"""
Implements the HPACK header compression algorithm as detailed by RFC 7541.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .exceptions import HPACKDecodingError, InvalidTableSizeError, OversizedHeaderListError
from .huffman import HuffmanEncoder
from .huffman_constants import REQUEST_CODES, REQUEST_CODES_LENGTH
from .huffman_table import HuffmanDecoder
from .struct import HeaderTuple, HeaderWeaklyTyped, NeverIndexedHeaderTuple
from .table import HeaderTable, table_entry_size

if TYPE_CHECKING:
    from collections.abc import Iterable  # pragma: no cover

log = logging.getLogger(__name__)

INDEX_NONE = b"\x00"
INDEX_NEVER = b"\x10"
INDEX_INCREMENTAL = b"\x40"

VARINT_MAX_LENGTH = 5 # octets, enough for encoding prefix + uint32

# Precompute 2^i for 1-8 for use in prefix calcs.
# Zero index is not used but there to save a subtraction
# as prefix numbers are not zero indexed.
_PREFIX_BIT_MAX_NUMBERS = [(2 ** i) - 1 for i in range(9)]

# We default the maximum header list we're willing to accept to 64kB. That's a
# lot of headers, but if applications want to raise it they can do.
DEFAULT_MAX_HEADER_LIST_SIZE = 2 ** 16


def _unicode_if_needed(header: HeaderWeaklyTyped, raw: bool) -> HeaderTuple:
    """
    Provides a header as a unicode string if raw is False, otherwise returns
    it as a bytestring.
    """
    name = bytes(header[0])  # type: ignore
    value = bytes(header[1])  # type: ignore

    if not raw:
        return header.__class__(name.decode("utf-8"), value.decode("utf-8"))  # type: ignore
    return header.__class__(name, value)  # type: ignore


def encode_integer(integer: int, prefix_bits: int) -> bytearray:
    """
    Encodes an integer according to the wacky integer encoding rules
    defined in the HPACK spec.
    """
    log.debug("Encoding %d with %d bits", integer, prefix_bits)

    if integer < 0:
        msg = f"Can only encode positive integers, got {integer}"
        raise ValueError(msg)

    if prefix_bits < 1 or prefix_bits > 8:
        msg = f"Prefix bits must be between 1 and 8, got {prefix_bits}"
        raise ValueError(msg)

    max_number = _PREFIX_BIT_MAX_NUMBERS[prefix_bits]

    if integer < max_number:
        return bytearray([integer])  # Seriously?
    elements = [max_number]
    integer -= max_number

    while integer >= 128:
        elements.append((integer & 127) + 128)
        integer >>= 7

    elements.append(integer)

    return bytearray(elements)


def decode_integer(data: bytes | memoryview, prefix_bits: int) -> tuple[int, int]:
    """
    Decodes an integer according to the wacky integer encoding rules
    defined in the HPACK spec. Returns a tuple of the decoded integer and the
    number of bytes that were consumed from ``data`` in order to get that
    integer.
    """
    if prefix_bits < 1 or prefix_bits > 8:
        msg = f"Prefix bits must be between 1 and 8, got {prefix_bits}"
        raise ValueError(msg)

    max_number = _PREFIX_BIT_MAX_NUMBERS[prefix_bits]
    index = 1
    shift = 0
    mask = (0xFF >> (8 - prefix_bits))

    try:
        number = data[0] & mask
        if number == max_number:
            while True:
                next_byte = data[index]
                index += 1

                if next_byte >= 128:
                    number += (next_byte - 128) << shift
                else:
                    number += next_byte << shift
                    break
                shift += 7

                if index > VARINT_MAX_LENGTH:  # sanity check to prevent infinite loops
                    # We have consumed more than enough bytes for a typical unsigned integer.
                    # The maximum size is not defined in HPACK RFC7541 Section 5.1:
                    #   > This integer representation allows for values of indefinite size.  It
                    #   > is also possible for an encoder to send a large number of zero
                    #   > values, which can waste octets and could be used to overflow integer
                    #   > values.  Integer encodings that exceed implementation limits -- in
                    #   > value or octet length -- MUST be treated as decoding errors.
                    msg = f"Variable integer representation is too long: {data!r}"
                    raise HPACKDecodingError(msg)

    except IndexError as err:
        msg = f"Unable to decode HPACK integer representation from {data!r}"
        raise HPACKDecodingError(msg) from err

    log.debug("Decoded %d, consumed %d bytes", number, index)

    return number, index


def _dict_to_iterable(header_dict: dict[bytes | str, bytes | str]) \
        -> Iterable[tuple[bytes | str, bytes | str]]:
    """
    Converts a dictionary to an iterable of key-value tuples. This is a
    HPACK-specific function because it pulls "special-headers" out first and
    then emits them.
    """
    if not isinstance(header_dict, dict):  # pragma: no cover
        msg = f"header_dict not a dict, but {type(header_dict)}"
        raise TypeError(msg)

    keys = sorted(
        header_dict.keys(),
        key=lambda k: not _to_bytes(k).startswith(b":"),
    )
    for key in keys:
        yield key, header_dict[key]


def _to_bytes(value: bytes | str | Any) -> bytes:
    """
    Convert anything to bytes through a UTF-8 encoded string
    """
    t = type(value)
    if t is bytes:
        return value  # type: ignore
    if t is not str:
        value = str(value)
    return value.encode("utf-8")  # type: ignore


class Encoder:
    """
    An HPACK encoder object. This object takes HTTP headers and emits encoded
    HTTP/2 header blocks.
    """

    def __init__(self) -> None:
        self.header_table = HeaderTable()
        self.huffman_coder = HuffmanEncoder(
            REQUEST_CODES, REQUEST_CODES_LENGTH,
        )
        self.table_size_changes: list[int] = []

    @property
    def header_table_size(self) -> int:
        """
        Controls the size of the HPACK header table.
        """
        return self.header_table.maxsize

    @header_table_size.setter
    def header_table_size(self, value: int) -> None:
        self.header_table.maxsize = value
        if self.header_table.resized:
            self.table_size_changes.append(value)

    def encode(self,
               headers: Iterable[\
                   HeaderTuple | \
                   tuple[bytes | str, bytes | str] | \
                   tuple[bytes | str, bytes | str, bool | None]] | \
                   dict[bytes | str, bytes | str],
               huffman: bool = True) -> bytes:
        """
        Takes a set of headers and encodes them into a HPACK-encoded header
        block.

        :param headers: The headers to encode. Must be either an iterable of
                        tuples, an iterable of :class:`HeaderTuple
                        <hpack.HeaderTuple>`, or a ``dict``.

                        If an iterable of tuples, the tuples may be either
                        two-tuples or three-tuples. If they are two-tuples, the
                        tuples must be of the format ``(name, value)``. If they
                        are three-tuples, they must be of the format
                        ``(name, value, sensitive)``, where ``sensitive`` is a
                        boolean value indicating whether the header should be
                        added to header tables anywhere. If not present,
                        ``sensitive`` defaults to ``False``.

                        If an iterable of :class:`HeaderTuple
                        <hpack.HeaderTuple>`, the tuples must always be
                        two-tuples. Instead of using ``sensitive`` as a third
                        tuple entry, use :class:`NeverIndexedHeaderTuple
                        <hpack.NeverIndexedHeaderTuple>` to request that
                        the field never be indexed.

                        .. warning:: HTTP/2 requires that all special headers
                            (headers whose names begin with ``:`` characters)
                            appear at the *start* of the header block. While
                            this method will ensure that happens for ``dict``
                            subclasses, callers using any other iterable of
                            tuples **must** ensure they place their special
                            headers at the start of the iterable.

                            For efficiency reasons users should prefer to use
                            iterables of two-tuples: fixing the ordering of
                            dictionary headers is an expensive operation that
                            should be avoided if possible.

        :param huffman: (optional) Whether to Huffman-encode any header sent as
                        a literal value. Except for use when debugging, it is
                        recommended that this be left enabled.

        :returns: A bytestring containing the HPACK-encoded header block.
        """
        # Transforming the headers into a header block is a procedure that can
        # be modeled as a chain or pipe. First, the headers are encoded. This
        # encoding can be done a number of ways. If the header name-value pair
        # are already in the header table we can represent them using the
        # indexed representation: the same is true if they are in the static
        # table. Otherwise, a literal representation will be used.
        header_block = []

        # Before we begin, if the header table size has been changed we need
        # to signal all changes since last emission appropriately.
        if self.header_table.resized:
            header_block.append(self._encode_table_size_change())
            self.header_table.resized = False

        if isinstance(headers, dict):
            # Turn the headers into a list of tuples if possible. This is the
            # natural way to interact with them in HPACK. Because dictionaries are
            # un-ordered, we need to make sure we grab the "special" headers first.
            hpack_headers = _dict_to_iterable(headers)
        else:
            """
            Assume headers is an iterable of HeaderTuples, or plain 2-tuples, or plain 3-tuples:

            examples:
            [
                HeaderTuple(':method', 'GET'),
                NeverIndexedHeaderTuple('customkey', 'sensitiveinfo'),
            ]
            or
            [
                (':method', 'GET'),
                ('customkey', 'some-data'),
            ]
            or
            [
                (':method', 'GET', True),
                ('customkey', 'sensitiveinfo', True),
            ]
            """
            hpack_headers = iter(headers)  # type: ignore

        # Add each header to the header block
        for header in hpack_headers:
            sensitive = False
            if isinstance(header, HeaderTuple):
                # HeaderTuple implies it's a 2-tuple with the sensitive information stored as instance attribute
                sensitive = not header.indexable
            elif len(header) > 2:
                sensitive = header[2]

            new_header = (_to_bytes(header[0]), _to_bytes(header[1]))
            header_block.append(self.add(new_header, sensitive, huffman))

        encoded = b"".join(header_block)

        log.debug("Encoded header block to %s", encoded)

        return encoded

    def add(self, to_add: tuple[bytes, bytes], sensitive: bool, huffman: bool = False) -> bytes:
        """
        Serializes a header key-value tuple.

        When sensitive is True, the header will not be added to the header table
        (see https://www.rfc-editor.org/rfc/rfc7541.html#section-7.1.3 for details),
        furthermore, the header value will be redacted in debug logs, as "SENSITIVE_REDACTED",
        to prevent accidental exposure of sensitive information.
        """
        name, value = to_add

        display_value = value if not sensitive else b"SENSITIVE_REDACTED"
        log.debug(
            "Adding %s=%s to the header table, sensitive:%s, huffman:%s",
            name,
            display_value,
            sensitive,
            huffman,
        )

        # Set our indexing mode
        indexbit = INDEX_INCREMENTAL if not sensitive else INDEX_NEVER

        # Search for a matching header in the header table.
        match = self.header_table.search(name, value)

        if match is None:
            # Not in the header table. Encode using the literal syntax,
            # and add it to the header table.
            encoded = self._encode_literal(name, value, indexbit, huffman)
            if not sensitive:
                self.header_table.add(name, value)
            return encoded

        # The header is in the table, break out the values. If we matched
        # perfectly, we can use the indexed representation: otherwise we
        # can use the indexed literal.
        index, name, perfect = match

        if perfect is not None:
            # Indexed representation.
            encoded = self._encode_indexed(index)
        else:
            # Indexed literal. We are going to add header to the
            # header table unconditionally. It is a future todo to
            # filter out headers which are known to be ineffective for
            # indexing since they just take space in the table and
            # pushed out other valuable headers.
            encoded = self._encode_indexed_literal(
                index, value, indexbit, huffman,
            )
            if not sensitive:
                self.header_table.add(name, value)

        return encoded

    def _encode_indexed(self, index: int) -> bytes:
        """
        Encodes a header using the indexed representation.
        """
        field = encode_integer(index, 7)
        field[0] |= 0x80  # we set the top bit
        return bytes(field)

    def _encode_literal(self, name: bytes, value: bytes, indexbit: bytes, huffman: bool = False) -> bytes:
        """
        Encodes a header with a literal name and literal value. If ``indexing``
        is True, the header will be added to the header table: otherwise it
        will not.
        """
        if huffman:
            name = self.huffman_coder.encode(name)
            value = self.huffman_coder.encode(value)

        name_len = encode_integer(len(name), 7)
        value_len = encode_integer(len(value), 7)

        if huffman:
            name_len[0] |= 0x80
            value_len[0] |= 0x80

        return b"".join(
            [indexbit, bytes(name_len), name, bytes(value_len), value],
        )

    def _encode_indexed_literal(self, index: int, value: bytes, indexbit: bytes, huffman: bool = False) -> bytes:
        """
        Encodes a header with an indexed name and a literal value and performs
        incremental indexing.
        """
        if indexbit != INDEX_INCREMENTAL:
            prefix = encode_integer(index, 4)
        else:
            prefix = encode_integer(index, 6)

        prefix[0] |= ord(indexbit)

        if huffman:
            value = self.huffman_coder.encode(value)

        value_len = encode_integer(len(value), 7)

        if huffman:
            value_len[0] |= 0x80

        return b"".join([bytes(prefix), bytes(value_len), value])

    def _encode_table_size_change(self) -> bytes:
        """
        Produces the encoded form of all header table size change context
        updates.
        """
        block = b""
        for size_bytes in self.table_size_changes:
            b = encode_integer(size_bytes, 5)
            b[0] |= 0x20
            block += bytes(b)
        self.table_size_changes = []
        return block


class _IntegerReader:
    __slots__ = (
        "awaiting_continuation", "continuation_count", "first_byte", "number",
        "prefix_bits", "shift",
    )

    def __init__(self, first_byte: int, prefix_bits: int) -> None:
        self.first_byte = first_byte
        self.prefix_bits = prefix_bits
        self.number = first_byte & _PREFIX_BIT_MAX_NUMBERS[prefix_bits]
        self.shift = 0
        self.continuation_count = 0
        self.awaiting_continuation = self.number == _PREFIX_BIT_MAX_NUMBERS[prefix_bits]

    @property
    def needs_continuation(self) -> bool:
        return self.awaiting_continuation

    def feed(self, byte: int) -> bool:
        self.continuation_count += 1
        if self.continuation_count >= VARINT_MAX_LENGTH:
            msg = "Variable integer representation is too long"
            raise HPACKDecodingError(msg)

        if byte >= 128:
            self.number += (byte - 128) << self.shift
            self.shift += 7
            return False

        self.number += byte << self.shift
        self.awaiting_continuation = False
        return True


class _HeaderBlockParser:
    _STATE_REPRESENTATION = 0
    _STATE_INTEGER = 1
    _STATE_STRING = 2

    _TARGET_INDEXED = "indexed"
    _TARGET_TABLE_SIZE = "table_size"
    _TARGET_NAME_INDEX = "name_index"
    _TARGET_NAME_LENGTH = "name_length"
    _TARGET_VALUE_LENGTH = "value_length"

    __slots__ = (
        "buffer", "decoder", "headers", "headers_seen", "inflated_size",
        "integer_prefix_bits", "integer_reader", "integer_target",
        "literal_mode", "max_header_list_size", "name", "state",
        "string_huffman", "string_is_name", "string_length", "string_output",
        "huffman_decoder", "table",
    )

    def __init__(self, decoder: Decoder, max_header_list_size: int) -> None:
        self.decoder = decoder
        self.table = decoder.header_table
        self.max_header_list_size = max_header_list_size
        self.buffer = bytearray()
        self.headers: list[HeaderTuple] = []
        self.headers_seen = False
        self.inflated_size = 0
        self.state = self._STATE_REPRESENTATION
        self.integer_reader: _IntegerReader | None = None
        self.integer_target = ""
        self.integer_prefix_bits = 0
        self.literal_mode = 0
        self.name: bytes | None = None
        self.string_huffman = False
        self.string_is_name = False
        self.string_length = 0
        self.string_output = bytearray()
        self.huffman_decoder: HuffmanDecoder | None = None

    def feed(self, data: bytes | bytearray | memoryview) -> list[HeaderTuple]:
        before = len(self.headers)
        self.buffer.extend(data)
        consumed = self._drain()
        del self.buffer[:consumed]
        return self.headers[before:]

    def end(self) -> None:
        if self.state != self._STATE_REPRESENTATION or self.buffer:
            msg = "Truncated header block"
            raise HPACKDecodingError(msg)
        self.decoder._assert_valid_table_size()

    def _drain(self) -> int:
        consumed = 0

        while consumed < len(self.buffer):
            if self.state == self._STATE_REPRESENTATION:
                self._start_representation(self.buffer[consumed])
                consumed += 1

            if self.state == self._STATE_INTEGER:
                if self.integer_reader is None:
                    if consumed >= len(self.buffer):
                        return consumed
                    self._begin_integer(
                        self.buffer[consumed],
                        self.integer_prefix_bits,
                        self.integer_target,
                    )
                    consumed += 1

                if self.integer_reader is not None and self.integer_reader.needs_continuation:
                    complete = False
                    while consumed < len(self.buffer):
                        complete = self.integer_reader.feed(self.buffer[consumed])
                        consumed += 1
                        if complete:
                            self._finish_integer(self.integer_reader.number)
                            break
                    if not complete:
                        return consumed
                elif self.integer_reader is not None:
                    self._finish_integer(self.integer_reader.number)

            if self.state == self._STATE_STRING:
                consumed = self._consume_string(consumed)
                if self.state == self._STATE_STRING:
                    return consumed

        return consumed

    def _start_representation(self, first_byte: int) -> None:
        if first_byte & 0x80:
            self._begin_integer(first_byte, 7, self._TARGET_INDEXED)
            return

        if first_byte & 0x40:
            self.literal_mode = 0x40
            indexed_name = first_byte & 0x3F
            if indexed_name:
                self._begin_integer(first_byte, 6, self._TARGET_NAME_INDEX)
            else:
                self._await_string_length(self._TARGET_NAME_LENGTH, 7)
            return

        if first_byte & 0x20:
            if self.headers_seen:
                msg = "Table size update not at the start of the block"
                raise HPACKDecodingError(msg)
            self._begin_integer(first_byte, 5, self._TARGET_TABLE_SIZE)
            return

        self.literal_mode = first_byte & 0x10
        indexed_name = first_byte & 0x0F
        if indexed_name:
            self._begin_integer(first_byte, 4, self._TARGET_NAME_INDEX)
        else:
            self._await_string_length(self._TARGET_NAME_LENGTH, 7)

    def _begin_integer(self, first_byte: int, prefix_bits: int, target: str) -> None:
        self.integer_target = target
        self.integer_prefix_bits = prefix_bits
        self.integer_reader = _IntegerReader(first_byte, prefix_bits)
        if not self.integer_reader.needs_continuation:
            self._finish_integer(self.integer_reader.number)
        else:
            self.state = self._STATE_INTEGER

    def _await_string_length(self, target: str, prefix_bits: int) -> None:
        self.state = self._STATE_INTEGER
        self.integer_target = target
        self.integer_prefix_bits = prefix_bits
        self.integer_reader = None

    def _finish_integer(self, value: int) -> None:
        reader = self.integer_reader
        self.integer_reader = None
        target = self.integer_target

        if target == self._TARGET_INDEXED:
            header = HeaderTuple(*self.table.get_by_index(value))
            self._emit(header)
            self.state = self._STATE_REPRESENTATION
        elif target == self._TARGET_TABLE_SIZE:
            if value > self.decoder.max_allowed_table_size:
                msg = "Encoder exceeded max allowable table size"
                raise InvalidTableSizeError(msg)
            self.decoder.header_table_size = value
            self.state = self._STATE_REPRESENTATION
        elif target == self._TARGET_NAME_INDEX:
            self.name = self.table.get_by_index(value)[0]
            self._await_string_length(self._TARGET_VALUE_LENGTH, 7)
        elif target == self._TARGET_NAME_LENGTH:
            self.string_is_name = True
            assert reader is not None
            self._begin_string(value, reader.first_byte & 0x80)
        elif target == self._TARGET_VALUE_LENGTH:
            self.string_is_name = False
            assert reader is not None
            self._begin_string(value, reader.first_byte & 0x80)

    def _begin_string(self, length: int, huffman: int) -> None:
        self.string_length = length
        self.string_huffman = bool(huffman)
        self.string_output = bytearray()
        self.huffman_decoder = HuffmanDecoder() if self.string_huffman else None
        self.state = self._STATE_STRING

    def _consume_string(self, offset: int) -> int:
        available = min(self.string_length, len(self.buffer) - offset)
        if available:
            chunk = self.buffer[offset:offset + available]
            if self.string_huffman:
                assert self.huffman_decoder is not None
                self.huffman_decoder.feed(chunk)
            else:
                self.string_output.extend(chunk)
            self.string_length -= available
            offset += available
            self._check_partial_header_size()

        if self.string_length:
            return offset

        if self.string_huffman:
            assert self.huffman_decoder is not None
            value = self.huffman_decoder.finish()
        else:
            value = bytes(self.string_output)
        self._finish_string(value)
        return offset

    def _finish_string(self, value: bytes) -> None:
        self.huffman_decoder = None
        self.string_output = bytearray()

        if self.string_is_name:
            self.name = value
            self._await_string_length(self._TARGET_VALUE_LENGTH, 7)
            return

        assert self.name is not None
        if self.literal_mode == 0x10:
            header: HeaderTuple = NeverIndexedHeaderTuple(self.name, value)
        else:
            header = HeaderTuple(self.name, value)

        if self.literal_mode == 0x40:
            self.table.add(self.name, value)
        self.name = None
        self._emit(header)
        self.state = self._STATE_REPRESENTATION

    def _check_partial_header_size(self) -> int:
        name_length = len(self.name) if self.name is not None else 0
        decoded_length = len(self.string_output)
        if self.string_is_name:
            name_length = 0
        if self.string_huffman:
            assert self.huffman_decoder is not None
            decoded_length = len(self.huffman_decoder.decoded_bytes)

        if self.string_is_name:
            name_length = decoded_length
            value_length = 0
        else:
            value_length = decoded_length

        current_size = self.inflated_size + 32 + name_length + value_length
        if current_size > self.max_header_list_size:
            msg = f"A header list larger than {self.max_header_list_size} has been received"
            raise OversizedHeaderListError(msg)
        return current_size

    def _emit(self, header: HeaderTuple) -> None:
        self.headers_seen = True
        self.headers.append(header)
        self.inflated_size += table_entry_size(header[0], header[1])
        if self.inflated_size > self.max_header_list_size:
            msg = f"A header list larger than {self.max_header_list_size} has been received"
            raise OversizedHeaderListError(msg)


class Decoder:
    """
    An HPACK decoder object.

    .. versionchanged:: 2.3.0
       Added ``max_header_list_size`` argument.

    :param max_header_list_size: The maximum decompressed size we will allow
        for any single header block. This is a protection against DoS attacks
        that attempt to force the application to expand a relatively small
        amount of data into a really large header list, allowing enormous
        amounts of memory to be allocated.

        If this amount of data is exceeded, a `OversizedHeaderListError
        <hpack.OversizedHeaderListError>` exception will be raised. At this
        point the connection should be shut down, as the HPACK state will no
        longer be usable.

        Defaults to 64kB.
    :type max_header_list_size: ``int``

    Header blocks may be supplied either as one complete bytestring to
    :meth:`decode` or in chunks to :meth:`begin_header_block`,
    :meth:`feed_header_block` and :meth:`end_header_block`. If decoding fails,
    :attr:`failed` becomes ``True`` and the persistent HPACK context must be
    discarded.
    """

    def __init__(self, max_header_list_size: int = DEFAULT_MAX_HEADER_LIST_SIZE) -> None:
        self.header_table = HeaderTable()

        #: The maximum decompressed size we will allow for any single header
        #: block. This is a protection against DoS attacks that attempt to
        #: force the application to expand a relatively small amount of data
        #: into a really large header list, allowing enormous amounts of memory
        #: to be allocated.
        #:
        #: If this amount is exceeded, a `OversizedHeaderListError
        #: <hpack.OversizedHeaderListError>` exception will be raised. At this
        #: point the connection should be shut down, as the HPACK state will no
        #: longer be usable.
        #:
        #: Defaults to 64kB.
        #:
        #: .. versionadded:: 2.3.0
        self.max_header_list_size = max_header_list_size

        #: Maximum allowed header table size.
        #:
        #: A HTTP/2 implementation should set this to the most recent value of
        #: SETTINGS_HEADER_TABLE_SIZE that it sent *and has received an ACK
        #: for*. Once this setting is set, the actual header table size will be
        #: checked at the end of each decoding run and whenever it is changed,
        #: to confirm that it fits in this size.
        self.max_allowed_table_size = self.header_table.maxsize

        #: Whether a prior decoding error has made this context unusable.
        self.failed = False
        self._block_parser: _HeaderBlockParser | None = None
        self._block_raw = False

    @property
    def header_table_size(self) -> int:
        """
        Controls the size of the HPACK header table.
        """
        return self.header_table.maxsize

    @header_table_size.setter
    def header_table_size(self, value: int) -> None:
        self.header_table.maxsize = value

    def begin_header_block(self, raw: bool = False) -> None:
        """
        Begin decoding one HPACK header block supplied in chunks.
        """
        self._ensure_usable()
        if self._block_parser is not None:
            msg = "A HPACK header block has already been started"
            raise HPACKDecodingError(msg)
        self._block_raw = raw
        self._block_parser = _HeaderBlockParser(self, self.max_header_list_size)

    def feed_header_block(self, data: bytes | bytearray | memoryview) -> list[HeaderTuple]:
        """
        Feed one byte chunk of a started HPACK header block and return complete
        headers that became available in this chunk.
        """
        self._ensure_usable()
        parser = self._require_active_block()
        try:
            headers = parser.feed(data)
            return self._convert_headers(headers)
        except HPACKDecodingError:
            self.fail()
            raise

    def end_header_block(self) -> list[HeaderTuple]:
        """
        Mark a streamed HPACK header block complete and validate its boundary.

        :raises HPACKDecodingError: If any integer, string payload, Huffman
                                    symbol or other representation remains
                                    incomplete.
        """
        self._ensure_usable()
        parser = self._require_active_block()
        try:
            parser.end()
            self._block_parser = None
            return []
        except HPACKDecodingError:
            self.fail()
            raise

    def fail(self) -> None:
        """
        Mark the persistent HPACK compression context as unusable.
        """
        self.failed = True
        self._block_parser = None

    def decode(
        self,
        data: bytes | bytearray | memoryview,
        raw: bool = False,
    ) -> list[HeaderTuple]:
        """
        Takes an HPACK-encoded header block and decodes it into a header set.

        :param data: A bytestring representing a complete HPACK-encoded header
                     block.
        :param raw: (optional) Whether to return the headers as tuples of raw
                    byte strings or to decode them as UTF-8 before returning
                    them. The default value is False, which returns tuples of
                    Unicode strings
        :returns: A list of two-tuples of ``(name, value)`` representing the
                  HPACK-encoded headers, in the order they were decoded.
        :raises HPACKDecodingError: If an error is encountered while decoding
                                    the header block.
        """
        log.debug("Decoding %s", data)
        self._ensure_usable()

        parser = _HeaderBlockParser(self, self.max_header_list_size)
        try:
            headers = parser.feed(data)
            parser.end()
        except HPACKDecodingError:
            self.fail()
            raise

        try:
            return self._convert_headers(headers, raw)
        except HPACKDecodingError:
            self.fail()
            raise

    def _ensure_usable(self) -> None:
        if self.failed:
            msg = "HPACK decoder has failed and must not be reused"
            raise HPACKDecodingError(msg)

    def _require_active_block(self) -> _HeaderBlockParser:
        parser = self._block_parser
        if parser is None:
            msg = "No HPACK header block has been started"
            raise HPACKDecodingError(msg)
        return parser

    def _convert_headers(self, headers: list[HeaderTuple], raw: bool | None = None) -> list[HeaderTuple]:
        if raw is None:
            raw = self._block_raw
        try:
            return [_unicode_if_needed(header, raw) for header in headers]
        except UnicodeDecodeError as err:
            msg = "Unable to decode headers as UTF-8"
            raise HPACKDecodingError(msg) from err

    def _assert_valid_table_size(self) -> None:
        """
        Check that the table size set by the encoder is lower than the maximum
        we expect to have.
        """
        if self.header_table_size > self.max_allowed_table_size:
            msg = "Encoder did not shrink table size to within the max"
            raise InvalidTableSizeError(msg)
