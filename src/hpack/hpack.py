"""
Implements the HPACK header compression algorithm as detailed by RFC 7541.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .exceptions import HPACKDecodingError, InvalidTableSizeError, OversizedHeaderListError
from .huffman import HuffmanEncoder
from .huffman_constants import REQUEST_CODES, REQUEST_CODES_LENGTH
from .huffman_table import decode_huffman
from .struct import HeaderTuple, HeaderWeaklyTyped, NeverIndexedHeaderTuple
from .table import HeaderTable, table_entry_size
from .trace import (
    KIND_INDEXED,
    KIND_LITERAL_INCREMENTAL,
    KIND_LITERAL_NEVER_INDEXED,
    KIND_LITERAL_NO_INDEX,
    KIND_TABLE_SIZE_UPDATE,
    BlockTrace,
    EntryOrigin,
    Eviction,
    RepresentationTrace,
    TableClear,
)

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


def _literal_kind(should_index: bool, not_indexable: bool) -> str:
    """
    Returns the trace representation kind for a literal header field.
    """
    if should_index:
        return KIND_LITERAL_INCREMENTAL
    if not_indexable:
        return KIND_LITERAL_NEVER_INDEXED
    return KIND_LITERAL_NO_INDEX


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

    :param trace: (optional) Whether to record a structural trace of every
                  decoded header block. When enabled, each call to ``decode``
                  appends a :class:`BlockTrace <hpack.trace.BlockTrace>` to
                  ``self.traces``, describing every representation decoded
                  (its byte range, kind, Huffman usage and never-indexed
                  flag), the origin of any dynamic table entry referenced
                  (the block and field that created it, so that reused
                  numeric indices still resolve to the correct generation),
                  and any table size changes, evictions and table clears.

                  If decoding raises an exception, the trace for that block
                  is still appended, with ``complete`` set to ``False``: it
                  contains only the representations that were fully decoded
                  before the error.

                  Traces are structural only and never contain header names
                  or values unless ``trace_values`` is enabled, making them
                  safe for diagnostic logs.

                  .. versionadded:: 4.3.0
    :type trace: ``bool``

    :param trace_values: (optional) Whether traces should also capture header
                         names and values. Fields the peer marked as
                         never-indexed never have their values captured.
                         Defaults to ``False``.

                         .. versionadded:: 4.3.0
    :type trace_values: ``bool``
    """

    def __init__(self,
                 max_header_list_size: int = DEFAULT_MAX_HEADER_LIST_SIZE,
                 trace: bool = False,
                 trace_values: bool = False) -> None:
        self.header_table = HeaderTable()

        #: The maximum decompressed size we will allow for any single header
        #: block. This is a protection against DoS attacks that attempt to
        #: force the application to expand a relatively small amount of data
        #: into a really large header list, allowing enormous amounts of memory
        #: to be allocated.
        #:
        #: If this amount of data is exceeded, a `OversizedHeaderListError
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

        #: Whether to record a structural trace of each decoded header block.
        self.trace_enabled = trace

        #: Whether traces should capture header names and values for
        #: indexable fields. Never-indexed fields are never captured.
        self.trace_values = trace_values

        #: The traces of all blocks decoded while tracing was enabled, in
        #: decode order.
        self.traces: list[BlockTrace] = []

        self._block_index = -1
        self._active_trace: BlockTrace | None = None
        self._trace_offset = 0
        self.header_table.eviction_observer = self._on_table_eviction
        self.header_table.clear_observer = self._on_table_clear

    @property
    def header_table_size(self) -> int:
        """
        Controls the size of the HPACK header table.
        """
        return self.header_table.maxsize

    @header_table_size.setter
    def header_table_size(self, value: int) -> None:
        self.header_table.maxsize = value

    def decode(self, data: bytes, raw: bool = False) -> Iterable[HeaderTuple]:
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

        data_mem = memoryview(data)
        headers: list[HeaderTuple] = []
        data_len = len(data)
        inflated_size = 0
        current_index = 0

        self._block_index += 1
        block_trace: BlockTrace | None = None
        if self.trace_enabled:
            block_trace = BlockTrace(
                block_index=self._block_index,
                block_length=data_len,
            )
            self.traces.append(block_trace)
        self._active_trace = block_trace

        completed = False
        try:
            while current_index < data_len:
                self._trace_offset = current_index
                header, consumed = self._decode_field(
                    data_mem, current_index, len(headers),
                )

                if header:
                    headers.append(header)
                    inflated_size += table_entry_size(header[0], header[1])

                    if inflated_size > self.max_header_list_size:
                        msg = f"A header list larger than {self.max_header_list_size} has been received"
                        raise OversizedHeaderListError(msg)

                current_index += consumed

            # Confirm that the table size is lower than the maximum. We do this
            # here to ensure that we catch when the max has been *shrunk* and the
            # remote peer hasn't actually done that.
            self._assert_valid_table_size()

            try:
                result = [_unicode_if_needed(h, raw) for h in headers]
            except UnicodeDecodeError as err:
                msg = "Unable to decode headers as UTF-8"
                raise HPACKDecodingError(msg) from err
            completed = True
            return result
        finally:
            if block_trace is not None and not completed:
                # Decoding failed partway through the block: the trace only
                # covers the representations decoded so far and must not
                # claim the remaining bytes were decoded.
                block_trace.complete = False
            self._active_trace = None

    def _decode_field(self,
                      data_mem: memoryview,
                      current_index: int,
                      field_index: int) -> tuple[HeaderTuple | None, int]:
        """
        Decodes a single field representation from ``data_mem``, starting at
        ``current_index``. Returns the decoded header (or ``None`` for table
        size updates) and the number of bytes consumed.
        """
        # Work out what kind of header we're decoding.
        # If the high bit is 1, it's an indexed field.
        current = data_mem[current_index]
        indexed = bool(current & 0x80)

        # Otherwise, if the second-highest bit is 1 it's a field that does
        # alter the header table.
        literal_index = bool(current & 0x40)

        # Otherwise, if the third-highest bit is 1 it's an encoding context
        # update.
        encoding_update = bool(current & 0x20)

        if indexed:
            return self._decode_indexed(
                data_mem[current_index:], current_index,
            )
        if literal_index:
            # It's a literal header that does affect the header table.
            return self._decode_literal_index(
                data_mem[current_index:], current_index, field_index,
            )
        if encoding_update:
            # It's an update to the encoding context. These are forbidden
            # in a header block after any actual header.
            if field_index:
                msg = "Table size update not at the start of the block"
                raise HPACKDecodingError(msg)
            consumed = self._update_encoding_context(
                data_mem[current_index:], current_index,
            )
            return None, consumed
        # It's a literal header that does not affect the header table.
        return self._decode_literal_no_index(
            data_mem[current_index:], current_index, field_index,
        )

    def _assert_valid_table_size(self) -> None:
        """
        Check that the table size set by the encoder is lower than the maximum
        we expect to have.
        """
        if self.header_table_size > self.max_allowed_table_size:
            msg = "Encoder did not shrink table size to within the max"
            raise InvalidTableSizeError(msg)

    def _update_encoding_context(self, data: bytes | memoryview, start: int = 0) -> int:
        """
        Handles a byte that updates the encoding context.
        """
        # We've been asked to resize the header table.
        new_size, consumed = decode_integer(data, 5)
        if new_size > self.max_allowed_table_size:
            msg = "Encoder exceeded max allowable table size"
            raise InvalidTableSizeError(msg)
        self.header_table_size = new_size
        self._trace_representation(
            RepresentationTrace(
                start=start,
                end=start + consumed,
                kind=KIND_TABLE_SIZE_UPDATE,
                table_size=new_size,
            ),
        )
        return consumed

    def _decode_indexed(self, data: bytes | memoryview, start: int = 0) -> tuple[HeaderTuple, int]:
        """
        Decodes a header represented using the indexed representation.
        """
        index, consumed = decode_integer(data, 7)
        header = HeaderTuple(*self.header_table.get_by_index(index))
        log.debug("Decoded %s, consumed %d", header, consumed)
        self._trace_representation(
            RepresentationTrace(
                start=start,
                end=start + consumed,
                kind=KIND_INDEXED,
                index=index,
                origin=self.header_table.get_origin_by_index(index),
            ),
        )
        return header, consumed

    def _decode_literal_no_index(self, data: bytes | memoryview, start: int = 0, field_index: int = 0) -> tuple[HeaderTuple, int]:
        return self._decode_literal(data, should_index=False, start=start, field_index=field_index)

    def _decode_literal_index(self, data: bytes | memoryview, start: int = 0, field_index: int = 0) -> tuple[HeaderTuple, int]:
        return self._decode_literal(data, should_index=True, start=start, field_index=field_index)

    def _decode_literal(self,
                        data: bytes | memoryview,
                        should_index: bool,
                        start: int = 0,
                        field_index: int = 0) -> tuple[HeaderTuple, int]:
        """
        Decodes a header represented with a literal.
        """
        if isinstance(data, memoryview):
            data = data.tobytes()  # pragma: no cover

        not_indexable = not should_index and bool(data[0] & 0x10)
        name, name_index, name_origin, name_huffman, consumed = self._decode_literal_name(
            data, should_index,
        )
        total_consumed = consumed
        data = data[consumed:]

        # The header value is definitely length-based.
        length, consumed = decode_integer(data, 7)
        value = data[consumed:consumed + length]
        if len(value) != length:
            msg = "Truncated header block"
            raise HPACKDecodingError(msg)

        value_huffman = bool(data[0] & 0x80)
        if value_huffman:
            value = decode_huffman(value)

        # Updated the total consumed length.
        total_consumed += length + consumed

        # If we have been told never to index the header field, encode that in
        # the tuple we use.
        header: HeaderTuple
        if not_indexable:
            header = NeverIndexedHeaderTuple(name, value)
        else:
            header = HeaderTuple(name, value)

        # Record the representation before mutating the header table:
        # adding a new entry shifts all dynamic indices, so the origin of an
        # indexed name must be resolved against the pre-insertion table.
        self._trace_representation(
            RepresentationTrace(
                start=start,
                end=start + total_consumed,
                kind=_literal_kind(should_index, not_indexable),
                huffman=name_huffman or value_huffman,
                never_indexed=not_indexable,
                index=name_index,
                origin=name_origin,
                name=name,
                value=value,
            ),
        )

        self._index_literal(name, value, should_index, field_index)

        log.debug(
            "Decoded %s, total consumed %d bytes, indexed %s",
            header,
            total_consumed,
            should_index,
        )

        return header, total_consumed

    def _decode_literal_name(self,
                             data: bytes,
                             should_index: bool) -> tuple[bytes, int | None, EntryOrigin | None, bool, int]:
        """
        Decodes the name portion of a literal header field.

        Returns the name, its table index and the origin of that entry (when
        the name is indexed), whether the name was Huffman-coded, and the
        number of bytes consumed.
        """
        # When should_index is true, if the low six bits of the first byte are
        # nonzero, the header name is indexed.
        # When should_index is false, if the low four bits of the first byte
        # are nonzero the header name is indexed.
        if should_index:
            indexed_name = data[0] & 0x3F
            name_len = 6
        else:
            indexed_name = data[0] & 0x0F
            name_len = 4

        if indexed_name:
            # Indexed header name.
            index, consumed = decode_integer(data, name_len)
            name = self.header_table.get_by_index(index)[0]
            # Resolve the origin now: adding this field to the table later
            # shifts all dynamic indices.
            origin = self.header_table.get_origin_by_index(index)
            return name, index, origin, False, consumed

        # Literal header name. The first byte was consumed, so we need to
        # move forward.
        data = data[1:]

        length, consumed = decode_integer(data, 7)
        name = data[consumed:consumed + length]
        if len(name) != length:
            msg = "Truncated header block"
            raise HPACKDecodingError(msg)

        huffman = bool(data[0] & 0x80)
        if huffman:
            name = decode_huffman(name)
        # Add 1 to account for the byte we moved forward above.
        return name, None, None, huffman, consumed + length + 1

    def _index_literal(self, name: bytes, value: bytes, should_index: bool, field_index: int) -> None:
        """
        Adds a decoded literal header field to the header table, recording
        which decoded field created the entry so that later references to it
        can be traced back to this block even after its numeric index is
        reused.
        """
        if not should_index:
            return
        self.header_table.add(name, value)
        if (
            self._active_trace is not None
            and self.header_table.dynamic_entries
            and self.header_table.dynamic_entries[0] == (name, value)
        ):
            self.header_table.entry_origins[0] = EntryOrigin(
                self._block_index, field_index,
            )

    def _trace_representation(self, record: RepresentationTrace) -> None:
        """
        Appends a representation record to the active trace, if any.
        """
        trace = self._active_trace
        if trace is None:
            return

        if not self.trace_values or record.never_indexed:
            # Traces are structural by default, and never-indexed fields
            # never have their values captured: both keep sensitive header
            # values out of diagnostic logs.
            record = replace(record, name=None, value=None)

        trace.representations.append(record)

    def _on_table_eviction(self, name: bytes, value: bytes, origin: EntryOrigin | None) -> None:
        """
        Header table observer: records evictions in the active trace, if any.
        """
        trace = self._active_trace
        if trace is None:
            return
        trace.evictions.append(
            Eviction(
                at=self._trace_offset,
                entry_size=table_entry_size(name, value),
                origin=origin,
            ),
        )

    def _on_table_clear(self, reason: str, entries_discarded: int) -> None:
        """
        Header table observer: records table clears in the active trace, if any.
        """
        trace = self._active_trace
        if trace is None:
            return
        trace.clears.append(
            TableClear(
                at=self._trace_offset,
                reason=reason,
                entries_discarded=entries_discarded,
            ),
        )
