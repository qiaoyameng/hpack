"""
Tests for ``Encoder.encode_shortest``.

The centrepiece is an *independent* brute-force enumerator (``bf_best``)
that re-derives, from RFC 7541 first principles, the length of the shortest
legal encoding of a small header block: it has its own integer-length,
Huffman-length, table-search and table-eviction logic and simply tries every
legal representation of every field. ``encode_shortest`` must always match
that length, and the emitted block must round-trip and leave encoder and
decoder in sync for subsequent blocks.
"""
import itertools

import pytest

from hpack import (
    Decoder,
    Encoder,
    HeaderTuple,
    NeverIndexedHeaderTuple,
)
from hpack.hpack import SHORTEST_MAX_HEADERS
from hpack.huffman_constants import REQUEST_CODES_LENGTH
from hpack.table import HeaderTable

STATIC_TABLE = HeaderTable.STATIC_TABLE


def bf_int_len(integer, prefix_bits):
    """Independent re-implementation of the RFC 7541 integer encoding length."""
    max_number = (1 << prefix_bits) - 1
    if integer < max_number:
        return 1
    integer -= max_number
    length = 1
    while integer >= 128:
        integer >>= 7
        length += 1
    return length + 1


def bf_huffman_len(data):
    """Length in bytes of the Huffman-coded form of ``data``."""
    return (sum(REQUEST_CODES_LENGTH[b] for b in data) + 7) // 8


def bf_search(entries, name, value):
    """
    Independent table search. Returns ``(name_index, perfect_index)``: the
    lowest index whose entry has a matching name (usable for the name of a
    literal, and not necessarily the perfect-match entry), and the lowest
    index whose entry matches name and value (or ``None``).
    """
    name_index = None
    perfect_index = None
    for i, (n, v) in enumerate(STATIC_TABLE, 1):
        if n == name:
            if name_index is None:
                name_index = i
            if v == value:
                return name_index, i
    offset = len(STATIC_TABLE) + 1
    for j, (n, v) in enumerate(entries):
        if n == name:
            if name_index is None:
                name_index = offset + j
            if v == value:
                return name_index, offset + j
    return name_index, None


def bf_add(entries, maxsize, name, value):
    """Independent dynamic-table insertion, including eviction."""
    if 32 + len(name) + len(value) > maxsize:
        return ()
    entries = ((name, value),) + entries
    kept = []
    total = 0
    for n, v in entries:
        size = 32 + len(n) + len(v)
        if total + size > maxsize:
            break
        total += size
        kept.append((n, v))
    return tuple(kept)


def bf_best(entries, maxsize, headers):
    """
    Exhaustively computes the shortest legal encoding length (in bytes) of
    ``headers`` given the dynamic table state ``entries``. Sensitive fields
    keep their never-indexed marker; every other field may use any legal
    representation, and the name/value Huffman choices are independent.
    """
    if not headers:
        return 0

    (name, value, sensitive), rest = headers[0], headers[1:]
    name_index, perfect_index = bf_search(entries, name, value)

    options = []  # (cost, new_entries)
    if not sensitive and perfect_index is not None:
        options.append((bf_int_len(perfect_index, 7), entries))

    indexbits = (0x10,) if sensitive else (0x40, 0x00)
    for indexbit in indexbits:
        if indexbit == 0x40:
            new_entries = bf_add(entries, maxsize, name, value)
            prefix_bits = 6
        else:
            new_entries = entries
            prefix_bits = 4

        if name_index is not None:
            for huffman_value in (False, True):
                vlen = bf_huffman_len(value) if huffman_value else len(value)
                cost = bf_int_len(name_index, prefix_bits) + bf_int_len(vlen, 7) + vlen
                options.append((cost, new_entries))

        for huffman_name in (False, True):
            nlen = bf_huffman_len(name) if huffman_name else len(name)
            for huffman_value in (False, True):
                vlen = bf_huffman_len(value) if huffman_value else len(value)
                cost = (
                    1
                    + bf_int_len(nlen, 7) + nlen
                    + bf_int_len(vlen, 7) + vlen
                )
                options.append((cost, new_entries))

    return min(
        cost + bf_best(new_entries, maxsize, rest)
        for cost, new_entries in options
    )


def expected_tuple(name, value, sensitive):
    cls = NeverIndexedHeaderTuple if sensitive else HeaderTuple
    return cls(name, value)


# A small pool of header fields exercising: name reuse across fields, a
# static-table perfect match, a static-table name-only match, a longer
# Huffman-friendly pair, and a sensitive field.
POOL = [
    (b"a", b"b", False),
    (b"a", b"c", False),
    (b":method", b"GET", False),
    (b":method", b"PUT", False),
    (b"custom-key", b"custom-header", False),
    (b"a", b"b", True),
]

PRELOADS = [
    [],
    [(b"a", b"b"), (b"d", b"e")],
]


def small_header_blocks():
    for length in (0, 1, 2, 3):
        for seq in itertools.product(POOL, repeat=length):
            yield list(seq)


class TestShortestExhaustive:
    @pytest.mark.parametrize("preload", PRELOADS)
    @pytest.mark.parametrize("headers", list(small_header_blocks()))
    def test_matches_independent_exhaustive_search(self, preload, headers):
        encoder = Encoder()
        decoder = Decoder()
        if preload:
            decoder.decode(encoder.encode(preload, huffman=True), raw=True)

        initial_entries = tuple(encoder.header_table.dynamic_entries)
        maxsize = encoder.header_table.maxsize

        block = encoder.encode_shortest(headers)

        # The new mode must be exactly as short as the independent
        # brute-force optimum.
        assert len(block) == bf_best(initial_entries, maxsize, headers)

        # The block must round-trip, preserving order and sensitive markers.
        decoded = decoder.decode(block, raw=True)
        assert list(decoded) == [
            expected_tuple(n, v, s) for n, v, s in headers
        ]

        # Encoder and decoder must still be in sync: a follow-up block (in
        # the classic mode) must round-trip too.
        follow_up = [(b"post", b"script"), (b"a", b"b")]
        follow_block = encoder.encode(follow_up, huffman=True)
        assert list(decoder.decode(follow_block, raw=True)) == [
            HeaderTuple(n, v) for n, v in follow_up
        ]


def primed_high_index_pair():
    """
    Builds an encoder/decoder pair whose dynamic table holds a target entry
    at an index high enough that its indexed representation needs 2 bytes.
    """
    encoder, decoder = Encoder(), Decoder()
    dummies = [(b"x", bytes([i])) for i in range(130)]
    target = (b"x", b"\xfe")
    for block in (dummies[:60], [target], dummies[60:]):
        decoder.decode(encoder.encode(block, huffman=False), raw=True)
    return encoder, decoder, target


class TestShortestMode:
    def test_spend_more_now_to_help_later(self):
        """
        A field that already has a perfect match at a high table index is
        cheaper to *re-add* (paying for a literal now) so that its later
        repetitions use a one-byte indexed representation. A per-field
        greedy encoder would take the 2-byte indexed representation every
        time; the joint optimum pays 3 bytes once and then 1 byte twice.
        """
        # Sanity check the setup: the target sits at an index whose indexed
        # representation needs two bytes.
        encoder, decoder, target = primed_high_index_pair()
        index, _, perfect = encoder.header_table.search(*target)
        assert perfect is not None
        assert index >= 127

        seq = [target, target, target]

        greedy_encoder, greedy_decoder, _ = primed_high_index_pair()
        naive = greedy_encoder.encode(seq, huffman=True)
        assert len(naive) == 6
        greedy_decoder.decode(naive, raw=True)

        initial_entries = tuple(encoder.header_table.dynamic_entries)
        block = encoder.encode_shortest(seq)

        # 3 bytes to re-add the field with an indexed name, then 1 byte per
        # repetition: strictly better than 3 x 2-byte indexed fields.
        assert len(block) == 5
        assert len(block) < len(naive)
        assert len(block) == bf_best(
            initial_entries, encoder.header_table.maxsize,
            [(n, v, False) for n, v in seq],
        )

        decoded = decoder.decode(block, raw=True)
        assert list(decoded) == [HeaderTuple(*target)] * 3

        # The encoder state must reflect the emitted plan: a further block
        # still round-trips.
        more = encoder.encode_shortest([target])
        assert list(decoder.decode(more, raw=True)) == [HeaderTuple(*target)]

    def test_rejects_blocks_above_limit(self):
        encoder = Encoder()
        too_many = [(b"a", b"b")] * (SHORTEST_MAX_HEADERS + 1)
        with pytest.raises(ValueError, match="at most"):
            encoder.encode_shortest(too_many)
        # The failed call must not have changed the encoder state.
        assert list(encoder.header_table.dynamic_entries) == []

    def test_accepts_blocks_at_limit(self):
        encoder = Encoder()
        decoder = Decoder()
        headers = [(b"a", b"b")] * SHORTEST_MAX_HEADERS
        block = encoder.encode_shortest(headers)
        assert list(decoder.decode(block, raw=True)) == [
            HeaderTuple(b"a", b"b")
        ] * SHORTEST_MAX_HEADERS

    def test_pending_table_size_update_is_counted(self):
        encoder = Encoder()
        decoder = Decoder()

        preload = [(b"k", b"v")] * 5
        decoder.decode(encoder.encode(preload, huffman=False), raw=True)

        # Shrink the table: the update bytes are pending and some entries
        # have already been evicted from the encoder's table.
        encoder.header_table_size = 128
        initial_entries = tuple(encoder.header_table.dynamic_entries)
        assert len(initial_entries) < 5

        headers = [(b"k", b"v", False), (b"a", b"b", False)]
        block = encoder.encode_shortest(headers)

        # The block starts with the table size update (001xxxxx), and those
        # bytes are part of the emitted (and accounted) length.
        assert block[0] & 0xE0 == 0x20
        update_len = 2  # 128 does not fit in a 5-bit prefix
        assert len(block) == update_len + bf_best(initial_entries, 128, headers)

        decoded = decoder.decode(block, raw=True)
        assert list(decoded) == [HeaderTuple(b"k", b"v"), HeaderTuple(b"a", b"b")]
        assert decoder.header_table_size == 128

        follow_up = encoder.encode([(b"k", b"v")], huffman=True)
        assert list(decoder.decode(follow_up, raw=True)) == [HeaderTuple(b"k", b"v")]

    def test_sensitive_headers_stay_never_indexed(self):
        encoder = Encoder()
        decoder = Decoder()
        headers = [
            (b"a", b"b", False),
            (b"secret", b"value", True),
            (b"a", b"b", False),
        ]
        block = encoder.encode_shortest(headers)

        decoded = list(decoder.decode(block, raw=True))
        assert isinstance(decoded[1], NeverIndexedHeaderTuple)
        assert isinstance(decoded[0], HeaderTuple)
        assert not isinstance(decoded[0], NeverIndexedHeaderTuple)
        assert isinstance(decoded[2], HeaderTuple)
        assert not isinstance(decoded[2], NeverIndexedHeaderTuple)

        # The sensitive field never entered either table.
        assert (b"secret", b"value") not in encoder.header_table.dynamic_entries
        assert (b"secret", b"value") not in decoder.header_table.dynamic_entries

    def test_dict_input_and_never_longer_than_encode(self):
        headers = {
            ":method": "GET",
            ":path": "/",
            "host": "example.com",
            "custom-key": "custom-header",
        }
        shortest = Encoder().encode_shortest(headers)
        assert len(shortest) <= len(Encoder().encode(headers, huffman=True))
        assert len(shortest) <= len(Encoder().encode(headers, huffman=False))

        decoded = Decoder().decode(shortest)
        assert {tuple(h) for h in decoded} == {
            (n, v) for n, v in headers.items()
        }

    def test_header_tuple_and_three_tuple_inputs(self):
        headers = [
            HeaderTuple(":method", "GET"),
            NeverIndexedHeaderTuple("password", "hunter2"),
            ("custom", "value", True),
            ("other", "value"),
        ]
        encoder = Encoder()
        decoder = Decoder()
        decoded = list(decoder.decode(encoder.encode_shortest(headers)))
        assert isinstance(decoded[1], NeverIndexedHeaderTuple)
        assert isinstance(decoded[2], NeverIndexedHeaderTuple)
        assert isinstance(decoded[3], HeaderTuple)
        assert not isinstance(decoded[3], NeverIndexedHeaderTuple)
        assert [tuple(h) for h in decoded] == [
            (":method", "GET"),
            ("password", "hunter2"),
            ("custom", "value"),
            ("other", "value"),
        ]

    def test_empty_block(self):
        assert Encoder().encode_shortest([]) == b""
