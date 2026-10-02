from __future__ import annotations

from collections.abc import Iterable
from itertools import product

import pytest

from hpack import Decoder, Encoder, HeaderTuple, NeverIndexedHeaderTuple
from hpack.huffman_constants import REQUEST_CODES, REQUEST_CODES_LENGTH
from hpack.huffman import HuffmanEncoder
from hpack.table import HeaderTable, table_entry_size

huffman_encoder = HuffmanEncoder(REQUEST_CODES, REQUEST_CODES_LENGTH)
SHORT_CASES = [
    [(b"a", b"b"), (b"a", b"b")],
    [(b":method", b"GET"), (b"x", b"y"), (b"x", b"y")],
    [
        (b"long-custom-name", b"v"),
        (b"long-custom-name", b"v"),
        (b"long-custom-name", b"other-value"),
    ],
    [(b"z", b"z"), (b":path", b"/"), (b"z", b"z")],
    [
        (b":method", b"GET", True),
        (b"secret-name", b"secret-value", True),
        (b"x", b"y"),
    ],
]


def encode_integer(value: int, prefix_bits: int) -> bytes:
    maximum = (1 << prefix_bits) - 1
    if value < maximum:
        return bytes([value])

    output = bytearray([maximum])
    value -= maximum
    while value >= 128:
        output.append((value & 127) | 128)
        value >>= 7
    output.append(value)
    return bytes(output)


def string_options(value: bytes) -> list[tuple[bytes, bool]]:
    raw_length = encode_integer(len(value), 7)
    huffman_value = huffman_encoder.encode(value)
    huffman_length = bytearray(encode_integer(len(huffman_value), 7))
    huffman_length[0] |= 0x80
    return [
        (bytes(raw_length) + value, False),
        (bytes(huffman_length) + huffman_value, True),
    ]


def normalize_headers(
    headers: Iterable[tuple[bytes, bytes] | tuple[bytes, bytes, bool] | HeaderTuple],
) -> list[tuple[bytes, bytes, bool]]:
    normalized = []
    for header in headers:
        if isinstance(header, HeaderTuple):
            normalized.append((header[0], header[1], isinstance(header, NeverIndexedHeaderTuple)))
        else:
            sensitive = len(header) == 3 and bool(header[2])
            normalized.append((header[0], header[1], sensitive))
    return normalized


def matching_indexes(
    entries: tuple[tuple[bytes, bytes], ...], name: bytes, value: bytes | None = None,
) -> list[int]:
    table = HeaderTable.STATIC_TABLE + entries
    return [
        index
        for index, (entry_name, entry_value) in enumerate(table, 1)
        if entry_name == name and (value is None or entry_value == value)
    ]


def advance_state(
    entries: tuple[tuple[bytes, bytes], ...],
    current_size: int,
    name: bytes,
    value: bytes,
    maxsize: int,
) -> tuple[tuple[tuple[bytes, bytes], ...], int]:
    entry_size = table_entry_size(name, value)
    if entry_size > maxsize:
        return (), 0

    entries = ((name, value),) + entries
    current_size += entry_size
    while current_size > maxsize:
        evicted_name, evicted_value = entries[-1]
        entries = entries[:-1]
        current_size -= table_entry_size(evicted_name, evicted_value)
    return entries, current_size


def brute_force_shortest(
    headers: Iterable[tuple[bytes, bytes] | tuple[bytes, bytes, bool] | HeaderTuple],
) -> int:
    headers = normalize_headers(headers)
    states: dict[tuple[tuple[tuple[bytes, bytes], ...], int], dict[bytes, None]] = {
        ((), 0): {b"": None},
    }

    for name, value, sensitive in headers:
        next_states: dict[tuple[tuple[tuple[bytes, bytes], ...], int], dict[bytes, None]] = {}
        value_options = string_options(value)
        name_options = string_options(name)

        for (entries, current_size), wires in states.items():
            candidates: list[tuple[bytes, bool]] = []
            exact_indexes = matching_indexes(entries, name, value)
            name_indexes = matching_indexes(entries, name)

            if sensitive:
                for index, (value_fragment, _) in product(name_indexes, value_options):
                    prefix = bytearray(encode_integer(index, 4))
                    prefix[0] |= 0x10
                    candidates.append((bytes(prefix) + value_fragment, False))
                for (name_fragment, _), (value_fragment, _) in product(name_options, value_options):
                    candidates.append((b"\x10" + name_fragment + value_fragment, False))
            else:
                for index in exact_indexes:
                    prefix = bytearray(encode_integer(index, 7))
                    prefix[0] |= 0x80
                    candidates.append((bytes(prefix), False))

                for index, (value_fragment, _) in product(name_indexes, value_options):
                    no_index = bytearray(encode_integer(index, 4))
                    incremental = bytearray(encode_integer(index, 6))
                    incremental[0] |= 0x40
                    candidates.extend([
                        (bytes(no_index) + value_fragment, False),
                        (bytes(incremental) + value_fragment, True),
                    ])

                for (name_fragment, _), (value_fragment, _) in product(name_options, value_options):
                    candidates.extend([
                        (b"\x00" + name_fragment + value_fragment, False),
                        (b"\x40" + name_fragment + value_fragment, True),
                    ])

            for prefix, already_wire in product(candidates, wires):
                fragment, adds_entry = prefix
                next_state = (entries, current_size)
                if adds_entry:
                    next_state = advance_state(entries, current_size, name, value, HeaderTable.DEFAULT_SIZE)
                next_states.setdefault(next_state, {})[already_wire + fragment] = None

        states = next_states

    all_wires = [wire for wires in states.values() for wire in wires]
    decoder = Decoder()
    for wire in all_wires:
        decoder = Decoder()
        decoded = decoder.decode(wire, raw=True)
        assert [(name, value) for name, value, _ in headers] == decoded
        for (_, _, sensitive), decoded_header in zip(headers, decoded, strict=True):
            assert isinstance(decoded_header, NeverIndexedHeaderTuple) is sensitive
    return min(map(len, all_wires))


@pytest.mark.parametrize("headers", SHORT_CASES)
def test_shortest_encoding_matches_independent_exhaustive_search(headers):
    encoder = Encoder()
    decoder = Decoder()

    encoded = encoder.encode(headers, shortest=True)
    decoded = decoder.decode(encoded, raw=True)

    assert len(encoded) == brute_force_shortest(headers)
    assert [(header[0], header[1]) for header in normalize_headers(headers)] == decoded
    for expected, actual in zip(normalize_headers(headers), decoded, strict=True):
        assert isinstance(actual, NeverIndexedHeaderTuple) is expected[2]
    assert list(encoder.header_table.dynamic_entries) == list(decoder.header_table.dynamic_entries)
    assert encoder.header_table._current_size == decoder.header_table._current_size


def test_shortest_encoding_spends_now_to_save_in_later_headers():
    setup_headers = [
        (bytes([65 + index // 26, 65 + index - (26 * (index // 26))]), b"")
        for index in range(120)
    ] + [(b"x", b"y")] + [
        (bytes([65 + index]), b"")
        for index in range(65)
    ]
    target_headers = [(b"x", b"y")] * 4

    standard_encoder = Encoder()
    standard_decoder = Decoder()
    setup = standard_encoder.encode(setup_headers, huffman=False)
    assert standard_decoder.decode(setup, raw=True) == setup_headers
    standard_target = standard_encoder.encode(target_headers, huffman=False)

    shortest_encoder = Encoder()
    shortest_decoder = Decoder()
    setup = shortest_encoder.encode(setup_headers, huffman=False)
    assert shortest_decoder.decode(setup, raw=True) == setup_headers
    shortest_target = shortest_encoder.encode(target_headers, shortest=True)

    assert len(shortest_target) < len(standard_target)
    assert shortest_target[0] == 0x7f
    assert shortest_decoder.decode(shortest_target, raw=True) == target_headers
    assert (
        list(shortest_encoder.header_table.dynamic_entries)
        == list(shortest_decoder.header_table.dynamic_entries)
    )

    next_headers = [(b"x", b"y"), (b"after", b"z")]
    next_block = shortest_encoder.encode(next_headers, huffman=False)
    assert shortest_decoder.decode(next_block, raw=True) == next_headers
    assert (
        list(shortest_encoder.header_table.dynamic_entries)
        == list(shortest_decoder.header_table.dynamic_entries)
    )


def test_shortest_encoding_rejects_more_than_ten_headers_without_mutating_state():
    encoder = Encoder()
    encoder.header_table_size = 0
    headers = [(f"name-{index}".encode(), b"value") for index in range(11)]

    with pytest.raises(ValueError, match="shortest encoding is limited to 10 headers; got 11"):
        encoder.encode(headers, shortest=True)

    assert encoder.header_table.resized is True
    assert encoder.table_size_changes == [0]

    fallback = encoder.encode([(b":method", b"GET")], huffman=False)
    assert fallback == b"\x20\x82"
    assert encoder.header_table.resized is False


def test_shortest_encoding_includes_pending_table_size_update():
    encoder = Encoder()
    decoder = Decoder()
    encoder.header_table_size = 0

    encoded = encoder.encode([(b"name", b"value")], shortest=True)

    assert encoded.startswith(b"\x20\x00")
    assert encoded[1] == 0x00
    assert encoded[2] == 0x83
    assert decoder.decode(encoded, raw=True) == [(b"name", b"value")]
    assert list(encoder.header_table.dynamic_entries) == []
    assert list(decoder.header_table.dynamic_entries) == []


def test_shortest_encoding_prefers_static_exact_index_over_dynamic_duplicate():
    encoder = Encoder()
    decoder = Decoder()
    setup = encoder.encode([(b":method", b"GET"), (b"x", b"y")], huffman=False)
    decoder.decode(setup, raw=True)

    encoded = encoder.encode([(b":method", b"GET")], shortest=True)

    assert encoded == b"\x82"
    assert decoder.decode(encoded, raw=True) == [(b":method", b"GET")]


def test_shortest_encoding_uses_visible_dynamic_table_after_shrink():
    encoder = Encoder()
    decoder = Decoder()
    encoder.encode([(b"zz", b"zz"), (b"yy", b"yy")], huffman=False)

    encoder.header_table_size = 35
    encoded = encoder.encode([(b"yy", b"yy")], shortest=True)

    assert encoded.startswith(b"?\x04\x00")
    assert decoder.decode(encoded, raw=True) == [(b"yy", b"yy")]
    assert list(encoder.header_table.dynamic_entries) == []
    assert list(decoder.header_table.dynamic_entries) == []


