import pytest

from hpack import Decoder, Encoder, HPACKDecodingError, OversizedHeaderListError


REQUEST_CASES = [
    [
        (":method", "GET"),
        (":scheme", "http"),
        (":path", "/"),
        (":authority", "www.example.com"),
    ],
    [
        (":method", "GET"),
        (":scheme", "http"),
        (":path", "/"),
        (":authority", "www.example.com"),
        ("cache-control", "no-cache"),
    ],
    [
        (":method", "GET"),
        (":scheme", "https"),
        (":path", "/index.html"),
        (":authority", "www.example.com"),
        ("custom-key", "custom-value"),
    ],
]


def encoded_requests(huffman):
    encoder = Encoder()
    return [encoder.encode(headers, huffman=huffman) for headers in REQUEST_CASES]


def dynamic_table(decoder):
    return list(decoder.header_table.dynamic_entries)


def split_at(data, cut):
    if cut == 0:
        return [b"", data]
    if cut == len(data):
        return [data, b""]
    return [data[:cut], b"", data[cut:]]


def stream_blocks(chunks_by_block, raw=False):
    decoder = Decoder()
    received = []

    for chunks in chunks_by_block:
        decoder.begin_header_block(raw=raw)
        for chunk in chunks:
            received.extend(decoder.feed_header_block(chunk))
        assert decoder.end_header_block() == []

    return decoder, received


@pytest.mark.parametrize("huffman", [False, True])
@pytest.mark.parametrize("mode", ["split", "byte"])
def test_rfc_request_sequence_every_feed_boundary(huffman, mode):
    wires = encoded_requests(huffman)

    reference = Decoder()
    expected = []
    reference_tables = []
    for wire in wires:
        expected.extend(reference.decode(wire))
        reference_tables.append(dynamic_table(reference))

    if mode == "split":
        chunkings = []
        for target_index, wire in enumerate(wires):
            for cut in range(len(wire) + 1):
                blocks = []
                for index, complete_wire in enumerate(wires):
                    if index == target_index:
                        blocks.append(split_at(complete_wire, cut))
                    else:
                        blocks.append([complete_wire])
                chunkings.append(blocks)
    else:
        chunkings = []
        for target_index, wire in enumerate(wires):
            blocks = []
            for index, complete_wire in enumerate(wires):
                if index == target_index:
                    blocks.append([b"", *(bytes((byte,)) for byte in complete_wire), b""])
                else:
                    blocks.append([complete_wire])
            chunkings.append(blocks)

    for blocks in chunkings:
        decoder, received = stream_blocks(blocks)

        assert received == expected
        assert dynamic_table(decoder) == reference_tables[-1]


def test_all_rfc_request_tables_match_after_each_block():
    wires = encoded_requests(False)
    reference = Decoder()
    decoder = Decoder()
    expected = []

    for index, wire in enumerate(wires):
        expected.extend(reference.decode(wire))

        decoder.begin_header_block()
        received = decoder.feed_header_block(wire)
        assert decoder.end_header_block() == []
        assert dynamic_table(decoder) == dynamic_table(reference)

    assert received[-1] == REQUEST_CASES[-1][-1]


def test_empty_feeds_do_not_complete_a_truncated_block():
    wires = encoded_requests(False)
    data = wires[1]

    decoder = Decoder()
    decoder.begin_header_block()
    decoder.feed_header_block(wires[0])
    decoder.end_header_block()

    decoder.begin_header_block()
    decoder.feed_header_block(data[:-1])
    decoder.feed_header_block(b"")
    decoder.feed_header_block(b"")

    with pytest.raises(HPACKDecodingError, match="Truncated"):
        decoder.end_header_block()

    assert decoder.failed
    assert dynamic_table(decoder) == [(b":authority", b"www.example.com")]

    with pytest.raises(HPACKDecodingError, match="failed"):
        decoder.begin_header_block()
    with pytest.raises(HPACKDecodingError, match="failed"):
        decoder.feed_header_block(b"")
    with pytest.raises(HPACKDecodingError, match="failed"):
        decoder.end_header_block()
    with pytest.raises(HPACKDecodingError, match="failed"):
        decoder.decode(b"")


@pytest.mark.parametrize("cut", range(1, 6))
def test_dynamic_table_update_spans_feeds_once(cut):
    data = b"?a?\xe1\x1f\x82\x87\x84A\x8a\x08\x9d\\\x0b\x81p\xdcy\xa6\x99"

    decoder = Decoder()
    decoder.begin_header_block(raw=True)
    received = []
    for chunk in (data[:cut], b"", data[cut:]):
        received.extend(decoder.feed_header_block(chunk))
    decoder.end_header_block()

    assert received == [
        (b":method", b"GET"),
        (b":scheme", b"https"),
        (b":path", b"/"),
        (b":authority", b"127.0.0.1:8443"),
    ]


def test_incremental_header_is_not_inserted_twice_after_partial_retry():
    data = b"@\ncustom-key\x0ccustom-value"
    decoder = Decoder()

    decoder.begin_header_block(raw=True)
    decoder.feed_header_block(data[:15])
    decoder.feed_header_block(b"")
    decoder.feed_header_block(data[15:])
    decoder.end_header_block()

    assert dynamic_table(decoder) == [(b"custom-key", b"custom-value")]


def test_incomplete_huffman_string_is_not_an_error_until_end():
    data = encoded_requests(True)[0]
    cut = data.index(b"\xf1") + 1

    decoder = Decoder()
    decoder.begin_header_block()
    received = decoder.feed_header_block(data[:cut])
    decoder.feed_header_block(b"")

    assert received == [
        (":method", "GET"),
        (":scheme", "http"),
        (":path", "/"),
    ]
    with pytest.raises(HPACKDecodingError):
        decoder.end_header_block()
    assert decoder.failed


def test_multibyte_varint_string_length_spans_feeds():
    encoder = Encoder()
    value = "v" * 200
    data = encoder.encode([("name", value)], huffman=False)

    cut = data.index(b"\x7f") + 1
    decoder = Decoder()
    decoder.begin_header_block()
    received = decoder.feed_header_block(data[:cut])
    assert received == []
    received.extend(decoder.feed_header_block(data[cut:]))
    decoder.end_header_block()

    assert received == [("name", value)]
    assert dynamic_table(decoder) == [(b"name", value.encode())]


def test_overlong_varint_spanning_feeds_is_rejected():
    decoder = Decoder()

    decoder.begin_header_block(raw=True)
    decoder.feed_header_block(b"\x3f")
    for byte in (0xFF, 0xFF, 0xFF, 0xFF):
        decoder.feed_header_block(bytes((byte,)))

    with pytest.raises(HPACKDecodingError, match="too long"):
        decoder.feed_header_block(b"\x7f")

    assert decoder.failed


def test_oversized_header_fails_decoder_contract():
    data = b"\x14\x0c/sample/path"
    decoder = Decoder(max_header_list_size=44)

    decoder.begin_header_block()
    with pytest.raises(OversizedHeaderListError):
        decoder.feed_header_block(data)

    assert decoder.failed
    with pytest.raises(HPACKDecodingError, match="failed"):
        decoder.begin_header_block()


def test_partial_oversized_header_is_rejected_before_end():
    decoder = Decoder(max_header_list_size=64)
    data = b"\x40\x01a\x80\x01"

    decoder.begin_header_block(raw=True)
    decoder.feed_header_block(data)
    with pytest.raises(OversizedHeaderListError):
        decoder.feed_header_block(b"x" * 100)

    assert decoder.failed


def test_lifecycle_boundaries_are_explicit():
    decoder = Decoder()

    with pytest.raises(HPACKDecodingError, match="No HPACK"):
        decoder.feed_header_block(b"")
    with pytest.raises(HPACKDecodingError, match="No HPACK"):
        decoder.end_header_block()

    decoder.begin_header_block()
    with pytest.raises(HPACKDecodingError, match="already"):
        decoder.begin_header_block()

    assert decoder.feed_header_block(b"") == []
    assert decoder.end_header_block() == []


def test_streaming_raw_headers_remain_bytes():
    data = b"@\x01a\x01b"
    decoder = Decoder()

    decoder.begin_header_block(raw=True)
    received = decoder.feed_header_block(data)
    decoder.end_header_block()

    assert received == [(b"a", b"b")]
