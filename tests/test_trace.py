import logging

import pytest

from hpack import (
    DecodeTrace,
    Decoder,
    Encoder,
    HPACKDecodingError,
    RepresentationType,
    TableClearReason,
    TableEventType,
)


def test_trace_records_representation_structure():
    decoder = Decoder()
    trace = DecodeTrace()
    encoder = Encoder()
    huffman_data = bytearray(
        encoder.encode([("custom-key", "custom-header")], huffman=True),
    )
    huffman_data[0] = 0x10

    data = b"\x82" + bytes(huffman_data)
    decoder.decode(data, trace=trace)

    indexed, never_indexed = trace.representations

    assert indexed.byte_range == (0, 1)
    assert indexed.representation_type == RepresentationType.INDEXED
    assert indexed.huffman is False
    assert indexed.never_indexed is False
    assert indexed.index == 2

    assert never_indexed.byte_range == (1, len(data))
    assert (
        never_indexed.representation_type
        == RepresentationType.LITERAL_NEVER_INDEXED
    )
    assert never_indexed.huffman is True
    assert never_indexed.name_huffman is True
    assert never_indexed.value_huffman is True
    assert never_indexed.never_indexed is True


def test_dynamic_references_track_index_movement_eviction_and_clear():
    traced = Decoder()
    traced.max_allowed_table_size = 102
    untraced = Decoder()
    untraced.max_allowed_table_size = 102
    trace = DecodeTrace()

    resize_to_102 = b"?G"
    entry_a = b"@\x01a\x01b"
    entry_b = b"@\x01b\x01c"
    entry_c = b"@\x01c\x01d"
    entry_d = b"@\x01d\x01e"
    block_zero = resize_to_102 + entry_a + entry_b + entry_c
    block_one = entry_d + b"\xbe\xbf"
    block_two = b"\x20"
    block_three = b"\x20" + resize_to_102 + entry_a + b"\xbe"

    blocks = (block_zero, block_one, block_two, block_three)
    traced_blocks = [
        traced.decode(block, trace=trace)
        for block in blocks
    ]
    untraced_blocks = [untraced.decode(block) for block in blocks]

    assert traced_blocks == untraced_blocks
    assert trace.block_sequence == 3

    traces = [DecodeTrace() for _ in blocks]
    single_block_traces = []
    trace_decoder = Decoder()
    trace_decoder.max_allowed_table_size = 102
    for block, block_trace in zip(blocks, traces):
        single_block_traces.append(
            trace_decoder.decode(block, trace=block_trace),
        )
    assert single_block_traces == untraced_blocks

    recreated_d = traces[1].representations[1].entry_reference
    moved_c = traces[1].representations[2].entry_reference
    recreated_a = traces[3].representations[3].entry_reference

    assert recreated_d is not None
    assert recreated_d.index == 62
    assert (
        recreated_d.origin.block_sequence,
        recreated_d.origin.header_sequence,
    ) == (1, 0)

    created_c = traces[0].representations[3]
    assert created_c.byte_range == (12, 17)
    assert created_c.inserted_index == 62

    assert moved_c is not None
    assert moved_c.index == 63
    assert (
        moved_c.origin.block_sequence,
        moved_c.origin.header_sequence,
    ) == (0, 2)
    assert moved_c.origin.byte_range == created_c.byte_range

    assert recreated_a is not None
    assert recreated_a.index == 62
    assert (recreated_a.origin.block_sequence, recreated_a.origin.header_sequence) == (3, 0)

    [eviction] = [
        event
        for event in traces[1].table_events
        if event.event_type == TableEventType.ENTRY_EVICTION
    ]
    assert eviction.entry.index == 65
    assert (
        eviction.entry.origin.block_sequence,
        eviction.entry.origin.header_sequence,
    ) == (0, 0)
    assert eviction.entry_size == 34
    assert eviction.new_size == 102

    [cleared] = [
        event
        for event in traces[2].table_events
        if event.event_type == TableEventType.TABLE_CLEAR
    ]
    assert cleared.reason == TableClearReason.TABLE_SIZE_UPDATE
    assert cleared.old_size == 102
    assert cleared.new_size == 0
    assert [entry.index for entry in cleared.entries] == [64, 63, 62]
    assert [
        (entry.origin.block_sequence, entry.origin.header_sequence)
        for entry in cleared.entries
    ] == [(0, 1), (0, 2), (1, 0)]
    assert [
        event.event_type
        for event in traces[2].table_events
    ] == [
        TableEventType.TABLE_SIZE_CHANGE,
        TableEventType.TABLE_CLEAR,
    ]


def test_trace_does_not_contain_or_log_sensitive_values(caplog):
    decoder = Decoder()
    trace = DecodeTrace()
    data = b"\x10\x0dauthorization\x06secret"

    with caplog.at_level(logging.DEBUG, logger="hpack"):
        headers = decoder.decode(data, trace=trace)

    assert headers == [("authorization", "secret")]
    assert isinstance(headers[0][0], str)
    assert "secret" not in repr(trace)
    assert "authorization" not in repr(trace)
    assert not any("secret" in record.getMessage() for record in caplog.records)


def test_trace_keeps_completed_representations_on_error():
    decoder = Decoder()
    trace = DecodeTrace()
    data = b"\x82\xbf"

    with pytest.raises(HPACKDecodingError) as exc_info:
        decoder.decode(data, trace=trace)

    assert exc_info.value.trace is trace
    assert trace.complete is False
    assert trace.error_offset == 2
    assert trace.error_representation_start == 1
    assert len(trace.representations) == 1
    assert trace.representations[0].byte_range == (0, 1)


def test_trace_reports_actual_truncation_offset():
    decoder = Decoder()
    trace = DecodeTrace()
    data = b"\x82\x00\x85\xf2\xb2J"

    with pytest.raises(HPACKDecodingError) as exc_info:
        decoder.decode(data, trace=trace)

    assert exc_info.value.trace is trace
    assert trace.complete is False
    assert trace.error_offset == len(data)
    assert trace.error_representation_start == 1
    assert [representation.byte_range for representation in trace.representations] == [(0, 1)]
