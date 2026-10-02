"""
Tests for the optional decode tracing added to ``hpack.Decoder``.
"""
import pytest

from hpack import (
    Decoder,
    Encoder,
    HPACKDecodingError,
    InvalidTableIndexError,
    NeverIndexedHeaderTuple,
)
from hpack.table import table_entry_size
from hpack.trace import (
    CLEAR_OVERSIZED_ENTRY,
    CLEAR_RESIZE,
    KIND_INDEXED,
    KIND_LITERAL_INCREMENTAL,
    KIND_LITERAL_NEVER_INDEXED,
    KIND_LITERAL_NO_INDEX,
    KIND_TABLE_SIZE_UPDATE,
    EntryOrigin,
)


def encode_blocks(encoder, headers_per_block, **encode_kwargs):
    """
    Encodes each entry of ``headers_per_block`` into its own header block.
    """
    return [encoder.encode(headers, **encode_kwargs) for headers in headers_per_block]


class TestRepresentationTraces:
    def test_indexed_static_representation(self):
        encoder = Encoder()
        decoder = Decoder(trace=True)
        block = encoder.encode([(b":method", b"GET")], huffman=False)

        decoder.decode(block)

        (trace,) = decoder.traces
        assert trace.block_index == 0
        assert trace.block_length == len(block)
        assert trace.complete
        assert trace.evictions == []
        assert trace.clears == []
        (representation,) = trace.representations
        assert representation.start == 0
        assert representation.end == len(block)
        assert representation.kind == KIND_INDEXED
        assert representation.index == 2
        # Static table entries have no origin.
        assert representation.origin is None
        assert not representation.huffman
        assert not representation.never_indexed
        assert representation.name is None
        assert representation.value is None

    def test_literal_representation_offsets_tile_block(self):
        encoder = Encoder()
        decoder = Decoder(trace=True)
        block = encoder.encode(
            [(b"custom-key", b"custom-value"), (b":method", b"GET")],
            huffman=True,
        )

        decoder.decode(block)

        (trace,) = decoder.traces
        first, second = trace.representations
        assert first.kind == KIND_LITERAL_INCREMENTAL
        assert first.huffman
        assert first.start == 0
        assert first.end == second.start
        assert second.kind == KIND_INDEXED
        assert not second.huffman
        assert second.end == len(block)

    def test_literal_without_indexing(self):
        decoder = Decoder(trace=True)
        # Literal header field without indexing, literal name and value.
        block = b"\x00\x08password\x06secret"

        headers = decoder.decode(block)

        assert headers == [("password", "secret")]
        (representation,) = decoder.traces[0].representations
        assert representation.kind == KIND_LITERAL_NO_INDEX
        assert not representation.never_indexed
        assert not representation.huffman
        assert representation.start == 0
        assert representation.end == len(block)

    def test_never_indexed_representation(self):
        encoder = Encoder()
        decoder = Decoder(trace=True)
        block = encoder.encode(
            [NeverIndexedHeaderTuple(b"password", b"secret")], huffman=False,
        )

        decoder.decode(block)

        (representation,) = decoder.traces[0].representations
        assert representation.kind == KIND_LITERAL_NEVER_INDEXED
        assert representation.never_indexed
        assert representation.name is None
        assert representation.value is None

    def test_literal_with_indexed_name(self):
        encoder = Encoder()
        decoder = Decoder(trace=True)
        blocks = encode_blocks(encoder, [
            [(b"custom-a", b"value-a")],
            [(b"custom-a", b"other-value")],
        ], huffman=False)
        for block in blocks:
            decoder.decode(block)

        representation = decoder.traces[1].representations[0]
        assert representation.kind == KIND_LITERAL_INCREMENTAL
        # The name was taken from the dynamic table entry created by block 0.
        assert representation.index == 62
        assert representation.origin == EntryOrigin(block_index=0, field_index=0)


class TestGenerations:
    def test_index_movement_resolves_to_correct_generation(self):
        encoder = Encoder()
        decoder = Decoder(trace=True)
        blocks = encode_blocks(encoder, [
            [(b"custom-a", b"value-a")],
            [(b"custom-b", b"value-b")],
            [(b"custom-a", b"value-a")],  # indexed reference after movement
            [(b"custom-b", b"value-b")],  # reuses numeric index 62
        ], huffman=False)
        for block in blocks:
            decoder.decode(block)

        # Block 0: a literal with incremental indexing creates the entry.
        created = decoder.traces[0].representations[0]
        assert created.kind == KIND_LITERAL_INCREMENTAL

        # Block 2: inserting custom-b moved custom-a from index 62 to 63.
        moved = decoder.traces[2].representations[0]
        assert moved.kind == KIND_INDEXED
        assert moved.index == 63
        assert moved.origin == EntryOrigin(block_index=0, field_index=0)

        # Block 3: numeric index 62 is reused, but it now refers to the
        # entry created by block 1.
        reused = decoder.traces[3].representations[0]
        assert reused.kind == KIND_INDEXED
        assert reused.index == 62
        assert reused.origin == EntryOrigin(block_index=1, field_index=0)

    def test_eviction_is_observable_with_origin(self):
        encoder = Encoder()
        decoder = Decoder(trace=True)
        encoder.header_table_size = 64
        blocks = encode_blocks(encoder, [
            [(b"custom-a", b"value-a")],
            [(b"custom-b", b"value-b")],  # evicts custom-a
            [(b"custom-b", b"value-b")],  # indexed reference to the survivor
        ], huffman=False)
        for block in blocks:
            decoder.decode(block)

        # The first block announces the table size change.
        update = decoder.traces[0].representations[0]
        assert update.kind == KIND_TABLE_SIZE_UPDATE
        assert update.table_size == 64
        assert update.start == 0

        # The second block evicts custom-a: the eviction names the block and
        # field that created the evicted entry.
        (eviction,) = decoder.traces[1].evictions
        assert eviction.entry_size == table_entry_size(b"custom-a", b"value-a")
        assert eviction.origin == EntryOrigin(block_index=0, field_index=0)
        assert eviction.at == decoder.traces[1].representations[0].start

        # Index 62 used to be custom-a's generation; after the eviction it
        # resolves to custom-b's generation.
        survivor = decoder.traces[2].representations[0]
        assert survivor.kind == KIND_INDEXED
        assert survivor.index == 62
        assert survivor.origin == EntryOrigin(block_index=1, field_index=0)

    def test_table_clear_is_observable(self):
        encoder = Encoder()
        decoder = Decoder(trace=True)
        blocks = encode_blocks(encoder, [
            [(b"custom-a", b"value-a")],
            [(b"custom-b", b"value-b")],
        ], huffman=False)
        encoder.header_table_size = 0
        blocks += encode_blocks(encoder, [[(b"custom-c", b"value-c")]], huffman=False)
        encoder.header_table_size = 4096
        blocks += encode_blocks(encoder, [
            [(b"custom-d", b"value-d")],
            [(b"custom-d", b"value-d")],
        ], huffman=False)
        for block in blocks:
            decoder.decode(block)

        # Block 2 shrinks the table to zero, discarding both entries.
        clear = decoder.traces[2].clears[0]
        assert clear.reason == CLEAR_RESIZE
        assert clear.entries_discarded == 2
        assert clear.at == 0

        # custom-c does not fit in a zero-sized table, so it clears the
        # (already empty) table again instead of being added.
        oversized = decoder.traces[2].clears[1]
        assert oversized.reason == CLEAR_OVERSIZED_ENTRY
        assert oversized.entries_discarded == 0

        # After the clear, new entries get fresh generations: index 62 now
        # resolves to the entry created by block 3.
        recreated = decoder.traces[4].representations[0]
        assert recreated.kind == KIND_INDEXED
        assert recreated.index == 62
        assert recreated.origin == EntryOrigin(block_index=3, field_index=0)


class TestFailedBlocks:
    def test_failed_block_keeps_partial_trace(self):
        encoder = Encoder()
        decoder = Decoder(trace=True)
        good = encoder.encode([(b"custom-a", b"value-a")], huffman=False)
        truncated = good + b"\x40\x05cus"  # truncated literal name

        with pytest.raises(HPACKDecodingError):
            decoder.decode(truncated)

        (trace,) = decoder.traces
        assert not trace.complete
        # Only the first representation was fully decoded: the trace must
        # not claim the truncated bytes were decoded.
        (representation,) = trace.representations
        assert representation.end == len(good)
        assert representation.end < trace.block_length

    def test_failed_block_without_any_representation(self):
        decoder = Decoder(trace=True)

        with pytest.raises(InvalidTableIndexError):
            decoder.decode(b"\xbe")  # indexed field, index 62, empty table

        (trace,) = decoder.traces
        assert not trace.complete
        assert trace.representations == []

    def test_failed_block_without_tracing_has_no_trace(self):
        decoder = Decoder()

        with pytest.raises(InvalidTableIndexError):
            decoder.decode(b"\xbe")

        assert decoder.traces == []


class TestTraceValues:
    def test_values_captured_only_when_enabled(self):
        encoder = Encoder()
        block = encoder.encode([(b"custom-a", b"value-a")], huffman=False)

        structural = Decoder(trace=True)
        structural.decode(block)
        (representation,) = structural.traces[0].representations
        assert representation.name is None
        assert representation.value is None

        capturing = Decoder(trace=True, trace_values=True)
        capturing.decode(block)
        (representation,) = capturing.traces[0].representations
        assert representation.name == b"custom-a"
        assert representation.value == b"value-a"

    def test_never_indexed_values_never_captured(self):
        encoder = Encoder()
        block = encoder.encode(
            [NeverIndexedHeaderTuple(b"password", b"secret")], huffman=False,
        )

        decoder = Decoder(trace=True, trace_values=True)
        decoder.decode(block)

        (representation,) = decoder.traces[0].representations
        assert representation.never_indexed
        assert representation.name is None
        assert representation.value is None


class TestTracingDisabled:
    def test_output_identical_with_and_without_tracing(self):
        encoder = Encoder()
        plain = Decoder()
        traced = Decoder(trace=True)

        encoder.header_table_size = 64
        blocks = encode_blocks(encoder, [
            [(b"custom-a", b"value-a")],
            [(b"custom-b", b"value-b")],  # evicts custom-a
            [NeverIndexedHeaderTuple(b"password", b"secret")],
            [(b"custom-b", b"value-b")],  # indexed reference
        ], huffman=True)
        encoder.header_table_size = 0
        blocks += encode_blocks(encoder, [[(b"custom-c", b"value-c")]], huffman=True)

        for block in blocks:
            assert plain.decode(block) == traced.decode(block)

        assert plain.traces == []
        assert len(traced.traces) == len(blocks)
