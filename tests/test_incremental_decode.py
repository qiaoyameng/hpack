"""
Tests for incremental (streaming) HPACK decoding via
``Decoder.start_header_block``/``Decoder.feed``/``Decoder.end_header_block``.

The RFC 7541 appendix examples used by the one-shot decoder tests are fed to
the incremental decoder split at every possible single cut point, byte-by-byte
and with empty fragments interleaved, and the decoded header sequences and the
final dynamic table state are compared against the one-shot ``decode`` API.
"""
import pytest

from hpack import Decoder, HPACKDecodingError, OversizedHeaderListError


# RFC 7541 Appendix C.3: request examples without Huffman coding.
REQUEST_EXAMPLES_NO_HUFFMAN = [
    b'\x82\x86\x84\x01\x0fwww.example.com',
    b'\x82\x86\x84\x01\x0fwww.example.com\x0f\t\x08no-cache',
    b'\x82\x87\x85\x01\x0fwww.example.com@\ncustom-key\x0ccustom-value',
]

# RFC 7541 Appendix C.4: request examples with Huffman coding.
REQUEST_EXAMPLES_HUFFMAN = [
    b'\x82\x86\x84\x01\x8c\xf1\xe3\xc2\xe5\xf2:k\xa0\xab\x90\xf4\xff',
    (
        b'\x82\x86\x84\x01\x8c\xf1\xe3\xc2\xe5\xf2:k\xa0\xab\x90\xf4\xff'
        b'\x0f\t\x86\xa8\xeb\x10d\x9c\xbf'
    ),
    (
        b'\x82\x87\x85\x01\x8c\xf1\xe3\xc2\xe5\xf2:k\xa0\xab\x90\xf4\xff@'
        b'\x88%\xa8I\xe9[\xa9}\x7f\x89%\xa8I\xe9[\xb8\xe8\xb4\xbf'
    ),
]

# A block that starts with two dynamic table size updates, exercising
# encoding-context updates that may be split across fragments.
TABLE_SIZE_UPDATE_BLOCK = [
    b'?a?\xe1\x1f\x82\x87\x84A\x8a\x08\x9d\\\x0b\x81p\xdcy\xa6\x99',
]

# A large block that populates much of the dynamic table.
RESIZING_BLOCK = [
    (
        b'\x82\x87D\x87a\x07\xa4\xacV4\xcfA\x8c\xf1\xe3\xc2\xe5\xf2:k\xa0'
        b'\xab\x90\xf4\xff@\x88%\xa8I\xe9[\xa9}\x7f\x89%\xa8I\xe9[\xb8\xe8'
        b'\xb4\xbfz\xbc\xd0\x7ff\xa2\x81\xb0\xda\xe0S\xfa\xd02\x1a\xa4\x9d'
        b'\x13\xfd\xa9\x92\xa4\x96\x854\x0c\x8aj\xdc\xa7\xe2\x81\x02\xef}'
        b'\xa9g{\x81qp\x7fjb):\x9d\x81\x00 \x00@\x150\x9a\xc2\xca\x7f,\x05'
        b'\xc5\xc1S\xb0I|\xa5\x89\xd3M\x1fC\xae\xba\x0cA\xa4\xc7\xa9\x8f3'
        b'\xa6\x9a?\xdf\x9ah\xfa\x1du\xd0b\r&=Ly\xa6\x8f\xbe\xd0\x01w\xfe'
        b'\xbeX\xf9\xfb\xed\x00\x17{@\x8a\xfc[=\xbdF\x81\xad\xbc\xa8O\x84y'
        b'\xe7\xde\x7f'
    ),
]

# Assorted single-representation blocks: literal with incremental indexing,
# literal without indexing, indexed, never indexed, and an empty block.
MISC_BLOCKS = [
    b'\x40\x0acustom-key\x0dcustom-header',
    b'\x04\x0c/sample/path',
    b'\x82',
    b'\x10\x0acustom-key\x0dcustom-header',
    b'\x14\x0c/sample/path',
    b'',
]

RFC_SEQUENCES = {
    'rfc7541-c3-no-huffman': REQUEST_EXAMPLES_NO_HUFFMAN,
    'rfc7541-c4-huffman': REQUEST_EXAMPLES_HUFFMAN,
    'table-size-updates': TABLE_SIZE_UPDATE_BLOCK,
    'resizing': RESIZING_BLOCK,
    'misc': MISC_BLOCKS,
}


def decode_one_shot(blocks):
    """
    Decodes the block sequence with the one-shot API, returning the decoder
    and the list of per-block header lists.
    """
    decoder = Decoder()
    headers = [decoder.decode(block, raw=True) for block in blocks]
    return decoder, headers


def decode_incrementally(blocks, chunkers):
    """
    Decodes the block sequence incrementally, chunking each block with the
    corresponding chunker function. Returns the decoder and the list of
    per-block header lists.

    Also asserts the core streaming invariant: the headers emitted
    progressively by ``feed`` are exactly the headers returned by
    ``end_header_block`` -- none are lost and none are emitted twice.
    """
    decoder = Decoder()
    all_headers = []
    for block, chunker in zip(blocks, chunkers):
        decoder.start_header_block()
        emitted = []
        for fragment in chunker(block):
            emitted.extend(decoder.feed(fragment))
        final = decoder.end_header_block(raw=True)
        assert [tuple(h) for h in emitted] == [tuple(h) for h in final]
        all_headers.append(final)
    return decoder, all_headers


def assert_matches_one_shot(blocks, chunkers):
    """
    Decodes the same sequence one-shot and incrementally, and compares the
    header sequences and the final dynamic table state.
    """
    ref_decoder, ref_headers = decode_one_shot(blocks)
    decoder, headers = decode_incrementally(blocks, chunkers)

    assert headers == ref_headers
    # The final dynamic table must be identical: in particular, no header
    # may have been inserted twice because of re-parsing.
    assert (
        list(decoder.header_table.dynamic_entries)
        == list(ref_decoder.header_table.dynamic_entries)
    )
    assert decoder.header_table_size == ref_decoder.header_table_size
    return decoder, headers


def whole(block):
    return [block]


def byte_by_byte(block):
    return [block[i:i + 1] for i in range(len(block))]


def byte_by_byte_with_empty(block):
    fragments = []
    for i in range(len(block)):
        fragments.append(b'')
        fragments.append(block[i:i + 1])
        fragments.append(b'')
    return fragments


class TestIncrementalDecodeRFCExamples:
    @pytest.mark.parametrize('sequence_name', sorted(RFC_SEQUENCES))
    def test_single_cut_points(self, sequence_name):
        """
        Each block of the sequence is split at every possible single cut
        point (one block at a time) and fed as two fragments.
        """
        blocks = RFC_SEQUENCES[sequence_name]
        for split_block in range(len(blocks)):
            block = blocks[split_block]
            for cut in range(len(block) + 1):
                chunkers = [whole] * len(blocks)
                chunkers[split_block] = (
                    lambda b, cut=cut: [b[:cut], b[cut:]]
                )
                assert_matches_one_shot(blocks, chunkers)

    @pytest.mark.parametrize('sequence_name', sorted(RFC_SEQUENCES))
    def test_byte_by_byte(self, sequence_name):
        """
        The whole sequence is fed one byte at a time.
        """
        blocks = RFC_SEQUENCES[sequence_name]
        chunkers = [byte_by_byte] * len(blocks)
        assert_matches_one_shot(blocks, chunkers)

    @pytest.mark.parametrize('sequence_name', sorted(RFC_SEQUENCES))
    def test_empty_fragments_interleaved(self, sequence_name):
        """
        Empty fragments may be fed at any point without affecting decoding.
        """
        blocks = RFC_SEQUENCES[sequence_name]
        chunkers = [byte_by_byte_with_empty] * len(blocks)
        assert_matches_one_shot(blocks, chunkers)

    def test_empty_feed_returns_no_headers(self):
        decoder = Decoder()
        decoder.start_header_block()
        assert decoder.feed(b'') == []
        assert decoder.feed(b'') == []
        assert decoder.end_header_block() == []

    def test_no_double_insertion_into_dynamic_table(self):
        """
        Feeding byte-by-byte must not insert any header into the dynamic
        table twice: the third RFC C.3 block indexes exactly one entry.
        """
        decoder = Decoder()
        for block in REQUEST_EXAMPLES_NO_HUFFMAN:
            decoder.start_header_block()
            for i in range(len(block)):
                decoder.feed(block[i:i + 1])
            decoder.end_header_block()
        assert list(decoder.header_table.dynamic_entries) == [
            (b'custom-key', b'custom-value'),
        ]

    def test_feed_emits_headers_as_soon_as_complete(self):
        """
        Headers completed by a fragment are returned by that feed call,
        before the block ends.
        """
        decoder = Decoder()
        block = REQUEST_EXAMPLES_NO_HUFFMAN[0]
        decoder.start_header_block()
        # The first three bytes are three complete indexed fields.
        assert decoder.feed(block[:3]) == [
            (b':method', b'GET'),
            (b':scheme', b'http'),
            (b':path', b'/'),
        ]
        # The remainder is an incomplete literal: nothing more is emitted
        # until the final bytes arrive.
        assert decoder.feed(block[3:-1]) == []
        assert decoder.feed(block[-1:]) == [
            (b':authority', b'www.example.com'),
        ]
        assert decoder.end_header_block() == [
            (':method', 'GET'),
            (':scheme', 'http'),
            (':path', '/'),
            (':authority', 'www.example.com'),
        ]

    def test_incremental_and_one_shot_can_be_mixed(self):
        """
        A block decoded incrementally and a block decoded with decode()
        share the same compression context.
        """
        decoder = Decoder()
        decoder.start_header_block()
        decoder.feed(REQUEST_EXAMPLES_NO_HUFFMAN[0])
        decoder.end_header_block()
        headers = decoder.decode(REQUEST_EXAMPLES_NO_HUFFMAN[1])
        ref = Decoder()
        ref.decode(REQUEST_EXAMPLES_NO_HUFFMAN[0])
        assert headers == ref.decode(REQUEST_EXAMPLES_NO_HUFFMAN[1])


class TestIncrementalDecodeBoundaries:
    def test_feed_without_start_rejected(self):
        decoder = Decoder()
        with pytest.raises(HPACKDecodingError):
            decoder.feed(b'\x82')

    def test_end_without_start_rejected(self):
        decoder = Decoder()
        with pytest.raises(HPACKDecodingError):
            decoder.end_header_block()

    def test_start_twice_rejected(self):
        decoder = Decoder()
        decoder.start_header_block()
        with pytest.raises(HPACKDecodingError):
            decoder.start_header_block()

    def test_decode_during_block_rejected(self):
        decoder = Decoder()
        decoder.start_header_block()
        with pytest.raises(HPACKDecodingError):
            decoder.decode(b'\x82')
        # The block is still in progress and can be completed normally.
        assert decoder.feed(b'\x82') == [(b':method', b'GET')]
        assert decoder.end_header_block() == [(':method', 'GET')]

    def test_truncated_block_at_end_rejected(self):
        """
        Bytes that are merely missing mid-block become a truncation error
        once the block is explicitly ended. Prefixes that happen to be
        complete header blocks must still decode successfully.
        """
        block = b'\x40\x0acustom-key\x0dcustom-header'
        for cut in range(len(block) + 1):
            decoder = Decoder()
            decoder.start_header_block()
            decoder.feed(block[:cut])
            try:
                expected = Decoder().decode(block[:cut])
            except HPACKDecodingError:
                with pytest.raises(HPACKDecodingError):
                    decoder.end_header_block()
            else:
                assert decoder.end_header_block() == expected

    def test_truncated_varint_at_end_rejected(self):
        # A dynamic table size update whose variable-length integer is
        # missing its continuation byte.
        decoder = Decoder()
        decoder.start_header_block()
        assert decoder.feed(b'\x3f') == []
        with pytest.raises(HPACKDecodingError):
            decoder.end_header_block()

    def test_incomplete_data_mid_block_is_not_an_error(self):
        # Half of a Huffman-coded string, a split string length and a split
        # variable-length integer must all be tolerated mid-block.
        block = REQUEST_EXAMPLES_HUFFMAN[0]
        decoder = Decoder()
        decoder.start_header_block()
        for i in range(len(block)):
            decoder.feed(block[i:i + 1])
        assert decoder.end_header_block() == [
            (':method', 'GET'),
            (':scheme', 'http'),
            (':path', '/'),
            (':authority', 'www.example.com'),
        ]


class TestIncrementalDecodeFailureContract:
    """
    After any decoding failure the decoder is permanently failed: every
    further decoding operation raises HPACKDecodingError, and the caller
    never has to guess whether the decoder can still be used.
    """
    def assert_failed(self, decoder):
        with pytest.raises(HPACKDecodingError):
            decoder.start_header_block()
        with pytest.raises(HPACKDecodingError):
            decoder.feed(b'\x82')
        with pytest.raises(HPACKDecodingError):
            decoder.end_header_block()
        with pytest.raises(HPACKDecodingError):
            decoder.decode(b'\x82')

    def test_truncation_failure_is_terminal(self):
        decoder = Decoder()
        decoder.start_header_block()
        decoder.feed(b'\x40\x0acustom')
        with pytest.raises(HPACKDecodingError):
            decoder.end_header_block()
        self.assert_failed(decoder)

    def test_invalid_data_failure_is_terminal(self):
        decoder = Decoder()
        decoder.start_header_block()
        with pytest.raises(HPACKDecodingError):
            # An index well beyond the end of the table.
            decoder.feed(b'\xbe')
        self.assert_failed(decoder)

    def test_oversized_header_list_failure_is_terminal(self):
        decoder = Decoder(max_header_list_size=44)
        decoder.start_header_block()
        with pytest.raises(OversizedHeaderListError):
            decoder.feed(b'\x14\x0c/sample/path')
        self.assert_failed(decoder)

    def test_oversized_header_list_checked_incrementally(self):
        """
        The cumulative header list size limit applies across fragments of
        the same block, not just to whole blocks.
        """
        decoder = Decoder(max_header_list_size=90)
        decoder.start_header_block()
        decoder.feed(b'\x04\x0c/sample/path')
        with pytest.raises(OversizedHeaderListError):
            decoder.feed(b'\x04\x0c/sample/path')
        self.assert_failed(decoder)

    def test_invalid_table_size_failure_is_terminal(self):
        decoder = Decoder()
        decoder.max_allowed_table_size = 127
        decoder.start_header_block()
        with pytest.raises(HPACKDecodingError):
            decoder.feed(b'?a')
        self.assert_failed(decoder)

    def test_fresh_decoder_still_works(self):
        decoder = Decoder()
        decoder.start_header_block()
        with pytest.raises(HPACKDecodingError):
            decoder.feed(b'\xbe')
        # A new decoder is unaffected by the failure of the old one.
        assert Decoder().decode(b'\x82') == [(':method', 'GET')]
