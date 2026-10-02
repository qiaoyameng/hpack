hpack API
=========

This document provides the HPACK API.

.. autoclass:: hpack.Encoder
   :members: header_table_size, encode

.. autoclass:: hpack.Decoder
   :members: header_table_size, begin_header_block, feed_header_block, end_header_block, decode, fail

Streamed header blocks use ``begin_header_block()``,
``feed_header_block()`` and ``end_header_block()``. Each call to
``feed_header_block()`` returns the complete headers made available by that
chunk. If a block ends while an integer, string or Huffman code is still
incomplete, ``end_header_block()`` raises an error. Any decoding error sets
``Decoder.failed`` to ``True`` and the persistent HPACK context must not be
reused.

.. autoclass:: hpack.HeaderTuple
   :members: indexable

.. autoclass:: hpack.NeverIndexedHeaderTuple
   :members: indexable

.. autoclass:: hpack.HPACKError

.. autoclass:: hpack.HPACKDecodingError

.. autoclass:: hpack.InvalidTableIndexError

.. autoclass:: hpack.InvalidTableIndex

.. autoclass:: hpack.OversizedHeaderListError

.. autoclass:: hpack.InvalidTableSizeError
