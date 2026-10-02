hpack API
=========

This document provides the HPACK API.

.. autoclass:: hpack.Encoder
   :members: header_table_size, encode

``SHORTEST_HEADER_BLOCK_LIMIT`` is the maximum number of header fields that
``Encoder.encode(..., shortest=True)`` accepts. This mode performs a complete
search over the legal indexed and literal representations, dynamic table
insertions, and per-field Huffman choices. Inputs above the limit raise
``ValueError`` and callers should use the regular streaming encoder.

.. autoclass:: hpack.Decoder
   :members: header_table_size, decode

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
