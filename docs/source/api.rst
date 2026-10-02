hpack API
=========

This document provides the HPACK API.

.. autoclass:: hpack.Encoder
   :members: header_table_size, encode

.. autoclass:: hpack.Decoder
   :members: header_table_size, decode

.. autoclass:: hpack.DecodeTrace

.. autoclass:: hpack.RepresentationTrace

.. autoclass:: hpack.DynamicTableReference

.. autoclass:: hpack.DynamicEntryOrigin

.. autoclass:: hpack.ByteRange

.. autoclass:: hpack.RepresentationType
   :members:
   :undoc-members:

.. autoclass:: hpack.TableSizeChange

.. autoclass:: hpack.EntryEviction

.. autoclass:: hpack.TableCleared

.. autoclass:: hpack.TableEventType
   :members:
   :undoc-members:

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
