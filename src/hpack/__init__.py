"""
HTTP/2 header encoding for Python.
"""
from __future__ import annotations

from .exceptions import HPACKDecodingError, HPACKError, InvalidTableIndex, InvalidTableIndexError, InvalidTableSizeError, OversizedHeaderListError
from .hpack import Decoder, Encoder
from .struct import HeaderTuple, NeverIndexedHeaderTuple
from .trace import (
    ByteRange,
    DecodeTrace,
    DynamicEntryOrigin,
    DynamicTableReference,
    EntryEviction,
    RepresentationTrace,
    RepresentationType,
    TableClearReason,
    TableCleared,
    TableEvent,
    TableEventType,
    TableSizeChange,
)

__all__ = [
    "Decoder",
    "ByteRange",
    "DecodeTrace",
    "DynamicEntryOrigin",
    "DynamicTableReference",
    "Encoder",
    "EntryEviction",
    "HPACKDecodingError",
    "HPACKError",
    "HeaderTuple",
    "InvalidTableIndex",
    "InvalidTableIndexError",
    "InvalidTableSizeError",
    "NeverIndexedHeaderTuple",
    "OversizedHeaderListError",
    "RepresentationTrace",
    "RepresentationType",
    "TableClearReason",
    "TableCleared",
    "TableEvent",
    "TableEventType",
    "TableSizeChange",
]

__version__ = "4.2.0"
