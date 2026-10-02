"""
HTTP/2 header encoding for Python.
"""
from __future__ import annotations

from .exceptions import HPACKDecodingError, HPACKError, InvalidTableIndex, InvalidTableIndexError, InvalidTableSizeError, OversizedHeaderListError
from .hpack import Decoder, Encoder
from .struct import HeaderTuple, NeverIndexedHeaderTuple
from .trace import BlockTrace, EntryOrigin, Eviction, RepresentationTrace, TableClear

__all__ = [
    "BlockTrace",
    "Decoder",
    "Encoder",
    "EntryOrigin",
    "Eviction",
    "HPACKDecodingError",
    "HPACKError",
    "HeaderTuple",
    "InvalidTableIndex",
    "InvalidTableIndexError",
    "InvalidTableSizeError",
    "NeverIndexedHeaderTuple",
    "OversizedHeaderListError",
    "RepresentationTrace",
    "TableClear",
]

__version__ = "4.2.0"
