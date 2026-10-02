"""
Structural tracing types for HPACK decoding.

These types deliberately contain no header names or header values. They are
intended for diagnosing compression state errors without exposing sensitive
header data through logs or trace output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple


class RepresentationType(str, Enum):
    """Identifies an HPACK header field representation."""

    INDEXED = "indexed"
    LITERAL_WITH_INCREMENTAL_INDEXING = "literal_with_incremental_indexing"
    LITERAL_WITHOUT_INDEXING = "literal_without_indexing"
    LITERAL_NEVER_INDEXED = "literal_never_indexed"
    DYNAMIC_TABLE_SIZE_UPDATE = "dynamic_table_size_update"


class TableEventType(str, Enum):
    """Identifies an observable dynamic table mutation."""

    TABLE_SIZE_CHANGE = "table_size_change"
    ENTRY_EVICTION = "entry_eviction"
    TABLE_CLEAR = "table_clear"


class TableClearReason(str, Enum):
    """Describes why a dynamic table was cleared."""

    TABLE_SIZE_UPDATE = "table_size_update"
    ENTRY_TOO_LARGE = "entry_too_large"


class ByteRange(NamedTuple):
    """A half-open byte range within one header block."""

    start: int
    end: int


@dataclass(frozen=True)
class DynamicEntryOrigin:
    """
    Identifies the representation that created a dynamic table entry.

    Sequence numbers are zero-based. Table size updates do not consume a header
    sequence number.
    """

    block_sequence: int
    header_sequence: int
    byte_range: ByteRange | None = None
    representation_type: RepresentationType | None = None

    @property
    def block_number(self) -> int:
        return self.block_sequence + 1

    @property
    def header_number(self) -> int:
        return self.header_sequence + 1

    @property
    def block(self) -> int:
        return self.block_number

    @property
    def header(self) -> int:
        return self.header_number


@dataclass(frozen=True)
class DynamicTableReference:
    """A reference to a dynamic table entry and its creation origin."""

    index: int
    origin: DynamicEntryOrigin | None
    name_only: bool = False


@dataclass(frozen=True)
class RepresentationTrace:
    """Structural information about one decoded HPACK representation."""

    byte_range: ByteRange
    representation_type: RepresentationType
    huffman: bool = False
    never_indexed: bool = False
    name_huffman: bool = False
    value_huffman: bool = False
    index: int | None = None
    inserted_index: int | None = None
    entry_reference: DynamicTableReference | None = None
    name_reference: DynamicTableReference | None = None

    @property
    def start(self) -> int:
        return self.byte_range.start

    @property
    def end(self) -> int:
        return self.byte_range.end

    @property
    def dynamic_reference(self) -> DynamicTableReference | None:
        return self.entry_reference or self.name_reference

    @property
    def dynamic_references(self) -> tuple[DynamicTableReference, ...]:
        return tuple(
            reference
            for reference in (self.entry_reference, self.name_reference)
            if reference is not None
        )


@dataclass(frozen=True)
class TableEvent:
    """Base class for dynamic table events."""

    byte_range: ByteRange
    event_type: TableEventType


@dataclass(frozen=True)
class TableSizeChange(TableEvent):
    """A dynamic table maximum size change."""

    old_max_size: int = 0
    new_max_size: int = 0
    old_size: int = 0
    new_size: int = 0
    event_type: TableEventType = field(default=TableEventType.TABLE_SIZE_CHANGE, init=False)


@dataclass(frozen=True)
class EntryEviction(TableEvent):
    """Eviction of one dynamic table entry."""

    entry: DynamicTableReference
    entry_size: int
    new_size: int
    event_type: TableEventType = field(default=TableEventType.ENTRY_EVICTION, init=False)


@dataclass(frozen=True)
class TableCleared(TableEvent):
    """Removal of every dynamic table entry."""

    reason: TableClearReason
    entries: tuple[DynamicTableReference, ...]
    old_size: int
    new_size: int = 0
    event_type: TableEventType = field(default=TableEventType.TABLE_CLEAR, init=False)


class DecodeTrace:
    """
    A collection of structural traces produced while decoding one block.

    The object is reset at the start of every ``Decoder.decode`` call it is
    passed to. It never stores header names or values.
    """

    def __init__(self) -> None:
        self.block_sequence: int | None = None
        self.representations: list[RepresentationTrace] = []
        self.table_events: list[TableEvent] = []
        self.complete = False
        self.error_offset: int | None = None
        self.error_representation_start: int | None = None

    @property
    def block_number(self) -> int | None:
        if self.block_sequence is None:
            return None
        return self.block_sequence + 1

    def reset(self, block_sequence: int) -> None:
        self.block_sequence = block_sequence
        self.representations.clear()
        self.table_events.clear()
        self.complete = False
        self.error_offset = None
        self.error_representation_start = None

    def add_representation(self, representation: RepresentationTrace) -> None:
        self.representations.append(representation)

    def add_table_event(self, event: TableEvent) -> None:
        self.table_events.append(event)

    @property
    def events(self) -> list[TableEvent]:
        return self.table_events

    @property
    def position(self) -> int | None:
        return self.error_offset

    @property
    def error_start(self) -> int | None:
        return self.error_representation_start

    @property
    def error_end(self) -> int | None:
        return self.error_offset

    def mark_error(self, offset: int, representation_start: int | None = None) -> None:
        if self.error_offset is None:
            self.error_offset = offset
        if self.error_representation_start is None and representation_start is not None:
            self.error_representation_start = representation_start
        self.complete = False

    def mark_complete(self) -> None:
        self.complete = True
