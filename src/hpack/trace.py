"""
Structures for optionally tracing HPACK header block decoding.

When decoding fails it is rarely enough to know that "index 62" was used:
after the next insertion, index 62 refers to a different entry. The structures
in this module allow a :class:`hpack.Decoder` to record, for every decoded
header block, exactly which representations were decoded, which byte ranges
they occupied, and which dynamic table entries they referenced, expressed in
terms of the block and field that originally created the entry (its
"generation") rather than a bare numeric index.

Traces are structural only: by default they contain byte offsets, field
kinds, table indices, entry sizes and entry origins, but never header names
or values, so they are safe to write to diagnostic logs. Even when value
capture is explicitly enabled, fields that were marked never-indexed by the
peer never have their values recorded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: Representation kinds recorded in :attr:`RepresentationTrace.kind`.
KIND_INDEXED = "indexed"
KIND_LITERAL_INCREMENTAL = "literal_incremental_indexing"
KIND_LITERAL_NO_INDEX = "literal_without_indexing"
KIND_LITERAL_NEVER_INDEXED = "literal_never_indexed"
KIND_TABLE_SIZE_UPDATE = "table_size_update"

#: Reasons recorded in :attr:`TableClear.reason`.
CLEAR_RESIZE = "resize"
CLEAR_OVERSIZED_ENTRY = "oversized_entry"


@dataclass(frozen=True)
class EntryOrigin:
    """
    Identifies the header field whose decoding created a dynamic table entry.

    This acts as a "generation" marker: even when a numeric table index is
    later reused for a different entry, the origin still unambiguously
    identifies which entry a reference resolved to.
    """

    #: Sequence number of the decoded header block (0-based, per decoder).
    block_index: int

    #: Sequence number of the header field within that block (0-based).
    field_index: int


@dataclass(frozen=True)
class RepresentationTrace:
    """
    A single field representation (or table size update) within a block.
    """

    #: Byte offset of the first byte of this representation in the block.
    start: int

    #: Byte offset one past the last byte of this representation.
    end: int

    #: One of the ``KIND_*`` constants from this module.
    kind: str

    #: Whether any string literal in this representation was Huffman-coded.
    huffman: bool = False

    #: Whether this representation forbids adding the field to any table.
    never_indexed: bool = False

    #: The HPACK table index referenced, if this representation uses one.
    index: int | None = None

    #: The origin of the referenced dynamic table entry, or ``None`` if the
    #: index refers to the static table, no index is used, or the entry
    #: predates tracing.
    origin: EntryOrigin | None = None

    #: For ``KIND_TABLE_SIZE_UPDATE`` representations, the new table size.
    table_size: int | None = None

    #: The decoded header name. Only populated when the decoder was created
    #: with ``trace_values=True`` and the field is indexable.
    name: bytes | None = None

    #: The decoded header value. Only populated when the decoder was created
    #: with ``trace_values=True`` and the field is indexable.
    value: bytes | None = None


@dataclass(frozen=True)
class Eviction:
    """
    Records the eviction of a dynamic table entry.
    """

    #: Byte offset of the representation whose processing caused the eviction.
    at: int

    #: The size of the evicted entry as defined by RFC 7541 Section 4.1.
    entry_size: int

    #: The origin of the evicted entry, or ``None`` if it predates tracing.
    origin: EntryOrigin | None = None


@dataclass(frozen=True)
class TableClear:
    """
    Records the dynamic table being emptied.
    """

    #: Byte offset of the representation whose processing caused the clear.
    at: int

    #: One of the ``CLEAR_*`` constants from this module.
    reason: str

    #: How many entries were discarded.
    entries_discarded: int


@dataclass
class BlockTrace:
    """
    The trace of decoding a single header block.
    """

    #: Sequence number of this block (0-based, per decoder).
    block_index: int

    #: The length of the encoded header block in bytes.
    block_length: int

    #: The representations decoded, in order of appearance.
    representations: list[RepresentationTrace] = field(default_factory=list)

    #: Dynamic table evictions that occurred while decoding this block.
    evictions: list[Eviction] = field(default_factory=list)

    #: Times the dynamic table was emptied while decoding this block.
    clears: list[TableClear] = field(default_factory=list)

    #: Whether the whole block was decoded. If decoding raised an exception,
    #: this is ``False`` and the trace contains only the representations that
    #: were fully decoded before the error: no claim is made about the bytes
    #: that follow the last recorded representation.
    complete: bool = True
