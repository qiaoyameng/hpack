# flake8: noqa
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
from typing import Optional

from .exceptions import InvalidTableIndex
from .trace import DynamicEntryOrigin, TableClearReason

log = logging.getLogger(__name__)


def table_entry_size(name: bytes, value: bytes) -> int:
    """
    Calculates the size of a single entry

    This size is mostly irrelevant to us and defined
    specifically to accommodate memory management for
    lower level implementations. The 32 extra bytes are
    considered the "maximum" overhead that would be
    required to represent each entry in the table.

    See RFC7541 Section 4.1
    """
    return 32 + len(name) + len(value)


class DynamicTableEntry(tuple[bytes, bytes]):
    """A dynamic table entry with optional creation provenance."""

    def __new__(
        cls,
        name: bytes,
        value: bytes,
        origin: DynamicEntryOrigin | None = None,
    ) -> DynamicTableEntry:
        entry = tuple.__new__(cls, (name, value))
        entry.origin = origin
        return entry


@dataclass(frozen=True)
class RemovedDynamicEntry:
    """An entry removed from the dynamic table."""

    index: int
    entry_size: int
    origin: DynamicEntryOrigin | None


@dataclass(frozen=True)
class DynamicTableMutation:
    """Structural changes made by one table operation."""

    inserted_index: int | None = None
    evicted_entries: tuple[RemovedDynamicEntry, ...] = ()
    cleared: bool = False
    clear_reason: TableClearReason | None = None
    old_size: int = 0
    new_size: int = 0


@dataclass(frozen=True)
class DynamicTableResize:
    """Result of changing the dynamic table maximum size."""

    old_max_size: int
    new_max_size: int
    mutation: DynamicTableMutation


class HeaderTable:
    """
    Implements the combined static and dynamic header table

    The name and value arguments for all the functions
    should ONLY be byte strings (b'') however this is not
    strictly enforced in the interface.

    See RFC7541 Section 2.3
    """
    #: Default maximum size of the dynamic table. See
    #:  RFC7540 Section 6.5.2.
    DEFAULT_SIZE = 4096

    #: Constant list of static headers. See RFC7541 Section
    #:  2.3.1 and Appendix A
    STATIC_TABLE = (
        (b':authority'                  , b''             ),  # noqa
        (b':method'                     , b'GET'          ),  # noqa
        (b':method'                     , b'POST'         ),  # noqa
        (b':path'                       , b'/'            ),  # noqa
        (b':path'                       , b'/index.html'  ),  # noqa
        (b':scheme'                     , b'http'         ),  # noqa
        (b':scheme'                     , b'https'        ),  # noqa
        (b':status'                     , b'200'          ),  # noqa
        (b':status'                     , b'204'          ),  # noqa
        (b':status'                     , b'206'          ),  # noqa
        (b':status'                     , b'304'          ),  # noqa
        (b':status'                     , b'400'          ),  # noqa
        (b':status'                     , b'404'          ),  # noqa
        (b':status'                     , b'500'          ),  # noqa
        (b'accept-charset'              , b''             ),  # noqa
        (b'accept-encoding'             , b'gzip, deflate'),  # noqa
        (b'accept-language'             , b''             ),  # noqa
        (b'accept-ranges'               , b''             ),  # noqa
        (b'accept'                      , b''             ),  # noqa
        (b'access-control-allow-origin' , b''             ),  # noqa
        (b'age'                         , b''             ),  # noqa
        (b'allow'                       , b''             ),  # noqa
        (b'authorization'               , b''             ),  # noqa
        (b'cache-control'               , b''             ),  # noqa
        (b'content-disposition'         , b''             ),  # noqa
        (b'content-encoding'            , b''             ),  # noqa
        (b'content-language'            , b''             ),  # noqa
        (b'content-length'              , b''             ),  # noqa
        (b'content-location'            , b''             ),  # noqa
        (b'content-range'               , b''             ),  # noqa
        (b'content-type'                , b''             ),  # noqa
        (b'cookie'                      , b''             ),  # noqa
        (b'date'                        , b''             ),  # noqa
        (b'etag'                        , b''             ),  # noqa
        (b'expect'                      , b''             ),  # noqa
        (b'expires'                     , b''             ),  # noqa
        (b'from'                        , b''             ),  # noqa
        (b'host'                        , b''             ),  # noqa
        (b'if-match'                    , b''             ),  # noqa
        (b'if-modified-since'           , b''             ),  # noqa
        (b'if-none-match'               , b''             ),  # noqa
        (b'if-range'                    , b''             ),  # noqa
        (b'if-unmodified-since'         , b''             ),  # noqa
        (b'last-modified'               , b''             ),  # noqa
        (b'link'                        , b''             ),  # noqa
        (b'location'                    , b''             ),  # noqa
        (b'max-forwards'                , b''             ),  # noqa
        (b'proxy-authenticate'          , b''             ),  # noqa
        (b'proxy-authorization'         , b''             ),  # noqa
        (b'range'                       , b''             ),  # noqa
        (b'referer'                     , b''             ),  # noqa
        (b'refresh'                     , b''             ),  # noqa
        (b'retry-after'                 , b''             ),  # noqa
        (b'server'                      , b''             ),  # noqa
        (b'set-cookie'                  , b''             ),  # noqa
        (b'strict-transport-security'   , b''             ),  # noqa
        (b'transfer-encoding'           , b''             ),  # noqa
        (b'user-agent'                  , b''             ),  # noqa
        (b'vary'                        , b''             ),  # noqa
        (b'via'                         , b''             ),  # noqa
        (b'www-authenticate'            , b''             ),  # noqa
    )  # noqa

    STATIC_TABLE_LENGTH = len(STATIC_TABLE)

    STATIC_TABLE_MAPPING: dict[bytes, tuple[int, dict[bytes, int]]]

    def __init__(self) -> None:
        self._maxsize = HeaderTable.DEFAULT_SIZE
        self._current_size = 0
        self.resized = False
        self.block_sequence = 0
        self.dynamic_entries: deque[DynamicTableEntry] = deque()

    def advance_block_sequence(self) -> int:
        """
        Marks the start of another decoded header block and returns its
        zero-based sequence number.
        """
        block_sequence = self.block_sequence
        self.block_sequence += 1
        return block_sequence

    def get_by_index(self, index: int) -> tuple[bytes, bytes]:
        """
        Returns the entry specified by index

        Note that the table is 1-based ie an index of 0 is
        invalid.  This is due to the fact that a zero value
        index signals that a completely unindexed header
        follows.

        The entry will either be from the static table or
        the dynamic table depending on the value of index.
        """
        original_index = index
        index -= 1
        if 0 <= index:
            if index < HeaderTable.STATIC_TABLE_LENGTH:
                return HeaderTable.STATIC_TABLE[index]

            index -= HeaderTable.STATIC_TABLE_LENGTH
            if index < len(self.dynamic_entries):
                return self.dynamic_entries[index]

        raise InvalidTableIndex("Invalid table index %d" % original_index)

    def __repr__(self) -> str:
        return "HeaderTable(%d, %s, %r)" % (
            self._maxsize,
            self.resized,
            self.dynamic_entries
        )

    def get_entry_by_index(
        self,
        index: int,
    ) -> tuple[bytes, bytes, DynamicEntryOrigin | None]:
        """
        Returns the name, value, and optional origin of the specified entry.
        """
        original_index = index
        index -= 1
        if 0 <= index:
            if index < HeaderTable.STATIC_TABLE_LENGTH:
                name, value = HeaderTable.STATIC_TABLE[index]
                return name, value, None

            index -= HeaderTable.STATIC_TABLE_LENGTH
            if index < len(self.dynamic_entries):
                entry = self.dynamic_entries[index]
                return entry[0], entry[1], entry.origin

        raise InvalidTableIndex("Invalid table index %d" % original_index)

    def add(
        self,
        name: bytes,
        value: bytes,
        origin: DynamicEntryOrigin | None = None,
    ) -> DynamicTableMutation:
        """
        Adds a new entry to the table

        We reduce the table size if the entry will make the
        table size greater than maxsize.
        """
        # We just clear the table if the entry is too big
        size = table_entry_size(name, value)
        if size > self._maxsize:
            removed = self._clear(TableClearReason.ENTRY_TOO_LARGE)
            self.dynamic_entries.clear()
            self._current_size = 0
            return DynamicTableMutation(
                inserted_index=None,
                cleared=True,
                clear_reason=TableClearReason.ENTRY_TOO_LARGE,
                evicted_entries=removed.evicted_entries,
                old_size=removed.old_size,
                new_size=0,
            )
        else:
            # Add new entry
            old_size = self._current_size
            self.dynamic_entries.appendleft(DynamicTableEntry(name, value, origin))
            self._current_size += size
            evicted_entries = self._shrink()
            return DynamicTableMutation(
                inserted_index=HeaderTable.STATIC_TABLE_LENGTH + 1,
                evicted_entries=tuple(evicted_entries),
                old_size=old_size,
                new_size=self._current_size,
            )

    def search(self, name: bytes, value: bytes) -> Optional[tuple[int, bytes, Optional[bytes]]]:
        """
        Searches the table for the entry specified by name
        and value

        Returns one of the following:
            - ``None``, no match at all
            - ``(index, name, None)`` for partial matches on name only.
            - ``(index, name, value)`` for perfect matches.
        """
        partial = None

        header_name_search_result = HeaderTable.STATIC_TABLE_MAPPING.get(name)
        if header_name_search_result:
            index = header_name_search_result[1].get(value)
            if index is not None:
                return index, name, value
            else:
                partial = (header_name_search_result[0], name, None)

        offset = HeaderTable.STATIC_TABLE_LENGTH + 1
        for (i, (n, v)) in enumerate(self.dynamic_entries):
            if n == name:
                if v == value:
                    return i + offset, n, v
                elif partial is None:
                    partial = (i + offset, n, None)
        return partial

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @maxsize.setter
    def maxsize(self, newmax: int) -> None:
        self.resize(newmax)

    def resize(self, newmax: int) -> DynamicTableResize:
        """
        Changes the maximum dynamic table size and reports structural changes.
        """
        newmax = int(newmax)
        log.debug("Resizing header table to %d from %d", newmax, self._maxsize)
        oldmax = self._maxsize
        old_size = self._current_size
        self._maxsize = newmax
        self.resized = (newmax != oldmax)
        mutation: DynamicTableMutation
        if newmax <= 0:
            removed = self._clear(TableClearReason.TABLE_SIZE_UPDATE)
            self.dynamic_entries.clear()
            self._current_size = 0
            mutation = DynamicTableMutation(
                cleared=True,
                clear_reason=TableClearReason.TABLE_SIZE_UPDATE,
                evicted_entries=removed.evicted_entries,
                old_size=old_size,
                new_size=0,
            )
        elif oldmax > newmax:
            evicted_entries = self._shrink()
            mutation = DynamicTableMutation(
                evicted_entries=tuple(evicted_entries),
                old_size=old_size,
                new_size=self._current_size,
            )
        else:
            mutation = DynamicTableMutation(old_size=old_size, new_size=old_size)
        return DynamicTableResize(oldmax, newmax, mutation)

    def _clear(self, reason: TableClearReason) -> DynamicTableMutation:
        removed_entries = []
        for position in range(len(self.dynamic_entries), 0, -1):
            entry = self.dynamic_entries[position - 1]
            name, value = entry
            removed_entries.append(
                RemovedDynamicEntry(
                    index=HeaderTable.STATIC_TABLE_LENGTH + position,
                    entry_size=table_entry_size(name, value),
                    origin=getattr(entry, "origin", None),
                ),
            )
        return DynamicTableMutation(
            cleared=True,
            clear_reason=reason,
            evicted_entries=tuple(removed_entries),
            old_size=self._current_size,
            new_size=0,
        )

    def _shrink(
        self,
    ) -> list[RemovedDynamicEntry]:
        """
        Shrinks the dynamic table to be at or below maxsize
        """
        cursize = self._current_size
        removed_entries = []
        while cursize > self._maxsize:
            entry = self.dynamic_entries.pop()
            name, value = entry
            entry_size = table_entry_size(name, value)
            index = HeaderTable.STATIC_TABLE_LENGTH + len(self.dynamic_entries) + 1
            removed_entries.append(
                RemovedDynamicEntry(index, entry_size, entry.origin),
            )
            cursize -= entry_size
            log.debug("Evicting dynamic table entry at index %d", index)
        self._current_size = cursize
        return removed_entries


def _build_static_table_mapping() -> dict[bytes, tuple[int, dict[bytes, int]]]:
    """
    Build static table mapping from header name to tuple with next structure:
    (<minimal index of header>, <mapping from header value to it index>).

    static_table_mapping used for hash searching.
    """
    static_table_mapping: dict[bytes, tuple[int, dict[bytes, int]]] = {}
    for index, (name, value) in enumerate(HeaderTable.STATIC_TABLE, 1):
        header_name_search_result = static_table_mapping.setdefault(name, (index, {}))
        header_name_search_result[1][value] = index
    return static_table_mapping


HeaderTable.STATIC_TABLE_MAPPING = _build_static_table_mapping()
