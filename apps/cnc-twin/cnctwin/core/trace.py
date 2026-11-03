"""The chain of cause: every physical event a finding depends on, as JSON lines.

A domain records an event when it computes something a finding may rest on (a force peak, a
flute fill, a temperature, a coolant shadow) with the ids of the events it was derived from.
A finding then carries the chain of ids back to its roots, and ``twin explain`` prints it.

Only an id -> byte-offset index is kept in memory (events are read back from the file when a
chain is followed), so a long run's trace costs little RAM. :meth:`TraceLog.resume` reopens the
file at a checkpoint's byte offset, truncating what was written after it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TraceEvent:
    id: int
    time_s: float
    tick: int
    domain: str
    event: str
    quantities: dict[str, float] = field(default_factory=dict)
    parents: list[int] = field(default_factory=list)
    line: int = 0


class TraceLog:
    """Append-only event log with sequential ids (deterministic)."""

    def __init__(self, path: Path | None = None, truncate_at: int | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._offsets: list[int] = []
        self._memory: list[TraceEvent] = []      # used when there is no file (tests)
        self._file = None
        if self.path is not None:
            mode = "r+b" if (truncate_at is not None and self.path.exists()) else "wb"
            self._file = open(self.path, mode)
            if truncate_at is not None:
                self._file.truncate(truncate_at)
                self._file.seek(truncate_at)
                self._rebuild_index(truncate_at)

    def _rebuild_index(self, length: int) -> None:
        self._file.seek(0)
        position = 0
        while position < length:
            raw = self._file.readline()
            if not raw:
                break
            self._offsets.append(position)
            position += len(raw)
        self._file.seek(length)

    def record(self, time_s: float, tick: int, domain: str, event: str,
               quantities: dict | None = None, parents: list[int] | None = None, line: int = 0) -> int:
        event_id = len(self._offsets) if self._file is not None else len(self._memory)
        trace_event = TraceEvent(event_id, float(time_s), int(tick), domain, event,
                                 {k: _plain(v) for k, v in sorted((quantities or {}).items())},
                                 [int(p) for p in (parents or [])], int(line))
        if self._file is None:
            self._memory.append(trace_event)
            return event_id
        self._offsets.append(self._file.tell())
        self._file.write((json.dumps(trace_event.__dict__, sort_keys=True) + "\n").encode("utf-8"))
        return event_id

    def get(self, event_id: int) -> TraceEvent | None:
        if self._file is None:
            return self._memory[event_id] if 0 <= event_id < len(self._memory) else None
        if not 0 <= event_id < len(self._offsets):
            return None
        position = self._file.tell()
        self._file.flush()
        with open(self.path, "rb") as reader:
            reader.seek(self._offsets[event_id])
            record = json.loads(reader.readline())
        self._file.seek(position)
        return TraceEvent(**record)

    def chain(self, event_id: int) -> list[TraceEvent]:
        """The event and all its ancestors, breadth first, each once."""
        seen: set[int] = set()
        order: list[TraceEvent] = []
        queue = [event_id]
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            event = self.get(current)
            if event is None:
                continue
            seen.add(current)
            order.append(event)
            queue.extend(event.parents)
        return order

    def chain_ids(self, event_id: int) -> list[int]:
        return [e.id for e in self.chain(event_id)]

    def byte_length(self) -> int:
        """Checkpoint cut: the file length now."""
        if self._file is None:
            return 0
        self._file.flush()
        return self._file.tell()

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @classmethod
    def resume(cls, path: Path, byte_offset: int) -> "TraceLog":
        """Reopen after a checkpoint: drop everything written after ``byte_offset``."""
        return cls(path, truncate_at=byte_offset)


def _plain(value):
    """JSON-safe scalar (NumPy scalars become Python floats/ints)."""
    if hasattr(value, "item") and not hasattr(value, "__len__"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value
