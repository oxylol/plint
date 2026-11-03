"""JSON in and out: atomic writes, cached reads of run files, NaN-safe responses.

JSON has no NaN or infinity; channels pad with NaN (architecture 10) and the simulator writes
Python's ``NaN`` tokens, so every response goes through :func:`plain`, which turns non-finite
numbers into ``null`` (NumPy arrays are converted in one vectorised pass).
"""

from __future__ import annotations

import json
import math
import os
import threading
from collections import OrderedDict
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path, PurePath

import numpy as np


def array_to_list(array: np.ndarray) -> list:
    if array.dtype.kind in "fc":
        finite = np.isfinite(array)
        if finite.all():
            return array.tolist()
        values = array.astype(object)
        values[~finite] = None
        return values.tolist()
    if array.dtype.kind == "V":
        raise TypeError("structured arrays are not JSON-serialisable; convert each field")
    return array.tolist()


def plain(value):
    """A JSON-safe copy of ``value``: non-finite floats become None, NumPy values Python values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return array_to_list(value)
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, Enum):
        return plain(value.value)
    if isinstance(value, PurePath):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return plain(asdict(value))
    raise TypeError(f"not JSON-serialisable: {type(value).__name__}")


def dumps(value, *, indent: int | None = None) -> str:
    separators = (",", ":") if indent is None else (",", ": ")
    return json.dumps(plain(value), ensure_ascii=False, allow_nan=False, indent=indent, separators=separators)


def json_response(value, status_code: int = 200, headers: dict | None = None):
    """A Starlette response of ``plain(value)`` (imported here: the worker uses this module too)."""
    from starlette.responses import Response
    return Response(dumps(value).encode("utf-8"), status_code=status_code, headers=headers,
                    media_type="application/json")


def write_json_atomic(path: Path, value, *, fsync: bool = True, indent: int | None = 1) -> None:
    """Write next to the target and rename over it, so a reader sees the old or the new file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    data = dumps(value, indent=indent) + "\n"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())
    os.replace(tmp, path)


def read_json(path: Path):
    with open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def read_jsonl(path: Path, offset: int = 0) -> tuple[list, int]:
    """Complete lines from ``offset`` on (a line still being written is left for the next read).
    Returns the records and the offset after the last complete line."""
    records = []
    with open(path, "rb") as handle:
        handle.seek(offset)
        data = handle.read()
    end = data.rfind(b"\n")
    if end < 0:
        return records, offset
    for raw in data[: end + 1].splitlines():
        if raw.strip():
            records.append(json.loads(raw.decode("utf-8")))
    return records, offset + end + 1


class FileCache:
    """Parsed JSON files keyed by path and (inode, size, mtime): a replaced or appended file is
    re-read. Values are shared, so callers never mutate them."""

    def __init__(self, max_entries: int = 256, max_bytes: int = 256 * 1024 * 1024) -> None:
        self._entries: OrderedDict[str, tuple[tuple, int, object]] = OrderedDict()
        self._bytes = 0
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._lock = threading.Lock()

    @staticmethod
    def signature(path: Path) -> tuple:
        stat = os.stat(path)
        return (stat.st_ino, stat.st_size, stat.st_mtime_ns)

    def json(self, path: Path):
        key = str(path)
        signature = self.signature(path)
        with self._lock:
            hit = self._entries.get(key)
            if hit is not None and hit[0] == signature:
                self._entries.move_to_end(key)
                return hit[2]
        value = read_json(path)
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._bytes -= old[1]
            self._entries[key] = (signature, signature[1], value)
            self._bytes += signature[1]
            while self._entries and (len(self._entries) > self._max_entries or self._bytes > self._max_bytes):
                _, (_, size, _) = self._entries.popitem(last=False)
                self._bytes -= size
        return value
