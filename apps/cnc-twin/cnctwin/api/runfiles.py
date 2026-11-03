"""Reading run directories (architecture 16). The API and the UI read only these files.

Layouts assumed (written by the simulation core; tests/fixtures/make_synthetic_run.py writes the
same, and docs/api.md "Run directory as read by the API" lists them):

* ``run.json`` (complete run, no volatile fields), ``status.json`` (volatile: ``status`` and
  ``progress``). A folder with neither is not a run. A run without status.json but with run.json
  reads as ``done``.
* ``findings.json``: a list of findings, or ``{"findings": [...]}``; ``trace.jsonl``: one event per
  line, ``id`` = line number (cnctwin.core.trace), so an event is found by binary search on byte
  offsets without an index.
* ``scene.json["frames"]`` describes the records of ``frames.bin``: ``record_bytes`` and
  ``fields`` [{name, dtype, shape, offset}] with an int64 ``tick`` field, records in tick order.
* ``keyframes/index.jsonl``: ``{k, tick, time_s, glb, fields, chips, ...}`` per keyframe;
  ``keyframes/bricks.idx``: JSON ``{"bricks": {"<brick>": [[k, file, offset, length], ...]}}``
  (JSON lines ``{brick, k, file, offset, length}`` are read too).
* ``checkpoints/ck_<tick>.json`` is the commit marker of ``ck_<tick>.npz``.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

import numpy as np

from .jsonio import FileCache, read_jsonl
from .safety import ApiError, check_id, resolve_inside

DEFAULT_TICK_S = 1e-6           # architecture 5.1: the base clock is fixed
TERMINAL = ("done", "failed", "cancelled")
_CHECKPOINT = re.compile(r"^ck_(\d+)\.json$")
_RUN_ID = re.compile(r"^(\d{5,})-")


class RunDir:
    """One run directory."""

    def __init__(self, run_id: str, path: Path, cache: FileCache) -> None:
        self.id = run_id
        self.path = path
        self.cache = cache

    # ------------------------------------------------------------------ plain files
    def file(self, relative: str) -> Path:
        return resolve_inside(self.path, relative)

    def has(self, relative: str) -> bool:
        return (self.path / relative).is_file()

    def json(self, relative: str, *, missing: str | None = None):
        path = self.path / relative
        try:
            return self.cache.json(path)
        except FileNotFoundError:
            raise ApiError(404, missing or f"run {self.id!r} has no {relative} (yet)") from None
        except ValueError as error:
            raise ApiError(500, f"run {self.id!r}: {relative} is not valid JSON ({error})") from None

    def optional_json(self, relative: str):
        try:
            return self.cache.json(self.path / relative)
        except FileNotFoundError:
            return None
        except ValueError:
            return None

    def run_json(self) -> dict | None:
        return self.optional_json("run.json")

    def status(self) -> dict:
        try:
            status = self.cache.json(self.path / "status.json")
        except FileNotFoundError:
            status = None
        except ValueError as error:
            return {"format": "cnctwin-run-status", "version": 1, "status": "unknown", "progress": {},
                    "synthesized": True, "error": {"type": "UnreadableStatus",
                                                   "message": f"status.json is not valid JSON ({error})"}}
        if isinstance(status, dict):
            return status
        if self.has("run.json"):
            return {"format": "cnctwin-run-status", "version": 1, "status": "done", "progress": {},
                    "synthesized": True}
        return {"format": "cnctwin-run-status", "version": 1, "status": "unknown", "progress": {},
                "synthesized": True}

    def state(self) -> str:
        return str(self.status().get("status", "unknown"))

    def tick_s(self) -> float:
        run = self.run_json() or {}
        clock = run.get("clock") if isinstance(run.get("clock"), dict) else {}
        if isinstance(clock.get("tick_s"), (int, float)):
            return float(clock["tick_s"])
        scene = self.optional_json("scene.json") or {}
        frames = scene.get("frames") if isinstance(scene, dict) else None
        if isinstance(frames, dict) and isinstance(frames.get("tick_s"), (int, float)):
            return float(frames["tick_s"])
        return DEFAULT_TICK_S

    # ------------------------------------------------------------------ findings, channels, timeline
    def findings(self) -> list[dict]:
        if not self.has("findings.json"):
            return []
        document = self.json("findings.json")
        items = document.get("findings", []) if isinstance(document, dict) else document
        if not isinstance(items, list):
            raise ApiError(500, f"run {self.id!r}: findings.json holds no list of findings")
        return items

    def channels(self) -> list[dict]:
        document = self.json("channels.json")
        items = document.get("channels", []) if isinstance(document, dict) else document
        return items if isinstance(items, list) else []

    def channel_units(self) -> dict[str, str]:
        try:
            return {c["name"]: c.get("unit", "") for c in self.channels() if isinstance(c, dict) and "name" in c}
        except ApiError:
            return {}

    def timeline_blocks(self) -> tuple[dict, list[dict]]:
        document = self.json("timeline.json")
        if isinstance(document, list):
            return {}, document
        blocks = document.get("blocks", []) if isinstance(document, dict) else []
        meta = {k: v for k, v in document.items() if k != "blocks"} if isinstance(document, dict) else {}
        return meta, blocks if isinstance(blocks, list) else []

    # ------------------------------------------------------------------ trace
    def trace_event(self, event_id: int) -> dict | None:
        path = self.path / "trace.jsonl"
        if not path.is_file():
            raise ApiError(404, f"run {self.id!r} has no trace.jsonl (yet)")
        with open(path, "rb") as handle:
            return _trace_search(handle, os.path.getsize(path), event_id)

    def trace_chain(self, event_id: int, max_events: int = 200) -> list[dict]:
        """The event and its ancestors, breadth first, each once (as cnctwin.core.trace.TraceLog.chain)."""
        path = self.path / "trace.jsonl"
        if not path.is_file():
            raise ApiError(404, f"run {self.id!r} has no trace.jsonl (yet)")
        size = os.path.getsize(path)
        seen: set[int] = set()
        order: list[dict] = []
        queue = [event_id]
        with open(path, "rb") as handle:
            while queue and len(order) < max_events:
                current = queue.pop(0)
                if current in seen:
                    continue
                event = _trace_search(handle, size, current)
                if event is None:
                    continue
                seen.add(current)
                order.append(event)
                queue.extend(int(p) for p in event.get("parents", []))
        return order

    # ------------------------------------------------------------------ frames.bin
    def frames_header(self) -> dict:
        scene = self.json("scene.json")
        header = scene.get("frames") if isinstance(scene, dict) else None
        if not isinstance(header, dict) or "fields" not in header or "record_bytes" not in header:
            raise ApiError(500, f"run {self.id!r}: scene.json has no frames header")
        return header

    def frame_records(self):
        """(memmap of records or None, header, dtype)."""
        header = self.frames_header()
        dtype = frame_dtype(header)
        path = self.path / header.get("file", "frames.bin")
        if not path.is_file():
            raise ApiError(404, f"run {self.id!r} has no frames.bin (yet)")
        count = os.path.getsize(path) // dtype.itemsize
        if count == 0:
            return None, header, dtype
        return np.memmap(path, dtype=dtype, mode="r", shape=(count,)), header, dtype

    # ------------------------------------------------------------------ keyframes
    def keyframes(self) -> list[dict]:
        path = self.path / "keyframes" / "index.jsonl"
        if not path.is_file():
            return []
        records, _ = read_jsonl(path)
        return sorted((r for r in records if isinstance(r, dict) and "k" in r), key=lambda r: int(r["k"]))

    def keyframe(self, k: int) -> dict:
        for entry in self.keyframes():
            if int(entry["k"]) == k:
                return entry
        raise ApiError(404, f"run {self.id!r} has no keyframe {k}")

    def keyframe_at(self, time_s: float | None) -> dict:
        frames = self.keyframes()
        if not frames:
            raise ApiError(404, f"run {self.id!r} has no keyframes (yet)")
        if time_s is None:
            return frames[-1]
        chosen = frames[0]
        tick_s = self.tick_s()
        for entry in frames:
            if _entry_time(entry, tick_s) <= time_s + 1e-12:
                chosen = entry
            else:
                break
        return chosen

    def keyframe_time(self, k: int) -> float | None:
        for entry in self.keyframes():
            if int(entry["k"]) == k:
                return _entry_time(entry, self.tick_s())
        return None

    def bricks_index(self) -> dict[str, list[tuple[int, str, int, int]]]:
        path = self.path / "keyframes" / "bricks.idx"
        if not path.is_file():
            raise ApiError(404, f"run {self.id!r} has no keyframes/bricks.idx (yet)")
        text = path.read_text(encoding="utf-8")
        out: dict[str, list] = {}
        try:
            document = json.loads(text)
            bricks = document.get("bricks", {}) if isinstance(document, dict) else {}
            for brick, entries in bricks.items():
                out[str(brick)] = [_brick_entry(e) for e in entries]
        except json.JSONDecodeError:
            for line in text.splitlines():
                if line.strip():
                    record = json.loads(line)
                    out.setdefault(str(record["brick"]), []).append(_brick_entry(record))
        for entries in out.values():
            entries.sort(key=lambda e: e[0])
        return out

    def npz(self, relative: str, names: list[str] | None = None) -> dict[str, np.ndarray]:
        path = self.file(relative)
        try:
            with np.load(path, allow_pickle=False) as archive:
                keys = list(archive.files)
                if names:
                    missing = [n for n in names if n not in keys]
                    if missing:
                        raise ApiError(404, f"{relative} has no array {missing[0]!r} (it has {', '.join(keys)})")
                    keys = [k for k in keys if k in names]
                return {key: archive[key] for key in keys}
        except (ValueError, OSError) as error:
            raise ApiError(500, f"run {self.id!r}: {relative} is not a readable .npz ({error})") from None

    # ------------------------------------------------------------------ checkpoints
    def checkpoints(self) -> list[dict]:
        """Committed checkpoints (marker .json present and its .npz too), oldest first."""
        directory = self.path / "checkpoints"
        if not directory.is_dir():
            return []
        found = []
        for path in directory.iterdir():
            match = _CHECKPOINT.match(path.name)
            if not match or not path.with_suffix(".npz").is_file():
                continue
            try:
                marker = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            found.append({"tick": int(match.group(1)), "file": f"checkpoints/{path.name}", "path": str(path),
                          "time_s": marker.get("time_s"), "op": marker.get("op")})
        return sorted(found, key=lambda c: c["tick"])

    def latest_checkpoint(self) -> dict | None:
        found = self.checkpoints()
        return found[-1] if found else None


def _entry_time(entry: dict, tick_s: float) -> float:
    if isinstance(entry.get("time_s"), (int, float)):
        return float(entry["time_s"])
    if isinstance(entry.get("tick"), (int, float)):
        return float(entry["tick"]) * tick_s
    return 0.0


def _brick_entry(entry) -> tuple[int, str, int, int]:
    if isinstance(entry, dict):
        return int(entry["k"]), str(entry["file"]), int(entry["offset"]), int(entry["length"])
    return int(entry[0]), str(entry[1]), int(entry[2]), int(entry[3])


def frame_dtype(header: dict) -> np.dtype:
    names, formats, offsets = [], [], []
    for item in header["fields"]:
        shape = tuple(int(n) for n in (item.get("shape") or ()))
        names.append(str(item["name"]))
        formats.append((item["dtype"], shape) if shape else item["dtype"])
        offsets.append(int(item["offset"]))
    if "tick" not in names:
        raise ApiError(500, "frames header has no 'tick' field")
    return np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": int(header["record_bytes"])})


def _first_line_at(handle, position: int):
    if position <= 0:
        handle.seek(0)
    else:
        handle.seek(position - 1)
        handle.readline()
    start = handle.tell()
    line = handle.readline()
    if not line or not line.endswith(b"\n"):
        return None
    return start, line


def _trace_search(handle, size: int, event_id: int) -> dict | None:
    """The trace line whose id is ``event_id``: ids rise one per line, so bisect byte offsets."""
    if event_id < 0:
        return None
    lo, hi = 0, size
    while lo < hi:
        mid = (lo + hi) // 2
        found = _first_line_at(handle, mid)
        if found is None or json.loads(found[1])["id"] >= event_id:
            hi = mid
        else:
            lo = mid + 1
    found = _first_line_at(handle, lo)
    if found is None:
        return None
    record = json.loads(found[1])
    return record if record.get("id") == event_id else None


class RunStore:
    """DATA_DIR/runs/<run_id>/ folders."""

    def __init__(self, runs_dir: Path, cache: FileCache) -> None:
        self.dir = runs_dir
        self.cache = cache
        self._lock = threading.Lock()

    def path(self, run_id: str) -> Path:
        return self.dir / check_id(run_id, "run id")

    def get(self, run_id: str) -> RunDir:
        path = self.path(run_id)
        if not path.is_dir() or not ((path / "run.json").is_file() or (path / "status.json").is_file()):
            raise ApiError(404, f"no run {run_id!r}")
        return RunDir(run_id, path, self.cache)

    def ids(self) -> list[str]:
        if not self.dir.is_dir():
            return []
        out = []
        for path in self.dir.iterdir():
            if path.is_dir() and ((path / "run.json").is_file() or (path / "status.json").is_file()):
                out.append(path.name)
        return sorted(out)

    def allocate(self, slug: str) -> tuple[str, Path]:
        """A new run folder ``<nnnnn>-<slug>`` (the number one above every numbered run)."""
        with self._lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            highest = 0
            for path in self.dir.iterdir():
                match = _RUN_ID.match(path.name)
                if match:
                    highest = max(highest, int(match.group(1)))
            number = highest + 1
            while True:
                run_id = f"{number:05d}-{slug}"[:128]
                path = self.dir / run_id
                try:
                    path.mkdir()
                    return run_id, path
                except FileExistsError:
                    number += 1
