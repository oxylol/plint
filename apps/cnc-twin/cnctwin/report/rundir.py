"""Reading one run directory (docs/architecture.md 16) for reports and comparisons.

:class:`RunData` reads only the run directory, never writes, and is tolerant: an absent or
unreadable file is not an error but an entry of :attr:`RunData.missing` (what, why), which the
report prints as its "not in this run directory" list, so nothing is left out silently.

Series are read part by part (``series/<tier>/part-<n>.parquet``) and reduced on the fly to a fixed
number of time buckets per column (min, max, sum, count, tick range), so the memory needed does not
grow with the run's length; fast-tier windows are scanned part by part and only the window with
the highest peak of each channel is kept raw.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

TIERS = ("fast", "tick", "macro")
BUCKETS = 600
_PART = re.compile(r"^part-(\d+)\.parquet$")
_COLUMN = re.compile(r"^(?P<name>.+?)(?P<index>(\[\d+\])*)$")
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml",
               ".gif": "image/gif", ".webp": "image/webp"}


def split_column(column: str) -> tuple[str, str, str]:
    """``cutting.force_N[0].max`` -> (``cutting.force_N``, ``[0]``, ``max``); stat '' if not an envelope
    column (architecture 10: ``<name>[i]`` vector columns, ``.min/.mean/.max`` tick-tier envelopes)."""
    base, stat = column, ""
    for candidate in ("min", "mean", "max"):
        suffix = "." + candidate
        if column.endswith(suffix) and "." in column[: -len(suffix)]:
            base, stat = column[: -len(suffix)], candidate
            break
    match = _COLUMN.match(base)
    return match.group("name"), match.group("index"), stat


# ----------------------------------------------------------------------------- trace

class TraceFile:
    """trace.jsonl with an id -> byte offset index (id = line number); a line being written is ignored."""

    def __init__(self, path: Path):
        self.path = path
        self._offsets: list[int] | None = None

    def _index(self) -> list[int]:
        if self._offsets is None:
            offsets = []
            if self.path.is_file():
                position = 0
                with open(self.path, "rb") as handle:
                    for raw in handle:
                        if raw.endswith(b"\n"):
                            offsets.append(position)
                        position += len(raw)
            self._offsets = offsets
        return self._offsets

    def __len__(self) -> int:
        return len(self._index())

    def get(self, event_id: int) -> dict | None:
        offsets = self._index()
        if not 0 <= int(event_id) < len(offsets):
            return None
        with open(self.path, "rb") as handle:
            handle.seek(offsets[int(event_id)])
            try:
                return json.loads(handle.readline())
            except json.JSONDecodeError:
                return None

    def events(self, ids) -> list[dict]:
        out = []
        for event_id in ids:
            event = self.get(event_id)
            if event is not None:
                out.append(event)
        return out

    def domains(self) -> set[str]:
        found = set()
        if self.path.is_file():
            with open(self.path, "rb") as handle:
                for raw in handle:
                    if raw.endswith(b"\n"):
                        try:
                            found.add(str(json.loads(raw).get("domain", "")))
                        except json.JSONDecodeError:
                            continue
        return found


# ----------------------------------------------------------------------------- series reduction

@dataclass
class ColumnStats:
    name: str
    rows: int = 0
    finite: int = 0
    vmin: float = np.inf
    vmax: float = -np.inf
    vsum: float = 0.0
    b_min: np.ndarray = field(default_factory=lambda: np.full(BUCKETS, np.inf))
    b_max: np.ndarray = field(default_factory=lambda: np.full(BUCKETS, -np.inf))
    b_sum: np.ndarray = field(default_factory=lambda: np.zeros(BUCKETS))
    b_cnt: np.ndarray = field(default_factory=lambda: np.zeros(BUCKETS, np.int64))

    @property
    def mean(self) -> float | None:
        return self.vsum / self.finite if self.finite else None

    @property
    def min(self) -> float | None:
        return float(self.vmin) if self.finite else None

    @property
    def max(self) -> float | None:
        return float(self.vmax) if self.finite else None

    @property
    def max_abs(self) -> float | None:
        return max(abs(self.vmin), abs(self.vmax)) if self.finite else None


@dataclass
class TierSummary:
    tier: str
    tick0: int
    tick1: int
    tick_s: float
    rows: int = 0
    parts: int = 0
    columns: dict[str, ColumnStats] = field(default_factory=dict)
    b_tick_lo: np.ndarray = field(default_factory=lambda: np.full(BUCKETS, np.iinfo(np.int64).max))
    b_tick_hi: np.ndarray = field(default_factory=lambda: np.full(BUCKETS, np.iinfo(np.int64).min))

    def curve(self, column: str):
        """(t_s, mean, lo, hi) over the non-empty buckets of ``column`` (None if absent or all NaN)."""
        stats = self.columns.get(column)
        if stats is None or not stats.finite:
            return None
        used = stats.b_cnt > 0
        t_s = 0.5 * (self.b_tick_lo[used] + self.b_tick_hi[used]).astype(float) * self.tick_s
        mean = stats.b_sum[used] / stats.b_cnt[used]
        return t_s, mean, stats.b_min[used], stats.b_max[used]

    def at_end(self, column: str):
        """The last finite bucket mean of a column (the value at the end of the run)."""
        stats = self.columns.get(column)
        if stats is None or not stats.finite:
            return None
        used = np.flatnonzero(stats.b_cnt > 0)
        last = used[-1]
        return float(stats.b_sum[last] / stats.b_cnt[last])


def _reduce_part(summary: TierSummary, ticks: np.ndarray, table, names: list[str]) -> None:
    span = max(summary.tick1 - summary.tick0 + 1, 1)
    order = np.argsort(ticks, kind="stable")
    ticks = ticks[order]
    bucket = np.clip(((ticks - summary.tick0).astype(np.float64) * BUCKETS / span).astype(np.int64), 0, BUCKETS - 1)
    starts = np.flatnonzero(np.r_[True, bucket[1:] != bucket[:-1]])
    ids = bucket[starts]
    np.minimum.at(summary.b_tick_lo, ids, np.minimum.reduceat(ticks, starts))
    np.maximum.at(summary.b_tick_hi, ids, np.maximum.reduceat(ticks, starts))
    summary.rows += len(ticks)
    for name in names:
        values = np.asarray(table.column(name).to_numpy(zero_copy_only=False), dtype=np.float64)[order]
        stats = summary.columns.setdefault(name, ColumnStats(name))
        finite = np.isfinite(values)
        stats.rows += len(values)
        count = int(finite.sum())
        if not count:
            continue
        stats.finite += count
        stats.vmin = min(stats.vmin, float(values[finite].min()))
        stats.vmax = max(stats.vmax, float(values[finite].max()))
        stats.vsum += float(values[finite].sum())
        np.minimum.at(stats.b_min, ids, np.fmin.reduceat(np.where(finite, values, np.inf), starts))
        np.maximum.at(stats.b_max, ids, np.fmax.reduceat(np.where(finite, values, -np.inf), starts))
        np.add.at(stats.b_sum, ids, np.add.reduceat(np.where(finite, values, 0.0), starts))
        np.add.at(stats.b_cnt, ids, np.add.reduceat(finite.astype(np.int64), starts))


@dataclass
class Window:
    """One recorded fast window (raw samples) of the tick range [tick0, tick1]."""

    tick0: int
    tick1: int
    rows: int
    t_s: np.ndarray | None = None
    values: dict[str, np.ndarray] = field(default_factory=dict)
    peak: float = 0.0


# ----------------------------------------------------------------------------- the run directory

class RunData:
    """Everything a report needs from one run directory, read lazily."""

    def __init__(self, path):
        self.path = Path(path).absolute()
        self.missing: list[dict] = []
        self._seen: set[str] = set()

    # ------------------------------------------------------------------ files
    @property
    def id(self) -> str:
        return self.path.name

    def has(self, rel: str) -> bool:
        return (self.path / rel).is_file()

    def note(self, item: str, reason: str) -> None:
        if item not in self._seen:
            self._seen.add(item)
            self.missing.append({"item": item, "reason": reason})

    def json(self, rel: str, what: str = ""):
        path = self.path / rel
        if not path.is_file():
            self.note(rel, f"{what or rel} is not in the run directory")
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            self.note(rel, f"{rel} cannot be read ({type(error).__name__}: {error})")
            return None

    def jsonl(self, rel: str, what: str = "") -> list[dict]:
        path = self.path / rel
        if not path.is_file():
            self.note(rel, f"{what or rel} is not in the run directory")
            return []
        records = []
        for raw in path.read_bytes().splitlines(keepends=True):
            if not raw.endswith(b"\n") or not raw.strip():
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError:
                self.note(f"{rel}:line", f"{rel} has a line that is not JSON (skipped)")
        return records

    def npz(self, rel: str) -> dict | None:
        path = self.path / rel
        if not path.is_file():
            self.note(rel, f"{rel} is not in the run directory")
            return None
        try:
            with np.load(path, allow_pickle=False) as archive:
                return {name: archive[name] for name in sorted(archive.files)}
        except (OSError, ValueError) as error:
            self.note(rel, f"{rel} cannot be read ({type(error).__name__}: {error})")
            return None

    # ------------------------------------------------------------------ documents
    @cached_property
    def run(self) -> dict:
        document = self.json("run.json", "run.json (verdict, summary, resolved project)")
        return document if isinstance(document, dict) else {}

    @cached_property
    def status(self) -> dict:
        path = self.path / "status.json"
        if not path.is_file():
            return {}
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return document if isinstance(document, dict) else {}

    @property
    def finished(self) -> bool:
        return bool(self.run)

    @property
    def project(self) -> dict:
        project = self.run.get("project")
        return project if isinstance(project, dict) else {}

    @property
    def name(self) -> str:
        return str(self.project.get("name") or self.id)

    @property
    def summary(self) -> dict:
        summary = self.run.get("summary")
        return summary if isinstance(summary, dict) else {}

    @property
    def verdict(self) -> dict:
        verdict = self.run.get("verdict")
        if isinstance(verdict, dict) and "safe" in verdict:
            return verdict
        reasons = [f"{f['category']} ({f['severity']}) at {f.get('op', '')} line {f.get('line', 0)}: {f['message']}"
                   for f in self.findings if f.get("severity") in ("error", "fatal")]
        self.note("run.json:verdict", "run.json has no verdict: it was derived from findings.json "
                                      "(safe when no finding is an error or fatal)")
        return {"safe": not reasons, "reasons": reasons, "derived": True}

    @cached_property
    def tick_s(self) -> float:
        for source in (self.run.get("clock") or {}, self.timeline_doc or {}, self.production or {},
                       (self.scene or {}).get("frames") or {}):
            value = source.get("tick_s") if isinstance(source, dict) else None
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
        from ..project.schema import TICK_S
        self.note("tick_s", "no file states the base clock: the architecture's fixed 1 us tick was used")
        return float(TICK_S)

    @cached_property
    def findings(self) -> list[dict]:
        document = self.json("findings.json", "findings.json")
        items = document.get("findings", []) if isinstance(document, dict) else (document or [])
        items = [f for f in items if isinstance(f, dict)]
        return sorted(items, key=lambda f: (int(f.get("id", 0)), float(f.get("time_s", 0.0))))

    @cached_property
    def trace(self) -> TraceFile:
        if not self.has("trace.jsonl"):
            self.note("trace.jsonl", "trace.jsonl is not in the run directory: cause chains cannot be expanded")
        return TraceFile(self.path / "trace.jsonl")

    @cached_property
    def timeline_doc(self) -> dict | None:
        document = self.json("timeline.json", "timeline.json (executed blocks)")
        if isinstance(document, list):
            return {"blocks": document}
        return document if isinstance(document, dict) else None

    @property
    def timeline(self) -> list[dict]:
        return [b for b in (self.timeline_doc or {}).get("blocks", []) if isinstance(b, dict)]

    @cached_property
    def timeline_blocks(self):
        """cnctwin.production.timeline blocks (tools assigned as the production metrics assign them)."""
        if not self.has("timeline.json"):
            return []
        from ..core.errors import InputError
        from ..production.timeline import read_timeline
        try:
            return list(read_timeline(self.path).blocks)
        except (InputError, KeyError, TypeError, ValueError) as error:
            self.note("timeline.json:blocks", f"timeline.json cannot be read as executed blocks ({error})")
            return []

    @cached_property
    def channels(self) -> list[dict]:
        document = self.json("channels.json", "channels.json (channel registry)")
        items = document.get("channels", []) if isinstance(document, dict) else (document or [])
        return sorted((c for c in items if isinstance(c, dict) and "name" in c), key=lambda c: c["name"])

    @cached_property
    def scene(self) -> dict:
        document = self.json("scene.json", "scene.json (static scene, frames header)")
        return document if isinstance(document, dict) else {}

    @cached_property
    def inspection(self) -> dict | None:
        return self._result("inspection.json", "inspection.json (measured characteristics)")

    @cached_property
    def surface(self) -> dict | None:
        return self._result("surface.json", "surface.json (finish and burrs)")

    @cached_property
    def tools(self) -> dict | None:
        return self._result("tools.json", "tools.json (wear and life)")

    @cached_property
    def production(self) -> dict | None:
        return self._result("production.json", "production.json (cycle time, energy, consumables, cost)")

    def _result(self, rel: str, what: str) -> dict | None:
        document = self.json(rel, what)
        return document if isinstance(document, dict) else None

    @cached_property
    def sld(self) -> dict[str, dict]:
        folder = self.path / "sld"
        if not folder.is_dir():
            self.note("sld/", "sld/ (stability lobe diagrams) is not in the run directory")
            return {}
        out = {}
        for path in sorted(folder.glob("*.json")):
            document = self.json(f"sld/{path.name}")
            if isinstance(document, dict):
                out[path.stem] = document
        return out

    @cached_property
    def keyframes(self) -> list[dict]:
        entries = self.jsonl("keyframes/index.jsonl", "keyframes/index.jsonl")
        return sorted((e for e in entries if "k" in e), key=lambda e: int(e["k"]))

    def keyframe_fields(self, entry: dict) -> dict | None:
        return self.npz(str(entry.get("fields") or f"keyframes/kf_{int(entry['k'])}.npz"))

    def keyframe_chips(self, entry: dict) -> dict | None:
        return self.npz(str(entry.get("chips") or f"chips/kf_{int(entry['k'])}.npz"))

    @cached_property
    def images(self) -> list[tuple[str, bytes, str]]:
        folder = self.path / "images"
        if not folder.is_dir():
            self.note("images/", "images/ (rendered pictures) is not in the run directory")
            return []
        out = []
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            mime = IMAGE_TYPES.get(path.suffix.lower())
            if mime is None:
                self.note(f"images/{path.name}", f"images/{path.relative_to(folder).as_posix()} is not an image "
                                                 "type the report embeds (png, jpg, svg, gif, webp)")
                continue
            out.append((path.relative_to(self.path).as_posix(), path.read_bytes(), mime))
        return out

    @cached_property
    def attributes(self) -> dict[str, np.ndarray]:
        """Final per-vertex attributes of every brick (keyframes/attr/brick_<b>.npz), concatenated."""
        folder = self.path / "keyframes" / "attr"
        if not folder.is_dir():
            self.note("keyframes/attr/", "keyframes/attr/ (final surface attributes) is not in the run directory")
            return {}
        paths = sorted(folder.glob("*.npz"), key=lambda p: (int(re.sub(r"\D", "", p.stem) or 0), p.name))
        merged: dict[str, list] = {}
        for path in paths:
            arrays = self.npz(f"keyframes/attr/{path.name}")
            for name, array in (arrays or {}).items():
                merged.setdefault(name, []).append(np.asarray(array, dtype=np.float64).reshape(-1))
        return {name: np.concatenate(parts) for name, parts in sorted(merged.items())}

    # ------------------------------------------------------------------ program
    @property
    def operations(self) -> list[dict]:
        return [op for op in self.project.get("operations", []) if isinstance(op, dict)]

    def programs(self) -> list[tuple[str, str, str | None]]:
        """(op, program file, text or None) for every program of every operation."""
        out = []
        for op in self.operations:
            name = str(op.get("name", ""))
            texts = op.get("program_texts")
            if isinstance(texts, dict) and texts:
                files = op.get("programs") if isinstance(op.get("programs"), dict) else {}
                for path_key in sorted(texts, key=str):
                    label = str(files.get(path_key, f"path {path_key}")) if files else f"path {path_key}"
                    out.append((name, Path(label).name, texts[path_key]))
                continue
            program = str(op.get("program") or "")
            text = op.get("program_text")
            if text is None:
                self.note(f"program:{name}", f"run.json has no program_text for {name} ({program})")
            out.append((name, Path(program).name if program else "", text))
        return out

    # ------------------------------------------------------------------ cycle
    @property
    def cycle_time_s(self) -> float | None:
        for value in (self.summary.get("cycle_time_s"), (self.production or {}).get("cycle_time_s")):
            if isinstance(value, (int, float)):
                return float(value)
        if self.timeline:
            return (max(int(b["tick1"]) for b in self.timeline) - min(int(b["tick0"]) for b in self.timeline)) * self.tick_s
        return None

    # ------------------------------------------------------------------ series
    def series_parts(self, tier: str) -> list[dict]:
        folder = self.path / "series" / tier
        indexed = [e for e in self.jsonl("series/index.jsonl", "series/index.jsonl")
                   if e.get("tier") == tier and e.get("file")] if self.has("series/index.jsonl") else []
        if indexed:
            parts = [e for e in indexed if (self.path / e["file"]).is_file()]
            for entry in indexed:
                if entry not in parts:
                    self.note(entry["file"], f"{entry['file']} is listed in series/index.jsonl but absent")
            return sorted(parts, key=lambda e: int(e.get("part", 0)))
        if not folder.is_dir():
            return []
        found = []
        for path in folder.iterdir():
            match = _PART.match(path.name)
            if match:
                found.append({"tier": tier, "part": int(match.group(1)), "file": f"series/{tier}/{path.name}"})
        return sorted(found, key=lambda e: e["part"])

    def _parquet(self, entry: dict):
        import pyarrow.parquet as pq
        try:
            return pq.read_table(self.path / entry["file"])
        except (OSError, ValueError) as error:
            self.note(entry["file"], f"{entry['file']} cannot be read ({type(error).__name__}: {error})")
            return None

    def _tier_span(self, tier: str, parts: list[dict]) -> tuple[int, int]:
        lo, hi = None, None
        for entry in parts:
            if "tick0" in entry and "tick1" in entry:
                a, b = int(entry["tick0"]), int(entry["tick1"])
            else:
                import pyarrow.parquet as pq
                ticks = pq.read_table(self.path / entry["file"], columns=["tick"]).column("tick").to_numpy()
                if not len(ticks):
                    continue
                a, b = int(ticks.min()), int(ticks.max())
            lo = a if lo is None else min(lo, a)
            hi = b if hi is None else max(hi, b)
        return (lo or 0), (hi if hi is not None else 0)

    def tier_summary(self, tier: str) -> TierSummary | None:
        key = f"_tier_{tier}"
        if key in self.__dict__:
            return self.__dict__[key]
        parts = self.series_parts(tier)
        if not parts:
            self.note(f"series/{tier}", f"no {tier}-tier series in the run directory")
            self.__dict__[key] = None
            return None
        tick0, tick1 = self._tier_span(tier, parts)
        summary = TierSummary(tier, tick0, tick1, self.tick_s)
        for entry in parts:
            table = self._parquet(entry)
            if table is None or "tick" not in table.column_names:
                continue
            ticks = np.asarray(table.column("tick").to_numpy(zero_copy_only=False), dtype=np.int64)
            if not len(ticks):
                continue
            names = [c for c in table.column_names if c not in ("tick", "time_s")]
            _reduce_part(summary, ticks, table, names)
            summary.parts += 1
        self.__dict__[key] = summary
        return summary

    def fast_windows(self, channel_columns: dict[str, list[str]]) -> tuple[list[Window], dict[str, Window]]:
        """All fast windows (tick ranges, row counts) and, per channel, the window with the highest
        |value| (raw samples of that channel's columns)."""
        windows: list[Window] = []
        best: dict[str, Window] = {}
        for entry in self.series_parts("fast"):
            table = self._parquet(entry)
            if table is None or "tick" not in table.column_names or not table.num_rows:
                continue
            ticks = np.asarray(table.column("tick").to_numpy(zero_copy_only=False), dtype=np.int64)
            order = np.argsort(ticks, kind="stable")
            ticks = ticks[order]
            steps = np.diff(ticks)
            typical = float(np.median(steps)) if len(steps) else 0.0
            cuts = np.flatnonzero(steps > 2.0 * typical) + 1 if typical > 0 else np.array([], np.int64)
            bounds = np.r_[0, cuts, len(ticks)]
            columns = {name: np.asarray(table.column(name).to_numpy(zero_copy_only=False), np.float64)[order]
                       for cols in channel_columns.values() for name in cols if name in table.column_names}
            for a, b in zip(bounds[:-1], bounds[1:]):
                window = Window(int(ticks[a]), int(ticks[b - 1]), int(b - a))
                windows.append(window)
                for channel, cols in channel_columns.items():
                    present = [c for c in cols if c in columns]
                    if not present:
                        continue
                    block = np.column_stack([columns[c][a:b] for c in present])
                    finite = np.abs(block[np.isfinite(block)])
                    peak = float(finite.max()) if finite.size else -1.0
                    if channel not in best or peak > best[channel].peak:
                        best[channel] = Window(window.tick0, window.tick1, window.rows,
                                               ticks[a:b].astype(float) * self.tick_s,
                                               {c: columns[c][a:b].copy() for c in present}, peak)
        return windows, best

    def per_tool(self, tier: str, columns: list[str]) -> dict:
        """Per tool number (None: no tool), for each column: (min, max, mean of |value|, rows) over the
        rows of that tier whose tick falls in a block of that tool (path 0's timeline)."""
        blocks = [b for b in self.timeline_blocks if int(getattr(b, "path", 0)) == 0]
        if not blocks:
            return {}
        starts = np.array([b.tick0 for b in blocks], np.int64)
        order = np.argsort(starts, kind="stable")
        starts = starts[order]
        tools = np.array([-1 if blocks[i].tool is None else int(blocks[i].tool) for i in order], np.int64)
        out: dict = {}
        for entry in self.series_parts(tier):
            table = self._parquet(entry)
            if table is None or "tick" not in table.column_names:
                continue
            ticks = np.asarray(table.column("tick").to_numpy(zero_copy_only=False), dtype=np.int64)
            index = np.clip(np.searchsorted(starts, ticks, side="right") - 1, 0, len(starts) - 1)
            row_tools = tools[index]
            for column in columns:
                if column not in table.column_names:
                    continue
                values = np.asarray(table.column(column).to_numpy(zero_copy_only=False), np.float64)
                for tool in np.unique(row_tools):
                    picked = values[(row_tools == tool) & np.isfinite(values)]
                    if not picked.size:
                        continue
                    key = None if tool < 0 else int(tool)
                    record = out.setdefault(key, {}).setdefault(column, [np.inf, -np.inf, 0.0, 0])
                    record[0] = min(record[0], float(picked.min()))
                    record[1] = max(record[1], float(picked.max()))
                    record[2] += float(np.abs(picked).sum())
                    record[3] += int(picked.size)
        return {tool: {c: (v[0], v[1], v[2] / v[3] if v[3] else None, v[3]) for c, v in cols.items()}
                for tool, cols in out.items()}

    # ------------------------------------------------------------------ frames
    def frames(self, max_records: int = 20000) -> dict | None:
        """Up to ``max_records`` evenly spaced frames.bin records: tick, tip_S_m (path 0), axis values."""
        header = self.scene.get("frames") if isinstance(self.scene, dict) else None
        path = self.path / "frames.bin"
        if not isinstance(header, dict) or "fields" not in header:
            self.note("scene.json:frames", "scene.json has no frames header: the tool path cannot be drawn")
            return None
        if not path.is_file():
            self.note("frames.bin", "frames.bin is not in the run directory: the tool path cannot be drawn")
            return None
        fields = sorted(header["fields"], key=lambda f: int(f["offset"]))
        dtype = np.dtype({"names": [f["name"] for f in fields],
                          "formats": [(f["dtype"], tuple(f.get("shape") or ())) if f.get("shape") else f["dtype"]
                                      for f in fields],
                          "offsets": [int(f["offset"]) for f in fields], "itemsize": int(header["record_bytes"])})
        count = path.stat().st_size // dtype.itemsize
        if count == 0:
            return None
        records = np.memmap(path, dtype=dtype, mode="r", shape=(count,))
        picks = np.unique(np.linspace(0, count - 1, min(count, max_records)).astype(np.int64))
        chosen = np.asarray(records[picks])
        out = {"tick": chosen["tick"].astype(np.int64), "count": int(count)}
        if "tip_S_m" in dtype.names:
            out["tip_S_m"] = np.asarray(chosen["tip_S_m"], np.float64).reshape(len(chosen), -1, 3)[:, 0, :]
        if "axis" in dtype.names:
            out["axis"] = np.asarray(chosen["axis"], np.float64).reshape(len(chosen), -1)
        del records
        return out


__all__ = ["RunData", "TraceFile", "TierSummary", "ColumnStats", "Window", "split_column", "TIERS", "BUCKETS"]
