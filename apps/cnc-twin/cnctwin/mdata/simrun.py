"""A simulated run directory (docs/architecture.md 16) as channels comparable with machine data.

:class:`SimulatedRun` reads ``run.json`` (clock), ``timeline.json`` (executed blocks: line, block
text, ``[tick0, tick1)``, tool, ``ticks`` by category), ``scene.json["frames"]`` + ``frames.bin``
(per L1 tick: axis values relative to the stock origin, spindle angle wrapped to [0, 2 pi) with
its revolution count), the tick-tier series (``series/index.jsonl`` and
``series/tick/part-<n>.parquet``) and, when present, ``tools.json`` and ``inspection.json``.

:meth:`SimulatedRun.to_log` gives a :class:`MachineLog` in the run's own time base
(``t_s = tick * tick_s`` from the start of the run) with the canonical channel names of the
machine data: ``spindle.speed_rad_s`` (magnitude of the forward difference of the spindle angle
per L1 tick), ``spindle.shaft_power_W`` (``cutting.spindle_power_W``: the shaft power of cutting),
``spindle.load_frac`` (``cutting.spindle_load_fraction``), ``path.feed_m_s`` (``motion.feed_m_s``),
``axis.<A>.following_error_m`` (``motion.following_error_m[i]``, axis names from the frames
header), ``axis.<A>.pos_stock_m`` (frames, relative to the stock origin), ``program.line``,
``program.sequence`` (the N word of the block text, where written), ``program.block`` and
``tool.number`` (timeline, at each block start). :meth:`markers` lists the block starts.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.errors import InputError
from .log import Channel, MachineLog

_N_WORD = re.compile(r"^\s*/?\d?\s*N(\d+)", re.IGNORECASE)

SERIES_CHANNELS = {
    "cutting.spindle_power_W": ("spindle.shaft_power_W", "sample"),
    "cutting.spindle_load_fraction": ("spindle.load_frac", "sample"),
    "motion.feed_m_s": ("path.feed_m_s", "sample"),
}


@dataclass(frozen=True)
class Marker:
    t_s: float
    tick: int
    index: int                  # timeline block index
    line: int
    sequence: int | None        # N word
    block: str
    program: str
    kind: str


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise InputError(f"cannot read {path.name}: {error.strerror}", str(path)) from None
    except ValueError as error:
        raise InputError(f"{path.name} is not JSON: {error}", str(path)) from None


class SimulatedRun:
    def __init__(self, run_dir) -> None:
        self.dir = Path(run_dir)
        if not self.dir.is_dir():
            raise InputError("the run directory does not exist", str(self.dir))
        self.run = _read_json(self.dir / "run.json") if (self.dir / "run.json").exists() else {}
        clock = self.run.get("clock") or {}
        self.tick_s = float(clock.get("tick_s") or self._frames_header().get("tick_s") or 1e-6)
        self._frames = None
        self._timeline = None

    # ------------------------------------------------------------------ files
    def _frames_header(self) -> dict:
        scene_path = self.dir / "scene.json"
        if not scene_path.exists():
            return {}
        scene = _read_json(scene_path)
        return scene.get("frames") or {} if isinstance(scene, dict) else {}

    def timeline(self) -> list[dict]:
        if self._timeline is None:
            path = self.dir / "timeline.json"
            if not path.exists():
                raise InputError("the run has no timeline.json", str(self.dir))
            document = _read_json(path)
            blocks = document.get("blocks") if isinstance(document, dict) else document
            if not isinstance(blocks, list):
                raise InputError("timeline.json has no block list", str(path))
            self._timeline = sorted(blocks, key=lambda b: (int(b["tick0"]), int(b.get("index", 0))))
        return self._timeline

    def frames(self) -> dict:
        """tick [n] int64, axis [n, n_axes] (SI, relative to the stock origin), axis_names, spindle angle
        unwrapped [n, n_paths]."""
        if self._frames is not None:
            return self._frames
        header = self._frames_header()
        if not header or "fields" not in header:
            raise InputError("scene.json has no frames header", str(self.dir / "scene.json"))
        fields = sorted(header["fields"], key=lambda f: int(f["offset"]))
        dtype = np.dtype({"names": [f["name"] for f in fields],
                          "formats": [(np.dtype(f["dtype"]), tuple(f.get("shape") or ())) if f.get("shape")
                                      else np.dtype(f["dtype"]) for f in fields],
                          "offsets": [int(f["offset"]) for f in fields], "itemsize": int(header["record_bytes"])})
        path = self.dir / "frames.bin"
        size = path.stat().st_size if path.exists() else 0
        count = size // dtype.itemsize
        data = np.fromfile(path, dtype=dtype, count=count) if count else np.zeros(0, dtype)
        names = [a["name"] for a in header.get("axes", [])]
        axis = np.asarray(data["axis"], dtype=np.float64).reshape(count, -1) if "axis" in dtype.names else \
            np.zeros((count, 0))
        angle = np.asarray(data["spindle_angle_rad"], dtype=np.float64).reshape(count, -1) \
            if "spindle_angle_rad" in dtype.names else np.zeros((count, 0))
        revs = np.asarray(data["spindle_rev"], dtype=np.float64).reshape(count, -1) \
            if "spindle_rev" in dtype.names else np.zeros_like(angle)
        self._frames = {"tick": np.asarray(data["tick"], dtype=np.int64), "axis": axis,
                        "axis_names": tuple(names) if len(names) == axis.shape[1] else
                        tuple(f"A{i}" for i in range(axis.shape[1])),
                        "spindle_angle_rad": angle + 2.0 * math.pi * revs,
                        "ipo_ticks": int(header.get("ipo_ticks") or 0)}
        return self._frames

    def series(self, tier: str, columns: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """(tick, {column: values}) over every part of a tier (columns a part lacks are NaN there)."""
        import pyarrow.parquet as pq

        index = self.dir / "series" / "index.jsonl"
        files = []
        if index.exists():
            for raw in index.read_text(encoding="utf-8").splitlines():
                if raw.strip():
                    entry = json.loads(raw)
                    if entry.get("tier") == tier:
                        files.append((int(entry.get("part", 0)), self.dir / entry["file"]))
        else:
            for path in (self.dir / "series" / tier).glob("part-*.parquet"):
                files.append((int(path.stem.split("-")[1]), path))
        files.sort()
        ticks, values = [], {c: [] for c in columns}
        for _, path in files:
            table = pq.read_table(path)
            names = set(table.column_names)
            tick = table.column("tick").to_numpy().astype(np.int64)
            ticks.append(tick)
            for column in columns:
                values[column].append(table.column(column).to_numpy().astype(np.float64) if column in names
                                      else np.full(len(tick), np.nan))
        if not ticks:
            return np.zeros(0, np.int64), {c: np.zeros(0) for c in columns}
        tick = np.concatenate(ticks)
        order = np.argsort(tick, kind="stable")
        return tick[order], {c: np.concatenate(v)[order] for c, v in values.items()}

    def tick_columns(self) -> list[str]:
        import pyarrow.parquet as pq

        for path in sorted((self.dir / "series" / "tick").glob("part-*.parquet")):
            return list(pq.read_schema(path).names)
        return []

    # ------------------------------------------------------------------ derived
    def cycle_time_s(self) -> float:
        blocks = self.timeline()
        if not blocks:
            return 0.0
        return (max(int(b["tick1"]) for b in blocks) - min(int(b["tick0"]) for b in blocks)) * self.tick_s

    def markers(self) -> list[Marker]:
        out = []
        for b in self.timeline():
            text = str(b.get("block") or "")
            match = _N_WORD.match(text)
            out.append(Marker(int(b["tick0"]) * self.tick_s, int(b["tick0"]), int(b.get("index", 0)),
                              int(b.get("line") or 0), int(match.group(1)) if match else None, text,
                              str(b.get("program") or ""), str(b.get("kind") or "")))
        return out

    def spindle_speed(self) -> tuple[np.ndarray, np.ndarray]:
        """(t_s, |omega| rad/s) per L1 tick: forward difference of the unwrapped angle of path 0."""
        frames = self.frames()
        tick = frames["tick"]
        if len(tick) < 2 or frames["spindle_angle_rad"].shape[1] == 0:
            return tick * self.tick_s, np.zeros(len(tick))
        angle = frames["spindle_angle_rad"][:, 0]
        dt = np.diff(tick) * self.tick_s
        omega = np.abs(np.diff(angle)) / dt
        omega = np.concatenate([omega, omega[-1:]])
        return tick * self.tick_s, omega

    def to_log(self) -> MachineLog:
        log = MachineLog(epoch_ns=None, source={"kind": "simulation", "run_dir": str(self.dir),
                                                "run": self.run.get("project", {}).get("name")})
        t, omega = self.spindle_speed()
        if len(t):
            log.channels["spindle.speed_rad_s"] = Channel("spindle.speed_rad_s", t, omega, "sample", "frames.bin")
            frames = self.frames()
            for i, name in enumerate(frames["axis_names"]):
                channel = f"axis.{name}.pos_stock_m"
                log.channels[channel] = Channel(channel, t, frames["axis"][:, i], "sample", "frames.bin")
        columns = self.tick_columns()
        wanted = [c for c in SERIES_CHANNELS if c in columns]
        follow = sorted((c for c in columns if c.startswith("motion.following_error_m[")),
                        key=lambda c: int(c[c.index("[") + 1:c.index("]")]))
        ticks, values = self.series("tick", wanted + follow)
        if len(ticks):
            ts = ticks * self.tick_s
            for column in wanted:
                name, kind = SERIES_CHANNELS[column]
                log.channels[name] = Channel(name, ts, values[column], kind, f"series/tick:{column}")
            names = self.frames()["axis_names"] if (self.dir / "scene.json").exists() else ()
            for column in follow:
                i = int(column[column.index("[") + 1:column.index("]")])
                axis = names[i] if i < len(names) else f"A{i}"
                channel = f"axis.{axis}.following_error_m"
                log.channels[channel] = Channel(channel, ts, values[column], "sample", f"series/tick:{column}")
        markers = self.markers()
        if markers:
            times = np.array([m.t_s for m in markers])
            log.channels["program.line"] = Channel("program.line", times, np.array([m.line for m in markers], float),
                                                   "event", "timeline.json")
            sequence = np.array([np.nan if m.sequence is None else m.sequence for m in markers], float)
            if np.isfinite(sequence).any():
                log.channels["program.sequence"] = Channel("program.sequence", times, sequence, "event", "timeline.json")
            blocks = np.empty(len(markers), dtype=object)
            blocks[:] = [m.block for m in markers]
            log.channels["program.block"] = Channel("program.block", times, blocks, "event", "timeline.json")
            tools = np.array([np.nan if b.get("tool") is None else float(b["tool"]) for b in self.timeline()])
            if np.isfinite(tools).any():
                log.channels["tool.number"] = Channel("tool.number", times, tools, "event", "timeline.json")
        return log


__all__ = ["SimulatedRun", "Marker", "SERIES_CHANNELS"]
