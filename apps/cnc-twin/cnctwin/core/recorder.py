"""Time-series recording (docs/architecture.md 10; files in docs/formats.md "Run directory").

Channels (cnctwin.core.channels) are recorded in three tiers:

* **fast** (L0): the compiled loop fills a ring buffer (:class:`cnctwin.core.abi.RingBuffer`, 0.5 s
  of samples) with one row per L0 step in contact: :func:`ring_begin` stamps the tick and NaN-fills
  the row, the domains' kernels write their columns (:meth:`Recorder.fast_columns`). At L2
  (:meth:`Recorder.end_step`) windows are cut from it: the first ``fast_window_revs`` revolutions of
  every engagement (a new engagement starts where the rows resume after more than one L1 tick
  without contact, or where the loop calls :meth:`Recorder.engagement`), capped by a minimum spacing
  and a maximum number per second; a window centred on every finding raised; a periodic window every
  ``fast_period_s``. ``fast: all`` stores every row as float32, ``fast: none`` none.
* **tick** (L1): the step's per-tick values (:meth:`Recorder.record_tick`) and, for every fast column,
  ``<column>.min``, ``.mean`` and ``.max`` over the L0 rows of each tick (:func:`tick_envelope`).
* **macro** (L2): one row per macro step (:meth:`Recorder.record_macro`).

Rows go to one Parquet file per tier per checkpoint interval, ``series/<tier>/part-<n>.parquet`` (zstd;
written row group by row group into a temporary file, closed and renamed at each checkpoint), listed in
the append-only ``series/index.jsonl`` (first line the header ``{"format": "cnctwin-series-index", "version": 1}``;
an index written before it has none). :func:`iter_series` reads them back in time order (the API's
backend), :func:`read_series` with M4 (min/max-preserving) downsampling, :func:`export` writes CSV or
Parquet (``twin export``). The recorder is a checkpoint participant (its windows, counters and ring
are part of the cut).

Kernels: :func:`ring_begin`, :func:`ring_put`, :func:`tick_envelope`, :func:`ring_envelope` (explicit
signatures; plain Python references ``*_reference``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from numba import njit, typeof, types

from .. import FORMAT_VERSION
from ..db.models import params as model_params
from .abi import RingBuffer, ring_buffer
from .channels import Channel, ChannelRegistry, Tier
from .errors import InputError, TwinError
from .rundir import (SERIES_INDEX_FORMAT, TICK_S, fsync_dir, fsync_file, is_jsonl_header, jsonable, jsonl_header,
                     ticks_to_seconds, write_json_atomic)

TIERS = ("fast", "tick", "macro")
FAST_MODES = ("windows", "all", "none")
CHANNELS_FORMAT = "cnctwin-channels"
SERIES_FORMAT = "cnctwin-series"
ENVELOPE = ("min", "mean", "max")
_PART = re.compile(r"^part-(\d+)\.parquet$")
_SUFFIX = r"(\[\d+\])*(\.(min|mean|max))?$"

RING_T = typeof(ring_buffer(2, 1))
OPTS = dict(cache=True, nogil=True, error_model="numpy")
i8 = types.int64
f8 = types.float64


class RecorderError(TwinError):
    """The recorder was used against its contract (rows lost from the ring, ticks out of order ...)."""


# =============================================================================== kernels

@njit(i8(RING_T, i8), **OPTS)
def ring_begin(ring, tick):
    """Start the ring row of one L0 sample at base tick ``tick``: returns its slot (row index), whose
    columns are NaN until the domains write them."""
    n = ring.tick.shape[0]
    head = ring.head[0]
    slot = head % n
    ring.tick[slot] = tick
    for c in range(ring.data.shape[1]):
        ring.data[slot, c] = np.nan
    ring.head[0] = head + 1
    return slot


@njit(i8(RING_T, i8, types.float64[::1]), **OPTS)
def ring_put(ring, tick, values):
    """Push one whole row (``values`` has one entry per fast column); returns its slot."""
    n = ring.tick.shape[0]
    head = ring.head[0]
    slot = head % n
    ring.tick[slot] = tick
    for c in range(ring.data.shape[1]):
        ring.data[slot, c] = values[c]
    ring.head[0] = head + 1
    return slot


@njit(types.void(types.int64[::1], types.float64[:, ::1], i8, i8, types.float64[:, ::1], types.float64[:, ::1],
                 types.float64[:, ::1]), **OPTS)
def tick_envelope(row_tick, rows, tick0, width, out_min, out_mean, out_max):
    """Per bucket ``[tick0 + b width, tick0 + (b + 1) width)`` and column: minimum, mean and maximum of the
    finite values of the rows in it (rows in tick order; NaN where a bucket has none). Sums run in row
    order, so the mean is the same on every run."""
    nb = out_min.shape[0]
    nc = rows.shape[1]
    count = np.zeros((nb, nc), np.int64)
    total = np.zeros((nb, nc))
    for b in range(nb):
        for c in range(nc):
            out_min[b, c] = np.inf
            out_max[b, c] = -np.inf
    for r in range(row_tick.shape[0]):
        offset = row_tick[r] - tick0
        if offset < 0:
            continue
        b = offset // width
        if b >= nb:
            continue
        for c in range(nc):
            v = rows[r, c]
            if v == v and v != np.inf and v != -np.inf:
                count[b, c] += 1
                total[b, c] += v
                if v < out_min[b, c]:
                    out_min[b, c] = v
                if v > out_max[b, c]:
                    out_max[b, c] = v
    for b in range(nb):
        for c in range(nc):
            if count[b, c] > 0:
                out_mean[b, c] = total[b, c] / count[b, c]
            else:
                out_min[b, c] = np.nan
                out_mean[b, c] = np.nan
                out_max[b, c] = np.nan


@njit(types.void(RING_T, i8, i8, i8, i8, types.float64[:, ::1]), **OPTS)
def ring_envelope(ring, head0, head1, tick0, width, out):
    """Envelope of the ring rows ``head0 <= h < head1`` (in place, no copy) per bucket ``[tick0 + b width,
    tick0 + (b + 1) width)``: ``out[b, 3 c + 0, 1, 2]`` = minimum, mean, maximum of the finite values of
    fast column c (NaN where a bucket has none); sums in row order."""
    n = ring.tick.shape[0]
    nb = out.shape[0]
    nc = ring.data.shape[1]
    count = np.zeros((nb, nc), np.int64)
    for b in range(nb):
        for c in range(nc):
            out[b, 3 * c] = np.inf
            out[b, 3 * c + 1] = 0.0
            out[b, 3 * c + 2] = -np.inf
    for h in range(head0, head1):
        slot = h % n
        offset = ring.tick[slot] - tick0
        if offset < 0:
            continue
        b = offset // width
        if b >= nb:
            continue
        for c in range(nc):
            v = ring.data[slot, c]
            if v == v and v != np.inf and v != -np.inf:
                count[b, c] += 1
                out[b, 3 * c + 1] += v
                if v < out[b, 3 * c]:
                    out[b, 3 * c] = v
                if v > out[b, 3 * c + 2]:
                    out[b, 3 * c + 2] = v
    for b in range(nb):
        for c in range(nc):
            if count[b, c] > 0:
                out[b, 3 * c + 1] = out[b, 3 * c + 1] / count[b, c]
            else:
                out[b, 3 * c] = np.nan
                out[b, 3 * c + 1] = np.nan
                out[b, 3 * c + 2] = np.nan


def ring_envelope_reference(ring: RingBuffer, head0: int, head1: int, tick0: int, width: int, out) -> None:
    n = len(ring.tick)
    slots = np.arange(head0, head1, dtype=np.int64) % n
    nb = out.shape[0]
    nc = ring.data.shape[1]
    lo, mean, hi = (np.empty((nb, nc)) for _ in range(3))
    tick_envelope_reference(ring.tick[slots], ring.data[slots], tick0, width, lo, mean, hi)
    out[:, 0::3], out[:, 1::3], out[:, 2::3] = lo, mean, hi


def ring_begin_reference(ring: RingBuffer, tick: int) -> int:
    n = len(ring.tick)
    slot = int(ring.head[0]) % n
    ring.tick[slot] = tick
    ring.data[slot, :] = np.nan
    ring.head[0] += 1
    return slot


def ring_put_reference(ring: RingBuffer, tick: int, values) -> int:
    slot = ring_begin_reference(ring, tick)
    ring.data[slot, :] = values
    return slot


def tick_envelope_reference(row_tick, rows, tick0, width, out_min, out_mean, out_max) -> None:
    nb = out_min.shape[0]
    for b in range(nb):
        lo, hi = tick0 + b * width, tick0 + (b + 1) * width
        inside = (row_tick >= lo) & (row_tick < hi)
        block = rows[inside]
        for c in range(rows.shape[1]):
            values = block[:, c]
            values = values[np.isfinite(values)]
            if len(values) == 0:
                out_min[b, c] = out_mean[b, c] = out_max[b, c] = np.nan
                continue
            total = 0.0
            for v in values:
                total += v
            out_min[b, c] = values.min()
            out_max[b, c] = values.max()
            out_mean[b, c] = total / len(values)


# =============================================================================== settings

_TEMPLATE_RECORDING: dict = {}


def template_recording() -> dict:
    """The ``recording:`` section of db/_templates/project.yaml: the defaults of a project without one."""
    if not _TEMPLATE_RECORDING:
        from ..db.yamlio import load_file
        from ..paths import DB
        data = load_file(Path(DB) / "_templates" / "project.yaml").data
        _TEMPLATE_RECORDING.update(dict(data.get("recording") or {}))
    return dict(_TEMPLATE_RECORDING)


def _get(section, name):
    if section is None:
        return None
    if isinstance(section, Mapping):
        return section.get(name) if name in section else None
    return getattr(section, name, None)


@dataclass(frozen=True)
class RecordingPolicy:
    """What the recorder keeps: the project's ``recording`` section over the template's, and the
    model settings of db/models/recording.yaml (docs/formats.md "Run directory")."""

    fast: str
    window_revs: float
    period_s: float
    keyframe_period_s: float
    ring_s: float
    ring_max_bytes: int
    buffer_max_bytes: int
    window_min_s: float
    window_max_s: float
    min_spacing_s: float
    max_windows_per_second: int
    checkpoint_interval_s: float

    @classmethod
    def load(cls, recording=None, settings: Mapping | None = None) -> "RecordingPolicy":
        template = template_recording()
        values = {}
        for name in ("fast", "fast_window_revs", "fast_period_s", "keyframe_period_s"):
            given = _get(recording, name)
            values[name] = given if given is not None else template.get(name)
        s = dict(model_params("recording"))
        s.update(settings or {})
        mode = str(values["fast"])
        if mode not in FAST_MODES:
            raise InputError(f"recording.fast {mode!r} is not one of {', '.join(FAST_MODES)}")
        policy = cls(mode, float(values["fast_window_revs"]), float(values["fast_period_s"]),
                     float(values["keyframe_period_s"]), float(s["ring_buffer_s"]), int(s["ring_max_bytes"]),
                     int(s["series_buffer_max_bytes"]), float(s["fast_window_min_s"]), float(s["fast_window_max_s"]),
                     float(s["fast_min_spacing_s"]), int(s["fast_max_windows_per_second"]),
                     float(s["checkpoint_interval_s"]))
        if policy.window_revs <= 0 or policy.period_s <= 0 or policy.window_min_s <= 0 \
                or policy.window_max_s < policy.window_min_s:
            raise InputError(f"recording settings are inconsistent: {policy}")
        return policy


# =============================================================================== storage of rows

_STORAGE = {"float64": "float64", "float32": "float32", "int64": "int64", "int32": "int32", "int8": "int8",
            "bool": "int8"}


def storage_dtype(channel: Channel, tier: str, fast_mode: str) -> str:
    """Parquet column type of a channel's columns: float64/float32 as declared (the fast tier in mode
    ``all``: float32), integers as declared, booleans as int8 (0/1); padding is NaN (floats) or null."""
    dtype = channel.dtype
    if dtype not in _STORAGE:
        raise InputError(f"channel {channel.name}: dtype {dtype!r} is not one of {', '.join(_STORAGE)}")
    if tier == "fast" and fast_mode == "all" and dtype in ("float64", "float32"):
        return "float32"
    return _STORAGE[dtype]


def _is_int(storage: str) -> bool:
    return storage.startswith("int")


def _channel_width(channel: Channel) -> int:
    count = 1
    for size in channel.max_shape:
        count *= int(size)
    return count


def block_of(channel: Channel, value, n_rows: int, per_row: bool) -> np.ndarray:
    """``[n_rows, columns]`` float64 block of one channel value, NaN-padded to ``max_shape``.

    ``per_row``: the value has a leading row axis (tick tier), else it is one row's value (macro tier)
    and is repeated. A scalar channel takes a number; a vector channel an array whose shape is at most
    ``max_shape`` along every axis (the rest is padding)."""
    width = _channel_width(channel)
    shape = tuple(int(s) for s in channel.max_shape)
    array = np.asarray(value, dtype=np.float64)
    if per_row:
        if array.ndim == 0:
            array = np.full((n_rows,) + (() if not shape else (1,) * len(shape)), float(array))
        if array.shape[0] != n_rows:
            raise RecorderError(f"channel {channel.name}: {array.shape[0]} rows given for {n_rows} ticks")
        inner = array.shape[1:]
    else:
        inner = array.shape
        array = array.reshape((1,) + inner)
    if not shape:
        if inner not in ((), (1,)):
            raise RecorderError(f"channel {channel.name} is scalar but got shape {inner}")
        out = array.reshape(array.shape[0], 1)
        return np.repeat(out, n_rows, axis=0) if out.shape[0] != n_rows else np.ascontiguousarray(out)
    if len(inner) != len(shape) or any(a > b for a, b in zip(inner, shape)):
        raise RecorderError(f"channel {channel.name}: shape {inner} exceeds max_shape {shape}")
    full = np.full((array.shape[0],) + shape, np.nan)
    full[(slice(None),) + tuple(slice(0, n) for n in inner)] = array
    out = full.reshape(array.shape[0], width)
    return np.repeat(out, n_rows, axis=0) if out.shape[0] != n_rows else out


class TierTable:
    """The column layout of one tier and its buffered rows (float64 values; converted to the storage
    types when a table is built)."""

    def __init__(self, tier: str, channels: list[Channel], fast_mode: str, envelope_of: list[Channel] = ()):
        self.tier = tier
        self.channels = channels
        self.columns: list[tuple[str, str]] = []           # (column name, storage dtype)
        self.slices: dict[str, slice] = {}
        offset = 0
        for channel in channels:
            storage = storage_dtype(channel, tier, fast_mode)
            names = channel.columns()
            self.columns.extend((name, storage) for name in names)
            self.slices[channel.name] = slice(offset, offset + len(names))
            offset += len(names)
        self.envelope_columns: list[str] = []
        for channel in envelope_of:
            for column in channel.columns():
                self.envelope_columns.extend(f"{column}.{stat}" for stat in ENVELOPE)
        self.columns.extend((name, "float64") for name in self.envelope_columns)
        self.width = len(self.columns)
        self.row_bytes = 16 + sum(np.dtype(s).itemsize for _, s in self.columns)
        self.ticks: list[np.ndarray] = []
        self.blocks: list[np.ndarray] = []
        self.rows = 0

    def schema(self):
        import pyarrow as pa
        fields = [pa.field("tick", pa.int64(), nullable=False), pa.field("time_s", pa.float64(), nullable=False)]
        fields += [pa.field(name, getattr(pa, storage)()) for name, storage in self.columns]
        return pa.schema(fields)

    def append(self, ticks: np.ndarray, block: np.ndarray) -> None:
        if block.shape != (len(ticks), self.width):
            raise RecorderError(f"{self.tier} tier: block {block.shape} for {len(ticks)} rows of {self.width} columns")
        if len(ticks):
            self.ticks.append(np.asarray(ticks, dtype=np.int64).copy())
            self.blocks.append(np.ascontiguousarray(block, dtype=np.float64))
            self.rows += len(ticks)

    @property
    def buffered_bytes(self) -> int:
        return self.rows * self.row_bytes

    def take(self, tick_s: float):
        """The buffered rows as a pyarrow Table (and the buffer emptied); None when empty."""
        import pyarrow as pa
        if not self.rows:
            return None
        ticks = np.concatenate(self.ticks)
        data = np.concatenate(self.blocks, axis=0)
        self.ticks, self.blocks, self.rows = [], [], 0
        arrays = [pa.array(ticks, pa.int64()), pa.array(ticks_to_seconds(ticks, tick_s), pa.float64())]
        for j, (name, storage) in enumerate(self.columns):
            column = data[:, j]
            if _is_int(storage):
                valid = np.isfinite(column)
                values = np.where(valid, np.rint(np.where(valid, column, 0.0)), 0.0).astype(storage)
                arrays.append(pa.array(values, type=getattr(pa, storage)(), mask=~valid))
            else:
                arrays.append(pa.array(column.astype(storage), type=getattr(pa, storage)()))
        return pa.Table.from_arrays(arrays, schema=self.schema())


# =============================================================================== the recorder

class Recorder:
    """Records the channels of a run (module docstring).

    ``registry`` must hold every channel before the recorder is built (the layouts are fixed then).
    ``l0_ticks_min``: the smallest L0 stride of the run (base ticks; it sizes the ring). ``recording``:
    the project's recording section (``ctx.project.recording``) or None for the template's.
    ``replay=(tick0, tick1)``: a deterministic replay (``twin replay --window``) that writes only the
    fast rows of that window, into ``series/fast/replay-<tick0>-<tick1>.parquet``.
    """

    def __init__(self, run_dir, registry: ChannelRegistry, *, l0_ticks_min: int, ipo_ticks: int, macro_ipo: int,
                 tick_s: float = TICK_S, recording=None, settings: Mapping | None = None,
                 replay: tuple[int, int] | None = None) -> None:
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.registry = registry
        self.policy = RecordingPolicy.load(recording, settings)
        self.tick_s = float(tick_s)
        self.ipo_ticks = int(ipo_ticks)
        self.macro_ipo = int(macro_ipo)
        self.l0_ticks_min = int(l0_ticks_min)
        if self.l0_ticks_min <= 0 or self.ipo_ticks <= 0 or self.macro_ipo <= 0:
            raise InputError("the recorder needs positive l0_ticks_min, ipo_ticks and macro_ipo")
        self.ticks_per_second = int(round(1.0 / self.tick_s))
        self.replay = (int(replay[0]), int(replay[1])) if replay is not None else None
        fast = registry.tier(Tier.FAST)
        self.fast_channels = fast
        mode = "windows" if self.replay is not None else self.policy.fast
        self.tables = {"fast": TierTable("fast", fast, "windows" if self.replay else self.policy.fast),
                       "tick": TierTable("tick", registry.tier(Tier.TICK), mode, envelope_of=fast),
                       "macro": TierTable("macro", registry.tier(Tier.MACRO), mode)}
        self.fast_slices = dict(self.tables["fast"].slices)
        # name -> (tier, column slice, scalar): one lookup per value instead of registry calls
        self._lookup = {c.name: (c.tier, self.tables[c.tier.value].slices[c.name], not c.max_shape, c)
                        for c in registry.all()}
        self.n_fast = self.tables["fast"].width
        self.ring = ring_buffer(self._ring_rows(), self.n_fast)
        # window state (checkpointed)
        self.windows: list[list[int]] = []             # pending [t0, t1) in base ticks, sorted, disjoint
        self.written_until = -1                        # last fast tick written
        self.prev_row_tick: int | None = None
        self.last_engagement: int | None = None
        self.rate = [-1, 0]                            # [second, engagement windows in it]
        self.next_periodic = 0
        self.findings_seen = 0
        self.read_head = 0
        self.lost_windows = 0                          # windows that started before the ring's oldest row
        self.window_count = {"engagement": 0, "finding": 0, "periodic": 0, "explicit": 0, "skipped": 0}
        # parts (checkpointed)
        self.parts = {tier: 0 for tier in TIERS}
        self.closed: dict[str, list[str]] = {tier: [] for tier in TIERS}
        self.index_bytes = 0
        self.index_lines = 0
        self._writers: dict = {}
        self._part_range: dict[str, list] = {}
        self._step_tick: tuple | None = None
        if self.run_dir is not None and self.replay is not None:
            self._init_replay()

    # ------------------------------------------------------------------ layout
    def _ring_rows(self) -> int:
        macro_ticks = self.macro_ipo * self.ipo_ticks
        minimum = int(math.ceil(macro_ticks / self.l0_ticks_min)) + int(math.ceil(self.ipo_ticks / self.l0_ticks_min))
        wanted = int(math.ceil(self.policy.ring_s / (self.l0_ticks_min * self.tick_s)))
        by_bytes = self.policy.ring_max_bytes // (8 * max(1, self.n_fast) + 8)
        rows = min(wanted, by_bytes)
        if rows < minimum:
            if self.n_fast and by_bytes < minimum:
                raise InputError(f"the fast ring cannot hold one macro step: {self.n_fast} fast columns at an L0 "
                                 f"stride of {self.l0_ticks_min} ticks need {minimum} rows, the "
                                 f"{self.policy.ring_max_bytes}-byte budget allows {by_bytes} (fewer fast channels, "
                                 f"or a larger ring_max_bytes in db/models/recording.yaml)")
            rows = minimum
        return int(rows)

    @staticmethod
    def fast_all_bytes_per_second(n_columns: int, l0_ticks: int, tick_s: float = TICK_S) -> float:
        """Uncompressed size of ``fast: all`` per simulated second while in contact (float32 columns plus
        the int64 tick and float64 time of every L0 row): what ``twin check`` prints as its estimate."""
        rows_per_s = 1.0 / (int(l0_ticks) * float(tick_s))
        return rows_per_s * (4.0 * int(n_columns) + 16.0)

    @property
    def ring_s(self) -> float:
        return len(self.ring.tick) * self.l0_ticks_min * self.tick_s

    def fast_columns(self, name: str) -> tuple[int, int]:
        """(first column, number of columns) of a fast channel in ``ring.data``."""
        if name not in self.fast_slices:
            raise KeyError(f"{name} is not a fast channel of this run")
        s = self.fast_slices[name]
        return s.start, s.stop - s.start

    def fast_column_names(self) -> list[str]:
        return [name for name, _ in self.tables["fast"].columns]

    def window_ticks(self, omega_rad_s: float) -> int:
        """Length of a window: ``fast_window_revs`` revolutions at |omega|, within [min, max]."""
        omega = abs(float(omega_rad_s))
        seconds = self.policy.window_revs * 2.0 * math.pi / omega if omega > 0.0 else self.policy.window_min_s
        seconds = min(max(seconds, self.policy.window_min_s), self.policy.window_max_s)
        return max(1, int(round(seconds / self.tick_s)))

    # ------------------------------------------------------------------ channels.json
    def channels_document(self) -> dict:
        return {"format": CHANNELS_FORMAT, "version": FORMAT_VERSION,
                "channels": [{"name": c.name, "unit": c.unit, "tier": c.tier.value, "domain": c.domain, "doc": c.doc,
                              "max_shape": list(c.max_shape), "dtype": c.dtype,
                              "storage": storage_dtype(c, c.tier.value, self.policy.fast)}
                             for c in self.registry.all()],
                "fast": {"mode": self.policy.fast, "ring_rows": len(self.ring.tick), "ring_s": self.ring_s,
                         "l0_ticks_min": self.l0_ticks_min, "columns": self.fast_column_names(),
                         "window_revs": self.policy.window_revs, "period_s": self.policy.period_s,
                         "min_spacing_s": self.policy.min_spacing_s,
                         "max_windows_per_second": self.policy.max_windows_per_second,
                         "window_min_s": self.policy.window_min_s, "window_max_s": self.policy.window_max_s},
                "tick_s": self.tick_s, "ipo_ticks": self.ipo_ticks, "macro_ipo": self.macro_ipo}

    def write_channels(self) -> None:
        if self.run_dir is not None:
            write_json_atomic(self.run_dir / "channels.json", self.channels_document())

    # ------------------------------------------------------------------ per step: values
    def record_tick(self, ticks, values: Mapping) -> None:
        """The step's L1 rows: ``ticks`` [n] (base ticks, one per L1 tick), ``values`` channel ->
        [n, *shape] (or a number for every tick). Channels not given are NaN/null."""
        if self.replay is not None:
            return
        if self._step_tick is not None:
            raise RecorderError("record_tick was called twice in one macro step (end_step closes a step)")
        ticks = np.asarray(ticks, dtype=np.int64).reshape(-1)
        table = self.tables["tick"]
        block = np.full((len(ticks), table.width), np.nan)
        n = len(ticks)
        for name, value in values.items():
            channel, columns, scalar = self._channel(name, Tier.TICK)
            if scalar:
                array = np.asarray(value, dtype=np.float64)
                if array.shape in ((), (n,)):
                    block[:, columns.start] = array
                    continue
            block[:, columns] = block_of(channel, value, n, per_row=True)
        self._step_tick = (ticks, block)

    def record_macro(self, tick: int, values: Mapping) -> None:
        """One macro-tier row at ``tick`` (the end of the step)."""
        if self.replay is not None:
            return
        table = self.tables["macro"]
        block = np.full((1, table.width), np.nan)
        for name, value in values.items():
            channel, columns, scalar = self._channel(name, Tier.MACRO)
            if scalar and isinstance(value, (int, float, np.number, bool, np.bool_)):
                block[0, columns.start] = float(value)
                continue
            block[:, columns] = block_of(channel, value, 1, per_row=False)
        table.append(np.array([int(tick)], np.int64), block)
        self._maybe_flush("macro")

    def _channel(self, name: str, tier: Tier):
        """(channel, column slice, scalar) of a registered channel of ``tier``."""
        entry = self._lookup.get(name)
        if entry is None:
            raise RecorderError(f"channel {name!r} is not registered")
        if entry[0] != tier:
            raise RecorderError(f"channel {name} is a {entry[0].value}-tier channel, not {tier.value}")
        return entry[3], entry[1], entry[2]

    # ------------------------------------------------------------------ per step: windows
    def request_window(self, tick0: int, tick1: int, kind: str = "explicit") -> None:
        """Keep the fast rows of ``[tick0, tick1)`` (merged with the other pending windows). Rows already
        written, and rows no longer in the ring, cannot be added any more."""
        tick0, tick1 = int(tick0), int(tick1)
        if tick1 <= tick0:
            return
        if tick1 <= self.written_until + 1:
            self.window_count["skipped"] += 1
            return
        self.window_count[kind] = self.window_count.get(kind, 0) + 1
        n = len(self.ring.tick)
        head = int(self.ring.head[0])
        if head > n and tick0 < int(self.ring.tick[head % n]) and tick0 > self.written_until:
            self.lost_windows += 1          # its first rows were overwritten before the window was asked for
        merged = []
        for a, b in self.windows:
            if b < tick0 or a > tick1:
                merged.append([a, b])
            else:
                tick0, tick1 = min(a, tick0), max(b, tick1)
        merged.append([tick0, tick1])
        self.windows = sorted(merged)

    def engagement(self, tick: int, omega_rad_s: float) -> bool:
        """An engagement starts at ``tick``: its first revolutions are kept unless an engagement window
        started less than ``fast_min_spacing_s`` ago or this second already has the maximum number."""
        tick = int(tick)
        spacing = int(round(self.policy.min_spacing_s / self.tick_s))
        if self.last_engagement is not None and tick - self.last_engagement < spacing:
            self.window_count["skipped"] += 1
            return False
        second = tick // self.ticks_per_second
        if self.rate[0] == second and self.rate[1] >= self.policy.max_windows_per_second:
            self.window_count["skipped"] += 1
            return False
        if self.rate[0] != second:
            self.rate = [second, 0]
        self.rate[1] += 1
        self.last_engagement = tick
        self.request_window(tick, tick + self.window_ticks(omega_rad_s), "engagement")
        return True

    def end_step(self, tick0: int, tick1: int, omega_rad_s: float = 0.0, findings=None) -> None:
        """Close the macro step ``[tick0, tick1)``: read the ring rows pushed in it, cut the fast windows,
        add the tick tier (values and envelopes), flush full buffers. ``omega_rad_s``: the largest
        spindle speed of the step (window lengths); ``findings``: the run's FindingList (windows around
        the new ones)."""
        tick0, tick1 = int(tick0), int(tick1)
        ring = self.ring
        n = len(ring.tick)
        head = int(ring.head[0])
        new = head - self.read_head
        if new < 0:
            raise RecorderError("the ring's head went backwards")
        if new > n:
            raise RecorderError(f"{new - n} fast rows were overwritten before L2 read them (ring of {n} rows); "
                                f"the ring must hold one macro step")
        slots = (self.read_head + np.arange(new, dtype=np.int64)) % n
        new_ticks = ring.tick[slots]
        if new and (np.any(np.diff(new_ticks) <= 0) or new_ticks[0] < tick0 or new_ticks[-1] >= tick1
                    or (self.prev_row_tick is not None and new_ticks[0] <= self.prev_row_tick)):
            raise RecorderError(f"fast rows of the step [{tick0}, {tick1}) are not in rising tick order inside it")
        if self.replay is not None:
            self._replay_rows(new_ticks, ring.data[slots])
        else:
            if self.policy.fast == "windows":
                self._windows(new_ticks, tick1, omega_rad_s, findings)
                self._cut_windows(tick1)
            elif self.policy.fast == "all" and new:
                self.tables["fast"].append(new_ticks, ring.data[slots])
                self.written_until = int(new_ticks[-1])
            self._tick_rows(tick0, tick1, self.read_head, head)
            for tier in TIERS:
                self._maybe_flush(tier)
        self.read_head = head
        if new:
            self.prev_row_tick = int(new_ticks[-1])

    def _windows(self, new_ticks: np.ndarray, tick1: int, omega_rad_s: float, findings) -> None:
        if len(new_ticks):
            previous = np.concatenate(([self.prev_row_tick if self.prev_row_tick is not None else -10 ** 18],
                                       new_ticks[:-1]))
            for start in new_ticks[(new_ticks - previous) > self.ipo_ticks]:
                self.engagement(int(start), omega_rad_s)
        period = max(1, int(round(self.policy.period_s / self.tick_s)))
        while self.next_periodic < tick1:
            self.request_window(self.next_periodic, self.next_periodic + self.window_ticks(omega_rad_s), "periodic")
            self.next_periodic += period
        if findings is not None and len(findings) > self.findings_seen:
            items = findings.items if hasattr(findings, "items") and isinstance(findings.items, list) else list(findings)
            for finding in items[self.findings_seen:]:
                length = self.window_ticks(omega_rad_s)
                centre = int(finding.tick)
                self.request_window(centre - length // 2, centre + (length - length // 2), "finding")
            self.findings_seen = len(items)

    def _first_head(self, tick: int) -> int:
        """The first head (row number since the start) still in the ring whose tick is >= ``tick``: rows
        rise in tick with their head, so a bisection over heads (O(log n), no copy of the ring)."""
        n = len(self.ring.tick)
        hi = int(self.ring.head[0])
        lo = max(0, hi - n)
        ticks = self.ring.tick
        while lo < hi:
            mid = (lo + hi) // 2
            if ticks[mid % n] < tick:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _cut_windows(self, tick1: int) -> None:
        """Copy the ring rows inside pending windows and after the last row written into the fast
        buffer; windows that end by ``tick1`` are done."""
        if not self.windows:
            return
        n = len(self.ring.tick)
        chosen = []
        remaining = []
        for a, b in self.windows:
            lo = max(a, self.written_until + 1)
            if b > lo:
                h0, h1 = self._first_head(lo), self._first_head(b)
                if h1 > h0:
                    chosen.append(np.arange(h0, h1, dtype=np.int64))
            if b > tick1:
                remaining.append([a, b])
        self.windows = remaining
        if chosen:
            picked = np.unique(np.concatenate(chosen)) % n
            rows_ticks = self.ring.tick[picked]
            self.tables["fast"].append(rows_ticks, self.ring.data[picked])
            self.written_until = int(rows_ticks[-1])

    def _tick_rows(self, tick0: int, tick1: int, head0: int, head1: int) -> None:
        table = self.tables["tick"]
        if self._step_tick is None:
            ticks = tick0 + self.ipo_ticks * np.arange(max(0, (tick1 - tick0) // self.ipo_ticks), dtype=np.int64)
            block = np.full((len(ticks), table.width), np.nan)
        else:
            ticks, block = self._step_tick
            self._step_tick = None
        if len(ticks) == 0:
            return
        if len(ticks) > 1 and np.any(np.diff(ticks) != self.ipo_ticks):
            raise RecorderError("tick-tier rows must be one per L1 tick (uniform spacing ipo_ticks)")
        if self.n_fast:
            envelope = np.empty((len(ticks), 3 * self.n_fast))
            ring_envelope(self.ring, int(head0), int(head1), int(ticks[0]), self.ipo_ticks, envelope)
            block[:, table.width - 3 * self.n_fast:] = envelope
        table.append(ticks, block)

    # ------------------------------------------------------------------ parts
    def _maybe_flush(self, tier: str) -> None:
        if self.tables[tier].buffered_bytes >= self.policy.buffer_max_bytes:
            self._flush(tier)

    def _part_paths(self, tier: str) -> tuple[Path, str]:
        if self.replay is not None:
            relative = f"series/fast/replay-{self.replay[0]}-{self.replay[1]}.parquet"
        else:
            relative = f"series/{tier}/part-{self.parts[tier]}.parquet"
        return self.run_dir / relative, relative

    def _flush(self, tier: str) -> None:
        """Write the buffered rows of a tier as one row group of its open part (opened at the first)."""
        import pyarrow.parquet as pq
        table = self.tables[tier]
        if self.run_dir is None or table.rows == 0:
            return
        first, last, rows = int(table.ticks[0][0]), int(table.ticks[-1][-1]), table.rows
        arrow = table.take(self.tick_s)
        if tier not in self._writers:
            final, relative = self._part_paths(tier)
            final.parent.mkdir(parents=True, exist_ok=True)
            tmp = final.with_name(f".{final.name}.tmp")
            meta = {"cnctwin": json.dumps({"format": SERIES_FORMAT, "version": FORMAT_VERSION, "tier": tier,
                                           "part": self.parts[tier], "tick_s": self.tick_s}, sort_keys=True)}
            writer = pq.ParquetWriter(tmp, arrow.schema.with_metadata(meta), compression="zstd",
                                      use_dictionary=False, write_statistics=True)
            self._writers[tier] = (writer, tmp, final, relative)
            self._part_range[tier] = [first, last, 0]
        self._writers[tier][0].write_table(arrow)
        self._part_range[tier][1] = last
        self._part_range[tier][2] += rows

    def close_parts(self) -> list[dict]:
        """Write what is buffered and close every open part (renamed into place, listed in
        series/index.jsonl). Called at each checkpoint and at the end of the run."""
        entries = []
        if self.run_dir is None:
            for table in self.tables.values():
                table.ticks, table.blocks, table.rows = [], [], 0
            return entries
        for tier in TIERS:
            self._flush(tier)
            if tier not in self._writers:
                continue
            writer, tmp, final, relative = self._writers.pop(tier)
            writer.close()
            fsync_file(tmp)
            os.replace(tmp, final)
            tick0, tick1, rows = self._part_range.pop(tier)
            entry = {"tier": tier, "part": self.parts[tier], "file": relative, "tick0": tick0, "tick1": tick1,
                     "t0_s": float(ticks_to_seconds([tick0], self.tick_s)[0]),
                     "t1_s": float(ticks_to_seconds([tick1], self.tick_s)[0]), "rows": rows}
            if self.replay is not None:
                entry["replay"] = True
            line = (json.dumps(entry, sort_keys=True) + "\n").encode("utf-8")
            index = self.run_dir / "series" / "index.jsonl"
            index.parent.mkdir(parents=True, exist_ok=True)
            if not index.exists() or index.stat().st_size == 0:
                line = jsonl_header(SERIES_INDEX_FORMAT) + line       # counted in index_bytes, not in index_lines
            with open(index, "ab") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            self.index_bytes += len(line)
            self.index_lines += 1
            self.closed[tier].append(relative)
            self.parts[tier] += 1
            entries.append(entry)
        if entries:
            fsync_dir(self.run_dir / "series")
        return entries

    def finish(self) -> list[dict]:
        """End of the run (or of a replay): close the parts."""
        return self.close_parts()

    # ------------------------------------------------------------------ replay
    def _init_replay(self) -> None:
        parts = [r for r in _index_records(self.run_dir) if r.get("tier") == "fast"]
        self.parts["fast"] = 1 + max([int(r.get("part", -1)) for r in parts] + [-1])
        self.index_bytes = (self.run_dir / "series" / "index.jsonl").stat().st_size \
            if (self.run_dir / "series" / "index.jsonl").exists() else 0

    def _replay_rows(self, ticks: np.ndarray, data: np.ndarray) -> None:
        inside = (ticks >= self.replay[0]) & (ticks < self.replay[1])
        if np.any(inside):
            self.tables["fast"].append(ticks[inside], data[inside])
            self.written_until = int(ticks[inside][-1])
            self._maybe_flush("fast")

    # ------------------------------------------------------------------ checkpoint participant
    def checkpoint_cut(self) -> dict:
        self.close_parts()
        return {"series": {tier: list(files) for tier, files in self.closed.items()},
                "index_bytes": self.index_bytes, "index_lines": self.index_lines}

    def checkpoint_state(self) -> dict:
        if self._writers or any(t.rows for t in self.tables.values()):
            raise RecorderError("the recorder's state is taken at a checkpoint cut, after close_parts()")
        return {"windows": [list(w) for w in self.windows], "written_until": self.written_until,
                "prev_row_tick": self.prev_row_tick, "last_engagement": self.last_engagement, "rate": list(self.rate),
                "next_periodic": self.next_periodic, "findings_seen": self.findings_seen, "read_head": self.read_head,
                "lost_windows": self.lost_windows, "window_count": dict(self.window_count),
                "parts": dict(self.parts), "closed": {t: list(f) for t, f in self.closed.items()},
                "index_bytes": self.index_bytes, "index_lines": self.index_lines,
                "ring_data": self.ring.data, "ring_tick": self.ring.tick, "ring_head": self.ring.head}

    def restore_state(self, state: Mapping) -> None:
        for name, target in (("ring_data", self.ring.data), ("ring_tick", self.ring.tick), ("ring_head", self.ring.head)):
            source = np.asarray(state[name])
            if source.shape != target.shape or source.dtype != target.dtype:
                raise RecorderError(f"checkpointed ring {name} {source.shape} {source.dtype} does not fit this run's "
                                    f"ring {target.shape} {target.dtype} (other channels or L0 stride)")
            np.copyto(target, source)
        self.windows = [[int(a), int(b)] for a, b in state["windows"]]
        self.written_until = int(state["written_until"])
        self.prev_row_tick = None if state["prev_row_tick"] is None else int(state["prev_row_tick"])
        self.last_engagement = None if state["last_engagement"] is None else int(state["last_engagement"])
        self.rate = [int(x) for x in state["rate"]]
        self.next_periodic = int(state["next_periodic"])
        self.findings_seen = int(state["findings_seen"])
        self.read_head = int(state["read_head"])
        self.lost_windows = int(state["lost_windows"])
        self.window_count = {str(k): int(v) for k, v in state["window_count"].items()}
        self.parts = {str(k): int(v) for k, v in state["parts"].items()}
        self.closed = {str(k): [str(f) for f in v] for k, v in state["closed"].items()}
        self.index_bytes = int(state["index_bytes"])
        self.index_lines = int(state["index_lines"])
        self._writers = {}
        self._part_range = {}
        self._step_tick = None
        for table in self.tables.values():
            table.ticks, table.blocks, table.rows = [], [], 0

    def truncate_outputs(self, cut: Mapping) -> None:
        """Remove the series written after the cut: parts not closed by then, temporary parts, and the
        index lines after its byte length."""
        if self.run_dir is None:
            return
        keep = {f for files in cut["series"].values() for f in files}
        series = self.run_dir / "series"
        if series.is_dir():
            for path in sorted(series.rglob("*.parquet")):
                if path.relative_to(self.run_dir).as_posix() not in keep:
                    path.unlink()
            for path in sorted(series.rglob(".*.tmp")):
                path.unlink()
        index = series / "index.jsonl"
        length = int(cut["index_bytes"])
        if index.exists():
            if index.stat().st_size < length:
                raise RecorderError(f"series/index.jsonl is shorter than its checkpoint cut ({length} bytes)")
            with open(index, "r+b") as handle:
                handle.truncate(length)
        elif length:
            raise RecorderError("series/index.jsonl is missing but the checkpoint cut lists parts")
        missing = [f for f in keep if not (self.run_dir / f).is_file()]
        if missing:
            raise RecorderError(f"series parts of the checkpoint are missing: {', '.join(sorted(missing))}")


# =============================================================================== reading

def _index_records(run_dir: Path) -> list[dict]:
    """The part records of series/index.jsonl (its header line skipped; an index without one reads the same)."""
    path = Path(run_dir) / "series" / "index.jsonl"
    if not path.is_file():
        return []
    out = []
    for raw in path.read_bytes().splitlines(keepends=True):
        if raw.endswith(b"\n") and raw.strip():
            record = json.loads(raw)
            if not is_jsonl_header(record):
                out.append(record)
    return out


def series_parts(run_dir, tier: str) -> list[dict]:
    """The closed parts of a tier in time order (index.jsonl; without it, part files by number)."""
    if tier not in TIERS:
        raise InputError(f"unknown tier {tier!r}: use {', '.join(TIERS)}")
    run_dir = Path(run_dir)
    parts = [r for r in _index_records(run_dir) if r.get("tier") == tier and isinstance(r.get("file"), str)]
    if not parts:
        folder = run_dir / "series" / tier
        if folder.is_dir():
            for path in folder.iterdir():
                match = _PART.match(path.name)
                if match:
                    parts.append({"tier": tier, "part": int(match.group(1)), "file": f"series/{tier}/{path.name}"})
    return sorted(parts, key=lambda r: (int(r.get("tick0", 0)), int(r.get("part", 0)), r["file"]))


def series_columns(run_dir, tier: str) -> list[str]:
    import pyarrow.parquet as pq
    names: list[str] = []
    seen: set[str] = set()
    for part in series_parts(run_dir, tier):
        for name in pq.read_schema(Path(run_dir) / part["file"]).names:
            if name not in seen and name not in ("tick", "time_s"):
                seen.add(name)
                names.append(name)
    return names


def select_columns(available: Sequence[str], tokens: Sequence[str]) -> list[str]:
    """Columns named by tokens: a column exactly, or every column of a channel (``name``, ``name[i]``,
    ``name.min`` ...)."""
    selected: list[str] = []
    for token in tokens:
        token = token.strip()
        if not token or token in ("tick", "time_s"):
            continue
        pattern = re.compile(re.escape(token) + _SUFFIX)
        matches = [c for c in available if pattern.match(c)]
        if not matches:
            raise InputError(f"no channel or column {token!r} in this tier")
        for column in matches:
            if column not in selected:
                selected.append(column)
    return selected


def iter_series(run_dir, tier: str, columns: Sequence[str], tick0: int | None = None,
                tick1: int | None = None) -> Iterator[tuple[np.ndarray, dict]]:
    """``(ticks, {column: float64 values})`` chunks of a tier in time order, ticks in [tick0, tick1]
    (inclusive); a column a part lacks reads NaN, nulls read NaN. Rows a replay part repeats are
    yielded once."""
    import pyarrow.parquet as pq
    run_dir = Path(run_dir)
    last = None
    for part in series_parts(run_dir, tier):
        if tick0 is not None and isinstance(part.get("tick1"), int) and part["tick1"] < tick0:
            continue
        if tick1 is not None and isinstance(part.get("tick0"), int) and part["tick0"] > tick1:
            continue
        path = run_dir / part["file"]
        present = set(pq.read_schema(path).names)
        wanted = [c for c in columns if c in present]
        filters = []
        if tick0 is not None:
            filters.append(("tick", ">=", int(tick0)))
        if tick1 is not None:
            filters.append(("tick", "<=", int(tick1)))
        table = pq.read_table(path, columns=["tick"] + wanted, filters=filters or None, use_threads=False)
        if table.num_rows == 0:
            continue
        ticks = np.asarray(table.column("tick").to_numpy(), dtype=np.int64)
        keep = np.ones(len(ticks), bool) if last is None else ticks > last
        if not np.any(keep):
            continue
        data = {}
        for column in columns:
            if column in present:
                data[column] = np.asarray(table.column(column).to_numpy(zero_copy_only=False), np.float64)[keep]
            else:
                data[column] = np.full(int(keep.sum()), np.nan)
        ticks = ticks[keep]
        last = int(ticks[-1])
        yield ticks, data


class M4:
    """Streaming M4 aggregation (Jugel et al. 2014): per bucket of ``[lo, hi]`` the first, last, minimum
    and maximum sample with their ticks; a bucket with samples but no finite one gives one NaN point.
    Every point returned is a recorded sample, so peaks are never smoothed away."""

    def __init__(self, lo: int, hi: int, buckets: int) -> None:
        self.lo, self.hi, self.n = int(lo), int(hi), max(1, int(buckets))
        self.span = max(1, self.hi - self.lo + 1)
        self.first_t = np.full(self.n, -1, np.int64)
        self.first_v = np.full(self.n, np.nan)
        self.last_t = np.full(self.n, -1, np.int64)
        self.last_v = np.full(self.n, np.nan)
        self.min_t = np.full(self.n, -1, np.int64)
        self.min_v = np.full(self.n, np.inf)
        self.max_t = np.full(self.n, -1, np.int64)
        self.max_v = np.full(self.n, -np.inf)
        self.nan_t = np.full(self.n, -1, np.int64)

    def add(self, ticks: np.ndarray, values: np.ndarray) -> None:
        ticks = np.asarray(ticks, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        b = np.clip((ticks - self.lo) * self.n // self.span, 0, self.n - 1)
        for k in range(len(ticks)):
            bucket, t, v = int(b[k]), int(ticks[k]), float(values[k])
            if not math.isfinite(v):
                if self.nan_t[bucket] < 0:
                    self.nan_t[bucket] = t
                continue
            if self.first_t[bucket] < 0:
                self.first_t[bucket], self.first_v[bucket] = t, v
            self.last_t[bucket], self.last_v[bucket] = t, v
            if v < self.min_v[bucket]:
                self.min_t[bucket], self.min_v[bucket] = t, v
            if v > self.max_v[bucket]:
                self.max_t[bucket], self.max_v[bucket] = t, v

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        ticks, values = [], []
        for bucket in range(self.n):
            if self.first_t[bucket] < 0:
                if self.nan_t[bucket] >= 0:
                    ticks.append(int(self.nan_t[bucket]))
                    values.append(np.nan)
                continue
            points = {int(self.first_t[bucket]): self.first_v[bucket], int(self.last_t[bucket]): self.last_v[bucket],
                      int(self.min_t[bucket]): self.min_v[bucket], int(self.max_t[bucket]): self.max_v[bucket]}
            for t in sorted(points):
                ticks.append(t)
                values.append(float(points[t]))
        return np.asarray(ticks, dtype=np.int64), np.asarray(values, dtype=np.float64)


def read_series(run_dir, tier: str, tokens: Sequence[str], t0_s: float | None = None, t1_s: float | None = None,
                max_points: int = 2000, tick_s: float = TICK_S) -> dict:
    """Columns of a tier between t0 and t1 (inclusive): as recorded up to ``max_points`` rows, else M4
    with ``max_points / 4`` buckets. Returns ``{column: {t_s, v, n}}`` plus ``downsampled``."""
    columns = select_columns(series_columns(run_dir, tier), tokens)
    tick0 = int(round(t0_s / tick_s)) if t0_s is not None else None
    tick1 = int(round(t1_s / tick_s)) if t1_s is not None else None
    chunks = list(iter_series(run_dir, tier, columns, tick0, tick1))
    rows = sum(len(t) for t, _ in chunks)
    out = {"tier": tier, "rows": rows, "downsampled": rows > max_points, "columns": {}}
    if rows <= max_points:
        ticks = np.concatenate([t for t, _ in chunks]) if chunks else np.zeros(0, np.int64)
        for column in columns:
            values = np.concatenate([d[column] for _, d in chunks]) if chunks else np.zeros(0)
            out["columns"][column] = {"t_s": ticks_to_seconds(ticks, tick_s), "v": values, "n": len(ticks)}
        return out
    lo = tick0 if tick0 is not None else int(chunks[0][0][0])
    hi = tick1 if tick1 is not None else int(chunks[-1][0][-1])
    for column in columns:
        acc = M4(lo, hi, max_points // 4)
        for ticks, data in chunks:
            acc.add(ticks, data[column])
        ticks, values = acc.result()
        out["columns"][column] = {"t_s": ticks_to_seconds(ticks, tick_s), "v": values, "n": len(ticks)}
    return out


# =============================================================================== export

def _arrow_table(ticks: np.ndarray, data: Mapping, columns: Sequence[str], tick_s: float):
    import pyarrow as pa
    import pyarrow.compute as pc
    arrays = {"tick": pa.array(ticks, pa.int64()), "time_s": pa.array(ticks_to_seconds(ticks, tick_s), pa.float64())}
    for column in columns:
        values = pa.array(np.asarray(data[column], dtype=np.float64), pa.float64())
        arrays[column] = pc.if_else(pc.is_nan(values), pa.scalar(None, pa.float64()), values)
    return pa.table(arrays)


def export(run_dir, tier: str, tokens: Sequence[str] = (), fmt: str = "csv", out=None, t0_s: float | None = None,
           t1_s: float | None = None, tick_s: float = TICK_S) -> int:
    """Full-resolution export of a tier: CSV (``tick,time_s,<columns>``, header quoted, NaN as an empty
    field) or Parquet (zstd, NaN as null). ``out``: a path, or None for standard output (CSV only).
    Returns the number of rows."""
    import pyarrow.csv as pacsv
    import pyarrow.parquet as pq
    if fmt not in ("csv", "parquet"):
        raise InputError(f"export format {fmt!r}: use csv or parquet")
    available = series_columns(run_dir, tier)
    columns = select_columns(available, tokens) if tokens else list(available)
    tick0 = int(round(t0_s / tick_s)) if t0_s is not None else None
    tick1 = int(round(t1_s / tick_s)) if t1_s is not None else None
    rows = 0
    if fmt == "parquet":
        if out is None:
            raise InputError("a Parquet export needs --out FILE")
        writer = None
        try:
            for ticks, data in iter_series(run_dir, tier, columns, tick0, tick1):
                table = _arrow_table(ticks, data, columns, tick_s)
                if writer is None:
                    writer = pq.ParquetWriter(out, table.schema, compression="zstd")
                writer.write_table(table)
                rows += table.num_rows
            if writer is None:
                empty = _arrow_table(np.zeros(0, np.int64), {c: np.zeros(0) for c in columns}, columns, tick_s)
                writer = pq.ParquetWriter(out, empty.schema, compression="zstd")
                writer.write_table(empty)
        finally:
            if writer is not None:
                writer.close()
        return rows
    target = open(out, "wb") if out is not None else sys.stdout.buffer
    try:
        header = True
        for ticks, data in iter_series(run_dir, tier, columns, tick0, tick1):
            import io
            buffer = io.BytesIO()
            pacsv.write_csv(_arrow_table(ticks, data, columns, tick_s), buffer,
                            write_options=pacsv.WriteOptions(include_header=header))
            header = False
            target.write(buffer.getvalue())
            rows += len(ticks)
        if header:
            target.write((",".join(f'"{c}"' for c in ["tick", "time_s"] + list(columns)) + "\n").encode("utf-8"))
    finally:
        if out is not None:
            target.close()
        else:
            target.flush()
    return rows


def export_main(argv=None) -> int:
    """``twin export RUN_DIR [--tier T] [--channels a,b] [--format csv|parquet] [--t0 S] [--t1 S] [--out F]``."""
    parser = argparse.ArgumentParser(prog="twin export", description="export a run's time series (CSV or Parquet)")
    parser.add_argument("run_dir", help="run directory")
    parser.add_argument("--tier", default="tick", choices=TIERS, help="series tier (default: tick)")
    parser.add_argument("--channels", default="", help="comma-separated channels or columns (default: all)")
    parser.add_argument("--format", default="csv", choices=("csv", "parquet"))
    parser.add_argument("--t0", type=float, default=None, help="first time (s, inclusive)")
    parser.add_argument("--t1", type=float, default=None, help="last time (s, inclusive)")
    parser.add_argument("--out", default=None, help="output file (CSV: default standard output)")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    tick_s = _run_tick_s(run_dir)
    tokens = [t for t in args.channels.split(",") if t.strip()]
    rows = export(run_dir, args.tier, tokens, args.format, args.out, args.t0, args.t1, tick_s)
    if args.out is not None:
        print(f"{rows} rows -> {args.out}", file=sys.stderr)
    return 0


def _run_tick_s(run_dir: Path) -> float:
    for name, path in (("run.json", ("clock", "tick_s")), ("channels.json", ("tick_s",))):
        try:
            document = json.loads((run_dir / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        value = document
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)):
            return float(value)
    return TICK_S


__all__ = ["Recorder", "RecordingPolicy", "RecorderError", "TierTable", "M4", "ring_begin", "ring_put", "tick_envelope",
           "ring_envelope", "ring_begin_reference", "ring_put_reference", "tick_envelope_reference",
           "ring_envelope_reference", "iter_series", "read_series",
           "series_parts", "series_columns", "select_columns", "export", "export_main", "block_of", "storage_dtype",
           "template_recording", "jsonable", "TIERS", "FAST_MODES"]
