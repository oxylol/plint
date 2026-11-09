"""Reductions of a run's time series for the dataset: one streaming pass over the tick tier and one over the macro
tier (cnctwin.core.recorder.iter_series, part by part), giving run-level statistics and per-block statistics.

Rows and blocks (docs/formats.md "series/", "timeline.json"): a tick row at tick ``t`` belongs to the block of its
path whose ``[tick0, tick1)`` holds ``t``; a macro row at tick ``T`` (the end of its step) summarises
``(T - macro_ticks, T]`` and counts for every block of its path that overlaps that interval (a block shorter than
a step shares the step's values with its neighbours; the schema says so for every macro label). Channels with a
per-path shape are read at the block's path, per-workpiece and per-spindle channels at their maximum (the
spindle speed at the path's spindle). Missing channels read NaN (cnctwin.core.recorder fills them), so a run
without a domain gives NaN, never a wrong number. Sums are accumulated in time order: the same run gives the same
bits.
"""

from __future__ import annotations

import math

import numpy as np


class Stat:
    """max, min, sum and count of finite values per slot (a block, or slot 0 for the whole run)."""

    __slots__ = ("max", "min", "sum", "count")

    def __init__(self, n: int) -> None:
        self.max = np.full(n, np.nan)
        self.min = np.full(n, np.nan)
        self.sum = np.zeros(n)
        self.count = np.zeros(n, dtype=np.int64)

    def add(self, idx: np.ndarray, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(values)
        if not np.any(finite):
            return
        idx, values = np.asarray(idx)[finite], values[finite]
        np.fmax.at(self.max, idx, values)
        np.fmin.at(self.min, idx, values)
        np.add.at(self.sum, idx, values)
        np.add.at(self.count, idx, 1)

    def mean(self) -> np.ndarray:
        out = np.full(len(self.sum), np.nan)
        ok = self.count > 0
        out[ok] = self.sum[ok] / self.count[ok]
        return out


class Stats:
    """Named Stats over ``n`` slots."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.items: dict[str, Stat] = {}

    def __getitem__(self, name: str) -> Stat:
        if name not in self.items:
            self.items[name] = Stat(self.n)
        return self.items[name]

    def get(self, name: str, what: str, slot: int = 0) -> float:
        stat = self.items.get(name)
        if stat is None:
            return math.nan
        if what == "mean":
            return float(stat.mean()[slot])
        return float(getattr(stat, what)[slot])


def _norm(*arrays) -> np.ndarray:
    total = np.zeros_like(np.asarray(arrays[0], dtype=np.float64))
    for a in arrays:
        total = total + np.asarray(a, dtype=np.float64) ** 2
    return np.sqrt(total)


class BlockIndex:
    """The blocks of the run grouped by path: sorted starts and ends, and each block's row in the table."""

    def __init__(self, blocks: list[dict]) -> None:
        self.paths: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        groups: dict[int, list[tuple[int, int, int]]] = {}
        for row, block in enumerate(blocks):
            t0, t1 = int(block["tick0"]), int(block["tick1"])
            if t1 > t0:
                groups.setdefault(int(block.get("path", 0) or 0), []).append((t0, t1, row))
        for path, items in groups.items():
            items.sort()
            self.paths[path] = (np.array([i[0] for i in items], np.int64), np.array([i[1] for i in items], np.int64),
                                np.array([i[2] for i in items], np.int64))

    def at_ticks(self, path: int, ticks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(rows of ``ticks`` inside a block of ``path``, their block rows)."""
        if path not in self.paths:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        starts, ends, rows = self.paths[path]
        k = np.searchsorted(starts, ticks, side="right") - 1
        inside = (k >= 0) & (ticks < ends[np.clip(k, 0, len(ends) - 1)])
        return np.nonzero(inside)[0], rows[k[inside]]

    def overlapping(self, path: int, ticks: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
        """Pairs (row, block row) of every row interval ``(T - width, T]`` and every block of ``path`` it overlaps."""
        if path not in self.paths or len(ticks) == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        starts, ends, rows = self.paths[path]
        lo = np.searchsorted(ends, ticks - width, side="right")
        hi = np.searchsorted(starts, ticks, side="left")
        counts = np.maximum(hi - lo, 0)
        total = int(counts.sum())
        if total == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        row_idx = np.repeat(np.arange(len(ticks), dtype=np.int64), counts)
        offsets = np.repeat(np.cumsum(counts) - counts, counts)
        block_pos = np.arange(total, dtype=np.int64) - offsets + np.repeat(lo, counts)
        return row_idx, rows[block_pos]


def _columns(run_dir, tier: str) -> list[str]:
    from ..core.recorder import series_columns
    try:
        return series_columns(run_dir, tier)
    except Exception:       # a run without series: no columns, every statistic NaN
        return []


def _indexed(available: list[str], name: str) -> list[int]:
    prefix = name + "["
    out = []
    for column in available:
        if column.startswith(prefix) and column.endswith("]") and column[len(prefix):-1].isdigit():
            out.append(int(column[len(prefix):-1]))
    return sorted(out)


def _col(data: dict, name: str, n: int) -> np.ndarray:
    value = data.get(name)
    return np.asarray(value, dtype=np.float64) if value is not None else np.full(n, np.nan)


def _max_over(data: dict, name: str, indexes: list[int], n: int) -> np.ndarray:
    if not indexes:
        return _col(data, name, n)
    stack = np.vstack([_col(data, f"{name}[{i}]", n) for i in indexes])
    with np.errstate(all="ignore"):
        out = np.full(n, np.nan)
        finite = np.isfinite(stack)
        any_finite = finite.any(axis=0)
        out[any_finite] = np.nanmax(np.where(finite, stack, -np.inf)[:, any_finite], axis=0)
    return out


# The channels the dataset reads (docs/ml.md "Columns"): tier and what their index is
TICK_PATH = ("cutting.force_x_N", "cutting.force_y_N", "cutting.force_z_N", "cutting.torque_Nm",
             "dynamics.tool_deflection_x_m", "dynamics.tool_deflection_y_m", "motion.path_speed_m_s")
TICK_SPINDLE = ("motion.spindle_omega_rad_s",)
MACRO_SCALAR = ("thermal.tool_edge_max_K", "thermal.tool_flank_max_K", "chips.max_temp_K", "coolant.on",
                "coolant.flow_m3_s", "coolant.tank_temp_K", "wear.VB_max_m", "holding.tool_margin")
MACRO_PATH = ("cutting.force_peak_N", "cutting.spindle_power_W", "cutting.spindle_load_fraction",
              "cutting.spindle_thermal_load", "cutting.chip_volume_rate_m3_s", "cutting.in_cut_fraction",
              "removal.ae_m", "removal.ap_m", "removal.engaged_angle_rad", "dynamics.vibration_pp_path_m",
              "dynamics.chatter_index_path", "dynamics.stability_margin")
MACRO_MAX_OVER = ("thermal.work_bulk_max_K", "thermal.work_surface_K", "thermal.spindle_temp_K", "holding.part_slip_m")
MACRO_SLOT = ("wear.life_used_slot",)
CHANNELS = {**{n: ("tick", "path") for n in TICK_PATH}, **{n: ("tick", "spindle") for n in TICK_SPINDLE},
            **{n: ("macro", None) for n in MACRO_SCALAR}, **{n: ("macro", "path") for n in MACRO_PATH},
            **{n: ("macro", "max") for n in MACRO_MAX_OVER}, **{n: ("macro", "slot") for n in MACRO_SLOT}}


def reduce_series(run_dir, blocks: list[dict], *, paths: list[int], spindle_of_path: dict[int, int],
                  slot_of_block: list[int], macro_ticks: int) -> tuple[Stats, Stats]:
    """(run statistics in slot 0, block statistics, one slot per block of ``blocks``)."""
    from ..core.recorder import iter_series
    index = BlockIndex(blocks)
    run = Stats(1)
    per = Stats(max(1, len(blocks)))
    paths = sorted(set(paths) | set(index.paths))
    tick_cols = _columns(run_dir, "tick")
    if tick_cols:
        wanted = [f"{name}[{p}]" for p in paths for name in TICK_PATH]
        wanted += [f"{name}[{spindle_of_path.get(p, 0)}]" for p in paths for name in TICK_SPINDLE]
        wanted = sorted(set(c for c in wanted if c in tick_cols))
        for ticks, data in iter_series(run_dir, "tick", wanted):
            _tick_chunk(ticks, data, paths, spindle_of_path, index, run, per)
    macro_cols = _columns(run_dir, "macro")
    if macro_cols:
        shapes = {name: _indexed(macro_cols, name) for name in MACRO_MAX_OVER}
        slots = sorted({s for s in slot_of_block if s >= 0})
        wanted = list(MACRO_SCALAR)
        wanted += [f"{name}[{i}]" for name, indexes in shapes.items() for i in indexes]
        wanted += [f"{name}[{k}]" for name in MACRO_SLOT for k in slots]
        wanted += [f"{name}[{p}]" for p in paths for name in MACRO_PATH]
        wanted = sorted(set(c for c in wanted if c in macro_cols))
        for ticks, data in iter_series(run_dir, "macro", wanted):
            _macro_chunk(ticks, data, paths, shapes, slots, slot_of_block, index, macro_ticks, run, per)
    return run, per


def _tick_chunk(ticks, data, paths, spindle_of_path, index: BlockIndex, run: Stats, per: Stats) -> None:
    n = len(ticks)
    zero = np.zeros(n, np.int64)
    for p in paths:
        force = _norm(_col(data, f"cutting.force_x_N[{p}]", n), _col(data, f"cutting.force_y_N[{p}]", n),
                      _col(data, f"cutting.force_z_N[{p}]", n))
        torque = np.abs(_col(data, f"cutting.torque_Nm[{p}]", n))
        deflection = _norm(_col(data, f"dynamics.tool_deflection_x_m[{p}]", n),
                           _col(data, f"dynamics.tool_deflection_y_m[{p}]", n))
        speed = _col(data, f"motion.path_speed_m_s[{p}]", n)
        omega = np.abs(_col(data, f"motion.spindle_omega_rad_s[{spindle_of_path.get(p, 0)}]", n))
        incut = np.isfinite(force) & (force > 0.0)
        run["force"].add(zero, force)
        run["force_cut"].add(zero[incut], force[incut])
        run["torque"].add(zero, torque)
        run["torque_cut"].add(zero[incut], torque[incut])
        run["deflection"].add(zero, deflection)
        rows, blocks = index.at_ticks(p, np.asarray(ticks, np.int64))
        if len(rows) == 0:
            continue
        cut = incut[rows]
        per["force"].add(blocks, force[rows])
        per["force_cut"].add(blocks[cut], force[rows][cut])
        per["torque"].add(blocks, torque[rows])
        per["torque_cut"].add(blocks[cut], torque[rows][cut])
        per["deflection"].add(blocks, deflection[rows])
        per["speed"].add(blocks, speed[rows])
        per["speed_cut"].add(blocks[cut], speed[rows][cut])
        per["omega"].add(blocks, omega[rows])


def _macro_chunk(ticks, data, paths, shapes, slots, slot_of_block, index: BlockIndex, macro_ticks: int, run: Stats,
                 per: Stats) -> None:
    n = len(ticks)
    zero = np.zeros(n, np.int64)
    scalars = {"edge": _col(data, "thermal.tool_edge_max_K", n), "flank": _col(data, "thermal.tool_flank_max_K", n),
               "chip_temp": _col(data, "chips.max_temp_K", n), "coolant_on": _col(data, "coolant.on", n),
               "coolant_flow": _col(data, "coolant.flow_m3_s", n), "tank": _col(data, "coolant.tank_temp_K", n),
               "vb": _col(data, "wear.VB_max_m", n), "holder_margin": _col(data, "holding.tool_margin", n),
               "work": _max_over(data, "thermal.work_bulk_max_K", shapes["thermal.work_bulk_max_K"], n),
               "surface_temp": _max_over(data, "thermal.work_surface_K", shapes["thermal.work_surface_K"], n),
               "spindle_temp": _max_over(data, "thermal.spindle_temp_K", shapes["thermal.spindle_temp_K"], n),
               "slip": _max_over(data, "holding.part_slip_m", shapes["holding.part_slip_m"], n)}
    for name, values in scalars.items():
        run[name].add(zero, values)
    life = np.vstack([_col(data, f"wear.life_used_slot[{k}]", n) for k in slots]) if slots else None
    slot_pos = {k: i for i, k in enumerate(slots)}
    ticks = np.asarray(ticks, np.int64)
    for p in paths:
        series = {"force_peak": _col(data, f"cutting.force_peak_N[{p}]", n),
                  "power": _col(data, f"cutting.spindle_power_W[{p}]", n),
                  "load": _col(data, f"cutting.spindle_load_fraction[{p}]", n),
                  "thermal_load": _col(data, f"cutting.spindle_thermal_load[{p}]", n),
                  "mrr": _col(data, f"cutting.chip_volume_rate_m3_s[{p}]", n),
                  "in_cut": _col(data, f"cutting.in_cut_fraction[{p}]", n),
                  "ae": _col(data, f"removal.ae_m[{p}]", n), "ap": _col(data, f"removal.ap_m[{p}]", n),
                  "angle": _col(data, f"removal.engaged_angle_rad[{p}]", n),
                  "vibration": _col(data, f"dynamics.vibration_pp_path_m[{p}]", n),
                  "chatter": _col(data, f"dynamics.chatter_index_path[{p}]", n),
                  "margin": _col(data, f"dynamics.stability_margin[{p}]", n)}
        cutting = np.isfinite(series["in_cut"]) & (series["in_cut"] > 0.0)
        margin = series["margin"]
        valid_margin = np.isfinite(margin) & (margin > 0.0)
        for name in ("force_peak", "power", "load", "thermal_load", "vibration", "chatter"):
            run[name].add(zero, series[name])
        run["power_cut"].add(zero[cutting], series["power"][cutting])
        run["mrr_cut"].add(zero[cutting], series["mrr"][cutting])
        run["margin"].add(zero[valid_margin], margin[valid_margin])
        rows, blocks = index.overlapping(p, ticks, int(macro_ticks))
        if len(rows) == 0:
            continue
        cut = cutting[rows]
        for name in ("force_peak", "power", "load", "ae", "ap", "angle", "vibration", "chatter"):
            per[name].add(blocks, series[name][rows])
        per["power_cut"].add(blocks[cut], series["power"][rows][cut])
        per["mrr_cut"].add(blocks[cut], series["mrr"][rows][cut])
        good = valid_margin[rows]
        per["margin"].add(blocks[good], margin[rows][good])
        for name in ("edge", "work", "coolant_on", "coolant_flow", "vb"):
            per[name].add(blocks, scalars[name][rows])
        if life is not None:
            positions = np.array([slot_pos.get(slot_of_block[b], -1) for b in blocks], dtype=np.int64)
            ok = positions >= 0
            if np.any(ok):
                per["life"].add(blocks[ok], life[positions[ok], rows[ok]])


__all__ = ["reduce_series", "Stats", "Stat", "BlockIndex", "CHANNELS"]
