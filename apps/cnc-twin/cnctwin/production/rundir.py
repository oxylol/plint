"""production.json from a finished run directory (docs/architecture.md 16).

Reads ``timeline.json`` (executed blocks), ``frames.bin`` with its header ``scene.json["frames"]``
(per L1 tick: ``tick``, ``axis`` values relative to the stock origin, ``spindle_angle_rad`` wrapped
to [0, 2 pi) and ``spindle_rev``), the tick-tier series (``cutting.spindle_power_W``, the shaft power
of cutting), ``tools.json`` (per tool ``number``, ``life_used_fraction``, ``VB_max_m``, ``cut_time_s``,
optional ``per_part`` list and ``wear_offsets_m``) and ``run.json`` (clock, summary: removed mass,
feed work) unless the caller passes them. Spindle speed and axis velocities are the forward
differences of the recorded angles and positions over each L1 tick; changes within the float32
quantisation of frames.bin are held (they would otherwise add up as accelerations).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core.errors import InputError
from ..sanity.context import ToolFacts
from ..sanity.si import si_view
from .batch import EnergyInputs, PartInputs, ProductionAggregator, ToolState, write_production
from .timeline import read_timeline

SHAFT_POWER_CHANNEL = "cutting.spindle_power_W"


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise InputError(f"cannot read {path.name}: {error.strerror}", str(path)) from None


def read_frames(run_dir) -> dict:
    """Per-tick arrays of frames.bin: ``tick`` [n], ``axis`` [n, n_axes] (SI), ``axis_names``,
    ``spindle_angle_rad`` [n, n_paths] unwrapped with the revolution count."""
    run_dir = Path(run_dir)
    scene = _json(run_dir / "scene.json")
    header = scene.get("frames") if isinstance(scene, dict) else None
    if not isinstance(header, dict) or "fields" not in header:
        raise InputError("scene.json has no frames header", str(run_dir / "scene.json"))
    fields = []
    for entry in sorted(header["fields"], key=lambda f: int(f["offset"])):
        shape = tuple(int(s) for s in entry.get("shape", []))
        fields.append((entry["name"], np.dtype(entry["dtype"]), shape, int(entry["offset"])))
    dtype = np.dtype({"names": [f[0] for f in fields],
                      "formats": [(f[1], f[2]) if f[2] else f[1] for f in fields],
                      "offsets": [f[3] for f in fields], "itemsize": int(header["record_bytes"])})
    names = set(dtype.names)
    for required in ("tick", "axis", "spindle_angle_rad", "spindle_rev"):
        if required not in names:
            raise InputError(f"frames.bin header has no field {required!r}", str(run_dir / "scene.json"))
    data = np.fromfile(run_dir / "frames.bin", dtype=dtype)
    angle = np.asarray(data["spindle_angle_rad"], dtype=float).reshape(len(data), -1)
    revs = np.asarray(data["spindle_rev"], dtype=float).reshape(len(data), -1)
    axes = [a["name"] for a in header.get("axes", [])]
    axis_values = np.asarray(data["axis"], dtype=float).reshape(len(data), -1)
    if axes and len(axes) != axis_values.shape[1]:
        raise InputError(f"frames.bin has {axis_values.shape[1]} axis values, the header names {len(axes)} axes")
    kinds = [a.get("kind", "linear") for a in header.get("axes", [])]
    return {"tick": np.asarray(data["tick"], dtype=np.int64), "axis": axis_values, "axis_names": tuple(axes),
            "axis_kinds": tuple(kinds), "spindle_angle_rad": angle + 2.0 * np.pi * revs,
            "tick_s": float(header.get("tick_s", 1e-6))}


def read_series_column(run_dir, tier: str, column: str) -> tuple[np.ndarray, np.ndarray] | None:
    """``(tick, values)`` of one column over every closed series part of ``tier`` (None if absent)."""
    import pyarrow.parquet as pq

    run_dir = Path(run_dir)
    index = run_dir / "series" / "index.jsonl"
    if not index.exists():
        return None
    ticks, values = [], []
    for raw in index.read_text(encoding="utf-8").splitlines():
        entry = json.loads(raw)
        if entry.get("tier") != tier:
            continue
        table = pq.read_table(run_dir / entry["file"])
        if column not in table.column_names:
            return None
        ticks.append(np.asarray(table.column("tick").to_numpy(), dtype=np.int64))
        values.append(np.asarray(table.column(column).to_numpy(), dtype=float))
    if not ticks:
        return None
    tick = np.concatenate(ticks)
    order = np.argsort(tick, kind="stable")
    return tick[order], np.concatenate(values)[order]


def energy_inputs(frames: dict, power: tuple | None, tick0: int, tick1: int, tick_s: float,
                  feed_work_J: float = 0.0) -> EnergyInputs:
    """Signals of the ticks in ``[tick0, tick1)``: forward differences of angle and positions."""
    tick = frames["tick"]
    select = (tick >= tick0) & (tick < tick1)
    t = tick[select]
    if len(t) == 0:
        raise InputError(f"frames.bin has no record between ticks {tick0} and {tick1}")
    following = np.append(t[1:], tick1)
    dt = (following - t) * tick_s
    angle = frames["spindle_angle_rad"][select][:, 0]
    position = frames["axis"][select]
    omega = np.zeros(len(t))
    velocity = np.zeros_like(position)
    if len(t) > 1:
        omega[:-1] = np.diff(angle) / dt[:-1]
        velocity[:-1] = np.diff(position, axis=0) / dt[:-1, None]
    linear = [k for k, kind in enumerate(frames["axis_kinds"]) if kind == "linear"] or list(range(position.shape[1]))
    shaft = np.zeros(len(t))
    if power is not None:
        power_tick, power_values = power
        at = np.searchsorted(power_tick, t, side="right") - 1
        valid = at >= 0
        shaft[valid] = np.nan_to_num(power_values[at[valid]])
    names = tuple(frames["axis_names"][k] for k in linear) if frames["axis_names"] else tuple(f"A{k}" for k in linear)
    step = float(np.min(dt[dt > 0])) if np.any(dt > 0) else 1.0
    # frames.bin stores float32: two units in the last place of the angle (< 2 pi) and of the largest
    # position over one tick are the noise a difference can carry
    omega_tol = 2.0 * float(np.spacing(np.float32(2.0 * np.pi))) / step
    span = np.max(np.abs(position[:, linear]), axis=0) if len(position) else np.zeros(len(linear))
    velocity_tol = 2.0 * np.spacing(span.astype(np.float32)).astype(float) / step
    return EnergyInputs(dt, omega, shaft, velocity[:, linear], names, None, feed_work_J, omega_tol, velocity_tol)


def tool_states(document: dict, part: int, parts: int) -> list[ToolState]:
    states = []
    for entry in document.get("tools", []):
        rows = entry.get("per_part")
        record = entry
        if isinstance(rows, list) and rows:
            record = next((r for r in rows if int(r.get("part", -1)) == part), None)
            if record is None:
                continue
        elif parts > 1:
            raise InputError("tools.json gives one state per tool but the run has several parts: give per_part")
        states.append(ToolState(number=int(entry.get("number", 0)), slot=int(entry.get("slot", -1)),
                                tool_id=str(entry.get("tool_id", "")),
                                life_used_frac=_float(record.get("life_used_fraction")),
                                life_used_part_frac=_float(record.get("life_used_part_fraction")),
                                cut_time_s=_float(record.get("cut_time_s")), VB_max_m=_float(record.get("VB_max_m")),
                                wear_offsets_m={k: float(v) for k, v in (record.get("wear_offsets_m") or {}).items()},
                                changes=int(record.get("changes", 0)), broken=bool(record.get("broken", False))))
    return states


def _float(value):
    return None if value is None else float(value)


def production_from_run_dir(run_dir, *, project=None, machine=None, coolant=None,
                            tools: dict[int, ToolFacts] | None = None, tick_s: float | None = None,
                            removed_mass_kg: list[float] | float | None = None, feed_work_J: float | None = None,
                            parts_planned: int | None = None, payload_kg: dict[str, float] | None = None,
                            write: bool = True) -> dict:
    """Compose production.json for a finished run (and write it unless ``write`` is False).

    ``project`` is the resolved project (SI view or mapping; run.json's when not given) and
    ``machine``/``coolant`` override its machine and coolant fluid (both are needed resolved)."""
    run_dir = Path(run_dir)
    run = _json(run_dir / "run.json") if (run_dir / "run.json").exists() else {}
    project = si_view(project if project is not None else run.get("project", {}))
    if tick_s is None:
        tick_s = float(run.get("clock", {}).get("tick_s", 1e-6)) if run else 1e-6
    aggregator = ProductionAggregator.from_project(project, tick_s=tick_s, machine=machine, coolant=coolant,
                                                   tools=tools, parts_planned=parts_planned, payload_kg=payload_kg)
    timeline = read_timeline(run_dir)
    frames = read_frames(run_dir)
    power = read_series_column(run_dir, "tick", SHAFT_POWER_CHANNEL)
    tools_document = _json(run_dir / "tools.json") if (run_dir / "tools.json").exists() else {"tools": []}
    summary = run.get("summary", {}) if run else {}
    parts = timeline.parts()
    masses = _per_part(removed_mass_kg if removed_mass_kg is not None else _summary_masses(summary), len(parts))
    feed = feed_work_J if feed_work_J is not None else _float(summary.get("feed_work_J"))
    if power is None:
        aggregator.assumptions.append(f"no {SHAFT_POWER_CHANNEL} series: spindle cutting power taken as 0")
    if feed is None:
        aggregator.assumptions.append("no feed work in the run summary: axis process power taken as 0 "
                                      "(it is typically below 1 % of the spindle power)")
    for k, part in enumerate(parts):
        blocks = timeline.of_part(part)
        tick0 = min(b.tick0 for b in blocks)
        tick1 = max(b.tick1 for b in blocks)
        part_feed = (feed or 0.0) / len(parts)
        inputs = PartInputs(part=part, blocks=blocks,
                            energy=energy_inputs(frames, power, tick0, tick1, tick_s, part_feed),
                            removed_mass_kg=masses[k], tools=tool_states(tools_document, part, len(parts)))
        aggregator.add_part(inputs)
    document = aggregator.result()
    if write:
        write_production(run_dir, document)
    return document


def _summary_masses(summary: dict):
    rows = summary.get("parts")
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        return [_float(r.get("removed_mass_kg")) for r in rows]
    return _float(summary.get("removed_mass_kg"))


def _per_part(value, n: int) -> list[float | None]:
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise InputError(f"removed mass given for {len(value)} parts, the timeline has {n}")
        return [None if v is None else float(v) for v in value]
    if value is None:
        return [None] * n
    if n == 1:
        return [float(value)]
    raise InputError("one removed mass for several parts: give one per part")
