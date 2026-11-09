"""The thermal domain's SimState field (checkpointable: to_arrays / from_arrays).

Everything needed for a bit-identical continuation is here: tool ladders and edge counters,
workpiece grids (fractions, enthalpies, temperatures, contact masks, pending heat rows), fixture,
spindle, axis and structure bodies, the fraction tables in force, the solver ledgers, the damage
findings already raised and the counters of the cadence. The workpiece geometry, edge and contact
providers are not state: ThermalDomain attaches them after a resume.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .abi import MatTable
from .damage import DamageMonitor
from .machine import AxisThermal, SpindleThermal, StructureThermal
from .partition import FractionTable
from .schema import ThermalSettings
from .toolmodel import ToolThermal
from .workpiece import WorkThermal

_FRACTION_ARRAYS = ("h_grid_m", "shear_to_chip", "rake_to_tool", "flank_to_tool", "chip_ratio", "shear_angle_rad")


def _f(value) -> float:
    return float(value)


@dataclass
class ThermalState:
    settings: ThermalSettings
    ambient_K: float
    material: MatTable
    material_family: str
    damage: DamageMonitor
    structure: StructureThermal
    time_s: float = 0.0
    tools: dict = field(default_factory=dict)           # slot -> ToolThermal
    works: dict = field(default_factory=dict)           # workpiece -> WorkThermal
    spindles: dict = field(default_factory=dict)        # spindle id -> SpindleThermal
    axes: dict = field(default_factory=dict)            # axis name -> AxisThermal
    fractions: dict = field(default_factory=dict)       # slot -> FractionTable in force
    published: dict = field(default_factory=dict)       # workpiece -> published temperature grid
    macro_since_fv: int = 0
    engaged_s: dict = field(default_factory=dict)       # contact -> continuous engagement (s)
    last_axis_m: dict = field(default_factory=dict)     # axis -> last actual position (SI)
    unload: list = field(default_factory=list)          # operation ends: {op, workpiece, mean_K, max_K, time_s}
    totals: dict = field(default_factory=dict)          # cumulative J by destination
    maxima: dict = field(default_factory=dict)          # report maxima
    values: dict = field(default_factory=dict)          # channel values of the last step
    estimated: list = field(default_factory=list)       # parameters taken from model defaults
    part_index: int = 0
    op_index: int = 0
    fixture_mass_kg: float = 0.0
    trace_notes: dict = field(default_factory=dict)     # what was already written to the trace once
    spindle_pending: dict = field(default_factory=dict)  # spindle id -> {n, dt, heat_J, omega}: not yet stepped

    def tool_temperature_at(self, slot: int, points_T) -> np.ndarray:
        """Tool temperatures (K) at points [n, 3] of the tool frame T of a slot (rotating tools at spindle
        angle 0): the 3-D cutting-region field where it exists (docs/models/thermal.md 2.4), else the
        tool ladder's body temperature at the point (no local field)."""
        tool = self.tools.get(int(slot))
        if tool is None:
            raise KeyError(f"thermal: no tool in slot {slot}")
        return tool.temperature_at_T(points_T)

    # ------------------------------------------------------------------ checkpoint
    def to_arrays(self) -> tuple[dict, dict]:
        arrays: dict = {}
        meta: dict = {"settings": dict(self.settings.__dict__), "ambient_K": self.ambient_K,
                      "material_family": self.material_family, "damage": self.damage.to_meta(),
                      "structure": self.structure.to_dict(), "time_s": self.time_s,
                      "macro_since_fv": self.macro_since_fv,
                      "engaged_s": [[int(k), _f(v)] for k, v in sorted(self.engaged_s.items())],
                      "last_axis_m": [[str(k), _f(v)] for k, v in sorted(self.last_axis_m.items())],
                      "unload": list(self.unload), "totals": {k: _f(v) for k, v in sorted(self.totals.items())},
                      "maxima": {k: _f(v) for k, v in sorted(self.maxima.items())},
                      "estimated": list(self.estimated), "part_index": self.part_index, "op_index": self.op_index,
                      "fixture_mass_kg": self.fixture_mass_kg, "trace_notes": dict(sorted(self.trace_notes.items())),
                      "tools": [], "works": [], "fractions": [], "published": [],
                      "spindles": [m.to_dict() for _, m in sorted(self.spindles.items())],
                      "axes": [m.to_dict() for _, m in sorted(self.axes.items())],
                      "spindle_pending": {sid: {"n": int(v["n"]), "dt": _f(v["dt"]), "heat_J": _f(v["heat_J"]),
                                                "omega": [_f(w) for w in v["omega"]]}
                                          for sid, v in sorted(self.spindle_pending.items())}}
        for name in MatTable._fields:
            arrays[f"material.{name}"] = np.array(getattr(self.material, name))
        for slot, tool in sorted(self.tools.items()):
            a, m = tool.to_arrays(f"tool.{slot}.")
            arrays.update(a)
            meta["tools"].append(m)
        for w, work in sorted(self.works.items()):
            a, m = work.to_arrays(f"work.{w}.")
            arrays.update(a)
            meta["works"].append(m)
        for slot, table in sorted(self.fractions.items()):
            for name in _FRACTION_ARRAYS:
                arrays[f"frac.{slot}.{name}"] = np.array(getattr(table, name))
            meta["fractions"].append([int(slot), int(table.version)])
        for w, grid in sorted(self.published.items()):
            arrays[f"pub.{w}"] = np.array(grid)
            meta["published"].append(int(w))
        return arrays, meta

    @classmethod
    def from_arrays(cls, arrays: dict, meta: dict) -> "ThermalState":
        settings = ThermalSettings(**meta["settings"])
        material = MatTable(*(np.ascontiguousarray(arrays[f"material.{name}"]).copy() for name in MatTable._fields))
        obj = cls(settings=settings, ambient_K=float(meta["ambient_K"]), material=material,
                  material_family=meta["material_family"], damage=DamageMonitor.from_meta(meta["damage"]),
                  structure=StructureThermal.from_dict(meta["structure"]), time_s=float(meta["time_s"]))
        obj.macro_since_fv = int(meta["macro_since_fv"])
        obj.engaged_s = {int(k): float(v) for k, v in meta["engaged_s"]}
        obj.last_axis_m = {str(k): float(v) for k, v in meta["last_axis_m"]}
        obj.unload = list(meta["unload"])
        obj.totals = {k: float(v) for k, v in meta["totals"].items()}
        obj.maxima = {k: float(v) for k, v in meta["maxima"].items()}
        obj.estimated = list(meta["estimated"])
        obj.part_index = int(meta["part_index"])
        obj.op_index = int(meta["op_index"])
        obj.fixture_mass_kg = float(meta["fixture_mass_kg"])
        obj.trace_notes = dict(meta["trace_notes"])
        for m in meta["tools"]:
            slot = int(m["slot"])
            obj.tools[slot] = ToolThermal.from_arrays(arrays, m, f"tool.{slot}.")
        for m in meta["works"]:
            w = int(m["workpiece"])
            obj.works[w] = WorkThermal.from_arrays(arrays, m, f"work.{w}.", None, settings)
        for slot, version in meta["fractions"]:
            values = [np.ascontiguousarray(arrays[f"frac.{slot}.{name}"]).copy() for name in _FRACTION_ARRAYS]
            obj.fractions[int(slot)] = FractionTable(*values, version=int(version))
        for w in meta["published"]:
            obj.published[int(w)] = np.array(arrays[f"pub.{w}"])
        obj.spindles = {d["sid"]: SpindleThermal.from_dict(d) for d in meta["spindles"]}
        obj.axes = {d["name"]: AxisThermal.from_dict(d) for d in meta["axes"]}
        obj.spindle_pending = {str(sid): {"n": int(v["n"]), "dt": float(v["dt"]), "heat_J": float(v["heat_J"]),
                                          "omega": [float(w) for w in v["omega"]]}
                               for sid, v in sorted((meta.get("spindle_pending") or {}).items())}
        return obj
