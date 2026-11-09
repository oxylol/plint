"""The structure domain's SimState field (checkpointable: to_arrays / from_arrays).

Everything a bit-identical continuation needs: per tool slot the published ToolModes and the inputs they were
computed for (pull-out, length lost, block orientation); per workpiece the grid, the boundary conditions in force
(supports, clamped, setup), the published solution, the latest computed one (the next Ritz basis), the pending
jobs with the macro step at which they apply, the residual-stress reference of the as-supplied stock and the
counters. The assembled matrices and multigrid hierarchies are runtime objects rebuilt from this state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.bus import ToolModes
from .lathe import LatheSupports
from .schema import StructureSettings, from_dict
from .voxel import Supports, VoxelGrid
from .work import WorkSolution

FORMAT_VERSION = 1


def json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.floating):
        return json_safe(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return value


TOOL_ARRAYS = ("freq_Hz", "zeta", "shape", "station_z_m", "static_residual_m_per_N", "sections_z_m",
               "sections_modulus_m3", "tip_compliance_m_per_N")


@dataclass
class ToolStructState:
    slot: int
    modes: ToolModes = field(default_factory=ToolModes)
    omega_rad_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pullout_m: float = 0.0
    length_loss_m: float = 0.0
    R_M_T0: np.ndarray = field(default_factory=lambda: np.eye(3))
    rotating: bool = True
    notes: list = field(default_factory=list)
    builds: int = 0

    def to_arrays(self, prefix: str) -> tuple[dict, dict]:
        arrays = {prefix + n: np.asarray(getattr(self.modes, n)) for n in TOOL_ARRAYS}
        arrays[prefix + "omega"] = self.omega_rad_s
        arrays[prefix + "R_M_T0"] = self.R_M_T0
        meta = {"slot": self.slot, "pullout_m": self.pullout_m, "length_loss_m": self.length_loss_m,
                "rotating": self.rotating, "notes": list(self.notes), "builds": self.builds,
                "fit_error": float(self.modes.fit_error), "version": int(self.modes.version)}
        return arrays, meta

    @classmethod
    def from_arrays(cls, a: dict, meta: dict, prefix: str) -> "ToolStructState":
        values = {n: np.ascontiguousarray(np.array(a[prefix + n])) for n in TOOL_ARRAYS}
        modes = ToolModes(**{n: values[n] for n in TOOL_ARRAYS if n != "tip_compliance_m_per_N"})
        modes.tip_compliance_m_per_N = values["tip_compliance_m_per_N"]
        modes.fit_error = float(meta["fit_error"])
        modes.version = int(meta["version"])
        return cls(int(meta["slot"]), modes, np.array(a[prefix + "omega"]), float(meta["pullout_m"]),
                   float(meta["length_loss_m"]), np.array(a[prefix + "R_M_T0"]), bool(meta["rotating"]),
                   list(meta["notes"]), int(meta["builds"]))


@dataclass
class WorkStructState:
    workpiece: int
    grid: VoxelGrid | None = None
    kind: str = ""
    clamped: bool = False
    op_index: int = -1
    supports_version: int = -1
    supports: Supports = field(default_factory=Supports.empty)
    lathe: LatheSupports = field(default_factory=LatheSupports)
    solution: WorkSolution | None = None
    latest: WorkSolution | None = None
    pending: list = field(default_factory=list)          # [(apply_macro, WorkSolution)] in start order
    reference: np.ndarray | None = None
    reference_digest: str = ""
    modes_version: int = 0
    disp_version: int = 0
    counters: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    unload: list = field(default_factory=list)
    fixture: dict = field(default_factory=dict)          # C3, C6, centroid, point (arrays)
    stock_fill: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))   # as-supplied fractions
    stock_lo: np.ndarray = field(default_factory=lambda: np.zeros(3))
    stock_hi: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def to_arrays(self, prefix: str) -> tuple[dict, dict]:
        arrays = dict(self.supports.to_arrays(prefix + "sup."))
        arrays[prefix + "stock_fill"] = self.stock_fill
        arrays[prefix + "stock_lo"] = self.stock_lo
        arrays[prefix + "stock_hi"] = self.stock_hi
        meta = {"workpiece": self.workpiece, "grid": self.grid.to_dict() if self.grid is not None else None,
                "kind": self.kind, "clamped": self.clamped, "op_index": self.op_index,
                "supports_version": self.supports_version, "lathe": json_safe(self.lathe.to_dict()),
                "reference_digest": self.reference_digest, "modes_version": self.modes_version,
                "disp_version": self.disp_version, "counters": json_safe(self.counters),
                "history": json_safe(self.history), "unload": json_safe(self.unload),
                "fixture": sorted(self.fixture), "has_solution": self.solution is not None,
                "latest_is_solution": self.latest is self.solution, "has_latest": self.latest is not None,
                "pending": [], "has_reference": self.reference is not None}
        if self.solution is not None:
            a, m = self.solution.to_arrays(prefix + "sol.")
            arrays.update(a)
            meta["solution"] = m
        if self.latest is not None and self.latest is not self.solution:
            a, m = self.latest.to_arrays(prefix + "lat.")
            arrays.update(a)
            meta["latest"] = m
        for i, (apply_macro, sol) in enumerate(self.pending):
            is_latest = sol is self.latest
            if is_latest and self.latest is not self.solution:
                meta["pending"].append({"apply_macro": int(apply_macro), "latest": True})
                continue
            a, m = sol.to_arrays(prefix + f"p{i}.")
            arrays.update(a)
            meta["pending"].append({"apply_macro": int(apply_macro), "latest": False, "meta": m})
        if self.reference is not None:
            arrays[prefix + "ref"] = self.reference
        for k, v in sorted(self.fixture.items()):
            arrays[prefix + "fx." + k] = np.asarray(v, dtype=float)
        return arrays, meta

    @classmethod
    def from_arrays(cls, a: dict, meta: dict, prefix: str) -> "WorkStructState":
        ws = cls(int(meta["workpiece"]))
        ws.grid = VoxelGrid.from_dict(meta["grid"]) if meta["grid"] is not None else None
        ws.kind = str(meta["kind"])
        ws.clamped = bool(meta["clamped"])
        ws.op_index = int(meta["op_index"])
        ws.supports_version = int(meta["supports_version"])
        ws.supports = Supports.from_arrays(a, prefix + "sup.")
        ws.stock_fill = np.array(a[prefix + "stock_fill"], dtype=float)
        ws.stock_lo = np.array(a[prefix + "stock_lo"], dtype=float)
        ws.stock_hi = np.array(a[prefix + "stock_hi"], dtype=float)
        lathe = meta["lathe"] or {}
        ws.lathe = LatheSupports(lathe.get("chuck"), lathe.get("tailstock"), list(lathe.get("steady") or []))
        ws.reference_digest = str(meta["reference_digest"])
        ws.modes_version = int(meta["modes_version"])
        ws.disp_version = int(meta["disp_version"])
        ws.counters = dict(meta["counters"])
        ws.history = list(meta["history"])
        ws.unload = list(meta["unload"])
        if meta["has_solution"]:
            ws.solution = WorkSolution.from_arrays(a, meta["solution"], prefix + "sol.")
        if meta["has_latest"]:
            ws.latest = ws.solution if meta["latest_is_solution"] else WorkSolution.from_arrays(a, meta["latest"],
                                                                                               prefix + "lat.")
        for i, entry in enumerate(meta["pending"]):
            sol = ws.latest if entry["latest"] else WorkSolution.from_arrays(a, entry["meta"], prefix + f"p{i}.")
            ws.pending.append((int(entry["apply_macro"]), sol))
        if meta["has_reference"]:
            ws.reference = np.array(a[prefix + "ref"])
        for k in meta["fixture"]:
            ws.fixture[k] = np.array(a[prefix + "fx." + k])
        return ws


@dataclass
class StructureState:
    settings: StructureSettings = field(default_factory=StructureSettings)
    tools: dict = field(default_factory=dict)            # slot -> ToolStructState
    works: dict = field(default_factory=dict)            # workpiece -> WorkStructState
    version: int = 0
    values: dict = field(default_factory=dict)           # macro channel values of the last step
    notes: list = field(default_factory=list)
    form: str = ""

    def to_arrays(self) -> tuple[dict, dict]:
        arrays: dict = {}
        meta = {"format": FORMAT_VERSION, "settings": self.settings.to_dict(), "version": self.version,
                "notes": list(self.notes), "form": self.form, "tools": [], "works": [],
                "values": json_safe(self.values)}
        for slot, ts in sorted(self.tools.items()):
            a, m = ts.to_arrays(f"t{slot}.")
            arrays.update(a)
            meta["tools"].append(m)
        for w, ws in sorted(self.works.items()):
            a, m = ws.to_arrays(f"w{w}.")
            arrays.update(a)
            meta["works"].append(m)
        return arrays, meta

    @classmethod
    def from_arrays(cls, arrays: dict, meta: dict) -> "StructureState":
        st = cls(from_dict(meta["settings"]))
        st.version = int(meta["version"])
        st.notes = list(meta["notes"])
        st.form = str(meta["form"])
        st.values = {k: (math.nan if v is None else v) for k, v in dict(meta["values"]).items()}
        for m in meta["tools"]:
            slot = int(m["slot"])
            st.tools[slot] = ToolStructState.from_arrays(arrays, m, f"t{slot}.")
        for m in meta["works"]:
            w = int(m["workpiece"])
            st.works[w] = WorkStructState.from_arrays(arrays, m, f"w{w}.")
        return st


__all__ = ["StructureState", "ToolStructState", "WorkStructState", "json_safe", "FORMAT_VERSION"]
