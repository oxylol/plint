"""The cutting domain's SimState field (checkpointable: to_arrays / from_arrays).

Per tool slot (:class:`ToolCut`): the table rows (geometry, coefficients, the slow state they were last
computed for), the kernel table and tool layout in force, the table top, versions and the coupling-lag
history. Per contact (:class:`ContactCut`, key (tool slot, workpiece, path)): the surface memory, the
engaged intervals, the refill planes, the kernel state (CutState) and the staleness cut. Plus the
spindle thermal loads, totals, the findings already raised and the coefficient model's cache contexts
(maps are rebuilt from the stored contexts on resume, so a resumed run is bit-identical).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from ..core.abi import EngagedIntervals, SurfaceMemory
from .abi import CutState, CutTable, CutTool, Refill
from .coefficients import RowGeometry, Rows, RowState
from .schema import CuttingSettings

_ROW_ARRAYS = ("h_grid_m", "Kt_Pa", "Kr_Pa", "Ka_Pa", "power_frac_shear", "power_frac_rake", "interface_normal_Pa",
               "chip_ratio", "T_chip_K", "frac_shear_to_chip", "frac_rake_to_tool", "frac_flank_to_tool",
               "shear_angle_rad", "T_shear_K", "T_interface_K")
_ROW_SCALARS = ("Kte_N_per_m", "Kre_N_per_m", "Kae_N_per_m", "h_min_m", "K_sp_N_per_m3", "v_min_m_s", "mu_eff",
                "flank_sigma0_Pa", "flank_tau0_Pa", "flank_land_star_m", "edge_radius_m")


@dataclass
class ToolCut:
    slot: int
    rows_geom: list = field(default_factory=list)       # RowGeometry per table row
    elem_row: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.int32))
    rows: list = field(default_factory=list)            # Rows per table row (current coefficients)
    row_states: list = field(default_factory=list)      # RowState of the last computation (None: never)
    row_radius: list = field(default_factory=list)      # edge radius of the last computation
    table: CutTable | None = None
    tool: CutTool | None = None
    h_hi_m: float = 0.0
    version: int = 0                                    # ProcessCoefficients version
    tooling_version: int = -1
    omega_plan_rad_s: float = 0.0                       # spindle speed the low-speed rows were planned for
    lag: list = field(default_factory=list)             # last two changes of the reference coefficient
    ref_coeff: float = 0.0
    computed: int = 0                                   # row computations (cost report)


@dataclass
class ContactCut:
    key: tuple
    mem: SurfaceMemory
    iv: EngagedIntervals
    refill: Refill
    kstate: CutState
    stale_before: int = -(1 << 60)
    last_tick: int = -1
    capacity: int = 4096


@dataclass
class CuttingState:
    settings: CuttingSettings
    tools: dict = field(default_factory=dict)           # slot -> ToolCut
    contacts: dict = field(default_factory=dict)        # (slot, workpiece, path) -> ContactCut
    spindle_theta: dict = field(default_factory=dict)   # spindle id -> thermal load (1 = S1 limit)
    totals: dict = field(default_factory=dict)
    peaks: dict = field(default_factory=dict)
    raised: dict = field(default_factory=dict)          # finding key -> last tick raised
    model_meta: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    stale_before: int = -(1 << 60)                      # global cut (operation start)
    # path -> tool slot seen by the last kernel_inputs (a tool change makes the contact's memory stale). Kept in the
    # state, not on the domain object, so a resumed run continues bit-identically (added by core.hotloop's
    # integration, docs/models/hotloop.md "Changes in domain packages").
    last_paths: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ checkpoint
    def to_arrays(self) -> tuple[dict, dict]:
        arrays: dict = {}
        meta: dict = {"settings": dataclasses.asdict(self.settings), "tools": [], "contacts": [],
                      "spindle_theta": [[k, float(v)] for k, v in sorted(self.spindle_theta.items())],
                      "totals": {k: float(v) for k, v in sorted(self.totals.items())},
                      "peaks": {k: float(v) for k, v in sorted(self.peaks.items())},
                      "raised": [[k, int(v)] for k, v in sorted(self.raised.items())],
                      "model_meta": self.model_meta, "notes": list(self.notes), "stale_before": int(self.stale_before),
                  "last_paths": [[int(k), int(v)] for k, v in sorted(self.last_paths.items())]}
        for slot, tc in sorted(self.tools.items()):
            p = f"t{slot}."
            arrays[p + "elem_row"] = np.array(tc.elem_row)
            if tc.table is not None:
                for name in CutTable._fields:
                    arrays[p + "table." + name] = np.array(getattr(tc.table, name))
            if tc.tool is not None:
                for name in CutTool._fields:
                    arrays[p + "tool." + name] = np.array(getattr(tc.tool, name))
            rows_meta = []
            for r, rows in enumerate(tc.rows):
                for name in _ROW_ARRAYS:
                    arrays[f"{p}row{r}.{name}"] = np.array(getattr(rows, name))
                if rows.plough_edge is not None:
                    arrays[f"{p}row{r}.plough_edge"] = np.array(rows.plough_edge)
                rows_meta.append({"scalars": {k: float(getattr(rows, k)) for k in _ROW_SCALARS},
                                  "rubbing": bool(rows.rubbing), "forming": bool(rows.forming),
                                  "source": rows.source, "notes": list(rows.notes),
                                  "plough": rows.plough_edge is not None})
            meta["tools"].append({
                "slot": int(slot), "rows_geom": [dataclasses.asdict(g) for g in tc.rows_geom], "rows": rows_meta,
                "row_states": [None if s is None else s.as_list() for s in tc.row_states],
                "row_radius": [float(v) for v in tc.row_radius], "h_hi_m": float(tc.h_hi_m), "version": int(tc.version),
                "tooling_version": int(tc.tooling_version), "omega_plan_rad_s": float(tc.omega_plan_rad_s),
                "lag": [float(v) for v in tc.lag], "ref_coeff": float(tc.ref_coeff), "computed": int(tc.computed),
                "has_table": tc.table is not None, "has_tool": tc.tool is not None})
        for i, (key, cc) in enumerate(sorted(self.contacts.items())):
            p = f"c{i}."
            for group, nt in (("mem", cc.mem), ("iv", cc.iv), ("refill", cc.refill), ("ks", cc.kstate)):
                for name in nt._fields:
                    arrays[p + group + "." + name] = np.array(getattr(nt, name))
            meta["contacts"].append({"key": [int(k) for k in key], "index": i, "stale_before": int(cc.stale_before),
                                     "last_tick": int(cc.last_tick), "capacity": int(cc.capacity)})
        return arrays, meta

    @classmethod
    def from_arrays(cls, arrays: dict, meta: dict) -> "CuttingState":
        def arr(name):
            return np.ascontiguousarray(np.array(arrays[name]))

        obj = cls(settings=CuttingSettings(**meta["settings"]))
        obj.spindle_theta = {str(k): float(v) for k, v in meta["spindle_theta"]}
        obj.totals = {k: float(v) for k, v in meta["totals"].items()}
        obj.peaks = {k: float(v) for k, v in meta["peaks"].items()}
        obj.raised = {str(k): int(v) for k, v in meta["raised"]}
        obj.model_meta = dict(meta["model_meta"])
        obj.notes = list(meta["notes"])
        obj.stale_before = int(meta["stale_before"])
        obj.last_paths = {int(k): int(v) for k, v in meta.get("last_paths", [])}
        for tm in meta["tools"]:
            slot = int(tm["slot"])
            p = f"t{slot}."
            tc = ToolCut(slot=slot)
            tc.rows_geom = [RowGeometry(**g) for g in tm["rows_geom"]]
            tc.elem_row = arr(p + "elem_row")
            tc.row_states = [None if s is None else RowState.from_list(s) for s in tm["row_states"]]
            tc.row_radius = list(tm["row_radius"])
            tc.h_hi_m = float(tm["h_hi_m"])
            tc.version = int(tm["version"])
            tc.tooling_version = int(tm["tooling_version"])
            tc.omega_plan_rad_s = float(tm["omega_plan_rad_s"])
            tc.lag = list(tm["lag"])
            tc.ref_coeff = float(tm["ref_coeff"])
            tc.computed = int(tm["computed"])
            for r, rm in enumerate(tm["rows"]):
                values = {name: arr(f"{p}row{r}.{name}") for name in _ROW_ARRAYS}
                values.update(rm["scalars"])
                rows = Rows(**values, rubbing=rm["rubbing"], forming=rm["forming"], source=rm["source"],
                            notes=list(rm["notes"]))
                if rm["plough"]:
                    rows.plough_edge = arr(f"{p}row{r}.plough_edge")
                tc.rows.append(rows)
            if tm["has_table"]:
                tc.table = CutTable(*(arr(p + "table." + name) for name in CutTable._fields))
            if tm["has_tool"]:
                tc.tool = CutTool(*(arr(p + "tool." + name) for name in CutTool._fields))
            obj.tools[slot] = tc
        for cm in meta["contacts"]:
            p = f"c{cm['index']}."
            key = tuple(int(k) for k in cm["key"])
            cc = ContactCut(key=key, mem=SurfaceMemory(*(arr(p + "mem." + n) for n in SurfaceMemory._fields)),
                            iv=EngagedIntervals(*(arr(p + "iv." + n) for n in EngagedIntervals._fields)),
                            refill=Refill(*(arr(p + "refill." + n) for n in Refill._fields)),
                            kstate=CutState(*(arr(p + "ks." + n) for n in CutState._fields)),
                            stale_before=int(cm["stale_before"]), last_tick=int(cm["last_tick"]),
                            capacity=int(cm["capacity"]))
            obj.contacts[key] = cc
        return obj


__all__ = ["CuttingState", "ToolCut", "ContactCut"]
