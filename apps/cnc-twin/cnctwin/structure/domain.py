"""StructureDomain: tool and workpiece structural models (docs/models/structure.md; contract: core/domain.py).

An L2 slow domain. Per macro step it reads ``prev``: holding's tools (pull-out) and parts (supports, support
stiffness, lathe supports, free), wear's tools (a broken tool's lost length), thermal's work temperature grids,
motion's paths (the orientation of non-rotating tools in M); it writes only ``nxt.structure``
(:class:`cnctwin.core.bus.StructureFeedback`): ToolModes per tool slot, WorkModes and Displacement per workpiece,
fixture compliance.

* Tools: computed at init, again at once when a slot's pull-out changes by ``pullout_refresh``, when it breaks
  (length lost) or when a non-rotating tool's block orientation changes (a few ms each).
* Workpieces: at a new setup (holding's supports_version, the operation, clamped or released) the model is built
  and solved at once (full eigen solve); every ``every_n_macro`` steps a refresh job starts (Ritz on the latest
  modes, displacement grids) and its result is applied exactly ``REFRESH_LATENCY_MACRO`` (50) steps later
  (architecture 5.1); a job whose inputs did not change since the previous one is skipped. At the end of an
  operation the released state (no clamp loads, minimal 3-2-1 support) is computed at once.

Geometry comes through :class:`cnctwin.structure.protocols.WorkGeometry` (removal's FieldGeometry); without it the
uncut stock of the project is used (recorded in the notes). Tool geometry from ``state.tooling`` (tooling's init
runs first) or the ``tools`` given to the constructor (slot -> ToolGeometry).
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from ..core.bus import Displacement, StructureFeedback, ToolModes, WorkModes
from ..core.errors import InputError
from ..units import REFERENCE_TEMP_K
from . import fixture as FX
from . import lathe as LA
from . import toolmodel as TM
from . import voxel as VX
from . import work as WK
from .abi import StructureInputs
from .channels import MAX_PARTS, MAX_SLOTS, channels as structure_channels
from .materials import work_material
from .protocols import CylinderGeometry, stock_geometry
from .schema import REFRESH_LATENCY_MACRO, settings as read_settings
from .state import StructureState, ToolStructState, WorkStructState

DOMAIN = "structure"
ORIENTATION_TOL = 1e-6


def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        value = obj.get(key, default)
    else:
        try:
            value = obj[key]
        except (KeyError, TypeError, IndexError):
            value = getattr(obj, key, default)
    return default if value is None else value


def _plain(value):
    return getattr(value, "value", value)


class StructureDomain:
    """Tool-holder-spindle modes, workpiece modes, displacement grids and fixture compliance."""

    name = DOMAIN

    def __init__(self, geometry=None, tools=None):
        self.geometry = geometry
        self.tools = tools
        self._rt: dict = {}

    # ------------------------------------------------------------------ Domain interface
    def channels(self):
        return structure_channels()

    def kernel_inputs(self, ctx, state, bus) -> StructureInputs:
        """The total displacement grid of every workpiece (published by the previous macro step)."""
        st: StructureState = state.structure
        n = max(list(st.works) + [-1]) + 1
        origin = np.zeros((max(n, 1), 3))
        cell = np.zeros(max(n, 1))
        grids = []
        for w in range(max(n, 1)):
            ws = st.works.get(w)
            d = ws.solution.displacement if (ws is not None and ws.solution is not None) else None
            if d is None or d.cell_m <= 0.0 or d.total_m.size == 0:
                grids.append(np.zeros((1, 1, 1, 3)))
                continue
            origin[w] = d.origin_S_m
            cell[w] = d.cell_m
            grids.append(np.ascontiguousarray(d.total_m))
        return StructureInputs(origin, cell, tuple(grids))

    def absorb(self, ctx, state, outputs, bus_next) -> None:
        """Structure has no accumulators in the compiled loop."""
        return None

    def init(self, ctx, state) -> None:
        project = ctx.project
        physics = _get(project, "physics")
        st = StructureState(read_settings(_get(physics, "structure")))
        stock = _get(project, "stock")
        st.form = str(_plain(_get(stock, "form", "")) or "")
        self._rt = {}
        self._attach(ctx, state, st)
        for slot in state.tools.slots:
            ts = ToolStructState(slot.slot)
            self._build_tool(ctx, state, st, ts, slot, 0.0, 0.0, np.eye(3))
            st.tools[slot.slot] = ts
        workpieces = sorted({int(_plain(_get(_get(op, "setup"), "workpiece", 0)) or 0)
                             for op in (_get(project, "operations") or [])} or {0})
        for w in workpieces:
            ws = WorkStructState(w)
            lo, hi = self._geometry().stock_bounds_S(w)
            ws.grid = VX.choose_grid(lo, hi, st.settings.max_dof, st.settings.min_cell_m)
            st.works[w] = ws
        state.structure = st

    def macro_step(self, ctx, state, prev, nxt) -> None:
        st: StructureState = state.structure
        self._attach(ctx, state, st)
        macro = int(state.clock.macro)
        tools = {}
        for slot_obj in state.tools.slots:
            ts = st.tools.get(slot_obj.slot)
            if ts is None:
                continue
            self._tool_step(ctx, state, st, ts, slot_obj, prev)
            tools[slot_obj.slot] = ts.modes
        fb = StructureFeedback(tools=tools)
        for w, ws in sorted(st.works.items()):
            self._work_step(ctx, state, st, ws, prev, macro)
            if ws.solution is not None:
                fb.work_modes[w] = ws.solution.modes
                fb.displacement[w] = ws.solution.displacement
            self._fixture(ctx, ws, prev)
            if ws.fixture:
                fb.fixture_compliance_m_per_N[w] = ws.fixture["C3"]
                fb.fixture_point_S_m[w] = ws.fixture["point"]
                fb.fixture_compliance_6_m_per_N[w] = ws.fixture["C6"]
                fb.fixture_centroid_S_m[w] = ws.fixture["centroid"]
        fb.version = st.version
        nxt.structure = fb
        self._values(st, prev)

    def end_operation(self, ctx, state) -> None:
        """The part is unclamped: the released state (no clamp loads, minimal 3-2-1 support, free thermal
        expansion and residual distortion) is computed at once and published from the next macro step."""
        st: StructureState = state.structure
        self._attach(ctx, state, st)
        op = int(state.clock.op_index)
        for w, ws in sorted(st.works.items()):
            if ws.solution is None:
                continue
            ws.clamped = False
            ws.pending = []
            ws.counters["released_op"] = op
            inputs = self._inputs(ctx, st, ws, state.bus_prev)
            sol = self._compute(ctx, state, st, ws, inputs, None, "release")
            ws.solution = ws.latest = sol
            d = sol.displacement
            ws.unload.append({"op_index": op, "part_index": int(state.clock.part_index),
                              "thermal_max_m": _max_norm(d.thermal_m), "residual_max_m": _max_norm(d.residual_m),
                              "total_max_m": _max_norm(d.total_m), "kind": sol.kind})

    def finalize(self, ctx, state) -> dict:
        st: StructureState = state.structure
        tools = []
        for slot, ts in sorted(st.tools.items()):
            tm = ts.modes
            tools.append({"slot": slot, "freq_Hz": [float(v) for v in tm.freq_Hz[:6]],
                          "zeta": [float(v) for v in tm.zeta[:6]],
                          "tip_compliance_m_per_N": [float(v) for v in np.diag(tm.tip_compliance_m_per_N)],
                          "fit_error": float(tm.fit_error), "builds": ts.builds, "pullout_m": ts.pullout_m,
                          "length_loss_m": ts.length_loss_m, "notes": list(ts.notes)})
        works = []
        for w, ws in sorted(st.works.items()):
            sol = ws.solution
            entry = {"workpiece": w, "kind": ws.kind, "grid": ws.grid.to_dict() if ws.grid else None,
                     "counters": dict(ws.counters), "unload": list(ws.unload), "history": list(ws.history[-20:])}
            if sol is not None:
                entry.update({"freq_Hz": [float(v) for v in sol.modes.freq_Hz],
                              "zeta": [float(v) for v in sol.modes.zeta], "n_wall_modes": int(sol.modes.n_wall_modes),
                              "stats": dict(sol.stats), "displacement_max_m": _max_norm(sol.displacement.total_m),
                              "clamped": bool(sol.displacement.clamped)})
            works.append(entry)
        return {"tools": tools, "works": works, "notes": list(st.notes),
                "geometry": type(self._geometry()).__name__ if self._rt else ""}

    def macro_values(self, state) -> dict:
        return dict(state.structure.values)

    def channel_values(self, ctx, state, bus=None) -> dict:
        return self.macro_values(state)

    # ------------------------------------------------------------------ runtime
    def _attach(self, ctx, state, st: StructureState) -> None:
        if "geometry" not in self._rt:
            project = ctx.project
            if self.geometry is not None:
                geometry = self.geometry
            else:
                geometry = stock_geometry(_get(project, "stock"))
                note = "workpiece geometry from the project stock (no removal query wired): the part is not cut"
                if note not in st.notes:
                    st.notes.append(note)
            self._rt["geometry"] = geometry
            self._rt["stock"] = stock_geometry(_get(project, "stock")) if _get(project, "stock") is not None \
                else geometry
            self._rt["material"] = work_material(_get(project, "material"))
            self._rt["machine"] = _get(project, "machine")
            self._rt["datum"] = FX.datum_point_S(_get(project, "stock_pose_W_S"))

    def _geometry(self):
        return self._rt["geometry"]

    def _tool_geometry(self, state, slot: int):
        src = self.tools
        if src is not None:
            if isinstance(src, dict):
                return src.get(slot)
            if hasattr(src, "geometry"):
                return src.geometry(slot)
        tooling = getattr(state, "tooling", None)
        if tooling is not None and hasattr(tooling, "geometry") and slot < len(tooling):
            return tooling.geometry(slot)
        return None

    # ------------------------------------------------------------------ tools
    def _build_tool(self, ctx, state, st, ts: ToolStructState, slot_obj, pullout, loss, R) -> None:
        geo = self._tool_geometry(state, slot_obj.slot)
        if geo is None:
            note = f"tool slot {slot_obj.slot}: no tool geometry (tooling not run): no ToolModes"
            if note not in st.notes:
                st.notes.append(note)
            return
        res = TM.tool_modes(geo, self._rt["machine"], st.settings, pullout_m=pullout, length_loss_m=loss, R_M_T0=R,
                            db=getattr(ctx, "db", None), spindle_id=slot_obj.spindle_id)
        tm = res.modes
        tm.tip_compliance_m_per_N = res.tip_compliance_m_per_N
        tm.fit_error = float(res.fit_error)
        tm.version = ts.modes.version + 1 if ts.builds else 0
        ts.modes = tm
        ts.omega_rad_s = res.omega_rad_s
        ts.pullout_m, ts.length_loss_m = float(pullout), float(loss)
        ts.R_M_T0 = np.array(R, dtype=float)
        ts.rotating = bool(geo.rotating)
        ts.notes = list(res.notes)
        ts.builds += 1
        st.version += 1
        tick = int(state.clock.tick)
        ctx.trace.record(tick * ctx.tick_s, tick, DOMAIN, "structure:tool_modes",
                         {"slot": slot_obj.slot, "f1_Hz": float(tm.freq_Hz[0]) if len(tm.freq_Hz) else 0.0,
                          "tip_compliance_m_per_N": float(tm.tip_compliance_m_per_N[0, 0]),
                          "fit_error": float(tm.fit_error), "pullout_m": float(pullout), "length_loss_m": float(loss)},
                         [])

    def _tool_step(self, ctx, state, st, ts: ToolStructState, slot_obj, prev) -> None:
        hold = prev.holding.tools.get(slot_obj.slot) if prev is not None else None
        pullout = float(getattr(hold, "pullout_m", 0.0) or 0.0) if hold is not None else ts.pullout_m
        wear = prev.wear.tools.get(slot_obj.slot) if prev is not None else None
        loss = float(wear.length_loss_m) if (wear is not None and bool(getattr(wear, "broken", False))) else \
            ts.length_loss_m
        R = ts.R_M_T0
        if not ts.rotating and prev is not None:
            for frame in prev.motion.paths:
                if int(frame.tool_slot) == int(slot_obj.slot):
                    pose = np.asarray(frame.pose_M_T, dtype=float)
                    if pose.shape == (4, 4):
                        R = pose[:3, :3]
                    break
        changed = abs(pullout - ts.pullout_m) > st.settings.pullout_refresh_m or abs(loss - ts.length_loss_m) > 1e-9 \
            or float(np.max(np.abs(np.asarray(R) - ts.R_M_T0))) > ORIENTATION_TOL
        if changed:
            self._build_tool(ctx, state, st, ts, slot_obj, pullout, loss, R)

    # ------------------------------------------------------------------ workpieces
    def _supports(self, prev, w):
        part = prev.holding.parts.get(w) if prev is not None else None
        if part is None:
            return None, False, -1, {}
        sup = list(getattr(part, "supports", []) or [])
        held = (not bool(getattr(part, "free", False))) and bool(sup)
        return sup, held, int(getattr(part, "supports_version", 0)), dict(getattr(part, "lathe", {}) or {})

    def _kind(self, st: StructureState, ws: WorkStructState, lathe: LA.LatheSupports) -> str:
        geometry = self._geometry()
        lo, hi = geometry.stock_bounds_S(ws.workpiece)
        cylinder = isinstance(self._rt["stock"], CylinderGeometry)
        if cylinder and lathe.chuck is not None:
            L, D = LA.free_length(lo, hi, lathe)
            if D > 0.0 and L / D > st.settings.lathe_slenderness:
                return "beam"
        return "voxel"

    def _inputs(self, ctx, st, ws: WorkStructState, prev) -> WK.WorkInputs:
        geometry = self._geometry()
        lo, hi = self._rt["stock"].stock_bounds_S(ws.workpiece)
        thermal = None
        th = prev.thermal if prev is not None else None
        T = th.work_temp_K.get(ws.workpiece) if th is not None else None
        if T is not None and np.size(T):
            thermal = (np.asarray(th.work_temp_origin_S_m[ws.workpiece], dtype=float),
                       float(th.work_temp_cell_m[ws.workpiece]), np.asarray(T, dtype=float))
        return WK.WorkInputs(ws.workpiece, ws.grid, self._rt["material"], st.settings, geometry, self._rt["stock"],
                             st.form, np.asarray(lo, float), np.asarray(hi, float), ws.kind or "voxel", ws.supports,
                             ws.lathe, ws.clamped, thermal, isinstance(self._rt["stock"], CylinderGeometry))

    def _input_key(self, inputs: WK.WorkInputs) -> str:
        """Identity of a job's inputs: material fractions, temperatures and boundary conditions."""
        g = inputs.grid
        h = hashlib.sha256()
        fill = np.asarray(inputs.geometry.fill_fraction(inputs.workpiece, g.origin, g.cell_m, g.shape), dtype=float)
        h.update(np.ascontiguousarray(fill).tobytes())
        if inputs.thermal is not None:
            h.update(np.ascontiguousarray(np.nan_to_num(inputs.thermal[2], nan=-1.0)).tobytes())
        h.update(WK.digest(inputs).encode())
        return h.hexdigest()[:24]

    def _compute(self, ctx, state, st, ws: WorkStructState, inputs: WK.WorkInputs, prev_sol, why: str):
        reference = None
        if inputs.kind == "voxel" and any(e["form"] == inputs.form for e in inputs.material.residual):
            dig = WK.digest(inputs)
            if ws.reference_digest != dig or ws.reference is None:
                ws.reference = WK.residual_reference(inputs)
                ws.reference_digest = dig
            reference = ws.reference
        sol = WK.compute(inputs, prev_sol, reference)
        ws.modes_version += 1
        ws.disp_version += 1
        sol.modes.version = ws.modes_version
        sol.displacement.version = ws.disp_version
        sol.stats["key"] = self._input_key(inputs)
        key = sol.stats.get("solve", "dense")
        ws.counters[key] = int(ws.counters.get(key, 0)) + 1
        tick = int(state.clock.tick)
        f1 = float(sol.modes.freq_Hz[0]) if len(sol.modes.freq_Hz) else 0.0
        ctx.trace.record(tick * ctx.tick_s, tick, DOMAIN, f"structure:work_{why}",
                         {"workpiece": ws.workpiece, "f1_Hz": f1, "dof": float(sol.stats.get("dof", 0)),
                          "ritz_residual": float(sol.stats.get("ritz_residual") or 0.0),
                          "displacement_max_m": _max_norm(sol.displacement.total_m)}, [])
        ws.history.append({"macro": int(state.clock.macro), "why": why, "solve": key, "f1_Hz": f1,
                           "dof": int(sol.stats.get("dof", 0))})
        return sol

    def _work_step(self, ctx, state, st, ws: WorkStructState, prev, macro: int) -> None:
        sup, held, sv, lathe = self._supports(prev, ws.workpiece)
        op = int(state.clock.op_index)
        if ws.counters.get("released_op") == op and ws.solution is not None:
            return
        new_setup = ws.solution is None or sv != ws.supports_version or op != ws.op_index or held != ws.clamped
        if new_setup:
            ws.supports = VX.supports_from_holding(sup) if held else VX.Supports.empty()
            ws.lathe = LA.LatheSupports.from_holding(lathe)
            ws.clamped = held
            ws.supports_version = sv
            ws.op_index = op
            ws.kind = self._kind(st, ws, ws.lathe)
            ws.pending = []
            inputs = self._inputs(ctx, st, ws, prev)
            sol = self._compute(ctx, state, st, ws, inputs, None, "setup")
            ws.solution = ws.latest = sol
            st.version += 1
            return
        due = [p for p in ws.pending if p[0] <= macro]
        if due:
            ws.pending = [p for p in ws.pending if p[0] > macro]
            ws.solution = due[-1][1]
            st.version += 1
        if macro > 0 and macro % st.settings.every_n_macro == 0:
            inputs = self._inputs(ctx, st, ws, prev)
            if self._input_key(inputs) == (ws.latest.stats.get("key") if ws.latest is not None else None):
                ws.counters["skipped"] = int(ws.counters.get("skipped", 0)) + 1
                return
            sol = self._compute(ctx, state, st, ws, inputs, ws.latest, "refresh")
            ws.pending.append((macro + REFRESH_LATENCY_MACRO, sol))
            ws.latest = sol

    def _fixture(self, ctx, ws: WorkStructState, prev) -> None:
        part = prev.holding.parts.get(ws.workpiece) if prev is not None else None
        if part is None:
            return
        K6 = np.asarray(getattr(part, "support_stiffness_S", np.zeros((6, 6))), dtype=float)
        if K6.shape != (6, 6) or not np.any(K6 != 0.0):
            ws.fixture = {}
            return
        centroid = np.asarray(getattr(part, "centroid_S_m", np.zeros(3)), dtype=float)
        C6 = FX.compliance6(K6)
        point = self._rt["datum"]
        ws.fixture = {"C6": C6, "C3": FX.point_compliance(C6, centroid, point), "centroid": centroid.copy(),
                      "point": np.asarray(point, dtype=float).copy()}

    # ------------------------------------------------------------------ channel values
    def _values(self, st: StructureState, prev) -> None:
        nan = math.nan
        slot_f1 = np.full(MAX_SLOTS, nan)
        for slot, ts in st.tools.items():
            if slot < MAX_SLOTS and len(ts.modes.freq_Hz):
                slot_f1[slot] = float(ts.modes.freq_Hz[0])
        active = None
        if prev is not None:
            for frame in prev.motion.paths:
                if int(frame.tool_slot) >= 0 and int(frame.tool_slot) in st.tools:
                    active = st.tools[int(frame.tool_slot)]
                    break
        values = {"structure.tool_f1_slot_Hz": slot_f1,
                  "structure.tool_f1_Hz": float(active.modes.freq_Hz[0]) if active is not None and
                  len(active.modes.freq_Hz) else nan,
                  "structure.tool_tip_compliance_m_per_N": float(active.modes.tip_compliance_m_per_N[0, 0])
                  if active is not None else nan,
                  "structure.tool_fit_error": float(active.modes.fit_error) if active is not None else nan}
        per = {k: np.full(MAX_PARTS, nan) for k in ("work_f1_Hz", "work_dof", "ritz_residual", "displacement_max_m",
                                                   "thermal_max_m", "clamp_max_m", "residual_max_m",
                                                   "fixture_compliance_m_per_N")}
        pending = 0
        for w, ws in st.works.items():
            pending += len(ws.pending)
            if w >= MAX_PARTS or ws.solution is None:
                continue
            sol = ws.solution
            per["work_f1_Hz"][w] = float(sol.modes.freq_Hz[0]) if len(sol.modes.freq_Hz) else nan
            per["work_dof"][w] = float(sol.stats.get("dof", nan))
            r = sol.stats.get("ritz_residual")
            per["ritz_residual"][w] = float(r) if r is not None else nan
            d = sol.displacement
            per["displacement_max_m"][w] = _max_norm(d.total_m)
            per["thermal_max_m"][w] = _max_norm(d.thermal_m)
            per["clamp_max_m"][w] = _max_norm(d.clamp_m)
            per["residual_max_m"][w] = _max_norm(d.residual_m)
            if ws.fixture:
                per["fixture_compliance_m_per_N"][w] = float(np.max(np.linalg.eigvalsh(ws.fixture["C3"])))
        for k, v in per.items():
            values[f"structure.{k}"] = v
        values["structure.jobs_pending"] = float(pending)
        st.values = values


def _max_norm(grid) -> float:
    a = np.asarray(grid, dtype=float)
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.max(np.sum(a * a, axis=-1))))


__all__ = ["StructureDomain", "DOMAIN"]
