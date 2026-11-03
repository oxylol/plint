"""CuttingDomain (architecture 5.3, 8, 19; docs/models/cutting.md "Domain").

* ``init``: settings, the work material and every tool's coefficient model (cnctwin.mechanics), the
  machine's spindle models and axis map; one ToolCut per tool slot (state.cutting).
* ``kernel_inputs(bus)``: per controller path with a tool, a :class:`ContactInputs` with every argument of
  :func:`cnctwin.cutting.kernels.cut_step` (rotating tools) or :func:`~cnctwin.cutting.kernels.turn_step`
  (non-rotating tools): the tool layout and linearised table of the coming step, the contact's memory,
  intervals and refill planes (L2 queries for stale bins), the frozen chip occupancy and packing loads,
  the axis map, fresh accumulators and event buffers, and per-step scratch (step, istep, pose, defl)
  the integration fills at every L0 step.
* ``absorb``: accumulators and events -> bus_next.cutting.contacts (CuttingContact per contact, with
  the additive fields of docs/models/cutting.md "Bus record"), ledger terms, buffer growth and replay
  requests.
* ``macro_step(prev, nxt)``: the coefficient pipeline for the next step (nxt.cutting.coefficients),
  spindle load and duty (spindle_overload findings), the coupling-lag monitor, finite checks.
* ``end_operation``: the memories become stale (new setup); ``finalize``: per-tool summary.

Bus read: prev.motion (paths: tool slot, spindle speed, pose, tip), prev.thermal.tools / contact
preheat, prev.coolant.friction_factor, prev.wear.tools, prev.chips (occupancy, packing loads),
prev.removal (contact geometry: workpiece). Bus written: cutting (contacts, coefficients).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from ..core.bus import Bus, CuttingContact, CuttingFrame
from ..core.errors import TwinError
from ..core.findings import Category, Severity
from ..units import REFERENCE_TEMP_K
from . import abi as A
from . import memory as MEM
from .channels import channels as cutting_channels
from .channels import macro_values, tick_values
from .coefficients import (LinearModel, MechanicsModel, RowGeometry, RowState, Tolerances, build_table, coeff_table,
                           flank_stress, log_grid, process_record, state_changed)
from .geometry import ToolOptions, build_cut_tool, row_plan
from .kernels import copy_element_accumulators, cut_step, turn_step
from .schema import model_params
from .schema import settings as read_settings
from .spindle import spindles_from_machine, stall_speed, thermal_step
from .state import ContactCut, CuttingState, ToolCut

DOMAIN = "cutting"
TWO_PI = 2.0 * math.pi


class ContactInputs(NamedTuple):
    """Kernel arguments of one contact for the coming macro step (see the module doc). The integration
    fills step / istep / pose / defl at every L0 step (abi.ST_*, SI_*) and calls :meth:`call` or the
    kernel directly; ``polar`` must be removal's radius view for non-rotating tools."""

    contact: int
    slot: int
    workpiece: int
    path: int
    rotating: bool
    tool: A.CutTool
    table: A.CutTable
    mem: A.SurfaceMemory
    iv: A.EngagedIntervals
    refill: A.Refill
    step: np.ndarray
    istep: np.ndarray
    pose: A.StepPose
    defl: A.Deflection
    occ: A.CutOccupancy
    pack: A.CutPacking
    axes: A.AxisMap
    st: A.CutState
    acc: A.Accum
    cacc: A.CutAccum
    out: A.StepOut
    ev: A.CutEvents
    polar: A.PolarView

    def call(self) -> None:
        if self.rotating:
            cut_step(self.tool, self.table, self.mem, self.iv, self.refill, self.step, self.istep, self.pose, self.defl,
                     self.occ, self.pack, self.axes, self.st, self.acc, self.cacc, self.out, self.ev)
        else:
            turn_step(self.tool, self.table, self.polar, self.step, self.istep, self.pose, self.defl, self.axes,
                      self.st, self.acc, self.cacc, self.out, self.ev)


class CuttingInputs(NamedTuple):
    contacts: tuple


def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    value = getattr(obj, key, None)
    if value is None:
        try:
            value = obj[key]
        except (KeyError, TypeError, IndexError):
            value = None
    return default if value is None else value


def axis_map_from_machine(machine) -> tuple[A.AxisMap, list[str]]:
    """Axis directions in M from the kinematic chains (home pose), side +1 for tool-chain axes and -1 for
    workpiece-chain axes, rotary axis points."""
    names, dirs, points, rotary, side = [], [], [], [], []
    kin = _get(machine, "kinematics")
    axes = _get(machine, "axes") or {}
    for chain, sgn in (("tool_chain", 1.0), ("workpiece_chain", -1.0)):
        for link in (_get(kin, chain) or []):
            axis = _get(link, "axis")
            if axis is None:
                continue
            d = np.asarray(_get(link, "direction", [0.0, 0.0, 1.0]), float)
            n = np.linalg.norm(d)
            names.append(str(axis))
            dirs.append(d / n if n > 0 else d)
            pt = _get(link, "axis_point_m")
            points.append(np.asarray(pt, float) if pt is not None else np.zeros(3))
            kind = _get(_get(axes, str(axis)), "type", "linear")
            rotary.append(1 if str(kind) == "rotary" else 0)
            side.append(sgn)
    if not names:
        return A.empty_axes(1), []
    return (A.AxisMap(np.ascontiguousarray(np.array(dirs)), np.ascontiguousarray(np.array(points)),
                      np.array(rotary, np.int8), np.array(side, float)), names)


class CuttingDomain:
    """Cutting mechanics: coefficient pipeline, surface memory and force kernels (module doc)."""

    name = DOMAIN

    def __init__(self, model_factory=None, surface_query=None, tool_sides: dict | None = None):
        """``model_factory(slot, geometry) -> CoefficientModel`` (default: cnctwin.mechanics per tool);
        ``surface_query``: removal's SurfaceQuery for the refill planes (memory.SurfaceQuery)."""
        self.model_factory = model_factory
        self.query = surface_query
        self.tool_sides = dict(tool_sides or {})
        self.models: dict = {}
        self.spindles: dict = {}
        self.axes = A.empty_axes(1)
        self.axis_names: list[str] = []
        self.params: dict = {}
        self.work = None
        self.replay_requested = False
        self._inputs: CuttingInputs | None = None
        self._last_paths: dict = {}

    # ================================================================== interface
    def channels(self):
        return cutting_channels()

    def init(self, ctx, state) -> None:
        project = ctx.project
        settings = read_settings(_get(project, "physics"))
        self.params = model_params()
        machine = _get(project, "machine")
        self.spindles = spindles_from_machine(machine, self.params) if machine is not None else {}
        self.axes, self.axis_names = axis_map_from_machine(machine) if machine is not None else (A.empty_axes(1), [])
        tooling = getattr(state, "tooling", None)
        if tooling is None:
            raise TwinError("cutting: the tooling domain must be initialised before cutting (state.tooling)")
        cs = CuttingState(settings=settings)
        material = _get(project, "material")
        self.rho = self._density(material)
        for slot in range(len(tooling)):
            cs.tools[slot] = ToolCut(slot=slot)
            self.models[slot] = self._model(ctx, slot, tooling.geometry(slot), material)
        for sid, sp in sorted(self.spindles.items()):
            cs.spindle_theta[sid] = 0.0
            cs.notes.extend(f"spindle {sid}: {n}" for n in sp.notes)
        state.cutting = cs

    def _density(self, material) -> float:
        physical = _get(material, "physical")
        rho = _get(physical, "density_kg_m3")
        return float(rho) if rho is not None else 7850.0

    def _model(self, ctx, slot, geometry, material):
        if self.model_factory is not None:
            return self.model_factory(slot, geometry)
        from ..mechanics import ToolSide, WorkMaterial
        from ..mechanics.materials import TOOL_PROCESS
        if self.work is None:
            self.work = WorkMaterial.from_db(material)
        side = self.tool_sides.get(slot)
        if side is None:
            side = self._tool_side(ctx, geometry)
        tol = Tolerances(speed_rel=ctx_settings(ctx).tol_speed_rel)
        return MechanicsModel(self.work, side, TOOL_PROCESS.get(geometry.type, "milling"), tol, self.params)

    @staticmethod
    def _tool_side(ctx, geometry):
        from ..mechanics import ToolSide
        spec = getattr(geometry, "spec", None)
        db = getattr(ctx, "db", None)
        if spec is None or db is None:
            return None
        try:
            tm = db.tool_material(spec.tool_material_id) if spec.tool_material_id else None
        except (KeyError, LookupError, TwinError):
            tm = None
        if tm is None:
            return None
        coat = None
        if spec.coating_id and spec.coating_id not in ("none", ""):
            try:
                coat = db.coating(spec.coating_id)
            except (KeyError, LookupError, TwinError):
                coat = None
        item = getattr(spec, "raw", None)
        try:
            return ToolSide.from_db(item if isinstance(item, dict) else None, tm, coat)
        except (ValueError, KeyError, TypeError):
            return None

    # ================================================================== coefficient pipeline
    def _speed(self, state, slot, prev) -> tuple[float, float]:
        """(|spindle omega|, contact radius for turning) of the slot from bus_prev.motion, else the own
        accumulators of the last step, else the last plan."""
        tc = state.cutting.tools[slot]
        omega = 0.0
        r_contact = 0.0
        for frame in getattr(prev.motion, "paths", []) or []:
            if int(frame.tool_slot) == slot and len(frame.spindle_omega_rad_s):
                omega = float(np.mean(np.abs(frame.spindle_omega_rad_s)))
                tip = np.asarray(frame.tip_S_m, float)
                if tip.size:
                    r_contact = float(math.hypot(tip[-1, 0], tip[-1, 1]))
        if omega <= 0.0:
            omega = tc.omega_plan_rad_s
        return omega, r_contact

    def _row_state(self, ctx, state, slot, r, geom: RowGeometry, prev, omega, r_contact, members, rotating) -> RowState:
        thermal = prev.thermal.tools.get(slot) if hasattr(prev.thermal, "tools") else None
        wear = prev.wear.tools.get(slot) if hasattr(prev.wear, "tools") else None
        speed = omega * (geom.radius_m if rotating else r_contact)
        if speed <= 0.0:
            speed = omega * max(geom.radius_m, 1e-3) if omega > 0.0 else 1.0

        def member_mean(values):
            arr = np.asarray(values, float) if values is not None else np.zeros(0)
            if arr.shape != members.shape or not members.any():
                return None
            return float(np.mean(arr[members]))

        tool_T = member_mean(getattr(thermal, "rake_temp_mean_K", None)) if thermal is not None else None
        flank_T = None
        if thermal is not None:
            q = np.asarray(getattr(thermal, "flank_temp_q_K", np.zeros((0, 0, 0))), float)
            w = np.asarray(getattr(thermal, "q_weights", np.zeros(0)), float)
            if q.ndim == 3 and q.shape[:2] == members.shape and w.size == q.shape[2] and w.sum() > 0:
                flank_T = float(np.mean((q @ w)[members] / w.sum()))
        work_T = REFERENCE_TEMP_K
        pre = getattr(prev.thermal, "contact_preheat_K", {}) or {}
        wpre = getattr(prev.thermal, "work_preheat_K", {}) or {}
        for key in sorted(state.cutting.contacts):
            if key[0] == slot:
                idx = self._contact_index(key)
                if idx in pre:
                    work_T = float(pre[idx])
                elif key[1] in wpre:
                    work_T = float(wpre[key[1]])
        friction = 1.0
        ff = getattr(prev.coolant, "friction_factor", {}) or {}
        for key in sorted(state.cutting.contacts):
            if key[0] == slot and self._contact_index(key) in ff:
                friction = float(ff[self._contact_index(key)])
        vb = member_mean(getattr(wear, "VB_m", None)) if wear is not None else None
        bue = member_mean(getattr(wear, "bue_height_m", None)) if wear is not None else None
        width = None
        ctime = None
        for c in getattr(prev.cutting, "contacts", []) or []:
            if int(c.tool_slot) == slot and getattr(c, "engaged_edge_m", 0.0) > 0.0:
                width = float(c.engaged_edge_m)
                if getattr(c, "pass_time_s", 0.0) > 0.0:
                    ctime = float(c.pass_time_s)
        return RowState(speed_m_s=float(speed), tool_temp_K=tool_T, flank_temp_K=flank_T, work_temp_K=work_T,
                        friction_factor=friction, VB_m=vb or 0.0, bue_m=bue or 0.0, contact_width_m=width,
                        contact_time_s=ctime)

    @staticmethod
    def _contact_index(key) -> int:
        return int(key[2])

    def update_tool(self, ctx, state, slot: int, prev: Bus, force: bool = False) -> bool:
        """Coefficient pipeline of one tool slot for the next step; True when its table changed."""
        cs = state.cutting
        st_ = cs.settings
        tc = cs.tools[slot]
        tooling = state.tooling
        geo = tooling.geometry(slot)
        edge = tooling.effective[slot]
        frames = tooling.frames[slot]
        tver = int(tooling.version[slot])
        omega, r_contact = self._speed(state, slot, prev)
        tol = Tolerances(speed_rel=st_.tol_speed_rel, temp_K=st_.tol_temp_K, work_temp_K=st_.tol_work_temp_K,
                         friction_rel=st_.tol_friction_rel, VB_m=st_.tol_VB_m, bue_m=st_.tol_bue_m,
                         edge_radius_m=st_.tol_edge_radius_m)
        v_min = float(self._v_min())
        replan = (tc.tooling_version != tver or not tc.rows_geom
                  or (geo.rotating and omega > 0.0 and abs(omega - tc.omega_plan_rad_s) > 0.2 * max(tc.omega_plan_rad_s, 1e-9)))
        changed = force
        if replan:
            rows_geom, elem_row = row_plan(geo, edge, omega, v_min)
            if rows_geom != tc.rows_geom:
                keep = {g: (tc.rows[i], tc.row_states[i], tc.row_radius[i]) for i, g in enumerate(tc.rows_geom)
                        if i < len(tc.rows)}
                tc.rows, tc.row_states, tc.row_radius = [], [], []
                for g in rows_geom:
                    old = keep.get(g)
                    tc.rows.append(old[0] if old else None)
                    tc.row_states.append(old[1] if old else None)
                    tc.row_radius.append(old[2] if old else 0.0)
                tc.rows_geom = rows_geom
                changed = True
            if not np.array_equal(elem_row, tc.elem_row):
                tc.elem_row = elem_row
                changed = True
            if omega > 0.0:
                tc.omega_plan_rad_s = omega
            if tc.tooling_version != tver:
                changed = True
            tc.tooling_version = tver
        valid = np.asarray(edge.elem_valid) != 0
        edge_r = np.asarray(edge.elem_edge_radius_m, float)
        h_grid = log_grid(st_.h_log_lo_m, max(st_.h_hi_m, 2e-3), st_.log_nodes)
        model = self.models[slot]
        for r, geom in enumerate(tc.rows_geom):
            members = valid & (tc.elem_row == r)
            radius = float(np.mean(edge_r[members])) if members.any() else geom.edge_radius_m
            g_now = RowGeometry(geom.kind, geom.rake_rad, geom.incl_rad, geom.clear_rad, radius, geom.forming,
                                geom.low_speed, geom.radius_m, geom.lobes, geom.diameter_m)
            rs = self._row_state(ctx, state, slot, r, g_now, prev, omega, r_contact, members, geo.rotating)
            if tc.rows[r] is None or state_changed(tc.row_states[r], rs, tc.row_radius[r], radius, tol):
                tc.rows[r] = model.rows((slot, r), g_now, rs, h_grid)
                tc.row_states[r] = rs
                tc.row_radius[r] = radius
                tc.computed += 1
                changed = True
        h_obs = cs.peaks.get(f"h_max:{slot}", 0.0)
        h_hi = max(st_.h_hi_m, st_.h_grow * h_obs)
        if h_hi > tc.h_hi_m * 1.1 or tc.h_hi_m <= 0.0:
            tc.h_hi_m = h_hi
            changed = True
        wear = prev.wear.tools.get(slot) if hasattr(prev.wear, "tools") else None
        vb = np.asarray(getattr(wear, "VB_m", np.zeros(0)), float) if wear is not None else np.zeros(0)
        if vb.shape != np.asarray(edge.elem_r_m).shape:
            vb = np.zeros(np.asarray(edge.elem_r_m).shape)
        if tc.tool is None or changed or not np.array_equal(vb.reshape(-1), tc.tool.VB_m):
            if changed or tc.table is None:
                tc.table = build_table(tc.rows, tc.h_hi_m, st_.n_lo, st_.n_hi, v_min)
            options = ToolOptions(tangent_vn=st_.tangent_vn, turn_colwell=bool(st_.turn_colwell),
                                  surface_event_rad=st_.surface_event_rad, h_cap_m=st_.h_cap_m,
                                  recovery_ratio=float(self.params["recovery_transition_ratio"]),
                                  pd_max_flank_m=float(self.params["pd_max_flank_m"]),
                                  pd_edge_land_per_radius=float(self.params["pd_edge_land_per_radius"]))
            tc.tool = build_cut_tool(geo, edge, frames, tc.rows, tc.elem_row, vb, options)
            changed = True
        if changed:
            tc.version += 1
        self._lag_monitor(ctx, state, tc, slot, prev)
        return changed

    def _v_min(self) -> float:
        from ..mechanics.contact import min_shear_speed
        return min_shear_speed()

    def record(self, state, slot) -> "object":
        """ProcessCoefficients (bus) of a slot: rows of the coefficient groups, the per-element wear-land stress."""
        tc = state.cutting.tools[slot]
        geo = state.tooling.geometry(slot)
        ng = int(np.asarray(geo.groups.kind).shape[0])
        base = tc.rows[:ng]
        vb = tc.tool.VB_m.reshape(np.asarray(tc.elem_row).shape) if tc.tool is not None else np.zeros(tc.elem_row.shape)
        flank = np.zeros(vb.shape)
        groups = np.asarray(state.tooling.effective[slot].elem_group)
        for g, rows in enumerate(base):
            mask = groups == g
            if mask.any():
                flank[mask] = flank_stress(rows, vb[mask])[0]
        return process_record(base, slot, tc.version, flank)

    def _lag_monitor(self, ctx, state, tc, slot, prev) -> None:
        """Coupling-lag monitor (architecture 5.3): two successive changes of the frozen reference
        coefficient (tangential force at the reference h of the main row) with opposite sign and ratio > 0.5."""
        if tc.table is None or tc.tool is None:
            return
        from .coefficients import table_forces
        row = int(tc.tool.scal[A.TC_RECUT_ROW])
        h_ref = max(state.cutting.peaks.get(f"h_max:{slot}", 0.0) * 0.5, 1e-5)
        value = float(table_forces(tc.table, row, h_ref)[A.COL_FT])
        if tc.ref_coeff != 0.0:
            delta = value - tc.ref_coeff
            tc.lag.append(delta)
            tc.lag = tc.lag[-2:]
            if len(tc.lag) == 2 and tc.lag[0] * tc.lag[1] < 0.0 and abs(tc.lag[0]) > 1e-6 * abs(value) \
                    and abs(tc.lag[1]) > 0.5 * abs(tc.lag[0]):
                tick = int(getattr(prev.motion, "tick", 0))
                ctx.trace.record(tick * ctx.tick_s, tick, DOMAIN, "cutting:coupling_lag",
                                 {"slot": slot, "delta_prev": tc.lag[0], "delta": tc.lag[1], "coeff": value}, [])
                key = f"lag:{slot}"
                if key not in state.cutting.raised:
                    state.cutting.raised[key] = tick
                    ctx.finding(state, DOMAIN, Category.SANITY, Severity.WARNING,
                                f"tool slot {slot}: the cutting coefficients oscillate between macro steps (staggered "
                                "thermo-mechanical coupling); compare physics.lag_macro 1 and 2", tick,
                                quantities={"delta_prev": tc.lag[0], "delta": tc.lag[1]})
        tc.ref_coeff = value

    # ================================================================== kernel inputs
    def kernel_inputs(self, ctx, state, bus: Bus) -> CuttingInputs:
        cs = state.cutting
        st_ = cs.settings
        contacts = []
        paths = getattr(bus.motion, "paths", []) or []
        n_ticks = max(int(getattr(ctx, "macro_ipo", 1)), 1)
        tick_now = int(getattr(bus.motion, "tick", 0))
        macro_ticks = int(getattr(ctx, "ipo_ticks", 1000)) * n_ticks
        for p, frame in enumerate(paths):
            slot = int(frame.tool_slot)
            if slot < 0 or slot not in cs.tools:
                self._last_paths[p] = -1
                continue
            tc = cs.tools[slot]
            if tc.table is None or tc.tool is None:
                self.update_tool(ctx, state, slot, bus, force=True)
            workpiece = int(getattr(frame, "workpiece", 0))
            key = (slot, workpiece, p)
            geo = state.tooling.geometry(slot)
            nt, ne = np.asarray(state.tooling.effective[slot].elem_r_m).shape
            nst = int(tc.tool.st_z_m.shape[0])
            n_rows = A.memory_rows(tc.tool)
            cc = cs.contacts.get(key)
            if cc is None or cc.mem.point_S_m.shape[0] != n_rows or cc.iv.count.shape[0] != nst:
                cc = ContactCut(key=key, mem=A.empty_memory(n_rows, st_.memory_bins), iv=A.empty_intervals(nst, 4),
                                refill=A.empty_refill(n_rows, st_.memory_bins), kstate=A.empty_state(nt * ne, nt, nst),
                                capacity=st_.event_capacity)
                cs.contacts[key] = cc
            if self._last_paths.get(p) != slot:
                cc.stale_before = tick_now          # tool change: the memory is refilled from the field
                self._last_paths[p] = slot
            omega = float(np.mean(np.abs(frame.spindle_omega_rad_s))) if len(frame.spindle_omega_rad_s) else \
                tc.omega_plan_rad_s
            stale = self._stale_ticks(ctx, omega, nt)
            cc.refill.valid[:] = 0
            if self.query is not None and geo.rotating:
                pose = np.asarray(frame.pose_S_T, float)
                theta = float(frame.spindle_angle_rad[-1]) if len(frame.spindle_angle_rad) else 0.0
                c, s = math.cos(theta), math.sin(theta)
                R_S_T0 = pose[:3, :3] @ np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
                pairs = MEM.candidates(tc.tool, cc.mem, cc.iv.lo_rad, cc.iv.hi_rad, cc.iv.count, tick_now + macro_ticks,
                                       stale, max(cc.stale_before, cs.stale_before), st_.refill_margin_rad,
                                       st_.max_refill_queries)
                MEM.compute_planes(self.query, workpiece, tc.tool, cc.refill, pairs, R_S_T0, pose[:3, 3], theta,
                                   4.0 * float(tc.tool.scal[A.TC_R_CUT]) + 1e-3)
            step = np.zeros(A.N_ST)
            step[A.ST_TICK_S] = ctx.tick_s
            step[A.ST_RHO_KG_M3] = self.rho
            istep = np.zeros(A.N_SI, np.int64)
            istep[A.SI_COMMIT] = 1
            istep[A.SI_CONTACT] = p
            istep[A.SI_SLOT] = slot
            istep[A.SI_STALE_TICKS] = stale
            istep[A.SI_STALE_BEFORE] = max(cc.stale_before, cs.stale_before)
            nsec = int(tc.tool.sec_z_m.shape[0])
            n_axes = max(self.axes.dir_M.shape[0], 1)
            contacts.append(ContactInputs(
                contact=p, slot=slot, workpiece=workpiece, path=p, rotating=bool(geo.rotating), tool=tc.tool,
                table=tc.table, mem=cc.mem, iv=cc.iv, refill=cc.refill, step=step, istep=istep, pose=A.identity_pose(),
                defl=A.empty_deflection(nst), occ=self._occupancy(bus), pack=self._packing(bus, slot), axes=self.axes,
                st=cc.kstate, acc=A.core_accum(n_ticks, n_axes, nt, ne), cacc=A.empty_accum(nsec),
                out=A.empty_out(nst, nsec), ev=A.empty_events(cc.capacity), polar=A.empty_polar()))
        self._inputs = CuttingInputs(tuple(contacts))
        return self._inputs

    @staticmethod
    def _stale_ticks(ctx, omega, n_teeth) -> int:
        """One revolution plus one tooth period, in base ticks (architecture 5.4)."""
        if omega <= 0.0:
            return 1 << 60
        rev = TWO_PI / omega
        return int(math.ceil((rev + rev / max(n_teeth, 1)) / ctx.tick_s))

    @staticmethod
    def _occupancy(bus) -> A.CutOccupancy:
        chips = bus.chips
        mass = np.asarray(getattr(chips, "occupancy_mass_kg", np.zeros((0, 0, 0))), float)
        cell = float(getattr(chips, "occupancy_cell_m", 0.0) or 0.0)
        if mass.ndim != 3 or mass.size == 0 or cell <= 0.0:
            return A.empty_occupancy()
        size = np.asarray(getattr(chips, "occupancy_size_m", np.zeros_like(mass)), float)
        thick = np.asarray(getattr(chips, "occupancy_thickness_m", np.zeros_like(mass)), float)
        if size.shape != mass.shape:
            size = np.zeros_like(mass)
        if thick.shape != mass.shape:
            thick = np.zeros_like(mass)
        c = np.ascontiguousarray
        return A.CutOccupancy(c(np.asarray(chips.occupancy_origin_S_m, float)), np.array([cell]), c(mass.copy()),
                              c(size.copy()), c(thick.copy()), np.zeros_like(mass))

    @staticmethod
    def _packing(bus, slot) -> A.CutPacking:
        chips = bus.chips
        torque = float((getattr(chips, "packing_torque_Nm", {}) or {}).get(slot, 0.0))
        force = np.asarray((getattr(chips, "packing_force_T0_N", {}) or {}).get(slot, np.zeros(3)), float)
        if force.shape != (3,):
            force = np.zeros(3)
        return A.CutPacking(np.array([torque]), np.ascontiguousarray(force.copy()))

    # ================================================================== absorb
    def absorb(self, ctx, state, outputs, bus_next: Bus) -> None:
        cs = state.cutting
        frame = bus_next.cutting if isinstance(bus_next.cutting, CuttingFrame) else CuttingFrame()
        frame.contacts = []
        self.replay_requested = False
        inputs = outputs if outputs is not None else self._inputs
        dt_macro = ctx.macro_s
        for ci in (inputs.contacts if inputs is not None else ()):
            ne = int(ci.tool.scal[A.TC_N_ELEM])
            copy_element_accumulators(ci.st, ci.acc, ne)
            cc = cs.contacts.get((ci.slot, ci.workpiece, ci.path))
            record = self._contact_record(ctx, state, ci, dt_macro)
            frame.contacts.append(record)
            self._ledger(state, ci)
            self._buffers(ctx, state, ci, cc)
            self._accumulators_finite(ci)
            if cc is not None:
                cc.last_tick = int(ci.istep[A.SI_TICK_END])
                ci.st.elem_acc[:] = 0.0
            h_max = float(ci.cacc.scalars[A.CA_H_MAX])
            key = f"h_max:{ci.slot}"
            if h_max > cs.peaks.get(key, 0.0):
                cs.peaks[key] = h_max
        bus_next.cutting = frame

    def _contact_record(self, ctx, state, ci: ContactInputs, dt_macro: float) -> CuttingContact:
        acc = ci.acc
        a = acc.scalars
        c = ci.cacc.scalars
        tw = acc.tick_wrench
        t_cut = float(c[A.CA_TIME]) if c[A.CA_TIME] > 0.0 else dt_macro
        rec = CuttingContact(contact=ci.contact, tool_slot=ci.slot, in_cut_s=float(a[A.ACC_IN_CUT_S]))
        rec.wrench_tool_T0_mean = tw[:, 0, :].copy()
        rec.wrench_tool_T0_peak = tw[:, 1, :].copy()
        rec.wrench_work_W_mean = tw[:, 2, :].copy()
        rec.wrench_work_W_peak = tw[:, 3, :].copy()
        rec.force_peak_M_N = np.array([c[A.CA_FPEAK_MX], c[A.CA_FPEAK_MY], c[A.CA_FPEAK_MZ]])
        rec.torque_mean_Nm = float(c[A.CA_TORQUE_INT]) / dt_macro
        rec.torque_peak_Nm = float(a[A.ACC_TORQUE_PEAK_NM])
        rec.spindle_power_mean_W = float(a[A.ACC_SHAFT_WORK_J]) / dt_macro
        rec.spindle_power_peak_W = float(a[A.ACC_POWER_PEAK_W])
        rec.feed_power_mean_W = float(a[A.ACC_FEED_WORK_J]) / dt_macro
        rec.axis_force_N = acc.tick_axis_force_N.copy()
        rec.spindle_torque_Nm = acc.tick_spindle_torque_Nm.copy()
        rec.shaft_work_J = float(a[A.ACC_SHAFT_WORK_J])
        rec.feed_work_J = float(a[A.ACC_FEED_WORK_J])
        rec.heat_shear_J = float(a[A.ACC_HEAT_SHEAR_J])
        rec.heat_rake_J = float(a[A.ACC_HEAT_RAKE_J])
        rec.heat_flank_J = float(a[A.ACC_HEAT_FLANK_J])
        # process damping (flank indentation) is dissipated at the edge-flank contact: booked with ploughing
        rec.ploughing_J = float(a[A.ACC_PLOUGHING_J] + a[A.ACC_PROCESS_DAMPING_J])
        rec.analytic_chip_volume_m3 = float(a[A.ACC_CHIP_VOLUME_M3])
        e = acc.elem
        rec.in_cut_time_s = e[A.ELEM_IN_CUT_S].copy()
        rec.rake_sliding_m = e[A.ELEM_RAKE_SLIDING_M].copy()
        rec.flank_sliding_m = e[A.ELEM_FLANK_SLIDING_M].copy()
        rec.int_sigma_v_rake = e[A.ELEM_INT_SIGMA_V_RAKE].copy()
        rec.int_sigma_v_flank = e[A.ELEM_INT_SIGMA_V_FLANK].copy()
        rec.int_h_dt = e[A.ELEM_INT_H_DT].copy()
        rec.h_max_m = e[A.ELEM_H_MAX].copy()
        rec.entries = e[A.ELEM_ENTRIES].astype(np.int32)
        ev = ci.ev
        rec.shocks = ev.shocks.rec[:int(ev.shocks.n[0])].copy()
        rec.work_heat_S = ev.deposits.rec[:int(ev.deposits.n[0])].copy()
        rec.surface_events = ev.surface.rec[:int(ev.surface.n[0])].copy()
        rec.recut = ev.recut.rec[:int(ev.recut.n[0])].copy()
        rec.section_moment_peak_Nm = ci.cacc.sections[:, 0].copy()
        rec.section_stress_peak_Pa = ci.cacc.sections[:, 1].copy()
        # additive fields (docs/models/cutting.md "Bus record")
        rec.emissions = ev.passes.rec[:int(ev.passes.n[0])].copy()
        rec.displaced_volume_m3 = float(c[A.CA_DISPLACED_M3])
        rec.forming_J = float(c[A.CA_FORMING_J])
        rec.process_damping_J = float(a[A.ACC_PROCESS_DAMPING_J])
        rec.recut_J = float(c[A.CA_RECUT_J])
        rec.packing_J = float(c[A.CA_PACKING_J])
        rec.deflection_work_J = float(c[A.CA_DEFLECTION_J])
        rec.h_max_step_m = float(c[A.CA_H_MAX])
        rec.engaged_edge_m = float(c[A.CA_EDGE_LEN_INT]) / t_cut if t_cut > 0.0 else 0.0
        entries = float(c[A.CA_ENTRIES])
        rec.pass_time_s = float(a[A.ACC_IN_CUT_S]) / entries if entries > 0.0 else 0.0
        rec.mean_speed_m_s = float(c[A.CA_V_INT]) / max(float(np.sum(e[A.ELEM_IN_CUT_S])), 1e-300)
        rec.omega_mean_rad_s = float(c[A.CA_OMEGA_INT]) / t_cut if t_cut > 0.0 else 0.0
        rec.refill_missing = int(c[A.CA_REFILL_MISSING])
        rec.table_over = int(c[A.CA_TABLE_OVER])
        rec.float_u_max_m = float(c[A.CA_FLOAT_U_MAX])
        rec.axis_names = tuple(self.axis_names)
        rec.spindle_load_fraction, rec.spindle_available_Nm = self._load_fraction(state, ci.slot, rec)
        return rec

    def _load_fraction(self, state, slot, rec) -> tuple[float, float]:
        sp = self._spindle_of(state, slot)
        if sp is None or rec.omega_mean_rad_s <= 0.0:
            return 0.0, 0.0
        avail = sp.available(rec.omega_mean_rad_s)
        return (abs(rec.torque_mean_Nm) / avail if avail > 0.0 and math.isfinite(avail) else 0.0,
                float(avail) if math.isfinite(avail) else 0.0)

    def _spindle_of(self, state, slot):
        slots = getattr(state.tools, "slots", ())
        sid = slots[slot].spindle_id if slot < len(slots) else "main"
        return self.spindles.get(sid) or (next(iter(self.spindles.values())) if self.spindles else None)

    @staticmethod
    def _ledger(state, ci) -> None:
        led = getattr(state, "ledger", None)
        if led is None:
            return
        a = ci.acc.scalars
        c = ci.cacc.scalars
        led.shaft_work_J += float(a[A.ACC_SHAFT_WORK_J])
        led.feed_work_J += float(a[A.ACC_FEED_WORK_J])
        led.heat_shear_J += float(a[A.ACC_HEAT_SHEAR_J])
        led.heat_rake_J += float(a[A.ACC_HEAT_RAKE_J])
        led.heat_flank_J += float(a[A.ACC_HEAT_FLANK_J])
        led.ploughing_J += float(a[A.ACC_PLOUGHING_J])
        led.process_damping_J += float(a[A.ACC_PROCESS_DAMPING_J])
        led.friction_J += float(c[A.CA_PACKING_J] + c[A.CA_RECUT_J])

    def _buffers(self, ctx, state, ci, cc) -> None:
        """Event-buffer overflow doubles the capacity and asks for a replay; so do refill requests (after the
        planes for them were computed with the surface query)."""
        over = any(int(buf.n[1]) for buf in ci.ev)
        if over and cc is not None:
            cc.capacity *= 2
            self.replay_requested = True
        missing = int(ci.cacc.scalars[A.CA_REFILL_MISSING])
        if missing and self.query is not None and cc is not None:
            rows = ci.ev.refill.rec[:int(ci.ev.refill.n[0])]
            if MEM.planes_for_requests(self.query, ci.workpiece, cc.refill, rows, 4.0 * float(ci.tool.scal[A.TC_R_CUT])
                                       + 1e-3):
                self.replay_requested = True
        elif missing:
            key = f"refill:{ci.slot}"
            if key not in state.cutting.raised:
                state.cutting.raised[key] = int(ci.istep[A.SI_TICK_END])
                ctx.trace.record(ci.istep[A.SI_TICK_END] * ctx.tick_s, int(ci.istep[A.SI_TICK_END]), DOMAIN,
                                 "cutting:stale_memory_without_field", {"slot": ci.slot, "bins": missing}, [])

    @staticmethod
    def _accumulators_finite(ci) -> None:
        for name, arr in (("scalars", ci.acc.scalars), ("elem", ci.acc.elem), ("tick_wrench", ci.acc.tick_wrench),
                          ("cutting scalars", ci.cacc.scalars)):
            if not np.all(np.isfinite(arr)):
                raise TwinError(f"cutting: non-finite accumulator {name} for tool slot {ci.slot} (architecture 5.5)")

    # ================================================================== macro step
    def macro_step(self, ctx, state, prev: Bus, nxt: Bus) -> None:
        cs = state.cutting
        frame = nxt.cutting if isinstance(nxt.cutting, CuttingFrame) else CuttingFrame()
        active = sorted({int(p.tool_slot) for p in getattr(prev.motion, "paths", []) or [] if int(p.tool_slot) >= 0}
                        | {int(c.tool_slot) for c in frame.contacts})
        every = max(int(cs.settings.every_n_macro), 1)
        macro = int(getattr(state.clock, "macro", 0)) if hasattr(state, "clock") else 0
        for slot in active:
            if slot not in cs.tools:
                continue
            if macro % every == 0 or cs.tools[slot].table is None:
                self.update_tool(ctx, state, slot, prev)
            frame.coefficients[slot] = self.record(state, slot)
        self._spindle_checks(ctx, state, frame, ctx.macro_s)
        nxt.cutting = frame

    def _spindle_checks(self, ctx, state, frame, dt_s) -> None:
        cs = state.cutting
        tau = float(self.params["motor_thermal_time_constant_s"])
        demand: dict = {}
        for rec in frame.contacts:
            sp = self._spindle_of(state, rec.tool_slot)
            if sp is None:
                continue
            d = demand.setdefault(sp.id, [0.0, 0.0, None])
            d[0] += abs(rec.torque_mean_Nm)
            d[1] = max(d[1], getattr(rec, "omega_mean_rad_s", 0.0))
            d[2] = rec
        for sid, sp in sorted(self.spindles.items()):
            torque, omega, rec = demand.get(sid, [0.0, 0.0, None])
            s1 = sp.s1_torque(omega) if omega > 0.0 else math.inf
            cs.spindle_theta[sid] = thermal_step(cs.spindle_theta.get(sid, 0.0), torque, s1, dt_s, tau)
            if rec is None or omega <= 0.0:
                continue
            tick = self._tick_of(state)
            avail = sp.available(omega)
            if torque > avail:
                m = float(self.params["default_force_exponent"])
                w_eq, stalls = stall_speed(sp, omega, torque, m, float(self.params["stall_speed_ratio"]))
                key = f"stall:{sid}"
                if cs.raised.get(key, -(1 << 60)) < tick - int(1.0 / ctx.tick_s):
                    cs.raised[key] = tick
                    msg = (f"spindle {sid}: cutting torque {torque:.1f} N m exceeds the available {avail:.1f} N m at "
                           f"{omega * 60 / TWO_PI:.0f} rpm: " + ("the spindle stalls" if stalls else
                                                                 f"the speed drops to {w_eq * 60 / TWO_PI:.0f} rpm"))
                    ctx.finding(state, DOMAIN, Category.SPINDLE_OVERLOAD, Severity.ERROR, msg, tick,
                                quantities={"torque_Nm": torque, "available_Nm": avail, "omega_rad_s": omega,
                                            "omega_eq_rad_s": w_eq, "stall": 1.0 if stalls else 0.0})
            if cs.spindle_theta[sid] > 1.0:
                key = f"duty:{sid}"
                if key not in cs.raised:
                    cs.raised[key] = tick
                    ctx.finding(state, DOMAIN, Category.SPINDLE_OVERLOAD, Severity.WARNING,
                                f"spindle {sid}: thermal load {cs.spindle_theta[sid]:.2f} of the continuous (S1) "
                                "rating: the drive derates or trips by duty", tick,
                                quantities={"thermal_load": cs.spindle_theta[sid], "torque_Nm": torque,
                                            "s1_torque_Nm": s1 if math.isfinite(s1) else 0.0})

    @staticmethod
    def _tick_of(state) -> int:
        clock = getattr(state, "clock", None)
        return int(getattr(clock, "tick", 0)) if clock is not None else 0

    # ================================================================== operations
    def end_operation(self, ctx, state) -> None:
        """New setup: every memory bin is stale (refilled from the field when read), passes are closed."""
        cs = state.cutting
        cs.stale_before = self._tick_of(state) + 1
        for cc in cs.contacts.values():
            MEM.invalidate(cc.mem)
            cc.iv.count[:] = 0
            ks = cc.kstate
            ks.flags[:] = 0
            ks.tick_last[:] = -1
            ks.pend[:, A.PD_BIN] = -1.0
            ks.tooth[:, A.TS_ACTIVE] = 0.0
            ks.tooth[:, A.TS_PASS_ON] = 0.0
            ks.tooth[:, A.TS_LOADING] = 0.0
            ks.misc[A.MS_FLOAT_U] = 0.0
        self._last_paths = {}

    def finalize(self, ctx, state) -> dict:
        cs = state.cutting
        tools = []
        for slot, tc in sorted(cs.tools.items()):
            tools.append({"slot": slot, "rows": len(tc.rows_geom), "row_computations": tc.computed,
                          "table_version": tc.version, "h_hi_m": tc.h_hi_m,
                          "h_max_m": cs.peaks.get(f"h_max:{slot}", 0.0),
                          "sources": sorted({r.source for r in tc.rows if r is not None}),
                          "notes": sorted({n for r in tc.rows if r is not None for n in r.notes})})
        return {"tools": tools, "spindle_thermal_load": dict(sorted(cs.spindle_theta.items())), "notes": list(cs.notes),
                "model": {str(k): (m.checkpoint() if hasattr(m, "checkpoint") else {}) for k, m in sorted(self.models.items())}}

    # ================================================================== recorder
    def channel_values(self, ctx, state, bus: Bus) -> dict:
        contacts = [None] * len(getattr(bus.motion, "paths", []) or [])
        for rec in bus.cutting.contacts:
            if 0 <= rec.contact < len(contacts):
                contacts[rec.contact] = rec
        n_ticks = max(int(getattr(ctx, "macro_ipo", 1)), 1)
        loads = []
        for rec in contacts:
            sp = self._spindle_of(state, rec.tool_slot) if rec is not None else None
            loads.append(state.cutting.spindle_theta.get(sp.id, 0.0) if sp is not None else np.nan)
        out = tick_values(contacts, n_ticks)
        out.update(macro_values(contacts, ctx.macro_s, loads))
        return out

    # ================================================================== checkpoint of the model caches
    def model_checkpoint(self, state) -> None:
        state.cutting.model_meta = {str(k): m.checkpoint() for k, m in sorted(self.models.items())
                                    if hasattr(m, "checkpoint")}

    def model_restore(self, state) -> None:
        for k, meta in (state.cutting.model_meta or {}).items():
            m = self.models.get(int(k))
            if m is not None and hasattr(m, "restore"):
                m.restore(meta)


def ctx_settings(ctx):
    return read_settings(_get(getattr(ctx, "project", None), "physics"))


__all__ = ["CuttingDomain", "ContactInputs", "CuttingInputs", "axis_map_from_machine", "LinearModel"]
