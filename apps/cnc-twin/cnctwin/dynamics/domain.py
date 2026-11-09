"""The dynamics domain (docs/models/dynamics.md section 6): machine dynamics and vibration.

One **system** per controller path with an active tool: the tool's modes (structure, else the spindle / turret
modes carrying a rigid tool), the machine's frame, table and rotary-axis modes and the workpiece's modes at the
contact (structure's grids). Per macro step:

* ``kernel_inputs`` (from ``bus_prev``): builds each system's kernel arrays (DynSystem: exact first-order-hold
  discretisation at the L0 step, energy integrals, relaxation from the operating point's mean cutting stiffness,
  unbalance and base-excitation projections, shapes at the stations and the contact); carries the modal state, or
  projects it onto a new basis when the structure's modes or the tool change (energy difference booked); catches
  up, out of contact and from the motion frame, ticks the hot loop did not advance;
* the compiled loop runs cnctwin.dynamics.kernels per L0 step in contact and per L1 tick out of contact;
* ``absorb``: energy identity check (dE = W_cut + W_unbalance + W_base - D, exact to round-off), the bus record
  (DynamicsFrame), the ledger's external-work term, the vibration samples into the chatter window;
* ``macro_step``: chatter detection (cnctwin.dynamics.chatter) with findings and cause chains, stability limits at
  the operating points and lobe diagrams per tool and engagement (cnctwin.dynamics.sld), the semi-discretisation
  check of new operating points (cnctwin.dynamics.semidisc);
* ``finalize``: sld/<tool>.json per tool and a summary.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass

import numpy as np

from ..core.bus import Bus, DynamicsContact, DynamicsFrame
from ..core.errors import TwinError
from ..core.findings import Category, Severity
from ..core.rundir import write_json_atomic
from . import channels as CH
from . import discretize as DZ
from . import kernels as K
from . import modal as MD
from . import params as P
from . import semidisc as SD
from . import sld as SL
from .abi import (DA_DAMPING_J, DA_ENERGY_MAX_J, DA_N_FREE, DA_N_L0, DA_UMAX_X, DA_UMIN_X, DA_W_BASE_J, DA_W_CUT_J,
                  DA_W_CUT_QUAD_J, DA_W_UNB_J, FL_STEP_ERROR, FL_STEP_SEEN, MAX_ROTARY, MI_L0_TICKS, N_BASE, AxisBase,
                  DynSystem, reset_extremes)
from .chatter import ChatterSettings, analyse
from .operating import OperatingPoint, _arc_from_geometry, cut_model, mean_chip_thickness
from .schema import settings as read_settings
from .state import ChatterWindow, DynamicsState, SystemRecord

DOMAIN = "dynamics"
TWO_PI = 2.0 * math.pi
ENERGY_TOL = 1e-9          # relative energy identity of one step (closed forms: ~1e-13 per interval)


def _rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def _get(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        v = obj.get(name, default)
    else:
        v = getattr(obj, name, default)
    return default if v is None else v


@dataclass
class DynamicsInputs:
    """What the hot loop needs from dynamics for one macro step (frozen; the arrays are written in place)."""

    systems: dict                      # path -> DynSystem (the kernels' argument)
    grids: dict                        # path -> WorkGrid (refresh_work_shapes)
    axes: dict                         # path -> AxisBase (relative_axis_acceleration)
    contact_offset_S: dict             # path -> [3] contact point minus tool tip (S), end of the previous step
    ring_columns: dict                 # path -> first fast column of the path in Recorder.ring.data (-1: none)
    tick0: int = 0
    n_ticks: int = 0


def l0_ticks_rule(ipo_ticks: int, tick_s: float, omega_max_rad_s: float, n_teeth: int, f_max_Hz: float,
                  max_rad_per_step: float, steps_per_tooth: int, omega_dt_max: float) -> int:
    """Largest divisor of ipo_ticks with |omega| dt <= max angle per step, 2 pi f_max dt <= omega_dt_max and at least
    steps_per_tooth steps per tooth period (architecture 5.1)."""
    best = 1
    for d in range(1, int(ipo_ticks) + 1):
        if ipo_ticks % d:
            continue
        dt = d * tick_s
        if omega_max_rad_s > 0.0:
            if omega_max_rad_s * dt > max_rad_per_step + 1e-15:
                continue
            tooth = TWO_PI / (max(n_teeth, 1) * omega_max_rad_s)
            if tooth / dt < steps_per_tooth - 1e-9:
                continue
        if f_max_Hz > 0.0 and TWO_PI * f_max_Hz * dt > omega_dt_max + 1e-15:
            continue
        best = d
    return best


def axis_base(machine, axis_names) -> AxisBase:
    """AxisBase of a path: link directions (M, home), side (+1 tool chain, -1 workpiece chain), rotary flags and
    the base-input column of each rotary axis (in axis order, at most MAX_ROTARY)."""
    names = list(axis_names)
    n = max(len(names), 1)
    dir_M = np.zeros((n, 3))
    side = np.zeros(n)
    rotary = np.zeros(n, np.int8)
    column = np.full(n, -1, np.int32)
    kin = _get(machine, "kinematics")
    axes = _get(machine, "axes") or {}
    found = {}
    for chain, sgn in (("workpiece_chain", -1.0), ("tool_chain", 1.0)):
        for link in _get(kin, chain) or []:
            name = _get(link, "axis")
            if name is not None and _get(link, "direction") is not None:
                found[str(name)] = (MD._vec(_get(link, "direction")), sgn)
    col = 0
    for i, name in enumerate(names):
        if name not in found:
            continue
        d, sgn = found[name]
        dir_M[i] = d
        side[i] = sgn
        try:
            kind = _get(axes[name], "type", "linear")
        except (KeyError, TypeError):
            kind = "linear"
        if kind == "rotary":
            rotary[i] = 1
            if col < MAX_ROTARY:
                column[i] = col
                col += 1
    return AxisBase(dir_M, side, rotary, column)


class DynamicsDomain:
    name = DOMAIN

    def __init__(self) -> None:
        self.settings = None
        self.machine = None
        self.chatter = None
        self.l0_override: dict[int, int] = {}
        self.inputs: DynamicsInputs | None = None
        self.last_channels: dict = {}
        self._disc: dict = {}
        self._op: dict = {}
        self._modes_event: dict = {}
        self._dropped_J = 0.0

    # ================================================================== interface
    def channels(self):
        return CH.channels()

    def init(self, ctx, state) -> None:
        project = getattr(ctx, "project", None)
        physics = _get(project, "physics")
        self.settings = read_settings(_get(physics, "dynamics"))
        self.machine = _get(project, "machine")
        self.chatter = ChatterSettings(P.p(P.CHATTER, "chatter_index_threshold"), P.p(P.CHATTER, "min_amplitude_m"),
                                       P.p(P.CHATTER, "min_amplitude_per_feed"), P.p(P.CHATTER, "flip_growth_ratio"),
                                       P.p(P.CHATTER, "flip_min_share"))
        # the project schema declares max_deg_per_step with the SI key max_rad_per_step (architecture 5.1 defaults)
        rad = _get(physics, "max_rad_per_step")
        self.max_rad_per_step = float(rad) if rad is not None else 0.5 * math.pi / 180.0
        self.steps_per_tooth = int(_get(physics, "steps_per_tooth", 96))
        if getattr(state, "dynamics", None) is None:
            state.dynamics = DynamicsState()
        state.dynamics.meta.setdefault("caught_up_ticks", 0)
        state.dynamics.meta.setdefault("caught_up_in_contact", 0)

    def set_l0_ticks(self, path: int, l0_ticks: int) -> None:
        """The integration's L0 stride for a path's coming steps (base ticks; must divide ipo_ticks)."""
        self.l0_override[int(path)] = int(l0_ticks)

    # ------------------------------------------------------------------ kernel inputs
    def kernel_inputs(self, ctx, state, bus: Bus) -> DynamicsInputs:
        ds = state.dynamics
        tick0 = int(getattr(bus.motion, "tick", 0))
        n_ticks = int(getattr(ctx, "macro_ipo", 1))
        self._dropped_J = 0.0
        systems, grids, axes, offsets, rings = {}, {}, {}, {}, {}
        seen = set()
        for p, frame in enumerate(bus.motion.paths):
            slot = int(frame.tool_slot)
            seen.add(p)
            if slot < 0:
                self._drop(ctx, state, p, tick0, "no tool")
                continue
            built = self._build(ctx, state, bus, p, frame, slot, tick0, n_ticks)
            systems[p], grids[p], axes[p], offsets[p] = built
            rings[p] = self._ring_column(ctx, p)
        for p in sorted(set(ds.systems) - seen):
            self._drop(ctx, state, p, tick0, "path gone")
        self.inputs = DynamicsInputs(systems, grids, axes, offsets, rings, tick0, n_ticks)
        return self.inputs

    def _ring_column(self, ctx, path: int) -> int:
        rec = getattr(ctx, "recorder", None)
        if rec is None or path >= CH.MAX_PATHS:
            return -1
        try:
            first, width = rec.fast_columns(CH.FAST_VIBRATION)
        except (KeyError, AttributeError):
            return -1
        return int(first + 3 * path) if width >= 3 * (path + 1) else -1

    def _drop(self, ctx, state, path: int, tick: int, why: str) -> None:
        rec = state.dynamics.systems.pop(path, None)
        if rec is None:
            return
        energy = DZ.modal_energy(rec.x, np.array([m.omega_rad_s for m in rec.modes])) if rec.modes else 0.0
        state.dynamics.totals["basis_change_J"] = state.dynamics.totals.get("basis_change_J", 0.0) - energy
        self._dropped_J = getattr(self, "_dropped_J", 0.0) - energy
        ctx.trace.record(tick * ctx.tick_s, tick, DOMAIN, "dynamics:system_removed",
                         {"path": path, "slot": rec.slot, "energy_J": energy}, [])
        state.dynamics.windows.pop(path, None)
        state.dynamics.episodes.pop(path, None)

    def _tool(self, state, slot: int):
        tooling = getattr(state, "tooling", None)
        edge = frames = None
        rotating = True
        if tooling is not None and slot < len(getattr(tooling, "tools", [])):
            geo = tooling.tools[slot]
            rotating = bool(getattr(geo, "rotating", True))
            eff = getattr(tooling, "effective", [])
            edge = eff[slot] if slot < len(eff) and eff[slot] is not None else geo.edge
            fr = getattr(tooling, "frames", [])
            frames = fr[slot] if slot < len(fr) else getattr(geo, "frames", None)
        return edge, frames, rotating

    def _discretise(self, omega: np.ndarray, zeta: np.ndarray, dt: float):
        """Phi, G0, G1, gram_D, gram_W per mode, cached by the exact bits of (omega, zeta, dt)."""
        nm = len(omega)
        Phi = np.zeros((nm, 2, 2))
        G0 = np.zeros((nm, 2))
        G1 = np.zeros((nm, 2))
        gD = np.zeros((nm, 4, 4))
        gW = np.zeros((nm, 2, 4))
        for k in range(nm):
            key = (_bits(omega[k]), _bits(zeta[k]), _bits(dt))
            hit = self._disc.get(key)
            if hit is None:
                ph, g0, g1 = DZ.foh(float(omega[k]), float(zeta[k]), dt)
                d, w = DZ.energy_grams(float(omega[k]), float(zeta[k]), dt)
                hit = (ph, g0, g1, d, w)
                if len(self._disc) > 4096:
                    self._disc.clear()
                self._disc[key] = hit
            Phi[k], G0[k], G1[k], gD[k], gW[k] = hit
        return Phi, G0, G1, gD, gW

    def _l0_ticks(self, ctx, p: int, frame, n_teeth: int, modes, ref: int, rotating: bool) -> int:
        if p in self.l0_override:
            return self.l0_override[p]
        ipo = int(ctx.ipo_ticks)
        omega = float(np.max(np.abs(frame.spindle_omega_rad_s))) if len(frame.spindle_omega_rad_s) else 0.0
        f_max = MD.f_max_resolved_Hz(modes, ref, P.p(P.MODES, "stride_compliance_share"))
        return l0_ticks_rule(ipo, ctx.tick_s, omega, n_teeth if rotating else 1, f_max, self.max_rad_per_step,
                             self.steps_per_tooth, P.p(P.MODES, "stride_omega_dt"))

    def _contact(self, bus: Bus, path: int, slot: int):
        for c in bus.removal.contacts:
            if int(c.tool_slot) == slot and int(c.path) == path and bool(c.geometry.in_cut):
                return c
        return None

    def _build(self, ctx, state, bus: Bus, p: int, frame, slot: int, tick0: int, n_ticks: int):
        st = self.settings
        ds = state.dynamics
        edge, frames, rotating = self._tool(state, slot)
        station_z = np.asarray(edge.station_z_m, float) if edge is not None else np.zeros(1)
        nst = max(len(station_z), 1)
        n_teeth = len(np.asarray(edge.tooth_phase_rad)) if edge is not None else 1
        theta_end = float(frame.spindle_angle_rad[-1]) if len(frame.spindle_angle_rad) else 0.0
        undo = _rz(-theta_end) if rotating else np.eye(3)
        R_M_T0 = np.asarray(frame.pose_M_T, float)[:3, :3] @ undo
        R_S_T0 = np.asarray(frame.pose_S_T, float)[:3, :3] @ undo
        tip_M = np.asarray(frame.pose_M_T, float)[:3, 3]
        tip_S = np.asarray(frame.pose_S_T, float)[:3, 3]
        contact = self._contact(bus, p, slot)
        # the contact point: removal's centroid of the engaged region for a rotating tool; a turning tool cuts at
        # its nose (removal's turning contacts carry no centroid: point_S_m is zero, docs/models/loop.md 15)
        point_S = np.asarray(contact.geometry.point_S_m, float) if (contact is not None and rotating) else \
            tip_S.copy()
        station_M = tip_M[None, :] + (R_M_T0 @ np.stack([np.zeros(nst), np.zeros(nst), station_z]).reshape(3, nst)).T
        names = list(frame.axis_names)
        values = {n: float(v) for n, v in zip(names, np.asarray(frame.axis_actual)[-1])} \
            if np.asarray(frame.axis_actual).size else {}
        base = axis_base(self.machine, names)
        rot_cols = {names[i]: int(base.column[i]) for i in range(len(names)) if base.rotary[i]}
        # ---------------------------------------------------------------- modes
        hold = bus.holding.tools.get(slot)
        z_u = float(_get(hold, "unbalance_z_T_m", 0.0))
        tm = bus.structure.tools.get(slot)
        version = int(getattr(bus.structure, "version", 0))
        tool_modes, residual = ([], None)
        if tm is not None and len(np.asarray(_get(tm, "freq_Hz", np.zeros(0)))):
            tool_modes, residual = MD.tool_modes_from_structure(tm, station_z, z_u, version)
            structure_version = version
        else:
            tool_modes = MD.fallback_tool_modes(self.machine, frame.spindle_id, rotating, nst, R_M_T0)
            structure_version = -1
        mach = MD.machine_modes(self.machine, station_M, R_M_T0, values, rot_cols)
        wm = bus.structure.work_modes.get(int(frame.workpiece))
        fixture = bus.structure.fixture_compliance_m_per_N.get(int(frame.workpiece))
        has_work = wm is not None and len(np.asarray(_get(wm, "freq_Hz", np.zeros(0))))
        work, grid = MD.work_modes(wm if has_work else None, version, R_S_T0,
                                   None if has_work or fixture is None else np.asarray(fixture, float))
        modes = MD.order_modes(tool_modes + mach + work)
        # ---------------------------------------------------------------- reference station and operating point
        d_c = R_S_T0.T @ (point_S - tip_S)
        ref = int(np.argmin(np.abs(station_z - d_c[2]))) if contact is not None else int(np.argmin(station_z))
        op = self._operating_point(state, bus, p, frame, slot, edge, frames, rotating, R_S_T0, R_M_T0, tip_S,
                                   contact, n_teeth, tick0)
        self._op[p] = op
        K_c = op.ap_m * op.cut.mean() if (op is not None and op.cut is not None and st.relax_estimate) else None
        l0 = self._l0_ticks(ctx, p, frame, n_teeth, modes, ref, rotating)
        unb = None
        if hold is not None and rotating:
            unb = (np.asarray(_get(hold, "unbalance_static_kg_m", np.zeros(3)), float),
                   np.asarray(_get(hold, "unbalance_couple_kg_m2", np.zeros(3)), float))
        steps = max(int(math.ceil(n_ticks * int(ctx.ipo_ticks) / max(l0, 1))), 1)
        capacity = int(math.ceil(steps * st.sample_margin)) + 1
        sys, layout = MD.build_system(modes, nst=nst, ref_station=ref, residual=residual, grid=grid, point_S=point_S,
                                      l0_ticks=l0, ipo_ticks=int(ctx.ipo_ticks), tick_s=ctx.tick_s, n_ticks=n_ticks,
                                      capacity=capacity, rotating=rotating, unbalance_T=unb, K_c_T0=K_c,
                                      R_S_T0=R_S_T0, discretise=self._discretise)
        work_shape0 = sys.model.work_shape_at_contact
        # ---------------------------------------------------------------- carried state or basis change
        signature = tuple(m.key for m in modes) + (("slot", slot), ("work", int(frame.workpiece)))
        rec = ds.systems.get(p)
        nm = len(modes)
        basis_J = 0.0
        carry_terms = np.zeros(4)
        if rec is not None and (rec.slot != slot or rec.workpiece != int(frame.workpiece)):
            basis_J -= DZ.modal_energy(rec.x, np.array([m.omega_rad_s for m in rec.modes])) if rec.modes else 0.0
            ctx.trace.record(tick0 * ctx.tick_s, tick0, DOMAIN, "dynamics:tool_changed",
                             {"path": p, "slot_old": rec.slot, "slot": slot, "energy_J": -basis_J}, [])
            rec = None
        if rec is not None:
            carry_terms = self._catch_up(ctx, state, rec, frame, bus, p, slot, tick0)
            if rec.mode_signature == signature and len(rec.modes) == nm:
                self._carry(sys, rec, nm)
            else:
                old_ws = rec.work_shape if rec.work_shape.size else None
                length = st.rotation_length_m
                if edge is not None and np.asarray(edge.station_r_m).size:
                    length = max(length, float(np.max(np.asarray(edge.station_r_m))))
                x_new, e_old, e_new, tracked = MD.project_basis(
                    rec.modes, rec.x, modes, ref_station=ref, old_work_shape=old_ws, new_work_shape=work_shape0,
                    length_m=length, mac_min=P.p(P.MODES, "basis_mac_threshold"),
                    freq_tol=P.p(P.MODES, "basis_freq_tol"))
                if nm:
                    sys.state.x[:nm] = x_new
                sys.ds.node_tick[0] = rec.node_tick
                basis_J += e_new - e_old
                self._modes_event[p] = ctx.trace.record(
                    tick0 * ctx.tick_s, tick0, DOMAIN, "dynamics:basis_change",
                    {"path": p, "slot": slot, "modes_old": len(rec.modes), "modes_new": nm, "tracked": tracked,
                     "energy_old_J": e_old, "energy_new_J": e_new}, [])
        else:
            sys.ds.node_tick[0] = tick0
        if rec is None or rec.mode_signature != signature:
            freqs = {f"f{k + 1}_Hz": m.freq_Hz for k, m in enumerate(modes[:8])}
            parents = [int(self._modes_event[p])] if p in self._modes_event else []
            self._modes_event[p] = ctx.trace.record(tick0 * ctx.tick_s, tick0, DOMAIN, "dynamics:modes",
                                                    {"path": p, "slot": slot, "n_modes": nm,
                                                     "structure_version": structure_version, **freqs}, parents)
        energy0 = DZ.modal_energy(sys.state.x[:nm], np.array([m.omega_rad_s for m in modes])) if nm else 0.0
        new = SystemRecord(p, slot, int(frame.workpiece), structure_version, modes, nst, ref, l0, rotating, n_teeth,
                           sys.state.x.copy(), sys.state.f_prev.copy(), sys.ds.f_cut.copy(), sys.ds.f_cut_prev.copy(),
                           sys.ds.f_unb.copy(), sys.ds.f_base.copy(), sys.ds.st_force.copy(),
                           sys.ds.work_force_S.copy(), int(sys.ds.node_tick[0]), sys.ds.flags.copy(), energy0, basis_J,
                           signature, np.array(work_shape0, float).copy(), carry_terms)
        ds.systems[p] = new
        reset_extremes(sys.da.scal)
        offset = point_S - tip_S
        return sys, grid, base, offset

    @staticmethod
    def _carry(sys: DynSystem, rec: SystemRecord, nm: int) -> None:
        sys.state.x[:nm] = rec.x[:nm]
        sys.state.f_prev[:nm] = rec.f_prev[:nm]
        sys.ds.f_cut[:nm] = rec.f_cut[:nm]
        sys.ds.f_cut_prev[:nm] = rec.f_cut_prev[:nm]
        sys.ds.f_unb[:nm] = rec.f_unb[:nm]
        sys.ds.f_base[:nm] = rec.f_base[:nm]
        n = min(sys.ds.st_force.shape[0], rec.st_force.shape[0])
        sys.ds.st_force[:n] = rec.st_force[:n]
        sys.ds.work_force_S[:] = rec.work_force_S
        sys.ds.node_tick[0] = rec.node_tick
        sys.ds.flags[:] = rec.flags
        sys.ds.flags[FL_STEP_ERROR] = 0

    def _catch_up(self, ctx, state, rec: SystemRecord, frame, bus: Bus, p: int, slot: int, tick0: int) -> np.ndarray:
        """Propagate a system the hot loop did not advance (its node before the step start) through the previous
        step's ticks out of contact, from the motion frame (spindle angle and speed per tick, axis accelerations by
        second differences of the actual positions). Returns the interval's exact (W_unbalance, W_base, D, dE),
        which balance among themselves and are booked with the coming step."""
        terms = np.zeros(4)
        if rec.node_tick >= tick0 or not len(frame.tick):
            return terms
        nm = len(rec.modes)
        sys, _ = MD.build_system(rec.modes, nst=rec.nst, ref_station=rec.ref_station, residual=None,
                                 grid=MD.work_modes(None, 0, np.eye(3), None)[1], point_S=np.zeros(3),
                                 l0_ticks=rec.l0_ticks, ipo_ticks=int(ctx.ipo_ticks), tick_s=ctx.tick_s, n_ticks=1,
                                 capacity=1, rotating=rec.rotating, unbalance_T=self._unbalance(bus, slot, rec.rotating),
                                 K_c_T0=None, R_S_T0=np.eye(3), discretise=self._discretise)
        self._carry(sys, rec, nm)
        omega_k = np.array([m.omega_rad_s for m in rec.modes])
        e_before = DZ.modal_energy(sys.state.x[:nm], omega_k) if nm else 0.0
        ticks = np.asarray(frame.tick, np.int64)
        angle = np.asarray(frame.spindle_angle_rad, float) + TWO_PI * np.asarray(frame.spindle_rev, float)
        omega = np.asarray(frame.spindle_omega_rad_s, float)
        acc = self._axis_acc(ctx, frame)
        base = axis_base(self.machine, frame.axis_names)
        bins = np.zeros((len(ticks), N_BASE))
        for i in range(len(ticks)):
            K.relative_axis_acceleration(base, np.ascontiguousarray(acc[i]), bins[i])
        caught = 0
        for i in range(len(ticks)):
            if ticks[i] <= sys.ds.node_tick[0]:
                continue
            if i > 0:
                th0, w0, b0 = angle[i - 1], omega[i - 1], bins[i - 1]
            else:
                th0 = angle[0] - omega[0] * (ticks[0] - sys.ds.node_tick[0]) * ctx.tick_s
                w0, b0 = omega[0], bins[0]
            K.modal_free(sys, float(th0), float(angle[i]), float(w0), float(omega[i]), np.ascontiguousarray(b0),
                         np.ascontiguousarray(bins[i]), int(ticks[i]))
            caught += 1
        in_contact = self._contact(bus, p, slot) is not None
        meta = state.dynamics.meta
        meta["caught_up_ticks"] = meta.get("caught_up_ticks", 0) + caught
        if in_contact:
            meta["caught_up_in_contact"] = meta.get("caught_up_in_contact", 0) + caught
        if caught:
            ctx.trace.record(tick0 * ctx.tick_s, tick0, DOMAIN, "dynamics:free_catch_up",
                             {"path": p, "slot": slot, "ticks": caught, "in_contact": float(in_contact)}, [])
        sc = sys.da.scal
        e_after = DZ.modal_energy(sys.state.x[:nm], omega_k) if nm else 0.0
        terms[:] = (float(sc[DA_W_UNB_J]), float(sc[DA_W_BASE_J]), float(sc[DA_DAMPING_J]), e_after - e_before)
        rec.x = sys.state.x.copy()
        rec.f_prev = sys.state.f_prev.copy()
        rec.f_cut = sys.ds.f_cut.copy()
        rec.f_cut_prev = sys.ds.f_cut_prev.copy()
        rec.f_unb = sys.ds.f_unb.copy()
        rec.f_base = sys.ds.f_base.copy()
        rec.st_force = sys.ds.st_force.copy()
        rec.work_force_S = sys.ds.work_force_S.copy()
        rec.node_tick = int(sys.ds.node_tick[0])
        rec.flags = sys.ds.flags.copy()
        return terms

    @staticmethod
    def _unbalance(bus: Bus, slot: int, rotating: bool):
        hold = bus.holding.tools.get(slot)
        if hold is None or not rotating:
            return None
        return (np.asarray(_get(hold, "unbalance_static_kg_m", np.zeros(3)), float),
                np.asarray(_get(hold, "unbalance_couple_kg_m2", np.zeros(3)), float))

    @staticmethod
    def _axis_acc(ctx, frame) -> np.ndarray:
        """Axis accelerations per tick by second differences of the actual positions (one-sided at the ends)."""
        x = np.asarray(frame.axis_actual, float)
        n = x.shape[0]
        if n < 3 or x.ndim != 2:
            return np.zeros((max(n, 1), max(x.shape[1] if x.ndim == 2 else 1, 1)))
        dt = ctx.ipo_ticks * ctx.tick_s
        a = np.zeros_like(x)
        a[1:-1] = (x[2:] - 2.0 * x[1:-1] + x[:-2]) / (dt * dt)
        a[0] = a[1]
        a[-1] = a[-2]
        return a

    # ------------------------------------------------------------------ operating point
    def _operating_point(self, state, bus: Bus, p, frame, slot, edge, frames, rotating, R_S_T0, R_M_T0, tip_S,
                         contact, n_teeth, tick) -> OperatingPoint | None:
        coeffs = bus.cutting.coefficients.get(slot)
        if coeffs is None or edge is None or contact is None or not len(frame.spindle_omega_rad_s):
            return None
        g = contact.geometry
        speed = float(np.mean(np.abs(frame.spindle_omega_rad_s)))
        if speed <= 0.0:
            return None
        omega_signed = float(np.mean(frame.spindle_omega_rad_s))
        spin = 1.0 if omega_signed > 0.0 else -1.0
        v_T0 = R_M_T0.T @ np.asarray(frame.feed_vel_M_m_s, float)[-1] if len(frame.feed_vel_M_m_s) else np.zeros(3)
        cut_rec = None
        for c in bus.cutting.contacts:
            if int(c.tool_slot) == slot:
                cut_rec = c
        nt, ne = np.asarray(edge.elem_r_m).shape
        in_cut = np.asarray(cut_rec.in_cut_time_s, float) if cut_rec is not None else np.zeros((0, 0))
        ap = float(g.ap_m)
        if in_cut.shape == (nt, ne) and np.any(in_cut > 0.0):
            engaged = in_cut > 0.0
            if ap <= 0.0 and not rotating:
                # removal's turning contacts carry no depth (docs/models/loop.md 15): the engaged chip width; the
                # linearised cut's stiffness (the relaxation of the step's predictor-corrector) does not depend on
                # it, only the reported depth of cut does
                valid = np.asarray(edge.elem_valid) != 0
                ap = float(np.sum(np.asarray(edge.elem_db_m, float)[engaged & valid]))
        elif ap > 0.0:
            d = R_S_T0.T @ (np.asarray(g.point_S_m, float) - tip_S)
            z = np.asarray(edge.elem_z_m, float)
            engaged = np.abs(z - d[2]) <= 0.5 * ap + 1e-12
        else:
            return None
        if ap <= 0.0:
            return None
        if rotating:
            lo, hi = _arc_from_geometry(R_S_T0, tip_S, g.point_S_m, g.engaged_angle_rad)
            fz = math.hypot(v_T0[0], v_T0[1]) * TWO_PI / (max(n_teeth, 1) * speed)
            h_mean = mean_chip_thickness(fz, math.atan2(v_T0[1], v_T0[0]), lo, hi)
            work_r = 0.0
        else:
            lo, hi = 0.0, TWO_PI
            fz = float(np.linalg.norm(v_T0)) * TWO_PI / speed
            vn = v_T0 / np.linalg.norm(v_T0) if np.linalg.norm(v_T0) > 0.0 else np.zeros(3)
            if frames is not None and np.any(engaged):
                proj = np.abs(np.einsum("ijk,k->ij", np.asarray(frames.normal, float), vn))[engaged]
                h_mean = float(fz * np.mean(proj))
            else:
                h_mean = fz
            tw = getattr(frame, "tip_W0_m", None)
            work_r = float(math.hypot(tw[-1][0], tw[-1][1])) if tw is not None and len(tw) else \
                float(math.hypot(g.point_S_m[0], g.point_S_m[1]))
        if h_mean <= 0.0:
            h_mean = float(np.asarray(coeffs.h_grid_m)[0]) if len(np.asarray(coeffs.h_grid_m)) else 1e-6
        wear = bus.wear.tools.get(slot)
        VB = np.asarray(_get(wear, "VB_m", np.zeros((0, 0))), float) if wear is not None else None
        cut = cut_model(edge, frames, coeffs, engaged=engaged, ap_m=ap, h_mean_m=h_mean, arc=(lo, hi),
                        rotating=rotating, spin=spin, K_sp_scale=np.asarray(coeffs.K_sp_N_per_m3, float),
                        land_per_radius=P.p(P.PROCESS_DAMPING, "edge_land_per_radius"), VB_m=VB,
                        work_radius_m=work_r)
        tol = max(self.settings.engagement_tol, 1e-9)
        sig = (slot, bool(rotating), round(float(hi - lo) / (TWO_PI * tol)),
               round(math.log(max(h_mean, 1e-12)) / tol), int(getattr(coeffs, "version", 0)),
               int(getattr(bus.structure, "version", 0)), int(np.count_nonzero(engaged)))
        return OperatingPoint(p, slot, int(tick), speed, ap, float(g.ae_m), lo, hi, fz, h_mean,
                              max(n_teeth, 1), rotating, spin, sig, cut)

    # ------------------------------------------------------------------ absorb
    def absorb(self, ctx, state, outputs, bus_next: Bus) -> None:
        inputs = self.inputs
        ds = state.dynamics
        frame = DynamicsFrame()
        if inputs is None:
            bus_next.dynamics = frame
            return
        st = self.settings
        tot = ds.totals
        ext_total = 0.0
        frame.basis_change_J += getattr(self, "_dropped_J", 0.0)
        self._dropped_J = 0.0
        for p, sys in sorted(inputs.systems.items()):
            rec = ds.systems[p]
            nm = len(rec.modes)
            if sys.ds.flags[FL_STEP_ERROR]:
                raise TwinError(f"dynamics: path {p} was stepped by {int(sys.ds.flags[FL_STEP_SEEN])} base ticks "
                                f"where its model expects {int(sys.dm.iscal[MI_L0_TICKS])} (L0) or a positive free "
                                f"interval: the integration must call set_l0_ticks before kernel_inputs")
            sc = sys.da.scal
            w_cut, w_unb, w_base, dis = (float(sc[DA_W_CUT_J]), float(sc[DA_W_UNB_J]), float(sc[DA_W_BASE_J]),
                                         float(sc[DA_DAMPING_J]))
            omega = np.array([m.omega_rad_s for m in rec.modes])
            e_end = DZ.modal_energy(sys.state.x[:nm], omega) if nm else 0.0
            e0 = rec.energy_start_J
            dE = e_end - e0
            scale = abs(e0) + abs(e_end) + abs(w_cut) + abs(w_unb) + abs(w_base) + abs(dis)
            resid = dE - (w_cut + w_unb + w_base - dis)
            if scale > 0.0 and abs(resid) > ENERGY_TOL * scale + 1e-300:
                raise TwinError(f"dynamics: energy identity of path {p} violated: dE {dE:.12g} J, work - damping "
                                f"{w_cut + w_unb + w_base - dis:.12g} J (relative {abs(resid) / scale:.3g})")
            cu = rec.carry_terms
            if np.any(cu != 0.0):
                # the catch-up of ticks the hot loop did not advance: its exact terms balance among themselves and
                # are booked with this step
                w_unb += float(cu[0])
                w_base += float(cu[1])
                dis += float(cu[2])
                dE += float(cu[3])
                rec.carry_terms = np.zeros(4)
            # carry the state back into the record
            rec.x = sys.state.x.copy()
            rec.f_prev = sys.state.f_prev.copy()
            rec.f_cut = sys.ds.f_cut.copy()
            rec.f_cut_prev = sys.ds.f_cut_prev.copy()
            rec.f_unb = sys.ds.f_unb.copy()
            rec.f_base = sys.ds.f_base.copy()
            rec.st_force = sys.ds.st_force.copy()
            rec.work_force_S = sys.ds.work_force_S.copy()
            rec.node_tick = int(sys.ds.node_tick[0])
            rec.flags = sys.ds.flags.copy()
            # vibration samples into the chatter window: only while the tool cuts (chatter is the cut's self-excited
            # vibration; samples of a step without cutting - the broad phase near the stock, the machine's frame
            # still ringing from a rapid - end the window's run instead of feeding it)
            n = int(sys.ds.samp_n[0])
            win = ds.windows.setdefault(p, ChatterWindow())
            keep = st.chatter_window_max_s
            if self._cutting(bus_next, rec.slot):
                win.append(sys.ds.samp_tick[:n], sys.ds.samp_u[:n], sys.ds.samp_w[:n], keep, ctx.tick_s)
            elif len(win.tick):
                win.clear()
            if int(sys.ds.samp_n[1]):
                ds.meta["samples_dropped"] = ds.meta.get("samples_dropped", 0) + int(sys.ds.samp_n[1])
            umin = sc[DA_UMIN_X:DA_UMIN_X + 3]
            umax = sc[DA_UMAX_X:DA_UMAX_X + 3]
            pp = float(np.max(umax - umin)) if np.all(np.isfinite(umin)) and np.all(np.isfinite(umax)) else 0.0
            peak = np.where(np.isfinite(umax), np.maximum(np.abs(umax), np.abs(umin)), 0.0)
            c = DynamicsContact(path=p, tool_slot=rec.slot, workpiece=rec.workpiece, n_modes=nm, vibration_pp_m=pp,
                                modal_energy_J=e_end, modal_energy_change_J=dE, damping_dissipation_J=dis,
                                cutting_work_J=w_cut, cutting_work_quadrature_J=float(sc[DA_W_CUT_QUAD_J]),
                                unbalance_work_J=w_unb, base_work_J=w_base, basis_change_J=rec.basis_change_J,
                                deflection_peak_T0_m=np.array(peak, float), structure_version=rec.structure_version)
            op = self._op.get(p)
            if op is not None:
                c.depth_of_cut_m = op.ap_m
                lp = ds.last_point.get(p)
                if lp is not None:
                    c.depth_limit_m = float(lp.get("depth_limit_m", math.inf))
            frame.contacts.append(c)
            frame.vibration_pp_m = max(frame.vibration_pp_m, pp)
            frame.modal_energy_change_J += dE
            frame.damping_dissipation_J += dis
            frame.basis_change_J += rec.basis_change_J
            frame.cutting_work_J += w_cut
            frame.cutting_work_quadrature_J += float(sc[DA_W_CUT_QUAD_J])
            frame.unbalance_work_J += w_unb
            frame.base_work_J += w_base
            frame.modal_energy_J += e_end
            ext_total += w_unb + w_base
            for name, value in (("cutting_work_J", w_cut), ("unbalance_work_J", w_unb), ("base_work_J", w_base),
                                ("damping_J", dis), ("basis_change_J", rec.basis_change_J)):
                tot[name] = tot.get(name, 0.0) + value
            tot["energy_max_J"] = max(tot.get("energy_max_J", 0.0), float(sc[DA_ENERGY_MAX_J]))
            tot["l0_steps"] = tot.get("l0_steps", 0.0) + float(sc[DA_N_L0])
            tot["free_ticks"] = tot.get("free_ticks", 0.0) + float(sc[DA_N_FREE])
            rec.energy_start_J = e_end
            rec.basis_change_J = 0.0
        frame.process_damping_total_J = self._process_damping_total(outputs)
        led = getattr(state, "ledger", None)
        if led is not None:
            # the ledger lists unbalance_work_J on its out side: minus the work the external forces did on the modes
            led.unbalance_work_J -= ext_total
        bus_next.dynamics = frame
        self.last_channels = self.channel_values(ctx, state)

    @staticmethod
    def _cutting(bus: Bus, slot: int) -> bool:
        """Whether the tool in ``slot`` was in cut during the step (cutting's record of the step, absorbed before
        dynamics': any element in cut)."""
        for c in getattr(bus.cutting, "contacts", []) or []:
            if int(c.tool_slot) == int(slot) and float(getattr(c, "in_cut_s", 0.0) or 0.0) > 0.0:
                return True
        return False

    @staticmethod
    def _process_damping_total(outputs) -> float:
        """Sum of ACC_PROCESS_DAMPING_J over the contact accumulators the integration passes (``outputs.accums``
        or a list of core Accum); 0 when none are given."""
        from ..core.abi import ACC_PROCESS_DAMPING_J
        accs = getattr(outputs, "accums", outputs if isinstance(outputs, (list, tuple)) else None)
        total = 0.0
        for a in accs or []:
            scal = getattr(a, "scalars", None)
            if scal is not None:
                total += float(scal[ACC_PROCESS_DAMPING_J])
        return total

    # ------------------------------------------------------------------ macro step
    def macro_step(self, ctx, state, prev: Bus, nxt: Bus) -> None:
        ds = state.dynamics
        frame = nxt.dynamics
        best = None
        for c in frame.contacts:
            p = c.path
            rec = ds.systems.get(p)
            if rec is None:
                continue
            self._stability(ctx, state, p, rec, c)
            res = self._detect(ctx, state, p, rec, c)
            if res is not None and (best is None or res.index > best[0].index):
                best = (res, c)
        if best is not None:
            res, c = best
            frame.chatter_index = float(res.index)
            frame.chatter_freq_Hz = float(res.freq_Hz) if res.chatter else 0.0
            frame.chatter_amp_m = float(res.amp_m)
            frame.chatter_onset_tick = int(c.chatter_onset_tick)
        ds.values = self._macro_values(ctx, state, frame)

    def _detect(self, ctx, state, p: int, rec: SystemRecord, c: DynamicsContact):
        st = self.settings
        ds = state.dynamics
        win = ds.windows.get(p)
        if win is None or len(win.tick) < 16:
            return None
        w = win.w[win.w > 0.0]
        if not len(w):
            return None
        f_lo, f_hi = float(np.min(w)) / TWO_PI, float(np.max(w)) / TWO_PI
        span_s = (int(win.tick[-1]) - int(win.tick[0])) * ctx.tick_s
        revs = span_s * 0.5 * (f_lo + f_hi)
        if revs < st.chatter_min_revs:
            return None
        keep = min(st.chatter_window_revs / max(0.5 * (f_lo + f_hi), 1e-9), st.chatter_window_max_s)
        start = int(np.searchsorted(win.tick, int(win.tick[-1]) - int(round(keep / ctx.tick_s)), side="left"))
        op = self._op.get(p)
        fz = op.feed_per_tooth_m if op is not None else 0.0
        res = analyse(win.tick[start:], win.u[start:], ctx.tick_s, f_lo, f_hi, rec.n_teeth if rec.rotating else 1, fz,
                      self.chatter)
        if res.chatter and self._decaying(rec, res):
            # a free vibration dying out at its structure's own damping (an entry shock, a mode excited before the
            # cut) is not the self-excited vibration of an unstable cut
            res.chatter = False
            res.onset_tick = -1
        c.chatter_index = float(res.index)
        c.chatter_freq_Hz = float(res.freq_Hz)
        c.chatter_amp_m = float(res.amp_m)
        ep = ds.episodes.setdefault(p, {"active": False, "onset_tick": -1, "clear": 0, "slot": rec.slot})
        clear_index = P.p(P.CHATTER, "clear_index")
        if res.chatter and not ep["active"]:
            ep.update(active=True, onset_tick=int(res.onset_tick), clear=0, slot=rec.slot)
            self._chatter_finding(ctx, state, p, rec, res, fz)
        elif ep["active"]:
            if res.index < clear_index:
                ep["clear"] = int(ep["clear"]) + 1
                if ep["clear"] >= st.chatter_clear_steps:
                    ep.update(active=False, clear=0)
                    ctx.trace.record(int(win.tick[-1]) * ctx.tick_s, int(win.tick[-1]), DOMAIN, "dynamics:chatter_end",
                                     {"path": p, "slot": rec.slot}, [])
            else:
                ep["clear"] = 0
        c.chatter = bool(ep["active"])
        c.chatter_onset_tick = int(ep["onset_tick"]) if ep["active"] else -1
        return res

    @staticmethod
    def _decaying(rec: SystemRecord, res) -> bool:
        """The non-forced vibration decays at least at half the structural rate zeta omega of the mode nearest its
        frequency (its envelope's fitted growth rate): a transient. Chatter grows, or holds a limit cycle, because
        the cut feeds it energy at least at that rate."""
        if not rec.modes or res.freq_Hz <= 0.0:
            return False
        w_c = TWO_PI * float(res.freq_Hz)
        mode = min(rec.modes, key=lambda m: abs(m.omega_rad_s - w_c))
        return float(res.growth_per_s) < -0.5 * float(mode.zeta) * float(mode.omega_rad_s)

    def _chatter_finding(self, ctx, state, p, rec, res, fz: float) -> None:
        ds = state.dynamics
        st = self.settings
        tick = int(res.onset_tick) if res.onset_tick >= 0 else int(ds.windows[p].tick[-1])
        parents = []
        lp = ds.last_point.get(p)
        if lp is not None and lp.get("event") is not None:
            parents.append(int(lp["event"]))
        if p in self._modes_event:
            parents.append(int(self._modes_event[p]))
        q = {"chatter_freq_Hz": res.freq_Hz, "chatter_amp_m": res.amp_m, "chatter_index": res.index,
             "growth_per_s": res.growth_per_s, "path": p, "slot": rec.slot, "flip": float(res.flip),
             "feed_per_tooth_m": fz}
        ev = ctx.trace.record(tick * ctx.tick_s, tick, DOMAIN, "dynamics:chatter_onset", q, parents)
        key = f"chatter:{rec.slot}:{getattr(state.clock, 'op_index', 0)}"
        n = ds.counters.get(key, 0)
        ds.counters[key] = n + 1
        if n >= st.max_findings:
            return
        severe = fz > 0.0 and res.amp_m >= P.p(P.CHATTER, "severe_amplitude_per_feed") * fz
        sev = Severity.ERROR if severe else Severity.WARNING
        kind = "period-doubling" if res.flip else "regenerative"
        msg = (f"Chatter ({kind}) at {res.freq_Hz:.0f} Hz, amplitude {res.amp_m * 1e6:.2f} um"
               + (f" ({res.amp_m / fz:.2f} x feed per tooth)" if fz > 0.0 else "")
               + (f"; lobe diagram limit {lp['depth_limit_m'] * 1e3:.3g} mm at ap {lp['ap_m'] * 1e3:.3g} mm"
                  if lp is not None and math.isfinite(lp.get("depth_limit_m", math.inf)) else ""))
        quantities = {"chatter_freq_Hz": res.freq_Hz, "chatter_amp_m": res.amp_m, "chatter_index": res.index}
        if lp is not None and math.isfinite(lp.get("depth_limit_m", math.inf)):
            quantities["depth_limit_m"] = lp["depth_limit_m"]
            quantities["ap_m"] = lp["ap_m"]
        ctx.finding(state, DOMAIN, Category.CHATTER, sev, msg, tick, path=p, quantities=quantities, parents=[ev])

    def _stability(self, ctx, state, p: int, rec: SystemRecord, c: DynamicsContact) -> None:
        """Limit at the operating point, recorded per tool (docs/models/dynamics.md "Stability during the run").

        The run needs only the limit at the operating speed (bus depth_limit_m, the stability findings): one speed,
        cached by engagement class and quantised speed (state.dynamics.meta, checkpointed). Lobe diagrams and the
        semi-discretisation check are report artefacts (sld/<tool>.json) and an input to nothing in the loop: the
        engagement class is kept (state.dynamics.pending) and both are computed once at the end of the operation
        (:meth:`finish_stability`); chatter detection stays the time-domain index of every step."""
        op = self._op.get(p)
        ds = state.dynamics
        st = self.settings
        if op is None or op.cut is None or len(op.cut.M) == 0:
            return
        lp = ds.last_point.get(p)
        tick = op.tick
        sig = list(op.signature)
        if lp is not None and lp.get("signature") == _plain_sig(sig) and tick - int(lp["tick"]) < \
                st.sld_refresh_macro * ctx.ipo_ticks * ctx.macro_ipo and \
                abs(op.speed_rad_s - lp["speed_rad_s"]) <= st.engagement_tol * lp["speed_rad_s"] and \
                abs(op.ap_m - lp["ap_m"]) <= st.engagement_tol * lp["ap_m"]:
            c.depth_limit_m = float(lp["depth_limit_m"])
            c.depth_of_cut_m = op.ap_m
            return
        inputs = self.inputs
        sys = inputs.systems[p]
        frf = self._frf(sys, rec)
        if frf is None:
            return
        method = "mf" if st.mf_harmonics > 0 and op.cut.uniform_pitch and op.rotating else "zoa"
        lim = self._limit(ds, op, frf, method)
        margin = lim.depth_limit_m / op.ap_m if op.ap_m > 0.0 else math.inf
        cls = self._remember_class(ds, rec, op, frf, sig)
        stable = bool(margin > 1.0)
        q = {"path": p, "slot": rec.slot, "speed_rad_s": op.speed_rad_s, "ap_m": op.ap_m, "ae_m": op.ae_m,
             "depth_limit_m": lim.depth_limit_m if math.isfinite(lim.depth_limit_m) else -1.0,
             "margin": margin if math.isfinite(margin) else -1.0, "chatter_freq_Hz": lim.chatter_freq_Hz,
             "stable": float(stable)}
        parents = [int(self._modes_event[p])] if p in self._modes_event else []
        ev = ctx.trace.record(tick * ctx.tick_s, tick, DOMAIN, "dynamics:stability", q, parents)
        ds.last_point[p] = {"signature": _plain_sig(sig), "tick": tick, "speed_rad_s": op.speed_rad_s, "ap_m": op.ap_m,
                            "depth_limit_m": lim.depth_limit_m, "event": ev}
        c.depth_limit_m = float(lim.depth_limit_m)
        c.depth_of_cut_m = op.ap_m
        line = self._line(ctx, state, tick, p)
        point = {"op": line[0], "program": line[1], "line": line[2], "time_s": tick * ctx.tick_s, "tick": tick,
                 "speed_rad_s": op.speed_rad_s, "ap_m": op.ap_m, "ae_m": op.ae_m, "stable": stable,
                 "depth_limit_m": lim.depth_limit_m, "margin": margin, "chatter_freq_Hz": lim.chatter_freq_Hz,
                 "method": method + ("+pd" if len(op.cut.damping_M) else ""), "diagram": None,
                 "sd_stable": None, "path": p, "class": cls, "event": ev}
        pts = ds.points.setdefault(rec.slot, [])
        for old in pts:
            if old["line"] == point["line"] and old["op"] == point["op"] and \
                    abs(old["speed_rad_s"] - point["speed_rad_s"]) <= 0.01 * point["speed_rad_s"] and \
                    abs(old["ap_m"] - point["ap_m"]) <= 0.01 * point["ap_m"]:
                if point["margin"] < old["margin"]:
                    old.update(point)
                break
        else:
            if len(pts) < st.max_operating_points:
                pts.append(point)
        if not stable:
            self._stability_finding(ctx, state, rec.slot, point, ev, sd_agrees=False)

    def _stability_finding(self, ctx, state, slot: int, point: dict, ev, sd_agrees: bool) -> None:
        ds = state.dynamics
        st = self.settings
        key = f"stability:{slot}:{getattr(state.clock, 'op_index', 0)}"
        n = ds.counters.get(key, 0)
        ds.counters[key] = n + 1
        if n >= st.max_findings:
            return
        lim = float(point["depth_limit_m"])
        msg = (f"Lobe diagram predicts chatter: depth of cut {point['ap_m'] * 1e3:.3g} mm above the stability limit "
               f"{lim * 1e3:.3g} mm at {point['speed_rad_s'] * 60.0 / TWO_PI:.0f} rpm "
               f"(chatter frequency {point['chatter_freq_Hz']:.0f} Hz)" + ("; semi-discretisation agrees"
                                                                           if sd_agrees else ""))
        if point.get("sd_stable") is False and point["margin"] > 1.0:
            msg = (f"Semi-discretisation predicts chatter at depth of cut {point['ap_m'] * 1e3:.3g} mm, "
                   f"{point['speed_rad_s'] * 60.0 / TWO_PI:.0f} rpm (the zero-order limit {lim * 1e3:.3g} mm misses "
                   "a period-doubling lobe)")
        ctx.finding(state, DOMAIN, Category.CHATTER, Severity.WARNING, msg, int(point["tick"]), path=int(point["path"]),
                    quantities={"depth_limit_m": lim, "ap_m": point["ap_m"], "chatter_freq_Hz": point["chatter_freq_Hz"],
                                "margin": point["margin"]}, parents=[ev] if ev is not None else [])

    def _limit(self, ds, op: OperatingPoint, frf: SL.ModalFRF, method: str) -> SL.SpeedLimit:
        """Limit at the operating speed for the engagement class, cached by (class, FRF, quantised speed) in
        state.dynamics.meta["limits"] (checkpointed: a resumed run reads the same values)."""
        st = self.settings
        pd = bool(len(op.cut.damping_M))
        key = "|".join([json.dumps(_plain_sig(list(op.signature))), _frf_key(frf), method, "pd" if pd else "-",
                        str(int(round(math.log(max(op.speed_rad_s, 1e-9)) / LIMIT_SPEED_QUANTUM)))])
        cache = ds.meta.setdefault("limits", {})
        hit = cache.get(key)
        if hit is not None:
            return SL.SpeedLimit(op.speed_rad_s, float(hit[0]), float(hit[1]))
        omegas = frf.frequency_grid(st.sld_n_freq, st.sld_freq_lo_factor, st.sld_freq_hi_factor)
        if pd:
            lim, _ = SL.limit_with_process_damping(op.cut, frf, op.speed_rad_s, omegas, tol_rel=st.pd_tol,
                                                   max_iter=st.pd_max_iter, method=method,
                                                   harmonics=st.mf_harmonics)
        else:
            lim = SL.limit_at_speed(op.cut, frf, op.speed_rad_s, omegas, method=method, harmonics=st.mf_harmonics)
        if len(cache) >= LIMIT_CACHE_MAX:
            cache.pop(next(iter(cache)))
        cache[key] = [float(lim.depth_limit_m), float(lim.chatter_freq_Hz)]
        return lim

    def _remember_class(self, ds, rec: SystemRecord, op: OperatingPoint, frf: SL.ModalFRF, sig) -> str:
        """Key of the operating point's engagement class; the class (its linearised cut and FRF) is kept until the
        end of the operation, when its lobe diagram is computed (state.dynamics.pending, checkpointed)."""
        key = json.dumps(_plain_sig(sig)) + "|" + _frf_key(frf)
        classes = ds.pending.setdefault(str(rec.slot), {})
        if key not in classes and len(classes) < self.settings.sld_max_classes:
            classes[key] = {"cut": _cut_doc(op.cut), "frf": _frf_doc(frf), "signature": _plain_sig(sig),
                            "ae_m": op.ae_m, "arc_lo_rad": op.arc_lo_rad, "arc_hi_rad": op.arc_hi_rad,
                            "n_teeth": op.n_teeth, "rotating": op.rotating, "h_mean_m": op.h_mean_m,
                            "tick": op.tick, "speed_rad_s": op.speed_rad_s,
                            "modes": [{"label": m.label, "freq_Hz": m.freq_Hz, "zeta": m.zeta} for m in rec.modes]}
        return key

    def finish_stability(self, ctx, state) -> None:
        """End of an operation (or of the run): the lobe diagram of every engagement class seen (at most
        sld_max_diagrams per tool, the classes of the operating points with the smallest margins first; cached by
        class in the tool's diagram list) and the semi-discretisation check of the operating points with the
        smallest margins (at most sd_max_points per run); a point the check finds unstable raises the stability
        finding then (at the point's tick)."""
        ds = state.dynamics
        st = self.settings
        for slot_key in sorted(ds.pending):
            classes = ds.pending[slot_key]
            slot = int(slot_key)
            pts = ds.points.get(slot, [])
            worst: dict = {}
            for q in pts:
                k = q.get("class")
                if k in classes and q.get("diagram") is None:
                    m = q["margin"] if math.isfinite(q["margin"]) else math.inf
                    worst[k] = min(worst.get(k, math.inf), m)
            order = sorted(worst, key=lambda k: (worst[k], k))[:max(st.sld_max_diagrams, 0)]
            index = {}
            for k in order:
                entry = classes[k]
                index[k] = self._diagram(ctx, state, slot, entry)
            for q in pts:
                k = q.get("class")
                if k in index:
                    q["diagram"] = index[k]
            if st.sd_check:
                todo = sorted((q for q in pts if q.get("sd_stable") is None and q.get("class") in classes),
                              key=lambda q: (q["margin"] if math.isfinite(q["margin"]) else math.inf, q["tick"]))
                for q in todo:
                    n_checked = ds.meta.get("sd_checked", 0)
                    if n_checked >= st.sd_max_points:
                        break
                    entry = classes[q["class"]]
                    cut = _cut_from(entry["cut"])
                    frf = _frf_from(entry["frf"])
                    sd_frf = _dominant(frf, None, P.p(P.MODES, "stride_compliance_share"))
                    rho = SD.spectral_radius(cut, sd_frf, q["speed_rad_s"], q["ap_m"], st.sd_m, st.sd_points_per_period,
                                             process_damping=bool(len(cut.damping_M)))
                    ds.meta["sd_checked"] = n_checked + 1
                    q["sd_stable"] = bool(rho < 1.0)
                    was = bool(q["stable"])
                    q["stable"] = bool(q["margin"] > 1.0) and q["sd_stable"]
                    if was and not q["stable"]:
                        self._stability_finding(ctx, state, slot, q, q.get("event"), sd_agrees=False)
        ds.pending.clear()

    def _frf(self, sys: DynSystem, rec: SystemRecord) -> SL.ModalFRF | None:
        nm = len(rec.modes)
        if nm == 0:
            return None
        psi = np.array(sys.dm.contact_shape[:nm], float)
        ref = rec.ref_station
        static = np.zeros((3, 3))
        R = sys.model.static_residual
        if R.shape[0] >= 3 * (ref + 1):
            static += R[3 * ref:3 * ref + 3, 3 * ref:3 * ref + 3]
        Rs = sys.dm.R_S_T0
        static += Rs.T @ sys.dm.res_work_S @ Rs
        return SL.ModalFRF(np.array([m.omega_rad_s for m in rec.modes]), np.array([m.zeta for m in rec.modes]), psi,
                           static)

    def _diagram(self, ctx, state, slot: int, entry: dict) -> int:
        """Index of the tool's lobe diagram for an engagement class (computed when new; cached by class)."""
        st = self.settings
        ds = state.dynamics
        diagrams = ds.diagrams.setdefault(slot, [])
        plain = entry["signature"]
        for i, d in enumerate(diagrams):
            if d["signature"] == plain:
                return i
        cut = _cut_from(entry["cut"])
        frf = _frf_from(entry["frf"])
        w_max = self._max_speed_of(float(entry["speed_rad_s"]))
        speeds = np.linspace(st.sld_speed_lo_frac * w_max, w_max, st.sld_n_speeds)
        if cut.uniform_pitch:
            lob = SL.zoa_lobes(cut, frf, speeds, n_freq=st.sld_n_freq, n_lobes=st.sld_n_lobes,
                               lo_factor=st.sld_freq_lo_factor, hi_factor=st.sld_freq_hi_factor)
            env, freq = lob.depth_limit_m, lob.chatter_freq_Hz
            lobes = [{"k": int(l["k"]), "branch": int(l["branch"]), "speed_rad_s": l["speed_rad_s"],
                      "depth_limit_m": l["depth_limit_m"], "chatter_freq_Hz": l["chatter_freq_Hz"]}
                     for l in lob.lobes]
            method = "zoa"
        else:
            coarse = speeds[::max(1, len(speeds) // 80)]
            om = frf.frequency_grid(max(st.sld_n_freq // 4, 64), st.sld_freq_lo_factor, st.sld_freq_hi_factor)
            res = [SL.limit_at_speed(cut, frf, w, om) for w in coarse]
            speeds = coarse
            env = np.array([r.depth_limit_m for r in res])
            freq = np.array([r.chatter_freq_Hz for r in res])
            lobes = []
            method = "zoa-variable-pitch"
        mc = SL.mode_coupling_limit(cut, frf, a_max=st.mode_coupling_max_m, n_scan=40, tol_rel=1e-3)
        diagrams.append({"signature": plain, "method": method, "ae_m": entry["ae_m"], "arc_lo_rad": entry["arc_lo_rad"],
                         "arc_hi_rad": entry["arc_hi_rad"], "n_teeth": entry["n_teeth"], "rotating": entry["rotating"],
                         "h_mean_m": entry["h_mean_m"], "speed_rad_s": np.asarray(speeds, float),
                         "depth_limit_m": np.asarray(env, float), "chatter_freq_Hz": np.asarray(freq, float),
                         "lobes": lobes, "mode_coupling_limit_m": mc, "tick": entry["tick"], "modes": entry["modes"]})
        return len(diagrams) - 1

    def _max_speed(self, op: OperatingPoint) -> float:
        return self._max_speed_of(op.speed_rad_s)

    def _max_speed_of(self, speed_rad_s: float) -> float:
        best = 0.0
        for sp in _get(self.machine, "spindles") or []:
            v = _get(sp, "max_rad_s")
            if v is not None:
                best = max(best, float(v))
        return best if best > 0.0 else 2.0 * speed_rad_s

    def _line(self, ctx, state, tick: int, path: int) -> tuple:
        timeline = getattr(ctx, "timeline", None)
        for bus in (state.bus_next, state.bus_prev):
            paths = bus.motion.paths
            if path < len(paths) and len(paths[path].tick):
                fr = paths[path]
                i = int(np.searchsorted(fr.tick, tick, side="right")) - 1
                i = min(max(i, 0), len(fr.tick) - 1)
                ref = timeline.line_ref(int(fr.block_index[i])) if timeline is not None else None
                if ref is not None:
                    return (str(getattr(ref, "op", "")), str(ref.program), int(ref.line))
                return ("", "", 0)
        return ("", "", 0)

    # ------------------------------------------------------------------ channels
    def channel_values(self, ctx, state) -> dict:
        """Tick-tier columns {name: [n_ticks, MAX_PATHS]} and the macro values of the step (recorder input)."""
        inputs = self.inputs
        n = max(int(getattr(ctx, "macro_ipo", 1)), 1)
        out = {name: np.full((n, CH.MAX_PATHS), np.nan) for name in CH.TICK_NAMES}
        if inputs is not None:
            for p, sys in inputs.systems.items():
                if p >= CH.MAX_PATHS:
                    continue
                rows = int(sys.da.tick_n[0])
                tu = sys.da.tick_u[:min(rows, n)]
                for c, name in enumerate(CH.TICK_NAMES):
                    out[name][:tu.shape[0], p] = tu[:, c]
        out.update(state.dynamics.values)
        return out

    def _macro_values(self, ctx, state, frame: DynamicsFrame) -> dict:
        out = {"dynamics.vibration_pp_m": frame.vibration_pp_m, "dynamics.chatter_index": frame.chatter_index,
               "dynamics.chatter_freq_Hz": frame.chatter_freq_Hz, "dynamics.chatter_amp_m": frame.chatter_amp_m}
        per = {name: np.full(CH.MAX_PATHS, np.nan) for name in CH.MACRO_PATH_NAMES}
        dt = max(ctx.macro_s, 1e-300)
        for c in frame.contacts:
            p = c.path
            if p >= CH.MAX_PATHS:
                continue
            per["dynamics.vibration_pp_path_m"][p] = c.vibration_pp_m
            per["dynamics.chatter_index_path"][p] = c.chatter_index
            per["dynamics.chatter_freq_path_Hz"][p] = c.chatter_freq_Hz
            per["dynamics.modal_energy_J"][p] = c.modal_energy_J
            per["dynamics.damping_power_W"][p] = c.damping_dissipation_J / dt
            per["dynamics.cutting_power_W"][p] = c.cutting_work_J / dt
            per["dynamics.unbalance_power_W"][p] = c.unbalance_work_J / dt
            per["dynamics.base_power_W"][p] = c.base_work_J / dt
            per["dynamics.basis_change_J"][p] = c.basis_change_J
            per["dynamics.depth_limit_m"][p] = c.depth_limit_m if math.isfinite(c.depth_limit_m) else np.nan
            per["dynamics.stability_margin"][p] = (c.depth_limit_m / c.depth_of_cut_m
                                                   if c.depth_of_cut_m > 0.0 and math.isfinite(c.depth_limit_m)
                                                   else np.nan)
            per["dynamics.n_modes"][p] = c.n_modes
        out.update(per)
        return out

    # ------------------------------------------------------------------ end of operation / run
    def end_operation(self, ctx, state) -> None:
        ds = state.dynamics
        self.finish_stability(ctx, state)
        for win in ds.windows.values():
            win.clear()
        for ep in ds.episodes.values():
            ep.update(active=False, onset_tick=-1, clear=0)
        ds.last_point.clear()

    def finalize(self, ctx, state) -> dict:
        ds = state.dynamics
        if ds.pending:
            self.finish_stability(ctx, state)
        tools = getattr(state, "tools", None)
        files = []
        run_dir = getattr(ctx, "run_dir", None)
        for slot in sorted(set(ds.diagrams) | set(ds.points)):
            number = slot
            slots = getattr(tools, "slots", ()) if tools is not None else ()
            if slot < len(slots):
                number = int(slots[slot].number)
            doc = self.sld_document(state, slot, number)
            if run_dir is not None:
                path = run_dir / "sld" / f"T{number}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                write_json_atomic(path, doc)
                files.append(str(path.relative_to(run_dir)))
        return {"files": files, "totals": dict(ds.totals), "caught_up_ticks": ds.meta.get("caught_up_ticks", 0),
                "caught_up_in_contact": ds.meta.get("caught_up_in_contact", 0),
                "samples_dropped": ds.meta.get("samples_dropped", 0),
                "operating_points": {int(k): len(v) for k, v in ds.points.items()}}

    def sld_document(self, state, slot: int, number: int) -> dict:
        """sld/<tool>.json (docs/report.md section 3): the most critical diagram's lower envelope as speed_rad_s /
        depth_limit_m, every diagram with its lobes, and the operating points."""
        from .. import FORMAT_VERSION
        ds = state.dynamics
        diagrams = ds.diagrams.get(slot, [])
        points = ds.points.get(slot, [])
        critical = 0
        if points:
            worst = min(points, key=lambda q: q["margin"] if math.isfinite(q["margin"]) else math.inf)
            critical = int(worst.get("diagram") or 0)
        main = diagrams[critical] if diagrams else None
        return {"format": "cnctwin-sld", "version": FORMAT_VERSION, "tool": f"T{number}", "slot": int(slot),
                "number": int(number),
                "speed_rad_s": main["speed_rad_s"] if main else [],
                "depth_limit_m": main["depth_limit_m"] if main else [],
                "chatter_freq_Hz": main["chatter_freq_Hz"] if main else [],
                "radial_immersion_m": main["ae_m"] if main else None,
                "method": main["method"] if main else "",
                "diagrams": [{k: v for k, v in d.items() if k != "signature"} for d in diagrams],
                "operating_points": [{k: v for k, v in q.items() if k not in _POINT_PRIVATE} for q in points]}


# keys of an operating point kept for the end-of-operation analysis only (not in sld/<tool>.json)
_POINT_PRIVATE = ("class", "event")
LIMIT_SPEED_QUANTUM = 1e-3      # relative spindle-speed step of the limit cache (state.dynamics.meta["limits"])
LIMIT_CACHE_MAX = 4096          # cached limits (oldest dropped first)


def _frf_key(frf: SL.ModalFRF) -> str:
    """Content key of an FRF (modes and residual): equal FRFs give equal keys."""
    import hashlib
    h = hashlib.sha256()
    for a in (frf.omega_rad_s, frf.zeta, frf.psi, frf.static):
        h.update(np.ascontiguousarray(np.asarray(a, float)).tobytes())
    return h.hexdigest()[:16]


def _cut_doc(cut: SL.CutModel) -> dict:
    return {"M": np.asarray(cut.M, float), "alpha0_rad": np.asarray(cut.alpha0_rad, float),
            "tooth": np.asarray(cut.tooth, np.int64), "n_teeth": int(cut.n_teeth),
            "pitch_rad": np.asarray(cut.pitch_rad, float), "arc_lo_rad": float(cut.arc_lo_rad),
            "arc_hi_rad": float(cut.arc_hi_rad), "spin": float(cut.spin), "rotating": bool(cut.rotating),
            "damping_M": np.asarray(cut.damping_M, float), "radius_m": np.asarray(cut.radius_m, float),
            "work_radius_m": float(cut.work_radius_m)}


def _cut_from(doc: dict) -> SL.CutModel:
    return SL.CutModel(np.asarray(doc["M"], float).reshape(-1, 3, 3), np.asarray(doc["alpha0_rad"], float),
                       np.asarray(doc["tooth"], np.int64), int(doc["n_teeth"]), np.asarray(doc["pitch_rad"], float),
                       float(doc["arc_lo_rad"]), float(doc["arc_hi_rad"]), float(doc["spin"]), bool(doc["rotating"]),
                       np.asarray(doc["damping_M"], float).reshape(-1, 3, 3), np.asarray(doc["radius_m"], float),
                       float(doc["work_radius_m"]))


def _frf_doc(frf: SL.ModalFRF) -> dict:
    return {"omega_rad_s": np.asarray(frf.omega_rad_s, float), "zeta": np.asarray(frf.zeta, float),
            "psi": np.asarray(frf.psi, float), "static": np.asarray(frf.static, float)}


def _frf_from(doc: dict) -> SL.ModalFRF:
    return SL.ModalFRF(np.asarray(doc["omega_rad_s"], float), np.asarray(doc["zeta"], float),
                       np.asarray(doc["psi"], float).reshape(-1, 3), np.asarray(doc["static"], float).reshape(3, 3))


def _plain_sig(sig) -> list:
    return [bool(v) if isinstance(v, bool) else (int(v) if isinstance(v, (int, np.integer)) else v) for v in sig]


def _dominant(frf: SL.ModalFRF, rec: SystemRecord | None, share: float) -> SL.ModalFRF:
    """The modes carrying at least ``share`` of the static compliance at the contact (the semi-discretisation check
    runs on them; at most 8)."""
    c = np.sum(frf.psi ** 2, axis=1) / frf.omega_rad_s ** 2
    order = np.argsort(-c, kind="stable")
    keep = [int(k) for k in order if c[k] >= share * float(np.sum(c))][:8]
    if not keep:
        keep = [int(order[0])]
    keep = sorted(keep)
    return SL.ModalFRF(frf.omega_rad_s[keep], frf.zeta[keep], frf.psi[keep], np.zeros((3, 3)))


__all__ = ["DynamicsDomain", "DynamicsInputs", "axis_base", "l0_ticks_rule", "DOMAIN"]
