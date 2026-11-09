"""HoldingDomain: tool holding and balance, and workholding (docs/models/holding.md; contract: core/domain.py).

An L2 slow domain (no part in the hot loop). Per macro step it reads ``prev``: motion's paths (active tool,
spindle speed and angle per tick, workpiece, poses, tool tip in M and S), cutting's per-contact wrenches
(``wrench_tool_T0_mean/_peak`` on the tool, ``wrench_work_W_mean/_peak`` on the part, ``float_u_max_m``),
removal's contacts (contact -> workpiece, removed volume) and loose bodies, chips' ``locator_debris_m``,
thermal's holder and tool body temperatures (interference fits); it writes only ``nxt.holding``: per tool
slot a :class:`cnctwin.core.bus.ToolHolding` (pull-out, slip, unbalance vector and couple, capacities,
margin, walk) and per workpiece a :class:`cnctwin.core.bus.PartHolding` (pose delta, free, seated,
location error, supports for structure, margin, lathe support conditions).

Geometry comes through the protocols of :mod:`cnctwin.holding.protocols` (wired by the integration to
removal's queries): ``HoldingDomain(geometry=..., tabs=..., burrs=..., tools=...)``; without them the stock
defines the part, tabs keep their nominal section and burr records are absent (each fallback is recorded in
the results' notes).
"""

from __future__ import annotations

import math

import numpy as np

from ..core.bus import HoldingFeedback, PartHolding, ToolHolding
from ..core.errors import InputError
from ..core.findings import Category, Severity
from ..units import REFERENCE_TEMP_K, STANDARD_GRAVITY_M_S2
from . import abi as A
from . import balance as B
from . import params as P
from . import setup as SU
from . import toolhold as TH
from . import workhold as WH
from .channels import MAX_PARTS, MAX_SLOTS
from .channels import channels as holding_channels
from .locators import HoldingLocators
from .protocols import NoBurrs, NominalTabs, StockGeometry, stock_geometry
from .schema import settings as read_settings
from .state import HoldingState, PartHoldState, ToolHoldState

DOMAIN = "holding"
LATHE_TYPES = ("lathe", "mill_turn", "swiss")


def gravity_M(machine) -> np.ndarray:
    """g in M: the machine's kinematics.gravity_direction, else -z for mills and -x for lathes (turret above
    the spindle), times standard gravity (the convention chips uses)."""
    kin = P.field_of(machine, "kinematics") if machine is not None else None
    direction = P.numlist(kin, "gravity_direction") if kin is not None else None
    if direction is not None and len(direction) == 3:
        d = np.asarray(direction, dtype=float)
    else:
        d = np.array([-1.0, 0.0, 0.0]) if P.text(machine, "type", "") in LATHE_TYPES else np.array([0.0, 0.0, -1.0])
    return STANDARD_GRAVITY_M_S2 * d / np.linalg.norm(d)


def _frame_for_slot(prev, slot):
    for frame in prev.motion.paths:
        if int(frame.tool_slot) == int(slot):
            return frame
    return None


def _frame_for_workpiece(prev, w):
    for frame in prev.motion.paths:
        if int(frame.workpiece) == int(w):
            return frame
    return None


def _R_M_S(frame) -> np.ndarray | None:
    if frame is None:
        return None
    pose_M_T = np.asarray(frame.pose_M_T, dtype=float)
    pose_S_T = np.asarray(frame.pose_S_T, dtype=float)
    if pose_M_T.shape != (4, 4) or pose_S_T.shape != (4, 4):
        return None
    return pose_M_T[:3, :3] @ pose_S_T[:3, :3].T


def _wrench_to_H(w_W: np.ndarray, T_H_W: np.ndarray, O_H: np.ndarray) -> np.ndarray:
    """Wrenches [n, 6] on the part about the W origin in W -> about O in H."""
    R = T_H_W[:3, :3]
    r = T_H_W[:3, 3] - O_H
    out = np.empty_like(w_W)
    f = w_W[:, :3] @ R.T
    out[:, :3] = f
    out[:, 3:] = w_W[:, 3:] @ R.T
    out[:, 3] += r[1] * f[:, 2] - r[2] * f[:, 1]
    out[:, 4] += r[2] * f[:, 0] - r[0] * f[:, 2]
    out[:, 5] += r[0] * f[:, 1] - r[1] * f[:, 0]
    return out


def _tightest_tolerance(project) -> float | None:
    cad = P.field_of(project, "cad")
    tols = P.field_of(cad, "tolerances") if cad is not None else None
    values = []
    for tol in tols or []:
        for key in ("tolerance_m", "plus_m", "minus_m"):
            v = P.num(tol, key)
            if v is not None and v > 0.0:
                values.append(v)
    return min(values) if values else None


class HoldingDomain:
    """Tool holding and balance, and workholding."""

    name = DOMAIN

    def __init__(self, geometry=None, tabs=None, burrs=None, tools=None):
        self.geometry = geometry
        self.tabs = tabs
        self.burrs = burrs
        self.tools = tools
        self._rt: dict = {}

    # ------------------------------------------------------------------ Domain interface
    def channels(self):
        return holding_channels()

    def kernel_inputs(self, ctx, state, bus):
        return None

    def absorb(self, ctx, state, outputs, bus_next) -> None:
        return None

    def init(self, ctx, state) -> None:
        project = ctx.project
        st = read_settings(P.field_of(P.field_of(project, "physics"), "holding"))
        hs = HoldingState(st, int(ctx.seed))
        self._rt = {}
        self._providers(ctx, hs)
        for slot in state.tools.slots:
            td = self._tool_data(ctx, state, slot.slot)
            clamp = TH.build_clamp(td)
            clamp.detail["teeth"] = int(max(td.flutes, 1))
            unbalance = B.assembly_unbalance(td, hs.seed)
            ts = ToolHoldState(slot.slot, slot.number, clamp.to_dict(), unbalance.to_dict())
            hs.tools[slot.slot] = ts
            for note in clamp.notes + unbalance.notes:
                if note not in hs.notes:
                    hs.notes.append(note)
            q = {"slot": slot.slot, "torque_capacity_Nm": clamp.torque_cap_Nm if clamp.slips else -1.0,
                 "axial_retention_N": clamp.retention_N if math.isfinite(clamp.retention_N) else -1.0,
                 "pressure_Pa": clamp.pressure_Pa, "unbalance_kg_m": float(np.linalg.norm(unbalance.static_T_kg_m)),
                 "unbalance_worst_kg_m": unbalance.worst_kg_m, "required_grade_m_s": unbalance.required_grade_m_s}
            hs.trace_notes[f"tool:{slot.slot}"] = int(ctx.trace.record(0.0, 0, DOMAIN, f"holding:tool_clamp:{clamp.model}",
                                                                      q, []))
        for op in list(P.field_of(project, "operations") or []):
            w = SU.op_workpiece(op)
            if w not in hs.parts:
                hs.parts[w] = PartHoldState(w)
        hs.op_index = int(state.clock.op_index)
        hs.part_index = int(state.clock.part_index)
        state.holding = hs
        self._attach(ctx, state)

    def macro_step(self, ctx, state, prev, nxt) -> None:
        hs: HoldingState = state.holding
        self._attach(ctx, state)
        hs.op_index = int(state.clock.op_index)
        hs.part_index = int(state.clock.part_index)
        tools = {slot: self._tool_step(ctx, state, hs, ts, prev) for slot, ts in sorted(hs.tools.items())}
        parts = {w: self._part_step(ctx, state, hs, ps, prev) for w, ps in sorted(hs.parts.items())}
        nxt.holding = HoldingFeedback(tools=tools, parts=parts)
        self._values(hs, prev, tools, parts)

    def end_operation(self, ctx, state) -> None:
        """Close the operation: its setup summary goes to the results; the part is unclamped (the next
        operation's setup is built at its first macro step, with the chips left on its locating faces).
        Tool pull-out and slip carry over (the tools stay in their holders)."""
        hs: HoldingState = state.holding
        for w, ps in sorted(hs.parts.items()):
            if ps.op_index < 0 or ps.flags.get("closed_op") == [ps.op_index, ps.flags.get("part", 0)]:
                continue
            hs.setups.append({"op_index": ps.op_index, "part_index": int(ps.flags.get("part", 0)), "workpiece": w,
                              "min_margin": ps.min_margin, "max_slip_m": ps.max_slip_m, "max_lift_m": ps.max_lift_m,
                              "location_error_S_m": [float(v) for v in ps.location_error_S], "seated": ps.seated,
                              "seat_gap_m": ps.seat_gap_m, "free": ps.free, "free_cause": ps.free_cause,
                              "friction_J": ps.friction_J, "slip_pose_delta": np.asarray(ps.slip_pose).tolist(),
                              "summary": dict(ps.flags.get("summary", {}))})
            ps.flags["closed_op"] = [ps.op_index, ps.flags.get("part", 0)]

    def finalize(self, ctx, state) -> dict:
        hs: HoldingState = state.holding
        self.end_operation(ctx, state)
        tools = []
        for slot, ts in sorted(hs.tools.items()):
            tools.append({"slot": slot, "number": ts.number, "clamp": dict(ts.clamp), "unbalance": dict(ts.unbalance),
                          "pullout_m": ts.pullout_m, "slip_rad": ts.slip_rad, "friction_J": ts.friction_J,
                          "min_margin": ts.min_margin if math.isfinite(ts.min_margin) else None,
                          "max_walk_per_rev_m": ts.max_pullout_rate_m_per_rev,
                          "findings": dict(sorted(ts.counters.items()))})
        return {"tools": tools, "setups": list(hs.setups), "events": list(hs.events), "notes": list(hs.notes),
                "protocols": {"geometry": type(self._rt["geometry"]).__name__, "tabs": type(self._rt["tabs"]).__name__,
                              "burrs": type(self._rt["burrs"]).__name__}}

    def macro_values(self, state) -> dict:
        return dict(state.holding.values)

    def locator_faces(self, ctx) -> HoldingLocators:
        """The LocatorFaces chips needs: the next setup's locating and clamping patches in S."""
        settings = read_settings(P.field_of(P.field_of(ctx.project, "physics"), "holding"))
        geometry = self._rt.get("geometry") if self._rt else None
        if geometry is None:
            geometry = self.geometry or stock_geometry(P.field_of(ctx.project, "stock"))
        return HoldingLocators(ctx.project, geometry, settings)

    # ------------------------------------------------------------------ runtime
    def _providers(self, ctx, hs) -> None:
        project = ctx.project
        if self.geometry is not None:
            geometry = self.geometry
        else:
            workpieces = sorted({SU.op_workpiece(op) for op in (P.field_of(project, "operations") or [])} or {0})
            geometry = stock_geometry(P.field_of(project, "stock"), workpieces)
            note = "part geometry from the stock (no removal query wired): mass falls with the removed volume, shape " \
                   "and centroid stay"
            if note not in hs.notes:
                hs.notes.append(note)
        tabs = self.tabs if self.tabs is not None else NominalTabs()
        burrs = self.burrs if self.burrs is not None else NoBurrs()
        self._rt.update({"geometry": geometry, "tabs": tabs, "burrs": burrs})

    def _attach(self, ctx, state) -> None:
        hs: HoldingState = state.holding
        if "geometry" not in self._rt:
            self._providers(ctx, hs)
            if isinstance(self._rt["geometry"], StockGeometry):
                for w, ps in hs.parts.items():
                    self._rt["geometry"].removed_m3[w] = ps.removed_m3
        if "clamps" not in self._rt:
            self._rt["clamps"] = {slot: TH.ToolClamp.from_dict(ts.clamp) for slot, ts in hs.tools.items()}
            self._rt["unbalance"] = {slot: B.Unbalance.from_dict(ts.unbalance) for slot, ts in hs.tools.items()}
            self._rt["setups"] = {}
            self._rt["gravity_M"] = gravity_M(P.field_of(ctx.project, "machine"))
            self._rt["speed_rel"] = float(hs.settings.speed_rebuild_rel)

    def _tool_data(self, ctx, state, slot: int) -> TH.ToolData:
        src = self.tools
        if src is not None:
            if isinstance(src, dict):
                if slot not in src:
                    raise InputError(f"holding: no tool data for slot {slot}")
                return src[slot]
            if hasattr(src, "tool_data"):
                return src.tool_data(slot)
            if hasattr(src, "geometry"):
                return TH.tool_data(src.geometry(slot), getattr(ctx, "db", None))
        tooling = getattr(state, "tooling", None)
        if tooling is not None and hasattr(tooling, "geometry") and slot < len(tooling):
            return TH.tool_data(tooling.geometry(slot), getattr(ctx, "db", None))
        raise InputError(f"holding: tool slot {slot} has no geometry (the tooling domain must run its init first, or "
                         f"tool data must be given)")

    # ------------------------------------------------------------------ findings
    def _finding(self, ctx, state, hs, counters: dict, key: str, category, severity, message: str, tick: int,
                 path: int, quantities: dict, parents: list) -> bool:
        ck = f"{hs.part_index}:{hs.op_index}:{key}"
        counters[ck] = int(counters.get(ck, 0)) + 1
        if counters[ck] > hs.settings.max_findings:
            return False
        ctx.finding(state, DOMAIN, category, severity, message, int(tick), path=int(path),
                    quantities={k: float(v) for k, v in quantities.items()}, parents=[int(p) for p in parents])
        return True

    # ------------------------------------------------------------------ tools
    def _tool_step(self, ctx, state, hs, ts: ToolHoldState, prev) -> ToolHolding:
        clamp = self._rt["clamps"][ts.slot]
        ub = self._rt["unbalance"][ts.slot]
        rec = ToolHolding(pullout_m=ts.pullout_m, slip_rad=ts.slip_rad,
                          unbalance_static_kg_m=np.array(ub.static_T_kg_m, dtype=float), unbalance_z_T_m=ub.z_T_m,
                          unbalance_couple_kg_m2=np.array(ub.couple_T_kg_m2, dtype=float),
                          torque_capacity_Nm=clamp.torque_cap_Nm, axial_retention_N=clamp.retention_N,
                          max_rad_s=ub.max_rad_s)
        frame = _frame_for_slot(prev, ts.slot)
        if frame is None or not np.size(frame.spindle_omega_rad_s):
            ts.omega_prev = math.nan
            self._stop_tool_slip(state, ts, clamp, rec)
            return rec
        omega = np.asarray(frame.spindle_omega_rad_s, dtype=float)
        ticks = np.asarray(frame.tick, dtype=np.int64)
        w_max = float(np.max(np.abs(omega)))
        tick_end = int(ticks[-1]) if ticks.size else int(prev.motion.tick)
        path = int(frame.path)
        parent = hs.trace_notes.get(f"tool:{ts.slot}")
        parents = [parent] if parent is not None else []
        rec.unbalance_force_N = float(np.linalg.norm(ub.static_T_kg_m)) * w_max ** 2
        rec.balance_quality_m_s = B.effective_grade_m_s(ub.worst_kg_m, ub.mass_kg, w_max)
        if w_max > ub.max_rad_s * (1.0 + 1e-9):
            k = int(np.argmax(np.abs(omega)))
            limiter = "holder" if ub.max_rad_s_holder <= ub.max_rad_s_tool else "tool"
            ev = ctx.trace.record(ticks[k] * ctx.tick_s, int(ticks[k]), DOMAIN, "holding:overspeed",
                                  {"slot": ts.slot, "speed_rad_s": w_max, "max_rad_s": ub.max_rad_s}, parents)
            self._finding(ctx, state, hs, ts.counters, f"overspeed:{ts.slot}", Category.OVERSPEED, Severity.ERROR,
                          f"T{ts.number} runs at {w_max * 30 / math.pi:.0f} rpm, above the {limiter}'s limit "
                          f"{ub.max_rad_s * 30 / math.pi:.0f} rpm", int(ticks[k]), path,
                          {"speed_rad_s": w_max, "max_rad_s": ub.max_rad_s}, [ev])
        if ub.mass_kg > 0.0 and rec.balance_quality_m_s > ub.required_grade_m_s * (1.0 + 1e-9):
            ev = ctx.trace.record(tick_end * ctx.tick_s, tick_end, DOMAIN, "holding:unbalance",
                                  {"slot": ts.slot, "G_eff_m_s": rec.balance_quality_m_s,
                                   "G_required_m_s": ub.required_grade_m_s, "speed_rad_s": w_max,
                                   "unbalance_worst_kg_m": ub.worst_kg_m, "unbalance_force_N": rec.unbalance_force_N},
                                  parents)
            u_per = B.permissible_unbalance_kg_m(ub.required_grade_m_s, ub.mass_kg, w_max)
            self._finding(ctx, state, hs, ts.counters, f"unbalance:{ts.slot}", Category.UNBALANCE, Severity.WARNING,
                          f"T{ts.number}: residual unbalance up to {ub.worst_kg_m * 1e6:.1f} g mm against "
                          f"{u_per * 1e6:.1f} g mm allowed by G {ub.required_grade_m_s * 1e3:.1f} at "
                          f"{w_max * 30 / math.pi:.0f} rpm (balance quality G {rec.balance_quality_m_s * 1e3:.2f})",
                          tick_end, path, {"G_eff_m_s": rec.balance_quality_m_s, "G_required_m_s":
                                           ub.required_grade_m_s, "unbalance_worst_kg_m": ub.worst_kg_m,
                                           "permissible_kg_m": u_per, "speed_rad_s": w_max}, [ev])
        contacts = [cc for cc in prev.cutting.contacts if int(cc.tool_slot) == ts.slot]
        if clamp.model == "tapping" and clamp.float_travel_m > 0.0 and contacts:
            u_float = max(float(cc.float_u_max_m) for cc in contacts)
            if u_float >= clamp.float_travel_m * (1.0 - 1e-6):
                rec.float_bottomed = True
                ev = ctx.trace.record(tick_end * ctx.tick_s, tick_end, DOMAIN, "holding:float_bottomed",
                                      {"slot": ts.slot, "float_u_m": u_float, "travel_m": clamp.float_travel_m}, parents)
                self._finding(ctx, state, hs, ts.counters, f"float:{ts.slot}", Category.TOOL_PULLOUT, Severity.WARNING,
                              f"T{ts.number}: the floating tapping holder reaches the end of its axial compensation "
                              f"({clamp.float_travel_m * 1e3:.2f} mm): the tap is forced axially (pitch and feed do "
                              f"not match)", tick_end, path, {"float_u_m": u_float, "travel_m": clamp.float_travel_m},
                              [ev])
        if clamp.slips and contacts and w_max > 0.0:
            self._tool_slip(ctx, state, hs, ts, clamp, contacts, frame, omega, ticks, prev, rec, parents)
        ts.omega_prev = float(omega[-1])
        rec.pullout_m = ts.pullout_m
        rec.slip_rad = ts.slip_rad
        return rec

    @staticmethod
    def _stop_tool_slip(state, ts, clamp, rec) -> None:
        """The tool's slip ends without a spindle record (tool change, spindle gone): what is left of its kinetic
        energy is dissipated at the interface (booked as friction, architecture 12)."""
        r = clamp.radius_m if clamp.radius_m > 0.0 else 0.0
        m1 = clamp.tool_I_axial_kgm2 / (r * r) if r > 0.0 else 0.0
        ke = 0.5 * (m1 * float(ts.ratchet[2]) ** 2 + float(clamp.tool_mass_kg) * float(ts.ratchet[3]) ** 2)
        ts.ratchet[2:] = 0.0
        if ke > 0.0:
            rec.slip_loss_J += ke
            rec.slip_mech_change_J -= ke
            rec.slip_friction_J += ke
            state.ledger.friction_J += ke

    def _tool_slip(self, ctx, state, hs, ts, clamp, contacts, frame, omega, ticks, prev, rec, parents) -> None:
        dt = ctx.ipo_s
        n = min(len(omega), *(np.asarray(cc.wrench_tool_T0_mean).shape[0] for cc in contacts))
        if n <= 0:
            return
        mean = np.zeros((n, 6))
        peak = np.zeros((n, 6))
        for cc in contacts:
            mean += np.asarray(cc.wrench_tool_T0_mean, dtype=float)[:n]
            peak += np.asarray(cc.wrench_tool_T0_peak, dtype=float)[:n]
        om = omega[:n]
        sgn = np.sign(om)
        prev_w = ts.omega_prev if math.isfinite(ts.omega_prev) else float(om[0])
        alpha = np.diff(np.concatenate([[prev_w], om])) / dt
        r = clamp.radius_m
        J = clamp.tool_I_axial_kgm2
        d1_m = (-sgn * mean[:, 5] + sgn * J * alpha) / r
        d1_p = (-sgn * peak[:, 5] + sgn * J * alpha) / r
        if clamp.model == "side_lock":
            d1_m[:] = 0.0
            d1_p[:] = 0.0
        d2_m = -mean[:, 2]
        d2_p = -peak[:, 2]
        heat = prev.thermal.tools.get(ts.slot) if prev.thermal is not None else None
        factor = TH.pressure_factor(clamp, getattr(heat, "holder_temp_K", None) if heat is not None else None,
                                    getattr(heat, "body_temp_K", None) if heat is not None else None,
                                    REFERENCE_TEMP_K)
        if clamp.model == "side_lock":
            cap = clamp.retention_N * factor
        else:
            cap = min(clamp.torque_cap_Nm / r, clamp.retention_N) * factor
        teeth = int(clamp.detail.get("teeth", 1))
        theta = np.asarray(frame.spindle_angle_rad, dtype=float)[:n] if np.size(frame.spindle_angle_rad) >= n \
            else np.zeros(n)
        wave = A.empty_wave(n, 2)
        wave.mean[:, 0], wave.mean[:, 1] = d1_m, d2_m
        wave.amp[:, 0], wave.amp[:, 1] = d1_p - d1_m, d2_p - d2_m
        wave.phase0_rad[:] = teeth * sgn * theta
        wave.rate_rad_s[:] = teeth * np.abs(om)
        # the cutting torque is the teeth's pulse train (it resists the tool's turning, it never drives it): its
        # tooth-period shape from each tick's mean / peak (abi.pulse_alpha; the axial pull in phase with it); the
        # tool stalls at the spindle speed (the torque cannot turn it backwards against the work)
        cut_m = -sgn * mean[:, 5]
        cut_p = -sgn * peak[:, 5]
        for k in range(n):
            frac = cut_m[k] / cut_p[k] if cut_p[k] > 0.0 and cut_m[k] >= 0.0 else 1.0
            wave.alpha_rad[k, :] = A.pulse_alpha(frac)
        wave.v_max[:, 0] = np.abs(om) * r
        # static check: the demand over the tooth harmonic never leaves the friction circle and the tool is at
        # rest in its holder -> no slip this step (the ratchet is not needed)
        reach = np.maximum(np.hypot(d1_m + wave.amp[:, 0], d2_m + wave.amp[:, 1]),
                           np.hypot(d1_m - wave.amp[:, 0], d2_m - wave.amp[:, 1]))
        if clamp.back_stop and ts.ratchet[1] <= 0.0:
            reach = np.maximum(np.hypot(d1_m + wave.amp[:, 0], np.maximum(d2_m + wave.amp[:, 1], 0.0)),
                               np.hypot(d1_m - wave.amp[:, 0], np.maximum(d2_m - wave.amp[:, 1], 0.0)))
        demand_max = float(np.max(reach))
        if ts.ratchet[2] == 0.0 and ts.ratchet[3] == 0.0 and demand_max <= cap:
            margin = cap / demand_max if demand_max > 0.0 else math.inf
            rec.holding_margin = margin
            ts.min_margin = min(ts.min_margin, margin)
            return
        out = np.zeros(A.N_RT)
        from . import kernels as K
        m1, m2 = float(J / (r * r)), float(clamp.tool_mass_kg)
        ke0 = 0.5 * (m1 * float(ts.ratchet[2]) ** 2 + m2 * float(ts.ratchet[3]) ** 2)
        K.ratchet_tool(wave, float(dt), float(cap), float(cap * clamp.kinetic_ratio), m1, m2,
                       1 if clamp.back_stop else 0,
                       float(clamp.walk_limit_m) if math.isfinite(clamp.walk_limit_m) else 1e9,
                       int(hs.settings.sub_steps_per_tooth), int(hs.settings.max_sub_steps), ts.ratchet, out)
        dslip = float(out[A.RT_DX1]) / r
        dpull = float(out[A.RT_DX2])
        loss = float(out[A.RT_WORK_J])
        engaged = clamp.clamp_length_m if clamp.clamp_length_m > 0.0 else math.inf
        if ts.ratchet[1] >= engaged:
            # the shank has walked out of its bore: the tool leaves the holder (its axial motion ends there)
            dpull -= float(ts.ratchet[1]) - engaged
            ts.ratchet[1] = engaged
            loss += 0.5 * m2 * float(ts.ratchet[3]) ** 2
            ts.ratchet[3] = 0.0
            ts.flags["out"] = True
        ke1 = 0.5 * (m1 * float(ts.ratchet[2]) ** 2 + m2 * float(ts.ratchet[3]) ** 2)
        # the slip's energy into the run's ledger: the dissipation as friction (architecture 12), the loads' work
        # on the slip and the change of its kinetic energy as the run ledger's pair of holding terms
        rec.slip_in_J += float(out[A.RT_IN_J])
        rec.slip_mech_change_J += ke1 - ke0
        rec.slip_loss_J += loss
        state.ledger.friction_J += loss
        ts.slip_rad += dslip
        ts.pullout_m = float(ts.ratchet[1])
        revs = float(np.sum(np.abs(om)) * dt / (2.0 * math.pi))
        walk = dpull / revs if revs > 0.0 else 0.0
        rec.walk_per_rev_m = walk
        ts.max_pullout_rate_m_per_rev = max(ts.max_pullout_rate_m_per_rev, walk)
        rec.slip_friction_J = loss
        ts.friction_J += rec.slip_friction_J
        margin = 1.0 / float(out[A.RT_PEAK_RATIO]) if out[A.RT_PEAK_RATIO] > 0.0 else math.inf
        rec.holding_margin = margin
        ts.min_margin = min(ts.min_margin, margin)
        if dslip == 0.0 and dpull == 0.0:
            return
        k = int(np.argmax(np.abs(d1_p) + np.abs(d2_p)))
        tick = int(ticks[min(k, len(ticks) - 1)])
        path = int(frame.path)
        q = {"slot": ts.slot, "slip_rad": ts.slip_rad, "pullout_m": ts.pullout_m, "walk_per_rev_m": walk,
             "torque_peak_Nm": float(np.max(np.abs(peak[:, 5]))), "axial_peak_N": float(np.max(-peak[:, 2])),
             "capacity_N": cap, "pressure_factor": factor, "margin": margin, "friction_J": rec.slip_friction_J}
        ev = ctx.trace.record(tick * ctx.tick_s, tick, DOMAIN, "holding:tool_slip", q, parents)
        hs.events.append({"kind": "tool_slip", "slot": ts.slot, "tick": tick, "slip_rad": ts.slip_rad,
                          "pullout_m": ts.pullout_m, "walk_per_rev_m": walk})
        limit = P.p(P.TOOL, "pullout_error_m")
        tol = _tightest_tolerance(ctx.project)
        if tol is not None:
            limit = min(limit, tol)
        if abs(dslip) > 0.0:
            sev = Severity.ERROR if abs(ts.slip_rad) >= P.p(P.TOOL, "slip_error_rad") else Severity.WARNING
            self._finding(ctx, state, hs, ts.counters, f"slip:{ts.slot}:{sev.value}", Category.TOOL_PULLOUT, sev,
                          f"T{ts.number} turns in its holder: {ts.slip_rad * 180 / math.pi:.3f} deg so far (cutting "
                          f"torque {q['torque_peak_Nm']:.1f} Nm at a capacity of {cap * r:.1f} Nm)", tick, path, q,
                          [ev])
        if ts.flags.get("out") and not ts.flags.get("out_reported"):
            ts.flags["out_reported"] = True
            self._finding(ctx, state, hs, ts.counters, f"out:{ts.slot}", Category.TOOL_PULLOUT, Severity.FATAL,
                          f"T{ts.number} is pulled out of its holder over its whole clamped length "
                          f"({ts.pullout_m * 1e3:.1f} mm): the tool leaves the holder", tick, path,
                          dict(q, depth_error_m=ts.pullout_m), [ev])
        if dpull > 0.0:
            sev = Severity.ERROR if ts.pullout_m >= limit else Severity.WARNING
            self._finding(ctx, state, hs, ts.counters, f"pullout:{ts.slot}:{sev.value}", Category.TOOL_PULLOUT, sev,
                          f"T{ts.number} pulls out of its holder: {ts.pullout_m * 1e3:.4f} mm so far "
                          f"({walk * 1e6:.3f} um per revolution); the tool cuts that much deeper (depth error)", tick,
                          path, dict(q, depth_error_m=ts.pullout_m), [ev])

    # ------------------------------------------------------------------ parts
    def _setup_key(self, hs, op_index) -> list:
        return [int(op_index), int(hs.part_index)]

    def _build(self, ctx, state, hs, ps, op_index: int, prev, fresh: bool, resume: bool = False):
        """(Re)build the setup of operation op_index for the part. ``resume``: after a checkpoint resume, at the
        pose, gravity, speed and volume of the build in force (bit-identical continuation)."""
        project = ctx.project
        if fresh:
            ps.slip_pose = np.eye(4)
            ps.location_pose = np.eye(4)
            ps.location_error_S = np.zeros(3)
            ps.free, ps.free_cause, ps.free_tick = False, "", -1
            ps.ratchet = {}
            ps.bond_failed = {}
            ps.vacuum_lost = []
            ps.slip_since_build_m = 0.0
            ps.o_hist_n = 0
            ps.min_margin, ps.max_slip_m, ps.max_lift_m, ps.friction_J = math.inf, 0.0, 0.0, 0.0
            ps.debris = self._debris(ctx, state, hs, ps, op_index, prev)
            ps.op_index = int(op_index)
            ps.flags["part"] = int(hs.part_index)
            ps.flags.pop("closed_op", None)
        if resume:
            gravity_H, omega, pose = ps.build_gravity.copy(), ps.build_omega, ps.build_pose.copy()
        else:
            gravity_H = self._gravity_H(ps, prev, op_index, ctx)
            omega = self._work_speed(prev, ps.workpiece)
            pose = ps.slip_pose.copy()
        setup = SU.build_setup(project, op_index, self._rt["geometry"], self._rt["tabs"], hs.settings,
                               debris=ps.debris, gravity_H=gravity_H, slip_pose=pose,
                               bond_failed=ps.bond_failed, vacuum_lost=set(ps.vacuum_lost), omega_rad_s=omega)
        if resume:
            setup.volume_built = ps.build_volume
        else:
            ps.build_pose, ps.build_gravity, ps.build_omega = pose, np.asarray(gravity_H, float).copy(), float(omega)
            ps.build_volume = float(setup.volume_built)
            ps.supports_version += 1
        self._rt["setups"][ps.workpiece] = setup
        for g in setup.groups:
            ps.ratchet.setdefault(g.name, np.zeros(3))
        return setup

    def _debris(self, ctx, state, hs, ps, op_index, prev) -> dict:
        """Chips and burrs on the setup's locating patches (patch id -> height)."""
        debris = {}
        chips = getattr(prev, "chips", None)
        for key, value in (getattr(chips, "locator_debris_m", {}) or {}).items():
            if float(value) > 0.0:
                debris[str(key)] = float(value)
        locators = HoldingLocators(ctx.project, self._rt["geometry"], hs.settings)
        faces = locators.faces_for_op(op_index)
        burrs = self._rt["burrs"].burr_heights_m(op_index, faces) or {}
        for key, value in burrs.items():
            debris[str(key)] = max(float(value), debris.get(str(key), 0.0))
        if not getattr(self._rt["burrs"], "wired", True):
            note = "no burr record wired (surface domain): burrs under locators come only through chips' debris"
            if note not in hs.notes:
                hs.notes.append(note)
        return debris

    def _gravity_H(self, ps, prev, op_index, ctx) -> np.ndarray:
        g_M = self._rt["gravity_M"]
        frame = _frame_for_workpiece(prev, ps.workpiece)
        R_M_S = _R_M_S(frame)
        op = SU.operation(ctx.project, op_index)
        T_H_S = ps.slip_pose @ ps.location_pose @ SU.stock_pose(op)
        if R_M_S is None:
            return np.array([0.0, 0.0, -float(np.linalg.norm(g_M))]) if g_M[2] != 0.0 else g_M.copy()
        R_M_H = R_M_S @ T_H_S[:3, :3].T
        return R_M_H.T @ g_M

    def _work_speed(self, prev, w) -> float:
        frame = _frame_for_workpiece(prev, w)
        if frame is None or not np.size(frame.spindle_omega_rad_s):
            return 0.0
        return float(np.max(np.abs(np.asarray(frame.spindle_omega_rad_s, dtype=float))))

    def _part_step(self, ctx, state, hs, ps: PartHoldState, prev) -> PartHolding:
        op_index = hs.op_index
        op = SU.operation(ctx.project, op_index)
        current = op is not None and SU.op_workpiece(op) == ps.workpiece
        geometry = self._rt["geometry"]
        for rc in getattr(prev.removal, "contacts", []) or []:
            if int(rc.workpiece) == ps.workpiece and float(rc.removed_volume_m3) > 0.0:
                ps.removed_m3 += float(rc.removed_volume_m3)
                if isinstance(geometry, StockGeometry):
                    geometry.note_removed(ps.workpiece, float(rc.removed_volume_m3))
        if not current:
            return self._record(ps, None, None, [])
        fresh = ps.op_index != op_index or ps.flags.get("part", 0) != hs.part_index
        setup = self._rt["setups"].get(ps.workpiece)
        if fresh or setup is None or setup.op_index != op_index:
            setup = self._build(ctx, state, hs, ps, op_index, prev, fresh, resume=not fresh)
            if fresh:
                self._setup_findings(ctx, state, hs, ps, setup, prev)
        if ps.free:
            return self._record(ps, setup, None, [])
        if setup.empty:
            note = f"{setup.op_name}: no workholding in the setup: holding of workpiece {ps.workpiece} not evaluated"
            if note not in hs.notes:
                hs.notes.append(note)
            return self._record(ps, setup, None, [])
        tick_end = int(prev.motion.tick)
        self._released(ctx, state, hs, ps, setup, prev, tick_end)
        if ps.free:
            return self._record(ps, setup, None, [])
        self._device_speed_limits(ctx, state, hs, ps, setup, prev)
        if self._chuck_speed_changed(setup, prev, ps):
            setup = self._rebuild_or_free(ctx, state, hs, ps, setup, prev, "the gripping force is lost at speed "
                                          "(centrifugal jaw loss)")
            if setup is None:
                return self._record(ps, self._rt["setups"].get(ps.workpiece), None, [])
        if self._tabs_changed(setup, ps):
            setup = self._rebuild_or_free(ctx, state, hs, ps, setup, prev, "tabs released")
            if setup is None:
                return self._record(ps, self._rt["setups"].get(ps.workpiece), None, [])
        results = {}
        support_rows = []
        for group in setup.groups:
            loads = self._loads(ctx, setup, group, ps, prev)
            res = WH.evaluate_group(group, loads, hs.settings, ps.ratchet.setdefault(group.name, np.zeros(3)))
            results[group.name] = (group, res, loads)
            if res.bond_failed:
                for i in res.bond_failed:
                    patch = group.patches[int(group.node_patch[i])]
                    mask = ps.bond_failed.setdefault(patch.id, np.zeros(patch.n_nodes, bool))
                    mask[int(group.node_local[i])] = True
            if res.seal_open:
                for hd in group.holddown:
                    if hd["kind"] == "vacuum" and hd["device"] not in ps.vacuum_lost:
                        ps.vacuum_lost.append(hd["device"])
        record_extra = self._outcome(ctx, state, hs, ps, setup, results, prev, tick_end)
        setup = self._rt["setups"][ps.workpiece]
        T_S_H = np.linalg.inv(setup.T_H_S)
        for name, (group, res, _) in sorted(results.items()):
            g_now = setup.group(name) or group
            support_rows.extend(WH.supports(g_now, res.forces_H if g_now is group else None,
                                            res.states if g_now is group else None, T_S_H))
        return self._record(ps, setup, record_extra, support_rows)

    def _released(self, ctx, state, hs, ps, setup, prev, tick_end) -> None:
        """Removal's loose bodies: a released body that holds a large share of the workpiece is the part."""
        volume_now, _, _ = self._rt["geometry"].mass_properties_S(ps.workpiece)
        for body in getattr(prev.removal, "loose_bodies", []) or []:
            if int(body.get("workpiece", 0)) != ps.workpiece:
                continue
            v = float(body.get("volume_m3", 0.0))
            if v < P.p(P.WORK, "free_volume_fraction") * (volume_now + v):
                continue
            tick = int(body.get("tick", tick_end))
            q = {"volume_m3": v, "mass_kg": float(body.get("mass_kg", 0.0)), "share": v / (volume_now + v)}
            ev = ctx.trace.record(tick * ctx.tick_s, tick, DOMAIN, "holding:part_released", q,
                                  [hs.trace_notes[k] for k in (f"setup:{ps.workpiece}",) if k in hs.trace_notes])
            sev = Severity.WARNING if setup.rotating else Severity.ERROR
            what = "the part is parted off (check the part catcher)" if setup.rotating else \
                "the part comes loose from the held material while the program runs"
            self._set_free(ps, "released by removal (connectivity)", tick)
            self._finding(ctx, state, hs, ps.counters, f"released:{ps.workpiece}", Category.PART_FREE, sev,
                          f"{setup.op_name}: {what} ({v * 1e9:.0f} mm^3, {q['share']:.0%} of the workpiece)", tick, 0,
                          q, [ev])
            return

    def _chuck_speed_changed(self, setup, prev, ps) -> bool:
        """The centrifugal jaw loss changed the gripping force by more than speed_rebuild_rel of a chuck's static
        force since the build (the loss grows with omega^2: the change between the build speed and now from the
        loss at the device's reference speed). A spindle running up rebuilds the setup only when the grip moved
        that much, not at every step of the run-up (each rebuild sets structure's work model up again)."""
        if not setup.rotating:
            return False
        omega = self._work_speed(prev, ps.workpiece)
        built = float(getattr(setup, "omega_built", 0.0))
        rel = self._rt["speed_rel"]
        for d in getattr(setup, "devices", []) or []:
            static = float(d.get("static_force_N", 0.0) or 0.0)
            ref, w_ref = float(d.get("loss_ref_N", 0.0) or 0.0), float(d.get("loss_ref_rad_s", 0.0) or 0.0)
            if static <= 0.0 or ref <= 0.0 or w_ref <= 0.0:
                continue
            if ref * abs(omega * omega - built * built) / (w_ref * w_ref) > rel * static:
                return True
        return False

    def _tabs_changed(self, setup, ps) -> bool:
        tabs = self._rt["tabs"]
        if not getattr(setup, "tab_fraction", None) or not getattr(tabs, "wired", True):
            return False
        group = setup.group("tabs")
        if group is None:
            return False
        T_S_H = np.linalg.inv(setup.T_H_S)
        for p in group.patches:
            c_S = T_S_H[:3, :3] @ p.centre_H + T_S_H[:3, 3]
            n_S = T_S_H[:3, :3] @ (-p.normal_H)
            a_S = p.axes_H @ T_S_H[:3, :3].T
            area, _ = tabs.section_S(ps.workpiece, c_S, n_S, a_S[0], a_S[1], float(p.half_m[0]), float(p.half_m[1]))
            nominal = 4.0 * float(p.half_m[0]) * float(p.half_m[1])
            frac = min(max(area / nominal, 0.0), 1.0) if nominal > 0.0 else 0.0
            if abs(frac - setup.tab_fraction.get(p.id, 1.0)) > 1e-9:
                return True
        return False

    def _device_speed_limits(self, ctx, state, hs, ps, setup, prev) -> None:
        if not setup.speed_limits:
            return
        omega = self._work_speed(prev, ps.workpiece)
        for device, kind, limit in setup.speed_limits:
            if omega > limit * (1.0 + 1e-9):
                tick = int(prev.motion.tick)
                ev = ctx.trace.record(tick * ctx.tick_s, tick, DOMAIN, "holding:device_overspeed",
                                      {"speed_rad_s": omega, "max_rad_s": limit}, [])
                self._finding(ctx, state, hs, ps.counters, f"overspeed:{device}", Category.OVERSPEED, Severity.ERROR,
                              f"{setup.op_name}: the work spindle turns at {omega * 30 / math.pi:.0f} rpm, above the "
                              f"{kind.replace('_', ' ')} {device}'s limit {limit * 30 / math.pi:.0f} rpm", tick, 0,
                              {"speed_rad_s": omega, "max_rad_s": limit}, [ev])

    def _loads(self, ctx, setup, group, ps, prev) -> WH.StepLoads:
        dt = ctx.ipo_s
        frame = _frame_for_workpiece(prev, ps.workpiece)
        contact_w = {int(rc.contact): int(rc.workpiece) for rc in getattr(prev.removal, "contacts", []) or []}
        ccs = [cc for cc in prev.cutting.contacts if contact_w.get(int(cc.contact), 0) == ps.workpiece]
        n = 0
        if frame is not None and np.size(frame.tick):
            n = int(np.size(frame.tick))
        for cc in ccs:
            rows = int(np.asarray(cc.wrench_work_W_mean).shape[0])
            n = rows if n == 0 else min(n, rows) if rows > 0 else n
        if n == 0:
            n = 1
        O = group.body.centroid_H
        T_H_W = ps.slip_pose @ ps.location_pose
        mean = np.zeros((n, 6))
        peak = np.zeros((n, 6))
        cutting = False
        for cc in ccs:
            wm = np.asarray(cc.wrench_work_W_mean, dtype=float)
            wp = np.asarray(cc.wrench_work_W_peak, dtype=float)
            if wm.shape[0] < n or wp.shape[0] < n:
                continue
            mean += _wrench_to_H(wm[:n], T_H_W, O)
            peak += _wrench_to_H(wp[:n], T_H_W, O)
            cutting = cutting or bool(np.any(wm != 0.0))
        # the cut's tooth-period pulse (|mean force| / |peak force| per tick) before the inertia is added
        f_mean = np.linalg.norm(mean[:, :3], axis=1)
        f_peak = np.linalg.norm(peak[:, :3], axis=1)
        cut_fraction = np.where(f_peak > 0.0, np.minimum(f_mean / np.where(f_peak > 0.0, f_peak, 1.0), 1.0), 1.0)
        v_limit, contact_H = self._cut_speed_limit(ctx, setup, ps, prev, frame, ccs, n, T_H_W, O)
        # gravity now (orientation and mass) against the clamped state
        volume, g_now = self._step_cache(ctx, setup, ps, prev)
        m_now = group.body.mass_kg * (volume / setup.volume_built) if setup.volume_built > 0.0 else group.body.mass_kg
        gravity_delta = np.zeros(6)
        gravity_delta[:3] = m_now * g_now - group.body.mass_kg * group.gravity_H
        inertia = self._inertia(ctx, setup, group, ps, prev, frame, n, m_now)
        mean += inertia
        peak += inertia
        phase0 = np.zeros(n)
        rate = np.zeros(n)
        if frame is not None and not setup.rotating and np.size(frame.spindle_omega_rad_s) >= n:
            om = np.asarray(frame.spindle_omega_rad_s, dtype=float)[:n]
            th = np.asarray(frame.spindle_angle_rad, dtype=float)[:n] if np.size(frame.spindle_angle_rad) >= n \
                else np.zeros(n)
            slot = int(frame.tool_slot)
            clamp = self._rt["clamps"].get(slot)
            teeth = int(clamp.detail.get("teeth", 1)) if clamp is not None else 1
            phase0 = teeth * np.sign(om) * th
            rate = teeth * np.abs(om)
        T_H_S = T_H_W @ SU.stock_pose(SU.operation(ctx.project, setup.op_index))
        corners = setup.shape.corners_S() @ T_H_S[:3, :3].T + T_H_S[:3, 3]
        return WH.StepLoads(mean, peak, gravity_delta, phase0, rate, dt, cutting, corners, cut_fraction, v_limit,
                            contact_H)

    def _cut_speed_limit(self, ctx, setup, ps, prev, frame, ccs, n, T_H_W, O):
        """(speed per tick the cut can push the part at its contact, contact point relative to O in H) - the
        slip's physical bound (workhold.StepLoads): a table-mounted part cannot outrun the tool (twice its feed
        speed relative to the part: the feed itself plus the chip thinning per tooth), a part turning in a chuck
        cannot be turned backwards by the cutting torque (the cutting speed at the contact). (None, None) without
        a cut or a motion record."""
        if frame is None:
            return None, None
        key = (int(prev.motion.tick), ps.workpiece)
        cache = self._rt.setdefault("speed_cache", {})
        if cache.get("key") == key:
            return cache["value"]
        speeds = self._tip_speeds(ctx, ps, frame, n)
        value = self._cut_speed_limit_now(ctx, setup, ps, prev, frame, n, T_H_W, O, speeds) if ccs else (None, None)
        cache.clear()
        cache.update({"key": key, "value": value})
        return value

    @staticmethod
    def _tip_speeds(ctx, ps, frame, n):
        """The tool tip's speed relative to the part per tick (S; from the last tick of the previous step), kept up
        to date every step (None without a tip record)."""
        tip = np.asarray(frame.tip_S_m, dtype=float)
        if tip.ndim != 2 or tip.shape[0] < n or n <= 0:
            return None
        prev_tip = ps.tip_prev if getattr(ps, "tip_prev", None) is not None else tip[0]
        steps = np.diff(np.vstack([np.asarray(prev_tip, dtype=float)[None, :], tip[:n]]), axis=0)
        ps.tip_prev = tip[n - 1].copy()
        return np.linalg.norm(steps, axis=1) / ctx.ipo_s

    def _cut_speed_limit_now(self, ctx, setup, ps, prev, frame, n, T_H_W, O, speeds):
        point_S = None
        for rc in getattr(prev.removal, "contacts", []) or []:
            if int(rc.workpiece) == ps.workpiece and bool(rc.geometry.in_cut):
                point_S = np.asarray(rc.geometry.point_S_m, dtype=float)
                break
        if point_S is None:
            return None, None
        if setup.rotating:
            # a turned part is cut at the tool's nose (removal's turning contacts carry no centroid of the removed
            # ring: their point_S_m is zero, docs/models/loop.md 15)
            tips = np.asarray(frame.tip_S_m, dtype=float)
            if tips.ndim == 2 and tips.shape[0]:
                point_S = tips[-1]
        op = SU.operation(ctx.project, setup.op_index)
        T_H_S = T_H_W @ SU.stock_pose(op)
        contact_H = T_H_S[:3, :3] @ point_S + T_H_S[:3, 3] - O
        if setup.rotating:
            om = np.abs(np.asarray(frame.spindle_omega_rad_s, dtype=float))
            if om.size < n:
                return None, None
            r_c = float(math.hypot(point_S[0], point_S[1]))
            return om[:n] * r_c, contact_H
        if speeds is None:
            return None, None
        return 2.0 * speeds, contact_H

    def _step_cache(self, ctx, setup, ps, prev):
        """(part volume now, gravity in H now), computed once per macro step and workpiece."""
        key = (int(prev.motion.tick), ps.workpiece, id(setup))
        cache = self._rt.setdefault("step_cache", {})
        if cache.get("key") != key:
            volume = float(self._rt["geometry"].mass_properties_S(ps.workpiece)[0])
            cache.clear()
            cache.update({"key": key, "volume": volume, "g": self._gravity_H(ps, prev, setup.op_index, ctx)})
        return cache["volume"], cache["g"]

    def _inertia(self, ctx, setup, group, ps, prev, frame, n, mass) -> np.ndarray:
        """-m a (table-mounted parts) or -I alpha about the spindle axis (parts in a chuck), per tick."""
        out = np.zeros((n, 6))
        dt = ctx.ipo_s
        if frame is None:
            return out
        if setup.rotating:
            om = np.asarray(frame.spindle_omega_rad_s, dtype=float)
            if om.size < n:
                return out
            prev_w = ps.omega_prev if math.isfinite(ps.omega_prev) else float(om[0])
            alpha = np.diff(np.concatenate([[prev_w], om[:n]])) / dt
            ps.omega_prev = float(om[n - 1])
            if group.axis_H is not None:
                a = np.asarray(group.axis_H[1], dtype=float)
                I_ax = float(a @ group.body.inertia_H @ a) * (mass / group.body.mass_kg if group.body.mass_kg else 1.0)
                out[:, 3:] = -I_ax * alpha[:, None] * a[None, :]
            return out
        R_M_S = _R_M_S(frame)
        tip_M = np.asarray(frame.tip_M_m, dtype=float)
        tip_S = np.asarray(frame.tip_S_m, dtype=float)
        if R_M_S is None or tip_M.shape[0] < n or tip_S.shape[0] < n:
            return out
        origins = tip_M[:n] - tip_S[:n] @ R_M_S.T
        hist = [ps.o_hist[i] for i in range(2 - min(ps.o_hist_n, 2), 2)]
        series = np.vstack(hist + [origins]) if hist else origins
        offset = len(hist)
        op = SU.operation(ctx.project, setup.op_index)
        T_H_S = ps.slip_pose @ ps.location_pose @ SU.stock_pose(op)
        R_M_H = R_M_S @ T_H_S[:3, :3].T
        for k in range(n):
            j = k + offset
            if j >= 2:
                a_M = (series[j] - 2.0 * series[j - 1] + series[j - 2]) / (dt * dt)
                out[k, :3] = -mass * (R_M_H.T @ a_M)
        tail = series[-2:] if series.shape[0] >= 2 else series
        ps.o_hist[2 - tail.shape[0]:] = tail
        ps.o_hist_n = min(ps.o_hist_n + n, 2)
        return out

    def _outcome(self, ctx, state, hs, ps, setup, results, prev, tick_end) -> dict:
        """Pose update, free checks and findings of the step; returns the record's extra values."""
        extra = {"margin": math.inf, "mode": "", "slip_m": 0.0, "lift_m": 0.0, "friction_J": 0.0, "in_J": 0.0,
                 "loss_J": 0.0, "mech_J": 0.0}
        for _, res, _ in results.values():
            extra["in_J"] += res.in_J
            extra["loss_J"] += res.loss_J
            extra["mech_J"] += res.mech_change_J
        state.ledger.friction_J += extra["loss_J"]
        parent = hs.trace_notes.get(f"setup:{ps.workpiece}")
        parents = [parent] if parent is not None else []
        rebuild = False
        cutting = any(loads.cutting for _, _, loads in results.values())
        safety = P.p(P.WORK, "chuck_safety_factor") if setup.rotating else P.p(P.WORK, "part_safety_factor")
        worst = math.inf
        for name, (group, res, loads) in sorted(results.items()):
            extra["margin"] = min(extra["margin"], res.margin)
            if not res.held and res.margin < worst:
                worst = res.margin
                extra["mode"] = "slip" if res.mode == A.MODE_FRICTION else "lift"
            ps.min_margin = min(ps.min_margin, res.margin) if cutting else ps.min_margin
            tick = tick_end
            if res.bond_failed:
                rebuild = True
                q = {"failed_nodes": float(len(res.bond_failed)),
                     "remaining_nodes": float(np.count_nonzero(group.contacts.kind == A.KIND_BONDED))}
                ctx.trace.record(tick * ctx.tick_s, tick, DOMAIN, f"holding:bond_failure:{name}", q, parents)
            if res.seal_open:
                rebuild = True
                ev = ctx.trace.record(tick * ctx.tick_s, tick, DOMAIN, "holding:vacuum_seal_open",
                                      {"margin": res.margin}, parents)
                hs.trace_notes[f"seal:{ps.workpiece}"] = int(ev)
            if not np.any(group.contacts.kind != A.KIND_OFF):
                cause = "tabs released" if name == "tabs" else "every contact failed"
                self._free(ctx, state, hs, ps, setup, cause, tick, parents, {"margin": 0.0})
                return extra
            if res.held:
                if cutting and res.margin < safety:
                    ev = ctx.trace.record(tick * ctx.tick_s, tick, DOMAIN, "holding:low_margin",
                                          {"margin": res.margin, "safety_factor": safety}, parents)
                    self._finding(ctx, state, hs, ps.counters, f"margin:{ps.workpiece}:{name}", Category.PART_SLIP,
                                  Severity.WARNING,
                                  f"{setup.op_name}: holding margin {res.margin:.2f} against the cutting load (safety "
                                  f"factor {safety:.1f} recommended)", tick, 0, {"margin": res.margin,
                                                                                  "safety_factor": safety}, [ev])
                continue
            q = {"margin": res.margin, "slip_m": res.slip_extent_m, "lift_m": res.lift_extent_m,
                 "generalized_slip_m": res.dq_m, "generalized_lift_m": res.q_max_m, "capacity_N": res.q_cap_N,
                 "demand_peak_N": res.demand_peak_N, "failing_ticks": float(res.failing_ticks),
                 "friction_J": res.work_J}
            ev = ctx.trace.record(tick * ctx.tick_s, tick, DOMAIN, f"holding:part_motion:{name}", q, parents)
            hs.events.append({"kind": "part_motion", "workpiece": ps.workpiece, "group": name, "tick": tick,
                              "mode": "slip" if res.mode == A.MODE_FRICTION else "lift", "slip_m": res.slip_extent_m,
                              "lift_m": res.lift_extent_m, "margin": res.margin})
            if res.mode == A.MODE_FRICTION:
                extra["slip_m"] += res.slip_extent_m
                extra["friction_J"] += res.work_J
                if name == "main" and res.dq_m > 0.0:
                    T_step = SU.twist_matrix(res.twist, group.body.centroid_H)
                    ps.slip_pose = T_step @ ps.slip_pose
                    ps.slip_since_build_m += float(np.linalg.norm(res.twist[:3]) +
                                                   group.body.l_c_m * np.linalg.norm(res.twist[3:]))
                    ps.max_slip_m = max(ps.max_slip_m, self._slip_magnitude(ctx, ps, setup))
                    if ps.slip_since_build_m >= hs.settings.rebuild_slip_m:
                        rebuild = True
                elif name == "tabs" and res.dq_m > 0.0:
                    self._free(ctx, state, hs, ps, setup, "tabs break (slip at the tabs)", tick, [ev], q)
                    return extra
                if res.slip_extent_m > P.p(P.WORK, "slip_report_m"):
                    self._finding(ctx, state, hs, ps.counters, f"slip:{ps.workpiece}:{name}", Category.PART_SLIP,
                                  Severity.ERROR,
                                  f"{setup.op_name}: the part slips in its holding (up to {res.slip_extent_m * 1e3:.4f} "
                                  f"mm this step, margin {res.margin:.2f})", tick, 0, q, [ev])
            else:
                extra["lift_m"] = max(extra["lift_m"], res.lift_extent_m)
                ps.max_lift_m = max(ps.max_lift_m, res.lift_extent_m)
                if np.any(group.seal) and res.lift_extent_m > P.p(P.WORK, "seat_tolerance_m"):
                    # a part lifting or tipping off a vacuum plate opens its seal
                    for hd in group.holddown:
                        if hd["kind"] == "vacuum" and hd["device"] not in ps.vacuum_lost:
                            ps.vacuum_lost.append(hd["device"])
                    rebuild = True
                if res.lift_extent_m > P.p(P.WORK, "lift_report_m"):
                    self._finding(ctx, state, hs, ps.counters, f"lift:{ps.workpiece}:{name}", Category.PART_SLIP,
                                  Severity.ERROR,
                                  f"{setup.op_name}: the part lifts off or tips in its holding (up to "
                                  f"{res.lift_extent_m * 1e3:.4f} mm, margin {res.margin:.2f})", tick, 0, q, [ev])
                if res.lift_extent_m > hs.settings.max_disp_m and self._escapes(ctx, hs, ps, setup, group, res, prev):
                    self._free(ctx, state, hs, ps, setup, "lifted out of its holding", tick, [ev], q)
                    return extra
        if rebuild:
            cause = "vacuum seal broken" if ps.vacuum_lost else "holding lost after slip"
            if self._rebuild_or_free(ctx, state, hs, ps, setup, prev, cause) is not None:
                ps.slip_since_build_m = 0.0
        return extra

    def _rebuild_or_free(self, ctx, state, hs, ps, setup, prev, cause: str):
        """Place the contacts again on the moved or changed part; the part is free when the devices no longer
        reach it or cannot hold it. Returns the new setup, or None when the part came free."""
        tick = int(prev.motion.tick)
        parent = hs.trace_notes.get(f"setup:{ps.workpiece}")
        parents = [parent] if parent is not None else []
        try:
            new = self._build(ctx, state, hs, ps, setup.op_index, prev, False)
        except InputError as error:
            self._free(ctx, state, hs, ps, setup, f"{cause}: the part left its holding ({error})", tick, parents,
                       {"margin": 0.0})
            return None
        if any(not g.held for g in new.groups) or (new.no_grip and not any(g.n for g in new.groups)) \
                or self._grip_lost(new) or self._only_gravity(new, ps):
            self._free(ctx, state, hs, ps, new, cause, tick, parents, {"margin": 0.0})
            return None
        return new

    @staticmethod
    def _grip_lost(setup) -> bool:
        """A part held only by chucks or collets whose gripping force has fallen to zero is not held."""
        grips = [d for d in setup.devices if d["type"] in ("chuck_3jaw", "chuck_4jaw", "collet_chuck", "collet")]
        others = [d for d in setup.devices if d["type"] in ("tailstock_centre", "steady_rest")]
        return bool(grips) and not others and all(float(d.get("gripping_force_N", 0.0)) <= 0.0 for d in grips)

    @staticmethod
    def _only_gravity(setup, ps) -> bool:
        """After a vacuum seal broke: nothing clamps, bonds or pulls the part down any more."""
        if not ps.vacuum_lost:
            return False
        main = setup.group("main")
        if main is None:
            return True
        clamps = any(p.law in ("locked", "bond") for p in main.patches)
        pull = any(not (hd["kind"] == "vacuum" and hd["device"] in ps.vacuum_lost) for hd in main.holddown)
        return not clamps and not pull

    @staticmethod
    def _holds_weight(group) -> bool:
        if group.n == 0:
            return False
        status, _, _, _ = WH.solve(group, group.w_ref)
        return status == A.STATUS_HELD

    def _escapes(self, ctx, hs, ps, setup, group, res, prev) -> bool:
        """A lift-off that reaches q_max: do the clamps still engage the part at that displaced pose?"""
        T = SU.twist_matrix(res.mech * res.q_max_m, group.body.centroid_H)
        try:
            trial = SU.build_setup(ctx.project, setup.op_index, self._rt["geometry"], self._rt["tabs"], hs.settings,
                                   debris=ps.debris, gravity_H=setup.gravity_H, slip_pose=T @ ps.slip_pose,
                                   bond_failed=ps.bond_failed, vacuum_lost=set(ps.vacuum_lost),
                                   omega_rad_s=getattr(setup, "omega_built", 0.0))
        except InputError:
            return True
        g = trial.group(group.name)
        if g is None or not g.held:
            return True
        had = sum(1 for p in group.patches if p.law in ("locked", "bond"))
        has = sum(1 for p in g.patches if p.law in ("locked", "bond"))
        return had > 0 and has == 0

    def _slip_magnitude(self, ctx, ps, setup) -> float:
        """Largest displacement of the part's bounding-box corners by the accumulated slip (m)."""
        T_H_S = ps.location_pose @ SU.stock_pose(SU.operation(ctx.project, setup.op_index))
        c = setup.shape.corners_S() @ T_H_S[:3, :3].T + T_H_S[:3, 3]
        moved = c @ ps.slip_pose[:3, :3].T + ps.slip_pose[:3, 3]
        return float(np.max(np.linalg.norm(moved - c, axis=1)))

    def _set_free(self, ps, cause, tick) -> None:
        ps.free = True
        ps.free_cause = cause
        ps.free_tick = int(tick)

    def _free(self, ctx, state, hs, ps, setup, cause, tick, parents, q) -> None:
        self._set_free(ps, cause, tick)
        ev = ctx.trace.record(tick * ctx.tick_s, tick, DOMAIN, "holding:part_free", dict(q), parents)
        hs.events.append({"kind": "part_free", "workpiece": ps.workpiece, "tick": int(tick), "cause": cause})
        self._finding(ctx, state, hs, ps.counters, f"free:{ps.workpiece}", Category.PART_FREE, Severity.ERROR,
                      f"{setup.op_name}: the part comes free ({cause})", tick, 0, dict(q), [ev])

    def _setup_findings(self, ctx, state, hs, ps, setup, prev) -> None:
        tick = int(prev.motion.tick)
        main = setup.group("main")
        loc = main.location if main is not None else np.zeros(6)
        T_S_H = np.linalg.inv(setup.T_H_S)
        if main is not None:
            ps.location_pose = SU.twist_matrix(loc, main.body.centroid_H)
            ps.location_error_S = T_S_H[:3, :3] @ loc[:3]
            ps.seated = bool(main.seated)
            ps.seat_gap_m = float(main.seat_gap_m)
        clamp_force = sum(float(d.get("clamp_force_N", 0.0) + d.get("gripping_force_N", 0.0) + d.get("holddown_N", 0.0))
                          for d in setup.devices)
        q = {"workpiece": ps.workpiece, "op_index": setup.op_index, "devices": len(setup.devices),
             "contacts": sum(g.n for g in setup.groups), "clamp_force_N": clamp_force,
             "location_error_m": float(np.linalg.norm(ps.location_error_S)), "seat_gap_m": ps.seat_gap_m,
             "mass_kg": main.body.mass_kg if main is not None else 0.0}
        ev = ctx.trace.record(tick * ctx.tick_s, tick, DOMAIN, "holding:setup", q, [])
        hs.trace_notes[f"setup:{ps.workpiece}"] = int(ev)
        summary = {"op": setup.op_name, "devices": [{k: v for k, v in d.items() if not k.startswith("_")}
                                                     for d in setup.devices], "notes": list(setup.notes),
                   "debris_m": dict(ps.debris), "clamp_force_N": clamp_force, "lathe": setup.lathe}
        ps.flags["summary"] = summary
        for note in setup.notes:
            ctx.trace.record(tick * ctx.tick_s, tick, DOMAIN, "holding:setup_note", {"op_index": setup.op_index}, [ev])
            if note not in hs.notes:
                hs.notes.append(note)
        if setup.empty:
            return
        debris_events = []
        for group in setup.groups:
            for pid, (node, h) in sorted(group.debris_node.items()):
                debris_events.append(ctx.trace.record(tick * ctx.tick_s, tick, DOMAIN, "holding:debris_under_locator",
                                                      {"height_m": h, "node": float(node)}, [ev]))
        for d in setup.devices:
            if d["type"] == "magnetic" and float(d.get("material_factor", 1.0)) <= 0.0:
                self._finding(ctx, state, hs, ps.counters, f"magnet:{ps.workpiece}", Category.PART_FREE,
                              Severity.ERROR, f"{setup.op_name}: the work material is not ferromagnetic: the magnetic "
                              f"chuck {d['device']} does not hold it (only its weight and friction do)", tick, 0,
                              {"material_factor": 0.0}, [ev])
        if setup.no_grip and (main is None or main.n == 0):
            self._free(ctx, state, hs, ps, setup, f"the workholding ({', '.join(setup.no_grip)}) does not reach the "
                       f"part", tick, [ev], q)
            return
        if main is not None and (not main.held or not self._holds_weight(main)):
            self._free(ctx, state, hs, ps, setup, "the setup cannot hold the part", tick, [ev], q)
            return
        if self._grip_lost(setup):
            self._free(ctx, state, hs, ps, setup, "the gripping force is zero at the spindle speed", tick, [ev], q)
            return
        err = float(np.linalg.norm(ps.location_error_S) + (main.body.l_c_m if main else 0.0) *
                    float(np.linalg.norm(loc[3:])))
        limit = P.p(P.WORK, "mislocation_error_m")
        tol = _tightest_tolerance(ctx.project)
        if tol is not None:
            limit = min(limit, tol)
        if err > P.p(P.WORK, "mislocation_warn_m") or err > limit:
            sev = Severity.ERROR if err > limit else Severity.WARNING
            qq = {"location_error_m": err, "x_S_m": float(ps.location_error_S[0]), "y_S_m": float(ps.location_error_S[1]),
                  "z_S_m": float(ps.location_error_S[2]), "rotation_rad": float(np.linalg.norm(loc[3:])),
                  "limit_m": limit}
            self._finding(ctx, state, hs, ps.counters, f"mislocation:{ps.workpiece}", Category.MISLOCATION, sev,
                          f"{setup.op_name}: chips or burrs under a locating face put the part "
                          f"{err * 1e3:.3f} mm off its nominal location", tick, 0, qq, debris_events or [ev])
        if not ps.seated:
            self._finding(ctx, state, hs, ps.counters, f"seat:{ps.workpiece}", Category.MISLOCATION, Severity.WARNING,
                          f"{setup.op_name}: the part is not seated on its primary datum (gap "
                          f"{ps.seat_gap_m * 1e3:.3f} mm)", tick, 0, {"seat_gap_m": ps.seat_gap_m},
                          debris_events or [ev])

    # ------------------------------------------------------------------ the record
    def _record(self, ps, setup, extra, support_rows) -> PartHolding:
        rec = PartHolding(pose_delta=ps.slip_pose @ ps.location_pose, free=bool(ps.free), seated=bool(ps.seated),
                          location_error_S_m=np.array(ps.location_error_S, dtype=float), supports=support_rows,
                          slip_pose_delta=np.array(ps.slip_pose, dtype=float),
                          location_pose_delta=np.array(ps.location_pose, dtype=float), seat_gap_m=ps.seat_gap_m,
                          supports_version=ps.supports_version, free_cause=ps.free_cause)
        if setup is not None:
            main = setup.group("main")
            rec.lathe = dict(setup.lathe)
            rec.clamp_force_N = float(sum(float(d.get("clamp_force_N", 0.0) + d.get("gripping_force_N", 0.0)
                                                + d.get("holddown_N", 0.0)) for d in setup.devices))
            if main is not None and main.n:
                T_S_H = np.linalg.inv(setup.T_H_S)
                R = T_S_H[:3, :3]
                Tb = np.zeros((6, 6))
                Tb[:3, :3] = R
                Tb[3:, 3:] = R
                rec.support_stiffness_S = Tb @ main.k_support_H @ Tb.T
                rec.centroid_S_m = R @ main.body.centroid_H + T_S_H[:3, 3]
        if extra is not None:
            rec.margin = float(extra["margin"])
            rec.mode = str(extra["mode"])
            rec.slip_m = float(extra["slip_m"])
            rec.lift_m = float(extra["lift_m"])
            rec.slip_friction_J = float(extra["loss_J"])
            rec.slip_in_J = float(extra["in_J"])
            rec.slip_mech_change_J = float(extra["mech_J"])
            ps.friction_J += rec.slip_friction_J
        if ps.free:
            rec.supports = []
            rec.margin = 0.0
        return rec

    def _values(self, hs, prev, tools, parts) -> None:
        nan = float("nan")
        values = {}
        active = None
        for frame in prev.motion.paths:
            if int(frame.tool_slot) in tools:
                active = int(frame.tool_slot)
                break
        if active is not None:
            rec = tools[active]
            values.update({
                "holding.tool_margin": rec.holding_margin if math.isfinite(rec.holding_margin) else nan,
                "holding.tool_torque_capacity_Nm": rec.torque_capacity_Nm if math.isfinite(rec.torque_capacity_Nm)
                else nan,
                "holding.tool_pullout_m": rec.pullout_m, "holding.tool_slip_rad": rec.slip_rad,
                "holding.tool_walk_per_rev_m": rec.walk_per_rev_m, "holding.unbalance_force_N": rec.unbalance_force_N,
                "holding.balance_quality_m_s": rec.balance_quality_m_s})
        else:
            for name in ("holding.tool_margin", "holding.tool_torque_capacity_Nm", "holding.tool_pullout_m",
                         "holding.tool_slip_rad", "holding.tool_walk_per_rev_m", "holding.unbalance_force_N",
                         "holding.balance_quality_m_s"):
                values[name] = nan
        per_slot = np.full(MAX_SLOTS, nan)
        for slot, rec in tools.items():
            if slot < MAX_SLOTS:
                per_slot[slot] = rec.pullout_m
        values["holding.pullout_slot_m"] = per_slot
        cols = {name: np.full(MAX_PARTS, nan) for name in ("holding.part_margin", "holding.part_slip_m",
                                                          "holding.part_lift_m", "holding.location_error_m",
                                                          "holding.clamp_force_N")}
        for w, rec in parts.items():
            if w >= MAX_PARTS:
                continue
            ps = hs.parts[w]
            cols["holding.part_margin"][w] = rec.margin if math.isfinite(rec.margin) else nan
            cols["holding.part_slip_m"][w] = ps.max_slip_m
            cols["holding.part_lift_m"][w] = rec.lift_m
            cols["holding.location_error_m"][w] = float(np.linalg.norm(rec.location_error_S_m))
            cols["holding.clamp_force_N"][w] = rec.clamp_force_N
        values.update(cols)
        hs.values = values


__all__ = ["HoldingDomain", "gravity_M"]
