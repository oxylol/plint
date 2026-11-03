"""A small driver of the cutting kernels over a prescribed path (docs/models/cutting.md "Verification").

The simulator's hot loop (core.hotloop.run_macro) composes the kernels with motion, removal and dynamics;
this driver replaces those with prescribed inputs so the force models can be checked against closed
forms and validation data in isolation:

* kinematics: constant spindle speed, the tool tip on a straight (or helical) path in S, tool axis along
  S z (or a given rotation);
* engaged intervals: a callable per step (straight-path immersion, full circles below a plane ...);
* memory: prefilled with the surface the previous tooth left on a steady path, then warmed up;
* deflection: optional callable u(t) per station (regeneration checks);
* turning: a polar radius view updated after each step by the swept tool profile (a stand-in for
  removal's polar update).

It records the step wrenches and the element chip thicknesses for the checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import abi as A
from .kernels import cut_step, turn_step

TWO_PI = 2.0 * math.pi


@dataclass
class Record:
    t_s: list = field(default_factory=list)
    wrench: list = field(default_factory=list)       # [6] step means (T0)
    force_S: list = field(default_factory=list)
    torque: list = field(default_factory=list)
    h: list = field(default_factory=list)            # [n] element h at the step midpoint (h_last)
    station: list = field(default_factory=list)


def straight_immersion(phi_st_rad: float, phi_ex_rad: float, feed_angle_rad: float = 0.0):
    """Engaged interval of a straight-path cut in T0 right-handed angles: Altintas' immersion angle phi is
    measured clockwise from the feed direction turned by +90 deg, so alpha = feed_angle + pi/2 - phi."""
    lo = feed_angle_rad + 0.5 * math.pi - phi_ex_rad
    hi = feed_angle_rad + 0.5 * math.pi - phi_st_rad
    lo_w = lo % TWO_PI
    return lo_w, lo_w + (hi - lo)


class Driver:
    """Runs cut_step / turn_step for one contact over n steps of a prescribed motion."""

    def __init__(self, tool: A.CutTool, table: A.CutTable, n_teeth: int, n_elem: int, *, memory_bins: int = 720,
                 rotating: bool = True, polar: A.PolarView | None = None, event_capacity: int = 20000,
                 n_axes: int = 3, n_ticks: int = 1, rho_kg_m3: float = 7850.0, tick_s: float = 1e-6):
        self.tool = tool
        self.table = table
        self.nt = n_teeth
        self.ne = n_elem
        self.nst = tool.st_z_m.shape[0]
        self.n_rows = A.memory_rows(tool)
        self.rotating = rotating
        self.mem = A.empty_memory(self.n_rows, memory_bins)
        self.iv = A.empty_intervals(self.nst, 4)
        self.refill = A.empty_refill(self.n_rows, memory_bins)
        self.polar = polar if polar is not None else A.empty_polar()
        self.state = A.empty_state(n_teeth * n_elem, n_teeth, self.nst)
        nsec = tool.sec_z_m.shape[0]
        self.acc = A.core_accum(n_ticks, n_axes, n_teeth, n_elem)
        self.cacc = A.empty_accum(nsec)
        self.out = A.empty_out(self.nst, nsec)
        self.ev = A.empty_events(event_capacity)
        self.pose = A.identity_pose()
        self.defl = A.empty_deflection(self.nst)
        self.occ = A.empty_occupancy()
        self.pack = A.empty_packing()
        self.axes = A.empty_axes(n_axes)
        self.step = np.zeros(A.N_ST)
        self.istep = np.zeros(A.N_SI, np.int64)
        self.step[A.ST_TICK_S] = tick_s
        self.step[A.ST_RHO_KG_M3] = rho_kg_m3
        self.istep[A.SI_COMMIT] = 1
        self.istep[A.SI_STALE_TICKS] = 1 << 60
        self.istep[A.SI_STALE_BEFORE] = -(1 << 60)
        self.tick_s = tick_s
        self.tick = 0
        self.minor_side = 0
        self._polar_last = None
        self.record = Record()

    # ---------------------------------------------------------------- set-up
    def set_full_circle(self, stations=None) -> None:
        idx = range(self.nst) if stations is None else stations
        for s in idx:
            self.iv.lo_rad[s, 0] = 0.0
            self.iv.hi_rad[s, 0] = TWO_PI
            self.iv.count[s] = 1

    def set_interval(self, lo_rad: float, hi_rad: float, stations=None) -> None:
        idx = range(self.nst) if stations is None else stations
        for s in idx:
            self.iv.lo_rad[s, 0] = lo_rad % TWO_PI
            self.iv.hi_rad[s, 0] = lo_rad % TWO_PI + (hi_rad - lo_rad)
            self.iv.count[s] = 1

    def prefill_memory(self, tip_fn, t0_s: float, omega_rad_s: float, R_S_T0: np.ndarray | None = None,
                       theta0_rad: float | None = None) -> None:
        """Steady-state memory on a path: bin b of memory row s holds the point the element writing that row
        that passed the bin angle most recently left there, at the time it passed (exact for any pitch, runout,
        lead shift or radius of the teeth; without rotation the tool's current position). Rows nobody writes
        (a tap's handover rows without a predecessor) stay empty: their reads meet the field."""
        R = np.eye(3) if R_S_T0 is None else np.asarray(R_S_T0, float)
        nb = self.mem.point_S_m.shape[1]
        theta0 = omega_rad_s * t0_s if theta0_rad is None else theta0_rad
        ang = (np.arange(nb) + 0.5) * TWO_PI / nb
        rev = TWO_PI / abs(omega_rad_s) if omega_rad_s != 0.0 else 0.0
        self.mem.valid[:] = 0
        valid = self.tool.valid != 0
        for s in range(self.n_rows):
            elems = [int(e) for e in np.nonzero(valid & (self.tool.write_row == s))[0]]
            if not elems:
                continue
            for b in range(nb):
                if omega_rad_s == 0.0:
                    e = elems[0]
                    t_pass = t0_s
                    theta = theta0
                else:
                    # time since each element last passed the bin angle; the most recent one wrote the bin
                    since = [np.mod((theta0 + self.tool.alpha0_rad[e] - ang[b]) / omega_rad_s, rev) for e in elems]
                    k = int(np.argmin(since))
                    e = elems[k]
                    t_pass = t0_s - since[k]
                    theta = theta0 - omega_rad_s * since[k]
                c, sn = math.cos(theta), math.sin(theta)
                p = self.tool.p_T_m[e]
                p0 = np.array([c * p[0] - sn * p[1], sn * p[0] + c * p[1], p[2]])
                self.mem.point_S_m[s, b] = np.asarray(tip_fn(t_pass), float) + R @ p0
            self.mem.valid[s, :] = 1
        self.mem.tick[:] = int(round(t0_s / self.tick_s))

    def set_planes(self, stock, tip_S_m=None, R_S_T0=None, theta_rad: float = 0.0, workpiece: int = 0) -> int:
        """Field planes of every memory row and bin from an analytic stock (memory.SurfaceQuery) at the given
        tool pose: what removal's query provides at L2 for stale bins. Exact for stocks whose boundary is
        invariant along the tool's motion (a bore along the tool axis, a plane parallel to the path)."""
        from .memory import compute_planes
        nb = self.refill.valid.shape[1]
        pairs = [(r, b) for r in range(self.n_rows) for b in range(nb)]
        tip = np.zeros(3) if tip_S_m is None else np.asarray(tip_S_m, float)
        R = np.eye(3) if R_S_T0 is None else np.asarray(R_S_T0, float)
        reach = 4.0 * float(self.tool.scal[A.TC_R_CUT]) + 1e-3
        return compute_planes(stock, workpiece, self.tool, self.refill, pairs, R, tip, theta_rad, reach)

    # ---------------------------------------------------------------- stepping
    def run_rotating(self, n_steps: int, omega_rad_s: float, dt_s: float, tip_fn, theta0_rad: float | None = None,
                     t0_s: float = 0.0, intervals_fn=None, deflection_fn=None, v_tip_fn=None, R_S_T0=None,
                     record: bool = True, commit: bool = True) -> Record:
        """Constant spindle speed: theta(t) = theta0 + omega (t - t0), theta0 = omega t0 by default (so runs
        continue each other without an angle jump)."""
        if theta0_rad is None:
            theta0_rad = omega_rad_s * t0_s
        l0 = int(round(dt_s / self.tick_s))
        self.step[A.ST_DT_S] = dt_s
        self.step[A.ST_OMEGA_RAD_S] = omega_rad_s
        self.step[A.ST_TICK_FRAC] = 0.0
        self.step[A.ST_TOOTH_PERIOD_S] = TWO_PI / (self.nt * abs(omega_rad_s)) if omega_rad_s else 0.0
        self.istep[A.SI_L0_TICKS] = l0
        self.istep[A.SI_COMMIT] = 1 if commit else 0
        if R_S_T0 is not None:
            self.pose.R_S_T0[:] = R_S_T0
        h = 1e-6 * dt_s
        for k in range(n_steps):
            t_end = t0_s + (k + 1) * dt_s
            t_mid = t_end - 0.5 * dt_s
            self.tick = int(round(t_end / self.tick_s))
            theta = (theta0_rad + omega_rad_s * (t_mid - t0_s)) % TWO_PI
            self.step[A.ST_THETA_MID_RAD] = theta
            self.step[A.ST_COS_MID] = math.cos(theta)
            self.step[A.ST_SIN_MID] = math.sin(theta)
            self.istep[A.SI_TICK_END] = self.tick
            tip = np.asarray(tip_fn(t_mid), float)
            self.pose.tip_S_m[:] = tip
            if v_tip_fn is not None:
                self.pose.v_tip_S_m_s[:] = v_tip_fn(t_mid)
            else:
                self.pose.v_tip_S_m_s[:] = (np.asarray(tip_fn(t_mid + h)) - np.asarray(tip_fn(t_mid - h))) / (2 * h)
            if intervals_fn is not None:
                intervals_fn(self, t_mid, theta, tip)
            if deflection_fn is not None:
                deflection_fn(self, t_mid)
            cut_step(self.tool, self.table, self.mem, self.iv, self.refill, self.step, self.istep, self.pose, self.defl,
                     self.occ, self.pack, self.axes, self.state, self.acc, self.cacc, self.out, self.ev)
            if record:
                self._record(t_mid)
        return self.record

    def run_turning(self, n_steps: int, omega_rad_s: float, dt_s: float, pose_fn, t0_s: float = 0.0,
                    record: bool = True, update_field: bool = True) -> Record:
        """pose_fn(t) -> (R_S_T, tip_S, v_tip_S, w_T_S): the tool in the rotating work frame S."""
        l0 = int(round(dt_s / self.tick_s))
        self.step[A.ST_DT_S] = dt_s
        self.step[A.ST_OMEGA_RAD_S] = omega_rad_s
        self.step[A.ST_TOOTH_PERIOD_S] = TWO_PI / abs(omega_rad_s) if omega_rad_s else 0.0
        self.istep[A.SI_L0_TICKS] = l0
        for k in range(n_steps):
            t_end = t0_s + (k + 1) * dt_s
            t_mid = t_end - 0.5 * dt_s
            self.tick = int(round(t_end / self.tick_s))
            self.istep[A.SI_TICK_END] = self.tick
            R, tip, v, w = pose_fn(t_mid)
            self.pose.R_S_T0[:] = R
            self.pose.tip_S_m[:] = tip
            self.pose.v_tip_S_m_s[:] = v
            self.pose.w_T0_S_rad_s[:] = w
            turn_step(self.tool, self.table, self.polar, self.step, self.istep, self.pose, self.defl, self.axes,
                      self.state, self.acc, self.cacc, self.out, self.ev)
            if record:
                self._record(t_mid)
            if update_field:
                self.sweep_polar()
        return self.record

    def sweep_polar(self) -> None:
        """Stand-in for removal's polar update: a bin the tool has left takes the tool profile of its last
        step in that bin (the edge elements' radius as a function of z, linear between elements, through the
        tool tip), so the bin under the tool still holds the previous revolution (the polar view requirement
        of docs/models/cutting.md "Turning"). With ``minor_side`` = +1 / -1 the tool's minor edge is flat
        (kappa_r' = 0) on that side of the tip along z: the surface behind the tip is left at the tip radius."""
        lp = self.state.last_p
        valid = [e for e in range(self.tool.p_T_m.shape[0]) if self.tool.valid[e]]
        if not valid:
            return
        tip = self.pose.tip_S_m
        r_tip = float(math.hypot(tip[0], tip[1]))
        pts = sorted([(float(lp[e, 2]), float(math.hypot(lp[e, 0], lp[e, 1])), float(math.atan2(lp[e, 1], lp[e, 0])))
                      for e in valid] + [(float(tip[2]), r_tip, float(math.atan2(tip[1], tip[0])))])
        z = np.array([p[0] for p in pts])
        r = np.array([p[1] for p in pts])
        th = float(math.atan2(np.mean([math.sin(p[2]) for p in pts]), np.mean([math.cos(p[2]) for p in pts]))) % TWO_PI
        dth, z0, dz = self.polar.geom[0], self.polar.geom[1], self.polar.geom[2]
        nth, nz = self.polar.r_out_m.shape
        b = int(th / dth) % nth
        zc = z0 + (np.arange(nz) + 0.5) * dz
        inside = (zc >= z.min()) & (zc <= z.max())
        rt = np.interp(zc, z, r) if z.max() > z.min() else np.full(nz, r.min())
        if self.minor_side != 0:
            behind = (zc - float(tip[2])) * self.minor_side > 0.0
            rt = np.where(behind, r_tip, rt)
            inside |= behind
        last = self._polar_last
        self._polar_last = (b, inside, rt)
        if last is None or last[0] == b:
            return
        b, inside, rt = last
        row = self.polar.r_out_m[b]
        row[inside] = np.minimum(row[inside], rt[inside])

    def _record(self, t_mid: float) -> None:
        r = self.record
        r.t_s.append(t_mid)
        r.wrench.append(self.out.wrench_T0.copy())
        r.force_S.append(self.out.force_S_N.copy())
        r.torque.append(float(self.out.misc[A.SO_TORQUE_NM]))
        r.h.append(self.state.h_last_m.copy())
        r.station.append(self.out.station_T0.copy())


__all__ = ["Driver", "Record", "straight_immersion"]
