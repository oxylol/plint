"""Kinematics of one path: FK through the compiled chain, closed-form and numeric inverse kinematics,
singularity detection and the tilted-plane rotary solutions (docs/models/motion.md "Kinematics").

**Orientation** (closed form for every 5-axis type: table A/C, table B/C, head B/C, head A/B,
head-table B/C, mill-turn B head with C in C-axis mode, nutating heads): the tool axis in the
workpiece frame W depends only on the rotary joints of the closed loop W <- workpiece chain <- M ->
tool chain -> T. With the joint axes omega_j expressed in W at the zero configuration and the joints
ordered from W (workpiece-chain joints traversed in reverse with angle -theta, tool-chain joints
forward with +theta), the product of exponentials gives

    z_W(theta) = exp(omega_1^ phi_1) exp(omega_2^ phi_2) z_W(0)          (Murray, Li & Sastry 1994 ch. 3)

so two orientation joints are Paden-Kahan subproblem 2 (two solutions, one, or none) and one is
subproblem 1. **Position**: with the rotary values fixed the tip is affine in the linear axes
(up to the tiny angular error motions), solved by Newton steps on the 3x3 linear system.
Machines with a different number of orientation joints, or positions the closed form cannot place,
use damped Gauss-Newton on the full residual from the reference configuration (deterministic).

**Singularity**: a joint whose axis is parallel to the vector it must turn (tool axis parallel to a
rotary axis, within ``singularity_cone_rad``) is undetermined; it keeps its reference value and the
solve reports ``singular``.

**Choice among solutions**: ``nearest`` (smallest weighted joint motion from the reference; used per
tick in TCP interpolation), ``positive`` / ``negative`` (sign of the master rotary axis: Heidenhain
PLANE SEQ+/SEQ-, Siemens CYCLE800 direction +/-, FANUC G53.1 rotary direction), ``travel_shortest``
(Heidenhain without SEQ: the shortest solution within the travel, else the one within travel). The
master axis is the orientation joint nearest the machine base in the tool chain, else in the
workpiece chain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .abi import CHAIN_TOOL, CHAIN_WORK, FK_M_T0, FK_M_W, LINK_ROTARY, N_FK
from .kernels import fk_poses

TWO_PI = 2.0 * math.pi
RULES = ("nearest", "positive", "negative", "travel_shortest")
# below this perpendicular component a subproblem is exactly degenerate (numerics); the singularity
# cone of the model parameters only flags near-singular configurations
_DEGENERATE = 1e-10


@dataclass
class Joint:
    axis: int          # global axis index
    side: int          # CHAIN_WORK / CHAIN_TOOL
    link: int
    sign: float        # phi = sign * theta (-1 for workpiece-chain joints)
    omega_W: np.ndarray


@dataclass
class IKResult:
    q: np.ndarray
    ok: bool
    singular: bool
    residual_m: float
    solutions: int


def sp1(omega: np.ndarray, u: np.ndarray, v: np.ndarray, eps: float) -> tuple[float, bool]:
    """Paden-Kahan subproblem 1: angle phi with exp(omega^ phi) u = v (directions). Degenerate when u is
    parallel to omega (returns 0, True)."""
    up = u - omega * (omega @ u)
    vp = v - omega * (omega @ v)
    nu = float(np.linalg.norm(up))
    nv = float(np.linalg.norm(vp))
    if nu < eps or nv < eps:
        return 0.0, True
    return math.atan2(float(omega @ np.cross(up, vp)), float(up @ vp)), False


def sp2(omega1, omega2, p, q, eps: float):
    """Paden-Kahan subproblem 2 for directions: all (phi1, phi2) with exp(w1 phi1) exp(w2 phi2) p = q.

    Returns a list of (phi1, phi2, degenerate1, degenerate2); empty if q is not reachable."""
    w12 = float(omega1 @ omega2)
    cross = np.cross(omega1, omega2)
    nc2 = float(cross @ cross)
    if nc2 < 1e-24:
        return []
    den = w12 * w12 - 1.0
    alpha = (w12 * float(omega2 @ p) - float(omega1 @ q)) / den
    beta = (w12 * float(omega1 @ q) - float(omega2 @ p)) / den
    g2 = (float(p @ p) - alpha * alpha - beta * beta - 2.0 * alpha * beta * w12) / nc2
    if g2 < -1e-9:
        return []
    g = math.sqrt(max(g2, 0.0))
    out = []
    for sign in ((1.0, -1.0) if g > 1e-12 else (1.0,)):
        c = alpha * omega1 + beta * omega2 + sign * g * cross
        phi2, d2 = sp1(omega2, p, c, eps)
        phi1, d1 = sp1(omega1, c, q, eps)
        out.append((phi1, phi2, d1, d2))
    return out


class PathKinematics:
    """FK/IK of one path on the compiled chain (the chain object may be replaced; see :meth:`bind`)."""

    def __init__(self, model, chain, p: int, cone_rad: float | None = None):
        self.model = model
        self.p = int(p)
        self.cone = float(cone_rad if cone_rad is not None else model.p["singularity_cone_rad"])
        self.n_axes = len(model.axes)
        self._out = np.zeros((N_FK, 4, 4))
        self._work = np.zeros((6, 4, 4))
        self.bind(chain)

    # ------------------------------------------------------------------ set-up
    def bind(self, chain) -> None:
        self.chain = chain
        p = self.p
        model = self.model
        lin = []
        for ax in model.path_axes(p):
            if not ax.rotary and not ax.synthetic and ax.letter in ("X", "Y", "Z"):
                lin.append(ax.index)
        self.linear = lin
        self._lin_arr = np.array(lin, np.int32)
        q0 = np.zeros(self.n_axes)
        poses = self.fk(q0)
        R_W0 = poses[FK_M_W][:3, :3]
        joints = []
        # workpiece chain joints from W toward M (reverse), then tool chain joints from M toward T
        for side in (CHAIN_WORK, CHAIN_TOOL):
            n = int(chain.n_links[p, side])
            order = range(n - 1, -1, -1) if side == CHAIN_WORK else range(n)
            for i in order:
                if chain.kind[p, side, i] != LINK_ROTARY:
                    continue
                a = int(chain.axis[p, side, i])
                if a < 0 or model.axes[a].synthetic:
                    continue
                R_parent = self._parent_rotation(side, i, q0)
                omega_M = R_parent @ chain.direction[p, side, i]
                omega_W = R_W0.T @ omega_M
                joints.append(Joint(a, side, i, -1.0 if side == CHAIN_WORK else 1.0, omega_W / np.linalg.norm(omega_W)))
        self.joints = joints
        self.z0_W = R_W0.T @ poses[FK_M_T0][:3, 2]
        tool = [j for j in joints if j.side == CHAIN_TOOL]
        work = [j for j in joints if j.side == CHAIN_WORK]
        self.master = min(tool, key=lambda j: j.link) if tool else (min(work, key=lambda j: j.link) if work else None)

    def _parent_rotation(self, side, i, q) -> np.ndarray:
        """Rotation of the parent frame of link i (chain side) at configuration q."""
        from .kernels import link_motion
        R = np.eye(4)
        m = np.zeros((4, 4))
        r = np.zeros((4, 4))
        t = np.zeros((4, 4))
        for k in range(i):
            link_motion(self.chain, self.p, side, k, q, -1, m, r, t)
            R = R @ m
        return R[:3, :3]

    # ------------------------------------------------------------------ forward
    def fk(self, q) -> np.ndarray:
        qq = np.ascontiguousarray(q, dtype=float)
        fk_poses(self.chain, self.p, qq, self._out, self._work)
        return self._out.copy()

    def tip_axis_W(self, q) -> tuple[np.ndarray, np.ndarray]:
        poses = self.fk(q)
        M_W = poses[FK_M_W]
        M_T = poses[FK_M_T0]
        R = M_W[:3, :3]
        return R.T @ (M_T[:3, 3] - M_W[:3, 3]), R.T @ M_T[:3, 2]

    # ------------------------------------------------------------------ inverse
    def solve(self, tip_W, axis_W, q_ref, rule: str = "nearest", position: bool = True) -> IKResult:
        """Axis values putting the tool tip at tip_W with the tool axis along axis_W (both in W).

        Axes not involved keep their q_ref values. ``position=False`` solves the orientation only."""
        if rule not in RULES:
            raise ValueError(f"IK rule {rule!r} is not one of {', '.join(RULES)}")
        q_ref = np.asarray(q_ref, dtype=float)
        tip_W = np.asarray(tip_W, dtype=float)
        target = np.asarray(axis_W, dtype=float)
        target = target / np.linalg.norm(target)
        candidates, singular, n_solutions = self._orientations(target, q_ref, rule)
        if not candidates:
            return self._numeric(tip_W, target, q_ref, singular)
        best = None
        for q in candidates:
            if position:
                q, res = self._positions(q, tip_W)
            else:
                res = 0.0
            ok = res < 1e-9 and self._within_travel(q)
            key = (0 if ok else 1, self._rule_key(q, q_ref, rule))
            if best is None or key < best[0]:
                best = (key, q, res, ok)
        _, q, res, ok = best
        if position and res >= 1e-9:
            num = self._numeric(tip_W, target, q, singular)
            if num.residual_m < res:
                return num
        return IKResult(q, ok, singular, res, n_solutions)

    def solve_position(self, tip_W, q) -> IKResult:
        """Linear axes for the tip at tip_W with every rotary axis as given in q (TCP with programmed rotary
        positions: the rotary axes are interpolated, the linear axes keep the tip on its path). Compiled
        Newton (cnctwin.motion.kernels.ik_position)."""
        from .kernels import ik_position
        q2 = np.ascontiguousarray(q, dtype=float).copy()
        res = ik_position(self.chain, self.p, q2, np.ascontiguousarray(tip_W, dtype=float), self._lin_arr, 6,
                          self._out, self._work)
        return IKResult(q2, res < 1e-9 and self._within_travel(q2), False, res, 1)

    def _rule_key(self, q, q_ref, rule) -> tuple:
        motion = float(sum(abs(q[j.axis] - q_ref[j.axis]) for j in self.joints))
        if rule in ("positive", "negative") and self.master is not None:
            value = q[self.master.axis]
            good = value >= -1e-12 if rule == "positive" else value <= 1e-12
            return (0 if good else 1, motion)
        return (motion,)

    def _within_travel(self, q) -> bool:
        for j in self.joints:
            ax = self.model.axes[j.axis]
            if q[j.axis] < ax.lo - 1e-9 or q[j.axis] > ax.hi + 1e-9:
                return False
        return True

    def _orientations(self, target, q_ref, rule):
        """Candidate configurations (rotary joints set, other axes = q_ref), singular flag, count."""
        joints = self.joints
        if len(joints) == 0:
            return [q_ref.copy()], False, 1
        z0 = self.z0_W
        if len(joints) == 1:
            j = joints[0]
            phi, degenerate = sp1(j.omega_W, z0, target, _DEGENERATE)
            q = q_ref.copy()
            if not degenerate:
                q[j.axis] = self._nearest_turn(j, phi * j.sign, q_ref[j.axis])
            return [q], degenerate, 1
        if len(joints) != 2:
            return [], False, 0
        j1, j2 = joints
        sols = sp2(j1.omega_W, j2.omega_W, z0, target, _DEGENERATE)
        out = []
        singular = False
        for phi1, phi2, d1, d2 in sols:
            q = q_ref.copy()
            if d1:
                # the first joint is free (target along its axis): keep it, solve the second for the target
                singular = True
                phi1 = j1.sign * q_ref[j1.axis]
                phi2, _ = sp1(j2.omega_W, z0, _rot(j1.omega_W, -phi1) @ target, _DEGENERATE)
            elif d2:
                singular = True
                phi2 = j2.sign * q_ref[j2.axis]
                phi1, _ = sp1(j1.omega_W, _rot(j2.omega_W, phi2) @ z0, target, _DEGENERATE)
            q[j1.axis] = self._nearest_turn(j1, phi1 * j1.sign, q_ref[j1.axis])
            q[j2.axis] = self._nearest_turn(j2, phi2 * j2.sign, q_ref[j2.axis])
            out.extend(self._turn_variants(q, q_ref, rule))
        # within the cone around the first joint's axis (the one applied last to the tool axis) the
        # solution is singular even when the subproblem was regular: q . omega_1 = c . omega_1
        if abs(float(j1.omega_W @ target)) > math.cos(self.cone):
            singular = True
        return out, singular, len(sols)

    def _turn_variants(self, q, q_ref, rule) -> list[np.ndarray]:
        """For rules other than nearest, also the 2 pi shifted values of axes with room for them."""
        if rule == "nearest":
            return [q]
        out = [q]
        for j in self.joints:
            ax = self.model.axes[j.axis]
            for shift in (-TWO_PI, TWO_PI):
                v = q[j.axis] + shift
                if ax.lo - 1e-9 <= v <= ax.hi + 1e-9:
                    qq = q.copy()
                    qq[j.axis] = v
                    out.append(qq)
        return out

    def _nearest_turn(self, j: Joint, theta: float, ref: float) -> float:
        """theta + 2 pi k nearest ref, within the travel when possible."""
        ax = self.model.axes[j.axis]
        k = round((ref - theta) / TWO_PI)
        best = theta + TWO_PI * k
        if ax.lo - 1e-9 <= best <= ax.hi + 1e-9 or not (math.isfinite(ax.lo) and math.isfinite(ax.hi)):
            return best
        for kk in (k - 1, k + 1, k - 2, k + 2):
            v = theta + TWO_PI * kk
            if ax.lo - 1e-9 <= v <= ax.hi + 1e-9:
                return v
        return best

    def _positions(self, q, tip_W, iterations: int = 4) -> tuple[np.ndarray, float]:
        """Newton on the linear axes for the tip position (rotary values fixed)."""
        q = q.copy()
        lin = self.linear
        if not lin:
            tip, _ = self.tip_axis_W(q)
            return q, float(np.linalg.norm(tip - tip_W))
        res = math.inf
        for _ in range(iterations):
            tip, _ = self.tip_axis_W(q)
            r = tip - tip_W
            res = float(np.linalg.norm(r))
            if res < 1e-13:
                break
            G = np.zeros((3, len(lin)))
            h = 1e-6
            for c, a in enumerate(lin):
                qq = q.copy()
                qq[a] += h
                tip_h, _ = self.tip_axis_W(qq)
                G[:, c] = (tip_h - tip) / h
            step, *_ = np.linalg.lstsq(G, -r, rcond=None)
            for c, a in enumerate(lin):
                q[a] += step[c]
        tip, _ = self.tip_axis_W(q)
        return q, float(np.linalg.norm(tip - tip_W))

    def _numeric(self, tip_W, axis_W, q_ref, singular: bool, iterations: int = 60) -> IKResult:
        """Damped Gauss-Newton on (tip error, scaled axis error) over the path's linear and orientation axes."""
        unknowns = list(self.linear) + [j.axis for j in self.joints]
        q = np.asarray(q_ref, dtype=float).copy()
        if not unknowns:
            tip, _ = self.tip_axis_W(q)
            return IKResult(q, False, singular, float(np.linalg.norm(tip - tip_W)), 0)
        w_rot = self.model.rot_len
        lam = 1e-6

        def residual(qv):
            tip, ax = self.tip_axis_W(qv)
            return np.concatenate([tip - tip_W, w_rot * np.cross(ax, axis_W)])

        r = residual(q)
        for _ in range(iterations):
            if float(np.linalg.norm(r)) < 1e-12:
                break
            Jm = np.zeros((6, len(unknowns)))
            for c, a in enumerate(unknowns):
                qq = q.copy()
                h = 1e-7
                qq[a] += h
                Jm[:, c] = (residual(qq) - r) / h
            A = Jm.T @ Jm + lam * np.eye(len(unknowns))
            step = np.linalg.solve(A, -Jm.T @ r)
            q_new = q.copy()
            for c, a in enumerate(unknowns):
                q_new[a] += step[c]
            r_new = residual(q_new)
            if float(np.linalg.norm(r_new)) < float(np.linalg.norm(r)):
                q, r = q_new, r_new
                lam = max(lam * 0.3, 1e-12)
            else:
                lam *= 10.0
                if lam > 1e6:
                    break
        tip, ax = self.tip_axis_W(q)
        res = float(np.linalg.norm(tip - tip_W))
        ang = float(np.linalg.norm(np.cross(ax, axis_W)))
        return IKResult(q, res < 1e-9 and ang < 1e-9 and self._within_travel(q), singular, res, 1)

    # ------------------------------------------------------------------ singularities and tilted planes
    def singular_at(self, axis_W) -> list[int]:
        """Rotary axes (global index) whose line is within the cone of the tool axis axis_W (W)."""
        axis_W = np.asarray(axis_W, float) / np.linalg.norm(axis_W)
        out = []
        for j in self.joints:
            if abs(float(j.omega_W @ axis_W)) > math.cos(self.cone):
                out.append(j.axis)
        return out

    def tilted_plane(self, R_W_plane, q_ref, rule: str) -> IKResult:
        """Rotary values that orient the tool axis along the tilted plane's z (G68.2/G53.1, CYCLE800, PLANE),
        chosen by the control's rule; linear axes stay at q_ref."""
        z = np.asarray(R_W_plane, float)[:3, 2]
        return self.solve(np.zeros(3), z, q_ref, rule=rule, position=False)


def _rot(omega, phi) -> np.ndarray:
    from .reference import axis_rotation
    return axis_rotation(omega, phi)


def default_plane_rule(dialect: str) -> str:
    """The rotary-solution rule a control applies to a tilted plane without an explicit choice."""
    if dialect.startswith("heidenhain"):
        return "travel_shortest"
    if dialect == "siemens840d":
        return "positive"
    return "nearest"
