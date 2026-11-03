"""Spindle commands and their timing on the planner side (docs/models/motion.md "Spindles").

The compiled loop runs the spindle dynamics (speed loop with the ramp and CSS, position loop); the
planner keeps the *commanded* state of every spindle (mode, signed target speed, CSS surface speed
and clamp, gear, position command) as a list of changes at L1 ticks, and predicts times it must wait
for: speed arrival before a cutting feed (|omega - omega_c| <= arrival band x |omega_c|), stopping
before an orientation, a gear change or C-axis mode. The prediction integrates

    d|omega|/dt = min(ramp rate, (T_max(|omega|) -+ T_drag(|omega|)) / J)

(accelerating: torque limit minus drag; decelerating: plus drag) from the actual speed at the start
of the macro step, in 1 ms steps. Its error against the loop is the PI settling of the speed loop
(about 1 / omega_b, 8 ms at 20 Hz) and the cutting torque, which is zero before a cut starts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .abi import SPM_CAXIS, SPM_POSITION, SPM_SPEED

_STEP_S = 1e-3


@dataclass
class SpindleCmd:
    mode: int = SPM_SPEED
    omega: float = 0.0              # signed target speed (M03 < 0)
    css_vc: float = 0.0             # m/s (0 = off)
    clamp: float = math.inf         # rad/s
    gear: int = 0
    angle: float = 0.0              # position command (SPM_POSITION)
    omega_ff: float = 0.0
    owner: int = -1                 # path that commanded it last

    def copy(self) -> "SpindleCmd":
        return SpindleCmd(self.mode, self.omega, self.css_vc, self.clamp, self.gear, self.angle, self.omega_ff,
                          self.owner)

    def to_dict(self) -> dict:
        return {"mode": self.mode, "omega": self.omega, "css_vc": self.css_vc,
                "clamp": self.clamp if math.isfinite(self.clamp) else None, "gear": self.gear, "angle": self.angle,
                "omega_ff": self.omega_ff, "owner": self.owner}

    @classmethod
    def from_dict(cls, d: dict) -> "SpindleCmd":
        clamp = d.get("clamp")
        return cls(int(d["mode"]), float(d["omega"]), float(d["css_vc"]), math.inf if clamp is None else float(clamp),
                   int(d["gear"]), float(d["angle"]), float(d["omega_ff"]), int(d["owner"]))


def drag_torque(sp, omega: float) -> float:
    if sp.max_omega <= 0:
        return 0.0
    return sp.drag_max * (abs(omega) / sp.max_omega) ** (2.0 / 3.0)


def speed_change_s(sp, omega0: float, target: float, gear: int, band: float = 0.0) -> float:
    """Predicted time from |omega0| to within band x |target| of target (signs: a reversal passes 0)."""
    if sp is None:
        return 0.0
    if omega0 * target < 0.0:
        return speed_change_s(sp, omega0, 0.0, gear) + speed_change_s(sp, 0.0, target, gear, band)
    w = abs(omega0)
    goal = abs(target)
    tol = band * goal
    if abs(w - goal) <= tol + 1e-12:
        return 0.0
    inertia = max(sp.inertia, 1e-12)
    t = 0.0
    up = goal > w
    rate = sp.accel_rate if up else sp.decel_rate
    for _ in range(600000):
        torque = sp.torque_at(w, gear)
        drag = drag_torque(sp, w)
        acc = (torque - drag) / inertia if up else (torque + drag) / inertia
        acc = min(acc, rate)
        if acc <= 0.0:
            return math.inf
        step = acc * _STEP_S
        if up:
            if w + step >= goal - tol:
                t += max(0.0, (goal - tol - w) / acc)
                return t
            w += step
        else:
            if w - step <= goal + tol:
                t += max(0.0, (w - goal - tol) / acc)
                return t
            w -= step
        t += _STEP_S
    return t


def evolve_speed(sp, omega0: float, target: float, gear: int, seconds: float) -> float:
    """Predicted speed after ``seconds`` from omega0 toward target (same model as speed_change_s)."""
    if seconds <= 0.0 or omega0 == target:
        return omega0
    if sp is None:
        return target
    w = omega0
    t = 0.0
    inertia = max(sp.inertia, 1e-12)
    for _ in range(int(math.ceil(seconds / _STEP_S)) + 1):
        if t >= seconds - 1e-15:
            break
        h = min(_STEP_S, seconds - t)
        if w * target < 0.0 or abs(target) < abs(w):
            # decelerating (toward zero first on a reversal)
            goal = 0.0 if w * target < 0.0 else abs(target)
            acc = min((sp.torque_at(w, gear) + drag_torque(sp, w)) / inertia, sp.decel_rate)
            mag = max(abs(w) - acc * h, goal)
            w = math.copysign(mag, w) if mag > 0.0 else (0.0 if goal == 0.0 else math.copysign(mag, w))
        else:
            goal = abs(target)
            acc = min((sp.torque_at(w, gear) - drag_torque(sp, w)) / inertia, sp.accel_rate)
            if acc <= 0.0:
                break
            mag = min(abs(w) + acc * h, goal)
            w = math.copysign(mag, target) if target != 0.0 else 0.0
        t += h
        if w == target:
            break
    return w


def css_omega(cmd: SpindleCmd, radius: float, sp, r_min: float) -> float:
    """Spindle speed of the constant-surface-speed law at a radius (clamped), signed like the command."""
    if cmd.css_vc <= 0.0 or cmd.omega == 0.0:
        return cmd.omega
    w = cmd.css_vc / max(abs(radius), r_min)
    w = min(w, cmd.clamp, sp.max_omega if sp is not None and sp.max_omega > 0 else math.inf)
    return w if cmd.omega > 0 else -w


@dataclass
class SpindleTimeline:
    """Commanded spindle states within one macro step: changes (tick index, spindle, SpindleCmd)."""

    start: list = field(default_factory=list)       # SpindleCmd per spindle at the step start (k_from)
    changes: list = field(default_factory=list)     # (k, spindle index, SpindleCmd)

    def fill(self, setp, k_from: int, n: int, angle_cmds: dict | None = None) -> None:
        """Write the spindle columns of Setpoint rows k_from .. n-1."""
        current = [c.copy() for c in self.start]
        changes = sorted(self.changes, key=lambda c: (c[0], c[1]))
        j = 0
        for k in range(k_from, n):
            while j < len(changes) and changes[j][0] <= k:
                _, s, cmd = changes[j]
                current[s] = cmd.copy()
                j += 1
            for s, cmd in enumerate(current):
                setp.sp_mode[k, s] = cmd.mode
                setp.sp_omega[k, s] = cmd.omega
                setp.css_vc[k, s] = cmd.css_vc
                setp.css_clamp[k, s] = cmd.clamp
                if cmd.mode == SPM_POSITION:
                    if angle_cmds is not None and (k, s) in angle_cmds:
                        setp.sp_angle[k, s], setp.sp_omega_ff[k, s] = angle_cmds[(k, s)]
                    else:
                        setp.sp_angle[k, s] = cmd.angle
                        setp.sp_omega_ff[k, s] = cmd.omega_ff
                elif cmd.mode == SPM_CAXIS:
                    setp.sp_angle[k, s] = cmd.angle
                    setp.sp_omega_ff[k, s] = 0.0
