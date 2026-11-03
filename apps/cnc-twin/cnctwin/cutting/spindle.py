"""Spindle load against the machine's torque / power curves (docs/models/cutting.md "Spindle load").

* Available torque at speed and gear: T_av(omega) = min(T_curve(omega), P_curve(omega) / omega) of the gear
  range containing |omega| (the lowest such gear; machine.spindles[].gear_ranges[]).
* Rating and duty: a curve published as continuous (S1) is the thermal limit itself; an S6-d rating
  (cyclic duration factor d) is a multiple of it with P_S1 = P_S6 sqrt(d), and a short-time / "peak, t
  min" rating with P_S1 = P_t sqrt(1 - exp(-t / tau)) (equal winding temperature of a first-order
  thermal model with copper losses ~ T^2; duty types of IEC 60034-1). The thermal load
  theta (1 = the S1 limit) follows d theta / dt = ((T / T_S1)^2 - theta) / tau, integrated exactly
  over each macro step with the step's mean torque.
* Stall: when the cutting torque exceeds T_av the speed loop saturates and the spindle slows; at a
  constant feed rate the chip thickness grows as 1 / omega, so T_cut(omega) = T_0 (omega_0 / omega)^(1 - m)
  (Kienzle exponent m); the speed settles where T_cut = T_av if that is above stall_speed_ratio x
  omega_0, otherwise the spindle stalls. The deceleration time is J (omega_0 - omega_eq) / (T_cut - T_av).

Findings (core.findings Category.SPINDLE_OVERLOAD): error when the demand exceeds the available torque
(speed drop / stall), warning when the thermal load exceeds the S1 limit (derating by duty).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Gear:
    lo_rad_s: float
    hi_rad_s: float
    torque_x: np.ndarray
    torque_y: np.ndarray
    power_x: np.ndarray
    power_y: np.ndarray
    rating: str
    s1_factor: float

    def available(self, omega_abs: float) -> float:
        tq = float(np.interp(omega_abs, self.torque_x, self.torque_y)) if self.torque_x.size else math.inf
        if self.power_x.size and omega_abs > 0.0:
            tq = min(tq, float(np.interp(omega_abs, self.power_x, self.power_y)) / omega_abs)
        return tq


@dataclass
class Spindle:
    id: str
    gears: list
    inertia_kgm2: float
    max_rad_s: float
    notes: list = field(default_factory=list)

    def gear_for(self, omega_abs: float) -> int:
        for i, g in enumerate(self.gears):
            if g.lo_rad_s - 1e-9 <= omega_abs <= g.hi_rad_s + 1e-9:
                return i
        if not self.gears:
            return -1
        return int(np.argmin([min(abs(omega_abs - g.lo_rad_s), abs(omega_abs - g.hi_rad_s)) for g in self.gears]))

    def available(self, omega_abs: float, gear: int | None = None) -> float:
        g = self.gear_for(omega_abs) if gear is None or gear < 0 else gear
        if g < 0:
            return math.inf
        return self.gears[g].available(omega_abs)

    def s1_torque(self, omega_abs: float, gear: int | None = None) -> float:
        g = self.gear_for(omega_abs) if gear is None or gear < 0 else gear
        if g < 0:
            return math.inf
        return self.gears[g].available(omega_abs) * self.gears[g].s1_factor


def s1_factor(rating: str, tau_s: float, default_peak_s: float, default_s6: float) -> tuple[float, str]:
    """Continuous (S1) capacity / rated capacity of a curve from its rating text, and how it was read."""
    text = (rating or "").lower()
    if not text or "s1" in text or "continuous" in text or "cont" in text:
        return 1.0, "S1"
    if "s6" in text or "s3" in text:
        m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
        d = float(m.group(1)) / 100.0 if m else default_s6
        return math.sqrt(min(max(d, 1e-3), 1.0)), f"S6 {d * 100:.0f} %"
    m = re.search(r"(\d+(?:\.\d+)?)\s*min", text)
    t = float(m.group(1)) * 60.0 if m else default_peak_s
    return math.sqrt(1.0 - math.exp(-t / tau_s)), f"short-time {t / 60.0:.0f} min"


def _num(v, default=0.0):
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _get(obj, key):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    try:
        return obj[key]
    except (KeyError, TypeError, IndexError):
        return getattr(obj, key, None)


def _curve(raw) -> tuple[np.ndarray, np.ndarray]:
    if raw is None:
        return np.zeros(0), np.zeros(0)
    x = getattr(raw, "x", None)
    y = getattr(raw, "y", None)
    if x is not None and y is not None and not callable(x):
        return np.asarray(x, float), np.asarray(y, float)
    rows = np.asarray(getattr(raw, "value", raw), float)
    if rows.ndim == 2 and rows.shape[1] == 2:
        order = np.argsort(rows[:, 0], kind="stable")
        return rows[order, 0], rows[order, 1]
    return np.zeros(0), np.zeros(0)


def spindles_from_machine(machine, params: dict) -> dict:
    """Spindle models of a machine (SIMap of db/machines, SI keys: gear_ranges[].rad_s, torque_Nm as
    [[rad_s, Nm]], power_W as [[rad_s, W]], rating; rotor_inertia_kgm2, max_rad_s)."""
    out = {}
    tau = float(params["motor_thermal_time_constant_s"])
    peak = float(params["default_peak_rating_s"])
    s6 = float(params["default_s6_duty"])
    for sp in (_get(machine, "spindles") or []):
        sid = str(_get(sp, "id") or "main")
        gears = []
        notes = []
        max_w = _num(_get(sp, "max_rad_s"), 0.0)
        for g in (_get(sp, "gear_ranges") or []):
            rng = _get(g, "rad_s")
            lo, hi = (float(rng[0]), float(rng[1])) if rng is not None and len(rng) == 2 else (0.0, max_w or math.inf)
            tx, ty = _curve(_get(g, "torque_Nm"))
            px, py = _curve(_get(g, "power_W"))
            rating = str(_get(g, "rating") or "")
            factor, how = s1_factor(rating, tau, peak, s6)
            notes.append(f"gear {len(gears) + 1}: rating '{rating or 'none'}' read as {how} (S1 = {factor:.2f} x curve)")
            gears.append(Gear(lo, hi, tx, ty, px, py, rating, factor))
        out[sid] = Spindle(sid, gears, _num(_get(sp, "rotor_inertia_kgm2"), 0.0), max_w, notes)
    return out


def thermal_step(theta: float, torque_Nm: float, s1_torque_Nm: float, dt_s: float, tau_s: float) -> float:
    """Exact update of the normalised winding temperature theta over dt with constant torque."""
    if not math.isfinite(s1_torque_Nm) or s1_torque_Nm <= 0.0:
        return theta
    u = (torque_Nm / s1_torque_Nm) ** 2
    return u + (theta - u) * math.exp(-dt_s / tau_s)


def stall_speed(spindle: Spindle, omega0_abs: float, torque0_Nm: float, exponent: float, stall_ratio: float,
                gear: int | None = None) -> tuple[float, bool]:
    """Highest speed below omega0 where the cutting torque T0 (omega0 / omega)^(1 - m) meets the available
    torque (scan down to stall_ratio omega0, then bisection); (omega0, False) when it is not exceeded,
    (0, True) for a stall."""
    if spindle.available(omega0_abs, gear) >= torque0_Nm or omega0_abs <= 0.0:
        return omega0_abs, False

    def excess(w):
        return torque0_Nm * (omega0_abs / w) ** (1.0 - exponent) - spindle.available(w, gear)

    # the spindle slows from omega0 until the torques balance: the first crossing below omega0 (below it the
    # cut may again exceed the curve - that equilibrium is never reached); none above stall_ratio omega0: stall
    grid = omega0_abs * np.geomspace(1.0, stall_ratio, 65)[1:]
    b = omega0_abs
    a = None
    for w in grid:
        if excess(w) <= 0.0:
            a = w
            break
        b = w
    if a is None:
        return 0.0, True
    for _ in range(60):
        mid = 0.5 * (a + b)
        if excess(mid) > 0.0:
            b = mid
        else:
            a = mid
    return a, False


__all__ = ["Gear", "Spindle", "s1_factor", "spindles_from_machine", "thermal_step", "stall_speed"]
