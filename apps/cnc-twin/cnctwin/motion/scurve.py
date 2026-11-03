"""Analytic jerk-limited 7-segment ("double S") profiles (Biagiotti & Melchiorri 2008, s. 3.4).

Used where an axis moves on its own (dog-leg rapids, rotary positioning, the tool-change Z move,
spindle orientation) and as the analytic reference the look-ahead generator is tested against.

Rest-to-rest over a distance D with limits V, A, J (all > 0):

* A reached when V J >= A^2: Tj = A / J, Ta = Tj + V / A; else Tj = sqrt(V / J), Ta = 2 Tj;
* cruise Tv = D / V - Ta; when Tv < 0 the speed limit is not reached:
  D >= 2 A^3 / J^2: Tj = A / J, Ta = Tj / 2 + sqrt((Tj / 2)^2 + D / A); else Tj = (D / (2 J))^(1/3),
  Ta = 2 Tj; Tv = 0.
* T = 2 Ta + Tv; jerk +J, 0, -J over the acceleration phase, 0 at cruise, -J, 0, +J decelerating.

The profile is time-optimal for zero boundary velocity and acceleration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DoubleS:
    distance: float
    Tj: float
    Ta: float
    Tv: float
    J: float

    @property
    def duration(self) -> float:
        return 2.0 * self.Ta + self.Tv

    @property
    def a_lim(self) -> float:
        return self.J * self.Tj

    @property
    def v_lim(self) -> float:
        return self.J * self.Tj * (self.Ta - self.Tj)

    def segments(self) -> list[tuple[float, float]]:
        """(duration, jerk) of the seven phases."""
        Tj, Ta, Tv, J = self.Tj, self.Ta, self.Tv, self.J
        mid = max(Ta - 2.0 * Tj, 0.0)
        return [(Tj, J), (mid, 0.0), (Tj, -J), (Tv, 0.0), (Tj, -J), (mid, 0.0), (Tj, J)]

    def state(self, t: float) -> tuple[float, float, float]:
        """(position, velocity, acceleration) at time t (clamped to [0, T])."""
        if t <= 0.0 or self.distance <= 0.0:
            return 0.0, 0.0, 0.0
        if t >= self.duration:
            return self.distance, 0.0, 0.0
        s = v = a = 0.0
        left = t
        for duration, jerk in self.segments():
            if duration <= 0.0:
                continue
            h = min(left, duration)
            s += v * h + 0.5 * a * h * h + jerk * h * h * h / 6.0
            v += a * h + 0.5 * jerk * h * h
            a += jerk * h
            left -= h
            if left <= 0.0:
                break
        return s, v, a

    def position(self, t: float) -> float:
        return self.state(t)[0]


def double_s(distance: float, v_max: float, a_max: float, j_max: float) -> DoubleS:
    """Time-optimal rest-to-rest double-S profile over ``distance`` (>= 0)."""
    D = abs(float(distance))
    V, A, J = float(v_max), float(a_max), float(j_max)
    if D <= 0.0:
        return DoubleS(0.0, 0.0, 0.0, 0.0, J)
    if V <= 0.0 or A <= 0.0 or J <= 0.0:
        raise ValueError(f"double-S limits must be positive (V={V}, A={A}, J={J})")
    if V * J >= A * A:
        Tj = A / J
        Ta = Tj + V / A
    else:
        Tj = math.sqrt(V / J)
        Ta = 2.0 * Tj
    Tv = D / V - Ta
    if Tv < 0.0:
        Tv = 0.0
        if D >= 2.0 * A ** 3 / (J * J):
            Tj = A / J
            Ta = 0.5 * Tj + math.sqrt((0.5 * Tj) ** 2 + D / A)
        else:
            Tj = (D / (2.0 * J)) ** (1.0 / 3.0)
            Ta = 2.0 * Tj
    return DoubleS(D, Tj, Ta, Tv, J)


def ticks_for(profile: DoubleS, dt: float) -> int:
    """Whole L1 ticks the profile needs (its last setpoint falls on or after its end)."""
    if profile.duration <= 0.0:
        return 0
    return int(math.ceil(profile.duration / dt - 1e-9))
