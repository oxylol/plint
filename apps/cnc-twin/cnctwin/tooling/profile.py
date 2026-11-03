"""Profiles in the meridian half-plane (z along +z_T, r from the axis): straight pieces and arcs.

A cutting edge is described by the profile of its revolved sweep (the curve every tooth
follows in the (z, r) plane) plus, per tooth, its angular position along that curve (the lag:
helix, drill-lip web offset, insert inclination). Profiles are traversed counter-clockwise around
the tool solid (bottom outward, side upward, tops inward), so the normal into the material is the
direction turned by -90 degrees: with the direction (dr, dz) = (cos kappa, sin kappa),
n = sin(kappa) e_r - cos(kappa) e_z. kappa = 0 is an end edge (n = -e_z), pi/2 a peripheral edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

LINE = "line"
ARC = "arc"
CURVE = "curve"     # parametric curve: extra["fn"](u) -> (z, r) (drill lips)


@dataclass
class Seg:
    """One profile piece. Points are (z, r).

    ``kind`` is the EDGE_* kind for cutting pieces (None for envelope-only pieces); ``rule`` tells
    the tooth how to place the piece angularly (see cnctwin.tooling.edges); ``face`` describes
    the rake and flank faces; ``tag`` is free text for builders (which edge this is).
    """

    shape: str
    kind: int | None = None
    p0: tuple[float, float] = (0.0, 0.0)
    p1: tuple[float, float] = (0.0, 0.0)
    centre: tuple[float, float] = (0.0, 0.0)
    rho: float = 0.0
    a0: float = 0.0
    a1: float = 0.0
    rule: str = "helix"
    face: object = None
    tag: str = ""
    min_elements: int = 1
    element_scale: float = 1.0          # element length limit = edge_element * element_scale
    part: int = 0                       # envelope part (PART_*), for non-cutting pieces
    extra: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- evaluation
    def point(self, u):
        u = np.asarray(u, dtype=float)
        if self.shape == CURVE:
            return self.extra["fn"](u)
        if self.shape == LINE:
            z = self.p0[0] + u * (self.p1[0] - self.p0[0])
            r = self.p0[1] + u * (self.p1[1] - self.p0[1])
            return z, r
        a = self.a0 + u * (self.a1 - self.a0)
        return self.centre[0] - self.rho * np.cos(a), self.centre[1] + self.rho * np.sin(a)

    def deriv(self, u):
        """(dz/du, dr/du)."""
        u = np.asarray(u, dtype=float)
        if self.shape == CURVE:
            step = 1e-6
            z1, r1 = self.extra["fn"](np.clip(u + step, 0.0, 1.0))
            z0, r0 = self.extra["fn"](np.clip(u - step, 0.0, 1.0))
            span = np.clip(u + step, 0.0, 1.0) - np.clip(u - step, 0.0, 1.0)
            return (np.asarray(z1) - np.asarray(z0)) / span, (np.asarray(r1) - np.asarray(r0)) / span
        if self.shape == LINE:
            return (np.full_like(u, self.p1[0] - self.p0[0]), np.full_like(u, self.p1[1] - self.p0[1]))
        a = self.a0 + u * (self.a1 - self.a0)
        da = self.a1 - self.a0
        return self.rho * np.sin(a) * da, self.rho * np.cos(a) * da

    @property
    def length(self) -> float:
        if self.shape == CURVE:
            z, r = self.point(np.linspace(0.0, 1.0, 513))
            return float(np.sum(np.hypot(np.diff(z), np.diff(r))))
        if self.shape == LINE:
            return math.hypot(self.p1[0] - self.p0[0], self.p1[1] - self.p0[1])
        return abs(self.a1 - self.a0) * self.rho

    @property
    def start(self) -> tuple[float, float]:
        z, r = self.point(0.0)
        return float(z), float(r)

    @property
    def end(self) -> tuple[float, float]:
        z, r = self.point(1.0)
        return float(z), float(r)

    def kappa(self, u):
        dz, dr = self.deriv(u)
        return np.arctan2(dz, dr)

    def kappa_span(self) -> float:
        return abs(self.a1 - self.a0) if self.shape == ARC else 0.0


def line(p0, p1, kind=None, **kw) -> Seg:
    return Seg(LINE, kind, p0=(float(p0[0]), float(p0[1])), p1=(float(p1[0]), float(p1[1])), **kw)


def curve(fn, kind=None, **kw) -> Seg:
    """Parametric piece: ``fn(u) -> (z, r)`` for u in [0, 1] (vectorised)."""
    extra = dict(kw.pop("extra", {}))
    extra["fn"] = fn
    seg = Seg(CURVE, kind, extra=extra, **kw)
    seg.p0 = seg.start
    seg.p1 = seg.end
    return seg


def arc(centre, rho, a0, a1, kind=None, **kw) -> Seg:
    """Arc with point(a) = (zc - rho cos a, rc + rho sin a); a0 -> a1 (increasing a is convex CCW)."""
    if rho <= 0.0:
        raise ValueError("arc radius must be positive")
    seg = Seg(ARC, kind, centre=(float(centre[0]), float(centre[1])), rho=float(rho), a0=float(a0), a1=float(a1),
              **kw)
    seg.p0 = seg.start
    seg.p1 = seg.end
    return seg


def fillet_between(p_corner, dir_in, dir_out, radius):
    """Tangent arc of ``radius`` rounding the corner where a piece arriving along ``dir_in`` (unit
    (dz, dr)) turns into ``dir_out``. Returns (tangent point in, tangent point out, centre, a0, a1)
    with the arc parametrised as in :func:`arc` (convex corner: kappa increases from in to out)."""
    k_in = math.atan2(dir_in[0], dir_in[1])
    k_out = math.atan2(dir_out[0], dir_out[1])
    turn = _wrap(k_out - k_in)
    if turn <= 0.0:
        raise ValueError("fillet_between: the corner is not convex")
    if turn >= math.pi:
        raise ValueError("fillet_between: the corner turns by pi or more")
    # distance from the corner to each tangent point: radius * tan(turn / 2)
    reach = radius * math.tan(turn / 2.0)
    t_in = (p_corner[0] - dir_in[0] * reach, p_corner[1] - dir_in[1] * reach)
    t_out = (p_corner[0] + dir_out[0] * reach, p_corner[1] + dir_out[1] * reach)
    # the centre lies against the normal into the material, n = (sin k e_r - cos k e_z) -> (z, r) = (-cos k, sin k)
    n_in = (-math.cos(k_in), math.sin(k_in))
    centre = (t_in[0] - n_in[0] * radius, t_in[1] - n_in[1] * radius)
    return t_in, t_out, centre, k_in, k_in + turn


def _wrap(angle: float) -> float:
    """Angle in (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))
