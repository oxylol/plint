"""Plane geometry of cutter / tool nose radius compensation and of chamfers and corner rounding.

Elements live in the 2-D coordinates of the compensation plane: ``u`` along the plane's first axis and
``v`` along its second (G17: X, Y; G18: Z, X; G19: Y, Z), so "left of the path" is the left of the
program path whatever mirror or rotation maps it into the work frame. Arcs are signed: ``sweep > 0``
is counter-clockwise in (u, v).

Offsets (FANUC type C, fanuc-30ibplus-om-mc-2019 6.6): a signed offset ``d`` (> 0 on the left, G41; the
sign of the D value swaps the side) moves a line by ``d`` along its left normal and changes an arc's
radius to ``R - d sign(sweep)``; a radius <= 0 means the tool cannot follow the arc (interference).

Corner between an incoming element A and an outgoing element B at the programmed point P, by the turn
angle phi toward the tool side (phi > 0: the tool is inside the corner, FANUC "inner side", alpha >= 180;
phi < 0: outer side, alpha = 180 deg + phi):

* tangent (|phi| below 1e-9 rad): one vector, the common normal point;
* inner: one vector, the intersection of the offset paths (lines, circles); none: PS0033;
* outer, 90 <= alpha < 180 (linear connection, parameter 19607#2 CCC = 0): the intersection;
* outer, alpha < 90 (linear connection): two vectors, each offset path extended by |d| along its tangent,
  joined by a line;
* outer, circular connection (CCC = 1): an arc of radius |d| about P.

Interference checks (direction check, circular angle check, 6.6.6) compare a block's programmed
direction with its compensated one; a reversal (angle of 90 deg or more) or a compensated arc angle that
differs from the programmed one by 180 deg or more is an interference.

Chamfer ``,C c``: the points at distance c from the hypothetical corner on both elements (on an arc the
chord distance), joined by a line. Corner rounding ``,R r``: the arc of radius r tangent to both
elements (centre on both elements offset by r toward the inside of the turn). Outside the elements:
PS0055.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ..alarms import ControllerAlarm

EPS = 1e-12
TWO_PI = 2.0 * math.pi
TANGENT_RAD = 1e-9


@dataclass(frozen=True)
class Seg:
    kind: str                                   # "line" | "arc"
    p0: tuple[float, float]
    p1: tuple[float, float]
    c: tuple[float, float] = (0.0, 0.0)
    sweep: float = 0.0                          # arc: signed, CCW positive

    @property
    def radius(self) -> float:
        return math.hypot(self.p0[0] - self.c[0], self.p0[1] - self.c[1])

    def length(self) -> float:
        if self.kind == "line":
            return math.hypot(self.p1[0] - self.p0[0], self.p1[1] - self.p0[1])
        return abs(self.sweep) * self.radius

    def tangent(self, at_end: bool) -> tuple[float, float]:
        if self.kind == "line":
            dx, dy = self.p1[0] - self.p0[0], self.p1[1] - self.p0[1]
            n = math.hypot(dx, dy)
            return (dx / n, dy / n) if n > 0.0 else (0.0, 0.0)
        p = self.p1 if at_end else self.p0
        rx, ry = p[0] - self.c[0], p[1] - self.c[1]
        r = math.hypot(rx, ry)
        sign = 1.0 if self.sweep > 0.0 else -1.0
        return (-sign * ry / r, sign * rx / r) if r > 0.0 else (0.0, 0.0)


def left(t: tuple[float, float]) -> tuple[float, float]:
    return (-t[1], t[0])


def add(p, q, k: float = 1.0) -> tuple[float, float]:
    return (p[0] + k * q[0], p[1] + k * q[1])


def sub(p, q) -> tuple[float, float]:
    return (p[0] - q[0], p[1] - q[1])


def dot(p, q) -> float:
    return p[0] * q[0] + p[1] * q[1]


def cross(p, q) -> float:
    return p[0] * q[1] - p[1] * q[0]


def dist(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def offset_point(seg: Seg, d: float, at_end: bool) -> tuple[float, float]:
    p = seg.p1 if at_end else seg.p0
    return add(p, left(seg.tangent(at_end)), d)


def offset(seg: Seg, d: float) -> Seg | None:
    """The element offset by d to its left (None: an arc whose offset radius is <= 0)."""
    if seg.kind == "line":
        n = left(seg.tangent(False))
        return Seg("line", add(seg.p0, n, d), add(seg.p1, n, d))
    r = seg.radius
    new_r = r - d * (1.0 if seg.sweep > 0.0 else -1.0)
    if new_r <= EPS:
        return None
    k = new_r / r
    p0 = (seg.c[0] + (seg.p0[0] - seg.c[0]) * k, seg.c[1] + (seg.p0[1] - seg.c[1]) * k)
    p1 = (seg.c[0] + (seg.p1[0] - seg.c[0]) * k, seg.c[1] + (seg.p1[1] - seg.c[1]) * k)
    return Seg("arc", p0, p1, seg.c, seg.sweep)


def turn(a: Seg, b: Seg) -> float:
    """Signed turn angle from A's end direction to B's start direction (left turn positive)."""
    ta, tb = a.tangent(True), b.tangent(False)
    return math.atan2(cross(ta, tb), dot(ta, tb))


def intersections(a: Seg, b: Seg) -> list[tuple[float, float]]:
    """Intersections of the supporting line / circle of a and b."""
    if a.kind == "line" and b.kind == "line":
        d1, d2 = sub(a.p1, a.p0), sub(b.p1, b.p0)
        den = cross(d1, d2)
        if abs(den) < EPS * max(1.0, math.hypot(*d1) * math.hypot(*d2)):
            return []
        t = cross(sub(b.p0, a.p0), d2) / den
        return [add(a.p0, d1, t)]
    if a.kind == "arc" and b.kind == "line":
        return intersections(b, a)
    if a.kind == "line":
        cx, cy, r = b.c[0], b.c[1], b.radius
        dx, dy = a.p1[0] - a.p0[0], a.p1[1] - a.p0[1]
        fx, fy = a.p0[0] - cx, a.p0[1] - cy
        qa = dx * dx + dy * dy
        qb = 2.0 * (fx * dx + fy * dy)
        qc = fx * fx + fy * fy - r * r
        disc = qb * qb - 4.0 * qa * qc
        if qa <= 0.0 or disc < -1e-18 * max(1.0, qb * qb):
            return []
        root = math.sqrt(max(disc, 0.0))
        return [(a.p0[0] + t * dx, a.p0[1] + t * dy) for t in ((-qb - root) / (2 * qa), (-qb + root) / (2 * qa))]
    (x0, y0), r0 = a.c, a.radius
    (x1, y1), r1 = b.c, b.radius
    dd = math.hypot(x1 - x0, y1 - y0)
    if dd < EPS or dd > r0 + r1 + 1e-12 or dd < abs(r0 - r1) - 1e-12:
        return []
    along = (r0 * r0 - r1 * r1 + dd * dd) / (2.0 * dd)
    h = math.sqrt(max(r0 * r0 - along * along, 0.0))
    mx, my = x0 + along * (x1 - x0) / dd, y0 + along * (y1 - y0) / dd
    return [(mx + h * (y1 - y0) / dd, my - h * (x1 - x0) / dd), (mx - h * (y1 - y0) / dd, my + h * (x1 - x0) / dd)]


def nearest(points: list, ref) -> tuple[float, float] | None:
    if not points:
        return None
    return min(points, key=lambda p: dist(p, ref))


@dataclass
class Corner:
    """The compensated corner: ``points`` in path order; ``arc`` (centre, ccw) joins points[0] -> points[-1]
    when the outer corner is connected circularly; ``kind`` for reports and tests."""

    points: list
    kind: str
    arc: tuple | None = None


def corner(a: Seg, b: Seg, d: float, outer: str = "linear") -> Corner:
    """Corner vectors (absolute points, path order) between A and B, both offset by d."""
    p = a.p1
    if d == 0.0:
        return Corner([p], "zero")
    va = offset_point(a, d, True)
    vb = offset_point(b, d, False)
    phi = turn(a, b)
    if abs(phi) < TANGENT_RAD:
        return Corner([va], "tangent")
    oa, ob = offset(a, d), offset(b, d)
    if oa is None or ob is None:
        raise ControllerAlarm("crc_interference", "the tool cannot follow an arc smaller than its radius")
    mid = ((va[0] + vb[0]) / 2.0, (va[1] + vb[1]) / 2.0)
    toward = phi * (1.0 if d > 0.0 else -1.0)
    if toward > 0.0:
        hit = nearest(intersections(oa, ob), mid)
        if hit is None:
            raise ControllerAlarm("crc_no_intersection", "the offset paths of two blocks do not intersect")
        return Corner([hit], "inner")
    if outer == "circular":
        return Corner([va, vb], "outer_arc", (p, d > 0.0))      # around an outer corner in the turn's sense
    if toward >= -math.pi / 2.0:
        hit = nearest(intersections(oa, ob), mid)
        if hit is not None and dist(hit, p) <= abs(d) * math.sqrt(2.0) * (1.0 + 1e-9) + 1e-12:
            return Corner([hit], "outer_obtuse")
    ta, tb = a.tangent(True), b.tangent(False)
    return Corner([add(va, ta, abs(d)), add(vb, tb, -abs(d))], "outer_acute")


def side_change(a: Seg, b: Seg, d_a: float, d_b: float) -> Corner:
    """G41 <-> G42 (or a D sign change) between A and B: the intersection of A's and B's offset paths when
    it lies on both (FANUC 6.6.3 "change in the offset direction"), else A's end normal then B's start
    normal."""
    va = offset_point(a, d_a, True)
    vb = offset_point(b, d_b, False)
    oa, ob = offset(a, d_a), offset(b, d_b)
    if oa is not None and ob is not None:
        for hit in sorted(intersections(oa, ob), key=lambda q: dist(q, a.p1)):
            if _on_element(oa, hit) and _on_element(ob, hit):
                return Corner([hit], "change")
    return Corner([va, vb], "change")


def perpendicular(seg: Seg, d: float, at_end: bool) -> Corner:
    """Only the vector normal to one element (blocks without movement beyond N - 2, buffer stops ...)."""
    return Corner([offset_point(seg, d, at_end)], "normal")


def ijk_vector(point, direction: tuple[float, float], d: float) -> Corner:
    """FANUC "IJ type vector": the vector normal to the direction (I, J) on the offset side."""
    n = math.hypot(*direction)
    if n == 0.0:
        raise ControllerAlarm("improper_address", "I, J, K vector of length 0")
    t = (direction[0] / n, direction[1] / n)
    return Corner([add(point, left(t), d)], "ijk")


def sweep_between(c, start, end, reference: float, full: bool = False) -> float:
    """Sweep from start to end about c in the turning sense of ``reference`` (in (0, 2 pi]); ``full``
    keeps a full circle when start and end coincide."""
    a0 = math.atan2(start[1] - c[1], start[0] - c[0])
    a1 = math.atan2(end[1] - c[1], end[0] - c[0])
    if reference > 0.0:
        delta = math.fmod(a1 - a0 + 4.0 * math.pi, TWO_PI)
        if delta <= 1e-15:
            delta = TWO_PI if full else 0.0
        return delta
    delta = math.fmod(a0 - a1 + 4.0 * math.pi, TWO_PI)
    if delta <= 1e-15:
        delta = TWO_PI if full else 0.0
    return -delta


def compensated(seg: Seg, start, end, d: float) -> Seg:
    """The compensated element from start to end (an arc keeps its centre and sense)."""
    if seg.kind == "line":
        return Seg("line", start, end)
    full = abs(abs(seg.sweep) - TWO_PI) < 1e-9 and dist(start, end) < 1e-12
    return Seg("arc", start, end, seg.c, sweep_between(seg.c, start, end, seg.sweep, full))


def direction_interferes(prog_start, prog_end, comp_start, comp_end) -> bool:
    """Direction check: the compensated displacement runs against the programmed one (90 deg or more)."""
    pd = sub(prog_end, prog_start)
    cd = sub(comp_end, comp_start)
    if math.hypot(*pd) < 1e-12 or math.hypot(*cd) < 1e-12:
        return False
    return dot(pd, cd) < 0.0


def arc_interferes(seg: Seg, comp_start, comp_end) -> bool:
    """Circular angle check: the compensated arc's angle differs from the programmed one by 180 deg or more."""
    if seg.kind != "arc":
        return False
    full = abs(abs(seg.sweep) - TWO_PI) < 1e-9
    comp = sweep_between(seg.c, comp_start, comp_end, seg.sweep, full and dist(comp_start, comp_end) < 1e-12)
    return abs(abs(comp) - abs(seg.sweep)) >= math.pi - 1e-12


# ---------------------------------------------------------------------- chamfer and corner rounding

def point_at(seg: Seg, distance: float, from_end: bool) -> tuple[float, float] | None:
    """The point at chord distance ``distance`` from an end of the element, on the element (None if beyond)."""
    if seg.kind == "line":
        length = seg.length()
        if distance > length + 1e-12:
            return None
        t = seg.tangent(False)
        return add(seg.p1, t, -distance) if from_end else add(seg.p0, t, distance)
    r = seg.radius
    if distance > 2.0 * r + 1e-12:
        return None
    angle = 2.0 * math.asin(min(1.0, distance / (2.0 * r)))
    if angle > abs(seg.sweep) + 1e-12:
        return None
    ref = seg.p1 if from_end else seg.p0
    a0 = math.atan2(ref[1] - seg.c[1], ref[0] - seg.c[0])
    sign = 1.0 if seg.sweep > 0.0 else -1.0
    a = a0 - sign * angle if from_end else a0 + sign * angle
    return (seg.c[0] + r * math.cos(a), seg.c[1] + r * math.sin(a))


def trim_end(seg: Seg, point) -> Seg:
    if seg.kind == "line":
        return replace(seg, p1=point)
    return replace(seg, p1=point, sweep=sweep_between(seg.c, seg.p0, point, seg.sweep))


def trim_start(seg: Seg, point) -> Seg:
    if seg.kind == "line":
        return replace(seg, p0=point)
    return replace(seg, p0=point, sweep=sweep_between(seg.c, point, seg.p1, seg.sweep))


def _on_element(seg: Seg, point) -> bool:
    if seg.kind == "line":
        t = seg.tangent(False)
        s = dot(sub(point, seg.p0), t)
        return -1e-9 <= s <= seg.length() + 1e-9
    part = sweep_between(seg.c, seg.p0, point, seg.sweep)
    return abs(part) <= abs(seg.sweep) + 1e-9


def chamfer(a: Seg, b: Seg, size: float) -> tuple[Seg, Seg, Seg]:
    """(A trimmed, chamfer line, B trimmed) for ,C size."""
    pa = point_at(a, size, from_end=True)
    pb = point_at(b, size, from_end=False)
    if pa is None or pb is None:
        raise ControllerAlarm("chamfer_too_large", f",C{size * 1e3:.3f} mm exceeds the block")
    return trim_end(a, pa), Seg("line", pa, pb), trim_start(b, pb)


def round_corner(a: Seg, b: Seg, radius: float) -> tuple[Seg, Seg, Seg]:
    """(A trimmed, fillet arc, B trimmed) for ,R radius."""
    phi = turn(a, b)
    if abs(phi) < math.radians(1.0):
        return a, Seg("line", a.p1, a.p1), b          # FANUC: within 1 deg the corner block has no travel
    side = 1.0 if phi > 0.0 else -1.0                  # the fillet centre lies on the inside of the turn
    oa, ob = offset(a, side * radius), offset(b, side * radius)
    if oa is None or ob is None:
        raise ControllerAlarm("chamfer_too_large", f",R{radius * 1e3:.3f} mm larger than an arc of the corner")
    centre = nearest(intersections(oa, ob), a.p1)
    if centre is None:
        raise ControllerAlarm("chamfer_too_large", f",R{radius * 1e3:.3f} mm: no fillet")
    ta = _foot(a, centre)
    tb = _foot(b, centre)
    if not (_on_element(a, ta) and _on_element(b, tb)):
        raise ControllerAlarm("chamfer_too_large", f",R{radius * 1e3:.3f} mm exceeds the block")
    sweep = sweep_between(centre, ta, tb, side)
    return trim_end(a, ta), Seg("arc", ta, tb, centre, sweep), trim_start(b, tb)


def _foot(seg: Seg, point) -> tuple[float, float]:
    """Foot of the perpendicular from point to the element's supporting line / circle."""
    if seg.kind == "line":
        t = seg.tangent(False)
        s = dot(sub(point, seg.p0), t)
        return add(seg.p0, t, s)
    r = seg.radius
    vx, vy = point[0] - seg.c[0], point[1] - seg.c[1]
    n = math.hypot(vx, vy)
    return (seg.c[0] + r * vx / n, seg.c[1] + r * vy / n)


# ---------------------------------------------------------------------- imaginary tool nose (lathe)

# Direction from the tool nose centre to the imaginary tool nose, in plane coordinates (first, second axis
# of the plane: G18 Z, X): FANUC tip numbers 1-8 (0 and 9: the nose centre is the reference point),
# fanuc-30ibplus-om-lathe-2019 5.2.2 (figure as tabulated in Smid 2008, ch. 30, X up and Z right).
TIP_DIRECTIONS = {0: (0.0, 0.0), 1: (1.0, 1.0), 2: (-1.0, 1.0), 3: (-1.0, -1.0), 4: (1.0, -1.0),
                  5: (1.0, 0.0), 6: (0.0, 1.0), 7: (-1.0, 0.0), 8: (0.0, -1.0), 9: (0.0, 0.0)}


def tip_vector(tip: int, radius: float) -> tuple[float, float]:
    """Nose centre -> imaginary tool nose (plane coordinates) for tip number 0-9 and nose radius r."""
    du, dv = TIP_DIRECTIONS.get(int(tip), (0.0, 0.0))
    return (du * abs(radius), dv * abs(radius))


__all__ = ["Seg", "Corner", "corner", "side_change", "perpendicular", "ijk_vector", "offset", "offset_point", "turn",
           "intersections", "nearest", "compensated", "sweep_between", "direction_interferes",
           "arc_interferes", "chamfer", "round_corner", "point_at", "trim_end", "trim_start", "tip_vector",
           "TIP_DIRECTIONS", "left", "add", "sub", "dot", "cross", "dist"]
