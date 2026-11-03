"""Closed 2-D contours of lines and arcs: orientation, inward offset, polylines, containment.

Used by the Haas G150 general pocket (roughing region and finishing path). The inward offset of a
counter-clockwise contour by d: lines move left by d, counter-clockwise arcs shrink to r - d,
clockwise arcs grow to r + d; at a convex corner (left turn) the two offset elements are trimmed
to their intersection, at a concave corner (right turn) a clockwise arc of radius d about the
corner joins them (the tool rolls around the corner). Elements that vanish (an arc with r <= d, or
an element whose trimmed ends cross) are removed and the joints recomputed. A remaining global
self-intersection means the pocket is too narrow for the tool: the caller raises an alarm.
This is the exact offset of the contour for d below the smallest feature size (no sampling).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

EPS = 1e-12


@dataclass(frozen=True)
class Element:
    kind: str                                   # "line" | "arc"
    p0: tuple[float, float]
    p1: tuple[float, float]
    centre: tuple[float, float] = (0.0, 0.0)
    sweep: float = 0.0                          # arc: signed, counter-clockwise positive

    @property
    def radius(self) -> float:
        return math.hypot(self.p0[0] - self.centre[0], self.p0[1] - self.centre[1])

    def tangent(self, at_end: bool) -> tuple[float, float]:
        if self.kind == "line":
            dx, dy = self.p1[0] - self.p0[0], self.p1[1] - self.p0[1]
            length = math.hypot(dx, dy)
            return (dx / length, dy / length) if length > 0.0 else (0.0, 0.0)
        point = self.p1 if at_end else self.p0
        rx, ry = point[0] - self.centre[0], point[1] - self.centre[1]
        r = math.hypot(rx, ry)
        sign = 1.0 if self.sweep > 0.0 else -1.0
        return (-sign * ry / r, sign * rx / r)

    def reversed(self) -> "Element":
        return replace(self, p0=self.p1, p1=self.p0, sweep=-self.sweep)

    def length(self) -> float:
        if self.kind == "line":
            return math.hypot(self.p1[0] - self.p0[0], self.p1[1] - self.p0[1])
        return abs(self.sweep) * self.radius


def signed_area(elements: list[Element]) -> float:
    area = 0.0
    for e in elements:
        area += 0.5 * (e.p0[0] * e.p1[1] - e.p1[0] * e.p0[1])
        if e.kind == "arc":
            r = e.radius
            area += 0.5 * r * r * (e.sweep - math.sin(e.sweep))
    return area


def reverse(elements: list[Element]) -> list[Element]:
    return [e.reversed() for e in reversed(elements)]


def polyline(elements: list[Element], sagitta: float) -> list[tuple[float, float]]:
    """Points of the closed contour (the closing point not repeated), arcs as chords."""
    points: list[tuple[float, float]] = []
    for e in elements:
        points.append(e.p0)
        if e.kind == "arc":
            r = e.radius
            step = 2.0 * math.acos(max(-1.0, 1.0 - sagitta / r)) if r > sagitta else math.pi / 2.0
            count = max(1, int(math.ceil(abs(e.sweep) / max(step, 1e-9))))
            a0 = math.atan2(e.p0[1] - e.centre[1], e.p0[0] - e.centre[0])
            for k in range(1, count):
                angle = a0 + e.sweep * k / count
                points.append((e.centre[0] + r * math.cos(angle), e.centre[1] + r * math.sin(angle)))
    return points


def contains(points: list[tuple[float, float]], p: tuple[float, float]) -> bool:
    inside = False
    n = len(points)
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        if (y0 <= p[1] < y1) or (y1 <= p[1] < y0):
            x = x0 + (p[1] - y0) * (x1 - x0) / (y1 - y0)
            if x > p[0]:
                inside = not inside
    return inside


def _offset_element(e: Element, d: float) -> Element | None:
    if e.kind == "line":
        tx, ty = e.tangent(False)
        nx, ny = -ty, tx
        return Element("line", (e.p0[0] + d * nx, e.p0[1] + d * ny), (e.p1[0] + d * nx, e.p1[1] + d * ny))
    r = e.radius
    new_r = r - d if e.sweep > 0.0 else r + d
    if new_r <= EPS:
        return None
    scale = new_r / r
    p0 = (e.centre[0] + (e.p0[0] - e.centre[0]) * scale, e.centre[1] + (e.p0[1] - e.centre[1]) * scale)
    p1 = (e.centre[0] + (e.p1[0] - e.centre[0]) * scale, e.centre[1] + (e.p1[1] - e.centre[1]) * scale)
    return Element("arc", p0, p1, e.centre, e.sweep)


def _intersections(a: Element, b: Element) -> list[tuple[float, float]]:
    """Intersection points of the supporting line/circle of a and b."""
    if a.kind == "line" and b.kind == "line":
        d1 = (a.p1[0] - a.p0[0], a.p1[1] - a.p0[1])
        d2 = (b.p1[0] - b.p0[0], b.p1[1] - b.p0[1])
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < EPS:
            return []
        t = ((b.p0[0] - a.p0[0]) * d2[1] - (b.p0[1] - a.p0[1]) * d2[0]) / den
        return [(a.p0[0] + t * d1[0], a.p0[1] + t * d1[1])]
    if a.kind == "arc" and b.kind == "line":
        return _intersections(b, a)
    if a.kind == "line":
        cx, cy, r = b.centre[0], b.centre[1], b.radius
        dx, dy = a.p1[0] - a.p0[0], a.p1[1] - a.p0[1]
        fx, fy = a.p0[0] - cx, a.p0[1] - cy
        qa = dx * dx + dy * dy
        qb = 2.0 * (fx * dx + fy * dy)
        qc = fx * fx + fy * fy - r * r
        disc = qb * qb - 4.0 * qa * qc
        if disc < 0.0:
            return []
        root = math.sqrt(disc)
        return [(a.p0[0] + t * dx, a.p0[1] + t * dy) for t in ((-qb - root) / (2 * qa), (-qb + root) / (2 * qa))]
    (x0, y0), r0 = a.centre, a.radius
    (x1, y1), r1 = b.centre, b.radius
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist < EPS or dist > r0 + r1 or dist < abs(r0 - r1):
        return []
    along = (r0 * r0 - r1 * r1 + dist * dist) / (2.0 * dist)
    h = math.sqrt(max(r0 * r0 - along * along, 0.0))
    mx, my = x0 + along * (x1 - x0) / dist, y0 + along * (y1 - y0) / dist
    return [(mx + h * (y1 - y0) / dist, my - h * (x1 - x0) / dist), (mx - h * (y1 - y0) / dist, my + h * (x1 - x0) / dist)]


def _with_end(e: Element, point: tuple[float, float]) -> Element:
    if e.kind == "line":
        return replace(e, p1=point)
    return replace(e, p1=point, sweep=_arc_sweep(e.centre, e.p0, point, e.sweep))


def _with_start(e: Element, point: tuple[float, float]) -> Element:
    if e.kind == "line":
        return replace(e, p0=point)
    return replace(e, p0=point, sweep=_arc_sweep(e.centre, point, e.p1, e.sweep))


def _arc_sweep(centre, start, end, reference: float) -> float:
    """Sweep from start to end about centre in the turning sense of ``reference`` (0 if they coincide)."""
    a0 = math.atan2(start[1] - centre[1], start[0] - centre[0])
    a1 = math.atan2(end[1] - centre[1], end[0] - centre[0])
    delta = a1 - a0
    if reference > 0.0:
        return math.fmod(delta + 4.0 * math.pi, 2.0 * math.pi)
    return -math.fmod(-delta + 4.0 * math.pi, 2.0 * math.pi)


def _valid(original: Element, trimmed: Element) -> bool:
    """A trimmed element keeps its direction and does not grow (else it has vanished)."""
    if trimmed.kind == "line":
        ox, oy = original.p1[0] - original.p0[0], original.p1[1] - original.p0[1]
        tx, ty = trimmed.p1[0] - trimmed.p0[0], trimmed.p1[1] - trimmed.p0[1]
        return ox * tx + oy * ty > 0.0
    return trimmed.sweep * original.sweep > 0.0 and abs(trimmed.sweep) <= abs(original.sweep) + 1e-9


def offset_inward(elements: list[Element], d: float) -> list[Element]:
    """Inward offset of a counter-clockwise closed contour (see module docstring)."""
    work = list(elements)
    for _ in range(len(elements) + 1):
        offsets = [(_offset_element(e, d), e) for e in work]
        if any(o is None for o, _ in offsets):
            work = [e for (o, e) in offsets if o is not None]
            if len(work) < 2:
                return []
            continue
        result: list[Element] = []
        n = len(offsets)
        trimmed = [o for o, _ in offsets]
        joins: list[Element | None] = [None] * n
        for i in range(n):
            a, b = trimmed[i], trimmed[(i + 1) % n]
            vertex = work[i].p1
            ta, tb = work[i].tangent(True), work[(i + 1) % n].tangent(False)
            cross = ta[0] * tb[1] - ta[1] * tb[0]
            gap = math.hypot(a.p1[0] - b.p0[0], a.p1[1] - b.p0[1])
            if gap <= 1e-12:
                continue
            if cross < 0.0:
                sweep = _arc_sweep(vertex, a.p1, b.p0, -1.0)
                joins[i] = Element("arc", a.p1, b.p0, vertex, sweep)
                continue
            candidates = _intersections(a, b)
            if not candidates:
                joins[i] = Element("line", a.p1, b.p0)
                continue
            point = min(candidates, key=lambda c: math.hypot(c[0] - a.p1[0], c[1] - a.p1[1]))
            trimmed[i] = _with_end(trimmed[i], point)
            trimmed[(i + 1) % n] = _with_start(trimmed[(i + 1) % n], point)
        vanished = [i for i in range(n) if not _valid(offsets[i][0], trimmed[i])]
        if vanished:
            work = [e for k, e in enumerate(work) if k not in vanished]
            if len(work) < 2:
                return []
            continue
        for i in range(n):
            result.append(trimmed[i])
            if joins[i] is not None:
                result.append(joins[i])
        return result
    return []


def self_intersects(elements: list[Element], sagitta: float) -> bool:
    points = polyline(elements, sagitta)
    n = len(points)
    for i in range(n):
        a0, a1 = points[i], points[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            b0, b1 = points[j], points[(j + 1) % n]
            d1 = (a1[0] - a0[0]) * (b0[1] - a0[1]) - (a1[1] - a0[1]) * (b0[0] - a0[0])
            d2 = (a1[0] - a0[0]) * (b1[1] - a0[1]) - (a1[1] - a0[1]) * (b1[0] - a0[0])
            d3 = (b1[0] - b0[0]) * (a0[1] - b0[1]) - (b1[1] - b0[1]) * (a0[0] - b0[0])
            d4 = (b1[0] - b0[0]) * (a1[1] - b0[1]) - (b1[1] - b0[1]) * (a1[0] - b0[0])
            if d1 * d2 < -1e-24 and d3 * d4 < -1e-24:
                return True
    return False
