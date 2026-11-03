"""Coordinate pipeline and arc geometry of the executor (dialect-neutral).

Program frame -> work frame (FANUC order: programmable mirror image -> scaling -> coordinate
rotation; Haas G268 feature coordinate systems after rotation), all affine, so a point and its
inverse are exact compositions. Work frame -> machine: + work offset (G54..., G54.1, EXT) + G52
local shift + G92 shift + tool offsets (length along the tool axis; lathe X/Z offsets), done in
integer counts by the executor. Cutter compensation acts in the work frame between the two.

Arcs (G02/G03) are defined in the program frame from the start point, the end point and either
the centre offsets (I, J, K: always incremental from the start) or the radius R (R > 0: at most
180 deg, R < 0: more than 180 deg), then mapped point by point: the arc's normal in the work
frame is ``L a1 x L a2`` for the plane axes a1, a2 and the linear part L, so a mirror flips the
turning sense, uniform scaling and rotations keep it (FANUC behaviour of G51.1/G51/G68 on arcs).
"""

from __future__ import annotations

import math

from .alarms import ControllerAlarm

# plane -> (first axis, second axis, normal axis); the right-handed order of FANUC's G17/G18/G19
PLANES = {"XY": ("X", "Y", "Z"), "ZX": ("Z", "X", "Y"), "YZ": ("Y", "Z", "X")}
PLANE_OF = {"G17": "XY", "G18": "ZX", "G19": "YZ"}
UNIT = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}
TWO_PI = 2.0 * math.pi


def _matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _apply(m, v):
    return [m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]]


def _transpose(m):
    return [[m[j][i] for j in range(3)] for i in range(3)]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _norm(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def euler_matrix(angles_deg: tuple[float, float, float], order: str) -> list[list[float]]:
    """Intrinsic rotations about the axes named by the digits of ``order`` (1 = X, 2 = Y, 3 = Z), applied
    left to right (Haas G268 Q321: first about Z, then the new Y, then the new X); ``angles_deg``
    are the angles about X, Y and Z (I, J, K)."""
    total = IDENTITY
    for digit in order:
        axis = int(digit) - 1
        angle = math.radians(angles_deg[axis])
        c, s = math.cos(angle), math.sin(angle)
        if axis == 0:
            rot = [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
        elif axis == 1:
            rot = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
        else:
            rot = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
        total = _matmul(total, rot)
    return total


class Transform:
    """The programmable transformations of one path at one moment (built from PathState)."""

    def __init__(self, mirror: dict[str, float], scale: dict | None, rotation: dict | None,
                 feature: dict | None):
        self.mirror = mirror                     # letter -> centre (SI)
        self.scale = scale                       # {"centre": {l: SI}, "factor": {l: f}}
        self.rotation = rotation                 # {"plane", "centre": [SI, SI], "angle_deg"}
        self.feature = feature                   # {"origin": [SI]*3, "matrix": 3x3}
        self.identity = not mirror and scale is None and rotation is None and feature is None
        linear = [list(r) for r in IDENTITY]
        for letter in mirror:
            if letter in UNIT:
                i = "XYZ".index(letter)
                linear[i][i] = -linear[i][i]
        if scale is not None:
            for letter, factor in scale["factor"].items():
                if letter in UNIT:
                    i = "XYZ".index(letter)
                    linear = _matmul([[factor if (r == c and r == i) else (1.0 if r == c else 0.0) for c in range(3)]
                                      for r in range(3)], linear)
        if rotation is not None:
            linear = _matmul(self._rotation_matrix(), linear)
        if feature is not None:
            linear = _matmul(feature["matrix"], linear)
        self.linear = linear

    def _rotation_matrix(self):
        a, b, n = PLANES[self.rotation["plane"]]
        angle = math.radians(self.rotation["angle_deg"])
        c, s = math.cos(angle), math.sin(angle)
        m = [list(r) for r in IDENTITY]
        ia, ib = "XYZ".index(a), "XYZ".index(b)
        m[ia][ia], m[ia][ib], m[ib][ia], m[ib][ib] = c, -s, s, c
        return m

    def forward(self, p: dict[str, float]) -> dict[str, float]:
        if self.identity:
            return dict(p)
        q = dict(p)
        for letter, centre in self.mirror.items():
            if letter in q:
                q[letter] = 2.0 * centre - q[letter]
        if self.scale is not None:
            for letter, factor in self.scale["factor"].items():
                if letter in q:
                    centre = self.scale["centre"].get(letter, 0.0)
                    q[letter] = centre + factor * (q[letter] - centre)
        if self.rotation is not None:
            a, b, _ = PLANES[self.rotation["plane"]]
            if a in q and b in q:
                ca, cb = self.rotation["centre"]
                angle = math.radians(self.rotation["angle_deg"])
                c, s = math.cos(angle), math.sin(angle)
                da, db = q[a] - ca, q[b] - cb
                q[a], q[b] = ca + c * da - s * db, cb + s * da + c * db
        if self.feature is not None:
            vec = [q.get("X", 0.0), q.get("Y", 0.0), q.get("Z", 0.0)]
            out = _apply(self.feature["matrix"], vec)
            origin = self.feature["origin"]
            for i, letter in enumerate("XYZ"):
                if letter in q:
                    q[letter] = origin[i] + out[i]
        return q

    def inverse(self, q: dict[str, float]) -> dict[str, float]:
        if self.identity:
            return dict(q)
        p = dict(q)
        if self.feature is not None:
            origin = self.feature["origin"]
            vec = [p.get("X", 0.0) - origin[0], p.get("Y", 0.0) - origin[1], p.get("Z", 0.0) - origin[2]]
            out = _apply(_transpose(self.feature["matrix"]), vec)
            for i, letter in enumerate("XYZ"):
                if letter in p:
                    p[letter] = out[i]
        if self.rotation is not None:
            a, b, _ = PLANES[self.rotation["plane"]]
            if a in p and b in p:
                ca, cb = self.rotation["centre"]
                angle = -math.radians(self.rotation["angle_deg"])
                c, s = math.cos(angle), math.sin(angle)
                da, db = p[a] - ca, p[b] - cb
                p[a], p[b] = ca + c * da - s * db, cb + s * da + c * db
        if self.scale is not None:
            for letter, factor in self.scale["factor"].items():
                if letter in p:
                    centre = self.scale["centre"].get(letter, 0.0)
                    p[letter] = centre + (p[letter] - centre) / factor
        for letter, centre in self.mirror.items():
            if letter in p:
                p[letter] = 2.0 * centre - p[letter]
        return p

    def arc_normal(self, plane: str) -> tuple[list[float], bool]:
        """(work-frame normal of a program arc in ``plane``, True when the sense is reversed against
        the plane's own normal axis direction mapped by the rotation part)."""
        a, b, n = PLANES[plane]
        ea = _apply(self.linear, list(UNIT[a]))
        eb = _apply(self.linear, list(UNIT[b]))
        normal = _cross(ea, eb)
        length = _norm(normal)
        if length == 0.0:
            raise ControllerAlarm("illegal_scale", "scaling factor 0 in the arc plane")
        normal = [v / length for v in normal]
        reference = _apply(self.linear, list(UNIT[n]))
        flipped = sum(normal[i] * reference[i] for i in range(3)) < 0.0
        if flipped:
            normal = [-v for v in normal]
        return normal, flipped

    def scales_uniformly(self, plane: str) -> bool:
        if self.scale is None:
            return True
        a, b, _ = PLANES[plane]
        fa = abs(self.scale["factor"].get(a, 1.0))
        fb = abs(self.scale["factor"].get(b, 1.0))
        return fa == fb

    def radius_factor(self, plane: str) -> float:
        """Factor applied to an R word under scaling (the larger in-plane magnification when they differ)."""
        if self.scale is None:
            return 1.0
        a, b, _ = PLANES[plane]
        return max(abs(self.scale["factor"].get(a, 1.0)), abs(self.scale["factor"].get(b, 1.0)))


def arc_in_plane(start: tuple[float, float], end: tuple[float, float], direction: int,
                 centre_offset: tuple[float, float] | None, radius: float | None, tolerance: float,
                 same_point: bool) -> dict:
    """Arc in plane coordinates (a, b): centre, signed sweep and radii.

    ``direction`` +1 = G03 (counter-clockwise about the plane normal), -1 = G02. ``same_point``: the
    start and end are the same point in counts (full circle with I/J/K; no motion with R).
    Raises ControllerAlarm("radius_tolerance") when the end point is not on the circle within
    ``tolerance`` (FANUC PS0020) and ("arc_no_radius") when neither R nor I/J/K is given.
    """
    sa, sb = start
    ea, eb = end
    if centre_offset is None and radius is None:
        raise ControllerAlarm("arc_no_radius", "G02/G03 without R or I, J, K")
    if centre_offset is not None:
        ca, cb = sa + centre_offset[0], sb + centre_offset[1]
        r_start = math.hypot(sa - ca, sb - cb)
        r_end = math.hypot(ea - ca, eb - cb)
        if r_start == 0.0:
            raise ControllerAlarm("illegal_radius", "arc centre on the start point")
        if abs(r_end - r_start) > tolerance:
            raise ControllerAlarm("radius_tolerance", f"start radius {r_start * 1e3:.4f} mm, end radius "
                                                      f"{r_end * 1e3:.4f} mm (tolerance {tolerance * 1e3:.4f} mm)")
        sweep = _sweep(math.atan2(sb - cb, sa - ca), math.atan2(eb - cb, ea - ca), direction, same_point)
        return {"centre": (ca, cb), "sweep": sweep, "r_start": r_start, "r_end": r_end}
    if same_point:
        return {"centre": None, "sweep": 0.0, "r_start": abs(radius), "r_end": abs(radius)}
    da, db = ea - sa, eb - sb
    chord = math.hypot(da, db)
    r = abs(radius)
    half = chord / 2.0
    if r < half:
        if half - r > tolerance:
            raise ControllerAlarm("radius_tolerance", f"R {r * 1e3:.4f} mm is less than half the chord "
                                                      f"{half * 1e3:.4f} mm (tolerance {tolerance * 1e3:.4f} mm)")
        r = half
    h = math.sqrt(max(r * r - half * half, 0.0))
    ma, mb = (sa + ea) / 2.0, (sb + eb) / 2.0
    left_a, left_b = -db / chord, da / chord          # unit vector to the left of start -> end
    side = direction * (1.0 if radius > 0.0 else -1.0)  # +1: centre to the left
    ca, cb = ma + side * h * left_a, mb + side * h * left_b
    sweep = _sweep(math.atan2(sb - cb, sa - ca), math.atan2(eb - cb, ea - ca), direction, False)
    return {"centre": (ca, cb), "sweep": sweep, "r_start": r, "r_end": r}


def _sweep(theta_start: float, theta_end: float, direction: int, full: bool) -> float:
    if full:
        return TWO_PI * direction
    if direction > 0:
        delta = math.fmod(theta_end - theta_start, TWO_PI)
        if delta <= 0.0:
            delta += TWO_PI
        return delta
    delta = math.fmod(theta_start - theta_end, TWO_PI)
    if delta <= 0.0:
        delta += TWO_PI
    return -delta


def arc_mid_point(centre: tuple[float, float], start: tuple[float, float], sweep: float) -> tuple[float, float]:
    angle = math.atan2(start[1] - centre[1], start[0] - centre[0]) + sweep / 2.0
    r = math.hypot(start[0] - centre[0], start[1] - centre[1])
    return centre[0] + r * math.cos(angle), centre[1] + r * math.sin(angle)
