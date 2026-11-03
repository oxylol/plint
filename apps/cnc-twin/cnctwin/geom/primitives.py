"""Primitive solids as meshes with face groups, revolved profiles (polylines with arcs), and analytic
ray-vs-primitive kernels (docs/models/geom.md "Primitives").

Meshes are closed, outward-oriented and deterministic. Curved surfaces are tessellated so that the
chord error (sagitta) is at most ``chordal_tol_m`` (default ``geom.tessellation_chordal_tolerance_m``),
with at least ``geom.tessellation_min_segments`` segments per full circle.

Revolved profiles live in the meridian half-plane ``(z, r)`` of the solid's local frame (axis +z):
points ``(z_i, r_i)`` with ``r >= 0`` and per segment a bulge ``b = tan(theta / 4)`` (``theta`` the signed
included angle, positive counter-clockwise in the (z, r) axes; 0 for a straight segment). An open
profile starts and ends on the axis (r = 0); a closed one (a torus) loops back to its first point.
Face group ``k`` of a revolved mesh is the surface swept by profile segment ``k``.

The ray kernels work in the primitive's local frame where noted and return the first hit with
``t0 <= t <= t1`` as ``out = [t, nx, ny, nz]`` (outward unit normal); the return value is 1 on a hit,
0 otherwise. Ray directions need not be unit vectors (``t`` is in units of ``|d|``).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from numba import njit, types

from . import transform as tf
from .mesh import Mesh
from .settings import param, resolve

F1 = types.float64[::1]
F2 = types.float64[:, ::1]
f8 = types.float64
i8 = types.int64
OPTS = dict(cache=True, nogil=True, error_model="numpy")
_EPS = float(np.finfo(np.float64).eps)
# Root polishing stops when the bracket is this many ulps of the root (or after the iteration cap).
ROOT_ULPS = 8.0
ROOT_MAX_ITER = 200
# An arc whose centre lies within this fraction of its radius from the axis is a sphere patch (the bulge
# construction rounds an on-axis centre to ~1e-16 of the radius).
AXIS_CENTRE_REL = 1e-12
# Angular slack when testing whether a hit lies on an arc's span (rounding of atan2 is ~1e-16 rad).
ARC_ANGLE_TOL_RAD = 1e-12

BOX_GROUP_NAMES = {0: "-x", 1: "+x", 2: "-y", 3: "+y", 4: "-z", 5: "+z"}


# ============================================================================ tessellation helpers

def circle_segments(radius_m: float, chordal_tol_m: float | None = None, min_segments: int | None = None) -> int:
    """Segments of a full circle so that the sagitta ``r (1 - cos(pi / n))`` is at most the tolerance."""
    tol = float(resolve(chordal_tol_m, "tessellation_chordal_tolerance_m"))
    nmin = int(resolve(min_segments, "tessellation_min_segments"))
    if radius_m <= 0.0 or tol >= radius_m:
        return nmin
    n = int(math.ceil(math.pi / math.acos(1.0 - tol / radius_m)))
    return max(nmin, n)


def arc_segments(radius_m: float, sweep_rad: float, chordal_tol_m: float | None = None,
                 min_segments: int | None = None) -> int:
    """Pieces of an arc of ``sweep_rad`` under the same chord rule (at least one piece)."""
    full = circle_segments(radius_m, chordal_tol_m, min_segments)
    return max(1, int(math.ceil(full * abs(sweep_rad) / (2.0 * math.pi) - 1e-9)))


def arc_centre(p0, p1, bulge: float) -> tuple[np.ndarray, float, float, float]:
    """``(centre, radius, start angle, signed sweep)`` of a bulge arc in the (z, r) plane."""
    a = np.asarray(p0, dtype=np.float64)
    b = np.asarray(p1, dtype=np.float64)
    chord = b - a
    length = float(np.hypot(*chord))
    if length == 0.0:
        raise ValueError("an arc segment has coincident end points")
    theta = 4.0 * math.atan(bulge)
    left = np.array([-chord[1], chord[0]]) / length
    centre = 0.5 * (a + b) + left * (0.5 * length) * ((1.0 - bulge * bulge) / (2.0 * bulge))
    radius = 0.5 * length / abs(math.sin(0.5 * theta))
    start = math.atan2(a[1] - centre[1], a[0] - centre[0])
    return centre, radius, start, theta


def profile_polyline(z, r, bulge=None, closed: bool = False, chordal_tol_m: float | None = None,
                     conservative: bool = False):
    """Polyline of a profile with arcs: ``(zs, rs, segment_of_piece)``.

    ``conservative`` (for collision envelopes of z-monotone profiles) replaces every arc that bulges away
    from the axis by its circumscribed polygon (tangent intersections), so the polyline never lies inside
    the true profile; arcs bulging toward the axis are already enclosed by their chords.
    """
    zs, rs, bul = _profile_arrays(z, r, bulge, closed)
    nseg = len(zs) if closed else len(zs) - 1
    out_z, out_r, seg_of = [float(zs[0])], [float(rs[0])], []
    for k in range(nseg):
        p0 = np.array([zs[k], rs[k]])
        p1 = np.array([zs[(k + 1) % len(zs)], rs[(k + 1) % len(zs)]])
        if bul[k] == 0.0:
            out_z.append(float(p1[0]))
            out_r.append(float(p1[1]))
            seg_of.append(k)
            continue
        centre, radius, start, sweep = arc_centre(p0, p1, bul[k])
        pieces = arc_segments(radius, sweep, chordal_tol_m)
        step = sweep / pieces
        outward = sweep < 0.0     # clockwise in (z, r) while z increases: the arc bulges away from the axis
        if conservative and outward:
            grow = radius / math.cos(0.5 * step)
            for i in range(pieces):
                ang = start + (i + 0.5) * step
                out_z.append(float(centre[0] + grow * math.cos(ang)))
                out_r.append(max(0.0, float(centre[1] + grow * math.sin(ang))))
                seg_of.append(k)
            out_z.append(float(p1[0]))
            out_r.append(float(p1[1]))
            seg_of.append(k)
        else:
            for i in range(1, pieces):
                ang = start + i * step
                out_z.append(float(centre[0] + radius * math.cos(ang)))
                out_r.append(max(0.0, float(centre[1] + radius * math.sin(ang))))
                seg_of.append(k)
            out_z.append(float(p1[0]))
            out_r.append(float(p1[1]))
            seg_of.append(k)
    if closed:
        out_z.pop()
        out_r.pop()
    return np.array(out_z), np.array(out_r), np.array(seg_of, np.int32)


def _profile_arrays(z, r, bulge, closed):
    zs = np.asarray(z, dtype=np.float64).reshape(-1)
    rs = np.asarray(r, dtype=np.float64).reshape(-1)
    if zs.shape != rs.shape or len(zs) < 2:
        raise ValueError("a profile needs at least two (z, r) points")
    if np.any(rs < 0.0):
        raise ValueError("profile radii must not be negative")
    nseg = len(zs) if closed else len(zs) - 1
    bul = np.zeros(nseg) if bulge is None else np.asarray(bulge, dtype=np.float64).reshape(-1)
    if len(bul) != nseg:
        raise ValueError(f"a profile with {nseg} segments needs {nseg} bulge values")
    return zs, rs, bul


def profile_signed_area(z, r, bulge=None, closed: bool = False) -> float:
    """Signed area enclosed by the profile (closed through the axis when open), positive counter-clockwise
    in (z, r): the exact Green's integral, arcs included."""
    zs, rs, bul = _profile_arrays(z, r, bulge, closed)
    nseg = len(zs) if closed else len(zs) - 1
    total = 0.0
    for k in range(nseg):
        a = (zs[k], rs[k])
        b = (zs[(k + 1) % len(zs)], rs[(k + 1) % len(zs)])
        if bul[k] == 0.0:
            total += a[0] * b[1] - b[0] * a[1]
        else:
            c, rad, start, sweep = arc_centre(a, b, bul[k])
            end = start + sweep
            total += rad * rad * sweep + rad * (c[0] * (math.sin(end) - math.sin(start))
                                                - c[1] * (math.cos(end) - math.cos(start)))
    if not closed:
        total += zs[-1] * rs[0] - zs[0] * rs[-1]
    return 0.5 * total


# ============================================================================ meshes

def box_mesh(lo, hi, name: str = "box") -> Mesh:
    """Axis-aligned box; groups 0..5 are the -x, +x, -y, +y, -z, +z sides."""
    a = np.asarray(lo, dtype=np.float64).reshape(3)
    b = np.asarray(hi, dtype=np.float64).reshape(3)
    if np.any(b <= a):
        raise ValueError(f"box max {b.tolist()} must exceed min {a.tolist()} on every axis")
    v = np.array([[a[0], a[1], a[2]], [b[0], a[1], a[2]], [b[0], b[1], a[2]], [a[0], b[1], a[2]],
                  [a[0], a[1], b[2]], [b[0], a[1], b[2]], [b[0], b[1], b[2]], [a[0], b[1], b[2]]])
    f = np.array([[0, 4, 7], [0, 7, 3],      # -x
                  [1, 2, 6], [1, 6, 5],      # +x
                  [0, 1, 5], [0, 5, 4],      # -y
                  [3, 7, 6], [3, 6, 2],      # +y
                  [0, 3, 2], [0, 2, 1],      # -z
                  [4, 5, 6], [4, 6, 7]])     # +z
    g = np.repeat(np.arange(6), 2)
    return Mesh(v, f, g, group_names=dict(BOX_GROUP_NAMES), name=name)


def oriented_box_mesh(pose_m, half_extents_m, name: str = "box") -> Mesh:
    """Box of half extents about the origin of the frame ``pose_m`` (4x4)."""
    h = np.asarray(half_extents_m, dtype=np.float64).reshape(3)
    return box_mesh(-h, h, name).transformed(pose_m)


def revolved_mesh(z, r, bulge=None, closed: bool = False, pose=None, chordal_tol_m: float | None = None,
                  min_segments: int | None = None, group_names: dict | None = None, name: str = "revolved") -> Mesh:
    """Solid of revolution of a profile about the local z axis, placed by ``pose`` (4x4, default identity)."""
    zs, rs, seg = profile_polyline(z, r, bulge, closed, chordal_tol_m)
    sigma = 1.0 if profile_signed_area(z, r, bulge, closed) > 0.0 else -1.0
    rmax = float(np.max(rs))
    if rmax <= 0.0:
        raise ValueError("a profile on the axis has no volume")
    n = circle_segments(rmax, chordal_tol_m, min_segments)
    ang = 2.0 * math.pi * np.arange(n) / n
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    verts: list[np.ndarray] = []
    ring_start: list[int] = []
    count = 0
    for zi, ri in zip(zs, rs):
        ring_start.append(count)
        if ri == 0.0:
            verts.append(np.array([[0.0, 0.0, zi]]))
            count += 1
        else:
            verts.append(np.column_stack([ri * cos_a, ri * sin_a, np.full(n, zi)]))
            count += n
    faces: list[list[int]] = []
    groups: list[int] = []
    npts = len(zs)
    nbands = npts if closed else npts - 1
    for b in range(nbands):
        ia, ib = b, (b + 1) % npts
        on_a, on_b = rs[ia] == 0.0, rs[ib] == 0.0
        if on_a and on_b:
            continue
        g = int(seg[b])
        for k in range(n):
            k1 = (k + 1) % n
            a0 = ring_start[ia] + (0 if on_a else k)
            a1 = ring_start[ia] + (0 if on_a else k1)
            b0 = ring_start[ib] + (0 if on_b else k)
            b1 = ring_start[ib] + (0 if on_b else k1)
            quad = (a0, a1, b1, b0) if sigma < 0.0 else (a0, b0, b1, a1)
            for tri in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])):
                if len(set(tri)) == 3:
                    faces.append(list(tri))
                    groups.append(g)
    mesh = Mesh(np.concatenate(verts), np.array(faces, np.int64), np.array(groups, np.int64),
                group_names=dict(group_names or {}), name=name)
    return mesh.transformed(pose) if pose is not None else mesh


def _axis_pose(p0, p1) -> tuple[np.ndarray, float]:
    a = np.asarray(p0, dtype=np.float64).reshape(3)
    b = np.asarray(p1, dtype=np.float64).reshape(3)
    axis = b - a
    h = float(np.linalg.norm(axis))
    if h == 0.0:
        raise ValueError("the two end points of a cylinder or cone coincide")
    return tf.pose(tf.frame_from_z(axis), a), h


def cylinder_mesh(p0, p1, radius_m: float, chordal_tol_m: float | None = None, name: str = "cylinder") -> Mesh:
    """Capped cylinder from ``p0`` to ``p1``; groups 0 cap at p0, 1 side, 2 cap at p1."""
    if radius_m <= 0.0:
        raise ValueError("cylinder radius must be positive")
    T, h = _axis_pose(p0, p1)
    return revolved_mesh([0.0, 0.0, h, h], [0.0, radius_m, radius_m, 0.0], pose=T, chordal_tol_m=chordal_tol_m,
                         group_names={0: "cap0", 1: "side", 2: "cap1"}, name=name)


def cone_mesh(p0, p1, r0_m: float, r1_m: float, chordal_tol_m: float | None = None, name: str = "cone") -> Mesh:
    """Capped cone frustum (radius r0 at p0, r1 at p1, either may be 0); groups 0 cap0, 1 side, 2 cap1
    (a zero-radius end has no cap, its group is absent)."""
    if r0_m < 0.0 or r1_m < 0.0 or (r0_m == 0.0 and r1_m == 0.0):
        raise ValueError("cone radii must be non-negative and not both zero")
    T, h = _axis_pose(p0, p1)
    z = [0.0, 0.0, h, h]
    r = [0.0, r0_m, r1_m, 0.0]
    names = {0: "cap0", 1: "side", 2: "cap1"}
    mesh = revolved_mesh(z, r, pose=T, chordal_tol_m=chordal_tol_m, group_names=names, name=name)
    present = set(np.unique(mesh.face_group).tolist())
    mesh.group_names = {k: v for k, v in names.items() if k in present}
    return mesh


def sphere_mesh(centre, radius_m: float, chordal_tol_m: float | None = None, name: str = "sphere") -> Mesh:
    """Sphere (one group, 0)."""
    if radius_m <= 0.0:
        raise ValueError("sphere radius must be positive")
    T = tf.translation(centre)
    return revolved_mesh([-radius_m, radius_m], [0.0, 0.0], [-1.0], pose=T, chordal_tol_m=chordal_tol_m,
                         group_names={0: "sphere"}, name=name)


def torus_mesh(centre, axis, major_m: float, minor_m: float, chordal_tol_m: float | None = None,
               name: str = "torus") -> Mesh:
    """Ring torus about ``axis`` through ``centre`` (minor < major); groups 0 outer half, 1 inner half."""
    if not 0.0 < minor_m < major_m:
        raise ValueError("a ring torus needs 0 < minor radius < major radius")
    T = tf.pose(tf.frame_from_z(axis), centre)
    return revolved_mesh([-minor_m, minor_m], [major_m, major_m], [-1.0, -1.0], closed=True, pose=T,
                         chordal_tol_m=chordal_tol_m, group_names={0: "outer", 1: "inner"}, name=name)


def _si_vec(spec, key: str):
    for suffix, scale in (("_m", 1.0), ("_mm", 1e-3)):
        if key + suffix in spec and spec[key + suffix] is not None:
            value = spec[key + suffix]
            value = getattr(value, "value", value)
            return np.asarray([float(getattr(v, "value", v)) for v in value]) * scale
    raise KeyError(f"primitive needs {key}_mm")


def _si_num(spec, key: str) -> float:
    for suffix, scale in (("_m", 1.0), ("_mm", 1e-3)):
        if key + suffix in spec and spec[key + suffix] is not None:
            return float(getattr(spec[key + suffix], "value", spec[key + suffix])) * scale
    raise KeyError(f"primitive needs {key}_mm")


def mesh_from_primitive(spec, base_dir=None, chordal_tol_m: float | None = None, name: str = "") -> Mesh:
    """Mesh of one machine/fixture geometry primitive (db/_templates/machine.yaml ``geometry[].primitives``):
    ``{box: {min_mm, max_mm}}``, ``{cylinder: {p0_mm, p1_mm, radius_mm}}``, ``{cone: {p0_mm, p1_mm, r0_mm,
    r1_mm}}``, ``{sphere: {centre_mm, radius_mm}}``, ``{torus: {centre_mm, axis, major_mm, minor_mm}}``,
    ``{mesh: {file, scale, translate_mm, rotate_deg, order}}`` (file relative to ``base_dir``). Raw (mm) and
    SI views (``_m`` keys, cnctwin.db.simap) are both accepted."""
    if len(spec) != 1:
        raise ValueError(f"a primitive has exactly one kind key, not {sorted(spec)}")
    kind = next(iter(spec))
    body = spec[kind]
    if kind == "box":
        return box_mesh(_si_vec(body, "min"), _si_vec(body, "max"), name or "box")
    if kind == "cylinder":
        return cylinder_mesh(_si_vec(body, "p0"), _si_vec(body, "p1"), _si_num(body, "radius"), chordal_tol_m,
                             name or "cylinder")
    if kind == "cone":
        return cone_mesh(_si_vec(body, "p0"), _si_vec(body, "p1"), _si_num(body, "r0"), _si_num(body, "r1"),
                         chordal_tol_m, name or "cone")
    if kind == "sphere":
        return sphere_mesh(_si_vec(body, "centre"), _si_num(body, "radius"), chordal_tol_m, name or "sphere")
    if kind == "torus":
        axis = body.get("axis") if hasattr(body, "get") else body["axis"]
        return torus_mesh(_si_vec(body, "centre"), [float(getattr(v, "value", v)) for v in axis],
                          _si_num(body, "major"), _si_num(body, "minor"), chordal_tol_m, name or "torus")
    if kind == "mesh":
        from .io import read_mesh

        file = body["file"]
        path = Path(str(file))
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
        mesh = read_mesh(path)
        scale = float(getattr(body.get("scale", 1.0), "value", body.get("scale", 1.0)) or 1.0)
        if scale != 1.0:
            mesh = mesh.transformed(np.diag([scale, scale, scale, 1.0]))
        pose_keys = {k: body[k] for k in body if k in ("translate_mm", "translate_m", "rotate_deg", "rotate_rad",
                                                         "order")}
        if hasattr(body, "matrix") and callable(body.matrix):
            mesh = mesh.transformed(body.matrix())
        elif pose_keys:
            mesh = mesh.transformed(tf.pose_from_dict(pose_keys))
        mesh.name = name or path.stem
        return mesh
    raise ValueError(f"unknown primitive kind {kind!r} (box, cylinder, cone, sphere, torus, mesh)")


# ============================================================================ polynomial roots (Numba)

@njit(i8(f8, f8, f8, f8, f8, F1), **OPTS)
def roots2_nb(a, b, c, lo, hi, out):
    """Real roots of a t^2 + b t + c in [lo, hi], ascending, into out (at most 2); stable formulas
    (no cancellation: q = -(b + sign(b) sqrt(disc)) / 2)."""
    n = 0
    if a == 0.0:
        if b != 0.0:
            t = -c / b
            if lo <= t <= hi:
                out[0] = t
                n = 1
        return n
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return 0
    sq = math.sqrt(disc)
    q = -0.5 * (b + sq) if b >= 0.0 else -0.5 * (b - sq)
    if q == 0.0:
        r1 = 0.0
        r2 = 0.0
    else:
        r1 = q / a
        r2 = c / q
    if r1 > r2:
        r1, r2 = r2, r1
    if lo <= r1 <= hi:
        out[n] = r1
        n += 1
    if disc > 0.0 and lo <= r2 <= hi:
        out[n] = r2
        n += 1
    return n


@njit(f8(F1, i8, f8), **OPTS)
def poly_eval_nb(c, deg, t):
    """Horner evaluation of c[0] t^deg + ... + c[deg]."""
    v = c[0]
    for i in range(1, deg + 1):
        v = v * t + c[i]
    return v


@njit(f8(F1, i8, f8), **OPTS)
def _poly_scale(c, deg, t):
    """Sum of |c_i| |t|^(deg - i): the magnitude the evaluation's rounding is relative to."""
    v = abs(c[0])
    at = abs(t)
    for i in range(1, deg + 1):
        v = v * at + abs(c[i])
    return v


@njit(f8(F1, i8, f8, f8, f8, f8), **OPTS)
def _bracket_root(c, deg, a, b, fa, fb):
    """Root of a polynomial monotone on [a, b] with f(a) f(b) < 0: bisection safeguarding Newton steps
    (rtsafe, Press et al. 2007, 9.4)."""
    d = np.empty(deg)
    for i in range(deg):
        d[i] = c[i] * (deg - i)
    if fa > 0.0:
        a, b = b, a           # orient so f(a) < 0 < f(b)
    x = 0.5 * (a + b)
    for _ in range(ROOT_MAX_ITER):
        fx = poly_eval_nb(c, deg, x)
        if fx == 0.0:
            return x
        if fx < 0.0:
            a = x
        else:
            b = x
        dfx = poly_eval_nb(d, deg - 1, x)
        newton = x - fx / dfx if dfx != 0.0 else 0.5 * (a + b)
        lo_ = min(a, b)
        hi_ = max(a, b)
        x_new = newton if lo_ < newton < hi_ else 0.5 * (a + b)
        if abs(b - a) <= ROOT_ULPS * _EPS * max(abs(a), abs(b), 1e-300):
            return x_new
        if x_new == x:
            return x
        x = x_new
    return x


@njit(i8(F1, i8, f8, f8, F1), **OPTS)
def poly_roots_nb(c, deg, lo, hi, out):
    """All real roots in [lo, hi] (ascending) of a polynomial of degree <= 4 with coefficients c[0..deg]
    (highest first), by recursive bracketing between the roots of the derivative (each interval between
    critical points is monotone, so it holds at most one root). A critical point where |f| is within
    rounding of zero is a double root (tangency). Returns the number of roots written."""
    while deg > 0 and c[0] == 0.0:
        c = c[1:].copy()
        deg -= 1
    if deg <= 0:
        return 0
    if deg == 1:
        t = -c[1] / c[0]
        if lo <= t <= hi:
            out[0] = t
            return 1
        return 0
    if deg == 2:
        return roots2_nb(c[0], c[1], c[2], lo, hi, out)
    # every root satisfies |t| < 1 + max |c_i / c_0| (Cauchy's bound): an infinite or huge search
    # interval is clipped to it, so all knots below are finite
    bound = 0.0
    for i in range(1, deg + 1):
        bound = max(bound, abs(c[i] / c[0]))
    bound = 1.0 + bound
    lo = max(lo, -bound)
    hi = min(hi, bound)
    if lo > hi:
        return 0
    # critical points (roots of the derivative) split [lo, hi] into monotone pieces
    d = np.empty(deg)
    for i in range(deg):
        d[i] = c[i] * (deg - i)
    crit = np.empty(4)
    ncrit = poly_roots_nb(d, deg - 1, lo, hi, crit)
    knots = np.empty(ncrit + 2)
    knots[0] = lo
    for i in range(ncrit):
        knots[i + 1] = crit[i]
    knots[ncrit + 1] = hi
    n = 0
    fprev = poly_eval_nb(c, deg, knots[0])
    if fprev == 0.0:
        out[n] = knots[0]
        n += 1
    for i in range(1, ncrit + 2):
        x = knots[i]
        fx = poly_eval_nb(c, deg, x)
        tol = 64.0 * _EPS * _poly_scale(c, deg, x)
        if fx == 0.0 or (i < ncrit + 1 and abs(fx) <= tol):
            if n == 0 or out[n - 1] != x:
                out[n] = x
                n += 1
            fx = 0.0
        elif fprev != 0.0 and (fprev < 0.0) != (fx < 0.0):
            root = _bracket_root(c, deg, knots[i - 1], x, fprev, fx)
            if n == 0 or out[n - 1] != root:
                out[n] = root
                n += 1
        fprev = fx
    return n


def poly_roots_ref(coeffs, lo: float, hi: float) -> np.ndarray:
    """Reference: real roots in [lo, hi] from the companion-matrix eigenvalues (numpy.roots), polished by
    Newton steps, ascending."""
    c = np.trim_zeros(np.asarray(coeffs, dtype=np.float64), "f")
    if len(c) <= 1:
        return np.zeros(0)
    roots = np.roots(c)
    real = roots[np.abs(roots.imag) <= 1e-7 * np.maximum(1.0, np.abs(roots.real))].real
    dc = np.polyder(c)
    out = []
    for t in real:
        for _ in range(4):
            dv = np.polyval(dc, t)
            if dv != 0.0:
                t = t - np.polyval(c, t) / dv
        if lo <= t <= hi:
            out.append(float(t))
    return np.array(sorted(out))


# ============================================================================ ray kernels (Numba)

@njit(i8(F1, F1, F1, F1, f8, f8, F1), **OPTS)
def ray_box_nb(o, d, lo, hi, t0, t1, out):
    """First crossing of an axis-aligned box surface in [t0, t1] (slab method, Kay & Kajiya 1986, with
    zero direction components handled explicitly). From inside, the hit is the exit."""
    tn = -np.inf
    tf_ = np.inf
    an = -1
    af = -1
    sn = 0.0
    sf = 0.0
    for k in range(3):
        if d[k] == 0.0:
            if o[k] < lo[k] or o[k] > hi[k]:
                return 0
            continue
        inv = 1.0 / d[k]
        ta = (lo[k] - o[k]) * inv
        tb = (hi[k] - o[k]) * inv
        if ta <= tb:
            if ta > tn:
                tn = ta
                an = k
                sn = -1.0
            if tb < tf_:
                tf_ = tb
                af = k
                sf = 1.0
        else:
            if tb > tn:
                tn = tb
                an = k
                sn = 1.0
            if ta < tf_:
                tf_ = ta
                af = k
                sf = -1.0
    if tn > tf_:
        return 0
    if tn >= t0 and tn <= t1 and an >= 0:
        out[0] = tn
        out[1] = 0.0
        out[2] = 0.0
        out[3] = 0.0
        out[1 + an] = sn
        return 1
    if tf_ >= t0 and tf_ <= t1 and af >= 0:
        out[0] = tf_
        out[1] = 0.0
        out[2] = 0.0
        out[3] = 0.0
        out[1 + af] = sf
        return 1
    return 0


@njit(i8(F1, F1, F1, f8, f8, f8, F1), **OPTS)
def ray_sphere_nb(o, d, c, radius, t0, t1, out):
    """First hit of a sphere in [t0, t1]."""
    ox = o[0] - c[0]
    oy = o[1] - c[1]
    oz = o[2] - c[2]
    a = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    b = 2.0 * (ox * d[0] + oy * d[1] + oz * d[2])
    cc = ox * ox + oy * oy + oz * oz - radius * radius
    roots = np.empty(2)
    n = roots2_nb(a, b, cc, t0, t1, roots)
    if n == 0:
        return 0
    t = roots[0]
    out[0] = t
    for k in range(3):
        out[1 + k] = (o[k] + t * d[k] - c[k]) / radius
    return 1


@njit(types.void(F1, F2), **OPTS)
def _frame_of_axis(axis, R):
    """Columns x, y, z of a frame with z along the unit ``axis`` (deterministic perpendicular x)."""
    ax = abs(axis[0])
    ay = abs(axis[1])
    az = abs(axis[2])
    if ax <= ay and ax <= az:
        ex, ey, ez = 1.0, 0.0, 0.0
    elif ay <= az:
        ex, ey, ez = 0.0, 1.0, 0.0
    else:
        ex, ey, ez = 0.0, 0.0, 1.0
    # x = normalize(axis x e)
    xx = axis[1] * ez - axis[2] * ey
    xy = axis[2] * ex - axis[0] * ez
    xz = axis[0] * ey - axis[1] * ex
    n = math.sqrt(xx * xx + xy * xy + xz * xz)
    xx /= n
    xy /= n
    xz /= n
    R[0, 0] = xx
    R[1, 0] = xy
    R[2, 0] = xz
    R[0, 1] = axis[1] * xz - axis[2] * xy
    R[1, 1] = axis[2] * xx - axis[0] * xz
    R[2, 1] = axis[0] * xy - axis[1] * xx
    R[0, 2] = axis[0]
    R[1, 2] = axis[1]
    R[2, 2] = axis[2]


@njit(i8(f8, f8, f8, f8, f8, f8, F2, f8, f8, f8, F1, F1), **OPTS)
def _ray_profile_local(ox, oy, oz, dx, dy, dz, seg, sigma, t0, t1, scratch, out):
    """First hit of the surface swept by a profile in the local frame (axis z). ``seg`` rows are
    (z0, r0, z1, r1, bulge, zc, rc, radius, start, sweep); ``sigma`` +1 for a counter-clockwise profile.
    out = [t, n_x, n_y, n_z] with the meridian outward normal mapped to 3-D; ``scratch`` holds at least
    4 roots. Arcs are torus patches (spheres when the centre is on the axis), solved after shifting the
    ray origin to its closest approach to the arc centre (conditioning of the quartic)."""
    best = np.inf
    bnz = 0.0
    bnr = 0.0
    bx = 0.0
    by = 0.0
    for k in range(seg.shape[0]):
        z0 = seg[k, 0]
        r0 = seg[k, 1]
        z1 = seg[k, 2]
        r1 = seg[k, 3]
        bulge = seg[k, 4]
        if bulge == 0.0:
            dzs = z1 - z0
            drs = r1 - r0
            length = math.sqrt(dzs * dzs + drs * drs)
            if length == 0.0:
                continue
            nz = sigma * drs / length
            nr = -sigma * dzs / length
            if dzs == 0.0:
                if dz == 0.0:
                    continue
                t = (z0 - oz) / dz
                if t < t0 or t > t1 or t >= best:
                    continue
                px = ox + t * dx
                py = oy + t * dy
                rho = math.sqrt(px * px + py * py)
                if rho >= min(r0, r1) and rho <= max(r0, r1):
                    best = t
                    bnz = nz
                    bnr = nr
                    bx = px
                    by = py
                continue
            m = drs / dzs
            c = r0 - m * z0
            a2 = dx * dx + dy * dy - m * m * dz * dz
            a1 = 2.0 * (ox * dx + oy * dy - m * dz * (m * oz + c))
            a0 = ox * ox + oy * oy - (m * oz + c) * (m * oz + c)
            nroot = roots2_nb(a2, a1, a0, t0, min(t1, best), scratch)
            zlo = min(z0, z1)
            zhi = max(z0, z1)
            for i in range(nroot):
                t = scratch[i]
                pz = oz + t * dz
                if pz >= zlo and pz <= zhi and m * pz + c >= 0.0 and t < best:
                    best = t
                    bnz = nz
                    bnr = nr
                    bx = ox + t * dx
                    by = oy + t * dy
                    break
        else:
            zc = seg[k, 5]
            rc = seg[k, 6]
            rad = seg[k, 7]
            start = seg[k, 8]
            sweep = seg[k, 9]
            orient = sigma if sweep > 0.0 else -sigma
            # shift the origin along the ray toward the arc centre (conditioning of the quartic)
            pzc = oz - zc
            dd = dx * dx + dy * dy + dz * dz
            sh = -(ox * dx + oy * dy + pzc * dz) / dd
            qx = ox + sh * dx
            qy = oy + sh * dy
            qz = pzc + sh * dz
            lo_t = t0 - sh
            hi_t = min(t1, best) - sh
            if hi_t < lo_t:
                continue
            nroot = 0
            if rc <= AXIS_CENTRE_REL * rad:
                nroot = roots2_nb(dd, 2.0 * (qx * dx + qy * dy + qz * dz), qx * qx + qy * qy + qz * qz - rad * rad,
                                  lo_t, hi_t, scratch)
            else:
                # (|p|^2 + rc^2 - rad^2)^2 = 4 rc^2 (px^2 + py^2), ray p = q + t d
                alpha = qx * dx + qy * dy + qz * dz
                beta = qx * qx + qy * qy + qz * qz + rc * rc - rad * rad
                e = dx * dx + dy * dy
                f = qx * dx + qy * dy
                g = qx * qx + qy * qy
                cf = np.empty(5)
                cf[0] = dd * dd
                cf[1] = 4.0 * dd * alpha
                cf[2] = 4.0 * alpha * alpha + 2.0 * dd * beta - 4.0 * rc * rc * e
                cf[3] = 4.0 * alpha * beta - 8.0 * rc * rc * f
                cf[4] = beta * beta - 4.0 * rc * rc * g
                nroot = poly_roots_nb(cf, 4, lo_t, hi_t, scratch)
            for i in range(nroot):
                tq = scratch[i]
                px = qx + tq * dx
                py = qy + tq * dy
                pz = qz + tq * dz
                rho = math.sqrt(px * px + py * py)
                uz = pz
                ur = rho - rc
                res_true = abs(uz * uz + ur * ur - rad * rad)
                res_mirror = abs(uz * uz + (rho + rc) * (rho + rc) - rad * rad)
                if rc > AXIS_CENTRE_REL * rad and res_mirror < res_true:
                    continue
                phi = math.atan2(ur, uz)
                rel = phi - start
                if sweep > 0.0:
                    rel = rel % (2.0 * math.pi)
                    inside = rel <= sweep + ARC_ANGLE_TOL_RAD
                else:
                    rel = (-rel) % (2.0 * math.pi)
                    inside = rel <= -sweep + ARC_ANGLE_TOL_RAD
                if not inside:
                    continue
                t = tq + sh
                if t < best and t >= t0 and t <= t1:
                    un = math.sqrt(uz * uz + ur * ur)
                    best = t
                    bnz = orient * uz / un
                    bnr = orient * ur / un
                    bx = ox + t * dx
                    by = oy + t * dy
                    break
    if best == np.inf:
        return 0
    out[0] = best
    rho = math.sqrt(bx * bx + by * by)
    if rho > 0.0:
        out[1] = bnr * bx / rho
        out[2] = bnr * by / rho
    else:
        out[1] = 0.0
        out[2] = 0.0
    out[3] = bnz
    nn = math.sqrt(out[1] * out[1] + out[2] * out[2] + out[3] * out[3])
    if nn > 0.0:
        out[1] /= nn
        out[2] /= nn
        out[3] /= nn
    return 1


@njit(i8(F1, F1, F2, F2, f8, f8, f8, F1), **OPTS)
def ray_revolved_nb(o, d, T_L_W, seg, sigma, t0, t1, out):
    """First hit of a solid of revolution; ``T_L_W`` maps world to the solid's local frame (axis z),
    ``seg`` the profile table of :func:`profile_table`, ``sigma`` its orientation. Normal in world."""
    ox = T_L_W[0, 0] * o[0] + T_L_W[0, 1] * o[1] + T_L_W[0, 2] * o[2] + T_L_W[0, 3]
    oy = T_L_W[1, 0] * o[0] + T_L_W[1, 1] * o[1] + T_L_W[1, 2] * o[2] + T_L_W[1, 3]
    oz = T_L_W[2, 0] * o[0] + T_L_W[2, 1] * o[1] + T_L_W[2, 2] * o[2] + T_L_W[2, 3]
    dx = T_L_W[0, 0] * d[0] + T_L_W[0, 1] * d[1] + T_L_W[0, 2] * d[2]
    dy = T_L_W[1, 0] * d[0] + T_L_W[1, 1] * d[1] + T_L_W[1, 2] * d[2]
    dz = T_L_W[2, 0] * d[0] + T_L_W[2, 1] * d[1] + T_L_W[2, 2] * d[2]
    scratch = np.empty(8)
    loc = np.empty(4)
    hit = _ray_profile_local(ox, oy, oz, dx, dy, dz, seg, sigma, t0, t1, scratch, loc)
    if hit == 0:
        return 0
    out[0] = loc[0]
    # normal back to world: R^T n (T_L_W is rigid)
    for k in range(3):
        out[1 + k] = T_L_W[0, k] * loc[1] + T_L_W[1, k] * loc[2] + T_L_W[2, k] * loc[3]
    return 1


@njit(i8(F1, F1, F1, F1, f8, f8, f8, f8, F1), **OPTS)
def ray_cone_nb(o, d, p0, p1, r0, r1, t0, t1, out):
    """First hit of a capped cone frustum (r0 at p0, r1 at p1; a cylinder when equal)."""
    ax = np.empty(3)
    h = math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2 + (p1[2] - p0[2]) ** 2)
    for k in range(3):
        ax[k] = (p1[k] - p0[k]) / h
    R = np.empty((3, 3))
    _frame_of_axis(ax, R)
    T = np.empty((4, 4))
    for r in range(3):
        for c in range(3):
            T[r, c] = R[c, r]
        T[r, 3] = -(R[0, r] * p0[0] + R[1, r] * p0[1] + R[2, r] * p0[2])
    T[3, 0] = 0.0
    T[3, 1] = 0.0
    T[3, 2] = 0.0
    T[3, 3] = 1.0
    seg = np.zeros((3, 10))
    seg[0, 0] = 0.0
    seg[0, 1] = 0.0
    seg[0, 2] = 0.0
    seg[0, 3] = r0
    seg[1, 0] = 0.0
    seg[1, 1] = r0
    seg[1, 2] = h
    seg[1, 3] = r1
    seg[2, 0] = h
    seg[2, 1] = r1
    seg[2, 2] = h
    seg[2, 3] = 0.0
    return ray_revolved_nb(o, d, T, seg, -1.0, t0, t1, out)


@njit(i8(F1, F1, F1, F1, f8, f8, f8, F1), **OPTS)
def ray_cylinder_nb(o, d, p0, p1, radius, t0, t1, out):
    """First hit of a capped cylinder from p0 to p1."""
    return ray_cone_nb(o, d, p0, p1, radius, radius, t0, t1, out)


@njit(i8(F1, F1, F1, F1, f8, f8, f8, f8, F1), **OPTS)
def ray_torus_nb(o, d, c, axis, major, minor, t0, t1, out):
    """First hit of a ring torus about the unit ``axis`` through ``c``."""
    R = np.empty((3, 3))
    _frame_of_axis(axis, R)
    T = np.empty((4, 4))
    for r in range(3):
        for k in range(3):
            T[r, k] = R[k, r]
        T[r, 3] = -(R[0, r] * c[0] + R[1, r] * c[1] + R[2, r] * c[2])
    T[3, 0] = 0.0
    T[3, 1] = 0.0
    T[3, 2] = 0.0
    T[3, 3] = 1.0
    seg = np.zeros((2, 10))
    # closed profile: two half circles about (zc, rc) = (0, major), counter-clockwise loop (sigma = +1)
    seg[0, 0] = 0.0
    seg[0, 1] = major - minor
    seg[0, 2] = 0.0
    seg[0, 3] = major + minor
    seg[0, 4] = 1.0
    seg[0, 5] = 0.0
    seg[0, 6] = major
    seg[0, 7] = minor
    seg[0, 8] = -0.5 * math.pi
    seg[0, 9] = math.pi
    seg[1, 0] = 0.0
    seg[1, 1] = major + minor
    seg[1, 2] = 0.0
    seg[1, 3] = major - minor
    seg[1, 4] = 1.0
    seg[1, 5] = 0.0
    seg[1, 6] = major
    seg[1, 7] = minor
    seg[1, 8] = 0.5 * math.pi
    seg[1, 9] = math.pi
    return ray_revolved_nb(o, d, T, seg, 1.0, t0, t1, out)


def profile_table(z, r, bulge=None, closed: bool = False) -> tuple[np.ndarray, float]:
    """``(seg [k, 10], sigma)`` for :func:`ray_revolved_nb`: per segment (z0, r0, z1, r1, bulge, zc, rc,
    radius, start, sweep) in the (z, r) plane; arc columns are 0 for straight segments."""
    zs, rs, bul = _profile_arrays(z, r, bulge, closed)
    nseg = len(zs) if closed else len(zs) - 1
    seg = np.zeros((nseg, 10))
    for k in range(nseg):
        a = (zs[k], rs[k])
        b = (zs[(k + 1) % len(zs)], rs[(k + 1) % len(zs)])
        seg[k, :5] = [a[0], a[1], b[0], b[1], bul[k]]
        if bul[k] != 0.0:
            c, rad, start, sweep = arc_centre(a, b, bul[k])
            seg[k, 5:] = [c[0], c[1], rad, start, sweep]
    sigma = 1.0 if profile_signed_area(z, r, bulge, closed) > 0.0 else -1.0
    return np.ascontiguousarray(seg), sigma


def _hit(fn, *args):
    out = np.zeros(4)
    if fn(*args, out):
        return float(out[0]), out[1:].copy()
    return None


def _v(x) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(-1))


def ray_box(o, d, lo, hi, t0: float = 0.0, t1: float = math.inf):
    """``(t, normal)`` of the first crossing of a box surface, or None."""
    return _hit(ray_box_nb, _v(o), _v(d), _v(lo), _v(hi), float(t0), float(t1))


def ray_sphere(o, d, centre, radius: float, t0: float = 0.0, t1: float = math.inf):
    return _hit(ray_sphere_nb, _v(o), _v(d), _v(centre), float(radius), float(t0), float(t1))


def ray_cylinder(o, d, p0, p1, radius: float, t0: float = 0.0, t1: float = math.inf):
    return _hit(ray_cylinder_nb, _v(o), _v(d), _v(p0), _v(p1), float(radius), float(t0), float(t1))


def ray_cone(o, d, p0, p1, r0: float, r1: float, t0: float = 0.0, t1: float = math.inf):
    return _hit(ray_cone_nb, _v(o), _v(d), _v(p0), _v(p1), float(r0), float(r1), float(t0), float(t1))


def ray_torus(o, d, centre, axis, major: float, minor: float, t0: float = 0.0, t1: float = math.inf):
    return _hit(ray_torus_nb, _v(o), _v(d), _v(centre), _v(tf.unit(axis)), float(major), float(minor),
                float(t0), float(t1))


def ray_revolved(o, d, z, r, bulge=None, closed: bool = False, pose=None, t0: float = 0.0, t1: float = math.inf):
    """First hit of the solid of revolution of a profile placed by ``pose`` (4x4 local -> world)."""
    seg, sigma = profile_table(z, r, bulge, closed)
    T = np.ascontiguousarray(tf.inverse(np.eye(4) if pose is None else pose))
    return _hit(ray_revolved_nb, _v(o), _v(d), T, seg, float(sigma), float(t0), float(t1))


KERNELS = (roots2_nb, poly_eval_nb, _poly_scale, _bracket_root, poly_roots_nb, ray_box_nb, ray_sphere_nb,
           _frame_of_axis, _ray_profile_local, ray_revolved_nb, ray_cone_nb, ray_cylinder_nb, ray_torus_nb)

__all__ = ["box_mesh", "oriented_box_mesh", "cylinder_mesh", "cone_mesh", "sphere_mesh", "torus_mesh",
           "revolved_mesh", "mesh_from_primitive", "profile_polyline", "profile_signed_area", "profile_table",
           "arc_centre", "circle_segments", "arc_segments", "ray_box", "ray_sphere", "ray_cylinder", "ray_cone",
           "ray_torus", "ray_revolved", "roots2_nb", "poly_eval_nb", "poly_roots_nb", "poly_roots_ref",
           "ray_box_nb", "ray_sphere_nb", "ray_cylinder_nb", "ray_cone_nb", "ray_torus_nb", "ray_revolved_nb",
           "BOX_GROUP_NAMES", "KERNELS", "param"]
