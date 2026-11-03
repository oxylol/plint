"""Overlap (interference) tests: triangle-triangle, triangle-box, box-box, and mesh-vs-mesh, mesh-vs-box,
mesh-vs-revolved-solid through the trees of :mod:`cnctwin.geom.bvh` (docs/models/geom.md "Overlap").

* Triangle-triangle: Moller (1997), interval overlap on the line where the planes meet; coplanar pairs
  by 2-D edge-edge and containment tests. A plane distance within the rounding of its evaluation counts
  as zero (relative threshold, no absolute epsilon). Touching counts as overlap.
* Triangle-box: separating axes (13) of Akenine-Moller (2001).
* Box-box: separating axes (15) of Gottschalk, Lin & Manocha (1996), with the parallel-edge guard of
  Ericson (2005, 4.4.1).
* Mesh-vs-mesh: simultaneous descent of two trees with the relative pose ``pose_A_B`` (tree B's frame
  in tree A's frame), oriented node boxes tested by separating axes, leaf triangles by Moller.
* Mesh-vs-revolved solid (tool and holder envelopes, a z-monotone profile r(z) about the solid's z axis):
  broad phase by the solid's oriented bounding box; narrow phase exact per triangle and profile segment
  by minimising the radial clearance ``h(z) = g(z) - R(z)`` where ``g(z)`` is the distance of the
  triangle's chord at height z from the axis. ``g`` is convex (partial minimisation of the convex
  distance-to-axis over the triangle, Boyd & Vandenberghe 2004, 3.2.5) and R is linear on a segment, so
  ``h`` is convex and a golden-section search (Kiefer 1953) finds its minimum to ``geom.clearance_z_
  tolerance_m``. A negative clearance is a radial penetration depth.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, types

from . import transform as tf
from .abi import STACK_SIZE, TREE_TYPE, MeshTree, contiguous
from .settings import resolve

F1 = types.float64[::1]
F2 = types.float64[:, ::1]
I1 = types.int32[::1]
I2 = types.int32[:, ::1]
f8 = types.float64
i8 = types.int64
b1 = types.boolean
OPTS = dict(cache=True, nogil=True, error_model="numpy")
TREE = TREE_TYPE
_EPS = float(np.finfo(np.float64).eps)
# A signed plane distance counts as zero below this many ulps of |n| |p - v0| (the rounding of the
# dot product of two differences); larger values are geometric, not rounding.
PLANE_ULPS = 32.0
# Ericson's guard on |R_ij| against the null cross product of near-parallel box edges.
SAT_PARALLEL_EPS = 64.0 * _EPS
INV_PHI = (math.sqrt(5.0) - 1.0) / 2.0


# ============================================================================ triangle-triangle (Moller 1997)

@njit(b1(f8, f8, f8, f8, f8, f8, f8, f8), **OPTS)
def _seg_seg_2d(ax, ay, bx, by, cx, cy, dx, dy):
    """Closed segments ab and cd intersect (orientation predicates, collinear overlap included)."""
    d1 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx)
    d2 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx)
    d3 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    d4 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax)
    if ((d1 > 0.0 and d2 < 0.0) or (d1 < 0.0 and d2 > 0.0)) and ((d3 > 0.0 and d4 < 0.0) or (d3 < 0.0 and d4 > 0.0)):
        return True
    if d1 == 0.0 and min(cx, dx) <= ax <= max(cx, dx) and min(cy, dy) <= ay <= max(cy, dy):
        return True
    if d2 == 0.0 and min(cx, dx) <= bx <= max(cx, dx) and min(cy, dy) <= by <= max(cy, dy):
        return True
    if d3 == 0.0 and min(ax, bx) <= cx <= max(ax, bx) and min(ay, by) <= cy <= max(ay, by):
        return True
    if d4 == 0.0 and min(ax, bx) <= dx <= max(ax, bx) and min(ay, by) <= dy <= max(ay, by):
        return True
    return False


@njit(b1(f8, f8, F1, F1, F1), **OPTS)
def _point_in_tri_2d(px, py, a, b, c):
    """Point inside or on the 2-D triangle (a, b, c) given as [x, y] pairs."""
    d1 = (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])
    d2 = (c[0] - b[0]) * (py - b[1]) - (c[1] - b[1]) * (px - b[0])
    d3 = (a[0] - c[0]) * (py - c[1]) - (a[1] - c[1]) * (px - c[0])
    neg = d1 < 0.0 or d2 < 0.0 or d3 < 0.0
    pos = d1 > 0.0 or d2 > 0.0 or d3 > 0.0
    return not (neg and pos)


@njit(b1(F1, F2, F2), **OPTS)
def _coplanar_tri_tri(n, V, U):
    """Overlap of two coplanar triangles: projected on the coordinate plane where |n| is largest."""
    ax = abs(n[0])
    ay = abs(n[1])
    az = abs(n[2])
    if ax >= ay and ax >= az:
        i0 = 1
        i1 = 2
    elif ay >= az:
        i0 = 0
        i1 = 2
    else:
        i0 = 0
        i1 = 1
    for e in range(3):
        a = V[e]
        b = V[(e + 1) % 3]
        for f in range(3):
            c = U[f]
            d = U[(f + 1) % 3]
            if _seg_seg_2d(a[i0], a[i1], b[i0], b[i1], c[i0], c[i1], d[i0], d[i1]):
                return True
    v2 = np.empty((3, 2))
    u2 = np.empty((3, 2))
    for k in range(3):
        v2[k, 0] = V[k, i0]
        v2[k, 1] = V[k, i1]
        u2[k, 0] = U[k, i0]
        u2[k, 1] = U[k, i1]
    if _point_in_tri_2d(v2[0, 0], v2[0, 1], u2[0], u2[1], u2[2]):
        return True
    if _point_in_tri_2d(u2[0, 0], u2[0, 1], v2[0], v2[1], v2[2]):
        return True
    return False


@njit(types.void(f8, f8, f8, f8, f8, f8, F1), **OPTS)
def _isect(vv0, vv1, vv2, d0, d1, d2, out):
    """Interval of a triangle on the intersection line: vertex 0 alone on its side of the other plane."""
    out[0] = vv0 + (vv1 - vv0) * d0 / (d0 - d1)
    out[1] = vv0 + (vv2 - vv0) * d0 / (d0 - d2)
    if out[0] > out[1]:
        t = out[0]
        out[0] = out[1]
        out[1] = t


@njit(i8(f8, f8, f8, f8, f8, f8, F1), **OPTS)
def _interval(vv0, vv1, vv2, d0, d1, d2, out):
    """Moller's COMPUTE_INTERVALS: 1 with out = [lo, hi], or 0 when the triangle is coplanar."""
    if d0 * d1 > 0.0:
        _isect(vv2, vv0, vv1, d2, d0, d1, out)
    elif d0 * d2 > 0.0:
        _isect(vv1, vv0, vv2, d1, d0, d2, out)
    elif d1 * d2 > 0.0 or d0 != 0.0:
        _isect(vv0, vv1, vv2, d0, d1, d2, out)
    elif d1 != 0.0:
        _isect(vv1, vv0, vv2, d1, d0, d2, out)
    elif d2 != 0.0:
        _isect(vv2, vv0, vv1, d2, d0, d1, out)
    else:
        return 0
    return 1


@njit(b1(F2, F2), **OPTS)
def _degenerate_tri_tri(V, U):
    """A zero-area triangle (segment or point) against the other: test its longest edge as a segment."""
    for which in range(2):
        A = V if which == 0 else U
        B = U if which == 0 else V
        bx1 = B[1, 0] - B[0, 0]
        by1 = B[1, 1] - B[0, 1]
        bz1 = B[1, 2] - B[0, 2]
        bx2 = B[2, 0] - B[0, 0]
        by2 = B[2, 1] - B[0, 1]
        bz2 = B[2, 2] - B[0, 2]
        nx = by1 * bz2 - bz1 * by2
        ny = bz1 * bx2 - bx1 * bz2
        nz = bx1 * by2 - by1 * bx2
        ax1 = A[1, 0] - A[0, 0]
        ay1 = A[1, 1] - A[0, 1]
        az1 = A[1, 2] - A[0, 2]
        ax2 = A[2, 0] - A[0, 0]
        ay2 = A[2, 1] - A[0, 1]
        az2 = A[2, 2] - A[0, 2]
        if (ay1 * az2 - az1 * ay2) ** 2 + (az1 * ax2 - ax1 * az2) ** 2 + (ax1 * ay2 - ay1 * ax2) ** 2 != 0.0:
            continue
        if nx == 0.0 and ny == 0.0 and nz == 0.0:
            # both degenerate: compare their extents only along their segments (points/segments overlap)
            for i in range(3):
                for j in range(3):
                    if A[i, 0] == B[j, 0] and A[i, 1] == B[j, 1] and A[i, 2] == B[j, 2]:
                        return True
            return False
        # segment test: each of A's three edges against triangle B
        for e in range(3):
            p = A[e]
            q = A[(e + 1) % 3]
            dp = nx * (p[0] - B[0, 0]) + ny * (p[1] - B[0, 1]) + nz * (p[2] - B[0, 2])
            dq = nx * (q[0] - B[0, 0]) + ny * (q[1] - B[0, 1]) + nz * (q[2] - B[0, 2])
            if (dp > 0.0 and dq > 0.0) or (dp < 0.0 and dq < 0.0):
                continue
            if dp == dq:
                s = 0.0
            else:
                s = dp / (dp - dq)
            x = np.empty(3)
            for k in range(3):
                x[k] = p[k] + s * (q[k] - p[k])
            # point in triangle B by same-side tests with the normal
            inside = True
            for k in range(3):
                a = B[k]
                b = B[(k + 1) % 3]
                ex = b[0] - a[0]
                ey = b[1] - a[1]
                ez = b[2] - a[2]
                wx = x[0] - a[0]
                wy = x[1] - a[1]
                wz = x[2] - a[2]
                cx = ey * wz - ez * wy
                cy = ez * wx - ex * wz
                cz = ex * wy - ey * wx
                if cx * nx + cy * ny + cz * nz < 0.0:
                    inside = False
            if inside:
                return True
        return False
    return False


@njit(b1(F2, F2), **OPTS)
def tri_tri_overlap_nb(V, U):
    """True when the closed triangles V [3, 3] and U [3, 3] intersect (Moller 1997)."""
    n1 = np.empty(3)
    n2 = np.empty(3)
    e1x = V[1, 0] - V[0, 0]
    e1y = V[1, 1] - V[0, 1]
    e1z = V[1, 2] - V[0, 2]
    e2x = V[2, 0] - V[0, 0]
    e2y = V[2, 1] - V[0, 1]
    e2z = V[2, 2] - V[0, 2]
    n1[0] = e1y * e2z - e1z * e2y
    n1[1] = e1z * e2x - e1x * e2z
    n1[2] = e1x * e2y - e1y * e2x
    nn1 = math.sqrt(n1[0] ** 2 + n1[1] ** 2 + n1[2] ** 2)
    du = np.empty(3)
    for k in range(3):
        rx = U[k, 0] - V[0, 0]
        ry = U[k, 1] - V[0, 1]
        rz = U[k, 2] - V[0, 2]
        du[k] = n1[0] * rx + n1[1] * ry + n1[2] * rz
        if abs(du[k]) <= PLANE_ULPS * _EPS * nn1 * math.sqrt(rx * rx + ry * ry + rz * rz):
            du[k] = 0.0
    if (du[0] > 0.0 and du[1] > 0.0 and du[2] > 0.0) or (du[0] < 0.0 and du[1] < 0.0 and du[2] < 0.0):
        return False
    f1x = U[1, 0] - U[0, 0]
    f1y = U[1, 1] - U[0, 1]
    f1z = U[1, 2] - U[0, 2]
    f2x = U[2, 0] - U[0, 0]
    f2y = U[2, 1] - U[0, 1]
    f2z = U[2, 2] - U[0, 2]
    n2[0] = f1y * f2z - f1z * f2y
    n2[1] = f1z * f2x - f1x * f2z
    n2[2] = f1x * f2y - f1y * f2x
    nn2 = math.sqrt(n2[0] ** 2 + n2[1] ** 2 + n2[2] ** 2)
    dv = np.empty(3)
    for k in range(3):
        rx = V[k, 0] - U[0, 0]
        ry = V[k, 1] - U[0, 1]
        rz = V[k, 2] - U[0, 2]
        dv[k] = n2[0] * rx + n2[1] * ry + n2[2] * rz
        if abs(dv[k]) <= PLANE_ULPS * _EPS * nn2 * math.sqrt(rx * rx + ry * ry + rz * rz):
            dv[k] = 0.0
    if (dv[0] > 0.0 and dv[1] > 0.0 and dv[2] > 0.0) or (dv[0] < 0.0 and dv[1] < 0.0 and dv[2] < 0.0):
        return False
    if nn1 == 0.0 or nn2 == 0.0:
        return _degenerate_tri_tri(V, U)
    if du[0] == 0.0 and du[1] == 0.0 and du[2] == 0.0:
        return _coplanar_tri_tri(n1, V, U)
    # direction of the intersection line and its largest component
    Dx = n1[1] * n2[2] - n1[2] * n2[1]
    Dy = n1[2] * n2[0] - n1[0] * n2[2]
    Dz = n1[0] * n2[1] - n1[1] * n2[0]
    idx = 0
    mx = abs(Dx)
    if abs(Dy) > mx:
        mx = abs(Dy)
        idx = 1
    if abs(Dz) > mx:
        idx = 2
    iv = np.empty(2)
    iu = np.empty(2)
    if _interval(V[0, idx], V[1, idx], V[2, idx], dv[0], dv[1], dv[2], iv) == 0:
        return _coplanar_tri_tri(n1, V, U)
    if _interval(U[0, idx], U[1, idx], U[2, idx], du[0], du[1], du[2], iu) == 0:
        return _coplanar_tri_tri(n1, V, U)
    if iv[1] < iu[0] or iu[1] < iv[0]:
        return False
    return True


# ============================================================================ triangle-box (Akenine-Moller 2001)

@njit(b1(F1, F1, F2), **OPTS)
def tri_box_overlap_nb(centre, half, tri):
    """True when the closed triangle tri [3, 3] and the axis-aligned box (centre, half extents) overlap."""
    v0x = tri[0, 0] - centre[0]
    v0y = tri[0, 1] - centre[1]
    v0z = tri[0, 2] - centre[2]
    v1x = tri[1, 0] - centre[0]
    v1y = tri[1, 1] - centre[1]
    v1z = tri[1, 2] - centre[2]
    v2x = tri[2, 0] - centre[0]
    v2y = tri[2, 1] - centre[1]
    v2z = tri[2, 2] - centre[2]
    hx = half[0]
    hy = half[1]
    hz = half[2]
    # box face normals
    if min(v0x, v1x, v2x) > hx or max(v0x, v1x, v2x) < -hx:
        return False
    if min(v0y, v1y, v2y) > hy or max(v0y, v1y, v2y) < -hy:
        return False
    if min(v0z, v1z, v2z) > hz or max(v0z, v1z, v2z) < -hz:
        return False
    e = np.empty((3, 3))
    e[0, 0] = v1x - v0x
    e[0, 1] = v1y - v0y
    e[0, 2] = v1z - v0z
    e[1, 0] = v2x - v1x
    e[1, 1] = v2y - v1y
    e[1, 2] = v2z - v1z
    e[2, 0] = v0x - v2x
    e[2, 1] = v0y - v2y
    e[2, 2] = v0z - v2z
    vx = np.empty(3)
    vy = np.empty(3)
    vz = np.empty(3)
    vx[0] = v0x
    vx[1] = v1x
    vx[2] = v2x
    vy[0] = v0y
    vy[1] = v1y
    vy[2] = v2y
    vz[0] = v0z
    vz[1] = v1z
    vz[2] = v2z
    # nine axes a_ij = e_i x edge_j
    for j in range(3):
        ex = e[j, 0]
        ey = e[j, 1]
        ez = e[j, 2]
        # x axis cross edge: (0, -ez, ey)
        pmin = np.inf
        pmax = -np.inf
        for k in range(3):
            p = -ez * vy[k] + ey * vz[k]
            pmin = min(pmin, p)
            pmax = max(pmax, p)
        rad = abs(ez) * hy + abs(ey) * hz
        if pmin > rad or pmax < -rad:
            return False
        # y axis cross edge: (ez, 0, -ex)
        pmin = np.inf
        pmax = -np.inf
        for k in range(3):
            p = ez * vx[k] - ex * vz[k]
            pmin = min(pmin, p)
            pmax = max(pmax, p)
        rad = abs(ez) * hx + abs(ex) * hz
        if pmin > rad or pmax < -rad:
            return False
        # z axis cross edge: (-ey, ex, 0)
        pmin = np.inf
        pmax = -np.inf
        for k in range(3):
            p = -ey * vx[k] + ex * vy[k]
            pmin = min(pmin, p)
            pmax = max(pmax, p)
        rad = abs(ey) * hx + abs(ex) * hy
        if pmin > rad or pmax < -rad:
            return False
    # triangle plane against the box
    nx = e[0, 1] * e[1, 2] - e[0, 2] * e[1, 1]
    ny = e[0, 2] * e[1, 0] - e[0, 0] * e[1, 2]
    nz = e[0, 0] * e[1, 1] - e[0, 1] * e[1, 0]
    d = -(nx * v0x + ny * v0y + nz * v0z)
    r = abs(nx) * hx + abs(ny) * hy + abs(nz) * hz
    if d > r or d < -r:
        return False
    return True


# ============================================================================ box-box (Gottschalk et al. 1996)

@njit(b1(F1, F1, F2, F1, F1, F1), **OPTS)
def obb_overlap_nb(ca, ha, R, t, cb, hb):
    """Box A (centre ca, half ha, axis-aligned in frame A) against box B (centre cb, half hb, axis-aligned in
    frame B), with p_A = R p_B + t. Separating axis theorem on 15 axes; touching counts as overlap."""
    T = np.empty(3)
    for i in range(3):
        T[i] = R[i, 0] * cb[0] + R[i, 1] * cb[1] + R[i, 2] * cb[2] + t[i] - ca[i]
    A = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            A[i, j] = abs(R[i, j]) + SAT_PARALLEL_EPS
    for i in range(3):
        rb = hb[0] * A[i, 0] + hb[1] * A[i, 1] + hb[2] * A[i, 2]
        if abs(T[i]) > ha[i] + rb:
            return False
    for j in range(3):
        ra = ha[0] * A[0, j] + ha[1] * A[1, j] + ha[2] * A[2, j]
        tb = T[0] * R[0, j] + T[1] * R[1, j] + T[2] * R[2, j]
        if abs(tb) > ra + hb[j]:
            return False
    for i in range(3):
        i1 = (i + 1) % 3
        i2 = (i + 2) % 3
        for j in range(3):
            j1 = (j + 1) % 3
            j2 = (j + 2) % 3
            ra = ha[i1] * A[i2, j] + ha[i2] * A[i1, j]
            rb = hb[j1] * A[i, j2] + hb[j2] * A[i, j1]
            if abs(T[i2] * R[i1, j] - T[i1] * R[i2, j]) > ra + rb:
                return False
    return True


# ============================================================================ tree traversals

@njit(types.void(F2, F2, F1, F2), **OPTS)
def _transform_tri(tri, R, t, out):
    for k in range(3):
        for r in range(3):
            out[k, r] = R[r, 0] * tri[k, 0] + R[r, 1] * tri[k, 1] + R[r, 2] * tri[k, 2] + t[r]


@njit(i8(TREE, TREE, F2, F1, b1, I2), **OPTS)
def tree_tree_overlap_nb(ta, tb, R, t, first_only, out_pairs):
    """Pairs (face of A, face of B) of intersecting triangles; tree B's coordinates map to A's by
    p_A = R p_B + t. Writes at most len(out_pairs) pairs and returns the total (1 with first_only)."""
    sa = np.empty(2 * STACK_SIZE, np.int64)
    sb = np.empty(2 * STACK_SIZE, np.int64)
    ca = np.empty(3)
    ha = np.empty(3)
    cb = np.empty(3)
    hb = np.empty(3)
    tri_b = np.empty((3, 3))
    cap = out_pairs.shape[0]
    n = 0
    sp = 1
    sa[0] = 0
    sb[0] = 0
    while sp > 0:
        sp -= 1
        na = sa[sp]
        nb = sb[sp]
        for k in range(3):
            ca[k] = 0.5 * (ta.node_lo[na, k] + ta.node_hi[na, k])
            ha[k] = 0.5 * (ta.node_hi[na, k] - ta.node_lo[na, k])
            cb[k] = 0.5 * (tb.node_lo[nb, k] + tb.node_hi[nb, k])
            hb[k] = 0.5 * (tb.node_hi[nb, k] - tb.node_lo[nb, k])
        if not obb_overlap_nb(ca, ha, R, t, cb, hb):
            continue
        leaf_a = ta.node_right[na] < 0
        leaf_b = tb.node_right[nb] < 0
        if leaf_a and leaf_b:
            for sb_ in range(tb.node_start[nb], tb.node_start[nb] + tb.node_count[nb]):
                _transform_tri(tb.tri_v[sb_], R, t, tri_b)
                for sa_ in range(ta.node_start[na], ta.node_start[na] + ta.node_count[na]):
                    if tri_tri_overlap_nb(ta.tri_v[sa_], tri_b):
                        if n < cap:
                            out_pairs[n, 0] = ta.tri_face[sa_]
                            out_pairs[n, 1] = tb.tri_face[sb_]
                        n += 1
                        if first_only:
                            return n
            continue
        size_a = ha[0] + ha[1] + ha[2]
        size_b = hb[0] + hb[1] + hb[2]
        if leaf_b or (not leaf_a and size_a >= size_b):
            sa[sp] = ta.node_right[na]
            sb[sp] = nb
            sp += 1
            sa[sp] = na + 1
            sb[sp] = nb
            sp += 1
        else:
            sa[sp] = na
            sb[sp] = tb.node_right[nb]
            sp += 1
            sa[sp] = na
            sb[sp] = nb + 1
            sp += 1
    return n


@njit(i8(TREE, F2, F1, F1, f8, I1), **OPTS)
def tree_box_overlap_nb(tree, R, t, half, margin, out_faces):
    """Faces of the tree overlapping an oriented box: the box is axis-aligned about the origin of its frame
    B (half extents ``half`` grown by ``margin``), p_A = R p_B + t. Returns the total count."""
    stack = np.empty(STACK_SIZE, np.int64)
    zero = np.zeros(3)
    hb = np.empty(3)
    for k in range(3):
        hb[k] = half[k] + margin
    # box B's frame as seen from A, and A's points mapped into B: p_B = R^T (p_A - t)
    Rt = np.empty((3, 3))
    tt = np.empty(3)
    for i in range(3):
        for j in range(3):
            Rt[i, j] = R[j, i]
    for i in range(3):
        tt[i] = -(Rt[i, 0] * t[0] + Rt[i, 1] * t[1] + Rt[i, 2] * t[2])
    ca = np.empty(3)
    ha = np.empty(3)
    tri = np.empty((3, 3))
    cap = out_faces.shape[0]
    n = 0
    sp = 1
    stack[0] = 0
    while sp > 0:
        sp -= 1
        node = stack[sp]
        for k in range(3):
            ca[k] = 0.5 * (tree.node_lo[node, k] + tree.node_hi[node, k])
            ha[k] = 0.5 * (tree.node_hi[node, k] - tree.node_lo[node, k])
        if not obb_overlap_nb(ca, ha, R, t, zero, hb):
            continue
        right = tree.node_right[node]
        if right < 0:
            for s in range(tree.node_start[node], tree.node_start[node] + tree.node_count[node]):
                _transform_tri(tree.tri_v[s], Rt, tt, tri)
                if tri_box_overlap_nb(zero, hb, tri):
                    if n < cap:
                        out_faces[n] = tree.tri_face[s]
                    n += 1
            continue
        stack[sp] = right
        sp += 1
        stack[sp] = node + 1
        sp += 1
    return n


@njit(f8(f8, f8, f8, f8), **OPTS)
def _seg_origin_dist(ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    l2 = dx * dx + dy * dy
    if l2 == 0.0:
        return math.sqrt(ax * ax + ay * ay)
    s = -(ax * dx + ay * dy) / l2
    if s < 0.0:
        s = 0.0
    elif s > 1.0:
        s = 1.0
    x = ax + s * dx
    y = ay + s * dy
    return math.sqrt(x * x + y * y)


@njit(f8(F2, f8), **OPTS)
def chord_axis_distance_nb(tri, z):
    """Distance from the z axis to the chord of triangle tri (local frame) at height z (the triangle
    clipped to the plane z); +inf when the plane misses the triangle."""
    ia = 0
    ib = 1
    ic = 2
    if tri[ib, 2] < tri[ia, 2]:
        ia, ib = ib, ia
    if tri[ic, 2] < tri[ib, 2]:
        ib, ic = ic, ib
    if tri[ib, 2] < tri[ia, 2]:
        ia, ib = ib, ia
    za = tri[ia, 2]
    zb = tri[ib, 2]
    zc = tri[ic, 2]
    if z < za or z > zc:
        return np.inf
    if zc == za:
        # the triangle lies in the plane: 2-D distance from the origin to the triangle
        px = np.empty(3)
        py = np.empty(3)
        for k in range(3):
            px[k] = tri[k, 0]
            py[k] = tri[k, 1]
        d1 = (px[1] - px[0]) * (0.0 - py[0]) - (py[1] - py[0]) * (0.0 - px[0])
        d2 = (px[2] - px[1]) * (0.0 - py[1]) - (py[2] - py[1]) * (0.0 - px[1])
        d3 = (px[0] - px[2]) * (0.0 - py[2]) - (py[0] - py[2]) * (0.0 - px[2])
        if not ((d1 < 0.0 or d2 < 0.0 or d3 < 0.0) and (d1 > 0.0 or d2 > 0.0 or d3 > 0.0)):
            return 0.0
        best = np.inf
        for k in range(3):
            best = min(best, _seg_origin_dist(px[k], py[k], px[(k + 1) % 3], py[(k + 1) % 3]))
        return best
    s = (z - za) / (zc - za)
    p0x = tri[ia, 0] + s * (tri[ic, 0] - tri[ia, 0])
    p0y = tri[ia, 1] + s * (tri[ic, 1] - tri[ia, 1])
    if z <= zb:
        if zb > za:
            s2 = (z - za) / (zb - za)
            p1x = tri[ia, 0] + s2 * (tri[ib, 0] - tri[ia, 0])
            p1y = tri[ia, 1] + s2 * (tri[ib, 1] - tri[ia, 1])
        else:
            p1x = tri[ib, 0]
            p1y = tri[ib, 1]
    else:
        if zc > zb:
            s3 = (z - zb) / (zc - zb)
            p1x = tri[ib, 0] + s3 * (tri[ic, 0] - tri[ib, 0])
            p1y = tri[ib, 1] + s3 * (tri[ic, 1] - tri[ib, 1])
        else:
            p1x = tri[ib, 0]
            p1y = tri[ib, 1]
    return _seg_origin_dist(p0x, p0y, p1x, p1y)


@njit(f8(F2, f8, f8, f8, f8, f8, i8, f8), **OPTS)
def _zone_value(tri, z, za, ra, zb, rb, mode, zref):
    """Clearance function on one zone: radial gap g(z) - R(z) (mode 0), or beyond an end of the solid the
    larger of the axial gap and the radial gap to the end radius (mode 1 below the start, 2 above the end)."""
    slope = (rb - ra) / (zb - za) if zb > za else 0.0
    radial = chord_axis_distance_nb(tri, z) - (ra + slope * (z - za))
    if mode == 1:
        return max(zref - z, radial)
    if mode == 2:
        return max(z - zref, radial)
    return radial


@njit(f8(F2, f8, f8, f8, f8, f8, i8, f8), **OPTS)
def _segment_clearance(tri, za, ra, zb, rb, ztol, mode, zref):
    """Minimum of the (convex) zone function over [za, zb]: golden section to ztol, end points included."""
    lo = za
    hi = zb
    best = min(_zone_value(tri, lo, za, ra, zb, rb, mode, zref), _zone_value(tri, hi, za, ra, zb, rb, mode, zref))
    if zb <= za:
        return best
    x1 = hi - INV_PHI * (hi - lo)
    x2 = lo + INV_PHI * (hi - lo)
    f1 = _zone_value(tri, x1, za, ra, zb, rb, mode, zref)
    f2 = _zone_value(tri, x2, za, ra, zb, rb, mode, zref)
    while hi - lo > ztol:
        if f1 <= f2:
            hi = x2
            x2 = x1
            f2 = f1
            x1 = hi - INV_PHI * (hi - lo)
            f1 = _zone_value(tri, x1, za, ra, zb, rb, mode, zref)
        else:
            lo = x1
            x1 = x2
            f1 = f2
            x2 = lo + INV_PHI * (hi - lo)
            f2 = _zone_value(tri, x2, za, ra, zb, rb, mode, zref)
    return min(best, f1, f2)


@njit(f8(F2, F1, F1, f8, f8), **OPTS)
def triangle_clearance_nb(tri, pz, pr, margin, ztol):
    """Clearance of triangle tri (in the solid's frame) to the solid of revolution of the z-monotone
    polyline (pz, pr): within the profile's z range the radial gap g(z) - R(z) (negative: radial
    penetration); within ``margin`` beyond either end the larger of the axial gap and the radial gap to
    the end radius. +inf when the triangle lies wholly beyond the grown solid's ends."""
    tz0 = min(tri[0, 2], tri[1, 2], tri[2, 2])
    tz1 = max(tri[0, 2], tri[1, 2], tri[2, 2])
    n = pz.shape[0]
    z_first = pz[0]
    z_last = pz[n - 1]
    r_first = 0.0
    r_last = 0.0
    for k in range(n):
        if pz[k] == z_first and pr[k] > r_first:
            r_first = pr[k]
        if pz[k] == z_last and pr[k] > r_last:
            r_last = pr[k]
    best = np.inf
    for k in range(-1, n):
        mode = 0
        zref = 0.0
        if k == -1:
            za = z_first - margin
            zb = z_first
            ra = r_first
            rb = r_first
            mode = 1
            zref = z_first
        elif k == n - 1:
            za = z_last
            zb = z_last + margin
            ra = r_last
            rb = r_last
            mode = 2
            zref = z_last
        else:
            za = pz[k]
            zb = pz[k + 1]
            ra = pr[k]
            rb = pr[k + 1]
            if zb <= za:
                continue
        if mode != 0 and margin <= 0.0:
            continue
        lo = max(za, tz0)
        hi = min(zb, tz1)
        if lo > hi:
            continue
        if zb > za:
            rlo = ra + (rb - ra) * (lo - za) / (zb - za)
            rhi = ra + (rb - ra) * (hi - za) / (zb - za)
        else:
            rlo = ra
            rhi = rb
        c = _segment_clearance(tri, lo, rlo, hi, rhi, ztol, mode, zref)
        if c < best:
            best = c
    return best


@njit(i8(TREE, F2, F1, F1, F1, f8, f8, I1, F1, F1), **OPTS)
def tree_revolved_overlap_nb(tree, R, t, pz, pr, margin, ztol, out_faces, out_clear, stats):
    """Faces of the tree within ``margin`` (radially, and axially past the ends) of a solid of revolution.
    The solid's frame S (axis z) maps to the tree's frame A by p_A = R p_S + t; (pz, pr) is its z-monotone
    profile. out_clear holds each reported face's clearance (negative: penetration). stats = [min
    clearance, face of min (-1 if none)]. Returns the total count."""
    stack = np.empty(STACK_SIZE, np.int64)
    n = pz.shape[0]
    rmax = 0.0
    for k in range(n):
        rmax = max(rmax, pr[k])
    cs = np.empty(3)
    hs = np.empty(3)
    cs[0] = 0.0
    cs[1] = 0.0
    cs[2] = 0.5 * (pz[0] + pz[n - 1])
    hs[0] = rmax + margin
    hs[1] = rmax + margin
    hs[2] = 0.5 * (pz[n - 1] - pz[0]) + margin
    Rt = np.empty((3, 3))
    tt = np.empty(3)
    for i in range(3):
        for j in range(3):
            Rt[i, j] = R[j, i]
    for i in range(3):
        tt[i] = -(Rt[i, 0] * t[0] + Rt[i, 1] * t[1] + Rt[i, 2] * t[2])
    ca = np.empty(3)
    ha = np.empty(3)
    tri = np.empty((3, 3))
    cap = out_faces.shape[0]
    count = 0
    stats[0] = np.inf
    stats[1] = -1.0
    sp = 1
    stack[0] = 0
    while sp > 0:
        sp -= 1
        node = stack[sp]
        for k in range(3):
            ca[k] = 0.5 * (tree.node_lo[node, k] + tree.node_hi[node, k])
            ha[k] = 0.5 * (tree.node_hi[node, k] - tree.node_lo[node, k])
        if not obb_overlap_nb(ca, ha, R, t, cs, hs):
            continue
        right = tree.node_right[node]
        if right < 0:
            for s in range(tree.node_start[node], tree.node_start[node] + tree.node_count[node]):
                _transform_tri(tree.tri_v[s], Rt, tt, tri)
                c = triangle_clearance_nb(tri, pz, pr, margin, ztol)
                if c <= margin:
                    face = tree.tri_face[s]
                    if count < cap:
                        out_faces[count] = face
                        out_clear[count] = c
                    count += 1
                    if c < stats[0] or (c == stats[0] and face < stats[1]):
                        stats[0] = c
                        stats[1] = face
            continue
        stack[sp] = right
        sp += 1
        stack[sp] = node + 1
        sp += 1
    return count


# ============================================================================ Python API

def _tri(x) -> np.ndarray:
    return contiguous(np.asarray(x, dtype=np.float64).reshape(3, 3))


def tri_tri_overlap(a, b) -> bool:
    return bool(tri_tri_overlap_nb(_tri(a), _tri(b)))


def tri_box_overlap(tri, centre, half) -> bool:
    return bool(tri_box_overlap_nb(contiguous(np.asarray(centre, float).reshape(3)),
                                   contiguous(np.asarray(half, float).reshape(3)), _tri(tri)))


def box_box_overlap(centre_a, half_a, pose_A_B, centre_b, half_b) -> bool:
    """Box A (axis-aligned in frame A) against box B (axis-aligned in frame B, placed by pose_A_B)."""
    T = np.asarray(pose_A_B, dtype=np.float64)
    return bool(obb_overlap_nb(contiguous(np.asarray(centre_a, float).reshape(3)),
                               contiguous(np.asarray(half_a, float).reshape(3)), contiguous(T[:3, :3]),
                               contiguous(T[:3, 3]), contiguous(np.asarray(centre_b, float).reshape(3)),
                               contiguous(np.asarray(half_b, float).reshape(3))))


def mesh_mesh_overlap(tree_a: MeshTree, tree_b: MeshTree, pose_A_B=None, first_only: bool = False,
                      max_pairs: int = 4096) -> np.ndarray:
    """``[k, 2]`` face pairs (A, B) of intersecting triangles (all of them unless ``first_only``)."""
    T = np.eye(4) if pose_A_B is None else np.asarray(pose_A_B, dtype=np.float64)
    if not tf.is_rigid(T, 1e-9):
        raise ValueError("mesh_mesh_overlap needs a rigid relative pose")
    R, t = contiguous(T[:3, :3]), contiguous(T[:3, 3])
    cap = max(1, int(max_pairs))
    while True:
        out = np.empty((cap, 2), np.int32)
        n = int(tree_tree_overlap_nb(tree_a, tree_b, R, t, bool(first_only), out))
        if n <= cap:
            return out[:n].copy()
        cap = n


def mesh_box_overlap(tree: MeshTree, pose_A_box, half_extents, margin: float = 0.0) -> np.ndarray:
    """Face ids overlapping an oriented box (half extents about its frame's origin, the frame placed by
    pose_A_box in the tree's frame), the box grown by ``margin``."""
    T = np.asarray(pose_A_box, dtype=np.float64)
    if not tf.is_rigid(T, 1e-9):
        raise ValueError("mesh_box_overlap needs a rigid box pose")
    R, t = contiguous(T[:3, :3]), contiguous(T[:3, 3])
    h = contiguous(np.asarray(half_extents, dtype=np.float64).reshape(3))
    cap = 1024
    while True:
        out = np.empty(cap, np.int32)
        n = int(tree_box_overlap_nb(tree, R, t, h, float(margin), out))
        if n <= cap:
            return out[:n].copy()
        cap = n


def mesh_revolved_overlap(tree: MeshTree, pose_A_S, z, r, margin: float = 0.0, z_tolerance: float | None = None):
    """Faces within ``margin`` of a z-monotone solid of revolution (tool/holder envelope; profile points
    (z_k, r_k) in its frame S, axis z, placed by pose_A_S): ``(faces [k], clearance [k], min_clearance,
    min_face)``. Clearance (see :func:`triangle_clearance_nb`) is the radial gap within the profile's z
    range (negative = radial penetration) and the larger of the axial and radial gaps beyond its ends; it
    equals the Euclidean gap on cylindrical portions and end faces and exceeds it by 1/cos(slope) on a
    conical portion. Overlap (clearance <= 0) is exact. min_face is -1 when nothing is near."""
    zs = contiguous(np.asarray(z, dtype=np.float64).reshape(-1))
    rs = contiguous(np.asarray(r, dtype=np.float64).reshape(-1))
    if len(zs) < 2 or zs.shape != rs.shape:
        raise ValueError("a revolved envelope needs at least two (z, r) points")
    if np.any(np.diff(zs) < 0.0) or np.any(rs < 0.0):
        raise ValueError("a revolved envelope profile must have non-decreasing z and non-negative r")
    T = np.asarray(pose_A_S, dtype=np.float64)
    if not tf.is_rigid(T, 1e-9):
        raise ValueError("mesh_revolved_overlap needs a rigid pose")
    ztol = float(resolve(z_tolerance, "clearance_z_tolerance_m"))
    R, t = contiguous(T[:3, :3]), contiguous(T[:3, 3])
    stats = np.zeros(2)
    cap = 1024
    while True:
        faces = np.empty(cap, np.int32)
        clear = np.empty(cap)
        n = int(tree_revolved_overlap_nb(tree, R, t, zs, rs, float(margin), ztol, faces, clear, stats))
        if n <= cap:
            return faces[:n].copy(), clear[:n].copy(), float(stats[0]), int(stats[1])
        cap = n


KERNELS = (_seg_seg_2d, _point_in_tri_2d, _coplanar_tri_tri, _isect, _interval, tri_tri_overlap_nb,
           _degenerate_tri_tri, tri_box_overlap_nb, obb_overlap_nb, _transform_tri, tree_tree_overlap_nb,
           tree_box_overlap_nb, chord_axis_distance_nb, _seg_origin_dist, _zone_value, _segment_clearance,
           triangle_clearance_nb, tree_revolved_overlap_nb)

__all__ = ["tri_tri_overlap", "tri_box_overlap", "box_box_overlap", "mesh_mesh_overlap", "mesh_box_overlap",
           "mesh_revolved_overlap", "tri_tri_overlap_nb", "tri_box_overlap_nb", "obb_overlap_nb",
           "tree_tree_overlap_nb", "tree_box_overlap_nb", "tree_revolved_overlap_nb", "triangle_clearance_nb",
           "chord_axis_distance_nb", "KERNELS"]
