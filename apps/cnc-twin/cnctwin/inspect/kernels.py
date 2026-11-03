"""Numba kernels of inspection (docs/models/inspect.md "Kernels"; rules: docs/architecture.md 5.5).

Every kernel has one explicit signature, ``cache=True, nogil=True, error_model="numpy"``, no fastmath and no
parallel loops; arguments are writable C-contiguous arrays, scalars and the NamedTuples of
:mod:`cnctwin.geom.abi` / :mod:`cnctwin.inspect.abi`. The plain-Python twins are in
:mod:`cnctwin.inspect.reference`; ``tests/unit/inspect/test_kernels.py`` compares each pair and checks
``len(kernel.signatures) == 1``.

* :func:`trilinear_nb` - displacement d(X) of points from a node grid (structure's displacement grids), the same
  clamped trilinear rule as removal's meshing, so displaced meshes agree with removal's.
* :func:`deviation_nb` - signed distance of points to a closed mesh (+ outside the material) with the closest
  triangle, point and feature (face / edge / vertex) - the deviation map and the point-to-face assignment.
* :func:`columns_from_hits_nb` - dexel intervals of rays from their sorted crossings (winding count by the
  crossing sense; duplicate crossings at a shared edge merged, zero-length intervals dropped).
* :func:`columns_difference_nb` - per column, the intervals of A outside B (remaining material = field minus
  target, gouges = target minus field), pieces shorter than a threshold dropped.
* :func:`label_components_nb` / :func:`component_stats_nb` - connected regions of intervals (face-adjacent
  columns whose intervals overlap in z), union-find with the smallest index as root (deterministic), and per
  region volume, centroid, box, largest thickness.
* :func:`chord_deviation_nb` - signed distance along toolpath segments (lines and arcs) minus its linear
  interpolation between the end points, sampled and refined by golden-section search: the chord error.
* :func:`probe_nb` - a stylus ball moving along its approach vector until it touches a mesh (sphere tracing on
  the exact unsigned distance: it never passes through the surface), with the contact point.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, types

from ..geom.abi import TREE_TYPE
from ..geom.bvh import closest_point_nb, pseudo_normal_sign_nb
from . import abi as A

F1 = types.float64[::1]
F2 = types.float64[:, ::1]
F4 = types.float64[:, :, :, ::1]
I1 = types.int64[::1]
I4 = types.int32[::1]
B1 = types.int8[::1]
f8 = types.float64
i8 = types.int64
OPTS = dict(cache=True, nogil=True, error_model="numpy")
TREE = TREE_TYPE

GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0       # golden-section ratio (Kiefer 1953)


# ============================================================================ displacement grids

@njit(types.void(F2, F1, f8, F4, F2), **OPTS)
def trilinear_nb(P, origin, cell, grid, out):
    """out[i] = d(P[i]) trilinear in the node grid ``grid[ix, iy, iz] = d(origin + (ix, iy, iz) cell)``, clamped at
    the border (constant extrapolation); an empty grid gives zero."""
    nx = grid.shape[0]
    ny = grid.shape[1]
    nz = grid.shape[2]
    for p in range(P.shape[0]):
        for c in range(3):
            out[p, c] = 0.0
        if nx == 0 or ny == 0 or nz == 0 or cell <= 0.0:
            continue
        fx = (P[p, 0] - origin[0]) / cell
        fy = (P[p, 1] - origin[1]) / cell
        fz = (P[p, 2] - origin[2]) / cell
        i = int(math.floor(fx))
        j = int(math.floor(fy))
        k = int(math.floor(fz))
        i = min(max(i, 0), max(nx - 2, 0))
        j = min(max(j, 0), max(ny - 2, 0))
        k = min(max(k, 0), max(nz - 2, 0))
        ax = min(max(fx - i, 0.0), 1.0) if nx > 1 else 0.0
        ay = min(max(fy - j, 0.0), 1.0) if ny > 1 else 0.0
        az = min(max(fz - k, 0.0), 1.0) if nz > 1 else 0.0
        for c in range(3):
            acc = 0.0
            for di in range(2):
                ii = min(i + di, nx - 1)
                wx = ax if di == 1 else 1.0 - ax
                for dj in range(2):
                    jj = min(j + dj, ny - 1)
                    wy = ay if dj == 1 else 1.0 - ay
                    for dk in range(2):
                        kk = min(k + dk, nz - 1)
                        wz = az if dk == 1 else 1.0 - az
                        acc += wx * wy * wz * grid[ii, jj, kk, c]
            out[p, c] = acc


# ============================================================================ distances to a mesh

@njit(types.void(TREE, F2, f8, F1, I4, F2, B1), **OPTS)
def deviation_nb(tree, P, max_dist, out_d, out_face, out_point, out_feature):
    """Signed distance (negative inside a closed mesh), closest triangle, point and feature code (geom.abi
    FEATURE_*) per point; points beyond max_dist give NaN, triangle -1, feature -1."""
    res = np.empty(5)
    c = np.empty(3)
    for i in range(P.shape[0]):
        f = closest_point_nb(tree, P[i], max_dist, res)
        out_face[i] = f
        if f < 0:
            out_d[i] = np.nan
            out_feature[i] = -1
            for k in range(3):
                out_point[i, k] = np.nan
            continue
        for k in range(3):
            c[k] = res[k]
            out_point[i, k] = res[k]
        feature = int(res[4])
        out_feature[i] = feature
        out_d[i] = pseudo_normal_sign_nb(tree, f, feature, P[i], c) * res[3]


@njit(f8(TREE, F1, F1, F1), **OPTS)
def _signed_nb(tree, p, res, c):
    """Signed distance of p (NaN for an empty tree); res[5] receives the triangle."""
    f = closest_point_nb(tree, p, np.inf, res)
    res[5] = f
    if f < 0:
        return np.nan
    for k in range(3):
        c[k] = res[k]
    return pseudo_normal_sign_nb(tree, f, int(res[4]), p, c) * res[3]


# ============================================================================ dexel columns

@njit(i8(I1, F1, I4, F1, f8, f8, I1, F2), **OPTS)
def columns_from_hits_nb(offsets, t, face, face_nz, z0, merge, out_start, out_seg):
    """Material intervals of +z rays from their crossings (CSR ``offsets``, ``t`` sorted per ray, ``face`` the
    triangle of each crossing, ``face_nz`` the z component of each triangle's outward normal). A crossing with
    nz < 0 enters the material, nz > 0 leaves it, nz = 0 grazes (ignored); consecutive crossings of the same
    sense within ``merge`` count once. Material is where the running count is positive. z = z0 + t. Returns
    the number of intervals (written while they fit in out_seg)."""
    n = offsets.shape[0] - 1
    cap = out_seg.shape[0]
    m = 0
    for r in range(n):
        out_start[r] = m
        w = 0
        last_s = 0
        last_t = -np.inf
        lo = 0.0
        for h in range(offsets[r], offsets[r + 1]):
            nz = face_nz[face[h]]
            if nz == 0.0:
                continue
            s = 1 if nz < 0.0 else -1
            th = t[h]
            if s == last_s and th - last_t <= merge:
                continue
            last_s = s
            last_t = th
            before = w
            w += s
            z = z0 + th
            if before <= 0 and w > 0:
                if m > out_start[r] and m - 1 < cap and out_seg[m - 1, 1] >= z:
                    m -= 1                      # touches the previous interval: reopen it
                    lo = out_seg[m, 0]
                else:
                    lo = z
            elif before > 0 and w <= 0:
                if z > lo:
                    if m < cap:
                        out_seg[m, 0] = lo
                        out_seg[m, 1] = z
                    m += 1
    out_start[n] = m
    return m


@njit(i8(I1, F2, I1, F2, f8, I1, F2), **OPTS)
def columns_difference_nb(a_start, a_seg, b_start, b_seg, min_len, out_start, out_seg):
    """Per column the intervals of A not covered by B; pieces shorter than min_len (or empty) are dropped.
    Returns the number of intervals (written while they fit)."""
    n = a_start.shape[0] - 1
    cap = out_seg.shape[0]
    m = 0
    for c in range(n):
        out_start[c] = m
        j = b_start[c]
        jb = b_start[c + 1]
        for i in range(a_start[c], a_start[c + 1]):
            lo = a_seg[i, 0]
            hi = a_seg[i, 1]
            while j < jb and b_seg[j, 1] <= lo:
                j += 1
            cur = lo
            k = j
            while k < jb and b_seg[k, 0] < hi:
                bl = b_seg[k, 0]
                bh = b_seg[k, 1]
                if bl > cur:
                    top = bl if bl < hi else hi
                    if top - cur >= min_len and top > cur:
                        if m < cap:
                            out_seg[m, 0] = cur
                            out_seg[m, 1] = top
                        m += 1
                if bh > cur:
                    cur = bh
                if cur >= hi:
                    break
                k += 1
            if cur < hi and hi - cur >= min_len:
                if m < cap:
                    out_seg[m, 0] = cur
                    out_seg[m, 1] = hi
                m += 1
    out_start[n] = m
    return m


@njit(i8(I1, i8), **OPTS)
def _find_nb(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@njit(i8(I1, I1, F2, I1, I1), **OPTS)
def label_components_nb(dims, start, seg, parent, label):
    """Connected regions of the intervals: intervals of face-adjacent columns ((i, j)-(i+1, j), (i, j)-(i, j+1))
    whose z ranges overlap by a positive length are connected. ``label`` gets a region index per interval,
    numbered by first interval; returns the number of regions. ``parent`` is work space (len = intervals)."""
    m = seg.shape[0]
    for s in range(m):
        parent[s] = s
    nx = dims[0]
    ny = dims[1]
    for i in range(nx):
        for j in range(ny):
            c = i * ny + j
            for side in range(2):
                if side == 0:
                    if i + 1 >= nx:
                        continue
                    c2 = (i + 1) * ny + j
                else:
                    if j + 1 >= ny:
                        continue
                    c2 = c + 1
                p = start[c]
                pe = start[c + 1]
                q = start[c2]
                qe = start[c2 + 1]
                while p < pe and q < qe:
                    lo = seg[p, 0] if seg[p, 0] > seg[q, 0] else seg[q, 0]
                    hi = seg[p, 1] if seg[p, 1] < seg[q, 1] else seg[q, 1]
                    if hi > lo:
                        rp = _find_nb(parent, p)
                        rq = _find_nb(parent, q)
                        if rp < rq:
                            parent[rq] = rp
                        elif rq < rp:
                            parent[rp] = rq
                    if seg[p, 1] < seg[q, 1]:
                        p += 1
                    else:
                        q += 1
    count = 0
    for s in range(m):
        r = _find_nb(parent, s)
        if r == s:
            label[s] = count
            count += 1
        else:
            label[s] = label[r]
    return count


@njit(types.void(F1, F1, I1, I1, F2, I1, I1, F2), **OPTS)
def component_stats_nb(origin, cell, dims, start, seg, label, last, out):
    """Per region (rows of out, abi.CS_*): volume, centroid, box, largest interval, columns, intervals, first
    interval. ``last`` is work space (len = regions)."""
    nr = out.shape[0]
    for r in range(nr):
        for k in range(A.N_CS):
            out[r, k] = 0.0
        for k in (A.CS_XMIN, A.CS_YMIN, A.CS_ZMIN):
            out[r, k] = np.inf
        for k in (A.CS_XMAX, A.CS_YMAX, A.CS_ZMAX):
            out[r, k] = -np.inf
        out[r, A.CS_THICK] = -1.0
        out[r, A.CS_FIRST] = -1.0
        last[r] = -1
    ny = dims[1]
    area = cell[0] * cell[1]
    n = start.shape[0] - 1
    for c in range(n):
        i = c // ny
        j = c - i * ny
        x = origin[0] + i * cell[0]
        y = origin[1] + j * cell[1]
        for s in range(start[c], start[c + 1]):
            r = label[s]
            lo = seg[s, 0]
            hi = seg[s, 1]
            length = hi - lo
            vol = length * area
            out[r, A.CS_VOLUME] += vol
            out[r, A.CS_CX] += vol * x
            out[r, A.CS_CY] += vol * y
            out[r, A.CS_CZ] += vol * 0.5 * (lo + hi)
            if x - 0.5 * cell[0] < out[r, A.CS_XMIN]:
                out[r, A.CS_XMIN] = x - 0.5 * cell[0]
            if x + 0.5 * cell[0] > out[r, A.CS_XMAX]:
                out[r, A.CS_XMAX] = x + 0.5 * cell[0]
            if y - 0.5 * cell[1] < out[r, A.CS_YMIN]:
                out[r, A.CS_YMIN] = y - 0.5 * cell[1]
            if y + 0.5 * cell[1] > out[r, A.CS_YMAX]:
                out[r, A.CS_YMAX] = y + 0.5 * cell[1]
            if lo < out[r, A.CS_ZMIN]:
                out[r, A.CS_ZMIN] = lo
            if hi > out[r, A.CS_ZMAX]:
                out[r, A.CS_ZMAX] = hi
            if length > out[r, A.CS_THICK]:
                out[r, A.CS_THICK] = length
                out[r, A.CS_TX] = x
                out[r, A.CS_TY] = y
                out[r, A.CS_TZ] = 0.5 * (lo + hi)
            out[r, A.CS_SEGMENTS] += 1.0
            if last[r] != c:
                last[r] = c
                out[r, A.CS_COLUMNS] += 1.0
            if out[r, A.CS_FIRST] < 0.0:
                out[r, A.CS_FIRST] = s
    for r in range(nr):
        v = out[r, A.CS_VOLUME]
        if v > 0.0:
            out[r, A.CS_CX] /= v
            out[r, A.CS_CY] /= v
            out[r, A.CS_CZ] /= v


# ============================================================================ toolpath chord error

@njit(types.void(F2, i8, f8, F1), **OPTS)
def segment_point_nb(seg, row, s, out):
    """Point at parameter s in [0, 1] of segment ``row`` (abi.SG_*): a line, or an arc about the axis SG_N through
    SG_C swept by SG_SWEEP with the radius and the axial position interpolated linearly (helix, spiral)."""
    if seg[row, A.SG_KIND] == A.SEG_LINE:
        for k in range(3):
            out[k] = seg[row, A.SG_P0 + k] + s * (seg[row, A.SG_P1 + k] - seg[row, A.SG_P0 + k])
        return
    nx = seg[row, A.SG_N]
    ny = seg[row, A.SG_N + 1]
    nz = seg[row, A.SG_N + 2]
    a0 = 0.0
    a1 = 0.0
    for k in range(3):
        a0 += (seg[row, A.SG_P0 + k] - seg[row, A.SG_C + k]) * seg[row, A.SG_N + k]
        a1 += (seg[row, A.SG_P1 + k] - seg[row, A.SG_C + k]) * seg[row, A.SG_N + k]
    v0x = seg[row, A.SG_P0] - seg[row, A.SG_C] - a0 * nx
    v0y = seg[row, A.SG_P0 + 1] - seg[row, A.SG_C + 1] - a0 * ny
    v0z = seg[row, A.SG_P0 + 2] - seg[row, A.SG_C + 2] - a0 * nz
    v1x = seg[row, A.SG_P1] - seg[row, A.SG_C] - a1 * nx
    v1y = seg[row, A.SG_P1 + 1] - seg[row, A.SG_C + 1] - a1 * ny
    v1z = seg[row, A.SG_P1 + 2] - seg[row, A.SG_C + 2] - a1 * nz
    r0 = math.sqrt(v0x * v0x + v0y * v0y + v0z * v0z)
    r1 = math.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
    if r0 <= 0.0:
        for k in range(3):
            out[k] = seg[row, A.SG_P0 + k] + s * (seg[row, A.SG_P1 + k] - seg[row, A.SG_P0 + k])
        return
    ex = v0x / r0
    ey = v0y / r0
    ez = v0z / r0
    fx = ny * ez - nz * ey
    fy = nz * ex - nx * ez
    fz = nx * ey - ny * ex
    ang = s * seg[row, A.SG_SWEEP]
    ca = math.cos(ang)
    sa = math.sin(ang)
    r = r0 + s * (r1 - r0)
    ax = a0 + s * (a1 - a0)
    out[0] = seg[row, A.SG_C] + r * (ca * ex + sa * fx) + ax * nx
    out[1] = seg[row, A.SG_C + 1] + r * (ca * ey + sa * fy) + ax * ny
    out[2] = seg[row, A.SG_C + 2] + r * (ca * ez + sa * fz) + ax * nz


@njit(f8(TREE, F2, i8, f8, f8, f8, F1, F1, F1), **OPTS)
def _chord_error_nb(tree, seg, row, s, d0, d1, p, res, c):
    segment_point_nb(seg, row, s, p)
    return _signed_nb(tree, p, res, c) - ((1.0 - s) * d0 + s * d1)


@njit(types.void(TREE, F2, i8, i8, f8, F2), **OPTS)
def chord_deviation_nb(tree, seg, n_samples, n_iter, max_dist, out):
    """Chord error of every segment (rows abi.CH_*): e(s) = d(P(s)) - ((1 - s) d0 + s d1) with d the signed distance
    to the mesh; the extremes over ``n_samples`` interior samples refined by ``n_iter`` golden-section steps on
    the bracket of the best sample. Segments with both ends farther than max_dist get NaN."""
    p = np.empty(3)
    res = np.empty(6)
    c = np.empty(3)
    q = np.empty(3)
    for row in range(seg.shape[0]):
        for k in range(A.N_CH):
            out[row, k] = np.nan
        segment_point_nb(seg, row, 0.0, p)
        d0 = _signed_nb(tree, p, res, c)
        segment_point_nb(seg, row, 1.0, q)
        d1 = _signed_nb(tree, q, res, c)
        # length: polyline of the samples (exact for lines)
        length = 0.0
        prev0 = p[0]
        prev1 = p[1]
        prev2 = p[2]
        n_len = 1 if seg[row, A.SG_KIND] == A.SEG_LINE else max(n_samples + 1, 16)
        for k in range(1, n_len + 1):
            segment_point_nb(seg, row, k / n_len, q)
            length += math.sqrt((q[0] - prev0) ** 2 + (q[1] - prev1) ** 2 + (q[2] - prev2) ** 2)
            prev0 = q[0]
            prev1 = q[1]
            prev2 = q[2]
        out[row, A.CH_LEN] = length
        out[row, A.CH_D0] = d0
        out[row, A.CH_D1] = d1
        if not (abs(d0) <= max_dist or abs(d1) <= max_dist):
            continue
        best_lo = 0.0
        s_lo = 0.0
        best_hi = 0.0
        s_hi = 0.0
        k_lo = 0
        k_hi = 0
        for k in range(1, n_samples + 1):
            s = k / (n_samples + 1.0)
            e = _chord_error_nb(tree, seg, row, s, d0, d1, p, res, c)
            if e < best_lo:
                best_lo = e
                s_lo = s
                k_lo = k
            if e > best_hi:
                best_hi = e
                s_hi = s
                k_hi = k
        for side in range(2):
            k_best = k_lo if side == 0 else k_hi
            if k_best == 0:
                continue
            sign = 1.0 if side == 0 else -1.0       # minimise sign * e
            a = (k_best - 1.0) / (n_samples + 1.0)
            b = (k_best + 1.0) / (n_samples + 1.0)
            x1 = b - GOLDEN * (b - a)
            x2 = a + GOLDEN * (b - a)
            f1 = sign * _chord_error_nb(tree, seg, row, x1, d0, d1, p, res, c)
            f2 = sign * _chord_error_nb(tree, seg, row, x2, d0, d1, p, res, c)
            for _ in range(n_iter):
                if f1 < f2:
                    b = x2
                    x2 = x1
                    f2 = f1
                    x1 = b - GOLDEN * (b - a)
                    f1 = sign * _chord_error_nb(tree, seg, row, x1, d0, d1, p, res, c)
                else:
                    a = x1
                    x1 = x2
                    f1 = f2
                    x2 = a + GOLDEN * (b - a)
                    f2 = sign * _chord_error_nb(tree, seg, row, x2, d0, d1, p, res, c)
            xm = x1 if f1 < f2 else x2
            fm = f1 if f1 < f2 else f2
            if side == 0 and sign * fm < best_lo:
                best_lo = sign * fm
                s_lo = xm
            if side == 1 and sign * fm > best_hi:
                best_hi = sign * fm
                s_hi = xm
        out[row, A.CH_EMIN] = best_lo
        out[row, A.CH_SMIN] = s_lo
        out[row, A.CH_EMAX] = best_hi
        out[row, A.CH_SMAX] = s_hi
        _chord_error_nb(tree, seg, row, s_lo, d0, d1, p, res, c)
        out[row, A.CH_FMIN] = res[5]
        _chord_error_nb(tree, seg, row, s_hi, d0, d1, p, res, c)
        out[row, A.CH_FMAX] = res[5]


# ============================================================================ probing

@njit(types.void(TREE, F2, F2, f8, f8, f8, i8, F2), **OPTS)
def probe_nb(tree, O, D, radius, max_travel, tol, max_iter, out):
    """A ball of ``radius`` starting at O[i] moves along D[i] (any length) until its free distance to the mesh is
    at most ``tol``; rows of out are abi.PR_*. Each step advances by the free distance (sphere tracing on the
    exact unsigned distance), so the ball never crosses the surface."""
    res = np.empty(5)
    cpt = np.empty(3)
    for i in range(O.shape[0]):
        for k in range(A.N_PR):
            out[i, k] = np.nan
        dn = math.sqrt(D[i, 0] ** 2 + D[i, 1] ** 2 + D[i, 2] ** 2)
        if dn <= 0.0:
            out[i, A.PR_STATUS] = A.PROBE_NO_CONTACT
            continue
        dx = D[i, 0] / dn
        dy = D[i, 1] / dn
        dz = D[i, 2] / dn
        t = 0.0
        status = A.PROBE_NOT_CONVERGED
        steps = 0
        face = -1
        for it in range(max_iter):
            steps = it + 1
            cpt[0] = O[i, 0] + t * dx
            cpt[1] = O[i, 1] + t * dy
            cpt[2] = O[i, 2] + t * dz
            face = closest_point_nb(tree, cpt, radius + (max_travel - t) + tol, res)
            if face < 0:
                status = A.PROBE_NO_CONTACT
                break
            free = res[3] - radius
            if free <= tol:
                status = A.PROBE_START_IN_CONTACT if (it == 0 and free < -tol) else A.PROBE_CONTACT
                break
            t += free
            if t > max_travel:
                status = A.PROBE_NO_CONTACT
                break
        out[i, A.PR_STATUS] = status
        out[i, A.PR_ITER] = steps
        out[i, A.PR_T] = t
        out[i, A.PR_CX] = O[i, 0] + t * dx
        out[i, A.PR_CY] = O[i, 1] + t * dy
        out[i, A.PR_CZ] = O[i, 2] + t * dz
        if status == A.PROBE_CONTACT or status == A.PROBE_START_IN_CONTACT:
            out[i, A.PR_QX] = res[0]
            out[i, A.PR_QY] = res[1]
            out[i, A.PR_QZ] = res[2]
            out[i, A.PR_FACE] = face


KERNELS = (trilinear_nb, deviation_nb, _signed_nb, columns_from_hits_nb, columns_difference_nb, _find_nb,
           label_components_nb, component_stats_nb, segment_point_nb, _chord_error_nb, chord_deviation_nb, probe_nb)

__all__ = ["trilinear_nb", "deviation_nb", "columns_from_hits_nb", "columns_difference_nb", "label_components_nb",
           "component_stats_nb", "segment_point_nb", "chord_deviation_nb", "probe_nb", "KERNELS", "GOLDEN"]
