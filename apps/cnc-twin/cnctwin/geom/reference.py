"""Plain-Python references of the geometry kernels (docs/architecture.md 5.5: every kernel has one and a test
comparing the two). Where it is cheap the reference uses a *different* algorithm, so a shared mistake is
unlikely: Moller-Trumbore rays against the watertight test, clipping against the separating-axis tests,
edge-segment tests against Moller's interval method, brute force against the trees, winding numbers
against parity, dense sampling against the golden-section clearance, per-cell slab tests against the DDA.
None of this is used by the simulation.
"""

from __future__ import annotations

import math

import numpy as np

from .abi import MAX_DEPTH
from .mesh import Mesh

_EPS = float(np.finfo(np.float64).eps)


# ============================================================================ triangles

def ray_triangle_ref(o, d, v0, v1, v2):
    """Moller & Trumbore (1997): ``(t, u, v)`` (barycentrics of v1, v2) or None; double-sided."""
    o, d, v0, v1, v2 = (np.asarray(x, dtype=np.float64) for x in (o, d, v0, v1, v2))
    e1 = v1 - v0
    e2 = v2 - v0
    p = np.cross(d, e2)
    det = float(np.dot(e1, p))
    if det == 0.0:
        return None
    inv = 1.0 / det
    s = o - v0
    u = float(np.dot(s, p)) * inv
    if u < 0.0 or u > 1.0:
        return None
    q = np.cross(s, e1)
    v = float(np.dot(d, q)) * inv
    if v < 0.0 or u + v > 1.0:
        return None
    return float(np.dot(e2, q)) * inv, u, v


def closest_point_segment_ref(p, a, b) -> np.ndarray:
    p, a, b = (np.asarray(x, dtype=np.float64) for x in (p, a, b))
    ab = b - a
    denom = float(np.dot(ab, ab))
    s = 0.0 if denom == 0.0 else min(1.0, max(0.0, float(np.dot(p - a, ab)) / denom))
    return a + s * ab


def closest_point_triangle_ref(p, a, b, c) -> np.ndarray:
    """Projection on the plane if it falls inside the triangle, else the nearest of the three edges."""
    p, a, b, c = (np.asarray(x, dtype=np.float64) for x in (p, a, b, c))
    n = np.cross(b - a, c - a)
    nn = float(np.dot(n, n))
    if nn > 0.0:
        q = p - (float(np.dot(p - a, n)) / nn) * n
        # barycentric by sub-areas
        wa = float(np.dot(np.cross(c - b, q - b), n))
        wb = float(np.dot(np.cross(a - c, q - c), n))
        wc = float(np.dot(np.cross(b - a, q - a), n))
        if wa >= 0.0 and wb >= 0.0 and wc >= 0.0:
            return q
    cands = [closest_point_segment_ref(p, a, b), closest_point_segment_ref(p, b, c), closest_point_segment_ref(p, c, a)]
    return min(cands, key=lambda x: float(np.sum((x - p) ** 2)))


def _segment_triangle(p, q, tri) -> bool:
    """Closed segment pq meets the closed triangle (non-coplanar case by plane crossing + inside test)."""
    a, b, c = tri
    n = np.cross(b - a, c - a)
    dp = float(np.dot(p - a, n))
    dq = float(np.dot(q - a, n))
    if dp * dq > 0.0:
        return False
    if dp == dq:
        return False
    s = dp / (dp - dq)
    x = p + s * (q - p)
    return _inside_triangle(x, tri, n)


def _inside_triangle(x, tri, n) -> bool:
    a, b, c = tri
    for u, v in ((a, b), (b, c), (c, a)):
        if float(np.dot(np.cross(v - u, x - u), n)) < -1e-12 * float(np.dot(n, n)):
            return False
    return True


def _seg2(a, b, c, d) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on(p, q, r):
        return min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1])

    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    return (o1 == 0 and on(a, b, c)) or (o2 == 0 and on(a, b, d)) or (o3 == 0 and on(c, d, a)) or (o4 == 0 and on(c, d, b))


def tri_tri_overlap_ref(t1, t2) -> bool:
    """Two closed triangles intersect: coplanar pairs by 2-D edge and containment tests, otherwise iff an edge of
    one meets the other triangle."""
    T1 = np.asarray(t1, dtype=np.float64)
    T2 = np.asarray(t2, dtype=np.float64)
    n1 = np.cross(T1[1] - T1[0], T1[2] - T1[0])
    d2 = (T2 - T1[0]) @ n1
    scale = np.linalg.norm(n1) * max(1e-300, float(np.max(np.abs(np.concatenate([T1, T2]) - T1[0]))))
    if np.all(np.abs(d2) <= 1e-12 * scale):
        k = int(np.argmax(np.abs(n1)))
        i0, i1 = [i for i in range(3) if i != k]
        A = T1[:, [i0, i1]]
        B = T2[:, [i0, i1]]
        for e in range(3):
            for f in range(3):
                if _seg2(A[e], A[(e + 1) % 3], B[f], B[(f + 1) % 3]):
                    return True

        def inside2(p, P):
            s = [(P[(i + 1) % 3][0] - P[i][0]) * (p[1] - P[i][1]) - (P[(i + 1) % 3][1] - P[i][1]) * (p[0] - P[i][0])
                 for i in range(3)]
            return all(v >= 0 for v in s) or all(v <= 0 for v in s)

        return inside2(A[0], B) or inside2(B[0], A)
    for e in range(3):
        if _segment_triangle(T1[e], T1[(e + 1) % 3], T2) or _segment_triangle(T2[e], T2[(e + 1) % 3], T1):
            return True
    return False


def tri_box_overlap_ref(tri, centre, half) -> bool:
    """Sutherland-Hodgman clipping of the triangle by the six (closed) box slabs: overlap iff anything is left."""
    poly = [np.asarray(v, dtype=np.float64) for v in np.asarray(tri, dtype=np.float64)]
    c = np.asarray(centre, dtype=np.float64)
    h = np.asarray(half, dtype=np.float64)
    for k in range(3):
        for sign in (1.0, -1.0):
            limit = c[k] + sign * h[k]
            out = []
            for i in range(len(poly)):
                p = poly[i]
                q = poly[(i + 1) % len(poly)]
                fp = sign * (limit - p[k])
                fq = sign * (limit - q[k])
                if fp >= 0.0:
                    out.append(p)
                if (fp >= 0.0) != (fq >= 0.0) and fp != fq:
                    s = fp / (fp - fq)
                    out.append(p + s * (q - p))
            poly = out
            if not poly:
                return False
    return True


def box_box_overlap_ref(ca, ha, T_AB, cb, hb) -> bool:
    """Box B's surface triangles against box A, plus the containment of either centre."""
    from .primitives import box_mesh

    ca = np.asarray(ca, dtype=np.float64)
    ha = np.asarray(ha, dtype=np.float64)
    cb = np.asarray(cb, dtype=np.float64)
    hb = np.asarray(hb, dtype=np.float64)
    T = np.asarray(T_AB, dtype=np.float64)
    mb = box_mesh(cb - hb, cb + hb).transformed(T)
    for tri in mb.triangles():
        if tri_box_overlap_ref(tri, ca, ha):
            return True
    cb_A = T[:3, :3] @ cb + T[:3, 3]
    if np.all(np.abs(cb_A - ca) <= ha):
        return True
    ca_B = T[:3, :3].T @ (ca - T[:3, 3])
    return bool(np.all(np.abs(ca_B - cb) <= hb))


# ============================================================================ tree build (same algorithm)

def build_tree_ref(mesh: Mesh, leaf_size: int, n_bins: int, trav_cost: float):
    """Plain-Python binned-SAH build, the same algorithm and arithmetic as bvh.build_nodes_nb: returns
    (order, node_lo, node_hi, node_right, node_start, node_count, depth)."""
    tris = mesh.triangles()
    tri_lo = tris.min(axis=1)
    tri_hi = tris.max(axis=1)
    cen = 0.5 * (tri_lo + tri_hi)
    order = list(range(len(tris)))
    nodes = []

    def area(lo, hi):
        d = hi - lo
        return 2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])

    stack = []
    s, e, depth, parent = 0, len(tris), 0, -1
    max_depth = 0
    while True:
        node = len(nodes)
        rec = {"right": -1, "start": s, "count": 0}
        nodes.append(rec)
        if parent >= 0:
            nodes[parent]["right"] = node
        idx = order[s:e]
        rec["lo"] = np.min(tri_lo[idx], axis=0)
        rec["hi"] = np.max(tri_hi[idx], axis=0)
        cmin = np.min(cen[idx], axis=0)
        cmax = np.max(cen[idx], axis=0)
        count = e - s
        leaf = count <= leaf_size or depth >= MAX_DEPTH
        mid = s
        if not leaf:
            ext3 = cmax - cmin
            axis = 0
            for k in (1, 2):
                if ext3[k] > ext3[axis]:
                    axis = k
            ext = ext3[axis]
            if ext <= 0.0:
                mid = s + count // 2
            else:
                scale = n_bins / ext

                def bin_of(t):
                    b = int((cen[t, axis] - cmin[axis]) * scale)
                    return min(b, n_bins - 1)

                cnt = [0] * n_bins
                blo = [np.full(3, np.inf) for _ in range(n_bins)]
                bhi = [np.full(3, -np.inf) for _ in range(n_bins)]
                for t in idx:
                    b = bin_of(t)
                    cnt[b] += 1
                    blo[b] = np.minimum(blo[b], tri_lo[t])
                    bhi[b] = np.maximum(bhi[b], tri_hi[t])
                right_area = [0.0] * n_bins
                right_cnt = [0] * n_bins
                rlo = np.full(3, np.inf)
                rhi = np.full(3, -np.inf)
                rc = 0
                for b in range(n_bins - 1, 0, -1):
                    rc += cnt[b]
                    rlo = np.minimum(rlo, blo[b])
                    rhi = np.maximum(rhi, bhi[b])
                    right_cnt[b] = rc
                    right_area[b] = area(rlo, rhi) if rc > 0 else 0.0
                llo = np.full(3, np.inf)
                lhi = np.full(3, -np.inf)
                lc = 0
                best_cost = np.inf
                best = -1
                pa = area(rec["lo"], rec["hi"])
                for b in range(n_bins - 1):
                    lc += cnt[b]
                    llo = np.minimum(llo, blo[b])
                    lhi = np.maximum(lhi, bhi[b])
                    if lc == 0 or right_cnt[b + 1] == 0:
                        continue
                    la = area(llo, lhi)
                    cost = trav_cost + (lc * la + right_cnt[b + 1] * right_area[b + 1]) / pa if pa > 0.0 \
                        else trav_cost + 0.5 * count
                    if cost < best_cost:
                        best_cost = cost
                        best = b
                if best < 0:
                    mid = s + count // 2
                else:
                    i, j = s, e - 1
                    while i <= j:
                        t = order[i]
                        if bin_of(t) <= best:
                            i += 1
                        else:
                            order[i], order[j] = order[j], t
                            j -= 1
                    mid = i
                    if mid == s or mid == e:
                        mid = s + count // 2
        if leaf:
            rec["count"] = count
            max_depth = max(max_depth, depth)
            if not stack:
                break
            s, e, depth, parent = stack.pop()
            continue
        stack.append((mid, e, depth + 1, node))
        e = mid
        depth += 1
        parent = -1
    lo = np.array([n["lo"] for n in nodes])
    hi = np.array([n["hi"] for n in nodes])
    return (np.array(order, np.int32), lo, hi, np.array([n["right"] for n in nodes], np.int32),
            np.array([n["start"] for n in nodes], np.int32), np.array([n["count"] for n in nodes], np.int32), max_depth)


# ============================================================================ brute-force mesh queries

def ray_hits_ref(mesh: Mesh, o, d, t_min=0.0, t_max=math.inf):
    """Every (t, face) by Moller-Trumbore over all faces, sorted by (t, face)."""
    hits = []
    for f, tri in enumerate(mesh.triangles()):
        h = ray_triangle_ref(o, d, tri[0], tri[1], tri[2])
        if h is not None and t_min <= h[0] <= t_max:
            hits.append((h[0], f))
    return sorted(hits)


def closest_point_ref(mesh: Mesh, p):
    """(point, distance, face) by brute force (lowest face id on ties)."""
    best = (math.inf, -1, None)
    for f, tri in enumerate(mesh.triangles()):
        q = closest_point_triangle_ref(p, tri[0], tri[1], tri[2])
        d = float(np.linalg.norm(q - np.asarray(p, dtype=np.float64)))
        if d < best[0]:
            best = (d, f, q)
    return best[2], best[0], best[1]


def winding_number_ref(mesh: Mesh, p) -> float:
    """Generalised winding number (Van Oosterom & Strackee solid angles, NumPy)."""
    t = mesh.triangles() - np.asarray(p, dtype=np.float64)
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    la, lb, lc = (np.linalg.norm(x, axis=1) for x in (a, b, c))
    det = np.einsum("ij,ij->i", a, np.cross(b, c))
    den = la * lb * lc + np.einsum("ij,ij->i", a, b) * lc + np.einsum("ij,ij->i", a, c) * lb \
        + np.einsum("ij,ij->i", b, c) * la
    return float(np.sum(2.0 * np.arctan2(det, den)) / (4.0 * math.pi))


def signed_distance_ref(mesh: Mesh, p) -> float:
    """Unsigned brute-force distance with the sign of the winding number (inside negative)."""
    _, d, _ = closest_point_ref(mesh, p)
    return -d if winding_number_ref(mesh, p) > 0.5 else d


def mesh_mesh_overlap_ref(a: Mesh, b: Mesh, T_AB) -> set:
    tb = b.transformed(T_AB).triangles()
    ta = a.triangles()
    return {(i, j) for i in range(len(ta)) for j in range(len(tb)) if tri_tri_overlap_ref(ta[i], tb[j])}


def mesh_box_overlap_ref(mesh: Mesh, T_A_box, half, margin=0.0) -> set:
    T = np.asarray(T_A_box, dtype=np.float64)
    local = mesh.transformed(np.linalg.inv(T)).triangles()
    h = np.asarray(half, dtype=np.float64) + margin
    return {f for f, tri in enumerate(local) if tri_box_overlap_ref(tri, np.zeros(3), h)}


def triangle_clearance_ref(tri, z, r, samples: int = 200) -> float:
    """Radial clearance min (rho - R(z)) over a dense barycentric sampling of the triangle (solid frame),
    restricted to the profile's z range; +inf when the triangle misses that range."""
    T = np.asarray(tri, dtype=np.float64)
    best = math.inf
    zs = np.asarray(z, dtype=np.float64)
    rs = np.asarray(r, dtype=np.float64)
    for i in range(samples + 1):
        for j in range(samples + 1 - i):
            a = i / samples
            b = j / samples
            p = T[0] + a * (T[1] - T[0]) + b * (T[2] - T[0])
            if p[2] < zs[0] or p[2] > zs[-1]:
                continue
            rho = math.hypot(p[0], p[1])
            radius = _profile_radius(zs, rs, p[2])
            best = min(best, rho - radius)
    return best


def _profile_radius(zs, rs, zq) -> float:
    """Largest radius of the z-monotone polyline at height zq (steps take their larger radius)."""
    best = 0.0
    for k in range(len(zs) - 1):
        if zs[k] <= zq <= zs[k + 1]:
            if zs[k + 1] == zs[k]:
                best = max(best, rs[k], rs[k + 1])
            else:
                best = max(best, rs[k] + (rs[k + 1] - rs[k]) * (zq - zs[k]) / (zs[k + 1] - zs[k]))
    return best


# ============================================================================ grid traversal

def dda_cells_ref(o, d, t0, t1, lo, cell, dims):
    """Every cell whose closed box the ray meets within [t0, t1] with a positive length: ``[(t_in, t_out, (i, j,
    k)), ...]`` sorted by t_in (slab test per cell, brute force over the grid)."""
    o = np.asarray(o, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    cell = np.broadcast_to(np.asarray(cell, dtype=np.float64), (3,))
    out = []
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                a = lo + np.array([i, j, k]) * cell
                b = a + cell
                tn, tf = t0, t1
                ok = True
                for ax in range(3):
                    if d[ax] == 0.0:
                        if o[ax] < a[ax] or o[ax] > b[ax]:
                            ok = False
                            break
                        continue
                    ta = (a[ax] - o[ax]) / d[ax]
                    tb = (b[ax] - o[ax]) / d[ax]
                    tn = max(tn, min(ta, tb))
                    tf = min(tf, max(ta, tb))
                if ok and tf > tn:
                    out.append((tn, tf, (i, j, k)))
    return sorted(out)


__all__ = ["ray_triangle_ref", "closest_point_triangle_ref", "closest_point_segment_ref", "tri_tri_overlap_ref",
           "tri_box_overlap_ref", "box_box_overlap_ref", "build_tree_ref", "ray_hits_ref", "closest_point_ref",
           "winding_number_ref", "signed_distance_ref", "mesh_mesh_overlap_ref", "mesh_box_overlap_ref",
           "triangle_clearance_ref", "dda_cells_ref"]
