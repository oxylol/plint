"""Axis-aligned bounding-box trees over triangle meshes and the queries on them (docs/models/geom.md "BVH").

* **Build** (:func:`build_tree`): binned surface-area heuristic (MacDonald & Booth 1990; Wald 2007; the
  constants of pbrt, Pharr et al. 2016 4.3), deterministic: no parallelism, in-place partitions in index
  order, the median of the index order when all centroids coincide, a forced leaf at ``abi.MAX_DEPTH``.
  The same mesh always gives bit-identical arrays. Static meshes (machine, fixture) are built once:
  :func:`static_tree` memoises by content hash, :func:`save_tree`/:func:`load_tree` persist.
* **Rays** (first hit, all hits, segments, batches): watertight ray/triangle test of Woop, Benthin &
  Wald (2013) (no epsilon; a ray through a shared edge or vertex is never lost between the triangles)
  with conservative box tests (far distance scaled by 1 + 2 gamma(3), Ize 2013).
* **Closest point and signed distance**: closest point on a triangle by Voronoi regions (Ericson 2005,
  5.1.5); the sign from the angle-weighted pseudo-normal of the closest feature (Baerentzen & Aanaes
  2005), valid for closed, consistently oriented meshes (``info[INFO_CLOSED]``).
* **Point in mesh**: parity of crossings along fixed, non-axis-aligned directions; a ray that meets an
  edge or a vertex exactly (an edge function of the watertight test is exactly zero) is discarded and the
  next direction is used; the majority of ``geom.point_in_mesh_rays`` decided rays wins. The generalised
  winding number (Jacobson et al. 2013, solid angles by Van Oosterom & Strackee 1983) is the
  orientation-aware alternative for meshes with holes.

Kernels have explicit signatures (one compiled specialisation each); plain-Python references are in
:mod:`cnctwin.geom.reference`.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
from numba import njit, types

from . import abi
from .abi import (FEATURE_EDGE_0, FEATURE_EDGE_1, FEATURE_EDGE_2, FEATURE_FACE, FEATURE_VERTEX_0, FEATURE_VERTEX_1,
                  FEATURE_VERTEX_2, INFO_CLOSED, INFO_DEPTH, INFO_LEAF_SIZE, INFO_NODES, INSIDE, MAX_DEPTH,
                  MeshTree, N_INFO, ON_SURFACE, OUTSIDE, STACK_SIZE, TREE_TYPE, UNDECIDED, contiguous)
from .mesh import Mesh
from .settings import resolve

F1 = types.float64[::1]
F2 = types.float64[:, ::1]
F3 = types.float64[:, :, ::1]
I1 = types.int32[::1]
I8_1 = types.int64[::1]
B1 = types.int8[::1]
f8 = types.float64
i8 = types.int64
OPTS = dict(cache=True, nogil=True, error_model="numpy")
TREE = TREE_TYPE

_EPS = float(np.finfo(np.float64).eps)
GAMMA3 = 3.0 * (0.5 * _EPS) / (1.0 - 3.0 * (0.5 * _EPS))      # Higham's gamma(3) with unit roundoff eps/2
BOX_FAR_SCALE = 1.0 + 2.0 * GAMMA3                              # Ize 2013: conservative far-plane distance

if (FEATURE_FACE, FEATURE_EDGE_0, FEATURE_EDGE_1, FEATURE_EDGE_2, FEATURE_VERTEX_0, FEATURE_VERTEX_1,
        FEATURE_VERTEX_2) != (0, 1, 2, 3, 4, 5, 6) or (OUTSIDE, INSIDE, ON_SURFACE, UNDECIDED) != (0, 1, 2, -1):
    raise ImportError("cnctwin.geom.bvh: feature/classification codes differ from cnctwin.geom.abi")


# ============================================================================ build

@njit(f8(F1, F1), **OPTS)
def _area(lo, hi):
    dx = hi[0] - lo[0]
    dy = hi[1] - lo[1]
    dz = hi[2] - lo[2]
    return 2.0 * (dx * dy + dy * dz + dz * dx)


@njit(i8(F2, F2, F2, i8, i8, f8, I1, F2, F2, I1, I1, I1, I8_1), **OPTS)
def build_nodes_nb(tri_lo, tri_hi, cen, leaf_size, n_bins, trav_cost, order, node_lo, node_hi, node_right,
                   node_start, node_count, stats):
    """Binned-SAH build into preallocated node arrays (capacity 2 m - 1). ``order`` holds the triangle ids
    and is permuted into leaf order. Returns the number of nodes; stats[0] = depth of the deepest leaf."""
    m = order.shape[0]
    # explicit stack of pending second children: (start, end, depth, parent)
    st_s = np.empty(MAX_DEPTH + 2, np.int64)
    st_e = np.empty(MAX_DEPTH + 2, np.int64)
    st_d = np.empty(MAX_DEPTH + 2, np.int64)
    st_p = np.empty(MAX_DEPTH + 2, np.int64)
    sp = 0
    n_nodes = 0
    max_depth_seen = 0
    s = 0
    e = m
    depth = 0
    parent = -1
    bin_cnt = np.zeros(n_bins, np.int64)
    bin_lo = np.empty((n_bins, 3))
    bin_hi = np.empty((n_bins, 3))
    right_area = np.empty(n_bins)
    right_cnt = np.empty(n_bins, np.int64)
    while True:
        node = n_nodes
        n_nodes += 1
        if parent >= 0:
            node_right[parent] = node
        # bounds of the node and of its centroids
        for k in range(3):
            node_lo[node, k] = np.inf
            node_hi[node, k] = -np.inf
        cmin = np.full(3, np.inf)
        cmax = np.full(3, -np.inf)
        for i in range(s, e):
            t = order[i]
            for k in range(3):
                if tri_lo[t, k] < node_lo[node, k]:
                    node_lo[node, k] = tri_lo[t, k]
                if tri_hi[t, k] > node_hi[node, k]:
                    node_hi[node, k] = tri_hi[t, k]
                if cen[t, k] < cmin[k]:
                    cmin[k] = cen[t, k]
                if cen[t, k] > cmax[k]:
                    cmax[k] = cen[t, k]
        count = e - s
        make_leaf = count <= leaf_size or depth >= MAX_DEPTH
        mid = s
        if not make_leaf:
            axis = 0
            ext = cmax[0] - cmin[0]
            for k in range(1, 3):
                if cmax[k] - cmin[k] > ext:
                    ext = cmax[k] - cmin[k]
                    axis = k
            if ext <= 0.0:
                mid = s + count // 2           # all centroids coincide: split the index order
            else:
                for b in range(n_bins):
                    bin_cnt[b] = 0
                    for k in range(3):
                        bin_lo[b, k] = np.inf
                        bin_hi[b, k] = -np.inf
                scale = n_bins / ext
                for i in range(s, e):
                    t = order[i]
                    b = int((cen[t, axis] - cmin[axis]) * scale)
                    if b >= n_bins:
                        b = n_bins - 1
                    bin_cnt[b] += 1
                    for k in range(3):
                        if tri_lo[t, k] < bin_lo[b, k]:
                            bin_lo[b, k] = tri_lo[t, k]
                        if tri_hi[t, k] > bin_hi[b, k]:
                            bin_hi[b, k] = tri_hi[t, k]
                # sweep from the right: area and count of bins b .. n_bins-1
                rlo = np.full(3, np.inf)
                rhi = np.full(3, -np.inf)
                rc = 0
                for b in range(n_bins - 1, 0, -1):
                    rc += bin_cnt[b]
                    for k in range(3):
                        rlo[k] = min(rlo[k], bin_lo[b, k])
                        rhi[k] = max(rhi[k], bin_hi[b, k])
                    right_cnt[b] = rc
                    right_area[b] = _area(rlo, rhi) if rc > 0 else 0.0
                llo = np.full(3, np.inf)
                lhi = np.full(3, -np.inf)
                lc = 0
                best_cost = np.inf
                best_split = -1
                parent_area = _area(node_lo[node], node_hi[node])
                for b in range(n_bins - 1):
                    lc += bin_cnt[b]
                    for k in range(3):
                        llo[k] = min(llo[k], bin_lo[b, k])
                        lhi[k] = max(lhi[k], bin_hi[b, k])
                    if lc == 0 or right_cnt[b + 1] == 0:
                        continue
                    la = _area(llo, lhi)
                    cost = trav_cost + (lc * la + right_cnt[b + 1] * right_area[b + 1]) / parent_area \
                        if parent_area > 0.0 else trav_cost + 0.5 * count
                    if cost < best_cost:
                        best_cost = cost
                        best_split = b
                if best_split < 0:
                    mid = s + count // 2
                else:
                    # in-place partition: bin <= best_split to the left (two pointers, index order)
                    i = s
                    j = e - 1
                    while i <= j:
                        t = order[i]
                        b = int((cen[t, axis] - cmin[axis]) * scale)
                        if b >= n_bins:
                            b = n_bins - 1
                        if b <= best_split:
                            i += 1
                        else:
                            order[i] = order[j]
                            order[j] = t
                            j -= 1
                    mid = i
                    if mid == s or mid == e:
                        mid = s + count // 2
        if make_leaf:
            node_right[node] = -1
            node_start[node] = s
            node_count[node] = count
            if depth > max_depth_seen:
                max_depth_seen = depth
            if sp == 0:
                break
            sp -= 1
            s = st_s[sp]
            e = st_e[sp]
            depth = st_d[sp]
            parent = st_p[sp]
            continue
        node_start[node] = s
        node_count[node] = 0
        # second child is pending; the first child follows immediately
        st_s[sp] = mid
        st_e[sp] = e
        st_d[sp] = depth + 1
        st_p[sp] = node
        sp += 1
        e = mid
        depth = depth + 1
        parent = -1
    stats[0] = max_depth_seen
    return n_nodes


def pseudo_normals(mesh: Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(face_normal [m, 3], edge_normal [m, 3, 3], vertex_normal [n, 3])``: unit face normals, edge
    pseudo-normals (sum of the adjacent faces' normals, i.e. weights pi each, normalised) and
    angle-weighted vertex pseudo-normals (Baerentzen & Aanaes 2005)."""
    fn = mesh.face_normals()
    m = mesh.n_faces
    _, face_edge, _ = mesh.edges()
    acc = np.zeros((int(face_edge.max()) + 1 if m else 0, 3))
    for k in range(3):
        np.add.at(acc, face_edge[:, k], fn)
    en = np.zeros((m, 3, 3))
    for k in range(3):
        v = acc[face_edge[:, k]]
        n = np.linalg.norm(v, axis=1, keepdims=True)
        en[:, k] = np.divide(v, n, out=fn.copy(), where=n > 0.0)
    vn = mesh.vertex_normals()
    return np.ascontiguousarray(fn), np.ascontiguousarray(en), np.ascontiguousarray(vn)


def build_tree(mesh: Mesh, leaf_size: int | None = None, bins: int | None = None,
               traversal_cost: float | None = None) -> MeshTree:
    """Tree over every face of ``mesh`` (defaults: ``geom.bvh_leaf_size``, ``geom.bvh_bins``,
    ``geom.bvh_traversal_cost``)."""
    leaf = int(resolve(leaf_size, "bvh_leaf_size"))
    n_bins = int(resolve(bins, "bvh_bins"))
    trav = float(resolve(traversal_cost, "bvh_traversal_cost"))
    if leaf < 1 or n_bins < 2:
        raise ValueError("leaf size must be >= 1 and bins >= 2")
    m = mesh.n_faces
    if m == 0:
        raise ValueError("cannot build a tree over a mesh without faces")
    tris = mesh.triangles()
    tri_lo = contiguous(tris.min(axis=1))
    tri_hi = contiguous(tris.max(axis=1))
    cen = contiguous(0.5 * (tri_lo + tri_hi))
    order = np.arange(m, dtype=np.int32)
    cap = 2 * m - 1
    node_lo = np.zeros((cap, 3))
    node_hi = np.zeros((cap, 3))
    node_right = np.full(cap, -1, np.int32)
    node_start = np.zeros(cap, np.int32)
    node_count = np.zeros(cap, np.int32)
    stats = np.zeros(1, np.int64)
    nn = int(build_nodes_nb(tri_lo, tri_hi, cen, leaf, n_bins, trav, order, node_lo, node_hi, node_right,
                            node_start, node_count, stats))
    fn, en, vn = pseudo_normals(mesh)
    info = np.zeros(N_INFO, np.int64)
    info[INFO_NODES] = nn
    info[INFO_DEPTH] = int(stats[0])
    info[INFO_CLOSED] = 1 if mesh.is_watertight() else 0
    info[INFO_LEAF_SIZE] = leaf
    return MeshTree(contiguous(node_lo[:nn]), contiguous(node_hi[:nn]), contiguous(node_right[:nn]),
                    contiguous(node_start[:nn]), contiguous(node_count[:nn]), contiguous(tris[order]),
                    contiguous(order), contiguous(mesh.vertices), contiguous(mesh.faces, np.int32), fn, en, vn, info)


_STATIC: OrderedDict[str, MeshTree] = OrderedDict()
STATIC_CACHE_SIZE = 64


def static_tree(mesh: Mesh) -> MeshTree:
    """Tree of a static mesh (machine component, fixture), built once per process and content."""
    key = mesh.content_hash()
    tree = _STATIC.get(key)
    if tree is None:
        tree = build_tree(mesh)
        _STATIC[key] = tree
        while len(_STATIC) > STATIC_CACHE_SIZE:
            _STATIC.popitem(last=False)
    else:
        _STATIC.move_to_end(key)
    return tree


def save_tree(path, tree: MeshTree) -> None:
    np.savez(Path(path), **{name: getattr(tree, name) for name in MeshTree._fields})


def load_tree(path) -> MeshTree:
    with np.load(Path(path)) as data:
        return MeshTree(**{name: contiguous(data[name]) for name in MeshTree._fields})


# ============================================================================ ray kernels

@njit(f8(F1, F1, F1, F1, f8, f8), **OPTS)
def ray_box_entry_nb(o, d, lo, hi, tmin, tmax):
    """Entry distance of the ray into a box clipped to [tmin, tmax], or +inf if it misses (conservative:
    the far distance is scaled by 1 + 2 gamma(3))."""
    t0 = tmin
    t1 = tmax
    for k in range(3):
        if d[k] == 0.0:
            if o[k] < lo[k] or o[k] > hi[k]:
                return np.inf
            continue
        inv = 1.0 / d[k]
        tn = (lo[k] - o[k]) * inv
        tf = (hi[k] - o[k]) * inv
        if tn > tf:
            tn, tf = tf, tn
        tf *= BOX_FAR_SCALE
        if tn > t0:
            t0 = tn
        if tf < t1:
            t1 = tf
        if t0 > t1:
            return np.inf
    return t0


@njit(types.void(F1, F1), **OPTS)
def ray_setup_nb(d, setup):
    """Per-ray constants of the watertight test: setup = [kx, ky, kz, Sx, Sy, Sz] (Woop et al. 2013)."""
    kz = 0
    if abs(d[1]) > abs(d[kz]):
        kz = 1
    if abs(d[2]) > abs(d[kz]):
        kz = 2
    kx = kz + 1
    if kx == 3:
        kx = 0
    ky = kx + 1
    if ky == 3:
        ky = 0
    if d[kz] < 0.0:
        kx, ky = ky, kx
    setup[0] = kx
    setup[1] = ky
    setup[2] = kz
    setup[3] = d[kx] / d[kz]
    setup[4] = d[ky] / d[kz]
    setup[5] = 1.0 / d[kz]


@njit(i8(F1, F1, F2, f8, f8, F1), **OPTS)
def ray_triangle_nb(o, setup, v, tmin, tmax, out):
    """Watertight ray/triangle test (Woop, Benthin & Wald 2013). ``v`` [3, 3] corners, ``setup`` from
    :func:`ray_setup_nb`. Returns 1 on a hit with tmin <= t <= tmax (out = [t, b0, b1, b2, degenerate]),
    else 0; degenerate = 1 when an edge function is exactly zero (the ray meets an edge or a vertex)."""
    kx = int(setup[0])
    ky = int(setup[1])
    kz = int(setup[2])
    sx = setup[3]
    sy = setup[4]
    sz = setup[5]
    ax = v[0, kx] - o[kx]
    ay = v[0, ky] - o[ky]
    az = v[0, kz] - o[kz]
    bx = v[1, kx] - o[kx]
    by = v[1, ky] - o[ky]
    bz = v[1, kz] - o[kz]
    cx = v[2, kx] - o[kx]
    cy = v[2, ky] - o[ky]
    cz = v[2, kz] - o[kz]
    ax = ax - sx * az
    ay = ay - sy * az
    bx = bx - sx * bz
    by = by - sy * bz
    cx = cx - sx * cz
    cy = cy - sy * cz
    u = cx * by - cy * bx
    vv = ax * cy - ay * cx
    w = bx * ay - by * ax
    if (u < 0.0 or vv < 0.0 or w < 0.0) and (u > 0.0 or vv > 0.0 or w > 0.0):
        return 0
    det = u + vv + w
    if det == 0.0:
        return 0
    az = sz * az
    bz = sz * bz
    cz = sz * cz
    T = u * az + vv * bz + w * cz
    t = T / det
    if t < tmin or t > tmax:
        return 0
    out[0] = t
    out[1] = u / det
    out[2] = vv / det
    out[3] = w / det
    out[4] = 1.0 if (u == 0.0 or vv == 0.0 or w == 0.0) else 0.0
    return 1


@njit(i8(TREE, F1, F1, f8, f8, F1), **OPTS)
def ray_first_hit_nb(tree, o, d, tmin, tmax, out):
    """Nearest hit with tmin <= t <= tmax: returns the face id (-1 for none), out = [t, b0, b1, b2,
    degenerate] (barycentrics of the face's corners v0, v1, v2)."""
    setup = np.empty(6)
    ray_setup_nb(d, setup)
    stack = np.empty(STACK_SIZE, np.int64)
    hit = np.empty(5)
    sp = 0
    stack[sp] = 0
    sp += 1
    best = tmax
    best_face = -1
    while sp > 0:
        sp -= 1
        node = stack[sp]
        if ray_box_entry_nb(o, d, tree.node_lo[node], tree.node_hi[node], tmin, best) == np.inf:
            continue
        right = tree.node_right[node]
        if right < 0:
            st = tree.node_start[node]
            for slot in range(st, st + tree.node_count[node]):
                if ray_triangle_nb(o, setup, tree.tri_v[slot], tmin, best, hit) == 1:
                    face = tree.tri_face[slot]
                    if hit[0] < best or (hit[0] == best and (best_face < 0 or face < best_face)):
                        best = hit[0]
                        best_face = face
                        for k in range(5):
                            out[k] = hit[k]
            continue
        left = node + 1
        tl = ray_box_entry_nb(o, d, tree.node_lo[left], tree.node_hi[left], tmin, best)
        tr = ray_box_entry_nb(o, d, tree.node_lo[right], tree.node_hi[right], tmin, best)
        if tl <= tr:
            if tr != np.inf:
                stack[sp] = right
                sp += 1
            if tl != np.inf:
                stack[sp] = left
                sp += 1
        else:
            if tl != np.inf:
                stack[sp] = left
                sp += 1
            stack[sp] = right
            sp += 1
    return best_face


@njit(i8(TREE, F1, F1, f8, f8, F1, I1, B1), **OPTS)
def ray_all_hits_nb(tree, o, d, tmin, tmax, out_t, out_face, out_degenerate):
    """Every hit with tmin <= t <= tmax, sorted by (t, face). Writes at most len(out_t) and returns the
    total count (call again with larger buffers if it exceeds them)."""
    setup = np.empty(6)
    ray_setup_nb(d, setup)
    stack = np.empty(STACK_SIZE, np.int64)
    hit = np.empty(5)
    cap = out_t.shape[0]
    n = 0
    sp = 0
    stack[sp] = 0
    sp += 1
    while sp > 0:
        sp -= 1
        node = stack[sp]
        if ray_box_entry_nb(o, d, tree.node_lo[node], tree.node_hi[node], tmin, tmax) == np.inf:
            continue
        right = tree.node_right[node]
        if right < 0:
            st = tree.node_start[node]
            for slot in range(st, st + tree.node_count[node]):
                if ray_triangle_nb(o, setup, tree.tri_v[slot], tmin, tmax, hit) == 1:
                    if n < cap:
                        out_t[n] = hit[0]
                        out_face[n] = tree.tri_face[slot]
                        out_degenerate[n] = np.int8(hit[4])
                    n += 1
            continue
        stack[sp] = right
        sp += 1
        stack[sp] = node + 1
        sp += 1
    # insertion sort of the written hits by (t, face): deterministic regardless of traversal order
    w = min(n, cap)
    for i in range(1, w):
        tt = out_t[i]
        ff = out_face[i]
        dg = out_degenerate[i]
        j = i - 1
        while j >= 0 and (out_t[j] > tt or (out_t[j] == tt and out_face[j] > ff)):
            out_t[j + 1] = out_t[j]
            out_face[j + 1] = out_face[j]
            out_degenerate[j + 1] = out_degenerate[j]
            j -= 1
        out_t[j + 1] = tt
        out_face[j + 1] = ff
        out_degenerate[j + 1] = dg
    return n


@njit(types.void(TREE, F2, F2, F1, F1, F1, I1), **OPTS)
def rays_first_hit_nb(tree, O, D, tmin, tmax, out_t, out_face):
    """Batch of first hits (rows of O, D); misses give t = inf, face = -1."""
    hit = np.empty(5)
    for i in range(O.shape[0]):
        f = ray_first_hit_nb(tree, O[i], D[i], tmin[i], tmax[i], hit)
        out_face[i] = f
        out_t[i] = hit[0] if f >= 0 else np.inf


@njit(types.void(TREE, F2, F2, F1, F1, I8_1), **OPTS)
def rays_count_hits_nb(tree, O, D, tmin, tmax, counts):
    """Number of hits per ray (first pass of the CSR batch)."""
    t0 = np.empty(0)
    f0 = np.empty(0, np.int32)
    g0 = np.empty(0, np.int8)
    for i in range(O.shape[0]):
        counts[i] = ray_all_hits_nb(tree, O[i], D[i], tmin[i], tmax[i], t0, f0, g0)


@njit(types.void(TREE, F2, F2, F1, F1, I8_1, F1, I1), **OPTS)
def rays_fill_hits_nb(tree, O, D, tmin, tmax, offsets, out_t, out_face):
    """Second pass of the CSR batch: hits of ray i in out[offsets[i]:offsets[i + 1]], sorted by t."""
    for i in range(O.shape[0]):
        a = offsets[i]
        b = offsets[i + 1]
        g = np.empty(b - a, np.int8)
        ray_all_hits_nb(tree, O[i], D[i], tmin[i], tmax[i], out_t[a:b], out_face[a:b], g)


# ============================================================================ closest point / distance

@njit(i8(F1, F2, F1), **OPTS)
def closest_point_triangle_nb(p, v, out):
    """Closest point of triangle v [3, 3] to p (Ericson 2005, 5.1.5); out = [x, y, z]; returns the
    feature code (abi.FEATURE_*)."""
    abx = v[1, 0] - v[0, 0]
    aby = v[1, 1] - v[0, 1]
    abz = v[1, 2] - v[0, 2]
    acx = v[2, 0] - v[0, 0]
    acy = v[2, 1] - v[0, 1]
    acz = v[2, 2] - v[0, 2]
    apx = p[0] - v[0, 0]
    apy = p[1] - v[0, 1]
    apz = p[2] - v[0, 2]
    d1 = abx * apx + aby * apy + abz * apz
    d2 = acx * apx + acy * apy + acz * apz
    if d1 <= 0.0 and d2 <= 0.0:
        out[0] = v[0, 0]
        out[1] = v[0, 1]
        out[2] = v[0, 2]
        return 4
    bpx = p[0] - v[1, 0]
    bpy = p[1] - v[1, 1]
    bpz = p[2] - v[1, 2]
    d3 = abx * bpx + aby * bpy + abz * bpz
    d4 = acx * bpx + acy * bpy + acz * bpz
    if d3 >= 0.0 and d4 <= d3:
        out[0] = v[1, 0]
        out[1] = v[1, 1]
        out[2] = v[1, 2]
        return 5
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        s = d1 / (d1 - d3)
        out[0] = v[0, 0] + s * abx
        out[1] = v[0, 1] + s * aby
        out[2] = v[0, 2] + s * abz
        return 1
    cpx = p[0] - v[2, 0]
    cpy = p[1] - v[2, 1]
    cpz = p[2] - v[2, 2]
    d5 = abx * cpx + aby * cpy + abz * cpz
    d6 = acx * cpx + acy * cpy + acz * cpz
    if d6 >= 0.0 and d5 <= d6:
        out[0] = v[2, 0]
        out[1] = v[2, 1]
        out[2] = v[2, 2]
        return 6
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        s = d2 / (d2 - d6)
        out[0] = v[0, 0] + s * acx
        out[1] = v[0, 1] + s * acy
        out[2] = v[0, 2] + s * acz
        return 3
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        s = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        out[0] = v[1, 0] + s * (v[2, 0] - v[1, 0])
        out[1] = v[1, 1] + s * (v[2, 1] - v[1, 1])
        out[2] = v[1, 2] + s * (v[2, 2] - v[1, 2])
        return 2
    den = 1.0 / (va + vb + vc)
    sv = vb * den
    sw = vc * den
    out[0] = v[0, 0] + abx * sv + acx * sw
    out[1] = v[0, 1] + aby * sv + acy * sw
    out[2] = v[0, 2] + abz * sv + acz * sw
    return 0


@njit(f8(F1, F1, F1), **OPTS)
def _box_dist2(p, lo, hi):
    s = 0.0
    for k in range(3):
        if p[k] < lo[k]:
            s += (lo[k] - p[k]) ** 2
        elif p[k] > hi[k]:
            s += (p[k] - hi[k]) ** 2
    return s


@njit(i8(TREE, F1, f8, F1), **OPTS)
def closest_point_nb(tree, p, max_dist, out):
    """Closest surface point within max_dist: returns the face id (-1 if none), out = [x, y, z, distance,
    feature]. Depth-first, nearer child first, boxes farther than the best distance pruned."""
    stack = np.empty(STACK_SIZE, np.int64)
    sdist = np.empty(STACK_SIZE)
    q = np.empty(3)
    best2 = max_dist * max_dist if max_dist < np.inf else np.inf
    best_face = -1
    sp = 0
    stack[0] = 0
    sdist[0] = _box_dist2(p, tree.node_lo[0], tree.node_hi[0])
    sp = 1
    while sp > 0:
        sp -= 1
        node = stack[sp]
        if sdist[sp] > best2:
            continue
        right = tree.node_right[node]
        if right < 0:
            st = tree.node_start[node]
            for slot in range(st, st + tree.node_count[node]):
                feat = closest_point_triangle_nb(p, tree.tri_v[slot], q)
                d2 = (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 + (q[2] - p[2]) ** 2
                face = tree.tri_face[slot]
                if d2 < best2 or (d2 == best2 and best_face >= 0 and face < best_face):
                    best2 = d2
                    best_face = face
                    out[0] = q[0]
                    out[1] = q[1]
                    out[2] = q[2]
                    out[4] = feat
            continue
        left = node + 1
        dl = _box_dist2(p, tree.node_lo[left], tree.node_hi[left])
        dr = _box_dist2(p, tree.node_lo[right], tree.node_hi[right])
        if dl <= dr:
            stack[sp] = right
            sdist[sp] = dr
            sp += 1
            stack[sp] = left
            sdist[sp] = dl
            sp += 1
        else:
            stack[sp] = left
            sdist[sp] = dl
            sp += 1
            stack[sp] = right
            sdist[sp] = dr
            sp += 1
    if best_face >= 0:
        out[3] = math.sqrt(best2)
    return best_face


@njit(f8(TREE, i8, i8, F1, F1), **OPTS)
def pseudo_normal_sign_nb(tree, face, feature, p, c):
    """Sign (+1 outside, -1 inside, 0 on the surface) of p relative to its closest point c on ``face``."""
    if feature == 0:
        nx = tree.face_normal[face, 0]
        ny = tree.face_normal[face, 1]
        nz = tree.face_normal[face, 2]
    elif feature <= 3:
        k = feature - 1
        nx = tree.edge_normal[face, k, 0]
        ny = tree.edge_normal[face, k, 1]
        nz = tree.edge_normal[face, k, 2]
    else:
        vtx = tree.faces[face, feature - 4]
        nx = tree.vertex_normal[vtx, 0]
        ny = tree.vertex_normal[vtx, 1]
        nz = tree.vertex_normal[vtx, 2]
    s = (p[0] - c[0]) * nx + (p[1] - c[1]) * ny + (p[2] - c[2]) * nz
    if s > 0.0:
        return 1.0
    if s < 0.0:
        return -1.0
    return 0.0


@njit(types.void(TREE, F2, f8, F1, I1, F2), **OPTS)
def signed_distance_nb(tree, P, max_dist, out_d, out_face, out_point):
    """Signed distance per point (negative inside), with the closest face and point. Points farther than
    max_dist (a narrow band asked by the caller) get a NaN distance and face -1."""
    res = np.empty(5)
    c = np.empty(3)
    for i in range(P.shape[0]):
        f = closest_point_nb(tree, P[i], max_dist, res)
        out_face[i] = f
        if f < 0:
            out_d[i] = np.nan
            for k in range(3):
                out_point[i, k] = np.nan
            continue
        for k in range(3):
            c[k] = res[k]
            out_point[i, k] = res[k]
        s = pseudo_normal_sign_nb(tree, f, int(res[4]), P[i], c)
        out_d[i] = s * res[3]


# ============================================================================ inside / outside

@njit(i8(TREE, F1, F2, i8), **OPTS)
def point_in_mesh_nb(tree, p, dirs, needed):
    """INSIDE / OUTSIDE / ON_SURFACE / UNDECIDED by the majority of ``needed`` non-degenerate parity rays
    taken in the order of ``dirs``."""
    cap = 256
    ts = np.empty(cap)
    fs = np.empty(cap, np.int32)
    gs = np.empty(cap, np.int8)
    votes_in = 0
    votes = 0
    for r in range(dirs.shape[0]):
        n = ray_all_hits_nb(tree, p, dirs[r], 0.0, np.inf, ts, fs, gs)
        if n > cap:
            big_t = np.empty(n)
            big_f = np.empty(n, np.int32)
            big_g = np.empty(n, np.int8)
            ray_all_hits_nb(tree, p, dirs[r], 0.0, np.inf, big_t, big_f, big_g)
            t_arr = big_t
            g_arr = big_g
        else:
            t_arr = ts
            g_arr = gs
        degenerate = False
        for i in range(n):
            if t_arr[i] == 0.0:
                return 2
            if g_arr[i] != 0:
                degenerate = True
        if degenerate:
            continue
        votes += 1
        if n % 2 == 1:
            votes_in += 1
        if votes == needed:
            break
    if votes == 0:
        return -1
    if 2 * votes_in > votes:
        return 1
    if 2 * votes_in < votes:
        return 0
    return -1


@njit(types.void(TREE, F2, F2, i8, B1), **OPTS)
def points_in_mesh_nb(tree, P, dirs, needed, out):
    for i in range(P.shape[0]):
        out[i] = np.int8(point_in_mesh_nb(tree, P[i], dirs, needed))


@njit(types.void(F2, types.int32[:, ::1], F2, F1), **OPTS)
def winding_numbers_nb(vertices, faces, P, out):
    """Generalised winding number per point: sum of the faces' signed solid angles / 4 pi
    (Van Oosterom & Strackee 1983). O(faces) per point."""
    for i in range(P.shape[0]):
        px = P[i, 0]
        py = P[i, 1]
        pz = P[i, 2]
        total = 0.0
        for f in range(faces.shape[0]):
            a0 = vertices[faces[f, 0], 0] - px
            a1 = vertices[faces[f, 0], 1] - py
            a2 = vertices[faces[f, 0], 2] - pz
            b0 = vertices[faces[f, 1], 0] - px
            b1 = vertices[faces[f, 1], 1] - py
            b2 = vertices[faces[f, 1], 2] - pz
            c0 = vertices[faces[f, 2], 0] - px
            c1 = vertices[faces[f, 2], 1] - py
            c2 = vertices[faces[f, 2], 2] - pz
            la = math.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
            lb = math.sqrt(b0 * b0 + b1 * b1 + b2 * b2)
            lc = math.sqrt(c0 * c0 + c1 * c1 + c2 * c2)
            det = a0 * (b1 * c2 - b2 * c1) - a1 * (b0 * c2 - b2 * c0) + a2 * (b0 * c1 - b1 * c0)
            den = la * lb * lc + (a0 * b0 + a1 * b1 + a2 * b2) * lc + (a0 * c0 + a1 * c1 + a2 * c2) * lb \
                + (b0 * c0 + b1 * c1 + b2 * c2) * la
            total += 2.0 * math.atan2(det, den)
        out[i] = total / (4.0 * math.pi)


# ============================================================================ Python API

def _vec(x) -> np.ndarray:
    return contiguous(np.asarray(x, dtype=np.float64).reshape(3))


def _rows(x) -> np.ndarray:
    return contiguous(np.asarray(x, dtype=np.float64).reshape(-1, 3))


def ray_first_hit(tree: MeshTree, origin, direction, t_min: float = 0.0, t_max: float = math.inf):
    """``(t, face, barycentrics[3])`` of the nearest hit, or None. ``t`` is in units of |direction|."""
    out = np.zeros(5)
    face = int(ray_first_hit_nb(tree, _vec(origin), _vec(direction), float(t_min), float(t_max), out))
    if face < 0:
        return None
    return float(out[0]), face, out[1:4].copy()


def ray_all_hits(tree: MeshTree, origin, direction, t_min: float = 0.0, t_max: float = math.inf):
    """``(t [k], face [k])`` of every hit sorted by t (ties by face id)."""
    o, d = _vec(origin), _vec(direction)
    cap = 64
    while True:
        ts = np.empty(cap)
        fs = np.empty(cap, np.int32)
        gs = np.empty(cap, np.int8)
        n = int(ray_all_hits_nb(tree, o, d, float(t_min), float(t_max), ts, fs, gs))
        if n <= cap:
            return ts[:n].copy(), fs[:n].copy()
        cap = n


def segment_hit(tree: MeshTree, p0, p1):
    """First crossing of the segment p0 -> p1: ``(fraction in [0, 1], face, point)`` or None."""
    a, b = _vec(p0), _vec(p1)
    hit = ray_first_hit(tree, a, b - a, 0.0, 1.0)
    if hit is None:
        return None
    s, face, _ = hit
    return s, face, a + s * (b - a)


def _batch_limits(n, t_min, t_max):
    tmin = contiguous(np.broadcast_to(np.asarray(t_min, dtype=np.float64), (n,)))
    tmax = contiguous(np.broadcast_to(np.asarray(t_max, dtype=np.float64), (n,)))
    return tmin, tmax


def rays_first_hit(tree: MeshTree, origins, directions, t_min=0.0, t_max=math.inf):
    """Batch first hits: ``(t [n] (inf for a miss), face [n] (-1))``; directions may be one row for all."""
    O = _rows(origins)
    D = contiguous(np.broadcast_to(np.asarray(directions, dtype=np.float64).reshape(-1, 3), O.shape))
    tmin, tmax = _batch_limits(len(O), t_min, t_max)
    out_t = np.empty(len(O))
    out_f = np.empty(len(O), np.int32)
    rays_first_hit_nb(tree, O, D, tmin, tmax, out_t, out_f)
    return out_t, out_f


def segments_hit(tree: MeshTree, p0, p1):
    """Batch of segment tests: ``(fraction [n] (inf for none), face [n])``."""
    a = _rows(p0)
    b = _rows(p1)
    return rays_first_hit(tree, a, b - a, 0.0, 1.0)


def rays_all_hits(tree: MeshTree, origins, directions, t_min=0.0, t_max=math.inf):
    """Batch of all hits in CSR form: ``(offsets [n + 1], t [k], face [k])``, hits of ray i in
    ``offsets[i]:offsets[i + 1]`` sorted by t (dexelisation of a stock mesh, line-of-sight counts)."""
    O = _rows(origins)
    D = contiguous(np.broadcast_to(np.asarray(directions, dtype=np.float64).reshape(-1, 3), O.shape))
    tmin, tmax = _batch_limits(len(O), t_min, t_max)
    counts = np.zeros(len(O), np.int64)
    rays_count_hits_nb(tree, O, D, tmin, tmax, counts)
    offsets = np.zeros(len(O) + 1, np.int64)
    np.cumsum(counts, out=offsets[1:])
    out_t = np.empty(int(offsets[-1]))
    out_f = np.empty(int(offsets[-1]), np.int32)
    rays_fill_hits_nb(tree, O, D, tmin, tmax, offsets, out_t, out_f)
    return offsets, out_t, out_f


def closest_point(tree: MeshTree, point, max_dist: float = math.inf):
    """``(point [3], distance, face, feature)`` of the closest surface point, or None beyond max_dist."""
    out = np.zeros(5)
    face = int(closest_point_nb(tree, _vec(point), float(max_dist), out))
    if face < 0:
        return None
    return out[:3].copy(), float(out[3]), face, int(out[4])


def signed_distance(tree: MeshTree, points, max_dist: float = math.inf, allow_open: bool = False):
    """``(distance [n], face [n], closest [n, 3])``: negative inside a closed mesh. Raises for a mesh that is
    not closed (the sign has no meaning there) unless ``allow_open``. Points beyond max_dist give NaN."""
    if not allow_open and tree.info[INFO_CLOSED] != 1:
        raise ValueError("signed distance needs a closed, consistently oriented mesh (Mesh.is_watertight)")
    P = _rows(points)
    d = np.empty(len(P))
    f = np.empty(len(P), np.int32)
    c = np.empty((len(P), 3))
    signed_distance_nb(tree, P, float(max_dist), d, f, c)
    return d, f, c


def parity_directions(count: int = 9) -> np.ndarray:
    """Fixed ray directions for the parity test: a Fibonacci spiral on the sphere (none along a
    coordinate axis or diagonal, so faceted CAD meshes rarely make a ray meet an edge exactly)."""
    golden = math.pi * (3.0 - math.sqrt(5.0))
    k = np.arange(count, dtype=np.float64)
    z = 1.0 - (2.0 * k + 1.0) / count
    rho = np.sqrt(1.0 - z * z)
    phi = golden * k + 0.5
    return contiguous(np.column_stack([rho * np.cos(phi), rho * np.sin(phi), z]))


def points_in_mesh(tree: MeshTree, points, rays: int | None = None) -> np.ndarray:
    """int8 per point: 1 inside, 0 outside, 2 on the surface. Undecided points (every ray degenerate) are
    settled by the generalised winding number (> 1/2 inside)."""
    P = _rows(points)
    needed = int(resolve(rays, "point_in_mesh_rays"))
    dirs = parity_directions(max(9, 3 * needed))
    out = np.empty(len(P), np.int8)
    points_in_mesh_nb(tree, P, dirs, needed, out)
    undecided = np.flatnonzero(out == UNDECIDED)
    if len(undecided):
        w = winding_numbers(tree, P[undecided])
        out[undecided] = np.where(w > 0.5, INSIDE, OUTSIDE).astype(np.int8)
    return out


def winding_numbers(tree_or_mesh, points) -> np.ndarray:
    """Generalised winding number per point (1 inside, 0 outside a closed outward mesh; fractional near
    holes)."""
    P = _rows(points)
    v = contiguous(tree_or_mesh.vertices)
    f = contiguous(tree_or_mesh.faces, np.int32)
    out = np.empty(len(P))
    winding_numbers_nb(v, f, P, out)
    return out


KERNELS = (build_nodes_nb, _area, ray_box_entry_nb, ray_setup_nb, ray_triangle_nb, ray_first_hit_nb, ray_all_hits_nb,
           rays_first_hit_nb, rays_count_hits_nb, rays_fill_hits_nb, closest_point_triangle_nb, _box_dist2,
           closest_point_nb, pseudo_normal_sign_nb, signed_distance_nb, point_in_mesh_nb, points_in_mesh_nb,
           winding_numbers_nb)

__all__ = ["build_tree", "static_tree", "save_tree", "load_tree", "pseudo_normals", "ray_first_hit", "ray_all_hits",
           "segment_hit", "rays_first_hit", "segments_hit", "rays_all_hits", "closest_point", "signed_distance",
           "points_in_mesh", "winding_numbers", "parity_directions", "MeshTree", "KERNELS", "build_nodes_nb",
           "ray_box_entry_nb", "ray_setup_nb", "ray_triangle_nb", "ray_first_hit_nb", "ray_all_hits_nb",
           "rays_first_hit_nb", "rays_count_hits_nb", "rays_fill_hits_nb", "closest_point_triangle_nb",
           "closest_point_nb", "pseudo_normal_sign_nb", "signed_distance_nb", "point_in_mesh_nb",
           "points_in_mesh_nb", "winding_numbers_nb", "GAMMA3", "BOX_FAR_SCALE"]
