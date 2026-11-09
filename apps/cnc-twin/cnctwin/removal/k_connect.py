"""Connectivity of the material after subtraction (docs/models/removal.md "Loose bodies").

The material graph is the level-0 z family: every z segment is a node, and two segments of 4-neighbouring
z rays are joined when their z ranges overlap by more than zero (face-connected columns of the column
model whose volume the ledger books). Components come from union-find with path halving and union by
index (deterministic: the smaller root wins). A body that is no longer joined to the held material is
loose: its volume, mass centre and box are measured on the same columns, and it is removed from every
ray of every level whose segment contains a node of it (x and y families: the segment's nodes are looked
up in the z labels). Pass only when a level-0 z ray changed its segment count (FI_TOPO): shortening
columns cannot separate the column graph without a column losing or splitting a segment.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, types

from . import abi as A
from . import k_core as C
from .nbt import F1, F2, FIELD_T, I1, I4_1, OPTS, SCRATCH_T, f8, i8, void

Q = A.QUANTUM_M
K = A.K_INLINE


@njit(i8(I4_1, i8), **OPTS)
def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@njit(void(I4_1, types.int32[:, :, ::1], I4_1, types.int32[:, :, ::1], I4_1, i8, I1, I1), **OPTS)
def _segs(cnt, seg, link, pseg, pnext, ray, lo, hi):
    n = np.int64(cnt[ray])
    b = np.int64(link[ray])
    s = 0
    for k in range(n):
        if k < K:
            lo[k] = seg[ray, k, 0]
            hi[k] = seg[ray, k, 1]
        else:
            if s == K:
                b = np.int64(pnext[b])
                s = 0
            lo[k] = pseg[b, s, 0]
            hi[k] = pseg[b, s, 1]
            s += 1


@njit(i8(FIELD_T, I1, I4_1), **OPTS)
def label_z(f, first, parent):
    """Union-find over the level-0 z segments. ``first[ray_local]`` gets the id of the ray's first segment
    (CSR, size nrays_z + 1); ``parent`` (size >= number of segments) ends with every segment's root.
    Returns the number of segments."""
    cnt = f.cnt
    seg = f.seg
    link = f.link
    pseg = f.pseg
    pnext = f.pnext
    nx = f.lv_dims[0, 0]
    ny = f.lv_dims[0, 1]
    r0 = f.lv_ray0[0, 2]
    total = 0
    for i in range(nx):
        for j in range(ny):
            first[i * ny + j] = total
            total += cnt[r0 + i * ny + j]
    first[nx * ny] = total
    for s in range(total):
        parent[s] = s
    lo_a = np.empty(A.MAX_SEG, np.int64)
    hi_a = np.empty(A.MAX_SEG, np.int64)
    lo_b = np.empty(A.MAX_SEG, np.int64)
    hi_b = np.empty(A.MAX_SEG, np.int64)
    for i in range(nx):
        for j in range(ny):
            ra = i * ny + j
            na = first[ra + 1] - first[ra]
            if na == 0:
                continue
            _segs(cnt, seg, link, pseg, pnext, r0 + ra, lo_a, hi_a)
            for nb_dir in range(2):
                ii = i + 1 if nb_dir == 0 else i
                jj = j if nb_dir == 0 else j + 1
                if ii >= nx or jj >= ny:
                    continue
                rb = ii * ny + jj
                nb = first[rb + 1] - first[rb]
                if nb == 0:
                    continue
                _segs(cnt, seg, link, pseg, pnext, r0 + rb, lo_b, hi_b)
                ka = 0
                kb = 0
                while ka < na and kb < nb:
                    if lo_a[ka] < hi_b[kb] and lo_b[kb] < hi_a[ka]:
                        x = _find(parent, first[ra] + ka)
                        y = _find(parent, first[rb] + kb)
                        if x != y:
                            if x < y:
                                parent[y] = x
                            else:
                                parent[x] = y
                    if hi_a[ka] < hi_b[kb]:
                        ka += 1
                    else:
                        kb += 1
    for s in range(total):
        parent[s] = _find(parent, s)
    return total


@njit(i8(FIELD_T, I1, I4_1, i8, F2), **OPTS)
def component_stats(f, first, parent, total, out):
    """Per root segment id r: out[r] = [volume (m^3), sum x V, sum y V, sum z V, x_lo, y_lo, z_lo, x_hi, y_hi,
    z_hi] (columns of a^2 x length, level 0). Returns the number of components."""
    cnt = f.cnt
    seg = f.seg
    link = f.link
    pseg = f.pseg
    pnext = f.pnext
    nx = f.lv_dims[0, 0]
    ny = f.lv_dims[0, 1]
    r0 = f.lv_ray0[0, 2]
    a = f.lv_cell_m[0]
    o = f.lv_origin_m[0]
    for s in range(total):
        if parent[s] == s:
            out[s, 0] = 0.0
            out[s, 1] = 0.0
            out[s, 2] = 0.0
            out[s, 3] = 0.0
            out[s, 4] = 1e300
            out[s, 5] = 1e300
            out[s, 6] = 1e300
            out[s, 7] = -1e300
            out[s, 8] = -1e300
            out[s, 9] = -1e300
    lo = np.empty(A.MAX_SEG, np.int64)
    hi = np.empty(A.MAX_SEG, np.int64)
    ncomp = 0
    for s in range(total):
        if parent[s] == s:
            ncomp += 1
    for i in range(nx):
        x = o[0] + i * a
        for j in range(ny):
            ra = i * ny + j
            n = first[ra + 1] - first[ra]
            if n == 0:
                continue
            y = o[1] + j * a
            _segs(cnt, seg, link, pseg, pnext, r0 + ra, lo, hi)
            for k in range(n):
                r = parent[first[ra] + k]
                z0 = o[2] + lo[k] * Q
                z1 = o[2] + hi[k] * Q
                v = (z1 - z0) * a * a
                out[r, 0] += v
                out[r, 1] += v * x
                out[r, 2] += v * y
                out[r, 3] += v * 0.5 * (z0 + z1)
                out[r, 4] = min(out[r, 4], x - 0.5 * a)
                out[r, 5] = min(out[r, 5], y - 0.5 * a)
                out[r, 6] = min(out[r, 6], z0)
                out[r, 7] = max(out[r, 7], x + 0.5 * a)
                out[r, 8] = max(out[r, 8], y + 0.5 * a)
                out[r, 9] = max(out[r, 9], z1)
    return ncomp


@njit(void(FIELD_T, I1, I4_1, i8, F1, F1, types.int8[::1]), **OPTS)
def components_in_box(f, first, parent, total, lo, hi, flags):
    """flags[root] = 1 for every component with a level-0 z column (node inside the box's x-y range) whose
    segment overlaps the box's z range (the held regions of the workholding)."""
    cnt = f.cnt
    seg = f.seg
    link = f.link
    pseg = f.pseg
    pnext = f.pnext
    nx = f.lv_dims[0, 0]
    ny = f.lv_dims[0, 1]
    r0 = f.lv_ray0[0, 2]
    a = f.lv_cell_m[0]
    o = f.lv_origin_m[0]
    qlo = (lo[2] - o[2]) / Q
    qhi = (hi[2] - o[2]) / Q
    zl = np.empty(A.MAX_SEG, np.int64)
    zh = np.empty(A.MAX_SEG, np.int64)
    i0 = max(0, int(math.ceil((lo[0] - o[0]) / a)))
    i1 = min(nx - 1, int(math.floor((hi[0] - o[0]) / a)))
    j0 = max(0, int(math.ceil((lo[1] - o[1]) / a)))
    j1 = min(ny - 1, int(math.floor((hi[1] - o[1]) / a)))
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            rz = i * ny + j
            n = first[rz + 1] - first[rz]
            if n == 0:
                continue
            _segs(cnt, seg, link, pseg, pnext, r0 + rz, zl, zh)
            for k in range(n):
                if zl[k] < qhi and zh[k] > qlo:
                    flags[parent[first[rz] + k]] = 1


@njit(i8(FIELD_T, I1, I4_1, i8, types.int8[::1], f8, f8, f8, i8, SCRATCH_T), **OPTS)
def remove_loose(f, first, parent, total, loose, x_unused, y_unused, z_unused, ev, sc):
    """Remove the segments of loose components (``loose[root] = 1``) from every ray of every level: z rays
    of level 0 by their labels; every other ray (x, y families of level 0; all rays of finer levels) by the
    label of the level-0 z segment at the node inside (or nearest to the middle of) each of its segments.
    Brick material and flags are booked. Returns the removed level-0 z length (quanta)."""
    cnt = f.cnt
    seg = f.seg
    sev = f.sev
    link = f.link
    pseg = f.pseg
    psev = f.psev
    pnext = f.pnext
    info = f.info
    nx0 = f.lv_dims[0, 0]
    ny0 = f.lv_dims[0, 1]
    nz0 = f.lv_dims[0, 2]
    r0z = f.lv_ray0[0, 2]
    a0 = f.lv_cell_m[0]
    o0 = f.lv_origin_m[0]
    removed_z = np.int64(0)
    zl = np.empty(A.MAX_SEG, np.int64)
    zh = np.empty(A.MAX_SEG, np.int64)
    nl = info[A.FI_NLEVELS]
    for lvl in range(nl):
        a = f.lv_cell_m[lvl]
        o = f.lv_origin_m[lvl]
        for fam in range(3):
            u = 1 if fam == 0 else 0
            v = 2 if fam < 2 else 1
            nu = f.lv_dims[lvl, u]
            nv = f.lv_dims[lvl, v]
            r0 = f.lv_ray0[lvl, fam]
            for iu in range(nu):
                for iv in range(nv):
                    ray = r0 + iu * nv + iv
                    n = C.load(cnt, seg, sev, link, pseg, psev, pnext, ray, sc.lo, sc.hi, sc.elo, sc.ehi)
                    if n <= 0:
                        continue
                    k = 0
                    npc = 0
                    for s in range(n):
                        drop = False
                        if lvl == 0 and fam == 2:
                            drop = loose[parent[first[iu * ny0 + iv] + s]] == 1
                        else:
                            # node of this segment (middle) in level-0 coordinates
                            mid = 0.5 * (np.int64(sc.lo[s]) + np.int64(sc.hi[s])) * Q + o[fam]
                            pu = o[u] + iu * a
                            pv = o[v] + iv * a
                            if fam == 0:
                                px = mid
                                py = pu
                                pz = pv
                            elif fam == 1:
                                px = pu
                                py = mid
                                pz = pv
                            else:
                                px = pu
                                py = pv
                                pz = mid
                            i0 = int(math.floor((px - o0[0]) / a0 + 0.5))
                            j0 = int(math.floor((py - o0[1]) / a0 + 0.5))
                            i0 = min(max(i0, 0), nx0 - 1)
                            j0 = min(max(j0, 0), ny0 - 1)
                            qz = (pz - o0[2]) / Q
                            rz = i0 * ny0 + j0
                            nzs = first[rz + 1] - first[rz]
                            if nzs > 0:
                                _segs(cnt, seg, link, pseg, pnext, r0z + rz, zl, zh)
                                best = -1
                                bd = 1e300
                                for t in range(nzs):
                                    if qz >= zl[t] and qz < zh[t]:
                                        best = t
                                        break
                                    d = min(abs(qz - zl[t]), abs(qz - zh[t]))
                                    if d < bd:
                                        bd = d
                                        best = t
                                if best >= 0 and (bd <= 1.5 * f.lv_cell_q[0] or (qz >= zl[best] and qz < zh[best])):
                                    drop = loose[parent[first[rz] + best]] == 1
                        if drop:
                            sc.plo[npc] = sc.lo[s]
                            sc.phi[npc] = sc.hi[s]
                            npc += 1
                            if lvl == 0 and fam == 2:
                                removed_z += np.int64(sc.hi[s]) - np.int64(sc.lo[s])
                        else:
                            sc.nlo[k] = sc.lo[s]
                            sc.nhi[k] = sc.hi[s]
                            sc.nelo[k] = sc.elo[s]
                            sc.nehi[k] = sc.ehi[s]
                            k += 1
                    if npc == 0:
                        continue
                    if C.store(cnt, seg, sev, link, pseg, psev, pnext, info, ray, sc.nlo, sc.nhi, sc.nelo, sc.nehi,
                               k) < 0:
                        return -1
                    C.account(f.lv_cell_q, f.lv_bdims, f.lv_brick0, f.bmat, f.bdirty, f.bconn, f.bver, lvl, fam, iu,
                              iv, sc.plo, sc.phi, npc, -1)
    info[A.FI_VERSION] += 1
    return removed_z


@njit(i8(FIELD_T, I1), **OPTS)
def local_connected(f, box):
    """Incremental connectivity of the level-0 column graph after removals. Precondition (the caller's): at the last
    check the material was one component, and since then material was only removed, each change marked in bconn.
    Every changed z column lies in the rectangle R of node columns of the brick columns with a bconn mark (a brick's
    nodes and the next boundary node, which a change marks too); the columns of R and a ring of one column around
    it (unchanged) are labelled. When that labelling has one component and it reaches the ring, the material is
    still one component: every path from a column outside into R crosses the ring, and every column outside R
    joins the ring's component as it did before (its path into R enters through the ring). Returns 1 (still one
    component), 0 (not proven: label the whole field), -1 (no change marked). box = [i0, i1, j0, j1] of R and the
    columns and segments labelled (box[4], box[5])."""
    cnt = f.cnt
    seg = f.seg
    link = f.link
    pseg = f.pseg
    pnext = f.pnext
    bconn = f.bconn
    nx = f.lv_dims[0, 0]
    ny = f.lv_dims[0, 1]
    r0 = f.lv_ray0[0, 2]
    nbx = f.lv_bdims[0, 0]
    nby = f.lv_bdims[0, 1]
    nbz = f.lv_bdims[0, 2]
    b0 = f.lv_brick0[0]
    BR = A.BRICK
    bx0 = nbx
    bx1 = -1
    by0 = nby
    by1 = -1
    for bx in range(nbx):
        for by in range(nby):
            base = b0 + (bx * nby + by) * nbz
            for bz in range(nbz):
                if bconn[base + bz] != 0:
                    if bx < bx0:
                        bx0 = bx
                    if bx > bx1:
                        bx1 = bx
                    if by < by0:
                        by0 = by
                    if by > by1:
                        by1 = by
                    break
    if bx1 < 0:
        return -1
    # R (changed columns) and I = R and its ring of one column, clipped to the lattice
    ri0 = BR * bx0
    ri1 = min(BR * bx1 + BR, nx - 1)
    rj0 = BR * by0
    rj1 = min(BR * by1 + BR, ny - 1)
    ia = max(ri0 - 1, 0)
    ib = min(ri1 + 1, nx - 1)
    ja = max(rj0 - 1, 0)
    jb = min(rj1 + 1, ny - 1)
    box[0] = ri0
    box[1] = ri1
    box[2] = rj0
    box[3] = rj1
    ni = ib - ia + 1
    nj = jb - ja + 1
    first = np.empty(ni * nj + 1, np.int64)
    total = 0
    for i in range(ni):
        for j in range(nj):
            first[i * nj + j] = total
            total += cnt[r0 + (ia + i) * ny + ja + j]
    first[ni * nj] = total
    box[4] = ni * nj
    box[5] = total
    if total == 0:
        return 0
    parent = np.empty(total, np.int32)
    for k in range(total):
        parent[k] = k
    lo_a = np.empty(A.MAX_SEG, np.int64)
    hi_a = np.empty(A.MAX_SEG, np.int64)
    lo_b = np.empty(A.MAX_SEG, np.int64)
    hi_b = np.empty(A.MAX_SEG, np.int64)
    unions = 0
    for i in range(ni):
        for j in range(nj):
            ca = i * nj + j
            na = first[ca + 1] - first[ca]
            if na == 0:
                continue
            ra = r0 + (ia + i) * ny + ja + j
            # one inline segment (the common case) is read in place
            lo1 = np.int64(seg[ra, 0, 0])
            hi1 = np.int64(seg[ra, 0, 1])
            if na > 1:
                _segs(cnt, seg, link, pseg, pnext, ra, lo_a, hi_a)
            for nb_dir in range(2):
                ii = i + 1 if nb_dir == 0 else i
                jj = j if nb_dir == 0 else j + 1
                if ii >= ni or jj >= nj:
                    continue
                cb = ii * nj + jj
                nb = first[cb + 1] - first[cb]
                if nb == 0:
                    continue
                rb = r0 + (ia + ii) * ny + ja + jj
                if na == 1 and nb == 1:
                    if lo1 < np.int64(seg[rb, 0, 1]) and np.int64(seg[rb, 0, 0]) < hi1:
                        x = _find(parent, first[ca])
                        y = _find(parent, first[cb])
                        if x != y:
                            unions += 1
                            if x < y:
                                parent[y] = x
                            else:
                                parent[x] = y
                    continue
                if na == 1:
                    lo_a[0] = lo1
                    hi_a[0] = hi1
                _segs(cnt, seg, link, pseg, pnext, rb, lo_b, hi_b)
                ka = 0
                kb = 0
                while ka < na and kb < nb:
                    if lo_a[ka] < hi_b[kb] and lo_b[kb] < hi_a[ka]:
                        x = _find(parent, first[ca] + ka)
                        y = _find(parent, first[cb] + kb)
                        if x != y:
                            unions += 1
                            if x < y:
                                parent[y] = x
                            else:
                                parent[x] = y
                    if hi_a[ka] < hi_b[kb]:
                        ka += 1
                    else:
                        kb += 1
    # one component (every successful union joins two), and it reaches the ring
    if total - unions != 1:
        return 0
    for i in range(ni):
        gi = ia + i
        for j in range(nj):
            gj = ja + j
            if ri0 <= gi <= ri1 and rj0 <= gj <= rj1:
                continue
            if first[i * nj + j + 1] > first[i * nj + j]:
                return 1
    return 0


KERNELS = (_find, _segs, label_z, component_stats, components_in_box, remove_loose, local_connected)
