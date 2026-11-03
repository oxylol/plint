"""Marching cubes on the tri-dexel lattice, per brick (docs/models/removal.md "Meshing").

A node's state is the majority of the three rays through it (x, y and z family agree except where a
boundary passes within one quantum of the node or a feature is thinner than a cell in a family's
transverse directions). A cell edge along axis f lies on a ray of family f, so its vertex is that ray's
exact endpoint inside the edge (the endpoint of the right kind: a segment end when the lower node is
inside, a start when it is outside); only where the majority state and the edge's own ray disagree is the
nearest endpoint in the edge (or the edge middle) used. Faces come from the face-consistent case table
(cnctwin.removal.mctable), so the mesh is closed; the nodes on the lattice boundary count as air, which
closes refine-level meshes at their box. Every vertex carries the surface event id of its endpoint and a
global edge id (level node index * 3 + axis) that welds bricks exactly. An optional displacement grid d(X)
(trilinear) moves the vertices to X + d(X) (as-cut, cold, unload geometry, architecture 4.1).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, types

from . import abi as A
from . import k_core as C
from .nbt import B3, F1, F2, F4, FIELD_T, I1, I4_1, I4_2, I4_3, INLINE, OPTS, f8, i8, void

Q = A.QUANTUM_M
I3_64 = types.int64[:, :, ::1]
I4_4 = types.int32[:, :, :, ::1]


@njit(types.UniTuple(i8, 3)(I4_1, I4_3, I4_3, I4_1, I4_3, I4_3, I4_1, i8, i8, i8, i8), **INLINE)
def _crossing(cnt, seg, sev, link, pseg, psev, pnext, ray, q0, q1, lower_inside):
    """(endpoint in [q0, q1], its surface event id, found) of the kind the edge needs: the first segment end
    in (q0, q1] when the lower node is inside, the first segment start in [q0, q1) otherwise; when none
    exists, the endpoint nearest the edge middle inside [q0, q1]."""
    n = np.int64(cnt[ray])
    b = np.int64(link[ray])
    s = 0
    near = np.int64(-1)
    near_ev = np.int64(-1)
    near_d = 1e300
    mid = 0.5 * (q0 + q1)
    K = A.K_INLINE
    for k in range(n):
        if k < K:
            lo = np.int64(seg[ray, k, 0])
            hi = np.int64(seg[ray, k, 1])
            elo = np.int64(sev[ray, k, 0])
            ehi = np.int64(sev[ray, k, 1])
        else:
            if s == K:
                b = np.int64(pnext[b])
                s = 0
            lo = np.int64(pseg[b, s, 0])
            hi = np.int64(pseg[b, s, 1])
            elo = np.int64(psev[b, s, 0])
            ehi = np.int64(psev[b, s, 1])
            s += 1
        if lo > q1:
            break
        if lower_inside == 1:
            if hi > q0 and hi <= q1:
                return hi, ehi, np.int64(1)
        else:
            if lo >= q0 and lo < q1:
                return lo, elo, np.int64(1)
        for e in range(2):
            x = lo if e == 0 else hi
            ev = elo if e == 0 else ehi
            if x >= q0 and x <= q1 and abs(x - mid) < near_d:
                near_d = abs(x - mid)
                near = x
                near_ev = ev
    if near >= 0:
        return near, near_ev, np.int64(1)
    return np.int64((q0 + q1) // 2), np.int64(A.SEV_NONE), np.int64(0)


@njit(types.UniTuple(i8, 3)(FIELD_T, i8, i8, i8, i8), **OPTS)
def ray_crossing(f, ray, q0, q1, lower_inside):
    """(endpoint in [q0, q1], its surface event id, found) of the kind an edge needs (see _crossing)."""
    return _crossing(f.cnt, f.seg, f.sev, f.link, f.pseg, f.psev, f.pnext, ray, q0, q1, lower_inside)


@njit(void(F1, f8, F4, f8, f8, f8, F1), **OPTS)
def displace(origin, cell, d, x, y, z, out):
    """out = (x, y, z) + d(x, y, z), trilinear in the grid (clamped at its border); no grid: unchanged."""
    out[0] = x
    out[1] = y
    out[2] = z
    nx = d.shape[0]
    ny = d.shape[1]
    nz = d.shape[2]
    if nx == 0 or ny == 0 or nz == 0 or cell <= 0.0:
        return
    fx = (x - origin[0]) / cell
    fy = (y - origin[1]) / cell
    fz = (z - origin[2]) / cell
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
                    acc += wx * wy * wz * d[ii, jj, kk, c]
        out[c] += acc


@njit(i8(FIELD_T, i8, i8, i8, i8, I1, I3_64, F2, I4_2, I4_1, I1, I1, F1, f8, F4, B3, I4_4), **OPTS)
def mesh_brick(f, lvl, bx, by, bz, tri_count, tri_edges, out_v, out_f, out_sev, out_eid, counts, d_origin, d_cell,
               disp, votes, vid):
    """Mesh one brick of one level into out_* (appending at counts[0] vertices, counts[1] faces). Returns 0,
    or 1 when the output arrays were too small (counts[2] = 1, the brick's output is incomplete: call
    again with larger arrays and counts reset)."""
    K = A.K_INLINE
    cnt = f.cnt
    seg = f.seg
    sev = f.sev
    link = f.link
    pseg = f.pseg
    psev = f.psev
    pnext = f.pnext
    lv_ray0 = f.lv_ray0
    lv_dims = f.lv_dims
    cq = f.lv_cell_q[lvl]
    a = f.lv_cell_m[lvl]
    o = f.lv_origin_m[lvl]
    dims0 = f.lv_dims[lvl, 0]
    dims1 = f.lv_dims[lvl, 1]
    dims2 = f.lv_dims[lvl, 2]
    n0 = np.empty(3, np.int64)
    ni = np.empty(3, np.int64)
    n0[0] = bx * A.BRICK
    n0[1] = by * A.BRICK
    n0[2] = bz * A.BRICK
    for i in range(3):
        top = n0[i] + A.BRICK
        if top > f.lv_dims[lvl, i] - 1:
            top = f.lv_dims[lvl, i] - 1
        ni[i] = top - n0[i] + 1
    node0 = 0
    for l in range(lvl):
        node0 += f.lv_dims[l, 0] * f.lv_dims[l, 1] * f.lv_dims[l, 2]
    for i in range(ni[0]):
        for j in range(ni[1]):
            for k in range(ni[2]):
                votes[i, j, k] = 0
    for fam in range(3):
        u = 1 if fam == 0 else 0
        v = 2 if fam < 2 else 1
        for lu in range(ni[u]):
            for lv in range(ni[v]):
                ray = C.rindex(lv_ray0, lv_dims, lvl, fam, n0[u] + lu, n0[v] + lv)
                n = np.int64(cnt[ray])
                if n == 0:
                    continue
                b = np.int64(link[ray])
                s = 0
                for kk in range(n):
                    if kk < K:
                        lo = np.int64(seg[ray, kk, 0])
                        hi = np.int64(seg[ray, kk, 1])
                    else:
                        if s == K:
                            b = np.int64(pnext[b])
                            s = 0
                        lo = np.int64(pseg[b, s, 0])
                        hi = np.int64(pseg[b, s, 1])
                        s += 1
                    k0 = (lo + cq - 1) // cq
                    k1 = (hi + cq - 1) // cq - 1
                    if k0 < n0[fam]:
                        k0 = n0[fam]
                    if k1 > n0[fam] + ni[fam] - 1:
                        k1 = n0[fam] + ni[fam] - 1
                    for kn in range(k0, k1 + 1):
                        lw = kn - n0[fam]
                        if fam == 0:
                            votes[lw, lu, lv] += 1
                        elif fam == 1:
                            votes[lu, lw, lv] += 1
                        else:
                            votes[lu, lv, lw] += 1
    # majority, the lattice boundary layer is air
    for i in range(ni[0]):
        gi = n0[0] + i
        for j in range(ni[1]):
            gj = n0[1] + j
            for k in range(ni[2]):
                gk = n0[2] + k
                st = 1 if votes[i, j, k] >= 2 else 0
                if gi == 0 or gj == 0 or gk == 0 or gi == dims0 - 1 or gj == dims1 - 1 or gk == dims2 - 1:
                    st = 0
                votes[i, j, k] = st
    for ax in range(3):
        for i in range(ni[0]):
            for j in range(ni[1]):
                for k in range(ni[2]):
                    vid[ax, i, j, k] = -1
    nv = counts[0]
    nf = counts[1]
    pos = np.empty(3)
    tmp = np.empty(3)
    ev_ids = np.empty(3, np.int64)
    for i in range(ni[0] - 1):
        for j in range(ni[1] - 1):
            for k in range(ni[2] - 1):
                case = 0
                for c in range(8):
                    if votes[i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1)] == 1:
                        case |= 1 << c
                if case == 0 or case == 255:
                    continue
                nt = tri_count[case]
                for t in range(nt):
                    for q in range(3):
                        e = tri_edges[case, t, q]
                        ax = e // 4
                        jj = e % 4
                        # lower node of edge e (local)
                        li = i
                        lj = j
                        lk = k
                        if ax == 0:
                            lj += jj & 1
                            lk += (jj >> 1) & 1
                        elif ax == 1:
                            li += jj & 1
                            lk += (jj >> 1) & 1
                        else:
                            li += jj & 1
                            lj += (jj >> 1) & 1
                        idx = vid[ax, li, lj, lk]
                        if idx < 0:
                            if nv >= out_v.shape[0]:
                                counts[0] = nv
                                counts[1] = nf
                                counts[2] = 1
                                return 1
                            gi = n0[0] + li
                            gj = n0[1] + lj
                            gk = n0[2] + lk
                            pos[0] = o[0] + gi * a
                            pos[1] = o[1] + gj * a
                            pos[2] = o[2] + gk * a
                            if ax == 0:
                                ray = C.rindex(lv_ray0, lv_dims, lvl, 0, gj, gk)
                                kq0 = gi * cq
                            elif ax == 1:
                                ray = C.rindex(lv_ray0, lv_dims, lvl, 1, gi, gk)
                                kq0 = gj * cq
                            else:
                                ray = C.rindex(lv_ray0, lv_dims, lvl, 2, gi, gj)
                                kq0 = gk * cq
                            low_in = votes[li, lj, lk]
                            ep, sev_e, found = _crossing(cnt, seg, sev, link, pseg, psev, pnext, ray, kq0, kq0 + cq,
                                                         np.int64(low_in))
                            pos[ax] = o[ax] + ep * Q
                            displace(d_origin, d_cell, disp, pos[0], pos[1], pos[2], tmp)
                            out_v[nv, 0] = tmp[0]
                            out_v[nv, 1] = tmp[1]
                            out_v[nv, 2] = tmp[2]
                            out_sev[nv] = sev_e
                            out_eid[nv] = ((node0 + (gi * dims1 + gj) * dims2 + gk) * 3 + ax)
                            vid[ax, li, lj, lk] = nv
                            idx = nv
                            nv += 1
                        ev_ids[q] = idx
                    if nf >= out_f.shape[0]:
                        counts[0] = nv
                        counts[1] = nf
                        counts[2] = 1
                        return 1
                    out_f[nf, 0] = ev_ids[0]
                    out_f[nf, 1] = ev_ids[1]
                    out_f[nf, 2] = ev_ids[2]
                    nf += 1
    counts[0] = nv
    counts[1] = nf
    return 0


KERNELS = (ray_crossing, displace, mesh_brick)
