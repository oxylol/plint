"""Collision kernels of removal (docs/models/removal.md "Collision").

* :func:`collide_envelope` - material inside the non-cutting parts of a tool envelope (holder, shank,
  neck, spindle nose, insert body) at a pose: nothing is removed, an EV_COLLISION row is pushed; only the
  box of the non-cutting pieces is searched (rejected by the brick occupancy while the tool cuts);
* :func:`env_ray` / :func:`envelope_ray` - the envelope intervals of one ray, compiled (off the hot path);
* :func:`mesh_field_overlap` - a triangle mesh (machine part, fixture) at a pose against the field:
  the mesh's interior intervals on the field's rays inside the overlap box (watertight rasterisation,
  k_stock) intersected with the material;
* swept checks between two poses sub-sample the motion so that no point of the moving solid travels more
  than ``step_m`` between samples (a fast rapid cannot tunnel through a feature thicker than step_m).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, types

from ..tooling.abi import PART_EDGE
from . import abi as A
from . import k_core as C
from .k_envelope import _aabb, _env_ray, _event, node_range
from .k_stock import rasterize, resolve_hits
from .nbt import ACCUM_T, EVENTS_T, F1, F2, FIELD_T, I4_1, OPTS, PROFILE_T, RAW, SCRATCH_T, f8, i8, void

Q = A.QUANTUM_M


@njit(i8(F1, F1, types.int32[::1], F1, F1, F1, F1, types.int32[::1], f8, f8, f8, i8, F1, F2, F1, F1, f8, f8,
         F2, I4_1, i8, F1), **OPTS)
def env_ray(pz0, pz1, pkind, pc0, pc1, pc2, pc3, ppart, ox, oy, oz, fam, w, slab_c, slab_z, slab_dr, t_lo, t_hi, out,
            tags, merge, work):
    """Compiled (not inlined) _env_ray, for the callers off the subtraction's hot path."""
    return _env_ray(pz0, pz1, pkind, pc0, pc1, pc2, pc3, ppart, ox, oy, oz, fam, w, slab_c, slab_z, slab_dr, t_lo,
                    t_hi, out, tags, merge, work)


@njit(i8(PROFILE_T, f8, f8, f8, i8, F1, F2, F1, F1, f8, f8, F2, I4_1, i8), **OPTS)
def envelope_ray(prof, ox, oy, oz, fam, w, slab_c, slab_z, slab_dr, t_lo, t_hi, out, tags, merge):
    """Intervals of the ray o + t e_fam (t in [t_lo, t_hi], metres) inside the slabbed envelope, sorted and
    merged when ``merge``; returns their number (-1: the interval buffer is full)."""
    work = np.empty(16)
    return env_ray(prof.z0_m, prof.z1_m, prof.kind, prof.c0, prof.c1, prof.c2, prof.c3, prof.part, ox, oy, oz, fam,
                   w, slab_c, slab_z, slab_dr, t_lo, t_hi, out, tags, merge, work)


# --------------------------------------------------------------------------- collision of the non-cutting parts
@njit(i8(FIELD_T, PROFILE_T, PROFILE_T, F1, F2, F1, i8, i8, f8, ACCUM_T, EVENTS_T, SCRATCH_T), **RAW)
def collide_envelope(f, prof_all, prof_cut, axis, tip_c, unused, tick, contact, tol, acc, ev, sc):
    """Material inside the whole envelope (holder, shank, neck, spindle nose, insert body) but outside the
    cutting envelope, at the pose (tip_c [1, 3], unit axis), level 0. Only the axial range and radius of
    the non-cutting pieces of ``prof_all`` (part != PART_EDGE) are searched: their box is rejected at once
    when no brick in it holds material (the usual case while cutting), else every ray of the box with
    material there is intersected with both envelopes. Nothing is removed. Pushes one EV_COLLISION row
    (largest overlap along a ray, its point and the part there; volume as the mean of the three families'
    estimates) when the overlap exceeds ``tol`` (m). Returns the number of rays in contact.
    Allocation-free (work arrays from the scratch), reference counting off."""
    cnt = f.cnt
    seg = f.seg
    link = f.link
    pseg = f.pseg
    pnext = f.pnext
    lv_cell_m = f.lv_cell_m
    lv_origin = f.lv_origin_m
    lv_dims = f.lv_dims
    lv_cell_q = f.lv_cell_q
    lv_ray0 = f.lv_ray0
    lv_bdims = f.lv_bdims
    lv_brick0 = f.lv_brick0
    bmat = f.bmat
    az0 = prof_all.z0_m
    az1 = prof_all.z1_m
    akind = prof_all.kind
    ac0 = prof_all.c0
    ac1 = prof_all.c1
    ac2 = prof_all.c2
    ac3 = prof_all.c3
    apart = prof_all.part
    armax = prof_all.rmax_m
    cz0 = prof_cut.z0_m
    cz1 = prof_cut.z1_m
    ckind = prof_cut.kind
    cc0 = prof_cut.c0
    cc1 = prof_cut.c1
    cc2 = prof_cut.c2
    cc3 = prof_cut.c3
    cpart = prof_cut.part
    half = sc.ivt.shape[0] // 2
    ivt = sc.ivt[0:half]
    ivp = sc.ivp[0:half]
    cut_iv = sc.ivt[half:2 * half]
    cut_tag = sc.ivp[half:2 * half]
    fw = sc.fw
    if az0.shape[0] == 0:
        return 0
    # the axial range and radius of the non-cutting pieces
    zlo_nc = 1e300
    zhi_nc = -1e300
    r_nc = 0.0
    for k in range(az0.shape[0]):
        if apart[k] != PART_EDGE:
            if az0[k] < zlo_nc:
                zlo_nc = az0[k]
            if az1[k] > zhi_nc:
                zhi_nc = az1[k]
            if armax[k] > r_nc:
                r_nc = armax[k]
    if zhi_nc <= zlo_nc:
        return 0
    slab_z = fw[0:2]
    slab_z[0] = zlo_nc
    slab_z[1] = zhi_nc
    slab_dr = fw[2:3]
    slab_dr[0] = 0.0
    lo = fw[3:6]
    hi = fw[6:9]
    p = fw[9:12]
    work = fw[A.FW_WORK:A.FW_WORK + 16]
    bb0 = sc.iw[0:3]
    bb1 = sc.iw[3:6]
    _aabb(axis, slab_dr, tip_c, slab_z, slab_dr, zlo_nc, zhi_nc, r_nc, 0.0, lo, hi)
    lvl = 0
    a = lv_cell_m[lvl]
    a2 = a * a
    best = 0.0
    bx_ = 0.0
    by_ = 0.0
    bz_ = 0.0
    bpart = -1
    vol = 0.0
    n_contact = 0
    for i in range(3):
        n0 = int(math.floor((lo[i] - lv_origin[lvl, i]) / a)) - 1
        n1 = int(math.ceil((hi[i] - lv_origin[lvl, i]) / a)) + 1
        k0 = n0 // A.BRICK
        k1 = n1 // A.BRICK
        if k0 < 0:
            k0 = 0
        if k1 > lv_bdims[lvl, i] - 1:
            k1 = lv_bdims[lvl, i] - 1
        if k1 < k0:
            return 0
        bb0[i] = k0
        bb1[i] = k1
    occupied = False
    for bxx in range(bb0[0], bb1[0] + 1):
        for byy in range(bb0[1], bb1[1] + 1):
            for bzz in range(bb0[2], bb1[2] + 1):
                idx = C.bindex(lv_brick0, lv_bdims, lvl, bxx, byy, bzz)
                if bmat[idx, 0] > 0 or bmat[idx, 1] > 0 or bmat[idx, 2] > 0:
                    occupied = True
    if not occupied:
        return 0
    for fam in range(3):
        u = 1 if fam == 0 else 0
        v = 2 if fam < 2 else 1
        ou = lv_origin[lvl, u]
        ov = lv_origin[lvl, v]
        of = lv_origin[lvl, fam]
        nu = lv_dims[lvl, u]
        nv = lv_dims[lvl, v]
        iu0, iu1 = node_range((lo[u] - ou) / a, (hi[u] - ou) / a, 0.0, nu)
        iv0, iv1 = node_range((lo[v] - ov) / a, (hi[v] - ov) / a, 0.0, nv)
        lq = (lv_dims[lvl, fam] - 1) * lv_cell_q[lvl]
        t_lo = lo[fam] - of
        t_hi = hi[fam] - of
        if t_lo < 0.0:
            t_lo = 0.0
        if t_hi > lq * Q:
            t_hi = lq * Q
        if t_hi <= t_lo:
            continue
        q_lo = np.int64(np.floor(t_lo / Q))
        q_hi = np.int64(np.ceil(t_hi / Q))
        r0 = lv_ray0[lvl, fam]
        for iu in range(iu0, iu1 + 1):
            for iv in range(iv0, iv1 + 1):
                ray = r0 + iu * nv + iv
                if cnt[ray] == 0:
                    continue
                p[fam] = of
                p[u] = ou + iu * a
                p[v] = ov + iv * a
                mat, touched = C.overlap(cnt, seg, link, pseg, pnext, ray, q_lo, q_hi)
                if mat == 0:
                    continue
                # the whole envelope's intervals, then the cutting envelope's (one inlined call site)
                na = 0
                nc = 0
                for which in range(2):
                    if which == 0:
                        n_iv = _env_ray(az0, az1, akind, ac0, ac1, ac2, ac3, apart, p[0], p[1], p[2], fam, axis,
                                        tip_c, slab_z, slab_dr, t_lo, t_hi, ivt, ivp, 0, work)
                        na = n_iv
                        if na <= 0:
                            break
                    else:
                        n_iv = _env_ray(cz0, cz1, ckind, cc0, cc1, cc2, cc3, cpart, p[0], p[1], p[2], fam, axis,
                                        tip_c, slab_z, slab_dr, t_lo, t_hi, cut_iv, cut_tag, 1, work)
                        nc = n_iv if n_iv > 0 else 0
                if na <= 0:
                    continue
                ray_hit = False
                for i in range(na):
                    # interval minus the cutting intervals
                    cur = ivt[i, 0]
                    end = ivt[i, 1]
                    for k in range(nc + 1):
                        if k < nc:
                            ca = cut_iv[k, 0]
                            cb = cut_iv[k, 1]
                        else:
                            ca = end
                            cb = end
                        if cb <= cur:
                            continue
                        seg_end = ca if ca < end else end
                        if seg_end > cur:
                            qa = np.int64(np.floor(cur / Q + 0.5))
                            qb = np.int64(np.floor(seg_end / Q + 0.5))
                            ovl, tch = C.overlap(cnt, seg, link, pseg, pnext, ray, qa, qb)
                            if ovl > 0:
                                ovm = ovl * Q
                                vol += ovm * a2 / 3.0
                                if ovm > tol:
                                    ray_hit = True
                                if ovm > best:
                                    best = ovm
                                    p[fam] = of + 0.5 * (cur + seg_end)
                                    bx_ = p[0]
                                    by_ = p[1]
                                    bz_ = p[2]
                                    bpart = ivp[i]
                        if cb > cur:
                            cur = cb
                        if cur >= end:
                            break
                if ray_hit:
                    n_contact += 1
    if best > tol:
        _event(ev.rec, ev.n, A.EV_COLLISION, tick, contact, bx_, by_, bz_, best, vol, bpart, -1, 0, -1)
    return n_contact


# --------------------------------------------------------------------------- mesh against the field
@njit(i8(FIELD_T, F2, types.int32[:, ::1], F1, F1, f8, F1, SCRATCH_T), **OPTS)
def mesh_field_overlap(f, V, Fc, box_lo, box_hi, tol, out, sc):
    """A closed triangle mesh (vertices in S) against the material of level 0, over the rays inside the box
    (the mesh box clipped to the field). For every family the mesh's interior intervals on each ray
    (k_stock watertight rasterisation, winding > 0) are intersected with the ray's material. out = [overlap
    volume (m^3, mean of the three families' estimates), largest overlap along a ray (m), x, y, z of its
    middle, rays in overlap above tol, open rays]. Returns 1 when the largest overlap exceeds tol."""
    a = f.lv_cell_m[0]
    o = f.lv_origin_m[0]
    out[0] = 0.0
    out[1] = 0.0
    out[2] = 0.0
    out[3] = 0.0
    out[4] = 0.0
    out[5] = 0.0
    out[6] = 0.0
    stats_f = np.zeros(5)
    stats_i = np.zeros(3, np.int64)
    dummy_seg = np.zeros((1, 2), np.int32)
    dummy_n = np.zeros(1, np.int64)
    for fam in range(3):
        u = 1 if fam == 0 else 0
        v = 2 if fam < 2 else 1
        iu0 = int(math.ceil((box_lo[u] - o[u]) / a))
        iu1 = int(math.floor((box_hi[u] - o[u]) / a))
        iv0 = int(math.ceil((box_lo[v] - o[v]) / a))
        iv1 = int(math.floor((box_hi[v] - o[v]) / a))
        iu0 = max(iu0, 0)
        iv0 = max(iv0, 0)
        iu1 = min(iu1, f.lv_dims[0, u] - 1)
        iv1 = min(iv1, f.lv_dims[0, v] - 1)
        if iu1 < iu0 or iv1 < iv0:
            continue
        nw = (iu1 - iu0 + 1) * (iv1 - iv0 + 1)
        counts = np.zeros(nw, np.int64)
        offsets = np.zeros(nw + 1, np.int64)
        tdum = np.zeros(1)
        ddum = np.zeros(1, np.int8)
        total = rasterize(f, 0, fam, V, Fc, iu0, iu1, iv0, iv1, counts, offsets, tdum, ddum, 0)
        if total == 0:
            continue
        for k in range(nw):
            offsets[k + 1] = offsets[k] + counts[k]
            counts[k] = 0
        ts = np.zeros(total)
        ds = np.zeros(total, np.int8)
        rasterize(f, 0, fam, V, Fc, iu0, iu1, iv0, iv1, counts, offsets, ts, ds, 1)
        before = stats_f[0]
        resolve_hits(f, 0, fam, iu0, iu1, iv0, iv1, offsets, ts, ds, 2, A.SEV_NONE, dummy_seg, dummy_n, stats_i,
                     stats_f, sc)
        out[0] += (stats_f[0] - before) * a * a / 3.0
    out[1] = stats_f[1]
    out[2] = stats_f[2]
    out[3] = stats_f[3]
    out[4] = stats_f[4]
    out[5] = stats_i[1]
    out[6] = stats_i[0]
    return 1 if stats_f[1] > tol else 0


@njit(void(F2, F1, F1, F2, F1, f8, F2), **OPTS)
def pose_lerp(R0, t0, unused, R1, t1, lam, out):
    """out (3 x 4) = the rigid pose a fraction lam from (R0, t0) to (R1, t1): translation linear, rotation
    along the geodesic R0 exp(lam log(R0^T R1)) (Rodrigues)."""
    # relative rotation
    Rr = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            acc = 0.0
            for k in range(3):
                acc += R0[k, i] * R1[k, j]
            Rr[i, j] = acc
    c = 0.5 * (Rr[0, 0] + Rr[1, 1] + Rr[2, 2] - 1.0)
    c = min(1.0, max(-1.0, c))
    ang = math.acos(c)
    ax = Rr[2, 1] - Rr[1, 2]
    ay = Rr[0, 2] - Rr[2, 0]
    az = Rr[1, 0] - Rr[0, 1]
    nrm = math.sqrt(ax * ax + ay * ay + az * az)
    Rl = np.eye(3)
    if ang > 1e-15 and nrm > 1e-15:
        ax /= nrm
        ay /= nrm
        az /= nrm
        th = lam * ang
        ct = math.cos(th)
        st = math.sin(th)
        vt = 1.0 - ct
        Rl[0, 0] = ct + ax * ax * vt
        Rl[0, 1] = ax * ay * vt - az * st
        Rl[0, 2] = ax * az * vt + ay * st
        Rl[1, 0] = ay * ax * vt + az * st
        Rl[1, 1] = ct + ay * ay * vt
        Rl[1, 2] = ay * az * vt - ax * st
        Rl[2, 0] = az * ax * vt - ay * st
        Rl[2, 1] = az * ay * vt + ax * st
        Rl[2, 2] = ct + az * az * vt
    for i in range(3):
        for j in range(3):
            acc = 0.0
            for k in range(3):
                acc += R0[i, k] * Rl[k, j]
            out[i, j] = acc
        out[i, 3] = t0[i] + lam * (t1[i] - t0[i])


@njit(f8(FIELD_T, F2, types.int32[:, ::1], F2, F1, F2, F1, f8, f8, F1, SCRATCH_T), **OPTS)
def swept_mesh_field(f, Vl, Fc, R0, t0, R1, t1, step, tol, out, sc):
    """A mesh (vertices in its own frame) moving from pose (R0, t0) to (R1, t1) in S: sub-sampled so no
    vertex moves more than ``step`` between samples (a feature thicker than step cannot be tunnelled), each
    sample tested with mesh_field_overlap. Returns the first fraction in contact (-1: none); out as
    mesh_field_overlap at that sample."""
    nvx = Vl.shape[0]
    maxd = 0.0
    p0 = np.empty(3)
    p1 = np.empty(3)
    for i in range(nvx):
        for r in range(3):
            p0[r] = R0[r, 0] * Vl[i, 0] + R0[r, 1] * Vl[i, 1] + R0[r, 2] * Vl[i, 2] + t0[r]
            p1[r] = R1[r, 0] * Vl[i, 0] + R1[r, 1] * Vl[i, 1] + R1[r, 2] * Vl[i, 2] + t1[r]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        dz = p1[2] - p0[2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if d > maxd:
            maxd = d
    n = int(math.ceil(maxd / step)) if step > 0.0 else 1
    if n < 1:
        n = 1
    P = np.empty((3, 4))
    Vs = np.empty((nvx, 3))
    lo = np.empty(3)
    hi = np.empty(3)
    for s in range(n + 1):
        lam = s / n
        pose_lerp(R0, t0, t0, R1, t1, lam, P)
        for i in range(nvx):
            for r in range(3):
                Vs[i, r] = P[r, 0] * Vl[i, 0] + P[r, 1] * Vl[i, 1] + P[r, 2] * Vl[i, 2] + P[r, 3]
        for r in range(3):
            lo[r] = 1e300
            hi[r] = -1e300
        for i in range(nvx):
            for r in range(3):
                if Vs[i, r] < lo[r]:
                    lo[r] = Vs[i, r]
                if Vs[i, r] > hi[r]:
                    hi[r] = Vs[i, r]
        if mesh_field_overlap(f, Vs, Fc, lo, hi, tol, out, sc) == 1:
            return lam
    return -1.0


KERNELS = (env_ray, envelope_ray, collide_envelope, mesh_field_overlap, pose_lerp, swept_mesh_field)
