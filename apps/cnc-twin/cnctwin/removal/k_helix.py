"""Swept-profile subtraction of taps along the relative helical path (architecture 5.4, docs/models/removal.md
"Taps").

Between two poses of one call the tool tip moves linearly in proportion to the spindle angle theta (so the
pitch, and any feed / speed mismatch of a floating or rigid tapping cycle, comes from the motion). Land l
lies in the meridian half-plane of right-handed azimuth theta + h phase_l. A point x of the work at T0
azimuth alpha (frame of the first pose) is passed by land l at theta* = alpha - h phase_l (+ 2 pi k) inside
the call's angle range (every such passage when the range exceeds a turn); there the tool has advanced to
p(theta*) and the point is inside the swept region when (z_T*, r*) = ((x - p) . w, |(x - p) x w|) lies in
one of the land's polygons. Along a ray the inside
function (largest outward edge distance, <= 0 inside a convex polygon) is sampled at most every
min(a / 2, pitch / 16) and every sign change refined by bisection to 1e-10 m; only the polygons whose z range
holds z_T* are tested. Rays without material in the tap's bounding box are skipped after one scan.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, types

from . import abi as A
from . import k_core as C
from .k_ray import iv_subtract
from .nbt import ACCUM_T, EVENTS_T, F1, FIELD_T, HELIX_T, OPTS, SCRATCH_T, TARGET_T, f8, i8

Q = A.QUANTUM_M
TWO_PI = 2.0 * math.pi


@njit(f8(HELIX_T, f8, f8, f8, f8, f8, f8, F1, F1, F1, F1, F1, f8, f8, f8), **OPTS)
def helix_inside(hx, x, y, z, th_a, th_b, unused, pa, pb, w, x0, y0, hand, zlo, zhi):
    """Largest of the per-land inside values at a point (<= 0: inside the swept region of some land)."""
    best = 1e300
    nl = hx.land_phase.shape[0]
    span = th_b - th_a
    if span == 0.0:
        return best
    lo_t = th_a if th_a < th_b else th_b
    hi_t = th_b if th_a < th_b else th_a
    # azimuth in T0 of the first pose
    dx = x - pa[0]
    dy = y - pa[1]
    dz = z - pa[2]
    ax_ = dx * x0[0] + dy * x0[1] + dz * x0[2]
    ay_ = dx * y0[0] + dy * y0[1] + dz * y0[2]
    alpha = math.atan2(ay_, ax_)
    for land in range(nl):
        base = alpha - hand * hx.land_phase[land]
        # every passage theta* = base + 2 pi k of this land inside [lo_t, hi_t]
        k = math.ceil((lo_t - base) / TWO_PI)
        ts = base + k * TWO_PI
        while ts <= hi_t:
            lam = (ts - th_a) / span
            px = pa[0] + lam * (pb[0] - pa[0])
            py = pa[1] + lam * (pb[1] - pa[1])
            pz = pa[2] + lam * (pb[2] - pa[2])
            ts += TWO_PI
            qx = x - px
            qy = y - py
            qz = z - pz
            zt = qx * w[0] + qy * w[1] + qz * w[2]
            if zt < zlo or zt > zhi:
                continue
            ex = qx - zt * w[0]
            ey = qy - zt * w[1]
            ez = qz - zt * w[2]
            rr = math.sqrt(ex * ex + ey * ey + ez * ez)
            for j in range(hx.land_first[land], hx.land_first[land + 1]):
                if zt < hx.zr_box[j, 0] or zt > hx.zr_box[j, 1] or rr > hx.zr_box[j, 3]:
                    continue
                npt = hx.n[j]
                dmax = -1e300
                for e in range(npt):
                    az = hx.poly[j, e, 0]
                    ar = hx.poly[j, e, 1]
                    bz = hx.poly[j, (e + 1) % npt, 0]
                    br = hx.poly[j, (e + 1) % npt, 1]
                    tz = bz - az
                    tr = br - ar
                    ln = math.sqrt(tz * tz + tr * tr)
                    if ln <= 0.0:
                        continue
                    d = ((zt - az) * tr - (rr - ar) * tz) / ln
                    if d > dmax:
                        dmax = d
                if dmax < best:
                    best = dmax
    return best


@njit(i8(FIELD_T, HELIX_T, F1, F1, F1, F1, F1, f8, f8, F1, F1, i8, i8, i8, TARGET_T, ACCUM_T, EVENTS_T,
         SCRATCH_T), **OPTS)
def helix_subtract(f, hx, pa, pb, w, x0, y0, th_a, th_b, box_lo, box_hi, ev_id, tick, contact, tgt, acc, ev, sc):
    """Subtract the regions the tap's lands sweep while the spindle turns from th_a to th_b and the tip moves
    from pa to pb (S). Books removed lengths in ``acc`` (level 0 z family for the volume); returns 0, 1 on a
    pool overflow (idempotent: grow and repeat)."""
    cnt = f.cnt
    seg = f.seg
    sev = f.sev
    link = f.link
    pseg = f.pseg
    psev = f.psev
    pnext = f.pnext
    info = f.info
    if hx.land_phase.shape[0] == 0 or hx.n.shape[0] == 0:
        return 0
    hand = hx.info[0]
    zlo = hx.info[1]
    zhi = hx.info[2]
    rhi = hx.info[3]
    # box of the swept region at both poses
    lo = np.empty(3)
    hi = np.empty(3)
    for i in range(3):
        e = rhi * math.sqrt(max(0.0, 1.0 - w[i] * w[i]))
        vals0 = pa[i] + zlo * w[i]
        vals1 = pa[i] + zhi * w[i]
        vals2 = pb[i] + zlo * w[i]
        vals3 = pb[i] + zhi * w[i]
        lo[i] = min(min(vals0, vals1), min(vals2, vals3)) - e
        hi[i] = max(max(vals0, vals1), max(vals2, vals3)) + e
        lo[i] = max(lo[i], box_lo[i])
        hi[i] = min(hi[i], box_hi[i])
        if hi[i] < lo[i]:
            return 0
    pitch = zhi - zlo
    for j in range(hx.n.shape[0]):
        hz = hx.zr_box[j, 1] - hx.zr_box[j, 0]
        if hz > 0.0 and hz < pitch:
            pitch = hz
    status = 0
    any_removed = False
    p = np.empty(3)
    nlv = info[A.FI_NLEVELS]
    for lvl in range(nlv):
        a = f.lv_cell_m[lvl]
        step = 0.5 * a
        if pitch / 16.0 < step:
            step = pitch / 16.0
        for fam in range(3):
            u = 1 if fam == 0 else 0
            v = 2 if fam < 2 else 1
            ou = f.lv_origin_m[lvl, u]
            ov = f.lv_origin_m[lvl, v]
            of = f.lv_origin_m[lvl, fam]
            nu = f.lv_dims[lvl, u]
            nv = f.lv_dims[lvl, v]
            iu0 = max(0, int(math.ceil((lo[u] - ou) / a)))
            iu1 = min(nu - 1, int(math.floor((hi[u] - ou) / a)))
            iv0 = max(0, int(math.ceil((lo[v] - ov) / a)))
            iv1 = min(nv - 1, int(math.floor((hi[v] - ov) / a)))
            lq = (f.lv_dims[lvl, fam] - 1) * f.lv_cell_q[lvl]
            t_lo = max(0.0, lo[fam] - of)
            t_hi = min(lq * Q, hi[fam] - of)
            if t_hi <= t_lo:
                continue
            q_lo = np.int64(np.floor(t_lo / Q))
            q_hi = np.int64(np.ceil(t_hi / Q))
            r0 = f.lv_ray0[lvl, fam]
            for iu in range(iu0, iu1 + 1):
                for iv in range(iv0, iv1 + 1):
                    ray = r0 + iu * nv + iv
                    if cnt[ray] == 0:
                        continue
                    mat, tch = C.overlap(cnt, seg, link, pseg, pnext, ray, q_lo, q_hi)
                    if mat == 0:
                        continue
                    p[u] = ou + iu * a
                    p[v] = ov + iv * a
                    ns = int(math.ceil((t_hi - t_lo) / step))
                    if ns < 1:
                        ns = 1
                    m = 0
                    prev_in = False
                    start = 0.0
                    tp = t_lo
                    for k in range(ns + 1):
                        t = t_lo + (t_hi - t_lo) * k / ns
                        p[fam] = of + t
                        g = helix_inside(hx, p[0], p[1], p[2], th_a, th_b, 0.0, pa, pb, w, x0, y0, hand, zlo, zhi)
                        ins = g <= 0.0
                        if k > 0 and ins != prev_in:
                            a_ = tp
                            b_ = t
                            for _ in range(48):
                                if b_ - a_ <= 1e-10:
                                    break
                                mid = 0.5 * (a_ + b_)
                                p[fam] = of + mid
                                gm = helix_inside(hx, p[0], p[1], p[2], th_a, th_b, 0.0, pa, pb, w, x0, y0, hand,
                                                  zlo, zhi)
                                if (gm <= 0.0) == prev_in:
                                    a_ = mid
                                else:
                                    b_ = mid
                            tb = 0.5 * (a_ + b_)
                            if ins:
                                start = tb
                            else:
                                qa = np.int64(np.floor(start / Q + 0.5))
                                qb = np.int64(np.floor(tb / Q + 0.5))
                                if qb > qa and m < sc.rlo.shape[0]:
                                    sc.rlo[m] = qa
                                    sc.rhi[m] = qb
                                    m += 1
                        elif k == 0 and ins:
                            start = t
                        prev_in = ins
                        tp = t
                    if prev_in:
                        qa = np.int64(np.floor(start / Q + 0.5))
                        qb = np.int64(np.floor(t_hi / Q + 0.5))
                        if qb > qa and m < sc.rlo.shape[0]:
                            sc.rlo[m] = qa
                            sc.rhi[m] = qb
                            m += 1
                    if m == 0:
                        continue
                    n = C.load(cnt, seg, sev, link, pseg, psev, pnext, ray, sc.lo, sc.hi, sc.elo, sc.ehi)
                    if n < 0:
                        info[A.FI_SCRATCH_OVF] = 1
                        continue
                    nn, npc, removed = iv_subtract(n, sc.lo, sc.hi, sc.elo, sc.ehi, m, sc.rlo, sc.rhi, ev_id, sc.nlo,
                                                   sc.nhi, sc.nelo, sc.nehi, sc.plo, sc.phi)
                    if nn < 0 or removed == 0:
                        continue
                    if C.store(cnt, seg, sev, link, pseg, psev, pnext, info, ray, sc.nlo, sc.nhi, sc.nelo, sc.nehi,
                               nn) < 0:
                        status = 1
                        continue
                    C.account(f.lv_cell_q, f.lv_bdims, f.lv_brick0, f.bmat, f.bdirty, f.bconn, f.bver, lvl, fam, iu,
                              iv, sc.plo, sc.phi, npc, -1)
                    acc.removed_q[lvl, fam] += removed
                    acc.ivals[A.IA_RAYS_CHANGED] += 1
                    any_removed = True
                    if lvl == 0 and fam == 2:
                        if nn != n:
                            info[A.FI_TOPO] += 1
                        for k in range(npc):
                            vol = (sc.phi[k] - sc.plo[k]) * Q * a * a
                            p[fam] = of + 0.5 * (sc.plo[k] + sc.phi[k]) * Q
                            acc.fvals[A.FA_CX] += vol * p[0]
                            acc.fvals[A.FA_CY] += vol * p[1]
                            acc.fvals[A.FA_CZ] += vol * p[2]
                            acc.fvals[A.FA_W] += vol
    acc.ivals[A.IA_SUBTRACTIONS] += 1
    if any_removed:
        info[A.FI_VERSION] += 1
        acc.ivals[A.IA_IN_CUT_CALLS] += 1
        acc.ivals[A.IA_LAST_TICK] = tick
        if acc.ivals[A.IA_FIRST_TICK] < 0:
            acc.ivals[A.IA_FIRST_TICK] = tick
    if status == 1:
        acc.ivals[A.IA_OVERFLOW] += 1
    return status


KERNELS = (helix_inside, helix_subtract)
