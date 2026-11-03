"""Outer-boundary gating (architecture 5.4, docs/models/removal.md "Gating"): per station the T0 angle
intervals where the element circle lies in material, the contact geometry of a contact, and the bounding
box of the engaged sector that limits the subtraction.

Per station circle c + r (cos phi x0 + sin phi y0):

1. reject when no brick with material meets the circle's box (one cell of slack for shared nodes);
2. classify samples spaced ``sample_m`` (default one lattice cell) along the circle (k_core.classify: the
   family most aligned with the tool axis first, certain far from boundaries, the most normal family's
   interpolated boundary near them); a sample whose stencil lies in bricks without material is outside
   without classification (exactly what classify would return);
3. locate every state change between neighbouring samples by root finding on D_f of the family with the
   smallest endpoint spread at the bracket (Illinois regula falsi, Dowell & Jarratt 1971), started at the
   previous interval boundary when one lies in the bracket (warm start), to ``tol_m`` along the arc; a
   bracket no family brackets falls back to bisection of the classification;
4. write intervals [lo, hi] with lo in [0, 2 pi), lo < hi <= lo + 2 pi (a full circle is [0, 2 pi]).

Error: planar boundaries exact to tol_m (and the 10 nm quantum); curved boundaries a^2 kappa / (8 cos^2
theta) <= a^2 kappa / 4 normal to the boundary (k_sd), divided by sin(crossing angle) along the circle;
features narrower than the sample spacing along the circle are found once they reach it (the
undetected depth of a new engagement is below s^2 / (8 r)).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from . import abi as A
from . import k_core as C
from .nbt import (ACCUM_T, B1, ENGAGED_T, F1, F2, FIELD_T, I1, I2, I4_1, I4_3, INLINE, OPTS, RAW, b1, f8, i8, void)

TWO_PI = 2.0 * math.pi
MAX_SAMPLES = 4096


@njit(b1(I2, I1, I2, F1, F2, i8, f8, f8, f8, f8, f8, f8), **INLINE)
def _box_material(bmat, lv_brick0, lv_bdims, lv_cell_m, lv_origin, lvl, x0, y0, z0, x1, y1, z1):
    """Any brick of ``lvl`` meeting the box (grown by one cell) holds material."""
    a = lv_cell_m[lvl]
    k0x = (int(math.floor((x0 - lv_origin[lvl, 0]) / a)) - 1) // A.BRICK
    k1x = (int(math.ceil((x1 - lv_origin[lvl, 0]) / a)) + 1) // A.BRICK
    k0y = (int(math.floor((y0 - lv_origin[lvl, 1]) / a)) - 1) // A.BRICK
    k1y = (int(math.ceil((y1 - lv_origin[lvl, 1]) / a)) + 1) // A.BRICK
    k0z = (int(math.floor((z0 - lv_origin[lvl, 2]) / a)) - 1) // A.BRICK
    k1z = (int(math.ceil((z1 - lv_origin[lvl, 2]) / a)) + 1) // A.BRICK
    k0x = max(k0x, 0)
    k0y = max(k0y, 0)
    k0z = max(k0z, 0)
    k1x = min(k1x, lv_bdims[lvl, 0] - 1)
    k1y = min(k1y, lv_bdims[lvl, 1] - 1)
    k1z = min(k1z, lv_bdims[lvl, 2] - 1)
    for bx in range(k0x, k1x + 1):
        for by in range(k0y, k1y + 1):
            for bz in range(k0z, k1z + 1):
                idx = C.bindex(lv_brick0, lv_bdims, lvl, bx, by, bz)
                if bmat[idx, 0] > 0 or bmat[idx, 1] > 0 or bmat[idx, 2] > 0:
                    return True
    return False


@njit(b1(I2, I1, I2, I2, F1, F2, i8, f8, f8, f8), **INLINE)
def _empty_near(bmat, lv_brick0, lv_bdims, lv_dims, lv_cell_m, lv_origin, lvl, x, y, z):
    """No ray of any family holds material in the bricks owning the point's stencil (the nodes of its cell
    across, its cell along each family), or the point lies beyond the outermost nodes (the margin layers
    are kept empty, field.make_field): then every family's four rays are outside at the point and
    k_core.classify would answer outside. Usually one brick is read."""
    a = lv_cell_m[lvl]
    ix = int(math.floor((x - lv_origin[lvl, 0]) / a))
    iy = int(math.floor((y - lv_origin[lvl, 1]) / a))
    iz = int(math.floor((z - lv_origin[lvl, 2]) / a))
    if (ix < 0 or iy < 0 or iz < 0 or ix >= lv_dims[lvl, 0] - 1 or iy >= lv_dims[lvl, 1] - 1
            or iz >= lv_dims[lvl, 2] - 1):
        return True
    nbx = lv_bdims[lvl, 0]
    nby = lv_bdims[lvl, 1]
    nbz = lv_bdims[lvl, 2]
    bx0 = min(max(ix, 0) // A.BRICK, nbx - 1)
    bx1 = min(max(ix + 1, 0) // A.BRICK, nbx - 1)
    by0 = min(max(iy, 0) // A.BRICK, nby - 1)
    by1 = min(max(iy + 1, 0) // A.BRICK, nby - 1)
    bz0 = min(max(iz, 0) // A.BRICK, nbz - 1)
    bz1 = min(max(iz + 1, 0) // A.BRICK, nbz - 1)
    for bx in range(bx0, bx1 + 1):
        for by in range(by0, by1 + 1):
            for bz in range(bz0, bz1 + 1):
                idx = C.bindex(lv_brick0, lv_bdims, lvl, bx, by, bz)
                if bmat[idx, 0] > 0 or bmat[idx, 1] > 0 or bmat[idx, 2] > 0:
                    return False
    return True


@njit(b1(FIELD_T, i8, f8, f8, f8, f8, f8, f8), **OPTS)
def box_has_material(f, lvl, x0, y0, z0, x1, y1, z1):
    return _box_material(f.bmat, f.lv_brick0, f.lv_bdims, f.lv_cell_m, f.lv_origin_m, lvl, x0, y0, z0, x1, y1, z1)


@njit(f8(F1, F2, I2, I1, I2, I4_1, I4_3, I4_1, I4_3, I4_1, i8, F1, F1, F1, f8, f8, f8, b1, f8, f8, F1, F1, F1),
      **RAW)
def _refine(lv_cell_m, lv_origin, lv_dims, lv_cell_q, lv_ray0, cnt, seg, link, pseg, pnext, lvl, c, x0, y0, r, pa,
            pb, a_inside, guess, tol, p, work, stats):
    """Angle in [pa, pb] where the material state changes (``a_inside``: the state at pa). Compiled once
    (called a few times per station: the call boundary costs less than inlining it everywhere)."""
    best = -1
    best_s = 1e300
    ga = 0.0
    gb = 0.0
    for fam in range(3):
        sp = 0.0
        va = 0.0
        vb = 0.0
        ok = True
        for q in range(3):
            phi = 0.5 * (pa + pb) if q == 0 else (pa if q == 1 else pb)
            cq = math.cos(phi)
            sq = math.sin(phi)
            v = C.sd(lv_cell_m, lv_origin, lv_dims, lv_cell_q, lv_ray0, cnt, seg, link, pseg, pnext, lvl, fam,
                     c[0] + r * (cq * x0[0] + sq * y0[0]), c[1] + r * (cq * x0[1] + sq * y0[1]),
                     c[2] + r * (cq * x0[2] + sq * y0[2]), work)
            if q == 0:
                if work[1] != 1.0:
                    ok = False
                    break
                sp = work[0]
            elif q == 1:
                va = v
            else:
                vb = v
        if not ok:
            continue
        if ((va > 0.0) == a_inside) and ((vb > 0.0) != a_inside) and sp < best_s:
            best = fam
            best_s = sp
            ga = va
            gb = vb
    lo = pa
    hi = pb
    if best < 0:
        for _ in range(64):
            if (hi - lo) * r <= tol:
                break
            mid = 0.5 * (lo + hi)
            cm = math.cos(mid)
            sm = math.sin(mid)
            ins, d, fm = C.classify(lv_cell_m, lv_origin, lv_dims, lv_cell_q, lv_ray0, cnt, seg, link, pseg, pnext,
                                    lvl, c[0] + r * (cm * x0[0] + sm * y0[0]), c[1] + r * (cm * x0[1] + sm * y0[1]),
                                    c[2] + r * (cm * x0[2] + sm * y0[2]), 2, work)
            stats[1] += 1.0
            if ins == a_inside:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    glo = ga
    ghi = gb
    side = 0
    x = guess if (guess > pa and guess < pb) else lo - glo * (hi - lo) / (ghi - glo)
    for _ in range(100):
        cx = math.cos(x)
        sx = math.sin(x)
        gx = C.sd(lv_cell_m, lv_origin, lv_dims, lv_cell_q, lv_ray0, cnt, seg, link, pseg, pnext, lvl, best,
                  c[0] + r * (cx * x0[0] + sx * y0[0]), c[1] + r * (cx * x0[1] + sx * y0[1]),
                  c[2] + r * (cx * x0[2] + sx * y0[2]), work)
        stats[1] += 1.0
        if gx == 0.0:
            return x
        if (gx > 0.0) == (glo > 0.0):
            lo = x
            glo = gx
            if side == -1:
                ghi *= 0.5
            side = -1
        else:
            hi = x
            ghi = gx
            if side == 1:
                glo *= 0.5
            side = 1
        if (hi - lo) * r <= tol:
            break
        x = lo - glo * (hi - lo) / (ghi - glo)
        if not (x > lo and x < hi):
            x = 0.5 * (lo + hi)
    return 0.5 * (lo + hi)


@njit(void(FIELD_T, F1, F1, F1, F1, F1, F1, B1, ENGAGED_T, i8, F1, F1, F1, F1, F1, B1, F1, F1, F1, B1), **RAW)
def _engaged(f, tip, x0, y0, w, st_z, st_r, active, out, tick, params, stats, work, p, c, states, prev_lo, prev_hi,
             bd_phi, bd_in):
    """engaged_intervals with its work arrays (states, bd_phi, bd_in: MAX_SAMPLES + 1; prev_lo, prev_hi: the
    intervals per station of ``out``). Allocation-free, reference counting off."""
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
    lv_lo = f.lv_box_lo_m
    lv_hi = f.lv_box_hi_m
    bmat = f.bmat
    nl = f.info[A.FI_NLEVELS]
    o_lo = out.lo_rad
    o_hi = out.hi_rad
    o_cnt = out.count
    nst = st_z.shape[0]
    nmax = o_lo.shape[1]
    tol = params[1] if params[1] > 0.0 else 1e-8
    margin = params[2]
    fam0 = 0
    if abs(w[1]) > abs(w[fam0]):
        fam0 = 1
    if abs(w[2]) > abs(w[fam0]):
        fam0 = 2
    for k in range(nst):
        if active[k] == 0:
            continue
        stats[3] += 1.0
        pc = o_cnt[k]
        for j in range(pc):
            prev_lo[j] = o_lo[k, j]
            prev_hi[j] = o_hi[k, j]
        o_cnt[k] = 0
        for i in range(3):
            c[i] = tip[i] + st_z[k] * w[i]
        r = st_r[k] + margin
        lvl = C.level_at(lv_lo, lv_hi, lv_cell_m, nl, c[0], c[1], c[2])
        if lvl > 0:
            ok = True
            for i in range(3):
                e = r * math.sqrt(max(0.0, 1.0 - w[i] * w[i]))
                if c[i] - e < lv_lo[lvl, i] or c[i] + e > lv_hi[lvl, i]:
                    ok = False
            if not ok:
                lvl = 0
        a = lv_cell_m[lvl]
        point = r <= 1e-9
        if point:
            r = 0.0
        ex = r * math.sqrt(max(0.0, 1.0 - w[0] * w[0]))
        ey = r * math.sqrt(max(0.0, 1.0 - w[1] * w[1]))
        ez = r * math.sqrt(max(0.0, 1.0 - w[2] * w[2]))
        if not _box_material(bmat, lv_brick0, lv_bdims, lv_cell_m, lv_origin, lvl, c[0] - ex, c[1] - ey, c[2] - ez,
                             c[0] + ex, c[1] + ey, c[2] + ez):
            continue
        spacing = params[0] if params[0] > 0.0 else a
        n = int(math.ceil(TWO_PI * r / spacing))
        if n < 16:
            n = 16
        if n > MAX_SAMPLES:
            n = MAX_SAMPLES
        if point:
            n = 1
        dphi = TWO_PI / n
        cd = math.cos(dphi)
        sdd = math.sin(dphi)
        cp = 1.0
        sp = 0.0
        n_in = 0
        for i in range(n):
            if i % 64 == 0:
                cp = math.cos(i * dphi)
                sp = math.sin(i * dphi)
            px = c[0] + r * (cp * x0[0] + sp * y0[0])
            py = c[1] + r * (cp * x0[1] + sp * y0[1])
            pz = c[2] + r * (cp * x0[2] + sp * y0[2])
            if _empty_near(bmat, lv_brick0, lv_bdims, lv_dims, lv_cell_m, lv_origin, lvl, px, py, pz):
                ins = False
            else:
                ins, d, fm = C.classify(lv_cell_m, lv_origin, lv_dims, lv_cell_q, lv_ray0, cnt, seg, link, pseg,
                                        pnext, lvl, px, py, pz, fam0, work)
            states[i] = 1 if ins else 0
            if ins:
                n_in += 1
            cn = cp * cd - sp * sdd
            sp = sp * cd + cp * sdd
            cp = cn
        stats[0] += n
        if n_in == 0:
            continue
        if n_in == n or point:
            o_cnt[k] = 1
            o_lo[k, 0] = 0.0
            o_hi[k, 0] = TWO_PI
            continue
        states[n] = states[0]
        nbd = 0
        for i in range(n):
            if states[i] == states[i + 1]:
                continue
            pa = i * dphi
            pb = (i + 1) * dphi
            entering = states[i + 1] == 1
            guess = -1.0
            for j in range(pc):
                cand = prev_lo[j] if entering else prev_hi[j]
                cand = cand - TWO_PI * math.floor(cand / TWO_PI)
                if cand > pa and cand < pb:
                    guess = cand
                    break
            phi = _refine(lv_cell_m, lv_origin, lv_dims, lv_cell_q, lv_ray0, cnt, seg, link, pseg, pnext, lvl, c, x0,
                          y0, r, pa, pb, states[i] == 1, guess, tol, p, work, stats)
            bd_phi[nbd] = phi
            bd_in[nbd] = 1 if entering else 0
            nbd += 1
        first = -1
        for j in range(nbd):
            if bd_in[j] == 1:
                first = j
                break
        if first < 0:
            continue
        cnt_k = 0
        j = first
        visited = 0
        while visited < nbd:
            if bd_in[j] == 1:
                lo = bd_phi[j]
                jj = (j + 1) % nbd
                hi = bd_phi[jj]
                if hi <= lo:
                    hi += TWO_PI
                if cnt_k < nmax:
                    o_lo[k, cnt_k] = lo
                    o_hi[k, cnt_k] = hi
                    cnt_k += 1
                else:
                    # merge: the smallest gap between consecutive intervals closes (deterministic)
                    stats[2] += 1.0
                    best = 0
                    best_gap = 1e300
                    for q in range(cnt_k - 1):
                        gap = o_lo[k, q + 1] - o_hi[k, q]
                        if gap < best_gap:
                            best_gap = gap
                            best = q
                    last_gap = lo - o_hi[k, cnt_k - 1]
                    if last_gap <= best_gap:
                        o_hi[k, cnt_k - 1] = hi
                    else:
                        o_hi[k, best] = o_hi[k, best + 1]
                        for q in range(best + 1, cnt_k - 1):
                            o_lo[k, q] = o_lo[k, q + 1]
                            o_hi[k, q] = o_hi[k, q + 1]
                        o_lo[k, cnt_k - 1] = lo
                        o_hi[k, cnt_k - 1] = hi
                j = (j + 2) % nbd
                visited += 2
            else:
                j = (j + 1) % nbd
                visited += 1
        o_cnt[k] = cnt_k
    out.tick[0] = tick


@njit(void(FIELD_T, F1, F1, F1, F1, F1, F1, B1, ENGAGED_T, i8, F1, F1), **OPTS)
def engaged_intervals(f, tip, x0, y0, w, st_z, st_r, active, out, tick, params, stats):
    """Engaged intervals of the active stations (``out`` updated in place; its previous contents are the
    warm start). params = [sample spacing (m, <= 0: one lattice cell), tolerance along the arc (m),
    radial gate margin (m)]. stats += [samples, refinement evaluations, interval overflows, stations].
    The work arrays are made here once per call; the station loop (_engaged) runs without reference counting."""
    nmax = out.lo_rad.shape[1]
    _engaged(f, tip, x0, y0, w, st_z, st_r, active, out, tick, params, stats, np.empty(4), np.empty(3), np.empty(3),
             np.empty(MAX_SAMPLES + 1, np.int8), np.empty(max(nmax, 1)), np.empty(max(nmax, 1)),
             np.empty(MAX_SAMPLES + 1), np.empty(MAX_SAMPLES + 1, np.int8))


@njit(f8(f8, f8, f8), **INLINE)
def _sin_range_max(lo, hi, shift):
    """max of sin(phi - shift) over [lo, hi]."""
    best = max(math.sin(lo - shift), math.sin(hi - shift))
    target = shift + 0.5 * math.pi
    k = math.ceil((lo - target) / TWO_PI)
    if target + k * TWO_PI <= hi:
        best = 1.0
    return best


@njit(void(ENGAGED_T, F1, F1, F1, f8, f8, ACCUM_T), **RAW)
def contact_geometry(eng, st_z, st_r, st_len, feed_x, feed_y, acc):
    """Fold one gating result into the contact accumulator: largest engaged angle and arc, radial engagement
    ae across the feed direction (feed_x, feed_y: unit feed direction in the T0 xy plane; zero vector: the
    T0 x axis) at the largest engaged radius, and the engaged edge length (sum of the element lengths of
    the engaged stations)."""
    nst = st_z.shape[0]
    shift = math.atan2(feed_y, feed_x) if (feed_x != 0.0 or feed_y != 0.0) else 0.0
    rmax = -1.0
    kbest = -1
    length = 0.0
    fv = acc.fvals
    for k in range(nst):
        cnt = eng.count[k]
        if cnt == 0:
            continue
        ang = 0.0
        for j in range(cnt):
            ang += eng.hi_rad[k, j] - eng.lo_rad[k, j]
        if ang > fv[A.FA_ANGLE_MAX]:
            fv[A.FA_ANGLE_MAX] = ang
        if ang * st_r[k] > fv[A.FA_ARC_MAX]:
            fv[A.FA_ARC_MAX] = ang * st_r[k]
        length += st_len[k]
        if st_r[k] > rmax:
            rmax = st_r[k]
            kbest = k
    if length > fv[A.FA_ENGAGED_LEN]:
        fv[A.FA_ENGAGED_LEN] = length
    if kbest < 0:
        return
    smax = -2.0
    smin = 2.0
    for j in range(eng.count[kbest]):
        lo = eng.lo_rad[kbest, j]
        hi = eng.hi_rad[kbest, j]
        top = _sin_range_max(lo, hi, shift)
        bot = -_sin_range_max(lo, hi, shift + math.pi)
        if top > smax:
            smax = top
        if bot < smin:
            smin = bot
    ae = rmax * (smax - smin)
    if ae > fv[A.FA_AE_MAX]:
        fv[A.FA_AE_MAX] = ae


@njit(i8(ENGAGED_T, F1, F1, F1, F1, F1, f8, f8, f8, f8, F1, F1), **RAW)
def sector_box(eng, tip, x0, y0, w, st_z, r_env, z_lo, z_hi, margin, box_lo, box_hi):
    """Axis-aligned box (S) of the engaged sector of the whole envelope: the union of the engaged angle
    intervals of all stations, as a wedge of radius ``r_env`` (the envelope's largest radius) from the axis,
    swept along the tool axis from ``z_lo`` to ``z_hi`` (the envelope's axial range), grown by ``margin``.
    Material outside the box but inside the envelope lies in the trailing half, which the previous tooth
    pass removed (architecture 5.5). Returns the number of engaged stations (0: lo > hi, nothing to do)."""
    for i in range(3):
        box_lo[i] = 1e300
        box_hi[i] = -1e300
    umin = 0.0
    umax = 0.0
    vmin = 0.0
    vmax = 0.0
    n = 0
    for k in range(st_z.shape[0]):
        cnt = eng.count[k]
        if cnt == 0:
            continue
        n += 1
        for j in range(cnt):
            lo = eng.lo_rad[k, j]
            hi = eng.hi_rad[k, j]
            # extremes of cos and sin over [lo, hi]
            cmax = max(math.cos(lo), math.cos(hi))
            cmin = min(math.cos(lo), math.cos(hi))
            smax = max(math.sin(lo), math.sin(hi))
            smin = min(math.sin(lo), math.sin(hi))
            for q in range(4):
                ang = 0.5 * math.pi * q
                kk = math.ceil((lo - ang) / TWO_PI)
                if ang + kk * TWO_PI <= hi:
                    if q == 0:
                        cmax = 1.0
                    elif q == 1:
                        smax = 1.0
                    elif q == 2:
                        cmin = -1.0
                    else:
                        smin = -1.0
            umin = min(umin, r_env * cmin)
            umax = max(umax, r_env * cmax)
            vmin = min(vmin, r_env * smin)
            vmax = max(vmax, r_env * smax)
    if n == 0:
        return 0
    for cu in range(2):
        u = umin if cu == 0 else umax
        for cv in range(2):
            v = vmin if cv == 0 else vmax
            for cz in range(2):
                z = z_lo if cz == 0 else z_hi
                for i in range(3):
                    x = tip[i] + u * x0[i] + v * y0[i] + z * w[i]
                    if x < box_lo[i]:
                        box_lo[i] = x
                    if x > box_hi[i]:
                        box_hi[i] = x
    for i in range(3):
        box_lo[i] -= margin
        box_hi[i] += margin
    return n


KERNELS = (box_has_material, _refine, _engaged, engaged_intervals, contact_geometry, sector_box)
INLINED = (_box_material, _empty_near, _sin_range_max)
