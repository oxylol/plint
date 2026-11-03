"""Field queries (docs/models/removal.md "Queries").

* :func:`ray_cast` - first entry into (or exit from) material along a ray or segment: brick-level 3D-DDA
  (Amanatides & Woo 1987) skips bricks without material, samples every half cell inside the others and
  refines the crossing on the interpolated boundary D_f (k_sd) by Illinois regula falsi to ``tol``.
* :func:`surface_point` / :func:`refill_points` - the surface point along a direction (stale surface-memory
  bins, architecture 5.4): ahead when the point is in air, behind when it is in material.
* :func:`surface_height` - highest (sign +1) or lowest (-1) material surface along an S axis (chip piles).
* :func:`fill_fraction` - material fraction of the cells of a coarse grid from the level-0 z family
  (column model, exact for it: every z ray is an a x a column).
* :func:`wall_thickness` - material thickness behind a surface point along -n (thin-wall model).
* :func:`broad_phase` - occupied bricks inside the box of an envelope at a pose (contact at L1).
* :func:`sweep_contact` - first contact fraction of a probe stylus (ball + shank) moving linearly (G31).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, types

from . import abi as A
from . import k_core as C
from .k_ray import brick_index, ray_index
from .k_sd import classify, dexel_sd, gradient, level_at
from .nbt import B1, F1, F2, F3, FIELD_T, I1, I2, I4_1, I4_3, INLINE, OPTS, PROFILE_T, b1, f8, i8, void

Q = A.QUANTUM_M


@njit(b1(I2, I1, I2, i8, i8, i8, i8), **INLINE)
def _occ(bmat, lv_brick0, lv_bdims, lvl, bx, by, bz):
    idx = C.bindex(lv_brick0, lv_bdims, lvl, bx, by, bz)
    return bmat[idx, 0] > 0 or bmat[idx, 1] > 0 or bmat[idx, 2] > 0


@njit(b1(FIELD_T, i8, i8, i8, i8), **OPTS)
def _brick_occupied(f, lvl, bx, by, bz):
    return _occ(f.bmat, f.lv_brick0, f.lv_bdims, lvl, bx, by, bz)


@njit(f8(F1, F2, I2, I1, I2, I4_1, I4_3, I4_1, I4_3, I4_1, F2, F2, i8, f8, f8, f8, f8, f8, f8, f8, f8, b1, f8, F1),
      **OPTS)
def _refine_t(lv_cell_m, lv_origin, lv_dims, lv_cell_q, lv_ray0, cnt, seg, link, pseg, pnext, lv_lo, lv_hi, nl, ox, oy,
              oz, dx, dy, dz, ta, tb, a_inside, tol, work):
    """Crossing on [ta, tb] of a ray whose material state is ``a_inside`` at ta and the opposite at tb."""
    xm = ox + 0.5 * (ta + tb) * dx
    ym = oy + 0.5 * (ta + tb) * dy
    zm = oz + 0.5 * (ta + tb) * dz
    lvl = C.level_at(lv_lo, lv_hi, lv_cell_m, nl, xm, ym, zm)
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
            tq = 0.5 * (ta + tb) if q == 0 else (ta if q == 1 else tb)
            v = C.sd(lv_cell_m, lv_origin, lv_dims, lv_cell_q, lv_ray0, cnt, seg, link, pseg, pnext, lvl, fam,
                     ox + tq * dx, oy + tq * dy, oz + tq * dz, work)
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
    lo = ta
    hi = tb
    if best < 0:
        for _ in range(64):
            if hi - lo <= tol:
                break
            mid = 0.5 * (lo + hi)
            ins, d, fm = C.classify(lv_cell_m, lv_origin, lv_dims, lv_cell_q, lv_ray0, cnt, seg, link, pseg, pnext,
                                    lvl, ox + mid * dx, oy + mid * dy, oz + mid * dz, 2, work)
            if ins == a_inside:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    glo = ga
    ghi = gb
    side = 0
    x = lo - glo * (hi - lo) / (ghi - glo)
    for _ in range(100):
        gx = C.sd(lv_cell_m, lv_origin, lv_dims, lv_cell_q, lv_ray0, cnt, seg, link, pseg, pnext, lvl, best,
                  ox + x * dx, oy + x * dy, oz + x * dz, work)
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
        if hi - lo <= tol:
            break
        x = lo - glo * (hi - lo) / (ghi - glo)
        if not (x > lo and x < hi):
            x = 0.5 * (lo + hi)
    return 0.5 * (lo + hi)


@njit(b1(F1, F2, I2, I1, I2, I4_1, I4_3, I4_1, I4_3, I4_1, F2, F2, i8, f8, f8, f8, i8, F1), **OPTS)
def _state(lv_cell_m, lv_origin, lv_dims, lv_cell_q, lv_ray0, cnt, seg, link, pseg, pnext, lv_lo, lv_hi, nl, x, y, z,
           fam0, work):
    """Material state at a point (finest level); work[3] returns the deciding family."""
    lvl = C.level_at(lv_lo, lv_hi, lv_cell_m, nl, x, y, z)
    ins, d, fm = C.classify(lv_cell_m, lv_origin, lv_dims, lv_cell_q, lv_ray0, cnt, seg, link, pseg, pnext, lvl, x, y,
                            z, fam0, work)
    work[3] = fm
    return ins


@njit(i8(FIELD_T, f8, f8, f8, f8, f8, f8, f8, i8, f8, F1), **OPTS)
def ray_cast(f, ox, oy, oz, dx, dy, dz, t_max, mode, tol, out):
    """First crossing along o + t d (|d| = 1), 0 <= t <= t_max: mode 0 entry into material, mode 1 exit.
    out = [t, x, y, z, nx, ny, nz (outward surface normal), started (1: the start is already in the state
    sought)]. Returns 1 on a hit, 0 otherwise."""
    work = np.empty(4)
    nrm = np.empty(3)
    lv_cell_m = f.lv_cell_m
    lv_origin = f.lv_origin_m
    lv_dims = f.lv_dims
    lv_cell_q = f.lv_cell_q
    lv_ray0 = f.lv_ray0
    lv_lo = f.lv_box_lo_m
    lv_hi = f.lv_box_hi_m
    lv_bdims = f.lv_bdims
    lv_brick0 = f.lv_brick0
    cnt = f.cnt
    seg = f.seg
    link = f.link
    pseg = f.pseg
    pnext = f.pnext
    bmat = f.bmat
    nl = f.info[A.FI_NLEVELS]
    a = lv_cell_m[0]
    o = lv_origin[0]
    if tol <= 0.0:
        tol = 1e-8
    fam0 = 0
    if abs(dy) > abs(dx):
        fam0 = 1
    if abs(dz) > abs(dx) and abs(dz) > abs(dy):
        fam0 = 2
    want_inside = mode == 0
    ins0 = _state(lv_cell_m, lv_origin, lv_dims, lv_cell_q, lv_ray0, cnt, seg, link, pseg, pnext, lv_lo, lv_hi, nl,
                  ox, oy, oz, fam0, work)
    out[7] = 0.0
    if ins0 == want_inside:
        out[0] = 0.0
        out[1] = ox
        out[2] = oy
        out[3] = oz
        out[4] = 0.0
        out[5] = 0.0
        out[6] = 0.0
        out[7] = 1.0
        return 1
    # clip to the level-0 lattice box
    t0 = 0.0
    t1 = t_max
    dd = np.empty(3)
    dd[0] = dx
    dd[1] = dy
    dd[2] = dz
    oo = np.empty(3)
    oo[0] = ox
    oo[1] = oy
    oo[2] = oz
    for i in range(3):
        lo_b = o[i]
        hi_b = o[i] + (lv_dims[0, i] - 1) * a
        if dd[i] == 0.0:
            if oo[i] < lo_b or oo[i] > hi_b:
                return 0
        else:
            ta = (lo_b - oo[i]) / dd[i]
            tb = (hi_b - oo[i]) / dd[i]
            if ta > tb:
                ta, tb = tb, ta
            if ta > t0:
                t0 = ta
            if tb < t1:
                t1 = tb
    if t1 < t0:
        return 0
    # brick DDA on level 0
    bs = A.BRICK * a
    eps = 1e-12
    pos = np.empty(3)
    for i in range(3):
        pos[i] = oo[i] + (t0 + eps) * dd[i]
    bi = np.empty(3, np.int64)
    step = np.empty(3, np.int64)
    tmax_ = np.empty(3)
    tdel = np.empty(3)
    for i in range(3):
        b = int(math.floor((pos[i] - o[i]) / bs))
        if b < 0:
            b = 0
        if b > lv_bdims[0, i] - 1:
            b = lv_bdims[0, i] - 1
        bi[i] = b
        if dd[i] > 0.0:
            step[i] = 1
            tmax_[i] = (o[i] + (b + 1) * bs - oo[i]) / dd[i]
            tdel[i] = bs / dd[i]
        elif dd[i] < 0.0:
            step[i] = -1
            tmax_[i] = (o[i] + b * bs - oo[i]) / dd[i]
            tdel[i] = -bs / dd[i]
        else:
            step[i] = 0
            tmax_[i] = 1e300
            tdel[i] = 1e300
    h = 0.5 * a
    t_last = t0
    tb0 = t0
    while True:
        tb1 = min(tmax_[0], min(tmax_[1], tmax_[2]))
        if tb1 > t1:
            tb1 = t1
        occ = _occ(bmat, lv_brick0, lv_bdims, 0, bi[0], bi[1], bi[2])
        # a brick's last cell layer depends on the next brick's rays: sample it unless both are empty
        if not occ:
            for i in range(3):
                nb = bi[i] + step[i]
                if step[i] != 0 and nb >= 0 and nb < lv_bdims[0, i]:
                    if i == 0 and _occ(bmat, lv_brick0, lv_bdims, 0, nb, bi[1], bi[2]):
                        occ = True
                    elif i == 1 and _occ(bmat, lv_brick0, lv_bdims, 0, bi[0], nb, bi[2]):
                        occ = True
                    elif i == 2 and _occ(bmat, lv_brick0, lv_bdims, 0, bi[0], bi[1], nb):
                        occ = True
        if occ or not want_inside:
            t = tb0 if tb0 > t_last else t_last
            while True:
                tn = t + h
                if tn > tb1:
                    tn = tb1
                x = ox + tn * dx
                y = oy + tn * dy
                z = oz + tn * dz
                fm = fam0
                if occ:
                    ins = _state(lv_cell_m, lv_origin, lv_dims, lv_cell_q, lv_ray0, cnt, seg, link, pseg, pnext,
                                 lv_lo, lv_hi, nl, x, y, z, fam0, work)
                    fm = np.int64(work[3])
                else:
                    ins = False
                if ins == want_inside:
                    th = _refine_t(lv_cell_m, lv_origin, lv_dims, lv_cell_q, lv_ray0, cnt, seg, link, pseg, pnext,
                                   lv_lo, lv_hi, nl, ox, oy, oz, dx, dy, dz, t_last, tn, not want_inside, tol, work)
                    out[0] = th
                    out[1] = ox + th * dx
                    out[2] = oy + th * dy
                    out[3] = oz + th * dz
                    gradient(f, out[1], out[2], out[3], fm, 0.25 * a, nrm)
                    out[4] = -nrm[0]
                    out[5] = -nrm[1]
                    out[6] = -nrm[2]
                    return 1
                t_last = tn
                t = tn
                if tn >= tb1:
                    break
        else:
            t_last = tb1
        if tb1 >= t1:
            return 0
        # next brick
        ax = 0
        if tmax_[1] < tmax_[ax]:
            ax = 1
        if tmax_[2] < tmax_[ax]:
            ax = 2
        bi[ax] += step[ax]
        if bi[ax] < 0 or bi[ax] >= lv_bdims[0, ax]:
            return 0
        tb0 = tmax_[ax]
        tmax_[ax] += tdel[ax]
    return 0


@njit(i8(FIELD_T, f8, f8, f8, f8, f8, f8, f8, F1), **OPTS)
def surface_point(f, px, py, pz, nx, ny, nz, max_dist, out):
    """Surface point along n from p (n into the material): ahead (+n) when p is in air, behind (-n) when p
    is in material. out = [x, y, z, h] with h = (p - M) . n (the chip thickness of an element at p).
    Returns 1 when found within max_dist."""
    work = np.empty(4)
    hit = np.empty(8)
    ins, d, fm = classify(f, level_at(f, px, py, pz), px, py, pz, 2, work)
    if ins:
        ok = ray_cast(f, px, py, pz, -nx, -ny, -nz, max_dist, 1, 1e-8, hit)
    else:
        ok = ray_cast(f, px, py, pz, nx, ny, nz, max_dist, 0, 1e-8, hit)
    if ok == 0:
        return 0
    out[0] = hit[1]
    out[1] = hit[2]
    out[2] = hit[3]
    out[3] = (px - hit[1]) * nx + (py - hit[2]) * ny + (pz - hit[3]) * nz
    return 1


@njit(void(FIELD_T, F2, F2, f8, F2, B1), **OPTS)
def refill_points(f, P, N, max_dist, out, found):
    """surface_point for many points (out [n, 4] = x, y, z, h; found [n] 1/0)."""
    tmp = np.empty(4)
    for i in range(P.shape[0]):
        ok = surface_point(f, P[i, 0], P[i, 1], P[i, 2], N[i, 0], N[i, 1], N[i, 2], max_dist, tmp)
        found[i] = ok
        for k in range(4):
            out[i, k] = tmp[k] if ok else 0.0


@njit(void(FIELD_T, F2, B1), **OPTS)
def points_inside(f, P, out):
    work = np.empty(4)
    lv_cell_m = f.lv_cell_m
    lv_origin = f.lv_origin_m
    lv_dims = f.lv_dims
    lv_cell_q = f.lv_cell_q
    lv_ray0 = f.lv_ray0
    lv_lo = f.lv_box_lo_m
    lv_hi = f.lv_box_hi_m
    cnt = f.cnt
    seg = f.seg
    link = f.link
    pseg = f.pseg
    pnext = f.pnext
    nl = f.info[A.FI_NLEVELS]
    for i in range(P.shape[0]):
        x = P[i, 0]
        y = P[i, 1]
        z = P[i, 2]
        lvl = C.level_at(lv_lo, lv_hi, lv_cell_m, nl, x, y, z)
        ins, d, fm = C.classify(lv_cell_m, lv_origin, lv_dims, lv_cell_q, lv_ray0, cnt, seg, link, pseg, pnext, lvl, x,
                                y, z, 2, work)
        out[i] = 1 if ins else 0


@njit(types.UniTuple(i8, 2)(I4_1, I4_3, I4_1, I4_3, I4_1, i8), **INLINE)
def _extreme(cnt, seg, link, pseg, pnext, ray):
    """(top, bottom) endpoints (quanta) of a ray; (-1, -1) when empty."""
    n = np.int64(cnt[ray])
    if n == 0:
        return np.int64(-1), np.int64(-1)
    bottom = np.int64(seg[ray, 0, 0])
    if n <= A.K_INLINE:
        top = np.int64(seg[ray, n - 1, 1])
    else:
        b = np.int64(link[ray])
        k = A.K_INLINE
        top = np.int64(0)
        while b >= 0:
            for s in range(A.K_INLINE):
                if k < n:
                    top = np.int64(pseg[b, s, 1])
                    k += 1
            b = np.int64(pnext[b])
    return top, bottom


@njit(types.UniTuple(i8, 2)(FIELD_T, i8, i8, i8, i8), **OPTS)
def _extreme_endpoint(f, lvl, fam, iu, iv):
    """(top, bottom) endpoints (quanta) of a ray; (-1, -1) when empty."""
    return _extreme(f.cnt, f.seg, f.link, f.pseg, f.pnext, C.rindex(f.lv_ray0, f.lv_dims, lvl, fam, iu, iv))


@njit(void(FIELD_T, i8, i8, F1, F1, F1, I1), **OPTS)
def surface_height(f, axis, sign, av, bv, out_h, out_kind):
    """Highest (sign +1) / lowest (sign -1) material surface along S axis ``axis`` above the points whose
    other two coordinates (increasing axis order) are (av, bv): out_h = coordinate x sign (-inf: none),
    out_kind = 1 (part) / 0 (none). Four rays of the axis family around the point: bilinear in their
    extreme endpoints when they agree within one cell (a plane), else the nearest ray (walls, edges)."""
    u = 1 if axis == 0 else 0
    v = 2 if axis < 2 else 1
    lvl = 0
    a = f.lv_cell_m[lvl]
    o = f.lv_origin_m[lvl]
    nu = f.lv_dims[lvl, u]
    nv = f.lv_dims[lvl, v]
    vals = np.empty(4)
    have = np.empty(4, np.int64)
    cnt = f.cnt
    seg = f.seg
    link = f.link
    pseg = f.pseg
    pnext = f.pnext
    lv_ray0 = f.lv_ray0
    lv_dims = f.lv_dims
    for i in range(av.shape[0]):
        fu = (av[i] - o[u]) / a
        fv = (bv[i] - o[v]) / a
        iu = int(math.floor(fu))
        iv = int(math.floor(fv))
        if iu < 0 or iv < 0 or iu > nu - 2 or iv > nv - 2:
            # outside the lattice: nearest boundary ray if within half a cell, else nothing
            out_h[i] = -np.inf
            out_kind[i] = 0
            continue
        al = fu - iu
        be = fv - iv
        nh = 0
        vmin = 1e300
        vmax = -1e300
        for c in range(4):
            du = c & 1
            dv = c >> 1
            top, bot = _extreme(cnt, seg, link, pseg, pnext, C.rindex(lv_ray0, lv_dims, lvl, axis, iu + du, iv + dv))
            if top < 0:
                have[c] = 0
                vals[c] = 0.0
                continue
            e = top if sign > 0 else bot
            val = (o[axis] + e * Q) * sign
            vals[c] = val
            have[c] = 1
            nh += 1
            if val < vmin:
                vmin = val
            if val > vmax:
                vmax = val
        if nh == 0:
            out_h[i] = -np.inf
            out_kind[i] = 0
            continue
        if nh == 4 and vmax - vmin <= a:
            out_h[i] = ((1 - al) * (1 - be) * vals[0] + al * (1 - be) * vals[1] + (1 - al) * be * vals[2]
                        + al * be * vals[3])
            out_kind[i] = 1
            continue
        cn = (1 if al >= 0.5 else 0) + (2 if be >= 0.5 else 0)
        if have[cn] == 1:
            out_h[i] = vals[cn]
            out_kind[i] = 1
        else:
            # nearest ray holding material
            best = -1
            bd = 1e300
            for c in range(4):
                if have[c] == 0:
                    continue
                du = c & 1
                dv = c >> 1
                dist = (al - du) ** 2 + (be - dv) ** 2
                if dist < bd:
                    bd = dist
                    best = c
            out_h[i] = vals[best]
            out_kind[i] = 1


@njit(void(FIELD_T, F1, f8, I1, F3), **OPTS)
def fill_fraction(f, origin, cell, shape, out):
    """Material fraction of the cells of a grid (origin, cubic cell, shape [nx, ny, nz]) from the level-0 z
    family: each z ray is the column [x +- a/2] x [y +- a/2]; its segments are distributed over the cells
    by exact overlap (lengths in z, areas in x-y)."""
    nx = shape[0]
    ny = shape[1]
    nz = shape[2]
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                out[i, j, k] = 0.0
    lvl = 0
    a = f.lv_cell_m[lvl]
    o = f.lv_origin_m[lvl]
    half = 0.5 * a
    gx1 = origin[0] + nx * cell
    gy1 = origin[1] + ny * cell
    i0 = int(math.floor((origin[0] - half - o[0]) / a))
    i1 = int(math.ceil((gx1 + half - o[0]) / a))
    j0 = int(math.floor((origin[1] - half - o[1]) / a))
    j1 = int(math.ceil((gy1 + half - o[1]) / a))
    if i0 < 0:
        i0 = 0
    if j0 < 0:
        j0 = 0
    if i1 > f.lv_dims[lvl, 0] - 1:
        i1 = f.lv_dims[lvl, 0] - 1
    if j1 > f.lv_dims[lvl, 1] - 1:
        j1 = f.lv_dims[lvl, 1] - 1
    inv = 1.0 / (cell * cell * cell)
    K = A.K_INLINE
    cnt = f.cnt
    seg = f.seg
    link = f.link
    pseg = f.pseg
    pnext = f.pnext
    r0 = f.lv_ray0[lvl, 2]
    ny_l = f.lv_dims[lvl, 1]
    for i in range(i0, i1 + 1):
        x = o[0] + i * a
        cx0 = int(math.floor((x - half - origin[0]) / cell))
        cx1 = int(math.floor((x + half - origin[0]) / cell))
        for j in range(j0, j1 + 1):
            ray = r0 + i * ny_l + j
            n = np.int64(cnt[ray])
            if n == 0:
                continue
            y = o[1] + j * a
            cy0 = int(math.floor((y - half - origin[1]) / cell))
            cy1 = int(math.floor((y + half - origin[1]) / cell))
            b = np.int64(link[ray])
            s = 0
            for kk in range(n):
                if kk < K:
                    lo = seg[ray, kk, 0]
                    hi = seg[ray, kk, 1]
                else:
                    if s == K:
                        b = np.int64(pnext[b])
                        s = 0
                    lo = pseg[b, s, 0]
                    hi = pseg[b, s, 1]
                    s += 1
                z0 = o[2] + lo * Q
                z1 = o[2] + hi * Q
                k0 = int(math.floor((z0 - origin[2]) / cell))
                k1 = int(math.floor((z1 - origin[2]) / cell))
                if k0 < 0:
                    k0 = 0
                if k1 > nz - 1:
                    k1 = nz - 1
                for cz in range(k0, k1 + 1):
                    za = origin[2] + cz * cell
                    lz = min(z1, za + cell) - max(z0, za)
                    if lz <= 0.0:
                        continue
                    for cxi in range(cx0, cx1 + 1):
                        if cxi < 0 or cxi >= nx:
                            continue
                        xa = origin[0] + cxi * cell
                        lx = min(x + half, xa + cell) - max(x - half, xa)
                        if lx <= 0.0:
                            continue
                        for cyi in range(cy0, cy1 + 1):
                            if cyi < 0 or cyi >= ny:
                                continue
                            ya = origin[1] + cyi * cell
                            ly = min(y + half, ya + cell) - max(y - half, ya)
                            if ly <= 0.0:
                                continue
                            out[cxi, cyi, cz] += lx * ly * lz * inv


@njit(f8(FIELD_T, f8, f8, f8, f8, f8, f8, f8), **OPTS)
def wall_thickness(f, px, py, pz, nx, ny, nz, max_dist):
    """Material thickness behind the surface point p along -n (n the outward normal): distance from the
    entry into material to the next exit; 0 when no material is met within max_dist, max_dist when it does
    not end."""
    hit = np.empty(8)
    if ray_cast(f, px, py, pz, -nx, -ny, -nz, max_dist, 0, 1e-8, hit) == 0:
        return 0.0
    t_in = hit[0]
    ex = px - nx * t_in
    ey = py - ny * t_in
    ez = pz - nz * t_in
    rest = max_dist - t_in
    if ray_cast(f, ex - nx * 1e-9, ey - ny * 1e-9, ez - nz * 1e-9, -nx, -ny, -nz, rest, 1, 1e-8, hit) == 0:
        return max_dist
    return hit[0] + 1e-9


@njit(i8(FIELD_T, PROFILE_T, F1, F1, f8), **OPTS)
def broad_phase(f, prof, tip, w, margin):
    """Occupied level-0 bricks inside the box of the profile at the pose (tip, unit axis w), grown by
    ``margin``: 0 means the envelope cannot touch material."""
    if prof.z0_m.shape[0] == 0:
        return 0
    zmin = prof.info[A.PI_ZMIN]
    zmax = prof.info[A.PI_ZMAX]
    R = prof.info[A.PI_RMAX] + margin
    a = f.lv_cell_m[0]
    o = f.lv_origin_m[0]
    b0 = np.empty(3, np.int64)
    b1_ = np.empty(3, np.int64)
    for i in range(3):
        pa = tip[i] + zmin * w[i]
        pb = tip[i] + zmax * w[i]
        e = R * math.sqrt(max(0.0, 1.0 - w[i] * w[i]))
        lo = min(pa, pb) - e - margin
        hi = max(pa, pb) + e + margin
        n0 = int(math.floor((lo - o[i]) / a)) - 1
        n1 = int(math.ceil((hi - o[i]) / a)) + 1
        k0 = n0 // A.BRICK
        k1 = n1 // A.BRICK
        bd = f.lv_bdims[0, i]
        if k0 < 0:
            k0 = 0
        if k1 > bd - 1:
            k1 = bd - 1
        if k1 < 0 or k0 > bd - 1:
            return 0
        b0[i] = k0
        b1_[i] = k1
    count = 0
    bmat = f.bmat
    lv_brick0 = f.lv_brick0
    lv_bdims = f.lv_bdims
    for bx in range(b0[0], b1_[0] + 1):
        for by in range(b0[1], b1_[1] + 1):
            for bz in range(b0[2], b1_[2] + 1):
                if _occ(bmat, lv_brick0, lv_bdims, 0, bx, by, bz):
                    count += 1
    return count


@njit(f8(FIELD_T, PROFILE_T, F1, F1, F1, F1, i8, i8, f8, F1), **OPTS)
def sweep_contact(f, prof, c0, c1, w, x0, n_rings, n_around, tol, out):
    """First contact of a stylus (solid of revolution ``prof`` about w, tip at c0 -> c1, translation only)
    with material: the fraction lambda in [0, 1] (-1: none). The leading half of the stylus surface is
    sampled (n_rings per sub-piece, n_around per ring), every sample cast along the move against the
    interpolated field boundary, and the earliest sample refined three times by 4x local sub-sampling.
    out = [lambda, x, y, z (contact point at lambda), nx, ny, nz (outward surface normal), started_inside]."""
    dx = c1[0] - c0[0]
    dy = c1[1] - c0[1]
    dz = c1[2] - c0[2]
    L = math.sqrt(dx * dx + dy * dy + dz * dz)
    out[0] = -1.0
    if L <= 0.0 or prof.z0_m.shape[0] == 0:
        return -1.0
    ux = dx / L
    uy = dy / L
    uz = dz / L
    y0 = np.empty(3)
    y0[0] = w[1] * x0[2] - w[2] * x0[1]
    y0[1] = w[2] * x0[0] - w[0] * x0[2]
    y0[2] = w[0] * x0[1] - w[1] * x0[0]
    hit = np.empty(8)
    best_t = 1e300
    best_z = 0.0
    best_phi = 0.0
    best_dz = 0.0
    best_dphi = 0.0
    zmin = prof.info[A.PI_ZMIN]
    zmax = prof.info[A.PI_ZMAX]
    nz_all = n_rings * prof.z0_m.shape[0]
    dzs = (zmax - zmin) / max(1, nz_all)
    dph = 2.0 * math.pi / max(4, n_around)
    started = 0.0
    for level_ref in range(4):
        if level_ref == 0:
            zs_n = nz_all + 1
            ph_n = max(4, n_around)
            z_start = zmin
            ph_start = 0.0
            zstep = dzs
            pstep = dph
        else:
            zstep = best_dz / 4.0
            pstep = best_dphi / 4.0
            zs_n = 9
            ph_n = 9
            z_start = best_z - 4.0 * zstep
            ph_start = best_phi - 4.0 * pstep
        cur_best = best_t
        for iz in range(zs_n):
            z = z_start + iz * zstep
            if z < zmin:
                z = zmin
            if z > zmax:
                z = zmax
            # radius and outward meridian normal at z
            R = 0.0
            dRdz = 0.0
            for p in range(prof.z0_m.shape[0]):
                if prof.z0_m[p] - 1e-15 <= z <= prof.z1_m[p] + 1e-15:
                    if prof.kind[p] == A.PK_LINE:
                        R = prof.c0[p] + prof.c1[p] * z
                        dRdz = prof.c1[p]
                    else:
                        hh = prof.c2[p] * prof.c2[p] - (z - prof.c0[p]) ** 2
                        sq = math.sqrt(hh) if hh > 0.0 else 0.0
                        R = prof.c1[p] + prof.c3[p] * sq
                        dRdz = -prof.c3[p] * (z - prof.c0[p]) / sq if sq > 1e-15 else 0.0
                    break
            for ip in range(ph_n):
                phi = ph_start + ip * pstep
                cp = math.cos(phi)
                sp = math.sin(phi)
                ex = cp * x0[0] + sp * y0[0]
                ey = cp * x0[1] + sp * y0[1]
                ez = cp * x0[2] + sp * y0[2]
                # outward normal of the revolved surface: e_r - dR/dz w (unnormalised); the bottom pole: -w
                nxx = ex - dRdz * w[0]
                nyy = ey - dRdz * w[1]
                nzz = ez - dRdz * w[2]
                if R <= 1e-15:
                    nxx = -w[0]
                    nyy = -w[1]
                    nzz = -w[2]
                if nxx * ux + nyy * uy + nzz * uz < 0.0:
                    continue
                sx = c0[0] + z * w[0] + R * ex
                sy = c0[1] + z * w[1] + R * ey
                sz = c0[2] + z * w[2] + R * ez
                if ray_cast(f, sx, sy, sz, ux, uy, uz, L, 0, tol, hit) == 1:
                    if hit[7] == 1.0:
                        started = 1.0
                    if hit[0] < best_t:
                        best_t = hit[0]
                        best_z = z
                        best_phi = phi
                        out[1] = hit[1]
                        out[2] = hit[2]
                        out[3] = hit[3]
                        out[4] = hit[4]
                        out[5] = hit[5]
                        out[6] = hit[6]
        if level_ref == 0:
            best_dz = dzs
            best_dphi = dph
        else:
            best_dz = zstep
            best_dphi = pstep
        if best_t >= 1e299:
            return -1.0
        if cur_best - best_t <= tol and level_ref > 0:
            break
    out[0] = best_t / L
    out[7] = started
    return best_t / L


KERNELS = (_brick_occupied, _refine_t, _state, ray_cast, surface_point, refill_points, points_inside, _extreme_endpoint,
           surface_height, fill_fraction, wall_thickness, broad_phase, sweep_contact)
