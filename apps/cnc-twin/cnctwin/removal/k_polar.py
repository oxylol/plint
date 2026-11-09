"""Polar field kernels for turning (architecture 5.4 "Turning", docs/models/removal.md "Polar field").

The workpiece turns about the S z axis; S rotates with it, so the polar field is fixed material. Bin b
covers theta in [b dtheta, (b + 1) dtheta) and row i covers z in [z0 + i dz, z0 + (i + 1) dz); the radial
ray of (b, i) at the bin and row centres holds the material's r segments (quanta from the axis), the axial
ray of (b, j) at radius (j + 1/2) dr holds its z segments (quanta from z0).

* :func:`polar_cut` - one L0 step of a non-rotating tool: every bin whose centre the tool's reference
  plane passes in the step (rotating-frame angle th0 -> th1) is cut with the insert region at the tool
  position interpolated to that angle (helical feed marks and threads come out exactly), radial and axial
  rays alike; the removed volume is the exact integer sum of delta(q_out^2 - q_in^2) of the radial family.
* :func:`polar_radius` - outer material radius at (theta, z) (chip area of an element, architecture 5.4).
* :func:`polar_to_ray` / :func:`rebuild_tridexel` - the exact tri-dexel content of the polar region (sync).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, types

from . import abi as A
from .k_ray import brick_account, ray_index, ray_load, ray_store, target_overlap
from .nbt import (ACCUM_T, EVENTS_T, F1, F2, FIELD_T, I1, INLINE, INSERT_T, OPTS, POLAR_T, RAW, SCRATCH_T,
                  TARGET_T, f8, i8, void)

Q = A.QUANTUM_M
TWO_PI = 2.0 * math.pi
PB = A.POLAR_BLOCK


# --------------------------------------------------------------------------- insert region
@njit(f8(F1, F1, f8, i8), **INLINE)
def _chain(zs, xs, z, upper):
    """Piecewise-linear chain value at z (outside: -inf for the upper chain, +inf for the lower)."""
    n = zs.shape[0]
    if n == 0 or z < zs[0] or z > zs[n - 1]:
        return -1e300 if upper == 1 else 1e300
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if zs[mid] <= z:
            lo = mid
        else:
            hi = mid
    dz = zs[hi] - zs[lo]
    if dz <= 0.0:
        return max(xs[lo], xs[hi]) if upper == 1 else min(xs[lo], xs[hi])
    return xs[lo] + (xs[hi] - xs[lo]) * (z - zs[lo]) / dz


@njit(types.Tuple((f8, i8))(F1, F1, f8, i8, i8), **INLINE)
def _chain_w(zs, xs, z, upper, k):
    """_chain with a bracket hint ``k`` (returns (value, bracket)): the same bracket as the bisection (the
    largest k <= n - 2 with zs[k] <= z), found by stepping from the hint, so a monotone sequence of z
    costs O(1) per evaluation and the values are identical to _chain's."""
    n = zs.shape[0]
    if n == 0 or z < zs[0] or z > zs[n - 1]:
        return (-1e300 if upper == 1 else 1e300), k
    if n == 1:
        return xs[0], 0
    if k < 0 or k > n - 2:
        k = 0
    while k < n - 2 and zs[k + 1] <= z:
        k += 1
    while k > 0 and zs[k] > z:
        k -= 1
    dz = zs[k + 1] - zs[k]
    if dz <= 0.0:
        return (max(xs[k], xs[k + 1]) if upper == 1 else min(xs[k], xs[k + 1])), k
    return xs[k] + (xs[k + 1] - xs[k]) * (z - zs[k]) / dz, k


@njit(types.UniTuple(f8, 2)(F1, F1, F1, F1, F1, f8), **INLINE)
def _x_range(zu, xu, zl, xl, nose, zt):
    hi = _chain(zu, xu, zt, 1)
    lo = _chain(zl, xl, zt, 0)
    rho = nose[2]
    if rho > 0.0:
        d = zt - nose[0]
        h2 = rho * rho - d * d
        if h2 >= 0.0:
            h = math.sqrt(h2)
            top = nose[1] + h
            if top > hi:
                hi = top
            bot = nose[1] - h
            if bot < lo:
                lo = bot
    return lo, hi


@njit(types.UniTuple(f8, 2)(F1, F1, F1, F1, F1, f8), **INLINE)
def _z_range(xa, za, xb, zb, nose, xt):
    left = _chain(xa, za, xt, 0)
    right = _chain(xb, zb, xt, 1)
    rho = nose[2]
    if rho > 0.0:
        d = xt - nose[1]
        h2 = rho * rho - d * d
        if h2 >= 0.0:
            h = math.sqrt(h2)
            l2 = nose[0] - h
            if l2 < left:
                left = l2
            r2 = nose[0] + h
            if r2 > right:
                right = r2
    return left, right


@njit(types.Tuple((f8, f8, i8, i8))(F1, F1, F1, F1, F1, f8, i8, i8), **INLINE)
def _z_range_w(xa, za, xb, zb, nose, xt, ka, kb):
    """_z_range with bracket hints (identical values)."""
    left, ka = _chain_w(xa, za, xt, 0, ka)
    right, kb = _chain_w(xb, zb, xt, 1, kb)
    rho = nose[2]
    if rho > 0.0:
        d = xt - nose[1]
        h2 = rho * rho - d * d
        if h2 >= 0.0:
            h = math.sqrt(h2)
            l2 = nose[0] - h
            if l2 < left:
                left = l2
            r2 = nose[0] + h
            if r2 > right:
                right = r2
    return left, right, ka, kb


@njit(types.UniTuple(f8, 2)(INSERT_T, f8), **OPTS)
def insert_x_range(ins, zt):
    """(x_lo, x_hi) of the insert at z_T (x_lo > x_hi: no insert there); the nose arc is exact."""
    return _x_range(ins.zu, ins.xu, ins.zl, ins.xl, ins.nose, zt)


@njit(types.UniTuple(f8, 2)(INSERT_T, f8), **OPTS)
def insert_z_range(ins, xt):
    """(z_left, z_right) of the insert at x_T (left > right: none)."""
    return _z_range(ins.xa, ins.za, ins.xb, ins.zb, ins.nose, xt)


# --------------------------------------------------------------------------- inline ray algebra
@njit(types.Tuple((i8, i8, f8, i8, i8))(types.int32[:, :, ::1], types.int32[:, :, ::1], types.int8[::1], i8, i8, i8,
                                         i8, i8), **INLINE)
def _cut_inline(seg, sev, cnt, ray, qa, qb, ev, kp):
    """Remove [qa, qb) from an inline ray (at most kp segments) in place. Returns (removed sum of q^2
    differences (radial use), status: 0 changed, 1 nothing removed, -1 overflow (unchanged), the removed sum of q^3
    differences (float: the radial centroid's moment), the first and last removed quantum)."""
    n = np.int64(cnt[ray])
    if n == 0 or qb <= qa:
        return np.int64(0), np.int64(1), 0.0, np.int64(0), np.int64(0)
    if qa >= np.int64(seg[ray, n - 1, 1]) or qb <= np.int64(seg[ray, 0, 0]):
        return np.int64(0), np.int64(1), 0.0, np.int64(0), np.int64(0)
    f0 = 0
    while f0 < n and np.int64(seg[ray, f0, 1]) <= qa:
        f0 += 1
    l0 = n - 1
    while l0 >= 0 and np.int64(seg[ray, l0, 0]) >= qb:
        l0 -= 1
    if l0 < f0:
        return np.int64(0), np.int64(1), 0.0, np.int64(0), np.int64(0)
    a_f = np.int64(seg[ray, f0, 0])
    e_af = np.int64(sev[ray, f0, 0])
    b_l = np.int64(seg[ray, l0, 1])
    e_bl = np.int64(sev[ray, l0, 1])
    keep_lo = 1 if a_f < qa else 0
    keep_hi = 1 if b_l > qb else 0
    m = n - (l0 - f0 + 1) + keep_lo + keep_hi
    if m > kp:
        return np.int64(0), np.int64(-1), 0.0, np.int64(0), np.int64(0)
    q2 = np.int64(0)
    q3 = 0.0
    x_lo = a_f if a_f > qa else qa
    x_hi = b_l if b_l < qb else qb
    for k in range(f0, l0 + 1):
        a = np.int64(seg[ray, k, 0])
        b = np.int64(seg[ray, k, 1])
        x0 = a if a > qa else qa
        x1 = b if b < qb else qb
        q2 += x1 * x1 - x0 * x0
        fx0 = float(x0)
        fx1 = float(x1)
        q3 += fx1 * fx1 * fx1 - fx0 * fx0 * fx0
    # the segments after the cut move to their new place (from the end when they move up)
    src = l0 + 1
    dst = f0 + keep_lo + keep_hi
    if dst > src:
        for k in range(n - 1, src - 1, -1):
            for e in range(2):
                seg[ray, k + dst - src, e] = seg[ray, k, e]
                sev[ray, k + dst - src, e] = sev[ray, k, e]
    elif dst < src:
        for k in range(src, n):
            for e in range(2):
                seg[ray, k + dst - src, e] = seg[ray, k, e]
                sev[ray, k + dst - src, e] = sev[ray, k, e]
    pos = f0
    if keep_lo == 1:
        seg[ray, pos, 0] = a_f
        seg[ray, pos, 1] = qa
        sev[ray, pos, 0] = e_af
        sev[ray, pos, 1] = ev
        pos += 1
    if keep_hi == 1:
        seg[ray, pos, 0] = qb
        seg[ray, pos, 1] = b_l
        sev[ray, pos, 0] = ev
        sev[ray, pos, 1] = e_bl
    for k in range(m, n):
        for e in range(2):
            seg[ray, k, e] = 0
            sev[ray, k, e] = 0
    cnt[ray] = m
    return q2, np.int64(0), q3, x_lo, x_hi


@njit(f8(F1, F1, F1, F1, f8, f8), **INLINE)
def _xhi_max(zu, xu, nose, box, za, zb):
    """Largest x_T of the insert over z_T in [za, zb] (-inf when it has none there): O(1) for a unimodal upper
    chain (box[5] = 1: the chain at the range point nearest its highest vertex), else a scan of its vertices;
    the nose arc's top is concave (its value at the range point nearest the arc centre)."""
    best = -1e300
    if box[5] == 1.0:
        zp = box[4]
        zz = zp if (zp >= za and zp <= zb) else (za if zp < za else zb)
        best = _chain(zu, xu, zz, 1)
    else:
        for k in range(zu.shape[0]):
            if zu[k] >= za and zu[k] <= zb and xu[k] > best:
                best = xu[k]
        v = _chain(zu, xu, za, 1)
        if v > best:
            best = v
        v = _chain(zu, xu, zb, 1)
        if v > best:
            best = v
    rho = nose[2]
    if rho > 0.0:
        zc = nose[0]
        zz = zc if (zc >= za and zc <= zb) else (za if abs(za - zc) < abs(zb - zc) else zb)
        d = zz - zc
        if rho * rho - d * d >= 0.0:
            top = nose[1] + math.sqrt(rho * rho - d * d)
            if top > best:
                best = top
    return best


@njit(types.Tuple((f8, i8))(F1, F1, F1, F1, f8, f8, i8), **INLINE)
def _xhi_max_w(zu, xu, nose, box, za, zb, k):
    """_xhi_max with a bracket hint for the unimodal chain (identical values; returns (value, bracket))."""
    if box[5] != 1.0:
        return _xhi_max(zu, xu, nose, box, za, zb), k
    zp = box[4]
    zz = zp if (zp >= za and zp <= zb) else (za if zp < za else zb)
    best, k = _chain_w(zu, xu, zz, 1, k)
    rho = nose[2]
    if rho > 0.0:
        zc = nose[0]
        zz = zc if (zc >= za and zc <= zb) else (za if abs(za - zc) < abs(zb - zc) else zb)
        d = zz - zc
        if rho * rho - d * d >= 0.0:
            top = nose[1] + math.sqrt(rho * rho - d * d)
            if top > best:
                best = top
    return best, k


@njit(f8(INSERT_T, f8, f8), **OPTS)
def insert_xhi_max(ins, za, zb):
    """Largest x_T of the insert over z_T in [za, zb] (-inf when it has none there)."""
    return _xhi_max(ins.zu, ins.xu, ins.nose, ins.box, za, zb)


@njit(void(types.int8[::1], types.int32[:, :, ::1], I1, i8, i8, i8), **INLINE)
def _rescan(rcnt, rseg, zmax, nz, b, blk):
    nblk = (nz + PB - 1) // PB
    best = np.int64(0)
    for i in range(blk * PB, min(nz, blk * PB + PB)):
        ray = b * nz + i
        n = rcnt[ray]
        if n > 0:
            r = np.int64(rseg[ray, n - 1, 1])
            if r > best:
                best = r
    zmax[b * nblk + blk] = best


@njit(void(POLAR_T, i8, i8), **OPTS)
def _rescan_block(pf, b, blk):
    _rescan(pf.rcnt, pf.rseg, pf.zmax_q, np.int64(pf.geom[A.PG_NZ]), b, blk)


@njit(i8(f8), **INLINE)
def _q_of(r):
    """Nearest quantum of a length (m)."""
    return np.int64(math.floor(r / Q + 0.5))


@njit(f8(f8, f8), **INLINE)
def _rad(rho, yo):
    """Distance from the axis of a point at rho in the reference plane offset yo from the axis."""
    return math.sqrt(rho * rho + yo * yo) if yo != 0.0 else rho


@njit(types.UniTuple(i8, 4)(types.int32[:, :, ::1], types.int32[:, :, ::1], types.int8[::1], I1, F1, F1, F1, F1, F1, i8,
                             i8, i8, i8, i8, f8, f8, f8, f8, f8, f8, f8, i8, i8, i8, F1, f8, f8, f8), **RAW)
def _polar_rows(rseg, rsev, rcnt, zmax, zu, xu, zl, xl, nose, b, nz, blk, ra0, ra1, z0f, dz, zt, sz, rt, sx, yo,
                top_blk, ev, kp, fv, dth, dnr, dnz):
    """The radial rays of rows ra0..ra1 of bin b under the insert (one block of polar_cut; compiled apart
    so the hot loop keeps its values in registers). Returns (removed q^2 sum, overflow (1/0), first and
    last row changed (-1: none)); zmax_q of the block is rescanned when its highest row was cut.

    The turning contact's geometry goes into ``fv`` (the contact's RemovalAccum.fvals, abi FA_*): removed volume,
    its radius and z moments, the radial sign of the generated surface's outward normal, the extent along the tool
    axis and from it relative to the tip, and across the feed ((dnr, dnz): the unit normal of the step's feed in
    the r-z plane)."""
    rho_n = nose[0]
    rho_x = nose[1]
    rho_r = nose[2]
    ku = 0
    kl = 0
    q2_total = np.int64(0)
    status = 0
    d0 = -1
    d1 = -1
    rescan = False
    for i in range(ra0, ra1 + 1):
        ray = b * nz + i
        n_old = np.int64(rcnt[ray])
        if n_old == 0:
            continue
        ztl = (z0f + (i + 0.5) * dz - zt) * sz
        # deep side first: the upper chain (sx < 0) or the lower chain (sx > 0), with the nose arc
        if sx < 0.0:
            xd, ku = _chain_w(zu, xu, ztl, 1, ku)
            if rho_r > 0.0:
                d = ztl - rho_n
                h2 = rho_r * rho_r - d * d
                if h2 >= 0.0:
                    t = rho_x + math.sqrt(h2)
                    if t > xd:
                        xd = t
            if xd <= -1e299:
                continue
            rho_a = rt - xd
            if rho_a < 0.0:
                rho_a = 0.0
            r_a = _rad(rho_a, yo)
            old_top = np.int64(rseg[ray, n_old - 1, 1])
            if r_a >= old_top * Q:
                continue
            qa = _q_of(r_a)
            if qa >= old_top:
                continue
            xo, kl = _chain_w(zl, xl, ztl, 0, kl)
            if rho_r > 0.0:
                d = ztl - rho_n
                h2 = rho_r * rho_r - d * d
                if h2 >= 0.0:
                    t = rho_x - math.sqrt(h2)
                    if t < xo:
                        xo = t
            if xo > xd:
                continue
            rho_b = rt - xo
            if rho_b <= 0.0:
                continue
            qb = _q_of(_rad(rho_b, yo))
        else:
            xd, ku = _chain_w(zu, xu, ztl, 1, ku)
            xo, kl = _chain_w(zl, xl, ztl, 0, kl)
            if rho_r > 0.0:
                d = ztl - rho_n
                h2 = rho_r * rho_r - d * d
                if h2 >= 0.0:
                    h = math.sqrt(h2)
                    if rho_x + h > xd:
                        xd = rho_x + h
                    if rho_x - h < xo:
                        xo = rho_x - h
            if xo > xd:
                continue
            rho_b = rt + xd
            if rho_b <= 0.0:
                continue
            qb = _q_of(_rad(rho_b, yo))
            if qb <= np.int64(rseg[ray, 0, 0]):
                continue
            rho_a = rt + xo
            if rho_a < 0.0:
                rho_a = 0.0
            qa = _q_of(_rad(rho_a, yo))
            old_top = np.int64(rseg[ray, n_old - 1, 1])
        q2, st, q3, x_lo, x_hi = _cut_inline(rseg, rsev, rcnt, ray, qa, qb, ev, kp)
        if st < 0:
            status = 1
            continue
        if st == 0:
            q2_total += q2
            if old_top >= top_blk:
                rescan = True
            if d0 < 0:
                d0 = i
            d1 = i
            # the contact's geometry: volume, its radius and z moments, extents relative to the tip
            zi = z0f + (i + 0.5) * dz
            vol = 0.5 * float(q2) * Q * Q * dth * dz
            fv[A.FA_W] += vol
            fv[A.FA_CX] += q3 * (Q * Q * Q / 3.0) * dth * dz
            fv[A.FA_CZ] += vol * zi
            fv[A.FA_NX] += -sx * vol
            zr = (zi - zt) * sz
            if zr < fv[A.FA_ZREL_MIN]:
                fv[A.FA_ZREL_MIN] = zr
            if zr > fv[A.FA_ZREL_MAX]:
                fv[A.FA_ZREL_MAX] = zr
            for e in range(2):
                rr = (x_lo if e == 0 else x_hi) * Q
                dr_ = abs(rr - rt)
                if dr_ > fv[A.FA_RREL_MAX]:
                    fv[A.FA_RREL_MAX] = dr_
                dd = (rr - rt) * dnr + (zi - zt) * dnz
                if dd < fv[A.FA_DEPTH_MIN]:
                    fv[A.FA_DEPTH_MIN] = dd
                if dd > fv[A.FA_DEPTH_MAX]:
                    fv[A.FA_DEPTH_MAX] = dd
    if rescan:
        _rescan(rcnt, rseg, zmax, nz, b, blk)
    return q2_total, np.int64(status), np.int64(d0), np.int64(d1)


@njit(types.UniTuple(i8, 3)(types.int32[:, :, ::1], types.int32[:, :, ::1], types.int8[::1], F1, F1, F1, F1, F1, i8, i8,
                             i8, i8, f8, f8, f8, f8, f8, f8, f8, i8, i8, i8, i8), **RAW)
def _polar_axial(aseg, asev, acnt, xa, za, xb, zb, nose, b, nr, j0, j1, dr, yo, rt, sx, zt, sz, z0f, g_lo, g_hi, ev,
                 kpa):
    """The axial rays j0..j1 of bin b under the insert (polar_cut). Returns (overflow (1/0), first and last
    ray changed (-1: none))."""
    yo2 = yo * yo
    ka = 0
    kb = 0
    status = 0
    d0 = -1
    d1 = -1
    for j in range(j0, j1 + 1):
        ray = b * nr + j
        na = np.int64(acnt[ray])
        if na == 0:
            continue
        if np.int64(aseg[ray, 0, 0]) >= g_hi or np.int64(aseg[ray, na - 1, 1]) <= g_lo:
            continue
        rj = (j + 0.5) * dr
        rho = math.sqrt(max(rj * rj - yo2, 0.0)) if yo != 0.0 else rj
        xt = (rho - rt) * sx
        zlft, zrgt, ka, kb = _z_range_w(xa, za, xb, zb, nose, xt, ka, kb)
        if zlft > zrgt:
            continue
        ga = zt + sz * zlft
        gb = zt + sz * zrgt
        if ga > gb:
            ga, gb = gb, ga
        qa = _q_of(ga - z0f)
        qb = _q_of(gb - z0f)
        q2, st, _q3, _xl, _xh = _cut_inline(aseg, asev, acnt, ray, qa, qb, ev, kpa)
        if st < 0:
            status = 1
            continue
        if st == 0:
            if d0 < 0:
                d0 = j
            d1 = j
    return np.int64(status), np.int64(d0), np.int64(d1)


# --------------------------------------------------------------------------- the L0 cut
@njit(i8(POLAR_T, INSERT_T, F1, i8, ACCUM_T), **RAW)
def polar_cut(pf, ins, ps, ev, acc):
    """Cut the bins the tool's reference plane passes in one L0 step (PolarStep values ``ps``). Returns 0,
    or 1 when a ray would need more than its inline segments (PI_OVERFLOW bit 1 radial, 2 axial; that ray is
    unchanged: polar.grow_kp and repeat the step - the cut is idempotent).

    Per bin: the rows under the insert's z extent are taken A.POLAR_BLOCK at a time; a block is skipped
    when its largest material radius (zmax_q) is not above the insert's deepest point there (first the
    insert's overall deepest point, then its deepest point over the block's rows); in the remaining rows
    (_polar_rows) the deep side of the insert is evaluated first and the other side only when the ray holds
    material beyond it. The axial rays under the insert's radial extent follow (_polar_axial).
    Allocation-free, reference counting off."""
    th0 = ps[A.PS_TH0]
    th1 = ps[A.PS_TH1]
    a_lo = th0 if th0 < th1 else th1
    a_hi = th1 if th0 < th1 else th0
    if a_hi - a_lo <= 0.0:
        return 0
    g = pf.geom
    dth = g[A.PG_DTHETA]
    # bin centre c_b = (b + 1/2) dth + 2 pi m: first index whose centre exceeds a_lo
    kfirst = int(math.floor((a_lo / dth) - 0.5)) + 1
    klast = int(math.floor((a_hi / dth) - 0.5))
    aiv = acc.ivals
    aiv[A.IA_POLAR_STEPS] += 1
    if klast < kfirst:
        return 0
    nth = np.int64(g[A.PG_NTHETA])
    z0f = g[A.PG_Z0]
    dz = g[A.PG_DZ]
    nz = np.int64(g[A.PG_NZ])
    dr = g[A.PG_DR]
    nr = np.int64(g[A.PG_NR])
    kp = pf.info[A.PI_KP]
    kpa = pf.info[A.PI_KPA]
    rseg = pf.rseg
    rsev = pf.rsev
    rcnt = pf.rcnt
    aseg = pf.aseg
    asev = pf.asev
    acnt = pf.acnt
    zmax = pf.zmax_q
    pinfo = pf.info
    zu = ins.zu
    xu = ins.xu
    zl = ins.zl
    xl = ins.xl
    xa = ins.xa
    za_ = ins.za
    xb = ins.xb
    zb_ = ins.zb
    nose = ins.nose
    box = ins.box
    sz = ps[A.PS_SZ]
    sx = ps[A.PS_SX]
    yo = ps[A.PS_YOFF]
    # the depth of cut is the removed material's extent across the feed: the unit normal of the step's feed in the
    # r-z plane (radial when the tool does not move)
    fr_ = ps[A.PS_R1] - ps[A.PS_R0]
    fz_ = ps[A.PS_Z1] - ps[A.PS_Z0]
    fl_ = math.sqrt(fr_ * fr_ + fz_ * fz_)
    if fl_ > 1e-15:
        dnr = -fz_ / fl_
        dnz = fr_ / fl_
    else:
        dnr = 1.0
        dnz = 0.0
    afv = acc.fvals
    status = 0
    zt_min = box[0]
    zt_max = box[1]
    xt_min = box[2]
    xt_max = box[3]
    q2_total = np.int64(0)
    nblk = (nz + PB - 1) // PB
    dirty_z0 = pinfo[A.PI_DIRTY_Z0]
    dirty_z1 = pinfo[A.PI_DIRTY_Z1]
    dirty_r0 = pinfo[A.PI_DIRTY_R0]
    dirty_r1 = pinfo[A.PI_DIRTY_R1]
    touched = False
    t_lo = 1e300
    t_hi = -1.0
    kx = 0
    for kk in range(kfirst, klast + 1):
        cb = (kk + 0.5) * dth
        lam = (cb - th0) / (th1 - th0)
        zt = ps[A.PS_Z0] + lam * (ps[A.PS_Z1] - ps[A.PS_Z0])
        rt = ps[A.PS_R0] + lam * (ps[A.PS_R1] - ps[A.PS_R0])
        b = kk % nth
        if b < 0:
            b += nth
        # the insert's smallest radius (sx < 0: an external tool toward the axis, sx > 0: away from it)
        rho_deep = rt - xt_max if sx < 0.0 else rt + xt_min
        if rho_deep < 0.0:
            rho_deep = 0.0
        q_deep = _q_of(_rad(rho_deep, yo))
        # rows covered by the insert at this position, block by block
        za = zt + sz * zt_min
        zb = zt + sz * zt_max
        if za > zb:
            za, zb = zb, za
        i0 = int(math.floor((za - z0f) / dz - 0.5)) + 1
        i1 = int(math.floor((zb - z0f) / dz - 0.5))
        if i0 < 0:
            i0 = 0
        if i1 > nz - 1:
            i1 = nz - 1
        blk0 = i0 // PB
        blk1 = i1 // PB if i1 >= 0 else -1
        for blk in range(blk0, blk1 + 1):
            top_blk = zmax[b * nblk + blk]
            if top_blk <= q_deep:
                continue
            ra0 = blk * PB
            ra1 = min(nz - 1, blk * PB + PB - 1)
            if ra0 < i0:
                ra0 = i0
            if ra1 > i1:
                ra1 = i1
            if sx < 0.0:
                # the insert's deepest point over the block's rows against the block's largest radius
                zr0 = (z0f + (ra0 + 0.5) * dz - zt) * sz
                zr1 = (z0f + (ra1 + 0.5) * dz - zt) * sz
                if zr0 > zr1:
                    zr0, zr1 = zr1, zr0
                xmax, kx = _xhi_max_w(zu, xu, nose, box, zr0, zr1, kx)
                if xmax <= -1e299:
                    continue
                rho_min = rt - xmax
                if rho_min < 0.0:
                    rho_min = 0.0
                # nearest quantum of the deepest radius >= top_blk (the float test is the sufficient half)
                r_min = _rad(rho_min, yo)
                if r_min >= top_blk * Q or _q_of(r_min) >= top_blk:
                    continue
            q2, st, d0, d1 = _polar_rows(rseg, rsev, rcnt, zmax, zu, xu, zl, xl, nose, b, nz, blk, ra0, ra1, z0f, dz,
                                         zt, sz, rt, sx, yo, top_blk, ev, kp, afv, dth, dnr, dnz)
            if st != 0:
                status = 1
                pinfo[A.PI_OVERFLOW] |= 1
            q2_total += q2
            if d0 >= 0:
                touched = True
                if dirty_z0 < 0 or d0 < dirty_z0:
                    dirty_z0 = d0
                if d1 > dirty_z1:
                    dirty_z1 = d1
        # axial rays inside the insert's radial range
        if sx < 0.0:
            rho_lo = rt - xt_max
            rho_hi = rt - xt_min
        else:
            rho_lo = rt + xt_min
            rho_hi = rt + xt_max
        if rho_lo < 0.0:
            rho_lo = 0.0
        r_lo = _rad(rho_lo, yo)
        r_hi = _rad(rho_hi, yo)
        if r_lo < t_lo:
            t_lo = r_lo
        if r_hi > t_hi:
            t_hi = r_hi
        j0 = int(math.ceil(r_lo / dr - 0.5))
        j1 = int(math.floor(r_hi / dr - 0.5))
        if j0 < 0:
            j0 = 0
        if j1 > nr - 1:
            j1 = nr - 1
        g_lo = _q_of(za - z0f)
        g_hi = _q_of(zb - z0f)
        if j1 >= j0:
            st, d0, d1 = _polar_axial(aseg, asev, acnt, xa, za_, xb, zb_, nose, b, nr, j0, j1, dr, yo, rt, sx, zt, sz,
                                      z0f, g_lo, g_hi, ev, kpa)
            if st != 0:
                status = 1
                pinfo[A.PI_OVERFLOW] |= 2
            if d0 >= 0:
                touched = True
                if dirty_r0 < 0 or d0 < dirty_r0:
                    dirty_r0 = d0
                if d1 > dirty_r1:
                    dirty_r1 = d1
    if touched and t_hi >= 0.0:
        # the radii the step can have changed: the insert's radial extent (the sync rebuilds only those)
        q_lo = np.int64(math.floor(t_lo / Q))
        q_hi = np.int64(math.ceil(t_hi / Q))
        if q_lo < 0:
            q_lo = 0
        for k0, k1 in ((A.PI_TOUCH_Q0, A.PI_TOUCH_Q1), (A.PI_OPTOUCH_Q0, A.PI_OPTOUCH_Q1)):
            if pinfo[k0] < 0 or q_lo < pinfo[k0]:
                pinfo[k0] = q_lo
            if q_hi > pinfo[k1]:
                pinfo[k1] = q_hi
    pinfo[A.PI_DIRTY_Z0] = dirty_z0
    pinfo[A.PI_DIRTY_Z1] = dirty_z1
    pinfo[A.PI_DIRTY_R0] = dirty_r0
    pinfo[A.PI_DIRTY_R1] = dirty_r1
    aiv[A.IA_POLAR_Q2] += q2_total
    if q2_total > 0:
        pinfo[A.PI_VERSION] += 1
        aiv[A.IA_IN_CUT_CALLS] += 1
        afv[A.FA_POLAR_TH] = th1
    return status


@njit(f8(POLAR_T, f8, f8), **RAW)
def polar_radius(pf, theta, z):
    """Outer material radius at (theta, z) in the rotating frame (0: no material)."""
    g = pf.geom
    nth = np.int64(g[A.PG_NTHETA])
    dth = g[A.PG_DTHETA]
    nz = np.int64(g[A.PG_NZ])
    i = int(math.floor((z - g[A.PG_Z0]) / g[A.PG_DZ]))
    if i < 0 or i >= nz:
        return 0.0
    t = theta - TWO_PI * math.floor(theta / TWO_PI)
    b = int(math.floor(t / dth))
    if b >= nth:
        b = nth - 1
    ray = b * nz + i
    n = pf.rcnt[ray]
    if n == 0:
        return 0.0
    return pf.rseg[ray, n - 1, 1] * Q


@njit(void(POLAR_T, F1, F1, F1), **RAW)
def polar_radii(pf, theta, z, out):
    for k in range(theta.shape[0]):
        out[k] = polar_radius(pf, theta[k], z[k])


@njit(i8(POLAR_T, i8), **OPTS)
def polar_q2_sum(pf, dummy):
    """Sum over radial rays of q_hi^2 - q_lo^2 (the exact integer volume / (dtheta / 2 dz quantum^2));
    int64 suffices up to ~1e19, i.e. a polar field of 1.8 m^3 at 0.5 deg and 12.5 um rows."""
    total = np.int64(0)
    for ray in range(pf.rcnt.shape[0]):
        for k in range(pf.rcnt[ray]):
            a = np.int64(pf.rseg[ray, k, 0])
            b = np.int64(pf.rseg[ray, k, 1])
            total += b * b - a * a
    return total


# --------------------------------------------------------------------------- polar -> tri-dexel
@njit(i8(F1, F1, I1, I1, i8, f8, f8, i8, i8, i8, I1, I1, I1, I1, i8), **INLINE)
def _add_piece(lo_t, hi_t, el, eh, m, a, b, ea, eb, dummy, o_lo, o_hi, o_el, o_eh, cap):
    """Append the piece [a, b) (metres along the ray) to the work list; returns the new count (-1 full)."""
    if b <= a:
        return m
    if m >= lo_t.shape[0]:
        return -1
    lo_t[m] = a
    hi_t[m] = b
    el[m] = ea
    eh[m] = eb
    return m + 1


@njit(i8(POLAR_T, f8, f8, f8, i8, f8, f8, I1, I1, I1, I1, i8, F1, F1, I1, I1, F2), **RAW)
def polar_to_ray(pf, ox, oy, oz, fam, t_lo, t_hi, out_lo, out_hi, out_elo, out_ehi, cap, lo_t, hi_t, el, eh, trig):
    """Material of the polar field along the ray o + t e_fam, t in [t_lo, t_hi] (metres from o): sorted,
    merged intervals written as quanta of t; returns their count (-1: a list is full).

    z rays read the radial ray of their bin row by row; a change of state between two rows is placed at
    the endpoint of the axial ray of the nearest radius that lies between the row centres (exact faces),
    else half way. x and y rays (constant z) cross the bins of their row: in bin b the line's points have
    s between the values at the bin's edge angles (s = c cot(theta) for an x ray at y = c, s = c tan(theta)
    for a y ray at x = c), and the radial segment [r0, r1) keeps c^2 + s^2 in [r0^2, r1^2). Only the bins
    the line crosses within the field's largest radius are visited. ``trig`` [ntheta + 1, 4] holds sin and
    cos of the bin edges b dtheta and of the bin middles (b + 1/2) dtheta (bin_trig: the same values as
    computed in place). Allocation-free, reference counting off."""
    g = pf.geom
    nth = np.int64(g[A.PG_NTHETA])
    dth = g[A.PG_DTHETA]
    z0f = g[A.PG_Z0]
    dz = g[A.PG_DZ]
    nz = np.int64(g[A.PG_NZ])
    dr = g[A.PG_DR]
    nr = np.int64(g[A.PG_NR])
    n = 0
    if fam == 2:
        r = math.sqrt(ox * ox + oy * oy)
        th = math.atan2(oy, ox)
        if th < 0.0:
            th += TWO_PI
        b = int(math.floor(th / dth))
        if b >= nth:
            b = nth - 1
        rq = r / Q
        j = int(math.floor(r / dr))
        if j > nr - 1:
            j = nr - 1
        aray = b * nr + j
        i_lo = int(math.floor((oz + t_lo - z0f) / dz))
        i_hi = int(math.floor((oz + t_hi - z0f) / dz))
        if i_lo < 0:
            i_lo = 0
        if i_hi > nz - 1:
            i_hi = nz - 1
        prev = False
        start = 0.0
        ev_start = np.int64(A.SEV_NONE)
        zrcnt = pf.rcnt
        zrseg = pf.rseg
        zrsev = pf.rsev
        for i in range(i_lo, i_hi + 2):
            inside = False
            ev_row = np.int64(A.SEV_NONE)
            if i <= i_hi:
                ray = b * nz + i
                for k in range(zrcnt[ray]):
                    if rq >= zrseg[ray, k, 0] and rq < zrseg[ray, k, 1]:
                        inside = True
                        ev_row = np.int64(zrsev[ray, k, 1])
                        break
            if inside == prev:
                continue
            # boundary between the centres of rows i - 1 and i
            zc0 = z0f + (i - 0.5) * dz
            zc1 = z0f + (i + 0.5) * dz
            zb = z0f + i * dz
            ev_b = ev_row
            for k in range(pf.acnt[aray]):
                e = pf.aseg[aray, k, 0] if inside else pf.aseg[aray, k, 1]
                ze = z0f + e * Q
                if ze >= zc0 and ze <= zc1:
                    zb = ze
                    ev_b = np.int64(pf.asev[aray, k, 0] if inside else pf.asev[aray, k, 1])
                    break
            if i == i_lo and inside:
                zb = oz + t_lo
            if i == i_hi + 1 and not inside:
                if zb > oz + t_hi:
                    zb = oz + t_hi
            if inside:
                start = zb
                ev_start = ev_b
            else:
                a_t = start - oz
                b_t = zb - oz
                if a_t < t_lo:
                    a_t = t_lo
                if b_t > t_hi:
                    b_t = t_hi
                if b_t > a_t:
                    if n >= cap:
                        return -1
                    out_lo[n] = np.int64(np.floor(a_t / Q + 0.5))
                    out_hi[n] = np.int64(np.floor(b_t / Q + 0.5))
                    out_elo[n] = ev_start
                    out_ehi[n] = ev_b
                    n += 1
            prev = inside
        return n
    i = int(math.floor((oz - z0f) / dz))
    if i < 0 or i >= nz:
        return 0
    c = oy if fam == 0 else ox
    s_origin = ox if fam == 0 else oy
    rcnt = pf.rcnt
    rseg = pf.rseg
    rsev = pf.rsev
    m = 0
    # the bins the line crosses within the field's largest radius (one bin of slack each side)
    rfield = nr * dr
    k_first = 0
    k_last = nth - 1
    if c != 0.0 and abs(c) < rfield:
        hh = math.sqrt(rfield * rfield - c * c)
        # angles of the line's ends at s = -hh and s = +hh: the line (off the axis) spans less than pi between
        if fam == 0:
            t1 = math.atan2(c, -hh)
            t2 = math.atan2(c, hh)
        else:
            t1 = math.atan2(-hh, c)
            t2 = math.atan2(hh, c)
        d = t2 - t1
        if d > math.pi:
            d -= TWO_PI
        elif d <= -math.pi:
            d += TWO_PI
        ta = t2 if d < 0.0 else t1
        tb = ta + abs(d)
        k_first = int(math.floor(ta / dth)) - 1
        k_last = int(math.floor(tb / dth)) + 1
        if k_last - k_first >= nth:
            k_first = 0
            k_last = nth - 1
    elif c != 0.0:
        k_last = -1
    for kk in range(k_first, k_last + 1):
        b = kk % nth
        ray = b * nz + i
        cntb = rcnt[ray]
        if cntb == 0:
            continue
        e0 = b * dth
        e1 = (b + 1) * dth
        # s range of the bin on the line
        if c == 0.0:
            # the line through the axis: s >= 0 at theta = 0 (x ray) / pi / 2 (y ray), s < 0 opposite
            pos_angle = 0.0 if fam == 0 else 0.5 * math.pi
            neg_angle = math.pi if fam == 0 else 1.5 * math.pi
            if e0 <= pos_angle < e1:
                sa = 0.0
                sb = 1e300
            elif e0 <= neg_angle < e1:
                sa = -1e300
                sb = 0.0
            else:
                continue
        else:
            sin0 = trig[b, 0]
            cos0 = trig[b, 1]
            sin1 = trig[b + 1, 0]
            cos1 = trig[b + 1, 1]
            sinm = trig[b, 2]
            cosm = trig[b, 3]
            if fam == 0:
                sn0 = sin0
                sn1 = sin1
                ok0 = sn0 != 0.0 and (sn0 > 0.0) == (c > 0.0)
                ok1 = sn1 != 0.0 and (sn1 > 0.0) == (c > 0.0)
                v0 = c * cos0 / sn0 if ok0 else 0.0
                v1 = c * cos1 / sn1 if ok1 else 0.0
            else:
                cs0 = cos0
                cs1 = cos1
                ok0 = cs0 != 0.0 and (cs0 > 0.0) == (c > 0.0)
                ok1 = cs1 != 0.0 and (cs1 > 0.0) == (c > 0.0)
                v0 = c * sin0 / cs0 if ok0 else 0.0
                v1 = c * sin1 / cs1 if ok1 else 0.0
            # the bin must touch the line's half plane: its middle direction, or one edge, on the line's side
            if fam == 0:
                mid_ok = (sinm > 0.0) == (c > 0.0) and sinm != 0.0
                s_mid = c * cosm / sinm if mid_ok else 0.0
            else:
                mid_ok = (cosm > 0.0) == (c > 0.0) and cosm != 0.0
                s_mid = c * sinm / cosm if mid_ok else 0.0
            if not ok0 and not ok1 and not mid_ok:
                continue
            if ok0 and ok1:
                sa = v0 if v0 < v1 else v1
                sb = v1 if v0 < v1 else v0
            elif ok0:
                # the other edge is beyond the line's direction: open to the side of the middle
                if s_mid >= v0:
                    sa = v0
                    sb = 1e300
                else:
                    sa = -1e300
                    sb = v0
            elif ok1:
                if s_mid >= v1:
                    sa = v1
                    sb = 1e300
                else:
                    sa = -1e300
                    sb = v1
            else:
                sa = -1e300
                sb = 1e300
        for k in range(cntb):
            r0 = rseg[ray, k, 0] * Q
            r1 = rseg[ray, k, 1] * Q
            if r1 <= abs(c):
                continue
            s1 = math.sqrt(r1 * r1 - c * c)
            ev_k = np.int64(rsev[ray, k, 1])
            if r0 <= abs(c):
                pa = -s1 if -s1 > sa else sa
                pb = s1 if s1 < sb else sb
                m = _add_piece(lo_t, hi_t, el, eh, m, pa - s_origin, pb - s_origin, ev_k, ev_k, 0, out_lo,
                               out_hi, out_elo, out_ehi, cap)
            else:
                s0 = math.sqrt(r0 * r0 - c * c)
                pa = s0 if s0 > sa else sa
                pb = s1 if s1 < sb else sb
                m = _add_piece(lo_t, hi_t, el, eh, m, pa - s_origin, pb - s_origin, ev_k, ev_k, 0, out_lo,
                               out_hi, out_elo, out_ehi, cap)
                pa = -s1 if -s1 > sa else sa
                pb = -s0 if -s0 < sb else sb
                m = _add_piece(lo_t, hi_t, el, eh, m, pa - s_origin, pb - s_origin, ev_k, ev_k, 0, out_lo,
                               out_hi, out_elo, out_ehi, cap)
            if m < 0:
                return -1
    for p in range(1, m):
        a = lo_t[p]
        bb = hi_t[p]
        ea = el[p]
        eb = eh[p]
        q = p - 1
        while q >= 0 and lo_t[q] > a:
            lo_t[q + 1] = lo_t[q]
            hi_t[q + 1] = hi_t[q]
            el[q + 1] = el[q]
            eh[q + 1] = eh[q]
            q -= 1
        lo_t[q + 1] = a
        hi_t[q + 1] = bb
        el[q + 1] = ea
        eh[q + 1] = eb
    for p in range(m):
        a = lo_t[p] if lo_t[p] > t_lo else t_lo
        bb = hi_t[p] if hi_t[p] < t_hi else t_hi
        if bb <= a:
            continue
        qa = np.int64(np.floor(a / Q + 0.5))
        qb = np.int64(np.floor(bb / Q + 0.5))
        if n > 0 and qa <= out_hi[n - 1]:
            if qb > out_hi[n - 1]:
                out_hi[n - 1] = qb
                out_ehi[n - 1] = eh[p]
        else:
            if n >= cap:
                return -1
            out_lo[n] = qa
            out_hi[n] = qb
            out_elo[n] = el[p]
            out_ehi[n] = eh[p]
            n += 1
    return n


@njit(F2(i8, f8), **OPTS)
def bin_trig(nth, dth):
    """[nth + 1, 4]: sin, cos of the bin edges b dth (rows 0..nth) and of the bin middles 0.5 (b dth + (b + 1)
    dth) (columns 2, 3 of rows 0..nth-1), evaluated exactly as polar_to_ray's per-bin expressions."""
    out = np.zeros((nth + 1, 4))
    for b in range(nth + 1):
        e0 = b * dth
        out[b, 0] = math.sin(e0)
        out[b, 1] = math.cos(e0)
        if b < nth:
            mid = 0.5 * (e0 + (b + 1) * dth)
            out[b, 2] = math.sin(mid)
            out[b, 3] = math.cos(mid)
    return out


@njit(i8(FIELD_T, POLAR_T, i8, i8, i8, i8, F1, f8, f8, TARGET_T, f8, i8, SCRATCH_T, I1, I1, I1, I1, F1, F1, I1, I1,
          F2, F1), **OPTS)
def _rebuild_span(f, pf, lvl, fam, iu, iv, p, t_lo, t_hi, tgt, gtol, sev_id, sc, out_lo, out_hi, out_el, out_eh, w_lo,
                  w_hi, w_el, w_eh, trig, gacc):
    """One span [t_lo, t_hi) (metres from the ray's start p) of ray (fam, iu, iv) becomes the polar field's: the old
    material outside it stays. Material it removes that the target holds adds to gacc [depth, x, y, z, volume] (the
    deepest point). Returns 0, 1 on pool overflow, 2 when a scratch list was full (the ray is left)."""
    a = f.lv_cell_m[lvl]
    o = f.lv_origin_m[lvl]
    check = tgt.info[0] == 1
    u = 1 if fam == 0 else 0
    v = 2 if fam < 2 else 1
    m = polar_to_ray(pf, p[0], p[1], p[2], fam, t_lo, t_hi, out_lo, out_hi, out_el, out_eh, A.MAX_SEG,
                     w_lo, w_hi, w_el, w_eh, trig)
    if m < 0:
        f.info[A.FI_SCRATCH_OVF] = 1
        return 2
    ray = ray_index(f, lvl, fam, iu, iv)
    n = ray_load(f, ray, sc.lo, sc.hi, sc.elo, sc.ehi)
    if n < 0:
        f.info[A.FI_SCRATCH_OVF] = 1
        return 2
    qa = np.int64(np.floor(t_lo / Q + 0.5))
    qb = np.int64(np.floor(t_hi / Q + 0.5))
    # new list: old outside [qa, qb) + polar inside
    k = 0
    for s in range(n):
        lo = np.int64(sc.lo[s])
        hi = np.int64(sc.hi[s])
        if lo < qa:
            e = hi if hi < qa else qa
            sc.nlo[k] = lo
            sc.nhi[k] = e
            sc.nelo[k] = sc.elo[s]
            sc.nehi[k] = sc.ehi[s] if hi <= qa else sev_id
            k += 1
    for s in range(m):
        if k > 0 and out_lo[s] <= sc.nhi[k - 1]:
            if out_hi[s] > sc.nhi[k - 1]:
                sc.nhi[k - 1] = out_hi[s]
                sc.nehi[k - 1] = out_eh[s]
        else:
            sc.nlo[k] = out_lo[s]
            sc.nhi[k] = out_hi[s]
            sc.nelo[k] = out_el[s]
            sc.nehi[k] = out_eh[s]
            k += 1
    for s in range(n):
        lo = np.int64(sc.lo[s])
        hi = np.int64(sc.hi[s])
        if hi > qb:
            st = lo if lo > qb else qb
            if k > 0 and st <= sc.nhi[k - 1]:
                sc.nhi[k - 1] = hi
                sc.nehi[k - 1] = sc.ehi[s]
            else:
                sc.nlo[k] = st
                sc.nhi[k] = hi
                sc.nelo[k] = sc.elo[s] if lo >= qb else sev_id
                sc.nehi[k] = sc.ehi[s]
                k += 1
    # identical? skip
    same = k == n
    if same:
        for s in range(n):
            if sc.nlo[s] != sc.lo[s] or sc.nhi[s] != sc.hi[s]:
                same = False
                break
    if same:
        return 0
    # removed material (old minus new) against the target
    if check:
        for s in range(n):
            lo = np.int64(sc.lo[s])
            hi = np.int64(sc.hi[s])
            cur = lo
            for q in range(k):
                nlo_ = np.int64(sc.nlo[q])
                nhi_ = np.int64(sc.nhi[q])
                if nhi_ <= cur:
                    continue
                if nlo_ >= hi:
                    break
                if nlo_ > cur:
                    ov, first = target_overlap(tgt.start, tgt.seg, ray, cur, nlo_)
                    if ov * Q > gtol:
                        gacc[4] += ov * Q * a * a / 3.0
                        if ov * Q > gacc[0]:
                            gacc[0] = ov * Q
                            gacc[1 + fam] = o[fam] + 0.5 * (cur + nlo_) * Q
                            gacc[1 + u] = p[u]
                            gacc[1 + v] = p[v]
                cur = nhi_ if nhi_ > cur else cur
            if cur < hi:
                ov, first = target_overlap(tgt.start, tgt.seg, ray, cur, hi)
                if ov * Q > gtol:
                    gacc[4] += ov * Q * a * a / 3.0
                    if ov * Q > gacc[0]:
                        gacc[0] = ov * Q
                        gacc[1 + fam] = o[fam] + 0.5 * (cur + hi) * Q
                        gacc[1 + u] = p[u]
                        gacc[1 + v] = p[v]
    if ray_store(f, ray, sc.nlo, sc.nhi, sc.nelo, sc.nehi, k) < 0:
        return 1
    for s in range(n):
        sc.plo[s] = sc.lo[s]
        sc.phi[s] = sc.hi[s]
    brick_account(f, lvl, fam, iu, iv, sc.plo, sc.phi, n, -1)
    for s in range(k):
        sc.plo[s] = sc.nlo[s]
        sc.phi[s] = sc.nhi[s]
    brick_account(f, lvl, fam, iu, iv, sc.plo, sc.phi, k, 1)
    f.info[A.FI_VERSION] += 1
    return 0


@njit(i8(FIELD_T, POLAR_T, i8, f8, f8, f8, f8, TARGET_T, f8, EVENTS_T, i8, i8, SCRATCH_T), **OPTS)
def rebuild_tridexel(f, pf, lvl, zlo, zhi, r_lo, r_hi, tgt, gtol, ev, tick, sev_id, sc):
    """Replace the material of every ray of level ``lvl`` inside the polar region's annulus r_lo <= r <= r_hi
    (r_hi at most the polar field's largest radius), z in [zlo, zhi], by the polar field's (exact: per bin and row):
    z rays whose radius is in the annulus, the spans of x and y rays inside it. The cuts change only the radii their
    insert reaches (polar_cut records them, PI_TOUCH_* / PI_OPTOUCH_*): rebuilding the whole disc replaced the stock's
    exact rays near unturned features (the lathe-shaft bar's centre hole) by their polar sampling and booked the
    difference as removed part material (gouge errors of 7-292 um at every revolution's sync). Material the rebuild
    removes that the target holds is a gouge (one aggregated event). Returns 0, 1 on pool overflow (the rays already
    done stay done: call again after growing the pool)."""
    g = pf.geom
    a = f.lv_cell_m[lvl]
    o = f.lv_origin_m[lvl]
    rmax = g[A.PG_NR] * g[A.PG_DR]
    if r_hi > rmax:
        r_hi = rmax
    if r_lo < 0.0:
        r_lo = 0.0
    if r_hi <= r_lo:
        return 0
    status = 0
    out_lo = np.empty(A.MAX_SEG, np.int64)
    out_hi = np.empty(A.MAX_SEG, np.int64)
    out_el = np.empty(A.MAX_SEG, np.int64)
    out_eh = np.empty(A.MAX_SEG, np.int64)
    w_lo = np.empty(8192)
    w_hi = np.empty(8192)
    w_el = np.empty(8192, np.int64)
    w_eh = np.empty(8192, np.int64)
    p = np.empty(3)
    trig = bin_trig(np.int64(g[A.PG_NTHETA]), g[A.PG_DTHETA])
    gacc = np.zeros(5)
    for fam in range(3):
        u = 1 if fam == 0 else 0
        v = 2 if fam < 2 else 1
        lq = (f.lv_dims[lvl, fam] - 1) * f.lv_cell_q[lvl]
        for iu in range(f.lv_dims[lvl, u]):
            for iv in range(f.lv_dims[lvl, v]):
                p[fam] = o[fam]
                p[u] = o[u] + iu * a
                p[v] = o[v] + iv * a
                if fam == 2:
                    rr = p[0] * p[0] + p[1] * p[1]
                    if rr > r_hi * r_hi or rr < r_lo * r_lo:
                        continue
                    n_span = 1
                    s0a = zlo - o[2]
                    s1a = zhi - o[2]
                    s0b = 0.0
                    s1b = 0.0
                else:
                    if p[2] < zlo or p[2] > zhi:
                        continue
                    across = p[1] if fam == 0 else p[0]
                    if abs(across) >= r_hi:
                        continue
                    h1 = math.sqrt(r_hi * r_hi - across * across)
                    if r_lo > abs(across):
                        # the annulus cuts the line twice: the spans on either side of the inner circle
                        h0 = math.sqrt(r_lo * r_lo - across * across)
                        n_span = 2
                        s0a = -h1 - o[fam]
                        s1a = -h0 - o[fam]
                        s0b = h0 - o[fam]
                        s1b = h1 - o[fam]
                    else:
                        n_span = 1
                        s0a = -h1 - o[fam]
                        s1a = h1 - o[fam]
                        s0b = 0.0
                        s1b = 0.0
                for k_span in range(n_span):
                    t_lo = s0a if k_span == 0 else s0b
                    t_hi = s1a if k_span == 0 else s1b
                    if t_lo < 0.0:
                        t_lo = 0.0
                    if t_hi > lq * Q:
                        t_hi = lq * Q
                    if t_hi <= t_lo:
                        continue
                    st = _rebuild_span(f, pf, lvl, fam, iu, iv, p, t_lo, t_hi, tgt, gtol, sev_id, sc, out_lo, out_hi,
                                       out_el, out_eh, w_lo, w_hi, w_el, w_eh, trig, gacc)
                    if st == 1:
                        status = 1
    if gacc[0] > 0.0:
        i = ev.n[0]
        if i < ev.rec.shape[0]:
            ev.rec[i, A.EC_KIND] = A.EV_GOUGE
            ev.rec[i, A.EC_TICK] = tick
            ev.rec[i, A.EC_CONTACT] = -1
            ev.rec[i, A.EC_X] = gacc[1]
            ev.rec[i, A.EC_Y] = gacc[2]
            ev.rec[i, A.EC_Z] = gacc[3]
            ev.rec[i, A.EC_DEPTH] = gacc[0]
            ev.rec[i, A.EC_VOLUME] = gacc[4]
            ev.rec[i, A.EC_PART] = -1
            ev.rec[i, A.EC_SEV_ID] = sev_id
            ev.rec[i, A.EC_LEVEL] = lvl
            ev.rec[i, A.EC_FAMILY] = -1
            ev.n[0] = i + 1
        else:
            ev.n[1] = 1
    return status


@njit(void(POLAR_T, i8), **OPTS)
def refresh_zmax(pf, dummy):
    """zmax_q[b * nblk + blk] = largest outer radius (quanta) of the rows of each block in each bin."""
    nth = np.int64(pf.geom[A.PG_NTHETA])
    nz = np.int64(pf.geom[A.PG_NZ])
    nblk = (nz + PB - 1) // PB
    for b in range(nth):
        for blk in range(nblk):
            _rescan_block(pf, b, blk)


KERNELS = (_chain, _chain_w, _x_range, _z_range, _z_range_w, insert_x_range, insert_z_range, _cut_inline, _xhi_max,
           _xhi_max_w, insert_xhi_max, _rescan, _rescan_block, _q_of, _rad, _polar_rows, _polar_axial, polar_cut,
           polar_radius, polar_radii, polar_q2_sum,
           _add_piece, polar_to_ray, bin_trig, rebuild_tridexel, refresh_zmax)
