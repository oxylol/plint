"""Array-level primitives of the tri-dexel field, inlined into the kernels (docs/models/removal.md "Kernels").

A NamedTuple argument costs reference counting of every member array at each call boundary (measured about
7 ns per member and call on this box), which would dominate the per-ray and per-sample work. The hot
kernels therefore unpack the DexelField once and call these helpers with plain arrays; every helper is
``inline="always"`` (no call boundary). Their FIELD_T-taking twins in k_ray / k_sd serve Python and tests.

Arrays of a field, in the order the helpers take them:
``cnt, seg, sev, link, pseg, psev, pnext, info`` (rays and pool) and ``lv_cell_m, lv_origin_m, lv_dims,
lv_cell_q, lv_ray0, lv_bdims, lv_brick0`` (levels), ``bmat, bdirty, bconn, bver`` (bricks).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, types

from . import abi as A
from .nbt import F1, F2, I1, I2, I4_1, I4_3, INLINE, b1, f8, i8, void

K = A.K_INLINE
BR = A.BRICK
Q = A.QUANTUM_M
SPREAD_CELLS = 2.5
B1_ = types.int8[::1]


# --------------------------------------------------------------------------- indices
@njit(i8(I2, I2, i8, i8, i8, i8), **INLINE)
def rindex(lv_ray0, lv_dims, lvl, fam, iu, iv):
    v = 2 if fam < 2 else 1
    return lv_ray0[lvl, fam] + iu * lv_dims[lvl, v] + iv


@njit(i8(I1, I2, i8, i8, i8, i8), **INLINE)
def bindex(lv_brick0, lv_bdims, lvl, bx, by, bz):
    return lv_brick0[lvl] + (bx * lv_bdims[lvl, 1] + by) * lv_bdims[lvl, 2] + bz


# --------------------------------------------------------------------------- lookups
@njit(types.UniTuple(i8, 3)(I4_1, I4_3, I4_1, I4_3, I4_1, i8, f8), **INLINE)
def nearest(cnt, seg, link, pseg, pnext, ray, q):
    """(inside, nearest endpoint (quanta), has_endpoint) of a ray at q (ray >= 0: indexed unsigned, without
    Numba's negative-index wrap-around)."""
    ur = np.uint64(ray)
    n = np.int64(cnt[ur])
    if n == 0:
        return np.int64(0), np.int64(0), np.int64(0)
    if n == 1:
        # one segment (the common case): the loop below in closed form (a tie keeps the lower end)
        lo = np.int64(seg[ur, 0, 0])
        hi = np.int64(seg[ur, 0, 1])
        ins1 = np.int64(1) if (q >= lo and q < hi) else np.int64(0)
        return ins1, (lo if abs(q - lo) <= abs(q - hi) else hi), np.int64(1)
    best = -1.0
    be = np.int64(0)
    ins = np.int64(0)
    b = np.int64(link[ur])
    s = 0
    for k in range(n):
        if k < K:
            lo = np.int64(seg[ur, k, 0])
            hi = np.int64(seg[ur, k, 1])
        else:
            if s == K:
                b = np.int64(pnext[b])
                s = 0
            lo = np.int64(pseg[b, s, 0])
            hi = np.int64(pseg[b, s, 1])
            s += 1
        if q >= lo and q < hi:
            ins = 1
        d = abs(q - lo)
        if best < 0.0 or d < best:
            best = d
            be = lo
        d = abs(q - hi)
        if d < best:
            best = d
            be = hi
        if lo > q + best:
            break
    return ins, be, np.int64(1)


@njit(b1(I4_1, I4_3, I4_1, I4_3, I4_1, i8, f8), **INLINE)
def inside(cnt, seg, link, pseg, pnext, ray, q):
    n = np.int64(cnt[ray])
    b = np.int64(link[ray])
    s = 0
    for k in range(n):
        if k < K:
            lo = seg[ray, k, 0]
            hi = seg[ray, k, 1]
        else:
            if s == K:
                b = np.int64(pnext[b])
                s = 0
            lo = pseg[b, s, 0]
            hi = pseg[b, s, 1]
            s += 1
        if q < lo:
            return False
        if q < hi:
            return True
    return False


@njit(types.UniTuple(i8, 2)(I4_1, I4_3, I4_1, I4_3, I4_1, i8, i8, i8), **INLINE)
def overlap(cnt, seg, link, pseg, pnext, ray, q0, q1):
    """(material length in [q0, q1), segments touching it)."""
    n = np.int64(cnt[ray])
    b = np.int64(link[ray])
    s = 0
    total = np.int64(0)
    touched = np.int64(0)
    for k in range(n):
        if k < K:
            lo = np.int64(seg[ray, k, 0])
            hi = np.int64(seg[ray, k, 1])
        else:
            if s == K:
                b = np.int64(pnext[b])
                s = 0
            lo = np.int64(pseg[b, s, 0])
            hi = np.int64(pseg[b, s, 1])
            s += 1
        if lo >= q1:
            break
        a = lo if lo > q0 else q0
        c = hi if hi < q1 else q1
        if c > a:
            total += c - a
            touched += 1
    return total, touched


@njit(types.UniTuple(i8, 3)(I4_1, I4_3, I4_1, I4_3, I4_1, i8, i8, i8), **INLINE)
def extent(cnt, seg, link, pseg, pnext, ray, q0, q1):
    """(material length in [q0, q1), first and last quantum + 1 of it): overlap with the material's hull."""
    n = np.int64(cnt[ray])
    b = np.int64(link[ray])
    s = 0
    total = np.int64(0)
    first = np.int64(0)
    last = np.int64(0)
    for k in range(n):
        if k < K:
            lo = np.int64(seg[ray, k, 0])
            hi = np.int64(seg[ray, k, 1])
        else:
            if s == K:
                b = np.int64(pnext[b])
                s = 0
            lo = np.int64(pseg[b, s, 0])
            hi = np.int64(pseg[b, s, 1])
            s += 1
        if lo >= q1:
            break
        a = lo if lo > q0 else q0
        c = hi if hi < q1 else q1
        if c > a:
            if total == 0:
                first = a
            total += c - a
            last = c
    return total, first, last


# --------------------------------------------------------------------------- load / store with the pool
@njit(i8(i8), **INLINE)
def blocks_for(n):
    if n <= K:
        return 0
    return (n - K + K - 1) // K


@njit(i8(I4_1, I4_3, I4_3, I4_1, I4_3, I4_3, I4_1, i8, I4_1, I4_1, I4_1, I4_1), **INLINE)
def load(cnt, seg, sev, link, pseg, psev, pnext, ray, lo, hi, elo, ehi):
    n = np.int64(cnt[ray])
    if n > lo.shape[0]:
        return -1
    m = n if n < K else K
    for k in range(m):
        lo[k] = seg[ray, k, 0]
        hi[k] = seg[ray, k, 1]
        elo[k] = sev[ray, k, 0]
        ehi[k] = sev[ray, k, 1]
    b = np.int64(link[ray])
    k = m
    while k < n and b >= 0:
        for s in range(K):
            if k >= n:
                break
            lo[k] = pseg[b, s, 0]
            hi[k] = pseg[b, s, 1]
            elo[k] = psev[b, s, 0]
            ehi[k] = psev[b, s, 1]
            k += 1
        b = np.int64(pnext[b])
    return n


@njit(i8(I4_3, I4_3, I4_1, I1), **INLINE)
def _alloc(pseg, psev, pnext, info):
    b = info[A.FI_FREE_HEAD]
    info[A.FI_FREE_HEAD] = pnext[b]
    pnext[b] = -1
    info[A.FI_N_FREE] -= 1
    used = info[A.FI_N_BLOCKS] - info[A.FI_N_FREE]
    if used > info[A.FI_PEAK_USED]:
        info[A.FI_PEAK_USED] = used
    return b


@njit(void(I4_3, I4_3, I4_1, I1, i8), **INLINE)
def _free(pseg, psev, pnext, info, b):
    for s in range(K):
        pseg[b, s, 0] = 0
        pseg[b, s, 1] = 0
        psev[b, s, 0] = 0
        psev[b, s, 1] = 0
    pnext[b] = info[A.FI_FREE_HEAD]
    info[A.FI_FREE_HEAD] = b
    info[A.FI_N_FREE] += 1


@njit(i8(I4_1, I4_3, I4_3, I4_1, I4_3, I4_3, I4_1, I1, i8, I4_1, I4_1, I4_1, I4_1, i8), **INLINE)
def store(cnt, seg, sev, link, pseg, psev, pnext, info, ray, lo, hi, elo, ehi, n):
    """Write n segments to a ray (inline, then its pool chain); -1 when the pool is short (unchanged)."""
    old = np.int64(cnt[ray])
    have = blocks_for(old)
    need = blocks_for(n)
    if need > have and need - have > info[A.FI_N_FREE]:
        info[A.FI_OVERFLOW] = 1
        return -1
    m = n if n < K else K
    for k in range(K):
        if k < m:
            seg[ray, k, 0] = lo[k]
            seg[ray, k, 1] = hi[k]
            sev[ray, k, 0] = elo[k]
            sev[ray, k, 1] = ehi[k]
        else:
            seg[ray, k, 0] = 0
            seg[ray, k, 1] = 0
            sev[ray, k, 0] = 0
            sev[ray, k, 1] = 0
    b = np.int64(link[ray])
    if need == 0:
        while b >= 0:
            nb = np.int64(pnext[b])
            _free(pseg, psev, pnext, info, b)
            b = nb
        link[ray] = -1
    else:
        if b < 0:
            b = _alloc(pseg, psev, pnext, info)
            link[ray] = b
        k = K
        used = 0
        while True:
            for s in range(K):
                if k < n:
                    pseg[b, s, 0] = lo[k]
                    pseg[b, s, 1] = hi[k]
                    psev[b, s, 0] = elo[k]
                    psev[b, s, 1] = ehi[k]
                    k += 1
                else:
                    pseg[b, s, 0] = 0
                    pseg[b, s, 1] = 0
                    psev[b, s, 0] = 0
                    psev[b, s, 1] = 0
            used += 1
            nb = np.int64(pnext[b])
            if used == need:
                pnext[b] = -1
                while nb >= 0:
                    nn = np.int64(pnext[nb])
                    _free(pseg, psev, pnext, info, nb)
                    nb = nn
                break
            if nb < 0:
                nb = _alloc(pseg, psev, pnext, info)
                pnext[b] = nb
            b = nb
    cnt[ray] = n
    if n > info[A.FI_MAX_SEG]:
        info[A.FI_MAX_SEG] = n
    return 0


# --------------------------------------------------------------------------- bricks
@njit(i8(i8, i8, f8), **INLINE)
def floordiv(a, b, binv):
    """a // b for b > 0 (binv = 1 / b) without an integer division: the float quotient is within one of the
    exact one for |a| < 2^52, and the integer corrections make it exact."""
    q = np.int64(math.floor(a * binv))
    if q * b > a:
        q -= 1
    elif (q + 1) * b <= a:
        q += 1
    return q


@njit(void(I1, I2, I1, B1_, B1_, I4_1, i8, i8, i8, i8, i8, i8), **INLINE)
def mark(lv_cell_q, lv_bdims, lv_brick0, bdirty, bconn, bver, lvl, fam, iu, iv, q0, q1):
    """Mark changed the bricks whose closed node ranges hold the ray part [q0, q1]."""
    u = 1 if fam == 0 else 0
    v = 2 if fam < 2 else 1
    bdu = lv_bdims[lvl, u]
    bdv = lv_bdims[lvl, v]
    bdw = lv_bdims[lvl, fam]
    bq = BR * lv_cell_q[lvl]
    binv = 1.0 / bq
    bu1 = min(iu // BR, bdu - 1)
    bu0 = bu1 - 1 if (iu % BR == 0 and iu > 0 and iu // BR <= bdu - 1) else bu1
    bv1 = min(iv // BR, bdv - 1)
    bv0 = bv1 - 1 if (iv % BR == 0 and iv > 0 and iv // BR <= bdv - 1) else bv1
    w0 = floordiv(q0, bq, binv)
    if q0 - w0 * bq == 0 and w0 > 0:
        w0 -= 1
    w1 = floordiv(q1, bq, binv)
    if w0 < 0:
        w0 = 0
    if w1 > bdw - 1:
        w1 = bdw - 1
    if w0 > bdw - 1:
        w0 = bdw - 1
    for bu in range(bu0, bu1 + 1):
        for bv in range(bv0, bv1 + 1):
            for bw in range(w0, w1 + 1):
                if fam == 0:
                    bx = bw
                    by = bu
                    bz = bv
                elif fam == 1:
                    bx = bu
                    by = bw
                    bz = bv
                else:
                    bx = bu
                    by = bv
                    bz = bw
                idx = lv_brick0[lvl] + (bx * lv_bdims[lvl, 1] + by) * lv_bdims[lvl, 2] + bz
                bdirty[idx] = 1
                bconn[idx] = 1
                bver[idx] += 1


@njit(void(I1, I2, I1, I2, B1_, B1_, I4_1, i8, i8, i8, i8, I1, I1, i8, i8), **INLINE)
def account(lv_cell_q, lv_bdims, lv_brick0, bmat, bdirty, bconn, bver, lvl, fam, iu, iv, plo, phi, npc, sign):
    """Book pieces of one ray into the owning bricks' material (sign -1 removed, +1 added) and mark them."""
    u = 1 if fam == 0 else 0
    v = 2 if fam < 2 else 1
    bq = BR * lv_cell_q[lvl]
    binv = 1.0 / bq
    bdw = lv_bdims[lvl, fam]
    bu = min(iu // BR, lv_bdims[lvl, u] - 1)
    bv = min(iv // BR, lv_bdims[lvl, v] - 1)
    for p in range(npc):
        p0 = plo[p]
        p1 = phi[p]
        if p1 <= p0:
            continue
        w0 = min(floordiv(p0, bq, binv), bdw - 1)
        w1 = min(floordiv(p1 - 1, bq, binv), bdw - 1)
        for bw in range(w0, w1 + 1):
            s0 = p0 if bw == w0 else bw * bq
            s1 = p1 if bw == w1 else (bw + 1) * bq
            if fam == 0:
                bx = bw
                by = bu
                bz = bv
            elif fam == 1:
                bx = bu
                by = bw
                bz = bv
            else:
                bx = bu
                by = bv
                bz = bw
            idx = lv_brick0[lvl] + (bx * lv_bdims[lvl, 1] + by) * lv_bdims[lvl, 2] + bz
            bmat[idx, fam] += sign * (s1 - s0)
        mark(lv_cell_q, lv_bdims, lv_brick0, bdirty, bconn, bver, lvl, fam, iu, iv, p0, p1)


@njit(i8(I2, I1, I2, B1_, B1_, I4_1, i8, i8, i8, i8, i8, i8, i8, i8, f8), **INLINE)
def book_piece(lv_bdims, lv_brick0, bmat, bdirty, bconn, bver, lvl, fam, iu, iv, p0, p1, sign, bq, binv):
    """account of the one piece [p0, p1) (p1 > p0; bq = BRICK cells in quanta, binv = 1 / bq): the brick material
    and the marks of account(..., npc=1) (its mark of [p0, p1] included), sharing the brick indices; one brick
    (the common case) is updated directly. Returns that brick's index, or -1 when several were touched."""
    u = 1 if fam == 0 else 0
    v = 2 if fam < 2 else 1
    bdu = lv_bdims[lvl, u]
    bdv = lv_bdims[lvl, v]
    bdw = lv_bdims[lvl, fam]
    # iu, iv >= 0 (node indices): unsigned division and remainder by the constant are the floor ones (shifts)
    qu = np.int64(np.uint64(iu) // np.uint64(BR))
    qv = np.int64(np.uint64(iv) // np.uint64(BR))
    ru = np.int64(np.uint64(iu) % np.uint64(BR))
    rv = np.int64(np.uint64(iv) % np.uint64(BR))
    bu1 = min(qu, bdu - 1)
    bv1 = min(qv, bdv - 1)
    f0 = floordiv(p0, bq, binv)
    g1 = floordiv(p1 - 1, bq, binv)
    w0 = min(f0, bdw - 1)
    w1 = min(g1, bdw - 1)
    bu0 = bu1 - 1 if (ru == 0 and iu > 0 and qu <= bdu - 1) else bu1
    bv0 = bv1 - 1 if (rv == 0 and iv > 0 and qv <= bdv - 1) else bv1
    m0 = f0
    if p0 - f0 * bq == 0 and m0 > 0:
        m0 -= 1
    m1 = g1 + 1 if p1 == (g1 + 1) * bq else g1
    if m0 < 0:
        m0 = 0
    if m1 > bdw - 1:
        m1 = bdw - 1
    if m0 > bdw - 1:
        m0 = bdw - 1
    b0 = lv_brick0[lvl]
    nb1 = lv_bdims[lvl, 1]
    nb2 = lv_bdims[lvl, 2]
    if bu0 == bu1 and bv0 == bv1 and w0 == w1 and m0 == w0 and m1 == w0:
        if fam == 0:
            idx = b0 + (w0 * nb1 + bu1) * nb2 + bv1
        elif fam == 1:
            idx = b0 + (bu1 * nb1 + w0) * nb2 + bv1
        else:
            idx = b0 + (bu1 * nb1 + bv1) * nb2 + w0
        bmat[idx, fam] += sign * (p1 - p0)
        bdirty[idx] = 1
        bconn[idx] = 1
        bver[idx] += 1
        return idx
    for bw in range(w0, w1 + 1):
        s0 = p0 if bw == w0 else bw * bq
        s1 = p1 if bw == w1 else (bw + 1) * bq
        if fam == 0:
            idx = b0 + (bw * nb1 + bu1) * nb2 + bv1
        elif fam == 1:
            idx = b0 + (bu1 * nb1 + bw) * nb2 + bv1
        else:
            idx = b0 + (bu1 * nb1 + bv1) * nb2 + bw
        bmat[idx, fam] += sign * (s1 - s0)
    for bu in range(bu0, bu1 + 1):
        for bv in range(bv0, bv1 + 1):
            for bw in range(m0, m1 + 1):
                if fam == 0:
                    idx = b0 + (bw * nb1 + bu) * nb2 + bv
                elif fam == 1:
                    idx = b0 + (bu * nb1 + bw) * nb2 + bv
                else:
                    idx = b0 + (bu * nb1 + bv) * nb2 + bw
                bdirty[idx] = 1
                bconn[idx] = 1
                bver[idx] += 1
    return np.int64(-1)


# --------------------------------------------------------------------------- interpolated boundary
@njit(i8(F2, F2, F1, i8, f8, f8, f8), **INLINE)
def level_at(lv_box_lo, lv_box_hi, lv_cell_m, nl, x, y, z):
    best = 0
    best_a = lv_cell_m[0]
    for l in range(1, nl):
        if (x >= lv_box_lo[l, 0] and x <= lv_box_hi[l, 0] and y >= lv_box_lo[l, 1] and y <= lv_box_hi[l, 1]
                and z >= lv_box_lo[l, 2] and z <= lv_box_hi[l, 2]):
            if lv_cell_m[l] < best_a:
                best = l
                best_a = lv_cell_m[l]
    return best


@njit(f8(I2, I1, I2, I4_1, I4_3, I4_1, I4_3, I4_1, i8, i8, f8, f8, f8, F1), **INLINE)
def sd_pre(lv_dims, lv_cell_q, lv_ray0, cnt, seg, link, pseg, pnext, lvl, fam, fu, fv, q, out):
    """sd from the point's lattice coordinates across the family (fu, fv: cells from the level origin) and
    along it (q: quanta), each computed as sd computes it (so callers can share the divisions)."""
    v = 2 if fam < 2 else 1
    u = 1 if fam == 0 else 0
    nu = lv_dims[lvl, u]
    nv = lv_dims[lvl, v]
    iu = int(math.floor(fu))
    iv = int(math.floor(fv))
    if iu < 0:
        iu = 0
    if iu > nu - 2:
        iu = nu - 2
    if iv < 0:
        iv = 0
    if iv > nv - 2:
        iv = nv - 2
    al = fu - iu
    be = fv - iv
    if al < 0.0:
        al = 0.0
    if al > 1.0:
        al = 1.0
    if be < 0.0:
        be = 0.0
    if be > 1.0:
        be = 1.0
    far = SPREAD_CELLS * lv_cell_q[lvl]
    emin = 1e300
    emax = -1e300
    valid = 1
    n_in = 0
    n_out = 0
    min_abs = 1e300
    D = 0.0
    r0 = lv_ray0[lvl, fam]
    for c in range(4):
        du = c & 1
        dv = c >> 1
        ray = r0 + (iu + du) * nv + iv + dv
        ins, e, has = nearest(cnt, seg, link, pseg, pnext, ray, q)
        wgt = (al if du == 1 else 1.0 - al) * (be if dv == 1 else 1.0 - be)
        if has == 0:
            valid = 0
            n_out += 1
            D -= wgt * far
            continue
        d = abs(q - e)
        psi = d if ins == 1 else -d
        if ins == 1:
            n_in += 1
        else:
            n_out += 1
        if e < emin:
            emin = e
        if e > emax:
            emax = e
        if d > far:
            valid = 0
        if d < min_abs:
            min_abs = d
        D += wgt * psi
    spread = (emax - emin) if emax >= emin else far
    if spread > far:
        valid = 0
    out[0] = spread * Q
    out[1] = valid
    out[2] = 1.0 if n_in == 4 else (-1.0 if n_out == 4 else 0.0)
    out[3] = min_abs * Q
    return D * Q


@njit(f8(F1, F2, I2, I1, I2, I4_1, I4_3, I4_1, I4_3, I4_1, i8, i8, f8, f8, f8, F1), **INLINE)
def sd(lv_cell_m, lv_origin_m, lv_dims, lv_cell_q, lv_ray0, cnt, seg, link, pseg, pnext, lvl, fam, x, y, z, out):
    """D_fam at (x, y, z) (m, positive inside); out = [spread (m), valid, unanimous (+1 in, -1 out, 0),
    smallest |psi| (m) of the rays with endpoints]."""
    u = 1 if fam == 0 else 0
    v = 2 if fam < 2 else 1
    a = lv_cell_m[lvl]
    if fam == 0:
        pu = y
        pv = z
        pw = x
    elif fam == 1:
        pu = x
        pv = z
        pw = y
    else:
        pu = x
        pv = y
        pw = z
    fu = (pu - lv_origin_m[lvl, u]) / a
    fv = (pv - lv_origin_m[lvl, v]) / a
    q = (pw - lv_origin_m[lvl, fam]) / Q
    return sd_pre(lv_dims, lv_cell_q, lv_ray0, cnt, seg, link, pseg, pnext, lvl, fam, fu, fv, q, out)


@njit(f8(F1, F2, I2, I1, I2, I4_1, I4_3, I4_1, I4_3, I4_1, i8, i8, f8, f8, f8), **INLINE)
def sd_d(lv_cell_m, lv_origin_m, lv_dims, lv_cell_q, lv_ray0, cnt, seg, link, pseg, pnext, lvl, fam, x, y, z):
    """The D of sd (the same arithmetic) without its spread and validity outputs (no work array)."""
    u = 1 if fam == 0 else 0
    v = 2 if fam < 2 else 1
    a = lv_cell_m[lvl]
    if fam == 0:
        pu = y
        pv = z
        pw = x
    elif fam == 1:
        pu = x
        pv = z
        pw = y
    else:
        pu = x
        pv = y
        pw = z
    fu = (pu - lv_origin_m[lvl, u]) / a
    fv = (pv - lv_origin_m[lvl, v]) / a
    q = (pw - lv_origin_m[lvl, fam]) / Q
    nu = lv_dims[lvl, u]
    nv = lv_dims[lvl, v]
    iu = int(math.floor(fu))
    iv = int(math.floor(fv))
    if iu < 0:
        iu = 0
    if iu > nu - 2:
        iu = nu - 2
    if iv < 0:
        iv = 0
    if iv > nv - 2:
        iv = nv - 2
    al = fu - iu
    be = fv - iv
    if al < 0.0:
        al = 0.0
    if al > 1.0:
        al = 1.0
    if be < 0.0:
        be = 0.0
    if be > 1.0:
        be = 1.0
    far = SPREAD_CELLS * lv_cell_q[lvl]
    D = 0.0
    r0 = lv_ray0[lvl, fam]
    for c in range(4):
        du = c & 1
        dv = c >> 1
        ray = r0 + (iu + du) * nv + iv + dv
        ins, e, has = nearest(cnt, seg, link, pseg, pnext, ray, q)
        wgt = (al if du == 1 else 1.0 - al) * (be if dv == 1 else 1.0 - be)
        if has == 0:
            D -= wgt * far
            continue
        d = abs(q - e)
        psi = d if ins == 1 else -d
        D += wgt * psi
    return D * Q


@njit(f8(I2, I1, I2, I4_1, I4_3, I4_1, I4_3, I4_1, i8, i8, f8, f8, f8, f8, F1), **INLINE)
def sd_beats(lv_dims, lv_cell_q, lv_ray0, cnt, seg, link, pseg, pnext, lvl, fam, fu, fv, q, best_s, out):
    """sd_pre for a family that only matters when it is valid with a spread below best_s (m): the same D and
    out[0:2] as sd_pre when it completes; out[1] = 0 as soon as a ray shows that it cannot be (no endpoint, an
    endpoint farther than SPREAD_CELLS cells, or the endpoint spread already at best_s)."""
    v = 2 if fam < 2 else 1
    u = 1 if fam == 0 else 0
    nu = lv_dims[lvl, u]
    nv = lv_dims[lvl, v]
    iu = int(math.floor(fu))
    iv = int(math.floor(fv))
    if iu < 0:
        iu = 0
    if iu > nu - 2:
        iu = nu - 2
    if iv < 0:
        iv = 0
    if iv > nv - 2:
        iv = nv - 2
    al = fu - iu
    be = fv - iv
    if al < 0.0:
        al = 0.0
    if al > 1.0:
        al = 1.0
    if be < 0.0:
        be = 0.0
    if be > 1.0:
        be = 1.0
    far = SPREAD_CELLS * lv_cell_q[lvl]
    emin = 1e300
    emax = -1e300
    D = 0.0
    r0 = lv_ray0[lvl, fam]
    out[1] = 0.0
    for c in range(4):
        du = c & 1
        dv = c >> 1
        ray = r0 + (iu + du) * nv + iv + dv
        ins, e, has = nearest(cnt, seg, link, pseg, pnext, ray, q)
        if has == 0:
            return 0.0
        wgt = (al if du == 1 else 1.0 - al) * (be if dv == 1 else 1.0 - be)
        d = abs(q - e)
        if d > far:
            return 0.0
        psi = d if ins == 1 else -d
        if e < emin:
            emin = e
        if e > emax:
            emax = e
        if (emax - emin) * Q >= best_s:
            return 0.0
        D += wgt * psi
    spread = emax - emin
    if spread > far:
        return 0.0
    out[0] = spread * Q
    out[1] = 1.0
    return D * Q


@njit(types.Tuple((b1, f8, i8))(F1, I2, I1, I2, I4_1, I4_3, I4_1, I4_3, I4_1, i8, f8, f8, f8, f8, f8, f8, i8, F1),
      **INLINE)
def classify_pre(lv_cell_m, lv_dims, lv_cell_q, lv_ray0, cnt, seg, link, pseg, pnext, lvl, fx, fy, fz, qx, qy, qz, fam0,
                 work):
    """classify from the point's lattice coordinates: fx = (x - o_x) / a ... (cells) and qx = (x - o_x) / Q ...
    (quanta), the values sd computes for each family. Once a family is valid, the others are only followed
    while they can still beat its spread (sd_beats), and a valid zero spread decides at once (the choice is
    strict): the result of the full evaluation."""
    far = SPREAD_CELLS * lv_cell_m[lvl]
    best_f = -1
    best_s = 1e300
    best_d = 0.0
    any_f = fam0
    any_s = 1e300
    any_d = 0.0
    for idx in range(3):
        fam = fam0 if idx == 0 else ((fam0 + idx) % 3)
        if fam == 0:
            fu = fy
            fv = fz
            q = qx
        elif fam == 1:
            fu = fx
            fv = fz
            q = qy
        else:
            fu = fx
            fv = fy
            q = qz
        if idx == 0:
            d = sd_pre(lv_dims, lv_cell_q, lv_ray0, cnt, seg, link, pseg, pnext, lvl, fam, fu, fv, q, work)
            if work[2] != 0.0 and work[3] > far:
                return work[2] > 0.0, d, fam0
            if work[1] == 1.0:
                best_f = fam
                best_s = work[0]
                best_d = d
            any_s = work[0]
            any_d = d
        else:
            # validity first: an invalid family stops at its first ray that shows it (the smallest-spread choice
            # among invalid families is only needed when none is valid: then they are evaluated in full below)
            d = sd_beats(lv_dims, lv_cell_q, lv_ray0, cnt, seg, link, pseg, pnext, lvl, fam, fu, fv, q, best_s, work)
            if work[1] == 1.0 and work[0] < best_s:
                best_f = fam
                best_s = work[0]
                best_d = d
        if best_f >= 0 and best_s <= 0.0:
            return best_d > 0.0, best_d, best_f
    if best_f >= 0:
        return best_d > 0.0, best_d, best_f
    for idx in range(1, 3):
        fam = (fam0 + idx) % 3
        if fam == 0:
            fu = fy
            fv = fz
            q = qx
        elif fam == 1:
            fu = fx
            fv = fz
            q = qy
        else:
            fu = fx
            fv = fy
            q = qz
        d = sd_pre(lv_dims, lv_cell_q, lv_ray0, cnt, seg, link, pseg, pnext, lvl, fam, fu, fv, q, work)
        if work[0] < any_s:
            any_f = fam
            any_s = work[0]
            any_d = d
    return any_d > 0.0, any_d, any_f


@njit(types.Tuple((b1, f8, i8))(F1, F2, I2, I1, I2, I4_1, I4_3, I4_1, I4_3, I4_1, i8, f8, f8, f8, i8, F1),
      **INLINE)
def classify(lv_cell_m, lv_origin_m, lv_dims, lv_cell_q, lv_ray0, cnt, seg, link, pseg, pnext, lvl, x, y, z, fam0,
             work):
    """(inside, D of the deciding family, that family): family fam0 decides alone when its four rays agree
    and are all farther than SPREAD_CELLS cells from an endpoint; otherwise the valid family with the
    smallest endpoint spread, or the smallest spread when none is valid (classify_pre)."""
    a = lv_cell_m[lvl]
    fx = (x - lv_origin_m[lvl, 0]) / a
    fy = (y - lv_origin_m[lvl, 1]) / a
    fz = (z - lv_origin_m[lvl, 2]) / a
    qx = (x - lv_origin_m[lvl, 0]) / Q
    qy = (y - lv_origin_m[lvl, 1]) / Q
    qz = (z - lv_origin_m[lvl, 2]) / Q
    return classify_pre(lv_cell_m, lv_dims, lv_cell_q, lv_ray0, cnt, seg, link, pseg, pnext, lvl, fx, fy, fz, qx, qy, qz,
                        fam0, work)
