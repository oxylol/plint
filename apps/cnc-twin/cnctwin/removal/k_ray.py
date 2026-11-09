"""Ray-level kernels of the tri-dexel field: segment storage with the overflow pool, interval
subtraction and union in fixed point, brick bookkeeping, per-ray lookups (docs/models/removal.md
"Field", "Kernels").

A ray's segments are read into a scratch list (:func:`ray_load`), changed there, and written back by
:func:`ray_store`, which refuses the write (returns -1, sets ``FI_OVERFLOW``, leaves the ray as it was)
when the chain would need more pool blocks than are free: the caller grows the pool
(cnctwin.removal.field.grow_pool) and repeats the operation. Subtraction and union are idempotent per
ray, so a repeat after a partial pass gives the same field as one pass with a large enough pool.
"""

from __future__ import annotations

import numpy as np
from numba import njit, types

from . import abi as A
from . import k_core as C
from .nbt import FIELD_T, I1, I4_1, INLINE, OPTS, TARGET_T, b1, f8, i4, i8, void

K = A.K_INLINE
BR = A.BRICK


# --------------------------------------------------------------------------- pool (FIELD_T wrappers of k_core)
@njit(i8(i8), **OPTS)
def blocks_for(n):
    """Pool blocks a ray with n segments uses."""
    return C.blocks_for(n)


@njit(i8(FIELD_T), **OPTS)
def pool_alloc(f):
    return C._alloc(f.pseg, f.psev, f.pnext, f.info)


@njit(void(FIELD_T, i8), **OPTS)
def pool_free(f, b):
    C._free(f.pseg, f.psev, f.pnext, f.info, b)


# --------------------------------------------------------------------------- load / store
@njit(i8(FIELD_T, i8, I4_1, I4_1, I4_1, I4_1), **OPTS)
def ray_load(f, ray, lo, hi, elo, ehi):
    """Copy the segments of ``ray`` into the scratch lists; returns their number (-1: more than fit)."""
    return C.load(f.cnt, f.seg, f.sev, f.link, f.pseg, f.psev, f.pnext, ray, lo, hi, elo, ehi)


@njit(i8(FIELD_T, i8, I4_1, I4_1, I4_1, I4_1, i8), **OPTS)
def ray_store(f, ray, lo, hi, elo, ehi, n):
    """Write n segments to ``ray`` (inline first, then its pool chain). Returns 0, or -1 when the pool
    has too few free blocks (``FI_OVERFLOW`` set, the ray unchanged)."""
    return C.store(f.cnt, f.seg, f.sev, f.link, f.pseg, f.psev, f.pnext, f.info, ray, lo, hi, elo, ehi, n)


# --------------------------------------------------------------------------- interval algebra (quanta)
@njit(types.UniTuple(i8, 3)(i8, I4_1, I4_1, I4_1, I4_1, i8, I1, I1, i8, I4_1, I4_1, I4_1, I4_1, I1, I1), **INLINE)
def iv_subtract(n, lo, hi, elo, ehi, m, rlo, rhi, ev, nlo, nhi, nelo, nehi, plo, phi):
    """Segments minus sorted, disjoint removal intervals ``[rlo, rhi)``: (new count, removed pieces,
    removed length); new endpoints carry ``ev``. Count -1: a scratch list was too small.

    A piece of a segment the removal leaves that touches a removal interval and is shorter than A.SLIVER_Q goes
    with it (a slab-discretisation leftover between two passes; docs/models/removal.md "Slivers"): it joins the
    adjacent removed piece. Segments no interval touches are copied unchanged, however short."""
    cap = nlo.shape[0]
    pcap = plo.shape[0]
    sliver = np.int64(A.SLIVER_Q)
    nn = 0
    npc = 0
    removed = 0
    j = 0
    for i in range(n):
        a = np.int64(lo[i])
        b = np.int64(hi[i])
        while j < m and rhi[j] <= a:
            j += 1
        cur = a
        ce = elo[i]
        jj = j
        cut = False
        while jj < m and rlo[jj] < b:
            ra = rlo[jj]
            rb = rhi[jj]
            if rb <= cur or rb <= ra:
                jj += 1
                continue
            if ra > cur:
                if ra - cur < sliver:
                    # a sliver below this removal: removed with it (joined to the last piece when they touch)
                    if cut and npc > 0 and phi[npc - 1] == cur:
                        p0 = plo[npc - 1]
                        npc -= 1
                        removed -= phi[npc] - plo[npc]
                    else:
                        p0 = cur
                else:
                    if nn >= cap:
                        return np.int64(-1), np.int64(npc), np.int64(removed)
                    nlo[nn] = cur
                    nhi[nn] = ra
                    nelo[nn] = ce
                    nehi[nn] = ev
                    nn += 1
                    p0 = ra
            elif cut and npc > 0 and phi[npc - 1] == cur:
                p0 = plo[npc - 1]
                npc -= 1
                removed -= phi[npc] - plo[npc]
            else:
                p0 = cur
            p1 = rb if rb < b else b
            if npc >= pcap:
                return np.int64(-1), np.int64(npc), np.int64(removed)
            plo[npc] = p0
            phi[npc] = p1
            npc += 1
            removed += p1 - p0
            cut = True
            if rb >= b:
                cur = b
                break
            cur = rb
            ce = ev
            jj += 1
        if cur < b:
            if cut and b - cur < sliver and npc > 0 and phi[npc - 1] == cur:
                # a sliver above the last removal: removed with it
                removed += b - cur
                phi[npc - 1] = b
            else:
                if nn >= cap:
                    return np.int64(-1), np.int64(npc), np.int64(removed)
                nlo[nn] = cur
                nhi[nn] = b
                nelo[nn] = ce
                nehi[nn] = ehi[i]
                nn += 1
    return np.int64(nn), np.int64(npc), np.int64(removed)


@njit(types.UniTuple(i8, 3)(i8, I4_1, I4_1, I4_1, I4_1, i8, I1, I1, i8, I4_1, I4_1, I4_1, I4_1, I1, I1), **INLINE)
def iv_union(n, lo, hi, elo, ehi, m, alo, ahi, ev, nlo, nhi, nelo, nehi, plo, phi):
    """Segments united with sorted, disjoint intervals ``[alo, ahi)``: (new count, added pieces, added
    length). Added pieces are the parts of the intervals not already material; touching segments merge;
    an endpoint keeps the id of the segment that provides it (``ev`` where an added interval does)."""
    cap = nlo.shape[0]
    pcap = plo.shape[0]
    # added pieces = intervals minus segments
    npc = 0
    added = 0
    i = 0
    for j in range(m):
        cur = alo[j]
        b = ahi[j]
        if b <= cur:
            continue
        while i < n and hi[i] <= cur:
            i += 1
        ii = i
        while cur < b:
            if ii < n and lo[ii] < b:
                if lo[ii] > cur:
                    if npc >= pcap:
                        return np.int64(-1), np.int64(npc), np.int64(added)
                    plo[npc] = cur
                    phi[npc] = lo[ii]
                    npc += 1
                    added += np.int64(lo[ii]) - cur
                if hi[ii] > cur:
                    cur = np.int64(hi[ii])
                ii += 1
            else:
                if npc >= pcap:
                    return np.int64(-1), np.int64(npc), np.int64(added)
                plo[npc] = cur
                phi[npc] = b
                npc += 1
                added += b - cur
                cur = b
    # merged list
    nn = 0
    i = 0
    j = 0
    while i < n or j < m:
        if j >= m or (i < n and np.int64(lo[i]) <= alo[j]):
            a = np.int64(lo[i])
            b = np.int64(hi[i])
            ea = elo[i]
            eb = ehi[i]
            i += 1
        else:
            a = alo[j]
            b = ahi[j]
            ea = ev
            eb = ev
            j += 1
            if b <= a:
                continue
        if nn > 0 and a <= nhi[nn - 1]:
            if b > nhi[nn - 1]:
                nhi[nn - 1] = b
                nehi[nn - 1] = eb
        else:
            if nn >= cap:
                return np.int64(-1), np.int64(npc), np.int64(added)
            nlo[nn] = a
            nhi[nn] = b
            nelo[nn] = ea
            nehi[nn] = eb
            nn += 1
    return np.int64(nn), np.int64(npc), np.int64(added)


# --------------------------------------------------------------------------- indices and bricks
@njit(i8(FIELD_T, i8, i8, i8, i8), **OPTS)
def ray_index(f, lvl, fam, iu, iv):
    return C.rindex(f.lv_ray0, f.lv_dims, lvl, fam, iu, iv)


@njit(i8(FIELD_T, i8, i8, i8, i8), **OPTS)
def brick_index(f, lvl, bx, by, bz):
    return C.bindex(f.lv_brick0, f.lv_bdims, lvl, bx, by, bz)


@njit(void(FIELD_T, i8, i8, i8, i8, i8, i8, i8), **OPTS)
def mark_range(f, lvl, fam, iu, iv, q0, q1, dummy):
    """Mark changed the bricks whose closed node ranges contain the ray part [q0, q1] (both ends
    inclusive, both bricks of a shared boundary node)."""
    C.mark(f.lv_cell_q, f.lv_bdims, f.lv_brick0, f.bdirty, f.bconn, f.bver, lvl, fam, iu, iv, q0, q1)


@njit(void(FIELD_T, i8, i8, i8, i8, I1, I1, i8, i8), **OPTS)
def brick_account(f, lvl, fam, iu, iv, plo, phi, npc, sign):
    """Book removed (sign -1) or added (+1) pieces of one ray into the owning bricks' material and
    mark the touched bricks changed."""
    C.account(f.lv_cell_q, f.lv_bdims, f.lv_brick0, f.bmat, f.bdirty, f.bconn, f.bver, lvl, fam, iu, iv, plo, phi,
              npc, sign)


# --------------------------------------------------------------------------- lookups
@njit(types.UniTuple(i8, 3)(FIELD_T, i8, f8), **OPTS)
def ray_nearest(f, ray, q):
    """(inside, nearest endpoint in quanta, has_endpoint) of ``ray`` at the coordinate q (quanta)."""
    return C.nearest(f.cnt, f.seg, f.link, f.pseg, f.pnext, ray, q)


@njit(b1(FIELD_T, i8, f8), **OPTS)
def ray_inside(f, ray, q):
    return C.inside(f.cnt, f.seg, f.link, f.pseg, f.pnext, ray, q)


@njit(types.UniTuple(i8, 2)(FIELD_T, i8, i8, i8), **OPTS)
def ray_overlap(f, ray, q0, q1):
    """(material length inside [q0, q1), segments touching it) of ``ray``."""
    return C.overlap(f.cnt, f.seg, f.link, f.pseg, f.pnext, ray, q0, q1)


@njit(i8(FIELD_T, i8, i8), **OPTS)
def family_length(f, lvl, fam):
    """Total material length (quanta) of one family of one level: the exact integer the volume is."""
    r0 = f.lv_ray0[lvl, fam]
    u = 1 if fam == 0 else 0
    v = 2 if fam < 2 else 1
    nr = f.lv_dims[lvl, u] * f.lv_dims[lvl, v]
    cnt = f.cnt
    seg = f.seg
    link = f.link
    pseg = f.pseg
    pnext = f.pnext
    total = np.int64(0)
    for ray in range(r0, r0 + nr):
        n = np.int64(cnt[ray])
        b = np.int64(link[ray])
        s = 0
        for k in range(n):
            if k < K:
                total += np.int64(seg[ray, k, 1]) - np.int64(seg[ray, k, 0])
            else:
                if s == K:
                    b = np.int64(pnext[b])
                    s = 0
                total += np.int64(pseg[b, s, 1]) - np.int64(pseg[b, s, 0])
                s += 1
    return total


@njit(types.UniTuple(i8, 2)(TARGET_T, i8, i8, i8), **OPTS)
def target_overlap_t(t, ray, q0, q1):
    """(target length inside [q0, q1), index of the first target segment touched or -1)."""
    a = t.start[ray]
    b = t.start[ray + 1]
    total = np.int64(0)
    first = np.int64(-1)
    for k in range(a, b):
        lo = np.int64(t.seg[k, 0])
        hi = np.int64(t.seg[k, 1])
        if lo >= q1:
            break
        x0 = lo if lo > q0 else q0
        x1 = hi if hi < q1 else q1
        if x1 > x0:
            total += x1 - x0
            if first < 0:
                first = k
    return total, first


@njit(types.UniTuple(i8, 2)(I1, types.int32[:, ::1], i8, i8, i8), **INLINE)
def target_overlap(start, tseg, ray, q0, q1):
    """(target length inside [q0, q1), index of the first target segment touched or -1)."""
    a = start[ray]
    b = start[ray + 1]
    total = np.int64(0)
    first = np.int64(-1)
    for k in range(a, b):
        lo = np.int64(tseg[k, 0])
        hi = np.int64(tseg[k, 1])
        if lo >= q1:
            break
        x0 = lo if lo > q0 else q0
        x1 = hi if hi < q1 else q1
        if x1 > x0:
            total += x1 - x0
            if first < 0:
                first = k
    return total, first


@njit(i8(f8), **OPTS)
def quantize(x):
    """Fixed-point coordinate (quanta) of a float coordinate in quanta: round half up, saturated."""
    if x != x:
        return np.int64(0)
    if x > 2.0e9:
        return np.int64(2000000000)
    if x < -2.0e9:
        return np.int64(-2000000000)
    return np.int64(np.floor(x + 0.5))


KERNELS = (blocks_for, pool_alloc, pool_free, ray_load, ray_store, iv_subtract, iv_union, ray_index, brick_index,
           mark_range, brick_account, ray_nearest, ray_inside, ray_overlap, family_length, target_overlap_t, quantize)
