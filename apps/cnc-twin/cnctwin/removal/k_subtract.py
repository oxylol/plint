"""Subtraction of revolved envelopes from the tri-dexel field (docs/models/removal.md "Subtraction").

:func:`subtract_envelope` runs in two passes per call: the hot pass walks the rays of the engaged-sector
box (quick rejects: no segments, farther from the axis than R_max + max|dr| + the slab spread, no material
in the box), computes their envelope intervals exactly (k_envelope), quantises them and subtracts them,
recording every removed piece; the second pass (:func:`post_pieces`, compiled separately) books the pieces
into the bricks, checks them against the target (gouge, breakthrough), the non-cutting centre (plunge),
accumulates the contact's centroid, normal and tool-frame extent, and the events are written once per call
(architecture 5.5 "subtract only within the bounding box of the engaged sector"). The field, profile, spec
and scratch arrays are unpacked once per call (k_core: no reference counting per ray).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from . import abi as A
from . import k_core as C
from .k_envelope import _aabb, _dead_cyl, _env_ray, _event, node_range
from .k_ray import iv_subtract, target_overlap
from .nbt import ACCUM_T, EVENTS_T, F1, FIELD_T, I2, PROFILE_T, RAW, SCRATCH_T, SPEC_T, TARGET_T, i8, void

Q = A.QUANTUM_M


# aggregate slots of post_pieces
AG_G_DEPTH = 0
AG_G_X = 1
AG_G_VOL = 4
AG_G_LVL = 5
AG_G_FAM = 6
AG_B_DEPTH = 7
AG_B_X = 8
AG_B_LVL = 11
AG_B_FAM = 12
AG_B_N = 13
AG_D_LEN = 14
AG_R_VOL = 15
AG_R_X = 16
AG_ZR_MIN = 19
AG_ZR_MAX = 20
AG_RR_MAX = 21
AG_GF = 24          # largest target overlap along a ray, per family (3)
AG_GP = 27          # its middle point, per family (3 x 3)
N_AG = 40


@njit(void(FIELD_T, TARGET_T, SPEC_T, F1, ACCUM_T, I2, i8, F1, SCRATCH_T), **RAW)
def post_pieces(f, tgt, spec, pinfo, acc, pcs, npc, agg, sc):
    """Second pass of a subtraction over the recorded pieces (ray, level, family, iu, iv, q0, q1): brick
    material and change flags, target checks (gouge: removed target material deeper than the tolerance
    along the ray; breakthrough: a target segment left without material), the non-cutting centre while
    plunging, the tool-frame extent, and (level 0, z family) the removed-volume centroid and the
    machined-surface normal sums. Aggregates go to ``agg`` (AG_*). Allocation-free (work arrays from the
    scratch ``sc``), reference counting off."""
    cnt = f.cnt
    seg = f.seg
    link = f.link
    pseg = f.pseg
    pnext = f.pnext
    lv_cell_m = f.lv_cell_m
    lv_origin = f.lv_origin_m
    lv_cell_q = f.lv_cell_q
    lv_bdims = f.lv_bdims
    lv_brick0 = f.lv_brick0
    bmat = f.bmat
    bdirty = f.bdirty
    bconn = f.bconn
    bver = f.bver
    tstart = tgt.start
    tseg = tgt.seg
    w = spec.axis_S
    afv = acc.fvals
    ns = spec.slab_dr.shape[0]
    check = spec.ivals[A.SI_CHECK_TARGET] == 1 and tgt.info[0] == 1
    rapid = spec.ivals[A.SI_RAPID] == 1
    plunge = spec.ivals[A.SI_PLUNGE] == 1
    gtol = spec.fvals[A.SF_GOUGE_TOL]
    r_dead = spec.fvals[A.SF_R_DEAD]
    z_dead = spec.fvals[A.SF_Z_DEAD]
    dead_ok = plunge and r_dead > 0.0 and z_dead > 0.0
    c0x = spec.slab_c[0, 0]
    c0y = spec.slab_c[0, 1]
    c0z = spec.slab_c[0, 2]
    zmin = max(pinfo[A.PI_ZMIN], spec.slab_z[0])
    zmax = min(pinfo[A.PI_ZMAX], spec.slab_z[ns])
    one_lo = sc.iw[0:1]
    one_hi = sc.iw[1:2]
    div = sc.iv2
    dtag = sc.tg
    p = sc.fw[A.FW_PP:A.FW_PP + 3]
    for k in range(npc):
        ray = pcs[k, 0]
        lvl = pcs[k, 1]
        fam = pcs[k, 2]
        iu = pcs[k, 3]
        iv = pcs[k, 4]
        p0 = pcs[k, 5]
        p1 = pcs[k, 6]
        one_lo[0] = p0
        one_hi[0] = p1
        C.account(lv_cell_q, lv_bdims, lv_brick0, bmat, bdirty, bconn, bver, lvl, fam, iu, iv, one_lo, one_hi, 1, -1)
        u = 1 if fam == 0 else 0
        v = 2 if fam < 2 else 1
        a = lv_cell_m[lvl]
        a2 = a * a
        of = lv_origin[lvl, fam]
        p[u] = lv_origin[lvl, u] + iu * a
        p[v] = lv_origin[lvl, v] + iv * a
        lm = (p1 - p0) * Q
        if check:
            ov, first = target_overlap(tstart, tseg, ray, p0, p1)
            if ov > 0:
                ovm = ov * Q
                agg[AG_G_VOL] += ovm * a2 / 3.0
                if ovm > gtol and ovm > agg[AG_GF + fam]:
                    agg[AG_GF + fam] = ovm
                    p[fam] = of + 0.5 * (p0 + p1) * Q
                    agg[AG_GP + 3 * fam] = p[0]
                    agg[AG_GP + 3 * fam + 1] = p[1]
                    agg[AG_GP + 3 * fam + 2] = p[2]
                    agg[AG_G_LVL] = lvl
                if ovm > gtol:
                    for kk in range(first, tstart[ray + 1]):
                        tl = np.int64(tseg[kk, 0])
                        th = np.int64(tseg[kk, 1])
                        if tl >= p1:
                            break
                        if th <= p0:
                            continue
                        left, _t = C.overlap(cnt, seg, link, pseg, pnext, ray, tl, th)
                        if left == 0:
                            agg[AG_B_N] += 1.0
                            tm = (th - tl) * Q
                            if tm > agg[AG_B_DEPTH]:
                                agg[AG_B_DEPTH] = tm
                                p[fam] = of + 0.5 * (tl + th) * Q
                                agg[AG_B_X] = p[0]
                                agg[AG_B_X + 1] = p[1]
                                agg[AG_B_X + 2] = p[2]
                                agg[AG_B_LVL] = lvl
                                agg[AG_B_FAM] = fam
        if dead_ok:
            p[fam] = of
            nd = _dead_cyl(p[0], p[1], p[2], fam, w, c0x, c0y, c0z, r_dead, zmin, zmin + z_dead, p0 * Q, p1 * Q,
                           div, dtag)
            for kk in range(nd):
                x0 = div[kk, 0] if div[kk, 0] > p0 * Q else p0 * Q
                x1 = div[kk, 1] if div[kk, 1] < p1 * Q else p1 * Q
                if x1 > x0:
                    agg[AG_D_LEN] += (x1 - x0) * a2 / 3.0
        for e in range(2):
            qq = p0 if e == 0 else p1
            p[fam] = of + qq * Q
            dx = p[0] - c0x
            dy = p[1] - c0y
            dz = p[2] - c0z
            zr = dx * w[0] + dy * w[1] + dz * w[2]
            if zr < agg[AG_ZR_MIN]:
                agg[AG_ZR_MIN] = zr
            if zr > agg[AG_ZR_MAX]:
                agg[AG_ZR_MAX] = zr
            ex = dx - zr * w[0]
            ey = dy - zr * w[1]
            ez = dz - zr * w[2]
            rr = math.sqrt(ex * ex + ey * ey + ez * ez)
            if rr > agg[AG_RR_MAX]:
                agg[AG_RR_MAX] = rr
        if lvl == 0 and fam == 2:
            vol = lm * a2
            p[fam] = of + 0.5 * (p0 + p1) * Q
            afv[A.FA_CX] += vol * p[0]
            afv[A.FA_CY] += vol * p[1]
            afv[A.FA_CZ] += vol * p[2]
            afv[A.FA_W] += vol
            dx = p[0] - c0x
            dy = p[1] - c0y
            dz = p[2] - c0z
            zr = dx * w[0] + dy * w[1] + dz * w[2]
            if zr < zmin:
                zr = zmin
            if zr > zmax:
                zr = zmax
            nx = c0x + zr * w[0] - p[0]
            ny = c0y + zr * w[1] - p[1]
            nz = c0z + zr * w[2] - p[2]
            nn_ = math.sqrt(nx * nx + ny * ny + nz * nz)
            if nn_ > 0.0:
                afv[A.FA_NX] += vol * nx / nn_
                afv[A.FA_NY] += vol * ny / nn_
                afv[A.FA_NZ] += vol * nz / nn_
            if rapid:
                agg[AG_R_VOL] += vol
                agg[AG_R_X] += vol * p[0]
                agg[AG_R_X + 1] += vol * p[1]
                agg[AG_R_X + 2] += vol * p[2]


@njit(i8(FIELD_T, PROFILE_T, SPEC_T, TARGET_T, ACCUM_T, EVENTS_T, SCRATCH_T), **RAW)
def subtract_envelope(f, prof, spec, tgt, acc, ev, sc):
    """Subtract the slabbed revolved envelope of a SubtractSpec from every level of the field (bit l of the
    level mask), inside the spec's box. Books removed lengths per level and family, the centroid, normal
    and tool-frame extent in ``acc``; pushes one aggregated event per kind (rapid into material, gouge,
    breakthrough, non-centre-cutting plunge, pool overflow). Returns 0, or 1 when a ray could not be
    written for lack of pool blocks (grow the pool and call again: the operation is idempotent).
    Allocation-free (work arrays from the scratch), reference counting off (nbt.RAW)."""
    cnt = f.cnt
    seg = f.seg
    sev = f.sev
    link = f.link
    pseg = f.pseg
    psev = f.psev
    pnext = f.pnext
    info = f.info
    lv_cell_m = f.lv_cell_m
    lv_origin = f.lv_origin_m
    lv_dims = f.lv_dims
    lv_cell_q = f.lv_cell_q
    lv_ray0 = f.lv_ray0
    pz0 = prof.z0_m
    pz1 = prof.z1_m
    pkind = prof.kind
    pc0 = prof.c0
    pc1 = prof.c1
    pc2 = prof.c2
    pc3 = prof.c3
    ppart = prof.part
    w = spec.axis_S
    slab_c = spec.slab_c
    slab_z = spec.slab_z
    slab_dr = spec.slab_dr
    s_lo = sc.lo
    s_hi = sc.hi
    s_elo = sc.elo
    s_ehi = sc.ehi
    s_nlo = sc.nlo
    s_nhi = sc.nhi
    s_nelo = sc.nelo
    s_nehi = sc.nehi
    s_plo = sc.plo
    s_phi = sc.phi
    s_rlo = sc.rlo
    s_rhi = sc.rhi
    ivt = sc.ivt
    ivp = sc.ivp
    pcs = sc.pcs
    removed_q = acc.removed_q
    aiv = acc.ivals
    afv = acc.fvals
    status = 0
    ns = slab_dr.shape[0]
    if ns == 0 or pz0.shape[0] == 0:
        return 0
    fw = sc.fw
    elo_ = fw[A.FW_ELO:A.FW_ELO + 3]
    ehi_ = fw[A.FW_EHI:A.FW_EHI + 3]
    Rq = _aabb(w, slab_dr, slab_c, slab_z, slab_dr, prof.info[A.PI_ZMIN], prof.info[A.PI_ZMAX],
               prof.info[A.PI_RMAX], spec.fvals[A.SF_SLACK], elo_, ehi_)
    blo = fw[A.FW_BLO:A.FW_BLO + 3]
    bhi = fw[A.FW_BHI:A.FW_BHI + 3]
    for i in range(3):
        blo[i] = max(elo_[i], spec.box_lo[i])
        bhi[i] = min(ehi_[i], spec.box_hi[i])
        if bhi[i] < blo[i]:
            return 0
    ev_id = spec.ivals[A.SI_EVENT_ID]
    tick = spec.ivals[A.SI_TICK]
    contact = spec.ivals[A.SI_CONTACT]
    rapid = spec.ivals[A.SI_RAPID] == 1
    mask = spec.ivals[A.SI_LEVEL_MASK]
    c0x = slab_c[0, 0]
    c0y = slab_c[0, 1]
    c0z = slab_c[0, 2]
    zmin = max(prof.info[A.PI_ZMIN], slab_z[0])
    agg = fw[A.FW_AGG:A.FW_AGG + N_AG]
    for i in range(N_AG):
        agg[i] = 0.0
    agg[AG_ZR_MIN] = afv[A.FA_ZREL_MIN]
    agg[AG_ZR_MAX] = afv[A.FA_ZREL_MAX]
    agg[AG_RR_MAX] = afv[A.FA_RREL_MAX]
    any_removed = False
    overflowed = False
    p = fw[A.FW_P:A.FW_P + 3]
    work = fw[A.FW_WORK:A.FW_WORK + 16]
    npc_all = 0
    cap_p = pcs.shape[0]
    nl = info[A.FI_NLEVELS]
    for lvl in range(nl):
        if mask >= 0 and (mask >> lvl) & 1 == 0:
            continue
        a = lv_cell_m[lvl]
        for fam in range(3):
            u = 1 if fam == 0 else 0
            v = 2 if fam < 2 else 1
            ou = lv_origin[lvl, u]
            ov = lv_origin[lvl, v]
            of = lv_origin[lvl, fam]
            nu = lv_dims[lvl, u]
            nv = lv_dims[lvl, v]
            iu0, iu1 = node_range((blo[u] - ou) / a, (bhi[u] - ou) / a, 0.0, nu)
            iv0, iv1 = node_range((blo[v] - ov) / a, (bhi[v] - ov) / a, 0.0, nv)
            if iu1 < iu0 or iv1 < iv0:
                continue
            lq = (lv_dims[lvl, fam] - 1) * lv_cell_q[lvl]
            t_lo = blo[fam] - of
            t_hi = bhi[fam] - of
            if t_lo < 0.0:
                t_lo = 0.0
            if t_hi > lq * Q:
                t_hi = lq * Q
            if t_hi <= t_lo:
                continue
            q_lo = np.int64(np.floor(t_lo / Q))
            q_hi = np.int64(np.ceil(t_hi / Q))
            if fam == 0:
                dx0 = 0.0
                dx1 = -w[2]
                dx2 = w[1]
            elif fam == 1:
                dx0 = w[2]
                dx1 = 0.0
                dx2 = -w[0]
            else:
                dx0 = -w[1]
                dx1 = w[0]
                dx2 = 0.0
            ndx = math.sqrt(dx0 * dx0 + dx1 * dx1 + dx2 * dx2)
            r0 = lv_ray0[lvl, fam]
            for iu in range(iu0, iu1 + 1):
                for iv in range(iv0, iv1 + 1):
                    ray = r0 + iu * nv + iv
                    if cnt[ray] == 0:
                        continue
                    p[fam] = of
                    p[u] = ou + iu * a
                    p[v] = ov + iv * a
                    rx = p[0] - c0x
                    ry = p[1] - c0y
                    rz = p[2] - c0z
                    if ndx > 1e-12:
                        dist = abs(rx * dx0 + ry * dx1 + rz * dx2) / ndx
                    else:
                        zz = rx * w[0] + ry * w[1] + rz * w[2]
                        px = rx - zz * w[0]
                        py = ry - zz * w[1]
                        pz = rz - zz * w[2]
                        dist = math.sqrt(px * px + py * py + pz * pz)
                    if dist > Rq:
                        continue
                    mat, touched = C.overlap(cnt, seg, link, pseg, pnext, ray, q_lo, q_hi)
                    if mat == 0:
                        continue
                    niv = _env_ray(pz0, pz1, pkind, pc0, pc1, pc2, pc3, ppart, p[0], p[1], p[2], fam, w, slab_c,
                                   slab_z, slab_dr, t_lo, t_hi, ivt, ivp, 1, work)
                    if niv <= 0:
                        continue
                    m = 0
                    for i in range(niv):
                        qa = np.int64(np.floor(ivt[i, 0] / Q + 0.5))
                        qb = np.int64(np.floor(ivt[i, 1] / Q + 0.5))
                        if qa < 0:
                            qa = 0
                        if qb > lq:
                            qb = lq
                        if qb <= qa:
                            continue
                        if m > 0 and qa <= s_rhi[m - 1]:
                            if qb > s_rhi[m - 1]:
                                s_rhi[m - 1] = qb
                        else:
                            s_rlo[m] = qa
                            s_rhi[m] = qb
                            m += 1
                    if m == 0:
                        continue
                    n = C.load(cnt, seg, sev, link, pseg, psev, pnext, ray, s_lo, s_hi, s_elo, s_ehi)
                    if n < 0:
                        info[A.FI_SCRATCH_OVF] = 1
                        aiv[A.IA_SCRATCH_OVF] += 1
                        continue
                    nn, npc, removed = iv_subtract(n, s_lo, s_hi, s_elo, s_ehi, m, s_rlo, s_rhi, ev_id, s_nlo, s_nhi,
                                                   s_nelo, s_nehi, s_plo, s_phi)
                    if nn < 0:
                        info[A.FI_SCRATCH_OVF] = 1
                        aiv[A.IA_SCRATCH_OVF] += 1
                        continue
                    if removed == 0:
                        continue
                    if C.store(cnt, seg, sev, link, pseg, psev, pnext, info, ray, s_nlo, s_nhi, s_nelo, s_nehi, nn) < 0:
                        status = 1
                        overflowed = True
                        continue
                    any_removed = True
                    removed_q[lvl, fam] += removed
                    aiv[A.IA_RAYS_CHANGED] += 1
                    if lvl == 0 and fam == 2 and nn != n:
                        info[A.FI_TOPO] += 1
                    if npc_all + npc > cap_p:
                        post_pieces(f, tgt, spec, prof.info, acc, pcs, npc_all, agg, sc)
                        npc_all = 0
                    for k in range(npc):
                        pcs[npc_all, 0] = ray
                        pcs[npc_all, 1] = lvl
                        pcs[npc_all, 2] = fam
                        pcs[npc_all, 3] = iu
                        pcs[npc_all, 4] = iv
                        pcs[npc_all, 5] = s_plo[k]
                        pcs[npc_all, 6] = s_phi[k]
                        npc_all += 1
    if npc_all > 0:
        post_pieces(f, tgt, spec, prof.info, acc, pcs, npc_all, agg, sc)
    # gouge depth: the family whose largest overlap is smallest (the one most normal to the gouged face)
    gfam = -1
    for fam in range(3):
        if agg[AG_GF + fam] > 0.0 and (gfam < 0 or agg[AG_GF + fam] < agg[AG_GF + gfam]):
            gfam = fam
    if gfam >= 0:
        agg[AG_G_DEPTH] = agg[AG_GF + gfam]
        agg[AG_G_X] = agg[AG_GP + 3 * gfam]
        agg[AG_G_X + 1] = agg[AG_GP + 3 * gfam + 1]
        agg[AG_G_X + 2] = agg[AG_GP + 3 * gfam + 2]
        agg[AG_G_FAM] = gfam
    afv[A.FA_ZREL_MIN] = agg[AG_ZR_MIN]
    afv[A.FA_ZREL_MAX] = agg[AG_ZR_MAX]
    afv[A.FA_RREL_MAX] = agg[AG_RR_MAX]
    aiv[A.IA_SUBTRACTIONS] += 1
    if any_removed:
        info[A.FI_VERSION] += 1
        aiv[A.IA_IN_CUT_CALLS] += 1
        aiv[A.IA_LAST_TICK] = tick
        if aiv[A.IA_FIRST_TICK] < 0:
            aiv[A.IA_FIRST_TICK] = tick
    if agg[AG_D_LEN] > 0.0:
        afv[A.FA_DEAD_V] += agg[AG_D_LEN]
    rows = sc.rows
    for k in range(5):
        for j in range(10):
            rows[k, j] = 0.0
    nrow = 0
    if overflowed:
        aiv[A.IA_OVERFLOW] += 1
        rows[nrow, 0] = A.EV_POOL_OVERFLOW
        rows[nrow, 1] = c0x
        rows[nrow, 2] = c0y
        rows[nrow, 3] = c0z
        rows[nrow, 6] = -1
        nrow += 1
    if agg[AG_G_DEPTH] > 0.0:
        rows[nrow, 0] = A.EV_GOUGE
        rows[nrow, 1] = agg[AG_G_X]
        rows[nrow, 2] = agg[AG_G_X + 1]
        rows[nrow, 3] = agg[AG_G_X + 2]
        rows[nrow, 4] = agg[AG_G_DEPTH]
        rows[nrow, 5] = agg[AG_G_VOL]
        rows[nrow, 6] = -1
        rows[nrow, 7] = agg[AG_G_LVL]
        rows[nrow, 8] = agg[AG_G_FAM]
        nrow += 1
    if agg[AG_B_N] > 0.0:
        rows[nrow, 0] = A.EV_BREAKTHROUGH
        rows[nrow, 1] = agg[AG_B_X]
        rows[nrow, 2] = agg[AG_B_X + 1]
        rows[nrow, 3] = agg[AG_B_X + 2]
        rows[nrow, 4] = agg[AG_B_DEPTH]
        rows[nrow, 5] = agg[AG_G_VOL]
        rows[nrow, 6] = -1
        rows[nrow, 7] = agg[AG_B_LVL]
        rows[nrow, 8] = agg[AG_B_FAM]
        nrow += 1
    if agg[AG_D_LEN] > 0.0:
        rows[nrow, 0] = A.EV_NON_CENTRE_PLUNGE
        rows[nrow, 1] = c0x + zmin * w[0]
        rows[nrow, 2] = c0y + zmin * w[1]
        rows[nrow, 3] = c0z + zmin * w[2]
        rows[nrow, 5] = agg[AG_D_LEN]
        rows[nrow, 6] = -1
        rows[nrow, 8] = 2
        nrow += 1
    if rapid and any_removed:
        rv = agg[AG_R_VOL]
        rows[nrow, 0] = A.EV_RAPID_INTO_MATERIAL
        rows[nrow, 1] = agg[AG_R_X] / rv if rv > 0.0 else c0x
        rows[nrow, 2] = agg[AG_R_X + 1] / rv if rv > 0.0 else c0y
        rows[nrow, 3] = agg[AG_R_X + 2] / rv if rv > 0.0 else c0z
        rows[nrow, 5] = rv
        rows[nrow, 6] = -1
        rows[nrow, 8] = 2
        nrow += 1
    erec = ev.rec
    en = ev.n
    for k in range(nrow):
        _event(erec, en, np.int64(rows[k, 0]), tick, contact, rows[k, 1], rows[k, 2], rows[k, 3], rows[k, 4],
               rows[k, 5], np.int64(rows[k, 6]), ev_id, np.int64(rows[k, 7]), np.int64(rows[k, 8]))
    return status


KERNELS = (post_pieces, subtract_envelope)
