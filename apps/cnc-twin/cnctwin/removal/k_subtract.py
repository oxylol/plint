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
from numba import njit, types

from . import abi as A
from . import k_core as C
from .k_envelope import (ENV_CX, ENV_CY, ENV_CZ, ENV_HINV, ENV_KEY, ENV_MAGIC, ENV_NS, ENV_SPREAD, ENV_THI,
                         ENV_TLO, ENV_Z0, ENV_Z1, _aabb, _dead_cyl, _env_ray, _event, node_range)
from .k_ray import iv_subtract, target_overlap
from .nbt import ACCUM_T, EVENTS_T, F1, FIELD_T, I2, INLINE, PROFILE_T, RAW, SCRATCH_T, SPEC_T, TARGET_T, f8, i8, void

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
AG_R_D = 36         # largest piece removed during a rapid along a ray, per family (3)
N_AG = 40
PCS_BATCH = 512     # pieces booked per post_pieces call


@njit(types.UniTuple(f8, 3)(i8, f8, f8, f8), **INLINE)
def _point(fam, pf, pu, pv):
    """(x, y, z) of the point at pf along family fam's axis and (pu, pv) across it."""
    if fam == 0:
        return pf, pu, pv
    if fam == 1:
        return pu, pf, pv
    return pu, pv, pf


@njit(void(FIELD_T, TARGET_T, SPEC_T, F1, ACCUM_T, I2, i8, F1, SCRATCH_T), **RAW)
def post_pieces(f, tgt, spec, pinfo, acc, pcs, npc, agg, sc):
    """Second pass of a subtraction over the recorded pieces (ray, level, family, iu, iv, q0, q1): brick
    material and change flags, target checks (gouge: removed target material deeper than the tolerance
    along the ray; breakthrough: a target segment left without material), the non-cutting centre while
    plunging, the tool-frame extent (piece ends; widened across the ray by half and by a whole dual cell: FA_ZE_*),
    (level 0, z family) the removed-volume centroid and the machined-surface normal sums, and (level 0, every
    family) the same sums per family with the piece counts (FA_FAM0 ..). Aggregates go to ``agg`` (AG_*).
    Allocation-free (work arrays from the scratch ``sc``), reference counting off."""
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
    div = sc.iv2
    dtag = sc.tg
    blvl = -1
    bq = np.int64(1)
    binv = 1.0
    # the brick of the last piece booked into one brick: a piece of the same level and family whose column lies
    # strictly inside the same brick column and whose ends lie strictly inside its brick along the ray, short of the
    # last half spacing (which also marks the next brick: k_core.mark), is booked into it directly (book_piece would
    # find that one brick)
    m_lvl = -1
    m_fam = -1
    m_qu = -1
    m_qv = -1
    m_lo = np.int64(0)
    m_hi = np.int64(0)
    m_idx = np.int64(0)
    # running values in registers (written back below; every sum takes its terms in the same order)
    zr_min = agg[AG_ZR_MIN]
    zr_max = agg[AG_ZR_MAX]
    rr_max = agg[AG_RR_MAX]
    ze_lo = afv[A.FA_ZE_LO]
    ze_lo2 = afv[A.FA_ZE_LO2]
    ze_hi = afv[A.FA_ZE_HI]
    ze_hi2 = afv[A.FA_ZE_HI2]
    afv[A.FA_ENV_ZMIN] = zmin
    afv[A.FA_ENV_ZMAX] = zmax
    s_cx = afv[A.FA_CX]
    s_cy = afv[A.FA_CY]
    s_cz = afv[A.FA_CZ]
    s_w = afv[A.FA_W]
    s_nx = afv[A.FA_NX]
    s_ny = afv[A.FA_NY]
    s_nz = afv[A.FA_NZ]
    w0 = w[0]
    w1 = w[1]
    w2 = w[2]
    for k in range(npc):
        ray = pcs[k, 0]
        lvl = pcs[k, 1]
        fam = pcs[k, 2]
        iu = pcs[k, 3]
        iv = pcs[k, 4]
        p0 = pcs[k, 5]
        p1 = pcs[k, 6]
        if p1 > p0:
            if lvl != blvl:
                blvl = lvl
                bq = C.BR * lv_cell_q[lvl]
                binv = 1.0 / bq
            qu = np.int64(np.uint64(iu) // np.uint64(C.BR))
            qv = np.int64(np.uint64(iv) // np.uint64(C.BR))
            if (lvl == m_lvl and fam == m_fam and qu == m_qu and qv == m_qv and p0 > m_lo and p1 < m_hi
                    and 2 * p1 + lv_cell_q[lvl] <= 2 * m_hi
                    and (iu - qu * C.BR != 0 or iu == 0) and (iv - qv * C.BR != 0 or iv == 0)):
                bmat[m_idx, fam] -= p1 - p0
                bdirty[m_idx] = 1
                bconn[m_idx] = 1
                bver[m_idx] += 1
            else:
                idx = C.book_piece(lv_bdims, lv_brick0, bmat, bdirty, bconn, bver, lvl, fam, iu, iv, p0, p1, -1, bq,
                                   binv)
                m_lvl = -1
                if idx >= 0:
                    # the piece went into one brick: remember it (its range along the ray, strictly inside)
                    wb = C.floordiv(p0, bq, binv)
                    m_lvl = lvl
                    m_fam = fam
                    m_qu = qu
                    m_qv = qv
                    m_lo = wb * bq
                    m_hi = (wb + 1) * bq
                    m_idx = idx
        u = 1 if fam == 0 else 0
        v = 2 if fam < 2 else 1
        a = lv_cell_m[lvl]
        a2 = a * a
        of = lv_origin[lvl, fam]
        pu = lv_origin[lvl, u] + iu * a
        pv = lv_origin[lvl, v] + iv * a
        lm = (p1 - p0) * Q
        if rapid and lm > agg[AG_R_D + fam]:
            agg[AG_R_D + fam] = lm
        if check:
            ov, first = target_overlap(tstart, tseg, ray, p0, p1)
            if ov > 0:
                ovm = ov * Q
                agg[AG_G_VOL] += ovm * a2 / 3.0
                if ovm > gtol and ovm > agg[AG_GF + fam]:
                    agg[AG_GF + fam] = ovm
                    x, y, z = _point(fam, of + 0.5 * (p0 + p1) * Q, pu, pv)
                    agg[AG_GP + 3 * fam] = x
                    agg[AG_GP + 3 * fam + 1] = y
                    agg[AG_GP + 3 * fam + 2] = z
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
                                x, y, z = _point(fam, of + 0.5 * (tl + th) * Q, pu, pv)
                                agg[AG_B_X] = x
                                agg[AG_B_X + 1] = y
                                agg[AG_B_X + 2] = z
                                agg[AG_B_LVL] = lvl
                                agg[AG_B_FAM] = fam
        if dead_ok:
            x, y, z = _point(fam, of, pu, pv)
            nd = _dead_cyl(x, y, z, fam, w, c0x, c0y, c0z, r_dead, zmin, zmin + z_dead, p0 * Q, p1 * Q, div, dtag)
            for kk in range(nd):
                x0 = div[kk, 0] if div[kk, 0] > p0 * Q else p0 * Q
                x1 = div[kk, 1] if div[kk, 1] < p1 * Q else p1 * Q
                if x1 > x0:
                    agg[AG_D_LEN] += (x1 - x0) * a2 / 3.0
        # half the dual cell across the ray, projected on the tool axis (0 for the family along the axis)
        if fam == 0:
            half = 0.5 * a * (abs(w1) + abs(w2))
        elif fam == 1:
            half = 0.5 * a * (abs(w0) + abs(w2))
        else:
            half = 0.5 * a * (abs(w0) + abs(w1))
        for e in range(2):
            qq = p0 if e == 0 else p1
            x, y, z = _point(fam, of + qq * Q, pu, pv)
            dx = x - c0x
            dy = y - c0y
            dz = z - c0z
            zr = dx * w0 + dy * w1 + dz * w2
            if zr < zr_min:
                zr_min = zr
            if zr > zr_max:
                zr_max = zr
            if zr - half < ze_lo:
                ze_lo = zr - half
            if zr - 2.0 * half < ze_lo2:
                ze_lo2 = zr - 2.0 * half
            if zr + half > ze_hi:
                ze_hi = zr + half
            if zr + 2.0 * half > ze_hi2:
                ze_hi2 = zr + 2.0 * half
            ex = dx - zr * w0
            ey = dy - zr * w1
            ez = dz - zr * w2
            # the square root only when the point can be farther than the largest distance so far
            r2 = ex * ex + ey * ey + ez * ez
            if r2 > rr_max * rr_max * (1.0 - 1e-12):
                rr = math.sqrt(r2)
                if rr > rr_max:
                    rr_max = rr
        if lvl == 0:
            vol = lm * a2
            x, y, z = _point(fam, of + 0.5 * (p0 + p1) * Q, pu, pv)
            dx = x - c0x
            dy = y - c0y
            dz = z - c0z
            zr = dx * w0 + dy * w1 + dz * w2
            if zr < zmin:
                zr = zmin
            if zr > zmax:
                zr = zmax
            nx = c0x + zr * w0 - x
            ny = c0y + zr * w1 - y
            nz = c0z + zr * w2 - z
            nn_ = math.sqrt(nx * nx + ny * ny + nz * nz)
            fb = A.FA_FAM0 + A.FA_FSTRIDE * fam
            afv[fb + A.FF_CX] += vol * x
            afv[fb + A.FF_CY] += vol * y
            afv[fb + A.FF_CZ] += vol * z
            afv[fb + A.FF_W] += vol
            afv[fb + A.FF_N] += 1.0
            if nn_ > 0.0:
                afv[fb + A.FF_NX] += vol * nx / nn_
                afv[fb + A.FF_NY] += vol * ny / nn_
                afv[fb + A.FF_NZ] += vol * nz / nn_
            if fam == 2:
                s_cx += vol * x
                s_cy += vol * y
                s_cz += vol * z
                s_w += vol
                if nn_ > 0.0:
                    s_nx += vol * nx / nn_
                    s_ny += vol * ny / nn_
                    s_nz += vol * nz / nn_
                if rapid:
                    agg[AG_R_VOL] += vol
                    agg[AG_R_X] += vol * x
                    agg[AG_R_X + 1] += vol * y
                    agg[AG_R_X + 2] += vol * z
    agg[AG_ZR_MIN] = zr_min
    agg[AG_ZR_MAX] = zr_max
    agg[AG_RR_MAX] = rr_max
    afv[A.FA_ZE_LO] = ze_lo
    afv[A.FA_ZE_LO2] = ze_lo2
    afv[A.FA_ZE_HI] = ze_hi
    afv[A.FA_ZE_HI2] = ze_hi2
    afv[A.FA_CX] = s_cx
    afv[A.FA_CY] = s_cy
    afv[A.FA_CZ] = s_cz
    afv[A.FA_W] = s_w
    afv[A.FA_NX] = s_nx
    afv[A.FA_NY] = s_ny
    afv[A.FA_NZ] = s_nz


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
    # slab bounds for _env_ray (work[16:]): the spread of the slab axis points, when the slab heights ascend
    env = fw[A.FW_ENV:A.FW_ENV + A.N_ENV]
    env[ENV_KEY] = 0.0
    mono = True
    spread = 0.0
    for k in range(ns):
        if slab_z[k + 1] < slab_z[k]:
            mono = False
        dx = slab_c[k, 0] - c0x
        dy = slab_c[k, 1] - c0y
        dz = slab_c[k, 2] - c0z
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if d > spread:
            spread = d
    if mono:
        env[ENV_NS] = ns
        env[ENV_CX] = c0x
        env[ENV_CY] = c0y
        env[ENV_CZ] = c0z
        env[ENV_SPREAD] = spread * (math.sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]) + 1e-12)
        env[ENV_Z0] = slab_z[0]
        env[ENV_Z1] = slab_z[ns]
        env[ENV_HINV] = ns / (slab_z[ns] - slab_z[0]) if slab_z[ns] > slab_z[0] else 0.0
        env[ENV_KEY] = ENV_MAGIC
    work = fw[A.FW_WORK:A.FW_ENV + A.N_ENV]
    cap_seg = s_lo.shape[0]
    npc_all = 0
    # pieces are booked in batches small enough to stay in the cache (the order, and so every sum, is the same)
    cap_p = min(pcs.shape[0], PCS_BATCH)
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
            # quick-reject bounds: certainly outside above rq_hi, certainly inside below rq_lo (only the rays in
            # between take the original test, so the decision is the original one)
            rq_hi = Rq * (1.0 + 1e-12)
            rq_lo = Rq * (1.0 - 1e-12)
            rq2_hi = rq_hi * rq_hi
            rq2_lo = rq_lo * rq_lo
            nq_hi = rq_hi * ndx * (1.0 + 1e-12)
            nq_lo = rq_lo * ndx * (1.0 - 1e-12)
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
                        num = abs(rx * dx0 + ry * dx1 + rz * dx2)
                        if num > nq_hi:
                            continue
                        if num > nq_lo and num / ndx > Rq:
                            continue
                    else:
                        zz = rx * w[0] + ry * w[1] + rz * w[2]
                        px = rx - zz * w[0]
                        py = ry - zz * w[1]
                        pz = rz - zz * w[2]
                        d2 = px * px + py * py + pz * pz
                        if d2 > rq2_hi:
                            continue
                        if d2 > rq2_lo and math.sqrt(d2) > Rq:
                            continue
                    mat, m_lo, m_hi = C.extent(cnt, seg, link, pseg, pnext, ray, q_lo, q_hi)
                    if mat == 0:
                        continue
                    env[ENV_TLO] = m_lo * Q
                    env[ENV_THI] = m_hi * Q
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
                    # no material in the removal intervals: nothing to subtract (iv_subtract would remove 0; a ray
                    # with more segments than the scratch holds takes the full path for its overflow flags)
                    if cnt[ray] <= cap_seg:
                        hit = False
                        for i in range(m):
                            mt, tt = C.overlap(cnt, seg, link, pseg, pnext, ray, s_rlo[i], s_rhi[i])
                            if mt > 0:
                                hit = True
                                break
                        if not hit:
                            continue
                    # one inline segment trimmed at one end by one interval (the common case): written in place, as
                    # load / iv_subtract / store would (the new end carries the event id; the other inline slots and
                    # the count stay; no pool block involved)
                    if m == 1 and cnt[ray] == 1 and link[ray] < 0:
                        sa = np.int64(seg[ray, 0, 0])
                        sb = np.int64(seg[ray, 0, 1])
                        ra = s_rlo[0]
                        rb = s_rhi[0]
                        trim = 0
                        # (a remainder shorter than a sliver takes the general path: iv_subtract removes it too)
                        if sa < ra and ra < sb and sb <= rb and ra - sa >= A.SLIVER_Q:
                            trim = 1
                            seg[ray, 0, 1] = ra
                            sev[ray, 0, 1] = ev_id
                            p0 = ra
                            p1 = sb
                        elif ra <= sa and sa < rb and rb < sb and sb - rb >= A.SLIVER_Q:
                            trim = 1
                            seg[ray, 0, 0] = rb
                            sev[ray, 0, 0] = ev_id
                            p0 = sa
                            p1 = rb
                        if trim == 1:
                            for kk in range(1, C.K):
                                seg[ray, kk, 0] = 0
                                seg[ray, kk, 1] = 0
                                sev[ray, kk, 0] = 0
                                sev[ray, kk, 1] = 0
                            if info[A.FI_MAX_SEG] < 1:
                                info[A.FI_MAX_SEG] = 1
                            any_removed = True
                            removed_q[lvl, fam] += p1 - p0
                            aiv[A.IA_RAYS_CHANGED] += 1
                            if npc_all + 1 > cap_p:
                                post_pieces(f, tgt, spec, prof.info, acc, pcs, npc_all, agg, sc)
                                npc_all = 0
                            pcs[npc_all, 0] = ray
                            pcs[npc_all, 1] = lvl
                            pcs[npc_all, 2] = fam
                            pcs[npc_all, 3] = iu
                            pcs[npc_all, 4] = iv
                            pcs[npc_all, 5] = p0
                            pcs[npc_all, 6] = p1
                            npc_all += 1
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
        # depth: the family whose largest removed piece is smallest (the one most normal to the face the rapid
        # moved into, as the gouge depth); the domain reports a crash only beyond the collision tolerance
        rdepth = 0.0
        for fam in range(3):
            if agg[AG_R_D + fam] > 0.0 and (rdepth == 0.0 or agg[AG_R_D + fam] < rdepth):
                rdepth = agg[AG_R_D + fam]
        rows[nrow, 0] = A.EV_RAPID_INTO_MATERIAL
        rows[nrow, 1] = agg[AG_R_X] / rv if rv > 0.0 else c0x
        rows[nrow, 2] = agg[AG_R_X + 1] / rv if rv > 0.0 else c0y
        rows[nrow, 3] = agg[AG_R_X + 2] / rv if rv > 0.0 else c0z
        rows[nrow, 4] = rdepth
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
INLINED = (_point,)
