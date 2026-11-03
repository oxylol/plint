"""Composition kernels for core.hotloop.run_macro (docs/models/removal.md "Integration").

* :func:`tooth_pass` - one tooth of a rotating tool at its generation instant: engaged intervals of the
  tool's stations (warm-started from ``eng``), contact geometry, the engaged-sector box, the tooth's slab
  layout from ToothSlabs with the per-slab axis points the loop computed (tip at the slab's generation
  instant + u_rel, mapped to the material configuration), then the subtraction. Returns the subtraction
  status (1: pool overflow, the loop replays the step after removal grows the pool).
* :func:`l1_contact` - per L1 tick and path: broad phase of the cutting envelope against the occupied
  bricks; feed motion out of contact is air cutting (ticks and the run's first / last tip in the
  accumulator).
* :func:`excess_material` - material of the field outside the target (remaining material).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from . import abi as A
from .k_gate import contact_geometry, engaged_intervals, sector_box
from .k_query import broad_phase
from .k_subtract import subtract_envelope
from .nbt import (ACCUM_T, ENGAGED_T, EVENTS_T, F1, F2, FIELD_T, OPTS, PROFILE_T, RAW, SCRATCH_T, SLABS_T, SPEC_T,
                  TARGET_T, B1, b1, f8, i8)

Q = A.QUANTUM_M


@njit(i8(FIELD_T, PROFILE_T, SLABS_T, i8, F2, F1, F1, F1, F1, F1, F1, B1, ENGAGED_T, F1, f8, f8, F1, SPEC_T,
         TARGET_T, ACCUM_T, EVENTS_T, SCRATCH_T, F1), **RAW)
def tooth_pass(f, prof, slabs, tooth, slab_c, tip, x0, y0, w, st_z, st_r, active, eng, st_len, feed_x, feed_y,
               gate_params, spec, tgt, acc, ev, sc, stats):
    """Gate, bound and subtract one tooth pass. ``slab_c`` [n[tooth], 3] holds the axis point of every slab
    of the tooth (S, material configuration); ``spec`` is a work SubtractSpec whose ivals / fvals the loop
    filled (event id, tick, contact, rapid, plunge, gouge tolerance, dead zone, slack) and whose arrays are
    at least as large as the tooth's slab count (they are filled here)."""
    tick = spec.ivals[A.SI_TICK]
    engaged_intervals(f, tip, x0, y0, w, st_z, st_r, active, eng, tick, gate_params, stats)
    contact_geometry(eng, st_z, st_r, st_len, feed_x, feed_y, acc)
    margin = gate_params[3] if gate_params.shape[0] > 3 else f.lv_cell_m[0]
    rmax = prof.info[A.PI_RMAX]
    drmax = 0.0
    for k in range(slabs.n[tooth]):
        if slabs.dr[tooth, k] > drmax:
            drmax = slabs.dr[tooth, k]
    n_eng = sector_box(eng, tip, x0, y0, w, st_z, rmax + drmax, prof.info[A.PI_ZMIN], prof.info[A.PI_ZMAX], margin,
                       spec.box_lo, spec.box_hi)
    if n_eng == 0 and spec.ivals[A.SI_RAPID] == 0:
        return 0
    if n_eng == 0:
        # a rapid: the whole envelope (no engaged sector is known)
        for i in range(3):
            spec.box_lo[i] = -1e300
            spec.box_hi[i] = 1e300
    ns = slabs.n[tooth]
    for i in range(3):
        spec.axis_S[i] = w[i]
    for k in range(ns):
        for i in range(3):
            spec.slab_c[k, i] = slab_c[k, i]
        spec.slab_dr[k] = slabs.dr[tooth, k]
    for k in range(ns + 1):
        spec.slab_z[k] = slabs.z[tooth, k]
    # the spec's arrays may be longer than this tooth's slab count: collapse the surplus slabs
    for k in range(ns, spec.slab_dr.shape[0]):
        for i in range(3):
            spec.slab_c[k, i] = slab_c[ns - 1, i] if ns > 0 else 0.0
        spec.slab_dr[k] = -1e9
    for k in range(ns + 1, spec.slab_z.shape[0]):
        spec.slab_z[k] = spec.slab_z[ns]
    return subtract_envelope(f, prof, spec, tgt, acc, ev, sc)


@njit(b1(FIELD_T, PROFILE_T, F1, F1, f8, b1, i8, f8, ACCUM_T), **OPTS)
def l1_contact(f, prof, tip, w, margin, feed, tick, tick_s, acc):
    """Broad phase of one path at one L1 tick; books feed ticks and air-cut ticks (feed motion, spindle on,
    no occupied brick inside the envelope's box) with the run's first and last tip. Returns contact."""
    n = broad_phase(f, prof, tip, w, margin)
    contact = n > 0
    if feed:
        acc.ivals[A.IA_FEED_TICKS] += 1
        if not contact:
            if acc.ivals[A.IA_AIR_TICKS] == 0:
                acc.fvals[A.FA_AIR_X0] = tip[0]
                acc.fvals[A.FA_AIR_Y0] = tip[1]
                acc.fvals[A.FA_AIR_Z0] = tip[2]
            acc.ivals[A.IA_AIR_TICKS] += 1
            acc.fvals[A.FA_AIR_X1] = tip[0]
            acc.fvals[A.FA_AIR_Y1] = tip[1]
            acc.fvals[A.FA_AIR_Z1] = tip[2]
    return contact


@njit(i8(FIELD_T, TARGET_T, f8, F1), **OPTS)
def excess_material(f, tgt, tol, out):
    """Material of level 0 outside the target, from the z family: out = [volume (m^3), largest excess
    along a z ray (m), x, y, z of its middle, rays with excess above tol]. Returns the rays with excess."""
    cnt = f.cnt
    seg = f.seg
    link = f.link
    pseg = f.pseg
    pnext = f.pnext
    nx = f.lv_dims[0, 0]
    ny = f.lv_dims[0, 1]
    r0 = f.lv_ray0[0, 2]
    a = f.lv_cell_m[0]
    o = f.lv_origin_m[0]
    K = A.K_INLINE
    for i in range(6):
        out[i] = 0.0
    nray = 0
    for i in range(nx):
        for j in range(ny):
            ray = r0 + i * ny + j
            n = np.int64(cnt[ray])
            if n == 0:
                continue
            b = np.int64(link[ray])
            s = 0
            t0 = tgt.start[ray]
            t1 = tgt.start[ray + 1]
            ray_excess = False
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
                # segment minus target segments
                cur = lo
                for q in range(t0, t1 + 1):
                    if q < t1:
                        tl = np.int64(tgt.seg[q, 0])
                        th = np.int64(tgt.seg[q, 1])
                    else:
                        tl = hi
                        th = hi
                    if th <= cur:
                        continue
                    e_hi = tl if tl < hi else hi
                    if e_hi > cur:
                        ex = (e_hi - cur) * Q
                        out[0] += ex * a * a
                        if ex > tol:
                            ray_excess = True
                        if ex > out[1]:
                            out[1] = ex
                            out[2] = o[0] + i * a
                            out[3] = o[1] + j * a
                            out[4] = o[2] + 0.5 * (cur + e_hi) * Q
                    if th > cur:
                        cur = th
                    if cur >= hi:
                        break
            if ray_excess:
                nray += 1
    out[5] = nray
    return nray


KERNELS = (tooth_pass, l1_contact, excess_material)
