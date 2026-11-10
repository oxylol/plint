"""Compiled kernels of the cutting domain (docs/models/cutting.md "Kernels"; architecture 5.4, 5.5).

* :func:`cut_step` - one L0 step of one contact of a rotating tool (milling with every envelope,
  drilling, reaming, boring heads, thread milling, chamfering, engraving, slitting, cutting and
  forming taps): gating by the engaged intervals, element positions (runout, tilt, wear, growth and
  pull-out are in the effective edge; float and deflection added here), chip thickness from the
  surface memory (vector form), forces from the linearised table integrated over the swept angle
  clipped to the intervals, wear-land and process-damping forces, wrenches, per-element integrals,
  heat by zone, analytic chip / displaced volume, memory update with elastic recovery (delayed by one
  bin), entry / exit / breakthrough / breakout shocks, tooth-pass rows (chip emission), surface events,
  work-heat deposits, critical-section loads, recut impulses, packing loads, per-tick wrenches, axis
  loads, spindle torque and the floating tapping holder.
* :func:`turn_step` - the same for a non-rotating tool on the rotating workpiece: chip thickness per
  element from the polar field (radius view) along the element normal, optionally Colwell's chord
  with the equivalent chip thickness.
* :func:`deflection_from_modal` - station deflections from dynamics' ModalModel / ModalState.
* :func:`copy_element_accumulators` - CutState.elem_acc -> core Accum.elem (once per macro step).

Rules (architecture 5.5): explicit signatures, cache, nogil, error_model numpy, no fastmath, no
parallel; only C-contiguous arrays, scalars and NamedTuples of them; per element and step only
table interpolation and products (sin/cos of the spindle angle by angle addition in the caller).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, typeof, types

from ..core.abi import (ACC_CHIP_VOLUME_M3, ACC_FEED_WORK_J, ACC_FORCE_PEAK_N, ACC_HEAT_FLANK_J, ACC_HEAT_RAKE_J,
                        ACC_HEAT_SHEAR_J, ACC_IN_CUT_S, ACC_PLOUGHING_J, ACC_POWER_PEAK_W, ACC_PROCESS_DAMPING_J,
                        ACC_RECUT_MASS_KG, ACC_SHAFT_WORK_J, ACC_TORQUE_PEAK_NM, ELEM_ENTRIES, ELEM_FLANK_SLIDING_M,
                        ELEM_H_MAX, ELEM_IN_CUT_S, ELEM_INT_H_DT, ELEM_INT_SIGMA_V_FLANK, ELEM_INT_SIGMA_V_RAKE,
                        ELEM_RAKE_SLIDING_M, N_ELEM_ACC, WRENCH_TOOL_MEAN, WRENCH_TOOL_PEAK, WRENCH_WORK_MEAN,
                        WRENCH_WORK_PEAK, ModalModel, ModalState)
from .abi import (ACCUM_T, AXES_T, CA_AXIAL_K, CA_DEFLECTION_J, CA_DISPLACED_M3, CA_EDGE_LEN_INT, CA_ELEM_EVALS,
                  CA_ENTRIES, CA_EVENTS_LOST, CA_FEED_POWER_PEAK, CA_FLOAT_U_MAX, CA_FORMING_J, CA_FX_INT, CA_FY_INT,
                  CA_FZ_INT, CA_H_MAX, CA_PACKING_J, CA_POWER_INT, CA_PROCESS_DAMPING_J, CA_RECUT_J, CA_RECUT_KG,
                  CA_REFILL_MISSING, CA_STALE_READS, CA_STEPS, CA_STEPS_IN_CUT, CA_TABLE_OVER, CA_TORQUE_INT,
                  CA_V_INT, CA_OMEGA_INT, CA_TIME, CA_FPEAK_MX, CA_FPEAK_MY, CA_FPEAK_MZ, CA_FPEAK_INST_N, CA_H_BEYOND,
                  CA_H_BEYOND_MAX, CACCUM_T, COL_EA, COL_ER, COL_ET, COL_FA, COL_FR, COL_FT, COL_PR, COL_PS, COL_RC,
                  COL_SIG_RAKE, COL_TCHIP, COL_WF, COL_WS, DEFLECTION_T, EVENTS_T, FL_EVENT_DONE, FL_IN_CUT,
                  FL_NONDEGEN, FL_VN_POS, INTERVALS_T, MEMORY_T, MS_FLOAT_F, MS_FLOAT_K, MS_FLOAT_U, MS_LAST_TICK,
                  MS_RECUT_ANGLE, MS_STEPS, MS_TURN_EVENT_ANGLE, OCCUPANCY_T, OUT_T, PACKING_T, PD_BIN, PD_H, PD_X,
                  PD_Y, PD_Z, POLAR_T, POSE_T, REFILL_T, SHOCK_BREAKOUT, SHOCK_BREAKTHROUGH, SHOCK_ENTRY, SHOCK_EXIT,
                  SI_COMMIT, SI_CONTACT, SI_L0_TICKS, SI_SLOT, SI_STALE_BEFORE, SI_STALE_TICKS, SI_TICK_END,
                  SI_TICK_ROW, SO_AXIAL_K, SO_ENGAGED, SO_POWER_W, SO_TORQUE_NM, ST_COS_MID, ST_DT_S, ST_OMEGA_RAD_S,
                  ST_RHO_KG_M3, ST_SIN_MID, ST_THETA_MID_RAD, ST_TICK_FRAC, ST_TICK_S, ST_TOOTH_PERIOD_S, STATE_T,
                  TABLE_T, TC_TANGENT_VN, TC_FLOAT_K, TC_FLOAT_TRAVEL, TC_H_CAP, TC_HOLE_MAKING, TC_N_ELEM,
                  TC_N_TEETH, TC_PD_MAX_LEN, TC_R_CUT, TC_RECOVERY_RATIO, TC_RECUT_ROW, TC_SURF_EVENT_RAD,
                  TC_TURN_COLWELL, TC_Z_HI, TC_Z_LO, TOOL_T, TS_ACTIVE, TS_B, TS_BREAKOUT, TS_CENTRE_ON,
                  TS_CENTRE_WAS, TS_DOM_KIND, TS_DOM_ROW, TS_DOM_VOL, TS_EMIT_TICK, TS_ENTRY_ANGLE, TS_ENTRY_ELEM,
                  TS_ENTRY_H, TS_ENTRY_TICK, TS_EXIT_FIRST_S, TS_EXITS, TS_FIRST_ENTRY_S, TS_FMAX, TS_FX, TS_FY,
                  TS_FZ, TS_HV, TS_INCLV, TS_LAST_ANGLE, TS_LAST_ELEM, TS_LAST_ENTRY_S, TS_LAST_EXIT_S, TS_LAST_H,
                  TS_LOADING, TS_LV, TS_NEW, TS_NOW, TS_NX, TS_NY, TS_NZ, TS_PASS_ON, TS_PX, TS_PY, TS_PZ,
                  TS_RAKEV, TS_RV, TS_S_NX, TS_S_NY, TS_S_NZ, TS_S_PX, TS_S_PY, TS_S_PZ, TS_S_VX, TS_S_VY, TS_S_VZ,
                  TS_STEP_B, TS_STEP_DV, TS_STEP_E, TS_TV, TS_UX, TS_UY, TS_UZ, TS_VOL, TS_VV, TS_WORK_J, TS_AVAIL,
                  PK_ALPHA0, PK_ARM, PK_C, PK_CHIP, PK_DB, PK_DIR, PK_DS, PK_EDGE_R, PK_FLANK, PK_INCL, PK_N,
                  PK_NPXY, PK_P, PK_PD, PK_RAKE, PK_RXY, PK_SIG_FLANK, PK_VB, PK_CURV, PKI_AXIAL, PKI_ELEM, PKI_FORMING,
                  PKI_KIND, PKI_READ, PKI_ROW, PKI_TOOTH, PKI_WRITE, PD_RING_N, PDA_COUNT, PDA_UX, PDA_UY, PDA_UZ,
                  PDA_X, RA_FX, RA_FY, RA_FZ, RA_MX, RA_MY, RA_MZ, RA_T, RA_T0, RA_Z, SE_D_HI, SE_D_LO, SE_ON, SE_TICK)

OPTS = dict(cache=True, nogil=True, error_model="numpy")
# kernels that allocate nothing are compiled with Numba's reference counting off for that function: every array
# they touch arrives as an argument, and an inlined helper would otherwise pay an atomic incref / decref per
# array argument (as removal.nbt RAW; docs/performance.md)
RAW = dict(OPTS, _nrt=False)
# small per-element helpers are inlined at the Numba IR level (no call boundary in the element loop; they are
# still compiled for their explicit signature as kernels of their own)
INL = dict(OPTS, inline="always")
TWO_PI = 2.0 * math.pi
f8 = types.float64
i8 = types.int64
b1 = types.boolean
F1 = types.float64[::1]
F2 = types.float64[:, ::1]
F3 = types.float64[:, :, ::1]
I1 = types.int64[::1]
GATE_T = types.UniTuple(f8, 4)
SIX_T = types.UniTuple(f8, 6)


def _modal_example() -> ModalModel:
    return ModalModel(np.zeros((1, 2, 2)), np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 1, 6)), np.zeros((1, 3)),
                      np.zeros(1, np.int8), np.zeros(1, np.int32), np.zeros((3, 3)), np.zeros(1))


MODAL_T = typeof(_modal_example())
MSTATE_T = typeof(ModalState(np.zeros((1, 2)), np.zeros(1)))


# =============================================================================== event rows
@njit(b1(F2, I1), **INL)
def row_ok(rec, n):
    """True when one more row fits (else sets the overflow flag n[1])."""
    if n[0] >= rec.shape[0]:
        n[1] = 1
        return False
    return True


@njit(types.void(F2, I1, f8, f8, f8, f8, f8, f8, f8, f8, CACCUM_T), **OPTS)
def put_shock(rec, n, tick, tooth, elem, h, angle, force, loading, kind, cacc):
    if not row_ok(rec, n):
        cacc.scalars[CA_EVENTS_LOST] += 1.0
        return
    i = n[0]
    rec[i, 0] = tick
    rec[i, 1] = tooth
    rec[i, 2] = elem
    rec[i, 3] = h
    rec[i, 4] = angle
    rec[i, 5] = force
    rec[i, 6] = loading
    rec[i, 7] = kind
    n[0] = i + 1


@njit(types.void(F2, I1, f8, f8, f8, f8, f8, f8, f8, CACCUM_T), **OPTS)
def put_deposit(rec, n, x, y, z, nx, ny, nz, joule, cacc):
    if not row_ok(rec, n):
        cacc.scalars[CA_EVENTS_LOST] += 1.0
        return
    i = n[0]
    rec[i, 0] = x
    rec[i, 1] = y
    rec[i, 2] = z
    rec[i, 3] = nx
    rec[i, 4] = ny
    rec[i, 5] = nz
    rec[i, 6] = joule
    n[0] = i + 1


@njit(types.void(F2, I1, f8, f8, f8, f8, f8, f8, CACCUM_T), **OPTS)
def put_recut(rec, n, tick, tooth, elem, mass, x, y, cacc):
    if not row_ok(rec, n):
        cacc.scalars[CA_EVENTS_LOST] += 1.0
        return
    i = n[0]
    rec[i, 0] = tick
    rec[i, 1] = tooth
    rec[i, 2] = elem
    rec[i, 3] = mass
    rec[i, 4] = x
    rec[i, 5] = y
    n[0] = i + 1


@njit(types.void(F2, I1, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, CACCUM_T), **OPTS)
def put_surface(rec, n, tick, tooth, elem, x, y, z, nx, ny, nz, h, radius, vb, ux, uy, uz, slot, cacc):
    if not row_ok(rec, n):
        cacc.scalars[CA_EVENTS_LOST] += 1.0
        return
    i = n[0]
    rec[i, 0] = tick
    rec[i, 1] = tooth
    rec[i, 2] = elem
    rec[i, 3] = x
    rec[i, 4] = y
    rec[i, 5] = z
    rec[i, 6] = nx
    rec[i, 7] = ny
    rec[i, 8] = nz
    rec[i, 9] = h
    rec[i, 10] = radius
    rec[i, 11] = vb
    rec[i, 12] = ux
    rec[i, 13] = uy
    rec[i, 14] = uz
    rec[i, 15] = slot
    n[0] = i + 1


# =============================================================================== model helpers
@njit(f8(f8, f8, f8, f8), **INL)
def flank_interference_area(slope, land_m, tan_clear, max_len_m):
    """Area (m2 per m of edge) between the flank and the surface the same edge has just generated,
    for a path entering the material with slope = v_n / V (Wu 1989; Altintas, Eynian & Onozuka 2008;
    Ahmadi & Ismail 2011): a zero-clearance land of length ``land_m`` (wear land + edge contact)
    followed by the clearance face, tan(gamma) = ``tan_clear``; the interference ends where
    slope x = (x - L) tan(gamma), at most ``max_len_m`` behind the edge.
    delta(x) = slope x - max(0, x - L) tan(gamma), A = integral of delta over [0, x*]."""
    if slope <= 0.0:
        return 0.0
    L = land_m
    if L > max_len_m:
        L = max_len_m
    if L < 0.0:
        L = 0.0
    if slope >= tan_clear:
        xs = max_len_m
    else:
        xs = L * tan_clear / (tan_clear - slope)
        if xs > max_len_m:
            xs = max_len_m
    if xs < L:
        xs = L
    d = xs - L
    return 0.5 * slope * xs * xs - 0.5 * tan_clear * d * d


@njit(f8(f8, f8, f8, f8), **INL)
def _pos_area(x0, f0, x1, f1):
    """Integral over [x0, x1] of the positive part of the linear function from f0 (at x0) to f1 (at x1)."""
    w = x1 - x0
    if w <= 0.0:
        return 0.0
    if f0 >= 0.0 and f1 >= 0.0:
        return 0.5 * (f0 + f1) * w
    if f0 <= 0.0 and f1 <= 0.0:
        return 0.0
    if f0 > 0.0:
        return 0.5 * f0 * f0 / (f0 - f1) * w
    return 0.5 * f1 * f1 / (f1 - f0) * w


@njit(f8(f8, f8, f8, f8, f8, f8, f8), **INL)
def flank_segment(xa, da, xb, db, land, tan_clear, x_max):
    """Indented area over one segment [xa, xb] behind the edge (clipped at x_max): the generated surface stands
    da .. db (linear) above the flank's zero-clearance land, the clearance face recedes max(0, x - land) tan(gamma)
    behind it; the positive part of the difference, integrated exactly (split at the land's end)."""
    if xb <= xa or xa >= x_max:
        return 0.0
    if xb > x_max:
        db = da + (db - da) * (x_max - xa) / (xb - xa)
        xb = x_max
    area = 0.0
    if xa < land < xb:
        dl = da + (db - da) * (land - xa) / (xb - xa)
        area += _pos_area(xa, da, land, dl)
        xa = land
        da = dl
    ca = (xa - land) * tan_clear if xa > land else 0.0
    cb = (xb - land) * tan_clear if xb > land else 0.0
    return area + _pos_area(xa, da - ca, xb, db - cb)


@njit(f8(F3, F2, i8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8), **INL)
def wave_area(ring, aux, e, ux, uy, uz, nx, ny, nz, travel, land, tan_clear, x_max, dx):
    """Area (m2 per m of edge) between element e's flank and the surface the element itself generated behind it
    (docs/models/cutting.md "Process damping"; Altintas, Eynian & Onozuka 2008; Budak & Tunc 2010 indentation
    volume). At the distance x behind the edge the surface was left where the edge was when it passed there, so it
    stands (u(now) - u(then)) . n above the land, u the element's relative vibration displacement (T0) and n its
    current normal (into the material); the steady path is the surface the edge itself follows. Nodes: the current
    point (x = 0), the last committed evaluation (x = travel - aux X) and the wave memory's samples (sample g at
    x = travel - g dx), u linear between them; the interference ends at x_max (the flank length, or where the
    surface behind was generated by another tooth). A vibration much shorter than the flank indents it by at most
    its own amplitude over the contact; a long wave of slope s gives the classical s L^2 / 2 of the linearised
    model (flank_interference_area)."""
    cnt = int(aux[e, PDA_COUNT])
    if cnt <= 0 or x_max <= 0.0:
        return 0.0
    L = land
    if L > x_max:
        L = x_max
    if L < 0.0:
        L = 0.0
    xb = travel - aux[e, PDA_X]
    db = (ux - aux[e, PDA_UX]) * nx + (uy - aux[e, PDA_UY]) * ny + (uz - aux[e, PDA_UZ]) * nz
    area = flank_segment(0.0, 0.0, xb, db, L, tan_clear, x_max)
    if xb >= x_max:
        return area
    xa = xb
    da = db
    nk = ring.shape[1]
    g = cnt - 1
    g_end = cnt - nk
    if g_end < 0:
        g_end = 0
    while g >= g_end:
        k = g % nk
        xb = travel - g * dx
        db = (ux - ring[e, k, 0]) * nx + (uy - ring[e, k, 1]) * ny + (uz - ring[e, k, 2]) * nz
        area += flank_segment(xa, da, xb, db, L, tan_clear, x_max)
        if xb >= x_max:
            break
        xa = xb
        da = db
        g -= 1
    return area


@njit(types.void(F3, F2, i8, f8, f8, f8, f8, f8, b1), **INL)
def wave_push(ring, aux, e, ux, uy, uz, travel_step, dx, entering):
    """Commit element e's evaluation to its wave memory: at an entry into the cut the memory restarts (sample 0 at
    travel 0: the surface behind an entering edge was not generated by it); otherwise the travel advances by
    travel_step and the samples at the multiples of dx passed since the last evaluation are written, u linear
    between the two evaluations (only the last PD_RING_N of a long step)."""
    nk = ring.shape[1]
    if entering or aux[e, PDA_COUNT] < 1.0:
        ring[e, 0, 0] = ux
        ring[e, 0, 1] = uy
        ring[e, 0, 2] = uz
        aux[e, PDA_X] = 0.0
        aux[e, PDA_UX] = ux
        aux[e, PDA_UY] = uy
        aux[e, PDA_UZ] = uz
        aux[e, PDA_COUNT] = 1.0
        return
    x0 = aux[e, PDA_X]
    x1 = x0 + travel_step
    g = int(aux[e, PDA_COUNT])
    g_last = int(math.floor(x1 / dx)) if dx > 0.0 else g - 1
    if g_last >= g:
        if g_last - g >= nk:
            g = g_last - nk + 1
        span = x1 - x0
        u0x = aux[e, PDA_UX]
        u0y = aux[e, PDA_UY]
        u0z = aux[e, PDA_UZ]
        for gg in range(g, g_last + 1):
            w = (gg * dx - x0) / span if span > 0.0 else 1.0
            if w > 1.0:
                w = 1.0
            elif w < 0.0:
                w = 0.0
            k = gg % nk
            ring[e, k, 0] = u0x + w * (ux - u0x)
            ring[e, k, 1] = u0y + w * (uy - u0y)
            ring[e, k, 2] = u0z + w * (uz - u0z)
        aux[e, PDA_COUNT] = float(g_last + 1)
    aux[e, PDA_X] = x1
    aux[e, PDA_UX] = ux
    aux[e, PDA_UY] = uy
    aux[e, PDA_UZ] = uz


@njit(GATE_T(INTERVALS_T, i8, f8, f8, f8), **INL)
def gate(iv, s, a, half, inv_dal):
    """Engaged part of the arc an element sweeps in the step: (fraction, centroid s, s_lo, s_hi) with
    s in [0, 1] along the step (angle(s) = a + (s - 1/2) dal, a the midpoint angle in [0, 2 pi)).
    Intervals are right-handed T0 angles, lo in [0, 2 pi), lo <= hi <= lo + 2 pi."""
    cnt = iv.count[s]
    lo_arc = a - half
    hi_arc = a + half
    total = 0.0
    first = 1e300
    last = -1e300
    point = 0
    for i in range(cnt):
        lo = iv.lo_rad[s, i]
        hi = iv.hi_rad[s, i]
        for m in range(-1, 2):
            sh = m * TWO_PI
            if half == 0.0:
                if lo + sh <= a <= hi + sh:
                    point = 1
                continue
            x0 = lo + sh
            if x0 < lo_arc:
                x0 = lo_arc
            x1 = hi + sh
            if x1 > hi_arc:
                x1 = hi_arc
            if x1 > x0:
                total += x1 - x0
                if x0 < first:
                    first = x0
                if x1 > last:
                    last = x1
    if half == 0.0:
        if point == 1:
            return 1.0, 0.5, 0.0, 1.0
        return 0.0, 0.0, 0.0, 0.0
    if total <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    frac = total / (2.0 * half)
    if frac >= 1.0 - 1e-12:
        return 1.0, 0.5, 0.0, 1.0
    s_a = 0.5 + (first - a) * inv_dal
    s_b = 0.5 + (last - a) * inv_dal
    if s_a > s_b:
        tmp = s_a
        s_a = s_b
        s_b = tmp
    if s_a < 0.0:
        s_a = 0.0
    if s_b > 1.0:
        s_b = 1.0
    return frac, 0.5 * (s_a + s_b), s_a, s_b


@njit(b1(INTERVALS_T, i8, f8), **INL)
def in_intervals(iv, s, a):
    """True when the right-handed T0 angle a (any value) lies in one of station s's engaged intervals."""
    x = a - TWO_PI * np.floor(a / TWO_PI)
    for i in range(iv.count[s]):
        lo = iv.lo_rad[s, i]
        hi = iv.hi_rad[s, i]
        if lo <= x <= hi or lo <= x + TWO_PI <= hi:
            return True
    return False


@njit(b1(INTERVALS_T, i8, f8, f8), **INL)
def near_edge(iv, s, a, width):
    """True when angle a is within ``width`` of a boundary of station s's intervals."""
    x = a - TWO_PI * np.floor(a / TWO_PI)
    for i in range(iv.count[s]):
        lo = iv.lo_rad[s, i]
        hi = iv.hi_rad[s, i]
        if hi - lo >= TWO_PI - 1e-12:
            continue
        for m in range(-1, 2):
            sh = m * TWO_PI
            if abs(x + sh - lo) < width or abs(x + sh - hi) < width:
                return True
    return False


@njit(types.void(MEMORY_T, i8, i8, f8, f8, f8, f8, f8, f8, i8), **INL)
def memory_commit(mem, s, b, x, y, z, h, h_min, ratio, tick):
    """Write one passage into memory bin (s, b) with the elastic-recovery rule (docs/models/cutting.md
    "Surface memory"): unchanged below h_min (the ploughed layer recovers), moved by
    w = (h - h_min) / (ratio h_min) toward the edge point between h_min and (1 + ratio) h_min, set
    to it above. A bin without a point is set whenever h >= h_min."""
    if h < h_min:
        return
    if h_min <= 0.0 or ratio <= 0.0 or h >= (1.0 + ratio) * h_min or mem.valid[s, b] == 0:
        w = 1.0
    else:
        w = (h - h_min) / (ratio * h_min)
    if w <= 0.0:
        return
    mem.point_S_m[s, b, 0] += w * (x - mem.point_S_m[s, b, 0])
    mem.point_S_m[s, b, 1] += w * (y - mem.point_S_m[s, b, 1])
    mem.point_S_m[s, b, 2] += w * (z - mem.point_S_m[s, b, 2])
    mem.tick[s, b] = tick
    mem.valid[s, b] = 1


@njit(b1(MEMORY_T, i8, i8, i8, i8, i8), **INL)
def bin_fresh(mem, s, b, tick, stale_ticks, stale_before):
    if mem.valid[s, b] == 0:
        return False
    t = mem.tick[s, b]
    return t >= stale_before and tick - t <= stale_ticks


@njit(f8(REFILL_T, i8, i8, f8, f8, f8, f8, f8, f8, i8, b1, EVENTS_T, CACCUM_T), **INL)
def field_depth(refill, s, b, px, py, pz, nx, ny, nz, tick, request, ev, cacc):
    """Depth of an element point below the field plane of bin (s, b) along its normal n (positive inside the
    material), for elements that read the field directly (no predecessor on the thread helix). Without a
    plane: 0 and, when ``request``, a refill request row (the integration replays after the L2 query)."""
    if refill.valid[s, b] != 0:
        mx = refill.m_S[s, b, 0]
        my = refill.m_S[s, b, 1]
        mz = refill.m_S[s, b, 2]
        d = (refill.q_S_m[s, b, 0] - px) * mx + (refill.q_S_m[s, b, 1] - py) * my + (refill.q_S_m[s, b, 2] - pz) * mz
        den = -(nx * mx + ny * my + nz * mz)
        return d / den if den > 0.2 else d
    if request:
        rec = ev.refill.rec
        n = ev.refill.n
        if row_ok(rec, n):
            i = n[0]
            rec[i, 0] = tick
            rec[i, 1] = s
            rec[i, 2] = b
            rec[i, 3] = px
            rec[i, 4] = py
            rec[i, 5] = pz
            rec[i, 6] = nx
            rec[i, 7] = ny
            rec[i, 8] = nz
            n[0] = i + 1
        else:
            cacc.scalars[CA_EVENTS_LOST] += 1.0
        cacc.scalars[CA_REFILL_MISSING] += 1.0
    return 0.0


@njit(b1(MEMORY_T, REFILL_T, i8, i8, f8, f8, f8, f8, f8, f8, i8, b1, EVENTS_T, CACCUM_T), **INL)
def refill_bin(mem, refill, s, b, px, py, pz, nx, ny, nz, tick, commit, ev, cacc):
    """Refill a stale bin from its field plane: the surface point along -n from the element. Without a
    plane, write a refill request (the integration replays the step after the L2 query) and leave the
    bin as it is. Returns True when the bin holds a usable point afterwards."""
    if refill.valid[s, b] != 0:
        mx = refill.m_S[s, b, 0]
        my = refill.m_S[s, b, 1]
        mz = refill.m_S[s, b, 2]
        d = (refill.q_S_m[s, b, 0] - px) * mx + (refill.q_S_m[s, b, 1] - py) * my + (refill.q_S_m[s, b, 2] - pz) * mz
        den = -(nx * mx + ny * my + nz * mz)
        hb = d / den if den > 0.2 else d
        if commit:
            mem.point_S_m[s, b, 0] = px - hb * nx
            mem.point_S_m[s, b, 1] = py - hb * ny
            mem.point_S_m[s, b, 2] = pz - hb * nz
            mem.tick[s, b] = tick
            mem.valid[s, b] = 1
            cacc.scalars[CA_STALE_READS] += 1.0
            return True
        return False
    if commit:
        rec = ev.refill.rec
        n = ev.refill.n
        if row_ok(rec, n):
            i = n[0]
            rec[i, 0] = tick
            rec[i, 1] = s
            rec[i, 2] = b
            rec[i, 3] = px
            rec[i, 4] = py
            rec[i, 5] = pz
            rec[i, 6] = nx
            rec[i, 7] = ny
            rec[i, 8] = nz
            n[0] = i + 1
        else:
            cacc.scalars[CA_EVENTS_LOST] += 1.0
        cacc.scalars[CA_REFILL_MISSING] += 1.0
    return mem.valid[s, b] != 0


@njit(i8(TABLE_T, i8, f8, CACCUM_T), **INL)
def table_segment(table, row, h, cacc):
    """Segment of chip thickness h (> 0) on a table row; above the table the last segment
    (linear extrapolation, counted)."""
    hm = table.h_min_m[row]
    nlo = table.n_lo[0]
    if h < hm:
        k = int(h * table.inv_dlo[row])
        if k >= nlo:
            k = nlo - 1
        return k
    nseg = table.coef.shape[1]
    x = (h - hm) * table.inv_dhi[row]
    # compared before the integer conversion: a chip far beyond the table (a diverged state) must not wrap the index
    if not x < nseg - nlo:
        cacc.scalars[CA_TABLE_OVER] += 1.0
        return nseg - 1
    return nlo + int(x)


# =============================================================================== deflection
@njit(types.void(MODAL_T, MSTATE_T, F2, DEFLECTION_T), **OPTS)
def deflection_from_modal(model, mstate, residual_force_T0_N, out):
    """Station deflections for the force kernels from the modal state (architecture 5.4 "Dynamics"):
    tool modes (is_work 0) u_st = sum_m tool_shape[m, st, :] q_m and the rates with qdot_m;
    workpiece modes (is_work 1) at the contact point into u_work; machine modes (is_work 2) are
    relative tool-work modes and enter the tool side through their tool_shape row like tool modes when
    the shape array has one, else they are skipped; plus the quasi-static residual flexibility times
    ``residual_force_T0_N`` [nst, 3] (the predicted station forces, or zeros)."""
    nst = out.u_st_T0_m.shape[0]
    for s in range(nst):
        for c in range(6):
            out.u_st_T0_m[s, c] = 0.0
            out.v_st_T0_m_s[s, c] = 0.0
    for c in range(3):
        out.u_work_S_m[c] = 0.0
        out.v_work_S_m_s[c] = 0.0
    nm = model.Phi.shape[0]
    n_shapes = model.tool_shape.shape[0]
    ns = model.tool_shape.shape[1]
    shape_row = 0
    for m in range(nm):
        q = mstate.x[m, 0]
        qd = mstate.x[m, 1]
        if model.is_work[m] == 1:
            w = model.work_index[m]
            if 0 <= w < model.work_shape_at_contact.shape[0]:
                for c in range(3):
                    out.u_work_S_m[c] += model.work_shape_at_contact[w, c] * q
                    out.v_work_S_m_s[c] += model.work_shape_at_contact[w, c] * qd
            continue
        if shape_row < n_shapes:
            for s in range(min(ns, nst)):
                for c in range(6):
                    out.u_st_T0_m[s, c] += model.tool_shape[shape_row, s, c] * q
                    out.v_st_T0_m_s[s, c] += model.tool_shape[shape_row, s, c] * qd
            shape_row += 1
    nres = model.static_residual.shape[0] // 3
    if nres > 0 and residual_force_T0_N.shape[0] >= nres:
        for a in range(min(nres, nst)):
            for ca in range(3):
                acc = 0.0
                for b in range(nres):
                    for cb in range(3):
                        acc += model.static_residual[3 * a + ca, 3 * b + cb] * residual_force_T0_N[b, cb]
                out.u_st_T0_m[a, ca] += acc


# =============================================================================== element accumulators
@njit(types.void(STATE_T, ACCUM_T, i8), **OPTS)
def copy_element_accumulators(st, acc, n_elem):
    """CutState.elem_acc [n, N_EA] -> Accum.elem [N_ELEM_ACC, nt, ne] (once per macro step)."""
    nt = acc.elem.shape[1]
    ne = acc.elem.shape[2]
    for e in range(st.elem_acc.shape[0]):
        j = e // n_elem
        k = e - j * n_elem
        if j >= nt or k >= ne:
            continue
        for c in range(N_ELEM_ACC):
            acc.elem[c, j, k] = st.elem_acc[e, c]




# =============================================================================== tooth passes
@njit(types.void(STATE_T, EVENTS_T, CACCUM_T, POSE_T, F1, I1, i8, f8), **OPTS)
def emit_pass(st, ev, cacc, pose, step, istep, j, tick):
    """Chip-emission row (abi.PASS_COLUMNS, positions and velocities in M) and work-heat deposit (S) of
    tooth j's pass or emission period, then reset the period accumulators."""
    ts = st.tooth
    vol = ts[j, TS_VOL]
    px = ts[j, TS_PX]
    py = ts[j, TS_PY]
    pz = ts[j, TS_PZ]
    if vol > 0.0:
        R = pose.R_M_S
        t = pose.t_M_S_m
        rec = ev.passes.rec
        n = ev.passes.n
        if row_ok(rec, n):
            i = n[0]
            rec[i, 0] = tick
            rec[i, 1] = istep[SI_CONTACT]
            rec[i, 2] = istep[SI_SLOT]
            rec[i, 3] = j
            rec[i, 4] = step[ST_RHO_KG_M3] * vol
            rec[i, 5] = R[0, 0] * px + R[0, 1] * py + R[0, 2] * pz + t[0]
            rec[i, 6] = R[1, 0] * px + R[1, 1] * py + R[1, 2] * pz + t[1]
            rec[i, 7] = R[2, 0] * px + R[2, 1] * py + R[2, 2] * pz + t[2]
            rec[i, 8] = ts[j, TS_UX]
            rec[i, 9] = ts[j, TS_UY]
            rec[i, 10] = ts[j, TS_UZ]
            rec[i, 11] = ts[j, TS_TV] / vol
            rec[i, 12] = ts[j, TS_HV] / vol
            rec[i, 13] = ts[j, TS_B]
            rec[i, 14] = ts[j, TS_RV] / vol * ts[j, TS_LV]
            rec[i, 15] = ts[j, TS_VV] / vol
            rec[i, 16] = ts[j, TS_RAKEV] / vol
            rec[i, 17] = ts[j, TS_INCLV] / vol
            rec[i, 18] = ts[j, TS_DOM_KIND]
            rec[i, 19] = ts[j, TS_DOM_ROW]
            n[0] = i + 1
        else:
            cacc.scalars[CA_EVENTS_LOST] += 1.0
    if ts[j, TS_WORK_J] > 0.0:
        put_deposit(ev.deposits.rec, ev.deposits.n, px, py, pz, ts[j, TS_NX], ts[j, TS_NY], ts[j, TS_NZ],
                    ts[j, TS_WORK_J], cacc)
    ts[j, TS_VOL] = 0.0
    ts[j, TS_HV] = 0.0
    ts[j, TS_B] = 0.0
    ts[j, TS_LV] = 0.0
    ts[j, TS_VV] = 0.0
    ts[j, TS_RAKEV] = 0.0
    ts[j, TS_INCLV] = 0.0
    ts[j, TS_TV] = 0.0
    ts[j, TS_RV] = 0.0
    ts[j, TS_DOM_VOL] = 0.0
    ts[j, TS_WORK_J] = 0.0
    ts[j, TS_EMIT_TICK] = tick


@njit(types.void(TOOL_T, TABLE_T, MEMORY_T, STATE_T, EVENTS_T, CACCUM_T, i8, f8, f8, f8, i8, b1, f8, f8), **INL)
def element_exit(tool, table, mem, st, ev, cacc, e, t_ex, h_ex, ang_ex, tick, commit, ratio, slot):
    """An element left the cut: commit its pending memory write and update the tooth's exit bookkeeping."""
    if not commit:
        return
    j = tool.tooth[e]
    b = int(st.pend[e, PD_BIN])
    if b >= 0:
        if tool.write_row[e] >= 0:
            memory_commit(mem, tool.write_row[e], b, st.pend[e, PD_X], st.pend[e, PD_Y], st.pend[e, PD_Z],
                          st.pend[e, PD_H], table.h_write_m[tool.row[e]], ratio, st.pend_tick[e])
        st.pend[e, PD_BIN] = -1.0
    ts = st.tooth
    ts[j, TS_EXITS] += 1.0
    if ts[j, TS_EXIT_FIRST_S] < 0.0 or t_ex < ts[j, TS_EXIT_FIRST_S]:
        ts[j, TS_EXIT_FIRST_S] = t_ex
    if t_ex >= ts[j, TS_LAST_EXIT_S]:
        ts[j, TS_LAST_EXIT_S] = t_ex
        ts[j, TS_LAST_ELEM] = tool.elem[e]
        ts[j, TS_LAST_H] = h_ex
        ts[j, TS_LAST_ANGLE] = ang_ex
    st.flags[e] = 0


@njit(types.void(TOOL_T, STATE_T, EVENTS_T, CACCUM_T, POSE_T, F1, I1, b1, f8, f8, f8, f8, f8, f8, f8, f8, f8,
                 f8), **OPTS)
def close_teeth(tool, st, ev, cacc, pose, step, istep, commit, g00, g01, g02, g10, g11, g12, g20, g21, g22, dal):
    """End of a step for every tooth: separation point, entry shock when loading completes, breakthrough /
    breakout of hole-making tools, emission once per revolution of a continuous cut, exit shock and
    emission row when the pass ends."""
    ts = st.tooth
    nt = ts.shape[0]
    tick_end = istep[SI_TICK_END]
    tick_s = step[ST_TICK_S]
    omega = step[ST_OMEGA_RAD_S]
    abs_om = abs(omega)
    rev = TWO_PI / abs_om if abs_om > 0.0 else 0.0
    hole = tool.scal[TC_HOLE_MAKING] > 0.5
    RM = pose.R_M_S
    for j in range(nt):
        now = ts[j, TS_NOW]
        prev = ts[j, TS_ACTIVE]
        fm = math.sqrt(ts[j, TS_FX] ** 2 + ts[j, TS_FY] ** 2 + ts[j, TS_FZ] ** 2)
        if now > 0.0:
            dv = ts[j, TS_STEP_DV]
            if commit and dv > 0.0:
                inv = 1.0 / dv
                ax = ts[j, TS_S_PX] * inv
                ay = ts[j, TS_S_PY] * inv
                az = ts[j, TS_S_PZ] * inv
                psx = g00 * ax + g01 * ay + g02 * az
                psy = g10 * ax + g11 * ay + g12 * az
                psz = g20 * ax + g21 * ay + g22 * az
                ax = ts[j, TS_S_VX] * inv
                ay = ts[j, TS_S_VY] * inv
                az = ts[j, TS_S_VZ] * inv
                vsx = g00 * ax + g01 * ay + g02 * az
                vsy = g10 * ax + g11 * ay + g12 * az
                vsz = g20 * ax + g21 * ay + g22 * az
                ax = ts[j, TS_S_NX]
                ay = ts[j, TS_S_NY]
                az = ts[j, TS_S_NZ]
                nsx = g00 * ax + g01 * ay + g02 * az
                nsy = g10 * ax + g11 * ay + g12 * az
                nsz = g20 * ax + g21 * ay + g22 * az
                nn = math.sqrt(nsx * nsx + nsy * nsy + nsz * nsz)
                if nn > 0.0:
                    nsx /= nn
                    nsy /= nn
                    nsz /= nn
                ts[j, TS_PX] = psx
                ts[j, TS_PY] = psy
                ts[j, TS_PZ] = psz
                ts[j, TS_NX] = -nsx
                ts[j, TS_NY] = -nsy
                ts[j, TS_NZ] = -nsz
                # chip velocity in M: (edge velocity relative to the material + chip flow) + material velocity there
                rpx = RM[0, 0] * psx + RM[0, 1] * psy + RM[0, 2] * psz
                rpy = RM[1, 0] * psx + RM[1, 1] * psy + RM[1, 2] * psz
                rpz = RM[2, 0] * psx + RM[2, 1] * psy + RM[2, 2] * psz
                w = pose.w_S_M_rad_s
                ts[j, TS_UX] = RM[0, 0] * vsx + RM[0, 1] * vsy + RM[0, 2] * vsz + pose.v_S_M_m_s[0] + w[1] * rpz - w[2] * rpy
                ts[j, TS_UY] = RM[1, 0] * vsx + RM[1, 1] * vsy + RM[1, 2] * vsz + pose.v_S_M_m_s[1] + w[2] * rpx - w[0] * rpz
                ts[j, TS_UZ] = RM[2, 0] * vsx + RM[2, 1] * vsy + RM[2, 2] * vsz + pose.v_S_M_m_s[2] + w[0] * rpy - w[1] * rpx
            if commit:
                ts[j, TS_LV] += abs(dal)
                if ts[j, TS_STEP_B] > ts[j, TS_B]:
                    ts[j, TS_B] = ts[j, TS_STEP_B]
                if fm > ts[j, TS_FMAX]:
                    ts[j, TS_FMAX] = fm
                if ts[j, TS_LOADING] > 0.5 and (now >= ts[j, TS_AVAIL] or ts[j, TS_EXITS] > 0.0 or
                                                 ts[j, TS_EXIT_FIRST_S] >= 0.0):
                    put_shock(ev.shocks.rec, ev.shocks.n, ts[j, TS_ENTRY_TICK], float(j), ts[j, TS_ENTRY_ELEM],
                              ts[j, TS_ENTRY_H], ts[j, TS_ENTRY_ANGLE], fm,
                              ts[j, TS_LAST_ENTRY_S] - ts[j, TS_FIRST_ENTRY_S], float(SHOCK_ENTRY), cacc)
                    ts[j, TS_LOADING] = 0.0
                if hole and ts[j, TS_BREAKOUT] < 0.5:
                    if ts[j, TS_CENTRE_WAS] > 0.5 and ts[j, TS_CENTRE_ON] < 0.5:
                        put_shock(ev.shocks.rec, ev.shocks.n, float(tick_end), float(j), ts[j, TS_LAST_ELEM],
                                  ts[j, TS_LAST_H], ts[j, TS_LAST_ANGLE], ts[j, TS_FMAX], 0.0,
                                  float(SHOCK_BREAKTHROUGH), cacc)
                        ts[j, TS_BREAKOUT] = 1.0
                    elif ts[j, TS_EXITS] > 0.0 and now < prev:
                        put_shock(ev.shocks.rec, ev.shocks.n, float(tick_end), float(j), ts[j, TS_LAST_ELEM],
                                  ts[j, TS_LAST_H], ts[j, TS_LAST_ANGLE], ts[j, TS_FMAX], 0.0, float(SHOCK_BREAKOUT),
                                  cacc)
                        ts[j, TS_BREAKOUT] = 1.0
                if rev > 0.0 and ts[j, TS_EMIT_TICK] >= 0.0 and (tick_end - ts[j, TS_EMIT_TICK]) * tick_s >= rev:
                    emit_pass(st, ev, cacc, pose, step, istep, j, float(tick_end))
        elif prev > 0.0 and commit:
            if ts[j, TS_LOADING] > 0.5:
                put_shock(ev.shocks.rec, ev.shocks.n, ts[j, TS_ENTRY_TICK], float(j), ts[j, TS_ENTRY_ELEM],
                          ts[j, TS_ENTRY_H], ts[j, TS_ENTRY_ANGLE], ts[j, TS_FMAX],
                          ts[j, TS_LAST_ENTRY_S] - ts[j, TS_FIRST_ENTRY_S], float(SHOCK_ENTRY), cacc)
            exit_tick = float(np.floor(ts[j, TS_LAST_EXIT_S] / tick_s + 0.5)) if ts[j, TS_LAST_EXIT_S] > 0.0 \
                else float(tick_end)
            put_shock(ev.shocks.rec, ev.shocks.n, exit_tick, float(j), ts[j, TS_LAST_ELEM], ts[j, TS_LAST_H],
                      ts[j, TS_LAST_ANGLE], ts[j, TS_FMAX], ts[j, TS_LAST_EXIT_S] - ts[j, TS_EXIT_FIRST_S],
                      float(SHOCK_EXIT), cacc)
            emit_pass(st, ev, cacc, pose, step, istep, j, exit_tick)
            ts[j, TS_PASS_ON] = 0.0
            ts[j, TS_LOADING] = 0.0
            ts[j, TS_FMAX] = 0.0
            ts[j, TS_BREAKOUT] = 0.0
            ts[j, TS_EXIT_FIRST_S] = -1.0
            ts[j, TS_LAST_EXIT_S] = -1.0
            ts[j, TS_EMIT_TICK] = -1.0
        if commit:
            ts[j, TS_ACTIVE] = now


# =============================================================================== recut
CROSS_EPS = 1e-7        # bins: round-off margin of the memory crossings of a step (cut_step)
CROSS_EPS_S = 1e-6      # the same margin in the step's own coordinate
RECUT_R_MIN = 0.1       # struck point at least this share of the cutting radius from the axis (the teeth meet it)


@njit(types.void(F2, f8, f8, f8, f8, f8, f8, f8, f8, f8), **INL)
def recut_load(act, fx, fy, fz, mx, my, mz, t0, t1, z_t):
    """A new recut load (force and moment about the tip on the tool, T0, acting from t0 to t1, absolute s) into a free
    row of st.recut_act (a row that has ended by t0); when every row is still acting it joins the row that ends last
    with its remaining impulse kept: the merged load acts from t0 to max(t1, t_end) with (F_old (t_end - t0) +
    F_new (t1 - t0)) / (max(t1, t_end) - t0)."""
    n = act.shape[0]
    free = -1
    last = 0
    for q in range(n):
        if act[q, RA_T] <= t0:
            free = q
            break
        if act[q, RA_T] > act[last, RA_T]:
            last = q
    if free >= 0:
        act[free, RA_FX] = fx
        act[free, RA_FY] = fy
        act[free, RA_FZ] = fz
        act[free, RA_MX] = mx
        act[free, RA_MY] = my
        act[free, RA_MZ] = mz
        act[free, RA_T0] = t0
        act[free, RA_T] = t1
        act[free, RA_Z] = z_t
        return
    q = last
    s_old = act[q, RA_T] - (t0 if t0 > act[q, RA_T0] else act[q, RA_T0])
    t_end = act[q, RA_T] if act[q, RA_T] > t1 else t1
    span = t_end - t0
    a = s_old / span
    b = (t1 - t0) / span
    act[q, RA_FX] = a * act[q, RA_FX] + b * fx
    act[q, RA_FY] = a * act[q, RA_FY] + b * fy
    act[q, RA_FZ] = a * act[q, RA_FZ] + b * fz
    act[q, RA_MX] = a * act[q, RA_MX] + b * mx
    act[q, RA_MY] = a * act[q, RA_MY] + b * my
    act[q, RA_MZ] = a * act[q, RA_MZ] + b * mz
    act[q, RA_Z] = (a * act[q, RA_Z] + b * z_t) / (a + b)
    act[q, RA_T0] = t0
    act[q, RA_T] = t_end


RECUT_ARC_PER_CELL = 16.0  # angle samples of a station's circle: at most 1/16 of a pile cell apart (0.125 mm)
RECUT_N_ANG_MIN = 36
RECUT_N_ANG_MAX = 2048


@njit(types.void(TOOL_T, TABLE_T, OCCUPANCY_T, POSE_T, F1, I1, CACCUM_T, ACCUM_T, EVENTS_T, STATE_T, f8, INTERVALS_T),
      **OPTS)
def recut_scan(tool, table, occ, pose, step, istep, cacc, acc, ev, st, theta, iv):
    """Chips of the frozen occupancy grid that the teeth meet in the coming tooth period (scanned once per tooth
    period) are recut like material (docs/models/cutting.md "Recutting"):

    * which chips: the edges meet loose chips only where they advance into air. In a tooth period the teeth sweep
      every angle of the tool's surface of revolution once, while that surface moves by d = v T_tooth relative to
      the stock: the volume they meet is the layer the surface's leading side sweeps, max(0, n . d) per unit area
      (n the outward meridian normal of each station's band, r db per radian), less the part of it in material
      (the station's engaged intervals: there the layer is the chip being formed). The voxel's chips are spread
      over its volume, so the struck mass is that layer's volume in the voxel times the voxel's chip density, less
      what was already struck from the voxel in this macro step (occ.taken_kg). Chips inside the envelope (in the
      flutes, carried round and out by the rake faces) are not cut: their impulse m V is ~1e-7 N s;
    * the struck chips are cut across their section: uncut thickness h_r = the chips' thickness, width b_r = their
      width sqrt(V_chip / h_r) (V_chip from the volume-equivalent size) up to the edge length in the voxel, times
      the chips side by side along the edge (packing: solid fraction x edge length / chip width, at least one);
      force F = f(h_r) b_r n (table row TC_RECUT_ROW: tangential and thrust per unit width at h_r, edge forces
      included) at the struck layer's centroid (at least RECUT_R_MIN of the cutting radius from the axis), opposite
      to the cutting direction there plus the thrust along the struck surface's inward normal;
    * it acts the crossing time V_struck / (h_r b_r n V) (the struck chip volume cut at that section at the edge
      speed V): impulse F V_struck / (A V) and energy f(h_r) V_struck / h_r (the table's specific cutting energy
      times the volume), as a load in st.recut_act applied step by step by cut_step.

    Before 2026-11-10 the teeth met every chip inside the envelope cylinder: a drill took the chips of its own flutes
    (the D6.8 of examples/drill-tap-plate-1045: 0.39 mg struck at 0.17 m/s, 431 N sideways for 2 ms, the drill
    thrown into a 0.4 mm vibration; docs/models/loop.md 15). Before 2026-11-09 a voxel's whole mass took
    J = m V + u m / (rho V) within one L0 step. Recut rows (struck mass, voxel position in M) go to the chips
    domain."""
    cell = occ.cell_m[0]
    rcut = tool.scal[TC_R_CUT]
    if cell <= 0.0 or rcut <= 0.0:
        return
    omega = step[ST_OMEGA_RAD_S]
    abs_om = abs(omega)
    rho = step[ST_RHO_KG_M3]
    row = int(tool.scal[TC_RECUT_ROW])
    if abs_om <= 0.0 or rho <= 0.0 or row < 0 or row >= table.coef.shape[0]:
        return
    n_teeth = tool.scal[TC_N_TEETH]
    if n_teeth < 1.0:
        n_teeth = 1.0
    t_tooth = TWO_PI / (abs_om * n_teeth)
    sgn = 1.0 if omega > 0.0 else -1.0
    nxg = occ.mass_kg.shape[0]
    nyg = occ.mass_kg.shape[1]
    nzg = occ.mass_kg.shape[2]
    R = pose.R_S_T0
    tip = pose.tip_S_m
    # the advance of the tool's surface relative to the stock in the tooth period, T0
    vs = pose.v_tip_S_m_s
    dx = (R[0, 0] * vs[0] + R[1, 0] * vs[1] + R[2, 0] * vs[2]) * t_tooth
    dy = (R[0, 1] * vs[0] + R[1, 1] * vs[1] + R[2, 1] * vs[2]) * t_tooth
    dz = (R[0, 2] * vs[0] + R[1, 2] * vs[1] + R[2, 2] * vs[2]) * t_tooth
    if dx * dx + dy * dy + dz * dz <= 0.0:
        return
    zlo = tool.scal[TC_Z_LO]
    zhi = tool.scal[TC_Z_HI]
    lox = 1e300
    loy = 1e300
    loz = 1e300
    hix = -1e300
    hiy = -1e300
    hiz = -1e300
    for zz in (zlo, zhi):
        ax = tip[0] + R[0, 2] * zz
        ay = tip[1] + R[1, 2] * zz
        az = tip[2] + R[2, 2] * zz
        lox = min(lox, ax - rcut)
        loy = min(loy, ay - rcut)
        loz = min(loz, az - rcut)
        hix = max(hix, ax + rcut)
        hiy = max(hiy, ay + rcut)
        hiz = max(hiz, az + rcut)
    o = occ.origin_S_m
    i0 = max(0, int(np.floor((lox - o[0]) / cell)))
    i1 = min(nxg - 1, int(np.floor((hix - o[0]) / cell)))
    j0 = max(0, int(np.floor((loy - o[1]) / cell)))
    j1 = min(nyg - 1, int(np.floor((hiy - o[1]) / cell)))
    k0 = max(0, int(np.floor((loz - o[2]) / cell)))
    k1 = min(nzg - 1, int(np.floor((hiz - o[2]) / cell)))
    if i1 < i0 or j1 < j0 or k1 < k0:
        return
    ni = i1 - i0 + 1
    nj = j1 - j0 + 1
    nk = k1 - k0 + 1
    # per voxel: the swept layer's volume in air, its centroid (T0, from the tip) and its normal, volume weighted
    sw = np.zeros((ni, nj, nk, 10))
    for i in range(ni):
        for jj in range(nj):
            for k in range(nk):
                sw[i, jj, k, 7] = -1.0
    n_ang = int(math.ceil(TWO_PI * rcut * RECUT_ARC_PER_CELL / cell))
    if n_ang < RECUT_N_ANG_MIN:
        n_ang = RECUT_N_ANG_MIN
    if n_ang > RECUT_N_ANG_MAX:
        n_ang = RECUT_N_ANG_MAX
    dang = TWO_PI / n_ang
    nst = tool.st_z_m.shape[0]
    pk = tool.pk
    any_sw = False
    for s in range(nst):
        kk = tool.st_ptr[s]
        if tool.st_ptr[s + 1] <= kk:
            continue
        px = pk[kk, PK_P]
        py = pk[kk, PK_P + 1]
        pz = pk[kk, PK_P + 2]
        r = pk[kk, PK_RXY]
        n_r = 0.0
        if r > 0.0:
            n_r = (pk[kk, PK_N] * px + pk[kk, PK_N + 1] * py) / r
        n_z = pk[kk, PK_N + 2]
        band = r * pk[kk, PK_DB] * dang
        if band <= 0.0:
            continue
        for ia in range(n_ang):
            a = (ia + 0.5) * dang
            ca = math.cos(a)
            sa = math.sin(a)
            dn = n_r * (dx * ca + dy * sa) + n_z * dz
            if dn <= 0.0:
                continue
            if iv.count[s] > 0 and in_intervals(iv, s, a):
                continue
            qx = r * ca
            qy = r * sa
            sx = tip[0] + R[0, 0] * qx + R[0, 1] * qy + R[0, 2] * pz
            sy = tip[1] + R[1, 0] * qx + R[1, 1] * qy + R[1, 2] * pz
            sz = tip[2] + R[2, 0] * qx + R[2, 1] * qy + R[2, 2] * pz
            i = int(np.floor((sx - o[0]) / cell)) - i0
            jj = int(np.floor((sy - o[1]) / cell)) - j0
            k = int(np.floor((sz - o[2]) / cell)) - k0
            if i < 0 or i >= ni or jj < 0 or jj >= nj or k < 0 or k >= nk:
                continue
            if occ.mass_kg[i + i0, jj + j0, k + k0] <= 1e-18:
                continue
            dv = dn * band
            sw[i, jj, k, 0] += dv
            sw[i, jj, k, 1] += dv * qx
            sw[i, jj, k, 2] += dv * qy
            sw[i, jj, k, 3] += dv * pz
            sw[i, jj, k, 4] += dv * n_r * ca
            sw[i, jj, k, 5] += dv * n_r * sa
            sw[i, jj, k, 6] += dv * n_z
            # the arc of the band in the voxel (sample offsets from the first, unwrapped within half a turn)
            if sw[i, jj, k, 7] < 0.0:
                sw[i, jj, k, 7] = float(ia)
            off = ia - int(sw[i, jj, k, 7])
            if off > n_ang // 2:
                off -= n_ang
            elif off < -(n_ang // 2):
                off += n_ang
            if off < sw[i, jj, k, 8]:
                sw[i, jj, k, 8] = float(off)
            if off > sw[i, jj, k, 9]:
                sw[i, jj, k, 9] = float(off)
            any_sw = True
    if not any_sw:
        return
    tick = float(istep[SI_TICK_END])
    t_now = istep[SI_TICK_END] * step[ST_TICK_S] - step[ST_DT_S]      # the step's start: the loads act from here
    RM = pose.R_M_S
    tM = pose.t_M_S_m
    nt = tool.tooth_a0_rad.shape[0]
    l_edge = zhi - zlo
    if l_edge > cell:
        l_edge = cell
    if l_edge <= 0.0:
        l_edge = cell
    r_min = RECUT_R_MIN * rcut
    coef = table.coef
    vol = cell * cell * cell
    for i in range(ni):
        for jj in range(nj):
            for k in range(nk):
                v_sw = sw[i, jj, k, 0]
                if v_sw <= 0.0:
                    continue
                gi = i + i0
                gj = jj + j0
                gk = k + k0
                m_vox = occ.mass_kg[gi, gj, gk]
                m_s = m_vox * v_sw / vol
                if m_s > m_vox - occ.taken_kg[gi, gj, gk]:
                    m_s = m_vox - occ.taken_kg[gi, gj, gk]
                if m_s <= 1e-18:
                    continue
                occ.taken_kg[gi, gj, gk] += m_s
                cx = sw[i, jj, k, 1] / v_sw
                cy = sw[i, jj, k, 2] / v_sw
                cz = sw[i, jj, k, 3] / v_sw
                nnx = sw[i, jj, k, 4]
                nny = sw[i, jj, k, 5]
                nnz = sw[i, jj, k, 6]
                nn = math.sqrt(nnx * nnx + nny * nny + nnz * nnz)
                if nn > 0.0:
                    nnx /= nn
                    nny /= nn
                    nnz /= nn
                rc = math.sqrt(cx * cx + cy * cy)
                if rc > 0.0:
                    urx = cx / rc
                    ury = cy / rc
                else:
                    urx = 1.0
                    ury = 0.0
                if rc < r_min:
                    rc = r_min
                V = abs_om * rc
                # the chips' section: thickness, width (platelet of the volume-equivalent size), side by side
                h_r = occ.thickness_m[gi, gj, gk]
                d_c = occ.size_m[gi, gj, gk]
                if h_r <= 0.0:
                    h_r = d_c
                if h_r <= 0.0:
                    continue
                if d_c <= 0.0:
                    d_c = h_r
                v_chip = (math.pi / 6.0) * d_c * d_c * d_c
                w_c = math.sqrt(v_chip / h_r)
                if w_c < h_r:
                    w_c = h_r
                b_r = w_c if w_c < l_edge else l_edge
                phi = m_vox / (rho * vol)
                n_side = phi * l_edge / w_c
                if n_side < 1.0:
                    n_side = 1.0
                seg = table_segment(table, row, h_r, cacc)
                f_t = (coef[row, seg, 0, COL_FT] + coef[row, seg, 1, COL_FT] * h_r) + \
                    (coef[row, seg, 0, COL_ET] + coef[row, seg, 1, COL_ET] * h_r)
                f_r = (coef[row, seg, 0, COL_FR] + coef[row, seg, 1, COL_FR] * h_r) + \
                    (coef[row, seg, 0, COL_ER] + coef[row, seg, 1, COL_ER] * h_r)
                if f_t <= 0.0:
                    continue
                width = b_r * n_side
                Ft = f_t * width
                Fr = f_r * width
                # the voxel's chips are spread uniformly over the layer, so the edges meet them all along the band's
                # arc in the voxel, over the time they take to sweep it (each angle once per tooth period): the mean
                # force cuts the struck volume at the specific energy f_t / h_r over that time, up to the section
                # force of whole chips side by side (dense chips). Same impulse and energy either way
                v_s = m_s / rho
                arc = (sw[i, jj, k, 9] - sw[i, jj, k, 8] + 1.0) * dang
                t_pass = arc / abs_om
                if t_pass > t_tooth:
                    t_pass = t_tooth
                F_mean = v_s * (f_t / h_r) / (t_pass * V)
                if F_mean < Ft:
                    Fr *= F_mean / Ft
                    Ft = F_mean
                t_cross = v_s * (f_t / h_r) / (Ft * V)
                # on the tool: -Ft along the cutting direction, the thrust against the struck surface's normal
                cdx = -sgn * ury
                cdy = sgn * urx
                fx = -Ft * cdx - Fr * nnx
                fy = -Ft * cdy - Fr * nny
                fz = -Fr * nnz
                px = rc * urx
                py = rc * ury
                recut_load(st.recut_act, fx, fy, fz, py * fz - cz * fy, cz * fx - px * fz, px * fy - py * fx, t_now,
                           t_now + t_cross, cz)
                cacc.scalars[CA_RECUT_KG] += m_s
                acc.scalars[ACC_RECUT_MASS_KG] += m_s
                # the tooth that strikes it: the next one to reach its angle in the cutting sense
                av = math.atan2(ury, urx)
                best = 0
                best_d = 1e300
                for t in range(nt):
                    d = sgn * (av - (theta + tool.tooth_a0_rad[t]))
                    d -= TWO_PI * np.floor(d / TWO_PI)
                    if d < best_d:
                        best_d = d
                        best = t
                near = 0
                near_d = 1e300
                for s in range(nst):
                    d = abs(tool.st_z_m[s] - cz)
                    if d < near_d:
                        near_d = d
                        near = s
                vx = o[0] + (gi + 0.5) * cell
                vy = o[1] + (gj + 0.5) * cell
                vz = o[2] + (gk + 0.5) * cell
                put_recut(ev.recut.rec, ev.recut.n, tick, float(best), float(near), m_s,
                          RM[0, 0] * vx + RM[0, 1] * vy + RM[0, 2] * vz + tM[0],
                          RM[1, 0] * vx + RM[1, 1] * vy + RM[1, 2] * vz + tM[1], cacc)


# =============================================================================== step totals -> outputs
@njit(types.void(POSE_T, F1, I1, AXES_T, ACCUM_T, CACCUM_T, OUT_T, b1, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8,
                 f8, f8, f8, f8, f8, f8, f8), **OPTS)
def finish_step(pose, step, istep, axes, acc, cacc, out, commit, Fx, Fy, Fz, Mx, My, Mz, shaft, feed,
                spindle_torque, power, n_in, pF0, pF1, pF2, pM0, pM1, pM2):
    """Impulses of the step (on the tool, T0, moment about the tip) -> StepOut means, the per-tick wrenches of
    tool and part, axis loads, spindle torque, peaks and integrals (architecture 5.4 "Loads to L1"). The peaks
    (largest |F| of the macro step and of each tick, their wrenches on tool and part) take the step's peak wrench
    (pF, pM: T0, about the tip): the instantaneous wrench at the step's start or end where it exceeds the mean."""
    dt = step[ST_DT_S]
    inv = 1.0 / dt
    wF0 = Fx * inv
    wF1 = Fy * inv
    wF2 = Fz * inv
    wM0 = Mx * inv
    wM1 = My * inv
    wM2 = Mz * inv
    out.wrench_T0[0] = wF0
    out.wrench_T0[1] = wF1
    out.wrench_T0[2] = wF2
    out.wrench_T0[3] = wM0
    out.wrench_T0[4] = wM1
    out.wrench_T0[5] = wM2
    R = pose.R_S_T0
    fs0 = R[0, 0] * wF0 + R[0, 1] * wF1 + R[0, 2] * wF2
    fs1 = R[1, 0] * wF0 + R[1, 1] * wF1 + R[1, 2] * wF2
    fs2 = R[2, 0] * wF0 + R[2, 1] * wF1 + R[2, 2] * wF2
    ms0 = R[0, 0] * wM0 + R[0, 1] * wM1 + R[0, 2] * wM2
    ms1 = R[1, 0] * wM0 + R[1, 1] * wM1 + R[1, 2] * wM2
    ms2 = R[2, 0] * wM0 + R[2, 1] * wM1 + R[2, 2] * wM2
    out.force_S_N[0] = fs0
    out.force_S_N[1] = fs1
    out.force_S_N[2] = fs2
    out.misc[SO_TORQUE_NM] = spindle_torque * inv
    out.misc[SO_POWER_W] = power * inv
    out.misc[SO_ENGAGED] = n_in
    if not commit:
        return
    a = acc.scalars
    a[ACC_SHAFT_WORK_J] += shaft
    a[ACC_FEED_WORK_J] += feed
    if n_in > 0.0:
        a[ACC_IN_CUT_S] += dt
        cacc.scalars[CA_STEPS_IN_CUT] += 1.0
    fmag = math.sqrt(pF0 * pF0 + pF1 * pF1 + pF2 * pF2)
    ps0 = R[0, 0] * pF0 + R[0, 1] * pF1 + R[0, 2] * pF2
    ps1 = R[1, 0] * pF0 + R[1, 1] * pF1 + R[1, 2] * pF2
    ps2 = R[2, 0] * pF0 + R[2, 1] * pF1 + R[2, 2] * pF2
    pm0 = R[0, 0] * pM0 + R[0, 1] * pM1 + R[0, 2] * pM2
    pm1 = R[1, 0] * pM0 + R[1, 1] * pM1 + R[1, 2] * pM2
    pm2 = R[2, 0] * pM0 + R[2, 1] * pM1 + R[2, 2] * pM2
    if fmag > cacc.scalars[CA_FPEAK_INST_N]:
        cacc.scalars[CA_FPEAK_INST_N] = fmag
    tq = abs(spindle_torque * inv)
    pw = abs(spindle_torque * inv * step[ST_OMEGA_RAD_S])
    if tq > a[ACC_TORQUE_PEAK_NM]:
        a[ACC_TORQUE_PEAK_NM] = tq
    if pw > a[ACC_POWER_PEAK_W]:
        a[ACC_POWER_PEAK_W] = pw
    if fmag > a[ACC_FORCE_PEAK_N]:
        a[ACC_FORCE_PEAK_N] = fmag
        RMp = pose.R_M_S
        cacc.scalars[CA_FPEAK_MX] = RMp[0, 0] * ps0 + RMp[0, 1] * ps1 + RMp[0, 2] * ps2
        cacc.scalars[CA_FPEAK_MY] = RMp[1, 0] * ps0 + RMp[1, 1] * ps1 + RMp[1, 2] * ps2
        cacc.scalars[CA_FPEAK_MZ] = RMp[2, 0] * ps0 + RMp[2, 1] * ps1 + RMp[2, 2] * ps2
    if abs(feed * inv) > cacc.scalars[CA_FEED_POWER_PEAK]:
        cacc.scalars[CA_FEED_POWER_PEAK] = abs(feed * inv)
    c = cacc.scalars
    c[CA_STEPS] += 1.0
    c[CA_OMEGA_INT] += abs(step[ST_OMEGA_RAD_S]) * dt
    c[CA_TIME] += dt
    c[CA_TORQUE_INT] += spindle_torque
    c[CA_FX_INT] += Fx
    c[CA_FY_INT] += Fy
    c[CA_FZ_INT] += Fz
    c[CA_POWER_INT] += power
    row = istep[SI_TICK_ROW]
    if row < 0 or row >= acc.tick_wrench.shape[0]:
        return
    tf = step[ST_TICK_FRAC]
    tw = acc.tick_wrench
    tw[row, WRENCH_TOOL_MEAN, 0] += wF0 * tf
    tw[row, WRENCH_TOOL_MEAN, 1] += wF1 * tf
    tw[row, WRENCH_TOOL_MEAN, 2] += wF2 * tf
    tw[row, WRENCH_TOOL_MEAN, 3] += wM0 * tf
    tw[row, WRENCH_TOOL_MEAN, 4] += wM1 * tf
    tw[row, WRENCH_TOOL_MEAN, 5] += wM2 * tf
    # on the part (W, about the W origin): -F at the tip
    RW = pose.R_W_S
    tW = pose.t_W_S_m
    tip = pose.tip_S_m
    fw0 = -(RW[0, 0] * fs0 + RW[0, 1] * fs1 + RW[0, 2] * fs2)
    fw1 = -(RW[1, 0] * fs0 + RW[1, 1] * fs1 + RW[1, 2] * fs2)
    fw2 = -(RW[2, 0] * fs0 + RW[2, 1] * fs1 + RW[2, 2] * fs2)
    px = RW[0, 0] * tip[0] + RW[0, 1] * tip[1] + RW[0, 2] * tip[2] + tW[0]
    py = RW[1, 0] * tip[0] + RW[1, 1] * tip[1] + RW[1, 2] * tip[2] + tW[1]
    pz = RW[2, 0] * tip[0] + RW[2, 1] * tip[1] + RW[2, 2] * tip[2] + tW[2]
    mw0 = -(RW[0, 0] * ms0 + RW[0, 1] * ms1 + RW[0, 2] * ms2) + (py * fw2 - pz * fw1)
    mw1 = -(RW[1, 0] * ms0 + RW[1, 1] * ms1 + RW[1, 2] * ms2) + (pz * fw0 - px * fw2)
    mw2 = -(RW[2, 0] * ms0 + RW[2, 1] * ms1 + RW[2, 2] * ms2) + (px * fw1 - py * fw0)
    tw[row, WRENCH_WORK_MEAN, 0] += fw0 * tf
    tw[row, WRENCH_WORK_MEAN, 1] += fw1 * tf
    tw[row, WRENCH_WORK_MEAN, 2] += fw2 * tf
    tw[row, WRENCH_WORK_MEAN, 3] += mw0 * tf
    tw[row, WRENCH_WORK_MEAN, 4] += mw1 * tf
    tw[row, WRENCH_WORK_MEAN, 5] += mw2 * tf
    old = tw[row, WRENCH_TOOL_PEAK, 0] ** 2 + tw[row, WRENCH_TOOL_PEAK, 1] ** 2 + tw[row, WRENCH_TOOL_PEAK, 2] ** 2
    if fmag * fmag > old:
        gw0 = -(RW[0, 0] * ps0 + RW[0, 1] * ps1 + RW[0, 2] * ps2)
        gw1 = -(RW[1, 0] * ps0 + RW[1, 1] * ps1 + RW[1, 2] * ps2)
        gw2 = -(RW[2, 0] * ps0 + RW[2, 1] * ps1 + RW[2, 2] * ps2)
        tw[row, WRENCH_TOOL_PEAK, 0] = pF0
        tw[row, WRENCH_TOOL_PEAK, 1] = pF1
        tw[row, WRENCH_TOOL_PEAK, 2] = pF2
        tw[row, WRENCH_TOOL_PEAK, 3] = pM0
        tw[row, WRENCH_TOOL_PEAK, 4] = pM1
        tw[row, WRENCH_TOOL_PEAK, 5] = pM2
        tw[row, WRENCH_WORK_PEAK, 0] = gw0
        tw[row, WRENCH_WORK_PEAK, 1] = gw1
        tw[row, WRENCH_WORK_PEAK, 2] = gw2
        tw[row, WRENCH_WORK_PEAK, 3] = -(RW[0, 0] * pm0 + RW[0, 1] * pm1 + RW[0, 2] * pm2) + (py * gw2 - pz * gw1)
        tw[row, WRENCH_WORK_PEAK, 4] = -(RW[1, 0] * pm0 + RW[1, 1] * pm1 + RW[1, 2] * pm2) + (pz * gw0 - px * gw2)
        tw[row, WRENCH_WORK_PEAK, 5] = -(RW[2, 0] * pm0 + RW[2, 1] * pm1 + RW[2, 2] * pm2) + (px * gw1 - py * gw0)
    acc.tick_spindle_torque_Nm[row] += spindle_torque * inv * tf
    # axis loads: force the process exerts on each axis' moving link (side +1 tool side, -1 work side)
    RM = pose.R_M_S
    tM = pose.t_M_S_m
    fm0 = RM[0, 0] * fs0 + RM[0, 1] * fs1 + RM[0, 2] * fs2
    fm1 = RM[1, 0] * fs0 + RM[1, 1] * fs1 + RM[1, 2] * fs2
    fm2 = RM[2, 0] * fs0 + RM[2, 1] * fs1 + RM[2, 2] * fs2
    mm0 = RM[0, 0] * ms0 + RM[0, 1] * ms1 + RM[0, 2] * ms2
    mm1 = RM[1, 0] * ms0 + RM[1, 1] * ms1 + RM[1, 2] * ms2
    mm2 = RM[2, 0] * ms0 + RM[2, 1] * ms1 + RM[2, 2] * ms2
    qx = RM[0, 0] * tip[0] + RM[0, 1] * tip[1] + RM[0, 2] * tip[2] + tM[0]
    qy = RM[1, 0] * tip[0] + RM[1, 1] * tip[1] + RM[1, 2] * tip[2] + tM[1]
    qz = RM[2, 0] * tip[0] + RM[2, 1] * tip[1] + RM[2, 2] * tip[2] + tM[2]
    na = min(axes.dir_M.shape[0], acc.tick_axis_force_N.shape[1])
    for ia in range(na):
        side = axes.side[ia]
        if side == 0.0:
            continue
        d0 = axes.dir_M[ia, 0]
        d1 = axes.dir_M[ia, 1]
        d2 = axes.dir_M[ia, 2]
        if axes.rotary[ia] != 0:
            rx = qx - axes.point_M_m[ia, 0]
            ry = qy - axes.point_M_m[ia, 1]
            rz = qz - axes.point_M_m[ia, 2]
            load = d0 * (ry * fm2 - rz * fm1 + mm0) + d1 * (rz * fm0 - rx * fm2 + mm1) + d2 * (rx * fm1 - ry * fm0 + mm2)
        else:
            load = d0 * fm0 + d1 * fm1 + d2 * fm2
        acc.tick_axis_force_N[row, ia] += side * load * tf


# =============================================================================== critical sections
@njit(types.void(TOOL_T, OUT_T, CACCUM_T, f8), **OPTS)
def section_peaks(tool, out, cacc, dt):
    """Bending moment and von Mises stress at the tool's critical sections from the station impulses
    binned below each section (out.sec rows: sum F (3), sum z F (3), sum M about the station point (3))."""
    nsec = tool.sec_z_m.shape[0]
    if nsec == 0 or cacc.sections.shape[0] < nsec:
        return
    sF0 = 0.0
    sF1 = 0.0
    sF2 = 0.0
    sZ0 = 0.0
    sZ1 = 0.0
    sZ2 = 0.0
    sM0 = 0.0
    sM1 = 0.0
    sM2 = 0.0
    inv = 1.0 / dt
    for k in range(nsec):
        sF0 += out.sec[k, 0]
        sF1 += out.sec[k, 1]
        sF2 += out.sec[k, 2]
        sZ0 += out.sec[k, 3]
        sZ1 += out.sec[k, 4]
        sZ2 += out.sec[k, 5]
        sM0 += out.sec[k, 6]
        sM1 += out.sec[k, 7]
        sM2 += out.sec[k, 8]
        zk = tool.sec_z_m[k]
        # moment about (0, 0, z_k) of the forces below: sum M_st + (z_st - z_k) e_z x F_st
        mx = (sM0 - (sZ1 - zk * sF1)) * inv
        my = (sM1 + (sZ0 - zk * sF0)) * inv
        mz = sM2 * inv
        mb = math.sqrt(mx * mx + my * my)
        zb = tool.sec_Zb_m3[k]
        zp = tool.sec_Zp_m3[k]
        sig = mb / zb if zb > 0.0 else 0.0
        tau = abs(mz) / zp if zp > 0.0 else 0.0
        vm = math.sqrt(sig * sig + 3.0 * tau * tau)
        if mb > cacc.sections[k, 0]:
            cacc.sections[k, 0] = mb
        if vm > cacc.sections[k, 1]:
            cacc.sections[k, 1] = vm


# =============================================================================== surface events
@njit(types.void(EVENTS_T, CACCUM_T, TOOL_T, DEFLECTION_T, i8, i8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8,
                 f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8), **OPTS)
def _surface_event(ev, cacc, tool, defl, e, j, tick, dTx, dTy, dTz, toff, vex, vey, vez, h, nx, ny, nz, ucx, ucy, uz,
                   rcx, rcy, rz, pzr, px, py, e00, e01, e02, e10, e11, e12, e20, e21, e22, slot):
    """One surface event (bus SURFACE_EVENT_COLUMNS): the element point toff seconds from its evaluation
    (S), the outward normal of the generated surface (-n), h, edge radius, VB and u_rel at generation."""
    qx = dTx + toff * vex
    qy = dTy + toff * vey
    qz = dTz + toff * vez
    uqx = ucx + (rcy * pzr - rz * py)
    uqy = ucy + (rz * px - rcx * pzr)
    uqz = uz + (rcx * py - rcy * px)
    put_surface(ev.surface.rec, ev.surface.n, tick, float(j), float(tool.elem[e]), e00 * qx + e01 * qy + e02 * qz,
                e10 * qx + e11 * qy + e12 * qz, e20 * qx + e21 * qy + e22 * qz, -(e00 * nx + e01 * ny + e02 * nz),
                -(e10 * nx + e11 * ny + e12 * nz), -(e20 * nx + e21 * ny + e22 * nz), h, tool.edge_radius_m[e],
                tool.VB_m[e], e00 * uqx + e01 * uqy + e02 * uqz - defl.u_work_S_m[0],
                e10 * uqx + e11 * uqy + e12 * uqz - defl.u_work_S_m[1],
                e20 * uqx + e21 * uqy + e22 * uqz - defl.u_work_S_m[2], slot, cacc)


# =============================================================================== rotating tools
GATE_EPS = 1e-12        # rad: an arc this close to an interval boundary takes the general gate (identical results)


@njit(types.void(TOOL_T, TABLE_T, MEMORY_T, INTERVALS_T, REFILL_T, F1, I1, POSE_T, DEFLECTION_T, OCCUPANCY_T,
                 PACKING_T, AXES_T, STATE_T, ACCUM_T, CACCUM_T, OUT_T, EVENTS_T), **RAW)
def cut_step(tool, table, mem, iv, refill, step, istep, pose, defl, occ, pack, axes, st, acc, cacc, out, ev):
    """One L0 step of one contact of a rotating tool (see the module doc and docs/models/cutting.md).

    Everything is evaluated at the engaged centroid of each element's swept arc (the step midpoint for
    elements fully inside an engaged interval), and the impulse is the force there times the engaged
    duration: the closed-form integral of the piecewise-linear force over the swept angle clipped to
    the intervals (exact for forces linear in h and h linear in the angle within a step).

    Cost (docs/performance.md "cut_step"): only stations with engaged intervals (or loads written in the last
    two steps) are visited; an element whose arc lies inside or outside a station's single interval is
    classified with two comparisons (the general :func:`gate` within GATE_EPS of a boundary); the element
    constants come from one packed row per element in station-list order (``tool.pk`` / ``tool.pki``, row kk for
    element st_elem[kk]); compiled without reference counting (it allocates nothing)."""
    dt = step[ST_DT_S]
    if dt <= 0.0:
        return
    commit = istep[SI_COMMIT] != 0
    omega = step[ST_OMEGA_RAD_S]
    tick_end = istep[SI_TICK_END]
    l0 = istep[SI_L0_TICKS]
    tick0 = tick_end - l0
    tick_s = step[ST_TICK_S]
    t_start = tick_end * tick_s - dt
    theta = step[ST_THETA_MID_RAD]
    cm = step[ST_COS_MID]
    sm = step[ST_SIN_MID]
    dal = omega * dt
    half = 0.5 * abs(dal)
    inv_dal = 1.0 / dal if dal != 0.0 else 0.0
    abs_dal = abs(dal)
    nst = tool.st_z_m.shape[0]
    nt = st.tooth.shape[0]
    nb = mem.point_S_m.shape[1]
    inv_bin = nb / TWO_PI
    bin_w = TWO_PI / nb
    sag_c = 0.5 * bin_w * bin_w
    bps = dal * inv_bin
    inv_bps = 1.0 / bps if bps != 0.0 else 0.0
    stale_ticks = istep[SI_STALE_TICKS]
    stale_before = istep[SI_STALE_BEFORE]
    ratio = tool.scal[TC_RECOVERY_RATIO]
    pd_max = tool.scal[TC_PD_MAX_LEN]
    eps_t = tool.scal[TC_TANGENT_VN]
    slot = float(istep[SI_SLOT])
    float_k = tool.scal[TC_FLOAT_K]
    u_f = st.misc[MS_FLOAT_U] if float_k > 0.0 else 0.0
    nlo = table.n_lo[0]
    nseg = table.coef.shape[1]
    coef = table.coef
    t_hmin = table.h_min_m
    t_dlo = table.inv_dlo
    t_dhi = table.inv_dhi
    t_hwrite = table.h_write_m
    pk = tool.pk
    pki = tool.pki
    st_ptr = tool.st_ptr
    st_elem = tool.st_elem
    st_zs = tool.st_z_m
    st_edge = st.st_edge
    iv_tick = float(iv.tick[0])
    h_reach = 2.0 * tool.scal[TC_R_CUT] if tool.scal[TC_R_CUT] > 0.0 else 1.0
    pts = mem.point_S_m
    mtick = mem.tick
    mvalid = mem.valid
    fl_a = st.flags
    h_last = st.h_last_m
    t_last = st.tick_last
    pend = st.pend
    pend_tick = st.pend_tick
    vn_last = st.vn_last_m_s
    ea = st.elem_acc
    ts = st.tooth
    written = st.st_written
    ust = defl.u_st_T0_m
    vst = defl.v_st_T0_m_s
    uw = defl.u_work_S_m
    vw = defl.v_work_S_m_s
    sto = out.station_T0
    sec = out.sec
    iv_cnt = iv.count
    iv_lo = iv.lo_rad
    iv_hi = iv.hi_rad
    R = pose.R_S_T0
    # G = R Rz(theta_mid): T (at the step midpoint) -> S
    g00 = R[0, 0] * cm + R[0, 1] * sm
    g01 = R[0, 1] * cm - R[0, 0] * sm
    g02 = R[0, 2]
    g10 = R[1, 0] * cm + R[1, 1] * sm
    g11 = R[1, 1] * cm - R[1, 0] * sm
    g12 = R[1, 2]
    g20 = R[2, 0] * cm + R[2, 1] * sm
    g21 = R[2, 1] * cm - R[2, 0] * sm
    g22 = R[2, 2]
    # tool tip minus the work deflection, in T: cT = G^T (tip - u_work)
    ox = pose.tip_S_m[0] - uw[0]
    oy = pose.tip_S_m[1] - uw[1]
    oz = pose.tip_S_m[2] - uw[2]
    cTx = g00 * ox + g10 * oy + g20 * oz
    cTy = g01 * ox + g11 * oy + g21 * oz
    cTz = g02 * ox + g12 * oy + g22 * oz
    # tip velocity relative to the (vibrating) material, in T
    wvx = pose.v_tip_S_m_s[0] - vw[0]
    wvy = pose.v_tip_S_m_s[1] - vw[1]
    wvz = pose.v_tip_S_m_s[2] - vw[2]
    vTx = g00 * wvx + g10 * wvy + g20 * wvz
    vTy = g01 * wvx + g11 * wvy + g21 * wvz
    vTz = g02 * wvx + g12 * wvy + g22 * wvz
    # the work's deflection and vibration velocity in T0 (process damping's wave memory, the chips' loads)
    uw0x = R[0, 0] * uw[0] + R[1, 0] * uw[1] + R[2, 0] * uw[2]
    uw0y = R[0, 1] * uw[0] + R[1, 1] * uw[1] + R[2, 1] * uw[2]
    uw0z = R[0, 2] * uw[0] + R[1, 2] * uw[1] + R[2, 2] * uw[2]
    vw0x = R[0, 0] * vw[0] + R[1, 0] * vw[1] + R[2, 0] * vw[2]
    vw0y = R[0, 1] * vw[0] + R[1, 1] * vw[1] + R[2, 1] * vw[2]
    vw0z = R[0, 2] * vw[0] + R[1, 2] * vw[1] + R[2, 2] * vw[2]
    pd_ring = st.pd_ring
    pd_aux = st.pd_aux
    pd_dx = pd_max / (PD_RING_N - 1)
    n_teeth_f = tool.scal[TC_N_TEETH]
    inv_nt = 1.0 / n_teeth_f if n_teeth_f >= 1.0 else 1.0
    # work vibration velocity alone in T (its share of the force power goes to the deflection work)
    vWx = g00 * vw[0] + g10 * vw[1] + g20 * vw[2]
    vWy = g01 * vw[0] + g11 * vw[1] + g21 * vw[2]
    vWz = g02 * vw[0] + g12 * vw[1] + g22 * vw[2]
    # angular velocity of T relative to S, in T: omega e_z + G^T w_frame
    fwx = pose.w_T0_S_rad_s[0]
    fwy = pose.w_T0_S_rad_s[1]
    fwz = pose.w_T0_S_rad_s[2]
    OTx = g00 * fwx + g10 * fwy + g20 * fwz
    OTy = g01 * fwx + g11 * fwy + g21 * fwz
    OTz = omega + g02 * fwx + g12 * fwy + g22 * fwz
    # tangency threshold of boundary surface events, and the trailing angle (T0) of the feed for floor events
    v_rel = math.sqrt(vTx * vTx + vTy * vTy + vTz * vTz)
    vt_tol = eps_t * v_rel
    t0x = cm * vTx - sm * vTy
    t0y = sm * vTx + cm * vTy
    floor_on = commit and t0x * t0x + t0y * t0y > (0.5 * v_rel) ** 2 and v_rel > 0.0
    a_trail = math.atan2(-t0y, -t0x) if floor_on else 0.0
    if a_trail < 0.0:
        a_trail += TWO_PI
    # station rows written last step are cleared (they are rewritten when engaged); the stations to visit lie in
    # [s_first, s_last]: engaged now or written in the last two steps
    contact_any = False
    s_first = nst
    s_last = -1
    for s in range(nst):
        if written[s] != 0:
            for c in range(6):
                sto[s, c] = 0.0
        if iv_cnt[s] > 0:
            contact_any = True
        if written[s] != 0 or iv_cnt[s] > 0:
            if s < s_first:
                s_first = s
            s_last = s
    for k in range(sec.shape[0]):
        for c in range(9):
            sec[k, c] = 0.0
    if commit:
        for j in range(nt):
            ts[j, TS_FX] = 0.0
            ts[j, TS_FY] = 0.0
            ts[j, TS_FZ] = 0.0
            ts[j, TS_NEW] = 0.0
            ts[j, TS_NOW] = 0.0
            ts[j, TS_EXITS] = 0.0
            ts[j, TS_CENTRE_WAS] = ts[j, TS_CENTRE_ON]
            ts[j, TS_CENTRE_ON] = 0.0
            ts[j, TS_STEP_DV] = 0.0
            ts[j, TS_STEP_B] = 0.0
            ts[j, TS_AVAIL] = 0.0
            for c in range(TS_S_PX, TS_S_NZ + 1):
                ts[j, c] = 0.0
    floor_test = floor_on and contact_any
    FTx = 0.0
    FTy = 0.0
    FTz = 0.0
    MTx = 0.0
    MTy = 0.0
    MTz = 0.0
    e_shear = 0.0
    e_rake = 0.0
    e_plough = 0.0
    e_flank = 0.0
    e_pd = 0.0
    e_forming = 0.0
    e_def = 0.0
    vol_chip = 0.0
    vol_disp = 0.0
    n_in = 0.0
    edge_len = 0.0
    v_int = 0.0
    h_max = 0.0
    Kz = 0.0
    # instantaneous wrench (T, about the station axis points as the element sums) at the step's start and end
    pS0 = 0.0
    pS1 = 0.0
    pS2 = 0.0
    pS3 = 0.0
    pS4 = 0.0
    pS5 = 0.0
    pE0 = 0.0
    pE1 = 0.0
    pE2 = 0.0
    pE3 = 0.0
    pE4 = 0.0
    pE5 = 0.0
    for s in range(s_first, s_last + 1):
        k0 = st_ptr[s]
        k1 = st_ptr[s + 1]
        cnt = iv_cnt[s]
        if k1 <= k0 or (cnt <= 0 and written[s] == 0):
            continue
        # station deflection: T0 -> T (rotate by -theta_mid); the float shifts the tool along z
        ux = ust[s, 0]
        uy = ust[s, 1]
        uTx = cm * ux + sm * uy
        uTy = cm * uy - sm * ux
        uTz = ust[s, 2] + u_f
        rx = ust[s, 3]
        ry = ust[s, 4]
        rTx = cm * rx + sm * ry
        rTy = cm * ry - sm * rx
        rTz = ust[s, 5]
        vx = vst[s, 0]
        vy = vst[s, 1]
        vuTx = cm * vx + sm * vy
        vuTy = cm * vy - sm * vx
        vuTz = vst[s, 2]
        vx = vst[s, 3]
        vy = vst[s, 4]
        vrTx = cm * vx + sm * vy
        vrTy = cm * vy - sm * vx
        vrTz = vst[s, 5]
        zst = st_zs[s]
        # the station's interval for the fast classification (one interval, not a full circle, wide margins)
        fast = False
        full = False
        lo_s = 0.0
        wid_s = 0.0
        if cnt == 1 and half > 0.0:
            lo_s = iv_lo[s, 0]
            wid_s = iv_hi[s, 0] - lo_s
            if wid_s >= TWO_PI - 1e-12:
                full = True
            elif lo_s >= 0.0 and lo_s < TWO_PI and wid_s >= 0.0 and wid_s < TWO_PI - 4.0 * (half + bin_w):
                fast = True
        sFx = 0.0
        sFy = 0.0
        sFz = 0.0
        sMx = 0.0
        sMy = 0.0
        sMz = 0.0
        any_in = False
        centre = tool.st_centre[s] != 0
        for kk in range(k0, k1):
            e = st_elem[kk]
            j = pki[kk, PKI_TOOTH]
            flags = fl_a[e]
            if commit and cnt > 0:
                ts[j, TS_AVAIL] += 1.0
            a = theta + pk[kk, PK_ALPHA0]
            if a >= TWO_PI:
                a -= TWO_PI
            if floor_test and pki[kk, PKI_AXIAL] != 0:
                # axial-normal edges (end, corner bottom, wiper) leave the floor where they pass the trailing side
                d_ang = a_trail - a
                if d_ang > math.pi:
                    d_ang -= TWO_PI
                elif d_ang < -math.pi:
                    d_ang += TWO_PI
                if abs(d_ang) <= half:
                    fpx = pk[kk, PK_P]
                    fpy = pk[kk, PK_P + 1]
                    fpz = pk[kk, PK_P + 2]
                    fpr = fpz - zst
                    fux = uTx + (rTy * fpr - rTz * fpy)
                    fuy = uTy + (rTz * fpx - rTx * fpr)
                    fuz = uTz + (rTx * fpy - rTy * fpx)
                    qx = cTx + fpx + fux
                    qy = cTy + fpy + fuy
                    qz = cTz + fpz + fuz
                    fnx = pk[kk, PK_N]
                    fny = pk[kk, PK_N + 1]
                    fnz = pk[kk, PK_N + 2]
                    sx_ = g00 * qx + g01 * qy + g02 * qz
                    sy_ = g10 * qx + g11 * qy + g12 * qz
                    sz_ = g20 * qx + g21 * qy + g22 * qz
                    nx_ = -(g00 * fnx + g01 * fny + g02 * fnz)
                    ny_ = -(g10 * fnx + g11 * fny + g12 * fnz)
                    nz_ = -(g20 * fnx + g21 * fny + g22 * fnz)
                    ux_ = g00 * fux + g01 * fuy + g02 * fuz - uw[0]
                    uy_ = g10 * fux + g11 * fuy + g12 * fuz - uw[1]
                    uz_ = g20 * fux + g21 * fuy + g22 * fuz - uw[2]
                    tick_f = float(tick0 + l0 // 2)
                    put_surface(ev.surface.rec, ev.surface.n, tick_f, float(j), float(pki[kk, PKI_ELEM]), sx_, sy_,
                                sz_, nx_, ny_, nz_, 0.0, pk[kk, PK_EDGE_R], pk[kk, PK_VB], ux_, uy_, uz_, slot, cacc)
            # ---------------------------------------------------- gating (engaged part of the swept arc)
            interior = False
            if cnt <= 0:
                frac = 0.0
                s_c = 0.5
                s_lo = 0.0
                s_hi = 0.0
            elif full:
                frac = 1.0
                s_c = 0.5
                s_lo = 0.0
                s_hi = 1.0
                interior = True
            else:
                general = True
                frac = 0.0
                s_c = 0.0
                s_lo = 0.0
                s_hi = 0.0
                if fast:
                    x = a - lo_s
                    if x < 0.0:
                        x += TWO_PI
                    if x - half > GATE_EPS and x + half < wid_s - GATE_EPS:
                        frac = 1.0
                        s_c = 0.5
                        s_lo = 0.0
                        s_hi = 1.0
                        general = False
                        interior = x > bin_w + GATE_EPS and x < wid_s - bin_w - GATE_EPS
                    elif x - half > wid_s + GATE_EPS and x + half < TWO_PI - GATE_EPS:
                        general = False
                if general:
                    frac, s_c, s_lo, s_hi = gate(iv, s, a, half, inv_dal)
            if frac <= 0.0:
                if (flags & FL_IN_CUT) != 0:
                    element_exit(tool, table, mem, st, ev, cacc, e, t_start, h_last[e], a - 0.5 * dal, tick0,
                                 commit, ratio, slot)
                elif commit and pend[e, PD_BIN] >= 0.0:
                    wsr = pki[kk, PKI_WRITE]
                    if wsr >= 0:
                        memory_commit(mem, wsr, int(pend[e, PD_BIN]), pend[e, PD_X], pend[e, PD_Y], pend[e, PD_Z],
                                      pend[e, PD_H], t_hwrite[pki[kk, PKI_ROW]], ratio, pend_tick[e])
                    pend[e, PD_BIN] = -1.0
                if commit:
                    t_last[e] = -1
                continue
            # ---------------------------------------------------- evaluation frame (centroid of the engaged arc)
            if frac < 1.0:
                dl = (s_c - 0.5) * dal
                cd = 1.0 - 0.5 * dl * dl
                sd = dl - dl * dl * dl / 6.0
                e00 = g00 * cd + g01 * sd
                e01 = g01 * cd - g00 * sd
                e10 = g10 * cd + g11 * sd
                e11 = g11 * cd - g10 * sd
                e20 = g20 * cd + g21 * sd
                e21 = g21 * cd - g20 * sd
                tsh = (s_c - 0.5) * dt
                qx = ox + tsh * wvx
                qy = oy + tsh * wvy
                qz = oz + tsh * wvz
                ccx = e00 * qx + e10 * qy + e20 * qz
                ccy = e01 * qx + e11 * qy + e21 * qz
                ccz = g02 * qx + g12 * qy + g22 * qz
                ucx = cd * uTx + sd * uTy
                ucy = cd * uTy - sd * uTx
                rcx = cd * rTx + sd * rTy
                rcy = cd * rTy - sd * rTx
                # velocities in the evaluation frame (rotated by -dl from the midpoint frame)
                bvx = cd * (vTx + vuTx) + sd * (vTy + vuTy)
                bvy = cd * (vTy + vuTy) - sd * (vTx + vuTx)
                wrx = cd * vrTx + sd * vrTy
                wry = cd * vrTy - sd * vrTx
                ofx = cd * OTx + sd * OTy
                ofy = cd * OTy - sd * OTx
                ac = a + dl
                if ac >= TWO_PI:
                    ac -= TWO_PI
                elif ac < 0.0:
                    ac += TWO_PI
            else:
                e00 = g00
                e01 = g01
                e10 = g10
                e11 = g11
                e20 = g20
                e21 = g21
                ccx = cTx
                ccy = cTy
                ccz = cTz
                ucx = uTx
                ucy = uTy
                rcx = rTx
                rcy = rTy
                bvx = vTx + vuTx
                bvy = vTy + vuTy
                wrx = vrTx
                wry = vrTy
                ofx = OTx
                ofy = OTy
                cd = 1.0
                sd = 0.0
                ac = a
            px = pk[kk, PK_P]
            py = pk[kk, PK_P + 1]
            pz = pk[kk, PK_P + 2]
            pzr = pz - zst
            dTx = ccx + px + ucx + (rcy * pzr - rTz * py)
            dTy = ccy + py + ucy + (rTz * px - rcx * pzr)
            dTz = ccz + pz + uTz + (rcx * py - rcy * px)
            nx = pk[kk, PK_N]
            ny = pk[kk, PK_N + 1]
            nz = pk[kk, PK_N + 2]
            # ---------------------------------------------------- memory read at the centroid angle
            xb = ac * inv_bin - 0.5
            fb = math.floor(xb)
            w = xb - fb
            u_b = w
            b0 = int(fb)
            if b0 < 0:
                b0 += nb
            bb = b0 + 1
            if bb >= nb:
                bb = 0
            if frac < 1.0 or (not interior and near_edge(iv, s, ac, bin_w)):
                # next to an interval boundary one bin of the stencil can lie outside the material, where no tooth
                # renews the memory: read the inside bin only
                in0 = in_intervals(iv, s, (b0 + 0.5) * bin_w)
                in1 = in_intervals(iv, s, (b0 + 1.5) * bin_w)
                if in0 and not in1:
                    w = 0.0
                elif in1 and not in0:
                    w = 1.0
            rs = pki[kk, PKI_READ]
            if rs < 0:
                # no predecessor on the thread helix: the field surface (refill plane of the station, uncached)
                psx = e00 * dTx + e01 * dTy + g02 * dTz
                psy = e10 * dTx + e11 * dTy + g12 * dTz
                psz = e20 * dTx + e21 * dTy + g22 * dTz
                nsx = e00 * nx + e01 * ny + g02 * nz
                nsy = e10 * nx + e11 * ny + g12 * nz
                nsz = e20 * nx + e21 * ny + g22 * nz
                hb0 = field_depth(refill, s, b0, psx, psy, psz, nsx, nsy, nsz, tick_end, commit, ev, cacc)
                hb1 = field_depth(refill, s, bb, psx, psy, psz, nsx, nsy, nsz, tick_end, commit, ev, cacc)
                hf = hb0 + w * (hb1 - hb0)
                m0x = psx - hf * nsx
                m0y = psy - hf * nsy
                m0z = psz - hf * nsz
                m1x = m0x
                m1y = m0y
                m1z = m0z
            else:
                ok0 = mvalid[rs, b0] != 0
                if ok0:
                    tk = mtick[rs, b0]
                    ok0 = tk >= stale_before and tick_end - tk <= stale_ticks
                ok1 = mvalid[rs, bb] != 0
                if ok1:
                    tk = mtick[rs, bb]
                    ok1 = tk >= stale_before and tick_end - tk <= stale_ticks
                if not (ok0 and ok1):
                    psx = e00 * dTx + e01 * dTy + g02 * dTz
                    psy = e10 * dTx + e11 * dTy + g12 * dTz
                    psz = e20 * dTx + e21 * dTy + g22 * dTz
                    nsx = e00 * nx + e01 * ny + g02 * nz
                    nsy = e10 * nx + e11 * ny + g12 * nz
                    nsz = e20 * nx + e21 * ny + g22 * nz
                    if not ok0:
                        ok0 = refill_bin(mem, refill, rs, b0, psx, psy, psz, nsx, nsy, nsz, tick_end, commit, ev,
                                         cacc)
                    if not ok1:
                        ok1 = refill_bin(mem, refill, rs, bb, psx, psy, psz, nsx, nsy, nsz, tick_end, commit, ev,
                                         cacc)
                    if ok0:
                        m0x = pts[rs, b0, 0]
                        m0y = pts[rs, b0, 1]
                        m0z = pts[rs, b0, 2]
                    else:
                        m0x = psx
                        m0y = psy
                        m0z = psz
                    if ok1:
                        m1x = pts[rs, bb, 0]
                        m1y = pts[rs, bb, 1]
                        m1z = pts[rs, bb, 2]
                    else:
                        m1x = psx
                        m1y = psy
                        m1z = psz
                else:
                    m0x = pts[rs, b0, 0]
                    m0y = pts[rs, b0, 1]
                    m0z = pts[rs, b0, 2]
                    m1x = pts[rs, bb, 0]
                    m1y = pts[rs, bb, 1]
                    m1z = pts[rs, bb, 2]
            Mx_ = m0x + w * (m1x - m0x)
            My_ = m0y + w * (m1y - m0y)
            Mz_ = m0z + w * (m1z - m0z)
            # memory point in T, chip thickness along n (vector form, architecture 5.4)
            MTx_ = e00 * Mx_ + e10 * My_ + e20 * Mz_
            MTy_ = e01 * Mx_ + e11 * My_ + e21 * Mz_
            MTz_ = g02 * Mx_ + g12 * My_ + g22 * Mz_
            h = (dTx - MTx_) * nx + (dTy - MTy_) * ny + (dTz - MTz_) * nz
            if rs >= 0:
                # the bins hold points on the swept circle: the linear stencil reads a chord, inside the circle
                # by r dphi^2 [(1 - w) u^2 + w (1 - u)^2] / 2 at the element's angle (u: its position between the
                # bin centres, w: the stencil weight); along n that is (n . p_xy) times it
                h -= sag_c * ((1.0 - w) * u_b * u_b + w * (1.0 - u_b) * (1.0 - u_b)) * pk[kk, PK_NPXY]
            hl = h_last[e]
            slope = h - hl if t_last[e] == tick0 else 0.0
            # element velocity relative to the material (T): tip + frame rotation + deflection rates
            vex = bvx + (ofy * pz - OTz * py) + (wry * pzr - vrTz * py)
            vey = bvy + (OTz * px - ofx * pz) + (vrTz * px - wrx * pzr)
            vez = vTz + vuTz + (ofx * py - ofy * px) + (wrx * py - wry * px)
            vn = vex * nx + vey * ny + vez * nz
            Vc = vex * pk[kk, PK_C] + vey * pk[kk, PK_C + 1] + vez * pk[kk, PK_C + 2]
            row = pki[kk, PKI_ROW]
            # ---------------------------------------------------- memory crossings (delayed by one bin)
            if commit and bps != 0.0:
                hmin_m = t_hwrite[row]
                xs = xb - (s_c - 0.5) * bps - 0.5 * bps
                xe = xs + bps
                # a bin centre at a step's end belongs to both steps: xs and xe of consecutive steps are computed
                # separately and may leave it outside both by round-off (a commensurate stride, e.g. 0.25 deg on
                # 0.5 deg bins, skipped the write of every other bin: the next tooth read a surface three passes old,
                # h = 3 f_z). The range is widened by CROSS_EPS bins and a bin already pending is not crossed again
                if bps > 0.0:
                    cA = int(math.floor(xs - CROSS_EPS)) + 1
                    cB = int(math.floor(xe + CROSS_EPS))
                    step_c = 1
                else:
                    cA = int(math.ceil(xs + CROSS_EPS)) - 1
                    cB = int(math.ceil(xe - CROSS_EPS))
                    step_c = -1
                ws = pki[kk, PKI_WRITE]
                if (step_c > 0 and cB >= cA) or (step_c < 0 and cB <= cA):
                    c = cA
                    while True:
                        sx = (c - xs) * inv_bps
                        if s_lo - CROSS_EPS_S < sx < s_lo:
                            sx = s_lo
                        elif s_hi < sx < s_hi + CROSS_EPS_S:
                            sx = s_hi
                        bw = c
                        while bw >= nb:
                            bw -= nb
                        while bw < 0:
                            bw += nb
                        if s_lo <= sx <= s_hi and bw != int(pend[e, PD_BIN]):
                            hx = h + (sx - s_c) * slope
                            pb = int(pend[e, PD_BIN])
                            if pb >= 0:
                                if ws >= 0:
                                    # the pending write of the previous bin (memory_commit, elastic recovery)
                                    hp = pend[e, PD_H]
                                    if hp >= hmin_m:
                                        if hmin_m <= 0.0 or ratio <= 0.0 or hp >= (1.0 + ratio) * hmin_m or \
                                                mvalid[ws, pb] == 0:
                                            wc = 1.0
                                        else:
                                            wc = (hp - hmin_m) / (ratio * hmin_m)
                                        if wc > 0.0:
                                            pts[ws, pb, 0] += wc * (pend[e, PD_X] - pts[ws, pb, 0])
                                            pts[ws, pb, 1] += wc * (pend[e, PD_Y] - pts[ws, pb, 1])
                                            pts[ws, pb, 2] += wc * (pend[e, PD_Z] - pts[ws, pb, 2])
                                            mtick[ws, pb] = pend_tick[e]
                                            mvalid[ws, pb] = 1
                                pend[e, PD_BIN] = -1.0
                            bi = c
                            while bi >= nb:
                                bi -= nb
                            while bi < 0:
                                bi += nb
                            off = (sx - s_c) * dt
                            # the element at the crossing: its velocity carries the motion to first order; the
                            # rotation's second-order term keeps the point on the swept circle (what the read's
                            # chord correction assumes)
                            dl2 = 0.5 * ((sx - s_c) * dal) ** 2
                            qx = dTx + off * vex - dl2 * px
                            qy = dTy + off * vey - dl2 * py
                            qz = dTz + off * vez
                            ex = e00 * qx + e01 * qy + g02 * qz
                            ey = e10 * qx + e11 * qy + g12 * qz
                            ez = e20 * qx + e21 * qy + g22 * qz
                            if ws < 0:
                                pass
                            elif rs != ws:
                                have = False
                                rx_ = 0.0
                                ry_ = 0.0
                                rz_ = 0.0
                                rtick = tick_end
                                if rs >= 0:
                                    if mvalid[rs, bi] != 0:
                                        have = True
                                        rx_ = pts[rs, bi, 0]
                                        ry_ = pts[rs, bi, 1]
                                        rz_ = pts[rs, bi, 2]
                                        rtick = mtick[rs, bi]
                                elif refill.valid[s, bi] != 0:
                                    nsx = e00 * nx + e01 * ny + g02 * nz
                                    nsy = e10 * nx + e11 * ny + g12 * nz
                                    nsz = e20 * nx + e21 * ny + g22 * nz
                                    hf = field_depth(refill, s, bi, ex, ey, ez, nsx, nsy, nsz, tick_end, False, ev,
                                                     cacc)
                                    have = True
                                    rx_ = ex - hf * nsx
                                    ry_ = ey - hf * nsy
                                    rz_ = ez - hf * nsz
                                if have:
                                    # thread helix: the element writes another row than it reads. The surface it
                                    # passes on is the one it read, renewed by the elastic-recovery rule
                                    if hmin_m <= 0.0 or ratio <= 0.0:
                                        wr = 1.0 if hx >= hmin_m else 0.0
                                    elif hx < hmin_m:
                                        wr = 0.0
                                    elif hx >= (1.0 + ratio) * hmin_m:
                                        wr = 1.0
                                    else:
                                        wr = (hx - hmin_m) / (ratio * hmin_m)
                                    pend[e, PD_BIN] = float(bi)
                                    pend[e, PD_X] = rx_ + wr * (ex - rx_)
                                    pend[e, PD_Y] = ry_ + wr * (ey - ry_)
                                    pend[e, PD_Z] = rz_ + wr * (ez - rz_)
                                    pend[e, PD_H] = 1e300
                                    pend_tick[e] = tick0 + int(math.floor(sx * l0 + 0.5)) if wr > 0.0 else rtick
                                elif hx >= hmin_m:
                                    pend[e, PD_BIN] = float(bi)
                                    pend[e, PD_X] = ex
                                    pend[e, PD_Y] = ey
                                    pend[e, PD_Z] = ez
                                    pend[e, PD_H] = hx
                                    pend_tick[e] = tick0 + int(math.floor(sx * l0 + 0.5))
                            elif hx >= hmin_m:
                                pend[e, PD_BIN] = float(bi)
                                pend[e, PD_X] = ex
                                pend[e, PD_Y] = ey
                                pend[e, PD_Z] = ez
                                pend[e, PD_H] = hx
                                pend_tick[e] = tick0 + int(math.floor(sx * l0 + 0.5))
                        if c == cB:
                            break
                        c += step_c
            if h <= 0.0:
                if (flags & FL_IN_CUT) != 0:
                    q = hl / (hl - h) if hl > h else 0.0
                    element_exit(tool, table, mem, st, ev, cacc, e, t_start + (q - 0.5) * dt, 0.0,
                                 a + (q - 1.0) * dal, tick0 + int(math.floor((q - 0.5) * l0 + 0.5)), commit, ratio,
                                 slot)
                if commit:
                    h_last[e] = h
                    t_last[e] = tick_end if frac >= 1.0 else -1
                continue
            # ---------------------------------------------------- forces
            if not h < h_reach:
                # a chip beyond the tool's reach (or not finite): the state has diverged. Counted for the domain's
                # finding; the table is read at its last segment (no wrapped index)
                cacc.scalars[CA_H_BEYOND] += 1.0
                if not h <= cacc.scalars[CA_H_BEYOND_MAX]:
                    cacc.scalars[CA_H_BEYOND_MAX] = h if h == h else 1e300
            hm = t_hmin[row]
            if h < hm:
                sg = int(h * t_dlo[row])
                if sg >= nlo:
                    sg = nlo - 1
            else:
                xg = (h - hm) * t_dhi[row]
                if xg < nseg - nlo:
                    sg = nlo + int(xg)
                else:
                    sg = nseg - 1
                    cacc.scalars[CA_TABLE_OVER] += 1.0
            # h is the projection on n of the surface's shift between passes at the element's own station (the memory),
            # so h db tiles the chip exactly (strips along the shift, Cavalieri) and the edge's curvature does not
            # narrow it (turn_step measures h along n instead: abi.edge_curvature)
            db = pk[kk, PK_DB]
            dS = pk[kk, PK_DS]
            # the element at the material's boundary along the edge (the depth-of-cut line, a step's or a floor's
            # edge): where a neighbouring station is out of the material at this angle, the element's half toward it
            # is replaced by the engaged length up to the boundary (st_edge, measured at the interval refresh:
            # core.hot_kernels.edge_reach). Before 2026-11-10 the element was engaged whole or not at all: the forces
            # erred by (engaged length - ap) / ap, up to half an element's height over ap (-4.8 % at ap 3 mm, -32 % at
            # 0.6 mm with 0.5 mm elements; docs/convergence.md 3)
            wgt = 1.0
            if st_edge[s, SE_ON] > 0.0 and st_edge[s, SE_TICK] == iv_tick:
                if s > 0 and st_edge[s, SE_D_LO] != 0.0 and not in_intervals(iv, s - 1, ac):
                    wgt += st_edge[s, SE_D_LO]
                if s + 1 < nst and st_edge[s, SE_D_HI] != 0.0 and not in_intervals(iv, s + 1, ac):
                    wgt += st_edge[s, SE_D_HI]
                if wgt < 0.0:
                    wgt = 0.0
                db *= wgt
                dS *= wgt
            ft0 = (coef[row, sg, 0, COL_FT] + coef[row, sg, 1, COL_FT] * h) * db + \
                (coef[row, sg, 0, COL_ET] + coef[row, sg, 1, COL_ET] * h) * dS
            fr0 = (coef[row, sg, 0, COL_FR] + coef[row, sg, 1, COL_FR] * h) * db + \
                (coef[row, sg, 0, COL_ER] + coef[row, sg, 1, COL_ER] * h) * dS
            fa0 = (coef[row, sg, 0, COL_FA] + coef[row, sg, 1, COL_FA] * h) * db + \
                (coef[row, sg, 0, COL_EA] + coef[row, sg, 1, COL_EA] * h) * dS
            fft = pk[kk, PK_FLANK] * wgt
            ffr = pk[kk, PK_FLANK + 1] * wgt
            fpd = 0.0
            kpd = pk[kk, PK_PD] * wgt
            dtw = frac * dt
            was_in = (flags & FL_IN_CUT) != 0
            pux = 0.0
            puy = 0.0
            puz = 0.0
            if kpd > 0.0:
                # process damping: the flank indents the surface the edge itself generated behind it, held in the
                # element's wave memory as its relative vibration displacement (station translation and rotation
                # minus the work's deflection, T0); the steady path (feed) is the surface the edge follows
                if frac < 1.0:
                    ce = cm * cd - sm * sd
                    se = sm * cd + cm * sd
                else:
                    ce = cm
                    se = sm
                ax0 = ce * px - se * py
                ay0 = se * px + ce * py
                pux = ust[s, 0] + ust[s, 4] * pzr - ust[s, 5] * ay0 - uw0x
                puy = ust[s, 1] + ust[s, 5] * ax0 - ust[s, 3] * pzr - uw0y
                puz = ust[s, 2] + u_f + ust[s, 3] * ay0 - ust[s, 4] * ax0 - uw0z
                if Vc > 0.0 and was_in:
                    # the surface behind the edge was generated by this tooth only back to the previous one
                    xlim = TWO_PI * pk[kk, PK_RXY] * inv_nt
                    if xlim > pd_max:
                        xlim = pd_max
                    fpd = kpd * wave_area(pd_ring, pd_aux, e, pux, puy, puz, ce * nx - se * ny, se * nx + ce * ny,
                                          nz, pd_aux[e, PDA_X] + Vc * dtw, pk[kk, PK_PD + 1], pk[kk, PK_PD + 2], xlim,
                                          pd_dx)
            mu_pd = pk[kk, PK_PD + 3]
            ft = ft0 + fft + mu_pd * fpd
            fr = fr0 + ffr + fpd
            fa = fa0
            Fx = ft * pk[kk, PK_DIR] + fr * pk[kk, PK_DIR + 3] + fa * pk[kk, PK_DIR + 6]
            Fy = ft * pk[kk, PK_DIR + 1] + fr * pk[kk, PK_DIR + 4] + fa * pk[kk, PK_DIR + 7]
            Fz = ft * pk[kk, PK_DIR + 2] + fr * pk[kk, PK_DIR + 5] + fa * pk[kk, PK_DIR + 8]
            Mx = ft * pk[kk, PK_ARM] + fr * pk[kk, PK_ARM + 3] + fa * pk[kk, PK_ARM + 6]
            My = ft * pk[kk, PK_ARM + 1] + fr * pk[kk, PK_ARM + 4] + fa * pk[kk, PK_ARM + 7]
            Mz = ft * pk[kk, PK_ARM + 2] + fr * pk[kk, PK_ARM + 5] + fa * pk[kk, PK_ARM + 8]
            # power dissipated by the element (-F . v), split by zone below
            va = vex * pk[kk, PK_DIR + 6] + vey * pk[kk, PK_DIR + 7] + vez * pk[kk, PK_DIR + 8]
            p_tab = ft0 * Vc + fr0 * vn - fa0 * va
            p_fl = fft * Vc + ffr * vn + mu_pd * fpd * Vc
            p_pd = fpd * vn
            # the instantaneous force at the step's ends (the peak; a step mean is 5-11 % below the peak at the default
            # stride, docs/convergence.md 1): the element's force at its chip of that instant (h linear over the step,
            # the table read at that chip: its own segment), where it is engaged then, in T (the same frame for every
            # element). Before 2026-11-10 the end forces took the slope of the centroid chip's segment: a chip that
            # fell from 49 um to 0.09 um within a step (below h_min, the steep low segment) gave 1.7 kN at the start
            # (stride_slot F_peak 850 N at 1 deg against 258 N converged)
            dft = coef[row, sg, 1, COL_FT] * db + coef[row, sg, 1, COL_ET] * dS
            dfr = coef[row, sg, 1, COL_FR] * db + coef[row, sg, 1, COL_ER] * dS
            dfa = coef[row, sg, 1, COL_FA] * db + coef[row, sg, 1, COL_EA] * dS
            for end_ in range(2):
                if end_ == 0:
                    if s_lo > 1e-9:
                        continue
                    hq = h - s_c * slope
                else:
                    if s_hi < 1.0 - 1e-9:
                        continue
                    hq = h + (1.0 - s_c) * slope
                if not hq > 0.0:
                    continue
                if hq < t_hmin[row]:
                    sq = int(hq * t_dlo[row])
                    if sq >= nlo:
                        sq = nlo - 1
                else:
                    xq = (hq - t_hmin[row]) * t_dhi[row]
                    sq = nlo + int(xq) if xq < nseg - nlo else nseg - 1
                gt = (coef[row, sq, 0, COL_FT] + coef[row, sq, 1, COL_FT] * hq) * db + \
                    (coef[row, sq, 0, COL_ET] + coef[row, sq, 1, COL_ET] * hq) * dS + (ft - ft0)
                gr = (coef[row, sq, 0, COL_FR] + coef[row, sq, 1, COL_FR] * hq) * db + \
                    (coef[row, sq, 0, COL_ER] + coef[row, sq, 1, COL_ER] * hq) * dS + (fr - fr0)
                ga = (coef[row, sq, 0, COL_FA] + coef[row, sq, 1, COL_FA] * hq) * db + \
                    (coef[row, sq, 0, COL_EA] + coef[row, sq, 1, COL_EA] * hq) * dS
                gx = gt * pk[kk, PK_DIR] + gr * pk[kk, PK_DIR + 3] + ga * pk[kk, PK_DIR + 6]
                gy = gt * pk[kk, PK_DIR + 1] + gr * pk[kk, PK_DIR + 4] + ga * pk[kk, PK_DIR + 7]
                gz = gt * pk[kk, PK_DIR + 2] + gr * pk[kk, PK_DIR + 5] + ga * pk[kk, PK_DIR + 8]
                # moment about the tip: about the station's axis point plus z_station x F
                g3 = gt * pk[kk, PK_ARM] + gr * pk[kk, PK_ARM + 3] + ga * pk[kk, PK_ARM + 6] - zst * gy
                g4 = gt * pk[kk, PK_ARM + 1] + gr * pk[kk, PK_ARM + 4] + ga * pk[kk, PK_ARM + 7] + zst * gx
                g5 = gt * pk[kk, PK_ARM + 2] + gr * pk[kk, PK_ARM + 5] + ga * pk[kk, PK_ARM + 8]
                if end_ == 0:
                    pS0 += gx
                    pS1 += gy
                    pS2 += gz
                    pS3 += g3
                    pS4 += g4
                    pS5 += g5
                else:
                    pE0 += gx
                    pE1 += gy
                    pE2 += gz
                    pE3 += g3
                    pE4 += g4
                    pE5 += g5
            if frac < 1.0:
                # forces at the engaged centroid act in the frame rotated by dl from the midpoint frame
                tmp = cd * Fx - sd * Fy
                Fy = sd * Fx + cd * Fy
                Fx = tmp
                tmp = cd * Mx - sd * My
                My = sd * Mx + cd * My
                Mx = tmp
            sFx += Fx * dtw
            sFy += Fy * dtw
            sFz += Fz * dtw
            sMx += Mx * dtw
            sMy += My * dtw
            sMz += Mz * dtw
            any_in = True
            if float_k > 0.0:
                Kz += frac * nz * (dft * pk[kk, PK_DIR + 2] + dfr * pk[kk, PK_DIR + 5] + dfa * pk[kk, PK_DIR + 8])
            n_in += 1.0
            if h > h_max:
                h_max = h
            if not commit:
                continue
            # ---------------------------------------------------- energies, volumes, element integrals
            ps = coef[row, sg, 0, COL_PS] + coef[row, sg, 1, COL_PS] * h
            pr = coef[row, sg, 0, COL_PR] + coef[row, sg, 1, COL_PR] * h
            rc = coef[row, sg, 0, COL_RC] + coef[row, sg, 1, COL_RC] * h
            et = p_tab * dtw
            forming = pki[kk, PKI_FORMING] != 0
            r_xy = pk[kk, PK_RXY]
            dV = h * db * r_xy * abs_dal * frac
            if not forming:
                # the chip is the material the pass removes: the share the surface memory takes (the elastic-recovery
                # rule: none below h_write, ploughed and recovering; the full chip above (1 + ratio) h_write).
                # Counting every h > 0 counted a ploughed layer again at the pass that finally removed it (+3.7 %
                # on a side step of ae/D 0.05 with h_min 3 um)
                hw = t_hwrite[row]
                if h < hw:
                    dV = 0.0
                elif hw > 0.0 and ratio > 0.0 and h < (1.0 + ratio) * hw:
                    dV *= (h - hw) / (ratio * hw)
            if forming:
                e_plough += et
                e_forming += et
                vol_disp += dV
            else:
                e_shear += et * ps
                e_rake += et * pr
                e_plough += et * (1.0 - ps - pr)
                vol_chip += dV
            e_flank += p_fl * dtw
            e_pd += p_pd * dtw
            edge_len += dS * frac * dt
            v_int += Vc * dtw
            ea[e, ELEM_IN_CUT_S] += dtw
            # wear's integrals: the sliding distance the faces actually travel (|V|: an edge driven backwards by a
            # vibration still slides on the work)
            sl = abs(Vc) * dtw
            ea[e, ELEM_RAKE_SLIDING_M] += rc * sl
            ea[e, ELEM_FLANK_SLIDING_M] += sl
            ea[e, ELEM_INT_SIGMA_V_RAKE] += (coef[row, sg, 0, COL_SIG_RAKE] + coef[row, sg, 1, COL_SIG_RAKE] * h) \
                * rc * sl
            ea[e, ELEM_INT_SIGMA_V_FLANK] += pk[kk, PK_SIG_FLANK] * sl
            ea[e, ELEM_INT_H_DT] += h * dtw
            if h > ea[e, ELEM_H_MAX]:
                ea[e, ELEM_H_MAX] = h
            # ---------------------------------------------------- entries
            if (flags & FL_IN_CUT) == 0:
                ea[e, ELEM_ENTRIES] += 1.0
                ts[j, TS_NEW] += 1.0
                if s_lo > 0.0:
                    s_en = s_lo
                elif slope > 0.0 and h < slope:
                    s_en = s_c - h / slope
                    if s_en < 0.0:
                        s_en = 0.0
                else:
                    s_en = 0.0
                t_en = t_start + s_en * dt
                ang = a + (s_en - 0.5) * dal
                if ang < 0.0:
                    ang += TWO_PI
                elif ang >= TWO_PI:
                    ang -= TWO_PI
                h_en = h + (s_en - s_c) * slope
                if h_en < 0.0:
                    h_en = 0.0
                if ts[j, TS_PASS_ON] < 0.5:
                    ts[j, TS_PASS_ON] = 1.0
                    ts[j, TS_ENTRY_TICK] = float(tick0 + int(math.floor(s_en * l0 + 0.5)))
                    ts[j, TS_ENTRY_ELEM] = float(pki[kk, PKI_ELEM])
                    ts[j, TS_ENTRY_H] = h_en
                    ts[j, TS_ENTRY_ANGLE] = ang
                    ts[j, TS_LOADING] = 1.0
                    ts[j, TS_FIRST_ENTRY_S] = t_en
                    ts[j, TS_LAST_ENTRY_S] = t_en
                    ts[j, TS_FMAX] = 0.0
                    ts[j, TS_BREAKOUT] = 0.0
                    ts[j, TS_EXIT_FIRST_S] = -1.0
                    ts[j, TS_LAST_EXIT_S] = -1.0
                    ts[j, TS_EMIT_TICK] = float(tick_end)
                    cacc.scalars[CA_ENTRIES] += 1.0
                elif ts[j, TS_LOADING] > 0.5 and t_en > ts[j, TS_LAST_ENTRY_S]:
                    ts[j, TS_LAST_ENTRY_S] = t_en
                flags = FL_IN_CUT
                if s_lo > 0.0 and abs(vn) <= vt_tol:
                    # entry through the outer boundary where the path is tangent: the entry point is on the
                    # generated surface (up-milling side wall)
                    tick_x = float(tick0 + int(math.floor(s_en * l0 + 0.5)))
                    toff = (s_en - s_c) * dt
                    _surface_event(ev, cacc, tool, defl, e, j, tick_x, dTx, dTy, dTz, toff, vex, vey, vez, h_en, nx,
                                   ny, nz, ucx, ucy, uTz, rcx, rcy, rTz, pzr, px, py, e00, e01, g02, e10, e11, g12,
                                   e20, e21, g22, slot)
            if kpd > 0.0:
                wave_push(pd_ring, pd_aux, e, pux, puy, puz, Vc * dtw if Vc > 0.0 else 0.0, pd_dx, not was_in)
            ts[j, TS_NOW] += 1.0
            ts[j, TS_FX] += Fx * frac
            ts[j, TS_FY] += Fy * frac
            ts[j, TS_FZ] += Fz * frac
            ts[j, TS_STEP_B] += db
            if centre:
                ts[j, TS_CENTRE_ON] = 1.0
            # ---------------------------------------------------- tooth-pass accumulation (chip emission, deposits)
            if not forming:
                tch = coef[row, sg, 0, COL_TCHIP] + coef[row, sg, 1, COL_TCHIP] * h
                ts[j, TS_VOL] += dV
                ts[j, TS_HV] += h * dV
                ts[j, TS_VV] += Vc * dV
                ts[j, TS_RAKEV] += pk[kk, PK_RAKE] * dV
                ts[j, TS_INCLV] += pk[kk, PK_INCL] * dV
                ts[j, TS_TV] += tch * dV
                ts[j, TS_RV] += r_xy * dV
                if dV > ts[j, TS_DOM_VOL]:
                    ts[j, TS_DOM_VOL] = dV
                    ts[j, TS_DOM_KIND] = float(pki[kk, PKI_KIND])
                    ts[j, TS_DOM_ROW] = float(row)
                ts[j, TS_STEP_DV] += dV
                ts[j, TS_S_PX] += dTx * dV
                ts[j, TS_S_PY] += dTy * dV
                ts[j, TS_S_PZ] += dTz * dV
                vch = rc * Vc
                ts[j, TS_S_VX] += (vex + vch * pk[kk, PK_CHIP]) * dV
                ts[j, TS_S_VY] += (vey + vch * pk[kk, PK_CHIP + 1]) * dV
                ts[j, TS_S_VZ] += (vez + vch * pk[kk, PK_CHIP + 2]) * dV
                ts[j, TS_S_NX] += nx * dV
                ts[j, TS_S_NY] += ny * dV
                ts[j, TS_S_NZ] += nz * dV
            wsh = coef[row, sg, 0, COL_WS] + coef[row, sg, 1, COL_WS] * h
            wfl = coef[row, sg, 0, COL_WF] + coef[row, sg, 1, COL_WF] * h
            ts[j, TS_WORK_J] += et * ps * wsh + (et * (1.0 - ps - pr) + p_fl * dtw) * wfl
            # ---------------------------------------------------- surface events (envelope: v.n from + to -)
            if (flags & FL_VN_POS) != 0 and vn <= 0.0 and (flags & FL_EVENT_DONE) == 0:
                vl = vn_last[e]
                q = vl / (vl - vn) if vl > vn else 1.0
                hq = hl + q * (h - hl)
                tick_x = float(tick0 + int(math.floor((q - 0.5) * l0 + 0.5)))
                toff = (q - 1.0) * dt
                _surface_event(ev, cacc, tool, defl, e, j, tick_x, dTx, dTy, dTz, toff, vex, vey, vez, hq, nx, ny, nz,
                               ucx, ucy, uTz, rcx, rcy, rTz, pzr, px, py, e00, e01, g02, e10, e11, g12, e20, e21, g22,
                               slot)
                flags |= FL_EVENT_DONE
            if vn > 0.0:
                flags |= FL_VN_POS
            else:
                flags &= ~FL_VN_POS
            vn_last[e] = vn
            h_last[e] = h
            t_last[e] = tick_end if frac >= 1.0 else -1
            # an arc clipped at its end: the element leaves the interval within this step
            if s_hi < 1.0 and frac < 1.0 and s_lo <= 0.0:
                if abs(vn) <= vt_tol and (flags & FL_EVENT_DONE) == 0:
                    # exit through the outer boundary where the path is tangent (down-milling side wall)
                    hx = h + (s_hi - s_c) * slope
                    if hx < 0.0:
                        hx = 0.0
                    tick_x = float(tick0 + int(math.floor(s_hi * l0 + 0.5)))
                    toff = (s_hi - s_c) * dt
                    _surface_event(ev, cacc, tool, defl, e, j, tick_x, dTx, dTy, dTz, toff, vex, vey, vez, hx, nx, ny,
                                   nz, ucx, ucy, uTz, rcx, rcy, rTz, pzr, px, py, e00, e01, g02, e10, e11, g12, e20,
                                   e21, g22, slot)
                fl_a[e] = flags
                element_exit(tool, table, mem, st, ev, cacc, e, t_start + s_hi * dt, h, a + (s_hi - 0.5) * dal,
                             tick0 + int(math.floor(s_hi * l0 + 0.5)), commit, ratio, slot)
                t_last[e] = -1
            else:
                fl_a[e] = flags
        # ---------------------------------------------------- station totals
        if any_in:
            inv = 1.0 / dt
            sto[s, 0] = (cm * sFx - sm * sFy) * inv
            sto[s, 1] = (sm * sFx + cm * sFy) * inv
            sto[s, 2] = sFz * inv
            sto[s, 3] = (cm * sMx - sm * sMy) * inv
            sto[s, 4] = (sm * sMx + cm * sMy) * inv
            sto[s, 5] = sMz * inv
            written[s] = 2
            FTx += sFx
            FTy += sFy
            FTz += sFz
            MTx += sMx - zst * sFy
            MTy += sMy + zst * sFx
            MTz += sMz
            e_def += sFx * vuTx + sFy * vuTy + sFz * vuTz + sMx * vrTx + sMy * vrTy + sMz * vrTz
            sb = tool.st_sec[s]
            if 0 <= sb < sec.shape[0]:
                sec[sb, 0] += sFx
                sec[sb, 1] += sFy
                sec[sb, 2] += sFz
                sec[sb, 3] += zst * sFx
                sec[sb, 4] += zst * sFy
                sec[sb, 5] += zst * sFz
                sec[sb, 6] += sMx
                sec[sb, 7] += sMy
                sec[sb, 8] += sMz
    for s in range(s_first, s_last + 1):
        if written[s] == 2:
            written[s] = 1
        elif written[s] == 1:
            written[s] = 0
    # the work's vibration velocity is part of vT above: its share of the force power is deflection work
    e_def -= FTx * vWx + FTy * vWy + FTz * vWz
    # ---------------------------------------------------- tooth ends, shocks, emission
    close_teeth(tool, st, ev, cacc, pose, step, istep, commit, g00, g01, g02, g10, g11, g12, g20, g21, g22, dal)
    # ---------------------------------------------------- totals in T0 (rotate the T sums by theta_mid)
    F0x = cm * FTx - sm * FTy
    F0y = sm * FTx + cm * FTy
    F0z = FTz
    M0x = cm * MTx - sm * MTy
    M0y = sm * MTx + cm * MTy
    M0z = MTz
    # the element impulses alone (the chip loads below are constant over the step: their mean joins both ends)
    E0x = F0x
    E0y = F0y
    E0z = F0z
    E0mx = M0x
    E0my = M0y
    E0mz = M0z
    # rigid-body work of the cutting forces: shaft (spindle rotation) and feed (axes, incl. rotary)
    shaft = -omega * M0z
    feed = -(vTx * FTx + vTy * FTy + vTz * FTz + (OTx * MTx + OTy * MTy + (OTz - omega) * MTz))
    feed -= FTx * vWx + FTy * vWy + FTz * vWz
    # ---------------------------------------------------- loads of the chips on the tool: recut strikes, packing
    # (docs/models/cutting.md "Recutting"). Each acts at a station dynamics takes loads at (the engaged range, as the
    # hot loop's DynSystem.act; none when no station is engaged: the work modes still take it through the total
    # force); its work on the deflection rates goes to e_def, the rest of its work is dissipated in the chips
    # (recut_J, packing_J), so the energy identity holds and agrees with the modes' work
    if commit and occ.cell_m[0] > 0.0 and n_teeth_f > 0.0:
        st.misc[MS_RECUT_ANGLE] += abs_dal
        if st.misc[MS_RECUT_ANGLE] >= TWO_PI * inv_nt:
            st.misc[MS_RECUT_ANGLE] = 0.0
            recut_scan(tool, table, occ, pose, step, istep, cacc, acc, ev, st, theta, iv)
    tp = pack.torque_Nm[0]
    pkx = pack.force_T0_N[0]
    pky = pack.force_T0_N[1]
    pkz = pack.force_T0_N[2]
    packing_on = tp != 0.0 or pkx != 0.0 or pky != 0.0 or pkz != 0.0
    ra = st.recut_act
    n_ra = ra.shape[0]
    t_end_step = t_start + dt
    any_load = packing_on
    for q in range(n_ra):
        if ra[q, RA_T] > t_start and ra[q, RA_T] > ra[q, RA_T0]:
            any_load = True
            break
    if any_load:
        lo_e = -1
        hi_e = -1
        for s in range(nst):
            if iv_cnt[s] > 0:
                if lo_e < 0:
                    lo_e = s
                hi_e = s
        inv_dt = 1.0 / dt
        sgn = 1.0 if omega > 0.0 else (-1.0 if omega < 0.0 else 0.0)
        vtx = pose.v_tip_S_m_s[0]
        vty = pose.v_tip_S_m_s[1]
        vtz = pose.v_tip_S_m_s[2]
        for q in range(n_ra + 1):
            if q < n_ra:
                # the part of the load's time inside this step (a load whose time passed while the tool was out of
                # contact has ended: it is not carried to the next engagement)
                a0 = ra[q, RA_T0] if ra[q, RA_T0] > t_start else t_start
                a1 = ra[q, RA_T] if ra[q, RA_T] < t_end_step else t_end_step
                if a1 <= a0:
                    continue
                te = a1 - a0
                jx = ra[q, RA_FX] * te
                jy = ra[q, RA_FY] * te
                jz = ra[q, RA_FZ] * te
                mx = ra[q, RA_MX] * te
                my = ra[q, RA_MY] * te
                mz = ra[q, RA_MZ] * te
                z_l = ra[q, RA_Z]
            else:
                if not packing_on:
                    continue
                # the flute packing at the middle of the engaged edge, its torque opposing the rotation
                z_l = 0.5 * (st_zs[lo_e] + st_zs[hi_e]) if lo_e >= 0 else 0.0
                jx = pkx * dt
                jy = pky * dt
                jz = pkz * dt
                mx = -z_l * jy
                my = z_l * jx
                mz = -sgn * tp * dt
            sl = -1
            if lo_e >= 0:
                sl = lo_e
                best = abs(st_zs[lo_e] - z_l)
                for s in range(lo_e + 1, hi_e + 1):
                    d = abs(st_zs[s] - z_l)
                    if d < best:
                        best = d
                        sl = s
            e_l = 0.0
            if sl >= 0:
                zs = st_zs[sl]
                msx = mx + zs * jy
                msy = my - zs * jx
                msz = mz
                sto[sl, 0] += jx * inv_dt
                sto[sl, 1] += jy * inv_dt
                sto[sl, 2] += jz * inv_dt
                sto[sl, 3] += msx * inv_dt
                sto[sl, 4] += msy * inv_dt
                sto[sl, 5] += msz * inv_dt
                if written[sl] == 0:
                    written[sl] = 1
                sb = tool.st_sec[sl]
                if 0 <= sb < sec.shape[0]:
                    sec[sb, 0] += jx
                    sec[sb, 1] += jy
                    sec[sb, 2] += jz
                    sec[sb, 3] += zs * jx
                    sec[sb, 4] += zs * jy
                    sec[sb, 5] += zs * jz
                    sec[sb, 6] += msx
                    sec[sb, 7] += msy
                    sec[sb, 8] += msz
                e_l = jx * vst[sl, 0] + jy * vst[sl, 1] + jz * vst[sl, 2] + msx * vst[sl, 3] + msy * vst[sl, 4] + \
                    msz * vst[sl, 5]
            e_l -= jx * vw0x + jy * vw0y + jz * vw0z
            jsx = R[0, 0] * jx + R[0, 1] * jy + R[0, 2] * jz
            jsy = R[1, 0] * jx + R[1, 1] * jy + R[1, 2] * jz
            jsz = R[2, 0] * jx + R[2, 1] * jy + R[2, 2] * jz
            msX = R[0, 0] * mx + R[0, 1] * my + R[0, 2] * mz
            msY = R[1, 0] * mx + R[1, 1] * my + R[1, 2] * mz
            msZ = R[2, 0] * mx + R[2, 1] * my + R[2, 2] * mz
            w_sh = -omega * mz
            w_fd = -(jsx * vtx + jsy * vty + jsz * vtz) - (fwx * msX + fwy * msY + fwz * msZ)
            F0x += jx
            F0y += jy
            F0z += jz
            M0x += mx
            M0y += my
            M0z += mz
            shaft += w_sh
            feed += w_fd
            e_def += e_l
            if commit:
                if q < n_ra:
                    cacc.scalars[CA_RECUT_J] += w_sh + w_fd - e_l
                else:
                    cacc.scalars[CA_PACKING_J] += w_sh + w_fd - e_l
    power = e_shear + e_rake + e_plough + e_flank + e_pd
    if commit:
        a_ = acc.scalars
        a_[ACC_HEAT_SHEAR_J] += e_shear
        a_[ACC_HEAT_RAKE_J] += e_rake
        a_[ACC_HEAT_FLANK_J] += e_flank
        a_[ACC_PLOUGHING_J] += e_plough
        a_[ACC_PROCESS_DAMPING_J] += e_pd
        a_[ACC_CHIP_VOLUME_M3] += vol_chip
        c_ = cacc.scalars
        c_[CA_DISPLACED_M3] += vol_disp
        c_[CA_FORMING_J] += e_forming
        c_[CA_DEFLECTION_J] += e_def
        c_[CA_PROCESS_DAMPING_J] += e_pd
        c_[CA_ELEM_EVALS] += n_in
        c_[CA_EDGE_LEN_INT] += edge_len
        c_[CA_V_INT] += v_int
        if h_max > c_[CA_H_MAX]:
            c_[CA_H_MAX] = h_max
        section_peaks(tool, out, cacc, dt)
        st.misc[MS_STEPS] += 1.0
        st.misc[MS_LAST_TICK] = float(tick_end)
    # ---------------------------------------------------- the step's peak: the instantaneous wrench at its start or
    # end, or its mean, whichever has the largest |F| (T0; the ends rotated by the spindle angle at that instant)
    inv_dt = 1.0 / dt
    cxF = (F0x - E0x) * inv_dt
    cyF = (F0y - E0y) * inv_dt
    czF = (F0z - E0z) * inv_dt
    cxM = (M0x - E0mx) * inv_dt
    cyM = (M0y - E0my) * inv_dt
    czM = (M0z - E0mz) * inv_dt
    pkx = F0x * inv_dt
    pky = F0y * inv_dt
    pkz = F0z * inv_dt
    pmx = M0x * inv_dt
    pmy = M0y * inv_dt
    pmz = M0z * inv_dt
    best = pkx * pkx + pky * pky + pkz * pkz
    ch = math.cos(0.5 * dal)
    sh = math.sin(0.5 * dal)
    for end in range(2):
        if end == 0:
            ce = cm * ch + sm * sh
            se = sm * ch - cm * sh
            qx = ce * pS0 - se * pS1 + cxF
            qy = se * pS0 + ce * pS1 + cyF
            qz = pS2 + czF
            rx = ce * pS3 - se * pS4 + cxM
            ry = se * pS3 + ce * pS4 + cyM
            rz = pS5 + czM
        else:
            ce = cm * ch - sm * sh
            se = sm * ch + cm * sh
            qx = ce * pE0 - se * pE1 + cxF
            qy = se * pE0 + ce * pE1 + cyF
            qz = pE2 + czF
            rx = ce * pE3 - se * pE4 + cxM
            ry = se * pE3 + ce * pE4 + cyM
            rz = pE5 + czM
        q2 = qx * qx + qy * qy + qz * qz
        if q2 > best:
            best = q2
            pkx = qx
            pky = qy
            pkz = qz
            pmx = rx
            pmy = ry
            pmz = rz
    finish_step(pose, step, istep, axes, acc, cacc, out, commit, F0x, F0y, F0z, M0x, M0y, M0z, shaft, feed,
                M0z, power, n_in, pkx, pky, pkz, pmx, pmy, pmz)
    out.misc[SO_AXIAL_K] = Kz
    # ---------------------------------------------------- floating tapping holder (implicit static equilibrium)
    if float_k > 0.0 and commit:
        fz = FTz / dt
        u_new = (fz - Kz * u_f) / (float_k - Kz) if float_k - Kz > 0.0 else u_f
        trav = tool.scal[TC_FLOAT_TRAVEL]
        if trav > 0.0:
            if u_new > trav:
                u_new = trav
            elif u_new < -trav:
                u_new = -trav
        st.misc[MS_FLOAT_U] = u_new
        st.misc[MS_FLOAT_F] = fz
        st.misc[MS_FLOAT_K] = Kz
        if abs(u_new) > cacc.scalars[CA_FLOAT_U_MAX]:
            cacc.scalars[CA_FLOAT_U_MAX] = abs(u_new)
        cacc.scalars[CA_AXIAL_K] = Kz


# =============================================================================== turning (polar field)
POLAR_BISECT = 12       # sign bisections of polar_depth's bracketing half-cell (dz / 2 / 4096)


@njit(f8(POLAR_T, f8, f8, f8), **OPTS)
def polar_inside(polar, theta, r, z):
    """Signed radial distance into the material at (theta, r, z) of the rotating S (> 0 inside):
    min(r_out - r, r - r_in), bilinear in (theta, z) between the bin centres, less the air gap of the
    nearest cell (gap_lo < r < gap_hi: minus the distance to the gap's nearer side); outside the z range
    of the view there is no material."""
    nth = polar.r_out_m.shape[0]
    nz = polar.r_out_m.shape[1]
    dth = polar.geom[0]
    z0 = polar.geom[1]
    dz = polar.geom[2]
    fz = (z - z0) / dz - 0.5
    if fz < -0.5 or fz > nz - 0.5:
        return -1.0
    if fz < 0.0:
        fz = 0.0
    i0 = int(np.floor(fz))
    if i0 > nz - 2:
        i0 = nz - 2 if nz >= 2 else 0
    wz = fz - i0
    if wz > 1.0:
        wz = 1.0
    i1 = i0 + 1 if nz >= 2 else 0
    ft = theta / dth - 0.5
    bf = np.floor(ft)
    wt = ft - bf
    b0 = int(bf) % nth
    if b0 < 0:
        b0 += nth
    b1 = b0 + 1
    if b1 >= nth:
        b1 = 0
    ro0 = polar.r_out_m[b0, i0] + wz * (polar.r_out_m[b0, i1] - polar.r_out_m[b0, i0])
    ro1 = polar.r_out_m[b1, i0] + wz * (polar.r_out_m[b1, i1] - polar.r_out_m[b1, i0])
    ri0 = polar.r_in_m[b0, i0] + wz * (polar.r_in_m[b0, i1] - polar.r_in_m[b0, i0])
    ri1 = polar.r_in_m[b1, i0] + wz * (polar.r_in_m[b1, i1] - polar.r_in_m[b1, i0])
    ro = ro0 + wt * (ro1 - ro0)
    ri = ri0 + wt * (ri1 - ri0)
    g = ro - r
    g2 = r - ri
    if g2 < g:
        g = g2
    if ro <= ri:
        return -1.0
    # air gap of the nearest cell (two segments on the field's ray; additive, docs/models/hotloop.md)
    bn = int(np.floor(ft + 0.5)) % nth
    if bn < 0:
        bn += nth
    iz = int(np.floor(fz + 0.5))
    if iz > nz - 1:
        iz = nz - 1
    gl = polar.gap_lo_m[bn, iz]
    gh = polar.gap_hi_m[bn, iz]
    if gh > gl:
        d = r - gl
        d2 = gh - r
        if d2 < d:
            d = d2
        if d > 0.0 and -d < g:
            g = -d
    return g


@njit(types.UniTuple(f8, 2)(POLAR_T, f8, f8, f8, f8, f8, f8), **OPTS)
def polar_depth(polar, theta, r, z, ur, uz, cap):
    """Distance along the ray (r, z) + s (ur, uz) from a point inside the material to the boundary
    (the chip thickness when (ur, uz) = -n): march in half-cells of the z spacing, then bisect the bracketing
    half-cell on the sign (POLAR_BISECT halvings). polar_inside is a radial distance that jumps where the ray
    enters an air gap or the next row's surface, so the linear root of the two samples sat near the outside one
    (+2 % on the chip of a G71 pass at 36 um rows). Returns (h, 1 if the ray reached cap)."""
    g0 = polar_inside(polar, theta, r, z)
    if g0 <= 0.0:
        return 0.0, 0.0
    dz = polar.geom[2]
    ds = 0.5 * dz
    if ds <= 0.0:
        return 0.0, 0.0
    s0 = 0.0
    while s0 < cap:
        s1 = s0 + ds
        if s1 > cap:
            s1 = cap
        g1 = polar_inside(polar, theta, r + s1 * ur, z + s1 * uz)
        if g1 <= 0.0:
            for _ in range(POLAR_BISECT):
                sm = 0.5 * (s0 + s1)
                if polar_inside(polar, theta, r + sm * ur, z + sm * uz) > 0.0:
                    s0 = sm
                else:
                    s1 = sm
            return 0.5 * (s0 + s1), 0.0
        s0 = s1
        g0 = g1
    return cap, 1.0


@njit(types.void(TOOL_T, STATE_T, i8, f8, f8), **OPTS)
def turn_exit(tool, st, e, t_ex, h_ex):
    """A turning element left the material (h fell to 0): tooth exit bookkeeping (the polar field is the
    memory of turning: nothing to commit)."""
    j = tool.tooth[e]
    ts = st.tooth
    ts[j, TS_EXITS] += 1.0
    if ts[j, TS_EXIT_FIRST_S] < 0.0 or t_ex < ts[j, TS_EXIT_FIRST_S]:
        ts[j, TS_EXIT_FIRST_S] = t_ex
    if t_ex >= ts[j, TS_LAST_EXIT_S]:
        ts[j, TS_LAST_EXIT_S] = t_ex
        ts[j, TS_LAST_ELEM] = tool.elem[e]
        ts[j, TS_LAST_H] = h_ex
        ts[j, TS_LAST_ANGLE] = 0.0
    st.flags[e] = 0


@njit(types.void(TOOL_T, TABLE_T, POLAR_T, F1, I1, POSE_T, DEFLECTION_T, AXES_T, STATE_T, ACCUM_T, CACCUM_T,
                 OUT_T, EVENTS_T), **OPTS)
def turn_step(tool, table, polar, step, istep, pose, defl, axes, st, acc, cacc, out, ev):
    """One L0 step of a non-rotating tool (turning, facing, grooving, parting, threading, boring bars, form
    tools) on the rotating workpiece. Chip thickness of every element = depth of the element inside
    the polar field along -n (previous revolution or pass, so threading passes and interrupted cuts need
    no special case); forces per element from the table (TC_TURN_COLWELL 0) or lumped on the
    equivalent chip thickness A / L with the thrust normal to Colwell's chord (1); per-element
    integrals, heat by zone, chip volume per revolution, shocks, surface events every
    TC_SURF_EVENT_RAD of rotation at the generating element (smallest |v.n| / V)."""
    dt = step[ST_DT_S]
    if dt <= 0.0:
        return
    commit = istep[SI_COMMIT] != 0
    omega = step[ST_OMEGA_RAD_S]
    tick_end = istep[SI_TICK_END]
    l0 = istep[SI_L0_TICKS]
    tick0 = tick_end - l0
    tick_s = step[ST_TICK_S]
    t_start = tick_end * tick_s - dt
    dal = omega * dt
    abs_dal = abs(dal)
    nst = tool.st_z_m.shape[0]
    nt = st.tooth.shape[0]
    n = tool.p_T_m.shape[0]
    ratio = tool.scal[TC_RECOVERY_RATIO]
    pd_max = tool.scal[TC_PD_MAX_LEN]
    cap = tool.scal[TC_H_CAP]
    colwell = tool.scal[TC_TURN_COLWELL] > 0.5
    slot = float(istep[SI_SLOT])
    nlo = table.n_lo[0]
    nseg = table.coef.shape[1]
    coef = table.coef
    R = pose.R_S_T0
    g00 = R[0, 0]
    g01 = R[0, 1]
    g02 = R[0, 2]
    g10 = R[1, 0]
    g11 = R[1, 1]
    g12 = R[1, 2]
    g20 = R[2, 0]
    g21 = R[2, 1]
    g22 = R[2, 2]
    ox = pose.tip_S_m[0] - defl.u_work_S_m[0]
    oy = pose.tip_S_m[1] - defl.u_work_S_m[1]
    oz = pose.tip_S_m[2] - defl.u_work_S_m[2]
    wvx = pose.v_tip_S_m_s[0] - defl.v_work_S_m_s[0]
    wvy = pose.v_tip_S_m_s[1] - defl.v_work_S_m_s[1]
    wvz = pose.v_tip_S_m_s[2] - defl.v_work_S_m_s[2]
    vTx = g00 * wvx + g10 * wvy + g20 * wvz
    vTy = g01 * wvx + g11 * wvy + g21 * wvz
    vTz = g02 * wvx + g12 * wvy + g22 * wvz
    wkx = defl.v_work_S_m_s[0]
    wky = defl.v_work_S_m_s[1]
    wkz = defl.v_work_S_m_s[2]
    vWx = g00 * wkx + g10 * wky + g20 * wkz
    vWy = g01 * wkx + g11 * wky + g21 * wkz
    vWz = g02 * wkx + g12 * wky + g22 * wkz
    # the work's deflection in T0 (process damping's wave memory)
    uw = defl.u_work_S_m
    uw0x = g00 * uw[0] + g10 * uw[1] + g20 * uw[2]
    uw0y = g01 * uw[0] + g11 * uw[1] + g21 * uw[2]
    uw0z = g02 * uw[0] + g12 * uw[1] + g22 * uw[2]
    pd_ring = st.pd_ring
    pd_aux = st.pd_aux
    pd_dx = pd_max / (PD_RING_N - 1)
    fwx = pose.w_T0_S_rad_s[0]
    fwy = pose.w_T0_S_rad_s[1]
    fwz = pose.w_T0_S_rad_s[2]
    OTx = g00 * fwx + g10 * fwy + g20 * fwz
    OTy = g01 * fwx + g11 * fwy + g21 * fwz
    OTz = g02 * fwx + g12 * fwy + g22 * fwz
    ts = st.tooth
    lp = st.last_p
    for s in range(nst):
        if st.st_written[s] != 0:
            for c in range(6):
                out.station_T0[s, c] = 0.0
    for k in range(out.sec.shape[0]):
        for c in range(9):
            out.sec[k, c] = 0.0
    if commit:
        for j in range(nt):
            ts[j, TS_FX] = 0.0
            ts[j, TS_FY] = 0.0
            ts[j, TS_FZ] = 0.0
            ts[j, TS_NEW] = 0.0
            ts[j, TS_NOW] = 0.0
            ts[j, TS_EXITS] = 0.0
            ts[j, TS_CENTRE_WAS] = 0.0
            ts[j, TS_CENTRE_ON] = 0.0
            ts[j, TS_STEP_DV] = 0.0
            ts[j, TS_STEP_B] = 0.0
            for c in range(TS_S_PX, TS_S_NZ + 1):
                ts[j, c] = 0.0
    # ---------------------------------------------------- pass 1: chip thickness of every element from the field
    A_tot = 0.0
    L_tot = 0.0
    first = -1
    last = -1
    best_row = 0
    best_A = -1.0
    for s in range(nst):
        for kk in range(tool.st_ptr[s], tool.st_ptr[s + 1]):
            e = tool.st_elem[kk]
            px = tool.p_T_m[e, 0]
            py = tool.p_T_m[e, 1]
            pz = tool.p_T_m[e, 2]
            pzr = pz - tool.st_z_m[s]
            ux = defl.u_st_T0_m[s, 0] + (defl.u_st_T0_m[s, 4] * pzr - defl.u_st_T0_m[s, 5] * py)
            uy = defl.u_st_T0_m[s, 1] + (defl.u_st_T0_m[s, 5] * px - defl.u_st_T0_m[s, 3] * pzr)
            uz = defl.u_st_T0_m[s, 2] + (defl.u_st_T0_m[s, 3] * py - defl.u_st_T0_m[s, 4] * px)
            qx = px + ux
            qy = py + uy
            qz = pz + uz
            psx = ox + g00 * qx + g01 * qy + g02 * qz
            psy = oy + g10 * qx + g11 * qy + g12 * qz
            psz = oz + g20 * qx + g21 * qy + g22 * qz
            nx = tool.n_T[e, 0]
            ny = tool.n_T[e, 1]
            nz = tool.n_T[e, 2]
            nsx = g00 * nx + g01 * ny + g02 * nz
            nsy = g10 * nx + g11 * ny + g12 * nz
            nsz = g20 * nx + g21 * ny + g22 * nz
            r = math.sqrt(psx * psx + psy * psy)
            th = math.atan2(psy, psx)
            if th < 0.0:
                th += TWO_PI
            if r > 0.0:
                n_r = (nsx * psx + nsy * psy) / r
            else:
                n_r = 0.0
            n_z = nsz
            nn = math.sqrt(n_r * n_r + n_z * n_z)
            h = 0.0
            if nn > 1e-12:
                h, capped = polar_depth(polar, th, r, psz, -n_r / nn, -n_z / nn, cap)
                if capped > 0.5:
                    cacc.scalars[CA_TABLE_OVER] += 1.0
            lp[e, 0] = psx
            lp[e, 1] = psy
            lp[e, 2] = psz
            lp[e, 3] = h
            lp[e, 4] = nsx
            lp[e, 5] = nsy
            lp[e, 6] = nsz
            lp[e, 7] = g00 * ux + g01 * uy + g02 * uz - defl.u_work_S_m[0]
            lp[e, 8] = g10 * ux + g11 * uy + g12 * uz - defl.u_work_S_m[1]
            lp[e, 9] = g20 * ux + g21 * uy + g22 * uz - defl.u_work_S_m[2]
            if h > 0.0:
                # the chip section of the element: the strip behind its curved edge (abi.edge_curvature)
                hk = h * tool.pk[kk, PK_CURV]
                if hk > 1.0:
                    hk = 1.0
                Ae = h * tool.db_m[e] * (1.0 - 0.5 * hk)
                A_tot += Ae
                L_tot += tool.db_m[e]
                if first < 0 or e < first:
                    first = e
                if e > last:
                    last = e
                if Ae > best_A:
                    best_A = Ae
                    best_row = tool.row[e]
    # Colwell (1954) chord and Woxen's equivalent chip thickness (Young, Mathew & Oxley 1987)
    h_eq = A_tot / L_tot if L_tot > 0.0 else 0.0
    cnx = 0.0
    cnz = 0.0
    chx = 0.0
    chz = 0.0
    if colwell and first >= 0:
        dx_ = tool.p_T_m[last, 0] - tool.p_T_m[first, 0]
        dz_ = tool.p_T_m[last, 2] - tool.p_T_m[first, 2]
        dl_ = math.sqrt(dx_ * dx_ + dz_ * dz_)
        if dl_ > 1e-12:
            chx = dx_ / dl_
            chz = dz_ / dl_
            cnx = chz
            cnz = -chx
            mnx = 0.0
            mnz = 0.0
            for e in range(first, last + 1):
                if lp[e, 3] > 0.0:
                    mnx += tool.n_T[e, 0] * lp[e, 3] * tool.db_m[e]
                    mnz += tool.n_T[e, 2] * lp[e, 3] * tool.db_m[e]
            if cnx * mnx + cnz * mnz < 0.0:
                cnx = -cnx
                cnz = -cnz
        else:
            cnx = tool.n_T[first, 0]
            cnz = tool.n_T[first, 2]
            chx = cnz
            chz = -cnx
    Ft_l = 0.0
    Fr_l = 0.0
    Fa_l = 0.0
    if colwell and h_eq > 0.0:
        row = best_row
        hm = table.h_min_m[row]
        if h_eq < hm:
            sg = int(h_eq * table.inv_dlo[row])
            if sg >= nlo:
                sg = nlo - 1
        else:
            xg = (h_eq - hm) * table.inv_dhi[row]
            sg = nlo + int(xg) if xg < nseg - nlo else nseg - 1
        Ft_l = (coef[row, sg, 0, COL_FT] + coef[row, sg, 1, COL_FT] * h_eq) * L_tot
        Fr_l = (coef[row, sg, 0, COL_FR] + coef[row, sg, 1, COL_FR] * h_eq) * L_tot
        Fa_l = (coef[row, sg, 0, COL_FA] + coef[row, sg, 1, COL_FA] * h_eq) * L_tot
    # ---------------------------------------------------- pass 2: forces and bookkeeping
    FTx = 0.0
    FTy = 0.0
    FTz = 0.0
    MTx = 0.0
    MTy = 0.0
    MTz = 0.0
    e_shear = 0.0
    e_rake = 0.0
    e_plough = 0.0
    e_flank = 0.0
    e_pd = 0.0
    e_def = 0.0
    vol_chip = 0.0
    n_in = 0.0
    edge_len = 0.0
    v_int = 0.0
    h_max = 0.0
    gen_e = -1
    gen_q = 1e300
    for s in range(nst):
        k0 = tool.st_ptr[s]
        k1 = tool.st_ptr[s + 1]
        if k1 <= k0:
            continue
        zst = tool.st_z_m[s]
        vuTx = defl.v_st_T0_m_s[s, 0]
        vuTy = defl.v_st_T0_m_s[s, 1]
        vuTz = defl.v_st_T0_m_s[s, 2]
        vrTx = defl.v_st_T0_m_s[s, 3]
        vrTy = defl.v_st_T0_m_s[s, 4]
        vrTz = defl.v_st_T0_m_s[s, 5]
        sFx = 0.0
        sFy = 0.0
        sFz = 0.0
        sMx = 0.0
        sMy = 0.0
        sMz = 0.0
        any_in = False
        for kk in range(k0, k1):
            e = tool.st_elem[kk]
            j = tool.tooth[e]
            flags = st.flags[e]
            h = lp[e, 3]
            px = tool.p_T_m[e, 0]
            py = tool.p_T_m[e, 1]
            pz = tool.p_T_m[e, 2]
            pzr = pz - zst
            nx = tool.n_T[e, 0]
            ny = tool.n_T[e, 1]
            nz = tool.n_T[e, 2]
            vex = vTx + vuTx + (OTy * pz - OTz * py) + (vrTy * pzr - vrTz * py)
            vey = vTy + vuTy + (OTz * px - OTx * pz) + (vrTz * px - vrTx * pzr)
            vez = vTz + vuTz + (OTx * py - OTy * px) + (vrTx * py - vrTy * px)
            vn = vex * nx + vey * ny + vez * nz
            Vc = vex * tool.c_T[e, 0] + vey * tool.c_T[e, 1] + vez * tool.c_T[e, 2]
            if h <= 0.0:
                if (flags & FL_IN_CUT) != 0 and commit:
                    hl = st.h_last_m[e]
                    q = hl / (hl - h) if hl > h else 0.0
                    turn_exit(tool, st, e, t_start + q * dt, hl)
                if commit:
                    st.h_last_m[e] = 0.0
                    st.tick_last[e] = tick_end
                continue
            row = tool.row[e]
            hm = table.h_min_m[row]
            if h < hm:
                sg = int(h * table.inv_dlo[row])
                if sg >= nlo:
                    sg = nlo - 1
            else:
                xg = (h - hm) * table.inv_dhi[row]
                if xg < nseg - nlo:
                    sg = nlo + int(xg)
                else:
                    sg = nseg - 1
                    cacc.scalars[CA_TABLE_OVER] += 1.0
            db0 = tool.db_m[e]
            hk = h * tool.pk[kk, PK_CURV]
            if hk > 1.0:
                hk = 1.0
            db = db0 * (1.0 - 0.5 * hk)
            dS = tool.dS_m[e]
            fft = tool.flank_N[e, 0]
            ffr = tool.flank_N[e, 1]
            fpd = 0.0
            was_in = (flags & FL_IN_CUT) != 0
            kpd = tool.pd[e, 0]
            pux = 0.0
            puy = 0.0
            puz = 0.0
            if kpd > 0.0:
                # process damping from the surface the edge generated behind it (wave memory, see cut_step): the
                # element's relative vibration displacement in T0 (= T for a non-rotating tool)
                us = defl.u_st_T0_m
                pux = us[s, 0] + us[s, 4] * pzr - us[s, 5] * py - uw0x
                puy = us[s, 1] + us[s, 5] * px - us[s, 3] * pzr - uw0y
                puz = us[s, 2] + us[s, 3] * py - us[s, 4] * px - uw0z
                if Vc > 0.0 and was_in:
                    # behind the edge the work surface goes round once: beyond its circumference the surface is
                    # from the previous revolution
                    xlim = TWO_PI * math.sqrt(lp[e, 0] * lp[e, 0] + lp[e, 1] * lp[e, 1])
                    if xlim > pd_max:
                        xlim = pd_max
                    fpd = kpd * wave_area(pd_ring, pd_aux, e, pux, puy, puz, nx, ny, nz, pd_aux[e, PDA_X] + Vc * dt,
                                          tool.pd[e, 1], tool.pd[e, 2], xlim, pd_dx)
            mu_pd = tool.pd[e, 3]
            if colwell and A_tot > 0.0:
                wgt = h * db / A_tot
                ft0 = Ft_l * wgt + (coef[row, sg, 0, COL_ET] + coef[row, sg, 1, COL_ET] * h) * dS
                frc = Fr_l * wgt
                fac = Fa_l * wgt
                fre = (coef[row, sg, 0, COL_ER] + coef[row, sg, 1, COL_ER] * h) * dS
                fae = (coef[row, sg, 0, COL_EA] + coef[row, sg, 1, COL_EA] * h) * dS
                # shearing thrust along -n_c (Colwell), lateral along the chord; edge parts on the element's own axes
                ft = ft0 + fft + mu_pd * fpd
                fr = fre + ffr + fpd
                fa = fae
                Fx = ft * tool.dir_T[e, 0, 0] + fr * tool.dir_T[e, 1, 0] + fa * tool.dir_T[e, 2, 0] - frc * cnx + fac * chx
                Fy = ft * tool.dir_T[e, 0, 1] + fr * tool.dir_T[e, 1, 1] + fa * tool.dir_T[e, 2, 1]
                Fz = ft * tool.dir_T[e, 0, 2] + fr * tool.dir_T[e, 1, 2] + fa * tool.dir_T[e, 2, 2] - frc * cnz + fac * chz
                Mx = py * Fz - pzr * Fy
                My = pzr * Fx - px * Fz
                Mz = px * Fy - py * Fx
                fr0 = fre + frc
                fa0 = fae + fac
                va = vex * tool.dir_T[e, 2, 0] + vey * tool.dir_T[e, 2, 1] + vez * tool.dir_T[e, 2, 2]
                vch = vex * chx + vez * chz
                p_tab = ft0 * Vc + fre * vn - fae * va + frc * (vex * cnx + vez * cnz) - fac * vch
            else:
                ft0 = (coef[row, sg, 0, COL_FT] + coef[row, sg, 1, COL_FT] * h) * db + \
                    (coef[row, sg, 0, COL_ET] + coef[row, sg, 1, COL_ET] * h) * dS
                fr0 = (coef[row, sg, 0, COL_FR] + coef[row, sg, 1, COL_FR] * h) * db + \
                    (coef[row, sg, 0, COL_ER] + coef[row, sg, 1, COL_ER] * h) * dS
                fa0 = (coef[row, sg, 0, COL_FA] + coef[row, sg, 1, COL_FA] * h) * db + \
                    (coef[row, sg, 0, COL_EA] + coef[row, sg, 1, COL_EA] * h) * dS
                ft = ft0 + fft + mu_pd * fpd
                fr = fr0 + ffr + fpd
                fa = fa0
                Fx = ft * tool.dir_T[e, 0, 0] + fr * tool.dir_T[e, 1, 0] + fa * tool.dir_T[e, 2, 0]
                Fy = ft * tool.dir_T[e, 0, 1] + fr * tool.dir_T[e, 1, 1] + fa * tool.dir_T[e, 2, 1]
                Fz = ft * tool.dir_T[e, 0, 2] + fr * tool.dir_T[e, 1, 2] + fa * tool.dir_T[e, 2, 2]
                Mx = ft * tool.arm_T[e, 0, 0] + fr * tool.arm_T[e, 1, 0] + fa * tool.arm_T[e, 2, 0]
                My = ft * tool.arm_T[e, 0, 1] + fr * tool.arm_T[e, 1, 1] + fa * tool.arm_T[e, 2, 1]
                Mz = ft * tool.arm_T[e, 0, 2] + fr * tool.arm_T[e, 1, 2] + fa * tool.arm_T[e, 2, 2]
                va = vex * tool.dir_T[e, 2, 0] + vey * tool.dir_T[e, 2, 1] + vez * tool.dir_T[e, 2, 2]
                p_tab = ft0 * Vc + fr0 * vn - fa0 * va
            p_fl = fft * Vc + ffr * vn + mu_pd * fpd * Vc
            p_pd = fpd * vn
            sFx += Fx * dt
            sFy += Fy * dt
            sFz += Fz * dt
            sMx += Mx * dt
            sMy += My * dt
            sMz += Mz * dt
            any_in = True
            n_in += 1.0
            if h > h_max:
                h_max = h
            if not commit:
                continue
            ps = coef[row, sg, 0, COL_PS] + coef[row, sg, 1, COL_PS] * h
            pr = coef[row, sg, 0, COL_PR] + coef[row, sg, 1, COL_PR] * h
            rc = coef[row, sg, 0, COL_RC] + coef[row, sg, 1, COL_RC] * h
            et = p_tab * dt
            r_e = math.sqrt(lp[e, 0] * lp[e, 0] + lp[e, 1] * lp[e, 1])
            # Pappus: the chip strip turns about the spindle axis at its centroid, h / 2 along the depth ray -n
            r_c = r_e
            if r_e > 0.0:
                n_r = (lp[e, 4] * lp[e, 0] + lp[e, 5] * lp[e, 1]) / r_e
                nn = math.sqrt(n_r * n_r + lp[e, 6] * lp[e, 6])
                if nn > 1e-12:
                    r_c = r_e - 0.5 * h * n_r / nn
            dV = h * db * r_c * abs_dal
            e_shear += et * ps
            e_rake += et * pr
            e_plough += et * (1.0 - ps - pr)
            vol_chip += dV
            e_flank += p_fl * dt
            e_pd += p_pd * dt
            edge_len += dS * dt
            v_int += Vc * dt
            ea = st.elem_acc
            ea[e, ELEM_IN_CUT_S] += dt
            sl = abs(Vc) * dt
            ea[e, ELEM_RAKE_SLIDING_M] += rc * sl
            ea[e, ELEM_FLANK_SLIDING_M] += sl
            ea[e, ELEM_INT_SIGMA_V_RAKE] += (coef[row, sg, 0, COL_SIG_RAKE] + coef[row, sg, 1, COL_SIG_RAKE] * h) \
                * rc * sl
            ea[e, ELEM_INT_SIGMA_V_FLANK] += tool.sig_flank_Pa[e] * sl
            ea[e, ELEM_INT_H_DT] += h * dt
            if h > ea[e, ELEM_H_MAX]:
                ea[e, ELEM_H_MAX] = h
            if (flags & FL_IN_CUT) == 0:
                ea[e, ELEM_ENTRIES] += 1.0
                ts[j, TS_NEW] += 1.0
                hl = st.h_last_m[e]
                q = hl / (hl - h) if (st.tick_last[e] == tick0 and hl < h and hl <= 0.0) else 0.5
                t_en = t_start + (q - 0.5) * dt + 0.5 * dt
                if ts[j, TS_PASS_ON] < 0.5:
                    ts[j, TS_PASS_ON] = 1.0
                    ts[j, TS_ENTRY_TICK] = float(tick0 + int(np.floor(q * l0 + 0.5)))
                    ts[j, TS_ENTRY_ELEM] = float(tool.elem[e])
                    ts[j, TS_ENTRY_H] = h
                    ts[j, TS_ENTRY_ANGLE] = 0.0
                    ts[j, TS_LOADING] = 1.0
                    ts[j, TS_FIRST_ENTRY_S] = t_en
                    ts[j, TS_LAST_ENTRY_S] = t_en
                    ts[j, TS_FMAX] = 0.0
                    ts[j, TS_BREAKOUT] = 0.0
                    ts[j, TS_EXIT_FIRST_S] = -1.0
                    ts[j, TS_LAST_EXIT_S] = -1.0
                    ts[j, TS_EMIT_TICK] = float(tick_end)
                    cacc.scalars[CA_ENTRIES] += 1.0
                elif ts[j, TS_LOADING] > 0.5 and t_en > ts[j, TS_LAST_ENTRY_S]:
                    ts[j, TS_LAST_ENTRY_S] = t_en
                flags = FL_IN_CUT
            if kpd > 0.0:
                wave_push(pd_ring, pd_aux, e, pux, puy, puz, Vc * dt if Vc > 0.0 else 0.0, pd_dx, not was_in)
            ts[j, TS_NOW] += 1.0
            ts[j, TS_FX] += Fx
            ts[j, TS_FY] += Fy
            ts[j, TS_FZ] += Fz
            ts[j, TS_STEP_B] += db0
            tch = coef[row, sg, 0, COL_TCHIP] + coef[row, sg, 1, COL_TCHIP] * h
            ws = coef[row, sg, 0, COL_WS] + coef[row, sg, 1, COL_WS] * h
            wf = coef[row, sg, 0, COL_WF] + coef[row, sg, 1, COL_WF] * h
            ts[j, TS_VOL] += dV
            ts[j, TS_HV] += h * dV
            ts[j, TS_VV] += Vc * dV
            ts[j, TS_RAKEV] += tool.rake_rad[e] * dV
            ts[j, TS_INCLV] += tool.incl_rad[e] * dV
            ts[j, TS_TV] += tch * dV
            ts[j, TS_RV] += r_e * dV
            if dV > ts[j, TS_DOM_VOL]:
                ts[j, TS_DOM_VOL] = dV
                ts[j, TS_DOM_KIND] = float(tool.kind[e])
                ts[j, TS_DOM_ROW] = float(row)
            ts[j, TS_STEP_DV] += dV
            ts[j, TS_S_PX] += (g00 * lp[e, 0] + g10 * lp[e, 1] + g20 * lp[e, 2]) * dV
            ts[j, TS_S_PY] += (g01 * lp[e, 0] + g11 * lp[e, 1] + g21 * lp[e, 2]) * dV
            ts[j, TS_S_PZ] += (g02 * lp[e, 0] + g12 * lp[e, 1] + g22 * lp[e, 2]) * dV
            vch = rc * Vc
            ts[j, TS_S_VX] += (vex + vch * tool.chip_T[e, 0]) * dV
            ts[j, TS_S_VY] += (vey + vch * tool.chip_T[e, 1]) * dV
            ts[j, TS_S_VZ] += (vez + vch * tool.chip_T[e, 2]) * dV
            ts[j, TS_S_NX] += nx * dV
            ts[j, TS_S_NY] += ny * dV
            ts[j, TS_S_NZ] += nz * dV
            ts[j, TS_WORK_J] += et * ps * ws + (et * (1.0 - ps - pr) + p_fl * dt) * wf
            qv = abs(vn) / Vc if Vc > 0.0 else 1e300
            if qv < gen_q:
                gen_q = qv
                gen_e = e
            st.flags[e] = flags
            st.h_last_m[e] = h
            st.tick_last[e] = tick_end
        if any_in:
            inv = 1.0 / dt
            out.station_T0[s, 0] = sFx * inv
            out.station_T0[s, 1] = sFy * inv
            out.station_T0[s, 2] = sFz * inv
            out.station_T0[s, 3] = sMx * inv
            out.station_T0[s, 4] = sMy * inv
            out.station_T0[s, 5] = sMz * inv
            st.st_written[s] = 2
            FTx += sFx
            FTy += sFy
            FTz += sFz
            MTx += sMx - zst * sFy
            MTy += sMy + zst * sFx
            MTz += sMz
            e_def += sFx * vuTx + sFy * vuTy + sFz * vuTz + sMx * vrTx + sMy * vrTy + sMz * vrTz
            sb = tool.st_sec[s]
            if 0 <= sb < out.sec.shape[0]:
                out.sec[sb, 0] += sFx
                out.sec[sb, 1] += sFy
                out.sec[sb, 2] += sFz
                out.sec[sb, 3] += zst * sFx
                out.sec[sb, 4] += zst * sFy
                out.sec[sb, 5] += zst * sFz
                out.sec[sb, 6] += sMx
                out.sec[sb, 7] += sMy
                out.sec[sb, 8] += sMz
    for s in range(nst):
        if st.st_written[s] == 2:
            st.st_written[s] = 1
        elif st.st_written[s] == 1:
            st.st_written[s] = 0
    e_def -= FTx * vWx + FTy * vWy + FTz * vWz
    # surface events: the generating element (envelope v.n = 0 along the edge) every TC_SURF_EVENT_RAD
    if commit and gen_e >= 0:
        st.misc[MS_TURN_EVENT_ANGLE] += abs_dal
        if st.misc[MS_TURN_EVENT_ANGLE] >= tool.scal[TC_SURF_EVENT_RAD]:
            st.misc[MS_TURN_EVENT_ANGLE] = 0.0
            e = gen_e
            put_surface(ev.surface.rec, ev.surface.n, float(tick_end), float(tool.tooth[e]), float(tool.elem[e]),
                        lp[e, 0], lp[e, 1], lp[e, 2], -lp[e, 4], -lp[e, 5], -lp[e, 6], lp[e, 3],
                        tool.edge_radius_m[e], tool.VB_m[e], lp[e, 7], lp[e, 8], lp[e, 9], slot, cacc)
    if commit:
        # turning: full contact when no element entered during the step
        for j in range(nt):
            ts[j, TS_AVAIL] = ts[j, TS_NOW] if ts[j, TS_NEW] == 0.0 else ts[j, TS_NOW] + 1.0
    close_teeth(tool, st, ev, cacc, pose, step, istep, commit, g00, g01, g02, g10, g11, g12, g20, g21, g22, abs_dal)
    # shaft: work spindle (rotation of S about its z axis, w_sp = -omega e_z relative to the tool); feed: the rest
    fsx = g00 * FTx + g01 * FTy + g02 * FTz
    fsy = g10 * FTx + g11 * FTy + g12 * FTz
    fsz = g20 * FTx + g21 * FTy + g22 * FTz
    msx = g00 * MTx + g01 * MTy + g02 * MTz
    msy = g10 * MTx + g11 * MTy + g12 * MTz
    msz = g20 * MTx + g21 * MTy + g22 * MTz
    tip = pose.tip_S_m
    vrx = omega * tip[1]
    vry = -omega * tip[0]
    shaft = -(vrx * fsx + vry * fsy - omega * msz)
    feed = -((pose.v_tip_S_m_s[0] - vrx) * fsx + (pose.v_tip_S_m_s[1] - vry) * fsy + pose.v_tip_S_m_s[2] * fsz
             + fwx * msx + fwy * msy + (fwz + omega) * msz)
    torque_work = -(tip[0] * fsy - tip[1] * fsx + msz)
    power = e_shear + e_rake + e_plough + e_flank + e_pd
    if commit:
        a_ = acc.scalars
        a_[ACC_HEAT_SHEAR_J] += e_shear
        a_[ACC_HEAT_RAKE_J] += e_rake
        a_[ACC_HEAT_FLANK_J] += e_flank
        a_[ACC_PLOUGHING_J] += e_plough
        a_[ACC_PROCESS_DAMPING_J] += e_pd
        a_[ACC_CHIP_VOLUME_M3] += vol_chip
        c_ = cacc.scalars
        c_[CA_DEFLECTION_J] += e_def
        c_[CA_PROCESS_DAMPING_J] += e_pd
        c_[CA_ELEM_EVALS] += n_in
        c_[CA_EDGE_LEN_INT] += edge_len
        c_[CA_V_INT] += v_int
        if h_max > c_[CA_H_MAX]:
            c_[CA_H_MAX] = h_max
        section_peaks(tool, out, cacc, dt)
        st.misc[MS_STEPS] += 1.0
        st.misc[MS_LAST_TICK] = float(tick_end)
    finish_step(pose, step, istep, axes, acc, cacc, out, commit, FTx, FTy, FTz, MTx, MTy, MTz, shaft, feed,
                torque_work * dt, power, n_in, FTx / dt, FTy / dt, FTz / dt, MTx / dt, MTy / dt, MTz / dt)
