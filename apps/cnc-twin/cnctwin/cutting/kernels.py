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
                  CA_V_INT, CA_OMEGA_INT, CA_TIME, CA_FPEAK_MX, CA_FPEAK_MY, CA_FPEAK_MZ, CACCUM_T, COL_EA, COL_ER, COL_ET, COL_FA, COL_FR, COL_FT, COL_PR, COL_PS, COL_RC,
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
                  TS_STEP_B, TS_STEP_DV, TS_STEP_E, TS_TV, TS_UX, TS_UY, TS_UZ, TS_VOL, TS_VV, TS_WORK_J, TS_AVAIL)

OPTS = dict(cache=True, nogil=True, error_model="numpy")
TWO_PI = 2.0 * math.pi
f8 = types.float64
i8 = types.int64
b1 = types.boolean
F1 = types.float64[::1]
F2 = types.float64[:, ::1]
I1 = types.int64[::1]
GATE_T = types.UniTuple(f8, 4)
SIX_T = types.UniTuple(f8, 6)


def _modal_example() -> ModalModel:
    return ModalModel(np.zeros((1, 2, 2)), np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 1, 6)), np.zeros((1, 3)),
                      np.zeros(1, np.int8), np.zeros(1, np.int32), np.zeros((3, 3)), np.zeros(1))


MODAL_T = typeof(_modal_example())
MSTATE_T = typeof(ModalState(np.zeros((1, 2)), np.zeros(1)))


# =============================================================================== event rows
@njit(b1(F2, I1), **OPTS)
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
@njit(f8(f8, f8, f8, f8), **OPTS)
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


@njit(GATE_T(INTERVALS_T, i8, f8, f8, f8), **OPTS)
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


@njit(b1(INTERVALS_T, i8, f8), **OPTS)
def in_intervals(iv, s, a):
    """True when the right-handed T0 angle a (any value) lies in one of station s's engaged intervals."""
    x = a - TWO_PI * np.floor(a / TWO_PI)
    for i in range(iv.count[s]):
        lo = iv.lo_rad[s, i]
        hi = iv.hi_rad[s, i]
        if lo <= x <= hi or lo <= x + TWO_PI <= hi:
            return True
    return False


@njit(b1(INTERVALS_T, i8, f8, f8), **OPTS)
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


@njit(types.void(MEMORY_T, i8, i8, f8, f8, f8, f8, f8, f8, i8), **OPTS)
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


@njit(b1(MEMORY_T, i8, i8, i8, i8, i8), **OPTS)
def bin_fresh(mem, s, b, tick, stale_ticks, stale_before):
    if mem.valid[s, b] == 0:
        return False
    t = mem.tick[s, b]
    return t >= stale_before and tick - t <= stale_ticks


@njit(f8(REFILL_T, i8, i8, f8, f8, f8, f8, f8, f8, i8, b1, EVENTS_T, CACCUM_T), **OPTS)
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


@njit(b1(MEMORY_T, REFILL_T, i8, i8, f8, f8, f8, f8, f8, f8, i8, b1, EVENTS_T, CACCUM_T), **OPTS)
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


@njit(i8(TABLE_T, i8, f8, CACCUM_T), **OPTS)
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
    k = nlo + int((h - hm) * table.inv_dhi[row])
    nseg = table.coef.shape[1]
    if k >= nseg:
        cacc.scalars[CA_TABLE_OVER] += 1.0
        k = nseg - 1
    return k


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


@njit(types.void(TOOL_T, TABLE_T, MEMORY_T, STATE_T, EVENTS_T, CACCUM_T, i8, f8, f8, f8, i8, b1, f8, f8), **OPTS)
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
@njit(SIX_T(TOOL_T, TABLE_T, OCCUPANCY_T, POSE_T, F1, I1, CACCUM_T, ACCUM_T, EVENTS_T, f8), **OPTS)
def recut_scan(tool, table, occ, pose, step, istep, cacc, acc, ev, theta):
    """Chips of the frozen occupancy grid inside the tool's cutting envelope are struck once (scan once per
    tooth period): each takes the impulse J = m V + E_cut / V along the edge motion (an inelastic impact
    that brings it to the edge speed V, plus the cutting energy E_cut = u(h_chip) m / rho of the table row
    TC_RECUT_ROW), applied to the tool at once (returned: impulse and moment about the tip, T0); recut
    rows go to the chips domain."""
    cell = occ.cell_m[0]
    rcut = tool.scal[TC_R_CUT]
    if cell <= 0.0 or rcut <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    nxg = occ.mass_kg.shape[0]
    nyg = occ.mass_kg.shape[1]
    nzg = occ.mass_kg.shape[2]
    R = pose.R_S_T0
    tip = pose.tip_S_m
    zlo = tool.scal[TC_Z_LO]
    zhi = tool.scal[TC_Z_HI]
    lo = np.empty(3)
    hi = np.empty(3)
    for c in range(3):
        a0 = tip[c] + R[c, 2] * zlo
        a1 = tip[c] + R[c, 2] * zhi
        lo[c] = min(a0, a1) - rcut
        hi[c] = max(a0, a1) + rcut
    o = occ.origin_S_m
    i0 = max(0, int(np.floor((lo[0] - o[0]) / cell)))
    i1 = min(nxg - 1, int(np.floor((hi[0] - o[0]) / cell)))
    j0 = max(0, int(np.floor((lo[1] - o[1]) / cell)))
    j1 = min(nyg - 1, int(np.floor((hi[1] - o[1]) / cell)))
    k0 = max(0, int(np.floor((lo[2] - o[2]) / cell)))
    k1 = min(nzg - 1, int(np.floor((hi[2] - o[2]) / cell)))
    omega = step[ST_OMEGA_RAD_S]
    abs_om = abs(omega)
    if abs_om <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    sgn = 1.0 if omega > 0.0 else -1.0
    rho = step[ST_RHO_KG_M3]
    row = int(tool.scal[TC_RECUT_ROW])
    nrow = table.coef.shape[0]
    tick = float(istep[SI_TICK_END])
    RM = pose.R_M_S
    tM = pose.t_M_S_m
    nt = tool.tooth_a0_rad.shape[0]
    nst = tool.st_z_m.shape[0]
    Jx = 0.0
    Jy = 0.0
    Mx = 0.0
    My = 0.0
    Mz = 0.0
    for i in range(i0, i1 + 1):
        for jj in range(j0, j1 + 1):
            for k in range(k0, k1 + 1):
                m = occ.mass_kg[i, jj, k] - occ.taken_kg[i, jj, k]
                if m <= 1e-18:
                    continue
                cx = o[0] + (i + 0.5) * cell
                cy = o[1] + (jj + 0.5) * cell
                cz = o[2] + (k + 0.5) * cell
                dx = cx - tip[0]
                dy = cy - tip[1]
                dz = cz - tip[2]
                tx = R[0, 0] * dx + R[1, 0] * dy + R[2, 0] * dz
                ty = R[0, 1] * dx + R[1, 1] * dy + R[2, 1] * dz
                tz = R[0, 2] * dx + R[1, 2] * dy + R[2, 2] * dz
                if tz < zlo or tz > zhi:
                    continue
                r2 = tx * tx + ty * ty
                if r2 > rcut * rcut or r2 <= 1e-18:
                    continue
                r = math.sqrt(r2)
                V = abs_om * r
                occ.taken_kg[i, jj, k] += m
                hch = occ.thickness_m[i, jj, k]
                if hch <= 0.0:
                    hch = occ.size_m[i, jj, k]
                u = 0.0
                if hch > 0.0 and 0 <= row < nrow:
                    seg = table_segment(table, row, hch, cacc)
                    u = (table.coef[row, seg, 0, COL_FT] + table.coef[row, seg, 1, COL_FT] * hch) / hch
                e_cut = u * m / rho if rho > 0.0 else 0.0
                J = m * V + e_cut / V
                cdx = -sgn * ty / r
                cdy = sgn * tx / r
                jx = -J * cdx
                jy = -J * cdy
                Jx += jx
                Jy += jy
                Mx += -tz * jy
                My += tz * jx
                Mz += tx * jy - ty * jx
                cacc.scalars[CA_RECUT_J] += J * V
                cacc.scalars[CA_RECUT_KG] += m
                acc.scalars[ACC_RECUT_MASS_KG] += m
                # the tooth that strikes it: the next one to reach its angle in the cutting sense
                av = math.atan2(ty, tx)
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
                    d = abs(tool.st_z_m[s] - tz)
                    if d < near_d:
                        near_d = d
                        near = s
                put_recut(ev.recut.rec, ev.recut.n, tick, float(best), float(near),
                          m, RM[0, 0] * cx + RM[0, 1] * cy + RM[0, 2] * cz + tM[0],
                          RM[1, 0] * cx + RM[1, 1] * cy + RM[1, 2] * cz + tM[1], cacc)
    return Jx, Jy, 0.0, Mx, My, Mz


# =============================================================================== step totals -> outputs
@njit(types.void(POSE_T, F1, I1, AXES_T, ACCUM_T, CACCUM_T, OUT_T, b1, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8,
                 f8), **OPTS)
def finish_step(pose, step, istep, axes, acc, cacc, out, commit, Fx, Fy, Fz, Mx, My, Mz, shaft, feed,
                spindle_torque, power, n_in):
    """Impulses of the step (on the tool, T0, moment about the tip) -> StepOut means, the per-tick wrenches of
    tool and part, axis loads, spindle torque, peaks and integrals (architecture 5.4 "Loads to L1")."""
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
    fmag = math.sqrt(wF0 * wF0 + wF1 * wF1 + wF2 * wF2)
    tq = abs(spindle_torque * inv)
    pw = abs(spindle_torque * inv * step[ST_OMEGA_RAD_S])
    if tq > a[ACC_TORQUE_PEAK_NM]:
        a[ACC_TORQUE_PEAK_NM] = tq
    if pw > a[ACC_POWER_PEAK_W]:
        a[ACC_POWER_PEAK_W] = pw
    if fmag > a[ACC_FORCE_PEAK_N]:
        a[ACC_FORCE_PEAK_N] = fmag
        RMp = pose.R_M_S
        cacc.scalars[CA_FPEAK_MX] = RMp[0, 0] * fs0 + RMp[0, 1] * fs1 + RMp[0, 2] * fs2
        cacc.scalars[CA_FPEAK_MY] = RMp[1, 0] * fs0 + RMp[1, 1] * fs1 + RMp[1, 2] * fs2
        cacc.scalars[CA_FPEAK_MZ] = RMp[2, 0] * fs0 + RMp[2, 1] * fs1 + RMp[2, 2] * fs2
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
        tw[row, WRENCH_TOOL_PEAK, 0] = wF0
        tw[row, WRENCH_TOOL_PEAK, 1] = wF1
        tw[row, WRENCH_TOOL_PEAK, 2] = wF2
        tw[row, WRENCH_TOOL_PEAK, 3] = wM0
        tw[row, WRENCH_TOOL_PEAK, 4] = wM1
        tw[row, WRENCH_TOOL_PEAK, 5] = wM2
        tw[row, WRENCH_WORK_PEAK, 0] = fw0
        tw[row, WRENCH_WORK_PEAK, 1] = fw1
        tw[row, WRENCH_WORK_PEAK, 2] = fw2
        tw[row, WRENCH_WORK_PEAK, 3] = mw0
        tw[row, WRENCH_WORK_PEAK, 4] = mw1
        tw[row, WRENCH_WORK_PEAK, 5] = mw2
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
@njit(types.void(TOOL_T, TABLE_T, MEMORY_T, INTERVALS_T, REFILL_T, F1, I1, POSE_T, DEFLECTION_T, OCCUPANCY_T,
                 PACKING_T, AXES_T, STATE_T, ACCUM_T, CACCUM_T, OUT_T, EVENTS_T), **OPTS)
def cut_step(tool, table, mem, iv, refill, step, istep, pose, defl, occ, pack, axes, st, acc, cacc, out, ev):
    """One L0 step of one contact of a rotating tool (see the module doc and docs/models/cutting.md).

    Everything is evaluated at the engaged centroid of each element's swept arc (the step midpoint for
    elements fully inside an engaged interval), and the impulse is the force there times the engaged
    duration: the closed-form integral of the piecewise-linear force over the swept angle clipped to
    the intervals (exact for forces linear in h and h linear in the angle within a step)."""
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
    ox = pose.tip_S_m[0] - defl.u_work_S_m[0]
    oy = pose.tip_S_m[1] - defl.u_work_S_m[1]
    oz = pose.tip_S_m[2] - defl.u_work_S_m[2]
    cTx = g00 * ox + g10 * oy + g20 * oz
    cTy = g01 * ox + g11 * oy + g21 * oz
    cTz = g02 * ox + g12 * oy + g22 * oz
    # tip velocity relative to the (vibrating) material, in T
    wvx = pose.v_tip_S_m_s[0] - defl.v_work_S_m_s[0]
    wvy = pose.v_tip_S_m_s[1] - defl.v_work_S_m_s[1]
    wvz = pose.v_tip_S_m_s[2] - defl.v_work_S_m_s[2]
    vTx = g00 * wvx + g10 * wvy + g20 * wvz
    vTy = g01 * wvx + g11 * wvy + g21 * wvz
    vTz = g02 * wvx + g12 * wvy + g22 * wvz
    # work vibration velocity alone in T (its share of the force power goes to the deflection work)
    wkx = defl.v_work_S_m_s[0]
    wky = defl.v_work_S_m_s[1]
    wkz = defl.v_work_S_m_s[2]
    vWx = g00 * wkx + g10 * wky + g20 * wkz
    vWy = g01 * wkx + g11 * wky + g21 * wkz
    vWz = g02 * wkx + g12 * wky + g22 * wkz
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
    ts = st.tooth
    # station rows written last step are cleared (they are rewritten when engaged)
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
            ts[j, TS_CENTRE_WAS] = ts[j, TS_CENTRE_ON]
            ts[j, TS_CENTRE_ON] = 0.0
            ts[j, TS_STEP_DV] = 0.0
            ts[j, TS_STEP_B] = 0.0
            ts[j, TS_AVAIL] = 0.0
            for c in range(TS_S_PX, TS_S_NZ + 1):
                ts[j, c] = 0.0
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
    contact_any = False
    for s in range(nst):
        if iv.count[s] > 0:
            contact_any = True
            break
    for s in range(nst):
        k0 = tool.st_ptr[s]
        k1 = tool.st_ptr[s + 1]
        cnt = iv.count[s]
        if k1 <= k0 or (cnt <= 0 and st.st_written[s] == 0):
            continue
        # station deflection: T0 -> T (rotate by -theta_mid); the float shifts the tool along z
        ux = defl.u_st_T0_m[s, 0]
        uy = defl.u_st_T0_m[s, 1]
        uTx = cm * ux + sm * uy
        uTy = cm * uy - sm * ux
        uTz = defl.u_st_T0_m[s, 2] + u_f
        rx = defl.u_st_T0_m[s, 3]
        ry = defl.u_st_T0_m[s, 4]
        rTx = cm * rx + sm * ry
        rTy = cm * ry - sm * rx
        rTz = defl.u_st_T0_m[s, 5]
        vx = defl.v_st_T0_m_s[s, 0]
        vy = defl.v_st_T0_m_s[s, 1]
        vuTx = cm * vx + sm * vy
        vuTy = cm * vy - sm * vx
        vuTz = defl.v_st_T0_m_s[s, 2]
        vx = defl.v_st_T0_m_s[s, 3]
        vy = defl.v_st_T0_m_s[s, 4]
        vrTx = cm * vx + sm * vy
        vrTy = cm * vy - sm * vx
        vrTz = defl.v_st_T0_m_s[s, 5]
        zst = tool.st_z_m[s]
        sFx = 0.0
        sFy = 0.0
        sFz = 0.0
        sMx = 0.0
        sMy = 0.0
        sMz = 0.0
        any_in = False
        centre = tool.st_centre[s] != 0
        for kk in range(k0, k1):
            e = tool.st_elem[kk]
            j = tool.tooth[e]
            flags = st.flags[e]
            if commit and cnt > 0:
                ts[j, TS_AVAIL] += 1.0
            a = theta + tool.alpha0_rad[e]
            if a >= TWO_PI:
                a -= TWO_PI
            if floor_on and contact_any and abs(tool.n_T[e, 2]) > 0.7:
                # axial-normal edges (end, corner bottom, wiper) leave the floor where they pass the trailing side
                d_ang = a_trail - a
                if d_ang > math.pi:
                    d_ang -= TWO_PI
                elif d_ang < -math.pi:
                    d_ang += TWO_PI
                if abs(d_ang) <= half:
                    fpx = tool.p_T_m[e, 0]
                    fpy = tool.p_T_m[e, 1]
                    fpz = tool.p_T_m[e, 2]
                    fpr = fpz - zst
                    fux = uTx + (rTy * fpr - rTz * fpy)
                    fuy = uTy + (rTz * fpx - rTx * fpr)
                    fuz = uTz + (rTx * fpy - rTy * fpx)
                    qx = cTx + fpx + fux
                    qy = cTy + fpy + fuy
                    qz = cTz + fpz + fuz
                    fnx = tool.n_T[e, 0]
                    fny = tool.n_T[e, 1]
                    fnz = tool.n_T[e, 2]
                    sx_ = g00 * qx + g01 * qy + g02 * qz
                    sy_ = g10 * qx + g11 * qy + g12 * qz
                    sz_ = g20 * qx + g21 * qy + g22 * qz
                    nx_ = -(g00 * fnx + g01 * fny + g02 * fnz)
                    ny_ = -(g10 * fnx + g11 * fny + g12 * fnz)
                    nz_ = -(g20 * fnx + g21 * fny + g22 * fnz)
                    ux_ = g00 * fux + g01 * fuy + g02 * fuz - defl.u_work_S_m[0]
                    uy_ = g10 * fux + g11 * fuy + g12 * fuz - defl.u_work_S_m[1]
                    uz_ = g20 * fux + g21 * fuy + g22 * fuz - defl.u_work_S_m[2]
                    tick_f = float(tick0 + l0 // 2)
                    put_surface(ev.surface.rec, ev.surface.n, tick_f, float(j), float(tool.elem[e]), sx_, sy_, sz_,
                                nx_, ny_, nz_, 0.0, tool.edge_radius_m[e], tool.VB_m[e], ux_, uy_, uz_, slot, cacc)
            if cnt > 0:
                frac, s_c, s_lo, s_hi = gate(iv, s, a, half, inv_dal)
            else:
                frac = 0.0
                s_c = 0.5
                s_lo = 0.0
                s_hi = 0.0
            if frac <= 0.0:
                if (flags & FL_IN_CUT) != 0:
                    element_exit(tool, table, mem, st, ev, cacc, e, t_start, st.h_last_m[e], a - 0.5 * dal, tick0,
                                 commit, ratio, slot)
                elif commit and st.pend[e, PD_BIN] >= 0.0:
                    if tool.write_row[e] >= 0:
                        memory_commit(mem, tool.write_row[e], int(st.pend[e, PD_BIN]), st.pend[e, PD_X],
                                      st.pend[e, PD_Y], st.pend[e, PD_Z], st.pend[e, PD_H],
                                      table.h_write_m[tool.row[e]], ratio, st.pend_tick[e])
                    st.pend[e, PD_BIN] = -1.0
                if commit:
                    st.tick_last[e] = -1
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
            px = tool.p_T_m[e, 0]
            py = tool.p_T_m[e, 1]
            pz = tool.p_T_m[e, 2]
            pzr = pz - zst
            dTx = ccx + px + ucx + (rcy * pzr - rTz * py)
            dTy = ccy + py + ucy + (rTz * px - rcx * pzr)
            dTz = ccz + pz + uTz + (rcx * py - rcy * px)
            nx = tool.n_T[e, 0]
            ny = tool.n_T[e, 1]
            nz = tool.n_T[e, 2]
            # ---------------------------------------------------- memory read at the centroid angle
            xb = ac * inv_bin - 0.5
            fb = np.floor(xb)
            w = xb - fb
            u_b = w
            b0 = int(fb)
            if b0 < 0:
                b0 += nb
            bb = b0 + 1
            if bb >= nb:
                bb = 0
            if frac < 1.0 or near_edge(iv, s, ac, bin_w):
                # next to an interval boundary one bin of the stencil can lie outside the material, where no tooth
                # renews the memory: read the inside bin only
                in0 = in_intervals(iv, s, (b0 + 0.5) * bin_w)
                in1 = in_intervals(iv, s, (b0 + 1.5) * bin_w)
                if in0 and not in1:
                    w = 0.0
                elif in1 and not in0:
                    w = 1.0
            rs = tool.read_row[e]
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
            elif not (bin_fresh(mem, rs, b0, tick_end, stale_ticks, stale_before)
                      and bin_fresh(mem, rs, bb, tick_end, stale_ticks, stale_before)):
                ok0 = bin_fresh(mem, rs, b0, tick_end, stale_ticks, stale_before)
                ok1 = bin_fresh(mem, rs, bb, tick_end, stale_ticks, stale_before)
                psx = e00 * dTx + e01 * dTy + g02 * dTz
                psy = e10 * dTx + e11 * dTy + g12 * dTz
                psz = e20 * dTx + e21 * dTy + g22 * dTz
                nsx = e00 * nx + e01 * ny + g02 * nz
                nsy = e10 * nx + e11 * ny + g12 * nz
                nsz = e20 * nx + e21 * ny + g22 * nz
                if not ok0:
                    ok0 = refill_bin(mem, refill, rs, b0, psx, psy, psz, nsx, nsy, nsz, tick_end, commit, ev, cacc)
                if not ok1:
                    ok1 = refill_bin(mem, refill, rs, bb, psx, psy, psz, nsx, nsy, nsz, tick_end, commit, ev, cacc)
                if ok0:
                    m0x = mem.point_S_m[rs, b0, 0]
                    m0y = mem.point_S_m[rs, b0, 1]
                    m0z = mem.point_S_m[rs, b0, 2]
                else:
                    m0x = psx
                    m0y = psy
                    m0z = psz
                if ok1:
                    m1x = mem.point_S_m[rs, bb, 0]
                    m1y = mem.point_S_m[rs, bb, 1]
                    m1z = mem.point_S_m[rs, bb, 2]
                else:
                    m1x = psx
                    m1y = psy
                    m1z = psz
            else:
                m0x = mem.point_S_m[rs, b0, 0]
                m0y = mem.point_S_m[rs, b0, 1]
                m0z = mem.point_S_m[rs, b0, 2]
                m1x = mem.point_S_m[rs, bb, 0]
                m1y = mem.point_S_m[rs, bb, 1]
                m1z = mem.point_S_m[rs, bb, 2]
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
                h -= sag_c * ((1.0 - w) * u_b * u_b + w * (1.0 - u_b) * (1.0 - u_b)) * (nx * px + ny * py)
            slope = h - st.h_last_m[e] if st.tick_last[e] == tick0 else 0.0
            # element velocity relative to the material (T): tip + frame rotation + deflection rates
            vex = bvx + (ofy * pz - OTz * py) + (wry * pzr - vrTz * py)
            vey = bvy + (OTz * px - ofx * pz) + (vrTz * px - wrx * pzr)
            vez = vTz + vuTz + (ofx * py - ofy * px) + (wrx * py - wry * px)
            vn = vex * nx + vey * ny + vez * nz
            Vc = vex * tool.c_T[e, 0] + vey * tool.c_T[e, 1] + vez * tool.c_T[e, 2]
            # ---------------------------------------------------- memory crossings (delayed by one bin)
            if commit and bps != 0.0:
                row_m = tool.row[e]
                hmin_m = table.h_write_m[row_m]
                xs = xb - (s_c - 0.5) * bps - 0.5 * bps
                xe = xs + bps
                if bps > 0.0:
                    cA = int(np.floor(xs)) + 1
                    cB = int(np.floor(xe))
                    step_c = 1
                else:
                    cA = int(np.ceil(xs)) - 1
                    cB = int(np.ceil(xe))
                    step_c = -1
                ws = tool.write_row[e]
                if (step_c > 0 and cB >= cA) or (step_c < 0 and cB <= cA):
                    c = cA
                    while True:
                        sx = (c - xs) / bps
                        if s_lo <= sx <= s_hi:
                            hx = h + (sx - s_c) * slope
                            pb = int(st.pend[e, PD_BIN])
                            if pb >= 0:
                                if ws >= 0:
                                    memory_commit(mem, ws, pb, st.pend[e, PD_X], st.pend[e, PD_Y],
                                                  st.pend[e, PD_Z], st.pend[e, PD_H], hmin_m, ratio,
                                                  st.pend_tick[e])
                                st.pend[e, PD_BIN] = -1.0
                            bi = c % nb
                            if bi < 0:
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
                            have = False
                            rx = 0.0
                            ry = 0.0
                            rz = 0.0
                            rtick = tick_end
                            if ws >= 0 and rs != ws:
                                if rs >= 0:
                                    if mem.valid[rs, bi] != 0:
                                        have = True
                                        rx = mem.point_S_m[rs, bi, 0]
                                        ry = mem.point_S_m[rs, bi, 1]
                                        rz = mem.point_S_m[rs, bi, 2]
                                        rtick = mem.tick[rs, bi]
                                elif refill.valid[s, bi] != 0:
                                    nsx = e00 * nx + e01 * ny + g02 * nz
                                    nsy = e10 * nx + e11 * ny + g12 * nz
                                    nsz = e20 * nx + e21 * ny + g22 * nz
                                    hf = field_depth(refill, s, bi, ex, ey, ez, nsx, nsy, nsz, tick_end, False, ev,
                                                     cacc)
                                    have = True
                                    rx = ex - hf * nsx
                                    ry = ey - hf * nsy
                                    rz = ez - hf * nsz
                            if ws < 0:
                                pass
                            elif have:
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
                                st.pend[e, PD_BIN] = float(bi)
                                st.pend[e, PD_X] = rx + wr * (ex - rx)
                                st.pend[e, PD_Y] = ry + wr * (ey - ry)
                                st.pend[e, PD_Z] = rz + wr * (ez - rz)
                                st.pend[e, PD_H] = 1e300
                                st.pend_tick[e] = tick0 + int(np.floor(sx * l0 + 0.5)) if wr > 0.0 else rtick
                            elif hx >= hmin_m:
                                st.pend[e, PD_BIN] = float(bi)
                                st.pend[e, PD_X] = ex
                                st.pend[e, PD_Y] = ey
                                st.pend[e, PD_Z] = ez
                                st.pend[e, PD_H] = hx
                                st.pend_tick[e] = tick0 + int(np.floor(sx * l0 + 0.5))
                        if c == cB:
                            break
                        c += step_c
            if h <= 0.0:
                if (flags & FL_IN_CUT) != 0:
                    q = st.h_last_m[e] / (st.h_last_m[e] - h) if st.h_last_m[e] > h else 0.0
                    element_exit(tool, table, mem, st, ev, cacc, e, t_start + (q - 0.5) * dt, 0.0,
                                 a + (q - 1.0) * dal, tick0 + int(np.floor((q - 0.5) * l0 + 0.5)), commit, ratio,
                                 slot)
                if commit:
                    st.h_last_m[e] = h
                    st.tick_last[e] = tick_end if frac >= 1.0 else -1
                continue
            # ---------------------------------------------------- forces
            row = tool.row[e]
            hm = table.h_min_m[row]
            if h < hm:
                sg = int(h * table.inv_dlo[row])
                if sg >= nlo:
                    sg = nlo - 1
            else:
                sg = nlo + int((h - hm) * table.inv_dhi[row])
                if sg >= nseg:
                    sg = nseg - 1
                    cacc.scalars[CA_TABLE_OVER] += 1.0
            db = tool.db_m[e]
            dS = tool.dS_m[e]
            ft0 = (coef[row, sg, 0, COL_FT] + coef[row, sg, 1, COL_FT] * h) * db + \
                (coef[row, sg, 0, COL_ET] + coef[row, sg, 1, COL_ET] * h) * dS
            fr0 = (coef[row, sg, 0, COL_FR] + coef[row, sg, 1, COL_FR] * h) * db + \
                (coef[row, sg, 0, COL_ER] + coef[row, sg, 1, COL_ER] * h) * dS
            fa0 = (coef[row, sg, 0, COL_FA] + coef[row, sg, 1, COL_FA] * h) * db + \
                (coef[row, sg, 0, COL_EA] + coef[row, sg, 1, COL_EA] * h) * dS
            fft = tool.flank_N[e, 0]
            ffr = tool.flank_N[e, 1]
            fpd = 0.0
            if vn > 0.0 and Vc > 0.0 and tool.pd[e, 0] > 0.0:
                fpd = tool.pd[e, 0] * flank_interference_area(vn / Vc, tool.pd[e, 1], tool.pd[e, 2], pd_max)
            mu_pd = tool.pd[e, 3]
            ft = ft0 + fft + mu_pd * fpd
            fr = fr0 + ffr + fpd
            fa = fa0
            Fx = ft * tool.dir_T[e, 0, 0] + fr * tool.dir_T[e, 1, 0] + fa * tool.dir_T[e, 2, 0]
            Fy = ft * tool.dir_T[e, 0, 1] + fr * tool.dir_T[e, 1, 1] + fa * tool.dir_T[e, 2, 1]
            Fz = ft * tool.dir_T[e, 0, 2] + fr * tool.dir_T[e, 1, 2] + fa * tool.dir_T[e, 2, 2]
            Mx = ft * tool.arm_T[e, 0, 0] + fr * tool.arm_T[e, 1, 0] + fa * tool.arm_T[e, 2, 0]
            My = ft * tool.arm_T[e, 0, 1] + fr * tool.arm_T[e, 1, 1] + fa * tool.arm_T[e, 2, 1]
            Mz = ft * tool.arm_T[e, 0, 2] + fr * tool.arm_T[e, 1, 2] + fa * tool.arm_T[e, 2, 2]
            dtw = frac * dt
            # power dissipated by the element (-F . v), split by zone below
            va = vex * tool.dir_T[e, 2, 0] + vey * tool.dir_T[e, 2, 1] + vez * tool.dir_T[e, 2, 2]
            p_tab = ft0 * Vc + fr0 * vn - fa0 * va
            p_fl = fft * Vc + ffr * vn + mu_pd * fpd * Vc
            p_pd = fpd * vn
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
                dft = coef[row, sg, 1, COL_FT] * db + coef[row, sg, 1, COL_ET] * dS
                dfr = coef[row, sg, 1, COL_FR] * db + coef[row, sg, 1, COL_ER] * dS
                dfa = coef[row, sg, 1, COL_FA] * db + coef[row, sg, 1, COL_EA] * dS
                Kz += frac * nz * (dft * tool.dir_T[e, 0, 2] + dfr * tool.dir_T[e, 1, 2] + dfa * tool.dir_T[e, 2, 2])
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
            forming = tool.forming[e] != 0
            dV = h * db * tool.r_xy_m[e] * abs_dal * frac
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
            ea = st.elem_acc
            ea[e, ELEM_IN_CUT_S] += dtw
            ea[e, ELEM_RAKE_SLIDING_M] += rc * Vc * dtw
            ea[e, ELEM_FLANK_SLIDING_M] += Vc * dtw
            ea[e, ELEM_INT_SIGMA_V_RAKE] += (coef[row, sg, 0, COL_SIG_RAKE] + coef[row, sg, 1, COL_SIG_RAKE] * h) \
                * rc * Vc * dtw
            ea[e, ELEM_INT_SIGMA_V_FLANK] += tool.sig_flank_Pa[e] * Vc * dtw
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
                    ts[j, TS_ENTRY_TICK] = float(tick0 + int(np.floor(s_en * l0 + 0.5)))
                    ts[j, TS_ENTRY_ELEM] = float(tool.elem[e])
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
                    tick_x = float(tick0 + int(np.floor(s_en * l0 + 0.5)))
                    toff = (s_en - s_c) * dt
                    _surface_event(ev, cacc, tool, defl, e, j, tick_x, dTx, dTy, dTz, toff, vex, vey, vez, h_en, nx,
                                   ny, nz, ucx, ucy, uTz, rcx, rcy, rTz, pzr, px, py, e00, e01, g02, e10, e11, g12,
                                   e20, e21, g22, slot)
            ts[j, TS_NOW] += 1.0
            ts[j, TS_FX] += Fx * frac
            ts[j, TS_FY] += Fy * frac
            ts[j, TS_FZ] += Fz * frac
            ts[j, TS_STEP_B] += db
            if centre:
                ts[j, TS_CENTRE_ON] = 1.0
            # ---------------------------------------------------- tooth-pass accumulation (chip emission, deposits)
            tch = coef[row, sg, 0, COL_TCHIP] + coef[row, sg, 1, COL_TCHIP] * h
            ws = coef[row, sg, 0, COL_WS] + coef[row, sg, 1, COL_WS] * h
            wf = coef[row, sg, 0, COL_WF] + coef[row, sg, 1, COL_WF] * h
            if not forming:
                ts[j, TS_VOL] += dV
                ts[j, TS_HV] += h * dV
                ts[j, TS_VV] += Vc * dV
                ts[j, TS_RAKEV] += tool.rake_rad[e] * dV
                ts[j, TS_INCLV] += tool.incl_rad[e] * dV
                ts[j, TS_TV] += tch * dV
                ts[j, TS_RV] += tool.r_xy_m[e] * dV
                if dV > ts[j, TS_DOM_VOL]:
                    ts[j, TS_DOM_VOL] = dV
                    ts[j, TS_DOM_KIND] = float(tool.kind[e])
                    ts[j, TS_DOM_ROW] = float(row)
                ts[j, TS_STEP_DV] += dV
                ts[j, TS_S_PX] += dTx * dV
                ts[j, TS_S_PY] += dTy * dV
                ts[j, TS_S_PZ] += dTz * dV
                vch = rc * Vc
                ts[j, TS_S_VX] += (vex + vch * tool.chip_T[e, 0]) * dV
                ts[j, TS_S_VY] += (vey + vch * tool.chip_T[e, 1]) * dV
                ts[j, TS_S_VZ] += (vez + vch * tool.chip_T[e, 2]) * dV
                ts[j, TS_S_NX] += nx * dV
                ts[j, TS_S_NY] += ny * dV
                ts[j, TS_S_NZ] += nz * dV
            ts[j, TS_WORK_J] += et * ps * ws + (et * (1.0 - ps - pr) + p_fl * dtw) * wf
            # ---------------------------------------------------- surface events (envelope: v.n from + to -)
            if (flags & FL_VN_POS) != 0 and vn <= 0.0 and (flags & FL_EVENT_DONE) == 0:
                vl = st.vn_last_m_s[e]
                q = vl / (vl - vn) if vl > vn else 1.0
                hq = st.h_last_m[e] + q * (h - st.h_last_m[e])
                tick_x = float(tick0 + int(np.floor((q - 0.5) * l0 + 0.5)))
                toff = (q - 1.0) * dt
                _surface_event(ev, cacc, tool, defl, e, j, tick_x, dTx, dTy, dTz, toff, vex, vey, vez, hq, nx, ny, nz,
                               ucx, ucy, uTz, rcx, rcy, rTz, pzr, px, py, e00, e01, g02, e10, e11, g12, e20, e21, g22,
                               slot)
                flags |= FL_EVENT_DONE
            if vn > 0.0:
                flags |= FL_VN_POS
            else:
                flags &= ~FL_VN_POS
            st.vn_last_m_s[e] = vn
            st.h_last_m[e] = h
            st.tick_last[e] = tick_end if frac >= 1.0 else -1
            # an arc clipped at its end: the element leaves the interval within this step
            if s_hi < 1.0 and frac < 1.0 and s_lo <= 0.0:
                if abs(vn) <= vt_tol and (flags & FL_EVENT_DONE) == 0:
                    # exit through the outer boundary where the path is tangent (down-milling side wall)
                    hx = h + (s_hi - s_c) * slope
                    if hx < 0.0:
                        hx = 0.0
                    tick_x = float(tick0 + int(np.floor(s_hi * l0 + 0.5)))
                    toff = (s_hi - s_c) * dt
                    _surface_event(ev, cacc, tool, defl, e, j, tick_x, dTx, dTy, dTz, toff, vex, vey, vez, hx, nx, ny,
                                   nz, ucx, ucy, uTz, rcx, rcy, rTz, pzr, px, py, e00, e01, g02, e10, e11, g12, e20,
                                   e21, g22, slot)
                st.flags[e] = flags
                element_exit(tool, table, mem, st, ev, cacc, e, t_start + s_hi * dt, h, a + (s_hi - 0.5) * dal,
                             tick0 + int(np.floor(s_hi * l0 + 0.5)), commit, ratio, slot)
                st.tick_last[e] = -1
            else:
                st.flags[e] = flags
        # ---------------------------------------------------- station totals
        if any_in:
            inv = 1.0 / dt
            out.station_T0[s, 0] = (cm * sFx - sm * sFy) * inv
            out.station_T0[s, 1] = (sm * sFx + cm * sFy) * inv
            out.station_T0[s, 2] = sFz * inv
            out.station_T0[s, 3] = (cm * sMx - sm * sMy) * inv
            out.station_T0[s, 4] = (sm * sMx + cm * sMy) * inv
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
    # rigid-body work of the cutting forces: shaft (spindle rotation) and feed (axes, incl. rotary)
    shaft = -omega * M0z
    feed = -(vTx * FTx + vTy * FTy + vTz * FTz + (OTx * MTx + OTy * MTy + (OTz - omega) * MTz))
    feed -= FTx * vWx + FTy * vWy + FTz * vWz
    # recut impulses (once per tooth period) and chip packing, both applied to the tool at once
    if commit and occ.cell_m[0] > 0.0 and tool.scal[TC_N_TEETH] > 0.0:
        st.misc[MS_RECUT_ANGLE] += abs_dal
        if st.misc[MS_RECUT_ANGLE] >= TWO_PI / tool.scal[TC_N_TEETH]:
            st.misc[MS_RECUT_ANGLE] = 0.0
            rj = recut_scan(tool, table, occ, pose, step, istep, cacc, acc, ev, theta)
            F0x += rj[0]
            F0y += rj[1]
            F0z += rj[2]
            M0x += rj[3]
            M0y += rj[4]
            M0z += rj[5]
            shaft += -omega * rj[5]
            rfx = R[0, 0] * rj[0] + R[0, 1] * rj[1] + R[0, 2] * rj[2]
            rfy = R[1, 0] * rj[0] + R[1, 1] * rj[1] + R[1, 2] * rj[2]
            rfz = R[2, 0] * rj[0] + R[2, 1] * rj[1] + R[2, 2] * rj[2]
            f_rec = -(rfx * pose.v_tip_S_m_s[0] + rfy * pose.v_tip_S_m_s[1] + rfz * pose.v_tip_S_m_s[2])
            feed += f_rec
            cacc.scalars[CA_RECUT_J] += f_rec
    sgn = 1.0 if omega > 0.0 else (-1.0 if omega < 0.0 else 0.0)
    tp = pack.torque_Nm[0]
    if tp != 0.0 or pack.force_T0_N[0] != 0.0 or pack.force_T0_N[1] != 0.0 or pack.force_T0_N[2] != 0.0:
        F0x += pack.force_T0_N[0] * dt
        F0y += pack.force_T0_N[1] * dt
        F0z += pack.force_T0_N[2] * dt
        M0z += -sgn * tp * dt
        w_pack = abs(omega) * tp * dt
        pfx = R[0, 0] * pack.force_T0_N[0] + R[0, 1] * pack.force_T0_N[1] + R[0, 2] * pack.force_T0_N[2]
        pfy = R[1, 0] * pack.force_T0_N[0] + R[1, 1] * pack.force_T0_N[1] + R[1, 2] * pack.force_T0_N[2]
        pfz = R[2, 0] * pack.force_T0_N[0] + R[2, 1] * pack.force_T0_N[1] + R[2, 2] * pack.force_T0_N[2]
        f_pack = -(pfx * pose.v_tip_S_m_s[0] + pfy * pose.v_tip_S_m_s[1] + pfz * pose.v_tip_S_m_s[2]) * dt
        shaft += abs(omega) * tp * dt
        feed += f_pack
        if commit:
            cacc.scalars[CA_PACKING_J] += w_pack + f_pack
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
    finish_step(pose, step, istep, axes, acc, cacc, out, commit, F0x, F0y, F0z, M0x, M0y, M0z, shaft, feed,
                M0z, power, n_in)
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
    (the chip thickness when (ur, uz) = -n): march in half-cells of the z spacing, linear root between
    the last inside and the first outside sample. Returns (h, 1 if the ray reached cap)."""
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
            return s0 + (s1 - s0) * g0 / (g0 - g1), 0.0
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
                Ae = h * tool.db_m[e]
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
            sg = nlo + int((h_eq - hm) * table.inv_dhi[row])
            if sg >= nseg:
                sg = nseg - 1
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
                sg = nlo + int((h - hm) * table.inv_dhi[row])
                if sg >= nseg:
                    sg = nseg - 1
                    cacc.scalars[CA_TABLE_OVER] += 1.0
            db = tool.db_m[e]
            dS = tool.dS_m[e]
            fft = tool.flank_N[e, 0]
            ffr = tool.flank_N[e, 1]
            fpd = 0.0
            if vn > 0.0 and Vc > 0.0 and tool.pd[e, 0] > 0.0:
                fpd = tool.pd[e, 0] * flank_interference_area(vn / Vc, tool.pd[e, 1], tool.pd[e, 2], pd_max)
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
            dV = h * db * r_e * abs_dal
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
            ea[e, ELEM_RAKE_SLIDING_M] += rc * Vc * dt
            ea[e, ELEM_FLANK_SLIDING_M] += Vc * dt
            ea[e, ELEM_INT_SIGMA_V_RAKE] += (coef[row, sg, 0, COL_SIG_RAKE] + coef[row, sg, 1, COL_SIG_RAKE] * h) \
                * rc * Vc * dt
            ea[e, ELEM_INT_SIGMA_V_FLANK] += tool.sig_flank_Pa[e] * Vc * dt
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
            ts[j, TS_NOW] += 1.0
            ts[j, TS_FX] += Fx
            ts[j, TS_FY] += Fy
            ts[j, TS_FZ] += Fz
            ts[j, TS_STEP_B] += db
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
                torque_work * dt, power, n_in)
