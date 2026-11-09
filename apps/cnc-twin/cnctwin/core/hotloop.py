"""The compiled hot loop: one macro step of every L1 tick and L0 step in one Numba call (architecture 5.2-5.5,
docs/models/hotloop.md).

:func:`run_macro` advances ticks ``k_from .. macro_ipo - 1`` of the macro step:

per L1 tick k
    motion's setpoint for the tick, feed drives with the previous tick's cutting load per axis, spindle loops
    with the previous tick's torque, kinematics (motion_tick); the probe sweep of G31 moves (removal's
    sweep_contact); per contact the broad phase (removal's l1_contact) and the collision check of the
    non-cutting envelope (collide_envelope);
per contact in contact, the L0 steps of the tick at the contact's stride
    pose by axis-space interpolation and FK (S_T0, S in M), spindle angle from the spindle servo state; the
    dynamics predictor (modal_begin: unbalance, base excitation, predicted deflections); the element
    positions (architecture 4.1: chain pose with error transforms and spindle growth from motion, tool and
    holder growth, pull-out, slip, wear, runout and tilt in cutting's effective tool, deflection here, the
    structure displacement d(X) in the workpiece deflection); refill planes of the stale memory bins from
    the field; cutting forces with the surface memory and engaged intervals (cut_step / turn_step: analytic
    chip volume, surface events, work-heat deposits, shocks, recut, chip emission rows); the dynamics
    corrector (modal_correct); tooth-period events: each slab of each tooth records its axis point
    X_e + u_rel when it passes its generation angle, and the tooth's pass (engaged-interval refresh and
    subtraction per slab, removal's tooth_pass) runs when its last slab has passed; turning: the polar cut
    every L0 step and the polar view update one bin behind the tool; taps: the helical sweep per step; the
    recorder ring row;
per contact out of contact
    exact free propagation of the dynamics over the tick (unbalance, base excitation);
then
    dynamics' per-tick record, the tick's mean cutting load per machine axis and spindle torque (the next
    tick's disturbances), out.in_cut; return at motion's sync codes (the controller runs) or at a split.

Splits (docs/models/hotloop.md "Splits"): before every L0 step the loop checks that every event buffer has room
for the rows one step can write; if not it returns HOT_SPLIT (SPLIT_DRAIN) inside the tick, the wrapper moves
the rows out and calls again, and the loop resumes at that L0 step. A removal pool or polar overflow returns
SPLIT_POOL / SPLIT_POLAR; the wrapper grows the field and the refused pass or cut is repeated at the resume.
No row is lost and no state is rolled back, so a run is bit-identical whatever the buffer capacities.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, types

from ..cutting import abi as CA
from ..cutting.kernels import cut_step, turn_step
from ..dynamics.abi import ACT_HI, ACT_LO, ACT_LOAD_HI, ACT_LOAD_LO
from ..dynamics.kernels import modal_begin, modal_correct, modal_free, record_tick, refresh_work_shapes, \
    relative_axis_acceleration, ring_write
from ..motion.abi import SYNC_NONE, SYNC_PROBE_TRIGGER, SYNC_QUEUE_EMPTY
from ..motion.kernels import motion_tick, replan_before
from ..removal import abi as RA
from ..removal.k_collide import collide_envelope
from ..removal.k_drive import l1_contact, tooth_pass
from ..removal.k_gate import engaged_intervals
from ..removal.k_helix import helix_subtract
from ..removal.k_polar import polar_cut
from ..removal.k_query import broad_phase, sweep_contact
from . import hot_abi as H
from .hot_abi import (BUF_DEPOSIT, BUF_PASS, BUF_RECUT, BUF_REFILL, BUF_REMOVAL, BUF_SHOCK, BUF_SURFACE, CF_COLLISION_TOL,
                      CF_FEED_FRAC, CF_HAND, CF_LATTICE, CF_MOUNT_X, CF_MOUNT_Y, CF_MOUNT_Z, CF_POLAR_SX, CF_POLAR_SZ,
                      CF_R_CUT, CF_REACH, CF_WORK_SIGN,
                      CI_ACTIVE, CI_DEFL_SHARED, CI_DYN, CI_KIND, CI_L0_TICKS, CI_N_TEETH, CI_NS_MAX, CI_PATH, CI_PROBE,
                      CI_ST_MARGIN,
                      CI_RING_DYN, CI_RING_FORCE, CI_RING_TORQUE, CI_ROTATING, CI_STALE_BEFORE, CI_TOOL_SPINDLE,
                      CI_WORK, CI_WORK_SPINDLE, CNT_BROAD, CNT_FREE, CNT_HELIX_CALLS, CNT_L0, CNT_PASSES,
                      CNT_POLAR_CUTS, CNT_PROBE_SWEEPS, CNT_REFILLS, CNT_SPLITS, CNT_TICKS, CONTACT_T, CONTACTS_T,
                      CS_ENGAGED_TICK, CS_FIRST, CS_FREE_TICKS, CS_IN_CONTACT, CS_L0_STEPS, CS_LAST_RAPID,
                      CS_OVERFLOW_TOOTH, CS_PASSES, CS_PB_PENDING, CS_REFILL_MISS, CS_REFILLS, CS_SUBTRACT_CALLS,
                      CS_FORCED, CS_GATINGS, CSF_BETA, CSF_GATE_ANGLE, CSF_THETA_PREV, GF_PROBE_TOL, GF_TICK_S, GI_HAS_DISP, GI_IPO_TICKS, GI_N_CONTACTS,
                      GI_PROBE_AROUND, GI_PROBE_RINGS, GI_RESUME, GI_RESUME_C, GI_RESUME_K, GI_RESUME_S,
                      GI_ROWS_DEPOSIT, GI_ROWS_PASS, GI_ROWS_RECUT, GI_ROWS_REFILL, GI_ROWS_REMOVAL,
                      GI_ROWS_SHOCK, GI_ROWS_SURFACE, GI_TICK0, GLOBAL_T, HOT_SPLIT, MAX_CONTACTS, MOTION_T,
                      REMOVAL_T, RES_CODE, RES_CONTACT, RES_DETAIL, RES_PATH, RES_REASON, RES_TICK, RING_T,
                      SPLIT_DRAIN, SPLIT_POLAR, SPLIT_POOL)
from .hot_abi import N_TW, TW_LO, TW_N_DONE, TW_N_ENG, TW_WIDTH
from .hot_kernels import (FR_R, FR_TIP, WK_A0, WK_A1, WK_DISP, WK_DV, WK_FEED, WK_MID, WK_PASS, WK_PFX, WK_PFY, WK_PROBE,
                          WK_PSLACK, WK_TIPC, WK_TSIG, WK_W0, WK_X0, WK_XP, WK_Y0, clear_planes, displacement_at,
                          fk_frame, hermite, interp_axes, polar_max_radius, polar_view_update, refill_planes,
                          set_step_pose, slab_point)
from .recorder import ring_begin

OPTS = dict(cache=True, nogil=True, error_model="numpy")
# every kernel of the loop allocates nothing (scratch lives in HotContact.work) and is compiled without reference
# counting: the loop body takes members of large NamedTuples, which otherwise costs an atomic incref / decref per
# member array (docs/performance.md "Hot loop composition"; removal.nbt RAW)
RAW = dict(OPTS, _nrt=False)
f8 = types.float64
i8 = types.int64
i4 = types.int32
b1 = types.boolean
void = types.void
TWO_PI = 2.0 * math.pi

GI_RESUME_CODE = 21         # motion's sync code of the tick a split interrupted (HotGlobal.ipar; hot_abi N_GI = 24)
WIN_EPS = 1e-8              # rad: margin of the tooth-window test (the exact per-slab test decides inside it)


# =============================================================================== small helpers
@njit(i8(CONTACT_T, REMOVAL_T, GLOBAL_T), **RAW)
def headroom(ct, rm, g):
    """-1 when every event buffer of the contact (and removal's) has room for one L0 step, else the BUF_* index of
    the first that has not (docs/models/hotloop.md "Splits")."""
    ev = ct.cut.ev
    ip = g.ipar
    if ev.shocks.rec.shape[0] - ev.shocks.n[0] < ip[GI_ROWS_SHOCK]:
        return BUF_SHOCK
    if ev.surface.rec.shape[0] - ev.surface.n[0] < ip[GI_ROWS_SURFACE]:
        return BUF_SURFACE
    if ev.deposits.rec.shape[0] - ev.deposits.n[0] < ip[GI_ROWS_DEPOSIT]:
        return BUF_DEPOSIT
    if ev.recut.rec.shape[0] - ev.recut.n[0] < ip[GI_ROWS_RECUT]:
        return BUF_RECUT
    if ev.passes.rec.shape[0] - ev.passes.n[0] < ip[GI_ROWS_PASS]:
        return BUF_PASS
    if ev.refill.rec.shape[0] - ev.refill.n[0] < ip[GI_ROWS_REFILL]:
        return BUF_REFILL
    if rm.events.rec.shape[0] - rm.events.n[0] < ip[GI_ROWS_REMOVAL]:
        return BUF_REMOVAL
    return -1


@njit(void(GLOBAL_T, i8, i8, i8, i8, i8, i8), **RAW)
def _split(g, k, c, s, reason, detail, code):
    g.ipar[GI_RESUME] = 1
    g.ipar[GI_RESUME_K] = k
    g.ipar[GI_RESUME_C] = c
    g.ipar[GI_RESUME_S] = s
    g.ipar[GI_RESUME_CODE] = code
    r = g.result
    r[RES_CODE] = HOT_SPLIT
    r[RES_TICK] = k
    r[RES_CONTACT] = c
    r[RES_REASON] = reason
    r[RES_DETAIL] = detail
    g.count[CNT_SPLITS] += 1


@njit(f8(f8, f8), **RAW)
def _unwrap_near(a, ref):
    """a + 2 pi n closest to ref."""
    return a + TWO_PI * math.floor((ref - a) / TWO_PI + 0.5)


# =============================================================================== tooth passes (rotating tools)
@njit(i8(REMOVAL_T, CONTACT_T, GLOBAL_T, i8, i8, f8, f8, f8, b1), **RAW)
def do_pass(rm, ct, g, j, tick, feed_x, feed_y, slack, rapid):
    """Removal's tooth_pass for tooth j of the contact: engaged intervals of every station refreshed against the
    field (the warm start is cutting's EngagedIntervals), contact geometry, engaged-sector box and the
    subtraction of the tooth's envelope per slab at the recorded axis points. The pose of the pass is
    HotContact.work[WK_PASS:WK_PASS + 12] (tip, x0, y0, w in S). Returns tooth_pass's status (1: pool overflow)."""
    p = ct.ipar[CI_PATH]
    w_ = ct.ipar[CI_WORK]
    f = rm.fields[w_]
    prof = rm.cut[p]
    slabs = rm.slabs[p]
    racc = rm.accum[p]
    tgt = rm.targets[w_]
    wk = ct.work
    tip = wk[WK_PASS:WK_PASS + 3]
    x0 = wk[WK_PASS + 3:WK_PASS + 6]
    y0 = wk[WK_PASS + 6:WK_PASS + 9]
    w = wk[WK_PASS + 9:WK_PASS + 12]
    spec = ct.spec
    npaths = rm.iparams[RA.RI_NPATHS]
    sev_at = RA.RI_PATH0 + RA.RI_PSTRIDE * npaths + w_
    spec.ivals[RA.SI_EVENT_ID] = rm.iparams[sev_at]
    spec.ivals[RA.SI_TICK] = tick
    spec.ivals[RA.SI_CONTACT] = p
    spec.ivals[RA.SI_RAPID] = 1 if rapid else 0
    fb = RA.RF_PATH0 + RA.RF_PSTRIDE * p
    plunge = 0
    vz = wk[WK_FEED + 2]
    vxy = math.sqrt(wk[WK_FEED] ** 2 + wk[WK_FEED + 1] ** 2)
    if vz < 0.0 and -vz > rm.fparams[fb + RA.RQ_TAN_RAMP_MAX] * vxy:
        plunge = 1
    spec.ivals[RA.SI_PLUNGE] = plunge
    spec.ivals[RA.SI_CHECK_TARGET] = 1 if tgt.info[0] != 0 else 0
    spec.ivals[RA.SI_LEVEL_MASK] = -1
    spec.fvals[RA.SF_GOUGE_TOL] = rm.fparams[RA.RF_GOUGE_TOL]
    spec.fvals[RA.SF_R_DEAD] = rm.fparams[fb + RA.RQ_R_DEAD]
    spec.fvals[RA.SF_Z_DEAD] = rm.fparams[fb + RA.RQ_Z_DEAD]
    spec.fvals[RA.SF_SLACK] = slack
    geo = ct.geo
    status = tooth_pass(f, prof, slabs, j, ct.cst.slab_c[j], tip, x0, y0, w, geo.st_z, geo.st_r, geo.st_active,
                        ct.cut.iv, geo.st_len, feed_x, feed_y, ct.gate, spec, tgt, racc, rm.events, rm.scratch,
                        ct.stats)
    if status == 0:
        rm.iparams[sev_at] += 1
        ct.cst.ival[CS_PASSES] += 1
        ct.cst.ival[CS_SUBTRACT_CALLS] += 1
        g.count[CNT_PASSES] += 1
    return status


@njit(void(CONTACT_T, f8), **RAW)
def tooth_windows(ct, sigma):
    """HotContact.teeth from ContactState.gamma / slab_done (call after either changes): per tooth the engaged
    slabs, those recorded since the last pass, and the arc of the engaged slabs' phases sigma (gamma - a0) (the
    spindle angle, in the direction of rotation, at which each slab passes its generation angle). A tooth whose
    arc the step's swept angle does not reach has no generation instant in the step (tooth_events)."""
    cst = ct.cst
    tw = ct.teeth
    a0 = ct.geo.slab_a0
    nt = min(cst.gamma.shape[0], tw.shape[0], a0.shape[0])
    ns = min(cst.gamma.shape[1], a0.shape[1])
    for j in range(tw.shape[0]):
        for c in range(N_TW):
            tw[j, c] = 0.0
    for j in range(nt):
        n_eng = 0
        n_done = 0
        ref = 0.0
        lo = 0.0
        hi = 0.0
        for k in range(ns):
            gam = cst.gamma[j, k]
            if gam != gam:
                continue
            ph = sigma * (gam - a0[j, k])
            ph -= TWO_PI * math.floor(ph / TWO_PI)
            if n_eng == 0:
                ref = ph
            else:
                rel = ph - ref
                rel -= TWO_PI * math.floor(rel / TWO_PI + 0.5)
                if rel < lo:
                    lo = rel
                if rel > hi:
                    hi = rel
            n_eng += 1
            if cst.slab_done[j, k] != 0:
                n_done += 1
        tw[j, TW_N_ENG] = n_eng
        tw[j, TW_N_DONE] = n_done
        start = ref + lo
        start -= TWO_PI * math.floor(start / TWO_PI)
        tw[j, TW_LO] = start
        tw[j, TW_WIDTH] = hi - lo if hi - lo < math.pi else -1.0
    ct.work[WK_TSIG] = sigma


@njit(void(CONTACT_T, f8), **RAW)
def update_gamma(ct, sigma):
    """Generation angle of every slab after an engaged-interval refresh: the end, in the direction of rotation
    sigma, of the largest engaged interval among the slab's stations (NaN when none is engaged); a full circle
    (continuous cut: drilling, plunging) takes its start. The tooth windows follow (tooth_windows)."""
    iv = ct.cut.iv
    geo = ct.geo
    cst = ct.cst
    nt = cst.gamma.shape[0]
    ns = cst.gamma.shape[1]
    for j in range(nt):
        for k in range(ns):
            idx = j * ns + k
            best = -1.0
            lo = 0.0
            hi = 0.0
            for q in range(geo.slab_ptr[idx], geo.slab_ptr[idx + 1]):
                s = geo.slab_sts[q]
                if s < 0 or s >= iv.count.shape[0]:
                    continue
                for i in range(iv.count[s]):
                    ext = iv.hi_rad[s, i] - iv.lo_rad[s, i]
                    if ext > best:
                        best = ext
                        lo = iv.lo_rad[s, i]
                        hi = iv.hi_rad[s, i]
            if best < 0.0:
                cst.gamma[j, k] = np.nan
            elif best >= TWO_PI - 1e-9:
                cst.gamma[j, k] = lo
            else:
                cst.gamma[j, k] = hi if sigma > 0.0 else lo
    tooth_windows(ct, sigma)


@njit(void(REMOVAL_T, CONTACT_T, i8, f8), **RAW)
def regate(rm, ct, tick, sigma):
    """Engaged intervals of every station at the step-end pose (removal's engaged_intervals, warm-started from the
    current ones), then the slabs' generation angles."""
    wk = ct.work
    f = rm.fields[ct.ipar[CI_WORK]]
    for r in range(3):
        wk[WK_X0 + r] = wk[WK_A1 + FR_R + 3 * r + 0]
        wk[WK_Y0 + r] = wk[WK_A1 + FR_R + 3 * r + 1]
        wk[WK_W0 + r] = wk[WK_A1 + FR_R + 3 * r + 2]
    geo = ct.geo
    engaged_intervals(f, wk[WK_A1 + FR_TIP:WK_A1 + FR_TIP + 3], wk[WK_X0:WK_X0 + 3], wk[WK_Y0:WK_Y0 + 3],
                      wk[WK_W0:WK_W0 + 3], geo.st_z, geo.st_r, geo.st_active, ct.cut.iv, tick, ct.gate3, ct.stats)
    ct.cst.ival[CS_GATINGS] += 1
    update_gamma(ct, sigma)


@njit(i8(REMOVAL_T, CONTACT_T, GLOBAL_T, i8, b1, i8, f8, f8), **RAW)
def tooth_events(rm, ct, g, tick, rapid, n_teeth, th0, th1):
    """Generation instants of the step just taken (docs/models/hotloop.md "Tooth passes"): a slab of a tooth is
    generated when it passes its generation angle (the exit of its station's engaged interval in the direction
    of rotation): its axis point X_e + u_rel is recorded then. A tooth's pass (removal's tooth_pass: engaged
    intervals refreshed against the field, then the tooth's envelope subtracted per slab at the recorded points)
    runs once every engaged slab of the tooth has been generated, or when the tooth has had no pass for longer
    than a bin stays fresh (one revolution and one tooth period: a slab missed its angle while the engagement
    changed); slabs not engaged take the step end. Without a pass for a tooth period the engaged intervals are
    refreshed alone. Returns -1, or the tooth whose pass overflowed the pool (repeated at the resume).

    Cost: a tooth whose window of slab phases (HotContact.teeth, tooth_windows) the step does not reach is skipped
    without visiting its slabs (the exact per-slab test runs for the others); the slab counts of the pass test
    are kept with the slabs' states, so a step costs O(teeth) without a generation instant."""
    p = ct.ipar[CI_PATH]
    slabs = rm.slabs[p]
    cst = ct.cst
    geo = ct.geo
    wk = ct.work
    tw = ct.teeth
    defl = ct.cut.defl
    R = ct.cut.pose.R_S_T0
    dth = th1 - th0
    adth = abs(dth)
    if adth <= 0.0:
        return -1
    sigma = 1.0 if dth > 0.0 else -1.0
    if wk[WK_TSIG] != sigma:
        tooth_windows(ct, sigma)
    # feed direction of the tip relative to the material in the T0 xy plane (contact geometry, plunge flag)
    vx = wk[WK_FEED]
    vy = wk[WK_FEED + 1]
    vz = wk[WK_FEED + 2]
    vxy = math.sqrt(vx * vx + vy * vy)
    vn = math.sqrt(vxy * vxy + vz * vz)
    fx = 0.0
    fy = 0.0
    if vxy > ct.fpar[CF_FEED_FRAC] * vn and vxy > 0.0:
        fx = vx / vxy
        fy = vy / vxy
    a0 = geo.slab_a0
    zm = geo.slab_zm
    nt = min(n_teeth, slabs.n.shape[0], cst.gamma.shape[0], tw.shape[0])
    for j in range(nt):
        if tw[j, TW_N_ENG] <= 0.0:
            continue
        width = tw[j, TW_WIDTH]
        if width >= 0.0:
            # start of the tooth's phase window after the step start, in the direction of rotation
            x = tw[j, TW_LO] - sigma * th0
            x -= TWO_PI * math.floor(x / TWO_PI)
            if x > adth + WIN_EPS and x + width < TWO_PI - WIN_EPS:
                continue
        for k in range(slabs.n[j]):
            gam = cst.gamma[j, k]
            if gam != gam:
                continue
            d0 = sigma * (gam - (th0 + a0[j, k]))
            d0 -= TWO_PI * math.floor(d0 / TWO_PI)
            if d0 < adth:
                slab_point(defl.u_st_T0_m, R, defl.u_work_S_m, geo.st_z, wk, d0 / adth, zm[j, k], cst.slab_c[j, k])
                if cst.slab_done[j, k] == 0:
                    tw[j, TW_N_DONE] += 1.0
                cst.slab_done[j, k] = 1
    stale = ct.cut.istep[CA.SI_STALE_TICKS]
    passed = False
    for j in range(nt):
        ns = slabs.n[j]
        if ns <= 0:
            continue
        n_eng = tw[j, TW_N_ENG]
        n_done = tw[j, TW_N_DONE]
        complete = n_done >= n_eng
        forced = n_done > 0.0 and not complete and tick - cst.last_pass[j] > stale
        if n_eng <= 0.0 or n_done <= 0.0 or not (complete or forced):
            continue
        if forced:
            cst.ival[CS_FORCED] += 1
        for k in range(ns):
            if cst.slab_done[j, k] == 0:
                slab_point(defl.u_st_T0_m, R, defl.u_work_S_m, geo.st_z, wk, 1.0, zm[j, k], cst.slab_c[j, k])
        # pose of the pass: the step end
        for r in range(3):
            wk[WK_PASS + r] = wk[WK_A1 + FR_TIP + r]
            wk[WK_PASS + 3 + r] = wk[WK_A1 + FR_R + 3 * r + 0]
            wk[WK_PASS + 6 + r] = wk[WK_A1 + FR_R + 3 * r + 1]
            wk[WK_PASS + 9 + r] = wk[WK_A1 + FR_R + 3 * r + 2]
        # slack: spread of the slab axis points about the axis through the tip
        slack = 0.0
        for k in range(ns):
            dx = cst.slab_c[j, k, 0] - wk[WK_PASS]
            dy = cst.slab_c[j, k, 1] - wk[WK_PASS + 1]
            dz = cst.slab_c[j, k, 2] - wk[WK_PASS + 2]
            along = dx * wk[WK_PASS + 9] + dy * wk[WK_PASS + 10] + dz * wk[WK_PASS + 11]
            rx = dx - along * wk[WK_PASS + 9]
            ry = dy - along * wk[WK_PASS + 10]
            rz = dz - along * wk[WK_PASS + 11]
            d = math.sqrt(rx * rx + ry * ry + rz * rz)
            if d > slack:
                slack = d
        status = do_pass(rm, ct, g, j, tick, fx, fy, slack, rapid)
        if status != 0:
            wk[WK_PFX] = fx
            wk[WK_PFY] = fy
            wk[WK_PSLACK] = slack
            return j
        for k in range(ns):
            cst.slab_done[j, k] = 0
        cst.last_pass[j] = tick
        cst.ival[CS_GATINGS] += 1
        update_gamma(ct, sigma)
        passed = True
    # the engaged intervals are refreshed at least once per tooth period
    cst.fval[CSF_GATE_ANGLE] += adth
    if passed:
        cst.fval[CSF_GATE_ANGLE] = 0.0
    elif cst.fval[CSF_GATE_ANGLE] >= TWO_PI / max(n_teeth, 1):
        cst.fval[CSF_GATE_ANGLE] = 0.0
        regate(rm, ct, tick, sigma)
    return -1


# =============================================================================== L0 steps of one contact
@njit(i8(MOTION_T, REMOVAL_T, CONTACT_T, RING_T, GLOBAL_T, i8, i8, i8, i8), **RAW)
def contact_steps(mo, rm, ct, ring, g, k, s0, c, code):
    """L0 steps s0 .. n - 1 of tick k for contact c (in contact). Returns -1 when the tick is done, or the step
    index to resume at after a split (g.result filled)."""
    chain = mo.chain
    servo = mo.servo
    ss = mo.servo_state
    sps = mo.spindle_state
    setp = mo.setpoint
    out = mo.out
    cut = ct.cut
    cst = ct.cst
    geo = ct.geo
    wk = ct.work
    ip = ct.ipar
    p = ip[CI_PATH]
    w_ = ip[CI_WORK]
    f = rm.fields[w_]
    ipo = g.ipar[GI_IPO_TICKS]
    tick0 = g.ipar[GI_TICK0]
    tick_s = g.fpar[GF_TICK_S]
    h_tick = ipo * tick_s
    l0 = ip[CI_L0_TICKS]
    n_sub = ipo // l0
    dt = l0 * tick_s
    kind = ip[CI_KIND]
    rotating = ip[CI_ROTATING] != 0
    st_t = ip[CI_TOOL_SPINDLE]
    st_w = ip[CI_WORK_SPINDLE]
    n_teeth = ip[CI_N_TEETH]
    dyn = ip[CI_DYN] != 0
    sys = ct.sys
    rapid = setp.rapid[k, p] != 0
    step = cut.step
    istep = cut.istep
    hand = ct.fpar[CF_HAND]
    W_S = chain.pose_W_S[p]
    mx = ct.fpar[CF_MOUNT_X]
    my = ct.fpar[CF_MOUNT_Y]
    mz = ct.fpar[CF_MOUNT_Z]
    mounted = mx != 1.0 or my != 1.0 or mz != 1.0
    th1_all = sps.angle
    w1_all = sps.omega
    # repeat of a pass refused by a pool overflow (the field was grown by the wrapper)
    if cst.ival[CS_OVERFLOW_TOOTH] >= 0:
        j = cst.ival[CS_OVERFLOW_TOOTH]
        status = do_pass(rm, ct, g, j, istep[CA.SI_TICK_END], wk[WK_PFX], wk[WK_PFY], wk[WK_PSLACK], rapid)
        if status != 0:
            _split(g, k, c, s0, SPLIT_POOL, w_, code)
            return s0
        cst.ival[CS_OVERFLOW_TOOTH] = -1
        for kk in range(cst.slab_done.shape[1]):
            cst.slab_done[j, kk] = 0
        cst.last_pass[j] = istep[CA.SI_TICK_END]
        update_gamma(ct, 1.0 if step[CA.ST_OMEGA_RAD_S] > 0.0 else -1.0)
    # the per-tooth slab windows of tooth_events (derived scratch: rebuilt once per call)
    if rotating and kind == RA.TK_ENVELOPE:
        sig0 = 1.0
        if st_t >= 0 and (th1_all[st_t] - g.th0[st_t] < 0.0 or (th1_all[st_t] == g.th0[st_t] and w1_all[st_t] < 0.0)):
            sig0 = -1.0
        tooth_windows(ct, sig0)
    # the layouts the step's kernels take (members of CutArgs: taken once, not per step)
    dfl = cut.defl
    tool = cut.tool
    table = cut.table
    mem = cut.mem
    iv = cut.iv
    refill = cut.refill
    pose = cut.pose
    occ = cut.occ
    pack = cut.pack
    axes = cut.axes
    kst = cut.st
    acc = cut.acc
    cacc = cut.cacc
    cout = cut.out
    ev = cut.ev
    polar = cut.polar
    # frame at the start of the first step taken in this call
    a_start = s0 / n_sub
    interp_axes(servo, ss, g.q0, out.actual, k, g.th0, g.w0, th1_all, w1_all, a_start, h_tick, ct.qv)
    fk_frame(chain, p, ct.qv, ct.poses, ct.fkw, wk, WK_A0, mx, my, mz)
    # base excitation of the tick (axis accelerations of the path, rotary spindles from their speed change)
    if dyn:
        pa = geo.path_axes
        for i in range(min(pa.shape[0], ct.acc_axis.shape[0])):
            ax = pa[i]
            sl = servo.spindle[ax]
            if sl >= 0:
                ct.acc_axis[i] = servo.kappa[ax] * (w1_all[sl] - g.w0[sl]) / h_tick
            else:
                ct.acc_axis[i] = ss.acc[ax]
        relative_axis_acceleration(ct.base, ct.acc_axis, ct.base1)
    alpha_t = 0.0
    if st_t >= 0:
        alpha_t = (w1_all[st_t] - g.w0[st_t]) / h_tick
    istep[CA.SI_L0_TICKS] = l0
    istep[CA.SI_TICK_ROW] = k
    istep[CA.SI_COMMIT] = 1
    sb = ip[CI_STALE_BEFORE]
    if cst.ival[CS_LAST_RAPID] > sb:
        sb = cst.ival[CS_LAST_RAPID]
    istep[CA.SI_STALE_BEFORE] = sb
    step[CA.ST_DT_S] = dt
    step[CA.ST_TICK_FRAC] = 1.0 / n_sub
    step[CA.ST_TICK_S] = tick_s
    nb_mem = mem.point_S_m.shape[1]
    for s in range(s0, n_sub):
        tick_end = tick0 + k * ipo + (s + 1) * l0
        # ---------------------------------------------------- headroom of the event buffers (split rule)
        hb = headroom(ct, rm, g)
        if hb >= 0:
            _split(g, k, c, s, SPLIT_DRAIN, hb, code)
            return s
        am = (s + 0.5) / n_sub
        a1 = (s + 1.0) / n_sub
        # ---------------------------------------------------- spindle angles, poses
        th_t = 0.0
        om_t = 0.0
        th_t1 = 0.0
        om_t1 = 0.0
        th_t0 = 0.0
        if st_t >= 0:
            th_t0, _ = hermite(g.th0[st_t], g.w0[st_t], th1_all[st_t], w1_all[st_t], s / n_sub, h_tick)
            th_t, om_t = hermite(g.th0[st_t], g.w0[st_t], th1_all[st_t], w1_all[st_t], am, h_tick)
            th_t1, om_t1 = hermite(g.th0[st_t], g.w0[st_t], th1_all[st_t], w1_all[st_t], a1, h_tick)
        om_w = 0.0
        if st_w >= 0:
            _, om_w = hermite(g.th0[st_w], g.w0[st_w], th1_all[st_w], w1_all[st_w], am, h_tick)
        interp_axes(servo, ss, g.q0, out.actual, k, g.th0, g.w0, th1_all, w1_all, am, h_tick, ct.qv)
        fk_frame(chain, p, ct.qv, ct.poses, ct.fkw, wk, WK_MID, mx, my, mz)
        interp_axes(servo, ss, g.q0, out.actual, k, g.th0, g.w0, th1_all, w1_all, a1, h_tick, ct.qv)
        fk_frame(chain, p, ct.qv, ct.poses, ct.fkw, wk, WK_A1, mx, my, mz)
        set_step_pose(pose, wk, W_S, dt)
        # feed velocity relative to the material in T0 (generation angle, plunge flag)
        R = pose.R_S_T0
        v = pose.v_tip_S_m_s
        for r in range(3):
            wk[WK_FEED + r] = R[0, r] * v[0] + R[1, r] * v[1] + R[2, r] * v[2]
        # ---------------------------------------------------- step scalars
        istep[CA.SI_TICK_END] = tick_end
        if rotating:
            theta_mid = th_t - TWO_PI * math.floor(th_t / TWO_PI)
            step[CA.ST_THETA_MID_RAD] = theta_mid
            step[CA.ST_COS_MID] = math.cos(th_t)
            step[CA.ST_SIN_MID] = math.sin(th_t)
            step[CA.ST_OMEGA_RAD_S] = om_t
            step[CA.ST_TOOTH_PERIOD_S] = TWO_PI / (n_teeth * abs(om_t)) if (om_t != 0.0 and n_teeth > 0) else 0.0
        else:
            theta_mid = 0.0
            step[CA.ST_THETA_MID_RAD] = 0.0
            step[CA.ST_COS_MID] = 1.0
            step[CA.ST_SIN_MID] = 0.0
            om_s = om_w * ct.fpar[CF_WORK_SIGN]
            step[CA.ST_OMEGA_RAD_S] = om_s
            step[CA.ST_TOOTH_PERIOD_S] = TWO_PI / abs(om_s) if om_s != 0.0 else 0.0
        # ---------------------------------------------------- dynamics predictor, deflections
        if dyn:
            # stations of the step for dynamics (deflections, load projection): the engaged ones and CI_ST_MARGIN
            # neighbours on each side (slab points interpolate between the stations around a slab's mid height;
            # every other station has no load)
            act = sys.act
            n_d = min(dfl.u_st_T0_m.shape[0], sys.defl.u_st_T0_m.shape[0])
            if rotating:
                lo_e = -1
                hi_e = -2
                for st_i in range(min(iv.count.shape[0], n_d)):
                    if iv.count[st_i] > 0:
                        if lo_e < 0:
                            lo_e = st_i
                        hi_e = st_i
                if lo_e >= 0:
                    m_st = ip[CI_ST_MARGIN]
                    act[ACT_LO] = max(lo_e - m_st, 0)
                    act[ACT_HI] = min(hi_e + m_st, n_d - 1)
                    act[ACT_LOAD_LO] = lo_e
                    act[ACT_LOAD_HI] = hi_e
                else:
                    act[ACT_LO] = 0
                    act[ACT_HI] = -1
                    act[ACT_LOAD_LO] = 0
                    act[ACT_LOAD_HI] = -1
            else:
                act[ACT_LO] = -1
                act[ACT_HI] = -1
                act[ACT_LOAD_LO] = -1
                act[ACT_LOAD_HI] = -1
            modal_begin(sys, th_t1, om_t1, alpha_t, ct.base1)
            md = sys.defl
            if ip[CI_DEFL_SHARED] == 0:
                d_lo = 0 if act[ACT_LO] < 0 else act[ACT_LO]
                d_hi = n_d - 1 if act[ACT_LO] < 0 else act[ACT_HI]
                # dynamics works in the chain's T0 (motion's pose_S_T); the tool mount turns it into the tool's
                for st_i in range(d_lo, d_hi + 1):
                    for cc in range(6):
                        sg = mx if cc % 3 == 0 else (my if cc % 3 == 1 else mz)
                        dfl.u_st_T0_m[st_i, cc] = sg * md.u_st_T0_m[st_i, cc]
                        dfl.v_st_T0_m_s[st_i, cc] = sg * md.v_st_T0_m_s[st_i, cc]
            for cc in range(3):
                dfl.u_work_S_m[cc] = md.u_work_S_m[cc] + wk[WK_DISP + cc]
                dfl.v_work_S_m_s[cc] = md.v_work_S_m_s[cc]
        else:
            for cc in range(3):
                dfl.u_work_S_m[cc] = wk[WK_DISP + cc]
        # ---------------------------------------------------- refill planes, forces
        if rotating:
            n_ref = refill_planes(tool.st_ptr, tool.st_elem, tool.alpha0_rad, tool.read_row, mem.valid, mem.tick,
                                  refill.q_S_m, refill.m_S, refill.valid, pose.R_S_T0, pose.tip_S_m, iv.lo_rad,
                                  iv.hi_rad, iv.count, step, istep, f, geo.st_z, geo.st_r, geo.st_nr, geo.st_nz,
                                  tick_end, theta_mid, ct.fpar[CF_REACH], ct.plane, cst.ival, CS_REFILLS,
                                  CS_REFILL_MISS, ct.planes)
            g.count[CNT_REFILLS] += n_ref
            cut_step(tool, table, mem, iv, refill, step, istep, pose, dfl, occ, pack, axes, kst, acc, cacc, cout, ev)
            if n_ref > 0 or ct.planes[0, 1] != 0:
                clear_planes(refill.valid, ct.planes)
        else:
            turn_step(tool, table, polar, step, istep, pose, dfl, axes, kst, acc, cacc, cout, ev)
        if cout.misc[CA.SO_ENGAGED] > 0.0:
            cst.ival[CS_ENGAGED_TICK] = 1
        if dyn:
            # station loads back into the chain's T0 for dynamics (sign flips: exact, restored after)
            stw = cout.station_T0
            if mounted:
                for st_i in range(stw.shape[0]):
                    for cc in range(6):
                        stw[st_i, cc] *= mx if cc % 3 == 0 else (my if cc % 3 == 1 else mz)
            modal_correct(sys, stw, cout.force_S_N, tick_end, acc)
            if mounted:
                for st_i in range(stw.shape[0]):
                    for cc in range(6):
                        stw[st_i, cc] *= mx if cc % 3 == 0 else (my if cc % 3 == 1 else mz)
        cst.ival[CS_L0_STEPS] += 1
        g.count[CNT_L0] += 1
        # ---------------------------------------------------- recorder ring
        cf = ip[CI_RING_FORCE]
        cd = ip[CI_RING_DYN]
        ctq = ip[CI_RING_TORQUE]
        if cf >= 0 or cd >= 0 or ctq >= 0:
            nrow = ring.tick.shape[0]
            slot = (ring.head[0] - 1) % nrow
            if ring.head[0] == 0 or ring.tick[slot] != tick_end:
                slot = ring_begin(ring, tick_end)
            if cf >= 0:
                for cc in range(3):
                    ring.data[slot, cf + cc] = cout.wrench_T0[cc]
            if ctq >= 0:
                ring.data[slot, ctq] = cout.misc[CA.SO_TORQUE_NM]
            if cd >= 0 and dyn:
                ring_write(sys, ring, slot, cd)
        # ---------------------------------------------------- removal
        cst.fval[CSF_THETA_PREV] = th_t1
        if kind == RA.TK_ENVELOPE:
            jj = tooth_events(rm, ct, g, tick_end, rapid, n_teeth, th_t0, th_t1)
            if jj >= 0:
                cst.ival[CS_OVERFLOW_TOOTH] = jj
                for r in range(3):
                    wk[WK_A0 + r] = wk[WK_A1 + r]
                _split(g, k, c, s + 1, SPLIT_POOL, w_, code)
                return s + 1
        elif kind == RA.TK_HELIX:
            hx = rm.helix[p]
            tgt = rm.targets[w_]
            npaths = rm.iparams[RA.RI_NPATHS]
            sev_at = RA.RI_PATH0 + RA.RI_PSTRIDE * npaths + w_
            tip_a = wk[WK_A0 + FR_TIP:WK_A0 + FR_TIP + 3]
            tip_b = wk[WK_A1 + FR_TIP:WK_A1 + FR_TIP + 3]
            wax = wk[WK_W0:WK_W0 + 3]
            x0 = wk[WK_X0:WK_X0 + 3]
            y0 = wk[WK_Y0:WK_Y0 + 3]
            for r in range(3):
                x0[r] = wk[WK_A1 + FR_R + 3 * r]
                y0[r] = wk[WK_A1 + FR_R + 3 * r + 1]
                wax[r] = wk[WK_A1 + FR_R + 3 * r + 2]
            lo = ct.spec.box_lo
            hi = ct.spec.box_hi
            for r in range(3):
                lo[r] = -1e300
                hi[r] = 1e300
            status = helix_subtract(f, hx, tip_a, tip_b, wax, x0, y0, th_t0, th_t1, lo, hi, rm.iparams[sev_at],
                                    tick_end, p, tgt, rm.accum[p], rm.events, rm.scratch)
            g.count[CNT_HELIX_CALLS] += 1
            if status != 0:
                _split(g, k, c, s, SPLIT_POOL, w_, code)
                return s
            rm.iparams[sev_at] += 1
        elif kind == RA.TK_TURNING and ip[CI_WORK] >= 0:
            pf = rm.polars[w_]
            ins = rm.inserts[p]
            ps = ct.polar_step
            sz = ct.fpar[CF_POLAR_SZ]
            sx = ct.fpar[CF_POLAR_SX]
            # the field is cut where the deflected edge is (architecture 4.1: X_e = x_e + u_tool - u_work, as
            # cutting composes its elements): the tip's displacement from the station nearest it (translation
            # and rotation, T0) and the workpiece's (deflection and d(X), S); the insert as a rigid body
            ust = dfl.u_st_T0_m
            stz = tool.st_z_m
            s_tip = 0
            for st_i in range(min(stz.shape[0], ust.shape[0])):
                if abs(stz[st_i]) < abs(stz[s_tip]):
                    s_tip = st_i
            zst = stz[s_tip] if stz.shape[0] > 0 else 0.0
            utx = ust[s_tip, 0] - ust[s_tip, 4] * zst
            uty = ust[s_tip, 1] + ust[s_tip, 3] * zst
            utz = ust[s_tip, 2]
            tip_ang = 0.0
            for e_i in range(2):
                off = WK_A0 if e_i == 0 else WK_A1
                tx = wk[off + FR_TIP] - dfl.u_work_S_m[0]
                ty = wk[off + FR_TIP + 1] - dfl.u_work_S_m[1]
                tz = wk[off + FR_TIP + 2] - dfl.u_work_S_m[2]
                tx += wk[off + FR_R + 0] * utx + wk[off + FR_R + 1] * uty + wk[off + FR_R + 2] * utz
                ty += wk[off + FR_R + 3] * utx + wk[off + FR_R + 4] * uty + wk[off + FR_R + 5] * utz
                tz += wk[off + FR_R + 6] * utx + wk[off + FR_R + 7] * uty + wk[off + FR_R + 8] * utz
                # radial direction of the tool's reference plane: e_r = sx x_T (projected on S xy)
                erx = sx * wk[off + FR_R + 0]
                ery = sx * wk[off + FR_R + 3]
                n_e = math.sqrt(erx * erx + ery * ery)
                if n_e > 0.0:
                    erx /= n_e
                    ery /= n_e
                th = math.atan2(ery, erx)
                rho = tx * erx + ty * ery
                yoff = -tx * ery + ty * erx
                if e_i == 1:
                    tip_ang = math.atan2(ty, tx)
                if e_i == 0:
                    ps[RA.PS_Z0] = tz
                    ps[RA.PS_R0] = rho
                    ps[RA.PS_TH0] = _unwrap_near(th, cst.fval[CSF_BETA])
                else:
                    ps[RA.PS_Z1] = tz
                    ps[RA.PS_R1] = rho
                    ps[RA.PS_TH1] = _unwrap_near(th, ps[RA.PS_TH0])
                    ps[RA.PS_YOFF] = yoff
            ps[RA.PS_SZ] = sz
            ps[RA.PS_SX] = sx
            cst.fval[CSF_BETA] = ps[RA.PS_TH1]
            npaths = rm.iparams[RA.RI_NPATHS]
            sev_at = RA.RI_PATH0 + RA.RI_PSTRIDE * npaths + w_
            status = polar_cut(pf, ins, ps, rm.iparams[sev_at], rm.accum[p])
            g.count[CNT_POLAR_CUTS] += 1
            if status != 0:
                _split(g, k, c, s, SPLIT_POLAR, w_, code)
                return s
            rm.iparams[sev_at] += 1
            # the bins the tool's plane passed: their view is refreshed once the tool has left them
            dth = pf.geom[RA.PG_DTHETA]
            a_lo = min(ps[RA.PS_TH0], ps[RA.PS_TH1])
            a_hi = max(ps[RA.PS_TH0], ps[RA.PS_TH1])
            kf = int(math.floor(a_lo / dth - 0.5)) + 1
            kl = int(math.floor(a_hi / dth - 0.5))
            z0f = pf.geom[RA.PG_Z0]
            dzf = pf.geom[RA.PG_DZ]
            z_a = min(ps[RA.PS_Z0], ps[RA.PS_Z1]) + min(sz * ins.box[0], sz * ins.box[1])
            z_b = max(ps[RA.PS_Z0], ps[RA.PS_Z1]) + max(sz * ins.box[0], sz * ins.box[1])
            i_lo = int(math.floor((z_a - z0f) / dzf)) - 1
            i_hi = int(math.floor((z_b - z0f) / dzf)) + 1
            pw = cst.polar_win
            # a bin's view keeps the previous revolution while any element can still read it: cutting interpolates
            # between the two bin centres around an element's angle, so bin b is refreshed once the angle that
            # lags (the reference plane or the deflected tip: a tangential deflection or a centre-height offset
            # puts the edge off the plane) has passed the centre of the next bin in the direction of motion
            inc = ps[RA.PS_TH1] >= ps[RA.PS_TH0]
            a_tip = _unwrap_near(tip_ang, ps[RA.PS_TH1])
            a_lag = min(ps[RA.PS_TH1], a_tip) if inc else max(ps[RA.PS_TH1], a_tip)
            b_st = int(math.floor(a_lag / dth - 0.5))           # lower bin of the lagging angle's stencil
            if kl >= kf:
                if pw[0] == -1 or pw[4] < 0:
                    pw[0] = kf
                    pw[1] = kl
                    pw[2] = i_lo
                    pw[3] = i_hi
                    pw[4] = 1
                else:
                    pw[0] = min(pw[0], kf)
                    pw[1] = max(pw[1], kl)
                    pw[2] = min(pw[2], i_lo)
                    pw[3] = max(pw[3], i_hi)
            if pw[0] != -1 and pw[4] >= 0:
                if inc:
                    if b_st > pw[0]:
                        hi_b = min(pw[1], b_st - 1)
                        polar_view_update(pf, polar, pw[0], hi_b, pw[2], pw[3], sx)
                        if hi_b >= pw[1]:
                            pw[0] = -1
                            pw[4] = -1
                        else:
                            pw[0] = hi_b + 1
                elif b_st + 1 < pw[1]:
                    lo_b = max(pw[0], b_st + 2)
                    polar_view_update(pf, polar, lo_b, pw[1], pw[2], pw[3], sx)
                    if lo_b <= pw[0]:
                        pw[0] = -1
                        pw[4] = -1
                    else:
                        pw[1] = lo_b - 1
        # next step starts where this one ended
        for r in range(24):
            wk[WK_A0 + r] = wk[WK_A1 + r]
    return -1


# =============================================================================== one macro step
@njit(i4(MOTION_T, REMOVAL_T, CONTACTS_T, RING_T, GLOBAL_T), **RAW)
def run_macro(mo, rm, cs, ring, g):
    """Ticks mo.k_from .. macro_ipo - 1 of the macro step (module doc). Returns motion's sync code (mo.result =
    (code, tick, path) as motion.kernels.run_motion), SYNC_NONE at the step end, or HOT_SPLIT (g.result: tick,
    contact, reason, detail; resume by calling again with the same arguments)."""
    chain = mo.chain
    servo = mo.servo
    ss = mo.servo_state
    sp = mo.spindle
    sps = mo.spindle_state
    setp = mo.setpoint
    ls = mo.loop
    out = mo.out
    n_ticks = setp.mode.shape[0]
    n_paths = setp.mode.shape[1]
    n_axes = servo.kv.shape[0]
    n_sp = sp.inertia.shape[0]
    ipo = g.ipar[GI_IPO_TICKS]
    tick0 = g.ipar[GI_TICK0]
    tick_s = g.fpar[GF_TICK_S]
    h_tick = ipo * tick_s
    n_c = min(g.ipar[GI_N_CONTACTS], MAX_CONTACTS)
    res = g.result
    for i in range(res.shape[0]):
        res[i] = 0
    res[RES_PATH] = -1
    res[RES_CONTACT] = -1
    k_from = np.int64(mo.k_from[0])
    mo.result[0] = SYNC_NONE
    mo.result[1] = n_ticks - 1
    mo.result[2] = -1
    resume = g.ipar[GI_RESUME] != 0
    k = k_from
    if resume:
        k = g.ipar[GI_RESUME_K]
    dist_force = mo.dist_force
    dist_torque = mo.dist_torque
    while k < n_ticks:
        c_start = 0
        s_start = 0
        if resume:
            resume = False
            c_start = g.ipar[GI_RESUME_C]
            s_start = g.ipar[GI_RESUME_S]
            code = np.int64(g.ipar[GI_RESUME_CODE])
            g.ipar[GI_RESUME] = 0
        else:
            if k > k_from:
                p_end = replan_before(setp, k)
                if p_end >= 0:
                    out.sync[k - 1, p_end] = SYNC_QUEUE_EMPTY
                    mo.result[0] = SYNC_QUEUE_EMPTY
                    mo.result[1] = k - 1
                    mo.result[2] = p_end
                    res[RES_CODE] = SYNC_QUEUE_EMPTY
                    res[RES_TICK] = k - 1
                    res[RES_PATH] = p_end
                    return SYNC_QUEUE_EMPTY
            # tick-start state of the L0 interpolation
            for a in range(n_axes):
                g.q0[a] = ss.load[a]
            for s_ in range(n_sp):
                g.th0[s_] = sps.angle[s_]
                g.w0[s_] = sps.omega[s_]
            code = np.int64(motion_tick(chain, servo, ss, sp, sps, setp, ls, out, k, dist_force, dist_torque))
            g.count[CNT_TICKS] += 1
            tick_end = tick0 + (k + 1) * ipo
            # ------------------------------------------------ probe sweep of G31 moves
            for c in range(n_c):
                ct = cs[c]
                p = ct.ipar[CI_PATH]
                if ct.ipar[CI_PROBE] == 0 or p < 0 or p >= n_paths or setp.probe[k, p] == 0:
                    continue
                f = rm.fields[ct.ipar[CI_WORK]]
                c1 = out.tip_S[k, p]
                wkc = ct.work
                wv = wkc[WK_W0:WK_W0 + 3]
                xv = wkc[WK_X0:WK_X0 + 3]
                probe_out = wkc[WK_PROBE:WK_PROBE + 8]
                st0 = out.poses[p, 3]
                for r in range(3):
                    wv[r] = st0[r, 2]
                    xv[r] = st0[r, 0]
                lam = sweep_contact(f, ct.probe, g.tip0[p], c1, wv, xv, g.ipar[GI_PROBE_RINGS],
                                    g.ipar[GI_PROBE_AROUND], g.fpar[GF_PROBE_TOL], probe_out)
                g.count[CNT_PROBE_SWEEPS] += 1
                if lam >= 0.0 and lam <= 1.0 and code == SYNC_NONE:
                    out.probe_hit[k, p] = 1
                    out.sync[k, p] = SYNC_PROBE_TRIGGER
                    code = SYNC_PROBE_TRIGGER
                    for r in range(8):
                        ct.probe_out[r] = probe_out[r]
            # ------------------------------------------------ broad phase, collision, rapid flags
            for c in range(n_c):
                ct = cs[c]
                if ct.ipar[CI_ACTIVE] == 0:
                    continue
                p = ct.ipar[CI_PATH]
                cst = ct.cst
                cst.ival[CS_ENGAGED_TICK] = 0
                if setp.rapid[k, p] != 0:
                    cst.ival[CS_LAST_RAPID] = tick_end
                w_ = ct.ipar[CI_WORK]
                f = rm.fields[w_]
                tip = out.tip_S[k, p]
                st0 = out.poses[p, 3]
                wkc = ct.work
                wv = wkc[WK_W0:WK_W0 + 3]
                for r in range(3):
                    wv[r] = st0[r, 2]
                move = 0.0
                for r in range(3):
                    move += (tip[r] - g.tip0[p, r]) ** 2
                margin = math.sqrt(move) + ct.fpar[CF_LATTICE]
                kind = ct.ipar[CI_KIND]
                if kind == RA.TK_TURNING:
                    ins = rm.inserts[p]
                    pf = rm.polars[w_]
                    sz = ct.fpar[CF_POLAR_SZ]
                    za = tip[2] + min(sz * ins.box[0], sz * ins.box[1]) - margin
                    zb = tip[2] + max(sz * ins.box[0], sz * ins.box[1]) + margin
                    r_tip = math.sqrt(tip[0] * tip[0] + tip[1] * tip[1])
                    extent = max(abs(ins.box[2]), abs(ins.box[3]))
                    r_mat = polar_max_radius(pf, za, zb)
                    contact = r_mat > 0.0 and r_tip - extent - margin <= r_mat
                else:
                    prof = rm.cut[p]
                    feed = setp.rapid[k, p] == 0
                    contact = l1_contact(f, prof, tip, wv, margin, feed, tick_end, tick_s, rm.accum[p])
                    if contact:
                        tipc = wkc[WK_TIPC:WK_TIPC + 3].reshape((1, 3))
                        for r in range(3):
                            tipc[0, r] = tip[r]
                        collide_envelope(f, rm.whole[p], prof, wv, tipc, wv, tick_end, p, ct.fpar[CF_COLLISION_TOL],
                                         rm.accum[p], rm.events, rm.scratch)
                g.count[CNT_BROAD] += 1
                if contact and cst.ival[CS_IN_CONTACT] == 0:
                    cst.ival[CS_FIRST] = 1
                    if kind == RA.TK_ENVELOPE:
                        # gating at the start of an engagement (the passes refresh it once per tooth period)
                        xv = wkc[WK_X0:WK_X0 + 3]
                        yv = wkc[WK_Y0:WK_Y0 + 3]
                        for r in range(3):
                            xv[r] = st0[r, 0]
                            yv[r] = st0[r, 1]
                        geo = ct.geo
                        engaged_intervals(f, tip, xv, yv, wv, geo.st_z, geo.st_r, geo.st_active, ct.cut.iv, tick_end,
                                          ct.gate3, ct.stats)
                        st_t = ct.ipar[CI_TOOL_SPINDLE]
                        sg = 1.0
                        if st_t >= 0 and sps.omega[st_t] < 0.0:
                            sg = -1.0
                        update_gamma(ct, sg)
                        cst.fval[CSF_GATE_ANGLE] = 0.0
                        for jj in range(cst.slab_done.shape[0]):
                            for kk in range(cst.slab_done.shape[1]):
                                cst.slab_done[jj, kk] = 0
                        tooth_windows(ct, sg)
                cst.ival[CS_IN_CONTACT] = 1 if contact else 0
                # structure displacement d(X) at the contact point for this tick
                wk = ct.work
                xp = wk[WK_XP:WK_XP + 3]
                for r in range(3):
                    xp[r] = tip[r] + ct.offset_S[r]
                if g.ipar[GI_HAS_DISP] != 0:
                    dv = wk[WK_DV:WK_DV + 3]
                    displacement_at(g.disp_origin, g.disp_cell, g.disp, xp, dv)
                    for r in range(3):
                        wk[WK_DISP + r] = dv[r]
                # workpiece mode shapes at the contact point
                if ct.ipar[CI_DYN] != 0:
                    refresh_work_shapes(ct.sys, ct.grid, xp)
        # ---------------------------------------------------- L0 steps of the contacts in contact
        for c in range(c_start, n_c):
            ct = cs[c]
            if ct.ipar[CI_ACTIVE] == 0 or ct.cst.ival[CS_IN_CONTACT] == 0:
                continue
            s0 = s_start if c == c_start else 0
            r_ = contact_steps(mo, rm, ct, ring, g, k, s0, c, code)
            if r_ >= 0:
                return HOT_SPLIT
        # ---------------------------------------------------- out of contact: free dynamics; tick records
        tick_end = tick0 + (k + 1) * ipo
        for c in range(n_c):
            ct = cs[c]
            if ct.ipar[CI_ACTIVE] == 0 or ct.ipar[CI_DYN] == 0:
                continue
            sys = ct.sys
            if ct.cst.ival[CS_IN_CONTACT] == 0:
                p = ct.ipar[CI_PATH]
                pa = ct.geo.path_axes
                for i in range(min(pa.shape[0], ct.acc_axis.shape[0])):
                    ax = pa[i]
                    sl = servo.spindle[ax]
                    if sl >= 0:
                        ct.acc_axis[i] = servo.kappa[ax] * (sps.omega[sl] - g.w0[sl]) / h_tick
                    else:
                        ct.acc_axis[i] = ss.acc[ax]
                relative_axis_acceleration(ct.base, ct.acc_axis, ct.base1)
                st_t = ct.ipar[CI_TOOL_SPINDLE]
                th0 = 0.0
                th1 = 0.0
                w0 = 0.0
                w1 = 0.0
                if st_t >= 0 and ct.ipar[CI_ROTATING] != 0:
                    th0 = g.th0[st_t]
                    th1 = sps.angle[st_t]
                    w0 = g.w0[st_t]
                    w1 = sps.omega[st_t]
                if sys.ds.node_tick[0] < tick_end:
                    modal_free(sys, th0, th1, w0, w1, ct.base0, ct.base1, tick_end)
                ct.cst.ival[CS_FREE_TICKS] += 1
                g.count[CNT_FREE] += 1
            record_tick(sys, k)
            for r in range(ct.base0.shape[0]):
                ct.base0[r] = ct.base1[r]
        # ---------------------------------------------------- loads to L1 (next tick's disturbances), in_cut
        for a in range(n_axes):
            dist_force[a] = 0.0
        for s_ in range(n_sp):
            dist_torque[s_] = 0.0
        for c in range(n_c):
            ct = cs[c]
            if ct.ipar[CI_ACTIVE] == 0:
                continue
            p = ct.ipar[CI_PATH]
            acc = ct.cut.acc
            if ct.cst.ival[CS_IN_CONTACT] != 0 and k < acc.tick_axis_force_N.shape[0]:
                ca = ct.geo.cut_axes
                for ia in range(min(ca.shape[0], acc.tick_axis_force_N.shape[1])):
                    ax = ca[ia]
                    if ax >= 0:
                        dist_force[ax] += acc.tick_axis_force_N[k, ia]
                tq = acc.tick_spindle_torque_Nm[k]
                if ct.ipar[CI_ROTATING] != 0:
                    st_t = ct.ipar[CI_TOOL_SPINDLE]
                    if st_t >= 0:
                        dist_torque[st_t] -= tq
                else:
                    st_w = ct.ipar[CI_WORK_SPINDLE]
                    if st_w >= 0:
                        dist_torque[st_w] -= tq * ct.fpar[CF_WORK_SIGN]
            out.in_cut[k, p] = 1 if ct.cst.ival[CS_ENGAGED_TICK] != 0 else 0
            g.tick_flags[k, p] = out.in_cut[k, p]
        for a in range(n_axes):
            g.load_force[a] = dist_force[a]
        for s_ in range(n_sp):
            g.load_torque[s_] = dist_torque[s_]
        for p in range(n_paths):
            for r in range(3):
                g.tip0[p, r] = out.tip_S[k, p, r]
        if code != SYNC_NONE:
            mo.result[0] = code
            mo.result[1] = k
            for p in range(n_paths):
                if out.sync[k, p] != SYNC_NONE:
                    mo.result[2] = p
                    break
            res[RES_CODE] = code
            res[RES_TICK] = k
            res[RES_PATH] = mo.result[2]
            return np.int32(code)
        k += 1
    res[RES_CODE] = SYNC_NONE
    res[RES_TICK] = n_ticks - 1
    return SYNC_NONE


KERNELS = (headroom, _split, _unwrap_near, do_pass, tooth_windows, update_gamma, tooth_events, regate, contact_steps,
           run_macro)


# =============================================================================== Python wrapper
FAST_DOMAIN = "hotloop"


def channels():
    """Fast (L0) channels the hot loop writes into the recorder ring (dynamics writes its own)."""
    from .channels import Channel, Tier
    from .hot_abi import FAST_FORCE, FAST_TORQUE, MAX_PATHS
    return [Channel(FAST_FORCE, "N", Tier.FAST, FAST_DOMAIN, "tool force in T0 per path (x, y, z), L0 step mean",
                    (MAX_PATHS, 3)),
            Channel(FAST_TORQUE, "N m", Tier.FAST, FAST_DOMAIN,
                    "process torque about the spindle axis on the rotating body per path, L0 step mean", (MAX_PATHS,))]


class StepReport:
    """What one macro step of the hot loop did (the driver's per-step record)."""

    __slots__ = ("tick0", "tick1", "chunks", "splits", "syncs", "contacts", "l0_ticks", "wall_s", "cpu_s",
                 "cpu_macro_s", "cpu_build_s", "cpu_absorb_s", "l0_steps")

    def __init__(self, tick0: int):
        self.tick0 = int(tick0)
        self.tick1 = int(tick0)
        self.chunks = 0
        self.splits: dict = {}
        self.syncs: dict = {}
        self.contacts: list = []
        self.l0_ticks: dict = {}
        self.wall_s = 0.0
        self.cpu_s = 0.0                # process CPU time of run_step (build, run_macro chunks, motion, absorb)
        self.cpu_macro_s = 0.0          # inside run_macro
        self.cpu_build_s = 0.0          # kernel_inputs of every domain and the contact layouts
        self.cpu_absorb_s = 0.0         # cutting, removal and dynamics absorb
        self.l0_steps = 0


class HotLoop:
    """Per macro step: the frozen inputs from every domain's kernel_inputs (motion, tooling, dynamics, cutting,
    removal), run_macro in chunks between the sync ticks (the feeder runs the controller there), drains and field
    growth at splits, then absorb on cutting, removal and dynamics (motion absorbs after every chunk, its
    protocol). docs/models/hotloop.md.

    ``feeder``: an object with ``feed(now_tick)`` (controller produce -> motion accept for every path) and
    ``apply(actions)`` (motion's controller actions: retire, feedback, cycle_start, program_end).

    ``use_dynamics=False`` runs with rigid tool and workpiece (zero deflection; dynamics' kernels are not called
    and its absorb gets an empty step). The loop's own state (contact phases, carried loads, polar views) is
    in :meth:`to_arrays` / :meth:`from_arrays` for checkpoints."""

    def __init__(self, ctx, state, *, motion, removal, cutting, tooling, dynamics=None, use_dynamics: bool = True,
                 recorder=None, l0_override: dict | None = None):
        self.ctx = ctx
        self.state = state
        self.motion = motion
        self.removal = removal
        self.cutting = cutting
        self.tooling = tooling
        self.dynamics = dynamics
        self.use_dynamics = bool(use_dynamics) and dynamics is not None
        self.recorder = recorder
        self.l0_override = dict(l0_override or {})
        self.cstate: dict = {}            # path -> (signature, ContactState)
        self.base0: dict = {}             # path -> base-excitation input at the step start
        self.views: dict = {}             # workpiece -> cutting PolarView kept one bin behind the tool
        self.g = None
        self.ring = None
        self.last = None                  # the step's built inputs (tests, absorb)
        self.reports: list = []
        self.keep_reports = False
        self._geo_cache: dict = {}
        self._placeholders: dict = {}     # padding contacts and rigid-dynamics placeholders (read by no kernel)
        self._phys = getattr(ctx.project, "physics", None) if ctx.project is not None else None

    # ------------------------------------------------------------------ settings
    def _physics(self, name, default):
        value = getattr(self._phys, name, None) if self._phys is not None else None
        return default if value is None else value

    def stride(self, p: int, slot: int, omega_max: float, n_teeth: int) -> int:
        """L0 stride of path p for the coming step (architecture 5.1): an override, else the largest divisor of
        ipo_ticks with |omega| dt <= max_deg_per_step, steps_per_tooth per tooth period and 2 pi f_max dt <= 0.3
        for the modes of the path's last dynamics system (dynamics' l0_ticks_rule)."""
        if p in self.l0_override:
            return int(self.l0_override[p])
        from ..dynamics import modal as MD
        from ..dynamics import params as DP
        from ..dynamics.domain import l0_ticks_rule
        ctx = self.ctx
        f_max = 0.0
        rec = getattr(self.state.dynamics, "systems", {}).get(p) if self.use_dynamics and \
            getattr(self.state, "dynamics", None) is not None else None
        if rec is not None and rec.modes:
            f_max = float(MD.f_max_resolved_Hz(rec.modes, rec.ref_station, DP.p(DP.MODES, "stride_compliance_share")))
        rad = self._physics("max_rad_per_step", 0.5 * math.pi / 180.0)
        spt = int(self._physics("steps_per_tooth", 96))
        omega_dt = DP.p(DP.MODES, "stride_omega_dt")
        return int(l0_ticks_rule(int(ctx.ipo_ticks), float(ctx.tick_s), float(omega_max), max(int(n_teeth), 1),
                                 f_max, float(rad), spt, float(omega_dt)))

    # ------------------------------------------------------------------ the step
    def run_step(self, feeder) -> StepReport:
        import time as _time
        t_wall = _time.perf_counter()
        t_cpu = _time.process_time()
        ctx, state = self.ctx, self.state
        md = self.motion
        tick0 = int(state.clock.tick)
        rep = StepReport(tick0)
        feeder.feed(tick0)
        mi = md.kernel_inputs(ctx, state, state.bus_prev)
        self._ensure_global(mi)
        g = self.g
        if int(mi.k_from[0]) == 0:
            mi.dist_force[:] = g.load_force[:mi.dist_force.shape[0]]
            mi.dist_torque[:] = g.load_torque[:mi.dist_torque.shape[0]]
        built = self._build(mi)
        rep.cpu_build_s = _time.process_time() - t_cpu
        self.last = built
        rm, cs = built["rm"], built["contacts"]
        g.ipar[H.GI_TICK0] = tick0
        g.ipar[H.GI_RESUME] = 0
        l0_before = int(g.count[H.CNT_L0])
        while True:
            self._check_tools(built)
            t_m = _time.process_time()
            code = int(run_macro(mi, rm, cs, self.ring, g))
            rep.cpu_macro_s += _time.process_time() - t_m
            rep.chunks += 1
            if code == H.HOT_SPLIT:
                reason = int(g.result[H.RES_REASON])
                rep.splits[reason] = rep.splits.get(reason, 0) + 1
                rm = self._handle_split(built, reason)
                continue
            rep.syncs[code] = rep.syncs.get(code, 0) + 1
            md.absorb(ctx, state, mi, state.bus_next)
            feeder.apply(md.controller_actions(state))
            if md.step_complete(state):
                break
            now = tick0 + int(state.motion.meta["k_next"]) * int(ctx.ipo_ticks)
            feeder.feed(now)
            mi = md.kernel_inputs(ctx, state, state.bus_prev)
        t_a = _time.process_time()
        self._absorb(built)
        rep.cpu_absorb_s = _time.process_time() - t_a
        rep.l0_steps = int(g.count[H.CNT_L0]) - l0_before
        rep.tick1 = tick0 + int(ctx.macro_ipo) * int(ctx.ipo_ticks)
        rep.contacts = [c for c in built["paths"]]
        rep.l0_ticks = dict(built["l0"])
        rep.wall_s = _time.perf_counter() - t_wall
        rep.cpu_s = _time.process_time() - t_cpu
        if self.keep_reports:
            self.reports.append(rep)
        return rep

    # ------------------------------------------------------------------ inputs
    def _ensure_global(self, mi) -> None:
        if self.g is not None:
            return
        ctx = self.ctx
        na = int(mi.servo.kv.shape[0])
        ns = int(mi.spindle.inertia.shape[0])
        P = int(mi.setpoint.mode.shape[1])
        g = H.empty_global(na, ns, P, int(ctx.macro_ipo))
        g.fpar[H.GF_TICK_S] = float(ctx.tick_s)
        g.fpar[H.GF_PROBE_TOL] = 1e-8
        g.ipar[H.GI_IPO_TICKS] = int(ctx.ipo_ticks)
        g.ipar[H.GI_MACRO_IPO] = int(ctx.macro_ipo)
        g.ipar[H.GI_N_AXES] = na
        g.ipar[H.GI_N_SPINDLES] = ns
        g.ipar[H.GI_N_PATHS] = P
        rs = getattr(self.state, "removal", None)
        settings = getattr(rs, "settings", None)
        g.ipar[H.GI_PROBE_RINGS] = int(getattr(settings, "probe_rings", 8))
        g.ipar[H.GI_PROBE_AROUND] = int(getattr(settings, "probe_around", 32))
        # tool tips at the start (FK of the current axis values)
        from ..motion.abi import FK_S_T0, eye_stack
        from ..motion.kernels import fk_poses
        poses = eye_stack(7)
        work = np.zeros((6, 4, 4))
        q = np.ascontiguousarray(mi.servo_state.load, dtype=float)
        for p in range(P):
            fk_poses(mi.chain, p, q, poses, work)
            g.tip0[p] = poses[FK_S_T0][:3, 3]
        restore = getattr(self, "_restore_g", None)
        if restore:
            for name, value in restore.items():
                getattr(g, name)[...] = value
            self._restore_g = None
        self.g = g
        if self.recorder is not None:
            self.ring = self.recorder.ring
        else:
            from .abi import ring_buffer
            self.ring = ring_buffer(2, 1)

    def _ring_col(self, name: str, p: int, width: int) -> int:
        rec = self.recorder
        if rec is None:
            return -1
        try:
            first, count = rec.fast_columns(name)
        except KeyError:
            return -1
        return int(first + width * p) if count >= width * (p + 1) else -1

    def _check_tools(self, built) -> None:
        """A contact whose path changed its tool inside the step (tool change) stops cutting for the rest of the step."""
        for c, p in enumerate(built["paths"]):
            ct = built["contacts"][c]
            if int(ct.ipar[H.CI_ACTIVE]) and int(self.motion.planners[p].st.tool_slot) != int(ct.ipar[H.CI_SLOT]):
                ct.ipar[H.CI_ACTIVE] = 0

    def _build(self, mi) -> dict:
        ctx, state = self.ctx, self.state
        bus = state.bus_prev
        md = self.motion
        P = int(mi.setpoint.mode.shape[1])
        frames = list(getattr(bus.motion, "paths", []) or [])
        # strides first: dynamics builds its systems at the stride
        l0 = {}
        teeth = {}
        for p in range(min(P, len(frames))):
            slot = int(frames[p].tool_slot)
            if slot < 0 or slot >= len(state.tooling):
                continue
            edge = state.tooling.effective[slot]
            n_teeth = int(np.asarray(edge.tooth_phase_rad).shape[0])
            teeth[p] = n_teeth
            s_tool = int(mi.chain.tool_spindle[p])
            omega = 0.0
            sp_index = s_tool if s_tool >= 0 else self._work_spindle(mi, p)
            if sp_index >= 0:
                omega = max(float(np.max(np.abs(mi.setpoint.sp_omega[:, sp_index]))),
                            float(abs(mi.spindle_state.omega[sp_index])))
            l0[p] = self.stride(p, slot, omega, n_teeth if s_tool >= 0 else 1)
            if self.use_dynamics:
                self.dynamics.set_l0_ticks(p, l0[p])
        ti = self.tooling.kernel_inputs(ctx, state, bus)
        di = self.dynamics.kernel_inputs(ctx, state, bus) if self.use_dynamics else None
        ci = self.cutting.kernel_inputs(ctx, state, bus)
        ri = self.removal.kernel_inputs(ctx, state, bus)
        rm = self._pad_removal(ri)
        contacts = []
        paths = []
        for cin in ci.contacts:
            p = int(cin.path)
            if p not in l0 or len(contacts) >= H.MAX_CONTACTS:
                continue
            contacts.append(self._contact(mi, ri, ti, di, cin, p, l0[p]))
            paths.append(p)
        n_c = len(contacts)
        n_axes = int(mi.servo.kv.shape[0])
        while len(contacts) < H.MAX_CONTACTS:
            contacts.append(self._padding(n_axes))
        g = self.g
        g.ipar[H.GI_N_CONTACTS] = n_c
        self._rows(contacts, n_c)
        g.tick_flags[:] = 0
        return {"rm": rm, "ri": ri, "ci": ci, "di": di, "ti": ti, "contacts": tuple(contacts), "paths": paths,
                "l0": l0, "drained": {c: {b: [] for b in range(7)} for c in range(n_c)}}

    def _padding(self, n_axes: int):
        """An inactive contact of the right types for the contact tuple (CI_ACTIVE = 0: run_macro never reads or
        writes it), made once per axis count."""
        key = ("pad", int(n_axes))
        pad = self._placeholders.get(key)
        if pad is None:
            pad = H.empty_contact(n_axes)
            self._placeholders[key] = pad
        return pad

    def _work_spindle(self, mi, p: int) -> int:
        from ..motion.abi import CHAIN_WORK
        wl = int(mi.chain.work_spindle_link[p])
        if wl < 0:
            return -1
        a = int(mi.chain.axis[p, CHAIN_WORK, wl])
        return int(mi.servo.spindle[a]) if a >= 0 else -1

    def _pad_removal(self, ri):
        from ..removal import nbt as RN
        from .abi import EventBuffer  # noqa: F401  (layout documented in hot_abi)

        def pad(items, n, filler):
            items = tuple(items)
            if len(items) > n:
                raise ValueError(f"the hot loop handles at most {n} entries here (core.hot_abi), got {len(items)}")
            return items + tuple(filler() for _ in range(n - len(items)))
        return RA.RemovalInputs(pad(ri.fields, H.MAX_WORK, RN.example_field), pad(ri.targets, H.MAX_WORK, RN.example_target),
                                pad(ri.polars, H.MAX_WORK, RN.example_polar), pad(ri.cut, H.MAX_PATHS, RN.example_profile),
                                pad(ri.whole, H.MAX_PATHS, RN.example_profile),
                                pad(ri.inserts, H.MAX_PATHS, RN.example_insert),
                                pad(ri.helix, H.MAX_PATHS, RN.example_helix), pad(ri.slabs, H.MAX_PATHS, RA.empty_slabs),
                                pad(ri.accum, H.MAX_PATHS, RN.example_accum), ri.events, ri.scratch, ri.iparams,
                                ri.fparams)

    def _geo(self, mi, cin, p: int, slot: int, slabs, edge, n_axes_path: list[int]):
        from ..cutting.memory import station_geometry
        tool = cin.tool
        key = (p, slot, id(tool), id(slabs))
        hit = self._geo_cache.get(key)
        if hit is not None:
            return hit
        radius, z, n_r, n_z = station_geometry(tool)
        nst = int(tool.st_z_m.shape[0])
        st_len = np.zeros(nst)
        active = np.zeros(nst, np.int8)
        for s in range(nst):
            lo, hi = int(tool.st_ptr[s]), int(tool.st_ptr[s + 1])
            if hi > lo:
                active[s] = 1
                st_len[s] = float(np.sum(tool.dS_m[tool.st_elem[lo:hi]]))
        nt = int(np.asarray(edge.tooth_phase_rad).shape[0])
        ns_max = int(max(1, np.max(slabs.n))) if slabs.n.size else 1
        a0 = np.zeros((max(nt, 1), ns_max))
        zm = np.zeros((max(nt, 1), ns_max))
        hand = float(np.asarray(edge.hand_sign)[0])
        for j in range(min(nt, slabs.n.shape[0])):
            valid = np.asarray(edge.elem_valid[j]) != 0
            zz = np.asarray(edge.elem_z_m[j], float)[valid]
            lag = np.asarray(edge.elem_lag_rad[j], float)[valid]
            order = np.argsort(zz, kind="stable")
            zz, lag = zz[order], lag[order]
            for k in range(int(slabs.n[j])):
                zmid = 0.5 * (float(slabs.z[j, k]) + float(slabs.z[j, k + 1]))
                zm[j, k] = zmid
                lk = float(np.interp(zmid, zz, lag)) if zz.size else 0.0
                a0[j, k] = hand * (float(edge.tooth_phase_rad[j]) + lk)
        # stations of every slab (by height; the nearest active station when none lies in it)
        ptr = [0]
        sts: list[int] = []
        act_idx = np.nonzero(active)[0]
        for j in range(max(nt, 1)):
            for k in range(ns_max):
                if j < slabs.n.shape[0] and k < int(slabs.n[j]):
                    z0s, z1s = float(slabs.z[j, k]), float(slabs.z[j, k + 1])
                    inside = [int(s_) for s_ in act_idx if z0s - 1e-12 <= z[s_] < z1s + 1e-12]
                    if not inside and act_idx.size:
                        inside = [int(act_idx[int(np.argmin(np.abs(z[act_idx] - zm[j, k])))])]
                    sts.extend(inside)
                ptr.append(len(sts))
        names = self.cutting.axis_names
        index = self.motion.model.axis_index
        cut_axes = np.array([int(index.get(n, -1)) for n in names] or [-1], np.int64)
        geo = H.ContactGeo(np.ascontiguousarray(z, float), np.ascontiguousarray(radius, float), st_len, active,
                           np.ascontiguousarray(n_r, float), np.ascontiguousarray(n_z, float), a0, zm,
                           np.array(ptr, np.int64), np.array(sts or [-1], np.int64),
                           np.array(n_axes_path or [0], np.int64), cut_axes)
        self._geo_cache = {k: v for k, v in self._geo_cache.items() if k[0] != p}
        self._geo_cache[key] = geo
        return geo

    def _contact(self, mi, ri, ti, di, cin, p: int, l0: int):
        from ..dynamics import abi as DA
        from ..removal import nbt as RN
        ctx, state = self.ctx, self.state
        slot = int(cin.slot)
        w = int(cin.workpiece)
        tool = cin.tool
        edge = state.tooling.effective[slot]
        geo_t = state.tooling.geometry(slot)
        nt = int(np.asarray(edge.tooth_phase_rad).shape[0])
        nst = int(tool.st_z_m.shape[0])
        slabs = ri.slabs[p]
        path_axes = list(self.motion.planners[p].nc_axes)
        geo = self._geo(mi, cin, p, slot, slabs, edge, path_axes)
        ns_max = int(geo.slab_a0.shape[1])
        sig = (slot, nt, ns_max)
        old = self.cstate.get(p)
        if old is None or old[0] != sig:
            cst = H.empty_contact_state(max(nt, 1), ns_max)
            cst.ival[H.CS_FIRST] = 1
            self.cstate[p] = (sig, cst)
        cst = self.cstate[p][1]
        ip = np.zeros(H.N_CI, np.int64)
        fp = np.zeros(H.N_CF)
        kind = int(ri.iparams[RA.RI_PATH0 + RA.RI_PSTRIDE * p + RA.RP_KIND])
        rotating = bool(cin.rotating)
        ip[H.CI_ACTIVE] = 1
        ip[H.CI_PATH] = p
        ip[H.CI_SLOT] = slot
        ip[H.CI_WORK] = w
        ip[H.CI_KIND] = kind
        ip[H.CI_ROTATING] = 1 if rotating else 0
        ip[H.CI_TOOL_SPINDLE] = int(mi.chain.tool_spindle[p]) if rotating else -1
        ip[H.CI_WORK_SPINDLE] = self._work_spindle(mi, p)
        ip[H.CI_N_TEETH] = nt
        ip[H.CI_N_ELEM] = int(tool.scal[CA.TC_N_ELEM])
        ip[H.CI_NST] = nst
        ip[H.CI_L0_TICKS] = int(l0)
        ip[H.CI_NS_MAX] = ns_max
        ip[H.CI_N_AXES] = len(path_axes)
        ip[H.CI_POLAR] = 1 if (w < len(ri.polars) and kind == RA.TK_TURNING) else 0
        ip[H.CI_STALE_BEFORE] = int(cin.istep[CA.SI_STALE_BEFORE])
        ptr = np.asarray(geo.slab_ptr, np.int64)
        ip[H.CI_ST_MARGIN] = 2 + (int(np.max(np.diff(ptr))) if ptr.shape[0] > 1 else 0)
        ip[H.CI_RING_FORCE] = self._ring_col(H.FAST_FORCE, p, 3)
        ip[H.CI_RING_TORQUE] = self._ring_col(H.FAST_TORQUE, p, 1)
        fp[H.CF_HAND] = float(np.asarray(edge.hand_sign)[0])
        fp[H.CF_R_CUT] = float(tool.scal[CA.TC_R_CUT])
        fp[H.CF_REACH] = 4.0 * float(tool.scal[CA.TC_R_CUT]) + 1e-3
        fp[H.CF_LATTICE] = float(ri.fields[w].lv_cell_m[0])
        fp[H.CF_GATE_SAMPLE] = float(ri.fparams[RA.RF_GATE_SAMPLE])
        fp[H.CF_GATE_TOL] = float(ri.fparams[RA.RF_GATE_TOL])
        fp[H.CF_GATE_MARGIN] = float(ri.fparams[RA.RF_GATE_MARGIN])
        fp[H.CF_SECTOR_MARGIN] = float(ri.fparams[RA.RF_SECTOR_MARGIN])
        fp[H.CF_COLLISION_TOL] = float(ri.fparams[RA.RF_COLLISION_TOL])
        fp[H.CF_FEED_FRAC] = FEED_DIRECTION_FRACTION
        fp[H.CF_WORK_SIGN] = 1.0
        fp[H.CF_MOUNT_X] = fp[H.CF_MOUNT_Y] = fp[H.CF_MOUNT_Z] = 1.0
        if not rotating:
            shank = np.asarray(getattr(getattr(geo_t, "props", None), "shank_dir_T", (-1.0, 0.0, 0.0)), float)
            (fp[H.CF_POLAR_SZ], fp[H.CF_POLAR_SX], fp[H.CF_WORK_SIGN], fp[H.CF_MOUNT_X], fp[H.CF_MOUNT_Y],
             fp[H.CF_MOUNT_Z]) = self._turning_signs(mi, p, slot, external=bool(shank[0] < -0.5),
                                                     hand=float(fp[H.CF_HAND]))
        gate = np.array([fp[H.CF_GATE_SAMPLE], fp[H.CF_GATE_TOL], fp[H.CF_GATE_MARGIN], fp[H.CF_SECTOR_MARGIN]])
        # cutting arguments; the event buffers get the headroom of a step (docs/models/hotloop.md "Splits")
        polar = self._view(ri, w, float(fp[H.CF_POLAR_SX])) if ip[H.CI_POLAR] else cin.polar
        defl = cin.defl
        if self.use_dynamics and di is not None and p in di.systems:
            md = di.systems[p].defl
            identity = fp[H.CF_MOUNT_X] == 1.0 and fp[H.CF_MOUNT_Y] == 1.0 and fp[H.CF_MOUNT_Z] == 1.0
            if identity and md.u_st_T0_m.shape == defl.u_st_T0_m.shape:
                # the tool's T0 is the chain's: cutting reads dynamics' station deflections in place (no copy)
                defl = CA.Deflection(md.u_st_T0_m, md.v_st_T0_m_s, defl.u_work_S_m, defl.v_work_S_m_s)
                ip[H.CI_DEFL_SHARED] = 1
        cut = H.CutArgs(tool, cin.table, cin.mem, cin.iv, cin.refill, cin.step, cin.istep, cin.pose, defl,
                        cin.occ, cin.pack, cin.axes, cin.st, cin.acc, cin.cacc, cin.out, cin.ev, polar)
        # dynamics
        if self.use_dynamics and di is not None and p in di.systems:
            sys = di.systems[p]
            base = di.axes[p]
            grid = di.grids[p]
            offset = np.ascontiguousarray(np.asarray(di.contact_offset_S[p], float))
            ip[H.CI_DYN] = 1
            ip[H.CI_RING_DYN] = int(di.ring_columns.get(p, -1))
        else:
            # rigid contact: placeholders of the right types that no kernel touches (CI_DYN = 0), made once
            key = ("rigid", nst, int(ctx.macro_ipo), max(len(path_axes), 1))
            held = self._placeholders.get(key)
            if held is None:
                held = (DA.empty_system(1, nst, 1, int(ctx.macro_ipo), 1), DA.empty_axes(max(len(path_axes), 1)),
                        DA.empty_work_grid())
                self._placeholders[key] = held
            sys, base, grid = held
            offset = np.zeros(3)
            ip[H.CI_DYN] = 0
            ip[H.CI_RING_DYN] = -1
        b0 = self.base0.get(p)
        if b0 is None or b0.shape != (DA.N_BASE,):
            b0 = np.zeros(DA.N_BASE)
            self.base0[p] = b0
        spec = RA.SubtractSpec(np.zeros(3), np.zeros((ns_max, 3)), np.zeros(ns_max + 1), np.zeros(ns_max), np.zeros(3),
                               np.zeros(3), np.zeros(RA.N_SI, np.int64), np.zeros(RA.N_SF))
        from ..motion.abi import N_FK, eye_stack
        return H.HotContact(ip, fp, cut, sys, base, grid, offset, geo, cst, spec, np.zeros(4), gate,
                            np.ascontiguousarray(gate[:3].copy()), np.zeros(H.WORK_SIZE), b0, np.zeros(DA.N_BASE),
                            np.zeros(max(len(path_axes), 1)), eye_stack(N_FK), np.zeros((6, 4, 4)),
                            np.zeros(int(mi.servo.kv.shape[0])), np.zeros(RA.N_PS), np.zeros(8),
                            np.zeros((H.REFILL_PAIRS + 1, 2), np.int64), RA.empty_profile(), np.zeros(8),
                            np.zeros((max(nt, 1), H.N_TW)))

    def _turning_signs(self, mi, p: int, slot: int, external: bool = True, hand: float = -1.0) -> tuple:
        """(sz, sx, work spindle sign, mx, my, mz) of a turning contact.

        Tool mount (docs/models/hotloop.md "Turning tool frame"): motion's chain ends in the turret station's frame
        (the machine's axes at home), tooling's non-rotating tools have x_T into the work (toward the axis for
        external tools, shank along -x_T; away from it for internal ones) and z_T from the tip into the tool body
        along the work axis (+S z for right-hand and neutral tools, which feed toward the chuck; -S z for
        left-hand ones). The mount (mx, my, mz) scales the chain's T0 axes into that frame (a proper rotation:
        identity or a half turn); decided where the tip is off the axis and kept per path and slot.
        sz, sx: z_T along +/- S z, x_T along +/- the radial direction of the tip (removal's PolarStep); the
        work spindle sign: the work spindle's +z along +/- S z."""
        from ..motion.abi import CHAIN_WORK, FK_S_T0, eye_stack
        from ..motion.kernels import fk_poses
        poses = eye_stack(7)
        work = np.zeros((6, 4, 4))
        q = np.ascontiguousarray(mi.servo_state.load, dtype=float).copy()
        fk_poses(mi.chain, p, q, poses, work)
        R = poses[FK_S_T0][:3, :3]
        tip = poses[FK_S_T0][:3, 3]
        er = np.array([tip[0], tip[1], 0.0])
        n = np.linalg.norm(er)
        sz_req = -1.0 if hand > 0.0 else 1.0
        sx_req = -1.0 if external else 1.0
        cache = getattr(self, "_mounts", None)
        if cache is None:
            cache = self._mounts = {}
        key = (int(p), int(slot))
        if n > 1e-4 or key not in cache:
            cz = 1.0 if R[2, 2] >= 0.0 else -1.0
            cx = (1.0 if float(R[:, 0] @ (er / n)) > 0.0 else -1.0) if n > 0.0 else sx_req
            mx, mz = sx_req * cx, sz_req * cz
            cache[key] = (mx, mx * mz, mz)
        mx, my, mz = cache[key]
        sz, sx = sz_req, sx_req
        sign = 1.0
        wl = int(mi.chain.work_spindle_link[p])
        if wl >= 0:
            a = int(mi.chain.axis[p, CHAIN_WORK, wl])
            if a >= 0:
                q2 = q.copy()
                q2[a] += 1e-3
                poses2 = eye_stack(7)
                fk_poses(mi.chain, p, q2, poses2, work)
                # the tool (fixed in M) turns by -dq about the work axis in S: its angle about S z decreases for +z
                t1 = poses2[FK_S_T0][:3, 3]
                d = math.atan2(t1[1], t1[0]) - math.atan2(tip[1], tip[0])
                d = (d + math.pi) % (2.0 * math.pi) - math.pi
                sign = 1.0 if d < 0.0 else -1.0
        return sz, sx, sign, mx, my, mz

    def _view(self, ri, w: int, side: float = -1.0):
        """Cutting's PolarView of workpiece w on the polar field's grid (built once from the field, then kept by
        the loop one bin behind the tool; rebuilt when the tool side changes: hot_kernels.polar_view_update)."""
        view = self.views.get(w)
        pf = ri.polars[w]
        nth = int(pf.geom[RA.PG_NTHETA])
        nz = int(pf.geom[RA.PG_NZ])
        sides = getattr(self, "_view_side", None)
        if sides is None:
            sides = self._view_side = {}
        if view is None or view.r_out_m.shape != (nth, nz) or sides.get(w) != side:
            sides[w] = side
            from .hot_kernels import polar_view_update
            view = CA.PolarView(np.zeros((nth, nz)), np.zeros((nth, nz)),
                                np.array([float(pf.geom[RA.PG_DTHETA]), float(pf.geom[RA.PG_Z0]),
                                          float(pf.geom[RA.PG_DZ]), 0.0]), np.zeros((nth, nz)), np.zeros((nth, nz)))
            polar_view_update(pf, view, 0, nth - 1, 0, nz - 1, float(side))
            self.views[w] = view
        return view

    def _rows(self, contacts: list, n_c: int) -> None:
        """Rows one L0 step may write per buffer (the headroom rule) and buffers large enough for several steps."""
        g = self.g
        n_el = 1
        nt = 1
        for ct in contacts[:n_c]:
            n_el = max(n_el, int(np.count_nonzero(ct.cut.tool.valid)))
            nt = max(nt, int(ct.ipar[H.CI_N_TEETH]))
        rows = {H.GI_ROWS_SHOCK: 4 * nt + 4, H.GI_ROWS_SURFACE: 3 * n_el + 8, H.GI_ROWS_DEPOSIT: 2 * nt + 4,
                H.GI_ROWS_RECUT: n_el + 4 * nt + 4, H.GI_ROWS_PASS: 2 * nt + 4, H.GI_ROWS_REFILL: 2 * n_el + 4,
                H.GI_ROWS_REMOVAL: 8 * nt + 8}
        for key, value in rows.items():
            g.ipar[key] = value
        need = 4 * max(rows.values())
        for c in range(n_c):
            ct = contacts[c]
            ev = ct.cut.ev
            if any(buf.rec.shape[0] < need for buf in ev):
                bigger = CA.CutEvents(*(type(buf)(np.zeros((max(need, buf.rec.shape[0]), buf.rec.shape[1])),
                                                  buf.n) for buf in ev))
                contacts[c] = ct._replace(cut=ct.cut._replace(ev=bigger))
        rs = self.state.removal
        if rs.events.rec.shape[0] < 4 * rows[H.GI_ROWS_REMOVAL]:
            from .abi import event_buffer
            rs.events = event_buffer(4 * rows[H.GI_ROWS_REMOVAL], RA.EVENT_WIDTH)

    # ------------------------------------------------------------------ splits
    def _handle_split(self, built, reason: int):
        g = self.g
        c = int(g.result[H.RES_CONTACT])
        if reason == H.SPLIT_DRAIN:
            self._drain(built, c)
            return built["rm"]
        if reason in (H.SPLIT_POOL, H.SPLIT_POLAR):
            removal = self.removal
            state = self.state
            removal.prepare_replay(self.ctx, state)
            rs = state.removal
            for wp in rs.workpieces:
                wp.field.info[RA.FI_OVERFLOW] = 0
                if wp.polar is not None:
                    wp.polar.info[RA.PI_OVERFLOW] = 0
            ri = rs.last_inputs._replace(fields=tuple(wp.field for wp in rs.workpieces),
                                         polars=tuple(wp.polar if wp.polar is not None else ri_p
                                                      for wp, ri_p in zip(rs.workpieces, rs.last_inputs.polars)))
            rs.last_inputs = ri
            built["ri"] = ri
            built["rm"] = self._pad_removal(ri)
            return built["rm"]
        raise RuntimeError(f"hot loop: unknown split reason {reason}")

    def _drain(self, built, c: int) -> None:
        """Move the rows of every event buffer of contact c (and removal's) out of the kernel buffers."""
        ct = built["contacts"][c]
        store = built["drained"][c]
        for b, buf in enumerate(ct.cut.ev):
            n = int(buf.n[0])
            if int(buf.n[1]):
                raise RuntimeError("hot loop: a cutting event buffer overflowed despite the headroom rule")
            if n:
                store[b].append(buf.rec[:n].copy())
                buf.n[0] = 0
        ev = self.state.removal.events
        n = int(ev.n[0])
        if n:
            built.setdefault("drained_removal", []).append(ev.rec[:n].copy())
            ev.n[0] = 0

    # ------------------------------------------------------------------ absorb
    def _absorb(self, built) -> None:
        from .abi import EventBuffer
        ctx, state = self.ctx, self.state
        contacts = built["contacts"]
        ci = built["ci"]
        by_path = {int(c.ipar[H.CI_PATH]): k for k, c in enumerate(contacts[:len(built["paths"])])}
        merged = []
        for cin in ci.contacts:
            k = by_path.get(int(cin.path))
            if k is None:
                merged.append(cin)
                continue
            ct = contacts[k]
            bufs = []
            for b, buf in enumerate(ct.cut.ev):
                parts = built["drained"][k][b] + [buf.rec[:int(buf.n[0])]]
                rows = np.concatenate(parts, axis=0) if parts else np.zeros((0, buf.rec.shape[1]))
                cap = max(rows.shape[0], 1)
                rec = np.zeros((cap, buf.rec.shape[1]))
                rec[:rows.shape[0]] = rows
                bufs.append(EventBuffer(rec, np.array([rows.shape[0], int(buf.n[1])], np.int64)))
            merged.append(cin._replace(ev=CA.CutEvents(*bufs)))
        from ..cutting.domain import CuttingInputs
        self.cutting.absorb(ctx, state, CuttingInputs(tuple(merged)), state.bus_next)
        rs = state.removal
        keep = rs.events
        extra = built.get("drained_removal", [])
        if extra:
            rows = np.concatenate(extra + [keep.rec[:int(keep.n[0])]], axis=0)
            rs.events = EventBuffer(np.ascontiguousarray(rows) if rows.shape[0] else np.zeros((1, RA.EVENT_WIDTH)),
                                    np.array([rows.shape[0], int(keep.n[1])], np.int64))
        self.removal.absorb(ctx, state, None, state.bus_next)
        if extra:
            rs.events = keep
        if self.use_dynamics:
            self.dynamics.absorb(ctx, state, _Accums([cin.acc for cin in ci.contacts]), state.bus_next)
        else:
            from .bus import DynamicsFrame
            state.bus_next.dynamics = DynamicsFrame()

    # ------------------------------------------------------------------ checkpoint
    def to_arrays(self) -> tuple[dict, dict]:
        """The loop's own state (bit-identical continuation): contact phases and flags, the carried L1 loads and
        tips, the base-excitation inputs and the polar views."""
        arrays: dict = {}
        meta: dict = {"paths": [], "views": sorted(int(w) for w in self.views),
                      "mounts": [[int(k[0]), int(k[1]), *map(float, v)] for k, v in
                                 sorted(getattr(self, "_mounts", {}).items())],
                      "view_sides": [[int(w), float(v)] for w, v in sorted(getattr(self, "_view_side", {}).items())]}
        if self.g is not None:
            for name in ("load_force", "load_torque", "tip0", "count"):
                arrays[f"g.{name}"] = np.array(getattr(self.g, name))
        for p, (sig, cst) in sorted(self.cstate.items()):
            meta["paths"].append({"path": int(p), "sig": [int(v) for v in sig]})
            for name in cst._fields:
                arrays[f"c{p}.{name}"] = np.array(getattr(cst, name))
        for p, b0 in sorted(self.base0.items()):
            arrays[f"b{p}"] = np.array(b0)
        for w, view in sorted(self.views.items()):
            for name in view._fields:
                arrays[f"v{w}.{name}"] = np.array(getattr(view, name))
        return arrays, meta

    def from_arrays(self, arrays: dict, meta: dict) -> None:
        self.cstate = {}
        for entry in meta.get("paths", []):
            p = int(entry["path"])
            cst = H.ContactState(*(np.ascontiguousarray(np.array(arrays[f"c{p}.{name}"]))
                                   for name in H.ContactState._fields))
            self.cstate[p] = (tuple(entry["sig"]), cst)
        self.base0 = {int(k[1:]): np.ascontiguousarray(np.array(v)) for k, v in arrays.items()
                      if k.startswith("b") and k[1:].isdigit()}
        self.views = {}
        for w in meta.get("views", []):
            self.views[int(w)] = CA.PolarView(*(np.ascontiguousarray(np.array(arrays[f"v{w}.{name}"]))
                                                for name in CA.PolarView._fields))
        self._restore_g = {name: np.array(arrays[f"g.{name}"]) for name in ("load_force", "load_torque", "tip0", "count")
                           if f"g.{name}" in arrays}
        self._mounts = {(int(m[0]), int(m[1])): (float(m[2]), float(m[3]), float(m[4]))
                        for m in meta.get("mounts", [])}
        self._view_side = {int(w): float(v) for w, v in meta.get("view_sides", [])}
        self.g = None
        self._geo_cache = {}


class _Accums:
    def __init__(self, accums):
        self.accums = accums


FEED_DIRECTION_FRACTION = 0.05      # |v_xy| / |v| below which a pass's generation angle is fixed (axial feed)
