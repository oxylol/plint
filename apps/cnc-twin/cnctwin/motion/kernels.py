"""Compiled kernels of the motion domain (docs/models/motion.md "Kernels").

L1 (called by ``core.hotloop.run_macro`` once per L1 tick, or by :func:`run_motion` when motion
runs alone):

* :func:`motion_tick` - one L1 tick of every path: spindles (speed loop with CSS and ramp, position
  loop, C-axis mode), spindle-synchronous setpoints (per revolution, threading) from the actual
  spindle angle, feed drives (position loop, velocity feed-forward, PI velocity loop on the moving
  mass, Coulomb friction, backlash and its compensation, force limit), kinematics (tool pose in M,
  W0 and S with geometric and thermal error transforms), travel and following-error checks,
  in-position detection and the sync events.
* :func:`fk_poses` - the kinematic chain of one path (callable per L0 step by the hot loop).
* :func:`spin_z` - right-multiply a pose by a rotation about its z axis (tool frame T from T0).

Planner kernels (called from Python per macro step):

* :func:`plan_ticks` - jerk-limited look-ahead path velocity generator over a window of pieces
  (time-optimal bang-bang jerk with exact switching on the braking curves of every node ahead).
* :func:`brake_distance` - minimum distance from (v, a) to (u, 0) with |a| <= A, |j| <= J.
* :func:`geo_eval_many` - positions of the path axes on a geometry piece.

Every kernel has a plain-Python twin in :mod:`cnctwin.motion.reference`; the tests compare them.
Rules (architecture 5.5): explicit signatures, cache, nogil, error_model numpy, no fastmath.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, typeof, types

from .abi import (ALARM_FOLLOWING_ERROR, ALARM_OVERTRAVEL_NEG, ALARM_OVERTRAVEL_POS, FK_M_F, FK_M_N, FK_M_T0, FK_M_W,
                  FK_M_W0, FK_S_T0, FK_W0_T0, GEO_ARC, LINK_LINEAR, LINK_ROTARY, SP_END, SP_HOLD_WAIT, SP_PER_REV,
                  SP_THREAD, SPM_CAXIS, SPM_POSITION, SPM_SPEED, SYNC_ALARM, SYNC_IN_POSITION, SYNC_NONE,
                  SYNC_PATH_WAIT, SYNC_PROBE_TRIGGER, SYNC_QUEUE_EMPTY, SYNC_SEGMENT_END, WAIT_PATH, empty_chain,
                  empty_loop_state, empty_out, empty_servo, empty_servo_state, empty_setpoint, empty_spindle,
                  empty_spindle_state)

OPTS = dict(cache=True, nogil=True, error_model="numpy")
# no reference counting in fk_poses (it allocates nothing: every array is an argument; removal.nbt RAW)
RAW = dict(OPTS, _nrt=False)

CHAIN_T = typeof(empty_chain())
SERVO_T = typeof(empty_servo())
SSTATE_T = typeof(empty_servo_state())
SPINDLE_T = typeof(empty_spindle())
SPSTATE_T = typeof(empty_spindle_state())
SETP_T = typeof(empty_setpoint())
LOOP_T = typeof(empty_loop_state())
OUT_T = typeof(empty_out())

F1 = types.float64[::1]
F2 = types.float64[:, ::1]
F3 = types.float64[:, :, ::1]
I1 = types.int32[::1]
I8_1 = types.int64[::1]
B1 = types.int8[::1]
f8 = types.float64
i4 = types.int32
i8 = types.int64

TWO_PI = 2.0 * math.pi
# generator numerics (pure numerics, not physics: see docs/models/motion.md "Generator")
_BISECT = 14                 # bisection steps on the jerk (resolution J * 2^-14)
_SNAP_M = 2e-9               # a stop node closer than this with v below _SNAP_V is reached
_SNAP_V = 1e-6               # m/s
_TOL_M = 1e-12               # distance tolerance of the feasibility tests


# =============================================================================== small linear algebra

@njit(types.void(F1, f8, F2), **OPTS)
def axis_rotation(d, angle, out):
    """out[:3, :3] = rotation by ``angle`` about the unit vector d (Rodrigues), right-handed."""
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    x = d[0]
    y = d[1]
    z = d[2]
    out[0, 0] = c + x * x * t
    out[0, 1] = x * y * t - z * s
    out[0, 2] = x * z * t + y * s
    out[1, 0] = y * x * t + z * s
    out[1, 1] = c + y * y * t
    out[1, 2] = y * z * t - x * s
    out[2, 0] = z * x * t - y * s
    out[2, 1] = z * y * t + x * s
    out[2, 2] = c + z * z * t


@njit(types.void(f8, f8, f8, F2), **OPTS)
def rotvec_rotation(rx, ry, rz, out):
    """out[:3, :3] = exact rotation of the rotation vector (rx, ry, rz)."""
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle < 1e-300:
        for i in range(3):
            for j in range(3):
                out[i, j] = 1.0 if i == j else 0.0
        return
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    x = rx / angle
    y = ry / angle
    z = rz / angle
    out[0, 0] = c + x * x * t
    out[0, 1] = x * y * t - z * s
    out[0, 2] = x * z * t + y * s
    out[1, 0] = y * x * t + z * s
    out[1, 1] = c + y * y * t
    out[1, 2] = y * z * t - x * s
    out[2, 0] = z * x * t - y * s
    out[2, 1] = z * y * t + x * s
    out[2, 2] = c + z * z * t


@njit(types.void(F2, F2, F2), **OPTS)
def mul4(a, b, out):
    """out = a @ b for rigid 4x4 poses (out may not alias a or b)."""
    for i in range(3):
        for j in range(4):
            out[i, j] = a[i, 0] * b[0, j] + a[i, 1] * b[1, j] + a[i, 2] * b[2, j] + (a[i, 3] if j == 3 else 0.0)
    out[3, 0] = 0.0
    out[3, 1] = 0.0
    out[3, 2] = 0.0
    out[3, 3] = 1.0


@njit(types.void(F2, F2), **OPTS)
def inv4(a, out):
    """out = inverse of the rigid pose a."""
    for i in range(3):
        for j in range(3):
            out[i, j] = a[j, i]
    for i in range(3):
        out[i, 3] = -(a[0, i] * a[0, 3] + a[1, i] * a[1, 3] + a[2, i] * a[2, 3])
    out[3, 0] = 0.0
    out[3, 1] = 0.0
    out[3, 2] = 0.0
    out[3, 3] = 1.0


@njit(types.void(F2, F2), **OPTS)
def copy4(a, out):
    for i in range(4):
        for j in range(4):
            out[i, j] = a[i, j]


@njit(types.void(F2), **OPTS)
def eye4(out):
    for i in range(4):
        for j in range(4):
            out[i, j] = 1.0 if i == j else 0.0


@njit(types.void(F2, f8, F2), **OPTS)
def spin_z(pose, angle, out):
    """out = pose @ Rz(angle) (the rotating tool frame T from T0)."""
    c = math.cos(angle)
    s = math.sin(angle)
    for i in range(4):
        x = pose[i, 0]
        y = pose[i, 1]
        out[i, 0] = c * x + s * y
        out[i, 1] = -s * x + c * y
        out[i, 2] = pose[i, 2]
        out[i, 3] = pose[i, 3]


# =============================================================================== kinematic chains

@njit(types.void(CHAIN_T, i8, i8, i8, F1, i8, F2, F2, F2), **OPTS)
def link_motion(chain, p, c, i, q, zero_link, out, rot, tmp):
    """Joint motion x error transform x fixed pose of link i of chain c of path p (4x4 into out; rot and tmp
    are 4x4 scratch)."""
    kind = chain.kind[p, c, i]
    eye4(out)
    qv = 0.0
    if kind == LINK_LINEAR or kind == LINK_ROTARY:
        ax = chain.axis[p, c, i]
        if ax >= 0 and i != zero_link:
            qv = q[ax]
        d = chain.direction[p, c, i]
        if kind == LINK_LINEAR:
            out[0, 3] = d[0] * qv
            out[1, 3] = d[1] * qv
            out[2, 3] = d[2] * qv
        else:
            axis_rotation(d, qv, rot)
            pt = chain.point[p, c, i]
            for r in range(3):
                for k in range(3):
                    out[r, k] = rot[r, k]
                out[r, 3] = pt[r] - (rot[r, 0] * pt[0] + rot[r, 1] * pt[1] + rot[r, 2] * pt[2])
    # error transform: constant part + position polynomial + thermal screw strain
    e0 = chain.err_const[p, c, i, 0]
    e1 = chain.err_const[p, c, i, 1]
    e2 = chain.err_const[p, c, i, 2]
    e3 = chain.err_const[p, c, i, 3]
    e4 = chain.err_const[p, c, i, 4]
    e5 = chain.err_const[p, c, i, 5]
    if kind == LINK_LINEAR or kind == LINK_ROTARY:
        xi = (qv - chain.err_lo[p, c, i]) / chain.err_span[p, c, i]
        if xi < 0.0:
            xi = 0.0
        elif xi > 1.0:
            xi = 1.0
        poly = chain.err_poly[p, c, i]
        e0 += poly[0, 0] + xi * (poly[0, 1] + xi * poly[0, 2])
        e1 += poly[1, 0] + xi * (poly[1, 1] + xi * poly[1, 2])
        e2 += poly[2, 0] + xi * (poly[2, 1] + xi * poly[2, 2])
        e3 += poly[3, 0] + xi * (poly[3, 1] + xi * poly[3, 2])
        e4 += poly[4, 0] + xi * (poly[4, 1] + xi * poly[4, 2])
        e5 += poly[5, 0] + xi * (poly[5, 1] + xi * poly[5, 2])
        if kind == LINK_LINEAR:
            grow = chain.strain[p, c, i] * (qv - chain.strain_ref[p, c, i])
            d = chain.direction[p, c, i]
            e0 += d[0] * grow
            e1 += d[1] * grow
            e2 += d[2] * grow
    if e0 != 0.0 or e1 != 0.0 or e2 != 0.0 or e3 != 0.0 or e4 != 0.0 or e5 != 0.0:
        eye4(tmp)
        rotvec_rotation(e3, e4, e5, tmp)
        tmp[0, 3] = e0
        tmp[1, 3] = e1
        tmp[2, 3] = e2
        copy4(out, rot)
        mul4(rot, tmp, out)
    copy4(out, rot)
    mul4(rot, chain.fixed[p, c, i], out)


@njit(types.void(CHAIN_T, i8, i8, F1, i8, F2, F3), **OPTS)
def chain_pose(chain, p, c, q, zero_link, out, work):
    """Pose of the end of chain c of path p in M for axis values q (zero_link: joint taken at 0)."""
    eye4(out)
    for i in range(chain.n_links[p, c]):
        link_motion(chain, p, c, i, q, zero_link, work[0], work[1], work[2])
        copy4(out, work[3])
        mul4(work[3], work[0], out)


@njit(types.void(CHAIN_T, i8, F1, F3, F3), **RAW)
def fk_poses(chain, p, q, out, work):
    """FK of path p: out[FK_*] (4x4 poses). work: scratch [>= 6, 4, 4].

    The poses of chain_pose / link_motion / mul4 / inv4, bit for bit (the same operations in the same order,
    written out in this one function): it runs twice per L0 step in the hot loop, where each call of a helper
    (an array descriptor copy and a reference count per array argument) cost more than its arithmetic; compiled
    without reference counting (it allocates nothing)."""
    lnk = work[0]
    rot = work[1]
    tmp = work[2]
    prv = work[3]
    zl = chain.work_spindle_link[p]
    for sel in range(3):
        if sel == 2 and zl < 0:
            break
        c = 1 if sel == 0 else 0
        zero_link = -1 if sel < 2 else zl
        dst = out[FK_M_N] if sel == 0 else (out[FK_M_F] if sel == 1 else work[4])
        # ------------------------------------------------ chain_pose: dst = product of the link transforms
        for a in range(4):
            for b in range(4):
                dst[a, b] = 1.0 if a == b else 0.0
        for i in range(chain.n_links[p, c]):
            # ------------------------------------------ link_motion into lnk (rot, tmp scratch)
            kind = chain.kind[p, c, i]
            for a in range(4):
                for b in range(4):
                    lnk[a, b] = 1.0 if a == b else 0.0
            qv = 0.0
            if kind == LINK_LINEAR or kind == LINK_ROTARY:
                ax = chain.axis[p, c, i]
                if ax >= 0 and i != zero_link:
                    qv = q[ax]
                d = chain.direction[p, c, i]
                if kind == LINK_LINEAR:
                    lnk[0, 3] = d[0] * qv
                    lnk[1, 3] = d[1] * qv
                    lnk[2, 3] = d[2] * qv
                else:
                    cq = math.cos(qv)
                    sq = math.sin(qv)
                    t = 1.0 - cq
                    x = d[0]
                    y = d[1]
                    z = d[2]
                    rot[0, 0] = cq + x * x * t
                    rot[0, 1] = x * y * t - z * sq
                    rot[0, 2] = x * z * t + y * sq
                    rot[1, 0] = y * x * t + z * sq
                    rot[1, 1] = cq + y * y * t
                    rot[1, 2] = y * z * t - x * sq
                    rot[2, 0] = z * x * t - y * sq
                    rot[2, 1] = z * y * t + x * sq
                    rot[2, 2] = cq + z * z * t
                    pt = chain.point[p, c, i]
                    for r in range(3):
                        for k in range(3):
                            lnk[r, k] = rot[r, k]
                        lnk[r, 3] = pt[r] - (rot[r, 0] * pt[0] + rot[r, 1] * pt[1] + rot[r, 2] * pt[2])
            e0 = chain.err_const[p, c, i, 0]
            e1 = chain.err_const[p, c, i, 1]
            e2 = chain.err_const[p, c, i, 2]
            e3 = chain.err_const[p, c, i, 3]
            e4 = chain.err_const[p, c, i, 4]
            e5 = chain.err_const[p, c, i, 5]
            if kind == LINK_LINEAR or kind == LINK_ROTARY:
                xi = (qv - chain.err_lo[p, c, i]) / chain.err_span[p, c, i]
                if xi < 0.0:
                    xi = 0.0
                elif xi > 1.0:
                    xi = 1.0
                poly = chain.err_poly[p, c, i]
                e0 += poly[0, 0] + xi * (poly[0, 1] + xi * poly[0, 2])
                e1 += poly[1, 0] + xi * (poly[1, 1] + xi * poly[1, 2])
                e2 += poly[2, 0] + xi * (poly[2, 1] + xi * poly[2, 2])
                e3 += poly[3, 0] + xi * (poly[3, 1] + xi * poly[3, 2])
                e4 += poly[4, 0] + xi * (poly[4, 1] + xi * poly[4, 2])
                e5 += poly[5, 0] + xi * (poly[5, 1] + xi * poly[5, 2])
                if kind == LINK_LINEAR:
                    grow = chain.strain[p, c, i] * (qv - chain.strain_ref[p, c, i])
                    d = chain.direction[p, c, i]
                    e0 += d[0] * grow
                    e1 += d[1] * grow
                    e2 += d[2] * grow
            if e0 != 0.0 or e1 != 0.0 or e2 != 0.0 or e3 != 0.0 or e4 != 0.0 or e5 != 0.0:
                for a in range(4):
                    for b in range(4):
                        tmp[a, b] = 1.0 if a == b else 0.0
                angle = math.sqrt(e3 * e3 + e4 * e4 + e5 * e5)
                if angle < 1e-300:
                    for a in range(3):
                        for b in range(3):
                            tmp[a, b] = 1.0 if a == b else 0.0
                else:
                    cq = math.cos(angle)
                    sq = math.sin(angle)
                    t = 1.0 - cq
                    x = e3 / angle
                    y = e4 / angle
                    z = e5 / angle
                    tmp[0, 0] = cq + x * x * t
                    tmp[0, 1] = x * y * t - z * sq
                    tmp[0, 2] = x * z * t + y * sq
                    tmp[1, 0] = y * x * t + z * sq
                    tmp[1, 1] = cq + y * y * t
                    tmp[1, 2] = y * z * t - x * sq
                    tmp[2, 0] = z * x * t - y * sq
                    tmp[2, 1] = z * y * t + x * sq
                    tmp[2, 2] = cq + z * z * t
                tmp[0, 3] = e0
                tmp[1, 3] = e1
                tmp[2, 3] = e2
                # lnk = lnk @ tmp (copy4 to rot, mul4)
                for a in range(4):
                    for b in range(4):
                        rot[a, b] = lnk[a, b]
                for a in range(3):
                    for b in range(4):
                        lnk[a, b] = (rot[a, 0] * tmp[0, b] + rot[a, 1] * tmp[1, b] + rot[a, 2] * tmp[2, b]
                                     + (rot[a, 3] if b == 3 else 0.0))
                lnk[3, 0] = 0.0
                lnk[3, 1] = 0.0
                lnk[3, 2] = 0.0
                lnk[3, 3] = 1.0
            # lnk = lnk @ fixed
            fx = chain.fixed[p, c, i]
            for a in range(4):
                for b in range(4):
                    rot[a, b] = lnk[a, b]
            for a in range(3):
                for b in range(4):
                    lnk[a, b] = (rot[a, 0] * fx[0, b] + rot[a, 1] * fx[1, b] + rot[a, 2] * fx[2, b]
                                 + (rot[a, 3] if b == 3 else 0.0))
            lnk[3, 0] = 0.0
            lnk[3, 1] = 0.0
            lnk[3, 2] = 0.0
            lnk[3, 3] = 1.0
            # ------------------------------------------ dst = dst @ lnk (copy4 to prv, mul4)
            for a in range(4):
                for b in range(4):
                    prv[a, b] = dst[a, b]
            for a in range(3):
                for b in range(4):
                    dst[a, b] = (prv[a, 0] * lnk[0, b] + prv[a, 1] * lnk[1, b] + prv[a, 2] * lnk[2, b]
                                 + (prv[a, 3] if b == 3 else 0.0))
            dst[3, 0] = 0.0
            dst[3, 1] = 0.0
            dst[3, 2] = 0.0
            dst[3, 3] = 1.0
        # ------------------------------------------------ the chain's end frame times its pose
        if sel == 0:
            m = chain.pose_N_T0[p]
            res = out[FK_M_T0]
        elif sel == 1:
            m = chain.pose_F_W[p]
            res = out[FK_M_W]
        else:
            m = chain.pose_F_W[p]
            res = out[FK_M_W0]
        for a in range(3):
            for b in range(4):
                res[a, b] = dst[a, 0] * m[0, b] + dst[a, 1] * m[1, b] + dst[a, 2] * m[2, b] + (dst[a, 3] if b == 3 else 0.0)
        res[3, 0] = 0.0
        res[3, 1] = 0.0
        res[3, 2] = 0.0
        res[3, 3] = 1.0
    mw = out[FK_M_W]
    mw0 = out[FK_M_W0]
    if zl < 0:
        for a in range(4):
            for b in range(4):
                mw0[a, b] = mw[a, b]
    # S_T0 = (M_W W_S)^-1 M_T0
    ws = chain.pose_W_S[p]
    w4 = work[4]
    w5 = work[5]
    mt = out[FK_M_T0]
    for a in range(3):
        for b in range(4):
            w4[a, b] = mw[a, 0] * ws[0, b] + mw[a, 1] * ws[1, b] + mw[a, 2] * ws[2, b] + (mw[a, 3] if b == 3 else 0.0)
    w4[3, 0] = 0.0
    w4[3, 1] = 0.0
    w4[3, 2] = 0.0
    w4[3, 3] = 1.0
    for g in range(2):
        src = w4 if g == 0 else mw0
        dst = out[FK_S_T0] if g == 0 else out[FK_W0_T0]
        # w5 = inverse of the rigid pose src
        for a in range(3):
            for b in range(3):
                w5[a, b] = src[b, a]
        for a in range(3):
            w5[a, 3] = -(src[0, a] * src[0, 3] + src[1, a] * src[1, 3] + src[2, a] * src[2, 3])
        w5[3, 0] = 0.0
        w5[3, 1] = 0.0
        w5[3, 2] = 0.0
        w5[3, 3] = 1.0
        for a in range(3):
            for b in range(4):
                dst[a, b] = w5[a, 0] * mt[0, b] + w5[a, 1] * mt[1, b] + w5[a, 2] * mt[2, b] + (w5[a, 3] if b == 3 else 0.0)
        dst[3, 0] = 0.0
        dst[3, 1] = 0.0
        dst[3, 2] = 0.0
        dst[3, 3] = 1.0


@njit(types.void(CHAIN_T, i8, F1, F3, F3, F1), **OPTS)
def tip_in_W(chain, p, q, poses, work, out):
    """Tool tip of path p in the workpiece frame W for axis values q (out[3])."""
    fk_poses(chain, p, q, poses, work)
    mw = poses[FK_M_W]
    mt = poses[FK_M_T0]
    for r in range(3):
        out[r] = (mw[0, r] * (mt[0, 3] - mw[0, 3]) + mw[1, r] * (mt[1, 3] - mw[1, 3])
                  + mw[2, r] * (mt[2, 3] - mw[2, 3]))


@njit(f8(CHAIN_T, i8, F1, F1, I1, i8, F3, F3), **OPTS)
def ik_position(chain, p, q, target, lin, n_iter, poses, work):
    """Newton on the linear axes ``lin`` (at most 3) putting the tool tip at target (W); q is updated in place
    (rotary axes as given). The tip is affine in the linear axes up to the angular error motions, so the
    finite-difference Jacobian is exact to rounding and two or three steps converge. Returns the residual (m)."""
    n = lin.shape[0]
    tip = np.empty(3)
    tip_h = np.empty(3)
    jac = np.zeros((3, 3))
    res = 0.0
    for _ in range(n_iter):
        tip_in_W(chain, p, q, poses, work, tip)
        r0 = tip[0] - target[0]
        r1 = tip[1] - target[1]
        r2 = tip[2] - target[2]
        res = math.sqrt(r0 * r0 + r1 * r1 + r2 * r2)
        if res < 1e-13 or n == 0:
            break
        h = 1e-6
        for c in range(n):
            a = lin[c]
            keep = q[a]
            q[a] = keep + h
            tip_in_W(chain, p, q, poses, work, tip_h)
            q[a] = keep
            for r in range(3):
                jac[r, c] = (tip_h[r] - tip[r]) / h
        # least squares J step = -r through the normal equations (n <= 3), Gaussian elimination with pivoting
        m = np.zeros((3, 4))
        for i in range(n):
            for j in range(n):
                m[i, j] = jac[0, i] * jac[0, j] + jac[1, i] * jac[1, j] + jac[2, i] * jac[2, j]
            m[i, 3] = -(jac[0, i] * r0 + jac[1, i] * r1 + jac[2, i] * r2)
        for col in range(n):
            piv = col
            for row in range(col + 1, n):
                if abs(m[row, col]) > abs(m[piv, col]):
                    piv = row
            if abs(m[piv, col]) < 1e-300:
                return res
            if piv != col:
                for j in range(4):
                    tmp = m[col, j]
                    m[col, j] = m[piv, j]
                    m[piv, j] = tmp
            for row in range(col + 1, n):
                f = m[row, col] / m[col, col]
                for j in range(col, 4):
                    m[row, j] -= f * m[col, j]
        step = np.zeros(3)
        for i in range(n - 1, -1, -1):
            acc = m[i, 3]
            for j in range(i + 1, n):
                acc -= m[i, j] * step[j]
            step[i] = acc / m[i, i]
        for c in range(n):
            q[lin[c]] += step[c]
    tip_in_W(chain, p, q, poses, work, tip)
    r0 = tip[0] - target[0]
    r1 = tip[1] - target[1]
    r2 = tip[2] - target[2]
    return math.sqrt(r0 * r0 + r1 * r1 + r2 * r2)


# =============================================================================== feed drives

@njit(types.void(SERVO_T, i8, f8, f8, f8, f8, f8, f8, f8, F1), **OPTS)
def _axis_rhs(servo, a, x, v, integ, xc, load, vff, kv, out):
    """d/dt (x, v, integ) of one axis into out[0:3]; out[3] = motor force. out[4] is an input: the external
    force on the axis (N along +axis), which the caller stores there (out has >= 5 entries)."""
    meas = load if servo.scale_fb[a] == 1 else x
    vref = kv * (xc - meas) + vff
    ev = vref - v
    fu = servo.kp_v[a] * ev + integ
    fmax = servo.force_max[a]
    fm = fu
    dint = servo.ki_v[a] * ev
    if fu > fmax:
        fm = fmax
        if ev > 0.0:
            dint = 0.0
    elif fu < -fmax:
        fm = -fmax
        if ev < 0.0:
            dint = 0.0
    vs = servo.v_stribeck[a]
    fric = servo.friction[a] * math.tanh(v / vs) if vs > 0.0 else 0.0
    out[0] = v
    out[1] = (fm + out[4] + servo.gravity[a] - fric) / servo.mass[a]
    out[2] = dint
    out[3] = fm


@njit(types.void(SERVO_T, SSTATE_T, i8, f8, f8, f8, F1), **OPTS)
def servo_axis_step(servo, st, a, cmd, kv, f_ext, work):
    """Advance axis a over one L1 tick to the command ``cmd`` (fine-interpolated linearly inside the
    tick), with external force f_ext (N along +axis) and loop gain kv. work: scratch [>= 25]."""
    dt = servo.dt[0]
    n = servo.n_sub[0]
    h = dt / n
    c0 = st.cmd[a]
    dc = cmd - c0
    # backlash compensation: the control adds +-comp/2 when the commanded direction reverses
    if dc > 1e-15:
        dnew = 1.0
    elif dc < -1e-15:
        dnew = -1.0
    else:
        dnew = st.dirn[a]
    if dnew != 0.0 and dnew != st.dirn[a]:
        st.dirn[a] = dnew
        st.comp[a] = 0.5 * servo.comp[a] * dnew
    vff = servo.kff[a] * dc / dt
    comp = st.comp[a]
    x = st.x[a]
    v = st.v[a]
    integ = st.integ[a]
    load = st.load[a]
    half = 0.5 * servo.backlash[a]
    v_start = v
    k1 = work[0:5]
    k2 = work[5:10]
    k3 = work[10:15]
    k4 = work[15:20]
    k1[4] = f_ext
    k2[4] = f_ext
    k3[4] = f_ext
    k4[4] = f_ext
    fm = st.force[a]
    for sub in range(n):
        xa = c0 + dc * (sub / n) + comp
        xm = c0 + dc * ((sub + 0.5) / n) + comp
        xb = c0 + dc * ((sub + 1.0) / n) + comp
        _axis_rhs(servo, a, x, v, integ, xa, load, vff, kv, k1)
        _axis_rhs(servo, a, x + 0.5 * h * k1[0], v + 0.5 * h * k1[1], integ + 0.5 * h * k1[2], xm, load, vff, kv, k2)
        _axis_rhs(servo, a, x + 0.5 * h * k2[0], v + 0.5 * h * k2[1], integ + 0.5 * h * k2[2], xm, load, vff, kv, k3)
        _axis_rhs(servo, a, x + h * k3[0], v + h * k3[1], integ + h * k3[2], xb, load, vff, kv, k4)
        x += h / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        v += h / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        integ += h / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        fm = (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0
        # backlash (dead band of width 2 half between motor and load)
        if x - load > half:
            load = x - half
        elif x - load < -half:
            load = x + half
    st.x[a] = x
    st.v[a] = v
    st.integ[a] = integ
    st.load[a] = load
    st.force[a] = fm
    st.acc[a] = (v - v_start) / dt
    st.cmd[a] = cmd


# =============================================================================== spindles

@njit(f8(SPINDLE_T, i8, i8, f8), **OPTS)
def torque_limit(sp, s, gear, omega):
    """Drive torque limit (N m) of spindle s in gear range ``gear`` at |omega| (the gear's curve)."""
    w = abs(omega)
    n = sp.n_curve[s, gear]
    cw = sp.curve_w[s, gear]
    ct = sp.curve_t[s, gear]
    if n <= 0:
        return 0.0
    if w <= cw[0]:
        return ct[0]
    for i in range(1, n):
        if w <= cw[i]:
            span = cw[i] - cw[i - 1]
            if span <= 0.0:
                return ct[i]
            return ct[i - 1] + (ct[i] - ct[i - 1]) * (w - cw[i - 1]) / span
    return ct[n - 1]


@njit(types.void(SPINDLE_T, i8, i8, f8, f8, f8, f8, f8, f8, f8, i8, f8, F1), **OPTS)
def _spindle_rhs(sp, s, gear, theta, omega, integ, wref_speed, theta_c, wff, kv, mode, t_load, out):
    """d/dt (theta, omega, integ) of spindle s into out[0:3]; out[3] = motor torque."""
    if mode == SPM_SPEED:
        wref = wref_speed
    else:
        wref = kv * (theta_c - theta) + wff
    e = wref - omega
    tu = sp.kp[s] * e + integ
    tmax = torque_limit(sp, s, gear, omega)
    tm = tu
    dint = sp.ki[s] * e
    if tu > tmax:
        tm = tmax
        if e > 0.0:
            dint = 0.0
    elif tu < -tmax:
        tm = -tmax
        if e < 0.0:
            dint = 0.0
    ratio = abs(omega) / sp.omega_max[s] if sp.omega_max[s] > 0.0 else 0.0
    drag = sp.drag_max[s] * ratio ** (2.0 / 3.0)
    if omega < 0.0:
        drag = -drag
    out[0] = omega
    out[1] = (tm - t_load - drag) / sp.inertia[s]
    out[2] = dint
    out[3] = tm


@njit(types.void(SPINDLE_T, SPSTATE_T, i8, i8, f8, f8, f8, f8, f8, F1), **OPTS)
def spindle_step(sp, st, s, mode, wref, theta_c, wff, kv, t_load, work):
    """Advance spindle s over one L1 tick. SPEED: track wref; POSITION/CAXIS: theta_c with feed-forward
    wff (the position command is fine-interpolated from the previous one). t_load: cutting torque
    acting against the motor (N m, positive opposes positive omega). work: scratch [>= 16]."""
    dt = sp.dt[0]
    n = sp.n_sub[0]
    h = dt / n
    theta = st.angle[s]
    omega = st.omega[s]
    integ = st.integ[s]
    c0 = st.cmd_angle[s]
    dcmd = theta_c - c0
    if mode == SPM_SPEED:
        dcmd = 0.0
        c0 = theta
    k1 = work[0:4]
    k2 = work[4:8]
    k3 = work[8:12]
    k4 = work[12:16]
    g = st.gear[s]
    tm = st.torque[s]
    for sub in range(n):
        ca = c0 + dcmd * (sub / n)
        cm = c0 + dcmd * ((sub + 0.5) / n)
        cb = c0 + dcmd * ((sub + 1.0) / n)
        _spindle_rhs(sp, s, g, theta, omega, integ, wref, ca, wff, kv, mode, t_load, k1)
        _spindle_rhs(sp, s, g, theta + 0.5 * h * k1[0], omega + 0.5 * h * k1[1], integ + 0.5 * h * k1[2], wref, cm,
                     wff, kv, mode, t_load, k2)
        _spindle_rhs(sp, s, g, theta + 0.5 * h * k2[0], omega + 0.5 * h * k2[1], integ + 0.5 * h * k2[2], wref, cm,
                     wff, kv, mode, t_load, k3)
        _spindle_rhs(sp, s, g, theta + h * k3[0], omega + h * k3[1], integ + h * k3[2], wref, cb, wff, kv, mode,
                     t_load, k4)
        theta += h / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        omega += h / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        integ += h / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        tm = (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0
    st.angle[s] = theta
    st.omega[s] = omega
    st.integ[s] = integ
    st.torque[s] = tm
    st.cmd_angle[s] = theta_c if mode != SPM_SPEED else theta


@njit(f8(f8, f8, f8, f8, f8), **OPTS)
def ramp_toward(current, target, up_rate, down_rate, dt):
    """Speed command ramp: |omega| rises at most up_rate * dt and falls at most down_rate * dt per tick."""
    if current * target < 0.0:
        target_mag = 0.0          # through zero first (a reversal decelerates to rest, then accelerates)
    else:
        target_mag = abs(target)
    cur = abs(current)
    if target_mag > cur:
        step = up_rate * dt
        new = cur + step if target_mag - cur > step else target_mag
    else:
        step = down_rate * dt
        new = cur - step if cur - target_mag > step else target_mag
    if current == 0.0:
        sign = 1.0 if target >= 0.0 else -1.0
    elif current * target < 0.0 and new == 0.0:
        return 0.0
    else:
        sign = 1.0 if current > 0.0 else -1.0
    return sign * new


# =============================================================================== geometry pieces

@njit(types.void(i8, f8, F2, F1, f8, I1, F1), **OPTS)
def geo_eval(kind, length, vec, par, s, axes, out):
    """Axis values of a geometry piece at path position s (0..length) for the axes listed in ``axes``.

    LINE: q = v0 + v1 u; ARC: q = v0 + r(u)(cos(phi u) v1 + sin(phi u) v2) + v3 u with
    r(u) = r0 + (r1 - r0) u, phi = par[2], u = s / length."""
    u = s / length if length > 0.0 else 1.0
    if u < 0.0:
        u = 0.0
    elif u > 1.0:
        u = 1.0
    if kind == GEO_ARC:
        r = par[0] + (par[1] - par[0]) * u
        ph = par[2] * u
        cr = r * math.cos(ph)
        sr = r * math.sin(ph)
        for j in range(axes.shape[0]):
            a = axes[j]
            out[a] = vec[0, a] + cr * vec[1, a] + sr * vec[2, a] + vec[3, a] * u
    else:
        for j in range(axes.shape[0]):
            a = axes[j]
            out[a] = vec[0, a] + vec[1, a] * u


@njit(types.void(i8, f8, F2, F1, F1, I1, F2), **OPTS)
def geo_eval_many(kind, length, vec, par, s, axes, out):
    """out[i, axes] = geo_eval at every s[i]."""
    for i in range(s.shape[0]):
        geo_eval(kind, length, vec, par, s[i], axes, out[i])


# =============================================================================== the L1 tick

@njit(f8(SETP_T, i8, f8, F1, F1), **OPTS)
def _table_s(setp, p, dtheta, theta_tab, s_tab):
    """Planned path position at spindle angle progress dtheta: linear in the table, the last slope beyond
    its end, its first value before its start."""
    n = setp.sync_n[p]
    if n <= 0:
        return 0.0
    if n == 1 or dtheta <= theta_tab[0]:
        return s_tab[0]
    for i in range(1, n):
        if dtheta <= theta_tab[i]:
            span = theta_tab[i] - theta_tab[i - 1]
            if span <= 0.0:
                return s_tab[i]
            return s_tab[i - 1] + (s_tab[i] - s_tab[i - 1]) * (dtheta - theta_tab[i - 1]) / span
    span = theta_tab[n - 1] - theta_tab[n - 2]
    slope = (s_tab[n - 1] - s_tab[n - 2]) / span if span > 0.0 else 0.0
    return s_tab[n - 1] + slope * (dtheta - theta_tab[n - 1])


@njit(types.void(SETP_T, i8, f8, I1, F1), **OPTS)
def _stretch_command(setp, p, s, axes, out):
    """Commanded axis values of path p at stretch position s (piece lookup)."""
    n = setp.geo_n[p]
    g = 0
    for j in range(n):
        if s >= setp.geo_s0[p, j]:
            g = j
    geo_eval(setp.geo_kind[p, g], setp.geo_len[p, g], setp.geo_vec[p, g], setp.geo_par[p, g], s - setp.geo_s0[p, g],
             axes, out)


@njit(i4(CHAIN_T, SERVO_T, SSTATE_T, SPINDLE_T, SPSTATE_T, SETP_T, LOOP_T, OUT_T, i8, F1, F1), **OPTS)
def motion_tick(chain, servo, ss, sp, sps, setp, ls, out, k, dist_force, dist_torque):
    """One L1 tick (index k of the macro step) of every spindle, axis and path.

    dist_force[a]: external (cutting) force on axis a along +axis (N), dist_torque[s]: cutting torque
    on spindle s against +omega (N m) - the previous tick's values (one-tick lag, architecture 5.4).
    Returns the first non-zero sync code (out.sync[k, p] holds every path's)."""
    dt = servo.dt[0]
    n_axes = servo.kv.shape[0]
    n_paths = setp.mode.shape[1]
    n_sp = sp.inertia.shape[0]
    work = np.empty(32)
    cmdk = np.empty(n_axes)
    tmpq = np.empty(n_axes)
    all_axes = np.empty(n_axes, np.int32)
    for a in range(n_axes):
        cmdk[a] = setp.cmd[k, a]
        all_axes[a] = a
    # ---------------------------------------------------------------- stretch starts (angle before the step)
    for p in range(n_paths):
        if setp.sync_start[k, p] == 1:
            s_id = setp.sync_spindle[p]
            ls.theta0[p] = sps.angle[s_id] if s_id >= 0 else 0.0
            ls.theta_prev[p] = ls.theta0[p]
            ls.stage[p] = 0
            ls.s_act[p] = 0.0
            ls.s_filt[p] = 0.0
    # ---------------------------------------------------------------- spindles
    for s in range(n_sp):
        mode = setp.sp_mode[k, s]
        if mode == SPM_SPEED:
            target = setp.sp_omega[k, s]
            vc = setp.css_vc[k, s]
            ca = setp.css_axis[s]
            if vc > 0.0 and ca >= 0 and target != 0.0:
                r = abs(ss.cmd[ca] - setp.css_centre[s])
                rmin = setp.css_rmin[0]
                if r < rmin:
                    r = rmin
                w = vc / r
                lim = setp.css_clamp[k, s]
                if w > lim:
                    w = lim
                if w > sp.omega_max[s]:
                    w = sp.omega_max[s]
                target = w if target > 0.0 else -w
            sps.cmd_omega[s] = ramp_toward(sps.cmd_omega[s], target, sp.accel_rate[s], sp.decel_rate[s], dt)
            spindle_step(sp, sps, s, SPM_SPEED, sps.cmd_omega[s], 0.0, 0.0, 0.0, dist_torque[s], work)
        elif mode == SPM_POSITION:
            sps.cmd_omega[s] = setp.sp_omega_ff[k, s]
            spindle_step(sp, sps, s, SPM_POSITION, 0.0, setp.sp_angle[k, s], sp.pos_kff[s] * setp.sp_omega_ff[k, s],
                         sp.pos_kv[s], dist_torque[s], work)
        else:
            ca = sp.c_axis[s]
            kap = servo.kappa[ca] if ca >= 0 else 1.0
            th = (cmdk[ca] - ss.c_off[ca]) / kap if ca >= 0 else sps.cmd_angle[s]
            wff = (th - sps.cmd_angle[s]) / dt
            sps.cmd_omega[s] = wff
            spindle_step(sp, sps, s, SPM_CAXIS, 0.0, th, sp.c_kff[s] * wff, sp.c_kv[s], dist_torque[s], work)
        out.sp_angle[k, s] = sps.angle[s]
        out.sp_omega[k, s] = sps.omega[s]
        out.sp_torque[k, s] = sps.torque[s]
    # ---------------------------------------------------------------- spindle-synchronous setpoints
    code_first = SYNC_NONE
    for p in range(n_paths):
        out.sync[k, p] = SYNC_NONE
        out.alarm[k, p] = 0
        out.alarm_axis[k, p] = -1
    seg_end = np.zeros(n_paths, np.int8)
    for p in range(n_paths):
        m = setp.mode[k, p]
        if m != SP_PER_REV and m != SP_THREAD:
            continue
        s_id = setp.sync_spindle[p]
        theta = sps.angle[s_id] if s_id >= 0 else 0.0
        s_end = setp.sync_s_end[p]
        if m == SP_PER_REV:
            dth = abs(theta - ls.theta0[p])
            s_now = _table_s(setp, p, dth, setp.sync_theta[p], setp.sync_s[p])
            if s_now > s_end:
                s_now = s_end
            if s_now < ls.s_act[p]:
                s_now = ls.s_act[p]
            ls.s_act[p] = s_now
            if s_now >= s_end - 1e-12:
                seg_end[p] = 1
        else:
            if ls.stage[p] == 0:
                trig = setp.thread_trig[p]
                prev = ls.theta_prev[p]
                n0 = math.floor((prev - trig) / TWO_PI)
                n1 = math.floor((theta - trig) / TWO_PI)
                if n1 != n0:
                    # crossing of trig + 2 pi n (direction of rotation either way)
                    edge = trig + TWO_PI * (n1 if n1 > n0 else n0)
                    ls.stage[p] = 1
                    ls.theta_trig[p] = edge
                    ls.s_filt[p] = 0.0
            if ls.stage[p] == 1:
                raw = setp.thread_lead[p] * abs(theta - ls.theta_trig[p]) / TWO_PI
                if raw > s_end:
                    raw = s_end
                tc = setp.thread_tc[p]
                if tc > 0.0:
                    ls.s_filt[p] += (raw - ls.s_filt[p]) * (1.0 - math.exp(-dt / tc))
                    if raw >= s_end and s_end - ls.s_filt[p] < 1e-7:
                        ls.s_filt[p] = s_end
                else:
                    ls.s_filt[p] = raw
                s_now = ls.s_filt[p]
            else:
                s_now = 0.0
            ls.s_act[p] = s_now
            if ls.stage[p] == 1 and s_now >= s_end - 1e-12:
                seg_end[p] = 1
        ls.theta_prev[p] = theta
        _stretch_command(setp, p, ls.s_act[p], all_axes, tmpq)
        for a in range(n_axes):
            if servo.path[a] == p:
                cmdk[a] = tmpq[a]
        out.sync_s[k, p] = ls.s_act[p]
    # ---------------------------------------------------------------- feed drives
    for a in range(n_axes):
        out.cmd[k, a] = cmdk[a]
        s_link = servo.spindle[a]
        p = servo.path[a]
        if s_link >= 0:
            act = servo.kappa[a] * sps.angle[s_link] + ss.c_off[a]
            out.actual[k, a] = act
            if setp.sp_mode[k, s_link] != SPM_CAXIS:
                out.cmd[k, a] = act          # not in C-axis mode: the axis turns with the spindle
            out.ferr[k, a] = cmdk[a] - act if setp.sp_mode[k, s_link] == SPM_CAXIS else 0.0
            ss.load[a] = act
            ss.x[a] = act
            ss.cmd[a] = cmdk[a] if setp.sp_mode[k, s_link] == SPM_CAXIS else act
            out.force[k, a] = 0.0
            out.inertial[k, a] = 0.0
            out.overload[k, a] = 0
            continue
        kv = servo.tap_kv[a] if setp.tap[k, p] == 1 else servo.kv[a]
        v_prev = ss.vel_prev[a]
        vc = (cmdk[a] - ss.cmd[a]) / dt
        servo_axis_step(servo, ss, a, cmdk[a], kv, dist_force[a], work)
        out.actual[k, a] = ss.load[a]
        out.ferr[k, a] = cmdk[a] - ss.load[a]
        acc_c = (vc - v_prev) / dt
        ss.vel_prev[a] = vc
        vsr = servo.v_stribeck[a]
        fric = servo.friction[a] * math.tanh(vc / vsr) if vsr > 0.0 else 0.0
        demand = servo.mass[a] * acc_c + fric - dist_force[a] - servo.gravity[a]
        out.force[k, a] = demand
        out.inertial[k, a] = servo.mass[a] * ss.acc[a]
        out.overload[k, a] = 1 if abs(demand) > servo.force_max[a] else 0
        # travel (stored stroke check on the command) and excess following error
        if cmdk[a] > servo.hi[a]:
            out.alarm[k, p] = ALARM_OVERTRAVEL_POS
            out.alarm_axis[k, p] = a
        elif cmdk[a] < servo.lo[a]:
            out.alarm[k, p] = ALARM_OVERTRAVEL_NEG
            out.alarm_axis[k, p] = a
        meas = ss.load[a] if servo.scale_fb[a] == 1 else ss.x[a]
        if abs(cmdk[a] + ss.comp[a] - meas) > servo.ferr_max[a] and out.alarm[k, p] == 0:
            out.alarm[k, p] = ALARM_FOLLOWING_ERROR
            out.alarm_axis[k, p] = a
    # ---------------------------------------------------------------- kinematics, checks, sync
    q = out.actual[k]
    fkw = np.empty((6, 4, 4))
    for p in range(n_paths):
        poses = out.poses[p]
        fk_poses(chain, p, q, poses, fkw)
        m_t0 = poses[FK_M_T0]
        s_t0 = poses[FK_S_T0]
        w0_t0 = poses[FK_W0_T0]
        for r in range(3):
            out.tip_M[k, p, r] = m_t0[r, 3]
            out.tip_S[k, p, r] = s_t0[r, 3]
            out.tip_W0[k, p, r] = w0_t0[r, 3]
            out.axis_M[k, p, r] = m_t0[r, 2]
        code = SYNC_NONE
        m = setp.mode[k, p]
        if out.alarm[k, p] != 0:
            code = SYNC_ALARM
        elif setp.probe[k, p] == 1 and out.probe_hit[k, p] == 1:
            code = SYNC_PROBE_TRIGGER
        elif setp.setter[k, p] == 1:
            pt = setp.setter_point[p]
            nn = setp.setter_normal[p]
            rad = setp.setter_radius[p]
            ext = setp.setter_extra[p]
            dx = out.tip_M[k, p, 0] - out.axis_M[k, p, 0] * ext - pt[0]
            dy = out.tip_M[k, p, 1] - out.axis_M[k, p, 1] * ext - pt[1]
            dz = out.tip_M[k, p, 2] - out.axis_M[k, p, 2] * ext - pt[2]
            if rad <= 0.0:
                if dx * nn[0] + dy * nn[1] + dz * nn[2] <= 0.0:
                    code = SYNC_PROBE_TRIGGER
            else:
                ax0 = out.axis_M[k, p, 0]
                ax1 = out.axis_M[k, p, 1]
                ax2 = out.axis_M[k, p, 2]
                along = -(dx * ax0 + dy * ax1 + dz * ax2)
                px = -dx - along * ax0
                py = -dy - along * ax1
                pz = -dz - along * ax2
                if 0.0 <= along <= 5.0 * rad and math.sqrt(px * px + py * py + pz * pz) <= rad:
                    code = SYNC_PROBE_TRIGGER
        if code == SYNC_NONE:
            if m == SP_END:
                code = SYNC_QUEUE_EMPTY
            elif (m == SP_PER_REV or m == SP_THREAD) and seg_end[p] == 1:
                code = SYNC_SEGMENT_END
            elif m == SP_HOLD_WAIT:
                inpos = True
                for a in range(n_axes):
                    if servo.path[a] != p:
                        continue
                    s_link = servo.spindle[a]
                    if s_link >= 0:
                        if setp.sp_mode[k, s_link] == SPM_CAXIS and abs(out.ferr[k, a]) > servo.in_pos[a]:
                            inpos = False
                        continue
                    meas = ss.load[a] if servo.scale_fb[a] == 1 else ss.x[a]
                    if abs(cmdk[a] + ss.comp[a] - meas) > servo.in_pos[a]:
                        inpos = False
                if inpos:
                    code = SYNC_PATH_WAIT if setp.wait_kind[p] == WAIT_PATH else SYNC_IN_POSITION
        out.sync[k, p] = code
        if code != SYNC_NONE and code_first == SYNC_NONE:
            code_first = code
    out.k_done[0] = k
    return code_first


@njit(i4(SETP_T, i8), **OPTS)
def replan_before(setp, k):
    """Path whose plan ends before tick k (setpoint SP_END at k), or -1. The loop checks it before every
    tick after the first of a call and returns without executing tick k (sync code SYNC_QUEUE_EMPTY at
    tick k - 1), so a re-plan point costs no simulated time."""
    for p in range(setp.mode.shape[1]):
        if setp.mode[k, p] == SP_END:
            return p
    return -1


@njit(i4(CHAIN_T, SERVO_T, SSTATE_T, SPINDLE_T, SPSTATE_T, SETP_T, LOOP_T, OUT_T, i8, i8, F1, F1, I8_1), **OPTS)
def run_motion(chain, servo, ss, sp, sps, setp, ls, out, k_from, k_to, dist_force, dist_torque, result):
    """Ticks k_from .. k_to - 1 of the step (motion without cutting: constant disturbances). Returns at
    the first sync event: result = (code, tick index, path) (code 0 and tick k_to - 1 if none). Before a
    tick after the first whose setpoint is SP_END, it returns SYNC_QUEUE_EMPTY at the previous tick."""
    result[0] = SYNC_NONE
    result[1] = k_to - 1
    result[2] = -1
    for k in range(k_from, k_to):
        if k > k_from:
            p_end = replan_before(setp, k)
            if p_end >= 0:
                out.sync[k - 1, p_end] = SYNC_QUEUE_EMPTY
                result[0] = SYNC_QUEUE_EMPTY
                result[1] = k - 1
                result[2] = p_end
                return SYNC_QUEUE_EMPTY
        code = motion_tick(chain, servo, ss, sp, sps, setp, ls, out, k, dist_force, dist_torque)
        if code != SYNC_NONE:
            result[0] = code
            result[1] = k
            for p in range(setp.mode.shape[1]):
                if out.sync[k, p] != SYNC_NONE:
                    result[2] = p
                    break
            return code
    return SYNC_NONE


# =============================================================================== look-ahead generator

@njit(f8(f8, f8, f8, f8, f8), **OPTS)
def brake_distance(v, a, u, amax, jmax):
    """Minimum distance to bring (v, a) to velocity <= u with acceleration 0 (|a| <= amax, |j| <= jmax).

    Time-optimal jerk-limited braking: jerk -J from a to a_p (a_p = -sqrt((a^2 - 2 J dv)/2), clipped to
    -amax with a constant-deceleration phase), then +J back to 0 (Biagiotti & Melchiorri 2008 s. 3.4).
    When ramping the acceleration to 0 already ends below u, the distance until v first reaches u."""
    J = jmax
    if a > 0.0:
        vn = v + a * a / (2.0 * J)
    else:
        vn = v - a * a / (2.0 * J)
    if vn <= u:
        if v <= u:
            return 0.0
        disc = a * a - 2.0 * J * (v - u)
        if disc < 0.0:
            disc = 0.0
        t = (-a - math.sqrt(disc)) / J
        if t < 0.0:
            t = 0.0
        return v * t + 0.5 * a * t * t + J * t * t * t / 6.0
    dv = u - v
    ap2 = 0.5 * (a * a - 2.0 * J * dv)
    ap = -math.sqrt(ap2) if ap2 > 0.0 else 0.0
    t2 = 0.0
    if ap < -amax:
        ap = -amax
        t2 = (dv - (a * a - 2.0 * amax * amax) / (2.0 * J)) / ap
        if t2 < 0.0:
            t2 = 0.0
    t1 = (a - ap) / J
    if t1 < 0.0:
        t1 = 0.0
    t3 = -ap / J
    d1 = v * t1 + 0.5 * a * t1 * t1 - J * t1 * t1 * t1 / 6.0
    v1 = v + a * t1 - 0.5 * J * t1 * t1
    d2 = v1 * t2 + 0.5 * ap * t2 * t2
    v2 = v1 + ap * t2
    d3 = v2 * t3 + 0.5 * ap * t3 * t3 + J * t3 * t3 * t3 / 6.0
    return d1 + d2 + d3


@njit(f8(f8, f8, f8, f8, f8), **OPTS)
def natural_velocity(v, a, jj, dt, jmax):
    """Velocity reached when, after one step of jerk jj, the acceleration is ramped to 0 at jmax."""
    a1 = a + jj * dt
    v1 = v + a * dt + 0.5 * jj * dt * dt
    if a1 > 0.0:
        return v1 + a1 * a1 / (2.0 * jmax)
    return v1 - a1 * a1 / (2.0 * jmax)


@njit(f8(f8, f8, f8, f8, f8, f8), **OPTS)
def track_jerk(v, a, vt, amax, jmax, dt):
    """Largest jerk for the next dt keeping the velocity approach to vt time-optimal without overshoot:
    after the step, ramping the acceleration to 0 must not end above vt (from below or above) and the
    acceleration stays within +-amax. Exact switching by bisection (monotone in the jerk)."""
    J = jmax
    j_hi = J
    if (amax - a) / dt < j_hi:
        j_hi = (amax - a) / dt
    j_lo = -J
    if (-amax - a) / dt > j_lo:
        j_lo = (-amax - a) / dt
    if j_hi < j_lo:
        j_hi = j_lo
    if natural_velocity(v, a, j_hi, dt, J) <= vt:
        return j_hi
    if natural_velocity(v, a, j_lo, dt, J) > vt:
        return j_lo
    lo = j_lo
    hi = j_hi
    for _ in range(_BISECT + 10):
        mid = 0.5 * (lo + hi)
        if natural_velocity(v, a, mid, dt, J) <= vt:
            lo = mid
        else:
            hi = mid
    return lo


@njit(types.boolean(f8, f8, f8, f8, F1, F1, F1, F1, i8), **OPTS)
def _nodes_ok(v1, a1, moved, tol, node_x, node_u, node_a, node_j, n_nodes):
    if v1 < -1e-9:
        return False
    for j in range(n_nodes):
        d = node_x[j] - moved
        if d < 0.0:
            continue
        if brake_distance(v1 if v1 > 0.0 else 0.0, a1, node_u[j], node_a[j], node_j[j]) > d + tol:
            return False
    return True


@njit(i4(F1, F1, F1, F1, F1, I8_1, B1, i8, i8, I8_1, F1, i8, i8, f8, i8, I1, F1, F1, F1, B1, I1), **OPTS)
def plan_ticks(L, vmax, amax, jmax, ustart, ready, stop_end, n_pieces, lookahead, gi, gf, t_first, ipo, dt, n_ticks,
               out_piece, out_s, out_v, out_a, out_lim, out_count):
    """Jerk-limited look-ahead profile along a chain of pieces.

    Piece j: length L[j], speed limit vmax[j] (feed with override, axis velocity limits projected on
    the path, centripetal sqrt(A R), rotary limits), path acceleration amax[j] and jerk jmax[j]
    (axis limits projected), bound ustart[j] at its start node (corner speed; <= vmax of both
    sides), ready[j] the base tick from which the control has the block (look-ahead never passes a
    block before its ready tick), stop_end[j] = 1 if the path must stop at its end (exact stop,
    in-position, a non-motion command next). The window is the ready pieces from the current one up
    to ``lookahead`` pieces; its end is a stop node (the control can only plan to a standstill at the
    end of what it knows, which is data starvation on short blocks).

    Per tick the jerk is the largest in [-J, J] that keeps the cruise approach to vmax and the braking
    distance to every node ahead feasible (brake_distance), found by bisection: bang-bang jerk with
    exact switching, i.e. the 7-segment double-S profile, re-planned every tick as blocks arrive.

    State: gi[0] piece index, gf = (s within the piece, v, a). Tick i ends at base tick t_first + i ipo.
    Outputs per tick: piece index, s, v, a and out_lim: 0 free (cruise or accelerating at the limits),
    1 held back by a node ahead (corner, stop), 2 held back by the end of the window where blocks are
    not prepared yet or lie beyond the look-ahead (data starvation). Returns 0 = all n_ticks generated
    (moving or waiting at the window end), 1 = stopped at the end of a stop_end piece after
    out_count[0] ticks, 2 = every piece done after out_count[0] ticks."""
    node_x = np.empty(lookahead + 2)
    node_u = np.empty(lookahead + 2)
    node_a = np.empty(lookahead + 2)
    node_j = np.empty(lookahead + 2)
    out_count[0] = 0
    for i in range(n_ticks):
        t = t_first + i * ipo
        k = gi[0]
        s = gf[0]
        v = gf[1]
        a = gf[2]
        if k >= n_pieces:
            out_count[0] = i
            return 2
        if ready[k] > t:
            out_piece[i] = k
            out_s[i] = s
            out_v[i] = 0.0
            out_a[i] = 0.0
            out_lim[i] = 2
            gf[1] = 0.0
            gf[2] = 0.0
            continue
        # window end
        e = k
        while stop_end[e] == 0 and e + 1 < n_pieces and e + 1 < k + lookahead and ready[e + 1] <= t:
            e += 1
        # nodes ahead (relative to the current position), with the braking limits of the range
        n_nodes = 0
        x = L[k] - s
        a_run = amax[k]
        j_run = jmax[k]
        for j in range(k + 1, e + 1):
            node_x[n_nodes] = x
            node_u[n_nodes] = ustart[j]
            node_a[n_nodes] = a_run
            node_j[n_nodes] = j_run
            n_nodes += 1
            if amax[j] < a_run:
                a_run = amax[j]
            if jmax[j] < j_run:
                j_run = jmax[j]
            x += L[j]
        node_x[n_nodes] = x
        node_u[n_nodes] = 0.0
        node_a[n_nodes] = a_run
        node_j[n_nodes] = j_run
        n_nodes += 1
        x_end = x
        A = amax[k]
        J = jmax[k]
        vcap = vmax[k]
        tol = _TOL_M + 1e-9 * x_end
        j_cap = track_jerk(v, a, vcap, A, J, dt)
        j_lo = -J
        if (-A - a) / dt > j_lo:
            j_lo = (-A - a) / dt
        if j_cap < j_lo:
            j_cap = j_lo

        artificial = stop_end[e] == 0 and e < n_pieces - 1
        a1 = a + j_cap * dt
        v1 = v + a * dt + 0.5 * j_cap * dt * dt
        d1 = v * dt + 0.5 * a * dt * dt + j_cap * dt * dt * dt / 6.0
        lim = 0
        if _nodes_ok(v1, a1, d1, tol, node_x, node_u, node_a, node_j, n_nodes):
            jj = j_cap
        else:
            lim = 1
            if artificial and _nodes_ok(v1, a1, d1, tol, node_x, node_u, node_a, node_j, n_nodes - 1):
                lim = 2
            a1 = a + j_lo * dt
            v1 = v + a * dt + 0.5 * j_lo * dt * dt
            d1 = v * dt + 0.5 * a * dt * dt + j_lo * dt * dt * dt / 6.0
            if not _nodes_ok(v1, a1, d1, tol, node_x, node_u, node_a, node_j, n_nodes):
                jj = j_lo
            else:
                lo = j_lo
                hi = j_cap
                for _ in range(_BISECT):
                    mid = 0.5 * (lo + hi)
                    a1 = a + mid * dt
                    v1 = v + a * dt + 0.5 * mid * dt * dt
                    d1 = v * dt + 0.5 * a * dt * dt + mid * dt * dt * dt / 6.0
                    if _nodes_ok(v1, a1, d1, tol, node_x, node_u, node_a, node_j, n_nodes):
                        lo = mid
                    else:
                        hi = mid
                jj = lo
        a1 = a + jj * dt
        v1 = v + a * dt + 0.5 * jj * dt * dt
        d1 = v * dt + 0.5 * a * dt * dt + jj * dt * dt * dt / 6.0
        if v1 < 0.0:
            # comes to rest inside the tick: stop there (time of v = 0 from the cubic)
            tz = dt
            lo_t = 0.0
            hi_t = dt
            for _ in range(40):
                mid = 0.5 * (lo_t + hi_t)
                if v + a * mid + 0.5 * jj * mid * mid > 0.0:
                    lo_t = mid
                else:
                    hi_t = mid
            tz = lo_t
            d1 = v * tz + 0.5 * a * tz * tz + jj * tz * tz * tz / 6.0
            v1 = 0.0
            a1 = 0.0
        if d1 < 0.0:
            d1 = 0.0
        # snap onto the window end (a stop node) when within rounding of it
        remaining = x_end - d1
        if remaining < _SNAP_M and v1 < 1e-3 or (remaining < 1e-7 and v1 < _SNAP_V) or remaining < 0.0:
            d1 = x_end
            v1 = 0.0
            a1 = 0.0
        # advance along the pieces
        s1 = s + d1
        while k < e and s1 > L[k]:
            s1 -= L[k]
            k += 1
        if s1 > L[k]:
            s1 = L[k]
        gi[0] = k
        gf[0] = s1
        gf[1] = v1
        gf[2] = a1
        out_piece[i] = k
        out_s[i] = s1
        out_v[i] = v1
        out_a[i] = a1
        if lim == 0 and artificial and d1 == 0.0:
            lim = 2
        out_lim[i] = lim
        if k == e and s1 >= L[e] and v1 == 0.0:
            if stop_end[e] == 1 or e == n_pieces - 1:
                gi[0] = e + 1
                gf[0] = 0.0
                out_count[0] = i + 1
                return 1 if stop_end[e] == 1 else 2
    out_count[0] = n_ticks
    return 0
