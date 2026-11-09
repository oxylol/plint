"""Compiled helpers of the hot loop (docs/models/hotloop.md; architecture 4.1, 5.4, 5.5).

* :func:`hermite` - spindle angle and speed inside an L1 tick (cubic Hermite on the tick's end values);
* :func:`interp_axes` - axis values at a fraction of the tick: linear axes in axis space, spindle-coupled
  axes (C axes on a spindle rotor) from the interpolated spindle angle (never a matrix lerp);
* :func:`fk_frame` - the kinematic chain of one path at given axis values: tool tip and T0 axes in S, and
  S in M (motion's fk_poses);
* :func:`set_step_pose` - cutting's StepPose of an L0 step from its start, mid and end frames;
* :func:`refill_planes` - field planes of the stale memory bins the coming step reads (removal's ray cast
  against the field at the moment of reading: no L2 query, no replay);
* :func:`displacement_at` - structure's displacement d(X) at a point (trilinear on its grid in S);
* :func:`slab_point` - a slab's axis point at its generation instant (tip + u_tool - u_work, material
  configuration, architecture 4.1);
* :func:`polar_view_update` - cutting's PolarView rows from removal's polar field (turning);
* :func:`rotvec` - rotation vector of the relative rotation of two frames.

Every helper has one explicit signature (architecture 5.5); the small ones are inlined at the Numba IR level.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, types

from ..cutting import abi as CA
from ..motion.abi import FK_M_W, FK_S_T0
from ..motion.kernels import CHAIN_T, SERVO_T, SSTATE_T, fk_poses
from ..removal import abi as RA
from ..removal.k_sd import dexel_sd, gradient, level_at
from ..removal.nbt import FIELD_T, POLAR_T
from . import hot_abi as H

OPTS = dict(cache=True, nogil=True, error_model="numpy")
INLINE = dict(OPTS, inline="always")
f8 = types.float64
i8 = types.int64
b1 = types.boolean
void = types.void
F1 = types.float64[::1]
F2 = types.float64[:, ::1]
F3 = types.float64[:, :, ::1]
F4 = types.float64[:, :, :, ::1]
I1 = types.int64[::1]
I2 = types.int64[:, ::1]
CUT_T = H.CUTARGS_T
VIEW_T = CA.POLAR_T
TWO_PI = 2.0 * math.pi

# frame records in HotContact.work (fk_frame output): tip_S [3], R_S_T0 [9] row-major, R_M_S [9], t_M_S [3]
FR_TIP = 0
FR_R = 3
FR_RM = 12
FR_TM = 21
FR_SIZE = 24
WK_A0 = 0                   # frame at the step start
WK_MID = 24                 # at the step mid
WK_A1 = 48                  # at the step end
WK_DISP = 72                # [3] structure displacement d at the contact (this tick)
WK_UREL = 75                # [3] scratch
WK_FEED = 78                # [3] feed velocity of the tip relative to the material in T0 (last step)
WK_PASS = 81                # [12] pose of a tooth pass: tip (3), x0 (3), y0 (3), w (3) (kept for a repeat)
WK_PFX = 93                 # feed direction (T0 xy) and slack of a pass refused by a pool overflow
WK_PFY = 94
WK_PSLACK = 95
WK_SIZE = 112


@njit(types.UniTuple(f8, 2)(f8, f8, f8, f8, f8, f8), **INLINE)
def hermite(th0, w0, th1, w1, a, h):
    """Angle and rate at fraction a of an interval of length h from the end values and rates (cubic Hermite:
    exact for a rate linear in time)."""
    a2 = a * a
    a3 = a2 * a
    th = (2.0 * a3 - 3.0 * a2 + 1.0) * th0 + (a3 - 2.0 * a2 + a) * h * w0 + (3.0 * a2 - 2.0 * a3) * th1 \
        + (a3 - a2) * h * w1
    w = (6.0 * a2 - 6.0 * a) * (th0 - th1) / h + (3.0 * a2 - 4.0 * a + 1.0) * w0 + (3.0 * a2 - 2.0 * a) * w1
    return th, w


@njit(void(SERVO_T, SSTATE_T, F1, F2, i8, F1, F1, F1, F1, f8, f8, F1), **INLINE)
def interp_axes(servo, ss, q0, actual, k, th0, w0, th1, w1, a, h, out):
    """Axis values at fraction a of tick k: linear axes between the tick's start values q0 and its end values
    actual[k]; a C axis on a spindle rotor from the spindle angle (q = kappa theta + c_off)."""
    for ax in range(out.shape[0]):
        s = servo.spindle[ax]
        if s >= 0:
            th, _ = hermite(th0[s], w0[s], th1[s], w1[s], a, h)
            out[ax] = servo.kappa[ax] * th + ss.c_off[ax]
        else:
            out[ax] = q0[ax] + a * (actual[k, ax] - q0[ax])


@njit(void(CHAIN_T, i8, F1, F3, F3, F1, i8, f8, f8, f8), **OPTS)
def fk_frame(chain, p, q, poses, fkw, dst, off, mx, my, mz):
    """FK of path p at axis values q into dst[off:off + FR_SIZE]: tip and T0 axes in S, S in M. (mx, my, mz):
    the tool mount, signs of the chain's T0 axes in the tool's own T0 (HotContact.fpar CF_MOUNT_*)."""
    fk_poses(chain, p, q, poses, fkw)
    st = poses[FK_S_T0]
    mw = poses[FK_M_W]
    ws = chain.pose_W_S[p]
    for r in range(3):
        dst[off + FR_TIP + r] = st[r, 3]
        dst[off + FR_R + 3 * r] = st[r, 0] * mx
        dst[off + FR_R + 3 * r + 1] = st[r, 1] * my
        dst[off + FR_R + 3 * r + 2] = st[r, 2] * mz
    for r in range(3):
        for c in range(3):
            dst[off + FR_RM + 3 * r + c] = mw[r, 0] * ws[0, c] + mw[r, 1] * ws[1, c] + mw[r, 2] * ws[2, c]
        dst[off + FR_TM + r] = mw[r, 0] * ws[0, 3] + mw[r, 1] * ws[1, 3] + mw[r, 2] * ws[2, 3] + mw[r, 3]


@njit(void(F1, i8, i8, f8, F1), **INLINE)
def rotvec(w, off0, off1, dt, out):
    """Angular velocity (rotation vector / dt, in the parent frame) of the rotation R1 R0^T of the row-major 3x3
    blocks w[off0:off0 + 9] (R0) and w[off1:off1 + 9] (R1)."""
    d = np.empty(9)
    for r in range(3):
        for c in range(3):
            acc = 0.0
            for m in range(3):
                acc += w[off1 + 3 * r + m] * w[off0 + 3 * c + m]
            d[3 * r + c] = acc
    vx = d[7] - d[5]
    vy = d[2] - d[6]
    vz = d[3] - d[1]
    s = 0.5 * math.sqrt(vx * vx + vy * vy + vz * vz)
    cth = 0.5 * (d[0] + d[4] + d[8] - 1.0)
    if s > 1e-300:
        scale = math.atan2(s, cth) / (2.0 * s * dt)
    else:
        scale = 0.5 / dt
    out[0] = vx * scale
    out[1] = vy * scale
    out[2] = vz * scale


@njit(void(CUT_T, F1, F2, f8), **OPTS)
def set_step_pose(cut, wk, W_S, dt):
    """Cutting's StepPose of an L0 step: pose at the mid frame, velocities from the end frames (mean over the
    step, second order at the midpoint)."""
    pose = cut.pose
    w = np.empty(3)
    for r in range(3):
        pose.tip_S_m[r] = wk[WK_MID + FR_TIP + r]
        pose.v_tip_S_m_s[r] = (wk[WK_A1 + FR_TIP + r] - wk[WK_A0 + FR_TIP + r]) / dt
        pose.t_M_S_m[r] = wk[WK_MID + FR_TM + r]
        pose.v_S_M_m_s[r] = (wk[WK_A1 + FR_TM + r] - wk[WK_A0 + FR_TM + r]) / dt
        pose.t_W_S_m[r] = W_S[r, 3]
        for c in range(3):
            pose.R_S_T0[r, c] = wk[WK_MID + FR_R + 3 * r + c]
            pose.R_M_S[r, c] = wk[WK_MID + FR_RM + 3 * r + c]
            pose.R_W_S[r, c] = W_S[r, c]
    rotvec(wk, WK_A0 + FR_R, WK_A1 + FR_R, dt, w)
    for r in range(3):
        pose.w_T0_S_rad_s[r] = w[r]
    rotvec(wk, WK_A0 + FR_RM, WK_A1 + FR_RM, dt, w)
    for r in range(3):
        pose.w_S_M_rad_s[r] = w[r]


@njit(void(F1, F1, F4, F1, F1), **OPTS)
def displacement_at(origin, cell, disp, x, out):
    """d(X) at point x (S) from a grid of cell-centred values (origin + (i + 1/2) cell), trilinear, clamped
    at the grid's outer centres; zero without a grid."""
    out[0] = 0.0
    out[1] = 0.0
    out[2] = 0.0
    h = cell[0]
    if h <= 0.0:
        return
    nx = disp.shape[0]
    ny = disp.shape[1]
    nz = disp.shape[2]
    idx = np.empty(3, np.int64)
    fr = np.empty(3)
    dims = (nx, ny, nz)
    for c in range(3):
        u = (x[c] - origin[c]) / h - 0.5
        n = dims[c]
        if u <= 0.0:
            idx[c] = 0
            fr[c] = 0.0
        elif u >= n - 1:
            idx[c] = max(n - 2, 0)
            fr[c] = 1.0 if n > 1 else 0.0
        else:
            i = int(math.floor(u))
            idx[c] = i
            fr[c] = u - i
    for a in range(2):
        ia = min(idx[0] + a, nx - 1)
        wa = fr[0] if a == 1 else 1.0 - fr[0]
        for b in range(2):
            ib = min(idx[1] + b, ny - 1)
            wb = fr[1] if b == 1 else 1.0 - fr[1]
            for cc in range(2):
                ic = min(idx[2] + cc, nz - 1)
                wc = fr[2] if cc == 1 else 1.0 - fr[2]
                wt = wa * wb * wc
                if wt == 0.0:
                    continue
                for d in range(3):
                    out[d] += wt * disp[ia, ib, ic, d]


@njit(void(CUT_T, F1, F1, f8, F1), **INLINE)
def tool_deflection_S(cut, st_z, wk, z, out):
    """Tool deflection (translation, S) at height z_T from the station deflections (linear in z between the
    stations, end values held), rotated with the step's R_S_T0."""
    u = cut.defl.u_st_T0_m
    n = st_z.shape[0]
    ux = 0.0
    uy = 0.0
    uz = 0.0
    if n == 1 or z <= st_z[0]:
        ux = u[0, 0]
        uy = u[0, 1]
        uz = u[0, 2]
    elif z >= st_z[n - 1]:
        ux = u[n - 1, 0]
        uy = u[n - 1, 1]
        uz = u[n - 1, 2]
    else:
        lo = 0
        hi = n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if st_z[mid] <= z:
                lo = mid
            else:
                hi = mid
        span = st_z[hi] - st_z[lo]
        t = (z - st_z[lo]) / span if span > 0.0 else 0.0
        ux = u[lo, 0] + t * (u[hi, 0] - u[lo, 0])
        uy = u[lo, 1] + t * (u[hi, 1] - u[lo, 1])
        uz = u[lo, 2] + t * (u[hi, 2] - u[lo, 2])
    R = cut.pose.R_S_T0
    for r in range(3):
        out[r] = R[r, 0] * ux + R[r, 1] * uy + R[r, 2] * uz


@njit(void(CUT_T, F1, F1, f8, f8, F1), **OPTS)
def slab_point(cut, st_z, wk, s_frac, z, out):
    """Axis point of a slab at its generation instant (fraction s_frac of the step): the tip there plus the
    tool deflection at the slab height minus the workpiece deflection and structure displacement (the
    cutting Deflection's u_work), i.e. X_e of the slab's axis (architecture 4.1)."""
    u = np.empty(3)
    tool_deflection_S(cut, st_z, wk, z, u)
    uw = cut.defl.u_work_S_m
    for r in range(3):
        t0 = wk[WK_A0 + FR_TIP + r]
        t1 = wk[WK_A1 + FR_TIP + r]
        out[r] = t0 + s_frac * (t1 - t0) + u[r] - uw[r]


@njit(b1(CUT_T, i8, i8, i8, i8, i8), **INLINE)
def _stale(cut, row, b, tick, stale_ticks, stale_before):
    mem = cut.mem
    if mem.valid[row, b] == 0:
        return True
    t = mem.tick[row, b]
    return t < stale_before or tick - t > stale_ticks


@njit(b1(CUT_T, i8, f8, f8), **INLINE)
def _in_intervals(cut, s, a, margin):
    """Angle a (T0, [0, 2 pi)) inside one of station s's engaged intervals widened by margin."""
    iv = cut.iv
    for j in range(iv.count[s]):
        lo = iv.lo_rad[s, j] - margin
        hi = iv.hi_rad[s, j] + margin
        if lo <= a <= hi or lo <= a + TWO_PI <= hi or lo <= a - TWO_PI <= hi:
            return True
    return False


@njit(b1(FIELD_T, f8, f8, f8, f8, f8, f8, F1), **OPTS)
def surface_plane(f, px, py, pz, nx, ny, nz, out):
    """Field plane near point p for a chip-thickness normal n (unit, into the material): the interpolated
    boundary of the dexel family most aligned with n (removal's D_f, exact for planes, a^2 kappa / 4 for
    curved boundaries), outward normal from its gradient. out[1:4] = surface point (p moved along the outward
    normal by its depth), out[4:7] = outward unit normal. False when the family's four rays do not all see the
    boundary within SPREAD_CELLS (concave edges, far from the surface) or the boundary does not face the
    element (refill_planes then refills the bin at the element's own position)."""
    lvl = level_at(f, px, py, pz)
    ax = abs(nx)
    ay = abs(ny)
    az = abs(nz)
    fam = 0
    if ay > ax and ay >= az:
        fam = 1
    elif az > ax and az > ay:
        fam = 2
    sdo = np.empty(4)
    D = dexel_sd(f, lvl, fam, px, py, pz, sdo)
    if sdo[1] == 0.0:
        return False
    g = np.empty(3)
    gradient(f, px, py, pz, fam, 0.25 * f.lv_cell_m[lvl], g)
    mx = -g[0]
    my = -g[1]
    mz = -g[2]
    if mx * mx + my * my + mz * mz <= 0.0:
        return False
    if mx * nx + my * ny + mz * nz > -0.3:
        return False
    me = mx if fam == 0 else (my if fam == 1 else mz)
    depth = D * abs(me)
    out[1] = px + depth * mx
    out[2] = py + depth * my
    out[3] = pz + depth * mz
    out[4] = mx
    out[5] = my
    out[6] = mz
    return True


@njit(i8(CUT_T, FIELD_T, F1, F1, F1, F1, i8, f8, f8, F1, I1, i8, i8, I2), **OPTS)
def refill_planes(cut, f, st_z, st_r, st_nr, st_nz, tick, theta, reach, hit, count, i_found, i_miss, pairs):
    """Field planes (cutting Refill) of the memory bins the coming L0 step reads that are stale (or that an
    element reads from the field directly, read_row -1) and have none yet in this macro step: the material
    boundary along the station's chip-thickness normal from the station circle at the bin centre (step mid
    pose), found by removal's ray cast against the field as it is now (architecture 5.4: stale bins are
    refilled from the field; core.hot_kernels.surface_plane). A plane lives for this step only: the (row, bin)
    of every plane set is listed in pairs[1:] (pairs[0, 0] = count; capacity exceeded: pairs[0, 1] = 1) and
    clear_planes removes them after the force kernel, so a later tooth never reads a plane of an earlier pose.
    count[i_found] += planes found, count[i_miss] += bins refilled at the element itself. Returns planes set."""
    tool = cut.tool
    mem = cut.mem
    refill = cut.refill
    pose = cut.pose
    iv = cut.iv
    nb = mem.point_S_m.shape[1]
    inv_bin = nb / TWO_PI
    bw = TWO_PI / nb
    stale_ticks = cut.istep[CA.SI_STALE_TICKS]
    stale_before = cut.istep[CA.SI_STALE_BEFORE]
    nst = st_z.shape[0]
    R = pose.R_S_T0
    tip = pose.tip_S_m
    half = 0.5 * abs(cut.step[CA.ST_OMEGA_RAD_S] * cut.step[CA.ST_DT_S]) + bw
    found = 0
    for s in range(min(nst, iv.count.shape[0])):
        if iv.count[s] == 0:
            continue
        for idx in range(tool.st_ptr[s], tool.st_ptr[s + 1]):
            e = tool.st_elem[idx]
            a = tool.alpha0_rad[e] + theta
            a -= TWO_PI * math.floor(a / TWO_PI)
            if not _in_intervals(cut, s, a, half):
                continue
            row = tool.read_row[e]
            direct = row < 0
            r = s if direct else row
            b_first = int(math.floor(a * inv_bin - 0.5)) - 1
            for d in range(4):
                b = (b_first + d) % nb
                if refill.valid[r, b] != 0:
                    continue
                if not direct and not _stale(cut, r, b, tick, stale_ticks, stale_before):
                    continue
                npair = pairs[0, 0]
                if npair + 1 < pairs.shape[0]:
                    pairs[npair + 1, 0] = r
                    pairs[npair + 1, 1] = b
                    pairs[0, 0] = npair + 1
                else:
                    pairs[0, 1] = 1
                c_b = (b + 0.5) * bw
                cb = math.cos(c_b)
                sb = math.sin(c_b)
                lx = st_r[s] * cb
                ly = st_r[s] * sb
                lz = st_z[s]
                nx_ = st_nr[s] * cb
                ny_ = st_nr[s] * sb
                nz_ = st_nz[s]
                ox = tip[0] + R[0, 0] * lx + R[0, 1] * ly + R[0, 2] * lz
                oy = tip[1] + R[1, 0] * lx + R[1, 1] * ly + R[1, 2] * lz
                oz = tip[2] + R[2, 0] * lx + R[2, 1] * ly + R[2, 2] * lz
                dx = R[0, 0] * nx_ + R[0, 1] * ny_ + R[0, 2] * nz_
                dy = R[1, 0] * nx_ + R[1, 1] * ny_ + R[1, 2] * nz_
                dz = R[2, 0] * nx_ + R[2, 1] * ny_ + R[2, 2] * nz_
                norm = math.sqrt(dx * dx + dy * dy + dz * dz)
                if norm <= 0.0:
                    continue
                dx /= norm
                dy /= norm
                dz /= norm
                if not surface_plane(f, ox, oy, oz, dx, dy, dz, hit):
                    # the field cannot place the boundary here within its resolution (within a cell of an edge,
                    # or farther than SPREAD_CELLS from any): the bin is refilled at the circle point itself, so
                    # this read sees no chip and the memory carries on from the next passage (never the stale point)
                    count[i_miss] += 1
                    hit[1] = ox
                    hit[2] = oy
                    hit[3] = oz
                    hit[4] = -dx
                    hit[5] = -dy
                    hit[6] = -dz
                mx = hit[4]
                my = hit[5]
                mz = hit[6]
                mn = 1.0
                refill.q_S_m[r, b, 0] = hit[1]
                refill.q_S_m[r, b, 1] = hit[2]
                refill.q_S_m[r, b, 2] = hit[3]
                refill.m_S[r, b, 0] = mx / mn
                refill.m_S[r, b, 1] = my / mn
                refill.m_S[r, b, 2] = mz / mn
                refill.valid[r, b] = 1
                found += 1
    count[i_found] += found
    return found


@njit(void(CUT_T, I2), **OPTS)
def clear_planes(cut, pairs):
    """Remove the planes refill_planes set for this step (all of them when its list overflowed)."""
    valid = cut.refill.valid
    if pairs[0, 1] != 0:
        for r in range(valid.shape[0]):
            for b in range(valid.shape[1]):
                valid[r, b] = 0
    else:
        for i in range(pairs[0, 0]):
            valid[pairs[i + 1, 0], pairs[i + 1, 1]] = 0
    pairs[0, 0] = 0
    pairs[0, 1] = 0


@njit(void(POLAR_T, VIEW_T, i8, i8, i8, i8, f8), **OPTS)
def polar_view_update(pf, view, b_lo, b_hi, i_lo, i_hi, side):
    """Cutting's PolarView (outer / inner material radius per bin and row) of bins b_lo..b_hi (wrapped) and rows
    i_lo..i_hi from removal's polar field (radial rays, quanta from the axis).

    The view holds the hull of the ray's segments (first start, last end) and one air gap inside it; a ray of the
    field may hold several segments (behind a major edge that leans back, kappa_r > 90 deg, the previous
    revolution leaves a thin ring above its cut). The gap kept is the one the tool meets first: between the two
    outermost segments for external tools (side <= 0, PolarStep sx), the two innermost for internal ones
    (side > 0). docs/models/hotloop.md "Turning"."""
    nth = int(pf.geom[RA.PG_NTHETA])
    nz = int(pf.geom[RA.PG_NZ])
    q = pf.geom[RA.PG_QUANTUM]
    if nth <= 0 or nz <= 0 or view.r_out_m.shape[0] != nth or view.r_out_m.shape[1] != nz:
        return
    i0 = max(i_lo, 0)
    i1 = min(i_hi, nz - 1)
    for bb in range(b_lo, b_hi + 1):
        b = bb % nth
        if b < 0:
            b += nth
        for i in range(i0, i1 + 1):
            ray = b * nz + i
            n = pf.rcnt[ray]
            view.gap_lo_m[b, i] = 0.0
            view.gap_hi_m[b, i] = 0.0
            if n <= 0:
                view.r_out_m[b, i] = 0.0
                view.r_in_m[b, i] = 0.0
            else:
                view.r_in_m[b, i] = pf.rseg[ray, 0, 0] * q
                view.r_out_m[b, i] = pf.rseg[ray, n - 1, 1] * q
                if n >= 2:
                    if side > 0.0:
                        view.gap_lo_m[b, i] = pf.rseg[ray, 0, 1] * q
                        view.gap_hi_m[b, i] = pf.rseg[ray, 1, 0] * q
                    else:
                        view.gap_lo_m[b, i] = pf.rseg[ray, n - 2, 1] * q
                        view.gap_hi_m[b, i] = pf.rseg[ray, n - 1, 0] * q


@njit(f8(POLAR_T, f8, f8), **OPTS)
def polar_max_radius(pf, z_lo, z_hi):
    """Largest outer material radius of the polar field over the rows touching [z_lo, z_hi] (every bin), from
    its exact per-block maxima (broad phase of a turning contact)."""
    nth = int(pf.geom[RA.PG_NTHETA])
    nz = int(pf.geom[RA.PG_NZ])
    dz = pf.geom[RA.PG_DZ]
    z0 = pf.geom[RA.PG_Z0]
    if nth <= 0 or nz <= 0 or dz <= 0.0:
        return 0.0
    i0 = int(math.floor((z_lo - z0) / dz))
    i1 = int(math.floor((z_hi - z0) / dz))
    if i1 < 0 or i0 >= nz:
        return 0.0
    i0 = max(i0, 0)
    i1 = min(i1, nz - 1)
    nblk = (nz + RA.POLAR_BLOCK - 1) // RA.POLAR_BLOCK
    best = 0
    for b in range(nth):
        for blk in range(i0 // RA.POLAR_BLOCK, i1 // RA.POLAR_BLOCK + 1):
            v = pf.zmax_q[b * nblk + blk]
            if v > best:
                best = v
    return best * pf.geom[RA.PG_QUANTUM]


__all__ = ["hermite", "interp_axes", "fk_frame", "rotvec", "set_step_pose", "displacement_at", "tool_deflection_S",
           "slab_point", "surface_plane", "refill_planes", "clear_planes", "polar_view_update", "polar_max_radius", "WK_A0", "WK_MID", "WK_A1", "WK_DISP", "WK_UREL", "WK_FEED", "WK_PASS", "WK_PFX", "WK_PFY", "WK_PSLACK", "WK_SIZE", "FR_TIP",
           "FR_R", "FR_RM", "FR_TM", "FR_SIZE"]
