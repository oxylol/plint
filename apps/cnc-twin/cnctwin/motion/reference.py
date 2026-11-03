"""Plain-Python references of the motion kernels (:mod:`cnctwin.motion.kernels`).

Each function mirrors one kernel with NumPy/float arithmetic in the same order, so the tests can
compare the compiled and the interpreted results tick by tick (architecture 5.5). They operate on
the same NamedTuples (:mod:`cnctwin.motion.abi`) and mutate state arrays the same way.
"""

from __future__ import annotations

import math

import numpy as np

from .abi import (ALARM_FOLLOWING_ERROR, ALARM_OVERTRAVEL_NEG, ALARM_OVERTRAVEL_POS, FK_M_F, FK_M_N, FK_M_T0, FK_M_W,
                  FK_M_W0, FK_S_T0, FK_W0_T0, GEO_ARC, LINK_LINEAR, LINK_ROTARY, N_FK, SP_END, SP_HOLD_WAIT,
                  SP_PER_REV, SP_THREAD, SPM_CAXIS, SPM_POSITION, SPM_SPEED, SYNC_ALARM, SYNC_IN_POSITION, SYNC_NONE,
                  SYNC_PATH_WAIT, SYNC_PROBE_TRIGGER, SYNC_QUEUE_EMPTY, SYNC_SEGMENT_END, WAIT_PATH)

TWO_PI = 2.0 * math.pi
_BISECT = 14
_SNAP_M = 2e-9
_SNAP_V = 1e-6
_TOL_M = 1e-12


# ------------------------------------------------------------------------------- rotations and poses

def axis_rotation(d, angle) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    x, y, z = float(d[0]), float(d[1]), float(d[2])
    return np.array([[c + x * x * t, x * y * t - z * s, x * z * t + y * s],
                     [y * x * t + z * s, c + y * y * t, y * z * t - x * s],
                     [z * x * t - y * s, z * y * t + x * s, c + z * z * t]])


def rotvec_rotation(rx, ry, rz) -> np.ndarray:
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle < 1e-300:
        return np.eye(3)
    return axis_rotation((rx / angle, ry / angle, rz / angle), angle)


def pose(R=None, t=None) -> np.ndarray:
    out = np.eye(4)
    if R is not None:
        out[:3, :3] = R
    if t is not None:
        out[:3, 3] = t
    return out


def inv(T) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def spin_z(T, angle) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return T @ np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


# ------------------------------------------------------------------------------- chains

def link_motion(chain, p, c, i, q, zero_link=-1) -> np.ndarray:
    kind = int(chain.kind[p, c, i])
    M = np.eye(4)
    qv = 0.0
    if kind in (LINK_LINEAR, LINK_ROTARY):
        ax = int(chain.axis[p, c, i])
        if ax >= 0 and i != zero_link:
            qv = float(q[ax])
        d = chain.direction[p, c, i]
        if kind == LINK_LINEAR:
            M[:3, 3] = d * qv
        else:
            R = axis_rotation(d, qv)
            pt = chain.point[p, c, i]
            M[:3, :3] = R
            M[:3, 3] = pt - R @ pt
    e = chain.err_const[p, c, i].astype(float).copy()
    if kind in (LINK_LINEAR, LINK_ROTARY):
        xi = (qv - chain.err_lo[p, c, i]) / chain.err_span[p, c, i]
        xi = min(max(xi, 0.0), 1.0)
        poly = chain.err_poly[p, c, i]
        e = e + poly[:, 0] + xi * (poly[:, 1] + xi * poly[:, 2])
        if kind == LINK_LINEAR:
            e[:3] = e[:3] + chain.direction[p, c, i] * (chain.strain[p, c, i] * (qv - chain.strain_ref[p, c, i]))
    if np.any(e != 0.0):
        M = M @ pose(rotvec_rotation(e[3], e[4], e[5]), e[:3])
    return M @ chain.fixed[p, c, i]


def chain_pose(chain, p, c, q, zero_link=-1) -> np.ndarray:
    T = np.eye(4)
    for i in range(int(chain.n_links[p, c])):
        T = T @ link_motion(chain, p, c, i, q, zero_link)
    return T


def fk_poses(chain, p, q) -> np.ndarray:
    out = np.zeros((N_FK, 4, 4))
    out[FK_M_N] = chain_pose(chain, p, 1, q)
    out[FK_M_T0] = out[FK_M_N] @ chain.pose_N_T0[p]
    out[FK_M_F] = chain_pose(chain, p, 0, q)
    out[FK_M_W] = out[FK_M_F] @ chain.pose_F_W[p]
    zl = int(chain.work_spindle_link[p])
    out[FK_M_W0] = (chain_pose(chain, p, 0, q, zl) @ chain.pose_F_W[p]) if zl >= 0 else out[FK_M_W]
    out[FK_S_T0] = inv(out[FK_M_W] @ chain.pose_W_S[p]) @ out[FK_M_T0]
    out[FK_W0_T0] = inv(out[FK_M_W0]) @ out[FK_M_T0]
    return out


def tip_in_W(chain, p, q) -> np.ndarray:
    poses = fk_poses(chain, p, q)
    mw, mt = poses[FK_M_W], poses[FK_M_T0]
    return mw[:3, :3].T @ (mt[:3, 3] - mw[:3, 3])


def ik_position(chain, p, q, target, lin, n_iter) -> float:
    n = len(lin)
    res = 0.0
    for _ in range(n_iter):
        tip = tip_in_W(chain, p, q)
        r = tip - target
        res = float(math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]))
        if res < 1e-13 or n == 0:
            break
        h = 1e-6
        jac = np.zeros((3, n))
        for c, a in enumerate(lin):
            keep = q[a]
            q[a] = keep + h
            jac[:, c] = (tip_in_W(chain, p, q) - tip) / h
            q[a] = keep
        step = np.linalg.solve(jac.T @ jac, -(jac.T @ r))
        for c, a in enumerate(lin):
            q[a] += step[c]
    r = tip_in_W(chain, p, q) - target
    return float(math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]))


# ------------------------------------------------------------------------------- feed drives

def _axis_rhs(servo, a, x, v, integ, xc, load, vff, kv, f_ext):
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
    return v, (fm + f_ext + servo.gravity[a] - fric) / servo.mass[a], dint, fm


def servo_axis_step(servo, st, a, cmd, kv, f_ext) -> None:
    dt = float(servo.dt[0])
    n = int(servo.n_sub[0])
    h = dt / n
    c0 = float(st.cmd[a])
    dc = cmd - c0
    if dc > 1e-15:
        dnew = 1.0
    elif dc < -1e-15:
        dnew = -1.0
    else:
        dnew = float(st.dirn[a])
    if dnew != 0.0 and dnew != st.dirn[a]:
        st.dirn[a] = dnew
        st.comp[a] = 0.5 * servo.comp[a] * dnew
    vff = servo.kff[a] * dc / dt
    comp = float(st.comp[a])
    x, v, integ, load = float(st.x[a]), float(st.v[a]), float(st.integ[a]), float(st.load[a])
    half = 0.5 * servo.backlash[a]
    v_start = v
    fm = float(st.force[a])
    for sub in range(n):
        xa = c0 + dc * (sub / n) + comp
        xm = c0 + dc * ((sub + 0.5) / n) + comp
        xb = c0 + dc * ((sub + 1.0) / n) + comp
        k1 = _axis_rhs(servo, a, x, v, integ, xa, load, vff, kv, f_ext)
        k2 = _axis_rhs(servo, a, x + 0.5 * h * k1[0], v + 0.5 * h * k1[1], integ + 0.5 * h * k1[2], xm, load, vff, kv,
                       f_ext)
        k3 = _axis_rhs(servo, a, x + 0.5 * h * k2[0], v + 0.5 * h * k2[1], integ + 0.5 * h * k2[2], xm, load, vff, kv,
                       f_ext)
        k4 = _axis_rhs(servo, a, x + h * k3[0], v + h * k3[1], integ + h * k3[2], xb, load, vff, kv, f_ext)
        x += h / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        v += h / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        integ += h / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        fm = (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0
        if x - load > half:
            load = x - half
        elif x - load < -half:
            load = x + half
    st.x[a], st.v[a], st.integ[a], st.load[a], st.force[a] = x, v, integ, load, fm
    st.acc[a] = (v - v_start) / dt
    st.cmd[a] = cmd


# ------------------------------------------------------------------------------- spindles

def torque_limit(sp, s, gear, omega) -> float:
    w = abs(omega)
    n = int(sp.n_curve[s, gear])
    cw, ct = sp.curve_w[s, gear], sp.curve_t[s, gear]
    if n <= 0:
        return 0.0
    if w <= cw[0]:
        return float(ct[0])
    for i in range(1, n):
        if w <= cw[i]:
            span = cw[i] - cw[i - 1]
            if span <= 0.0:
                return float(ct[i])
            return float(ct[i - 1] + (ct[i] - ct[i - 1]) * (w - cw[i - 1]) / span)
    return float(ct[n - 1])


def _spindle_rhs(sp, s, gear, theta, omega, integ, wref_speed, theta_c, wff, kv, mode, t_load):
    wref = wref_speed if mode == SPM_SPEED else kv * (theta_c - theta) + wff
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
    return omega, (tm - t_load - drag) / sp.inertia[s], dint, tm


def spindle_step(sp, st, s, mode, wref, theta_c, wff, kv, t_load) -> None:
    dt = float(sp.dt[0])
    n = int(sp.n_sub[0])
    h = dt / n
    theta, omega, integ = float(st.angle[s]), float(st.omega[s]), float(st.integ[s])
    c0 = float(st.cmd_angle[s])
    dcmd = theta_c - c0
    if mode == SPM_SPEED:
        dcmd = 0.0
        c0 = theta
    g = int(st.gear[s])
    tm = float(st.torque[s])
    for sub in range(n):
        ca = c0 + dcmd * (sub / n)
        cm = c0 + dcmd * ((sub + 0.5) / n)
        cb = c0 + dcmd * ((sub + 1.0) / n)
        k1 = _spindle_rhs(sp, s, g, theta, omega, integ, wref, ca, wff, kv, mode, t_load)
        k2 = _spindle_rhs(sp, s, g, theta + 0.5 * h * k1[0], omega + 0.5 * h * k1[1], integ + 0.5 * h * k1[2], wref, cm,
                          wff, kv, mode, t_load)
        k3 = _spindle_rhs(sp, s, g, theta + 0.5 * h * k2[0], omega + 0.5 * h * k2[1], integ + 0.5 * h * k2[2], wref, cm,
                          wff, kv, mode, t_load)
        k4 = _spindle_rhs(sp, s, g, theta + h * k3[0], omega + h * k3[1], integ + h * k3[2], wref, cb, wff, kv, mode,
                          t_load)
        theta += h / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        omega += h / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        integ += h / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        tm = (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0
    st.angle[s], st.omega[s], st.integ[s], st.torque[s] = theta, omega, integ, tm
    st.cmd_angle[s] = theta_c if mode != SPM_SPEED else theta


def ramp_toward(current, target, up_rate, down_rate, dt) -> float:
    target_mag = 0.0 if current * target < 0.0 else abs(target)
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


# ------------------------------------------------------------------------------- geometry

def geo_eval(kind, length, vec, par, s, axes, out) -> None:
    u = s / length if length > 0.0 else 1.0
    u = min(max(u, 0.0), 1.0)
    if kind == GEO_ARC:
        r = par[0] + (par[1] - par[0]) * u
        ph = par[2] * u
        cr, sr = r * math.cos(ph), r * math.sin(ph)
        for a in axes:
            out[a] = vec[0, a] + cr * vec[1, a] + sr * vec[2, a] + vec[3, a] * u
    else:
        for a in axes:
            out[a] = vec[0, a] + vec[1, a] * u


def geo_eval_many(kind, length, vec, par, s, axes, out) -> None:
    for i in range(len(s)):
        geo_eval(kind, length, vec, par, float(s[i]), axes, out[i])


# ------------------------------------------------------------------------------- the L1 tick

def _table_s(setp, p, dtheta, theta_tab, s_tab) -> float:
    n = int(setp.sync_n[p])
    if n <= 0:
        return 0.0
    if n == 1 or dtheta <= theta_tab[0]:
        return float(s_tab[0])
    for i in range(1, n):
        if dtheta <= theta_tab[i]:
            span = theta_tab[i] - theta_tab[i - 1]
            if span <= 0.0:
                return float(s_tab[i])
            return float(s_tab[i - 1] + (s_tab[i] - s_tab[i - 1]) * (dtheta - theta_tab[i - 1]) / span)
    span = theta_tab[n - 1] - theta_tab[n - 2]
    slope = (s_tab[n - 1] - s_tab[n - 2]) / span if span > 0.0 else 0.0
    return float(s_tab[n - 1] + slope * (dtheta - theta_tab[n - 1]))


def _stretch_command(setp, p, s, axes, out) -> None:
    g = 0
    for j in range(int(setp.geo_n[p])):
        if s >= setp.geo_s0[p, j]:
            g = j
    geo_eval(int(setp.geo_kind[p, g]), float(setp.geo_len[p, g]), setp.geo_vec[p, g], setp.geo_par[p, g],
             s - float(setp.geo_s0[p, g]), axes, out)


def motion_tick(chain, servo, ss, sp, sps, setp, ls, out, k, dist_force, dist_torque) -> int:
    dt = float(servo.dt[0])
    n_axes = len(servo.kv)
    n_paths = setp.mode.shape[1]
    n_sp = len(sp.inertia)
    cmdk = setp.cmd[k].astype(float).copy()
    for p in range(n_paths):
        if setp.sync_start[k, p] == 1:
            s_id = int(setp.sync_spindle[p])
            ls.theta0[p] = sps.angle[s_id] if s_id >= 0 else 0.0
            ls.theta_prev[p] = ls.theta0[p]
            ls.stage[p] = 0
            ls.s_act[p] = 0.0
            ls.s_filt[p] = 0.0
    for s in range(n_sp):
        mode = int(setp.sp_mode[k, s])
        if mode == SPM_SPEED:
            target = float(setp.sp_omega[k, s])
            vc = float(setp.css_vc[k, s])
            ca = int(setp.css_axis[s])
            if vc > 0.0 and ca >= 0 and target != 0.0:
                r = max(abs(ss.cmd[ca] - setp.css_centre[s]), float(setp.css_rmin[0]))
                w = min(vc / r, float(setp.css_clamp[k, s]), float(sp.omega_max[s]))
                target = w if target > 0.0 else -w
            sps.cmd_omega[s] = ramp_toward(float(sps.cmd_omega[s]), target, float(sp.accel_rate[s]),
                                           float(sp.decel_rate[s]), dt)
            spindle_step(sp, sps, s, SPM_SPEED, float(sps.cmd_omega[s]), 0.0, 0.0, 0.0, float(dist_torque[s]))
        elif mode == SPM_POSITION:
            sps.cmd_omega[s] = setp.sp_omega_ff[k, s]
            spindle_step(sp, sps, s, SPM_POSITION, 0.0, float(setp.sp_angle[k, s]),
                         float(sp.pos_kff[s] * setp.sp_omega_ff[k, s]), float(sp.pos_kv[s]), float(dist_torque[s]))
        else:
            ca = int(sp.c_axis[s])
            kap = float(servo.kappa[ca]) if ca >= 0 else 1.0
            th = (cmdk[ca] - float(ss.c_off[ca])) / kap if ca >= 0 else float(sps.cmd_angle[s])
            wff = (th - sps.cmd_angle[s]) / dt
            sps.cmd_omega[s] = wff
            spindle_step(sp, sps, s, SPM_CAXIS, 0.0, th, float(sp.c_kff[s] * wff), float(sp.c_kv[s]),
                         float(dist_torque[s]))
        out.sp_angle[k, s] = sps.angle[s]
        out.sp_omega[k, s] = sps.omega[s]
        out.sp_torque[k, s] = sps.torque[s]
    code_first = SYNC_NONE
    out.sync[k, :] = SYNC_NONE
    out.alarm[k, :] = 0
    out.alarm_axis[k, :] = -1
    seg_end = np.zeros(n_paths, np.int8)
    all_axes = list(range(n_axes))
    for p in range(n_paths):
        m = int(setp.mode[k, p])
        if m not in (SP_PER_REV, SP_THREAD):
            continue
        s_id = int(setp.sync_spindle[p])
        theta = float(sps.angle[s_id]) if s_id >= 0 else 0.0
        s_end = float(setp.sync_s_end[p])
        if m == SP_PER_REV:
            dth = abs(theta - ls.theta0[p])
            s_now = min(_table_s(setp, p, dth, setp.sync_theta[p], setp.sync_s[p]), s_end)
            s_now = max(s_now, float(ls.s_act[p]))
            ls.s_act[p] = s_now
            if s_now >= s_end - 1e-12:
                seg_end[p] = 1
        else:
            if ls.stage[p] == 0:
                trig = float(setp.thread_trig[p])
                prev = float(ls.theta_prev[p])
                n0 = math.floor((prev - trig) / TWO_PI)
                n1 = math.floor((theta - trig) / TWO_PI)
                if n1 != n0:
                    ls.stage[p] = 1
                    ls.theta_trig[p] = trig + TWO_PI * (n1 if n1 > n0 else n0)
                    ls.s_filt[p] = 0.0
            if ls.stage[p] == 1:
                raw = min(setp.thread_lead[p] * abs(theta - ls.theta_trig[p]) / TWO_PI, s_end)
                tc = float(setp.thread_tc[p])
                if tc > 0.0:
                    ls.s_filt[p] += (raw - ls.s_filt[p]) * (1.0 - math.exp(-dt / tc))
                    if raw >= s_end and s_end - ls.s_filt[p] < 1e-7:
                        ls.s_filt[p] = s_end
                else:
                    ls.s_filt[p] = raw
                s_now = float(ls.s_filt[p])
            else:
                s_now = 0.0
            ls.s_act[p] = s_now
            if ls.stage[p] == 1 and s_now >= s_end - 1e-12:
                seg_end[p] = 1
        ls.theta_prev[p] = theta
        tmpq = np.empty(n_axes)
        _stretch_command(setp, p, float(ls.s_act[p]), all_axes, tmpq)
        for a in range(n_axes):
            if servo.path[a] == p:
                cmdk[a] = tmpq[a]
        out.sync_s[k, p] = ls.s_act[p]
    for a in range(n_axes):
        out.cmd[k, a] = cmdk[a]
        s_link = int(servo.spindle[a])
        p = int(servo.path[a])
        if s_link >= 0:
            act = servo.kappa[a] * sps.angle[s_link] + ss.c_off[a]
            caxis = setp.sp_mode[k, s_link] == SPM_CAXIS
            out.actual[k, a] = act
            if not caxis:
                out.cmd[k, a] = act
            out.ferr[k, a] = cmdk[a] - act if caxis else 0.0
            ss.load[a] = act
            ss.x[a] = act
            ss.cmd[a] = cmdk[a] if caxis else act
            out.force[k, a] = 0.0
            out.inertial[k, a] = 0.0
            out.overload[k, a] = 0
            continue
        kv = float(servo.tap_kv[a]) if setp.tap[k, p] == 1 else float(servo.kv[a])
        v_prev = float(ss.vel_prev[a])
        vc = (cmdk[a] - ss.cmd[a]) / dt
        servo_axis_step(servo, ss, a, float(cmdk[a]), kv, float(dist_force[a]))
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
    q = out.actual[k]
    for p in range(n_paths):
        poses = fk_poses(chain, p, q)
        out.poses[p] = poses
        out.tip_M[k, p] = poses[FK_M_T0][:3, 3]
        out.tip_S[k, p] = poses[FK_S_T0][:3, 3]
        out.tip_W0[k, p] = poses[FK_W0_T0][:3, 3]
        out.axis_M[k, p] = poses[FK_M_T0][:3, 2]
        code = SYNC_NONE
        m = int(setp.mode[k, p])
        if out.alarm[k, p] != 0:
            code = SYNC_ALARM
        elif setp.probe[k, p] == 1 and out.probe_hit[k, p] == 1:
            code = SYNC_PROBE_TRIGGER
        elif setp.setter[k, p] == 1:
            d = out.tip_M[k, p] - out.axis_M[k, p] * setp.setter_extra[p] - setp.setter_point[p]
            rad = float(setp.setter_radius[p])
            if rad <= 0.0:
                if float(d @ setp.setter_normal[p]) <= 0.0:
                    code = SYNC_PROBE_TRIGGER
            else:
                axis = out.axis_M[k, p]
                along = -float(d @ axis)
                perp = -d - along * axis
                if 0.0 <= along <= 5.0 * rad and float(np.linalg.norm(perp)) <= rad:
                    code = SYNC_PROBE_TRIGGER
        if code == SYNC_NONE:
            if m == SP_END:
                code = SYNC_QUEUE_EMPTY
            elif m in (SP_PER_REV, SP_THREAD) and seg_end[p] == 1:
                code = SYNC_SEGMENT_END
            elif m == SP_HOLD_WAIT:
                inpos = True
                for a in range(n_axes):
                    if servo.path[a] != p:
                        continue
                    s_link = int(servo.spindle[a])
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


def replan_before(setp, k) -> int:
    for p in range(setp.mode.shape[1]):
        if setp.mode[k, p] == SP_END:
            return p
    return -1


def run_motion(chain, servo, ss, sp, sps, setp, ls, out, k_from, k_to, dist_force, dist_torque, result) -> int:
    result[0], result[1], result[2] = SYNC_NONE, k_to - 1, -1
    for k in range(k_from, k_to):
        if k > k_from:
            p_end = replan_before(setp, k)
            if p_end >= 0:
                out.sync[k - 1, p_end] = SYNC_QUEUE_EMPTY
                result[0], result[1], result[2] = SYNC_QUEUE_EMPTY, k - 1, p_end
                return SYNC_QUEUE_EMPTY
        code = motion_tick(chain, servo, ss, sp, sps, setp, ls, out, k, dist_force, dist_torque)
        if code != SYNC_NONE:
            result[0], result[1] = code, k
            hits = np.nonzero(out.sync[k] != SYNC_NONE)[0]
            result[2] = int(hits[0]) if len(hits) else -1
            return code
    return SYNC_NONE


# ------------------------------------------------------------------------------- generator

def brake_distance(v, a, u, amax, jmax) -> float:
    J = jmax
    vn = v + a * a / (2.0 * J) if a > 0.0 else v - a * a / (2.0 * J)
    if vn <= u:
        if v <= u:
            return 0.0
        disc = max(a * a - 2.0 * J * (v - u), 0.0)
        t = max((-a - math.sqrt(disc)) / J, 0.0)
        return v * t + 0.5 * a * t * t + J * t * t * t / 6.0
    dv = u - v
    ap2 = 0.5 * (a * a - 2.0 * J * dv)
    ap = -math.sqrt(ap2) if ap2 > 0.0 else 0.0
    t2 = 0.0
    if ap < -amax:
        ap = -amax
        t2 = max((dv - (a * a - 2.0 * amax * amax) / (2.0 * J)) / ap, 0.0)
    t1 = max((a - ap) / J, 0.0)
    t3 = -ap / J
    d1 = v * t1 + 0.5 * a * t1 * t1 - J * t1 * t1 * t1 / 6.0
    v1 = v + a * t1 - 0.5 * J * t1 * t1
    d2 = v1 * t2 + 0.5 * ap * t2 * t2
    v2 = v1 + ap * t2
    d3 = v2 * t3 + 0.5 * ap * t3 * t3 + J * t3 * t3 * t3 / 6.0
    return d1 + d2 + d3


def natural_velocity(v, a, jj, dt, jmax) -> float:
    a1 = a + jj * dt
    v1 = v + a * dt + 0.5 * jj * dt * dt
    return v1 + a1 * a1 / (2.0 * jmax) if a1 > 0.0 else v1 - a1 * a1 / (2.0 * jmax)


def track_jerk(v, a, vt, amax, jmax, dt) -> float:
    J = jmax
    j_hi = min(J, (amax - a) / dt)
    j_lo = max(-J, (-amax - a) / dt)
    if j_hi < j_lo:
        j_hi = j_lo
    if natural_velocity(v, a, j_hi, dt, J) <= vt:
        return j_hi
    if natural_velocity(v, a, j_lo, dt, J) > vt:
        return j_lo
    lo, hi = j_lo, j_hi
    for _ in range(_BISECT + 10):
        mid = 0.5 * (lo + hi)
        if natural_velocity(v, a, mid, dt, J) <= vt:
            lo = mid
        else:
            hi = mid
    return lo


def _nodes_ok(v1, a1, moved, tol, nodes) -> bool:
    if v1 < -1e-9:
        return False
    for x, u, A, J in nodes:
        d = x - moved
        if d < 0.0:
            continue
        if brake_distance(max(v1, 0.0), a1, u, A, J) > d + tol:
            return False
    return True


def plan_ticks(L, vmax, amax, jmax, ustart, ready, stop_end, n_pieces, lookahead, gi, gf, t_first, ipo, dt, n_ticks,
               out_piece, out_s, out_v, out_a, out_lim, out_count) -> int:
    out_count[0] = 0
    for i in range(n_ticks):
        t = t_first + i * ipo
        k = int(gi[0])
        s, v, a = float(gf[0]), float(gf[1]), float(gf[2])
        if k >= n_pieces:
            out_count[0] = i
            return 2
        if ready[k] > t:
            out_piece[i], out_s[i], out_v[i], out_a[i] = k, s, 0.0, 0.0
            out_lim[i] = 2
            gf[1] = gf[2] = 0.0
            continue
        e = k
        while stop_end[e] == 0 and e + 1 < n_pieces and e + 1 < k + lookahead and ready[e + 1] <= t:
            e += 1
        nodes = []
        x = L[k] - s
        a_run, j_run = amax[k], jmax[k]
        for j in range(k + 1, e + 1):
            nodes.append((x, ustart[j], a_run, j_run))
            a_run = min(a_run, amax[j])
            j_run = min(j_run, jmax[j])
            x += L[j]
        nodes.append((x, 0.0, a_run, j_run))
        x_end = x
        A, J, vcap = amax[k], jmax[k], vmax[k]
        tol = _TOL_M + 1e-9 * x_end
        j_cap = track_jerk(v, a, vcap, A, J, dt)
        j_lo = max(-J, (-A - a) / dt)
        j_cap = max(j_cap, j_lo)

        def step(jj):
            return (a + jj * dt, v + a * dt + 0.5 * jj * dt * dt, v * dt + 0.5 * a * dt * dt + jj * dt * dt * dt / 6.0)

        artificial = stop_end[e] == 0 and e < n_pieces - 1
        a1, v1, d1 = step(j_cap)
        lim = 0
        if _nodes_ok(v1, a1, d1, tol, nodes):
            jj = j_cap
        else:
            lim = 1
            if artificial and _nodes_ok(v1, a1, d1, tol, nodes[:-1]):
                lim = 2
            a1, v1, d1 = step(j_lo)
            if not _nodes_ok(v1, a1, d1, tol, nodes):
                jj = j_lo
            else:
                lo, hi = j_lo, j_cap
                for _ in range(_BISECT):
                    mid = 0.5 * (lo + hi)
                    a1, v1, d1 = step(mid)
                    if _nodes_ok(v1, a1, d1, tol, nodes):
                        lo = mid
                    else:
                        hi = mid
                jj = lo
        a1, v1, d1 = step(jj)
        if v1 < 0.0:
            lo_t, hi_t = 0.0, dt
            for _ in range(40):
                mid = 0.5 * (lo_t + hi_t)
                if v + a * mid + 0.5 * jj * mid * mid > 0.0:
                    lo_t = mid
                else:
                    hi_t = mid
            tz = lo_t
            d1 = v * tz + 0.5 * a * tz * tz + jj * tz * tz * tz / 6.0
            v1 = a1 = 0.0
        d1 = max(d1, 0.0)
        remaining = x_end - d1
        if remaining < _SNAP_M and v1 < 1e-3 or (remaining < 1e-7 and v1 < _SNAP_V) or remaining < 0.0:
            d1, v1, a1 = x_end, 0.0, 0.0
        s1 = s + d1
        while k < e and s1 > L[k]:
            s1 -= L[k]
            k += 1
        s1 = min(s1, L[k])
        gi[0] = k
        gf[0], gf[1], gf[2] = s1, v1, a1
        out_piece[i], out_s[i], out_v[i], out_a[i] = k, s1, v1, a1
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
