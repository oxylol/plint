"""Feed drives and spindles in kernel form (docs/models/motion.md "Servo").

Feed drive of each axis (Altintas 2012 ch. 6; Erkorkmaz & Altintas 2001 part II):

    v_ref = Kv (x_c + c_b - x_meas) + k_ff dx_c/dt                  position loop + velocity feed-forward
    F_m   = sat(Kp_v (v_ref - v) + Ki_v int(v_ref - v) dt)           PI velocity loop (conditional anti-windup)
    m dv/dt = F_m + F_ext + F_g - F_c tanh(v / v_s)                  moving mass, cutting load, gravity, friction
    backlash: the load follows the motor through a dead band of width b (Nordin & Gutman 2002)

with Kp_v = 2 zeta omega_v m, Ki_v = omega_v^2 m (closed velocity loop of bandwidth omega_v =
2 pi velocity_bw_Hz and damping zeta), x_meas the motor position (rotary encoder) or the load position
(linear scale), c_b = +-c/2 the backlash compensation the control adds when the commanded direction
reverses, c = b (1 - residual fraction). The Coulomb friction F_c is calibrated so that its sign
change at a velocity reversal (a force step 2 F_c) produces the machine's published or estimated
reversal spike: F_c = spike / (2 h_peak), h_peak the peak position response of the closed loop to a
unit force step (computed with the drive kernel itself). Integration: RK4 on n_sub sub-steps per L1
tick with the command fine-interpolated linearly (the drive's position loop runs faster than IPO).

Spindle: J domega/dt = T_m - T_cut - T_drag, T_m = sat(Kp e + Ki int e) within the gear range's torque
curve, T_drag = (P_idle / omega_max) (|omega| / omega_max)^(2/3) (bearing drag following Palmgren's
viscous moment, Palmgren 1959), Kp = 2 zeta omega_b J, Ki = omega_b^2 J; the speed command ramps at
max speed / accel_time (decel_time). Position mode: omega_ref = Kv (theta_c - theta) + k_ff omega_c.
"""

from __future__ import annotations

import math

import numpy as np

from .abi import Servo, ServoState, Spindle, SpindleState, empty_servo, empty_servo_state
from .kernels import servo_axis_step

# numerics of the sub-stepping (physics.motion servo_omega_h / friction_h, cnctwin.motion.schema)
DEFAULT_OMEGA_H = 0.25
DEFAULT_FRICTION_H = 1.0


def _closed_loop_peak(kv: float, kp: float, ki: float, mass: float, fmax: float, dt: float, n_sub: int,
                      bw_Hz: float) -> float:
    """Peak |position| (per N) of the closed position/velocity loop after a unit force step at rest."""
    servo = empty_servo(1)
    servo.kv[0] = kv
    servo.kff[0] = 0.0
    servo.kp_v[0] = kp
    servo.ki_v[0] = ki
    servo.mass[0] = mass
    servo.force_max[0] = max(fmax, 1e6)
    servo.friction[0] = 0.0
    servo.backlash[0] = 0.0
    servo.comp[0] = 0.0
    servo.n_sub[0] = n_sub
    servo.dt[0] = dt
    st = empty_servo_state(1)
    work = np.zeros(32)
    horizon = max(10.0 / (2.0 * math.pi * bw_Hz), 6.0 / kv)
    peak = 0.0
    for _ in range(int(math.ceil(horizon / dt))):
        servo_axis_step(servo, st, 0, 0.0, kv, 1.0, work)
        peak = max(peak, abs(float(st.load[0])))
    return peak


def build_servo(model, dt: float, omega_h: float = DEFAULT_OMEGA_H, friction_h: float = DEFAULT_FRICTION_H) -> Servo:
    """Servo arrays for every axis of the machine model (global order)."""
    p = model.p
    axes = model.axes
    n = len(axes)
    zeta = p["velocity_loop_zeta"]
    rot_len = model.rot_len
    servo = empty_servo(n)
    friction = np.zeros(n)
    v_s = np.zeros(n)
    n_sub_need = 1
    for ax in axes:
        a = ax.index
        omega = 2.0 * math.pi * ax.bw_Hz
        servo.path[a] = ax.path
        servo.rotary[a] = 1 if ax.rotary else 0
        servo.kv[a] = ax.kv
        servo.kff[a] = ax.kff
        servo.kp_v[a] = 2.0 * zeta * omega * ax.mass
        servo.ki_v[a] = omega * omega * ax.mass
        servo.mass[a] = ax.mass
        servo.force_max[a] = ax.force_max
        servo.gravity[a] = ax.gravity_N
        servo.backlash[a] = ax.backlash
        servo.comp[a] = ax.backlash * (1.0 - p["backlash_comp_residual_frac"])
        servo.scale_fb[a] = 1 if ax.scale_fb else 0
        band = model.in_position / rot_len if ax.rotary else model.in_position
        servo.in_pos[a] = band
        # the stored stroke check works on the control's position resolution (half an input increment)
        half = 0.5 * (model.rot_increment if ax.rotary else model.lin_increment)
        servo.lo[a] = ax.lo - half
        servo.hi[a] = ax.hi + half
        servo.ferr_max[a] = p["ferr_limit_factor"] * ax.rapid / ax.kv if ax.kv > 0 else math.inf
        servo.spindle[a] = -1
        servo.kappa[a] = ax.kappa
        servo.tap_kv[a] = ax.kv
        v_s[a] = p["stribeck_velocity_m_s"] / rot_len if ax.rotary else p["stribeck_velocity_m_s"]
        n_sub_need = max(n_sub_need, int(math.ceil(dt * omega / omega_h)))
    servo.n_sub[0] = n_sub_need
    servo.dt[0] = dt
    # friction calibrated to the reversal spike
    for ax in axes:
        a = ax.index
        if ax.synthetic or ax.spike <= 0.0:
            continue
        h_peak = _closed_loop_peak(ax.kv, float(servo.kp_v[a]), float(servo.ki_v[a]), ax.mass, ax.force_max, dt,
                                   int(servo.n_sub[0]), ax.bw_Hz)
        if h_peak <= 0.0:
            continue
        fc = ax.spike / (2.0 * h_peak)
        friction[a] = min(fc, p["friction_max_frac"] * ax.force_max)
    for ax in axes:
        a = ax.index
        if friction[a] > 0.0:
            # RK4 stability of the regularised friction: h F_c / (m v_s) <= _FRICTION_H
            n_sub_need = max(n_sub_need, int(math.ceil(dt * friction[a] / (ax.mass * v_s[a] * friction_h))))
    servo.n_sub[0] = n_sub_need
    servo.friction[:] = friction
    servo.v_stribeck[:] = v_s
    for sp in model.spindles:
        if sp.c_axis is not None and sp.c_axis in model.axis_index:
            servo.spindle[model.axis_index[sp.c_axis]] = sp.index
    # rigid tapping: the feed axes of a path run with the loop gain of the path's tapping spindle (FANUC sets
    # the tapping axis' position gain equal to the spindle's, which removes the steady synchronisation error)
    for pd in model.paths:
        sid = pd.tool_spindle or pd.work_spindle
        sp = model.spindle(sid)
        if sp is None:
            continue
        for name in pd.axes:
            ax = model.axis(name)
            if not ax.rotary and not ax.synthetic:
                servo.tap_kv[ax.index] = sp.pos_kv
    return servo


def initial_servo_state(servo: Servo, q0: np.ndarray) -> ServoState:
    """Drives at rest at q0, gravity and friction preloads held by the integrators."""
    st = empty_servo_state(len(q0))
    st.x[:] = q0
    st.load[:] = q0
    st.cmd[:] = q0
    st.integ[:] = -servo.gravity
    st.force[:] = -servo.gravity
    return st


def build_spindles(model, dt: float, omega_h: float = DEFAULT_OMEGA_H) -> Spindle:
    """Spindle arrays (machine order). A spindle with no gear range gets a zero torque curve (it cannot
    turn: the project check refuses cutting with it)."""
    p = model.p
    S = max(1, len(model.spindles))
    NG = max([1] + [len(s.gears) for s in model.spindles])
    NC = max([2] + [len(g.curve_w) for s in model.spindles for g in s.gears])
    zeta = p["spindle_zeta"]
    inertia = np.ones(S)
    kp = np.ones(S)
    ki = np.ones(S)
    drag = np.zeros(S)
    wmax = np.ones(S)
    up = np.full(S, np.inf)
    down = np.full(S, np.inf)
    pos_kv = np.full(S, p["default_position_loop_kv_1_s"])
    pos_kff = np.zeros(S)
    c_kv = np.full(S, p["default_position_loop_kv_1_s"])
    c_kff = np.zeros(S)
    c_axis = np.full(S, -1, np.int32)
    curve_w = np.zeros((S, NG, NC))
    curve_t = np.zeros((S, NG, NC))
    n_curve = np.zeros((S, NG), np.int32)
    n_sub = 1
    for sp in model.spindles:
        s = sp.index
        omega = 2.0 * math.pi * sp.bw_Hz
        inertia[s] = sp.inertia
        kp[s] = 2.0 * zeta * omega * sp.inertia
        ki[s] = omega * omega * sp.inertia
        drag[s] = sp.drag_max
        wmax[s] = sp.max_omega if sp.max_omega > 0 else 1.0
        up[s] = sp.accel_rate
        down[s] = sp.decel_rate
        pos_kv[s] = sp.pos_kv
        pos_kff[s] = sp.pos_kff
        if sp.c_axis is not None and sp.c_axis in model.axis_index:
            ax = model.axis(sp.c_axis)
            c_axis[s] = ax.index
            if not ax.synthetic:
                c_kv[s] = ax.kv
                c_kff[s] = ax.kff
        for g, gear in enumerate(sp.gears):
            m = len(gear.curve_w)
            curve_w[s, g, :m] = gear.curve_w
            curve_t[s, g, :m] = gear.curve_t
            if m < NC:
                curve_w[s, g, m:] = gear.curve_w[-1]
                curve_t[s, g, m:] = gear.curve_t[-1]
            n_curve[s, g] = m
        n_sub = max(n_sub, int(math.ceil(dt * max(omega, pos_kv[s], c_kv[s]) / omega_h)))
    return Spindle(inertia, kp, ki, drag, wmax, up, down, pos_kv, pos_kff, c_kv, c_kff, c_axis, curve_w, curve_t,
                   n_curve, np.array([n_sub], np.int32), np.array([dt]))


def initial_spindle_state(model) -> SpindleState:
    S = max(1, len(model.spindles))
    return SpindleState(np.zeros(S), np.zeros(S), np.zeros(S), np.zeros(S), np.zeros(S), np.zeros(S, np.int32),
                        np.zeros(S))


def circle_radius_ratio(kv: float, kff: float, bw_Hz: float, zeta: float, omega_c: float) -> float:
    """|x/x_c| of the linear drive model on a circle of angular speed omega_c (steady state): the closed
    form the servo radius-reduction test compares against.

    Velocity loop V/V_ref = (2 zeta w s + w^2) / (s^2 + 2 zeta w s + w^2) =: H(s); position
    X = (Kv (Xc - X) + kff s Xc) H / s  ->  X / Xc = (Kv + kff s) H / (s + Kv H)."""
    w = 2.0 * math.pi * bw_Hz
    s = 1j * omega_c
    H = (2.0 * zeta * w * s + w * w) / (s * s + 2.0 * zeta * w * s + w * w)
    G = (kv + kff * s) * H / (s + kv * H)
    return float(abs(G))
