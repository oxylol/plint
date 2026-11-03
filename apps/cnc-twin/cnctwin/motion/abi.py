"""Kernel layouts of the motion domain (docs/architecture.md 5.2-5.5, docs/models/motion.md "Kernel ABI").

Everything is a NamedTuple of writable C-contiguous arrays so Numba types it once (one signature
per kernel). Sizes: ``P`` controller paths, ``NA`` machine axes (global order, every axis belongs to
one path), ``S`` spindles, ``L`` links per chain (padded), ``T`` L1 ticks of a macro step, ``G``
geometry pieces of a spindle-synchronous stretch, ``NG`` gear ranges, ``NC`` torque-curve points.

* :class:`Chain` - kinematic chains of every path (workpiece chain 0, tool chain 1), per-link fixed
  transforms, geometric error motions (ISO 230-1) as position polynomials, thermal error transforms
  (6-component, never axis offsets), the tool tip and the workpiece/stock frames.
* :class:`Servo` / :class:`ServoState` - position loop Kv with velocity feed-forward, PI velocity
  loop of given bandwidth on the moving mass, Coulomb friction (reversal spike), backlash with the
  control's compensation, force limit, gravity, travel limits, in-position band.
* :class:`Spindle` / :class:`SpindleState` - speed loop on the rotor inertia with the gear range's
  torque curve, drag torque, ramp rates; position loop (orientation, rigid tapping) and C-axis gains.
* :class:`Setpoint` - commanded axis values per L1 tick and the flags that make the compiled loop
  compute per-revolution, threading and C-axis setpoints from the actual spindle angle.
* :class:`LoopState` - per-path state of the synchronous evaluation (thread trigger, filter).
* :class:`MotionOut` - per-tick results (actual, command, following error, forces, tip and tool axis
  in M, S and W0, spindle angle/speed/torque, sync codes), plus fields the hot loop fills
  (``in_cut``, ``probe_hit``).

Sign conventions (architecture 4): spindle angles and speeds are right-handed about the spindle's
reference axis +z (from the workpiece toward the spindle for tool spindles; from the workpiece
toward the headstock for work spindles), so M03 gives omega < 0. A C axis that is a spindle has
``q_C = kappa * theta + c_off`` (``Servo.kappa``; ``ServoState.c_off`` a whole number of turns the
planner sets when the spindle enters C-axis mode, so the C coordinate continues the queued program).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

# ------------------------------------------------------------------------------- links and chains
LINK_FIXED = 0
LINK_LINEAR = 1
LINK_ROTARY = 2
CHAIN_WORK = 0
CHAIN_TOOL = 1

# FK output rows (fk_poses writes out[FK_*] as 4x4 poses)
FK_M_T0 = 0         # non-rotating tool frame in M
FK_M_W = 1          # workpiece frame in M (rotating with a work spindle)
FK_M_W0 = 2         # non-rotating workpiece frame in M (work spindle angle 0)
FK_S_T0 = 3         # tool frame in the stock frame
FK_W0_T0 = 4        # tool frame in the non-rotating workpiece frame
FK_M_N = 5          # spindle nose / turret station
FK_M_F = 6          # fixture mounting face (end of the workpiece chain)
N_FK = 7

# ------------------------------------------------------------------------------- sync codes
SYNC_NONE = 0           # the tick completed, nothing to report
SYNC_QUEUE_EMPTY = 1    # the plan of the step is exhausted (the planner must run: new commands may be ready)
SYNC_IN_POSITION = 2    # an in-position wait completed (exact stop, SYNC/STOP/PROBE/TOOL_SETTER end, dwell start)
SYNC_PROBE_TRIGGER = 3  # probe (removal's sweep_contact via the hot loop) or tool-setter trigger
SYNC_ALARM = 4          # machine alarm: overtravel, excess following error
SYNC_PATH_WAIT = 5      # in position at a path wait
SYNC_SEGMENT_END = 6    # a spindle-synchronous stretch (per revolution, threading) ended
SYNC_NAMES = ("none", "queue_empty", "in_position", "probe_trigger", "alarm", "path_wait", "segment_end")

# alarm detail codes (MotionOut.alarm_axis gives the axis)
ALARM_NONE = 0
ALARM_OVERTRAVEL_POS = 1
ALARM_OVERTRAVEL_NEG = 2
ALARM_FOLLOWING_ERROR = 3

# ------------------------------------------------------------------------------- setpoint modes (per path)
SP_TIME = 0         # cmd[k] is the commanded value
SP_PER_REV = 1      # path position from the actual spindle angle through the (theta, s) table
SP_THREAD = 2       # threading: wait for the one-revolution trigger angle, then s = lead * dtheta / 2 pi, filtered
SP_HOLD_WAIT = 3    # hold cmd[k] and report SYNC_IN_POSITION / SYNC_PATH_WAIT once in position
SP_END = 4          # nothing planned: report SYNC_QUEUE_EMPTY and hold
SP_IDLE = 5         # hold cmd[k] without reporting (waiting for the controller or the operator)

WAIT_INPOS = 1      # Setpoint.wait_kind: plain in-position wait
WAIT_PATH = 2       # in-position at a path wait

# ------------------------------------------------------------------------------- spindle modes
SPM_SPEED = 0       # speed loop on the ramped speed command (CSS applied in the loop)
SPM_POSITION = 1    # position loop (orientation, rigid tapping) with position_loop gains
SPM_CAXIS = 2       # C-axis contouring: position loop on the C axis command with the C axis gains

# ------------------------------------------------------------------------------- geometry piece kinds
GEO_LINE = 0        # q = v0 + v1 * u                       (u = s / length)
GEO_ARC = 1         # q = v0 + r(u) (cos(phi u) v1 + sin(phi u) v2) + v3 * u, r(u) = r0 + (r1 - r0) u


class Chain(NamedTuple):
    """Kinematic chains of every path: base -> workpiece (chain 0) and base -> spindle nose (chain 1).

    Link i of a chain: ``T_i = T_{i-1} J_i(q) E_i(q) X_i`` with J the joint motion (translation by q
    along ``direction``, or rotation by q about the line (``point``, ``direction``), both in the parent
    frame at q = 0), E the error transform of the 6-vector ``err_const + sum_m err_poly[.., m] xi^m``
    (xi = (q - err_lo) / err_span, components dx dy dz rx ry rz) plus the thermal screw strain
    ``strain * (q - strain_ref)`` along the link direction, and X the fixed pose (identity for axis
    links).
    """

    n_links: np.ndarray         # [P, 2] int32
    kind: np.ndarray            # [P, 2, L] int32 LINK_*
    axis: np.ndarray            # [P, 2, L] int32 global axis index (-1 fixed)
    direction: np.ndarray       # [P, 2, L, 3]
    point: np.ndarray           # [P, 2, L, 3]
    fixed: np.ndarray           # [P, 2, L, 4, 4]
    err_poly: np.ndarray        # [P, 2, L, 6, 3]
    err_lo: np.ndarray          # [P, 2, L]
    err_span: np.ndarray        # [P, 2, L]
    err_const: np.ndarray       # [P, 2, L, 6]
    strain: np.ndarray          # [P, 2, L]
    strain_ref: np.ndarray      # [P, 2, L]
    work_spindle_link: np.ndarray   # [P] int32: link of chain 0 that turns the workpiece as a spindle (-1)
    pose_N_T0: np.ndarray       # [P, 4, 4] tool frame (tip, +z_T toward the spindle) in the nose frame
    pose_F_W: np.ndarray        # [P, 4, 4] fixture pose * part pose * part pose delta
    pose_W_S: np.ndarray        # [P, 4, 4] stock pose
    tool_spindle: np.ndarray    # [P] int32 spindle turning the active tool (-1 for a non-rotating tool)
    tool_phase: np.ndarray      # [P] holder slip phase added to the tool spindle angle (rad)


class Servo(NamedTuple):
    """Feed drive of every axis (global axis order). Rotary axes in rad, N m, kg m^2."""

    path: np.ndarray            # [NA] int32
    rotary: np.ndarray          # [NA] int8
    kv: np.ndarray              # [NA] 1/s position loop gain
    kff: np.ndarray             # [NA] velocity feed-forward fraction
    kp_v: np.ndarray            # [NA] N/(m/s): PI velocity loop proportional gain (2 zeta omega m)
    ki_v: np.ndarray            # [NA] N/m: integral gain (omega^2 m)
    mass: np.ndarray            # [NA] kg (kg m^2)
    force_max: np.ndarray       # [NA] N (N m): drive force limit
    friction: np.ndarray        # [NA] N: Coulomb friction (calibrated to the reversal spike)
    v_stribeck: np.ndarray      # [NA] m/s: regularisation speed of the Coulomb friction (tanh)
    gravity: np.ndarray         # [NA] N: static load along the axis (m g . direction)
    backlash: np.ndarray        # [NA] m: mechanical lost motion
    comp: np.ndarray            # [NA] m: backlash compensation amount of the control
    scale_fb: np.ndarray        # [NA] int8: 1 = position loop on the load (linear scale)
    in_pos: np.ndarray          # [NA] in-position band
    lo: np.ndarray              # [NA] soft travel limits (machine coordinates, -inf / +inf if none)
    hi: np.ndarray              # [NA]
    ferr_max: np.ndarray        # [NA] excess following error (servo alarm)
    spindle: np.ndarray         # [NA] int32: spindle whose rotor this C axis is (-1)
    kappa: np.ndarray           # [NA] q_axis = kappa * spindle angle
    tap_kv: np.ndarray          # [NA] loop gain while rigid tapping (the spindle position loop gain)
    n_sub: np.ndarray           # [1] int32 servo sub-steps per L1 tick
    dt: np.ndarray              # [1] L1 period (s)


class ServoState(NamedTuple):
    x: np.ndarray               # [NA] motor-side position (axis units)
    v: np.ndarray               # [NA] motor-side velocity
    integ: np.ndarray           # [NA] velocity-loop integral (force)
    load: np.ndarray            # [NA] load position after the backlash (what the chain sees)
    comp: np.ndarray            # [NA] backlash compensation now applied (+- comp / 2)
    dirn: np.ndarray            # [NA] last commanded direction (+1, -1, 0 unknown)
    cmd: np.ndarray             # [NA] last command
    force: np.ndarray           # [NA] motor force of the last sub-step
    acc: np.ndarray             # [NA] acceleration over the last tick
    vel_prev: np.ndarray        # [NA] load velocity at the end of the previous tick
    c_off: np.ndarray           # [NA] spindle C axes: q_C - kappa * theta (whole turns, set at C-axis engage)


class Spindle(NamedTuple):
    """Spindles of the machine (machine order)."""

    inertia: np.ndarray         # [S] kg m^2 rotor (+ chuck when published so)
    kp: np.ndarray              # [S] N m s: speed PI proportional gain
    ki: np.ndarray              # [S] N m: integral gain
    drag_max: np.ndarray        # [S] N m: drag torque at max speed (idle power / max speed)
    omega_max: np.ndarray       # [S] rad/s
    accel_rate: np.ndarray      # [S] rad/s^2 of the speed command ramp (inf: no ramp)
    decel_rate: np.ndarray      # [S] rad/s^2
    pos_kv: np.ndarray          # [S] 1/s spindle position loop (orientation, rigid tapping)
    pos_kff: np.ndarray         # [S]
    c_kv: np.ndarray            # [S] 1/s C-axis contouring gains (the C axis' servo data)
    c_kff: np.ndarray           # [S]
    c_axis: np.ndarray          # [S] int32 global axis index of the C axis on this rotor (-1)
    curve_w: np.ndarray         # [S, NG, NC] rad/s points of the torque limit per gear range
    curve_t: np.ndarray         # [S, NG, NC] N m
    n_curve: np.ndarray         # [S, NG] int32
    n_sub: np.ndarray           # [1] int32
    dt: np.ndarray              # [1]


class SpindleState(NamedTuple):
    angle: np.ndarray           # [S] unwrapped angle (rad)
    omega: np.ndarray           # [S] rad/s
    integ: np.ndarray           # [S] N m
    torque: np.ndarray          # [S] motor torque of the last sub-step
    cmd_omega: np.ndarray       # [S] ramped speed command
    gear: np.ndarray            # [S] int32 active gear range
    cmd_angle: np.ndarray       # [S] last position command (for the feed-forward)


class Setpoint(NamedTuple):
    """Commands of one macro step (T ticks; tick k ends at tick0 + (k + 1) ipo_ticks)."""

    n_ticks: np.ndarray         # [1] int32
    cmd: np.ndarray             # [T, NA]
    mode: np.ndarray            # [T, P] int8 SP_*
    block: np.ndarray           # [T, P] int32 timeline index
    rapid: np.ndarray           # [T, P] int8
    tap: np.ndarray             # [T, P] int8 rigid tapping (tap_kv on the path's linear axes)
    probe: np.ndarray           # [T, P] int8 probe move: the hot loop may set MotionOut.probe_hit
    setter: np.ndarray          # [T, P] int8 tool-setter move: the loop tests the setter contact
    sync_start: np.ndarray      # [T, P] int8 first tick of a synchronous stretch: the loop latches its angle
    sp_mode: np.ndarray         # [T, S] int8 SPM_*
    sp_omega: np.ndarray        # [T, S] target speed (before the ramp)
    sp_angle: np.ndarray        # [T, S] position command (SPM_POSITION)
    sp_omega_ff: np.ndarray     # [T, S] speed feed-forward of the position command
    css_vc: np.ndarray          # [T, S] constant surface speed (m/s, 0 = off)
    css_clamp: np.ndarray       # [T, S] spindle speed clamp (rad/s)
    css_axis: np.ndarray        # [S] int32 radius axis of the CSS (-1)
    css_centre: np.ndarray      # [S] radius-axis value at the spindle centre line
    css_rmin: np.ndarray        # [1] smallest radius used in the CSS law (m)
    wait_kind: np.ndarray       # [P] int8 WAIT_* for SP_HOLD_WAIT ticks
    sync_spindle: np.ndarray    # [P] int32 spindle driving the synchronous stretch
    sync_theta: np.ndarray      # [P, T + 1] planned spindle angle travelled since the stretch start (abs)
    sync_s: np.ndarray          # [P, T + 1] planned path position at those angles
    sync_n: np.ndarray          # [P] int32 valid rows of the table
    sync_theta0: np.ndarray     # [P] spindle angle at the stretch start (unwrapped)
    sync_s_end: np.ndarray      # [P] stretch length
    thread_lead: np.ndarray     # [P] m per revolution
    thread_trig: np.ndarray     # [P] one-revolution trigger angle (rad, [0, 2 pi))
    thread_tc: np.ndarray       # [P] s, exponential acceleration filter of threading
    geo_n: np.ndarray           # [P] int32 pieces of the stretch
    geo_kind: np.ndarray        # [P, G] int32 GEO_*
    geo_s0: np.ndarray          # [P, G] stretch position where the piece starts
    geo_len: np.ndarray         # [P, G]
    geo_vec: np.ndarray         # [P, G, 4, NA] v0, v1, v2, v3 (see GEO_*)
    geo_par: np.ndarray         # [P, G, 3] r0, r1, sweep
    setter_point: np.ndarray    # [P, 3] tool-setter contact point (M)
    setter_normal: np.ndarray   # [P, 3] outward normal of the contact face (M)
    setter_radius: np.ndarray   # [P] tool radius used for the contact test (0: tip point)
    setter_extra: np.ndarray    # [P] physical tool length beyond the nominal tip (growth, pull-out) for the setter


class LoopState(NamedTuple):
    stage: np.ndarray           # [P] int32 threading: 0 waiting for the trigger, 1 running
    theta_trig: np.ndarray      # [P] spindle angle at the trigger (unwrapped)
    s_filt: np.ndarray          # [P] filtered thread position
    s_act: np.ndarray           # [P] synchronous path position reached
    theta_prev: np.ndarray      # [P] spindle angle of the previous tick (trigger crossing)
    theta0: np.ndarray          # [P] spindle angle latched at the start of the synchronous stretch


class MotionOut(NamedTuple):
    actual: np.ndarray          # [T, NA] load positions
    cmd: np.ndarray             # [T, NA] commands used (synchronous ones computed in the loop)
    ferr: np.ndarray            # [T, NA] following error cmd - actual
    force: np.ndarray           # [T, NA] demanded motor force (m a + friction + gravity + cutting load)
    inertial: np.ndarray        # [T, NA] m a
    overload: np.ndarray        # [T, NA] int8 demanded force above the drive limit
    tip_M: np.ndarray           # [T, P, 3]
    tip_S: np.ndarray           # [T, P, 3]
    tip_W0: np.ndarray          # [T, P, 3]
    axis_M: np.ndarray          # [T, P, 3] tool axis +z_T in M
    sp_angle: np.ndarray        # [T, S] unwrapped spindle angle
    sp_omega: np.ndarray        # [T, S]
    sp_torque: np.ndarray       # [T, S] motor torque
    sync: np.ndarray            # [T, P] int32 SYNC_*
    alarm: np.ndarray           # [T, P] int32 ALARM_*
    alarm_axis: np.ndarray      # [T, P] int32
    in_cut: np.ndarray          # [T, P] int8 set by the hot loop (tool in contact)
    probe_hit: np.ndarray       # [T, P] int8 set by the hot loop (probe stylus contact)
    sync_s: np.ndarray          # [T, P] synchronous path position
    poses: np.ndarray           # [P, N_FK, 4, 4] FK poses at the last tick processed
    k_done: np.ndarray          # [1] int32 last tick processed (-1 none)


# ------------------------------------------------------------------------------- factories

def _f(*shape) -> np.ndarray:
    return np.zeros(shape, dtype=np.float64)


def _i(*shape, dtype=np.int32) -> np.ndarray:
    return np.zeros(shape, dtype=dtype)


def eye_stack(*lead) -> np.ndarray:
    out = np.zeros(tuple(lead) + (4, 4))
    out[..., 0, 0] = out[..., 1, 1] = out[..., 2, 2] = out[..., 3, 3] = 1.0
    return out


def empty_chain(P: int = 1, L: int = 1) -> Chain:
    fixed = eye_stack(P, 2, L)
    return Chain(_i(P, 2), _i(P, 2, L), np.full((P, 2, L), -1, np.int32), _f(P, 2, L, 3), _f(P, 2, L, 3), fixed,
                 _f(P, 2, L, 6, 3), _f(P, 2, L), np.ones((P, 2, L)), _f(P, 2, L, 6), _f(P, 2, L), _f(P, 2, L),
                 np.full(P, -1, np.int32), eye_stack(P), eye_stack(P), eye_stack(P), np.full(P, -1, np.int32), _f(P))


def empty_servo(NA: int = 1) -> Servo:
    return Servo(_i(NA), _i(NA, dtype=np.int8), np.full(NA, 30.0), _f(NA), np.full(NA, 1.0), np.full(NA, 1.0),
                 np.full(NA, 1.0), np.full(NA, 1e9), _f(NA), np.full(NA, 1e-4), _f(NA), _f(NA), _f(NA),
                 _i(NA, dtype=np.int8), np.full(NA, 1e-5), np.full(NA, -np.inf), np.full(NA, np.inf),
                 np.full(NA, np.inf), np.full(NA, -1, np.int32), np.ones(NA), np.full(NA, 30.0),
                 np.ones(1, np.int32), np.full(1, 1e-3))


def empty_servo_state(NA: int = 1) -> ServoState:
    return ServoState(*(_f(NA) for _ in ServoState._fields))


def empty_spindle(S: int = 1, NG: int = 1, NC: int = 2) -> Spindle:
    w = np.zeros((S, NG, NC))
    w[..., 1] = 1.0
    return Spindle(np.ones(S), np.ones(S), np.ones(S), _f(S), np.ones(S), np.full(S, np.inf), np.full(S, np.inf),
                   np.full(S, 30.0), _f(S), np.full(S, 30.0), _f(S), np.full(S, -1, np.int32), w, np.ones((S, NG, NC)),
                   np.full((S, NG), 2, np.int32), np.ones(1, np.int32), np.full(1, 1e-3))


def empty_spindle_state(S: int = 1) -> SpindleState:
    return SpindleState(_f(S), _f(S), _f(S), _f(S), _f(S), _i(S), _f(S))


def empty_setpoint(T: int = 1, NA: int = 1, P: int = 1, S: int = 1, G: int = 1) -> Setpoint:
    return Setpoint(np.array([T], np.int32), _f(T, NA), np.full((T, P), SP_END, np.int8), _i(T, P),
                    _i(T, P, dtype=np.int8), _i(T, P, dtype=np.int8), _i(T, P, dtype=np.int8), _i(T, P, dtype=np.int8),
                    _i(T, P, dtype=np.int8), _i(T, S, dtype=np.int8), _f(T, S), _f(T, S), _f(T, S), _f(T, S), np.full((T, S), np.inf),
                    np.full(S, -1, np.int32), _f(S), np.full(1, 1e-3), _i(P, dtype=np.int8), np.full(P, -1, np.int32),
                    _f(P, T + 1), _f(P, T + 1), _i(P), _f(P), _f(P), _f(P), _f(P), _f(P), _i(P), _i(P, G), _f(P, G),
                    np.ones((P, G)), _f(P, G, 4, NA), _f(P, G, 3), _f(P, 3), _f(P, 3), _f(P), _f(P))


def empty_loop_state(P: int = 1) -> LoopState:
    return LoopState(_i(P), _f(P), _f(P), _f(P), _f(P), _f(P))


def empty_out(T: int = 1, NA: int = 1, P: int = 1, S: int = 1) -> MotionOut:
    return MotionOut(_f(T, NA), _f(T, NA), _f(T, NA), _f(T, NA), _f(T, NA), _i(T, NA, dtype=np.int8),
                     _f(T, P, 3), _f(T, P, 3), _f(T, P, 3), _f(T, P, 3), _f(T, S), _f(T, S), _f(T, S),
                     _i(T, P), _i(T, P), np.full((T, P), -1, np.int32), _i(T, P, dtype=np.int8),
                     _i(T, P, dtype=np.int8), _f(T, P), eye_stack(P, N_FK), np.full(1, -1, np.int32))


class MotionInputs(NamedTuple):
    """What MotionDomain.kernel_inputs hands to the compiled loop for the rest of the macro step."""

    chain: Chain
    servo: Servo
    servo_state: ServoState
    spindle: Spindle
    spindle_state: SpindleState
    setpoint: Setpoint
    loop: LoopState
    out: MotionOut
    k_from: np.ndarray          # [1] int32 first tick to run
    dist_force: np.ndarray      # [NA] cutting load per axis applied on the first tick (previous tick's value)
    dist_torque: np.ndarray     # [S] cutting torque per spindle on the first tick
    result: np.ndarray          # [3] int64: sync code, tick index, path of the first sync (written by the loop)
