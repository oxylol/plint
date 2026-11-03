"""Feed drives: circle radius reduction vs the closed form, following error, backlash, reversal spike."""

import copy
import math

import numpy as np
import pytest

from cnctwin.motion.abi import SP_TIME, empty_loop_state, empty_out, empty_setpoint
from cnctwin.motion.kernels import run_motion
from cnctwin.motion.machine import MachineModel
from cnctwin.motion.servo import build_servo, build_spindles, circle_radius_ratio, initial_servo_state, \
    initial_spindle_state
from mfix import MILL


def _machine(kv=40.0, ff=0.9, bw=80.0, backlash=0.0, spike=0.0):
    m = copy.deepcopy(MILL)
    for ax in m["axes"].values():
        ax["servo"] = {"kv_1_s": kv, "feedforward": ff, "velocity_bw_Hz": bw}
        ax["backlash_um"] = backlash
        ax["reversal_spike_um"] = spike
    return MachineModel(m)


def _drive(model, cmd: np.ndarray, friction: bool = True):
    """Run the drives on a time-based command [T, NA]; returns the actual positions."""
    T, NA = cmd.shape
    servo = build_servo(model, 1e-3)
    if not friction:
        servo.friction[:] = 0.0
    sp = build_spindles(model, 1e-3)
    chain = model.build_chain([np.zeros(3)], [np.eye(4)], [np.eye(4)], [-1])
    ss = initial_servo_state(servo, cmd[0])
    sps = initial_spindle_state(model)
    setp = empty_setpoint(T, NA, 1, max(1, len(model.spindles)), 1)
    setp.mode[:, 0] = SP_TIME
    setp.cmd[:] = cmd
    out = empty_out(T, NA, 1, max(1, len(model.spindles)))
    result = np.zeros(3, np.int64)
    run_motion(chain, servo, ss, sp, sps, setp, empty_loop_state(1), out, 0, T, np.zeros(NA), np.zeros(sp.inertia.shape[0]),
               result)
    return out, servo


@pytest.mark.parametrize("kv,ff,bw,feed_mm_min,radius_mm", [(40.0, 0.9, 80.0, 6000.0, 10.0),
                                                            (30.0, 0.0, 60.0, 3000.0, 25.0),
                                                            (50.0, 0.95, 120.0, 10000.0, 5.0)])
def test_circle_radius_reduction_matches_closed_form(kv, ff, bw, feed_mm_min, radius_mm):
    model = _machine(kv, ff, bw)
    R = radius_mm * 1e-3
    v = feed_mm_min * 1e-3 / 60.0
    w = v / R
    T = int(4 * 2 * math.pi / w / 1e-3) + 400
    t = (np.arange(T) + 1) * 1e-3
    c = np.array([-0.4, -0.25])
    cmd = np.zeros((T, 3))
    cmd[:, 0] = c[0] + R * np.cos(w * t)
    cmd[:, 1] = c[1] + R * np.sin(w * t)
    cmd[:, 2] = -0.1
    # start on the circle, at rest: let the transient die before measuring the last revolution
    out, _ = _drive(model, cmd, friction=False)
    last = slice(T - int(2 * math.pi / w / 1e-3), T)
    r_act = np.hypot(out.actual[last, 0] - c[0], out.actual[last, 1] - c[1])
    ratio = circle_radius_ratio(kv, ff, bw, 0.7, w)
    dR_model = R * (1.0 - ratio)
    dR_sim = R - float(np.mean(r_act))
    # the chord velocity feed-forward and the linear fine interpolation differ from the continuous form by
    # O((w dt)^2) of R; the radius reduction itself is at least 1e-3 of R here
    assert dR_sim == pytest.approx(dR_model, rel=0.03, abs=2e-8)
    if ff == 0.0:
        # the classic estimate of a P position loop: dR = v^2 / (2 R Kv^2) (fast velocity loop)
        assert dR_sim == pytest.approx(v * v / (2 * R * kv * kv), rel=0.15)


def test_steady_following_error_is_v_one_minus_ff_over_kv():
    model = _machine(40.0, 0.9, 80.0)
    T = 3000
    v = 0.1
    cmd = np.zeros((T, 3))
    cmd[:, 0] = -0.5 + v * np.clip((np.arange(T) + 1) * 1e-3 - 0.1, 0.0, None)
    out, _ = _drive(model, cmd, friction=False)
    e = out.ferr[-200:, 0]
    assert np.allclose(e, v * (1 - 0.9) / 40.0, rtol=1e-6)


def test_backlash_lost_motion_is_the_compensation_residual():
    b = 10.0
    model = _machine(backlash=b)
    T = 6000
    t = (np.arange(T) + 1) * 1e-3
    x = np.where(t < 2.0, -0.3 + 0.005 * np.clip(t - 0.5, 0, 1), -0.3 + 0.005 - 0.005 * np.clip(t - 3.5, 0, 1))
    cmd = np.zeros((T, 3))
    cmd[:, 0] = x
    out, servo = _drive(model, cmd, friction=False)
    res = model.p["backlash_comp_residual_frac"]
    # at rest after a positive move the load lags the command by half the residual; after the negative
    # move it leads by the same: the lost motion is b x residual fraction
    after_pos = out.actual[1999, 0] - cmd[1999, 0]
    after_neg = out.actual[-1, 0] - cmd[-1, 0]
    assert after_pos == pytest.approx(-0.5 * b * 1e-6 * res, abs=2e-9)
    assert after_neg == pytest.approx(0.5 * b * 1e-6 * res, abs=2e-9)
    assert after_neg - after_pos == pytest.approx(b * 1e-6 * res, abs=4e-9)


def test_reversal_spike_from_the_calibrated_friction():
    spike_um = 3.0
    model = _machine(spike=spike_um)
    T = 4000
    t = (np.arange(T) + 1) * 1e-3
    # 1 m/min triangle: the command reverses at t = 2 s
    v = 1.0 / 60.0
    x = -0.3 + v * np.where(t < 2.0, t, 4.0 - t)
    cmd = np.zeros((T, 3))
    cmd[:, 0] = x
    with_f, servo = _drive(model, cmd, friction=True)
    without, _ = _drive(model, cmd, friction=False)
    assert servo.friction[0] > 0.0
    glitch = np.max(np.abs(with_f.actual[1990:2300, 0] - without.actual[1990:2300, 0]))
    # the steady part of the friction difference (held by the velocity integrator) is zero; the transient at the
    # reversal reproduces the spike the friction was calibrated to (a step of 2 F_c at rest)
    assert glitch == pytest.approx(spike_um * 1e-6, rel=0.25)
    assert np.max(np.abs(with_f.actual[1000:1900, 0] - without.actual[1000:1900, 0])) < 0.05 * spike_um * 1e-6
