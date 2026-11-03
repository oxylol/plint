"""Compiled kernels against their plain-Python references, and one signature per kernel."""

import copy
import math

import numpy as np
import pytest

from cnctwin.motion import kernels as K
from cnctwin.motion import reference as R
from cnctwin.motion.abi import (SP_HOLD_WAIT, SP_PER_REV, SP_TIME, SPM_POSITION, SPM_SPEED, empty_loop_state, empty_out,
                                empty_setpoint)
from cnctwin.motion.machine import MachineModel
from cnctwin.motion.servo import build_servo, build_spindles, initial_servo_state, initial_spindle_state
from mfix import LATHE, MILL, five_axis

KERNELS = sorted(name for name, obj in vars(K).items()
                 if hasattr(obj, "signatures") and getattr(getattr(obj, "py_func", None), "__module__", "") == K.__name__)


def test_every_kernel_is_listed():
    assert {"motion_tick", "run_motion", "plan_ticks", "fk_poses", "ik_position", "_axis_rhs", "_spindle_rhs",
            "_table_s", "_stretch_command", "_nodes_ok"} <= set(KERNELS)
    assert len(KERNELS) == 30


@pytest.mark.parametrize("name", KERNELS)
def test_one_signature_per_kernel(name):
    assert len(getattr(K, name).signatures) == 1


def _model(machine):
    return MachineModel(copy.deepcopy(machine))


def _chain(model, tip=0.1, errors=True):
    P = len(model.paths)
    return model.build_chain([np.array([0.0, 0.0, -tip])] * P, [np.eye(4)] * P, [np.eye(4)] * P, [-1] * P,
                             nominal=not errors)


@pytest.mark.parametrize("kind", ["table_bc", "head_bc", "head_table_bc"])
def test_fk_matches_reference_with_errors(kind):
    m = five_axis(kind)
    m["error_map"] = {"squareness_urad": {"XY": 30, "YZ": -20}, "straightness_um_per_m": {"X": 8, "Z": 5},
                      "angular_urad": {"X": {"pitch": 15, "yaw": -10, "roll": 5}},
                      "pivot_errors": {"B": {"position_um": [10, -5, 20], "tilt_urad": [30, 0, -10]}}}
    model = _model(m)
    chain = _chain(model)
    chain.err_const[0, 1, 0, :] = [1e-6, -2e-6, 3e-6, 1e-5, -2e-5, 5e-6]
    chain.strain[0, 1, 0] = 1.2e-5
    chain.strain_ref[0, 1, 0] = -0.1
    rng = np.random.default_rng(3)
    out = np.zeros((7, 4, 4))
    work = np.zeros((6, 4, 4))
    for _ in range(10):
        q = rng.uniform(-0.3, 0.3, len(model.axes))
        K.fk_poses(chain, 0, q, out, work)
        ref = R.fk_poses(chain, 0, q)
        assert np.allclose(out, ref, atol=1e-13, rtol=0)


@pytest.mark.parametrize("kind", ["table_bc", "head_table_bc"])
def test_ik_position_matches_reference(kind):
    m = five_axis(kind)
    m["error_map"] = {"angular_urad": {"X": {"pitch": 40}}, "straightness_um_per_m": {"Y": 20}}
    model = _model(m)
    chain = _chain(model)
    rng = np.random.default_rng(9)
    lin = np.array([model.axis_index[n] for n in ("X", "Y", "Z")], np.int32)
    poses = np.zeros((7, 4, 4))
    work = np.zeros((6, 4, 4))
    for _ in range(8):
        q_true = rng.uniform(-0.3, 0.3, len(model.axes))
        target = R.tip_in_W(chain, 0, q_true)
        q_k = q_true.copy()
        q_k[lin] += rng.uniform(-0.01, 0.01, 3)
        q_r = q_k.copy()
        res_k = K.ik_position(chain, 0, q_k, target, lin, 6, poses, work)
        res_r = R.ik_position(chain, 0, q_r, target, lin, 6)
        assert res_k < 1e-11 and res_r < 1e-11
        assert np.allclose(q_k, q_r, atol=1e-10)
        assert np.allclose(q_k, q_true, atol=1e-9)


def test_geo_eval_matches_reference():
    vec = np.random.default_rng(1).normal(size=(4, 5))
    par = np.array([0.01, 0.012, 2.5])
    axes = np.arange(5, dtype=np.int32)
    s = np.linspace(0.0, 0.03, 17)
    for kind in (0, 1):
        out = np.zeros((17, 5))
        ref = np.zeros((17, 5))
        K.geo_eval_many(kind, 0.03, vec, par, s, axes, out)
        R.geo_eval_many(kind, 0.03, vec, par, s, axes, ref)
        assert np.array_equal(out, ref)
        one, one_ref = np.zeros(5), np.zeros(5)
        K.geo_eval(kind, 0.03, vec, par, 0.011, axes, one)
        R.geo_eval(kind, 0.03, vec, par, 0.011, axes, one_ref)
        assert np.array_equal(one, one_ref)


def test_small_kernels_match_reference():
    rng = np.random.default_rng(21)
    out3 = np.zeros((3, 3))
    for _ in range(20):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        ang = rng.uniform(-4, 4)
        K.axis_rotation(d, ang, out3)
        assert np.allclose(out3, R.axis_rotation(d, ang), atol=1e-15)
        rv = rng.normal(size=3) * 0.3
        K.rotvec_rotation(rv[0], rv[1], rv[2], out3)
        assert np.allclose(out3, R.rotvec_rotation(*rv), atol=1e-15)
    K.rotvec_rotation(0.0, 0.0, 0.0, out3)
    assert np.array_equal(out3, np.eye(3))
    a = R.pose(R.rotvec_rotation(0.1, -0.2, 0.3), [0.1, 0.2, -0.3])
    b = R.pose(R.rotvec_rotation(-0.4, 0.05, 0.2), [-0.02, 0.5, 0.1])
    out = np.zeros((4, 4))
    K.mul4(a, b, out)
    assert np.allclose(out, a @ b, atol=1e-15)
    K.inv4(a, out)
    assert np.allclose(out, R.inv(a), atol=1e-15)
    K.copy4(b, out)
    assert np.array_equal(out, b)
    K.eye4(out)
    assert np.array_equal(out, np.eye(4))
    K.spin_z(a, 0.7, out)
    assert np.allclose(out, R.spin_z(a, 0.7), atol=1e-15)


def test_chain_kernels_match_reference():
    m = five_axis("head_table_bc")
    m["error_map"] = {"squareness_urad": {"XY": 30}, "straightness_um_per_m": {"X": 8},
                      "pivot_errors": {"B": {"position_um": [10, -5, 20], "tilt_urad": [30, 0, -10]}}}
    model = _model(m)
    chain = _chain(model)
    chain.strain[0, 1, 0] = 1.2e-5
    q = np.random.default_rng(5).uniform(-0.3, 0.3, len(model.axes))
    out = np.zeros((4, 4))
    rot = np.zeros((4, 4))
    tmp = np.zeros((4, 4))
    work = np.zeros((6, 4, 4))
    for c in (0, 1):
        for i in range(int(chain.n_links[0, c])):
            K.link_motion(chain, 0, c, i, q, -1, out, rot, tmp)
            assert np.allclose(out, R.link_motion(chain, 0, c, i, q), atol=1e-14)
        for zl in (-1, 0):
            K.chain_pose(chain, 0, c, q, zl, out, work)
            assert np.allclose(out, R.chain_pose(chain, 0, c, q, zl), atol=1e-14)
    tip = np.zeros(3)
    poses = np.zeros((7, 4, 4))
    K.tip_in_W(chain, 0, q, poses, work, tip)
    assert np.allclose(tip, R.tip_in_W(chain, 0, q), atol=1e-14)


def test_rhs_table_stretch_and_node_kernels_match_reference():
    model, servo = _servo_fixture()
    o = np.zeros(5)
    for args in ((0, 0.01, 0.02, 3.0, 0.011, 0.0101, 0.015, 30.0, 40.0),
                 (1, -0.2, -0.5, -9.0, -0.25, -0.2, 0.0, 30.0, -900.0)):
        a, x, v, integ, xc, load, vff, kv, f_ext = args
        o[4] = f_ext                    # the external force is passed in out[4]
        K._axis_rhs(servo, a, x, v, integ, xc, load, vff, kv, o)
        ref = R._axis_rhs(servo, a, x, v, integ, xc, load, vff, kv, f_ext)
        assert np.allclose(o[:4], ref, atol=1e-12)
    lathe = _model(LATHE)
    sp = build_spindles(lathe, 1e-3)
    o = np.zeros(4)
    for mode in (SPM_SPEED, SPM_POSITION):
        K._spindle_rhs(sp, 0, 0, 0.3, -50.0, 2.0, -60.0, 0.31, 1.0, 30.0, mode, 4.0, o)
        ref = R._spindle_rhs(sp, 0, 0, 0.3, -50.0, 2.0, -60.0, 0.31, 1.0, 30.0, mode, 4.0)
        assert np.allclose(o, ref, atol=1e-12)
    setp = empty_setpoint(4, 3, 1, 1, 3)
    setp.sync_n[0] = 3
    theta = np.array([0.0, 0.5, 1.0, 0.0, 0.0])
    s_tab = np.array([0.0, 1e-4, 2.5e-4, 0.0, 0.0])
    for dth in (-0.1, 0.0, 0.2, 0.5, 0.75, 1.0, 1.6):
        assert K._table_s(setp, 0, dth, theta, s_tab) == pytest.approx(R._table_s(setp, 0, dth, theta, s_tab),
                                                                        abs=1e-18)
    setp.geo_n[0] = 2
    setp.geo_kind[0, :2] = [0, 1]
    setp.geo_len[0, :2] = [1e-3, 2e-3]
    setp.geo_s0[0, :2] = [0.0, 1e-3]
    rng = np.random.default_rng(2)
    setp.geo_vec[0, :2] = rng.normal(size=(2, 4, 3)) * 1e-2
    setp.geo_par[0, 1] = [0.01, 0.012, 0.2]
    axes = np.arange(3, dtype=np.int32)
    ok, ref = np.zeros(3), np.zeros(3)
    for s in (0.0, 4e-4, 1e-3, 2.2e-3, 3e-3):
        K._stretch_command(setp, 0, s, axes, ok)
        R._stretch_command(setp, 0, s, axes, ref)
        assert np.allclose(ok, ref, atol=1e-15)
    setp.mode[:, 0] = SP_TIME
    setp.mode[2, 0] = 4
    for k in range(4):
        assert K.replan_before(setp, k) == R.replan_before(setp, k)
    for v, acc, jj in ((0.1, 1.0, 50.0), (0.05, -2.0, -100.0), (0.0, 0.0, 200.0)):
        assert K.natural_velocity(v, acc, jj, 1e-3, 300.0) == pytest.approx(R.natural_velocity(v, acc, jj, 1e-3, 300.0),
                                                                          abs=1e-15)
    nx = np.array([2e-3, 5e-3])
    nu = np.array([0.02, 0.0])
    na = np.array([3.0, 3.0])
    nj = np.array([100.0, 100.0])
    for v1, a1, moved in ((0.1, 0.0, 0.0), (0.3, 1.0, 1e-3), (0.05, -1.0, 4e-3), (-1.0, 0.0, 0.0)):
        assert bool(K._nodes_ok(v1, a1, moved, 1e-9, nx, nu, na, nj, 2)) == \
            R._nodes_ok(v1, a1, moved, 1e-9, list(zip(nx, nu, na, nj)))


def test_motion_tick_c_axis_mode_with_a_turn_offset_matches_reference():
    from cnctwin.motion.abi import SPM_CAXIS
    model, chain, servo, sp, q0, P, NA, S = _tick_fixture(LATHE)
    c = model.axis_index["C"]
    T = 12
    states = []
    for _ in range(2):
        ss = initial_servo_state(servo, q0)
        sps = initial_spindle_state(model)
        sps.angle[0] = 7.3                       # the spindle has turned; C continues at q_C = theta + 4 pi
        sps.cmd_angle[0] = 7.3
        ss.c_off[c] = 4 * math.pi
        setp = empty_setpoint(T, NA, P, S, 2)
        setp.mode[:, 0] = SP_TIME
        for k in range(T):
            setp.cmd[k] = q0
            setp.cmd[k, c] = model.axes[c].kappa * 7.3 + 4 * math.pi + 2e-3 * k
            setp.sp_mode[k, 0] = SPM_CAXIS
        states.append((ss, sps, setp, empty_loop_state(P), empty_out(T, NA, P, S)))
    z = np.zeros(NA)
    for k in range(T):
        c1 = K.motion_tick(chain, servo, states[0][0], sp, states[0][1], states[0][2], states[0][3], states[0][4], k,
                           z, np.zeros(S))
        c2 = R.motion_tick(chain, servo, states[1][0], sp, states[1][1], states[1][2], states[1][3], states[1][4], k,
                           z, np.zeros(S))
        assert c1 == c2
    out = states[0][4]
    assert np.allclose(out.actual[:, c] - model.axes[c].kappa * out.sp_angle[:, 0], 4 * math.pi, atol=1e-12)
    assert np.max(np.abs(out.ferr[:, c])) < 0.05            # the spindle follows C (no jump of whole turns)
    for a, b in zip(states[0], states[1]):
        for name in a._fields:
            assert np.allclose(getattr(a, name), getattr(b, name), rtol=0, atol=1e-10, equal_nan=True), name


def _servo_fixture(backlash=3.0, spike=2.0):
    m = copy.deepcopy(MILL)
    for ax in m["axes"].values():
        ax["backlash_um"] = backlash
        ax["reversal_spike_um"] = spike
    model = _model(m)
    servo = build_servo(model, 1e-3)
    return model, servo


def test_servo_step_matches_reference():
    model, servo = _servo_fixture()
    q0 = np.array([ax.home for ax in model.axes]) - 0.1
    st_k = initial_servo_state(servo, q0)
    st_r = initial_servo_state(servo, q0)
    work = np.zeros(32)
    for k in range(300):
        cmd = q0[0] + 0.002 * math.sin(2 * math.pi * k / 120.0)
        K.servo_axis_step(servo, st_k, 0, cmd, float(servo.kv[0]), 50.0 * math.cos(k * 0.1), work)
        R.servo_axis_step(servo, st_r, 0, cmd, float(servo.kv[0]), 50.0 * math.cos(k * 0.1))
    for name in st_k._fields:
        assert np.allclose(getattr(st_k, name), getattr(st_r, name), rtol=0, atol=1e-12), name


def test_spindle_step_ramp_and_torque_limit_match_reference():
    model = _model(LATHE)
    sp = build_spindles(model, 1e-3)
    st_k = initial_spindle_state(model)
    st_r = initial_spindle_state(model)
    work = np.zeros(32)
    for k in range(400):
        target = -200.0 if k < 250 else 30.0
        mode = SPM_SPEED if k < 300 else SPM_POSITION
        st_k.cmd_omega[0] = K.ramp_toward(float(st_k.cmd_omega[0]), target, float(sp.accel_rate[0]),
                                          float(sp.decel_rate[0]), 1e-3)
        st_r.cmd_omega[0] = R.ramp_toward(float(st_r.cmd_omega[0]), target, float(sp.accel_rate[0]),
                                          float(sp.decel_rate[0]), 1e-3)
        angle = float(st_k.angle[0]) + 0.01
        K.spindle_step(sp, st_k, 0, mode, float(st_k.cmd_omega[0]), angle, 0.0, float(sp.pos_kv[0]), 3.0, work)
        R.spindle_step(sp, st_r, 0, mode, float(st_r.cmd_omega[0]), angle, 0.0, float(sp.pos_kv[0]), 3.0)
        assert K.torque_limit(sp, 0, 0, float(st_k.omega[0])) == R.torque_limit(sp, 0, 0, float(st_r.omega[0]))
    for name in st_k._fields:
        assert np.allclose(getattr(st_k, name), getattr(st_r, name), rtol=0, atol=1e-9), name


def test_generator_matches_reference():
    rng = np.random.default_rng(7)
    n = 12
    L = rng.uniform(1e-4, 5e-3, n)
    vmax = rng.uniform(0.02, 0.3, n)
    amax = rng.uniform(1.0, 6.0, n)
    jmax = rng.uniform(50.0, 300.0, n)
    ustart = np.minimum(vmax, np.roll(vmax, 1)) * rng.uniform(0.0, 1.0, n)
    ready = np.sort(rng.integers(0, 60000, n)).astype(np.int64)
    stop = np.zeros(n, np.int8)
    stop[7] = 1
    outs = []
    for fn in (K.plan_ticks, R.plan_ticks):
        gi = np.zeros(1, np.int64)
        gf = np.zeros(3)
        op = np.zeros(300, np.int32)
        os_ = np.zeros(300)
        ov = np.zeros(300)
        oa = np.zeros(300)
        ol = np.zeros(300, np.int8)
        oc = np.zeros(1, np.int32)
        status = fn(L, vmax, amax, jmax, ustart, ready, stop, n, 6, gi, gf, 1000, 1000, 1e-3, 300, op, os_, ov, oa,
                    ol, oc)
        outs.append((status, int(oc[0]), op.copy(), os_.copy(), ov.copy(), oa.copy(), ol.copy(), gi.copy(),
                     gf.copy()))
    a, b = outs
    assert a[0] == b[0] and a[1] == b[1]
    for x, y in zip(a[2:], b[2:]):
        assert np.allclose(x, y, rtol=0, atol=1e-12)


def test_brake_distance_and_track_jerk_match_reference():
    rng = np.random.default_rng(11)
    for _ in range(200):
        v, a, u = rng.uniform(0, 0.5), rng.uniform(-4, 4), rng.uniform(0, 0.4)
        A, J = rng.uniform(1, 6), rng.uniform(20, 400)
        assert K.brake_distance(v, a, u, A, J) == pytest.approx(R.brake_distance(v, a, u, A, J), abs=1e-15)
        assert K.track_jerk(v, a, u, A, J, 1e-3) == pytest.approx(R.track_jerk(v, a, u, A, J, 1e-3), abs=1e-12)


def _tick_fixture(machine):
    model = _model(machine)
    P = len(model.paths)
    NA = len(model.axes)
    S = max(1, len(model.spindles))
    chain = _chain(model)
    servo = build_servo(model, 1e-3)
    sp = build_spindles(model, 1e-3)
    q0 = np.array([ax.home for ax in model.axes]) - 0.05
    return model, chain, servo, sp, q0, P, NA, S


def test_motion_tick_matches_reference_time_and_per_rev():
    model, chain, servo, sp, q0, P, NA, S = _tick_fixture(LATHE)
    T = 10
    states = []
    for _ in range(2):
        ss = initial_servo_state(servo, q0)
        sps = initial_spindle_state(model)
        sps.omega[0] = -100.0
        sps.cmd_omega[0] = -100.0
        setp = empty_setpoint(T, NA, P, S, 4)
        setp.mode[:, 0] = SP_TIME
        setp.mode[5:, 0] = SP_PER_REV
        setp.sync_start[5, 0] = 1
        setp.sync_spindle[0] = 0
        setp.sync_theta[0, :3] = [0.0, 0.5, 1.0]
        setp.sync_s[0, :3] = [0.0, 1e-4, 2e-4]
        setp.sync_n[0] = 3
        setp.sync_s_end[0] = 1e-3
        setp.geo_n[0] = 1
        setp.geo_len[0, 0] = 1e-3
        setp.geo_vec[0, 0, 0] = q0
        setp.geo_vec[0, 0, 1, 1] = -1e-3
        for k in range(T):
            setp.cmd[k] = q0 + np.array([1e-4 * k, -2e-5 * k, 0.0])
            setp.sp_mode[k, 0] = SPM_SPEED
            setp.sp_omega[k, 0] = -100.0
        setp.css_vc[:, 0] = 3.0
        setp.css_axis[0] = 0
        setp.css_centre[0] = -0.2
        states.append((ss, sps, setp, empty_loop_state(P), empty_out(T, NA, P, S)))
    df = np.array([30.0, -20.0, 0.0])
    dtq = np.array([2.0])
    for k in range(T):
        c1 = K.motion_tick(chain, servo, states[0][0], sp, states[0][1], states[0][2], states[0][3], states[0][4], k,
                           df, dtq)
        c2 = R.motion_tick(chain, servo, states[1][0], sp, states[1][1], states[1][2], states[1][3], states[1][4], k,
                           df, dtq)
        assert c1 == c2
    for a, b in zip(states[0], states[1]):
        for name in a._fields:
            x, y = getattr(a, name), getattr(b, name)
            assert np.allclose(x, y, rtol=0, atol=1e-10, equal_nan=True), name


def test_run_motion_hold_wait_reports_in_position_like_reference():
    model, chain, servo, sp, q0, P, NA, S = _tick_fixture(MILL)
    T = 10
    res = []
    for fn in (K.run_motion, R.run_motion):
        ss = initial_servo_state(servo, q0)
        sps = initial_spindle_state(model)
        setp = empty_setpoint(T, NA, P, S, 2)
        setp.mode[:, 0] = SP_HOLD_WAIT
        setp.cmd[:] = q0 + 2e-6
        out = empty_out(T, NA, P, S)
        result = np.zeros(3, np.int64)
        code = fn(chain, servo, ss, sp, sps, setp, empty_loop_state(P), out, 0, T, np.zeros(NA), np.zeros(S), result)
        res.append((code, result.copy(), out.actual.copy()))
    assert res[0][0] == res[1][0] == 2
    assert np.array_equal(res[0][1], res[1][1])
    assert np.allclose(res[0][2], res[1][2], atol=1e-12)
