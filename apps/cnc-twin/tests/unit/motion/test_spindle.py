"""Spindles: ramps, torque limit, speed arrival, CSS, rigid tapping synchronisation."""

import copy
import math

import numpy as np
import pytest

from cnctwin.motion.machine import MachineModel
from cnctwin.motion.spindle import SpindleCmd, css_omega, speed_change_s
from mfix import LATHE, MILL, Script, run, stack


def test_ramp_time_from_accel_time_and_arrival_before_cutting():
    s = Script()
    s.spindle(9000)
    s.line(600, X=-1, Y=0, Z=0)            # cutting feed: waits for the speed arrival
    s.stop()
    res, dom, state, ctrl = run(MILL, s)
    om = np.abs(stack(res.frames, "spindle_omega_rad_s"))
    target = 9000 * 2 * math.pi / 60
    t = (np.arange(len(om)) + 1) * 1e-3
    # 0 -> 9000 rpm at 12000 rpm per 1.0 s: 0.75 s; the ramp starts with the block (the speed loop tracks it
    # within a few ms)
    t95 = t[np.nonzero(om >= 0.95 * target)[0][0]]
    assert t95 == pytest.approx(0.75 * 0.95, abs=0.005)
    # the spindle block takes its M-function time, the cutting block waits the rest of the run-up
    m_s = dom.model.p["m_function_s"]
    rec = dom.timeline.records[1]
    assert rec.ticks.get("spindle_ramp", 0) * 1e-6 == pytest.approx(0.95 * 0.75 - m_s, abs=0.002)
    # the cut starts only once the spindle is within the arrival band
    blk = stack(res.frames, "block_index")
    x = stack(res.frames, "axis_cmd")[:, 0]
    first_move = np.nonzero(np.abs(x) > 1e-12)[0][0]
    assert om[first_move] >= (1 - dom.model.p["spindle_arrival_frac"]) * target * 0.999


def test_torque_limited_acceleration_when_no_ramp_is_given():
    m = copy.deepcopy(LATHE)
    del m["spindles"][0]["accel_time_s"]
    del m["spindles"][0]["decel_time_s"]
    model = MachineModel(m)
    sp = model.spindles[0]
    # constant torque 180 N m to 800 rpm, constant power above; J = 0.2 kg m2 plus drag
    t = speed_change_s(sp, 0.0, 400 * 2 * math.pi / 60, 0)
    drag = sp.drag_max
    assert t == pytest.approx(0.2 * (400 * 2 * math.pi / 60) / 180.0, rel=0.02 + drag / 180.0)
    s = Script()
    s.spindle(400)
    s.dwell(s=0.3)
    s.stop()
    res, dom, state, ctrl = run(m, s)
    om = np.abs(stack(res.frames, "spindle_omega_rad_s"))
    tq = np.abs(stack(res.frames, "spindle_torque_Nm"))
    assert tq.max() <= 180.0 * (1 + 1e-9)
    t_reach = (np.nonzero(om >= 0.99 * 400 * 2 * math.pi / 60)[0][0] + 1) * 1e-3
    assert t_reach == pytest.approx(t, abs=0.004)


def test_css_law_with_clamp():
    model = MachineModel(copy.deepcopy(LATHE))
    sp = model.spindles[0]
    cmd = SpindleCmd(omega=-1.0, css_vc=200.0 / 60.0, clamp=2500 * 2 * math.pi / 60)
    assert css_omega(cmd, 0.05, sp, 1e-4) == pytest.approx(-(200.0 / 60.0) / 0.05)
    assert css_omega(cmd, 0.005, sp, 1e-4) == pytest.approx(-2500 * 2 * math.pi / 60)
    # in a facing cut the spindle speeds up as X goes toward the centre, at constant surface speed
    s = Script()
    s.spindle(500, css_m_min=150.0, clamp_rpm=3000)
    s.rapid(X=-100, Z=-200)
    s.dwell(s=1.0)
    s.line(0, per_rev_mm=0.2, X=-130, Z=-200)
    s.stop()
    res, dom, state, ctrl = run(LATHE, s, max_s=60)
    om = np.abs(stack(res.frames, "spindle_omega_rad_s"))
    x = stack(res.frames, "axis_actual")[:, 0]
    blk = stack(res.frames, "block_index")
    cut = np.nonzero(blk == 3)[0]
    info = dom.env.css_axis_info(0, 0)
    r = np.abs(x[cut] - info[1])
    expect = np.minimum(150.0 / 60.0 / r, 3000 * 2 * math.pi / 60)
    mid = cut[len(cut) // 4: len(cut) // 2]
    k = np.searchsorted(cut, mid)
    assert np.allclose(om[mid], expect[k], rtol=0.03)


def test_rigid_tapping_spindle_z_synchronisation_error():
    m = copy.deepcopy(MILL)
    m["spindles"][0]["position_loop"] = {"kv_1_s": 30, "feedforward": 0.5, "velocity_bw_Hz": 60}
    for ax in m["axes"].values():
        ax["servo"] = {"kv_1_s": 40, "feedforward": 0.9, "velocity_bw_Hz": 80}
    s = Script()
    s.rapid(X=-100, Y=-100, Z=-100)
    s.mode(rigid_tap=True)
    s.spindle(600)
    s.tap(1.5, X=-100, Y=-100, Z=-130)       # M10 x 1.5, 30 mm deep
    s.tap(1.5, X=-100, Y=-100, Z=-100)       # back out (reversed rotation)
    s.mode(rigid_tap=False)
    s.stop()
    res, dom, state, ctrl = run(m, s)
    blk = stack(res.frames, "block_index")
    z = stack(res.frames, "axis_actual")[:, 2]
    zc = stack(res.frames, "axis_cmd")[:, 2]
    ang = stack(res.frames, "spindle_angle_rad")
    rev = stack(res.frames, "spindle_rev")
    theta = 2 * math.pi * rev + ang
    tap_in = [r.index for r in dom.timeline.records if r.kind == "linear"][0]
    sel = np.nonzero(blk == tap_in)[0]
    z0 = zc[sel[0] - 1]
    th0 = theta[sel[0] - 1]
    pitch = 1.5e-3
    # tapping: 1.5 mm per spindle revolution, the spindle turning M03 (negative) going in
    z_from_theta = z0 + (theta[sel] - th0) / (2 * math.pi) * pitch
    err = z[sel] - z_from_theta
    v = 600 / 60 * pitch                                     # tapping speed at cruise
    # steady state: e = v [(1 - kff_sp) / Kv_sp - (1 - kff_z) / Kv_tap] with the tapping axis at the spindle's
    # loop gain (30 1/s): v (0.5 - 0.1) / 30
    cruise = sel[len(sel) // 3: 2 * len(sel) // 3]
    e_cruise = z[cruise] - (z0 + (theta[cruise] - th0) / (2 * math.pi) * pitch)
    assert np.median(np.abs(e_cruise)) == pytest.approx(v * (0.5 - 0.1) / 30.0, rel=0.05)
    # equal feed-forward on both: the steady error vanishes
    m2 = copy.deepcopy(m)
    m2["spindles"][0]["position_loop"]["feedforward"] = 0.9
    res2, dom2, _, _ = run(m2, s)
    z2 = stack(res2.frames, "axis_actual")[:, 2]
    th2 = 2 * math.pi * stack(res2.frames, "spindle_rev") + stack(res2.frames, "spindle_angle_rad")
    blk2 = stack(res2.frames, "block_index")
    sel2 = np.nonzero(blk2 == tap_in)[0]
    c2 = sel2[len(sel2) // 3: 2 * len(sel2) // 3]
    e2 = z2[c2] - (zc[sel[0] - 1] + (th2[c2] - th2[sel2[0] - 1]) / (2 * math.pi) * pitch)
    assert np.median(np.abs(e2)) < 0.1 * np.median(np.abs(e_cruise))


def test_orientation_turns_the_spindle_to_the_programmed_angle():
    s = Script()
    s.spindle(3000)
    s.dwell(s=1.0)
    s.orient(angle_deg=135.0)       # M19 R135: stop, then the position loop turns it to 135 degrees
    s.dwell(s=0.3)
    s.stop()
    res, dom, state, ctrl = run(MILL, s)
    spd = dom.model.spindles[0]
    rec = [r for r in dom.timeline.records if r.kind == "spindle"][1]
    assert (rec.tick1 - rec.tick0) * 1e-6 == pytest.approx(3000 / 12000 * 1.0 + spd.orient_s, abs=2e-3)
    angle = stack(res.frames, "spindle_angle_rad")
    omega = stack(res.frames, "spindle_omega_rad_s")
    blk = stack(res.frames, "block_index")
    held = blk == [r for r in dom.timeline.records if r.kind == "dwell"][1].index
    # held at the angle (position control at rest), and at rest
    err = np.angle(np.exp(1j * (angle[held] - math.radians(135.0))))
    assert np.max(np.abs(err[-100:])) < 1e-4
    assert np.max(np.abs(omega[held][-100:])) < 1e-3


def test_c_axis_moves_stop_the_spindle_and_position_the_work():
    s = Script()
    s.rapid(X=-50, Z=-100, C_deg=0)     # the first target of C is its modal value (not a move)
    s.spindle(1000)
    s.dwell(s=1.0)
    s.rapid(C_deg=90)               # C-axis mode: stop, reference, then position C
    s.line(3600, C_deg=180)         # 60 deg/s (1 deg = 1 mm of path)
    s.dwell(s=0.2)
    s.spindle(500)                  # back to speed control
    s.dwell(s=1.0)
    s.stop()
    res, dom, state, ctrl = run(LATHE, s)
    assert ctrl.done()
    model = dom.model
    spd = model.spindles[0]
    c = model.axis_index["C"]
    act = stack(res.frames, "axis_actual")[:, c]
    cmd = stack(res.frames, "axis_cmd")[:, c]
    omega = stack(res.frames, "spindle_omega_rad_s")
    angle = 2 * np.pi * stack(res.frames, "spindle_rev") + stack(res.frames, "spindle_angle_rad")
    blk = stack(res.frames, "block_index")
    recs = dom.timeline.records
    rap = [r for r in recs if r.kind == "rapid"][1]
    lin = [r for r in recs if r.kind == "linear"][0]
    dw = [r for r in recs if r.kind == "dwell"]
    # the C rapid first waits for the stop from 1000 rpm and the C-axis reference
    stop_s = 1000 / 4000 * spd.decel_time
    assert rap.ticks.get("spindle_ramp", 0) * 1e-6 == pytest.approx(stop_s + spd.c_engage_s, abs=2e-3)
    # C positions: 90 then 180 degrees (the C coordinate continues the program's)
    end_rap = blk == dw[1].index
    assert act[end_rap][0] == pytest.approx(math.radians(180.0), abs=1e-4)
    at90 = np.nonzero(blk == lin.index)[0][0]
    assert act[at90 - 1] == pytest.approx(math.radians(90.0), abs=2e-3)
    # the work turns with C: q_C = kappa * theta + whole turns
    kap = model.axes[c].kappa
    wrap = np.angle(np.exp(1j * (kap * angle[end_rap] - act[end_rap])))
    assert np.max(np.abs(wrap)) < 1e-9
    # the C feed: 60 deg/s commanded, the spindle follows within its C-axis loop
    moving = blk == lin.index
    rate = np.diff(cmd)[moving[1:]] / 1e-3
    assert np.max(np.abs(rate)) == pytest.approx(math.radians(60.0), rel=1e-3)
    assert np.max(np.abs(cmd - act)[moving]) < math.radians(60.0) / 10.0
    # speed control again afterwards
    assert omega[-1] == pytest.approx(-500 * 2 * math.pi / 60, rel=1e-3)


@pytest.mark.parametrize("capacity", [1000, 3])        # everything queued / blocks arriving while it runs
def test_c_axis_engage_inside_a_blended_chain_keeps_the_path_continuous(capacity):
    s = Script()
    s.rapid(X=-50, Z=-100, C_deg=0)
    s.spindle(1000)
    s.dwell(s=1.0)
    s.line(3600, tol_mm=0.01, C_deg=90)                # engages C; blended with the next blocks
    targets = [(-49.0, 100.0), (-48.0, 112.0), (-46.5, 120.0), (-45.0, 135.0)]
    for x, c in targets:
        s.line(3600, tol_mm=0.01, X=x, C_deg=c)
    s.rapid(X=-40)
    s.stop()
    res, dom, state, ctrl = run(LATHE, s, capacity=capacity)
    assert ctrl.done()
    model = dom.model
    c = model.axis_index["C"]
    x = model.axis_index["X"]
    cmd = stack(res.frames, "axis_cmd")
    act = stack(res.frames, "axis_actual")
    # no jump in the commanded path: every axis moves at most its rapid rate per tick (C modulo the whole
    # turns the engage relabels; before it the C command is the turning spindle's angle)
    assert np.max(np.abs(np.diff(cmd[:, x]))) <= model.axes[x].rapid * 1e-3 * (1 + 1e-9)
    dc = np.angle(np.exp(1j * np.diff(cmd[:, c])))
    w_max = np.max(np.abs(stack(res.frames, "spindle_omega_rad_s")))
    assert np.max(np.abs(dc)) <= max(model.axes[c].rapid, w_max) * 1e-3 * (1 + 1e-6)
    # the chain ends where the program says, the corners within the tolerance
    assert act[-1, c] == pytest.approx(math.radians(135.0), abs=2e-4)
    assert act[-1, x] == pytest.approx(-0.040, abs=1e-5)
    blk = stack(res.frames, "block_index")
    lines = [r.index for r in dom.timeline.records if r.kind == "linear"]
    assert len(lines) == 5
    sel = np.isin(blk, lines)
    pts = cmd[sel][:, [x, c]]
    # distance (scaled: 1 deg = 1 mm) of the commanded points from the programmed polyline
    poly = np.array([(-50.0, 90.0)] + targets)
    pp = np.column_stack([pts[:, 0] * 1e3, np.degrees(pts[:, 1])])
    d = np.full(len(pp), np.inf)
    for (a0, a1) in zip(poly[:-1], poly[1:]):
        seg = a1 - a0
        t = np.clip(((pp - a0) @ seg) / (seg @ seg), 0.0, 1.0)
        d = np.minimum(d, np.linalg.norm(pp - (a0 + np.outer(t, seg)), axis=1))
    first = np.nonzero(sel)[0][0]
    on_path = np.degrees(cmd[first:, c]) >= 90.0 - 1e-6       # after the C move reached the chain
    assert np.max(d[on_path[np.nonzero(sel)[0] - first]]) <= 0.01 * (1 + 1e-6)
