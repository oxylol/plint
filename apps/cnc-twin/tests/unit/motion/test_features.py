"""Probing, tool setter, ISO 230-1 error map effects, machine lock, spindle orientation and gear change."""

import copy
import math

import numpy as np
import pytest

from cnctwin.controller.ir import Command, Feed, FeedMode, Kind, Space
from mfix import MILL, Script, ref, run, stack


def _probe_program(z_end=-160):
    s = Script()
    s.rapid(X=-100, Y=-100, Z=-150)
    s.probe(300, X=-100, Y=-100, Z=z_end)
    s.stop()
    return s


def test_probe_skip_latches_where_the_stylus_touches_and_brakes():
    plane_M = 0.6 - 0.155            # the stylus (gauge line, no tool) touches at nose z = 600 - 155 mm

    def probe(p, k, inputs):
        return inputs.out.tip_M[k, p, 2] <= plane_M

    res, dom, state, ctrl = run(MILL, _probe_program(), probe=probe)
    (tick, fb), = ctrl.feedbacks[0]
    assert fb.probe.triggered
    pre = dom.model.p["probe_pretravel_m"]
    z_latched = fb.probe.machine_pos["Z"]
    # latched within one tick's travel past the touch point (plus the stylus pre-travel)
    assert z_latched == pytest.approx(-0.155 - pre, abs=300 / 60e3 * 1e-3 + 1e-9)
    # the axis then brakes to rest a little beyond and never reaches the programmed end
    z = stack(res.frames, "axis_actual")[:, 2]
    assert z.min() > -0.160 + 1e-4 and z.min() < -0.155
    assert state.motion.planners[0].item(1).triggered


def test_the_move_after_a_skip_starts_where_the_axes_stopped():
    """G31 then a rapid back up: the rapid is planned from the braked stop, not from the probe block's programmed
    end point (the command used to jump to that end point, 5 mm past the touch, and the servo followed it down
    before the rapid came back: in the full loop the stylus went 3 mm into the part after the trigger)."""
    s = Script()
    s.rapid(X=-100, Y=-100, Z=-150)
    s.probe(300, X=-100, Y=-100, Z=-160)
    s.rapid(X=-100, Y=-100, Z=-120)
    s.stop()
    plane_M = 0.6 - 0.155
    res, dom, state, ctrl = run(MILL, s, probe=lambda p, k, inputs: inputs.out.tip_M[k, p, 2] <= plane_M)
    assert ctrl.done()
    (tick, fb), = ctrl.feedbacks[0]
    assert fb.probe.triggered
    cmd = stack(res.frames, "axis_cmd")[:, 2]
    act = stack(res.frames, "axis_actual")[:, 2]
    # neither the command nor the axis goes further than the brake distance past the touch (-155 mm)
    assert cmd.min() > -0.1552 and act.min() > -0.1552
    assert act[-1] == pytest.approx(-0.120, abs=dom.model.in_position)


def test_probe_without_contact_reports_no_trigger_at_the_end():
    res, dom, state, ctrl = run(MILL, _probe_program(), probe=lambda p, k, inputs: False)
    (tick, fb), = ctrl.feedbacks[0]
    assert not fb.probe.triggered
    assert fb.probe.machine_pos["Z"] == pytest.approx(-0.160, abs=dom.model.in_position)


def test_tool_setter_measures_the_physical_tool_length():
    # setter disc at M z = 100 mm (machine file probes.tool_setter.position_mm); tool 100 mm long
    s = Script()
    s.tool_change(1, slot=0)
    s.rapid(X=-400, Y=-300, Z=-300)
    s._add(Command(Kind.TOOL_SETTER, ref(), Space.MACHINE, {"X": -0.4, "Y": -0.3, "Z": -0.45},
                   feed=Feed(FeedMode.PER_MIN, 200e-3 / 60.0), setter={"measure": "length", "tool": 1}))
    s.stop()
    res, dom, state, ctrl = run(MILL, s)
    (tick, fb), = ctrl.feedbacks[0]
    assert fb.probe.triggered and fb.setter.length_m == pytest.approx(0.1)
    # trigger where the tip (nose z - 100 mm) reaches the disc at z = 100 mm: Z = 100 + 100 - 600 = -400 mm
    assert fb.probe.machine_pos["Z"] == pytest.approx(-0.400, abs=200 / 60e3 * 1e-3 + 1e-9)


def test_squareness_and_straightness_move_the_tip_not_the_axes():
    e = 50e-6
    m = copy.deepcopy(MILL)
    m["axes"]["X"]["travel_mm"] = [-800, 0]
    m["error_map"] = {"squareness_urad": {"XY": 50}, "straightness_um_per_m": {"X": 10}}
    s = Script()
    s.rapid(X=-400, Y=-400, Z=-100)
    s.stop()
    res0, dom0, _, _ = run(MILL, s)
    res1, dom1, _, _ = run(m, s)
    t0 = stack(res0.frames, "tip_S_m")[-1]
    t1 = stack(res1.frames, "tip_S_m")[-1]
    d = t1 - t0
    # squareness XY = e: moving Y by 400 mm tilts toward X by e x 0.4 m (the angle between +X and +Y is 90 deg - e)
    # straightness of X (10 um/m over 800 mm = 8 um bow, 4 D xi (1 - xi) at xi = 0.5): 8 um in y and z
    assert abs(d[0]) == pytest.approx(e * 0.4, rel=1e-3)
    assert abs(d[1]) == pytest.approx(8e-6, rel=1e-3)
    assert abs(d[2]) == pytest.approx(8e-6, rel=1e-3)
    assert np.array_equal(stack(res0.frames, "axis_cmd"), stack(res1.frames, "axis_cmd"))


def test_machine_lock_holds_the_axes_but_times_the_program():
    from cnctwin.core.state import SimState
    from cnctwin.motion.driver import MotionRunner
    from mfix import ScriptedController, context, domain
    s = Script()
    s.rapid(X=-100, Y=-100, Z=-100)
    s.line(1200, X=-50, Y=-100, Z=-100)
    s.stop()
    ctrl = ScriptedController([s])
    ctrl.state.paths[0].operator.machine_lock = True
    dom = domain(MILL)
    res = MotionRunner(ctrl, dom, ctx=context(), state=SimState(), keep_frames=True).run(60)
    act = stack(res.frames, "axis_actual")
    assert np.max(np.abs(act)) < 1e-9
    lin = [r for r in dom.timeline.records if r.kind == "linear"][0]
    assert (lin.tick1 - lin.tick0) * 1e-6 == pytest.approx(50.0 / 1200.0 * 60.0, rel=0.02)


def test_orientation_and_gear_change_times():
    m = copy.deepcopy(MILL)
    m["spindles"][0]["gear_ranges"] = [
        {"rpm": [0, 3000], "power_kW": [[0, 0], [3000, 15]], "torque_Nm": [[0, 300], [3000, 48]]},
        {"rpm": [3000, 12000], "power_kW": [[0, 0], [12000, 15]], "torque_Nm": [[0, 100], [12000, 12]]}]
    m["spindles"][0]["gear_change_s"] = 2.5
    s = Script()
    s.spindle(2000)
    s.dwell(s=1.0)
    s.spindle(8000)                 # gear change: stop from 2000 rpm, shift, then run up in the background
    s.dwell(s=2.0)
    s.orient()                      # stop from 8000 rpm, then orient
    s.stop()
    res, dom, state, ctrl = run(m, s)
    sp = [r for r in dom.timeline.records if r.kind == "spindle"]
    model = dom.model
    spd = model.spindles[0]
    assert (sp[1].tick1 - sp[1].tick0) * 1e-6 == pytest.approx(2000 / 12000 * 1.0 + 2.5, abs=2e-3)
    assert (sp[2].tick1 - sp[2].tick0) * 1e-6 == pytest.approx(8000 / 12000 * 1.0 + spd.orient_s, abs=2e-3)


def test_long_per_revolution_contour_runs_as_one_stretch():
    from mfix import LATHE
    s = Script()
    s.spindle(1200)
    s.rapid(X=-100, Z=-200)
    s.dwell(s=1.5)
    for i in range(150):                 # 150 blended blocks of a gentle taper, 0.1 mm/rev
        s.line(0, per_rev_mm=0.1, tol_mm=0.01, X=-100 + 0.002 * (i + 1), Z=-200 - 0.1 * (i + 1))
    s.stop()
    res, dom, state, ctrl = run(LATHE, s)
    assert ctrl.done()
    blk = stack(res.frames, "block_index")
    z = stack(res.frames, "axis_cmd")[:, 1]
    theta = 2 * np.pi * stack(res.frames, "spindle_rev") + stack(res.frames, "spindle_angle_rad")
    lines = [r.index for r in dom.timeline.records if r.kind == "linear"]
    sel = np.isin(blk, lines[20:130])
    dz = np.diff(z)[sel[1:]]
    dth = np.diff(theta)[sel[1:]]
    # the axial feed per revolution: 0.1 mm/rev along the path, whose direction is (0.002, 0.1) per block
    assert np.median(np.abs(dz / dth)) * 2 * np.pi == pytest.approx(0.1e-3 / np.hypot(1.0, 0.02), rel=1e-3)
    assert len(lines) == 150 and all(r.tick1 >= r.tick0 for r in dom.timeline.records)
    # every block's programmed-speed time is its length at 0.1 mm/rev and 1200 rpm, blended blocks included
    recs = [r for r in dom.timeline.records if r.kind == "linear"]
    ideal = 0.1e-3 * np.hypot(1.0, 0.02) / (0.1e-3 * 1200 / 60)
    assert all(r.ideal_ticks * 1e-6 == pytest.approx(ideal, rel=1e-6) for r in recs[1:-1])


def test_per_revolution_corners_blend_and_finish():
    from mfix import LATHE
    s = Script()
    s.spindle(1000)
    s.rapid(X=-100, Z=-200)
    s.dwell(s=1.5)
    x, z = -100.0, -200.0
    for i in range(12):                  # a staircase: 90 degree corners, blended within 0.01 mm
        if i % 2 == 0:
            z -= 2.0
        else:
            x += 0.5
        s.line(0, per_rev_mm=0.2, tol_mm=0.01, X=x, Z=z)
    s.stop()
    res, dom, state, ctrl = run(LATHE, s)
    assert ctrl.done()
    recs = [r for r in dom.timeline.records if r.kind == "linear"]
    assert len(recs) == 12
    cmd = stack(res.frames, "axis_cmd")[:, :2]
    speed = np.linalg.norm(np.diff(cmd, axis=0), axis=1) / dom.env.ipo_s
    blk = stack(res.frames, "block_index")[1:]
    # the commanded path speed of the long Z steps is the per-revolution speed: 0.2 mm/rev at 1000 rpm
    z_steps = [r.index for r in recs[::2]]
    assert np.max(speed[np.isin(blk, z_steps)]) == pytest.approx(0.2e-3 * 1000 / 60, rel=2e-3)
    for r in recs:
        assert r.ideal_ticks > 0 and r.motion_ticks >= r.ideal_ticks - 2 * dom.ipo_ticks


def test_one_feedback_per_buffer_stop_when_a_live_read_starts_inside_a_step():
    """A live read (SYNC) prepared after the axes came to rest starts inside a macro step; its feedback is
    committed once, with the tick that started it, so the probe move after it gets its own feedback (a duplicate
    used to answer the probe's wait with 'not triggered')."""
    s = Script()
    s.rapid(X=-100, Y=-100, Z=-150)
    sync = s.sync("live_read")
    sync.ready_tick = 2_003_500                   # block preparation ends 3.5 ms into a macro step
    s.probe(300, X=-100, Y=-100, Z=-160)
    s.stop()
    plane_M = 0.6 - 0.155
    res, dom, state, ctrl = run(MILL, s, probe=lambda p, k, inputs: inputs.out.tip_M[k, p, 2] <= plane_M)
    assert ctrl.done()
    fbs = [fb for _, fb in ctrl.feedbacks[0]]
    assert len(fbs) == 2
    assert fbs[0].probe is None and fbs[1].probe is not None and fbs[1].probe.triggered
    retired = sum(1 for _ in ctrl.retire_ticks[0])
    assert retired == len(s.cmds)                 # every command retired exactly once


def test_per_revolution_run_after_a_stop_keeps_its_length_while_the_spindle_turns():
    """The lathe's C axis is the spindle (not in C-axis mode it turns with it): a per-revolution block after an exact
    stop, ending in a blended corner, is not rebuilt over the spindle angle the C command carries by then."""
    from mfix import LATHE
    s = Script()
    s.spindle(1000)
    s.rapid(X=-100, Z=-200)
    s.dwell(s=1.0)
    s.line(0, per_rev_mm=0.2, exact=True, X=-100, Z=-202)
    s.line(0, per_rev_mm=0.2, tol_mm=0.01, X=-99, Z=-204)
    s.line(0, per_rev_mm=0.2, tol_mm=0.01, X=-99, Z=-206)
    s.stop()
    res, dom, state, ctrl = run(LATHE, s, max_s=20.0)
    assert ctrl.done()
    pl = dom.planners[0]
    for j in range(pl.piece0, pl.n_pieces()):
        pc = pl.piece(j)
        assert pc.length <= 2.3e-3                # no piece is longer than its block (sqrt(1 + 4) mm at most)
    recs = [r for r in dom.timeline.records if r.kind == "linear"]
    speed = 0.2e-3 * 1000 / 60
    assert (recs[1].tick1 - recs[1].tick0) * 1e-6 < 2.0 * math.hypot(1e-3, 2e-3) / speed
