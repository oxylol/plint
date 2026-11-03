"""Times of non-motion commands: tool change (with the Z move and spindle stop), turret index, dwell,
coolant delays, M functions, operator stops."""

import pytest

from cnctwin.motion.scurve import double_s
from cnctwin.motion.timing import ticks_of
from mfix import LATHE, MILL, Script, run, stack


def _rec(dom, kind):
    return [r for r in dom.timeline.records if r.kind == kind]


def test_tool_change_with_spindle_running_and_z_move_home():
    s = Script()
    s.tool_change(1, slot=0)
    s.spindle(6000)
    s.rapid(X=-100, Y=-100, Z=-150)
    s.line(1000, X=-100, Y=-100, Z=-160)
    s.tool_change(2, slot=1)
    s.stop()
    res, dom, state, ctrl = run(MILL, s)
    tc = _rec(dom, "tool_change")
    assert len(tc) == 2
    # first change: Z already home, spindle stopped: just the exchange
    assert tc[0].tick1 - tc[0].tick0 == ticks_of(2.0, 1e-3) * 1000
    # second: stop from 6000 rpm (ramp 12000 rpm / 1.0 s) + orientation, while Z travels 160 mm home; then the
    # exchange
    model = dom.model
    sp = model.spindles[0]
    stop_s = 6000.0 / 12000.0 * 1.0 + sp.orient_s
    z = model.axis("Z")
    move_s = double_s(0.160, z.rapid, z.accel, z.jerk).duration
    expected = ticks_of(max(stop_s, move_s) + 2.0, 1e-3) * 1000
    assert tc[1].tick1 - tc[1].tick0 == expected
    assert tc[1].ticks == {"tool_change": expected}
    # the Z slide really went home during the change, and the spindle stopped
    z_act = stack(res.frames, "axis_actual")[:, 2]
    blk = stack(res.frames, "block_index")
    assert abs(z_act[blk == tc[1].index][-1]) < 1e-6
    om = stack(res.frames, "spindle_omega_rad_s")
    assert abs(om[blk == tc[1].index][-1]) < 1.0
    # chip-to-chip with the approach and spindle run-up is longer than the published exchange time
    assert (tc[1].tick1 - tc[1].tick0) * 1e-6 > model.tool_changer.tool_to_tool_s


def test_turret_index_time_by_station_steps():
    s = Script()
    s.turret(1)
    s.turret(2)
    s.turret(7)
    s.turret(12)
    s.stop()
    res, dom, state, ctrl = run(LATHE, s)
    tc = _rec(dom, "tool_change")
    durations = [(r.tick1 - r.tick0) * 1e-6 for r in tc]
    # from the start station (unknown): adjacent time; 1 -> 2 adjacent 0.4 s; 2 -> 7: 5 steps of 6 max ->
    # 0.4 + 0.6 * 4 / 5; 7 -> 12: 5 steps
    assert durations[0] == pytest.approx(0.4)
    assert durations[1] == pytest.approx(0.4)
    assert durations[2] == pytest.approx(0.4 + 0.6 * 4 / 5, abs=1e-3)
    assert durations[3] == pytest.approx(0.4 + 0.6 * 4 / 5, abs=1e-3)


def test_dwell_seconds_and_revolutions_at_the_actual_speed():
    s = Script()
    s.spindle(600)
    s.dwell(s=2.0)                  # also lets the spindle reach speed
    s.dwell(rev=5.0)                # 5 rev at 600 rpm = 0.5 s
    s.stop()
    res, dom, state, ctrl = run(MILL, s)
    d = _rec(dom, "dwell")
    assert (d[0].tick1 - d[0].tick0) == 2_000_000
    assert (d[1].tick1 - d[1].tick0) * 1e-6 == pytest.approx(0.5, abs=1.5e-3)


def test_coolant_and_m_function_times():
    s = Script()
    s.coolant(flood=True)
    s.coolant(tsc=True)
    s.coolant(flood=False, tsc=False)
    s.stop()
    res, dom, state, ctrl = run(MILL, s)
    c = _rec(dom, "coolant")
    p = dom.model.p
    assert (c[0].tick1 - c[0].tick0) * 1e-6 == pytest.approx(p["m_function_s"])
    assert (c[1].tick1 - c[1].tick0) * 1e-6 == pytest.approx(p["m_function_s"] + p["tsc_pressure_delay_s"])
    assert c[1].ticks == {"coolant_delay": c[1].tick1 - c[1].tick0}
    frames = res.frames
    assert any(f.paths[0].coolant_pumps.get("tsc") for f in frames)
    assert frames[-1].paths[0].coolant_pumps == {"flood": False, "tsc": False}


def test_program_stop_waits_for_the_operator_then_cycle_start():
    s = Script()
    s.rapid(X=-10, Y=-10, Z=-10)
    s.stop("M00")
    s.rapid(X=-20, Y=-10, Z=-10)
    s.stop()
    res, dom, state, ctrl = run(MILL, s)
    st = [r for r in dom.timeline.records if r.stop == "M00"]
    assert len(st) == 1
    t = dom.timing
    assert (st[0].tick1 - st[0].tick0) * 1e-6 == pytest.approx(t.stop_s + 2 * t.door_s)
    assert st[0].ticks == {"program_stop": st[0].tick1 - st[0].tick0}
    assert ctrl.starts[0] == [st[0].tick1]
