"""Look-ahead planner: feed limits, data starvation, corners, exact stop, overrides, rapids."""

import copy
import math

import numpy as np
import pytest

from mfix import MILL, Script, five_axis, run, stack


def _speeds(frames, blocks):
    cmd = stack(frames, "axis_cmd")
    blk = stack(frames, "block_index")
    v = np.r_[0.0, np.linalg.norm(np.diff(cmd[:, :3], axis=0), axis=1) / 1e-3]
    keep = np.isin(blk, blocks)
    return v[keep], cmd, blk


@pytest.mark.parametrize("radius_mm,feed", [(5.0, 10000.0), (2.0, 6000.0), (20.0, 20000.0)])
def test_arc_feed_is_clamped_by_centripetal_acceleration_and_jerk(radius_mm, feed):
    s = Script()
    s.rapid(X=-100, Y=-100, Z=-100)
    s.line(feed, X=-100 + radius_mm, Y=-100, Z=-100)
    s.arc(feed, centre_mm=(-100, -100, -100), sweep_rad=2 * math.pi, radius_mm=radius_mm, X=-100 + radius_mm, Y=-100,
          Z=-100)
    s.stop()
    res, dom, state, ctrl = run(MILL, s)
    v, cmd, blk = _speeds(res.frames, [2])
    R = radius_mm * 1e-3
    A, J = 5.0, 200.0
    clamp = min(feed / 60e3, math.sqrt(A * R), (J * R * R) ** (1.0 / 3.0))
    # the commanded chord speed over a tick underestimates the arc speed by at most (w dt)^2 / 24
    assert v.max() <= clamp * (1 + 1e-6)
    assert v.max() >= clamp * (1 - 5e-3)
    # and the commanded path stays on the circle
    r = np.hypot(cmd[blk == 2, 0] + 0.1, cmd[blk == 2, 1] + 0.1)
    assert np.max(np.abs(r - R)) < 1e-9


def test_data_starvation_on_short_segments_limits_feed_to_block_rate():
    s = Script()
    s.rapid(X=-100, Y=-100, Z=-100)
    for i in range(300):
        s.line(9000, X=-100 + 0.05 * (i + 1), Y=-100, Z=-100)
    s.stop()
    # 1 ms block processing, 40-block look-ahead buffer (the control's interpreter clock)
    res, dom, state, ctrl = run(MILL, s, block_ticks=1000, capacity=40)
    v, _, _ = _speeds(res.frames, list(range(30, 280)))
    bound = 0.05e-3 / 1e-3                       # one 0.05 mm block per millisecond
    assert np.median(v) == pytest.approx(bound, rel=2e-3)
    # once the 40 blocks prepared during the rapid are used up, the block rate holds the feed
    late, _, _ = _speeds(res.frames, list(range(150, 280)))
    assert late.max() <= bound * 1.01
    cats = {}
    for r in dom.timeline.records:
        for k, x in r.ticks.items():
            cats[k] = cats.get(k, 0) + x
    assert cats.get("data_starvation", 0) > 0.1 / 1e-6      # most of the lost time is starvation
    # the same program with slow block preparation absent runs at the programmed feed
    s2 = Script()
    s2.rapid(X=-100, Y=-100, Z=-100)
    for i in range(300):
        s2.line(1500, X=-100 + 0.05 * (i + 1), Y=-100, Z=-100)
    s2.stop()
    res2, _, _, _ = run(MILL, s2, block_ticks=1000, capacity=40)
    v2, _, _ = _speeds(res2.frames, list(range(30, 280)))
    assert np.median(v2) == pytest.approx(1500 / 60e3, rel=1e-6)


@pytest.mark.parametrize("tol_mm,angle_deg", [(0.01, 90.0), (0.005, 45.0), (0.05, 120.0)])
def test_corner_blend_deviation_within_tolerance(tol_mm, angle_deg):
    s = Script()
    s.rapid(X=-100, Y=-100, Z=-100)
    s.line(3000, tol_mm=tol_mm, X=-80, Y=-100, Z=-100)
    a = math.radians(angle_deg)
    s.line(3000, tol_mm=tol_mm, X=-80 + 20 * math.cos(a), Y=-100 + 20 * math.sin(a), Z=-100)
    s.stop()
    res, dom, state, ctrl = run(MILL, s)
    cmd = stack(res.frames, "axis_cmd")
    act = stack(res.frames, "axis_actual")
    corner = np.array([-0.08, -0.1])
    end = corner + 0.02 * np.array([math.cos(a), math.sin(a)])
    tol = tol_mm * 1e-3

    def dist_to_path(p):
        d = []
        for a0, a1 in ((np.array([-0.1, -0.1]), corner), (corner, end)):
            t = np.clip(((p - a0) @ (a1 - a0)) / ((a1 - a0) @ (a1 - a0)), 0.0, 1.0)
            d.append(np.linalg.norm(p - (a0 + t[:, None] * (a1 - a0)), axis=-1))
        return np.minimum(d[0], d[1])

    # every commanded setpoint of the two feed blocks lies within the tolerance band of the programmed path
    blk = stack(res.frames, "block_index")
    feed = np.isin(blk, [1, 2])
    assert np.max(dist_to_path(cmd[feed, :2])) <= tol * (1 + 1e-9)
    # the blend's closest approach to the programmed corner is the tolerance (the deviation it allows)
    pl = dom.planners[0]
    blends = [pl.piece(j) for j in range(pl.n_pieces()) if pl.piece(j).blend]
    assert len(blends) == 2
    pts = []
    for pc in blends:
        out = np.zeros((201, len(dom.model.axes)))
        pc.eval(np.linspace(0.0, pc.length, 201), np.arange(len(dom.model.axes), dtype=np.int32), out)
        pts.append(out[:, :2])
    pts = np.vstack(pts)
    assert np.min(np.linalg.norm(pts - corner, axis=1)) == pytest.approx(tol, rel=1e-6)
    # the blend rounds the corner (it does not stop): the commanded path passes within the tolerance band
    # and the time lost at the corner is far less than a full stop
    exact = Script()
    exact.rapid(X=-100, Y=-100, Z=-100)
    exact.line(3000, exact=True, X=-80, Y=-100, Z=-100)
    exact.line(3000, X=-80 + 20 * math.cos(a), Y=-100 + 20 * math.sin(a), Z=-100)
    exact.stop()
    res_e, dom_e, _, _ = run(MILL, exact)
    assert res.end_tick < res_e.end_tick
    # actual path: the servo lag adds at most (1 - kff) v_corner / Kv + the lag change inside the blend
    d_act = np.min(np.hypot(act[:, 0] - corner[0], act[:, 1] - corner[1]))
    assert d_act <= tol + (1 - 0.9) * (3000 / 60e3) / 40.0 + 1e-6


def test_exact_stop_waits_for_in_position():
    s = Script()
    s.rapid(X=-100, Y=-100, Z=-100)
    s.line(3000, exact=True, X=-80, Y=-100, Z=-100)
    s.line(3000, X=-80, Y=-80, Z=-100)
    s.stop()
    res, dom, state, ctrl = run(MILL, s)
    recs = dom.timeline.records
    assert recs[1].ticks.get("in_position", 0) > 0
    # after the exact stop the next block starts from rest at the corner, in position
    act = stack(res.frames, "axis_actual")
    blk = stack(res.frames, "block_index")
    first = np.nonzero(blk == 2)[0][0]
    assert abs(act[first - 1, 0] + 0.08) <= dom.model.in_position + 1e-12


def test_feed_override_and_feed_hold():
    def program():
        s = Script()
        s.rapid(X=-100, Y=-100, Z=-100)
        s.line(1200, X=-50, Y=-100, Z=-100)
        s.stop()
        return s
    from cnctwin.motion.driver import MotionRunner
    from mfix import ScriptedController, context, domain
    from cnctwin.core.state import SimState
    base, dom0, _, _ = run(MILL, program())
    ctrl = ScriptedController([program()])
    ctrl.state.paths[0].operator.feed_override = 0.5
    dom = domain(MILL)
    res = MotionRunner(ctrl, dom, ctx=context(), state=SimState()).run(60)
    feed_base = dom0.timeline.records[1].ticks["air_cutting"]
    feed_half = dom.timeline.records[1].ticks["air_cutting"]
    assert feed_base == pytest.approx(50.0 / 1200.0 * 60.0 / 1e-6, abs=1.0)
    assert feed_half == pytest.approx(2.0 * feed_base, abs=1.0)
    # feed hold: the axes brake to rest and wait; the rest of the cut follows after release
    ctrl = ScriptedController([program()])
    dom = domain(MILL)
    state = SimState()
    runner = MotionRunner(ctrl, dom, ctx=context(), state=state, keep_frames=True)
    panel = ctrl.state.paths[0].operator

    class Hold:
        def __init__(self):
            self.calls = 0

    orig = dom.kernel_inputs

    def kernel_inputs(ctx, st, bus):
        t = st.clock.tick * 1e-6
        panel.feed_hold = 1.0 <= t < 2.0
        return orig(ctx, st, bus)

    dom.kernel_inputs = kernel_inputs
    res = runner.run(60)
    held = dom.timeline.records[1].ticks.get("feed_hold", 0) * 1e-6
    assert 0.8 < held < 1.0
    v = np.linalg.norm(np.diff(stack(res.frames, "axis_cmd")[:, :3], axis=0), axis=1)
    t = np.arange(1, len(v) + 1) * 1e-3
    assert np.all(v[(t > 1.15) & (t < 1.95)] == 0.0)


def test_rapid_dogleg_per_axis_and_linear_for_siemens():
    s = Script()
    s.rapid(X=-200, Y=-50, Z=-10)
    s.stop()
    res, dom, state, ctrl = run(MILL, s)
    cmd = stack(res.frames, "axis_cmd")
    # dog-leg: Y and Z reach their targets long before X (independent axes at their own rapid)
    t_y = np.nonzero(np.abs(cmd[:, 1] + 0.05) < 1e-12)[0][0]
    t_x = np.nonzero(np.abs(cmd[:, 0] + 0.2) < 1e-12)[0][0]
    assert t_y < 0.5 * t_x
    m = copy.deepcopy(MILL)
    m["controller"]["dialect"] = "siemens840d"
    res2, dom2, _, _ = run(m, s)
    cmd2 = stack(res2.frames, "axis_cmd")
    moving = np.abs(cmd2[:, 0]) > 1e-9
    ratio = cmd2[moving, 1] / cmd2[moving, 0]
    assert np.allclose(ratio[ratio == ratio], 0.25, atol=1e-9)    # straight line X:Y = 200:50


def test_rotary_shortest_path_on_a_continuous_axis():
    m = five_axis("table_bc")
    s = Script()
    s.rapid(C_deg=300.0)                # from 0: the shortest path is -60 deg
    s.rapid(C_deg=10.0)                 # from 300 = -60: +70 deg, not -290 deg
    s.stop()
    res, dom, state, ctrl = run(m, s)
    c = stack(res.frames, "axis_cmd")[:, dom.model.axis_index["C"]]
    blk = stack(res.frames, "block_index")
    assert np.min(c) == pytest.approx(math.radians(-60.0), abs=1e-12)
    assert c[-1] == pytest.approx(math.radians(10.0), abs=1e-12)
    assert np.all(np.diff(c[blk == 1]) >= -1e-15)


def test_rebuilt_pieces_stay_in_path_order():
    """A rebuilt item's pieces replace its old ones in place: later items shift, joins stay adjacent."""
    from cnctwin.core.state import SimState
    from mfix import context, domain
    dom = domain(MILL)
    state = SimState()
    ctx = context()
    dom.init(ctx, state)
    s = Script()
    s.rapid(X=-100, Y=-100, Z=-150)
    for k in range(4):
        s.line(1200, tol_mm=0.01, X=-100 + 10 * (k + 1), Y=-100 + 5 * (k % 2), Z=-150)
    dom.accept(ctx, state, 0, s.cmds)
    pl = dom.planners[0]
    items = [it for it in pl.items if it.pieces]
    target = items[2]
    before = {it.index: list(it.pieces) for it in items}
    extra = copy.deepcopy(pl.piece(target.pieces[0]))
    pl._put_pieces(target, [copy.deepcopy(pl.piece(j)) for j in target.pieces] + [extra], pl.st)
    assert target.pieces == list(range(before[target.index][0], before[target.index][-1] + 2))
    for it in items:
        if it.index > target.index:
            assert it.pieces == [j + 1 for j in before[it.index]]
        elif it.index < target.index:
            assert it.pieces == before[it.index]
    owners = [pl.piece(j).item for j in range(pl.piece0, pl.n_pieces())]
    assert owners == sorted(owners)
    # a later join appends after the last item's pieces, which are still the last in the list
    s2 = Script()
    s2.seq = s.seq
    s2.line(1200, tol_mm=0.01, X=-40, Y=-90, Z=-150)
    dom.accept(ctx, state, 0, s2.cmds)
    owners = [pl.piece(j).item for j in range(pl.piece0, pl.n_pieces())]
    assert owners == sorted(owners)
    for it in pl.items:
        if it.pieces:
            assert it.pieces == list(range(it.pieces[0], it.pieces[-1] + 1))
