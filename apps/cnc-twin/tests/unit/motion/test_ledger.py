"""Timeline and cycle-time ledger: exact integer sums, contiguity, the production reader, path waits."""

import copy
import math

import numpy as np
import pytest

from cnctwin.production.timeline import block_from_record
from mfix import MILL, Script, run, stack


def _program():
    s = Script(ready_step_ticks=700)
    s.tool_change(1, slot=0)
    s.spindle(5000)
    s.coolant(flood=True)
    s.rapid(X=-100, Y=-100, Z=-150)
    s.line(800, X=-100, Y=-100, Z=-160)
    s.line(2400, tol_mm=0.01, X=-60, Y=-100, Z=-160)
    s.arc(2400, centre_mm=(-60, -90, -160), sweep_rad=math.pi / 2, radius_mm=10, X=-50, Y=-90, Z=-160)
    for i in range(40):
        s.line(4000, X=-50 + 0.1 * (i + 1), Y=-90 + 0.02 * i, Z=-160)
    s.dwell(s=0.25)
    s.rapid(X=-100, Y=-100, Z=-10)
    s.stop("M00")
    s.tool_change(2, slot=1)
    s.rapid(X=-20, Y=-30, Z=-50)
    s.coolant(flood=False)
    s.stop()
    return s


def test_records_sum_exactly_to_the_cycle_time():
    res, dom, state, ctrl = run(MILL, _program(), block_ticks=900, capacity=12)
    recs = dom.timeline.records
    assert len(recs) >= 50
    for r in recs:
        assert r.tick1 >= r.tick0
        assert sum(r.ticks.values()) == r.tick1 - r.tick0, (r.index, r.ticks)
        assert all(v > 0 for v in r.ticks.values())
        assert not r.pending_feed
    # contiguous: every record starts where the previous ended (waits belong to the next block)
    for a, b in zip(recs, recs[1:]):
        assert b.tick0 == a.tick1
    span = recs[-1].tick1 - recs[0].tick0
    assert state.ledger.cycle_time_ticks == span
    assert state.ledger.block_time_ticks == span
    assert sum(sum(r.ticks.values()) for r in recs) == span
    # every executed tick of the run is in one record: the program ends with the M30 at the last tick
    assert recs[-1].kind == "stop" and recs[-1].tick1 == recs[-1].tick0
    cats = {}
    for r in recs:
        for k, v in r.ticks.items():
            cats[k] = cats.get(k, 0) + v
    for name in ("tool_change", "spindle_ramp", "coolant_delay", "rapid", "air_cutting", "acceleration",
                 "dwell", "program_stop", "in_position"):
        assert cats.get(name, 0) > 0, name
    out = dom.finalize(None, state)
    assert out["paths"][0]["cycle_time_s"] == pytest.approx(span * 1e-6)


def test_timeline_records_read_by_production():
    res, dom, state, ctrl = run(MILL, _program())
    doc = dom.timeline_json()
    assert doc["format"] == "cnctwin-timeline"
    blocks = [block_from_record(r) for r in doc["blocks"]]
    for b, r in zip(blocks, dom.timeline.records):
        assert b.duration_ticks == r.tick1 - r.tick0
        assert dict(b.categories()) == {k: v for k, v in r.ticks.items() if v}
    tools = {b.tool for b in blocks if b.kind != "tool_change"}
    assert {1, 2} <= tools


def test_frames_point_at_their_records():
    res, dom, state, ctrl = run(MILL, _program())
    ticks = stack(res.frames, "tick")
    blocks = stack(res.frames, "block_index")
    recs = dom.timeline.records
    for t, b in zip(ticks[::37], blocks[::37]):
        r = recs[b]
        assert r.tick0 < t <= r.tick1 or (r.tick0 == t - 1000 and r.tick1 >= t - 1000)


def test_two_paths_wait_for_each_other():
    m = copy.deepcopy(MILL)
    m["type"] = "swiss"
    m["axes"] = {"X1": m["axes"]["X"], "Z1": m["axes"]["Z"], "X2": copy.deepcopy(m["axes"]["X"]),
                 "Z2": copy.deepcopy(m["axes"]["Z"])}
    m["kinematics"] = {"workpiece_chain": [{"link": "head", "axis": "Z1", "direction": [0, 0, 1]}],
                       "workpiece_chains": {"sub": [{"link": "sub", "axis": "Z2", "direction": [0, 0, 1]}]},
                       "tool_chain": [{"link": "gang", "axis": "X1", "direction": [1, 0, 0]}],
                       "tool_chains": {"back": [{"link": "back", "axis": "X2", "direction": [1, 0, 0]}]},
                       "home_mm": {"X1": 0, "Z1": 0, "X2": 0, "Z2": 0}, "tool_axis": [1, 0, 0]}
    m["controller"]["rapid_mm_min"] = {"X1": 30000, "Z1": 30000, "X2": 30000, "Z2": 30000}
    p1 = Script(path=0)
    p1.rapid(X1=-300, Z1=-10)          # long move
    p1.sync("path_wait", paths=(0, 1), code="M100")
    p1.rapid(X1=-310, Z1=-10)
    p1.stop()
    p2 = Script(path=1)
    p2.rapid(X2=-20, Z2=-10)           # short move: waits for path 1
    p2.sync("path_wait", paths=(0, 1), code="M100")
    p2.rapid(X2=-30, Z2=-10)
    p2.stop()
    res, dom, state, ctrl = run(m, [p1, p2])
    assert len(dom.model.paths) == 2
    (paths, release), = ctrl.releases
    assert paths == (0, 1)
    # both continue after the release; path 2 waited (path_wait time) for path 1
    r2 = [r for r in dom.timeline.records if r.path == 1]
    waited = sum(r.ticks.get("path_wait", 0) for r in r2)
    r1_first = [r for r in dom.timeline.records if r.path == 0][0]
    r2_first = r2[0]
    assert waited == pytest.approx((r1_first.tick1 - r2_first.tick1) - 0, abs=2000)
    # path 2's move after the wait starts no earlier than the release
    after = [r for r in r2 if r.kind == "rapid"][1]
    x2 = stack(res.frames, "axis_cmd", p=1)[:, 0]
    ticks = stack(res.frames, "tick", p=1)
    first_move = ticks[np.nonzero(np.abs(x2 + 0.02) > 1e-12)[0]]
    assert first_move[first_move > r2_first.tick1][0] > release
    for p in (0, 1):
        recs = [r for r in dom.timeline.records if r.path == p]
        for a, b in zip(recs, recs[1:]):
            assert b.tick0 == a.tick1
        for r in recs:
            assert sum(r.ticks.values()) == r.tick1 - r.tick0
