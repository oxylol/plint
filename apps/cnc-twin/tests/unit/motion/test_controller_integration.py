"""The real controller (FANUC dialect) in lock-step with motion: programs run to their end with the
look-ahead buffer, block-time model, retire and feedback protocol."""

import copy
import math

import numpy as np
import pytest

from cnctwin.controller import Controller, ControllerSetup, ProgramSource, ToolOffsetInit, dry_run
from cnctwin.controller.schema import ControllerSettings
from cnctwin.core.state import SimState
from cnctwin.motion.driver import MotionRunner
from mfix import LATHE, MILL, context, domain, stack

FANUC_DB = {
    "format": "cnctwin-controller", "version": 1, "id": "fixture", "dialect": "fanuc", "name": "fixture",
    "alarms": [{"code": "PS0010", "text": "IMPROPER G-CODE"}, {"code": "PS0011", "text": "FEED ZERO ( COMMAND )"},
               {"code": "OT0500", "text": "+ OVERTRAVEL ( SOFT 1 )", "trigger": "overtravel"},
               {"code": "OT0501", "text": "- OVERTRAVEL ( SOFT 1 )", "trigger": "overtravel"}],
}


def controller(program, machine, *, offsets=None, tools=None, programs=None):
    sources = [ProgramSource(f"p{k}.nc", t, path=k) for k, t in enumerate(programs)] if programs else \
        [ProgramSource("t.nc", program)]
    setup = ControllerSetup(copy.deepcopy(machine), sources, ControllerSettings(), controller_db=FANUC_DB,
                            db_id="fixture", work_offsets_m=[offsets or {}] + [{} for _ in range(3)],
                            tools=tools or [ToolOffsetInit(1, length_m=0.1, radius_m=0.005),
                                            ToolOffsetInit(2, length_m=0.1, radius_m=0.003)])
    return Controller(setup)


def run_real(program, machine, **kw):
    ctrl = controller(program, machine, **kw)
    dom = domain(machine)
    state = SimState()
    res = MotionRunner(ctrl, dom, ctx=context(), state=state, keep_frames=True).run(120)
    return res, dom, state, ctrl


MILL_PROGRAM = """%
O1000
G21 G90 G17 G40 G80
T1 M06
S6000 M03
M08
G54 G00 X-50. Y-40.
G43 H1 Z-20.
G01 Z-30. F600.
X-20. F1200.
G02 X-10. Y-30. I0. J10.
G01 Y-10.
G04 P250
G00 Z-5.
M09
M05
M30
%"""


def test_fanuc_mill_program_runs_to_its_end():
    res, dom, state, ctrl = run_real(MILL_PROGRAM, MILL, offsets={"G54": [-0.4, -0.25, -0.2]})
    assert ctrl.done()
    assert all(p.alarm is None for p in ctrl.state.paths)
    assert not [f for f in state.findings if f.severity.value in ("error", "fatal")]
    recs = dom.timeline.records
    assert [r.line for r in recs][-1] == 17                    # the M30 block
    for a, b in zip(recs, recs[1:]):
        assert b.tick0 == a.tick1
    for r in recs:
        assert sum(r.ticks.values()) == r.tick1 - r.tick0
    # the tool ends where the program left it (G00 Z-5. with H1 = 100 mm: machine Z = -0.2 - 0.005 + 0.1)
    final = stack(res.frames, "axis_actual")[-1]
    assert final == pytest.approx([-0.4 - 0.010, -0.25 - 0.010, -0.2 - 0.005 + 0.1], abs=2e-6)
    # the physical machine is slower than the ideal one (acceleration, in-position, spindle run-up); the ideal
    # machine books the published chip-to-chip time for M06, the physical one its sequence
    ideal = dry_run(controller(MILL_PROGRAM, MILL, offsets={"G54": [-0.4, -0.25, -0.2]}))
    tc_phys = sum(r.tick1 - r.tick0 for r in recs if r.kind == "tool_change")
    tc_ideal = int(round(MILL["tool_changer"]["chip_to_chip_s"] / 1e-6))
    assert recs[-1].tick1 - tc_phys > ideal.end_tick - tc_ideal
    # the arc was cut on its circle: radius 10 mm about (X-20, Y-30) in work coordinates
    blk = stack(res.frames, "block_index")
    cmd = stack(res.frames, "axis_cmd")
    arc = [r.index for r in recs if r.kind == "arc"][0]
    r_arc = np.hypot(cmd[blk == arc, 0] - (-0.42), cmd[blk == arc, 1] - (-0.28))
    assert np.allclose(r_arc, 0.010, atol=1e-9)


LATHE_PROGRAM = """%
O2000
T0101
G50 S2500
G96 S180 M03
G00 X60. Z2.
G99 G01 Z-20. F0.2
G00 U4. Z2.
G97 S800
G00 X40. Z5.
G32 Z-15. F1.5
G00 X50.
M05
M30
%"""


def test_fanuc_lathe_program_css_per_rev_and_threading():
    res, dom, state, ctrl = run_real(LATHE_PROGRAM, LATHE, offsets={"G54": [0.0, -0.3]},
                                     tools=[ToolOffsetInit(1, x_m=-0.30, z_m=0.0)])   # X offset in diameter
    assert ctrl.done() and all(p.alarm is None for p in ctrl.state.paths)
    recs = dom.timeline.records
    for a, b in zip(recs, recs[1:]):
        assert b.tick0 == a.tick1
    blk = stack(res.frames, "block_index")
    z = stack(res.frames, "axis_cmd")[:, 1]
    theta = 2 * math.pi * stack(res.frames, "spindle_rev") + stack(res.frames, "spindle_angle_rad")
    lines = {r.line: r.index for r in recs}
    turn = np.nonzero(blk == lines[7])[0]
    dz = np.diff(z[turn])
    dth = np.diff(theta[turn])
    moving = np.abs(dz) > 1e-9
    assert np.median(np.abs(dz[moving] / dth[moving])) * 2 * math.pi == pytest.approx(0.2e-3, rel=1e-6)
    thread = np.nonzero(blk == lines[11])[0]
    dz = np.diff(z[thread])
    dth = np.diff(theta[thread])
    moving = np.abs(dz) > 1e-9
    assert np.median(np.abs(dz[moving] / dth[moving])) * 2 * math.pi == pytest.approx(1.5e-3, rel=1e-9)
    # constant surface speed while turning at diameter 60 (radius 30 mm): 180 m/min -> 955 rpm
    om = np.abs(stack(res.frames, "spindle_omega_rad_s"))
    assert np.median(om[turn]) == pytest.approx(180.0 / 60.0 / 0.030, rel=0.01)


SWISS = {
    "format": "cnctwin-machine", "version": 1, "id": "t-swiss", "maker": "T", "model": "T", "type": "swiss",
    "controller": dict(copy.deepcopy(MILL["controller"]),
                       rapid_mm_min={"X1": 30000, "Z1": 30000, "X2": 30000, "Z2": 30000}, smoothing=[],
                       diameter_programming_X=True),
    "axes": {n: copy.deepcopy(MILL["axes"]["X"]) for n in ("X1", "Z1", "X2", "Z2")},
    "kinematics": {"workpiece_chain": [{"link": "head", "axis": "Z1", "direction": [0, 0, 1]}],
                   "workpiece_chains": {"sub": [{"link": "subz", "axis": "Z2", "direction": [0, 0, -1]}]},
                   "tool_chain": [{"link": "gang", "axis": "X1", "direction": [1, 0, 0]}],
                   "tool_chains": {"back": [{"link": "back", "axis": "X2", "direction": [1, 0, 0]}]},
                   "home_mm": {"X1": 0, "Z1": 0, "X2": 0, "Z2": 0}, "tool_axis": [1, 0, 0]},
    "spindles": [{"id": "main", "kind": "turning", "nose": "c", "max_rpm": 10000,
                  "gear_ranges": [{"rpm": [0, 10000], "power_kW": [[0, 0], [10000, 3.7]],
                                   "torque_Nm": [[0, 20], [10000, 3.5]]}]},
                 {"id": "sub", "kind": "sub", "nose": "c", "max_rpm": 8000,
                  "gear_ranges": [{"rpm": [0, 8000], "power_kW": [[0, 0], [8000, 2.2]],
                                   "torque_Nm": [[0, 12], [8000, 2.6]]}]}],
    "turrets": [{"id": "gang", "stations": 8, "index_time_s": {"adjacent": 0.3, "max": 0.6}}],
    "coolant": {"pumps": [{"id": "flood", "kind": "flood"}]},
}
for _ax in SWISS["axes"].values():
    _ax["travel_mm"] = [-300, 300]


def test_two_path_program_waits_with_m100():
    p1 = "%\nO0001\nG98 G00 X-100. Z-50.\nM100\nG01 Z-60. F600.\nM30\n%\n"
    p2 = "%\nO0002\nG98 G00 X-10.\nM100\nG01 Z-10. F300.\nM30\n%\n"
    res, dom, state, ctrl = run_real("", SWISS, programs=[p1, p2])
    assert ctrl.done() and all(p.alarm is None for p in ctrl.state.paths)
    recs = dom.timeline.records
    waits = [r for r in recs if r.kind == "sync"]
    assert len(waits) == 2
    # path 2 had the short move: it waited for path 1 at M100, both left together
    r2 = [r for r in recs if r.path == 1]
    assert sum(r.ticks.get("path_wait", 0) for r in r2) > 0
    # the moves after M100 start together, after both paths arrived (path 1 last)
    starts = []
    for p, axis in ((0, 1), (1, 1)):
        cmd = stack(res.frames, "axis_cmd", p=p)[:, axis]
        ticks = stack(res.frames, "tick", p=p)
        blk = stack(res.frames, "block_index", p=p)
        lin = [r.index for r in recs if r.path == p and r.kind == "linear"][0]
        sel = np.nonzero(blk == lin)[0]
        moving = sel[np.abs(cmd[sel] - cmd[sel[0] - 1]) > 1e-12]
        starts.append(int(ticks[moving[0]]))
    sync1 = [r for r in recs if r.path == 0 and r.kind == "sync"][0]
    assert abs(starts[0] - starts[1]) <= 2000
    assert min(starts) >= sync1.tick1
