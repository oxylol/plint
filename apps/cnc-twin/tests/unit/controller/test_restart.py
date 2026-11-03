"""Program restart through the state machine: search with modal reconstruction (P and Q types), restored
M/S/T state, the axis-by-axis approach, alarms, and the dangerous-restart analysis for the sanity domain."""

import pytest

from cnctwin.controller import ToolOffsetInit
from cnctwin.controller.dryrun import dry_run
from cnctwin.controller.ir import FeedMode, Kind
from cnctwin.controller.restart import restart, restart_analysis
from cnctwin.controller.schema import ControllerSettings
from harness import MILL, alarm, controller, moves, wrap

G54 = {"G54": [-0.4, -0.2, -0.3]}
TOOLS = [ToolOffsetInit(1, length_m=0.1, radius_m=0.005), ToolOffsetInit(2, length_m=0.12, radius_m=0.003)]
PROGRAM = """N10 G90 G54 G17 G21
N20 T1 M06
N30 S1000 M03
N40 G00 X10. Y10. Z5.
N50 M08
N60 G91
N70 G01 X5. F200.
N80 G90 G00 X50. Y20.
N90 G01 Z-2. F100.
N100 X60.
N110 S1200
N120 G00 Z50.
N130 M09"""


@pytest.fixture(autouse=True)
def _providers():
    from cnctwin.controller import hooks, restart as module
    hooks.load_plugins()
    module.register()


def ctl(body=PROGRAM, settings=None):
    return controller(wrap(body), MILL, tools=TOOLS, offsets=G54, settings=settings or ControllerSettings())


def mm(cmd):
    return {k: round(v * 1e3, 4) for k, v in cmd.target.items()}


def test_restart_rebuilds_modal_state_restores_mst_and_approaches_axis_by_axis():
    c = ctl()
    info = restart(c, n=100)
    assert info["line"] == 12 and info["n"] == 100 and info["steps"] == 6
    r = dry_run(c)
    assert alarm(r) is None
    kinds = [cmd.kind for cmd in r.commands[0]]
    assert kinds[:3] == [Kind.TOOL_CHANGE, Kind.SPINDLE, Kind.COOLANT], "the last T, S/M03 and M08 are output"
    assert r.commands[0][0].tool["number"] == 1 and r.commands[0][1].spindle["omega_rad_s"] < 0   # M03
    approach = [cmd for cmd in moves(r) if cmd.modal.get("program_restart")]
    assert [mm(cmd) for cmd in approach] == [{"X": -350.0, "Y": 0.0, "Z": 0.0}, {"X": -350.0, "Y": -180.0, "Z": 0.0},
                                             {"X": -350.0, "Y": -180.0, "Z": -302.0}]
    assert all(cmd.feed.value == pytest.approx(5000 / 60e3) for cmd in approach)
    resumed = [cmd for cmd in moves(r) if not cmd.modal.get("program_restart")][0]
    assert resumed.kind == Kind.LINEAR and mm(resumed)["X"] == -340.0, "N100 X60. runs in G90 G01 F100"
    assert resumed.feed.mode == FeedMode.PER_MIN and resumed.feed.value == pytest.approx(0.1 / 60)


def test_restart_by_line_frequency_and_output_modes():
    body = "N10 G90 G00 X0 Y0 Z10.\nS500 M03\nN20 X1.\nS700 M04\nN20 X2.\nN30 X3."
    c = ctl(body)
    restart(c, n=20, frequency=2)
    r = dry_run(c)
    assert r.commands[0][0].kind == Kind.SPINDLE and r.commands[0][0].spindle["omega_rad_s"] > 0   # M04
    assert [mm(cmd)["X"] for cmd in moves(r) if not cmd.modal.get("program_restart")][:2] == [-398.0, -397.0]
    c = ctl(body, ControllerSettings(cycle_parameters={"restart_output": "all"}))
    restart(c, line=7)
    r = dry_run(c)
    assert [cmd.kind for cmd in r.commands[0][:2]] == [Kind.SPINDLE, Kind.SPINDLE]
    c = ctl(body, ControllerSettings(cycle_parameters={"restart_output": "none"}))
    restart(c, line=7)
    assert dry_run(c).commands[0][0].kind == Kind.LINEAR


def test_restart_alarms_and_q_type():
    c = ctl()
    restart(c, n=999)
    assert c.state.paths[0].alarm["code"] == "PS0060"
    c = ctl("G90 G00 X0 Y0 Z10.\nM29 S500\nG84 X0 Y0 Z-10. R5. F1.\nN50 X10.\nG80")
    restart(c, n=50)
    assert c.state.paths[0].alarm["code"] == "TW0024"
    c = ctl()
    for tick in range(0, 400000, 20000):              # run part of the program, then restart from scratch
        c.produce(0, tick)
        if c.state.paths[0].buffer:
            c.retire(0, len(c.state.paths[0].buffer), tick)
    restart(c, n=90, kind="Q")
    r = dry_run(c)
    first = [cmd for cmd in moves(r) if cmd.modal.get("program_restart")][0]
    assert mm(first) == {"X": -350.0, "Y": 0.0, "Z": 0.0}, "Q type approaches from the reference position"
    with pytest.raises(ValueError):
        restart(ctl(), n=10, line=3)


def test_dangerous_restart_analysis_reports_unrestated_modal_state():
    body = ("G90 G54 G00 X0 Y0 Z50.\nT1 M06\nS1000 M03\nG00 X10. Y10.\nG43 H1 Z5.\nG91 G01 Z-1. F100.\n"
            "T2 M06\nX20.\nG90 G00 Z50.\nT1 M06\nG90 G00 G49 X0 Y0 S800 M03\nM30")
    c = ctl(body)
    records = restart_analysis(c.setup)
    assert [rec["line"] for rec in records] == [4, 9, 12]
    assert records[0]["issues"] == []
    groups = {issue["group"]: issue for issue in records[1]["issues"]}
    assert set(groups) == {"spindle", "length_comp", "distance", "motion"}
    assert groups["distance"]["value"] == "G91" and groups["distance"]["severity"] == "error"
    assert groups["spindle"]["first_move_line"] == 10 and "power-on 'M05'" in groups["spindle"]["message"]
    assert records[2]["issues"] == [], "the last section restates G90 G00 G49 and the spindle"
    from cnctwin.sanity.rules import _restart_from_controller

    class Check:
        program = type("P", (), {"name": "test.nc"})()
        restart_analysis = records

        def __init__(self):
            self.found = []

        def issue(self, rule, severity, message, line, program):
            self.found.append((rule, severity.value, line, message))

    check = Check()
    _restart_from_controller(check)
    assert len(check.found) == 4 and all(f[0] == "restart-modal" and f[2] == 9 for f in check.found)
