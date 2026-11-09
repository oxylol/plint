"""The controller's dry run as the scan's aid (macro offsets, Haas pattern cycles, TCP moves, lathe cycle
flow), probe tools, vise jaws in a flipped setup, through cuts measured to the stock bottom, arc ae."""

import math

import numpy as np
import pytest

from cnctwin.controller.ir import Command, Feed, FeedMode, Kind, LineRef, Space
from cnctwin.sanity import SanityContext, check_operation, check_program
from cnctwin.sanity.assist import Assist, Execution

from sanity_testkit import program, setup_data

HEAD = ("%", "O1000", "G21 G90 G17 G40 G49 G80 G94", "G54")
TAIL = ("G00 Z50.", "M09", "M05", "M30", "%")


def _issues(report):
    return sorted((i.rule, i.line, i.severity.value) for i in report.issues)


def test_macro_offsets_and_haas_patterns_come_from_the_controller():
    data = setup_data()
    data["operations"][0]["program_text"] = program(
        *HEAD, "T1 M06 (D10 3FL CARBIDE ENDMILL)", "S10000 M03", "M08", "G00 X50. Y40.", "#1 = 1",
        "G43 H#1 Z20.", "G00 Z2.", "G01 Z-2. F600.", "G13 I8. K12. Q2. Z-2. D01 F600.", "G00 Z20.",
        "G81 G98 Z-5. R2. F300. L0", "G70 I20. J0. L4", "G80", *TAIL)
    report = check_operation(data, 0)
    assert report.assist is not None and report.assist.available and report.assist.alarm is None
    assert not [i for i in report.issues if i.rule in ("offset-length", "scan-limits") and i.line in (10, 13)]
    holes = sorted((round(h.x_m * 1e3, 3), round(h.y_m * 1e3, 3)) for h in report.program.scan.holes)
    assert holes == [(30.0, 40.0), (50.0, 20.0), (50.0, 60.0), (70.0, 40.0)]      # 4 holes on R20 about X50 Y40
    pocket = [k for k, c in enumerate(report.program.commands) if c.ref.line == 13 and c.kind == Kind.ARC]
    ae = report.engagement.ae_m[pocket]
    # radial depth from each arc's centre: the first ring opens the plunge's D10 bore to R8 (3 mm), the
    # next rings step Q2 (the chord-capsule width overshoots these arcs by up to the tool radius)
    assert np.nanmax(ae) == pytest.approx(0.003, abs=0.6e-3) and ae[-1] == pytest.approx(0.002, abs=0.6e-3)
    static = check_operation(data, 0, use_controller=False)
    assert ("offset-length", 10, "error") in _issues(static)                        # H#1 read as H0 alone
    assert any(i.rule == "scan-limits" and i.line == 13 for i in static.issues)


def test_tcp_moves_take_the_controller_tip_and_axis():
    text = program(*HEAD, "T1 M06", "S8000 M03", "G43 H1 Z30.", "G234 H1 X20. Y20. Z5. B20. C0.",
                   "G01 X30. Z5. F500.", "G49", *TAIL)
    ctx = SanityContext.from_project(setup_data(), 0)
    tilt = (math.sin(math.radians(20.0)), 0.0, math.cos(math.radians(20.0)))

    def tip(line, kind, xyz):
        return Command(kind, LineRef("t.nc", line, ""), Space.TIP, {"X": -0.4, "Y": -0.2, "Z": -0.3, "B": 0.35},
                       tuple(v * 1e-3 for v in xyz), tilt, feed=Feed(FeedMode.PER_MIN, 0.5 / 60.0),
                       modal={"tip_interpolation": "tip"})

    assist = Assist({("t.nc", 8): [Execution(7, 1, {}, [tip(8, Kind.RAPID, (20.0, 20.0, 5.0))])],
                     ("t.nc", 9): [Execution(8, 1, {}, [tip(9, Kind.LINEAR, (30.0, 21.0, 5.0))])]},
                    ("X", "Y", "Z", "B"), ("X", "Y", "Z", "B"))
    report = check_program(text, "t.nc", ctx, assist=assist)
    moves = [k for k, c in enumerate(report.program.commands) if c.ref.line in (8, 9)]
    assert np.allclose(report.program.tip_end_W_m[moves[-1]], [0.030, 0.021, 0.005])   # the controller's tip
    assert report.program.commands[moves[-1]].tip_axis == pytest.approx(tilt)
    assert "tool axis tilted" in report.engagement.reasons[moves[-1]]
    assert not [i for i in report.issues if "tool centre point control" in i.message]


def test_lathe_g71_skips_the_shape_and_g70_runs_it():
    from cnctwin.sanity.scan import scan

    ctx = SanityContext.from_project(setup_data(), 0)
    ctx.machine_kind = "lathe"
    text = program("%", "O2000", "G21 G18 G40 G99", "T0101", "G00 X46. Z2.", "G71 P10 Q20 U0.5 W0.1 D2. F0.3",
                   "N10 G00 X20.", "G01 Z-30. F0.12", "N20 X46.", "G00 X100. Z50.", "T0202", "G00 X46. Z2.",
                   "G70 P10 Q20", "G00 X100. Z50.", "M30", "%")
    result = scan(text, "t.nc", ctx)
    lines = [(c.ref.line, c.modal.get("tool")) for c in result.commands if c.kind in (Kind.RAPID, Kind.LINEAR)]
    assert (7, 1) not in lines and (8, 1) not in lines            # T1 does not run the shape after G71
    assert (7, 2) in lines and (8, 2) in lines and (9, 2) in lines   # G70 runs it with T2
    back = [c for c in result.commands if c.ref.line == 13 and c.kind == Kind.RAPID]
    assert back and np.allclose(back[-1].tip_target_m[::2], [0.023, 0.002])       # back to the G70 start point


def test_probe_tool_entry_is_a_known_tool_that_never_cuts():
    data = setup_data()
    data["tools"]["list"].append({"number": 20, "kind": "probe", "probe": {"stylus_ball_mm": 6},
                                  "length_offset_mm": 125.0, "comment": "RENISHAW OMP40-2 SPINDLE PROBE"})
    data["operations"][0]["program_text"] = program(
        *HEAD, "T20 M06 (RENISHAW OMP40-2 SPINDLE PROBE, 6 MM RUBY BALL)", "M19", "G00 G90 X50. Y40.",
        "G43 H20 Z20.", "G01 Z2. F2000.", "G31 Z-5. F300.", "G00 Z50.", "M30", "%")
    report = check_operation(data, 0)
    ctx = report.context
    assert ctx.tools[20].is_probe and ctx.tools[20].diameter_m == pytest.approx(0.006)
    assert not [i for i in report.issues if i.severity.value in ("error", "warning")], _issues(report)
    skip = [k for k, c in enumerate(report.program.commands) if c.kind == Kind.PROBE]
    assert report.program.tip_end_W_m[skip[0]][2] == pytest.approx(0.0, abs=1e-6)  # latched on the stock top


def test_vise_jaws_stay_on_the_seat_in_a_flipped_setup():
    def flipped(data):
        setup = data["operations"][0]["setup"]
        setup["workholding"] = [{"device": {"id": "fixture-vise", "type": "vise", "jaw_width_mm": 60},
                                 "grip_depth_mm": 4}]
        setup["stock_pose"] = {"translate_mm": [-50, 40, 30], "rotate_deg": [180, 0, 0]}
    data = setup_data()
    flipped(data)
    ctx = SanityContext.from_project(data, 0)
    fixture_z = -0.430
    tops = sorted(round((b.hi[2] - fixture_z) * 1e3, 6) for b in ctx.clamps_M)
    bottoms = sorted(round((b.lo[2] - fixture_z) * 1e3, 6) for b in ctx.clamps_M)
    assert tops == [4.0, 4.0] and bottoms == [0.0, 0.0]          # on the seat, not on the turned part's top
    ys = sorted(round((b.lo[1] + 0.2432) * 1e3, 6) for b in ctx.clamps_M)
    assert ys == [-65.0, 40.0]                                    # at the part's y faces (+-40), 25 thick


def test_through_tap_counts_the_engaged_thread_a_drill_its_reach():
    data = setup_data()
    data["stock"]["size_mm"] = [100, 80, 10]
    tap = data["tools"]["list"][3]["tool"]
    tap["geometry"].update({"cutting_length_mm": 12, "flute_length_mm": 12})
    data["operations"][0]["setup"]["fixture_pose"]["translate_mm"][2] = -410.0
    data["operations"][0]["program_text"] = program(
        *HEAD, "T3 M06 (D6.8 DRILL 140DEG)", "S4000 M03", "M08", "G00 X50. Y40.", "G43 H3 Z20.",
        "G99 G81 Z-14. R2. F400.", "G80", "G00 Z20.", "T4 M06 (M8X1.25 TAP)", "S500 M03", "M08", "G00 X50. Y40.",
        "G43 H4 Z20.", "G99 G84 Z-13. R5. F625.", "G80", *TAIL)
    report = check_operation(data, 0, use_controller=False)
    flute = [i for i in report.issues if i.rule == "flute-length"]
    assert not [i for i in flute if "T4" in i.message]             # 10 mm of thread engaged, 12 of thread
    taps = [k for k, c in enumerate(report.program.commands) if c.modal.get("tool") == 4 and report.engagement.cut[k]]
    assert max(report.engagement.reach_m[taps]) > 0.012 > max(report.engagement.ap_m[taps])
