"""Haas cycle coverage (lathe D words, face/radial drilling, G77 back bore, G70-G72 bolt patterns, modal
I/J/K pecks), the dry run's feed semantics (G234 tip feed, constant surface speed, probe latch time) and the
machine model's tolerance of source keys in the axes mapping."""

import copy
from types import SimpleNamespace
import math

import pytest

from cnctwin.controller import ToolOffsetInit, dry_run
from cnctwin.controller.dryrun import BoxProbe
from cnctwin.controller.ir import Kind, Space
from cnctwin.controller.machine import build_machine
from harness import HAAS_MILL, LATHE, UMC, alarm, controller, moves, run, wrap

HAAS_LATHE = copy.deepcopy(LATHE)
HAAS_LATHE["controller"]["dialect"] = "haas"
LATHE_TOOL = [ToolOffsetInit(1, x_m=0.0, z_m=0.0, nose_radius_m=0.0)]


def _xz(cmd):
    return round(cmd.target["X"] * 2e3, 4), round(cmd.target["Z"] * 1e3, 4)     # X as a diameter (mm)


# ------------------------------------------------------------------ controller fixes

def test_haas_arcs_take_the_corner_argument():
    r = run(wrap("G90 G54 G00 X0 Y0\nG01 X20. F500. ,R2.\nG01 Y-20.\nG02 X30. Y-30. R10.\nG03 X40. Y-20. I0 J10."), HAAS_MILL)
    assert alarm(r) is None, alarm(r)
    kinds = [c.kind for c in moves(r)]
    assert kinds.count(Kind.ARC) == 3                       # the two programmed arcs and the ,R corner
    assert alarm(run(wrap("G90 G54 G00 X0 Y0\nG01 X20. F500. ,R2.\nG02 X30. Y-10. R10."), HAAS_MILL))["code"] \
        == "TW0031"                                         # Haas: ,C/,R only between two G01 blocks


def test_haas_lathe_g71_d_is_the_depth_of_cut():
    program = wrap("""T0101
G50 S2500
G97 S800 M03
G00 X46. Z2.
G71 P10 Q20 U0. W0. D2.5 F0.3
N10 G00 X20.
G01 Z0. F0.12
Z-30.
N20 X46.
G00 X100. Z50.""")
    r = run(program, HAAS_LATHE, tools=LATHE_TOOL)
    assert alarm(r) is None, alarm(r)
    cuts = [_xz(c)[0] for c in moves(r) if c.kind == Kind.LINEAR and abs(c.target["Z"] + 0.03) < 1e-9]
    # the roughing passes reach Z-30 at diameters 46 - 2 x 2.5 k (D is the depth of cut, a radius), then the
    # rough pass along the figure (X20) and its shoulder
    assert cuts == [41.0, 36.0, 31.0, 26.0, 21.0, 20.0, 46.0]


def test_haas_lathe_g76_d_first_pass_and_g73_d_passes_are_cycle_data():
    thread = wrap("""T0101
G97 S1000 M03
G00 X26. Z6.
G76 X22.058 Z-21.5 K0.971 D0.300 F1.5 A60 P1
G00 X30.""")
    r = run(thread, HAAS_LATHE, tools=LATHE_TOOL)
    assert alarm(r) is None, alarm(r)
    passes = [c for c in moves(r) if c.kind == Kind.LINEAR and c.feed is not None and c.feed.mode.value == "per_rev"]
    # crest = root X22.058 + 2 K0.971 = 24.0; the first pass cuts D0.3 (radius) below it
    assert _xz(passes[0])[0] == pytest.approx(24.0 - 2 * 0.3, abs=1e-3)
    repeat = run(thread.replace("G76 X22.058 Z-21.5 K0.971 D0.300 F1.5 A60 P1",
                                "G73 P10 Q20 U0.2 W0.1 I3. K0. D3 F0.2\nN10 G01 X20. Z0.\nN20 Z-10."),
                 HAAS_LATHE, tools=LATHE_TOOL)
    assert alarm(repeat) is None, alarm(repeat)                        # G73 D3: three passes, no offset alarm


def test_haas_lathe_face_and_radial_drilling_take_absolute_r():
    program = wrap("""T0101
G98
G97 S1200 M03
G00 X0. Z5.
G81 Z-12. R1. F100.
G80
G00 X50. Z-10.
G241 X20. Z-10. R44. F100.
G80
G00 X100.""")
    r = run(program, HAAS_LATHE, tools=LATHE_TOOL)
    assert alarm(r) is None, alarm(r)
    m = moves(r)
    z_moves = [_xz(c) for c in m if c.kind in (Kind.RAPID, Kind.LINEAR)]
    assert (0.0, 1.0) in z_moves and (0.0, -12.0) in z_moves          # R1. and Z-12. are positions
    assert (44.0, -10.0) in z_moves and (20.0, -10.0) in z_moves      # R44. and X20. are diameters
    assert z_moves[z_moves.index((20.0, -10.0)) + 1] == (50.0, -10.0)  # back to the starting plane


def test_haas_g77_back_bore_shifts_enters_below_and_returns_to_the_initial_level():
    program = wrap("""T1 M06
G90 G54 G00 X0 Y0 S1200 M03
G43 H1 Z2.5
G77 Z-12.7 R-40.6 Q2.54 F254.
G80""")
    r = run(program, HAAS_MILL)
    assert alarm(r) is None, alarm(r)
    m = [c for c in moves(r) if c.ref.line == 6]
    xs = [round(c.target["X"] * 1e3, 3) for c in m]
    zs = [round((c.target["Z"] - 0.1) * 1e3, 3) for c in m]          # program Z (H1 = 100 mm)
    assert 2.54 in xs and 0.0 in xs                                   # shifted by Q along Setting 27 (+X), back
    feed = [c for c in m if c.kind == Kind.LINEAR]
    assert len(feed) == 1 and zs[m.index(feed[0])] == -12.7           # feeds up from R to Z
    assert zs[m.index(feed[0]) - 2] == -40.6                          # entered at R below the part, shifted
    assert zs[-2] == 2.5 or zs[-1] == 2.5                             # returns to the initial level


def test_haas_bolt_hole_circle_arc_and_line():
    circle = wrap("""T1 M06
G90 G54 G00 X0 Y0 S1000 M03
G43 H1 Z2.5
G81 G98 Z-10. R1. F200. L0
G70 I50. J15. L12
G80""")
    r = run(circle, HAAS_MILL)
    assert alarm(r) is None, alarm(r)
    feeds = [c for c in moves(r) if c.kind == Kind.LINEAR]
    assert len(feeds) == 12
    for k, c in enumerate(feeds):
        angle = math.radians(15.0 + 30.0 * k)
        assert c.target["X"] == pytest.approx(0.05 * math.cos(angle), abs=1.5e-6)
        assert c.target["Y"] == pytest.approx(0.05 * math.sin(angle), abs=1.5e-6)
    arc = run(circle.replace("G70 I50. J15. L12", "G71 I50. J90. K-45. L3"), HAAS_MILL)
    pts = [(round(c.target["X"] * 1e3, 3), round(c.target["Y"] * 1e3, 3)) for c in moves(arc) if c.kind == Kind.LINEAR]
    assert pts == [(0.0, 50.0), (35.355, 35.355), (50.0, 0.0)]
    line = run(circle.replace("G70 I50. J15. L12", "G72 I10. J0. L3"), HAAS_MILL)
    pts = [round(c.target["X"] * 1e3, 3) for c in moves(line) if c.kind == Kind.LINEAR]
    assert pts == [0.0, 10.0, 20.0]                                   # haas_g72_first_hole_at_start (est)
    bare = run(wrap("G90 G54 G00 X0 Y0 Z10.\nG70 I50. L4"), HAAS_MILL)
    assert alarm(bare)["code"] == "TW0037"


def test_haas_peck_schedule_is_modal_for_the_next_holes():
    program = wrap("""T1 M06
G90 G54 G00 X0 Y0 S1000 M03
G43 H1 Z5.
G83 G99 Z-10. R1. I4. J1. K2. F200.
X20.
G80""")
    r = run(program, HAAS_MILL)
    assert alarm(r) is None, alarm(r)
    for line in (6, 7):
        depths = [round((c.target["Z"] - 0.1) * 1e3, 3) for c in moves(r) if c.ref.line == line
                  and c.kind == Kind.LINEAR]
        assert depths == [-3.0, -6.0, -8.0, -10.0]                    # 4, 3, then the 2 mm minimum


def test_axes_mapping_with_source_keys():
    machine = copy.deepcopy(HAAS_MILL)
    machine["axes"]["src"] = "maker-brochure"
    machine["axes"]["note"] = "travels as published"
    model = build_machine(machine)
    assert [a.name for a in model.paths[0]] == ["X", "Y", "Z"]


# ------------------------------------------------------------------ dry run feed semantics

def test_g234_feed_is_the_tool_tip_speed():
    program = wrap("""T1 M06
G90 G54 G00 X0 Y0 B0 C0 S1000 M03
G43 H1 Z50.
G234 H1
G01 X10. Y0 Z50. B30. C45. F600.
G01 X10. Y10. Z50. B0 C0 F600.
G49""")
    ctl = controller(program, UMC, offsets={"G54": [-0.3, -0.25, -0.4, 0.0, 0.0]})
    r = dry_run(ctl)
    assert alarm(r) is None, alarm(r)
    tip = [(k, c) for k, c in enumerate(r.commands[0]) if c.kind == Kind.LINEAR and c.space == Space.TIP]
    assert len(tip) == 2
    for k, c in tip:
        seconds = (r.end_ticks[0][k] - r.start_ticks[0][k]) * 1e-6
        assert seconds == pytest.approx(0.010 / (600e-3 / 60.0), rel=1e-3)   # 10 mm of tip path at F600


def test_constant_surface_speed_per_revolution_time():
    program = wrap("""T0101
G50 S3000
G96 S120 M03
G00 X60. Z0.
G01 X10. F0.2
G00 X60. Z5.""")
    r = run(program, HAAS_LATHE, tools=LATHE_TOOL)
    assert alarm(r) is None, alarm(r)
    k, c = next((k, c) for k, c in enumerate(r.commands[0]) if c.kind == Kind.LINEAR)
    seconds = (r.end_ticks[0][k] - r.start_ticks[0][k]) * 1e-6
    v, f, n_max = 120.0 / 60.0, 0.2e-3, 3000 * 2 * math.pi / 60.0
    r_c = v / n_max                                                  # 6.37 mm: inside it the clamp holds
    expected = 2 * math.pi / f * ((0.030 ** 2 - r_c ** 2) / (2 * v) + (r_c - 0.005) / n_max)
    assert r_c > 0.005 and seconds == pytest.approx(expected, rel=1e-4)


def test_box_probe_triggers_at_the_stock_face_and_cuts_the_move():
    program = wrap("""T1 M06
G90 G54 G00 X0 Y0
G43 H1 Z10.
G31 Z-10. F100.
#100 = #5063""")
    box = [((-0.5, -0.4, -0.5), (0.0, 0.0, -0.2))]                   # stock top at machine Z -200
    probe = BoxProbe(box, radius_m=0.003)
    ctl = controller(program, HAAS_MILL, offsets={"G54": [-0.1, -0.1, -0.2]})
    r = dry_run(ctl, probe=probe)
    assert alarm(r) is None, alarm(r)
    k, c = next((k, c) for k, c in enumerate(r.commands[0]) if c.kind == Kind.PROBE)
    assert ctl.executors[0].read(100) == pytest.approx(0.0, abs=1e-6)  # the ball's bottom touched Z0 (work)
    seconds = (r.end_ticks[0][k] - r.start_ticks[0][k]) * 1e-6
    assert seconds == pytest.approx(0.010 / (100e-3 / 60.0), rel=1e-3)  # 10 of the 20 mm at F100


def test_a_probe_t_number_is_a_listed_tool():
    tools = [ToolOffsetInit(1, length_m=0.1, radius_m=0.005),
             ToolOffsetInit(20, length_m=0.125, radius_m=0.002999, probe="spindle_probe")]

    class Table:                                            # a core ToolTable holding the cutting tools only
        def by_number(self, number, path=0):
            return SimpleNamespace(slot=0) if number == 1 else None

    program = wrap("T20 M06\nG90 G54 G00 X0 Y0\nG43 H20 Z10.\nG31 Z-10. F100.\nT1 M06")
    ctl = controller(program, HAAS_MILL, tools=tools)
    ctl.executors[0].ctx.tool_table = Table()
    r = dry_run(ctl)
    changes = [c for c in r.commands[0] if c.kind == Kind.TOOL_CHANGE]
    assert changes[0].tool["probe"] == "spindle_probe" and changes[0].tool["slot"] == -1
    assert "probe" not in changes[1].tool and changes[1].tool["slot"] == 0
    probe = next(c for c in r.commands[0] if c.kind == Kind.PROBE)
    assert probe.probe["stylus"] == "spindle_probe"
    assert probe.modal["length_offset_m"] == pytest.approx(0.125)
    assert not [e for e in r.events[0] if e.get("kind") == "warning" and "T20" in e.get("message", "")]


def test_block_log_records_the_evaluated_words():
    program = wrap("#1 = 1\nT1 M06\nG90 G54 G00 X10. Y5.\nG43 H#1 Z[#1 + 9.]\nG01 Z0. F#1")
    ctl = controller(program, HAAS_MILL)
    ctl.executors[0].block_log = []
    r = dry_run(ctl)
    assert alarm(r) is None, alarm(r)
    log = {rec["line"]: rec for rec in ctl.executors[0].block_log}
    assert log[3]["words"] == {} and log[6]["words"]["H"] == 1.0 and log[6]["words"]["Z"] == 10.0
    assert log[7]["words"]["F"] == 1.0 and log[6]["depth"] == 1
    assert [rec["seq"] for rec in ctl.executors[0].block_log] == sorted(rec["seq"] for rec in ctl.executors[0].block_log)


def test_project_work_offset_names_map_to_the_control_keys():
    from cnctwin.controller.domain import offset_key

    assert [offset_key(k) for k in ("G54", "g55", "G54.1 P3", "G54.1P3", "G154 P12", "G110", "G129", "PRESET 1",
                                    "P7", "EXT")] == ["G54", "G55", "P3", "P3", "P12", "P1", "P20", "P1", "P7", "EXT"]
