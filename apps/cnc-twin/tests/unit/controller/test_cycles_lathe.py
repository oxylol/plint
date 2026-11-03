"""Lathe cycles (FANUC T series system A, Haas lathe): box cycles, G70-G76, threading, through the state machine."""

import math

import pytest

from cnctwin.controller import ToolOffsetInit
from cnctwin.controller.ir import FeedMode, Kind
from harness import LATHE, alarm, run, wrap

TOOLS = [ToolOffsetInit(1, nose_radius_m=0.0008, tip=3), ToolOffsetInit(2, nose_radius_m=0.0004, tip=3)]


@pytest.fixture(autouse=True)
def _providers():
    from cnctwin.controller import comp, cycles, hooks
    hooks.load_plugins()
    cycles.register()
    comp.register()


def lathe(body, **kwargs):
    kwargs.setdefault("tools", TOOLS)
    kwargs.setdefault("offsets", {"G54": [0.0, 0.0]})
    return run(wrap(body), LATHE, **kwargs)


def path(result):
    """(kind, X diameter mm, Z mm) of every motion."""
    return [(c.kind.value, round(c.target["X"] * 2000.0, 4), round(c.target["Z"] * 1000.0, 4))
            for c in result.commands[0] if c.kind in (Kind.RAPID, Kind.LINEAR, Kind.ARC)]


def test_g90_box_cycle_taper_and_modal_repeat():
    r = lathe("T0100\nS500 M03\nG00 X50. Z2.\nG90 X40. Z-20. R-2. F0.2\nX36.\nG00 X60.")
    assert alarm(r) is None
    p = path(r)
    assert p[1:5] == [("rapid", 36.0, 2.0), ("linear", 40.0, -20.0), ("linear", 50.0, -20.0), ("rapid", 50.0, 2.0)]
    assert p[5:9] == [("rapid", 32.0, 2.0), ("linear", 36.0, -20.0), ("linear", 50.0, -20.0), ("rapid", 50.0, 2.0)]
    assert [c for c in r.commands[0] if c.kind == Kind.LINEAR][0].feed.mode == FeedMode.PER_REV


def test_g94_face_cycle_and_eob_block_repeats_the_cycle():
    r = lathe("T0100\nS500 M03\nG00 X60. Z5.\nG94 X20. Z-2. F0.15\n;\nG00 X80.")
    p = path(r)
    assert p[1:5] == [("rapid", 60.0, -2.0), ("linear", 20.0, -2.0), ("linear", 20.0, 5.0), ("rapid", 60.0, 5.0)]
    assert p[5:9] == p[1:5], "the empty (EOB-only) block repeats G94 (FANUC 4.1.6)"


def test_g92_thread_with_chamfer_and_lead_along_the_major_axis():
    r = lathe("T0100\nS500 M03\nG00 X30. Z5.\nG92 X19. Z-20. F1.5\nG00 X40.")
    threads = [c for c in r.commands[0] if c.kind == Kind.LINEAR]
    assert all(not c.feed.override_allowed and c.feed.mode == FeedMode.PER_REV for c in threads)
    assert [c.modal["thread"]["lead_major_m"] for c in threads] == pytest.approx([0.0015, 0.0015])
    assert threads[1].modal["thread"]["lead_m"] == pytest.approx(0.0015 * math.sqrt(2.0))
    assert path(r)[2:5] == [("linear", 19.0, -18.5), ("linear", 22.0, -20.0), ("rapid", 30.0, -20.0)]


def test_g32_taper_lead_is_along_the_major_axis():
    r = lathe("T0100\nS500 M03\nG00 X20. Z2.\nG32 X24. Z-18. F2.")
    thread = [c for c in r.commands[0] if c.kind == Kind.LINEAR][0]
    assert thread.modal["thread"]["major_axis"] == "Z"
    assert thread.modal["thread"]["lead_m"] == pytest.approx(0.002 * math.hypot(2.0, 20.0) / 20.0)


def test_g76_constant_area_infeed_flank_shift_and_finishing():
    r = lathe("T0100\nS500 M03\nG00 X30. Z5.\nG76 P021060 Q50 R0.02\nG76 X17.4 Z-20. P1300 Q400 F2.\nG00 X40.")
    assert alarm(r) is None
    starts = [c for c in r.commands[0] if c.kind == Kind.LINEAR and not c.modal["thread"]["chamfer"]]
    depths = [round((30.0 - (c.target["X"] * 2000.0 - 17.4) / 2.0 * 0 - 0) * 0, 0) for c in starts]
    radial = [round(1.3 - (c.target["X"] * 2000.0 - 17.4) / 2.0, 4) for c in starts]
    expected = []
    n, last = 1, 0.0
    while True:
        d = max(0.4 * math.sqrt(n), last + 0.05)
        if d >= 1.28 - 1e-12:
            expected.append(1.28)
            break
        expected.append(round(d, 4))
        last, n = d, n + 1
    assert radial == pytest.approx(expected + [1.3, 1.3], abs=1e-3) and depths
    shift = [round(c.target["Z"] * 1000.0 + 20.0 - 2.0, 4) for c in starts]
    assert shift[-1] == pytest.approx(0.0, abs=1e-6)
    assert shift[0] == pytest.approx((1.3 - 0.4) * math.tan(math.radians(30.0)), abs=1e-3)
    assert alarm(lathe("G00 X30. Z5.\nG76 P020075 Q50 R0.02"))["code"] == "PS0315"


SHAPE = """N10 G00 X20.
G01 Z-10. F0.1
X30. Z-20.
G02 X40. Z-25. R5.
G01 Z-40.
N20 X52."""


def test_g71_type1_levels_escape_and_rough_finish_pass():
    r = lathe("T0100\nS500 M03\nG00 X52. Z2.\nG71 U2. R0.5\nG71 P10 Q20 U0.4 W0.1 F0.25\n" + SHAPE + "\nG00 X100. Z50.")
    assert alarm(r) is None
    p = path(r)
    cuts = [q for q in p if q[0] == "linear"]
    levels = [q[1] for prev, q in zip(p, p[1:]) if prev[0] == "rapid" and q[0] == "linear" and prev[1] == q[1]
              and prev[2] == 2.0]
    assert levels == [48.0, 44.0, 40.0, 36.0, 32.0, 28.0, 24.0]
    assert ("linear", 48.0, -39.9) in cuts and ("linear", 49.0, -39.4) in cuts   # 45 deg escape of 0.5 mm
    tail = p[-7:]
    assert tail[0] == ("linear", 20.4, -9.9) and tail[3] == ("linear", 40.4, -39.9)
    assert tail[-1] == ("rapid", 100.0, 50.0) and tail[-2] == ("rapid", 52.0, 2.0)
    feeds = {round(c.feed.value * 1000, 4) for c in r.commands[0] if c.kind == Kind.LINEAR}
    assert feeds == {0.25}, "F of the shape blocks is ignored by G71"


def test_g70_executes_the_shape_blocks_and_returns():
    body = "T0100\nS500 M03\nG00 X52. Z2.\nG70 P10 Q20\nG00 X100. Z50.\nM30\n" + SHAPE
    r = run("%\nO0001\n" + body + "\n%\n", LATHE, tools=TOOLS, offsets={"G54": [0.0, 0.0]})
    assert alarm(r) is None
    p = path(r)
    assert p[1:7] == [("rapid", 20.0, 2.0), ("linear", 20.0, -10.0), ("linear", 30.0, -20.0), ("arc", 40.0, -25.0),
                      ("linear", 40.0, -40.0), ("linear", 52.0, -40.0)]
    assert p[7:] == [("rapid", 52.0, 2.0), ("rapid", 100.0, 50.0)]
    lin = [c for c in r.commands[0] if c.kind == Kind.LINEAR]
    assert lin[0].feed.value == pytest.approx(0.0001)


def test_g71_type2_pocket_is_cut_after_the_start_region():
    shape = """N10 G00 X30. Z0
G01 Z-10. F0.1
X20. Z-15.
Z-20.
X30. Z-25.
Z-35.
N20 X42."""
    r = lathe("T0100\nS500 M03\nG00 X42. Z2.\nG71 U2. R0.5\nG71 P10 Q20 U0 W0 F0.2\n" + shape + "\nG00 X100.")
    assert alarm(r) is None
    cuts = [q for q in path(r) if q[0] == "linear"]
    pocket = [q for q in cuts if q[1] < 30.0 - 1e-6]
    assert pocket and min(q[1] for q in pocket) >= 20.0 - 1e-6
    first_pocket = next(i for i, q in enumerate(cuts) if q[1] < 30.0 - 1e-6)
    assert all(q[1] >= 30.0 - 1e-6 for q in cuts[:first_pocket])


def test_g72_facing_and_g73_pattern_repeating():
    shape = "N10 G00 Z-12.\nG01 X30. F0.1\nZ-5.\nX20. Z0\nN20 Z2."
    r = lathe("T0100\nS500 M03\nG00 X42. Z2.\nG72 W2. R0.5\nG72 P10 Q20 U0.2 W0.1 F0.2\n" + shape)
    assert alarm(r) is None
    infeeds = sorted({q[2] for q in path(r) if q[0] == "rapid" and q[1] == 42.0})
    assert -10.0 in infeeds and -2.0 in infeeds
    r = lathe("T0100\nS500 M03\nG00 X50. Z2.\nG73 U3. W1. R3\nG73 P10 Q20 U0.4 W0.1 F0.2\n"
              "N10 G00 X20. Z0\nG01 Z-20. F0.1\nN20 X50.")
    starts = [q for q in path(r) if q[0] == "rapid" and q[2] <= 1.1 and q[1] < 50.0]
    assert [q[1:] for q in starts] == [(26.4, 1.1), (23.4, 0.6), (20.4, 0.1)]


def test_g74_peck_drilling_and_g75_grooving():
    r = lathe("T0100\nS500 M03\nG00 X0 Z2.\nG74 R1.\nG74 Z-10. Q4000 F0.1")
    zs = [(q[0], q[2]) for q in path(r)[1:]]
    assert zs == [("linear", -2.0), ("rapid", -1.0), ("linear", -6.0), ("rapid", -5.0), ("linear", -10.0),
                  ("rapid", 2.0)]
    r = lathe("T0100\nS500 M03\nG00 X42. Z-10.\nG75 R0.5\nG75 X30. Z-14. P3000 Q2000 F0.05")
    p = path(r)
    bottoms = [q for q in p if q[0] == "linear" and q[1] == 30.0]
    assert sorted({q[2] for q in bottoms}) == [-14.0, -12.0, -10.0]


def test_g90_with_tool_nose_radius_compensation_offsets_the_cut():
    r = lathe("T0101\nS500 M03\nG00 X50. Z2.\nG42\nG90 X40. Z-20. F0.2\nG40 G00 X60.")
    assert alarm(r) is None
    cut = [c for c in r.commands[0] if c.kind == Kind.LINEAR][0]
    assert round(cut.target["X"] * 2000.0, 4) == pytest.approx(40.0, abs=1e-4)


def test_css_parameters_reach_the_spindle_command_in_a_cycle():
    r = lathe("T0100\nG50 S1800\nG96 S180 M03\nG00 X52. Z2.\nG71 U2. R0.5\nG71 P10 Q20 U0.4 W0.1 F0.25\n" + SHAPE)
    spindle = [c for c in r.commands[0] if c.kind == Kind.SPINDLE][0]
    assert spindle.spindle["css_m_s"] == pytest.approx(3.0) and spindle.spindle["css_axis"] == "X"
    assert spindle.spindle["max_rad_s"] == pytest.approx(1800 * 2 * math.pi / 60)


def test_mrc_alarms():
    assert alarm(lathe("G00 X52. Z2.\nG71 P10 Q99 U0.4 W0.1 F0.25\nN10 G00 X20.\nG01 Z-10.\nN20 X52."))["code"] == "PS0063"
    assert alarm(lathe("G00 X52. Z2.\nG71 P10 Q20 U0.4 F0.25\nN10 X20.\nG01 Z-10.\nN20 X52."))["code"] == "PS0065"
    assert alarm(lathe("G00 X52. Z2.\nG71 P10 Q20 F0.25\nN10 G00 X20.\nG01 Z-10.\nX30. Z-5.\nN20 X52."))["code"] == "PS0064"
    assert alarm(lathe("G00 X52. Z2.\nG71 P10 Q20 F0.25\nN10 G00 X20.\nG01 Z-10.\nX15. Z-15.\nN20 X52."))["code"] == "PS0329"
