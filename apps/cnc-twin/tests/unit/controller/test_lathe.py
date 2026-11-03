import math

import pytest

from cnctwin.controller import ToolOffsetInit, hooks
from cnctwin.controller.hooks import Step
from cnctwin.controller.ir import FeedMode, Kind
from harness import LATHE, alarm, mm, moves, run, wrap

TOOLS = [ToolOffsetInit(1, x_m=-0.010, z_m=0.020, nose_radius_m=0.0008, tip=3),
         ToolOffsetInit(2, x_m=0.004, z_m=-0.005)]


def lathe(body, **kwargs):
    kwargs.setdefault("tools", TOOLS)
    kwargs.setdefault("offsets", {"G54": [0.0, -0.300]})
    return run(wrap(body), LATHE, **kwargs)


def test_t_word_indexes_the_turret_and_applies_the_offset_with_motion():
    r = lathe("T0101\nG00 X50. Z2.\nT0202\nG00 X50. Z2.\nT0200\nX50.")
    changes = [c for c in r.commands[0] if c.kind == Kind.TOOL_CHANGE]
    assert [c.tool["number"] for c in changes] == [1, 2] and changes[0].tool["offset"] == 1
    m = moves(r)
    assert mm(m[0], "X") == pytest.approx(25.0 - 5.0) and mm(m[0], "Z") == pytest.approx(-298.0 + 20.0)
    assert mm(m[1], "X") == pytest.approx(25.0 + 2.0) and mm(m[1], "Z") == pytest.approx(-298.0 - 5.0)
    assert mm(m[2], "X") == pytest.approx(25.0)


def test_css_g96_g50_clamp_and_g97():
    r = lathe("G50 S2000\nG96 S200 M03\nG00 X100. Z0\nG97 S800")
    sp = [c for c in r.commands[0] if c.kind == Kind.SPINDLE]
    assert sp[0].spindle["css_m_s"] == pytest.approx(200 / 60.0)
    assert sp[0].spindle["max_rad_s"] == pytest.approx(2000 * 2 * math.pi / 60)
    assert sp[0].spindle["omega_rad_s"] < 0 and sp[0].spindle["css_axis"] == "X"
    assert sp[-1].spindle["css_m_s"] is None
    assert sp[-1].spindle["omega_rad_s"] == pytest.approx(-800 * 2 * math.pi / 60)


def test_g98_g99_feed_modes_and_lathe_arcs():
    r = lathe("S500 M03\nG98 G01 X40. F120.\nG99 G01 Z-10. F0.15\nG02 X60. Z-20. R10.\nG03 X80. Z-30. I0 K-10.")
    m = moves(r)
    assert alarm(r) is None
    assert m[0].feed.mode == FeedMode.PER_MIN and m[1].feed.mode == FeedMode.PER_REV
    arc = m[2]
    assert arc.arc.plane == "ZX" and arc.arc.normal == (0.0, 1.0, 0.0)
    assert mm(arc, "X") == 30.0 and mm(arc, "Z") == -320.0
    assert arc.arc.sweep_rad == pytest.approx(-math.pi / 2)
    assert arc.arc.radius_start_m == pytest.approx(0.010)
    assert m[3].arc.sweep_rad == pytest.approx(math.pi / 2)


def test_g28_u0_w0_and_g50_coordinate_setting():
    m = moves(lathe("G00 X40. Z10.\nG28 U0 W0\nG50 X200. Z100.\nG00 X100. Z100."))
    assert (mm(m[1], "X"), mm(m[1], "Z")) == (20.0, -290.0)
    assert (mm(m[2], "X"), mm(m[2], "Z")) == (0.0, 0.0)
    assert (mm(m[3], "X"), mm(m[3], "Z")) == (-50.0, 0.0)


def test_g32_threading_lead_and_no_override():
    m = moves(lathe("G00 X20. Z2.\nS500 M03\nG32 Z-20. F1.5"))
    thread = m[-1]
    assert thread.feed.mode == FeedMode.PER_REV and thread.feed.value == pytest.approx(0.0015)
    assert thread.feed.override_allowed is False and thread.modal["thread"]["lead_m"] == pytest.approx(0.0015)


def test_lathe_g10_tool_offsets_and_variables():
    body = "G10 P10003 X20. Z-5. R0.4 Q3\nG10 P3 X-0.02\n#100 = #2703\n#101 = #2803\n#102 = #2003\n#103 = #2303"
    r = lathe(body)
    v = r.controller.state.paths[0].variables["c100"]
    assert v[:4] == pytest.approx([20.0, -5.0, -0.02, 3.0])


def test_lathe_cycles_route_through_the_cycle_hook():
    assert alarm(lathe("G71 U2. R0.5\nG71 P10 Q20 U0.2 W0.1 F0.25"))["code"] == "TW0001"
    assert alarm(lathe("G90 X40. Z-20. F0.2"))["code"] == "TW0001"

    class Box:
        calls = []

        def expand(self, ex, call):
            Box.calls.append(call)
            x = call.data["X"]
            z = call.data["Z"]
            start = dict(call.current)
            return [Step("rapid", {"X": x}), Step("linear", {"Z": z}, feed=call.data["F"], feed_mode=call.feed_mode),
                    Step("linear", {"X": start["X"]}, feed=call.data["F"], feed_mode=call.feed_mode),
                    Step("rapid", {"Z": start["Z"]})]

    hooks.register_cycle("G90", Box(), dialect="fanuc", replace=True)
    try:
        r = lathe("S500 M03\nG00 X50. Z2.\nG90 X40. Z-20. F0.2\nX35.")
        m = moves(r)
        assert alarm(r) is None
        assert len(Box.calls) == 2 and Box.calls[1].data["X"] == pytest.approx(0.0175)
        assert [mm(c, "X") for c in m[1:5]] == [20.0, 20.0, 25.0, 25.0]
        assert mm(m[2], "Z") == -320.0 and m[2].feed.mode == FeedMode.PER_REV
    finally:
        del hooks.CYCLES["fanuc:G90"]


def test_lathe_drilling_cycle_modal_data_and_drilling_axis():
    class Drill:
        calls = []

        def expand(self, ex, call):
            Drill.calls.append(call)
            return []

    hooks.register_cycle("G83", Drill(), dialect="fanuc", replace=True)
    try:
        r = lathe("G00 X0 Z5.\nG83 Z-20. R2. Q5. F0.1\nZ-30.\nG80")
        assert alarm(r) is None
        assert [c.extra["drilling_axis"] for c in Drill.calls] == ["Z", "Z"]
        assert Drill.calls[0].data["Z"] == pytest.approx(-0.020) and Drill.calls[1].data["Z"] == pytest.approx(-0.030)
        assert Drill.calls[1].data["Q"] == pytest.approx(0.005) and Drill.calls[0].initial_level == pytest.approx(0.005)
    finally:
        del hooks.CYCLES["fanuc:G83"]
