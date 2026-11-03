import math

import pytest

from cnctwin.controller.ir import FeedMode, Kind, Space
from harness import LATHE, alarm, mm, moves, run, wrap, xyz


def test_rapid_and_linear_targets_in_machine_coordinates():
    r = run(wrap("G90 G54 G21\nG00 X10. Y20.\nG01 X15. F600."), offsets={"G54": [-0.4, -0.2, -0.1]})
    m = moves(r)
    assert alarm(r) is None
    assert m[0].kind == Kind.RAPID and xyz(m[0]) == (-390.0, -180.0, 0.0)
    assert m[1].kind == Kind.LINEAR and xyz(m[1]) == (-385.0, -180.0, 0.0)
    assert m[1].feed.mode == FeedMode.PER_MIN and m[1].feed.value == pytest.approx(0.01)
    assert m[0].feed.mode == FeedMode.RAPID and m[0].space == Space.MACHINE
    assert m[1].ref.line == 5 and m[1].ref.block == "G01 X15. F600."


def test_decimal_point_rule_and_pocket_calculator():
    from cnctwin.controller.schema import ControllerSettings
    r = run(wrap("G90 G00 X10 Y10.\nX1000"))
    m = moves(r)
    assert xyz(m[0])[:2] == (0.01, 10.0)
    assert xyz(m[1])[0] == 1.0
    r = run(wrap("G90 G00 X10 Y10."), settings=ControllerSettings(pocket_calculator=True))
    assert xyz(moves(r)[0])[:2] == (10.0, 10.0)


def test_least_input_increment_rounding_mm_and_inch():
    r = run(wrap("G90 G21 G00 X1.23456 Y-1.23449\nG20 G00 X1.00005"))
    m = moves(r)
    assert xyz(m[0])[:2] == (1.235, -1.234)
    assert mm(m[1], "X") == 25.403          # 1.0001 in (rounded at 0.0001 in) = 25.40254 mm -> 0.001 mm


def test_incremental_moves_are_exact_in_counts():
    body = "G91 G01 F100.\n" + "X0.001\n" * 1000
    m = moves(run(wrap(body)))
    assert m[-1].target["X"] == pytest.approx(0.001, abs=0.0)
    assert round(m[-1].target["X"] / 1e-6) == 1000


def test_arc_r_minor_and_major():
    m = moves(run(wrap("G90 G17 G00 X0 Y0\nG02 X10. Y0 R5. F100.\nG03 X0 Y0 R-5.")))
    cw, ccw = m[1], m[2]
    assert cw.kind == Kind.ARC and cw.arc.sweep_rad == pytest.approx(-math.pi)
    assert cw.arc.centre[0] == pytest.approx(0.005) and cw.arc.centre[1] == pytest.approx(0.0, abs=1e-12)
    assert ccw.arc.sweep_rad == pytest.approx(math.pi)
    m = moves(run(wrap("G90 G00 X0 Y0\nG02 X10. Y10. R10. F100.\nG00 X0 Y0\nG02 X10. Y10. R-10.")))
    minor, major = m[1], m[3]
    assert minor.arc.sweep_rad == pytest.approx(-math.pi / 2)
    assert minor.arc.centre[:2] == pytest.approx((0.01, 0.0))
    assert major.arc.sweep_rad == pytest.approx(-3 * math.pi / 2)
    assert major.arc.centre[:2] == pytest.approx((0.0, 0.01))
    assert major.arc.radius_start_m == pytest.approx(0.01) and major.arc.radius_end_m == pytest.approx(0.01)


def test_arc_ijk_full_circle_and_quarter():
    m = moves(run(wrap("G90 G00 X10. Y0\nG03 I-10. F200.\nG02 X0 Y10. I-10. J0")))
    full, quarter = m[1], m[2]
    assert full.arc.sweep_rad == pytest.approx(2 * math.pi)
    assert full.arc.centre[:2] == pytest.approx((0.0, 0.0), abs=1e-12)
    assert xyz(full)[:2] == (10.0, 0.0)
    assert quarter.arc.sweep_rad == pytest.approx(-1.5 * math.pi)
    assert quarter.arc.normal == (0.0, 0.0, 1.0)


def test_helical_arc_moves_the_third_axis():
    m = moves(run(wrap("G90 G00 X10. Y0 Z0\nG03 X10. Y0 Z-2. I-10. J0 F100.")))
    helix = m[1]
    assert helix.arc.sweep_rad == pytest.approx(2 * math.pi)
    assert mm(helix, "Z") == -2.0
    assert helix.arc.centre[2] == pytest.approx(0.0)


def test_arc_planes_g18_g19():
    m = moves(run(wrap("G90 G18 G00 X0 Z0\nG02 X10. Z0 R5. F100.\nG19 G00 Y0 Z0\nG03 Y0 Z10. K5. F100.")))
    g18, g19 = m[1], m[3]
    assert g18.arc.plane == "ZX" and g18.arc.normal == (0.0, 1.0, 0.0)
    assert g18.arc.sweep_rad == pytest.approx(-math.pi)
    assert g19.arc.plane == "YZ" and g19.arc.normal == (1.0, 0.0, 0.0)
    assert g19.arc.sweep_rad == pytest.approx(math.pi)


def test_arc_end_point_radius_check():
    r = run(wrap("G90 G00 X0 Y0\nG02 X10. Y0 I5.1 J0 F100."))
    assert alarm(r)["code"] == "PS0020" and alarm(r)["text"] == "OVER TOLERANCE OF RADIUS"
    ok = run(wrap("G90 G00 X0 Y0\nG02 X10. Y0 I5.01 J0 F100."))
    arc = moves(ok)[1]
    assert alarm(ok) is None
    assert arc.arc.radius_start_m == pytest.approx(0.00501) and arc.arc.radius_end_m == pytest.approx(0.00499)
    r = run(wrap("G90 G00 X0 Y0\nG02 X10. Y0 R4. F100."))
    assert alarm(r)["code"] == "PS0020"


def test_arc_without_radius_and_feed_zero():
    r = run(wrap("G90 G00 X0 Y0\nG02 X10. Y0 F100."))
    assert alarm(r)["code"] == "PS0022"
    r = run(wrap("G90 G00 X0 Y0\nG01 X10."))
    assert alarm(r)["code"] == "PS0011" and alarm(r)["text"] == "FEED ZERO ( COMMAND )"


def test_feed_modes_and_inverse_time():
    m = moves(run(wrap("S1000 M03\nG95 G01 X10. F0.2\nG93 G01 X20. F2.\nG94 G01 X30. F500.")))
    assert m[0].feed.mode == FeedMode.PER_REV and m[0].feed.value == pytest.approx(0.0002)
    assert m[1].feed.mode == FeedMode.INVERSE_TIME and m[1].feed.value == pytest.approx(2.0 / 60.0)
    assert m[2].feed.mode == FeedMode.PER_MIN
    r = run(wrap("G93 G01 X10. F2.\nX20."))
    assert alarm(r)["code"] == "PS0011"
    r = run(wrap("G94 G01 X10. F100.\nG95\nG01 X20."))
    assert alarm(r)["code"] == "PS0011"


def test_feed_clamped_to_machine_maximum_with_warning():
    r = run(wrap("G01 X10. F50000."))
    assert moves(r)[0].feed.value == pytest.approx(20000 / 60000)
    assert any("clamped" in e.get("message", "") for e in r.events[0])


def test_dwell_p_and_x():
    r = run(wrap("G04 P1500\nG04 X2.5\nG04 X1000"))
    dwells = [c for c in r.commands[0] if c.kind == Kind.DWELL]
    assert [d.dwell_s for d in dwells] == pytest.approx([1.5, 2.5, 1.0])
    assert alarm(run(wrap("G04 P1.5")))["code"] == "PS0007"


def test_spindle_sign_rule_and_coolant():
    r = run(wrap("S1000 M03\nM04\nM05\nM08\nM09"))
    sp = [c for c in r.commands[0] if c.kind == Kind.SPINDLE]
    assert sp[0].spindle["omega_rad_s"] == pytest.approx(-1000 * 2 * math.pi / 60)
    assert sp[1].spindle["omega_rad_s"] == pytest.approx(1000 * 2 * math.pi / 60)
    assert sp[2].spindle["omega_rad_s"] == 0.0
    co = [c for c in r.commands[0] if c.kind == Kind.COOLANT]
    assert co[0].coolant["pumps"]["flood"] is True and co[1].coolant["pumps"]["flood"] is False


def test_spindle_clamped_to_machine_maximum():
    r = run(wrap("S20000 M03"))
    sp = [c for c in r.commands[0] if c.kind == Kind.SPINDLE][0]
    assert sp.spindle["omega_rad_s"] == pytest.approx(-12000 * 2 * math.pi / 60)


def test_tool_change_and_preselect():
    r = run(wrap("T5\nM06\nT7 M06"))
    kinds = [c.kind for c in r.commands[0]]
    assert kinds.count(Kind.TOOL_CHANGE) == 2
    changes = [c for c in r.commands[0] if c.kind == Kind.TOOL_CHANGE]
    assert changes[0].tool["number"] == 5 and changes[1].tool["number"] == 7
    assert alarm(run(wrap("T30 M06")))["code"] == "TW0013"


def test_optional_stop_program_stop_and_end():
    r = run(wrap("M00\nM01\nG00 X1."))
    stops = [c.stop for c in r.commands[0] if c.kind == Kind.STOP]
    assert stops == ["M00", "M30"]


def test_exact_stop_modes():
    m = moves(run(wrap("G61 G01 X1. F100.\nG64 X2.\nG09 X3.\nX4.")))
    assert [c.smoothing.corner for c in m] == ["exact_stop", "blend", "exact_stop", "blend"]


def test_g53_machine_coordinates_and_rapid():
    m = moves(run(wrap("G90 G00 X10. Y10.\nG53 Z-50.\nG91 G53 Z0"), offsets={"G54": [-0.4, -0.2, -0.1]}))
    assert xyz(m[1]) == (-390.0, -190.0, -50.0) and m[1].kind == Kind.RAPID
    assert len(m) == 2


def test_lathe_diameter_programming_and_incremental_addresses():
    r = run(wrap("S500 M03\nG00 X50. Z2.\nG01 U-10. W-5. F0.2\nX20."), LATHE, offsets={"G54": [0.0, -0.3]})
    m = moves(r)
    assert alarm(r) is None
    assert mm(m[0], "X") == 25.0 and mm(m[0], "Z") == -298.0
    assert mm(m[1], "X") == 20.0 and mm(m[1], "Z") == -303.0
    assert m[1].feed.mode == FeedMode.PER_REV and m[1].feed.value == pytest.approx(0.0002)
    assert mm(m[2], "X") == 10.0
