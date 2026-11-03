"""SINUMERIK 840D sl technology cycles: drilling, hole patterns, pockets, slots, swivel, high-speed settings,
TRAORI and the turning cycles (expected command streams)."""

from __future__ import annotations

import math

import pytest

from cnctwin.controller import ToolOffsetInit
from cnctwin.controller.ir import Kind, Space
from harness import LATHE, UMC
from test_siemens import arcs, head, path, run

DRILL = "S1000 M3 F100\nG0 X0 Y0 Z20\n"


def zs(res) -> list[tuple]:
    return [(c.kind.value, round(c.target["Z"] * 1000.0, 6)) for c in res.commands[0]
            if c.kind in (Kind.RAPID, Kind.LINEAR)][1:]


def test_cycle81_82_85_89_sequences():
    res = run(head(DRILL + "CYCLE81(10, 0, 2, -5)\nCYCLE82(10, 0, 2, , 5, 0.5)\nCYCLE85(10, 0, 2, -5, , 0, 200, 400)\n"
                           "CYCLE89(10, 0, 2, -5, , 1)"))
    assert res.alarms[0] is None
    assert zs(res) == [("rapid", 102.0), ("linear", 95.0), ("rapid", 110.0),
                       ("rapid", 102.0), ("linear", 95.0), ("rapid", 110.0),
                       ("rapid", 102.0), ("linear", 95.0), ("linear", 102.0), ("rapid", 110.0),
                       ("rapid", 102.0), ("linear", 95.0), ("linear", 102.0), ("rapid", 110.0)]
    dwells = [c.dwell_s for c in res.commands[0] if c.kind == Kind.DWELL]
    assert dwells == [0.5, 1.0]
    feeds = [round(c.feed.value * 60000, 6) for c in res.commands[0] if c.kind == Kind.LINEAR]
    assert feeds == [100.0, 100.0, 200.0, 400.0, 100.0, 100.0]


def test_cycle83_chip_breaking_and_chip_removal():
    breaking = run(head(DRILL + "CYCLE83(10, 0, 2, -12, , -5, , 2, 0, 0, 1, 0)"))
    assert breaking.alarms[0] is None
    assert zs(breaking) == [("rapid", 102.0), ("linear", 95.0), ("linear", 96.0), ("linear", 92.0), ("linear", 93.0),
                            ("linear", 90.0), ("linear", 91.0), ("linear", 88.0), ("rapid", 110.0)]
    removal = run(head(DRILL + "CYCLE83(10, 0, 2, -9, , -5, , 0, 0, 0, 1, 1)"))
    assert zs(removal) == [("rapid", 102.0), ("linear", 95.0), ("rapid", 102.0), ("rapid", 95.6), ("linear", 91.0),
                           ("rapid", 110.0)]


def test_cycle84_rigid_tapping_orients_taps_and_restores():
    res = run(head(DRILL + "CYCLE84(10, 0, 2, -10, , 0, 3, , 1.5, 30, 500, 800)"))
    assert res.alarms[0] is None
    spindle = [c.spindle for c in res.commands[0] if c.kind == Kind.SPINDLE][1:]
    assert spindle[0]["orient_rad"] == pytest.approx(math.radians(30))
    assert spindle[1]["omega_rad_s"] == pytest.approx(-500 * 2 * math.pi / 60)
    assert spindle[2]["omega_rad_s"] == pytest.approx(-1000 * 2 * math.pi / 60)
    taps = [c for c in res.commands[0] if c.kind == Kind.LINEAR]
    assert [c.modal["tap"]["phase"] for c in taps] == ["in", "out"]
    assert taps[0].feed.mode.value == "per_rev" and taps[0].feed.value == pytest.approx(0.0015)
    assert not taps[0].feed.override_allowed and taps[1].modal["tap"]["rpm"] == 800


def test_cycle840_floating_tap_and_metric_size():
    res = run(head(DRILL + "CYCLE840(10, 0, 2, -10, , 0, 0, 3, 0, 8)"))
    assert res.alarms[0] is None
    taps = [c for c in res.commands[0] if c.kind == Kind.LINEAR]
    assert taps[0].feed.value == pytest.approx(0.00125) and taps[0].modal["tap"]["floating"]
    reverse = [c.spindle["direction"] for c in res.commands[0] if c.kind == Kind.SPINDLE]
    assert reverse == [-1, 1, -1]


def test_cycle86_87_88_boring():
    res = run(head(DRILL + "CYCLE86(10, 0, 2, -10, , 0, 3, 0.5, 0.2, 0, 90)"))
    assert res.alarms[0] is None
    assert path(res)[-3:] == [("rapid", 0.5, 0.2, 90.0), ("rapid", 0.5, 0.2, 110.0), ("rapid", 0.0, 0.0, 110.0)]
    stop = run(head(DRILL + "CYCLE87(10, 0, 2, -10, , 3)"))
    assert any(c.kind == Kind.STOP and c.stop == "M0" for c in stop.commands[0])
    assert zs(stop)[-1] == ("rapid", 110.0)
    dwell = run(head(DRILL + "CYCLE88(10, 0, 2, -10, , 2, 4)"))
    assert [c.dwell_s for c in dwell.commands[0] if c.kind == Kind.DWELL] == [2.0]
    assert [c.spindle["direction"] for c in dwell.commands[0] if c.kind == Kind.SPINDLE][-1] == 1


def test_drilling_errors_reference_plane_and_feed():
    assert run(head(DRILL + "CYCLE81(0, 10, 2, -5)")).alarms[0]["code"] == "61101"
    assert run(head("S1000 M3\nG0 X0 Y0 Z20\nCYCLE81(10, 0, 2, -5)")).alarms[0]["code"] == "10860"
    assert run(head(DRILL + "CYCLE83(10, 0, 2, -12, , , , 2)")).alarms[0]["code"] == "61107"


def test_holes1_row_and_holes2_circle_with_the_modal_cycle():
    res = run(head(DRILL + "MCALL CYCLE81(10, 0, 2, -5)\nHOLES1(0, 0, 0, 10, 5, 3)\nMCALL\nG0 X0 Y0\n"
                           "MCALL CYCLE81(10, 0, 2, -5)\nHOLES2(0, 0, 20, 45, 90, 2)\nMCALL"))
    assert res.alarms[0] is None
    holes = [c for c in res.commands[0] if c.kind == Kind.LINEAR]
    xy = [(round(c.target["X"] * 1000, 6), round(c.target["Y"] * 1000, 6)) for c in holes]
    s = round(20 * math.sqrt(0.5), 3)            # positions are whole counts of 1 um
    assert xy == [(10.0, 0.0), (15.0, 0.0), (20.0, 0.0), (s, s), (-s, s)]


def test_pocket3_roughing_layers_and_contour_parallel_passes():
    res = run(head("S1000 M3\nG0 X0 Y0 Z20\nPOCKET3(10, 0, 2, -6, 40, 20, 5, 50, 50, 0, 3, 0.5, 0.2, 400, 100, 0, 11, 6)"))
    assert res.alarms[0] is None
    moves = path(res)
    assert moves[1:4] == [("rapid", 0.0, 0.0, 110.0), ("rapid", 50.0, 50.0, 110.0), ("rapid", 50.0, 50.0, 102.0)]
    layer = [m for m in moves if m[0] == "linear" and m[3] == 97.1]
    assert layer[:4] == [("linear", 50.0, 50.0, 97.1), ("linear", 60.0, 50.0, 97.1), ("linear", 40.0, 50.0, 97.1),
                         ("linear", 60.0, 50.0, 97.1)]
    outer = {(m[1], m[2]) for m in layer}
    assert {(64.5, 54.5), (35.5, 54.5), (35.5, 45.5), (64.5, 45.5)} <= outer
    assert moves[-1] == ("rapid", 50.0, 50.0, 110.0)
    assert {m[3] for m in moves if m[0] == "linear"} == {97.1, 94.2}


def test_pocket3_finishing_and_corner_dimensioning():
    res = run(head("S1000 M3\nG0 X0 Y0 Z20\nPOCKET3(10, 0, 2, -6, -40, -20, 5, 50, 50, 0, 6, 0, 0, 400, 100, 0, 2, 6)"))
    assert res.alarms[0] is None
    loop = [m for m in path(res) if m[0] == "linear" and m[3] == 94.0]
    xs = {m[1] for m in loop}
    assert max(xs) == 25.0 and min(xs) == -5.0 + 20.0 - 5.0 + 0.0 or min(xs) == 15.0


def test_pocket4_circular_pocket_with_helical_insertion():
    res = run(head("S1000 M3\nG0 X0 Y0 Z20\nPOCKET4(10, 0, 2, -4, 20, 30, 30, 4, 0, 0, 400, 100, 0, 21, 4, 0, 0, 3, 2)"))
    assert res.alarms[0] is None
    helix = [c for c in arcs(res) if abs(c.arc.radius_start_m - 0.003) < 1e-9]
    assert helix and all(c.arc.sweep_rad > 0 for c in helix)
    circles = [c for c in arcs(res) if abs(c.arc.radius_start_m - 0.015) < 1e-9]
    assert circles and round(abs(circles[0].arc.sweep_rad), 9) == round(2 * math.pi, 9)


def test_slot1_and_slot2():
    res = run(head("S1000 M3 F100\nG0 X0 Y0 Z20\nSLOT1(10, 0, 2, -3, , 2, 30, 12, 0, 0, 40, 0, 180, 100, 300, 3, 0, 0, 0)"))
    assert res.alarms[0] is None
    xy = {(m[1], m[2]) for m in path(res) if m[0] == "linear"}
    assert (50.0, 0.0) in xy and (-50.0, 0.0) in xy           # slot ends 40 +- 15, tool radius 5
    arc_slots = run(head("S1000 M3 F100\nG0 X0 Y0 Z20\nSLOT2(10, 0, 2, -3, , 1, 90, 10, 0, 0, 40, 0, 0, 100, 300, 3, 0, 0, 0)"))
    assert arc_slots.alarms[0] is None
    radii = {round(c.arc.radius_start_m * 1000, 6) for c in arcs(arc_slots)}
    assert 40.0 in radii


def test_tool_too_big_for_the_pocket_is_61105():
    assert run(head("S1000 M3\nG0 X0 Y0 Z20\nPOCKET4(10, 0, 2, -4, 4, 30, 30, 4, 0, 0, 400, 100, 0, 11, 4)")).alarms[0][
        "code"] == "61105"


def test_cycle800_swivels_the_table_and_tilts_the_frame():
    res = run("G54 G17 G90 T1 M6\nS1000 M3 F500\nG0 X0 Y0 Z50\n"
              "CYCLE800(1, \"\", 0, 57, 0, 0, 0, 0, 30, 0, 0, 0, 0, 1)\nG0 X0 Y0 Z10\nM30\n", UMC)
    assert res.alarms[0] is None
    swivel = [c for c in res.commands[0] if c.kind == Kind.RAPID and c.space == Space.MACHINE][-1]
    assert math.degrees(swivel.target["B"]) == pytest.approx(30.0)
    assert math.degrees(swivel.target["C"]) == pytest.approx(0.0, abs=1e-9)
    tip = [c for c in res.commands[0] if c.space == Space.TIP][0]
    assert [round(v * 1000, 6) for v in tip.tip_target_m] == [5.0, 0.0, 8.660254]
    assert tip.tip_axis == pytest.approx((0.5, 0.0, math.sqrt(0.75)))


SWIVEL = "G54 G17 G90 T1 M6\nS1000 M3 F500\nG0 X0 Y0 Z50\n"


def swivelled(call: str, base=UMC):
    res = run(SWIVEL + call + "\nG0 X10 Y0 Z10\nM30\n", base)
    tip = [c for c in res.commands[0] if c.space == Space.TIP]
    return res, (tip[0] if tip else None)


def test_cycle800_mode_bits_axis_sequence_solid_and_projection_angles():
    tilted_about_x = (0.0, -0.5, math.sqrt(0.75))
    for call in ('CYCLE800(1, "", 0, 57, 0, 0, 0, 30, 0, 0, 0, 0, 0, 1)',      # X then Y then Z: _A about X
                 'CYCLE800(1, "", 0, 27, 0, 0, 0, 0, 0, 30, 0, 0, 0, 1)',      # Z, Y, X: _C about X
                 'CYCLE800(1, "", 0, 137, 0, 0, 0, 30, 0, 0, 0, 0, 0, 1)'):    # projection angles X, Y
        res, tip = swivelled(call)
        assert res.alarms[0] is None, call
        assert tip.tip_axis == pytest.approx(tilted_about_x, abs=1e-12), call
        assert [round(v * 1000, 6) for v in tip.tip_target_m] == [10.0, -5.0, 8.660254]
    res, tip = swivelled('CYCLE800(1, "", 0, 64, 0, 0, 0, 90, 30, 0, 0, 0, 0, 1)')     # solid angle
    assert tip.tip_axis == pytest.approx((0.0, 0.5, math.sqrt(0.75)), abs=1e-12)
    assert [round(v * 1000, 6) for v in tip.tip_target_m] == [0.0, 13.660254, 3.660254]
    alarm = run(SWIVEL + 'CYCLE800(1, "", 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 1)\nM30\n', UMC).alarms[0]
    assert alarm["code"] == "61019"
    assert run(SWIVEL + "CYCLE800()\nM30\n", UMC).alarms[0] is None         # the zero plane


def test_cycle800_rotary_axes_direct_positions_the_axes_and_aligns_the_frame():
    res, tip = swivelled('CYCLE800(1, "", 0, 192, 0, 0, 0, 30, 20, 0, 0, 0, 0, 1)')
    assert res.alarms[0] is None
    assert math.degrees(tip.target["B"]) == pytest.approx(30.0) and math.degrees(tip.target["C"]) == pytest.approx(20.0)
    c20, s20 = math.cos(math.radians(20)), math.sin(math.radians(20))
    assert tip.tip_axis == pytest.approx((0.5 * c20, 0.5 * s20, math.sqrt(0.75)), abs=1e-12)
    # the frame keeps the machine axes: X10 Z10 of the swivelled plane is R_z(20) R_y(30) (10, 0, 10)
    in_plane = 10.0 * (math.cos(math.radians(30)) + 0.5)                    # R_y(30) (10, 0, 10): x
    assert [v * 1000 for v in tip.tip_target_m] == pytest.approx([in_plane * c20, in_plane * s20,
                                                                  10.0 * (math.cos(math.radians(30)) - 0.5)], abs=1e-9)
    assert run(SWIVEL + 'CYCLE800(1, "", 0, 192, 0, 0, 0, 200, 0, 0, 0, 0, 0, 1)\nM30\n', UMC).alarms[0][
        "code"] == "61189"                                                      # B travel -35..120 deg
    assert run(SWIVEL + 'CYCLE800(1, "", 0, 192, 0, 0, 0, 30, 0, 0, 0, 0, 0, 1)\nM30\n').alarms[0]["code"] == "61153"


def test_traori_programs_the_tool_tip():
    res = run("G54 G90 T1 M6\nS1000 M3 F500\nTRAORI\nG1 X10 Y0 Z0 B20 C0\nTRAFOOF\nM30\n", UMC)
    assert res.alarms[0] is None
    tip = [c for c in res.commands[0] if c.space == Space.TIP]
    assert len(tip) == 1 and tip[0].modal["tip_interpolation"] == "tip"
    assert [round(v * 1000, 6) for v in tip[0].tip_target_m] == [10.0, 0.0, 0.0]


def test_traori_orientation_vector_a3_b3_c3_picks_the_table_solution():
    res = run("G54 G90 T1 M6\nS1000 M3 F500\nTRAORI\nG1 X10 Y0 Z0 A3=0 B3=0.5 C3=COS(30)\nTRAFOOF\nM30\n", UMC)
    assert res.alarms[0] is None
    tip = [c for c in res.commands[0] if c.space == Space.TIP]
    assert len(tip) == 1
    assert tip[0].tip_axis == pytest.approx((0.0, 0.5, math.sqrt(0.75)), abs=1e-12)
    assert [round(v * 1000, 6) for v in tip[0].tip_target_m] == [10.0, 0.0, 0.0]
    assert run("G54 G90 T1 M6\nS1000 M3 F500\nTRAORI\nG1 X10 A3=0 B3=0 C3=0\nM30\n", UMC).alarms[0] is not None


def test_cycle832_high_speed_settings():
    res = run(head("CYCLE832(0.02, _FINISH, 1)\nCYCLE832()"))
    assert res.alarms[0] is None
    modes = [c for c in res.commands[0] if c.kind == Kind.MODE and "cycle832" in c.modal]
    assert modes[0].smoothing.tolerance_m == pytest.approx(2e-5) and modes[0].modal["cycle832"]["mode"] == "_FINISH"
    assert modes[1].smoothing.tolerance_m == 0.0


LATHE_TOOLS = [ToolOffsetInit(1, x_m=0.0, z_m=0.0, nose_radius_m=0.0, radius_m=0.0015)]
CONTOUR = ("KONTUR.SPF", "G1 X20 Z0\nG1 Z-20\nG1 X40 Z-30\nG1 Z-50\nG1 X60\nM17\n")


def lathe(body: str, extra=(CONTOUR,)):
    return run("G18 G90 G95 DIAMON T1 D1\nS500 M3\nG0 X70 Z5\n" + body + "\nG0 X100 Z50\nM30\n", LATHE, extra=extra,
               tools=LATHE_TOOLS)


def xz(res) -> list[tuple]:
    return [(c.kind.value, round(c.target["X"] * 2000.0, 6), round(c.target["Z"] * 1000.0, 6))
            for c in res.commands[0] if c.kind in (Kind.RAPID, Kind.LINEAR, Kind.ARC)]


def test_cycle95_complete_roughing_follows_the_contour_and_finishes_it():
    res = lathe('CYCLE95("KONTUR", 2.5, 0, 0, 0, 0.3, 0.1, 0.1, 9, 0, 0, 1)')
    assert res.alarms[0] is None
    moves = xz(res)
    finish = moves[-7:-1]
    assert finish == [("linear", 20.0, 0.0), ("linear", 20.0, -20.0), ("linear", 40.0, -30.0), ("linear", 40.0, -50.0),
                      ("linear", 60.0, -50.0), ("rapid", 70.0, 1.0)]
    cuts = [m for m in moves if m[0] == "linear"][:-5]
    assert min(m[1] for m in cuts) == pytest.approx(20.0)
    assert all(m[1] >= 20.0 - 1e-9 for m in cuts)


def test_cycle95_allowances_leave_material():
    res = lathe('CYCLE95("KONTUR", 2.5, 0.2, 0.3, 0, 0.3, 0.1, 0.1, 1, 0, 0, 1)')
    assert res.alarms[0] is None
    cuts = [m for m in xz(res) if m[0] == "linear"]
    assert min(m[1] for m in cuts) == pytest.approx(20.6)


def test_cycle97_thread_passes_depths_and_lead():
    res = lathe("CYCLE97(1.5, , 0, -20, 30, 30, 2, 1, 0.92, 0.05, 0, 0, 3, 1, 1, 1)")
    assert res.alarms[0] is None
    threads = [c for c in res.commands[0] if c.kind == Kind.LINEAR]
    assert len(threads) == 3 + 1 + 1
    assert all(c.feed.value == pytest.approx(0.0015) and not c.feed.override_allowed for c in threads)
    depths = [round(30.0 - c.target["X"] * 2000.0, 6) for c in threads]
    assert depths[:3] == pytest.approx([2 * 0.29, 2 * 0.58, 2 * 0.87])
    assert depths[3] == pytest.approx(2 * 0.92) and depths[4] == pytest.approx(2 * 0.92)


def test_cycle93_groove_roughing_and_finishing():
    res = lathe("F0.1\nCYCLE93(40, -20, 6, 4, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.2, 2, 0, 1, 1)")
    assert res.alarms[0] is None
    cuts = [m for m in xz(res) if m[0] == "linear"]
    assert min(m[1] for m in cuts) == pytest.approx(32.0)
    zs_ = {m[2] for m in cuts}
    assert min(zs_) == pytest.approx(-20.0) and max(zs_) == pytest.approx(-17.0)


def test_turning_cycles_need_a_lathe():
    assert run(head('CYCLE95("KONTUR", 2, 0, 0, 0, 0.3, 0.1, 0.1, 9)'), extra=[CONTOUR]).alarms[0]["code"] == "61004"
