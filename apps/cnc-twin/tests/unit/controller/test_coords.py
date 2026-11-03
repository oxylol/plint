import math

import pytest

from cnctwin.controller import ToolOffsetInit
from harness import alarm, mm, moves, run, wrap, xyz

OFFSETS = {"G54": [-0.400, -0.200, -0.300], "G55": [-0.100, -0.050, -0.250]}


def test_work_offset_chain_g54_g55_g54_1():
    body = """G90 G54 G00 X10. Y10.
G55 X10. Y10.
G10 L20 P3 X-0.2 Y-0.1 Z-0.05
G54.1 P3 X10. Y10.
G54 X0 Y0"""
    m = moves(run(wrap(body), offsets=OFFSETS))
    assert xyz(m[0])[:2] == (-390.0, -190.0)
    assert xyz(m[1])[:2] == (-90.0, -40.0)
    assert xyz(m[2])[:2] == (9.8, 9.9)
    assert xyz(m[3])[:2] == (-400.0, -200.0)


def test_g10_l2_sets_and_adds_work_offsets():
    body = "G10 L2 P1 X-100. Y-50. Z-20.\nG90 G00 X0 Y0 Z0\nG91 G10 L2 P1 X-1.\nG90 G00 X0"
    m = moves(run(wrap(body)))
    assert xyz(m[0]) == (-100.0, -50.0, -20.0)
    assert mm(m[1], "X") == -101.0
    assert alarm(run(wrap("G10 L2 P9 X0")))["code"] == "PS0031"


def test_external_offset_adds_to_every_work_system():
    m = moves(run(wrap("G10 L2 P0 X-1. Y-2. Z-3.\nG90 G54 G00 X0 Y0 Z0"), offsets=OFFSETS))
    assert xyz(m[0]) == (-401.0, -202.0, -303.0)


def test_g52_local_shift_and_g92():
    body = """G90 G54 G00 X0 Y0
G52 X10. Y5.
G00 X0 Y0
G52 X0 Y0
G00 X0 Y0
G92 X100. Y100.
G00 X100. Y100.
G00 X0 Y0"""
    m = moves(run(wrap(body), offsets=OFFSETS))
    assert xyz(m[1])[:2] == (-390.0, -195.0)
    assert xyz(m[2])[:2] == (-400.0, -200.0)
    assert xyz(m[3])[:2] == (-400.0, -200.0)
    assert xyz(m[4])[:2] == (-500.0, -300.0)


def test_g92_cancels_local_shift():
    body = "G90 G00 X0 Y0\nG52 X10.\nG00 X0\nG92 X50.\nG00 X50.\nX0"
    m = moves(run(wrap(body), offsets=OFFSETS))
    assert mm(m[1], "X") == -390.0
    assert mm(m[2], "X") == -390.0
    assert mm(m[3], "X") == -440.0


def test_tool_length_offset_g43_g44_g49_and_wear():
    tools = [ToolOffsetInit(1, length_m=0.100, length_wear_m=-0.0005), ToolOffsetInit(2, length_m=0.050)]
    body = "G90 G00 G43 Z10. H1\nG44 Z10. H2\nG49 Z10.\nG43 H2\nX1."
    m = moves(run(wrap(body), offsets=OFFSETS, tools=tools))
    assert mm(m[0], "Z") == pytest.approx(-300.0 + 10.0 + 99.5)
    assert mm(m[1], "Z") == pytest.approx(-300.0 + 10.0 - 50.0)
    assert mm(m[2], "Z") == pytest.approx(-290.0)
    assert mm(m[3], "Z") == pytest.approx(-290.0 + 50.0)
    assert m[0].modal["length_offset_m"] == pytest.approx(0.0995) and m[0].modal["h"] == 1


def test_g10_tool_offsets_l10_to_l13_and_system_variables():
    body = """G10 L10 P3 R120.
G10 L11 P3 R-0.2
G10 L12 P3 R5.
G10 L13 P3 R0.01
#101 = #2203
#102 = #2003
#103 = #2603
#104 = #2403
#11003 = 130.
#105 = #2203
G90 G43 G00 Z0 H3"""
    r = run(wrap(body))
    ctl = r.controller
    v = ctl.state.paths[0].variables["c100"]
    assert v[1:6] == pytest.approx([120.0, -0.2, 5.0, 0.01, 130.0])
    assert mm(moves(r)[0], "Z") == pytest.approx(129.8)


def test_scaling_about_a_centre_with_arcs():
    body = """G90 G00 X0 Y0
G51 X10. Y10. P2000
G01 X20. Y10. F100.
G02 X30. Y10. R5.
G50
G00 X0 Y0"""
    m = moves(run(wrap(body)))
    assert xyz(m[1])[:2] == (30.0, 10.0)
    assert xyz(m[2])[:2] == (50.0, 10.0)
    assert m[2].arc.radius_start_m == pytest.approx(0.010) and m[2].arc.sweep_rad == pytest.approx(-math.pi)
    assert xyz(m[3])[:2] == (0.0, 0.0)


def test_per_axis_scaling_and_negative_scale_mirrors():
    m = moves(run(wrap("G90 G00 X0 Y0\nG51 X0 Y0 I2000 J-1000\nG01 X10. Y10. F100.\nG50")))
    assert xyz(m[1])[:2] == (20.0, -10.0)


def test_programmable_mirror_reverses_arcs():
    body = """G90 G00 X0 Y0
G51.1 X0
G01 X10. Y5. F100.
G02 X20. Y5. R5.
G50.1
G01 X10. Y5."""
    m = moves(run(wrap(body)))
    assert xyz(m[1])[:2] == (-10.0, 5.0)
    assert xyz(m[2])[:2] == (-20.0, 5.0)
    assert m[2].arc.sweep_rad == pytest.approx(math.pi)
    assert m[2].arc.centre[0] == pytest.approx(-0.015)
    assert xyz(m[3])[:2] == (10.0, 5.0)


def test_coordinate_rotation_and_incremental_moves():
    body = """G90 G00 X0 Y0
G68 X0 Y0 R90.
G01 X10. Y0 F100.
G91 X5.
G90 G69
G01 X10. Y0"""
    m = moves(run(wrap(body)))
    assert xyz(m[1])[:2] == pytest.approx((0.0, 10.0))
    assert xyz(m[2])[:2] == pytest.approx((0.0, 15.0))
    assert xyz(m[3])[:2] == (10.0, 0.0)


def test_rotation_with_arc_keeps_turning_sense():
    m = moves(run(wrap("G90 G00 X0 Y0\nG68 X0 Y0 R45.\nG03 X10. Y0 R5. F100.\nG69")))
    arc = m[1]
    assert xyz(arc)[:2] == pytest.approx((7.071, 7.071))
    assert arc.arc.sweep_rad == pytest.approx(math.pi)
    assert arc.arc.centre[:2] == pytest.approx((0.005 * math.cos(math.pi / 4), 0.005 * math.sin(math.pi / 4)))


def test_mirror_then_scale_then_rotate_order():
    m = moves(run(wrap("G90 G00 X0 Y0\nG51.1 X0\nG51 X0 Y0 P2000\nG68 X0 Y0 R90.\nG01 X10. Y0 F100.\nG69 G50 G50.1")))
    assert xyz(m[1])[:2] == pytest.approx((0.0, -20.0))


def test_polar_coordinates_absolute_and_incremental():
    body = """G90 G17 G00 X0 Y0
G16
G01 X10. Y30. F100.
Y90.
G91 X5. Y90.
G90 G15
G00 X0 Y0"""
    m = moves(run(wrap(body)))
    assert xyz(m[1])[:2] == pytest.approx((8.660, 5.0))
    assert xyz(m[2])[:2] == pytest.approx((0.0, 10.0))
    assert xyz(m[3])[:2] == pytest.approx((-5.0, 10.0))


def test_g28_via_intermediate_point_g29_and_g30():
    body = """G90 G00 X10. Y10. Z10.
G91 G28 Z0
G90 G28 X20. Y20.
G29 X5. Y5.
G30 P2 Z0"""
    r = run(wrap(body), offsets=OFFSETS)
    m = moves(r)
    assert xyz(m[1]) == (-390.0, -190.0, -290.0)
    assert xyz(m[2]) == (-390.0, -190.0, 0.0)
    assert xyz(m[3])[:2] == (-380.0, -180.0)
    assert xyz(m[4])[:2] == (0.0, 0.0)
    assert xyz(m[5])[:2] == (-380.0, -180.0)
    assert xyz(m[6])[:2] == (-395.0, -195.0)
    assert alarm(r) is None


def test_g27_reference_check():
    assert alarm(run(wrap("G90 G53 X0 Y0 Z0\nG27 X400. Y200.")))["code"] == "PS0092"
    r = run(wrap("G90 G27 X400. Y200. Z300."), offsets=OFFSETS)
    assert alarm(r) is None


def test_continuous_rotary_axis_shortest_path():
    from harness import UMC
    m = moves(run(wrap("G90 G00 C350.\nC10.\nG91 C-30.\nG90 C-90."), UMC))
    degrees = [round(math.degrees(c.target["C"]), 6) for c in m]
    assert degrees == [-10.0, 10.0, -20.0, -90.0]
