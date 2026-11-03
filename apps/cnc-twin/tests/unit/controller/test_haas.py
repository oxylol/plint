import math

import pytest

from cnctwin.controller import ToolOffsetInit
from cnctwin.controller.ir import Kind, Space
from harness import HAAS_MILL, UMC, alarm, mm, moves, run, wrap, xyz

TOOLS = [ToolOffsetInit(1, length_m=0.1, radius_m=0.005), ToolOffsetInit(2, length_m=0.1, radius_m=0.003)]


def haas(body, machine=HAAS_MILL, **kwargs):
    kwargs.setdefault("tools", TOOLS)
    return run(wrap(body), machine, **kwargs)


def test_haas_feed_dwell_and_one_m_code_per_block():
    r = haas("G01 X10. F100\nG04 P2.5\nG04 P500")
    assert moves(r)[0].feed.value == pytest.approx(0.1 / 60.0)
    assert [c.dwell_s for c in r.commands[0] if c.kind == Kind.DWELL] == pytest.approx([2.5, 0.5])
    assert alarm(haas("S1000 M03 M08"))["code"] == "339"
    assert alarm(haas("G04"))["code"] == "326"


def test_haas_m_codes_act_at_the_end_of_the_block():
    r = haas("S500 M03\nG00 X1. M08")
    kinds = [c.kind for c in r.commands[0]]
    assert kinds.index(Kind.RAPID) < kinds.index(Kind.COOLANT)


def test_m88_m89_tsc_m83_air_m34_nozzle_conveyor_door():
    r = haas("M88\nM89\nM83\nM84\nM34\nM34\nM35\nM31\nM33\nM80\nM81")
    co = [c for c in r.commands[0] if c.kind == Kind.COOLANT]
    assert co[0].coolant["pumps"]["tsc"] is True and co[1].coolant["pumps"]["tsc"] is False
    assert co[2].coolant["air"] is True and co[3].coolant["air"] is False
    assert [c.coolant["nozzles"]["pcool"] for c in co[4:7]] == [2, 3, 2]
    assert [c.modal["device"]["action"] for c in r.commands[0] if c.kind == Kind.CONVEYOR] == ["forward", "stop"]
    assert [c.modal["device"]["action"] for c in r.commands[0] if c.kind == Kind.DOOR] == ["open", "close"]
    assert alarm(haas("M50"))["code"] == "TW0010"


def test_m97_local_subprogram_and_haas_macro_alarm_numbers():
    body = "M97 P100 L2\nG00 Y5.\nM30\nN100 G91 G00 X1.\nM99"
    m = moves(run("%\nO01000\n" + body + "\n%\n", HAAS_MILL))
    assert [mm(c, "X") for c in m[:2]] == [1.0, 2.0] and mm(m[2], "Y") == 5.0
    a = alarm(haas("#3000 = 15 (CHECK FIXTURE)"))
    assert a["code"] == "1015" and a["text"] == "CHECK FIXTURE"
    assert alarm(haas("M97 P900"))["code"] == "337"
    assert alarm(haas("M98 P5555"))["code"] == "314"


def test_haas_variables_ngc_and_legacy_numbers():
    r = haas("#100 = 5\n#10101 = 6\n#5221 = -20.\n#101 = #5221\n#102 = #3026\n#103 = #4124\nG00 X3.")
    path = r.controller.state.paths[0]
    assert path.variables["c10000"][100] == 5.0 and path.variables["c10000"][101] == -20.0
    assert path.variables["c10000"][103] is None
    assert alarm(haas("#100 = #720"))["code"] == "331"


def test_g187_smoothing_levels_and_g103_lookahead():
    r = haas("G187 P1\nG01 X1. F100.\nG187 P3 E0.05\nX2.\nG187\nX3.\nG103 P2\nX4.")
    m = moves(r)
    assert m[0].smoothing.tolerance_m == pytest.approx(25e-6) and m[0].smoothing.mode_code == "G187 P1"
    assert m[1].smoothing.tolerance_m == pytest.approx(50e-6) and m[1].smoothing.mode_code == "G187 P3"
    assert m[2].smoothing.tolerance_m == 0.0
    assert r.controller.executors[0].capacity() == 2
    modes = [c for c in r.commands[0] if c.kind == Kind.MODE and c.smoothing is not None]
    assert len(modes) == 3


def test_g110_and_g154_work_offsets():
    body = "G10 L20 P1 X-11.\nG10 L20 P25 X-22.\nG110 G00 X0\nG154 P25 X0\nG10 L2 P0 X-1.\nG110 X0"
    m = moves(haas(body))
    assert [mm(c, "X") for c in m] == [-11.0, -22.0, -12.0]


def test_g12_circular_pocket_path():
    r = haas("G90 G00 X50. Y50. Z1.\nG13 I10. K20. Q5. D1 Z-2. F200.")
    m = moves(r)
    arcs = [c for c in m if c.kind == Kind.ARC]
    assert alarm(r) is None
    assert mm(m[1], "Z") == -2.0
    radii = sorted({round(c.arc.radius_start_m * 1000, 6) for c in arcs if abs(c.arc.sweep_rad) > 6.0})
    assert radii == [5.0, 10.0, 15.0]
    assert all(c.arc.sweep_rad > 0 for c in arcs)
    assert xyz(m[-1])[:2] == (50.0, 50.0)
    assert alarm(haas("G12 I3. D1 F100."))["code"] == "369"
    assert alarm(haas("G12 I10. K20. F100."))["code"] == "305"


def test_g47_engraving_strokes_and_serial_number():
    r = haas("G90 G00 X0 Y0 Z5.\nG47 P0 X10. Y10. I0 J6. R2. Z-0.1 F300. E50. (LI)")
    m = moves(r)
    assert alarm(r) is None
    depth = [c for c in m if abs(mm(c, "Z") + 0.1) < 1e-9 and c.kind == Kind.LINEAR]
    xs = sorted({mm(c, "X") for c in depth})
    assert min(xs) == 10.0 and max(xs) == pytest.approx(10.0 + 6.0 + 3.0)
    assert any(c.feed.value == pytest.approx(0.05 / 60) for c in m if c.kind == Kind.LINEAR)
    assert mm(m[-1], "Z") == 2.0
    r = haas("#599 = 41\nG47 P1 X0 Y0 J6. R1. Z-0.1 F300. (###)")
    assert r.controller.state.paths[0].variables["c10000"][599] == 42.0
    assert alarm(haas("G47 P0 X0 Y0 J6. R1. Z-0.1 F300. (é)"))["code"] == "TW0014"


def test_g150_general_pocket_rough_and_finish():
    main = "%\nO01000\nG90 G00 X10. Y5. Z1.\nG150 X10. Y5. Z-2. J3. R1. Q1. D2 K0.2 P2000 F300.\nM30\n"
    pocket = "O02000\nG01 Y10.\nX0\nY0\nX20.\nY10.\nX10.\nM99\n%\n"
    r = run(main + pocket, HAAS_MILL, tools=TOOLS)
    assert alarm(r) is None
    m = moves(r)
    cutting = [c for c in m if c.kind in (Kind.LINEAR, Kind.ARC)]
    depths = sorted({mm(c, "Z") for c in cutting})
    assert depths[0] == -2.0 and -1.0 in depths
    finish = [c for c in cutting if mm(c, "Z") == -2.0 and c.kind == Kind.LINEAR]
    xs = [mm(c, "X") for c in finish]
    ys = [mm(c, "Y") for c in finish]
    assert min(xs) == pytest.approx(3.0) and max(xs) == pytest.approx(17.0)
    assert min(ys) == pytest.approx(3.0) and max(ys) == pytest.approx(7.0)
    level_y = {round(mm(c, "Y"), 6) for c in cutting if mm(c, "Z") == -1.0 and c.kind == Kind.LINEAR}
    assert {3.2, 5.0, 6.8} <= level_y and {3.0, 7.0} <= level_y
    assert min(level_y) == 3.0 and max(level_y) == 7.0
    bad = run(main + "O02000\nG01 Y10.\nX0\nY0\nM99\n%\n", HAAS_MILL, tools=TOOLS)
    assert alarm(bad)["code"] == "370" and alarm(bad)["text"] == "Pocket Definition Error"


def test_g254_dwo_and_g234_tcpc_on_a_table_machine():
    r = haas("G90 G54 G00 B0 C0 X0 Y0 Z10.\nG254\nG00 B90. C0\nX0 Y0 Z10.\nG255\nG234 H1 Z10.\nG01 X5. F500.\nG49",
             UMC, offsets={"G54": [-0.3, -0.25, -0.65, 0.0, 0.0]})
    assert alarm(r) is None
    m = moves(r)
    dwo = [c for c in m if c.space == Space.TIP and c.modal.get("tip_interpolation") == "joint"]
    tcpc = [c for c in m if c.space == Space.TIP and c.modal.get("tip_interpolation") == "tip"]
    assert dwo and tcpc
    point = dwo[-1]
    assert point.tip_target_m == pytest.approx((0.0, 0.0, 0.01))
    assert point.tip_axis == pytest.approx((1.0, 0.0, 0.0), abs=1e-12)
    assert point.target["B"] == pytest.approx(math.pi / 2)
    assert (point.target["X"], point.target["Y"], point.target["Z"]) == pytest.approx((-0.31, -0.25, -0.65), abs=1e-9)
    cut = tcpc[-1]
    assert cut.kind == Kind.LINEAR and cut.tip_target_m == pytest.approx((0.005, 0.0, 0.01))
    assert (cut.target["X"], cut.target["Y"], cut.target["Z"]) == pytest.approx((-0.31, -0.25, -0.545), abs=1e-9)
    assert cut.modal["work_origin_M0_m"] == pytest.approx([-0.3, -0.25, -0.65])
    assert alarm(haas("G254"))["code"] == "TW0003"


def test_g268_feature_coordinate_system():
    m = moves(haas("G90 G00 X0 Y0 Z0\nG268 X10. Y0 Z0 I0 J0 K90. Q321\nG01 X5. Y0 F100.\nG269\nG01 X1. Y1."))
    assert xyz(m[1]) == pytest.approx((10.0, 5.0, 0.0))
    assert xyz(m[2])[:2] == (1.0, 1.0)


def test_haas_g68_centre_uses_a_and_b():
    m = moves(haas("G90 G00 X0 Y0\nG68 A10. B0 R90.\nG01 X20. Y0 F100.\nG69"))
    assert xyz(m[1])[:2] == pytest.approx((10.0, 10.0))


def test_m78_m79_after_skip():
    assert alarm(haas("G31 X10. F100.\nM79"))["code"] == "379"
    assert alarm(haas("G31 X10. F100.\nM78")) is None


def test_g150_haas_manual_example_with_arc_corners():
    program = """%
O01001
G20 G90 G54 G00 X3.25 Y4.5 Z1.
G150 X3.25 Y4.5 Z-1.5 G41 J0.35 R0.1 Q0.4 D01 K0.01 P2001 F15.
G40 G01 X3.25 Y4.5
G00 Z1.0
M30
O02001
G01 Y7.0
X1.5
G03 Y5.25 R0.875
G01 Y2.25
G03 Y0.5 R0.875
G01 X5.0
G03 Y2.25 R0.875
G01 Y5.25
G03 Y7.0 R0.875
G01 X3.25
M99
%
"""
    tools = [ToolOffsetInit(1, length_m=0.1, radius_m=0.125 * 0.0254)]
    r = run(program, HAAS_MILL, tools=tools)
    assert alarm(r) is None
    m = moves(r)
    depths = sorted({round(c.target["Z"] / 0.0254, 4) for c in m if c.kind == Kind.LINEAR})
    assert depths[:4] == [-1.5, -1.1, -0.7, -0.3]
    finish_arcs = [c for c in m if c.kind == Kind.ARC and round(c.target["Z"] / 0.0254, 4) == -1.5]
    assert {round(c.arc.radius_start_m / 0.0254, 4) for c in finish_arcs} == {0.75, 0.125}
    assert all(c.arc.sweep_rad > 0 for c in finish_arcs if round(c.arc.radius_start_m / 0.0254, 4) == 0.75)
    xs = [c.target["X"] / 0.0254 for c in m if round(c.target["Z"] / 0.0254, 4) == -1.5]
    assert min(xs) > 1.5 - 0.875 + 0.125 - 1e-6 and max(xs) < 5.0 + 0.875 - 0.125 + 1e-6
