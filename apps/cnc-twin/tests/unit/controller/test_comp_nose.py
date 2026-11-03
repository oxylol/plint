"""Tool nose radius compensation (imaginary tool nose 0-9) and optional chamfering / corner rounding."""

import math

import pytest

from cnctwin.controller import ToolOffsetInit
from cnctwin.controller.comp import geometry as G
from cnctwin.controller.ir import Kind
from harness import LATHE, MILL, alarm, moves, run, wrap

R = 0.8
TOOLS = [ToolOffsetInit(1, nose_radius_m=R * 1e-3, tip=3), ToolOffsetInit(2, nose_radius_m=R * 1e-3, tip=0),
         ToolOffsetInit(3, nose_radius_m=R * 1e-3, tip=2)]
CONTOUR = "S500 M03\nG00 X50. Z2.\nG42 G01 X40. Z0 F0.2\nZ-20.\nX60. Z-30.\nZ-40.\nG40 G00 X80."


@pytest.fixture(autouse=True)
def _providers():
    from cnctwin.controller import comp, hooks
    hooks.load_plugins()
    comp.register()


def xz(result):
    return [(c.kind.value[0], round(c.target["X"] * 2000, 4), round(c.target["Z"] * 1000, 4)) for c in moves(result)]


def lathe(body):
    return run(wrap(body), LATHE, tools=TOOLS, offsets={"G54": [0.0, 0.0]})


def test_tip_3_keeps_straight_cuts_on_the_programmed_line_and_corrects_the_taper():
    r = lathe("T0101\n" + CONTOUR)
    assert alarm(r) is None
    shoulder = R * math.sqrt(2.0) - (20.0 + R) - R          # nose centre at Z = r sqrt2 - 20.8, tip r further
    assert xz(r)[2][:2] == ("l", 40.0) and xz(r)[2][2] == pytest.approx(shoulder, abs=6e-4)   # 1 um counts
    assert xz(r)[3][:2] == ("l", 60.0) and xz(r)[3][2] == pytest.approx(shoulder - 10.0, abs=6e-4)
    assert xz(r)[-2:] == [("l", 60.0, -40.0), ("r", 80.0, -40.0)]


def test_tip_0_programs_the_nose_centre():
    r = lathe("T0202\n" + CONTOUR)
    centre = R * math.sqrt(2.0) - (20.0 + R)
    assert xz(r)[2][1] == pytest.approx(40.0 + 2 * R, abs=1e-3) and xz(r)[2][2] == pytest.approx(centre, abs=6e-4)


def test_tip_directions_table_and_negative_radius_swaps_the_side():
    assert G.tip_vector(3, R) == (-R, -R) and G.tip_vector(2, R) == (-R, R) and G.tip_vector(8, R) == (0.0, -R)
    assert G.tip_vector(9, R) == (0.0, 0.0) and G.tip_vector(-5, R) == (0.0, 0.0)
    r = lathe("T0303\nS500 M03\nG00 X10. Z2.\nG41 G01 X20. Z0 F0.2\nZ-20.\nG40 G00 X10.")
    assert alarm(r) is None
    assert xz(r)[2][1] == pytest.approx(20.0, abs=1e-4)      # tip 2 bore: the imaginary nose stays on the bore


def test_g40_with_i_k_gives_the_retract_direction():
    r = lathe("T0101\nS500 M03\nG00 X60. Z2.\nG42 G01 X40. Z0 F0.2\nZ-20.\nG40 G00 X80. Z0 I10. K10.")
    assert alarm(r) is None and xz(r)[-1] == ("r", 80.0, 0.0)


def test_chamfer_and_corner_rounding_without_compensation():
    r = run(wrap("G90 G00 X0 Y0 Z5.\nG01 X50. ,C5. F500.\nY25.\nX0 ,R8.\nY0"), MILL)
    assert alarm(r) is None
    path = [(c.kind.value[0], round(c.target["X"] * 1000, 4), round(c.target["Y"] * 1000, 4)) for c in moves(r)]
    assert path[1:] == [("l", 45.0, 0.0), ("l", 50.0, 5.0), ("l", 50.0, 25.0), ("l", 8.0, 25.0), ("a", 0.0, 17.0),
                        ("l", 0.0, 0.0)]
    fillet = [c for c in moves(r) if c.kind == Kind.ARC][0]
    assert tuple(round(v * 1000, 4) for v in fillet.arc.centre[:2]) == (8.0, 17.0)
    assert fillet.arc.sweep_rad == pytest.approx(math.pi / 2) and r.controller.state.paths[0].comp_state is None


def test_chamfer_with_cutter_compensation_and_chamfer_alarms():
    tools = [ToolOffsetInit(1, length_m=0.1, radius_m=0.005)]
    r = run(wrap("G90 G00 X-20. Y-20. Z5.\nG41 G01 X0 Y0 D1 F500.\nY30. ,C5.\nX40.\nG40 X60."), MILL, tools=tools)
    k = 5.0 * math.tan(math.radians(22.5))
    path = [(round(c.target["X"] * 1000, 3), round(c.target["Y"] * 1000, 3)) for c in moves(r)][2:4]
    assert path == [(-5.0, round(25.0 + k, 3)), (round(5.0 - k, 3), 35.0)]
    assert alarm(run(wrap("G90 G00 X0 Y0 Z5.\nG01 X50. ,C5. F500.\nG00 Y25."), MILL))["code"] == "PS0052"
    assert alarm(run(wrap("G90 G00 X0 Y0 Z5.\nG01 X50. ,C5. F500.\nM08\nM09\nY25."), MILL))["code"] == "PS0051"
    assert alarm(run(wrap("G90 G00 X0 Y0 Z5.\nG01 X50. ,C60. F500.\nY25."), MILL))["code"] == "PS0055"


def test_geometry_offsets_intersections_and_arc_checks():
    line = G.Seg("line", (0.0, 0.0), (10.0, 0.0))
    up = G.Seg("line", (10.0, 0.0), (10.0, 10.0))
    assert G.corner(line, up, 1.0).points == [pytest.approx((9.0, 1.0))]
    acute = G.corner(line, G.Seg("line", (10.0, 0.0), (0.0, 1.0)), -1.0)
    assert acute.kind == "outer_acute" and len(acute.points) == 2
    arc = G.Seg("arc", (10.0, 0.0), (0.0, 10.0), (0.0, 0.0), math.pi / 2)
    assert G.offset(arc, 2.0).radius == pytest.approx(8.0) and G.offset(arc, 11.0) is None
    assert G.arc_interferes(arc, (0.0, 8.0), (8.0, 0.0))
    assert not G.direction_interferes((0, 0), (1, 0), (0, 1), (2, 1)) and G.direction_interferes((0, 0), (1, 0), (2, 1), (0, 1))
