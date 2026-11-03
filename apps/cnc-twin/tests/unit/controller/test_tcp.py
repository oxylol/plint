"""Five-axis functions: G43.4 / G43.5 tool centre point control, G68.2 / G68.3 / G68.4, G53.1, G69."""

import copy
import math

import numpy as np
import pytest

from cnctwin.controller.ir import Kind, Space
from harness import MILL, UMC, alarm, moves, run, wrap

FANUC_UMC = copy.deepcopy(UMC)
FANUC_UMC["controller"]["dialect"] = "fanuc"
OFFSETS = {"G54": [-0.3, -0.25, -0.65, 0.0, 0.0]}

HEAD = copy.deepcopy(MILL)
HEAD["id"] = "test-head"
HEAD["type"] = "five_axis_mill"
HEAD["axes"]["B"] = {"type": "rotary", "travel_deg": [-110, 110], "continuous": False}
HEAD["axes"]["C"] = {"type": "rotary", "travel_deg": [-360, 360], "continuous": True}
HEAD["kinematics"] = {
    "workpiece_chain": [{"link": "table", "fixed": {"translate_mm": [0, 0, 0]}}],
    "tool_chain": [{"link": "x", "axis": "X", "direction": [1, 0, 0]}, {"link": "y", "axis": "Y", "direction": [0, 1, 0]},
                   {"link": "z", "axis": "Z", "direction": [0, 0, 1]},
                   {"link": "c_head", "axis": "C", "axis_point_mm": [0, 0, 400], "direction": [0, 0, 1]},
                   {"link": "b_head", "axis": "B", "axis_point_mm": [0, 0, 400], "direction": [0, 1, 0]},
                   {"link": "nose", "fixed": {"translate_mm": [0, 0, 250]}}],
    "home_mm": {"X": 0, "Y": 0, "Z": 0, "B": 0, "C": 0}, "tool_axis": [0, 0, 1]}


@pytest.fixture(autouse=True)
def _providers():
    from cnctwin.controller import hooks, tcp
    hooks.load_plugins()
    tcp.register()


def tips(result):
    return [c for c in moves(result) if c.space == Space.TIP]


def test_g43_4_matches_the_table_tcp_of_the_core_and_interpolates_the_tip():
    program = "G90 G54 G00 B0 C0 X0 Y0 Z10.\n{code} H1 B30. C45.\nG01 X5. Y3. Z2. F500.\nG49"
    fanuc = run(wrap(program.format(code="G43.4")), FANUC_UMC, offsets=OFFSETS)
    haas = run(wrap(program.format(code="G234")), UMC, offsets=OFFSETS)
    assert alarm(fanuc) is None and alarm(haas) is None
    for a, b in zip(tips(fanuc), tips(haas)):
        assert a.target == pytest.approx(b.target, abs=1e-12)
        assert a.tip_target_m == pytest.approx(b.tip_target_m, abs=1e-12)
        assert a.tip_axis == pytest.approx(b.tip_axis, abs=1e-12)
    first, cut = tips(fanuc)
    assert first.modal["tip_interpolation"] == "joint" and cut.modal["tip_interpolation"] == "tip"
    assert cut.tip_target_m == pytest.approx((0.005, 0.003, 0.002))
    assert cut.tip_axis == pytest.approx((math.sin(math.radians(30)) * math.cos(math.radians(45)),
                                          math.sin(math.radians(30)) * math.sin(math.radians(45)),
                                          math.cos(math.radians(30))), abs=1e-12)


def test_g43_5_tool_vector_is_solved_to_rotary_axes_and_kept_without_ijk():
    r = run(wrap("G90 G54 G00 B0 C0 X0 Y0 Z10.\nG43.5 H1 X0 Y0 Z10.\nG01 X5. I0.5 J0 K0.8660254 F500.\nY3.\nG49"),
            FANUC_UMC, offsets=OFFSETS)
    assert alarm(r) is None
    _, tilted, kept = tips(r)
    assert tilted.modal["tcp_type"] == "vector"
    assert tilted.target["B"] == pytest.approx(math.radians(30), abs=1e-7) and tilted.target["C"] == pytest.approx(0.0)
    assert tilted.tip_axis == pytest.approx((0.5, 0.0, 0.8660254), abs=1e-6)
    assert kept.target["B"] == tilted.target["B"] and kept.tip_target_m == pytest.approx((0.005, 0.003, 0.01))


def test_head_head_kinematics_tool_vector_and_tip_position():
    tools_axis = (0.0, -math.sin(math.radians(40)), math.cos(math.radians(40)))
    r = run(wrap(f"G90 G54 G00 B0 C0 X0 Y0 Z100.\nG43.5 H1 X0 Y0 Z100.\n"
                 f"G01 X10. Y20. Z30. I{tools_axis[0]:.9f} J{tools_axis[1]:.9f} K{tools_axis[2]:.9f} F500.\nG49"),
            HEAD, offsets={"G54": [0.0, 0.0, -0.5, 0.0, 0.0]})
    assert alarm(r) is None
    cmd = tips(r)[-1]
    assert cmd.tip_axis == pytest.approx(tools_axis, abs=1e-9)
    from cnctwin.controller.tcp import kinematics
    ex = r.controller.executors[0]
    kin = kinematics(ex)
    tip = kin.tip(dict(cmd.target), 0.1) - kin.tip({"X": 0.0, "Y": 0.0, "Z": -0.5, "B": 0.0, "C": 0.0}, 0.0)
    assert tip == pytest.approx(np.array([0.01, 0.02, 0.03]), abs=0.6e-6)      # linear axes in 1 um counts


def test_g68_2_euler_zxz_feature_frame_and_g53_1():
    r = run(wrap("G90 G54 G00 B0 C0 X0 Y0 Z50.\nG43 H1 Z50.\nG68.2 X10. Y0 Z0 I0 J30. K0\nG53.1\nG00 X0 Y0 Z20.\n"
                 "G01 X5. F300.\nG69\nG00 Z100."), FANUC_UMC, offsets=OFFSETS)
    assert alarm(r) is None
    oriented = [c for c in moves(r) if c.modal.get("tool_axis_control") == "G53.1"][0]
    assert oriented.target["B"] == pytest.approx(math.radians(-30)) and oriented.target["C"] == pytest.approx(math.pi / 2)
    first, cut = tips(r)
    assert first.tip_target_m == pytest.approx((0.010, -0.010, 0.020 * math.cos(math.radians(30))), abs=1e-9)
    assert cut.tip_target_m == pytest.approx((0.015, -0.010, 0.020 * math.cos(math.radians(30))), abs=1e-9)
    assert first.tip_axis == pytest.approx((0.0, -0.5, math.cos(math.radians(30))), abs=1e-9)
    state = r.controller.state.paths[0]
    assert state.feature is None and state.tcp == "" and state.modal["16"] == "G69"


def test_euler_rpy_three_points_and_incremental_frames():
    from cnctwin.controller.tcp import _euler_zxz, _rpy
    zxz = _euler_zxz(math.radians(90), math.radians(30), math.radians(0))
    assert zxz @ np.array([0, 0, 1.0]) == pytest.approx([math.sin(math.radians(30)), 0.0, math.cos(math.radians(30))])
    rpy = _rpy((math.radians(30), 0.0, math.radians(90)), "123")
    assert rpy @ np.array([1.0, 0, 0]) == pytest.approx([0.0, math.cos(math.radians(30)), math.sin(math.radians(30))])
    body = ("G90 G00 X0 Y0 Z0\nG68.2 P2 Q1 X10. Y0 Z0\nG68.2 P2 Q2 X20. Y0 Z0\nG68.2 P2 Q3 X10. Y10. Z10.\n"
            "G01 X0 Y1. Z0 F100.\nG69")
    r = run(wrap(body), MILL)
    assert alarm(r) is None
    move = [c for c in moves(r) if c.kind == Kind.LINEAR][0]
    s = math.sqrt(0.5)
    assert (move.target["X"], move.target["Y"], move.target["Z"]) == pytest.approx((0.010, s * 0.001, s * 0.001), abs=1e-6)
    r = run(wrap("G90 G00 X0 Y0 Z0\nG68.2 X0 Y0 Z0 I90. J0 K0\nG68.4 X5. Y0 Z0 I0 J0 K0\nG01 X1. Y0 F100.\nG69"), MILL)
    move = [c for c in moves(r) if c.kind == Kind.LINEAR][0]
    assert (move.target["X"], move.target["Y"]) == pytest.approx((0.0, 0.006), abs=1e-9)


def test_g68_3_uses_the_tool_axis_and_alarms():
    r = run(wrap("G90 G54 G00 B0 C0 X0 Y0 Z50.\nG68.3 X0 Y0 Z0\nG01 X5. F100.\nG69"), FANUC_UMC, offsets=OFFSETS)
    assert alarm(r) is None
    assert run(wrap("G90 G00 X0 Y0\nG53.1"), MILL).alarms[0]["code"] == "PS5458"
    assert run(wrap("G90 G00 X0 Y0\nG68.2 X0 Y0 Z0 I0 J0 K0"), MILL).alarms[0]["code"] == "PS5457"
    assert run(wrap("G90 G00 X0 Y0\nG68.3 X0 Y0"), FANUC_UMC, offsets=OFFSETS).alarms[0]["code"] == "PS5457"


def test_program_position_follows_the_feature_frame():
    r = run(wrap("G90 G54 G00 B0 C0 X0 Y0 Z50.\nG68.2 X10. Y0 Z0 I0 J30. K0\nG53.1\nG00 X1. Y2. Z3.\n#100 = #5001\n"
                 "#101 = #5002\n#102 = #5003\nG69"), FANUC_UMC, offsets=OFFSETS)
    assert alarm(r) is None
    v = r.controller.state.paths[0].variables["c100"]
    assert v[:3] == pytest.approx([1.0, 2.0, 3.0], abs=1e-6)


def test_haas_g268_and_g254_match_the_fanuc_functions():
    from harness import HAAS_MILL
    fanuc = run(wrap("G90 G00 X0 Y0 Z10.\nG68.2 X10. Y0 Z0 I0 J30. K0\nG01 X5. Y2. Z1. F300.\nG69"), MILL)
    haas = run(wrap("G90 G00 X0 Y0 Z10.\nG268 X10. Y0 Z0 I30. J0 K0 Q123\nG01 X5. Y2. Z1. F300.\nG269"), HAAS_MILL)
    a, b = ([c for c in moves(r) if c.kind == Kind.LINEAR][0] for r in (fanuc, haas))
    assert alarm(fanuc) is None and alarm(haas) is None
    assert a.target == pytest.approx(b.target, abs=1e-12)          # Euler beta about X = Haas I about X
    dwo = run(wrap("G90 G54 G00 B0 C0 X0 Y0 Z10.\nG254\nG00 B30. C45.\nG01 X5. F500.\nG255"), UMC, offsets=OFFSETS)
    assert alarm(dwo) is None and tips(dwo)[-1].kind == Kind.LINEAR
