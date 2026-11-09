"""Probe entries in tools.list (kind: probe), lathe tool geometry offsets, workholding outline and seat
height, and the from_previous convention (the turned stock frame; stock_pose is the extra placement)."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from cnctwin.controller.domain import setup_from_project, stock_probe
from cnctwin.core.errors import InputError
from cnctwin.project import resolve
from conftest import PROJECT

PROBE_LINE = "    - {number: 1, tool: test-em-10, holder: test-er32, stickout_mm: 35, length_offset_mm: 120.5, runout_um: 5}\n"


def _with(project_dir, old: str, new: str) -> None:
    path = project_dir / "project.yaml"
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new, 1))


def _setup(project, op_index=0):
    ctx = SimpleNamespace(project=project, tick_s=1e-6)
    state = SimpleNamespace(tools=None, clock=SimpleNamespace(op_index=op_index))
    return setup_from_project(ctx, state, op_index)


# ------------------------------------------------------------------ probes

def test_probe_entry_resolves_against_the_machine_probe(project_dir, fixture_db):
    _with(project_dir, PROBE_LINE, PROBE_LINE + (
        "    - {number: 20, kind: probe, probe: {extends: spindle_probe, stylus_ball_mm: 6, stylus_length_mm: 50,\n"
        "       body: {diameter_mm: 40, length_mm: 50}}, length_offset_mm: 125.0, radius_offset_mm: 2.999,\n"
        "       comment: \"SPINDLE PROBE\"}\n"))
    p = resolve(project_dir, fixture_db)
    assert [int(t.number) for t in p.tools.list] == [1, 2]          # the tool table: cutting tools only
    probe = p.tools.probes[0]
    assert int(probe.number) == 20 and probe.list_index == 1        # its place in the file's tools.list
    assert probe.probe.name == "spindle_probe" and probe.probe.model == "probe"       # from the machine
    assert float(probe.probe.stylus_ball_m) == pytest.approx(0.006)
    assert float(probe.probe.body.diameter_m) == pytest.approx(0.040)
    assert float(probe.probe.pretravel_m) == pytest.approx(1e-6)    # the machine's spindle_probe_extra
    setup = _setup(p)
    by_number = {t.number: t for t in setup.tools}
    assert by_number[20].probe == "spindle_probe" and by_number[1].probe is None
    assert by_number[20].length_m == pytest.approx(0.125) and by_number[20].radius_m == pytest.approx(0.002999)


def test_probe_entry_problems(project_dir, fixture_db):
    _with(project_dir, PROBE_LINE, PROBE_LINE + "    - {number: 20, kind: probe, probe: spindle_probe}\n")
    with pytest.raises(InputError, match="stylus_ball_mm"):           # the fixture machine gives no ball
        resolve(project_dir, fixture_db)
    _with(project_dir, "probe: spindle_probe}", "probe: laser_probe}")
    with pytest.raises(InputError, match="no probe 'laser_probe'"):
        resolve(project_dir, fixture_db)
    _with(project_dir, "probe: laser_probe}", "probe: {model: x}}")
    with pytest.raises(InputError, match="stylus_ball_mm"):
        resolve(project_dir, fixture_db)
    _with(project_dir, "number: 20, kind: probe, probe: {model: x}}", "number: 2, kind: probe, probe: {stylus_ball_mm: 4}}")
    with pytest.raises(InputError, match="used twice"):
        resolve(project_dir, fixture_db)


# ------------------------------------------------------------------ lathe offsets and workholding keys

def test_lathe_geometry_offsets_reach_the_controller(project_dir, fixture_db):
    _with(project_dir, "stickout_mm: 25\n", "stickout_mm: 25\n      x_offset_mm: -210.5\n      z_offset_mm: -330.25\n"
          "      nose_radius_mm: 0.8\n      tip: 3\n      wear_offsets_mm: {x: 0.02, z: -0.01}\n")
    p = resolve(project_dir, fixture_db)
    tool = {t.number: t for t in _setup(p).tools}[2]
    assert (tool.x_m, tool.z_m, tool.nose_radius_m, tool.tip) == pytest.approx((-0.2105, -0.33025, 0.0008, 3))
    assert (tool.x_wear_m, tool.z_wear_m) == pytest.approx((2e-5, -1e-5))


def test_workholding_outline_and_seat_height(project_dir, fixture_db):
    _with(project_dir, "input_torque_Nm: 80}", "input_torque_Nm: 80, seat_height_mm: 95,\n"
          "           outline_mm: [[-60, -80, -95], [60, 80, 8]]}")
    p = resolve(project_dir, fixture_db)
    entry = p.operations[0].setup.workholding[0]
    assert float(entry.seat_height_m) == pytest.approx(0.095)
    assert np.allclose(np.asarray(entry.outline_m, dtype=float), [[-0.06, -0.08, -0.095], [0.06, 0.08, 0.008]])
    _with(project_dir, "outline_mm: [[-60, -80, -95], [60, 80, 8]]", "outline_mm: [[-60, -80, -95], [60, 80, 8], [0, 0, 0]]")
    with pytest.raises(InputError, match="outline_mm is a box"):
        resolve(project_dir, fixture_db)


# ------------------------------------------------------------------ from_previous: the turned stock frame

OP20 = """  - name: OP20
    program: op10.nc
    setup:
      workholding:
        - {device: test-vise, pose: {translate_mm: [0, 0, 0], rotate_deg: [0, 0, 0]}}
      stock_pose: STOCK_POSE
      work_offsets_mm: {G55: [-250, -200, -400]}
      fixture_pose: {translate_mm: [-250, -200, -460], rotate_deg: [0, 0, 0]}
    from_previous: FROM
physics:"""


def _two_ops(project_dir, stock_pose: str, from_previous: str):
    _with(project_dir, "      stock_pose: {translate_mm: [10, 20, 0], rotate_deg: [0, 0, 90], order: xyz}",
          "      stock_pose: {translate_mm: [-50, -40, 0], rotate_deg: [0, 0, 0]}")
    _with(project_dir, "physics:", OP20.replace("STOCK_POSE", stock_pose).replace("FROM", from_previous))


def test_flip_turns_the_stock_frame_and_stock_pose_is_the_extra_placement(project_dir, fixture_db):
    _two_ops(project_dir, "{translate_mm: [-50, -40, -2.5], rotate_deg: [0, 0, 0]}",
             "{op: OP10, flip: {axis: X, angle_deg: 180}}")
    p = resolve(project_dir, fixture_db)
    op = p.operation("OP20")
    assert np.allclose(op.transfer_S_Sprev, [[1, 0, 0, 0], [0, -1, 0, 0.08], [0, 0, -1, 0.03], [0, 0, 0, 1]])
    assert np.allclose(op.stock_box_S_m, [[0, 0, 0], [0.1, 0.08, 0.03]])
    point = op.stock_pose_W_S0 @ np.array([0.01, 0.01, 0.0275, 1.0])  # 2.5 below the OP10 blank top
    assert np.allclose(point[:3], [-0.04, 0.03, 0.0])                # lies on the seat, turned over
    assert np.allclose(op.stock_pose_W_S, op.stock_pose_W_S0 @ np.linalg.inv(op.pose_S_S0))


def test_a_stock_pose_that_repeats_the_turn_is_refused_with_the_meant_placement(project_dir, fixture_db):
    _two_ops(project_dir, "{translate_mm: [-50, 40, 27.5], rotate_deg: [180, 0, 0]}",
             "{op: OP10, flip: {axis: X, angle_deg: 180}}")
    with pytest.raises(InputError, match=r"repeats the from_previous turn .*translate_mm: \[-50.0, -40.0, -2.5\], "
                                         r"rotate_deg: \[0.0, 0.0, 0.0\]"):
        resolve(project_dir, fixture_db)


def test_quarter_flip_permutes_the_box_and_no_flip_keeps_the_frame(project_dir, fixture_db):
    _two_ops(project_dir, "{translate_mm: [0, 0, 0], rotate_deg: [0, 0, 0]}", "{op: OP10, flip: {axis: Y, angle_deg: 90}}")
    p = resolve(project_dir, fixture_db)
    op = p.operation("OP20")
    assert np.allclose(op.stock_box_S_m, [[0, 0, 0], [0.03, 0.08, 0.1]])
    rotation = op.transfer_S_Sprev[:3, :3]
    assert np.allclose(rotation, [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], atol=1e-12)
    _with(project_dir, "{op: OP10, flip: {axis: Y, angle_deg: 90}}", "{op: OP10, transfer: manual}")
    op = resolve(project_dir, fixture_db).operation("OP20")
    assert np.allclose(op.transfer_S_Sprev, np.eye(4)) and np.allclose(op.pose_S_S0, np.eye(4))


def test_stock_probe_of_the_set_up_stock(project_dir, fixture_db):
    _with(project_dir, PROBE_LINE, PROBE_LINE + "    - {number: 20, kind: probe, probe: {stylus_ball_mm: 6}}\n")
    _with(project_dir, "      stock_pose: {translate_mm: [10, 20, 0], rotate_deg: [0, 0, 90], order: xyz}",
          "      stock_pose: {translate_mm: [-50, -40, 0], rotate_deg: [0, 0, 0]}")
    probe = stock_probe(resolve(project_dir, fixture_db), 0)
    (lo, hi), = probe.boxes
    assert np.allclose(lo, [-0.3, -0.24, -0.46]) and np.allclose(hi, [-0.2, -0.16, -0.43])
    assert probe.radii == {20: pytest.approx(0.003)} and math.isclose(probe.radius, 0.003)
