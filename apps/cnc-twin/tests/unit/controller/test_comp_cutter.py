"""Cutter radius compensation (FANUC type C) against hand-computed offsets, through the state machine."""

import copy
import io
import json
import math

import numpy as np
import pytest

from cnctwin.controller import Controller, ControllerState, ToolOffsetInit
from cnctwin.controller.ir import Kind
from cnctwin.controller.schema import ControllerSettings
from harness import HAAS_MILL, MILL, alarm, controller, moves, run, wrap

TOOLS = [ToolOffsetInit(1, length_m=0.1, radius_m=0.005), ToolOffsetInit(2, length_m=0.1, radius_m=0.003),
         ToolOffsetInit(3, radius_m=0.006)]
R2 = 5.0 * (math.sqrt(2.0) - 1.0)        # 2.0711: inner start-up intersection of a 45 deg approach, r = 5


@pytest.fixture(autouse=True)
def _providers():
    from cnctwin.controller import comp, cycles, hooks
    hooks.load_plugins()
    comp.register()
    cycles.register()


def xy(result, machine=MILL):
    return [(c.kind.value[0], round(c.target["X"] * 1000, 3), round(c.target["Y"] * 1000, 3)) for c in moves(result)][1:]


def crc(body, settings=None, machine=MILL):
    return run(wrap(body), machine, tools=TOOLS, settings=settings or ControllerSettings())


SQUARE = "G90 G00 X-20. Y-20. Z5.\nG41 G01 X0 Y0 D1 F500.\nY30.\nX40.\nY0\nX0\nG40 G00 X-20. Y-20."


def test_outside_contour_inner_start_up_obtuse_corners_and_inner_cancel():
    r = crc(SQUARE)
    assert alarm(r) is None
    assert xy(r) == [("l", -5.0, round(R2, 3)), ("l", -5.0, 35.0), ("l", 45.0, 35.0), ("l", 45.0, -5.0),
                     ("l", round(R2, 3), -5.0), ("r", -20.0, -20.0)]
    assert r.controller.state.paths[0].comp_state is None


def test_inside_pocket_corners_are_intersections():
    r = crc("G90 G00 X20. Y10. Z5.\nG41 G01 X20. Y0 D1 F500.\nX40.\nY30.\nX0\nY0\nX20.\nG40 X20. Y10.")
    assert xy(r) == [("l", 25.0, 5.0), ("l", 35.0, 5.0), ("l", 35.0, 25.0), ("l", 5.0, 25.0), ("l", 5.0, 5.0),
                     ("l", 15.0, 5.0), ("l", 20.0, 10.0)]


def test_acute_outer_corner_gets_two_extension_vectors():
    r = crc("G90 G00 X-20. Y0 Z5.\nG42 G01 X0 Y0 D1 F500.\nX20. Y0\nX0 Y10.\nG40 X-20. Y10.")
    s5 = math.sqrt(5.0)
    v2 = (20.0 + s5 + 2 * s5, 2 * s5 - s5)                       # B's offset start pulled back by r along B
    assert xy(r) == [("l", 0.0, -5.0), ("l", 25.0, -5.0), ("l", round(v2[0], 3), round(v2[1], 3)),
                     ("l", round(s5, 3), round(10 + 2 * s5, 3)), ("l", -20.0, 10.0)]


def test_start_up_type_b_and_circular_outer_corners():
    r = crc("G90 G00 X-20. Y-20. Z5.\nG42 G01 X0 Y0 D1 F500.\nY30.\nX40.\nG40 X60.",
            ControllerSettings(cycle_parameters={"crc_startup_type": "B"}))
    h = 5.0 / math.sqrt(2.0)
    assert xy(r)[:2] == [("l", round(h, 3), round(-h, 3)), ("l", 5.0, round(-R2, 3))]
    r = crc(SQUARE.replace("G40 G00", "G40 G01"), ControllerSettings(cycle_parameters={"crc_outer_corner": "circular"}))
    arcs = [c for c in moves(r) if c.kind == Kind.ARC]
    assert [tuple(round(v * 1000, 3) for v in c.arc.centre[:2]) for c in arcs] == [(0.0, 30.0), (40.0, 30.0),
                                                                                   (40.0, 0.0)]
    assert all(c.arc.sweep_rad == pytest.approx(-math.pi / 2) and c.arc.radius_start_m == pytest.approx(0.005)
               for c in arcs)


def test_d_change_takes_effect_at_the_end_of_its_block_and_ij_vector():
    r = crc("G90 G00 X-20. Y-20. Z5.\nG41 G01 X0 Y0 D1 F500.\nY30.\nX40. D2\nY0\nG40 X60.")
    assert xy(r)[1:3] == [("l", -5.0, 35.0), ("l", 43.0, 33.0)]
    r = crc("G90 G00 X-20. Y-20. Z5.\nG41 G01 X0 Y0 I1. J0 D1 F500.\nY30.\nG40 X-20.")
    assert xy(r)[0] == ("l", 0.0, 5.0)


def test_g39_inserts_a_corner_arc_and_g40_alone_types_a_and_c():
    r = crc("G90 G00 X-20. Y-20. Z5.\nG41 G01 X0 Y0 D1 F500.\nY30.\nG39\nX40.\nG40 Y60.")
    arc = [c for c in moves(r) if c.kind == Kind.ARC][0]
    assert tuple(round(v * 1000, 3) for v in arc.arc.centre[:2]) == (0.0, 30.0)
    assert xy(r)[1:3] == [("l", -5.0, 30.0), ("a", 0.0, 35.0)]
    body = "G90 G00 X-20. Y-20. Z5.\nG41 G01 X0 Y0 D1 F500.\nY30.\nX40.\nG40\nG00 X60. Y60."
    assert xy(crc(body))[2:] == [("l", 40.0, 35.0), ("r", 60.0, 60.0)]
    assert xy(crc(body, ControllerSettings(cycle_parameters={"crc_startup_type": "C"})))[2:] == \
        [("l", 40.0, 35.0), ("l", 40.0, 30.0), ("r", 60.0, 60.0)]


def test_interference_and_start_up_alarms():
    r = crc("G90 G00 X0 Y10. Z5.\nG41 G01 X0 Y0 D1 F500.\nY-10.\nX6.\nY0\nG40 Y10.")
    a = alarm(r)
    assert a["code"] == "PS0041" and a["text"].startswith("INTERFERENCE")
    assert xy(r) == [("l", 5.0, 0.0)], "the alarm stops the tool before the block preceding the overcut"
    assert alarm(crc("G90 G00 X0 Y0 Z5.\nG41 G02 X10. Y0 R5. D1 F500."))["code"] == "PS0034"
    r = crc("G90 G00 X-10. Y-15. Z5.\nG41 G01 X-10. Y0 D3 F500.\nG03 X10. Y0 I10. J0\nG03 X30. Y0 I10. J0\nG40 G01 Y-15.")
    assert alarm(r)["code"] == "PS0033"
    assert alarm(crc("G90 G00 X0 Y0 Z5.\nG41 G01 X10. D1 F500.\nG18 X20."))["code"] == "PS0037"


def test_z_moves_between_blocks_keep_the_corner_and_no_lookahead_gives_normal_vectors():
    r = crc("G90 G00 X-20. Y-20. Z5.\nG41 G01 X0 Y0 D1 F500.\nZ-2.\nY30.\nX40.\nG40 X60.")
    assert xy(r)[:3] == [("l", -5.0, round(R2, 3)), ("l", -5.0, round(R2, 3)), ("l", -5.0, 35.0)]
    r = crc("G90 G00 X-20. Y-20. Z5.\nG41 G01 X0 Y0 D1 F500.\nY30.\nM08\nM09\nX40.\nG40 X60.")
    assert xy(r)[1] == ("l", -5.0, 30.0), "more than N - 2 blocks without movement: normal vector"


def test_mirror_image_and_rotation_keep_the_side_in_the_program_frame():
    r = crc("G90 G00 X-20. Y-20. Z5.\nG51.1 X0\nG41 G01 X0 Y0 D1 F500.\nY30.\nX40.\nG40 X60.\nG50.1 X0")
    assert xy(r)[1] == ("l", 5.0, 35.0)
    r = crc("G90 G00 X0 Y0 Z5.\nG68 X0 Y0 R90.\nG41 G01 X10. Y0 D1 F500.\nX20.\nG40 X30.\nG69")
    assert xy(r)[1] == ("l", -5.0, 20.0)


def test_haas_compensation_uses_the_same_rules():
    r = run(wrap(SQUARE), HAAS_MILL, tools=TOOLS)
    assert alarm(r) is None and xy(r)[1] == ("l", -5.0, 35.0)


def test_checkpoint_inside_a_compensated_contour_resumes_identically():
    small = copy.deepcopy(MILL)
    small["controller"]["lookahead_blocks"] = 2
    body = SQUARE.replace("X0\nG40", "X0\nY10.\nX5.\nG40")

    def drive(ctl, stop_after=None):
        out, tick = [], 0
        while True:
            out.extend(ctl.produce(0, tick))
            path = ctl.state.paths[0]
            if path.alarm is not None or (path.waiting is not None and path.waiting["kind"] == "end"):
                return out, tick
            if path.buffer:
                tick += 3000
                ctl.retire(0, 1, tick)
            else:
                tick += 5000
            if stop_after is not None and len(out) >= stop_after:
                return out, tick

    reference, _ = drive(controller(wrap(body), small, tools=TOOLS))
    first = controller(wrap(body), small, tools=TOOLS)
    part, tick = drive(first, stop_after=3)
    assert first.state.paths[0].comp_state and first.state.paths[0].comp_state["queue"]
    arrays, meta = first.state.to_arrays()
    blob = io.BytesIO()
    np.savez(blob, **arrays)
    restored = Controller(first.setup, ControllerState.from_arrays(dict(np.load(io.BytesIO(blob.getvalue()))),
                                                                   json.loads(json.dumps(meta))))
    rest = []
    while True:
        rest.extend(restored.produce(0, tick))
        path = restored.state.paths[0]
        if path.alarm is not None or (path.waiting is not None and path.waiting["kind"] == "end"):
            break
        if path.buffer:
            tick += 3000
            restored.retire(0, 1, tick)
        else:
            tick += 5000
    assert part + rest == reference
