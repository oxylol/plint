"""HEIDENHAIN TNC 640 fixed cycles: drilling, tapping, bore milling, centring, pockets and slots, coordinate
transformations (datum shift, mirroring, rotation, scaling in the TNC's order), tolerance, presetting."""

from __future__ import annotations

import math

import pytest

from cnctwin.controller.ir import Kind
from test_heidenhain import TOOL, path, run


def cycle(number: int, name: str, params: dict) -> str:
    lines = [f"CYCL DEF {number} {name} ~"] + [f"  Q{k}={v:+g} ;P ~" for k, v in params.items()]
    lines[-1] = lines[-1].rstrip(" ~")
    return "\n".join(lines)


def program(*blocks: str) -> str:
    """Klartext with continuation lines: the block numbers are given only to the first line of a block."""
    out, k = [], 0
    for block in ("BEGIN PGM T MM", TOOL.split("\n")[0], TOOL.split("\n")[1]) + blocks + ("END PGM T MM",):
        for j, line in enumerate(block.strip().splitlines()):
            out.append(f"{k} {line.strip()}" if j == 0 else "  " + line.strip())
        k += 1
    return "\n".join(out) + "\n"


DRILL = {200: 2, 201: -10, 206: 150, 202: 4, 210: 0, 203: 0, 204: 20, 211: 0}


def zs(res) -> list[tuple]:
    return [(c.kind.value, round(c.target["Z"] * 1000.0, 6)) for c in res.commands[0]
            if c.kind in (Kind.RAPID, Kind.LINEAR)]


def test_cycle_200_pecks_with_clearance_and_second_clearance():
    res = run(program("L X+10 Y+10 Z+50 R0 FMAX", cycle(200, "DRILLING", DRILL), "CYCL CALL"))
    assert res.alarms[0] is None
    assert zs(res)[2:] == [("rapid", 102.0), ("linear", 96.0), ("rapid", 102.0), ("rapid", 98.0), ("linear", 92.0),
                           ("rapid", 102.0), ("rapid", 94.0), ("linear", 90.0), ("rapid", 120.0)]


def test_depth_sign_error_and_zero_depth():
    assert run(program("L Z+50 R0 FMAX", cycle(200, "DRILLING", {**DRILL, 201: 5}), "CYCL CALL")).alarms[0][
        "code"] == "280-0435"
    res = run(program("L Z+50 R0 FMAX", cycle(200, "DRILLING", {**DRILL, 201: 0}), "CYCL CALL"))
    assert res.alarms[0] is None and len(zs(res)) == 2
    assert run(program("L Z+50 R0 FMAX", "CYCL CALL")).alarms[0]["code"] == "280-0403"


def test_cycles_201_202_203_205_206_207_208_209_240():
    common = {200: 2, 201: -10, 203: 0, 204: 0}
    blocks = [
        cycle(201, "REAMING", {**common, **{206: 100, 211: 0.5, 208: 300}}),
        cycle(202, "BORING", {**common, **{206: 100, 211: 0, 208: 300, 214: 1, 336: 90}}),
        cycle(203, "UNIVERSAL DRILLING", {**common, **{206: 100, 202: 4, 210: 0, 212: 1, 213: 2, 205: 2, 211: 0,
                                                         208: 500, 256: 0.2}}),
        cycle(205, "UNIVERSAL PECKING", {**common, **{206: 100, 202: 5, 212: 0, 205: 0, 258: 0.5, 259: 0.5, 257: 0,
                                                        256: 0.2, 211: 0, 379: 0, 253: 750, 208: 0}}),
        cycle(206, "TAPPING", {**common, **{206: 150, 211: 0}}),
        cycle(207, "RIGID TAPPING", {**common, **{239: 1.5}}),
        cycle(208, "BORE MILLING", {**common, **{206: 150, 334: 2, 335: 20, 342: 0, 351: 1}}),
        cycle(209, "TAPPING W/ CHIP BRKG", {**common, **{239: -1.0, 257: 4, 256: 0, 336: 0, 403: 2}}),
        cycle(240, "CENTERING", {**common, **{343: 0, 344: 0, 206: 100, 211: 0}}),
    ]
    for block in blocks:
        res = run(program("L X+0 Y+0 Z+50 R0 FMAX", block, "CYCL CALL"))
        assert res.alarms[0] is None, block.splitlines()[0]
        assert min(z for _, z in zs(res)) == pytest.approx(90.0), block.splitlines()[0]
    taps = run(program("L X+0 Y+0 Z+50 R0 FMAX", blocks[5], "CYCL CALL"))
    lin = [c for c in taps.commands[0] if c.kind == Kind.LINEAR]
    assert [c.modal["tap"]["phase"] for c in lin] == ["in", "out"] and lin[0].feed.value == pytest.approx(0.0015)
    chip = run(program("L X+0 Y+0 Z+50 R0 FMAX", blocks[7], "CYCL CALL"))
    lin = [c for c in chip.commands[0] if c.kind == Kind.LINEAR]
    assert [c.modal["tap"]["phase"] for c in lin] == ["in", "out", "in", "out", "in", "out"]
    assert lin[1].modal["tap"]["rpm"] == pytest.approx(2000.0) and lin[0].modal["tap"]["direction"] == -1
    helix = run(program("L X+0 Y+0 Z+50 R0 FMAX", blocks[6], "CYCL CALL"))
    arcs = [c for c in helix.commands[0] if c.kind == Kind.ARC]
    assert len(arcs) == 7 and all(abs(a.arc.radius_start_m - 0.005) < 1e-9 for a in arcs)   # 12 mm / 2 + closing circle


def test_cycl_call_pos_and_m99_m89():
    res = run(program("L X+0 Y+0 Z+50 R0 FMAX", cycle(200, "DRILLING", DRILL), "CYCL CALL POS X+30 Y+20 Z+0 FMAX",
                      "L X+40 Y+20 FMAX M99", "L X+50 FMAX M89", "L X+60 FMAX", cycle(200, "DRILLING", DRILL),
                      "L X+70 FMAX"))
    assert res.alarms[0] is None
    bottoms = [(m[1], m[2]) for m in path(res) if m[0] == "linear" and m[3] == 90.0]
    assert bottoms == [(30.0, 20.0), (40.0, 20.0), (50.0, 20.0), (60.0, 20.0)]


BACK = {200: 2, 249: 5, 250: 20, 251: 3.5, 252: 15, 253: 750, 254: 200, 255: 0, 203: 0, 204: 50, 214: 1, 336: 0}


def test_cycle_204_back_boring_sequence():
    res = run(program("L X+10 Y+10 Z+50 R0 FMAX", cycle(204, "BACK BORING", BACK), "CYCL CALL"))
    assert res.alarms[0] is None
    steps = []
    for c in res.commands[0]:
        if c.kind in (Kind.RAPID, Kind.LINEAR):
            feed = None if c.kind == Kind.RAPID else round(c.feed.value * 60000.0, 6)
            steps.append((c.kind.value, round(c.target["X"] * 1000, 6), round(c.target["Y"] * 1000, 6),
                          round(c.target["Z"] * 1000, 6), feed))
        elif c.kind == Kind.SPINDLE:
            steps.append(("spindle", c.spindle["direction"]))
    m3 = next(step[1] for step in steps if step[0] == "spindle")           # the direction M3 started
    cycle_part = steps[steps.index(("rapid", 10.0, 10.0, 150.0, None)) + 1:]
    assert m3 != 0 and cycle_part == [
        ("rapid", 10.0, 10.0, 102.0, None),               # set-up clearance above Q203 (tool length 100)
        ("spindle", 0),                                   # oriented stop Q336
        ("rapid", 6.5, 10.0, 102.0, None),                # off centre Q251 toward -X (Q214 = 1)
        ("linear", 6.5, 10.0, 63.0, 750.0),               # edge Q252 = 15 above the tip, Q200 under the underside
        ("linear", 10.0, 10.0, 63.0, 750.0),
        ("spindle", m3),
        ("linear", 10.0, 10.0, 70.0, 200.0),              # counterbore Q249 = 5 above the underside -20
        ("spindle", 0),
        ("linear", 6.5, 10.0, 70.0, 750.0),
        ("linear", 6.5, 10.0, 102.0, 750.0),
        ("rapid", 6.5, 10.0, 150.0, None),                # 2nd set-up clearance
        ("rapid", 10.0, 10.0, 150.0, None),
        ("spindle", m3)]
    assert run(program("L X+10 Y+10 Z+50 R0 FMAX", cycle(204, "BACK BORING", {**BACK, 214: 0}),
                       "CYCL CALL")).alarms[0]["code"] == "280-03F9"


def test_cycl_call_pat_expands_centre_based_cycles_at_each_point():
    bore = cycle(202, "BORING", {200: 2, 201: -10, 206: 100, 211: 0, 208: 300, 203: 0, 204: 20, 214: 3, 336: 0})
    res = run(program("L X+0 Y+0 Z+50 R0 FMAX", bore, "PATTERN DEF ~\nPOS1 (X+30 Y+0 Z+0) ~\nPOS2 (X+60 Y+5 Z+0)",
                      "CYCL CALL PAT FMAX", "CYCL CALL POS X+90 Y+0 Z+0 FMAX"))
    assert res.alarms[0] is None
    disengaged = [(m[1], m[2]) for m in path(res) if m[0] == "rapid" and m[3] == 90.0]
    # Q214 = 3: off the bottom toward +X by the disengaging distance, at every hole of its own
    assert [round(x - round(x), 6) > 0 for x, _ in disengaged] == [True, True, True]
    assert [(round(x), y) for x, y in disengaged] == [(30, 0.0), (60, 5.0), (90, 0.0)]


def test_pattern_def_row_and_circle():
    res = run(program("L X+0 Y+0 Z+50 R0 FMAX", cycle(200, "DRILLING", DRILL),
                      "PATTERN DEF ~\nROW1 (X+0 Y+0 D+10 NUM3 ROT+0 Z+0) ~\nCIRC1 (X+0 Y+0 D+40 START+90 NUM2 Z-1)",
                      "CYCL CALL PAT FMAX"))
    assert res.alarms[0] is None
    bottoms = [(m[1], m[2], m[3]) for m in path(res) if m[0] == "linear" and m[3] in (90.0, 89.0)]
    assert [(b[0], b[1]) for b in bottoms] == [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (0.0, 20.0), (0.0, -20.0)]
    assert [b[2] for b in bottoms] == [90.0, 90.0, 90.0, 89.0, 89.0]      # CIRC Z-1 lowers the surface


def test_pattern_def_pos_incremental_sheared_pat_and_frame():
    res = run(program("L X+5 Y+5 Z+50 R0 FMAX", cycle(200, "DRILLING", DRILL),
                      "PATTERN DEF ~\nPOS1 (X+0 Y+0 Z+0) ~\nPOS2 (IX+5 IY+2 Z+0) ~\n"
                      "PAT1 (X+20 Y+0 DX+10 DY+10 NUMX2 NUMY2 ROT+0 ROTX+0 ROTY+90 Z+0) ~\n"
                      "FRAME1 (X+50 Y+0 DX+10 DY+10 NUMX3 NUMY3 ROT+0 ROTX+0 ROTY+0 Z+0)",
                      "CYCL CALL PAT FMAX"))
    assert res.alarms[0] is None
    bottoms = [(m[1], m[2]) for m in path(res) if m[0] == "linear" and m[3] == 90.0]
    # POS2 is 5 / 2 from POS1; ROTY+90 turns the secondary axis onto -X (a fully sheared row);
    # FRAME1 drops the inner point (60, 10)
    assert bottoms == [(0.0, 0.0), (5.0, 2.0), (20.0, 0.0), (30.0, 0.0), (10.0, 0.0), (20.0, 0.0),
                       (50.0, 0.0), (60.0, 0.0), (70.0, 0.0), (50.0, 10.0), (70.0, 10.0),
                       (50.0, 20.0), (60.0, 20.0), (70.0, 20.0)]
    bad = run(program("PATTERN DEF ~\nROW1 (IX+0 Y+0 D+10 NUM3 ROT+0 Z+0)"))
    assert bad.alarms[0] is not None and bad.alarms[0]["code"] == "160-0003"


def test_axis_specific_scaling_refuses_circles_with_unequal_factors():
    circle = ("L X+0 Y+0 Z-1 R0 F100", "CC X+5 Y+0", "C X+10 Y+0 DR-")
    unequal = run(program("CYCL DEF 26.0 AXIS-SPECIFIC SCALING", "CYCL DEF 26.1 X2 Y1", *circle))
    assert unequal.alarms[0]["code"] == "TW0018"
    equal = run(program("CYCL DEF 26.0 AXIS-SPECIFIC SCALING", "CYCL DEF 26.1 X2 Y2", *circle))
    assert equal.alarms[0] is None


def pocket(number: int, name: str, extra: dict) -> str:
    base = {215: 0, 368: 0.5, 207: 400, 351: 1, 201: -6, 202: 3, 369: 0.2, 206: 100, 338: 0, 200: 2, 203: 0,
            204: 10, 370: 1, 366: 0, 385: 300, 439: 0}
    base.update(extra)
    return cycle(number, name, base)


def test_cycle_251_252_253_rough_and_finish():
    rect = run(program("L X+50 Y+50 Z+20 R0 FMAX", pocket(251, "RECTANGULAR POCKET",
                                                          {218: 40, 219: 20, 220: 5, 224: 0, 367: 0}), "CYCL CALL"))
    assert rect.alarms[0] is None
    moves = path(rect)
    finish = [m for m in moves if m[0] in ("linear", "arc") and m[3] == 94.0]
    assert max(m[1] for m in finish) == 65.0 and min(m[2] for m in finish) == 45.0
    rough = [m for m in moves if m[0] == "linear" and m[3] == 97.1]
    assert rough and max(m[1] for m in rough) <= 64.5
    circle = run(program("L X+50 Y+50 Z+20 R0 FMAX", pocket(252, "CIRCULAR POCKET", {223: 30}), "CYCL CALL"))
    assert circle.alarms[0] is None
    radii = {round(c.arc.radius_start_m * 1000, 6) for c in circle.commands[0] if c.kind == Kind.ARC}
    assert 10.0 in radii and 9.5 in radii
    slot = run(program("L X+50 Y+50 Z+20 R0 FMAX", pocket(253, "SLOT MILLING", {218: 40, 219: 12, 374: 90, 367: 0}),
                       "CYCL CALL"))
    assert slot.alarms[0] is None
    ys = [m[2] for m in path(slot) if m[0] in ("linear", "arc")]
    assert max(ys) == pytest.approx(65.0) and min(ys) == pytest.approx(35.0)


def test_pocket_errors():
    assert run(program("L X+0 Y+0 Z+20 R0 FMAX", pocket(253, "SLOT MILLING", {218: 10, 219: 40, 374: 0, 367: 0}),
                       "CYCL CALL")).alarms[0]["code"] == "280-0408"
    assert run(program("L X+0 Y+0 Z+20 R0 FMAX", pocket(252, "CIRCULAR POCKET", {223: 30, 366: 7}),
                       "CYCL CALL")).alarms[0]["code"] == "280-0450"


def test_coordinate_transformations_in_the_tnc_order():
    res = run(program("L X+0 Y+0 Z+0 R0 FMAX",
                      "CYCL DEF 10.0 ROTATION", "CYCL DEF 10.1 ROT+90",
                      "CYCL DEF 7.0 DATUM SHIFT", "CYCL DEF 7.1 X+10", "CYCL DEF 7.2 Y+20",
                      "L X+5 Y+0 FMAX",
                      "CYCL DEF 8.0 MIRRORING", "CYCL DEF 8.1 X",
                      "L X+5 Y+0 FMAX",
                      "CYCL DEF 11.0 SCALING", "CYCL DEF 11.1 SCL2",
                      "L X+5 Y+0 FMAX",
                      "CYCL DEF 26.0 AXIS-SPECIFIC SCALING", "CYCL DEF 26.1 X1 Y1 CCX+0 CCY+0",
                      "CYCL DEF 8.0 MIRRORING", "CYCL DEF 8.1",
                      "CYCL DEF 10.0 ROTATION", "CYCL DEF 10.1 IROT-90",
                      "CYCL DEF 7.0 DATUM SHIFT", "CYCL DEF 7.1 IX+1",
                      "L X+0 Y+0 FMAX"))
    assert res.alarms[0] is None
    # x_machine = datum + M R S x: rotation 90 then the datum (10, 20); mirroring X acts on the rotated point
    # (0, 5), scaling 2 before the rotation; then everything reset except the datum, shifted by IX+1
    assert path(res)[2:] == [("rapid", 10.0, 25.0, 100.0), ("rapid", 10.0, 25.0, 100.0), ("rapid", 10.0, 30.0, 100.0),
                             ("rapid", 11.0, 20.0, 100.0)]


def test_tolerance_cycle_32_and_preset_247():
    res = run(program("L X+0 Y+0 Z+0 R0 FMAX", "CYCL DEF 32.0 TOLERANCE", "CYCL DEF 32.1 T0.02",
                      cycle(247, "PRESETTING", {339: 2}), "L X+0 Y+0 FMAX"),
              offsets={"P2": [0.1, 0.2, 0.0]})
    assert res.alarms[0] is None
    mode = [c for c in res.commands[0] if c.kind == Kind.MODE and "tolerance_cycle" in c.modal][0]
    assert mode.smoothing.tolerance_m == pytest.approx(2e-5)
    assert path(res)[-1] == ("rapid", 100.0, 200.0, 100.0)
    assert run(program(cycle(247, "PRESETTING", {339: 7}))).alarms[0]["code"] == "280-03F1"


def test_dwell_and_orientation_cycles_act_when_defined():
    res = run(program("L X+0 Y+0 Z+0 R0 FMAX", "CYCL DEF 9.0 DWELL TIME", "CYCL DEF 9.1 DWELL 1.5",
                      "CYCL DEF 13.0 ORIENTATION", "CYCL DEF 13.1 ANGLE 45"))
    assert res.alarms[0] is None
    assert [c.dwell_s for c in res.commands[0] if c.kind == Kind.DWELL] == [1.5]
    assert [c.spindle["orient_rad"] for c in res.commands[0] if c.kind == Kind.SPINDLE][-1] == pytest.approx(math.pi / 4)
