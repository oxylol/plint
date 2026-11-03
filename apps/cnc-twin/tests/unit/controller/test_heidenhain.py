"""HEIDENHAIN TNC 640 dialects (Klartext and ISO): paths, Q parameters, program flow, approach and
departure, FK, tilted planes, TCPM, NC errors (docs/dialects.md "Heidenhain TNC")."""

from __future__ import annotations

import copy
import math

import pytest

from cnctwin.controller import Controller, ControllerSetup, ProgramSource, ToolOffsetInit, dry_run
from cnctwin.controller.alarms import load_controller_db
from cnctwin.controller.ir import Kind, Space
from harness import MILL, UMC

MOTION = (Kind.RAPID, Kind.LINEAR, Kind.ARC)
TNC_DB = {
    "format": "cnctwin-controller", "version": 1, "id": "fixture", "dialect": "heidenhain_klartext", "name": "fixture",
    "alarms": [{"code": "1A0-0084", "text": "Circle incorrectly programmed"},
               {"code": "160-001B", "text": "Label number not found"},
               {"code": "280-03E8", "text": "Spindle must be turning"},
               {"code": "280-0064", "text": "FN 14: error code %1"}],
}


def machine(base=MILL, dialect: str = "heidenhain_klartext") -> dict:
    m = copy.deepcopy(base)
    m["controller"]["dialect"] = dialect
    m["controller"]["alarms"] = "fixture"
    for axis in m["axes"].values():
        if axis.get("type") == "linear":
            axis["travel_mm"] = [-2000, 2000]
    return m


def controller(program: str, base=MILL, *, extra=(), tools=None, offsets=None, db=TNC_DB,
               dialect="heidenhain_klartext", name="TEST.H") -> Controller:
    sources = [ProgramSource(name, program)] + [ProgramSource(n, t, path=-1, main=False) for n, t in extra]
    setup = ControllerSetup(machine(base, dialect), sources, controller_db=db, db_id="fixture",
                            work_offsets_m=[offsets or {}],
                            tools=tools if tools is not None else [ToolOffsetInit(1, length_m=0.1, radius_m=0.005),
                                                                   ToolOffsetInit(2, length_m=0.05, radius_m=0.003)])
    return Controller(setup)


def run(program: str, base=MILL, **kwargs):
    return dry_run(controller(program, base, **kwargs))


def path(res, p: int = 0) -> list[tuple]:
    return [(c.kind.value,) + tuple(round(c.target[a] * 1000.0, 6) for a in ("X", "Y", "Z") if a in c.target)
            for c in res.commands[p] if c.kind in MOTION]


def pgm(body: str, name: str = "TEST") -> str:
    lines = ["BEGIN PGM %s MM" % name] + [line.strip() for line in body.strip().splitlines()] + ["END PGM %s MM" % name]
    return "\n".join(f"{k} {line}" for k, line in enumerate(lines)) + "\n"


TOOL = "TOOL CALL 1 Z S1000 F300\nL Z+0 R0 FMAX M3"


# ====================================================================== paths

def test_lines_circles_polar_and_incremental():
    res = run(pgm(TOOL + """
L X+0 Y+0 Z-1 R0 F200
L IX+10
CC X+10 Y+10
C X+20 Y+10 DR-
CR X+30 Y+0 R+10 DR+
CT X+40 Y+10
CC X+40 Y+0
LP PR+10 PA+0
CP IPA+90 DR+
L X+10 Y+10 FMAX
L Y+20"""))
    assert res.alarms[0] is None
    moves = path(res)
    assert moves[2:5] == [("linear", 10.0, 0.0, 99.0), ("arc", 20.0, 10.0, 99.0), ("arc", 30.0, 0.0, 99.0)]
    arc = [c for c in res.commands[0] if c.kind == Kind.ARC]
    assert round(arc[0].arc.sweep_rad, 9) == round(-1.5 * math.pi, 9)
    assert [round(v * 1000, 6) for v in arc[2].arc.centre[:2]] == [30.0, 10.0]     # CT continues the CR arc
    assert [round(v * 1000, 6) for v in arc[3].arc.centre[:2]] == [40.0, 0.0]
    assert moves[6] == ("linear", 50.0, 0.0, 99.0) and moves[7] == ("arc", 40.0, 10.0, 99.0)
    assert moves[8][0] == "rapid" and moves[9] == ("linear", 10.0, 20.0, 99.0)
    feeds = [round(c.feed.value * 60000, 6) for c in res.commands[0] if c.kind == Kind.LINEAR]
    assert feeds[-1] == 200.0


def test_circle_end_point_error_1a0_0084_and_missing_pole():
    bad = run(pgm(TOOL + "\nL X+0 Y+0 R0 F100\nCC X+10 Y+0\nC X+20.5 Y+0 DR+"))
    assert bad.alarms[0]["code"] == "1A0-0084" and bad.alarms[0]["text"] == "Circle incorrectly programmed"
    assert run(pgm(TOOL + "\nL X+0 Y+0 R0 F100\nC X+10 Y+0 DR+")).alarms[0]["code"] == "1A0-000F"
    assert run(pgm(TOOL + "\nL X+0 Y+0 R0 F100\nCC X+5 Y+0\nC X+10 Y+0")).alarms[0]["code"] == "1A0-000C"


def test_fauto_fu_fz_and_m91_m92():
    res = run(pgm("""TOOL CALL 1 Z S1000 F250
L X+0 Y+0 Z+0 R0 FMAX M3
L X+10 FAUTO
L X+20 FU0.1
L Z+5 R0 FMAX M91"""))
    assert res.alarms[0] is None
    linear = [c for c in res.commands[0] if c.kind == Kind.LINEAR]
    assert round(linear[0].feed.value * 60000, 6) == 250.0
    assert linear[1].feed.mode.value == "per_rev" and linear[1].feed.value == pytest.approx(1e-4)
    assert path(res)[-1][3] == 105.0
    assert run(pgm(TOOL + "\nL X+10 FZ0.05")).alarms[0]["code"] == "160-00E0"


def test_m140_retracts_along_the_tool_axis():
    res = run(pgm(TOOL + "\nL X+10 Y+0 Z-5 R0 F200\nL X+20 F100 M140 MB 50 F750\nL X+30 FMAX M140 MB MAX"))
    assert res.alarms[0] is None
    moves = [(c.kind.value, round(c.target["X"] * 1000, 6), round(c.target["Z"] * 1000, 6),
              None if c.kind == Kind.RAPID else round(c.feed.value * 60000.0, 6))
             for c in res.commands[0] if c.kind in MOTION]
    assert moves[-4:] == [("linear", 20.0, 95.0, 100.0), ("linear", 20.0, 145.0, 750.0),   # after the move: +50
                          ("rapid", 30.0, 145.0, None), ("rapid", 30.0, 2000.0, None)]     # to the end of travel
    assert run(pgm(TOOL + "\nL X+10 M140 MB")).alarms[0]["code"] == "160-0003"
    assert run(pgm(TOOL + "\nL X+10 FMAX M140 MB 5000")).alarms[0]["code"] == "1A1-0010"


def test_rnd_and_chf_between_elements():
    res = run(pgm(TOOL + """
L X+0 Y+0 Z-1 R0 F200
L X+50
RND R8
L Y+50
CHF 5
L X+0"""))
    assert res.alarms[0] is None
    moves = path(res)
    assert moves[2] == ("linear", 42.0, 0.0, 99.0)
    arc = [c for c in res.commands[0] if c.kind == Kind.ARC][0]
    assert arc.arc.radius_start_m == pytest.approx(0.008)
    assert ("linear", 50.0, 45.0, 99.0) in moves and ("linear", 45.0, 50.0, 99.0) in moves


# ====================================================================== tool and spindle

def test_tool_call_length_axis_deltas_and_coolant_m13():
    res = run(pgm("""TOOL CALL 2 Z S2000 DL+0.5 DR-0.1
L X+0 Y+0 Z+0 R0 FMAX M13
L Z+10 FMAX M9
L Z+20 FMAX M5"""))
    assert res.alarms[0] is None
    assert [m[3] for m in path(res)] == [50.5, 60.5, 70.5]
    kinds = [c.kind for c in res.commands[0]]
    assert kinds[:3] == [Kind.TOOL_CHANGE, Kind.SPINDLE, Kind.COOLANT]
    assert controller(pgm("TOOL CALL 9 Z")).dialect is not None
    assert run(pgm("TOOL CALL 9 Z")).alarms[0]["code"] == "160-00C5"


# ====================================================================== Q parameters and program flow

def test_q_parameters_fn_functions_formulas_and_jumps():
    res = run(pgm(TOOL + """
FN 0: Q1 = +3
FN 1: Q2 = +Q1 + +2
FN 3: Q3 = +Q2 * +Q1
FN 4: Q4 = +Q3 DIV +5
FN 5: Q5 = SQRT +16
FN 8: Q6 = +3 LEN +4
FN 13: Q7 = +1 ANG +1
Q8 = SQ(Q1) + SIN(30) * 2 + INT(2.7) + 7 % 4 + 2 ^ 3
QL1 = Q8 + 1
QR2 = QL1 * 2
FN 18: SYSREAD Q9 = ID20 NR1
FN 12: IF +Q1 LT +5 GOTO LBL 7
L X+999
LBL 7
L X+Q2 Y+Q6 Z+QR2 R0 FMAX"""))
    assert res.alarms[0] is None
    assert path(res)[-1] == ("rapid", 5.0, 5.0, 148.0)
    q = res.controller.state.paths[0].variables["Q"]
    assert q[3:10] == [15.0, 3.0, 4.0, 5.0, 45.0, 23.0, 1.0]


def test_string_parameters_and_string_functions():
    res = run(pgm(TOOL + """
QS1 = "Part"
QS2 = "-A7"
QS3 = STRING FORMULA QS1 || QS2 || "_x"
Q1 = STRLEN ( SRC_QS3 )
Q2 = 2.5
QS4 = TOCHAR ( DAT+Q2 DECIMALS2 )
QS5 = SUBSTR ( SRC_QS3 BEG4 LEN3 )
QS6 = "12.75"
Q3 = TONUMB ( SRC_QS6 ) * 2
Q4 = INSTR ( SRC_QS3 SEA_QS2 BEG0 )
Q5 = STRCOMP ( SRC_QS1 SEA_QS2 )
Q6 = STRLEN ( SRC_QS99 )
Q7 = INSTR ( SRC_QS3 SEA_QS6 )"""))
    assert res.alarms[0] is None
    state = res.controller.state.paths[0]
    assert {k: state.sys["qs"][k] for k in ("3", "4", "5")} == {"3": "Part-A7_x", "4": "2.50", "5": "-A7"}
    assert state.variables["Q"][1:8] == [9.0, 2.5, 25.5, 4.0, 1.0, -1.0, 9.0]
    assert run(pgm(TOOL + '\nQS1 = "ABC"\nQ1 = TONUMB ( SRC_QS1 )')).alarms[0]["code"] == "160-0028"
    assert run(pgm(TOOL + '\nQS1 = "ABC"\nQS2 = SUBSTR ( SRC_QS1 BEG2 LEN5 )')).alarms[0]["code"] == "160-0028"


def test_undefined_q_parameter_drops_the_axis_and_fails_in_calculations():
    res = run(pgm(TOOL + "\nQ1 = SET UNDEFINED\nL X+5 Y+Q1 R0 FMAX"))
    assert res.alarms[0] is None and path(res)[-1][1:3] == (5.0, 0.0)
    assert run(pgm(TOOL + "\nQ1 = SET UNDEFINED\nQ2 = Q1 + 1")).alarms[0]["code"] == "160-0028"
    assert run(pgm(TOOL + "\nQ105 = 1")).alarms[0]["code"] == "160-0028"


def test_labels_section_repeat_subprogram_and_program_call():
    res = run(pgm(TOOL + """
L X+0 Y+0 R0 FMAX
Q1 = 0
LBL 1
Q1 = Q1 + 1
L IX+1 FMAX
CALL LBL 1 REP 2
CALL LBL 5
CALL PGM SUB
L Z+50 FMAX
M30
LBL 5
L IY+10 FMAX
LBL 0"""), extra=[("SUB.H", pgm("L IY+5 R0 FMAX", "SUB"))])
    assert res.alarms[0] is None
    assert path(res)[2:] == [("rapid", 1.0, 0.0, 100.0), ("rapid", 2.0, 0.0, 100.0), ("rapid", 3.0, 0.0, 100.0),
                             ("rapid", 3.0, 10.0, 100.0), ("rapid", 3.0, 15.0, 100.0), ("rapid", 3.0, 15.0, 150.0)]


def test_sel_pgm_and_program_cycles_cycl_def_12_and_sel_cycle():
    hole = ("HOLE.H", pgm("L IZ-1 F100\nL IZ+1 FMAX", "HOLE"))
    res = run(pgm(TOOL + '\nSEL PGM "HOLE"\nL X+3 FMAX\nCALL SELECTED PGM'), extra=[hole])
    assert res.alarms[0] is None and [m[1:] for m in path(res)][-2:] == [(3.0, 0.0, 99.0), (3.0, 0.0, 100.0)]
    res = run(pgm(TOOL + "\nCYCL DEF 12.0 PGM CALL\nCYCL DEF 12.1 PGM TNC:\\HOLE.H\nL X+5 FMAX M99\n"
                         "L X+10 FMAX M89\nL X+15 FMAX"), extra=[hole])
    assert res.alarms[0] is None
    bottoms = [m[1] for m in path(res) if m[3] == 99.0]
    assert bottoms == [5.0, 10.0, 15.0]          # M99 once, M89 after every positioning (not inside HOLE)
    res = run(pgm(TOOL + "\nSEL CYCLE HOLE\nL X+7 FMAX\nCYCL CALL"), extra=[hole])
    assert res.alarms[0] is None and [m[1] for m in path(res) if m[3] == 99.0] == [7.0]
    assert run(pgm(TOOL + "\nSEL CYCLE HOLE\nCYCL CALL PAT"), extra=[hole]).alarms[0]["code"] == "TW0012"
    assert run(pgm(TOOL + "\nCALL SELECTED PGM")).alarms[0]["code"] == "201-0804"


def test_program_flow_errors():
    assert run(pgm(TOOL + "\nCALL LBL 9")).alarms[0]["code"] == "160-001B"
    assert run(pgm(TOOL + "\nCALL LBL 0")).alarms[0]["code"] == "160-0018"
    assert run(pgm(TOOL + "\nCALL PGM NOPE")).alarms[0]["code"] == "201-0804"
    assert run(pgm(TOOL + "\nCALL PGM SUB"), extra=[("SUB.H", pgm("M30", "SUB"))]).alarms[0]["code"] == "160-0160"
    assert run(pgm(TOOL + "\nM77")).alarms[0]["code"] == "160-00C3"
    assert run(pgm(TOOL + "\nL X+1 X+2 FMAX")).alarms[0]["code"] == "1A0-0008"
    assert run(pgm(TOOL + "\nFOO")).alarms[0]["code"] == "160-0003"


def test_fn14_errors_map_to_the_predefined_numbers():
    known = run(pgm(TOOL + "\nFN 14: ERROR = 1000"))
    assert known.alarms[0]["code"] == "280-03E8" and known.alarms[0]["text"] == "Spindle must be turning"
    other = run(pgm(TOOL + "\nFN 14: ERROR = 7"))
    assert other.alarms[0]["code"] == "280-0064" and other.alarms[0]["text"] == "FN 14: error code 7"


def test_fn18_positions_tool_data_and_fn17_tool_write():
    res = run(pgm(TOOL + """
L X+5 Y+6 R0 FMAX
FN 18: SYSREAD Q1 = ID270 NR1 IDX1
FN 18: SYSREAD Q2 = ID50 NR2 IDX1
FN 17: SYSWRITE ID50 NR1 IDX1 = +80
TOOL CALL 1 Z S1000
L X+Q1 Y+Q2 Z+0 FMAX"""))
    assert res.alarms[0] is None
    assert path(res)[-1] == ("rapid", 5.0, 5.0, 80.0)


# ====================================================================== approach and departure (core cutter compensation)

def test_appr_lt_and_dep_lt_with_radius_compensation():
    res = run(pgm(TOOL + """
L X-20 Y-20 Z-1 R0 FMAX
APPR LT X+0 Y+0 LEN10 RL F300
L X+50
L Y+30
DEP LT LEN10
L Z+50 R0 FMAX"""))
    assert res.alarms[0] is None
    moves = path(res)
    assert moves[2:4] == [("linear", -10.0, 5.0, 99.0), ("linear", 0.0, 5.0, 99.0)]
    assert moves[4] == ("linear", 45.0, 5.0, 99.0)
    assert moves[-2][1:3] == (50.0, 40.0)


def test_appr_ln_perpendicular_and_appr_ct_arc():
    ln = run(pgm(TOOL + "\nL X+0 Y+30 Z-1 R0 FMAX\nAPPR LN X+0 Y+0 LEN10 RL F300\nL X+50\nDEP LN LEN5\nL Z+50 R0 FMAX"))
    assert ln.alarms[0] is None
    moves = path(ln)
    assert moves[2] == ("linear", 0.0, 15.0, 99.0)       # PH = PA + (LEN + r) n, reached without compensation
    assert moves[4][1:3] == (45.0, 5.0)                  # then on the compensated contour (tool left)
    ct = run(pgm(TOOL + "\nL X-30 Y+30 Z-1 R0 FMAX\nAPPR CT X+0 Y+0 CCA90 R+10 RL F300\nL X+50\nDEP CT CCA90 R+10"))
    assert ct.alarms[0] is None
    arc = [c for c in ct.commands[0] if c.kind == Kind.ARC][0]
    assert arc.arc.sweep_rad == pytest.approx(math.pi / 2)
    assert run(pgm(TOOL + "\nAPPR LN X+0 Y+0 LEN10 R0 F300\nL X+50")).alarms[0]["code"] == "1A0-0071"


# ====================================================================== FK free contour programming

def test_fk_determined_elements_and_fselect():
    res = run(pgm(TOOL + """
L X+0 Y+0 Z-1 R0 F200
FPOL X+0 Y+0
FL X+20 Y+0
FLT LEN10
FCT DR+ R10 Y+20
FL LEN10 X+20
FSELECT 2
FL AN+180 X+0"""))
    assert res.alarms[0] is None
    moves = path(res)
    assert moves[2:5] == [("linear", 20.0, 0.0, 99.0), ("linear", 30.0, 0.0, 99.0), ("arc", 30.0, 20.0, 99.0)]
    assert moves[5][1] == 20.0 and moves[5][2] == pytest.approx(20.0 + math.sqrt(100 - 100), abs=1e-3)
    assert moves[6] == ("linear", 0.0, moves[5][2], 99.0)


def test_fk_undetermined_and_several_solutions_without_fselect():
    assert run(pgm(TOOL + "\nL X+0 Y+0 Z-1 R0 F200\nFL X+20")).alarms[0]["code"] == "TW0040"
    assert run(pgm(TOOL + "\nL X+0 Y+0 Z-1 R0 F200\nFL LEN10 X+5")).alarms[0]["code"] == "TW0040"
    assert run(pgm(TOOL + "\nL X+0 Y+0 Z-1 R0 F200\nFL X+20 Y+0 AN+90")).alarms[0]["code"] == "TW0041"


def test_fk_closed_contour_clsd_ends_at_the_start_point():
    res = run(pgm(TOOL + """
L X+0 Y+0 Z-1 R0 F200
FL X+20 Y+0 CLSD+
FL Y+20 AN+90
FC DR+ R10 X+0 Y+20
FL CLSD-"""))
    assert res.alarms[0] is None
    assert path(res)[-3:] == [("linear", 20.0, 20.0, 99.0), ("arc", 0.0, 20.0, 99.0), ("linear", 0.0, 0.0, 99.0)]
    semicircle = [c for c in res.commands[0] if c.kind == Kind.ARC][0]
    assert [round(v * 1000, 6) for v in semicircle.arc.centre[:2]] == [10.0, 20.0]     # one centre: no FSELECT
    assert run(pgm(TOOL + "\nL X+0 Y+0 Z-1 R0 F200\nFL X+20 Y+0 CLSD-")).alarms[0]["code"] == "TW0041"


# ====================================================================== tilted plane and TCPM

FIVE = [ToolOffsetInit(1, length_m=0.1, radius_m=0.005)]


def test_endless_rotary_axis_moves_from_its_shown_position_and_m126_takes_the_shortest_way():
    res = run(pgm("TOOL CALL 1 Z S1000\nL Z+50 R0 FMAX M3\nL C+350 FMAX\nL C+10 FMAX M126\nL C+190 FMAX\n"
                  "L C+10 FMAX M127\nL C+350 FMAX\nL IC-370 FMAX\nL C+10 FMAX"), UMC, tools=FIVE)
    assert res.alarms[0] is None
    c = [round(math.degrees(cmd.target["C"]), 6) for cmd in res.commands[0] if cmd.kind == Kind.RAPID][1:]
    # 350; M126: +20, +180 (a tie turns positive); M127: from the shown 190 to 10, from 10 to 350; IC; 340 -> 10
    assert c == [350.0, 370.0, 550.0, 370.0, 710.0, 340.0, 10.0]


def test_plane_spatial_turn_with_seq_and_q120():
    res = run(pgm("""TOOL CALL 1 Z S1000
L Z+50 R0 FMAX M3
PLANE SPATIAL SPA+0 SPB+30 SPC+0 TURN FMAX SEQ+
L X+0 Y+0 Z+10 FMAX
Q1 = Q121"""), UMC, tools=FIVE)
    assert res.alarms[0] is None
    swivel = [c for c in res.commands[0] if c.kind == Kind.RAPID and c.space == Space.MACHINE][-1]
    assert math.degrees(swivel.target["B"]) == pytest.approx(30.0)
    tip = [c for c in res.commands[0] if c.space == Space.TIP][0]
    assert [round(v * 1000, 6) for v in tip.tip_target_m] == [5.0, 0.0, 8.660254]
    assert res.controller.state.paths[0].variables["Q"][1] == pytest.approx(30.0)


def test_plane_variants_agree_and_reset():
    from cnctwin.controller.dialects.heidenhain.plane import rotation

    ctl = controller(pgm("L X+0 R0 FMAX"), UMC, tools=FIVE)
    h = ctl.executors[0].handler
    spatial = rotation(h, "SPATIAL", {"SPB": 30.0})
    projected = rotation(h, "PROJECTED", {"PROPR": 30.0})
    euler = rotation(h, "EULER", {"EULPR": 90.0, "EULNU": 30.0, "EULROT": -90.0})
    vector = rotation(h, "VECTOR", {"BX": math.cos(math.radians(30)), "BZ": -0.5, "NX": 0.5, "NZ": math.sqrt(0.75)})
    points = rotation(h, "POINTS", {"P2X": math.cos(math.radians(30)), "P2Z": -0.5, "P3Y": 1.0})
    for other in (projected, euler, vector, points):
        assert other == pytest.approx(spatial, abs=1e-12)
    res = run(pgm("""TOOL CALL 1 Z S1000
L Z+50 R0 FMAX M3
PLANE SPATIAL SPA+0 SPB+30 SPC+0 MOVE FMAX SEQ+
PLANE RESET TURN FMAX
L X+0 Y+0 Z+10 FMAX"""), UMC, tools=FIVE)
    assert res.alarms[0] is None
    last = [c for c in res.commands[0] if c.kind == Kind.RAPID][-1]
    assert last.space == Space.MACHINE and math.degrees(last.target["B"]) == pytest.approx(0.0, abs=1e-9)


def test_m128_keeps_the_tool_tip_and_m129_ends_it():
    res = run(pgm("""TOOL CALL 1 Z S1000
L Z+50 R0 FMAX M3
M128
L X+10 Y+0 Z+0 B+20 C+0 F500
M129
L B+0 FMAX"""), UMC, tools=FIVE)
    assert res.alarms[0] is None
    tip = [c for c in res.commands[0] if c.space == Space.TIP]
    assert len(tip) == 1 and tip[0].modal["tip_interpolation"] == "tip"
    assert res.commands[0][-2].space == Space.MACHINE
    function = "TOOL CALL 1 Z S1000\nL Z+50 R0 FMAX M3\nFUNCTION TCPM {}\nL X+10 B+20 F500\nFUNCTION RESET TCPM"
    assert run(pgm(function.format("F TCP AXIS POS PATHCTRL AXIS")), UMC, tools=FIVE).alarms[0] is None
    for unmodelled in ("F CONT", "AXIS SPAT", "PATHCTRL VECTOR"):
        assert run(pgm(function.format(unmodelled)), UMC, tools=FIVE).alarms[0]["code"] == "TW0012"


# ====================================================================== ISO

ISO = """%ISO1 G71 *
N10 T1 G17 S1000 *
N20 G00 G90 X+0 Y+0 Z+5 M03 *
N30 G01 Z-1 F200 *
N40 X+10 Y+0 *
N50 I+10 J+10 *
N60 G02 X+20 Y+10 *
N70 G03 X+30 Y+0 R+10 *
N80 D00 Q1 P01 +0 *
N90 G98 L1 *
N100 D01 Q1 P01 +Q1 P02 +1 *
N110 G91 G01 X+1 *
N120 G90 *
N130 D12 P01 +Q1 P02 +3 P03 1 *
N140 G54 X+5 Y+5 *
N150 G00 X+0 Y+0 Z+50 *
N160 G200 BOHREN Q200=2 Q201=-5 Q206=100 Q202=5 Q210=0 Q203=+0 Q204=10 Q211=0 *
N170 G79 *
N180 G28 X *
N190 G00 X+10 *
N200 G73 G90 H+90 *
N210 G28 *
N220 G00 X+10 Y+0 *
N99999999 %ISO1 G71 *
"""


def test_iso_program_motion_d_functions_labels_cycles_and_transformations():
    res = run(ISO, dialect="heidenhain_iso", name="ISO1.I")
    assert res.alarms[0] is None
    moves = path(res)
    assert moves[:8] == [("rapid", 0.0, 0.0, 105.0), ("linear", 0.0, 0.0, 99.0), ("linear", 10.0, 0.0, 99.0),
                         ("arc", 20.0, 10.0, 99.0), ("arc", 30.0, 0.0, 99.0), ("linear", 31.0, 0.0, 99.0),
                         ("linear", 32.0, 0.0, 99.0), ("linear", 33.0, 0.0, 99.0)]
    assert moves[8:12] == [("rapid", 5.0, 5.0, 150.0), ("rapid", 5.0, 5.0, 102.0), ("linear", 5.0, 5.0, 95.0),
                           ("rapid", 5.0, 5.0, 110.0)]
    assert moves[12] == ("rapid", -5.0, 5.0, 110.0)
    assert moves[13] == ("rapid", 5.0, 15.0, 110.0)


def test_iso_errors():
    assert run("%A G71 *\nN10 G01 X+1 F100 *\nN20 G77 *\nN30 %A G71 *\n", dialect="heidenhain_iso",
               name="A.I").alarms[0]["code"] == "160-0003"
    assert run("%A G71 *\nN10 G26 R5 *\nN30 %A G71 *\n", dialect="heidenhain_iso", name="A.I").alarms[0]["code"] == \
        "TW0012"
    assert run("%A G71 *\nN10 G53 *\nN30 %A G71 *\n", dialect="heidenhain_iso", name="A.I").alarms[0]["code"] == "TW0012"
    assert run(pgm(TOOL + "\nCYCL DEF 19.0 WORKING PLANE")).alarms[0]["code"] == "TW0012"     # exists, not modelled
    assert run(pgm(TOOL + "\nCYCL DEF 999 FOO")).alarms[0]["code"] == "160-000B"


def test_iso_g05_and_g15_reuse_the_last_programmed_direction():
    res = run("%A G71 *\nN10 T1 G17 S1000 *\nN20 G00 X+10 Y+0 Z+5 M03 *\nN30 I+0 J+0 *\n"
              "N40 G13 G91 H+90 F100 *\nN50 G15 G91 H+90 *\nN60 G02 X+0 Y+10 R+10 *\nN70 G05 X+10 Y+0 R+10 *\n"
              "N80 %A G71 *\n", dialect="heidenhain_iso", name="A.I")
    assert res.alarms[0] is None
    sweeps = [round(math.degrees(c.arc.sweep_rad), 6) for c in res.commands[0] if c.kind == Kind.ARC]
    assert sweeps == [90.0, 90.0, -60.0, -60.0]          # G15 after G13 turns CCW, G05 after G02 turns CW
    first = run("%A G71 *\nN10 T1 G17 S1000 *\nN20 G00 X+10 Y+0 Z+5 M03 *\nN30 I+0 J+0 *\n"
                "N40 G15 G91 H+90 F100 *\nN80 %A G71 *\n", dialect="heidenhain_iso", name="A.I")
    assert first.alarms[0]["code"] == "1A0-000C"


def test_tool_call_feed_is_modal_and_m_only_iso_blocks_do_not_move():
    res = run(pgm("TOOL CALL 1 Z S1000 F300\nL Z+0 R0 FMAX M3\nL X+10\nL X+20 F150\nL X+30 FAUTO"))
    assert res.alarms[0] is None
    feeds = [round(c.feed.value * 60000.0, 6) for c in res.commands[0] if c.kind == Kind.LINEAR]
    assert feeds == [300.0, 150.0, 300.0]
    iso = run("%A G71 *\nN10 T1 G17 S1000 *\nN20 G00 X+0 Y+0 Z+5 M03 *\nN30 G01 Z-1 F100 *\nN40 G01 X+10 Y+0 *\n"
              "N50 G06 X+20 Y+10 *\nN60 F250 *\nN70 M09 *\nN80 G01 X+30 *\nN90 M30 *\nN99999999 %A G71 *\n",
              dialect="heidenhain_iso", name="A.I")
    assert iso.alarms[0] is None
    moves = [c for c in iso.commands[0] if c.kind in MOTION]
    assert [c.kind for c in moves[-2:]] == [Kind.ARC, Kind.LINEAR]
    assert round(moves[-1].feed.value * 60000.0, 6) == 250.0


def test_real_error_database_gives_the_texts_when_present():
    db = load_controller_db("heidenhain-tnc640")
    if db is None:
        pytest.skip("db/controllers/heidenhain-tnc640.yaml not present")
    res = run(pgm(TOOL + "\nCALL LBL 9"), db=db)
    assert res.alarms[0]["code"] == "160-001B" and res.alarms[0]["text"] == "Label number not found"
    assert res.alarms[0]["source"].startswith("db:")
