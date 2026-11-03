"""SINUMERIK 840D sl dialect: motion, arcs, frames, programming language, tools, alarms (docs/dialects.md)."""

from __future__ import annotations

import copy
import math

import pytest

from cnctwin.controller import Controller, ControllerSetup, ProgramSource, ToolOffsetInit, dry_run
from cnctwin.controller.alarms import load_controller_db
from cnctwin.controller.ir import Kind
from harness import LATHE, MILL, SWISS, UMC

MOTION = (Kind.RAPID, Kind.LINEAR, Kind.ARC)
SIEMENS_DB = {
    "format": "cnctwin-controller", "version": 1, "id": "fixture", "dialect": "siemens840d", "name": "fixture",
    "alarms": [{"code": "12080", "text": "[Channel %1: ] Block %2 syntax error in text %3"},
               {"code": "14080", "text": "[Channel %1: ] Block %2 jump destination %3 not found"},
               {"code": "10720", "text": "[Channel %1: ] Block %3 axis %2 software limit switch %4"},
               {"code": "61101", "text": "[Channel %1: ] Block %2: Reference plane incorrectly defined"}],
}


def machine(base=MILL, travel: bool = True) -> dict:
    m = copy.deepcopy(base)
    m["controller"]["dialect"] = "siemens840d"
    m["controller"]["alarms"] = "fixture"
    if travel:
        for axis in m["axes"].values():
            if axis.get("type") == "linear":
                axis["travel_mm"] = [-2000, 2000]
    return m


def controller(program: str, base=MILL, *, extra=(), tools=None, offsets=None, db=SIEMENS_DB, travel=True,
               programs=None) -> Controller:
    if programs is not None:
        sources = [ProgramSource(f"P{k}.MPF", text, path=k) for k, text in enumerate(programs)]
    else:
        sources = [ProgramSource("TEST.MPF", program)]
    sources += [ProgramSource(name, text, path=-1, main=False) for name, text in extra]
    setup = ControllerSetup(machine(base, travel), sources, controller_db=db, db_id="fixture",
                            work_offsets_m=[offsets or {}] + [{} for _ in range(3)],
                            tools=tools if tools is not None else [ToolOffsetInit(1, length_m=0.1, radius_m=0.005),
                                                                   ToolOffsetInit(2, length_m=0.05, radius_m=0.003)])
    return Controller(setup)


def run(program: str, base=MILL, **kwargs):
    return dry_run(controller(program, base, **kwargs))


def path(res, p: int = 0) -> list[tuple]:
    out = []
    for c in res.commands[p]:
        if c.kind in MOTION:
            out.append((c.kind.value,) + tuple(round(c.target[a] * 1000.0, 6) for a in ("X", "Y", "Z")
                                               if a in c.target))
    return out


def arcs(res) -> list:
    return [c for c in res.commands[0] if c.kind == Kind.ARC]


def head(body: str, defs: str = "") -> str:
    """DEF statements first (the definition part), then the set-up blocks and the body."""
    return (defs.strip() + "\n" if defs else "") + "G90 G17 G71 G94\nT1 M6\n" + body.strip() + "\nM30\n"


# ====================================================================== motion and arcs

def test_linear_absolute_incremental_and_per_word_modifiers():
    res = run(head("""G0 X10 Y20 Z5
G1 X=IC(5) F200
G91 Y10 Z=AC(-1)
X=AC(0)
G90 X5 Y=IC(-5)"""))
    assert res.alarms[0] is None
    assert path(res) == [("rapid", 10.0, 20.0, 105.0), ("linear", 15.0, 20.0, 105.0), ("linear", 15.0, 30.0, 99.0),
                         ("linear", 0.0, 30.0, 99.0), ("linear", 5.0, 25.0, 99.0)]


def test_arcs_by_centre_radius_aperture_intermediate_point_and_tangent():
    res = run(head("""G0 X0 Y0 Z0 F300
G1 X10
G2 X20 Y10 I0 J10
G3 X30 Y0 CR=10
G2 X40 Y0 AR=180
G3 I=AC(45) J0 AR=90
G1 X60 Y0
CIP X80 Y0 I1=70 J1=10
CT X100 Y20"""))
    assert res.alarms[0] is None
    a = arcs(res)
    assert [round(c.arc.sweep_rad, 9) for c in a[:4]] == [round(-1.5 * math.pi, 9), round(0.5 * math.pi, 9),
                                                          round(-math.pi, 9), round(0.5 * math.pi, 9)]
    assert [round(v * 1000, 6) for v in a[0].arc.centre[:2]] == [10.0, 10.0]
    assert [round(v * 1000, 6) for v in a[1].arc.centre[:2]] == [30.0, 10.0]
    assert [round(v * 1000, 6) for v in a[2].arc.centre[:2]] == [35.0, 0.0]
    assert path(res)[5][1:3] == (45.0, -5.0)
    cip = a[4]
    assert [round(v * 1000, 6) for v in cip.arc.centre[:2]] == [70.0, 0.0] and cip.arc.sweep_rad < 0
    ct = a[5]                                   # the CIP arc ends heading -Y: tangential 270 deg turn
    assert round(ct.arc.sweep_rad, 9) == round(1.5 * math.pi, 9)
    assert [round(v * 1000, 6) for v in ct.arc.centre[:2]] == [100.0, 0.0]


def test_aperture_angle_outside_0_to_360_is_14910():
    for ar in ("360", "-10", "0"):
        alarm = run(head(f"G0 X0 Y0 Z0 F100\nG2 X10 Y0 AR={ar}")).alarms[0]
        assert alarm["code"] == "14910" and alarm["detail"] == f"AR={ar}"
    assert run(head("G0 X0 Y0 Z0 F100\nG2 X10 Y0 I5 AR=90")).alarms[0]["code"] == "14900"


def test_helix_with_turn_and_cr_larger_than_half_circle():
    res = run(head("""G0 X10 Y0 Z0 F300
G3 X10 Y0 Z-6 I-10 J0 TURN=2
G2 X30 Y0 CR=-10"""))
    assert res.alarms[0] is None
    helix, big = arcs(res)
    assert round(helix.arc.sweep_rad, 9) == round(6 * math.pi, 9)
    assert round(helix.target["Z"] * 1000, 6) == 94.0
    assert round(big.arc.sweep_rad, 9) == round(-2 * math.pi, 9) or abs(big.arc.sweep_rad) > math.pi


def test_arc_end_point_outside_tolerance_is_14040_and_within_is_corrected():
    bad = run(head("G0 X0 Y0 Z0 F100\nG2 X20.5 Y0 I10 J0"))
    assert bad.alarms[0]["code"] == "14040"
    good = run(head("G0 X0 Y0 Z0 F100\nG2 X20.005 Y0 I10 J0"))
    assert good.alarms[0] is None
    arc = arcs(good)[0].arc
    assert abs(arc.radius_start_m - arc.radius_end_m) < 1e-12


def test_polar_coordinates_with_pole_g111_g110_and_ap_rp():
    res = run(head("""G0 X0 Y0 Z0 F100
G111 X10 Y10
G1 AP=0 RP=10
AP=90
G110 X0 Y0
G0 AP=IC(90) RP=5"""))
    assert res.alarms[0] is None
    assert path(res)[1:] == [("linear", 20.0, 10.0, 100.0), ("linear", 10.0, 20.0, 100.0), ("rapid", 10.0, 25.0, 100.0)]


def test_modulo_rotary_axis_dc_acp_acn_and_plain_absolute():
    res = run("""G90 G0 C350
C=DC(10)
C=ACP(5)
C=ACN(10)
C20
M30
""", UMC)
    assert res.alarms[0] is None
    c = [round(math.degrees(cmd.target["C"]), 6) for cmd in res.commands[0] if cmd.kind == Kind.RAPID]
    assert c == [350.0, 370.0, 725.0, 370.0, 380.0]


# ====================================================================== frames

def test_programmable_frames_trans_rot_scale_mirror_and_suppression():
    res = run(head("""G54 G0 X0 Y0 Z0
TRANS X10 Y10
AROT RPL=90
G1 X10 Y0 F200
ATRANS X5
G1 X0 Y0
TRANS
SCALE X2 Y2
G1 X5 Y5
ASCALE X0.5
G1 X10 Y0
MIRROR X0
G1 X10 Y5
MIRROR
G53 G0 Z0
SUPA X0
G0 X1"""), offsets={"G54": [0.1, 0.2, 0.0]})
    assert res.alarms[0] is None
    assert path(res) == [("rapid", 100.0, 200.0, 100.0), ("linear", 110.0, 220.0, 100.0),
                         ("linear", 110.0, 215.0, 100.0), ("linear", 110.0, 210.0, 100.0),
                         ("linear", 110.0, 200.0, 100.0), ("linear", 90.0, 205.0, 100.0),
                         ("rapid", 90.0, 205.0, 100.0), ("rapid", 0.0, 205.0, 100.0), ("rapid", 101.0, 205.0, 100.0)]


def test_circles_under_unequal_scale_factors_are_refused():
    assert run(head("SCALE X2 Y1\nG0 X0 Y0 Z0 F100\nG2 X10 Y0 CR=5")).alarms[0]["code"] == "TW0018"
    assert run(head("SCALE X2 Y2\nG0 X0 Y0 Z0 F100\nG2 X10 Y0 CR=5")).alarms[0] is None
    assert run(head("SCALE X2 Y1\nG0 X0 Y0 Z0 F100\nG19 G2 Y10 Z0 CR=5")).alarms[0] is None    # Y and Z equal
    assert run(head("SCALE X2 Y1\nS1000 M3\nG0 X0 Y0 Z20\n"
                    "POCKET4(10, 0, 2, -4, 20, 50, 50, 4, 0, 0, 400, 100, 0, 11, 4)")).alarms[0]["code"] == "61011"


def test_frame_variables_ctrans_crot_and_p_pframe():
    res = run(head("""F1 = CTRANS(X, 10, Y, 0) : CROT(Z, 90)
$P_PFRAME = F1
G0 X5 Y0 Z0
R1 = F1[X, TR]
R2 = F1[Z, RT]
$P_PFRAME = CTRANS()
G0 X=R1 Y=R2""", defs="DEF FRAME F1"))
    assert res.alarms[0] is None
    assert path(res) == [("rapid", 10.0, 5.0, 100.0), ("rapid", 10.0, 90.0, 100.0)]


def test_settable_frames_g500_g505_and_p_uifr():
    res = run(head("""G505 G0 X0 Y0 Z0
G500 X0
$P_UIFR[1] = CTRANS(X, 7, Y, 8)
G54 X0 Y0"""), offsets={"G505": [0.05, 0.0, 0.0]})
    assert res.alarms[0] is None
    assert path(res) == [("rapid", 50.0, 0.0, 100.0), ("rapid", 0.0, 0.0, 100.0), ("rapid", 7.0, 8.0, 100.0)]


# ====================================================================== programming language

def test_control_structures_if_while_for_loop_repeat_and_jumps():
    res = run(head("""R1=0
WHILE R1<2
R1=R1+1
ENDWHILE
FOR N=1 TO 3
R2=R2+N
ENDFOR
LOOP
R3=R3+1
IF R3==2 GOTOF OUT
ENDLOOP
OUT: REPEAT
R4=R4+1
UNTIL R4>=3
IF R1==2
G0 X=R1
ELSE
G0 X99
ENDIF
R5=0
BACK: R5=R5+1
IF R5<2 GOTOB BACK
GOTOF FWD
G0 Y99
FWD: G0 Y=R2 Z=R3*10+R4+R5
GOTOC NOWHERE
G0 X1""", defs="DEF INT N=0"))
    assert res.alarms[0] is None
    assert path(res) == [("rapid", 2.0, 0.0, 0.0), ("rapid", 2.0, 6.0, 125.0), ("rapid", 1.0, 6.0, 125.0)]
    # D1 (tool length 100 mm) became active with M6 while Z stood at 0: Z keeps its machine position until
    # it is programmed (the length is applied with the next motion of the tool axis)


def test_program_section_repetition_repeat_repeatb_and_endlabel():
    def xs(body: str):
        res = run(head(body), extra=[("SUB9.SPF", "G91 G0 Z1\nM17\n")])
        assert res.alarms[0] is None, res.alarms[0]
        return [m[1:] for m in path(res)][1:]
    assert xs("G0 X0 Y0 Z0\nLAB1: G91 G0 X1\nREPEATB LAB1 P=2\nG90 G0 Y5") == [
        (1.0, 0.0, 100.0), (2.0, 0.0, 100.0), (3.0, 0.0, 100.0), (3.0, 5.0, 100.0)]
    # label up to the REPEAT line, two more passes
    assert len(xs("G0 X0 Y0 Z0\nG91\nMA1: G0 X1\nG0 Y1\nREPEAT MA1 P=2\nG90")) == 6
    # label to ENDLABEL; two labels; a section after the REPEAT jumped over afterwards; a subprogram ends one
    assert xs("G0 X0 Y0 Z0\nG91\nMA1: G0 X1\nENDLABEL: G0 Y1\nG0 Z1\nREPEAT MA1 P=1\nG90")[-3:] == [
        (1.0, 1.0, 101.0), (2.0, 1.0, 101.0), (2.0, 2.0, 101.0)]
    assert xs("G0 X0 Y0 Z0\nREPEAT SEC1 SEC2 P=2\nGOTOF ENDE\nSEC1: G91 G0 X1\nSEC2: G90\nENDE: G0 Y7") == [
        (1.0, 0.0, 100.0), (2.0, 0.0, 100.0), (2.0, 7.0, 100.0)]
    assert xs("G0 X0 Y0 Z0\nG91\nAA1: G0 X1\nBB1: SUB9\nREPEAT AA1 BB1 P=2\nG90")[-1] == (3.0, 0.0, 103.0)
    assert run(head("G0 X0\nREPEAT NOPE P=2")).alarms[0]["code"] == "14080"


def test_jump_destination_not_found_is_14080_with_its_text():
    res = run(head("GOTOF MISSING"))
    assert res.alarms[0]["code"] == "14080"
    assert res.alarms[0]["text"] == "Block line 3 jump destination MISSING not found"
    assert res.alarms[0]["source"] == "db:fixture"


def test_expressions_functions_and_types():
    res = run(head("""A = SQRT(POT(3)+POT(4)) + ABS(-1) + TRUNC(2.7) + ROUND(2.5) + 10 MOD 4 + 9 DIV 2
B = SIN(30) + COS(60) + TAN(45) + ASIN(1) + ACOS(0) + ATAN2(1, 1)
CNT = 2.5
TXT = TXT << "C" << CNT
IF (OK AND (CNT==3)) AND (TXT=="ABC3")
G0 X=A Y=B Z=CNT
ENDIF""", defs='DEF REAL A, B\nDEF INT CNT=7\nDEF BOOL OK=TRUE\nDEF STRING[10] TXT="AB"'))
    assert res.alarms[0] is None
    assert path(res) == [("rapid", 17.0, 227.0, 103.0)]       # 5+1+2+3+2+4, 0.5+0.5+1+90+90+45


def test_subprograms_with_repeat_parameters_var_save_and_ret():
    res = run(head("""G0 X0 Y0 Z0
R5 = 0
SUB1 P2
SUB2(3, R5)
G0 Z=R5"""), extra=[("SUB1.SPF", "G91 G1 X1 F100\nM17\n"),
                    ("SUB2.SPF", "PROC SUB2(REAL A, VAR REAL B) SAVE\nDEF REAL C=2\nG91 G1 Y=A*C F100\nB=A*2\nRET\n")])
    assert res.alarms[0] is None
    assert path(res) == [("rapid", 0.0, 0.0, 100.0), ("linear", 1.0, 0.0, 100.0), ("linear", 2.0, 0.0, 100.0),
                         ("linear", 2.0, 6.0, 100.0), ("rapid", 2.0, 6.0, 106.0)]


def test_mcall_cycle_after_every_positioning_and_cancel():
    res = run(head("""S1000 M3 F100
G0 X0 Y0 Z20
MCALL CYCLE81(10, 0, 2, -5)
X10
X20
MCALL
X30"""))
    assert res.alarms[0] is None
    assert path(res) == [("rapid", 0.0, 0.0, 120.0), ("rapid", 10.0, 0.0, 120.0), ("rapid", 10.0, 0.0, 102.0),
                         ("linear", 10.0, 0.0, 95.0), ("rapid", 10.0, 0.0, 110.0), ("rapid", 20.0, 0.0, 110.0),
                         ("rapid", 20.0, 0.0, 102.0), ("linear", 20.0, 0.0, 95.0), ("rapid", 20.0, 0.0, 110.0),
                         ("rapid", 30.0, 0.0, 110.0)]


def test_undefined_names_missing_programs_and_unsupported_words_alarm():
    assert run(head("FOO X1")).alarms[0]["code"] == "12550"
    assert run(head("A=B", defs="DEF REAL A")).alarms[0]["code"] == "12550"
    assert run(head("CALL \"NOPE\"")).alarms[0]["code"] == "14011"
    assert run(head("G0 X1\nDEF REAL LATE")).alarms[0]["code"] == "14500"
    assert run(head("ASPLINE X1 Y1")).alarms[0]["code"] == "TW0012"
    assert run(head("G1 X10 X20 F100")).alarms[0]["code"] == "16420"
    assert run(head("G0 G1 X1")).alarms[0]["code"] == "12060"
    assert run(head("IF R1==1\nG0 X1")).alarms[0]["code"] == "12640"
    assert run(head("G0 X1 Y")).alarms[0]["code"] == "12080"
    assert run(head("CYCLE71(50, 2, 2, 0, 0, 0, 80, 60)")).alarms[0]["code"] == "TW0012"     # standard, not modelled
    assert run(head("CYCLE71(1)"), extra=[("CYCLE71.SPF", "PROC CYCLE71(REAL A)\nG0 X=A\nM17\n")]).alarms[0] is None


def test_syntax_error_text_is_formatted_with_the_block_number():
    res = run("G90 G17\nN10 G0 X=AC(\nM30\n")
    alarm = res.alarms[0]
    assert alarm["code"] == "12080" and alarm["text"].startswith("Block N10 syntax error in text")


def test_software_limit_switch_10720_before_the_motion_is_emitted():
    res = run(head("G0 X-850 Y-10 Z-10"), travel=False)
    assert res.alarms[0]["code"] == "10720"
    assert res.alarms[0]["text"] == "Block line 3 axis X software limit switch -"
    assert not [c for c in res.commands[0] if c.kind == Kind.RAPID]


def test_msg_setal_stopre_and_system_variables():
    res = run(head("""MSG("hello " << 3)
G0 X5 Y6 Z7
R1 = $P_EP[X] + $P_TOOLNO * 100 + $TC_DP3[1,1]
STOPRE
R2 = $AA_IW[Y]
G0 X=R1 Y=R2"""))
    assert res.alarms[0] is None
    syncs = [c for c in res.commands[0] if c.kind == Kind.SYNC]
    assert syncs[0].sync["code"] == "STOPRE" and syncs[1].sync["reason"] == "live_read"
    assert path(res)[-1] == ("rapid", 205.0, 6.0, 107.0)
    assert any(e.get("message") == "MSG: hello 3" for e in res.events[0])
    alarm = run(head("SETAL(61101)")).alarms[0]
    assert alarm["code"] == "61101" and alarm["text"] == "Block line 3: Reference plane incorrectly defined"


def test_dates_and_tool_data_variables():
    res = run(head("""R1 = $A_YEAR
R2 = $TC_DP6[1,1]
$TC_DP3[1,1] = 50
D1
G0 X=R1-2000 Y=R2 Z0"""))
    assert res.alarms[0] is None
    assert path(res) == [("rapid", 26.0, 5.0, 50.0)]


# ====================================================================== tools, spindle, coolant

def test_tool_change_d_offsets_spindle_css_spos_and_coolant():
    res = run("""G90 T2
M6
D1
S1000 M3
M8
G0 X0 Y0 Z0
D0
G0 Z0
G96 S200 LIMS=2000
SPOS=90
M9
M30
""")
    assert res.alarms[0] is None
    kinds = [c.kind for c in res.commands[0]]
    assert kinds[:3] == [Kind.MODE, Kind.TOOL_CHANGE, Kind.SPINDLE]
    spindle = [c.spindle for c in res.commands[0] if c.kind == Kind.SPINDLE]
    assert spindle[0]["omega_rad_s"] == pytest.approx(-1000 * 2 * math.pi / 60)
    assert spindle[-1]["orient_rad"] == pytest.approx(math.pi / 2)
    assert path(res) == [("rapid", 0.0, 0.0, 50.0), ("rapid", 0.0, 0.0, 0.0)]
    coolant = [c.coolant["pumps"]["flood"] for c in res.commands[0] if c.kind == Kind.COOLANT]
    assert coolant == [True, False]


def test_illegal_d_number_and_no_tool():
    assert run("G90\nD1\nM30\n").alarms[0]["code"] == "17160"
    assert run("G90 T1 M6\nD3\nM30\n").alarms[0]["code"] == "17181"


def test_cutter_compensation_g41_offsets_a_straight_contour():
    res = run(head("""G0 X-20 Y-20 Z0
G41 G1 X0 Y0 F100
X50
G40 G1 X60 Y-20"""))
    assert res.alarms[0] is None
    moves = path(res)
    assert moves[2][1:3] == (50.0, 5.0)


def test_corner_rounding_rnd_and_chamfer_chr():
    res = run(head("""G0 X0 Y0 Z0
G1 X50 F100 RND=5
Y50 CHR=5
X0"""))
    assert res.alarms[0] is None
    moves = path(res)
    assert moves[1][1:3] == (45.0, 0.0)
    assert [c for c in res.commands[0] if c.kind == Kind.ARC][0].arc.radius_start_m == pytest.approx(0.005)
    assert moves[3][1:3] == (50.0, 45.0) and moves[4][1:3] == (45.0, 50.0)


# ====================================================================== multi-channel

def test_waitm_synchronises_two_channels():
    sw = copy.deepcopy(SWISS)
    ctl = controller("", sw, programs=["G0 X1 Z1\nWAITM(1, 1, 2)\nG0 X2\nM30\n", "G0 X3 Z3\nWAITM(1, 1, 2)\nG0 X4\nM30\n"],
                     tools=[], travel=True)
    res = dry_run(ctl)
    assert res.alarms == [None, None]
    sync = [c for c in res.commands[0] if c.kind == Kind.SYNC][0]
    assert sync.sync["reason"] == "path_wait" and sync.sync["paths"] == [0, 1]
    assert res.end_tick > 0


# ====================================================================== real database

def test_real_alarm_database_gives_the_texts_when_present():
    db = load_controller_db("siemens-840d-sl")
    if db is None:
        pytest.skip("db/controllers/siemens-840d-sl.yaml not present")
    res = dry_run(controller(head("GOTOF NIX"), db=db))
    alarm = res.alarms[0]
    assert alarm["code"] == "14080" and alarm["source"].startswith("db:")
    assert "jump destination NIX not found" in alarm["text"]


def test_lathe_diameter_programming_and_g18():
    res = run("G18 G90 DIAMON T1 D1\nS500 M3\nG0 X40 Z2\nG1 X20 Z0 F0.2\nDIAMOF\nG1 X5\nM30\n", LATHE,
              tools=[ToolOffsetInit(1, x_m=0.0, z_m=0.0)])
    assert res.alarms[0] is None
    assert [(c.kind.value, round(c.target["X"] * 1000, 6), round(c.target["Z"] * 1000, 6))
            for c in res.commands[0] if c.kind in MOTION] == [("rapid", 20.0, 2.0), ("linear", 10.0, 0.0),
                                                              ("linear", 5.0, 0.0)]
