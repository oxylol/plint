"""One part programmed for FANUC 0i-MF, SINUMERIK 840D sl, HEIDENHAIN Klartext and HEIDENHAIN ISO gives the
same toolpath (machine targets, arc centres and sweeps, feeds) within rounding, and the same spindle,
coolant and tool-change states in the same order."""

from __future__ import annotations

import copy

from cnctwin.controller import Controller, ControllerSetup, ProgramSource, ToolOffsetInit, dry_run
from cnctwin.controller.ir import Kind
from harness import FANUC_DB, MILL

FANUC = """%
O1000
G90 G17 G21 G54 G94 G40 G49 G80
T1 M06
G43 H1
S3000 M03
M08
G00 X0 Y0 Z5.
G01 Z-2. F200.
X40.
G02 X50. Y10. I0 J10.
G01 Y30.
G03 X40. Y40. R10.
G01 X0
Y0
G00 Z5.
G52 X60. Y0
G68 X0 Y0 R30.
G00 X0 Y0
G01 Z-1. F200.
M98 P2000 L2
G69
G52 X0 Y0
G00 Z50.
M09
M05
M30
O2000
G91 G01 X10.
Y10.
X-10.
Y-10.
G90
M99
%
"""

SIEMENS = """G90 G17 G71 G54 G94 G40
T1 M6
S3000 M3
M8
G0 X0 Y0 Z5
G1 Z-2 F200
X40
G2 X50 Y10 I0 J10
G1 Y30
G3 X40 Y40 CR=10
G1 X0
Y0
G0 Z5
TRANS X60 Y0
AROT RPL=30
G0 X0 Y0
G1 Z-1 F200
SQUARE P2
TRANS
G0 Z50
M9
M5
M30
"""
SQUARE = "G91 G1 X10\nY10\nX-10\nY-10\nG90\nM17\n"

KLARTEXT = """0 BEGIN PGM PART MM
1 TOOL CALL 1 Z S3000
2 L X+0 Y+0 Z+5 R0 FMAX M3 M8
3 L Z-2 F200
4 L X+40
5 CC X+40 Y+10
6 C X+50 Y+10 DR-
7 L Y+30
8 CR X+40 Y+40 R+10 DR+
9 L X+0
10 L Y+0
11 L Z+5 FMAX
12 CYCL DEF 7.0 DATUM SHIFT
13 CYCL DEF 7.1 X+60
14 CYCL DEF 7.2 Y+0
15 CYCL DEF 10.0 ROTATION
16 CYCL DEF 10.1 ROT+30
17 L X+0 Y+0 FMAX
18 L Z-1 F200
19 CALL LBL 1 REP 1
20 CYCL DEF 10.0 ROTATION
21 CYCL DEF 10.1 ROT+0
22 CYCL DEF 7.0 DATUM SHIFT
23 CYCL DEF 7.1 X+0
24 CYCL DEF 7.2 Y+0
25 L Z+50 FMAX M9
26 M5
27 M30
28 LBL 1
29 L IX+10
30 L IY+10
31 L IX-10
32 L IY-10
33 LBL 0
34 END PGM PART MM
"""

ISO = """%PART G71 *
N10 T1 G17 S3000 *
N20 G00 G40 G90 X+0 Y+0 Z+5 M03 M08 *
N30 G01 Z-2 F200 *
N40 X+40 *
N50 I+40 J+10 *
N60 G02 X+50 Y+10 *
N70 G01 Y+30 *
N80 G03 X+40 Y+40 R+10 *
N90 G01 X+0 *
N100 Y+0 *
N110 G00 Z+5 *
N120 G54 X+60 Y+0 *
N130 G73 G90 H+30 *
N140 G00 X+0 Y+0 *
N150 G01 Z-1 F200 *
N160 L1,1 *
N170 G73 G90 H+0 *
N180 G54 X+0 Y+0 *
N190 G00 Z+50 M09 *
N200 M05 *
N210 M30 *
N220 G98 L1 *
N230 G91 G01 X+10 *
N240 Y+10 *
N250 X-10 *
N260 Y-10 *
N270 G90 *
N280 G98 L0 *
N99999999 %PART G71 *
"""

OFFSET = {"G54": [-0.4, -0.3, -0.2]}


def run(dialect: str, text: str, name: str, extra=()):
    machine = copy.deepcopy(MILL)
    machine["controller"]["dialect"] = dialect
    sources = [ProgramSource(name, text)] + [ProgramSource(n, t, path=-1, main=False) for n, t in extra]
    setup = ControllerSetup(machine, sources, controller_db=FANUC_DB if dialect == "fanuc" else None,
                            db_id="fixture" if dialect == "fanuc" else None, load_db=False, work_offsets_m=[OFFSET],
                            tools=[ToolOffsetInit(1, length_m=0.1, radius_m=0.005)])
    result = dry_run(Controller(setup))
    assert result.alarms[0] is None, (dialect, result.alarms[0])
    return result


def toolpath(result) -> list[tuple]:
    out = []
    for c in result.commands[0]:
        if c.kind in (Kind.RAPID, Kind.LINEAR, Kind.ARC):
            entry = (c.kind.value, tuple(round(c.target[a] * 1e6) for a in ("X", "Y", "Z")))
            if c.kind != Kind.RAPID:
                entry += (round(c.feed.value * 60000.0, 6),)
            if c.arc is not None:
                entry += (tuple(round(v * 1e6) for v in c.arc.centre), round(c.arc.sweep_rad, 9))
            out.append(entry)
    return out


def actions(result) -> list[tuple]:
    out = []
    for c in result.commands[0]:
        if c.kind == Kind.SPINDLE:
            out.append(("spindle", round(c.spindle["omega_rad_s"], 6)))
        elif c.kind == Kind.COOLANT:
            out.append(("coolant", c.coolant["pumps"]["flood"]))
        elif c.kind == Kind.TOOL_CHANGE:
            out.append(("tool", c.tool["number"]))
    return out


def test_one_part_in_four_dialects_gives_one_toolpath():
    fanuc = run("fanuc", FANUC, "O1000.NC")
    siemens = run("siemens840d", SIEMENS, "PART.MPF", extra=[("SQUARE.SPF", SQUARE)])
    klartext = run("heidenhain_klartext", KLARTEXT, "PART.H")
    iso = run("heidenhain_iso", ISO, "PART.I")
    reference = toolpath(fanuc)
    assert len(reference) == 20
    for other in (siemens, klartext, iso):
        assert toolpath(other) == reference
    assert actions(siemens) == actions(fanuc) == actions(klartext) == actions(iso)
    corners = {c[1][:2] for c in reference[-8:]}
    assert (-340000 + 8660, -300000 + 5000) in corners      # rotated square corner (10, 0) -> (8.660, 5.000)
