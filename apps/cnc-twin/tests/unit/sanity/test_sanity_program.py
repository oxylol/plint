"""Lexer and tool-comment parser of the program view."""

import math

from cnctwin.sanity.program import lex, lex_line, parse_tool_comment


def test_words_comments_and_markers():
    line = lex_line("/N120 G01 X-12.5 Y.5 F500. (FINISH) ; trailing", 7)
    assert line.block_delete and line.n_number == 120
    assert [w.letter for w in line.words] == ["G", "X", "Y", "F"]
    assert line.first("X").value == -12.5 and line.first("Y").value == 0.5
    assert line.first("F").has_point and not line.first("G").has_point
    assert line.codes("G") == ["G1"]
    assert line.comments == ("FINISH", "trailing")
    assert lex_line("%", 1).percent
    assert lex_line("O1000 (PART)", 2).program_number == "O1000"
    assert lex_line(":2000", 3).program_number == "O2000"
    assert lex_line("G54.1 P3", 4).codes("G") == ["G54.1"]


def test_macro_lines_are_marked():
    assert lex_line("#100 = 2.5", 1).macro
    assert lex_line("IF [#1 GT 5] GOTO 10", 2).macro
    expression = lex_line("G00 X#101 Z[#1 - 2.]", 3)
    assert expression.macro
    assert expression.first("G").value == 0.0
    assert expression.first("X").value is None and expression.first("X").text == "#101"
    assert expression.first("Z").value is None and expression.first("Z").text.startswith("[")
    assert not lex_line("G00 X1.", 4).macro


def test_lex_keeps_every_physical_line():
    lines = lex("%\nO1\n\nG00 X0\n%\n")
    assert [l.number for l in lines] == [1, 2, 3, 4, 5]
    assert lines[2].is_empty


def test_tool_comments():
    endmill = parse_tool_comment("T1 D10 ENDMILL", 3, 1e-3)
    assert endmill.number == 1 and math.isclose(endmill.diameter_m, 0.010) and endmill.kind == "endmill"
    drill = parse_tool_comment("T3 6.8 DRILL", 4, 1e-3)
    assert drill.kind == "drill" and math.isclose(drill.diameter_m, 0.0068)
    tap = parse_tool_comment("T4 M8X1.25 TAP", 5, 1e-3)
    assert tap.kind == "tap" and math.isclose(tap.thread_major_m, 0.008) and math.isclose(tap.thread_pitch_m, 0.00125)
    assert tap.diameter_m is None
    spot = parse_tool_comment("TOOL 2: SPOT DRILL DIA 10", 6, 1e-3)
    assert spot.number == 2 and spot.kind == "spotdrill" and math.isclose(spot.diameter_m, 0.010)
    fraction = parse_tool_comment("T5 1/2 EM", 7, 0.0254)
    assert math.isclose(fraction.diameter_m, 0.0127) and fraction.kind == "endmill"
    face = parse_tool_comment("T6 FACE MILL D50 R0.8", 8, 1e-3)
    assert face.kind == "facemill" and math.isclose(face.diameter_m, 0.050)
    tap_drill = parse_tool_comment("T7 6.8 TAP DRILL", 9, 1e-3)
    assert tap_drill.kind == "drill"
    operation = parse_tool_comment("ROUGH POCKET", 10, 1e-3)
    assert not operation.describes_tool
