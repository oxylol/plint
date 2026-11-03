from cnctwin.controller.dialects.fanuc_tables import LETTERS, MAX_DIGITS
from cnctwin.controller.expr import ParseError, parse_expression, parse_statement
from cnctwin.controller.lexer import LexerRules, parse_block, parse_file
from cnctwin.controller.program import ProgramSource
from cnctwin.controller.ir import Kind
from harness import alarm, moves, run, wrap

RULES = LexerRules(letters=LETTERS, max_digits=dict(MAX_DIGITS))


def test_comments_semicolon_and_spaces():
    b = parse_block("N10 G01 (MOVE) X 10. Y-5.5 ; trailing note", 3, RULES)
    assert b.n == 10 and b.line == 3 and b.comment == "MOVE trailing note"
    assert [(w.letter, w.value, w.point) for w in b.words] == [("G", 1.0, False), ("X", 10.0, True), ("Y", -5.5, True)]
    assert parse_block("(ONLY A COMMENT)", 1, RULES).words == ()


def test_block_delete_leading_and_mid_block():
    b = parse_block("/2 N20 G00 X1.", 1, RULES)
    assert b.delete == 2 and b.n == 20
    b = parse_block("G00 X1. /3 Y2.", 1, RULES)
    assert b.mid_delete_at == 2 and b.mid_delete == 3


def test_programs_numbers_and_tape_marks():
    text = "leader text\n%\nO1000 (MAIN)\nG00 X1.\nM30\nO2000\nM99\n%\ntrailer"
    programs = parse_file("f.nc", text, RULES)
    assert [p.number for p in programs] == [1000, 2000]
    assert programs[0].name == "O1000" and len(programs[0].blocks) == 2 and programs[0].blocks[0].line == 4
    unnumbered = parse_file("g.nc", "G00 X1.\nM30\n", RULES)
    assert unnumbered[0].number is None and unnumbered[0].name == "g.nc"


def test_lexical_errors_are_deferred_to_the_block():
    assert parse_block("G01 X", 1, RULES).error[0] == "no_data_after_address"
    assert parse_block("G01 X1.2.3", 1, RULES).error[0] == "illegal_decimal_point"
    assert parse_block("G01 X--1", 1, RULES).error[0] == "illegal_minus"
    assert parse_block("G01 X1234567890", 1, RULES).error[0] == "too_many_digits"
    assert parse_block("N123456 G01", 1, RULES).error[0] == "too_many_digits"
    assert parse_block("G01 @1", 1, RULES).error[0] == "improper_address"


def test_lexical_alarms_raised_when_reached():
    r = run(wrap("G00 X1.\nG01 X"))
    assert alarm(r)["code"] == "PS0005" and alarm(r)["text"] == "NO DATA AFTER ADDRESS"
    assert len(moves(r)) == 1
    assert alarm(run(wrap("G01 X1.2.3 F1.")))["code"] == "PS0007"
    assert alarm(run(wrap("G01 X--1 F1.")))["code"] == "PS0006"
    r = run(wrap("G00 X1.\nM30\nG01 X"))
    assert alarm(r) is None


def test_macro_statement_detection():
    assert parse_block("#1=#2+3", 1, RULES).stmt[0] == "assign"
    assert parse_block("IF [#1 GT 2] GOTO 10", 1, RULES).stmt[0] == "if_goto"
    assert parse_block("WHILE [#1 LT 5] DO 1", 1, RULES).stmt == ("while", ("brk", ("bin", "LT", ("var", 1), ("num", 5.0))), 1)
    assert parse_block("END 1", 1, RULES).stmt == ("end", 1)
    assert parse_block("GOTO #10", 1, RULES).stmt == ("goto", ("var", 10))
    assert parse_block("D01", 1, RULES).words[0].letter == "D"
    assert parse_block("#1=5 X10.", 1, RULES).error[0] == "nc_macro_same_block"


def test_word_values_from_expressions():
    b = parse_block("G01 X#1 Y-#2 Z[#3+1] F#[#4]", 1, RULES)
    letters = [w.letter for w in b.words]
    assert letters == ["G", "X", "Y", "Z", "F"]
    assert b.words[1].expr == ("var", 1) and b.words[2].expr == ("neg", ("var", 2))
    assert b.terms > 0


def test_expression_precedence_and_brackets():
    node, _ = parse_expression("1+2*3")
    assert node == ("bin", "+", ("num", 1.0), ("bin", "*", ("num", 2.0), ("num", 3.0)))
    node, _ = parse_expression("#1GT0AND#2")
    assert node[0] == "bin" and node[1] == "GT"
    try:
        parse_expression("[[[[[[1]]]]]]")
    except ParseError as error:
        assert error.key == "bracket_nesting"
    else:
        raise AssertionError("six bracket levels accepted")
    node, _ = parse_expression("ATAN[1]/[2]")
    assert node == ("fn", "ATAN", (("num", 1.0), ("brk", ("num", 2.0))))


def test_statement_errors():
    for text, key in (("#1=", "expression_format"), ("IF[#1]X1", "macro_format"), ("DO4", "illegal_loop_number")):
        try:
            stmt, _ = parse_statement(text)
        except ParseError as error:
            assert error.key == key
        else:
            assert key == "illegal_loop_number" and stmt == ("do", 4)


def test_several_programs_and_library_calls():
    prog = "%\nO0001\nM98 P2000\nM30\nO2000\nG00 X5.\nM99\n%\n"
    m = moves(run(prog))
    assert len(m) == 1 and m[0].ref.call_stack == ("O0001:3",)


def test_lowercase_is_read_and_reported():
    r = run(wrap("g00 x1."))
    assert len(moves(r)) == 1
    assert any("lower-case" in e.get("message", "") for e in r.events[0])


def test_program_digest_detects_changed_text():
    from cnctwin.controller.program import source_digest
    a = source_digest([ProgramSource("a.nc", "G00 X1.")])
    b = source_digest([ProgramSource("a.nc", "G00 X2.")])
    assert a != b
    assert Kind.RAPID.value == "rapid"
