import math

import pytest

from cnctwin.controller.ir import Kind, ProbeResult
from harness import alarm, mm, moves, run, wrap


def values(result, first=100, count=20):
    path = result.controller.state.paths[0]
    return path.variables["c100"][first - 100:first - 100 + count]


def test_arithmetic_precedence_functions_and_assignment():
    body = """#100 = 1 + 2 * 3
#101 = [1 + 2] * 3
#102 = SIN[30] + COS[60]
#103 = ATAN[1]/[-1]
#104 = SQRT[16] + ABS[-2]
#105 = ROUND[2.5] + ROUND[-2.5]
#106 = FIX[-1.7] + FUP[1.2]
#107 = FUP[-1.2]
#108 = 7 MOD 3
#109 = 12 AND 10
#110 = 12 OR 3
#111 = 6 XOR 3
#112 = BIN[37] + BCD[25]
#113 = POW[2, 10]
#114 = LN[EXP[2]]
#115 = ASIN[-0.5]
#116 = ACOS[-1]
#117 = TAN[45]
#118 = [3 GT 2] + [2 EQ 2]
#119 = -#100"""
    r = run(wrap(body))
    v = values(r)
    assert alarm(r) is None
    assert v[0] == 7.0 and v[1] == 9.0
    assert v[2] == pytest.approx(1.0)
    assert v[3] == pytest.approx(135.0)
    assert v[4] == 6.0 and v[5] == 0.0
    assert v[6] == pytest.approx(1.0) and v[7] == -2.0
    assert v[8] == 1.0 and v[9] == 8.0 and v[10] == 15.0 and v[11] == 5.0
    assert v[12] == 25.0 + 37.0 and v[13] == 1024.0
    assert v[14] == pytest.approx(2.0) and v[15] == pytest.approx(330.0) and v[16] == pytest.approx(180.0)
    assert v[17] == pytest.approx(1.0) and v[18] == 2.0 and v[19] == -7.0


def test_vacant_semantics():
    body = """#100 = #1
#101 = #1 * 5
#102 = #1 + #1
#103 = [#1 EQ #0]
#104 = [#1 EQ 0]
#105 = [#1 GE 0]
#106 = [#1 NE 0]
G90 G00 X5. Y#1
#107 = #5002"""
    r = run(wrap(body))
    v = values(r)
    assert v[0] is None and v[1] == 0.0 and v[2] == 0.0
    assert v[3] == 1.0 and v[4] == 0.0 and v[5] == 1.0 and v[6] == 1.0
    assert v[7] == 0.0
    assert mm(moves(r)[0], "X") == 5.0


def test_macro_runtime_alarms():
    assert alarm(run(wrap("#100 = 1 / 0")))["code"] == "PS0112"
    assert alarm(run(wrap("#100 = SQRT[-1]")))["code"] == "PS0119"
    assert alarm(run(wrap("#0 = 1")))["code"] == "PS0116"
    assert alarm(run(wrap("#100 = #40")))["code"] == "PS0115"
    assert alarm(run(wrap("#100 = ASIN[2]")))["code"] == "PS0119"


def test_if_goto_while_nesting_and_goto_backward():
    body = """#100 = 0
#101 = 0
WHILE [#100 LT 3] DO 1
#102 = 0
WHILE [#102 LT 2] DO 2
#101 = #101 + 1
#102 = #102 + 1
END 2
#100 = #100 + 1
END 1
N50 #103 = 0
N60 #103 = #103 + 1
IF [#103 LT 4] GOTO 60
IF [#103 EQ 4] THEN #104 = 99
GOTO 90
#104 = -1
N90 G00 X#101"""
    r = run(wrap(body))
    v = values(r)
    assert alarm(r) is None
    assert v[1] == 6.0 and v[3] == 4.0 and v[4] == 99.0
    assert mm(moves(r)[0], "X") == 6.0


def test_loop_errors():
    assert alarm(run(wrap("WHILE [1 EQ 1] DO 1\nG00 X1.")))["code"] == "PS0124"
    assert alarm(run(wrap("END 1")))["code"] == "PS0126"
    assert alarm(run(wrap("WHILE [1 EQ 2] DO 4\nEND 4")))["code"] == "PS0126"
    assert alarm(run(wrap("GOTO 77")))["code"] == "PS0078"


def test_g65_argument_specifications_and_locals():
    main = wrap("#1 = 11\nG65 P9010 A1. B2 X3. I4. J5. K6. I7. J8.\n#110 = #1")
    macro = "O9010\n#100 = #1\n#101 = #2\n#102 = #24\n#103 = #4\n#104 = #6\n#105 = #7\n#106 = #8\n#107 = #26\nM99\n"
    r = run(main, extra=[("lib.nc", macro)])
    v = values(r)
    assert alarm(r) is None
    assert v[0] == 1.0 and v[1] == pytest.approx(0.002) and v[2] == 3.0
    assert v[3] == 4.0 and v[4] == 6.0 and v[5] == 7.0 and v[6] == 8.0 and v[7] is None
    assert v[10] == 11.0


def test_adp_restores_the_decimal_point():
    r = run(wrap("G65 P9011 X10 A10"), extra=[("lib.nc", "O9011\n#100 = #24\n#101 = ADP[#24]\n#102 = ADP[#1]\nM99\n")])
    v = values(r)
    assert v[0] == pytest.approx(0.01) and v[1] == pytest.approx(10.0) and v[2] == pytest.approx(10.0)


def test_g65_repeat_and_nesting_limit():
    r = run(wrap("#100 = 0\nG65 P9012 L3"), extra=[("lib.nc", "O9012\n#100 = #100 + 1\nM99\n")])
    assert values(r)[0] == 3.0
    deep = "O9013\n#100 = #100 + 1\nG65 P9013\nM99\n"
    r = run(wrap("#100 = 0\nG65 P9013"), extra=[("lib.nc", deep)])
    assert alarm(r)["code"] == "PS0077" and values(r)[0] == 5.0
    assert alarm(run(wrap("G65 P1234")))["code"] == "PS0076"
    assert alarm(run(wrap("G65 P9014 G01"), extra=[("lib.nc", "O9014\nM99\n")]))["code"] == "PS0129"


def test_m98_subprogram_repeats_and_m99_return_to_n():
    main = wrap("M98 P2000 L3\nM98 P2001\nN30 G00 X99.\nN40 G00 Y7.")
    subs = "O2000\nG91 G00 X1.\nM99\nO2001\nM99 P40\n"
    m = moves(run(main, extra=[("lib.nc", subs)]))
    assert [mm(c, "X") for c in m[:3]] == [1.0, 2.0, 3.0]
    assert len(m) == 4 and mm(m[3], "Y") == 7.0
    m = moves(run(wrap("M98 P30002"), extra=[("lib.nc", "O0002\nG91 X1.\nM99\n")]))
    assert len(m) == 3


def test_g66_modal_call_after_each_move():
    main = wrap("G66 P9020 Z-2.\nG90 G00 X10.\nX20.\nG67\nX30.")
    macro = "O9020\nG91 G01 Z#26 F100.\nG00 Z-#26\nG90\nM99\n"
    m = moves(run(main, extra=[("lib.nc", macro)]))
    assert len(m) == 7
    assert [round(c.target["X"] * 1000, 3) for c in m] == [10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 30.0]
    assert [round(c.target["Z"] * 1000, 3) for c in m[:3]] == [0.0, -2.0, 0.0]


def test_macro_alarm_and_message_stop():
    r = run(wrap("#3000 = 12 (BAD SETUP)"))
    assert alarm(r)["code"] == "3012" and alarm(r)["text"] == "BAD SETUP" and alarm(r)["source"] == "program"
    r = run(wrap("#3006 = 1 (CHECK PART)\nG00 X1."))
    msgs = [c for c in r.commands[0] if c.kind == Kind.MESSAGE]
    assert msgs[0].message == "CHECK PART" and len(moves(r)) == 1


def test_system_variables_positions_offsets_modal_and_clock():
    body = """G90 G54 G00 X10. Y20. Z30.
G01 X12. F250.
#100 = #5001
#101 = #5002
#102 = #4001
#103 = #4014
#104 = #4109
#105 = #5221
#106 = #3901
#107 = #3001
#5221 = -100.
G00 X0
#108 = #5021"""
    r = run(wrap(body), offsets={"G54": [-0.4, -0.2, -0.3]})
    v = values(r)
    assert alarm(r) is None
    assert v[0] == 12.0 and v[1] == 20.0 and v[2] == 1.0 and v[3] == 54.0 and v[4] == 250.0
    assert v[5] == -400.0 and v[6] == 0.0 and v[7] >= 0.0
    assert mm(moves(r)[2], "X") == -100.0
    assert v[8] == -100.0


def test_live_position_read_stops_the_buffer():
    r = run(wrap("G00 X10.\nG01 X20. F600.\n#100 = #5021\nG00 X#100"))
    kinds = [c.kind for c in r.commands[0]]
    sync = kinds.index(Kind.SYNC)
    assert kinds[:sync] == [Kind.RAPID, Kind.LINEAR]
    assert r.commands[0][sync].sync["reason"] == "live_read"
    after = r.commands[0][sync + 1]
    assert after.ready_tick >= r.end_ticks[0][sync]
    assert values(r)[0] == 20.0


def test_skip_g31_sets_skip_variables():
    def probe(path, cmd):
        return ProbeResult(True, {"X": 0.0123, "Y": 0.0, "Z": 0.0}, 0.0)

    r = run(wrap("G90 G00 X0 Y0 Z0\nG31 X50. F100.\n#100 = #5061\nG91 G00 X1."), probe=probe)
    assert values(r)[0] == pytest.approx(12.3)
    probe_cmds = [c for c in r.commands[0] if c.kind == Kind.PROBE]
    assert probe_cmds[0].probe["skip"] == 1 and mm(probe_cmds[0], "X") == 50.0
    assert mm(moves(r)[-1], "X") == pytest.approx(13.3)
    r = run(wrap("G31 X50. F100.\n#100 = #5061"))
    assert values(r)[0] == 50.0


def test_dprnt_output():
    r = run(wrap("#100 = 12.3456\nPOPEN\nDPRNT[X#100[33]*OK]\nPCLOS"))
    assert r.controller.state.paths[0].printed == ["X 12.346 OK"]


def test_common_variables_500_and_indirect_access():
    r = run(wrap("#500 = 3\n#[#500 + 100] = 42\n#100 = #103"))
    assert values(r)[0] == 42.0 and r.controller.state.paths[0].variables["c500"][0] == 3.0


def test_macro_statement_math_in_addresses_rounds_at_increment():
    m = moves(run(wrap("#1 = 12.3456\nG00 X#1 Y[ROUND[#1]]")))
    assert mm(m[0], "X") == 12.346 and mm(m[0], "Y") == 12.346
    assert math.isclose(m[0].target["X"], 0.012346)


def test_builder_macro_call_codes_g_m_and_t():
    from cnctwin.controller.schema import ControllerSettings
    settings = ControllerSettings(macro_call_codes={"M06": 9001, "G100": 9010, "T": 9000})
    lib = "O9001\n#100 = #100 + 1\nM06\nM99\nO9010\n#101 = #24\nM99\nO9000\n#102 = #149\nT#149\nM99\n"
    r = run(wrap("#100 = 0\nT5\nM06\nG100 X7."), settings=settings, extra=[("lib.nc", lib)])
    v = values(r)
    assert alarm(r) is None and v[:3] == [1.0, 7.0, 5.0]
    changes = [c for c in r.commands[0] if c.kind == Kind.TOOL_CHANGE]
    assert [c.tool["number"] for c in changes] == [5]
