from cnctwin.controller.alarms import AlarmTable, load_controller_db
from cnctwin.controller.dialects.fanuc_tables import ALARM_CODES, ALARM_TEXTS
from cnctwin.controller.ir import Kind
from harness import FANUC_DB, HAAS_MILL, MILL, alarm, moves, run, wrap


def test_unknown_g_code_is_ps0010():
    r = run(wrap("G00 X1.\nG99.9 X2."))
    a = alarm(r)
    assert a["code"] == "PS0010" and a["text"] == "IMPROPER G-CODE" and a["source"] == "db:fixture"
    assert len(moves(r)) == 1
    alarms = [c for c in r.commands[0] if c.kind == Kind.ALARM]
    assert alarms and alarms[0].alarm["code"] == "PS0010" and alarms[0].ref.line == 4


def test_address_without_function_or_axis_is_ps0009():
    assert alarm(run(wrap("G00 A10.")))["code"] == "PS0009"
    assert alarm(run(wrap("G00 X1. Q5.")))["code"] == "PS0009"
    assert alarm(run(wrap("G01 X1. F100. L2")))["code"] == "PS0009"


def test_same_group_rule_fanuc_last_wins_haas_alarms():
    r = run(wrap("G00 G01 X1. F100."))
    assert alarm(r) is None and moves(r)[0].kind == Kind.LINEAR
    assert any("G00 and G01" in e.get("message", "") for e in r.events[0])
    h = run(wrap("G00 G01 X1. F100."), HAAS_MILL)
    assert alarm(h)["code"] == "339" and alarm(h)["text"] == "Multiple Codes"


def test_two_axis_word_one_shots_in_one_block():
    assert alarm(run(wrap("G28 G53 X0")))["code"] == "PS0010"


def test_undefined_m_code_uses_the_database_kind_entry():
    a = alarm(run(wrap("M77")))
    assert a["code"] == "TW-M" and a["text"].startswith("M-CODE ERROR")
    assert alarm(run(wrap("M77"), HAAS_MILL))["code"] == "329"


def test_builtin_text_when_the_database_lacks_the_code():
    a = alarm(run(wrap("#100 = 1 / 0")))
    assert a["code"] == "PS0112" and a["text"] == "ZERO DIVIDE" and a["source"] == "builtin"


def test_unsupported_features_raise_twin_alarms(monkeypatch):
    from cnctwin.controller import hooks
    hooks.load_plugins()
    for table, key in ((hooks.COMPENSATION, "cutter"), (hooks.CYCLES, "G81"), (hooks.TRANSFORMS, "fanuc:G43.4")):
        monkeypatch.delitem(table, key, raising=False)      # the missing-provider path (providers exist now)
    assert alarm(run(wrap("G41 G01 X1. D1 F100.")))["code"] == "TW0002"
    assert alarm(run(wrap("G81 X1. Y1. Z-1. R1. F100.")))["code"] == "TW0001"
    assert alarm(run(wrap("G43.4 H1")))["code"] == "TW0003"
    assert alarm(run(wrap("#100 = #1000")))["code"] == "TW0007"
    assert alarm(run(wrap("#100 = #4201")))["code"] == "TW0006"


def test_end_of_program_text_without_m30():
    assert alarm(run("%\nO0001\nG00 X1.\n%\n"))["code"] == "PS5010"


def test_alarm_table_resolution_order():
    table = AlarmTable("fanuc", ALARM_CODES, ALARM_TEXTS, FANUC_DB, "fixture")
    assert table.resolve("improper_g_code").as_dict()["source"] == "db:fixture"
    assert table.resolve("zero_divide").source == "builtin"
    assert table.resolve("undefined_m_code").code == "TW-M"
    assert table.resolve("cycle_unavailable").code == "TW0001"
    assert table.program_alarm(5, "X", 3000).code == "3005"


def test_real_controller_database_files_load_when_present():
    data = load_controller_db("haas-ngc")
    if data is None:
        return
    assert data["dialect"] == "haas"
    table = AlarmTable("haas", {"pocket_definition": "370"}, {"370": "x"}, data, "haas-ngc")
    assert table.resolve("pocket_definition").source == "db:haas-ngc"
    assert load_controller_db("no-such-controller") is None
    assert MILL["controller"]["dialect"] == "fanuc"
