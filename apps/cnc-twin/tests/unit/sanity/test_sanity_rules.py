"""Single rules on small programs (conditions, severities, edge paths)."""

import pytest

from cnctwin.sanity import check_program

from sanity_testkit import context_with, program

HEAD = ("G21 G17 G40 G49 G80 G90 G94", "T1 M06", "G90 G54 G00 X30. Y25. S12000 M03", "G43 H01 Z50. M08", "Z2.")
TAIL = ("G00 Z50.", "M05", "M09", "G53 Z0.", "M30")


def _issues(ctx, *lines, **kw):
    return [(i.rule, i.line, i.severity.value) for i in check_program(program(*lines), "t.nc", ctx, **kw).issues]


def test_clean_skeleton(ctx):
    assert _issues(ctx, *HEAD, "G01 Z-2. F1800.", "X70.", *TAIL) == []


def test_unsupported_dialect_without_commands_is_reported():
    ctx = context_with(lambda d: d["machine"]["controller"].update({"dialect": "heidenhain_klartext"}))
    assert _issues(ctx, "L X+10 Y+5 R0 FMAX") == [("scan-limits", 0, "info")]


def test_macros_are_reported_not_ignored(ctx):
    issues = _issues(ctx, *HEAD, "#100 = -2.", "G01 Z#100 F1800.", "X70.", *TAIL)
    assert ("scan-limits", 6, "info") in issues and ("scan-limits", 7, "info") in issues


def test_no_end_of_program(ctx):
    assert _issues(ctx, *HEAD, "G01 Z-2. F1800.", "X70.", "G00 Z50.", "M05", "M09") == [("program-end", 10, "error")]


def test_arc_alarm_and_unreachable_radius(ctx):
    alarm = _issues(ctx, *HEAD, "G01 Z-2. F1800.", "X70.", "G02 X75.05 Y20. I0. J-5.", *TAIL)
    assert ("arc-endpoint", 8, "error") in alarm
    short = _issues(ctx, *HEAD, "G01 Z-2. F1800.", "X70.", "G02 X70. Y45. R5.", *TAIL)
    assert ("arc-endpoint", 8, "error") in short


def test_length_offset_of_another_tool(ctx):
    other = _issues(ctx, *HEAD[:3], "G43 H02 Z50. M08", "Z2.", "G01 Z-2. F1800.", "X70.", *TAIL)
    assert other == [("offset-length", 4, "error")]
    same = context_with(lambda d: d["tools"]["list"][1].update({"length_offset_mm": 120.0}))
    assert _issues(same, *HEAD[:3], "G43 H02 Z50. M08", "Z2.", "G01 Z-2. F1800.", "X70.", *TAIL) == \
        [("offset-length", 4, "warning")]


def test_unset_length_offset_and_measurement(ctx):
    unset = context_with(lambda d: d["tools"]["list"][0].update({"length_offset_mm": None}))
    assert ("offset-length", 4, "error") in _issues(unset, *HEAD, "G01 Z-2. F1800.", "X70.", *TAIL)
    measured = _issues(unset, "T1 M06", "G37 H01", "G90 G54 G00 X30. Y25. S12000 M03", "G43 H01 Z50. M08", "Z2.",
                       "G01 Z-2. F1800.", "X70.", *TAIL)
    assert not [i for i in measured if i[0] == "offset-length"]


def test_tapping_feed_must_equal_pitch(ctx):
    lines = ("T3 M06", "G90 G54 G00 X20. Y15. S3000 M03", "G43 H03 Z25. M08", "G98 G81 Z-16. R2. F240.", "G80",
             "G53 Z0.", "T4 M06", "G90 G54 G00 X20. Y15. S500 M03", "G43 H04 Z25. M08",
             "G95 G98 G84 Z-12. R4. F1.0", "G80", "G94", "M05", "M09", "G53 Z0.", "M30")
    assert _issues(ctx, *lines) == [("cutting-data", 10, "error")]


def test_coolant_rules_for_material_and_tool_class():
    magnesium = context_with(lambda d: d["material"].update({"family": "magnesium", "iso_group": "N"}))
    assert ("coolant-required", 6, "error") in _issues(magnesium, *HEAD, "G01 Z-2. F1800.", "X70.", *TAIL)
    ceramic = context_with(lambda d: d["tools"]["list"][0]["tool"]["tool_material"].update({"class": "ceramic"}))
    assert ("coolant-required", 6, "warning") in _issues(ceramic, *HEAD, "G01 Z-2. F1800.", "X70.", *TAIL)


def test_through_spindle_coolant_on_a_tool_without_holes():
    def tsc(d):
        d["machine"]["coolant"]["pumps"].append({"id": "tsc", "kind": "through_spindle", "power_kW": 1.1})
    ctx = context_with(tsc)
    assert _issues(ctx, *HEAD[:3], "G43 H01 Z50. M88", "Z2.", "G01 Z-2. F1800.", "X70.", "G00 Z50.", "M05",
                   "M89", "G53 Z0.", "M30") == [("coolant-unavailable", 4, "warning")]


def test_vise_jaws_and_retract_plane():
    def vise(d):
        d["operations"][0]["setup"]["workholding"] = [
            {"device": {"id": "fixture-vise", "type": "vise", "jaw_width_mm": 150, "jaw_height_mm": 40},
             "grip_depth_mm": 35}]
    ctx = context_with(vise)
    assert len(ctx.clamps_M) == 2 and ctx.clamps_W("G54")[0].top == pytest.approx(0.005)
    lines = ("T2 M06", "G90 G54 G00 X20. Y15. S3000 M03", "G43 H02 Z25. M08", "G99 G81 Z-3. R2. F300.", "X80.",
             "G80", "M05", "M09", "G53 Z0.", "M30")
    assert _issues(ctx, *lines) == [("retract-below-clamps", 4, "warning")]


def test_unknown_clamp_geometry_is_reported():
    ctx = context_with(lambda d: d["operations"][0]["setup"]["workholding"][0].pop("outline_mm"))
    assert ("retract-below-clamps", 0, "info") in _issues(ctx, *HEAD, "G01 Z-2. F1800.", "X70.", *TAIL)


def test_rapid_into_material_and_power_on_offset(ctx):
    lines = ("T1 M06", "G00 X30. Y25. S12000 M03", "G43 H01 Z50. M08", "Z-1.", "G01 X70. F1800.", *TAIL)
    issues = _issues(ctx, *lines)
    assert ("rapid-below-stock", 4, "error") in issues and ("work-offset-unset", 2, "warning") in issues


def test_unresolved_tool_is_reported():
    ctx = context_with(lambda d: d["tools"]["list"][0].update({"tool": "some-db-id"}))
    issues = _issues(ctx, *HEAD, "G01 Z-2. F1800.", "X70.", *TAIL)
    assert issues == [("scan-limits", 3, "info")]


def test_lathe_rules():
    def lathe(d):
        d["machine"]["type"] = "lathe"
        d["stock"] = {"kind": "cylinder", "size_mm": [50, 100]}
        d["tools"]["list"][0]["tool"]["type"] = "turning_insert"
        d["tools"]["list"][0]["tool"]["cutting_data"] = [
            {"iso_group": "N", "vc_m_min": {"min": 200, "max": 800}, "f_mm": {"min": 0.05, "max": 0.4}}]
    ctx = context_with(lathe)
    good = ("G18 G21 G99 G54", "T0101", "G50 S3000", "G96 S300 M03", "G00 X52. Z2. M08", "G01 X40. F0.2",
            "Z-20.", "G00 X60. Z5.", "G28 U0. W0.", "M05", "M09", "M30")
    assert _issues(ctx, *good) == []
    no_offset = list(good)
    no_offset[1] = "T0100"
    assert ("offset-length", 6, "error") in _issues(ctx, *no_offset)
    absolute = list(good)
    absolute[8] = "G28 X0. Z0."
    assert ("g28-absolute", 9, "error") in _issues(ctx, *absolute)
