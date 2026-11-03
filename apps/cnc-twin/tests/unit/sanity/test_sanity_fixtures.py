"""The program with every mistake once, the clean program, and both input paths."""

import copy

import pytest

from cnctwin.controller.ir import Space
from cnctwin.core.findings import Category, FindingList, Severity
from cnctwin.core.trace import TraceLog
from cnctwin.db.yamlio import load_file
from cnctwin.sanity import RULE_IDS, check_operation, check_program

from sanity_testkit import FIXTURES, fixture_text, setup_data


def _expected():
    return load_file(FIXTURES / "sanity_all_mistakes.expected.yaml").data["findings"]


def _triples(issues):
    return sorted((i.rule, i.line, i.severity.value) for i in issues)


def test_every_mistake_once(ctx):
    report = check_program(fixture_text("sanity_all_mistakes.nc"), "sanity_all_mistakes.nc", ctx)
    expected = sorted((e["rule"], e["line"], e["severity"]) for e in _expected())
    assert _triples(report.issues) == expected
    rules = {e["rule"] for e in _expected()}
    assert rules == set(RULE_IDS) - {"scan-limits"}          # every rule is exercised exactly once
    assert len(rules) == len(_expected())
    assert not report.safe()


def test_clean_program_has_no_findings(ctx):
    report = check_program(fixture_text("sanity_clean.nc"), "sanity_clean.nc", ctx)
    assert [f"{i.line} {i.rule}: {i.message}" for i in report.issues] == []
    assert report.safe()
    assert report.engagement.cut.sum() > 20      # the engagement estimate saw the cuts


def test_findings_carry_line_block_position_and_cause(ctx):
    trace = TraceLog()
    timeline = {"blocks": [{"program": "sanity_all_mistakes.nc", "line": 50, "start_tick": 5_000_000}]}
    report = check_program(fixture_text("sanity_all_mistakes.nc"), "sanity_all_mistakes.nc", ctx, trace=trace,
                           timeline=timeline)
    findings = FindingList()
    report.add_to(findings)
    assert len(findings) == len(_expected()) and [f.id for f in findings] == list(range(len(findings)))
    assert all(f.category == Category.SANITY and f.domain == "sanity" for f in findings)
    inherited = next(f for f in findings if f.message.startswith("feed-inherited"))
    assert inherited.block == "G99 G81 Z-3. R2." and inherited.tick == 5_000_000
    assert inherited.time_s == pytest.approx(5.0)
    chain = trace.chain(inherited.cause_chain[0])
    assert [e.line for e in chain] == [50, 29]          # the finding, then the line that set F600
    assert inherited.pos_W_m is not None and inherited.pos_M_m is not None
    assert inherited.pos_M_m[0] == pytest.approx(inherited.pos_W_m[0] - 0.431)
    travel = next(f for f in findings if f.message.startswith("part-outside-travel"))
    assert travel.severity == Severity.ERROR and travel.quantities["machine_pos_m"] == pytest.approx(-0.840)


def test_controller_command_stream_gives_the_same_findings(ctx):
    text = fixture_text("sanity_all_mistakes.nc")
    own = check_program(text, "sanity_all_mistakes.nc", ctx)
    commands = []
    for command in own.program.commands:
        clone = copy.copy(command)
        clone.space = Space.MACHINE
        clone.tip_target_m = None
        # a controller snapshot with a few keys: the rest is completed from the own scan of the same line
        clone.modal = {k: v for k, v in command.modal.items() if k in ("tool", "work_offset", "length_comp",
                                                                       "length_offset", "feed", "feed_mode")}
        commands.append(clone)
    report = check_program(text, "sanity_all_mistakes.nc", ctx, commands=commands)
    assert report.program.source == "controller"
    assert _triples(report.issues) == _triples(own.issues)


def test_restart_analysis_from_the_controller(ctx):
    records = [{"line": 54, "program": "sanity_clean.nc",
                "issues": [{"group": "distance", "message": "G91 active from line 48", "severity": "error"}]}]
    report = check_program(fixture_text("sanity_clean.nc"), "sanity_clean.nc", ctx, restart_analysis=records)
    assert [(i.rule, i.line, i.severity.value) for i in report.issues] == [("restart-modal", 54, "error")]


def test_check_operation_reads_the_project():
    data = setup_data()
    data["operations"][0]["program_text"] = fixture_text("sanity_clean.nc")
    assert check_operation(data, 0).issues == []
    data["operations"][0].pop("program_text")
    report = check_operation(data, 0, project_dir=FIXTURES)
    assert len(report.issues) == len(_expected())
