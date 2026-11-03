"""Comparison of two runs: findings matched by category and the same execution of a line, by line or
feature and program progress, or not at all; characteristics, cycle time per category, line and tool,
tool wear, energy and cost, inputs; the API entry, the command line and the HTML diff."""

from __future__ import annotations

import json

import pytest
from htmlcheck import check

from cnctwin.report.compare import compare, compare_html, compare_runs, main


def test_unsafe_against_safe(runs):
    doc = compare(runs["unsafe"], runs["safe"])
    assert doc["verdict"] == {"a_safe": False, "b_safe": True, "changed": True}
    findings = doc["findings"]
    assert findings["counts"]["matched"] == 4 and findings["counts"]["resolved"] == 1 and findings["counts"]["new"] == 0
    assert [f["category"] for f in findings["only_a"]] == ["out_of_tolerance"]
    assert {p["matched_by"] for p in findings["matched"]} == {"same execution of the line"}
    chatter = next(p for p in findings["matched"] if p["category"] == "chatter")
    assert set(chatter["quantities"]) == {"chatter_freq_Hz", "chatter_index"}
    depth = next(d for d in doc["dimensions"] if d["id"] == "pocket-depth")
    assert depth["change"] == "better" and depth["a_pass"] is False and depth["b_pass"] is True
    assert depth["delta_m"] == pytest.approx(0.002011 - 0.002062)
    assert doc["cycle"]["total"]["delta"] == pytest.approx(0.0) and doc["cycle"]["total"]["change"] == "same"
    assert any(i["file"] == "project.yaml" and i["state"] == "changed" for i in doc["inputs"])
    assert doc["summary"][0] == "verdict changed: not safe to run → safe to run"
    json.dumps(doc, allow_nan=False)


def test_a_run_against_itself_changes_nothing(runs):
    doc = compare(runs["unsafe"], runs["twin"])
    assert doc["findings"]["counts"]["matched"] == 5 and not doc["findings"]["only_a"] and not doc["findings"]["only_b"]
    assert all(p["severity_change"] == "same" and p["dt_s"] == 0.0 for p in doc["findings"]["matched"])
    assert all(d["change"] == "same" and d["delta_m"] == 0.0 for d in doc["dimensions"])
    assert all(r["delta"] == 0.0 for r in doc["cycle"]["lines"])
    assert all(i["state"] == "same" for i in doc["inputs"])
    assert doc["energy"]["total_J"]["change"] == "same"


def _write_run(folder, *, scale: int, findings: list[dict], with_timeline: bool = True):
    """A minimal finished run: program lines 20, 21, 20 (a loop) of ``1000 * scale`` ticks each."""
    folder.mkdir(parents=True)
    blocks = [{"index": i, "op": "OP10", "program": "p.nc", "line": line, "block": f"N{line}", "path": 0,
               "kind": "linear", "tick0": i * 1000 * scale, "tick1": (i + 1) * 1000 * scale}
              for i, line in enumerate((20, 21, 20))]
    if with_timeline:
        (folder / "timeline.json").write_text(json.dumps({"tick_s": 1e-6, "blocks": blocks}))
    (folder / "findings.json").write_text(json.dumps({"findings": findings}))
    (folder / "inspection.json").write_text(json.dumps({"tolerances": [
        {"id": "bore-dia", "kind": "diameter", "nominal_m": 0.008, "plus_m": 1.5e-5, "minus_m": 0.0,
         "measured_m": 0.008004, "pass": True}]}))
    (folder / "run.json").write_text(json.dumps({"clock": {"tick_s": 1e-6}, "verdict": {"safe": True, "reasons": []},
                                                 "summary": {"cycle_time_s": 3000 * scale * 1e-6}}))


def _finding(ident, category, line, tick, message="m", severity="warning"):
    return {"id": ident, "category": category, "severity": severity, "message": message, "time_s": tick * 1e-6,
            "tick": tick, "domain": "test", "op": "OP10", "program": "p.nc", "line": line, "quantities": {},
            "cause_chain": [ident]}


def test_matching_passes(tmp_path):
    a = [_finding(0, "chatter", 20, 2500), _finding(1, "gouge", 21, 100),
         _finding(2, "out_of_tolerance", 0, 3000, "bore-dia 8.004 mm"), _finding(3, "burr", 21, 300)]
    b = [_finding(0, "chatter", 20, 5000), _finding(1, "gouge", 21, 240),
         _finding(2, "out_of_tolerance", 0, 6000, "bore-dia 8.016 mm", "error"), _finding(3, "burr", 21, 5900)]
    _write_run(tmp_path / "a", scale=1, findings=a)
    _write_run(tmp_path / "b", scale=2, findings=b)                          # every block twice as long
    doc = compare(tmp_path / "a", tmp_path / "b")
    by_category = {p["category"]: p for p in doc["findings"]["matched"]}
    assert by_category["chatter"]["matched_by"] == "same execution of the line"      # second run of line 20 in both
    assert by_category["chatter"]["a"]["execution"] == 1 and by_category["chatter"]["b"]["execution"] == 1
    assert by_category["gouge"]["matched_by"] == "same line or feature, nearest progress"   # tick in no line-21 block
    assert by_category["out_of_tolerance"]["matched_by"] == "same line or feature, nearest progress"
    assert by_category["out_of_tolerance"]["feature"] == "bore-dia"
    assert by_category["out_of_tolerance"]["severity_change"] == "worse"
    assert "burr" not in by_category                                          # 10 % against 98 % of the cycle
    assert [f["id"] for f in doc["findings"]["only_a"]] == [3] and [f["id"] for f in doc["findings"]["only_b"]] == [3]


def test_progress_window(tmp_path):
    a = [_finding(0, "chip_packing", 21, 1500), _finding(1, "chip_packing", 21, 1600)]
    b = [_finding(0, "chip_packing", 21, 1550), _finding(1, "chip_packing", 21, 2000)]
    _write_run(tmp_path / "a", scale=1, findings=a, with_timeline=False)
    _write_run(tmp_path / "b", scale=1, findings=b, with_timeline=False)
    doc = compare(tmp_path / "a", tmp_path / "b", {"finding_match_progress_frac": 0.2})
    pairs = sorted((p["a"]["id"], p["b"]["id"]) for p in doc["findings"]["matched"])
    assert pairs == [(0, 0), (1, 1)]                                           # optimal assignment by progress
    doc = compare(tmp_path / "a", tmp_path / "b", {"finding_match_progress_frac": 0.01})
    assert not doc["findings"]["matched"] and len(doc["findings"]["only_a"]) == 2


def test_api_entry_cli_and_html(runs, tmp_path, capsys):
    calls = []
    request = {"run_a": "a", "run_b": "b", "run_a_dir": str(runs["unsafe"]), "run_b_dir": str(runs["safe"]),
               "options": {}}
    result = compare_runs(request, tmp_path / "job", lambda *a, **k: calls.append(k))
    assert calls and (tmp_path / "job" / "compare.html").is_file()
    stored = json.loads((tmp_path / "job" / "compare.json").read_text())
    assert stored["summary"] == result["summary"] and stored["format"] == "cnctwin-compare"
    checker = check((tmp_path / "job" / "compare.html").read_text(encoding="utf-8"))
    assert {"findings", "dimensions", "cycle", "tools", "energy", "surface", "inputs"} <= set(checker.ids)
    assert compare_html(stored) == (tmp_path / "job" / "compare.html").read_text(encoding="utf-8")
    assert main([str(runs["unsafe"]), str(runs["safe"]), "--out", str(tmp_path / "cli")]) == 0
    assert "verdict changed" in capsys.readouterr().out
    assert main([str(runs["unsafe"]), str(tmp_path / "missing")]) == 1
