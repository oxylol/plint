"""The twin command line (cnctwin/cli.py, bin/twin): dispatch, missing modules, explain, run/check/replay."""

import io
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from cnctwin import cli
from cnctwin.core.findings import Category, Finding, FindingList, Severity
from cnctwin.core.rundir import RunDirectory, write_json_atomic
from cnctwin.core.trace import TraceLog

ROOT = Path(__file__).resolve().parents[3]


def traced_run(tmp_path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    trace = TraceLog(run / "trace.jsonl")
    spindle = trace.record(0.01, 10_000, "motion", "spindle_on", {"omega_rad_s": -1047.2}, [], 3)
    entry = trace.record(0.3, 300_000, "removal", "engagement:13", {"ae_m": 0.004}, [spindle], 13)
    coolant = trace.record(0.05, 50_000, "coolant", "coolant_on", {"flow_m3_s": 2.5e-4}, [], 4)
    force = trace.record(1.2, 1_200_000, "cutting", "force_peak", {"force_N": 271.3}, [entry, spindle], 13)
    modal = trace.record(1.25, 1_250_000, "dynamics", "modal_energy", {"chatter_index": 0.41}, [force, coolant], 13)
    root = trace.record(1.25, 1_250_000, "dynamics", "finding:chatter", {"chatter_freq_Hz": 1834.0}, [modal], 13)
    trace.close()
    findings = FindingList()
    findings.add(Finding(Category.CHATTER, Severity.WARNING, "Chatter at 1834 Hz", 1.25, 1_250_000, "dynamics",
                         op="OP10", program="op10.nc", line=13, block="G1 X40 F800", pos_W_m=(0.01, 0.02, -0.002),
                         quantities={"chatter_freq_Hz": 1834.0}, cause_chain=[root, modal, force, coolant, entry, spindle]))
    findings.add(Finding(Category.AIR_CUTTING, Severity.INFO, "air", 0.1, 100_000, "removal", cause_chain=[]))
    RunDirectory(run).write_findings(findings)
    return run


def test_explain_prints_the_cause_chain(tmp_path):
    run = traced_run(tmp_path)
    out = io.StringIO()
    assert cli.explain(run, 0, out=out) == 0
    text = out.getvalue()
    assert "Finding 0: chatter (warning) at OP10 op10.nc line 13" in text
    assert "tool tip in W: 10.0000, 20.0000, -2.0000 mm" in text
    lines = text.splitlines()
    chain = [line for line in lines if "#" in line and "t=" in line]
    assert chain[0].lstrip().startswith("#5") and "finding:chatter" in chain[0]
    assert any("└─ #4" in line for line in chain) and any("(see above)" in line for line in chain)
    assert "Root causes: #0 motion spindle_on; #2 coolant coolant_on" in text
    listing = io.StringIO()
    cli.explain(run, None, out=listing)
    assert "chatter" in listing.getvalue() and "air_cutting" in listing.getvalue()
    as_json = io.StringIO()
    cli.explain(run, 0, as_json=True, out=as_json)
    document = json.loads(as_json.getvalue())
    assert [e["id"] for e in document["chain"]][:3] == [5, 4, 3] and {r["id"] for r in document["roots"]} == {0, 2}
    empty = io.StringIO()
    cli.explain(run, 1, out=empty)
    assert "Cause chain: none recorded" in empty.getvalue()
    with pytest.raises(cli.UsageError):
        cli.explain(run, 9)


def test_explain_reads_the_synthetic_run(tmp_path):
    sys.path.insert(0, str(ROOT / "tests" / "fixtures"))
    try:
        from make_synthetic_run import make_synthetic_run
    finally:
        sys.path.pop(0)
    manifest = make_synthetic_run(tmp_path / "syn", seed=2, until_s=1.0, media=False)
    findings = json.loads((tmp_path / "syn" / "findings.json").read_text())["findings"]
    assert manifest["findings"] == len(findings) >= 1
    for finding in findings:
        out = io.StringIO()
        assert cli.explain(tmp_path / "syn", finding["id"], out=out) == 0
        assert f"Finding {finding['id']}:" in out.getvalue()


def test_main_dispatch_and_missing_modules(capsys):
    assert cli.main([]) == cli.EXIT_USAGE
    # a subcommand whose module is missing from a build exits EXIT_MISSING and names the module
    with pytest.raises(cli.Missing, match="cnctwin.not_in_this_build.main is not available in this build"):
        cli.resolve("cnctwin.not_in_this_build", "main")
    cli.FORWARDED["ghost"] = ("cnctwin.not_in_this_build", "main", "test-only command")
    try:
        assert cli.main(["ghost"]) == cli.EXIT_MISSING
        assert "not available in this build (no module 'cnctwin.not_in_this_build')" in capsys.readouterr().err
    finally:
        del cli.FORWARDED["ghost"]
    assert cli.main(["db", "check", "--help"]) == 0
    assert "twin db check" in capsys.readouterr().out
    assert cli.main(["new", "--help"]) == 0
    assert cli.main(["explain", "/nonexistent/run"]) == cli.EXIT_ERROR
    assert "has no findings.json" in capsys.readouterr().err


@pytest.fixture
def fake_loop(monkeypatch):
    calls = {}
    module = types.ModuleType("cnctwin.core.loop")

    def run_project(project_path, run_dir, options, progress_cb):
        calls["run"] = (project_path, run_dir, dict(options))
        progress_cb(sim_time_s=0.5, est_total_s=1.0, op="OP10")
        safe = not options.get("unsafe")
        write_json_atomic(Path(run_dir) / "run.json", {"verdict": {"safe": safe, "reasons": [] if safe else ["collision"]}})
        return {"ok": True}

    def check_project(request, out_dir, progress_cb):
        calls["check"] = (request, out_dir)
        return {"ok": not request["options"].get("bad"), "summary": "checked", "errors": []}

    def replay_window(run_dir, tick0, tick1, options, progress_cb):
        calls["replay"] = (run_dir, tick0, tick1)
        return {"files": ["series/fast/replay.parquet"]}

    module.run_project, module.check_project, module.replay_window = run_project, check_project, replay_window
    monkeypatch.setitem(sys.modules, "cnctwin.core.loop", module)
    return calls


def test_run_check_and_replay_through_the_loop(tmp_path, fake_loop, capsys):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "project.yaml").write_text("format: cnctwin-project\n")
    out = tmp_path / "runs" / "r1"
    assert cli.main(["run", str(project), "--out", str(out), "--set", "a.b=2", "--set", "name=x", "--quiet"]) == 0
    path, run_dir, options = fake_loop["run"]
    assert path == (project / "project.yaml").resolve() and run_dir == out and options == {"a": {"b": 2}, "name": "x"}
    assert "SAFE TO RUN" in capsys.readouterr().out
    status = json.loads((out / "status.json").read_text())
    assert status["status"] == "done" and status["progress"]["sim_time_s"] == 0.5 and status["op"] == "OP10"
    assert cli.main(["run", str(project), "--out", str(out), "--set", "unsafe=true", "--quiet"]) == cli.EXIT_UNSAFE
    assert "collision" in capsys.readouterr().out
    (out / "checkpoints").mkdir()
    (out / "checkpoints" / "ck_5000.json").write_text("{}")
    (out / "checkpoints" / "ck_5000.npz").write_bytes(b"")
    assert cli.main(["run", str(project), "--out", str(out), "--resume", "--quiet"]) == 0
    assert fake_loop["run"][2]["resume_from"] == str((out / "checkpoints" / "ck_5000.json").resolve())
    assert cli.main(["run", str(project), "--resume"]) == cli.EXIT_ERROR
    assert cli.main(["check", str(project), "--out", str(tmp_path / "c")]) == 0
    request, _ = fake_loop["check"]
    assert request["read_only"] and request["project_path"] == str((project / "project.yaml").resolve())
    assert cli.main(["check", str(project), "--write", "--set", "bad=1", "--out", str(tmp_path / "c")]) == cli.EXIT_ERROR
    write_json_atomic(out / "run.json", {"clock": {"tick_s": 1e-6}, "verdict": {"safe": True}})
    assert cli.main(["replay", str(out), "--window", "0.5", "0.75", "--quiet"]) == 0
    assert fake_loop["replay"] == (out, 500_000, 750_000)
    assert cli.main(["run", str(tmp_path / "nothing"), "--out", str(out)]) == cli.EXIT_ERROR

    def broken(project_path, run_dir, options, progress_cb):
        from cnctwin.core.errors import ConservationError
        raise ConservationError("mass", 1.0, 2.0, 1e-9, 0.5)
    sys.modules["cnctwin.core.loop"].run_project = broken
    assert cli.main(["run", str(project), "--out", str(out), "--quiet"]) == cli.EXIT_ERROR
    status = json.loads((out / "status.json").read_text())
    assert status["status"] == "failed" and status["error"]["type"] == "ConservationError"


def test_operator_options_run_through_the_runner_that_applies_them(tmp_path, fake_loop, monkeypatch, capsys):
    """The loop's run_project does not read options.operator: twin run sends such a run through cnctwin.sweep.runner
    (normalised, with that runner in its cache key); overrides at 100 % are the plain run through the loop."""
    from cnctwin import cache
    from cnctwin.core.errors import InputError
    from cnctwin.sweep import runner
    project = tmp_path / "proj"
    project.mkdir()
    (project / "project.yaml").write_text("format: cnctwin-project\n")
    calls = {}

    def override_run(project_path, run_dir, options, progress_cb):
        calls["options"] = dict(options)
        write_json_atomic(Path(run_dir) / "run.json", {"verdict": {"safe": True, "reasons": []}})
        return {"operator": options["operator"]}

    def key_for(project_path, options, runner=None, **kwargs):
        calls["key_runner"] = runner
        raise InputError("not a resolvable project (test)")

    monkeypatch.setattr(runner, "run_project", override_run)
    monkeypatch.setattr(cache, "key_for", key_for)
    monkeypatch.setenv("CNCTWIN_CACHE", "on")
    for name, value in cache.DETERMINISTIC_ENV.items():
        monkeypatch.setenv(name, value)
    out = tmp_path / "runs" / "r1"
    assert cli.main(["run", str(project), "--out", str(out), "--set", "operator.feed_override=1.2",
                     "--set", "operator.rapid_override=1", "--quiet"]) == 0
    assert calls["options"] == {"operator": {"feed_override": {"*": 1.2}}}         # normalised: 100 % dropped
    assert calls["key_runner"] == cache.SWEEP_RUNNER and "run" not in fake_loop
    calls.clear()
    assert cli.main(["run", str(project), "--out", str(out), "--set", "operator.feed_override=1.0", "--quiet"]) == 0
    assert fake_loop["run"][2] == {} and "options" not in calls and calls["key_runner"] == cache.DEFAULT_RUNNER
    assert cli.main(["run", str(project), "--out", str(out), "--set", "operator.torque_override=2"]) == cli.EXIT_ERROR
    assert "options.operator.torque_override: one of feed_override" in capsys.readouterr().err


def test_check_writes_into_the_data_folder_not_the_project(tmp_path, fake_loop, monkeypatch, capsys):
    project = tmp_path / "examples" / "pocket"
    project.mkdir(parents=True)
    (project / "project.yaml").write_text("format: cnctwin-project\n")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    assert cli.main(["check", str(project)]) == 0
    _, out_dir = fake_loop["check"]
    assert out_dir.parent == tmp_path / "data" / "checks" and out_dir.name.startswith("pocket-") and out_dir.is_dir()
    assert not (project / ".twin-check").exists() and f"check files: {out_dir}" in capsys.readouterr().err
    other = tmp_path / "elsewhere" / "pocket"                   # the same folder name elsewhere: another folder
    other.mkdir(parents=True)
    (other / "project.yaml").write_text("format: cnctwin-project\n")
    assert cli.default_check_dir(other / "project.yaml") != out_dir
    import tempfile
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    third = tmp_path / "third" / "pocket"
    third.mkdir(parents=True)
    (tmp_path / "data" / "checks").chmod(0o500)               # the data folder cannot be written: the temporary one
    try:
        fallback = cli.default_check_dir(third / "project.yaml")
    finally:
        (tmp_path / "data" / "checks").chmod(0o700)
    assert fallback.parent == tmp_path / "tmp" / "cnctwin-check" and fallback.name.startswith("pocket-")


def test_run_allocates_a_deterministic_run_id(tmp_path, fake_loop, monkeypatch):
    project = tmp_path / "pocket"
    project.mkdir()
    (project / "project.yaml").write_text("format: cnctwin-project\n")
    monkeypatch.setattr(cli, "_project_hash", lambda p: "ab" * 32)
    assert cli.main(["run", str(project), "--runs", str(tmp_path / "runs"), "--quiet"]) == 0
    from cnctwin.core.rundir import run_id
    assert fake_loop["run"][1].name == run_id("ab" * 32, {}, 1, "pocket")


def test_export_through_the_cli(tmp_path, capsys):
    from cnctwin.core.channels import Channel, ChannelRegistry, Tier
    from cnctwin.core.recorder import Recorder
    reg = ChannelRegistry()
    reg.register(Channel("m.x_m", "m", Tier.MACRO, "motion", "x"))
    rec = Recorder(tmp_path, reg, l0_ticks_min=100, ipo_ticks=1000, macro_ipo=10, recording={"fast": "none"})
    for n in range(3):
        rec.record_macro((n + 1) * 10_000, {"m.x_m": 0.5 * n})
        rec.end_step(n * 10_000, (n + 1) * 10_000)
    rec.close_parts()
    assert cli.main(["export", str(tmp_path), "--tier", "macro"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == ['"tick","time_s","m.x_m"', "10000,0.01,0", "20000,0.02,0.5", "30000,0.03,1"]


def test_bin_twin_runs_the_cli():
    result = subprocess.run([str(ROOT / "bin" / "twin"), "--help"], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0 and "usage: twin" in result.stdout
    unknown = subprocess.run([str(ROOT / "bin" / "twin"), "no-such-command"], capture_output=True, text=True,
                             timeout=120)
    assert unknown.returncode == cli.EXIT_USAGE
