"""twin run through the run cache, twin run --no-cache, the cache/sweep/dataset subcommands and twin replay of a
run made with operator overrides (cnctwin/cli.py). The loop is a fake module (fake_runner.run_project)."""

from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path

import pytest

import fake_runner
from cnctwin import cache as C
from cnctwin import cli
from cnctwin.core.warmup import mill_project


@pytest.fixture
def project(tmp_path_factory) -> Path:
    return mill_project(tmp_path_factory.mktemp("mill") / "mill")


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "build_id", lambda package_dir=None: "b" * 64)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("CNCTWIN_CACHE_DIR", raising=False)
    log = tmp_path / "calls.log"
    monkeypatch.setenv("FAKE_RUNNER_LOG", str(log))
    module = types.ModuleType("cnctwin.core.loop")
    module.run_project = fake_runner.run_project
    replays = []
    module.replay_window = lambda run_dir, t0, t1, options, progress: replays.append(("loop", t0, t1)) or {}
    monkeypatch.setitem(sys.modules, "cnctwin.core.loop", module)
    return {"log": log, "replays": replays, "data": tmp_path / "data"}


def calls(log: Path) -> int:
    return len(log.read_text().splitlines()) if log.is_file() else 0


def test_twin_run_reuses_the_cached_run(project, env, tmp_path, capsys):
    assert cli.main(["run", str(project), "--quiet"]) == 0
    assert calls(env["log"]) == 1
    first = capsys.readouterr()
    run_dir = Path(first.err.split("run directory: ")[1].splitlines()[0])
    assert run_dir.parent == env["data"] / "runs" and "SAFE TO RUN" in first.out
    entry = C.RunCache(env["data"] / "cache").entries()[0]
    assert entry.run_dir == str(run_dir.resolve()) and not entry.owned          # twin run's runs are references
    assert cli.main(["run", str(project), "--quiet"]) == 0
    second = capsys.readouterr()
    assert calls(env["log"]) == 1 and f"cached run: {run_dir.resolve()}" in second.err and "SAFE TO RUN" in second.out
    out = tmp_path / "copy"
    assert cli.main(["run", str(project), "--out", str(out), "--quiet"]) == 0
    assert calls(env["log"]) == 1 and (out / "run.json").is_file()
    assert json.loads((out / "status.json").read_text())["status"] == "done"
    assert C.run_digest(out)[0] == entry.digest
    assert cli.main(["run", str(project), "--no-cache", "--quiet"]) == 0
    assert calls(env["log"]) == 2
    assert cli.main(["run", str(project), "--set", "max_sim_s=0.5", "--quiet"]) == 0
    assert calls(env["log"]) == 3


def test_twin_run_without_the_deterministic_environment_is_not_cached(project, env, monkeypatch, capsys):
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    assert cli.main(["run", str(project), "--quiet"]) == 0
    assert "not cached" in capsys.readouterr().err and C.RunCache(env["data"] / "cache").entries() == []


def test_replay_of_an_override_run_goes_through_the_sweep_runner(env, tmp_path, monkeypatch):
    import cnctwin.sweep.runner as runner
    seen = []
    monkeypatch.setattr(runner, "replay_window", lambda run_dir, t0, t1, options, progress: seen.append((t0, t1)) or {})
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"clock": {"tick_s": 1e-6}, "options": {"operator": {
        "feed_override": {"*": 1.2}}}}))
    assert cli.main(["replay", str(run), "--window", "0.1", "0.2", "--quiet"]) == 0
    assert seen == [(100_000, 200_000)] and env["replays"] == []
    (run / "run.json").write_text(json.dumps({"clock": {"tick_s": 1e-6}, "options": {}}))
    assert cli.main(["replay", str(run), "--window", "0.1", "0.2", "--quiet"]) == 0
    assert env["replays"] == [("loop", 100_000, 200_000)]


def test_the_new_subcommands_are_dispatched(tmp_path, capsys):
    assert cli.main(["cache", "--root", str(tmp_path / "c"), "list"]) == 0
    assert "0 entries" in capsys.readouterr().out
    assert cli.main(["sweep", "--help"]) == 0 and "twin sweep" in capsys.readouterr().out
    assert cli.main(["dataset", "--help"]) == 0 and "twin dataset" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["run", "--help"])
    assert "--no-cache" in capsys.readouterr().out
