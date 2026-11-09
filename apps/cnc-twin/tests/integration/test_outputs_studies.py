"""The studies cnctwin.core.outputs builds on whole runs (docs/outputs.md 3): the simulate entry resolves to the
run loop of this build, a replay window goes through the runner and lands in the run's series index, two runs
compare. The runner here is tests/integration/test_outputs_fakes.py (the real loop's replay is tested in
test_loop_determinism.py)."""

from __future__ import annotations

import json
import shutil

import pytest

from cnctwin.core.outputs import (SIMULATE_ENTRY, compare_study, nearest_checkpoint, replay_window, resolve_entry,
                                  run_document)
from tests.integration import test_outputs_fakes as fakes

FAKE = "tests.integration.test_outputs_fakes:fake_run_project"


def test_the_simulate_entry_is_the_run_loop(monkeypatch):
    monkeypatch.delenv("CNCTWIN_ENTRY_SIMULATE", raising=False)
    from cnctwin.core.loop import run_project
    assert SIMULATE_ENTRY == "cnctwin.core.loop:run_project" and resolve_entry() is run_project


@pytest.fixture()
def run_copy(tmp_path_factory, tmp_path):
    runs = fakes.templates(tmp_path_factory)
    target = tmp_path / "00003-run"
    shutil.copytree(runs["safe"], target)
    document = json.loads((target / "run.json").read_text())
    document.update({"project_dir": str(runs["safe_project"].parent), "project_file": runs["safe_project"].name})
    (target / "run.json").write_text(json.dumps(document))
    return target, runs


def test_a_replay_window_goes_through_the_runner(run_copy):
    run_dir, _ = run_copy
    marker = nearest_checkpoint(run_dir, 1_500_000)
    calls = []
    result = replay_window(run_dir, 1_000_000, 1_200_000, None, lambda p=None, **f: calls.append(p), runner=FAKE)
    assert calls
    assert result["window"] == [1_000_000, 1_200_000] and result["window_s"] == pytest.approx([1.0, 1.2])
    assert result["checkpoint"] == (marker.relative_to(run_dir).as_posix() if marker is not None else None)
    part = result["parts"][0]
    assert part["replay"] and (run_dir / part["file"]).is_file()
    index = {e["path"] for e in run_document(run_dir)["outputs"]}
    assert part["file"] in index
    with pytest.raises(Exception, match="end must come after"):
        replay_window(run_dir, 5, 5, None, lambda p=None, **f: None, runner=FAKE)


def test_two_runs_compare(run_copy, tmp_path):
    _, runs = run_copy
    document = compare_study(runs["safe"], runs["unsafe"], tmp_path / "compare")
    assert document["format"] == "cnctwin-compare" and document["verdict"]["changed"] is True
    assert (tmp_path / "compare" / "compare.json").is_file() and (tmp_path / "compare" / "compare.html").is_file()
