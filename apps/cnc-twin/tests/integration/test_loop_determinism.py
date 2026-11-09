"""Determinism of the whole loop (architecture 9; docs/models/loop.md "Checkpoints and resume"): two runs of a
project give the same result set, byte for byte; a run stopped after a checkpoint and resumed from it gives the
same result set as one that never stopped; a replay of a window reproduces the fast rows the run recorded."""

from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from cnctwin.core.loop import replay_window, run_project
from cnctwin.core.rundir import result_set_digest
from tests.integration import loopfix as LF

CHECKPOINT_S = 1.0          # the tiny project runs about 7 s: checkpoints at 1, 2, ... s


class Stop(BaseException):
    """What the API worker's cancel raises from the progress callback."""


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("loop-det")
    project = LF.tiny_mill(base / "project")
    options = {"outputs": False, "checkpoint_interval_s": CHECKPOINT_S}
    digests = []
    for k in range(2):
        run_project(project, base / f"run{k}", options, None)
        digests.append(result_set_digest(base / f"run{k}"))
    return base, project, options, digests


def test_two_runs_give_the_same_result_set(runs):
    base, _, _, digests = runs
    assert digests[0] == digests[1]
    for name in ("findings.json", "trace.jsonl", "frames.bin", "tools.json", "production.json", "surface.json",
                 "inspection.json"):
        assert name in digests[0], name
    assert any(k.startswith("series/") for k in digests[0]) and any(k.startswith("keyframes/") for k in digests[0])


def test_a_resumed_run_gives_the_same_result_set(runs):
    base, project, options, digests = runs
    run_dir = base / "resumed"
    seen = []

    def stop_after_the_second_checkpoint(progress=None, **fields):
        fields = dict(progress or {}, **fields)
        if fields.get("checkpoint"):
            seen.append(fields["checkpoint"])
            if len(seen) == 2:
                raise Stop()

    with pytest.raises(Stop):
        run_project(project, run_dir, options, stop_after_the_second_checkpoint)
    assert not (run_dir / "run.json").exists()
    marker = run_dir / seen[-1]["file"]
    assert marker.is_file() and int(seen[-1]["tick"]) > 0
    result = run_project(project, run_dir, dict(options, resume_from=str(marker)), None)
    assert result["complete"]
    resumed = result_set_digest(run_dir)
    different = sorted(k for k in set(resumed) | set(digests[0]) if resumed.get(k) != digests[0].get(k))
    assert not different, different


def test_a_replay_reproduces_the_recorded_fast_rows(runs):
    base, _, _, _ = runs
    run_dir = base / "run0"
    index = [json.loads(line) for line in (run_dir / "series" / "index.jsonl").read_text().splitlines()]
    fast = [e for e in index if e["tier"] == "fast" and not e.get("replay")]
    assert fast, "the run recorded no fast window"
    table = pq.read_table(run_dir / fast[0]["file"])
    ticks = table.column("tick").to_numpy()
    tick0 = int(ticks[0])
    tick1 = int(ticks[min(len(ticks) - 1, 200)]) + 1
    result = replay_window(run_dir, tick0, tick1, {"outputs": False}, None)
    assert result["parts"] and result["window"] == [tick0, tick1]
    replayed = pq.read_table(run_dir / result["parts"][0]["file"])
    keep = (ticks >= tick0) & (ticks < tick1)
    assert replayed.num_rows == int(np.count_nonzero(keep)) > 0
    assert set(replayed.column_names) == set(table.column_names)
    for column in table.column_names:
        a = table.column(column).to_numpy(zero_copy_only=False)[keep]
        b = replayed.column(column).to_numpy(zero_copy_only=False)
        assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), column
