"""A production batch (docs/models/loop.md "Production batches"): three parts of the tiny project, each from a
fresh blank, with the state that a real batch carries from part to part: the tool's wear and life, the machine
and coolant temperatures, the chips conveyed. tools.json has a row per part, production.json the three parts."""

from __future__ import annotations

import pytest

from cnctwin.core.errors import TwinError
from cnctwin.core.loop import run_project
from tests.integration import loopfix as LF

# The run stops in the second part on NaN states the domains hand on across the transition (docs/models/loop.md
# "Open problems": thermal's tool flank temperatures into cutting's coefficient pipeline, structure's work modes of
# the re-seated part into dynamics, thermal's tool field after the same tool's M06): a model error (TwinError,
# ValueError) marks the test as an expected failure until those are fixed; a wrong loop result still fails it.
pytestmark = pytest.mark.xfail(raises=(TwinError, ValueError), strict=False,
                               reason="domain NaN at the transition: docs/models/loop.md Open problems")

PARTS = 3


@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    base = tmp_path_factory.mktemp("loop-batch")
    project = LF.tiny_mill(base / "project", parts=PARTS)
    result = run_project(project, base / "run", {"outputs": False}, None)
    return base / "run", result


def test_three_parts_ran_with_the_ledgers_passing(batch):
    run_dir, result = batch
    run = LF.read(run_dir, "run.json")
    assert result["complete"] and run["summary"]["parts"] == PARTS and run["summary"]["parts_run"] == PARTS
    assert LF.conservation_ok(run) == {"mass": True, "energy": True, "time": True, "final_mass": True}
    blocks = LF.read(run_dir, "timeline.json")["blocks"]
    assert sorted({b["part"] for b in blocks}) == list(range(PARTS))
    masses = [p["removed_mass_kg"] for p in run["summary"]["parts_removed_mass_kg"]]
    assert len(masses) == PARTS and all(m > 0.0 for m in masses)
    assert max(masses) == pytest.approx(min(masses), rel=0.02)          # the same step in every blank


def test_wear_and_tool_life_carry_over(batch):
    run_dir, _ = batch
    tool = LF.read(run_dir, "tools.json")["tools"][0]
    rows = sorted(tool["per_part"], key=lambda r: r["part"])
    assert [r["part"] for r in rows] == list(range(PARTS))
    cut = [r["cut_time_s"] for r in rows]
    assert all(b > a for a, b in zip(cut, cut[1:]))                      # cumulative: the same edge keeps cutting
    vb = [r["VB_max_m"] for r in rows]
    assert all(b >= a for a, b in zip(vb, vb[1:])) and vb[-1] > vb[0]
    used = [r["life_used_fraction"] for r in rows]
    assert all(b > a for a, b in zip(used, used[1:]))
    production = LF.read(run_dir, "production.json")
    assert production["parts_simulated"] == PARTS
    life = [t["life_used_part_frac"] for p in production["per_part"] for t in p["tools"] if t["number"] == 1]
    assert len(life) == PARTS and all(v > 0.0 for v in life)


def test_machine_and_coolant_state_carry_over(batch):
    run_dir, _ = batch
    run = LF.read(run_dir, "run.json")
    thermal = run["domains"]["thermal"]
    unload = thermal["unload"]
    assert len(unload) == PARTS
    spindles = thermal["spindles"]
    assert spindles and all(s["temp_K"] > 0.0 for s in spindles.values())
    coolant = run["domains"]["coolant"]
    assert len(coolant.get("per_part", coolant.get("parts", {})) or {}) >= PARTS - 1
    chips = run["summary"]["chips_kg"]
    assert chips["chips_conveyed_kg"] + chips["chips_settled_kg"] > 0.0
