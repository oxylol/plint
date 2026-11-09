"""A production batch (docs/models/loop.md "Production batches"): three parts of the tiny project, each from a
fresh blank, with the state that a real batch carries from part to part: the tool's wear and life, the machine
and coolant temperatures, the chips conveyed. tools.json has a row per part, production.json the three parts."""

from __future__ import annotations

import pytest

from cnctwin.core.loop import run_project
from tests.integration import loopfix as LF

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
    """The same edge cuts every part: each part's own cutting time (wear's per-part rows: the part's share), the
    same for the same program on the same blank, adds up to the tool's; its wear and used life never go back.
    (The tool comes in used: 720 s of earlier cutting, VB 131 um; three 0.13 s side steps in Al add little.)"""
    run_dir, _ = batch
    tool = LF.read(run_dir, "tools.json")["tools"][0]
    rows = sorted(tool["per_part"], key=lambda r: r["part"])
    assert [r["part"] for r in rows] == list(range(PARTS))
    cut = [r["cut_time_s"] for r in rows]
    assert all(c > 0.0 for c in cut) and max(cut) <= 1.05 * min(cut)
    assert sum(cut) == pytest.approx(tool["cut_time_s"], rel=1e-9)        # the tool's time carries over the parts
    vb = [r["VB_max_m"] for r in rows]
    assert all(b >= a for a, b in zip(vb, vb[1:]))
    used = [r["life_used_fraction"] for r in rows]
    assert all(b >= a for a, b in zip(used, used[1:]))
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
