"""``twin check`` (cnctwin.core.check.check_project; docs/models/loop.md "twin check"): the L0 strides of every tool
by architecture 5.1, the lattice rule, the CPU and disk estimates, the controller's and the planner's cycle times
and sanity's findings; ``--write`` stores the strides in physics.resolved and a run uses them."""

from __future__ import annotations

import math

import pytest
import yaml

from cnctwin.core.check import check_project
from cnctwin.core.loop import run_project
from tests.integration import loopfix as LF


@pytest.fixture(scope="module")
def checked(tmp_path_factory):
    base = tmp_path_factory.mktemp("loop-check")
    project = LF.tiny_mill(base / "project")
    request = {"project_path": str(project), "project_dir": str(project.parent), "read_only": False,
               "options": {"write": True}}
    result = check_project(request, base / "out", None)
    return base, project, result


def test_the_check_resolves_the_strides(checked):
    _, _, result = checked
    assert result["ok"] and result["exit_code"] == 0, result["errors"]
    op = result["operations"][0]
    stride = op["strides"][0]
    assert stride["tool"] == 1 and 1000 % stride["l0_ticks"] == 0
    # 10 000 rpm, 0.5 deg per step: dt <= 8.3 us (the D10's 3 teeth at 96 steps per tooth would allow 20.8 us)
    omega = 10000.0 * 2.0 * math.pi / 60.0
    dt = stride["l0_ticks"] * 1e-6
    assert omega * dt <= math.radians(0.5) + 1e-12
    assert stride["rule"] in ("angle", "tooth", "mode")
    assert op["dry_run_cycle_time_s"] > 0.0 and op["planner_cycle_time_s"] >= op["dry_run_cycle_time_s"] - 1e-9
    assert op["estimated_cpu_s_per_s"] > 0.0 and op["fast_all_bytes"] > 0.0
    assert op["lattice"]["ok"] in (True, False) and "summary" in result


def test_write_stores_the_strides_and_a_run_uses_them(checked):
    base, project, result = checked
    assert result["written"]
    data = yaml.safe_load(project.read_text())
    resolved = data["physics"]["resolved"]
    assert resolved["operations"]["OP10"][0]["l0_ticks"] == result["operations"][0]["strides"][0]["l0_ticks"]
    run = run_project(project, base / "run", {"outputs": False, "max_sim_s": 4.5}, None)
    document = LF.read(base / "run", "run.json")
    assert document["clock"]["l0_ticks"] == {"OP10": resolved["operations"]["OP10"][0]["l0_ticks"]}
    assert run["complete"] is False and document["summary"]["truncated"]["at_s"] >= 4.5


def test_a_project_with_errors_is_not_written(tmp_path):
    project = LF.tiny_mill(tmp_path / "project")
    text = project.parent / "warmup.nc"
    text.write_text(text.read_text().replace(" S10000 M03", ""))      # cutting with the spindle stopped
    data = yaml.safe_load(project.read_text())
    result = check_project({"project_path": str(project), "options": {"write": True}}, tmp_path / "out", None)
    assert not result["ok"] and result["exit_code"] == 1 and result["written"] is None
    assert yaml.safe_load(project.read_text()) == data
