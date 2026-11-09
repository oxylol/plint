"""G31 probing through the whole loop (docs/models/hotloop.md 9a, docs/models/loop.md 15): the tiny project's
block with the pocket example's spindle probe (T20, 6 mm ruby ball, length 125 mm to the bottom of the ball). A
G31 skip move from 10 mm above the block's top towards 5 mm below it must stop at the top: the program checks the
latched Z (#5063, work coordinates) and raises a macro alarm (#3000) when the probe did not touch or touched
elsewhere, so a run that ends without an alarm proves the trigger, and the frames show the gauge line stopping
125 mm above the top. Before the probe contacts existed the move ran to its end point (#5063 = -5): alarm."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from cnctwin.core.loop import run_project
from cnctwin.core.warmup import mill_project
from tests.integration.loopfix import read

PROBE_PROGRAM = """%
O09010 (LOOP TEST: G31 PROBE OF THE BLOCK TOP)
G90 G17 G21 G40 G49 G80 G94 G54
G53 G00 Z0.
T20 M06
G90 G54 G00 X15. Y10.
G43 H20 Z10.
G31 Z-5. F300.
#100 = #5063
IF [#100 LT -0.05] THEN #3000 = 1 (PROBE DID NOT TOUCH)
IF [#100 GT 0.05] THEN #3000 = 2 (PROBE TOUCHED ABOVE THE TOP)
G00 Z25.
G53 G49 Z0.
M30
%
"""


def probe_project(tmp: Path) -> Path:
    import cnctwin.core.warmup as W
    project = mill_project(tmp)
    data = yaml.safe_load(project.read_text())
    (Path(project).parent / "warmup.nc").write_text(PROBE_PROGRAM)
    src = yaml.safe_load((Path(W.__file__).resolve().parents[2] / "examples" / "pocket-islands-al6061" /
                          "project.yaml").read_text())
    probes = [t for t in src["tools"]["list"] if t.get("kind") == "probe"]
    assert probes, "the pocket example lists its spindle probe"
    data["tools"]["list"] = data["tools"]["list"] + probes
    project.write_text(yaml.safe_dump(data, sort_keys=False))
    return project


@pytest.fixture(scope="module")
def probe_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("probe")
    project = probe_project(tmp / "project")
    result = run_project(project, tmp / "run", {"outputs": False, "sanity": False}, None)
    return tmp / "run", result


def test_the_g31_move_stops_at_the_block_top(probe_run):
    run_dir, result = probe_run
    run = read(run_dir, "run.json")
    assert result["complete"], run["summary"]
    assert not run["summary"].get("stopped"), run["summary"].get("stopped")
    alarms = [f for f in read(run_dir, "findings.json")["findings"] if f["category"] == "alarm"]
    assert not alarms, alarms
    from cnctwin.core.rundir import read_frames
    records, _ = read_frames(run_dir)
    tip = np.asarray(records["tip_S_m"][:, 0, :], float)
    # the probe has no tool slot: motion's tip is the gauge line, 125 mm above the bottom of the ball. After the touch
    # at the block's top (10 mm in S) the axis brakes along the move: at F300 jerk-limited braking takes about 25 um,
    # plus the tick of travel in which the stylus touched; then the retract goes straight up (before the planner
    # started it from the G31 end point, 5 mm below the touch, the gauge line went down to 132 mm)
    z_min = float(tip[:, 2].min())
    assert 0.010 + 0.125 - 1e-4 < z_min < 0.010 + 0.125
