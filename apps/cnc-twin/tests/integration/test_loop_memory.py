"""Memory of the whole loop per macro step (docs/convergence.md 9 item 10, docs/models/loop.md 15).

With 2 ms macro steps the convergence study's run exceeded 1.8 GB about 26 s in, before the first cut: at the tool
change the tip moved to the new tool's point between two ticks, motion booked that jump (102 mm in 1 ms) as the
tick's feed velocity, and chips sized its frozen occupancy grid by the travel at that velocity over the lag (8e7
voxels). The 10 ms run survived only because the jump did not fall on a macro step's last tick, the sample chips
reads. With one tick per macro step (macro_ipo 1) every tool change falls on a last tick: the tiny project runs
through its tool change and memory per macro step must stay flat.
"""

import math
import os
from pathlib import Path

import numpy as np
import yaml

from cnctwin.chips.scene import tool_speed_max
from cnctwin.core.loop import Run
from tests.integration.loopfix import tiny_mill

PAGE = os.sysconf("SC_PAGE_SIZE")
WARM_STEPS = 200                     # domain set-up allocations and the first jobs settle in these
STEP_JUMP_MB = 64.0                  # no macro step may add more (the bug added 270 MB here, 1.9 GB at 2 ms; the
                                     # tool change's own allocations, the new tool's tables, are about 40 MB)
SETTLE_STEPS = 50                    # after the tool change the new tool's first jobs run in these
FLAT_STEPS = 450                     # the steps after that: memory per macro step flat over them
FLAT_GROWTH_MB = 16.0
FLAT_SLOPE_MB = 0.02                 # per macro step (20 kB: 9 MB over the window at most)


def _rss_mb() -> float:
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * PAGE / 2 ** 20


def test_memory_per_macro_step_stays_flat_through_a_tool_change(tmp_path):
    project = tiny_mill(tmp_path / "p")
    data = yaml.safe_load(project.read_text())
    data["physics"]["macro_ipo"] = 1
    project.write_text(yaml.safe_dump(data, sort_keys=False))
    rows = []

    class Observed(Run):
        def step(self):
            super().step()
            bus = self.state.bus_prev
            feed = max((float(np.max(np.linalg.norm(np.asarray(p.feed_vel_M_m_s, float), axis=1)))
                        for p in bus.motion.paths if len(p.feed_vel_M_m_s)), default=0.0)
            tool_event = any(e.get("kind") == "tool" for p in bus.motion.paths for e in p.events)
            rows.append((_rss_mb(), feed, int(np.asarray(self.state.chips.occ_mass_kg).size), tool_event))

    run = Observed(project, tmp_path / "run", {"outputs": False, "sanity": False, "max_sim_s": 2.2})
    run.execute()
    rss = np.array([r[0] for r in rows])
    feed = np.array([r[1] for r in rows])
    voxels = np.array([r[2] for r in rows])
    changes = [k for k, r in enumerate(rows) if r[3]]
    assert changes, "the run must pass the tool change"
    flat = rss[changes[-1] + SETTLE_STEPS:]
    assert len(flat) >= FLAT_STEPS
    flat = flat[:FLAT_STEPS]
    # no tick's feed velocity beyond what the machine can do (the tool change's jump is no motion)
    v_max = tool_speed_max(run.project.machine)
    assert math.isfinite(v_max) and float(feed.max()) <= v_max * (1 + 1e-9)
    # the frozen occupancy grid stays the size of what a tool can sweep (a few thousand 2 mm voxels here)
    assert int(voxels.max()) <= 200_000
    # no macro step adds STEP_JUMP_MB, and memory per macro step is flat once the tool change has settled
    assert float(np.diff(rss[WARM_STEPS:]).max()) <= STEP_JUMP_MB
    assert float(flat.max() - flat[0]) <= FLAT_GROWTH_MB
    slope = float(np.polyfit(np.arange(len(flat), dtype=float), flat, 1)[0])
    assert slope <= FLAT_SLOPE_MB
