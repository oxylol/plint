"""(i) Cost: process CPU seconds per simulated second of the hot loop (run_macro alone, the HotLoop step with
its kernel_inputs and absorb, and the whole driver step with the L2 macro steps of motion, cutting, removal and
dynamics) for the full slot and for the pocket-islands example's roughing (helical entry of its D10 3FL at
12000 rpm, full depth), both rigid, against the architecture's 0.45 s/s budget (architecture 12).

The budget is not met (docs/models/hotloop.md "Open problems": cutting's cut_step is ~94 % of the loop); the
budget assertion is an expected failure that turns into an error once it passes (strict). The regression
bounds hold the measured cost within 2 x."""

import time

import numpy as np
import pytest

from cnctwin.core.drive import OperationDriver
from tests.integration import hotfix as HF

BUDGET_S_PER_S = 0.45


def _measure(drv, max_steps=5000, in_cut_s=None):
    """Totals over the steps entirely in cut (and over the run): CPU of run_macro, of the hot step, of the driver
    step; simulated time; L0 steps."""
    tot = {"macro": 0.0, "hot": 0.0, "step": 0.0, "build": 0.0, "absorb": 0.0, "l2": 0.0, "l0_steps": 0.0,
           "sim_s": 0.0}
    run = {"step": 0.0, "sim_s": 0.0}
    n = 0
    t0 = time.process_time()
    cut_time = 0.0
    while not drv.feeder.finished() and n < max_steps:
        rec = drv.step()
        n += 1
        dt = (rec.tick1 - rec.tick0) * drv.ctx.tick_s
        run["step"] += rec.cpu["step"]
        run["sim_s"] += dt
        if rec.in_cut_s >= 0.999 * dt:
            for k in ("macro", "hot", "step", "build", "absorb", "l2", "l0_steps"):
                tot[k] += rec.cpu[k]
            tot["sim_s"] += dt
            cut_time += dt
            if in_cut_s is not None and cut_time >= in_cut_s:
                break
    run["wall_cpu"] = time.process_time() - t0
    return tot, run


def _report(name, tot, run):
    s = tot["sim_s"]
    out = {"macro_s_per_s": tot["macro"] / s, "hot_s_per_s": tot["hot"] / s, "step_s_per_s": tot["step"] / s,
           "us_per_l0": 1e6 * tot["macro"] / max(tot["l0_steps"], 1.0), "in_cut_sim_s": s,
           "run_s_per_s": run["step"] / run["sim_s"], "build_share": tot["build"] / tot["hot"],
           "absorb_share": tot["absorb"] / tot["hot"], "l2_share": tot["l2"] / tot["step"]}
    print(name, {k: round(v, 4) for k, v in out.items()})
    return out


@pytest.fixture(scope="module")
def slot_cost(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("perf_slot")
    probe = OperationDriver(HF.slot_project(tmp / "probe", 3.0), use_dynamics=False)
    probe.cutting.update_tool(probe.ctx, probe.state, 0, probe.state.bus_prev, force=True)
    ap_mm = HF.station_ap_mm(probe.state.cutting.tools[0].tool, 14)
    drv = OperationDriver(HF.slot_project(tmp / "run", ap_mm, x1_mm=14.0), use_dynamics=False)
    return _report("slot D10 4FL rigid", *_measure(drv))


@pytest.fixture(scope="module")
def pocket_cost(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("perf_pocket")
    drv = OperationDriver(HF.pocket_project(tmp / "run", n_arcs=0, helix_turns=2), use_dynamics=False)
    return _report("pocket-islands helical entry rigid", *_measure(drv, in_cut_s=0.3))


def test_slot_cost_regression(slot_cost):
    assert slot_cost["in_cut_sim_s"] >= 0.3
    assert slot_cost["us_per_l0"] < 110.0                # 53 us measured (one core of the i5-4690, tsafe nice 10)
    assert slot_cost["macro_s_per_s"] < 14.0
    # the wrapper (kernel_inputs, absorb) and the L2 domains stay a small share of a cutting step
    assert slot_cost["hot_s_per_s"] < 1.25 * slot_cost["macro_s_per_s"]


def test_pocket_cost_regression(pocket_cost):
    assert pocket_cost["in_cut_sim_s"] >= 0.29
    assert pocket_cost["us_per_l0"] < 250.0


@pytest.mark.xfail(strict=True, reason="0.45 s/s budget not met: cutting's cut_step dominates (docs/models/hotloop.md)")
def test_budget(slot_cost, pocket_cost):
    assert slot_cost["step_s_per_s"] <= BUDGET_S_PER_S
    assert pocket_cost["step_s_per_s"] <= BUDGET_S_PER_S
