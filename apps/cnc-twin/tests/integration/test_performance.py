"""(i) Cost: process CPU seconds per simulated second (docs/performance.md, docs/models/hotloop.md section 13).

Measured: run_macro alone, the HotLoop step with its kernel_inputs and absorb, and the whole driver step with the L2
macro steps of motion, cutting, removal and dynamics, for

* the full slot (D10 4FL, 10000 rpm, rigid), the steps entirely in cut;
* the pocket-islands example's roughing (helical entry of its D10 3FL at 12000 rpm, full depth, rigid), 0.3 s in cut;
* the slot with the machine's dynamics modes (the dynamics kernels in every L0 step), 0.2 s in cut;
* the whole pocket-islands helical-entry run (two turns) with the default physics settings (no fast-L2 overrides:
  cutting's coefficient pipeline every macro step, dynamics' default grids), rigid.

Against the architecture's budget (section 20: hot loop <= 0.45 s/s) and the whole-run target (<= 1 CPU s per
simulated s including L2). Both are missed: removal's tooth pass (engaged intervals and the envelope subtraction,
once per tooth per revolution) alone costs about 33 us per L0 step (4-7 s/s), more than the rest of the loop
together (docs/performance.md "Where the time goes"); the assertions are expected failures that turn into errors
once they pass (strict). The regression bounds hold the measured costs within about 2 x (one core of the
i5-4690, shared box, bin/tsafe nice 10: timings vary by +-30 % with the load)."""

import time

import pytest

from cnctwin.core.drive import OperationDriver
from tests.integration import hotfix as HF

BUDGET_S_PER_S = 0.45            # architecture 20: hot loop per simulated second
WHOLE_RUN_TARGET_S_PER_S = 1.0   # whole pocket helical-entry run including L2 (the performance task's target)


def _measure(drv, max_steps=5000, in_cut_s=None):
    """Totals over the steps entirely in cut (and over the run): CPU of run_macro, of the hot step, of the driver
    step; simulated time; L0 steps."""
    tot = {"macro": 0.0, "hot": 0.0, "step": 0.0, "build": 0.0, "absorb": 0.0, "l2": 0.0, "l0_steps": 0.0,
           "sim_s": 0.0}
    run = {"step": 0.0, "sim_s": 0.0, "l2": 0.0}
    n = 0
    t0 = time.process_time()
    cut_time = 0.0
    while not drv.feeder.finished() and n < max_steps:
        rec = drv.step()
        n += 1
        dt = (rec.tick1 - rec.tick0) * drv.ctx.tick_s
        run["step"] += rec.cpu["step"]
        run["l2"] += rec.cpu["l2"]
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
           "run_s_per_s": run["step"] / run["sim_s"], "run_l2_s_per_s": run["l2"] / run["sim_s"],
           "run_sim_s": run["sim_s"], "build_share": tot["build"] / tot["hot"],
           "absorb_share": tot["absorb"] / tot["hot"], "l2_share": tot["l2"] / tot["step"]}
    print(name, {k: round(v, 4) for k, v in out.items()})
    return out


def _slot(tmp, use_dynamics):
    probe = OperationDriver(HF.slot_project(tmp / "probe", 3.0), use_dynamics=False)
    probe.cutting.update_tool(probe.ctx, probe.state, 0, probe.state.bus_prev, force=True)
    ap_mm = HF.station_ap_mm(probe.state.cutting.tools[0].tool, 14)
    return OperationDriver(HF.slot_project(tmp / "run", ap_mm, x1_mm=14.0), use_dynamics=use_dynamics)


@pytest.fixture(scope="module")
def slot_cost(tmp_path_factory):
    drv = _slot(tmp_path_factory.mktemp("perf_slot"), False)
    return _report("slot D10 4FL rigid", *_measure(drv))


@pytest.fixture(scope="module")
def slot_dynamics_cost(tmp_path_factory):
    drv = _slot(tmp_path_factory.mktemp("perf_slot_dyn"), True)
    out = _report("slot D10 4FL with the machine's modes", *_measure(drv, in_cut_s=0.2))
    sys = drv.hot.last["contacts"][0].sys
    out["n_modes"] = int(sys.model.Phi.shape[0])
    return out


@pytest.fixture(scope="module")
def pocket_cost(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("perf_pocket")
    drv = OperationDriver(HF.pocket_project(tmp / "run", n_arcs=0, helix_turns=2), use_dynamics=False)
    return _report("pocket-islands helical entry rigid", *_measure(drv, in_cut_s=0.3))


@pytest.fixture(scope="module")
def pocket_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("perf_pocket_run")
    drv = OperationDriver(HF.pocket_project(tmp / "run", n_arcs=0, helix_turns=2, fast_l2=False), use_dynamics=False)
    out = _report("pocket-islands helical entry, whole run, default L2", *_measure(drv))
    out["finished"] = drv.feeder.finished()
    return out


def test_slot_cost_regression(slot_cost):
    assert slot_cost["in_cut_sim_s"] >= 0.3
    assert slot_cost["us_per_l0"] < 90.0                 # 49-59 us measured (2026-11-09, load 3-6; 80 us before)
    assert slot_cost["macro_s_per_s"] < 11.0             # 6.2-7.4 s/s measured (10.0 before)
    # the wrapper (kernel_inputs, absorb) and the L2 domains stay a small share of a cutting step
    assert slot_cost["hot_s_per_s"] < 1.25 * slot_cost["macro_s_per_s"]


def test_slot_dynamics_cost_regression(slot_dynamics_cost, slot_cost):
    assert slot_dynamics_cost["in_cut_sim_s"] >= 0.19
    assert slot_dynamics_cost["n_modes"] >= 4
    # the dynamics kernels (predictor, deflections and corrector over the engaged stations) add little to a step
    assert slot_dynamics_cost["us_per_l0"] < slot_cost["us_per_l0"] + 10.0


def test_pocket_cost_regression(pocket_cost):
    assert pocket_cost["in_cut_sim_s"] >= 0.29
    assert pocket_cost["us_per_l0"] < 80.0               # 39-45 us measured (71 us before)


def test_pocket_run_regression(pocket_run):
    assert pocket_run["finished"]
    assert pocket_run["run_s_per_s"] < 7.5               # 3.9-4.7 s/s measured: whole run (6.5 s, 0.8 s in cut)
    # L2 1.9-2.2 s/s measured, of which removal's connectivity check (find_bodies every few macro steps on the
    # pocket's 0.075 mm field: 67 ms per call) is 1.7-1.9 s/s and cutting's coefficient pipeline 0.16 s/s
    # (docs/performance.md)
    assert pocket_run["run_l2_s_per_s"] < 3.5


@pytest.mark.xfail(strict=True, reason="0.45 s/s hot-loop budget not met: 7.6-9.0 s/s for the slot and 9.7-11.2 "
                                       "s/s for the pocket's helical entry (driver step in cut, 2026-11-09), of which "
                                       "removal's tooth pass is 33-37 us per L0 step = 4-7.5 s/s (docs/performance.md)")
def test_budget(slot_cost, pocket_cost):
    assert slot_cost["step_s_per_s"] <= BUDGET_S_PER_S
    assert pocket_cost["step_s_per_s"] <= BUDGET_S_PER_S


@pytest.mark.xfail(strict=True, reason="whole pocket helical-entry run including L2 at 3.9-4.7 s/s (2026-11-09), "
                                       "above the 1 s/s target: removal's tooth pass in cut and its connectivity check "
                                       "at L2 (docs/performance.md)")
def test_whole_run_target(pocket_run):
    assert pocket_run["run_s_per_s"] <= WHOLE_RUN_TARGET_S_PER_S
