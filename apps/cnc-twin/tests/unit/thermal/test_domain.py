"""ThermalDomain with a fake bus and state: ownership, heat bookkeeping, checkpoint, findings, cool-down."""

import copy
import json
import math

import numpy as np
import pytest

from cnctwin.core.bus import Bus, ThermalFeedback
from cnctwin.core.findings import Category, Severity
from cnctwin.core.state import ToolSlot, ToolTable
from cnctwin.db.simap import SIMap
from cnctwin.thermal.domain import ThermalDomain
from cnctwin.thermal.state import ThermalState

import fixtures as F


def run(steps, proj=None, removal_at=None, power=(300.0, 100.0, 30.0, 5.0), coolant=False, state=None, dom=None,
        ctx=None, start=0, geo=None):
    proj = proj or F.project()
    ctx = ctx or F.context(proj)
    if state is None:
        state = F.sim_state()
        geo = geo or F.stock()
        dom = ThermalDomain(geometry=geo, edges=F.Edges(F.edge_model()))
        dom.init(ctx, state)
    outputs = []
    for n in range(start, start + steps):
        prev = F.bus_step(n, power_W=power, coolant_on=coolant)
        nxt = Bus()
        state.clock.tick = int(prev.motion.tick)
        dom.macro_step(ctx, state, prev, nxt)
        outputs.append(nxt)
        if removal_at is not None and n == removal_at:
            geo.remove_box([0.015, 0.0, 0.017], [0.025, 0.03, 0.02])
    return state, dom, ctx, outputs, geo


def test_writes_only_its_own_record_and_books_every_joule():
    state, dom, ctx, outs, _ = run(30, removal_at=8)
    default = Bus()
    last = outs[-1]
    for name in Bus.WRITERS:
        if name != "thermal":
            assert repr(getattr(last, name)) == repr(getattr(default, name)), name
    assert isinstance(last.thermal, ThermalFeedback)
    generated = 30 * 0.01 * sum((300.0, 100.0, 30.0, 5.0))
    ledger = state.ledger
    assert ledger.heat_to_chip_J + ledger.heat_to_tool_J + ledger.heat_to_work_J == pytest.approx(generated, rel=1e-12)
    th = state.thermal
    work = th.works[0]
    pending = work.pending_energy()
    assert ledger.heat_to_work_J == pytest.approx(work.ledger.sources_J + pending + th.totals["unplaced"], rel=1e-12)
    assert ledger.removed_enthalpy_J > 0.0
    for name, rel in last.thermal.ledger_residual_rel.items():
        assert rel < 1e-12, name
    assert set(last.thermal.heat_to_chip_J) == {0}


def test_tool_heat_and_machine_outputs():
    state, dom, ctx, outs, _ = run(20)
    fb = outs[-1].thermal
    heat = fb.tools[0]
    assert heat.edge_temp_q_K.shape == (4, 10, 3)
    assert heat.q_weights.sum() == pytest.approx(1.0)
    assert heat.edge_cycle_amp_K.shape == (4, 10) and np.all(heat.edge_cycle_amp_K > 0)
    assert np.all(heat.edge_peak_q_K > heat.body_temp_K)
    assert heat.length_growth_m > 0.0
    growth = fb.spindle_growth_M_m["main"]
    assert growth[2] < 0.0 and growth[0] == 0.0             # the nose moves toward the work (-z_T in M)
    assert fb.spindle_growth_abs_M_m["main"][2] < growth[2] or fb.spindle_growth_abs_M_m["main"][2] <= 0.0
    assert set(fb.axis_drift) == {"X", "Y", "Z"} and fb.axis_drift["X"].shape == (6,)
    assert fb.work_temp_K[0].shape == state.thermal.works[0].shape
    assert np.isnan(fb.work_temp_K[0]).sum() == 0          # unmachined block: no empty cells
    assert fb.work_surface_temp_K[0] > fb.work_bulk_mean_K[0]
    assert fb.work_preheat_K[0] >= fb.work_bulk_mean_K[0] - 1e-9
    values = dom.macro_values(state)
    assert values["thermal.heat_to_chip_W"] > 0 and np.isfinite(values["thermal.tool_edge_max_K"])
    names = {c.name for c in dom.channels()}
    assert set(values) <= names


def test_checkpoint_resume_is_bit_identical():
    geo_a = F.stock()
    state, dom, ctx, outs, _ = run(12, geo=geo_a)
    arrays, meta = state.thermal.to_arrays()
    meta = json.loads(json.dumps(meta))
    ledger_snapshot = copy.deepcopy(state.ledger)
    clock = copy.deepcopy(state.clock)
    _, _, _, cont_a, _ = run(8, state=state, dom=dom, ctx=ctx, start=12)
    state_b = F.sim_state()
    state_b.ledger = ledger_snapshot
    state_b.clock = clock
    state_b.thermal = ThermalState.from_arrays({k: np.array(v) for k, v in arrays.items()}, meta)
    dom_b = ThermalDomain(geometry=F.stock(), edges=F.Edges(F.edge_model()))
    _, _, _, cont_b, _ = run(8, state=state_b, dom=dom_b, ctx=ctx, start=12)
    for a, b in zip(cont_a, cont_b):
        ta, tb = a.thermal, b.thermal
        assert np.array_equal(ta.work_temp_K[0], tb.work_temp_K[0], equal_nan=True)
        assert np.array_equal(ta.tools[0].edge_temp_q_K, tb.tools[0].edge_temp_q_K)
        assert ta.spindle_temp_K == tb.spindle_temp_K
        assert ta.contact_surface_temp_K == tb.contact_surface_temp_K


def test_run_is_deterministic():
    _, _, _, a, _ = run(10)
    _, _, _, b, _ = run(10)
    assert np.array_equal(a[-1].thermal.work_temp_K[0], b[-1].thermal.work_temp_K[0], equal_nan=True)
    assert a[-1].thermal.tools[0].edge_peak_q_K.tobytes() == b[-1].thermal.tools[0].edge_peak_q_K.tobytes()


def test_burn_and_white_layer_findings():
    state, dom, ctx, outs, _ = run(6, power=(6000.0, 500.0, 2500.0, 500.0))
    kinds = {(f.category, f.severity) for f in state.findings}
    assert (Category.THERMAL_DAMAGE, Severity.WARNING) in kinds
    finding = next(f for f in state.findings if f.category == Category.THERMAL_DAMAGE)
    assert finding.quantities["surface_temp_K"] > finding.quantities["threshold_K"]
    assert {"x_S_m", "y_S_m", "z_S_m"} <= set(finding.quantities)
    assert finding.tick > 0 and finding.domain == "thermal"
    # one finding per (op, workpiece, kind, block): repeats are counted, not raised again
    count = len(state.findings)
    run(4, state=state, dom=dom, ctx=ctx, start=6, power=(6000.0, 500.0, 2500.0, 500.0))
    assert len(state.findings) == count
    assert state.thermal.damage.counts["burn"] > 1


def test_cool_down_between_operations_and_unload():
    proj = F.project(wait_min=10.0)
    state, dom, ctx, outs, _ = run(30, proj=proj)
    th = state.thermal
    before_mean, before_max = th.works[0].bulk()
    spindle_before = th.spindles["main"].body.T_K
    dom.end_operation(ctx, state)
    unload = th.unload[-1]
    assert unload["mean_K"] == pytest.approx(before_mean, rel=1e-3)
    after_mean, after_max = th.works[0].bulk()
    assert after_max < before_max
    assert th.spindles["main"].body.T_K < spindle_before
    assert th.time_s == pytest.approx(0.3 + 600.0)
    assert th.works[0].clamped and th.op_index == 1
    for name, ledger in dom._solver_ledgers(th):
        assert ledger.worst_rel < 1e-10, name           # contract: 1e-9
    report = dom.finalize(ctx, state)
    json.dumps(report)
    assert report["unload"][0]["op"] == 0


def test_turning_band_spreads_heat_around_the_axis():
    proj = F.project()
    tool_raw = {"id": "turn", "type": "turning_insert", "tool_material": {"class": "carbide"},
                "geometry": {"insert": {"ic_mm": 12.7, "thickness_mm": 4.76}, "holder": {"shank_mm": [25, 25],
                                                                                        "overhang_mm": 30}}}
    proj.tools = {"list": [{"number": 1, "tool": SIMap(tool_raw), "holder": None, "stickout_mm": 30}]}
    ctx = F.context(proj)
    state = F.sim_state()
    geo = F.SampledStockCylinder(radius=0.02, length=0.03)
    dom = ThermalDomain(geometry=geo, edges=None)
    dom.init(ctx, state)
    for n in range(5):
        prev = F.bus_step(n, point=(0.02, 0.0, 0.015))
        prev.removal.contacts[0].geometry.ap_m = 0.002
        nxt = Bus()
        dom.macro_step(ctx, state, prev, nxt)
    work = state.thermal.works[0]
    rise = np.nan_to_num(work.grid.T - 293.15)
    ring = rise[:, :, 7]
    hot = np.argwhere(ring > 0.05 * ring.max())
    angles = np.arctan2(hot[:, 1] - ring.shape[1] / 2 + 0.5, hot[:, 0] - ring.shape[0] / 2 + 0.5)
    assert np.ptp(angles) > math.pi / 2                  # heat spread over a large arc of the band
    assert state.thermal.tools[0].layout.turning
    assert nxt.thermal.tools[0].body_temp_K > 293.15
