"""Tool life: the order of the end-of-life criteria (VB, dimensional drift, finish), the remaining-life
prediction, carry-over across operations and parts, checkpoint resume, and empty steps (zero rates, no NaN)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.unit.wear import wearfix as fx
from cnctwin.core.bus import Bus
from cnctwin.core.findings import Category
from cnctwin.wear.domain import WearDomain
from cnctwin.wear.state import WearState

HOT = dict(flank_K=1350.0, rake_K=1350.0)


def _setup(tolerances=None, macro_ipo=100, edge=None, **proj):
    project = fx.project(tolerances=tolerances, **proj)
    ctx = fx.context(project, macro_ipo=macro_ipo)
    state = fx.sim_state()
    edge = edge if edge is not None else fx.edge_model()
    dom = WearDomain(tools=fx.tool_info(edge), response=fx.AnalyticResponse(), catalog=fx.Catalog())
    dom.init(ctx, state)
    return dom, ctx, state, edge


def _until_expired(dom, ctx, state, edge, max_steps=2000, **cut):
    heat = fx.heat(edge, **HOT)
    for i in range(max_steps):
        record = fx.run(dom, ctx, state, edge, 1, start=i, heat_rec=heat, duty=1.0, entries=0, **cut)
        if record.tools[0].expired:
            return record, i
    raise AssertionError("the tool never expired")


def test_flank_wear_ends_the_life_without_tolerances():
    dom, ctx, state, edge = _setup()
    record, i = _until_expired(dom, ctx, state, edge)
    sw = state.wear.slots[0]
    assert record.tools[0].life_cause == "vb" and record.expired == [1]
    assert sw.life["values"]["vb"] >= 3e-4 > sw.life["values"]["vb"] - 3e-4 * 0.05
    findings = [f for f in state.findings if f.category == Category.TOOL_END_OF_LIFE]
    assert len(findings) == 1 and "VB_B" in findings[0].message
    assert sw.life["expired_tick"] <= state.clock.tick
    assert 0.99 <= record.tools[0].life_used_fraction < 1.1


def test_a_tight_tolerance_ends_the_life_by_dimensional_drift_first():
    tol = [{"id": "wall", "kind": "distance", "nominal_mm": 10, "plus_mm": 0.005, "minus_mm": 0.005}]
    dom, ctx, state, edge = _setup(tolerances=tol)
    record, i = _until_expired(dom, ctx, state, edge, surface=fx.surface_rows(edge))
    sw = state.wear.slots[0]
    assert sw.life["cause"] == "dimension"
    assert sw.life["values"]["dimension"] >= 5e-6 and sw.life["values"]["vb"] < 3e-4


def test_a_finish_specification_ends_the_life_by_roughness_first():
    tol = [{"id": "floor", "kind": "roughness", "feature": {"face": "plane"}, "Ra_um": 1.6}]
    dom, ctx, state, edge = _setup(tolerances=tol)
    record, i = _until_expired(dom, ctx, state, edge, surface=fx.surface_rows(edge))
    sw = state.wear.slots[0]
    assert sw.life["cause"] == "finish"
    assert sw.life["Ra_m"] >= 1.6e-6 and sw.life["values"]["vb"] < 3e-4
    # the new edge met the specification at this feed: the criterion was reached by wear
    assert sw.life["limits"]["finish"] == 1.0


def test_the_remaining_life_prediction_matches_the_actual_life():
    dom, ctx, state, edge = _setup()
    heat = fx.heat(edge, **HOT)
    fx.run(dom, ctx, state, edge, 60, heat_rec=heat, duty=1.0, entries=0)
    sw = state.wear.slots[0]
    predicted = sw.life["remaining_s"]
    cut_before = sw.counters["in_cut_s"]
    assert sw.life["remaining_cause"] == "vb" and math.isfinite(predicted)
    assert sw.life["replacement_s"] == pytest.approx(state.wear.time_s + predicted, rel=1e-6)   # duty 1
    record, i = _until_expired(dom, ctx, state, edge)
    actual = sw.counters["in_cut_s"] - cut_before
    assert actual == pytest.approx(predicted, rel=0.03, abs=ctx.macro_s)


def test_wear_and_counters_carry_over_operations_and_parts():
    dom, ctx, state, edge = _setup(macro_ipo=10)
    fx.run(dom, ctx, state, edge, 5)
    sw = state.wear.slots[0]
    vn = sw.elem.vn.copy()
    assert sw.counters["uses"] == 1 and sw.counters["parts"] == 1
    dom.end_operation(ctx, state)
    away = fx.path_frame(tick0=state.clock.tick, slot=-1)                                     # the tool leaves
    record = fx.step(dom, ctx, state, fx.bus(edge, [], None, away, tick=state.clock.tick + 10000))
    np.testing.assert_array_equal(sw.elem.vn, vn)
    state.clock.part_index = 1
    fx.run(dom, ctx, state, edge, 5, start=10)
    assert np.all(sw.elem.vn[sw.geo.valid != 0] > vn[sw.geo.valid != 0])     # continues from the worn state
    assert sw.counters["uses"] == 2 and sw.counters["parts"] == 2
    doc = dom.finalize(ctx, state)["tools"][0]
    assert [row["part"] for row in doc["per_part"]] == [0, 1]
    assert doc["per_part"][1]["cut_time_s"] == pytest.approx(5 * 0.25 * ctx.macro_s)
    assert doc["per_part"][0]["life_used_fraction"] <= doc["per_part"][1]["life_used_fraction"]
    assert doc["cut_time_s"] == pytest.approx(10 * 0.25 * ctx.macro_s)


def test_checkpoint_resume_continues_bit_identically():
    dom, ctx, state, edge = _setup(macro_ipo=10)
    shocks = np.array([[1000.0 * k, k % 4, 3, 1e-4, 0.1, 900.0, 1e-4, 1] for k in range(8)])
    heat = fx.heat(edge, cycle_dK=250.0)
    fx.run(dom, ctx, state, edge, 4, heat_rec=heat, shocks=shocks)
    arrays, meta = state.wear.to_arrays()
    other = fx.sim_state()
    other.clock.tick = state.clock.tick
    other.wear = WearState.from_arrays({k: np.array(v) for k, v in arrays.items()}, meta)
    dom2 = WearDomain(tools=fx.tool_info(edge), response=fx.AnalyticResponse(), catalog=fx.Catalog())
    a = fx.run(dom, ctx, state, edge, 6, start=4, heat_rec=heat, shocks=shocks)
    b = fx.run(dom2, ctx, other, edge, 6, start=4, heat_rec=heat, shocks=shocks)
    for name in ("VB_m", "KT_m", "notch_m", "edge_radius_m", "radial_wear_m", "coating_remaining_m", "chip_depth_m",
                 "thermal_damage", "comb_crack_depth_m", "bue_height_m"):
        np.testing.assert_array_equal(getattr(a.tools[0], name), getattr(b.tools[0], name), err_msg=name)
    for name in state.wear.slots[0].elem._fields:
        np.testing.assert_array_equal(getattr(state.wear.slots[0].elem, name), getattr(other.wear.slots[0].elem, name))
    assert a.tools[0].remaining_life_s == b.tools[0].remaining_life_s


def test_a_checkpoint_without_the_edge_fatigue_sum_resumes_with_an_empty_one():
    """Checkpoints written before the edge's fatigue sum existed (ElemState.edge_fat, 2026-11-09) still load: the sum
    starts empty, every other array as stored."""
    dom, ctx, state, edge = _setup(macro_ipo=10)
    fx.run(dom, ctx, state, edge, 2)
    arrays, meta = state.wear.to_arrays()
    old = {k: np.array(v) for k, v in arrays.items() if not k.endswith("elem.edge_fat")}
    back = WearState.from_arrays(old, meta)
    sw, sw0 = back.slots[0], state.wear.slots[0]
    assert np.all(sw.elem.edge_fat == 0.0) and sw.elem.edge_fat.shape == sw0.elem.vn.shape
    np.testing.assert_array_equal(sw.elem.vn, sw0.elem.vn)


def test_empty_steps_give_zero_rates_and_no_nan():
    dom, ctx, state, edge = _setup(macro_ipo=10)
    for i in range(3):
        record = fx.step(dom, ctx, state, fx.bus(edge, [], None, fx.path_frame(tick0=i * 10000), tick=(i + 1) * 10000))
    tool = record.tools[0]
    for name in ("VB_m", "KT_m", "notch_m", "radial_wear_m", "axial_wear_m", "coating_remaining_m", "bue_height_m"):
        arr = getattr(tool, name)
        assert arr.shape == np.shape(edge.elem_z_m) and np.all(np.isfinite(arr)) and np.all(arr == 0.0), name
    assert np.all(np.isfinite(tool.edge_radius_m)) and tool.life_used_fraction == 0.0
    assert math.isinf(tool.remaining_life_s) and not tool.broken and record.expired == []
    values = state.wear.values
    assert values["wear.VB_max_m"] == 0.0 and values["wear.flank_rate_max_m_s"] == 0.0
    # a contact whose per-element arrays are all zero (air cut) changes nothing either
    zero = fx.cutting(edge, duty=0.0)
    record = fx.step(dom, ctx, state, fx.bus(edge, [zero], fx.heat(edge), fx.path_frame(tick0=40000), tick=50000))
    assert np.all(record.tools[0].VB_m == 0.0) and np.all(np.isfinite(record.tools[0].thermal_damage))
    assert all(np.all(np.isfinite(v)) for k, v in state.wear.values.items() if k != "wear.coating_min_m"
               and k != "wear.remaining_life_s" and not k.endswith("tooth_m") and k != "wear.life_used_slot")
