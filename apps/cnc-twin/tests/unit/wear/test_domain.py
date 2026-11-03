"""WearDomain with a fake bus and state: the published record, notch at the depth-of-cut line, the skin
factor, BUE growth and protection, coating first then substrate, thermal cracks raising the chipping
probability, fallbacks when thermal is missing, settings, channels and the tools.json document."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.unit.wear import wearfix as fx
from cnctwin.core.bus import ToolWear
from cnctwin.core.errors import InputError
from cnctwin.db.simap import SIMap
from cnctwin.wear import abi as A
from cnctwin.wear.channels import NAMES, channels
from cnctwin.wear.domain import WearDomain
from cnctwin.wear.protocols import ProjectTolerances, StockShell
from cnctwin.wear.schema import PHYSICS_SCHEMA, settings


def _setup(material=None, tool=None, macro_ipo=10, edge=None, physics=None, stock=None, catalog=None):
    proj = fx.project(material=material, tool=tool, physics=physics, stock=stock)
    ctx = fx.context(proj, macro_ipo=macro_ipo)
    state = fx.sim_state()
    edge = edge if edge is not None else fx.edge_model()
    dom = WearDomain(tools=fx.tool_info(edge), response=fx.AnalyticResponse(), catalog=catalog or fx.Catalog())
    dom.init(ctx, state)
    return dom, ctx, state, edge


def test_the_record_has_every_field_with_the_edge_shape():
    dom, ctx, state, edge = _setup()
    record = fx.run(dom, ctx, state, edge, 3)
    assert set(record.tools) == {0} and record.expired == [] and record.broken == []
    tool = record.tools[0]
    assert isinstance(tool, ToolWear)
    shape = np.shape(edge.elem_z_m)
    for name in ("VB_m", "KT_m", "notch_m", "edge_radius_m", "radial_wear_m", "axial_wear_m", "coating_remaining_m",
                 "chipped", "bue_height_m", "rake_coating_remaining_m", "chip_depth_m", "thermal_damage",
                 "comb_crack_depth_m"):
        assert getattr(tool, name).shape == shape, name
    assert tool.chipped.dtype == bool and tool.section_fatigue_damage.shape == (2,)
    assert np.all(tool.VB_m > 0.0) and np.all(tool.radial_wear_m > 0.0) and np.allclose(tool.axial_wear_m, 0.0)
    assert tool.VB_max_m == pytest.approx(float(np.max(tool.VB_m)))
    assert np.all(tool.edge_radius_m > 8e-6)            # the hone radius rounds as the edge recedes
    assert tool.in_cut_s == pytest.approx(3 * 0.25 * ctx.macro_s) and tool.uses == 1 and tool.parts == 1
    # a new record object each step: the previous record is never mutated (double buffering)
    first = tool.VB_m.copy()
    fx.run(dom, ctx, state, edge, 1, start=3)
    np.testing.assert_array_equal(tool.VB_m, first)


def test_notch_grows_at_the_depth_of_cut_line_of_a_work_hardening_material():
    dom, ctx, state, edge = _setup()
    engaged = np.zeros(np.shape(edge.elem_z_m), bool)
    engaged[:, :6] = True                              # ap ends between elements 5 and 6
    record = fx.run(dom, ctx, state, edge, 4, engaged=engaged)
    tool = record.tools[0]
    assert np.all(tool.notch_m[:, 5] > tool.VB_m[:, 5])
    assert np.all(tool.notch_m[:, :5] == 0.0) and np.all(tool.notch_m[:, 6:] == 0.0)
    excess = (1.48 ** 2 - 1.0)                         # (1 + 48 % hardness increase)^q - 1, q = 2
    sw = state.wear.slots[0]
    assert sw.elem.vn_notch[0, 5] == pytest.approx(excess * sw.elem.vn[0, 5], rel=1e-9)
    assert np.all(tool.radial_wear_m[:, 5] > tool.radial_wear_m[:, 4])
    assert np.all(tool.VB_m[:, 6:] == 0.0)             # never in cut


def test_the_skin_layer_multiplies_the_wear_rate():
    material = dict(fx.MATERIAL, skin={"type": "mill_scale", "thickness_mm": {"value": 0.05},
                                       "hardness_factor": {"value": 2.0}, "abrasiveness_factor": {"value": 2.0}})
    pose = np.eye(4)
    pose[:3, 3] = [-4.9e-3, 40e-3, 26e-3]              # the tool grazes the x = 0 wall of the block
    shell = StockShell("block", [0.1, 0.08, 0.03], 5e-5, 64)
    edge = fx.edge_model()
    frac = shell.skin_fraction(0, edge, True, pose, np.ones(np.shape(edge.elem_z_m)))
    assert 0.0 < frac[0, 0] <= 1.0
    results = []
    for mat in (fx.MATERIAL, material):
        dom, ctx, state, edge = _setup(material=mat, edge=edge)
        frame = fx.path_frame()
        frame.pose_S_T = pose
        fx.step(dom, ctx, state, fx.bus(edge, [fx.cutting(edge)], fx.heat(edge), frame))
        results.append(state.wear.slots[0].elem.vn.copy())
    np.testing.assert_allclose(results[1], results[0] * (1.0 + frac), rtol=1e-12)


def test_bue_grows_in_its_window_and_protects_the_edge():
    no_bue = dict(fx.MATERIAL, chip={"tendency": "continuous", "bue": None})
    heat = None
    out = []
    for mat in (fx.MATERIAL, no_bue):
        dom, ctx, state, edge = _setup(material=mat)
        heat = fx.heat(edge, flank_K=650.0, rake_K=600.0, spread=0.0, weights=(1.0,))
        fx.run(dom, ctx, state, edge, 20, heat_rec=heat, speed=0.5)
        out.append(state.wear.slots[0])
    with_bue, without = out
    h = 1e-4
    # heights relax toward 0.8 h; recent break-offs leave 20 % of the height (mechanics' BUE model)
    assert np.all(with_bue.elem.bue_m > 0.1 * h) and float(np.mean(with_bue.elem.bue_m)) > 0.3 * h
    assert np.all(without.elem.bue_m == 0.0) and with_bue.counters["bue_breakoffs"] > 0
    attrition = with_bue.counters["bue_breakoffs"] * float(with_bue.law.scal[A.LS_PLUCK_M])
    continuous = float(np.sum(with_bue.elem.vn)) - attrition
    assert continuous < float(np.sum(without.elem.vn))


def test_coating_first_then_substrate():
    tool = dict(fx.TOOL, coating={"id": "test-tialn", "thickness_um": 0.3})
    dom_c, ctx_c, state_c, edge = _setup(tool=tool, macro_ipo=100)
    dom_u, ctx_u, state_u, _ = _setup(macro_ipo=100, edge=edge)
    heat = fx.heat(edge, flank_K=1050.0, rake_K=1050.0, spread=0.0, weights=(1.0,))
    sw_c, sw_u = state_c.wear.slots[0], state_u.wear.slots[0]
    assert np.all(sw_c.elem.coat_f == 3e-7)
    fx.run(dom_c, ctx_c, state_c, edge, 1, heat_rec=heat, duty=1.0, entries=0)
    fx.run(dom_u, ctx_u, state_u, edge, 1, heat_rec=heat, duty=1.0, entries=0)
    first_c, first_u = float(sw_c.elem.vn[0, 0]), float(sw_u.elem.vn[0, 0])
    assert first_c < 0.75 * first_u                     # the coating halves the Usui part, its hardness the rest
    for i in range(1, 400):
        fx.run(dom_c, ctx_c, state_c, edge, 1, start=i, heat_rec=heat, duty=1.0, entries=0)
        if np.all(sw_c.elem.coat_f == 0.0):
            break
    assert np.all(sw_c.elem.coat_f == 0.0)
    before = sw_c.elem.vn.copy()
    fx.run(dom_c, ctx_c, state_c, edge, 1, start=500, heat_rec=heat, duty=1.0, entries=0)
    np.testing.assert_allclose(sw_c.elem.vn - before, first_u, rtol=1e-9)    # substrate rate once worn through
    assert any(e.event == "wear:flank_coating_worn_through" for e in ctx_c.trace._memory)


def test_thermal_cycles_crack_the_edge_and_raise_the_chipping_probability():
    dom, ctx, state, edge = _setup(macro_ipo=10)
    heat = fx.heat(edge, cycle_dK=600.0)
    sw = state.wear.slots[0]
    for i in range(400):
        fx.run(dom, ctx, state, edge, 1, start=i, heat_rec=heat, entries=20)
        if sw.flags["comb_cracks"]:
            break
    assert sw.flags["comb_cracks"] and np.all(sw.elem.crack_m > 0.0)
    assert any(e.event == "wear:comb_cracks" for e in ctx.trace._memory)
    # same entry shock on the cracked tool and on a fresh one
    rows = np.array([[0.0, 0, 3, 1e-4, 0.1, 700.0, 1e-4, 1]])
    fresh_dom, fresh_ctx, fresh_state, _ = _setup(edge=edge)
    from cnctwin.wear import step as S
    probe = fx.cutting(edge, shocks=rows)
    g_cracked = S.gather(fx.bus(edge, [probe]), 0, np.shape(edge.elem_z_m), 2)
    inp = A.new_input(*np.shape(edge.elem_z_m), 1)
    inp.in_cut_s[:] = 1.0
    p_cracked = S.chip_step(sw, inp, g_cracked, 7)[0, A.CO_P]
    p_fresh = S.chip_step(fresh_state.wear.slots[0], inp, g_cracked, 7)[0, A.CO_P]
    assert p_cracked > 2.0 * p_fresh


def test_missing_thermal_record_uses_the_response_table_and_says_so():
    dom, ctx, state, edge = _setup()
    fx.step(dom, ctx, state, fx.bus(edge, [fx.cutting(edge)], None, fx.path_frame()))
    sw = state.wear.slots[0]
    assert sw.flags["temperature_fallback"] and np.all(sw.elem.vn > 0.0)
    assert any(e.event == "wear:temperature_from_response_table" for e in ctx.trace._memory)
    with pytest.raises(Exception):
        bad = fx.heat(fx.edge_model(nt=2, ne=3))
        fx.step(dom, ctx, state, fx.bus(edge, [fx.cutting(edge)], bad, fx.path_frame(tick0=10000)))


def test_contact_stress_fallback_when_cutting_gives_no_wear_integrals():
    dom, ctx, state, edge = _setup()
    contact = fx.cutting(edge, sigma_f=0.0, sigma_r=0.0)
    fx.step(dom, ctx, state, fx.bus(edge, [contact], fx.heat(edge), fx.path_frame()))
    sw = state.wear.slots[0]
    assert sw.flags["stress_fallback"] and np.all(sw.elem.vn > 0.0)


def test_settings_are_declared_and_checked():
    assert settings(None).strength == "median"
    s = settings({"breakage_risk_warn": 0.05, "strength": "sample", "rate_window_s": 30})
    assert s.breakage_risk_warn == 0.05 and s.strength == "sample" and s.rate_window_s == 30.0
    with pytest.raises(InputError):
        settings({"calib_nodes": 1})
    with pytest.raises(InputError):
        settings({"strength": "mean"})
    from cnctwin.db.schema import Ctx
    ctx = Ctx(None, obj="test")
    PHYSICS_SCHEMA.check({"strength": "sample", "calib_nodes": 8}, (), None, ctx)
    assert not [p for p in ctx.problems if p.is_error]


def test_sampled_strengths_are_deterministic_per_seed():
    dom_a, ctx_a, state_a, _ = _setup(physics={"strength": "sample"})
    dom_b, ctx_b, state_b, _ = _setup(physics={"strength": "sample"})
    a = state_a.wear.slots[0].sections["strength_b_Pa"]
    b = state_b.wear.slots[0].sections["strength_b_Pa"]
    np.testing.assert_array_equal(a, b)
    median = _setup()[2].wear.slots[0].sections["strength_b_Pa"]
    assert not np.array_equal(a, median)


def test_channels_and_values():
    names = [c.name for c in channels()]
    assert names == list(NAMES) and all(c.domain == "wear" and c.name.startswith("wear.") for c in channels())
    dom, ctx, state, edge = _setup()
    fx.run(dom, ctx, state, edge, 2)
    values = dom.macro_values(state)
    assert set(values) == set(NAMES)
    assert values["wear.VB_max_m"] > 0.0 and values["wear.VB_tooth_m"].shape == (16,)
    assert values["wear.life_used_slot"][0] == pytest.approx(state.wear.slots[0].life["used_fraction"])


def test_tolerances_from_the_project():
    tol = ProjectTolerances(SIMap({"cad": {"tolerances": [
        {"id": "d", "kind": "distance", "nominal_mm": 10, "plus_mm": 0.02, "minus_mm": 0.02},
        {"id": "h", "kind": "diameter", "diameter_mm": {"nominal": 8, "plus": 0.015, "minus": 0.0}},
        {"id": "f", "kind": "flatness", "feature": {"face": "plane"}, "tolerance_mm": 0.01},
        {"id": "fit", "kind": "diameter", "diameter_mm": {"nominal": 8, "fit": "H7"}},
        {"id": "r", "kind": "roughness", "feature": {"face": "plane"}, "Ra_um": 0.8}]}}))
    assert tol.surface_tolerance_m(0) == pytest.approx(0.015e-3 / 4)
    assert tol.finish_spec(0) == (pytest.approx(0.8e-6), None)
    assert tol.unused == ["fit"] and tol.applies(0, True) and not tol.applies(0, False)
    tol.assign(0, 1e-5, 1.6e-6)
    assert tol.surface_tolerance_m(0) == 1e-5 and tol.applies(0, False)


def test_finalize_writes_the_tools_document():
    dom, ctx, state, edge = _setup()
    fx.run(dom, ctx, state, edge, 3)
    doc = dom.finalize(ctx, state)
    tool = doc["tools"][0]
    for key in ("number", "slot", "tool_id", "VB_m", "VB_max_m", "VB_limit_m", "life_used_fraction", "cut_time_s",
                "broken", "remaining_life_s", "per_part", "wear_offsets_m", "law", "criteria"):
        assert key in tool, key
    assert tool["law"]["flank"]["source"] == "taylor" and len(tool["VB_m"]) == 4
    from cnctwin.production.rundir import tool_states
    parsed = tool_states(doc, 0, 1)
    assert parsed[0].number == 1 and parsed[0].VB_max_m == pytest.approx(tool["VB_max_m"])
    import json
    json.dumps(doc)                                     # JSON-safe (no inf, no NumPy types)


def test_a_slot_without_cutting_edges_keeps_counters_only():
    edge = fx.edge_model()
    edge = edge._replace(elem_valid=np.zeros(np.shape(edge.elem_valid), np.int8))
    dom, ctx, state, edge = _setup(edge=edge)
    sw = state.wear.slots[0]
    assert sw.info["law"]["flank"]["source"] == "abrasive_only"
    assert any("no cutting-edge elements" in e for e in state.wear.estimated)
    record = fx.run(dom, ctx, state, edge, 2)
    assert np.all(record.tools[0].VB_m == 0.0) and record.tools[0].life_used_fraction == 0.0
