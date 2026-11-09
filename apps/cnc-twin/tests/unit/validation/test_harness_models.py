"""Fast checks of the second-round harness pieces against closed forms: straight-cut engagement, force
statistics, the cantilever compliance, the complex-residue FRF, the kinematic roughness profile, the name
parsers, and the one-sided prediction bound of the metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cnctwin.validation import budgets as B
from cnctwin.validation import metrics as M
from cnctwin.validation import registry as R
from cnctwin.validation.harnesses import cutforce as CF
from cnctwin.validation.harnesses import cutsim as CS
from cnctwin.validation.harnesses import roughness as RH
from cnctwin.validation.harnesses import stability as ST
from cnctwin.validation.harnesses import wallerror as WE


def test_straight_cut_strips_reproduce_the_immersion_angles():
    from cnctwin.dynamics.sld import immersion_angles
    R_m = 5e-3
    for ae, direction in ((1e-3, "down"), (2.5e-3, "up"), (10e-3, "slot")):
        lo, hi = CS.strip(direction, ae, R_m)
        arc = CS.strip_arc(R_m, lo, hi)
        if direction == "slot":
            assert arc == pytest.approx((0.0, math.pi))
        else:
            assert arc == pytest.approx(immersion_angles(ae / (2 * R_m), direction == "down"), abs=1e-12)
    # a station inside the cutter radius sees the strip only where its circle crosses it
    assert CS.strip_arc(2e-3, -5e-3, -4e-3) is None


def test_milling_statistics_follow_their_definitions():
    t = np.linspace(0.0, 1.0, 400, endpoint=False)
    F = np.stack([np.sin(2 * math.pi * 4 * t), 2.0 + np.cos(2 * math.pi * 4 * t), -np.ones_like(t)], axis=1)
    run = CS.MillRun(t, F, np.full(len(t), -0.5), -2 * math.pi, 4, t[1] - t[0], 1)
    cond = CF.MillCond(60.0, 1e-4, 1e-3, 1e-3, "down")
    assert CF.statistic(run, CF.MillRequest(cond, "mean", "y")) == pytest.approx(2.0)
    assert CF.statistic(run, CF.MillRequest(cond, "mean", "y", on_workpiece=True, magnitude=False)) == \
        pytest.approx(-2.0)
    assert CF.statistic(run, CF.MillRequest(cond, "max", "y", magnitude=False)) == pytest.approx(3.0, abs=1e-3)
    rms = math.sqrt((0.5 + 4.5 + 1.0) / 3.0)
    assert CF.statistic(run, CF.MillRequest(cond, "rms3")) == pytest.approx(rms, rel=1e-3)
    assert CF.statistic(run, CF.MillRequest(cond, "torque")) == pytest.approx(0.5)
    assert len(run.per_tooth_peaks(F[:, 1])) == 4


def test_cantilever_compliance_is_the_beam_formula_on_the_clamp_spring():
    L, EI, k = 0.05, 210e9 * 1e-9, 1e7
    c = WE.cantilever(L, EI, k, np.array([0.0, 0.02]), np.array([0.0, 0.02]))
    assert c[0, 0] == pytest.approx(L ** 3 / (3 * EI) + 1 / k)          # tip load, tip deflection
    a = L - 0.02
    assert c[1, 1] == pytest.approx(a ** 3 / (3 * EI) + 1 / k)
    assert c[0, 1] == pytest.approx(c[1, 0])                             # Maxwell reciprocity


def test_residue_frf_of_a_real_mode_equals_the_modal_frf():
    from cnctwin.dynamics.sld import ModalFRF
    f, z, m = 600.0, 0.03, 0.4
    wn = 2 * math.pi * f
    wd = wn * math.sqrt(1 - z * z)
    frf = ST.ResidueFRF([(f, z, 0.0, -1.0 / (2.0 * m * wd), 0)])
    ref = ModalFRF.single(f, z, m, (1.0, 0.0, 0.0))
    w = np.array([0.3 * wn, wn, 2.0 * wn])
    assert np.allclose(frf(w)[:, 0, 0], ref(w)[:, 0, 0], rtol=1e-12)


def test_kinematic_turning_profile_gives_the_nose_radius_roughness():
    from cnctwin.surface.roughness import IsoRules
    rules = IsoRules()
    r, f = 0.8e-3, 0.2e-3
    tpl = RH.corner_template(r, math.radians(93.0), math.radians(27.0), 1e-3, rules.spacing_for(f))
    rough, kin = RH.profile_roughness(tpl, f, 0.0, 0.0, rules)
    assert kin.Rt_m == pytest.approx(r - math.sqrt(r * r - f * f / 4.0), rel=0.02)
    assert rough.Ra_m == pytest.approx(f * f / (18.0 * math.sqrt(3.0) * r), rel=0.05)


def _rough_case(tool: dict, conditions: dict | None = None):
    from cnctwin.validation.cases import Case
    return Case(id="t", quantity="roughness", file="", raw={"setup": {"tool": tool, "conditions": conditions or {}}})


def test_roughness_edge_default_follows_the_tooling_rule():
    """Without a stated edge radius: uncoated solid tools take tooling's ground (sharp) edge, coated ones the light
    hone, inserts the insert hone (tooling.common.edge_prep); a stated radius wins."""
    from cnctwin.db.models import params
    from cnctwin.validation.harnesses.common import Collector
    p = params("tooling")
    col = Collector()
    assert RH._edge_radius(_rough_case({"type": "ball_endmill", "tool_class": "cbn"}), col, "solid") == \
        pytest.approx(float(p["sharp_edge_radius_m"]))
    assert RH._edge_radius(_rough_case({"type": "square_endmill", "coating": "none"}), col, "solid") == \
        pytest.approx(float(p["sharp_edge_radius_m"]))
    assert RH._edge_radius(_rough_case({"type": "square_endmill", "coating": "TiAlN"}), col, "solid") == \
        pytest.approx(float(p["hone_edge_radius_m"]))
    assert RH._edge_radius(_rough_case({"type": "square_endmill", "coating": "none", "edge_radius_um": 12}), col,
                           "solid") == pytest.approx(12e-6)
    assert RH._edge_radius(_rough_case({"type": "turning_insert", "coating": "none"}), col, "insert") == \
        pytest.approx(float(p["insert_edge_radius_m"]))
    from cnctwin.validation.harnesses import common as H     # the force, wall and coolant harnesses share the rule
    assert H.edge_radius_m(_rough_case({"type": "square_endmill", "coating": "none"}), None, col, "solid") == \
        pytest.approx(float(p["sharp_edge_radius_m"]))


def test_roughness_reads_the_instruments_stated_cutoff():
    from cnctwin.validation.harnesses.common import Collector
    col = Collector()
    for text, cutoff in (("Talysurf CCI-Lite, Gaussian filter 0.8 mm cut-off", 0.8e-3),
                         ("Surtronic 3+ portable stylus, sampling length 0.8 mm", 0.8e-3),
                         ("stylus, cut-off of 2.5 mm", 2.5e-3), ("Mitutoyo 201 roughness tester, ISO 4287", None)):
        got = RH._stated_cutoff(_rough_case({}, {"roughness_instrument": text}), col)
        assert got == (None if cutoff is None else pytest.approx(cutoff)), text


def test_ball_profile_is_bounded_by_the_window_and_carries_the_minimum_chip_term():
    """The ball's meridian up to the depth of cut and a wider one give the same feed-mark profile (the surface
    model's window bounds each pass; the former 3 fz + 20 um template was narrower than that window and changed
    it); below sqrt(2 R h_min) the profile carries R_t = h_min, and the scallops across the step-over carry
    Brammertz's term like a turned profile at the feed ae."""
    from cnctwin.surface.roughness import IsoRules
    rules = IsoRules()
    R, fz, h_min = 1e-3, 0.0356e-3, 5e-6
    du = rules.spacing_for(fz)
    deep = RH.profile_roughness(RH.ball_template(R, 0.1e-3, du), fz, h_min, 0.0, rules)
    deeper = RH.profile_roughness(RH.ball_template(R, 0.3e-3, du), fz, h_min, 0.0, rules)
    narrow = RH.profile_roughness(RH.circle_template(R, 3.0 * fz + 2e-5, du), fz, h_min, 0.0, rules)
    assert deep[1].Rt_m == pytest.approx(deeper[1].Rt_m, rel=0.02)
    assert narrow[1].Rt_m < 0.85 * deep[1].Rt_m
    assert deep[1].Rt_m == pytest.approx(h_min, rel=0.1)
    ae, h_small = 0.1e-3, 0.75e-6
    cross = RH.profile_roughness(RH.ball_template(R, 0.1e-3, rules.spacing_for(ae)), ae, h_small, 0.0, rules)
    brammertz = ae * ae / (8.0 * R) + 0.5 * h_small * (1.0 + R * h_small / (ae * ae))
    assert cross[1].Rt_m == pytest.approx(brammertz, rel=0.03)


def test_reference_name_and_dimension_parsers():
    assert CF._token_value("Fc f=0.05", ("f", "h")) == pytest.approx(0.05)
    assert CF._token_value("Fc dry vc361", ("vc",)) == pytest.approx(361.0)
    assert CF.coolant_mode(type("C", (), {"section": lambda self, n: {"coolant": "dry"}})(), "Ft flood f0.1") == "flood"
    assert ST._label("light_chatter_test") == "light_chatter"


def test_a_prediction_bound_makes_the_reference_one_sided():
    budget = B.Budget("q", "m", 25.0)
    stable = M.compare_one(where="value", index=0, point=None, name="stable_test", reference=1.0, unit="mm",
                           pred=R.Prediction(1.4, bound="lower"), budget=budget)
    assert stable.error_rel == 0.0 and stable.within_tolerance and stable.bound == M.LOWER
    chatter = M.compare_one(where="value", index=0, point=None, name="chatter_test", reference=1.0, unit="mm",
                            pred=R.Prediction(1.4, bound="upper"), budget=budget)
    assert chatter.error_rel == pytest.approx(0.4) and chatter.within_tolerance is False
    with pytest.raises(ValueError):
        R.Prediction(1.0, bound="sideways")
