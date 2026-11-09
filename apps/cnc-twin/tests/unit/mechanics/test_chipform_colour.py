"""Chip form (curl, breaking, ISO 3685 class, segmentation, BUE, ejection) and chip colour."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cnctwin.core.bus import CHIP_FORMS
from cnctwin.mechanics import chipform as CH, colour as COL, oxley
from cnctwin.mechanics.materials import ChipBreaker

from conftest import STEEL_1045, work_from


@pytest.fixture(scope="module")
def steel_state(steel, carbide):
    res = oxley.solve(steel, 2e-4, 2.5, math.radians(6), tool=carbide, tool_temp_K=600.0)
    return CH.ChipState.from_orthogonal(res)


def test_chip_geometry_and_ejection(steel, carbide, steel_state):
    cond = CH.ChipConditions(h_m=2e-4, width_m=2e-3, speed_m_s=2.5, rake_rad=math.radians(6))
    cf = CH.chip_form(steel, carbide, cond, steel_state)
    assert cf.thickness_m == pytest.approx(2e-4 / steel_state.chip_ratio)
    assert cf.width_m == pytest.approx(2e-3)
    v = cf.velocity_rel_tool_m_s
    assert np.linalg.norm(v) == pytest.approx(2.5 * steel_state.chip_ratio)
    assert v[1] > 0 and v[0] < 0 and v[2] == pytest.approx(0.0, abs=1e-12)    # up a positive rake face
    assert cf.kinetic_energy_J_per_kg == pytest.approx(0.5 * float(cf.velocity_rel_work_m_s @ cf.velocity_rel_work_m_s))
    assert cf.form_code in CHIP_FORMS and cf.form_name == CHIP_FORMS[cf.form_code]


def test_curl_radii():
    assert CH.natural_curl_radius(4e-4) == pytest.approx(15.0 * 4e-4)
    breaker = ChipBreaker(land_m=1.5e-4, groove_width_m=1e-3, groove_depth_m=2e-4, rake_rad=0.26,
                          feed_window_m=(1e-4, 4e-4), depth_window_m=None)
    in_groove = CH.breaker_curl_radius(breaker, 3e-4, 1.0)
    assert in_groove == pytest.approx((1e-6 + 4 * 4e-8) / (8 * 2e-4))
    before = CH.breaker_curl_radius(breaker, 1e-4, 1.0)
    run = 1.5e-4 + 1e-3 - 1e-4
    assert before == pytest.approx((run ** 2 + 4e-8) / (4e-4))
    assert CH.breaker_curl_radius(breaker, 3e-4, 1e-4) == 1e-4          # natural curl tighter


def test_breaker_window(steel, coated_carbide, steel_state):
    def form(feed):
        cond = CH.ChipConditions(h_m=feed, width_m=2e-3, speed_m_s=2.5, rake_rad=math.radians(6), feed_m=feed,
                                 depth_m=2e-3, nose_radius_m=0.0)
        return CH.chip_form(steel, coated_carbide, cond, steel_state)
    inside = form(2e-4)
    below = form(5e-5)
    assert inside.breaker_state == "in_window" and inside.curl_mode == "breaker"
    assert inside.broken and inside.form_code in (42, 52, 62, 22, 12, 32, 31)
    assert below.breaker_state == "below_window" and below.curl_mode == "natural"
    assert below.curl_radius_m > inside.curl_radius_m


def test_nakayama_breaking_criterion(steel):
    # carbon steel: the chip keeps Nakayama's measured fraction of the bulk true fracture strain
    eps_b = CH.chip_fracture_strain(steel)
    assert eps_b == pytest.approx(0.055 * 0.6)
    # Nakayama 1962 Fig. 13, S45C: a 0.30 mm chip breaks on the R_c = 4.1 mm flow circle, t2 / (2 R_c)
    assert 0.30e-3 / (2.0 * 4.1e-3) == pytest.approx(eps_b, rel=0.15)
    free_cutting = work_from(STEEL_1045, chip__breakability="good")
    assert CH.chip_fracture_strain(free_cutting) == pytest.approx(0.5 * eps_b)
    # the 'poor' rating does not change a steel class Nakayama measured (carbon content made no difference)
    assert CH.chip_fracture_strain(work_from(STEEL_1045, chip__breakability="poor")) == pytest.approx(eps_b)
    # austenitic stainless: its own measured fraction (less ductile chips despite the more ductile bulk)
    stainless = work_from(STEEL_1045, family="stainless", physical__fracture_strain=1.1)
    assert CH.chip_fracture_strain(stainless) == pytest.approx(0.020 * 1.1)
    # families Nakayama did not test: carbon steel's fraction, scaled by the database rating
    titanium = work_from(STEEL_1045, family="titanium", physical__fracture_strain=0.49, chip__breakability="poor")
    assert CH.chip_fracture_strain(titanium) == pytest.approx(0.055 * 0.49 * 2.0)
    # a measured chip breaking strain in the material file is used as it is
    assert CH.chip_fracture_strain(work_from(STEEL_1045, chip__breaking_strain=0.02)) == pytest.approx(0.02)


def test_classification_table():
    # straight unbroken ribbon, snarled after many turns
    assert CH.classify(math.inf, 2e-4, 2e-3, math.inf, False, 0.0, False, "turning") == 13
    assert CH.classify(math.inf, 2e-4, 2e-3, 3.0, True, 0.0, False, "turning") == 12
    # arcs
    assert CH.classify(2e-3, 2e-4, 2e-3, 0.5, True, 0.0, False, "milling") == 62
    assert CH.classify(2e-3, 2e-4, 2e-3, 0.5, False, 0.0, False, "turning") == 61
    # planar coils: flat spiral, conical spiral
    assert CH.classify(2e-3, 2e-4, 2e-3, 2.0, True, 0.0, False, "milling") == 31
    assert CH.classify(2e-3, 2e-4, 2e-3, 2.0, True, 0.0, True, "turning") == 32
    # helices: washer-type (pitch <= width) and tubular, long/short/snarled
    assert CH.classify(2e-3, 2e-4, 2e-3, 3.0, True, math.radians(8), False, "turning") == 42
    assert CH.classify(2e-3, 2e-4, 2e-3, 20.0, False, math.radians(8), False, "turning") == 41
    assert CH.classify(2e-3, 2e-4, 2e-3, math.inf, False, math.radians(8), False, "turning") == 43
    assert CH.classify(6e-3, 2e-4, 1e-3, 3.0, True, math.radians(30), False, "turning") == 22
    assert CH.classify(6e-3, 2e-4, 1e-3, 30.0, False, math.radians(30), False, "turning") == 21
    assert CH.classify(2e-3, 2e-4, 2e-3, 30.0, False, math.radians(10), False, "drilling") == 51


def test_segmentation_rules(steel, steel_state):
    cond = CH.ChipConditions(h_m=2e-4, width_m=2e-3, speed_m_s=2.5, rake_rad=0.1)
    assert CH.is_segmented(steel, steel_state, cond)[0] is False          # below the 300 m/min onset
    fast = CH.ChipConditions(h_m=2e-4, width_m=2e-3, speed_m_s=6.0, rake_rad=0.1)
    assert CH.is_segmented(steel, steel_state, fast)[0] is True
    recht = work_from(STEEL_1045, chip__segmentation={"criterion": "thermal_instability"})
    seg, ratio = CH.is_segmented(recht, steel_state, cond)
    assert math.isfinite(ratio) and ratio > 0.0 and seg == (ratio < 1.0)
    # Recht's ratio falls with speed (less time for heat to leave the band)
    slow_state = CH.ChipState(**{**steel_state.__dict__, "shear_strain_rate_1_s": steel_state.shear_strain_rate_1_s / 20})
    assert CH.recht_ratio(recht, slow_state, 2e-4, 0.1) > ratio
    continuous = work_from(STEEL_1045, chip__segmentation=None)
    assert CH.is_segmented(continuous, steel_state, fast)[0] is False


def test_bue_window_and_dynamics(steel):
    assert CH.bue_in_window(steel, 0.5, 600.0)
    assert not CH.bue_in_window(steel, 0.5, 1000.0) and not CH.bue_in_window(steel, 3.0, 600.0)
    h = np.zeros(4)
    counters = np.zeros(4, dtype=np.int64)
    keys = np.arange(4, dtype=np.int64)
    window = np.array([1.0, 1.0, 0.0, 1.0])
    uncut = np.full(4, 1e-4)
    breaks = np.zeros(4, dtype=np.int64)
    for _ in range(500):
        breaks += CH.bue_advance(h, window, uncut, 5e-4, 11, keys, counters)
    assert h[2] == 0.0 and np.all(h[[0, 1, 3]] > 0.0) and breaks[[0, 1, 3]].sum() > 0
    h2 = np.zeros(4)
    c2 = np.zeros(4, dtype=np.int64)
    for _ in range(500):
        CH.bue_advance(h2, window, uncut, 5e-4, 11, keys, c2)
    assert np.array_equal(h, h2) and np.array_equal(counters, c2)


def test_polymer_composite_and_discontinuous(polymer, composite, carbide):
    hot = CH.ChipState(0.5, 0.6, 3e-4, 560.0, 600.0, 500.0, 2.0, 1e4)
    cool = CH.ChipState(0.5, 0.6, 3e-4, 400.0, 420.0, 380.0, 2.0, 1e4)
    cond = CH.ChipConditions(h_m=2e-4, width_m=2e-3, speed_m_s=5.0, rake_rad=0.2)
    assert CH.chip_form(polymer, carbide, cond, hot).form_code == 95
    assert CH.chip_form(polymer, carbide, cond, cool).form_code != 95
    dust = CH.chip_form(composite, carbide, cond, cool)
    assert dust.form_code == 90 and dust.dust_fraction == 1.0
    brass = work_from(STEEL_1045, chip__tendency="discontinuous", chip__segmentation=None)
    assert CH.chip_form(brass, carbide, cond, cool).form_code == 70


def test_milling_chips_are_pieces(steel, carbide, steel_state):
    cond = CH.ChipConditions(h_m=1e-4, width_m=5e-3, speed_m_s=2.5, rake_rad=0.1, incl_rad=math.radians(30),
                             cut_length_m=4e-3, process="milling")
    cf = CH.chip_form(steel, carbide, cond, steel_state)
    assert cf.broken and cf.length_m == pytest.approx(4e-3 * steel_state.chip_ratio, rel=1e-9)
    assert cf.form_code in (61, 62, 31, 42, 22)
    assert cf.chip_flow_angle_rad == pytest.approx(math.radians(30))
    assert cf.velocity_rel_tool_m_s[2] < 0.0


# ------------------------------------------------------------------------------ colour
def test_colour_round_trip(steel):
    q = COL.activation_temperature(steel)
    t_ref = 1800.0
    for row in steel.oxide_table:
        rgb, name, teq = COL.chip_colour(steel, COL.exposure_hold(row.temp_K, t_ref, q))
        assert teq == pytest.approx(row.temp_K, rel=1e-12)
        assert rgb == row.rgb
    rgb, name, teq = COL.chip_colour(steel, COL.exposure_hold(400.0, 1.0, q))
    assert name == "bare" and rgb == COL.bare_colour(steel)


def test_short_exposure_needs_higher_temperature(steel):
    q = COL.activation_temperature(steel)
    long_hold = COL.equivalent_temperature(COL.exposure_hold(573.15, 1800.0, q), q)
    flash = COL.equivalent_temperature(COL.exposure_hold(573.15, 1e-3, q), q)
    assert flash < long_hold
    # a 1 ms flash shows the 300 degC (30 min) colour only near the temperature that compensates
    t_needed = q / (q / 573.15 - math.log(1800.0 / 1e-3))
    assert COL.equivalent_temperature(COL.exposure_hold(t_needed, 1e-3, q), q) == pytest.approx(573.15, rel=1e-9)


def test_exposure_integrals(steel):
    q = COL.activation_temperature(steel)
    temps = np.linspace(900.0, 400.0, 2001)
    expect = np.trapezoid(np.exp(-q / temps), dx=1e-4)
    assert COL.exposure_from_history(temps, 1e-4, q) == pytest.approx(expect, rel=1e-12)
    # exponential cooling: Laplace estimate tau R T0^2 / (Q (T0 - Ta)) e^{-Q/T0} for Q >> T0
    tau = 0.2
    val = COL.exposure_cooling(900.0, 300.0, tau, q)
    laplace = tau * 900.0 ** 2 / (q * 600.0) * math.exp(-q / 900.0)
    assert val == pytest.approx(laplace, rel=0.1)
    rgb, name, exposure = COL.emitted_chip_colour(steel, 900.0, 1e-4, 4e-4, 50.0, 293.15)
    assert exposure > 0.0 and len(rgb) == 3


def test_bare_colours_for_other_families(polymer, composite):
    assert COL.chip_colour(polymer, 1e3)[1] == "bare"
    assert COL.bare_colour(composite) == (0x2E, 0x2E, 0x2E)
