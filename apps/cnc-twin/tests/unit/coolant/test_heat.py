"""Heat transfer: Martin (1977) values, Rohsenow and Zuber against Incropera Example 10.1, film boiling,
Leidenfrost switching, the pressure shift of saturation, aerosols."""

import math

import numpy as np
import pytest

from cnctwin.coolant import fluid as F
from cnctwin.coolant import heat as H

import coolfix as CF

TOL, ITER = 1e-12, 200


def water_at_saturation():
    """Saturated water at 1 atm, the property values of Incropera Example 10.1 (Table A.6 lists Pr = 1.76; its
    rounded mu, cp and k give 1.73, so k is set to reproduce the tabulated Pr the example uses)."""
    return F.Fluid(id="water-ex10.1", kind="water", concentration_frac=0.0, density=lambda t: 957.9,
                   viscosity=lambda t: 279e-6, specific_heat=lambda t: 4217.0,
                   conductivity=lambda t: 279e-6 * 4217.0 / 1.76,
                   sigma_N_per_m=58.9e-3, boiling_K=373.15, leidenfrost_K=466.15, latent_J_kg=2257e3)


def test_martin_value_hand_calc():
    # Re = 1e4, Pr = 0.7, r/D = 5, H/D = 6: G = 0.2 (1 - 0.22) = 0.156, F = 2 * 100 * sqrt(1 + 0.005 * 10^2.2)
    G = 0.2 * (1 - 1.1 * 0.2)
    Fv = 2.0 * 100.0 * math.sqrt(1.0 + 0.005 * 10.0 ** 2.2)
    hand = 0.7 ** 0.42 * G * Fv
    assert H.martin_nusselt(1e4, 0.7, 5.0, 6.0) == pytest.approx(hand, rel=1e-12)
    assert hand == pytest.approx(35.96, abs=0.01)
    # Re = 5e4, Pr = 7, r/D = 3, H/D = 10
    G = (1 / 3) * (1 - 1.1 / 3) / (1 + 0.1 * 4 / 3)
    Fv = 2.0 * math.sqrt(5e4) * math.sqrt(1.0 + 0.005 * 5e4 ** 0.55)
    assert H.martin_nusselt(5e4, 7.0, 3.0, 10.0) == pytest.approx(7.0 ** 0.42 * G * Fv, rel=1e-12)


def test_martin_range_handling():
    # below r/D = 2.5 the stagnation-zone value at 2.5 is used; H/D is clamped to [2, 12]
    assert H.martin_nusselt(2e4, 1.0, 1.0, 6.0) == pytest.approx(H.martin_nusselt(2e4, 1.0, 2.5, 6.0))
    assert H.martin_nusselt(2e4, 1.0, 4.0, 30.0) == pytest.approx(H.martin_nusselt(2e4, 1.0, 4.0, 12.0))
    air = F.air_props(300.0)
    d, v, r = 0.002, 100.0, 0.02
    # a gas jet 24 D away acts as the momentum-equivalent nozzle D' = H / 12 at H/D' = 12 (same Re)
    far = H.jet_h(True, v, d, 24 * d, r, air)
    d_eq = 24 * d / 12.0
    re = v * d / air.nu_m2_s
    assert far == pytest.approx(H.martin_nusselt(re, air.pr, r / d_eq, 12.0) * air.k_W_mK / d_eq, rel=1e-12)
    # a coherent liquid jet arrives undecayed: H/D beyond the potential core does not matter
    liquid = F.Props(998.0, 1e-3, 0.6, 4180.0, 0.07)
    assert H.jet_h(False, 10.0, 0.004, 0.5, 0.02, liquid) == pytest.approx(
        H.jet_h(False, 10.0, 0.004, 5.8 * 0.004, 0.02, liquid), rel=1e-12)


def test_rohsenow_against_incropera_example_10_1():
    # Incropera Ex. 10.1: water at 1 atm on a surface 18 K above saturation, C_sf = 0.0128 -> 836 kW/m2;
    # the model's C_sf of 0.0130 scales it by (0.0128/0.0130)^3
    fluid = water_at_saturation()
    liq = fluid.props(373.15)
    vap = fluid.vapour(373.15)
    q = H.nucleate_flux(fluid, 18.0, liq, vap)
    csf = H.P()["rohsenow_csf"]
    assert q == pytest.approx(836e3 * (0.0128 / csf) ** 3, rel=5e-3)


def test_zuber_peak_flux_against_incropera_example_10_1():
    # Ex. 10.1: q_max = 1.26 MW/m2 for saturated water (C = 0.149); the vapour table holds 0.586 kg/m3 at 380 K
    fluid = water_at_saturation()
    q = H.peak_flux(fluid, fluid.props(373.15), fluid.vapour(373.15), 0.0, 0.0)
    assert q == pytest.approx(1.26e6, rel=0.015)
    # subcooling (Ivey & Morris) raises it: 80 K subcooled water about 4.7x
    sub = H.peak_flux(fluid, fluid.props(373.15), fluid.vapour(373.15), 80.0, 0.0)
    factor = 1 + 0.1 * (957.9 / fluid.vapour(373.15).rho_kg_m3) ** 0.75 * 4217.0 * 80.0 / 2257e3
    assert sub == pytest.approx(q * factor, rel=1e-12)


def test_film_boiling_formula_and_radiation():
    fluid = water_at_saturation()
    T_w, T_sat = 900.0, 373.15
    liq = fluid.props(T_sat)
    D = 0.01
    h = H.film_h(fluid, T_w, T_sat, liq, H.CYLINDER, D, 0.0, 101325.0, TOL, ITER)
    vap = fluid.vapour(0.5 * (T_w + T_sat))
    dT = T_w - T_sat
    hfg = 2257e3 + 0.8 * vap.cp_J_kgK * dT
    hc = 0.62 * vap.k_W_mK / D * (9.80665 * (957.9 - vap.rho_kg_m3) * hfg * D ** 3 /
                                  (vap.nu_m2_s * vap.k_W_mK * dT)) ** 0.25
    hr = H.P()["film_emissivity"] * 5.670374419e-8 * (T_w ** 4 - T_sat ** 4) / dT
    assert h ** (4 / 3) == pytest.approx(hc ** (4 / 3) + hr * h ** (1 / 3), rel=1e-10)
    # a fast jet thins the film (forced film boiling, Bromley, LeRoy & Robbers 1953)
    fast = H.film_h(fluid, T_w, T_sat, liq, H.CYLINDER, D, 20.0, 101325.0, TOL, ITER)
    assert fast > h


def test_leidenfrost_switching_and_continuity():
    fluid = F.fluid_from_db(CF.fluid_db("emulsion-8pct"))
    T_f, h_fc = 295.0, 2.0e4
    temps = np.linspace(300.0, 800.0, 501)
    results = [H.boiling(fluid, T, T_f, h_fc, H.PLATE, 0.01, 5.0, 0.0, TOL, ITER) for T in temps]
    regimes = np.array([r.regime for r in results])
    # single phase, nucleate, (transition), film - in that order and nothing else
    changes = regimes[np.r_[True, np.diff(regimes) != 0]]
    assert list(changes) in ([0, 1, 2, 3], [0, 1, 3])
    T_L = fluid.leidenfrost_K
    assert all(r.regime == H.FILM for T, r in zip(temps, results) if T >= T_L)
    assert all(r.regime == H.SINGLE_PHASE for T, r in zip(temps, results) if T <= fluid.boiling_K)
    # the coefficient collapses at the Leidenfrost point
    peak = max(r.h_W_m2K for r in results)
    film = next(r.h_W_m2K for T, r in zip(temps, results) if T >= T_L)
    assert film < 0.2 * peak
    # q is continuous where transition meets film boiling
    eps = 1e-6
    below = H.boiling(fluid, T_L - eps, T_f, h_fc, H.PLATE, 0.01, 5.0, 0.0, TOL, ITER)
    above = H.boiling(fluid, T_L + eps, T_f, h_fc, H.PLATE, 0.01, 5.0, 0.0, TOL, ITER)
    assert below.q_W_m2 == pytest.approx(above.q_W_m2, rel=1e-4)
    # below saturation the coefficient is the jet's forced convection
    assert results[0].h_W_m2K == pytest.approx(h_fc)


def test_impingement_pressure_raises_saturation():
    fluid = F.fluid_from_db(CF.fluid_db("emulsion-8pct"))
    # Clausius-Clapeyron from 100 degC: 10 bar abs gives 180 degC (steam tables 179.9 degC)
    assert fluid.saturation_K(10e5) == pytest.approx(453.0, abs=2.0)
    assert fluid.saturation_K(5e5) == pytest.approx(425.0, abs=2.0)
    # a wall at 140 degC boils under a gentle jet but not under a 70 bar impingement
    gentle = H.boiling(fluid, 413.15, 295.0, 1e4, H.PLATE, 0.01, 5.0, 0.0, TOL, ITER)
    strong = H.boiling(fluid, 413.15, 295.0, 1e4, H.PLATE, 0.01, 100.0, 69e5, TOL, ITER)
    assert gentle.regime == H.NUCLEATE and strong.regime == H.SINGLE_PHASE


def test_aerosol_sensible_and_latent_heat():
    fluid = F.fluid_from_db(CF.fluid_db("mql-ester"))
    flux = 0.05                       # kg/(m2 s) deposited oil
    T_f = 293.15
    cool = H.aerosol_h(300.0, flux, 400.0, T_f, fluid)
    cp = fluid.props(T_f).cp_J_kgK
    assert cool == pytest.approx(300.0 + flux * cp * (400.0 - T_f) / (400.0 - T_f), rel=1e-12)
    hot = H.aerosol_h(300.0, flux, 650.0, T_f, fluid)          # above boiling, below Leidenfrost
    assert hot == pytest.approx(300.0 + flux * (cp * (fluid.boiling_K - T_f) + fluid.latent_J_kg) / (650.0 - T_f),
                                rel=1e-12)
    above = H.aerosol_h(300.0, flux, 700.0, T_f, fluid)        # drops bounce off: no evaporation
    assert above == pytest.approx(300.0 + flux * cp * (fluid.boiling_K - T_f) / (700.0 - T_f), rel=1e-12)
