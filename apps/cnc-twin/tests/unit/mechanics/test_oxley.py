"""Oxley's theory: code-to-code check against the published extended-Oxley predictions, the
kernel against the independent SciPy solver, the solution conditions, trends, fallbacks."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cnctwin.mechanics import abi, kernels as K, oxley
from cnctwin.mechanics.oxley import kernel_inputs

from conftest import STEEL_1045, work_from

T0 = 273.15


@pytest.fixture(scope="module")
def lalwani_1045():
    """AISI 1045 inputs of the extended-Oxley model (ExtOxley code of Pantale 2022 = Lalwani 2009):
    JC 553.1/600.8/0.234/0.0134/1 with T_ref = 0 degC and T_m = 1460 degC, rho = 8000,
    K = 52.61 - 0.0281 (T - T0), c_p = 420 + 0.504 (T - T0), T_w = 25 degC."""
    temps = np.linspace(0.0, 1500.0, 16)
    return work_from(STEEL_1045,
                     flow_stress__T_ref_degC=0.0, flow_stress__T_melt_degC=1460.0, physical__density_kg_m3=8000.0,
                     physical__specific_heat_J_kgK={"table": [[t, 420.0 + 0.504 * t] for t in temps]},
                     physical__conductivity_W_mK={"table": [[t, 52.61 - 0.0281 * t] for t in temps]})


# Ning & Liang 2019, Table 2, tests 5-8: extended-Oxley predictions of Lalwani et al. 2009 for
# AISI 1045, alpha = -7 deg, T0 = 25 degC: (V m/min, t1 mm) -> (Fc N, T_int degC), width 1.6 mm.
LALWANI = {(200, 0.15): (576, 942), (200, 0.30): (1007, 1042), (300, 0.15): (533, 1017), (300, 0.30): (1041, 1025)}


@pytest.mark.parametrize("case", sorted(LALWANI))
def test_code_to_code_extended_oxley(lalwani_1045, case):
    speed, t1 = case
    fc_doc, tint_doc = LALWANI[case]
    res = oxley.solve(lalwani_1045, t1 * 1e-3, speed / 60.0, math.radians(-7.0), work_temp_K=T0 + 25.0,
                      coulomb_limit=False)
    assert res.converged and res.regime == "sticking"
    fc = res.cutting_force_N_per_m * 1.6e-3
    tint = res.T_interface_K - T0
    # three of the four documented cases agree within 1-2 %; at 300 m/min, 0.3 mm the published
    # solution sits about 10 % higher in force (a different local delta optimum of the
    # published grid search, delta step 0.005)
    tol = 0.12 if case == (300, 0.30) else 0.03
    assert abs(fc / fc_doc - 1.0) < tol, (fc, fc_doc)
    assert abs(tint / tint_doc - 1.0) < tol, (tint, tint_doc)


CONDITIONS = [
    dict(t1_m=1.5e-4, speed_m_s=200 / 60, rake_rad=math.radians(-7), tool=False, tool_temp_K=None, limit=False),
    dict(t1_m=1e-4, speed_m_s=150 / 60, rake_rad=0.0, tool=True, tool_temp_K=500.0, limit=True),
    dict(t1_m=3e-4, speed_m_s=100 / 60, rake_rad=math.radians(10), tool=True, tool_temp_K=800.0, limit=True),
]


@pytest.mark.parametrize("cond", CONDITIONS)
def test_kernel_against_scipy_reference(steel, carbide, cond):
    tool = carbide if cond["tool"] else None
    res = oxley.solve(steel, cond["t1_m"], cond["speed_m_s"], cond["rake_rad"], tool=tool,
                      tool_temp_K=cond["tool_temp_K"], coulomb_limit=cond["limit"])
    ref = oxley.solve_reference(steel, cond["t1_m"], cond["speed_m_s"], cond["rake_rad"], tool=tool,
                                tool_temp_K=cond["tool_temp_K"], coulomb_limit=cond["limit"])
    assert res.converged and ref["status"] == "oxley"
    assert res.cutting_force_N_per_m == pytest.approx(ref["Fc"], rel=3e-3)
    assert res.thrust_force_N_per_m == pytest.approx(ref["Ft"], rel=1e-2)
    assert res.shear_angle_rad == pytest.approx(ref["phi"], rel=3e-3)
    assert res.T_interface_K == pytest.approx(ref["T_int"], rel=3e-3)
    assert res.T_shear_K == pytest.approx(ref["T_AB"], rel=3e-3)


def test_solution_satisfies_oxleys_conditions(steel, carbide):
    res = oxley.solve(steel, 2e-4, 2.5, math.radians(5), tool=carbide, tool_temp_K=600.0)
    v = res.vector
    assert v[abi.OX_C0_RES] < 1e-6 and v[abi.OX_G1_RES] < 1e-6
    assert v[abi.OX_SIGMA_N] == pytest.approx(v[abi.OX_SIGMA_NP], rel=1e-6)
    # theta and lambda relations, forces from the resultant, energy balance F_C V = F_S V_s + F V_c
    assert v[abi.OX_THETA] == pytest.approx(v[abi.OX_PHI] + v[abi.OX_LAMBDA] - math.radians(5), rel=1e-12)
    assert v[abi.OX_FT] / v[abi.OX_FC] == pytest.approx(math.tan(v[abi.OX_LAMBDA] - math.radians(5)), rel=1e-12)
    power = v[abi.OX_FC] * 2.5
    assert v[abi.OX_FS] * v[abi.OX_VS] + v[abi.OX_F] * v[abi.OX_VC] == pytest.approx(power, rel=1e-12)
    assert v[abi.OX_P_SHEAR] == pytest.approx(v[abi.OX_FS] * v[abi.OX_VS] / power, rel=1e-12)
    assert 0.0 < res.chip_ratio < 1.0 and res.chip_thickness_m == pytest.approx(2e-4 / res.chip_ratio)


def test_delta_minimises_cutting_force(steel):
    """Perturbing delta and re-solving phi gives no smaller cutting force (Oxley's criterion)."""
    inputs = kernel_inputs(steel, None, tool_temp_K=None, mu_limit=-1.0, mu_fallback=0.5)
    case = np.array([1.5e-4, 2.5, 0.0, 293.15])
    out = np.zeros(abi.N_OUT)
    K.oxley_point(case, *inputs, out)
    best = out[abi.OX_FC]
    nphi = int(inputs.prm[abi.P_PHI_N])
    phis = np.linspace(inputs.prm[abi.P_PHI_MIN], inputs.prm[abi.P_PHI_MAX], nphi)
    states = np.zeros((nphi, K.N_ST))
    valid = np.zeros(nphi, np.int64)
    st = np.zeros(K.N_ST)
    for i, phi in enumerate(phis):
        if K.solve_strain_rate_constant(phi, case, *inputs[:5], inputs.prm, st) == 1:
            valid[i] = 1
            states[i] = st
    res = np.zeros(3)
    for factor in (0.8, 0.9, 1.1, 1.25):
        fc = K.phi_root_on_grid(out[abi.OX_DELTA] * factor, phis, states, valid, case, *inputs, res)
        assert fc >= best * (1.0 - 2e-4)


def test_trends(steel, carbide):
    base = dict(tool=carbide, tool_temp_K=500.0)
    f1 = oxley.solve(steel, 1e-4, 2.5, 0.0, **base)
    f2 = oxley.solve(steel, 2e-4, 2.5, 0.0, **base)
    assert f2.cutting_force_N_per_m > f1.cutting_force_N_per_m
    assert f2.cutting_force_N_per_m / 2e-4 < f1.cutting_force_N_per_m / 1e-4        # size effect
    slow = oxley.solve(steel, 1.5e-4, 1.0, 0.0, **base)
    fast = oxley.solve(steel, 1.5e-4, 6.0, 0.0, **base)
    assert fast.T_interface_K > slow.T_interface_K
    low_rake = oxley.solve(steel, 1.5e-4, 2.5, math.radians(-5), **base)
    high_rake = oxley.solve(steel, 1.5e-4, 2.5, math.radians(15), **base)
    assert high_rake.shear_angle_rad > low_rake.shear_angle_rad
    assert high_rake.cutting_force_N_per_m < low_rake.cutting_force_N_per_m
    hot_work = oxley.solve(steel, 1.5e-4, 2.5, 0.0, work_temp_K=600.0, **base)
    assert hot_work.cutting_force_N_per_m < oxley.solve(steel, 1.5e-4, 2.5, 0.0, **base).cutting_force_N_per_m


def test_tool_temperature_and_partition(steel, carbide):
    adiabatic = oxley.solve(steel, 1.5e-4, 2.5, 0.0)
    assert adiabatic.rake_heat_to_chip == 1.0
    cold = oxley.solve(steel, 1.5e-4, 2.5, 0.0, tool=carbide, tool_temp_K=293.15)
    hot = oxley.solve(steel, 1.5e-4, 2.5, 0.0, tool=carbide, tool_temp_K=900.0)
    assert 0.0 < cold.rake_heat_to_chip < hot.rake_heat_to_chip <= 1.0
    # Oxley's condition (the default): a hotter tool gives a hotter interface, the adiabatic tool the hottest
    assert cold.regime == hot.regime == "sticking"
    assert 293.15 < cold.T_interface_K < hot.T_interface_K <= adiabatic.T_interface_K + 1e-6
    # the Coulomb-Tresca alternative: the cold tool slides (delta at its lower bound, surface source) and the
    # hot tool sticks at a larger delta, so the interface temperatures differ by < 1 %
    cold_c = oxley.solve(steel, 1.5e-4, 2.5, 0.0, tool=carbide, tool_temp_K=293.15, coulomb_limit=True)
    hot_c = oxley.solve(steel, 1.5e-4, 2.5, 0.0, tool=carbide, tool_temp_K=900.0, coulomb_limit=True)
    assert hot_c.T_interface_K == pytest.approx(cold_c.T_interface_K, rel=0.01)
    short = oxley.solve(steel, 1.5e-4, 2.5, 0.0, tool=carbide, tool_temp_K=293.15, contact_time_s=2e-4)
    assert short.rake_heat_to_chip < cold.rake_heat_to_chip   # a fresh contact draws more heat into the tool


def test_default_interface_is_oxleys_condition(steel, carbide):
    """Friction is an output: at the solution the interface shear stress equals the chip's shear
    flow stress at the interface state, and the friction angle is not atan of the database mu."""
    res = oxley.solve(steel, 1.5e-4, 2.5, 0.0, tool=carbide, tool_temp_K=500.0)
    v = res.vector
    assert res.converged and res.regime == "sticking"
    assert v[abi.OX_TAU_INT] == pytest.approx(v[abi.OX_K_CHIP], rel=1e-6)
    # the database friction coefficient does not enter the rake-face solution (only the reported
    # Zorev split), while it decides the Coulomb-Tresca alternative
    low_mu = work_from(STEEL_1045, friction=[{"tool_class": "carbide", "coating": "uncoated", "mu_sliding": 0.2}])
    other = oxley.solve(low_mu, 1.5e-4, 2.5, 0.0, tool=carbide, tool_temp_K=500.0)
    slots = [abi.OX_PHI, abi.OX_FC, abi.OX_FT, abi.OX_LAMBDA, abi.OX_T_INT, abi.OX_LC, abi.OX_DELTA]
    assert np.array_equal(other.vector[slots], v[slots])
    assert oxley.solve(low_mu, 1.5e-4, 2.5, 0.0, tool=carbide, tool_temp_K=500.0,
                       coulomb_limit=True).friction_angle_rad == pytest.approx(math.atan(0.2), rel=1e-6)
    assert res.effective_rake_rad == 0.0 and not res.bue_active and res.fallback_reference == ""
    # identical to the explicit request of Oxley's condition
    explicit = oxley.solve(steel, 1.5e-4, 2.5, 0.0, tool=carbide, tool_temp_K=500.0, coulomb_limit=False)
    assert np.array_equal(explicit.vector, v)
    assert oxley.interface_coulomb(None) is False and oxley.interface_coulomb(True) is True


def test_coulomb_alternative_only_on_request(steel, carbide):
    sliding = oxley.solve(steel, 1e-4, 2.0, 0.0, tool=carbide, tool_temp_K=500.0, coulomb_limit=True)
    assert sliding.regime == "sliding"
    assert sliding.friction_angle_rad == pytest.approx(math.atan(0.5), rel=1e-6)
    assert sliding.interface_shear_Pa == pytest.approx(0.5 * sliding.interface_normal_Pa, rel=1e-6)


def test_coolant_friction_lowers_forces(steel, carbide):
    """A lubricant at the interface carries the coolant's friction factor f times the chip's shear
    flow stress (Tresca factor form of Oxley's condition); f = 1 is the dry, seized interface."""
    dry = oxley.solve(steel, 1e-4, 2.0, 0.0, tool=carbide, tool_temp_K=500.0)
    wet = oxley.solve(steel, 1e-4, 2.0, 0.0, tool=carbide, tool_temp_K=500.0, friction_factor=0.6)
    assert dry.regime == "sticking" and wet.regime == "lubricated"
    assert wet.interface_shear_Pa == pytest.approx(0.6 * wet.vector[abi.OX_K_CHIP], rel=1e-6)
    assert wet.cutting_force_N_per_m < dry.cutting_force_N_per_m
    assert wet.shear_angle_rad > dry.shear_angle_rad
    s, tau_p, sigma_max = wet.sticking(0.3)
    assert 0.0 <= s <= 1.0
    # the Coulomb alternative with the same coolant: mu x f limits the interface
    wet_c = oxley.solve(steel, 1e-4, 2.0, 0.0, tool=carbide, tool_temp_K=500.0, friction_factor=0.6,
                        coulomb_limit=True)
    assert wet_c.regime == "sliding" and wet_c.friction_angle_rad == pytest.approx(math.atan(0.3), rel=1e-6)


def test_low_speed_has_no_degenerate_root(steel, carbide):
    """At low speed the strain-hardened interface layer is stronger than the shear plane can load it
    (no admissible root, or a root whose force-minimising delta is at the top of its range: a chip
    ten times thicker than the cut); the fallback is continued from the edge of the admissible
    solutions and never uses such a root."""
    res = oxley.solve(steel, 1.4e-4, 0.5, 0.0, tool=carbide, tool_temp_K=293.15)
    assert res.status == "merchant" and res.fallback_reference.startswith("edge of the admissible")
    assert res.shear_angle_rad > math.radians(8.0) and res.chip_ratio > 0.15
    # the reference is an admissible Oxley state: its delta is inside the range
    inputs = oxley.interface_inputs(steel, carbide, tool_temp_K=293.15)
    ref, _ = oxley.continuation_reference(np.array([1.4e-4, 0.5, 0.0, 293.15]), inputs)
    assert ref[abi.OX_STATUS] == abi.STATUS_OXLEY and ref[abi.OX_DELTA] < 0.2 * K.DELTA_EDGE
    assert res.shear_angle_rad == pytest.approx(ref[abi.OX_PHI], rel=1e-9)
    # the kernel alone rejects the boundary-delta root and returns its own fallback
    out = np.zeros(abi.N_OUT)
    K.oxley_point(np.array([1.4e-4, 0.5, 0.0, 293.15]), *inputs, out)
    assert out[abi.OX_STATUS] != abi.STATUS_OXLEY or out[abi.OX_DELTA] < 0.2 * K.DELTA_EDGE


def test_fallback_is_grid_independent(steel, carbide):
    """A map node without an admissible solution takes the same value as the point solver (both
    continue from the bisected edge of the admissible region)."""
    from cnctwin.mechanics import coefficients as CF
    t1 = 1.4e-4
    for grid in (np.array([t1, 4e-4, 1e-3]), np.geomspace(2e-5, 2e-3, 9) * (t1 / np.geomspace(2e-5, 2e-3, 9)[3])):
        pmap = CF.build_process_map(steel, carbide, 0.0, speed_grid_m_s=np.array([0.5]), t1_grid_m=grid,
                                    tool_temp_grid_K=np.array([293.15]))
        node = pmap.values[0, 0, int(np.argmin(np.abs(grid - t1)))]
        direct = oxley.solve(steel, t1, 0.5, 0.0, tool=carbide, tool_temp_K=293.15)
        assert node[abi.OX_STATUS] == abi.STATUS_MERCHANT
        assert node[abi.OX_PHI] == pytest.approx(direct.shear_angle_rad, rel=0.01)
        assert node[abi.OX_FC] == pytest.approx(direct.cutting_force_N_per_m, rel=0.01)


def test_bue_option(steel, carbide):
    """Inside the database BUE window (fixture: 10-50 m/min, interface 200-550 degC) the chip flows
    over a built-up edge of effective rake bue_rake_rad (25 deg): a larger shear angle and a
    thinner chip; outside the window and by default nothing changes."""
    plain = oxley.solve(steel, 1e-4, 0.5, 0.0, tool=carbide, tool_temp_K=293.15)
    bue = oxley.solve(steel, 1e-4, 0.5, 0.0, tool=carbide, tool_temp_K=293.15, bue=True)
    assert bue.bue_active and bue.effective_rake_rad == pytest.approx(math.radians(25.0))
    assert bue.shear_angle_rad > plain.shear_angle_rad and bue.chip_ratio > plain.chip_ratio
    assert bue.cutting_force_N_per_m < plain.cutting_force_N_per_m
    fast = oxley.solve(steel, 1e-4, 3.0, 0.0, tool=carbide, tool_temp_K=293.15, bue=True)
    assert not fast.bue_active and fast.effective_rake_rad == 0.0
    half = oxley.solve(steel, 1e-4, 0.5, 0.0, tool=carbide, tool_temp_K=293.15, bue_height_m=0.5e-4)
    assert half.bue_active and half.effective_rake_rad == pytest.approx(math.radians(12.5))
    assert oxley.bue_effective_rake(steel, 1e-4, 0.5, math.radians(30.0), 500.0, True) == math.radians(30.0)


def test_zorev_split():
    s, tau_p, sigma_max = oxley.zorev_split(300e6, 900e6, 0.5, 3.0)
    assert sigma_max == pytest.approx(3.6e9)
    assert (1 - s) ** 3 * (1 + 3 * s) == pytest.approx(300 / 450, rel=1e-12)
    assert tau_p == pytest.approx(0.5 * sigma_max * (1 - s) ** 3)
    # mean shear of the distribution equals tau_int
    x = np.linspace(0.0, 1.0, 200001)
    tau = np.where(x <= s, tau_p, 0.5 * sigma_max * (1 - x) ** 3)
    assert np.trapezoid(tau, x) == pytest.approx(300e6, rel=1e-6)
    assert oxley.zorev_split(500e6, 900e6, 0.5, 3.0)[0] == 0.0


@pytest.mark.parametrize("rake_deg,t1_m,speed", [(-40.0, 1e-4, 2.0), (-55.0, 5e-5, 0.5), (0.0, 2e-7, 0.1),
                                                 (30.0, 1e-3, 10.0), (0.0, 1e-4, 1e-3)])
def test_extreme_conditions_stay_finite(steel, rake_deg, t1_m, speed):
    res = oxley.solve(steel, t1_m, speed, math.radians(rake_deg))
    assert res.status in ("oxley", "merchant", "indentation")
    assert np.all(np.isfinite(res.vector))
    if res.status != "indentation":
        assert res.cutting_force_N_per_m > 0.0 and res.T_interface_K >= res.T_shear_K - 1e-6


def test_fallback_continuity(steel):
    """The modified-Merchant fallback with the neighbour's constants reproduces Oxley at the
    neighbour's own condition (exactly) and stays within 15 % one thickness node away."""
    inputs = kernel_inputs(steel, None, tool_temp_K=None, mu_limit=0.5, mu_fallback=0.5)
    out = np.zeros(abi.N_OUT)
    case = np.array([1e-4, 2.5, 0.0, 293.15])
    K.oxley_point(case, *inputs, out)
    prm = inputs.prm.copy()
    prm[abi.P_REF_LAMBDA] = out[abi.OX_LAMBDA]
    prm[abi.P_REF_CM] = 2 * out[abi.OX_PHI] + out[abi.OX_LAMBDA]
    prm[abi.P_REF_C0] = out[abi.OX_C0]
    prm[abi.P_REF_DELTA] = out[abi.OX_DELTA]
    fb = np.zeros(abi.N_OUT)
    K.merchant_point(case, *inputs[:-1], prm, fb)
    assert fb[abi.OX_STATUS] == abi.STATUS_MERCHANT
    assert fb[abi.OX_FC] == pytest.approx(out[abi.OX_FC], rel=1e-6)
    case2 = np.array([1.72e-4, 2.5, 0.0, 293.15])
    K.merchant_point(case2, *inputs[:-1], prm, fb)
    ox2 = np.zeros(abi.N_OUT)
    K.oxley_point(case2, *inputs, ox2)
    assert fb[abi.OX_FC] == pytest.approx(ox2[abi.OX_FC], rel=0.15)
