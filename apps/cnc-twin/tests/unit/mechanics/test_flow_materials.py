"""Johnson-Cook flow stress (with its clamps), the equivalent hardening index, and the material
and tool adaptors."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cnctwin.db.simap import wrap
from cnctwin.mechanics import abi, kernels as K
from cnctwin.mechanics.materials import ToolSide, coating_matches, flow_stress_fallback

from conftest import CARBIDE, STEEL_1045, work_from


def jc_by_hand(eps, rate, temp, A, B, n, C, m, eps0, tref, tmelt):
    ts = min(max((temp - tref) / (tmelt - tref), 0.0), 0.99)
    ratio = max(rate / eps0, 1.0)
    return (A + B * eps ** n) * max(1.0 + C * math.log(ratio), 0.01) * (1.0 - ts ** m)


def test_johnson_cook_values(steel):
    mat = steel.jc
    for eps, rate, temp in ((0.3, 1e4, 600.0), (1.0, 3e4, 900.0), (2.5, 1e5, 1300.0)):
        expect = jc_by_hand(eps, rate, temp, 553.1e6, 600.8e6, 0.234, 0.0134, 1.0, 1.0, 293.15, 1763.15)
        assert math.isclose(K.jc_flow_stress(eps, rate, temp, mat, 0.99, 1.0), expect, rel_tol=1e-13)


def test_johnson_cook_clamps(steel):
    mat = steel.jc
    cold = K.jc_flow_stress(0.5, 1e3, 100.0, mat, 0.99, 1.0)          # below T_ref: T* clamped to 0
    room = K.jc_flow_stress(0.5, 1e3, 293.15, mat, 0.99, 1.0)
    assert cold == room
    hot = K.jc_flow_stress(0.5, 1e3, 5000.0, mat, 0.99, 1.0)           # above melt: T* = 0.99
    assert hot > 0.0 and math.isclose(hot, room * 0.01, rel_tol=1e-12)
    slow = K.jc_flow_stress(0.5, 1e-6, 600.0, mat, 0.99, 1.0)          # below the reference rate: no softening
    ref = K.jc_flow_stress(0.5, 1.0, 600.0, mat, 0.99, 1.0)
    assert slow == ref
    assert K.jc_flow_stress(-1.0, 1e3, 600.0, mat, 0.99, 1.0) == K.jc_flow_stress(0.0, 1e3, 600.0, mat, 0.99, 1.0)


def test_hardening_index_is_log_derivative(steel):
    mat = steel.jc
    for eps in (0.05, 0.5, 2.0):
        d = 1e-6
        s1 = K.jc_flow_stress(eps * (1 - d), 1e3, 600.0, mat, 0.99, 1.0)
        s2 = K.jc_flow_stress(eps * (1 + d), 1e3, 600.0, mat, 0.99, 1.0)
        numeric = (math.log(s2) - math.log(s1)) / (math.log(1 + d) - math.log(1 - d))
        assert math.isclose(K.jc_hardening_index(eps, mat), numeric, rel_tol=1e-6)
    assert K.jc_hardening_index(0.0, mat) == 0.0


def test_work_material_adaptor(steel, polymer, composite):
    assert steel.jc[abi.MAT_A_PA] == pytest.approx(553.1e6)
    assert steel.jc[abi.MAT_TMELT_K] == pytest.approx(1763.15)
    assert steel.jc[abi.MAT_RHO_KG_M3] == 7850.0
    assert steel.cp(293.15) == pytest.approx(460.0) and steel.conductivity(1273.15) == pytest.approx(24.02)
    assert steel.kienzle[0].kc11_Pa == pytest.approx(1.672e9) and steel.kienzle[0].h_ref_m == pytest.approx(1e-3)
    assert steel.kienzle[0].calibration.edge_radius_m == pytest.approx(37.5e-6)
    assert steel.bue.speed_m_s == pytest.approx((10 / 60, 50 / 60))
    assert steel.indentation_estimated
    assert steel.friction_for("carbide", "none").mu == 0.5
    assert steel.friction_for("carbide", "tin").mu == 0.4
    assert steel.friction_for("hss", "none").mu == 0.5          # nearest entry of another class
    assert polymer.jc is None and polymer.melt_smear_K == pytest.approx(531.15)
    assert composite.dust_mass_fraction == 1.0
    assert flow_stress_fallback(polymer, 293.15) == pytest.approx(84e6)
    assert flow_stress_fallback(polymer, 600.0) == 0.0


def test_constant_properties_become_one_point_tables():
    work = work_from(STEEL_1045, physical__specific_heat_J_kgK={"value": 470, "src": "est"},
                     physical__conductivity_W_mK=45.0)
    assert work.cp_T_K.shape == (1,) and work.cp(800.0) == 470.0 and work.conductivity(300.0) == 45.0
    assert np.all(np.isfinite(work.cp_J_kgK))


def test_coating_matching():
    assert coating_matches("TiAlN", "tialn") == 3
    assert coating_matches("CVD TiCN/Al2O3", "ticn") == 3
    assert coating_matches("uncoated", "none") == 3
    assert coating_matches("uncoated", "tialn") == 0
    assert coating_matches("any", "tialn") == 2


def test_tool_side_adaptor(carbide, coated_carbide):
    assert carbide.tool_class == "carbide" and carbide.coating_resistance_m2K_W == 0.0
    assert coated_carbide.coating == "tialn"
    assert coated_carbide.coating_resistance_m2K_W == pytest.approx(3e-6 / 3.0)
    assert coated_carbide.process == "turning" and coated_carbide.corner_radius_m == pytest.approx(0.8e-3)
    cb = coated_carbide.chip_breaker
    assert cb.groove_width_m == pytest.approx(1e-3) and cb.feed_window_m == pytest.approx((1e-4, 4e-4))
    typical = ToolSide.from_db(wrap({"coating": {"id": "tialn"}}), wrap(CARBIDE),
                               wrap({"conductivity_W_mK": 3.0, "typical_thickness_um": [1, 4]}))
    assert typical.coating_resistance_m2K_W == pytest.approx(2.5e-6 / 3.0)
