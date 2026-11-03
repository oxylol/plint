"""Lubrication: the database factor at its reference conditions, the capillary speed law, reach, floor, MQL
supply, the friction-factor lookup."""

import math

import pytest

from cnctwin.coolant import fluid as F
from cnctwin.coolant import lubrication as L

import coolfix as CF


def emulsion():
    fluid = F.fluid_from_db(CF.fluid_db("emulsion-8pct"))
    return fluid, fluid.props(295.0)


def reference_cut():
    """Cutting conditions equal to the database factors' reference (chip ratio 0.5)."""
    p = L.P()
    ratio = 0.5
    v_cut = p["reference_chip_speed_m_s"] / ratio
    h = p["reference_contact_length_m"] * ratio / p["contact_length_ratio"]
    return v_cut, h, ratio


def test_reference_conditions_give_the_database_factor():
    fluid, props = emulsion()
    v, h, r = reference_cut()
    lube = L.friction(fluid, props, 1.0, "carbide", "steel", v, h, r, 0.005, False, 0.0)
    assert lube.database_factor == 0.8 and lube.source == "database"
    assert lube.penetration == pytest.approx(1.0, rel=1e-12)
    assert lube.factor == pytest.approx(0.8, rel=1e-12)
    assert lube.v_chip_m_s == pytest.approx(L.P()["reference_chip_speed_m_s"])


def test_capillary_access_falls_as_inverse_square_root_of_speed():
    fluid, props = emulsion()
    v, h, r = reference_cut()
    phi_ref = L.penetration(props, L.P()["reference_chip_speed_m_s"], L.P()["reference_contact_length_m"])
    assert phi_ref < 0.5                                     # emulsion capillaries are not filled at the reference
    fast = L.friction(fluid, props, 1.0, "carbide", "steel", 4 * v, h, r, 0.005, False, 0.0)
    assert fast.penetration == pytest.approx(0.5, rel=1e-12)
    assert fast.factor == pytest.approx(1 - 0.2 * 0.5, rel=1e-12)
    # Washburn: phi = sqrt(sigma r cos / (2 mu v l))
    p = L.P()
    hand = math.sqrt(props.sigma_N_per_m * p["capillary_radius_m"] * p["wetting_cos"] /
                     (2 * props.mu_Pa_s * p["reference_chip_speed_m_s"] * p["reference_contact_length_m"]))
    assert phi_ref == pytest.approx(hand, rel=1e-12)


def test_slow_cutting_lubricates_more_down_to_the_floor():
    fluid, props = emulsion()
    v, h, r = reference_cut()
    slow = L.friction(fluid, props, 1.0, "carbide", "steel", v / 4, h, r, 0.005, False, 0.0)
    assert slow.factor < 0.8
    crawl = L.friction(fluid, props, 1.0, "carbide", "steel", v / 1e4, h, r, 0.005, False, 0.0)
    assert crawl.factor == pytest.approx(L.P()["friction_floor"])


def test_reach_scales_the_lubricating_effect():
    fluid, props = emulsion()
    v, h, r = reference_cut()
    for reach in (0.0, 0.25, 0.6):
        lube = L.friction(fluid, props, reach, "carbide", "steel", v, h, r, 0.005, False, 0.0)
        assert lube.factor == pytest.approx(1 - reach * 0.2, rel=1e-12)


def test_mql_supply_limit():
    fluid = F.fluid_from_db(CF.fluid_db("mql-ester"))
    props = fluid.props(295.0)
    v, h, r = reference_cut()
    p = L.P()
    need = p["boundary_film_m"] * 0.005 * v * r
    ample = L.friction(fluid, props, 1.0, "carbide", "steel", v, h, r, 0.005, True, 10 * need)
    starved = L.friction(fluid, props, 1.0, "carbide", "steel", v, h, r, 0.005, True, 0.25 * need)
    assert ample.supply == 1.0 and starved.supply == pytest.approx(0.25, rel=1e-12)
    assert 1.0 - starved.factor == pytest.approx(0.25 * (1.0 - ample.factor), rel=1e-12)


def test_friction_factor_lookup_falls_back_to_the_closest_data():
    fluid = F.fluid_from_db(CF.fluid_db("emulsion-8pct"))
    assert fluid.friction_factor("carbide", "aluminium") == (0.65, "database")
    value, how = fluid.friction_factor("carbide", "nickel")
    assert value == pytest.approx((0.8 + 0.65 + 0.9) / 3) and "carbide" in how
    value, how = fluid.friction_factor("cbn", "steel")
    assert value == pytest.approx((0.8 + 0.7) / 2) and "steel" in how
    air = F.fluid_from_db(CF.fluid_db("air"))
    assert air.friction_factor("carbide", "steel") == (1.0, "database")
