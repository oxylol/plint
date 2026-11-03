"""SIMap without a schema (units by key suffix) and with one (units as declared, contract v2)."""

import math

import pytest

from cnctwin.db import schemas as S
from cnctwin.db.simap import Curve, SIMap, Sourced, Table

MATERIAL = {
    "id": "al6061-t6",
    "physical": {
        "src": "asm",
        "density_kg_m3": {"value": 2700},
        "specific_heat_J_kgK": {"table": [[20, 896], [200, 963]]},
        "youngs_modulus_GPa": {"table": [[20, 68.9], [200, 61.8]], "src": "other"},
        "solidus_degC": {"value": 582, "src": "est", "note": "from liquidus"},
    },
    "cutting": {"kienzle": [{"tool_class": "carbide", "kc11_N_mm2": 700, "mc": 0.25}]},
}

MACHINE = {
    "axes": {"X": {"travel_mm": [-762, 0], "rapid_mm_min": 35560, "accel_m_s2": {"value": 5.0, "src": "est", "note": "n"}}},
    "controller": {"rapid_mm_min": {"X": 35560, "Y": 35560}},
    "spindles": [{"max_rpm": 12000, "gear_ranges": [{"rpm": [0, 12000],
                  "power_kW": [[0, 0], [2000, 22.4]], "torque_Nm": [[0, 122], [12000, 18]]}]}],
    "tool": {"fz_mm": {"min": 0.03, "start": 0.06, "max": 0.1}, "clearance_deg": {"primary": 10}},
    "clamp": {"clamping_force_kN": {"value": 28.5, "at_handle_torque_Nm": 108, "src": "kurt"}},
}


def test_sourced_and_tables_convert_to_si_and_inherit_src():
    m = SIMap(MATERIAL)
    density = m.physical.density_kg_m3
    assert isinstance(density, Sourced) and float(density) == 2700 and density.src == "asm"
    cp = m.physical.specific_heat_J_kgK
    assert isinstance(cp, Table) and math.isclose(cp(293.15), 896) and cp.src == "asm"
    assert math.isclose(m.physical.youngs_modulus_Pa(473.15), 61.8e9)
    assert m.physical.youngs_modulus_Pa.src == "other"
    assert math.isclose(float(m.physical.solidus_K), 855.15) and m.physical.solidus_K.estimated
    assert m.cutting.kienzle[0].kc11_Pa == 7e8 and m.cutting.kienzle[0].mc == 0.25
    assert cp(2000.0) == 963 and cp.clamped == 1


def test_units_pass_down_and_pair_curves():
    m = SIMap(MACHINE)
    assert m.axes.X.travel_m == [-0.762, 0.0]
    assert math.isclose(m.controller.rapid_m_s.X, 35560e-3 / 60)
    assert math.isclose(m.spindles[0].max_rad_s, 12000 * 2 * math.pi / 60)
    gear = m.spindles[0].gear_ranges[0]
    assert isinstance(gear.power_W, Curve)          # schema-less: the curve columns the schemas declare by key
    assert math.isclose(gear.power_W[1][0], 2000 * 2 * math.pi / 60) and gear.power_W[1][1] == 22400
    assert math.isclose(gear.torque_Nm[1][0], 12000 * 2 * math.pi / 60) and gear.torque_Nm[1][1] == 18
    assert math.isclose(m.tool.fz_m.start, 6e-5)
    assert math.isclose(m.tool.clearance_rad.primary, math.radians(10))
    force = m.clamp.clamping_force_N
    assert float(force) == 28500 and force.at_handle_torque_Nm == 108 and force.src == "kurt"


def test_missing_keys():
    m = SIMap(MACHINE)
    assert m.nothing_here is None
    with pytest.raises(KeyError):
        m["nothing_here"]


def test_schema_view_converts_by_declared_unit():
    raw = {"type": "rotary", "travel_deg": [-120, 30], "max_rpm": 85, "accel_rad_s2": 30, "jerk_rad_s3": 1000,
           "servo": {"kv_1_s": 40, "velocity_bw_Hz": 80}, "backlash_arcsec": 5, "index_90deg_s": 0.5}
    axis = S.AXIS.build(raw, None, None, "B")
    assert axis.travel_rad == pytest.approx([math.radians(-120), math.radians(30)])
    assert axis.max_rad_s == pytest.approx(85 * 2 * math.pi / 60)
    assert axis.servo.kv_1_s == 40 and axis.index_90deg_s == 0.5           # 1/s and s are SI already
    assert axis.backlash_rad == pytest.approx(5 * math.pi / (180 * 3600))
    assert axis.to_dict()["travel_rad"][0] == pytest.approx(math.radians(-120))
    assert axis.raw is raw
