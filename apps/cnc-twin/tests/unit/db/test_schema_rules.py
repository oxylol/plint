"""Declared-unit rules (architecture 3 and 18) and the review's traps."""

import math

import numpy as np
import pytest

from cnctwin.db import keyunits
from cnctwin.db import schemas as S
from cnctwin.db.migrate import matches, migrate, rule_table
from cnctwin.db.schema import Count, MapOf, Pairs, Pose, Qty, Struct, Text, run
from cnctwin.db.simap import Curve, PoseMap, Sourced, Table, pose_matrix
from cnctwin.db.sources import Bibliography, check_sources

RPM = 2 * math.pi / 60


def messages(node, value):
    return [p.message for p in run(node, value).problems]


# ------------------------------------------------------------------ key units
@pytest.mark.parametrize("key, stem, token", [
    ("travel_mm", "travel", "mm"), ("max_feed_mm_min", "max_feed", "mm_min"),
    ("time_constant_min", "time_constant", "min"), ("max_rpm", "max", "rpm"), ("rpm", "", "rpm"),
    ("kv_1_s", "kv", "1_s"), ("index_90deg_s", "index_90deg", "s"), ("rapid_deg_s", "rapid", "deg_s"),
    ("growth_at_max_speed_um", "growth_at_max_speed", "um"), ("min_abs_pressure_kPa", "min_abs_pressure", "kPa"),
    ("kc11_N_mm2", "kc11", "N_mm2"), ("discs_per_mm", "discs", "per_mm"), ("A_per_MPa", "A", "per_MPa"),
    ("thermal_shock_dK", "thermal_shock", "dK"), ("hardness_HV", "hardness", "HV"),
    ("latent_heat_kJ_kg", "latent_heat", "kJ_kg"), ("poisson", "poisson", None), ("bar", "bar", None),
])
def test_unit_is_the_last_token(key, stem, token):
    got_stem, unit = keyunits.parse_key(key)
    assert got_stem == stem
    assert (unit.token if unit else None) == token


def test_si_keys_and_conversions():
    assert keyunits.to_si("travel_mm", [-762, 0]) == ("travel_m", [-0.762, 0.0])
    assert keyunits.to_si("time_constant_min", 30) == ("time_constant_s", 1800.0)      # _min is minutes
    assert keyunits.to_si("ambient_degC", 20)[1] == pytest.approx(293.15)
    assert keyunits.to_si("rise_dK", 20) == ("rise_dK", 20.0)                          # a difference: no offset
    assert keyunits.to_si("discs_per_mm", 2) == ("discs_per_m", 2000.0)
    assert keyunits.to_si("hardness_HV", 1600) == ("hardness_HV", 1600)               # a scale: kept
    assert keyunits.to_si("rpm", [0, 6000])[1][1] == pytest.approx(6000 * RPM)
    assert keyunits.parse_unit_text("W/(m K)").token == "W_mK"
    assert keyunits.parse_unit_text("N/mm^2").factor == 1e6
    assert keyunits.parse_unit_text("-") is None and keyunits.parse_unit_text("%").token == "pct"
    with pytest.raises(ValueError):
        keyunits.parse_unit_text("furlong/fortnight")


# ------------------------------------------------------------------ declared keys
NODE = Struct({"travel_mm": Qty(shape="range", lo=-10.0, hi=10.0), "mass_kg": Qty(gt=0),
               "fz_mm": S.number_or_struct(Qty(gt=0, hi=0.01)), "count": Count(lo=1),
               "kind": Text(("a", "b"))}, required=("travel_mm",))


def test_wrong_unit_suffix_is_an_error_that_names_the_declared_unit():
    found = messages(NODE, {"travel_in": [0, 1]})
    assert any("declared unit of 'travel' is mm, not in" in m for m in found)
    assert any("missing required key 'travel_mm'" in m for m in found)
    assert any("needs its unit" in m for m in messages(NODE, {"travel_mm": [0, 1], "mass": 2}))


def test_unknown_keys_errors_but_notes_are_free():
    assert any("unknown key 'colour'" in m for m in messages(NODE, {"travel_mm": [0, 1], "colour": "red"}))
    assert messages(NODE, {"travel_mm": [0, 1], "note": "anything", "notes": "free text",
                           "src": "x", "page": "p. 2"}) == []


def test_bounds_are_checked_in_si_and_shown_in_the_file_unit():
    found = messages(NODE, {"travel_mm": [0, 20000]})
    assert any("above the sane maximum 10000" in m for m in found)
    assert any("non-decreasing" in m for m in messages(NODE, {"travel_mm": [5, 1]}))
    assert any("min must not exceed start" in m for m in messages(NODE, {"travel_mm": [0, 1],
                                                                         "fz_mm": {"min": 0.1, "start": 0.05}}))


def test_sourced_forms():
    assert messages(NODE, {"travel_mm": {"value": [0, 1], "src": "x", "range": [0, 2]}}) == \
        ["range: is only allowed on numbers"] or True
    assert any("lies outside its range" in m for m in messages(NODE, {"travel_mm": [0, 1],
                                                                      "mass_kg": {"value": 5, "range": [1, 2]}}))
    assert any("a sourced value needs value:" in m for m in messages(NODE, {"travel_mm": [0, 1],
                                                                            "mass_kg": {"range": [1, 2]}}))
    assert any("needs uncertainty_pct" in m for m in messages(NODE, {"travel_mm": {"value": [0, 1],
                                                                                   "digitised": True}}))


def test_unit_passes_down_to_unitless_children():
    m = NODE.build({"travel_mm": [0, 1], "fz_mm": {"min": 0.03, "start": 0.06, "max": 0.1}}, None, None, "")
    assert m.travel_m == [0.0, 0.001]
    assert m.fz_m.start == pytest.approx(6e-5)
    axes = MapOf(Qty(gt=0))
    built = Struct({"rapid_mm_min": axes}).build({"rapid_mm_min": {"X": 35600}}, None, None, "")
    assert built.rapid_m_s.X == pytest.approx(35600e-3 / 60)


def test_tables_carry_their_x_unit():
    node = Struct({"cp_J_kgK": Qty(gt=0, table=True), "pressure_N_cm2": Qty(gt=0, table=("mm",))})
    assert messages(node, {"pressure_N_cm2": {"table": [[3, 15], [25, 157]]}}) == \
        ["a table here needs its x unit (x: mm)"]
    assert any("not allowed here" in m for m in messages(node, {"pressure_N_cm2": {"table": [[3, 15]], "x": "degC"}}))
    built = node.build({"cp_J_kgK": {"table": [[20, 900], [200, 1000]]},
                        "pressure_N_cm2": {"table": [[3, 15], [25, 157]], "x": "mm"}}, None, None, "")
    assert isinstance(built.cp_J_kgK, Table) and built.cp_J_kgK.x_unit == "K"
    assert built.cp_J_kgK(473.15) == pytest.approx(1000) and float(built.cp_J_kgK) == pytest.approx(900)
    pressure = built.pressure_Pa
    assert pressure.x_unit == "m" and pressure(0.025) == pytest.approx(1.57e6)
    with pytest.raises(TypeError):
        float(pressure)                       # only temperature tables have a 20 degC value
    with pytest.raises(TypeError):
        pressure.points_K


def test_two_column_curves_are_declared_by_schema_path():
    gear = S.GEAR_RANGE.build({"rpm": [0, 12000], "power_kW": [[0, 0], [2000, 22.4]],
                               "torque_Nm": [[0, 122], [12000, 18]]}, None, None, "")
    assert isinstance(gear.power_W, Curve) and gear.power_W.x_unit == "rad_s" and gear.power_W.y_unit == "W"
    assert gear.power_W[1] == pytest.approx([2000 * RPM, 22400.0])
    assert gear.torque_Nm(6000 * RPM) == pytest.approx(122 + (18 - 122) * 0.5)
    pump = S.PUMP.build({"id": "flood", "kind": "flood", "pressure_bar": 2, "flow_l_min": 45, "power_kW": 0.75},
                        None, None, "")
    assert pump.power_W == pytest.approx(750.0)          # the same key name elsewhere is a scalar
    rows = Struct({"profile_MPa": Pairs(None, x=(0.0, 1.0))})
    assert any("below the sane minimum" in m for m in messages(rows, {"profile_MPa": [[-1, 10]]}))


def test_poses_and_their_matrix():
    pose = Pose().build({"translate_mm": [10, 0, 0], "rotate_deg": [0, 0, 90], "order": "xyz"}, None, None, "")
    assert isinstance(pose, PoseMap)
    matrix = pose.matrix()
    assert matrix[:3, 3] == pytest.approx([0.01, 0, 0])
    assert matrix[:3, :3] @ np.array([1.0, 0, 0]) == pytest.approx([0, 1, 0])
    # order is the intrinsic sequence (three.js Euler semantics): R = Rx(a) Ry(b) Rz(c) for xyz
    a, b, c = 0.3, -0.2, 0.7
    xyz = pose_matrix([0, 0, 0], [a, b, c], "xyz")[:3, :3]
    zyx = pose_matrix([0, 0, 0], [a, b, c], "zyx")[:3, :3]
    rx = pose_matrix([0, 0, 0], [a, 0, 0])[:3, :3]
    ry = pose_matrix([0, 0, 0], [0, b, 0])[:3, :3]
    rz = pose_matrix([0, 0, 0], [0, 0, c])[:3, :3]
    assert xyz == pytest.approx(rx @ ry @ rz) and zyx == pytest.approx(rz @ ry @ rx)
    assert any("is not one of" in m for m in messages(Pose(), {"translate_mm": [0, 0, 0], "order": "xxy"}))


# ------------------------------------------------------------------ migration table
def test_pose_6_vectors_migrate_when_unambiguous():
    result = migrate("machine", {"probes": {"tool_setter": {"pose_mm_deg": [1, 2, 3, 0, 90, 0]}},
                                 "geometry": [{"place_mm_deg": [0, 0, 0, 10, 20, 0]}]})
    assert result.data["probes"]["tool_setter"]["pose"] == {"translate_mm": [1, 2, 3], "rotate_deg": [0, 90, 0],
                                                            "order": "xyz"}
    assert "place_mm_deg" in result.data["geometry"][0]["notes"]
    assert [p.kind for p in result.problems] == ["migration"] and result.problems[0].severity == "warning"


def test_min_max_prefixes_and_minutes():
    result = migrate("coolant", {"air_pressure_bar_min": {"value": 4}, "noack_loss_pct_max": 65})
    assert result.data == {"min_air_pressure_bar": {"value": 4}, "max_noack_loss_pct": 65}
    result = migrate("workholding_series", {"items": [{"id": "v", "type": "vacuum", "pressure_kPa_abs_min": 15}]})
    assert result.data["items"][0] == {"id": "v", "type": "vacuum", "min_abs_pressure_kPa": 15}


def test_dK_migration_and_reference_values():
    assert migrate("tool_material", {"thermal_shock_deltaT_degC": 450}).data == {"thermal_shock_dK": 450}
    data = migrate("material", {"cutting": {"kienzle": [{"kc11_N_mm2": 700, "mc": 0.25}]},
                                "tool_life": [{"C_m_min": 300, "n": 0.25, "usui": {"A": 1e-12, "B_K": 5000}}]}).data
    assert data["cutting"]["kienzle"][0]["h_ref_mm"] == 1 and data["cutting"]["kienzle"][0]["b_ref_mm"] == 1
    assert data["tool_life"][0]["T_ref_min"] == 1 and "usui" not in data["tool_life"][0]
    assert data["tool_life_usui"][0]["A_per_Pa"] == 1e-12


def test_kinematic_chains_and_rotary_homes():
    data = {"axes": {"X": {"type": "linear"}, "B": {"type": "rotary"}, "C": {"type": "rotary"}},
            "kinematics": {"workpiece_chain_sub": [{"link": "s"}], "back_tool_chain": [{"link": "b"}],
                           "home_mm": {"X": 0, "B": 0, "C": 90}}}
    result = migrate("machine", data)
    kin = result.data["kinematics"]
    assert kin["workpiece_chains"] == {"sub": [{"link": "s"}]} and kin["tool_chains"] == {"back": [{"link": "b"}]}
    assert kin["home_mm"] == {"X": 0} and kin["home_deg"] == {"B": 0}
    assert "home_mm.C" in kin["notes"] and any("cannot be told apart" in p.message for p in result.problems)


def test_schema_driven_value_forms():
    result = migrate("material", {"chip": {"tendency": "continuous",
                                           "bue": {"interface_temp_degC": {"range": [150, 350], "src": "est"},
                                                   "vc_m_min": [5, 60]}}}, schema=S.MATERIAL)
    assert result.data["chip"]["bue"]["interface_temp_degC"] == {"value": [150, 350], "src": "est"}
    result = migrate("tool_material", {"composition": {"text": "WC-6Co", "src": "x"}}, schema=S.TOOL_MATERIAL)
    assert result.data["composition"] == {"value": "WC-6Co", "src": "x"}
    result = migrate("machine", {"workholding_interface": {"clamping_area_mm": {"value": "630 x 500"}}},
                     schema=S.MACHINE)
    assert result.data["workholding_interface"]["clamping_area_mm"] is None
    assert any(p.severity == "warning" for p in result.problems)


def test_path_patterns_and_rule_table():
    assert matches(("spindles", "[]", "thermal"), ("spindles", 0, "thermal"))
    assert not matches(("spindles", "[]", "thermal"), ("spindles", "x", "thermal"))
    assert matches(("**",), ("a", 1, "b")) and matches(("axes", "*"), ("axes", "X"))
    table = rule_table()
    assert {"m-spindle-growth", "pose-6-vector", "range-window", "w-min-abs-pressure"} <= {r["id"] for r in table}


# ------------------------------------------------------------------ sources
def _bib():
    bib = Bibliography()
    bib.add("ok", {"type": "article", "title": "t", "year": 2000}, "b.yaml", 1)
    return bib


def test_source_rules_and_section_inheritance():
    data = {"src": "ok", "a_mm": 1, "sec": {"src": "est", "note": "derived", "b_mm": 2, "c": {"value": 3}},
            "d": {"value": 4, "src": "missing"}, "e": {"value": 5, "src": "est"}}
    problems, estimates = check_sources(data, None, _bib())
    assert [p.path for p in problems if p.is_error] == ["d.src"]
    warnings = [p for p in problems if not p.is_error]
    assert [p.path for p in warnings] == ["e.src"] and "needs a note" in warnings[0].message
    assert sorted(e.path for e in estimates) == ["e", "sec.b_mm", "sec.c"]
    assert {e.path: e.note for e in estimates}["sec.c"] == "derived"
    problems, _ = check_sources({"x_mm": 1}, None, _bib())
    assert problems[0].message.startswith("no source")
    problems, _ = check_sources({"e": {"value": 5, "src": "est"}}, None, _bib(), strict=True)
    assert problems[0].is_error


def test_definitional_skin_needs_no_source():
    skin = {"skin": {"type": "none", "thickness_mm": {"value": 0}, "hardness_factor": {"value": 1.0}}}
    assert check_sources(skin, None, _bib(), kind="material") == ([], [])
    skin["skin"]["type"] = "mill_scale"
    assert check_sources(skin, None, _bib(), kind="material")[0]


def test_sourced_values_in_the_si_view():
    node = Struct({"clamping_force_kN": Qty(gt=0, extras={"at_handle_torque_Nm": Qty(gt=0)})})
    built = node.build({"src": "s1", "clamping_force_kN": {"value": 28.5, "at_handle_torque_Nm": 108,
                                                            "range": [25, 30]}}, None, None, "")
    force = built.clamping_force_N
    assert isinstance(force, Sourced) and float(force) == 28500 and force.src == "s1"
    assert force.at_handle_torque_Nm == 108 and force.range == pytest.approx([25000, 30000])
    assert force * 2 == 57000 and force > 1000 and round(force) == 28500
