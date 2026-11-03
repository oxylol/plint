"""Envelope radii, chains and the property models against closed forms."""

import math

import numpy as np
import pytest

from cnctwin.tooling import build_tool
from cnctwin.tooling import params as P
from cnctwin.tooling.abi import CHAIN_HOLDER, CHAIN_SPINDLE, CHAIN_TOOL, PART_EDGE
from cnctwin.tooling.envelope import SpindleNose, spindle_nose_from_machine
from cnctwin.tooling.props import _revolved, balance, combine, envelope_radius, principal, sector_moments
from cnctwin.tooling.spec import parse_balance_grade, read_tool
from cnctwin.tooling.props import MassProps
from tests.unit.tooling.fixtures import HOLDER, TOOLS, endmill, instance

R = 5e-3


def test_endmill_envelope_radii_and_chains():
    g = build_tool(endmill(neck={"diameter_mm": 9.6, "length_mm": 32}), HOLDER, instance(stickout_mm=45),
                   spindle_nose=SpindleNose(0.085, 0.13))
    env = g.envelope
    assert set(env.chain.tolist()) == {CHAIN_TOOL, CHAIN_HOLDER, CHAIN_SPINDLE}
    tool_r = lambda z: float(envelope_radius(env, z, chain=CHAIN_TOOL)[0])  # noqa: E731
    assert tool_r(0.010) == pytest.approx(R)                      # cutting part
    assert tool_r(0.021) == pytest.approx(R)                      # fluted body to the flute length (22 mm)
    assert tool_r(0.025) == pytest.approx(4.8e-3)                 # neck to 32 mm
    assert tool_r(0.040) == pytest.approx(R)                      # shank
    union = lambda z: float(envelope_radius(env, z)[0])           # noqa: E731
    assert union(0.045 + 0.050) == pytest.approx(0.5 * (27e-3 + (44e-3 - 27e-3) * 0.5))   # holder profile
    assert union(0.045 + 0.080 + 0.05) == pytest.approx(0.085)    # spindle nose above the gauge line
    assert np.all(env.cutting[env.part == PART_EDGE] == 1)
    assert g.layout["z_gauge"] == pytest.approx(0.045 + 0.080)
    # every chain starts and ends on the axis
    for chain in (CHAIN_TOOL, CHAIN_HOLDER, CHAIN_SPINDLE):
        idx = np.nonzero(env.chain == chain)[0]
        assert env.r0_m[idx[0]] == pytest.approx(0.0, abs=1e-15)
        assert env.r1_m[idx[-1]] == pytest.approx(0.0, abs=1e-15)


def test_spindle_nose_from_machine_geometry():
    machine = {"kinematics": {"tool_chain": [{"link": "head", "axis": "Z"},
                                             {"link": "spindle_nose", "fixed": {"translate_m": [0.381, 0.203, 0.61]}}],
                              "tool_axis": [0, 0, 1]},
               "geometry": [{"name": "spindle_head", "attached_to": "head", "primitives": [
                   {"cylinder": {"p0_m": [0.381, 0.203, 0.61], "p1_m": [0.381, 0.203, 0.74], "radius_m": 0.085}},
                   {"cylinder": {"p0_m": [0.0, 0.0, 0.0], "p1_m": [0.0, 0.0, 0.1], "radius_m": 0.3}}]}]}
    nose = spindle_nose_from_machine(machine)
    assert nose is not None and nose.radius_m == pytest.approx(0.085) and nose.length_m == pytest.approx(0.13)


def test_gullet_and_fluted_section():
    g = build_tool(endmill(), HOLDER, instance())
    props = g.props
    assert props.gullet_area_per_flute_m2 == pytest.approx(9.5e-6)
    assert props.gullet_volume_per_length_m3_per_m == pytest.approx(4 * 9.5e-6)
    assert props.chip_space_m3 == pytest.approx(4 * 9.5e-6 * 22e-3)
    # sector model with the tool's own core and gullet: isotropic for 4 evenly spaced teeth
    r_c = 3e-3
    area = math.pi * R * R - 4 * 9.5e-6
    f_s = (area - math.pi * r_c ** 2) / (math.pi * (R * R - r_c * r_c))
    I = math.pi * r_c ** 4 / 4 + f_s * math.pi * (R ** 4 - r_c ** 4) / 4
    assert props.fluted.I_min_m4 == pytest.approx(I, rel=1e-12)
    assert props.fluted.I_max_m4 == pytest.approx(I, rel=1e-12)
    assert props.fluted.d_eq_m == pytest.approx((64 * I / math.pi) ** 0.25)
    assert 0.75 * 2 * R < props.fluted.d_eq_m < 0.9 * 2 * R


def test_sector_model_limits_and_two_flute_anisotropy():
    # full solid: pi R^4 / 4 about both axes
    ixx, iyy, ixy = sector_moments(R, 0.0, 1.0, [0.0, math.pi])
    assert ixx == pytest.approx(math.pi * R ** 4 / 4) and iyy == pytest.approx(ixx) and ixy == pytest.approx(0, abs=1e-30)
    # two opposite teeth of width alpha: I = (R^4 - rc^4)/4 (alpha +- sin alpha) + core
    rc, fs = 2.5e-3, 0.4
    alpha = 2 * math.pi * fs / 2
    I1, I2 = principal(*sector_moments(R, rc, fs, [0.0, math.pi]))
    ring = (R ** 4 - rc ** 4) / 4
    core = math.pi * rc ** 4 / 4
    assert I1 == pytest.approx(core + ring * (alpha + math.sin(alpha)))
    assert I2 == pytest.approx(core + ring * (alpha - math.sin(alpha)))


def test_kops_vo_fallback_without_core_and_gullet():
    item = endmill()
    del item["geometry"]["core_diameter_mm"], item["geometry"]["gullet_area_mm2"]
    g = build_tool(item, HOLDER, instance())
    D = 2 * R
    mean = math.pi * (P.pf("kops_vo_ratio") * D) ** 4 / 64
    assert 0.5 * (g.props.fluted.I_min_m4 + g.props.fluted.I_max_m4) == pytest.approx(mean, rel=1e-12)


def test_critical_sections():
    g = build_tool(endmill(neck={"diameter_mm": 9.6, "length_mm": 32}), HOLDER, instance(stickout_mm=45))
    sections = {s.name: s for s in g.props.sections}
    assert sections["neck"].Z_bending_m3 == pytest.approx(math.pi * 9.6e-3 ** 3 / 32)
    assert sections["neck"].z_m == pytest.approx(0.032)
    assert sections["shank_at_holder"].Z_polar_m3 == pytest.approx(math.pi * 10e-3 ** 3 / 16)
    assert sections["shank_at_holder"].z_m == pytest.approx(0.045)
    assert sections["flute_runout"].z_m == pytest.approx(0.022)
    turning = build_tool(TOOLS["turning_insert"], None, {"number": 1})
    shank = {s.name: s for s in turning.props.sections}["shank_at_clamp"]
    assert shank.Z_bending_m3 == pytest.approx(25e-3 ** 3 / 6)


def test_revolved_solid_closed_forms():
    z = np.linspace(0.0, 0.1, 20001)
    r = np.full_like(z, 0.01)
    mass, com, i_ax, i_tr = _revolved(z, r, np.ones_like(z), 7850.0)
    m = 7850.0 * math.pi * 0.01 ** 2 * 0.1
    assert mass == pytest.approx(m, rel=1e-4)
    assert com == pytest.approx(0.05, rel=1e-9)
    assert i_ax == pytest.approx(0.5 * m * 0.01 ** 2, rel=1e-4)
    assert i_tr == pytest.approx(m * (0.01 ** 2 / 4 + 0.1 ** 2 / 12), rel=1e-3)


def test_masses_scale_to_catalogue_and_combine():
    g = build_tool(endmill(), HOLDER, instance(stickout_mm=35))
    holder = g.props.holder_mass
    assert holder.mass_kg == pytest.approx(1.2) and holder.source == "catalogue"
    assert 0.035 < holder.com_z_m < 0.035 + 0.08
    tool = g.props.tool_mass
    assert tool.source == "geometry" and 0.04 < tool.mass_kg < 0.08
    both = g.props.assembly_mass
    assert both.mass_kg == pytest.approx(tool.mass_kg + holder.mass_kg)
    assert both.com_z_m == pytest.approx((tool.mass_kg * tool.com_z_m + holder.mass_kg * holder.com_z_m) / both.mass_kg)
    item = endmill()
    item["mass_g"] = 40
    scaled = build_tool(item, HOLDER, instance()).props.tool_mass
    assert scaled.mass_kg == pytest.approx(0.040) and scaled.source == "catalogue"
    assert scaled.com_z_m == pytest.approx(tool.com_z_m)
    assert combine(None, holder) is holder


def test_balance_and_speed():
    g, speed = parse_balance_grade("G2.5 at 25000 rpm")
    assert g == pytest.approx(2.5e-3) and speed == pytest.approx(25000 * 2 * math.pi / 60)
    b = balance("G6.3", MassProps(2.0, 0.0, 0.0, 0.0, "catalogue"), 1000.0)
    assert b.e_per_m == pytest.approx(6.3e-3 / 1000.0) and b.U_per_kg_m == pytest.approx(2.0 * 6.3e-6)
    tool = build_tool(endmill(), HOLDER, instance())
    assert tool.props.max_rad_s == pytest.approx(30000 * 2 * math.pi / 60)
    assert tool.props.balance_holder.speed_rad_s == pytest.approx(25000 * 2 * math.pi / 60)


def test_coolant_holes_and_hand():
    g = build_tool(TOOLS["twist_drill"], HOLDER, instance(stickout_mm=70))
    holes = g.props.coolant
    assert holes.count == 2 and holes.area_m2 == pytest.approx(2 * math.pi * 0.8e-3 ** 2 / 4)
    radii = np.hypot(holes.exit_pos_T_m[:, 0], holes.exit_pos_T_m[:, 1])
    assert np.allclose(radii, P.pf("coolant_exit_radius_ratio") * R)
    assert g.props.hand == "right" and g.props.hand_sign == -1.0
    assert read_tool(TOOLS["twist_drill"]).hand_sign == -1.0
