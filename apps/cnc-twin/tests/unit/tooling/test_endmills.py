"""End mills against closed forms: element counts and lengths, end teeth, helix lag, variable pitch
and helix, rake/clearance/inclination relations, runout radii, roughing waves."""

import math

import numpy as np
import pytest

from cnctwin.core.abi import EDGE_CORNER, EDGE_END, EDGE_PERIPHERAL
from cnctwin.tooling import build_tool, element_positions_T
from cnctwin.tooling import params as P
from cnctwin.tooling.schema import ToolingSettings
from tests.unit.tooling.fixtures import HOLDER, endmill, instance

R = 5e-3
L = 20e-3
BETA = math.radians(35)


def _square(**changes):
    return build_tool(endmill(**changes), HOLDER, instance(runout_um=0))


def test_square_periphery_counts_and_lengths():
    g = _square()
    e = g.edge
    per = e.elem_kind[0] == EDGE_PERIPHERAL
    # 3-D edge length per tooth L / cos(beta) split into ceil(L / cos(beta) / 0.5 mm) equal elements
    n_expect = math.ceil(L / math.cos(BETA) / 0.5e-3)
    assert per.sum() == n_expect
    for j in range(4):
        assert e.elem_db_m[j, per].sum() == pytest.approx(L, rel=1e-12)
        assert e.elem_len_m[j, per].sum() == pytest.approx(L / math.cos(BETA), rel=1e-6)
        assert np.all(e.elem_len_m[j, per] <= 0.5e-3 * (1 + 1e-9))
        assert np.allclose(e.elem_r_m[j, per], R)
        assert np.allclose(e.elem_kappa_rad[j, per], math.pi / 2)


def test_square_end_teeth():
    g = _square()
    e = g.edge
    dish = P.pf("end_dish_rad")
    end = e.elem_kind[0] == EDGE_END
    assert np.allclose(e.elem_kappa_rad[:, end], -dish)
    # long teeth 0 and 2 reach the axis: chip width = radial extent along the dished edge
    for j in (0, 2):
        assert e.elem_db_m[j, end][e.elem_valid[j, end] == 1].sum() == pytest.approx(R / math.cos(dish), rel=1e-9)
    # short teeth start at the core radius (mask at the element midpoint: within one element)
    core = 3e-3
    for j in (1, 3):
        width = e.elem_db_m[j, end][e.elem_valid[j, end] == 1].sum()
        assert abs(width - (R - core)) <= 0.5e-3
    # end edges are radial straight lines: no lag, finite db
    assert np.allclose(e.elem_lag_rad[:, end], 0.0)
    assert np.all(e.elem_db_m[:, end] > 0)


def test_rake_clearance_inclination_relations():
    e = _square().edge
    per = e.elem_kind == EDGE_PERIPHERAL
    gamma_r, alpha_r = math.radians(8), math.radians(10)
    assert np.allclose(e.elem_incl_rad[per], BETA, atol=1e-6)
    assert np.allclose(np.tan(e.elem_rake_rad[per]), math.tan(gamma_r) * math.cos(BETA), atol=1e-6)
    assert np.allclose(np.tan(e.elem_clear_rad[per]), math.tan(alpha_r) / math.cos(BETA), atol=1e-5)


def test_helix_lag_is_constant_lead():
    e = _square().edge
    per = e.elem_kind[0] == EDGE_PERIPHERAL
    for j in range(4):
        expect = -e.elem_z_m[j, per] * math.tan(BETA) / R
        assert np.allclose(e.elem_lag_rad[j, per], expect, atol=1e-12)


def test_variable_pitch_and_helix():
    g = build_tool(endmill(pitch_deg=[86, 94, 86, 94], helix_deg=[36, 38, 36, 38]), HOLDER, instance())
    e = g.edge
    expect = np.radians([0.0, -86.0, -180.0, -266.0]) % (2 * math.pi)
    assert np.allclose(e.tooth_phase_rad, expect, atol=1e-12)
    per = e.elem_kind[0] == EDGE_PERIPHERAL
    for j, beta in enumerate((36, 38, 36, 38)):
        expect_lag = -e.elem_z_m[j, per] * math.tan(math.radians(beta)) / R
        assert np.allclose(e.elem_lag_rad[j, per], expect_lag, atol=1e-12)
        assert np.allclose(e.elem_incl_rad[j, per], math.radians(beta), atol=1e-6)
    # stations are shared: same station index and nominal position on every tooth
    assert np.array_equal(e.elem_station[0], e.elem_station[1])
    assert np.allclose(e.station_z_m[per], e.elem_z_m[0, per])


def test_runout_radii_and_tilt():
    tir, phase = 10e-6, math.radians(30)
    g = build_tool(endmill(), HOLDER, instance(runout_um=10, runout_phase_deg=30, tilt_urad=50))
    e = g.edge
    assert np.linalg.norm(e.runout_T_m) == pytest.approx(0.5 * tir)
    pos = element_positions_T(e, 0.0)
    per = e.elem_kind[0] == EDGE_PERIPHERAL
    k0 = int(np.nonzero(per)[0][0])
    offset_angle = -phase            # right-hand tool: right-handed angle = -(cutting-sense angle)
    for j in range(4):
        alpha = -(e.tooth_phase_rad[j] + e.elem_lag_rad[j, k0])
        e_vec = 0.5 * tir * np.array([math.cos(offset_angle), math.sin(offset_angle)])
        z = e.elem_z_m[j, k0]
        tilt = 50e-6 * z * np.array([math.cos(offset_angle), math.sin(offset_angle)])
        p = R * np.array([math.cos(alpha), math.sin(alpha)]) + e_vec + tilt
        assert np.hypot(*pos[j, k0, :2]) == pytest.approx(np.hypot(*p), abs=1e-12)
        # first order: R + (e + tau z) cos(alpha - offset), up to (e + tau z)^2 / (2 R)
        offset = 0.5 * tir + 50e-6 * z
        first = R + offset * math.cos(alpha - offset_angle)
        assert np.hypot(*pos[j, k0, :2]) == pytest.approx(first, abs=offset ** 2 / (2 * R) * 1.01)


def test_toroidal_corner_elements():
    g = build_tool(endmill("toroidal_endmill", corner_radius_mm=1.0), HOLDER, instance())
    e = g.edge
    corner = e.elem_kind[0] == EDGE_CORNER
    rc = 1e-3
    kappa = e.elem_kappa_rad[0, corner]
    assert e.elem_db_m[0, corner].sum() == pytest.approx(0.5 * math.pi * rc, rel=1e-12)
    assert np.all(np.diff(kappa) > 0)
    assert np.all(np.diff(kappa) <= ToolingSettings().max_kappa_step_rad * 1.0001)
    assert np.allclose(e.elem_r_m[0, corner], R - rc + rc * np.sin(kappa), atol=2e-8)


def test_roughing_wave_shifted_per_flute():
    g = build_tool(endmill("roughing_endmill"), HOLDER, instance())
    e = g.edge
    per = e.elem_kind[0] == EDGE_PERIPHERAL
    lam = P.pf("knuckle_wavelength_per_D") * 2 * R
    depth = P.pf("knuckle_depth_per_D") * 2 * R
    z = e.elem_z_m[0, per]
    late = z > lam                     # past the ramp-in
    for j in range(4):
        expect = R - depth * 0.5 * (1 - np.cos(2 * math.pi * (z[late] / lam + j / 4)))
        assert np.allclose(e.elem_r_m[j, per][late], expect, atol=1e-6 * R)
    # the radius envelope over all teeth stays close to R (the crests of the teeth interleave)
    assert np.max(R - e.elem_r_m[:, per][:, late].max(axis=0)) < depth * 0.5


def test_left_hand_tool_mirrors_positions():
    right = build_tool(endmill(), HOLDER, instance())
    left_item = endmill()
    left_item["hand"] = "left"
    left = build_tool(left_item, HOLDER, instance())
    assert left.edge.hand_sign[0] == 1.0 and right.edge.hand_sign[0] == -1.0
    pr = element_positions_T(right.edge, 0.0, apply_runout=False)
    pl = element_positions_T(left.edge, 0.0, apply_runout=False)
    assert np.allclose(pr[..., 0], pl[..., 0])
    assert np.allclose(pr[..., 1], -pl[..., 1])
    assert np.allclose(right.edge.elem_rake_rad, left.edge.elem_rake_rad)
