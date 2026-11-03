"""Ball nose (Lee & Altintas 1996 geometry) and twist drill (Armarego / Wiriyacosol lip geometry)
against closed forms."""

import math

import numpy as np
import pytest

from cnctwin.core.abi import EDGE_CHISEL, EDGE_CORNER, EDGE_LIP, EDGE_MARGIN
from cnctwin.tooling import build_tool
from cnctwin.tooling.holemaking import chisel_angles
from cnctwin.tooling.schema import ToolingSettings
from tests.unit.tooling.fixtures import HOLDER, TOOLS, endmill, instance, item

R = 5e-3


def _ball():
    return build_tool(endmill("ball_endmill", corner_radius_mm=5, flutes=2), HOLDER, instance())


def test_ball_chip_width_and_projections():
    e = _ball().edge
    ball = e.elem_kind[0] == EDGE_CORNER
    kappa = e.elem_kappa_rad[0, ball]
    db = e.elem_db_m[0, ball]
    # meridian arc of the quarter ball, exact
    assert db.sum() == pytest.approx(0.5 * math.pi * R, rel=1e-12)
    # projected widths (Lee & Altintas db = dz / sin(kappa)): axial and radial extent of the ball;
    # midpoint rule with kappa steps <= 7.5 deg: relative error < dk^2 / 24
    dk = ToolingSettings().max_kappa_step_rad
    assert (db * np.sin(kappa)).sum() == pytest.approx(R, rel=dk ** 2 / 24)
    assert (db * np.cos(kappa)).sum() == pytest.approx(R, rel=dk ** 2 / 24)
    # element radius and height on the sphere
    assert np.allclose(e.elem_r_m[0, ball], R * np.sin(kappa), atol=1e-12)
    assert np.allclose(e.elem_z_m[0, ball], R * (1 - np.cos(kappa)), atol=1e-12)


def test_ball_local_helix_rake_and_tip_speed():
    e = _ball().edge
    ball = e.elem_kind[0] == EDGE_CORNER
    kappa = e.elem_kappa_rad[0, ball]
    r = e.elem_r_m[0, ball]
    beta0 = math.radians(35)
    # constant lead: lag = -z tan(i0) / R0 (Lee & Altintas), true inclination tan(l) = r tan(i0) sin(kappa) / R0
    assert np.allclose(e.elem_lag_rad[0, ball], -e.elem_z_m[0, ball] * math.tan(beta0) / R, atol=1e-12)
    lam = np.arctan(r * math.tan(beta0) * np.sin(kappa) / R)
    assert np.allclose(e.elem_incl_rad[0, ball], lam, atol=2e-4)
    # local normal rake from the constant radial rake (chord tangents: agreement to second order)
    assert np.allclose(np.tan(e.elem_rake_rad[0, ball]), math.tan(math.radians(8)) * np.cos(e.elem_incl_rad[0, ball]),
                       atol=1e-6)
    # the element next to the tip has a small radius: near-zero cutting speed handled by v_e = omega x r_e
    assert r.min() < 0.05 * R and r.min() > 0.0


def _drill(**changes):
    tool = TOOLS["twist_drill"]
    tool = {**tool, "geometry": {**tool["geometry"], **changes}}
    return build_tool(tool, HOLDER, instance(stickout_mm=70))


def test_drill_lip_length_and_chisel():
    g = _drill()
    e = g.edge
    p = math.radians(59)
    psi = math.radians(125)
    w = 0.75e-3
    x_c = -w / math.tan(psi)
    x_R = math.sqrt(R * R - w * w)
    lip = e.elem_kind[0] == EDGE_LIP
    chisel = e.elem_kind[0] == EDGE_CHISEL
    # the lip is a straight edge from the chisel corner to the outer corner
    assert e.elem_len_m[0, lip].sum() == pytest.approx((x_R - x_c) / math.sin(p), rel=1e-9)
    # chisel half from the axis to the corner r_c = w / sin(psi); chisel length 2 r_c
    r_c = w / math.sin(psi)
    assert e.elem_db_m[0, chisel].sum() == pytest.approx(r_c, rel=1e-12)
    assert g.info["chisel_length"] == pytest.approx(2 * r_c)
    assert np.allclose(e.elem_kappa_rad[0, chisel], 0.0)
    assert np.allclose(e.elem_incl_rad[0, chisel], 0.0, atol=1e-12)
    rake_ch, clear_ch, _ = chisel_angles(p, psi)
    assert np.allclose(e.elem_rake_rad[0, chisel], rake_ch)
    tan_ac = 1 / (math.tan(math.pi - psi) * math.tan(p))
    assert rake_ch == pytest.approx(math.atan(math.sqrt(1 / math.tan(p) ** 2 + tan_ac ** 2)) - math.pi / 2)
    assert -math.radians(65) < rake_ch < -math.radians(45)
    # outer corner height and point
    assert g.info["point_angle"] == pytest.approx(2 * p)
    assert e.elem_z_m[0, lip].max() < (x_R - x_c) / math.tan(p)


def test_drill_lip_lag_inclination_rake():
    g = _drill()
    e = g.edge
    p = math.radians(59)
    w = 0.75e-3
    lip = e.elem_kind[0] == EDGE_LIP
    r = e.elem_r_m[0, lip]
    assert np.allclose(e.elem_lag_rad[0, lip], np.arcsin(w / r) - math.asin(w / R), atol=1e-12)
    assert np.allclose(np.sin(e.elem_incl_rad[0, lip]), w / r * math.sin(p), atol=2e-3)
    # rake falls toward the centre (positive at the periphery, negative near the chisel)
    rake = e.elem_rake_rad[0, lip]
    assert rake[-1] > rake[0] and rake[-1] > 0 and rake[0] < 0


def test_drill_rake_tends_to_closed_form_for_thin_web():
    g = _drill(web_thickness_mm=0.001)
    e = g.edge
    lip = e.elem_kind[0] == EDGE_LIP
    r = e.elem_r_m[0, lip]
    beta_r = np.arctan(r / R * math.tan(math.radians(30)))
    expect = np.arctan(np.tan(beta_r) / math.sin(math.radians(59)))
    outer = r > 0.3 * R
    assert np.allclose(e.elem_rake_rad[0, lip][outer], expect[outer], atol=2e-3)


def test_drill_margin_back_taper_and_lag():
    g = _drill()
    e = g.edge
    margin = e.elem_kind[0] == EDGE_MARGIN
    assert np.allclose(e.elem_db_m[0, margin], 0.7e-3)
    assert np.all(np.diff(e.elem_r_m[0, margin]) < 0)          # back taper
    z0 = g.info["head_length"]
    assert e.elem_z_m[0, margin].max() < z0
    assert np.allclose(e.elem_rake_rad[0, margin], 0.0) and np.allclose(e.elem_clear_rad[0, margin], 0.0)
    two = e.tooth_phase_rad
    assert np.allclose(np.sort(two), [0.0, math.pi])


def test_spot_and_centre_drill_kinds():
    spot = build_tool(TOOLS["spot_drill"], HOLDER, instance(stickout_mm=40))
    kinds = set(spot.edge.elem_kind[spot.edge.elem_valid == 1].tolist())
    assert kinds == {EDGE_LIP, EDGE_CHISEL}
    centre = build_tool(TOOLS["centre_drill"], HOLDER, instance(stickout_mm=30))
    e = centre.edge
    sink = (e.elem_kind[0] == EDGE_LIP) & (e.elem_r_m[0] > 1.25e-3 + 1e-9)
    assert np.allclose(e.elem_kappa_rad[0, sink], math.radians(60), atol=1e-9)    # 60 deg countersink
    assert e.elem_r_m[0].max() <= 3.15e-3 + 1e-12


def test_gun_drill_point_and_pads():
    g = build_tool(TOOLS["gun_drill"], HOLDER, instance(stickout_mm=300))
    e = g.edge
    assert e.elem_z_m.shape[0] == 3               # cutting edge + two pads
    cutting = e.elem_valid[0] == 1
    assert set(e.elem_kind[0, cutting].tolist()) == {EDGE_LIP}
    for j in (1, 2):
        assert set(e.elem_kind[j, e.elem_valid[j] == 1].tolist()) == {EDGE_MARGIN}
    # the point (lowest element) lies near the point offset D/4, not on the axis
    k_low = int(np.argmin(np.where(cutting, e.elem_z_m[0], np.inf)))
    assert abs(e.elem_r_m[0, k_low] - g.info["point_offset"]) < 0.5e-3


def item_reamer():
    return item("reamer", diameter_mm=10, flutes=6, cutting_length_mm=30, overall_length_mm=120, rake_deg=0,
                clearance_deg={"primary": 8}, shank={"diameter_mm": 10}, corner_chamfer_mm=0.4, tip_angle_deg=90)


def test_reamer_chamfer_lead():
    g = build_tool(item_reamer(), HOLDER, instance(stickout_mm=60))
    e = g.edge
    lead = (e.elem_kind[0] != EDGE_MARGIN) & (e.elem_valid[0] == 1)
    assert np.allclose(e.elem_kappa_rad[0, lead], math.radians(45), atol=1e-9)
    assert e.elem_db_m[0, lead].sum() == pytest.approx(0.4e-3 / math.cos(math.radians(45)), rel=1e-9)
