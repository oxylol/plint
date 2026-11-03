"""Taps (chamfer teeth sharing the thread depth along the helical path), thread mills, turning,
grooving and threading inserts, indexable cutters."""

import math

import numpy as np
import pytest

from cnctwin.core.abi import EDGE_INSERT, EDGE_NOSE, EDGE_THREAD_FLANK, EDGE_WIPER
from cnctwin.tooling import build_tool
from cnctwin.tooling.inserts import psi_on_plane, seat_angles
from tests.unit.tooling.fixtures import HOLDER, TOOLS, instance


def test_tap_chamfer_teeth_share_the_depth():
    g = build_tool(TOOLS["tap_cutting"], HOLDER, instance(stickout_mm=40))
    e = g.edge
    info = g.info
    N, P, h = 3, 1.5e-3, info["thread"]["depth"]
    L_ch, R = info["chamfer_length"], 5e-3
    r_tip = R - h
    lead_shift = P / N
    centres = []
    for j in range(N):
        for m in range(info["rows"]):
            sel = (e.elem_valid[j] == 1) & (g.element_rows == m) & (np.array(g.element_tags) == "crest")
            if sel.sum() == 0:
                continue
            z_c = info["z_first"] + m * P + j * lead_shift
            k = int(np.argmin(np.where(sel, np.abs(e.elem_z_m[j] - z_c), np.inf)))
            centres.append((e.elem_z_m[j, k], e.elem_r_m[j, k]))
    centres.sort()
    z = np.array([c[0] for c in centres])
    r = np.array([c[1] for c in centres])
    # every crest is ground down to the chamfer cone r_tip + (R - r_tip) z / L_ch
    cone = np.minimum(r_tip + (R - r_tip) * np.clip(z, 0, None) / L_ch, R)
    assert np.allclose(r, cone, atol=1e-12)
    # successive teeth along the helix (land 0, 1, 2, then the next row) rise by h / (threads x lands)
    inside = (z > 0) & (z < L_ch - lead_shift)
    steps = np.diff(r[inside])
    assert len(steps) >= 6
    assert np.allclose(steps, h / (info["chamfer_threads"] * N), rtol=1e-6)
    assert np.isclose(r.max(), R, atol=1e-12)
    # every land carries the thread shifted by lead x trailing angle / 2 pi
    flank = (e.elem_kind == EDGE_THREAD_FLANK) & (e.elem_valid == 1) & (g.element_rows[None, :] == 6)
    k = int(np.nonzero(flank[0] & flank[1] & flank[2])[0][0])
    assert np.allclose(np.diff(e.elem_z_m[:, k]), lead_shift, atol=1e-12)


def test_tap_thread_profile_polygons():
    g = build_tool(TOOLS["tap_cutting"], HOLDER, instance(stickout_mm=40))
    thread = g.thread
    assert thread is not None
    assert thread.pitch_m[0] == pytest.approx(1.5e-3)
    assert thread.lead_m[0] == pytest.approx(1.5e-3)
    areas = []
    for j in range(thread.n_pts.shape[0]):
        for m in range(thread.n_pts.shape[1]):
            n = thread.n_pts[j, m]
            if n < 3:
                continue
            pts = thread.pts_zr_m[j, m, :n]
            area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
            assert area > 0
            areas.append((thread.row_z_m[j, m], area))
    areas.sort()
    # the full tooth trapezoid: (crest + base) / 2 x depth
    c, hd = 0.125 * 1.5e-3, g.info["thread"]["depth"]
    full = (c + (c + 2 * hd * math.tan(math.radians(30)))) / 2 * hd
    assert max(a for _, a in areas) == pytest.approx(full, rel=1e-9)
    assert min(a for _, a in areas) < 0.5 * full        # chamfer teeth are truncated


def test_thread_mill_rows_and_flanks():
    g = build_tool(TOOLS["thread_mill"], HOLDER, instance())
    e = g.edge
    flank = (e.elem_kind[0] == EDGE_THREAD_FLANK) & (e.elem_valid[0] == 1)
    kappa = np.degrees(e.elem_kappa_rad[0, flank])
    assert set(np.round(kappa, 6)) == {30.0, 150.0}
    h = g.info["thread"]["depth"]
    rows = g.info["rows"]
    assert e.elem_db_m[0, flank].sum() == pytest.approx(2 * rows * h / math.cos(math.radians(30)), rel=1e-9)
    assert g.thread.lead_m[0] == 0.0


def test_turning_insert_nose_and_angles():
    g = build_tool(TOOLS["turning_insert"], None, {"number": 1})
    e = g.edge
    nose = e.elem_kind[0] == EDGE_NOSE
    r_eps = 0.8e-3
    eps = math.radians(80)
    assert e.elem_db_m[0, nose].sum() == pytest.approx(r_eps * (math.pi - eps), rel=1e-12)
    kappa = e.elem_kappa_rad[0, nose]
    assert kappa.min() > math.radians(-5) - 1e-9 and kappa.max() < math.radians(95) + 1e-9
    # imaginary tip at the origin: the nose circle touches x_T = 0 and z_T = 0
    centre_z = e.elem_z_m[0, nose] + r_eps * np.cos(kappa)
    centre_x = e.elem_r_m[0, nose] - r_eps * np.sin(kappa)
    assert np.allclose(centre_z, r_eps, atol=1e-12) and np.allclose(centre_x, -r_eps, atol=1e-12)
    assert np.all(e.elem_r_m <= 1e-15)
    major = (e.elem_kind[0] == EDGE_INSERT) & (e.elem_kappa_rad[0] < 0)
    assert np.allclose(e.elem_kappa_rad[0, major], math.radians(-5))
    # negative insert in a -6/-6 holder with a 12 deg chip breaker: rake 6, clearance 6, inclination -6
    assert np.allclose(e.elem_rake_rad[0, major], math.radians(6), atol=1e-9)
    assert np.allclose(e.elem_clear_rad[0, major], math.radians(6), atol=1e-9)
    assert np.allclose(e.elem_incl_rad[0, major], math.radians(-6), atol=1e-6)
    assert np.allclose(e.elem_lag_rad, 0.0) and np.all(e.runout_T_m == 0)
    profile = g.insert
    assert profile.cutting.sum() > 0 and profile.pts_zx_m.shape[1] == 2


def test_grooving_and_threading_inserts():
    groove = build_tool(TOOLS["grooving_insert"], None, {"number": 1}).edge
    front = (groove.elem_kind[0] == EDGE_INSERT) & np.isclose(groove.elem_kappa_rad[0], math.pi / 2)
    # the corner radii meet the side relief (3 deg): each fillet reaches r tan((90 + 3) / 2) along the front
    side = math.radians(3)
    expect = 3e-3 - 2 * 0.2e-3 * math.tan(0.5 * (math.pi / 2 + side))
    assert groove.elem_db_m[0, front].sum() == pytest.approx(expect, rel=1e-9)
    assert np.allclose(groove.elem_r_m[0, front], 0.0, atol=1e-15)
    thread = build_tool(TOOLS["threading_insert"], None, {"number": 1})
    e = thread.edge
    flank = e.elem_kind[0] == EDGE_THREAD_FLANK
    assert set(np.round(np.degrees(e.elem_kappa_rad[0, flank]), 6)) == {30.0, 150.0}
    assert e.elem_r_m.max() == pytest.approx(0.0, abs=1e-6)
    assert thread.thread.n_pts[0, 0] >= 3


def test_insert_face_model_reproduces_the_seat_angles():
    from cnctwin.tooling.edges import Face, face_angles, lift_tangent
    from cnctwin.tooling.inserts import face_planes
    kappa_r, gamma_p, gamma_f = math.radians(45), math.radians(6), math.radians(-6.5)
    gam, lam = seat_angles(gamma_p, gamma_f, kappa_r)
    # classic relations (Armarego & Brown): check against the explicit formulas
    assert math.tan(lam) == pytest.approx(math.tan(gamma_p) * math.sin(kappa_r) - math.tan(gamma_f) * math.cos(kappa_r))
    alpha_i = math.radians(7)
    N_r, _ = face_planes(kappa_r, gam, lam, alpha_i - gam)
    e_r, e_v, e_z = np.eye(3)
    t_m = math.cos(kappa_r) * e_r + math.sin(kappa_r) * e_z
    n = math.sin(kappa_r) * e_r - math.cos(kappa_r) * e_z
    t3 = lift_tangent(t_m, e_v, N_r)
    assert math.asin(-float(t3 @ e_v)) == pytest.approx(lam, abs=1e-12)
    rake, clear, _ = face_angles(Face(mode="insert", N_rake=tuple(N_r), rake=0.0, clear=alpha_i), t3, n, e_v,
                                 np.zeros(3))
    assert rake == pytest.approx(gam, abs=1e-12)
    assert clear == pytest.approx(alpha_i - gam, abs=1e-12)


def test_face_mill_inserts_on_the_rake_plane():
    g = build_tool(TOOLS["face_mill"], HOLDER, instance(stickout_mm=50))
    e = g.edge
    gam, lam = seat_angles(math.radians(6), math.radians(-6.5), math.radians(45))
    major = (e.elem_kind[0] == EDGE_INSERT) & np.isclose(e.elem_kappa_rad[0], math.radians(45), atol=1e-3)
    assert major.sum() > 3
    # a straight inclined edge: the classic angles hold at the corner and drift along the edge as
    # the edge moves around the cutter (here by -0.2 deg per mm)
    first = int(np.nonzero(major)[0][0])
    assert e.elem_incl_rad[0, first] == pytest.approx(lam, abs=3e-3)
    assert e.elem_rake_rad[0, first] == pytest.approx(gam, abs=3e-3)
    assert np.all(np.diff(e.elem_incl_rad[0, major]) < 0)
    assert np.allclose(e.tooth_phase_rad, np.radians([0, 288, 216, 144, 72]), atol=1e-12)
    wiper = e.elem_kind[0] == EDGE_WIPER
    assert wiper.sum() >= 1 and np.allclose(e.elem_z_m[0, wiper], 0.0, atol=1e-12)
    assert e.elem_z_m[:, e.elem_valid[0] == 1].min() >= -1e-12


def test_psi_on_plane_solves_the_plane():
    N = np.array([0.1, 0.98, -0.17])
    N /= np.linalg.norm(N)
    Q = np.array([0.025, 0.0, 0.0])
    z = np.array([0.0, 0.001, 0.003])
    r = np.array([0.025, 0.0252, 0.0258])
    psi = psi_on_plane(z, r, N, Q)
    pts = np.stack([r * np.cos(psi), r * np.sin(psi), z], axis=1)
    assert np.allclose((pts - Q) @ N, 0.0, atol=1e-15)


@pytest.mark.parametrize("shape, kappa_r, sharp_area_mm2", [
    ("C", 95, 12.7 ** 2 / math.sin(math.radians(80))),
    ("T", 91, math.sqrt(3) / 4 * (math.sqrt(3) * 12.7) ** 2),
    ("S", 45, 12.7 ** 2),
])
def test_insert_outline_areas(shape, kappa_r, sharp_area_mm2):
    from tests.unit.tooling.fixtures import item
    r_eps = 0.8e-3
    insert = {"shape": shape, "ic_mm": 12.7, "nose_radius_mm": 0.8, "clearance_deg": 7}
    g = build_tool(item("turning_insert", insert=insert, holder={"approach_angle_deg": kappa_r, "overhang_mm": 30}),
                   None, {"number": 1})
    p = g.insert.pts_zx_m
    area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    corner = {"C": 80, "T": 60, "S": 90}[shape]
    turn = math.radians(180 - corner)
    # sharp polygon minus the material the nose radius rounds off (r^2 (tan(t/2) - t/2)), polygonised nose
    expect = sharp_area_mm2 * 1e-6 - r_eps ** 2 * (math.tan(turn / 2) - turn / 2)
    assert area == pytest.approx(expect, rel=2e-4)
    assert g.insert.cutting.sum() > 0


def test_round_insert_is_a_full_circle():
    from tests.unit.tooling.fixtures import item
    g = build_tool(item("turning_insert", insert={"shape": "R", "ic_mm": 12.7, "nose_radius_mm": 0, "clearance_deg": 7},
                        holder={"approach_angle_deg": 90, "overhang_mm": 30}), None, {"number": 1})
    p = g.insert.pts_zx_m
    area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    assert area == pytest.approx(math.pi * 6.35e-3 ** 2, rel=2e-3)
    nose = g.edge.elem_kind[0] == EDGE_NOSE
    assert g.edge.elem_db_m[0, nose].sum() == pytest.approx(math.pi * 6.35e-3, rel=1e-12)
