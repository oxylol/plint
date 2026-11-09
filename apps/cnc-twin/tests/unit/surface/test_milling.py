"""Milled surfaces through SurfaceDomain: side-wall feed marks of the trochoidal tooth paths, runout marks,
floor marks of the end edges (trailing side, and the leading side of a tilted tool), ball-nose scallops of the
step-over and the effective cutting profile of inclined ball and toroidal cutters, witness marks."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.unit.surface import surffix as fx
from cnctwin.surface import faces as FC
from cnctwin.surface.protocols import ball_outline, endmill_outline
from cnctwin.surface.store import PF

R = 5e-3
FZ = 0.05e-3
RPM = 10000.0


def _r_eff(v, omega):
    return R * (1.0 + v / (R * abs(omega))) ** 2


def test_side_wall_feed_marks_follow_the_trochoid():
    _, _, state, v = fx.run_wall(fz=FZ)
    omega = 2.0 * math.pi * RPM / 60.0
    got = fx.patch_values(state)
    assert got["Pt_m"] == pytest.approx(FZ ** 2 / (8.0 * _r_eff(v, omega)), rel=0.02)
    assert got["direction"] == "feed"


@pytest.mark.parametrize("runout", [3e-6, 10e-6])
def test_runout_leaves_one_mark_per_revolution(runout):
    """A tooth standing out by more than the per-revolution cusp generates alone: marks spaced N fz with the
    height of the revolution feed (Martellotti 1941)."""
    nt = 4
    _, _, state, v = fx.run_wall(fz=FZ, nt=nt, runout=runout)
    omega = 2.0 * math.pi * RPM / 60.0
    got = fx.patch_values(state)
    f_rev = nt * FZ
    assert got["Pt_m"] == pytest.approx(f_rev ** 2 / (8.0 * _r_eff(v, omega)), rel=0.02)
    pf = state.surface.store.pf[0]
    rsm = pf[PF["f_rsm"]] / pf[PF["f_len"]]
    assert rsm == pytest.approx(f_rev, rel=0.05)


def test_small_runout_mixes_the_marks():
    """Runout below the per-revolution cusp: every tooth still touches, the marks lie between the two limits."""
    _, _, state, v = fx.run_wall(fz=FZ, runout=0.3e-6)
    omega = 2.0 * math.pi * RPM / 60.0
    pt = fx.patch_values(state)["Pt_m"]
    assert FZ ** 2 / (8.0 * _r_eff(v, omega)) < pt < (4 * FZ) ** 2 / (8.0 * _r_eff(v, omega))


def test_deflection_at_generation_moves_the_valleys():
    """u_rel along the wall normal at generation (here alternating by tooth) is in the event positions: the
    profile gains the same per-revolution pattern as runout."""
    amp = 2e-6
    _, _, state, _ = fx.run_wall(fz=FZ, u_rel=lambda q, j, k: (0.0, amp * (1 if j == 0 else 0), 0.0))
    _, _, clean, _ = fx.run_wall(fz=FZ)
    assert fx.patch_values(state)["Pt_m"] > 3.0 * fx.patch_values(clean)["Pt_m"]


def _floor_template(outline_pts):
    """The end mill's meridian as the trailing-side floor template: s = -(r - r_low), w = z - z_low."""
    r, z = outline_pts[:, 0], outline_pts[:, 1]
    low = int(np.argmin(z - 1e-12 * r))
    return -(r - r[low]), z - z[low]


def test_floor_marks_are_the_end_edge_profile_repeated_every_feed_per_tooth():
    edge, outline = fx.endmill(R, 4)
    elems = list(range(9))                           # end elements and the corner-bottom element
    _, _, state = fx.run_floor(edge, outline, elems, fz=FZ)
    s, w = _floor_template(outline.pts)
    expect, _, _ = fx.envelope_pt(s, w, np.arange(60) * FZ, lo=20 * FZ, hi=40 * FZ)
    got = fx.patch_values(state)
    assert got["Pt_m"] == pytest.approx(expect, rel=0.02)
    assert 0.3e-6 < expect < 0.7e-6                  # the dished end edge meets the corner arc


def test_axial_runout_of_one_end_tooth_marks_the_floor_once_per_revolution():
    """One end tooth 3 um lower than the others generates the floor alone: the floor marks repeat every
    revolution feed N fz (the edge profile at the revolution spacing)."""
    edge, outline = fx.endmill(R, 4)
    elems = list(range(9))
    _, _, state = fx.run_floor(edge, outline, elems, fz=FZ, axial_runout=[-3e-6, 0.0, 0.0, 0.0], teeth_passes=200)
    s, w = _floor_template(outline.pts)
    expect, _, _ = fx.envelope_pt(s, w, np.arange(40) * 4 * FZ, lo=10 * 4 * FZ, hi=30 * 4 * FZ)
    got = fx.patch_values(state)
    assert got["Pt_m"] == pytest.approx(expect, rel=0.03)
    pf = state.surface.store.pf[0]
    assert pf[PF["f_rsm"]] / pf[PF["f_len"]] == pytest.approx(4 * FZ, rel=0.05)


def test_a_tilted_tool_cuts_the_floor_with_its_leading_side():
    """The axis leaning toward the feed lowers the leading side by 2 r tau: the leading passages (mirrored
    template, half a revolution earlier) generate the floor instead of the trailing ones."""
    edge, outline = fx.endmill(R, 4)
    elems = list(range(9))
    tau = 2e-4
    # the travel must exceed the distance between the two corners for a floor point to see both sides
    _, _, state = fx.run_floor(edge, outline, elems, fz=FZ, tilt_y=tau, teeth_passes=320)
    s, w = _floor_template(outline.pts)
    lead, _, _ = fx.envelope_pt(-s, w, np.arange(60) * FZ, lo=20 * FZ, hi=40 * FZ)
    got = fx.patch_values(state)
    assert got["Pt_m"] == pytest.approx(lead, rel=0.03)


def _ball_tool(nt=2, n_ball=24):
    edge, outline = fx.ballmill(R, nt, n_ball)
    elems = list(range(4))                           # the elements near the tip
    return edge, outline, elems


def test_ball_nose_scallop_height_matches_the_geometry():
    edge, outline, elems = _ball_tool()
    step = 0.5e-3
    passes = [(k * step, 0.0, 0.0) for k in range(6)]
    dom, ctx, state = fx.run_floor(edge, outline, elems, fz=FZ, passes=passes, teeth_passes=60)
    results = [r for r in state.surface.results if "cross_patch" in r]
    assert results, "no step-over profile was evaluated"
    scallop = R - math.sqrt(R * R - step * step / 4.0)
    cross = state.surface.store.pf[results[0]["cross_patch"]]
    assert cross[PF["c_rt_max"]] == pytest.approx(scallop, rel=0.03)
    assert results[0]["track_spacing_m"] == pytest.approx(step, rel=1e-6)
    rough = FC.patch_roughness(cross)
    assert rough["direction"] == "cross"             # the scallops dominate the feed marks: lay along the feed


def test_step_over_profile_carries_the_minimum_chip_thickness():
    """Across the tracks the step-over plays the feed: a pass meets the layer the neighbouring track left, and with an
    edge radius the scallop peaks follow Brammertz's R_t = s^2 / 8 R + (h_min / 2)(1 + R h_min / s^2) instead of the
    kinematic scallop (h_min = 0.3 x 10 um in the test material)."""
    edge, outline, elems = _ball_tool()
    step, edge_r = 0.5e-3, 10e-6
    passes = [(k * step, 0.0, 0.0) for k in range(6)]
    _, _, state = fx.run_floor(edge, outline, elems, fz=FZ, passes=passes, teeth_passes=60, edge_r=edge_r)
    results = [r for r in state.surface.results if "cross_patch" in r]
    assert results, "no step-over profile was evaluated"
    h_min = 0.3 * edge_r
    brammertz = step * step / (8.0 * R) + 0.5 * h_min * (1.0 + R * h_min / (step * step))
    cross = state.surface.store.pf[results[0]["cross_patch"]]
    assert cross[PF["c_rt_max"]] == pytest.approx(brammertz, rel=0.04)
    assert cross[PF["c_rt_max"]] > 1.15 * (R - math.sqrt(R * R - step * step / 4.0))


@pytest.mark.parametrize("kind,lead", [("ball", 20.0), ("torus", 30.0)])
def test_effective_cutting_profile_of_inclined_cutters(kind, lead):
    """The step-over profile of a cutter whose axis leans by the lead angle toward the feed is its silhouette
    seen along the feed: a ball stays a circle of radius R (exactly: w = R cos(lead) - sqrt(R^2 - u^2) about
    the tip); a torus (corner r on the radius R0 = R - r) gets the effective radius r + R0 / sin(lead) at its
    lowest point (the corner-centre circle projects to an ellipse of curvature radius R0 / sin(lead) there,
    offset by r)."""
    beta = math.radians(lead)
    if kind == "ball":
        pts = ball_outline(R, 0.012)
    else:
        pts = endmill_outline(R, 1.0e-3, 0.0, 0.012)
    d = np.array([1.0, 0.0, 0.0])
    nn = np.array([0.0, 0.0, 1.0])
    tc = np.cross(nn, d)
    a = np.array([math.sin(beta), 0.0, math.cos(beta)])
    poly, gmin = FC.silhouette(pts, a, tc, nn, 1440, 2e-6)
    if kind == "ball":
        for u in (-0.4e-3, -0.2e-3, 0.0, 0.25e-3):
            exact = R * math.cos(beta) - math.sqrt(R * R - u * u)
            assert np.interp(u, poly[:, 0], poly[:, 1]) == pytest.approx(exact, abs=5e-8)   # 1440 x 10 um facets
    u0 = np.interp(0.0, poly[:, 0], poly[:, 1])
    near = np.abs(poly[:, 0]) < 0.2e-3
    c2 = np.polyfit(poly[near, 0], poly[near, 1] - u0, 2)[0]
    r_eff = 1.0 / (2.0 * c2)
    expect = R if kind == "ball" else 1.0e-3 + (R - 1.0e-3) / math.sin(beta)
    assert r_eff == pytest.approx(expect, rel=0.01)


def test_witness_mark_between_passes_of_different_height():
    """Two overlapping floor passes 3 um apart in height (1 um of it deflection) leave a 3 um witness step."""
    edge, outline = fx.endmill(R, 4)
    elems = list(range(9))
    passes = [(0.0, 0.0, 0.0), (6e-3, -3e-6, 1e-6)]
    _, _, state = fx.run_floor(edge, outline, elems, fz=FZ, passes=passes)
    marks = [m for m in state.surface.marks if m["kind"] == "witness"]
    assert len(marks) == 1
    assert marks[0]["height_m"] == pytest.approx(3e-6, rel=1e-3)
    assert marks[0]["deflection_m"] == pytest.approx(1e-6, rel=1e-6)
