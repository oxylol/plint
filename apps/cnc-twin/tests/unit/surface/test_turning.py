"""Turned surfaces through SurfaceDomain: nose-radius feed marks (f^2 / 8 r), the minimum-chip-thickness term
(Brammertz 1961), published measured Ra (Cirstoiu 2024), flank-wear changes of the generating outline, BUE."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from tests.unit.surface import surffix as fx
from cnctwin.core.abi import EDGE_NOSE
from cnctwin.surface.protocols import ToolOutline, turning_insert_outline

NOSE = 0.8e-3


@pytest.mark.parametrize("f", [0.1e-3, 0.2e-3, 0.3e-3])
def test_feed_marks_follow_the_nose_radius(f):
    _, _, state = fx.run_turning(f)
    v = fx.patch_values(state)
    exact = NOSE - math.sqrt(NOSE * NOSE - f * f / 4.0)
    assert v["Pt_m"] == pytest.approx(exact, rel=0.01)                     # kinematic peak-to-valley
    assert v["Pt_m"] == pytest.approx(f * f / (8.0 * NOSE), rel=0.015)     # the parabolic approximation
    # ISO 4287 Ra of the filtered profile vs the arithmetic mean of the parabolic profile f^2 / (18 sqrt 3 r)
    assert v["Ra_m"] == pytest.approx(f * f / (18.0 * math.sqrt(3.0) * NOSE), rel=0.03)
    assert v["direction"] == "feed" and v["short"] == 0


@pytest.mark.parametrize("f", [0.125e-3, 0.208e-3, 0.353e-3])
def test_minimum_chip_thickness_term_matches_brammertz(f):
    edge_r = 25e-6
    _, _, state = fx.run_turning(f, edge_r=edge_r, revolutions=60,
                                 physics={"half_spacings": 6.0})
    h_min = 0.3 * edge_r
    brammertz = f * f / (8.0 * NOSE) + 0.5 * h_min * (1.0 + NOSE * h_min / (f * f))
    _, _, sharp = fx.run_turning(f, edge_r=0.0, revolutions=60)
    assert fx.patch_values(state)["Pt_m"] == pytest.approx(brammertz, rel=0.03)
    assert fx.patch_values(state)["Pt_m"] > fx.patch_values(sharp)["Pt_m"]


# Cirstoiu 2024 Table 2: dry turning of 42MoCr11, TNMG 220408 (r 0.8 mm, kappa_r 93 deg, kappa_r' 27 deg); the
# measured Ra at each feed (mean of the repeated rows)
MEASURED = {0.125e-3: [1.90], 0.167e-3: [2.25, 2.12, 2.62, 2.30], 0.208e-3: [2.20, 2.18, 2.20, 2.25],
            0.250e-3: [3.50, 3.37, 3.37, 3.25], 0.353e-3: [4.25]}


def test_published_measured_Ra_of_turning():
    """Simulated Ra within +-30 % of every measured feed (edge radius 25 um, h_min = 0.3 r_beta: a honed P15
    insert), and the low-feed rise above the kinematic value the measurements show."""
    errors = []
    for f, meas in sorted(MEASURED.items()):
        _, _, state = fx.run_turning(f, kr=math.radians(93.0), kr2=math.radians(27.0), edge_r=25e-6, revolutions=60,
                                     physics={"half_spacings": 6.0})
        ra = fx.patch_values(state)["Ra_m"] * 1e6
        m = float(np.mean(meas))
        errors.append(ra / m - 1.0)
        assert abs(ra / m - 1.0) < 0.30, (f, ra, m)
    assert float(np.mean(np.abs(errors))) < 0.20
    _, _, low = fx.run_turning(0.125e-3, edge_r=25e-6, revolutions=60, physics={"half_spacings": 6.0})
    kinematic = (0.125e-3) ** 2 / (18.0 * math.sqrt(3.0) * NOSE)
    assert fx.patch_values(low)["Ra_m"] > 2.0 * kinematic


def _multi_element_insert(n=7):
    """Insert whose nose carries n elements on the arc (the deepest one, index n // 2, generates)."""
    kr, kr2 = math.radians(93.0), math.radians(27.0)
    a = np.sort(np.concatenate([np.linspace(kr2, -kr, n + 2)[1:-1], [0.0]]))[::-1]
    x = -NOSE + NOSE * np.cos(a)
    z = NOSE + NOSE * np.sin(a)
    kappa = math.pi / 2.0 - a
    edge = fx.edge_from(1, x, z, kappa, [EDGE_NOSE] * len(a), edge_radius=0.0, phase=[0.0])
    outline = ToolOutline(False, turning_insert_outline(NOSE, kr, kr2), 1, NOSE, 0.0, "turning_insert",
                          kappa_major_rad=kr)
    gen = int(np.argmin(np.abs(a)))
    return edge, outline, a, gen


def _worn_envelope_pt(outline_pts, elem_x, elem_z, wear_x, gen, f, n=40):
    """Brute-force kinematic envelope of the worn outline (independent of the surface code)."""
    seg = np.linalg.norm(np.diff(outline_pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    pos = np.array([arc[np.argmin(np.hypot(outline_pts[:, 0] - x, outline_pts[:, 1] - z))]
                    for x, z in zip(elem_x, elem_z)])
    order = np.argsort(pos)
    worn = outline_pts.copy()
    worn[:, 0] -= np.interp(arc, pos[order], wear_x[order])
    ex, ez = elem_x[gen] - wear_x[gen], elem_z[gen]
    # profile coordinate along the feed (-z), height = -(x - ex) (away from the material)
    s = -(worn[:, 1] - ez)
    w = -(worn[:, 0] - ex)
    grid = np.arange(10 * f, 30 * f, 2.5e-7)
    best = np.full(len(grid), np.inf)
    order_s = np.argsort(s)
    for k in range(n):
        best = np.minimum(best, np.interp(grid - k * f, s[order_s], w[order_s], left=np.inf, right=np.inf))
    return float(np.max(best) - np.min(best))


def test_uniform_wear_shifts_the_outline_and_local_wear_reshapes_it():
    edge, outline, a, gen = _multi_element_insert()
    f = 0.2e-3
    shape = np.shape(edge.elem_r_m)
    base_state = fx.run_turning(f, edge=edge, outline=outline, elem=gen)[2]
    base = fx.patch_values(base_state)["Pt_m"]
    uniform = SimpleNamespace(radial_wear_m=np.full(shape, 8e-6), axial_wear_m=np.zeros(shape),
                              bue_height_m=np.zeros(shape))
    shifted = fx.patch_values(fx.run_turning(f, edge=edge, outline=outline, wear={0: uniform}, elem=gen)[2])["Pt_m"]
    assert shifted == pytest.approx(base, rel=1e-3)        # the same shape (window resampled 8 um away)
    local = np.zeros(shape)
    trailing = [i for i in range(len(a)) if a[i] > 0.0]
    local[0, trailing[-1]] = 6e-6                       # the trailing element next to the deepest point
    worn = SimpleNamespace(radial_wear_m=local, axial_wear_m=np.zeros(shape), bue_height_m=np.zeros(shape))
    got = fx.patch_values(fx.run_turning(f, edge=edge, outline=outline, wear={0: worn}, elem=gen)[2])["Pt_m"]
    expect = _worn_envelope_pt(outline.pts, np.asarray(edge.elem_r_m)[0], np.asarray(edge.elem_z_m)[0], local[0], gen, f)
    assert got == pytest.approx(expect, rel=0.01)
    assert abs(got - base) > 0.2e-6                     # the worn trailing side changes the feed marks


def test_built_up_edge_roughens_the_surface_deterministically():
    f = 0.2e-3
    shape = (1, 1)
    bue = SimpleNamespace(radial_wear_m=np.zeros(shape), axial_wear_m=np.zeros(shape),
                          bue_height_m=np.full(shape, 20e-6))
    clean = fx.patch_values(fx.run_turning(f)[2])["Pt_m"]
    a = fx.patch_values(fx.run_turning(f, wear={0: bue}, seed=5)[2])["Pt_m"]
    b = fx.patch_values(fx.run_turning(f, wear={0: bue}, seed=5)[2])["Pt_m"]
    c = fx.patch_values(fx.run_turning(f, wear={0: bue}, seed=6)[2])["Pt_m"]
    assert a == b
    assert a != c
    assert a > clean + 2e-6
    assert a < clean + 0.5 * 20e-6 + 0.3 * 20e-6 + 1e-6    # bounded by the overcut and fragment heights
