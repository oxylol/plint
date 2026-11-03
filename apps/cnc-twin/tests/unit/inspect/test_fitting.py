"""Least-squares and minimum-zone fits on synthetic points against known geometry and hand examples."""

import math

import numpy as np
import pytest

from cnctwin.geom import transform as tf
from cnctwin.inspect import fitting as F
from cnctwin.inspect.schema import defaults

S = defaults()
SIGMA = 1e-6          # 1 um point noise


def _rot(axis):
    return tf.frame_from_z(axis)


def test_plane_fit_recovers_a_tilted_plane():
    rng = np.random.default_rng(1)
    n_true = tf.unit([0.02, -0.01, 1.0])
    R = _rot(n_true)
    uv = rng.uniform(-0.02, 0.02, (400, 2))
    P = np.array([0.01, 0.02, 0.005]) + uv @ R[:, :2].T + rng.normal(0, SIGMA, 400)[:, None] * n_true
    fit = F.fit_plane(P, [0, 0, 1])
    assert math.acos(min(1.0, fit.params["normal"] @ n_true)) < 4 * fit.uncertainty["tilt_rad"] + 1e-12
    assert abs((fit.params["point_m"] - [0.01, 0.02, 0.005]) @ n_true) < 4 * fit.uncertainty["offset_m"]
    assert fit.rms_m == pytest.approx(SIGMA, rel=0.15)


def test_cylinder_fit_recovers_axis_and_radius():
    rng = np.random.default_rng(2)
    u = tf.unit([0.01, 0.005, 1.0])
    R = _rot(u)
    th = rng.uniform(0, 2 * math.pi, 600)
    z = rng.uniform(0.0, 0.02, 600)
    radius = 0.004
    a = np.array([0.03, 0.02, 0.0])
    P = a + radius * (np.cos(th)[:, None] * R[:, 0] + np.sin(th)[:, None] * R[:, 1]) + z[:, None] * u
    normals = np.cos(th)[:, None] * R[:, 0] + np.sin(th)[:, None] * R[:, 1]
    P = P + rng.normal(0, SIGMA, 600)[:, None] * normals
    fit = F.fit_cylinder(P, a + [1e-4, -1e-4, 0], [0, 0, 1], 0.0041, S)
    assert fit.converged
    assert fit.params["radius_m"] == pytest.approx(radius, abs=4 * fit.uncertainty["radius_m"])
    assert math.acos(min(1.0, abs(fit.params["axis"] @ u))) < 5e-4
    w = fit.params["axis_point_m"] - a
    assert np.linalg.norm(w - (w @ u) * u) < 5 * fit.uncertainty["axis_x_m"]


def test_sphere_circle_cone_torus_fits():
    rng = np.random.default_rng(3)
    # sphere
    d = rng.normal(size=(500, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    c, r = np.array([0.01, -0.01, 0.02]), 0.006
    sph = F.fit_sphere(c + (r + rng.normal(0, SIGMA, 500))[:, None] * d, c + 2e-4, 0.0058, S)
    assert np.linalg.norm(sph.params["centre_m"] - c) < 5 * max(sph.uncertainty["centre_x_m"], 1e-9)
    assert sph.params["radius_m"] == pytest.approx(r, abs=4 * sph.uncertainty["radius_m"])
    # circle in a tilted plane
    n = tf.unit([0.0, 0.1, 1.0])
    R = _rot(n)
    th = np.linspace(0, 2 * math.pi, 90, endpoint=False)
    P = c + 0.005 * (np.cos(th)[:, None] * R[:, 0] + np.sin(th)[:, None] * R[:, 1])
    P += rng.normal(0, SIGMA, 90)[:, None] * (np.cos(th)[:, None] * R[:, 0] + np.sin(th)[:, None] * R[:, 1])
    cir = F.fit_circle(P, c + 1e-4, n, S)
    assert cir.params["radius_m"] == pytest.approx(0.005, abs=4 * cir.uncertainty["radius_m"])
    assert np.linalg.norm(cir.params["centre_m"] - c) < 1e-6
    # cone: half angle 10 deg, axis +z from the apex
    psi = math.radians(10.0)
    apex = np.array([0.0, 0.0, -0.03])
    z = rng.uniform(0.01, 0.03, 800)
    th = rng.uniform(0, 2 * math.pi, 800)
    rad = z * math.tan(psi)
    er = np.column_stack([np.cos(th), np.sin(th), np.zeros(800)])
    nrm = math.cos(psi) * er - math.sin(psi) * np.array([0, 0, 1.0])
    P = apex + rad[:, None] * er + z[:, None] * np.array([0, 0, 1.0]) + rng.normal(0, SIGMA, 800)[:, None] * nrm
    cone = F.fit_cone(P, apex + [2e-4, 0, 5e-4], [0.01, 0, 1], math.radians(10.3), S)
    assert cone.params["semi_angle_rad"] == pytest.approx(psi, abs=4 * cone.uncertainty["semi_angle_rad"])
    assert np.linalg.norm(cone.params["apex_m"] - apex) < 2e-5
    assert cone.rms_m == pytest.approx(SIGMA, rel=0.15)
    # torus R 10 mm, r 2 mm
    u = tf.unit([0.0, 0.05, 1.0])
    R = _rot(u)
    a, b = rng.uniform(0, 2 * math.pi, 900), rng.uniform(0, 2 * math.pi, 900)
    ring = np.cos(a)[:, None] * R[:, 0] + np.sin(a)[:, None] * R[:, 1]
    tube = np.cos(b)[:, None] * ring + np.sin(b)[:, None] * u
    P = c + 0.010 * ring + (0.002 + rng.normal(0, SIGMA, 900))[:, None] * tube
    tor = F.fit_torus(P, c + 1e-4, [0, 0, 1], 0.0101, 0.0019, S)
    assert tor.params["major_radius_m"] == pytest.approx(0.010, abs=4 * tor.uncertainty["major_radius_m"])
    assert tor.params["minor_radius_m"] == pytest.approx(0.002, abs=4 * tor.uncertainty["minor_radius_m"])


def test_minimum_zone_flatness_hand_example():
    """An extruded profile x = 0, 10, 20, 30 mm with heights 0, 0, 0, 10 um. The minimum zone is parallel to the
    line from the first to the last point; its width is the distance of the third point from that line,
    z3 x2 / sqrt(x3^2 + z3^2). The least-squares plane (slope 0.3 z3 / x step) leaves a range of 0.7 z3 / sqrt(1 +
    slope^2) - 5 % more."""
    x = np.array([0.0, 0.01, 0.02, 0.03])
    z = np.array([0.0, 0.0, 0.0, 1e-5])
    P = np.array([[xi, yi, zi] for yi in (0.0, 0.01) for xi, zi in zip(x, z)])
    mz = F.zone_plane(P, [0, 0, 1], S)
    expected = z[3] * x[2] / math.hypot(x[3], z[3])
    assert mz.zone_m == pytest.approx(expected, rel=1e-9)
    ls = F.fit_plane(P, [0, 0, 1])
    slope = 0.3 * z[3] / 0.01
    assert ls.zone_m == pytest.approx(0.7 * z[3] / math.sqrt(1 + slope ** 2), rel=1e-9)
    assert ls.zone_m > 1.04 * mz.zone_m
    tangent = F.tangent_plane(P, [0, 0, 1], S)
    assert np.all(tangent.residuals_m <= 1e-15)            # every point on the material side
    assert -tangent.residuals_m.min() == pytest.approx(expected, rel=1e-9)


def test_minimum_zone_straightness_and_roundness():
    s = np.array([0.0, 0.01, 0.02, 0.03])
    h = np.array([0.0, 0.0, 0.0, 1e-5])
    mz, ls, _ = F.zone_line(np.column_stack([s, h]), S)
    assert mz == pytest.approx(1e-5 * 0.02 / math.hypot(0.03, 1e-5), rel=1e-9)
    assert ls > mz
    # a three-lobed circle: the minimum zone is twice the lobe amplitude
    th = np.linspace(0, 2 * math.pi, 360, endpoint=False)
    r = 0.01 + 3e-6 * np.cos(3 * th)
    P = np.column_stack([r * np.cos(th), r * np.sin(th), np.zeros_like(th)])
    rz = F.zone_circle(P + [1e-5, 0, 0], [0, 0, 0], [0, 0, 1], "MZ", S)
    assert rz.zone_m == pytest.approx(6e-6, rel=1e-6)
    np.testing.assert_allclose(rz.params["centre_m"][:2], [1e-5, 0.0], atol=1e-10)
    # one dent: the inscribed circle is the dent's depth smaller than the circumscribed one
    r2 = np.full_like(th, 0.01)
    r2[90] -= 4e-6
    P2 = np.column_stack([r2 * np.cos(th), r2 * np.sin(th), np.zeros_like(th)])
    mic = F.zone_circle(P2, [0, 0, 0], [0, 0, 1], "MIC", S)
    mcc = F.zone_circle(P2, [0, 0, 0], [0, 0, 1], "MCC", S)
    assert mcc.params["radius_m"] == pytest.approx(0.01, abs=1e-12)
    assert mic.params["radius_m"] > 0.01 - 4e-6               # the inscribed circle moves away from the dent
    assert mic.params["radius_m"] < 0.01


def test_minimum_zone_cylindricity_of_a_bent_cylinder():
    th = np.linspace(0, 2 * math.pi, 72, endpoint=False)
    z = np.linspace(0.0, 0.02, 11)
    TH, Z = np.meshgrid(th, z)
    bow = 4e-6 * (1 - ((Z - 0.01) / 0.01) ** 2)                  # axis bent by 4 um in x
    P = np.column_stack([(0.005 * np.cos(TH) + bow).ravel(), (0.005 * np.sin(TH)).ravel(), Z.ravel()])
    mz = F.zone_cylinder(P, [0, 0, 0], [0, 0, 1], "MZ", S)
    # a straight cylinder through the bow's mid height halves the bow: zone = bow (2 x half the bow)
    assert mz.zone_m == pytest.approx(4e-6, rel=0.02)
    ls = F.fit_cylinder(P, [0, 0, 0], [0, 0, 1], 0.005, S)
    assert ls.zone_m >= mz.zone_m - 1e-12


def test_zones_linear_in_free_displacements():
    # deviations 1, 3 at x = 0, 1 with a free tilt (column x) and offset: best fit leaves zero
    d = np.array([1.0, 3.0, 5.0])
    J = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    half, _ = F.symmetric_zone(d, J)
    assert half == pytest.approx(0.0, abs=1e-12)
    width, _ = F.band_zone(d, J[:, 1:])
    assert width == pytest.approx(0.0, abs=1e-12)
    radius, xi = F.radial_zone(np.array([[1.0, 0.0], [-1.0, 0.0]]) + [0.5, 0.2],
                               np.tile(np.eye(2)[None, :, :], (2, 1, 1)), 64)
    assert radius == pytest.approx(1.0, rel=1e-9)
    np.testing.assert_allclose(xi, [-0.5, -0.2], atol=1e-9)
    c, r = F.min_enclosing_circle(np.array([[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3) / 2], [0.5, 0.3]]))
    assert r == pytest.approx(1 / math.sqrt(3), rel=1e-12)
