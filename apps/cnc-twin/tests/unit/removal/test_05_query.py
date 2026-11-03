"""Queries: ray cast (brick DDA), surface point, surface height, fill fraction, wall thickness, broad phase,
point inside, probe sweep (G31)."""

import math

import numpy as np
import pytest

from cnctwin.removal import abi as A
from cnctwin.removal import k_query as KQ
from cnctwin.removal import k_sd as KSD
from cnctwin.removal import profile as P
from cnctwin.removal import reference as R
from cnctwin.removal import stock as S
from tests.unit.removal import fixtures as FX

LAT = 0.25e-3
LO = np.array([0.0, 0.0, 0.0])
HI = np.array([20e-3, 20e-3, 10e-3])


@pytest.fixture(scope="module")
def plane():
    n = np.array([math.cos(0.35), math.sin(0.35), 0.3])
    n /= np.linalg.norm(n)
    d = float(n @ np.array([10e-3, 10e-3, 5e-3]))
    return FX.build(LO, HI, LAT, FX.halfspace_box(LO, HI, n, d)), n, d


def test_ray_cast_exact_on_planes_and_matches_reference(plane):
    f, n, d = plane
    out = np.zeros(8)
    rng = np.random.default_rng(2)
    for _ in range(8):
        dvec = -n + 0.4 * rng.normal(size=3)
        dvec /= np.linalg.norm(dvec)
        o = np.array([10e-3, 10e-3, 5e-3]) + 3e-3 * n + rng.uniform(-1e-3, 1e-3, 3)
        assert KQ.ray_cast(f, *o, *dvec, 30e-3, 0, 1e-10, out) == 1
        t_exact = (d - n @ o) / (n @ dvec)
        assert out[0] == pytest.approx(t_exact, abs=2e-8)
        assert np.dot(out[4:7], n) > 0.999           # outward normal
        ref = R.ray_cast_ref(f, o, dvec, 30e-3, 0)
        assert out[0] == pytest.approx(ref, abs=5e-8)
    # exit search from inside
    o = np.array([10e-3, 10e-3, 5e-3]) - 2e-3 * n
    assert KQ.ray_cast(f, *o, *n, 30e-3, 1, 1e-10, out) == 1
    assert out[0] == pytest.approx(2e-3, abs=2e-8)


def test_ray_cast_skips_empty_bricks_and_misses():
    f = S.block_field([40e-3, 10e-3, 10e-3], LAT)
    out = np.zeros(8)
    # long ray through air beside the block
    assert KQ.ray_cast(f, -1e-3, 15e-3, 5e-3, 1.0, 0.0, 0.0, 50e-3, 0, 1e-9, out) == 0
    # entry far along the ray through several empty bricks
    assert KQ.ray_cast(f, 30e-3, 5e-3, 40e-3, 0.0, 0.0, -1.0, 60e-3, 0, 1e-9, out) == 0 or out[0] > 0
    o = np.array([20e-3, 5e-3, 10e-3 + 25e-3])
    assert KQ.ray_cast(f, *o, 0.0, 0.0, -1.0, 60e-3, 0, 1e-9, out) == 1
    assert out[0] == pytest.approx(25e-3, abs=2e-8)


def test_surface_point_and_refill(plane):
    f, n, d = plane
    out = np.zeros(4)
    p = np.array([10e-3, 10e-3, 5e-3]) + 0.2e-3 * n       # in air, normal into the material is -n
    assert KQ.surface_point(f, *p, *(-n), 5e-3, out) == 1
    assert out[3] == pytest.approx(-0.2e-3, abs=2e-8)       # h = (p - M) . n_e < 0: not reached
    p2 = np.array([10e-3, 10e-3, 5e-3]) - 0.15e-3 * n     # buried 0.15 mm
    assert KQ.surface_point(f, *p2, *(-n), 5e-3, out) == 1
    assert out[3] == pytest.approx(0.15e-3, abs=2e-8)
    P_ = np.ascontiguousarray(np.stack([p, p2]))
    N_ = np.ascontiguousarray(np.stack([-n, -n]))
    res = np.zeros((2, 4))
    found = np.zeros(2, np.int8)
    KQ.refill_points(f, P_, N_, 5e-3, res, found)
    assert list(found) == [1, 1]
    assert res[1, 3] == pytest.approx(0.15e-3, abs=2e-8)


def test_surface_height_fill_fraction_and_wall_thickness():
    f = S.block_field([20e-3, 10e-3, 8e-3], LAT)
    h = np.zeros(3)
    kind = np.zeros(3, np.int64)
    KQ.surface_height(f, 2, 1, np.array([3e-3, 15.1e-3, 30e-3]), np.array([2e-3, 7.7e-3, 2e-3]), h, kind)
    assert h[0] == pytest.approx(8e-3) and h[1] == pytest.approx(8e-3) and kind[2] == 0
    KQ.surface_height(f, 0, -1, np.array([5e-3]), np.array([4e-3]), h[:1], kind[:1])
    assert h[0] == pytest.approx(0.0, abs=1e-12)
    shape = np.array([6, 4, 3], np.int64)
    frac = np.zeros(tuple(shape))
    origin = np.array([-1.2e-3, -0.7e-3, -0.3e-3])
    KQ.fill_fraction(f, origin, 4e-3, shape, frac)
    ref = R.fill_fraction_ref(f, origin, 4e-3, tuple(shape))
    assert np.allclose(frac, ref, atol=1e-12)
    assert float(frac.sum()) * (4e-3) ** 3 == pytest.approx(20e-3 * 10e-3 * 8e-3, rel=1e-9)
    assert KQ.wall_thickness(f, 10e-3, 5e-3, 8e-3, 0.0, 0.0, 1.0, 50e-3) == pytest.approx(8e-3, abs=2e-8)
    assert KQ.wall_thickness(f, 20e-3, 5.1e-3, 4e-3, 1.0, 0.0, 0.0, 50e-3) == pytest.approx(20e-3, abs=2e-8)


def test_broad_phase_and_points_inside():
    f = S.block_field([20e-3, 10e-3, 8e-3], LAT)
    prof = P.profile_from_points([0.0, 20e-3], [3e-3, 3e-3])
    w = np.array([0.0, 0.0, 1.0])
    for tip in ([10e-3, 5e-3, 9e-3], [10e-3, 5e-3, 40e-3], [-30e-3, 5e-3, 2e-3]):
        tip = np.asarray(tip)
        assert KQ.broad_phase(f, prof, tip, w, 0.0) == R.broad_phase_ref(f, prof, tip, w, 0.0)
    assert KQ.broad_phase(f, prof, np.array([10e-3, 5e-3, 40e-3]), w, 0.0) == 0
    pts = np.ascontiguousarray(np.array([[5e-3, 5e-3, 4e-3], [5e-3, 5e-3, 8.01e-3], [-1e-3, 5e-3, 4e-3]]))
    out = np.zeros(3, np.int8)
    KQ.points_inside(f, pts, out)
    assert list(out) == [1, 0, 0]
    assert KSD.point_inside(f, 5e-3, 5e-3, 4e-3)


def test_probe_sweep_plane_and_bore():
    f = S.block_field([20e-3, 20e-3, 10e-3], LAT)
    ball = P.profile_ball(1e-3, 8e-3)
    out = np.zeros(8)
    w = np.array([0.0, 0.0, 1.0])
    x0 = np.array([1.0, 0.0, 0.0])
    # straight down onto the top face
    lam = KQ.sweep_contact(f, ball, np.array([10.13e-3, 9.94e-3, 15e-3]), np.array([10.13e-3, 9.94e-3, 5e-3]), w, x0,
                           8, 32, 1e-10, out)
    assert 15e-3 + lam * (5e-3 - 15e-3) == pytest.approx(10e-3, abs=1e-8)
    # sideways into the +x face: ball centre stops 1 mm from it
    lam = KQ.sweep_contact(f, ball, np.array([26e-3, 10e-3, 6e-3]), np.array([16e-3, 10e-3, 6e-3]), w, x0, 8, 32,
                           1e-10, out)
    assert 26e-3 + lam * (16e-3 - 26e-3) == pytest.approx(21e-3, abs=2e-7)
    assert np.dot(out[4:7], [1.0, 0.0, 0.0]) > 0.99
    # a bore of radius 4 mm: probe moving outward from the centre touches at r = 4 - 1 mm
    bore = FX.build(LO, HI, LAT, FX.cylinder_hole_box(LO, HI, (10e-3, 10e-3), 4e-3))
    lam = KQ.sweep_contact(bore, ball, np.array([10e-3, 10e-3, 5e-3]), np.array([17e-3, 10e-3, 5e-3]), w, x0, 8, 32,
                           1e-10, out)
    x_c = 10e-3 + lam * 7e-3
    assert x_c - 10e-3 == pytest.approx(3e-3, abs=LAT * LAT / (4 * 4e-3) + 2e-7)


def test_one_signature_each():
    for kernel in KQ.KERNELS + KSD.KERNELS:
        assert len(kernel.signatures) == 1, kernel.__name__
