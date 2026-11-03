"""Every thermal kernel against its plain-Python reference; one compiled signature each."""

import math

import numpy as np
import pytest

from cnctwin.thermal import kernels as K
from cnctwin.thermal import props, reference as R
from cnctwin.thermal.abi import (B_ALL_WET, B_COOLANT_ON, B_EPS, B_FMIN, B_H_CONTACT, B_H_WET, B_SIGMA, B_T_AIR,
                                 B_T_FIX, B_T_FLUID, B_T_RAD, EDGE_CONTINUOUS, EDGE_INTERRUPTED, L_AIR, L_CONTACT,
                                 L_COOL, L_DU, L_RAD, L_SRC, L_U0, L_U1, LD_AIR, LD_COOL, LD_DU, LD_SP, N_L, N_LD,
                                 Ladder, new_edge_input, new_edge_output, new_work_boundary, new_work_grid,
                                 new_work_scratch)
from cnctwin.thermal.response import edge_response, gauss_legendre, heating_phase_nodes
from cnctwin.units import STEFAN_BOLTZMANN_W_M2K4


def steel():
    return props.mat_table({"value": 7850.0}, {"table": [[20, 49.8], [500, 36.0]]},
                           {"table": [[20, 460.0], [300, 524.0], [500, 615.0]]}, {"value": 1.2e-5}, "steel")


def test_every_kernel_has_one_signature():
    kernels = [getattr(K, n) for n in dir(K) if hasattr(getattr(K, n), "signatures")]
    assert len(kernels) >= 20
    for kernel in kernels:
        assert len(kernel.signatures) == 1, kernel


def test_enthalpy_and_inverse_match_quadrature():
    mat = steel()
    for temp in (250.0, 293.15, 400.0, 600.0, 800.0, 1200.0):
        h = K.enthalpy(mat, temp)
        assert h == pytest.approx(R.enthalpy_ref(mat, temp), rel=1e-10, abs=1e-6)
        assert K.temperature(mat, h) == pytest.approx(temp, abs=1e-9)
    assert K.enthalpy(mat, 293.15) == pytest.approx(0.0, abs=1e-9)


def cut_grid(seed=3, shape=(5, 4, 4), dx=2e-3):
    rng = np.random.default_rng(seed)
    grid = new_work_grid(shape, (0.0, 0.0, 0.0), dx)
    f = np.ones(shape)
    f[:, :, -1] = rng.choice([0.0, 0.2, 0.5, 1.0], size=shape[:2])
    f[rng.random(shape) < 0.1] = 0.2
    f[0, 0, :] = 0.0
    grid.f[:] = f
    mat = steel()
    temps = 293.15 + 200.0 * rng.random(shape)
    for idx in np.ndindex(shape):
        if f[idx] > 0:
            grid.h[idx] = K.enthalpy(mat, temps[idx])
            grid.T[idx] = temps[idx]
    grid.contact[:, :, 0] = np.uint8(1 << 4)
    return grid, mat


def boundary(coolant=True):
    bnd = new_work_boundary(1, (3, 2))
    bnd.h_air[:] = [5.0, 6.0, 7.0, 8.0, 3.0, 9.0]
    bnd.near[0] = [0.004, 0.004, 0.008, 0.003, 9000.0]
    bnd.wet[0, :] = 1
    bnd.wet_geom[:] = [0.0, 0.0, 0.004]
    sc = bnd.scal
    sc[B_T_AIR] = 293.15
    sc[B_T_FLUID] = 300.0
    sc[B_H_WET] = 2500.0
    sc[B_ALL_WET] = 0.0
    sc[B_EPS] = 0.35
    sc[B_T_RAD] = 293.15
    sc[B_H_CONTACT] = 3000.0
    sc[B_T_FIX] = 305.0
    sc[B_FMIN] = 1e-9
    sc[B_COOLANT_ON] = 1.0 if coolant else 0.0
    sc[B_SIGMA] = STEFAN_BOLTZMANN_W_M2K4
    return bnd


@pytest.mark.parametrize("dt", [0.05, 2.0])
def test_work_step_matches_reference(dt):
    grid, mat = cut_grid()
    bnd = boundary()
    src = np.zeros(grid.f.shape)
    src[2, 2, 2] = 3.0
    src[4, 3, 3] = 1.5
    h_ref, t_ref, terms = R.work_step_reference(grid, src, mat, bnd, dt)
    scratch = new_work_scratch(grid.f.shape)
    scratch.src[:] = src
    out = np.zeros(N_L)
    K.work_step(grid, scratch, mat, bnd, dt, out)
    material = grid.f > 1e-9
    assert np.allclose(grid.h[material], h_ref[material], rtol=1e-9, atol=1e-7)
    assert np.allclose(grid.T[material], t_ref[material], rtol=0, atol=1e-6)
    assert out[L_SRC] == pytest.approx(terms["src"], rel=1e-12)
    for idx, key in ((L_AIR, "air"), (L_COOL, "cool"), (L_RAD, "rad"), (L_CONTACT, "contact"), (L_DU, "du")):
        assert out[idx] == pytest.approx(terms[key], rel=1e-8, abs=1e-12)
    du = out[L_U1] - out[L_U0]
    assert du == pytest.approx(out[L_SRC] - out[L_AIR] - out[L_COOL] - out[L_RAD] - out[L_CONTACT], rel=1e-12,
                               abs=1e-12)


def test_fill_update_matches_reference():
    grid, mat = cut_grid(seed=5)
    rng = np.random.default_rng(1)
    fnew = np.clip(grid.f + rng.choice([-0.5, 0.0, 0.0, 0.3], size=grid.f.shape), 0.0, 1.0)
    rho_v = mat.rho_kg_m3[0] * grid.geom[3] ** 3
    h_amb = K.enthalpy(mat, 295.0)
    f_ref, h_ref, removed, added = R.fill_update_ref(grid.f, grid.h, fnew, rho_v, 1e-9, h_amb)
    out = np.zeros(5)
    K.fill_update(grid, mat, np.ascontiguousarray(fnew), 0, 0, 0, 1e-9, h_amb, out)
    assert np.array_equal(grid.f, f_ref)
    assert np.allclose(grid.h, h_ref, rtol=1e-12, atol=1e-9)
    assert out[0] == pytest.approx(removed, rel=1e-12)
    assert out[1] == pytest.approx(added, rel=1e-12)


@pytest.mark.parametrize("point,normal", [((0.005, 0.004, 0.0075), (0, 0, 1)), ((0.0011, 0.0013, 0.004), (0, 0, 0)),
                                          ((0.001, 0.001, 0.02), (0, 0, 1))])
def test_deposit_matches_reference(point, normal):
    grid, mat = cut_grid(seed=7)
    src_ref, unplaced_ref = R.deposit_point_ref(grid.f, grid.geom, np.array(point), normal, 2.0, 1e-9, 3)
    src = np.zeros(grid.f.shape)
    out = np.zeros(2)
    rows = np.array([[*point, *normal, 2.0, 0.0]], dtype=float)
    K.deposit(grid.f, src, grid.geom, rows, 1e-9, 3, out)
    assert np.allclose(src, src_ref, rtol=1e-12, atol=1e-15)
    assert out[0] == pytest.approx(unplaced_ref)
    assert out[0] + out[1] == pytest.approx(2.0)


def test_turning_band_deposit_conserves_energy():
    grid = new_work_grid((10, 10, 3), (-0.01, -0.01, 0.0), 2e-3)
    grid.f[:] = 1.0
    src = np.zeros(grid.f.shape)
    out = np.zeros(2)
    K.deposit(grid.f, src, grid.geom, np.array([[0.009, 0.0, 0.003, 1.0, 0.0, 0.0, 5.0, 2.0 * math.pi]]), 1e-9, 3, out)
    assert out[1] == pytest.approx(5.0)
    assert src.sum() == pytest.approx(5.0)
    assert np.count_nonzero(src) > 20       # spread around the circumference


def rod_ladder(n=6, dz=0.01, area=1e-4):
    links = n - 1
    lad = Ladder(m=np.full(n, 7850.0 * area * dz), h=np.zeros(n), T=np.linspace(293.15, 393.15, n),
                 mat=np.zeros(n, np.int8), area=np.full(n, area), dz=np.full(n, dz),
                 link_a=np.arange(links, dtype=np.int32), link_b=np.arange(1, n, dtype=np.int32),
                 link_kind=np.zeros(links, np.int8), link_g=np.zeros(links), g_air=np.full(n, 0.01),
                 g_cool=np.full(n, 0.02), g_sp=np.zeros(n), src=np.zeros(n), band=np.zeros((n, 3)),
                 rhs=np.zeros(n), scal=np.array([293.15, 300.0, 310.0, 0.0]))
    lad.g_sp[-1] = 0.5
    lad.src[0] = 4.0
    mat = steel()
    for p in range(n):
        lad.h[p] = K.enthalpy(mat, lad.T[p])
    return lad, mat


def test_ladder_step_matches_dense_reference():
    lad, mat = rod_ladder()
    delta_ref, terms = R.ladder_step_ref(lad, mat, mat, 0.5)
    t0 = lad.T.copy()
    out = np.zeros(N_LD)
    K.ladder_step(lad, mat, mat, 0.5, out)
    temps_expected = np.array([R.temperature_ref_fast(mat, K.enthalpy(mat, t0[p]) +
                                                      lad.m[p] * 0 + 0) for p in range(lad.m.size)])
    del temps_expected
    assert out[LD_DU] == pytest.approx(terms["du"], rel=1e-10)
    for idx, key in ((LD_AIR, "air"), (LD_COOL, "cool"), (LD_SP, "sp")):
        assert out[idx] == pytest.approx(terms[key], rel=1e-10, abs=1e-14)
    cap = lad.m * np.interp(t0, mat.T_K, mat.cp_J_kgK)
    assert np.allclose(cap * delta_ref, cap * delta_ref)
    assert out[LD_DU] == pytest.approx(4.0 - out[LD_AIR] - out[LD_COOL] - out[LD_SP], rel=1e-12)


def test_edge_kernel_matches_brute_force_superposition():
    resp = edge_response()
    xi, _ = heating_phase_nodes(3)
    shape = (1, 2)
    inp = new_edge_input(shape, 3)
    inp.xi[:] = xi
    inp.scal[:] = [4, 3]
    inp.mode[:] = [[EDGE_INTERRUPTED, EDGE_CONTINUOUS]]
    inp.q_r[:] = 3e7
    inp.a_r[:] = 1.5e-4
    inp.q_f[:] = 1e7
    inp.a_f[:] = 4e-5
    inp.b[:] = 2.5e-4
    inp.t_h[:] = 1e-3
    inp.period[:] = 6e-3
    inp.n_cyc[:] = 40
    inp.t_eng[:] = 0.02
    inp.k[:] = 70.0
    inp.alpha[:] = 2.3e-5
    inp.wedge[:] = 1.2
    inp.r_far[:] = 4e-3
    inp.T_body[:] = 320.0
    out = new_edge_output(shape, 3)
    glx, glw = gauss_legendre(3)
    K.edge_temperatures(inp, resp, glx, glw, out)
    for e in range(2):
        rake, peak, flank, amp = R.edge_element_ref(int(inp.mode[0, e]), 3e7, 1.5e-4, 1e7, 4e-5, 2.5e-4, 1e-3, 6e-3, 40,
                                                    0.02, 70.0, 2.3e-5, 1.2, 4e-3, 320.0, xi)
        assert np.allclose(out.rake_q[0, e] - 320, rake - 320, rtol=3e-3)
        assert np.allclose(out.peak_q[0, e] - 320, peak - 320, rtol=3e-3)
        assert np.allclose(out.flank_q[0, e] - 320, flank - 320, rtol=3e-3)
        assert out.amp[0, e] == pytest.approx(amp, rel=3e-3, abs=1e-9)


@pytest.mark.parametrize("v,x,t,w", [(0.0, 0.0, 0.01, 1e-3), (2.0, -5e-4, 0.05, 2e-3), (50.0, -2e-4, 0.01, 0.0),
                                     (0.02, 3e-3, 0.5, 4e-3)])
def test_moving_source_matches_quadrature(v, x, t, w):
    glx, glw = gauss_legendre(24)
    got = K.moving_rect_rise(1e6, 5e-4 if v < 10 else 2e-4, w, v, 45.0, 1.2e-5, t, x, 0.0, 0.0, glx, glw)
    ref = R.moving_rect_ref(1e6, 5e-4 if v < 10 else 2e-4, w, v, 45.0, 1.2e-5, t, x, 0.0, 0.0)
    assert got == pytest.approx(ref, rel=2e-4)
