"""Kernel rules (architecture 5.5): one explicit signature each, compiled code equal to the
plain-Python execution of the same source, and independent references for the lookups."""

from __future__ import annotations

import math

import numpy as np
from numba import types
from numba.core.registry import CPUDispatcher
from scipy.interpolate import RegularGridInterpolator

from cnctwin.mechanics import abi, kernels as K, params as P
from cnctwin.mechanics.oxley import kernel_inputs

DISPATCHERS = sorted(name for name, obj in vars(K).items() if isinstance(obj, CPUDispatcher))


def test_every_kernel_has_exactly_one_signature():
    assert len(DISPATCHERS) >= 20
    for name in DISPATCHERS:
        kernel = getattr(K, name)
        assert len(kernel.signatures) == 1, name
        assert kernel.targetoptions.get("nogil") is True, name
        assert kernel.targetoptions.get("fastmath", False) is False, name
        assert kernel.targetoptions.get("parallel", False) is False, name


def _close(a, b, rel=1e-12, abs_=1e-300):
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)


def test_leaf_kernels_match_python(steel):
    mat = steel.jc
    prm = P.oxley_vector(0.5, 0.5)
    xs = np.array([300.0, 400.0, 700.0])
    ys = np.array([1.0, 3.0, 2.0])
    for x in (100.0, 300.0, 350.0, 401.0, 650.0, 900.0):
        assert _close(K.interp1(xs, ys, x), K.interp1.py_func(xs, ys, x))
        assert _close(K.interp1(xs, ys, x), float(np.interp(x, xs, ys)))
    for eps, rate, temp in ((0.0, 1e3, 293.15), (0.5, 1e4, 600.0), (2.0, 1e6, 1500.0), (1.0, 1e-3, 200.0),
                            (1.0, 1e4, 2000.0)):
        assert _close(K.jc_flow_stress(eps, rate, temp, mat, 0.99, 1.0),
                      K.jc_flow_stress.py_func(eps, rate, temp, mat, 0.99, 1.0))
        assert _close(K.jc_hardening_index(eps, mat), K.jc_hardening_index.py_func(eps, mat))
    for x in (0.001, 0.04, 0.5, 10.0, 11.0, 1e3):
        assert _close(K.beta_partition(x, prm), K.beta_partition.py_func(x, prm))
    for m in (1e-3, 0.5, 1.0, 1.5, 30.0, 1e4):
        assert _close(K.rect_area_factor(m), K.rect_area_factor.py_func(m))
    tool = np.array([500.0, 15000.0, 1e-6, 2e-3, 0.0])
    kt = np.array([293.15, 1273.15])
    kv = np.array([80.0, 55.0])
    for time_s in (0.0, 1e-4, 1.0):
        tool[abi.TOOL_TIME_S] = time_s
        assert _close(K.tool_resistance(3e-4, 700.0, tool, kt, kv, kt, kv * 3, prm),
                      K.tool_resistance.py_func(3e-4, 700.0, tool, kt, kv, kt, kv * 3, prm))
    temps = np.linspace(400.0, 900.0, 50)
    assert _close(K.oxide_exposure(temps, 1e-3, 16838.0), K.oxide_exposure.py_func(temps, 1e-3, 16838.0))
    grid = np.geomspace(1e-6, 1e-3, 9)
    row = np.linspace(5.0, 1.0, 9)
    for h in (1e-7, 3e-6, 2e-4, 5e-3):
        assert _close(K.interp_log_h(grid, row, h), K.interp_log_h.py_func(grid, row, h))
        assert _close(K.interp_log_h(grid, row, h), float(np.interp(math.log(h), np.log(grid), row)))
        assert _close(K.interp_lin_h(grid, row, h), K.interp_lin_h.py_func(grid, row, h))
        assert _close(K.interp_lin_h(grid, row, h), float(np.interp(h, grid, row)))


def test_brent_reverse_communication():
    def run(init, step):
        bs = np.zeros(K.N_BRENT)
        init(bs, 0.0, 2.0, -2.0, 2.0, 1e-14)
        for _ in range(100):
            if step(bs) == 1:
                break
            bs[4] = bs[1] ** 2 - 2.0
        return bs[1]
    assert abs(run(K.brent_init, K.brent_step) - math.sqrt(2.0)) < 1e-12
    assert run(K.brent_init, K.brent_step) == run(K.brent_init.py_func, K.brent_step.py_func)


def test_edge_contact_matches_python():
    out_c = np.zeros(abi.N_PL)
    out_p = np.zeros(abi.N_PL)
    for depth, radius, stag, rake, incl in ((1e-5, 4e-5, 1.1e-5, 0.0, 0.0), (5e-6, 4e-5, 1.1e-5, 0.1, 0.3),
                                            (1e-4, 3e-5, 9e-6, -0.95, 0.1), (2e-5, 0.0, 5e-6, -0.2, 0.0)):
        K.edge_contact_forces(depth, radius, stag, rake, incl, 2.5e9, 5e8, out_c)
        K.edge_contact_forces.py_func(depth, radius, stag, rake, incl, 2.5e9, 5e8, out_p)
        assert np.allclose(out_c, out_p, rtol=1e-12, atol=0.0)
        assert np.all(np.isfinite(out_c))


def test_shear_plane_and_interface_match_python(steel, carbide):
    inputs = kernel_inputs(steel, carbide, tool_temp_K=600.0, mu_limit=0.5, mu_fallback=0.5)
    case = np.array([1e-4, 2.5, 0.1, 293.15])
    st_c = np.zeros(K.N_ST)
    st_p = np.zeros(K.N_ST)
    for phi in (0.2, 0.35, 0.5):
        a = K.shear_plane_state(phi, 5.0, abi.NO_REF, case, inputs.mat, inputs.cp_T, inputs.cp_v, inputs.k_T,
                                inputs.k_v, inputs.prm, st_c)
        b = K.shear_plane_state.py_func(phi, 5.0, abi.NO_REF, case, inputs.mat, inputs.cp_T, inputs.cp_v,
                                        inputs.k_T, inputs.k_v, inputs.prm, st_p)
        assert a == b
        assert np.allclose(st_c, st_p, rtol=1e-10, atol=1e-12)
        if a == 1:
            it_c = np.zeros(K.N_IT)
            it_p = np.zeros(K.N_IT)
            g_c = K.interface_state(st_c, 0.05, case, *inputs[:5], *inputs[5:10], inputs.prm, it_c)
            g_p = K.interface_state.py_func(st_c, 0.05, case, *inputs[:5], *inputs[5:10], inputs.prm, it_p)
            assert math.isclose(g_c, g_p, rel_tol=1e-10, abs_tol=1e-3)
            assert np.allclose(it_c, it_p, rtol=1e-10, atol=1e-9)


def test_oxley_point_matches_python(steel, carbide):
    inputs = kernel_inputs(steel, carbide, tool_temp_K=500.0, mu_limit=0.5, mu_fallback=0.5)
    case = np.array([1.5e-4, 2.5, 0.0, 293.15])
    out_c = np.zeros(abi.N_OUT)
    out_p = np.zeros(abi.N_OUT)
    K.oxley_point(case, *inputs, out_c)
    K.oxley_point.py_func(case, *inputs, out_p)
    assert out_c[abi.OX_STATUS] == abi.STATUS_OXLEY
    assert np.allclose(out_c, out_p, rtol=1e-9, atol=1e-9)


def test_map_lookup_against_scipy():
    rng = np.random.default_rng(7)
    t1 = np.geomspace(1e-6, 1e-3, 5)
    v = np.geomspace(0.5, 8.0, 4)
    tt = np.array([293.15, 600.0, 900.0])
    values = rng.uniform(1.0, 2.0, size=(3, 4, 5, abi.N_OUT))
    scaled = np.array(abi.SCALES_WITH_T1, dtype=np.int64)
    discrete = np.array(abi.DISCRETE_SLOTS, dtype=np.int64)
    query = np.array([2e-6, 3.3e-5, 7e-4])
    out = np.zeros((3, abi.N_OUT))
    out_p = np.zeros((3, abi.N_OUT))
    K.map_lookup(t1, v, tt, values, scaled, discrete, query, 1.7, 700.0, out)
    K.map_lookup.py_func(t1, v, tt, values, scaled, discrete, query, 1.7, 700.0, out_p)
    assert np.allclose(out, out_p, rtol=1e-12)
    points = (tt, np.log(v), np.log(t1))
    for q in range(abi.N_OUT):
        if q in abi.DISCRETE_SLOTS:
            continue
        data = values[..., q] / t1[None, None, :] if q in abi.SCALES_WITH_T1 else values[..., q]
        ref = RegularGridInterpolator(points, data)
        expect = ref(np.array([[700.0, math.log(1.7), math.log(x)] for x in query]))
        if q in abi.SCALES_WITH_T1:
            expect = expect * query
        assert np.allclose(out[:, q], expect, rtol=1e-10), q


def test_bue_step_deterministic_and_matches_python():
    prm = P.bue_vector()
    keys = np.arange(6, dtype=np.int64)

    def run(fn, seed):
        h = np.full(6, 2e-5)
        counters = np.zeros(6, dtype=np.int64)
        broke = np.zeros(6, dtype=np.int64)
        total = np.zeros(6, dtype=np.int64)
        window = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 0.0])
        uncut = np.full(6, 1e-4)
        for _ in range(200):
            fn(h, window, uncut, 1e-3, prm, seed, 12345, keys, counters, broke)
            total += broke
        return h, counters, total

    h1, c1, b1 = run(K.bue_step, 3)
    h2, c2, b2 = run(K.bue_step.py_func, 3)
    h3, _, b3 = run(K.bue_step, 4)
    assert np.array_equal(h1, h2) and np.array_equal(c1, c2) and np.array_equal(b1, b2)
    assert np.all(c1[[0, 1, 2, 4]] == 200) and np.all(c1[[3, 5]] == 0)
    assert np.all(b1[[3, 5]] == 0) and b1[[0, 1, 2, 4]].sum() > 0
    assert not np.array_equal(b1, b3)
    assert np.all(h1[[3, 5]] < 1e-12)
    sat = prm[0] * 1e-4
    assert np.all(h1[[0, 1, 2, 4]] <= sat * (1 + 1e-12))


def test_array_arguments_are_c_contiguous():
    for name in DISPATCHERS:
        for arg in getattr(K, name).signatures[0]:
            if isinstance(arg, types.Array):
                assert arg.layout == "C", (name, arg)
