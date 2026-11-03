"""Every dynamics kernel against its plain-Python reference, and one compiled signature each (architecture 5.5)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cnctwin.core.abi import ACC_DAMPING_J, ACC_PROCESS_DAMPING_J, ACC_UNBALANCE_WORK_J, ring_buffer
from cnctwin.dynamics import kernels as K
from cnctwin.dynamics import reference as R
from cnctwin.dynamics.abi import (AxisBase, KIND_MACHINE, MI_HAS_TOOL_RES, MI_RELAX, core_accum, is_kernel_layout)
from cnctwin.dynamics.drive import bench_run

from tests.unit.dynamics.dynfix import assert_close_tuple, clone, system, tool_mode, work_grid, cantilever_mode

RNG = np.random.default_rng(20261103)


def _rich_system(l0_ticks=8, tick_s=1e-6, relax=True):
    """Tool bending modes at 4 stations, a machine mode, two workpiece modes, residuals, unbalance, base inputs."""
    z = np.array([0.0, 0.004, 0.008, 0.012])
    modes = [cantilever_mode(1800.0, 0.02, z + 0.05, 0.07, 0.08, axis=0),
             cantilever_mode(1850.0, 0.025, z + 0.05, 0.07, 0.09, axis=1),
             tool_mode(60.0, 0.05, 400.0, (1.0, 0.3, 0.0), nst=4, kind=KIND_MACHINE,
                       base=[-20.0, -6.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
    wmodes, grid = work_grid(2, (900.0, 1300.0))
    residual = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            residual[i, j] = 2e-9 * math.exp(-abs(i - j)) * (1.0 if (i % 3) == (j % 3) else 0.2)
    Kc = np.array([[-3e6, 1e6, 0.0], [-2e6, -4e6, 0.0], [0.0, 0.0, 0.0]]) if relax else None
    sys, layout = system(modes + wmodes, nst=4, ref=1, l0_ticks=l0_ticks, tick_s=tick_s, rotating=True,
                         unbalance=(np.array([2e-6, -1e-6, 0.0]), np.array([1e-8, 3e-8, 0.0])), K_c=Kc,
                         residual=residual, grid=grid, point_S=np.array([0.013, 0.01, 0.01]))
    nm = sys.state.x.shape[0]
    sys.state.x[:] = RNG.normal(size=(nm, 2)) * np.array([1e-6, 1e-3])
    sys.ds.f_cut[:] = RNG.normal(size=nm)
    sys.ds.f_cut_prev[:] = RNG.normal(size=nm)
    sys.ds.f_unb[:] = RNG.normal(size=nm) * 0.1
    sys.ds.f_base[:] = RNG.normal(size=nm) * 0.1
    sys.ds.st_force[:, :3] = RNG.normal(size=(4, 3)) * 50.0
    sys.ds.work_force_S[:] = RNG.normal(size=3) * 80.0
    sys.ds.flags[0] = 1
    sys.ds.node_tick[0] = 1000
    return sys, layout


def test_layout_is_kernel_ready():
    sys, _ = _rich_system()
    assert is_kernel_layout(sys)
    assert sys.dm.iscal[MI_RELAX] == 1
    assert sys.dm.iscal[MI_HAS_TOOL_RES] == 1


@pytest.mark.parametrize("case", range(40))
def test_free_interval(case):
    rng = np.random.default_rng(case)
    w = TWO_PI * 10 ** rng.uniform(1.0, 3.7)
    z = 10 ** rng.uniform(-3.0, -0.4)
    h = (10 ** rng.uniform(-1.6, 1.5)) / w           # omega h from 0.025 to 30: both branches
    args = (w, z, h, rng.normal() * 1e-6, rng.normal() * 1e-3, rng.normal() * 3, rng.normal() * 3,
            float(rng.choice([0.0, w * rng.uniform(0.7, 1.3), TWO_PI * rng.uniform(5, 300)])), rng.normal(),
            rng.normal() / h)
    out = np.zeros(5)
    K.free_interval(*args, out)
    ref = R.free_interval_ref(*args)
    scale = abs(ref[2]) + abs(ref[3]) + abs(ref[4]) + 1e-300
    np.testing.assert_allclose(out[0], ref[0], rtol=1e-9, atol=1e-9 * (abs(args[3]) + abs(args[4]) / w + 1e-12))
    np.testing.assert_allclose(out[1], ref[1], rtol=1e-9, atol=1e-9 * (abs(args[4]) + abs(args[3]) * w + 1e-12))
    np.testing.assert_allclose(out[2:], ref[2:], rtol=0, atol=2e-8 * scale)
    # the exact energy identity of the interval
    E0 = 0.5 * (args[4] ** 2 + w * w * args[3] ** 2)
    E1 = 0.5 * (out[1] ** 2 + w * w * out[0] ** 2)
    assert abs(E1 - E0 - (out[2] + out[3] - out[4])) <= 1e-9 * (abs(E0) + abs(E1) + scale)


TWO_PI = 2.0 * math.pi


def test_external_force():
    sys, _ = _rich_system()
    a, b = clone(sys), clone(sys)
    base = RNG.normal(size=7)
    K.external_force(a, 1.234, -900.0, 35.0, base)
    R.external_force_ref(b, 1.234, -900.0, 35.0, base)
    np.testing.assert_allclose(a.fe, b.fe, rtol=1e-12, atol=1e-15)
    assert np.any(a.fe[0] != 0.0) and np.any(a.fe[1] != 0.0)


def test_predict_and_deflection():
    sys, _ = _rich_system()
    base = RNG.normal(size=7)
    K.external_force(sys, 0.3, -1000.0, 0.0, base)
    a, b = clone(sys), clone(sys)
    K.modal_predict(a)
    R.modal_predict_ref(b)
    np.testing.assert_allclose(a.xp, b.xp, rtol=1e-12, atol=1e-18)
    K.modal_deflection(a)
    R.modal_deflection_ref(b)
    assert_close_tuple(a.defl, b.defl, rtol=1e-11, atol=1e-18)
    out = np.zeros(9)
    K.relative_deflection(a, out)
    np.testing.assert_allclose(out, R.relative_deflection_ref(b), rtol=1e-11, atol=1e-18)


def test_correct():
    sys, _ = _rich_system()
    K.external_force(sys, 0.3, -1000.0, 50.0, RNG.normal(size=7))
    K.modal_predict(sys)
    a, b = clone(sys), clone(sys)
    station = RNG.normal(size=(4, 6)) * 40.0
    fS = RNG.normal(size=3) * 60.0
    acc_a, acc_b = core_accum(), core_accum()
    K.modal_correct(a, station, fS, 1008, acc_a)
    R.modal_correct_ref(b, station, fS, 1008, acc_b)
    for name in ("x", "f_prev"):
        np.testing.assert_allclose(getattr(a.state, name), getattr(b.state, name), rtol=1e-11, atol=1e-18)
    for name in ("f_cut", "f_cut_prev", "f_unb", "f_base", "st_force", "work_force_S", "node_tick", "flags",
                 "samp_tick", "samp_u", "samp_w", "samp_n"):
        np.testing.assert_allclose(getattr(a.ds, name), getattr(b.ds, name), rtol=1e-11, atol=1e-18, err_msg=name)
    np.testing.assert_allclose(a.da.scal, b.da.scal, rtol=1e-10, atol=1e-20)
    for idx in (ACC_DAMPING_J, ACC_UNBALANCE_WORK_J):
        np.testing.assert_allclose(acc_a.scalars[idx], acc_b.scalars[idx], rtol=1e-10, atol=1e-22)
    assert a.ds.flags[2] == 0                      # the step matched the model's L0 stride


def test_correct_flags_wrong_step():
    sys, _ = _rich_system()
    K.modal_predict(sys)
    K.modal_correct(sys, np.zeros((4, 6)), np.zeros(3), 1005, core_accum())
    assert sys.ds.flags[2] == 1 and sys.ds.flags[3] == 5


def test_free_tick():
    sys, _ = _rich_system()
    a, b = clone(sys), clone(sys)
    b0, b1 = RNG.normal(size=7), RNG.normal(size=7)
    K.modal_free(a, 0.4, 0.4 - 1.05, -1040.0, -1050.0, b0, b1, 2000)
    R.modal_free_ref(b, 0.4, 0.4 - 1.05, -1040.0, -1050.0, b0, b1, 2000)
    np.testing.assert_allclose(a.state.x, b.state.x, rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(a.ds.f_unb, b.ds.f_unb, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(a.da.scal, b.da.scal, rtol=1e-8, atol=1e-16)
    assert a.ds.node_tick[0] == 2000 and a.ds.flags[0] == 0


def test_refresh_work_shapes():
    sys, _ = _rich_system()
    _, grid = work_grid(2, (900.0, 1300.0))
    for point in ([0.013, 0.01, 0.01], [-0.5, 0.0, 0.0], [0.9, 0.2, 0.03], [0.0151, 0.0049, 0.025]):
        a, b = clone(sys), clone(sys)
        K.refresh_work_shapes(a, grid, np.array(point))
        R.refresh_work_shapes_ref(b, grid, np.array(point))
        np.testing.assert_allclose(a.model.work_shape_at_contact, b.model.work_shape_at_contact, rtol=1e-12)
        np.testing.assert_allclose(a.dm.res_work_S, b.dm.res_work_S, rtol=1e-12)
        np.testing.assert_allclose(a.dm.contact_shape, b.dm.contact_shape, rtol=1e-12, atol=1e-18)


def test_axis_acceleration():
    axes = AxisBase(np.array([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0], [0, 0, 1.0]]), np.array([-1.0, -1.0, 1.0, -1.0]),
                    np.array([0, 0, 0, 1], np.int8), np.array([-1, -1, -1, 0], np.int32))
    acc = np.array([2.0, -3.0, 1.5, 40.0])
    a, b = np.zeros(7), np.zeros(7)
    K.relative_axis_acceleration(axes, acc, a)
    R.relative_axis_acceleration_ref(axes, acc, b)
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(a[:4], [2.0, -3.0, 1.5, 40.0])     # table -X moves the work: tool +X relative


@pytest.mark.parametrize("slope", [0.0, 1e-4, 0.02, 0.1, 0.2, 0.5])
def test_process_damping_element(slope):
    land, tanc, mx = 2e-4, math.tan(math.radians(7.0)), 5e-4
    np.testing.assert_allclose(K.interference_area(slope, land, tanc, mx), R.interference_area_ref(slope, land, tanc, mx),
                               rtol=2e-6, atol=1e-18)
    acc_a, acc_b = core_accum(), core_accum()
    V = 3.0
    fa = K.process_damping_element(4e13, 2e-3, land, tanc, mx, slope * V, V, 1e-5, acc_a)
    fb = R.process_damping_element_ref(4e13, 2e-3, land, tanc, mx, slope * V, V, 1e-5, acc_b)
    np.testing.assert_allclose(fa, fb, rtol=2e-6, atol=1e-18)
    np.testing.assert_allclose(acc_a.scalars[ACC_PROCESS_DAMPING_J], acc_b.scalars[ACC_PROCESS_DAMPING_J], rtol=2e-6,
                               atol=1e-24)


def test_small_slope_area_is_linear():
    """A(s) -> s L^2 / 2 for small slopes: the linearised damping K_sp L^2 / (2 V) of one pass (Altintas 2008)."""
    land = 2e-4
    s = 1e-5
    assert abs(K.interference_area(s, land, 0.12, 5e-4) / (0.5 * s * land * land) - 1.0) < 1e-3


def test_deflection_at_point_and_tick_record():
    sys, _ = _rich_system()
    K.modal_predict(sys)
    K.modal_deflection(sys)
    r = np.array([0.003, -0.002, 0.0005])
    out = np.zeros(6)
    K.deflection_at_point(sys.defl, 2, r, out)
    np.testing.assert_allclose(out, R.deflection_at_point_ref(sys.defl, 2, r), rtol=1e-12, atol=1e-20)
    K.record_tick(sys, 3)
    np.testing.assert_allclose(sys.da.tick_u[3], R.relative_deflection_ref(sys), rtol=1e-11, atol=1e-18)
    assert sys.da.tick_n[0] == 4
    ring = ring_buffer(8, 20)
    K.ring_write(sys, ring, 5, 6)
    np.testing.assert_allclose(ring.data[5, 6:9], R.relative_deflection_ref(sys)[6:9], rtol=1e-11, atol=1e-18)
    assert np.all(np.isnan(ring.data[5, :6]))


def test_one_signature_each():
    for kernel in K.KERNELS + (bench_run,):
        assert len(kernel.signatures) == 1, kernel.__name__
