"""The GPU workpiece finite-volume step equals thermal's work_step bit for bit: enthalpy, temperature, every
scratch array the CPU kernel defines (line_c, line_d, gprev are unspecified scratch) and the ledger terms
(cnctwin.gpu.k_thermal)."""

from __future__ import annotations

import numpy as np
import pytest

from cnctwin.gpu import cases, k_thermal as GT
from cnctwin.gpu.accel import Accelerator
from cnctwin.gpu.device import GpuError
from cnctwin.gpu.schema import GpuSettings
from cnctwin.thermal import kernels as K
from cnctwin.thermal.abi import L_AIR, L_CONTACT, L_COOL, L_LOST, L_SRC, N_L

from gpufix import differing

SCRATCH_SKIP = ("line_c", "line_d", "gprev")


def run(call, case):
    g, s, mat, bnd, dt = case
    g, s = cases.clone(g), cases.clone(s)
    out = np.zeros(N_L)
    call(g, s, mat, bnd, dt, out)
    return g, s, out


def assert_same(ref, got):
    assert differing(ref[0], got[0]) == []
    assert differing(ref[1], got[1], skip=SCRATCH_SKIP) == []
    assert np.array_equal(ref[2], got[2], equal_nan=True)


def test_case_exercises_every_term():
    g, s, out = run(K.work_step, cases.thermal_case(seed=1))
    assert out[L_SRC] > 0 and out[L_AIR] != 0 and out[L_COOL] != 0 and out[L_CONTACT] != 0 and out[L_LOST] > 0


@pytest.mark.gpu
@pytest.mark.parametrize("shape,dt,coolant,seed", [((10, 8, 6), 0.05, True, 1), ((7, 9, 5), 2.0, True, 2),
                                                   ((6, 5, 8), 0.5, False, 3), ((1, 4, 3), 0.1, True, 4)])
def test_gpu_equals_cpu(gpu_kind, shape, dt, coolant, seed):
    g, s, mat, bnd, _ = cases.thermal_case(shape=shape, coolant=coolant, seed=seed)
    case = (g, s, mat, bnd, dt)
    assert_same(run(K.work_step, case), run(GT.work_step, case))


@pytest.mark.gpu
def test_repeated_steps_stay_identical(gpu_kind):
    g, s, mat, bnd, dt = cases.thermal_case(seed=5)
    g1, s1, g2, s2 = cases.clone(g), cases.clone(s), cases.clone(g), cases.clone(s)
    for _ in range(3):
        o1, o2 = np.zeros(N_L), np.zeros(N_L)
        K.work_step(g1, s1, mat, bnd, dt, o1)
        GT.work_step(g2, s2, mat, bnd, dt, o2)
        assert np.array_equal(o1, o2, equal_nan=True)
        s1.src[:] = 0.0
        s2.src[:] = 0.0
    assert differing(g1, g2) == []


@pytest.mark.gpu
def test_accelerator_falls_back_and_stays_on_the_cpu(gpu_kind, monkeypatch):
    case = cases.thermal_case(seed=6)
    reference = run(K.work_step, case)
    accel = Accelerator(GpuSettings(mode="on"))
    assert_same(reference, run(accel.work_step, case))
    assert accel.usage["thermal_work_step"].gpu_calls == 1

    def broken():
        raise GpuError("simulated device failure")

    monkeypatch.setattr(GT, "kernels", broken)
    assert_same(reference, run(accel.work_step, case))           # CPU kernel on untouched inputs
    usage = accel.usage["thermal_work_step"]
    assert usage.fallbacks == 1 and usage.cpu_calls == 1 and "simulated device failure" in accel.disabled_reason
    monkeypatch.undo()
    assert_same(reference, run(accel.work_step, case))
    assert accel.usage["thermal_work_step"].cpu_calls == 2        # stays on the CPU for the rest of the run
    assert accel.run_info()["errors"]


def test_empty_grid_is_refused():
    from cnctwin.thermal import abi as T
    g = T.new_work_grid((0, 2, 2), (0.0, 0.0, 0.0), 1e-3)
    with pytest.raises(GpuError):
        GT.work_step(g, T.new_work_scratch((0, 2, 2)), cases.thermal_case()[2], cases.thermal_case()[3], 0.1,
                     np.zeros(N_L))
