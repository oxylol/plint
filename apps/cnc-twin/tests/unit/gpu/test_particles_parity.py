"""The GPU particle step equals chips' K1 (advance_particles) bit for bit in the simulator: particles, pile,
accumulators and events, including the pile coupling between particles of one call (recomputed on the host)
and heat logs too long for a GPU thread (cnctwin.gpu.k_particles)."""

from __future__ import annotations

import numpy as np
import pytest

from cnctwin.chips import abi
from cnctwin.chips import kernels as CK
from cnctwin.gpu import cases, k_particles as GP
from cnctwin.gpu.accel import Accelerator
from cnctwin.gpu.schema import GpuSettings

from gpufix import differing


@pytest.fixture(scope="module")
def model():
    return cases.chips_model()


def run(call, case, events=64):
    P, H, W, S, C = case
    P, H = cases.clone(P), cases.clone(H)
    out, ev = abi.step_out(events)
    stats = call(P, H, W, S, C, out, ev)
    return P, H, out, ev, stats


def assert_same(ref, got, exact=True):
    if exact:
        assert differing(ref[0], got[0]) == []
        assert differing(ref[1], got[1]) == []
        assert np.array_equal(ref[2], got[2])
        assert differing(ref[3], got[3]) == []
        return
    # a real device: libdevice exp/expm1/pow may differ from glibc by a few ulp (docs/models/gpu.md "Tolerances")
    assert np.array_equal(ref[0].status, got[0].status)
    for name in ("pos_M_m", "vel_M_m_s", "temp_K", "time_s"):
        np.testing.assert_allclose(getattr(got[0], name), getattr(ref[0], name), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(got[2], ref[2], rtol=1e-9, atol=1e-15)


def test_cpu_phases_equal_k1(model):
    case = cases.chips_case(n=24, seed=11, model=model)
    ref = run(CK.advance_particles, case)
    got = run(GP.advance_particles_cpu_twin, case)
    assert_same(ref, got)
    P, H, out, ev, _ = ref
    kinds = set(ev.rec[:ev.n[0], 0].astype(int).tolist())
    assert {abi.EV_WELD, abi.EV_WRAP, abi.EV_NOZZLE} <= kinds
    assert out[abi.OUT_PAN_MASS_KG] > 0 and out[abi.OUT_DEPOSIT_KG] > 0 and out[abi.OUT_HITS] > 0


@pytest.mark.gpu
@pytest.mark.parametrize("liquid,seed", [(True, 11), (False, 12)])
def test_gpu_equals_k1(gpu_kind, model, liquid, seed):
    case = cases.chips_case(n=24, seed=seed, liquid=liquid, model=model)
    assert_same(run(CK.advance_particles, case), run(GP.advance_particles, case), exact=gpu_kind == "simulator")


@pytest.mark.gpu
def test_pile_coupling_is_resolved_in_particle_order(gpu_kind, model):
    case = cases.stack_case(n=6, model=model)
    ref = run(CK.advance_particles, case)
    settled = ref[1].mass_kg[5, 7] - case[1].mass_kg[5, 7]
    assert settled > 2.5 * case[0].mass_kg[0]                  # several parcels landed on the same cell
    got = run(GP.advance_particles, case)
    assert got[4][1] > 0                                       # pile conflicts recomputed on the host
    assert_same(ref, got, exact=gpu_kind == "simulator")
    # negative control: integrated against the pile at the start of the call, the stack comes out different
    k = GP.pack(*case)
    rf, ri, glog = GP.results(k["n"])
    GP.cpu_entries()["flight_all"](np.int64(k["n"]), k["pf"], k["pi"], *[k[key] for key in GP.PILE_KEYS],
                                   *[k[key] for key in GP.SCENE_KEYS], rf, ri, glog)
    frozen_z = rf[:k["n"], GP.RF_Z]
    assert not np.array_equal(frozen_z, ref[0].pos_M_m[:k["n"], 2])


@pytest.mark.gpu
def test_long_heat_logs_are_recomputed_on_the_host(gpu_kind, model):
    P, H, W, S, C = cases.chips_case(n=12, seed=13, model=model)
    numerics = C.numerics.copy()
    numerics[0] = 2e-5                                          # 500 sub-steps in the 10 ms window
    C = C._replace(numerics=numerics)
    case = (P, H, W, S, C)
    ref = run(CK.advance_particles, case)
    got = run(GP.advance_particles, case)
    assert got[4][0] > 0 and ref[2][abi.OUT_SUBSTEPS] > GP.CAP_LOG
    assert_same(ref, got, exact=gpu_kind == "simulator")


@pytest.mark.gpu
def test_no_particles_and_the_accelerator(gpu_kind, model):
    P, H, W, S, C = cases.chips_case(n=10, seed=14, model=model)
    empty = P._replace(n=np.zeros(1, np.int64))
    stats = GP.advance_particles(empty, H, W, S, C, *abi.step_out(4))
    assert list(stats) == [0, 0]
    accel = Accelerator(GpuSettings(mode="on"))
    case = (P, H, W, S, C)
    assert_same(run(CK.advance_particles, case), run(lambda *a: accel.advance_particles(*a), case),
                exact=gpu_kind == "simulator")
    usage = accel.usage["chips_advance"]
    assert usage.gpu_calls == 1 and usage.gpu_items == 9 and usage.fallbacks == 0
