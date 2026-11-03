"""The GPU dexel subtraction equals removal's CPU kernel bit for bit: every field array (segments, event ids,
pool, bricks, info), the accumulators, the events and the return status (cnctwin.gpu.k_dexel)."""

from __future__ import annotations

import numpy as np
import pytest

from cnctwin.gpu import cases, k_dexel as GD
from cnctwin.gpu.accel import Accelerator
from cnctwin.gpu.schema import GpuSettings
from cnctwin.removal import abi as A
from cnctwin.removal import k_subtract as KS

from gpufix import differing


def assert_same(ref, got):
    (f1, a1, e1, s1), (f2, a2, e2, s2) = ref, got
    assert s1 == s2
    assert differing(f1, f2) == []
    assert differing(a1, a2) == []
    assert differing(e1, e2) == []


def kinds(ev) -> set[int]:
    return set(ev.rec[:ev.n[0], A.EC_KIND].astype(int).tolist())


def spilled(f, prof, spec, tgt, acc, ev, sc):
    """The commit with every ray marked SPILL: the host computes all intervals."""
    n, cand_f, cand_i = GD.gather(f, prof, spec)
    if n < 0:
        return KS.subtract_envelope(f, prof, spec, tgt, acc, ev, sc)
    out_n = np.full(max(n, 1), GD.SPILL, np.int64)
    out_iv = np.zeros((max(n, 1), GD.MAX_OUT, 2))
    status, hits, misses = GD.commit(f, prof, spec, tgt, acc, ev, sc, n, cand_f, cand_i, out_n, out_iv)
    assert hits == 0 and misses >= n
    return status


def shuffled_gather(f, prof, spec, tgt, acc, ev, sc):
    """gather foresees rays the CPU kernel skips and misses some it computes: still exact."""
    n, cand_f, cand_i = GD.gather(f, prof, spec)
    if n < 2:
        return GD.subtract_envelope_cpu_twin(f, prof, spec, tgt, acc, ev, sc)
    out_n = np.zeros(n, np.int64)
    out_iv = np.zeros((n, GD.MAX_OUT, 2))
    GD.cpu_rays()(np.int64(n), cand_f, cand_i, *GD._profile_args(prof, spec), out_n, out_iv,
                  np.zeros((GD.MAX_RAW, 2)), np.zeros(GD.MAX_RAW, np.int32), np.zeros(GD.WORK))
    keep = np.ones(n, bool)
    keep[1::3] = False                                   # forget every third candidate
    extra_f = cand_f[:1].copy()
    extra_f[0, GD.CF_OX] += 12.345e-3                    # a ray nobody asks for
    cf = np.concatenate([extra_f, cand_f[:n][keep]])
    ci = np.concatenate([cand_i[:1], cand_i[:n][keep]])
    on = np.concatenate([out_n[:1], out_n[keep]])
    oiv = np.concatenate([out_iv[:1], out_iv[keep]])
    status, hits, misses = GD.commit(f, prof, spec, tgt, acc, ev, sc, len(cf), cf, ci, on, oiv)
    assert misses > 0 and hits > 0
    return status


def test_case_exercises_the_kernel():
    case = cases.dexel_case(seed=3)
    f, acc, ev, statuses = cases.dexel_run(case, KS.subtract_envelope)
    assert acc.removed_q.sum() > 0 and acc.ivals[A.IA_RAYS_CHANGED] > 0
    assert {A.EV_GOUGE, A.EV_BREAKTHROUGH, A.EV_RAPID_INTO_MATERIAL, A.EV_NON_CENTRE_PLUNGE} <= kinds(ev)
    assert case.levels == 2


def test_cpu_phases_equal_the_cpu_kernel():
    case = cases.dexel_case(seed=3)
    assert_same(cases.dexel_run(case, KS.subtract_envelope), cases.dexel_run(case, GD.subtract_envelope_cpu_twin))


def test_spilled_rays_are_computed_on_the_host():
    case = cases.dexel_case(tool="ball", seed=4)
    assert_same(cases.dexel_run(case, KS.subtract_envelope), cases.dexel_run(case, spilled))


def test_lookup_survives_a_gather_that_drifted():
    case = cases.dexel_case(seed=6)
    assert_same(cases.dexel_run(case, KS.subtract_envelope), cases.dexel_run(case, shuffled_gather))


def test_nothing_to_do_changes_nothing():
    case = cases.dexel_case(cuts=1, seed=1)
    empty = A.empty_profile()
    assert GD.candidate_bound(case.field, empty, case.specs[0]) == -1
    f = type(case.field)(*(a.copy() for a in case.field))
    acc = A.new_accum(case.levels)
    assert GD.subtract_envelope(f, empty, case.specs[0], case.target, acc, _events(), A.make_scratch()) == 0
    assert differing(f, case.field) == [] and acc.ivals[A.IA_SUBTRACTIONS] == 0


def _events():
    from cnctwin.core.abi import event_buffer
    return event_buffer(4, A.EVENT_WIDTH)


@pytest.mark.gpu
@pytest.mark.parametrize("tool,refine,target", [("torus", True, True), ("ball", False, True), ("torus", False, False)])
def test_gpu_equals_cpu(gpu_kind, tool, refine, target):
    case = cases.dexel_case(tool=tool, refine=refine, target=target, seed=3)
    assert GD.candidate_bound(case.field, case.profile, case.specs[0]) > 0
    assert_same(cases.dexel_run(case, KS.subtract_envelope), cases.dexel_run(case, GD.subtract_envelope))


@pytest.mark.gpu
def test_gpu_equals_cpu_through_pool_overflow(gpu_kind):
    case = cases.dexel_case(size_mm=(6.0, 3.0, 3.0), cuts=6, refine=False, pool_blocks=1, radius_frac=0.1, seed=7)
    ref = cases.dexel_run(case, KS.subtract_envelope)
    assert 1 in ref[3]                                   # a subtraction ran out of pool blocks and was repeated
    assert_same(ref, cases.dexel_run(case, GD.subtract_envelope))


@pytest.mark.gpu
def test_accelerator_uses_the_gpu_and_records_it(gpu_kind):
    case = cases.dexel_case(cuts=3, refine=False, seed=2)
    accel = Accelerator(GpuSettings(mode="on"))
    got = cases.dexel_run(case, accel.subtract_envelope)
    assert_same(cases.dexel_run(case, KS.subtract_envelope), got)
    usage = accel.usage["dexel_subtract"]
    assert usage.gpu_calls == 3 and usage.cpu_calls == 0 and usage.fallbacks == 0 and usage.gpu_items > 0
    assert accel.run_info()["kernels"]["dexel_subtract"]["last_backend"] == "gpu"
