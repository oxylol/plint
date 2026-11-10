"""Field layout, overflow pool, interval algebra, exact volumes, checkpoint arrays, memory budget."""

import numpy as np
import pytest

from cnctwin.removal import abi as A
from cnctwin.removal import field as F
from cnctwin.removal import k_ray as KR
from cnctwin.removal import reference as R

Q = A.QUANTUM_M


def _scratch_lists(segs):
    n = len(segs)
    lo = np.zeros(A.MAX_SEG, np.int32)
    hi = np.zeros(A.MAX_SEG, np.int32)
    elo = np.zeros(A.MAX_SEG, np.int32)
    ehi = np.zeros(A.MAX_SEG, np.int32)
    for k, (a, b, ea, eb) in enumerate(segs):
        lo[k], hi[k], elo[k], ehi[k] = a, b, ea, eb
    return n, lo, hi, elo, ehi


def _out_lists():
    return (np.zeros(A.MAX_SEG, np.int32), np.zeros(A.MAX_SEG, np.int32), np.zeros(A.MAX_SEG, np.int32),
            np.zeros(A.MAX_SEG, np.int32), np.zeros(A.MAX_SEG, np.int64), np.zeros(A.MAX_SEG, np.int64))


@pytest.mark.parametrize("seed", range(6))
def test_interval_subtract_and_union_match_reference(seed):
    rng = np.random.default_rng(seed)
    pts = np.sort(rng.choice(np.arange(0, 2000), size=2 * int(rng.integers(1, 9)), replace=False))
    segs = [(int(pts[2 * k]), int(pts[2 * k + 1]), 7, 8) for k in range(len(pts) // 2)]
    rp = np.sort(rng.choice(np.arange(0, 2000), size=2 * int(rng.integers(1, 6)), replace=False))
    rem = [(int(rp[2 * k]), int(rp[2 * k + 1])) for k in range(len(rp) // 2)]
    n, lo, hi, elo, ehi = _scratch_lists(segs)
    rlo = np.array([a for a, _ in rem], np.int64)
    rhi = np.array([b for _, b in rem], np.int64)
    nlo, nhi, nelo, nehi, plo, phi = _out_lists()
    nn, npc, removed = KR.iv_subtract(n, lo, hi, elo, ehi, len(rem), rlo, rhi, 99, nlo, nhi, nelo, nehi, plo, phi)
    ref, pieces, rm = R.iv_subtract_ref(segs, rem, 99)
    assert removed == rm
    assert [(int(nlo[k]), int(nhi[k]), int(nelo[k]), int(nehi[k])) for k in range(nn)] == ref
    assert [(int(plo[k]), int(phi[k])) for k in range(npc)] == pieces
    nn, npc, added = KR.iv_union(n, lo, hi, elo, ehi, len(rem), rlo, rhi, 99, nlo, nhi, nelo, nehi, plo, phi)
    ref_u, add_ref = R.iv_union_ref(segs, rem, 99)
    assert added == add_ref
    assert [(int(nlo[k]), int(nhi[k])) for k in range(nn)] == [(a, b) for a, b, _, _ in ref_u]


def test_pool_chain_store_load_free_and_overflow():
    f = F.make_field([0, 0, 0], [2e-3, 2e-3, 2e-3], 0.25e-3, pool_blocks=2)
    ray = 0
    segs = [(10 * k, 10 * k + 5, k, k + 100) for k in range(11)]           # 11 segments: 2 pool blocks
    n, lo, hi, elo, ehi = _scratch_lists(segs)
    assert KR.ray_store(f, ray, lo, hi, elo, ehi, n) == 0
    assert f.info[A.FI_N_FREE] == 0
    assert R.ray_segments(f, ray) == segs
    out = _scratch_lists([])
    assert KR.ray_load(f, ray, out[1], out[2], out[3], out[4]) == 11
    # one more segment needs a third block: refused, ray unchanged
    segs13 = segs + [(200, 210, 1, 2), (220, 230, 3, 4)]
    n13, lo13, hi13, elo13, ehi13 = _scratch_lists(segs13)
    assert KR.ray_store(f, ray, lo13, hi13, elo13, ehi13, n13) == -1
    assert f.info[A.FI_OVERFLOW] == 1
    assert R.ray_segments(f, ray) == segs
    g = F.grow_pool(f, 4)
    assert g.info[A.FI_N_FREE] == 4 and g.info[A.FI_OVERFLOW] == 0
    assert KR.ray_store(g, ray, lo13, hi13, elo13, ehi13, n13) == 0
    assert R.ray_segments(g, ray) == segs13
    # shrinking frees the chain back to the pool
    n2, lo2, hi2, elo2, ehi2 = _scratch_lists(segs[:3])
    assert KR.ray_store(g, ray, lo2, hi2, elo2, ehi2, n2) == 0
    assert g.info[A.FI_N_FREE] == g.info[A.FI_N_BLOCKS]
    assert g.link[ray] == -1


def test_lookups_match_reference():
    f = F.make_field([0, 0, 0], [2e-3, 2e-3, 2e-3], 0.25e-3, pool_blocks=8)
    segs = [(100 * k, 100 * k + 40, 0, 0) for k in range(9)]
    n, lo, hi, elo, ehi = _scratch_lists(segs)
    KR.ray_store(f, 3, lo, hi, elo, ehi, n)
    for q in (0.0, 39.9, 40.0, 55.0, 455.0, 799.0, 840.0, 2000.0):
        assert KR.ray_nearest(f, 3, q) == R.ray_nearest_ref(f, 3, q)
        assert KR.ray_inside(f, 3, q) == bool(R.ray_nearest_ref(f, 3, q)[0])
    for q0, q1 in ((0, 1000), (35, 245), (850, 900)):
        assert KR.ray_overlap(f, 3, q0, q1) == R.ray_overlap_ref(f, 3, q0, q1)


def test_family_sums_exact_and_reference():
    from cnctwin.removal.stock import block_field

    f = block_field([3e-3, 2e-3, 1.5e-3], 0.25e-3)
    for fam in range(3):
        assert KR.family_length(f, 0, fam) == R.family_length_ref(f, 0, fam)
        assert F.volume_m3(f, 0, fam) == pytest.approx(3e-3 * 2e-3 * 1.5e-3, rel=1e-12)


def test_refine_levels_and_lattice_geometry():
    f = F.make_field([0, 0, 0], [10e-3, 6e-3, 4e-3], 0.25e-3, refine=[((2e-3, 1e-3, 0), (4e-3, 3e-3, 4e-3), 4)])
    assert F.n_levels(f) == 2
    assert f.lv_cell_q[1] * 4 == f.lv_cell_q[0]
    lo, hi = F.lattice_box(f, 0)
    assert np.all(lo < 0) and np.all(hi > [10e-3, 6e-3, 4e-3])
    assert F.finest_level_at(f, [3e-3, 2e-3, 1e-3]) == 1
    assert F.finest_level_at(f, [8e-3, 2e-3, 1e-3]) == 0


def test_checkpoint_round_trip_bitwise():
    from cnctwin.removal.stock import tube_field

    f = tube_field(6e-3, 4e-3, 0.25e-3, inner_diameter_m=2e-3)
    arrays = F.to_arrays(f, "w0.")
    g = F.from_arrays(arrays, "w0.")
    assert F.fields_equal(f, g)
    assert all(np.asarray(x).flags.c_contiguous and np.asarray(x).flags.writeable for x in g)
    # the checkpoint takes the field's own arrays (no copy: a refined field of 0.7 GB doubled the run's memory)
    assert all(np.shares_memory(arrays["w0." + name], getattr(f, name)) for name in A.DexelField._fields
               if getattr(f, name).size)
    t = F.target_to_arrays(A.TargetField(np.zeros(3, np.int64), np.zeros((0, 2), np.int32), np.zeros(4, np.int64)))
    assert F.target_from_arrays(t).start.flags.writeable


def test_memory_budget_reference_part():
    """300 x 200 x 100 mm at 0.25 mm: arrays of the field (estimate = allocation) within 300 MB."""
    est = F.estimate_bytes([0.3, 0.2, 0.1], 0.25e-3)
    assert est < 300e6
    small = F.make_field([0, 0, 0], [0.03, 0.02, 0.01], 0.25e-3)
    assert abs(F.memory_bytes(small) - F.estimate_bytes([0.03, 0.02, 0.01], 0.25e-3)) / F.memory_bytes(small) < 0.02


def test_one_signature_each():
    for kernel in KR.KERNELS:
        assert len(kernel.signatures) == 1, kernel.__name__
