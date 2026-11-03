"""3D-DDA against per-cell slab tests (brute force), first-occupied queries and batches."""

import math

import numpy as np

from cnctwin.geom import dda, reference as ref
from cnctwin.geom import transform as tf

RNG = np.random.default_rng(9)


def test_axis_aligned_and_diagonal_walks():
    g = dda.grid([0, 0, 0], 1.0, [4, 3, 2])
    ijk, t = dda.ray_cells(g, [-1, 0.5, 0.5], [1, 0, 0])
    assert ijk.tolist() == [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]
    assert np.allclose(t, [[1, 2], [2, 3], [3, 4], [4, 5]])
    ijk, t = dda.ray_cells(g, [-1, -1, 0.25], [1, 1, 0])
    # through the cell corners: the edge-touched cells appear with zero length (conservative)
    assert ijk.tolist() == [[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0], [2, 2, 0]]
    assert t[1, 0] == t[1, 1] == 2.0
    ijk, t = dda.segment_cells(g, [0.5, 0.5, 0.5], [0.6, 0.5, 0.5])
    assert ijk.tolist() == [[0, 0, 0]] and np.allclose(t, [[0.0, 1.0]])
    assert dda.ray_cells(g, [-1, 5, 0.5], [1, 0, 0])[0].shape == (0, 3)


def test_walk_matches_brute_force_cells():
    lo = np.array([0.1, -0.2, 0.05])
    cell = np.array([0.013, 0.021, 0.017])
    dims = [7, 5, 6]
    g = dda.grid(lo, cell, dims)
    for _ in range(300):
        o = lo + RNG.uniform(-0.05, 0.15, 3)
        d = tf.unit(RNG.normal(size=3))
        t1 = RNG.uniform(0.01, 0.3)
        ijk, t = dda.ray_cells(g, o, d, 0.0, t1)
        expected = ref.dda_cells_ref(o, d, 0.0, t1, lo, cell, dims)
        got = [(float(a), float(b), tuple(c)) for (a, b), c in zip(t, ijk.tolist()) if b - a > 1e-12]
        exp = [e for e in expected if e[1] - e[0] > 1e-12]
        assert [c for _, _, c in got] == [c for _, _, c in exp]
        for (a, b, _), (ea, eb, _) in zip(got, exp):
            assert math.isclose(a, ea, abs_tol=1e-12) and math.isclose(b, eb, abs_tol=1e-12)
        # consecutive cells are face, edge or corner neighbours and the intervals chain
        if len(ijk) > 1:
            assert np.all(np.abs(np.diff(ijk, axis=0)).sum(axis=1) == 1)
            assert np.allclose(t[1:, 0], t[:-1, 1])


def test_first_occupied_and_batch():
    g = dda.grid([0, 0, 0], [0.01, 0.01, 0.01], [10, 10, 10])
    occ = np.zeros((10, 10, 10), np.uint8)
    occ[6, 3, 2] = 1
    occ[8, 3, 2] = 1
    o = np.array([0.0005, 0.035, 0.025])
    hit = dda.first_occupied(g, occ, o, [1, 0, 0])
    assert hit[0] == (6, 3, 2) and math.isclose(hit[1], 0.0595, rel_tol=1e-12)
    assert dda.first_occupied(g, occ, o, [1, 0, 0], 0.0, 0.05) is None
    assert dda.first_occupied(g, occ, o, [-1, 0, 0]) is None
    found, rows = dda.first_occupied_batch(g, occ, [o, o, [0.095, 0.035, 0.025]], [[1, 0, 0], [0, 1, 0], [-1, 0, 0]])
    assert found.tolist() == [1, 0, 1]
    assert rows[0, :3].tolist() == [6, 3, 2] and rows[2, :3].tolist() == [8, 3, 2]
    assert np.all(np.isnan(rows[1]))


def test_walks_agree_with_occupied_queries():
    g = dda.grid([0, 0, 0], 0.005, [12, 12, 12])
    occ = (RNG.uniform(size=(12, 12, 12)) < 0.05).astype(np.uint8)
    for _ in range(200):
        o = RNG.uniform(-0.01, 0.07, 3)
        d = tf.unit(RNG.normal(size=3))
        ijk, t = dda.ray_cells(g, o, d)
        first = next(((tuple(c), a, b) for c, (a, b) in zip(ijk.tolist(), t) if occ[tuple(c)]), None)
        hit = dda.first_occupied(g, occ, o, d)
        if first is None:
            assert hit is None
        else:
            assert hit[0] == first[0] and hit[1] == first[1] and hit[2] == first[2]


def test_each_kernel_has_one_signature():
    for kernel in dda.KERNELS:
        assert len(kernel.signatures) == 1, kernel.__name__
