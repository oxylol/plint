"""Every structure kernel against its plain-Python reference, and one compiled signature each."""

from __future__ import annotations

import numpy as np
import pytest

from cnctwin.structure import hexfe
from cnctwin.structure import kernels as K
from cnctwin.structure import reference as R


@pytest.fixture(scope="module")
def lattice():
    rng = np.random.default_rng(3)
    active = rng.random((4, 3, 3)) > 0.3
    active[0, 0, 0] = True
    ok = np.zeros((5, 4, 4), bool)
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                ok[di:di + 4, dj:dj + 3, dk:dk + 3] |= active
    ijk = np.ascontiguousarray(np.argwhere(ok).astype(np.int64))
    index = np.full(ok.shape, -1, np.int64)
    index[tuple(ijk.T)] = np.arange(len(ijk))
    cells = np.ascontiguousarray(np.argwhere(active).astype(np.int64))
    return active, ok, ijk, index, cells


def _pattern(active, ijk, index):
    nbr = np.zeros((len(ijk), 27), np.int64)
    total = K.neighbour_table(np.ascontiguousarray(active.astype(np.int8)), index, ijk, nbr)
    counts = np.count_nonzero(nbr >= 0, axis=1)
    indptr = np.zeros(3 * len(ijk) + 1, np.int64)
    np.cumsum(np.repeat(3 * counts, 3), out=indptr[1:])
    indices = np.zeros(int(indptr[-1]), np.int32)
    K.fill_indices(ijk, index, nbr, indptr, indices)
    return nbr, total, indptr, indices


def test_node_grid_sampling_and_material_points():
    rng = np.random.default_rng(1)
    grid = np.ascontiguousarray(rng.standard_normal((4, 3, 5, 3)) * 1e-5)
    origin = np.array([0.01, -0.02, 0.0])
    cell = 0.004
    pts = np.ascontiguousarray(origin + rng.random((50, 3)) * np.array([0.016, 0.012, 0.02]) * 1.2 - 0.002)
    out = np.zeros((50, 3))
    K.sample_node_grid(origin, cell, grid, pts, out)
    assert np.allclose(out, R.sample_node_grid_ref(origin, cell, grid, pts), rtol=1e-13, atol=1e-20)
    # exact at the nodes
    node = origin + np.array([2, 1, 3]) * cell
    one = np.zeros((1, 3))
    K.sample_node_grid(origin, cell, grid, node[None, :].copy(), one)
    assert np.allclose(one[0], grid[2, 1, 3], rtol=1e-12)
    X = np.zeros((50, 3))
    K.material_points(origin, cell, grid, pts, 3, X)
    assert np.allclose(X, R.material_points_ref(origin, cell, grid, pts, 3), rtol=1e-13, atol=1e-20)
    K.material_points(origin, cell, np.zeros((0, 0, 0, 3)), pts, 1, X)
    assert np.array_equal(X, pts)


def test_pattern_and_assembly_against_dense_loops(lattice):
    active, ok, ijk, index, cells = lattice
    nbr, total, indptr, indices = _pattern(active, ijk, index)
    nbr_ref, total_ref = R.neighbour_table_ref(active, index, ijk)
    assert total == total_ref and np.array_equal(nbr, nbr_ref)
    Ke, Me, _ = hexfe.matrices(0.002, 70e9, 0.33, 2700.0)
    scale = np.linspace(0.5, 1.5, len(cells))
    data = np.zeros(len(indices))
    K.assemble_values(cells, index, nbr, indptr, np.ascontiguousarray(Ke), scale, data)
    import scipy.sparse as sp
    A = sp.csr_matrix((data, indices, indptr), shape=(3 * len(ijk),) * 2).toarray()
    D = R.dense_assembly_ref(cells, index, len(ijk), Ke, scale)
    assert np.allclose(A, D, rtol=1e-12, atol=1e-6)
    # springs at points spread by trilinear weights
    rng = np.random.default_rng(5)
    pc = np.ascontiguousarray(cells[:4])
    w = rng.random((4, 8))
    w /= w.sum(axis=1, keepdims=True)
    kmat = np.ascontiguousarray(rng.standard_normal((4, 3, 3)))
    data2 = np.zeros(len(indices))
    K.add_point_springs(index, nbr, indptr, pc, np.ascontiguousarray(w), kmat, data2)
    A2 = sp.csr_matrix((data2, indices, indptr), shape=(3 * len(ijk),) * 2).toarray()
    assert np.allclose(A2, R.point_springs_ref(index, len(ijk), pc, w, kmat), rtol=1e-12, atol=1e-14)


def test_scatter_and_cell_average(lattice):
    active, ok, ijk, index, cells = lattice
    rng = np.random.default_rng(2)
    coef = np.ascontiguousarray(rng.standard_normal((len(cells), 6)))
    Ge = np.ascontiguousarray(rng.standard_normal((24, 6)))
    out = np.zeros(3 * len(ijk))
    K.scatter_cell_vectors(cells, index, coef, Ge, out)
    assert np.allclose(out, R.scatter_cell_vectors_ref(cells, index, coef, Ge, 3 * len(ijk)), rtol=1e-12)
    vals = np.ascontiguousarray(rng.standard_normal(ok.shape + (3,)))
    cav = np.zeros((4, 3, 3, 3))
    has = np.zeros((4, 3, 3), np.int8)
    K.cell_average(vals, np.ascontiguousarray(ok.astype(np.int8)), cav, has)
    ref, has_ref = R.cell_average_ref(vals, ok)
    assert np.array_equal(has, has_ref) and np.allclose(cav, ref, rtol=1e-12, atol=1e-15)


def test_one_signature_per_kernel():
    for kernel in K.KERNELS:
        assert len(kernel.signatures) == 1, kernel.__name__
