"""Every inspection kernel against its plain-Python reference, and one compiled specialisation each."""

import math

import numpy as np
import pytest

from cnctwin.geom.bvh import build_tree, rays_all_hits
from cnctwin.geom.primitives import box_mesh, cylinder_mesh, sphere_mesh
from cnctwin.inspect import abi as A
from cnctwin.inspect import kernels as K
from cnctwin.inspect import reference as R
from cnctwin.inspect.columns import columns_from_mesh, difference, regions

from tests.unit.inspect.inspfix import pocket_block


def test_one_signature_each():
    for kernel in K.KERNELS:
        assert len(kernel.signatures) == 1, kernel.__name__


def test_trilinear_matches_reference():
    rng = np.random.default_rng(3)
    grid = A.contiguous(rng.normal(size=(4, 3, 5, 3)))
    origin = np.array([0.1, -0.2, 0.05])
    cell = 0.01
    P = A.contiguous(np.column_stack([rng.uniform(0.05, 0.16, 40), rng.uniform(-0.25, -0.17, 40),
                                      rng.uniform(0.0, 0.11, 40)]))
    out = np.zeros_like(P)
    K.trilinear_nb(P, origin, cell, grid, out)
    np.testing.assert_allclose(out, R.trilinear_ref(P, origin, cell, grid), rtol=0, atol=1e-14)
    # at a node the value is the node's
    node = A.contiguous((origin + np.array([2, 1, 3]) * cell)[None, :])
    K.trilinear_nb(node, origin, cell, grid, out[:1])
    np.testing.assert_allclose(out[0], grid[2, 1, 3], rtol=0, atol=1e-14)


def test_deviation_matches_brute_force():
    mesh = sphere_mesh([0.0, 0.0, 0.0], 0.01, chordal_tol_m=2e-4)
    tree = build_tree(mesh)
    rng = np.random.default_rng(5)
    P = A.contiguous(rng.uniform(-0.015, 0.015, (30, 3)))
    d = np.empty(30)
    tri = np.empty(30, np.int32)
    q = np.empty((30, 3))
    feat = np.empty(30, np.int8)
    K.deviation_nb(tree, P, np.inf, d, tri, q, feat)
    d_ref, q_ref = R.deviation_ref(mesh, P)
    np.testing.assert_allclose(d, d_ref, rtol=0, atol=1e-13)
    np.testing.assert_allclose(q, q_ref, rtol=0, atol=1e-13)


def test_columns_from_hits_matches_reference():
    mesh = pocket_block([0, 0, 0], [0.02, 0.016, 0.01], [0.005, 0.004], [0.015, 0.012], 0.004)
    tree = build_tree(mesh)
    xs, ys = np.meshgrid(np.arange(-1, 22) * 1e-3, np.arange(-1, 18) * 1e-3, indexing="ij")
    O = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, -0.001)])
    offsets, t, face = rays_all_hits(tree, O, [0.0, 0.0, 1.0])
    nz = A.contiguous(tree.face_normal[:, 2])
    start = np.zeros(len(O) + 1, np.int64)
    seg = np.zeros((len(t) // 2 + 1, 2))
    m = K.columns_from_hits_nb(offsets, t, face, nz, -0.001, 1e-12, start, seg)
    ref = R.columns_from_hits_ref(offsets, t, face, nz, -0.001, 1e-12)
    got = [[tuple(seg[k]) for k in range(start[c], start[c + 1])] for c in range(len(O))]
    assert m == sum(len(r) for r in ref)
    for a, b in zip(got, ref):
        assert len(a) == len(b)
        for (lo, hi), (rlo, rhi) in zip(a, b):
            assert lo == pytest.approx(rlo, abs=1e-15) and hi == pytest.approx(rhi, abs=1e-15)
    # rays exactly through the pocket's wall corners and edges still give the solid's intervals
    cols = columns_from_mesh(mesh, [0.0, 0.0], [1e-3, 1e-3], [21, 17])
    c = 5 * 17 + 4                                       # x = 5 mm, y = 4 mm: on the pocket corner edge
    assert len(cols.seg[cols.start[c]:cols.start[c + 1]]) == 1


def test_difference_matches_reference():
    rng = np.random.default_rng(11)

    def random_columns(n):
        start = [0]
        segs = []
        for _ in range(n):
            z = np.sort(rng.uniform(0, 1, 2 * rng.integers(0, 4)))
            pairs = z.reshape(-1, 2)
            segs.extend(pairs.tolist())
            start.append(len(segs))
        return A.make_columns([0, 0], [1, 1], [n, 1], start, np.array(segs).reshape(-1, 2))

    a, b = random_columns(60), random_columns(60)
    out = difference(a, b, 0.01)
    ref = R.columns_difference_ref([[tuple(a.seg[k]) for k in range(a.start[c], a.start[c + 1])] for c in range(60)],
                                   [[tuple(b.seg[k]) for k in range(b.start[c], b.start[c + 1])] for c in range(60)],
                                   0.01)
    for c in range(60):
        got = [tuple(out.seg[k]) for k in range(out.start[c], out.start[c + 1])]
        assert len(got) == len(ref[c])
        for g, r in zip(got, ref[c]):
            assert g == pytest.approx(r, abs=1e-15)


def test_components_match_reference():
    rng = np.random.default_rng(2)
    nx, ny = 12, 9
    start = [0]
    segs = []
    for c in range(nx * ny):
        if rng.uniform() < 0.35:
            lo = rng.uniform(0, 0.5)
            segs.append([lo, lo + rng.uniform(0.05, 0.5)])
            if rng.uniform() < 0.3:
                segs.append([1.2, 1.4])
        start.append(len(segs))
    cols = A.make_columns([0.0, 0.0], [0.5, 0.25], [nx, ny], start, np.array(segs))
    parent = np.zeros(len(segs), np.int64)
    label = np.zeros(len(segs), np.int64)
    count = K.label_components_nb(cols.dims, cols.start, cols.seg, parent, label)
    ref = R.components_ref(cols)
    assert count == len(ref)
    got = {}
    for s, lab in enumerate(label):
        got.setdefault(int(lab), set()).add(s)
    assert sorted(map(sorted, got.values())) == sorted(map(sorted, ref))
    stats = np.zeros((count, A.N_CS))
    K.component_stats_nb(cols.origin_m, cols.cell_m, cols.dims, cols.start, cols.seg, label, np.zeros(count, np.int64),
                         stats)
    for lab, members in got.items():
        r = R.component_stats_ref(cols, members)
        assert stats[lab, A.CS_VOLUME] == pytest.approx(r["volume_m3"], rel=1e-14)
        np.testing.assert_allclose(stats[lab, A.CS_CX:A.CS_CZ + 1], r["centroid_m"], rtol=1e-13, atol=1e-15)
        assert stats[lab, A.CS_THICK] == pytest.approx(r["thickness_m"], rel=1e-15)
    rows = regions(cols)
    assert np.all(np.diff(rows[:, A.CS_VOLUME]) <= 0.0)


def test_segment_point_matches_reference():
    row = np.zeros(A.N_SG)
    row[A.SG_KIND] = A.SEG_ARC
    row[A.SG_P0:A.SG_P0 + 3] = [0.01, 0.0, 0.0]
    row[A.SG_P1:A.SG_P1 + 3] = [0.0, 0.012, 0.002]
    row[A.SG_C:A.SG_C + 3] = [0.0, 0.0, 0.0]
    row[A.SG_N:A.SG_N + 3] = [0.0, 0.0, 1.0]
    row[A.SG_SWEEP] = 0.5 * math.pi
    seg = A.contiguous(row[None, :])
    out = np.zeros(3)
    for s in (0.0, 0.3, 0.71, 1.0):
        K.segment_point_nb(seg, 0, s, out)
        np.testing.assert_allclose(out, R.segment_point_ref(row, s), rtol=0, atol=1e-15)
    K.segment_point_nb(seg, 0, 1.0, out)
    np.testing.assert_allclose(out, [0.0, 0.012, 0.002], atol=1e-15)


def _chord(a0, a1, r):
    seg = np.zeros((1, A.N_SG))
    seg[0, A.SG_P0:A.SG_P0 + 3] = [r * math.cos(a0), r * math.sin(a0), 0.0]
    seg[0, A.SG_P1:A.SG_P1 + 3] = [r * math.cos(a1), r * math.sin(a1), 0.0]
    return seg


def test_chord_deviation_matches_dense_reference():
    # kernel against brute-force sampling on a coarse cylinder (the reference is O(triangles) per sample)
    coarse = cylinder_mesh([0, 0, -0.01], [0, 0, 0.01], 0.02, chordal_tol_m=2e-4)
    seg = _chord(0.1, 0.1 + math.radians(30.0), 0.022)
    out = np.zeros((1, A.N_CH))
    K.chord_deviation_nb(build_tree(coarse), seg, 8, 40, 0.01, out)
    lo, hi = R.chord_error_ref(coarse, seg[0], samples=121)
    # the refined extremes are values of e(s) (brute force at the kernel's s) at least as extreme as dense samples
    assert R.chord_error_at(coarse, seg[0], out[0, A.CH_SMIN]) == pytest.approx(out[0, A.CH_EMIN], abs=1e-12)
    assert out[0, A.CH_EMIN] <= lo + 1e-12
    assert out[0, A.CH_EMIN] == pytest.approx(lo, abs=2e-6)       # facets of 0.2 mm chord error: kinks in e(s)
    assert out[0, A.CH_EMAX] >= hi - 1e-12


def test_chord_deviation_is_the_sagitta():
    # a chord of 30 deg on the 22 mm offset circle of a finely tessellated cylinder: sagitta 22 (1 - cos 15 deg) mm
    fine = cylinder_mesh([0, 0, -0.01], [0, 0, 0.01], 0.02, chordal_tol_m=1e-6)
    seg = _chord(0.1, 0.1 + math.radians(30.0), 0.022)
    out = np.zeros((1, A.N_CH))
    K.chord_deviation_nb(build_tree(fine), seg, 8, 40, 0.01, out)
    assert out[0, A.CH_EMIN] == pytest.approx(-0.022 * (1 - math.cos(math.radians(15.0))), rel=2e-3)
    # the facets (1 um chords) ripple the flat bottom of e(s): its argmin moves by sqrt(1e-6 / (4 sagitta))
    assert out[0, A.CH_SMIN] == pytest.approx(0.5, abs=0.03)
    assert out[0, A.CH_LEN] == pytest.approx(2 * 0.022 * math.sin(math.radians(15.0)), rel=1e-12)


def test_probe_matches_reference():
    mesh = box_mesh([0, 0, 0], [0.02, 0.02, 0.01])
    tree = build_tree(mesh)
    O = A.contiguous(np.array([[0.01, 0.01, 0.015], [0.004, 0.012, 0.03], [0.03, 0.01, 0.005], [0.01, 0.01, 0.05]]))
    D = A.contiguous(np.array([[0.0, 0.0, -1.0], [0.3, -0.2, -1.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    out = np.zeros((4, A.N_PR))
    K.probe_nb(tree, O, D, 0.001, 0.05, 1e-12, 400, out)
    for i in range(3):
        ref = R.probe_ref(mesh, O[i], D[i], 0.001, 0.05)
        assert out[i, A.PR_STATUS] == A.PROBE_CONTACT
        assert out[i, A.PR_T] == pytest.approx(ref, abs=1e-9)
    assert out[0, A.PR_T] == pytest.approx(0.004, abs=1e-12)          # 15 mm - 10 mm - ball 1 mm
    np.testing.assert_allclose(out[0, A.PR_QX:A.PR_QZ + 1], [0.01, 0.01, 0.01], atol=1e-12)
    assert out[3, A.PR_STATUS] == A.PROBE_NO_CONTACT
