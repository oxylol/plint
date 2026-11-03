"""Stock and target construction: analytic block / cylinder / tube, watertight mesh rasterisation."""

import math

import numpy as np
import pytest

from cnctwin.core.errors import InputError
from cnctwin.removal import abi as A
from cnctwin.removal import field as F
from cnctwin.removal import k_stock as KS
from cnctwin.removal import reference as R
from cnctwin.removal import stock as S

Q = A.QUANTUM_M


def _box_mesh(lo, hi):
    from cnctwin.geom.primitives import box_mesh

    m = box_mesh(lo, hi)
    return np.asarray(m.vertices), np.asarray(m.faces)


def test_block_is_exact_on_every_family():
    f = S.block_field([5e-3, 4e-3, 3e-3], 0.25e-3)
    for fam in range(3):
        assert F.volume_m3(f, 0, fam) == pytest.approx(6e-8, rel=1e-12)
    # an x ray inside the block holds exactly [0, 5 mm)
    ray = R.ray_index(f, 0, 0, 3, 3)
    (lo, hi, elo, ehi), = R.ray_segments(f, ray)
    assert (hi - lo) * Q == pytest.approx(5e-3, abs=Q)
    assert elo == ehi == A.SEV_STOCK


def test_tube_endpoints_are_exact_chords():
    r_out, r_in = 4e-3, 1.5e-3
    f = S.tube_field(2 * r_out, 3e-3, 0.25e-3, inner_diameter_m=2 * r_in)
    o = f.lv_origin_m[0]
    a = f.lv_cell_m[0]
    for j in range(int(f.lv_dims[0, 1])):
        y = o[1] + j * a
        for k in range(int(f.lv_dims[0, 2])):
            z = o[2] + k * a
            segs = R.ray_segments(f, R.ray_index(f, 0, 0, j, k))
            if not (0.0 <= z < 3e-3) or abs(y) >= r_out:
                assert segs == []
                continue
            ho = math.sqrt(r_out ** 2 - y ** 2)
            xs = [(o[0] + lo * Q, o[0] + hi * Q) for lo, hi, _, _ in segs]
            if abs(y) < r_in:
                hi_ = math.sqrt(r_in ** 2 - y ** 2)
                assert xs[0] == pytest.approx((-ho, -hi_), abs=Q)
                assert xs[1] == pytest.approx((hi_, ho), abs=Q)
            else:
                assert xs == [pytest.approx((-ho, ho), abs=Q)]


def test_mesh_stock_equals_analytic_block_and_target_csr():
    lo, hi = [0, 0, 0], [5e-3, 4e-3, 3e-3]
    V, Fc = _box_mesh(lo, hi)
    fm, info = S.mesh_field(V, Fc, 0.25e-3, box=(lo, hi))
    fb = S.block_field(hi, 0.25e-3)
    assert F.fields_equal(fm, fb)
    tgt, tinfo = S.target_from_mesh(fb, V, Fc)
    assert tinfo["cad_volume_m3"] == pytest.approx(6e-8)
    # target rows equal the field rows
    for ray in range(0, int(fb.info[A.FI_NRAYS]), 37):
        t = [(int(tgt.seg[k, 0]), int(tgt.seg[k, 1])) for k in range(int(tgt.start[ray]), int(tgt.start[ray + 1]))]
        assert t == [(a, b) for a, b, _, _ in R.ray_segments(fb, ray)]


def test_mesh_through_lattice_nodes_is_watertight():
    """Vertices and edges exactly on lattice lines: every ray is counted once (top-left rule)."""
    a = 0.25e-3
    f = F.make_field([0, 0, 0], [4e-3, 4e-3, 4e-3], a)
    o = f.lv_origin_m[0]
    lo = o + 2 * a
    hi = o + 10 * a
    V, Fc = _box_mesh(lo, hi)
    m, _ = S.mesh_field(V, Fc, a, box=([0, 0, 0], [4e-3, 4e-3, 4e-3]))
    vol = F.volume_m3(m, 0, 2)
    assert vol == pytest.approx(64 * a * a * 8 * a, rel=1e-9) or vol == pytest.approx(81 * a * a * 8 * a, rel=1e-9)


def test_open_mesh_is_an_error():
    V, Fc = _box_mesh([0, 0, 0], [2e-3, 2e-3, 2e-3])
    with pytest.raises(InputError):
        S.mesh_field(V, Fc[:-2], 0.25e-3)


def test_inverted_mesh_is_turned_outward():
    V, Fc = _box_mesh([0, 0, 0], [2e-3, 2e-3, 2e-3])
    f, info = S.mesh_field(V, Fc[:, ::-1], 0.25e-3)
    assert F.volume_m3(f, 0, 2) == pytest.approx(8e-9, rel=1e-9)


def test_fill_box_matches_per_ray_reference():
    f = F.make_field([0, 0, 0], [3e-3, 3e-3, 3e-3], 0.25e-3)
    sc = A.make_scratch()
    lo = np.array([0.3e-3, 0.4e-3, 0.55e-3])
    hi = np.array([2.1e-3, 2.6e-3, 1.75e-3])
    assert KS.fill_box(f, 0, lo, hi, 3, sc) == 0
    o, a = f.lv_origin_m[0], f.lv_cell_m[0]
    for fam in range(3):
        u, v = A.TRANS_U[fam], A.TRANS_V[fam]
        for iu in range(int(f.lv_dims[0, u])):
            for iv in range(int(f.lv_dims[0, v])):
                pu, pv = o[u] + iu * a, o[v] + iv * a
                segs = R.ray_segments(f, R.ray_index(f, 0, fam, iu, iv))
                inside = lo[u] <= pu < hi[u] and lo[v] <= pv < hi[v]
                if not inside:
                    assert segs == []
                else:
                    (s0, s1, e0, e1), = segs
                    assert s0 == math.floor((lo[fam] - o[fam]) / Q + 0.5)
                    assert s1 == math.floor((hi[fam] - o[fam]) / Q + 0.5)


def test_one_signature_each():
    for kernel in KS.KERNELS:
        assert len(kernel.signatures) == 1, kernel.__name__
