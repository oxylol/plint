"""Collisions (non-cutting envelope parts, meshes, swept meshes, envelope vs fixture) and the tap sweep."""

import math

import numpy as np
import pytest

from cnctwin.core.abi import event_buffer
from cnctwin.removal import abi as A
from cnctwin.removal import collision as CO
from cnctwin.removal import field as F
from cnctwin.removal import k_collide as KC
from cnctwin.removal import k_helix as KH
from cnctwin.removal import profile as P
from cnctwin.removal import reference as R
from cnctwin.removal import stock as S
from cnctwin.tooling.abi import PART_HOLDER, PART_SHANK

LAT = 0.25e-3


def _box(lo, hi):
    from cnctwin.geom.primitives import box_mesh

    m = box_mesh(lo, hi)
    return np.asarray(m.vertices), np.asarray(m.faces, np.int32)


def test_shank_in_material_is_a_collision():
    f = S.block_field([20e-3, 20e-3, 10e-3], LAT)
    cut = P.profile_from_points([0.0, 5e-3], [3e-3, 3e-3])
    whole = P.build_profile([P.Piece(A.PK_LINE, 0.0, 5e-3, 3e-3, 0.0, 0, 0, 0, 0),
                             P.Piece(A.PK_LINE, 5e-3, 40e-3, 3e-3, 0.0, 0, 0, PART_SHANK, 1),
                             P.Piece(A.PK_LINE, 40e-3, 60e-3, 12e-3, 0.0, 0, 0, PART_HOLDER, 2)])
    acc = A.new_accum(1)
    sc = A.make_scratch()
    axis = np.array([0.0, 0.0, 1.0])
    # tip 8 mm deep in a 10 mm block: the shank (z_T 5..40) is in material from z_T 5 to 8 mm
    ev = event_buffer(8, A.EVENT_WIDTH)
    n = KC.collide_envelope(f, whole, cut, axis, np.array([[10e-3, 10e-3, 2e-3]]), axis, 5, 0, 1e-5, acc, ev, sc)
    assert n > 0 and ev.n[0] == 1
    assert int(ev.rec[0, A.EC_KIND]) == A.EV_COLLISION and int(ev.rec[0, A.EC_PART]) == PART_SHANK
    assert ev.rec[0, A.EC_DEPTH] == pytest.approx(6e-3, abs=LAT)    # chord of the shank at the axis
    # above the block: nothing
    ev2 = event_buffer(8, A.EVENT_WIDTH)
    assert KC.collide_envelope(f, whole, cut, axis, np.array([[10e-3, 10e-3, 6e-3]]), axis, 5, 0, 1e-5, acc, ev2,
                               sc) == 0
    assert ev2.n[0] == 0


def test_mesh_overlap_and_swept_tunnelling():
    f = S.block_field([20e-3, 20e-3, 1e-3], LAT)          # a 1 mm plate
    V, Fc = _box([-2e-3, -2e-3, -2e-3], [2e-3, 2e-3, 2e-3])
    hit = CO.mesh_vs_field(f, V + np.array([10e-3, 10e-3, 0.5e-3]), Fc, 1e-5)
    assert hit["hit"] and hit["volume_m3"] == pytest.approx(16e-6 * 1e-3, rel=0.02)
    assert not CO.mesh_vs_field(f, V + np.array([10e-3, 10e-3, 10e-3]), Fc, 1e-5)["hit"]
    # a fast move from below the plate to above it: both end poses are free, the sweep is not
    P0 = np.eye(4)
    P0[:3, 3] = [10e-3, 10e-3, -10e-3]
    P1 = np.eye(4)
    P1[:3, 3] = [10e-3, 10e-3, 12e-3]
    assert not CO.mesh_vs_field(f, V + P0[:3, 3], Fc, 1e-5)["hit"]
    assert not CO.mesh_vs_field(f, V + P1[:3, 3], Fc, 1e-5)["hit"]
    res = CO.swept_mesh_vs_field(f, V, Fc, P0, P1, 0.5 * LAT, 1e-5)
    assert res["hit"] and 0.3 < res["fraction"] < 0.6


def test_envelope_vs_fixture_mesh():
    prof = P.profile_from_points([0.0, 30e-3], [5e-3, 5e-3])
    V, Fc = _box([20e-3, -10e-3, 0.0], [30e-3, 10e-3, 20e-3])        # a vise jaw
    res = CO.envelope_vs_mesh(prof, [0.0, 0.0, 5e-3], [30e-3, 0.0, 5e-3], [0, 0, 1], V, Fc, 0.1e-3)
    assert res["fraction"] == pytest.approx(15e-3 / 30e-3, abs=0.1e-3 / 30e-3 + 1e-9)
    res = CO.envelope_vs_mesh(prof, [0.0, 0.0, 5e-3], [10e-3, 0.0, 5e-3], [0, 0, 1], V, Fc, 0.1e-3, margin_m=10e-3)
    assert res["fraction"] < 0 and res["clearance_m"] == pytest.approx(5e-3, rel=1e-6)


def _single_point_tap(r_hole, depth, half_deg=30.0):
    t = math.tan(math.radians(half_deg))
    eps = 0.1e-3
    poly = np.array([[[-(depth + eps) * t, r_hole - eps], [(depth + eps) * t, r_hole - eps], [0.0, r_hole + depth]]])
    box = np.array([[poly[0, :, 0].min(), poly[0, :, 0].max(), poly[0, :, 1].min(), poly[0, :, 1].max()]])
    info = np.array([-1.0, box[0, 0], box[0, 1], box[0, 3]])
    return A.HelixSweep(poly, np.array([3], np.int32), box, np.array([0, 1], np.int32), np.array([0.0]), info)


def test_tap_sweep_cuts_a_helical_groove():
    """One V tooth swept along a helix of pitch 2 mm in a 6 mm bore: groove volume by Pappus."""
    r_h, depth, pitch = 3e-3, 0.6e-3, 2e-3
    a = LAT
    f = S.block_field([12e-3, 12e-3, 8e-3], a)
    from cnctwin.removal.k_subtract import subtract_envelope

    bore = P.profile_from_points([0.0, 20e-3], [r_h, r_h])
    subtract_envelope(f, bore, A.subtract_spec([0, 0, 1], [[6e-3, 6e-3, -1e-3]], [0.0, 20e-3], [0.0], [-1] * 3,
                                               [1] * 3), A.empty_target(int(f.info[A.FI_NRAYS])), A.new_accum(1),
                      event_buffer(4, A.EVENT_WIDTH), A.make_scratch())
    hx = _single_point_tap(r_h, depth)
    v0 = F.volume_m3(f, 0, 2)
    acc = A.new_accum(1)
    ev = event_buffer(8, A.EVENT_WIDTH)
    sc = A.make_scratch()
    x0 = np.array([1.0, 0.0, 0.0])
    y0 = np.array([0.0, 1.0, 0.0])
    w = np.array([0.0, 0.0, 1.0])
    turns = 2
    steps = 4 * turns
    z = 6.5e-3
    for k in range(steps):
        th_a, th_b = -k * math.pi / 2, -(k + 1) * math.pi / 2
        pa = np.array([6e-3, 6e-3, z - k * pitch / 4])
        pb = np.array([6e-3, 6e-3, z - (k + 1) * pitch / 4])
        assert KH.helix_subtract(f, hx, pa, pb, w, x0, y0, th_a, th_b, np.full(3, -1.0), np.full(3, 1.0), 9, k, 0,
                                 A.empty_target(int(f.info[A.FI_NRAYS])), acc, ev, sc) == 0
    removed = v0 - F.volume_m3(f, 0, 2)
    t = math.tan(math.radians(30))
    area = depth * depth * t
    expect = turns * area * 2 * math.pi * (r_h + depth / 3)
    assert removed == pytest.approx(expect, rel=0.05)
    # a point on the helix is cut, the point half a pitch away is not
    g_in = KH.helix_inside(hx, 6e-3 + r_h + 0.2e-3, 6e-3, z - 0.0, 0.0, -2 * math.pi, 0.0, np.array([6e-3, 6e-3, z]),
                           np.array([6e-3, 6e-3, z - pitch]), w, x0, y0, -1.0, hx.info[1], hx.info[2])
    ref = R.helix_inside_ref(hx, [6e-3 + r_h + 0.2e-3, 6e-3, z], 0.0, -2 * math.pi, np.array([6e-3, 6e-3, z]),
                             np.array([6e-3, 6e-3, z - pitch]), w, x0, y0)
    assert g_in == pytest.approx(ref, abs=1e-12) and g_in <= 0.0      # passed at theta = 0, at its height
    g_out = KH.helix_inside(hx, 6e-3 + r_h + 0.2e-3, 6e-3, z + pitch / 2, 0.0, -2 * math.pi, 0.0,
                            np.array([6e-3, 6e-3, z]), np.array([6e-3, 6e-3, z - pitch]), w, x0, y0, -1.0,
                            hx.info[1], hx.info[2])
    assert g_out > 0.0


def test_one_signature_each():
    for kernel in KC.KERNELS + KH.KERNELS:
        assert len(kernel.signatures) == 1, kernel.__name__
