"""Connectivity after subtraction: loose bodies (volume, centroid), held regions, removal from every family."""

import math

import numpy as np
import pytest

from cnctwin.core.abi import event_buffer
from cnctwin.removal import abi as A
from cnctwin.removal import connect as CN
from cnctwin.removal import field as F
from cnctwin.removal import k_connect as KC
from cnctwin.removal import k_subtract as KS
from cnctwin.removal import profile as P
from cnctwin.removal import reference as R
from cnctwin.removal import stock as S

LAT = 0.25e-3


def _trepan(f, centre, r_mean, width, z_bottom):
    """Cut a ring (trepanning) with a small cutter stepped around the circle: frees the core."""
    prof = P.profile_from_points([0.0, 20e-3], [0.5 * width, 0.5 * width])
    acc = A.new_accum(F.n_levels(f))
    sc = A.make_scratch()
    tgt = A.empty_target(int(f.info[A.FI_NRAYS]))
    ev = event_buffer(16, A.EVENT_WIDTH)
    n = int(math.ceil(2 * math.pi * r_mean / (0.2 * LAT)))
    for k in range(n):
        ph = 2 * math.pi * k / n
        c = [centre[0] + r_mean * math.cos(ph), centre[1] + r_mean * math.sin(ph), z_bottom]
        spec = A.subtract_spec([0, 0, 1], [c], [0.0, 20e-3], [0.0], [-1] * 3, [1] * 3)
        assert KS.subtract_envelope(f, prof, spec, tgt, acc, ev, sc) == 0
    return acc


def test_through_ring_frees_the_core_and_it_is_removed():
    f = S.block_field([16e-3, 16e-3, 4e-3], LAT)
    _trepan(f, (8e-3, 8e-3), 4e-3, 1.2e-3, -0.5e-3)
    assert f.info[A.FI_TOPO] > 0
    bodies, first, parent, total = CN.find_bodies(f)
    ref = R.components_ref(f)
    assert len(bodies) == len(ref) == 2
    for b, (v, c) in zip(bodies, ref):
        assert b.volume_m3 == pytest.approx(v, rel=1e-12)
        assert np.allclose(b.centroid_S_m, c, atol=1e-12)
    core = bodies[1]
    r_core = 4e-3 - 0.6e-3
    assert core.volume_m3 == pytest.approx(math.pi * r_core ** 2 * 4e-3, rel=0.03)
    assert np.allclose(core.centroid_S_m[:2], [8e-3, 8e-3], atol=LAT)
    vol_before = [F.volume_m3(f, 0, fam) for fam in range(3)]
    released, held, removed = CN.release_loose(f, [((-1, -1, -1), (1e-3, 20e-3, 20e-3))])
    assert len(released) == 1 and released[0].volume_m3 == pytest.approx(core.volume_m3)
    assert removed * F.volume_factor_m3(f, 0) == pytest.approx(core.volume_m3, rel=1e-12)
    for fam in range(3):
        assert vol_before[fam] - F.volume_m3(f, 0, fam) == pytest.approx(core.volume_m3, rel=0.03)
    assert len(CN.find_bodies(f)[0]) == 1
    assert f.info[A.FI_TOPO] == 0


def test_blind_ring_keeps_one_body():
    f = S.block_field([16e-3, 16e-3, 4e-3], LAT)
    _trepan(f, (8e-3, 8e-3), 4e-3, 1.2e-3, 1.0e-3)
    released, held, removed = CN.release_loose(f)
    assert released == [] and len(held) == 1


def test_held_region_keeps_the_held_body_even_when_smaller():
    f = S.block_field([16e-3, 16e-3, 4e-3], LAT)
    _trepan(f, (8e-3, 8e-3), 4e-3, 1.2e-3, -0.5e-3)
    released, held, _ = CN.release_loose(f, [((7.5e-3, 7.5e-3, -1.0), (8.5e-3, 8.5e-3, 1.0))])
    assert len(held) == 1 and held[0].volume_m3 < released[0].volume_m3


def test_one_signature_each():
    for kernel in KC.KERNELS:
        assert len(kernel.signatures) == 1, kernel.__name__
