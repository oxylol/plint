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


def _cut_path(f, pts, z_bottom, width, height=20e-3):
    """Subtract a small cutter (diameter ``width``, ``height`` tall from z_bottom) at every point of pts."""
    prof = P.profile_from_points([0.0, height], [0.5 * width, 0.5 * width])
    acc = A.new_accum(F.n_levels(f))
    sc = A.make_scratch()
    tgt = A.empty_target(int(f.info[A.FI_NRAYS]))
    ev = event_buffer(16, A.EVENT_WIDTH)
    for x, y in pts:
        spec = A.subtract_spec([0, 0, 1], [[x, y, z_bottom]], [0.0, height], [0.0], [-1] * 3, [1] * 3)
        assert KS.subtract_envelope(f, prof, spec, tgt, acc, ev, sc) == 0
        yield


def _incremental_vs_full(f, steps, held):
    """Run the incremental check (the domain's per-macro-step rule: only when a z column changed its segment count)
    and, on a copy, the full labelling after every step; returns the step and volumes of every release of each."""
    g = F.copy_field(f)
    one_body = False
    inc, full, local, checks = [], [], 0, 0
    for k, _ in enumerate(steps(f, g)):
        if f.info[A.FI_TOPO] > 0:
            rel, _h, _r, one_body, was_local = CN.release_loose_incremental(f, held, one_body)
            checks += 1
            local += was_local
            if rel:
                inc.append((k, sorted(round(b.volume_m3 * 1e12) for b in rel)))
        rel_g, _h, _r = CN.release_loose(g, held)
        if rel_g:
            full.append((k, sorted(round(b.volume_m3 * 1e12) for b in rel_g)))
    return inc, full, local, checks


def test_incremental_check_separates_exactly_at_the_through_cut_and_the_last_tab():
    """The per-macro-step check labels only the columns around the changed bricks while the material is one body:
    a trepanned core comes loose at exactly the step that closes the ring through the stock, a part held by two
    tabs at exactly the step that cuts through the last one, a blind groove frees nothing; the released bodies are
    those of the full labelling. The local check decides when the changed region shows one body reaching its
    ring (every step of the groove); a through cut whose two sides join only outside it falls back to the whole
    field."""
    held = [((-1.0, -1.0, -1.0), (1e-3, 1.0, 1.0))]           # the vise jaw at the block's left face
    width = 1.0e-3

    def ring(z_bottom, height=20e-3):
        n = int(math.ceil(2 * math.pi * 4e-3 / (0.2 * LAT)))
        pts = [(20e-3 + 4e-3 * math.cos(2 * math.pi * k / n), 20e-3 + 4e-3 * math.sin(2 * math.pi * k / n))
               for k in range(n + 1)]

        def steps(f, g):
            for _ in zip(_cut_path(f, pts, z_bottom, width, height), _cut_path(g, pts, z_bottom, width, height)):
                yield
        return steps, n

    # through ring: the core separates when the ring closes (and only then)
    f = S.block_field([40e-3, 40e-3, 4e-3], LAT)
    steps, n = ring(-0.5e-3)
    inc, full, local, checks = _incremental_vs_full(f, steps, held)
    assert len(full) == 1 and inc == full
    assert full[0][0] > 0.9 * n                                # the ring's last steps cut the core free
    assert 0 < local < checks
    # a groove from 1 to 3 mm (columns split in two) on the same ring: nothing comes loose
    f = S.block_field([40e-3, 40e-3, 4e-3], LAT)
    steps, n = ring(1.0e-3, 2.0e-3)
    inc, full, local, checks = _incremental_vs_full(f, steps, held)
    assert inc == full == [] and checks > 100 and local == checks - 1   # the first check labels everything
    # a 12 x 12 mm square cut through around (20, 20) leaving tabs in the middle of its left and right sides,
    # then the tabs cut one after the other: the part separates with the second tab
    f = S.block_field([40e-3, 40e-3, 4e-3], LAT)
    lo, hi, tab = 14e-3, 26e-3, 2e-3
    d = 0.2 * LAT
    side = lambda a, b: [a + (b - a) * t for t in np.linspace(0.0, 1.0, int(math.ceil(abs(b - a) / d)) + 1)]  # noqa
    contour = ([(x, lo) for x in side(lo, hi)] + [(hi, y) for y in side(lo, 20e-3 - tab)]
               + [(hi, y) for y in side(20e-3 + tab, hi)] + [(x, hi) for x in side(hi, lo)]
               + [(lo, y) for y in side(hi, 20e-3 + tab)] + [(lo, y) for y in side(20e-3 - tab, lo)])
    tab_r = [(hi, y) for y in side(20e-3 - tab, 20e-3 + tab)]
    tab_l = [(lo, y) for y in side(20e-3 - tab, 20e-3 + tab)]
    pts = contour + tab_r + tab_l

    def steps(f, g):
        for _ in zip(_cut_path(f, pts, -0.5e-3, width), _cut_path(g, pts, -0.5e-3, width)):
            yield
    inc, full, local, checks = _incremental_vs_full(f, steps, held)
    assert len(full) == 1 and inc == full
    assert full[0][0] >= len(contour) + len(tab_r)            # not before the last tab is reached
    vol = full[0][1][0] * 1e-12
    assert vol == pytest.approx((hi - lo - width) ** 2 * 4e-3, rel=0.05)
    assert 0 < local < checks
