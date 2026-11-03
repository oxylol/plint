"""Polar field (turning): construction, the per-L0 insert cut (volume, radius, threads), the exact tri-dexel
rebuild, KP growth, the resolution rule, the insert region chains."""

import math

import numpy as np
import pytest

from cnctwin.core.abi import event_buffer
from cnctwin.removal import abi as A
from cnctwin.removal import field as F
from cnctwin.removal import k_polar as KP
from cnctwin.removal import polar as PO
from cnctwin.removal import reference as R
from cnctwin.removal import stock as S
from cnctwin.tooling.abi import InsertProfile

Q = A.QUANTUM_M


def _insert(rho=0.4e-3, kr_minor_deg=5.0, width=4e-3):
    arc = [(rho - rho * math.sin(a), -rho + rho * math.cos(a)) for a in np.linspace(0, math.pi / 2, 10)]
    top = (width, -width * math.tan(math.radians(kr_minor_deg)))
    pts = [(0.0, -6e-3), (width, -6e-3), top] + arc + [(0.0, -rho)]
    clean = []
    for p in pts:
        if not clean or abs(p[0] - clean[-1][0]) > 1e-15 or abs(p[1] - clean[-1][1]) > 1e-15:
            clean.append(p)
    return InsertProfile(np.array(clean), np.ones(len(clean), np.int8), np.array([rho, -rho]), np.array([rho]))


def _turn(pf, reg, z_start, z_end, rt, f_rev, dth=math.radians(0.5), ev=3):
    acc = A.new_accum(1)
    ps = PO.polar_step(0, 0, 0, 0, 0, 0, 1.0, -1.0, 0.0)
    th, zt = 0.0, z_start
    n = int(round((z_start - z_end) / f_rev * (2 * math.pi / dth)))
    for _ in range(n):
        th1 = th - dth
        zt1 = zt - f_rev * dth / (2 * math.pi)
        ps[:6] = (zt, rt, zt1, rt, th, th1)
        while KP.polar_cut(pf, reg, ps, ev, acc) == 1:
            pf = PO.grow_kp(pf)            # threads: axial rays gain a segment per crest
        th, zt = th1, zt1
    return acc, pf


def test_resolution_rule_and_memory_note():
    n_th, dz, nz, dr, nr, note = PO.resolution(0.04, 720, 0.1e-3, 0.4e-3, 0.02)
    assert n_th == 720 and dz <= min(0.1e-3 / 8, 0.4e-3 / 16) + 1e-15 and note == ""
    n_th, dz2, nz, dr, nr, note = PO.resolution(0.5, 720, 0.05e-3, 0.4e-3, 0.05, max_bytes=50e6)
    assert dz2 > dz and "coarsened" in note


def test_insert_region_chains_and_nose():
    reg = PO.insert_region(_insert())
    for zt in (0.05e-3, 0.4e-3, 2e-3):
        lo, hi = KP.insert_x_range(reg, zt)
        rlo, rhi = R._x_range_ref(reg, zt)
        assert lo == pytest.approx(rlo) and hi == pytest.approx(rhi)
    lo, hi = KP.insert_x_range(reg, 0.4e-3)
    assert hi == pytest.approx(0.0, abs=1e-15)       # the nose apex: x_T = 0 at z_T = rho


def test_turning_pass_volume_radius_and_reference():
    R0, L = 15e-3, 30e-3
    n_th, dz, nz, dr, nr, _ = PO.resolution(L, 360, 0.2e-3, 0.4e-3, R0)
    pf = PO.polar_tube(R0, 0.0, 0.0, L, n_th, dz, dr)
    reg = PO.insert_region(_insert())
    rt = R0 - 1e-3
    v0 = PO.volume_m3(pf)
    snap = PO.from_arrays(PO.to_arrays(pf))
    ps = PO.polar_step(20e-3, rt, 20e-3 - 1e-6, rt, 0.0, -math.radians(1.0), 1.0, -1.0, 0.0)
    ref_rows, ref_q2 = R.polar_cut_ref(snap, reg, ps)
    acc0 = A.new_accum(1)
    KP.polar_cut(snap, reg, ps, 9, acc0)
    assert acc0.ivals[A.IA_POLAR_Q2] == ref_q2
    for (b, i), segs in ref_rows.items():
        assert R.polar_radial(snap, b, i) == segs
    acc, pf = _turn(pf, reg, 20e-3, 5e-3, rt, 0.2e-3)
    removed = acc.ivals[A.IA_POLAR_Q2] * PO.q2_volume_factor(pf)
    assert removed == pytest.approx(v0 - PO.volume_m3(pf), rel=1e-12)
    # behind the tool the diameter is the programmed one (scallop height f^2 / (8 r) = 12.5 um)
    for z in (8e-3, 12e-3, 16e-3):
        r = KP.polar_radius(pf, 1.234, z)
        assert rt <= r <= rt + (0.2e-3) ** 2 / (8 * 0.4e-3) + 2 * dz
    assert KP.polar_radius(pf, 0.3, 28e-3) == pytest.approx(R0, abs=Q) or KP.polar_radius(pf, 0.3, 28e-3) < R0
    assert PO.volume_m3(pf) == pytest.approx(R.polar_volume_ref(pf) * PO.q2_volume_factor(pf), rel=1e-12)


def test_threading_leaves_a_helix():
    """Feed per revolution = pitch with a V insert: the groove bottom advances by the pitch per turn."""
    R0, L = 10e-3, 20e-3
    pitch = 1.5e-3
    n_th, dz, nz, dr, nr, _ = PO.resolution(L, 360, pitch, 0.1e-3, R0)
    pf = PO.polar_tube(R0, 0.0, 0.0, L, n_th, dz, dr)
    half = math.radians(30)
    depth = 0.8e-3
    pts = np.array([[-depth * math.tan(half), -depth], [depth * math.tan(half), -depth], [0.0, 0.0]])
    v_insert = InsertProfile(pts, np.ones(3, np.int8), np.array([0.0, 0.0]), np.array([0.0]))
    reg = PO.insert_region(v_insert)
    rt = R0 - 0.5e-3
    acc, pf = _turn(pf, reg, 15e-3, 6e-3, rt, pitch, dth=math.radians(1.0))
    assert pf.info[A.PI_KPA] > 2 and pf.info[A.PI_KP] == 2
    # at a fixed z, the radius goes through the thread profile once per revolution
    th = np.linspace(0, 2 * math.pi, 361)[:-1]
    r = np.array([KP.polar_radius(pf, t, 10e-3) for t in th])
    assert r.min() == pytest.approx(rt, abs=2 * dz * math.tan(half) + 1e-6)
    assert r.max() == pytest.approx(R0, abs=1e-6)
    # the deepest point moves by pitch / 360 deg per degree: one minimum per revolution
    k_min = int(np.argmin(r))
    z2 = 10e-3 - pitch * 0.25
    r2 = np.array([KP.polar_radius(pf, t, z2) for t in th])
    shift = (int(np.argmin(r2)) - k_min) % 360
    assert min(shift, 360 - shift) == pytest.approx(90, abs=3)


def test_rebuild_tridexel_is_exact_and_kp_grows_on_overflow():
    R0, L = 6e-3, 8e-3
    n_th, dz, nz, dr, nr, _ = PO.resolution(L, 360, 0.2e-3, 0.4e-3, R0)
    pf = PO.polar_tube(R0, 0.0, 0.0, L, n_th, dz, dr, kp=1)
    reg = PO.insert_region(_insert(width=2e-3))
    # a face groove: the insert plunges axially between r = 2 and 3 mm, splitting radial rays: KP = 1 overflows
    acc = A.new_accum(1)
    ps = PO.polar_step(L - 2e-3, 2.5e-3, L - 2e-3, 2.5e-3, 0.0, -2 * math.pi, 1.0, -1.0, 0.0)
    groove = np.array([[0.0, -0.4e-3], [3e-3, -0.4e-3], [3e-3, 0.0], [0.0, 0.0]])
    g_reg = PO.insert_region(InsertProfile(groove, np.ones(4, np.int8), np.zeros(2), np.array([0.0])))
    st = KP.polar_cut(pf, g_reg, ps, 4, acc)
    assert st == 1 and pf.info[A.PI_OVERFLOW] & 1
    pf = PO.grow_kp(pf, 2)
    acc = A.new_accum(1)
    assert KP.polar_cut(pf, g_reg, ps, 4, acc) == 0
    assert acc.ivals[A.IA_POLAR_Q2] > 0
    f = S.tube_field(2 * R0, L, 0.25e-3)
    sc = A.make_scratch()
    tgt = A.empty_target(int(f.info[A.FI_NRAYS]))
    assert KP.rebuild_tridexel(f, pf, 0, 0.0, L, tgt, 5e-6, event_buffer(8, A.EVENT_WIDTH), 0, 1, sc) == 0
    vp = PO.volume_m3(pf)
    for fam in range(3):
        assert F.volume_m3(f, 0, fam) == pytest.approx(vp, rel=0.01)
    # an x ray through the groove has two segments there
    o = f.lv_origin_m[0]
    a = f.lv_cell_m[0]
    j = int(round((0.0 - o[1]) / a))
    k = int(round((L - 1.5e-3 - o[2]) / a))
    y = o[1] + j * a
    xs = [(o[0] + lo * Q, o[0] + hi * Q) for lo, hi, _, _ in R.ray_segments(f, R.ray_index(f, 0, 0, j, k))]
    # near the axis: material, the groove (r 2.5 .. 2.9 mm), the core, the groove, material - exact chords
    assert len(xs) == 3
    assert xs[1][1] == pytest.approx(math.sqrt(2.5e-3 ** 2 - y * y), abs=2 * Q)
    assert xs[2][0] == pytest.approx(math.sqrt(2.9e-3 ** 2 - y * y), abs=2 * Q)


def test_one_signature_each():
    for kernel in KP.KERNELS:
        assert len(kernel.signatures) == 1, kernel.__name__


def test_inline_cut_matches_the_interval_reference():
    """_cut_inline (in place, no work array) against the plain interval subtraction, every placement of a cut
    over two segments, including splits, removals, misses and the capacity limit."""
    segs = [(100, 200), (300, 400)]
    cuts = [(50, 120), (150, 160), (120, 350), (200, 300), (0, 1000), (250, 260), (390, 500), (100, 400), (199, 301)]
    for kp in (2, 3):
        for qa, qb in cuts:
            seg = np.zeros((1, kp, 2), np.int32)
            sev = np.zeros((1, kp, 2), np.int32)
            cnt = np.zeros(1, np.int8)
            for k, (lo, hi) in enumerate(segs):
                seg[0, k] = (lo, hi)
                sev[0, k] = (10 + k, 20 + k)
            cnt[0] = len(segs)
            new, pieces, _ = R.iv_subtract_ref([(lo, hi, 10 + k, 20 + k) for k, (lo, hi) in enumerate(segs)],
                                               [(qa, qb)], 7)
            q2_ref = sum(b * b - a * a for a, b in pieces)
            q2, st = KP._cut_inline(seg, sev, cnt, 0, qa, qb, 7, kp)
            if len(new) > kp:
                assert st == -1 and cnt[0] == 2 and seg[0, 0, 0] == 100        # unchanged
                continue
            if not pieces:
                assert st == 1
                continue
            assert st == 0 and q2 == q2_ref
            got = [(int(seg[0, k, 0]), int(seg[0, k, 1]), int(sev[0, k, 0]), int(sev[0, k, 1])) for k in range(cnt[0])]
            assert got == [tuple(int(v) for v in x) for x in new], (qa, qb, got, new)
            assert np.all(seg[0, cnt[0]:] == 0)


@pytest.mark.parametrize("sx, yo", [(-1.0, 0.0), (-1.0, 0.3e-3), (1.0, 0.0), (1.0, -0.2e-3)])
def test_cut_matches_reference_over_successive_steps(sx, yo):
    """External (sx -1) and internal (sx +1, boring a tube) tools, with and without a centre-height offset:
    every radial row and the q^2 sum of several successive L0 steps equal the plain-Python reference (the
    block bounds and the deep-side-first evaluation skip nothing that cuts)."""
    R0, r_in, L = 15e-3, 8e-3, 30e-3
    n_th, dz, nz, dr, nr, _ = PO.resolution(L, 360, 0.2e-3, 0.4e-3, R0)
    pf = PO.polar_tube(R0, r_in, 0.0, L, n_th, dz, dr)
    reg = PO.insert_region(_insert())
    rt = R0 - 1e-3 if sx < 0 else r_in + 1e-3
    dth = math.radians(1.0)
    th, zt = 0.0, 20e-3
    for step in range(6):
        th1, zt1 = th - dth, zt - 0.2e-3 * dth / (2 * math.pi)
        ps = PO.polar_step(zt, rt, zt1, rt, th, th1, 1.0, sx, yo)
        ref_rows, ref_q2 = R.polar_cut_ref(pf, reg, ps)
        acc = A.new_accum(1)
        assert KP.polar_cut(pf, reg, ps, 9, acc) == 0
        assert acc.ivals[A.IA_POLAR_Q2] == ref_q2 and ref_q2 > 0
        for (b, i), segs in ref_rows.items():
            assert R.polar_radial(pf, b, i) == segs
        th, zt = th1, zt1
    # the block maxima stay exact
    keep = pf.zmax_q.copy()
    KP.refresh_zmax(pf, 0)
    assert np.array_equal(keep, pf.zmax_q)


def test_rebuilt_x_and_y_rays_agree_with_the_polar_field_pointwise():
    """polar_to_ray (through rebuild_tridexel) against point membership in the polar field: every sample of
    an x or y ray of the rebuilt tri-dexel is in material exactly when its (bin, row) radial segment holds
    its radius (samples within 2 quanta of an endpoint excepted: the endpoints are quantised). Covers the
    bins visited per line (only those it crosses within the field's radius), lines on either side of the
    axis and through it."""
    R0, L = 6e-3, 8e-3
    n_th, dz, nz, dr, nr, _ = PO.resolution(L, 360, 0.2e-3, 0.4e-3, R0)
    pf = PO.polar_tube(R0, 0.0, 0.0, L, n_th, dz, dr, kp=4)
    groove = np.array([[0.0, -0.4e-3], [3e-3, -0.4e-3], [3e-3, 0.0], [0.0, 0.0]])
    g_reg = PO.insert_region(InsertProfile(groove, np.ones(4, np.int8), np.zeros(2), np.array([0.0])))
    ps = PO.polar_step(L - 2e-3, 2.5e-3, L - 2e-3, 2.5e-3, 0.0, -2 * math.pi, 1.0, -1.0, 0.0)
    assert KP.polar_cut(pf, g_reg, ps, 4, A.new_accum(1)) == 0
    acc, pf = _turn(pf, PO.insert_region(_insert()), 4e-3, 1e-3, R0 - 1.2e-3, 0.2e-3, dth=math.radians(1.0))
    f = S.tube_field(2 * R0, L, 0.25e-3)
    tgt = A.empty_target(int(f.info[A.FI_NRAYS]))
    assert KP.rebuild_tridexel(f, pf, 0, 0.0, L, tgt, 5e-6, event_buffer(8, A.EVENT_WIDTH), 0, 1, A.make_scratch()) == 0
    a = float(f.lv_cell_m[0])
    o = f.lv_origin_m[0]
    nth = int(pf.geom[A.PG_NTHETA])
    dth = float(pf.geom[A.PG_DTHETA])
    rng = np.random.default_rng(7)
    checked = 0
    for fam in (0, 1):
        u, v = (1, 2) if fam == 0 else (0, 2)
        for iu in range(0, int(f.lv_dims[0, u]), 5):
            for iv in (int(round((z - o[2]) / a)) for z in (1.6e-3, 3.1e-3, 5.4e-3, 6.6e-3)):
                ray = int(f.lv_ray0[0, fam] + iu * f.lv_dims[0, v] + iv)
                segs = [(int(f.seg[ray, k, 0]), int(f.seg[ray, k, 1])) for k in range(min(int(f.cnt[ray]), A.K_INLINE))]
                assert int(f.cnt[ray]) <= A.K_INLINE
                c = o[u] + iu * a
                zz = o[2] + iv * a
                row = int(math.floor((zz - float(pf.geom[A.PG_Z0])) / float(pf.geom[A.PG_DZ])))
                for t in rng.uniform(0.0, (f.lv_dims[0, fam] - 1) * a, 400):
                    q = t / Q
                    if any(abs(q - e) < 2 for sg in segs for e in sg):
                        continue
                    s_ = o[fam] + t
                    x, y = (s_, c) if fam == 0 else (c, s_)
                    th = math.atan2(y, x) % (2 * math.pi)
                    b = min(int(th / dth), nth - 1)
                    r = math.hypot(x, y) / Q
                    in_polar = any(lo <= r < hi for lo, hi in R.polar_radial(pf, b, row))
                    in_field = any(lo <= q < hi for lo, hi in segs)
                    assert in_polar == in_field, (fam, iu, iv, t)
                    checked += 1
    assert checked > 10000


def test_bracket_hint_walkers_equal_the_bisection_exactly():
    """_chain_w / _z_range_w / _xhi_max_w (bracket hints, O(1) per monotone step) return bit-identical values
    to _chain / _z_range / _xhi_max (bisection), for monotone and random query orders and stale hints."""
    reg = PO.insert_region(_insert())
    rng = np.random.default_rng(3)
    zs = np.concatenate([np.linspace(-1e-3, 5e-3, 400), rng.uniform(-1e-3, 5e-3, 400), np.array(reg.zu)])
    for upper, (zc, xc) in ((1, (reg.zu, reg.xu)), (0, (reg.zl, reg.xl))):
        k = 0
        for z in zs:
            v, k = KP._chain_w(zc, xc, z, upper, k)
            assert v == KP._chain(zc, xc, z, upper)
    ka = kb = 0
    for x in np.concatenate([np.linspace(-7e-3, 1e-3, 300), rng.uniform(-7e-3, 1e-3, 300), np.array(reg.xa)]):
        lft, rgt, ka, kb = KP._z_range_w(reg.xa, reg.za, reg.xb, reg.zb, reg.nose, x, ka, kb)
        assert (lft, rgt) == KP._z_range(reg.xa, reg.za, reg.xb, reg.zb, reg.nose, x)
    assert reg.box[5] == 1.0
    kx = 0
    for za in rng.uniform(-1e-3, 5e-3, 300):
        zb = za + rng.uniform(0.0, 2e-3)
        v, kx = KP._xhi_max_w(reg.zu, reg.xu, reg.nose, reg.box, za, zb, kx)
        assert v == KP._xhi_max(reg.zu, reg.xu, reg.nose, reg.box, za, zb)
        # the unimodal shortcut equals the full vertex scan
        scan = reg.box.copy()
        scan[5] = 0.0
        assert v == pytest.approx(KP._xhi_max(reg.zu, reg.xu, reg.nose, scan, za, zb), abs=1e-15)
