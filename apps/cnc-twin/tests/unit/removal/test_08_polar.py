"""Polar field (turning): construction, the per-L0 insert cut (volume, radius, threads), the exact tri-dexel
rebuild, KP growth, the resolution rule, the insert region chains."""

import math

import numpy as np
import pytest

from cnctwin.core.abi import event_buffer
from cnctwin.removal import abi as A
from cnctwin.removal import field as F
from cnctwin.removal import k_polar as KP
from cnctwin.removal import k_ray as KR
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
    assert KP.rebuild_tridexel(f, pf, 0, 0.0, L, 0.0, 1.0, tgt, 5e-6, event_buffer(8, A.EVENT_WIDTH), 0, 1, sc) == 0
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
                                               [(qa, qb)], 7, sliver=0)
            q2_ref = sum(b * b - a * a for a, b in pieces)
            q2, st, q3, x_lo, x_hi = KP._cut_inline(seg, sev, cnt, 0, qa, qb, 7, kp)
            if len(new) > kp:
                assert st == -1 and cnt[0] == 2 and seg[0, 0, 0] == 100        # unchanged
                continue
            if not pieces:
                assert st == 1
                continue
            assert st == 0 and q2 == q2_ref
            # the radial centroid's moment and the removed range (turning contacts' geometry)
            assert q3 == pytest.approx(sum(float(b) ** 3 - float(a) ** 3 for a, b in pieces), rel=1e-15)
            assert (x_lo, x_hi) == (min(a for a, b in pieces), max(b for a, b in pieces))
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
    assert KP.rebuild_tridexel(f, pf, 0, 0.0, L, 0.0, 1.0, tgt, 5e-6, event_buffer(8, A.EVENT_WIDTH), 0, 1, A.make_scratch()) == 0
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


def test_a_turning_contact_carries_its_point_and_depth():
    """polar_cut accumulates the turning contact's geometry as the tri-dexel cuts do (removed volume, centroid,
    extents; docs/models/removal.md "Polar field"): the centroid's mean radius lies in the removed annulus, it is
    placed at the tool's angle of the last removal, and the depth of cut is the removed material's extent across the
    feed: radial for a longitudinal pass, axial for facing. Before 2026-11-09 a turning contact had point_S_m 0 and
    ap_m 0 (every consumer read the S origin)."""
    import dataclasses
    from types import SimpleNamespace
    from cnctwin.removal.domain import RemovalDomain
    R0, L, ap, f_rev = 15e-3, 30e-3, 1e-3, 0.2e-3
    n_th, dz, nz, dr, nr, _ = PO.resolution(L, 360, f_rev, 0.4e-3, R0)
    pf = PO.polar_tube(R0, 0.0, 0.0, L, n_th, dz, dr)
    reg = PO.insert_region(_insert())
    v0 = PO.volume_m3(pf)
    acc, pf = _turn(pf, reg, 20e-3, 17e-3, R0 - ap, f_rev)
    fv = acc.fvals
    removed = v0 - PO.volume_m3(pf)
    assert fv[A.FA_W] == pytest.approx(removed, rel=1e-9)
    r_c = fv[A.FA_CX] / fv[A.FA_W]
    exp_r = (2.0 / 3.0) * (R0 ** 3 - (R0 - ap) ** 3) / (R0 ** 2 - (R0 - ap) ** 2)
    assert r_c == pytest.approx(exp_r, abs=0.05e-3)
    depth = fv[A.FA_DEPTH_MAX] - fv[A.FA_DEPTH_MIN]
    assert depth == pytest.approx(ap, abs=2 * dz + (f_rev ** 2) / (8 * 0.4e-3))
    assert fv[A.FA_NX] > 0.0                                  # an external tool: the new surface faces outward
    th_last = fv[A.FA_POLAR_TH]
    # the domain's contact record: the centroid at the tool's angle, the depth of cut
    dom = RemovalDomain.__new__(RemovalDomain)
    ctx = SimpleNamespace(ipo_s=1e-3)
    g = dom._contact_geometry(ctx, SimpleNamespace(bus_prev=SimpleNamespace(motion=SimpleNamespace(paths=[]))), acc,
                              0)
    assert math.hypot(g.point_S_m[0], g.point_S_m[1]) == pytest.approx(r_c, rel=1e-12)
    assert math.atan2(g.point_S_m[1], g.point_S_m[0]) == pytest.approx(math.atan2(math.sin(th_last),
                                                                                  math.cos(th_last)), abs=1e-12)
    assert g.ap_m == pytest.approx(depth, rel=1e-12) and g.in_cut
    # facing: the tool fed toward the axis at a fixed z takes an axial layer, its depth the layer's thickness
    pf = PO.polar_tube(R0, 0.0, 0.0, L, n_th, dz, dr)
    acc = A.new_accum(1)
    ps = PO.polar_step(0, 0, 0, 0, 0, 0, 1.0, -1.0, 0.0)
    face = 0.5e-3
    zt, rt, th = L - face + 0.4e-3, R0 + 1e-3, 0.0
    dth = math.radians(0.5)
    for _ in range(int(round(2 * math.pi / dth)) * 4):
        rt1 = rt - f_rev * dth / (2 * math.pi)
        ps[:6] = (zt, rt, zt, rt1, th, th - dth)
        assert KP.polar_cut(pf, reg, ps, 3, acc) == 0
        th, rt = th - dth, rt1
    fv = acc.fvals
    if fv[A.FA_W] > 0.0:
        assert fv[A.FA_DEPTH_MAX] - fv[A.FA_DEPTH_MIN] == pytest.approx(face, abs=2 * dz + 0.05e-3)
    del dataclasses


def test_the_polar_rows_do_not_depend_on_the_inline_capacity():
    """The rows are sized for the memory budget with a fixed capacity (polar.SIZING_KP): the configured initial
    polar_kp (grown on demand anyway) no longer changes the field's resolution (kp 16 coarsened a d45 bar's rows
    from 6.25 um to 75 um, so its pass differed from kp 1's from the first contact; docs/models/loop.md 15)."""
    import dataclasses
    from types import SimpleNamespace
    from cnctwin.removal.domain import RemovalDomain
    from cnctwin.removal.schema import RemovalSettings
    project = SimpleNamespace(tools=SimpleNamespace(list=[SimpleNamespace(
        tool=SimpleNamespace(geometry=SimpleNamespace(insert=SimpleNamespace(nose_radius_m=0.8e-3))))]))
    dom = RemovalDomain.__new__(RemovalDomain)
    geoms = []
    for kp in (1, 2, 16):
        st = dataclasses.replace(RemovalSettings(), polar_kp=kp)
        pf, note = dom._make_polar(None, project, st, None, "cylinder", [45e-3, 60e-3])
        geoms.append((tuple(pf.geom), note))
        assert int(pf.info[A.PI_KP]) == kp
    assert geoms[0] == geoms[1] == geoms[2]


def _tube_mesh(r_out, r_in, length, n, phase=0.0):
    """Closed, outward-oriented mesh of an n-gon tube (r_in 0: a solid bar with a centre vertex on each cap)."""
    ang = phase + 2.0 * math.pi * np.arange(n) / n
    ring = lambda r, z: np.column_stack([r * np.cos(ang), r * np.sin(ang), np.full(n, z)])
    V = [ring(r_out, 0.0), ring(r_out, length)]
    o0, o1 = np.arange(n), n + np.arange(n)
    nx = lambda a: (a - a[0] + 1) % n + a[0]
    F = [np.column_stack([o0, nx(o0), nx(o1)]), np.column_stack([o0, nx(o1), o1])]
    if r_in > 0.0:
        V += [ring(r_in, 0.0), ring(r_in, length)]
        i0, i1 = 2 * n + np.arange(n), 3 * n + np.arange(n)
        F += [np.column_stack([i0, nx(i1), nx(i0)]), np.column_stack([i0, i1, nx(i1)]),
              np.column_stack([i1, o1, nx(o1)]), np.column_stack([i1, nx(o1), nx(i1)]),
              np.column_stack([i0, nx(o0), o0]), np.column_stack([i0, nx(i0), nx(o0)])]
    else:
        V += [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, length]])]
        c0, c1 = np.full(n, 2 * n), np.full(n, 2 * n + 1)
        F += [np.column_stack([c1, o1, nx(o1)]), np.column_stack([c0, nx(o0), o0])]
    return np.vstack(V), np.vstack(F)


def _ngon_radius(r, n, phase, th):
    """Distance from the axis to the n-gon of circumradius r (vertices at phase + 2 pi k / n) along th."""
    step = 2.0 * math.pi / n
    k = np.floor((th - phase) / step)
    mid = phase + (k + 0.5) * step
    return r * math.cos(step / 2) / np.cos(th - mid)


@pytest.mark.parametrize("n_th,phase_bins", [(96, 0.5), (96, 0.13), (95, 0.37)])
def test_polar_from_mesh_holds_the_tessellated_tube_exactly(n_th, phase_bins):
    """Every radial ray of a mesh tube meets the n-gons where geometry says, every axial ray covers the length
    exactly; with phase 0.5 bins the n-gon vertices sit on bin centres, so radial rays run through the walls' shared
    vertical edges and axial rays along the shared cap edges: the tie rule counts each crossing once. An odd bin
    count has no opposite bin on each line (one-sided lines)."""
    r_out, r_in, L, n = 4e-3, 1.5e-3, 6e-3, 48
    dth = 2.0 * math.pi / n_th
    phase = phase_bins * dth
    dz, dr = 0.05e-3, 0.1e-3
    for ri in (r_in, 0.0):
        V, Fc = _tube_mesh(r_out, ri, L, n, phase)
        pf = PO.polar_from_mesh(V, Fc, n_th, dz, dr, 0.0, L, r_out, kp=1)
        nz, nr = int(pf.geom[A.PG_NZ]), int(pf.geom[A.PG_NR])
        th = (np.arange(n_th) + 0.5) * dth
        qo = np.floor(_ngon_radius(r_out, n, phase, th) / Q + 0.5)
        qi = np.floor(_ngon_radius(ri, n, phase, th) / Q + 0.5) if ri > 0 else np.zeros(n_th)
        rc = pf.rcnt.reshape(n_th, nz)
        rs = pf.rseg.reshape(n_th, nz, -1, 2)
        assert np.all(rc == 1)
        assert np.all(np.abs(rs[:, :, 0, 1] - qo[:, None]) <= 1)
        assert np.all(np.abs(rs[:, :, 0, 0] - qi[:, None]) <= 1)
        # axial rays: [0, L] where the ray's point lies in the annulus of n-gons, none elsewhere
        rj = (np.arange(nr) + 0.5) * dr
        inside = (rj[None, :] < _ngon_radius(r_out, n, phase, th)[:, None])
        if ri > 0:
            inside &= rj[None, :] > _ngon_radius(ri, n, phase, th)[:, None]
        ac = pf.acnt.reshape(n_th, nr)
        asg = pf.aseg.reshape(n_th, nr, -1, 2)
        assert np.array_equal(ac, inside.astype(np.int8))
        assert np.all(asg[inside][:, 0, 0] == 0) and np.all(asg[inside][:, 0, 1] == round(L / Q))
        _, _, vol = S.oriented_mesh(V, Fc)
        # rows sample the n-gon at bin centres: the polar volume is the n_th-gon-of-the-n-gon area times the length
        area = 0.5 * dth * np.sum(_ngon_radius(r_out, n, phase, th) ** 2
                                  - (_ngon_radius(ri, n, phase, th) ** 2 if ri > 0 else 0.0))
        assert PO.volume_m3(pf) == pytest.approx(area * L, rel=1e-6)
        assert PO.volume_m3(pf) == pytest.approx(vol, rel=2e-3)


def test_polar_from_mesh_puts_shoulders_at_their_own_z():
    """A stepped bar (d8 x 3 on d5 x 3): rows below and above the shoulder hold their own radius, the shoulder is
    where the mesh has it (not on the nearest tri-dexel lattice plane), the axial rays end on the shoulder face."""
    n, L1, L2 = 64, 3.0e-3, 3.0e-3
    big_V, big_F = _tube_mesh(4e-3, 0.0, L1, n)
    small_V, small_F = _tube_mesh(2.5e-3, 0.0, L2, n)
    # union through one closed surface: the big cylinder's top cap becomes an annulus plus the small cylinder on it
    ang = 2.0 * math.pi * np.arange(n) / n
    ring = lambda r, z: np.column_stack([r * np.cos(ang), r * np.sin(ang), np.full(n, z)])
    V = np.vstack([ring(4e-3, 0.0), ring(4e-3, L1), ring(2.5e-3, L1), ring(2.5e-3, L1 + L2),
                   [[0.0, 0.0, 0.0], [0.0, 0.0, L1 + L2]]])
    a0, a1, b0, b1 = (k * n + np.arange(n) for k in range(4))
    nx = lambda a: (a - a[0] + 1) % n + a[0]
    c0, c1 = np.full(n, 4 * n), np.full(n, 4 * n + 1)
    Fc = np.vstack([np.column_stack([a0, nx(a0), nx(a1)]), np.column_stack([a0, nx(a1), a1]),
                    np.column_stack([b0, nx(b0), nx(b1)]), np.column_stack([b0, nx(b1), b1]),
                    np.column_stack([b0, a1, nx(a1)]), np.column_stack([b0, nx(a1), nx(b0)]),
                    np.column_stack([c1, b1, nx(b1)]), np.column_stack([c0, nx(a0), a0])])
    del big_V, big_F, small_V, small_F
    _, _, vol = S.oriented_mesh(V, Fc)
    n_th, dz, dr = 64, 0.04e-3, 0.1e-3
    pf = PO.polar_from_mesh(V, Fc, n_th, dz, dr, 0.0, L1 + L2, 4e-3, kp=1)
    nz, nr = int(pf.geom[A.PG_NZ]), int(pf.geom[A.PG_NR])
    zc = (np.arange(nz) + 0.5) * dz
    rs = pf.rseg.reshape(n_th, nz, -1, 2)[:, :, 0, 1]
    th = (np.arange(n_th) + 0.5) * 2 * math.pi / n_th
    want = np.where(zc[None, :] < L1, _ngon_radius(4e-3, n, 0.0, th)[:, None],
                    _ngon_radius(2.5e-3, n, 0.0, th)[:, None])
    assert np.all(np.abs(rs - np.floor(want / Q + 0.5)) <= 1)
    rj = (np.arange(nr) + 0.5) * dr
    asg = pf.aseg.reshape(n_th, nr, -1, 2)[:, :, 0, 1]
    ac = pf.acnt.reshape(n_th, nr)
    shoulder = (rj[None, :] > _ngon_radius(2.5e-3, n, 0.0, th)[:, None]) & (ac == 1)
    assert np.all(asg[shoulder] == round(L1 / Q))
    assert np.all(asg[(rj[None, :] < _ngon_radius(2.5e-3, n, 0.0, th)[:, None])] == round((L1 + L2) / Q))
    assert PO.volume_m3(pf) == pytest.approx(vol, rel=3e-3)


def test_polar_from_mesh_rejects_an_open_mesh():
    V, Fc = _tube_mesh(4e-3, 0.0, 6e-3, 32)
    with pytest.raises(Exception):
        PO.polar_from_mesh(V, Fc[:-3], 64, 0.1e-3, 0.2e-3, 0.0, 6e-3, 4e-3)


def test_polar_volume_is_exact_past_the_int64_range():
    """sum q^2 of a d600 x 1 m bar at 8 bins and 0.5 mm rows is 1.4e19, beyond int64 (k_polar.polar_q2_sum wraps):
    the volume is still pi r^2 L to round-off (the lathe-shaft d45 x 230 bar read 20 % of its volume before)."""
    pf = PO.polar_tube(0.3, 0.0, 0.0, 1.0, 8, 0.5e-3, 0.03)
    assert PO.q2_sum(pf) > 2 ** 63
    assert PO.volume_m3(pf) == pytest.approx(math.pi * 0.3 ** 2 * 1.0, rel=1e-12)


def _cup_mesh(r_out, r_hole, length, depth, n):
    """A bar of n-gons with a blind hole from its top (the lathe-shaft bar's centre hole, simplified)."""
    ang = 2.0 * math.pi * np.arange(n) / n
    ring = lambda r, z: np.column_stack([r * np.cos(ang), r * np.sin(ang), np.full(n, z)])
    V = np.vstack([ring(r_out, 0.0), ring(r_out, length), ring(r_hole, length), ring(r_hole, length - depth),
                   [[0.0, 0.0, 0.0], [0.0, 0.0, length - depth]]])
    o0, o1, h1, h0 = (k * n + np.arange(n) for k in range(4))
    nx = lambda a: (a - a[0] + 1) % n + a[0]
    c0, c1 = np.full(n, 4 * n), np.full(n, 4 * n + 1)
    F = np.vstack([np.column_stack([o0, nx(o0), nx(o1)]), np.column_stack([o0, nx(o1), o1]),        # outer wall
                   np.column_stack([h1, o1, nx(o1)]), np.column_stack([h1, nx(o1), nx(h1)]),        # top annulus
                   np.column_stack([h0, nx(h1), nx(h0)]), np.column_stack([h0, h1, nx(h1)]),        # hole wall (inward)
                   np.column_stack([c1, h0, nx(h0)]),                                               # hole bottom (+z)
                   np.column_stack([c0, nx(o0), o0])])                                              # bottom cap
    return V, F


def test_the_sync_rebuilds_only_the_annulus_the_insert_reached():
    """An OD pass on a bar with a blind centre hole (mesh stock; CAD: the bar turned below the pass, same hole): the
    polar -> tri-dexel sync rebuilds only the radii the insert reached (polar_cut's PI_TOUCH range), so the stock's
    exact rays at the hole stay unchanged and nothing inside the part is booked as removed. Rebuilding the whole disc
    replaced every ray of the dirty rows by its polar sampling; at the lathe-shaft bar's 60 deg centre cone that
    difference was booked as gouges at every revolution (here the whole-disc count is printed for information)."""
    R0, rh, L, depth = 6e-3, 1.6e-3, 8e-3, 3e-3
    V, Fc = _cup_mesh(R0, rh, L, depth, 24)
    f, _ = S.mesh_field(V, Fc, 0.25e-3, box=([-R0, -R0, 0.0], [R0, R0, L]))
    Vp, Fp = _cup_mesh(R0 - 0.6e-3, rh, L, depth, 24)            # the part: the bar turned below the pass, same hole
    tgt, _ = S.target_from_mesh(f, Vp, Fp)
    n_th, dz, dr = 720, 25e-6, 0.125e-3
    pf = PO.polar_from_mesh(V, Fc, n_th, dz, dr, 0.0, L, R0, kp=2)
    reg = PO.insert_region(_insert())
    f_full = F.copy_field(f)
    _, pf = _turn(pf, reg, L + 0.5e-3, L - 2.0e-3, R0 - 0.5e-3, 0.2e-3)
    touched = PO.touched_radii(pf)
    assert touched is not None and touched[0] > 3e-3          # the insert never came near the hole
    zlo, zhi = PO.dirty_z_range(pf)
    rays_before = [np.array(a) for a in (f.cnt, f.seg)]
    ev = event_buffer(8, A.EVENT_WIDTH)
    assert KP.rebuild_tridexel(f, pf, 0, zlo, zhi, touched[0], touched[1], tgt, 5e-6, ev, 0, 1, A.make_scratch()) == 0
    assert ev.n[0] == 0                                          # no gouge
    # the rays through the hole's region are the stock's (unchanged), the OD rows took the cut
    o, a = f.lv_origin_m[0], float(f.lv_cell_m[0])
    changed = 0
    for fam in range(3):
        u = 1 if fam == 0 else 0
        v = 2 if fam < 2 else 1
        for iu in range(int(f.lv_dims[0, u])):
            for iv in range(int(f.lv_dims[0, v])):
                ray = int(KR.ray_index(f, 0, fam, iu, iv))
                pu, pv = o[u] + iu * a, o[v] + iv * a
                same = int(f.cnt[ray]) == int(rays_before[0][ray]) and np.array_equal(f.seg[ray], rays_before[1][ray])
                if fam == 2 and math.hypot(pu, pv) < touched[0] - a:
                    assert same
                changed += 0 if same else 1
    assert changed > 0
    # the whole disc (as before the fix) books part material at the hole as removed
    ev2 = event_buffer(8, A.EVENT_WIDTH)
    KP.rebuild_tridexel(f_full, pf, 0, zlo, zhi, 0.0, R0, tgt, 5e-6, ev2, 0, 1, A.make_scratch())
    print("gouge events of the whole-disc rebuild", int(ev2.n[0]))
