"""Revolved-envelope subtraction: exact intervals (line, sphere, torus), slabs, box, events, pool overflow."""

import math

import numpy as np
import pytest

from cnctwin.core.abi import event_buffer
from cnctwin.removal import abi as A
from cnctwin.removal import field as F
from cnctwin.removal import k_envelope as KE
from cnctwin.removal import k_subtract as KS
from cnctwin.removal import profile as P
from cnctwin.removal import reference as R
from cnctwin.removal import stock as S
from cnctwin.tooling.abi import ENV_ARC, ENV_FRUSTUM, Envelope

Q = A.QUANTUM_M


def _intervals(prof, o, fam, w, c, z, dr, t_lo, t_hi):
    out = np.zeros((A.MAX_IV, 2))
    tags = np.zeros(A.MAX_IV, np.int32)
    from cnctwin.removal.k_collide import envelope_ray

    n = envelope_ray(prof, *o, fam, np.asarray(w, float), np.asarray(c, float).reshape(-1, 3),
                     np.asarray(z, float), np.asarray(dr, float), t_lo, t_hi, out, tags, 1)
    return [tuple(out[k]) for k in range(n)]


PROFILES = {
    "cylinder": P.profile_from_points([0.0, 12e-3], [3e-3, 3e-3]),
    "cone": P.profile_from_points([0.0, 4e-3, 12e-3], [0.5e-3, 3e-3, 3e-3]),
    "ball": P.profile_ball(3e-3, 12e-3),
    "torus": P.profile_torus(3e-3, 1e-3, 12e-3),
}


@pytest.mark.parametrize("name", sorted(PROFILES))
@pytest.mark.parametrize("w", [(0.0, 0.0, 1.0), (0.3, -0.2, 0.93), (1.0, 0.0, 0.0), (0.0, 0.6, 0.8)])
def test_envelope_intervals_match_reference(name, w):
    prof = PROFILES[name]
    w = np.asarray(w) / np.linalg.norm(w)
    c = np.array([[1e-3, 0.5e-3, -0.4e-3]])
    rng = np.random.default_rng(5)
    for trial in range(6):
        fam = trial % 3
        o = c[0] + rng.uniform(-3.5e-3, 3.5e-3, 3) + w * rng.uniform(0, 6e-3)
        o[fam] = -20e-3
        got = _intervals(prof, o, fam, w, c, [0.0, 12e-3], [0.0], 0.0, 40e-3)
        ref = R.envelope_ray_ref(prof, o, fam, w, c, [0.0, 12e-3], [0.0], 0.0, 40e-3)
        assert len(got) == len(ref), (got, ref)
        for (a, b), (ra, rb) in zip(got, ref):
            assert abs(a - ra) < 1e-9 and abs(b - rb) < 1e-9


def test_torus_with_tooth_offsets_and_slabs_match_reference():
    prof = PROFILES["ball"]
    w = np.array([0.1, 0.2, 0.97])
    w /= np.linalg.norm(w)
    c = np.array([[0.0, 0.0, 0.0], [2e-6, -1e-6, 0.0], [5e-6, 0.0, 0.0]])
    z = [0.0, 1e-3, 2.5e-3, 12e-3]
    dr = [4e-6, -3e-6, 6e-6]          # sphere with a radius offset is a spindle torus
    rng = np.random.default_rng(11)
    for trial in range(9):
        fam = trial % 3
        o = rng.uniform(-3e-3, 3e-3, 3) + w * rng.uniform(0, 4e-3)
        o[fam] = -20e-3
        got = _intervals(prof, o, fam, w, c, z, dr, 0.0, 40e-3)
        ref = R.envelope_ray_ref(prof, o, fam, w, c, z, dr, 0.0, 40e-3)
        assert len(got) == len(ref)
        for (a, b), (ra, rb) in zip(got, ref):
            assert abs(a - ra) < 1e-9 and abs(b - rb) < 1e-9


def test_poly4_roots_match_numpy():
    rng = np.random.default_rng(3)
    work = np.zeros(16)
    for _ in range(20):
        roots = np.sort(rng.uniform(-1, 1, 4))
        coeffs = np.poly(roots)[::-1]          # p0 .. p4
        n = KE.poly4_roots(*coeffs, -2.0, 2.0, work)
        assert n == 4
        assert np.allclose(work[:4], roots, atol=1e-10)
        assert R.poly4_roots_ref(coeffs, -2.0, 2.0) == pytest.approx(list(work[:4]), abs=1e-9)


def _spec(w, c, z, dr, **kw):
    return A.subtract_spec(w, c, z, dr, [-1.0] * 3, [1.0] * 3, **kw)


def _run(f, prof, spec, tgt=None, cap=64):
    acc = A.new_accum(F.n_levels(f))
    ev = event_buffer(cap, A.EVENT_WIDTH)
    tgt = A.empty_target(int(f.info[A.FI_NRAYS])) if tgt is None else tgt
    st = KS.subtract_envelope(f, prof, spec, tgt, acc, ev, A.make_scratch())
    return st, acc, ev


def test_subtraction_matches_per_ray_reference_oblique_ball():
    f = S.block_field([8e-3, 8e-3, 6e-3], 0.5e-3)
    prof = PROFILES["ball"]
    w = np.array([0.2, -0.3, 0.93])
    w /= np.linalg.norm(w)
    spec = _spec(w, [[4e-3, 4e-3, 4e-3]], [0.0, 12e-3], [0.0])
    ref_segs, ref_removed = R.subtract_envelope_ref(F.copy_field(f), prof, spec)
    st, acc, ev = _run(f, prof, spec)
    assert st == 0
    for ray, segs in ref_segs.items():
        got = [(a, b) for a, b, _, _ in R.ray_segments(f, ray)]
        assert len(got) == len(segs)
        for (a, b), (ra, rb) in zip(got, segs):
            assert abs(a - ra) <= 1 and abs(b - rb) <= 1
    for (lvl, fam), q in ref_removed.items():
        assert abs(int(acc.removed_q[lvl, fam]) - q) <= 4 * len(ref_segs)


def test_plunge_volume_is_exact_column_count_and_idempotent():
    a = 0.25e-3
    f = S.block_field([20e-3, 20e-3, 10e-3], a)
    R0 = 5e-3
    prof = PROFILES["cylinder"]._replace() if False else P.profile_from_points([0.0, 20e-3], [R0, R0])
    centre = np.array([10.1e-3, 9.93e-3, 6e-3])
    spec = _spec([0, 0, 1], [centre], [0.0, 20e-3], [0.0])
    v0 = F.volume_m3(f, 0, 2)
    st, acc, ev = _run(f, prof, spec)
    o = f.lv_origin_m[0]
    nodes = 0
    for i in range(int(f.lv_dims[0, 0])):
        for j in range(int(f.lv_dims[0, 1])):
            x, y = o[0] + i * a, o[1] + j * a
            if (x - centre[0]) ** 2 + (y - centre[1]) ** 2 <= R0 ** 2 and 0 <= x < 20e-3 and 0 <= y < 20e-3:
                nodes += 1
    expect = nodes * a * a * 4e-3
    assert v0 - F.volume_m3(f, 0, 2) == pytest.approx(expect, abs=nodes * a * a * Q)
    assert int(acc.removed_q[0, 2]) * a * a * Q == pytest.approx(expect, abs=nodes * a * a * Q)
    assert v0 - F.volume_m3(f, 0, 2) == pytest.approx(math.pi * R0 ** 2 * 4e-3, rel=0.01)
    snap = F.copy_field(f)
    st, acc2, ev2 = _run(f, prof, spec)
    assert int(acc2.removed_q.sum()) == 0
    assert all(np.array_equal(x, y) for x, y in zip(snap[:-1], f[:-1]))
    # contact geometry sums
    W = acc.fvals[A.FA_W]
    cen = acc.fvals[A.FA_CX:A.FA_CZ + 1] / W
    assert cen[0] == pytest.approx(centre[0], abs=a) and cen[1] == pytest.approx(centre[1], abs=a)
    assert acc.fvals[A.FA_ZREL_MIN] == pytest.approx(0.0, abs=1e-9)
    assert acc.fvals[A.FA_ZREL_MAX] == pytest.approx(4e-3, abs=1e-9)


def test_box_limits_the_subtraction():
    f = S.block_field([20e-3, 20e-3, 10e-3], 0.25e-3)
    prof = P.profile_from_points([0.0, 20e-3], [5e-3, 5e-3])
    full = F.copy_field(f)
    spec = A.subtract_spec([0, 0, 1], [[10e-3, 10e-3, 6e-3]], [0.0, 20e-3], [0.0], [10e-3, -1, -1], [1, 1, 1])
    _run(f, prof, spec)
    _run(full, prof, _spec([0, 0, 1], [[10e-3, 10e-3, 6e-3]], [0.0, 20e-3], [0.0]))
    v_half = F.volume_m3(S.block_field([20e-3, 20e-3, 10e-3], 0.25e-3), 0, 2) - F.volume_m3(f, 0, 2)
    v_full = F.volume_m3(S.block_field([20e-3, 20e-3, 10e-3], 0.25e-3), 0, 2) - F.volume_m3(full, 0, 2)
    assert v_half == pytest.approx(0.5 * v_full, rel=0.03)


def test_pool_overflow_is_refused_and_retry_is_deterministic():
    """A comb of thin slots makes x rays need more than K segments: with no free blocks the rays are
    refused; after growing the pool a repeated call gives the field of a single call with a big pool."""
    a = 0.25e-3
    base = S.block_field([20e-3, 6e-3, 4e-3], a, pool_blocks=1)
    prof = P.profile_from_points([0.0, 10e-3], [0.4e-3, 0.4e-3])
    specs = [_spec([0, 0, 1], [[1.5e-3 + 2e-3 * k, 3e-3, 1e-3]], [0.0, 10e-3], [0.0]) for k in range(9)]
    big = F.grow_pool(F.copy_field(base), 4096)
    for sp in specs:
        assert _run(big, prof, sp)[0] == 0
    small = F.copy_field(base)
    statuses = []
    for sp in specs:
        st, acc, ev = _run(small, prof, sp)
        statuses.append(st)
        while st == 1:
            assert ev.rec[0, A.EC_KIND] == A.EV_POOL_OVERFLOW or ev.n[0] >= 1
            small = F.grow_pool(small, 64)
            st, acc, ev = _run(small, prof, sp)
    assert 1 in statuses
    for name in ("cnt", "bmat"):
        assert np.array_equal(getattr(small, name), getattr(big, name)), name
    for ray in range(0, int(big.info[A.FI_NRAYS]), 7):
        assert R.ray_segments(small, ray) == R.ray_segments(big, ray)


def test_gouge_and_breakthrough_events():
    a = 0.25e-3
    f = S.block_field([10e-3, 10e-3, 4e-3], a)
    from cnctwin.geom.primitives import box_mesh

    m = box_mesh([0, 0, 0], [10e-3, 10e-3, 2e-3])         # target: a 2 mm plate
    tgt, _ = S.target_from_mesh(f, np.asarray(m.vertices), np.asarray(m.faces))
    prof = P.profile_from_points([0.0, 10e-3], [1e-3, 1e-3])
    # down to z = 2.5 mm: no gouge
    st, acc, ev = _run(f, prof, _spec([0, 0, 1], [[5e-3, 5e-3, 2.5e-3]], [0.0, 10e-3], [0.0]), tgt)
    assert ev.n[0] == 0
    # down to z = 1.0 mm: gouge 1 mm deep
    st, acc, ev = _run(f, prof, _spec([0, 0, 1], [[5e-3, 5e-3, 1.0e-3]], [0.0, 10e-3], [0.0]), tgt)
    kinds = [int(ev.rec[k, A.EC_KIND]) for k in range(int(ev.n[0]))]
    assert A.EV_GOUGE in kinds and A.EV_BREAKTHROUGH not in kinds
    row = ev.rec[kinds.index(A.EV_GOUGE)]
    assert row[A.EC_DEPTH] == pytest.approx(1e-3, abs=2 * Q)
    # through the plate: breakthrough
    st, acc, ev = _run(f, prof, _spec([0, 0, 1], [[5e-3, 5e-3, -1e-3]], [0.0, 10e-3], [0.0]), tgt)
    kinds = [int(ev.rec[k, A.EC_KIND]) for k in range(int(ev.n[0]))]
    assert A.EV_BREAKTHROUGH in kinds


def test_rapid_into_material_and_non_centre_plunge_events():
    f = S.block_field([10e-3, 10e-3, 4e-3], 0.25e-3)
    prof = P.profile_from_points([0.0, 10e-3], [2e-3, 2e-3])
    st, acc, ev = _run(f, prof, _spec([0, 0, 1], [[5e-3, 5e-3, 3e-3]], [0.0, 10e-3], [0.0], rapid=True))
    assert int(ev.rec[0, A.EC_KIND]) == A.EV_RAPID_INTO_MATERIAL
    assert ev.rec[0, A.EC_VOLUME] > 0
    st, acc, ev = _run(f, prof, _spec([0, 0, 1], [[5e-3, 5e-3, 2e-3]], [0.0, 10e-3], [0.0], plunge=True,
                                       r_dead_m=0.8e-3, z_dead_m=0.1e-3))
    kinds = [int(ev.rec[k, A.EC_KIND]) for k in range(int(ev.n[0]))]
    assert A.EV_NON_CENTRE_PLUNGE in kinds
    assert acc.fvals[A.FA_DEAD_V] > 0


def test_bent_tool_slabs_cut_at_their_offsets():
    """Per-slab axis points (u_rel): the lower slab bent by 50 um leaves a 50 um step on the wall."""
    a = 0.25e-3
    f = S.block_field([20e-3, 20e-3, 10e-3], a)
    prof = P.profile_from_points([0.0, 20e-3], [5e-3, 5e-3])
    c = np.array([[10e-3 + 50e-6, 10e-3, 5e-3], [10e-3, 10e-3, 5e-3]])
    _run(f, prof, _spec([0, 0, 1], c, [0.0, 2.5e-3, 20e-3], [0.0, 0.0]))
    o = f.lv_origin_m[0]
    j = int(round((10e-3 - o[1]) / a))
    for z, shift in ((6e-3, 50e-6), (9e-3, 0.0)):
        k = int(round((z - o[2]) / a))
        segs = R.ray_segments(f, R.ray_index(f, 0, 0, j, k))
        y = o[1] + j * a - 10e-3
        x_hi_cut = 10e-3 + shift + math.sqrt(25e-6 - y * y)
        assert o[0] + segs[1][0] * Q == pytest.approx(x_hi_cut, abs=2 * Q)


def test_profile_from_tooling_envelope_max_radius():
    """A frustum-and-arc envelope (square end mill with a corner radius and a narrower neck)."""
    R0, rc = 5e-3, 1e-3
    rows = [  # kind, part, cutting, z0, r0, z1, r1, zc, rc, rho, a0, a1
        (ENV_FRUSTUM, 0, 1, 0.0, 0.0, 0.0, R0 - rc, 0, 0, 0, 0, 0),
        (ENV_ARC, 0, 1, 0.0, R0 - rc, rc, R0, rc, R0 - rc, rc, 0.0, math.pi / 2),
        (ENV_FRUSTUM, 0, 1, rc, R0, 20e-3, R0, 0, 0, 0, 0, 0),
        (ENV_FRUSTUM, 2, 0, 20e-3, R0, 20e-3, 4e-3, 0, 0, 0, 0, 0),
        (ENV_FRUSTUM, 2, 0, 20e-3, 4e-3, 30e-3, 4e-3, 0, 0, 0, 0, 0),
        (ENV_FRUSTUM, 2, 0, 30e-3, 4e-3, 30e-3, 0.0, 0, 0, 0, 0, 0),
    ]
    cols = list(zip(*rows))
    env = Envelope(np.array(cols[0], np.int32), np.array(cols[1], np.int32), np.array(cols[2], np.int8),
                   *(np.array(cols[i], float) for i in range(3, 12)),
                   z_lo_m=np.array([min(r[3], r[5]) for r in rows]), z_hi_m=np.array([max(r[3], r[5]) for r in rows]),
                   r_max_m=np.array([max(r[4], r[6]) for r in rows]), chain=np.zeros(len(rows), np.int32))
    cut = P.profile_from_envelope(env, "cutting")
    whole = P.profile_from_envelope(env, "all")
    for z in (0.2e-3, 0.7e-3, 5e-3, 19e-3):
        dz = z - rc
        expect = (R0 - rc) + math.sqrt(max(rc * rc - dz * dz, 0.0)) if z < rc else R0
        assert P.radius_at(cut, z) == pytest.approx(expect, abs=1e-12)
    assert P.radius_at(whole, 25e-3) == pytest.approx(4e-3)
    assert cut.info[A.PI_ZMAX] == pytest.approx(20e-3) and whole.info[A.PI_ZMAX] == pytest.approx(30e-3)
    assert cut.info[A.PI_RDEAD] == 0.0


def test_one_signature_each():
    for kernel in KE.KERNELS + KS.KERNELS:
        assert len(kernel.signatures) == 1, kernel.__name__


def _quantised(out, n, lq):
    """The removal intervals subtract_envelope makes of n envelope intervals (quantised, clipped, merged)."""
    res = []
    for i in range(n):
        qa = max(int(math.floor(out[i, 0] / Q + 0.5)), 0)
        qb = min(int(math.floor(out[i, 1] / Q + 0.5)), lq)
        if qb <= qa:
            continue
        if res and qa <= res[-1][1]:
            res[-1][1] = max(res[-1][1], qb)
        else:
            res.append([qa, qb])
    return res


def _within(ivs, lo, hi):
    """Intervals intersected with [lo, hi) (what they remove from material confined to it)."""
    return [(max(a, lo), min(b, hi)) for a, b in ivs if min(b, hi) > max(a, lo)]


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_slab_window_removes_what_every_slab_removes(name):
    """With the slab bounds and a material extent in the work array (as subtract_envelope gives them), _env_ray
    evaluates only the slabs that can reach the material: the quantised intervals within the material are those
    of every slab (random tooth slabs with offset axis points and radius offsets, axes along a family and oblique,
    every family, material extents anywhere along the ray)."""
    from cnctwin.removal.k_collide import env_ray

    prof = PROFILES[name]
    rng = np.random.default_rng(11)
    lq = 4_000_000                                      # quanta: a 40 mm ray
    n_hit = 0
    for trial in range(240):
        w = [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.3, -0.2, 0.93)][trial % 4]
        w = np.asarray(w, float) / np.linalg.norm(w)
        ns = int(rng.integers(1, 30))
        z = np.sort(np.concatenate([[0.0, 12e-3], rng.uniform(0.0, 12e-3, ns - 1)]))
        base = np.array([20e-3, 20e-3, 20e-3]) - 6e-3 * w
        c = base + rng.normal(scale=2e-6, size=(ns, 3)) * (trial % 3)
        dr = rng.normal(scale=3e-6, size=ns)
        fam = int(rng.integers(0, 3))
        o = base + rng.uniform(-3.5e-3, 3.5e-3, 3)
        o[fam] = 0.0
        out = np.zeros((A.MAX_IV, 2))
        tags = np.zeros(A.MAX_IV, np.int32)
        args = (prof.z0_m, prof.z1_m, prof.kind, prof.c0, prof.c1, prof.c2, prof.c3, prof.part, o[0], o[1], o[2], fam, w,
                np.ascontiguousarray(c), z, dr, 1e-3, 39e-3, out, tags, 1)
        full = _quantised(out, env_ray(*args, np.zeros(16)), lq)
        for _ in range(4):
            m_lo, m_hi = sorted(int(v) for v in rng.integers(100_000, 3_900_000, 2))
            work = np.zeros(16 + A.N_ENV)
            e = work[16:]
            spread = max(float(np.linalg.norm(c[k] - c[0])) for k in range(ns))
            e[KE.ENV_NS] = ns
            e[KE.ENV_CX:KE.ENV_CZ + 1] = c[0]
            e[KE.ENV_SPREAD] = spread * (1.0 + 1e-12)
            e[KE.ENV_Z0] = z[0]
            e[KE.ENV_Z1] = z[ns]
            e[KE.ENV_HINV] = ns / (z[ns] - z[0])
            e[KE.ENV_TLO] = m_lo * Q
            e[KE.ENV_THI] = m_hi * Q
            e[KE.ENV_KEY] = KE.ENV_MAGIC
            got = _quantised(out, env_ray(*args, work), lq)
            assert _within(got, m_lo, m_hi) == _within(full, m_lo, m_hi)
            n_hit += bool(_within(full, m_lo, m_hi))
    assert n_hit > 100
