"""Jets: trajectory against the exact projectile, drag of drops (terminal velocity), Grant-Middleman break-up,
programmable nozzle aim, the tracer against its plain-Python reference, one signature per kernel."""

import math

import numpy as np
import pytest

from cnctwin.coolant import abi, jets as J, kernels as K, reference as R
from cnctwin.coolant import fluid as F
from cnctwin.coolant.protocols import OccupancyBricks, merge_bricks, radius_table

G = 9.80665


def params(ds_max=0.01, s_max=2.0, box=10.0):
    prm = np.zeros(abi.N_TP)
    prm[abi.TP_GZ] = -G
    prm[abi.TP_RHO_AIR] = 1.204
    prm[abi.TP_MU_AIR] = 1.81e-5
    prm[abi.TP_DS_MAX] = ds_max
    prm[abi.TP_DS_MIN] = 1e-4
    prm[abi.TP_SAGITTA] = 5e-5
    prm[abi.TP_S_MAX] = s_max
    prm[abi.TP_CF_COLUMN] = 0.005
    prm[abi.TP_SN_A] = 0.15
    prm[abi.TP_SN_N] = 0.687
    prm[abi.TP_CD_NEWTON] = 0.44
    prm[abi.TP_RE_NEWTON] = 1000.0
    prm[abi.TP_BOX_LO:abi.TP_BOX_LO + 3] = -box
    prm[abi.TP_BOX_HI:abi.TP_BOX_HI + 3] = box
    return prm


def rays(origins, vels, gas=0, breakup=np.inf, drop=0.0, column=0.0, rho_l=0.0, core=0.0, s_aim=0.0, skip=-1):
    n = len(origins)
    full = lambda v, dt=float: np.full(n, v, dtype=dt)
    return abi.RaySet(np.ascontiguousarray(origins, float), np.ascontiguousarray(vels, float), full(1e-5),
                      np.arange(n, dtype=np.int64), full(gas, np.int64), full(breakup), full(drop), full(column),
                      full(rho_l), full(core), full(s_aim), full(skip, np.int64))


def empty_scene():
    return abi.empty_revolved(), abi.empty_boxes(), abi.empty_bricks(), abi.empty_heightfield()


def test_trajectory_is_the_exact_projectile():
    # no drag: the midpoint step integrates constant gravity exactly, points lie on the parabola
    v0, z0 = 6.0, 0.5
    rs = rays([[0.0, 0.0, z0]], [[v0, 0.0, 0.0]])
    out = abi.trace_out(1, 0, 400)
    K.trace_rays(rs, params(s_max=1.0), *empty_scene(), abi.Zones(np.zeros((0, 3)), np.zeros(0)), out)
    n = int(out.npts[0])
    pts = out.pts_M[0, :n]
    expected = z0 - G * pts[:, 0] ** 2 / (2 * v0 ** 2)
    assert np.max(np.abs(pts[:-1, 2] - expected[:-1])) < 1e-12
    # the last point ends the trace inside a step: it lies on the chord, within the sagitta tolerance
    assert abs(pts[-1, 2] - expected[-1]) <= 5e-5
    assert np.all(np.abs(pts[:, 1]) == 0.0)
    # speed along the path follows energy conservation
    assert np.allclose(out.pts_v[0, :n - 1] ** 2, v0 ** 2 + 2 * G * (z0 - pts[:n - 1, 2]), rtol=1e-12)
    # chords stay within the sagitta tolerance of the arc
    assert n > 10


def test_drops_reach_terminal_velocity():
    # a 0.2 mm water drop falling from rest: Schiller-Naumann terminal velocity (solved here by bisection)
    d, rho_l, rho_a, mu = 2e-4, 998.0, 1.204, 1.81e-5

    def residual(v):
        re = rho_a * v * d / mu
        cd = 24 / re * (1 + 0.15 * re ** 0.687)
        return 0.75 * rho_a * cd / (rho_l * d) * v * v - G

    lo, hi = 1e-3, 10.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if residual(mid) < 0 else (lo, mid)
    vt = 0.5 * (lo + hi)
    rs = rays([[0.0, 0.0, 0.0]], [[0.0, 0.0, -0.05]], breakup=0.0, drop=d, column=d, rho_l=rho_l)
    out = abi.trace_out(1, 0, 4000)
    K.trace_rays(rs, params(s_max=8.0, ds_max=0.005), *empty_scene(), abi.Zones(np.zeros((0, 3)), np.zeros(0)), out)
    n = int(out.npts[0])
    assert out.pts_v[0, n - 1] == pytest.approx(vt, rel=2e-3)
    assert vt == pytest.approx(0.70, rel=0.15)          # order of magnitude of 0.2 mm drizzle drops


def test_drag_shortens_the_throw():
    o = [[0.0, 0.0, 1.0]]
    v = [[5.0, 0.0, 3.0]]
    plain = abi.trace_out(1, 0, 2000)
    K.trace_rays(rays(o, v), params(s_max=6.0), *empty_scene(), abi.Zones(np.zeros((0, 3)), np.zeros(0)), plain)
    spray = abi.trace_out(1, 0, 2000)
    K.trace_rays(rays(o, v, breakup=0.0, drop=3e-4, column=3e-4, rho_l=998.0), params(s_max=6.0), *empty_scene(),
                 abi.Zones(np.zeros((0, 3)), np.zeros(0)), spray)

    def x_at_floor(out):
        n = int(out.npts[0])
        pts = out.pts_M[0, :n]
        k = int(np.argmax(pts[:, 2] < 0.0))
        return pts[k, 0]

    # without drag the throw lands at x = 5 t, t = (3 + sqrt(9 + 2 g)) / g
    assert x_at_floor(plain) == pytest.approx(5.0 * (3.0 + math.sqrt(9.0 + 2 * G)) / G, rel=2e-3)
    assert x_at_floor(spray) < 0.8 * x_at_floor(plain)


def test_grant_middleman_breakup_length():
    props = F.Props(990.0, 1.05e-3, 0.56, 4020.0, 0.035)
    speed, d = 8.0, 0.004
    we = 990.0 * speed ** 2 * d / 0.035
    assert J.breakup_length(speed, d, props) == pytest.approx(8.51 * we ** 0.32 * d, rel=1e-12)
    slow = J.breakup_length(0.2, 0.001, props)               # Re = 190: laminar correlation
    we_s = 990.0 * 0.04 * 0.001 / 0.035
    oh = 1.05e-3 / math.sqrt(990.0 * 0.035 * 0.001)
    assert slow == pytest.approx(19.5 * (we_s ** 0.5 * (1 + 3 * oh)) ** 0.85 * 0.001, rel=1e-12)


def test_programmable_nozzle_aim_steps_downward():
    home = np.array([-0.7, 0.0, -0.7])
    step = math.radians(1.5)
    a1 = J.programmable_aim(home, [0.09, -0.04, -0.04], 1, step)
    a12 = J.programmable_aim(home, [0.09, -0.04, -0.04], 12, step)
    assert np.allclose(a1, home / np.linalg.norm(home))
    assert math.acos(float(np.clip(a1 @ a12, -1, 1))) == pytest.approx(11 * step, rel=1e-12)
    assert a12[2] < a1[2]                                     # higher positions aim lower (longer tools)
    assert abs(a12[1]) < 1e-15                                # tilts in the plane of the aim and the axis


def test_sunflower_rays_carry_the_jet_flow():
    jet = J.JetSpec("n1", "n1", "flood", "liquid", np.zeros(3), np.array([0.0, 0.0, -1.0]), 10.0, 2e-4, 0.004,
                    990.0, 2e-4, math.radians(2.0), 0.5, 0.0076, 0.0)
    rs = J.make_rays([jet], 64, 8, np.array([0.3]))
    assert len(rs.weight) == 65 and rs.weight[0] == 0.0
    assert rs.weight.sum() == pytest.approx(2e-4, rel=1e-12)
    tilt = np.degrees(np.arccos(np.clip(-rs.vel_M[:, 2] / 10.0, -1, 1)))
    assert tilt.max() <= 2.0 + 1e-9
    assert np.allclose(np.linalg.norm(rs.vel_M, axis=1), 10.0)


def scene_fixture():
    """A tool envelope, a box, a pocketed block (bricks) and a pile height field."""
    from coolfix import geometries
    geo = geometries()[0]
    z0, dz, Rr, part = radius_table(geo.envelope, 2.5e-4, 2048)
    n = len(Rr)
    tip = np.array([0.05, 0.04, 0.035])
    rev = abi.Revolved(tip[None, :].copy(), np.eye(3)[None, :, :].copy(), np.array([z0]), np.array([dz]),
                       np.array([n], np.int64), Rr[None, :].copy(), part[None, :].copy(), np.array([Rr.max()]))
    boxes = abi.Boxes(np.array([[-0.05, 0.04, 0.05]]), np.eye(3)[None].copy(), np.array([[0.01, 0.02, 0.01]]),
                      np.array([3], np.int64))
    pocket = lambda p: ~((p[:, 0] > 0.03) & (p[:, 0] < 0.07) & (p[:, 1] > 0.02) & (p[:, 1] < 0.06) & (p[:, 2] > 0.02))
    work = OccupancyBricks([0.0, 0.0, 0.0], [0.1, 0.08, 0.03], 5e-3, 2, inside=pocket)
    bricks = merge_bricks([(work.bricks(0), np.eye(3), np.zeros(3))])
    nu, nv, cell = 30, 25, 4e-3
    base = np.full((nu, nv), 0.0)
    base[:5, :] = -np.inf
    top = base.copy()
    top[10:14, 8:12] = 0.006
    hf = abi.HeightField(np.array([-0.01, -0.01, 0.0]), np.eye(3), np.array([cell]), base, top, np.eye(3),
                         np.array([0.0, 0.0, -0.002]), np.ones(1, np.int64))
    zones = abi.Zones(np.array([[0.055, 0.04, 0.025], [0.0, 0.0, 0.0]]), np.array([0.008, 0.003]))
    return rev, boxes, bricks, hf, zones


def deterministic_rays(n=60):
    k = np.arange(n)
    phi = k * 2.399963229728653
    origins = np.stack([0.05 + 0.12 * np.cos(phi), 0.04 + 0.12 * np.sin(phi), 0.08 + 0.04 * np.cos(3 * phi)], axis=1)
    target = np.stack([0.05 + 0.02 * np.cos(5 * phi), 0.04 + 0.02 * np.sin(7 * phi), 0.02 + 0.03 * np.sin(phi)],
                      axis=1)
    d = target - origins
    d /= np.linalg.norm(d, axis=1)[:, None]
    gas = (k % 4 == 0).astype(np.int64)
    rs = rays(origins, 8.0 * d, s_aim=0.4, column=0.004, rho_l=998.0, breakup=0.09, drop=0.0076, core=0.02)
    return rs._replace(gas=np.ascontiguousarray(gas), rho_l=np.where(gas == 1, 0.0, 998.0))


def test_tracer_matches_reference():
    rev, boxes, bricks, hf, zones = scene_fixture()
    rs = deterministic_rays(32)
    prm = params(ds_max=0.01, s_max=0.6, box=1.0)
    out = abi.trace_out(len(rs.weight), 2, 200)
    K.trace_rays(rs, prm, rev, boxes, bricks, hf, zones, out)
    ref = R.trace_rays_ref(rs, prm, rev, boxes, bricks, hf, zones, 200)
    kinds = set(out.hit_kind.tolist())
    assert {abi.HIT_TOOL, abi.HIT_WORK} <= kinds
    assert np.array_equal(out.hit_kind, ref.hit_kind)
    assert np.array_equal(out.npts, ref.npts)
    assert np.allclose(out.hit_s, ref.hit_s, rtol=0, atol=1e-9)
    assert np.allclose(out.hit_M, ref.hit_M, rtol=0, atol=1e-9)
    finite = np.isfinite(ref.zone_s)
    assert np.array_equal(np.isfinite(out.zone_s), finite)
    assert np.allclose(out.zone_s[finite], ref.zone_s[finite], atol=1e-12)
    assert np.allclose(out.zone_dmin, ref.zone_dmin, atol=1e-12)
    tool_hits = out.hit_kind == abi.HIT_TOOL
    assert np.array_equal(out.hit_part[tool_hits], ref.hit_part[tool_hits])


@pytest.mark.parametrize("seed", range(3))
def test_revolved_hit_against_dense_sampling(seed):
    from coolfix import geometries
    z0, dz, Rr, part = radius_table(geometries()[0].envelope, 2.5e-4, 2048)
    rng = np.random.default_rng(seed)
    tmp = np.zeros(1, np.int64)
    for _ in range(40):
        a = rng.uniform([-0.04, -0.04, -0.01], [0.04, 0.04, 0.14])
        b = rng.uniform([-0.04, -0.04, -0.01], [0.04, 0.04, 0.14])
        t = K.revolved_hit(a[0], a[1], a[2], b[0], b[1], b[2], z0, dz, len(Rr), Rr.max(), Rr, part, tmp)
        tr, pr = R.revolved_hit_ref(a, b, z0, dz, Rr, part)
        if tr >= 2.0:
            assert t >= 2.0
        else:
            assert t == pytest.approx(tr, abs=5e-6)
            assert int(tmp[0]) == pr


def test_brick_dda_against_brute_force():
    pocket = lambda p: ~((p[:, 0] > 0.01) & (p[:, 0] < 0.03) & (p[:, 2] > 0.012))
    work = OccupancyBricks([0.0, 0.0, 0.0], [0.04, 0.02, 0.02], 1e-3, 4, inside=pocket)
    G_ = merge_bricks([(work.bricks(0), np.eye(3), np.zeros(3))])
    rng = np.random.default_rng(5)
    scratch = np.zeros(8)
    hits = 0
    for _ in range(60):
        o = rng.uniform([-0.01, -0.01, -0.01], [0.05, 0.03, 0.04])
        e = rng.uniform([-0.01, -0.01, -0.01], [0.05, 0.03, 0.04])
        d = e - o
        t = K.brick_hit(o, d, G_, 0, scratch)
        tr = R.brick_hit_ref(o, d, G_, 0)
        assert (t >= 2.0) == (tr >= 2.0)
        if tr < 2.0:
            hits += 1
            assert t == pytest.approx(tr, abs=1e-12)
    assert hits > 10


def test_height_field_against_brute_force():
    rev, boxes, bricks, hf, zones = scene_fixture()
    rng = np.random.default_rng(9)
    kind = np.zeros(1, np.int64)
    for _ in range(80):
        o = rng.uniform([-0.02, -0.02, -0.005], [0.12, 0.1, 0.03])
        e = rng.uniform([-0.02, -0.02, -0.005], [0.12, 0.1, 0.03])
        t = K.height_hit(o, e - o, hf, kind)
        tr, kr = R.height_hit_ref(o, e - o, hf.base_w, hf.top_w, float(hf.cell[0]))
        assert (t >= 2.0) == (tr >= 2.0)
        if tr < 2.0:
            assert t == pytest.approx(tr, abs=1e-12)
            assert int(kind[0]) == kr


def test_mesh_pass_against_moeller_trumbore():
    from cnctwin.geom import bvh
    from cnctwin.geom.primitives import box_mesh
    mesh = box_mesh([-0.02, -0.02, -0.01], [0.02, 0.02, 0.0])
    tree = bvh.build_tree(mesh)
    rs = deterministic_rays(24)
    prm = params(ds_max=0.01, s_max=0.6, box=1.0)
    out = abi.trace_out(len(rs.weight), 0, 200)
    K.trace_rays(rs, prm, *empty_scene(), abi.Zones(np.zeros((0, 3)), np.zeros(0)), out)
    R_M_L = np.eye(3)
    o_M_L = np.array([0.05, 0.04, 0.03])
    limit = out.hit_s.copy()
    ref_limit, ref_M, flag = R.hit_meshes_ref(mesh.vertices, mesh.faces, R_M_L, o_M_L, out.pts_M, out.pts_s, out.npts,
                                              limit.copy())
    K.hit_meshes(tree, R_M_L, o_M_L, out.pts_M, out.pts_s, out.npts, limit, out.hit_M, 0, out.hit_kind,
                 out.hit_index)
    assert flag.sum() > 3
    assert np.allclose(limit, ref_limit, atol=1e-12)
    assert np.allclose(out.hit_M[flag], ref_M[flag], atol=1e-12)
    assert np.all(out.hit_kind[flag] == abi.HIT_MESH)


def test_every_kernel_has_one_signature():
    for kernel in K.KERNELS:
        assert len(kernel.signatures) == 1, kernel.__name__


def test_helper_kernels_against_references():
    rng = np.random.default_rng(21)
    out = np.zeros(2)
    for _ in range(200):
        a, b, c = rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3), rng.uniform(-0.5, 0.5, 3)
        r = rng.uniform(0.05, 0.6)
        t = K.sphere_entry(a, b, c, r, out)
        tr, dist, tc = R.sphere_entry_ref(a, b, c, r)
        assert (t >= 2.0) == (tr >= 2.0)
        if tr < 2.0:
            assert t == pytest.approx(tr, abs=1e-12)
        assert out[0] == pytest.approx(dist, abs=1e-12) and out[1] == pytest.approx(tc, abs=1e-12)
        lo = np.minimum(c, c + 0.3)
        hi = np.maximum(c, c + 0.3)
        assert K.slab_clip(a, b - a, lo, hi, 0.0, 1.0) == pytest.approx(R.box_entry(a, b - a, lo, hi), abs=1e-12)
        inside = 0.5 * (lo + hi)
        assert K.slab_exit(inside, b - inside, lo, hi) == pytest.approx(R._exit_ref(inside, b - inside, lo, hi),
                                                                         abs=1e-12)
        q = rng.uniform(-2, 2, 3)
        roots = [x.real for x in np.roots(q) if abs(x.imag) < 1e-12 and 0.0 < x.real <= 1.0]
        got = K.first_root(q[0], q[1], q[2], 0.0, 1.0)
        assert (got >= 2.0) == (not roots)
        if roots:
            assert got == pytest.approx(min(roots), abs=1e-9)
    prm = params()
    for v in ([3.0, 0.0, -1.0], [0.1, 0.2, -0.05]):
        for spray in (0, 1):
            got = K.accel(v[0], v[1], v[2], prm, 0.02, spray, 3e-4, 998.0)
            ref = R._accel_ref(v, prm, 0.02, bool(spray), 3e-4, 998.0)
            assert np.allclose(got, ref, rtol=1e-14, atol=0.0)
    fp = np.array([0.1833, 1.402, 1.0, 1e-3])
    for r in np.linspace(0.0, 0.05, 30):
        assert K.watson_speed(r, 2e-4, 10.0, 0.005, 1e-6, fp) == pytest.approx(
            R.watson_speed_ref(r, 2e-4, 10.0, 0.005, 1e-6, fp), rel=1e-14)


def test_segment_hit_against_reference():
    rev, boxes, bricks, hf, zones = scene_fixture()
    rng = np.random.default_rng(3)
    info = np.zeros(3, np.int64)
    scratch = np.zeros(K.N_SCRATCH)
    iscratch = np.zeros(2, np.int64)
    kinds = set()
    for _ in range(40):
        a = rng.uniform([-0.06, -0.02, -0.01], [0.16, 0.1, 0.15])
        b = rng.uniform([-0.06, -0.02, -0.01], [0.16, 0.1, 0.15])
        info[:] = (abi.HIT_NONE, -1, -1)
        t = K.segment_hit(a, b, rev, boxes, bricks, hf, -1, info, scratch, iscratch)
        tr, kind, idx, part = R.segment_hit_ref(a, b, rev, boxes, bricks, hf, -1)
        assert (t >= 2.0) == (tr >= 2.0)
        if tr < 2.0:
            kinds.add(kind)
            assert t == pytest.approx(tr, abs=5e-6)
            assert int(info[0]) == kind and int(info[1]) == idx
    assert len(kinds) >= 3
