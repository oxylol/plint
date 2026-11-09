"""Outer-boundary gating precision (architecture 5.4): engaged intervals against analytic geometry at a
0.25 mm lattice, warm start, contact geometry, the engaged-sector box, and the f_z < lattice slot with a
surface memory fixture."""

import math

import numpy as np
import pytest

from cnctwin.core.abi import EngagedIntervals, SurfaceMemory, event_buffer
from cnctwin.removal import abi as A
from cnctwin.removal import k_gate as KG
from cnctwin.removal import profile as P
from cnctwin.removal import reference as R
from cnctwin.removal import stock as S
from tests.unit.removal import fixtures as FX

TWO_PI = 2 * math.pi
LAT = 0.25e-3
LO = np.array([0.0, 0.0, 0.0])
HI = np.array([20e-3, 20e-3, 10e-3])
X0 = np.array([1.0, 0.0, 0.0])
Y0 = np.array([0.0, 1.0, 0.0])
W = np.array([0.0, 0.0, 1.0])


def _eng(n_st, n_max=4):
    return EngagedIntervals(np.zeros((n_st, n_max)), np.zeros((n_st, n_max)), np.zeros(n_st, np.int32),
                            np.zeros(1, np.int64))


def _gate(f, tip, st_z, st_r, eng=None, params=(0.0, 1e-9, 0.0)):
    st_z = np.asarray(st_z, float)
    st_r = np.asarray(st_r, float)
    eng = _eng(len(st_z)) if eng is None else eng
    stats = np.zeros(4)
    KG.engaged_intervals(f, np.asarray(tip, float), X0, Y0, W, st_z, st_r, np.ones(len(st_z), np.int8), eng, 7,
                         np.asarray(params, float), stats)
    return eng, stats


def _intervals(eng, k):
    return [(float(eng.lo_rad[k, j]), float(eng.hi_rad[k, j])) for j in range(int(eng.count[k]))]


def _ang_err(a, b):
    d = (a - b) % TWO_PI
    return min(d, TWO_PI - d)


def _plane_arc(c, r, n, d):
    """Material where n . x <= d on the circle c + r e(phi) (n in the circle plane or not): (lo, hi)."""
    A_ = r * math.hypot(n[0], n[1])
    rhs = d - float(n @ c)
    ph = math.atan2(n[1], n[0])
    if rhs >= A_:
        return [(0.0, TWO_PI)]
    if rhs <= -A_:
        return []
    half = math.acos(rhs / A_)
    lo = (ph + half) % TWO_PI
    return [(lo, lo + TWO_PI - 2 * half)]


@pytest.fixture(scope="module", params=[0, 15, 30, 45])
def wall(request):
    th = math.radians(request.param)
    n = np.array([math.cos(th), math.sin(th), 0.0])
    d = float(n @ np.array([10e-3, 10e-3, 0.0]))
    return FX.build(LO, HI, LAT, FX.halfspace_box(LO, HI, n, d)), n, d


def test_straight_walls(wall):
    f, n, d = wall
    r = 5e-3
    for cx, cy in ((10.3e-3, 10.1e-3), (8.0e-3, 11.7e-3), (12.2e-3, 9.1e-3)):
        tip = np.array([cx, cy, 2e-3])
        eng, _ = _gate(f, tip, [3e-3, 6e-3], [r, r])
        for k, z in enumerate((3e-3, 6e-3)):
            c = tip + z * W
            ref = _plane_arc(c, r, n, d)
            got = _intervals(eng, k)
            assert len(got) == len(ref)
            for (a, b), (ra, rb) in zip(got, ref):
                # planar boundary: exact up to the 10 nm quantum and the 1 nm tolerance
                beta = abs(math.sin(math.acos(max(-1.0, min(1.0, (d - n @ c) / r)))))
                bound = (2 * A.QUANTUM_M) / (r * max(beta, 1e-3))
                assert _ang_err(a, ra) <= bound and _ang_err(b, rb) <= bound


@pytest.mark.parametrize("boss", [True, False])
def test_circular_walls(boss):
    """A boss (material inside a cylinder of radius 8 mm) or a bore (material outside it): the station
    circle crosses the cylinder twice. Error bound normal to the wall: bilinear interpolation of the
    crossings of the most normal family (angle theta <= 45 deg to the wall normal) a^2 / (8 rho cos^2 theta)
    <= a^2 / (4 rho); along the circle that over sin(beta), beta the angle between the circles."""
    lo, hi = np.array([0.0, 0.0, 0.0]), np.array([30e-3, 30e-3, 10e-3])
    rho = 8e-3
    centre = (15e-3, 15e-3)
    f = FX.build(lo, hi, LAT, FX.cylinder_hole_box(lo, hi, centre, rho, inside_material=boss))
    r = 5e-3
    tip = np.array([15e-3 + 9.0e-3, 15.4e-3, 2e-3])
    eng, _ = _gate(f, tip, [4e-3], [r])
    c = tip + 4e-3 * W
    dvec = np.array(centre) - c[:2]
    dist = float(np.linalg.norm(dvec))
    a = math.acos((r * r + dist * dist - rho * rho) / (2 * r * dist))
    ph = math.atan2(dvec[1], dvec[0])
    inside_arc = ((ph - a) % TWO_PI, (ph - a) % TWO_PI + 2 * a)     # station arc inside the cylinder
    got = _intervals(eng, 0)
    assert len(got) == 1
    lo_, hi_ = got[0]
    # angle between the two circles at the crossing: sin(beta) with the triangle (r, rho, dist)
    gamma = math.acos((rho * rho + dist * dist - r * r) / (2 * rho * dist))
    beta = math.sin(a + gamma)
    bound = (LAT * LAT / (4 * rho) + 2 * A.QUANTUM_M) / (r * abs(beta)) + 1e-9
    if boss:
        assert _ang_err(lo_, inside_arc[0]) <= bound and _ang_err(hi_, inside_arc[1]) <= bound
    else:
        assert _ang_err(lo_, inside_arc[1]) <= bound and _ang_err(hi_, inside_arc[0]) <= bound


def test_slot_two_walls():
    """Material outside a slot |y - 10| < 3 mm: a 5 mm circle centred in the slot meets both walls."""
    def fn(o, fam, length):
        segs = FX.box_ray(o, fam, length, LO, HI)
        if not segs:
            return []
        t0, t1 = segs[0]
        if fam == 1:
            return FX.clip(t0, min(t1, 7e-3 - o[1]), length) + FX.clip(max(t0, 13e-3 - o[1]), t1, length)
        return segs if abs(o[1] - 10e-3) >= 3e-3 else []
    f = FX.build(LO, HI, LAT, fn)
    tip = np.array([10e-3, 10e-3, 2e-3])
    r = 5e-3
    eng, _ = _gate(f, tip, [4e-3], [r])
    got = _intervals(eng, 0)
    a = math.asin(3e-3 / r)
    ref = [(a, math.pi - a), (math.pi + a, TWO_PI - a)]
    assert len(got) == 2
    for (lo, hi), (ra, rb) in zip(sorted(got), ref):
        assert _ang_err(lo, ra) < 1e-6 and _ang_err(hi, rb) < 1e-6


def test_plunge_full_and_empty_circles():
    f = S.block_field(HI, LAT)
    eng, _ = _gate(f, [10e-3, 10e-3, 9e-3], [0.5e-3, 1.5e-3, 0.0], [2e-3, 2e-3, 0.0])
    assert _intervals(eng, 0) == [(0.0, TWO_PI)]          # inside the block
    assert _intervals(eng, 1) == []                       # above the top face
    assert _intervals(eng, 2) == [(0.0, TWO_PI)]          # a station on the axis (point)


def test_ramp_inclined_floor():
    alpha = math.radians(8.0)
    n = np.array([math.sin(alpha), 0.0, math.cos(alpha)])
    d = float(n @ np.array([10e-3, 10e-3, 5e-3]))
    f = FX.build(LO, HI, LAT, FX.halfspace_box(LO, HI, n, d))
    tip = np.array([10e-3, 10e-3, 5e-3 - 0.1e-3])
    st_r = [1e-3, 2.5e-3, 4e-3]
    eng, _ = _gate(f, tip, [0.0, 0.0, 0.0], st_r)
    for k, r in enumerate(st_r):
        ref = _plane_arc(tip, r, n, d)
        got = _intervals(eng, k)
        assert len(got) == len(ref)
        for (a, b), (ra, rb) in zip(got, ref):
            assert _ang_err(a, ra) < 5e-6 and _ang_err(b, rb) < 5e-6


def test_ball_nose_on_inclined_plane():
    R0 = 4e-3
    alpha = math.radians(25.0)
    n = np.array([math.sin(alpha), 0.0, math.cos(alpha)])
    tip = np.array([10e-3, 10e-3, 4e-3])
    d = float(n @ (tip + R0 * W - R0 * n)) + 0.3e-3            # the ball 0.3 mm into the plane
    f = FX.build(LO, HI, LAT, FX.halfspace_box(LO, HI, n, d))
    zs = np.linspace(0.2e-3, 3.8e-3, 10)
    rs = np.sqrt(R0 ** 2 - (R0 - zs) ** 2)
    eng, _ = _gate(f, tip, zs, rs)
    for k in range(len(zs)):
        c = tip + zs[k] * W
        ref = _plane_arc(c, rs[k], n, d)
        got = _intervals(eng, k)
        assert len(got) == len(ref)
        for (a, b), (ra, rb) in zip(got, ref):
            assert _ang_err(a, ra) < 1e-5 and _ang_err(b, rb) < 1e-5


def test_fz_below_lattice_slot_with_surface_memory():
    """Slotting with f_z = 50 um at a 250 um lattice: gating combined with the surface memory shows the
    whole advancing half engaged (up to where the chip thickness is below the field's error bound)."""
    from cnctwin.removal import k_subtract as KS

    R0 = 5e-3
    fz = 50e-6
    f = S.block_field(HI, LAT)
    prof = P.profile_from_points([0.0, 20e-3], [R0, R0])
    acc = A.new_accum(1)
    ev = event_buffer(64, A.EVENT_WIDTH)
    sc = A.make_scratch()
    tgt = A.empty_target(int(f.info[A.FI_NRAYS]))
    x = 3e-3
    while x < 10e-3:
        spec = A.subtract_spec(W, [[x, 10e-3, 6e-3]], [0.0, 20e-3], [0.0], [-1] * 3, [1] * 3)
        KS.subtract_envelope(f, prof, spec, tgt, acc, ev, sc)
        x_last = x
        x += fz
    c_prev = np.array([x_last, 10e-3])
    tip = np.array([x_last + fz, 10e-3, 6e-3])
    nb = 720
    eng, _ = _gate(f, tip, [2e-3], [R0], _eng(1, 8))
    phis = (np.arange(nb) + 0.5) * TWO_PI / nb
    pts = tip[:2] + R0 * np.stack([np.cos(phis), np.sin(phis)], axis=1)
    mem = SurfaceMemory(np.zeros((1, nb, 3)), np.zeros((1, nb), np.int64), np.ones((1, nb), np.int8))
    mem.point_S_m[0, :, :2] = c_prev + R0 * np.stack([np.cos(phis), np.sin(phis)], axis=1)
    normal = np.stack([np.cos(phis), np.sin(phis)], axis=1)
    h = np.einsum("ij,ij->i", pts - mem.point_S_m[0, :, :2], normal)
    gated = np.zeros(nb, bool)
    for lo, hi in _intervals(eng, 0):
        rel = (phis - lo) % TWO_PI
        gated |= rel <= (hi - lo)
    bound = LAT * LAT / (4 * R0) + 2 * A.QUANTUM_M
    must = h > bound                         # chip thicker than the field's error: must be engaged
    assert np.all(gated[must])
    engaged = gated & (h > 0.0)
    ang = np.degrees(np.arctan2(np.sin(phis[engaged]), np.cos(phis[engaged])))
    assert ang.min() < -88.0 and ang.max() > 88.0          # the full advancing half
    assert np.all(np.abs(ang) <= 90.0 + 1e-6)              # and nothing of the trailing half


def test_warm_start_gives_the_cold_result():
    """Intervals of the previous tooth period seed the refinement; the result equals a cold computation."""
    th = math.radians(20)
    n = np.array([math.cos(th), math.sin(th), 0.0])
    d = float(n @ np.array([10e-3, 10e-3, 0.0]))
    f = FX.build(LO, HI, LAT, FX.halfspace_box(LO, HI, n, d))
    tip = np.array([10.3e-3, 10.1e-3, 2e-3])
    eng, _ = _gate(f, tip, [4e-3], [5e-3])
    moved = tip + np.array([20e-6, -7e-6, 0.0])
    eng_warm, s_warm = _gate(f, moved, [4e-3], [5e-3], eng)
    eng_cold, s_cold = _gate(f, moved, [4e-3], [5e-3])
    for (a, b), (ra, rb) in zip(_intervals(eng_warm, 0), _intervals(eng_cold, 0)):
        assert _ang_err(a, ra) < 1e-8 and _ang_err(b, rb) < 1e-8
    assert s_warm[1] <= s_cold[1] + 4


def test_matches_reference_on_a_milled_corner():
    """Engaged intervals against the dense-sampling reference on a field shaped by two cuts."""
    from cnctwin.removal import k_subtract as KS

    f = S.block_field([12e-3, 12e-3, 6e-3], 0.5e-3)
    prof = P.profile_from_points([0.0, 20e-3], [3e-3, 3e-3])
    for c in ([3e-3, 6e-3, 3e-3], [6e-3, 3e-3, 3e-3]):
        KS.subtract_envelope(f, prof, A.subtract_spec(W, [c], [0.0, 20e-3], [0.0], [-1] * 3, [1] * 3),
                             A.empty_target(int(f.info[A.FI_NRAYS])), A.new_accum(1), event_buffer(8, A.EVENT_WIDTH),
                             A.make_scratch())
    tip = np.array([6.5e-3, 6.2e-3, 3.5e-3])
    eng, _ = _gate(f, tip, [1e-3], [3.2e-3], params=(0.0, 1e-10, 0.0))
    ref = R.engaged_ref(f, tip + 1e-3 * W, X0, Y0, 3.2e-3, n=1440)
    got = _intervals(eng, 0)
    assert len(got) == len(ref)
    for (a, b), (ra, rb) in zip(got, ref):
        assert _ang_err(a, ra) < 1e-6 and _ang_err(b, rb) < 1e-6


def test_contact_geometry_and_sector_box():
    f = S.block_field(HI, LAT)
    # slot: tool half in the block from the -x side, feed +x
    tip = np.array([0.0, 10e-3, 6e-3])
    R0 = 5e-3
    eng, _ = _gate(f, tip, [2e-3], [R0])
    acc = A.new_accum(1)
    KG.contact_geometry(eng, np.array([2e-3]), np.array([R0]), np.array([0.5e-3]), 1.0, 0.0, acc)
    assert acc.fvals[A.FA_AE_MAX] == pytest.approx(2 * R0, rel=1e-6)
    assert acc.fvals[A.FA_ANGLE_MAX] == pytest.approx(math.pi, abs=1e-5)
    lo = np.zeros(3)
    hi = np.zeros(3)
    n = KG.sector_box(eng, tip, X0, Y0, W, np.array([2e-3]), R0, 0.0, 20e-3, 0.0, lo, hi)
    assert n == 1
    # the advancing half (feed +x): x from the axis to R, y over the whole diameter, z the envelope's range
    assert lo[0] == pytest.approx(0.0, abs=1e-8) and hi[0] == pytest.approx(R0, rel=1e-9)
    assert lo[1] == pytest.approx(10e-3 - R0, rel=1e-9) and hi[1] == pytest.approx(10e-3 + R0, rel=1e-9)
    assert lo[2] == pytest.approx(6e-3) and hi[2] == pytest.approx(26e-3)
    # side milling with ae = 1 mm on the +y wall side
    tip = np.array([10e-3, 20e-3 + R0 - 1e-3, 6e-3])
    eng, _ = _gate(f, tip, [2e-3], [R0])
    acc = A.new_accum(1)
    KG.contact_geometry(eng, np.array([2e-3]), np.array([R0]), np.array([0.5e-3]), 1.0, 0.0, acc)
    assert acc.fvals[A.FA_AE_MAX] == pytest.approx(1e-3, abs=2e-6)


def test_one_signature_each():
    for kernel in KG.KERNELS:
        assert len(kernel.signatures) == 1, kernel.__name__


def test_a_station_within_a_cell_of_the_top_edge_of_a_side_step_is_engaged():
    """A side step along the block's front face (0.5 mm wide, 1 mm deep, f_z 53 um, lattice 0.25 mm): the station
    0.09 mm below the block's top lies within one cell of the top and of the cut front, where the rays above the top
    hold no material and the z columns are cut and uncut in turn: every dexel family was invalid and the fallback
    classified the crescent of material there as air, so the station had no engaged interval (the top cell of the
    side step: 17 % of its chip in the tiny project). Classification now takes the family's rays that hold material
    (docs/models/removal.md 5): the station sees the step's arc like the stations below it, and points just outside
    the block's edge stay air."""
    from cnctwin.removal import k_sd as KSD
    from cnctwin.removal import k_subtract as KS
    f = S.block_field([30e-3, 20e-3, 10e-3], LAT)
    work = np.zeros(4)
    for (y, z, inside) in ((0.05e-3, 9.97e-3, True), (0.144e-3, 10.04e-3, False), (-0.05e-3, 9.89e-3, False),
                           (-0.02e-3, 10.02e-3, False)):
        ins, d, fam = KSD.classify(f, 0, 12.3e-3, y, z, 2, work)
        assert bool(ins) == inside, (y, z, d, fam)
        assert R.classify_ref(f, 0, np.array([12.3e-3, y, z])) == inside
    r, yc, tip_z, fz = 5e-3, -4.5e-3, 9.0e-3, 53e-6
    prof = P.profile_from_points([0.0, 20e-3], [r, r])
    x_now = 12.0e-3
    for x in np.append(np.arange(0.0, x_now - fz, 0.25e-3), x_now - fz):
        spec = A.subtract_spec(W, [[x, yc, tip_z]], [0.0, 20e-3], [0.0], [-1.0] * 3, [1.0] * 3)
        KS.subtract_envelope(f, prof, spec, A.empty_target(int(f.info[A.FI_NRAYS])), A.new_accum(1),
                             event_buffer(64, A.EVENT_WIDTH), A.make_scratch())
    tip = np.array([x_now, yc, tip_z])
    st_z = np.array([0.3e-3, 0.7e-3, 0.91e-3])
    eng, _ = _gate(f, tip, st_z, [r, r, r])
    # the crescent: in front of the face (y >= 0) and outside the last circle (f_z behind)
    face = math.asin(-yc / r)                                     # 64.2 deg
    front = math.acos(-fz / (2 * r))                              # 90.3 deg
    for k in range(3):
        got = _intervals(eng, k)
        assert len(got) == 1, (k, got)
        assert _ang_err(got[0][0], face) < math.radians(0.5), (k, math.degrees(got[0][0]))
        assert _ang_err(got[0][1], front) < math.radians(3.0), (k, math.degrees(got[0][1]))
