"""Measured costs (architecture 20 budgets; docs/models/removal.md "Cost"): engaged intervals of 40 stations,
one tooth-pass subtraction of a 10 mm end mill in side milling at 0.25 mm, one L0 step of a turning insert on
the polar field, and the memory of the 300 x 200 x 100 mm reference part at 0.25 mm.

Each cost is the best of several repeats of a compiled driver loop (no Python call overhead), timed in process
CPU time (time.process_time: the time this process ran, so other jobs on the box add less noise than to wall
time). The asserts allow 4x the targets (regression guards on a shared, loaded machine); the measured values
are printed (pytest -s) and recorded in the model document."""

import math
import time

import numpy as np
import pytest
from numba import njit

from cnctwin.core.abi import EngagedIntervals, event_buffer
from cnctwin.removal import abi as A
from cnctwin.removal import field as F
from cnctwin.removal import k_gate as KG
from cnctwin.removal import k_polar as KP
from cnctwin.removal import k_subtract as KS
from cnctwin.removal import polar as PO
from cnctwin.removal import profile as P
from cnctwin.removal import stock as S
from cnctwin.tooling.abi import InsertProfile

LAT = 0.25e-3
R0 = 5e-3
W = np.array([0.0, 0.0, 1.0])
X0 = np.array([1.0, 0.0, 0.0])
Y0 = np.array([0.0, 1.0, 0.0])


@njit(cache=False)
def _gate_loop(f, tip, x0, y0, w, st_z, st_r, active, eng, params, stats, n, dx):
    for k in range(n):
        tip[0] += dx
        KG.engaged_intervals(f, tip, x0, y0, w, st_z, st_r, active, eng, k, params, stats)


@njit(cache=False)
def _sub_loop(f, prof, spec, tgt, acc, ev, sc, n, fz):
    st = 0
    for k in range(n):
        spec.slab_c[0, 0] += fz
        spec.box_lo[0] += fz
        spec.box_hi[0] += fz
        st |= KS.subtract_envelope(f, prof, spec, tgt, acc, ev, sc)
    return st


@njit(cache=False)
def _polar_loop(pf, reg, ps, acc, n, z0, rt, f_rev, dth):
    """n L0 steps of one bin each (0.5 deg at 720 bins: physics.max_deg_per_step), feeding toward -z."""
    th = 0.0
    zt = z0
    st = 0
    for k in range(n):
        th1 = th - dth
        zt1 = zt - f_rev * dth / (2 * math.pi)
        ps[0] = zt
        ps[1] = rt
        ps[2] = zt1
        ps[3] = rt
        ps[4] = th
        ps[5] = th1
        st |= KP.polar_cut(pf, reg, ps, 3, acc)
        th = th1
        zt = zt1
    return st


def _best(fn, repeats=5):
    best = math.inf
    for _ in range(repeats):
        t0 = time.process_time()
        fn()
        best = min(best, time.process_time() - t0)
    return best


@pytest.fixture(scope="module")
def side_milling_block():
    """A block side-milled by a D10 end mill (ae 1 mm, ap 10 mm, f_z 50 um) to a steady state."""
    f = S.block_field([60e-3, 30e-3, 20e-3], LAT)
    prof = P.profile_from_points([0.0, 22e-3], [R0, R0])
    y_c = 30e-3 + R0 - 1e-3
    tgt = A.empty_target(int(f.info[A.FI_NRAYS]))
    acc = A.new_accum(1)
    ev = event_buffer(64, A.EVENT_WIDTH)
    sc = A.make_scratch()
    full = A.subtract_spec(W, [[-5e-3, y_c, 10e-3]], [0.0, 22e-3], [0.0], [-1] * 3, [1] * 3)
    for k in range(200):
        full.slab_c[0, 0] += 50e-6
        KS.subtract_envelope(f, prof, full, tgt, acc, ev, sc)
    x_c = full.slab_c[0, 0]
    # the engaged sector's box (what tooth_pass gets from sector_box): ahead of the axis, the stock side
    spec = A.subtract_spec(W, [[x_c, y_c, 10e-3]], [0.0, 22e-3], [0.0], [x_c - LAT, y_c - R0 - LAT, 9e-3],
                           [x_c + R0 + LAT, y_c + R0, 33e-3])
    return f, prof, spec, tgt, acc, ev, sc, y_c


def test_engaged_intervals_40_stations(side_milling_block, capsys):
    f, prof, spec, tgt, acc, ev, sc, y_c = side_milling_block
    # 20 peripheral stations over ap and 20 end-edge stations on the tip plane
    st_z = np.concatenate([np.linspace(0.25e-3, 9.75e-3, 20), np.zeros(20)])
    st_r = np.concatenate([np.full(20, R0), np.linspace(0.125e-3, 4.875e-3, 20)])
    tip = np.array([spec.slab_c[0, 0], y_c, 10e-3])
    eng = EngagedIntervals(np.zeros((40, 8)), np.zeros((40, 8)), np.zeros(40, np.int32), np.zeros(1, np.int64))
    active = np.ones(40, np.int8)
    params = np.array([0.0, 1e-8, 0.0])
    stats = np.zeros(4)
    args = (X0.copy(), Y0.copy(), W.copy(), st_z, st_r, active, eng, params, stats)
    _gate_loop(f, tip.copy(), *args, 2, 1e-7)
    n = 50
    t = _best(lambda: _gate_loop(f, tip.copy(), *args, n, 1e-7)) / n
    engaged = int(np.sum(eng.count > 0))
    with capsys.disabled():
        print(f"\n[removal perf] engaged intervals, 40 stations ({engaged} engaged, "
              f"{stats[0] / max(stats[3], 1) * 40:.0f} samples): {t * 1e6:.1f} us per call (target 120)")
    assert engaged >= 20
    assert t < 4 * 120e-6


def test_tooth_pass_subtraction_d10_side_milling(side_milling_block, capsys):
    f, prof, spec, tgt, acc, ev, sc, y_c = side_milling_block
    changed0 = int(acc.ivals[A.IA_RAYS_CHANGED])
    _sub_loop(f, prof, spec, tgt, acc, ev, sc, 2, 50e-6)
    n = 40
    t = _best(lambda: _sub_loop(f, prof, spec, tgt, acc, ev, sc, n, 50e-6)) / n
    per_pass = (int(acc.ivals[A.IA_RAYS_CHANGED]) - changed0) / (2 + 5 * n)
    with capsys.disabled():
        print(f"[removal perf] tooth-pass subtraction D10, ae 1 mm, ap 10 mm, f_z 50 um, 0.25 mm "
              f"({per_pass:.0f} rays changed): {t * 1e6:.1f} us per pass (target 200)")
    assert acc.removed_q[0, 2] > 0
    assert t < 4 * 200e-6


def _cnmg_like(rho=0.4e-3, side=12.9e-3):
    """80 deg rhombic insert outline in (z_T, x_T): kappa_r 95 deg, nose radius rho at the tip (0.4 mm, 0)."""
    kr = math.radians(95.0)
    eps = math.radians(80.0)
    k_minor = math.pi - kr - eps
    major = np.array([-math.cos(kr), -math.sin(kr)])
    minor = np.array([math.cos(k_minor), -math.sin(k_minor)])
    nose = [(rho - rho * math.sin(a), -rho + rho * math.cos(a)) for a in np.linspace(0.0, math.pi / 2, 12)]
    pts = [tuple(np.array(nose[0]) + side * minor)] + nose + [tuple(np.array(nose[-1]) + side * major)]
    pts = pts + [tuple(np.array(pts[-1]) + np.array(pts[0]) - np.array(nose[0]))]
    poly = np.array(pts)
    area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    if area < 0:
        poly = poly[::-1]
    return PO.insert_region(InsertProfile(np.ascontiguousarray(poly), np.ones(len(poly), np.int8),
                                          np.array([rho, -rho]), np.array([rho])))


def test_polar_l0_step_turning_insert(capsys):
    reg = _cnmg_like()
    assert reg.box[5] == 1.0 and reg.box[4] == pytest.approx(0.4e-3)
    Rb, L = 20e-3, 60e-3
    f_rev = 0.2e-3
    n_th, dz, nz, dr, nr, _ = PO.resolution(L, 720, f_rev, 0.4e-3, Rb)
    pf = PO.polar_tube(Rb, 0.0, 0.0, L, n_th, dz, dr)
    ps = PO.polar_step(0, 0, 0, 0, 0, 0, 1.0, -1.0, 0.0)
    acc = A.new_accum(1)
    dth = 2 * math.pi / n_th
    rt = Rb - 2e-3
    # enter from the bar end and run until the whole insert is over machined surface (insert length / f)
    z = 61e-3
    revs = int(math.ceil(15e-3 / f_rev))
    _polar_loop(pf, reg, ps, acc, revs * n_th, z, rt, f_rev, dth)
    z -= revs * f_rev
    state = {"z": z}

    def one_rev():
        _polar_loop(pf, reg, ps, acc, n_th, state["z"], rt, f_rev, dth)
        state["z"] -= f_rev

    t = _best(one_rev, repeats=8) / n_th
    with capsys.disabled():
        print(f"[removal perf] polar L0 step (one 0.5 deg bin), CNMG-like insert, ap 2 mm, f 0.2 mm/rev, "
              f"dz {dz * 1e6:.1f} um: {t * 1e6:.2f} us per step (target 2)")
    assert acc.ivals[A.IA_POLAR_Q2] > 0
    assert t < 4 * 2e-6


def test_memory_of_the_reference_part(capsys):
    t0 = time.perf_counter()
    f = S.block_field([0.3, 0.2, 0.1], LAT)
    build = time.perf_counter() - t0
    mb = F.memory_bytes(f) / 1e6
    tgt_mb = (8 * (int(f.info[A.FI_NRAYS]) + 1) + 8 * int(f.info[A.FI_NRAYS])) / 1e6
    with capsys.disabled():
        print(f"[removal perf] 300 x 200 x 100 mm at 0.25 mm: {int(f.info[A.FI_NRAYS])} rays, field {mb:.1f} MB "
              f"(+ about {tgt_mb:.0f} MB for a one-segment-per-ray target), built in {build:.1f} s (budget 300 MB)")
    assert mb < 300.0
    assert F.volume_m3(f, 0, 2) == pytest.approx(0.3 * 0.2 * 0.1, rel=1e-12)
