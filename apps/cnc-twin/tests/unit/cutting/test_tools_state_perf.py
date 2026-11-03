"""Every tooling-built rotating tool through the milling kernel (finite results, energy closure, chip volume
against the swept material), checkpoint round trip of the cutting state, and the measured kernel cost."""

import json
import math
import time

import numba
import numpy as np
import pytest

from cnctwin.cutting import abi as A
from cnctwin.cutting import analytic as AN
from cnctwin.cutting import kernels as K
from cnctwin.cutting.drive import Driver, straight_immersion
from cnctwin.cutting.schema import CuttingSettings
from cnctwin.cutting.state import ContactCut, CuttingState, ToolCut
from tests.unit.cutting.cutfix import ALTINTAS, built, linear_layout, rev_steps

ROTATING = ["square_endmill", "ball_endmill", "face_mill", "reamer", "chamfer_mill", "engraver", "slitting_saw",
            "thread_mill", "boring_head", "twist_drill", "tap_cutting", "tap_forming"]
COEFFS = {"default": dict(ALTINTAS, h_min_m=2e-6), "forming": dict(flow_stress=400e6, mu=0.1)}


@pytest.mark.parametrize("kind", ROTATING)
def test_tooling_built_tools_run_through_the_kernel(kind):
    g = built(kind)
    nt, ne = g.edge.elem_r_m.shape
    omega = -2.0 * math.pi * 3000.0 / 60.0
    ct, table, _ = linear_layout(g, COEFFS, omega)
    assert len(K.cut_step.signatures) == 1
    drv = Driver(ct, table, nt, ne)
    R = float(ct.scal[A.TC_R_CUT])
    feed_tooth = 0.03e-3
    if kind in ("twist_drill", "tap_cutting", "tap_forming"):
        lead = g.info.get("lead") or 0.0
        vz = -(lead if lead else 0.1e-3) * abs(omega) / (2 * math.pi)
        tip = lambda t: np.array([0.0, 0.0, vz * t])
        drv.set_full_circle()
    else:
        vf = feed_tooth * nt * abs(omega) / (2 * math.pi)
        tip = lambda t: np.array([vf * t, 0.0, 0.0])
        lo, hi = straight_immersion(0.0, math.pi)
        drv.set_interval(lo, hi)
    drv.prefill_memory(tip, 0.0, omega)
    dt = 2.0 * math.pi / abs(omega) / 720.0
    n = rev_steps(omega, dt)
    rec = drv.run_rotating(n, omega, dt, tip)
    W = np.array(rec.wrench)
    assert np.all(np.isfinite(W))
    a = drv.acc.scalars
    heat = a[A.ACC_HEAT_SHEAR_J] + a[A.ACC_HEAT_RAKE_J] + a[A.ACC_HEAT_FLANK_J] + a[A.ACC_PLOUGHING_J] + \
        a[A.ACC_PROCESS_DAMPING_J]
    work = a[A.ACC_SHAFT_WORK_J] + a[A.ACC_FEED_WORK_J]
    assert work == pytest.approx(heat + drv.cacc.scalars[A.CA_DEFLECTION_J], rel=1e-9, abs=1e-12)
    assert a[A.ACC_SHAFT_WORK_J] >= 0.0
    if kind not in ("tap_cutting", "tap_forming"):
        # memory as left by the same steady path: the cut is the feed of one revolution
        assert np.max(np.array(rec.h)) <= 1.05 * max(feed_tooth * nt, 0.1e-3) + 1e-9
        assert R > 0.0


def test_cutting_state_checkpoint_round_trip():
    tool = AN.cylinder_mill(5e-3, 2, 2e-3, 4)
    ct, table, rows = linear_layout(tool, {"default": ALTINTAS}, -1047.0)
    drv = Driver(ct, table, 2, 4)
    lo, hi = straight_immersion(0.0, math.pi)
    drv.set_interval(lo, hi)
    tip = lambda t: np.array([0.05e-3 * 2 * 1047 / (2 * math.pi) * t, 0.0, 0.0])
    drv.prefill_memory(tip, 0.0, -1047.0)
    drv.run_rotating(100, -1047.0, 8e-6, tip, record=False)
    cs = CuttingState(settings=CuttingSettings())
    cs.tools[0] = ToolCut(slot=0, rows=list(rows), table=table, tool=ct, h_hi_m=6e-4, version=3)
    cs.contacts[(0, 0, 0)] = ContactCut(key=(0, 0, 0), mem=drv.mem, iv=drv.iv, refill=drv.refill, kstate=drv.state)
    cs.spindle_theta["main"] = 0.4
    arrays, meta = cs.to_arrays()
    meta = json.loads(json.dumps(meta))
    back = CuttingState.from_arrays({k: np.array(v) for k, v in arrays.items()}, meta)
    cc = back.contacts[(0, 0, 0)]
    assert np.array_equal(cc.mem.point_S_m, drv.mem.point_S_m) and np.array_equal(cc.mem.tick, drv.mem.tick)
    assert np.array_equal(cc.kstate.pend, drv.state.pend) and np.array_equal(cc.kstate.elem_acc, drv.state.elem_acc)
    t = back.tools[0]
    assert t.version == 3 and np.array_equal(t.table.coef, table.coef) and np.array_equal(t.tool.p_T_m, ct.p_T_m)
    assert back.spindle_theta["main"] == 0.4
    # the restored state continues bit-identically
    drv2 = Driver(t.tool, t.table, 2, 4)
    drv2.mem, drv2.iv, drv2.refill, drv2.state = cc.mem, cc.iv, cc.refill, cc.kstate
    drv2.acc, drv2.cacc = drv.acc, drv.cacc
    r1 = drv.run_rotating(50, -1047.0, 8e-6, tip, t0_s=100 * 8e-6)
    r2 = drv2.run_rotating(50, -1047.0, 8e-6, tip, t0_s=100 * 8e-6)
    assert np.array_equal(np.array(r1.wrench)[-50:], np.array(r2.wrench))


@numba.njit(cache=True)
def _hot_loop(n, k0, tool, table, mem, iv, refill, step, istep, pose, defl, occ, pack, axes, st, acc, cacc, out, ev,
              omega, dt, vf, tick_s):
    """A minimal compiled hot loop (what core.hotloop does around the kernel): advance the spindle angle, the
    tick and the tip, call cut_step. Called from compiled code the kernel's arguments are not boxed."""
    for k in range(k0, k0 + n):
        t_end = (k + 1) * dt
        t_mid = t_end - 0.5 * dt
        th = (omega * t_mid) % (2.0 * math.pi)
        step[A.ST_THETA_MID_RAD] = th
        step[A.ST_COS_MID] = math.cos(th)
        step[A.ST_SIN_MID] = math.sin(th)
        istep[A.SI_TICK_END] = int(round(t_end / tick_s))
        pose.tip_S_m[0] = vf * t_mid
        K.cut_step(tool, table, mem, iv, refill, step, istep, pose, defl, occ, pack, axes, st, acc, cacc, out, ev)


def test_milling_kernel_cost():
    """4 teeth x 40 engaged elements per L0 step (slotting, 30 deg helix, half of the 80 elements per tooth in
    the cut): measured ns per step and per element in a compiled loop (docs/models/cutting.md "Kernels and
    measured cost"). Calling the kernel from Python adds the unboxing of its 17 NamedTuple arguments
    (~40 us), which the compiled hot loop does not pay."""
    nz = 80
    tool = AN.cylinder_mill(5e-3, 4, 4e-3, nz, helix_rad=math.radians(30))
    omega, dt = -1047.0, 8e-6
    ct, table, _ = linear_layout(tool, {"default": ALTINTAS}, omega)
    drv = Driver(ct, table, 4, nz)
    lo, hi = straight_immersion(0.0, math.pi)
    drv.set_interval(lo, hi)
    vf = 0.05e-3 * 4 * abs(omega) / (2 * math.pi)
    tip = lambda t: np.array([vf * t, 0.0, 0.0])
    drv.prefill_memory(tip, 0.0, omega)
    drv.run_rotating(200, omega, dt, tip, record=False)
    args = (drv.tool, drv.table, drv.mem, drv.iv, drv.refill, drv.step, drv.istep, drv.pose, drv.defl, drv.occ,
            drv.pack, drv.axes, drv.state, drv.acc, drv.cacc, drv.out, drv.ev)
    _hot_loop(10, 200, *args, omega, dt, vf, drv.tick_s)                # compile
    evals0, steps0 = float(drv.cacc.scalars[A.CA_ELEM_EVALS]), float(drv.state.misc[A.MS_STEPS])
    n, k0, best = 750, 210, math.inf
    for rep in range(4):
        t0 = time.perf_counter()
        _hot_loop(n, k0, *args, omega, dt, vf, drv.tick_s)
        best = min(best, (time.perf_counter() - t0) / n)
        k0 += n
    n_in = (float(drv.cacc.scalars[A.CA_ELEM_EVALS]) - evals0) / (float(drv.state.misc[A.MS_STEPS]) - steps0)
    print(f"\ncut_step in a compiled loop: {best * 1e9:.0f} ns per step, {n_in:.1f} engaged elements, "
          f"{best * 1e9 / max(n_in, 1):.1f} ns per element")
    assert 150 <= n_in <= 170                 # 4 teeth x ~40 engaged elements
    # regression guard at 4x the cost measured on Plint (2026-11-03: ~50 us per step, ~290 ns per engaged
    # element + ~4 us per step); the 1 us target is not met (docs/models/cutting.md "Kernels and measured cost")
    assert best < 200e-6
