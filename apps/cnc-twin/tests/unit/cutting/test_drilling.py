"""Drilling: thrust and torque against the lip and chisel integrals, breakthrough, cross-hole breakout,
the lateral (walk) force of a one-sided entry, and a tooling-built twist drill."""

import math

import numpy as np
import pytest

from cnctwin.core.abi import EDGE_CHISEL, EDGE_LIP, EDGE_MARGIN
from cnctwin.cutting import abi as A
from cnctwin.cutting import analytic as AN
from cnctwin.cutting.drive import Driver
from tests.unit.cutting.cutfix import built, linear_layout, rev_steps

R, P, W = 5e-3, math.radians(59.0), 0.75e-3
OMEGA = -523.6
DT = 16e-6
F_REV = 0.2e-3
LIP = dict(Ktc=1.6e9, Krc=0.7e9, Kac=0.0, Kte=25e3, Kre=35e3, Kae=0.0)
CHISEL = dict(Ktc=0.0, Krc=0.0, Kac=0.0, Kte=40e3, Kre=120e3, Kae=0.0)
COEFFS = {EDGE_LIP: LIP, EDGE_CHISEL: CHISEL}


def _drill_run(revs=2, intervals=None, n_lip=40, n_chisel=8):
    tool = AN.two_lip_drill(R, P, W, n_lip, n_chisel)
    ct, table, _ = linear_layout(tool, COEFFS, OMEGA)
    nt, ne = tool.edge.elem_r_m.shape
    drv = Driver(ct, table, nt, ne)
    vz = -F_REV * abs(OMEGA) / (2 * math.pi)
    tip = lambda t: np.array([0.0, 0.0, vz * t])
    drv.set_full_circle()
    drv.prefill_memory(tip, 0.0, OMEGA)
    return tool, ct, drv, tip


def test_thrust_and_torque_match_the_lip_and_chisel_integrals():
    tool, ct, drv, tip = _drill_run()
    n = rev_steps(OMEGA, DT)
    drv.run_rotating(n, OMEGA, DT, tip, record=False)
    rec = drv.run_rotating(n, OMEGA, DT, tip, t0_s=n * DT)
    W_ = np.array(rec.wrench)
    thrust, torque = AN.drilling_lip_chisel(LIP["Ktc"], LIP["Krc"], LIP["Kte"], LIP["Kre"], F_REV, R, W, P,
                                            CHISEL["Kte"], CHISEL["Kre"])
    assert W_[:, 2].mean() == pytest.approx(thrust, rel=2e-3)          # thrust pushes the drill back
    assert W_[:, 5].mean() == pytest.approx(torque, rel=2e-3)          # resists the clockwise rotation (M03)
    # symmetric two-lip drill: no lateral force
    assert np.abs(W_[:, :2]).max() < 1e-6 * thrust
    # chip thickness of every lip element: h = (f / 2) sin p; chisel: f / 2
    lips = ct.kind == EDGE_LIP
    assert np.allclose(np.array(rec.h)[-1][lips], 0.5 * F_REV * math.sin(P), rtol=1e-6)
    assert np.allclose(np.array(rec.h)[-1][ct.kind == EDGE_CHISEL], 0.5 * F_REV, rtol=1e-6)


def test_breakthrough_and_cross_hole_breakout_events():
    tool, ct, drv, tip = _drill_run()
    n = rev_steps(OMEGA, DT)
    drv.run_rotating(n, OMEGA, DT, tip, record=False)
    thrust_before = drv.out.wrench_T0[2]
    # cross hole: the outer half of the lips meets a gap of 40 deg in every revolution
    outer = [s for s in range(ct.st_z_m.shape[0]) if ct.r_xy_m[ct.st_elem[ct.st_ptr[s]]] > 0.6 * R]
    for s in outer:
        drv.iv.lo_rad[s, 0] = 0.4
        drv.iv.hi_rad[s, 0] = 0.4 + 2 * math.pi - math.radians(40.0)
    drv.run_rotating(n, OMEGA, DT, tip, t0_s=n * DT, record=False)
    sh = drv.ev.shocks.rec[:drv.ev.shocks.n[0]]
    assert np.sum(sh[:, 7] == A.SHOCK_BREAKOUT) == 2                  # one per lip at the first breakout
    # breakthrough: the material under the chisel is gone, the lips still cut
    centre = [s for s in range(ct.st_z_m.shape[0]) if ct.st_centre[s]]
    assert centre
    for s in range(ct.st_z_m.shape[0]):
        if ct.kind[ct.st_elem[ct.st_ptr[s]]] == EDGE_CHISEL:
            drv.iv.count[s] = 0
    drv.ev.shocks.n[:] = 0
    for s in outer:
        drv.iv.lo_rad[s, 0] = 0.0
        drv.iv.hi_rad[s, 0] = 2 * math.pi
    drv.ev.shocks.n[:] = 0
    tool2, ct2, drv2, tip2 = _drill_run()
    drv2.run_rotating(n, OMEGA, DT, tip2, record=False)
    for s in range(ct2.st_z_m.shape[0]):
        if ct2.kind[ct2.st_elem[ct2.st_ptr[s]]] == EDGE_CHISEL:
            drv2.iv.count[s] = 0
    drv2.run_rotating(20, OMEGA, DT, tip2, t0_s=n * DT, record=False)
    sh2 = drv2.ev.shocks.rec[:drv2.ev.shocks.n[0]]
    assert np.sum(sh2[:, 7] == A.SHOCK_BREAKTHROUGH) == 2
    _, t_chisel = 0.0, 2 * CHISEL["Kre"] * W
    assert drv2.out.wrench_T0[2] == pytest.approx(thrust_before - t_chisel, rel=2e-3)


def test_one_sided_entry_gives_a_lateral_walk_force():
    tool, ct, drv, tip = _drill_run()
    n = rev_steps(OMEGA, DT)
    # entry on a surface inclined about y: only the side x > 0 (T0 angles -60..60 deg) is in material
    drv.set_interval(-math.radians(60.0), math.radians(60.0))
    rec = drv.run_rotating(2 * n, OMEGA, DT, tip)
    W_ = np.array(rec.wrench)[n:]
    fx, fy = W_[:, 0].mean(), W_[:, 1].mean()
    assert math.hypot(fx, fy) > 0.05 * W_[:, 2].mean()
    # the normal forces of the engaged lips push the drill away from the engaged side (toward -x); the
    # tangential forces add a component along -y for the clockwise rotation
    assert fx < 0.0


def test_tooling_twist_drill_runs_with_margins_rubbing_only_on_interference():
    g = built("twist_drill")
    nt, ne = g.edge.elem_r_m.shape
    coeffs = {EDGE_LIP: LIP, EDGE_CHISEL: CHISEL, EDGE_MARGIN: dict(K_sp=2e13, mu=0.3), "default": LIP}
    ct, table, _ = linear_layout(g, coeffs, OMEGA)
    drv = Driver(ct, table, nt, ne)
    vz = -F_REV * abs(OMEGA) / (2 * math.pi)
    tip = lambda t: np.array([0.0, 0.0, vz * t])
    drv.set_full_circle()
    drv.prefill_memory(tip, 0.0, OMEGA)
    n = rev_steps(OMEGA, DT)
    rec = drv.run_rotating(n, OMEGA, DT, tip)
    W_ = np.array(rec.wrench)
    assert W_[:, 2].mean() > 0.0 and W_[:, 5].mean() > 0.0
    assert np.all(np.isfinite(W_))
    h = np.array(rec.h)[-1]
    margins = ct.kind == EDGE_MARGIN
    assert np.all(h[margins] <= 1e-12)                 # a straight hole: the margins do not interfere
    assert np.all(h[ct.kind == EDGE_LIP] > 0.0)
