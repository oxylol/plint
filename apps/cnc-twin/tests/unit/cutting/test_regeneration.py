"""Regeneration through the surface memory and process damping by flank indentation."""

import math

import numpy as np
import pytest

from cnctwin.cutting import abi as A
from cnctwin.cutting import analytic as AN
from cnctwin.cutting.drive import Driver, straight_immersion
from tests.unit.cutting.cutfix import ALTINTAS, linear_layout

NB = 720
BIN = 2 * math.pi / NB
DT = 8e-6
OMEGA = -BIN / DT            # one memory bin per step: every read and write falls on a bin centre


def test_one_dof_regenerative_chip_thickness_is_exact():
    """h(t) = (c + u(t) - u(t - tau)) sin(phi) for a tool vibrating along the feed (Tlusty's regenerative
    chip thickness), to round-off, through the vector-form memory."""
    R, ap, c = 5e-3, 1e-3, 0.04e-3
    tool = AN.cylinder_mill(R, 1, ap, 1)
    ct, table, _ = linear_layout(tool, {"default": ALTINTAS}, OMEGA)
    drv = Driver(ct, table, 1, 1, memory_bins=NB)
    vf = c * abs(OMEGA) / (2 * math.pi)
    tip = lambda t: np.array([vf * t, 0.0, 0.0])
    amp, wv = 4e-6, 2 * math.pi * 1530.0
    u = lambda t: amp * math.sin(wv * t)

    def vibrate(d, t):
        d.defl.u_st_T0_m[:, 0] = u(t)
        d.defl.v_st_T0_m_s[:, 0] = amp * wv * math.cos(wv * t)

    lo, hi = straight_immersion(0.0, math.pi)
    drv.set_interval(lo, hi)
    drv.prefill_memory(tip, 0.0, OMEGA)
    n_rev = NB
    tau = n_rev * DT
    drv.run_rotating(n_rev, OMEGA, DT, tip, deflection_fn=vibrate, record=False)
    rec = drv.run_rotating(n_rev, OMEGA, DT, tip, t0_s=n_rev * DT, deflection_fn=vibrate)
    t = np.array(rec.t_s)
    h = np.array(rec.h)[:, 0]
    phi = np.mod(0.5 * math.pi - np.mod(OMEGA * t, 2 * math.pi), 2 * math.pi)
    inside = (phi > 0.05) & (phi < math.pi - 0.05)
    expect = (c + np.array([u(x) - u(x - tau) for x in t])) * np.sin(phi)
    assert np.max(np.abs(h[inside] - expect[inside])) < 1e-13


def test_process_damping_dissipates_and_scales_with_the_wear_land():
    R, ap, c = 5e-3, 2e-3, 0.05e-3
    tool = AN.cylinder_mill(R, 2, ap, 2, clear_rad=math.radians(7))
    out = {}
    for vb in (0.0, 0.1e-3, 0.2e-3):
        coeffs = {"default": dict(ALTINTAS, K_sp=3e13, sigma0=1.5e9, tau0=0.6e9, h_min_m=0.0)}
        nt, ne = tool.edge.elem_r_m.shape
        ct, table, _ = linear_layout(tool, coeffs, OMEGA, VB=np.full((nt, ne), vb))
        drv = Driver(ct, table, nt, ne, memory_bins=NB)
        vf = c * 2 * abs(OMEGA) / (2 * math.pi)
        tip = lambda t: np.array([vf * t, 0.0, 0.0])
        amp, wv = 3e-6, 2 * math.pi * 2100.0

        def vibrate(d, t):
            d.defl.u_st_T0_m[:, 1] = amp * math.sin(wv * t)
            d.defl.v_st_T0_m_s[:, 1] = amp * wv * math.cos(wv * t)

        lo, hi = straight_immersion(0.0, math.pi)
        drv.set_interval(lo, hi)
        drv.prefill_memory(tip, 0.0, OMEGA)
        drv.run_rotating(2 * NB, OMEGA, DT, tip, deflection_fn=vibrate, record=False)
        out[vb] = float(drv.acc.scalars[A.ACC_PROCESS_DAMPING_J])
        heat = sum(drv.acc.scalars[k] for k in (A.ACC_HEAT_SHEAR_J, A.ACC_HEAT_RAKE_J, A.ACC_HEAT_FLANK_J,
                                                A.ACC_PLOUGHING_J, A.ACC_PROCESS_DAMPING_J))
        # energy: shaft + feed = heat generated + work on the vibration
        assert drv.acc.scalars[A.ACC_SHAFT_WORK_J] + drv.acc.scalars[A.ACC_FEED_WORK_J] == \
            pytest.approx(heat + drv.cacc.scalars[A.CA_DEFLECTION_J], rel=1e-11)
    # a sharp edge without a wear land has no zero-clearance contact: nothing indents (the vibration slope
    # stays below the clearance); with a land L the indented area is ~ slope L^2 / 2
    assert out[0.0] == 0.0
    assert 0.0 < out[0.1e-3] < out[0.2e-3]
    assert out[0.2e-3] / out[0.1e-3] == pytest.approx(4.0, rel=0.25)


def test_process_damping_needs_vibration_not_the_steady_feed():
    """Only the vibration makes the wave the land indents: in a steady cut (no deflection, no work vibration) the
    feed slope v_f . n / V of the path the edge itself generates gives no process-damping force, also with a worn
    land and a large slope (feed per tooth 0.4 mm); the land's steady contact is the wear-land force."""
    R, ap, c = 5e-3, 2e-3, 0.4e-3
    tool = AN.cylinder_mill(R, 2, ap, 2, clear_rad=math.radians(7))
    nt, ne = tool.edge.elem_r_m.shape
    coeffs = {"default": dict(ALTINTAS, K_sp=3e13, sigma0=1.5e9, tau0=0.6e9, h_min_m=0.0)}
    ct, table, _ = linear_layout(tool, coeffs, OMEGA, VB=np.full((nt, ne), 0.2e-3))
    assert np.all(ct.pd[:, 0] > 0.0) and np.all(ct.pd[:, 1] > 0.0)
    drv = Driver(ct, table, nt, ne, memory_bins=NB)
    vf = c * 2 * abs(OMEGA) / (2 * math.pi)
    tip = lambda t: np.array([vf * t, 0.0, 0.0])
    lo, hi = straight_immersion(0.0, math.pi)
    drv.set_interval(lo, hi)
    drv.prefill_memory(tip, 0.0, OMEGA)
    drv.run_rotating(2 * NB, OMEGA, DT, tip, record=False)
    assert drv.acc.scalars[A.ACC_SHAFT_WORK_J] > 0.0
    assert drv.acc.scalars[A.ACC_PROCESS_DAMPING_J] == 0.0
    assert drv.acc.scalars[A.ACC_HEAT_FLANK_J] > 0.0              # the land's own contact
