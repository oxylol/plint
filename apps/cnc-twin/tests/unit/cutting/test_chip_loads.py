"""Loads of the chips on the tool (docs/models/cutting.md "Recutting"): a recut strike is cut like material, its force
bounded by the chips' section and spread over the crossing time; recut and packing loads act at an engaged station,
and their work on the deflection rates is booked with the element forces' (the run's energy balance)."""

import math

import numpy as np
import pytest

from cnctwin.cutting import abi as A
from cnctwin.cutting import analytic as AN
from cnctwin.cutting.drive import Driver, straight_immersion
from tests.unit.cutting.cutfix import ALTINTAS, linear_layout

OMEGA = -1047.0                # 10 000 rpm, M03
DT = 5e-6
RHO = 2700.0
R_TOOL = 5e-3
AP = 2e-3


def _driver(n_z=8):
    tool = AN.cylinder_mill(R_TOOL, 4, AP, n_z, helix_rad=math.radians(30))
    nt, ne = tool.edge.elem_r_m.shape
    ct, table, rows = linear_layout(tool, {"default": dict(ALTINTAS, h_min_m=0.0)}, OMEGA)
    return Driver(ct, table, nt, ne, rho_kg_m3=RHO, n_ticks=1), ct, table


def _voxel(drv, mass, x0, size, thick, cell=2e-3):
    occ = A.empty_occupancy(1, 1, 1)
    occ.origin_S_m[:] = [x0, -0.5 * cell, 0.0]
    occ.cell_m[0] = cell
    occ.mass_kg[0, 0, 0] = mass
    occ.size_m[0, 0, 0] = size
    occ.thickness_m[0, 0, 0] = thick
    drv.occ = occ


def test_a_recut_strike_is_cut_like_material_over_its_crossing_time():
    """A 2 mm voxel holding 2 mg of 40 um chips (0.6 mm volume-equivalent size), two thirds of it inside the D10's
    envelope: the teeth strike those two thirds once (not the voxel's whole mass), cut across the chips' section
    (thickness 40 um x width sqrt(V/h) = 1.68 mm): the force is the table's f(h) at h = 40 um on that width (about
    100 N tangential), lasting the crossing time V_struck / (h b V); the energy is the specific cutting energy times
    the struck volume. Before, the whole voxel took J = m V + u m / (rho V) within one L0 step."""
    drv, ct, table = _driver()
    mass, size, thick = 2e-6, 0.6e-3, 40e-6
    _voxel(drv, mass, 3.5e-3, size, thick)
    tip = lambda t: np.zeros(3)                       # standing tool: no chip of the work, only the struck chips
    n = 1200
    rec = drv.run_rotating(n, OMEGA, DT, tip, v_tip_fn=lambda t: np.zeros(3))
    struck = float(drv.cacc.scalars[A.CA_RECUT_KG])
    assert struck == pytest.approx(mass * 18.0 / 27.0, rel=1e-12)       # 18 of the 27 sub-cells inside r = 5 mm
    F = np.array(rec.wrench)[:, :3]
    on = np.linalg.norm(F, axis=1) > 0.0
    # the expected load: the linear table at h = 40 um, width min(sqrt(V_chip / h), the 2 mm of edge in the voxel)
    v_chip = math.pi / 6.0 * size ** 3
    width = min(math.sqrt(v_chip / thick), 2e-3)
    ft = (ALTINTAS["Ktc"] * thick + ALTINTAS["Kte"]) * width
    fr = (ALTINTAS["Krc"] * thick + ALTINTAS["Kre"]) * width
    peak = float(np.max(np.linalg.norm(F, axis=1)))
    assert peak == pytest.approx(math.hypot(ft, fr), rel=1e-9)
    assert peak < 200.0                               # the old spike: kilonewtons in one step
    # struck point: the centroid of the inside part (x 4.17 mm), speed |omega| r; crossing time V / (h b V)
    rc = (3.5e-3 + 2e-3 * (0.5 / 3)) * 0.5 + (3.5e-3 + 2e-3 * (1.5 / 3)) * 0.5
    t_cross = struck / RHO / (thick * width * abs(OMEGA) * rc)
    assert np.count_nonzero(on) * DT == pytest.approx(t_cross, abs=DT)
    impulse = np.sum(F[on], axis=0) * DT
    assert np.linalg.norm(impulse) == pytest.approx(math.hypot(ft, fr) * t_cross, rel=1e-9)
    # energy: all of it dissipated in the chips (no feed, no vibration): the table's specific energy x volume
    e_rec = float(drv.cacc.scalars[A.CA_RECUT_J])
    assert e_rec == pytest.approx(ft / (thick * width) * struck / RHO, rel=1e-9)
    a = drv.acc.scalars
    assert a[A.ACC_SHAFT_WORK_J] + a[A.ACC_FEED_WORK_J] == pytest.approx(e_rec, rel=1e-12)


def _vibrate(d, t):
    """Smooth tool and work vibration (station translations and rotations, the work's velocity) for the energy
    bookkeeping: rates that do work on every load."""
    w = 2 * math.pi * 1700.0
    nst = d.defl.u_st_T0_m.shape[0]
    for s in range(nst):
        ph = 0.3 * s
        d.defl.u_st_T0_m[s, :] = [2e-6 * math.sin(w * t + ph), 1e-6 * math.cos(w * t), 3e-7 * math.sin(w * t),
                                  1e-5 * math.sin(w * t), -2e-5 * math.cos(w * t + ph), 4e-6 * math.sin(w * t)]
        d.defl.v_st_T0_m_s[s, :] = [2e-6 * w * math.cos(w * t + ph), -1e-6 * w * math.sin(w * t),
                                    3e-7 * w * math.cos(w * t), 1e-5 * w * math.cos(w * t),
                                    2e-5 * w * math.sin(w * t + ph), 4e-6 * w * math.cos(w * t)]
    d.defl.u_work_S_m[:] = [5e-7 * math.sin(1.3 * w * t), 0.0, 2e-7 * math.cos(w * t)]
    d.defl.v_work_S_m_s[:] = [5e-7 * 1.3 * w * math.cos(1.3 * w * t), 0.0, -2e-7 * w * math.sin(w * t)]


def test_chip_loads_close_the_energy_identity_and_agree_with_the_modal_work():
    """A slot through a voxel of chips with flute packing, the tool and the work vibrating: shaft + feed = heat +
    recut + packing + work on the deflection rates, to round-off, and that deflection work is exactly what the modes
    receive from the published loads (station loads x station rates - total force x the work's velocity: dynamics'
    quadrature). Before, the recut impulse and the packing force reached the modes (through the total force) but
    not CA_DEFLECTION_J: the run's energy balance missed their work (the four examples that stopped at their first
    cut, docs/models/loop.md 14)."""
    drv, ct, table = _driver()
    _voxel(drv, 3e-6, 3.2e-3, 0.5e-3, 30e-6)
    drv.pack.torque_Nm[0] = 0.4
    drv.pack.force_T0_N[:] = [12.0, -7.0, 3.0]
    fz = 0.05e-3
    vf = fz * 4 * abs(OMEGA) / (2 * math.pi)
    tip = lambda t: np.array([vf * t, 0.0, 0.0])
    lo, hi = straight_immersion(0.0, math.pi)
    drv.set_interval(lo, hi)
    drv.prefill_memory(tip, 0.0, OMEGA)
    quad = 0.0
    n = 900
    for k in range(n):
        drv.run_rotating(1, OMEGA, DT, tip, t0_s=k * DT, deflection_fn=_vibrate, record=False)
        # what dynamics' corrector integrates with the predicted rates (modal_correct's quadrature)
        quad += DT * float(np.sum(drv.out.station_T0 * drv.defl.v_st_T0_m_s))
        quad -= DT * float(drv.out.force_S_N @ drv.defl.v_work_S_m_s)
    a = drv.acc.scalars
    c = drv.cacc.scalars
    assert c[A.CA_RECUT_KG] > 0.0 and c[A.CA_PACKING_J] > 0.0 and c[A.CA_RECUT_J] > 0.0
    heat = a[A.ACC_HEAT_SHEAR_J] + a[A.ACC_HEAT_RAKE_J] + a[A.ACC_HEAT_FLANK_J] + a[A.ACC_PLOUGHING_J] + \
        a[A.ACC_PROCESS_DAMPING_J]
    total_in = a[A.ACC_SHAFT_WORK_J] + a[A.ACC_FEED_WORK_J]
    assert total_in == pytest.approx(heat + c[A.CA_RECUT_J] + c[A.CA_PACKING_J] + c[A.CA_DEFLECTION_J], rel=1e-12)
    assert c[A.CA_DEFLECTION_J] == pytest.approx(quad, rel=1e-9, abs=1e-12 * abs(total_in))


def test_chip_loads_act_at_an_engaged_station():
    """The station loads (what the tool modes take) hold the recut and packing loads: their sum is the step's force
    on the tool, and only engaged stations carry load."""
    drv, ct, table = _driver()
    _voxel(drv, 3e-6, 3.2e-3, 0.5e-3, 30e-6)
    drv.pack.torque_Nm[0] = 0.4
    drv.pack.force_T0_N[:] = [12.0, -7.0, 3.0]
    lo, hi = straight_immersion(0.0, math.pi)
    drv.set_interval(lo, hi, stations=range(2, 6))
    tip = lambda t: np.zeros(3)
    for k in range(700):
        drv.run_rotating(1, OMEGA, DT, tip, t0_s=k * DT, v_tip_fn=lambda t: np.zeros(3), record=False)
        assert np.allclose(np.sum(drv.out.station_T0[:, :3], axis=0), drv.out.wrench_T0[:3], rtol=0, atol=1e-9)
        assert not np.any(drv.out.station_T0[:2]) and not np.any(drv.out.station_T0[6:])
