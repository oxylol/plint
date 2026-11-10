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


VF = 0.05e-3 * 4 * abs(OMEGA) / (2 * math.pi)       # f_z 0.05 mm
T_TOOTH = 2 * math.pi / (4 * abs(OMEGA))


def _grid(drv, mass_per_voxel, thick=40e-6, size=0.6e-3, cell=2e-3):
    """Chips everywhere around the tool: 7 x 7 x 2 voxels of 2 mm, the same mass in each."""
    occ = A.empty_occupancy(7, 7, 2)
    occ.origin_S_m[:] = [-7e-3, -7e-3, -1e-3]
    occ.cell_m[0] = cell
    occ.mass_kg[:] = mass_per_voxel
    occ.size_m[:] = size
    occ.thickness_m[:] = thick
    drv.occ = occ


def _one_scan(drv, tip, v_tip):
    """Steps over 1.5 tooth periods from rest: one recut scan (at the end of the first tooth period)."""
    n = int(1.5 * T_TOOTH / DT)
    return drv.run_rotating(n, OMEGA, DT, tip, v_tip_fn=lambda t: v_tip)


def test_the_teeth_meet_the_chips_in_the_layer_their_advance_sweeps_in_air():
    """Chips spread uniformly around a D10 cutter (ap 2 mm) feeding 0.05 mm per tooth along x through air: in one
    tooth period the teeth meet the chips of the layer the leading half of the envelope sweeps, f_z x 2R x ap, at the
    voxels' chip density (to the angle sampling's 0.1 %), not the chips inside the envelope."""
    drv, ct, table = _driver()
    m = 1e-6
    _grid(drv, m)
    v = np.array([VF, 0.0, 0.0])
    _one_scan(drv, lambda t: v * t, v)
    struck = float(drv.cacc.scalars[A.CA_RECUT_KG])
    swept = VF * T_TOOTH * 2 * R_TOOL * AP
    assert struck == pytest.approx(m / 8e-9 * swept, rel=2e-3)


def test_chips_in_the_flutes_and_ahead_in_material_are_not_cut():
    """The same chips with the cutter (1) plunging along its axis (a drill in its hole: the chips around it are in its
    flutes, carried out, never in front of a lip), (2) slotting, the leading half of every station engaged in
    material: nothing is struck. Before 2026-11-10 every chip inside the envelope cylinder was cut; the drill of
    examples/drill-tap-plate-1045 took 0.39 mg of its own flute chips at 431 N sideways (docs/models/loop.md 15)."""
    drv, ct, table = _driver()
    _grid(drv, 1e-6)
    v = np.array([0.0, 0.0, -VF])
    rec = _one_scan(drv, lambda t: v * t, v)
    assert float(drv.cacc.scalars[A.CA_RECUT_KG]) == 0.0
    assert not np.any(np.array(rec.wrench))
    drv, ct, table = _driver()
    _grid(drv, 1e-6)
    lo, hi = straight_immersion(0.0, math.pi)
    drv.set_interval(lo, hi)
    v = np.array([VF, 0.0, 0.0])
    drv.prefill_memory(lambda t: v * t, 0.0, OMEGA)
    _one_scan(drv, lambda t: v * t, v)
    assert float(drv.cacc.scalars[A.CA_RECUT_KG]) == 0.0


def _strike(mass, size=0.6e-3, thick=40e-6):
    drv, ct, table = _driver()
    _voxel(drv, mass, 3.5e-3, size, thick)
    v = np.array([VF, 0.0, 0.0])
    rec = _one_scan(drv, lambda t: v * t, v)
    F = np.array(rec.wrench)[:, :3]
    return drv, F


def test_a_recut_strike_is_cut_like_material():
    """A 2 mm voxel of 40 um chips (0.6 mm volume-equivalent size) straight ahead of the feeding D10. The chips are
    spread over the voxel, so the edges meet the struck ones all along the band's arc in the voxel (2 asin(1 / 4.97)
    = 23 deg) while they sweep it: the mean force cuts the struck volume at the table's specific energy f(h) / h over
    that time (14 N for 2 mg in the voxel), bounded by the section force of whole chips side by side (thickness
    40 um x width sqrt(V/h) = 1.68 mm: about 100 N), which a nearly solid voxel (20 mg) reaches; the spindle's work is
    the specific energy times the struck volume either way. Before 2026-11-10 every strike took the section force
    (the D6.8 drill: 431 N from 13 ug of chips near its axis); before 2026-11-09 the voxel took J = m V + u m / (rho V)
    within one L0 step."""
    v_chip = math.pi / 6.0 * 0.6e-3 ** 3
    arc = 2.0 * math.asin(1e-3 / 4.97e-3)
    l_edge = AP * 7.0 / 8.0                    # the edge length in the voxel: between the outer stations (8 per 2 mm)
    for mass, thick in ((2e-6, 40e-6), (20e-6, 10e-6)):
        b_r = min(math.sqrt(v_chip / thick), l_edge)
        ft = (ALTINTAS["Ktc"] * thick + ALTINTAS["Kte"]) * b_r
        fr = (ALTINTAS["Krc"] * thick + ALTINTAS["Kre"]) * b_r
        drv, F = _strike(mass, thick=thick)
        struck = float(drv.cacc.scalars[A.CA_RECUT_KG])
        # the leading face inside the voxel (|y| <= 1 mm, z 0-2 mm) sweeps f_z x 2 mm x 2 mm in the tooth period
        assert struck == pytest.approx(mass / 8e-9 * VF * T_TOOTH * 4e-6, rel=0.07)
        energy = ft / (thick * b_r) * struck / RHO                 # the specific energy x the struck volume
        a = drv.acc.scalars
        if mass < 1e-5:            # (the dense strike lasts beyond the half tooth period simulated after the scan)
            assert a[A.ACC_SHAFT_WORK_J] == pytest.approx(energy, rel=1e-6)
        assert a[A.ACC_SHAFT_WORK_J] + a[A.ACC_FEED_WORK_J] == pytest.approx(float(drv.cacc.scalars[A.CA_RECUT_J]),
                                                                             rel=1e-12)
        mag = np.linalg.norm(F, axis=1)
        on = mag > 0.0
        peak = float(mag.max())
        w_c = max(math.sqrt(v_chip / thick), thick)
        n_side = max(mass / (RHO * 8e-9) * l_edge / w_c, 1.0)
        section = math.hypot(ft, fr) * n_side
        if mass < 1e-5:
            # the mean force over the sweep of the arc: energy / (V t_pass), V at the face's centroid (r ~ 4.97 mm)
            t_pass = arc / abs(OMEGA)
            mean_ft = energy / (abs(OMEGA) * 4.97e-3 * t_pass)
            assert peak == pytest.approx(mean_ft * math.hypot(ft, fr) / ft, rel=0.05)
            assert peak < 0.2 * section
            assert np.count_nonzero(on) * DT == pytest.approx(t_pass, rel=0.05, abs=DT)
        else:
            # dense chips: whole sections side by side (the mean over the sweep would be about three times more)
            assert peak == pytest.approx(section, rel=1e-9)


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
    # half immersion (down milling): the leading face on the other side is in air, where the voxel's chips lie
    lo, hi = straight_immersion(0.5 * math.pi, math.pi)
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
    lo, hi = straight_immersion(0.5 * math.pi, math.pi)
    drv.set_interval(lo, hi, stations=range(2, 6))
    v = np.array([VF, 0.0, 0.0])
    tip = lambda t: v * t
    for k in range(700):
        drv.run_rotating(1, OMEGA, DT, tip, t0_s=k * DT, v_tip_fn=lambda t: v, record=False)
        assert np.allclose(np.sum(drv.out.station_T0[:, :3], axis=0), drv.out.wrench_T0[:3], rtol=0, atol=1e-9)
        assert not np.any(drv.out.station_T0[:2]) and not np.any(drv.out.station_T0[6:])
    assert float(drv.cacc.scalars[A.CA_RECUT_KG]) > 0.0
