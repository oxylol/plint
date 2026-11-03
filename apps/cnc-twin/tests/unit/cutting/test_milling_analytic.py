"""Milling forces against closed forms: Altintas' average forces (slotting, up / down side milling, helix),
instantaneous straight-tooth forces, runout tooth skipping, variable pitch delays, end-edge load of a
plunge and a helical ramp, energy closure, emission mass and shocks."""

import math

import numpy as np
import pytest

from cnctwin.cutting import abi as A
from cnctwin.cutting import analytic as AN
from cnctwin.cutting.drive import Driver, straight_immersion
from tests.unit.cutting.cutfix import ALTINTAS, linear_layout, rev_steps, steady_tip

R = 5e-3
AP = 2e-3
OMEGA = -1047.0
DT = 8e-6
C = 0.05e-3


def _run(tool, coeffs, phi_st, phi_ex, n_teeth, c=C, warm=2, revs=1, **kw):
    ct, table, _ = linear_layout(tool, coeffs, OMEGA, **kw)
    nt, ne = tool.edge.elem_r_m.shape
    drv = Driver(ct, table, nt, ne, rho_kg_m3=2700.0)
    vf = c * n_teeth * abs(OMEGA) / (2 * math.pi)
    tip = steady_tip(vf)
    lo, hi = straight_immersion(phi_st, phi_ex)
    drv.set_interval(lo, hi)
    drv.prefill_memory(tip, 0.0, OMEGA)
    n = rev_steps(OMEGA, DT)
    drv.run_rotating(warm * n, OMEGA, DT, tip, record=False)
    rec = drv.run_rotating(revs * n, OMEGA, DT, tip, t0_s=warm * n * DT)
    return drv, np.array(rec.wrench), rec


@pytest.mark.parametrize("case,phi_st,phi_ex", [
    ("slot", 0.0, math.pi),
    ("up", 0.0, math.acos(1.0 - 2e-3 / R)),
    ("down", math.pi - math.acos(1.0 - 2e-3 / R), math.pi),
])
def test_average_forces_straight_teeth(case, phi_st, phi_ex):
    tool = AN.cylinder_mill(R, 4, AP, 4)
    drv, W, _ = _run(tool, {"default": ALTINTAS}, phi_st, phi_ex, 4)
    mean = W[:, :3].mean(axis=0)
    exp = np.array(AN.milling_average_forces(*ALTINTAS.values(), 4, AP, C, phi_st, phi_ex))
    # trochoidal path and the 0.24 deg step: < 0.5 % of the force magnitude
    assert np.allclose(mean, exp, atol=5e-3 * np.linalg.norm(exp)), (mean, exp)


def test_average_forces_helical_cutter():
    """The mean force over a tooth period does not depend on the helix (Altintas 2012 ch. 2.4) when the edge
    coefficients are per unit axial depth; the kernel applies Ke per unit edge length (dS = dz / cos i), so
    the per-length coefficients are Altintas' times cos(i)."""
    beta = math.radians(30.0)
    tool = AN.cylinder_mill(R, 4, AP, 8, helix_rad=beta)
    coeffs = dict(ALTINTAS)
    for key in ("Kte", "Kre", "Kae"):
        coeffs[key] = ALTINTAS[key] * math.cos(beta)
    drv, W, _ = _run(tool, {"default": coeffs}, 0.0, math.pi, 4)
    mean = W[:, :3].mean(axis=0)
    exp = np.array(AN.milling_average_forces(*ALTINTAS.values(), 4, AP, C, 0.0, math.pi))
    assert np.allclose(mean, exp, atol=5e-3 * np.linalg.norm(exp)), (mean, exp)


def test_instantaneous_force_single_tooth():
    tool = AN.cylinder_mill(R, 1, AP, 2)
    drv, W, rec = _run(tool, {"default": ALTINTAS}, 0.0, math.pi, 1)
    t = np.array(rec.t_s)
    # immersion angle of the tooth (phi clockwise from +Y, tooth 0 at T0 angle omega t)
    alpha = np.mod(OMEGA * t, 2 * math.pi)
    phi = np.mod(0.5 * math.pi - alpha, 2 * math.pi)
    inside = (phi > 0.02) & (phi < math.pi - 0.02)
    for k in np.nonzero(inside)[0][::25]:
        exp = AN.milling_forces_at(*ALTINTAS.values(), AP, C, float(phi[k]))
        assert np.allclose(W[k, :3], exp, atol=4e-3 * np.linalg.norm(exp) + 0.05), (phi[k], W[k, :3], exp)


def test_runout_tooth_skipping_through_the_memory():
    e = 0.03e-3
    tool = AN.cylinder_mill(R, 2, AP, 1, runout_T_m=(e, 0.0, 0.0))
    drv, W, rec = _run(tool, {"default": ALTINTAS}, 0.0, math.pi, 2, warm=4, revs=2)
    H = np.array(rec.h)                      # [steps, n]: h of tooth 0 (elem 0) and tooth 1 (elem 1)
    peaks = np.clip(H.max(axis=0), 0.0, None)
    exp = AN.runout_chip_thickness(C, [R + e, R - e])
    assert peaks[1] == pytest.approx(0.0, abs=1e-9)          # the small tooth never reaches the surface
    assert peaks[0] == pytest.approx(exp[0], rel=2e-3)       # the large tooth cuts both feeds
    assert exp[0] == pytest.approx(2 * C, rel=1e-12)


def test_variable_pitch_delays():
    pitch = np.radians([80.0, 100.0, 80.0, 100.0])
    tool = AN.cylinder_mill(R, 4, AP, 1, pitch_rad=pitch)
    drv, W, rec = _run(tool, {"default": ALTINTAS}, 0.0, math.pi, 4, warm=3, revs=2)
    peaks = np.array(rec.h).max(axis=0)
    # tooth j trails tooth j-1 by pitch[j-1]: it cuts the feed of that delay
    exp = C * 4 * np.roll(pitch, 1) / (2 * math.pi)
    assert np.allclose(peaks, exp, rtol=3e-3), (peaks, exp)


def test_energy_closure_and_emission_mass():
    tool = AN.cylinder_mill(R, 4, AP, 4, helix_rad=math.radians(30))
    drv, W, rec = _run(tool, {"default": ALTINTAS}, 0.0, math.pi, 4, warm=1, revs=2)
    a = drv.acc.scalars
    heat = a[A.ACC_HEAT_SHEAR_J] + a[A.ACC_HEAT_RAKE_J] + a[A.ACC_HEAT_FLANK_J] + a[A.ACC_PLOUGHING_J] + \
        a[A.ACC_PROCESS_DAMPING_J]
    assert a[A.ACC_SHAFT_WORK_J] + a[A.ACC_FEED_WORK_J] == pytest.approx(heat, rel=1e-12)
    # analytic chip volume vs the removed volume of a slot: width 2R x depth AP x feed travelled
    vf = C * 4 * abs(OMEGA) / (2 * math.pi)
    t_total = 3 * rev_steps(OMEGA, DT) * DT
    assert a[A.ACC_CHIP_VOLUME_M3] == pytest.approx(2 * R * AP * vf * t_total, rel=2e-3)
    rows = drv.ev.passes.rec[:drv.ev.passes.n[0]]
    assert rows.shape[0] >= 4 * 3 - 1
    # each complete pass (the first ones began before the run started): mass = rho x 2 R AP C
    assert np.allclose(rows[4:, 4], 2700.0 * 2 * R * AP * C, rtol=5e-3)
    assert rows[:, 4].sum() <= 2700.0 * a[A.ACC_CHIP_VOLUME_M3] * (1 + 1e-12)
    assert np.all(rows[:, 11] > 0) and np.all(rows[:, 13] > 0)        # temperature and width filled


def test_entry_exit_shocks_and_helix_loading_time():
    beta = math.radians(30.0)
    tool = AN.cylinder_mill(R, 2, AP, 8, helix_rad=beta)
    drv, W, rec = _run(tool, {"default": ALTINTAS}, 0.0, math.pi, 2, warm=1, revs=1)
    sh = drv.ev.shocks.rec[:drv.ev.shocks.n[0]]
    entries = sh[sh[:, 7] == A.SHOCK_ENTRY]
    exits = sh[sh[:, 7] == A.SHOCK_EXIT]
    assert len(entries) >= 3 and len(exits) >= 3
    # the edge sweeps into the cut over the helix lag: (AP - one element) tan(beta) / R of rotation
    lag = (AP - AP / 8) * math.tan(beta) / R
    assert np.allclose(entries[1:, 6], lag / abs(OMEGA), rtol=0.02)
    # up-milling entry at phi = 0: the first element enters with a vanishing chip (row 0: the tooth that was
    # already inside the cut when the run started)
    assert np.all(entries[1:, 3] < 0.01 * C)
    # full contact: every element of the tooth is in cut; the force then is the tooth force at that angle
    assert np.all(entries[:, 5] > 0.0)
    # surface events at both slot walls (tangent boundary points), one pair per element pass
    ev = drv.ev.surface.rec[:drv.ev.surface.n[0]]
    y = ev[:, 4]
    assert np.any(y > R * 0.999) and np.any(y < -R * 0.999)
    assert np.all(np.abs(np.abs(y) - R) < 2e-6)


def test_plunge_end_edge_load_and_helical_ramp():
    n_teeth, n_r, r0 = 2, 10, 0.5e-3
    tool = AN.end_face_mill(R, n_teeth, n_r, r0_m=r0)
    coeffs = {"default": ALTINTAS}
    f_rev = 0.08e-3
    vz = -f_rev * abs(OMEGA) / (2 * math.pi)
    for helical in (False, True):
        ct, table, _ = linear_layout(tool, coeffs, OMEGA)
        drv = Driver(ct, table, n_teeth, n_r)
        if helical:
            rad, w = 2e-3, 30.0
            tip = lambda t: np.array([rad * math.cos(w * t), rad * math.sin(w * t), vz * t])
        else:
            tip = steady_tip(0.0, vz_m_s=vz)
        drv.set_full_circle()
        drv.prefill_memory(tip, 0.0, OMEGA)
        n = rev_steps(OMEGA, DT)
        drv.run_rotating(2 * n, OMEGA, DT, tip, record=False)
        rec = drv.run_rotating(n, OMEGA, DT, tip, t0_s=2 * n * DT)
        W = np.array(rec.wrench)
        h_t = f_rev / n_teeth
        thrust = n_teeth * (ALTINTAS["Krc"] * h_t + ALTINTAS["Kre"]) * (R - r0)
        torque = n_teeth * (ALTINTAS["Ktc"] * h_t + ALTINTAS["Kte"]) * 0.5 * (R * R - r0 * r0)
        assert W[:, 2].mean() == pytest.approx(thrust, rel=2e-3)          # pushes the tool toward the spindle
        assert W[:, 5].mean() == pytest.approx(torque, rel=2e-3)          # resists the clockwise rotation
        assert np.allclose(np.array(rec.h)[-1], h_t, rtol=2e-3)
