"""Stability lobes: semi-discretisation against the Insperger & Stepan (2004) benchmark charts (validation cases),
zero-order and multi-frequency solutions against semi-discretisation, Tlusty's turning limit, variable-pitch
stabilisation, mode coupling and process damping."""

from __future__ import annotations

import math

import numpy as np
import pytest
import yaml

from cnctwin.dynamics import semidisc as SD
from cnctwin.dynamics import sld as SL
from cnctwin.paths import VALIDATION

TWO_PI = 2.0 * math.pi
FN, ZETA, MASS = 922.0, 0.011, 0.03993           # Insperger & Stepan 2004, Sect. 4
KT, KN = 6e8, 2e8
RPM = TWO_PI / 60.0
FRF1 = SL.ModalFRF.single(FN, ZETA, MASS, [1.0, 0.0, 0.0])
FRF2 = SL.ModalFRF.combine(SL.ModalFRF.single(FN, ZETA, MASS, [1.0, 0.0, 0.0]),
                           SL.ModalFRF.single(FN, ZETA, MASS, [0.0, 1.0, 0.0]))


def _case(name):
    with open(VALIDATION / "cases" / "stability_lobe" / f"{name}.yaml") as fh:
        doc = yaml.safe_load(fh)
    curve = doc["reference_curves"][0]
    return np.array(curve["points"], float), float(curve["uncertainty"])


def _sd(cut, frf, rpm, a_max=0.02):
    return SD.limit_at_speed(cut, frf, rpm * RPM, m=40, a_max=a_max, n_scan=20, tol_rel=2e-3)[0]


@pytest.mark.parametrize("name,frf,ad", [("stability-insperger2004-sd-1dof-down-ad0.5", FRF1, 0.5),
                                         ("stability-insperger2004-sd-2dof-down-ad1", FRF2, 1.0)])
def test_insperger_stepan_benchmark(name, frf, ad):
    """In the lobe troughs (digitised depth within 1.3 x the chart's lowest point, the limit changing by less than
    8 % within the source's 50 rpm grid) the semi-discretisation (m = 40, the paper's own setting) reproduces the
    digitised boundary within its stated uncertainty plus 3 %. The steep flanks are blunted and shifted by the
    source's 400 x 200 grid and rendering (the case's notes); there the semi-discretisation, the time-domain
    simulation and the zero-order solution agree with each other instead (test_time_domain.py). The a/D = 0.05 chart
    reads 7-15 % above the converged semi-discretisation and the time-domain onset in its troughs (model doc 5.6);
    it is checked against the time domain there."""
    points, unc = _case(name)
    cut = SL.classic_milling(2, KT, KN, *SL.immersion_angles(ad, True))
    floor = float(np.min(points[:, 1]))
    compared = 0
    for rpm, mm in points:
        if mm > 1.3 * floor:
            continue
        vals = [_sd(cut, frf, rpm + d) * 1e3 for d in (-50.0, 0.0, 50.0)]
        if (max(vals) - min(vals)) > 0.08 * vals[1]:
            continue
        compared += 1
        assert abs(vals[1] - mm) <= unc + 0.03 * mm, (rpm, mm, vals)
    assert compared >= 4


def test_zero_order_against_semidiscretisation_slotting():
    """Full immersion: the zero-order solution is within 3 % of semi-discretisation at the lobe minima."""
    cut = SL.classic_milling(2, KT, KN, *SL.immersion_angles(1.0, True))
    om = FRF2.frequency_grid(3000, 0.5, 2.0)
    for rpm in (7700.0, 10500.0, 18000.0):
        a_z = SL.limit_at_speed(cut, FRF2, rpm * RPM, om).depth_limit_m
        a_s = _sd(cut, FRF2, rpm, a_max=0.005)
        assert abs(a_z / a_s - 1.0) < 0.03, (rpm, a_z, a_s)


def test_lobes_envelope_matches_speed_solver():
    """The frequency-scan lobes and the per-speed solver give the same zero-order limit."""
    cut = SL.classic_milling(2, KT, KN, *SL.immersion_angles(0.5, True))
    speeds = np.array([6400.0, 8358.0, 11908.0]) * RPM
    lob = SL.zoa_lobes(cut, FRF1, speeds, n_freq=6000, n_lobes=10, lo_factor=0.5, hi_factor=2.0)
    om = FRF1.frequency_grid(6000, 0.5, 2.0)
    for w, a_env in zip(speeds, lob.depth_limit_m):
        a = SL.limit_at_speed(cut, FRF1, w, om).depth_limit_m
        assert abs(a_env / a - 1.0) < 2e-3
    assert lob.lobes and all(np.all(np.diff(l["speed_rad_s"]) >= 0.0) for l in lob.lobes)


def test_multi_frequency_recovers_the_flip_lobe():
    """Low immersion (a/D = 0.05): the zero-order solution misses the period-doubling lobe near 18000 rpm (Bayly et
    al. 2003); the multi-frequency solution with 3 tooth-passing harmonics is within 6 % of semi-discretisation."""
    cut = SL.classic_milling(2, KT, KN, *SL.immersion_angles(0.05, True))
    om = FRF1.frequency_grid(2000, 0.3, 3.0)
    w = 18000.0 * RPM
    a_z = SL.limit_at_speed(cut, FRF1, w, om).depth_limit_m
    a_m = SL.limit_at_speed(cut, FRF1, w, om, method="mf", harmonics=3).depth_limit_m
    a_s = _sd(cut, FRF1, 18000.0, a_max=0.05)
    assert a_z > 2.0 * a_s
    assert abs(a_m / a_s - 1.0) < 0.06
    mu = SD.multipliers(cut, FRF1, w, 1.05 * a_s, 40)[0]
    assert SD.bifurcation_kind(mu) == "flip"


def test_turning_tlusty_limit():
    """Oriented single mode: the minimum of the lobes is Tlusty's 2 zeta (1 + zeta) k / (K_f mu) (Tlusty & Polacek
    1963); semi-discretisation of the same cut agrees within 1 %."""
    fn, zeta, k = 600.0, 0.03, 2e7
    frf = SL.ModalFRF.single(fn, zeta, k / (TWO_PI * fn) ** 2, [1.0, 0.0, 0.0])
    beta = math.radians(30.0)
    cut = SL.turning_cut(2e9, [1.0, 0.0, 0.0], [math.cos(beta), 0.0, math.sin(beta)])
    closed = 2.0 * zeta * (1.0 + zeta) * k / (2e9 * math.cos(beta))
    lob = SL.zoa_lobes(cut, frf, np.linspace(500.0, 4000.0, 2000) * RPM, n_freq=20000, n_lobes=60, lo_factor=0.8,
                       hi_factor=1.5)
    assert abs(float(np.min(lob.depth_limit_m)) / closed - 1.0) < 1e-5
    om = frf.frequency_grid(6000, 0.8, 1.5)
    for rpm in (2500.0, 3500.0):
        a_z = SL.limit_at_speed(cut, frf, rpm * RPM, om).depth_limit_m
        a_s = SD.limit_at_speed(cut, frf, rpm * RPM, m=40, a_max=0.2, n_scan=16, tol_rel=1e-3,
                                points_per_period=24)[0]
        assert abs(a_s / a_z - 1.0) < 0.01, (rpm, a_z, a_s)


def test_variable_pitch_stabilisation():
    """A 4-flute cutter with alternating pitch 90 +- d, d = pi Omega / (2 omega_c) (Altintas, Engin & Budak 1999):
    at the design speed its limit is far above the uniform pitch's; the variable-pitch zero-order solution agrees
    with semi-discretisation (per-tooth delays) within 6 %."""
    rpm = 8000.0
    w = rpm * RPM
    d = math.degrees(math.pi * w / (2.0 * TWO_PI * FN * 1.05))
    pitch = [90.0 + d, 90.0 - d, 90.0 + d, 90.0 - d]
    phi = SL.immersion_angles(1.0, True)
    uniform = SL.classic_milling(4, KT, KN, *phi)
    variable = SL.classic_milling(4, KT, KN, *phi, pitch_deg=pitch)
    assert not variable.uniform_pitch
    om = FRF2.frequency_grid(3000, 0.5, 2.0)
    a_u = SL.limit_at_speed(uniform, FRF2, w, om).depth_limit_m
    a_v = SL.limit_at_speed(variable, FRF2, w, om).depth_limit_m
    assert a_v > 1.3 * a_u
    s_u = SD.limit_at_speed(uniform, FRF2, w, m=30, a_max=0.005, n_scan=20, tol_rel=2e-3)[0]
    s_v = SD.limit_at_speed(variable, FRF2, w, m=30, a_max=0.005, n_scan=20, tol_rel=2e-3)[0]
    assert abs(a_u / s_u - 1.0) < 0.06 and abs(a_v / s_v - 1.0) < 0.06, (a_u, s_u, a_v, s_v)


def test_mode_coupling_limit():
    """Two orthogonal, nearly undamped modes with the force between them: the eigenvalue criterion meets the
    closed-form flutter onset tr(A)^2 = 4 det(A) of A = W^2 - a Psi^T K0 Psi (Tobias 1965; Tlusty & Polacek 1963)."""
    w1, w2 = TWO_PI * 500.0, TWO_PI * 560.0
    frf = SL.ModalFRF.combine(SL.ModalFRF.single(500.0, 1e-4, 1.0, [1, 0, 0]), SL.ModalFRF.single(560.0, 1e-4, 1.0, [0, 1, 0]))
    n = [math.cos(math.radians(45.0)), math.sin(math.radians(45.0)), 0.0]
    u = [math.cos(math.radians(-25.0)), math.sin(math.radians(-25.0)), 0.0]
    cut = SL.turning_cut(1e9, n, u)
    a_mc = SL.mode_coupling_limit(cut, frf, a_max=10.0, n_scan=200, tol_rel=1e-7)
    Kq = frf.psi @ cut.mean() @ frf.psi.T
    disc = lambda a: np.trace(np.diag([w1 * w1, w2 * w2]) - a * Kq) ** 2 - 4.0 * np.linalg.det(np.diag([w1 * w1, w2 * w2]) - a * Kq)
    from scipy.optimize import brentq
    grid = np.geomspace(1e-9, 10.0, 4000)
    i = next(i for i in range(1, len(grid)) if disc(grid[i]) < 0.0)
    closed = brentq(disc, grid[i - 1], grid[i])
    assert abs(a_mc / closed - 1.0) < 2e-3
    # without regeneration and with the force along the normal there is no coupling instability
    assert math.isinf(SL.mode_coupling_limit(SL.turning_cut(1e9, n, n), frf, a_max=10.0, n_scan=60, tol_rel=1e-6))


def test_process_damping_iteration():
    """The process-damped limit is the undamped one at high speed (the indentation damping ~ 1 / V vanishes) and
    rises at low speed; it is a fixed point of a = limit(G_pd(a))."""
    phi = SL.immersion_angles(0.5, True)
    Ksp, land, R = 1e13, 1e-4, 0.01
    cut = SL.classic_milling(2, KT, KN, *phi, K_sp_N_per_m3=Ksp, land_m=land, radius_m=R)
    bare = SL.classic_milling(2, KT, KN, *phi)
    om = FRF1.frequency_grid(3000, 0.5, 2.0)
    hi = 60000.0 * RPM
    a_pd, _ = SL.limit_with_process_damping(cut, FRF1, hi, om, tol_rel=1e-5, max_iter=60)
    a_0 = SL.limit_at_speed(bare, FRF1, hi, om).depth_limit_m
    # C ~ K_sp L^2 / (4 V): at V = 63 m/s the process damping is 3 % of the structural damping at the limit
    assert 1.0 <= a_pd.depth_limit_m / a_0 < 1.04
    lo = 1500.0 * RPM
    a_pd, it = SL.limit_with_process_damping(cut, FRF1, lo, om, tol_rel=1e-5, max_iter=60)
    a_0 = SL.limit_at_speed(bare, FRF1, lo, om).depth_limit_m
    assert a_pd.depth_limit_m > 1.5 * a_0 and it < 60
    check = SL.limit_at_speed(cut, FRF1, lo, om, depth_for_pd=a_pd.depth_limit_m).depth_limit_m
    assert abs(check / a_pd.depth_limit_m - 1.0) < 1e-3
