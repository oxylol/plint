"""Time-domain chatter onset with the real kernels (predictor-corrector, first-order hold) against the
semi-discretisation and zero-order limits, process damping, and chatter detection on the simulated vibration."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cnctwin.dynamics import semidisc as SD
from cnctwin.dynamics import sld as SL
from cnctwin.dynamics.chatter import ChatterSettings, analyse
from cnctwin.dynamics.drive import BenchModes, onset_depth, run_benchmark

TWO_PI = 2.0 * math.pi
RPM = TWO_PI / 60.0
FN, ZETA, MASS = 922.0, 0.011, 0.03993
KT, KN = 6e8, 2e8
FRF1 = SL.ModalFRF.single(FN, ZETA, MASS, [1.0, 0.0, 0.0])
FRF2 = SL.ModalFRF.combine(SL.ModalFRF.single(FN, ZETA, MASS, [1.0, 0.0, 0.0]),
                           SL.ModalFRF.single(FN, ZETA, MASS, [0.0, 1.0, 0.0]))
MODES1 = BenchModes([(FN, ZETA, MASS, (1.0, 0.0))])
MODES2 = BenchModes([(FN, ZETA, MASS, (1.0, 0.0)), (FN, ZETA, MASS, (0.0, 1.0))])


def _sd_converged(cut, frf, rpm, a_max, process_damping=False, points_per_period=48, n_scan=20):
    return SD.limit_at_speed(cut, frf, rpm * RPM, m=40, a_max=a_max, n_scan=n_scan, tol_rel=1e-3,
                             points_per_period=points_per_period, process_damping=process_damping)[0]


def _td(modes, ad, rpm, a_hi, steps=96, periods=150, **kw):
    phi = SL.immersion_angles(ad, True)
    return onset_depth(modes, a_lo=1e-6, a_hi=a_hi, tol_rel=2e-3, n_teeth=2, Kt_Pa=KT, Kr_Pa=KN, fz_m=1e-4,
                       phi_st_rad=phi[0], phi_ex_rad=phi[1], speed_rad_s=rpm * RPM, steps_per_tooth=steps,
                       periods=periods, **kw)


@pytest.mark.parametrize("rpm", [6000.0, 6642.0, 8358.0, 10000.0])
def test_one_dof_onset(rpm):
    """1 DOF, a/D = 0.5 down-milling (Insperger & Stepan 2004): the time-domain onset at 96 steps per tooth is
    within 2 % of the converged semi-discretisation, on lobe minima and flanks and where the first loss of stability
    is period doubling (10000 rpm: a flip lens below a Hopf boundary); the zero-order solution, which has no
    period-doubling boundary, within 15 %."""
    cut = SL.classic_milling(2, KT, KN, *SL.immersion_angles(0.5, True))
    a_sd = _sd_converged(cut, FRF1, rpm, 0.01)
    a_td = _td(MODES1, 0.5, rpm, 3.0 * a_sd)
    assert abs(a_td / a_sd - 1.0) < 0.02, (rpm, a_td, a_sd)
    a_z = SL.limit_at_speed(cut, FRF1, rpm * RPM, FRF1.frequency_grid(3000, 0.5, 2.0)).depth_limit_m
    mu = SD.multipliers(cut, FRF1, rpm * RPM, 1.01 * a_sd, 40)[0]
    if SD.bifurcation_kind(mu) == "hopf":
        assert abs(a_z / a_sd - 1.0) < 0.15, (rpm, a_z, a_sd)
    else:
        assert a_z > a_sd          # the zero-order solution has no period-doubling boundary: unconservative there


@pytest.mark.parametrize("rpm", [7700.0, 10500.0])
def test_two_dof_onset(rpm):
    """2 DOF symmetric tool, slotting (Insperger & Stepan 2004, Sect. 5): time domain within 3 % of
    semi-discretisation, zero order within 3 %."""
    cut = SL.classic_milling(2, KT, KN, *SL.immersion_angles(1.0, True))
    a_sd = _sd_converged(cut, FRF2, rpm, 0.002)
    a_td = _td(MODES2, 1.0, rpm, 3.0 * a_sd)
    assert abs(a_td / a_sd - 1.0) < 0.03, (rpm, a_td, a_sd)
    a_z = SL.limit_at_speed(cut, FRF2, rpm * RPM, FRF2.frequency_grid(3000, 0.5, 2.0)).depth_limit_m
    assert abs(a_z / a_sd - 1.0) < 0.03


def test_low_immersion_onset():
    """a/D = 0.05: the time domain at 192 steps per tooth (the short engagement arc) within 4 % of
    semi-discretisation in the regenerative trough at 8447 rpm."""
    cut = SL.classic_milling(2, KT, KN, *SL.immersion_angles(0.05, True))
    a_sd = _sd_converged(cut, FRF1, 8447.0, 0.02)
    a_td = _td(MODES1, 0.05, 8447.0, 3.0 * a_sd, steps=192)
    assert abs(a_td / a_sd - 1.0) < 0.04, (a_td, a_sd)


@pytest.mark.parametrize("rpm", [1500.0, 3000.0])
def test_process_damping_onset(rpm):
    """Low speed with flank indentation (K_sp 1e13 N/m^3, land 0.1 mm, R 10 mm): the time domain with the one-sided
    nonlinear indentation force (process_damping_element) and no forced vibration (f_z = 0) is within 4 % of the
    semi-discretisation with the energy-equivalent linear damping K_sp L^2 / (4 V) applied while each tooth cuts;
    the zero-order solution averages the damping over the revolution and overestimates the limit at 1500 rpm."""
    Ksp, land, R = 1e13, 1e-4, 0.01
    phi = SL.immersion_angles(0.5, True)
    cut = SL.classic_milling(2, KT, KN, *phi, K_sp_N_per_m3=Ksp, land_m=land, radius_m=R)
    a_sd = _sd_converged(cut, FRF1, rpm, 0.004, process_damping=True, points_per_period=24, n_scan=10)
    # small kick and a bracket near the limit: far above it the indentation of large vibrations (slope beyond the
    # clearance angle) saturates the growth within the run
    a_td = onset_depth(MODES1, a_lo=0.5 * a_sd, a_hi=1.5 * a_sd, tol_rel=3e-3, n_teeth=2, Kt_Pa=KT, Kr_Pa=KN,
                       fz_m=0.0, phi_st_rad=phi[0], phi_ex_rad=phi[1], speed_rad_s=rpm * RPM, steps_per_tooth=96,
                       periods=60, kick_m_s=1e-5, process_damping=(Ksp, land, math.tan(math.radians(7.0)), 5e-4, R))
    assert abs(a_td / a_sd - 1.0) < 0.04, (rpm, a_td, a_sd)
    bare = SL.classic_milling(2, KT, KN, *phi)
    assert a_sd > 1.3 * _sd_converged(bare, FRF1, rpm, 0.004, points_per_period=24, n_scan=10)
    if rpm < 2000.0:
        om = FRF1.frequency_grid(3000, 0.5, 2.0)
        a_z, _ = SL.limit_with_process_damping(cut, FRF1, rpm * RPM, om, tol_rel=1e-4, max_iter=60)
        assert a_z.depth_limit_m > 1.15 * a_sd


def test_chatter_detected_in_the_simulated_vibration():
    """Above the limit (with the edge leaving the cut) the spectral criterion flags chatter near the frequency of
    the critical Floquet multiplier; below it the vibration is forced only."""
    rpm = 8358.0
    cut = SL.classic_milling(2, KT, KN, *SL.immersion_angles(0.5, True))
    a_sd = _sd_converged(cut, FRF1, rpm, 0.01)
    phi = SL.immersion_angles(0.5, True)
    settings = ChatterSettings(0.5, 0.5e-6, 0.05, 1.5, 0.05)
    kw = dict(n_teeth=2, Kt_Pa=KT, Kr_Pa=KN, fz_m=1e-4, phi_st_rad=phi[0], phi_ex_rad=phi[1], speed_rad_s=rpm * RPM,
              steps_per_tooth=96, periods=160, linear=False)
    hot = run_benchmark(MODES1, depth_m=1.4 * a_sd, **kw)
    cold = run_benchmark(MODES1, depth_m=0.6 * a_sd, **kw)
    dt = hot.t_s[1] - hot.t_s[0]
    tick_s = 1e-7
    ticks = np.round(hot.t_s / tick_s).astype(np.int64)
    u = np.zeros((len(ticks), 3))
    f_s = rpm / 60.0
    late = slice(len(ticks) // 2, None)
    u[:, :2] = hot.u_m
    r_hot = analyse(ticks[late], u[late], tick_s, f_s, f_s, 2, 1e-4, settings)
    u[:, :2] = cold.u_m
    r_cold = analyse(ticks[late], u[late], tick_s, f_s, f_s, 2, 1e-4, settings)
    assert r_hot.chatter and not r_cold.chatter
    assert r_hot.index > 0.5 > r_cold.index
    mu = SD.multipliers(cut, FRF1, rpm * RPM, 1.02 * a_sd, 40)[0]
    # the chatter frequency of a Hopf multiplier mu: (arg mu + 2 pi k) / tau for the k nearest the natural frequency
    tau = 60.0 / (2.0 * rpm)
    base = abs(math.atan2(mu.imag, mu.real)) / (TWO_PI * tau)
    candidates = [abs(k / tau + s * base) for k in range(0, 8) for s in (-1.0, 1.0)]
    f_pred = min(candidates, key=lambda f: abs(f - r_hot.freq_Hz))
    assert abs(r_hot.freq_Hz / f_pred - 1.0) < 0.03
    assert r_hot.amp_m > 1e-6 and dt > 0.0
