"""The predictor-corrector does not delay the cutting stiffness: with a force F = -K_c u evaluated at the predicted
state, the numerical damping of the closed loop stays within 1 % of the exact damping up to 2 pi f dt = 0.3 and
K_c / k up to 1 (architecture 5.4; model doc section 2.3)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cnctwin.dynamics import discretize as DZ
from cnctwin.dynamics import kernels as K
from cnctwin.dynamics.abi import core_accum
from tests.unit.dynamics.dynfix import system, tool_mode

TWO_PI = 2.0 * math.pi


def _closed_loop_damping(omega_dt: float, ratio: float, zeta: float, relax: bool, n_steps: int = 600) -> tuple:
    """(numerical, exact) damping ratio and frequency of the kernels' closed loop F = -K_c u."""
    freq = 1000.0
    w = TWO_PI * freq
    mass = 1.0
    k = mass * w * w
    Kc = ratio * k
    tick_s = omega_dt / w / 100.0              # 100 base ticks per step (any value: dt = 100 tick_s)
    K_est = np.diag([-Kc, 0.0, 0.0]) if relax else None
    sys, _ = system([tool_mode(freq, zeta, mass, (1.0, 0.0, 0.0))], l0_ticks=100, ipo_ticks=100, tick_s=tick_s,
                    K_c=K_est)
    sys.state.x[0, 0] = 1e-6 * math.sqrt(mass)
    # the force at the initial state is part of the history: F_0 = -K_c u_0
    sys.ds.f_cut[0] = -Kc * 1e-6 / math.sqrt(mass)
    acc = core_accum()
    station = np.zeros((1, 6))
    fS = np.zeros(3)
    q = []
    for n in range(n_steps):
        K.external_force(sys, 0.0, 0.0, 0.0, np.zeros(7))
        K.modal_predict(sys)
        K.modal_deflection(sys)
        F = -Kc * sys.defl.u_st_T0_m[0, 0]
        station[0, 0] = F
        fS[0] = F
        K.modal_correct(sys, station, fS, (n + 1) * 100, acc)
        q.append(sys.state.x[0, 0])
    q = np.array(q)
    a = q[12:-2]
    b = q[13:-1]
    c = q[14:]
    A = np.stack([b, a], axis=1)
    c1, c0 = np.linalg.lstsq(A, c, rcond=None)[0]
    lam = np.roots([1.0, -c1, -c0])
    dt = 100 * tick_s
    s = np.log(lam[np.argmax(lam.imag)].astype(complex)) / dt
    zeta_num = -s.real / abs(s)
    w_c = w * math.sqrt(1.0 + ratio)
    zeta_exact = zeta * w / w_c
    return zeta_num, zeta_exact, abs(s) / w_c - 1.0


@pytest.mark.parametrize("zeta", [0.002, 0.01, 0.05])
@pytest.mark.parametrize("omega_dt", [0.05, 0.15, 0.3])
@pytest.mark.parametrize("ratio", [0.1, 0.5, 1.0])
def test_damping_with_stiffness_estimate(zeta, omega_dt, ratio):
    zn, ze, ferr = _closed_loop_damping(omega_dt, ratio, zeta, relax=True)
    assert abs(zn / ze - 1.0) <= 0.01, (zn, ze)
    assert abs(ferr) <= 0.005


@pytest.mark.parametrize("zeta,omega_dt,ratio", [(z, w, r) for z in (0.005, 0.01, 0.05)
                                                  for w, r in ((0.1, 1.0), (0.2, 1.0), (0.3, 0.5), (0.3, 0.25))
                                                  if not (z < 0.01 and w > 0.2)])
def test_damping_without_estimate(zeta, omega_dt, ratio):
    """Without a stiffness estimate (before the first operating point is known) the plain predictor-corrector
    (extrapolated force, one evaluation) stays within 1 % up to 2 pi f dt = 0.2 for K_c / k up to 1, and at 0.3 for
    K_c / k up to 0.5 (zeta >= 0.01); the corner 0.3 / 1.0 reaches +3.3 % at zeta 0.01 (model doc 2.3)."""
    zn, ze, _ = _closed_loop_damping(omega_dt, ratio, zeta, relax=False)
    assert abs(zn / ze - 1.0) <= 0.01, (zn, ze)


def test_delayed_stiffness_would_be_unstable():
    """Reference: holding the force of the previous node (no predictor-corrector) delays the cutting stiffness and
    produces negative numerical damping far beyond the physical damping (the reason for the scheme)."""
    w, zeta, dt = 1.0, 0.01, 0.3
    Kc = 1.0
    Phi, G0, G1 = DZ.foh(w, zeta, dt)
    C = np.array([[1.0, 0.0]])
    M = Phi - Kc * np.outer(G0 + G1, C[0])
    lam = np.linalg.eigvals(M)
    s = np.log(lam[np.argmax(lam.imag)].astype(complex)) / dt
    zeta_num = -s.real / abs(s)
    zeta_exact = zeta / math.sqrt(1.0 + Kc)
    assert zeta_num < -2.0 * zeta_exact          # unstable where the exact loop is damped
