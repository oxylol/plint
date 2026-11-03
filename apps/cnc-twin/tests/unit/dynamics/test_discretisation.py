"""Exact first-order-hold discretisation, its energy integrals, the free interval with harmonic forcing, and the
forced responses the kernels produce (analytic references)."""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp

from cnctwin.dynamics import discretize as DZ
from cnctwin.dynamics import kernels as K
from cnctwin.dynamics.abi import DA_DAMPING_J, DA_W_BASE_J, DA_W_CUT_J, DA_W_UNB_J, KIND_MACHINE, core_accum
from tests.unit.dynamics.dynfix import system, tool_mode

TWO_PI = 2.0 * math.pi


def _analytic_ramp(w, z, x0, f0, f1, h, t):
    """Exact response to f(t) = f0 + (f1 - f0) t / h from x0 (homogeneous + polynomial particular solution)."""
    s = (f1 - f0) / h
    wd = w * math.sqrt(1 - z * z)
    sig = z * w
    qp0 = f0 / w ** 2 - 2 * z * s / w ** 3
    a = x0[0] - qp0
    b = x0[1] - s / w ** 2
    B = (b + sig * a) / wd
    e = math.exp(-sig * t)
    q = e * (a * math.cos(wd * t) + B * math.sin(wd * t)) + (f0 + s * t) / w ** 2 - 2 * z * s / w ** 3
    v = e * ((-sig * a + wd * B) * math.cos(wd * t) + (-sig * B - wd * a) * math.sin(wd * t)) + s / w ** 2
    return np.array([q, v])


@pytest.mark.parametrize("wh", [1e-4, 0.01, 0.3, 3.0, 30.0])
def test_foh_is_exact_for_a_ramp(wh):
    w, z = TWO_PI * 800.0, 0.03
    h = wh / w
    Phi, G0, G1 = DZ.foh(w, z, h)
    x0 = np.array([2e-6, -1e-3])
    x1 = Phi @ x0 + G0 * 3.0 + G1 * -1.5
    np.testing.assert_allclose(x1, _analytic_ramp(w, z, x0, 3.0, -1.5, h, h), rtol=1e-10, atol=1e-18)


def test_foh_converges_on_a_harmonic_force():
    """Sampled harmonic forcing: second order in the step (the hold is the only approximation)."""
    w, z, Om = TWO_PI * 500.0, 0.02, TWO_PI * 650.0
    T = 0.05
    exact = solve_ivp(lambda t, y: [y[1], -w * w * y[0] - 2 * z * w * y[1] + math.sin(Om * t)], (0, T), [0.0, 0.0],
                      rtol=1e-12, atol=1e-16, method="DOP853").y[:, -1]
    errs = []
    for n in (500, 1000, 2000):
        h = T / n
        Phi, G0, G1 = DZ.foh(w, z, h)
        x = np.zeros(2)
        for k in range(n):
            x = Phi @ x + G0 * math.sin(Om * k * h) + G1 * math.sin(Om * (k + 1) * h)
        errs.append(np.linalg.norm(x - exact) / np.linalg.norm(exact))
    assert errs[0] / errs[1] > 3.5 and errs[1] / errs[2] > 3.5        # O(h^2)
    # the linear hold of a sinusoid errs by about (W h)^2 / 12 of its amplitude
    assert errs[-1] < 2.0 * (Om * T / 2000) ** 2 / 12.0


def test_energy_grams_against_quadrature():
    w, z, h = TWO_PI * 1200.0, 0.04, 2e-5
    MD, N = DZ.energy_grams(w, z, h)
    x0 = np.array([1e-6, 3e-3])
    f0, f1 = 5.0, -2.0
    s = (f1 - f0) / h
    zz = np.array([x0[0], x0[1], f0, s])
    v = lambda t: _analytic_ramp(w, z, x0, f0, f1, h, t)[1]
    D = quad(lambda t: 2 * z * w * v(t) ** 2, 0, h, epsabs=0, epsrel=1e-13)[0]
    W = quad(lambda t: (f0 + s * t) * v(t), 0, h, epsabs=0, epsrel=1e-13)[0]
    np.testing.assert_allclose(zz @ MD @ zz, D, rtol=1e-9)
    np.testing.assert_allclose(np.array([f0, s]) @ N @ zz, W, rtol=1e-9)
    x1 = _analytic_ramp(w, z, x0, f0, f1, h, h)
    E = lambda x: 0.5 * (x[1] ** 2 + w * w * x[0] ** 2)
    assert abs(E(x1) - E(x0) - (W - D)) < 1e-10 * (abs(W) + abs(D) + E(x0))


def _unbalanced_system(freq, zeta, mass, l0, ipo, tick_s, U, C=None, rot=None):
    m = tool_mode(freq, zeta, mass, (1.0, 0.0, 0.0), rot=rot)
    sys, _ = system([m], l0_ticks=l0, ipo_ticks=ipo, tick_s=tick_s,
                    unbalance=(np.array(U, float), np.zeros(3) if C is None else np.array(C, float)))
    return sys


def _receptance(w, z, Om):
    return 1.0 / complex(w * w - Om * Om, 2 * z * w * Om)


@pytest.mark.parametrize("Om_over_w", [0.3, 0.97, 1.6])
def test_unbalance_response_out_of_contact(Om_over_w):
    """Static unbalance U at constant speed: steady-state x(t) = Re(|phi|^2 U Om^2 H(Om) e^(i theta)) - amplitude and
    phase against the receptance (exact propagation of the free ticks)."""
    freq, zeta, mass = 900.0, 0.05, 2.0
    w = TWO_PI * freq
    Om = -Om_over_w * w                       # M03: negative
    tick_s, ipo = 1e-6, 1000
    U = (3e-6, 0.0, 0.0)
    sys = _unbalanced_system(freq, zeta, mass, 10, ipo, tick_s, U)
    theta = 0.0
    n = 400
    xs = []
    for k in range(n):
        th1 = theta + Om * ipo * tick_s
        K.modal_free(sys, theta, th1, Om, Om, np.zeros(7), np.zeros(7), (k + 1) * ipo)
        theta = th1
        xs.append(sys.state.x[0, 0] / math.sqrt(mass))
    # analytic steady state at the last tick: u = Re(phi_c conj ... ) along x
    P = (1.0 / math.sqrt(mass)) * complex(U[0], U[1])          # conj(phi_c) U_c with phi real along x
    q = (P * Om * Om * _receptance(w, zeta, Om) * cmath.exp(1j * theta)).real
    np.testing.assert_allclose(xs[-1], q / math.sqrt(mass), rtol=1e-6, atol=1e-6 * abs(q / math.sqrt(mass)))
    amp = abs(Om * Om * _receptance(w, zeta, Om)) * U[0] / mass
    assert abs(max(abs(v) for v in xs[-60:]) / amp - 1.0) < 0.02


def test_unbalance_response_in_contact_steps():
    """The same steady state reached by L0 steps (first-order hold on the unbalance samples): error O((W dt)^2)."""
    freq, zeta, mass = 700.0, 0.06, 1.5
    w = TWO_PI * freq
    Om = -1.2 * w
    tick_s, l0 = 1e-6, 8
    U = (0.0, 2e-6, 0.0)
    sys = _unbalanced_system(freq, zeta, mass, l0, 1000, tick_s, U)
    acc = core_accum()
    theta = 0.0
    station = np.zeros((1, 6))
    for k in range(4000):
        theta += Om * l0 * tick_s
        K.external_force(sys, theta, Om, 0.0, np.zeros(7))
        K.modal_predict(sys)
        K.modal_correct(sys, station, np.zeros(3), (k + 1) * l0, acc)
    P = (1.0 / math.sqrt(mass)) * complex(U[0], U[1])
    q = (P * Om * Om * _receptance(w, zeta, Om) * cmath.exp(1j * theta)).real
    amp = abs(P * Om * Om * _receptance(w, zeta, Om))
    assert abs(sys.state.x[0, 0] - q) < 2e-4 * amp


def test_couple_unbalance_through_rotations():
    """A couple C (kg m^2) loads a mode through its rotation shape: P = i conj(psi_c) C_c."""
    freq, zeta, mass = 1100.0, 0.04, 1.0
    w = TWO_PI * freq
    Om = -0.8 * w
    rot = (0.0, 2.0, 0.0)                         # r_y per unit modal coordinate (rad / sqrt(kg))
    C = (0.0, 1e-7, 0.0)
    sys = _unbalanced_system(freq, zeta, mass, 10, 1000, 1e-6, (0.0, 0.0, 0.0), C=C, rot=rot)
    P_expected = 1j * complex(rot[0], -rot[1]) * complex(C[0], C[1])
    np.testing.assert_allclose(sys.dm.unb[0], [P_expected.real, P_expected.imag], rtol=1e-12)
    theta = 0.0
    for k in range(500):
        th1 = theta + Om * 1e-3
        K.modal_free(sys, theta, th1, Om, Om, np.zeros(7), np.zeros(7), (k + 1) * 1000)
        theta = th1
    q = (P_expected * Om * Om * _receptance(w, zeta, Om) * cmath.exp(1j * theta)).real
    np.testing.assert_allclose(sys.state.x[0, 0], q, rtol=1e-6, atol=1e-9 * abs(q))


def test_unbalance_during_a_ramp():
    """Spindle ramp: tick-wise mean speed and alpha against a fine ODE integration of the exact forcing
    U (W(t)^2 - i alpha) e^(i theta(t)) (modal force along x); the tick model holds W^2 - i alpha per tick."""
    freq, zeta, mass = 400.0, 0.05, 1.0
    w = TWO_PI * freq
    alpha = -9000.0                                 # rad/s^2 (M03 ramp: omega more negative)
    U = 5e-6
    sys = _unbalanced_system(freq, zeta, mass, 10, 1000, 1e-6, (U, 0.0, 0.0))
    h = 1e-3
    n = 300
    om0 = -100.0
    q = []
    for k in range(n):
        t0, t1 = k * h, (k + 1) * h
        th0 = om0 * t0 + 0.5 * alpha * t0 * t0
        th1 = om0 * t1 + 0.5 * alpha * t1 * t1
        K.modal_free(sys, th0, th1, om0 + alpha * t0, om0 + alpha * t1, np.zeros(7), np.zeros(7), (k + 1) * 1000)
        q.append(sys.state.x[0, 0])

    def rhs(t, y):
        om = om0 + alpha * t
        th = om0 * t + 0.5 * alpha * t * t
        f = ((U / math.sqrt(mass)) * (om * om - 1j * alpha) * cmath.exp(1j * th)).real
        return [y[1], -w * w * y[0] - 2 * zeta * w * y[1] + f]
    ts = (np.arange(n) + 1) * h
    ref = solve_ivp(rhs, (0, n * h), [0.0, 0.0], t_eval=ts, rtol=1e-11, atol=1e-18, method="DOP853",
                    max_step=2e-5).y[0]
    err = np.max(np.abs(np.array(q) - ref)) / np.max(np.abs(ref))
    assert err < 5e-3


def test_base_excitation_quasi_static():
    """Constant relative axis acceleration a: the machine mode settles at u = -m a / k along its direction."""
    freq, zeta, mass = 50.0, 0.05, 500.0
    k = mass * (TWO_PI * freq) ** 2
    d = np.array([1.0, 0.0, 0.0])
    m = tool_mode(freq, zeta, mass, d, kind=KIND_MACHINE, base=[-math.sqrt(mass), 0, 0, 0, 0, 0, 0])
    sys, _ = system([m], l0_ticks=10, ipo_ticks=1000, tick_s=1e-6)
    a = np.array([3.0, 0, 0, 0, 0, 0, 0])
    for i in range(3000):
        K.modal_free(sys, 0.0, 0.0, 0.0, 0.0, a, a, (i + 1) * 1000)
    u = sys.state.x[0, 0] / math.sqrt(mass)
    np.testing.assert_allclose(u, -mass * 3.0 / k, rtol=1e-6)


def test_energy_identity_over_many_steps():
    """Both kernels close dE = W_cut + W_unbalance + W_base - D to round-off over a long mixed run."""
    m1 = tool_mode(900.0, 0.03, 2.0, (1.0, 0.2, 0.0))
    m2 = tool_mode(60.0, 0.05, 300.0, (0.0, 1.0, 0.0), kind=KIND_MACHINE, base=[0, -math.sqrt(300.0), 0, 0, 0, 0, 0])
    sys, _ = system([m1, m2], l0_ticks=20, ipo_ticks=1000, tick_s=1e-6, unbalance=(np.array([2e-6, 1e-6, 0.0]), np.zeros(3)))
    rng = np.random.default_rng(3)
    acc = core_accum()
    w = np.array([m1.omega_rad_s, m2.omega_rad_s])
    E0 = DZ.modal_energy(sys.state.x, w)
    tick, theta, Om = 0, 0.0, -800.0
    for block in range(30):
        if block % 2:
            for _ in range(50):                     # one L1 tick of L0 steps in contact
                tick += 20
                theta += Om * 20e-6
                K.external_force(sys, theta, Om, 0.0, rng.normal(size=7) * 0.5)
                K.modal_predict(sys)
                K.modal_correct(sys, rng.normal(size=(1, 6)) * 30.0, rng.normal(size=3) * 30.0, tick, acc)
        else:
            th1 = theta + Om * 1e-3
            b0, b1 = rng.normal(size=7) * 0.5, rng.normal(size=7) * 0.5
            K.modal_free(sys, theta, th1, Om, Om, b0, b1, tick + 1000)
            tick += 1000
            theta = th1
    E1 = DZ.modal_energy(sys.state.x, w)
    sc = sys.da.scal
    W = sc[DA_W_CUT_J] + sc[DA_W_UNB_J] + sc[DA_W_BASE_J] - sc[DA_DAMPING_J]
    scale = abs(E0) + abs(E1) + abs(sc[DA_W_CUT_J]) + abs(sc[DA_W_UNB_J]) + abs(sc[DA_W_BASE_J]) + abs(sc[DA_DAMPING_J])
    assert abs((E1 - E0) - W) <= 1e-11 * scale
