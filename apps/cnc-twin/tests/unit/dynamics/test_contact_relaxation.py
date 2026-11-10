"""Process damping's flank contact in the predictor-corrector (model doc 2.3): the hot loop evaluates the wave-memory
indentation K_sp dS A at the predicted state, a stiffness K_sp dS L along each element's normal with the remembered
surface held. Beyond the explicit limit of the scheme (a mode the step does not resolve, contact stiffness above its
own) the plain predictor-corrector turns it into a growing step-to-step alternation (examples/drill-tap-plate-1045's
spot drill at 6.8 s, stable at half the step); with the contact in the relaxation's stiffness estimate the loop is
the implicit one and decays."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from cnctwin.dynamics import kernels as K
from cnctwin.dynamics.abi import core_accum
from cnctwin.dynamics.operating import cut_model
from tests.unit.dynamics.dynfix import system, tool_mode

TWO_PI = 2.0 * math.pi


def _contact_loop(omega_dt: float, ratio: float, relax: bool, n_steps: int = 200) -> np.ndarray:
    """Displacement history of one mode (1 kHz, zeta 0.01) under a contact force F = -K u evaluated at the predicted
    state (the real kernels), K = ratio x the modal stiffness."""
    freq = 1000.0
    w = TWO_PI * freq
    k = w * w
    Kc = ratio * k
    tick_s = omega_dt / w / 100.0
    sys, _ = system([tool_mode(freq, 0.01, 1.0, (1.0, 0.0, 0.0))], l0_ticks=100, ipo_ticks=100, tick_s=tick_s,
                    K_c=np.diag([-Kc, 0.0, 0.0]) if relax else None)
    sys.state.x[0, 0] = 1e-6
    sys.ds.f_cut[0] = -Kc * 1e-6
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
    return np.abs(np.array(q))


@pytest.mark.parametrize("omega_dt,ratio", [(0.3, 20.0), (0.3, 50.0), (1.0, 2.0)])
def test_a_stiff_contact_needs_the_estimate(omega_dt, ratio):
    """Closed-loop 2 pi f dt of 1.4-2.1: the plain scheme diverges (it holds to about 1.2), the relaxed one decays
    (it holds to about 2; beyond, the step must resolve the contact-stiffened frequency: model doc 2.3)."""
    plain = _contact_loop(omega_dt, ratio, relax=False, n_steps=300)
    relaxed = _contact_loop(omega_dt, ratio, relax=True, n_steps=300)
    assert plain[-20:].max() > 1e3 * plain[:20].max()          # the explicit evaluation diverges
    assert relaxed[-20:].max() <= 1.5 * relaxed[:20].max()     # the relaxed (implicit) loop does not grow


def _edge(n_elem: int = 4):
    nt = 2
    r = np.tile(np.linspace(0.5e-3, 2e-3, n_elem), (nt, 1))
    return SimpleNamespace(
        elem_r_m=r, hand_sign=np.array([-1.0]), elem_valid=np.ones((nt, n_elem), np.int8),
        elem_group=np.zeros((nt, n_elem), np.int32), tooth_phase_rad=np.array([0.0, math.pi]),
        elem_lag_rad=np.zeros((nt, n_elem)), elem_kappa_rad=np.full((nt, n_elem), math.radians(45.0)),
        elem_db_m=np.full((nt, n_elem), 0.3e-3), elem_len_m=np.full((nt, n_elem), 0.25e-3),
        elem_edge_radius_m=np.full((nt, n_elem), 6e-6), elem_incl_rad=np.zeros((nt, n_elem)))


def test_cut_model_carries_the_flank_contact_stiffness():
    """contact_M = -K_sp dS L n n^T / a_p per engaged element (L = VB + land_per_radius r_e); its revolution mean for
    two opposite lips of a 90 deg point: axial K n_z^2, lateral K n_r^2 / 2 per axis."""
    edge = _edge()
    nt, ne = edge.elem_r_m.shape
    grid = np.array([1e-6, 1e-4])
    coeffs = SimpleNamespace(h_grid_m=grid, Kt_Pa=np.full((1, 2), 2e9), Kr_Pa=np.full((1, 2), 1e9),
                             Ka_Pa=np.zeros((1, 2)), K_sp_N_per_m3=np.array([3.4e14]))
    VB = np.full((nt, ne), 20e-6)
    ap = 2e-3
    cut = cut_model(edge, None, coeffs, engaged=np.ones((nt, ne), bool), ap_m=ap, h_mean_m=4e-5, arc=(0.0, TWO_PI),
                    rotating=True, spin=-1.0, K_sp_scale=coeffs.K_sp_N_per_m3, land_per_radius=1.0, VB_m=VB,
                    work_radius_m=0.0)
    k_e = 3.4e14 * 0.25e-3 * (20e-6 + 6e-6)
    assert cut.contact_M.shape == (nt * ne, 3, 3)
    n = nt * ne
    mean = ap * cut.contact_mean()
    s2 = 0.5
    assert mean[2, 2] == pytest.approx(-n * k_e * s2, rel=1e-9)
    assert mean[0, 0] == pytest.approx(-n * k_e * s2 / 2.0, rel=1e-9)
    assert mean[1, 1] == pytest.approx(-n * k_e * s2 / 2.0, rel=1e-9)
    assert abs(mean[0, 1]) < 1e-9 * n * k_e
    # the lobes' delay model is not changed by it
    assert cut.mean()[2, 2] != pytest.approx(cut.contact_mean()[2, 2])


def test_the_stride_resolves_the_contact_stiffened_mode():
    """f_contact: a 2 kHz tool mode (1 kg modal mass, shape 1 at the contact) under a 3e10 N/m contact along x is
    sqrt(4 pi^2 2000^2 + 3e10) / 2 pi (27.6 kHz); the L0 rule keeps 2 pi f_contact dt <= the contact limit."""
    from cnctwin.dynamics.domain import l0_ticks_rule
    from cnctwin.dynamics.modal import f_contact_Hz
    mode = tool_mode(2000.0, 0.02, 1.0, (1.0, 0.0, 0.0))
    K = np.diag([-3e10, 0.0, 0.0])
    w = TWO_PI * 2000.0
    expected = math.sqrt(w * w + 3e10) / TWO_PI
    assert f_contact_Hz([mode], 0, K) == pytest.approx(expected, rel=1e-12)
    assert f_contact_Hz([mode], 0, None) == 0.0
    free = l0_ticks_rule(1000, 1e-6, 0.0, 1, 2000.0, 1.0, 1, 0.3)
    stiff = l0_ticks_rule(1000, 1e-6, 0.0, 1, 2000.0, 1.0, 1, 0.3, expected, 1.0)
    assert TWO_PI * expected * stiff * 1e-6 <= 1.0 < TWO_PI * expected * free * 1e-6


def test_a_slow_tool_keeps_its_contact_between_subtractions():
    """Removal subtracts a tooth's material when the tooth has passed (every 36 ms for the D10 spot drill at 830 rpm
    against 10 ms steps): while cutting reports the tool in cut the operating point keeps the last subtracting step's
    geometry (persisted in the state's meta), and drops it when the tool leaves the cut or changes."""
    from cnctwin.dynamics.domain import DynamicsDomain
    from cnctwin.dynamics.state import DynamicsState
    dom = object.__new__(DynamicsDomain)
    state = SimpleNamespace(dynamics=DynamicsState())

    def bus(subtracted: bool, in_cut_s: float, slot: int = 0):
        g = SimpleNamespace(in_cut=subtracted, point_S_m=np.array([0.0, 0.0, -1e-3]), ap_m=3e-3, ae_m=5e-3,
                            engaged_angle_rad=math.pi)
        return SimpleNamespace(removal=SimpleNamespace(contacts=[SimpleNamespace(tool_slot=slot, path=0, geometry=g)]),
                               cutting=SimpleNamespace(contacts=[SimpleNamespace(tool_slot=slot, in_cut_s=in_cut_s)]))

    assert dom._engaged_contact(state, bus(False, 0.01), 0, 0) is None            # never subtracted: nothing held
    first = dom._engaged_contact(state, bus(True, 0.01), 0, 0)
    assert first is not None and first.geometry.in_cut
    held = dom._engaged_contact(state, bus(False, 0.01), 0, 0)
    assert held is not None and held.geometry.ap_m == 3e-3
    np.testing.assert_array_equal(held.geometry.point_S_m, [0.0, 0.0, -1e-3])
    # a resumed run reads the same held geometry from the persisted meta
    arrays, meta = state.dynamics.to_arrays()
    resumed = SimpleNamespace(dynamics=DynamicsState.from_arrays(arrays, meta))
    again = dom._engaged_contact(resumed, bus(False, 0.01), 0, 0)
    np.testing.assert_array_equal(again.geometry.point_S_m, held.geometry.point_S_m)
    assert dom._engaged_contact(state, bus(False, 0.0), 0, 0) is None             # out of the cut: dropped
    assert dom._engaged_contact(state, bus(False, 0.01), 0, 0) is None
    dom._engaged_contact(state, bus(True, 0.01), 0, 0)
    assert dom._engaged_contact(state, bus(False, 0.01, slot=1), 0, 1) is None    # another tool: not held


def _residual_loop(gain: float, relax: bool, n_steps: int = 200, estimate: float = 1.0) -> np.ndarray:
    """Contact force history of a stiff, well damped mode (20 kHz, zeta 0.5) with a tool residual compliance C at the contact, the force
    F = -K (u - s) evaluated at the predicted deflection (modal + C x the stored force), K C = gain; the estimate is
    `estimate` x the true K."""
    C = 1e-8
    Kc = gain / C
    sys, _ = system([tool_mode(20000.0, 0.5, 1.0, (1.0, 0.0, 0.0))], l0_ticks=2, ipo_ticks=100, tick_s=1e-6,
                    K_c=np.diag([-estimate * Kc, 0.0, 0.0]) if relax else None, residual=np.eye(3) * C)
    acc = core_accum()
    station = np.zeros((1, 6))
    fS = np.zeros(3)
    s = -1e-6                                         # the surface 1 um into the tool: a steady contact force
    out = []
    for n in range(n_steps):
        K.external_force(sys, 0.0, 0.0, 0.0, np.zeros(7))
        K.modal_predict(sys)
        K.modal_deflection(sys)
        F = -Kc * (sys.defl.u_st_T0_m[0, 0] - s)
        station[0, 0] = F
        fS[0] = F
        K.modal_correct(sys, station, fS, (n + 1) * 2, acc)
        out.append(F)
    return np.array(out)


@pytest.mark.parametrize("gain", [0.5, 2.4, 5.0])
def test_the_residual_flexibility_is_relaxed_against_the_contact(gain):
    """The residual flexibility takes the last evaluated force: against a contact the explicit iteration's gain is
    K C, and at 2.2-2.4 (the spot drill of examples/drill-tap-plate-1045) the force alternates and grows. With the
    estimate the stored force is the fixed point: the force settles to the series value K s / (1 + K C + K / k_modal)."""
    plain = _residual_loop(gain, relax=False)
    relaxed = _residual_loop(gain, relax=True)
    if gain > 1.0:
        assert np.abs(plain[-5:]).max() > 1e3 * np.abs(plain[:5]).max()      # explicit: alternates and grows
    else:
        assert np.ptp(plain[-10:]) <= 1e-6 * np.abs(plain[-10:]).max()       # below 1 the explicit one settles
    tail = relaxed[-10:]
    assert np.ptp(tail) <= 1e-6 * np.abs(tail).max()                         # relaxed: settled
    Kc = gain / 1e-8
    k_mode = (TWO_PI * 20000.0) ** 2
    expected = -Kc * 1e-6 / (1.0 + gain + Kc / k_mode)      # pushed back by the surface 1 um into the tool
    assert tail[-1] == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("estimate", [0.5, 3.0])
def test_a_wrong_estimate_still_converges(estimate):
    """The fixed-point update converges for a true gain below 1 + 2 x the estimated one (model doc 2.3)."""
    relaxed = _residual_loop(2.4, relax=True, n_steps=200, estimate=estimate)
    tail = relaxed[-10:]
    assert np.ptp(tail) <= 1e-6 * np.abs(tail).max()


def test_the_residual_relaxation_takes_its_own_stiffness():
    """build_system relaxes the residuals with K_res_T0 when given (the domain passes the flank contact alone) and
    with K_c_T0 otherwise; the identity (flag off) without either."""
    from cnctwin.dynamics.abi import MI_RES_RELAX, MS_RES_RELAX
    from cnctwin.dynamics import modal as MD
    from tests.unit.dynamics.dynfix import empty_grid
    mode = tool_mode(2000.0, 0.02, 1.0, (1.0, 0.0, 0.0))
    C = np.eye(3) * 1e-8
    Kc = np.diag([-3e8, -3e8, 0.0])
    Kr = np.diag([-1e8, 0.0, 0.0])

    def build(**kw):
        sys, _ = MD.build_system([mode], nst=1, ref_station=0, residual=C, grid=empty_grid(), point_S=np.zeros(3),
                                 l0_ticks=2, ipo_ticks=100, tick_s=1e-6, n_ticks=100, capacity=100, rotating=True,
                                 unbalance_T=None, R_S_T0=np.eye(3), **kw)
        return int(sys.dm.iscal[MI_RES_RELAX]), np.asarray(sys.dm.scal[MS_RES_RELAX:MS_RES_RELAX + 9]).reshape(3, 3)

    flag, R = build(K_c_T0=Kc)
    assert flag == 1 and R[0, 0] == pytest.approx(1.0 / 4.0) and R[1, 1] == pytest.approx(1.0 / 4.0)
    flag, R = build(K_c_T0=Kc, K_res_T0=Kr)
    assert flag == 1 and R[0, 0] == pytest.approx(1.0 / 2.0) and R[1, 1] == pytest.approx(1.0)
    flag, R = build(K_c_T0=Kc, K_res_T0=None)
    assert flag == 0 and np.array_equal(R, np.eye(3))
    flag, R = build(K_c_T0=None)
    assert flag == 0
