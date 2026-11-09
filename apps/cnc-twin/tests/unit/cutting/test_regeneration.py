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


# ------------------------------------------------------------------------------- the wave memory (2026-11-09)
PD_N = A.PD_RING_N


def _ring(history, dx, travel, n=PD_N):
    """A one-element wave memory holding u(x_g) at travel g dx (g = 0 .. len(history) - 1) and the last evaluation
    at `travel` with u(travel); history(x) -> u (3,) at travel x."""
    cnt = int(math.floor(travel / dx)) + 1
    ring = np.zeros((1, n, 3))
    for g in range(max(cnt - n, 0), cnt):
        ring[0, g % n] = history(g * dx)
    aux = np.zeros((1, A.N_PDA))
    aux[0, A.PDA_X] = travel
    aux[0, A.PDA_UX:A.PDA_UZ + 1] = history(travel)
    aux[0, A.PDA_COUNT] = cnt
    return ring, aux


def test_wave_area_against_quadrature_of_the_generated_surface():
    """The flank's interference with the surface the edge left (wave memory: u at the samples, linear between them)
    equals a fine quadrature of max(0, (u(now) - u(x)) . n - max(0, x - L) tan(gamma)) over the profile, for random
    smooth and rough histories, lands and clearances; the compiled kernel equals its py_func and the Python
    reference."""
    from cnctwin.cutting import kernels as K
    from cnctwin.cutting import reference as REF
    rng = np.random.default_rng(7)
    lmax = 5e-4
    dx = lmax / (PD_N - 1)
    for trial in range(120):
        amp = rng.uniform(1e-9, 3e-5)
        lam = rng.uniform(2e-5, 3e-3)
        ph = rng.uniform(0, 2 * math.pi)
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        noise = rng.normal(0.0, 0.2 * amp, (64, 3))
        def history(x, amp=amp, lam=lam, ph=ph, noise=noise):
            k = int(x / dx) % 64
            return amp * np.array([math.sin(2 * math.pi * x / lam + ph), math.cos(2 * math.pi * x / lam), 0.3]) + \
                (noise[k] if trial % 2 else 0.0)
        travel_prev = rng.uniform(0.0, 3 * lmax)
        step = rng.uniform(0.0, 2 * dx)
        ring, aux = _ring(history, dx, travel_prev)
        u_now = history(travel_prev) + rng.normal(0.0, amp, 3)
        travel = travel_prev + step
        L = rng.uniform(0.0, 3e-4)
        tg = math.tan(rng.uniform(0.02, 0.3))
        xmax = rng.uniform(0.3, 1.0) * lmax
        got = K.wave_area(ring, aux, 0, *u_now, *n, travel, L, tg, xmax, dx)
        assert got == K.wave_area.py_func(ring, aux, 0, *u_now, *n, travel, L, tg, xmax, dx)
        ref = REF.wave_area_ref(ring, aux, 0, u_now, n, travel, L, tg, xmax, dx)
        # nodes: x = 0 (now), the last evaluation, the samples; u linear between them
        cnt = int(aux[0, A.PDA_COUNT])
        xs = [0.0, travel - travel_prev] + [travel - g * dx for g in range(cnt - 1, max(cnt - PD_N, 0) - 1, -1)]
        us = [u_now, history(travel_prev)] + [ring[0, g % PD_N] for g in range(cnt - 1, max(cnt - PD_N, 0) - 1, -1)]
        x = np.linspace(0.0, min(xmax, xs[-1]), 200001)
        du = np.interp(x, xs, [float((u_now - v) @ n) for v in us])
        delta = du - np.maximum(x - min(L, xmax), 0.0) * tg
        exp = float(np.trapezoid(np.maximum(delta, 0.0), x))
        assert got == pytest.approx(exp, rel=2e-4, abs=1e-15)
        assert ref == pytest.approx(got, rel=1e-9, abs=1e-18)


def test_a_long_wave_gives_the_linearised_process_damping():
    """A wave much longer than the flank: the generated surface is the straight line of slope v_n / V behind the
    edge, and the wave memory gives exactly the linearised model's area (flank_interference_area)."""
    from cnctwin.cutting import kernels as K
    lmax = 5e-4
    dx = lmax / (PD_N - 1)
    n = np.array([1.0, 0.0, 0.0])
    for slope in (0.01, 0.05, 0.11, 0.3):
        hist = lambda x, slope=slope: np.array([slope * x, 0.0, 0.0])      # u(now) - u(then) = slope x
        travel = 2.0 * lmax
        ring, aux = _ring(hist, dx, travel)
        for L, tg in ((0.0, 0.12), (0.1e-3, 0.12), (0.2e-3, 0.05), (0.6e-3, 0.2)):
            got = K.wave_area(ring, aux, 0, *hist(travel), *n, travel, L, tg, lmax, dx)
            assert got == pytest.approx(K.flank_interference_area(slope, L, tg, lmax), rel=1e-9, abs=1e-18)


def test_a_wave_shorter_than_the_flank_indents_by_its_amplitude_only():
    """The cause of the turning blow-up and the drill entry (docs/models/loop.md 15): a vibration of nanometres at a
    frequency whose wavelength is far below the flank length (the work's 32-48 kHz modes at 3 m/s: 60-100 um, an L0
    step's alternation: 2 V dt) was read by the linearised model as a slope over the whole 0.5 mm flank (and near a
    drill's centre, where V -> 0, as a slope above the clearance for any vibration). The generated surface stands at
    most twice the amplitude above the land, over the land and the few microns behind it where the clearance has not
    yet risen above it."""
    from cnctwin.cutting import kernels as K
    lmax = 5e-4
    dx = lmax / (PD_N - 1)
    n = np.array([0.0, 0.0, 1.0])
    land, tg = 30e-6, math.tan(math.radians(6.0))
    for amp, lam in ((1e-9, 8e-5), (1e-9, 2 * 3.0 * 5e-6), (2e-6, 6e-5), (1e-8, 1e-5)):
        hist = lambda x, amp=amp, lam=lam: np.array([0.0, 0.0, amp * math.sin(2 * math.pi * x / lam)])
        travel = 3 * lmax + 0.37 * lam
        ring, aux = _ring(hist, dx, travel)
        for now in (hist(travel), np.array([0.0, 0.0, amp])):
            area = K.wave_area(ring, aux, 0, *now, *n, travel, land, tg, lmax, dx)
            bound = 2 * amp * (land + 2 * amp / tg + dx)
            assert 0.0 <= area <= bound
        # the linearised model at the wave's peak velocity: the slope 2 pi amp / lam over the flank
        old = K.flank_interference_area(2 * math.pi * amp / lam, land, tg, lmax)
        if 2 * math.pi * amp / lam > tg:
            assert old > 20 * bound
    # near a drill's centre (V -> 0) the travel since the entry is tiny: the interference is bounded by it
    hist = lambda x: np.array([0.0, 0.0, 1e-9 * math.sin(x / 1e-7)])
    ring, aux = _ring(hist, dx, 2e-7)
    assert K.wave_area(ring, aux, 0, 0.0, 0.0, 1e-9, *n, 2.5e-7, land, tg, lmax, dx) <= 2e-9 * 2.5e-7


def test_process_damping_does_not_relay_on_a_step_to_step_alternation():
    """A tool whose deflection alternates from one L0 step to the next (the predictor's period-2 noise) by a
    nanometre, with a worn land: the old model's force switched between nothing and the whole flank every step; the
    wave memory gives the land's indentation by the alternation's amplitude, a force of millinewtons."""
    R, ap, c = 5e-3, 2e-3, 0.05e-3
    tool = AN.cylinder_mill(R, 2, ap, 2, clear_rad=math.radians(7))
    nt, ne = tool.edge.elem_r_m.shape
    coeffs = {"default": dict(ALTINTAS, K_sp=3e13, sigma0=1.5e9, tau0=0.6e9, h_min_m=0.0)}
    ct, table, _ = linear_layout(tool, coeffs, OMEGA, VB=np.full((nt, ne), 0.1e-3))
    drv = Driver(ct, table, nt, ne, memory_bins=NB)
    vf = c * 2 * abs(OMEGA) / (2 * math.pi)
    tip = lambda t: np.array([vf * t, 0.0, 0.0])
    amp = 1e-9
    k = [0]

    def alternate(d, t):
        k[0] += 1
        sign = 1.0 if k[0] % 2 else -1.0
        d.defl.u_st_T0_m[:, 1] = sign * amp
        d.defl.v_st_T0_m_s[:, 1] = sign * 1.0     # the predictor's velocity spike (1 m/s), inconsistent with u

    lo, hi = straight_immersion(0.0, math.pi)
    drv.set_interval(lo, hi)
    drv.prefill_memory(tip, 0.0, OMEGA)
    drv.run_rotating(NB, OMEGA, DT, tip, deflection_fn=alternate, record=False)
    pd = abs(float(drv.acc.scalars[A.ACC_PROCESS_DAMPING_J]))
    shaft = float(drv.acc.scalars[A.ACC_SHAFT_WORK_J])
    assert shaft > 0.0
    # per element the land indents by at most 2 amp over the land and 2 amp / tan(clearance) behind it; its work on
    # the 1 m/s spikes stays far below the cutting work (the linearised model read a slope of 0.18 > tan(7 deg) and
    # indented the whole 0.5 mm flank: ~0.7 kN per element, more than the cutting work)
    assert pd < 1e-4 * shaft
