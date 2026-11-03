"""Spectral chatter detection on synthetic vibration: forced lines only, regenerative chatter, period doubling of an
even-tooth cutter, a speed spread, onset time and gaps."""

from __future__ import annotations

import math

import numpy as np

from cnctwin.dynamics.chatter import ChatterSettings, analyse, last_run

SET = ChatterSettings(index_threshold=0.5, min_amplitude_m=0.5e-6, min_amplitude_per_feed=0.05, flip_growth_ratio=1.5,
                      flip_min_share=0.05)
TICK_S = 1e-6
TWO_PI = 2.0 * math.pi


def _signal(f_s, n_teeth, revs, dt_ticks=10, chatter=None, half=None, spread=0.0, seed=0):
    rng = np.random.default_rng(seed)
    T = revs / f_s
    n = int(T / (dt_ticks * TICK_S))
    ticks = np.arange(n, dtype=np.int64) * dt_ticks
    t = ticks * TICK_S
    phase = TWO_PI * f_s * (t + 0.5 * spread * t * t / T)        # a slow speed change of +spread over the window
    u = np.zeros((n, 3))
    u[:, 0] += 3e-6 * np.cos(phase + 0.3)                       # runout: spindle frequency
    for k in range(1, 6):
        u[:, 0] += 8e-6 / k * np.cos(k * n_teeth * phase + 0.7 * k)
        u[:, 1] += 5e-6 / k * np.sin(k * n_teeth * phase + 0.2 * k)
    u += 2e-9 * rng.normal(size=u.shape)
    if chatter is not None:
        f_c, a_end, growth = chatter
        env = a_end * np.exp(growth * (t - t[-1]))
        u[:, 0] += env * np.cos(TWO_PI * f_c * t)
        u[:, 1] += 0.5 * env * np.sin(TWO_PI * f_c * t)
    if half is not None:
        a_end, growth = half
        env = a_end * np.exp(growth * (t - t[-1]))
        u[:, 0] += env * np.cos(0.5 * n_teeth * phase)          # (1/2) tooth passing: an N/2 spindle order
    return ticks, u


def test_forced_only():
    ticks, u = _signal(150.0, 4, 25)
    r = analyse(ticks, u, TICK_S, 150.0, 150.0, 4, 1e-4, SET)
    assert not r.chatter and r.index < 0.05


def test_regenerative_chatter():
    f_c = 1234.5
    growth = 12.0
    ticks, u = _signal(150.0, 4, 25, chatter=(f_c, 40e-6, growth))
    r = analyse(ticks, u, TICK_S, 150.0, 150.0, 4, 1e-4, SET)
    assert r.chatter and r.index > 0.5
    assert abs(r.freq_Hz / f_c - 1.0) < 2e-3
    # amplitude: the Hann-weighted band amplitude of the growing envelope, close to its mean over the window
    t = ticks * TICK_S
    env = 40e-6 * np.exp(growth * (t - t[-1])) * math.sqrt(1.25)
    w = np.hanning(len(t))
    weighted = math.sqrt(np.sum((w * env) ** 2) / np.sum(w * w))
    assert 0.85 * weighted < r.amp_m < 1.15 * weighted
    assert abs(r.growth_per_s / growth - 1.0) < 0.25
    # onset: where the envelope reaches the floor 5 um (0.05 x 0.1 mm feed) - before the window here
    t_on = t[-1] + math.log(5e-6 / (40e-6 * math.sqrt(1.25))) / growth
    assert r.onset_tick * TICK_S == t[0] if t_on < t[0] else abs(r.onset_tick * TICK_S - t_on) < 0.1 * (t[-1] - t[0])


def test_weak_onset_not_yet_chatter():
    """A component still small against the forced vibration over the window does not dominate: no chatter yet (the
    sliding window flags it once it does, and the onset is then dated back by the envelope)."""
    ticks, u = _signal(150.0, 4, 25, chatter=(1234.5, 20e-6, 30.0))
    r = analyse(ticks, u, TICK_S, 150.0, 150.0, 4, 1e-4, SET)
    assert not r.chatter and r.index < 0.5 and abs(r.freq_Hz - 1234.5) < 2.0


def test_small_component_is_not_chatter():
    ticks, u = _signal(150.0, 4, 25, chatter=(1234.5, 0.3e-6, 0.0))
    r = analyse(ticks, u, TICK_S, 150.0, 150.0, 4, 1e-4, SET)
    assert not r.chatter


def test_period_doubling_of_an_even_tooth_cutter():
    """A growing component at half the tooth-passing frequency (a spindle order for N = 4) is chatter; the same
    component without growth is forced (runout)."""
    grow = _signal(150.0, 4, 30, half=(40e-6, 8.0))
    r = analyse(*grow, TICK_S, 150.0, 150.0, 4, 1e-4, SET)
    assert r.flip and r.chatter
    assert abs(r.freq_Hz - 300.0) < 3.0
    steady = _signal(150.0, 4, 30, half=(4e-6, 0.0))
    r = analyse(*steady, TICK_S, 150.0, 150.0, 4, 1e-4, SET)
    assert not r.flip and not r.chatter


def test_odd_teeth_half_lines_are_not_forced():
    """For N = 3, (k + 1/2) tooth passing is not a spindle order: period doubling needs no growth test."""
    ticks, u = _signal(150.0, 3, 30)
    t = ticks * TICK_S
    u[:, 0] += 12e-6 * np.cos(TWO_PI * 225.0 * t)
    r = analyse(ticks, u, TICK_S, 150.0, 150.0, 3, 1e-4, SET)
    assert r.chatter and not r.flip and abs(r.freq_Hz - 225.0) < 2.0


def test_speed_spread_keeps_forced_lines_forced():
    ticks, u = _signal(150.0, 4, 25, spread=0.01)
    r = analyse(ticks, u, TICK_S, 150.0, 151.5, 4, 1e-4, SET)
    assert not r.chatter and r.index < 0.2


def test_last_contiguous_run():
    t = np.concatenate([np.arange(100) * 10, 5000 + np.arange(50) * 10]).astype(np.int64)
    s = last_run(t, 1.5)
    assert s.start == 100 and s.stop == 150
