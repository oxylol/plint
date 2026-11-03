"""Chatter detection from the vibration samples (docs/models/dynamics.md section 4; L2).

The relative tool-workpiece deflection at the reference station, sampled at every L0 node in contact, is analysed
over a sliding window of whole spindle revolutions:

* the window is the last contiguous run of samples (a gap longer than ``gap_steps`` steps starts a new run), its
  mean and linear trend removed per component (the static deflection follows the engagement);
* a Hann-windowed periodogram summed over the three components; a stable cut, with any runout, pitch or helix,
  is periodic with the spindle revolution, so its vibration lies on the **forced lines** k f_s (k >= 1, every
  tooth-passing harmonic included; Delio, Tlusty & Smith 1992; Schmitz, Medicus & Dutterer 2002). Each line
  masks +-(2 / T_window + k (f_s,max - f_s,min) / 2) (the Hann main lobe and the speed spread of the window);
* the **chatter index** is the non-forced share of the energy above f_s / 2 (slower drift is excluded):
  ``CI = E_non / (E_non + E_forced)``; the chatter frequency the largest non-forced peak (parabolic interpolation
  on the log power), its amplitude ``A = 2 sqrt(sum_band |X|^2 / (n_fft sum w^2))`` over +-the line width;
* **period doubling** of an even-tooth cutter lies on the lines (k + 1/2) f_T, which are spindle orders: they
  move to the non-forced side when their energy grew by ``flip_growth_ratio`` from the first to the second half of
  the window and carries at least ``flip_min_share`` of the second half's energy;
* **onset**: the band around the chatter frequency is filtered out of the window (FFT mask) and its Hilbert
  envelope taken; the onset is the first sample where the envelope reaches the amplitude floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert


@dataclass
class ChatterSettings:
    index_threshold: float          # dynamics-chatter chatter_index_threshold
    min_amplitude_m: float
    min_amplitude_per_feed: float
    flip_growth_ratio: float
    flip_min_share: float
    gap_steps: float = 1.5          # a sample gap longer than this many median steps ends a run


@dataclass
class ChatterResult:
    index: float = 0.0
    freq_Hz: float = 0.0
    amp_m: float = 0.0
    forced_energy: float = 0.0      # periodogram sums (m^2 scale, for the ratio)
    nonforced_energy: float = 0.0
    onset_tick: int = -1
    flip: bool = False
    growth_per_s: float = 0.0
    duration_s: float = 0.0
    n_samples: int = 0
    chatter: bool = False
    threshold_m: float = 0.0

    def to_dict(self) -> dict:
        return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in self.__dict__.items()}


def last_run(ticks: np.ndarray, gap_steps: float) -> slice:
    """Slice of the last contiguous run of samples (no gap above gap_steps x the median step)."""
    n = len(ticks)
    if n < 2:
        return slice(0, n)
    d = np.diff(ticks.astype(np.int64))
    med = float(np.median(d))
    if med <= 0.0:
        return slice(n - 1, n)
    breaks = np.nonzero(d > gap_steps * med)[0]
    start = int(breaks[-1]) + 1 if len(breaks) else 0
    return slice(start, n)


def _detrend(u: np.ndarray) -> np.ndarray:
    n = u.shape[0]
    t = np.arange(n, dtype=float)
    out = np.empty_like(u)
    for c in range(u.shape[1]):
        coef = np.polyfit(t, u[:, c], 1)
        out[:, c] = u[:, c] - (coef[0] * t + coef[1])
    return out


def _spectrum(x: np.ndarray, nfft: int) -> tuple[np.ndarray, float]:
    """One-sided |X_k|^2 summed over components of the Hann-windowed signal, and sum w^2."""
    n = x.shape[0]
    w = np.hanning(n)
    X = np.fft.rfft(x * w[:, None], n=nfft, axis=0)
    return np.sum(np.abs(X) ** 2, axis=1), float(np.sum(w * w))


def analyse(ticks: np.ndarray, u: np.ndarray, tick_s: float, spindle_Hz_min: float, spindle_Hz_max: float,
            n_teeth: int, feed_per_tooth_m: float, settings: ChatterSettings) -> ChatterResult:
    """Chatter metrics of one window (module doc). ``spindle_Hz_min/max``: |omega| / 2 pi range over the window."""
    res = ChatterResult()
    run = last_run(np.asarray(ticks), settings.gap_steps)
    t = np.asarray(ticks, dtype=np.int64)[run]
    x = np.asarray(u, float)[run]
    n = len(t)
    res.n_samples = n
    f_s = 0.5 * (spindle_Hz_min + spindle_Hz_max)
    floor = max(settings.min_amplitude_m, settings.min_amplitude_per_feed * max(feed_per_tooth_m, 0.0))
    res.threshold_m = floor
    if n < 16 or f_s <= 0.0:
        return res
    dt = float(np.median(np.diff(t))) * tick_s
    T = n * dt
    res.duration_s = T
    if T * f_s < 2.0:
        return res
    xd = _detrend(x)
    nfft = 1 << int(math.ceil(math.log2(2 * n)))
    P, w2 = _spectrum(xd, nfft)
    freqs = np.fft.rfftfreq(nfft, dt)
    df_res = 1.0 / T
    spread = max(spindle_Hz_max - spindle_Hz_min, 0.0)
    forced = np.zeros(len(freqs), bool)
    k_max = int(freqs[-1] / f_s) + 1
    half_lines = np.zeros(len(freqs), bool)
    even = n_teeth >= 2 and n_teeth % 2 == 0
    for k in range(1, k_max + 1):
        fk = k * f_s
        bw = 2.0 * df_res + 0.5 * k * spread
        mask = np.abs(freqs - fk) <= bw
        forced |= mask
        if even and (2 * k) % n_teeth == 0 and ((2 * k) // n_teeth) % 2 == 1:
            half_lines |= mask
    band = freqs >= 0.5 * f_s
    if even and np.any(half_lines & band):
        # the window halves: the lines' energy must have grown and carry a share of the recent half's energy
        m = n // 2
        P1, _ = _spectrum(_detrend(x[:m]), nfft)
        P2, _ = _spectrum(_detrend(x[m:2 * m]), nfft)
        e1 = float(np.sum(P1[half_lines & band]))
        e2 = float(np.sum(P2[half_lines & band]))
        total2 = float(np.sum(P2[band]))
        if total2 > 0.0 and e2 >= settings.flip_min_share * total2 and e2 >= settings.flip_growth_ratio * e1:
            forced &= ~half_lines
            res.flip = True
    E_f = float(np.sum(P[band & forced]))
    E_n = float(np.sum(P[band & ~forced]))
    res.forced_energy = E_f
    res.nonforced_energy = E_n
    res.index = E_n / (E_n + E_f) if E_n + E_f > 0.0 else 0.0
    cand = np.nonzero(band & ~forced)[0]
    if len(cand) == 0 or E_n <= 0.0:
        return res
    i = int(cand[np.argmax(P[cand])])
    f_c = float(freqs[i])
    if 0 < i < len(P) - 1 and P[i - 1] > 0.0 and P[i + 1] > 0.0 and P[i] > 0.0:
        a, b, c = math.log(P[i - 1]), math.log(P[i]), math.log(P[i + 1])
        den = a - 2.0 * b + c
        if den < 0.0:
            f_c += 0.5 * (a - c) / den * (freqs[1] - freqs[0])
    res.freq_Hz = f_c
    bw_c = 2.0 * df_res
    sel = (np.abs(freqs - f_c) <= bw_c) & ~forced
    res.amp_m = 2.0 * math.sqrt(float(np.sum(P[sel])) / (nfft * w2)) if np.any(sel) else 0.0
    # onset and growth from the band envelope
    X = np.fft.rfft(xd, n=nfft, axis=0)
    keep = (np.abs(freqs - f_c) <= max(bw_c, 0.05 * f_c)) & ~forced
    X[~keep, :] = 0.0
    xb = np.fft.irfft(X, n=nfft, axis=0)[:n]
    env = np.sqrt(np.sum(np.abs(hilbert(xb, axis=0)) ** 2, axis=1))
    res.chatter = bool(res.index >= settings.index_threshold and res.amp_m >= floor)
    above = np.nonzero(env >= floor)[0]
    if res.chatter and len(above):
        res.onset_tick = int(t[int(above[0])])
    pos = env > 0.0
    if np.count_nonzero(pos) >= 8:
        tt = (t[pos] - t[0]) * tick_s
        res.growth_per_s = float(np.polyfit(tt, np.log(env[pos]), 1)[0])
    return res


__all__ = ["ChatterSettings", "ChatterResult", "analyse", "last_run"]
