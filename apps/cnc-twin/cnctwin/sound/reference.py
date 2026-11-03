"""Plain-Python twins of the sound kernels (architecture 5.5): direct sums and SciPy's Hermite spline, for the
tests that compare them with the Numba kernels."""

from __future__ import annotations

import math

import numpy as np


def blackman_ref(x: np.ndarray, half: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    u = (x + half) / (2.0 * half)
    w = 0.42 - 0.5 * np.cos(2.0 * math.pi * u) + 0.08 * np.cos(4.0 * math.pi * u)
    return np.where(np.abs(x) < half, w, 0.0)


def lowpass_ref(dt: np.ndarray, cutoff: float, half_s: float) -> np.ndarray:
    return 2.0 * cutoff * np.sinc(2.0 * cutoff * np.asarray(dt, dtype=np.float64)) * blackman_ref(dt, half_s)


def sinc_resample_ref(t_first: float, dt: float, x: np.ndarray, n0: int, fs: float, cutoff: float,
                      half_width: int, n: int) -> np.ndarray:
    t_src = t_first + dt * np.arange(len(x))
    out = np.zeros(n)
    for i in range(n):
        t = (n0 + i) / fs
        if t < t_src[0] or t > t_src[-1]:
            continue
        centre = int(math.floor((t - t_first) / dt))
        j = np.arange(max(centre - half_width, 0), min(centre + half_width + 1, len(x) - 1) + 1)
        out[i] = float(np.sum(x[j] * lowpass_ref(t - t_src[j], cutoff, half_width * dt) * dt))
    return out


def place_impulses_ref(times: np.ndarray, strength: np.ndarray, n0: int, fs: float, cutoff: float,
                       half_width: int, n: int) -> np.ndarray:
    out = np.zeros(n)
    t = (n0 + np.arange(n)) / fs
    for tk, sk in zip(times, strength):
        centre = int(math.floor(tk * fs - n0))
        lo = max(centre - half_width, 0)
        hi = min(centre + half_width + 1, n - 1)
        if hi < lo:
            continue
        idx = np.arange(lo, hi + 1)
        out[idx] += sk * lowpass_ref(t[idx] - tk, cutoff, half_width / fs)
    return out


def hermite_ref(xs: np.ndarray, ys: np.ndarray, dys: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.interpolate import CubicHermiteSpline

    spline = CubicHermiteSpline(xs, ys, dys)
    y = spline(x)
    dy = spline(x, 1)
    y = np.where(x <= xs[0], ys[0], np.where(x >= xs[-1], ys[-1], y))
    dy = np.where((x <= xs[0]) | (x >= xs[-1]), 0.0, dy)
    return y, dy
