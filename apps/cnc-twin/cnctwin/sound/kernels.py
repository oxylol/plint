"""Numba kernels of the sound synthesis (docs/models/sound.md "Signal processing").

All kernels are stateless functions of absolute sample indices and their inputs, so any window of the run's
audio is computed with exactly the same arithmetic as the whole run:

* :func:`hash_noise` white Gaussian noise ``n[i] = hash_normal(seed, stream, n0 + i, 0, 0)`` (core.rng).
* :func:`sinc_resample` band-limited interpolation of a uniformly sampled signal (a fast-tier window) onto the
  audio grid: ``y(t) = sum_j x_j dt h(t - t_j)`` with ``h`` the Blackman-windowed sinc low-pass at ``cutoff``
  (Oppenheim & Schafer 2010, 4.6), ``half_width`` source samples on each side.
* :func:`place_impulses` band-limited impulses (chip impacts) of strength ``J`` (N s) at arbitrary times:
  ``y[n] += J h(t_n - t_k)`` with the same windowed-sinc kernel in the audio rate's own units.
* :func:`hermite` cubic Hermite interpolation of a sampled function with given derivatives (spindle angle).

Plain-Python twins are in :mod:`cnctwin.sound.reference`.
"""

from __future__ import annotations

import math

from numba import njit, types

from ..core.rng import hash_normal, hash_uniform

F1 = types.float64[::1]
OPTS = dict(cache=True, nogil=True, error_model="numpy")

TWO_PI = 2.0 * math.pi


@njit(types.float64(types.float64, types.float64), **OPTS)
def blackman(x, half):
    """Blackman window on [-half, half] (0 outside)."""
    if x <= -half or x >= half:
        return 0.0
    u = (x + half) / (2.0 * half)
    return 0.42 - 0.5 * math.cos(TWO_PI * u) + 0.08 * math.cos(2.0 * TWO_PI * u)


@njit(types.float64(types.float64, types.float64, types.float64), **OPTS)
def lowpass_kernel(dt, cutoff, half_s):
    """Windowed-sinc low-pass impulse response (1/s) at time offset ``dt``: 2 fc sinc(2 fc dt) w(dt)."""
    x = 2.0 * cutoff * dt
    if x == 0.0:
        s = 1.0
    else:
        s = math.sin(math.pi * x) / (math.pi * x)
    return 2.0 * cutoff * s * blackman(dt, half_s)


@njit(types.int64(types.int64, types.int64, types.int64, F1), **OPTS)
def hash_noise(seed, stream, n0, out):
    """Unit-variance white Gaussian noise of absolute samples n0 .. n0 + len(out) - 1."""
    for i in range(out.shape[0]):
        out[i] = hash_normal(seed, stream, n0 + i, 0, 0)
    return out.shape[0]


@njit(types.int64(types.int64, types.int64, types.int64, types.int64, F1), **OPTS)
def hash_uniforms(seed, stream, a0, b, out):
    """``out[i] = hash_uniform(seed, stream, a0 + i, b, 0)``: counter-based uniforms in [0, 1)."""
    for i in range(out.shape[0]):
        out[i] = hash_uniform(seed, stream, a0 + i, b, 0)
    return out.shape[0]


@njit(types.int64(types.int64, types.int64, types.int64, F1), **OPTS)
def hash_tpdf(seed, stream, n0, out):
    """Triangular-PDF dither in (-1, 1) LSB of absolute samples n0 .. (sum of two counter-based uniforms - 1)."""
    for i in range(out.shape[0]):
        out[i] = hash_uniform(seed, stream, n0 + i, 0, 0) + hash_uniform(seed, stream, n0 + i, 1, 0) - 1.0
    return out.shape[0]


@njit(types.int64(types.float64, types.float64, F1, types.int64, types.float64, types.float64, types.int64, F1),
      **OPTS)
def sinc_resample(t_first, dt, x, n0, fs, cutoff, half_width, out):
    """Band-limited value of the signal ``x`` sampled at ``t_first + j dt`` at the audio times ``(n0 + i) / fs``;
    samples whose kernel support leaves the window use the samples that exist (the caller crossfades the
    window edges). Returns the number of output samples inside the window's time span."""
    count = 0
    half_s = half_width * dt
    nx = x.shape[0]
    t_last = t_first + (nx - 1) * dt
    for i in range(out.shape[0]):
        t = (n0 + i) / fs
        if t < t_first or t > t_last:
            out[i] = 0.0
            continue
        centre = (t - t_first) / dt
        j0 = int(math.floor(centre)) - half_width
        j1 = int(math.floor(centre)) + half_width + 1
        if j0 < 0:
            j0 = 0
        if j1 > nx - 1:
            j1 = nx - 1
        acc = 0.0
        for j in range(j0, j1 + 1):
            acc += x[j] * lowpass_kernel(t - (t_first + j * dt), cutoff, half_s) * dt
        out[i] = acc
        count += 1
    return count


@njit(types.int64(F1, F1, types.int64, types.float64, types.float64, types.int64, F1), **OPTS)
def place_impulses(times, strength, n0, fs, cutoff, half_width, out):
    """Add band-limited impulses ``strength[k]`` (signal x seconds) at ``times[k]`` to the samples
    ``(n0 + i) / fs``; returns the number of impulses that touched the block."""
    touched = 0
    half_s = half_width / fs
    n_out = out.shape[0]
    for k in range(times.shape[0]):
        centre = times[k] * fs - n0
        i0 = int(math.floor(centre)) - half_width
        i1 = int(math.floor(centre)) + half_width + 1
        if i1 < 0 or i0 >= n_out:
            continue
        if i0 < 0:
            i0 = 0
        if i1 > n_out - 1:
            i1 = n_out - 1
        for i in range(i0, i1 + 1):
            out[i] += strength[k] * lowpass_kernel((n0 + i) / fs - times[k], cutoff, half_s)
        touched += 1
    return touched


@njit(types.int64(F1, F1, F1, F1, F1, F1), **OPTS)
def hermite(xs, ys, dys, x, y, dy):
    """Cubic Hermite interpolation of ``ys`` (derivatives ``dys``) at the sorted ``xs``, evaluated at ``x``;
    clamped (constant value, zero slope) outside. Writes ``y`` and its derivative ``dy``."""
    n = xs.shape[0]
    j = 0
    for i in range(x.shape[0]):
        xi = x[i]
        if n == 0:
            y[i] = 0.0
            dy[i] = 0.0
            continue
        if xi <= xs[0]:
            y[i] = ys[0]
            dy[i] = 0.0
            continue
        if xi >= xs[n - 1]:
            y[i] = ys[n - 1]
            dy[i] = 0.0
            continue
        while j > 0 and xs[j] > xi:
            j -= 1
        while j < n - 2 and xs[j + 1] <= xi:
            j += 1
        h = xs[j + 1] - xs[j]
        s = (xi - xs[j]) / h
        s2 = s * s
        s3 = s2 * s
        h00 = 2.0 * s3 - 3.0 * s2 + 1.0
        h10 = s3 - 2.0 * s2 + s
        h01 = -2.0 * s3 + 3.0 * s2
        h11 = s3 - s2
        y[i] = h00 * ys[j] + h10 * h * dys[j] + h01 * ys[j + 1] + h11 * h * dys[j + 1]
        d00 = (6.0 * s2 - 6.0 * s) / h
        d10 = 3.0 * s2 - 4.0 * s + 1.0
        d01 = (-6.0 * s2 + 6.0 * s) / h
        d11 = 3.0 * s2 - 2.0 * s
        dy[i] = d00 * ys[j] + d10 * dys[j] + d01 * ys[j + 1] + d11 * dys[j + 1]
    return x.shape[0]


KERNELS = (hash_noise, hash_uniforms, hash_tpdf, sinc_resample, place_impulses, hermite, lowpass_kernel, blackman)
