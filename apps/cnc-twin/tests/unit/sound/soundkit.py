"""Helpers of the sound tests: synthetic inputs (constant spindle speed, a cut that starts half-way, a chatter
episode), spectra and lines."""

from __future__ import annotations

import math
import numpy as np
from cnctwin.sound.inputs import FastWindow, SoundInputs

OMEGA = -1000.0          # rad/s, M03
TEETH = 3
F_TOOTH = TEETH * abs(OMEGA) / (2.0 * math.pi)       # 477.5 Hz
F_CHATTER = 1234.0
CUT_FROM_S = 1.0
CHATTER_FROM_S = 1.5


def make_inputs(end_s: float = 2.0, cut_from_s: float = CUT_FROM_S, chatter_from_s: float | None = CHATTER_FROM_S,
                fast: list | None = None) -> SoundInputs:
    ticks = np.arange(int(round(end_s * 1000)))
    t = ticks * 1e-3
    inp = SoundInputs(end_s=end_s, seed=3)
    inp.frame_t_s = t
    inp.spindle_angle_rad = OMEGA * t
    inp.axis_names = ["X", "Y", "Z"]
    inp.axis_linear = [True, True, True]
    inp.axis_pos = np.zeros((len(t), 3))
    inp.teeth = np.full(len(t), TEETH, np.int64)
    inp.tool_radius_m = np.full(len(t), 0.005)
    cutting = (t >= cut_from_s)[:, None]
    inp.env_t_s = t
    inp.env_span_s = 1e-3
    inp.env_min_N = np.zeros((len(t), 3))
    inp.env_max_N = np.where(cutting, [300.0, 200.0, 40.0], 0.0)
    inp.env_mean_N = np.where(cutting, [80.0, 50.0, 10.0], 0.0)
    macro_t = np.arange(1, int(round(end_s * 100)) + 1) * 0.01
    inp.macro_t_s = macro_t
    inp.macro_span_s = 0.01
    chattering = (macro_t > chatter_from_s) if chatter_from_s is not None else np.zeros(len(macro_t), bool)
    inp.chatter_freq_Hz = np.where(chattering, F_CHATTER, 0.0)
    inp.chatter_index = np.where(chattering, 0.5, 0.0)
    inp.vibration_pp_m = np.where(macro_t > cut_from_s, 2e-5, 0.0)
    inp.packing_torque_Nm = np.zeros(len(macro_t))
    inp.chip_rate_kg_s = np.zeros(len(macro_t))
    inp.spindle_modes = [(1200.0, 0.03, 5e7, "x"), (1200.0, 0.03, 5e7, "y")]
    inp.fast = list(fast or [])
    return inp


def spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    window = np.hanning(len(x))
    return np.fft.rfftfreq(len(x), 1.0 / fs), np.abs(np.fft.rfft(x * window))


def line_dB(freqs, spec, f0: float, width_Hz: float = 4.0) -> float:
    near = np.abs(freqs - f0) <= width_Hz
    return 20.0 * math.log10(float(spec[near].max()) + 1e-300)


def fast_window(f_Hz: float, amplitude_N: float, t0: float, t1: float, rate_Hz: float = 125_000.0) -> FastWindow:
    t = np.arange(t0, t1, 1.0 / rate_Hz)
    force = np.zeros((len(t), 3))
    force[:, 0] = amplitude_N * np.sin(2.0 * math.pi * f_Hz * t)
    return FastWindow(float(t[0]), 1.0 / rate_Hz, force)


def prominence_dB(freqs, spec, f0: float, width_Hz: float = 4.0, band_Hz: float = 40.0) -> float:
    """Level of the line at f0 above the median of the spectrum around it (a tone stands out, noise does not)."""
    near = np.abs(freqs - f0) <= width_Hz
    around = (np.abs(freqs - f0) <= band_Hz) & ~near
    return 20.0 * math.log10((float(spec[near].max()) + 1e-300) / (float(np.median(spec[around])) + 1e-300))
