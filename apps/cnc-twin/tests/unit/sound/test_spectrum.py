"""The synthesised sound carries the physics: tooth passing at N n / 60 with its harmonics while cutting,
nothing of it in air, chatter at the frequency the vibration channels give, the fast tier resampled as it was
recorded; any window equals the same samples of the whole run."""

import math

import numpy as np
import pytest

from cnctwin.sound import synth
from cnctwin.sound.synth import Synth, render_window

from soundkit import (CHATTER_FROM_S, CUT_FROM_S, F_CHATTER, F_TOOTH, OMEGA, fast_window, line_dB, make_inputs,
                      prominence_dB, spectrum)

FS = 44100.0


def test_spectrum_peaks_at_the_tooth_passing_and_chatter_frequencies(inputs):
    window = render_window(inputs, CHATTER_FROM_S + 0.1, 2.0, sources=("cutting", "chatter"))
    freqs, spec = spectrum(window.pressure_Pa, FS)
    peaks = [i for i in range(1, len(spec) - 1) if spec[i] >= spec[i - 1] and spec[i] >= spec[i + 1]]
    top = sorted(peaks, key=lambda i: spec[i], reverse=True)[:4]
    top_f = sorted(freqs[i] for i in top)
    for expected in (F_TOOTH, 2 * F_TOOTH, F_CHATTER):
        assert min(abs(f - expected) for f in top_f) < 3.0, (expected, top_f)
    # before the chatter starts the chatter line is absent, the tooth passing line is there
    calm = render_window(inputs, CUT_FROM_S + 0.1, CHATTER_FROM_S - 0.05, sources=("cutting", "chatter"))
    f2, s2 = spectrum(calm.pressure_Pa, FS)
    assert line_dB(f2, s2, F_CHATTER) < line_dB(f2, s2, F_TOOTH) - 40.0


def test_no_tooth_passing_in_air_but_the_spindle_still_sounds(inputs):
    cut_only = render_window(inputs, 0.2, 0.9, sources=("cutting",))
    assert np.max(np.abs(cut_only.pressure_Pa)) < 1e-12        # air: no tooth pass (FFT round-off of the next cut)
    sources = ("cutting", "unbalance", "bearings", "spindle_motor")
    air = render_window(inputs, 0.3, 0.8, sources=sources)
    cut = render_window(inputs, 1.1, 1.6, sources=sources)
    fa, sa = spectrum(air.pressure_Pa, FS)
    fc, sc = spectrum(cut.pressure_Pa, FS)
    assert prominence_dB(fc, sc, F_TOOTH) > 20.0                               # a line while cutting
    assert prominence_dB(fa, sa, F_TOOTH) < 12.0     # noise floor only (5-bin maxima of noise: < 12 dB, p ~ 1e-4)
    assert line_dB(fc, sc, F_TOOTH) > line_dB(fa, sa, F_TOOTH) + 25.0
    f_rot = abs(OMEGA) / (2.0 * math.pi)
    assert prominence_dB(fa, sa, f_rot) > 15.0                                 # unbalance at 1x in air
    assert np.sqrt(np.mean(air.pressure_Pa ** 2)) > 1e-3


def test_harmonics_follow_the_immersion_short_pulses_are_rich():
    def harmonic_ratio(mean_frac):
        inp = make_inputs(chatter_from_s=None)
        inp.env_mean_N = inp.env_max_N * mean_frac
        w = render_window(inp, 1.1, 1.9, sources=("cutting",))
        f, s = spectrum(w.pressure_Pa, FS)
        return line_dB(f, s, 3 * F_TOOTH) - line_dB(f, s, F_TOOTH)
    assert harmonic_ratio(0.05) > harmonic_ratio(0.4) + 6.0                  # light radial immersion: more harmonics


def test_fast_tier_is_resampled_band_limited_and_crossfaded():
    fast = fast_window(700.0, 120.0, 0.40, 0.60)
    inp = make_inputs(cut_from_s=5.0, chatter_from_s=None, fast=[fast])
    s = Synth(inp, sources=("cutting",))
    j = int(0.5 * FS) // s.B
    force = s.force_block(j)
    t = s._times(j)
    inside = (t > 0.41) & (t < 0.59)
    assert np.allclose(force[inside, 0], 120.0 * np.sin(2.0 * math.pi * 700.0 * t[inside]), atol=0.02)
    outside = (t < 0.399) | (t > 0.601)
    assert np.all(force[outside] == 0.0)
    w = render_window(inp, 0.42, 0.58, sources=("cutting",))
    f, sp = spectrum(w.pressure_Pa, FS)
    assert abs(f[np.argmax(sp)] - 700.0) < 4.0


def test_a_window_is_the_same_samples_as_the_whole_run(inputs):
    whole = render_window(inputs, 0.0, inputs.end_s)
    assert len(whole.pressure_Pa) == int(math.ceil(inputs.end_s * FS))
    B = Synth(inputs).B
    for t0, t1 in ((0.0, 0.1), (B / FS - 0.01, B / FS + 0.01), (1.234567, 1.5), (1.9, 2.0)):
        part = render_window(inputs, t0, t1)
        assert np.array_equal(part.pressure_Pa, whole.pressure_Pa[part.n0:part.n0 + len(part.pressure_Pa)])


def test_sources_and_listeners_are_checked(inputs):
    with pytest.raises(ValueError):
        Synth(inputs, sources=("cutting", "drums"))
    with pytest.raises(ValueError):
        Synth(inputs, listener="ceiling")
    outside = render_window(inputs, 1.1, 1.6, sources=("cutting",), listener="operator")
    inside = render_window(inputs, 1.1, 1.6, sources=("cutting",), listener="inside")
    level = lambda w: 20 * math.log10(np.sqrt(np.mean(w.pressure_Pa ** 2)))       # noqa: E731
    assert level(inside) > level(outside) + 10.0                                  # the enclosure attenuates the cut
    split = render_window(inputs, 1.1, 1.2, split=True)
    assert np.allclose(split.buses["inside"] + split.buses["outside"], split.pressure_Pa)
    assert set(synth.SOURCES) >= set(synth.FORCE_SOURCES)
