"""The sound kernels against their plain-Python references (architecture 5.5) and their own properties."""

import math

import numpy as np
import pytest

from cnctwin.core.rng import stream_key
from cnctwin.sound import dsp as D
from cnctwin.sound import kernels as K
from cnctwin.sound import reference as R

RNG = np.random.default_rng(21)


def test_sinc_resampling_matches_the_reference_and_reproduces_a_band_limited_sine():
    dt = 1.0 / 125_000.0
    t = np.arange(0.0, 0.02, dt)
    x = np.ascontiguousarray(np.sin(2 * math.pi * 3000.0 * t) + 0.3 * RNG.normal(size=len(t)))
    out = np.empty(600)
    K.sinc_resample(0.0, dt, x, 100, 44100.0, 20000.0, 48, out)
    ref = R.sinc_resample_ref(0.0, dt, x, 100, 44100.0, 20000.0, 48, 600)
    assert np.allclose(out, ref, atol=1e-12)
    clean = np.ascontiguousarray(np.sin(2 * math.pi * 3000.0 * t))
    K.sinc_resample(0.0, dt, clean, 100, 44100.0, 20000.0, 48, out)
    ts = (100 + np.arange(600)) / 44100.0
    inner = (ts > 48 * dt) & (ts < t[-1] - 48 * dt)
    assert np.max(np.abs(out[inner] - np.sin(2 * math.pi * 3000.0 * ts[inner]))) < 2e-3


def test_band_limited_impulses_match_the_reference_and_keep_their_area():
    times = np.ascontiguousarray(RNG.uniform(0.0, 0.01, 20))
    strength = np.ascontiguousarray(RNG.uniform(0.1, 1.0, 20))
    out = np.zeros(441)
    K.place_impulses(times, strength, 0, 44100.0, 20000.0, 48, out)
    assert np.allclose(out, R.place_impulses_ref(times, strength, 0, 44100.0, 20000.0, 48, 441), atol=1e-9)
    single = np.zeros(400)
    K.place_impulses(np.array([0.004]), np.array([2.0]), 0, 44100.0, 20000.0, 48, single)
    assert single.sum() / 44100.0 == pytest.approx(2.0, rel=1e-3)               # the impulse keeps its integral


def test_hermite_matches_scipy_and_clamps_outside():
    xs = np.ascontiguousarray(np.sort(RNG.uniform(0.0, 1.0, 30)))
    ys = np.ascontiguousarray(np.sin(5 * xs))
    dys = np.ascontiguousarray(5 * np.cos(5 * xs))
    x = np.ascontiguousarray(np.linspace(-0.1, 1.1, 777))
    y, dy = np.empty(len(x)), np.empty(len(x))
    K.hermite(xs, ys, dys, x, y, dy)
    y_ref, dy_ref = R.hermite_ref(xs, ys, dys, x)
    assert np.allclose(y, y_ref, atol=1e-12) and np.allclose(dy, dy_ref, atol=1e-9)


def test_counter_based_noise_and_dither_are_deterministic_and_standard():
    a, b = np.empty(50_000), np.empty(50_000)
    K.hash_noise(7, stream_key("sound.test"), 1000, a)
    K.hash_noise(7, stream_key("sound.test"), 1000, b)
    assert np.array_equal(a, b)
    assert abs(a.mean()) < 0.02 and a.std() == pytest.approx(1.0, abs=0.02)
    shifted = np.empty(49_000)
    K.hash_noise(7, stream_key("sound.test"), 2000, shifted)
    assert np.array_equal(shifted, a[1000:])                                    # a function of the sample index
    d = np.empty(50_000)
    K.hash_tpdf(7, stream_key("sound.dither"), 0, d)
    assert d.min() > -1.0 and d.max() < 1.0 and abs(d.mean()) < 0.01 and d.std() == pytest.approx(math.sqrt(1 / 6), abs=0.01)
    u = np.empty(10_000)
    K.hash_uniforms(7, stream_key("sound.test"), 0, 2, u)
    assert 0.0 <= u.min() and u.max() < 1.0 and abs(u.mean() - 0.5) < 0.01


def test_fir_design_and_block_convolution_on_the_absolute_grid():
    fs = 44100.0
    h = D.design_fir(lambda f: np.where(f < 5000.0, 1.0, 0.0), fs, 513)
    assert len(h) == 513 and np.allclose(h, h[::-1])                             # zero phase
    x = RNG.normal(size=5 * 1024)
    direct = np.convolve(x, h)[256:256 + len(x)]
    fir = D.CentredFIR(h, 1024)
    blocks = lambda j: x[j * 1024:(j + 1) * 1024] if 0 <= j < 5 else None       # noqa: E731
    streamed = np.concatenate([fir.output(J, blocks) for J in range(5)])
    assert np.allclose(streamed, direct, atol=1e-10)
    again = D.CentredFIR(h, 1024)
    assert np.array_equal(again.output(3, blocks), streamed[3 * 1024:4 * 1024])  # one block alone: same samples


def test_every_kernel_has_one_compiled_signature():
    for kernel in K.KERNELS:
        assert len(kernel.signatures) == 1, kernel.__name__
