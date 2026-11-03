"""Colour maps and colour bars, sRGB, blackbody colours (Planck x CIE 1931), the cue thresholds and the
deterministic sprites of sparks, smoke and steam."""

import math

import numpy as np
import pytest

from cnctwin.render import colour as C
from cnctwin.render import cues as Q
from cnctwin.render import params as P
from cnctwin.render.colour import apply_colormap, lut, nice_ticks, robust_range
from cnctwin.render.layers import LAYERS, SPECS


def test_srgb_round_trip_and_endpoints():
    x = np.linspace(0.0, 1.0, 1001)
    assert np.allclose(C.linear_to_srgb(C.srgb_to_linear(x)), x, atol=1e-12)
    assert C.srgb_to_linear(0.5) == pytest.approx(0.21404, abs=1e-5)          # IEC 61966-2-1 value
    assert C.to_uint8([0.0, 0.5, 1.0]).tolist() == [0, 128, 255]


def test_colour_maps_hit_their_ends_and_keep_lightness_monotonic():
    for layer in LAYERS:
        name = SPECS[layer].cmap()
        table = lut(name)
        assert table.shape == (256, 3)
        out = apply_colormap([-5.0, 0.0, 10.0, 99.0], 0.0, 10.0, name)
        assert np.allclose(out[0], table[0]) and np.allclose(out[1], table[0]) and np.allclose(out[2], table[-1])
    for name in ("plasma", "viridis", "cividis"):
        y = C.srgb_to_linear(lut(name)) @ np.array([0.2126, 0.7152, 0.0722])   # relative luminance
        assert np.all(np.diff(y) > -1e-4), name
    assert np.allclose(apply_colormap([np.nan], 0, 1, "viridis", nan_srgb=(0.1, 0.2, 0.3)), [[0.1, 0.2, 0.3]])


def test_ticks_and_ranges():
    assert nice_ticks(0.0, 1.0) == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert nice_ticks(20.0, 151.0) == pytest.approx([50.0, 100.0, 150.0])        # steps of 1, 2, 2.5, 5 x 10^k
    assert nice_ticks(20.0, 151.0, count=8) == pytest.approx([20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0])
    values = np.concatenate([np.linspace(-3.0, 2.0, 1000), [1e6]])           # one outlier does not set the range
    lo, hi = robust_range(values, percentiles=(1, 99))
    assert lo > -3.0 and hi < 3.0
    lo, hi = robust_range(values, percentiles=(1, 99), symmetric=True)
    assert lo == -hi


def test_planck_law_peaks_at_wien_wavelength():
    lam = np.linspace(200e-9, 20e-6, 200001)
    for temp in (800.0, 1500.0, 3000.0):
        peak = lam[np.argmax(C.planck_radiance(lam, temp))]
        assert peak * temp == pytest.approx(2.897771955e-3, rel=1e-3)       # Wien's displacement constant (CODATA)


def test_blackbody_colours_go_from_red_to_white_and_glow_starts_at_the_draper_point():
    red = C.blackbody_chromaticity_rgb(1000.0)
    assert red[0] == pytest.approx(1.0) and red[0] > red[1] > red[2]
    white = C.blackbody_chromaticity_rgb(6500.0)                              # D65-like: nearly neutral
    assert white.min() > 0.85
    draper = float(P.number(P.CUES, "draper_point_K"))
    level = C.glow_intensity([300.0, draper, draper + 50.0, 1100.0, 1400.0, 2500.0])
    assert level[0] == 0.0 and level[1] == 0.0
    assert 0.0 < level[2] < level[3] < level[4] <= 1.0 and level[4] == pytest.approx(1.0) and level[5] == 1.0
    assert np.all(C.glow_rgb([500.0, 700.0]) == 0.0)


def test_cie_fit_reproduces_the_1931_observer_peaks():
    lam, cmf, _ = C._observer()
    nm = lam * 1e9
    assert nm[np.argmax(cmf[:, 1])] == pytest.approx(555.0, abs=5.0)           # photopic peak
    assert cmf[:, 1].max() == pytest.approx(1.0, abs=0.03)
    assert nm[np.argmax(cmf[:, 2])] == pytest.approx(445.0, abs=5.5)           # table peak 445 nm, 5 nm grid


@pytest.mark.parametrize("kind, smoke, boil", [("emulsion", True, True), ("straight_oil", True, False),
                                                ("synthetic", False, True), ("mql", True, False), ("dry", False, False),
                                                ("air", False, False)])
def test_cue_thresholds_follow_the_coolant(kind, smoke, boil):
    th = Q.thresholds({"kind": kind})
    assert (th.smoke_K is not None) == smoke and (th.boil_K is not None) == boil
    assert th.glow_K == pytest.approx(798.0)
    if boil:
        assert th.boil_K == pytest.approx(373.124)
    own = Q.thresholds({"kind": kind, "smoke_point_K": 500.0, "boiling_K": 380.0})
    if smoke:
        assert own.smoke_K == 500.0


def test_smoke_density_and_puffs_are_deterministic_and_continuous():
    full = float(P.number(P.CUES, "cue_full_scale_dK"))
    assert Q.density([300.0, 400.0, 400.0 + full / 2, 400.0 + 2 * full], 400.0).tolist() == pytest.approx([0, 0, 0.5, 1])
    spot = [Q.HotSpot(7, np.array([0.01, 0.02, 0.0]), 470.0)]
    up = np.array([0.0, 0.0, 1.0])
    a = Q.puffs(spot, 1.0, 400.0, "smoke", 11, up)
    b = Q.puffs(spot, 1.0, 400.0, "smoke", 11, up)
    assert np.array_equal(a.p0, b.p0) and np.array_equal(a.alpha, b.alpha)
    later = Q.puffs(spot, 1.0 + 1e-4, 400.0, "smoke", 11, up)
    assert np.max(np.linalg.norm(later.p0 - a.p0, axis=1)) < 1e-3              # puffs move continuously
    other_seed = Q.puffs(spot, 1.0, 400.0, "smoke", 12, up)
    assert not np.array_equal(other_seed.p0, a.p0)
    assert np.all(a.p0[:, 2] >= spot[0].pos_S_m[2])                           # smoke rises
    assert Q.puffs(spot, 1.0, None, "steam", 11, up) is None
    assert Q.puffs([Q.HotSpot(1, np.zeros(3), 350.0)], 1.0, 400.0, "smoke", 11, up) is None


def test_sparks_only_above_the_draper_point_and_streak_behind_the_chip():
    pos = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    vel = np.array([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    sparks = Q.sparks(pos, vel, np.array([600.0, 1200.0]))
    assert len(sparks) == 1 and sparks.p1[0, 0] < sparks.p0[0, 0]
    assert sparks.rgb[0, 0] == pytest.approx(1.0) and sparks.rgb[0, 0] > sparks.rgb[0, 2]
    assert Q.sparks(pos, vel, np.array([600.0, 700.0])) is None
    assert Q.source_key(3, [0.0101, 0.0, 0.0]) == Q.source_key(3, [0.0102, 0.0003, 0.0002])   # same 2 mm cell
