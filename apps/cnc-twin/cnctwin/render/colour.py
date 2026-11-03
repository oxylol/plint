"""Colour science of the renderer: sRGB transfer functions (IEC 61966-2-1), colour maps with colour bars,
and the colour and luminance of incandescent bodies from Planck's law with the CIE 1931 observer
(docs/models/render.md "Colour").

Lighting runs in linear RGB; pictures are written in sRGB. Colour maps are matplotlib's published 256-entry
tables (viridis family, cividis, ColorBrewer Blues/RdBu), sampled at the layer value by linear
interpolation between entries (in sRGB, as published).
"""

from __future__ import annotations

import functools
import math

import numpy as np

from . import params as P

# ----------------------------------------------------------------------------- sRGB transfer (IEC 61966-2-1)
_SRGB_LINEAR_LIMIT = 0.0031308
_SRGB_ENCODED_LIMIT = 0.04045
_SRGB_SLOPE = 12.92
_SRGB_A = 0.055
_SRGB_GAMMA = 2.4


def srgb_to_linear(c) -> np.ndarray:
    """Encoded sRGB in [0, 1] -> linear RGB."""
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= _SRGB_ENCODED_LIMIT, c / _SRGB_SLOPE, ((c + _SRGB_A) / (1.0 + _SRGB_A)) ** _SRGB_GAMMA)


def linear_to_srgb(c) -> np.ndarray:
    """Linear RGB -> encoded sRGB in [0, 1] (values clipped to [0, 1] first)."""
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= _SRGB_LINEAR_LIMIT, _SRGB_SLOPE * c, (1.0 + _SRGB_A) * c ** (1.0 / _SRGB_GAMMA) - _SRGB_A)


def to_uint8(srgb) -> np.ndarray:
    """sRGB in [0, 1] -> uint8 by rounding half up (deterministic)."""
    return np.floor(np.clip(np.asarray(srgb, dtype=np.float64), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# ----------------------------------------------------------------------------- colour maps
@functools.lru_cache(maxsize=None)
def lut(name: str) -> np.ndarray:
    """The 256 x 3 sRGB table of a matplotlib colour map (read once per process)."""
    import matplotlib

    cmap = matplotlib.colormaps[name]
    table = np.asarray(cmap(np.linspace(0.0, 1.0, 256)), dtype=np.float64)[:, :3]
    table.setflags(write=False)
    return table


def apply_colormap(values, vmin: float, vmax: float, name: str, nan_srgb=(0.5, 0.5, 0.5)) -> np.ndarray:
    """sRGB colours [n, 3] of ``values`` on the map ``name`` between ``vmin`` and ``vmax`` (clipped); NaN values
    get ``nan_srgb``."""
    values = np.asarray(values, dtype=np.float64)
    table = lut(name)
    span = vmax - vmin
    if not math.isfinite(span) or span <= 0.0:
        span = 1.0
    x = np.clip((values - vmin) / span, 0.0, 1.0) * (len(table) - 1)
    x = np.where(np.isfinite(x), x, 0.0)
    i0 = np.minimum(np.floor(x).astype(np.int64), len(table) - 2)
    frac = (x - i0)[..., None]
    out = table[i0] * (1.0 - frac) + table[i0 + 1] * frac
    bad = ~np.isfinite(values)
    if np.any(bad):
        out[bad] = np.asarray(nan_srgb, dtype=np.float64)
    return out


def nice_ticks(vmin: float, vmax: float, count: int = 5) -> list[float]:
    """Round tick values inside [vmin, vmax] (1, 2, 2.5, 5 x 10^k steps), at least the two limits."""
    if not (math.isfinite(vmin) and math.isfinite(vmax)) or vmax <= vmin:
        return [float(vmin)]
    raw = (vmax - vmin) / max(count - 1, 1)
    power = 10.0 ** math.floor(math.log10(raw))
    step = power
    for factor in (1.0, 2.0, 2.5, 5.0, 10.0):
        if factor * power >= raw:
            step = factor * power
            break
    first = math.ceil(vmin / step - 1e-9) * step
    ticks = []
    value = first
    while value <= vmax + 1e-9 * step:
        ticks.append(round(value / step) * step)
        value += step
    return ticks or [float(vmin), float(vmax)]


def robust_range(values, percentiles=None, symmetric: bool = False) -> tuple[float, float]:
    """Colour bar limits: the given percentiles of the finite values (``render.range_percentiles``), symmetric
    about zero when asked (deviation maps)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    lo_pct, hi_pct = (P.vector(P.VIEW, "range_percentiles") if percentiles is None else percentiles)
    lo, hi = (float(x) for x in np.percentile(v, [lo_pct, hi_pct]))
    if symmetric:
        m = max(abs(lo), abs(hi))
        lo, hi = -m, m
    if hi <= lo:
        pad = abs(hi) * 1e-3 if hi != 0.0 else 1.0
        lo, hi = lo - pad, hi + pad
    return lo, hi


# ----------------------------------------------------------------------------- incandescence
def _cmf(fit, wavelength_nm: np.ndarray) -> np.ndarray:
    out = np.zeros_like(wavelength_nm)
    for a, b, g, d in fit:
        s = np.where(wavelength_nm < b, g, d)
        t = (wavelength_nm - b) * s
        out += a * np.exp(-0.5 * t * t)
    return out


@functools.lru_cache(maxsize=None)
def _observer() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wavelengths (m), the CIE 1931 colour-matching functions there [n, 3], and the integration weights."""
    p = P.cues()
    start, stop, step = (float(v) for v in p["cmf_wavelengths_m"])
    lam = np.arange(start, stop + 0.5 * step, step)
    nm = lam * 1e9
    cmf = np.stack([_cmf(p["cmf_fit_x"], nm), _cmf(p["cmf_fit_y"], nm), _cmf(p["cmf_fit_z"], nm)], axis=1)
    weights = np.full(len(lam), step)
    weights[0] = weights[-1] = 0.5 * step        # trapezoid rule
    return lam, cmf, weights


def planck_radiance(wavelength_m, temp_K: float) -> np.ndarray:
    """Spectral radiance of a black body (W sr^-1 m^-3), Planck's law."""
    from ..units import BOLTZMANN_J_K

    p = P.cues()
    h = float(p["planck_constant"])
    c = float(p["speed_of_light_m_s"])
    lam = np.asarray(wavelength_m, dtype=np.float64)
    x = h * c / (lam * BOLTZMANN_J_K * temp_K)
    # exp(700) is near the float64 limit; beyond it the radiance is zero to double precision anyway
    return 2.0 * h * c * c / lam ** 5 / np.expm1(np.minimum(x, 700.0))


def blackbody_xyz(temp_K: float) -> np.ndarray:
    """CIE XYZ (unnormalised: the luminance-like Y is proportional to the visible radiance) of a black body."""
    lam, cmf, weights = _observer()
    radiance = planck_radiance(lam, temp_K)
    return (cmf * (radiance * weights)[:, None]).sum(axis=0)


def blackbody_chromaticity_rgb(temp_K: float) -> np.ndarray:
    """Linear sRGB of a black body at ``temp_K`` scaled to a maximum component of 1 (the hue of the glow)."""
    xyz = blackbody_xyz(float(temp_K))
    rgb = np.asarray(P.cues()["xyz_to_linear_srgb"], dtype=np.float64) @ xyz
    rgb = np.maximum(rgb, 0.0)
    peak = float(rgb.max())
    return rgb / peak if peak > 0.0 else np.zeros(3)


def glow_intensity(temp_K, draper_K: float | None = None, saturation_K: float | None = None) -> np.ndarray:
    """Display intensity of incandescence in [0, 1]: 0 at and below the Draper point, rising with the logarithm
    of the visible (Y) radiance to 1 at ``glow_saturation_K``."""
    p = P.cues()
    t0 = float(p["draper_point_K"] if draper_K is None else draper_K)
    t1 = float(p["glow_saturation_K"] if saturation_K is None else saturation_K)
    temps = np.atleast_1d(np.asarray(temp_K, dtype=np.float64))
    y0 = blackbody_xyz(t0)[1]
    y1 = blackbody_xyz(t1)[1]
    out = np.zeros(temps.shape)
    for i, t in enumerate(temps):
        if not math.isfinite(t) or t <= t0:
            continue
        y = blackbody_xyz(float(t))[1]
        out[i] = min(1.0, math.log(y / y0) / math.log(y1 / y0))
    return out


def glow_rgb(temp_K) -> np.ndarray:
    """Linear RGB emission [n, 3] of incandescent chips/surfaces: hue from Planck x CIE, intensity from
    :func:`glow_intensity` (zero below the Draper point)."""
    temps = np.atleast_1d(np.asarray(temp_K, dtype=np.float64))
    level = glow_intensity(temps)
    out = np.zeros((len(temps), 3))
    for i, (t, a) in enumerate(zip(temps, level)):
        if a > 0.0:
            out[i] = blackbody_chromaticity_rgb(float(t)) * a
    return out
