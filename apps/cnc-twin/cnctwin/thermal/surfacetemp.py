"""Temperature history of the machined surface under the cutting edge (docs/models/thermal.md 3.4).

The new surface is heated by the work's share of three sources that pass over it at the cutting speed V, in the
frame of the edge (xi along the cutting direction, xi = 0 at the edge, xi < 0 the new surface behind it):

* the shear zone: the heat the primary zone conducts into the work (partition's ``Q_s (1 - f_sc)``; Boothroyd 1963,
  Weiner 1955) enters the material that stays below the cut line through the footprint of the shear plane on the
  cut plane, a band 0 < xi < L_s = h / tan(phi) ahead of the edge;
* the edge arc: the friction on the edge arc below the stagnation point, the work's share of the edge heat's friction
  part (contact.contact_power_parts; Blok's equal temperatures give the share, mechanics.heat.flank_heat_to_tool) on
  -l_a < xi < 0, l_a the arc length;
* the plastic work of the edge pressure (contact_power_parts' pressure part, its work share) generated uniformly to
  the depth delta of the ploughed zone under the arc (mechanics-contact ``ploughing_zone_depth_ratio`` x arc):
  buried plane sources and their images (adiabatic surface; Carslaw & Jaeger 1959 ch. X), three Gauss points over
  the depth;
* the wear land: the friction on the land (the hot loop's flank heat, its work share) on -(l_a + VB) < xi < -l_a.

The partition's flank share applies to the arc and land heat together over l_f = l_a + VB (thermal.partition); here
each part is put where it is generated.

Each source is Jaeger's (1942) uniform rectangle of width b (the engaged edge length) moving at V over a half space,
on for the passage time (``thermal.kernels.moving_rect_rise``, valid at any Peclet number). A material point at depth
z meets the sources in the order shear band, flank contact and then cools by conduction into the bulk: its history is
T(z, t) = T_b + theta(xi0 - V t, z), T_b the bulk plus feed-scale temperature of the two-scale model (3.3). The
history is sampled at fixed positions (ahead of the shear band, across the contacts and on a geometric ladder behind
them) and depths; damage.py applies the criteria to it.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import numpy as np

from . import kernels as K

# positions across the contacts (fractions of the band / contact) and behind them (multiples of the contact scale)
_AHEAD = (1.0, 0.5, 0.0)
_ACROSS = (0.25, 0.5, 0.75, 0.9, 0.95, 1.0)   # the maximum of a fast band lies just inside its trailing edge
_BEHIND = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)
_PLOUGH_NODES = 3


@dataclass
class EdgeSources:
    """The work-bound heat of one edge passage as the sources of the module doc (W, m, s)."""

    speed_m_s: float                 # cutting speed V (work relative to the edge)
    width_m: float                   # engaged edge length b
    duration_s: float                # time the sources have been on (one passage)
    shear_W: float = 0.0
    shear_length_m: float = 0.0      # L_s = h / tan(phi)
    arc_W: float = 0.0               # friction on the edge arc (surface source)
    plough_W: float = 0.0            # plastic work of the edge pressure (layer source under the arc)
    arc_length_m: float = 0.0        # l_a
    plough_depth_m: float = 0.0      # delta
    land_W: float = 0.0              # friction on the wear land (surface source behind the arc)
    land_length_m: float = 0.0       # VB

    @property
    def flank_length_m(self) -> float:
        return self.arc_length_m + self.land_length_m

    def active(self) -> bool:
        edge = (self.arc_W > 0.0 or self.plough_W > 0.0) and self.arc_length_m > 0.0
        land = self.land_W > 0.0 and self.land_length_m > 0.0
        shear = self.shear_W > 0.0 and self.shear_length_m > 0.0
        return self.speed_m_s > 0.0 and self.width_m > 0.0 and self.duration_s > 0.0 and (edge or land or shear)

    def contact_scale_m(self) -> float:
        return max(self.flank_length_m, self.shear_length_m, 1e-9)


@dataclass
class SurfaceHistory:
    """Sampled temperature history of the machined surface (K) at ``depth_m`` (rows) and ``time_s`` (columns)."""

    depth_m: np.ndarray
    time_s: np.ndarray
    temp_K: np.ndarray               # [n_depth, n_time]
    background_K: float
    edge_index: int                  # column of the edge (xi = 0)
    sources: EdgeSources | None = None
    k_W_mK: float = 0.0
    alpha_m2_s: float = 0.0
    extra: dict = field(default_factory=dict)

    def peak_K(self) -> np.ndarray:
        return np.max(self.temp_K, axis=1)

    def at_depth(self, z: float) -> np.ndarray:
        """History at depth z (linear interpolation between the sampled depths, clamped)."""
        d = self.depth_m
        if z <= d[0]:
            return self.temp_K[0].copy()
        if z >= d[-1]:
            return self.temp_K[-1].copy()
        j = int(np.searchsorted(d, z) - 1)
        w = (z - d[j]) / (d[j + 1] - d[j])
        return self.temp_K[j] * (1.0 - w) + self.temp_K[j + 1] * w


@functools.lru_cache(maxsize=8)
def _gauss(n: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(int(n))
    return np.ascontiguousarray(x, dtype=float), np.ascontiguousarray(w, dtype=float)


def _band(q, half, b, v, k, alpha, t, x, z, glx, glw) -> float:
    return float(K.moving_rect_rise(q, half, b, v, k, alpha, t, x, 0.0, z, glx, glw))


def rise(src: EdgeSources, k: float, alpha: float, xi: float, z: float, glx, glw) -> float:
    """Temperature rise (K) at xi (m, relative to the edge, + ahead) and depth z of the sources."""
    b = 0.5 * src.width_m
    v = src.speed_m_s
    t = src.duration_s
    total = 0.0
    if src.shear_W > 0.0 and src.shear_length_m > 0.0:
        ls = 0.5 * src.shear_length_m
        q = src.shear_W / (src.shear_length_m * src.width_m)
        total += _band(q, ls, b, v, k, alpha, t, xi - ls, z, glx, glw)
    la = 0.5 * src.arc_length_m
    if la > 0.0 and src.arc_W > 0.0:
        q = src.arc_W / (src.arc_length_m * src.width_m)
        total += _band(q, la, b, v, k, alpha, t, xi + la, z, glx, glw)
    if la > 0.0 and src.plough_W > 0.0:
        delta = src.plough_depth_m
        q_area = src.plough_W / (src.arc_length_m * src.width_m)
        if delta <= 0.0:
            total += _band(q_area, la, b, v, k, alpha, t, xi + la, z, glx, glw)
        else:
            gx, gw = _gauss(_PLOUGH_NODES)
            for node, weight in zip(gx, gw):
                zp = 0.5 * delta * (node + 1.0)
                share = 0.5 * weight       # of q_area (the weights sum to 2)
                # a plane source buried at zp in a half space with an adiabatic surface: half the surface-source
                # solution at |z - zp| plus its image at z + zp
                up = _band(q_area, la, b, v, k, alpha, t, xi + la, abs(z - zp), glx, glw)
                im = _band(q_area, la, b, v, k, alpha, t, xi + la, z + zp, glx, glw)
                total += share * 0.5 * (up + im)
    lw = 0.5 * src.land_length_m
    if lw > 0.0 and src.land_W > 0.0:
        q = src.land_W / (src.land_length_m * src.width_m)
        total += _band(q, lw, b, v, k, alpha, t, xi + 2.0 * la + lw, z, glx, glw)
    return total


def positions(src: EdgeSources) -> np.ndarray:
    """Sample positions xi (m), first ahead of the shear band, decreasing (the order a material point meets them)."""
    ls = src.shear_length_m if src.shear_W > 0.0 else 0.0
    lf = src.flank_length_m
    scale = src.contact_scale_m()
    la, lw = src.arc_length_m, src.land_length_m
    xs = [ls * a for a in _AHEAD] if ls > 0.0 else [scale, 0.0]
    xs += [-la * a for a in (0.5, 0.9, 0.95, 1.0)] if la > 0.0 else []
    xs += [-la - lw * a for a in _ACROSS] if lw > 0.0 else []
    xs += [-lf - scale * a for a in _BEHIND]
    out = []
    for x in xs:
        if not out or x < out[-1] - 1e-15:
            out.append(x)
    return np.asarray(out, dtype=float)


def surface_peak(src: EdgeSources, k: float, alpha: float, glx, glw) -> float:
    """Largest rise on the new surface (z = 0) across the contacts and just behind them (K)."""
    if not src.active():
        return 0.0
    lf = src.flank_length_m
    la = src.arc_length_m
    scale = src.contact_scale_m()
    xs = [0.0] + [-la * a for a in (0.5, 0.9, 0.95, 1.0)] + [-la - src.land_length_m * a for a in _ACROSS]
    xs += [-lf - scale * a for a in (0.05, 0.1, 0.25)]
    return max(rise(src, k, alpha, x, 0.0, glx, glw) for x in xs)


def contact_mean_rise(src: EdgeSources, k: float, alpha: float, glx, glw, n: int = 8) -> float:
    """Mean rise of the new surface over the tool-work contact -l_f < xi < 0 (what a tool-work thermocouple
    averages; Gauss-Legendre over the arc and over the land)."""
    gx, gw = _gauss(n)
    total = 0.0
    for a, c in ((0.0, src.arc_length_m), (src.arc_length_m, src.land_length_m)):
        if c <= 0.0:
            continue
        for node, weight in zip(gx, gw):
            total += 0.5 * weight * c * rise(src, k, alpha, -(a + 0.5 * c * (node + 1.0)), 0.0, glx, glw)
    return total / src.flank_length_m if src.flank_length_m > 0.0 else 0.0


def history(src: EdgeSources, background_K: float, k: float, alpha: float, depths_m, glx, glw) -> SurfaceHistory:
    """T(z, t) of a material point of the new surface as it passes the sources (module doc)."""
    xs = positions(src)
    depths = np.asarray(sorted(set(float(d) for d in depths_m)), dtype=float)
    temp = np.empty((depths.size, xs.size))
    for i, z in enumerate(depths):
        for j, x in enumerate(xs):
            temp[i, j] = background_K + rise(src, k, alpha, float(x), float(z), glx, glw)
    times = (xs[0] - xs) / src.speed_m_s
    edge = int(np.argmin(np.abs(xs)))
    return SurfaceHistory(depths, times, temp, float(background_K), edge, src, float(k), float(alpha))


def default_depths(src: EdgeSources, alpha: float, extra=()) -> list[float]:
    """Depths of the history: the surface, the heat-penetration ladder of the passage and ``extra`` (m)."""
    t_c = (src.flank_length_m + src.shear_length_m) / max(src.speed_m_s, 1e-12)
    d = math.sqrt(max(alpha * t_c, 0.0))
    ladder = [0.0] + [d * a for a in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)]
    return sorted(set([float(x) for x in ladder] + [float(x) for x in extra if x >= 0.0]))


__all__ = ["EdgeSources", "SurfaceHistory", "rise", "positions", "surface_peak", "contact_mean_rise", "history",
           "default_depths"]
