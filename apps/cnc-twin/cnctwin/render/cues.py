"""Visual cues: what a person at the machine would see glow, smoke or steam (docs/models/render.md
"Visual cues").

* **Glow and sparks.** Above the Draper point (798 K, Draper 1847) a body glows; its colour is the
  chromaticity of a black body at that temperature (Planck 1901 integrated with the CIE 1931 observer, the
  Wyman-Sloan-Shirley fit, converted to linear sRGB) and its drawn intensity rises with the logarithm of the
  visible radiance up to ``glow_saturation_K``. Flying chips above it are sparks: additive streaks of their
  velocity times ``spark_exposure_s``; glowing tool edges and part surfaces emit that light.
* **Smoke.** Where an oil-bearing coolant (straight oil, MQL, the concentrate of an emulsion) meets a surface
  above its smoke point (the fluid's ``smoke_point_degC``, else ``oil_smoke_point_K``) and the coolant is on.
* **Steam.** Where a water-based coolant meets a surface above its boiling point (the fluid's
  ``boiling_degC``, else 99.974 C at 1 atm, IAPWS-95).
* Density of smoke and steam rises linearly from the threshold to full at ``cue_full_scale_dK`` above it.
  Puffs rise along -g at the plume speeds of ``render-cues`` (Morton, Taylor & Turner 1956 scaling), grow and
  fade over their lifetime. A puff's phase and lateral drift are ``hash_uniform(seed, "render.smoke" or
  "render.steam", source, puff, axis)``: the same frame time always draws the same puffs, and a puff moves
  continuously from frame to frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.rng import hash_u64, hash_uniform, stream_key
from . import colour as C
from . import params as P
from .scene import SPRITE_ADD, SPRITE_ALPHA, Sprites

WATER_BASED = ("emulsion", "semi_synthetic", "synthetic", "solution", "water")
OIL_BEARING = ("straight_oil", "oil", "mql", "emulsion", "semi_synthetic")
DRY = ("dry", "air", "none", None)

# puff opacity envelope: rises over the first quarter of the puff's life, then falls as (1 - age/life)^2
# (dilution); a display shape, documented in docs/models/render.md
FADE_IN_FRACTION = 0.25

STREAM_SMOKE = "render.smoke"
STREAM_STEAM = "render.steam"
STREAM_CHIPS = "render.chip_orientation"
STREAM_SPOT = "render.hot_spot"


@dataclass
class Thresholds:
    glow_K: float
    smoke_K: float | None          # None: the fluid does not smoke
    boil_K: float | None           # None: no water phase


def thresholds(coolant: dict) -> Thresholds:
    """Cue thresholds of the run's coolant (database values first, ``render-cues`` fall-backs)."""
    p = P.cues()
    kind = (coolant.get("kind") or "").lower() or None
    smoke = coolant.get("smoke_point_K")
    boil = coolant.get("boiling_K")
    if kind in DRY:
        return Thresholds(float(p["draper_point_K"]), None, None)
    if smoke is None and kind in OIL_BEARING:
        smoke = float(p["oil_smoke_point_K"])
    if kind not in WATER_BASED:
        boil = None
    elif boil is None:
        boil = float(p["water_boiling_K"])
    return Thresholds(float(p["draper_point_K"]), None if smoke is None else float(smoke),
                      None if boil is None else float(boil))


def density(temp_K, threshold_K: float | None) -> np.ndarray:
    """Smoke/steam density in [0, 1] of hot spots at ``temp_K`` for one threshold."""
    t = np.atleast_1d(np.asarray(temp_K, dtype=np.float64))
    if threshold_K is None:
        return np.zeros(t.shape)
    full = float(P.number(P.CUES, "cue_full_scale_dK"))
    return np.clip(np.nan_to_num((t - threshold_K) / full, nan=0.0), 0.0, 1.0)


@dataclass
class HotSpot:
    key: int                       # stable source id (puffs follow it from frame to frame)
    pos_S_m: np.ndarray
    temp_K: float


def puffs(spots: list[HotSpot], time_s: float, threshold_K: float | None, kind: str, seed: int,
          up_S: np.ndarray) -> Sprites | None:
    """Smoke (kind "smoke") or steam ("steam") puffs of the hot spots at ``time_s``."""
    if threshold_K is None or not spots:
        return None
    p = P.cues()
    rise = float(p[f"{kind}_rise_m_s"])
    life = float(p[f"{kind}_lifetime_s"])
    r0, r1 = (float(v) for v in p[f"{kind}_radius_m"])
    opacity = float(p[f"{kind}_opacity"])
    rgb = C.srgb_to_linear(np.asarray(p[f"{kind}_srgb"], dtype=np.float64))
    per_source = int(round(float(p["puffs_per_source"])))
    stream = stream_key(STREAM_SMOKE if kind == "smoke" else STREAM_STEAM)
    up = np.asarray(up_S, dtype=np.float64)
    up = up / float(np.linalg.norm(up))
    side_a = np.cross(up, [1.0, 0.0, 0.0] if abs(up[0]) < 0.9 else [0.0, 1.0, 0.0])
    side_a /= float(np.linalg.norm(side_a))
    side_b = np.cross(up, side_a)
    pos, radius, alpha = [], [], []
    for spot in spots:
        strength = float(density(spot.temp_K, threshold_K)[0])
        count = int(math.ceil(strength * per_source))
        for i in range(count):
            phase = hash_uniform(seed, stream, spot.key, i, 0)
            age = ((time_s / life + phase) % 1.0) * life
            frac = age / life
            drift_a = (hash_uniform(seed, stream, spot.key, i, 1) - 0.5) * 2.0 * r1 * frac
            drift_b = (hash_uniform(seed, stream, spot.key, i, 2) - 0.5) * 2.0 * r1 * frac
            pos.append(spot.pos_S_m + up * (rise * age + r0) + side_a * drift_a + side_b * drift_b)
            radius.append(r0 + (r1 - r0) * frac)
            alpha.append(opacity * strength * min(1.0, frac / FADE_IN_FRACTION) * (1.0 - frac) ** 2)
    if not pos:
        return None
    pos = np.asarray(pos)
    return Sprites(pos, pos, np.asarray(radius), rgb, np.asarray(alpha), SPRITE_ALPHA, name=kind)


def sparks(pos_S: np.ndarray, vel_S: np.ndarray | None, temp_K: np.ndarray) -> Sprites | None:
    """Additive streaks for the flying chips above the Draper point."""
    temp = np.asarray(temp_K, dtype=np.float64).reshape(-1)
    hot = np.flatnonzero(temp > float(P.number(P.CUES, "draper_point_K")))
    if not len(hot):
        return None
    p = P.cues()
    exposure = float(p["spark_exposure_s"])
    radius = float(p["spark_radius_m"])
    pos = np.asarray(pos_S, dtype=np.float64).reshape(-1, 3)[hot]
    if vel_S is not None:
        vel = np.asarray(vel_S, dtype=np.float64).reshape(-1, 3)[hot]
        tail = pos - vel * exposure
    else:
        tail = pos.copy()
    rgb = C.glow_rgb(temp[hot])
    level = np.maximum(rgb.max(axis=1), 1e-12)
    return Sprites(pos, tail, radius, rgb / level[:, None], level, SPRITE_ADD, name="sparks")


def emission(temp_K) -> np.ndarray:
    """Linear emitted light [n, 3] of surfaces at ``temp_K`` (zero below the Draper point)."""
    return C.glow_rgb(temp_K)


def source_key(seed: int, point_S_m) -> int:
    """Stable id of a hot spot: a hash of the ``hot_spot_cell_m`` cell it lies in, so the same place keeps its
    puffs from one frame (and keyframe) to the next."""
    cell = float(P.number(P.CUES, "hot_spot_cell_m"))
    i, j, k = (int(math.floor(float(c) / cell)) for c in np.asarray(point_S_m, dtype=np.float64).reshape(3))
    return int(hash_u64(seed, stream_key(STREAM_SPOT), i, j, k) >> np.uint64(33))


def chip_orientation_seeds(seed: int, keys: np.ndarray) -> np.ndarray:
    """Three uniforms per chip from its stable key (orientation of the drawn octahedron)."""
    stream = stream_key(STREAM_CHIPS)
    keys = np.asarray(keys, dtype=np.int64).reshape(-1)
    out = np.empty((len(keys), 3))
    for i, key in enumerate(keys):
        for a in range(3):
            out[i, a] = hash_uniform(seed, stream, int(key), a, 0)
    return out
