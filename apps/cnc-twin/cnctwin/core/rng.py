"""Deterministic randomness.

Every stochastic element (edge chipping, chip particle scatter, recut noise, BUE break-off,
Monte-Carlo draws) is seeded from the project seed, and nothing in the loop holds a stateful
generator (so a checkpoint never has to capture one):

* inside the simulation every draw is ``hash_uniform(seed, stream, a, b, c)``: a counter-based
  value that depends only on its arguments. ``stream`` is ``stream_key("<domain>.<purpose>")``
  (each domain lists its stream names in its model doc); ``a, b, c`` are counters the owner
  keeps in its own state (tooth, event count, tick ...), so results never depend on evaluation
  order and resume is exact.
* :func:`stream` (a PCG64 generator per name) is for initialisation and for outer studies
  (uncertainty sampling plans), never inside the macro loop.
"""

from __future__ import annotations

import zlib

import numpy as np
from numba import njit

_GOLDEN_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_MIX_1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX_2 = np.uint64(0x94D049BB133111EB)
_TWO_POW_MINUS_53 = 1.0 / 9007199254740992.0


def stream_key(name: str) -> int:
    """Stable 32-bit key of a stream name (CRC-32, identical on every platform)."""
    return zlib.crc32(name.encode("utf-8"))


def stream(seed: int, name: str) -> np.random.Generator:
    """Independent generator for initialisation or outer studies (not for use in the loop)."""
    sequence = np.random.SeedSequence([int(seed) & 0xFFFFFFFF, stream_key(name)])
    return np.random.Generator(np.random.PCG64(sequence))


@njit("uint64(uint64)", cache=True, nogil=True)
def splitmix64(x):
    """One SplitMix64 step (Steele, Lea & Flood 2014)."""
    z = x + _GOLDEN_GAMMA
    z = (z ^ (z >> np.uint64(30))) * _MIX_1
    z = (z ^ (z >> np.uint64(27))) * _MIX_2
    return z ^ (z >> np.uint64(31))


@njit("uint64(int64, int64, int64, int64, int64)", cache=True, nogil=True)
def hash_u64(seed, stream_id, a, b, c):
    """Counter-based 64-bit hash of (seed, stream, a, b, c)."""
    h = splitmix64(np.uint64(seed))
    h = splitmix64(h ^ np.uint64(stream_id))
    h = splitmix64(h ^ np.uint64(a))
    h = splitmix64(h ^ np.uint64(b))
    return splitmix64(h ^ np.uint64(c))


@njit("float64(int64, int64, int64, int64, int64)", cache=True, nogil=True)
def hash_uniform(seed, stream_id, a, b, c):
    """Uniform float in [0, 1) that depends only on its arguments."""
    return float(hash_u64(seed, stream_id, a, b, c) >> np.uint64(11)) * _TWO_POW_MINUS_53


@njit("float64(int64, int64, int64, int64, int64)", cache=True, nogil=True)
def hash_normal(seed, stream_id, a, b, c):
    """Standard normal draw from two counter-based uniforms (Box-Muller)."""
    u1 = hash_uniform(seed, stream_id, a, b, 2 * c)
    u2 = hash_uniform(seed, stream_id, a, b, 2 * c + 1)
    if u1 < 1e-300:
        u1 = 1e-300
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
