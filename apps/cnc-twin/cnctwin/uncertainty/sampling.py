"""Sampling plans over the stated tolerances: Latin hypercube and simple Monte Carlo.

* **Latin hypercube** (McKay, Beckman & Conover 1979): for n samples each factor's probability axis is
  cut into n equal strata; every stratum gets exactly one sample, at a uniform position inside it,
  and the strata are paired across factors by independent random permutations. Marginals are exact
  stratified samples; the mean of a monotone response has a variance no larger than with random
  sampling (Stein 1987: it removes the variance of the additive main effects). Pairing is random,
  so two factors can correlate spuriously by about 1/sqrt(n) (Helton & Davis 2003); the sensitivity
  analysis reports partial rank correlations for that reason.
* **Monte Carlo** (JCGM 101:2008, 7): independent uniform draws per factor.

Draws are mapped to each factor's distribution by its inverse CDF. Every factor has its own seeded
stream (``cnctwin.core.rng.stream(seed, "uncertainty.<method>.<factor>")``): the plan of one factor
does not change when another factor is added or removed, and the same seed gives the same plan on
every platform (PCG64, SeedSequence). Unit draws lie strictly inside (0, 1) (a 53-bit draw k / 2^53
moved to (k + 1/2) / 2^53), so no inverse CDF is evaluated at 0 or 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri

from ..core.rng import stream

METHODS = ("lhs", "monte_carlo")
_TWO_POW_53 = 2.0 ** 53


@dataclass(frozen=True)
class Uniform:
    """Rectangular on [low, high]: the distribution of a quantity known only by its bounds (GUM 4.3.7)."""

    low: float
    high: float

    def ppf(self, u: np.ndarray) -> np.ndarray:
        return self.low + np.asarray(u, float) * (self.high - self.low)

    @property
    def mean(self) -> float:
        return 0.5 * (self.low + self.high)

    @property
    def std(self) -> float:
        return (self.high - self.low) / math.sqrt(12.0)

    def describe(self) -> dict:
        return {"kind": "uniform", "low": self.low, "high": self.high, "mean": self.mean, "std": self.std}


@dataclass(frozen=True)
class TruncatedNormal:
    """Normal (mean, sigma) restricted to [low, high]: a stated 1-sigma spread with physical bounds."""

    mean_value: float
    sigma: float
    low: float
    high: float

    def _bounds(self) -> tuple[float, float]:
        if self.sigma <= 0.0:
            return 0.0, 1.0
        return float(ndtr((self.low - self.mean_value) / self.sigma)), float(ndtr((self.high - self.mean_value) / self.sigma))

    def ppf(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, float)
        if self.sigma <= 0.0:
            return np.full(u.shape, self.mean_value)
        a, b = self._bounds()
        return np.clip(self.mean_value + self.sigma * ndtri(a + u * (b - a)), self.low, self.high)

    @property
    def mean(self) -> float:
        from scipy.stats import truncnorm
        if self.sigma <= 0.0:
            return self.mean_value
        a, b = (self.low - self.mean_value) / self.sigma, (self.high - self.mean_value) / self.sigma
        return float(truncnorm.mean(a, b, loc=self.mean_value, scale=self.sigma))

    @property
    def std(self) -> float:
        from scipy.stats import truncnorm
        if self.sigma <= 0.0:
            return 0.0
        a, b = (self.low - self.mean_value) / self.sigma, (self.high - self.mean_value) / self.sigma
        return float(truncnorm.std(a, b, loc=self.mean_value, scale=self.sigma))

    def describe(self) -> dict:
        return {"kind": "truncated_normal", "mean_param": self.mean_value, "sigma": self.sigma, "low": self.low,
                "high": self.high, "mean": self.mean, "std": self.std}


def open_unit(rng: np.random.Generator, n: int) -> np.ndarray:
    """n uniform draws strictly inside (0, 1)."""
    k = np.floor(rng.random(n) * _TWO_POW_53)
    return (k + 0.5) / _TWO_POW_53


def latin_hypercube(n: int, rng: np.random.Generator) -> np.ndarray:
    """One column of an LHS plan: exactly one draw in each stratum [i/n, (i+1)/n), in random order."""
    permutation = rng.permutation(n)
    return (permutation + open_unit(rng, n)) / n


def monte_carlo(n: int, rng: np.random.Generator) -> np.ndarray:
    return open_unit(rng, n)


def unit_column(method: str, n: int, seed: int, factor: str) -> np.ndarray:
    if method not in METHODS:
        raise ValueError(f"sampling method {method!r}: use lhs or monte_carlo")
    rng = stream(seed, f"uncertainty.{method}.{factor}")
    return latin_hypercube(n, rng) if method == "lhs" else monte_carlo(n, rng)


def plan(names: list[str], distributions: list, n: int, method: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(unit draws [n, k], values [n, k]) for factors ``names`` with their ``distributions``."""
    if n < 1:
        raise ValueError("an uncertainty study needs at least one run")
    units = np.zeros((n, len(names)))
    values = np.zeros((n, len(names)))
    for j, (name, distribution) in enumerate(zip(names, distributions)):
        units[:, j] = unit_column(method, n, seed, name)
        values[:, j] = distribution.ppf(units[:, j])
    return units, values


__all__ = ["Uniform", "TruncatedNormal", "latin_hypercube", "monte_carlo", "open_unit", "unit_column", "plan",
           "METHODS"]
