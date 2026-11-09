"""Designs of a sweep: which parameter values each point gets (docs/formats.md "Sweep", docs/ml.md "Designs").

* ``grid``: the full factorial of every parameter's values (``values``, or ``levels`` evenly spaced over the range,
  log-spaced with ``scale: log``), in parameter order with the last parameter varying fastest;
* ``lhs``: a Latin hypercube (McKay, Beckman & Conover 1979) of ``points`` (scipy.stats.qmc.LatinHypercube,
  ``optimization: random-cd`` improves the centred discrepancy, ``lloyd`` spreads the points by Lloyd-Max);
* ``sobol``: a scrambled Sobol' sequence (Owen scrambling; scipy.stats.qmc.Sobol): ``random_base2`` when
  ``points`` is a power of two (the balance properties of the sequence hold only then; otherwise the first
  ``points`` points and a note);
* ``random``: independent uniform draws;
* ``list``: the explicit points, a parameter a point omits taking its ``default``.

The stochastic designs draw from ``cnctwin.core.rng.stream(seed, "sweep.<design>")`` (PCG64 with SeedSequence):
the same spec and seed give the same points on every platform for a given SciPy. A unit draw ``u`` in [0, 1)
becomes a value by the parameter's inverse map: ``lo + u (hi - lo)`` (linear), ``lo (hi/lo)^u`` (log), then
``step`` (a multiple of the step above lo, the last one at most hi), then ``integer`` (rounded half to even); a
categorical parameter takes ``values[floor(u n)]``.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from ..core.errors import InputError
from .spec import Parameter, SweepSpec


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def unit_design(kind: str, n: int, d: int, seed: int, *, optimization: str = "none", scramble: bool = True):
    """(units [n, d] in [0, 1), notes)."""
    from scipy.stats import qmc

    from ..core.rng import stream
    rng = stream(seed, f"sweep.{kind}")
    notes: list[str] = []
    if kind == "lhs":
        sampler = qmc.LatinHypercube(d=d, optimization=None if optimization == "none" else optimization, rng=rng)
        units = sampler.random(n)
    elif kind == "sobol":
        sampler = qmc.Sobol(d=d, scramble=scramble, rng=rng)
        if _is_power_of_two(n):
            units = sampler.random_base2(int(round(math.log2(n))))
        else:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)      # the note below says it
                units = sampler.random(n)
            notes.append(f"{n} Sobol' points: the balance properties hold for powers of two only "
                         f"({2 ** int(math.floor(math.log2(n)))} or {2 ** int(math.ceil(math.log2(n)))})")
    elif kind == "random":
        units = rng.random((n, d))
    else:
        raise InputError(f"no unit design {kind!r}")
    units = np.asarray(units, dtype=np.float64).reshape(n, d)
    return np.clip(units, 0.0, np.nextafter(1.0, 0.0)), notes


def value_of(p: Parameter, u: float):
    """The value of a unit draw ``u`` in [0, 1)."""
    if p.categorical:
        n = len(p.values)
        return p.values[min(int(math.floor(u * n)), n - 1)]
    lo, hi = p.range
    if p.scale == "log":
        x = lo * (hi / lo) ** u
    else:
        x = lo + u * (hi - lo)
    return _quantised(p, x)


def _quantised(p: Parameter, x: float):
    lo, hi = p.range
    if p.step is not None:
        k = math.floor((x - lo) / p.step + 0.5)
        k_max = math.floor((hi - lo) / p.step + 1e-9)
        x = lo + min(max(k, 0), k_max) * p.step
        x = round(x, 12)
    if p.integer:
        return int(round(x))
    return float(min(max(x, lo), hi))


def levels_of(p: Parameter) -> list:
    """The grid values of a parameter: its values, or ``levels`` points over the range (ends included)."""
    if p.categorical:
        return list(p.values)
    lo, hi = p.range
    n = int(p.levels)
    if p.scale == "log":
        raw = [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)]
    else:
        raw = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    raw[0], raw[-1] = lo, hi
    out = []
    for x in raw:
        value = _quantised(p, x)
        if value not in out:
            out.append(value)
    return out


def design_points(spec: SweepSpec) -> tuple[list[dict], np.ndarray | None, list[str]]:
    """([{parameter: value}] per point in design order, the unit draws [n, d] (None for grid and list), notes)."""
    params = spec.parameters
    kind = spec.design.kind
    if kind == "grid":
        axes = [levels_of(p) for p in params]
        points = [dict(zip([p.name for p in params], combo)) for combo in itertools.product(*axes)]
        return points, None, []
    if kind == "list":
        points = []
        for entry in spec.design.list:
            points.append({p.name: entry.get(p.name, p.default) for p in params})
        return points, None, []
    units, notes = unit_design(kind, int(spec.design.points), len(params), spec.seed,
                               optimization=spec.design.optimization, scramble=spec.design.scramble)
    points = [{p.name: value_of(p, float(units[i, j])) for j, p in enumerate(params)} for i in range(units.shape[0])]
    return points, units, notes


__all__ = ["design_points", "unit_design", "value_of", "levels_of"]
