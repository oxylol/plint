"""ISO 286 limits of size from a tolerance class (``H7``, ``g6``, ``js5``, ``K7``) - docs/models/inspect.md
"Size limits".

* The standard tolerance IT of grades 1-18 comes from ISO 286-1:2010 Table 1 for the main size step of the
  nominal size (sizes up to 500 mm; IT14-IT18 not below 1 mm).
* Shafts (lower case): the fundamental deviation of positions a-g is the upper deviation es (ISO 286-2 tables,
  intermediate steps), h has es = 0, js is +-IT/2, k (IT4-IT7 from the table, else 0), m, n, p, r, s give the lower
  deviation ei; the other limit is the fundamental deviation -/+ IT.
* Holes (upper case), ISO 286-1 5.3.2: A-H: EI = -es of the shaft letter (general rule); JS +-IT/2;
  K, M, N up to IT8 and P-S up to IT7: ES = -ei + delta with delta = IT(n) - IT(n - 1) (special rule; delta = 0
  up to 3 mm); K above IT8: ES = 0; M above IT8: ES = -ei; N above IT8: ES = 0 (sizes above 3 mm); P-S above IT7:
  ES = -ei (general rule). The K rule up to IT8 uses the k value tabulated for IT4-IT7.

Positions j/J, t-zc / T-ZC, cd/ef/fg and IT01/IT0 are not encoded: they raise :class:`FitDesignationError` with
the reason, and the characteristic says so in inspection.json.
"""

from __future__ import annotations

import functools

import numpy as np

from ..core.errors import InputError
from ..db.models import params as model_params
from .specs import ISO_FIT

MODEL = "inspect-iso286"
SHAFT_UPPER = ("a", "b", "c", "d", "e", "f", "g")
SHAFT_LOWER = ("k", "m", "n", "p", "r", "s")


class FitDesignationError(InputError):
    """A tolerance class this module does not evaluate."""


@functools.lru_cache(maxsize=1)
def _tables() -> dict:
    p = model_params(MODEL)
    um = 1e6
    out = {"it_steps": np.round(np.asarray(p["it_steps_m"]) * 1e3, 6),
           "it": np.round(np.asarray(p["it_grades_m"]) * um, 6),
           "dev_steps": np.round(np.asarray(p["deviation_steps_m"]) * 1e3, 6)}
    for letter in SHAFT_UPPER:
        out["es_" + letter] = np.round(np.asarray(p[f"shaft_es_{letter}_m"]) * um, 6)
    for letter in SHAFT_LOWER:
        out["ei_" + letter] = np.round(np.asarray(p[f"shaft_ei_{letter}_m"]) * um, 6)
    return out


def _step(steps: np.ndarray, size_mm: float) -> int:
    """Index of the step (steps[i], steps[i + 1]] holding the size (the first step includes 0 < size <= 3)."""
    if not 0.0 < size_mm <= float(steps[-1]):
        raise FitDesignationError(f"ISO 286 limits are encoded for nominal sizes up to {steps[-1]:g} mm, "
                                  f"not {size_mm:g} mm")
    return int(np.searchsorted(steps, size_mm, side="left")) - 1


def it_um(grade: int, size_mm: float) -> float:
    """Standard tolerance IT<grade> in um for a nominal size in mm."""
    t = _tables()
    if not 1 <= grade <= 18:
        raise FitDesignationError(f"IT{grade}: grades IT1-IT18 are encoded (not IT01, IT0)")
    if grade >= 14 and size_mm <= 1.0:
        raise FitDesignationError(f"IT{grade} is not used for nominal sizes up to 1 mm (ISO 286-1 Table 1)")
    return float(t["it"][grade - 1, _step(t["it_steps"], size_mm)])


def _delta_um(grade: int, size_mm: float) -> float:
    if size_mm <= 3.0 or grade < 2:
        return 0.0
    return it_um(grade, size_mm) - it_um(grade - 1, size_mm)


def shaft_deviations_um(letter: str, grade: int, size_mm: float) -> tuple[float, float]:
    """(es, ei) in um of a shaft tolerance class."""
    t = _tables()
    it = it_um(grade, size_mm)
    if letter == "h":
        return 0.0, -it
    if letter == "js":
        return 0.5 * it, -0.5 * it
    i = _step(t["dev_steps"], size_mm)
    if letter in SHAFT_UPPER:
        es = float(t["es_" + letter][i])
        return es, es - it
    if letter in SHAFT_LOWER:
        ei = float(t["ei_" + letter][i])
        if letter == "k" and not 4 <= grade <= 7:
            ei = 0.0
        return ei + it, ei
    raise FitDesignationError(f"shaft position {letter!r} is not encoded (a-h, js, k, m, n, p, r, s are)")


def hole_deviations_um(letter: str, grade: int, size_mm: float) -> tuple[float, float]:
    """(ES, EI) in um of a hole tolerance class."""
    t = _tables()
    it = it_um(grade, size_mm)
    low = letter.lower()
    if letter == "H":
        return it, 0.0
    if letter == "JS":
        return 0.5 * it, -0.5 * it
    i = _step(t["dev_steps"], size_mm)
    if low in SHAFT_UPPER:
        ei = -float(t["es_" + low][i])
        return ei + it, ei
    if low not in SHAFT_LOWER:
        raise FitDesignationError(f"hole position {letter!r} is not encoded (A-H, JS, K, M, N, P, R, S are)")
    ei_shaft = float(t["ei_" + low][i])
    if low == "k":
        es = (-ei_shaft + _delta_um(grade, size_mm)) if grade <= 8 else 0.0
    elif low == "m":
        es = -ei_shaft + (_delta_um(grade, size_mm) if grade <= 8 else 0.0)
    elif low == "n":
        es = (-ei_shaft + _delta_um(grade, size_mm)) if (grade <= 8 or size_mm <= 3.0) else 0.0
    else:
        es = -ei_shaft + (_delta_um(grade, size_mm) if grade <= 7 else 0.0)
    return es, es - it


def parse(designation: str) -> tuple[str, int]:
    text = str(designation).strip()
    m = ISO_FIT.match(text)
    if m is None:
        raise FitDesignationError(f"{designation!r} is not an ISO 286 tolerance class (letters, then the grade)")
    letter, grade = m.group(1), int(m.group(2))
    if not (letter.islower() or letter.isupper()):
        raise FitDesignationError(f"{designation!r}: a tolerance class is all lower case (shaft) or upper (hole)")
    return letter, grade


def limits_m(nominal_m: float, designation: str) -> tuple[float, float, bool]:
    """(lower limit, upper limit, internal) in metres of a size with an ISO 286 tolerance class."""
    letter, grade = parse(designation)
    size_mm = round(float(nominal_m) * 1e3, 9)
    if letter.isupper():
        es, ei = hole_deviations_um(letter, grade, size_mm)
        internal = True
    else:
        es, ei = shaft_deviations_um(letter, grade, size_mm)
        internal = False
    return float(nominal_m) + ei * 1e-6, float(nominal_m) + es * 1e-6, internal


def clear_cache() -> None:
    _tables.cache_clear()


__all__ = ["limits_m", "parse", "it_um", "shaft_deviations_um", "hole_deviations_um", "FitDesignationError",
           "clear_cache"]
