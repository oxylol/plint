"""Harness registry and the data a harness returns (docs/validation.md "Harnesses").

A harness is ``function(case, db) -> HarnessOutput`` registered for one quantity. It simulates the
experiment of the case with the domain libraries at the level the experiment measured (never the
full run loop) and returns, for every reference value and every curve point of the case, a
:class:`Prediction` in the unit of that reference, plus the model chain, the settings, every input it
had to assume and the database values the model used (their sources decide whether the case is
calibration-dependent).

Prediction status:

* ``compared``: a simulated value; it is compared with the reference;
* ``no_prediction``: the model gives no value where the experiment measured one (e.g. no chip below
  the minimum chip thickness); it counts as a miss;
* ``not_compared``: the reference is outside what this harness validates (a duplicate of a counted
  value in another unit, a quantity of another harness, a point the case itself excludes); the reason
  is in the note and the value is listed but does not count.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable

COMPARED = "compared"
NO_PREDICTION = "no_prediction"
NOT_COMPARED = "not_compared"
STATUSES = (COMPARED, NO_PREDICTION, NOT_COMPARED)

SCALAR = "scalar"
CLASS = "class"


@dataclass
class Prediction:
    """Simulated counterpart of one reference value or curve point (in the reference's unit)."""

    value: float | str | None = None
    status: str = COMPARED
    note: str = ""
    kind: str = SCALAR                  # scalar | class (value and reference_class are labels)
    reference_class: str | None = None  # class of the reference (kind class): the harness maps the reference
    component: str | None = None        # budget component (budgets with components, e.g. cutting_force)
    uncertainty: float | None = None    # replaces the reference uncertainty (reference unit), e.g. propagated
    bound: str | None = None            # 'lower' / 'upper': the reference is a one-sided bound of the simulated
                                        # quantity (a stable test bounds a stability limit from below), else the
                                        # reference note decides ('lower bound' / 'upper bound')
    extra: dict = field(default_factory=dict)   # further simulated outputs shown in the report

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"unknown prediction status {self.status!r}")
        if self.status == COMPARED and self.value is None:
            raise ValueError("a compared prediction needs a value")
        if self.bound not in (None, "lower", "upper", "none"):
            raise ValueError(f"unknown bound {self.bound!r} (lower, upper, none)")


def no_prediction(note: str, **kw) -> Prediction:
    return Prediction(None, NO_PREDICTION, note, **kw)


def not_compared(note: str, value=None, **kw) -> Prediction:
    return Prediction(value, NOT_COMPARED, note, **kw)


@dataclass(frozen=True)
class Assumption:
    """An input the case does not state, with the value the harness used and where it comes from."""

    input: str
    value: str
    reason: str
    source: str = ""                    # database path / 'est' / 'case'


@dataclass(frozen=True)
class Provenance:
    """A database value the model chain used, with its source (bibliography key)."""

    what: str
    src: str
    note: str = ""


@dataclass
class HarnessOutput:
    values: list[Prediction]                       # aligned with case.reference_values
    curves: list[list[Prediction]]                 # aligned with case.reference_curves[i].points
    model_chain: list[str]
    settings: dict = field(default_factory=dict)
    assumptions: list[Assumption] = field(default_factory=list)
    provenance: list[Provenance] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    ambient_K: float | None = None                 # ambient of temperature references (rise above it)


@dataclass(frozen=True)
class Harness:
    quantity: str
    function: Callable
    models: tuple[str, ...]
    description: str


HARNESSES: dict[str, Harness] = {}
DISCUSSIONS: dict[str, Callable] = {}      # quantity -> function(QuantityResult) -> {"summary": text, case_id: text}
HARNESS_MODULES = ("orthogonal", "temperature", "taylor", "spindle", "coolant", "chipform", "cycletime",
                   "cutforce", "drilling", "roughness", "stability", "wallerror")
_LOADED = False


def discussion(quantity: str):
    """Decorator registering the report's 'failures and likely causes' text of a quantity:
    ``function(QuantityResult) -> {"summary": text, <case id>: text}`` (computed from the results)."""
    def wrap(function):
        DISCUSSIONS[quantity] = function
        return function
    return wrap


def harness(quantity: str, *, models: tuple[str, ...] = (), description: str = ""):
    """Decorator registering ``function(case, db) -> HarnessOutput`` for a quantity."""
    def wrap(function):
        register(quantity, function, models=models, description=description)
        return function
    return wrap


def register(quantity: str, function: Callable, *, models: tuple[str, ...] = (), description: str = "",
             replace: bool = False) -> None:
    if quantity in HARNESSES and not replace and HARNESSES[quantity].function is not function:
        raise ValueError(f"a harness for {quantity!r} is already registered "
                         f"({HARNESSES[quantity].function.__module__}); pass replace=True to override")
    HARNESSES[quantity] = Harness(quantity, function, tuple(models), description or (function.__doc__ or "").strip())


def load_harnesses() -> dict[str, Harness]:
    """Import every harness module of cnctwin.validation.harnesses (registration happens at import)."""
    global _LOADED
    if not _LOADED:
        for name in HARNESS_MODULES:
            importlib.import_module(f"{__package__}.harnesses.{name}")
        _LOADED = True
    return HARNESSES


def get(quantity: str) -> Harness | None:
    load_harnesses()
    return HARNESSES.get(quantity)


__all__ = ["Prediction", "Assumption", "Provenance", "HarnessOutput", "Harness", "HARNESSES", "DISCUSSIONS", "harness",
           "register", "discussion", "load_harnesses", "get", "no_prediction", "not_compared", "COMPARED",
           "NO_PREDICTION", "NOT_COMPARED", "SCALAR", "CLASS", "HARNESS_MODULES"]
