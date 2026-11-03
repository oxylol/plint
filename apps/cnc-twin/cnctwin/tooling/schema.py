"""Project settings of the tooling library: ``physics.edge_element_mm`` and ``physics.tooling``.

These are numerical settings (how finely the cutting edges are discretised and grouped), not
model coefficients; the defaults below are the documented starting values (docs/models/tooling.md
"Discretisation"). The project schema embeds :data:`PHYSICS_TOOLING` under ``physics.tooling``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from ..db.schema import Count, Qty, Struct

DEG = math.pi / 180.0

#: ``physics.tooling`` (keys in the file's units; ranges in SI as everywhere in cnctwin.db).
PHYSICS_TOOLING = Struct({
    "rake_class_deg": Qty(gt=0.0, hi=30 * DEG, doc="width of a rake class of the coefficient groups"),
    "incl_class_deg": Qty(gt=0.0, hi=45 * DEG, doc="width of an inclination class of the coefficient groups"),
    "max_kappa_step_deg": Qty(gt=0.0, hi=45 * DEG, doc="largest change of the edge normal across one element"),
    "max_lag_step_deg": Qty(gt=0.0, hi=45 * DEG, doc="largest change of the angular position across one element"),
    "margin_element_factor": Qty(gt=0.0, hi=100.0, dimensionless=True,
                                 doc="margin and pad elements are this many edge elements long"),
    "wave_elements": Count(lo=2, hi=1000, doc="elements per wavelength of a knuckle or serration profile"),
    "max_elements_per_tooth": Count(lo=10, hi=1000000, doc="refuse finer discretisations (memory guard)"),
}, doc="edge discretisation and coefficient grouping of the tooling library")

#: physics-level key read by tooling (declared in the project schema's physics section)
EDGE_ELEMENT_KEY = "edge_element_mm"

DEFAULTS = {
    "edge_element_m": 0.5e-3,
    "rake_class_rad": 2.0 * DEG,
    "incl_class_rad": 5.0 * DEG,
    "max_kappa_step_rad": 7.5 * DEG,
    "max_lag_step_rad": 5.0 * DEG,
    "margin_element_factor": 4.0,
    "wave_elements": 8,
    "max_elements_per_tooth": 20000,
}


@dataclass(frozen=True)
class ToolingSettings:
    edge_element_m: float = DEFAULTS["edge_element_m"]
    rake_class_rad: float = DEFAULTS["rake_class_rad"]
    incl_class_rad: float = DEFAULTS["incl_class_rad"]
    max_kappa_step_rad: float = DEFAULTS["max_kappa_step_rad"]
    max_lag_step_rad: float = DEFAULTS["max_lag_step_rad"]
    margin_element_factor: float = DEFAULTS["margin_element_factor"]
    wave_elements: int = DEFAULTS["wave_elements"]
    max_elements_per_tooth: int = DEFAULTS["max_elements_per_tooth"]


def settings_from(physics) -> ToolingSettings:
    """Settings from the project's ``physics`` section (SIMap or plain mapping in SI keys, or file
    keys with units; None gives the defaults)."""
    if physics is None:
        return ToolingSettings()
    values = dict(DEFAULTS)
    edge = _si(physics, "edge_element_m", "edge_element_mm", 1e-3)
    if edge is not None:
        values["edge_element_m"] = edge
    section = _get(physics, "tooling")
    if isinstance(section, Mapping):
        for si_name, file_name, factor in (("rake_class_rad", "rake_class_deg", DEG),
                                           ("incl_class_rad", "incl_class_deg", DEG),
                                           ("max_kappa_step_rad", "max_kappa_step_deg", DEG),
                                           ("max_lag_step_rad", "max_lag_step_deg", DEG),
                                           ("margin_element_factor", "margin_element_factor", 1.0)):
            value = _si(section, si_name, file_name, factor)
            if value is not None:
                values[si_name] = value
        for name in ("wave_elements", "max_elements_per_tooth"):
            value = _get(section, name)
            if value is not None:
                values[name] = int(value)
    if not values["edge_element_m"] > 0.0:
        raise ValueError("physics.edge_element_mm must be positive")
    return ToolingSettings(**values)


def _get(mapping, key):
    if mapping is None:
        return None
    if isinstance(mapping, Mapping):
        return mapping.get(key)
    return getattr(mapping, key, None)


def _si(mapping, si_name: str, file_name: str, factor: float):
    """The SI value whether the mapping is an SI view (``edge_element_m``) or the file mapping."""
    value = _get(mapping, si_name)
    if value is not None and si_name != file_name:
        return float(value)
    value = _get(mapping, file_name)
    if value is None:
        return None
    return float(value) * factor
