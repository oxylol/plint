"""Inspection settings: the numerical controls of cnctwin.inspect (docs/models/inspect.md "Settings").

The defaults, each with its source and the reason for its value, live in ``db/models/inspect.yaml`` (they are
numerical settings, not physics: like the geometry library's, they are read with
``cnctwin.db.models.params("inspect")``). A project may override them in a ``physics.inspect`` section once the
project schema routes it (:data:`PHYSICS_SCHEMA` declares it with the file keys and units); every function also
takes an :class:`InspectSettings` argument.

:func:`settings` reads a section (an SI view, a raw mapping with unit-suffixed keys, or None) and rejects values
outside the declared ranges.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass, fields

from ..core.errors import InputError
from ..db.models import params as model_params
from ..db.models import records as model_records

MODEL = "inspect"
COUNTS = ("max_points_per_face", "sections", "section_min_points", "ls_iterations", "lp_iterations",
          "zone_polygon_sides", "probe_points_plane", "probe_points_circle", "probe_levels",
          "probe_trace_iterations", "chordal_samples", "chordal_iterations")


@dataclass(frozen=True)
class InspectSettings:
    assign_band_m: float
    assign_angle_rad: float
    max_points_per_face: int
    sections: int
    section_min_points: int
    ls_iterations: int
    ls_step_tolerance: float
    lp_iterations: int
    lp_tolerance_m: float
    lp_trust_m: float
    zone_polygon_sides: int
    pair_angle_rad: float
    probe_radius_m: float
    probe_prehit_m: float
    probe_search_m: float
    probe_inset_m: float
    probe_points_plane: int
    probe_points_circle: int
    probe_levels: int
    probe_trace_iterations: int
    probe_trace_tolerance_m: float
    remaining_tolerance_m: float
    gouge_tolerance_m: float
    region_min_volume_m3: float
    column_merge_m: float
    chordal_samples: int
    chordal_iterations: int
    chordal_max_distance_m: float
    chordal_tolerance_m: float
    thread_match_rel: float

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def replace(self, **changes) -> "InspectSettings":
        values = self.to_dict()
        unknown = sorted(set(changes) - set(values))
        if unknown:
            raise InputError(f"inspection settings have no {', '.join(unknown)}")
        values.update(changes)
        return _checked(values)


@functools.lru_cache(maxsize=1)
def _defaults() -> dict:
    return dict(model_params(MODEL))


@functools.lru_cache(maxsize=1)
def _ranges() -> dict:
    """name -> (lo, hi) in SI (the file's range is written in the parameter's unit)."""
    from ..db import keyunits

    out = {}
    for name, entry in model_records(MODEL).items():
        rng = entry.get("range")
        if isinstance(rng, list) and len(rng) == 2:
            unit = keyunits.parse_unit_text(entry.get("unit"))
            out[name] = tuple(float(keyunits.convert(v, unit)) for v in rng)
    return out


def defaults() -> InspectSettings:
    return _checked(_defaults())


def _checked(values: dict) -> InspectSettings:
    out = {}
    for f in fields(InspectSettings):
        if f.name not in values:
            raise InputError(f"db/models/{MODEL}.yaml has no parameter {f.name}")
        number = float(values[f.name])
        lo, hi = _ranges().get(f.name, (-float("inf"), float("inf")))
        if not lo <= number <= hi:
            raise InputError(f"inspection setting {f.name} = {number:g} (SI) is outside [{lo:g}, {hi:g}]")
        out[f.name] = int(round(number)) if f.name in COUNTS else number
    return InspectSettings(**out)


def _get(section, key):
    if section is None:
        return None
    if isinstance(section, Mapping):
        return section.get(key)
    return getattr(section, key, None)


def _file_key(name: str) -> tuple[str, float]:
    """(file key, factor file unit -> SI) of a setting, by the unit the defaults file writes it in."""
    entry = model_records(MODEL)[name]
    unit = str(entry.get("unit") or "-")
    if unit in ("-", ""):
        return name, 1.0
    from ..db import keyunits

    parsed = keyunits.parse_unit_text(unit)
    stem = name[: -len("_" + parsed.si_token)] if name.endswith("_" + parsed.si_token) else name
    return f"{stem}_{parsed.token}", float(parsed.factor)


def settings(section=None) -> InspectSettings:
    """Settings from a ``physics.inspect``-like section (None: the defaults). Keys may be the SI names
    (``probe_radius_m``) or the file keys (``probe_radius_mm``)."""
    values = dict(_defaults())
    for f in fields(InspectSettings):
        found = _get(section, f.name)
        if found is None:
            key, factor = _file_key(f.name)
            raw = _get(section, key)
            raw = getattr(raw, "value", raw)
            found = float(raw) * factor if raw is not None else None
        found = getattr(found, "value", found)
        if found is not None:
            values[f.name] = float(found)
    return _checked(values)


def _schema():
    from ..db.schema import Count, Qty, Struct

    nodes = {}
    for f in fields(InspectSettings):
        key, _ = _file_key(f.name)
        lo, hi = _ranges().get(f.name, (None, None))
        entry = model_records(MODEL)[f.name]
        rng = entry.get("range") or [None, None]
        if f.name in COUNTS:
            nodes[key] = Count(lo=int(rng[0]) if rng[0] is not None else 0,
                               hi=int(rng[1]) if rng[1] is not None else None, nullable=True,
                               doc=str(entry.get("note", "")))
        else:
            nodes[key] = Qty(lo=rng[0], hi=rng[1], nullable=True, doc=str(entry.get("note", "")))
    return Struct(nodes, meta=False, nullable=True,
                  doc="numerical settings of the result inspection (defaults: db/models/inspect.yaml)")


def physics_schema():
    """The :mod:`cnctwin.db.schema` node of a ``physics.inspect`` section (built on first use)."""
    return _schema()


def clear_cache() -> None:
    _defaults.cache_clear()
    _ranges.cache_clear()


__all__ = ["InspectSettings", "settings", "defaults", "physics_schema", "clear_cache", "MODEL"]
