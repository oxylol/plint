"""``physics.chips`` project settings: the numerical settings of the chips domain (model coefficients
live in db/models/chips-*.yaml).

:data:`FIELDS` is the single declaration: :func:`settings` reads a project's ``physics`` section (the SI
view, where ``pile_cell_mm`` reads as ``pile_cell_m``, or the raw mapping) into :class:`ChipsSettings`
and rejects values out of range; :data:`PHYSICS_SCHEMA` is the :mod:`cnctwin.db.schema` node the
project schema checks ``physics.chips`` with.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields

from ..core.errors import InputError

# name (SI) -> (file key, kind, default (SI), lo, hi, doc)
FIELDS: dict[str, tuple[str, str, float, float, float, str]] = {
    "every_n_macro": ("every_n_macro", "count", 1, 1, 1,
                      "chips run every macro step (they integrate the window of the motion record they read)"),
    "particles_per_tooth": ("particles_per_tooth", "number", 0.1, 1e-4, 1.0,
                            "representative particles per emission row (tooth pass); default physics.particles_per_tooth"),
    "max_particles": ("max_particles", "count", 50000, 100, 10 ** 7,
                      "capacity of the particle set; above it new parcels merge into the previous one of the step"),
    "pile_cell_m": ("pile_cell_mm", "length", 2.0e-3, 2.0e-4, 0.02,
                    "cell of the pile height field and of the occupancy grid (also the recut resolution)"),
    "pile_margin_m": ("pile_margin_mm", "length", 0.05, 0.0, 2.0,
                      "the pile grid covers the stock footprint plus this margin on every side"),
    "dt_max_s": ("dt_max_ms", "time", 1.0e-3, 1.0e-6, 0.01, "largest particle sub-step"),
    "drag_dt_factor": ("drag_dt_factor", "number", 0.2, 0.01, 10.0,
                       "sub-step <= factor / drag rate: the frozen-drag exponential integrator errs by about "
                       "factor^2 / 2 in the velocity transient (0.2: 2 %), never at the terminal state"),
    "contact_offset_m": ("contact_offset_um", "length", 1.0e-6, 1.0e-9, 1.0e-4,
                         "a particle leaves a contact this far outside the surface"),
    "max_bounces": ("max_bounces", "count", 400, 1, 10 ** 6,
                    "contacts after which a particle that has not settled is laid down where it is"),
    "max_zero_hits": ("max_zero_hits", "count", 8, 1, 10 ** 4,
                      "consecutive contacts without time advance after which a particle is laid down"),
    "relax_max_iter": ("relax_max_iter", "count", 50, 1, 10 ** 5, "avalanche sweeps per macro step"),
    "relax_tol_fraction": ("relax_tol_fraction", "number", 1.0e-3, 1.0e-12, 1.0,
                           "avalanching stops when less than this fraction of one cell volume moved in a sweep"),
    "emission_capacity": ("emission_capacity", "count", 4096, 16, 10 ** 7,
                          "rows of the chips EventBuffer of the hot loop (doubled after an overflow)"),
    "mobilise_capacity": ("mobilise_capacity", "count", 16384, 16, 10 ** 7,
                          "rows of the flush/sweep/recut buffers per macro step"),
    "event_capacity": ("event_capacity", "count", 4096, 16, 10 ** 7, "rows of the particle event buffer"),
    "rotating_part_rad_s": ("rotating_part_rad_s", "number", 1.0, 0.0, 1.0e4,
                            "a workpiece spindle turning faster than this sheds its chips and holds no pile"),
    "slow_lag_steps": ("slow_lag_steps", "number", 2.0, 0.5, 100.0,
                       "flying particles slower than cell / (this x macro step) join the frozen occupancy grid"),
    "ring_cells": ("ring_cells", "count", 2, 1, 50,
                   "the confinement ring of a tool lies this many cells outside its radius"),
    "enclosure_margin_m": ("enclosure_margin_mm", "length", 1.0, 0.0, 20.0,
                           "enclosure around the stock when the machine file gives none (enclosure.box_mm)"),
    "formation_bin_rel": ("formation_bin_rel", "number", 0.02, 1e-4, 0.5,
                          "relative bin of h, speed, width and length in which the mechanics chip formation is "
                          "solved once"),
    "formation_bin_rad": ("formation_bin_rad", "number", 0.01, 1e-5, 0.2,
                          "bin of rake and inclination (rad) of the mechanics chip formation"),
}


@dataclass(frozen=True)
class ChipsSettings:
    every_n_macro: int = 1
    particles_per_tooth: float = 0.1
    max_particles: int = 50000
    pile_cell_m: float = 2.0e-3
    pile_margin_m: float = 0.05
    dt_max_s: float = 1.0e-3
    drag_dt_factor: float = 0.2
    contact_offset_m: float = 1.0e-6
    max_bounces: int = 400
    max_zero_hits: int = 8
    relax_max_iter: int = 50
    relax_tol_fraction: float = 1.0e-3
    emission_capacity: int = 4096
    mobilise_capacity: int = 16384
    event_capacity: int = 4096
    rotating_part_rad_s: float = 1.0
    slow_lag_steps: float = 2.0
    ring_cells: int = 2
    enclosure_margin_m: float = 1.0
    formation_bin_rel: float = 0.02
    formation_bin_rad: float = 0.01


_SCALE = {"length": {"_mm": 1e-3, "_um": 1e-6, "_m": 1.0}, "time": {"_ms": 1e-3, "_s": 1.0}}


def _get(section, key):
    if section is None:
        return None
    if isinstance(section, Mapping):
        return section.get(key)
    return getattr(section, key, None)


def _lookup(section, name: str, file_key: str, kind: str):
    """Value of a field from an SI view (SI key) or a raw mapping (file key, converted)."""
    for key in (name, file_key):
        raw = _get(section, key)
        if raw is None:
            continue
        number = float(raw)
        if key == file_key and kind in _SCALE:
            for suffix, factor in _SCALE[kind].items():
                if key.endswith(suffix):
                    number *= factor
                    break
        return number
    return None


def settings(physics) -> ChipsSettings:
    """Settings from a project's ``physics`` section (its ``chips`` mapping; ``particles_per_tooth``
    falls back to ``physics.particles_per_tooth``). None gives the defaults; out-of-range values raise."""
    section = _get(physics, "chips")
    values = {}
    for item in fields(ChipsSettings):
        file_key, kind, default, lo, hi, _ = FIELDS[item.name]
        found = _lookup(section, item.name, file_key, kind)
        if found is None and item.name == "particles_per_tooth":
            found = _lookup(physics, item.name, file_key, kind)
        if found is None:
            found = default
        if not lo <= found <= hi:
            raise InputError(f"physics.chips.{file_key} = {found:g} (SI) is outside [{lo:g}, {hi:g}]")
        values[item.name] = int(round(found)) if kind == "count" else float(found)
    return ChipsSettings(**values)


def _schema():
    from ..db.schema import Count, Qty, Struct
    nodes = {}
    for _, (file_key, kind, _default, lo, hi, doc) in FIELDS.items():
        nodes[file_key] = Count(lo=int(lo), hi=int(hi), doc=doc) if kind == "count" else Qty(lo=lo, hi=hi, doc=doc)
    return Struct(nodes, doc="chips settings (docs/models/chips.md)", meta=False)


PHYSICS_SCHEMA = _schema()
