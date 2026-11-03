"""``physics.thermal`` project settings (numerical settings of the thermal solvers).

The declaration :data:`FIELDS` is the single source: :func:`settings` reads a project's
``physics.thermal`` (an SI view, where ``cell_mm`` reads as ``cell_m``, or the raw mapping) into
:class:`ThermalSettings` and rejects out-of-range values; :func:`schema_node` builds the
:mod:`cnctwin.db.schema` node for the project schema (imported lazily).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields

from ..core.errors import InputError

# name -> (file key, kind, default (SI), lo, hi, doc)
FIELDS: dict[str, tuple[str, str, float, float, float, str]] = {
    "every_n_macro": ("every_n_macro", "count", 5, 1, 10000,
                      "workpiece finite-volume step every n macro steps (energy accumulated in between)"),
    "cell_m": ("cell_mm", "length", 2.0e-3, 1.0e-4, 0.05,
               "workpiece grid cell, co-registered with the stock lattice"),
    "tool_cell_m": ("tool_cell_mm", "length", 1.0e-3, 1.0e-4, 0.02, "axial cell of the tool and holder ladder"),
    "edge_quadrature": ("edge_quadrature", "count", 3, 1, 16,
                        "Gauss points over the in-cut interval for the edge temperatures (ToolHeat.q_weights)"),
    "edge_every_n_macro": ("edge_every_n_macro", "count", 2, 1, 1000,
                           "edge field re-evaluated every n macro steps (in between: last local rise on the current "
                           "body temperature)"),
    "sa_direct_cycles": ("sa_direct_cycles", "count", 4, 2, 512,
                         "cycles summed term by term in the periodic edge solution (the rest as an integral)"),
    "window_points": ("window_points", "count", 3, 2, 32, "Gauss points of the tail windows of the edge solution"),
    "jaeger_points": ("jaeger_points", "count", 24, 4, 128, "Gauss points per piece of the moving-source integral"),
    "min_fraction": ("min_fraction", "number", 1.0e-9, 0.0, 0.01, "cells at or below this fraction are empty"),
    "deposit_search_cells": ("deposit_search_cells", "count", 3, 1, 32,
                             "how far (cells) a heat deposit searches for remaining material"),
    "cooldown_steps": ("cooldown_steps", "count", 24, 1, 10000,
                       "implicit steps for the cool-down between operations and after unload"),
    "idle_tool_steps": ("idle_tool_steps", "count", 8, 1, 10000,
                        "implicit steps when an idle tool (magazine) is brought up to date"),
    "energy_tol": ("energy_tol", "number", 1.0e-9, 1.0e-15, 1.0e-3,
                   "relative tolerance of each solver's energy ledger (a violation fails the run)"),
}


@dataclass(frozen=True)
class ThermalSettings:
    every_n_macro: int = 5
    cell_m: float = 2.0e-3
    tool_cell_m: float = 1.0e-3
    edge_quadrature: int = 3
    edge_every_n_macro: int = 2
    sa_direct_cycles: int = 4
    window_points: int = 3
    jaeger_points: int = 24
    min_fraction: float = 1.0e-9
    deposit_search_cells: int = 3
    cooldown_steps: int = 24
    idle_tool_steps: int = 8
    energy_tol: float = 1.0e-9


def _lookup(section, name: str, file_key: str, kind: str):
    """Value of a field from an SI view (attribute/SI key) or a raw mapping (file key, converted)."""
    if section is None:
        return None
    candidates = [name, file_key]
    for key in candidates:
        value = None
        if isinstance(section, Mapping):
            value = section.get(key)
        else:
            value = getattr(section, key, None)
        if value is None:
            continue
        number = float(value)
        if key == file_key and file_key.endswith("_mm") and kind == "length":
            number *= 1e-3
        return number
    return None


def settings(section) -> ThermalSettings:
    """Settings from ``physics.thermal`` (None: defaults). Out-of-range values raise InputError."""
    values = {}
    for item in fields(ThermalSettings):
        file_key, kind, default, lo, hi, _ = FIELDS[item.name]
        value = _lookup(section, item.name, file_key, kind)
        if value is None:
            value = default
        if not lo <= value <= hi:
            raise InputError(f"physics.thermal.{file_key} = {value} is outside [{lo}, {hi}] (SI)")
        values[item.name] = int(round(value)) if kind == "count" else float(value)
    return ThermalSettings(**values)


def schema_node():
    """The ``physics.thermal`` node of the project schema (cnctwin.db.schema)."""
    from ..db.schema import Count, Qty, Struct
    nodes = {}
    for _, (file_key, kind, _default, lo, hi, doc) in FIELDS.items():
        if kind == "count":
            nodes[file_key] = Count(lo=int(lo), hi=int(hi), doc=doc)
        else:
            nodes[file_key] = Qty(lo=lo, hi=hi, doc=doc)
    return Struct(nodes, doc="thermal solver settings (docs/models/thermal.md)")
