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
                           "edge field re-evaluated every n macro steps when edge_interval_ms is 0 (in between: last "
                           "local rise on the current body temperature)"),
    "edge_interval_s": ("edge_interval_ms", "time", 0.02, 0.0, 10.0,
                        "edge field re-evaluated at every multiple of this run time (the first macro step ending at or "
                        "after it; in between: last local rise on the current body temperature), so the evaluations "
                        "do not move with the macro step; 0: every edge_every_n_macro macro steps"),
    "edge_window_periods": ("edge_window_periods", "number", 4.0, 1.0, 1000.0,
                            "time constant of an edge element's running sums (tool heat, in-cut time, passes) in pass "
                            "periods: the periodic edge solution's flux, heating time and period come from them, not "
                            "from one macro step's counts"),
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
    "tool_field": ("tool_field", "count", 1, 0, 1,
                   "3-D field of the tool's cutting region solved with the ladder (1), or the ladder and the analytic "
                   "edge field only (0)"),
    "tool_field_ring_m": ("tool_field_ring_mm", "length", 2.0e-6, 1.0e-7, 1.0e-4,
                          "outer radius of the innermost ring of the cutting-region mesh (at most a quarter of the "
                          "smallest edge radius)"),
    "tool_field_growth": ("tool_field_growth", "number", 1.4, 1.05, 3.0,
                          "ratio of successive ring radii of the cutting-region mesh"),
    "tool_field_sector_deg": ("tool_field_sector_deg", "number", 15.0, 2.0, 45.0,
                              "largest sector of the cutting-region mesh over the wedge angle"),
    "tool_field_dy_m": ("tool_field_dy_mm", "length", 0.5e-3, 2.0e-5, 0.01,
                        "largest cell of the cutting-region mesh along the edge"),
    "tool_field_every_n_macro": ("tool_field_every_n_macro", "count", 10, 1, 1000,
                                 "tool ladders with a cutting-region field solved every n macro steps (heat "
                                 "accumulated exactly in between) when tool_field_interval_ms is 0"),
    "tool_field_interval_s": ("tool_field_interval_ms", "time", 0.1, 0.0, 100.0,
                              "tool ladders with a cutting-region field solved at every multiple of this run time (the "
                              "first macro step ending at or after it; heat accumulated exactly in between), so the "
                              "solves do not move with the macro step; 0: every tool_field_every_n_macro macro steps"),
    "tool_field_length_m": ("tool_field_length_mm", "length", 0.012, 1.0e-3, 0.2,
                            "length of a rotating tool's field along its edge from the first element"),
    "tool_field_tol": ("tool_field_tol", "number", 1.0e-12, 1.0e-15, 1.0e-6,
                       "relative residual of the cutting-region field's conjugate-gradient solve"),
    "tool_field_max_iter": ("tool_field_max_iter", "count", 5000, 10, 100000,
                            "iteration limit of the cutting-region field's solve"),
    "spindle_every_n_macro": ("spindle_every_n_macro", "count", 10, 1, 100000,
                              "spindle networks stepped every n macro steps (time, speeds and holder heat accumulated "
                              "exactly in between; their time constants are minutes)"),
}


@dataclass(frozen=True)
class ThermalSettings:
    every_n_macro: int = 5
    cell_m: float = 2.0e-3
    tool_cell_m: float = 1.0e-3
    edge_quadrature: int = 3
    edge_every_n_macro: int = 2
    edge_interval_s: float = 0.02
    edge_window_periods: float = 4.0
    sa_direct_cycles: int = 4
    window_points: int = 3
    jaeger_points: int = 24
    min_fraction: float = 1.0e-9
    deposit_search_cells: int = 3
    cooldown_steps: int = 24
    idle_tool_steps: int = 8
    energy_tol: float = 1.0e-9
    tool_field: int = 1
    tool_field_ring_m: float = 2.0e-6
    tool_field_growth: float = 1.4
    tool_field_sector_deg: float = 15.0
    tool_field_dy_m: float = 0.5e-3
    tool_field_every_n_macro: int = 10
    tool_field_interval_s: float = 0.1
    tool_field_length_m: float = 0.012
    tool_field_tol: float = 1.0e-12
    tool_field_max_iter: int = 5000
    spindle_every_n_macro: int = 10


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
        if key == file_key and file_key.endswith("_ms") and kind == "time":
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
