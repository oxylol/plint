"""``physics.cutting`` project settings (numerical settings of the cutting domain; docs/models/cutting.md
"Settings"). Model coefficients are in db/models/cutting-*.yaml, not here.

:data:`FIELDS` is the single declaration: :func:`settings` reads a project's ``physics.cutting`` (an SI
view, where ``h_hi_mm`` reads as ``h_hi_m``, or the raw mapping) into :class:`CuttingSettings` and
rejects out-of-range values; :func:`schema_node` builds the project-schema node (found by
cnctwin.project.schema as ``schema_node``). ``physics.memory_bins`` (top level) sets the surface-memory
resolution.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields

from ..core.errors import InputError

DEG = math.pi / 180.0

# name -> (file key, kind, default (SI), lo, hi, doc)
FIELDS: dict[str, tuple[str, str, float, float, float, str]] = {
    "n_lo": ("table_low_segments", "count", 16, 2, 4096,
             "uniform segments of the coefficient table between 0 and h_min (ploughing branch)"),
    "n_hi": ("table_high_segments", "count", 384, 8, 65536,
             "uniform segments of the coefficient table between h_min and h_hi (linear interpolation error "
             "~ 0.02 (dh / h)^2 for Kienzle-type rows)"),
    "h_hi_m": ("h_hi_mm", "length", 0.6e-3, 1e-5, 0.05, "top of the coefficient table (grown with the observed h)"),
    "h_grow": ("h_grow", "number", 1.5, 1.05, 10.0,
               "the table top is raised to this factor times the largest observed chip thickness"),
    "log_nodes": ("log_nodes", "count", 28, 8, 400, "nodes of the log-spaced h grid the coefficient model is evaluated on"),
    "h_log_lo_m": ("h_log_lo_mm", "length", 5e-8, 1e-10, 1e-5, "lowest node of the log-spaced h grid"),
    "tol_speed_rel": ("tol_speed_rel", "number", 0.02, 0.0, 1.0, "rebuild a coefficient row when its speed moved by this"),
    "tol_temp_K": ("tol_temp_dK", "number", 10.0, 0.0, 500.0, "... or its tool temperature by this"),
    "tol_work_temp_K": ("tol_work_temp_dK", "number", 5.0, 0.0, 500.0, "... or the workpiece preheat by this"),
    "tol_friction_rel": ("tol_friction_rel", "number", 0.02, 0.0, 1.0, "... or the coolant friction factor by this"),
    "tol_VB_m": ("tol_VB_um", "length", 2e-6, 0.0, 1e-3, "... or VB by this"),
    "tol_bue_m": ("tol_bue_um", "length", 1e-6, 0.0, 1e-3, "... or the BUE height by this"),
    "tol_edge_radius_m": ("tol_edge_radius_um", "length", 0.5e-6, 0.0, 1e-4, "... or the edge radius by this"),
    "event_capacity": ("event_capacity", "count", 4096, 16, 10 ** 7,
                       "rows per event buffer and macro step (doubled when a buffer overflowed)"),
    "tangent_vn": ("tangent_vn", "number", math.sin(5.0 * DEG), 0.0, 1.0,
                   "a pass boundary is on the generated surface when |v.n| is below this fraction of the feed speed"),
    "surface_event_rad": ("surface_event_deg", "angle", 10.0 * DEG, 0.1 * DEG, 360.0 * DEG,
                          "turning: work rotation between two surface events"),
    "h_cap_m": ("h_cap_mm", "length", 4e-3, 1e-5, 0.1, "turning: longest chip-thickness ray into the polar field"),
    "turn_colwell": ("turn_colwell", "count", 0, 0, 1,
                     "turning: 0 per-element forces, 1 Colwell's chord with the equivalent chip thickness"),
    "refill_margin_rad": ("refill_margin_deg", "angle", 15.0 * DEG, 0.0, 180.0 * DEG,
                          "L2 refill planes are computed for bins this far around the last engaged intervals"),
    "max_refill_queries": ("max_refill_queries", "count", 20000, 0, 10 ** 7,
                           "surface-point queries per contact and macro step for the refill planes"),
}


@dataclass(frozen=True)
class CuttingSettings:
    n_lo: int = 16
    n_hi: int = 384
    h_hi_m: float = 0.6e-3
    h_grow: float = 1.5
    log_nodes: int = 28
    h_log_lo_m: float = 5e-8
    tol_speed_rel: float = 0.02
    tol_temp_K: float = 10.0
    tol_work_temp_K: float = 5.0
    tol_friction_rel: float = 0.02
    tol_VB_m: float = 2e-6
    tol_bue_m: float = 1e-6
    tol_edge_radius_m: float = 0.5e-6
    event_capacity: int = 4096
    tangent_vn: float = math.sin(5.0 * DEG)
    surface_event_rad: float = 10.0 * DEG
    h_cap_m: float = 4e-3
    turn_colwell: int = 0
    refill_margin_rad: float = 15.0 * DEG
    max_refill_queries: int = 20000
    memory_bins: int = 720
    every_n_macro: int = 1


_FACTORS = {"length": {"_mm": 1e-3, "_um": 1e-6, "_m": 1.0}, "angle": {"_deg": DEG, "_rad": 1.0}}


def _lookup(section, name: str, file_key: str, kind: str):
    if section is None:
        return None
    for key in (name, file_key):
        value = section.get(key) if isinstance(section, Mapping) else getattr(section, key, None)
        if value is None:
            continue
        number = float(value)
        if key == file_key and kind in _FACTORS:
            for suffix, factor in _FACTORS[kind].items():
                if key.endswith(suffix):
                    number *= factor
                    break
        return number
    return None


def _get(mapping, key):
    if mapping is None:
        return None
    return mapping.get(key) if isinstance(mapping, Mapping) else getattr(mapping, key, None)


def settings(physics) -> CuttingSettings:
    """Settings from the project's ``physics`` section (its ``cutting`` sub-section and ``memory_bins``);
    None gives the defaults. Out-of-range values raise InputError."""
    section = _get(physics, "cutting")
    values = {}
    for item in fields(CuttingSettings):
        if item.name in ("memory_bins", "every_n_macro"):
            continue
        file_key, kind, default, lo, hi, _ = FIELDS[item.name]
        value = _lookup(section, item.name, file_key, kind)
        if value is None:
            value = default
        if not lo <= value <= hi:
            raise InputError(f"physics.cutting.{file_key} = {value} is outside [{lo}, {hi}] (SI)")
        values[item.name] = int(round(value)) if kind == "count" else float(value)
    bins = _get(physics, "memory_bins")
    values["memory_bins"] = int(bins) if bins is not None else 720
    if not 16 <= values["memory_bins"] <= 1 << 20:
        raise InputError(f"physics.memory_bins = {values['memory_bins']} is outside [16, 1048576]")
    every = _get(section, "every_n_macro") if section is not None else None
    values["every_n_macro"] = int(every) if every is not None else 1
    return CuttingSettings(**values)


def schema_node():
    """The ``physics.cutting`` node of the project schema (cnctwin.db.schema)."""
    from ..db.schema import Count, Qty, Struct
    nodes = {}
    for _, (file_key, kind, _default, lo, hi, doc) in FIELDS.items():
        if kind == "count":
            nodes[file_key] = Count(lo=int(lo), hi=int(hi), doc=doc)
        elif kind == "number":
            nodes[file_key] = Qty(lo=lo, hi=hi, dimensionless=True, doc=doc)
        else:
            nodes[file_key] = Qty(lo=lo, hi=hi, doc=doc)
    nodes["every_n_macro"] = Count(lo=1, hi=10 ** 6, doc="cadence of the coefficient pipeline (macro steps)")
    return Struct(nodes, doc="cutting settings (docs/models/cutting.md)")


def model_params() -> dict:
    """Coefficients of db/models/cutting-contact.yaml and cutting-spindle.yaml (SI)."""
    from ..db.models import params
    out = dict(params("cutting-contact"))
    out.update(params("cutting-spindle"))
    return out


__all__ = ["FIELDS", "CuttingSettings", "settings", "schema_node", "model_params"]
