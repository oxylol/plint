"""``physics.removal`` project settings: the numerical settings of the removal domain (the hazard criteria
are model parameters in db/models/removal.yaml; the lattice, refine boxes and polar theta bins are the
top-level ``physics.lattice_mm``, ``physics.refine`` and ``physics.polar_theta_bins``).

:data:`FIELDS` is the single declaration: :func:`settings` reads a project's ``physics`` section (an SI view,
where ``gouge_tolerance_um`` reads as ``gouge_tolerance_m``, or the raw mapping) into
:class:`RemovalSettings` and rejects values out of range; :data:`PHYSICS_SCHEMA` is the node the project
schema checks ``physics.removal`` with.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields

from ..core.errors import InputError

DEG = math.pi / 180.0

# name (SI) -> (file key, kind, default (SI), lo, hi, doc)
FIELDS: dict[str, tuple[str, str, float, float, float, str]] = {
    "every_n_macro": ("every_n_macro", "count", 1, 1, 1, "removal absorbs and books every macro step"),
    "margin_nodes": ("margin_nodes", "count", 1, 1, 8, "empty node layers around the stock (closed meshes)"),
    "gouge_tolerance_m": ("gouge_tolerance_um", "length", 5.0e-6, 1.0e-8, 1.0e-3,
                          "removed target material deeper than this along a ray is a gouge (covers the CAD "
                          "tessellation chord error and the 10 nm quantum)"),
    "collision_tolerance_m": ("collision_tolerance_um", "length", 1.0e-5, 1.0e-8, 1.0e-3,
                              "overlap of a non-cutting part or a mesh with material beyond this is a collision"),
    "gate_sample_cells": ("gate_sample_cells", "number", 1.0, 0.1, 8.0,
                          "gating samples along a station circle, spaced this many lattice cells"),
    "gate_tolerance_m": ("gate_tolerance_nm", "length", 1.0e-8, 1.0e-10, 1.0e-5,
                         "engaged-interval boundaries are located to this distance along the circle"),
    "gate_margin_m": ("gate_margin_um", "length", 0.0, 0.0, 1.0e-3,
                      "the gated circle is grown radially by this much (0: the element circle itself)"),
    "sector_margin_cells": ("sector_margin_cells", "number", 1.0, 0.0, 16.0,
                            "the engaged-sector box of a subtraction is grown by this many lattice cells"),
    "pool_headroom": ("pool_headroom", "number", 2.0, 1.0, 64.0,
                      "free overflow blocks kept before a macro step: this times the most used in one step"),
    "pool_min_free": ("pool_min_free", "count", 1024, 16, 10 ** 8, "free overflow blocks kept at least"),
    "event_capacity": ("event_capacity", "count", 1024, 16, 10 ** 7,
                       "rows of the removal EventBuffer of the hot loop (doubled after an overflow)"),
    "sweep_step_cells": ("sweep_step_cells", "number", 0.5, 0.05, 8.0,
                         "swept collision checks sample motions so no point moves more than this many cells"),
    "connect_every_n_macro": ("connect_every_n_macro", "count", 1, 1, 10 ** 6,
                              "connectivity passes at most every this many macro steps (when topology changed)"),
    "polar_min_feed_m_per_rev": ("polar_min_feed_mm_rev", "length", 5.0e-5, 1.0e-7, 1.0e-2,
                                 "smallest feed per revolution the polar rows resolve (dz <= f_min / 8)"),
    "polar_max_bytes": ("polar_max_mb", "megabytes", 150e6, 1e6, 4e9,
                        "the polar field's rows are coarsened to stay within this memory (warning finding)"),
    "polar_kp": ("polar_kp", "count", 2, 1, 16, "inline segments per polar ray (grown on demand)"),
    "polar_axial_spacing_m": ("polar_axial_spacing_mm", "length", 0.0, 0.0, 5.0e-3,
                              "radial spacing of the polar axial rays (0: max(dz, lattice / 2))"),
    "polar_sync_revolutions": ("polar_sync_revolutions", "number", 1.0, 0.05, 1000.0,
                               "the tri-dexel is rebuilt from the polar field after this many revolutions"),
    "volume_consistency_tol": ("volume_consistency_tol", "number", 0.02, 1e-6, 1.0,
                               "a pass whose field / analytic chip volume leaves 1 +- this (plus the field's "
                               "own family spread) raises a finding (architecture 5.4)"),
    "air_cut_report_s": ("air_cut_report_s", "time", 2.0, 0.0, 1.0e6,
                         "consecutive feed motion out of contact longer than this is reported (info)"),
    "probe_rings": ("probe_rings", "count", 8, 2, 256, "stylus sampling rings per profile piece (G31)"),
    "probe_around": ("probe_around", "count", 32, 4, 4096, "stylus samples around each ring (G31)"),
    "mesh_capacity": ("mesh_capacity", "count", 20000, 1000, 10 ** 8, "initial vertex capacity per brick mesh"),
}


@dataclass(frozen=True)
class RemovalSettings:
    every_n_macro: int = 1
    margin_nodes: int = 1
    gouge_tolerance_m: float = 5.0e-6
    collision_tolerance_m: float = 1.0e-5
    gate_sample_cells: float = 1.0
    gate_tolerance_m: float = 1.0e-8
    gate_margin_m: float = 0.0
    sector_margin_cells: float = 1.0
    pool_headroom: float = 2.0
    pool_min_free: int = 1024
    event_capacity: int = 1024
    sweep_step_cells: float = 0.5
    connect_every_n_macro: int = 1
    polar_min_feed_m_per_rev: float = 5.0e-5
    polar_max_bytes: float = 150e6
    polar_kp: int = 2
    polar_axial_spacing_m: float = 0.0
    polar_sync_revolutions: float = 1.0
    volume_consistency_tol: float = 0.02
    air_cut_report_s: float = 2.0
    probe_rings: int = 8
    probe_around: int = 32
    mesh_capacity: int = 20000
    # top-level physics keys
    lattice_m: float = 0.25e-3
    refine: tuple = ()
    polar_theta_bins: int = 720


_SCALE = {"length": {"_mm": 1e-3, "_um": 1e-6, "_nm": 1e-9, "_m": 1.0, "_mm_rev": 1e-3},
          "time": {"_ms": 1e-3, "_s": 1.0}, "megabytes": {"_mb": 1e6}}


def _get(section, key):
    if section is None:
        return None
    if isinstance(section, Mapping):
        return section.get(key)
    return getattr(section, key, None)


def _lookup(section, name: str, file_key: str, kind: str):
    for key in (name, file_key):
        raw = _get(section, key)
        if raw is None:
            continue
        number = float(raw)
        if key == file_key and kind in _SCALE:
            for suffix in sorted(_SCALE[kind], key=len, reverse=True):
                if key.endswith(suffix):
                    number *= _SCALE[kind][suffix]
                    break
        return number
    return None


def _refine(physics) -> tuple:
    items = _get(physics, "refine") or []
    out = []
    for item in items:
        box = _get(item, "box")
        factor = _get(item, "factor")
        lo = _get(box, "min_m")
        hi = _get(box, "max_m")
        if lo is None:
            lo = [float(v) * 1e-3 for v in (_get(box, "min_mm") or [])]
            hi = [float(v) * 1e-3 for v in (_get(box, "max_mm") or [])]
        if len(lo) != 3 or len(hi) != 3 or factor is None:
            raise InputError("physics.refine entries need box {min_mm, max_mm} and factor")
        out.append((tuple(float(v) for v in lo), tuple(float(v) for v in hi), int(factor)))
    return tuple(out)


def settings(physics) -> RemovalSettings:
    """Settings from a project's ``physics`` section (its ``removal`` mapping plus the top-level lattice,
    refine and theta-bin keys). None gives the defaults; out-of-range values raise InputError."""
    section = _get(physics, "removal")
    values = {}
    for item in fields(RemovalSettings):
        if item.name not in FIELDS:
            continue
        file_key, kind, default, lo, hi, _ = FIELDS[item.name]
        found = _lookup(section, item.name, file_key, kind)
        if found is None:
            found = default
        if not lo <= found <= hi:
            raise InputError(f"physics.removal.{file_key} = {found:g} (SI) is outside [{lo:g}, {hi:g}]")
        values[item.name] = int(round(found)) if kind == "count" else float(found)
    lat = _lookup(physics, "lattice_m", "lattice_mm", "length")
    if lat is not None:
        if not 1e-6 <= lat <= 0.1:
            raise InputError(f"physics.lattice_mm = {lat * 1e3:g} mm is outside [0.001, 100]")
        values["lattice_m"] = lat
    bins = _get(physics, "polar_theta_bins")
    if bins is not None:
        values["polar_theta_bins"] = int(bins)
    values["refine"] = _refine(physics) if physics is not None else ()
    return RemovalSettings(**values)


def _schema():
    from ..db.schema import Count, Qty, Struct
    nodes = {}
    for _, (file_key, kind, _default, lo, hi, doc) in FIELDS.items():
        if kind == "count":
            nodes[file_key] = Count(lo=int(lo), hi=int(hi), doc=doc)
        elif kind == "megabytes":
            nodes[file_key] = Qty(lo=lo / 1e6, hi=hi / 1e6, dimensionless=True, doc=doc)
        else:
            nodes[file_key] = Qty(lo=lo, hi=hi, doc=doc)
    return Struct(nodes, doc="removal settings (docs/models/removal.md)", meta=False)


PHYSICS_SCHEMA = _schema()
