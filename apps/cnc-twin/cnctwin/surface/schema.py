"""``physics.surface`` project settings: the numerical settings of the surface domain (model coefficients
live in db/models/surface-*.yaml).

:data:`FIELDS` is the single declaration: :func:`settings` reads ``physics.surface`` (the SI view or the raw
mapping) into :class:`SurfaceSettings` and rejects out-of-range values; :data:`PHYSICS_SCHEMA` is the
:mod:`cnctwin.db.schema` node the project schema checks the section with.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields

from ..core.errors import InputError

DEG = math.pi / 180.0

# name (SI) -> (file key, kind, default (SI), lo, hi, doc)
FIELDS: dict[str, tuple] = {
    "every_n_macro": ("every_n_macro", "count", 1, 1, 1,
                      "surface runs every macro step (it gathers the surface events of every step)"),
    "face_angle_rad": ("face_angle_deg", "angle", 10.0 * DEG, 1.0 * DEG, 45.0 * DEG,
                       "events whose normals differ by more than this belong to different face patches"),
    "plane_tol_m": ("plane_tol_mm", "length", 3.0e-4, 1.0e-6, 1.0e-2,
                    "events of one brick farther than this from a facet's plane start another patch"),
    "pass_gap_s": ("pass_gap_s", "time", 0.05, 1.0e-4, 10.0,
                   "no event of a tool on a patch for this long (and 4 revolutions) closes its pass and its lines"),
    "lines_per_pass": ("lines_per_pass", "count", 1, 1, 64,
                       "profile lines kept per patch and pass (other events only update the patch sums); more lines "
                       "give more statistics at a proportional cost"),
    "line_rows": ("line_rows", "count", 1024, 64, 1_000_000,
                  "generating passes per profile line before it is evaluated and restarted (memory bound)"),
    "floor_line_rows": ("floor_line_rows", "count", 4096, 64, 1_000_000,
                        "the same for floor lines, which receive every axial-normal element of every tooth pass"),
    "half_spacings": ("half_spacings", "number", 4.0, 1.0, 50.0,
                      "half-width of the template window around a pass's lowest point, in feed spacings"),
    "turn_bins": ("turn_bins", "count", 36, 4, 3600,
                  "angular bins of a turned face: each bin is one candidate profile line"),
    "burr_limit_m": ("burr_limit_mm", "length", 1.0e-4, 0.0, 0.1,
                     "largest burr height the drawing allows (ISO 13715 edge state); 0: no burr finding"),
    "mark_limit_m": ("mark_limit_um", "length", 2.0e-6, 0.0, 1.0e-3,
                     "a dwell, witness or servo mark deeper than this (or than the face's Rz specification, the "
                     "larger) raises a finish finding; 0: no mark findings"),
    "dwell_speed_m_s": ("dwell_speed_mm_min", "speed", 1.0 / 60.0 * 1e-3, 0.0, 1.0,
                        "feed speed below which a tool in the cut is dwelling"),
    "spindle_stop_ratio": ("spindle_stop_ratio", "number", 0.05, 1e-3, 0.9,
                           "spindle speed below this share of its running speed counts as stopped"),
    "reversal_window_s": ("reversal_window_s", "time", 0.02, 1e-4, 1.0,
                          "half-width of the window around an axis reversal searched for the servo mark"),
    "servo_min_m": ("servo_min_um", "length", 1.0e-7, 0.0, 1e-3,
                    "smallest servo (reversal) mark that is recorded"),
    "witness_min_m": ("witness_min_um", "length", 1.0e-7, 0.0, 1e-3,
                      "smallest step-over / re-entry mismatch that is recorded as a witness mark"),
    "chatter_index": ("chatter_index", "number", 0.5, 0.0, 1.0,
                      "dynamics' chatter index above which the patches generated in the step carry chatter marks"),
    "silhouette_angles": ("silhouette_angles", "count", 720, 36, 100_000,
                          "angles at which the revolved tool outline is sampled for the step-over profile"),
    "thread_points": ("thread_points", "count", 8192, 256, 10_000_000,
                      "profile points kept per thread feature (older points are thinned by halves)"),
    "max_marks": ("max_marks", "count", 5000, 10, 10_000_000,
                  "marks kept individually (beyond: counted per patch only)"),
    "max_burr_edges": ("max_burr_edges", "count", 20000, 10, 10_000_000,
                       "burr edge records kept (beyond: the smallest are dropped and counted)"),
    "max_facets": ("max_facets", "count", 16, 1, 1024, "face patches per brick"),
}


@dataclass(frozen=True)
class SurfaceSettings:
    every_n_macro: int = 1
    face_angle_rad: float = 10.0 * DEG
    plane_tol_m: float = 3.0e-4
    pass_gap_s: float = 0.05
    lines_per_pass: int = 1
    line_rows: int = 1024
    floor_line_rows: int = 4096
    half_spacings: float = 4.0
    turn_bins: int = 36
    burr_limit_m: float = 1.0e-4
    mark_limit_m: float = 2.0e-6
    dwell_speed_m_s: float = 1.0 / 60.0 * 1e-3
    spindle_stop_ratio: float = 0.05
    reversal_window_s: float = 0.02
    servo_min_m: float = 1.0e-7
    witness_min_m: float = 1.0e-7
    chatter_index: float = 0.5
    silhouette_angles: int = 720
    thread_points: int = 8192
    max_marks: int = 5000
    max_burr_edges: int = 20000
    max_facets: int = 16


_CONVERT = {"angle": DEG, "length_mm": 1e-3, "length_um": 1e-6}


def _get(section, key):
    if section is None:
        return None
    if isinstance(section, Mapping):
        return section.get(key)
    return getattr(section, key, None)


def _raw_scale(file_key: str) -> float:
    """Factor from the file key's unit to SI (for raw mappings that are not SI views)."""
    if file_key.endswith("_deg"):
        return DEG
    if file_key.endswith("_mm"):
        return 1e-3
    if file_key.endswith("_um"):
        return 1e-6
    if file_key.endswith("_mm_min"):
        return 1e-3 / 60.0
    return 1.0


def settings(section) -> SurfaceSettings:
    """Settings from ``physics.surface`` (None: defaults). Out-of-range values raise InputError."""
    values = {}
    for item in fields(SurfaceSettings):
        file_key, kind, default, lo, hi, _ = FIELDS[item.name]
        found = _get(section, item.name)
        if found is None:
            raw = _get(section, file_key)
            raw = getattr(raw, "value", raw)
            found = float(raw) * _raw_scale(file_key) if raw is not None else None
        found = getattr(found, "value", found)
        if found is None:
            found = default
        number = float(found)
        if not lo <= number <= hi:
            raise InputError(f"physics.surface.{file_key} = {number:g} (SI) is outside [{lo:g}, {hi:g}]")
        values[item.name] = int(round(number)) if kind == "count" else number
    return SurfaceSettings(**values)


def _schema():
    from ..db.schema import Count, Qty, Struct
    nodes = {}
    for _, (file_key, kind, _default, lo, hi, doc) in FIELDS.items():
        if kind == "count":
            nodes[file_key] = Count(lo=int(lo), hi=int(hi), doc=doc)
        elif kind == "number":
            nodes[file_key] = Qty(lo=lo, hi=hi, doc=doc, dimensionless=True)
        else:
            nodes[file_key] = Qty(lo=lo, hi=hi, doc=doc)
    return Struct(nodes, doc="surface settings (docs/models/surface.md)", meta=False)


PHYSICS_SCHEMA = _schema()

__all__ = ["SurfaceSettings", "settings", "FIELDS", "PHYSICS_SCHEMA"]
