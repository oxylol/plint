"""``physics.structure`` project settings: the numerical settings of the structure domain (model coefficients
live in db/models/structure-*.yaml).

:data:`FIELDS` is the single declaration: :func:`settings` reads ``physics.structure`` (the SI view or the raw
mapping) into :class:`StructureSettings` and rejects out-of-range values; :data:`PHYSICS_SCHEMA` is the
:mod:`cnctwin.db.schema` node the project schema checks the section with.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields

from ..core.errors import InputError

# fixed by the contract (architecture 5.1): a structure refresh job started at macro n is applied at n + 50
REFRESH_LATENCY_MACRO = 50

# name (SI) -> (file key, kind, default (SI), lo, hi, doc); kinds: count | number | length (file key in mm) |
# frequency (file key in Hz)
FIELDS: dict[str, tuple] = {
    "every_n_macro": ("every_n_macro", "count", 50, 1, 10 ** 6,
                      "workpiece refresh cadence (modes by Rayleigh-Ritz, displacement grids); each job is applied "
                      f"{REFRESH_LATENCY_MACRO} macro steps after it starts (fixed latency)"),
    "max_dof": ("max_dof", "count", 50000, 300, 2 * 10 ** 6,
                "largest number of degrees of freedom of the workpiece voxel model (sets the cell size)"),
    "min_cell_m": ("min_cell_mm", "length", 1.0e-3, 1.0e-4, 0.1, "smallest voxel cell of the workpiece model"),
    "fill_threshold": ("fill_threshold", "number", 0.5, 0.01, 0.99,
                       "a cell is in the voxel model when its material fraction exceeds this"),
    "work_modes": ("work_modes", "count", 8, 1, 64, "workpiece modes published (WorkModes)"),
    "work_residual_modes": ("work_residual_modes", "count", 12, 0, 128,
                            "further modes computed and added quasi-statically to the residual compliance"),
    "eig_tol": ("eig_tol", "number", 1.0e-8, 0.0, 1.0e-3, "ARPACK tolerance of the full eigen solve (0: machine precision)"),
    "ritz_tol": ("ritz_tol", "number", 2.0e-3, 1.0e-8, 0.5,
                 "energy-norm eigen-residual sqrt(r^T K^-1 r / lambda), r = K phi - lambda M phi, of the published "
                 "modes (10 x for the residual modes) above which a Ritz refresh is replaced by a full eigen solve"),
    "direct_max_dof": ("direct_max_dof", "count", 8000, 0, 200000,
                       "systems up to this size are factorised (SuperLU); larger ones use multigrid-preconditioned CG"),
    "pcg_rtol": ("pcg_rtol", "number", 1.0e-10, 1.0e-14, 1.0e-4, "relative residual of the conjugate-gradient solves"),
    "pcg_max_iter": ("pcg_max_iter", "count", 3000, 10, 10 ** 6, "conjugate-gradient iteration limit"),
    "gmg_coarse_dof": ("gmg_coarse_dof", "count", 1500, 50, 50000, "multigrid levels are added until this size"),
    "gmg_degree": ("gmg_degree", "count", 3, 1, 10, "degree of the Chebyshev smoother"),
    "tool_max_freq_Hz": ("tool_max_freq_Hz", "frequency", 2.0e4, 100.0, 1.0e6, "highest tool mode published"),
    "tool_max_modes": ("tool_max_modes", "count", 16, 1, 128, "most tool modes published"),
    "tool_element_m": ("tool_element_mm", "length", 2.0e-3, 1.0e-4, 0.05, "longest beam element of tool and holder"),
    "tool_stations": ("tool_stations", "count", 24, 2, 512, "stations of ToolModes along the tool axis"),
    "pullout_refresh_m": ("pullout_refresh_mm", "length", 2.0e-5, 1.0e-7, 0.01,
                          "pull-out change after which a tool's modes are computed again"),
    "thin_wall_cells": ("thin_wall_cells", "number", 2.0, 0.0, 10.0,
                        "walls thinner than this many cells use the analytic plate model (blended with the FE)"),
    "wall_terms": ("wall_terms", "count", 6, 2, 12, "beam functions per direction of the Ritz plate model"),
    "wall_modes": ("wall_modes", "count", 3, 0, 16, "plate modes published per thin wall"),
    "max_walls": ("max_walls", "count", 8, 0, 256, "thin walls modelled per workpiece (largest first)"),
    "wall_min_fill": ("wall_min_fill", "number", 1.0e-3, 0.0, 0.5, "cells below this material fraction are air"),
    "lathe_slenderness": ("lathe_slenderness", "number", 3.0, 0.0, 1000.0,
                          "turned parts with free length / diameter above this use the beam model"),
    "lathe_radial_cells": ("lathe_radial_cells", "count", 12, 4, 128, "section sampling cells per radius (beam model)"),
    "soft_support_ratio": ("soft_support_ratio", "number", 1.0e-3, 1.0e-9, 0.1,
                           "force-controlled supports fix the rigid-body motions the locators leave free with this "
                           "fraction of their stiffness (clamping solve)"),
    "support_change_rel": ("support_change_rel", "number", 0.02, 0.0, 1.0,
                           "holding's supports placed again (supports_version) set the work model up again only when "
                           "a contact's stiffness or clamping force changed by more than this fraction of the "
                           "largest, a contact point moved by more than this fraction of the model's cell, or the "
                           "contact set changed (a chuck's grip during a spindle run-up does not)"),
}


@dataclass(frozen=True)
class StructureSettings:
    every_n_macro: int = 50
    max_dof: int = 50000
    min_cell_m: float = 1.0e-3
    fill_threshold: float = 0.5
    work_modes: int = 8
    work_residual_modes: int = 12
    eig_tol: float = 1.0e-8
    ritz_tol: float = 2.0e-3
    direct_max_dof: int = 8000
    pcg_rtol: float = 1.0e-10
    pcg_max_iter: int = 3000
    gmg_coarse_dof: int = 1500
    gmg_degree: int = 3
    tool_max_freq_Hz: float = 2.0e4
    tool_max_modes: int = 16
    tool_element_m: float = 2.0e-3
    tool_stations: int = 24
    pullout_refresh_m: float = 2.0e-5
    thin_wall_cells: float = 2.0
    wall_terms: int = 6
    wall_modes: int = 3
    max_walls: int = 8
    wall_min_fill: float = 1.0e-3
    lathe_slenderness: float = 3.0
    lathe_radial_cells: int = 12
    soft_support_ratio: float = 1.0e-3
    support_change_rel: float = 0.02

    def to_dict(self) -> dict:
        return asdict(self)


def _get(section, key):
    if section is None:
        return None
    if isinstance(section, Mapping):
        return section.get(key)
    return getattr(section, key, None)


def settings(section) -> StructureSettings:
    """Settings from ``physics.structure`` (None: defaults). Out-of-range values raise InputError."""
    values = {}
    for item in fields(StructureSettings):
        file_key, kind, default, lo, hi, _ = FIELDS[item.name]
        found = _get(section, item.name)
        scale = 1.0
        if found is None and file_key != item.name:
            found = _get(section, file_key)
            if found is not None and kind == "length" and file_key.endswith("_mm"):
                scale = 1e-3          # a raw file value (the SI view names it item.name)
        found = getattr(found, "value", found)
        if found is None:
            found = default
            scale = 1.0
        try:
            number = float(found) * scale
        except (TypeError, ValueError) as error:
            raise InputError(f"physics.structure.{file_key} = {found!r} is not a number") from error
        if not lo <= number <= hi:
            raise InputError(f"physics.structure.{file_key} = {number:g} (SI) is outside [{lo:g}, {hi:g}]")
        if kind == "count":
            if abs(number - round(number)) > 1e-9:
                raise InputError(f"physics.structure.{file_key} = {number:g} must be an integer")
            values[item.name] = int(round(number))
        else:
            values[item.name] = number
    return StructureSettings(**values)


def from_dict(data: dict) -> StructureSettings:
    return StructureSettings(**{f.name: data[f.name] for f in fields(StructureSettings) if f.name in data})


def _schema():
    from ..db.schema import Count, Qty, Struct
    nodes = {}
    for _, (file_key, kind, _default, lo, hi, doc) in FIELDS.items():
        if kind == "count":
            nodes[file_key] = Count(lo=int(lo), hi=int(hi), doc=doc)
        else:
            nodes[file_key] = Qty(lo=lo, hi=hi, doc=doc, dimensionless=(kind == "number"))
    return Struct(nodes, doc="structure settings (docs/models/structure.md)", meta=False)


PHYSICS_SCHEMA = _schema()

__all__ = ["StructureSettings", "settings", "from_dict", "FIELDS", "PHYSICS_SCHEMA", "REFRESH_LATENCY_MACRO"]
