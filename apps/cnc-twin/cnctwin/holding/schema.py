"""``physics.holding`` project settings: the numerical settings of the holding domain (model coefficients
live in db/models/holding-*.yaml).

:data:`FIELDS` is the single declaration: :func:`settings` reads ``physics.holding`` (the SI view or the raw
mapping) into :class:`HoldingSettings` and rejects out-of-range values; :data:`PHYSICS_SCHEMA` is the
:mod:`cnctwin.db.schema` node the project schema checks the section with.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields

import numpy as np

from ..core.errors import InputError
from .abi import cfg_array

# name (SI) -> (file key, kind, default (SI), lo, hi, doc)
FIELDS: dict[str, tuple] = {
    "every_n_macro": ("every_n_macro", "count", 1, 1, 1,
                      "holding runs every macro step (it integrates the cutting record of every step)"),
    "patch_nodes": ("patch_nodes", "count", 5, 1, 9,
                    "Simpson nodes per side of a contact patch (odd; 5 keeps the torsional friction capacity of a "
                    "rectangle within 1 %, 3 under-predicts it by 21-30 %)"),
    "centre_nodes": ("centre_nodes", "count", 8, 3, 64, "contact nodes around a tailstock centre"),
    "newton_tol": ("newton_tol", "number", 1e-9, 1e-14, 1e-4,
                   "relative residual of the contact equilibrium (force over the force scale)"),
    "newton_max_iter": ("newton_max_iter", "count", 60, 5, 1000, "Newton iterations per load increment"),
    "load_increments": ("load_increments", "count", 1, 1, 64,
                        "load increments of an equilibrium solve (8 times as many on the retry); the clamping solve "
                        "uses 4 times as many"),
    "max_disp_m": ("max_disp_m", "length", 1e-3, 1e-6, 0.1,
                   "translation of the part beyond which an equilibrium solve counts as motion (not held)"),
    "max_rot_rad": ("max_rot_rad", "number", 1e-2, 1e-6, 1.0,
                    "rotation (rad) beyond which an equilibrium solve counts as motion"),
    "regularisation": ("regularisation", "number", 1e-12, 0.0, 1e-6,
                       "regularising stiffness on the part's six rigid-body motions, relative to the stiffest contact"),
    "capacity_tol": ("capacity_tol", "number", 1e-3, 1e-9, 0.1,
                     "relative tolerance of the capacity (load factor) below a margin of 2; above it 10 %"),
    "margin_cap": ("margin_cap", "number", 1000.0, 2.0, 1e9, "largest load factor searched (the margin channel cap)"),
    "critical_eps": ("critical_eps", "number", 1e-2, 1e-6, 0.5,
                     "friction utilisation / normal-force share that marks a contact critical in the mechanism"),
    "sub_steps_per_tooth": ("sub_steps_per_tooth", "count", 256, 4, 65536,
                            "integration sub-steps per tooth period of the slip and walk ratchets (they run only "
                            "when the static check allows slip)"),
    "max_sub_steps": ("max_sub_steps", "count", 16384, 16, 10 ** 7, "largest number of sub-steps per L1 tick"),
    "speed_rebuild_rel": ("speed_rebuild_rel", "number", 0.02, 1e-6, 1.0,
                          "relative change of omega^2 (the centrifugal jaw loss) after which a chuck setup is clamped "
                          "again at the new gripping force"),
    "rebuild_slip_m": ("rebuild_slip_m", "length", 5e-5, 1e-7, 1e-2,
                       "slip of the part (centre of mass, plus rotation x size) after which its contacts are placed "
                       "again on the moved part"),
    "max_findings": ("max_findings", "count", 3, 1, 1000,
                     "findings of one kind raised per tool or part and operation (counts in the results are complete)"),
}


@dataclass(frozen=True)
class HoldingSettings:
    every_n_macro: int = 1
    patch_nodes: int = 5
    centre_nodes: int = 8
    newton_tol: float = 1e-9
    newton_max_iter: int = 60
    load_increments: int = 1
    max_disp_m: float = 1e-3
    max_rot_rad: float = 1e-2
    regularisation: float = 1e-12
    capacity_tol: float = 1e-3
    margin_cap: float = 1000.0
    critical_eps: float = 1e-2
    sub_steps_per_tooth: int = 256
    max_sub_steps: int = 16384
    speed_rebuild_rel: float = 0.02
    rebuild_slip_m: float = 5e-5
    max_findings: int = 3

    def cfg(self, k_max_N_per_m: float, l_c_m: float, increments: int | None = None) -> np.ndarray:
        """Solver settings array (abi.CFG_*) for a contact set with stiffest contact k_max and length l_c."""
        return cfg_array(self.newton_tol, self.newton_max_iter, increments or self.load_increments, self.max_disp_m,
                         self.max_rot_rad, self.regularisation * max(k_max_N_per_m, 1.0), max(l_c_m, 1e-6),
                         self.margin_cap, self.capacity_tol, self.critical_eps)


def _get(section, key):
    if section is None:
        return None
    if isinstance(section, Mapping):
        return section.get(key)
    return getattr(section, key, None)


def settings(section) -> HoldingSettings:
    """Settings from ``physics.holding`` (None: defaults). Out-of-range values raise InputError."""
    values = {}
    for item in fields(HoldingSettings):
        file_key, kind, default, lo, hi, _ = FIELDS[item.name]
        found = _get(section, item.name)
        if found is None:
            found = _get(section, file_key)
        found = getattr(found, "value", found)
        if found is None:
            found = default
        number = float(found)
        if not lo <= number <= hi:
            raise InputError(f"physics.holding.{file_key} = {number:g} (SI) is outside [{lo:g}, {hi:g}]")
        values[item.name] = int(round(number)) if kind == "count" else number
    if values["patch_nodes"] != 1 and values["patch_nodes"] % 2 == 0:
        raise InputError(f"physics.holding.patch_nodes = {values['patch_nodes']} must be odd (Simpson nodes)")
    return HoldingSettings(**values)


def _schema():
    from ..db.schema import Count, Qty, Struct
    nodes = {}
    for _, (file_key, kind, _default, lo, hi, doc) in FIELDS.items():
        if kind == "count":
            nodes[file_key] = Count(lo=int(lo), hi=int(hi), doc=doc)
        else:
            nodes[file_key] = Qty(lo=lo, hi=hi, doc=doc, dimensionless=(kind == "number"))
    return Struct(nodes, doc="holding settings (docs/models/holding.md)", meta=False)


PHYSICS_SCHEMA = _schema()

__all__ = ["HoldingSettings", "settings", "FIELDS", "PHYSICS_SCHEMA"]
