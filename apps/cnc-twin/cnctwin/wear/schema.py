"""``physics.wear`` project settings: the numerical settings of the wear domain (model coefficients live
in db/models/wear-*.yaml).

:data:`FIELDS` is the single declaration: :func:`settings` reads ``physics.wear`` (the SI view or the raw
mapping) into :class:`WearSettings` and rejects out-of-range values; :data:`PHYSICS_SCHEMA` is the
:mod:`cnctwin.db.schema` node the project schema checks the section with.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields

from ..core.errors import InputError

STRENGTH_MODES = ("median", "sample")

# name (SI) -> (file key, kind, default (SI), lo, hi, doc); kind text: lo is the tuple of choices
FIELDS: dict[str, tuple] = {
    "every_n_macro": ("every_n_macro", "count", 1, 1, 1,
                      "wear runs every macro step (it integrates the cutting record of every step)"),
    "strength": ("strength", "text", "median", STRENGTH_MODES, None,
                 "breakage threshold of a tool section: the median strength of its Weibull distribution, or a "
                 "counter-based draw per tool instance (stream wear.strength)"),
    "breakage_risk_warn": ("breakage_risk_warn", "number", 0.01, 1e-12, 1.0,
                           "a warning is raised once per operation when the fracture probability at a section's "
                           "peak stress exceeds this"),
    "calib_nodes": ("calib_nodes", "count", 6, 2, 64,
                    "Gauss-Legendre nodes per flank-land interval of the calibration's life integral"),
    "calib_pieces": ("calib_pieces", "count", 8, 1, 256,
                     "the calibration's life integral splits the flank land into intervals no longer than 1 / this of "
                     "it (besides its kinks: coating breakthrough, hone radius, plastic land)"),
    "rate_window_s": ("rate_window_s", "time", 10.0, 1e-3, 1e6,
                      "time constant (cutting time) of the moving average of the wear rates behind the remaining-"
                      "life prediction"),
    "chip_theta_nodes": ("chip_theta_nodes", "count", 64, 8, 4096,
                         "nodes of the angular integral of the chipping effective volume"),
    "skin_samples": ("skin_samples", "count", 64, 8, 4096, "samples per element circle of the skin query"),
    "max_chip_events": ("max_chip_events", "count", 20, 0, 10 ** 6,
                        "chipping and spallation events written to the trace per tool and operation (the counts "
                        "in tools.json are complete)"),
    "fallback_speeds": ("fallback_speeds", "count", 7, 2, 64,
                        "speeds of the response table used when a thermal or cutting record is missing"),
    "dimensional_share": ("dimensional_share", "number", 1.0, 0.01, 1.0,
                          "share of the tightest tolerance's allowed surface displacement the tool's wear may use "
                          "before the dimensional life criterion is reached"),
    "final_surface_band_m": ("final_surface_band_mm", "length", 5e-5, 1e-6, 1e-2,
                             "a surface event belongs to a toleranced plane feature (and its tool's wear to that "
                             "tolerance and finish) within this distance of the feature (or twice the tolerance's own "
                             "allowed displacement if larger): a roughing pass that leaves stock for finishing is not "
                             "the feature's final surface"),
}


@dataclass(frozen=True)
class WearSettings:
    every_n_macro: int = 1
    strength: str = "median"
    breakage_risk_warn: float = 0.01
    calib_nodes: int = 6
    calib_pieces: int = 8
    rate_window_s: float = 10.0
    chip_theta_nodes: int = 64
    skin_samples: int = 64
    max_chip_events: int = 20
    fallback_speeds: int = 7
    dimensional_share: float = 1.0
    final_surface_band_m: float = 5e-5


def _get(section, key):
    if section is None:
        return None
    if isinstance(section, Mapping):
        return section.get(key)
    return getattr(section, key, None)


def settings(section) -> WearSettings:
    """Settings from ``physics.wear`` (None: defaults). Out-of-range values raise InputError."""
    values = {}
    for item in fields(WearSettings):
        file_key, kind, default, lo, hi, _ = FIELDS[item.name]
        found = _get(section, item.name)
        if found is None:
            found = _get(section, file_key)
        found = getattr(found, "value", found)
        if found is None:
            found = default
        if kind == "text":
            if str(found) not in lo:
                raise InputError(f"physics.wear.{file_key} = {found!r} is not one of {', '.join(lo)}")
            values[item.name] = str(found)
            continue
        number = float(found)
        if not lo <= number <= hi:
            raise InputError(f"physics.wear.{file_key} = {number:g} (SI) is outside [{lo:g}, {hi:g}]")
        values[item.name] = int(round(number)) if kind == "count" else number
    return WearSettings(**values)


def _schema():
    from ..db.schema import Count, Qty, Struct, Text
    nodes = {}
    for _, (file_key, kind, _default, lo, hi, doc) in FIELDS.items():
        if kind == "count":
            nodes[file_key] = Count(lo=int(lo), hi=int(hi), doc=doc)
        elif kind == "text":
            nodes[file_key] = Text(tuple(lo), doc=doc, sourced=False)
        else:
            nodes[file_key] = Qty(lo=lo, hi=hi, doc=doc, dimensionless=(kind == "number"))
    return Struct(nodes, doc="wear settings (docs/models/wear.md)", meta=False)


PHYSICS_SCHEMA = _schema()

__all__ = ["WearSettings", "settings", "FIELDS", "PHYSICS_SCHEMA", "STRENGTH_MODES"]
