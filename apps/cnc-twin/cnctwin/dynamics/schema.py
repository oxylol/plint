"""``physics.dynamics`` project settings: the numerical settings of the dynamics domain (model coefficients live
in db/models/dynamics-*.yaml).

:data:`FIELDS` is the single declaration: :func:`settings` reads ``physics.dynamics`` (the SI view or the raw
mapping) into :class:`DynamicsSettings` and rejects out-of-range values; :data:`PHYSICS_SCHEMA` is the
:mod:`cnctwin.db.schema` node the project schema checks the section with.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields

from ..core.errors import InputError

# name (SI) -> (file key, kind, default (SI), lo, hi, doc)
FIELDS: dict[str, tuple] = {
    "every_n_macro": ("every_n_macro", "count", 1, 1, 1,
                      "dynamics runs every macro step (its kernels run in every L0 step and L1 tick)"),
    "sample_margin": ("sample_margin", "number", 1.25, 1.0, 10.0,
                      "vibration-sample buffer per macro step, relative to the L0 steps of a step"),
    "chatter_window_revs": ("chatter_window_revs", "count", 20, 4, 10000,
                            "chatter analysis window in spindle revolutions (frequency resolution f_s / revs)"),
    "chatter_window_max_s": ("chatter_window_max_s", "time", 0.5, 0.01, 10.0,
                             "longest chatter analysis window"),
    "chatter_min_revs": ("chatter_min_revs", "count", 6, 2, 10000,
                         "fewest spindle revolutions in the window before it is analysed"),
    "chatter_clear_steps": ("chatter_clear_steps", "count", 5, 1, 10000,
                            "macro steps below the clear index before a chatter episode ends"),
    "max_findings": ("max_findings", "count", 3, 1, 1000,
                     "chatter / stability findings per tool and operation (counts in the results are complete)"),
    "sld_n_freq": ("sld_n_freq", "count", 1500, 64, 200000, "chatter-frequency points of a lobe scan"),
    "sld_n_lobes": ("sld_n_lobes", "count", 12, 1, 200, "lobes of a diagram"),
    "sld_n_speeds": ("sld_n_speeds", "count", 300, 8, 100000, "speeds of a diagram's lower envelope"),
    "sld_speed_lo_frac": ("sld_speed_lo_frac", "number", 0.05, 0.001, 1.0,
                          "lowest diagram speed as a share of the spindle's maximum speed"),
    "sld_freq_lo_factor": ("sld_freq_lo_factor", "number", 0.5, 0.01, 1.0,
                           "lowest chatter frequency of a scan / lowest mode frequency"),
    "sld_freq_hi_factor": ("sld_freq_hi_factor", "number", 2.0, 1.0, 100.0,
                           "highest chatter frequency of a scan / highest mode frequency"),
    "sld_refresh_macro": ("sld_refresh_macro", "count", 50, 1, 10 ** 6,
                          "macro steps between operating-point limits of an unchanged cut"),
    "sld_max_classes": ("sld_max_classes", "count", 64, 1, 10 ** 5,
                        "engagement classes kept per tool and operation for the end-of-operation lobe diagrams"),
    "sld_max_diagrams": ("sld_max_diagrams", "count", 8, 0, 10 ** 4,
                         "lobe diagrams computed per tool at the end of an operation (classes of the operating points "
                         "with the smallest margins first)"),
    "engagement_tol": ("engagement_tol", "number", 0.05, 0.0, 1.0,
                       "relative change of radial immersion, speed range or coefficients that starts a new diagram"),
    "mf_harmonics": ("mf_harmonics", "count", 0, 0, 8,
                     "tooth-passing harmonics of the multi-frequency solution at the operating points (0: zero order)"),
    "pd_max_iter": ("pd_max_iter", "count", 30, 1, 1000, "iterations of the process-damped limit"),
    "pd_tol": ("pd_tol", "number", 1e-3, 1e-9, 0.1, "relative tolerance of the process-damped limit"),
    "mode_coupling_max_m": ("mode_coupling_max_m", "length", 0.05, 1e-4, 1.0,
                            "largest depth scanned for the mode-coupling limit"),
    "sd_check": ("sd_check", "count", 1, 0, 1,
                 "1: verify the recorded operating points by semi-discretisation at the end of the run"),
    "sd_m": ("sd_m", "count", 40, 8, 1000, "semi-discretisation intervals per tooth period (at least)"),
    "sd_points_per_period": ("sd_points_per_period", "count", 24, 4, 1000,
                             "semi-discretisation intervals per period of the highest checked mode over the delay"),
    "sd_max_points": ("sd_max_points", "count", 20, 0, 10000, "operating points checked by semi-discretisation"),
    "sd_a_max_m": ("sd_a_max_m", "length", 0.05, 1e-4, 1.0, "largest depth of the semi-discretisation scan"),
    "sd_scan": ("sd_scan", "count", 30, 4, 10000, "depths of the semi-discretisation scan before bisection"),
    "sd_tol": ("sd_tol", "number", 1e-3, 1e-9, 0.1, "relative tolerance of the semi-discretisation limit"),
    "max_operating_points": ("max_operating_points", "count", 200, 1, 10 ** 6,
                             "operating points recorded per tool (sld/<tool>.json)"),
    "relax_estimate": ("relax_estimate", "count", 1, 0, 1,
                       "1: the corrector uses the mean cutting stiffness of the operating point (implicit-like "
                       "accuracy); 0: plain predictor-corrector"),
    "rotation_length_m": ("rotation_length_m", "length", 0.01, 1e-5, 1.0,
                          "length that weights rotations against translations in a basis projection (the tool "
                          "radius is used when it is larger)"),
}


@dataclass(frozen=True)
class DynamicsSettings:
    every_n_macro: int = 1
    sample_margin: float = 1.25
    chatter_window_revs: int = 20
    chatter_window_max_s: float = 0.5
    chatter_min_revs: int = 6
    chatter_clear_steps: int = 5
    max_findings: int = 3
    sld_n_freq: int = 1500
    sld_n_lobes: int = 12
    sld_n_speeds: int = 300
    sld_speed_lo_frac: float = 0.05
    sld_freq_lo_factor: float = 0.5
    sld_freq_hi_factor: float = 2.0
    sld_refresh_macro: int = 50
    sld_max_classes: int = 64
    sld_max_diagrams: int = 8
    engagement_tol: float = 0.05
    mf_harmonics: int = 0
    pd_max_iter: int = 30
    pd_tol: float = 1e-3
    mode_coupling_max_m: float = 0.05
    sd_check: int = 1
    sd_m: int = 40
    sd_points_per_period: int = 24
    sd_max_points: int = 20
    sd_a_max_m: float = 0.05
    sd_scan: int = 30
    sd_tol: float = 1e-3
    max_operating_points: int = 200
    relax_estimate: int = 1
    rotation_length_m: float = 0.01


def _get(section, key):
    if section is None:
        return None
    if isinstance(section, Mapping):
        return section.get(key)
    return getattr(section, key, None)


def settings(section) -> DynamicsSettings:
    """Settings from ``physics.dynamics`` (None: defaults). Out-of-range values raise InputError."""
    values = {}
    for item in fields(DynamicsSettings):
        file_key, kind, default, lo, hi, _ = FIELDS[item.name]
        found = _get(section, item.name)
        if found is None:
            found = _get(section, file_key)
        found = getattr(found, "value", found)
        if found is None:
            found = default
        number = float(found)
        if not lo <= number <= hi:
            raise InputError(f"physics.dynamics.{file_key} = {number:g} (SI) is outside [{lo:g}, {hi:g}]")
        values[item.name] = int(round(number)) if kind == "count" else number
    return DynamicsSettings(**values)


def _schema():
    from ..db.schema import Count, Qty, Struct
    nodes = {}
    for _, (file_key, kind, _default, lo, hi, doc) in FIELDS.items():
        if kind == "count":
            nodes[file_key] = Count(lo=int(lo), hi=int(hi), doc=doc)
        else:
            nodes[file_key] = Qty(lo=lo, hi=hi, doc=doc, dimensionless=(kind == "number"))
    return Struct(nodes, doc="dynamics settings (docs/models/dynamics.md)", meta=False)


PHYSICS_SCHEMA = _schema()

__all__ = ["DynamicsSettings", "settings", "FIELDS", "PHYSICS_SCHEMA"]
