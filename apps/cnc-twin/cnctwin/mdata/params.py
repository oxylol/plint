"""The model parameter files of machine data import, alignment and calibration (db/models/mdata-*.yaml)."""

from __future__ import annotations

from ..db.models import params


def import_params() -> dict:
    """db/models/mdata-import.yaml (request sizes, time-outs, memory caps)."""
    return params("mdata-import")


def align_params() -> dict:
    """db/models/mdata-align.yaml (marker matching, robust fit, cross-correlation, resampling)."""
    return params("mdata-align")


def calibration_params() -> dict:
    """db/models/mdata-calibration.yaml (solver, interval level, identifiability flags, bounds)."""
    return params("mdata-calibration")


__all__ = ["import_params", "align_params", "calibration_params"]
