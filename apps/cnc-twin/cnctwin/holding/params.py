"""Model coefficients of the holding domain (db/models/holding-*.yaml) and small readers for the SI views
of project and database objects (architecture 18), which arrive as SIMap views, plain mappings with file
keys (wrapped here) or attribute objects (test fixtures)."""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping

import numpy as np

from ..core.errors import InputError
from ..db import models
from ..db.simap import Curve, SIMap, Sourced, Table, wrap

TOOL = "holding-tool"
WORK = "holding-work"
BALANCE = "holding-balance"


@functools.lru_cache(maxsize=None)
def get(model: str) -> dict:
    """name -> SI value of one holding model file (validated by cnctwin.db.models)."""
    return dict(models.params(model))


def p(model: str, name: str) -> float:
    try:
        return float(get(model)[name])
    except KeyError:
        raise InputError(f"db/models/{model}.yaml has no parameter {name!r}") from None


def plist(model: str, name: str) -> np.ndarray:
    try:
        return np.asarray(get(model)[name], dtype=float)
    except KeyError:
        raise InputError(f"db/models/{model}.yaml has no parameter {name!r}") from None


def records(model: str) -> dict:
    return models.records(model)


# ------------------------------------------------------------------------------------------ readers
def as_map(value):
    if value is None:
        return None
    if isinstance(value, SIMap):
        return value
    if isinstance(value, Mapping):
        return wrap(dict(value))
    return value


def field_of(obj, key):
    """``obj[key]`` of a mapping, ``obj.key`` of an object, None when absent."""
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(key) if key in obj else None
    return getattr(obj, key, None)


def plain(value):
    if isinstance(value, Sourced):
        return value.value
    return value


def num(obj, key, default=None):
    """A finite number under ``key`` (sourced or plain), else ``default``."""
    value = plain(field_of(obj, key))
    if isinstance(value, Table):
        return float(value) if value.x_unit == "K" else default
    if value is None or isinstance(value, (str, bool, list, tuple, Mapping)) or isinstance(value, Curve):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def numlist(obj, key) -> list[float] | None:
    value = plain(field_of(obj, key))
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            item = plain(item)
            if isinstance(item, (list, tuple)):
                return None
            out.append(float(item))
        return out
    return None


def text(obj, key, default: str = "") -> str:
    value = plain(field_of(obj, key))
    return default if value is None else str(value)


def extra(obj, key, name):
    """A declared extra of a sourced value (``clamping_force_kN: {value, at_handle_torque_Nm}``)."""
    value = field_of(obj, key)
    if isinstance(value, Sourced):
        got = value.extra.get(name)
        return float(got) if isinstance(got, (int, float)) and not isinstance(got, bool) else None
    if isinstance(value, Mapping):
        got = plain(value.get(name))
        return float(got) if isinstance(got, (int, float)) and not isinstance(got, bool) else None
    return None


def matrix(value) -> np.ndarray | None:
    """4x4 matrix of a pose (PoseMap or array), None when absent."""
    if value is None:
        return None
    if hasattr(value, "matrix"):
        return np.asarray(value.matrix(), dtype=float)
    arr = np.asarray(plain(value), dtype=float)
    return arr if arr.shape == (4, 4) else None


def curve(obj, key):
    """(x, y) arrays of a two-column curve (Curve or rows), None when absent."""
    value = plain(field_of(obj, key))
    if isinstance(value, Curve):
        return value.x, value.y
    if isinstance(value, (list, tuple)) and value and all(isinstance(plain(r), (list, tuple)) for r in value):
        rows = np.asarray([[float(plain(v)) for v in plain(r)] for r in value], dtype=float)
        if rows.ndim == 2 and rows.shape[1] == 2:
            order = np.argsort(rows[:, 0], kind="stable")
            return rows[order, 0], rows[order, 1]
    return None


def table(obj, key):
    """A Table (property over x), None when absent."""
    value = plain(field_of(obj, key))
    return value if isinstance(value, Table) else None


__all__ = ["TOOL", "WORK", "BALANCE", "get", "p", "plist", "records", "as_map", "field_of", "plain", "num",
           "numlist", "text", "extra", "matrix", "curve", "table"]
