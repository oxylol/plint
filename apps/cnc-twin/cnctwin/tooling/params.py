"""Model coefficients of the tooling library in SI (db/models/tooling.yaml).

Each entry is read with its written unit (``cnctwin.db.models.records``) and converted here by
the unit table of :mod:`cnctwin.units`, so a value written in degrees or micrometres arrives in
radians or metres whatever the loader's own conversion policy. Numeric entries of mappings and
lists convert with the entry's unit as well.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache

from ..db.models import records
from ..units import find_suffix

MODEL = "tooling"


@lru_cache(maxsize=None)
def _table() -> dict:
    out = {}
    for name, record in records(MODEL).items():
        out[name] = _to_si(record["value"], str(record.get("unit", "")))
    return out


def _to_si(value, unit: str):
    suffix = find_suffix("x_" + unit) if unit and unit != "-" else None
    return _scale(value, suffix)


def _scale(value, suffix):
    if suffix is None or value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Mapping):
        return {k: _scale(v, suffix) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scale(v, suffix) for v in value]
    return float(value) * suffix.factor + suffix.offset


def p(name: str):
    """One coefficient in SI (KeyError if the model file does not define it)."""
    return _table()[name]


def pf(name: str) -> float:
    return float(p(name))


def core_ratio(flutes: int) -> float:
    """Default core diameter / D for a flute count (db: core_ratio_by_flutes, 1..6+)."""
    table = p("core_ratio_by_flutes")
    index = min(max(int(flutes), 1), len(table)) - 1
    return float(table[index])


def clear_cache() -> None:
    _table.cache_clear()
