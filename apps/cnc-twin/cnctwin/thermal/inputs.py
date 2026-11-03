"""What the thermal domain reads from the resolved project and the machine file.

The project and database objects are SI views (cnctwin.db.simap.SIMap) whose raw mapping is
``.raw``; machine-thermal keys exist in more than one written form (``growth_um_at_max_rpm`` in the
machine files, ``growth_at_max_speed_um`` in the template), so this module reads the raw values
and converts them by the unit each accepted key names. A key that is absent reads as None.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping

from ..db.simap import SIMap, Sourced
from ..db.yamlio import load_file
from ..paths import DB
from ..units import ZERO_DEGC_K


def raw(obj):
    """The raw mapping of an SI view (or the mapping itself); None stays None."""
    if obj is None:
        return None
    if isinstance(obj, SIMap):
        return obj.raw
    if isinstance(obj, Mapping):
        return obj
    value = getattr(obj, "raw", None)
    return value if isinstance(value, Mapping) else obj


def plain(value):
    """The plain value of a possibly sourced raw entry."""
    if isinstance(value, Sourced):
        return value.value
    if isinstance(value, Mapping) and "value" in value:
        return value.get("value")
    return value


def number(mapping, forms: tuple[tuple[str, float, float], ...], default=None):
    """SI number of the first present key among ``forms`` = ((key, factor, offset), ...)."""
    if not isinstance(mapping, Mapping):
        return default
    for key, factor, offset in forms:
        if key in mapping:
            value = plain(mapping[key])
            if value is None:
                continue
            if isinstance(value, bool):
                return value
            try:
                return float(value) * factor + offset
            except (TypeError, ValueError):
                continue
    return default


def flag(mapping, key: str, default: bool = False) -> bool:
    if not isinstance(mapping, Mapping) or key not in mapping:
        return default
    value = plain(mapping[key])
    return bool(value) if value is not None else default


def get(mapping, *path, default=None):
    cur = mapping
    for key in path:
        if not isinstance(cur, Mapping):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


MM = 1e-3
UM = 1e-6
MIN = 60.0
DEGC = (1.0, ZERO_DEGC_K)


def ambient_K(project, default_K: float) -> float:
    env = raw(getattr(project, "environment", None)) if project is not None else None
    return number(env, (("ambient_degC", 1.0, ZERO_DEGC_K), ("ambient_K", 1.0, 0.0)), default_K)


def environment(project) -> Mapping:
    env = raw(getattr(project, "environment", None)) if project is not None else None
    return env if isinstance(env, Mapping) else {}


def time_between_ops_s(project) -> float:
    return number(environment(project), (("time_between_ops_min", MIN, 0.0), ("time_between_ops_s", 1.0, 0.0)), 0.0)


def machine_state(project):
    """'cold' | 'warm' | {spindle_K, axes_K, structure_K} (absent keys None)."""
    state = environment(project).get("machine_state", "warm")
    state = plain(state)
    if isinstance(state, Mapping):
        out = {}
        for name in ("spindle", "axes", "structure"):
            out[name + "_K"] = number(state, ((f"{name}_degC", 1.0, ZERO_DEGC_K), (f"{name}_K", 1.0, 0.0)))
        return out
    return str(state or "warm")


def machine_raw(project) -> Mapping:
    machine = getattr(project, "machine", None) if project is not None else None
    data = raw(machine)
    return data if isinstance(data, Mapping) else {}


def physics_thermal(project):
    physics = getattr(project, "physics", None) if project is not None else None
    if physics is None:
        return None
    section = physics.get("thermal") if isinstance(physics, Mapping) else getattr(physics, "thermal", None)
    return section


@functools.lru_cache(maxsize=None)
def _item_index(folder: str) -> dict:
    """item id -> raw item of every series file in db/<folder> (tools, holders, workholding)."""
    index = {}
    base = DB / folder
    if not base.is_dir():
        return index
    for path in sorted(base.glob("*.yaml")):
        try:
            data = load_file(path).data
        except Exception:          # a broken file is reported by the database checks, not here
            continue
        items = data.get("items") if isinstance(data, Mapping) else None
        for item in items or []:
            if isinstance(item, Mapping) and isinstance(item.get("id"), str):
                index.setdefault(item["id"], item)
    return index


def catalogue_item(folder: str, item):
    """A resolved item (SI view) for a project reference: SIMap/mapping as given, or an id looked up
    in the series files of db/<folder>."""
    if item is None:
        return None
    if isinstance(item, str):
        found = _item_index(folder).get(item)
        return SIMap(found) if found is not None else None
    if isinstance(item, Mapping) and not isinstance(item, SIMap):
        return SIMap(item)
    return item
