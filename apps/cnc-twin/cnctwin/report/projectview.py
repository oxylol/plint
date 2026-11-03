"""Reading the resolved project that run.json carries (architecture 16, 18).

run.json holds the project's SI view (``size_m``, ``stickout_m``, ``work_offsets_m``, poses as
``{translate_m, rotate_rad, order}``, database objects in place of their ids). These helpers also
accept the keys as written in project files (``size_mm``, ``rotate_deg``) and convert them with the
unit table of :mod:`cnctwin.db.keyunits`, and sourced values ``{value, src}`` as their value.
"""

from __future__ import annotations

from ..db import keyunits


def unwrap(value):
    if isinstance(value, dict) and "value" in value and not ({"translate_m", "translate_mm"} & set(value)):
        return value["value"]
    return value


def si(mapping, stem: str, si_token: str):
    """The SI value of ``<stem>_<unit>`` in ``mapping`` whatever unit it is written in (None if absent)."""
    if not isinstance(mapping, dict):
        return None
    key = f"{stem}_{si_token}" if stem else si_token
    if key in mapping:
        return unwrap(mapping[key])
    for name, value in mapping.items():
        found, unit = keyunits.parse_key(str(name))
        if found == stem and unit is not None and not unit.scale and unit.si_token == si_token:
            return keyunits.convert(unwrap(value), unit)
    return None


def ref_id(value) -> str:
    """The id of a database reference: the id text, or the ``id`` of a resolved (or inline) object."""
    if isinstance(value, dict):
        for key in ("id", "db_id", "name", "extends"):
            if value.get(key):
                return str(value[key])
        return "inline"
    return "" if value is None else str(value)


def pose(mapping) -> tuple[list[float], list[float], str] | None:
    """(translate_m, rotate_rad, order) of a pose mapping, or None."""
    if not isinstance(mapping, dict):
        return None
    translate = si(mapping, "translate", "m")
    rotate = si(mapping, "rotate", "rad")
    if translate is None and rotate is None:
        return None
    return ([float(v) for v in (translate or [0.0, 0.0, 0.0])], [float(v) for v in (rotate or [0.0, 0.0, 0.0])],
            str(mapping.get("order") or "xyz"))


def get(mapping, *path, default=None):
    current = mapping
    for key in path:
        if isinstance(current, dict):
            current = current.get(key)
        elif isinstance(current, list) and isinstance(key, int) and 0 <= key < len(current):
            current = current[key]
        else:
            return default
    return default if current is None else current


__all__ = ["unwrap", "si", "ref_id", "pose", "get"]
