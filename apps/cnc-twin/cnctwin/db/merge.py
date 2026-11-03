"""Deep merge for ``extends`` (database files, series defaults and inline project items).

``merged = deep_merge(base, override)``: mappings merge key by key, recursively; every other
value (numbers, text, lists, null) in ``override`` replaces the base value whole. Lists are not
merged element-wise: an extending file that changes one spindle repeats the spindle list. A
sourced value or table (``{value: ..}``, ``{table: ..}``) is a value, not a section: it is
replaced whole.

Sources stay with the file that wrote the value: before merging, :func:`pin_sources` writes the
effective (inherited) ``src`` into every mapping of each side that does not name its own (and the
``note`` of an ``est`` into sourced values), so a section ``src`` of the extending file does not
re-attribute the values it inherits from the base. Source rules and the estimate list are checked
per file as written (:mod:`cnctwin.db.sources`), never on the merged tree.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping

_SKIP = ("src", "note", "notes", "page", "pages", "digitised", "uncertainty_pct")


def is_value_mapping(mapping) -> bool:
    return isinstance(mapping, Mapping) and ("value" in mapping or "table" in mapping)


def deep_merge(base, override):
    if isinstance(base, Mapping) and isinstance(override, Mapping):
        result = {k: copy.deepcopy(v) for k, v in base.items()}
        for key, value in override.items():
            if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping) \
                    and not is_value_mapping(value) and not is_value_mapping(result[key]):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result
    return copy.deepcopy(override)


def pin_sources(data, inherited_src=None, inherited_note=None):
    """A copy of ``data`` where every mapping names the src in force for it (and est notes)."""
    if isinstance(data, Mapping):
        own = data.get("src") if isinstance(data.get("src"), str) else None
        src = own or inherited_src
        if own is not None:
            note = data.get("note") if own == "est" else None
        else:
            note = inherited_note
        out = {}
        for key, value in data.items():
            if key in _SKIP:
                out[key] = copy.deepcopy(value)
            else:
                out[key] = pin_sources(value, src, note)
        if own is None and src is not None:
            out["src"] = src
            if src == "est" and note and is_value_mapping(data) and "note" not in data:
                out["note"] = note
        return out
    if isinstance(data, list):
        return [pin_sources(v, inherited_src, inherited_note) for v in data]
    return data


def merge_pinned(base, override):
    """Deep merge with both sides' sources pinned first (``extends``)."""
    return deep_merge(pin_sources(base), pin_sources(override))


__all__ = ["deep_merge", "pin_sources", "merge_pinned", "is_value_mapping"]
