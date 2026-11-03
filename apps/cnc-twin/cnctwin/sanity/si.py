"""Reading project and database objects whatever form they come in.

The resolved project (``ctx.project``) and the databases are SI views (:class:`cnctwin.db.SIMap`,
docs/architecture.md 18); tests and tools may hand over the raw YAML mappings instead (unit
suffixed keys). :func:`si_view` turns a raw mapping into its SI view, :func:`get` follows a path
through views, mappings, lists and plain objects, :func:`num` reads a number (also a ``Sourced``
value or the 20 degC value of a ``Table``).
"""

from __future__ import annotations

from collections.abc import Mapping

from ..db.simap import SIMap, Sourced, Table, wrap


def si_view(obj):
    """SI view of ``obj``: a raw mapping is wrapped (``travel_mm`` -> ``travel_m``), anything else
    (an SIMap, a resolved-project object, None) is returned unchanged."""
    if obj is None or isinstance(obj, SIMap):
        return obj
    if isinstance(obj, Mapping):
        return wrap(obj)
    return obj


def get(obj, *path, default=None):
    """Value at ``path`` (keys, attribute names or list indices); ``default`` where it is absent."""
    node = obj
    for key in path:
        if node is None:
            return default
        if isinstance(key, int) and not isinstance(node, Mapping):
            if isinstance(node, (list, tuple)) and -len(node) <= key < len(node):
                node = node[key]
                continue
            return default
        if isinstance(node, Mapping):
            node = node.get(key)
        elif isinstance(node, Sourced):
            node = getattr(node, key)
        else:
            node = getattr(node, key, None)
    return default if node is None else node


def num(value, default=None):
    """A float from a number, a :class:`Sourced` value or a :class:`Table` (its 20 degC value)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (Sourced, Table)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return default


def plain(value):
    """The plain value of a :class:`Sourced` (lists stay lists), anything else unchanged."""
    if isinstance(value, Sourced):
        return value.value
    return value


def text(value, default=None):
    value = plain(value)
    if value is None:
        return default
    return str(value)


def items(value) -> list:
    """A list of entries (a single mapping becomes a one-entry list, None an empty list)."""
    value = plain(value)
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]
