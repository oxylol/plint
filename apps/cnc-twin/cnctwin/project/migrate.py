"""Project file versions, migrated forward (docs/formats.md "Project file").

* **Version 0** (before the ``format``/``version`` keys; a file without ``version`` is version 0):
  one operation written flat at the top (``program``, ``setup``, ``from_previous``,
  ``nozzle_positions``, optionally ``operation_name``), ``tool_list`` and ``tool_library`` beside
  each other, ``material`` and ``coolant`` as bare ids.
* **Version 1** (current, db/_templates/project.yaml): ``operations: [{name, program, setup ..}]``,
  ``tools: {library, list}``, ``material: {id, batch}``, ``coolant: {fluid, ...}``.

:func:`migrate` works on plain data; :func:`migrate_with_locations` also moves the YAML line
numbers so that problems in a migrated file still name the line they were written on.
"""

from __future__ import annotations

import copy

from ..core.errors import InputError
from .schema import CURRENT_VERSION, FORMAT

V0_OPERATION_KEYS = ("program", "setup", "from_previous", "nozzle_positions")
DEFAULT_V0_OPERATION = "OP10"


def version_of(raw) -> int:
    if not isinstance(raw, dict):
        raise InputError("a project file is a YAML mapping")
    version = raw.get("version", 0)
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise InputError(f"project version {version!r} is not an integer >= 0")
    if version > CURRENT_VERSION:
        raise InputError(f"project version {version} is newer than this build supports ({CURRENT_VERSION})")
    return version


def migrate(raw) -> dict:
    """The current form of a parsed project (a copy); raises InputError if it cannot be migrated."""
    return migrate_with_locations(raw, None)[0]


def migrate_with_locations(raw, locations):
    """``(data, locations, notes)``: the current form, moved line numbers, and what was migrated."""
    version = version_of(raw)
    data = copy.deepcopy(raw)
    locations = locations.copy() if locations is not None else None
    notes: list[str] = []
    if version == 0:
        data, notes = _from_v0(data, locations)
        version = 1
    return data, locations, notes


def _relocate(locations, old, new):
    if locations is not None:
        locations.relocate(tuple(old), tuple(new))


def _from_v0(data: dict, locations):
    notes = []
    if data.get("format", FORMAT) != FORMAT:
        raise InputError(f"format {data.get('format')!r} is not {FORMAT}")
    out: dict = {"format": FORMAT, "version": 1}
    if locations is not None:
        locations.set(("format",), locations.line(("format",)) or 1)
        locations.set(("version",), locations.line(("version",)) or 1)
    operation: dict = {}
    for key in V0_OPERATION_KEYS:
        if key in data:
            operation[key] = data[key]
    name = data.get("operation_name", DEFAULT_V0_OPERATION)
    tools: dict = {}
    for key, value in data.items():
        if key in ("format", "version", "operation_name") or key in V0_OPERATION_KEYS:
            continue
        if key == "tool_list":
            tools["list"] = value
            _relocate(locations, ("tool_list",), ("tools", "list"))
            notes.append("tool_list -> tools.list")
        elif key == "tool_library":
            tools["library"] = value
            _relocate(locations, ("tool_library",), ("tools", "library"))
            notes.append("tool_library -> tools.library")
        elif key == "material" and isinstance(value, str):
            out["material"] = {"id": value}
            _relocate(locations, ("material",), ("material", "id"))
            notes.append("material: <id> -> material: {id: <id>}")
        elif key == "coolant" and isinstance(value, str):
            out["coolant"] = {"fluid": value}
            _relocate(locations, ("coolant",), ("coolant", "fluid"))
            notes.append("coolant: <id> -> coolant: {fluid: <id>}")
        else:
            out[key] = value
    if "tools" not in out:
        out["tools"] = tools or {"list": []}
    if operation:
        operation = {"name": name, **operation}
        if "operations" in out:
            raise InputError("a version 0 project writes its single operation flat, not under operations:")
        out["operations"] = [operation]
        line = locations.line(("program",)) if locations is not None else 0
        for key in V0_OPERATION_KEYS:
            if key in operation:
                _relocate(locations, (key,), ("operations", 0, key))
        if locations is not None:
            locations.set(("operations",), line)
            locations.set(("operations", 0), line)
            locations.set(("operations", 0, "name"), line)
        notes.append("program/setup -> operations[0]")
    if locations is not None and ("tools",) not in locations.entries:
        locations.set(("tools",), locations.line(("tools", "list")))
    return out, notes


__all__ = ["migrate", "migrate_with_locations", "version_of", "V0_OPERATION_KEYS"]
