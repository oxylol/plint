"""The immutable tool table of a project (cnctwin.core.state.ToolTable / ToolSlot).

One slot per ``tools.list`` entry, in list order. ``tool_id`` is the database id the entry names
(``inline:<n>`` for an inline item, n = list index); ``holder_id`` likewise ('' without holder).
``path`` is 0-based (the project file counts paths from 1). ``spindle_id`` is the entry's
``spindle``, else the machine's first spindle that drives the tool (milling or live-tool
spindles for rotating tools, turning spindles for non-rotating tools), else ``main``.
Sister groups: ``life.group`` (>= 1) groups tools that replace each other; ``life.sister_of: N``
joins tool N's group (a new group is opened for N when it has none); 0 means no group.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..core.errors import InputError
from ..core.state import ToolSlot, ToolTable
from .spec import ROTATING_TYPES, as_simap, get, plain, text


def _raw(entry):
    raw = getattr(entry, "raw", None)
    return raw if isinstance(raw, Mapping) else (entry if isinstance(entry, Mapping) else {})


def entry_ids(entry, index: int) -> tuple[str, str]:
    raw = _raw(entry)
    tool_ref = raw.get("tool")
    holder_ref = raw.get("holder")
    explicit = raw.get("tool_id")
    if isinstance(explicit, str):
        tool_id = explicit
    elif isinstance(tool_ref, str):
        tool_id = tool_ref
    else:
        tool_id = f"inline:{index}"
    if isinstance(holder_ref, str):
        holder_id = holder_ref
    elif isinstance(holder_ref, Mapping):
        holder_id = f"inline:{index}"
    else:
        holder_id = ""
    return tool_id, holder_id


def _tool_type(entry) -> str | None:
    tool = get(entry, "tool")
    if isinstance(tool, Mapping):
        return text(tool, "type")
    return None


def _default_spindle(machine, kind: str | None) -> str:
    spindles = get(machine, "spindles") if machine is not None else None
    if not spindles:
        return "main"
    rotating = kind is None or kind in ROTATING_TYPES
    wanted = ("milling", "live_tool") if rotating else ("turning", "sub")
    for spindle in spindles:
        if text(spindle, "kind") in wanted:
            return text(spindle, "id", "main") or "main"
    return text(spindles[0], "id", "main") or "main"


def build_tool_table(entries, machine=None) -> ToolTable:
    entries = list(entries or [])
    numbers = []
    groups: dict[int, int] = {}
    sister_of: dict[int, int] = {}
    for index, entry in enumerate(entries):
        m = as_simap(entry) if isinstance(entry, Mapping) and not hasattr(entry, "raw") else entry
        number = plain(get(m, "number"))
        if number is None:
            raise InputError(f"tools.list[{index}] needs its T number")
        path_value = plain(get(m, "path"))
        path = int(path_value) - 1 if path_value is not None else 0
        key = (int(number), path)
        if key in numbers:
            raise InputError(f"tool number {number} is used twice on path {path + 1}")
        numbers.append(key)
        life = get(m, "life")
        group = plain(get(life, "group")) if life is not None else None
        sister = plain(get(life, "sister_of")) if life is not None else None
        if group is not None and int(group) >= 1:
            groups[index] = int(group)
        if sister is not None and int(sister) > 0:
            sister_of[index] = int(sister)
    next_group = max(groups.values(), default=0) + 1
    by_number = {num: i for i, (num, _) in enumerate(numbers)}
    for index in sorted(sister_of):
        target = sister_of[index]
        if target not in by_number:
            raise InputError(f"tools.list[{index}]: sister_of {target} is not in tools.list")
        t_index = by_number[target]
        if t_index not in groups:
            groups[t_index] = next_group
            next_group += 1
        if index in groups and groups[index] != groups[t_index]:
            raise InputError(f"tools.list[{index}]: life.group {groups[index]} contradicts sister_of {target}")
        groups[index] = groups[t_index]
    slots = []
    for index, entry in enumerate(entries):
        m = as_simap(entry) if isinstance(entry, Mapping) and not hasattr(entry, "raw") else entry
        tool_id, holder_id = entry_ids(entry, index)
        spindle = text(m, "spindle") or _default_spindle(machine, _tool_type(m))
        number, path = numbers[index]
        slots.append(ToolSlot(slot=index, number=number, tool_id=tool_id, holder_id=holder_id, path=path,
                              spindle_id=spindle, sister_group=groups.get(index, 0)))
    return ToolTable(tuple(slots))
