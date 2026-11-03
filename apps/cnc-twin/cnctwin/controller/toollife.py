"""Tool life management for the controller core (docs/dialects.md "Cycles and compensation", tool life).

FANUC basic tool life management (fanuc-30ibplus-om-2019 10.5, pp. 265-282) and Haas tool groups:

* **groups** come from the project (``physics.controller.tool_life``) or from G10 L3 data blocks:
  ``G10 L3;`` (delete every group, then register) / ``G10 L3 P1;`` (change the groups given) /
  ``G10 L3 P2;`` (delete the groups given), ``P_ L_ [Q_];`` starts a group (life L in uses, or in minutes
  with Q2), ``T_ [H_] [D_];`` adds a tool (lathes: the T code tool + offset, ``T0101; T0102;`` two offsets
  of tool 1), ``G11;`` ends. PS0149 other addresses, PS0150 group number above ``tool_life_max_groups``,
  PS0152 more than ``tool_life_max_tools`` tools, PS0153 a tool block without T, PS0156 a tool before
  ``P_ L_``. Haas has no G10 L3 (its groups come from the project, the ATM page): TW0012.
* **group call**: M series ``T(ignore number + group)`` (parameter 6810, ``tool_life_ignore_number``;
  ``T <= ignore`` is an ordinary T code, an unknown group too, p. 274); Haas ``T1000``-``T2999``; lathes
  (turret type) ``T(group)99`` selects and starts counting, ``T(group)88`` cancels the offset of the tool
  counted (PS0155 when that is not the group counted), p. 276. The tool is searched from the group's
  current tool toward the last and around; when none has life left, ``tool_life_all_expired``: alarm
  PS5317 ALL TOOL LIFE IS OVER (default) or the last tool (FANUC basic function, p. 281 note 1).
  The selected tool replaces the T number before the magazine / turret checks.
* **counting**: uses - +1 at M06 (lathe: at the T(group)99 call) for the first call of a group since
  reset (p. 280, FANUC); every M06 on Haas (the twin's reading of the Haas tool life page). Time - the
  cutting time of LINEAR/ARC commands while a managed tool is in the spindle, taken from the retirement
  ticks of the commands (the motion domain's clock), so rapids, dwells and stops are not counted
  (p. 280). A tool expires when its count reaches the life value.
* **H99 / D99** (``tool_life_offset_code``) select the H/D registered for the tool counted (PS0154 when
  none). Haas H/D words stay ordinary numbers (a Haas program names the spindle tool, ``H#3026``).
* **expired from the physics**: ``Feedback.tool_life_expired`` (the bus ``wear.expired``) marks those
  tool numbers expired in every group; the next group call takes a sister.

State: ``PathState.sys["tool_life"]`` (JSON, checkpointed with the path). Error bound of time counting:
exact when commands retire one by one (dry run, motion segment ends); otherwise the interval between two
retirements is shared equally by the commands retired together. A feed hold inside a cutting move is
counted (FANUC excludes it). No randomness; no bus fields of its own (``Feedback`` only).
"""

from __future__ import annotations

from ..db import keyunits
from . import hooks
from .alarms import ControllerAlarm, register_alarm_codes
from .cycles.common import params as cycle_params
from .ir import Kind

KEY = "tool_life"

FANUC_ALARMS = {"life_format": "PS0149", "life_group_number": "PS0150", "life_group_not_found": "PS0151",
                "life_too_many_tools": "PS0152", "life_t_missing": "PS0153", "life_not_in_group": "PS0154",
                "life_illegal_t": "PS0155", "life_pl_missing": "PS0156", "life_all_expired": "PS5317"}
FANUC_TEXTS = {"PS0149": "FORMAT ERROR IN G10L3", "PS0150": "ILLEGAL LIFE GROUP NUMBER",
               "PS0151": "GROUP NOT FOUND AT LIFE DATA", "PS0152": "OVER MAXIMUM TOOL NUMBER",
               "PS0153": "T-CODE NOT FOUND", "PS0154": "NOT USING TOOL IN LIFE GROUP",
               "PS0155": "ILLEGAL T-CODE COMMAND", "PS0156": "P/L COMMAND NOT FOUND",
               "PS5317": "ALL TOOL LIFE IS OVER"}
HAAS_ALARMS = {"life_format": "TW0028", "life_group_number": "TW0028", "life_group_not_found": "TW0033",
               "life_too_many_tools": "TW0028", "life_t_missing": "TW0028", "life_not_in_group": "TW0035",
               "life_illegal_t": "TW0028", "life_pl_missing": "TW0028", "life_all_expired": "TW0029"}
HAAS_TEXTS = {"TW0028": "TOOL LIFE DATA FORMAT ERROR", "TW0029": "ALL TOOLS OF THE TOOL GROUP EXPIRED",
              "TW0033": "TOOL GROUP NOT DEFINED", "TW0035": "NO TOOL OF A TOOL GROUP IN THE SPINDLE"}


# --------------------------------------------------------------------------- state

def _new() -> dict:
    return {"groups": {}, "selected": None, "current": None, "counted": [], "calls": {}, "spindle": None,
            "changes": {}, "last_tick": None, "g10": None}


def _life(entry: dict) -> tuple[str, float]:
    if "life_uses" in entry:
        return "uses", float(entry["life_uses"])
    for key, value in entry.items():
        if str(key).startswith("life_"):
            name, si = keyunits.to_si(str(key), value)
            if name != "life_s":
                raise ControllerAlarm("life_format", f"tool life {key}: give life_uses or a time (life_min, life_s)")
            return "time", float(si)
    raise ControllerAlarm("life_format", f"tool life group {entry.get('group')}: no life_uses / life_min")


def _used(entry: dict) -> float:
    for key, value in entry.items():
        if str(key) == "used_uses":
            return float(value)
        if str(key).startswith("used_"):
            return float(keyunits.to_si(str(key), value)[1])
    return 0.0


def _tool_entry(ex, item, lathe: bool) -> dict:
    if isinstance(item, dict):
        if "t" not in item:
            raise ControllerAlarm("life_t_missing", f"tool life tool {item}")
        number = int(item["t"])
        tool = {"h": int(item.get("h", number)), "d": int(item.get("d", number)), "used": _used(item)}
    else:
        number = int(item)
        tool = {"used": 0.0}
    if lathe:
        station, offset = divmod(number, 100)
        tool.update(t=station, offsets=[offset])
    else:
        tool.setdefault("h", number)
        tool.setdefault("d", number)
        tool["t"] = number
    tool["expired"] = False
    return tool


def _add_tool(group: dict, tool: dict, lathe: bool, p: dict) -> None:
    tools = group["tools"]
    if lathe and tools and tools[-1]["t"] == tool["t"]:
        tools[-1]["offsets"].extend(tool["offsets"])          # T0101; T0102: two offsets of tool 1
        return
    if len(tools) >= int(p["tool_life_max_tools"]):
        raise ControllerAlarm("life_too_many_tools", f"more than {int(p['tool_life_max_tools'])} tools in a group")
    tools.append(tool)


def _groups_from_settings(ex) -> dict:
    data = ex.p.get("tool_life") or {}
    if not isinstance(data, dict):
        raise ControllerAlarm("life_format", "physics.controller.tool_life must be a mapping with 'groups'")
    p = cycle_params(ex)
    lathe = _turret(ex)
    groups = {}
    for entry in data.get("groups") or []:
        if "group" not in entry:
            raise ControllerAlarm("life_format", f"physics.controller.tool_life group without 'group': {entry}")
        number = int(entry["group"])
        _check_group(ex, number)
        kind, life = _life(entry)
        group = {"count": kind, "life": life, "tools": [], "index": 0}
        for item in entry.get("tools") or []:
            tool = _tool_entry(ex, item, lathe)
            tool["expired"] = tool["used"] >= life
            _add_tool(group, tool, lathe, p)
        groups[str(number)] = group
    unknown = set(data) - {"groups"}
    if unknown:
        raise ControllerAlarm("life_format", f"physics.controller.tool_life: unknown keys {sorted(unknown)}")
    return groups


def state(ex, create: bool = True) -> dict | None:
    """The tool life state of a path, created from the project's groups on first use (``create=False``:
    None while neither the project nor a G10 L3 block has defined groups)."""
    s = ex.s
    st = s.sys.get(KEY)
    if st is None:
        if not create and not ex.p.get("tool_life"):
            return None
        st = _new()
        st["groups"] = _groups_from_settings(ex)
        s.sys[KEY] = st
    return st


def _haas(ex) -> bool:
    return ex.dialect.name == "haas"


def _turret(ex) -> bool:
    """FANUC lathe of the turret type: T(group)99 / T(group)88 (the twin's lathes have turrets)."""
    return bool(getattr(ex.handler, "T", False)) and not _haas(ex)


def _check_group(ex, number: int) -> None:
    p = cycle_params(ex)
    if _haas(ex):
        if not int(p["haas_tool_group_first"]) <= number <= int(p["haas_tool_group_last"]):
            raise ControllerAlarm("life_group_number", f"tool group {number}")
    elif not 1 <= number <= int(p["tool_life_max_groups"]):
        raise ControllerAlarm("life_group_number", f"tool group {number}")


# --------------------------------------------------------------------------- the provider

class ToolLife:
    """The ``TOOL_LIFE`` provider: group calls, sister selection, counting, H99/D99, expiry."""

    # ---------------------------------------------------------------- selection
    def _pick(self, ex, st: dict, key: str) -> int:
        group = st["groups"][key]
        tools = group["tools"]
        if not tools:
            raise ControllerAlarm("life_group_not_found", f"tool group {key} has no tools")
        start = int(group.get("index", 0)) % len(tools)
        if group["count"] == "uses" and key in st["counted"] and not _haas(ex):
            return start                            # once per run: the same tool until reset (p. 280 CAUTION)
        for k in range(len(tools)):
            index = (start + k) % len(tools)
            if not tools[index]["expired"]:
                group["index"] = index
                return index
        if str(cycle_params(ex)["tool_life_all_expired"]) == "last":
            ex.note("warning", f"tool group {key}: every tool expired, the last one is used (tool change signal)")
            group["index"] = len(tools) - 1
            return len(tools) - 1
        raise ControllerAlarm("life_all_expired", f"tool group {key}")

    def _count_use(self, ex, st: dict, key: str, index: int, every_call: bool) -> None:
        group = st["groups"][key]
        if group["count"] != "uses" or (key in st["counted"] and not every_call):
            return
        if key not in st["counted"]:
            st["counted"].append(key)
        tool = group["tools"][index]
        tool["used"] += 1.0
        if tool["used"] >= group["life"]:
            tool["expired"] = True

    def t_code(self, ex, number: int) -> int:
        """A T code: a group call returns the tool (lathe: T code) chosen; other T codes are returned as given."""
        st = state(ex, create=False)
        if st is None or not st["groups"]:
            return number
        p = cycle_params(ex)
        if _turret(ex):
            return self._lathe_call(ex, st, number, p)
        if _haas(ex):
            if not int(p["haas_tool_group_first"]) <= number <= int(p["haas_tool_group_last"]):
                return number
            key = str(number)
            if key not in st["groups"]:
                raise ControllerAlarm("life_group_not_found", f"T{number}")
        else:
            ignore = int(round(float(p["tool_life_ignore_number"])))
            if number <= ignore:
                return number
            group = number - ignore
            if group > int(p["tool_life_max_groups"]):
                raise ControllerAlarm("life_group_number", f"T{number}: group {group}")
            key = str(group)
            if key not in st["groups"]:
                return number                       # FANUC: an unknown group is an ordinary T code (p. 274)
        index = self._pick(ex, st, key)
        st["selected"] = {"group": key, "index": index}
        return int(st["groups"][key]["tools"][index]["t"])

    def _lathe_call(self, ex, st: dict, number: int, p: dict) -> int:
        group_number, code = divmod(number, 100)
        if code == 88:
            current = st["current"]
            if current is None or current["group"] != str(group_number):
                raise ControllerAlarm("life_illegal_t", f"T{number}: group {group_number} is not being counted")
            return int(st["groups"][current["group"]]["tools"][current["index"]]["t"]) * 100
        if code != 99:
            st["current"] = None                    # an ordinary T code ends the life counting
            return number
        key = str(group_number)
        if key not in st["groups"]:
            raise ControllerAlarm("life_group_not_found", f"T{number}")
        index = self._pick(ex, st, key)
        self._count_use(ex, st, key, index, every_call=False)
        st["current"] = {"group": key, "index": index}
        calls = st["calls"].get(key, 0)
        st["calls"][key] = calls + 1
        tool = st["groups"][key]["tools"][index]
        offsets = tool["offsets"]
        return int(tool["t"]) * 100 + int(offsets[min(calls, len(offsets) - 1)])

    def select(self, ex, number: int) -> int:
        """Called at the tool change (M06; lathe: the turret index) with the tool about to be used."""
        st = state(ex, create=False)
        if st is None or not st["groups"]:
            if number in ex.s.expired_tools:
                ex.note("warning", f"T{number} called after its tool life expired (no tool group)")
            return number
        if _turret(ex):
            current = st["current"]
            if current is not None:
                st["changes"][str(ex.s.block_seq)] = dict(current)
            return number
        selected = st["selected"]
        if selected is not None and int(st["groups"][selected["group"]]["tools"][selected["index"]]["t"]) == number:
            st["current"] = dict(selected)
            st["selected"] = None
            self._count_use(ex, st, selected["group"], selected["index"], every_call=_haas(ex))
            st["changes"][str(ex.s.block_seq)] = dict(selected)
        else:
            st["current"] = None
            st["selected"] = None
            if number in ex.s.expired_tools:
                ex.note("warning", f"T{number} called after its tool life expired (not called through its group)")
        return number

    # ---------------------------------------------------------------- offsets
    def offset_code(self, ex, kind: str, number: int) -> int:
        st = state(ex, create=False)
        if st is None or not st["groups"]:
            return number
        current = st["current"]
        if _haas(ex) or _turret(ex) or number != int(cycle_params(ex)["tool_life_offset_code"]):
            return number                           # Haas: ordinary H/D (G43 H#3026 takes the spindle tool)
        if current is None:
            raise ControllerAlarm("life_not_in_group", f"{kind}{number} without a tool of a life group")
        tool = st["groups"][current["group"]]["tools"][current["index"]]
        return int(tool["h" if kind == "H" else "d"])

    # ---------------------------------------------------------------- counting from the machine
    def expired(self, ex, numbers: list[int]) -> None:
        st = state(ex, create=False)
        if st is None:
            return                                  # no groups: the core keeps PathState.expired_tools
        wanted = {int(n) for n in numbers}
        for group in st["groups"].values():
            for tool in group["tools"]:
                if int(tool["t"]) in wanted:
                    tool["expired"] = True

    def retired(self, ex, done: list, tick: int) -> None:
        st = ex.s.sys.get(KEY)
        if st is None or not st["groups"]:
            return
        last = st["last_tick"]
        st["last_tick"] = int(tick)
        share = (int(tick) - int(last)) * ex.tick_s / len(done) if last is not None and done else 0.0
        for cmd in done:
            if cmd.kind == Kind.TOOL_CHANGE:
                st["spindle"] = st["changes"].pop(str(cmd.block_seq), None)
                continue
            spindle = st["spindle"]
            if spindle is None or share <= 0.0 or cmd.kind not in (Kind.LINEAR, Kind.ARC):
                continue
            group = st["groups"].get(spindle["group"])
            if group is None or group["count"] != "time" or spindle["index"] >= len(group["tools"]):
                continue
            tool = group["tools"][spindle["index"]]
            tool["used"] += share
            if tool["used"] >= group["life"]:
                tool["expired"] = True

    def reset(self, ex) -> None:
        """Reset / next run: group calls count again and multi-offset sequences restart (p. 280, p. 276)."""
        st = ex.s.sys.get(KEY)
        if st is None:
            return
        st["counted"] = []
        st["calls"] = {}
        st["selected"] = None


class LifeData:
    """``G10 L3`` ... ``G11`` tool life data blocks (``DATA_MODES["G10L3"]``)."""

    def begin(self, ex, p, bw) -> None:
        if _haas(ex):
            raise ControllerAlarm("unsupported_function", "G10 L3: Haas tool groups come from the project's tool_life")
        st = state(ex)
        mode = {None: "register", 0: "register", 1: "change", 2: "delete"}.get(p)
        if mode is None:
            raise ControllerAlarm("life_format", f"G10 L3 P{p}")
        self._leftover(bw)
        if mode == "register":
            st["groups"] = {}
        st["g10"] = {"mode": mode, "group": None}

    def _leftover(self, bw) -> None:
        extra = [letter for letter in bw.addr if letter not in bw.used]
        codes = [g for g in bw.g if g not in ("G10", "G11")] + [f"M{m}" for m in bw.m]
        if extra or codes:
            raise ControllerAlarm("life_format", "address " + " ".join(extra + codes) + " in tool life data")

    def block(self, ex, bw) -> None:
        st = state(ex)
        h = ex.handler
        data = st["g10"] or {"mode": "register", "group": None}
        st["g10"] = data
        p_word, t_word = bw.take("P"), bw.take("T")
        if p_word is not None:
            number = h.integer(p_word)
            _check_group(ex, number)
            key = str(number)
            if data["mode"] == "delete":
                st["groups"].pop(key, None)
                self._leftover(bw)
                return
            l_word, q_word = bw.take("L"), bw.take("Q")
            if l_word is None:
                raise ControllerAlarm("life_pl_missing", f"P{number} without L")
            q = h.integer(q_word) if q_word is not None else None
            if q not in (None, 1, 2):
                raise ControllerAlarm("life_format", f"Q{q}")
            params = cycle_params(ex)
            kind = {1: "uses", 2: "time"}.get(q, str(params["tool_life_count_default"]))
            value = float(h.integer(l_word))
            life = value if kind == "uses" else value * 60.0        # L in minutes (parameter 6805#1 FGL = 0)
            st["groups"][key] = {"count": kind, "life": life, "tools": [], "index": 0}
            data["group"] = key
            self._leftover(bw)
            return
        if data["mode"] == "delete":
            raise ControllerAlarm("life_format", "only P_ blocks between G10 L3 P2 and G11")
        if data["group"] is None:
            raise ControllerAlarm("life_pl_missing", "tool data before P_ L_")
        if t_word is None:
            raise ControllerAlarm("life_t_missing", "tool life data block without T")
        item = {"t": h.integer(t_word)}
        for letter in ("H", "D"):
            word = bw.take(letter)
            if word is not None:
                if _turret(ex) and letter == "H":
                    raise ControllerAlarm("life_format", "H in lathe tool life data")
                item[letter.lower()] = h.integer(word)
        self._leftover(bw)
        lathe = _turret(ex)
        tool = _tool_entry(ex, item, lathe)
        _add_tool(st["groups"][data["group"]], tool, lathe, cycle_params(ex))

    def end(self, ex) -> None:
        st = state(ex)
        st["g10"] = None


TOOL_LIFE = ToolLife()
LIFE_DATA = LifeData()


def register(replace: bool = True) -> None:
    """(Re-)register the tool life provider, the G10 L3 data mode and the alarm numbers."""
    hooks.register_tool_life(TOOL_LIFE, replace=replace)
    hooks.register_data_mode("G10L3", LIFE_DATA, replace=replace)
    register_alarm_codes("fanuc", FANUC_ALARMS, FANUC_TEXTS)
    register_alarm_codes("haas", HAAS_ALARMS, HAAS_TEXTS)


register()

__all__ = ["ToolLife", "LifeData", "TOOL_LIFE", "LIFE_DATA", "state", "register"]
