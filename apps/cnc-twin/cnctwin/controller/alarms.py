"""Alarms the controller raises: condition key -> the control's alarm number and text.

The executor never invents alarm text inline. It raises :class:`ControllerAlarm` with a
*condition key* (``"improper_g_code"``, ``"feed_zero"``, ...); the dialect's :class:`AlarmTable`
turns the key into the number and text the real control shows:

1. ``db/controllers/<id>.yaml`` (format ``cnctwin-controller``): an entry whose ``kind`` equals the
   key wins; otherwise the dialect's key -> code map gives the code and the database gives the
   text for that code.
2. When the database file is absent (it is being researched) or lacks the code, the dialect's
   built-in text is used and the alarm is marked ``source: builtin`` (the report lists it).

Keys without a counterpart on the real control (a feature this build does not model, a machine
code the machine definition does not declare) use the ``TW`` namespace (``TW0001`` ...); a
database entry with the same ``kind`` replaces them, so a machine builder's own alarm number
(e.g. the PMC alarm for an undefined M code) can be mapped without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DB_DIR = Path(__file__).resolve().parents[2] / "db" / "controllers"

# Twin alarms (no counterpart number on the real control). Key -> (code, text).
TWIN_ALARMS: dict[str, tuple[str, str]] = {
    "cycle_unavailable": ("TW0001", "CYCLE NOT AVAILABLE IN THIS BUILD"),
    "comp_unavailable": ("TW0002", "COMPENSATION NOT AVAILABLE IN THIS BUILD"),
    "transform_unavailable": ("TW0003", "COORDINATE TRANSFORMATION NOT AVAILABLE IN THIS BUILD"),
    "probe_unavailable": ("TW0004", "PROBING FUNCTION NOT AVAILABLE IN THIS BUILD"),
    "undefined_m_code": ("TW0005", "M CODE NOT DEFINED FOR THIS MACHINE"),
    "not_modelled_variable": ("TW0006", "SYSTEM VARIABLE NOT MODELLED"),
    "interface_signal": ("TW0007", "INTERFACE SIGNAL NOT MODELLED"),
    "wait_never_released": ("TW0008", "PATH WAIT CAN NEVER BE RELEASED"),
    "machine_alarm": ("TW0009", "MACHINE ALARM"),
    "machine_feature_missing": ("TW0010", "MACHINE HAS NO SUCH DEVICE"),
    "operator_input": ("TW0011", "OPERATOR INPUT NOT MODELLED"),
    "unsupported_function": ("TW0012", "FUNCTION NOT SUPPORTED BY THIS BUILD"),
    "tool_not_in_magazine": ("TW0013", "TOOL NUMBER EXCEEDS THE TOOL CHANGER CAPACITY"),
    "g47_character": ("TW0014", "CHARACTER NOT IN THE ENGRAVING FONT"),
    "pocket_definition": ("TW0015", "POCKET DEFINITION ERROR"),
    "lowercase": ("TW0016", "LOWER-CASE ADDRESS"),
    "tool_too_big": ("TW0017", "TOOL TOO BIG FOR THE CYCLE"),
    "arc_under_scaling": ("TW0018", "ARC NOT ALLOWED WITH THIS SCALING"),
    "skip_found": ("TW0019", "SKIP SIGNAL FOUND"),
    "skip_not_found": ("TW0020", "SKIP SIGNAL NOT FOUND"),
    "dwell_missing_p": ("TW0021", "DWELL WITHOUT TIME"),
}


@dataclass(frozen=True)
class AlarmInfo:
    """A resolved alarm: what the control would display, and where the text came from."""

    key: str
    code: str
    text: str
    dialect: str
    detail: str = ""
    source: str = "builtin"          # "db:<controller id>" | "builtin" | "program" (#3000)

    def as_dict(self) -> dict:
        return {"code": self.code, "text": self.text, "dialect": self.dialect, "key": self.key,
                "detail": self.detail, "source": self.source}


class ControllerAlarm(Exception):
    """Raised inside the executor; caught by the interpretation loop, which stops the path."""

    def __init__(self, key: str, detail: str = "", code: str | None = None, text: str | None = None):
        super().__init__(f"{key}: {detail}" if detail else key)
        self.key = key
        self.detail = detail
        self.code = code          # set for program-raised alarms (#3000), which carry their own number
        self.text = text


class AlarmTable:
    """Condition key -> (code, text) for one dialect, with an optional controller database."""

    def __init__(self, dialect: str, key_codes: dict[str, str], builtin_texts: dict[str, str],
                 db: dict | None = None, db_id: str = ""):
        self.dialect = dialect
        self.key_codes = dict(key_codes)
        self.builtin_texts = dict(builtin_texts)
        self.db_id = db_id
        self.by_code: dict[str, dict] = {}
        self.by_kind: dict[str, dict] = {}
        for entry in (db or {}).get("alarms") or []:
            code = str(entry.get("code"))
            self.by_code[code] = entry
            kind = entry.get("kind")
            if kind:
                self.by_kind[str(kind)] = entry
        self.messages = dict((db or {}).get("messages") or {})

    def resolve(self, key: str, detail: str = "") -> AlarmInfo:
        source_db = f"db:{self.db_id}" if self.db_id else "db"
        if key in self.by_kind:
            entry = self.by_kind[key]
            return AlarmInfo(key, str(entry["code"]), str(entry["text"]), self.dialect, detail, source_db)
        code = self.key_codes.get(key)
        if code is None:
            code, text = TWIN_ALARMS.get(key, ("TW0000", "INTERNAL: UNKNOWN ALARM KEY " + key.upper()))
            return AlarmInfo(key, code, text, self.dialect, detail, "twin")
        if code in self.by_code:
            return AlarmInfo(key, code, str(self.by_code[code]["text"]), self.dialect, detail, source_db)
        text = self.builtin_texts.get(code)
        if text is None:
            raise KeyError(f"dialect {self.dialect}: alarm code {code} (key {key}) has no text")
        return AlarmInfo(key, code, text, self.dialect, detail, "builtin")

    def program_alarm(self, number: int, message: str, base: int) -> AlarmInfo:
        """#3000 = n (MESSAGE): number base + n, or the database's ``messages.macro_alarm_format``."""
        fmt = self.messages.get("macro_alarm_format")
        code = fmt.format(n=number, number=base + number) if fmt else str(base + number)
        return AlarmInfo("macro_alarm", code, message, self.dialect, "", "program")

    def has_key(self, key: str) -> bool:
        return key in self.by_kind or key in self.key_codes or key in TWIN_ALARMS


def load_controller_db(controller_id: str | None, directory: Path | None = None) -> dict | None:
    """Plain data of ``db/controllers/<id>.yaml`` with ``extends`` resolved (alarms merged by code,
    the child's entries winning), or None when the file does not exist."""
    if not controller_id:
        return None
    base = directory or DB_DIR
    path = base / f"{controller_id}.yaml"
    if not path.exists():
        return None
    from ..db.yamlio import load_file

    data = load_file(path).data or {}
    parent_id = data.get("extends")
    if parent_id:
        parent = load_controller_db(str(parent_id), base)
        if parent is not None:
            merged_alarms = {str(a.get("code")): a for a in parent.get("alarms") or []}
            for entry in data.get("alarms") or []:
                merged_alarms[str(entry.get("code"))] = entry
            merged = dict(parent)
            merged.update(data)
            merged["alarms"] = list(merged_alarms.values())
            merged["messages"] = {**(parent.get("messages") or {}), **(data.get("messages") or {})}
            return merged
    return data
