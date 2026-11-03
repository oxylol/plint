"""Dialect objects and the dialect registry.

A dialect bundles everything that differs between controls: lexer rules, the word table, the
G-code table with modal groups and power-on codes, the cycle table (which codes route to which
cycle provider), the alarm table, the system-variable map, parameter defaults and the block
handler (the semantics of a block). The executor core (cnctwin.controller.executor) is the
same for every dialect.

``register_dialect(name, builder)``; ``builder(machine, settings, controller_db, db_id) -> Dialect``.
``build_dialect(name, ...)`` looks the builder up; the machine's ``controller.dialect`` is the name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..alarms import AlarmTable
from ..lexer import LexerRules


@dataclass(frozen=True)
class GSpec:
    group: int                      # modal group (0 = one-shot); the number #4001+ reports
    kind: str                       # semantic key the handler dispatches on
    arg: str = ""                   # e.g. "rapid", "XY", "inch", "G54"
    cycle: bool = False             # routed to the cycle hook
    axis_words: bool = False        # a one-shot that consumes the block's axis words


@dataclass
class Dialect:
    name: str                       # "fanuc" | "haas" | ...
    variant: str                    # "M" | "T"
    lexer: LexerRules
    gcodes: dict[str, GSpec]
    power_on: dict[int, str]        # group -> code
    alarms: AlarmTable
    sysvars: object                 # read(ex, n) / write(ex, n, v) / adp(ex, n, v)
    handler_class: type
    params: dict                    # dialect parameter defaults (db/models/controller-<name>.yaml)
    db: dict | None = None
    db_id: str = ""
    supported_g: frozenset | None = None     # G codes the controller database lists (None: no list)
    supported_m: frozenset | None = None
    letters: frozenset = frozenset()
    extra: dict = field(default_factory=dict)
    parser: object = None           # (sources, lexer rules) -> (ProgramStore, problems); None = ISO lexer

    def make_handler(self, executor):
        return self.handler_class(executor, self)


_BUILDERS: dict[str, object] = {}


def register_dialect(name: str, builder, *, replace: bool = False) -> None:
    if name in _BUILDERS and not replace and _BUILDERS[name] is not builder:
        raise ValueError(f"dialect {name!r} is already registered")
    _BUILDERS[name] = builder


def build_dialect(name: str, machine, settings, controller_db: dict | None, db_id: str = "") -> Dialect:
    builder = _BUILDERS.get(name)
    if builder is None:
        raise KeyError(f"no controller dialect {name!r} (registered: {', '.join(sorted(_BUILDERS))})")
    return builder(machine, settings, controller_db, db_id)


def dialect_names() -> list[str]:
    return sorted(_BUILDERS)


def normalize_g(text: str) -> str:
    """'1' -> 'G01', '01' -> 'G01', '54.1' -> 'G54.1', '5.1' -> 'G05.1', '187' -> 'G187'."""
    whole, _, fraction = text.partition(".")
    whole = whole.lstrip("+").lstrip("0") or "0"
    fraction = fraction.rstrip("0")
    code = "G" + whole.zfill(2)
    return code + "." + fraction if fraction else code


def g_from_value(value: float) -> str:
    tenths = int(round(abs(value) * 10.0))
    whole, fraction = divmod(tenths, 10)
    code = "G" + str(whole).zfill(2)
    return code + "." + str(fraction) if fraction else code


_CODE = re.compile(r"^\s*(\d+(?:\.\d+)?)(?:\s*-\s*(\d+))?\s*$")


def word_set(table, prefix: str) -> frozenset | None:
    """Codes a database word table lists: keys like "00", "54.1", "54-59", "110-129", "03-09, 19, 29"."""
    if not isinstance(table, dict) or not table:
        return None
    codes: set[str] = set()
    for key in table:
        for part in str(key).split(","):
            match = _CODE.match(part)
            if not match:
                continue
            first, last = match.group(1), match.group(2)
            if last is None:
                codes.add(normalize_g(first) if prefix == "G" else f"M{int(float(first))}")
            else:
                for number in range(int(first), int(last) + 1):
                    codes.add(normalize_g(str(number)) if prefix == "G" else f"M{number}")
    return frozenset(codes)
