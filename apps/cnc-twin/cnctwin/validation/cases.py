"""Validation cases as the suite reads them (docs/validation.md "Cases").

Cases are loaded through :mod:`cnctwin.db` (schema ``validation_case``, ``Database.validation_case``
and ``Database.ids("validation_case")``); :class:`Case` adds what the harnesses and the report need on
top of the SIMap: the plain values of the setup with their sources, the inputs the case marks as
assumed, per-test condition tables and the conversion of the reference units.

Setup leaves are written either as plain values or as ``{value, src, note, assumed}``; a leaf counts
as assumed when its ``src`` is ``est``, when ``assumed`` is true, or when its key is listed in an
``assumed_inputs`` list of the same section. Reference units are the free texts of the case files
(``um``, ``m/min``, ``degC``...), converted by :func:`cnctwin.db.keyunits.parse_unit_text`; ``index`` and
an empty unit are labels / dimensionless.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..db import keyunits
from ..units import REFERENCE_TEMP_K

LABEL_UNITS = ("index", "count", "test", "-", "")


# ------------------------------------------------------------------------------------------ units
def unit_of(text) -> keyunits.Unit | None:
    """Unit of a reference unit text; ``None`` for dimensionless values and labels."""
    if text is None:
        return None
    raw = str(text).strip()
    if raw.lower() in LABEL_UNITS or raw == "None":
        return None
    return keyunits.parse_unit_text(raw)


def to_si(value: float, unit_text) -> float:
    """A reference value in SI (temperatures absolute, in kelvin)."""
    unit = unit_of(unit_text)
    return float(value) if unit is None else unit.to_si(float(value))


def from_si(value_si: float, unit_text) -> float:
    unit = unit_of(unit_text)
    return float(value_si) if unit is None else unit.from_si(float(value_si))


def difference_to_si(delta: float, unit_text) -> float:
    """A difference (an uncertainty) in SI: no offset, only the factor (degC differences are kelvin)."""
    unit = unit_of(unit_text)
    return float(delta) if unit is None else float(delta) * unit.factor


def is_temperature_unit(unit_text) -> bool:
    unit = unit_of(unit_text)
    return unit is not None and unit.is_temperature


# ------------------------------------------------------------------------------------------ leaves
@dataclass(frozen=True)
class Leaf:
    """One setup value with its provenance in the case file."""

    path: str                       # e.g. "setup.tool.edge_radius_um"
    key: str                        # the file key (unit as its last token)
    value: Any                      # plain value as written (unwrapped from {value: ...})
    src: str = ""
    note: str = ""
    assumed: bool = False

    @property
    def unit(self) -> keyunits.Unit | None:
        return keyunits.parse_key(self.key)[1]

    def si(self):
        """The value in SI by the key's unit token (numbers and lists of numbers)."""
        return keyunits.convert(self.value, self.unit)


def unwrap(raw) -> tuple[Any, str, str, bool]:
    """(value, src, note, assumed flag) of a leaf written plainly or as {value, src, note, assumed}."""
    if isinstance(raw, Mapping) and "value" in raw:
        src = str(raw.get("src") or "")
        note = str(raw.get("note") or "")
        assumed = bool(raw.get("assumed")) or src == "est"
        return raw.get("value"), src, note, assumed
    return raw, "", "", False


def _plain(value):
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


# ------------------------------------------------------------------------------------------ case
@dataclass
class Case:
    """One validation case: the database object plus plain access to its parts."""

    id: str
    quantity: str
    file: str                                  # path relative to the repository root
    raw: dict                                  # the case file (as written, merged)
    simap: Any = None                          # cnctwin.db SIMap of the case (SI view of declared keys)
    problems: list = field(default_factory=list)

    # -------------------------------------------------------------- construction
    @classmethod
    def from_db(cls, db, case_id: str) -> "Case":
        obj = db.object("validation_case", case_id)
        simap = db.validation_case(case_id)
        raw = _plain(obj.merged if getattr(obj, "merged", None) is not None else obj.raw)
        return cls(id=case_id, quantity=str(raw.get("quantity")), file=str(obj.rel), raw=raw, simap=simap,
                   problems=[str(p.message) for p in (db.problems_of("validation_case", case_id) or [])])

    # -------------------------------------------------------------- parts
    @property
    def reference(self) -> dict:
        return dict(self.raw.get("reference") or {})

    @property
    def src(self) -> str:
        return str(self.reference.get("src") or "")

    @property
    def location(self) -> str:
        return str(self.reference.get("location") or "")

    @property
    def measured(self) -> bool:
        return bool(self.reference.get("measured", True))

    @property
    def digitised(self) -> bool:
        return bool(self.reference.get("digitised", False))

    @property
    def uncertainty_pct(self) -> float | None:
        value = self.reference.get("uncertainty_pct")
        return None if value is None else float(value)

    @property
    def setup(self) -> dict:
        return dict(self.raw.get("setup") or {})

    @property
    def operation(self) -> str:
        return str(self.setup.get("operation") or "")

    @property
    def material_id(self) -> str | None:
        material = self.setup.get("material") or {}
        return str(material.get("db")) if isinstance(material, Mapping) and material.get("db") else None

    @property
    def reference_values(self) -> list[dict]:
        return [dict(v) for v in (self.raw.get("reference_values") or [])]

    @property
    def reference_curves(self) -> list[dict]:
        return [dict(c) for c in (self.raw.get("reference_curves") or [])]

    @property
    def notes(self) -> str:
        return str(self.raw.get("notes") or "")

    def section(self, name: str) -> dict:
        value = self.setup.get(name)
        return dict(value) if isinstance(value, Mapping) else {}

    # -------------------------------------------------------------- leaves
    def leaf(self, section: str, *keys: str, default=None) -> Leaf | None:
        """The first of ``keys`` present in ``setup.<section>`` (None when none is)."""
        sec = self.section(section)
        assumed_keys = set(sec.get("assumed_inputs") or [])
        for key in keys:
            if key in sec and sec[key] is not None:
                value, src, note, assumed = unwrap(sec[key])
                if value is None:
                    continue
                return Leaf(f"setup.{section}.{key}", key, value, src, note, assumed or key in assumed_keys)
        if default is not None:
            return Leaf(f"setup.{section}.{keys[0]}", keys[0], default, "", "", True)
        return None

    def value(self, section: str, *keys: str, default=None):
        leaf = self.leaf(section, *keys)
        return default if leaf is None else leaf.value

    def si(self, section: str, *keys: str, default=None):
        """SI value of the first present key (converted by its unit token)."""
        leaf = self.leaf(section, *keys)
        return default if leaf is None else leaf.si()

    def assumed_inputs(self) -> list[Leaf]:
        """Every setup leaf the case marks as assumed (src: est, assumed: true or assumed_inputs)."""
        out: list[Leaf] = []
        for name, sec in self.setup.items():
            if not isinstance(sec, Mapping):
                continue
            listed = set(sec.get("assumed_inputs") or [])
            for key in sorted(sec):
                if key == "assumed_inputs":
                    continue
                value, src, note, assumed = unwrap(sec[key])
                if assumed or key in listed:
                    out.append(Leaf(f"setup.{name}.{key}", key, value, src, note, True))
        return out


# ------------------------------------------------------------------------------------------ per-test tables
PER_TEST = re.compile(r"^(?P<stem>.+?)_per_(?:test|curve)(?:_(?P<unit>.+))?$")


def _per_test_key(key: str) -> tuple[str, str] | None:
    """('vc_m_min', 'vc') for 'vc_per_test_m_min'; None for other keys."""
    m = PER_TEST.match(key)
    if not m:
        return None
    stem, unit = m.group("stem"), m.group("unit")
    return (f"{stem}_{unit}" if unit else stem), stem


@dataclass
class TestRow:
    """The conditions of one test of a case (file keys with their units, plain values)."""

    index: int
    test_id: str
    values: dict                       # key (with unit token) -> plain value
    sources: dict                      # key -> 'setup.<section>.<file key>'

    def si(self, *keys: str, default=None):
        for key in keys:
            if key in self.values and self.values[key] is not None:
                return keyunits.convert(self.values[key], keyunits.parse_key(key)[1])
        return default

    def raw(self, *keys: str, default=None):
        for key in keys:
            if key in self.values and self.values[key] is not None:
                return self.values[key]
        return default


def per_test_table(case: Case, sections: tuple[str, ...] = ("conditions", "tool", "stock", "coolant")) -> list[TestRow]:
    """Per-test rows: ``*_per_test_<unit>`` lists spread over the tests (``test_ids`` names them),
    scalar leaves shared by every test. A case without per-test lists has one row."""
    shared: dict = {}
    shared_src: dict = {}
    lists: dict = {}
    list_src: dict = {}
    ids = None
    for name in sections:
        sec = case.section(name)
        for key in sorted(sec):
            if key in ("assumed_inputs", "src", "page", "note"):
                continue
            value, _, _, _ = unwrap(sec[key])
            if key == "test_ids":
                ids = [str(v) for v in value]
                continue
            per = _per_test_key(key)
            if per is not None and isinstance(value, (list, tuple)):
                lists[per[0]] = list(value)
                list_src[per[0]] = f"setup.{name}.{key}"
            elif key not in shared:
                shared[key] = value
                shared_src[key] = f"setup.{name}.{key}"
    n = max([len(v) for v in lists.values()] + [len(ids) if ids else 0, 1])
    rows = []
    for i in range(n):
        values = dict(shared)
        sources = dict(shared_src)
        for key, seq in lists.items():
            if i < len(seq):
                values[key] = seq[i]
                sources[key] = list_src[key]
        rows.append(TestRow(i, ids[i] if ids and i < len(ids) else str(i + 1), values, sources))
    return rows


def name_tokens(name: str) -> list[str]:
    """Tokens of a reference name ("chip_ratio V0072 (derived)" -> ["chip_ratio", "V0072", "derived"])."""
    return [t for t in re.split(r"[\s(),;:]+", str(name)) if t]


def find_test(rows: list[TestRow], entry: Mapping, x_keys: tuple[str, ...] = ()) -> TestRow | None:
    """The test a reference entry belongs to: a test id among the tokens of its name ("V0072",
    "test 3", "A1"), else the row whose value of one of ``x_keys`` equals the entry's x (in the
    entry's x unit)."""
    tokens = name_tokens(entry.get("name", ""))
    ids = {r.test_id: r for r in rows}
    lowered = [t.lower() for t in tokens]
    for k, token in enumerate(tokens):
        if token in ids and len(rows) > 1:
            if token.isdigit() and not (k > 0 and lowered[k - 1] == "test"):
                continue
            return ids[token]
    x = entry.get("x")
    if x is not None and x_keys:
        x_si = to_si(float(x), entry.get("x_unit"))
        for row in rows:
            for key in x_keys:
                value = row.si(key)
                if isinstance(value, (int, float)) and math.isclose(float(value), x_si, rel_tol=1e-6, abs_tol=1e-12):
                    return row
    return rows[0] if len(rows) == 1 else None


def ambient_K(case: Case, default_K: float = REFERENCE_TEMP_K) -> tuple[float, bool]:
    """(ambient temperature, assumed flag): the case's ``ambient_temp_degC`` (conditions), else 20 degC."""
    leaf = case.leaf("conditions", "ambient_temp_degC", "ambient_degC")
    if leaf is None or not isinstance(leaf.value, (int, float)):
        return default_K, True
    return float(leaf.si()), leaf.assumed


def load_cases(db, quantities=None, case_ids=None) -> list[Case]:
    """Cases of the database in (quantity, id) order, filtered by quantity and id."""
    wanted_q = set(quantities) if quantities else None
    wanted_id = set(case_ids) if case_ids else None
    out = []
    for case_id in db.ids("validation_case"):
        if wanted_id is not None and case_id not in wanted_id:
            continue
        case = Case.from_db(db, case_id)
        if wanted_q is not None and case.quantity not in wanted_q:
            continue
        out.append(case)
    out.sort(key=lambda c: (c.quantity, c.id))
    return out


__all__ = ["Case", "Leaf", "TestRow", "per_test_table", "find_test", "name_tokens", "load_cases", "unit_of", "to_si",
           "from_si", "difference_to_si", "is_temperature_unit", "ambient_K", "unwrap"]
