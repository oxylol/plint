"""A small declared-schema language for database and project files (contract v2).

A schema is a tree of nodes (:class:`Struct`, :class:`Seq`, :class:`MapOf`, :class:`Tagged`,
:class:`OneKey`, :class:`ByType`, :class:`Qty`, :class:`Pairs`, :class:`Pose`, :class:`Count`,
:class:`Text`, :class:`Flag`, :class:`Ref`, :class:`FilePath`, :class:`Any_`). Every node can

* ``check(value, path, unit, ctx)``: add a :class:`~cnctwin.db.problems.Problem` (file and line from
  the :class:`~cnctwin.db.yamlio.Locations`) for every unknown key, key with the wrong unit,
  missing required key, wrong type, insane value or broken cross-field rule (nothing is coerced);
* ``build(value, unit, src, si_key)``: the SI view (:mod:`cnctwin.db.simap`) of a checked value.

Rules (docs/architecture.md 3, docs/formats.md "Databases"):

* A :class:`Struct` declares its keys exactly. The unit of a field is its key's last token
  (:mod:`cnctwin.db.keyunits`) or given explicitly (``Qty(unit="1_s")``); the SI key replaces the
  unit token by the SI one (``travel_mm`` -> ``travel_m``). A file key with the stem of a declared
  field but another unit (``travel_in`` for ``travel_mm``) is an error that says so; older key forms
  are renamed by :mod:`cnctwin.db.migrate` before checking, never accepted as synonyms.
* A unit-less key under a unit key inherits its unit (``fz_mm: {min, start, max}``,
  ``rapid_mm_min: {X: ..}``).
* Every leaf also accepts the sourced form ``{value: x, src: key, note: .., range: [lo, hi],
  digitised: true, uncertainty_pct: n, page: ..}``; a :class:`Qty` declared with ``table=`` also
  the table ``{table: [[x, v], ...], x: <unit>}`` (``x`` defaults to degC only for temperature
  tables). Two-column curves are :class:`Pairs` with their x unit declared at their path.
* ``note`` and ``notes`` are free text anywhere and are never checked.

Ranges are given in SI and checked on the converted value.
"""

from __future__ import annotations

import difflib
import math
import re
from collections.abc import Mapping

from ..units import ZERO_DEGC_K
from . import keyunits
from .keyunits import Unit
from .problems import Problem, format_path
from .simap import (CITATION_KEYS, NOTE_KEYS, ROTATION_ORDERS, Curve, PoseMap, SIMap, Sourced, Table, infer,
                    register_curve)

# Citation keys that may sit on any section (they describe its section-level src).
SECTION_META = ("src", "page", "pages", "digitised", "uncertainty_pct")
SOURCED_META = ("value", "src", "note", "notes", "range", "digitised", "uncertainty_pct", "page", "pages")

ID_PATTERN = r"[a-z0-9][a-z0-9._+-]*"


class Ctx:
    """State of one check: where problems go and what was referenced."""

    def __init__(self, locations=None, obj: str = "", file: str = ""):
        self.locations = locations
        self.obj = obj
        self.file = file
        self.problems: list[Problem] = []
        self.refs: list[tuple[str, str, tuple]] = []      # (kind, id, path)
        self.paths: list[tuple[tuple, str, object]] = []  # (path, text, FilePath node)

    def add(self, path, message: str, kind: str = "rule", severity: str = "error") -> None:
        if self.locations is not None:
            file, line = self.locations.get(path)
        else:
            file, line = self.file, 0
        self.problems.append(Problem(file or self.file, line, format_path(path), message, kind, severity,
                                     self.obj))

    def warn(self, path, message: str, kind: str = "rule") -> None:
        self.add(path, message, kind, "warning")


# --------------------------------------------------------------------------- value helpers

def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_number_like(value) -> bool:
    if is_number(value):
        return True
    if isinstance(value, (list, tuple)):
        return all(is_number_like(v) for v in value)
    return False


def unwrap(value):
    """The plain value of a possibly sourced entry."""
    if isinstance(value, Mapping) and "value" in value:
        return value.get("value")
    return value


def pick(mapping, key: str):
    """``(key, value)`` if ``mapping`` has ``key`` (exact), else ``(None, None)``."""
    if isinstance(mapping, Mapping) and key in mapping:
        return key, mapping[key]
    return None, None


def has(mapping, key: str) -> bool:
    return isinstance(mapping, Mapping) and unwrap(mapping.get(key)) is not None


def get(mapping, key: str):
    return unwrap(mapping.get(key)) if isinstance(mapping, Mapping) else None


def si_number(mapping, key: str, parent_unit: Unit | None = None):
    """SI value of a numeric entry (sourced or plain; a temperature table gives its 20 degC value)."""
    if not isinstance(mapping, Mapping) or key not in mapping:
        return None
    value = mapping[key]
    _, own = keyunits.parse_key(key)
    unit_ = own or parent_unit
    if isinstance(value, Mapping) and "table" in value:
        rows = value.get("table")
        if not isinstance(rows, list) or not rows or value.get("x", "degC") not in ("degC", None):
            return None
        try:
            temps = [float(r[0]) for r in rows]
            values = [float(r[1]) for r in rows]
        except (TypeError, ValueError, IndexError):
            return None
        reference = 20.0
        if reference <= temps[0]:
            raw = values[0]
        elif reference >= temps[-1]:
            raw = values[-1]
        else:
            raw = values[0]
            for i in range(1, len(temps)):
                if temps[i] >= reference:
                    t0, t1 = temps[i - 1], temps[i]
                    raw = values[i - 1] + (values[i] - values[i - 1]) * (reference - t0) / (t1 - t0)
                    break
        return keyunits.convert(raw, unit_)
    raw = unwrap(value)
    if is_number_like(raw):
        return keyunits.convert(raw, unit_)
    return None


def _show_bound(bound_si: float, unit_: Unit | None) -> str:
    """An SI bound shown in the unit of the file key."""
    if unit_ is None or unit_.scale:
        return f"{bound_si:g}"
    return f"{unit_.from_si(bound_si):.6g}"


def _describe(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return f"{str(value).lower()} (a boolean)"
    if isinstance(value, str):
        text = value if len(value) <= 40 else value[:37] + "..."
        return f"text {text!r}"
    if is_number(value):
        return f"the number {value}"
    if isinstance(value, list):
        return f"a list of {len(value)}"
    if isinstance(value, Mapping):
        return "a mapping"
    return type(value).__name__


def check_citation(mapping, path, ctx: Ctx) -> None:
    """Types of the citation keys of a section or sourced value."""
    if "src" in mapping and not isinstance(mapping["src"], str):
        ctx.add(path + ("src",), "src must be a bibliography key (text) or est", "type")
    for key in ("page", "pages"):
        if key in mapping and not (isinstance(mapping[key], str) or is_number(mapping[key])):
            ctx.add(path + (key,), f"{key} must be text", "type")
    if "digitised" in mapping:
        if not isinstance(mapping["digitised"], bool):
            ctx.add(path + ("digitised",), "digitised must be true or false", "type")
        elif mapping["digitised"] and "uncertainty_pct" not in mapping:
            ctx.add(path, "a digitised value needs uncertainty_pct", "rule")
    if "uncertainty_pct" in mapping:
        u = mapping["uncertainty_pct"]
        if not is_number(u) or u < 0 or u > 1000:
            ctx.add(path + ("uncertainty_pct",), "uncertainty_pct must be a number in [0, 1000]", "range")
    for key in NOTE_KEYS:
        if key in mapping:
            note = mapping[key]
            if not (note is None or isinstance(note, str)
                    or (isinstance(note, list) and all(isinstance(n, str) for n in note))):
                ctx.add(path + (key,), f"{key} must be text (or a list of text)", "type")


def _citations(raw: Mapping) -> dict:
    return {k: raw[k] for k in CITATION_KEYS if k in raw and k not in ("src", "note")}


# --------------------------------------------------------------------------- nodes

class Node:
    nullable: bool = False
    doc: str = ""
    takes_unit: bool = False       # whether the field key's unit applies to this node
    unit_token: str | None = None  # explicit unit (overrides the key's last token)
    si_key_override: str | None = None

    def check(self, value, path, unit, ctx: Ctx) -> None:  # pragma: no cover - interface
        raise TypeError("abstract node")

    def build(self, value, unit, src, si_key):
        return value

    def _null(self, value, path, ctx) -> bool:
        """True when the value is null (and reported if null is not allowed)."""
        if value is None:
            if not self.nullable:
                ctx.add(path, "must not be null", "type")
            return True
        return False


class Any_(Node):
    """Anything (free-form data, e.g. the setup of a validation case); built by suffix inference."""

    takes_unit = True

    def __init__(self, doc: str = ""):
        self.nullable = True
        self.doc = doc

    def check(self, value, path, unit, ctx):
        return

    def build(self, value, unit, src, si_key):
        if isinstance(value, Mapping):
            return SIMap(value, src, unit)
        if isinstance(value, list):
            return [self.build(v, unit, src, si_key) for v in value]
        return keyunits.convert(value, unit) if is_number_like(value) else value


class Raw(Node):
    """Free-form data kept exactly as written (``extensions`` sections): not checked beyond being a
    mapping, not converted (the SI view holds the raw mapping), read by no model."""

    def __init__(self, doc: str = "", nullable: bool = True):
        self.nullable = nullable
        self.doc = doc

    def check(self, value, path, unit, ctx):
        if self._null(value, path, ctx):
            return
        if not isinstance(value, Mapping):
            ctx.add(path, f"expected a mapping, got {_describe(value)}", "type")

    def build(self, value, unit, src, si_key):
        return value


class Leaf(Node):
    """A scalar-like value; also accepts the sourced form ``{value: x, src: .., note: ..}``."""

    expected = "a value"
    allow_table = False

    def __init__(self, nullable=False, doc="", sourced=True, extras=None):
        self.nullable = nullable
        self.doc = doc
        self.sourced = sourced
        self.extras = dict(extras or {})

    def check(self, value, path, unit, ctx):
        if self._null(value, path, ctx):
            return
        if isinstance(value, Mapping):
            if "value" in value or (self.allow_table and "table" in value):
                if not self.sourced:
                    ctx.add(path, f"expected {self.expected}, not a sourced mapping", "type")
                    return
                self._check_sourced(value, path, unit, ctx)
                return
            if "table" in value:
                ctx.add(path, f"this value cannot be a table (expected {self.expected})", "type")
                return
            if "range" in value:
                ctx.add(path, "a sourced value needs value: (range: is the source's range around it; "
                              "for a window write value: [lo, hi])", "type")
                return
            ctx.add(path, f"expected {self.expected} or {{value: ..., src: ...}}", "type")
            return
        self.check_plain(value, path, unit, ctx)

    def _allowed_keys(self) -> set:
        allowed = set(SOURCED_META) | set(self.extras)
        if self.allow_table:
            allowed |= {"table", "x"}
        return allowed

    def _check_sourced(self, value, path, unit, ctx):
        allowed = self._allowed_keys()
        for key in value:
            if key not in allowed:
                ctx.add(path + (key,), f"unknown key {key!r} in a sourced value "
                                       f"(allowed: {', '.join(sorted(allowed))})", "unknown")
        if "value" in value and "table" in value:
            ctx.add(path, "give either value: or table:, not both", "rule")
        check_citation(value, path, ctx)
        for key, node in self.extras.items():
            if key in value:
                node.check(value[key], path + (key,), keyunits.parse_key(key)[1], ctx)
        if "table" in value:
            self._check_table(value, path, unit, ctx)
            return
        inner = value.get("value")
        if inner is None:
            if not self.nullable:
                ctx.add(path + ("value",), "must not be null", "type")
            return
        self.check_plain(inner, path + ("value",), unit, ctx)
        if "range" in value:
            self._check_range_key(value["range"], inner, path, unit, ctx)

    def _check_range_key(self, rng, inner, path, unit, ctx):
        ctx.add(path + ("range",), "range: is only allowed on numbers", "type")

    def _check_table(self, value, path, unit, ctx):
        ctx.add(path, "this value cannot be a table", "type")

    def check_plain(self, value, path, unit, ctx):  # pragma: no cover - interface
        raise TypeError("abstract leaf")

    # build
    def convert_plain(self, value, unit):
        return value

    def build(self, value, unit, src, si_key):
        if value is None:
            return None
        if isinstance(value, Mapping):
            if "table" in value:
                return self._build_table(value, unit, src, si_key)
            if "value" in value:
                extra = _citations(value)
                if "range" in value:
                    extra["range"] = keyunits.convert(value["range"], unit)
                for key, node in self.extras.items():
                    if key in value:
                        own = keyunits.parse_key(key)[1]
                        extra[keyunits.si_key(key, own)] = node.build(value[key], own, src, key)
                own_src = value.get("src", src)
                return Sourced(self.convert_plain(value.get("value"), unit), value.get("value"), own_src,
                               value.get("note"), extra, si_key)
        return self.convert_plain(value, unit)

    def _build_table(self, value, unit, src, si_key):
        raise TypeError(f"{si_key}: not a table")


_NO_SHAPE = object()


class Qty(Leaf):
    """A physical quantity: number, list of numbers or table; unit from the key (or ``unit=``).

    ``shape``: ``None`` scalar, ``n`` list of n numbers, ``"list"`` list of >= 1 numbers,
    ``"range"`` [lo, hi] with lo <= hi (``open_range=True``: either bound may be null),
    ``("rows", n)`` list of rows of n numbers. A tuple of shapes allows several (``(None, "list")``:
    one number or one per flute). Bounds (SI): ``lo``/``hi`` inclusive, ``gt``/``lt`` exclusive.
    ``table``: allowed x units of the table form (``("degC",)`` for temperature-dependent
    properties, where a missing ``x:`` means degC; any other x unit must be written).
    """

    expected = "a number"
    takes_unit = True

    def __init__(self, lo=None, hi=None, *, gt=None, lt=None, shape=None, table=None, nullable=False,
                 doc="", extras=None, ordered=False, unit_vector=False, nonzero=False, strict=False,
                 unit=None, si_key=None, dimensionless=False, open_range=False):
        super().__init__(nullable=nullable, doc=doc, extras=extras)
        self.lo, self.hi, self.gt, self.lt = lo, hi, gt, lt
        self.shapes = shape if isinstance(shape, tuple) and not (len(shape) == 2 and shape[0] == "rows") \
            else (shape,)
        if table is True:
            table = ("degC",)
        self.table_x = tuple(table) if table else ()
        self.allow_table = bool(self.table_x)
        self.ordered = ordered
        self.strict = strict
        self.unit_vector = unit_vector
        self.nonzero = nonzero
        self.unit_token = unit
        self.si_key_override = si_key
        self.dimensionless = dimensionless
        self.open_range = open_range
        if dimensionless:
            self.takes_unit = False
        self.expected = _shape_text(self.shapes)

    def check_plain(self, value, path, unit, ctx):
        shape = self._match_shape(value)
        if shape is _NO_SHAPE:
            ctx.add(path, f"expected {self.expected}, got {_describe(value)}", "type")
            return
        for number, where in _numbers(value, path):
            if not math.isfinite(float(number)):
                ctx.add(where, f"{number!r} is not a finite number", "range")
                return
        si = keyunits.convert(value, unit)
        self._bounds(si, path, unit, ctx)
        if shape == "range" or self.ordered:
            if isinstance(si, list) and all(is_number(v) for v in si):
                for a, b in zip(si, si[1:]):
                    if (b < a) or (self.strict and b == a):
                        ctx.add(path, f"must be {'increasing' if self.strict else 'non-decreasing'} "
                                      f"(got {value})", "rule")
                        break
        if self.unit_vector or self.nonzero:
            if isinstance(si, list) and len(si) == 3 and all(is_number(v) for v in si):
                norm = math.sqrt(sum(float(v) * float(v) for v in si))
                if self.unit_vector and abs(norm - 1.0) > 1e-3:
                    ctx.add(path, f"must be a unit vector (length {norm:.6g})", "rule")
                if norm == 0.0:
                    ctx.add(path, "must not be the zero vector", "rule")

    def _match_shape(self, value):
        for shape in self.shapes:
            if shape is None and is_number(value):
                return None
            if isinstance(shape, int) and _is_flat(value) and len(value) == shape:
                return shape
            if shape == "list" and _is_flat(value) and len(value) >= 1:
                return shape
            if shape == "range" and isinstance(value, list) and len(value) == 2:
                if _is_flat(value) or (self.open_range and all(v is None or is_number(v) for v in value)):
                    return shape
            if isinstance(shape, tuple) and shape[0] == "rows" and isinstance(value, list) and value:
                n = shape[1]
                if all(_is_flat(row) and (n is None or len(row) == n) for row in value):
                    return ("rows",)
        return _NO_SHAPE

    def _bounds(self, si, path, unit, ctx):
        for number in _flat_numbers(si):
            x = float(number)
            if self.lo is not None and x < self.lo:
                ctx.add(path, f"{_show(number, unit)} is below the sane minimum {_show_bound(self.lo, unit)}",
                        "range")
                return
            if self.hi is not None and x > self.hi:
                ctx.add(path, f"{_show(number, unit)} is above the sane maximum {_show_bound(self.hi, unit)}",
                        "range")
                return
            if self.gt is not None and not x > self.gt:
                ctx.add(path, f"{_show(number, unit)} must be greater than {_show_bound(self.gt, unit)}", "range")
                return
            if self.lt is not None and not x < self.lt:
                ctx.add(path, f"{_show(number, unit)} must be less than {_show_bound(self.lt, unit)}", "range")
                return

    def _check_range_key(self, rng, inner, path, unit, ctx):
        where = path + ("range",)
        if not (_is_flat(rng) and len(rng) == 2):
            ctx.add(where, "range must be [lo, hi]", "type")
            return
        lo, hi = rng
        if lo > hi:
            ctx.add(where, f"range [{lo}, {hi}] has lo > hi", "rule")
            return
        if is_number(inner):
            tolerance = 1e-9 * max(1.0, abs(lo), abs(hi))
            if inner < lo - tolerance or inner > hi + tolerance:
                ctx.add(path, f"value {inner} lies outside its range [{lo}, {hi}]", "rule")

    def _table_x_unit(self, value, path, ctx):
        x_token = value.get("x")
        if x_token is None:
            if "degC" in self.table_x:
                return keyunits.unit("degC")
            ctx.add(path, f"a table here needs its x unit (x: {' | '.join(self.table_x)})", "required")
            return None
        if x_token not in self.table_x:
            ctx.add(path + ("x",), f"x unit {x_token!r} is not allowed here (allowed: {', '.join(self.table_x)})",
                    "type")
            return None
        return keyunits.unit(x_token) if x_token in keyunits.UNITS else None

    def _check_table(self, value, path, unit, ctx):
        x_unit = self._table_x_unit(value, path, ctx)
        rows = value.get("table")
        where = path + ("table",)
        if not isinstance(rows, list) or not rows:
            ctx.add(where, "table must be a list of [x, value] rows", "type")
            return
        previous = None
        for i, row in enumerate(rows):
            at = where + (i,)
            if not (_is_flat(row) and len(row) == 2):
                ctx.add(at, f"table row must be [x, value], got {_describe(row)}", "type")
                return
            x, number = row
            if x_unit is not None and x_unit.is_temperature and keyunits.convert(x, x_unit) < 0.0:
                ctx.add(at, f"temperature {x} {x_unit.token} is below absolute zero", "range")
            if previous is not None and x <= previous:
                ctx.add(at, "table rows must be sorted by increasing x", "rule")
            previous = x
            self._bounds(keyunits.convert(number, unit), at, unit, ctx)
        if "range" in value:
            rng = value["range"]
            if not (_is_flat(rng) and len(rng) == 2 and rng[0] <= rng[1]):
                ctx.add(path + ("range",), "range must be [lo, hi] with lo <= hi", "type")

    def convert_plain(self, value, unit):
        return keyunits.convert(value, unit)

    def _build_table(self, value, unit, src, si_key):
        x_token = value.get("x") or "degC"
        x_unit = keyunits.UNITS.get(x_token)
        rows = value.get("table") or []
        xs = [keyunits.convert(r[0], x_unit) for r in rows]
        values = [keyunits.convert(r[1], unit) for r in rows]
        extra = _citations(value)
        if "range" in value:
            extra["range"] = keyunits.convert(value["range"], unit)
        return Table(xs, values, value.get("src", src), value.get("note"), si_key, extra,
                     x_unit.si_token if x_unit else "")


def _shape_text(shapes) -> str:
    words = []
    for shape in shapes:
        if shape is None:
            words.append("a number")
        elif isinstance(shape, int):
            words.append(f"a list of {shape} numbers")
        elif shape == "list":
            words.append("a list of numbers")
        elif shape == "range":
            words.append("[lo, hi]")
        elif isinstance(shape, tuple):
            words.append(f"a list of rows of {shape[1]} numbers" if shape[1] else "a list of rows of numbers")
    return " or ".join(words)


def _is_flat(value) -> bool:
    return isinstance(value, list) and all(is_number(v) for v in value)


def _numbers(value, path):
    if is_number(value):
        yield value, path
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _numbers(v, path + (i,))


def _flat_numbers(value):
    if is_number(value):
        yield value
    elif isinstance(value, list):
        for v in value:
            yield from _flat_numbers(v)


def _show(number_si, unit_) -> str:
    return _show_bound(float(number_si), unit_)


class Count(Leaf):
    """An integer (counts, numbers of teeth, stations, T numbers)."""

    expected = "an integer"

    def __init__(self, lo=0, hi=None, *, nullable=False, doc=""):
        super().__init__(nullable=nullable, doc=doc)
        self.lo, self.hi = lo, hi

    def check_plain(self, value, path, unit, ctx):
        if not (isinstance(value, int) and not isinstance(value, bool)):
            ctx.add(path, f"expected an integer, got {_describe(value)}", "type")
            return
        if self.lo is not None and value < self.lo:
            ctx.add(path, f"{value} is below the minimum {self.lo}", "range")
        if self.hi is not None and value > self.hi:
            ctx.add(path, f"{value} is above the maximum {self.hi}", "range")


class Text(Leaf):
    """Text, optionally one of ``enum`` or matching ``pattern`` (full match)."""

    expected = "text"

    def __init__(self, enum=None, *, pattern=None, nullable=False, doc="", allow_empty=False, sourced=True):
        super().__init__(nullable=nullable, doc=doc, sourced=sourced)
        self.enum = tuple(enum) if enum is not None else None
        self.pattern = re.compile(pattern) if pattern else None
        self.allow_empty = allow_empty
        if self.enum is not None:
            self.expected = "one of " + ", ".join(self.enum)

    def check_plain(self, value, path, unit, ctx):
        if not isinstance(value, str):
            hint = " (quote it)" if is_number(value) or isinstance(value, bool) else ""
            ctx.add(path, f"expected {self.expected}, got {_describe(value)}{hint}", "type")
            return
        if not value and not self.allow_empty:
            ctx.add(path, "must not be empty", "type")
            return
        if self.enum is not None and value not in self.enum:
            close = difflib.get_close_matches(value, self.enum, n=1)
            hint = f" (did you mean {close[0]!r}?)" if close else ""
            ctx.add(path, f"{value!r} is not one of: {', '.join(self.enum)}{hint}", "type")
        elif self.pattern is not None and not self.pattern.fullmatch(value):
            ctx.add(path, f"{value!r} does not match the expected form {self.pattern.pattern}", "type")


class Flag(Leaf):
    expected = "true or false"

    def check_plain(self, value, path, unit, ctx):
        if not isinstance(value, bool):
            ctx.add(path, f"expected true or false, got {_describe(value)}", "type")


class Ref(Text):
    """The id of another database object (its existence is checked by the database)."""

    def __init__(self, kind: str, *, nullable=False, doc="", special=()):
        super().__init__(pattern=ID_PATTERN, nullable=nullable, doc=doc, sourced=False)
        self.kind = kind
        self.special = tuple(special)

    def check_plain(self, value, path, unit, ctx):
        if isinstance(value, str) and value in self.special:
            return
        before = len(ctx.problems)
        super().check_plain(value, path, unit, ctx)
        if len(ctx.problems) == before:
            ctx.refs.append((self.kind, value, path))


class FilePath(Text):
    """A path to a file or directory, relative to the project directory (checked by the loader)."""

    def __init__(self, *, kind="file", exts=None, nullable=False, doc=""):
        super().__init__(nullable=nullable, doc=doc, sourced=False)
        self.kind = kind
        self.exts = tuple(e.lower() for e in exts) if exts else None

    def check_plain(self, value, path, unit, ctx):
        before = len(ctx.problems)
        super().check_plain(value, path, unit, ctx)
        if len(ctx.problems) == before:
            ctx.paths.append((path, value, self))


class Pairs(Node):
    """A two-column curve ``[[x, y], ...]``: y in the field key's unit, x in the declared ``x_unit``
    (``power_kW`` under a gear range is ``[[rpm, kW], ...]``); also ``{value: rows, src: ..}``."""

    takes_unit = True

    def __init__(self, x_unit: str | None, *, x=(None, None), y=(None, None), increasing=True, min_len=1,
                 nullable=False, doc=""):
        self.x_token = x_unit
        self.x_unit = keyunits.unit(x_unit) if x_unit else None
        self.x = x
        self.y = y
        self.increasing = increasing
        self.min_len = min_len
        self.nullable = nullable
        self.doc = doc

    def check(self, value, path, unit, ctx):
        if self._null(value, path, ctx):
            return
        if isinstance(value, Mapping):
            if "value" not in value:
                ctx.add(path, "expected a list of [x, y] rows or {value: rows, src: ..}", "type")
                return
            for key in value:
                if key not in SOURCED_META:
                    ctx.add(path + (key,), f"unknown key {key!r} in a sourced curve", "unknown")
            check_citation(value, path, ctx)
            value, path = value.get("value"), path + ("value",)
            if value is None:
                if not self.nullable:
                    ctx.add(path, "must not be null", "type")
                return
        if not isinstance(value, list) or len(value) < self.min_len:
            ctx.add(path, f"expected a list of at least {self.min_len} [x, y] rows", "type")
            return
        previous = None
        for i, row in enumerate(value):
            where = path + (i,)
            if not (isinstance(row, list) and len(row) == 2 and all(is_number(v) for v in row)):
                ctx.add(where, f"expected [x, y], got {_describe(row)}", "type")
                return
            xs = keyunits.convert(row[0], self.x_unit)
            ys = keyunits.convert(row[1], unit)
            for v, (lo, hi), name, raw in ((xs, self.x, "x", row[0]), (ys, self.y, "y", row[1])):
                if lo is not None and v < lo:
                    ctx.add(where, f"{name} = {raw} is below the sane minimum", "range")
                if hi is not None and v > hi:
                    ctx.add(where, f"{name} = {raw} is above the sane maximum", "range")
            if self.increasing and previous is not None and xs < previous:
                ctx.add(where, "rows must be sorted by increasing x", "rule")
            previous = xs

    def build(self, value, unit, src, si_key):
        if value is None:
            return None
        if isinstance(value, Mapping):
            inner = self.build(value.get("value"), unit, src, si_key)
            extra = _citations(value)
            return Sourced(inner, value.get("value"), value.get("src", src), value.get("note"), extra, si_key)
        rows = [[keyunits.convert(r[0], self.x_unit), keyunits.convert(r[1], unit)] for r in value]
        return Curve(rows, self.x_unit.si_token if self.x_unit else "", unit.si_token if unit else "", si_key)


class Struct(Node):
    """A mapping with declared keys. ``required`` keys must be present (and not null unless the
    field node is nullable); ``checks`` are functions ``(mapping, path, unit, ctx)`` for rules
    across fields; ``hints`` maps a wrong key to an explanation. ``meta`` allows the citation
    keys (src, page, digitised, uncertainty_pct) on the section."""

    takes_unit = True
    map_class = SIMap

    def __init__(self, fields: dict, *, required=(), nullable=False, checks=(), meta=True, doc="",
                 hints=None):
        self.fields = dict(fields)
        self.units: dict[str, Unit | None] = {}
        self.si: dict[str, str] = {}
        self.stems: dict[str, list[str]] = {}
        seen_si: dict[str, str] = {}
        for key, node in self.fields.items():
            unit_, stem = _declared_unit(key, node)
            self.units[key] = unit_
            si = node.si_key_override or (keyunits.si_key(key, unit_) if node.takes_unit else key)
            if si in seen_si:
                raise ValueError(f"schema defines {key} and {seen_si[si]} (same SI key {si})")
            seen_si[si] = key
            self.si[key] = si
            self.stems.setdefault(stem, []).append(key)
            if isinstance(node, Pairs):
                register_curve(key, node.x_token or "", unit_.token if unit_ else "")
        for key in required:
            if key not in self.fields:
                raise ValueError(f"required key {key} is not a field")
        self.required = tuple(required)
        self.nullable = nullable
        self.checks = tuple(checks)
        self.meta = meta
        self.doc = doc
        self.hints = dict(hints or {})

    def extend(self, fields=None, *, required=(), checks=(), hints=None, doc=None, drop=()) -> "Struct":
        """A new Struct with more fields, required keys and checks (``drop`` removes fields)."""
        merged = {k: v for k, v in self.fields.items() if k not in drop}
        merged.update(fields or {})
        clone = type(self)(merged, required=tuple(k for k in self.required if k not in drop) + tuple(required),
                           nullable=self.nullable, checks=tuple(self.checks) + tuple(checks), meta=self.meta,
                           doc=self.doc if doc is None else doc, hints={**self.hints, **(hints or {})})
        return clone

    def child_unit(self, key, unit):
        own = self.units.get(key)
        if own is not None:
            return own
        node = self.fields[key]
        return unit if node.takes_unit and not getattr(node, "dimensionless", False) else None

    def check(self, value, path, unit, ctx):
        if self._null(value, path, ctx):
            return
        if not isinstance(value, Mapping):
            ctx.add(path, f"expected a mapping, got {_describe(value)}", "type")
            return
        for key, item in value.items():
            if not isinstance(key, str):
                ctx.add(path + (key,), f"key {key!r} must be text", "type")
                continue
            if key in NOTE_KEYS:
                continue
            if self.meta and key in SECTION_META:
                continue
            node = self.fields.get(key)
            if node is None:
                ctx.add(path + (key,), self._unknown_message(key), "unknown")
                continue
            node.check(item, path + (key,), self.child_unit(key, unit), ctx)
        if self.meta:
            check_citation(value, path, ctx)
        else:
            _check_notes(value, path, ctx)
        for key in self.required:
            if key not in value:          # a null value is reported by the field node ("must not be null")
                ctx.add(path, f"missing required key {key!r}", "required")
        for rule in self.checks:
            rule(value, path, unit, ctx)

    def _unknown_message(self, key: str) -> str:
        if key in self.hints:
            return f"unknown key {key!r}: {self.hints[key]}"
        stem, unit_ = keyunits.parse_key(key)
        same_stem = [k for k in self.stems.get(stem, ()) if k != key]
        if same_stem:
            declared = same_stem[0]
            declared_unit = self.units.get(declared)
            if declared_unit is None:
                return (f"key {key!r}: {declared!r} is declared without a unit (dimensionless or text); "
                        f"write {declared!r}")
            if unit_ is None:
                return f"key {key!r} needs its unit: write {declared!r} (value in {declared_unit.token})"
            return (f"key {key!r}: the declared unit of {stem!r} is {declared_unit.token}, not {unit_.token}; "
                    f"write {declared!r} with the value in {declared_unit.token}")
        if unit_ is None and key in self.stems:
            declared = self.stems[key][0]
            return f"key {key!r} needs its unit: write {declared!r}"
        close = difflib.get_close_matches(key, list(self.fields), n=1, cutoff=0.75)
        if close:
            return f"unknown key {key!r} (did you mean {close[0]!r}?)"
        return f"unknown key {key!r} (free text belongs under note or notes)"

    def build(self, value, unit, src, si_key):
        if value is None:
            return None
        if not isinstance(value, Mapping):
            return value
        own_src = value.get("src") if isinstance(value.get("src"), str) else src
        entries = {}
        for key, item in value.items():
            if key in CITATION_KEYS:
                entries[key] = item
                continue
            node = self.fields.get(key)
            if node is None:
                name, built = infer(str(key), item, own_src, unit)
                entries[name] = built
                continue
            entries[self.si[key]] = node.build(item, self.child_unit(key, unit), own_src, self.si[key])
        return self.map_class.from_entries(value, entries, own_src)


def _check_notes(value, path, ctx):
    for key in NOTE_KEYS:
        if key in value and not (value[key] is None or isinstance(value[key], (str, list))):
            ctx.add(path + (key,), f"{key} must be text", "type")


def _declared_unit(key: str, node: Node):
    """``(unit, stem)`` of a field: the node's explicit unit, else the key's last token."""
    if node.unit_token is not None:
        unit_ = keyunits.unit(node.unit_token)
        token = unit_.token
        if key.endswith("_" + token):
            return unit_, key[: -len(token) - 1]
        if key == token:
            return unit_, ""
        if node.si_key_override is None:
            raise ValueError(f"schema key {key} does not end with its declared unit {token}")
        return unit_, key
    if not node.takes_unit:
        return None, keyunits.parse_key(key)[0]
    stem, unit_ = keyunits.parse_key(key)
    return unit_, stem


class Pose(Struct):
    """``{translate_mm: [x, y, z], rotate_deg: [a, b, c], order: xyz}`` -> :class:`PoseMap`."""

    map_class = PoseMap

    def __init__(self, fields=None, *, required=(), nullable=False, checks=(), meta=True, doc="", hints=None,
                 translate_bound_m=100.0):
        base = {
            "translate_mm": Qty(lo=-translate_bound_m, hi=translate_bound_m, shape=3),
            "rotate_deg": Qty(lo=-4 * math.pi, hi=4 * math.pi, shape=3),
            "order": Text(ROTATION_ORDERS, sourced=False),
        }
        base.update(fields or {})
        super().__init__(base, required=required, nullable=nullable, checks=checks, meta=meta, doc=doc,
                         hints={"pose_mm_deg": "write {translate_mm: [x, y, z], rotate_deg: [a, b, c], order: xyz}",
                                **(hints or {})})
        self.translate_bound_m = translate_bound_m

    def extend(self, fields=None, *, required=(), checks=(), hints=None, doc=None, drop=()) -> "Pose":
        merged = {k: v for k, v in self.fields.items() if k not in drop and k not in ("translate_mm", "rotate_deg",
                                                                                         "order")}
        merged.update(fields or {})
        return Pose(merged, required=tuple(self.required) + tuple(required), nullable=self.nullable,
                    checks=tuple(self.checks) + tuple(checks), meta=self.meta,
                    doc=self.doc if doc is None else doc, hints=hints, translate_bound_m=self.translate_bound_m)


class MapOf(Node):
    """A mapping with free keys (axis names, offset names ...) and one kind of value; the keys are
    names, not unit keys, so a unit of the field passes down to the values."""

    takes_unit = True

    def __init__(self, value: Node, *, key: Node | None = None, min_len=0, nullable=False, meta=True, doc="",
                 int_keys=False):
        self.value = value
        self.key = key
        self.min_len = min_len
        self.nullable = nullable
        self.meta = meta
        self.doc = doc
        self.int_keys = int_keys

    def check(self, value, path, unit, ctx):
        if self._null(value, path, ctx):
            return
        if not isinstance(value, Mapping):
            ctx.add(path, f"expected a mapping, got {_describe(value)}", "type")
            return
        count = 0
        for key, item in value.items():
            if isinstance(key, str) and key in NOTE_KEYS:
                continue
            if self.meta and isinstance(key, str) and key in SECTION_META:
                continue
            count += 1
            if isinstance(key, int) and not isinstance(key, bool):
                if not self.int_keys:
                    ctx.add(path + (key,), f"key {key!r} must be text (quote it)", "type")
                    continue
            elif not isinstance(key, str):
                ctx.add(path + (key,), f"key {key!r} must be text", "type")
                continue
            if self.key is not None and isinstance(key, str):
                self.key.check(key, path + (key,), None, ctx)
            self.value.check(item, path + (key,), unit if self.value.takes_unit else None, ctx)
        if self.meta:
            check_citation(value, path, ctx)
        if count < self.min_len:
            ctx.add(path, f"needs at least {self.min_len} entr{'y' if self.min_len == 1 else 'ies'}", "required")

    def build(self, value, unit, src, si_key):
        if value is None:
            return None
        own_src = value.get("src") if isinstance(value.get("src"), str) else src
        entries = {}
        for key, item in value.items():
            if isinstance(key, str) and key in CITATION_KEYS:
                entries[key] = item
                continue
            entries[key] = self.value.build(item, unit if self.value.takes_unit else None, own_src, f"{si_key}.{key}")
        return SIMap.from_entries(value, entries, own_src)


class Seq(Node):
    """A list of items of one kind; ``unique`` names a field whose values must differ."""

    takes_unit = True

    def __init__(self, item: Node, *, min_len=0, max_len=None, nullable=False, unique=None, doc=""):
        self.item = item
        self.min_len = min_len
        self.max_len = max_len
        self.nullable = nullable
        self.unique = unique
        self.doc = doc

    def check(self, value, path, unit, ctx):
        if self._null(value, path, ctx):
            return
        if not isinstance(value, list):
            ctx.add(path, f"expected a list, got {_describe(value)}", "type")
            return
        if len(value) < self.min_len:
            ctx.add(path, f"needs at least {self.min_len} item{'s' if self.min_len != 1 else ''}", "required")
        if self.max_len is not None and len(value) > self.max_len:
            ctx.add(path, f"has {len(value)} items, at most {self.max_len} allowed", "range")
        seen = {}
        for i, item in enumerate(value):
            self.item.check(item, path + (i,), unit if self.item.takes_unit else None, ctx)
            if self.unique and isinstance(item, Mapping):
                key = unwrap(item.get(self.unique))
                if isinstance(key, (str, int)) and not isinstance(key, bool):
                    if key in seen:
                        ctx.add(path + (i, self.unique), f"{self.unique} {key!r} is used twice "
                                                         f"(also item {seen[key]})", "duplicate")
                    else:
                        seen[key] = i

    def build(self, value, unit, src, si_key):
        if value is None:
            return None
        return [self.item.build(v, unit if self.item.takes_unit else None, src, si_key) for v in value]


class Tagged(Node):
    """A mapping whose ``tag`` field selects the schema (axis type, tool type ...)."""

    def __init__(self, tag: str, cases: dict, *, nullable=False, doc=""):
        self.tag = tag
        self.cases = dict(cases)
        self.nullable = nullable
        self.doc = doc

    def select(self, value):
        tag = unwrap(value.get(self.tag)) if isinstance(value, Mapping) else None
        return self.cases.get(tag) if isinstance(tag, str) else None

    def check(self, value, path, unit, ctx):
        if self._null(value, path, ctx):
            return
        if not isinstance(value, Mapping):
            ctx.add(path, f"expected a mapping, got {_describe(value)}", "type")
            return
        tag = unwrap(value.get(self.tag))
        if tag is None:
            ctx.add(path, f"missing required key {self.tag!r} (one of: {', '.join(self.cases)})", "required")
            return
        node = self.select(value)
        if node is None:
            close = difflib.get_close_matches(str(tag), list(self.cases), n=1)
            hint = f" (did you mean {close[0]!r}?)" if close else ""
            ctx.add(path + (self.tag,), f"{tag!r} is not one of: {', '.join(self.cases)}{hint}", "type")
            return
        node.check(value, path, unit, ctx)

    def build(self, value, unit, src, si_key):
        node = self.select(value)
        return node.build(value, unit, src, si_key) if node is not None else value


class OneKey(Node):
    """A mapping with exactly one key out of ``cases`` (geometry primitives)."""

    def __init__(self, cases: dict, *, doc=""):
        self.cases = dict(cases)
        self.doc = doc

    def _keys(self, value):
        return [k for k in value if k not in NOTE_KEYS and k not in SECTION_META]

    def check(self, value, path, unit, ctx):
        if self._null(value, path, ctx):
            return
        if not isinstance(value, Mapping):
            ctx.add(path, f"expected a mapping, got {_describe(value)}", "type")
            return
        keys = self._keys(value)
        if len(keys) != 1:
            ctx.add(path, f"needs exactly one of: {', '.join(self.cases)} (got {', '.join(map(str, keys)) or 'none'})",
                    "rule")
            return
        key = keys[0]
        node = self.cases.get(key)
        if node is None:
            ctx.add(path + (key,), f"{key!r} is not one of: {', '.join(self.cases)}", "unknown")
            return
        node.check(value[key], path + (key,), None, ctx)

    def build(self, value, unit, src, si_key):
        own_src = value.get("src") if isinstance(value.get("src"), str) else src
        entries = {}
        for key, item in value.items():
            if key in CITATION_KEYS:
                entries[key] = item
            elif key in self.cases:
                entries[key] = self.cases[key].build(item, None, own_src, key)
        return SIMap.from_entries(value, entries, own_src)


class ByType(Node):
    """Different schemas for text, numbers, lists, mappings and booleans."""

    takes_unit = True

    def __init__(self, *, text=None, number=None, seq=None, mapping=None, boolean=None, nullable=False,
                 doc="", expected=""):
        self.cases = {"text": text, "number": number, "seq": seq, "mapping": mapping, "boolean": boolean}
        self.nullable = nullable
        self.doc = doc
        self.expected = expected or " or ".join(k for k, v in self.cases.items() if v is not None)

    def select(self, value):
        if isinstance(value, bool):
            return self.cases["boolean"]
        if isinstance(value, str):
            return self.cases["text"]
        if is_number(value):
            return self.cases["number"]
        if isinstance(value, list):
            return self.cases["seq"]
        if isinstance(value, Mapping):
            if "value" in value or "table" in value:
                inner = value.get("value")
                leaves = [(kind, c) for kind, c in self.cases.items() if isinstance(c, (Leaf, Pairs))]
                for kind, candidate in leaves:
                    if _kind_of(inner) == kind or (inner is None and "table" in value and isinstance(candidate, Qty)):
                        return candidate
                if inner is None and "table" not in value and leaves:
                    return leaves[0][1]
            return self.cases["mapping"]
        return None

    def check(self, value, path, unit, ctx):
        if self._null(value, path, ctx):
            return
        node = self.select(value)
        if node is None:
            ctx.add(path, f"expected {self.expected}, got {_describe(value)}", "type")
            return
        node.check(value, path, unit if node.takes_unit else None, ctx)

    def build(self, value, unit, src, si_key):
        if value is None:
            return None
        node = self.select(value)
        if node is None:
            return value
        return node.build(value, unit if node.takes_unit else None, src, si_key)


def _kind_of(value) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "text"
    if is_number(value):
        return "number"
    if isinstance(value, list):
        return "seq"
    if isinstance(value, Mapping):
        return "mapping"
    return "none"


class OxideRows(Node):
    """Temper / oxide colours ``[[T_degC, "#rrggbb", name], ...]`` or ``{table: rows, src, note}``;
    built as rows ``[T_K, colour, name]``."""

    def __init__(self, nullable=True):
        self.nullable = nullable

    def check(self, value, path, unit, ctx):
        if self._null(value, path, ctx):
            return
        if isinstance(value, Mapping):
            for key in value:
                if key not in ("table",) + SOURCED_META:
                    ctx.add(path + (key,), f"unknown key {key!r}", "unknown")
            check_citation(value, path, ctx)
            value, path = value.get("table"), path + ("table",)
        if not isinstance(value, list) or not value:
            ctx.add(path, "expected a list of [T_degC, \"#rrggbb\", name] rows", "type")
            return
        colour = re.compile(r"#[0-9a-fA-F]{6}")
        previous = None
        for i, row in enumerate(value):
            where = path + (i,)
            if not (isinstance(row, list) and len(row) == 3 and is_number(row[0]) and isinstance(row[1], str)
                    and isinstance(row[2], str)):
                ctx.add(where, "expected [T_degC, \"#rrggbb\", name]", "type")
                continue
            if not colour.fullmatch(row[1]):
                ctx.add(where, f"colour {row[1]!r} is not #rrggbb", "type")
            if not -ZERO_DEGC_K < row[0] < 3000:
                ctx.add(where, f"temperature {row[0]} degC out of range", "range")
            if previous is not None and row[0] <= previous:
                ctx.add(where, "rows must be sorted by increasing temperature", "rule")
            previous = row[0]

    def build(self, value, unit, src, si_key):
        if value is None:
            return None
        rows = value.get("table") if isinstance(value, Mapping) else value
        built = [[float(r[0]) + ZERO_DEGC_K, r[1], r[2]] for r in rows]
        if isinstance(value, Mapping):
            return Sourced(built, rows, value.get("src", src), value.get("note"), _citations(value), si_key)
        return built


def run(node: Node, value, locations=None, *, path=(), unit=None, obj: str = "", file: str = "") -> Ctx:
    """Check ``value`` against ``node``; the returned Ctx holds problems and references."""
    ctx = Ctx(locations, obj=obj, file=file)
    node.check(value, tuple(path), unit, ctx)
    return ctx


def build(node: Node, value, *, src=None, unit=None, si_key=""):
    """The SI view of a (checked) value."""
    return node.build(value, unit, src, si_key)


__all__ = ["Ctx", "Node", "Any_", "Raw", "Leaf", "Qty", "Count", "Text", "Flag", "Ref", "FilePath", "Pairs", "Struct",
           "Pose", "MapOf", "Seq", "Tagged", "OneKey", "ByType", "OxideRows", "run", "build", "is_number",
           "is_number_like", "unwrap", "pick", "has", "get", "si_number", "check_citation", "ID_PATTERN",
           "SECTION_META", "SOURCED_META"]
