"""SI views of database and project files (docs/architecture.md 18, docs/formats.md "Databases").

A file is YAML with unit-suffixed keys. :class:`SIMap` wraps one parsed mapping and exposes every
key converted to SI by its **declared** unit (``travel_mm`` -> ``travel_m``), with attribute and
item access:

* ``{value: x, src: k, note: ..}`` -> :class:`Sourced` (``float(x)`` is the SI value; it also acts
  as a number in arithmetic and comparisons, and as a list when the value is a list)
* ``{table: [[x, v], ...], x: degC}`` -> :class:`Table` (call it with the SI x: kelvin, metres ...)
* a two-column curve declared by schema path (``spindles[].gear_ranges[].power_kW`` is
  ``[[rpm, kW], ...]``) -> :class:`Curve` (a list of ``[x_SI, y_SI]`` rows, also callable)
* a pose ``{translate_mm, rotate_deg, order}`` -> :class:`PoseMap` (``.matrix()`` is the 4x4)
* a mapping under a unit key whose own keys carry no unit (``fz_mm: {min, start, max}``,
  ``rapid_mm_min: {X: .., Y: ..}``) passes the unit down to its entries
* lists of mappings are lists of SIMap; absent optional keys read as ``None`` through attribute
  access (``m.foo``) and raise ``KeyError`` through item access (``m["foo"]``)

Objects from :class:`cnctwin.db.Database` and :mod:`cnctwin.project` are built by their schema
(:mod:`cnctwin.db.schema`), which declares every unit and curve. ``SIMap(raw)`` without a schema
infers units from the key suffixes (:mod:`cnctwin.db.keyunits`) and two-column curves from the
curve keys the schemas declare (by key name); it is for ad-hoc data and tests.

``.raw`` is the mapping as read (after format migration and ``extends`` merging).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from ..units import REFERENCE_TEMP_K
from . import keyunits
from .keyunits import Unit

NOTE_KEYS = ("note", "notes")
CITATION_KEYS = ("src", "note", "notes", "page", "pages", "digitised", "uncertainty_pct")
ROTATION_ORDERS = ("xyz", "xzy", "yxz", "yzx", "zxy", "zyx")


def _num(other):
    return other.value if isinstance(other, Sourced) else other


class Sourced:
    """A value with its source. ``float(x)`` is the SI value.

    It acts as its value in arithmetic, comparisons, ``int()``/``range()`` (integral values),
    ``len()``/iteration/indexing (list values), ``numpy.asarray`` and formatting, so code may use
    it where a number is expected. ``bool(x)`` is ``bool(value)`` (``False`` for a null value).
    ``x.range`` is the source's range in SI (or None); other citation keys (``page``,
    ``digitised``, ``uncertainty_pct``) and declared extras (``at_handle_torque_Nm`` ...) are
    attributes too (``None`` when absent).
    """

    __slots__ = ("value", "raw_value", "src", "note", "extra", "si_key")

    def __init__(self, value, raw_value, src, note, extra, si_key):
        self.value = value
        self.raw_value = raw_value
        self.src = src
        self.note = note
        self.extra = extra
        self.si_key = si_key

    @property
    def estimated(self) -> bool:
        return self.src == "est"

    def __float__(self) -> float:
        if self.value is None:
            raise TypeError(f"{self.si_key} has no value")
        return float(self.value)

    def __int__(self) -> int:
        return int(float(self))

    def __index__(self) -> int:
        number = float(self)
        if not number.is_integer():
            raise TypeError(f"{self.si_key} = {self.value!r} is not an integer")
        return int(number)

    def __bool__(self) -> bool:
        return self.value is not None and bool(self.value)

    def __repr__(self) -> str:
        return f"Sourced({self.si_key}={self.value!r}, src={self.src!r})"

    def __str__(self) -> str:
        return str(self.value)

    def __format__(self, spec: str) -> str:
        return format(self.value, spec) if spec else str(self.value)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        extra = object.__getattribute__(self, "extra")
        return extra.get(name)

    # number protocol
    def __add__(self, other): return self.value + _num(other)
    def __radd__(self, other): return _num(other) + self.value
    def __sub__(self, other): return self.value - _num(other)
    def __rsub__(self, other): return _num(other) - self.value
    def __mul__(self, other): return self.value * _num(other)
    def __rmul__(self, other): return _num(other) * self.value
    def __truediv__(self, other): return self.value / _num(other)
    def __rtruediv__(self, other): return _num(other) / self.value
    def __floordiv__(self, other): return self.value // _num(other)
    def __rfloordiv__(self, other): return _num(other) // self.value
    def __mod__(self, other): return self.value % _num(other)
    def __rmod__(self, other): return _num(other) % self.value
    def __pow__(self, other): return self.value ** _num(other)
    def __rpow__(self, other): return _num(other) ** self.value
    def __neg__(self): return -self.value
    def __pos__(self): return +self.value
    def __abs__(self): return abs(self.value)
    def __round__(self, ndigits=None): return round(self.value, ndigits)
    def __floor__(self): return math.floor(self.value)
    def __ceil__(self): return math.ceil(self.value)
    def __trunc__(self): return math.trunc(self.value)
    def __lt__(self, other): return self.value < _num(other)
    def __le__(self, other): return self.value <= _num(other)
    def __gt__(self, other): return self.value > _num(other)
    def __ge__(self, other): return self.value >= _num(other)

    def __eq__(self, other):
        return self.value == _num(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(_hashable(self.value))

    # sequence protocol (list values)
    def __len__(self) -> int:
        return len(self.value)

    def __iter__(self):
        return iter(self.value)

    def __getitem__(self, index):
        return self.value[index]

    def __call__(self, *args):
        """A sourced curve or table is callable like the curve or table itself."""
        return self.value(*args)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.value, dtype=dtype)


def _hashable(value):
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    return value


class Table:
    """A tabulated property ``{table: [[x, v], ...], x: <unit>}``: call with the SI x, get SI.

    Linear interpolation; outside the table the end value is held and the clamp is counted
    (``clamped``) so the report can say a property was extrapolated. ``x_unit`` is the SI token
    of x ("K" for temperature tables, "m" for ``x: mm`` ...). ``float(table)`` is the value at
    20 degC and exists only for temperature tables.
    """

    def __init__(self, x_si, values, src, note, si_key, extra=None, x_unit: str = "K"):
        order = np.argsort(np.asarray(x_si, dtype=float), kind="stable")
        self.x = np.asarray(x_si, dtype=float)[order]
        self.values = np.asarray(values, dtype=float)[order]
        self.src = src
        self.note = note
        self.si_key = si_key
        self.extra = extra or {}
        self.x_unit = x_unit
        self.clamped = 0

    @property
    def points_K(self) -> np.ndarray:
        """The x points of a temperature table (kelvin)."""
        if self.x_unit != "K":
            raise TypeError(f"{self.si_key} is a table over {self.x_unit}, not temperature")
        return self.x

    @property
    def estimated(self) -> bool:
        return self.src == "est"

    def __call__(self, x):
        xs = np.asarray(x, dtype=float)
        outside = (xs < self.x[0]) | (xs > self.x[-1])
        self.clamped += int(np.count_nonzero(outside))
        result = np.interp(xs, self.x, self.values)
        return float(result) if np.ndim(result) == 0 else result

    def __float__(self) -> float:
        if self.x_unit != "K":
            raise TypeError(f"{self.si_key} is a table over {self.x_unit}: call it with x")
        return float(np.interp(REFERENCE_TEMP_K, self.x, self.values))

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return self.__dict__.get("extra", {}).get(name)

    def __repr__(self) -> str:
        return f"Table({self.si_key}, {len(self.x)} points over {self.x_unit}, src={self.src!r})"


class Curve(list):
    """A two-column curve: a list of ``[x_SI, y_SI]`` rows (as written, converted per column).

    ``curve.x`` and ``curve.y`` are arrays; ``curve(x)`` interpolates linearly, holding the end
    values outside the curve.
    """

    def __init__(self, rows, x_unit: str, y_unit: str, si_key: str = ""):
        super().__init__([list(r) for r in rows])
        self.x_unit = x_unit
        self.y_unit = y_unit
        self.si_key = si_key

    @property
    def x(self) -> np.ndarray:
        return np.array([r[0] for r in self], dtype=float)

    @property
    def y(self) -> np.ndarray:
        return np.array([r[1] for r in self], dtype=float)

    def __call__(self, x):
        result = np.interp(np.asarray(x, dtype=float), self.x, self.y)
        return float(result) if np.ndim(result) == 0 else result


class SIMap(Mapping):
    """Read-only SI view of one mapping.

    ``SIMap(raw)`` infers units from the key suffixes; schema nodes build SIMaps with
    :meth:`from_entries` (units as declared).
    """

    def __init__(self, raw: Mapping, inherited_src: str | None = None, unit: Unit | None = None):
        self.raw = raw
        self._src = _own_src(raw, inherited_src)
        self._data: dict = {}
        for key, value in raw.items():
            if key in CITATION_KEYS:
                self._data[key] = value
                continue
            name, built = infer(str(key), value, self._src, unit)
            self._data[name] = built

    @classmethod
    def from_entries(cls, raw: Mapping, entries: dict, src: str | None = None) -> "SIMap":
        obj = cls.__new__(cls)
        obj.raw = raw
        obj._src = src
        obj._data = dict(entries)
        return obj

    @property
    def src(self):
        """The section-level source in force here (own ``src`` or inherited)."""
        return self._src

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key) -> bool:
        return key in self._data

    def __getattr__(self, name):
        if name.startswith("_") or name == "raw":
            raise AttributeError(name)
        return self._data.get(name)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self._data)})"

    def to_dict(self) -> dict:
        """Plain SI values (JSON-serialisable): sourced values as their value, tables as points."""
        return {str(k): plain(v) for k, v in self._data.items()}


class PoseMap(SIMap):
    """A pose ``{translate_m, rotate_rad, order}``: ``matrix()`` maps child to parent coordinates.

    ``p_parent = R p_child + t`` with ``R = R_a(order[0]) R_a(order[1]) R_a(order[2])``; the angles
    ``rotate_rad = [rx, ry, rz]`` are indexed by axis, ``order`` (default ``xyz``) says in which
    sequence the rotations about the moving axes are applied (intrinsic x, then y', then z'' for
    ``xyz``: the same convention as three.js ``Euler(rx, ry, rz, 'XYZ')``).
    """

    def matrix(self) -> np.ndarray:
        translate = _plain_vector(self._data.get("translate_m"), 3)
        rotate = _plain_vector(self._data.get("rotate_rad"), 3)
        order = self._data.get("order") or "xyz"
        return pose_matrix(translate, rotate, str(order))


def rotation(axis: str, angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    if axis == "z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    raise ValueError(f"rotation axis {axis!r} is not x, y or z")


def pose_matrix(translate_m, rotate_rad, order: str = "xyz") -> np.ndarray:
    """4x4 homogeneous transform of a pose (see :class:`PoseMap`)."""
    if order not in ROTATION_ORDERS:
        raise ValueError(f"rotation order {order!r} is not one of {', '.join(ROTATION_ORDERS)}")
    angles = {"x": float(rotate_rad[0]), "y": float(rotate_rad[1]), "z": float(rotate_rad[2])}
    matrix = np.eye(4)
    rot = np.eye(3)
    for axis in order:
        rot = rot @ rotation(axis, angles[axis])
    matrix[:3, :3] = rot
    matrix[:3, 3] = [float(v) for v in translate_m]
    return matrix


def _plain_vector(value, n):
    if value is None:
        return [0.0] * n
    value = value.value if isinstance(value, Sourced) else value
    return [float(_num(v)) for v in value]


def _own_src(raw, inherited):
    if isinstance(raw, Mapping) and isinstance(raw.get("src"), str):
        return raw["src"]
    return inherited


# --------------------------------------------------------------------------- schema-less inference

# key name -> (x unit token, y unit token) of every two-column curve the schemas declare
# (filled by cnctwin.db.schema when a Struct declares a Pairs field; None if a key name is
# declared with different columns at different paths, which inference then does not guess).
CURVE_COLUMNS: dict[str, tuple[str, str] | None] = {}


def register_curve(key: str, x_token: str, y_token: str) -> None:
    columns = (x_token, y_token)
    if key in CURVE_COLUMNS and CURVE_COLUMNS[key] != columns:
        CURVE_COLUMNS[key] = None
        return
    CURVE_COLUMNS[key] = columns


def _curve_columns(key: str):
    if not CURVE_COLUMNS:
        from . import schemas  # noqa: F401  (declares the curves)
    return CURVE_COLUMNS.get(key)


def infer(key: str, value, src, parent_unit: Unit | None):
    """``(si_key, si_value)`` of one entry by its key suffix (no schema)."""
    stem, own = keyunits.parse_key(key)
    unit_ = own if own is not None else (parent_unit if _unit_capable(value) else None)
    name = keyunits.si_key(key, own)
    if isinstance(value, Mapping):
        if "table" in value:
            return name, _infer_table(name, value, src, unit_)
        if "value" in value:
            extra = {k: (keyunits.convert(v, unit_) if k == "range" else v)
                     for k, v in value.items() if k not in ("value", "src", "note")}
            inner = value.get("value")
            columns = _curve_columns(key) if own is not None else None
            if columns and _is_rows(inner, 2):
                built = _curve(name, inner, columns)
            else:
                built = keyunits.convert(inner, unit_)
            return name, Sourced(built, inner, value.get("src", src), value.get("note"), extra, name)
        if "translate_mm" in value or "rotate_deg" in value:
            return name, PoseMap(value, src, None)
        return name, SIMap(value, src, unit_)
    if isinstance(value, list) and value and all(isinstance(v, Mapping) for v in value):
        return name, [SIMap(v, src, None) for v in value]
    if own is not None and _is_rows(value, 2):
        columns = _curve_columns(key)
        if columns:
            return name, _curve(name, value, columns)
    return name, keyunits.convert(value, unit_)


def _curve(name, rows, columns):
    x_unit, y_unit = (keyunits.unit(t) if t else None for t in columns)
    converted = [[keyunits.convert(r[0], x_unit), keyunits.convert(r[1], y_unit)] for r in rows]
    return Curve(converted, x_unit.si_token if x_unit else "", y_unit.si_token if y_unit else "", name)


def _infer_table(name, raw, src, unit_):
    x_token = raw.get("x", "degC")
    x_unit = keyunits.UNITS.get(str(x_token)) if x_token is not None else None
    rows = raw.get("table") or []
    xs = [keyunits.convert(r[0], x_unit) for r in rows]
    values = [keyunits.convert(r[1], unit_) for r in rows]
    extra = {k: (keyunits.convert(v, unit_) if k == "range" else v)
             for k, v in raw.items() if k not in ("table", "src", "note", "x")}
    return Table(xs, values, raw.get("src", src), raw.get("note"), name, extra,
                 x_unit.si_token if x_unit else "")


def _unit_capable(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_unit_capable(v) for v in value)
    return isinstance(value, Mapping)


def _is_rows(value, n) -> bool:
    return (isinstance(value, list) and bool(value)
            and all(isinstance(r, (list, tuple)) and len(r) == n for r in value))


def plain(value):
    """JSON-safe plain SI value of anything SIMap holds."""
    if isinstance(value, SIMap):
        return value.to_dict()
    if isinstance(value, Curve):
        return [list(r) for r in value]
    if isinstance(value, list):
        return [plain(v) for v in value]
    if isinstance(value, Sourced):
        return plain(value.value)
    if isinstance(value, Table):
        return {"x": value.x.tolist(), "x_unit": value.x_unit, "values": value.values.tolist()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def wrap(raw: Mapping) -> SIMap:
    """SI view of a parsed YAML mapping (units inferred from the key suffixes)."""
    return SIMap(raw)


__all__ = ["SIMap", "Sourced", "Table", "Curve", "PoseMap", "wrap", "infer", "plain", "pose_matrix",
           "rotation", "register_curve", "CURVE_COLUMNS", "ROTATION_ORDERS", "NOTE_KEYS", "CITATION_KEYS"]
