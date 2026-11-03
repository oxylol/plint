"""Tool data from ISO 13399 exchange files (ISO 10303-21 "p21" text; docs/formats.md "Project file").

:func:`parse_p21` reads the exchange structure (``ISO-10303-21; HEADER; ... DATA; #n=ENTITY(...);
... ENDSEC; END-ISO-10303-21;``): strings (``''`` escapes, ``\\X2\\...\\X0\\`` and ``\\X\\hh``
encodings), numbers, enumerations (``.MM.``), references (``#12``), ``$`` (unset), ``*``, lists
and typed parameters (``LENGTH_MEASURE(10.)``), and complex instances ``#n=(A(..) B(..));``.

:func:`tool_items` finds ISO 13399 property symbols (``DC``, ``APMX`` ...) and their values in the
three encodings vendor files use:

1. a symbol and its measure in one instance: ``#5=PROPERTY_VALUE('DC',LENGTH_MEASURE(10.0),'mm');``
   (any entity whose first string parameter is a known symbol; the value is the first number or
   measure after it; a unit string after the value overrides the file's unit);
2. a symbol with description, number and unit: ``#6=PROPERTY('DC','cutting diameter',10.0,'mm');``
3. an ISO 10303 property chain: ``PROPERTY_DEFINITION('DC',..,#p)`` referenced by a
   ``PROPERTY_DEFINITION_REPRESENTATION(#def, #rep)`` whose ``REPRESENTATION`` holds a
   ``MEASURE_REPRESENTATION_ITEM`` (or any instance with a measure) with the value.

Lengths are in the file's length unit (``SI_UNIT(.MILLI.,.METRE.)`` = mm, a
``CONVERSION_BASED_UNIT('INCH',..)`` = inch; mm when the file names none), angles in its plane
angle unit (degrees when the file names none or a ``CONVERSION_BASED_UNIT('DEGREE',..)``; radians
for ``SI_UNIT($,.RADIAN.)``). The symbols map to tool template fields by :data:`SYMBOLS`; an item
id is the ``PRODUCT`` id (lower-cased to the database id form) or the file name. Symbols that
are not in the table are listed in the item's ``notes``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..core.errors import InputError

P21_SUFFIXES = (".p21", ".stp", ".step", ".spf")


# --------------------------------------------------------------------------- the exchange structure

@dataclass(frozen=True)
class Ref:
    id: int


@dataclass(frozen=True)
class Enum:
    name: str


@dataclass(frozen=True)
class Typed:
    type: str
    value: object


@dataclass
class Instance:
    id: int
    entity: str                  # upper case; "" for a complex instance
    params: list
    parts: list = field(default_factory=list)   # complex instance: [(entity, params), ...]
    line: int = 0


class _Lexer:
    def __init__(self, text: str, name: str):
        self.text = text
        self.pos = 0
        self.name = name

    def line(self) -> int:
        return self.text.count("\n", 0, self.pos) + 1

    def error(self, message):
        raise InputError(f"p21: {message}", self.name, self.line())

    def skip(self):
        text = self.text
        while self.pos < len(text):
            ch = text[self.pos]
            if ch.isspace():
                self.pos += 1
            elif text.startswith("/*", self.pos):
                end = text.find("*/", self.pos + 2)
                if end < 0:
                    self.error("unterminated comment")
                self.pos = end + 2
            else:
                break

    def peek(self) -> str:
        self.skip()
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def expect(self, ch: str):
        if self.peek() != ch:
            self.error(f"expected {ch!r}, found {self.text[self.pos:self.pos + 20]!r}")
        self.pos += 1

    def keyword(self) -> str:
        self.skip()
        match = re.compile(r"!?[A-Za-z_][A-Za-z0-9_\-]*").match(self.text, self.pos)
        if not match:
            self.error(f"expected a keyword at {self.text[self.pos:self.pos + 20]!r}")
        self.pos = match.end()
        return match.group(0).upper()

    def string(self) -> str:
        assert self.text[self.pos] == "'"
        self.pos += 1
        out = []
        while True:
            end = self.text.find("'", self.pos)
            if end < 0:
                self.error("unterminated string")
            out.append(self.text[self.pos:end])
            self.pos = end + 1
            if self.pos < len(self.text) and self.text[self.pos] == "'":
                out.append("'")
                self.pos += 1
                continue
            break
        return decode_string("".join(out))

    def number(self):
        match = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?").match(self.text, self.pos)
        if not match:
            self.error("bad number")
        self.pos = match.end()
        text = match.group(0)
        if "." in text or "e" in text or "E" in text:
            return float(text)
        return int(text)

    def param(self):
        ch = self.peek()
        if ch == "'":
            return self.string()
        if ch == "#":
            self.pos += 1
            match = re.compile(r"\d+").match(self.text, self.pos)
            if not match:
                self.error("bad reference")
            self.pos = match.end()
            return Ref(int(match.group(0)))
        if ch == "$":
            self.pos += 1
            return None
        if ch == "*":
            self.pos += 1
            return "*"
        if ch == ".":
            match = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)\.").match(self.text, self.pos)
            if match:
                self.pos = match.end()
                name = match.group(1).upper()
                if name in ("T", "TRUE"):
                    return True
                if name in ("F", "FALSE"):
                    return False
                return Enum(name)
            return self.number()
        if ch == "(":
            return self.params()
        if ch == '"':
            end = self.text.find('"', self.pos + 1)
            if end < 0:
                self.error("unterminated binary")
            value = self.text[self.pos + 1:end]
            self.pos = end + 1
            return value
        if ch and (ch.isdigit() or ch in "+-"):
            return self.number()
        if ch and (ch.isalpha() or ch == "_"):
            name = self.keyword()
            inner = self.params()
            return Typed(name, inner[0] if len(inner) == 1 else inner)
        self.error(f"unexpected {self.text[self.pos:self.pos + 20]!r}")

    def params(self) -> list:
        self.expect("(")
        out = []
        if self.peek() == ")":
            self.pos += 1
            return out
        while True:
            out.append(self.param())
            ch = self.peek()
            if ch == ",":
                self.pos += 1
                continue
            if ch == ")":
                self.pos += 1
                return out
            self.error(f"expected , or ) in a parameter list, found {self.text[self.pos:self.pos + 20]!r}")


def decode_string(text: str) -> str:
    """ISO 10303-21 string encodings: \\X2\\hhhh...\\X0\\ (UCS-2), \\X\\hh (ISO 8859-1), \\\\ (backslash)."""
    def x2(match):
        hexes = match.group(1)
        return "".join(chr(int(hexes[i:i + 4], 16)) for i in range(0, len(hexes), 4))
    text = re.sub(r"\\X2\\((?:[0-9A-Fa-f]{4})+)\\X0\\", x2, text)
    text = re.sub(r"\\X\\([0-9A-Fa-f]{2})", lambda m: chr(int(m.group(1), 16)), text)
    return text.replace("\\\\", "\\")


@dataclass
class P21File:
    name: str
    header: dict
    instances: dict[int, Instance]

    def by_entity(self, entity: str) -> list[Instance]:
        return [i for i in self.instances.values() if i.entity == entity or any(p[0] == entity for p in i.parts)]


def parse_p21(text: str, name: str = "<p21>") -> P21File:
    lexer = _Lexer(text, name)
    if lexer.keyword() != "ISO-10303-21":
        lexer.error("not an ISO 10303-21 file (first keyword must be ISO-10303-21)")
    lexer.expect(";")
    header: dict = {}
    instances: dict[int, Instance] = {}
    section = None
    while True:
        lexer.skip()
        if lexer.pos >= len(text):
            lexer.error("missing END-ISO-10303-21")
        if lexer.peek() == "#":
            if section != "DATA":
                lexer.error("an entity instance outside the DATA section")
            line = lexer.line()
            lexer.pos += 1
            match = re.compile(r"\d+").match(text, lexer.pos)
            if not match:
                lexer.error("bad instance name")
            lexer.pos = match.end()
            ident = int(match.group(0))
            lexer.expect("=")
            if lexer.peek() == "(":
                lexer.pos += 1
                parts = []
                while lexer.peek() != ")":
                    entity = lexer.keyword()
                    parts.append((entity, lexer.params()))
                lexer.pos += 1
                instance = Instance(ident, "", [], parts, line)
            else:
                entity = lexer.keyword()
                instance = Instance(ident, entity, lexer.params(), [], line)
            lexer.expect(";")
            if ident in instances:
                lexer.error(f"instance #{ident} is defined twice")
            instances[ident] = instance
            continue
        word = lexer.keyword()
        if word == "END-ISO-10303-21":
            lexer.expect(";")
            break
        if word in ("HEADER", "DATA"):
            section = word
            if word == "DATA" and lexer.peek() == "(":
                lexer.params()
            lexer.expect(";")
            continue
        if word == "ENDSEC":
            section = None
            lexer.expect(";")
            continue
        if section == "HEADER":
            header[word] = lexer.params()
            lexer.expect(";")
            continue
        lexer.error(f"unexpected {word}")
    return P21File(name, header, instances)


# --------------------------------------------------------------------------- ISO 13399 properties

@dataclass(frozen=True)
class Symbol:
    path: str             # tool template field ("geometry.diameter_mm")
    kind: str             # length | angle | count | rpm | mass | text
    note: str = ""


# ISO 13399 property symbols -> tool template fields (ISO/TS 13399-2, -3, -4 symbol definitions).
SYMBOLS: dict[str, Symbol] = {
    "DC": Symbol("geometry.diameter_mm", "length", "cutting diameter"),
    "DCX": Symbol("geometry.diameter_mm", "length", "cutting diameter maximum (used when DC is absent)"),
    "DCN": Symbol("geometry.tip_diameter_mm", "length", "cutting diameter minimum (chamfer/countersink tip)"),
    "APMX": Symbol("geometry.cutting_length_mm", "length", "depth of cut maximum"),
    "LCF": Symbol("geometry.flute_length_mm", "length", "length chip flute"),
    "OAL": Symbol("geometry.overall_length_mm", "length", "overall length"),
    "LF": Symbol("geometry.overall_length_mm", "length", "functional length (used when OAL is absent)"),
    "LU": Symbol("geometry.neck.length_mm", "length", "usable length (the neck length when DN is given)"),
    "DN": Symbol("geometry.neck.diameter_mm", "length", "neck diameter"),
    "DMM": Symbol("geometry.shank.diameter_mm", "length", "shank diameter"),
    "LS": Symbol("geometry.shank.length_mm", "length", "shank length"),
    "ZEFP": Symbol("geometry.flutes", "count", "peripheral effective cutting edge count"),
    "NOF": Symbol("geometry.flutes", "count", "number of flutes (used when ZEFP is absent)"),
    "RE": Symbol("geometry.corner_radius_mm", "length", "corner radius (insert: nose radius)"),
    "CHW": Symbol("geometry.corner_chamfer_mm", "length", "corner chamfer width"),
    "KAPR": Symbol("geometry.inserts.lead_angle_deg", "angle", "tool cutting edge angle kappa_r (turning: "
                   "holder.approach_angle_deg)"),
    "PSIR": Symbol("geometry.inserts.lead_angle_deg", "angle", "tool lead angle psi_r = 90 deg - kappa_r "
                   "(used when KAPR is absent)"),
    "GAMF": Symbol("geometry.rake_deg", "angle", "radial rake angle gamma_f"),
    "GAMP": Symbol("geometry.inserts.axial_rake_deg", "angle", "axial rake angle gamma_p"),
    "FHA": Symbol("geometry.helix_deg", "angle", "flute helix angle"),
    "SIG": Symbol("geometry.point_angle_deg", "angle", "point angle"),
    "TP": Symbol("geometry.thread.pitch_mm", "length", "thread pitch"),
    "CW": Symbol("geometry.width_mm", "length", "cutting width (grooving: insert.width_mm)"),
    "IC": Symbol("geometry.insert.ic_mm", "length", "inscribed circle diameter"),
    "S": Symbol("geometry.insert.thickness_mm", "length", "insert thickness"),
    "AN": Symbol("geometry.clearance_deg", "angle", "clearance angle major (insert: insert.clearance_deg)"),
    "EPSR": Symbol("geometry.insert.included_angle_deg", "angle", "corner (included) angle"),
    "WT": Symbol("mass_g", "mass", "weight of the item"),
    "RPMX": Symbol("max_rpm", "rpm", "rotational speed maximum"),
    "BMC": Symbol("tool_material.class", "text", "body material code (ISO 513)"),
    "HAND": Symbol("hand", "text", "hand (R, L, N)"),
}
PRIORITY = {"DC": 0, "DCX": 1, "OAL": 0, "LF": 1, "ZEFP": 0, "NOF": 1, "KAPR": 0, "PSIR": 1}
BODY_MATERIAL_CLASS = {"HW": "carbide", "HC": "carbide", "HF": "carbide", "HT": "cermet", "HS": "hss",
                       "HSS": "hss", "HSS-E": "hss", "HSS-PM": "hss", "CA": "ceramic", "CM": "ceramic",
                       "CN": "ceramic", "CR": "ceramic", "CC": "ceramic", "BL": "cbn", "BH": "cbn", "BC": "cbn",
                       "DP": "pcd", "DM": "pcd", "DC": "pcd"}
HANDS = {"R": "right", "L": "left", "N": "neutral"}
TYPE_WORDS = (("ball", "ball_endmill"), ("torus", "toroidal_endmill"), ("toroid", "toroidal_endmill"),
              ("corner radius", "toroidal_endmill"), ("chamfer", "chamfer_mill"), ("spot", "spot_drill"),
              ("centre drill", "centre_drill"), ("center drill", "centre_drill"), ("thread mill", "thread_mill"),
              ("tap", "tap_cutting"), ("reamer", "reamer"), ("face mill", "face_mill"),
              ("shoulder", "shoulder_mill"), ("drill", "twist_drill"), ("end mill", "square_endmill"),
              ("endmill", "square_endmill"), ("slitting", "slitting_saw"), ("keyseat", "keyseat"),
              ("engrav", "engraver"), ("boring bar", "boring_bar"), ("grooving", "grooving_insert"),
              ("parting", "parting_insert"), ("threading insert", "threading_insert"),
              ("turning insert", "turning_insert"), ("insert", "turning_insert"))
LENGTH_UNITS = {"mm": 1.0, "millimetre": 1.0, "millimeter": 1.0, "m": 1000.0, "metre": 1000.0, "inch": 25.4,
                "in": 25.4}
ANGLE_UNITS = {"deg": 1.0, "degree": 1.0, "rad": 180.0 / math.pi, "radian": 180.0 / math.pi}
MASS_UNITS = {"g": 1.0, "gram": 1.0, "kg": 1000.0, "kilogram": 1000.0, "lb": 453.59237}
RPM_UNITS = {"rpm": 1.0, "1/min": 1.0, "min-1": 1.0, "r/min": 1.0}


def _first_string(params):
    for p in params:
        if isinstance(p, str):
            return p
    return None


def _number_of(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Typed):
        return _number_of(value.value)
    return None


def _measure_after(params, start):
    """(number, measure type, unit string, unit reference) after index ``start``."""
    for i in range(start, len(params)):
        p = params[i]
        number = _number_of(p)
        if number is not None:
            measure = p.type if isinstance(p, Typed) else ""
            unit, ref = None, None
            for q in params[i + 1:]:
                if isinstance(q, str) and q.strip() and unit is None:
                    unit = q.strip()
                elif isinstance(q, Ref) and ref is None:
                    ref = q.id
            return number, measure, unit, ref
    return None


SI_PREFIX_MM = {"MILLI": 1.0, "CENTI": 10.0, "DECI": 100.0, "MICRO": 1e-3, None: 1000.0}


def unit_scales(p21: P21File) -> dict[int, tuple[str, float]]:
    """instance id -> ("length", mm per unit) or ("angle", degrees per unit) of every unit instance
    (SI units by prefix; conversion-based units through their measure-with-unit)."""
    scales: dict[int, tuple[str, float]] = {}

    def scale_of(ident, depth=0):
        if ident in scales or depth > 8 or ident not in p21.instances:
            return scales.get(ident)
        inst = p21.instances[ident]
        parts = inst.parts or [(inst.entity, inst.params)]
        names = [p[0] for p in parts]
        kind = "length" if "LENGTH_UNIT" in names else "angle" if "PLANE_ANGLE_UNIT" in names else None
        if kind is None:
            return None
        for entity, params in parts:
            if entity == "SI_UNIT" and len(params) >= 2:
                prefix, unit = params[-2], params[-1]
                prefix_name = prefix.name if isinstance(prefix, Enum) else None
                if kind == "length" and isinstance(unit, Enum) and unit.name == "METRE":
                    scales[ident] = ("length", SI_PREFIX_MM.get(prefix_name, 1.0))
                if kind == "angle" and isinstance(unit, Enum) and unit.name == "RADIAN":
                    scales[ident] = ("angle", 180.0 / math.pi)
            if entity == "CONVERSION_BASED_UNIT" and len(params) >= 2 and isinstance(params[1], Ref):
                factor = p21.instances.get(params[1].id)
                if factor is not None and len(factor.params) >= 2:
                    number = _number_of(factor.params[0])
                    base = factor.params[1].id if isinstance(factor.params[1], Ref) else None
                    base_scale = scale_of(base, depth + 1) if base is not None else None
                    if number is not None and base_scale is not None:
                        scales[ident] = (kind, number * base_scale[1])
                if ident not in scales and isinstance(params[0], str):
                    label = params[0].strip().lower()
                    if kind == "length" and label in LENGTH_UNITS:
                        scales[ident] = ("length", LENGTH_UNITS[label])
                    if kind == "angle" and label in ANGLE_UNITS:
                        scales[ident] = ("angle", ANGLE_UNITS[label])
        return scales.get(ident)

    for ident in p21.instances:
        scale_of(ident)
    return scales


def _file_units(p21: P21File) -> tuple[float, float]:
    """(mm per length unit, degrees per angle unit) for values that name no unit: the units of a
    GLOBAL_UNIT_ASSIGNED_CONTEXT if the file has one, else the units no conversion is based on
    (mm and degrees when the file defines none)."""
    scales = unit_scales(p21)
    for inst in p21.instances.values():
        for entity, params in inst.parts or [(inst.entity, inst.params)]:
            if entity == "GLOBAL_UNIT_ASSIGNED_CONTEXT" and params and isinstance(params[0], list):
                found = {scales[r.id][0]: scales[r.id][1] for r in params[0] if isinstance(r, Ref) and r.id in scales}
                return found.get("length", 1.0), found.get("angle", 1.0)
    bases = set()
    for inst in p21.instances.values():
        if inst.entity.endswith("MEASURE_WITH_UNIT") and len(inst.params) >= 2 and isinstance(inst.params[1], Ref):
            bases.add(inst.params[1].id)
    length = [s for i, (k, s) in sorted(scales.items()) if k == "length" and i not in bases]
    angle = [s for i, (k, s) in sorted(scales.items()) if k == "angle" and i not in bases]
    return (length[0] if length else 1.0), (angle[0] if angle else 1.0)


def properties(p21: P21File) -> dict[str, tuple]:
    """symbol -> (value, unit text or measure type, unit instance id, line) of every symbol in the file."""
    found: dict[str, tuple] = {}
    referrers: dict[int, list[Instance]] = {}
    for inst in p21.instances.values():
        for p in _flat(inst.params):
            if isinstance(p, Ref):
                referrers.setdefault(p.id, []).append(inst)
    for inst in p21.instances.values():
        params = inst.params if inst.entity else [q for _, ps in inst.parts for q in ps]
        index = next((i for i, p in enumerate(params) if isinstance(p, str)), None)
        if index is None:
            continue
        symbol = params[index].strip()
        if symbol not in SYMBOLS or symbol in found:
            continue
        kind = SYMBOLS[symbol].kind
        if kind == "text":
            strings = [p for p in params[index + 1:] if isinstance(p, str) and p.strip()]
            text = strings[-1] if strings else _text_by_reference(p21, inst, referrers)
            if text is not None:
                found[symbol] = (text, None, None, inst.line)
            continue
        measure = _measure_after(params, index + 1) or _measure_by_reference(p21, inst, referrers)
        if measure is not None:
            found[symbol] = (measure[0], measure[2] or measure[1], measure[3], inst.line)
    return found


def _flat(params):
    for p in params:
        if isinstance(p, list):
            yield from _flat(p)
        elif isinstance(p, Typed) and isinstance(p.value, list):
            yield from _flat(p.value)
        else:
            yield p


def _chain_targets(p21, inst, referrers):
    """Instances reachable from a property definition through the representations that reference it."""
    for ref in referrers.get(inst.id, []):
        for p in _flat(ref.params):
            if isinstance(p, Ref) and p.id != inst.id and p.id in p21.instances:
                rep = p21.instances[p.id]
                yield rep
                for q in _flat(rep.params):
                    if isinstance(q, Ref) and q.id in p21.instances:
                        yield p21.instances[q.id]


def _measure_by_reference(p21, inst, referrers):
    for target in _chain_targets(p21, inst, referrers):
        found = _measure_after(target.params, 0)
        if found is not None:
            return found
    return None


def _text_by_reference(p21, inst, referrers):
    for target in _chain_targets(p21, inst, referrers):
        strings = [p for p in target.params if isinstance(p, str) and p.strip()]
        if len(strings) >= 2:
            return strings[1]
    return None


def _scale(kind, value, unit, unit_ref, scales, length_mm, angle_deg):
    text = (unit or "").strip().lower()
    referenced = scales.get(unit_ref) if unit_ref is not None else None
    if kind == "length":
        if text in LENGTH_UNITS:
            return value * LENGTH_UNITS[text]
        if referenced is not None and referenced[0] == "length":
            return value * referenced[1]
        return value * length_mm
    if kind == "angle":
        if text in ANGLE_UNITS:
            return value * ANGLE_UNITS[text]
        if referenced is not None and referenced[0] == "angle":
            return value * referenced[1]
        return value * angle_deg
    if kind == "mass":
        return value * MASS_UNITS.get(text, MASS_UNITS["kg"])
    if kind == "rpm":
        return value * RPM_UNITS.get(text, 1.0)
    if kind == "count":
        return int(round(value))
    return value


def _set(item: dict, path: str, value) -> None:
    node = item
    parts = path.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _clean_id(text: str) -> str:
    out = re.sub(r"[^a-z0-9._+-]+", "-", text.strip().lower()).strip("-.")
    return out or "tool"


def _infer_type(words: str, props: dict) -> str | None:
    text = words.lower()
    for word, kind in TYPE_WORDS:
        if word in text:
            if kind == "square_endmill" and "RE" in props and "DC" in props:
                radius, diameter = props["RE"], props["DC"]
                if abs(radius - diameter / 2.0) <= 0.01 * diameter:
                    return "ball_endmill"
                if radius > 0:
                    return "toroidal_endmill"
            return kind
    return None


def tool_items(p21: P21File, *, source: str = "") -> list[dict]:
    """Tool items (tool template form, raw values with unit keys) of one exchange file."""
    found = properties(p21)
    length_mm, angle_deg = _file_units(p21)
    scales = unit_scales(p21)
    products = p21.by_entity("PRODUCT")
    product = products[0] if products else None
    item_id = None
    words = ""
    if product is not None:
        strings = [p for p in product.params if isinstance(p, str)]
        if strings:
            item_id = strings[0]
            words = " ".join(strings)
    stem = Path(p21.name).stem
    item: dict = {"id": _clean_id(item_id or stem), "order_code": item_id or stem}
    values: dict[str, float] = {}
    chosen: dict[str, str] = {}
    for symbol, (raw, unit, unit_ref, _line) in sorted(found.items(), key=lambda kv: PRIORITY.get(kv[0], 0)):
        spec = SYMBOLS[symbol]
        if spec.kind == "text":
            values[symbol] = raw
            continue
        value = _scale(spec.kind, raw, unit, unit_ref, scales, length_mm, angle_deg)
        values[symbol] = value
        if spec.path in chosen:
            continue
        chosen[spec.path] = symbol
    if "PSIR" in values and "KAPR" not in values:
        values["KAPR"] = 90.0 - values["PSIR"]
        chosen["geometry.inserts.lead_angle_deg"] = "KAPR"
    insert = "IC" in values
    tool_type = _infer_type(words, values)
    turning = tool_type in ("turning_insert", "grooving_insert", "parting_insert", "threading_insert") or (
        insert and tool_type is None)
    for path, symbol in sorted(chosen.items()):
        value = values[symbol]
        if symbol in ("RE",) and (turning or insert):
            path = "geometry.insert.nose_radius_mm" if turning else "geometry.inserts.nose_radius_mm"
        if symbol in ("ZEFP", "NOF") and insert and not turning:
            path = "geometry.inserts.count"
        if symbol in ("KAPR",) and turning:
            path = "geometry.holder.approach_angle_deg"
        if symbol == "CW" and turning:
            path = "geometry.insert.width_mm"
        if symbol == "AN" and insert:
            path = "geometry.insert.clearance_deg" if turning else "geometry.inserts.clearance_deg"
        if symbol == "IC" and not turning:
            path = "geometry.inserts.ic_mm"
        if symbol in ("LU",) and "DN" not in values:
            continue
        if symbol == "GAMP" and not insert:
            continue
        _set(item, path, value)
    if "BMC" in values:
        material_class = BODY_MATERIAL_CLASS.get(str(values["BMC"]).strip().upper())
        if material_class:
            _set(item, "tool_material.class", material_class)
    if "HAND" in values:
        hand = HANDS.get(str(values["HAND"]).strip().upper()[:1])
        if hand:
            item["hand"] = hand
    if tool_type is not None:
        item["type"] = tool_type
    unmapped = sorted(set(_all_symbols(p21)) - set(found))
    note = f"imported from {source or p21.name} (ISO 13399)"
    if unmapped:
        note += f"; symbols not mapped: {', '.join(unmapped)}"
    item["notes"] = note
    return [item]


def _all_symbols(p21: P21File) -> list[str]:
    """Upper-case short strings that look like ISO 13399 symbols (for the not-mapped list)."""
    out = set()
    for inst in p21.instances.values():
        names = [inst.entity] + [e for e, _ in inst.parts]
        if any(name.endswith("UNIT") or "CONTEXT" in name for name in names):
            continue
        first = _first_string(inst.params if inst.entity else [q for _, ps in inst.parts for q in ps])
        if first and re.fullmatch(r"[A-Z][A-Z0-9]{0,7}", first.strip()) and inst.entity not in ("PRODUCT",
                                                                                               "APPLICATION_CONTEXT"):
            out.add(first.strip())
    return sorted(out)


def read_items(path) -> list[dict]:
    """Tool items of one p21 file or of every p21 file in a directory (sorted by name)."""
    path = Path(path)
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in P21_SUFFIXES) if path.is_dir() else [path]
    items = []
    for file in files:
        try:
            text = file.read_text(encoding="utf-8", errors="strict")
        except (OSError, UnicodeDecodeError) as error:
            raise InputError(f"cannot read ISO 13399 file: {error}", str(file)) from None
        items.extend(tool_items(parse_p21(text, str(file)), source=file.name))
    return items


__all__ = ["parse_p21", "tool_items", "read_items", "properties", "unit_scales", "SYMBOLS", "P21File", "Instance", "Ref", "Enum",
           "Typed", "decode_string", "P21_SUFFIXES"]
