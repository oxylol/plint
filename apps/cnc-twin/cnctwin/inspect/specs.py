"""Datum and tolerance specifications: the project's ``cad.datums`` / ``cad.tolerances`` (db/_templates/project.yaml)
merged with the CAD's PMI (cnctwin.geom.cad ``pmi``, the same form in mm), read into SI records
(docs/models/inspect.md "Specifications").

Entries may come as SI views (``nominal_m``, ``near_m``: cnctwin.db.simap) or in the file form (``nominal_mm``,
``plus_deg``, ``Ra_um``, ``diameter_mm: {nominal, fit, plus, minus}``). Hand-written entries win over PMI entries of
the same id (tolerances) or label (datums); the order is: hand entries as written, then the PMI entries not
overridden. Selectors stay as given (CAD frame C) and are resolved by cnctwin.geom.selectors.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

DEG = math.pi / 180.0
THREAD_CLASS = re.compile(r"^(?:(\d)([a-hA-H])){1,2}$|^([123])([AB])$")
ISO_FIT = re.compile(r"^([A-Za-z]{1,2})(\d{1,2})$")


def _get(obj, key):
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(key)
    try:
        return getattr(obj, key)
    except AttributeError:
        return None


def _plain(value):
    value = getattr(value, "value", value)
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if hasattr(value, "raw") and isinstance(getattr(value, "raw"), Mapping):
        return _plain(value.raw)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _number(obj, si_key: str, file_keys: tuple) -> float | None:
    """SI value of ``si_key`` or of one of ``(file_key, factor)``."""
    v = _get(obj, si_key)
    v = getattr(v, "value", v)
    if v is not None and not isinstance(v, (list, tuple, Mapping)):
        return float(v)
    for key, factor in file_keys:
        v = _get(obj, key)
        v = getattr(v, "value", v)
        if v is not None and not isinstance(v, (list, tuple, Mapping)):
            return float(v) * factor
    return None


def _vector(obj, si_key: str, file_key: str, factor: float) -> list[float] | None:
    v = _get(obj, si_key)
    scale = 1.0
    if v is None:
        v = _get(obj, file_key)
        scale = factor
    v = getattr(v, "value", v)
    if v is None:
        return None
    return [float(getattr(x, "value", x)) * scale for x in v]


def _length(obj, stem: str) -> float | None:
    return _number(obj, stem + "_m", ((stem + "_mm", 1e-3), (stem + "_um", 1e-6)))


def _angle(obj, stem: str) -> float | None:
    return _number(obj, stem + "_rad", ((stem + "_deg", DEG),))


def selector_form(selector) -> dict:
    """A selector in plain file form (mm) for JSON output."""
    out = {}
    for key in ("face", "normal", "axis", "id"):
        v = _plain(_get(selector, key))
        if v is not None:
            out[key] = v
    near = _vector(selector, "near_m", "near_mm", 1e-3)
    if near is not None:
        out["near_mm"] = [round(x * 1e3, 9) for x in near]
    radius = _length(selector, "radius")
    if radius is not None:
        out["radius_mm"] = round(radius * 1e3, 9)
    return out


@dataclass
class DatumSpec:
    label: str
    feature: object
    source: str = "hand"


@dataclass
class SizeSpec:
    """A size of a feature of size: nominal and limits (from plus/minus or a fit designation)."""

    nominal_m: float
    fit: str | None = None
    plus_m: float | None = None
    minus_m: float | None = None

    @property
    def thread_class(self) -> bool:
        return bool(self.fit) and THREAD_CLASS.match(str(self.fit).replace(" ", "")) is not None


@dataclass
class ToleranceSpec:
    id: str
    kind: str
    feature: object = None
    between: list = field(default_factory=list)
    nominal_m: float | None = None
    plus_m: float | None = None
    minus_m: float | None = None
    tolerance_m: float | None = None
    nominal_rad: float | None = None
    plus_rad: float | None = None
    minus_rad: float | None = None
    datums: list = field(default_factory=list)
    basic_m: list | None = None
    size: SizeSpec | None = None
    Ra_m: float | None = None
    Rz_m: float | None = None
    Rt_m: float | None = None
    modifier: str = ""
    source: str = "hand"

    @property
    def material_condition(self) -> str:
        """``MMC``, ``LMC`` or ``RFS`` from the modifier text (M / L / MMC / LMC)."""
        text = self.modifier.upper().replace(" ", "")
        if text in ("M", "MMC", "(M)"):
            return "MMC"
        if text in ("L", "LMC", "(L)"):
            return "LMC"
        return "RFS"

    def describe_feature(self) -> str:
        if self.feature is not None:
            return _describe(self.feature)
        if self.between:
            return " to ".join(_describe(b) for b in self.between)
        return ""

    def to_json(self) -> dict:
        out = {"id": self.id, "kind": self.kind, "source": self.source}
        if self.feature is not None:
            out["feature_selector"] = selector_form(self.feature)
        if self.between:
            out["between"] = [selector_form(b) for b in self.between]
        return out


def _describe(selector) -> str:
    form = selector_form(selector)
    text = form.get("face", "any")
    if "id" in form:
        text += f" {form['id']}"
    if "near_mm" in form:
        text += " near (" + ", ".join(f"{v:g}" for v in form["near_mm"]) + ") mm"
    if "radius_mm" in form:
        text += f" r {form['radius_mm']:g} mm"
    return text


def _size(entry) -> SizeSpec | None:
    raw = _get(entry, "diameter_m")
    scale = 1.0
    if raw is None:
        raw = _get(entry, "diameter_mm")
        scale = 1e-3
    if raw is None:
        return None
    inner = getattr(raw, "value", raw)
    if isinstance(inner, (int, float)) and not isinstance(inner, bool):
        return SizeSpec(float(inner) * scale)
    nominal = _get(raw, "nominal")
    nominal = getattr(nominal, "value", nominal)
    if nominal is None:
        return None
    plus = getattr(_get(raw, "plus"), "value", _get(raw, "plus"))
    minus = getattr(_get(raw, "minus"), "value", _get(raw, "minus"))
    fit = _get(raw, "fit")
    fit = getattr(fit, "value", fit)
    return SizeSpec(float(nominal) * scale, str(fit) if fit else None,
                    float(plus) * scale if plus is not None else None,
                    float(minus) * scale if minus is not None else None)


def read_tolerance(entry, source: str = "hand") -> ToleranceSpec:
    ident = str(_plain(_get(entry, "id")) or "")
    kind = str(_plain(_get(entry, "kind")) or "")
    between = _get(entry, "between") or []
    datums = [str(_plain(d)) for d in (_get(entry, "datums") or [])]
    basic = _vector(entry, "basic_m", "basic_mm", 1e-3)
    modifier = str(_plain(_get(entry, "modifier")) or "")
    return ToleranceSpec(
        id=ident, kind=kind, feature=_get(entry, "feature"), between=list(between),
        nominal_m=_length(entry, "nominal"), plus_m=_length(entry, "plus"), minus_m=_length(entry, "minus"),
        tolerance_m=_length(entry, "tolerance"), nominal_rad=_angle(entry, "nominal"), plus_rad=_angle(entry, "plus"),
        minus_rad=_angle(entry, "minus"), datums=datums, basic_m=basic, size=_size(entry),
        Ra_m=_length(entry, "Ra"), Rz_m=_length(entry, "Rz"), Rt_m=_length(entry, "Rt"), modifier=modifier,
        source=source)


def read_datum(entry, source: str = "hand") -> DatumSpec:
    return DatumSpec(str(_plain(_get(entry, "label")) or ""), _get(entry, "feature"), source)


def merged(cad_section, pmi: Mapping | None) -> tuple[list[DatumSpec], list[ToleranceSpec]]:
    """(datums, tolerances) of the project's cad section merged with the model's PMI (hand entries win)."""
    hand_d = [read_datum(d, "hand") for d in (_get(cad_section, "datums") or [])]
    hand_t = [read_tolerance(t, "hand") for t in (_get(cad_section, "tolerances") or [])]
    pmi = pmi or {}
    pmi_d = [read_datum(d, "pmi") for d in (pmi.get("datums") or [])]
    pmi_t = [read_tolerance(t, "pmi") for t in (pmi.get("tolerances") or [])]
    labels = {d.label for d in hand_d}
    ids = {t.id for t in hand_t}
    datums = hand_d + [d for d in pmi_d if d.label not in labels]
    tolerances = hand_t + [t for t in pmi_t if t.id not in ids]
    seen: set[str] = set()
    for i, t in enumerate(tolerances):
        if not t.id or t.id in seen:
            t.id = f"{t.kind or 'tolerance'}-{i + 1}"
        seen.add(t.id)
    return datums, tolerances


__all__ = ["DatumSpec", "ToleranceSpec", "SizeSpec", "read_tolerance", "read_datum", "merged", "selector_form",
           "THREAD_CLASS", "ISO_FIT"]
