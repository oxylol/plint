"""Unit tokens that database and project keys end with (docs/architecture.md 3, docs/formats.md).

A key is ``<stem>_<unit>``: the unit is always the last token (``travel_mm``, ``kc11_N_mm2``,
``max_rpm``, ``growth_at_max_speed_um``, ``min_abs_pressure_kPa``). The schemas in
:mod:`cnctwin.db.schemas` declare the unit of every field; this module knows how each unit
converts to SI and what the SI key is (``travel_mm`` -> ``travel_m``).

The registry is :data:`cnctwin.units.SUFFIXES` (the shared table) plus the units the database
files need that the shared table does not name (:data:`EXTRA_UNITS`). Scales and labels that
look like units but are not converted (hardness scales, currencies, counts) keep their key.

Matching is on an underscore boundary and the longest token wins: ``max_feed_mm_min`` ends
in ``mm_min`` (mm/min), ``time_constant_min`` in ``min`` (minutes), ``index_90deg_s`` in ``s``
(``90deg`` is not a token), ``kv_1_s`` in ``1_s``. Only ``rpm`` may stand alone as a key
(the ``rpm`` span of a gear range).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .. import units as _core


@dataclass(frozen=True)
class Unit:
    """One unit a key may end with."""

    token: str                 # as written after the stem: "mm", "N_mm2", "rpm", "degC", "1_s"
    si_token: str              # token of the SI key: "m", "Pa", "rad_s", "K", "1_s"
    factor: float = 1.0
    offset: float = 0.0
    scale: bool = False        # a scale or label (HV, eur): no conversion, the key is kept

    @property
    def is_temperature(self) -> bool:
        return self.si_token == "K"

    def to_si(self, value: float) -> float:
        return float(value) * self.factor + self.offset

    def from_si(self, value_si: float) -> float:
        return (float(value_si) - self.offset) / self.factor

    def si_key(self, key: str) -> str:
        """SI name of a key that ends with this unit (``travel_mm`` -> ``travel_m``)."""
        if self.scale:
            return key
        if key == self.token:
            return self.si_token
        return key[: len(key) - len(self.token)] + self.si_token


def _from_core() -> dict[str, Unit]:
    out: dict[str, Unit] = {}
    for suffix in _core.SUFFIXES:
        token = suffix.suffix[1:] if suffix.suffix.startswith("_") else suffix.suffix
        si_token = suffix.si_suffix[1:] if suffix.si_suffix.startswith("_") else suffix.si_suffix
        out[token] = Unit(token, si_token, suffix.factor, suffix.offset)
    return out


DEG = math.pi / 180.0

# Units the database files need beyond cnctwin.units.SUFFIXES.
EXTRA_UNITS: tuple[Unit, ...] = (
    Unit("rad_s", "rad_s"),
    Unit("deg_s", "rad_s", DEG),                   # rotary axis speed
    Unit("1_s", "1_s"),                            # rates and loop gains (kv, strain rate)
    Unit("mm2_s", "m2_s", 1e-6),                   # kinematic viscosity (cSt)
    Unit("m2_s", "m2_s"),
    Unit("kJ_kg", "J_kg", 1e3),                    # specific (latent) heat
    Unit("J_kg", "J_kg"),
    Unit("kg_mm", "kg_m", 1e-3),                   # mass moment m r (chuck jaws)
    Unit("kg_m", "kg_m"),
    Unit("mJ", "J", 1e-3),
    Unit("V", "V"),
    Unit("VA", "VA"),
    Unit("kVA", "VA", 1e3),
    Unit("N_per_Nm", "N_per_Nm"),                  # clamping force per input torque
    Unit("per_MPa", "per_Pa", 1e-6),               # Usui-type constants (stress unit in the key)
    Unit("per_Pa", "per_Pa"),
    Unit("per_m", "per_m"),
    Unit("l_h", "m3_s", 1e-3 / 3600.0),
    Unit("ml_min", "m3_s", 1e-6 / 60.0),
    Unit("J_m2", "J_m2"),                          # fracture energy G_IC
    Unit("kJ_m2", "J_m2", 1e3),
    Unit("um_per_min", "m_per_s", 1e-6 / 60.0),
    Unit("s_per_rev", "s_per_rev"),
    # scales and labels: not converted, key kept
    Unit("HB", "HB", scale=True), Unit("HV", "HV", scale=True), Unit("HRC", "HRC", scale=True),
    Unit("HRB", "HRB", scale=True), Unit("HRA", "HRA", scale=True), Unit("HK", "HK", scale=True),
    Unit("eur", "eur", scale=True), Unit("usd", "usd", scale=True), Unit("eur_per_l", "eur_per_l", scale=True),
    Unit("count", "count", scale=True),
)

UNITS: dict[str, Unit] = _from_core()
for _unit in EXTRA_UNITS:
    UNITS[_unit.token] = _unit
# ``rpm`` appears twice in the core table (``_rpm`` and bare ``rpm``); one entry is enough here.
UNITS["rpm"] = Unit("rpm", "rad_s", 2.0 * math.pi / 60.0)

_BY_LENGTH = sorted(UNITS.values(), key=lambda u: len(u.token), reverse=True)

# Absolute temperatures; differences are _dK.
TEMPERATURE_TOKENS = ("degC", "K")


def unit(token: str) -> Unit:
    """The unit named by ``token`` (KeyError if unknown)."""
    return UNITS[token]


def parse_key(key: str) -> tuple[str, Unit | None]:
    """``(stem, unit)`` of a key by its last token; ``(key, None)`` for a dimensionless key.

    ``travel_mm`` -> ``("travel", mm)``; ``max_rpm`` -> ``("max", rpm)``; ``rpm`` -> ``("", rpm)``;
    ``poisson`` -> ``("poisson", None)``.
    """
    if key == "rpm":
        return "", UNITS["rpm"]
    for candidate in _BY_LENGTH:
        tail = "_" + candidate.token
        if key.endswith(tail) and len(key) > len(tail):
            return key[: len(key) - len(tail)], candidate
    return key, None


def key_with_unit(stem: str, unit_: Unit | None) -> str:
    if unit_ is None:
        return stem
    return f"{stem}_{unit_.token}" if stem else unit_.token


def si_key(key: str, unit_: Unit | None) -> str:
    if unit_ is None:
        return key
    return unit_.si_key(key)


def convert(value, unit_: Unit | None):
    """A file value (number, nested list of numbers, None) in SI; text and booleans pass."""
    if unit_ is None or unit_.scale or value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [convert(v, unit_) for v in value]
    if isinstance(value, (int, float)):
        return unit_.to_si(value)
    return value


_DIMENSIONLESS_TEXT = ("", "-", "1", "none", "dimensionless")


def parse_unit_text(text) -> Unit | None:
    """The unit named by a free unit text (``"mm"``, ``"1/s"``, ``"W/(m K)"``, ``"N/mm^2"``, ``"%"``);
    ``None`` for a dimensionless value. Raises ValueError for a unit this registry cannot convert."""
    if text is None:
        return None
    raw = str(text).strip()
    if raw.lower() in _DIMENSIONLESS_TEXT:
        return None
    norm = (raw.replace("°C", "degC").replace("°", "deg").replace("µ", "u").replace("μ", "u")
            .replace("^", "").replace("(", "").replace(")", "").replace("·", "").replace("*", "")
            .replace(" ", "").replace("%", "pct").replace("/", "_"))
    if norm in UNITS:
        return UNITS[norm]
    raise ValueError(f"unit {raw!r} is not a unit this database can convert (write SI, or one of the key "
                     f"unit tokens, e.g. mm, deg, rpm, N_mm2, 1/s)")


def to_si(key: str, value):
    """``(si_key, si_value)`` of one key by its own suffix (schema-less use)."""
    _, unit_ = parse_key(key)
    return si_key(key, unit_), convert(value, unit_)


__all__ = ["Unit", "UNITS", "EXTRA_UNITS", "TEMPERATURE_TOKENS", "unit", "parse_key", "key_with_unit",
           "si_key", "convert", "to_si", "parse_unit_text"]
