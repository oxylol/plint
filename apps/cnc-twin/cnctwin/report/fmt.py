"""Display formatting: SI values in the units a machinist reads (mm, um, degC, kW, kWh, rpm).

Everything here is deterministic text (no locale, no wall clock). Conversions use the unit
definitions of :mod:`cnctwin.db.keyunits` (one source of truth for factors); a missing or
non-finite value is shown as :data:`DASH`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..db import keyunits
from ..units import ZERO_DEGC_K

DASH = "–"
MINUS = "−"


def as_float(value) -> float | None:
    """The value as a finite float, else None (text, None, NaN, inf, bool)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _per_si(token: str) -> float:
    """Display units per SI unit of a key unit token (1000 for mm)."""
    return 1.0 / keyunits.unit(token).factor


MM_PER_M = _per_si("mm")
UM_PER_M = _per_si("um")
KW_PER_W = _per_si("kW")
KJ_PER_J = _per_si("kJ")
MIN_PER_S = _per_si("min")
H_PER_S = _per_si("h")
RPM_PER_RAD_S = _per_si("rpm")
DEG_PER_RAD = _per_si("deg")
MM_MIN_PER_M_S = _per_si("mm_min")
G_PER_KG = _per_si("g")
L_PER_M3 = _per_si("l")
MPA_PER_PA = _per_si("MPa")
BAR_PER_PA = _per_si("bar")
KWH_PER_J = KW_PER_W * H_PER_S
MM3_PER_M3 = MM_PER_M ** 3
L_MIN_PER_M3_S = _per_si("l_min")
CM3_PER_M3 = _per_si("cm") ** 3


def fixed(value, decimals: int = 3) -> str:
    number = as_float(value)
    if number is None:
        return DASH
    text = f"{number:.{decimals}f}"
    if text.startswith("-"):
        stripped = text[1:].strip("0").strip(".")
        return MINUS + text[1:] if stripped else text[1:]
    return text


def num(value, digits: int = 4) -> str:
    """``digits`` significant figures, plain notation between 1e-4 and 1e9, else scientific."""
    number = as_float(value)
    if number is None:
        return DASH
    if isinstance(value, int) and abs(value) < 10 ** 15:
        return str(value) if value >= 0 else MINUS + str(-value)
    if number == 0.0:
        return "0"
    magnitude = abs(number)
    if magnitude >= 1e9 or magnitude < 1e-4:
        mantissa, exponent = f"{number:.{digits - 1}e}".split("e")
        sign = MINUS if mantissa.startswith("-") else ""
        return f"{sign}{mantissa.lstrip('-')}e{int(exponent)}"
    exponent = int(math.floor(math.log10(magnitude)))
    decimals = max(0, digits - 1 - exponent)
    rounded = abs(round(number, decimals))
    if rounded >= 10.0 ** (exponent + 1):          # rounding carried into the next decade (0.99999 -> 1.000)
        decimals = max(0, decimals - 1)
    return fixed(number, decimals)


def signed(value, decimals: int = 3) -> str:
    """With an explicit sign (+0.012, −0.004)."""
    number = as_float(value)
    if number is None:
        return DASH
    text = fixed(number, decimals)
    return text if text.startswith(MINUS) else "+" + text


def integer(value) -> str:
    number = as_float(value)
    return DASH if number is None else f"{int(round(number)):d}"


def mm(value_m, decimals: int = 3) -> str:
    number = as_float(value_m)
    return DASH if number is None else fixed(number * MM_PER_M, decimals)


def um(value_m, decimals: int = 1) -> str:
    number = as_float(value_m)
    return DASH if number is None else fixed(number * UM_PER_M, decimals)


def signed_um(value_m, decimals: int = 1) -> str:
    number = as_float(value_m)
    return DASH if number is None else signed(number * UM_PER_M, decimals)


def degc(value_K, decimals: int = 1) -> str:
    number = as_float(value_K)
    return DASH if number is None else fixed(number - ZERO_DEGC_K, decimals)


def pct(value_frac, decimals: int = 1) -> str:
    number = as_float(value_frac)
    return DASH if number is None else fixed(number * 100.0, decimals)


def seconds(value_s, decimals: int = 3) -> str:
    return fixed(value_s, decimals)


def duration(value_s) -> str:
    """2.320 s, 3 min 12.4 s, 2 h 05 min."""
    number = as_float(value_s)
    if number is None:
        return DASH
    if abs(number) < 60.0:
        return f"{fixed(number, 3)} s"
    if abs(number) < 3600.0:
        minutes = int(number // 60.0)
        return f"{minutes} min {fixed(number - 60.0 * minutes, 1)} s"
    hours = int(number // 3600.0)
    minutes = int(round((number - 3600.0 * hours) / 60.0))
    return f"{hours} h {minutes:02d} min"


def kwh(value_J, decimals: int = 4) -> str:
    number = as_float(value_J)
    return DASH if number is None else num(number * KWH_PER_J, decimals)


def kj(value_J, decimals: int = 2) -> str:
    number = as_float(value_J)
    return DASH if number is None else fixed(number * KJ_PER_J, decimals)


def eur(value, decimals: int = 2) -> str:
    number = as_float(value)
    if number is None:
        return DASH
    if number != 0.0 and abs(number) < 1.0:          # cents and below: three significant figures
        return num(number, 3)
    return fixed(number, decimals)


def rpm(value_rad_s) -> str:
    number = as_float(value_rad_s)
    return DASH if number is None else fixed(abs(number) * RPM_PER_RAD_S, 0)


def grams(value_kg, digits: int = 4) -> str:
    number = as_float(value_kg)
    return DASH if number is None else num(number * G_PER_KG, digits)


def litres(value_m3, digits: int = 4) -> str:
    number = as_float(value_m3)
    return DASH if number is None else num(number * L_PER_M3, digits)


def yes_no(value) -> str:
    if value is None:
        return DASH
    return "yes" if bool(value) else "no"


# ----------------------------------------------------------------------------- channel display units

@dataclass(frozen=True)
class DisplayUnit:
    label: str
    factor: float = 1.0
    offset: float = 0.0

    def __call__(self, value_si):
        return value_si * self.factor + self.offset


_LENGTH = ((1e-3, DisplayUnit("µm", UM_PER_M)), (1.0, DisplayUnit("mm", MM_PER_M)),
           (math.inf, DisplayUnit("m")))
_POWER = ((1e3, DisplayUnit("W")), (math.inf, DisplayUnit("kW", KW_PER_W)))
_ENERGY = ((1e4, DisplayUnit("J")), (1e7, DisplayUnit("kJ", KJ_PER_J)), (math.inf, DisplayUnit("kWh", KWH_PER_J)))
_VOLUME = ((1e-6, DisplayUnit("mm³", MM3_PER_M3)), (math.inf, DisplayUnit("cm³", CM3_PER_M3)))
_MASS = ((1.0, DisplayUnit("g", G_PER_KG)), (math.inf, DisplayUnit("kg")))
_FIXED = {
    "K": DisplayUnit("°C", 1.0, -ZERO_DEGC_K), "m/s": DisplayUnit("mm/min", MM_MIN_PER_M_S),
    "m_s": DisplayUnit("mm/min", MM_MIN_PER_M_S), "rad/s": DisplayUnit("rpm", RPM_PER_RAD_S),
    "rad_s": DisplayUnit("rpm", RPM_PER_RAD_S), "rad": DisplayUnit("deg", DEG_PER_RAD),
    "Pa": DisplayUnit("MPa", MPA_PER_PA), "-": DisplayUnit("-"), "": DisplayUnit("-"),
}
_SCALED = {"m": _LENGTH, "W": _POWER, "J": _ENERGY, "m3": _VOLUME, "kg": _MASS}


def display_unit(si_unit: str, magnitude: float | None = None) -> DisplayUnit:
    """The display unit for a channel of SI unit ``si_unit`` whose values reach ``magnitude``."""
    unit = (si_unit or "").strip()
    if unit in _FIXED:
        return _FIXED[unit]
    if unit in _SCALED:
        size = abs(magnitude) if magnitude is not None and math.isfinite(magnitude) else 0.0
        for limit, choice in _SCALED[unit]:
            if size < limit:
                return choice
        return _SCALED[unit][-1][1]
    return DisplayUnit(unit)


_SI_TOKEN_TO_UNIT = {"rad_s": "rad/s", "m_s": "m/s"}


def quantity(key: str, value) -> str:
    """One named quantity in display units: ``deviation_m`` 6.2e-5 -> ``deviation 62.0 µm``.

    The key carries its unit as the last token (architecture 3); a non-SI token is converted first.
    """
    stem, unit = keyunits.parse_key(str(key))
    label = (stem or str(key)).replace("_", " ")
    number = as_float(value)
    if number is None:
        return f"{str(key).replace('_', ' ')} {value if isinstance(value, str) else DASH}"
    if unit is None or unit.scale:
        return f"{str(key).replace('_', ' ')} {num(number)}"
    si_value = unit.to_si(number)
    if unit.si_token == "frac":
        return f"{label} {pct(si_value)} %"
    shown = display_unit(_SI_TOKEN_TO_UNIT.get(unit.si_token, unit.si_token), si_value)
    text = num(shown(si_value))
    return f"{label} {text} {shown.label}" if shown.label not in ("-", "") else f"{label} {text}"


def channel_unit(si_unit: str) -> str:
    """The channel unit text of an SI key token (``rad_s`` -> ``rad/s``)."""
    return _SI_TOKEN_TO_UNIT.get(si_unit, si_unit)


__all__ = ["DASH", "as_float", "fixed", "num", "signed", "integer", "mm", "um", "signed_um", "degc", "pct",
           "seconds", "duration", "kwh", "kj", "eur", "rpm", "grams", "litres", "yes_no", "DisplayUnit",
           "display_unit", "quantity", "channel_unit", "MM_PER_M", "UM_PER_M", "KWH_PER_J", "RPM_PER_RAD_S",
           "MM_MIN_PER_M_S", "MM3_PER_M3", "L_MIN_PER_M3_S", "DEG_PER_RAD", "MIN_PER_S", "BAR_PER_PA"]
