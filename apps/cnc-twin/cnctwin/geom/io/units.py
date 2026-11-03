"""Length units of mesh files: the scale from a file unit to metres (meshes are SI inside)."""

from __future__ import annotations

from ...units import FOOT_M, INCH_M

# file unit name -> metres per unit (3MF names, STEP/CAD names and the key unit tokens)
UNIT_SCALE_M = {
    "m": 1.0, "meter": 1.0, "metre": 1.0,
    "mm": 1e-3, "millimeter": 1e-3, "millimetre": 1e-3,
    "cm": 1e-2, "centimeter": 1e-2, "centimetre": 1e-2,
    "um": 1e-6, "micron": 1e-6, "micrometer": 1e-6, "micrometre": 1e-6,
    "in": INCH_M, "inch": INCH_M,
    "ft": FOOT_M, "foot": FOOT_M,
}
# canonical 3MF unit attribute per scale
THREEMF_UNITS = {1e-6: "micron", 1e-3: "millimeter", 1e-2: "centimeter", INCH_M: "inch", FOOT_M: "foot", 1.0: "meter"}


def unit_scale(unit) -> float:
    """Metres per file unit for a unit name (``mm``, ``inch``, ``millimeter`` ...) or a number (metres)."""
    if isinstance(unit, (int, float)) and not isinstance(unit, bool):
        if unit <= 0:
            raise ValueError("a unit scale must be positive")
        return float(unit)
    key = str(unit).strip().lower()
    if key not in UNIT_SCALE_M:
        raise ValueError(f"unknown length unit {unit!r} (one of {', '.join(sorted(UNIT_SCALE_M))})")
    return UNIT_SCALE_M[key]


def unit_name(scale_m: float) -> str:
    """Short name of a unit scale (mm, m, in ...), or the scale in metres as text."""
    for name in ("mm", "m", "cm", "um", "in", "ft"):
        if abs(UNIT_SCALE_M[name] - scale_m) <= 1e-12 * scale_m:
            return name
    return repr(float(scale_m))


__all__ = ["UNIT_SCALE_M", "THREEMF_UNITS", "unit_scale", "unit_name"]
