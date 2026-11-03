"""SI units inside, unit-suffixed keys at the interfaces.

Database and project files name the unit in the key (``travel_mm``, ``power_kW``,
``max_rpm``). :func:`to_si` turns such a key and value into the SI key and value the
simulator uses internally (``travel_m``, ``power_W``, ``max_rad_s``). Hardness scales,
money and dimensionless keys pass through unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Physical constants (CODATA 2018 / ISO 80000)
STANDARD_GRAVITY_M_S2 = 9.80665
STEFAN_BOLTZMANN_W_M2K4 = 5.670374419e-8
GAS_CONSTANT_J_MOLK = 8.314462618
BOLTZMANN_J_K = 1.380649e-23
ZERO_DEGC_K = 273.15
REFERENCE_TEMP_K = 293.15          # ISO 1: geometrical product specifications refer to 20 degC
STANDARD_ATMOSPHERE_PA = 101325.0
AIR_DENSITY_20C_KG_M3 = 1.204      # dry air at 20 degC, 1 atm
AIR_VISCOSITY_20C_PA_S = 1.81e-5
AIR_CONDUCTIVITY_20C_W_MK = 0.0257
AIR_CP_20C_J_KGK = 1005.0
WATER_LATENT_HEAT_J_KG = 2.257e6   # at 100 degC

INCH_M = 0.0254
FOOT_M = 0.3048
POUND_FORCE_N = 4.4482216152605
FOOT_POUND_FORCE_NM = 1.3558179483314
HORSEPOWER_W = 745.69987158227
KILOGRAM_FORCE_N = 9.80665
PSI_PA = 6894.757293168


@dataclass(frozen=True)
class Suffix:
    """One interface unit: the key suffix, the SI suffix it maps to, and the conversion."""

    suffix: str
    si_suffix: str
    factor: float
    offset: float = 0.0


# Longest suffix wins, so compound units are listed explicitly.
SUFFIXES: tuple[Suffix, ...] = (
    # length
    Suffix("_mm", "_m", 1e-3),
    Suffix("_um", "_m", 1e-6),
    Suffix("_nm", "_m", 1e-9),
    Suffix("_cm", "_m", 1e-2),
    Suffix("_in", "_m", INCH_M),
    Suffix("_m", "_m", 1.0),
    Suffix("_um_per_m", "_m_per_m", 1e-6),
    Suffix("_um_per_K", "_m_per_K", 1e-6),
    Suffix("_mm2", "_m2", 1e-6),
    Suffix("_cm2", "_m2", 1e-4),
    Suffix("_mm3", "_m3", 1e-9),
    Suffix("_per_mm", "_per_m", 1e3),
    Suffix("_per_mm2", "_per_m2", 1e6),
    # angle
    Suffix("_deg", "_rad", math.pi / 180.0),
    Suffix("_arcsec", "_rad", math.pi / (180.0 * 3600.0)),
    Suffix("_urad", "_rad", 1e-6),
    Suffix("_rad", "_rad", 1.0),
    # speeds
    Suffix("_rpm", "_rad_s", 2.0 * math.pi / 60.0),
    Suffix("rpm", "rad_s", 2.0 * math.pi / 60.0),
    Suffix("_mm_min", "_m_s", 1e-3 / 60.0),
    Suffix("_m_min", "_m_s", 1.0 / 60.0),
    Suffix("_ipm", "_m_s", INCH_M / 60.0),
    Suffix("_sfm", "_m_s", FOOT_M / 60.0),
    Suffix("_m_s", "_m_s", 1.0),
    Suffix("_mm_rev", "_m_per_rev", 1e-3),
    Suffix("_mm_tooth", "_m_per_tooth", 1e-3),
    Suffix("_ipr", "_m_per_rev", INCH_M),
    Suffix("_ipt", "_m_per_tooth", INCH_M),
    Suffix("_m_s2", "_m_s2", 1.0),
    Suffix("_m_s3", "_m_s3", 1.0),
    Suffix("_rad_s2", "_rad_s2", 1.0),
    Suffix("_rad_s3", "_rad_s3", 1.0),
    # force, torque, stress, stiffness
    Suffix("_N", "_N", 1.0),
    Suffix("_kN", "_N", 1e3),
    Suffix("_kgf", "_N", KILOGRAM_FORCE_N),
    Suffix("_lbf", "_N", POUND_FORCE_N),
    Suffix("_Nm", "_Nm", 1.0),
    Suffix("_ft_lbf", "_Nm", FOOT_POUND_FORCE_NM),
    Suffix("_Pa", "_Pa", 1.0),
    Suffix("_kPa", "_Pa", 1e3),
    Suffix("_MPa", "_Pa", 1e6),
    Suffix("_GPa", "_Pa", 1e9),
    Suffix("_bar", "_Pa", 1e5),
    Suffix("_psi", "_Pa", PSI_PA),
    Suffix("_N_mm2", "_Pa", 1e6),
    Suffix("_N_cm2", "_Pa", 1e4),
    Suffix("_N_mm", "_N_per_m", 1e3),
    Suffix("_N_um", "_N_per_m", 1e6),
    Suffix("_N_mm3", "_N_per_m3", 1e9),
    Suffix("_N_per_m", "_N_per_m", 1.0),
    Suffix("_mN_m", "_N_per_m", 1e-3),
    Suffix("_Nm_rad", "_Nm_per_rad", 1.0),
    Suffix("_MPa_sqrt_m", "_Pa_sqrt_m", 1e6),
    # energy, power
    Suffix("_J", "_J", 1.0),
    Suffix("_kJ", "_J", 1e3),
    Suffix("_W", "_W", 1.0),
    Suffix("_kW", "_W", 1e3),
    Suffix("_hp", "_W", HORSEPOWER_W),
    # temperature
    Suffix("_degC", "_K", 1.0, ZERO_DEGC_K),
    Suffix("_K", "_K", 1.0),
    Suffix("_dK", "_dK", 1.0),          # temperature difference (no offset)
    # mass, density, flow, volume
    Suffix("_kg", "_kg", 1.0),
    Suffix("_g", "_kg", 1e-3),
    Suffix("_kg_m3", "_kg_m3", 1.0),
    Suffix("_g_cm3", "_kg_m3", 1e3),
    Suffix("_l", "_m3", 1e-3),
    Suffix("_l_min", "_m3_s", 1e-3 / 60.0),
    Suffix("_ml_h", "_m3_s", 1e-6 / 3600.0),
    Suffix("_ml_per_kg_chips", "_m3_per_kg", 1e-6),
    Suffix("_l_per_h_per_m2", "_m3_s_per_m2", 1e-3 / 3600.0),
    Suffix("_kg_h", "_kg_s", 1.0 / 3600.0),
    # time, frequency
    Suffix("_s", "_s", 1.0),
    Suffix("_ms", "_s", 1e-3),
    Suffix("_us", "_s", 1e-6),
    Suffix("_min", "_s", 60.0),
    Suffix("_h", "_s", 3600.0),
    Suffix("_Hz", "_Hz", 1.0),
    Suffix("_kHz", "_Hz", 1e3),
    # thermal and transport properties (already SI)
    Suffix("_J_kgK", "_J_kgK", 1.0),
    Suffix("_W_mK", "_W_mK", 1.0),
    Suffix("_W_m2K", "_W_m2K", 1.0),
    Suffix("_1_K", "_1_K", 1.0),
    Suffix("_kgm2", "_kgm2", 1.0),
    Suffix("_mPa_s", "_Pa_s", 1e-3),
    Suffix("_Pa_s", "_Pa_s", 1.0),
    # fractions
    Suffix("_pct", "_frac", 1e-2),
)

_BY_LENGTH = sorted(SUFFIXES, key=lambda s: len(s.suffix), reverse=True)

# Keys that look like a unit suffix but are not one (hardness scales, codes, money, counts).
PASS_THROUGH_SUFFIXES = ("_HB", "_HV", "_HRC", "_HRB", "_eur", "_eur_per_l", "_count", "_id")


def find_suffix(key: str) -> Suffix | None:
    """Return the interface unit a key ends with, or None for a dimensionless key."""
    if key.endswith(PASS_THROUGH_SUFFIXES):
        return None
    for suffix in _BY_LENGTH:
        if key.endswith(suffix.suffix) and (len(key) > len(suffix.suffix) or suffix.suffix == "rpm"):
            return suffix
    return None


def to_si(key: str, value):
    """Convert one interface key/value to its SI key/value.

    Numbers, lists of numbers and nested lists convert element-wise; strings and None pass
    through. A key without a unit suffix is returned unchanged.
    """
    suffix = find_suffix(key)
    if suffix is None:
        return key, value
    si_key = key[: len(key) - len(suffix.suffix)] + suffix.si_suffix
    return si_key, _convert(value, suffix)


def _convert(value, suffix: Suffix):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_convert(v, suffix) for v in value]
    return float(value) * suffix.factor + suffix.offset


def degC_to_K(temp_degC: float) -> float:
    return temp_degC + ZERO_DEGC_K


def K_to_degC(temp_K: float) -> float:
    return temp_K - ZERO_DEGC_K


def rpm_to_rad_s(speed_rpm: float) -> float:
    return speed_rpm * 2.0 * math.pi / 60.0


def rad_s_to_rpm(omega_rad_s: float) -> float:
    return omega_rad_s * 60.0 / (2.0 * math.pi)


def m_min_to_m_s(speed_m_min: float) -> float:
    return speed_m_min / 60.0


def m_s_to_m_min(speed_m_s: float) -> float:
    return speed_m_s * 60.0


def mm_min_to_m_s(feed_mm_min: float) -> float:
    return feed_mm_min * 1e-3 / 60.0


def m_s_to_mm_min(feed_m_s: float) -> float:
    return feed_m_s * 60.0 * 1e3
