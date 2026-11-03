"""Acoustic relations of the sound model (docs/models/sound.md "Radiation and propagation" and "Sources").

Every function is SI in and out. Frequencies in Hz, levels in dB re 20 uPa (pressure) or 1 pW (power).

* Radiation efficiency of a compact body: the rigid sphere, monopole ``(ka)^2 / (1 + (ka)^2)`` and dipole
  ``(ka)^4 / (4 + (ka)^4)`` (exact for spheres; Kinsler et al. 2000, Fahy & Gardonio 2007).
* Structure-borne path (statistical energy analysis, Lyon & DeJong 1995): a force F at a plate-like
  structure injects ``F_rms^2 Re(Y)`` with the infinite-plate point mobility ``Y = 1 / (8 sqrt(B m''))``
  (Cremer, Heckl & Petersson 2005); the structure radiates ``eta_rad / (eta_rad + eta_int)`` of it, with the
  radiation loss factor ``eta_rad = rho c sigma / (omega m'')`` and sigma of the sphere of the same area.
* Compact dipole path: a rigid body of radius a moving with velocity U radiates ``rho c (4 pi a^2 / 3) sigma U^2``.
* Listener: ``p_rms^2 = rho c W (Q / (4 pi r^2) + 4 / R)`` with the room constant ``R = S a / (1 - a)``
  (Hopkins & Stryker 1948; Beranek & Ver 1992).
* Enclosure: field-incidence mass law of sheet and window, leaks, internal absorption:
  ``IL = TL_composite + 10 log10(a_enclosure)``, not below 0 dB.
* Empirical sound power of electric motors and pumps (Bies & Hansen 2003), subsonic air jets (Lighthill 1952),
  splashing liquid (an acoustic efficiency of the stream's kinetic power), bubble resonance (Minnaert 1933),
  Hertzian impact duration (Johnson 1985), rolling bearing ball passage (Harris & Kotzalas 2006),
  residual unbalance (ISO 21940-11), A-weighting (IEC 61672-1).
"""

from __future__ import annotations

import math

import numpy as np

from ..units import AIR_DENSITY_20C_KG_M3, STANDARD_ATMOSPHERE_PA, rad_s_to_rpm
from . import params as P

# Hertz impact of a sphere on a half-space: T = 2.94 delta_max / V with delta_max = (15 m V^2 / (16 E* R^1/2))^(2/5)
# gives T = 2.94 (15/16)^(2/5) (m^2 / (R E*^2 V))^(1/5) = 2.868 (...)^(1/5) (Johnson 1985, 11.4)
HERTZ_DURATION = 2.94 * (15.0 / 16.0) ** 0.4
MIN_RPM_FOR_LEVEL = 1.0          # the motor formulas use log10(rpm): below 1 rpm the motor is silent
KILOWATT_W = 1e3                 # the empirical formulas take powers in kW


def air() -> tuple[float, float]:
    """(density kg/m3, speed of sound m/s) of the air at 20 C."""
    return AIR_DENSITY_20C_KG_M3, float(P.number(P.SOUND, "speed_of_sound_m_s"))


# ----------------------------------------------------------------------------- radiation
def sigma_monopole(ka) -> np.ndarray:
    ka = np.asarray(ka, dtype=np.float64)
    return ka * ka / (1.0 + ka * ka)


def sigma_dipole(ka) -> np.ndarray:
    ka2 = np.asarray(ka, dtype=np.float64) ** 2
    return ka2 * ka2 / (4.0 + ka2 * ka2)


def sphere_radius_of_area(area_m2: float) -> float:
    return math.sqrt(area_m2 / (4.0 * math.pi))


def plate_mobility(youngs_Pa: float, density_kg_m3: float, poisson: float, thickness_m: float) -> float:
    """Real point mobility (m/(N s)) of an infinite thin plate: 1 / (8 sqrt(B m''))."""
    bending = youngs_Pa * thickness_m ** 3 / (12.0 * (1.0 - poisson * poisson))
    return 1.0 / (8.0 * math.sqrt(bending * density_kg_m3 * thickness_m))


def sea_power_per_force2(freq_Hz, youngs_Pa: float, density_kg_m3: float, poisson: float, thickness_m: float,
                         loss_factor: float, area_m2: float) -> np.ndarray:
    """Radiated sound power per mean-square force (W/N^2) of a plate-like structure driven at a point."""
    rho0, c0 = air()
    f = np.maximum(np.asarray(freq_Hz, dtype=np.float64), 1e-9)
    omega = 2.0 * math.pi * f
    surface = density_kg_m3 * thickness_m
    sigma = sigma_dipole(omega / c0 * sphere_radius_of_area(area_m2))
    eta_rad = rho0 * c0 * sigma / (omega * surface)
    return plate_mobility(youngs_Pa, density_kg_m3, poisson, thickness_m) * eta_rad / (eta_rad + loss_factor)


def modal_compliance(freq_Hz, modes) -> np.ndarray:
    """|H(f)| (m/N) of a sum of modes ``[(fn_Hz, zeta, k_N_per_m), ...]``."""
    f = np.asarray(freq_Hz, dtype=np.float64)
    total = np.zeros(f.shape, dtype=np.complex128)
    for fn, zeta, k in modes:
        r = f / fn
        total += 1.0 / (k * (1.0 - r * r + 2j * zeta * r))
    return np.abs(total)


def dipole_power_per_force2(freq_Hz, modes, radius_m: float) -> np.ndarray:
    """Sound power per mean-square force (W/N^2) of a compact rigid body (sphere of ``radius_m``) whose velocity is
    omega |H| F, H the modal compliance of ``modes``."""
    rho0, c0 = air()
    f = np.maximum(np.asarray(freq_Hz, dtype=np.float64), 1e-9)
    omega = 2.0 * math.pi * f
    velocity2 = (omega * modal_compliance(f, modes)) ** 2
    return rho0 * c0 * (4.0 * math.pi * radius_m ** 2 / 3.0) * sigma_dipole(omega / c0 * radius_m) * velocity2


def room_constant(surface_m2: float, absorption: float) -> float:
    return surface_m2 * absorption / (1.0 - absorption)


def listener_factor(distance_m: float, directivity: float, room_constant_m2: float) -> float:
    """``Q / (4 pi r^2) + 4 / R`` (1/m^2): p_rms^2 = rho c W x this."""
    return directivity / (4.0 * math.pi * distance_m ** 2) + 4.0 / room_constant_m2


def mass_law_tl(freq_Hz, surface_density_kg_m2: float, field_correction_dB: float) -> np.ndarray:
    """Field-incidence transmission loss (dB) of a limp panel below coincidence, not below 0."""
    rho0, c0 = air()
    f = np.asarray(freq_Hz, dtype=np.float64)
    tl0 = 10.0 * np.log10(1.0 + (math.pi * f * surface_density_kg_m2 / (rho0 * c0)) ** 2)
    return np.maximum(tl0 - field_correction_dB, 0.0)


def enclosure_power_transmission(freq_Hz, params: dict | None = None):
    """Power transmission factor 10^(-IL/10) of the closed enclosure, IL = TL_composite + 10 log10(a), IL >= 0."""
    p = P.sound() if params is None else params
    sheet = float(p["sheet_density_kg_m3"]) * float(p["sheet_thickness_m"])
    window = float(p["window_density_kg_m3"]) * float(p["window_thickness_m"])
    correction = float(p["mass_law_field_correction_dB"])
    fw = float(p["window_area_fraction"])
    fl = float(p["leak_area_fraction"])
    tau_sheet = 10.0 ** (-mass_law_tl(freq_Hz, sheet, correction) / 10.0)
    tau_window = 10.0 ** (-mass_law_tl(freq_Hz, window, correction) / 10.0)
    tau = (1.0 - fw - fl) * tau_sheet + fw * tau_window + fl
    return np.minimum(tau / float(p["enclosure_absorption"]), 1.0)


def a_weighting_dB(freq_Hz, params: dict | None = None) -> np.ndarray:
    p = P.sound() if params is None else params
    f1, f2, f3, f4 = (float(v) for v in p["a_weighting_poles_Hz"])
    f = np.maximum(np.asarray(freq_Hz, dtype=np.float64), 1e-12)
    f_sq = f * f
    ra = (f4 * f4 * f_sq * f_sq) / ((f_sq + f1 * f1) * np.sqrt((f_sq + f2 * f2) * (f_sq + f3 * f3)) * (f_sq + f4 * f4))
    return 20.0 * np.log10(ra) + float(p["a_weighting_offset_dB"])


def level_dB(p_rms_Pa: float) -> float:
    ref = float(P.number(P.SOUND, "reference_pressure_Pa"))
    return 20.0 * math.log10(max(p_rms_Pa, 1e-30) / ref)


# ----------------------------------------------------------------------------- empirical source powers
def power_from_level_at(level_dB_: float, distance_m: float) -> float:
    """Sound power (W) of a hemispherical point source with pressure level ``level_dB_`` at ``distance_m``."""
    rho0, c0 = air()
    ref = float(P.number(P.SOUND, "reference_pressure_Pa"))
    p_rms = ref * 10.0 ** (level_dB_ / 20.0)
    return p_rms * p_rms * 2.0 * math.pi * distance_m ** 2 / (rho0 * c0)


def motor_level_dB(rated_power_W: float, speed_rad_s, offset_dB: float = 0.0) -> np.ndarray:
    """Overall L_p at the formula distance of an electric motor (Bies & Hansen 2003); -inf when stopped."""
    p = P.sources()
    small = rated_power_W < float(p["motor_level_break_W"])
    a, b, c = (float(v) for v in p["motor_level_small_dB" if small else "motor_level_large_dB"])
    rpm = np.abs(rad_s_to_rpm(np.asarray(speed_rad_s, dtype=np.float64)))
    capped = np.minimum(rpm, rad_s_to_rpm(float(p["motor_formula_max_speed_rad_s"])))
    with np.errstate(divide="ignore"):
        level = a + b * math.log10(rated_power_W / KILOWATT_W) + c * np.log10(capped) + offset_dB
    return np.where(rpm >= MIN_RPM_FOR_LEVEL, level, -np.inf)


def motor_power_W(rated_power_W: float, speed_rad_s, offset_dB: float = 0.0) -> np.ndarray:
    level = motor_level_dB(rated_power_W, speed_rad_s, offset_dB)
    distance = float(P.number(P.SOURCES, "motor_level_distance_m"))
    finite = np.isfinite(level)
    out = np.zeros(level.shape)
    out[finite] = [power_from_level_at(float(v), distance) for v in level[finite]]
    return out


def pump_power_W(rated_power_W: float) -> float:
    a, b = (float(v) for v in P.sources()["pump_level_dB"])
    level = a + b * math.log10(rated_power_W / KILOWATT_W)
    return power_from_level_at(level, float(P.number(P.SOURCES, "motor_level_distance_m")))


def lighthill_power_W(speed_m_s: float, diameter_m: float) -> float:
    """Subsonic air jet: efficiency K M^5 of the kinetic power (1/2) rho U^3 (pi D^2 / 4)."""
    rho0, c0 = air()
    mach = min(speed_m_s / c0, 1.0)
    kinetic = 0.5 * rho0 * speed_m_s ** 3 * math.pi * diameter_m ** 2 / 4.0
    return float(P.number(P.SOURCES, "lighthill_efficiency")) * mach ** 5 * kinetic


def splash_power_W(flow_m3_s: float, speed_m_s: float, density_kg_m3: float) -> float:
    return float(P.number(P.SOURCES, "splash_efficiency")) * 0.5 * density_kg_m3 * flow_m3_s * speed_m_s ** 2


def minnaert_Hz(radius_m: float, liquid_density_kg_m3: float) -> float:
    gamma = float(P.number(P.SOURCES, "air_ratio_of_specific_heats"))
    return math.sqrt(3.0 * gamma * STANDARD_ATMOSPHERE_PA / liquid_density_kg_m3) / (2.0 * math.pi * radius_m)


def hertz_duration_s(mass_kg, radius_m, speed_m_s, youngs_1_Pa: float, poisson_1: float, youngs_2_Pa: float,
                     poisson_2: float) -> np.ndarray:
    """Contact time of a sphere (mass, radius) hitting a massive flat at ``speed_m_s``."""
    e_star = 1.0 / ((1.0 - poisson_1 ** 2) / youngs_1_Pa + (1.0 - poisson_2 ** 2) / youngs_2_Pa)
    m = np.asarray(mass_kg, dtype=np.float64)
    r = np.asarray(radius_m, dtype=np.float64)
    v = np.maximum(np.asarray(speed_m_s, dtype=np.float64), 1e-12)
    return HERTZ_DURATION * (m * m / (r * e_star * e_star * v)) ** 0.2


def cage_ratio(ball_d_m: float, pitch_d_m: float, contact_rad: float) -> float:
    """Cage speed / shaft speed (outer ring fixed)."""
    return 0.5 * (1.0 - ball_d_m / pitch_d_m * math.cos(contact_rad))


def ball_pass_outer_ratio(balls: int, ball_d_m: float, pitch_d_m: float, contact_rad: float) -> float:
    """BPFO / shaft frequency (the varying-compliance frequency of a healthy bearing)."""
    return balls * cage_ratio(ball_d_m, pitch_d_m, contact_rad)


def unbalance_force_N(rotor_mass_kg: float, grade_m_s: float, omega_max_rad_s: float, omega_rad_s) -> np.ndarray:
    """Residual unbalance force m e_per omega^2 with e_per = G / omega_max (ISO 21940-11)."""
    w = np.asarray(omega_rad_s, dtype=np.float64)
    return rotor_mass_kg * grade_m_s / max(omega_max_rad_s, 1e-9) * w * w


def chatter_force_N(amplitude_m, freq_Hz, mode) -> np.ndarray:
    """Force amplitude that drives the mode ``(fn_Hz, zeta, k)`` to ``amplitude_m`` at ``freq_Hz``: x |k (1 - r^2 +
    2 i zeta r)| (the inverse of the modal receptance)."""
    fn, zeta, k = mode
    r = np.asarray(freq_Hz, dtype=np.float64) / fn
    return np.asarray(amplitude_m, dtype=np.float64) * k * np.abs(1.0 - r * r + 2j * zeta * r)
