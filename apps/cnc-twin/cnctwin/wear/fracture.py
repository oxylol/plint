"""Strength, fracture probability and fatigue of tool bodies and edges (docs/models/wear.md 7-10).

* Weibull (1951) weakest-link strength: P_f = 1 - exp(-(V_eff / V_ref)(sigma / sigma_0)^m). The
  database TRS is the mean bend strength of the three-point test piece, whose effective volume is
  V_ref = L b d / (2 (m + 1)^2) (ASTM C1683), so sigma_0 = TRS / Gamma(1 + 1/m) at V_ref.
* Effective volume of a round cantilever section of diameter d over the moment arm L: bending
  pi d^2 / 4 L k(m) / (m + 1), k(m) = Gamma((m + 1)/2) / (2 sqrt(pi) Gamma(m/2 + 2)) (tension side of
  the circle, linearly growing moment); torsion pi d^2 / 4 L 2 / (m + 2) (shear stress growing with r).
* Section criteria: brittle classes (carbide, cermet, ceramic, cBN, PCD) by the largest principal stress
  sigma/2 + sqrt(sigma^2/4 + tau^2) (Rankine; pure torsion fails at tau = strength); high-speed steels by
  von Mises sqrt(sigma^2 + 3 tau^2) (pure torsion at strength / sqrt 3).
* Fatigue: Basquin (1910) sigma_a = sigma_f' (2 N)^b with Smith-Watson-Topper's equivalent amplitude
  sqrt(sigma_max sigma_a) (rotating bending: sigma; pulsating 0 -> sigma: sigma / sqrt 2), Miner's sum.
  Hardmetal-type classes: sigma_f' = TRS, b from the 70 % fatigue limit at 1e7 cycles (Nakajima et al.
  2006); high-speed steels: the modified universal slopes (Muralidharan & Manson 1988).
* Thermal cycling of the edge has its own curve (thermal_basquin: comb cracks form after thousands of
  cycles at a quarter of the TRS, an estimate documented in wear-fracture.yaml).
* Comb cracks: the initial flaw of the edge a0 = (K_IC / (Y sigma_0))^2 / pi (the crack that fails at the
  characteristic strength), crack strength K_IC / (Y sqrt(pi a)) (Tada, Paris & Irwin 2000).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .params import FRACTURE, TOOL_CLASSES_BRITTLE, get


def reference_volume_m3(m: float) -> float:
    """Effective volume of the three-point TRS test piece at Weibull modulus m."""
    p = get(FRACTURE)
    return p["trs_specimen_width_m"] * p["trs_specimen_height_m"] * p["trs_span_m"] / (2.0 * (m + 1.0) ** 2)


def characteristic_strength_Pa(trs_Pa: float, m: float) -> float:
    """sigma_0 at V_ref whose Weibull mean is the TRS."""
    return trs_Pa / math.gamma(1.0 + 1.0 / m)


def circle_tension_factor(m: float) -> float:
    """(1 / A) int over the tension half of a circle of (y / R)^m dA."""
    return math.exp(math.lgamma(0.5 * (m + 1.0)) - math.lgamma(0.5 * m + 2.0)) / (2.0 * math.sqrt(math.pi))


def section_volume_m3(d_m: float, arm_m: float, m: float, torsion: bool) -> float:
    area = math.pi * d_m * d_m / 4.0
    if torsion:
        return area * arm_m * 2.0 / (m + 2.0)
    return area * arm_m * circle_tension_factor(m) / (m + 1.0)


def strength_Pa(volume_m3: float, sigma0_Pa: float, vref_m3: float, m: float, quantile: float = 0.5) -> float:
    """Strength of a volume at the failure-probability quantile (0.5: median)."""
    q = min(max(quantile, 1e-12), 1.0 - 1e-12)
    return sigma0_Pa * (vref_m3 / volume_m3) ** (1.0 / m) * (-math.log1p(-q)) ** (1.0 / m)


def failure_probability(sigma_Pa: float, volume_m3: float, sigma0_Pa: float, vref_m3: float, m: float) -> float:
    if sigma_Pa <= 0.0 or volume_m3 <= 0.0:
        return 0.0
    return -math.expm1(-(volume_m3 / vref_m3) * (sigma_Pa / sigma0_Pa) ** m)


def equivalent_stress_Pa(sigma_b: float, tau: float, tool_class: str) -> float:
    if tool_class in TOOL_CLASSES_BRITTLE:
        return 0.5 * sigma_b + math.sqrt(0.25 * sigma_b * sigma_b + tau * tau)
    return math.sqrt(sigma_b * sigma_b + 3.0 * tau * tau)


@dataclass(frozen=True)
class Basquin:
    sigma_f_Pa: float
    b: float
    src: str

    def cycles(self, amplitude_Pa: float) -> float:
        """Cycles to failure at a (fully reversed, SWT-equivalent) amplitude. Basquin's curve ends at one reversal
        (N_f = 1/2) at sigma_f': a load at or above it fails in its first reversal (the static fracture the section
        criterion reports), so its damage is that of one reversal, not the curve extrapolated beyond its end (a
        blown-up entry gave a Miner sum of 1.7e24)."""
        if amplitude_Pa <= 0.0:
            return math.inf
        return 0.5 * (min(amplitude_Pa, self.sigma_f_Pa) / self.sigma_f_Pa) ** (1.0 / self.b)


def basquin(tool_class: str, trs_Pa: float, youngs_Pa: float) -> Basquin:
    p = get(FRACTURE)
    if tool_class == "hss":
        sigma_f = p["steel_basquin_coeff"] * youngs_Pa * (trs_Pa / youngs_Pa) ** p["steel_basquin_strength_exponent"]
        return Basquin(sigma_f, p["steel_basquin_exponent"], "muralidharan-manson-1988-universal-slopes")
    b = math.log(p["hardmetal_fatigue_ratio"]) / math.log(2.0 * p["hardmetal_fatigue_cycles"])
    return Basquin(trs_Pa, b, "nakajima-2006-wc-co-fatigue")


def thermal_basquin(tool_class: str, trs_Pa: float, youngs_Pa: float) -> Basquin:
    """Fatigue curve of thermal cycling of the edge (comb cracks): hardmetal-type classes through
    (thermal_fatigue_cycles, thermal_fatigue_ratio x TRS) from TRS at one reversal; high-speed steels
    their isothermal curve."""
    p = get(FRACTURE)
    if tool_class == "hss":
        return basquin(tool_class, trs_Pa, youngs_Pa)
    b = math.log(p["thermal_fatigue_ratio"]) / math.log(2.0 * p["thermal_fatigue_cycles"])
    return Basquin(trs_Pa, b, "est (wear-fracture.yaml thermal_fatigue_ratio)")


def swt_amplitude(peak_Pa: float, rotating_bending: bool) -> float:
    """Equivalent fully reversed amplitude: rotating bending (+-peak) or pulsating (0 -> peak)."""
    return peak_Pa if rotating_bending else peak_Pa / math.sqrt(2.0)


def initial_flaw_m(kic_Pa_sqrt_m: float, sigma0_Pa: float) -> float:
    y = get(FRACTURE)["crack_shape_factor"]
    return (kic_Pa_sqrt_m / (y * sigma0_Pa)) ** 2 / math.pi


__all__ = ["reference_volume_m3", "characteristic_strength_Pa", "circle_tension_factor", "section_volume_m3",
           "strength_Pa", "failure_probability", "equivalent_stress_Pa", "Basquin", "basquin", "thermal_basquin",
           "swt_amplitude", "initial_flaw_m"]
