"""Heat generation by zone and its partition (docs/models/mechanics.md, "Heat").

Zones per unit width of cut: primary shear F_S V_s, rake friction F V_c (both from the Oxley
solution; F_S V_s + F V_c = F_C V exactly), flank rubbing tau_f VB V and edge ploughing
F_te V. Partitions:

* shear zone: fraction beta to the work (Boothroyd / Oxley), 1 - beta to the chip;
* rake: fraction R to the chip from equal mean interface temperatures of the chip (moving
  source, Oxley's dT_M) and the tool (stationary source on a quarter-infinite tool, Loewen &
  Shaw), solved inside the Oxley interface state (:func:`kernels.interface_state`);
* flank and edge: Blok's principle with Jaeger's moving band source on the work and the
  stationary source on the tool flank.
"""

from __future__ import annotations

import functools
import math

import numpy as np
from scipy import integrate, special

from . import abi
from . import kernels as K
from . import params as P
from .materials import ToolSide, WorkMaterial


@functools.lru_cache(maxsize=4096)
def band_mean_factor(peclet: float) -> float:
    """Mean surface temperature rise of a band source of half-width l moving at V over a
    half-space (Jaeger 1942), in units of q l / k, for L = V l / (2 kappa):
    (1 / (2 pi L^2)) int_0^{2L} e^{u}K0(u) (1 + e^{-2u}) (2L - u) du; -> 0.752 / sqrt(L) for large L."""
    L = float(peclet)
    if L <= 0.0:
        return math.inf

    def f(u):
        return special.k0e(u) * (1.0 + math.exp(-2.0 * u)) * (2.0 * L - u)

    split = min(1.0, 2.0 * L)
    total = integrate.quad(f, 0.0, split, limit=200)[0]
    if 2.0 * L > split:
        total += integrate.quad(f, split, 2.0 * L, limit=200)[0]
    return total / (2.0 * math.pi * L * L)


def stationary_rect_factor(aspect: float) -> float:
    """Mean rise of a uniformly heated a x b rectangle on a half-space in units of q a / k (b = aspect a)."""
    return float(K.rect_area_factor(float(aspect)))


def moving_strip_factor(peclet: float, aspect: float) -> float:
    """Mean rise of a strip of length 2l (along the motion) and width w = aspect 2l moving at
    Peclet number L, in units of q l / k: the infinite band (Jaeger) capped by the stationary
    rectangle 2 A(aspect), its finite-width limit at low speed."""
    return min(band_mean_factor(peclet), 2.0 * stationary_rect_factor(aspect))


def work_side_resistance(length_m: float, width_m: float, speed_m_s: float, work: WorkMaterial,
                         temp_K: float) -> float:
    """Mean surface temperature rise of the work per unit flux (m2 K / W) under a contact of
    ``length_m`` along the motion sliding at ``speed_m_s``."""
    half = 0.5 * length_m
    kappa = work.diffusivity(temp_K)
    peclet = speed_m_s * half / (2.0 * kappa)
    return moving_strip_factor(peclet, width_m / length_m) * half / work.conductivity(temp_K)


def tool_side_resistance(length_m: float, temp_K: float, tool: ToolSide, width_m: float | None = None,
                         contact_time_s: float | None = None) -> float:
    """Mean temperature rise of the tool face per unit flux (m2 K / W) over a contact adjacent to
    the edge (quarter-infinite tool by an image source, transient cap, coating resistance)."""
    vec = np.zeros(abi.N_TOOL)
    vec[abi.TOOL_RHO_KG_M3] = tool.rho_kg_m3
    vec[abi.TOOL_R_COAT_M2K_W] = tool.coating_resistance_m2K_W
    vec[abi.TOOL_WIDTH_M] = float(width_m) if width_m else 0.0
    vec[abi.TOOL_TIME_S] = float(contact_time_s) if contact_time_s else 0.0
    prm = P.oxley_vector(-1.0, 0.5)
    return float(K.tool_resistance(float(length_m), float(temp_K), vec, tool.k_T_K, tool.k_W_mK, tool.cp_T_K,
                                   tool.cp_J_kgK, prm))


def blok_partition_to_tool(flux_W_m2: float, work_resistance: float, tool_resistance: float, work_surface_K: float,
                           tool_surface_K: float) -> float:
    """Fraction of a sliding contact's heat entering the tool from equal mean temperatures:
    T_ws + (1 - f) q M_w = T_t + f q M_t (Blok 1937), clamped to [0, 1]."""
    if flux_W_m2 <= 0.0:
        return 0.0
    denom = flux_W_m2 * (work_resistance + tool_resistance)
    if denom <= 0.0:
        return 0.5
    frac = (work_surface_K - tool_surface_K + flux_W_m2 * work_resistance) / denom
    return min(max(frac, 0.0), 1.0)


def flank_partition_to_tool(work: WorkMaterial, tool: ToolSide | None, land_m: float, width_m: float,
                            speed_m_s: float, flux_W_m2: float, work_surface_K: float, tool_flank_K: float | None,
                            contact_time_s: float | None = None) -> float:
    """Share of flank-land (or edge-rubbing) heat entering the tool. An adiabatic tool
    (no tool data or temperature) takes none."""
    if tool is None or tool_flank_K is None or land_m <= 0.0 or speed_m_s <= 0.0:
        return 0.0
    mw = work_side_resistance(land_m, width_m, speed_m_s, work, work_surface_K)
    mt = tool_side_resistance(land_m, tool_flank_K, tool, width_m, contact_time_s)
    return blok_partition_to_tool(flux_W_m2, mw, mt, work_surface_K, tool_flank_K)


def zone_powers(result_vector: np.ndarray, speed_m_s: float) -> tuple[float, float]:
    """(primary shear, rake friction) power per unit width (W/m) of an Oxley output vector."""
    shear = float(result_vector[abi.OX_FS] * result_vector[abi.OX_VS])
    rake = float(result_vector[abi.OX_F] * result_vector[abi.OX_VC])
    return shear, rake


def chip_temperature_rise(power_to_chip_W_per_m: float, work: WorkMaterial, t1_m: float, speed_m_s: float,
                          temp_K: float) -> float:
    """Mean temperature rise of the chip from the heat it receives (per unit width)."""
    return power_to_chip_W_per_m / (work.rho_kg_m3 * speed_m_s * t1_m * work.cp(temp_K))


__all__ = ["band_mean_factor", "stationary_rect_factor", "moving_strip_factor", "work_side_resistance",
           "tool_side_resistance", "blok_partition_to_tool", "flank_partition_to_tool", "zone_powers",
           "chip_temperature_rise"]
