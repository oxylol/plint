"""Heat generation by zone and its partition (docs/models/mechanics.md, "Heat").

Zones per unit width of cut: primary shear F_S V_s, rake friction F V_c (both from the Oxley
solution; F_S V_s + F V_c = F_C V exactly), flank rubbing tau_f VB V and edge ploughing
F_te V. Partitions:

* shear zone: fraction beta to the work (Boothroyd / Oxley), 1 - beta to the chip;
* rake: fraction R to the chip from equal mean interface temperatures of the chip (moving
  source, Oxley's dT_M) and the tool (stationary source on a quarter-infinite tool, Loewen &
  Shaw), solved inside the Oxley interface state (:func:`kernels.interface_state`);
* flank and edge: Blok's principle with Jaeger's moving band source on the work and the
  stationary source on the tool flank, over the flank contact (wear land + edge arc), for the
  contact's friction heat and the plastic work of the edge pressure, the latter a source inside the
  work that reaches the interface in proportion to :func:`layer_source_factor`
  (:func:`flank_heat_to_tool`).
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


@functools.lru_cache(maxsize=4096)
def layer_band_mean_factor(peclet: float, depth: float) -> float:
    """Mean surface temperature rise over a band of half-width l moving at L = V l / (2 kappa) over a half-space with
    an adiabatic surface when its heat is generated uniformly in the layer 0 < z < delta beneath the band (Delta =
    V delta / (2 kappa) = ``depth``), in units of q l / k (q the heat per unit band area): the moving line source and its
    image (Carslaw & Jaeger 1959 10.7; Jaeger 1942) integrated over the layer,
    (1 / (2 pi L^2 Delta)) int_0^Delta dzeta int_0^{2L} (2L - u) (e^{-u} + e^{u}) K0(sqrt(u^2 + zeta^2)) du;
    -> band_mean_factor(L) as Delta -> 0."""
    L = float(peclet)
    D = float(depth)
    if L <= 0.0:
        return math.inf
    if D <= 0.0:
        return band_mean_factor(L)

    def inner(zeta):
        def f(u):
            r = math.hypot(u, zeta)
            if r <= 0.0:
                return 0.0
            return (2.0 * L - u) * special.k0e(r) * (math.exp(u - r) + math.exp(-u - r))
        split = min(max(zeta, 1e-12), 2.0 * L)
        total = integrate.quad(f, 0.0, split, limit=200)[0]
        if 2.0 * L > split:
            total += integrate.quad(f, split, 2.0 * L, limit=200)[0]
        return total

    outer = integrate.quad(inner, 0.0, D, limit=200)[0]
    return outer / (2.0 * math.pi * L * L * D)


def layer_source_factor(peclet: float, depth_ratio: float) -> float:
    """g = surface rise of heat generated uniformly to the depth depth_ratio x (2 l) under a band of length 2 l moving
    at Peclet number L, over the rise of the same heat at the surface (:func:`layer_band_mean_factor` /
    :func:`band_mean_factor`): 1 for a thin layer or a slow band (the heat reaches the surface during the passage),
    falling toward 0 when the band passes faster than the heat diffuses up from the layer. Evaluated at the Peclet
    number rounded to 4 significant digits (cached)."""
    L = float(f"{max(float(peclet), 1e-12):.4g}")
    ratio = float(f"{max(float(depth_ratio), 0.0):.4g}")
    if ratio <= 0.0:
        return 1.0
    band = band_mean_factor(L)
    if not math.isfinite(band) or band <= 0.0:
        return 1.0
    return min(max(layer_band_mean_factor(L, 2.0 * L * ratio) / band, 0.0), 1.0)


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


def flank_heat_to_tool(work: WorkMaterial, tool: ToolSide | None, length_m: float, width_m: float, speed_m_s: float,
                       friction_flux_W_m2: float, deformation_flux_W_m2: float, deformation_depth_m: float,
                       work_surface_K: float, tool_flank_K: float | None,
                       contact_time_s: float | None = None) -> tuple[float, dict]:
    """Share of the flank contact's heat (the friction on the edge arc and the wear land, plus the plastic work of the
    material pressed under the edge) that enters the tool, from equal mean temperatures of the two sides of the
    contact (Blok 1937), with the plastic work as a heat source inside the work:

        T_ws + M_w (q_f - H_t) + g M_w q_d = T_t + M_t H_t   ->   H_t = (T_ws - T_t + M_w (q_f + g q_d)) / (M_w + M_t)

    q_f: friction flux at the interface, q_d: plastic work per unit contact area generated uniformly to the depth
    ``deformation_depth_m`` in the work (layer_source_factor g: the surface rise it makes, relative to the same heat
    at the surface, Jaeger's moving band with the layer integrated), M_w the work side (moving band over the contact
    length, capped by the stationary rectangle), M_t the tool side (quarter-infinite, transient cap, coating). Returns
    (H_t / (q_f + q_d) clamped to [0, 1], details). An adiabatic tool (no tool data or temperature) takes none."""
    q_f = max(float(friction_flux_W_m2), 0.0)
    q_d = max(float(deformation_flux_W_m2), 0.0)
    total = q_f + q_d
    info = {"M_w": 0.0, "M_t": 0.0, "g": 1.0, "H_t": 0.0, "T_contact_K": float("nan")}
    if tool is None or tool_flank_K is None or length_m <= 0.0 or speed_m_s <= 0.0 or total <= 0.0:
        return 0.0, info
    mw = work_side_resistance(length_m, width_m, speed_m_s, work, work_surface_K)
    mt = tool_side_resistance(length_m, tool_flank_K, tool, width_m, contact_time_s)
    g = 1.0
    if q_d > 0.0 and deformation_depth_m > 0.0:
        peclet = speed_m_s * 0.5 * length_m / (2.0 * work.diffusivity(work_surface_K))
        g = layer_source_factor(peclet, deformation_depth_m / length_m)
    if mw + mt <= 0.0:
        return 0.5, info
    h_t = (work_surface_K - tool_flank_K + mw * (q_f + g * q_d)) / (mw + mt)
    frac = min(max(h_t / total, 0.0), 1.0)
    info.update(M_w=mw, M_t=mt, g=g, H_t=frac * total, T_contact_K=tool_flank_K + mt * frac * total)
    return frac, info


def zone_powers(result_vector: np.ndarray, speed_m_s: float) -> tuple[float, float]:
    """(primary shear, rake friction) power per unit width (W/m) of an Oxley output vector."""
    shear = float(result_vector[abi.OX_FS] * result_vector[abi.OX_VS])
    rake = float(result_vector[abi.OX_F] * result_vector[abi.OX_VC])
    return shear, rake


def chip_temperature_rise(power_to_chip_W_per_m: float, work: WorkMaterial, t1_m: float, speed_m_s: float,
                          temp_K: float) -> float:
    """Mean temperature rise of the chip from the heat it receives (per unit width)."""
    return power_to_chip_W_per_m / (work.rho_kg_m3 * speed_m_s * t1_m * work.cp(temp_K))


__all__ = ["band_mean_factor", "layer_band_mean_factor", "layer_source_factor", "stationary_rect_factor",
           "moving_strip_factor", "work_side_resistance", "tool_side_resistance", "blok_partition_to_tool",
           "flank_partition_to_tool", "flank_heat_to_tool", "zone_powers", "chip_temperature_rise"]
