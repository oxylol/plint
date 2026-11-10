"""Usui constants calibrated to the work material's Taylor life (docs/models/wear.md section 3).

The flank wear land grows as dVB/dt = g dVN/dt, g = cot(a) - tan(g_rake) (yen-2004-tool-wear-fem), with
the wear-depth rate (Takeyama & Murata 1963: thermally activated + mechanical term)

    dVN/dt = [f_c A sum_q w_q exp(-B / T_q) + sum_q w_q K a / H(T_q)] sigma_n V_s,

f_c the coating factor while the coating covers the contact (and T_q is below its service limit).
The life to the Taylor VB limit at constant conditions is the integral over the land,

    T(A, B; v) = int_{VB0}^{VB_lim} dVB / (g dVN/dt(VB, v)),

evaluated by composite Gauss-Legendre in VB (split where the integrand has a kink: the coating wearing through,
the hone radius, the plastic land's VB*; intervals no longer than 1 / ``calib_pieces`` of the land), with the response
(sigma_n V_s and the flank temperatures at the quadrature points) evaluated at every node through the
:class:`TaylorResponse` protocol, so stresses and temperatures may depend on the land. Two speeds v1,
v2 (the entry's vc range, else the speeds of the Taylor lives in wear-calibration.yaml) give two
equations: for each B, A follows from T(A, B; v1) = T_Taylor(v1) (closed form without the abrasive
term, a monotone root otherwise), and B is the root of ln(T(A(B), B; v2) / T_Taylor(v2)) (Brent, fixed
tolerances). The check speed v3 (the 15-min life) gives the reported prediction residual.

The life integral is edge **contact** time (the run advances each element by its own in-cut time), so
the targets are T_Taylor x the entry's contact fraction (``TaylorEntry.contact_life_s``): 1 for a
contact-time life and for a continuous cut (turning, boring, drilling, tapping: cutting time is contact
time); phi_ref / 2 pi for a milling life counted in cutting time, phi_ref the arc of the revolution the
edge is in the material in the entry's reference cut (``engaged_arc_ref_deg``, else the ISO 8688-2
reference end-milling cut, ``milling_reference_arc_rad``). An element engaged phi_e / 2 pi of each
revolution then reaches the VB limit after T_Taylor phi_ref / phi_e of cutting time: the Taylor life
itself at the reference engagement.

:class:`MechanicsResponse` is the default response: cnctwin.mechanics (Oxley with Johnson-Cook, or the
data-only model) for the chip and the rake contact, Smithey's flank-land stresses and, for the flank temperature,
the run's flank partition (mechanics.coefficients.flank_contact_heat: the land and edge-arc friction and the edge's
plastic work over the wear land plus the edge arc, Blok's equal temperatures with the Jaeger band source on the work
and the stationary source on the tool), with the work's flow stress in the contact softened at the contact's own
temperature (fixed point, :meth:`MechanicsResponse._softened_contact`).

:func:`anchor_published` scales the A of a published Usui set (B kept) so that this response reaches the
pairing's Taylor life at the 15-min speed: a published set belongs to the temperatures its source fitted it
with. The
integration may wire a response that runs cutting's coefficient pipeline and thermal's edge model at
the Taylor conditions instead (then the calibration and the run would use the same rake temperature too).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

import numpy as np
from scipy import optimize

from ..core.errors import InputError
from ..units import REFERENCE_TEMP_K
from .params import CALIBRATION, FLANK, CoatingData, TaylorEntry, get


@dataclass(frozen=True)
class Condition:
    """An equivalent continuous orthogonal cut at the Taylor reference conditions."""

    speed_m_s: float
    h_m: float                     # uncut chip thickness (the Taylor feed reference)
    depth_m: float                 # engaged edge length (the Taylor depth reference)
    VB_m: float                    # flank wear land
    rake_rad: float
    clearance_rad: float
    edge_radius_m: float
    incl_rad: float = 0.0
    coated: bool = True            # False: the response models the uncoated substrate (Taylor data of the substrate)


@dataclass(frozen=True)
class Response:
    """Wear-relevant state of a condition: sigma_n V_s on both faces and the contact temperatures at
    quadrature points (weights summing to 1)."""

    flank_sv_Pa_m_s: float
    flank_temp_K: np.ndarray
    rake_sv_Pa_m_s: float
    rake_temp_K: np.ndarray
    weights: np.ndarray
    flank_sigma_Pa: float = 0.0
    rake_sigma_Pa: float = 0.0
    chip_speed_ratio: float = 0.0


@runtime_checkable
class TaylorResponse(Protocol):
    def response(self, slot: int, cond: Condition) -> Response:
        """The response of tool slot ``slot`` at ``cond``."""


ABRASIVE_REDUCTIONS = 6          # quarterings of the abrasive floor tried before it is removed
FLANK_BISECTIONS = 60            # halvings of the flank-temperature bracket (below 1e-12 K for any bracket)


class CalibrationError(InputError):
    """The response model cannot evaluate the Taylor conditions."""


# ------------------------------------------------------------------------------------------ default response
class MechanicsResponse:
    """Default TaylorResponse from cnctwin.mechanics (see the module doc)."""

    def __init__(self, work, tool=None, ambient_K: float = REFERENCE_TEMP_K, friction_factor: float = 1.0,
                 substrate=None):
        self.work = work
        self.tool = tool
        self.substrate = substrate if substrate is not None else tool
        self.ambient_K = float(ambient_K)
        self.friction_factor = float(friction_factor)
        self._chip: dict = {}
        self.cache_key = ("mechanics", getattr(work, "id", ""), getattr(tool, "tool_class", ""),
                          getattr(tool, "coating", ""), getattr(self.substrate, "coating", ""), self.ambient_K,
                          self.friction_factor)

    def _side(self, cond: Condition):
        return self.tool if cond.coated else self.substrate

    def _chip_state(self, cond: Condition) -> tuple[float, float, float, float, float]:
        """(k_AB or flank sigma_0 marker, T_work_surface, T_rake, sigma_rake, chip speed) of the cut."""
        from ..mechanics import contact as C
        from ..mechanics import oxley
        from ..mechanics.coefficients import EdgeGroup, SlowState, group_coefficients

        key = (cond.speed_m_s, cond.h_m, cond.rake_rad, cond.clearance_rad, cond.edge_radius_m, cond.incl_rad,
               cond.coated)
        if key in self._chip:
            return self._chip[key]
        side = self._side(cond)
        if self.work.jc is None:
            gc = group_coefficients(self.work, side, EdgeGroup(cond.rake_rad, cond.incl_rad, cond.clearance_rad,
                                                                    cond.edge_radius_m),
                                    SlowState(cond.speed_m_s, work_temp_K=self.ambient_K,
                                              friction_factor=self.friction_factor),
                                    np.array([cond.h_m]), None)
            k_ref = gc.flank_sigma0_Pa / C.pressure_factor()
            if not k_ref > 0.0:
                # the data-only model's softened yield vanished at its (Merchant) temperature: the land then
                # presses on cold material, k = strength / sqrt(3) at 20 degC (same Prandtl pressure (2 + pi) k)
                strength = self.work.yield_Pa or self.work.ultimate_Pa
                if not strength:
                    raise CalibrationError(f"wear: material {getattr(self.work, 'id', '')} gives no flow stress or "
                                           "strength for the flank contact pressure")
                k_ref = float(strength) / math.sqrt(3.0)
            out = (k_ref, float(gc.T_shear_K[0]), float(gc.T_interface_K[0]), float(gc.interface_normal_Pa[0]),
                   cond.speed_m_s * float(gc.chip_ratio[0]))
        else:
            h_min = C.min_chip_thickness(self.work, cond.edge_radius_m)
            t1 = max(cond.h_m - h_min, 0.05 * cond.h_m)
            res = oxley.solve(self.work, t1, cond.speed_m_s, cond.rake_rad, work_temp_K=self.ambient_K,
                              tool=side, tool_temp_K=None, friction_factor=self.friction_factor)
            if res.status == "invalid":
                raise CalibrationError(f"wear: no chip-formation solution at v = {cond.speed_m_s:.3g} m/s, "
                                       f"h = {cond.h_m:.3g} m for material {getattr(self.work, 'id', '')}")
            out = (res.shear_flow_stress_Pa, res.T_shear_K, res.T_interface_K, res.interface_normal_Pa,
                   res.chip_speed_m_s)
        self._chip[key] = out
        return out

    def _thermal_factor(self, temp_K: float) -> float:
        """Flow stress of the work at ``temp_K`` relative to its value at the same strain and rate at 20 degC:
        Johnson-Cook's 1 - T*^m (T* clipped to [0, 1]), or the data-only model's linear fall to its softening
        temperature."""
        from ..mechanics.abi import MAT_M, MAT_TMELT_K, MAT_TREF_K
        from ..mechanics.materials import flow_stress_fallback

        jc = self.work.jc
        if jc is not None:
            span = float(jc[MAT_TMELT_K] - jc[MAT_TREF_K])
            ts = (temp_K - float(jc[MAT_TREF_K])) / span if span > 0.0 else 0.0
            ts = min(max(ts, 0.0), 1.0)
            return 1.0 - ts ** float(jc[MAT_M])
        strength = self.work.yield_Pa or self.work.ultimate_Pa
        return flow_stress_fallback(self.work, temp_K) / strength if strength else 1.0

    def flank_temperature(self, t_ws: float, t_base: float, t_ref_K: float, tau_Pa: float, speed: float,
                          m_w: float, m_t: float | None) -> float:
        """Mean flank-land temperature with the land's friction stress softened at that temperature.

        The land's shear stress tau (Smithey: min(mu sigma_0, k_AB) times the land's stress factor) is the work's
        shear flow stress at the shear plane, k_AB at T_AB; where the land runs hotter than the shear plane, the
        work under it is softer: tau(T) = tau x min(1, theta(T) / theta(T_AB)), theta the flow stress's thermal
        factor (Johnson-Cook 1 - T*^m). The heat flux tau(T) V and Blok's partition (Jaeger band source on the
        work, the stationary source on the tool) then fix T self-consistently, as Oxley's interface temperature
        is fixed with the chip's flow stress at that temperature; T stays below the melting temperature of the
        flow-stress law. The pressure sigma_0 is not softened: the hot layer at the land (Peclet depth
        sqrt(kappa VB / V), about ten microns) is thin against the plastic zone that carries it (of the order
        of VB). Bisection, fixed steps (deterministic).

        Blok's partition of the land's friction heat is taken in [0, 1] (mechanics.heat.blok_partition_to_tool): a
        small land, whose friction rises the work's surface less than the tool's base exceeds it, sits at the
        tool's base temperature (the rake-heated edge). Solving Blok's equality with the fraction unbounded (the
        tool face also giving heat to the colder work) was examined on 2026-11-09 and not adopted: the work
        surface entering the land is taken at the shear-plane temperature T_AB, without the heat of the edge's
        ploughing (the tertiary zone), so the unbounded flank of a small land follows T_AB, which hardly changes
        with speed; every Taylor calibration then gave activation temperatures of 50-100 kK (published Usui sets:
        5-30 kK) with check residuals of 6-34 %, and carbide in 6061 a law (B 43 kK, A 4e20 / Pa) that thermal's
        present flank temperature drives to kilometres of wear in one step (docs/models/wear.md 3, 17)."""
        from ..mechanics import heat as H

        theta_ref = self._thermal_factor(t_ref_K)

        def tool_side(temp_K: float) -> float:
            ratio = self._thermal_factor(temp_K) / theta_ref if theta_ref > 0.0 else 1.0
            flux = tau_Pa * min(1.0, ratio) * speed
            if m_t is None:
                return t_ws + flux * m_w
            share = H.blok_partition_to_tool(flux, m_w, m_t, t_ws, t_base)
            return t_base + share * flux * m_t

        lo = min(t_ws, t_base) if m_t is not None else t_ws
        hi = tool_side(lo)
        if not hi > lo:
            return hi
        for _ in range(FLANK_BISECTIONS):
            mid = 0.5 * (lo + hi)
            if tool_side(mid) > mid:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def interface_max_K(self) -> float | None:
        """The highest tool-work interface temperature the run's wear domain lets the law see: the work material's
        solidus (else liquidus, else its flow law's melting temperature; WearDomain._interface_max_K); None without
        one."""
        for value in (getattr(self.work, "solidus_K", None), getattr(self.work, "liquidus_K", None)):
            if value is not None and float(value) > 0.0:
                return float(value)
        jc = getattr(self.work, "jc", None)
        if jc is not None:
            from ..mechanics.abi import MAT_TMELT_K
            if float(jc[MAT_TMELT_K]) > 0.0:
                return float(jc[MAT_TMELT_K])
        return None

    def _edge_flow_stress(self, cond: Condition, k_chip: float) -> tuple[float, float]:
        """(shear flow stress of the material at the edge, h_min) as the run's coefficient rows take it
        (mechanics.coefficients._reference_shear_flow_stress): the primary-zone state at t1 = h_min with the strain
        limited to chip material's saturation strain; the data-only model's k (``k_chip``) as it is."""
        from ..mechanics import contact as C
        from ..mechanics import oxley
        from ..mechanics.abi import OX_EPS_AB
        from ..mechanics.coefficients import default_t1_grid, edge_shear_flow_stress

        h_min = C.min_chip_thickness(self.work, cond.edge_radius_m)
        if self.work.jc is None:
            return k_chip, h_min
        key = ("edge", cond.speed_m_s, cond.rake_rad, cond.edge_radius_m, cond.coated)
        if key not in self._chip:
            t1 = max(h_min, float(default_t1_grid()[0]))
            res = oxley.solve(self.work, t1, cond.speed_m_s, cond.rake_rad, work_temp_K=self.ambient_K,
                              tool=self._side(cond), tool_temp_K=None, friction_factor=self.friction_factor)
            k = res.shear_flow_stress_Pa if res.status != "invalid" else 0.0
            self._chip[key] = edge_shear_flow_stress(self.work, k, float(res.vector[OX_EPS_AB])) if k > 0.0 \
                else k_chip
        return self._chip[key], h_min

    def flank_contact(self, cond: Condition) -> dict:
        """The flank contact of the condition as the run's partition has it (mechanics.coefficients.flank_contact_heat,
        the share the coefficient rows give thermal): the wear land plus the edge arc below the stagnation point,
        the friction on both and the edge's plastic work with the buried-layer factor, Blok's equal mean
        temperatures between the work surface entering the contact (the shear plane's T_AB) and the tool flank
        (``flank_base_weight`` of the way from T_AB to the rake interface: the rake-heated edge). Returns the
        details of flank_contact_heat (at the shear plane's flow stress) plus ``T_flank_K`` (the contact's mean
        temperature, the tool face's at the partition, what thermal's flank contact mean is in the run, with the
        work's flow stress in the contact at that temperature), ``friction_softening`` (that flow stress over the
        shear plane's, s of :meth:`_softened_contact`), ``share``, ``length_m``, ``sigma_mean_Pa``, ``T_base_K``,
        ``T_work_surface_K``."""
        from ..mechanics import contact as C
        from ..mechanics import heat as H
        from ..mechanics.coefficients import flank_contact_heat
        from ..mechanics.oxley import effective_mu

        k_chip, t_ws, t_rake, _, _ = self._chip_state(cond)
        k_edge, h_min = self._edge_flow_stress(cond, k_chip)
        side = self._side(cond)
        mu = effective_mu(self.work, side, self.friction_factor)
        sigma0, tau0 = C.flank_reference_stresses(k_edge, mu)
        s_land, tau_land = C.flank_stresses(max(cond.VB_m, 0.0), sigma0, tau0)
        indentation = C.uses_indentation(0, cond.rake_rad)
        depth = cond.h_m if (indentation or cond.h_m < h_min) else h_min
        t_base = t_ws + get(FLANK)["flank_base_weight"] * (t_rake - t_ws)
        width = max(cond.depth_m, 1e-9)
        share, length, info = flank_contact_heat(self.work, side, cond.edge_radius_m, cond.rake_rad, cond.incl_rad,
                                                 depth, h_min, k_edge, mu, cond.VB_m, tau_land, cond.speed_m_s,
                                                 width, t_ws, t_base if side is not None else None, None,
                                                 indentation=indentation)
        if side is None and length > 0.0 and cond.speed_m_s > 0.0:
            # adiabatic tool: all the contact's heat goes into the work
            m_w = H.work_side_resistance(length, width, cond.speed_m_s, self.work, t_ws)
            q_d = info["q_deformation_W_m2"]
            g = 1.0
            if q_d > 0.0 and info["arc_m"] > 0.0:
                from ..mechanics import params as MP
                peclet = cond.speed_m_s * 0.5 * length / (2.0 * self.work.diffusivity(t_ws))
                g = H.layer_source_factor(peclet, MP.number(MP.CONTACT, "ploughing_zone_depth_ratio")
                                          * info["arc_m"] / length)
            t_flank = t_ws + m_w * (info["q_friction_W_m2"] + g * q_d)
        elif math.isfinite(info["T_contact_K"]):
            t_flank, soft = self._softened_contact(info, t_ws, t_base)
            info["friction_softening"] = soft
        else:
            t_flank = t_base
        info.update(T_flank_K=float(t_flank), share=float(share), length_m=float(length),
                    sigma_mean_Pa=float(s_land), sigma0_Pa=float(sigma0), tau_land_Pa=float(tau_land),
                    T_base_K=float(t_base), T_work_surface_K=float(t_ws), k_edge_Pa=float(k_edge))
        return info

    def _softened_contact(self, info: dict, t_ws: float, t_base: float) -> tuple[float, float]:
        """(contact temperature, flow-stress factor s) with the work's flow stress in the flank contact taken at the
        contact's own temperature: s = min(1, theta(T) / theta(T_AB)), theta the work's flow-stress thermal factor.
        The friction stress of the land and the arc is the work's shear flow stress at the interface, and the edge's
        pressure, (2 + pi) k, is carried by the ploughed zone just under it, whose heat (g near 1) reaches the
        interface during the passage: where the contact runs hotter than the shear plane both work against softer
        material, so both heats scale with s (the run's partition takes k at the shear plane's state). Blok's balance
        of the run's partition is linear in the fluxes, T = T_t + M_t H_t with H_t = clamp((T_ws - T_t + M_w s (q_f +
        g q_d)) / (s (q_f + q_d)), 0, 1) s (q_f + q_d), so the fixed point is found on that closed form (bisection,
        fixed steps); T stays below the melting temperature of the flow-stress law, where s = 0. Without it the
        contact of hard work materials at their Taylor speeds runs to the work's solidus at every speed (D2 at 60 HRC
        with PCBN, IN718 with ceramics: 1250-1260 degC flat), which leaves the calibration no speed dependence of the
        temperature to fit; with the friction alone softened the PCBN in D2 still sat at the solidus."""
        q_f, q_d = info["q_friction_W_m2"], info["q_deformation_W_m2"]
        m_w, m_t, g = info["M_w"], info["M_t"], info["g"]
        theta_ref = self._thermal_factor(t_ws)

        def contact(temp_K: float) -> tuple[float, float]:
            ratio = self._thermal_factor(temp_K) / theta_ref if theta_ref > 0.0 else 1.0
            soft = min(1.0, max(ratio, 0.0))
            total = soft * (q_f + q_d)
            if not total > 0.0 or not (m_w + m_t) > 0.0:
                return t_base, soft
            h_t = (t_ws - t_base + m_w * soft * (q_f + g * q_d)) / (m_w + m_t)
            frac = min(max(h_t / total, 0.0), 1.0)
            return t_base + m_t * frac * total, soft

        lo = t_base
        hi, soft = contact(lo)
        if not hi > lo:
            return hi, soft
        for _ in range(FLANK_BISECTIONS):
            mid = 0.5 * (lo + hi)
            if contact(mid)[0] > mid:
                lo = mid
            else:
                hi = mid
        return contact(0.5 * (lo + hi))[0], contact(0.5 * (lo + hi))[1]

    def response(self, slot: int, cond: Condition) -> Response:
        """sigma_n V_s on the land and the rake, and the flank and rake contact temperatures of the condition.

        The flank temperature is the flank contact's mean temperature from the run's partition (:meth:`flank_contact`;
        since 2026-11-10): the friction on the land and the edge arc and the edge's plastic work, over the contact
        thermal deposits the tool's share on, as the coefficient rows give it to thermal in the run, so that the
        Usui constants calibrated with this response and the run's flank temperatures belong to one partition; the
        work's flow stress in the contact is taken at the contact's temperature (:meth:`_softened_contact`). Until
        then the flank was Blok's share of the land friction alone over max(VB, r_e), softened at the land's
        temperature (:meth:`flank_temperature`, kept for comparison): 416-419 K at the aluminium side step of
        docs/convergence.md where this gives 437-465 K. Both temperatures are bounded by the work's solidus, as the
        wear domain bounds thermal's (:meth:`interface_max_K`)."""
        from ..mechanics import contact as C
        from ..mechanics.oxley import effective_mu

        k_chip, t_ws, t_rake, sigma_rake, v_chip = self._chip_state(cond)
        k_edge, _ = self._edge_flow_stress(cond, k_chip)
        side = self._side(cond)
        mu = effective_mu(self.work, side, self.friction_factor)
        sigma0, tau0 = C.flank_reference_stresses(k_edge, mu)
        if cond.VB_m > 0.0:
            s_mean, _ = C.flank_stresses(cond.VB_m, sigma0, tau0)
        else:
            s_mean = sigma0
        speed = cond.speed_m_s
        t_flank = self.flank_contact(cond)["T_flank_K"]
        cap = self.interface_max_K()
        if cap is not None:
            t_flank = min(t_flank, cap)
            t_rake = min(t_rake, cap)
        return Response(flank_sv_Pa_m_s=s_mean * speed, flank_temp_K=np.array([t_flank]),
                        rake_sv_Pa_m_s=sigma_rake * v_chip, rake_temp_K=np.array([t_rake]), weights=np.array([1.0]),
                        flank_sigma_Pa=s_mean, rake_sigma_Pa=sigma_rake,
                        chip_speed_ratio=v_chip / speed if speed > 0.0 else 0.0)


# ------------------------------------------------------------------------------------------ calibration
@dataclass(frozen=True)
class FlankGeometry:
    rake_rad: float
    clearance_rad: float
    edge_radius_m: float
    incl_rad: float
    g_vb: float                     # VB / VN
    c_coat: float                   # flank-coating depth at the edge per unit VN


@dataclass(frozen=True)
class Abrasion:
    """The Archard term K a / H(T) with its hardness tables (substrate, coating)."""

    coefficient: float              # K x abrasiveness
    substrate_T_K: np.ndarray
    substrate_Pa: np.ndarray
    coating_T_K: np.ndarray | None = None
    coating_Pa: np.ndarray | None = None

    def rate_factor(self, temps_K: np.ndarray, weights: np.ndarray, coated: bool) -> float:
        if self.coefficient <= 0.0:
            return 0.0
        if coated and self.coating_T_K is not None:
            hardness = np.interp(temps_K, self.coating_T_K, self.coating_Pa)
        else:
            hardness = np.interp(temps_K, self.substrate_T_K, self.substrate_Pa)
        return float(np.sum(weights * np.where(hardness > 0.0, self.coefficient / np.maximum(hardness, 1e-300), 0.0)))


@dataclass
class Calibration:
    A_per_Pa: float
    B_K: float
    abrasive_scale: float           # factor applied to the abrasive coefficient (1: as given)
    report: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _Node:
    weight_m: float                 # Gauss weight in VB
    coated: bool
    sv: float                       # sigma_n V_s
    temps: np.ndarray
    weights: np.ndarray
    coat_ok: np.ndarray             # quadrature points where the coating lowers the Usui rate
    abr: float                      # abrasive rate factor (1/Pa)


def _gauss(lo: float, hi: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (hi - lo) * x + 0.5 * (hi + lo), 0.5 * (hi - lo) * w


def land_breaks(vb0: float, vb_lim: float, geometry: FlankGeometry, coating: CoatingData | None,
                pieces: int = 1) -> tuple[list[float], float | None]:
    """(interval ends, coating breakthrough land) of the life integral over [vb0, vb_lim]: split where the integrand
    has a kink (the coating wearing through; the land reaching the hone radius, below which the heated contact is
    the rounded edge's; the land reaching the critical length VB* of the plastic land of mechanics' worn-tool
    contact, beyond which its mean stresses grow from a third of sigma_0, tau_0) and into intervals no longer than
    (vb_lim - vb0) / pieces. The wear rate varies by orders of magnitude over the land (Usui's Arrhenius factor of
    a flank that heats as the land grows): one Gauss rule over a kinked land had put Zhao's Ti6Al4V life to VB 0.3
    mm 10 % short of its own integral (148 against 164 s)."""
    from ..mechanics import params as MP

    points = {float(vb0), float(vb_lim)}
    vb_through = None
    if coating is not None and geometry.c_coat > 0.0:
        vb_through = coating.thickness_m / geometry.c_coat * geometry.g_vb
        points.add(vb_through)
    for p in (geometry.edge_radius_m, MP.number(MP.CONTACT, "flank_plastic_land_m")):
        points.add(float(p))
    ends = sorted(p for p in points if vb0 <= p <= vb_lim)
    longest = (vb_lim - vb0) / max(int(pieces), 1)
    out = [ends[0]]
    for lo, hi in zip(ends[:-1], ends[1:]):
        if hi - lo <= 1e-15:
            continue
        k = max(1, math.ceil((hi - lo) / longest - 1e-9))
        out.extend(lo + (hi - lo) * (i + 1) / k for i in range(k))
    return out, vb_through


def _nodes(response: Callable[[float, float], Response], speed: float, vb0: float, vb_lim: float,
           geometry: FlankGeometry, coating: CoatingData | None, abrasion: Abrasion, n: int,
           pieces: int = 1) -> list[_Node]:
    breaks, vb_through = land_breaks(vb0, vb_lim, geometry, coating, pieces)
    out = []
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        xs, ws = _gauss(lo, hi, n)
        for vb, w in zip(xs, ws):
            coated = coating is not None and (vb_through is None or vb < vb_through)
            r = response(speed, float(vb))
            if not r.flank_sv_Pa_m_s > 0.0:
                raise CalibrationError(f"wear: the response gives no flank contact stress at v = {speed:.3g} m/s, "
                                       f"VB = {vb:.3g} m")
            temps = np.asarray(r.flank_temp_K, dtype=float)
            weights = np.asarray(r.weights, dtype=float)
            coat_ok = (temps <= coating.max_service_K) if coated else np.zeros(temps.shape, bool)
            out.append(_Node(float(w), coated, float(r.flank_sv_Pa_m_s), temps, weights, coat_ok,
                             abrasion.rate_factor(temps, weights, coated)))
    return out


def _usui_factor(node: _Node, B: float, T_ref: float, coating: CoatingData | None) -> float:
    """sum_q w_q f_c exp(-B (1/T_q - 1/T_ref)): the Usui factor scaled by exp(B / T_ref), so that large
    activation temperatures neither underflow nor overflow (A' = A exp(-B / T_ref) carries the scale)."""
    factor = np.where(node.coat_ok, coating.factor if coating is not None else 1.0, 1.0)
    return float(np.sum(node.weights * factor * np.exp(-B * (1.0 / node.temps - 1.0 / T_ref))))


def life_s(nodes: list[_Node], A_scaled: float, B: float, T_ref: float, g_vb: float, coating: CoatingData | None,
           abrasive_scale: float = 1.0) -> float:
    """Time to wear the land over the nodes' interval at constant conditions, with A = A_scaled exp(B/T_ref)."""
    total = 0.0
    for node in nodes:
        rate = g_vb * (A_scaled * _usui_factor(node, B, T_ref, coating) + abrasive_scale * node.abr) * node.sv
        if not rate > 0.0:
            return math.inf
        total += node.weight_m / rate
    return total


def _solve_A(nodes, B, T_ref, target, g_vb, coating, scale) -> float:
    """Scaled A with life(A, B) = target (life decreases monotonically with A); 0 when the abrasive
    term alone already wears the land out in less than the target."""
    j = life_s(nodes, 1.0, B, T_ref, g_vb, coating, 0.0)
    if not math.isfinite(j):
        return 0.0
    if scale == 0.0 or all(n.abr == 0.0 for n in nodes):
        return j / target
    if life_s(nodes, 0.0, B, T_ref, g_vb, coating, scale) <= target:
        return 0.0
    hi = math.log(j / target) + 1.0

    def f(log_a):
        return math.log(life_s(nodes, math.exp(log_a), B, T_ref, g_vb, coating, scale) / target)

    lo = hi - 6.0
    while f(lo) <= 0.0:
        lo -= 6.0
    return math.exp(optimize.brentq(f, lo, hi, xtol=1e-13, rtol=1e-15, maxiter=500))


def calibration_speeds(entry: TaylorEntry) -> tuple[float, float, float]:
    p = get(CALIBRATION)
    f, ap = entry.feed_ref_m, entry.ap_ref_m
    if entry.vc_range_m_s is not None:
        v1, v2 = entry.vc_range_m_s
    else:
        v1 = entry.speed_for_life(p["calibration_life_long_s"], f, ap)
        v2 = entry.speed_for_life(p["calibration_life_short_s"], f, ap)
    v3 = entry.speed_for_life(p["check_life_s"], f, ap)
    return float(v1), float(v2), float(v3)


def calibrate(entry: TaylorEntry, response: Callable[[float, float], Response], geometry: FlankGeometry,
              coating: CoatingData | None, abrasion: Abrasion, vb0_m: float, nodes_per_interval: int,
              pieces: int = 1) -> Calibration:
    """A and B such that the wear model reaches the entry's VB limit in the Taylor life at two speeds
    (see the module doc): in ``entry.contact_life_s`` of edge contact, so that a milling cutting-time
    life is reached in cutting time at the reference engagement. ``response(speed, VB)`` is the
    response at the Taylor reference feed and depth."""
    p = get(CALIBRATION)
    v1, v2, v3 = calibration_speeds(entry)
    f, ap = entry.feed_ref_m, entry.ap_ref_m
    # the wear law integrates edge contact time: the targets are the Taylor lives on the entry's basis
    # times the share of them the edge is in contact in the reference cut (1, or engaged arc / 2 pi)
    t1, t2, t3 = (entry.contact_life_s(v, f, ap) for v in (v1, v2, v3))
    vb_lim = entry.VB_limit_m
    if not vb0_m < vb_lim:
        raise InputError(f"wear: the initial flank wear {vb0_m:.3g} m is not below the Taylor VB limit {vb_lim:.3g} m")
    n = int(nodes_per_interval)
    nodes1 = _nodes(response, v1, vb0_m, vb_lim, geometry, coating, abrasion, n, pieces)
    nodes2 = _nodes(response, v2, vb0_m, vb_lim, geometry, coating, abrasion, n, pieces)
    g = geometry.g_vb
    T_ref = float(np.sum(nodes1[0].weights * nodes1[0].temps))
    if not T_ref > 0.0:
        raise CalibrationError("wear: the response gives no flank temperature at the lower calibration speed")
    scale = 1.0 if abrasion.coefficient > 0.0 else 0.0
    share = p["max_abrasive_share"]
    for nodes, target in ((nodes1, t1), (nodes2, t2)):
        # abrasion alone at scale 1 wears the land in abr_life: its share of the Taylor rate is target / abr_life
        abr_life = life_s(nodes, 0.0, 1.0, T_ref, g, coating, 1.0) if scale > 0.0 else math.inf
        if math.isfinite(abr_life) and target / abr_life > share:
            scale = min(scale, share * abr_life / target)

    b_lo, b_hi = p["min_activation_K"], p["max_activation_K"]

    def residual(B, s):
        A = _solve_A(nodes1, B, T_ref, t1, g, coating, s)
        if not A > 0.0:
            return -math.inf
        return math.log(life_s(nodes2, A, B, T_ref, g, coating, s) / t2)

    # the abrasive floor is reduced (quartered, finally removed) until both Taylor lives can be met
    bracketed = False
    for attempt in range(ABRASIVE_REDUCTIONS + 1):
        r_lo, r_hi = residual(b_lo, scale), residual(b_hi, scale)
        bracketed = math.isfinite(r_lo) and math.isfinite(r_hi) and r_lo * r_hi < 0.0
        if bracketed or scale == 0.0:
            break
        scale = 0.0 if attempt == ABRASIVE_REDUCTIONS - 1 else 0.25 * scale
    if bracketed:
        B = optimize.brentq(lambda b: residual(b, scale), b_lo, b_hi, xtol=1e-9, rtol=1e-15, maxiter=500)
    else:
        grid = np.exp(np.linspace(math.log(b_lo), math.log(b_hi), 61))
        values = []
        for b in grid:
            r = residual(b, scale)
            values.append(abs(r) if math.isfinite(r) else math.inf)
        B = float(grid[int(np.argmin(values))])
    A_scaled = _solve_A(nodes1, B, T_ref, t1, g, coating, scale)
    nodes3 = _nodes(response, v3, vb0_m, vb_lim, geometry, coating, abrasion, n, pieces)
    lives = [life_s(nd, A_scaled, B, T_ref, g, coating, scale) for nd in (nodes1, nodes2, nodes3)]
    targets = [t1, t2, t3]
    A = A_scaled * math.exp(B / T_ref)
    report = {"speeds_m_s": [v1, v2, v3],
              "taylor_life_s": [entry.life_s(v, f, ap) for v in (v1, v2, v3)],
              "life_basis": entry.life_basis, "contact_fraction": entry.contact_fraction,
              "engagement": entry.engagement, "contact_life_s": targets,
              "model_life_s": [m if math.isfinite(m) else None for m in lives],
              "residual_rel": [(m / t - 1.0) if math.isfinite(m) else None for m, t in zip(lives, targets)],
              "bracketed": bool(bracketed), "abrasive_scale": scale, "VB0_m": vb0_m, "VB_limit_m": vb_lim,
              "entry_index": entry.index, "entry_src": entry.src, "entry_coating": entry.coating,
              "flank_temp_K": [float(np.sum(nd[0].weights * nd[0].temps)) for nd in (nodes1, nodes2, nodes3)]}
    return Calibration(float(A), float(B), float(scale), report)


def _branch_rate(A: np.ndarray, B: np.ndarray, lo: np.ndarray, temp_K: float) -> float:
    """A exp(-B / T) of the branch in force at T (kernels.usui_rate's rule: the last branch whose lower end is at
    or below T, the first below every range)."""
    if A.size == 0 or temp_K <= 0.0:
        return 0.0
    b = 0
    for i in range(A.size):
        if temp_K >= lo[i]:
            b = i
    return float(A[b] * math.exp(-B[b] / temp_K))


@dataclass
class Anchor:
    factor: float                   # on every published branch's A
    report: dict = field(default_factory=dict)


def anchor_published(entry: TaylorEntry, response: Callable[[float, float], Response], geometry: FlankGeometry,
                     coating: CoatingData | None, abrasion: Abrasion, vb0_m: float, nodes_per_interval: int,
                     A: np.ndarray, B: np.ndarray, lo: np.ndarray, pieces: int = 1) -> Anchor:
    """Factor s on the A of a published Usui set so that, with this response's flank temperatures, the set reaches
    the Taylor entry's VB limit in its contact life at the entry's check speed (the 15-min speed v15, at the
    entry's feed and depth references). B and the branch ranges stay as published.

    A published set (A, B) holds for the temperatures it was fitted with (often the source's FE model): a
    systematic difference dT of another temperature model changes its rate by about exp(B dT / T^2) near the
    fitted range, a factor A absorbs. The pairing's Taylor life fixes that factor; the published activation
    temperature B keeps the speed and temperature dependence of the mechanism. The residuals at the entry's two
    calibration speeds are reported (not fitted)."""
    p = get(CALIBRATION)
    v1, v2, v3 = calibration_speeds(entry)
    f, ap = entry.feed_ref_m, entry.ap_ref_m
    vb_lim = entry.VB_limit_m
    if not vb0_m < vb_lim:
        raise InputError(f"wear: the initial flank wear {vb0_m:.3g} m is not below the Taylor VB limit {vb_lim:.3g} m")
    n = int(nodes_per_interval)

    def life(nodes, scale: float) -> float:
        total = 0.0
        for node in nodes:
            usui = 0.0
            for temp, weight, ok in zip(node.temps, node.weights, node.coat_ok):
                fc = coating.factor if (coating is not None and ok) else 1.0
                usui += float(weight) * fc * _branch_rate(A, B, lo, float(temp))
            rate = geometry.g_vb * (scale * usui + node.abr) * node.sv
            if not rate > 0.0:
                return math.inf
            total += node.weight_m / rate
        return total

    nodes = {v: _nodes(response, v, vb0_m, vb_lim, geometry, coating, abrasion, n, pieces) for v in (v1, v2, v3)}
    target = entry.contact_life_s(v3, f, ap)
    if life(nodes[v3], 0.0) <= target:
        raise CalibrationError("wear: the abrasive term alone wears the land out before the Taylor life at v15; the "
                               "published Usui set cannot be anchored")
    published = life(nodes[v3], 1.0)
    if not math.isfinite(published):
        raise CalibrationError("wear: the published Usui set gives no wear at the Taylor check speed")
    hi = math.log(published / target) + 1.0
    lo_ = hi - 6.0

    def g(log_s: float) -> float:
        return math.log(life(nodes[v3], math.exp(log_s)) / target)

    while g(lo_) <= 0.0:
        lo_ -= 6.0
    while g(hi) > 0.0:
        hi += 6.0
    scale = math.exp(optimize.brentq(g, lo_, hi, xtol=1e-13, rtol=1e-15, maxiter=500))
    lives = [life(nodes[v], scale) for v in (v1, v2, v3)]
    targets = [entry.contact_life_s(v, f, ap) for v in (v1, v2, v3)]
    report = {"entry_index": entry.index, "entry_src": entry.src, "entry_coating": entry.coating,
              "speeds_m_s": [v1, v2, v3], "contact_life_s": targets, "published_life_v15_s": published,
              "factor": scale, "model_life_s": [m if math.isfinite(m) else None for m in lives],
              "residual_rel": [(m / t - 1.0) if math.isfinite(m) else None for m, t in zip(lives, targets)],
              "flank_temp_K": [float(np.sum(nodes[v][0].weights * nodes[v][0].temps)) for v in (v1, v2, v3)],
              "check_life_s": float(p["check_life_s"])}
    return Anchor(float(scale), report)


__all__ = ["Condition", "Response", "TaylorResponse", "MechanicsResponse", "CalibrationError", "FlankGeometry",
           "Abrasion", "Calibration", "calibrate", "calibration_speeds", "life_s", "Anchor", "anchor_published",
           "land_breaks"]
