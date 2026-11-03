"""Heat transfer coefficients of the coolant at the cut (docs/models/coolant.md "Heat transfer").

* **Impinging round jets** (Martin 1977, single round nozzle): Nu_avg = Pr^0.42 G(r/D, H/D) F1(Re),
  G = (D/r)(1 - 1.1 D/r)/(1 + 0.1 (H/D - 6) D/r), F1 = 2 Re^0.5 (1 + 0.005 Re^0.55)^0.5, averaged over
  the disc of radius r around the stagnation point, properties at the film temperature. A coherent
  liquid jet arrives undecayed (H/D capped at the potential-core length B); a gas jet farther than
  H/D = 12 is replaced by the momentum-equivalent nozzle D' = H/12 (Re = u D/nu is conserved by the
  self-similar jet). r/D below 2.5 uses 2.5 (stagnation zone); above 7.5 G is extrapolated (~ D/r).
* **Boiling** of liquids on walls hotter than the saturation temperature (raised by the impingement
  pressure, Clausius-Clapeyron): nucleate boiling (Rohsenow 1952) superposed on the jet's forced
  convection (Kutateladze 1961), peak heat flux of Zuber/Lienhard-Dhir with the subcooling factor of
  Ivey & Morris (1962) plus the forced-convection share at the subcooling; film boiling above the
  database Leidenfrost temperature: Bromley (1950) on tools (cylinders), Berenson (1961) on the work
  (plates), forced film boiling of Bromley, LeRoy & Robbers (1953) when larger, radiation across the
  vapour film by Bromley's h^(4/3) = h_c^(4/3) + h_r h^(1/3); transition boiling interpolated in log q
  - log dT between the peak and the Leidenfrost point. The coefficient returned is q / (T_w - T_fluid)
  (relative to the bulk fluid temperature, as thermal applies it).
* **Aerosols** (MQL, mist): the carrier-air jet plus the sensible heat and, above the boiling point
  and below the Leidenfrost point, the evaporation of the deposited liquid flux.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

from ..db.models import params as model_params
from ..units import STANDARD_ATMOSPHERE_PA, STANDARD_GRAVITY_M_S2, STEFAN_BOLTZMANN_W_M2K4
from . import fluid as F
from .jets import P as jet_params

SINGLE_PHASE = 0
NUCLEATE = 1
TRANSITION = 2
FILM = 3
REGIME_NAMES = ("single_phase", "nucleate", "transition", "film")

PLATE = 0
CYLINDER = 1


@functools.lru_cache(maxsize=1)
def P() -> dict:
    return model_params("coolant-heat")


# ============================================================================ Martin 1977

def martin_G(r_over_d: float, h_over_d: float) -> float:
    p = P()
    rd = max(r_over_d, p["martin_rd_lo"])
    hd = min(max(h_over_d, p["martin_hd_lo"]), p["martin_hd_hi"])
    x = 1.0 / rd
    return x * (1.0 - p["martin_g_a"] * x) / (1.0 + p["martin_g_b"] * (hd - p["martin_g_h0"]) * x)


def martin_F(re: float) -> float:
    p = P()
    re = max(re, 0.0)
    return p["martin_f_a"] * re ** p["martin_f_n1"] * (1.0 + p["martin_f_b"] * re ** p["martin_f_n2"]) ** p["martin_f_n1"]


def martin_nusselt(re: float, pr: float, r_over_d: float, h_over_d: float) -> float:
    """Area-averaged Nusselt number Nu = h D/k of a single round impinging jet (Martin 1977)."""
    return pr ** P()["martin_pr_exp"] * martin_G(r_over_d, h_over_d) * martin_F(re)


def jet_h(gas: bool, speed: float, diameter: float, distance: float, r_avg: float, props: F.Props) -> float:
    """Martin coefficient (W/m2K) of a jet of exit (or arrival) speed and diameter reaching a surface at
    arc length ``distance``, averaged over the disc of radius ``r_avg``."""
    if speed <= 0.0 or diameter <= 0.0:
        return 0.0
    p = P()
    re = speed * diameter / props.nu_m2_s
    hd = distance / diameter
    if gas and hd > p["martin_hd_hi"]:
        d_eq = distance / p["martin_hd_hi"]
        nu = martin_nusselt(re, props.pr, r_avg / d_eq, p["martin_hd_hi"])
        return nu * props.k_W_mK / d_eq
    if not gas:
        hd = min(hd, jet_params()["gas_jet_centreline_B"])
    nu = martin_nusselt(re, props.pr, r_avg / diameter, hd)
    return nu * props.k_W_mK / diameter


# ============================================================================ boiling curve

@dataclass(frozen=True)
class Boiling:
    h_W_m2K: float                  # q / (T_w - T_fluid)
    q_W_m2: float
    regime: int
    T_sat_K: float
    T_leid_K: float
    q_peak_W_m2: float


def _sat_props(fluid: F.Fluid, T_sat: float, pressure_Pa: float):
    liq = fluid.props(min(T_sat, _table_top(fluid)))
    vap = fluid.vapour(T_sat, pressure_Pa)
    return liq, vap


def _table_top(fluid: F.Fluid) -> float:
    """Highest temperature at which the liquid tables are evaluated (they hold their end values)."""
    return fluid.boiling_K if math.isfinite(fluid.boiling_K) else 400.0


def nucleate_flux(fluid: F.Fluid, dT_sat: float, liq: F.Props, vap: F.Props) -> float:
    """Rohsenow (1952) nucleate pool-boiling heat flux at a wall superheat."""
    if dT_sat <= 0.0:
        return 0.0
    p = P()
    n = p["rohsenow_n_water"] if fluid.water_based else p["rohsenow_n_other"]
    hfg = fluid.latent_J_kg
    lap = math.sqrt(STANDARD_GRAVITY_M_S2 * max(liq.rho_kg_m3 - vap.rho_kg_m3, 0.0) / max(fluid.sigma_N_per_m, 1e-6))
    return liq.mu_Pa_s * hfg * lap * (liq.cp_J_kgK * dT_sat / (p["rohsenow_csf"] * hfg * liq.pr ** n)) ** 3


def peak_flux(fluid: F.Fluid, liq: F.Props, vap: F.Props, dT_sub: float, h_fc: float) -> float:
    """Peak (critical) heat flux: Zuber/Lienhard-Dhir with Ivey & Morris subcooling, plus the jet's forced
    convection at the subcooling."""
    p = P()
    hfg = fluid.latent_J_kg
    sigma = max(fluid.sigma_N_per_m, 1e-6)
    q_sat = p["peak_flux_c"] * hfg * vap.rho_kg_m3 * (sigma * STANDARD_GRAVITY_M_S2 *
                                                     max(liq.rho_kg_m3 - vap.rho_kg_m3, 0.0) / vap.rho_kg_m3 ** 2) ** 0.25
    factor = 1.0 + p["subcooling_a"] * (liq.rho_kg_m3 / vap.rho_kg_m3) ** p["subcooling_n"] * \
        liq.cp_J_kgK * max(dT_sub, 0.0) / hfg
    return q_sat * factor + h_fc * max(dT_sub, 0.0)


def film_h(fluid: F.Fluid, T_w: float, T_sat: float, liq: F.Props, target: int, size_m: float, speed: float,
           pressure_Pa: float, tol: float, max_iter: int) -> float:
    """Film-boiling coefficient on (T_w - T_sat), convection plus radiation across the vapour film."""
    p = P()
    dT = max(T_w - T_sat, 1e-6)
    vap = fluid.vapour(0.5 * (T_w + T_sat), pressure_Pa)
    hfg = fluid.latent_J_kg + p["film_hfg_factor"] * vap.cp_J_kgK * dT
    drho = max(liq.rho_kg_m3 - vap.rho_kg_m3, 0.0)
    g = STANDARD_GRAVITY_M_S2
    size = max(size_m, 1e-6)
    if target == CYLINDER:
        h_pool = p["film_bromley_c"] * vap.k_W_mK / size * (g * drho * hfg * size ** 3 /
                                                            (vap.nu_m2_s * vap.k_W_mK * dT)) ** 0.25
    else:
        L = math.sqrt(max(fluid.sigma_N_per_m, 1e-6) / (g * max(drho, 1e-9)))
        h_pool = p["film_berenson_c"] * (vap.k_W_mK ** 3 * vap.rho_kg_m3 * g * drho * hfg /
                                         (vap.mu_Pa_s * dT * L)) ** 0.25
    h_forced = p["film_forced_c"] * math.sqrt(max(speed, 0.0) * vap.k_W_mK * vap.rho_kg_m3 * hfg / (size * dT))
    h_conv = max(h_pool, h_forced)
    eps = p["film_emissivity"]
    h_rad = eps * STEFAN_BOLTZMANN_W_M2K4 * (T_w ** 4 - T_sat ** 4) / dT
    h = h_conv + 0.75 * h_rad
    for _ in range(max_iter):
        new = (h_conv ** (4.0 / 3.0) + h_rad * h ** (1.0 / 3.0)) ** 0.75
        if abs(new - h) <= tol * new:
            h = new
            break
        h = new
    return h


def _peak_superheat(total, a_nb, h_fc, dT_sub, n, q_peak, dT_hi, tol, max_iter) -> float:
    """Superheat at which the superposed nucleate curve reaches the peak flux: Newton on
    g = q_total - q_peak from (q_peak / a)^(1/3), kept inside the bracket [0, dT_hi] (bisection when a
    Newton step leaves it); deterministic, stops at |step| <= tol x dT."""
    lo, hi = 0.0, dT_hi
    x = min(max((q_peak / a_nb) ** (1.0 / 3.0) if a_nb > 0.0 else 0.5 * dT_hi, 0.0), dT_hi)
    for _ in range(max_iter):
        q_fc = h_fc * (dT_sub + x)
        q_nb = a_nb * x ** 3
        q = total(x)
        g = q - q_peak
        if g < 0.0:
            lo = x
        else:
            hi = x
        dq = (q_fc ** (n - 1.0) * h_fc + q_nb ** (n - 1.0) * 3.0 * a_nb * x * x) * q ** (1.0 - n) if q > 0.0 else 0.0
        step = g / dq if dq > 0.0 else 0.0
        new = x - step
        if not (lo < new < hi) or dq <= 0.0:
            new = 0.5 * (lo + hi)
        if abs(new - x) <= tol * max(new, 1e-12):
            return new
        x = new
    return x


def boiling(fluid: F.Fluid, T_w: float, T_f: float, h_fc: float, target: int, size_m: float, speed: float,
            impact_Pa: float, tol: float, max_iter: int) -> Boiling:
    """Effective coefficient of a liquid on a wall at T_w (fluid bulk T_f), with the forced-convection
    coefficient h_fc of the jet, see the module docstring. ``impact_Pa``: mean gauge pressure of the
    impingement raising the saturation temperature."""
    if fluid.dry or fluid.is_gas or not math.isfinite(fluid.boiling_K):
        return Boiling(h_fc, h_fc * (T_w - T_f), SINGLE_PHASE, math.inf, math.inf, math.inf)
    pressure = STANDARD_ATMOSPHERE_PA + max(impact_Pa, 0.0)
    T_sat = fluid.saturation_K(pressure)
    T_L = max(fluid.leidenfrost_at_K(pressure), T_sat + 1.0)
    if T_w <= T_sat:
        return Boiling(h_fc, h_fc * (T_w - T_f), SINGLE_PHASE, T_sat, T_L, math.inf)
    liq, vap = _sat_props(fluid, T_sat, pressure)
    n = P()["superposition_n"]
    dT_sub = max(T_sat - T_f, 0.0)

    a_nb = nucleate_flux(fluid, 1.0, liq, vap)              # Rohsenow: q_nb = a dT^3

    def nucleate_total(dT):
        q_fc = h_fc * (T_sat + dT - T_f)
        q_nb = a_nb * dT ** 3
        return (q_fc ** n + q_nb ** n) ** (1.0 / n)

    q_peak = peak_flux(fluid, liq, vap, dT_sub, h_fc)
    dT_L = T_L - T_sat
    if nucleate_total(dT_L) <= q_peak:
        dT_peak = dT_L
    else:
        dT_peak = _peak_superheat(nucleate_total, a_nb, h_fc, T_sat - T_f, n, q_peak, dT_L, tol, max_iter)
    dT = T_w - T_sat
    if T_w >= T_L:
        q = film_h(fluid, T_w, T_sat, liq, target, size_m, speed, pressure, tol, max_iter) * dT
        regime = FILM
    elif dT <= dT_peak:
        q = nucleate_total(dT)
        regime = NUCLEATE
    else:
        q1 = nucleate_total(dT_peak)
        q2 = film_h(fluid, T_L, T_sat, liq, target, size_m, speed, pressure, tol, max_iter) * dT_L
        x = (math.log(dT) - math.log(dT_peak)) / (math.log(dT_L) - math.log(dT_peak))
        q = math.exp(math.log(q1) + x * (math.log(q2) - math.log(q1)))
        regime = TRANSITION
    h = q / (T_w - T_f) if T_w > T_f else h_fc
    return Boiling(h, q, regime, T_sat, T_L, q_peak)


# ============================================================================ aerosols

def aerosol_h(h_air: float, liquid_flux_kg_m2s: float, T_w: float, T_f: float, fluid: F.Fluid) -> float:
    """Carrier-air coefficient plus the deposited droplets' sensible heat and evaporation."""
    if T_w <= T_f or liquid_flux_kg_m2s <= 0.0 or fluid.dry or fluid.is_gas:
        return h_air
    liq = fluid.props(min(T_f, _table_top(fluid)))
    T_b = fluid.boiling_K
    sensible = liq.cp_J_kgK * (min(T_w, T_b) - T_f)
    latent = fluid.latent_J_kg if (T_b <= T_w < fluid.leidenfrost_K) else 0.0
    return h_air + liquid_flux_kg_m2s * (sensible + latent) / (T_w - T_f)
