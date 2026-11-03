"""The coolant tank: thermal mass, heat in and out, chiller, evaporation, concentration and consumption
(docs/models/coolant.md "Tank").

Energy (cp at the start temperature, so the ledger is exact):

    C dT/dt = P_thermal + P_chips + P_pump - Q_amb(T) - Q_evap(T) - Q_chill(T),    C = rho V cp

* Q_amb: free convection of the walls (Churchill & Chu 1975, vertical plate of the liquid depth) and
  the free surface (Lloyd & Moran 1974 hot plate up, 0.27 Ra^(1/4) cold plate up), air at the film
  temperature, plus radiation epsilon sigma (T^4 - T_a^4) of both;
* Q_evap = m_evap h_fg(T): the fluid file's evaporation rate scaled by Dalton's law with the
  vapour-pressure deficit (Magnus formula, Alduchov & Eskridge 1996) relative to its reference
  conditions, water-mixed fluids only (oils are not volatile at tank temperatures);
* Q_chill = capacity x clamp((T - T_set)/band, 0, 1) when the machine has a tank chiller.

Each step linearises the losses at the step's start and integrates the linear equation exactly
(T = T0 + (P - Q0)/G (1 - exp(-G dt/C))); the integral of every loss term follows from the same
solution, so C (T_end - T_start) = (heat in - losses) dt to round-off. Mass leaves with drag-out on
chips, mist and evaporation; drag-out and mist take mix at the current concentration, evaporation
takes water only (the concentration drifts up). No make-up during a run: the volumes to restore
level and concentration are reported.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

from ..db.models import params as model_params
from ..units import STEFAN_BOLTZMANN_W_M2K4, STANDARD_GRAVITY_M_S2, ZERO_DEGC_K
from . import fluid as F


@functools.lru_cache(maxsize=1)
def P() -> dict:
    return model_params("coolant-tank")


def saturation_pressure_Pa(temp_K: float) -> float:
    """Saturation vapour pressure of water (Magnus form, Alduchov & Eskridge 1996)."""
    p = P()
    t = temp_K - ZERO_DEGC_K
    return p["magnus_c1_Pa"] * math.exp(p["magnus_c2"] * t / (t + p["magnus_c3_dK"]))


def latent_heat_water(temp_K: float) -> float:
    """h_fg of water, linear between the NIST values at 25 and 100 degC."""
    fp = F.P()
    t25 = ZERO_DEGC_K + 25.0
    t100 = ZERO_DEGC_K + 100.0
    return fp["water_latent_heat_25C_J_kg"] + (temp_K - t25) / (t100 - t25) * (
        fp["water_latent_heat_J_kg"] - fp["water_latent_heat_25C_J_kg"])


def _free_convection_vertical(dT: float, height: float, T_film: float) -> float:
    p = P()
    air = F.air_props(T_film)
    beta = 1.0 / T_film
    ra = STANDARD_GRAVITY_M_S2 * beta * abs(dT) * height ** 3 / (air.nu_m2_s * air.nu_m2_s / air.pr)
    ra = max(ra, p["min_rayleigh"])
    nu = (p["cc_vertical_a"] + p["cc_vertical_b"] * ra ** (1.0 / 6.0)
          / (1.0 + (p["cc_vertical_pr"] / air.pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)) ** 2
    return nu * air.k_W_mK / height


def _free_convection_surface(dT: float, length: float, T_film: float) -> float:
    p = P()
    air = F.air_props(T_film)
    beta = 1.0 / T_film
    ra = STANDARD_GRAVITY_M_S2 * beta * abs(dT) * length ** 3 / (air.nu_m2_s * air.nu_m2_s / air.pr)
    ra = max(ra, p["min_rayleigh"])
    if dT >= 0.0:
        if ra <= p["plate_up_ra_switch"]:
            nu = p["plate_up_low_c"] * ra ** p["plate_up_low_n"]
        else:
            nu = p["plate_up_high_c"] * ra ** p["plate_up_high_n"]
    else:
        nu = p["plate_down_c"] * ra ** p["plate_down_n"]
    return nu * air.k_W_mK / length


@dataclass
class TankStep:
    """Energies (J) and masses of one step."""

    heat_in_J: float = 0.0
    ambient_J: float = 0.0
    evaporation_J: float = 0.0
    chiller_J: float = 0.0
    mass_out_enthalpy_J: float = 0.0
    evaporated_m3: float = 0.0
    residual_J: float = 0.0


@dataclass
class Tank:
    """State of the tank (checkpointed by CoolantState)."""

    fluid_kind: str
    capacity_m3: float
    temp_K: float
    volume_m3: float
    concentrate_m3: float
    rho_ref_kg_m3: float
    cp_ref_J_kgK: float
    ref_temp_K: float
    ambient_K: float
    evaporation_m3_s_per_m2: float
    chiller_W: float = 0.0
    chiller_set_K: float = math.inf
    energy_J: float = 0.0           # C (T - T_ref)
    totals: dict = field(default_factory=dict)

    @classmethod
    def create(cls, fluid: F.Fluid, capacity_m3: float, fill_fraction: float, temp_K: float, ambient_K: float,
               concentration: float, chiller_W: float = 0.0, chiller_set_K: float = math.inf) -> "Tank":
        volume = capacity_m3 * fill_fraction
        liquid = not (fluid.dry or fluid.is_gas or fluid.is_mql)
        props = fluid.props(temp_K) if liquid else F.air_props(temp_K)
        tank = cls(fluid.kind, capacity_m3, temp_K, volume if liquid else 0.0,
                   volume * concentration if liquid else 0.0, props.rho_kg_m3, props.cp_J_kgK, temp_K, ambient_K,
                   fluid.evaporation_m3_s_per_m2 if fluid.water_based else 0.0, chiller_W, chiller_set_K)
        tank.totals = {k: 0.0 for k in ("heat_in_J", "ambient_J", "evaporation_J", "chiller_J", "mass_out_J",
                                        "drag_out_m3", "mist_m3", "evaporated_m3", "concentrate_out_m3")}
        return tank

    @property
    def liquid(self) -> bool:
        return self.volume_m3 > 0.0

    @property
    def capacity_J_K(self) -> float:
        return self.rho_ref_kg_m3 * self.volume_m3 * self.cp_ref_J_kgK

    @property
    def concentration(self) -> float:
        return self.concentrate_m3 / self.volume_m3 if self.volume_m3 > 0.0 else 0.0

    def geometry(self) -> tuple[float, float, float]:
        """(free surface, wetted wall area, liquid depth) of a square tank of the database volume."""
        p = P()
        surface = self.capacity_m3 / p["tank_depth_m"]
        depth = self.volume_m3 / surface if surface > 0.0 else 0.0
        walls = 4.0 * math.sqrt(surface) * depth
        return surface, walls, depth

    def _losses(self, T: float) -> tuple[list[float], list[float]]:
        """(values W, slopes W/K) of [ambient, evaporation, chiller] at T."""
        p = P()
        surface, walls, depth = self.geometry()
        Ta = self.ambient_K
        dT = T - Ta
        film = 0.5 * (T + Ta)
        h_w = _free_convection_vertical(dT, max(depth, 1e-3), film) if walls > 0.0 else 0.0
        length = math.sqrt(surface) / 4.0 if surface > 0.0 else 1.0
        h_s = _free_convection_surface(dT, length, film)
        eps = p["wall_emissivity"]
        h_r = eps * STEFAN_BOLTZMANN_W_M2K4 * (T * T + Ta * Ta) * (T + Ta)
        area = surface + walls
        amb = (h_w * walls + h_s * surface + h_r * area) * dT
        amb_slope = h_w * walls + h_s * surface + h_r * area + 4.0 * eps * STEFAN_BOLTZMANN_W_M2K4 * T ** 3 * area \
            - h_r * area
        evap = 0.0
        evap_slope = 0.0
        if self.evaporation_m3_s_per_m2 > 0.0:
            rate, slope = self._evaporation(T, surface)
            evap = rate * self.rho_ref_kg_m3 * latent_heat_water(T)
            evap_slope = slope * self.rho_ref_kg_m3 * latent_heat_water(T)
        chill = 0.0
        chill_slope = 0.0
        if self.chiller_W > 0.0 and math.isfinite(self.chiller_set_K):
            band = p["chiller_band_dK"]
            x = (T - self.chiller_set_K) / band
            if x > 0.0:
                chill = self.chiller_W * min(x, 1.0)
                chill_slope = self.chiller_W / band if x < 1.0 else 0.0
        return [amb, evap, chill], [amb_slope, evap_slope, chill_slope]

    def _evaporation(self, T: float, surface: float) -> tuple[float, float]:
        """(volume rate m3/s of water, d rate/dT) by Dalton scaling of the database rate."""
        p = P()
        rh = p["relative_humidity"]
        ref = saturation_pressure_Pa(p["evaporation_ref_temp_K"]) - rh * saturation_pressure_Pa(
            p["evaporation_ref_ambient_K"])
        deficit = saturation_pressure_Pa(T) - rh * saturation_pressure_Pa(self.ambient_K)
        if deficit <= 0.0 or ref <= 0.0:
            return 0.0, 0.0
        t = T - ZERO_DEGC_K
        c2, c3 = p["magnus_c2"], p["magnus_c3_dK"]
        d_es = saturation_pressure_Pa(T) * c2 * c3 / (t + c3) ** 2
        base = self.evaporation_m3_s_per_m2 * surface / ref
        return base * deficit, base * d_es

    def step(self, dt_s: float, heat_in_W: float, drag_out_m3: float = 0.0, mist_m3: float = 0.0) -> TankStep:
        """Advance by dt with a constant heat input; remove drag-out and mist volumes at the end."""
        out = TankStep()
        if not self.liquid or dt_s <= 0.0:
            return out
        C = self.capacity_J_K
        T0 = self.temp_K
        values, slopes = self._losses(T0)
        Q0 = sum(values)
        G = sum(slopes)
        if G > 0.0:
            tau = C / G
            x = (heat_in_W - Q0) / G
            decay = math.exp(-dt_s / tau)
            T1 = T0 + x * (1.0 - decay)
            integral = x * (dt_s - tau * (1.0 - decay))        # int (T - T0) dt
        else:
            T1 = T0 + (heat_in_W - Q0) * dt_s / C
            integral = 0.5 * (T1 - T0) * dt_s
        energies = [v * dt_s + s * integral for v, s in zip(values, slopes)]
        out.heat_in_J = heat_in_W * dt_s
        out.ambient_J, out.evaporation_J, out.chiller_J = energies
        E0 = self.energy_J
        self.temp_K = T1
        self.energy_J = C * (T1 - self.ref_temp_K)
        evaporated = 0.0
        if self.evaporation_m3_s_per_m2 > 0.0:
            rate, slope = self._evaporation(T0, self.geometry()[0])
            evaporated = rate * dt_s + slope * integral
        out.evaporated_m3 = max(evaporated, 0.0)
        out.residual_J = (self.energy_J - E0) - (out.heat_in_J - sum(energies))
        self._remove(drag_out_m3 + mist_m3, out.evaporated_m3, out)
        t = self.totals
        t["heat_in_J"] += out.heat_in_J
        t["ambient_J"] += out.ambient_J
        t["evaporation_J"] += out.evaporation_J
        t["chiller_J"] += out.chiller_J
        t["mass_out_J"] += out.mass_out_enthalpy_J
        t["drag_out_m3"] += drag_out_m3
        t["mist_m3"] += mist_m3
        t["evaporated_m3"] += out.evaporated_m3
        return out

    def _remove(self, mix_m3: float, water_m3: float, out: TankStep) -> None:
        """Mix leaves at the current concentration, water alone by evaporation; the energy that leaves
        with the mass at the tank temperature is booked so the ledger stays closed."""
        if self.volume_m3 <= 0.0:
            return
        mix = min(max(mix_m3, 0.0), self.volume_m3)
        conc = self.concentration
        concentrate_out = mix * conc
        water = min(max(water_m3, 0.0), self.volume_m3 - mix - (self.concentrate_m3 - concentrate_out))
        removed = mix + max(water, 0.0)
        C_old = self.capacity_J_K
        self.volume_m3 -= removed
        self.concentrate_m3 -= concentrate_out
        self.totals["concentrate_out_m3"] += concentrate_out
        C_new = self.capacity_J_K
        out.mass_out_enthalpy_J = (C_old - C_new) * (self.temp_K - self.ref_temp_K)
        self.energy_J = C_new * (self.temp_K - self.ref_temp_K)

    def to_dict(self) -> dict:
        return {k: (v if not isinstance(v, dict) else dict(v)) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: dict) -> "Tank":
        tank = cls(**{k: v for k, v in data.items() if k != "totals"})
        tank.totals = dict(data.get("totals", {}))
        return tank
