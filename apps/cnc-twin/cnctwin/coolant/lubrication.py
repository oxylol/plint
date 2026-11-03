"""Lubrication of the tool-chip interface (docs/models/coolant.md "Lubrication").

The multiplier on the dry sliding friction coefficient the cutting coefficient pipeline uses is

    f = clamp(1 - reach_lub x (1 - f_DB) x Phi, f_floor, 1)

* f_DB - the fluid database factor for the tool class / work family (fluid.Fluid.friction_factor),
  measured with full fluid supply at the database's reference sliding speed and contact length;
* reach_lub - the flow-weighted reach of the coolant at the cut; for aerosols (MQL, mist) also
  limited by supply: the deposited liquid volume flow over the volume a renewed interface needs,
  delta x b x v_chip (boundary film delta, chip width b);
* Phi - capillary access relative to the reference conditions (Williams & Tabor 1977: the fluid
  reaches the interface through capillaries between asperities; Godlevski et al. 1997: the filled
  length falls with speed). A chip element stays on the rake for t = l_c / v_chip; Washburn (1921)
  filling of a capillary of radius r_c gives l_pen = sqrt(sigma r_c cos(theta) t / (2 mu)), so the
  lubricated share of the contact length is phi = min(1, sqrt(sigma r_c cos(theta) / (2 mu v_chip l_c)))
  and Phi = phi / phi_ref. Contact length l_c = k_c h / r (chip ratio r = h / h_c).

Consequences: lubrication falls as v_chip^(-1/2) above the reference speed and saturates at full
penetration below it (f bounded by f_floor); more viscous fluids penetrate less at equal speed (their
database factors already contain their better boundary lubrication at the reference conditions).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

from ..db.models import params as model_params
from . import fluid as F


@functools.lru_cache(maxsize=1)
def P() -> dict:
    return model_params("coolant-lubrication")


@dataclass(frozen=True)
class Lubrication:
    factor: float                   # multiplier on dry friction
    database_factor: float
    source: str                     # how the database factor was found
    penetration: float              # Phi
    supply: float                   # aerosol supply limit (1 for liquids)
    v_chip_m_s: float
    contact_length_m: float


def penetration(props: F.Props, v_chip_m_s: float, contact_length_m: float) -> float:
    """Lubricated share phi of the contact length (Washburn filling during the contact time)."""
    p = P()
    if v_chip_m_s <= 0.0 or contact_length_m <= 0.0 or props.mu_Pa_s <= 0.0:
        return 1.0
    sigma = props.sigma_N_per_m
    return min(1.0, math.sqrt(max(sigma, 0.0) * p["capillary_radius_m"] * p["wetting_cos"]
                              / (2.0 * props.mu_Pa_s * v_chip_m_s * contact_length_m)))


def friction(fluid: F.Fluid, props: F.Props, reach: float, tool_class: str, work_family: str, v_cut_m_s: float,
             h_m: float | None, chip_ratio: float | None, chip_width_m: float, aerosol: bool,
             deposited_liquid_m3_s: float) -> Lubrication:
    """Friction factor of one contact (see the module docstring)."""
    p = P()
    f_db, source = fluid.friction_factor(tool_class, work_family)
    ratio = chip_ratio if chip_ratio is not None and chip_ratio > 0.0 else p["default_chip_ratio"]
    h = h_m if h_m is not None and h_m > 0.0 else p["default_chip_thickness_m"]
    v_chip = max(v_cut_m_s, 0.0) * ratio
    l_c = p["contact_length_ratio"] * h / ratio
    phi = penetration(props, v_chip, l_c)
    phi_ref = penetration(props, p["reference_chip_speed_m_s"], p["reference_contact_length_m"])
    access = phi / phi_ref if phi_ref > 0.0 else 1.0
    supply = 1.0
    if aerosol:
        need = p["boundary_film_m"] * max(chip_width_m, 0.0) * v_chip
        supply = min(1.0, deposited_liquid_m3_s / need) if need > 0.0 else 1.0
    reach_lub = max(0.0, min(reach, 1.0)) * supply
    factor = 1.0 - reach_lub * (1.0 - f_db) * access
    factor = min(1.0, max(factor, p["friction_floor"])) if f_db < 1.0 else min(1.0, max(factor, p["friction_floor"]))
    return Lubrication(factor, f_db, source, access, supply, v_chip, l_c)
