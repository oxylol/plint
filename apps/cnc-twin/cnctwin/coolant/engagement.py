"""The coolant jet at the cutting engagement (docs/models/coolant.md 5.4): convective cooling of the tool and
work faces next to the contact and the shorter tool-chip contact under a jet that reaches under the chip.

* **Nozzle jet**: speed from the supply pressure, ``v = C_v sqrt(2 p / rho)`` (converging nozzle, no vena
  contracta: C_v = the supply model's ``nozzle_discharge_coefficient``); a stated flow sets the jet diameter by
  continuity, ``d = sqrt(4 Q / (pi v))``, otherwise the flow is ``Q = v pi d0^2 / 4`` of the nozzle bore d0.
* **Convective cooling of a face** at the engagement (rake face beyond the contact, flank face, work surface):
  Martin's (1977) round-jet coefficient (``heat.jet_h``, liquid jets arriving undecayed) with the component of
  the arriving speed normal to the face (an inclined jet transfers less heat than a normal one, its stagnation
  zone set by the normal momentum; Goldstein & Franchett 1988), averaged over a disc of radius r_avg around
  the impingement, times the arriving share of the jet (reach). These are the "where the coolant arrives"
  coefficients thermal applies over the reached part of a face (``ToolEnvironment.h_coolant_W_m2K``).
* **Hydraulic wedge** (Mazurkiewicz, Kubala & Chow 1989; Kaminski & Alvelid 2000): the part of the jet's
  momentum flux that runs along the rake face against the chip flow enters the gap between the curling chip
  and the rake face and lifts the chip at the end of contact, ``F_w = rho Q_arr v cos(a) max(0, -e . u_chip)``
  (a the jet's elevation above the rake face, e its unit direction in the rake plane, u_chip the chip flow
  direction). The chip's normal-force balance (rake normal force N' + F_w = N, the shear zone unchanged) and
  moment balance about the cutting edge (the rake normal stress falls linearly to zero at the end of contact,
  so N' acts at l_c' / 3; Zorev 1963, Oxley 1989) give ``l_c' = l_c / (1 + 2 F_w / N)``, not below the
  sticking zone (``wedge_min_contact_ratio``). A jet onto the chip's back (the normal component) does not
  reach under the chip and is not modelled as a force here.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np

from ..db.models import params as model_params
from . import fluid as F
from . import heat as H


@functools.lru_cache(maxsize=1)
def P() -> dict:
    return model_params("coolant-heat")


@functools.lru_cache(maxsize=1)
def _supply() -> dict:
    return model_params("coolant-supply")


@dataclass(frozen=True)
class NozzleJet:
    speed_m_s: float
    diameter_m: float
    flow_m3_s: float
    rho_kg_m3: float

    @property
    def momentum_N(self) -> float:
        """Momentum flux of the whole jet (N)."""
        return self.rho_kg_m3 * self.flow_m3_s * self.speed_m_s


def nozzle_jet(pressure_Pa: float, bore_m: float, props: F.Props, flow_m3_s: float | None = None) -> NozzleJet:
    """Jet of a nozzle at a supply pressure (gauge): Bernoulli speed with the nozzle coefficient; a stated flow
    gives the jet diameter by continuity, else the bore does."""
    if pressure_Pa <= 0.0 or props.rho_kg_m3 <= 0.0:
        raise ValueError("coolant: a nozzle jet needs a positive pressure and a liquid density")
    speed = float(_supply()["nozzle_discharge_coefficient"]) * math.sqrt(2.0 * pressure_Pa / props.rho_kg_m3)
    if flow_m3_s is not None and flow_m3_s > 0.0:
        diameter = math.sqrt(4.0 * flow_m3_s / (math.pi * speed))
        return NozzleJet(speed, diameter, float(flow_m3_s), props.rho_kg_m3)
    return NozzleJet(speed, bore_m, speed * math.pi * bore_m * bore_m / 4.0, props.rho_kg_m3)


def face_h(jet: NozzleJet, distance_m: float, incidence_rad: float, r_avg_m: float, props: F.Props,
           reach: float = 1.0) -> float:
    """Convective coefficient (W/m2K) the jet gives a face it meets at ``incidence_rad`` (angle between the jet
    and the face; pi/2 = normal), averaged over a disc of radius r_avg around the impingement: Martin's liquid
    jet with the normal speed component, times the arriving share."""
    s = math.sin(min(max(incidence_rad, 0.0), 0.5 * math.pi))
    if s <= 0.0 or reach <= 0.0:
        return 0.0
    return float(reach) * H.jet_h(False, jet.speed_m_s * s, jet.diameter_m, distance_m, r_avg_m, props)


def wedge_force_N(jet: NozzleJet, direction, rake_normal, chip_flow, reach: float = 1.0) -> float:
    """Momentum flux of the arriving jet that runs along the rake face against the chip flow (N): the part that
    enters the gap between the chip and the rake face."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    n = np.asarray(rake_normal, dtype=float)
    n = n / np.linalg.norm(n)
    u = np.asarray(chip_flow, dtype=float)
    u = u - np.dot(u, n) * n
    u = u / np.linalg.norm(u)
    in_plane = d - np.dot(d, n) * n                      # |in_plane| = cos(elevation)
    against = max(0.0, -float(np.dot(in_plane, u)))      # cos(elevation) x alignment against the chip flow
    return float(reach) * jet.momentum_N * against


def contact_length_ratio(wedge_N: float, rake_normal_N: float) -> float:
    """l_c' / l_c of the tool-chip contact under a lifting force at the end of contact (see the module notes)."""
    if wedge_N <= 0.0 or rake_normal_N <= 0.0:
        return 1.0
    p = P()
    return max(float(p["wedge_min_contact_ratio"]), 1.0 / (1.0 + p["wedge_moment_factor"] * wedge_N / rake_normal_N))


__all__ = ["NozzleJet", "nozzle_jet", "face_h", "wedge_force_N", "contact_length_ratio"]
