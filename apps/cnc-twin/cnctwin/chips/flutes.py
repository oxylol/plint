"""Chips in the flutes: confinement, fill, evacuation and packing loads per tool slot
(docs/models/chips.md "Flute fill and packing").

A tool whose cut is enclosed (the surface around it lies above its tip: drilling, tapping, reaming,
plunging, deep closed slots) cannot throw its chips out: they enter the flutes and travel up the
engaged length L_e to the hole/slot mouth. Per slot (scalars, L2):

* gullet volume of the engaged length  V_g = A_g L_e  (A_g: total flute cross-section);
* conveying velocity of the flutes as a screw turning in the stationary hole (high-speed limit of
  enclosed screw-conveyor theory, Roberts 1999: the chip's absolute velocity is normal to the flute
  face tilted by the friction angle):  v_s = |omega| R cos(delta) (sin(delta) - mu_f cos(delta)),
  delta the helix angle from the axis, >= 0 (no conveying when tan(delta) <= mu_f);
* depth attenuation and chip form:  v_eff = v_s f_form exp(-k L_e / D) + c_slip Q_tt / A_g
  (through-tool coolant carries chips at a slip ratio of its velocity in the flutes);
* transport with residence time tau = L_e / v_eff (exact over the step):
  dV/dt = Q_in - V / tau  ->  V(t + dt) = V_ss + (V - V_ss) exp(-dt/tau), V_ss = Q_in tau;
  the fill f = V / V_g (steady f = Q_in / (A_g v_eff), independent of depth for a constant v_eff);
* above the critical fill the chips are compacted (Kawakita & Luedde 1971):
  C = 1 - f_c / f,  sigma = P_half C / (a - C),  a = 1 - 1/R,  P_half = c sigma_y,  sigma <= 3 sigma_y;
* packing torque (wall friction on the hole wall) T = mu_w sigma (pi D L_e) R, axial force on the
  flute bottom F = sigma A_g (along +z_T), friction power P = T |omega| + F |v_axial|.

A retract that lifts the tip out of the enclosure (L_e -> 0: peck, tool change) empties the flutes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .protocols import ToolInfo


@dataclass(frozen=True)
class FluteStep:
    volume_m3: float        # bulk volume in the flutes at the end of the step
    out_m3: float           # bulk volume that left the flutes during the step (exact balance)
    fill: float
    v_eff_m_s: float
    stress_Pa: float
    torque_Nm: float
    force_N: float
    power_W: float


def conveying_velocity(tool: ToolInfo, omega_rad_s: float, mu_flute: float) -> float:
    """v_s of the flute as a screw conveyor (m/s, >= 0)."""
    delta = abs(tool.helix_rad)
    if not tool.rotating or omega_rad_s == 0.0 or delta <= 0.0:
        return 0.0
    eta_dir = math.sin(delta) - mu_flute * math.cos(delta)
    if eta_dir <= 0.0:
        return 0.0
    return abs(omega_rad_s) * tool.radius_m * math.cos(delta) * eta_dir


def effective_velocity(tool: ToolInfo, omega_rad_s: float, length_m: float, form_factor: float,
                       through_flow_m3_s: float, flute: dict) -> float:
    """v_eff of chips in the flutes (m/s)."""
    v = conveying_velocity(tool, omega_rad_s, float(flute["flute_friction"]))
    v *= form_factor * math.exp(-float(flute["depth_attenuation_per_diameter"]) * length_m / max(tool.diameter_m, 1e-12))
    if tool.through_coolant and through_flow_m3_s > 0.0 and tool.gullet_area_m2 > 0.0:
        v += float(flute["coolant_slip_ratio"]) * through_flow_m3_s / tool.gullet_area_m2
    return v


def compaction_stress(fill: float, bulk_ratio: float, yield_Pa: float, flute: dict) -> float:
    """Kawakita compaction stress needed to hold the chips of fill f in the gullet (Pa)."""
    f_c = float(flute["critical_fill"])
    if fill <= f_c or fill <= 0.0:
        return 0.0
    c = 1.0 - f_c / fill
    a = 1.0 - 1.0 / max(bulk_ratio, 1.0 + 1e-12)
    cap = float(flute["max_stress_yield_ratio"]) * yield_Pa
    if c >= a:
        return cap
    p_half = float(flute["kawakita_half_stress_ratio"]) * yield_Pa
    return min(p_half * c / (a - c), cap)


def advance(tool: ToolInfo, volume_m3: float, inflow_m3_s: float, length_m: float, omega_rad_s: float,
            v_axial_m_s: float, dt_s: float, bulk_ratio: float, form_factor: float, through_flow_m3_s: float,
            yield_Pa: float, flute: dict) -> FluteStep:
    """One step of the flute store of a tool (bulk volumes; the caller moves mass in proportion)."""
    if length_m <= 0.0 or tool.gullet_area_m2 <= 0.0:
        # out of the enclosure: everything that is in the flutes leaves, nothing new is confined
        return FluteStep(0.0, volume_m3 + inflow_m3_s * dt_s, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    v_eff = effective_velocity(tool, omega_rad_s, length_m, form_factor, through_flow_m3_s, flute)
    v_g = tool.gullet_area_m2 * length_m
    if v_eff > 0.0:
        tau = length_m / v_eff
        v_ss = inflow_m3_s * tau
        new = v_ss + (volume_m3 - v_ss) * math.exp(-dt_s / tau)
    else:
        new = volume_m3 + inflow_m3_s * dt_s
    out = volume_m3 + inflow_m3_s * dt_s - new
    if out < 0.0:
        out = 0.0
        new = volume_m3 + inflow_m3_s * dt_s
    fill = new / v_g
    stress = compaction_stress(fill, bulk_ratio, yield_Pa, flute)
    wall = math.pi * tool.diameter_m * length_m
    torque = float(flute["wall_friction"]) * stress * wall * tool.radius_m
    force = stress * tool.gullet_area_m2
    power = torque * abs(omega_rad_s) + force * abs(v_axial_m_s)
    return FluteStep(new, out, fill, v_eff, stress, torque, force, power)


def steady_fill(tool: ToolInfo, inflow_m3_s: float, length_m: float, omega_rad_s: float, form_factor: float,
                through_flow_m3_s: float, flute: dict) -> float:
    """Hand-calculation form of the steady fill Q_in / (A_g v_eff) (inf when nothing is conveyed)."""
    v_eff = effective_velocity(tool, omega_rad_s, length_m, form_factor, through_flow_m3_s, flute)
    if v_eff <= 0.0:
        return math.inf
    return inflow_m3_s / (tool.gullet_area_m2 * v_eff)
