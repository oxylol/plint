"""Recorded channels of the cutting domain (core.channels; docs/models/cutting.md "Channels").

Tick tier (L1): per path the tick means of the tool force in T0 and of the process torque about the
spindle axis (from Accum.tick_wrench / tick_spindle_torque_Nm). Macro tier (L2): per path the step
means of spindle power, feed power, spindle load fraction, heat generation by zone, chip volume rate,
the largest chip thickness, the in-cut fraction and the peak force. Vectors are per controller path
(max_shape (MAX_PATHS,), NaN padded).
"""

from __future__ import annotations

import numpy as np

from ..core.channels import Channel, Tier

DOMAIN = "cutting"
MAX_PATHS = 4

_TICK = (("force_x_N", "N", "tool force along x_T0, tick mean"), ("force_y_N", "N", "tool force along y_T0, tick mean"),
         ("force_z_N", "N", "tool force along z_T0 (+: toward the spindle), tick mean"),
         ("torque_Nm", "N m", "process torque about the spindle axis on the rotating body, tick mean"))
_MACRO = (("spindle_power_W", "W", "spindle shaft power of the cut, step mean"),
          ("feed_power_W", "W", "feed-drive power of the cut, step mean"),
          ("spindle_load_fraction", "-", "cutting torque / available torque at speed and gear"),
          ("heat_shear_W", "W", "primary-zone heat generation, step mean"),
          ("heat_rake_W", "W", "rake-face (secondary zone) heat generation, step mean"),
          ("heat_flank_W", "W", "flank wear-land heat generation, step mean"),
          ("ploughing_W", "W", "edge, ploughing, forming and process-damping dissipation, step mean"),
          ("chip_volume_rate_m3_s", "m3/s", "analytic chip volume per second"),
          ("h_max_m", "m", "largest uncut chip thickness of the step"),
          ("in_cut_fraction", "-", "share of the step with at least one element in cut"),
          ("force_peak_N", "N", "largest tool force of the step"),
          ("spindle_thermal_load", "-", "spindle motor thermal load (1 = continuous S1 limit)"))


def channels() -> list[Channel]:
    out = [Channel(f"{DOMAIN}.{name}", unit, Tier.TICK, DOMAIN, doc, (MAX_PATHS,)) for name, unit, doc in _TICK]
    out += [Channel(f"{DOMAIN}.{name}", unit, Tier.MACRO, DOMAIN, doc, (MAX_PATHS,)) for name, unit, doc in _MACRO]
    return out


def tick_values(contacts: list, n_ticks: int) -> dict:
    """Tick-tier columns {name: [n_ticks, MAX_PATHS]} from the step's CuttingContact records (path order)."""
    out = {f"{DOMAIN}.{name}": np.full((n_ticks, MAX_PATHS), np.nan) for name, _, _ in _TICK}
    for p, c in enumerate(contacts[:MAX_PATHS]):
        if c is None:
            continue
        w = np.asarray(c.wrench_tool_T0_mean, float)
        k = min(n_ticks, w.shape[0])
        out[f"{DOMAIN}.force_x_N"][:k, p] = w[:k, 0]
        out[f"{DOMAIN}.force_y_N"][:k, p] = w[:k, 1]
        out[f"{DOMAIN}.force_z_N"][:k, p] = w[:k, 2]
        tq = np.asarray(c.spindle_torque_Nm, float)
        out[f"{DOMAIN}.torque_Nm"][:min(k, tq.shape[0]), p] = tq[:min(k, tq.shape[0])]
    return out


def macro_values(contacts: list, dt_s: float, thermal_load: list) -> dict:
    """Macro-tier values {name: [MAX_PATHS]} from the step's CuttingContact records (path order)."""
    out = {f"{DOMAIN}.{name}": np.full(MAX_PATHS, np.nan) for name, _, _ in _MACRO}
    for p, c in enumerate(contacts[:MAX_PATHS]):
        if c is None:
            continue
        inv = 1.0 / dt_s if dt_s > 0.0 else 0.0
        out[f"{DOMAIN}.spindle_power_W"][p] = c.spindle_power_mean_W
        out[f"{DOMAIN}.feed_power_W"][p] = c.feed_power_mean_W
        out[f"{DOMAIN}.spindle_load_fraction"][p] = c.spindle_load_fraction
        out[f"{DOMAIN}.heat_shear_W"][p] = c.heat_shear_J * inv
        out[f"{DOMAIN}.heat_rake_W"][p] = c.heat_rake_J * inv
        out[f"{DOMAIN}.heat_flank_W"][p] = c.heat_flank_J * inv
        out[f"{DOMAIN}.ploughing_W"][p] = c.ploughing_J * inv
        out[f"{DOMAIN}.chip_volume_rate_m3_s"][p] = c.analytic_chip_volume_m3 * inv
        out[f"{DOMAIN}.h_max_m"][p] = getattr(c, "h_max_m", 0.0)
        out[f"{DOMAIN}.in_cut_fraction"][p] = c.in_cut_s * inv
        out[f"{DOMAIN}.force_peak_N"][p] = float(np.linalg.norm(c.force_peak_M_N))
        if p < len(thermal_load):
            out[f"{DOMAIN}.spindle_thermal_load"][p] = thermal_load[p]
    return out


__all__ = ["channels", "tick_values", "macro_values", "MAX_PATHS", "DOMAIN"]
