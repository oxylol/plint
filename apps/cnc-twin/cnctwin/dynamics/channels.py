"""Time series the dynamics domain records (docs/models/dynamics.md "Channels").

* fast (L0, written by cnctwin.dynamics.kernels.ring_write): the relative tool-workpiece deflection at the
  reference station of every path (T0);
* tick (L1, from DynAccum.tick_u via DynamicsDomain.channel_values): tool deflection at the reference station (T0),
  workpiece deflection at the contact (S), relative deflection (T0), per path; NaN where a path has no tool;
* macro (L2): the scalar chatter channels sound and surface read (worst system of the step), and per path the
  vibration, chatter and energy terms and the stability margin of the operating point.
"""

from __future__ import annotations

from ..core.channels import Channel, Tier

DOMAIN = "dynamics"
MAX_PATHS = 4

FAST_VIBRATION = "dynamics.vibration_rel_m"

_TICK = (
    ("tool_deflection_x_m", "m", "tool deflection along x_T0 at the reference station (modes + residual), tick end"),
    ("tool_deflection_y_m", "m", "tool deflection along y_T0 at the reference station, tick end"),
    ("tool_deflection_z_m", "m", "tool deflection along z_T0 at the reference station, tick end"),
    ("work_deflection_x_m", "m", "workpiece elastic deflection along x_S at the contact, tick end"),
    ("work_deflection_y_m", "m", "workpiece elastic deflection along y_S at the contact, tick end"),
    ("work_deflection_z_m", "m", "workpiece elastic deflection along z_S at the contact, tick end"),
    ("deflection_rel_x_m", "m", "relative tool - workpiece deflection along x_T0, tick end"),
    ("deflection_rel_y_m", "m", "relative tool - workpiece deflection along y_T0, tick end"),
    ("deflection_rel_z_m", "m", "relative tool - workpiece deflection along z_T0, tick end"),
)

_MACRO_SCALAR = (
    ("vibration_pp_m", "m", "largest peak-to-peak relative vibration of any path this step"),
    ("chatter_index", "-", "non-forced / total vibration energy of the most chattering path"),
    ("chatter_freq_Hz", "Hz", "chatter frequency of that path (0: none)"),
    ("chatter_amp_m", "m", "amplitude of the non-forced component of that path"),
)

_MACRO_PATH = (
    ("vibration_pp_path_m", "m", "peak-to-peak relative vibration per path this step"),
    ("chatter_index_path", "-", "chatter index per path"),
    ("chatter_freq_path_Hz", "Hz", "chatter frequency per path"),
    ("modal_energy_J", "J", "modal energy of each path's system at the end of the step"),
    ("damping_power_W", "W", "structural damping dissipation per path, step mean"),
    ("cutting_power_W", "W", "work of the cutting forces on the modes per path, step mean"),
    ("unbalance_power_W", "W", "work of the unbalance forces on the modes per path, step mean"),
    ("base_power_W", "W", "work of the base excitation (axis accelerations) on the modes per path, step mean"),
    ("basis_change_J", "J", "energy difference booked when the path's modal basis changed this step"),
    ("depth_limit_m", "m", "stability limit at the operating point (lobe diagram), per path"),
    ("stability_margin", "-", "depth limit / depth of cut at the operating point (below 1: chatter predicted)"),
    ("n_modes", "-", "modes of each path's system"),
)


def channels() -> list[Channel]:
    out = [Channel(FAST_VIBRATION, "m", Tier.FAST, DOMAIN,
                   "relative tool - workpiece deflection at the reference station per path (x, y, z in T0)",
                   (MAX_PATHS, 3))]
    out += [Channel(f"{DOMAIN}.{name}", unit, Tier.TICK, DOMAIN, doc, (MAX_PATHS,)) for name, unit, doc in _TICK]
    out += [Channel(f"{DOMAIN}.{name}", unit, Tier.MACRO, DOMAIN, doc) for name, unit, doc in _MACRO_SCALAR]
    out += [Channel(f"{DOMAIN}.{name}", unit, Tier.MACRO, DOMAIN, doc, (MAX_PATHS,)) for name, unit, doc in _MACRO_PATH]
    return out


TICK_NAMES = tuple(f"{DOMAIN}.{name}" for name, _, _ in _TICK)
MACRO_SCALAR_NAMES = tuple(f"{DOMAIN}.{name}" for name, _, _ in _MACRO_SCALAR)
MACRO_PATH_NAMES = tuple(f"{DOMAIN}.{name}" for name, _, _ in _MACRO_PATH)

__all__ = ["channels", "DOMAIN", "MAX_PATHS", "FAST_VIBRATION", "TICK_NAMES", "MACRO_SCALAR_NAMES",
           "MACRO_PATH_NAMES"]
