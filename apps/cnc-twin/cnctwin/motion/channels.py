"""Time series the motion domain records.

Tick tier (one value per L1 tick): per path the path speed, the tool tip in M, the cutting spindle;
per machine axis (global order, padded to MAX_AXES) actual and commanded values, following error and
the demanded drive force. Macro tier: look-ahead and ledger quantities per path. Values come from
:meth:`cnctwin.motion.domain.MotionDomain.channel_values` (``last_channels``).
"""

from __future__ import annotations

from ..core.channels import Channel, Tier

MAX_PATHS = 4
MAX_AXES = 16
MAX_SPINDLES = 4

_TICK = (
    ("motion.axis_actual", "m|rad", "actual (load-side) axis values, machine order", (MAX_AXES,)),
    ("motion.axis_cmd", "m|rad", "commanded axis values after the planner, machine order", (MAX_AXES,)),
    ("motion.following_error", "m|rad", "commanded - actual per axis (servo lag, backlash, reversal spike)",
     (MAX_AXES,)),
    ("motion.axis_force_N", "N|Nm", "drive force demanded per axis (m a + friction + gravity + cutting load)",
     (MAX_AXES,)),
    ("motion.path_speed_m_s", "m/s", "tool tip speed relative to the workpiece per path", (MAX_PATHS,)),
    ("motion.tip_M_m", "m", "tool tip in the machine frame, path 1", (3,)),
    ("motion.spindle_omega_rad_s", "rad/s", "spindle speed per spindle (right-handed about its reference axis; "
                                            "M03 < 0)", (MAX_SPINDLES,)),
    ("motion.spindle_torque_Nm", "Nm", "spindle motor torque per spindle", (MAX_SPINDLES,)),
)
_MACRO = (
    ("motion.queued_commands", "-", "commands received from the controller and not finished, per path", (MAX_PATHS,)),
    ("motion.phase", "-", "planner phase per path: 0 idle, 1 motion, 2 timed, 3 in-position, 4 controller, "
                          "5 ended, 6 alarm, 7 resync", (MAX_PATHS,)),
    ("motion.cycle_time_s", "s", "span of each path since the operation start", (MAX_PATHS,)),
)


def channels() -> list[Channel]:
    out = [Channel(name, unit, Tier.TICK, "motion", doc, shape) for name, unit, doc, shape in _TICK]
    out += [Channel(name, unit, Tier.MACRO, "motion", doc, shape, "int8" if name == "motion.phase" else "float64")
            for name, unit, doc, shape in _MACRO]
    return out
