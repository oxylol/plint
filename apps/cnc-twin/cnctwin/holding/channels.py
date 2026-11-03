"""Time series the holding domain records (macro tier; values from HoldingDomain.macro_values).

Per-tool channels describe the tool of the first path that has one (the active tool); per-part channels
have one column per workpiece (up to MAX_PARTS); the per-slot pull-out covers every tool of the table."""

from __future__ import annotations

from ..core.channels import Channel, Tier

MAX_PARTS = 8
MAX_SLOTS = 32

_DEFS = (
    ("holding.tool_margin", "-", "friction capacity / peak demand of the active tool's holder this step (NaN: no load "
     "or positive drive)", ()),
    ("holding.tool_torque_capacity_Nm", "Nm", "static transmissible torque of the active tool's holder (NaN: positive)",
     ()),
    ("holding.tool_pullout_m", "m", "pull-out of the active tool (depth error)", ()),
    ("holding.tool_slip_rad", "rad", "rotation of the active tool in its holder (lag)", ()),
    ("holding.tool_walk_per_rev_m", "m", "pull-out per spindle revolution of the active tool this step", ()),
    ("holding.unbalance_force_N", "N", "unbalance force |U| omega^2 of the active tool assembly at the step's speed",
     ()),
    ("holding.balance_quality_m_s", "m/s", "worst-case balance quality G of the active assembly at the step's speed",
     ()),
    ("holding.pullout_slot_m", "m", "pull-out per tool slot", (MAX_SLOTS,)),
    ("holding.part_margin", "-", "capacity factor of the step's load on each workpiece (NaN: no load or not held)",
     (MAX_PARTS,)),
    ("holding.part_slip_m", "m", "persistent slip of each workpiece's centre of mass since its setup", (MAX_PARTS,)),
    ("holding.part_lift_m", "m", "largest reversible lift-off / tipping of each workpiece this step", (MAX_PARTS,)),
    ("holding.location_error_m", "m", "location error of each workpiece's centre of mass in its setup", (MAX_PARTS,)),
    ("holding.clamp_force_N", "N", "total clamping / gripping force on each workpiece (dynamic for chucks)",
     (MAX_PARTS,)),
)


def channels() -> list[Channel]:
    return [Channel(name, unit, Tier.MACRO, "holding", doc, shape) for name, unit, doc, shape in _DEFS]


NAMES = tuple(d[0] for d in _DEFS)

__all__ = ["channels", "NAMES", "MAX_PARTS", "MAX_SLOTS"]
