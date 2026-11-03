"""Time series the coolant domain records (macro tier; values from CoolantDomain.macro_values)."""

from __future__ import annotations

from ..core.channels import Channel, Tier

MAX_CONTACTS = 4

_DEFS = (
    ("coolant.on", "-", "1 while coolant, MQL or air reaches the outlets", ()),
    ("coolant.fluid_temp_K", "K", "supply temperature of the fluid at the outlets", ()),
    ("coolant.tank_temp_K", "K", "tank temperature (liquid coolants)", ()),
    ("coolant.tank_volume_m3", "m3", "liquid in the tank", ()),
    ("coolant.concentration_frac", "-", "concentrate volume fraction in the tank", ()),
    ("coolant.flow_m3_s", "m3/s", "liquid delivered to all outlets", ()),
    ("coolant.through_tool_flow_m3_s", "m3/s", "liquid through the tool's coolant holes", ()),
    ("coolant.through_tool_exit_speed_m_s", "m/s", "exit speed of the through-tool jets", ()),
    ("coolant.pump_power_W", "W", "shaft power of the running pumps (heat into the fluid)", ()),
    ("coolant.air_flow_m3_s", "m3/s", "free-air consumption of air blast, MQL and mist", ()),
    ("coolant.mql_oil_m3_s", "m3/s", "MQL oil dosage", ()),
    ("coolant.reach_fraction", "-", "flow-weighted reach of the aimed coolant at each contact", (MAX_CONTACTS,)),
    ("coolant.h_flank_W_m2K", "W/m2K", "coefficient on the tool's wetted surface per contact", (MAX_CONTACTS,)),
    ("coolant.h_rake_W_m2K", "W/m2K", "coefficient on the exposed rake face per contact", (MAX_CONTACTS,)),
    ("coolant.h_work_W_m2K", "W/m2K", "coefficient on the work near the cut per contact", (MAX_CONTACTS,)),
    ("coolant.h_work_far_W_m2K", "W/m2K", "coefficient on wetted work surfaces away from the cut", ()),
    ("coolant.friction_factor", "-", "multiplier on dry tool-chip friction per contact", (MAX_CONTACTS,)),
    ("coolant.heat_in_W", "W", "heat into the tank: thermal, chips and pumps (step mean)", ()),
    ("coolant.heat_out_W", "W", "heat out of the tank: ambient, evaporation, chiller (step mean)", ()),
    ("coolant.wetted_fraction", "-", "fraction of the wetted-grid cells with a surface that are wet", ()),
)


def channels() -> list[Channel]:
    return [Channel(name, unit, Tier.MACRO, "coolant", doc, shape) for name, unit, doc, shape in _DEFS]


NAMES = tuple(d[0] for d in _DEFS)
