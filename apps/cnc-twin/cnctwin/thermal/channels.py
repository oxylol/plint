"""Time series the thermal domain records (macro tier; values from ThermalDomain.macro_values)."""

from __future__ import annotations

from ..core.channels import Channel, Tier

MAX_SPINDLES = 4
MAX_AXES = 9
MAX_WORKPIECES = 2

_DEFS = (
    ("thermal.tool_edge_max_K", "K", "highest edge temperature of the active tools over the in-cut points", ()),
    ("thermal.tool_flank_max_K", "K", "highest flank-contact temperature of the active tools", ()),
    ("thermal.tool_edge_cycle_amp_max_K", "K", "largest heating/cooling amplitude per revolution at an edge", ()),
    ("thermal.tool_body_K", "K", "mean body temperature of the active tool over its stickout", ()),
    ("thermal.holder_K", "K", "mean temperature of the active tool's holder", ()),
    ("thermal.tool_length_growth_m", "m", "thermal length growth of the active tool (stickout)", ()),
    ("thermal.holder_growth_m", "m", "thermal growth of the active holder (gauge line to nose)", ()),
    ("thermal.spindle_temp_K", "K", "spindle temperature per spindle (machine order)", (MAX_SPINDLES,)),
    ("thermal.spindle_growth_m", "m", "spindle growth since the start after compensation, per spindle", (MAX_SPINDLES,)),
    ("thermal.axis_drift_m", "m", "ballscrew error at the current position per linear axis (alphabetical)", (MAX_AXES,)),
    ("thermal.structure_drift_m", "m", "tool-to-table drift of the structure since the operation start", ()),
    ("thermal.work_surface_K", "K", "two-scale surface temperature at the contact, per workpiece", (MAX_WORKPIECES,)),
    ("thermal.work_preheat_K", "K", "temperature of the material entering the cut, per workpiece", (MAX_WORKPIECES,)),
    ("thermal.work_bulk_mean_K", "K", "mass-weighted mean workpiece temperature", (MAX_WORKPIECES,)),
    ("thermal.work_bulk_max_K", "K", "highest finite-volume temperature of the workpiece", (MAX_WORKPIECES,)),
    ("thermal.fixture_temp_K", "K", "fixture temperature per workpiece", (MAX_WORKPIECES,)),
    ("thermal.heat_to_chip_W", "W", "cutting heat leaving with the chips (step mean)", ()),
    ("thermal.heat_to_tool_W", "W", "cutting heat entering the tools (step mean)", ()),
    ("thermal.heat_to_work_W", "W", "cutting heat entering the workpieces (step mean)", ()),
    ("thermal.heat_to_coolant_W", "W", "convective heat from tools and workpieces into the coolant (step mean)", ()),
    ("thermal.removed_enthalpy_W", "W", "enthalpy leaving with removed material (step mean)", ()),
    ("thermal.ledger_residual_rel", "-", "largest relative energy residual of the thermal solvers this step", ()),
    ("thermal.work_deposit_scale", "-", "partitioned work heat / cutting's deposit total (consistency)", (MAX_WORKPIECES,)),
)


def channels() -> list[Channel]:
    return [Channel(name, unit, Tier.MACRO, "thermal", doc, shape) for name, unit, doc, shape in _DEFS]


NAMES = tuple(d[0] for d in _DEFS)
