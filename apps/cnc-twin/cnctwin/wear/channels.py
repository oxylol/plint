"""Time series the wear domain records (macro tier; values from WearDomain.macro_values).

Per-tool channels describe the tool of the first path that has one (the active tool); the per-slot
life channel covers every tool of the table (up to MAX_SLOTS)."""

from __future__ import annotations

from ..core.channels import Channel, Tier

MAX_TEETH = 16
MAX_SLOTS = 32

_DEFS = (
    ("wear.VB_max_m", "m", "largest flank wear land of the active tool", ()),
    ("wear.VB_tooth_m", "m", "largest flank wear land per tooth of the active tool", (MAX_TEETH,)),
    ("wear.KT_max_m", "m", "deepest crater of the active tool", ()),
    ("wear.notch_max_m", "m", "largest notch land (depth-of-cut line) of the active tool", ()),
    ("wear.edge_radius_max_m", "m", "largest worn hone radius of the active tool", ()),
    ("wear.radial_wear_max_m", "m", "largest radial wear of the active tool", ()),
    ("wear.coating_min_m", "m", "thinnest remaining flank coating over the elements that cut (active tool)", ()),
    ("wear.bue_height_max_m", "m", "highest built-up edge on the active tool", ()),
    ("wear.flank_rate_max_m_s", "m/s", "largest flank wear-depth rate per second in cut (active tool, this step)", ()),
    ("wear.chipped_elements", "-", "chipped edge elements of the active tool", ()),
    ("wear.thermal_damage_max", "-", "largest thermal-fatigue damage (Miner) of the active tool", ()),
    ("wear.section_fatigue_max", "-", "largest section fatigue damage (Miner) of the active tool", ()),
    ("wear.section_stress_max_Pa", "Pa", "largest equivalent section stress of the active tool this step", ()),
    ("wear.life_used_fraction", "-", "life used of the active tool (first criterion)", ()),
    ("wear.remaining_life_s", "s", "cutting time left to the first criterion (active tool; NaN when not growing)",
     ()),
    ("wear.life_used_slot", "-", "life used per tool slot", (MAX_SLOTS,)),
)


def channels() -> list[Channel]:
    return [Channel(name, unit, Tier.MACRO, "wear", doc, shape) for name, unit, doc, shape in _DEFS]


NAMES = tuple(d[0] for d in _DEFS)

__all__ = ["channels", "NAMES", "MAX_TEETH", "MAX_SLOTS"]
