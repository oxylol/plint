"""Channels of the tooling library (macro tier): the effective geometry of the active tool per path.

Values come from :func:`channel_values` (the recorder samples it once per macro step); a channel of
max_shape (16,) has one column per controller path, NaN where a path has no tool.
"""

from __future__ import annotations

import numpy as np

from ..core.channels import Channel, Tier
from ..core.abi import EDGE_MARGIN

MAX_PATHS = 16

CHANNELS = [
    Channel("tooling.slot", "-", Tier.MACRO, "tooling", "tool slot active on each path (-1 none)", (MAX_PATHS,),
            "int32"),
    Channel("tooling.radius_eff_max_m", "m", Tier.MACRO, "tooling",
            "largest effective cutting radius of the active tool (wear, growth; runout excluded)", (MAX_PATHS,)),
    Channel("tooling.radius_change_m", "m", Tier.MACRO, "tooling",
            "change of that radius from the nominal tool (growth - wear)", (MAX_PATHS,)),
    Channel("tooling.tip_shift_m", "m", Tier.MACRO, "tooling",
            "axial shift of the effective tip along -z_T (growth, holder growth, pull-out minus axial wear)",
            (MAX_PATHS,)),
    Channel("tooling.edge_version", "-", Tier.MACRO, "tooling", "updates of the effective edge of the active tool",
            (MAX_PATHS,), "int64"),
]


def channels() -> list[Channel]:
    return list(CHANNELS)


def _radius_max(edge) -> float:
    mask = (edge.elem_valid == 1) & (edge.elem_kind != EDGE_MARGIN)
    if not np.any(mask):
        return float("nan")
    return float(np.max(edge.elem_r_m[mask]))


def _tip_z(edge) -> float:
    mask = edge.elem_valid == 1
    if not np.any(mask):
        return float("nan")
    return float(np.min(edge.elem_z_m[mask]))


def channel_values(state, slots_by_path) -> dict[str, np.ndarray]:
    """Values of the tooling channels for the slots active on each path (list index = path)."""
    out = {name: np.full(MAX_PATHS, np.nan) for name in ("tooling.radius_eff_max_m", "tooling.radius_change_m",
                                                          "tooling.tip_shift_m")}
    slot_col = np.full(MAX_PATHS, -1, np.int32)
    version = np.zeros(MAX_PATHS, np.int64)
    tooling = state.tooling
    for path, slot in enumerate(list(slots_by_path)[:MAX_PATHS]):
        slot = int(slot)
        slot_col[path] = slot
        if slot < 0 or tooling is None or slot >= len(tooling):
            continue
        nominal = tooling.nominal(slot)
        effective = tooling.edge_model(slot)
        r_eff = _radius_max(effective)
        out["tooling.radius_eff_max_m"][path] = r_eff
        out["tooling.radius_change_m"][path] = r_eff - _radius_max(nominal)
        out["tooling.tip_shift_m"][path] = _tip_z(nominal) - _tip_z(effective)
        version[path] = int(tooling.version[slot])
    out["tooling.slot"] = slot_col
    out["tooling.edge_version"] = version
    return out
