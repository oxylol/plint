"""Adapters that answer other domains' tool questions from the tooling state.

* :class:`ToolingCatalog` - ``edge_model(slot)`` (thermal's ToolEdges protocol: the effective
  EdgeModel) and ``tool(slot)`` (the fields of chips' ToolInfo: kind, radius, flute length,
  flutes, mean helix, total gullet area of all flutes, holder radius and length, through-tool
  coolant, rotating, spiral).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ToolInfoView:
    slot: int
    kind: str
    radius_m: float
    flute_length_m: float
    n_flutes: int
    helix_rad: float
    gullet_area_m2: float           # all flutes together (cross-section of chip space)
    holder_radius_m: float
    holder_length_m: float
    through_coolant: bool
    rotating: bool
    spiral: str = ""

    @property
    def diameter_m(self) -> float:
        return 2.0 * self.radius_m


class ToolingCatalog:
    def __init__(self, tooling_state):
        self.tooling = tooling_state

    def edge_model(self, tool_slot: int):
        return self.tooling.edge_model(tool_slot)

    def tool(self, slot: int) -> ToolInfoView | None:
        if slot < 0 or slot >= len(self.tooling):
            return None
        geo = self.tooling.geometry(slot)
        props = geo.props
        helix = geo.info.get("helix")
        helix_rad = float(np.mean(helix)) if helix else float(np.mean(np.abs(geo.edge.elem_incl_rad[geo.edge.elem_valid == 1]))
                                                              if np.any(geo.edge.elem_valid == 1) else 0.0)
        holder_r = 0.0
        holder_len = 0.0
        if geo.layout.get("z_gauge") is not None and geo.holder is not None:
            holder_r = 0.5 * max(d for _, d in geo.holder.profile_m)
            holder_len = geo.layout["z_gauge"] - max(geo.layout.get("z_flute_end", props.flute_length_m), 0.0)
        return ToolInfoView(slot=slot, kind=geo.type, radius_m=0.5 * props.diameter_m,
                            flute_length_m=props.flute_length_m, n_flutes=props.flutes, helix_rad=helix_rad,
                            gullet_area_m2=props.gullet_volume_per_length_m3_per_m, holder_radius_m=holder_r,
                            holder_length_m=holder_len, through_coolant=props.coolant is not None,
                            rotating=geo.rotating, spiral=str(geo.info.get("spiral", "") or ""))
