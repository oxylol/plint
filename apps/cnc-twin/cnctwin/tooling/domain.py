"""ToolingDomain: builds the tool table and every tool's geometry at init, and the effective edges
of the next macro step from the previous step's wear, thermal and holding records (architecture
4.1, 5.3, 19; docs/models/tooling.md "Domain").

Bus: reads ``prev.wear.tools``, ``prev.thermal.tools``, ``prev.holding.tools`` and
``bus.motion.paths[*].tool_slot``; writes no bus record (tooling has none: its outputs are
``state.tools`` (the ToolTable), ``state.tooling`` and the kernel inputs). It must run its init
before the domains that size per-tool arrays from the ToolTable.
"""

from __future__ import annotations

import numpy as np

from ..core.abi import EdgeModel
from ..core.bus import Bus
from ..core.channels import Channel
from .abi import EdgeFrameT, EdgeTrig, ToolingInputs
from .build import build_tool
from .channels import channel_values, channels
from .envelope import spindle_nose_from_machine
from .schema import settings_from
from .spec import get, num, plain, text
from .state import ToolingState
from .table import build_tool_table
from .update import copy_edge, edge_state, effective_edge, rephase


def empty_edge() -> EdgeModel:
    """EdgeModel of a path without tool (zero teeth): same dtypes and dimensions as a real one."""
    i2 = np.zeros((0, 0), np.int32)
    f2 = np.zeros((0, 0))
    return EdgeModel(np.zeros(0), f2.copy(), f2.copy(), f2.copy(), f2.copy(), f2.copy(), f2.copy(), f2.copy(),
                     f2.copy(), f2.copy(), f2.copy(), i2.copy(), i2.copy(), i2.copy(), np.zeros((0, 0), np.int8),
                     np.zeros(3), np.zeros(2), np.zeros(0), np.zeros(0), np.array([-1.0]))


def _empty_trig() -> EdgeTrig:
    return EdgeTrig(np.zeros((0, 0)), np.zeros((0, 0)))


def _empty_frames() -> EdgeFrameT:
    return EdgeFrameT(*(np.zeros((0, 0, 3)) for _ in range(4)))


def _states_equal(a, b) -> bool:
    return all(np.array_equal(getattr(a, f), getattr(b, f)) for f in a._fields)


class ToolingDomain:
    name = "tooling"

    def channels(self) -> list[Channel]:
        return channels()

    # ------------------------------------------------------------------ init
    def init(self, ctx, state) -> None:
        project = ctx.project
        settings = settings_from(get(project, "physics"))
        tools_section = get(project, "tools")
        entries = list(get(tools_section, "list") or [])
        machine = get(project, "machine")
        table = build_tool_table(entries, machine)
        state.tools = table
        nose = spindle_nose_from_machine(machine)
        spindle_tir = {}
        for spindle in get(machine, "spindles") or []:
            spindle_tir[text(spindle, "id", "main")] = num(spindle, "runout_m")
        tooling = ToolingState()
        for slot in table.slots:
            entry = entries[slot.slot]
            geo = build_tool(get(entry, "tool"), get(entry, "holder"), entry, settings, slot=slot.slot,
                             tool_id=slot.tool_id, holder_id=slot.holder_id, db=getattr(ctx, "db", None),
                             spindle_nose=nose, spindle_runout_tir_m=spindle_tir.get(slot.spindle_id))
            tooling.tools.append(geo)
            initial = edge_state(geo, initial_edge_radius_m=geo.instance.initial_edge_radius_m)
            effective = effective_edge(geo.edge, initial)
            trig, frames = rephase(geo, effective)
            tooling.effective.append(effective)
            tooling.edge_state.append(initial)
            tooling.trig.append(trig)
            tooling.frames.append(frames)
            ctx.trace.record(0.0, 0, "tooling", "tooling.tool", self._summary(geo), [], 0)
        tooling.version = np.zeros(len(tooling.tools), np.int64)
        tooling.initial_edge_radius_m = np.array(
            [g.instance.initial_edge_radius_m if g.instance.initial_edge_radius_m is not None else np.nan
             for g in tooling.tools], dtype=float)
        state.tooling = tooling

    @staticmethod
    def _summary(geo) -> dict:
        props = geo.props
        out = {"slot": geo.slot, "number": geo.number, "diameter_m": props.diameter_m, "flutes": props.flutes,
               "runout_tir_m": geo.runout_tir_m, "elements": int(geo.edge.elem_valid.sum())}
        if props.stickout_m is not None:
            out["stickout_m"] = props.stickout_m
        if props.max_rad_s is not None:
            out["max_rad_s"] = props.max_rad_s
        return out

    # ------------------------------------------------------------------ hot loop
    def kernel_inputs(self, ctx, state, bus: Bus) -> ToolingInputs:
        """Per path: the effective edge of the path's active tool (bus.motion.paths[p].tool_slot)."""
        tooling = state.tooling
        edges, trig, frames, slots = [], [], [], []
        paths = bus.motion.paths if bus is not None else []
        for frame in paths:
            slot = int(frame.tool_slot)
            if 0 <= slot < len(tooling):
                edges.append(tooling.effective[slot])
                trig.append(tooling.trig[slot])
                frames.append(tooling.frames[slot])
            else:
                slot = -1
                edges.append(empty_edge())
                trig.append(_empty_trig())
                frames.append(_empty_frames())
            slots.append(slot)
        return ToolingInputs(tuple(edges), tuple(trig), tuple(frames), np.array(slots, np.int32))

    def absorb(self, ctx, state, outputs, bus_next: Bus) -> None:
        """Tooling has no accumulators or event buffers in the compiled loop: nothing to absorb."""
        return None

    # ------------------------------------------------------------------ L2
    def macro_step(self, ctx, state, prev: Bus, nxt: Bus) -> None:
        """Effective edges for the next step from the previous step's wear, heat and holding."""
        tooling = state.tooling
        for slot, geo in enumerate(tooling.tools):
            wear = prev.wear.tools.get(slot)
            heat = prev.thermal.tools.get(slot)
            hold = prev.holding.tools.get(slot)
            initial = float(tooling.initial_edge_radius_m[slot])
            new = edge_state(geo, wear, heat, hold, None if np.isnan(initial) else initial)
            if _states_equal(new, tooling.edge_state[slot]):
                continue
            effective_edge(geo.edge, new, tooling.effective[slot])
            tooling.trig[slot], tooling.frames[slot] = rephase(geo, tooling.effective[slot])
            tooling.edge_state[slot] = new
            tooling.version[slot] += 1

    def channel_values(self, ctx, state, bus: Bus) -> dict:
        """Macro-tier channel values for the recorder (active slot per path from ``bus``)."""
        slots = [frame.tool_slot for frame in bus.motion.paths]
        return channel_values(state, slots)

    def end_operation(self, ctx, state) -> None:
        """Tool geometry carries over between operations unchanged (wear, growth and pull-out are
        composed again from the bus at the next macro step): nothing to do."""
        return None

    def finalize(self, ctx, state) -> dict:
        tooling = state.tooling
        out = {"tools": []}
        for slot, geo in enumerate(tooling.tools):
            eff = tooling.effective[slot]
            valid = geo.edge.elem_valid == 1
            out["tools"].append({
                "slot": slot, "number": geo.number, "tool_id": geo.tool_id, "holder_id": geo.holder_id,
                "type": geo.type, "hand": geo.hand, "diameter_m": geo.props.diameter_m, "flutes": geo.props.flutes,
                "runout_tir_m": geo.runout_tir_m, "runout_source": geo.runout_source,
                "max_radius_change_m": float(np.max(np.abs(eff.elem_r_m - geo.edge.elem_r_m)[valid])) if valid.any()
                else 0.0,
                "max_axial_change_m": float(np.max(np.abs(eff.elem_z_m - geo.edge.elem_z_m)[valid])) if valid.any()
                else 0.0,
                "edge_updates": int(tooling.version[slot]), "estimates": list(geo.notes)})
        return out


__all__ = ["ToolingDomain", "empty_edge", "copy_edge", "plain"]
