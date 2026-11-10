"""ThermalDomain: the thermal slow domain (docs/models/thermal.md; Domain contract: core/domain.py).

Per macro step it reads ``prev`` (cutting zone heat and fraction tables, coolant, chips, motion,
removal geometry, wear) and writes only ``nxt.thermal``:

1. distributes each contact's zone heat once (partition.py) - chips, tool elements, workpiece;
2. steps the active tools' ladders, 3-D cutting-region fields (toolfield.py) and edge fields
   (toolmodel.py), idle tools lazily;
3. steps spindles (bearing heat, motor losses, holder conduction: the network of spindle.py, or a lumped
   body), ballscrew axes and the structure, which also takes the heat the spindle heads pass into it
   (machine.py);
4. accumulates the work heat rows and every ``every_n_macro`` steps runs the workpiece finite
   volumes and fixtures (workpiece.py); evaluates the two-scale surface temperature and preheat
   at every contact and raises thermal_damage findings (damage.py);
5. books every solver's energy ledger and the core ledger's heat destinations.

The domain takes no part in the hot loop (kernel_inputs and absorb return None: it is an L2
domain by the architecture, its coupling to the forces goes through ProcessCoefficients).
"""

from __future__ import annotations

import functools
import math

import numpy as np

from ..core.bus import ThermalFeedback
from ..core.errors import InputError
from ..units import REFERENCE_TEMP_K
from . import inputs as I
from . import machine as M
from . import props
from .abi import L_TMAX, L_TMEAN
from .channels import MAX_AXES, MAX_SPINDLES, MAX_WORKPIECES, channels as thermal_channels
from . import surfacetemp as ST
from .damage import DamageMonitor, hj_constant, thresholds
from .partition import FractionTable, split_contact
from .protocols import FixtureContact, ToolEdges, WorkGeometry
from .schema import settings as read_settings
from .state import ThermalState
from .toolmodel import ToolEnvironment, ToolThermal, tool_spec
from .workpiece import ContactSource, WorkEnvironment, WorkThermal, _p as work_params
from . import convection as cv


@functools.lru_cache(maxsize=4096)
def _edge_split_cached(edge_radius_m: float, h_min_m: float, rake_rad: float, incl_rad: float) -> tuple[float, float]:
    """(friction share of the edge force's power, contact arc) of material pressed under a rounded edge
    (contact.contact_power_parts at depth h_min). The pressure is Prandtl's (2 + pi) k and the friction stress
    min(mu p, k): k for every mu above 1 / (2 + pi) = 0.19, so the split does not depend on k or on mu there (mu is
    taken at that limit)."""
    from ..mechanics import contact as MC
    if not (edge_radius_m > 0.0 and h_min_m > 0.0):
        return 1.0, 0.0
    f_p, f_tau, arc = MC.contact_power_parts(h_min_m, edge_radius_m, h_min_m, rake_rad, incl_rad, 1.0, 1.0)
    total = f_p + f_tau
    return (float(f_tau / total) if total > 0.0 else 1.0), float(arc)


def _edge_split(th, cut, edges, heat, arc_m: float) -> tuple[float, float]:
    """(friction share of the contact's edge heat, depth of its pressure part's source) for one contact: the edge
    state weighted by the work's edge heat (elements' hone radius, rake, inclination, their group's h_min), the
    ploughed-zone depth mechanics-contact ploughing_zone_depth_ratio x arc. Without h_min (no coefficient table) the
    edge heat is all friction at the surface (the larger surface rise)."""
    from ..mechanics import params as MP
    table = th.fractions.get(int(cut.tool_slot))
    w = np.asarray(heat.work_plough_elem_J, dtype=float)
    if edges is None or table is None or table.h_min_m.size == 0 or not w.size or float(w.sum()) <= 0.0 or \
            np.shape(edges.elem_edge_radius_m) != w.shape:
        return 1.0, 0.0
    groups = np.clip(np.asarray(edges.elem_group, dtype=np.int64), 0, table.h_min_m.shape[0] - 1)
    mean = lambda a: float(np.sum(np.asarray(a, dtype=float) * w) / w.sum())
    r = mean(edges.elem_edge_radius_m)
    h_min = mean(table.h_min_m[groups])
    rake = mean(edges.elem_rake_rad)
    incl = mean(edges.elem_incl_rad)
    key = tuple(float(f"{x:.3g}") for x in (r, h_min, rake, incl))
    friction, arc = _edge_split_cached(*key)
    ratio = float(MP.number(MP.CONTACT, "ploughing_zone_depth_ratio"))
    return friction, ratio * (arc_m if arc_m > 0.0 else arc)


class ThermalDomain:
    """Heat in tools, holders, spindles, axes and workpieces; thermal damage; solver ledgers."""

    name = "thermal"

    def __init__(self, geometry: WorkGeometry | None = None, edges: ToolEdges | None = None,
                 contact: FixtureContact | None = None):
        self.geometry = geometry
        self.edges = edges
        self.contact = contact

    # ------------------------------------------------------------------ Domain interface
    def channels(self):
        return thermal_channels()

    def kernel_inputs(self, ctx, state, bus):
        return None

    def absorb(self, ctx, state, outputs, bus_next) -> None:
        return None

    def init(self, ctx, state) -> None:
        project = ctx.project
        st = read_settings(I.physics_thermal(project))
        ambient = I.ambient_K(project, REFERENCE_TEMP_K)
        start = I.machine_state(project)
        machine = I.machine_raw(project)
        material = getattr(project, "material", None) if project is not None else None
        mat_raw = I.raw(material) or {}
        mat_table = props.workpiece_table(material) if material is not None else None
        family = str(I.plain(mat_raw.get("family")) or "") if isinstance(mat_raw, dict) else ""
        name = str(I.plain(mat_raw.get("name")) or I.plain(mat_raw.get("id")) or "the workpiece material") \
            if isinstance(mat_raw, dict) else "the workpiece material"
        time_s = state.clock.tick * ctx.tick_s
        structure = M.structure_model(machine, ambient, start)
        th = ThermalState(st, ambient, mat_table if mat_table is not None else props.material_table_by_id("aisi1045"),
                          family, DamageMonitor(name, thresholds(material), hj_constant(material)), structure,
                          time_s=time_s)
        if mat_table is None:
            th.estimated.append("workpiece material: none in the project, aisi1045 used for any grid")
        th.spindles = M.spindle_models(machine, ambient, start, st.energy_tol, time_s, structure.body.T_K)
        M.link_structure(structure, th.spindles)        # the head's share of the loop is the spindle model's
        th.axes = M.axis_models(machine, ambient, start, st.energy_tol, time_s)
        for sid, model in sorted(th.spindles.items()):
            th.estimated.extend(f"spindle {sid}: {e}" for e in model.estimated)
        for axis, model in sorted(th.axes.items()):
            th.estimated.extend(f"axis {axis}: {e}" for e in model.estimated)
        th.estimated.extend(f"structure: {e}" for e in structure.estimated)
        th.fixture_mass_kg = self._fixture_mass(project)
        for slot in state.tools.slots:
            th.tools[slot.slot] = self._new_tool(project, slot, st, ambient, time_s)
            self._ensure_field(th.tools[slot.slot], st)
        if self.geometry is not None:
            th.works[0] = WorkThermal(0, self.geometry, th.material, st, ambient, family, time_s,
                                      th.fixture_mass_kg or None, self.contact)
            th.published[0] = th.works[0].published_grid()
        th.part_index = state.clock.part_index
        th.op_index = state.clock.op_index
        for key in ("chip", "tool", "work", "coolant", "removed", "unplaced", "generated"):
            th.totals[key] = 0.0
        state.thermal = th

    def macro_step(self, ctx, state, prev, nxt) -> None:
        th: ThermalState = state.thermal
        self._attach(th)
        st = th.settings
        dt = ctx.macro_s
        if state.clock.part_index != th.part_index:
            self._new_part(th)
            th.part_index = state.clock.part_index
        paths = self._paths(prev, th)
        coolant = prev.coolant
        removal = {c.contact: c for c in prev.removal.contacts}
        feedback = ThermalFeedback()
        step = {"chip": 0.0, "tool": 0.0, "work": 0.0, "coolant": 0.0, "removed": 0.0, "unplaced": 0.0,
                "generated": 0.0}
        # ---- 1. distribution (fraction tables in force = those published one step earlier)
        in_force = th.fractions
        next_tables = {}
        for slot, coeff in sorted(prev.cutting.coefficients.items()):
            table = FractionTable.from_coefficients(coeff)
            if table is not None:
                next_tables[int(slot)] = table
        contacts = []
        for cut in sorted(prev.cutting.contacts, key=lambda c: c.contact):
            rem = removal.get(cut.contact)
            workpiece = rem.workpiece if rem is not None else 0
            slot = int(cut.tool_slot)
            table = in_force.get(slot) or next_tables.get(slot)
            edges = self.edges.edge_model(slot) if (self.edges is not None and slot >= 0) else None
            vb = self._vb(prev, slot)
            tool = th.tools.get(slot)
            heat = split_contact(cut, workpiece, table, edges, vb,
                                 rotating=not bool(tool is not None and tool.layout.turning))
            if heat.fallback:
                self._note(ctx, th, f"partition_fallback:{slot}", "thermal:partition_fallback",
                           {"tool_slot": slot, "generated_J": heat.generated_J}, state.clock.tick)
            contacts.append((cut, rem, heat, edges, vb))
            feedback.heat_to_chip_J[cut.contact] = heat.chip_J
            feedback.heat_to_tool_J[cut.contact] = heat.tool_J
            feedback.heat_to_work_J[cut.contact] = heat.work_J
            step["chip"] += heat.chip_J
            step["tool"] += heat.tool_J
            step["work"] += heat.work_J
            step["generated"] += heat.generated_J
        th.fractions = {**th.fractions, **next_tables}
        # ---- 2. tools
        to_ends = self._step_tools(ctx, state, th, prev, paths, contacts, coolant, dt, feedback, step)
        # ---- 3. machine
        chips_machine = float(getattr(prev.chips, "heat_to_machine_J", 0.0) or 0.0)   # chips' added field
        self._step_machine(th, paths, to_ends, dt, feedback, chips_machine)
        # ---- 4. workpieces
        scales = self._work_heat(ctx, th, contacts, paths, prev, dt, feedback)
        fv_now = th.macro_since_fv + 1 >= st.every_n_macro
        th.macro_since_fv += 1
        window_steps = th.macro_since_fv
        if fv_now:
            self._fv_all(ctx, th, prev, paths, contacts, coolant, th.macro_since_fv * dt, step, feedback)
            th.macro_since_fv = 0
        self._surfaces(ctx, state, th, contacts, paths, coolant, window_steps * dt, fv_now, dt, feedback)
        th.time_s += dt
        # ---- 5. publish
        self._publish(th, feedback, paths)
        feedback.heat_to_coolant_J = step["coolant"]
        feedback.unplaced_heat_J = step["unplaced"]
        state.ledger.heat_to_chip_J += step["chip"]
        state.ledger.heat_to_tool_J += step["tool"]
        state.ledger.heat_to_work_J += step["work"]
        state.ledger.heat_to_coolant_J += step["coolant"]
        state.ledger.removed_enthalpy_J += step["removed"] + step["unplaced"]
        for key, value in step.items():
            th.totals[key] = th.totals.get(key, 0.0) + value
        self._values(th, feedback, step, dt, scales)
        nxt.thermal = feedback

    def end_operation(self, ctx, state) -> None:
        th: ThermalState = state.thermal
        self._attach(th)
        st = th.settings
        dry = WorkEnvironment(T_air_K=th.ambient_K, T_table_K=th.structure.body.T_K)
        step = {"coolant": 0.0, "removed": 0.0, "unplaced": 0.0}
        for w, work in sorted(th.works.items()):
            if th.macro_since_fv > 0 or work.pending:
                result = work.fv_step(max(th.macro_since_fv, 1) * ctx.macro_s, dry, th.time_s)
                self._book_fv(th, work, result, step)
            mean_K, max_K = work.bulk()
            th.unload.append({"op": int(th.op_index), "workpiece": int(w), "mean_K": mean_K, "max_K": max_K,
                              "fixture_K": work.fixture.T_K, "time_s": th.time_s,
                              "settle_s": self._settle_time(th, work, mean_K)})
            work.clamped = False
            work.update_contact()
        th.macro_since_fv = 0
        turret_flush = self._flush_fields(th, step) + self._flush_spindles(th, th.time_s)
        wait = I.time_between_ops_s(ctx.project)
        if turret_flush:
            th.structure.body.step(0.0, 0.0, th.ambient_K, turret_flush, st.energy_tol, th.time_s)
        if wait > 0.0:
            n = st.cooldown_steps
            for w, work in sorted(th.works.items()):
                for i in range(n):
                    result = work.fv_step(wait / n, dry, th.time_s + wait * (i + 1) / n)
                    self._book_fv(th, work, result, step)
                th.published[w] = work.published_grid()
            env = ToolEnvironment(in_spindle=False, T_air_K=th.ambient_K, T_end_K=th.structure.body.T_K,
                                  T_fluid_K=th.ambient_K)
            turret_heat = 0.0
            for slot, tool in sorted(th.tools.items()):
                turret_heat += tool.advance_idle(th.time_s + wait, env, st.idle_tool_steps, st.energy_tol)["spindle_J"]
            head_heat = 0.0
            for sid, spindle in sorted(th.spindles.items()):
                spindle.step(wait, np.zeros(1), 0.0, st.energy_tol, th.time_s + wait, T_struct_K=th.structure.body.T_K)
                head_heat += spindle.last_to_structure_J
            for axis, model in sorted(th.axes.items()):
                model.step(wait, 0.0, th.ambient_K, st.energy_tol, th.time_s + wait)
            th.structure.body.step(wait, 0.0, th.ambient_K, self._table_heat(th) + turret_heat + head_heat,
                                   st.energy_tol, th.time_s + wait)
            th.time_s += wait
        for axis, model in sorted(th.axes.items()):
            model.T0_K = model.body.T_K
        th.structure.T0_K = th.structure.body.T_K
        for w, work in sorted(th.works.items()):
            work.clamped = True
            work.update_contact()
        state.ledger.heat_to_coolant_J += step["coolant"]
        state.ledger.removed_enthalpy_J += step["removed"] + step["unplaced"]
        th.totals["coolant"] = th.totals.get("coolant", 0.0) + step["coolant"]
        th.totals["removed"] = th.totals.get("removed", 0.0) + step["removed"]
        th.totals["unplaced"] = th.totals.get("unplaced", 0.0) + step["unplaced"]
        th.op_index += 1
        th.engaged_s = {}

    def finalize(self, ctx, state) -> dict:
        th: ThermalState = state.thermal
        self._attach(th)
        flush_step = {"coolant": 0.0}
        flushed = self._flush_fields(th, flush_step) + self._flush_spindles(th, th.time_s)
        state.ledger.heat_to_coolant_J += flush_step["coolant"]
        th.totals["coolant"] = th.totals.get("coolant", 0.0) + flush_step["coolant"]
        leftover = self._table_heat(th) + flushed
        if leftover:
            th.structure.body.step(0.0, 0.0, th.ambient_K, leftover, th.settings.energy_tol, th.time_s)
        ledgers = self._ledgers(th)
        tools = {}
        for slot, tool in sorted(th.tools.items()):
            tools[str(slot)] = {"edge_max_K": th.maxima.get(f"tool.{slot}.edge_K", tool.last_heat.body_temp_K),
                                "body_max_K": th.maxima.get(f"tool.{slot}.body_K", tool.last_heat.body_temp_K),
                                "cycle_amp_max_K": th.maxima.get(f"tool.{slot}.amp_K", 0.0),
                                "length_growth_m": tool.last_heat.length_growth_m,
                                "holder_growth_m": tool.last_heat.holder_growth_m}
            if tool.field is not None:
                f = tool.field
                tools[str(slot)]["field"] = {"kind": f.kind, "cells": int(f.sys.T.size), "wedge_rad": f.beta,
                                             "teeth": f.mult, "last_iterations": f.iterations,
                                             "last_residual": f.residual}
        works = {}
        for w, work in sorted(th.works.items()):
            mean_K, max_K = work.bulk()
            works[str(w)] = {"mean_K": mean_K, "max_K": max_K, "surface_max_K": th.maxima.get(f"work.{w}.surface_K"),
                             "fixture_K": work.fixture.T_K, "settle_s": self._settle_time(th, work, mean_K),
                             "fv_steps": work.fv_count}
        return {
            "tools": tools, "workpieces": works, "unload": list(th.unload),
            "spindles": {sid: self._spindle_report(th, sid, m) for sid, m in sorted(th.spindles.items())},
            "axes": {a: {"temp_K": m.body.T_K, "strain": m.strain()} for a, m in sorted(th.axes.items())},
            "structure": {"temp_K": th.structure.body.T_K, "drift_m": th.structure.drift_m(),
                          "spindle_share_m_per_K": th.structure.spindle_share_m_per_K},
            "damage": {"counts": dict(sorted(th.damage.counts.items())), "max_K": dict(sorted(th.damage.max_K.items()))},
            "energy_J": {k: v for k, v in sorted(th.totals.items())},
            "ledgers": ledgers, "estimated": list(th.estimated),
        }

    @staticmethod
    def _spindle_report(th, sid, m) -> dict:
        out = {"temp_K": m.body.T_K, "growth_m": m.growth_rel_m(), "growth_abs_m": m.growth_abs_m(),
               "growth_max_m": th.maxima.get(f"spindle.{sid}.growth_m", 0.0), "model": m.model}
        if m.network is not None:
            net = m.network
            out["node_temps_K"] = {name: float(T) for name, T in zip(net.names, net.temps_K())}
            out["growth_parts_m"] = net.growth_parts_m()
            out["energy_J"] = {k: float(v) for k, v in sorted(net.totals.items())}
        return out

    # ------------------------------------------------------------------ results for the recorder
    def macro_values(self, state) -> dict:
        """Channel values of the last macro step (names of channels.py)."""
        return dict(state.thermal.values)

    # ------------------------------------------------------------------ helpers: setup
    def _attach(self, th: ThermalState) -> None:
        for work in th.works.values():
            if work.geometry is None:
                if self.geometry is None:
                    raise InputError("thermal: a resumed workpiece grid needs the WorkGeometry provider")
                work.geometry = self.geometry
                work.contact_provider = self.contact

    def _fixture_mass(self, project) -> float:
        try:
            ops = getattr(project, "operations", None) or []
            first = ops[0] if len(ops) else None
            holding = I.get(I.raw(getattr(first, "setup", None)) or {}, "workholding") if first is not None else None
            device = holding[0].get("device") if holding else None
            item = I.catalogue_item("workholding", I.plain(device)) if device is not None else None
            mass = I.number(I.raw(item) or {}, (("mass_kg", 1.0, 0.0),))
            return float(mass) if mass else 0.0
        except (TypeError, AttributeError, IndexError):
            return 0.0

    def _new_tool(self, project, slot, st, ambient, time_s) -> ToolThermal:
        tools = getattr(project, "tools", None) if project is not None else None
        entries = (tools.get("list") if isinstance(tools, dict) else getattr(tools, "list", None)) or []
        entry = None
        for candidate in entries:
            number = I.plain(candidate.get("number") if isinstance(candidate, dict) else getattr(candidate, "number", None))
            if number is not None and int(number) == int(slot.number):
                entry = candidate
                break
        if entry is None:
            raise InputError(f"thermal: tool slot {slot.slot} (T{slot.number}) has no entry in the project tools list")
        get = (lambda name: entry.get(name)) if isinstance(entry, dict) else (lambda name: getattr(entry, name, None))
        tool_item = I.catalogue_item("tools", get("tool") if get("tool") is not None else slot.tool_id)
        holder_item = I.catalogue_item("holders", get("holder") if get("holder") is not None else slot.holder_id)
        stick = get("stickout_m")
        if stick is None and get("stickout_mm") is not None:
            stick = float(I.plain(get("stickout_mm"))) * 1e-3
        spec = tool_spec({"stickout_m": stick}, tool_item, holder_item)
        mat_tool = props.tool_material_table_by_id(spec.tool_material_id)
        mat_holder = props.material_table_by_id(spec.holder_material_id)
        return ToolThermal(slot.slot, spec, mat_tool, mat_holder, st.tool_cell_m, ambient, time_s)

    def _ensure_field(self, tool: ToolThermal, st, edges=None) -> None:
        """Build the tool's 3-D cutting-region field once its EdgeModel is known (physics.thermal.tool_field)."""
        if not st.tool_field or tool.field is not None:
            return
        if edges is None and self.edges is not None:
            edges = self.edges.edge_model(tool.slot)
        if edges is not None:
            tool.enable_field(st, edges, every_n_steps=st.tool_field_every_n_macro, interval_s=st.tool_field_interval_s)

    def _flush_fields(self, th: ThermalState, step: dict) -> float:
        """Solve every tool field over the time it accumulated (end of an operation, end of the run);
        returns the heat its clamped end passed to the turret / table (to the structure body)."""
        env = ToolEnvironment(in_spindle=False, T_air_K=th.ambient_K, T_end_K=th.structure.body.T_K,
                              T_fluid_K=th.ambient_K)
        to_structure = 0.0
        for slot, tool in sorted(th.tools.items()):
            result = tool.flush_field(env, th.settings.energy_tol)
            step["coolant"] = step.get("coolant", 0.0) + result["coolant_J"]
            to_structure += result["spindle_J"]
        return to_structure

    def tool_temperature_at(self, state, slot: int, points_T) -> np.ndarray:
        """Tool temperatures (K) at points of the tool frame T of a slot (rotating tools at spindle angle 0):
        the 3-D cutting-region field where it exists, else the ladder's body temperature there."""
        return state.thermal.tool_temperature_at(slot, points_T)

    def _new_part(self, th: ThermalState) -> None:
        for w, work in sorted(th.works.items()):
            work.reset_material(th.ambient_K)
            th.published[w] = work.published_grid()
        th.macro_since_fv = 0
        th.engaged_s = {}

    def _note(self, ctx, th, key: str, event: str, quantities: dict, tick: int) -> int:
        """Write a trace event once per key (and return its id)."""
        if key in th.trace_notes:
            return int(th.trace_notes[key])
        event_id = ctx.trace.record(tick * ctx.tick_s, tick, "thermal", event, quantities, [])
        th.trace_notes[key] = int(event_id)
        return int(event_id)

    @staticmethod
    def _vb(prev, slot):
        wear = prev.wear.tools.get(slot) if prev.wear is not None else None
        return np.asarray(wear.VB_m) if wear is not None and np.size(wear.VB_m) else None

    # ------------------------------------------------------------------ helpers: motion
    def _paths(self, prev, th) -> list[dict]:
        out = []
        for frame in prev.motion.paths:
            omega = np.abs(np.asarray(frame.spindle_omega_rad_s, dtype=float))
            r_mt = np.asarray(frame.pose_M_T, dtype=float)[:3, :3]
            r_st = np.asarray(frame.pose_S_T, dtype=float)[:3, :3]
            r_ms = r_mt @ r_st.T
            axis_t = np.asarray(frame.axis_T_M, dtype=float)
            axis_t = axis_t[-1] if axis_t.ndim == 2 and axis_t.shape[0] else np.array([0.0, 0.0, 1.0])
            feed = np.asarray(frame.feed_vel_M_m_s, dtype=float)
            feed = feed[-1] if feed.ndim == 2 and feed.shape[0] else np.zeros(3)
            travel = {}
            positions = {}
            actual = np.asarray(frame.axis_actual, dtype=float)
            for a_i, name in enumerate(frame.axis_names):
                if actual.ndim != 2 or actual.shape[0] == 0 or a_i >= actual.shape[1]:
                    continue
                column = actual[:, a_i]
                last = th.last_axis_m.get(name, float(column[0]))
                travel[name] = float(abs(column[0] - last) + np.sum(np.abs(np.diff(column))))
                positions[name] = float(column[-1])
                th.last_axis_m[name] = float(column[-1])
            ticks = np.asarray(frame.tick)
            blocks = np.asarray(frame.block_index)
            tips = np.asarray(frame.tip_S_m, dtype=float)
            out.append({"path": frame.path, "slot": int(frame.tool_slot), "spindle": frame.spindle_id,
                        "workpiece": int(frame.workpiece), "omega": omega,
                        "omega_mean": float(omega.mean()) if omega.size else 0.0, "r_ms": r_ms, "axis_t": axis_t,
                        "feed": feed, "travel": travel, "positions": positions,
                        "tip_S": tips[-1].copy() if tips.ndim == 2 and tips.shape[0] else None,
                        "tick": int(ticks[-1]) if ticks.size else int(prev.motion.tick),
                        "block": int(blocks[-1]) if blocks.size else 0})
        return out

    @staticmethod
    def _path_of(paths, slot=None, workpiece=None, index=None):
        for p in paths:
            if index is not None and p["path"] == index:
                return p
        for p in paths:
            if slot is not None and p["slot"] == slot:
                return p
        for p in paths:
            if workpiece is not None and p["workpiece"] == workpiece:
                return p
        return paths[0] if paths else None

    # ------------------------------------------------------------------ helpers: tools
    def _tool_env(self, th, tool: ToolThermal, path, coolant, contact: int | None, ctx) -> ToolEnvironment:
        in_spindle = path is not None and not tool.layout.turning
        omega = path["omega_mean"] if path is not None and in_spindle else 0.0
        on = bool(coolant.on)
        h = 0.0
        reach = 0.0
        if on:
            if contact is not None and contact in coolant.h_flank_W_m2K:
                h = float(coolant.h_flank_W_m2K[contact])
                reach = float(coolant.reach_fraction.get(contact, 1.0))
            else:
                h = float(coolant.h_work_far_W_m2K)
                reach = 1.0 if h > 0.0 else 0.0
        channel_h = 0.0
        if on and coolant.through_tool_flow_m3_s > 0.0 and tool.spec.coolant_holes > 0 and in_spindle:
            value = self._channel_h(ctx, tool, coolant)
            if value is None:
                self._note(ctx, th, f"channel_props:{tool.slot}", "thermal:through_tool_coolant_unmodelled",
                           {"tool_slot": tool.slot, "flow_m3_s": float(coolant.through_tool_flow_m3_s)},
                           int(path["tick"]) if path is not None else 0)
            else:
                channel_h = value
        if tool.layout.turning:
            t_end = th.structure.body.T_K
        else:
            sid = path["spindle"] if path is not None else None
            t_end = th.spindles[sid].body.T_K if sid in th.spindles else th.ambient_K
        return ToolEnvironment(in_spindle=in_spindle, omega_rad_s=omega, T_air_K=th.ambient_K, T_end_K=t_end,
                               coolant_on=on, T_fluid_K=float(coolant.fluid_temp_K), h_coolant_W_m2K=h, reach=reach,
                               channel_h_W_m2K=channel_h)

    def _channel_h(self, ctx, tool: ToolThermal, coolant) -> float | None:
        """Film coefficient in the through-tool channels, None when the project's coolant fluid gives
        no density, viscosity, conductivity and specific heat (the channels are then not modelled and a
        trace event says so)."""
        fluid = getattr(getattr(ctx.project, "coolant", None), "fluid", None) if ctx.project is not None else None
        temp = float(coolant.fluid_temp_K)
        values = {}
        for key in ("density_kg_m3", "viscosity_Pa_s", "conductivity_W_mK", "specific_heat_J_kgK"):
            prop = getattr(fluid, key, None) if fluid is not None else None
            if prop is None:
                return None
            pts, vals = props.table_points(prop, key)
            values[key] = float(np.interp(temp, pts, vals))
        d = tool.spec.coolant_hole_diameter_m
        area = tool.spec.coolant_holes * math.pi * d * d / 4.0
        velocity = coolant.through_tool_flow_m3_s / area if area > 0.0 else 0.0
        return cv.h_pipe(velocity, d, values["density_kg_m3"], values["viscosity_Pa_s"], values["conductivity_W_mK"],
                         values["specific_heat_J_kgK"])

    def _step_tools(self, ctx, state, th, prev, paths, contacts, coolant, dt, feedback, step) -> dict:
        """Steps the tools of the paths and of the contacts; returns {"spindle id": J, ..., "turret": J}:
        the heat their holders passed into spindle noses (milling) or turrets (turning) this step."""
        st = th.settings
        to_ends = {"turret": 0.0}
        by_slot = {}
        for cut, rem, heat, edges, vb in contacts:
            by_slot.setdefault(int(cut.tool_slot), []).append((cut, rem, heat, edges, vb))
        active = {}
        for p in paths:
            if p["slot"] >= 0:
                active[p["slot"]] = p
        for slot in sorted(set(active) | set(by_slot)):
            if slot not in th.tools:
                raise InputError(f"thermal: tool slot {slot} is cutting but is not in the tool table")
            tool = th.tools[slot]
            path = active.get(slot)
            items = by_slot.get(slot, [])
            contact = items[0][0].contact if items else None
            env = self._tool_env(th, tool, path, coolant, contact, ctx)
            if tool.time_s < th.time_s - 1e-12:
                idle_env = ToolEnvironment(in_spindle=False, T_air_K=th.ambient_K, T_end_K=th.structure.body.T_K,
                                           T_fluid_K=th.ambient_K)
                result = tool.advance_idle(th.time_s, idle_env, st.idle_tool_steps, st.energy_tol)
                step["coolant"] += result["coolant_J"]
                to_ends["turret"] += result["spindle_J"]           # idle: only turret-clamped tools conduct
            for cut, rem, heat, edges, vb in items:
                if edges is not None and np.shape(edges.elem_z_m) == heat.tool_rake_elem_J.shape:
                    self._ensure_field(tool, st, edges)
                    if tool.field is not None:
                        tool.deposit_elements(edges, heat, vb)
                    else:
                        tool.deposit(edges.elem_z_m, heat.tool_rake_elem_J + heat.tool_flank_elem_J)
                else:
                    length = rem.geometry.ap_m if rem is not None and rem.geometry.ap_m > 0 else tool.spec.flute_length_m
                    tool.deposit_uniform(length, heat.tool_J)
            result = tool.step(dt, env, st.energy_tol)
            step["coolant"] += result["coolant_J"]
            if tool.layout.turning:
                to_ends["turret"] += result["spindle_J"]
            elif path is not None:
                to_ends[path["spindle"]] = to_ends.get(path["spindle"], 0.0) + result["spindle_J"]
            cut, rem, heat, edges, vb = items[0] if items else (None, None, None, None, None)
            if edges is None and self.edges is not None:
                edges = self.edges.edge_model(slot)
            omega = float(path["omega_mean"]) if path is not None and not tool.layout.turning else None
            heat_out = tool.edge_heat(edges, heat, cut.in_cut_time_s if cut is not None else None,
                                      cut.entries if cut is not None else None, vb if vb is not None else self._vb(prev, slot),
                                      dt, st, omega_rad_s=omega)
            feedback.tools[slot] = heat_out
            self._tool_maxima(th, slot, heat_out)
        for slot, tool in sorted(th.tools.items()):
            if slot not in feedback.tools:
                feedback.tools[slot] = tool.last_heat
        return to_ends

    @staticmethod
    def _tool_maxima(th, slot, heat) -> None:
        if np.size(heat.edge_peak_q_K):
            th.maxima[f"tool.{slot}.edge_K"] = max(th.maxima.get(f"tool.{slot}.edge_K", 0.0),
                                                   float(np.max(heat.edge_peak_q_K)))
            th.maxima[f"tool.{slot}.amp_K"] = max(th.maxima.get(f"tool.{slot}.amp_K", 0.0),
                                                  float(np.max(heat.edge_cycle_amp_K)))
        th.maxima[f"tool.{slot}.body_K"] = max(th.maxima.get(f"tool.{slot}.body_K", 0.0), float(heat.body_temp_K))

    # ------------------------------------------------------------------ helpers: machine
    def _step_machine(self, th, paths, to_ends, dt, feedback, chips_machine_J: float = 0.0) -> None:
        """Spindles (bearing heat, holder conduction), axes (travel), structure (fixture tables, turrets,
        chips lying on the machine)."""
        st = th.settings
        turret_heat = to_ends.get("turret", 0.0) + chips_machine_J
        for sid, spindle in sorted(th.spindles.items()):
            omega = np.zeros(1)
            for p in paths:
                if p["spindle"] == sid and p["omega"].size:
                    omega = p["omega"]
                    break
            if spindle.network is None:
                spindle.step(dt, omega, to_ends.get(sid, 0.0), st.energy_tol, th.time_s + dt,
                             T_struct_K=th.structure.body.T_K)
            else:                                                  # networks: every spindle_every_n_macro steps
                pend = th.spindle_pending.setdefault(sid, {"n": 0, "dt": 0.0, "heat_J": 0.0, "omega": []})
                pend["n"] += 1
                pend["dt"] += dt
                pend["heat_J"] += to_ends.get(sid, 0.0)
                pend["omega"].extend(float(w) for w in np.abs(np.asarray(omega, dtype=float)).ravel())
                if pend["n"] < st.spindle_every_n_macro:
                    continue
                turret_heat += self._flush_spindle(th, sid, th.time_s + dt)
                th.maxima[f"spindle.{sid}.growth_m"] = max(th.maxima.get(f"spindle.{sid}.growth_m", 0.0),
                                                           abs(spindle.growth_rel_m()))
                continue
            turret_heat += spindle.last_to_structure_J            # the head's mounting heats the structure
            th.maxima[f"spindle.{sid}.growth_m"] = max(th.maxima.get(f"spindle.{sid}.growth_m", 0.0),
                                                       abs(spindle.growth_rel_m()))
        travel = {}
        for p in paths:
            for name, value in p["travel"].items():
                travel[name] = travel.get(name, 0.0) + value
        for name, model in sorted(th.axes.items()):
            model.step(dt, travel.get(name, 0.0), th.ambient_K, st.energy_tol, th.time_s + dt)
        th.structure.body.step(dt, 0.0, th.ambient_K, self._table_heat(th) + turret_heat, st.energy_tol, th.time_s + dt)

    @staticmethod
    def _flush_spindle(th, sid: str, time_s: float) -> float:
        """Step a spindle network over its accumulated time (time-mean speeds, the holder heat of those steps);
        returns the heat its head passed into the structure."""
        pend = th.spindle_pending.pop(sid, None)
        if not pend or pend["dt"] <= 0.0:
            return 0.0
        spindle = th.spindles[sid]
        omega = np.asarray(pend["omega"], dtype=float) if pend["omega"] else np.zeros(1)
        spindle.step(pend["dt"], omega, pend["heat_J"], th.settings.energy_tol, time_s,
                     T_struct_K=th.structure.body.T_K)
        return spindle.last_to_structure_J

    def _flush_spindles(self, th, time_s: float) -> float:
        return sum(self._flush_spindle(th, sid, time_s) for sid in sorted(th.spindle_pending))

    @staticmethod
    def _table_heat(th) -> float:
        """Heat the fixtures passed into the table since the last call (given to the structure)."""
        total = 0.0
        for _, work in sorted(th.works.items()):
            total += work.fixture_to_table_J
            work.fixture_to_table_J = 0.0
        return total

    # ------------------------------------------------------------------ helpers: workpieces
    def _work(self, th, workpiece: int) -> WorkThermal:
        if workpiece not in th.works:
            if self.geometry is None:
                raise InputError(f"thermal: heat for workpiece {workpiece} but no WorkGeometry provider is wired")
            th.works[workpiece] = WorkThermal(workpiece, self.geometry, th.material, th.settings, th.ambient_K,
                                              th.material_family, th.time_s, th.fixture_mass_kg or None, self.contact)
            th.published[workpiece] = th.works[workpiece].published_grid()
        return th.works[workpiece]

    def _work_heat(self, ctx, th, contacts, paths, prev, dt, feedback) -> dict:
        scales = {}
        for cut, rem, heat, edges, vb in contacts:
            if heat.work_J == 0.0:
                continue
            workpiece = rem.workpiece if rem is not None else 0
            work = self._work(th, workpiece)
            rows = np.asarray(cut.work_heat_S, dtype=float).reshape(-1, 7)
            total = float(rows[:, 6].sum()) if rows.size else 0.0
            tool = th.tools.get(int(cut.tool_slot))
            turning = bool(tool is not None and tool.layout.turning)
            path = self._path_of(paths, slot=int(cut.tool_slot), workpiece=workpiece,
                                 index=rem.path if rem is not None else None)
            if rows.size and total > 0.0:
                scale = heat.work_J / total
                scales[workpiece] = scale
                out = np.zeros((rows.shape[0], 8))
                out[:, :6] = rows[:, :6]
                out[:, 6] = rows[:, 6] * scale
            else:
                point = rem.geometry.point_S_m if rem is not None else np.zeros(3)
                normal = self._normal_S(rem, path)
                out = np.array([[point[0], point[1], point[2], normal[0], normal[1], normal[2], heat.work_J, 0.0]])
                scales[workpiece] = float("nan")
                self._note(ctx, th, f"no_deposits:{cut.tool_slot}", "thermal:work_heat_at_contact_point",
                           {"tool_slot": int(cut.tool_slot), "work_J": heat.work_J}, prev.motion.tick)
            if turning and path is not None:
                # in the rotating S the tool angle runs as -omega t: the band it passed over during the
                # step lies at phi_p + omega t u, i.e. arc = -omega dt per row (kernel convention)
                sign = 1.0 if np.mean(prev_omega_signed(prev, path)) >= 0.0 else -1.0
                out[:, 7] = -sign * min(2.0 * math.pi, path["omega_mean"] * dt / max(out.shape[0], 1))
            work.add_rows(out)
        chips = prev.chips
        rows = np.asarray(chips.heat_to_part, dtype=float).reshape(-1, 4)
        if rows.size and len(th.works):
            out = np.zeros((rows.shape[0], 8))
            out[:, :3] = rows[:, :3]
            out[:, 6] = rows[:, 3]
            self._work(th, 0).add_rows(out)
        if chips.heat_to_fixture_J and len(th.works):
            self._work(th, 0).fixture_heat_J += float(chips.heat_to_fixture_J)
        return scales

    @staticmethod
    def _normal_S(rem, path) -> np.ndarray:
        if rem is None:
            return np.zeros(3)
        normal_M = np.asarray(rem.geometry.normal_M, dtype=float)
        if path is None or not np.any(normal_M):
            return np.zeros(3)
        return path["r_ms"].T @ normal_M

    def _work_env(self, th, work, contacts, coolant) -> WorkEnvironment:
        near = []
        ratio = work_params()["near_cut_radius_ratio"]
        for cut, rem, heat, edges, vb in contacts:
            if rem is None or rem.workpiece != work.workpiece:
                continue
            h = float(coolant.h_work_W_m2K.get(cut.contact, 0.0)) if coolant.on else 0.0
            if h <= 0.0:
                continue
            tool = th.tools.get(int(cut.tool_slot))
            diameter = tool.spec.diameter_m if tool is not None and not tool.layout.turning else \
                (tool.spec.insert_ic_m if tool is not None else work.grid.geom[3])
            p = rem.geometry.point_S_m
            near.append([p[0], p[1], p[2], ratio * diameter, h])
        wet = np.asarray(coolant.wetted) if coolant.on else None
        return WorkEnvironment(T_air_K=th.ambient_K, T_table_K=th.structure.body.T_K, coolant_on=bool(coolant.on),
                               T_fluid_K=float(coolant.fluid_temp_K), h_far_W_m2K=float(coolant.h_work_far_W_m2K),
                               near=np.asarray(near, dtype=float).reshape(-1, 5), wet=wet,
                               wet_origin_S_m=np.asarray(coolant.wetted_origin_S_m, dtype=float),
                               wet_cell_m=float(coolant.wetted_cell_m))

    def _fv_all(self, ctx, th, prev, paths, contacts, coolant, dt_fv, step, feedback) -> None:
        for w, work in sorted(th.works.items()):
            path = self._path_of(paths, workpiece=w)
            if path is not None:
                work.set_gravity(path["r_ms"].T @ np.array([0.0, 0.0, -1.0]))
            env = self._work_env(th, work, contacts, coolant)
            result = work.fv_step(dt_fv, env, th.time_s + ctx.macro_s)
            self._book_fv(th, work, result, step)
            th.published[w] = work.published_grid()
            if result["unplaced_J"] > 0.0:
                self._note(ctx, th, f"unplaced:{w}", "thermal:unplaced_heat",
                           {"workpiece": w, "unplaced_J": result["unplaced_J"]}, prev.motion.tick)

    @staticmethod
    def _book_fv(th, work, result, step) -> None:
        step["coolant"] += result["coolant_J"]
        step["removed"] += result["removed_J"] - result["added_J"]
        step["unplaced"] += result["unplaced_J"]

    def _surfaces(self, ctx, state, th, contacts, paths, coolant, window_s, fv_now, dt, feedback) -> None:
        engaged = {}
        band = {}
        for cut, rem, heat, edges, vb in contacts:
            if rem is None or heat.work_J <= 0.0:
                continue
            engaged[cut.contact] = th.engaged_s.get(cut.contact, 0.0) + dt
            work = th.works.get(rem.workpiece)
            if work is None:
                continue
            path = self._path_of(paths, slot=int(cut.tool_slot), workpiece=rem.workpiece, index=rem.path)
            source = self._source(th, cut, rem, heat, edges, vb, path, min(window_s, engaged[cut.contact]), dt)
            source.engaged_s = engaged[cut.contact]
            source.pending_s = 0.0 if fv_now else min(window_s, engaged[cut.contact])
            # the preheat's band carries the contact's work-bound heat rate averaged over edge_window_periods
            # revolutions: a step holds a whole number of passes only on average (at 5 ms steps of a 4-flute tool at
            # 10 000 rpm 3 or 4 passes), so the step's own rate would make the band follow the step
            omega = float(path["omega_mean"]) if path is not None else 0.0
            tau = th.settings.edge_window_periods * 2.0 * math.pi / omega if omega > 0.0 else 0.0
            decay = math.exp(-dt / tau) if tau > 0.0 else 0.0
            band[cut.contact] = self._band(th, cut.contact, source, path, heat.work_J, dt, decay)
            surface = work.contact_surface(source, fv_now)
            feedback.contact_surface_temp_K[cut.contact] = surface.surface_K
            feedback.contact_preheat_K[cut.contact] = surface.preheat_K
            w = rem.workpiece
            feedback.work_surface_temp_K[w] = max(feedback.work_surface_temp_K.get(w, -np.inf), surface.surface_K)
            feedback.work_preheat_K[w] = max(feedback.work_preheat_K.get(w, -np.inf), surface.preheat_K)
            th.maxima[f"work.{w}.surface_K"] = max(th.maxima.get(f"work.{w}.surface_K", 0.0), surface.surface_K)
            if source.edge is not None and source.edge.active() and th.damage.limits:
                feedback.contact_damage[cut.contact] = {}
            if th.damage.limits and surface.surface_K > th.damage.screen():
                verdicts, extra = self._damage_verdicts(th, source, surface, engaged[cut.contact])
                if verdicts is not None:
                    feedback.contact_damage[cut.contact] = {v.kind: (v.depth_m if v.damaged else 0.0)
                                                            for v in verdicts}
                if verdicts is None or any(v.damaged for v in verdicts):
                    tick = path["tick"] if path is not None else state.clock.tick
                    quantities = {"x_S_m": float(surface.point_S_m[0]), "y_S_m": float(surface.point_S_m[1]),
                                  "z_S_m": float(surface.point_S_m[2]), "background_K": surface.background_K,
                                  "feed_rise_K": surface.feed_rise_K, "flash_rise_K": surface.flash_rise_K,
                                  "work_heat_W": heat.work_J / dt,
                                  "film_boiling": float(bool(coolant.film_boiling.get(cut.contact, False))),
                                  **extra}
                    parent = ctx.trace.record(tick * ctx.tick_s, tick, "thermal", "thermal:surface_temperature",
                                              dict(quantities, surface_temp_K=surface.surface_K), [])
                    th.damage.check(ctx, state, th.op_index, w, path["block"] if path is not None else 0, tick,
                                    path["path"] if path is not None else 0, surface.surface_K, quantities, [parent],
                                    verdicts=verdicts)
        th.engaged_s = engaged
        th.band_power = band

    def _damage_verdicts(self, th, source: ContactSource, surface, engaged_s: float):
        """The damage criteria on the machined surface's temperature history of one contact (damage.py,
        surfacetemp.py): (verdicts, trace quantities); (None, {}) without edge sources (legacy threshold crossing).
        The bulk and feed-scale temperature under the flash is held for the feed source's dwell over a point
        (2 l / v_feed, at most the engagement)."""
        edge = source.edge
        if edge is None or not edge.active() or not surface.alpha_m2_s > 0.0:
            return None, {}
        from .damage import damage_params
        glx, glw = self._gl(th)
        base = surface.surface_K - surface.flash_rise_K
        depths = ST.default_depths(edge, surface.alpha_m2_s, extra=(float(damage_params()["min_depth_m"]),))
        hist = ST.history(edge, base, surface.k_W_mK, surface.alpha_m2_s, depths, glx, glw)
        dwell = 2.0 * source.feed_half_length_m / source.feed_speed_m_s if source.feed_speed_m_s > 0.0 else engaged_s
        hold_s = min(dwell, max(engaged_s, source.window_s)) if engaged_s > 0.0 else dwell
        verdicts = th.damage.verdicts(hist, base, hold_s)
        extra = {"edge_speed_m_s": edge.speed_m_s, "shear_work_W": edge.shear_W, "shear_length_m": edge.shear_length_m,
                 "arc_work_W": edge.arc_W, "plough_work_W": edge.plough_W, "arc_length_m": edge.arc_length_m,
                 "plough_depth_m": edge.plough_depth_m, "land_work_W": edge.land_W, "land_length_m": edge.land_length_m,
                 "edge_width_m": edge.width_m, "hold_s": hold_s}
        for v in verdicts:
            extra[f"{v.kind}_value"] = v.metric
            extra[f"{v.kind}_limit"] = v.limit
        return verdicts, extra

    @staticmethod
    def _gl(th):
        from .response import gauss_legendre
        return gauss_legendre(th.settings.jaeger_points)

    @staticmethod
    def _band(th, contact: int, source: ContactSource, path, work_J: float, dt: float, decay: float) -> list:
        """Running sums of the preheat band of a contact - [J, s, J x half length, J x width, offset of the contact
        point from the tool tip (3)] - and the band they give, set on ``source``: the heat rate, half length and width
        averaged over edge_window_periods revolutions, the contact point the tool tip plus its last offset where
        removal gives none (a zero point) in the step. Removal's contact geometry of one step moves with the passes it
        held (the engaged depth by up to a quarter in the macro-step case at 5 ms steps)."""
        old = th.band_power.get(contact) if contact in th.engaged_s else None
        old = list(old) + [0.0] * (7 - len(old)) if old is not None else [0.0] * 7
        sums = [old[0] * decay + work_J, old[1] * decay + dt, old[2] * decay + work_J * source.feed_half_length_m,
                old[3] * decay + work_J * source.width_m]
        point = np.asarray(source.point_S_m, dtype=float)
        tip = np.asarray(path["tip_S"], dtype=float) if path is not None and path.get("tip_S") is not None else None
        offset = np.asarray(old[4:7], dtype=float)
        if tip is not None and np.any(point != 0.0):
            offset = point - tip
        elif tip is not None:
            source.band_point_S_m = tip + offset
        source.band_power_W = sums[0] / sums[1]
        if sums[0] > 0.0:
            source.band_half_length_m = sums[2] / sums[0]
            source.band_width_m = sums[3] / sums[0]
        return sums + [float(offset[0]), float(offset[1]), float(offset[2])]

    def _source(self, th, cut, rem, heat, edges, vb, path, window_s, dt) -> ContactSource:
        pw = work_params()
        tool = th.tools.get(int(cut.tool_slot))
        turning = bool(tool is not None and tool.layout.turning)
        in_cut = np.asarray(cut.in_cut_time_s, dtype=float)
        entries = np.asarray(cut.entries, dtype=float)
        omega = path["omega_mean"] if path is not None else 0.0
        geo = rem.geometry
        feed = path["feed"] if path is not None else np.zeros(3)
        speed = float(np.linalg.norm(feed))
        direction = (path["r_ms"].T @ (feed / speed)) if (path is not None and speed > 0.0) else np.array([1.0, 0, 0])
        active = in_cut > 0.0
        weights = np.where(active, in_cut, 0.0)
        wsum = float(weights.sum())
        if edges is not None and np.shape(edges.elem_r_m) == in_cut.shape and wsum > 0.0:
            r_mean = float(np.sum(np.asarray(edges.elem_r_m) * weights) / wsum)
            h_mean = float(np.sum(heat.h_mean_m * weights) / wsum) if heat.h_mean_m.shape == in_cut.shape else 0.0
            phi = float(np.sum(heat.shear_angle_rad * weights) / wsum) if heat.shear_angle_rad.shape == in_cut.shape \
                else 0.0
            edge_r = float(np.sum(np.asarray(edges.elem_edge_radius_m) * weights) / wsum)
            vb_mean = float(np.sum(np.asarray(vb) * weights) / wsum) if vb is not None and np.shape(vb) == in_cut.shape \
                else 0.0
            db = np.asarray(edges.elem_db_m, dtype=float)
            per_tooth = np.sum(np.where(active, db, 0.0), axis=1)
            band_width = float(per_tooth[per_tooth > 0].mean()) if np.any(per_tooth > 0) else geo.ap_m
            t_cut = float(np.sum(np.max(np.where(active, in_cut, 0.0), axis=1)))
            passes = np.where(active, np.maximum(entries, 1.0), 0.0)
            t_pass = float(np.sum(np.where(active, in_cut, 0.0)) / max(np.sum(passes), 1.0))
        else:
            r_mean = (tool.spec.diameter_m / 2.0) if tool is not None else 0.0
            h_mean = float(np.sum(cut.int_h_dt) / wsum) if wsum > 0.0 else 0.0
            phi = 0.0
            edge_r = 0.0
            vb_mean = 0.0
            band_width = geo.ap_m
            t_cut = float(cut.in_cut_s) if cut.in_cut_s > 0 else dt
            t_pass = t_cut
        if phi <= 0.0:
            phi = pw["default_shear_angle_rad"]
        point = np.asarray(geo.point_S_m, dtype=float)
        if turning:
            # the generated surface of a turned part is where the tool's nose is: its radius sets the cutting speed
            # and the ring the feed source covers (removal's turning contacts carry no centroid of the removed
            # ring: their point_S_m is zero, docs/models/loop.md 15)
            if path is not None and path.get("tip_S") is not None:
                point = np.asarray(path["tip_S"], dtype=float)
            r_c = float(math.hypot(point[0], point[1]))
            v_cut = omega * r_c
            half = 0.5 * max(band_width, geo.ap_m, 1e-6)
            width = 2.0 * math.pi * max(r_c, 1e-6)
        else:
            radius = (tool.spec.diameter_m / 2.0) if tool is not None else r_mean
            v_cut = omega * r_mean
            psi = geo.engaged_angle_rad if geo.engaged_angle_rad > 0.0 else math.pi
            half = max(radius * math.sin(min(psi, math.pi) / 2.0), 1e-6)
            width = geo.ap_m if geo.ap_m > 0.0 else max(band_width, 1e-6)
        flank = vb_mean + edge_r
        band = flank + h_mean / math.tan(phi)
        flux = heat.work_J / (band * max(band_width, 1e-6) * t_cut) if band > 0.0 and t_cut > 0.0 else 0.0
        # the preheat's background is read one tool size beyond the leading edge (diameter / insert size)
        ahead = (tool.spec.insert_ic_m if turning else tool.spec.diameter_m) if tool is not None else 2.0 * half
        edge_src = self._edge_sources(th, cut, heat, edges, vb, h_mean, phi, edge_r, vb_mean, band_width, v_cut,
                                      t_cut, t_pass, turning, engaged=window_s)
        return ContactSource(point_S_m=point, feed_dir_S=np.asarray(direction),
                             power_W=heat.work_J / dt, feed_speed_m_s=speed, feed_half_length_m=half, width_m=width,
                             window_s=window_s, flash_flux_W_m2=flux, flash_half_length_m=0.5 * band,
                             flash_speed_m_s=v_cut, flash_duration_s=max(t_pass, 1e-9), ahead_m=float(ahead),
                             edge=edge_src)

    @staticmethod
    def _edge_sources(th, cut, heat, edges, vb, h_mean, phi, edge_r, vb_mean, band_width, v_cut, t_cut, t_pass,
                      turning, engaged) -> "ST.EdgeSources | None":
        """The work's shares of one edge passage as the machined surface's sources (surfacetemp.py, docs/models/
        thermal.md 3.4): heat rates while in cut (the step's work shares over the time in cut), the shear-plane
        footprint h / tan(phi), the edge arc of the flank contact the partition used (l_f - VB, edge-heat-weighted),
        the edge heat split into its friction (surface, on the arc) and pressure (plastic work to the depth of the
        ploughed zone under the arc) parts by contact.contact_power_parts, the wear-land friction on the land behind
        the arc (VB, in-cut mean; without a land it is put on the arc), the engaged edge length per tooth as the
        width, on for one passage (turning: the engagement)."""
        shear_J = float(getattr(heat, "work_shear_J", 0.0))
        flank_J = float(getattr(heat, "work_flank_J", 0.0))
        plough_J = float(getattr(heat, "work_plough_J", 0.0))
        if not (t_cut > 0.0 and v_cut > 0.0 and band_width > 0.0) or shear_J + flank_J + plough_J <= 0.0:
            return None
        lengths = np.asarray(heat.flank_length_m, dtype=float)
        weights = np.asarray(heat.work_plough_elem_J, dtype=float)
        if lengths.size and lengths.shape == weights.shape and float(weights.sum()) > 0.0:
            l_f = float(np.sum(lengths * weights) / weights.sum())
        else:
            l_f = vb_mean + edge_r
        arc = l_f - vb_mean
        if not arc > 0.0:
            arc = edge_r
        friction, delta = _edge_split(th, cut, edges, heat, arc)
        tan_phi = math.tan(phi) if phi > 0.0 else 0.0
        l_s = h_mean / tan_phi if tan_phi > 0.0 and h_mean > 0.0 else 0.0
        land_J = flank_J if vb_mean > 0.0 else 0.0
        return ST.EdgeSources(speed_m_s=float(v_cut), width_m=float(band_width),
                              duration_s=float(max(engaged, t_pass) if turning else max(t_pass, 1e-9)),
                              shear_W=shear_J / t_cut if l_s > 0.0 else 0.0, shear_length_m=float(l_s),
                              arc_W=(friction * plough_J + flank_J - land_J) / t_cut,
                              plough_W=(1.0 - friction) * plough_J / t_cut, arc_length_m=float(max(arc, 0.0)),
                              plough_depth_m=float(delta), land_W=land_J / t_cut, land_length_m=float(vb_mean))

    # ------------------------------------------------------------------ helpers: results
    @staticmethod
    def _settle_time(th, work, mean_K) -> float:
        """Lumped time for the part to come within the settle band of ambient (s; 0 if already)."""
        band = work_params()["unload_settle_band_dK"]
        excess = abs(mean_K - th.ambient_K)
        if not np.isfinite(excess) or excess <= band:
            return 0.0
        extent = work.material_extent()
        area = 2.0 * (extent[0] * extent[1] + extent[1] * extent[2] + extent[0] * extent[2])
        h = cv.h_vertical_plate(mean_K, th.ambient_K, max(extent[2], 1e-3)) + cv.h_radiation(work.emissivity, mean_K,
                                                                                            th.ambient_K)
        cp = float(np.interp(mean_K, work.mat.T_K, work.mat.cp_J_kgK))
        tau = work.mass_kg * cp / (h * area) if h * area > 0.0 else float("inf")
        return float(tau * math.log(excess / band))

    def _publish(self, th, feedback: ThermalFeedback, paths) -> None:
        for sid, spindle in sorted(th.spindles.items()):
            direction = self._spindle_direction(spindle, paths)
            feedback.spindle_temp_K[sid] = spindle.body.T_K
            feedback.spindle_growth_M_m[sid] = direction * spindle.growth_rel_m()
            feedback.spindle_growth_abs_M_m[sid] = direction * spindle.growth_abs_m()
        for name, axis in sorted(th.axes.items()):
            position = None
            for p in paths:
                if name in p["positions"]:
                    position = p["positions"][name]
                    break
            position = th.last_axis_m.get(name, axis.fixed_end_m) if position is None else position
            error = axis.error_at(position)
            vec = np.zeros(6)
            vec[:3] = np.asarray(axis.direction_M, dtype=float) * error
            feedback.axis_drift[name] = vec
            feedback.axis_drift_strain[name] = axis.strain()
            feedback.axis_drift_ref_m[name] = axis.fixed_end_m
        axis_t = paths[0]["axis_t"] if paths else np.array([0.0, 0.0, 1.0])
        feedback.structure_drift_M_m = np.asarray(axis_t, dtype=float) * th.structure.drift_m()
        for w, work in sorted(th.works.items()):
            g = work.grid
            feedback.work_temp_origin_S_m[w] = g.geom[:3].copy()
            feedback.work_temp_cell_m[w] = float(g.geom[3])
            feedback.work_temp_K[w] = th.published.get(w, work.published_grid())
            last = work.last
            feedback.work_bulk_mean_K[w] = float(last[L_TMEAN]) if np.isfinite(last[L_TMEAN]) else work.temp0_K
            feedback.work_bulk_max_K[w] = float(last[L_TMAX]) if np.isfinite(last[L_TMAX]) else work.temp0_K
            feedback.fixture_temp_K[w] = work.fixture.T_K
            feedback.pending_heat_J[w] = work.pending_energy()
            if w not in feedback.work_surface_temp_K:
                feedback.work_surface_temp_K[w] = feedback.work_bulk_mean_K[w]
                feedback.work_preheat_K[w] = feedback.work_bulk_mean_K[w]
        for entry in th.unload:
            feedback.work_unload_mean_K[int(entry["workpiece"])] = float(entry["mean_K"])
        for name, ledger in self._solver_ledgers(th):
            feedback.absorbed_J[name] = ledger.step_absorbed_J
            feedback.ledger_residual_rel[name] = ledger.worst_rel

    @staticmethod
    def _spindle_direction(spindle, paths) -> np.ndarray:
        if spindle.kind in ("turning",):
            return np.array([0.0, 0.0, 1.0])
        if spindle.kind == "sub":
            return np.array([0.0, 0.0, -1.0])
        for p in paths:
            if p["spindle"] == spindle.sid:
                axis = np.asarray(p["axis_t"], dtype=float)
                norm = np.linalg.norm(axis)
                return -axis / norm if norm > 0 else np.array([0.0, 0.0, -1.0])
        return np.array([0.0, 0.0, -1.0])

    @staticmethod
    def _solver_ledgers(th):
        out = []
        for slot, tool in sorted(th.tools.items()):
            out.append((f"tool.{slot}", tool.ledger))
        for w, work in sorted(th.works.items()):
            out.append((f"work.{w}", work.ledger))
            out.append((f"fixture.{w}", work.fixture.ledger))
        for sid, sp in sorted(th.spindles.items()):
            out.append((f"spindle.{sid}", sp.body.ledger))
        for name, ax in sorted(th.axes.items()):
            out.append((f"axis.{name}", ax.body.ledger))
        out.append(("structure", th.structure.body.ledger))
        return out

    def _ledgers(self, th) -> dict:
        return {name: {"u_J": led.u_J, "u0_J": led.u0_J, "sources_J": led.sources_J, "losses_J": led.losses_J,
                       "removed_J": led.removed_J, "worst_rel": led.worst_rel, "steps": led.steps}
                for name, led in self._solver_ledgers(th)}

    def _values(self, th, feedback, step, dt, scales) -> None:
        values = {}
        edge = [float(np.max(h.edge_peak_q_K)) for h in feedback.tools.values() if np.size(h.edge_peak_q_K)]
        flank = [float(np.max(h.flank_temp_q_K)) for h in feedback.tools.values() if np.size(h.flank_temp_q_K)]
        amp = [float(np.max(h.edge_cycle_amp_K)) for h in feedback.tools.values() if np.size(h.edge_cycle_amp_K)]
        values["thermal.tool_edge_max_K"] = max(edge) if edge else float("nan")
        values["thermal.tool_flank_max_K"] = max(flank) if flank else float("nan")
        values["thermal.tool_edge_cycle_amp_max_K"] = max(amp) if amp else 0.0
        active = [h for s, h in sorted(feedback.tools.items()) if np.size(h.edge_peak_q_K)]
        lead = active[0] if active else None
        values["thermal.tool_body_K"] = lead.body_temp_K if lead is not None else float("nan")
        values["thermal.holder_K"] = lead.holder_temp_K if lead is not None else float("nan")
        values["thermal.tool_length_growth_m"] = lead.length_growth_m if lead is not None else 0.0
        values["thermal.holder_growth_m"] = lead.holder_growth_m if lead is not None else 0.0
        sp = [m for _, m in sorted(th.spindles.items())][:MAX_SPINDLES]
        values["thermal.spindle_temp_K"] = np.array([m.body.T_K for m in sp])
        values["thermal.spindle_growth_m"] = np.array([m.growth_rel_m() for m in sp])
        values["thermal.axis_drift_m"] = np.array([float(np.linalg.norm(feedback.axis_drift[a][:3])) *
                                                   np.sign(th.axes[a].strain() or 1.0)
                                                   for a in sorted(th.axes)][:MAX_AXES])
        values["thermal.structure_drift_m"] = th.structure.drift_m()
        works = sorted(th.works)[:MAX_WORKPIECES]
        values["thermal.work_surface_K"] = np.array([feedback.work_surface_temp_K.get(w, np.nan) for w in works])
        values["thermal.work_preheat_K"] = np.array([feedback.work_preheat_K.get(w, np.nan) for w in works])
        values["thermal.work_bulk_mean_K"] = np.array([feedback.work_bulk_mean_K.get(w, np.nan) for w in works])
        values["thermal.work_bulk_max_K"] = np.array([feedback.work_bulk_max_K.get(w, np.nan) for w in works])
        values["thermal.fixture_temp_K"] = np.array([feedback.fixture_temp_K.get(w, np.nan) for w in works])
        values["thermal.heat_to_chip_W"] = step["chip"] / dt
        values["thermal.heat_to_tool_W"] = step["tool"] / dt
        values["thermal.heat_to_work_W"] = step["work"] / dt
        values["thermal.heat_to_coolant_W"] = step["coolant"] / dt
        values["thermal.removed_enthalpy_W"] = (step["removed"] + step["unplaced"]) / dt
        values["thermal.ledger_residual_rel"] = max(feedback.ledger_residual_rel.values()) \
            if feedback.ledger_residual_rel else 0.0
        values["thermal.work_deposit_scale"] = np.array([scales.get(w, np.nan) for w in works])
        th.values = values


def prev_omega_signed(prev, path) -> np.ndarray:
    """Signed spindle speed of a path over the step (sign: direction of rotation, M03 < 0)."""
    for frame in prev.motion.paths:
        if frame.path == path["path"]:
            return np.asarray(frame.spindle_omega_rad_s, dtype=float)
    return np.zeros(1)
