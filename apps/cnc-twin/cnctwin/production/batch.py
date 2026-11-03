"""Per-part records and the batch: what the loop calls after each part, and production.json.

:class:`ProductionAggregator` takes one :class:`PartInputs` per finished part (its executed blocks,
the sampled spindle and axis signals for the energy, the removed mass and the state of every tool)
and keeps the part's record: cycle-time breakdown, operator and takt time, energy by consumer,
coolant and air, tool life used, wear offsets, tool changes and tool cost. :meth:`result` adds the
batch: totals and means over the simulated parts, and when the project plans more parts than were
simulated, the projection of the remaining parts (each tool's life used per part continued at its
mean over the simulated parts, an edge change each time the life used passes a whole edge life,
cycle, energy and consumables those of the last simulated part, operator time recomputed per part).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.errors import InputError
from ..db.models import params as model_params
from ..sanity.context import ToolFacts, tool_facts
from ..sanity.scan import COOLANT_OFF, COOLANT_ON
from ..sanity.si import get, items, num, plain, si_view, text
from .consumables import air_use, coolant_use
from .cycle import CycleTime, cycle_time, cycle_to_dict, lines_to_dict
from .energy import (AxisModel, SpindleModel, air_specific_energy_J_m3, axis_energy, axis_models, base_energy,
                     conveyor_power_W, process_feed_energy, pump_energy, spindle_energy, supply_pressure_Pa)
from .operator import OperatorModel
from .timeline import CATEGORIES, TimelineBlock
from .toolcost import tool_cost

FORMAT = "cnctwin-production"
VERSION = 1


@dataclass
class ToolState:
    """A tool slot at the end of a part (wear domain / tools.json)."""

    number: int
    slot: int = -1
    tool_id: str = ""
    life_used_frac: float | None = None          # life used of the edge in use at the end of the part
    life_used_part_frac: float | None = None     # life used during this part (derived when None)
    cut_time_s: float | None = None              # cumulative cutting time of the edge in use
    VB_max_m: float | None = None
    wear_offsets_m: dict[str, float] = field(default_factory=dict)   # {"length": m, "radius": m}
    changes: int = 0                             # edge / sister changes during this part
    broken: bool = False


@dataclass
class EnergyInputs:
    """Signals of one part, one sample per entry of ``dt_s`` (piecewise constant)."""

    dt_s: np.ndarray
    spindle_omega_rad_s: np.ndarray
    spindle_shaft_power_W: np.ndarray
    axis_vel_m_s: np.ndarray | None = None       # [n, n_axes] in the order of axis_names
    axis_names: tuple[str, ...] = ()
    axis_process_power_W: np.ndarray | None = None
    feed_work_J: float = 0.0                     # used when axis_process_power_W is None
    omega_tolerance_rad_s: float = 0.0           # quantisation of a recorded speed (frames.bin float32)
    velocity_tolerance_m_s: float = 0.0


@dataclass
class PartInputs:
    part: int
    blocks: list[TimelineBlock]
    energy: EnergyInputs | None = None
    removed_mass_kg: float | None = None
    tools: list[ToolState] = field(default_factory=list)


class ProductionAggregator:
    """Collects the parts of a run and writes production.json (see the module docstring)."""

    def __init__(self, *, machine, tick_s: float = 1e-6, coolant=None, concentration_frac: float | None = None,
                 tools: dict[int, ToolFacts] | None = None, operator: OperatorModel | None = None,
                 parts_planned: int = 1, dialect: str | None = None, payload_kg: dict[str, float] | None = None,
                 initial_life_frac: dict[int, float] | None = None, life_limit_s: dict[int, float] | None = None,
                 spindle_extra_inertia_kgm2: float = 0.0):
        self.machine = si_view(machine)
        if self.machine is None:
            raise InputError("production needs the machine data (power, spindles, axes, coolant pumps)")
        self.tick_s = float(tick_s)
        self.coolant = si_view(coolant)
        self.concentration_frac = concentration_frac
        self.tools = dict(tools or {})
        self.operator = operator or OperatorModel.from_project({}, self.machine)
        self.parts_planned = max(int(parts_planned), 1)
        self.dialect = dialect or text(get(self.machine, "controller", "dialect"), "fanuc")
        self.payload_kg = dict(payload_kg or {})
        self.initial_life_frac = dict(initial_life_frac or {})
        self.life_limit_s = dict(life_limit_s or {})
        self.energy_params = model_params("production-energy")
        self.consumable_params = model_params("production-consumables")
        self.spindle = SpindleModel.from_machine(self.machine, extra_inertia_kgm2=spindle_extra_inertia_kgm2,
                                                 params=self.energy_params)
        self.axes = axis_models(self.machine, payload_kg=self.payload_kg, params=self.energy_params)
        self.records: list[dict] = []
        self.cycles: list[CycleTime] = []
        self._last_life: dict[int, float] = {}
        self.assumptions: list[str] = []

    @classmethod
    def from_project(cls, project, *, tick_s: float | None = None, machine=None, coolant=None,
                     tools: dict[int, ToolFacts] | None = None, parts_planned: int | None = None,
                     payload_kg: dict[str, float] | None = None) -> "ProductionAggregator":
        """From a resolved project (machine, coolant fluid and tool items as SI views)."""
        project = si_view(project)
        machine = si_view(machine) if machine is not None else si_view(get(project, "machine"))
        if machine is None or isinstance(plain(machine), str) or not items(get(machine, "spindles")):
            raise InputError("production: the machine data is not resolved (spindles, axes, power and coolant "
                             "pumps are needed): pass the machine file's SI view")
        fluid = coolant if coolant is not None else get(project, "coolant", "fluid")
        if isinstance(plain(fluid), str):
            fluid = None
        concentration = num(get(project, "coolant", "concentration_frac"))
        if tools is None:
            tools = {}
            for entry in items(get(project, "tools", "list")):
                facts = tool_facts(si_view(entry))
                tools[facts.number] = facts
        initial, limits = {}, {}
        for entry in items(get(project, "tools", "list")):
            number = int(num(get(entry, "number"), 0))
            limit = num(get(entry, "life", "limit_s"))
            used = num(get(entry, "life", "used_s"))
            if limit:
                limits[number] = limit
                initial[number] = (used or 0.0) / limit
        planned = parts_planned if parts_planned is not None else int(num(get(project, "production", "parts"), 1) or 1)
        step = tick_s if tick_s is not None else num(get(project, "physics", "tick_s"), 1e-6)
        return cls(machine=machine, tick_s=step, coolant=fluid, concentration_frac=concentration, tools=tools,
                   operator=OperatorModel.from_project(project, machine), parts_planned=planned,
                   payload_kg=payload_kg, initial_life_frac=initial, life_limit_s=limits)

    # ------------------------------------------------------------------ one part
    def add_part(self, inputs: PartInputs) -> dict:
        cycle = cycle_time(inputs.blocks, inputs.part)
        machine_s = cycle.machine_ticks * self.tick_s
        tools = self._tool_records(inputs)
        replacements = sum(t["changes"] for t in tools)
        operator = self.operator.part_time(operations=len(cycle.operations), stops=cycle.stops,
                                           part_index=inputs.part, machine_time_s=machine_s,
                                           tool_replacements=replacements)
        states = coolant_states(inputs.blocks, self.dialect, self.machine, self.coolant, self.tick_s)
        energy = self._energy(inputs, states, operator.takt_s, machine_s)
        consumables = self._consumables(inputs, states, operator.takt_s)
        energy["air_J"] = consumables["air"]["total_m3"] * energy["air_specific_J_m3"]
        energy["total_J"] = _total_energy(energy)
        energy["total_kWh"] = energy["total_J"] / 3.6e6
        cost = sum(t["cost"]["cost_per_part_eur"] for t in tools if t["cost"]["cost_per_part_eur"] is not None)
        record = {
            "part": inputs.part, "simulated": True,
            "machine_ticks": cycle.machine_ticks, "machine_s": machine_s,
            "breakdown": cycle.breakdown.to_dict(self.tick_s),
            "operations": [{"op": op.op, "span_ticks": op.span_ticks} for op in cycle.operations],
            "operator": operator.to_dict(), "takt_s": operator.takt_s,
            "removed_mass_kg": inputs.removed_mass_kg,
            "energy": energy, "consumables": consumables,
            "tools": tools, "tool_cost_eur": cost,
        }
        self.records.append(record)
        self.cycles.append(cycle)
        return record

    def _tool_records(self, inputs: PartInputs) -> list[dict]:
        records = []
        for state in sorted(inputs.tools, key=lambda s: (s.number, s.slot)):
            used = state.life_used_part_frac
            if used is None and state.life_used_frac is not None:
                before = self._last_life.get(state.number, self.initial_life_frac.get(state.number, 0.0))
                used = (state.changes - before + state.life_used_frac) if state.changes else \
                    max(state.life_used_frac - before, 0.0)
            if used is None and state.cut_time_s is not None and state.number in self.life_limit_s:
                used = state.cut_time_s / self.life_limit_s[state.number]
            if state.life_used_frac is not None:
                self._last_life[state.number] = state.life_used_frac
            facts = self.tools.get(state.number)
            cost = tool_cost(facts.price_eur if facts else None, facts.regrinds if facts else None, used)
            records.append({"number": state.number, "slot": state.slot, "tool_id": state.tool_id or
                            (facts.tool_id if facts else ""), "life_used_part_frac": used,
                            "life_used_frac": state.life_used_frac, "cut_time_s": state.cut_time_s,
                            "VB_max_m": state.VB_max_m, "wear_offsets_m": dict(sorted(state.wear_offsets_m.items())),
                            "changes": int(state.changes), "broken": bool(state.broken), "cost": cost})
        return records

    def _energy(self, inputs: PartInputs, states: "CoolantStates", takt_s: float, machine_s: float) -> dict:
        signals = inputs.energy
        result: dict = {"missing": []}
        if signals is None:
            result["spindle"] = None
            result["axes"] = None
            result["missing"].append("spindle and axis signals of the part (energy inputs)")
        else:
            result["spindle"] = spindle_energy(signals.dt_s, signals.spindle_omega_rad_s,
                                               signals.spindle_shaft_power_W, self.spindle,
                                               signals.omega_tolerance_rad_s)
            if signals.axis_vel_m_s is not None:
                models = self._axes_for(signals.axis_names)
                axes = axis_energy(signals.dt_s, signals.axis_vel_m_s, models, signals.axis_process_power_W,
                                   signals.velocity_tolerance_m_s)
                if signals.axis_process_power_W is None and signals.feed_work_J:
                    axes["process_feed_J"] = process_feed_energy(signals.feed_work_J, self.energy_params)
                    axes["electrical_J"] += axes["process_feed_J"]
                result["axes"] = axes
            else:
                result["axes"] = None
                result["missing"].append("axis velocities of the part")
        base = base_energy(takt_s, self.machine)
        result["base_J"] = base["base_J"]
        result["servo_standby_J"] = base["servo_standby_J"]
        result["pumps"] = pump_energy(states.pump_on_s, self.machine, self.energy_params)
        conveyor_s = states.conveyor_on_s if states.conveyor_commanded else machine_s
        conveyor_power = conveyor_power_W(self.machine, self.energy_params)
        result["conveyor"] = {"on_s": conveyor_s if conveyor_power > 0 else 0.0, "power_W": conveyor_power,
                              "electrical_J": conveyor_power * conveyor_s, "commanded": states.conveyor_commanded}
        result["air_specific_J_m3"] = air_specific_energy_J_m3(supply_pressure_Pa(self.machine, self.energy_params),
                                                                self.energy_params)
        return result

    def _axes_for(self, names: tuple[str, ...]) -> list[AxisModel]:
        by_name = {m.name: m for m in self.axes}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise InputError(f"axis signals for {', '.join(missing)}: not linear axes of the machine file")
        return [by_name[n] for n in names]

    def _consumables(self, inputs: PartInputs, states: "CoolantStates", takt_s: float) -> dict:
        pumps = {text(get(p, "id")): p for p in items(get(self.machine, "coolant", "pumps"))}
        pumped = sum(num(get(pumps.get(pid), "flow_m3_s"), 0.0) * seconds for pid, seconds in states.pump_on_s.items())
        tank = num(get(self.machine, "coolant", "tank_m3"))
        coolant = coolant_use(self.coolant, chip_mass_kg=inputs.removed_mass_kg, pumped_m3=pumped,
                              mql_on_s=states.mql_on_s, elapsed_s=takt_s, tank_m3=tank,
                              concentration_frac=self.concentration_frac, params=self.consumable_params)
        coolant["pumped_m3"] = pumped
        air = air_use(self.machine, machine_on_s=takt_s, blast_on_s=states.air_blast_on_s, mql_on_s=states.mql_on_s,
                      params=self.consumable_params)
        return {"coolant": coolant, "air": air, "pump_on_s": dict(sorted(states.pump_on_s.items())),
                "air_blast_on_s": states.air_blast_on_s, "mql_on_s": states.mql_on_s}

    # ------------------------------------------------------------------ the batch
    def result(self) -> dict:
        if not self.records:
            raise InputError("production: no part was recorded")
        parts = list(self.records)
        projected = self._project(parts)
        all_parts = parts + projected
        first = self.cycles[0]
        summary = self._summary(all_parts)
        legacy = _legacy_summary(parts[0], first, self.tick_s, self.operator, summary)
        document = {
            "format": FORMAT, "version": VERSION, "tick_s": self.tick_s, "categories": list(CATEGORIES),
            "parts_planned": self.parts_planned, "parts_simulated": len(parts), "parts_projected": len(projected),
            **legacy,
            "first_part": {"cycle": cycle_to_dict(first, self.tick_s), "lines": lines_to_dict(first, self.tick_s)},
            "per_part": all_parts,
            "batch": summary,
            "progression": _progression(all_parts),
            "operator_model": {k: getattr(self.operator, k) for k in (
                "load_s", "unload_s", "door_s", "clean_s", "inspect_every", "inspect_s", "stop_s", "optional_stop",
                "inspect_parallel", "first_article_inspection", "tool_replace_s", "auto_door_s", "pallets",
                "pallet_change_s")} | {"sources": dict(sorted(self.operator.sources.items()))},
            "assumptions": sorted(set(self.assumptions + ASSUMPTIONS)),
        }
        return _clean(document)

    def _project(self, parts: list[dict]) -> list[dict]:
        remaining = self.parts_planned - len(parts)
        if remaining <= 0:
            return []
        last = parts[-1]
        rates: dict[int, float] = {}
        for record in last["tools"]:
            values = [t["life_used_part_frac"] for p in parts for t in p["tools"]
                      if t["number"] == record["number"] and t["life_used_part_frac"] is not None]
            if values:
                rates[record["number"]] = sum(values) / len(values)
        life = {t["number"]: (t["life_used_frac"] if t["life_used_frac"] is not None else 0.0) for t in last["tools"]}
        projected = []
        for k in range(remaining):
            index = len(parts) + k
            tools = []
            changes_total = 0
            for record in last["tools"]:
                number = record["number"]
                rate = rates.get(number)
                if rate is None:
                    tools.append({**record, "life_used_part_frac": None, "changes": 0, "VB_max_m": None,
                                  "projected": True})
                    continue
                value = life[number] + rate
                changes = int(math.floor(value + 1e-12))
                life[number] = value - changes
                changes_total += changes
                facts = self.tools.get(number)
                cost = tool_cost(facts.price_eur if facts else None, facts.regrinds if facts else None, rate)
                tools.append({"number": number, "slot": record["slot"], "tool_id": record["tool_id"],
                              "life_used_part_frac": rate, "life_used_frac": life[number], "cut_time_s": None,
                              "VB_max_m": None, "wear_offsets_m": record["wear_offsets_m"], "changes": changes,
                              "broken": False, "cost": cost, "projected": True})
            operator = self.operator.part_time(operations=len(last["operations"]), stops=self.cycles[-1].stops,
                                               part_index=index, machine_time_s=last["machine_s"],
                                               tool_replacements=changes_total)
            energy = dict(last["energy"])
            ratio = operator.takt_s / last["takt_s"] if last["takt_s"] > 0 else 1.0
            for key in ("base_J", "servo_standby_J"):
                energy[key] = last["energy"][key] * ratio
            consumables = json.loads(json.dumps(last["consumables"]))
            air = consumables["air"]
            air["machine_m3"] = last["consumables"]["air"]["machine_m3"] * ratio
            air["total_m3"] = air["machine_m3"] + air["air_blast_m3"] + air["mql_m3"]
            energy["air_J"] = air["total_m3"] * energy["air_specific_J_m3"]
            energy["total_J"] = _total_energy(energy)
            energy["total_kWh"] = energy["total_J"] / 3.6e6
            cost = sum(t["cost"]["cost_per_part_eur"] for t in tools
                       if t.get("cost") and t["cost"]["cost_per_part_eur"] is not None)
            projected.append({"part": index, "simulated": False, "machine_ticks": last["machine_ticks"],
                              "machine_s": last["machine_s"], "breakdown": last["breakdown"],
                              "operations": last["operations"], "operator": operator.to_dict(),
                              "takt_s": operator.takt_s, "removed_mass_kg": last["removed_mass_kg"],
                              "energy": energy, "consumables": consumables, "tools": tools, "tool_cost_eur": cost})
        return projected

    def _summary(self, parts: list[dict]) -> dict:
        n = len(parts)
        machine = sum(p["machine_s"] for p in parts)
        takt = sum(p["takt_s"] for p in parts)
        energy = sum(p["energy"]["total_J"] for p in parts)
        consumers = {}
        for p in parts:
            for name, value in _energy_by_consumer(p["energy"]).items():
                consumers[name] = consumers.get(name, 0.0) + value
        coolant_keys = ("drag_out_m3", "mist_m3", "concentrate_m3", "water_m3", "oil_m3", "evaporated_water_m3",
                        "pumped_m3")
        coolant = {k: sum(p["consumables"]["coolant"].get(k) or 0.0 for p in parts) for k in coolant_keys}
        coolant_cost = [p["consumables"]["coolant"].get("cost_eur") for p in parts]
        coolant["cost_eur"] = None if any(c is None for c in coolant_cost) else sum(coolant_cost)
        air = {k: sum(p["consumables"]["air"][k] for p in parts) for k in ("machine_m3", "air_blast_m3", "mql_m3",
                                                                         "total_m3")}
        tools: dict[int, dict] = {}
        for p in parts:
            for t in p["tools"]:
                entry = tools.setdefault(t["number"], {"number": t["number"], "tool_id": t.get("tool_id", ""),
                                                       "life_used_frac": 0.0, "changes": 0, "cost_eur": 0.0,
                                                       "cost_known": True})
                if t["life_used_part_frac"] is not None:
                    entry["life_used_frac"] += t["life_used_part_frac"]
                entry["changes"] += int(t.get("changes", 0))
                value = (t.get("cost") or {}).get("cost_per_part_eur")
                if value is None:
                    entry["cost_known"] = False
                else:
                    entry["cost_eur"] += value
        tool_list = []
        for number, entry in sorted(tools.items()):
            facts = self.tools.get(number)
            known = entry.pop("cost_known")
            entry["cost_eur"] = entry["cost_eur"] if known else None
            entry["cost_per_part_eur"] = entry["cost_eur"] / n if known else None
            rate = entry["life_used_frac"] / n
            entry["parts_per_edge"] = (1.0 / rate) if rate > 0 else None
            entry["edge_lives"] = (facts.regrinds or 0) + 1 if facts else None
            tool_list.append(entry)
        tool_cost_total = sum(t["cost_eur"] for t in tool_list if t["cost_eur"] is not None)
        return {
            "parts": n, "machine_s": machine, "total_s": takt, "takt_mean_s": takt / n,
            "parts_per_hour": 3600.0 * n / takt if takt > 0 else None,
            "machine_utilisation_frac": machine / takt if takt > 0 else None,
            "energy_J": energy, "energy_kWh": energy / 3.6e6, "energy_per_part_J": energy / n,
            "energy_by_consumer_J": dict(sorted(consumers.items())),
            "coolant": coolant, "air": air, "tools": tool_list,
            "tool_cost_eur": tool_cost_total, "tool_cost_per_part_eur": tool_cost_total / n,
            "tool_changes": sum(t["changes"] for t in tool_list),
        }

    def write(self, run_dir) -> Path:
        return write_production(run_dir, self.result())


# ----------------------------------------------------------------------------- coolant, air and conveyor states

@dataclass
class CoolantStates:
    pump_on_s: dict[str, float]
    air_blast_on_s: float
    mql_on_s: float
    conveyor_on_s: float
    conveyor_commanded: bool


def coolant_states(blocks: list[TimelineBlock], dialect: str, machine, fluid, tick_s: float) -> CoolantStates:
    """Running time of every pump, the air blast, the MQL unit and the conveyor over the blocks.

    A block's ``pumps_on`` is used when the timeline gives it; otherwise the coolant M codes of
    the program (cnctwin.sanity.scan tables) are followed per path, a tool change and the program
    end switching coolant off; a block runs with the state its own codes leave."""
    machine = si_view(machine)
    pumps = items(get(machine, "coolant", "pumps"))
    by_kind: dict[str, list[str]] = {}
    for pump in pumps:
        by_kind.setdefault(text(get(pump, "kind"), ""), []).append(text(get(pump, "id"), ""))
    fluid_kind = text(get(si_view(fluid), "kind"), "")
    on_table = COOLANT_ON.get(dialect, COOLANT_ON["fanuc"])
    off_table = COOLANT_OFF.get(dialect, COOLANT_OFF["fanuc"])
    pump_s: dict[str, float] = {}
    blast = mql = conveyor = 0.0
    commanded = False
    active: dict[int, set] = {}
    conveying: dict[int, bool] = {}
    for block in sorted(blocks, key=lambda b: (b.path, b.tick0, b.index)):
        codes = block.codes("M")
        state = active.setdefault(block.path, set())
        if block.kind in ("tool_change", "turret_index") or any(c in ("M6", "M30", "M2") for c in codes):
            state.clear()
        for code in codes:
            if code in on_table:
                state.add(code)
            elif code in off_table:
                state.difference_update(off_table[code])
            elif dialect == "haas" and code in ("M31", "M33"):
                commanded = True
                conveying[block.path] = code == "M31"
        seconds = block.duration_ticks * tick_s
        if conveying.get(block.path):
            conveyor += seconds
        if block.pumps_on is not None:
            for pump_id in block.pumps_on:
                pump_s[pump_id] = pump_s.get(pump_id, 0.0) + seconds
            continue
        needs = {on_table[c] for c in state}
        if not needs:
            continue
        if fluid_kind == "mql_oil" and needs & {"flood", "through_spindle", "mist", "shower"}:
            mql += seconds
            needs -= {"flood", "through_spindle", "mist", "shower"}
        if fluid_kind == "air" and needs & {"flood", "through_spindle", "mist", "shower"}:
            blast += seconds
            needs -= {"flood", "through_spindle", "mist", "shower"}
        for need in sorted(needs):
            if need in ("air_blast", "through_tool_air"):
                blast += seconds
                continue
            kinds = {"flood": ("flood",), "shower": ("flood",), "mist": ("mist", "mql"),
                     "through_spindle": ("through_spindle", "high_pressure", "through_tool")}.get(need, ())
            ids = next((by_kind[k] for k in kinds if by_kind.get(k)), [])
            if ids:
                pump_s[ids[0]] = pump_s.get(ids[0], 0.0) + seconds
    return CoolantStates(pump_s, blast, mql, conveyor, commanded)


# ----------------------------------------------------------------------------- helpers

ASSUMPTIONS = [
    "machine time of an operation is its longest path's span; path waits and timeline gaps are booked path_wait",
    "base load, servo standby and the machine's own compressed air run for the whole takt time (cycle and waits)",
    "the chip conveyor runs for the whole cycle when the program never commands it",
    "spindle idle power scales with speed^(5/3) from the machine file's idle power at maximum speed (Palmgren)",
    "braking energy of spindle and axes is dissipated unless the energy model's recovered fractions say otherwise",
    "coolant follows the program's M codes when the timeline gives no pumps_on; M06 and the program end stop it",
    "tool cost is price / (regrinds + 1) per edge life times the life used per part; regrind service not included",
]


def _energy_by_consumer(energy: dict) -> dict[str, float]:
    consumers = {}
    if energy.get("spindle"):
        consumers["spindle"] = energy["spindle"]["electrical_J"]
    if energy.get("axes"):
        consumers["axes"] = energy["axes"]["electrical_J"]
    consumers["servo_standby"] = energy["servo_standby_J"]
    consumers["pumps"] = sum(p["electrical_J"] for p in energy["pumps"].values())
    consumers["conveyor"] = energy["conveyor"]["electrical_J"]
    consumers["base"] = energy["base_J"]
    consumers["compressed_air"] = energy["air_J"]
    return consumers


def _total_energy(energy: dict) -> float:
    return sum(_energy_by_consumer(energy).values())


def _progression(parts: list[dict]) -> dict:
    numbers = sorted({t["number"] for p in parts for t in p["tools"]})
    tools = {}
    for number in numbers:
        rows = [next((t for t in p["tools"] if t["number"] == number), None) for p in parts]
        offsets = sorted({k for r in rows if r for k in r.get("wear_offsets_m", {})})
        tools[f"T{number}"] = {
            "life_used_frac": [r["life_used_frac"] if r else None for r in rows],
            "life_used_part_frac": [r["life_used_part_frac"] if r else None for r in rows],
            "VB_max_m": [r["VB_max_m"] if r else None for r in rows],
            "changes": [r["changes"] if r else 0 for r in rows],
            "wear_offsets_m": {k: [r["wear_offsets_m"].get(k) if r else None for r in rows] for k in offsets},
        }
    return {"part": [p["part"] for p in parts], "simulated": [p["simulated"] for p in parts],
            "machine_s": [p["machine_s"] for p in parts], "takt_s": [p["takt_s"] for p in parts],
            "energy_J": [p["energy"]["total_J"] for p in parts], "tool_cost_eur": [p["tool_cost_eur"] for p in parts],
            "tools": tools}


def _legacy_summary(first: dict, cycle: CycleTime, tick_s: float, operator: OperatorModel, summary: dict) -> dict:
    """The summary keys the first run-directory layout used (kept, same meanings)."""
    ticks = cycle.breakdown.ticks
    cut = ticks.get("cutting", 0) * tick_s
    return {"parts": summary["parts"], "cycle_time_s": first["machine_s"], "cut_time_s": cut,
            "air_time_s": first["machine_s"] - cut, "tool_change_time_s": ticks.get("tool_change", 0) * tick_s,
            "operator_s": {"load_s": operator.load_s, "unload_s": operator.unload_s, "door_s": operator.door_s,
                           "clean_s": operator.clean_s},
            "per_part_s": first["takt_s"], "removed_mass_kg": first["removed_mass_kg"],
            "energy_J": first["energy"]["total_J"]}


def _clean(value):
    """JSON-safe: NumPy scalars to Python, non-finite floats to None, keys as written."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_production(run_dir, document: dict) -> Path:
    """Write production.json (sorted keys, fixed separators: identical bytes for identical results)."""
    path = Path(run_dir) / "production.json"
    data = (json.dumps(_clean(document), sort_keys=True, indent=1) + "\n").encode("utf-8")
    tmp = path.with_name(".production.json.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return path
