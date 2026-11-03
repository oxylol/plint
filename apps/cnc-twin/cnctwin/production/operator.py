"""Operator model: what a person does per part, and how long the machine waits for it (takt time).

Per part and operation the operator unloads the finished part, cleans the fixture, loads the next
blank and moves the door twice (open, close); at every M00, and at every M01 when the optional stop
is on, the door is opened, the stop action done and the door closed; every ``inspect_every``-th part
(and the first, when first-article inspection is on) is inspected; a worn tool is replaced. Without
a pallet changer the machine waits for all of it. With two or more pallets, loading, unloading and
cleaning happen while the machine cuts the other pallet: the machine waits for the pallet change,
and for the operator only when the handling takes longer than the machine time.

``takt_s`` = machine time + machine waiting time: the time between finished parts the cell keeps up.
Values come from ``production.operator`` of the project (template keys ``load_s, unload_s, door_s,
clean_s, inspect_every, inspect_s``; additive keys ``stop_s, optional_stop, inspect_parallel,
first_article_inspection, tool_replace_s``), else from db/models/production-operator.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..db.models import params as model_params
from ..sanity.si import get, num, plain, si_view

FIELDS = ("load_s", "unload_s", "door_s", "clean_s", "inspect_every", "inspect_s", "stop_s", "optional_stop",
          "inspect_parallel", "first_article_inspection", "tool_replace_s")


@dataclass
class OperatorModel:
    load_s: float
    unload_s: float
    door_s: float
    clean_s: float
    inspect_every: int
    inspect_s: float
    stop_s: float
    optional_stop: bool
    inspect_parallel: bool
    first_article_inspection: bool
    tool_replace_s: float
    auto_door_s: float | None = None
    pallets: int = 1
    pallet_change_s: float = 0.0
    sources: dict[str, str] = field(default_factory=dict)      # field -> "project" | "default"

    @classmethod
    def from_project(cls, project, machine=None, params=None) -> "OperatorModel":
        defaults = params or model_params("production-operator")
        operator = si_view(get(si_view(project), "production", "operator"))
        values, sources = {}, {}
        for name in FIELDS:
            given = plain(get(operator, name))
            if given is None:
                values[name] = defaults[name]
                sources[name] = "default"
            else:
                values[name] = given
                sources[name] = "project"
        machine = si_view(machine)
        doors = get(machine, "doors")
        auto = bool(plain(get(doors, "auto")))
        open_s = num(get(doors, "open_time_s"))
        pallet = get(machine, "pallet_changer")
        pallets = int(num(get(pallet, "pallets"), 1) or 1)
        return cls(load_s=float(values["load_s"]), unload_s=float(values["unload_s"]),
                   door_s=float(values["door_s"]), clean_s=float(values["clean_s"]),
                   inspect_every=int(values["inspect_every"]), inspect_s=float(values["inspect_s"]),
                   stop_s=float(values["stop_s"]), optional_stop=bool(values["optional_stop"]),
                   inspect_parallel=bool(values["inspect_parallel"]),
                   first_article_inspection=bool(values["first_article_inspection"]),
                   tool_replace_s=float(values["tool_replace_s"]),
                   auto_door_s=open_s if auto and open_s is not None else None,
                   pallets=pallets, pallet_change_s=float(num(get(pallet, "change_time_s"), 0.0) or 0.0),
                   sources=sources)

    @property
    def door_move_s(self) -> float:
        """One door movement: the machine's automatic door, else the operator's."""
        return self.auto_door_s if self.auto_door_s is not None else self.door_s

    def inspected(self, part_index: int) -> bool:
        if self.inspect_every <= 0:
            return False
        if part_index == 0 and self.first_article_inspection:
            return True
        return (part_index + 1) % self.inspect_every == 0

    def part_time(self, *, operations: int, stops: dict[str, int], part_index: int, machine_time_s: float,
                  tool_replacements: int = 0) -> "OperatorTime":
        door = self.door_move_s
        handling = operations * (self.load_s + self.unload_s + self.clean_s + 2.0 * door)
        m00 = int(stops.get("M00", 0))
        m01 = int(stops.get("M01", 0)) if self.optional_stop else 0
        stop_time = (m00 + m01) * (self.stop_s + 2.0 * door)
        inspection = self.inspect_s if self.inspected(part_index) else 0.0
        replace = tool_replacements * self.tool_replace_s
        if self.pallets >= 2:
            pallet_wait = operations * self.pallet_change_s
            handling_wait = max(0.0, handling - machine_time_s)
        else:
            pallet_wait = 0.0
            handling_wait = handling
        inspection_wait = max(0.0, inspection - machine_time_s) if self.inspect_parallel else inspection
        idle = pallet_wait + handling_wait + stop_time + inspection_wait + replace
        components = {"handling_s": handling, "stops_s": stop_time, "inspection_s": inspection,
                      "tool_replacement_s": replace, "pallet_change_s": pallet_wait}
        return OperatorTime(components, handling + stop_time + inspection + replace, idle, machine_time_s + idle,
                            m00, m01, inspection > 0.0)


@dataclass
class OperatorTime:
    components: dict[str, float]
    operator_s: float                # time the operator is busy with this part
    machine_wait_s: float            # time the machine stands still waiting for the operator
    takt_s: float                    # machine time + machine waiting time
    m00: int
    m01: int
    inspected: bool

    def to_dict(self) -> dict:
        return {"components_s": dict(self.components), "operator_s": self.operator_s,
                "machine_wait_s": self.machine_wait_s, "takt_s": self.takt_s, "m00": self.m00, "m01": self.m01,
                "inspected": self.inspected}
