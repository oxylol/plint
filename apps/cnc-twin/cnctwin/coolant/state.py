"""The coolant domain's SimState field (``state.coolant``), checkpointable as arrays + JSON metadata.

Everything the next macro step depends on is here: the tank (temperature, volume, concentrate,
energy and its totals), the applied and the pending occlusion results (a job in flight survives a
checkpoint), the bookkeeping cursors of the chips record, consumption totals per operation and part,
finding episode flags and the trace ids the cause chains point to. The supply operating point is a
pure function of (configuration, temperature bucket) and is recomputed after a resume.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from .occlusion import OcclusionResult
from .tank import Tank

FORMAT_VERSION = 1


def _totals() -> dict:
    return {k: 0.0 for k in ("pump_energy_J", "thermal_heat_J", "chips_heat_J", "air_m3", "mql_oil_m3", "mist_m3",
                             "drag_out_m3", "pumped_m3", "on_s", "chip_mass_kg")}


@dataclass
class CoolantState:
    tank: Tank
    macro: int = 0
    applied: OcclusionResult | None = None
    pending: OcclusionResult | None = None
    chips_booked_tick: int = -2
    chips_emitted_kg: float = 0.0
    totals: dict = field(default_factory=_totals)
    per_op: list = field(default_factory=list)          # [{op, ...totals}] closed at end_operation
    op_start: dict = field(default_factory=_totals)     # totals at the start of the current operation
    per_part: dict = field(default_factory=dict)        # part index -> totals at its end
    part_index: int = 0
    op_index: int = 0
    flags: dict = field(default_factory=dict)           # finding episodes ("shadow:<slot>", "holes:<pump>:<T>")
    trace_ids: dict = field(default_factory=dict)       # "supply", "jets", "occlusion:<key>" -> trace id
    config_key: str = ""
    nozzle_positions: dict = field(default_factory=dict)
    tank_min_K: float = math.inf
    tank_max_K: float = -math.inf
    worst_reach: dict = field(default_factory=dict)     # tool slot -> lowest reach seen while cutting
    notes: list = field(default_factory=list)           # fallbacks and estimates reported at the end

    # ------------------------------------------------------------------ checkpoint
    def to_arrays(self) -> tuple[dict, dict]:
        arrays: dict[str, np.ndarray] = {}
        meta = {"format_version": FORMAT_VERSION, "tank": self.tank.to_dict(), "macro": self.macro,
                "chips_booked_tick": self.chips_booked_tick, "chips_emitted_kg": self.chips_emitted_kg,
                "totals": dict(self.totals), "per_op": list(self.per_op), "op_start": dict(self.op_start),
                "per_part": {str(k): v for k, v in sorted(self.per_part.items())}, "part_index": self.part_index,
                "op_index": self.op_index, "flags": dict(sorted(self.flags.items())),
                "trace_ids": dict(sorted(self.trace_ids.items())), "config_key": self.config_key,
                "nozzle_positions": dict(sorted(self.nozzle_positions.items())), "tank_min_K": self.tank_min_K,
                "tank_max_K": self.tank_max_K, "worst_reach": {str(k): v for k, v in sorted(self.worst_reach.items())},
                "notes": list(self.notes)}
        for name in ("applied", "pending"):
            result = getattr(self, name)
            if result is None:
                meta[name] = None
                continue
            a, m = result.to_arrays(name + ".")
            arrays.update(a)
            meta[name] = m
        # JSON round trip now, so a checkpoint can never hold what from_arrays cannot read back
        meta = json.loads(json.dumps(meta))
        return arrays, meta

    @classmethod
    def from_arrays(cls, arrays: dict, meta: dict) -> "CoolantState":
        if int(meta.get("format_version", 0)) != FORMAT_VERSION:
            from ..core.errors import CheckpointError
            raise CheckpointError(f"coolant state format {meta.get('format_version')} is not {FORMAT_VERSION}")
        out = cls(tank=Tank.from_dict(meta["tank"]), macro=int(meta["macro"]),
                  chips_booked_tick=int(meta["chips_booked_tick"]), chips_emitted_kg=float(meta["chips_emitted_kg"]),
                  totals={k: float(v) for k, v in meta["totals"].items()}, per_op=list(meta["per_op"]),
                  op_start={k: float(v) for k, v in meta["op_start"].items()},
                  per_part={int(k): v for k, v in meta["per_part"].items()}, part_index=int(meta["part_index"]),
                  op_index=int(meta["op_index"]), flags={k: int(v) for k, v in meta["flags"].items()},
                  trace_ids={k: int(v) for k, v in meta["trace_ids"].items()}, config_key=str(meta["config_key"]),
                  nozzle_positions={k: int(v) for k, v in meta["nozzle_positions"].items()},
                  tank_min_K=float(meta["tank_min_K"]), tank_max_K=float(meta["tank_max_K"]),
                  worst_reach={int(k): float(v) for k, v in meta["worst_reach"].items()}, notes=list(meta["notes"]))
        for name in ("applied", "pending"):
            if meta.get(name) is not None:
                setattr(out, name, OcclusionResult.from_arrays(arrays, meta[name], name + "."))
        return out
