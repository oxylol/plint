"""Production metrics: cycle time, takt, batches, energy, coolant and air, tool cost per part
(docs/models/production.md).

The loop calls :class:`ProductionAggregator.add_part` after each part with the part's executed
blocks, signals and tool states, then :meth:`ProductionAggregator.write` for production.json; after a
run, :func:`production_from_run_dir` composes the same document from the run directory. The pieces
are pure functions: :func:`cycle_time` (exact breakdown in ticks per block, line, tool, operation),
:class:`OperatorModel` (takt), :func:`spindle_energy`, :func:`axis_energy`, :func:`pump_energy`,
:func:`coolant_use`, :func:`air_use`, :func:`tool_cost`.
"""

from .batch import (CoolantStates, EnergyInputs, PartInputs, ProductionAggregator, ToolState, coolant_states,
                    write_production)
from .consumables import air_use, coolant_use
from .cycle import Breakdown, CycleTime, block_breakdown, cycle_time, cycle_to_dict, lines_to_dict
from .energy import (AxisModel, SpindleModel, air_specific_energy_J_m3, axis_energy, axis_models, base_energy,
                     conveyor_power_W, process_feed_energy, pump_energy, spindle_energy, supply_pressure_Pa)
from .operator import OperatorModel, OperatorTime
from .rundir import energy_inputs, production_from_run_dir, read_frames, read_series_column, tool_states
from .timeline import CATEGORIES, Timeline, TimelineBlock, block_from_record, categorise, read_timeline, with_tools
from .toolcost import tool_cost

__all__ = [
    "ProductionAggregator", "PartInputs", "EnergyInputs", "ToolState", "CoolantStates", "coolant_states",
    "write_production", "production_from_run_dir", "read_frames", "read_series_column", "energy_inputs",
    "tool_states", "cycle_time", "block_breakdown", "cycle_to_dict", "lines_to_dict", "Breakdown", "CycleTime", "OperatorModel",
    "OperatorTime", "SpindleModel", "AxisModel", "axis_models", "spindle_energy", "axis_energy", "pump_energy",
    "conveyor_power_W", "base_energy", "air_specific_energy_J_m3", "supply_pressure_Pa", "process_feed_energy",
    "coolant_use", "air_use", "tool_cost", "CATEGORIES", "Timeline", "TimelineBlock", "block_from_record",
    "categorise", "read_timeline", "with_tools",
]
