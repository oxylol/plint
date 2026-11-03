"""Helpers of the production tests (unique module name: the test folders are not packages)."""

import copy

from cnctwin.production import TimelineBlock, block_from_record

MACHINE = {
    "id": "fixture-vmc", "type": "vertical_mill",
    "controller": {"dialect": "haas"},
    "axes": {"X": {"type": "linear", "travel_mm": [-762, 0], "moving_mass_kg": 300},
             "Y": {"type": "linear", "travel_mm": [-406, 0], "moving_mass_kg": 650},
             "Z": {"type": "linear", "travel_mm": [-508, 0], "moving_mass_kg": 400}},
    "kinematics": {"workpiece_chain": [{"link": "saddle", "axis": "Y", "direction": [0, -1, 0]},
                                       {"link": "table", "axis": "X", "direction": [-1, 0, 0]}],
                   "tool_chain": [{"link": "head", "axis": "Z", "direction": [0, 0, 1]}]},
    "spindles": [{"id": "main", "max_rpm": 12000, "rotor_inertia_kgm2": 0.05, "efficiency": 0.85,
                  "idle_power_W": 700}],
    "coolant": {"tank_l": 208,
                "pumps": [{"id": "flood", "kind": "flood", "flow_l_min": 45, "power_kW": 0.75},
                          {"id": "tsc", "kind": "through_spindle", "flow_l_min": 11, "power_kW": 1.1}],
                "air_blast": {"flow_l_min": 170, "pressure_bar": 6.9}},
    "chip_conveyor": {"type": "auger"},
    "doors": {"count": 2, "auto": False},
    "power": {"base_W": 900, "servo_standby_W": 300, "air_consumption_l_min": 113},
}
FLUID = {"id": "emulsion-8pct", "kind": "emulsion", "concentration_pct": 8, "drag_out_ml_per_kg_chips": 42,
         "evaporation_l_per_h_per_m2": 0.3, "mist_loss_fraction": 2.0e-6, "cost_eur_per_l": 4.0}


def machine(**changes) -> dict:
    data = copy.deepcopy(MACHINE)
    data.update(changes)
    return data


def block(index, tick0, tick1, kind="linear", line=None, text="G01 X1.", **extra) -> TimelineBlock:
    record = {"index": index, "op": extra.pop("op", "OP10"), "program": "p.nc",
              "line": index + 1 if line is None else line, "block": text, "path": extra.pop("path", 0),
              "kind": kind, "tick0": tick0, "tick1": tick1}
    record.update(extra)
    return block_from_record(record)
