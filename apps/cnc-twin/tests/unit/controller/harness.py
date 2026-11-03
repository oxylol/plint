"""Inline fixtures for the controller tests: small machine definitions, controller alarm tables, runners."""

from __future__ import annotations

import copy

from cnctwin.controller import Controller, ControllerSetup, ProgramSource, ToolOffsetInit, dry_run
from cnctwin.controller.ir import Kind
from cnctwin.controller.schema import ControllerSettings

MILL = {
    "format": "cnctwin-machine", "version": 1, "id": "test-mill", "maker": "T", "model": "T", "type": "vertical_mill",
    "controller": {"dialect": "fanuc", "model": "test", "alarms": "fixture", "ipo_period_ms": 1.0,
                   "block_processing_ms": 1.0, "lookahead_blocks": 20, "least_input_increment_mm": 0.001,
                   "least_input_increment_deg": 0.001, "max_feed_mm_min": 20000,
                   "rapid_mm_min": {"X": 30000, "Y": 30000, "Z": 30000},
                   "smoothing": [{"code": "G05.1 Q1", "name": "ai", "tolerance_mm": 0.01}]},
    "axes": {"X": {"type": "linear", "travel_mm": [-800, 0]}, "Y": {"type": "linear", "travel_mm": [-500, 0]},
             "Z": {"type": "linear", "travel_mm": [-500, 0]}},
    "kinematics": {"workpiece_chain": [{"link": "table", "axis": "X", "direction": [-1, 0, 0]}],
                   "tool_chain": [{"link": "head", "axis": "Z", "direction": [0, 0, 1]}],
                   "home_mm": {"X": 0, "Y": 0, "Z": 0}, "tool_axis": [0, 0, 1]},
    "spindles": [{"id": "main", "kind": "milling", "max_rpm": 12000, "gear_ranges": [{"rpm": [0, 12000]}]}],
    "tool_changer": {"type": "side_mount", "capacity": 24, "tool_to_tool_s": 2.0},
    "coolant": {"pumps": [{"id": "flood", "kind": "flood"}, {"id": "tsc", "kind": "through_spindle"}],
                "nozzles": [{"id": "pcool", "kind": "flood", "programmable": {"code": "M34/M35", "positions": 34}}],
                "air_blast": {"flow_l_min": 170}},
    "chip_conveyor": {"type": "auger"}, "doors": {"auto": True}, "pallet_changer": None,
}

HAAS_MILL = copy.deepcopy(MILL)
HAAS_MILL["controller"]["dialect"] = "haas"
HAAS_MILL["controller"]["smoothing"] = [{"code": "G187 P1", "name": "rough", "tolerance_mm": 0.025},
                                        {"code": "G187 P2", "name": "medium", "tolerance_mm": 0.010},
                                        {"code": "G187 P3", "name": "finish", "tolerance_mm": 0.0025}]

UMC = copy.deepcopy(HAAS_MILL)
UMC["id"] = "test-umc"
UMC["type"] = "five_axis_mill"
UMC["axes"]["B"] = {"type": "rotary", "travel_deg": [-35, 120], "continuous": False}
UMC["axes"]["C"] = {"type": "rotary", "travel_deg": [0, 360], "continuous": True}
UMC["controller"]["rapid_mm_min"] = {"X": 30000, "Y": 30000, "Z": 30000}
UMC["kinematics"] = {
    "workpiece_chain": [{"link": "trunnion", "axis": "B", "axis_point_mm": [0, 0, -50], "direction": [0, -1, 0]},
                        {"link": "platter", "axis": "C", "axis_point_mm": [0, 0, 0], "direction": [0, 0, -1]},
                        {"link": "top", "fixed": {"translate_mm": [0, 0, 0]}}],
    "tool_chain": [{"link": "column", "axis": "X", "direction": [1, 0, 0]},
                   {"link": "ram", "axis": "Y", "direction": [0, 1, 0]},
                   {"link": "head", "axis": "Z", "direction": [0, 0, 1]},
                   {"link": "nose", "fixed": {"translate_mm": [300, 250, 600]}}],
    "home_mm": {"X": 0, "Y": 0, "Z": 0, "B": 0, "C": 0}, "tool_axis": [0, 0, 1]}

LATHE = {
    "format": "cnctwin-machine", "version": 1, "id": "test-lathe", "maker": "T", "model": "T", "type": "lathe",
    "controller": {"dialect": "fanuc", "model": "test", "alarms": "fixture", "ipo_period_ms": 1.0,
                   "block_processing_ms": 1.0, "lookahead_blocks": 20, "least_input_increment_mm": 0.001,
                   "least_input_increment_deg": 0.001, "max_feed_mm_min": 12000,
                   "rapid_mm_min": {"X": 12000, "Z": 24000}, "diameter_programming_X": True},
    "axes": {"X": {"type": "linear", "travel_mm": [-200, 0]}, "Z": {"type": "linear", "travel_mm": [-500, 0]}},
    "kinematics": {"workpiece_chain": [{"link": "spindle", "axis": "C", "direction": [0, 0, -1]}],
                   "tool_chain": [{"link": "carriage", "axis": "Z", "direction": [0, 0, 1]},
                                  {"link": "slide", "axis": "X", "direction": [1, 0, 0]}],
                   "home_mm": {"X": 0, "Z": 0}, "tool_axis": [0, 0, 1]},
    "spindles": [{"id": "main", "kind": "turning", "max_rpm": 4000, "gear_ranges": [{"rpm": [0, 4000]}]}],
    "turrets": [{"id": "t1", "stations": 12}],
    "coolant": {"pumps": [{"id": "flood", "kind": "flood"}]},
}

SWISS = {
    "format": "cnctwin-machine", "version": 1, "id": "test-swiss", "maker": "T", "model": "T", "type": "swiss",
    "controller": {"dialect": "fanuc", "model": "test", "alarms": "fixture", "ipo_period_ms": 1.0,
                   "block_processing_ms": 1.0, "lookahead_blocks": 20, "least_input_increment_mm": 0.001,
                   "least_input_increment_deg": 0.001, "max_feed_mm_min": 20000,
                   "rapid_mm_min": {"X1": 30000, "Z1": 30000, "X2": 30000, "Z2": 30000},
                   "diameter_programming_X": True},
    "axes": {"X1": {"type": "linear", "travel_mm": [-50, 150]}, "Z1": {"type": "linear", "travel_mm": [0, 300]},
             "X2": {"type": "linear", "travel_mm": [-50, 150]}, "Z2": {"type": "linear", "travel_mm": [0, 300]}},
    "kinematics": {"workpiece_chain": [{"link": "h", "axis": "Z1", "direction": [0, 0, 1]}],
                   "tool_chain": [{"link": "g", "axis": "X1", "direction": [1, 0, 0]}],
                   "home_mm": {"X1": 0, "Z1": 0, "X2": 0, "Z2": 0}, "tool_axis": [1, 0, 0]},
    "spindles": [{"id": "main", "kind": "turning", "max_rpm": 10000}, {"id": "sub", "kind": "sub", "max_rpm": 8000}],
    "coolant": {"pumps": [{"id": "flood", "kind": "flood"}]},
}

FANUC_DB = {
    "format": "cnctwin-controller", "version": 1, "id": "fixture", "dialect": "fanuc", "name": "fixture",
    "alarms": [{"code": "PS0010", "text": "IMPROPER G-CODE"}, {"code": "PS0009", "text": "IMPROPER NC-ADDRESS"},
               {"code": "PS0005", "text": "NO DATA AFTER ADDRESS"}, {"code": "PS0011", "text": "FEED ZERO ( COMMAND )"},
               {"code": "PS0020", "text": "OVER TOLERANCE OF RADIUS"},
               {"code": "TW-M", "text": "M-CODE ERROR (MACHINE BUILDER)", "kind": "undefined_m_code"}],
}

HAAS_DB = {
    "format": "cnctwin-controller", "version": 1, "id": "fixture", "dialect": "haas", "name": "fixture",
    "alarms": [{"code": "310", "text": "Invalid G Code"}, {"code": "339", "text": "Multiple Codes"},
               {"code": "370", "text": "Pocket Definition Error"}, {"code": "312", "text": "Program End"}],
}


def controller(program: str, machine: dict = MILL, *, db: dict | None = None, settings: ControllerSettings | None = None,
               offsets: dict | None = None, tools: list | None = None, extra: list[tuple[str, str]] | None = None,
               programs: list[str] | None = None) -> Controller:
    if db is None:
        db = HAAS_DB if machine["controller"]["dialect"] == "haas" else FANUC_DB
    if programs is not None:
        sources = [ProgramSource(f"p{k}.nc", text, path=k) for k, text in enumerate(programs)]
    else:
        sources = [ProgramSource("test.nc", program)]
    for name, text in extra or []:
        sources.append(ProgramSource(name, text, path=-1, main=False))
    work = [offsets or {}] + [{} for _ in range(4)]
    setup = ControllerSetup(machine, sources, settings or ControllerSettings(), controller_db=db, db_id="fixture",
                            work_offsets_m=work, tools=tools or [ToolOffsetInit(1, length_m=0.1, radius_m=0.005)])
    return Controller(setup)


def run(program: str, machine: dict = MILL, **kwargs):
    probe = kwargs.pop("probe", None)
    setter = kwargs.pop("setter", None)
    parts = kwargs.pop("parts", 1)
    ctl = controller(program, machine, **kwargs)
    return dry_run(ctl, probe=probe, setter=setter, parts=parts)


MOTION = (Kind.RAPID, Kind.LINEAR, Kind.ARC, Kind.PROBE, Kind.TOOL_SETTER)


def moves(result, path: int = 0):
    return [c for c in result.commands[path] if c.kind in MOTION]


def mm(cmd, axis: str) -> float:
    return round(cmd.target[axis] * 1000.0, 6)


def xyz(cmd) -> tuple:
    return tuple(round(cmd.target[a] * 1000.0, 6) for a in ("X", "Y", "Z") if a in cmd.target)


def alarm(result, path: int = 0) -> dict | None:
    return result.alarms[path]


def wrap(body: str) -> str:
    return "%\nO0001\n" + body.strip() + "\nM30\n%\n"
