"""The machine as the control sees it: axes per path, spindles, coolant devices, magazine, options.

Built from a machine definition (``db/machines/<id>.yaml`` read as a :class:`cnctwin.db.SIMap`,
SI units) plus the controller settings. Paths (channels): a machine with ``controller.two_path``
or of type ``swiss`` groups its axes by the trailing digit of their names (``X1 Z1 Y1 C1`` are
path 1's ``X Z Y C``; ``X2 Z2 ...`` path 2's); any other machine is one path whose program
letters are the axis names (the first of ``C1, C2`` is ``C``). ``physics.controller.paths``
overrides the mapping.

Positions inside the executor are integer counts of the axis quantum: the machine's least
input increment for linear axes (half of it, i.e. radius counts, for an X axis programmed in
diameter) and for rotary axes. Counts make block end points, full-circle detection and
incremental chains exact, as on the control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..db.simap import SIMap, Sourced, wrap

LATHE_TYPES = ("lathe", "mill_turn", "swiss")


def plain(value):
    """The plain value of a possibly sourced database entry."""
    if isinstance(value, Sourced):
        return value.value
    return value


def number(value, default=None):
    value = plain(value)
    if value is None:
        return default
    return float(value)


@dataclass(frozen=True)
class AxisInfo:
    letter: str                     # program address of the axis on its path
    name: str                       # machine axis name (target key of commands)
    kind: str                       # "linear" | "rotary"
    quantum: float                  # SI value of one count (m, or rad)
    diameter: bool = False          # programmed in diameter (lathe X): counts are radius half-increments
    continuous: bool = False        # rotary with roll-over (shortest path in G90)
    home: int = 0                   # counts: machine coordinate of the 1st reference point
    rapid: float = 0.0              # m/s or rad/s (0 = unknown)
    index: int = 1                  # 1-based axis number for system variables (#5001 + index - 1)
    travel: tuple[float, float] | None = None   # SI machine coordinates


@dataclass(frozen=True)
class SpindleInfo:
    id: str
    kind: str                       # milling | turning | live_tool | sub
    max_rad_s: float
    gears: int = 1


@dataclass(frozen=True)
class RotaryLink:
    axis: str
    point_m: tuple[float, float, float]      # axis line point, machine coordinates at zero
    direction: tuple[float, float, float]    # unit vector, right-handed for +command


@dataclass
class MachineModel:
    id: str
    type: str
    variant: str                    # "M" (milling G codes) | "T" (turning, G code system A)
    paths: list[list[AxisInfo]]
    path_spindle: list[str]
    path_live_spindle: list[str | None]
    spindles: dict[str, SpindleInfo]
    pumps: dict[str, str]           # pump id -> kind
    nozzles: dict[str, dict]        # programmable nozzles: id -> {"positions": n, "code": "M34/M35"}
    air_blast: bool
    tool_capacity: int | None
    tool_change_s: float
    turrets: list[dict]
    pallets: int
    door_auto: bool
    conveyor: bool
    spindle_probe: bool
    tool_setter: dict | None
    smoothing: list[dict]
    corner_default: str
    lookahead_blocks: int
    block_processing_s: float
    ipo_s: float
    lin_increment_m: float
    rot_increment_rad: float
    max_feed_m_s: float
    diameter_programming: bool
    work_rotaries: list[RotaryLink] = field(default_factory=list)
    tool_rotaries: list[str] = field(default_factory=list)
    tool_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    raw: SIMap | None = None

    def axis(self, path: int, letter: str) -> AxisInfo | None:
        for info in self.paths[path]:
            if info.letter == letter:
                return info
        return None

    # ------------------------------------------------------------------ table rotary kinematics
    def table_transform(self, rotary_values: dict[str, float]) -> np.ndarray:
        """4x4 motion of a table-fixed point (machine coordinates) for the given rotary values (rad):
        p = T_1(v1) T_2(v2) ... p0, axes defined at zero (product of exponentials)."""
        total = np.eye(4)
        for link in self.work_rotaries:
            angle = rotary_values.get(link.axis, 0.0)
            total = total @ _axis_rotation(np.array(link.point_m), np.array(link.direction), angle)
        return total

    def table_kinematics_supported(self) -> bool:
        return bool(self.work_rotaries) and not self.tool_rotaries


def _axis_rotation(point: np.ndarray, direction: np.ndarray, angle: float) -> np.ndarray:
    axis = direction / np.linalg.norm(direction)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    one_c = 1.0 - c
    rotation = np.array([
        [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
        [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
        [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c]])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = point - rotation @ point
    return transform


def _rapid(controller, axes, name: str) -> float:
    rapid = controller.rapid_m_s if controller is not None else None
    if isinstance(rapid, SIMap):
        value = rapid.get(name)
        if value is not None:
            return float(plain(value))
    elif rapid is not None:
        return float(plain(rapid))
    axis = axes.get(name) if axes is not None else None
    if axis is not None and axis.rapid_m_s is not None:
        return float(plain(axis.rapid_m_s))
    if axis is not None and axis.max_rad_s is not None:
        return float(plain(axis.max_rad_s))
    return 0.0


def _order(names: list[str], lathe: bool) -> list[str]:
    first = ["X", "Z"] if lathe else ["X", "Y", "Z"]
    head = [n for n in first if n in names]
    return head + [n for n in names if n not in head]


# Keys a database mapping may carry beside its entries (docs/formats.md "Sources and estimates"):
# they describe the mapping, they are not axes.
META_KEYS = frozenset(("src", "note", "notes", "page", "pages", "est", "estimated", "accessed"))


def axis_names(axes) -> list[str]:
    """Names of the axes of a machine's ``axes`` mapping: the entries that are axis definitions
    (mappings), without the source keys a sourced mapping carries (``src``, ``note`` ...)."""
    if axes is None:
        return []
    names = []
    for name in axes.keys():
        if str(name) in META_KEYS:
            continue
        entry = axes[name]
        if isinstance(entry, SIMap) or isinstance(entry, dict):
            names.append(str(name))
    return names


def build_machine(machine, settings=None) -> MachineModel:
    """MachineModel from a machine definition (SIMap or plain mapping)."""
    m = machine if isinstance(machine, SIMap) else wrap(machine)
    controller = m.controller
    mtype = str(plain(m.type))
    lathe = mtype in LATHE_TYPES
    lin_inc = number(controller.least_input_increment_m, 1e-6)
    rot_inc = number(controller.least_input_increment_rad, math.radians(0.001))
    diameter = bool(plain(controller.diameter_programming_X)) if controller.diameter_programming_X is not None \
        else lathe
    axes = m.axes
    names = axis_names(axes)
    two_path = bool(plain(controller.two_path)) if controller.two_path is not None else False
    two_path = two_path or mtype == "swiss"
    overrides = getattr(settings, "paths", None) if settings is not None else None
    path_letters: list[dict[str, str]] = []
    if overrides:
        path_letters = [dict(p) for p in overrides]
    elif two_path:
        groups: dict[int, dict[str, str]] = {}
        for name in names:
            digit = name[-1] if name[-1].isdigit() else "1"
            groups.setdefault(int(digit), {})[name[0] if name[-1].isdigit() else name] = name
        path_letters = [groups[k] for k in sorted(groups)]
    else:
        single: dict[str, str] = {}
        for name in names:
            letter = name[0] if (len(name) > 1 and name[-1].isdigit()) else name
            if letter not in single:
                single[letter] = name
        path_letters = [single]
    home = m.kinematics.home_m if m.kinematics is not None else None
    paths: list[list[AxisInfo]] = []
    for letters in path_letters:
        infos: list[AxisInfo] = []
        ordered = _order(list(letters.keys()), lathe)
        for index, letter in enumerate(ordered, start=1):
            name = letters[letter]
            axis = axes[name]
            kind = str(plain(axis.type))
            is_diameter = diameter and lathe and letter == "X" and kind == "linear"
            quantum = (lin_inc / 2.0 if is_diameter else lin_inc) if kind == "linear" else rot_inc
            home_si = float(plain(home.get(name, 0.0))) if isinstance(home, SIMap) else 0.0
            travel = axis.travel_m if kind == "linear" else axis.travel_rad
            travel = plain(travel)
            infos.append(AxisInfo(letter, name, kind, quantum, is_diameter,
                                  bool(plain(axis.continuous)) if axis.continuous is not None else False,
                                  int(round(home_si / quantum)), _rapid(controller, axes, name), index,
                                  (float(travel[0]), float(travel[1])) if travel else None))
        paths.append(infos)
    spindles: dict[str, SpindleInfo] = {}
    for spindle in m.spindles or []:
        gears = spindle.gear_ranges or []
        spindles[str(plain(spindle.id))] = SpindleInfo(str(plain(spindle.id)), str(plain(spindle.kind)),
                                                       number(spindle.max_rad_s, 0.0), max(1, len(gears)))
    main_kinds = ("turning",) if lathe else ("milling",)
    ordered_spindles = list(spindles.values())
    main = next((s.id for s in ordered_spindles if s.kind in main_kinds), ordered_spindles[0].id
                if ordered_spindles else "main")
    sub = next((s.id for s in ordered_spindles if s.kind == "sub"), main)
    live = next((s.id for s in ordered_spindles if s.kind == "live_tool"), None)
    if lathe and main not in spindles and ordered_spindles:
        main = ordered_spindles[0].id
    path_spindle = [main] + [sub] * (len(paths) - 1)
    path_live = [live] * len(paths)
    coolant = m.coolant
    pumps: dict[str, str] = {}
    nozzles: dict[str, dict] = {}
    air = False
    if coolant is not None:
        for pump in coolant.pumps or []:
            pumps[str(plain(pump.id))] = str(plain(pump.kind))
        for nozzle in coolant.nozzles or []:
            programmable = nozzle.programmable
            if programmable is not None:
                nozzles[str(plain(nozzle.id))] = {"positions": int(plain(programmable.positions) or 0),
                                                  "code": str(plain(programmable.code) or "")}
        air = coolant.air_blast is not None
    changer = m.tool_changer
    capacity = None
    change_s = 0.0
    if changer is not None:
        capacity = int(plain(changer.capacity)) if plain(changer.capacity) is not None else None
        change_s = number(changer.chip_to_chip_s, None) or number(changer.tool_to_tool_s, 0.0) or 0.0
    turrets = []
    for turret in m.turrets or []:
        turrets.append({"id": str(plain(turret.id)), "stations": int(plain(turret.stations) or 0),
                        "index_s": number(turret.index_time_s.max if isinstance(turret.index_time_s, SIMap)
                                          else turret.index_time_s, 0.0) or 0.0})
    pallet = m.pallet_changer
    pallets = int(plain(pallet.pallets) or 0) if pallet is not None else 0
    doors = m.doors
    door_auto = bool(plain(doors.auto)) if doors is not None and doors.auto is not None else False
    conveyor_def = m.chip_conveyor
    conveyor = conveyor_def is not None and str(plain(conveyor_def.type)) not in ("none", "None", "")
    probes = m.probes
    spindle_probe = probes is not None and probes.spindle_probe is not None
    setter = None
    if probes is not None and probes.tool_setter is not None:
        position = probes.tool_setter.position_m
        setter = {"position_m": [float(v) for v in plain(position)] if position is not None else None}
    smoothing = []
    for entry in controller.smoothing or []:
        smoothing.append({"code": str(plain(entry.code)), "name": str(plain(entry.name) or ""),
                          "tolerance_m": number(entry.tolerance_m, 0.0)})
    kin = m.kinematics
    work_rotaries: list[RotaryLink] = []
    tool_rotaries: list[str] = []
    tool_axis = (0.0, 0.0, 1.0)
    if kin is not None:
        nose = np.zeros(3)
        for link in kin.tool_chain or []:
            if link.axis is not None:
                axis_def = axes.get(str(plain(link.axis)))
                if axis_def is not None and str(plain(axis_def.type)) == "rotary":
                    tool_rotaries.append(str(plain(link.axis)))
            elif link.fixed is not None and link.fixed.translate_m is not None:
                nose = nose + np.array([float(v) for v in plain(link.fixed.translate_m)])
        pose = np.zeros(3)
        for link in kin.workpiece_chain or []:
            if link.axis is not None:
                axis_def = axes.get(str(plain(link.axis)))
                if axis_def is not None and str(plain(axis_def.type)) == "rotary":
                    point = np.array([float(v) for v in plain(link.axis_point_m)]) if link.axis_point_m is not None \
                        else np.zeros(3)
                    direction = np.array([float(v) for v in plain(link.direction)])
                    work_rotaries.append(RotaryLink(str(plain(link.axis)), tuple(float(v) for v in pose + point - nose),
                                                    tuple(float(v) for v in direction / np.linalg.norm(direction))))
            elif link.fixed is not None and link.fixed.translate_m is not None:
                pose = pose + np.array([float(v) for v in plain(link.fixed.translate_m)])
        if kin.tool_axis is not None:
            tool_axis = tuple(float(v) for v in plain(kin.tool_axis))
    return MachineModel(
        id=str(plain(m.id) or ""), type=mtype, variant="T" if lathe else "M", paths=paths,
        path_spindle=path_spindle, path_live_spindle=path_live, spindles=spindles, pumps=pumps, nozzles=nozzles,
        air_blast=air, tool_capacity=capacity, tool_change_s=float(change_s), turrets=turrets, pallets=pallets,
        door_auto=door_auto, conveyor=conveyor, spindle_probe=spindle_probe, tool_setter=setter,
        smoothing=smoothing, corner_default=str(plain(controller.corner_default) or "blend"),
        lookahead_blocks=int(plain(controller.lookahead_blocks) or 1),
        block_processing_s=number(controller.block_processing_s, 0.0),
        ipo_s=number(controller.ipo_period_s, 1e-3), lin_increment_m=lin_inc, rot_increment_rad=rot_inc,
        max_feed_m_s=number(controller.max_feed_m_s, 0.0) or 0.0, diameter_programming=diameter and lathe,
        work_rotaries=work_rotaries, tool_rotaries=tool_rotaries, tool_axis=tool_axis, raw=m)
