"""Controller state: everything the executor knows, as plain data (architecture 6).

The executor is an explicit state machine: a program counter per call level (``Frame.pc``), the
call stack (``frames``), the loop stack of each level (``Frame.loops``), local variables per macro
level, common variables, modal state, positions in integer counts, offsets, the look-ahead buffer
(``buffer``: commands produced and not yet retired by motion), the pending cycle expansion
(``pending``: :class:`~cnctwin.controller.hooks.Step` data), the cutter-compensation queue and its
filter data, what the path is waiting for, and the block-time clock. Nothing lives in Python
generators, closures or the call stack between blocks, so :meth:`ControllerState.to_arrays` is
complete and a run resumed with :meth:`ControllerState.from_arrays` continues bit-identically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from . import serial
from .hooks import Step
from .ir import Command, OperatorPanel

STATE_VERSION = 1
N_LOCALS = 33


@dataclass
class LoopEntry:
    m: int                          # DO identification number (1..3)
    do_pc: int                      # block index of the WHILE ... DO m / DO m block
    end_pc: int = -1                # block index of the matching END m (known after the first pass)


@dataclass
class Frame:
    kind: str                       # "main" | "sub" (M98) | "local" (M97) | "macro" (G65, G/M-code call) | "modal" (G66)
    program: int                    # index into the ProgramStore
    pc: int                         # index of the block being (or about to be) executed
    level: int                      # local-variable level (index into PathState.locals)
    repeat_left: int = 0            # further repetitions after the current one
    entry_pc: int = 0               # first block of each repetition
    args: list[float | None] | None = None   # macro arguments (re-applied at each repetition)
    loops: list[LoopEntry] = field(default_factory=list)
    call_line: int = 0              # line of the calling block (LineRef.call_stack)
    call_program: str = ""          # name of the calling program


@dataclass
class Spindle:
    direction: int = 0              # -1 = M03 (CW seen from the spindle: omega < 0), +1 = M04, 0 = stopped
    s_value: float = 0.0            # S as programmed (rpm, or surface speed in G96 in m/min or ft/min)
    css: bool = False               # G96 active
    clamp_rad_s: float = 0.0        # G50 S / G92 S maximum speed (0 = machine maximum)
    gear: int = 1
    oriented: bool = False


@dataclass
class PathState:
    path: int = 0
    frames: list[Frame] = field(default_factory=list)
    locals: list[list[float | None]] = field(default_factory=list)
    variables: dict[str, list[float | None]] = field(default_factory=dict)   # common ranges, by name
    modal: dict[str, str] = field(default_factory=dict)       # modal group (str of group number) -> G code
    units: str = "mm"
    feed_si: float = 0.0            # modal F in SI of the feed mode
    feed_prog: float = 0.0          # modal F as programmed (#4109)
    feed_set: bool = False
    pos: dict[str, int] = field(default_factory=dict)         # program-frame position, counts per letter
    mach: dict[str, int] = field(default_factory=dict)        # commanded machine position, counts per letter
    applied: dict[str, int] = field(default_factory=dict)     # tool offset counts contained in mach
    polar_angle_deg: float = 0.0
    mirror: dict[str, int] = field(default_factory=dict)      # programmable mirror: letter -> centre counts
    scale: dict | None = None       # {"centre": {letter: counts}, "factor": {letter: f}}
    rotation: dict | None = None    # {"plane": "XY", "centre": [c1, c2] (counts), "angle_deg": a}
    feature: dict | None = None     # G268: {"origin": [x, y, z] counts, "matrix": [9 floats]}
    offsets: dict[str, list[int]] = field(default_factory=dict)   # "EXT", "G54".."G59", "P1".. -> counts per axis
    wcs: str = "G54"
    g52: dict[str, int] = field(default_factory=dict)
    g92: dict[str, int] = field(default_factory=dict)
    tool_offsets: dict[str, list[int]] = field(default_factory=dict)   # str(number) -> counts (layout per variant)
    tool: int = 0                   # tool in the spindle / turret station
    preselect: int | None = None
    h_number: int = 0
    d_number: int = 0
    offset_number: int = 0          # lathe T-code offset number
    spindle: Spindle = field(default_factory=Spindle)
    live: Spindle = field(default_factory=Spindle)
    pumps: dict[str, bool] = field(default_factory=dict)
    nozzles: dict[str, int] = field(default_factory=dict)
    outputs: dict[str, bool] = field(default_factory=dict)
    smoothing: dict = field(default_factory=dict)             # {"corner", "tolerance_m", "mode_code", "level"}
    cycle: dict | None = None       # canned cycle modal data
    cycle_initial: float | None = None
    comp_state: dict | None = None  # compensation filter data (provider's to_data())
    comp_kind: str = ""
    g28_mid: dict[str, int] = field(default_factory=dict)
    lookahead_limit: int = 0        # Haas G103 Pn (0 = machine look-ahead)
    interp_tick: int = 0
    blocked: bool = False           # stopped because the look-ahead buffer was full
    buffer: list[Command] = field(default_factory=list)
    issued: int = 0                 # commands produced so far
    retired: int = 0                # commands retired by motion so far
    block_seq: int = 0
    pending: list[Step] = field(default_factory=list)
    pending_call: dict | None = None
    waiting: dict | None = None     # {"kind", "cmd", "since", ...}
    live_values: dict | None = None # fresh live machine values for the current statement
    skip: dict[str, int] = field(default_factory=dict)
    skip_triggered: bool = False
    alarm: dict | None = None
    events: list[dict] = field(default_factory=list)
    reported: int = 0
    operator: OperatorPanel = field(default_factory=OperatorPanel)
    sys: dict = field(default_factory=dict)                   # clocks, #3003/#3004, part counters ...
    runs: int = 0
    suppress: bool = False          # restart search: interpret without producing commands
    restart_summary: dict = field(default_factory=dict)
    expired_tools: list[int] = field(default_factory=list)
    printed: list[str] = field(default_factory=list)          # DPRNT output
    last_words: dict[str, float] = field(default_factory=dict)   # address data of the last block (#41xx)
    last_n: int = 0
    tool_axis_letter: str = "Z"
    tcp: str = ""                   # active final-stage transform key


@dataclass
class ControllerState:
    paths: list[PathState] = field(default_factory=list)
    shared: dict[str, list[float | None]] = field(default_factory=dict)   # common variables shared by paths
    digest: str = ""
    wait_seq: int = 0
    version: int = STATE_VERSION

    def to_arrays(self) -> tuple[dict, dict]:
        """(arrays, metadata). Variables go to float64 arrays (NaN = <vacant>); the rest is JSON."""
        arrays: dict[str, np.ndarray] = {}
        meta = serial.dump(self)
        for p, path in enumerate(self.paths):
            for name, values in path.variables.items():
                arrays[f"p{p}.var.{name}"] = _to_array(values)
            arrays[f"p{p}.locals"] = np.array([[np.nan if v is None else v for v in level] for level in path.locals],
                                              dtype=float).reshape(len(path.locals), N_LOCALS)
            meta["paths"][p]["variables"] = sorted(path.variables)
            meta["paths"][p]["locals"] = len(path.locals)
        for name, values in self.shared.items():
            arrays[f"shared.{name}"] = _to_array(values)
        meta["shared"] = sorted(self.shared)
        json.dumps(meta)            # fails here, not in the checkpoint writer, if anything is not JSON
        return arrays, meta

    @classmethod
    def from_arrays(cls, arrays: dict, meta: dict) -> "ControllerState":
        data = json.loads(json.dumps(meta))
        names_by_path = []
        for path in data["paths"]:
            names_by_path.append((path.pop("variables"), path.pop("locals")))
            path["variables"] = {}
            path["locals"] = []
        shared_names = data.pop("shared")
        data["shared"] = {}
        state = serial.load(cls, data)
        for p, (names, n_levels) in enumerate(names_by_path):
            path = state.paths[p]
            path.variables = {name: _from_array(arrays[f"p{p}.var.{name}"]) for name in names}
            table = np.asarray(arrays[f"p{p}.locals"], dtype=float).reshape(n_levels, N_LOCALS)
            path.locals = [_from_array(row) for row in table]
        state.shared = {name: _from_array(arrays[f"shared.{name}"]) for name in shared_names}
        return state


def _to_array(values: list) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def _from_array(array) -> list:
    return [None if np.isnan(v) else float(v) for v in np.asarray(array, dtype=float)]


def new_locals() -> list[float | None]:
    return [None] * N_LOCALS
