"""What the control hands to the machine, and what the machine reports back.

The executor (any dialect) turns blocks into :class:`Command` objects: motion in machine
axes (after every offset, compensation and coordinate transformation the control applies),
or tool-tip motion in the workpiece frame when tool centre point control is active, plus
the non-motion actions (spindle, coolant, tool change, ...). ``motion`` plans and times
them; the physics reports back through :class:`Feedback` (probe hits, tool-setter
measurements, live positions for macro reads, alarms raised by the machine).

All values are SI: metres for linear axes, radians for rotary axes, seconds, rad/s.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Kind(str, Enum):
    RAPID = "rapid"                 # G00 (or G53 G00): axes move at rapid, linear or non-linear per control
    LINEAR = "linear"               # G01 / L
    ARC = "arc"                     # G02/G03, CW/CCW in the active plane, helical if the third axis moves
    DWELL = "dwell"                 # G04
    SPINDLE = "spindle"             # S/M03/M04/M05/M19, CSS G96/G97/G50 clamp, gear range
    COOLANT = "coolant"             # M07/M08/M09/M88/M89/M34/M35 ... (pump and nozzle states)
    TOOL_CHANGE = "tool_change"     # M06 (mill), T-word (lathe turret index with offset)
    TURRET_INDEX = "turret_index"
    PALLET = "pallet"
    DOOR = "door"
    CONVEYOR = "conveyor"
    CHUCK = "chuck"                 # clamp/unclamp, tailstock, steady rest, sub-spindle transfer, part catcher
    STOP = "stop"                   # M00, M01 (optional), M02/M30 (end), feed hold points
    PROBE = "probe"                 # G31 skip move (and probe-cycle moves); waits for feedback
    TOOL_SETTER = "tool_setter"     # tool length/diameter/breakage measurement; waits for feedback
    MESSAGE = "message"             # operator message (#3006, MSG, FN 14 with stop)
    ALARM = "alarm"                 # program-raised or executor alarm: the control stops
    SYNC = "sync"                   # buffer-stopping point: motion must finish before the executor goes on
    MODE = "mode"                   # smoothing / exact stop / look-ahead / feed mode change effective here


class FeedMode(str, Enum):
    PER_MIN = "per_min"             # G94 / G98 (lathe)
    PER_REV = "per_rev"             # G95 / G99 (lathe)
    INVERSE_TIME = "inverse_time"   # G93
    RAPID = "rapid"


class Space(str, Enum):
    MACHINE = "machine"             # targets are machine axis values (G53 frame), joint interpolation
    TIP = "tip"                     # targets are tool-tip position + axis in the workpiece frame (TCP active)


@dataclass(frozen=True)
class LineRef:
    program: str                    # file name as given in the project
    line: int                       # 1-based line number in that file
    block: str                      # the block text as written
    n_number: int | None = None
    call_stack: tuple[str, ...] = ()   # "O1000:12 > O9810:4" style frames for subprograms/macros


@dataclass
class Feed:
    mode: FeedMode = FeedMode.PER_MIN
    value: float = 0.0              # m/s (per_min), m/rev (per_rev), 1/s (inverse_time)
    override_allowed: bool = True   # False in tapping/threading (G63, G84, G32/G76 ...)


@dataclass
class ArcSpec:
    plane: str = "XY"               # XY (G17), ZX (G18), YZ (G19)
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0)   # in the same space as the targets
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)   # rotation axis, right-handed; CW = negative turn
    sweep_rad: float = 0.0          # signed sweep, full circles = +-2 pi * turns
    radius_start_m: float = 0.0
    radius_end_m: float = 0.0       # differs only within the control's tolerance (or spiral)


@dataclass
class Smoothing:
    corner: str = "blend"           # blend (G64) | exact_stop (G61 / G09 block) | tapping (G63)
    tolerance_m: float = 0.0        # contour tolerance band of the active high-speed mode (0 = control default)
    mode_code: str = ""             # "G187 P2", "G05.1 Q1", "CYCLE832 _ROUGH", "CYCL DEF 32" ...
    lookahead: bool = True
    # what tolerance_m bounds at a corner (additive, docs/models/motion.md 2.2): "deviation" from the programmed
    # corner (Heidenhain cycle 32 T, SINUMERIK CYCLE832 / G642, FANUC smoothing), "radius" of the rounded corner
    # (Haas Setting 85 / G187 E), "distance" of the rounding along each block (SINUMERIK G641 ADIS)
    tolerance_kind: str = "deviation"


@dataclass
class Command:
    kind: Kind
    ref: LineRef
    space: Space = Space.MACHINE
    target: dict[str, float] = field(default_factory=dict)    # machine axes {"X": m, "B": rad} (MACHINE)
    tip_target_m: tuple[float, float, float] | None = None    # tool tip in W (TIP)
    tip_axis: tuple[float, float, float] | None = None        # tool axis unit vector in W (TIP)
    arc: ArcSpec | None = None
    feed: Feed | None = None
    smoothing: Smoothing | None = None
    # non-motion payload (one of these is set according to kind)
    spindle: dict | None = None     # {"id": "main", "omega_rad_s": signed, "css_m_s": v or None, "max_rad_s": clamp, "gear": n, "orient_rad": a}
    coolant: dict | None = None     # {"pumps": {"flood": True}, "nozzles": {"n1": position_index}}
    tool: dict | None = None        # {"number": 12, "pocket": 12, "length_offset": 12, "radius_offset": 12, "turret": "T1", "station": 3}
    probe: dict | None = None       # {"skip": 1, "stylus": "spindle_probe", "expect": "trigger"}
    setter: dict | None = None      # {"measure": "length"|"diameter"|"broken", "tool": 12, "tolerance_m": 1e-4}
    dwell_s: float = 0.0
    dwell_rev: float = 0.0
    stop: str = ""                  # "M00" | "M01" | "M02" | "M30" | "M99-end"
    message: str = ""
    alarm: dict | None = None       # {"code": "PS0010", "text": "IMPROPER G-CODE", "dialect": "fanuc"}
    modal: dict = field(default_factory=dict)   # snapshot the physics may need: plane, units, work offset, comp side,
                                                # cycle name, tool number, length/radius offset values, feed mode, css
    # --- additive (controller core, docs/dialects.md "Structure of the controller") ---
    ready_tick: int = 0             # base tick at which the control has finished preparing this block (block-time
                                    #   model): the planner never starts it, nor looks ahead past it, before this tick
    path: int = 0                   # controller path (channel) that produced the command, 0-based
    block_seq: int = -1             # executed-block counter of that path (commands of one NC block share it)
    sync: dict | None = None        # SYNC payload: {"reason": "live_read"|"path_wait"|"single_block"|..., "code": "M100",
                                    #   "paths": [0, 1], "id": n}; None for other kinds


@dataclass
class OperatorPanel:
    """Operator switches of one controller path (set by the operator model, read by motion each macro step).

    Additive (controller core). Overrides are fractions (1.0 = 100 %); motion applies feed/rapid/spindle
    overrides only where the command allows it (``Feed.override_allowed``, ``modal["feed_hold_inhibit"]``).
    """

    feed_override: float = 1.0
    rapid_override: float = 1.0
    spindle_override: float = 1.0
    feed_hold: bool = False          # motion decelerates to a stop and holds while set
    single_block: bool = False       # the executor stops after every NC block (STOP "single_block")
    optional_stop: bool = False      # M01 stops only when set
    block_delete: list[bool] = field(default_factory=lambda: [False] * 9)   # switches /1 ... /9 (/ = /1)
    dry_run: bool = False            # motion: cutting feeds replaced by the dry-run feed
    machine_lock: bool = False       # motion: axes do not move (the program is still timed)


@dataclass
class ProbeResult:
    triggered: bool
    machine_pos: dict[str, float]   # axis values latched at the skip signal (SI)
    time_s: float


@dataclass
class SetterResult:
    tool: int
    length_m: float | None = None
    diameter_m: float | None = None
    broken: bool = False
    time_s: float = 0.0


@dataclass
class Feedback:
    """What the physics reports to the executor at a buffer-stopping point."""

    time_s: float
    machine_pos: dict[str, float] = field(default_factory=dict)   # current actual positions
    probe: ProbeResult | None = None
    setter: SetterResult | None = None
    alarm: dict | None = None       # machine-side alarm (overtravel, spindle overload ...): {"code", "text"}
    tool_life_expired: list[int] = field(default_factory=list)    # tool numbers whose life ended
    spindle_load_fraction: float = 0.0
