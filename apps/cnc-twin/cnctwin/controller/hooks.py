"""Extension points of the controller core (docs/dialects.md "Structure of the controller", hooks).

Six hook families, each a registry keyed by text, filled at import time by provider modules:

* **cycles** (``register_cycle``): a cycle code (``"G81"``, ``"G71"``, ``"haas:G150"``) -> a provider whose
  ``expand(ex, call: CycleCall) -> list[Step]`` turns one cycle execution into steps (program-frame
  moves, dwells, commands). Steps are plain data kept in the path state (``PathState.pending``), so a
  checkpoint can fall in the middle of an expansion.
* **compensation** (``register_compensation``): ``"cutter"`` (G41/G42), ``"nose"`` (lathe G41/G42),
  ``"3d"`` (G41.2/G141 ...), ``"corner"`` (``,C`` ``,R`` chamfer/corner rounding) -> a factory
  ``create(ex) -> CompFilter``. A filter is a block-stream filter with look-ahead on work-frame
  :class:`PathItem` objects: ``push(ex, item) -> list[PathItem]`` returns what may be released now,
  ``flush(ex, reason) -> list[PathItem]`` releases everything; its whole state is
  ``to_data()``/``from_data(ex, data)`` (JSON types) so it checkpoints with the path.
* **transforms** (``register_transform``): ``"haas:G234"``, ``"fanuc:G43.4"``, ``"fanuc:G68.2"``,
  ``"fanuc:G12.1"`` ... -> a final stage ``to_command(ex, item: PathItem) -> Command`` used instead of
  the default "work + offsets = machine" stage while its modal code is active (TCP, tilted planes,
  polar/cylindrical interpolation).
* **probing** (``register_probe``): ``"G31"``, ``"G37"``, ``"haas:G36"`` ... -> provider with
  ``start(ex, call) -> list[Step]`` and ``resume(ex, data, feedback) -> list[Step]``; a step of kind
  ``"probe"``/``"setter"`` is buffer-stopping and the provider continues on the feedback.
* **tool_life** (``register_tool_life``): one provider with ``select(ex, number) -> int`` (called at
  every tool call; returns the tool actually used, e.g. a sister) and ``expired(ex, numbers)``.
* **restart** (``register_restart``): one provider with ``reconstruct(ex, target) -> list[Step]``;
  the core offers ``ex.search(...)`` (run blocks without output, keeping modal state) for it.

A provider registered under an existing key raises unless ``replace=True`` (explicit override).
Plugin packages are imported by :func:`load_plugins` in a fixed order; a plugin that exists but
fails to import raises (nothing is skipped silently).
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass, field
from typing import Any, Protocol

from .ir import Command, Feed, LineRef, Smoothing

# --------------------------------------------------------------------------- data passed through hooks


@dataclass
class Step:
    """One element of a cycle expansion, in the program frame (before mirror/scale/rotation).

    kind: ``rapid`` | ``linear`` | ``arc`` | ``dwell`` | ``command`` | ``probe`` | ``setter`` | ``resume``
    | ``mode``. Targets are absolute program-frame values in SI (metres, radians; lathe X as radius)
    keyed by program letter; axes not named keep their position.
    """

    kind: str
    target: dict[str, float] = field(default_factory=dict)
    centre: dict[str, float] | None = None       # arc centre (absolute, program frame, plane axes)
    direction: int = 0                           # arc: -1 CW (G02), +1 CCW (G03)
    plane: str = ""                              # "" = modal plane; "XY" | "ZX" | "YZ"
    feed: float | None = None                    # SI in feed_mode; None = the modal F
    feed_mode: str = ""                          # "" = modal; "per_min" | "per_rev" | "inverse_time"
    dwell_s: float = 0.0
    dwell_rev: float = 0.0
    comp: bool = False                           # True: through cutter compensation (contour moves)
    machine: bool = False                        # target is in machine coordinates (G53-like)
    exact_stop: bool = False
    override: bool = True                        # feed override allowed
    command: Command | None = None               # kind "command": a non-motion command
    modal: dict = field(default_factory=dict)    # extra entries for the produced command's modal snapshot
    provider: str = ""                           # kind "resume"/"probe"/"setter": who continues
    data: dict = field(default_factory=dict)     # provider continuation data (JSON types only)
    probe: dict | None = None                    # kind "probe"/"setter": payload of the PROBE/TOOL_SETTER command
    ref: LineRef | None = None                   # block the step belongs to (set by the core when queued)


@dataclass
class CycleCall:
    """One execution of a cycle (one hole of a canned cycle, one G12/G150/G71 block)."""

    code: str                       # "G81", "G150", ...
    dialect: str
    plane: str                      # "XY" | "ZX" | "YZ"
    absolute: bool                  # G90 (or lathe X/Z words) at the call
    position: dict[str, float]      # hole position / start point, program frame absolute (SI)
    current: dict[str, float]       # tool position before the call, program frame absolute (SI)
    initial_level: float | None     # canned cycles: drilling-axis value when the cycle mode started
    return_to_initial: bool         # G98 (True) / G99 (False)
    data: dict[str, float]          # cycle data in SI: R Z Q P (s) F (SI feed) I J K L ... as given
    words: dict[str, float]         # the block's raw numeric word values (program units, after vacant removal)
    repeat_index: int = 0
    feed_mode: str = "per_min"
    units: str = "mm"
    extra: dict = field(default_factory=dict)


@dataclass
class PathItem:
    """A motion (or a non-motion command keeping its place) in the work frame of the control:
    after mirror, scaling, rotation and feature coordinate systems, before work offsets and tool
    length offsets. Lists are in the path's axis order (SI; lathe X as radius)."""

    kind: str                                    # "rapid" | "linear" | "arc" | "probe" | "command"
    start: list[float] = field(default_factory=list)
    end: list[float] = field(default_factory=list)
    centre: list[float] | None = None            # arc centre, work frame XYZ (missing axes 0)
    normal: list[float] | None = None            # arc rotation axis (unit), work frame XYZ
    sweep_rad: float = 0.0                       # signed about normal
    r_start: float = 0.0
    r_end: float = 0.0
    plane: str = "XY"
    feed: Feed | None = None
    smoothing: Smoothing | None = None
    offset: list[int] = field(default_factory=list)   # counts per axis: machine = round(work / q) + offset
    comp: str = "G40"                            # cutter compensation mode at interpretation
    comp_radius_m: float = 0.0                   # active D value (SI)
    space: str = "machine"                       # "machine" | "direct" (end already machine coordinates)
    transform: str = ""                          # active final-stage key ("" = default)
    command: Command | None = None               # kind "command": the command to release
    ref: LineRef | None = None
    block_seq: int = -1
    ready_tick: int = 0
    modal: dict = field(default_factory=dict)
    probe: dict | None = None
    corner: dict | None = None                   # ,C / ,R request of the block
    bypass: bool = False                         # cycle positioning: not offset by cutter compensation


# --------------------------------------------------------------------------- provider protocols


class CycleProvider(Protocol):
    def expand(self, ex: Any, call: CycleCall) -> list[Step]: ...


class CompFilter(Protocol):
    def push(self, ex: Any, item: PathItem) -> list[PathItem]: ...
    def flush(self, ex: Any, reason: str) -> list[PathItem]: ...
    def to_data(self) -> dict: ...


class CompFactory(Protocol):
    def create(self, ex: Any) -> CompFilter: ...
    def from_data(self, ex: Any, data: dict) -> CompFilter: ...


class TransformProvider(Protocol):
    def to_command(self, ex: Any, item: PathItem) -> Command: ...


class ProbeProvider(Protocol):
    def start(self, ex: Any, call: CycleCall) -> list[Step]: ...
    def resume(self, ex: Any, data: dict, feedback: Any) -> list[Step]: ...


class ToolLifeProvider(Protocol):
    def select(self, ex: Any, number: int) -> int: ...
    def expired(self, ex: Any, numbers: list[int]) -> None: ...


class RestartProvider(Protocol):
    def reconstruct(self, ex: Any, target: dict) -> list[Step]: ...


# --------------------------------------------------------------------------- registries

CYCLES: dict[str, CycleProvider] = {}
COMPENSATION: dict[str, CompFactory] = {}
TRANSFORMS: dict[str, TransformProvider] = {}
PROBES: dict[str, ProbeProvider] = {}
TOOL_LIFE: dict[str, ToolLifeProvider] = {}
RESTART: dict[str, RestartProvider] = {}
RESUMERS: dict[str, Any] = {}       # name -> object with resume(ex, data, feedback) (cycles that buffer-stop)


def _register(table: dict, key: str, provider, replace: bool) -> None:
    if key in table and not replace and table[key] is not provider:
        raise ValueError(f"a provider is already registered for {key!r} (pass replace=True to override)")
    table[key] = provider


def register_cycle(code: str, provider: CycleProvider, *, dialect: str = "", replace: bool = False) -> None:
    """``code`` is the G code ("G81"); ``dialect`` restricts it ("haas" -> key "haas:G81")."""
    _register(CYCLES, f"{dialect}:{code}" if dialect else code, provider, replace)


def register_compensation(kind: str, factory: CompFactory, *, replace: bool = False) -> None:
    _register(COMPENSATION, kind, factory, replace)


def register_transform(key: str, provider: TransformProvider, *, replace: bool = False) -> None:
    _register(TRANSFORMS, key, provider, replace)


def register_probe(key: str, provider: ProbeProvider, *, replace: bool = False) -> None:
    _register(PROBES, key, provider, replace)


def register_tool_life(provider: ToolLifeProvider, *, replace: bool = False) -> None:
    _register(TOOL_LIFE, "default", provider, replace)


def register_restart(provider: RestartProvider, *, replace: bool = False) -> None:
    _register(RESTART, "default", provider, replace)


def register_resumer(name: str, provider, *, replace: bool = False) -> None:
    _register(RESUMERS, name, provider, replace)


def find(table: dict, dialect: str, key: str):
    """Dialect-specific provider first ("haas:G81"), then the generic one ("G81")."""
    return table.get(f"{dialect}:{key}") or table.get(key)


# --------------------------------------------------------------------------- additive extension points
# (added by the cycles / compensation / TCP / probing / tool-life / restart providers; docs/dialects.md
# "Cycles and compensation"). Each is optional: without a registration the core behaves as before.

GCODES: dict[str, dict] = {}            # "<dialect>:<variant>" -> {code: GSpec} merged into the dialect's G-code table
GCODE_HANDLERS: dict[str, Any] = {}     # GSpec(kind="ext", arg=key) -> handler.execute(ex, code, spec, bw) -> moved
MOTION_WORDS: dict[str, Any] = {}       # name -> provider.motion_words(ex, code, bw) -> dict merged into the item modal
LIBRARIES: dict[str, Any] = {}          # name -> provider.sources(machine, dialect, settings, programs) -> [ProgramSource]
DATA_MODES: dict[str, Any] = {}         # "G10L3" -> provider with begin(ex, p, bw), block(ex, bw), end(ex)


def register_gcode(code: str, spec, *, variant: str, dialect: str, replace: bool = False) -> None:
    """Add (or with ``replace`` redefine) a G code of ``dialect`` ("fanuc", "haas") and ``variant`` ("M"/"T")."""
    table = GCODES.setdefault(f"{dialect}:{variant}", {})
    if code in table and not replace and table[code] != spec:
        raise ValueError(f"G code {code} is already registered for {dialect}/{variant}")
    table[code] = spec


def extra_gcodes(dialect: str, variant: str) -> dict:
    """The registered G codes of one dialect and variant (the dialect builders merge them)."""
    return dict(GCODES.get(f"{dialect}:{variant}", {}))


def register_gcode_handler(key: str, handler, *, replace: bool = False) -> None:
    _register(GCODE_HANDLERS, key, handler, replace)


def register_motion_words(name: str, provider, *, replace: bool = False) -> None:
    _register(MOTION_WORDS, name, provider, replace)


def register_library(name: str, provider, *, replace: bool = False) -> None:
    _register(LIBRARIES, name, provider, replace)


def register_data_mode(key: str, provider, *, replace: bool = False) -> None:
    _register(DATA_MODES, key, provider, replace)


def library_sources(machine, dialect, settings, programs: list) -> list:
    """Built-in programs every registered library provides for this machine (in registration order).
    ``dialect`` is the built :class:`~cnctwin.controller.dialects.base.Dialect` (name, variant, params);
    ``programs`` the project's own sources, so a library can leave out a number the project defines."""
    out = []
    for provider in LIBRARIES.values():
        out.extend(provider.sources(machine, dialect, settings, list(programs) + out) or [])
    return out


# Plugin packages of the agents that extend the core, imported in this order when present.
PLUGIN_PACKAGES = ("cycles", "comp", "probing", "toollife", "restart", "tcp",
                   "dialects.siemens840d", "dialects.heidenhain")
_loaded = False


def load_plugins() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    for name in PLUGIN_PACKAGES:
        module = f"cnctwin.controller.{name}"
        if importlib.util.find_spec(module) is not None:
            importlib.import_module(module)
