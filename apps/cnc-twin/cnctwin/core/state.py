"""The shared simulation state: exactly one field per domain, plus core bookkeeping.

Each domain defines its state class in ``cnctwin/<domain>/state.py`` (a dataclass of plain
values and NumPy arrays) with ``to_arrays() -> (dict[str, np.ndarray], dict)`` and a
classmethod ``from_arrays(arrays, meta)``, so the checkpoint writer stores it as ``.npz`` +
JSON without knowing its fields. Per-tool data is kept by each owner in arrays indexed by the
tool slot of the immutable :class:`ToolTable`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .bus import Bus
from .findings import FindingList


class Checkpointable(Protocol):
    def to_arrays(self) -> tuple[dict, dict]:
        """(arrays, metadata): NumPy arrays by name, and JSON-serialisable metadata."""

    @classmethod
    def from_arrays(cls, arrays: dict, meta: dict) -> "Checkpointable":
        """Rebuild the state exactly (bit-identical continuation)."""


@dataclass(frozen=True)
class ToolSlot:
    slot: int
    number: int                     # T number / station the program calls
    tool_id: str                    # db id (or "inline:<n>")
    holder_id: str
    path: int = 0                   # controller path that uses it
    spindle_id: str = "main"
    sister_group: int = 0


@dataclass(frozen=True)
class ToolTable:
    """Every tool instance of the project, fixed at init (sister tools included)."""

    slots: tuple[ToolSlot, ...] = ()

    def by_number(self, number: int, path: int = 0) -> ToolSlot | None:
        for slot in self.slots:
            if slot.number == number and slot.path == path:
                return slot
        return None


@dataclass
class Clock:
    tick: int = 0                   # base ticks (tick_s, 1 us) since the start of the run
    macro: int = 0                  # macro steps since the start of the run
    op_index: int = 0               # operation (setup) being run
    part_index: int = 0             # part number in a production run (0-based)
    op_start_tick: int = 0

    def time_s(self, tick_s: float) -> float:
        return self.tick * tick_s


@dataclass
class Ledger:
    """Conservation accumulators (SI), each term computed independently (architecture 12)."""

    # mass
    removed_volume_m3: float = 0.0          # exact field volume removed (material configuration)
    removed_mass_kg: float = 0.0
    displaced_volume_m3: float = 0.0        # formed/burr/side flow: re-added to the field, not chips
    freed_body_mass_kg: float = 0.0         # slugs and parted-off parts leaving the field
    chips_in_flight_kg: float = 0.0
    chips_settled_kg: float = 0.0
    chips_conveyed_kg: float = 0.0
    chips_adhered_kg: float = 0.0
    chips_in_flutes_kg: float = 0.0
    dust_kg: float = 0.0
    # energy in
    shaft_work_J: float = 0.0
    feed_work_J: float = 0.0
    # energy out (generation by zone, then where it went)
    heat_shear_J: float = 0.0
    heat_rake_J: float = 0.0
    heat_flank_J: float = 0.0
    ploughing_J: float = 0.0
    chip_kinetic_J: float = 0.0
    modal_energy_change_J: float = 0.0
    damping_J: float = 0.0
    process_damping_J: float = 0.0
    friction_J: float = 0.0                 # packing, recut, holder and part slip
    stored_cold_work_J: float = 0.0
    unbalance_work_J: float = 0.0
    # heat destinations (thermal distributes the generation)
    heat_to_chip_J: float = 0.0
    heat_to_tool_J: float = 0.0
    heat_to_work_J: float = 0.0
    heat_to_coolant_J: float = 0.0
    removed_enthalpy_J: float = 0.0         # heat leaving with removed preheated material
    # time
    block_time_ticks: int = 0
    cycle_time_ticks: int = 0


@dataclass
class SimState:
    clock: Clock = field(default_factory=Clock)
    ledger: Ledger = field(default_factory=Ledger)
    findings: FindingList = field(default_factory=FindingList)
    tools: ToolTable = field(default_factory=ToolTable)
    bus_prev: Bus = field(default_factory=Bus)
    bus_next: Bus = field(default_factory=Bus)
    rng_counters: dict[str, int] = field(default_factory=dict)   # counter per stream key (core.rng)
    # one field per domain (Domain.name), each owned by that domain
    controller: Any = None
    motion: Any = None
    tooling: Any = None
    removal: Any = None
    cutting: Any = None
    chips: Any = None
    thermal: Any = None
    coolant: Any = None
    wear: Any = None
    holding: Any = None
    structure: Any = None
    dynamics: Any = None
    surface: Any = None

    DOMAIN_FIELDS = ("controller", "motion", "tooling", "removal", "cutting", "chips", "thermal",
                     "coolant", "wear", "holding", "structure", "dynamics", "surface")

    def swap_bus(self) -> None:
        """End of a macro step: what was written becomes what is read."""
        self.bus_prev, self.bus_next = self.bus_next, self.bus_prev
