"""The chips domain's SimState field (``state.chips``), checkpointable as arrays + JSON metadata."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import abi

# columns of ChipsState.flute (per tool slot)
FL_MASS_KG = 0
FL_BULK_M3 = 1
FL_ENTHALPY_J = 2
FL_SIZE_MASS = 3
FL_THICK_MASS = 4
FL_WIDTH_MASS = 5
FL_LENGTH_MASS = 6
FL_FORM = 7
FL_FILL = 8
FL_LENGTH_E_M = 9          # evacuation (engaged) length
FL_TORQUE_NM = 10
FL_FORCE_N = 11            # axial packing force along +z_T
FL_POWER_W = 12
FL_INFLOW_M3_S = 13
FL_V_EFF_M_S = 14
FL_STRESS_PA = 15
FL_CONFINED = 16           # confined fraction of the chips of the slot this step
N_FL = 17

# columns of ChipsState.last (per tool slot): what the slot made last (fallback emissions, releases)
LS_FORM = 0
LS_THICK_M = 1
LS_WIDTH_M = 2
LS_LENGTH_M = 3
LS_CURL_M = 4
LS_TEMP_K = 5
LS_IN_CUT = 6              # 1 when the slot had a contact in the previous window
LS_VALID = 7
N_LS = 8

# finding flags per tool slot (0 = clear, else the severity rank + 1 raised in this episode)
F_PACKING = 0
F_RECUT = 1
F_WELD = 2
F_NEST = 3
F_WRAP = 4
N_FLAGS = 5

# global finding flags
G_DUST = 0
G_FINES = 1
G_IGNITION = 2
G_CONVEYOR = 3
G_PROBE = 4
G_BODY_WRAP = 5
G_UNSCALED = 6
N_GFLAGS = 7

# trace ids per tool slot
T_FORMATION = 0
T_FLUTE = 1
T_WELD = 2
T_WRAP = 3
T_RECUT = 4
N_TRACE = 5

# cumulative heat (J)
H_PART = 0
H_FIXTURE = 1
H_MACHINE = 2
H_COOLANT = 3
H_AIR = 4
N_HEAT = 5

# pan store
PAN_MASS = 0
PAN_ENTHALPY = 1
PAN_BULK = 2
PAN_LONG = 3


def _empty_rows() -> np.ndarray:
    return np.zeros((0, abi.EMISSION_WIDTH))


@dataclass
class ChipsState:
    particles: abi.Particles
    pile: abi.Pile
    n_slots: int
    colour: np.ndarray = None           # [cap, 3] uint8
    segmented: np.ndarray = None        # [cap] bool
    attach_rel: np.ndarray = None       # [cap, 3] offset from the body reference point (M)
    pile_workpiece: int = 0
    pile_axis: int = 2                  # S axis along e_w
    pile_sign: int = 1
    pile_version: int = -1              # geometry version the base heights were built from
    pan: np.ndarray = field(default_factory=lambda: np.zeros(4))
    conveyed: np.ndarray = field(default_factory=lambda: np.zeros(2))   # mass, enthalpy (cumulative)
    dust: np.ndarray = field(default_factory=lambda: np.zeros(2))
    emitted: np.ndarray = field(default_factory=lambda: np.zeros(2))
    heat: np.ndarray = field(default_factory=lambda: np.zeros(N_HEAT))
    flute: np.ndarray = None            # [n_slots, N_FL]
    last: np.ndarray = None             # [n_slots, N_LS]
    flags: np.ndarray = None            # [n_slots, N_FLAGS] int64
    gflags: np.ndarray = field(default_factory=lambda: np.zeros(N_GFLAGS, np.int64))
    trace_ids: np.ndarray = None        # [n_slots, N_TRACE] int64 (-1 none)
    recut_hist: np.ndarray = None       # [n_slots, n_hist, 2] recut kg, made kg per step (ring)
    recut_pos: int = 0
    pending: np.ndarray = field(default_factory=_empty_rows)   # emission rows not yet released
    occ_origin_S_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    occ_cell_m: float = 0.0
    occ_mass_kg: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    occ_size_m: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    occ_thickness_m: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    nozzle_ids: list = field(default_factory=list)                 # sorted ids
    nozzle_r0_m: np.ndarray = field(default_factory=lambda: np.zeros(0))
    nozzle_blocked: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    next_pid: int = 0
    group_counter: int = 0
    macro: int = 0
    emission_capacity: int = 4096
    window_end_tick: int = -1           # end of the last integrated window (base ticks)
    last_R_M_S: np.ndarray = field(default_factory=lambda: np.eye(3))    # pose of S in M at that window's end
    last_o_S_M: np.ndarray = field(default_factory=lambda: np.zeros(3))
    recut_missing_kg: float = 0.0       # recut mass the hot loop reported that was no longer there
    unscaled_kg: float = 0.0            # emissions booked without a removal record (analytic mass)
    locator_debris: dict = field(default_factory=dict)             # locator id -> chip/burr height (end of op)
    channel_values: dict = field(default_factory=dict)

    def __post_init__(self):
        cap = len(self.particles.mass_kg)
        if self.colour is None:
            self.colour = np.zeros((cap, 3), np.uint8)
        if self.segmented is None:
            self.segmented = np.zeros(cap, bool)
        if self.attach_rel is None:
            self.attach_rel = np.zeros((cap, 3))
        n = max(int(self.n_slots), 1)
        if self.flute is None:
            self.flute = np.zeros((n, N_FL))
        if self.last is None:
            self.last = np.zeros((n, N_LS))
        if self.flags is None:
            self.flags = np.zeros((n, N_FLAGS), np.int64)
        if self.trace_ids is None:
            self.trace_ids = np.full((n, N_TRACE), -1, np.int64)
        if self.recut_hist is None:
            self.recut_hist = np.zeros((n, 1, 2))

    # ---------------------------------------------------------------- particle capacity
    def ensure_capacity(self, capacity: int) -> None:
        cap = len(self.particles.mass_kg)
        if capacity <= cap:
            return
        new_cap = max(capacity, 2 * cap)
        self.particles = abi.grow_particles(self.particles, new_cap)
        colour = np.zeros((new_cap, 3), np.uint8)
        colour[:cap] = self.colour
        seg = np.zeros(new_cap, bool)
        seg[:cap] = self.segmented
        rel = np.zeros((new_cap, 3))
        rel[:cap] = self.attach_rel
        self.colour, self.segmented, self.attach_rel = colour, seg, rel

    # ---------------------------------------------------------------- checkpoint
    def to_arrays(self) -> tuple[dict, dict]:
        arrays: dict[str, np.ndarray] = {}
        for name in abi.Particles._fields:
            arrays[f"p.{name}"] = np.array(getattr(self.particles, name))
        for name in abi.Pile._fields:
            arrays[f"h.{name}"] = np.array(getattr(self.pile, name))
        for name in ("colour", "segmented", "attach_rel", "pan", "conveyed", "dust", "emitted", "heat", "flute", "last",
                     "flags", "gflags", "trace_ids", "recut_hist", "pending", "occ_origin_S_m", "occ_mass_kg",
                     "occ_size_m", "occ_thickness_m", "nozzle_r0_m", "nozzle_blocked", "last_R_M_S", "last_o_S_M"):
            arrays[name] = np.array(getattr(self, name))
        meta = {"n_slots": int(self.n_slots), "pile_workpiece": int(self.pile_workpiece),
                "pile_axis": int(self.pile_axis), "pile_sign": int(self.pile_sign),
                "pile_version": int(self.pile_version), "recut_pos": int(self.recut_pos),
                "occ_cell_m": float(self.occ_cell_m), "nozzle_ids": list(self.nozzle_ids),
                "next_pid": int(self.next_pid), "group_counter": int(self.group_counter), "macro": int(self.macro),
                "emission_capacity": int(self.emission_capacity), "window_end_tick": int(self.window_end_tick),
                "recut_missing_kg": float(self.recut_missing_kg), "unscaled_kg": float(self.unscaled_kg),
                "locator_debris": {k: float(v) for k, v in sorted(self.locator_debris.items())},
                "channel_values": {k: float(v) for k, v in sorted(self.channel_values.items())}}
        return arrays, meta

    @classmethod
    def from_arrays(cls, arrays: dict, meta: dict) -> "ChipsState":
        def arr(name):
            return np.ascontiguousarray(np.array(arrays[name]))

        particles = abi.Particles(*(arr(f"p.{name}") for name in abi.Particles._fields))
        pile = abi.Pile(*(arr(f"h.{name}") for name in abi.Pile._fields))
        state = cls(particles=particles, pile=pile, n_slots=int(meta["n_slots"]),
                    colour=arr("colour").astype(np.uint8), segmented=arr("segmented").astype(bool),
                    attach_rel=arr("attach_rel"))
        for name in ("pan", "conveyed", "dust", "emitted", "heat", "flute", "last", "recut_hist", "pending",
                     "occ_origin_S_m", "occ_mass_kg", "occ_size_m", "occ_thickness_m", "nozzle_r0_m", "last_R_M_S",
                     "last_o_S_M"):
            setattr(state, name, arr(name))
        state.flags = arr("flags").astype(np.int64)
        state.gflags = arr("gflags").astype(np.int64)
        state.trace_ids = arr("trace_ids").astype(np.int64)
        state.nozzle_blocked = arr("nozzle_blocked").astype(np.int64)
        for key in ("pile_workpiece", "pile_axis", "pile_sign", "pile_version", "recut_pos", "next_pid",
                    "group_counter", "macro", "emission_capacity", "window_end_tick"):
            setattr(state, key, int(meta[key]))
        state.occ_cell_m = float(meta["occ_cell_m"])
        state.nozzle_ids = list(meta["nozzle_ids"])
        state.recut_missing_kg = float(meta["recut_missing_kg"])
        state.unscaled_kg = float(meta["unscaled_kg"])
        state.locator_debris = dict(meta.get("locator_debris", {}))
        state.channel_values = dict(meta.get("channel_values", {}))
        return state

    # ---------------------------------------------------------------- mass terms (architecture 12)
    def mass_terms(self) -> dict[str, float]:
        p = self.particles
        n = int(p.n[0])
        status = p.status[:n]
        mass = p.mass_kg[:n]
        flying = float(mass[status == abi.FLYING].sum())
        adhered = float(mass[(status == abi.ON_TOOL) | (status == abi.WRAPPED) | (status == abi.ON_NOZZLE)].sum())
        return {"in_flight_kg": flying,
                "settled_kg": float(self.pile.mass_kg.sum()) + float(self.pan[PAN_MASS]),
                "conveyed_kg": float(self.conveyed[0]),
                "adhered_kg": adhered,
                "in_flutes_kg": float(self.flute[:, FL_MASS_KG].sum()),
                "dust_kg": float(self.dust[0])}

    def enthalpy_terms(self, specific_enthalpy) -> dict[str, float]:
        """Chip enthalpy now (J, relative to 20 degC) by where the chips are; specific_enthalpy(T) -> J/kg."""
        p = self.particles
        n = int(p.n[0])
        live = np.nonzero((p.status[:n] != abi.FREE) & (p.mass_kg[:n] > 0.0))[0]
        particles = float(sum(p.mass_kg[i] * specific_enthalpy(p.temp_K[i]) for i in live))
        return {"particles_J": particles, "pile_J": float(self.pile.enthalpy_J.sum()),
                "pan_J": float(self.pan[PAN_ENTHALPY]), "flutes_J": float(self.flute[:, FL_ENTHALPY_J].sum())}
