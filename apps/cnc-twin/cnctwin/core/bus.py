"""The coupling bus: what the domains exchange once per macro step (docs/architecture.md 7).

Rules
-----
* **One writer per record.** Each record below names its writer; no other domain assigns to it
  (a test enforces this).
* **Double-buffered.** The loop keeps ``prev`` (read-only, macro step n-1) and ``next`` (written
  in step n) and swaps them at the end of the step, so the order in which the slow domains run
  cannot change results and they may run concurrently. Every slow-to-slow coupling therefore
  lags one macro step; fast coupling lives inside the hot loop.
* **Frames.** Positions and directions that chips, coolant and the UI use are in M (machine,
  inertial). Geometry tied to the material (deposits, surface events, displacement grids) is in
  S, the stock lattice frame, in the *material configuration* (20 degC, unclamped, undistorted;
  architecture 4.1). Forces on the tool are in the non-rotating tool frame T0; forces on the part
  in W.
* **Indexing.** ``contact`` = an active (tool slot, workpiece, path) cutting pair (several may
  cut at once on lathes with two turrets, mill-turn and Swiss machines). ``tool`` = a tool slot of
  the core ToolTable. Per-element arrays are ``[n_teeth, n_elem]`` of that tool's EdgeModel.
* SI units; records may gain fields (additively, documented by the writer), never change meaning.
* Empty means are 0 with the matching time integral 0, never NaN.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _vec3() -> np.ndarray:
    return np.zeros(3)


def _empty() -> np.ndarray:
    return np.zeros(0)


def _empty2(cols: int):
    return lambda: np.zeros((0, cols))


def _pose() -> np.ndarray:
    return np.eye(4)


def _grid3():
    return np.zeros((0, 0, 0, 3))


# ISO 3685:1993 Annex G chip forms, encoded as class * 10 + subclass (0 = no subclass).
CHIP_FORMS = {
    11: "ribbon, long", 12: "ribbon, short", 13: "ribbon, snarled",
    21: "tubular, long", 22: "tubular, short", 23: "tubular, snarled",
    31: "spiral, flat", 32: "spiral, conical",
    41: "washer-type helical, long", 42: "washer-type helical, short", 43: "washer-type helical, snarled",
    51: "conical helical, long", 52: "conical helical, short",
    61: "arc, connected", 62: "arc, loose",
    70: "elemental",
    80: "needle",
    90: "dust",               # composites, MDF, cast-iron fines (not in ISO 3685; our code)
    95: "molten / smeared",   # polymers above Tg, smeared aluminium (not in ISO 3685; our code)
}


# --------------------------------------------------------------------------- motion (writer: motion)

@dataclass
class PathFrame:
    """One controller path/channel over the macro step. Per-tick arrays have length n_ticks."""

    path: int = 0
    tick0: int = 0                          # first L1 tick of the step (base-clock ticks)
    ipo_ticks: int = 1000                   # base ticks per L1 tick
    tick: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    block_index: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))  # into timeline.json
    tip_S_m: np.ndarray = field(default_factory=_empty2(3))     # tool tip in the stock frame (as moved)
    tip_M_m: np.ndarray = field(default_factory=_empty2(3))
    axis_T_M: np.ndarray = field(default_factory=_empty2(3))    # tool axis (+z_T) in M
    axis_names: tuple[str, ...] = ()
    axis_actual: np.ndarray = field(default_factory=_empty2(0))  # [n_ticks, n_axes] SI
    axis_cmd: np.ndarray = field(default_factory=_empty2(0))
    following_error: np.ndarray = field(default_factory=_empty2(0))
    spindle_angle_rad: np.ndarray = field(default_factory=_empty)   # right-handed about +z_T, wrapped [0, 2 pi)
    spindle_rev: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    spindle_omega_rad_s: np.ndarray = field(default_factory=_empty)  # right-handed about +z_T: M03 < 0
    feed_vel_M_m_s: np.ndarray = field(default_factory=_empty2(3))
    rapid: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    tool_slot: int = -1                     # active tool slot at the end of the step (-1 none)
    spindle_id: str = "main"
    workpiece: int = 0
    pose_M_T: np.ndarray = field(default_factory=_pose)          # end of step
    pose_S_T: np.ndarray = field(default_factory=_pose)
    pose_W0_T: np.ndarray = field(default_factory=_pose)         # non-rotating workpiece frame (lathes)
    coolant_pumps: dict[str, bool] = field(default_factory=dict)
    nozzle_settings: dict[str, float] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)  # completed non-motion actions with their tick
    axis_inertial_force_N: np.ndarray = field(default_factory=_empty2(0))   # m a per axis per tick (base excitation)


@dataclass
class MotionFrame:
    time_s: float = 0.0                     # end of the macro step
    tick: int = 0
    paths: list[PathFrame] = field(default_factory=list)


# ------------------------------------------------------------------------- removal (writer: removal)

@dataclass
class ToolContact:
    in_cut: bool = False
    point_M_m: np.ndarray = field(default_factory=_vec3)    # engaged-region centroid
    normal_M: np.ndarray = field(default_factory=_vec3)     # machined-surface outward normal there
    point_S_m: np.ndarray = field(default_factory=_vec3)
    ae_m: float = 0.0
    ap_m: float = 0.0
    engaged_length_m: float = 0.0
    engaged_angle_rad: float = 0.0
    in_cut_fraction: float = 0.0
    air_cut_time_s: float = 0.0             # feed motion out of contact this step


@dataclass
class RemovalContact:
    contact: int = 0
    tool_slot: int = -1
    workpiece: int = 0
    path: int = 0
    removed_volume_m3: float = 0.0          # exact field volume removed this step (Z family / polar radial)
    displaced_volume_m3: float = 0.0        # formed / burr / side flow: re-added, not chips
    geometry: ToolContact = field(default_factory=ToolContact)
    volume_consistency: float = 1.0         # field volume / analytic chip volume this step


@dataclass
class RemovalFrame:
    contacts: list[RemovalContact] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)       # collisions, gouges, breakthrough, rapids into material, with ticks
    loose_bodies: list[dict] = field(default_factory=list) # {workpiece, volume_m3, mass_kg, centroid_S_m, tick}
    field_version: dict[int, int] = field(default_factory=dict)   # workpiece -> increments when geometry changed


# ------------------------------------------------------------------------- cutting (writer: cutting)

@dataclass
class ProcessCoefficients:
    """Cutting coefficients for the next macro step, per element group, tabulated over h.

    Built at L2 by cutting's coefficient pipeline with the mechanics model (Oxley / Johnson-Cook)
    from the current local speed, rake, edge radius, wear, friction (coolant), tool temperature
    and workpiece preheat. The hot loop only interpolates (no transcendental work at L0).
    """

    tool_slot: int = -1
    h_grid_m: np.ndarray = field(default_factory=_empty)          # [n_h] log-spaced
    Kt_Pa: np.ndarray = field(default_factory=_empty2(0))         # [n_group, n_h] specific force on h
    Kr_Pa: np.ndarray = field(default_factory=_empty2(0))
    Ka_Pa: np.ndarray = field(default_factory=_empty2(0))
    Kte_N_per_m: np.ndarray = field(default_factory=_empty)       # [n_group] edge (ploughing) forces
    Kre_N_per_m: np.ndarray = field(default_factory=_empty)
    Kae_N_per_m: np.ndarray = field(default_factory=_empty)
    shear_angle_rad: np.ndarray = field(default_factory=_empty2(0))   # [n_group, n_h]
    chip_ratio: np.ndarray = field(default_factory=_empty2(0))
    frac_shear_to_chip: np.ndarray = field(default_factory=_empty2(0))
    frac_rake_to_tool: np.ndarray = field(default_factory=_empty2(0))
    frac_flank_to_tool: np.ndarray = field(default_factory=_empty2(0))
    T_shear_K: np.ndarray = field(default_factory=_empty2(0))
    T_interface_K: np.ndarray = field(default_factory=_empty2(0))
    h_min_m: np.ndarray = field(default_factory=_empty)           # [n_group] minimum chip thickness
    v_min_m_s: float = 0.0                                        # below: indentation/ploughing model
    K_sp_N_per_m3: np.ndarray = field(default_factory=_empty)     # [n_group] indentation (process damping)
    flank_coeff_N_per_m2: np.ndarray = field(default_factory=_empty2(0))  # [n_teeth, n_elem] ploughing per VB area
    version: int = 0


@dataclass
class CuttingContact:
    contact: int = 0
    tool_slot: int = -1
    in_cut_s: float = 0.0
    wrench_tool_T0_mean: np.ndarray = field(default_factory=_empty2(6))   # [n_ticks, 6] F (N), M about tip (Nm)
    wrench_tool_T0_peak: np.ndarray = field(default_factory=_empty2(6))   # sample at max |F| in each tick
    wrench_work_W_mean: np.ndarray = field(default_factory=_empty2(6))    # [n_ticks, 6] on the part about W origin
    wrench_work_W_peak: np.ndarray = field(default_factory=_empty2(6))
    force_peak_M_N: np.ndarray = field(default_factory=_vec3)     # the force vector at the instant of max |F|
    torque_mean_Nm: float = 0.0
    torque_peak_Nm: float = 0.0
    spindle_power_mean_W: float = 0.0
    spindle_power_peak_W: float = 0.0
    feed_power_mean_W: float = 0.0
    spindle_load_fraction: float = 0.0      # demand / available torque at speed and gear
    axis_force_N: np.ndarray = field(default_factory=_empty2(0))  # [n_ticks, n_axes] cutting load per axis
    spindle_torque_Nm: np.ndarray = field(default_factory=_empty)  # [n_ticks] for the spindle speed loop
    shaft_work_J: float = 0.0               # integral of torque * omega
    feed_work_J: float = 0.0                # integral of F . v_feed
    heat_shear_J: float = 0.0               # generation by zone (partition is applied by thermal
    heat_rake_J: float = 0.0                #   with ProcessCoefficients fractions)
    heat_flank_J: float = 0.0
    ploughing_J: float = 0.0                # below h_min (rubbing) and edge forces
    analytic_chip_volume_m3: float = 0.0    # integral of h db r dphi
    # per element integrals over the step: [n_teeth, n_elem]
    in_cut_time_s: np.ndarray = field(default_factory=_empty2(0))
    rake_sliding_m: np.ndarray = field(default_factory=_empty2(0))
    flank_sliding_m: np.ndarray = field(default_factory=_empty2(0))
    int_sigma_v_rake: np.ndarray = field(default_factory=_empty2(0))    # integral sigma_n V_s dt (Pa m)
    int_sigma_v_flank: np.ndarray = field(default_factory=_empty2(0))
    int_h_dt: np.ndarray = field(default_factory=_empty2(0))            # m s
    h_max_m: np.ndarray = field(default_factory=_empty2(0))
    entries: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.int32))
    shocks: np.ndarray = field(default_factory=_empty2(8))  # [tick, tooth, elem, entry_h_m, entry_angle_rad, force_first_full_N, loading_time_s, kind]
    work_heat_S: np.ndarray = field(default_factory=_empty2(7))   # [x, y, z, nx, ny, nz, J] deposits on the generated surface (S, material config)
    surface_events: np.ndarray = field(default_factory=_empty2(16))  # see SURFACE_EVENT_COLUMNS
    section_moment_peak_Nm: np.ndarray = field(default_factory=_empty)  # tool critical sections
    section_stress_peak_Pa: np.ndarray = field(default_factory=_empty)
    recut: np.ndarray = field(default_factory=_empty2(6))    # [tick, tooth, elem, mass_kg, x_M, y_M] (z in surface_events)


SURFACE_EVENT_COLUMNS = (
    "tick", "tooth", "elem", "x_S_m", "y_S_m", "z_S_m", "nx", "ny", "nz",
    "h_m", "edge_radius_m", "VB_m", "u_rel_x_m", "u_rel_y_m", "u_rel_z_m", "tool_slot",
)


@dataclass
class CuttingFrame:
    contacts: list[CuttingContact] = field(default_factory=list)
    coefficients: dict[int, ProcessCoefficients] = field(default_factory=dict)   # tool slot -> next step's tables


# ------------------------------------------------------------------- chips (writer: chips)

@dataclass
class EmissionBatch:
    """Chips emitted in the step: one row per representative particle (M frame)."""

    tick: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    tool_slot: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    mass_kg: np.ndarray = field(default_factory=_empty)
    temp_K: np.ndarray = field(default_factory=_empty)
    pos_M_m: np.ndarray = field(default_factory=_empty2(3))
    vel_M_m_s: np.ndarray = field(default_factory=_empty2(3))
    thickness_m: np.ndarray = field(default_factory=_empty)
    width_m: np.ndarray = field(default_factory=_empty)
    length_m: np.ndarray = field(default_factory=_empty)
    curl_radius_m: np.ndarray = field(default_factory=_empty)
    form_code: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int16))
    colour_rgb: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.uint8))
    segmented: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))

    def __len__(self) -> int:
        return len(self.mass_kg)


@dataclass
class ChipFormationContact:
    contact: int = 0
    shear_angle_rad: float = 0.0
    chip_ratio: float = 0.0
    shear_strain: float = 0.0
    strain_rate_1_s: float = 0.0
    chip_flow_angle_rad: float = 0.0
    chip_temp_K: float = 0.0
    form_code: int = 0
    bue_active: bool = False
    emissions: EmissionBatch = field(default_factory=EmissionBatch)


@dataclass
class ChipFeedback:
    formation: list[ChipFormationContact] = field(default_factory=list)
    flute_fill: dict[int, float] = field(default_factory=dict)            # tool slot -> fill fraction
    packing_torque_Nm: dict[int, float] = field(default_factory=dict)     # additive, with its friction power
    packing_force_T0_N: dict[int, np.ndarray] = field(default_factory=dict)
    occupancy_origin_S_m: np.ndarray = field(default_factory=_vec3)      # frozen chip-occupancy grid for recutting
    occupancy_cell_m: float = 0.0
    occupancy_mass_kg: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    occupancy_size_m: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    pile_origin_S_m: np.ndarray = field(default_factory=_vec3)           # settled height field (2-D)
    pile_cell_m: float = 0.0
    pile_height_m: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    blocked_nozzles: list[str] = field(default_factory=list)
    probe_blocked: bool = False
    wrap_mass_kg: float = 0.0
    dust_mass_rate_kg_s: float = 0.0
    heat_to_part: np.ndarray = field(default_factory=_empty2(4))       # [x_S, y_S, z_S, J] hot chips resting on the part
    heat_to_fixture_J: float = 0.0
    heat_to_coolant_J: float = 0.0
    locator_debris_m: dict[str, float] = field(default_factory=dict)  # locator id -> chip/burr height under it
    # additive (chips, docs/models/chips.md "Bus record"): the window these values describe, the grids' frames,
    # the mass terms of the ledger and the energy chips carry
    booked_until_tick: int = -1             # emissions of all windows ending at or before this base tick are booked
    occupancy_thickness_m: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))  # mass-weighted chip thickness per voxel
    pile_axes_S: np.ndarray = field(default_factory=lambda: np.eye(3))  # columns e_u, e_v, e_w of the pile grid in S (height along e_w)
    pile_base_m: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))   # w of the surface under each cell (-inf: none)
    pile_mass_kg: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    packing_power_W: dict[int, float] = field(default_factory=dict)     # friction power of the packing loads (estimate)
    flute_mass_kg: dict[int, float] = field(default_factory=dict)
    nest_fraction: dict[int, float] = field(default_factory=dict)      # long chips around the tool / space there
    part_chip_mass_kg: dict[int, float] = field(default_factory=dict)  # workpiece -> chips resting on it
    recut_mass_kg: float = 0.0              # chips removed by the recut rows of the window
    heat_to_machine_J: float = 0.0          # resting chips into the table and chip pan
    heat_to_air_J: float = 0.0              # convection to air and radiation to the enclosure
    kinetic_energy_J: float = 0.0           # 1/2 m v^2 (M) of the chips booked in the window, at separation
    in_flight_kg: float = 0.0               # ledger mass terms (architecture 12) at the end of the window
    settled_kg: float = 0.0
    conveyed_kg: float = 0.0
    adhered_kg: float = 0.0
    in_flutes_kg: float = 0.0
    dust_kg: float = 0.0
    emitted_kg: float = 0.0                 # cumulative chip mass booked (= the sum of the six terms above)
    conveyor_backlog_kg: float = 0.0        # chips in the pan / conveyor trough not yet removed


# ------------------------------------------------------------------- dynamics (writer: dynamics)

@dataclass
class DynamicsFrame:
    vibration_pp_m: float = 0.0             # peak-to-peak relative tool-work vibration this step
    chatter_index: float = 0.0              # non-forced / total vibration energy
    chatter_freq_Hz: float = 0.0
    modal_energy_change_J: float = 0.0
    damping_dissipation_J: float = 0.0      # structural damping (booked as heat to tool/holder/structure)
    process_damping_J: float = 0.0
    basis_change_J: float = 0.0             # energy difference booked when the modal basis was refreshed


# ------------------------------------------------------------------- thermal (writer: thermal)

@dataclass
class ToolHeat:
    """Per tool slot. Edge temperatures at in-cut quadrature points: [n_teeth, n_elem, n_q] with weights [n_q]."""

    edge_temp_q_K: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    flank_temp_q_K: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    q_weights: np.ndarray = field(default_factory=_empty)
    edge_cycle_amp_K: np.ndarray = field(default_factory=_empty2(0))   # [n_teeth, n_elem] per revolution
    rake_temp_mean_K: np.ndarray = field(default_factory=_empty2(0))   # for the coefficient pipeline
    body_temp_K: float = 293.15
    holder_temp_K: float = 293.15
    radial_growth_m: np.ndarray = field(default_factory=_empty2(0))   # [n_teeth, n_elem]
    length_growth_m: float = 0.0
    holder_growth_m: float = 0.0
    # added by thermal (docs/models/thermal.md):
    edge_peak_q_K: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))  # cutting-edge line (max) at the q points
    body_temp_at_elem_K: np.ndarray = field(default_factory=_empty2(0))  # [n_teeth, n_elem] 1-D body temperature there


@dataclass
class ThermalFeedback:
    tools: dict[int, ToolHeat] = field(default_factory=dict)
    spindle_temp_K: dict[str, float] = field(default_factory=dict)
    spindle_growth_M_m: dict[str, np.ndarray] = field(default_factory=dict)
    axis_drift: dict[str, np.ndarray] = field(default_factory=dict)   # axis -> 6-component link error (dx dy dz rx ry rz)
    work_temp_origin_S_m: dict[int, np.ndarray] = field(default_factory=dict)   # workpiece -> grid origin
    work_temp_cell_m: dict[int, float] = field(default_factory=dict)
    work_temp_K: dict[int, np.ndarray] = field(default_factory=dict)            # [nx, ny, nz], NaN outside material
    work_surface_temp_K: dict[int, float] = field(default_factory=dict)         # at the contact (two-scale)
    work_preheat_K: dict[int, float] = field(default_factory=dict)              # bulk temperature ahead of the cut
    work_bulk_mean_K: dict[int, float] = field(default_factory=dict)
    work_bulk_max_K: dict[int, float] = field(default_factory=dict)
    heat_to_coolant_J: float = 0.0          # convective heat into the fluid this step (coolant owns the tank)
    absorbed_J: dict[str, float] = field(default_factory=dict)   # per solver: energy actually absorbed (ledger)
    # added by thermal (docs/models/thermal.md):
    heat_to_chip_J: dict[int, float] = field(default_factory=dict)      # contact -> cutting heat leaving with the chips (chips reads it)
    heat_to_tool_J: dict[int, float] = field(default_factory=dict)      # contact -> cutting heat into the tool
    heat_to_work_J: dict[int, float] = field(default_factory=dict)      # contact -> cutting heat into the workpiece
    unplaced_heat_J: float = 0.0            # work heat that found no remaining material (left with removed material)
    pending_heat_J: dict[int, float] = field(default_factory=dict)      # workpiece -> heat accumulated for the next FV step
    contact_surface_temp_K: dict[int, float] = field(default_factory=dict)  # contact -> two-scale generated-surface peak
    contact_preheat_K: dict[int, float] = field(default_factory=dict)       # contact -> material entering the cut
    spindle_growth_abs_M_m: dict[str, np.ndarray] = field(default_factory=dict)  # physical growth vs the cold spindle
    axis_drift_strain: dict[str, float] = field(default_factory=dict)   # axis -> screw error per metre from the fixed end
    axis_drift_ref_m: dict[str, float] = field(default_factory=dict)    # axis -> position of the fixed (motor) end
    structure_drift_M_m: np.ndarray = field(default_factory=_vec3)      # tool-to-table drift of the structure (M)
    fixture_temp_K: dict[int, float] = field(default_factory=dict)      # workpiece -> lumped fixture temperature
    work_unload_mean_K: dict[int, float] = field(default_factory=dict)  # workpiece -> mean temperature at the last unload
    ledger_residual_rel: dict[str, float] = field(default_factory=dict)  # per solver: relative energy residual this step


# ------------------------------------------------------------------- coolant (writer: coolant)

@dataclass
class Jet:
    nozzle_id: str = ""
    kind: str = "flood"                     # flood | mist | mql | air | through_spindle | high_pressure
    origin_M_m: np.ndarray = field(default_factory=_vec3)
    direction_M: np.ndarray = field(default_factory=_vec3)
    speed_m_s: float = 0.0
    flow_m3_s: float = 0.0
    half_angle_rad: float = 0.0
    reach_fraction: float = 0.0


@dataclass
class CoolantFeedback:
    on: bool = False
    kind: str = "dry"
    reach_fraction: dict[int, float] = field(default_factory=dict)     # contact -> flow-weighted reach
    h_rake_W_m2K: dict[int, float] = field(default_factory=dict)
    h_flank_W_m2K: dict[int, float] = field(default_factory=dict)
    h_work_W_m2K: dict[int, float] = field(default_factory=dict)       # near the cut
    h_work_far_W_m2K: float = 0.0                                      # wetted surfaces away from the cut
    friction_factor: dict[int, float] = field(default_factory=dict)    # multiplier on dry sliding friction
    film_boiling: dict[int, bool] = field(default_factory=dict)
    fluid_temp_K: float = 293.15            # tank / supply temperature (owner: coolant)
    through_tool_exit_speed_m_s: float = 0.0
    through_tool_flow_m3_s: float = 0.0
    jets: list[Jet] = field(default_factory=list)
    shadows: dict[str, float] = field(default_factory=dict)            # occluder id -> fraction of flow blocked
    wetted_origin_S_m: np.ndarray = field(default_factory=_vec3)
    wetted_cell_m: float = 0.0
    wetted: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), bool))
    flush_pressure_Pa: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # on the pile grid


# ------------------------------------------------------------------- wear (writer: wear)

@dataclass
class ToolWear:
    VB_m: np.ndarray = field(default_factory=_empty2(0))             # [n_teeth, n_elem]
    KT_m: np.ndarray = field(default_factory=_empty2(0))
    notch_m: np.ndarray = field(default_factory=_empty2(0))
    edge_radius_m: np.ndarray = field(default_factory=_empty2(0))
    radial_wear_m: np.ndarray = field(default_factory=_empty2(0))
    axial_wear_m: np.ndarray = field(default_factory=_empty2(0))
    coating_remaining_m: np.ndarray = field(default_factory=_empty2(0))
    chipped: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), bool))
    bue_height_m: np.ndarray = field(default_factory=_empty2(0))
    broken: bool = False
    life_used_fraction: float = 0.0


@dataclass
class WearFeedback:
    tools: dict[int, ToolWear] = field(default_factory=dict)
    expired: list[int] = field(default_factory=list)                 # tool numbers whose life ended (controller)


# ------------------------------------------------------------------- structure (writer: structure)

@dataclass
class ToolModes:
    """Mass-normalised modes of the tool-holder-spindle chain at the edge elements of one tool."""

    freq_Hz: np.ndarray = field(default_factory=_empty)
    zeta: np.ndarray = field(default_factory=_empty)
    shape: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 6)))   # [mode, station, 6] in T0
    station_z_m: np.ndarray = field(default_factory=_empty)                # stations along the tool axis
    static_residual_m_per_N: np.ndarray = field(default_factory=_empty2(0))  # [3 n_station, 3 n_station]
    sections_z_m: np.ndarray = field(default_factory=_empty)                 # critical sections (neck, shank, flute root)
    sections_modulus_m3: np.ndarray = field(default_factory=_empty)          # bending section modulus there


@dataclass
class WorkModes:
    freq_Hz: np.ndarray = field(default_factory=_empty)
    zeta: np.ndarray = field(default_factory=_empty)
    origin_S_m: np.ndarray = field(default_factory=_vec3)
    cell_m: float = 0.0
    shape: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0, 0, 3)))  # [mode, nx, ny, nz, 3]
    static_compliance: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0, 3, 3)))  # residual, per cell


@dataclass
class Displacement:
    """d(X) on a grid in the material configuration: x = X + d(X)."""

    origin_S_m: np.ndarray = field(default_factory=_vec3)
    cell_m: float = 0.0
    total_m: np.ndarray = field(default_factory=_grid3)        # thermal + clamp + residual
    thermal_m: np.ndarray = field(default_factory=_grid3)
    clamp_m: np.ndarray = field(default_factory=_grid3)
    residual_m: np.ndarray = field(default_factory=_grid3)


@dataclass
class StructureFeedback:
    tools: dict[int, ToolModes] = field(default_factory=dict)
    work_modes: dict[int, WorkModes] = field(default_factory=dict)
    displacement: dict[int, Displacement] = field(default_factory=dict)
    fixture_compliance_m_per_N: dict[int, np.ndarray] = field(default_factory=dict)  # workpiece -> 3x3
    version: int = 0


# ------------------------------------------------------------------- holding (writer: holding)

@dataclass
class ToolHolding:
    pullout_m: float = 0.0
    slip_rad: float = 0.0
    unbalance_static_kg_m: np.ndarray = field(default_factory=_vec3)   # in the rotating tool frame T
    unbalance_z_T_m: float = 0.0
    unbalance_couple_kg_m2: np.ndarray = field(default_factory=_vec3)


@dataclass
class PartHolding:
    pose_delta: np.ndarray = field(default_factory=_pose)              # slip/lift/rotation (persistent)
    free: bool = False
    seated: bool = True
    location_error_S_m: np.ndarray = field(default_factory=_vec3)
    supports: list[dict] = field(default_factory=list)                 # chuck/tailstock/steady/locators for structure


@dataclass
class HoldingFeedback:
    tools: dict[int, ToolHolding] = field(default_factory=dict)
    parts: dict[int, PartHolding] = field(default_factory=dict)


# ------------------------------------------------------------------- the bus

@dataclass
class Bus:
    motion: MotionFrame = field(default_factory=MotionFrame)
    removal: RemovalFrame = field(default_factory=RemovalFrame)
    cutting: CuttingFrame = field(default_factory=CuttingFrame)
    chips: ChipFeedback = field(default_factory=ChipFeedback)
    dynamics: DynamicsFrame = field(default_factory=DynamicsFrame)
    thermal: ThermalFeedback = field(default_factory=ThermalFeedback)
    coolant: CoolantFeedback = field(default_factory=CoolantFeedback)
    wear: WearFeedback = field(default_factory=WearFeedback)
    structure: StructureFeedback = field(default_factory=StructureFeedback)
    holding: HoldingFeedback = field(default_factory=HoldingFeedback)

    WRITERS = {
        "motion": "motion", "removal": "removal", "cutting": "cutting", "chips": "chips",
        "dynamics": "dynamics", "thermal": "thermal", "coolant": "coolant", "wear": "wear",
        "structure": "structure", "holding": "holding",
    }
