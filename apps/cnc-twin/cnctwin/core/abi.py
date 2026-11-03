"""Kernel ABI: the array layouts every hot-loop kernel agrees on (docs/architecture.md 5.5).

Everything here is a NamedTuple of writable, C-contiguous NumPy arrays (Numba types them
concretely, caches them, and never needs a jitclass). Owners build them in Python at L2 and
the compiled loop only reads (or, for state and accumulators, writes) them.

Shapes: ``nt`` teeth, ``ne`` elements per tooth (padded; ``elem_valid``), ``nst`` stations
along the edge shared by all teeth (station k of every tooth sits at the same arc-length
position, so teeth sweep the same circle there), ``nb`` angle bins, ``ng`` coefficient
groups, ``nh`` chip-thickness nodes, ``nm`` modes.

Domain-internal layouts (the dexel field, the tool envelope, the kinematic chain and servo
state) are defined by their owners in ``<domain>/abi.py`` and passed through unchanged.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------- edge element kinds
EDGE_PERIPHERAL = 0     # side cutting edge of an end mill / flank of a reamer
EDGE_END = 1            # end (bottom) cutting edge, kappa near 0
EDGE_CORNER = 2         # corner radius / ball / nose radius
EDGE_LIP = 3            # drill lip
EDGE_CHISEL = 4         # drill chisel edge (indentation + extrusion)
EDGE_MARGIN = 5         # drill/reamer margin (rubbing only)
EDGE_THREAD_FLANK = 6   # tap / thread mill / threading insert flank
EDGE_WIPER = 7          # face-mill or turning wiper flat
EDGE_INSERT = 8         # straight edge of an indexable insert
EDGE_NOSE = 9           # turning / boring nose radius
EDGE_KINDS = ("peripheral", "end", "corner", "lip", "chisel", "margin", "thread_flank", "wiper", "insert", "nose")


class EdgeModel(NamedTuple):
    """Cutting edges of one tool instance (built by cnctwin.tooling).

    The element midpoint of tooth j, element k, at spindle angle theta (right-handed about +z_T)
    is, in the rotating tool frame T: angle psi = sign * theta + tooth_phase[j] + lag[j, k]
    (sign = spindle direction factor applied by the kernel), radius r[j, k], height z[j, k].
    Runout and tilt are applied on top by the kernel from runout_T_m / tilt_rad.
    """

    tooth_phase_rad: np.ndarray     # [nt]
    elem_z_m: np.ndarray            # [nt, ne] along +z_T from the tool tip
    elem_r_m: np.ndarray            # [nt, ne] distance from the tool axis
    elem_lag_rad: np.ndarray        # [nt, ne] helix / variable-helix lag
    elem_len_m: np.ndarray          # [nt, ne] edge length of the element (arc length)
    elem_db_m: np.ndarray           # [nt, ne] chip width (finite for kappa = 0)
    elem_kappa_rad: np.ndarray      # [nt, ne] axial immersion: 0 = end edge, pi/2 = peripheral
    elem_rake_rad: np.ndarray       # [nt, ne] normal rake
    elem_incl_rad: np.ndarray       # [nt, ne] inclination (helix / lead)
    elem_clear_rad: np.ndarray      # [nt, ne] clearance
    elem_edge_radius_m: np.ndarray  # [nt, ne] hone radius (grows with wear at L2)
    elem_kind: np.ndarray           # [nt, ne] int32, EDGE_*
    elem_group: np.ndarray          # [nt, ne] int32, coefficient group
    elem_station: np.ndarray        # [nt, ne] int32, shared station index (surface memory row)
    elem_valid: np.ndarray          # [nt, ne] int8
    runout_T_m: np.ndarray          # [3] axis offset at the tip in T (static radial runout)
    tilt_rad: np.ndarray            # [2] axis tilt about x_T and y_T
    station_z_m: np.ndarray         # [nst]
    station_r_m: np.ndarray         # [nst]
    # Added by tooling (additive, docs/models/tooling.md "EdgeModel conventions"): the tool's hand
    # h = -1 right-hand (cuts with omega < 0, M03), +1 left-hand. tooth_phase/lag are in the
    # cutting sense, so the right-handed angle of an element in T is alpha = theta + h * (phase +
    # lag); psi above is h * alpha. Tooling always sets it; the default serves fixtures only.
    hand_sign: np.ndarray = np.array([-1.0])   # [1] float64


class CoeffTable(NamedTuple):
    """ProcessCoefficients in kernel form (cnctwin.core.bus.ProcessCoefficients)."""

    h_grid_m: np.ndarray            # [nh] log-spaced, ascending
    K_Pa: np.ndarray                # [ng, nh, 3] tangential, radial, axial specific force on h
    Ke_N_per_m: np.ndarray          # [ng, 3] edge forces
    frac: np.ndarray                # [ng, nh, 3] shear->chip, rake->tool, flank->tool
    h_min_m: np.ndarray             # [ng]
    K_sp_N_per_m3: np.ndarray       # [ng] indentation coefficient (process damping, ploughing)
    flank_coeff_N_per_m2: np.ndarray  # [nt, ne] flank-wear ploughing per unit wear-land area
    v_min_m_s: np.ndarray           # [1] below: indentation model instead of shearing


class SurfaceMemory(NamedTuple):
    """Angle-domain memory of the surface the last effective edge passage left (owner: cutting).

    Row = station, column = angle bin in the non-rotating tool frame T0. h of an element is
    (p_e(t) - point[station, bin]) . n_e. A bin is refilled from the field when stale.
    """

    point_S_m: np.ndarray           # [nst, nb, 3] in the material configuration
    tick: np.ndarray                # [nst, nb] int64, base tick of the passage
    valid: np.ndarray               # [nst, nb] int8


class EngagedIntervals(NamedTuple):
    """Outer-boundary gating per station: angle intervals (T0, right-handed) where material exists."""

    lo_rad: np.ndarray              # [nst, n_max]
    hi_rad: np.ndarray              # [nst, n_max]
    count: np.ndarray               # [nst] int32
    tick: np.ndarray                # [1] int64, when computed


class ModalModel(NamedTuple):
    """Relative tool-workpiece modal model for one contact (owner: dynamics).

    First-order-hold exact discretisation at the L0 step: x_{n+1} = Phi x_n + G0 f_n + G1 f_{n+1},
    with x = (q, qdot) per mode and f the modal force.
    """

    Phi: np.ndarray                 # [nm, 2, 2]
    G0: np.ndarray                  # [nm, 2]
    G1: np.ndarray                  # [nm, 2]
    tool_shape: np.ndarray          # [nm_tool, nst, 6] mass-normalised, T0 (translations, rotations)
    work_shape_at_contact: np.ndarray  # [nm_work, 3] at the current contact point (refreshed per tick)
    is_work: np.ndarray             # [nm] int8: 0 tool mode, 1 workpiece mode, 2 machine mode
    work_index: np.ndarray          # [nm] int32 index into work_shape_at_contact rows (or -1)
    static_residual: np.ndarray     # [3 nst, 3 nst] residual flexibility (m/N) of the tool, T0
    dt_s: np.ndarray                # [1]


class ModalState(NamedTuple):
    x: np.ndarray                   # [nm, 2]
    f_prev: np.ndarray              # [nm] modal force at the previous step (first-order hold)


# ---------------------------------------------------------------- accumulators of one contact
ACC_SHAFT_WORK_J = 0
ACC_FEED_WORK_J = 1
ACC_HEAT_SHEAR_J = 2
ACC_HEAT_RAKE_J = 3
ACC_HEAT_FLANK_J = 4
ACC_PLOUGHING_J = 5
ACC_CHIP_VOLUME_M3 = 6
ACC_IN_CUT_S = 7
ACC_TORQUE_PEAK_NM = 8
ACC_POWER_PEAK_W = 9
ACC_FORCE_PEAK_N = 10
ACC_RECUT_MASS_KG = 11
ACC_DAMPING_J = 12
ACC_PROCESS_DAMPING_J = 13
ACC_UNBALANCE_WORK_J = 14
N_ACC = 16

ELEM_IN_CUT_S = 0
ELEM_RAKE_SLIDING_M = 1
ELEM_FLANK_SLIDING_M = 2
ELEM_INT_SIGMA_V_RAKE = 3
ELEM_INT_SIGMA_V_FLANK = 4
ELEM_INT_H_DT = 5
ELEM_H_MAX = 6
ELEM_ENTRIES = 7
N_ELEM_ACC = 8

WRENCH_TOOL_MEAN = 0
WRENCH_TOOL_PEAK = 1
WRENCH_WORK_MEAN = 2
WRENCH_WORK_PEAK = 3


class Accum(NamedTuple):
    scalars: np.ndarray             # [N_ACC]
    elem: np.ndarray                # [N_ELEM_ACC, nt, ne]
    tick_wrench: np.ndarray         # [n_ticks, 4, 6]
    tick_axis_force_N: np.ndarray   # [n_ticks, n_axes]
    tick_spindle_torque_Nm: np.ndarray  # [n_ticks]


class EventBuffer(NamedTuple):
    """Rows of float64 written by kernels; ``n[0]`` rows used. Overflow sets ``n[1]`` (the loop
    then splits the macro step and replays it with a larger buffer, deterministically)."""

    rec: np.ndarray                 # [cap, width]
    n: np.ndarray                   # [2] int64: used, overflowed


def event_buffer(capacity: int, width: int) -> EventBuffer:
    return EventBuffer(np.zeros((capacity, width)), np.zeros(2, np.int64))


def contiguous(array: np.ndarray, dtype=None) -> np.ndarray:
    """Writable C-contiguous copy when needed (read-only arrays would force a second compile)."""
    out = np.ascontiguousarray(array, dtype=dtype)
    if not out.flags.writeable:
        out = out.copy()
    return out


# ---------------------------------------------------------------- fast-tier ring buffer (owner: core.recorder)
# Added by the run infrastructure (additive; docs/architecture.md 10, docs/formats.md "Run directory").

class RingBuffer(NamedTuple):
    """The fast (L0) tier's ring buffer, filled by the compiled loop and read at L2 by the recorder.

    Row ``s = head[0] % n`` takes the next sample: ``cnctwin.core.recorder.ring_begin(ring, tick)``
    stamps the tick, fills the row with NaN and advances ``head``; the domains' kernels then write
    their channels' columns (``Recorder.fast_columns(name)`` gives the column range). ``head`` counts
    every row pushed since the start of the run (it never wraps), so the recorder knows which rows
    are new and whether any were overwritten before it read them.
    """

    data: np.ndarray                # [n, n_columns] float64, NaN where a channel has no value
    tick: np.ndarray                # [n] int64 base tick of each row (-1: never written)
    head: np.ndarray                # [1] int64 rows pushed since the start of the run


def ring_buffer(n_rows: int, n_columns: int) -> RingBuffer:
    """An empty ring of ``n_rows`` samples of ``n_columns`` fast columns."""
    if n_rows < 1 or n_columns < 0:
        raise ValueError(f"ring buffer needs n_rows >= 1 and n_columns >= 0 (got {n_rows}, {n_columns})")
    return RingBuffer(np.full((int(n_rows), int(n_columns)), np.nan), np.full(int(n_rows), -1, np.int64),
                      np.zeros(1, np.int64))
