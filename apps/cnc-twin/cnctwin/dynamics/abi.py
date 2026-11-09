"""Kernel layouts of the dynamics domain (docs/models/dynamics.md "Kernel ABI"; rules: architecture 5.5).

The shared layouts ModalModel and ModalState are in :mod:`cnctwin.core.abi`; this module adds what only
dynamics defines. Everything is a NamedTuple of writable, C-contiguous float64 / int64 / int32 / int8
arrays, so each kernel has exactly one compiled specialisation.

Modes of one system (the active tool of one controller path against its workpiece) are ordered
``[tool modes (is_work 0)] + [machine modes (is_work 2)] + [workpiece modes (is_work 1)]``; row r of
``ModalModel.tool_shape`` belongs to the r-th mode that is not a workpiece mode (the convention of
cnctwin.cutting.kernels.deflection_from_modal). Coordinates are mass-normalised: physical displacement
``u = sum_k phi_k q_k``, energy ``E = sum_k (qdot_k^2 + omega_k^2 q_k^2) / 2``.

Time grid: a node is the end of an L0 step in contact (the cutting kernel's force of the step, evaluated at
the step's mid angle with the deflection predicted for the step end, is the first-order-hold sample at that
node) or an L1 tick boundary out of contact. ``DynState.node_tick`` is the base tick of the current node.

* :class:`DynModel` - per system and macro step: modal frequencies and damping, the exact energy integrals
  of one L0 step, the corrector's relaxation matrix, unbalance and base-excitation projections, the work
  residual compliance and the relative shape at the contact (frozen for the step);
* :class:`DynState` - per system, carried between steps (checkpointed): force samples at the current node
  (cutting, unbalance, base), the last station wrenches (residual flexibility), the node tick, flags and the
  step's vibration samples for chatter detection;
* :class:`DynAccum` - per system and macro step: energy integrals, counters, vibration extremes, per-tick
  deflections;
* :class:`ModalDeflection` - the deflections a force kernel needs (same fields as cutting.abi.Deflection:
  the integration may wrap the same arrays as ``Deflection(*defl)`` without a copy);
* :class:`WorkGrid` - workpiece mode shapes and residual compliance on structure's grid (refresh kernel);
* :class:`AxisBase` - machine axes for the relative rigid-body acceleration (base excitation);
* :class:`DynSystem` - all of the above for one system plus scratch arrays (the kernels' single argument).
"""

from __future__ import annotations

from typing import NamedTuple

import numba
import numpy as np

from ..core.abi import N_ACC, N_ELEM_ACC, Accum, ModalModel, ModalState, RingBuffer, ring_buffer

# The constants below are read by the kernels as globals (frozen at compile time); this file is part of the
# kernel cache key of bin/tsafe and bin/twin through the @numba marker in this comment.

# ------------------------------------------------------------------------------- mode kinds (ModalModel.is_work)
KIND_TOOL = 0
KIND_WORK = 1
KIND_MACHINE = 2

# ------------------------------------------------------------------------------- base-excitation inputs
# column j of the base input vector: 0..2 relative rigid-body acceleration of the tool side with respect to the
# workpiece side in M (m/s2); 3 + r angular acceleration of rotary axis r (rad/s2)
BASE_LINEAR = 3
MAX_ROTARY = 4
N_BASE = BASE_LINEAR + MAX_ROTARY

# ------------------------------------------------------------------------------- DynModel.iscal
MI_REF_STATION = 0      # station of the vibration output, the chatter samples and the SLD FRF
MI_N_NONWORK = 1        # rows of tool_shape in use (tool + machine modes)
MI_L0_TICKS = 2         # base ticks per L0 step of ModalModel.dt_s
MI_IPO_TICKS = 3        # base ticks per L1 tick (free propagation)
MI_ROTATING = 4         # 1 rotating tool (unbalance applies)
MI_HAS_WORK_RES = 5     # 1 the workpiece residual compliance is non-zero
MI_HAS_TOOL_RES = 6     # 1 the tool residual compliance is non-zero
MI_RELAX = 7            # 1 DynModel.relax differs from the identity (the corrector multiplies by it)
N_MI = 8

# ------------------------------------------------------------------------------- DynModel.scal
MS_DT_L0_S = 0          # L0 step (= ModalModel.dt_s[0])
MS_DT_TICK_S = 1        # L1 tick
MS_TICK_S = 2           # base clock
N_MS = 4

# ------------------------------------------------------------------------------- DynState.flags
FL_HISTORY = 0          # 1: f_cut_prev holds the cutting force of the node before the current one
FL_CONTACT = 1          # 1: the current node ends an L0 step in contact
FL_STEP_ERROR = 2       # set when a kernel saw a step that does not match the model (absorb raises)
FL_STEP_SEEN = 3        # last offending step length in base ticks (diagnostic)
N_FL = 4

# ------------------------------------------------------------------------------- DynAccum.scal
DA_W_CUT_J = 0          # work of the cutting forces on the modes (exact for the first-order hold)
DA_W_UNB_J = 1          # work of the unbalance forces on the modes
DA_W_BASE_J = 2         # work of the base-excitation forces on the modes
DA_DAMPING_J = 3        # structural damping dissipation
DA_N_L0 = 4             # L0 steps in contact
DA_N_FREE = 5           # free ticks
DA_UMIN_X = 6           # extremes of the relative deflection at the reference station (T0), all nodes
DA_UMIN_Y = 7
DA_UMIN_Z = 8
DA_UMAX_X = 9
DA_UMAX_Y = 10
DA_UMAX_Z = 11
DA_W_CUT_QUAD_J = 12    # the same cutting work by the force kernel's quadrature (impulse x predicted rate)
DA_ENERGY_MAX_J = 13    # largest modal energy at a node
DA_FORCE_MAX_N = 14     # largest |F| on the tool at an evaluation
N_DA = 16
N_SCRATCH = 16
SC_OMEGA = 15           # scratch slot: |spindle speed| given to external_force (recorded with the samples)

# ------------------------------------------------------------------------------- DynSystem.act
ACT_LO = 0              # first station of the step (< 0: every station)
ACT_HI = 1              # last station of the step
ACT_PREV_LO = 2         # the range of the previous corrected step (its loads are cleared in DynState.st_force)
ACT_PREV_HI = 3
ACT_LOAD_LO = 4         # stations that may carry a load this step (the engaged ones; < 0: ACT_LO..ACT_HI)
ACT_LOAD_HI = 5
N_ACT = 6

# ------------------------------------------------------------------------------- DynAccum.tick_u columns
TU_TOOL = 0             # [0:3] tool deflection at the reference station (T0, modal + residual)
TU_WORK = 3             # [3:6] workpiece deflection at the contact (S, modal + residual)
TU_REL = 6              # [6:9] relative deflection tool - work at the reference station (T0)
N_TU = 9


class DynModel(NamedTuple):
    omega_rad_s: np.ndarray     # [nm] natural angular frequency
    zeta: np.ndarray            # [nm] damping ratio (0 <= zeta < 1)
    gram_D: np.ndarray          # [nm, 4, 4] L0 step: structural damping D = z^T gram_D z, z = [q, v, f0, s]
    gram_W: np.ndarray          # [nm, 2, 4] L0 step: work of a force part W = [f0_p, s_p] gram_W z
    relax: np.ndarray           # [nm, nm] corrector relaxation (I + K_m diag(G1[:, 0]))^-1 (identity: no estimate)
    unb: np.ndarray             # [nm, 2] Re, Im of P_k: unbalance modal force Re[P_k (W^2 - i alpha) e^(i theta)]
    base_proj: np.ndarray       # [nm, N_BASE] modal force per unit base input (BASE_* columns)
    res_work_S: np.ndarray      # [3, 3] residual compliance of the workpiece at the contact (S, m/N)
    contact_shape: np.ndarray   # [nm, 3] relative tool-work displacement shape at the reference station (T0)
    row: np.ndarray             # [nm] int32 row of the mode in ModalModel.tool_shape (non-work) or work_shape_at_contact
    R_S_T0: np.ndarray          # [3, 3] columns: T0 axes in S (workpiece residual into T0)
    iscal: np.ndarray           # [N_MI] int64
    scal: np.ndarray            # [N_MS]


class DynState(NamedTuple):
    f_cut: np.ndarray           # [nm] cutting modal force sample at the current node
    f_cut_prev: np.ndarray      # [nm] at the node before (linear extrapolation of the predictor)
    f_unb: np.ndarray           # [nm] unbalance modal force at the current node
    f_base: np.ndarray          # [nm] base-excitation modal force at the current node
    st_force: np.ndarray        # [nst, 6] station wrenches (T0) of the last evaluation (residual, one step lag)
    work_force_S: np.ndarray    # [3] force on the tool (S) of the last evaluation (work residual)
    node_tick: np.ndarray       # [1] int64 base tick of the current node
    flags: np.ndarray           # [N_FL] int64
    samp_tick: np.ndarray       # [cap] int64 base tick of each vibration sample (L0 nodes in contact)
    samp_u: np.ndarray          # [cap, 3] relative deflection at the reference station (T0)
    samp_w: np.ndarray          # [cap] |spindle speed| at the sample (the value external_force was given)
    samp_n: np.ndarray          # [2] int64 used, overflowed (samples dropped)


class DynAccum(NamedTuple):
    scal: np.ndarray            # [N_DA]
    tick_u: np.ndarray          # [n_ticks, N_TU] deflections at each L1 tick end of the step
    tick_n: np.ndarray          # [1] int64 rows written


class ModalDeflection(NamedTuple):
    """Same fields, order and shapes as cnctwin.cutting.abi.Deflection."""

    u_st_T0_m: np.ndarray       # [nst, 6] tool deflection at the stations: dx dy dz rx ry rz (T0)
    v_st_T0_m_s: np.ndarray     # [nst, 6] its rate (modal part)
    u_work_S_m: np.ndarray      # [3] elastic workpiece deflection at the contact (S): modes + residual
    v_work_S_m_s: np.ndarray    # [3] its rate (modal part)


class WorkGrid(NamedTuple):
    """Workpiece modes on structure's grid (cell centres origin + (i + 1/2) cell), refreshed per macro step."""

    shape: np.ndarray           # [nm_work, nx, ny, nz, 3] mass-normalised shapes (S)
    compliance: np.ndarray      # [nx, ny, nz, 3, 3] residual compliance per cell (S, m/N)
    origin_S_m: np.ndarray      # [3]
    cell_m: np.ndarray          # [1] (0: no grid)
    R_S_T0: np.ndarray          # [3, 3] columns: T0 axes in S (work rows of DynModel.contact_shape)
    extra_S: np.ndarray         # [3, 3] compliance added at every point (fixture, when the part has no modes)


class AxisBase(NamedTuple):
    """Machine axes for base excitation: link direction in M, side (+1 moves the tool, -1 the workpiece,
    0 ignored), rotary flag and the base-input column of a rotary axis (-1 none)."""

    dir_M: np.ndarray           # [na, 3]
    side: np.ndarray            # [na]
    rotary: np.ndarray          # [na] int8
    column: np.ndarray          # [na] int32


class DynSystem(NamedTuple):
    model: ModalModel           # core.abi: Phi, G0, G1 at the L0 step, shapes, residual, dt
    state: ModalState           # core.abi: x [nm, 2], f_prev [nm] = total force sample at the current node
    dm: DynModel
    ds: DynState
    da: DynAccum
    defl: ModalDeflection       # output of modal_deflection (scratch, overwritten every call)
    xp: np.ndarray              # [nm, 2] predicted state at the next node (scratch)
    fe: np.ndarray              # [2, nm] external modal force at the next node: unbalance, base (scratch)
    fx: np.ndarray              # [nm] modal force of the evaluated cutting wrenches (scratch)
    scratch: np.ndarray         # [N_SCRATCH] (scratch)
    act: np.ndarray             # [N_ACT] int64 stations of the step (additive): deflections are computed and loads
                                #   projected for stations ACT_LO..ACT_HI only (the caller's engaged stations with
                                #   their neighbours; every other station has zero load); ACT_LO < 0: all stations


# ------------------------------------------------------------------------------- constructors
def _f(*shape) -> np.ndarray:
    return np.zeros(shape, dtype=np.float64)


def _i64(*shape) -> np.ndarray:
    return np.zeros(shape, dtype=np.int64)


def empty_modal_model(nm: int = 1, nst: int = 1, nm_work: int = 1) -> ModalModel:
    nm = max(int(nm), 1)
    nst = max(int(nst), 1)
    return ModalModel(_f(nm, 2, 2), _f(nm, 2), _f(nm, 2), _f(nm, nst, 6), _f(max(int(nm_work), 1), 3),
                      np.zeros(nm, np.int8), np.full(nm, -1, np.int32), _f(3 * nst, 3 * nst), _f(1))


def empty_modal_state(nm: int = 1) -> ModalState:
    nm = max(int(nm), 1)
    return ModalState(_f(nm, 2), _f(nm))


def empty_dyn_model(nm: int = 1) -> DynModel:
    nm = max(int(nm), 1)
    return DynModel(_f(nm), _f(nm), _f(nm, 4, 4), _f(nm, 2, 4), np.eye(nm), _f(nm, 2), _f(nm, N_BASE), _f(3, 3),
                    _f(nm, 3), np.zeros(nm, np.int32), np.eye(3), _i64(N_MI), _f(N_MS))


def empty_dyn_state(nm: int = 1, nst: int = 1, capacity: int = 16) -> DynState:
    nm = max(int(nm), 1)
    nst = max(int(nst), 1)
    cap = max(int(capacity), 1)
    return DynState(_f(nm), _f(nm), _f(nm), _f(nm), _f(nst, 6), _f(3), _i64(1), _i64(N_FL), _i64(cap), _f(cap, 3),
                    _f(cap), _i64(2))


def empty_dyn_accum(n_ticks: int = 1) -> DynAccum:
    scal = _f(N_DA)
    reset_extremes(scal)
    return DynAccum(scal, _f(max(int(n_ticks), 1), N_TU), _i64(1))


def reset_extremes(scal: np.ndarray) -> None:
    for c in (DA_UMIN_X, DA_UMIN_Y, DA_UMIN_Z):
        scal[c] = np.inf
    for c in (DA_UMAX_X, DA_UMAX_Y, DA_UMAX_Z):
        scal[c] = -np.inf


def empty_deflection(nst: int = 1) -> ModalDeflection:
    nst = max(int(nst), 1)
    return ModalDeflection(_f(nst, 6), _f(nst, 6), _f(3), _f(3))


def empty_work_grid(nm_work: int = 1) -> WorkGrid:
    return WorkGrid(_f(max(int(nm_work), 1), 1, 1, 1, 3), _f(1, 1, 1, 3, 3), _f(3), _f(1), np.eye(3), _f(3, 3))


def empty_axes(n_axes: int = 1) -> AxisBase:
    n = max(int(n_axes), 1)
    return AxisBase(_f(n, 3), _f(n), np.zeros(n, np.int8), np.full(n, -1, np.int32))


def empty_system(nm: int = 1, nst: int = 1, nm_work: int = 1, n_ticks: int = 1, capacity: int = 16) -> DynSystem:
    nm = max(int(nm), 1)
    return DynSystem(empty_modal_model(nm, nst, nm_work), empty_modal_state(nm), empty_dyn_model(nm),
                     empty_dyn_state(nm, nst, capacity), empty_dyn_accum(n_ticks), empty_deflection(nst), _f(nm, 2),
                     _f(2, nm), _f(nm), _f(N_SCRATCH), all_stations())


def all_stations() -> np.ndarray:
    """DynSystem.act for every station (the default)."""
    return np.full(N_ACT, -1, np.int64)


def core_accum(n_ticks: int = 1, n_axes: int = 1, nt: int = 1, ne: int = 1) -> Accum:
    """A core.abi.Accum with the shapes the cutting kernels use (dynamics writes ACC_DAMPING_J and
    ACC_UNBALANCE_WORK_J into it while in contact)."""
    return Accum(_f(N_ACC), _f(N_ELEM_ACC, max(nt, 1), max(ne, 1)), _f(max(n_ticks, 1), 4, 6),
                 _f(max(n_ticks, 1), max(n_axes, 1)), _f(max(n_ticks, 1)))


def is_kernel_layout(nt) -> bool:
    """Every array of a (nested) layout is writable and C-contiguous (kernel argument rule)."""
    for value in nt:
        if isinstance(value, tuple):
            if not is_kernel_layout(value):
                return False
        elif not (isinstance(value, np.ndarray) and value.flags.c_contiguous and value.flags.writeable):
            return False
    return True


# ------------------------------------------------------------------------------- Numba types
MODEL_T = numba.typeof(empty_modal_model())
MSTATE_T = numba.typeof(empty_modal_state())
SYSTEM_T = numba.typeof(empty_system())
DEFL_T = numba.typeof(empty_deflection())
GRID_T = numba.typeof(empty_work_grid())
AXES_T = numba.typeof(empty_axes())
ACCUM_T = numba.typeof(core_accum())
RING_T = numba.typeof(ring_buffer(2, 1))

__all__ = [name for name in dir() if name.isupper() or name in (
    "DynModel", "DynState", "DynAccum", "ModalDeflection", "WorkGrid", "AxisBase", "DynSystem", "ModalModel",
    "ModalState", "Accum", "RingBuffer", "empty_modal_model", "empty_modal_state", "empty_dyn_model",
    "empty_dyn_state", "empty_dyn_accum", "empty_deflection", "empty_work_grid", "empty_axes", "empty_system",
    "core_accum", "is_kernel_layout", "reset_extremes", "all_stations")]
