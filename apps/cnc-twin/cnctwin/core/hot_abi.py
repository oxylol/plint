"""Layouts of the compiled hot loop (core.hotloop.run_macro; docs/models/hotloop.md "Layouts").

The hot loop composes the domains' kernels; its arguments are the domains' own kernel layouts plus the
layouts below (all NamedTuples of writable C-contiguous arrays, so ``run_macro`` has exactly one compiled
specialisation, architecture 5.5):

* :class:`CutArgs` - the cutting kernel arguments of one contact (cnctwin.cutting.domain.ContactInputs
  without its Python fields);
* :class:`ContactGeo` - per contact and macro step, the geometry the loop needs beyond the kernels' own:
  station circles (gating, refill planes), the tooth slabs' angular offsets (removal generation instants),
  the path's axis map (base excitation), the turning polar view's update window;
* :class:`ContactState` - per contact, carried between calls and macro steps (checkpointed by the
  wrapper): slab crossing phases, the recorded slab axis points of the tooth passes in progress, the
  contact flags;
* :class:`HotContact` - one contact: parameters, CutArgs, the dynamics system (cnctwin.dynamics.abi
  DynSystem), ContactGeo, ContactState, the removal work spec and small scratch;
* :class:`HotGlobal` - per call: parameters, the axis maps, the carried L1 loads, the tick-start state
  of the L0 interpolation (mid-tick resume), the structure displacement grid d(X), the probe stylus, the
  result and counters.

Contacts are passed as a homogeneous tuple of :data:`MAX_CONTACTS` HotContact (inactive ones padded with
placeholders of the same types); removal's per-workpiece tuples are padded to :data:`MAX_WORK` and its
per-path tuples to :data:`MAX_PATHS`. Raising a maximum is one recompilation of run_macro.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numba import typeof

from ..cutting import abi as CA
from ..dynamics import abi as DA
from ..motion import abi as MA
from ..removal import abi as RA
from ..removal import nbt as RN
from .abi import EventBuffer, RingBuffer, ring_buffer

# The constants below are read by the hot loop as globals (frozen at compile time); this file is part of the
# kernel cache key of bin/tsafe and bin/twin through the @numba marker in this comment.

MAX_CONTACTS = 2            # contacts (paths with a tool) per step; one contact per controller path
MAX_PATHS = 2               # removal's per-path tuples (profiles, slabs, accumulators) are padded to this
MAX_WORK = 2                # workpieces (main and sub spindle)
WORK_SIZE = 112             # HotContact.work (core.hot_kernels WK_SIZE)
REFILL_PAIRS = 4096         # refill planes one L0 step may set before clearing all (HotContact.planes)

# ------------------------------------------------------------------------------- HotContact.ipar
CI_ACTIVE = 0               # 1: the contact takes part in this call
CI_PATH = 1                 # controller path (motion path index, removal path index, cutting contact index)
CI_SLOT = 2                 # tool slot
CI_WORK = 3                 # workpiece (removal field index)
CI_KIND = 4                 # removal.abi TK_* (envelope, turning, helix)
CI_ROTATING = 5             # 1 rotating tool
CI_TOOL_SPINDLE = 6         # spindle turning the tool (-1 none)
CI_WORK_SPINDLE = 7         # spindle turning the workpiece (-1 none)
CI_N_TEETH = 8
CI_N_ELEM = 9               # elements per tooth of the EdgeModel
CI_NST = 10                 # stations
CI_L0_TICKS = 11            # L0 stride (base ticks); divides ipo_ticks
CI_DYN = 12                 # 1: the dynamics system is live (modes), 0: rigid tool and workpiece
CI_RING_FORCE = 13          # first fast column of hotloop.force_T0_N for this path (-1: not recorded)
CI_RING_DYN = 14            # first fast column of dynamics.vibration_rel_m for this path (-1)
CI_RING_TORQUE = 15         # fast column of hotloop.torque_Nm for this path (-1)
CI_NS_MAX = 16              # slabs of the longest tooth
CI_N_AXES = 17              # axes of the path (dynamics AxisBase order)
CI_POLAR = 18               # 1: the workpiece has a polar field (turning)
CI_STALE_BEFORE = 19        # cutting's SI_STALE_BEFORE from L2 (the loop raises it after a rapid)
CI_PROBE = 20               # 1: a probe stylus profile is set for this path (G31 sweep)
N_CI = 24

# ------------------------------------------------------------------------------- HotContact.fpar
CF_HAND = 0                 # hand sign h of the tool (-1 right-hand)
CF_R_CUT = 1                # largest cutting radius (m)
CF_REACH = 2                # refill-plane search distance (m)
CF_LATTICE = 3              # level-0 lattice spacing of the workpiece field (m)
CF_WORK_SIGN = 4            # +1 / -1: the work spindle's +z axis is +/- S z (turning torque)
CF_GATE_TOL = 5             # gating parameters (removal fparams RF_GATE_SAMPLE, RF_GATE_TOL, RF_GATE_MARGIN)
CF_GATE_SAMPLE = 6
CF_GATE_MARGIN = 7
CF_SECTOR_MARGIN = 8
CF_COLLISION_TOL = 9
CF_FEED_FRAC = 10           # |v_xy| / |v| below which the feed direction is taken as the T0 x axis (axial feed)
CF_POLAR_SZ = 11            # turning: z_T along +S z (+1) or -S z (-1)
CF_POLAR_SX = 12            # turning: x_T along -e_r (-1 external) or +e_r (+1 internal)
CF_MOUNT_X = 13             # tool mount: tooling's T0 = the chain's T0 with its axes scaled by (mx, my, mz), a proper
CF_MOUNT_Y = 14             # rotation (identity, or pi about one axis): non-rotating tools follow tooling's turning
CF_MOUNT_Z = 15             # convention (x_T into the work); docs/models/hotloop.md "Turning tool frame"
N_CF = 16

# ------------------------------------------------------------------------------- ContactState.ival
CS_IN_CONTACT = 0           # 1: broad phase in contact at the last tick
CS_FIRST = 1                # 1: the next L0 step is the first of an engagement (phases are initialised)
CS_PASSES = 2               # tooth passes executed
CS_LAST_RAPID = 3           # base tick of the last rapid tick of the path (memory staleness)
CS_ENGAGED_TICK = 4         # 1: at least one element was engaged during the current tick
CS_PB_PENDING = 5           # turning: polar bins waiting for the view update (count)
CS_REFILLS = 6              # refill planes computed by the loop
CS_REFILL_MISS = 7          # refill planes not found (no surface within reach)
CS_L0_STEPS = 8             # L0 steps in contact
CS_SUBTRACT_CALLS = 9
CS_FREE_TICKS = 10          # dynamics free ticks
CS_OVERFLOW_TOOTH = 11      # tooth whose pass hit a pool overflow (-1 none): repeated at resume
CS_GATINGS = 12             # engaged-interval refreshes (passes and timer)
CS_FORCED = 13              # passes forced by the staleness fallback (a slab missed its generation angle)
N_CS = 16

# ContactState.fval
CSF_BETA = 0                # turning: unwrapped angle of the tool's reference plane in S at the last step end
CSF_THETA_PREV = 1          # unwrapped tool spindle angle at the end of the last step
CSF_GATE_ANGLE = 2          # spindle angle turned since the last engaged-interval refresh (rad)
N_CSF = 4

# ------------------------------------------------------------------------------- HotGlobal.ipar
GI_N_CONTACTS = 0
GI_TICK0 = 1                # base tick of the macro step start
GI_IPO_TICKS = 2
GI_MACRO_IPO = 3
GI_RESUME = 4               # 1: resume inside tick GI_RESUME_K at contact GI_RESUME_C, L0 step GI_RESUME_S
GI_RESUME_K = 5
GI_RESUME_C = 6
GI_RESUME_S = 7
GI_N_AXES = 8               # machine axes (motion global order)
GI_N_SPINDLES = 9
GI_N_PATHS = 10             # controller paths of the machine
GI_PROBE_RINGS = 11
GI_PROBE_AROUND = 12
GI_ROWS_SHOCK = 13          # rows one L0 step may write per cutting buffer (headroom rule, docs/models/hotloop.md)
GI_ROWS_SURFACE = 14
GI_ROWS_DEPOSIT = 15
GI_ROWS_RECUT = 16
GI_ROWS_PASS = 17
GI_ROWS_REFILL = 18
GI_ROWS_REMOVAL = 19        # rows one tooth pass / L0 step may write into removal's event buffer
GI_HAS_DISP = 20            # 1: the structure displacement grid d(X) is set
N_GI = 24

# HotGlobal.fpar
GF_TICK_S = 0
GF_PROBE_TOL = 1
N_GF = 4

# ------------------------------------------------------------------------------- HotGlobal.result
RES_CODE = 0                # motion sync code (SYNC_*) or HOT_* split code
RES_TICK = 1                # tick index of the return
RES_PATH = 2                # path of the first sync (-1)
RES_CONTACT = 3             # contact of a split (-1)
RES_REASON = 4              # split reason (SPLIT_*)
RES_DETAIL = 5              # buffer index / workpiece of the split
N_RES = 8

HOT_SPLIT = 100             # returned code: the loop stopped inside a tick (resume after the wrapper acted)
SPLIT_DRAIN = 1             # an event buffer lacks the headroom of one L0 step: drain and resume
SPLIT_POOL = 2              # removal pool overflow: grow the pool and resume (the pass is repeated)
SPLIT_POLAR = 3             # polar field overflow: grow the inline segments and resume (the cut is repeated)

# buffer indices of a drain split (RES_DETAIL)
BUF_SHOCK = 0
BUF_SURFACE = 1
BUF_DEPOSIT = 2
BUF_RECUT = 3
BUF_PASS = 4
BUF_REFILL = 5
BUF_REMOVAL = 6

# ------------------------------------------------------------------------------- HotGlobal.count
CNT_L0 = 0                  # L0 steps (all contacts)
CNT_TICKS = 1               # L1 ticks
CNT_PASSES = 2              # tooth passes
CNT_REFILLS = 3             # refill planes
CNT_POLAR_CUTS = 4
CNT_HELIX_CALLS = 5
CNT_FREE = 6                # dynamics free ticks
CNT_SPLITS = 7
CNT_PROBE_SWEEPS = 8
CNT_BROAD = 9               # broad-phase calls
N_CNT = 12

# hot-loop fast channels (core.hotloop.channels)
FAST_FORCE = "hotloop.force_T0_N"       # [MAX_PATHS x 3] tool force (T0), L0 step mean
FAST_TORQUE = "hotloop.torque_Nm"       # [MAX_PATHS] process torque about the spindle axis


class CutArgs(NamedTuple):
    """cut_step / turn_step arguments of one contact (the arrays of cutting's ContactInputs)."""

    tool: CA.CutTool
    table: CA.CutTable
    mem: object                 # core.abi.SurfaceMemory
    iv: object                  # core.abi.EngagedIntervals
    refill: CA.Refill
    step: np.ndarray
    istep: np.ndarray
    pose: CA.StepPose
    defl: CA.Deflection
    occ: CA.CutOccupancy
    pack: CA.CutPacking
    axes: CA.AxisMap
    st: CA.CutState
    acc: object                 # core.abi.Accum
    cacc: CA.CutAccum
    out: CA.StepOut
    ev: CA.CutEvents
    polar: CA.PolarView


class ContactGeo(NamedTuple):
    st_z: np.ndarray            # [nst] station height (z_T)
    st_r: np.ndarray            # [nst] station radius (first valid element)
    st_len: np.ndarray          # [nst] engaged-edge length of the station (sum over its elements)
    st_active: np.ndarray       # [nst] int8 1: the station has valid elements
    st_nr: np.ndarray           # [nst] radial / axial components of the station's chip-thickness normal (T0 meridian)
    st_nz: np.ndarray
    slab_a0: np.ndarray         # [nt, ns_max] right-handed T0 angle offset h (phase + lag) of each slab's mid height
    slab_zm: np.ndarray         # [nt, ns_max] slab mid height (z_T)
    slab_ptr: np.ndarray        # [nt * ns_max + 1] int64 stations of slab (j, k): slab_sts[slab_ptr[j ns + k]:...]
    slab_sts: np.ndarray        # [m] int64 (the stations whose height lies in the slab; the nearest when none)
    path_axes: np.ndarray       # [na_path] int64 global motion axis of each path axis (dynamics AxisBase order)
    cut_axes: np.ndarray        # [na_cut] int64 global motion axis of each cutting AxisMap row (-1 none)


class ContactState(NamedTuple):
    gamma: np.ndarray           # [nt, ns_max] generation angle of each slab (T0, right-handed): the exit of its
                                #   station's main engaged interval in the direction of rotation (NaN: not engaged)
    slab_c: np.ndarray          # [nt, ns_max, 3] slab axis points (S, material configuration) at generation
    slab_done: np.ndarray       # [nt, ns_max] int8 1: slab recorded since the tooth's last pass
    last_pass: np.ndarray       # [nt] int64 base tick of the tooth's last pass
    polar_win: np.ndarray       # [8] int64 turning: polar view rows waiting for the tool to leave their bins
    ival: np.ndarray            # [N_CS] int64
    fval: np.ndarray            # [N_CSF]


class HotContact(NamedTuple):
    ipar: np.ndarray            # [N_CI] int64
    fpar: np.ndarray            # [N_CF]
    cut: CutArgs
    sys: object                 # cnctwin.dynamics.abi.DynSystem
    base: object                # cnctwin.dynamics.abi.AxisBase of the path
    grid: object                # cnctwin.dynamics.abi.WorkGrid
    offset_S: np.ndarray        # [3] contact point minus tool tip (S) for the workpiece mode shapes
    geo: ContactGeo
    cst: ContactState
    spec: object                # removal.abi.SubtractSpec work spec (arrays sized for the longest tooth)
    stats: np.ndarray           # [4] gating statistics
    gate: np.ndarray            # [4] gate parameters (RF_GATE_SAMPLE, RF_GATE_TOL, RF_GATE_MARGIN, RF_SECTOR_MARGIN)
    gate3: np.ndarray           # [3] the first three (engaged_intervals)
    work: np.ndarray            # [WK_SIZE] frames and vectors of the step (core.hot_kernels WK_*)
    base0: np.ndarray           # [N_BASE] base-excitation input at the tick start
    base1: np.ndarray           # [N_BASE] at the tick end
    acc_axis: np.ndarray        # [na_path] axis accelerations of the path (tick)
    poses: np.ndarray           # [N_FK, 4, 4] FK scratch of the L0 pose
    fkw: np.ndarray             # [6, 4, 4] FK work
    qv: np.ndarray              # [NA] interpolated axis values
    polar_step: np.ndarray      # [N_PS] removal polar step values (turning)
    plane: np.ndarray           # [8] surface-plane output (refill planes)
    planes: np.ndarray          # [REFILL_PAIRS + 1, 2] int64 (row, bin) of the refill planes of the current step
    probe: object               # removal.abi.Profile of the stylus (G31 sweep; empty: no probe)
    probe_out: np.ndarray       # [8] sweep_contact output


class HotGlobal(NamedTuple):
    ipar: np.ndarray            # [N_GI] int64
    fpar: np.ndarray            # [N_GF]
    load_force: np.ndarray      # [NA] cutting load per machine axis of the last tick (N along +axis)
    load_torque: np.ndarray     # [S] cutting torque per spindle against +omega of the last tick (N m)
    q0: np.ndarray              # [NA] axis values at the start of the current tick (L0 interpolation)
    th0: np.ndarray             # [S] spindle angles at the start of the current tick (unwrapped)
    w0: np.ndarray              # [S] spindle speeds at the start of the current tick
    tip0: np.ndarray            # [P, 3] tool tip in S at the start of the current tick (probe sweep)
    disp_origin: np.ndarray     # [3] structure displacement grid d(X) in S (cell centres origin + (i + 1/2) cell)
    disp_cell: np.ndarray       # [1] (0: no grid)
    disp: np.ndarray            # [nx, ny, nz, 3]
    result: np.ndarray          # [N_RES] int64
    count: np.ndarray           # [N_CNT] int64
    tick_flags: np.ndarray      # [T, P] int8 1: the path's tool was engaged during the tick (out.in_cut)


# ------------------------------------------------------------------------------- example instances (types)
def _f(*shape) -> np.ndarray:
    return np.zeros(shape, dtype=np.float64)


def _i8(*shape) -> np.ndarray:
    return np.zeros(shape, dtype=np.int64)


def empty_cut_args(n: int = 1, nt: int = 1, nst: int = 1, nsec: int = 1, nb: int = 4, capacity: int = 1) -> CutArgs:
    tool = CA.empty_tool(n, nst, nsec)
    return CutArgs(tool, CA.empty_table(), CA.empty_memory(nst, nb), CA.empty_intervals(nst, 4),
                   CA.empty_refill(nst, nb), np.zeros(CA.N_ST), np.zeros(CA.N_SI, np.int64), CA.identity_pose(),
                   CA.empty_deflection(nst), CA.empty_occupancy(), CA.empty_packing(), CA.empty_axes(1),
                   CA.empty_state(n, nt, nst), CA.core_accum(1, 1, nt, max(n // max(nt, 1), 1)), CA.empty_accum(nsec),
                   CA.empty_out(nst, nsec), CA.empty_events(capacity), CA.empty_polar())


def empty_geo(nst: int = 1, nt: int = 1, ns: int = 1, na: int = 1, na_cut: int = 1) -> ContactGeo:
    return ContactGeo(_f(nst), _f(nst), _f(nst), np.zeros(nst, np.int8), _f(nst), _f(nst), _f(nt, ns), _f(nt, ns),
                      np.zeros(nt * ns + 1, np.int64), np.zeros(1, np.int64), np.zeros(max(na, 1), np.int64),
                      np.full(max(na_cut, 1), -1, np.int64))


def empty_contact_state(nt: int = 1, ns: int = 1) -> ContactState:
    iv = _i8(N_CS)
    iv[CS_OVERFLOW_TOOTH] = -1
    iv[CS_LAST_RAPID] = -(1 << 60)
    return ContactState(np.full((nt, ns), np.nan), _f(nt, ns, 3), np.zeros((nt, ns), np.int8),
                        np.full(nt, -(1 << 60), np.int64), np.full(8, -1, np.int64), iv, _f(N_CSF))


def empty_contact(n_axes: int = 1) -> HotContact:
    """An inactive contact of the right types (padding of the contact tuple)."""
    na = max(int(n_axes), 1)
    return HotContact(_i8(N_CI), _f(N_CF), empty_cut_args(), DA.empty_system(), DA.empty_axes(1),
                      DA.empty_work_grid(), _f(3), empty_geo(), empty_contact_state(), RN.example_spec(), _f(4), _f(4),
                      _f(3), _f(WORK_SIZE), _f(DA.N_BASE), _f(DA.N_BASE), _f(1), MA.eye_stack(MA.N_FK), _f(6, 4, 4), _f(na),
                      _f(RA.N_PS), _f(8), np.zeros((REFILL_PAIRS + 1, 2), np.int64), RA.empty_profile(), _f(8))


def empty_global(n_axes: int = 1, n_spindles: int = 1, n_paths: int = 1, n_ticks: int = 1) -> HotGlobal:
    na = max(int(n_axes), 1)
    ns = max(int(n_spindles), 1)
    npth = max(int(n_paths), 1)
    return HotGlobal(_i8(N_GI), _f(N_GF), _f(na), _f(ns), _f(na), _f(ns), _f(ns), _f(npth, 3), _f(3), _f(1),
                     _f(1, 1, 1, 3), _i8(N_RES), _i8(N_CNT), np.zeros((max(int(n_ticks), 1), npth), np.int8))


def example_motion() -> MA.MotionInputs:
    return MA.MotionInputs(MA.empty_chain(), MA.empty_servo(), MA.empty_servo_state(), MA.empty_spindle(),
                           MA.empty_spindle_state(), MA.empty_setpoint(), MA.empty_loop_state(), MA.empty_out(),
                           np.zeros(1, np.int32), _f(1), _f(1), np.zeros(3, np.int64))


def example_removal() -> RA.RemovalInputs:
    return RA.RemovalInputs(tuple(RN.example_field() for _ in range(MAX_WORK)),
                            tuple(RN.example_target() for _ in range(MAX_WORK)),
                            tuple(RN.example_polar() for _ in range(MAX_WORK)),
                            tuple(RN.example_profile() for _ in range(MAX_PATHS)),
                            tuple(RN.example_profile() for _ in range(MAX_PATHS)),
                            tuple(RN.example_insert() for _ in range(MAX_PATHS)),
                            tuple(RN.example_helix() for _ in range(MAX_PATHS)),
                            tuple(RA.empty_slabs() for _ in range(MAX_PATHS)),
                            tuple(RN.example_accum() for _ in range(MAX_PATHS)),
                            EventBuffer(_f(1, RA.EVENT_WIDTH), np.zeros(2, np.int64)), RA.make_scratch(),
                            np.zeros(1, np.int64), _f(1))


MOTION_T = typeof(example_motion())
REMOVAL_T = typeof(example_removal())
CONTACT_T = typeof(empty_contact())
CONTACTS_T = typeof(tuple(empty_contact() for _ in range(MAX_CONTACTS)))
GLOBAL_T = typeof(empty_global())
RING_T = typeof(ring_buffer(2, 1))
CUTARGS_T = typeof(empty_cut_args())

__all__ = [name for name in dir() if name.isupper() or name in (
    "CutArgs", "ContactGeo", "ContactState", "HotContact", "HotGlobal", "RingBuffer", "empty_cut_args", "empty_geo",
    "empty_contact_state", "empty_contact", "empty_global", "example_motion", "example_removal")]
