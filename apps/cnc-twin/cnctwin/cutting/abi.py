"""Kernel layouts of the cutting domain (docs/models/cutting.md "Kernel ABI"; rules: architecture 5.5).

Everything a cutting kernel reads or writes is a NamedTuple of writable, C-contiguous NumPy arrays
(float64, int64, int32 or int8), so every kernel has exactly one compiled specialisation. The shared
layouts (EdgeModel, CoeffTable, SurfaceMemory, EngagedIntervals, Accum, EventBuffer) are in
:mod:`cnctwin.core.abi`; this module adds what only cutting defines:

* :class:`CutTool` - per tool slot and macro step, the effective edge elements in the kernel form
  (positions, normals, force directions and moment arms in T at spindle angle 0, widths, table rows,
  wear-land forces, process-damping constants), the per-station element lists and the critical sections;
* :class:`CutTable` - the coefficient table linearised at L2: per table row (coefficient group or a
  low-speed / forming variant) piecewise-linear functions of the chip thickness h on two uniform grids
  ([0, h_min] and [h_min, h_hi]), so the hot loop evaluates ``c0 + c1 h`` and never a pow or exp;
* :class:`Refill` - field planes per memory bin (L2 surface-point query) for stale bins;
* :class:`StepPose`, :class:`Deflection`, :class:`CutOccupancy`, :class:`CutPacking`, :class:`AxisMap`,
  :class:`PolarView` - the per-step inputs the integration fills from motion, dynamics, chips, the
  machine and removal;
* :class:`CutState` - per contact, the kernel state carried between steps (checkpointed);
* :class:`CutAccum`, :class:`StepOut`, :class:`CutEvents` - outputs (macro-step accumulators, this
  step's wrenches for dynamics and the loops, event rows).

Frames and signs (architecture 4): T rotates with the spindle about +z_T; T0 is T without the
spindle angle; S is the stock frame in the material configuration. The force components of a table
row follow cnctwin.mechanics: f_t resists the cutting motion, f_r pushes the tool off the machined
surface, f_a acts along a = -(t_edge + sin(i) c) / cos(i); on the tool F = f_t (-c) + f_r (-n) + f_a a
with c the cutting direction and n the chip-thickness normal (into the material).
"""

from __future__ import annotations

from typing import NamedTuple

import numba
import numpy as np

from ..core.abi import (ACC_CHIP_VOLUME_M3, ACC_DAMPING_J, ACC_FEED_WORK_J, ACC_FORCE_PEAK_N, ACC_HEAT_FLANK_J,  # noqa: F401
                        ACC_HEAT_RAKE_J, ACC_HEAT_SHEAR_J, ACC_IN_CUT_S, ACC_PLOUGHING_J, ACC_POWER_PEAK_W,
                        ACC_PROCESS_DAMPING_J, ACC_RECUT_MASS_KG, ACC_SHAFT_WORK_J, ACC_TORQUE_PEAK_NM,
                        ACC_UNBALANCE_WORK_J, ELEM_ENTRIES, ELEM_FLANK_SLIDING_M, ELEM_H_MAX, ELEM_IN_CUT_S,
                        ELEM_INT_H_DT, ELEM_INT_SIGMA_V_FLANK, ELEM_INT_SIGMA_V_RAKE, ELEM_RAKE_SLIDING_M, N_ACC,
                        N_ELEM_ACC, WRENCH_TOOL_MEAN, WRENCH_TOOL_PEAK, WRENCH_WORK_MEAN, WRENCH_WORK_PEAK, Accum,
                        CoeffTable, EngagedIntervals, EventBuffer, SurfaceMemory, event_buffer)

# The constants below are read by the kernels as globals (frozen at compile time): this file is part
# of the kernel cache key of bin/tsafe and bin/twin through the @numba marker in this comment.

# ------------------------------------------------------------------------------- table columns
COL_FT = 0          # shearing part per unit chip width (N/m): tangential
COL_FR = 1          #   thrust
COL_FA = 2          #   axial
COL_ET = 3          # edge / ploughing / forming part per unit edge length (N/m): tangential
COL_ER = 4          #   thrust
COL_EA = 5          #   axial
COL_PS = 6          # share of the element power dissipated in the primary (shear) zone
COL_PR = 7          # share dissipated at the rake face (secondary zone); the edge share is 1 - PS - PR
COL_SIG_RAKE = 8    # mean rake-face normal stress (Pa) for the wear integrals
COL_RC = 9          # chip ratio h / h_c (rake sliding speed = RC x cutting speed)
COL_TCHIP = 10      # mean chip temperature leaving the contact (K)
COL_WS = 11         # share of the primary-zone heat entering the work (1 - frac_shear_to_chip)
COL_WF = 12         # share of the edge and flank heat entering the work (1 - frac_flank_to_tool)
N_COL = 13

# ------------------------------------------------------------------------------- step scalars
ST_DT_S = 0             # L0 step duration
ST_THETA_MID_RAD = 1    # spindle angle at the step midpoint, right-handed about +z_T, wrapped to [0, 2 pi)
ST_COS_MID = 2          # cos / sin of it (angle addition by the caller)
ST_SIN_MID = 3
ST_OMEGA_RAD_S = 4      # spindle angular velocity (M03 < 0); turning: the work spindle (about +z_S)
ST_TICK_FRAC = 5        # dt / L1 tick duration: weight of this step in the per-tick means
ST_RHO_KG_M3 = 6        # work density at 20 degC (emission and recut masses)
ST_TOOTH_PERIOD_S = 7   # 2 pi / (n_teeth |omega|) (0 when the spindle stands)
ST_TICK_S = 8           # base clock (1e-6 s)
N_ST = 12

# ------------------------------------------------------------------------------- step ints
SI_TICK_END = 0         # base tick at the end of the step
SI_TICK_ROW = 1         # row of the per-tick arrays of Accum (L1 tick index within the macro step)
SI_COMMIT = 2           # 1: update memory, accumulators and events; 0: evaluate forces only
SI_CONTACT = 3          # contact index (bus CuttingContact.contact)
SI_SLOT = 4             # tool slot
SI_STALE_TICKS = 5      # a memory bin older than this many base ticks is stale (one revolution + one tooth period)
SI_STALE_BEFORE = 6     # bins written before this base tick are stale (last rapid, tool change, new operation)
SI_L0_TICKS = 7         # base ticks per L0 step
N_SI = 8

# ------------------------------------------------------------------------------- tool scalars (CutTool.scal)
TC_N_TEETH = 0
TC_ROTATING = 1         # 1 rotating tool (milling, drilling, tapping ...), 0 non-rotating (turning family)
TC_HOLE_MAKING = 2      # 1 drills, reamers, taps (breakthrough / breakout events)
TC_HAND = 3             # hand sign of the tool (-1 right-hand)
TC_FLOAT_K = 4          # axial float stiffness of a floating / synchro tapping holder (N/m, 0: rigid holder)
TC_FLOAT_TRAVEL = 5     # +- axial float travel (m)
TC_RECUT_ROW = 6        # table row used for the cutting energy of recut chips
TC_R_CUT = 7            # largest cutting radius (recut envelope)
TC_Z_LO = 8             # axial extent of the cutting edges (recut envelope)
TC_Z_HI = 9
TC_TANGENT_VN = 10      # |v.n| below this fraction of the relative tip speed: a boundary point of the pass lies on
                        #   the envelope (surface event at entry / exit)
TC_TURN_COLWELL = 11    # turning force model: 0 per element, 1 Colwell chord + equivalent chip thickness
TC_SURF_EVENT_RAD = 12  # turning: work rotation between surface events
TC_H_CAP = 13           # turning: longest chip-thickness ray (m)
TC_POLAR_SIDE = 14      # turning: polar field available (1) or not (0)
TC_RECOVERY_RATIO = 15  # elastic recovery: partial memory update between h_min and (1 + ratio) h_min
TC_PD_MAX_LEN = 16      # longest flank interference length of the process-damping model (m)
TC_N_ELEM = 17          # elements per tooth of the EdgeModel (flat index e = tooth * n_elem + elem)
N_TC = 20

# ------------------------------------------------------------------------------- CutTool.pk columns (additive)
# The per-element constants of the force kernel packed into one row per element (built at L2 from the fields
# above by pack_elements): the kernel reads one contiguous row instead of ~25 arrays (register pressure and
# array-descriptor reloads were most of its cost, docs/performance.md "cut_step").
PK_ALPHA0 = 0           # alpha0_rad
PK_P = 1                # p_T_m (3)
PK_N = 4                # n_T (3)
PK_C = 7                # c_T (3)
PK_DIR = 10             # dir_T (9): component j = t, r, a at PK_DIR + 3 j + (x, y, z)
PK_ARM = 19             # arm_T (9), same order
PK_DB = 28              # db_m
PK_DS = 29              # dS_m
PK_RXY = 30             # r_xy_m
PK_NPXY = 31            # n_x p_x + n_y p_y (the memory stencil's chord correction along n)
PK_PZR = 32             # p_z - z of the element's station (moment arm of the station rotation)
PK_FLANK = 33           # flank_N (2): tangential, normal
PK_PD = 35              # pd (4): K_sp dS, land, tan(clearance), friction
PK_SIG_FLANK = 39       # sig_flank_Pa
PK_RAKE = 40            # rake_rad
PK_INCL = 41            # incl_rad
PK_CHIP = 42            # chip_T (3)
PK_EDGE_R = 45          # edge_radius_m
PK_VB = 46              # VB_m
PK_CURV = 47            # signed curvature of the edge in its chip-section plane (1/m, edge_curvature): turning's chip
#                         strip of an element, h deep along -n, is db (1 - h curv / 2) wide (a nose radius r: 1 / r)
N_PK = 48
# CutTool.pki columns (int64)
PKI_ROW = 0             # row
PKI_TOOTH = 1           # tooth
PKI_STATION = 2         # station
PKI_READ = 3            # read_row
PKI_WRITE = 4           # write_row
PKI_KIND = 5            # kind
PKI_FORMING = 6         # forming
PKI_ELEM = 7            # elem
PKI_AXIAL = 8           # 1: axial-normal edge (|n_z| > 0.7: end, corner bottom, wiper) - floor surface events
N_PKI = 10

# ------------------------------------------------------------------------------- CutState.flags bits
FL_IN_CUT = 1           # the element was in cut (engaged and h > 0) at its last evaluation
FL_VN_POS = 2           # its relative normal velocity was positive (moving into the material)
FL_EVENT_DONE = 4       # a surface event was written in the current pass
FL_NONDEGEN = 8         # the envelope condition was non-degenerate somewhere in the current pass

# ------------------------------------------------------------------------------- CutState.pend columns
PD_BIN = 0              # pending memory write (delayed by one bin, so an element never reads its own trace)
PD_X = 1
PD_Y = 2
PD_Z = 3
PD_H = 4
N_PD = 5

# ------------------------------------------------------------------------------- process-damping wave memory (CutState.pd_ring / pd_aux)
# The surface an element generated behind it, held as its relative vibration displacement u (tool deflection at the
# element minus the work's, T0) sampled every pd_max_flank / (PD_RING_N - 1) of its travel along the material since it
# entered the cut: the flank meets the wave the edge left there (docs/models/cutting.md "Process damping")
PD_RING_N = 16
PDA_X = 0               # travel along the material since the element entered the cut, at its last committed evaluation
PDA_UX = 1              # u at that evaluation (T0)
PDA_UY = 2
PDA_UZ = 3
PDA_COUNT = 4           # samples written since the entry (sample g at travel g dx, ring slot g mod PD_RING_N)
N_PDA = 5

# ------------------------------------------------------------------------------- recut strikes in progress (CutState.recut_act)
# A struck chip voxel is cut like material: a load on the tool at the struck point for the crossing time
# (docs/models/cutting.md "Recutting"); each row is one load still acting
RECUT_SLOTS = 16
RA_FX = 0               # force on the tool (T0)
RA_FY = 1
RA_FZ = 2
RA_MX = 3               # its moment about the tool tip (T0)
RA_MY = 4
RA_MZ = 5
RA_T = 6                # time it ends (s, absolute; a row ends there whether the tool still cuts or not)
RA_Z = 7                # z_T of the struck point (the station that takes the load)
RA_T0 = 8               # time it starts (s, absolute); a free row has RA_T <= RA_T0
N_RA = 10

# ------------------------------------------------------------------------------- per-element accumulators (CutState.elem_acc)
# the first N_ELEM_ACC columns are the core.abi ELEM_* slots in the same order (copied to Accum.elem at absorb)
EA_LAST_H = N_ELEM_ACC  # h at the last surface-relevant evaluation (surface events of degenerate elements)
N_EA = N_ELEM_ACC + 2

# ------------------------------------------------------------------------------- per-tooth pass state (CutState.tooth)
TS_ACTIVE = 0           # elements of the tooth in cut at the last step
TS_VOL = 1              # analytic chip volume of the current pass (or emission period of a continuous cut)
TS_HV = 2               # sum h dV
TS_B = 3                # largest engaged chip width of a step (sum db)
TS_LV = 4               # spindle angle swept while the tooth was in cut (uncut chip length = mean r x this)
TS_VV = 5               # sum V dV
TS_RAKEV = 6            # sum rake dV
TS_INCLV = 7            # sum inclination dV
TS_TV = 8               # sum T_chip dV
TS_RV = 9               # sum r dV
TS_PX = 10              # chip separation point (S): the tooth's largest contributor at its latest step in cut
TS_PY = 11
TS_PZ = 12
TS_UX = 13              # chip velocity at separation (M)
TS_UY = 14
TS_UZ = 15
TS_NX = 16              # outward normal of the generated surface there (S, deposit normal)
TS_NY = 17
TS_NZ = 18
TS_DOM_VOL = 19         # largest single-element contribution of the period and its kind / table row
TS_DOM_KIND = 20
TS_DOM_ROW = 21
TS_WORK_J = 22          # estimated heat into the work of the period (deposit weight)
TS_EMIT_TICK = 23       # tick of the last emission row (continuous cuts emit once per tooth period)
TS_ENTRY_TICK = 24      # shocks: first contact of the pass
TS_ENTRY_ELEM = 25
TS_ENTRY_H = 26
TS_ENTRY_ANGLE = 27
TS_LOADING = 28         # 1 while the engaged element count is still growing
TS_FIRST_ENTRY_S = 29   # absolute time of the first and of the latest element entry
TS_LAST_ENTRY_S = 30
TS_FMAX = 31            # peak tooth force of the pass
TS_EXIT_FIRST_S = 32    # first and latest element exit of the pass (unloading time)
TS_LAST_EXIT_S = 33
TS_LAST_ELEM = 34       # element that left last, its h and angle
TS_LAST_H = 35
TS_LAST_ANGLE = 36
TS_BREAKOUT = 37        # a breakthrough / breakout row was written in this pass
TS_PASS_ON = 38         # a pass is open
TS_FX = 39              # step scratch: tooth force this step (T0), entries, elements in cut, exits
TS_FY = 40
TS_FZ = 41
TS_NEW = 42
TS_NOW = 43
TS_EXITS = 44
TS_CENTRE_ON = 45       # a centre station of the tooth is in cut this step / was at the last step
TS_CENTRE_WAS = 46
TS_STEP_DV = 47         # step scratch: chip volume of the tooth this step (sum dV)
TS_STEP_E = 48          # element with the largest contribution this step
TS_STEP_B = 49          # engaged chip width of the tooth this step (sum db)
TS_STEP_MAXDV = 50      # largest single contribution this step
TS_AVAIL = 51           # valid elements of the tooth at engaged stations this step (full contact = all in cut)
TS_S_PX = 52            # step scratch in T: sum p dV, sum (edge + chip-flow velocity) dV, sum n dV
TS_S_PY = 53
TS_S_PZ = 54
TS_S_VX = 55
TS_S_VY = 56
TS_S_VZ = 57
TS_S_NX = 58
TS_S_NY = 59
TS_S_NZ = 60
N_TS = 64

# ------------------------------------------------------------------------------- CutState.misc
MS_FLOAT_U = 0          # axial float displacement of the tool (m, +z_T)
MS_FLOAT_F = 1          # axial force of the last step (N, T0 z)
MS_FLOAT_K = 2          # axial cutting stiffness of the last step (N/m)
MS_RECUT_ANGLE = 3      # spindle angle travelled since the last recut scan
MS_LAST_TICK = 4        # tick of the last committed step
MS_TICK_PEAK_ROW = 5    # tick row whose peak sample is being tracked
MS_TURN_EVENT_ANGLE = 6  # turning: work rotation since the last surface event
MS_STEPS = 7
N_MS = 8

# ------------------------------------------------------------------------------- CutAccum.scalars
CA_DISPLACED_M3 = 0     # forming taps: displaced (re-added) volume, analytic
CA_FORMING_J = 1        # forming work (included in the ploughing heat)
CA_RECUT_J = 2          # work done on recut chips (impulse x edge speed + cutting energy)
CA_PACKING_J = 3        # work of the chip-packing torque and force
CA_DEFLECTION_J = 4     # work of the cutting forces on the tool / work deflection velocities (to dynamics)
CA_STALE_READS = 5      # memory bins refilled from the field planes
CA_REFILL_MISSING = 6   # stale bins without a plane (refill requests written; the step should be replayed)
CA_TABLE_OVER = 7       # evaluations above the table's h range (linear extrapolation)
CA_H_MAX = 8            # largest chip thickness of the step
CA_STEPS = 9            # committed steps
CA_STEPS_IN_CUT = 10
CA_ELEM_EVALS = 11      # element evaluations in cut
CA_TORQUE_INT = 12      # integral of the process torque about +z_T (N m s) (spindle mean)
CA_FX_INT = 13          # integral of the tool force in T0 (N s)
CA_FY_INT = 14
CA_FZ_INT = 15
CA_FEED_POWER_PEAK = 16
CA_EVENTS_LOST = 17     # rows that did not fit an event buffer (the buffer's n[1] is set too)
CA_PROCESS_DAMPING_J = 18
CA_RECUT_KG = 19
CA_EDGE_LEN_INT = 20    # integral of the engaged edge length (m s): contact width for the pipeline
CA_ENTRIES = 21         # tooth entries (pass starts)
CA_FLOAT_U_MAX = 22     # largest |axial float displacement|
CA_AXIAL_K = 23         # last axial cutting stiffness (N/m)
CA_POWER_INT = 24       # integral of the cutting power (J), = heat generated
CA_V_INT = 25           # integral of V dt over elements in cut (mean speed for the pipeline)
CA_OMEGA_INT = 26       # integral of |omega| dt (mean spindle speed of the step)
CA_TIME = 27            # integral of dt over committed steps
CA_FPEAK_MX = 28        # tool force at the instant of the largest |F| (M frame)
CA_FPEAK_MY = 29
CA_FPEAK_MZ = 30
N_CA = 32

# ------------------------------------------------------------------------------- StepOut.misc
SO_TORQUE_NM = 0        # process torque about the spindle axis this step (on the rotating body, right-handed)
SO_POWER_W = 1          # cutting power this step
SO_ENGAGED = 2          # elements in cut this step
SO_AXIAL_K = 3          # axial cutting stiffness (dF_z / du_z) of the step (N/m)
N_SO = 4

# ------------------------------------------------------------------------------- event rows
SHOCK_ENTRY = 1
SHOCK_EXIT = 2
SHOCK_BREAKTHROUGH = 3  # hole-making tool: a centre station left the material while the tooth still cuts
SHOCK_BREAKOUT = 4      # hole-making tool: part of the tooth left the material (cross hole, exit face)
SHOCK_COLUMNS = ("tick", "tooth", "elem", "entry_h_m", "entry_angle_rad", "force_first_full_N", "loading_time_s",
                 "kind")      # = core.bus CuttingContact.shocks
SURFACE_COLUMNS = ("tick", "tooth", "elem", "x_S_m", "y_S_m", "z_S_m", "nx", "ny", "nz", "h_m", "edge_radius_m",
                   "VB_m", "u_rel_x_m", "u_rel_y_m", "u_rel_z_m", "tool_slot")   # = core.bus SURFACE_EVENT_COLUMNS
DEPOSIT_COLUMNS = ("x_S_m", "y_S_m", "z_S_m", "nx", "ny", "nz", "J")             # = CuttingContact.work_heat_S
RECUT_COLUMNS = ("tick", "tooth", "elem", "mass_kg", "x_M_m", "y_M_m")           # = CuttingContact.recut
# one row per tooth pass (or tooth period of a continuous cut); identical to chips.abi.EMISSION_COLUMNS, so
# the integration may hand the chips emission EventBuffer to the kernel as CutEvents.passes
PASS_COLUMNS = ("tick", "contact", "tool_slot", "tooth", "mass_kg", "x_M_m", "y_M_m", "z_M_m", "vx_M_m_s",
                "vy_M_m_s", "vz_M_m_s", "temp_K", "h_m", "b_m", "l_m", "v_cut_m_s", "rake_rad", "incl_rad",
                "edge_kind", "group")
REFILL_COLUMNS = ("tick", "station", "bin", "x_S_m", "y_S_m", "z_S_m", "nx", "ny", "nz")
SHOCK_WIDTH = len(SHOCK_COLUMNS)
SURFACE_WIDTH = len(SURFACE_COLUMNS)
DEPOSIT_WIDTH = len(DEPOSIT_COLUMNS)
RECUT_WIDTH = len(RECUT_COLUMNS)
PASS_WIDTH = len(PASS_COLUMNS)
REFILL_WIDTH = len(REFILL_COLUMNS)


class CutTool(NamedTuple):
    """Effective cutting edges of one tool in kernel form (built at L2 by cnctwin.cutting.geometry).

    Element arrays are flat, index e = tooth * n_elem + elem of the tool's EdgeModel; vectors are in T
    at spindle angle 0 (for non-rotating tools T = T0). Only valid elements are listed in the
    station lists ``st_elem[st_ptr[s]:st_ptr[s + 1]]``.
    """

    p_T_m: np.ndarray           # [n, 3] element midpoint (runout and tilt included)
    n_T: np.ndarray             # [n, 3] chip-thickness normal, into the material
    c_T: np.ndarray             # [n, 3] cutting direction (unit)
    dir_T: np.ndarray           # [n, 3, 3] force direction on the tool of component j = t, r, a
    arm_T: np.ndarray           # [n, 3, 3] (p - z_station e_z) x dir_T[e, j]
    alpha0_rad: np.ndarray      # [n] right-handed angle about +z_T at spindle angle 0, [0, 2 pi)
    r_xy_m: np.ndarray          # [n] distance from the spindle axis
    db_m: np.ndarray            # [n] chip width
    dS_m: np.ndarray            # [n] edge length
    row: np.ndarray             # [n] int32 table row
    tooth: np.ndarray           # [n] int32
    elem: np.ndarray            # [n] int32
    station: np.ndarray         # [n] int32 station (engaged intervals, station outputs)
    read_row: np.ndarray        # [n] int32 memory row the element reads (its station; on a tap the first land
                                #   reads the handover row nst + station written by its predecessor on the
                                #   thread helix, docs/models/cutting.md "Surface memory")
    write_row: np.ndarray       # [n] int32 memory row it writes (-1: none): the row its successor on the helix reads
    kind: np.ndarray            # [n] int32 core.abi EDGE_*
    forming: np.ndarray         # [n] int8 1: displaces material (forming tap lobe), no chip
    valid: np.ndarray           # [n] int8
    flank_N: np.ndarray         # [n, 2] wear-land force while in contact: tangential, normal (Smithey et al. 2001)
    sig_flank_Pa: np.ndarray    # [n] flank normal stress for the wear integral
    pd: np.ndarray              # [n, 4] process damping: K_sp dS (N/m2), zero-clearance land (m), tan(clearance), friction
    edge_radius_m: np.ndarray   # [n]
    VB_m: np.ndarray            # [n]
    chip_T: np.ndarray          # [n, 3] chip flow direction relative to the tool (unit)
    rake_rad: np.ndarray        # [n]
    incl_rad: np.ndarray        # [n]
    st_z_m: np.ndarray          # [nst] station height (z_T)
    st_ptr: np.ndarray          # [nst + 1] int32
    st_elem: np.ndarray         # [n_valid] int32
    st_sec: np.ndarray          # [nst] int32 index of the first critical section above the station
    st_centre: np.ndarray       # [nst] int8 centre station of a hole-making tool (breakthrough)
    sec_z_m: np.ndarray         # [nsec] critical sections, ascending z
    sec_Zb_m3: np.ndarray       # [nsec] bending section modulus
    sec_Zp_m3: np.ndarray       # [nsec] polar section modulus
    tooth_a0_rad: np.ndarray    # [nt] right-handed angle of each tooth's reference at spindle angle 0 (recut rows)
    scal: np.ndarray            # [N_TC]
    pk: np.ndarray              # [n_valid, N_PK] the element constants above packed per element, rows in the order of
                                #   st_elem (row k holds element st_elem[k]: the kernel streams through them; additive)
    pki: np.ndarray             # [n_valid, N_PKI] int64 the element indices above, same rows (additive)


class CutTable(NamedTuple):
    """Coefficient rows linearised on two uniform h grids per row (see module doc).

    Segment k < n_lo covers [k, k + 1) h_min / n_lo; segment n_lo + m covers h_min + [m, m + 1) / inv_dhi.
    ``coef[row, seg, 0, col]`` is the intercept, ``coef[row, seg, 1, col]`` the slope in h.
    """

    coef: np.ndarray            # [nr, n_lo + n_hi, 2, N_COL]
    h_min_m: np.ndarray         # [nr]
    inv_dlo: np.ndarray         # [nr] n_lo / h_min (0 when h_min = 0)
    inv_dhi: np.ndarray         # [nr] n_hi / (h_hi - h_min)
    h_hi_m: np.ndarray          # [nr]
    forming: np.ndarray         # [nr] int8
    h_write_m: np.ndarray       # [nr] passages with h at or above this renew the surface memory (h_min for
                                #   cutting rows, +inf for rubbing rows such as margins, 0 for forming lobes)
    n_lo: np.ndarray            # [1] int64
    v_min_m_s: np.ndarray       # [1] below: indentation rows (chosen per element at L2)


class Refill(NamedTuple):
    """Field surface planes per memory bin (point q, outward unit normal m, both in S), from removal's
    surface-point query at L2; valid = 0 where none was computed."""

    q_S_m: np.ndarray           # [nst, nb, 3]
    m_S: np.ndarray             # [nst, nb, 3]
    valid: np.ndarray           # [nst, nb] int8


class StepPose(NamedTuple):
    """Kinematics of one L0 step at its midpoint (filled by the integration from motion's kinematics)."""

    R_S_T0: np.ndarray          # [3, 3] columns: T0 axes in S
    tip_S_m: np.ndarray         # [3] tool tip (T0 origin) in S, material configuration
    v_tip_S_m_s: np.ndarray     # [3] velocity of the tip relative to the material, S
    w_T0_S_rad_s: np.ndarray    # [3] angular velocity of T0 relative to S (rotary axes; lathes -omega_C e_z), S
    R_M_S: np.ndarray           # [3, 3] S -> M (emission and recut positions)
    t_M_S_m: np.ndarray         # [3]
    R_W_S: np.ndarray           # [3, 3] S -> W (wrench on the part)
    t_W_S_m: np.ndarray         # [3]
    v_S_M_m_s: np.ndarray       # [3] velocity of the S origin in M (moving table) - chip velocities in M
    w_S_M_rad_s: np.ndarray     # [3] angular velocity of S in M (rotary table, lathe spindle), M


class Deflection(NamedTuple):
    """Deflections at the step midpoint (dynamics' modal state; cnctwin.cutting.kernels.deflection_from_modal)."""

    u_st_T0_m: np.ndarray       # [nst, 6] tool deflection at the stations: dx dy dz rx ry rz (T0)
    v_st_T0_m_s: np.ndarray     # [nst, 6] its rate
    u_work_S_m: np.ndarray      # [3] workpiece deflection + structure displacement d(X) at the contact (S)
    v_work_S_m_s: np.ndarray    # [3]


class CutOccupancy(NamedTuple):
    """Frozen chip-occupancy grid of the step (ChipFeedback; fields as chips.abi.Occupancy) and the
    mass already recut from it in this macro step (cutting's scratch, zeroed by kernel_inputs)."""

    origin_S_m: np.ndarray      # [3]
    cell_m: np.ndarray          # [1] (0: no grid)
    mass_kg: np.ndarray         # [nx, ny, nz]
    size_m: np.ndarray          # [nx, ny, nz]
    thickness_m: np.ndarray     # [nx, ny, nz]
    taken_kg: np.ndarray        # [nx, ny, nz]


class CutPacking(NamedTuple):
    """Chip-packing loads of the contact's tool (ChipFeedback): the torque opposes the rotation."""

    torque_Nm: np.ndarray       # [1]
    force_T0_N: np.ndarray      # [3]


class AxisMap(NamedTuple):
    """Machine axes for the per-tick axis loads: direction of each axis' link motion in M, a point on a
    rotary axis, rotary flag and side (+1 the axis carries the tool, -1 the workpiece)."""

    dir_M: np.ndarray           # [na, 3]
    point_M_m: np.ndarray       # [na, 3]
    rotary: np.ndarray          # [na] int8
    side: np.ndarray            # [na]


class PolarView(NamedTuple):
    """Radius view of removal's polar field (turning): material at (theta, z) between r_in and r_out
    (rotating S, bins theta_b = (b + 0.5) dtheta, z_i = z0 + (i + 0.5) dz). Kept current by removal's
    polar update for the bins it changes (docs/models/cutting.md "Turning").

    ``gap_lo_m`` / ``gap_hi_m`` (additive, hot-loop integration, docs/models/hotloop.md "Turning"): an air
    gap inside [r_in, r_out] of the cell (nearest bin and row; none where gap_hi <= gap_lo). A radial ray of
    the polar field holds two segments behind a major edge that leans back (kappa_r > 90 deg): the previous
    revolution leaves a thin ring above its cut, the next chip."""

    r_out_m: np.ndarray         # [nth, nz]
    r_in_m: np.ndarray          # [nth, nz]
    geom: np.ndarray            # [4]: dtheta, z0, dz, unused
    gap_lo_m: np.ndarray        # [nth, nz] air gap inside the cell's interval (none: gap_hi <= gap_lo)
    gap_hi_m: np.ndarray        # [nth, nz]


class CutState(NamedTuple):
    """Kernel state of one contact carried between steps (checkpointed by cnctwin.cutting.state)."""

    h_last_m: np.ndarray        # [n]
    vn_last_m_s: np.ndarray     # [n]
    tick_last: np.ndarray       # [n] int64 (-1: never evaluated)
    flags: np.ndarray           # [n] int8 FL_*
    pend: np.ndarray            # [n, N_PD]
    pend_tick: np.ndarray       # [n] int64
    elem_acc: np.ndarray        # [n, N_EA]
    tooth: np.ndarray           # [nt, N_TS]
    misc: np.ndarray            # [N_MS]
    st_written: np.ndarray      # [nst] int8 station rows of StepOut written last step
    last_p: np.ndarray          # [n, 10] last in-cut sample of a degenerate element: p_S (3), h, n_S (3), u_rel_S (3)
    # additive (process damping from the generated surface, recut loads over their crossing time;
    # docs/models/cutting.md "Process damping", "Recutting")
    pd_ring: np.ndarray         # [n, PD_RING_N, 3] wave memory: u (T0) at travel g dx since the entry
    pd_aux: np.ndarray          # [n, N_PDA]
    recut_act: np.ndarray       # [RECUT_SLOTS, N_RA] recut loads still acting


class CutAccum(NamedTuple):
    scalars: np.ndarray         # [N_CA]
    sections: np.ndarray        # [nsec, 2] peak bending moment (N m), peak von Mises stress (Pa)


class StepOut(NamedTuple):
    """This step's loads (overwritten every step): wrench on the tool about the tip (T0), per station
    force and moment about the station's axis point (T0, for dynamics' modal forces), force in S."""

    wrench_T0: np.ndarray       # [6] mean over the step
    station_T0: np.ndarray      # [nst, 6]
    force_S_N: np.ndarray       # [3]
    misc: np.ndarray            # [N_SO]
    sec: np.ndarray             # [nsec + 1, 9] scratch: impulses of the stations below each section bin


class CutEvents(NamedTuple):
    shocks: EventBuffer         # SHOCK_COLUMNS
    surface: EventBuffer        # SURFACE_COLUMNS
    deposits: EventBuffer       # DEPOSIT_COLUMNS
    recut: EventBuffer          # RECUT_COLUMNS
    passes: EventBuffer         # PASS_COLUMNS
    refill: EventBuffer         # REFILL_COLUMNS


# ------------------------------------------------------------------------------- constructors
def _f(*shape) -> np.ndarray:
    return np.zeros(shape, dtype=np.float64)


def _i32(*shape) -> np.ndarray:
    return np.zeros(shape, dtype=np.int32)


def _i8(*shape) -> np.ndarray:
    return np.zeros(shape, dtype=np.int8)


def empty_tool(n: int = 1, nst: int = 1, nsec: int = 1) -> CutTool:
    n = max(int(n), 1)
    nst = max(int(nst), 1)
    nsec = max(int(nsec), 1)
    return CutTool(_f(n, 3), _f(n, 3), _f(n, 3), _f(n, 3, 3), _f(n, 3, 3),
                   _f(n), _f(n), _f(n), _f(n), _i32(n), _i32(n), _i32(n), _i32(n), _i32(n), _i32(n), _i32(n), _i8(n),
                   _i8(n),
                   _f(n, 2), _f(n), _f(n, 4), _f(n), _f(n), _f(n, 3), _f(n), _f(n), _f(nst), _i32(nst + 1),
                   _i32(n), _i32(nst), _i8(nst), _f(nsec), _f(nsec), _f(nsec), _f(1), _f(N_TC), _f(n, N_PK),
                   np.zeros((n, N_PKI), np.int64))


def pack_elements(tool) -> tuple[np.ndarray, np.ndarray]:
    """(pk [n_valid, N_PK], pki [n_valid, N_PKI]) of a CutTool from its per-element fields: the kernel's packed rows,
    in the order of the station lists st_elem (the element loop reads them consecutively)."""
    n = int(tool.p_T_m.shape[0])
    pk = np.zeros((n, N_PK))
    pki = np.zeros((n, N_PKI), np.int64)
    P = np.asarray(tool.p_T_m, float)
    N = np.asarray(tool.n_T, float)
    pk[:, PK_ALPHA0] = tool.alpha0_rad
    pk[:, PK_P:PK_P + 3] = P
    pk[:, PK_N:PK_N + 3] = N
    pk[:, PK_C:PK_C + 3] = tool.c_T
    pk[:, PK_DIR:PK_DIR + 9] = np.asarray(tool.dir_T, float).reshape(n, 9)
    pk[:, PK_ARM:PK_ARM + 9] = np.asarray(tool.arm_T, float).reshape(n, 9)
    pk[:, PK_DB] = tool.db_m
    pk[:, PK_DS] = tool.dS_m
    pk[:, PK_RXY] = tool.r_xy_m
    pk[:, PK_NPXY] = N[:, 0] * P[:, 0] + N[:, 1] * P[:, 1]
    nst = int(tool.st_z_m.shape[0])
    station = np.asarray(tool.station, np.int64)
    zst = np.asarray(tool.st_z_m, float)[np.clip(station, 0, max(nst - 1, 0))] if nst else np.zeros(n)
    pk[:, PK_PZR] = P[:, 2] - zst
    pk[:, PK_FLANK:PK_FLANK + 2] = tool.flank_N
    pk[:, PK_PD:PK_PD + 4] = tool.pd
    pk[:, PK_SIG_FLANK] = tool.sig_flank_Pa
    pk[:, PK_RAKE] = tool.rake_rad
    pk[:, PK_INCL] = tool.incl_rad
    pk[:, PK_CHIP:PK_CHIP + 3] = tool.chip_T
    pk[:, PK_EDGE_R] = tool.edge_radius_m
    pk[:, PK_VB] = tool.VB_m
    pk[:, PK_CURV] = edge_curvature(tool)
    pki[:, PKI_ROW] = tool.row
    pki[:, PKI_TOOTH] = tool.tooth
    pki[:, PKI_STATION] = station
    pki[:, PKI_READ] = tool.read_row
    pki[:, PKI_WRITE] = tool.write_row
    pki[:, PKI_KIND] = tool.kind
    pki[:, PKI_FORMING] = tool.forming
    pki[:, PKI_ELEM] = tool.elem
    pki[:, PKI_AXIAL] = (np.abs(N[:, 2]) > 0.7).astype(np.int64)
    order = np.asarray(tool.st_elem, np.int64)
    return np.ascontiguousarray(pk[order]), np.ascontiguousarray(pki[order])


def edge_curvature(tool) -> np.ndarray:
    """[n] signed curvature (1/m) of the cutting edge at each element in the plane of its chip section: the meridian
    plane (r_xy, z) of a rotating tool (the helix's twist about the axis is no curvature of the chip section), the
    rake plane of a non-rotating one. Positive where the outward normals n_T spread along the edge, so the chip,
    which lies behind the edge (on -n_T), narrows with depth: a nose or corner radius r gives 1 / r, a straight
    edge 0. turn_step measures the chip thickness h along -n into the polar field, so an element's chip is the
    strip between its neighbours' normals, db (1 - h curv / 2) wide at mid-depth (an annular sector: exact for
    h <= r); h db overstated the chip of a turning nose by h / 2r, 19 % at f 0.3 mm/rev on r 0.8 mm (+5 % on a
    G71 pass). cut_step does not use it: its h is the projection on n of the surface's shift between passes at the
    element's own station, and h db tiles the chip exactly (strips along the shift). Central difference over the
    neighbouring elements of the same tooth (one-sided at an edge's end or next to an invalid or distant element;
    0 for an isolated one)."""
    n = int(tool.p_T_m.shape[0])
    curv = np.zeros(n)
    if n < 2:
        return curv
    P = np.asarray(tool.p_T_m, float)
    N = np.asarray(tool.n_T, float)
    if float(tool.scal[TC_ROTATING]) > 0.5:
        r = np.hypot(P[:, 0], P[:, 1])
        safe = np.where(r > 0.0, r, 1.0)
        n_r = np.where(r > 0.0, (N[:, 0] * P[:, 0] + N[:, 1] * P[:, 1]) / safe, np.hypot(N[:, 0], N[:, 1]))
        P2 = np.column_stack([r, P[:, 2]])
        N2 = np.column_stack([n_r, N[:, 2]])
    else:
        P2, N2 = P, N
    norm = np.linalg.norm(N2, axis=1)
    N2 = N2 / np.where(norm > 0.0, norm, 1.0)[:, None]
    n_elem = int(round(float(tool.scal[TC_N_ELEM]))) if len(tool.scal) > TC_N_ELEM else 0
    if n_elem <= 0:
        n_elem = n
    tooth = np.arange(n) // n_elem
    valid = np.asarray(tool.valid) != 0
    dS = np.asarray(tool.dS_m, float)

    def neighbour(k):
        j = np.arange(n) + k
        ok = (j >= 0) & (j < n)
        jj = np.clip(j, 0, n - 1)
        ok &= (tooth[jj] == tooth) & valid[jj] & valid
        gap = np.linalg.norm(P[jj] - P, axis=1)
        ok &= gap <= 2.0 * np.maximum(dS, dS[jj]) + 1e-12
        return jj, ok

    jp, okp = neighbour(1)
    jm, okm = neighbour(-1)
    a = np.where(okm, jm, np.arange(n))
    b = np.where(okp, jp, np.arange(n))
    dp = P2[b] - P2[a]
    dn = N2[b] - N2[a]
    d2 = np.einsum("ij,ij->i", dp, dp)
    use = (okp | okm) & (d2 > 0.0)
    curv[use] = np.einsum("ij,ij->i", dn[use], dp[use]) / d2[use]
    return curv


def with_packed(tool):
    """The tool with pk / pki rebuilt from its per-element fields (after any change of those fields)."""
    pk, pki = pack_elements(tool)
    return tool._replace(pk=pk, pki=pki)


def memory_rows(tool: CutTool) -> int:
    """Rows of the surface memory (and refill planes) a tool needs: one per station, plus one handover row per
    station (index nst + s) when an element reads or writes one (taps)."""
    nst = max(int(tool.st_z_m.shape[0]), 1)
    top = max(int(np.max(tool.read_row)) if tool.read_row.size else 0,
              int(np.max(tool.write_row)) if tool.write_row.size else 0)
    return 2 * nst if top >= nst else nst


def row_station(tool: CutTool, r: int) -> int:
    """Station whose circle a memory row describes (handover rows nst + s describe station s)."""
    nst = max(int(tool.st_z_m.shape[0]), 1)
    return int(r) - nst if int(r) >= nst else int(r)


def empty_table(rows: int = 1, n_lo: int = 2, n_hi: int = 2) -> CutTable:
    rows = max(int(rows), 1)
    return CutTable(_f(rows, n_lo + n_hi, 2, N_COL), _f(rows), _f(rows), _f(rows), _f(rows), _i8(rows), _f(rows),
                    np.array([n_lo], np.int64), _f(1))


def empty_memory(nst: int = 1, nb: int = 720) -> SurfaceMemory:
    nst = max(int(nst), 1)
    nb = max(int(nb), 4)
    return SurfaceMemory(_f(nst, nb, 3), np.zeros((nst, nb), np.int64), _i8(nst, nb))


def empty_intervals(nst: int = 1, n_max: int = 4) -> EngagedIntervals:
    nst = max(int(nst), 1)
    return EngagedIntervals(_f(nst, n_max), _f(nst, n_max), _i32(nst), np.zeros(1, np.int64))


def empty_refill(nst: int = 1, nb: int = 720) -> Refill:
    nst = max(int(nst), 1)
    return Refill(_f(nst, nb, 3), _f(nst, nb, 3), _i8(nst, nb))


def identity_pose() -> StepPose:
    return StepPose(np.eye(3), _f(3), _f(3), _f(3), np.eye(3), _f(3), np.eye(3), _f(3), _f(3), _f(3))


def empty_deflection(nst: int = 1) -> Deflection:
    nst = max(int(nst), 1)
    return Deflection(_f(nst, 6), _f(nst, 6), _f(3), _f(3))


def empty_occupancy(nx: int = 1, ny: int = 1, nz: int = 1) -> CutOccupancy:
    return CutOccupancy(_f(3), _f(1), _f(nx, ny, nz), _f(nx, ny, nz), _f(nx, ny, nz), _f(nx, ny, nz))


def empty_packing() -> CutPacking:
    return CutPacking(_f(1), _f(3))


def empty_axes(n_axes: int = 0) -> AxisMap:
    n = max(int(n_axes), 1)
    rotary = _i8(n)
    side = _f(n)       # side 0: an unused row contributes nothing
    return AxisMap(_f(n, 3), _f(n, 3), rotary, side)


def empty_polar(nth: int = 1, nz: int = 1) -> PolarView:
    return PolarView(_f(nth, nz), _f(nth, nz), np.array([2.0 * np.pi / max(nth, 1), 0.0, 1.0, 0.0]), _f(nth, nz),
                     _f(nth, nz))


def empty_state(n: int = 1, nt: int = 1, nst: int = 1) -> CutState:
    n = max(int(n), 1)
    nt = max(int(nt), 1)
    nst = max(int(nst), 1)
    pend = _f(n, N_PD)
    pend[:, PD_BIN] = -1.0
    tooth = _f(nt, N_TS)
    tooth[:, TS_EMIT_TICK] = -1.0
    tooth[:, TS_STEP_E] = -1.0
    return CutState(_f(n), _f(n), np.full(n, -1, np.int64), _i8(n), pend, np.zeros(n, np.int64), _f(n, N_EA),
                    tooth, _f(N_MS), _i8(nst), _f(n, 10), _f(n, PD_RING_N, 3), _f(n, N_PDA), _f(RECUT_SLOTS, N_RA))


def empty_accum(nsec: int = 1) -> CutAccum:
    return CutAccum(_f(N_CA), _f(max(int(nsec), 1), 2))


def empty_out(nst: int = 1, nsec: int = 1) -> StepOut:
    return StepOut(_f(6), _f(max(int(nst), 1), 6), _f(3), _f(N_SO), _f(max(int(nsec), 1) + 1, 9))


def core_accum(n_ticks: int = 1, n_axes: int = 1, nt: int = 1, ne: int = 1) -> Accum:
    """A core.abi.Accum with the shapes the cutting kernels write."""
    return Accum(_f(N_ACC), _f(N_ELEM_ACC, max(nt, 1), max(ne, 1)), _f(max(n_ticks, 1), 4, 6),
                 _f(max(n_ticks, 1), max(n_axes, 1)), _f(max(n_ticks, 1)))


def empty_events(capacity: int = 64) -> CutEvents:
    cap = max(int(capacity), 1)
    return CutEvents(event_buffer(cap, SHOCK_WIDTH), event_buffer(cap, SURFACE_WIDTH), event_buffer(cap, DEPOSIT_WIDTH),
                     event_buffer(cap, RECUT_WIDTH), event_buffer(cap, PASS_WIDTH), event_buffer(cap, REFILL_WIDTH))


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
TOOL_T = numba.typeof(empty_tool())
TABLE_T = numba.typeof(empty_table())
MEMORY_T = numba.typeof(empty_memory())
INTERVALS_T = numba.typeof(empty_intervals())
REFILL_T = numba.typeof(empty_refill())
POSE_T = numba.typeof(identity_pose())
DEFLECTION_T = numba.typeof(empty_deflection())
OCCUPANCY_T = numba.typeof(empty_occupancy())
PACKING_T = numba.typeof(empty_packing())
AXES_T = numba.typeof(empty_axes())
POLAR_T = numba.typeof(empty_polar())
STATE_T = numba.typeof(empty_state())
ACCUM_T = numba.typeof(core_accum())
CACCUM_T = numba.typeof(empty_accum())
OUT_T = numba.typeof(empty_out())
EVENTS_T = numba.typeof(empty_events())
COEFF_T = numba.typeof(CoeffTable(_f(2), _f(1, 2, 3), _f(1, 3), _f(1, 2, 3), _f(1), _f(1), _f(1, 1), _f(1)))

__all__ = [name for name in dir() if name.isupper() or name in (
    "CutTool", "CutTable", "Refill", "StepPose", "Deflection", "CutOccupancy", "CutPacking", "AxisMap", "PolarView",
    "CutState", "CutAccum", "StepOut", "CutEvents", "empty_tool", "empty_table", "empty_memory", "empty_intervals",
    "memory_rows", "row_station", "pack_elements", "with_packed", "edge_curvature",
    "empty_refill", "identity_pose", "empty_deflection", "empty_occupancy", "empty_packing", "empty_axes",
    "empty_polar", "empty_state", "empty_accum", "empty_out", "core_accum", "empty_events", "is_kernel_layout",
    "Accum", "EventBuffer", "SurfaceMemory", "EngagedIntervals", "CoeffTable")]
