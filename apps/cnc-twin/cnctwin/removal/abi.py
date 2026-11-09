"""Kernel layouts owned by removal (docs/models/removal.md "Layouts"; rules: docs/architecture.md 5.5).

Every layout is a NamedTuple of writable, C-contiguous NumPy arrays, so Numba types it concretely and
every kernel has exactly one compiled specialisation. Python builds and grows them at L2 (cnctwin.removal
.field, .polar, .profile); the compiled kernels read and update them in place.

Tri-dexel field (:class:`DexelField`)
-------------------------------------
* Several **levels** share the arrays: level 0 is the base lattice over the stock, levels >= 1 are refine
  boxes with a finer spacing (``physics.refine``). Level l has nodes ``origin[l] + (i, j, k) * cell[l]``
  for ``0 <= i < dims[l, 0]`` ...; the spacing is an integer number of fixed-point quanta.
* Three **families** of rays per level run along the lattice lines through the nodes: family 0 along x
  through (y_j, z_k), family 1 along y through (x_i, z_k), family 2 along z through (x_i, y_j); the
  families share the nodes. Ray index = ``ray0[l, f] + iu * dims[l, v] + iv`` with the transverse axes
  ``(u, v) = TRANS[f]`` ((1, 2), (0, 2), (0, 1)).
* A ray holds sorted, disjoint, non-touching half-open segments ``[lo, hi)`` of material, endpoints as
  int32 **fixed-point coordinates along the ray in QUANTUM_M (10 nm) from the level origin**; the first
  ``K_INLINE`` segments inline, the rest in a chain of pool blocks (``link`` / ``pnext``, -1 ends). Each
  endpoint carries the int32 surface event id of the cut (or the stock) that made it.
* **Bricks** of ``BRICK`` (32) cells per axis: brick (bx, by, bz) of level l covers the nodes
  ``32 b .. 32 b + 32``; ``bmat[brick, f]`` is the material length (quanta) of the family-f rays it owns
  (node index // 32, clamped to the last brick), so a brick with all three zero is empty; ``bdirty`` marks
  changes since the last keyframe, ``bconn`` since the last connectivity pass.

Volumes: the field volume of a level is ``cell**2 * QUANTUM_M * sum(z-family lengths)``, an exact integer
sum (architecture 5.4, 10); the ledger uses level 0.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

QUANTUM_M = 1.0e-8          # fixed-point unit of every ray endpoint (10 nm)
K_INLINE = 4                # segments stored inline per ray (and per pool block)
BRICK = 32                  # cells per brick edge
MAX_SEG = 2048              # segments a kernel holds for one ray in its scratch (more: FI_SCRATCH_OVF)
NO_LINK = -1

FAM_X = 0
FAM_Y = 1
FAM_Z = 2
TRANS_U = (1, 0, 0)         # first transverse axis of family f
TRANS_V = (2, 2, 1)         # second transverse axis of family f

# surface event ids carried by endpoints
SEV_NONE = -1               # unknown
SEV_STOCK = 0               # the stock surface as delivered (and target surfaces)
SEV_FIRST_CUT = 1           # the first id a subtraction may use

# DexelField.info
FI_VERSION = 0              # increments on every geometry change (bus field_version)
FI_NLEVELS = 1
FI_NRAYS = 2
FI_OVERFLOW = 3             # a write needed more pool blocks than were free: caller grows the pool and retries
FI_FREE_HEAD = 4            # first free pool block (-1 none)
FI_N_FREE = 5
FI_N_BLOCKS = 6             # pool capacity in blocks
FI_MAX_SEG = 7              # largest segment count stored on one ray
FI_SCRATCH_OVF = 8          # a ray exceeded MAX_SEG segments in a kernel scratch (its update was refused)
FI_PEAK_USED = 9            # most pool blocks in use at once
FI_WORKPIECE = 10
FI_TOPO = 11                # level-0 z rays whose segment count changed since the last connectivity pass
N_FI = 12


class DexelField(NamedTuple):
    """Tri-dexel field of one workpiece in the stock frame S, material configuration (architecture 4.1)."""

    lv_origin_m: np.ndarray     # [nl, 3] f8 position of node (0, 0, 0) in S
    lv_cell_m: np.ndarray       # [nl] f8 lattice spacing
    lv_cell_q: np.ndarray       # [nl] i8 lattice spacing in quanta
    lv_dims: np.ndarray         # [nl, 3] i8 nodes per axis
    lv_ray0: np.ndarray         # [nl, 3] i8 first ray index of each family
    lv_bdims: np.ndarray        # [nl, 3] i8 bricks per axis
    lv_brick0: np.ndarray       # [nl] i8 first brick index
    lv_box_lo_m: np.ndarray     # [nl, 3] f8 region the level is authoritative for (queries pick the finest)
    lv_box_hi_m: np.ndarray     # [nl, 3]
    cnt: np.ndarray             # [nray] i4 segments of the ray
    seg: np.ndarray             # [nray, K, 2] i4 inline endpoints (lo, hi), quanta from the level origin
    sev: np.ndarray             # [nray, K, 2] i4 surface event id of each inline endpoint
    link: np.ndarray            # [nray] i4 first overflow block (-1 none)
    pseg: np.ndarray            # [nblk, K, 2] i4 pool endpoints
    psev: np.ndarray            # [nblk, K, 2] i4 pool event ids
    pnext: np.ndarray           # [nblk] i4 next block of a chain / of the free list (-1 ends)
    bmat: np.ndarray            # [nbrick, 3] i8 material length per family owned by the brick (quanta)
    bdirty: np.ndarray          # [nbrick] i1 changed since the last keyframe
    bconn: np.ndarray           # [nbrick] i1 changed since the last connectivity pass
    bver: np.ndarray            # [nbrick] i4 change counter
    info: np.ndarray            # [N_FI] i8


class TargetField(NamedTuple):
    """The CAD part on the same rays (static, compressed rows): ray r's target segments are
    ``seg[start[r]:start[r + 1]]`` (quanta, half-open). An empty target has ``start`` all zero."""

    start: np.ndarray           # [nray + 1] i8
    seg: np.ndarray             # [m, 2] i4
    info: np.ndarray            # [4] i8: [present, n_segments, 0, 0]


# --------------------------------------------------------------------------- revolved profiles
PK_LINE = 0                 # R(z) = c0 + c1 z
PK_ARC = 1                  # R(z) = c1 + c3 * sqrt(c2^2 - (z - c0)^2): centre (zc = c0, rc = c1), rho = c2, sigma = c3

PI_ZMIN = 0
PI_ZMAX = 1
PI_RMAX = 2
PI_RDEAD = 3                # radius of the non-cutting centre of the end face (0: centre cutting)
N_PI = 4


class Profile(NamedTuple):
    """Max-radius profile R(z) of a solid of revolution about +z (the tool axis, z from the tip): sorted,
    non-overlapping sub-pieces ``[z0, z1]`` on each of which R is one line or one z-monotone circular arc.
    The solid at height z is the disc of radius R(z) (filled to the axis; tooling Envelope convention)."""

    z0_m: np.ndarray            # [m] f8
    z1_m: np.ndarray            # [m] f8
    kind: np.ndarray            # [m] i4 PK_*
    c0: np.ndarray              # [m] f8 line: intercept; arc: zc
    c1: np.ndarray              # [m] f8 line: slope;     arc: rc
    c2: np.ndarray              # [m] f8 arc: rho
    c3: np.ndarray              # [m] f8 arc: sigma (+1 outer half, -1 inner half)
    part: np.ndarray            # [m] i4 tooling PART_* of the piece giving the maximum
    rmax_m: np.ndarray          # [m] f8 largest R on the sub-piece
    info: np.ndarray            # [N_PI] f8


# --------------------------------------------------------------------------- per-contact accumulators
# RemovalAccum.ivals
IA_SUBTRACTIONS = 0         # subtraction calls
IA_RAYS_CHANGED = 1
IA_OVERFLOW = 2             # calls that hit a pool overflow (retried by the caller)
IA_SCRATCH_OVF = 3
IA_EVENTS_DROPPED = 4       # event rows that did not fit (EventBuffer n[1] is set as well)
IA_LAST_TICK = 5            # tick of the last subtraction that removed material
IA_FIRST_TICK = 6           # tick of the first one in the step (-1: none)
IA_IN_CUT_CALLS = 7         # subtraction calls that removed material
IA_AIR_TICKS = 8            # L1 ticks of feed motion with the cutting envelope out of contact
IA_FEED_TICKS = 9           # L1 ticks of feed motion (in or out of contact)
IA_POLAR_Q2 = 10            # polar removed sum of delta(q_out^2 - q_in^2) (radial family, int64)
IA_POLAR_STEPS = 11
N_IA = 12

# RemovalAccum.fvals
FA_CX = 0                   # sum of removed z-family length (m) x a^2 x midpoint (m^4): centroid numerator
FA_CY = 1
FA_CZ = 2
FA_W = 3                    # sum of removed z-family volume (m^3) (level 0 + refined levels weighted)
FA_NX = 4                   # sum of volume x machined-surface outward normal at the removal (S)
FA_NY = 5
FA_NZ = 6
FA_ZREL_MIN = 7             # removed material along the tool axis from the tip (m): min / max
FA_ZREL_MAX = 8
FA_RREL_MAX = 9             # largest distance of removed material from the tool axis
FA_ANGLE_MAX = 10           # largest engaged angle of a station (rad), written by the gating kernel
FA_ARC_MAX = 11             # largest engaged arc length at one station (m)
FA_AE_MAX = 12              # largest radial engagement (m) seen in the step
FA_AIR_X0 = 13              # first air-cut tip position in the step (S)
FA_AIR_Y0 = 14
FA_AIR_Z0 = 15
FA_AIR_X1 = 16              # last air-cut tip position in the step (S)
FA_AIR_Y1 = 17
FA_AIR_Z1 = 18
FA_M_FROM_S = 19            # 12 values: rotation (row-major 3x3) + translation of M_from_S at the last removal
FA_ENGAGED_LEN = 31         # largest engaged edge length at one gating (m)
FA_DEAD_V = 32              # volume removed by a non-cutting centre while plunging (m^3)
N_FA = 36


class RemovalAccum(NamedTuple):
    """What the compiled loop accumulates for one contact over a macro step (absorb reads it)."""

    removed_q: np.ndarray       # [nl, 3] i8 removed length per level and family (quanta)
    ivals: np.ndarray           # [N_IA] i8
    fvals: np.ndarray           # [N_FA] f8


# --------------------------------------------------------------------------- events of the hot loop
EV_RAPID_INTO_MATERIAL = 1
EV_COLLISION = 2            # holder / shank / neck / spindle / insert body / machine part / fixture in material
EV_GOUGE = 3                # removal of target material deeper than the gouge tolerance
EV_BREAKTHROUGH = 4         # gouge through the whole target thickness along a ray (air on the other side)
EV_NON_CENTRE_PLUNGE = 5    # material removed by the non-cutting centre while feeding axially into material
EV_POOL_OVERFLOW = 6        # the subtraction was refused for lack of pool blocks (the caller retries)
EV_FIXTURE_CONTACT = 7      # tool envelope within the fixture / machine-part mesh
EV_PROBE_CONTACT = 8
EV_NAMES = {EV_RAPID_INTO_MATERIAL: "rapid_into_material", EV_COLLISION: "collision", EV_GOUGE: "gouge",
            EV_BREAKTHROUGH: "breakthrough", EV_NON_CENTRE_PLUNGE: "non_centre_cutting_plunge",
            EV_POOL_OVERFLOW: "pool_overflow", EV_FIXTURE_CONTACT: "fixture_collision",
            EV_PROBE_CONTACT: "probe_contact"}

# columns of an event row (EventBuffer.rec)
EC_KIND = 0
EC_TICK = 1
EC_CONTACT = 2
EC_X = 3                    # point in S
EC_Y = 4
EC_Z = 5
EC_DEPTH = 6                # largest overlap along a ray (m)
EC_VOLUME = 7               # overlap volume (m^3; z family, or the largest family for thin overlaps)
EC_PART = 8                 # tooling PART_* (collisions), -1 otherwise
EC_SEV_ID = 9               # surface event id of the subtraction
EC_LEVEL = 10
EC_FAMILY = 11
EVENT_WIDTH = 12

# --------------------------------------------------------------------------- subtraction parameters
# SubtractSpec.ivals
SI_EVENT_ID = 0             # surface event id given to the endpoints the cut creates
SI_TICK = 1
SI_CONTACT = 2
SI_RAPID = 3                # 1: the motion is a rapid (any removal is a rapid into material)
SI_PLUNGE = 4               # 1: the tool feeds axially into the material faster than the ramp limit
SI_CHECK_TARGET = 5         # 1: gouge / breakthrough checks against the target field
SI_LEVEL_MASK = 6           # bit l set: level l is updated (all levels: -1)
N_SI = 8

# SubtractSpec.fvals
SF_GOUGE_TOL = 0            # overlap with the target below this (m along the ray) is not a gouge
SF_R_DEAD = 1               # non-cutting centre radius (m)
SF_Z_DEAD = 2               # height of the dead zone above the lowest point of the envelope (m)
SF_SLACK = 3                # extra radius of the quick-reject cylinder (m)
N_SF = 4


class SubtractSpec(NamedTuple):
    """One subtraction of a revolved envelope at a pose, per axial slab (architecture 4.1, 5.4).

    Slab k covers ``z in [slab_z[k], slab_z[k + 1]]`` along the unit axis ``axis_S`` from its own axis point
    ``slab_c[k]`` (tool tip of that slab's generation instant + u_rel(k), mapped to X), with the radius
    ``R(z) + slab_dr[k]`` (tooth radius offset: runout, wear, growth). Only rays through the box
    ``[box_lo, box_hi]`` (S) are touched and removal is clipped to it (the engaged-sector bound)."""

    axis_S: np.ndarray          # [3] f8 unit tool axis (+z_T) in S
    slab_c: np.ndarray          # [ns, 3] f8
    slab_z: np.ndarray          # [ns + 1] f8 ascending
    slab_dr: np.ndarray         # [ns] f8
    box_lo: np.ndarray          # [3] f8
    box_hi: np.ndarray          # [3] f8
    ivals: np.ndarray           # [N_SI] i8
    fvals: np.ndarray           # [N_SF] f8


# --------------------------------------------------------------------------- helical swept profiles (taps)
class HelixSweep(NamedTuple):
    """Tooth polygons of a tap swept along the relative helical path (never a revolved body: architecture
    5.4). Land l owns polygons ``land_first[l] .. land_first[l + 1] - 1`` (ascending in z), each a closed
    counter-clockwise convex polygon ``poly[j, :n[j]]`` in the tool's meridian plane (z_T, r) at the land's
    cutting-sense phase ``land_phase[l]``. The kernel moves the tool between the two poses of a call in
    proportion to the spindle angle, so the helix (and any feed / speed mismatch) comes from the motion."""

    poly: np.ndarray            # [nj, nmax, 2] f8 (z, r)
    n: np.ndarray               # [nj] i4
    zr_box: np.ndarray          # [nj, 4] f8 z_lo, z_hi, r_lo, r_hi of each polygon
    land_first: np.ndarray      # [nlands + 1] i4
    land_phase: np.ndarray      # [nlands] f8
    info: np.ndarray            # [4] f8: hand sign h (alpha = theta + h phase), z_lo, z_hi, r_hi (all polygons)


# --------------------------------------------------------------------------- polar field (turning)
PG_NTHETA = 0               # PolarField.geom
PG_DTHETA = 1
PG_Z0 = 2                   # z of radial ray row 0 (S, along the spindle axis = S z)
PG_DZ = 3
PG_NZ = 4
PG_R0 = 5                   # radius of axial ray column 0
PG_DR = 6
PG_NR = 7
PG_QUANTUM = 8
N_PG = 10

PI_VERSION = 0              # PolarField.info
PI_OVERFLOW = 1             # bit 1: a radial ray needed more than KPR segments, bit 2: an axial ray more than KPA
PI_DIRTY_Z0 = 2             # z-row range changed since the last tri-dexel sync (inclusive; -1 none)
PI_DIRTY_Z1 = 3
PI_DIRTY_R0 = 4             # axial-column range changed since the last sync
PI_DIRTY_R1 = 5
PI_KP = 6                   # inline segments per radial ray (KPR)
PI_KPA = 7                  # inline segments per axial ray (KPA; threads leave one segment per crest)
N_PINFO = 8


POLAR_BLOCK = 16            # rows per zmax_q block of the polar field (the per-step skip granularity)


class PolarField(NamedTuple):
    """Rotating-workpiece field about the S z axis (architecture 5.4 "Turning"): radial rays at
    (theta_b, z_i) with segments in r (quanta from the axis) and axial rays at (theta_b, r_j) with segments
    in z (quanta from z0), KPR (radial) and KPA (axial) segments inline (no pool: a ray needing more sets
    PI_OVERFLOW and is left unchanged; the caller re-allocates with a larger capacity and repeats the
    update, which is idempotent).
    Radial ray index ``b * nz + i``, axial ``b * nr + j`` (theta-major). ``zmax_q[b * nblk + i // POLAR_BLOCK]`` is
    the largest outer radius of the rows of block i // POLAR_BLOCK in bin b (kept exact by every cut), so a step skips
    every block its insert cannot reach."""

    geom: np.ndarray            # [N_PG] f8
    rcnt: np.ndarray            # [ntheta * nz] i1
    rseg: np.ndarray            # [ntheta * nz, KPR, 2] i4
    rsev: np.ndarray            # [ntheta * nz, KPR, 2] i4
    acnt: np.ndarray            # [ntheta * nr] i1
    aseg: np.ndarray            # [ntheta * nr, KPA, 2] i4
    asev: np.ndarray            # [ntheta * nr, KPA, 2] i4
    zmax_q: np.ndarray          # [ntheta * ceil(nz / POLAR_BLOCK)] i8
    info: np.ndarray            # [N_PINFO] i8


class InsertRegion(NamedTuple):
    """Convex 2-D region of a non-rotating tool's insert in its reference plane (z_T, x_T), as monotone
    chains for fast per-ray evaluation (built from tooling's InsertProfile; exact nose arc added).

    For a given z_T the insert occupies ``x_T in [lo(z), hi(z)]`` (``zl, xl`` lower chain, ``zu, xu`` upper
    chain, both ascending in z); for a given x_T, ``z_T in [left(x), right(x)]`` (``xa, za`` left chain,
    ``xb, zb`` right chain, ascending in x). ``nose`` = (z_c, x_c, rho) of the nose arc whose upper / left
    sides are exact (rho = 0: none)."""

    zl: np.ndarray              # [nl] f8
    xl: np.ndarray
    zu: np.ndarray              # [nu] f8
    xu: np.ndarray
    xa: np.ndarray              # [na] f8
    za: np.ndarray
    xb: np.ndarray              # [nb] f8
    zb: np.ndarray
    nose: np.ndarray            # [3] f8
    box: np.ndarray             # [6] f8 z_min, z_max, x_min, x_max, z of the upper chain's highest vertex,
                                #     1 when the upper chain rises to it and falls after it (unimodal) else 0


# --------------------------------------------------------------------------- polar cut of one L0 step
# PolarStep.vals
PS_Z0 = 0                   # tool tip P (z along the work axis, r from it) at the start of the step
PS_R0 = 1
PS_Z1 = 2                   # ... at the end of the step
PS_R1 = 3
PS_TH0 = 4                  # angle of the tool's reference plane in the rotating frame S (unwrapped), start
PS_TH1 = 5                  # ... end
PS_SZ = 6                   # z_T along +S z (+1) or -S z (-1)
PS_SX = 7                   # x_T along -e_r (-1: external tool, into the work toward the axis) or +e_r (+1)
PS_YOFF = 8                 # offset of the reference plane from the axis (centre height error, m)
N_PS = 10


MAX_IV = 512                # removal intervals a kernel collects for one ray
MAX_PIECES = 16384          # removed pieces a subtraction records before it books them (then it books and goes on)


class Scratch(NamedTuple):
    """Work lists of the ray kernels (one per caller; kernels never allocate per ray)."""

    lo: np.ndarray              # [MAX_SEG] i4 segments of the ray being changed
    hi: np.ndarray
    elo: np.ndarray
    ehi: np.ndarray
    nlo: np.ndarray             # [MAX_SEG] i4 the new segments
    nhi: np.ndarray
    nelo: np.ndarray
    nehi: np.ndarray
    plo: np.ndarray             # [MAX_SEG] i8 pieces removed or added
    phi: np.ndarray
    rlo: np.ndarray             # [MAX_IV] i8 intervals to remove / add (quanta)
    rhi: np.ndarray
    ivt: np.ndarray             # [MAX_IV, 2] f8 intervals along the ray in metres before quantisation
    ivp: np.ndarray             # [MAX_IV] i4 part / piece tag of each interval
    pcs: np.ndarray             # [MAX_PIECES, 7] i8 removed pieces (ray, level, family, iu, iv, q0, q1)
    # small fixed work of the kernels compiled without reference counting (nbt.RAW: they allocate nothing)
    fw: np.ndarray              # [N_FW] f8, sub-ranges FW_*
    iv2: np.ndarray             # [16, 2] f8 short interval lists (dead-zone intervals of post_pieces)
    tg: np.ndarray              # [16] i4 their tags
    iw: np.ndarray              # [16] i8 small integer work
    rows: np.ndarray            # [8, 12] f8 event rows written once per call


# Scratch.fw sub-ranges (subtract_envelope / post_pieces)
FW_ELO = 0                  # [3] envelope box
FW_EHI = 3
FW_BLO = 6                  # [3] envelope box clipped to the spec's box
FW_BHI = 9
FW_P = 12                   # [3] ray origin
FW_PP = 15                  # [3] post_pieces' point
FW_WORK = 18                # [16] root-isolation workspace of the envelope intervals
FW_ENV = 34                 # [N_ENV] slab bounds of the current subtraction and the ray's material extent
                            #   (k_envelope ENV_*): right after the workspace, read as work[16:16 + N_ENV]
N_ENV = 11
FW_AGG = 45                 # [64] aggregates of one subtraction (k_subtract.AG_*)
N_FW = 128


def make_scratch() -> Scratch:
    m, n = MAX_SEG, MAX_IV
    return Scratch(*(np.zeros(m, np.int32) for _ in range(8)), np.zeros(m, np.int64), np.zeros(m, np.int64),
                   np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros((n, 2)), np.zeros(n, np.int32),
                   np.zeros((MAX_PIECES, 7), np.int64), np.zeros(N_FW), np.zeros((16, 2)), np.zeros(16, np.int32),
                   np.zeros(16, np.int64), np.zeros((8, 12)))


class Mesh3(NamedTuple):
    """Triangle mesh arrays produced by the meshing kernels (positions in S)."""

    vertices: np.ndarray        # [nv, 3] f8
    faces: np.ndarray           # [nf, 3] i4 counter-clockwise from outside
    sev: np.ndarray             # [nv] i4 surface event id of the crossing that made the vertex
    edge_id: np.ndarray         # [nv] i8 global lattice edge id (node * 3 + axis) for welding bricks


def i4(n: int, fill: int = 0) -> np.ndarray:
    return np.full(int(n), fill, np.int32)


def empty_target(nray: int) -> TargetField:
    return TargetField(np.zeros(int(nray) + 1, np.int64), np.zeros((0, 2), np.int32), np.zeros(4, np.int64))


def new_accum(n_levels: int) -> RemovalAccum:
    acc = RemovalAccum(np.zeros((max(1, int(n_levels)), 3), np.int64), np.zeros(N_IA, np.int64), np.zeros(N_FA))
    reset_accum(acc)
    return acc


def reset_accum(acc: RemovalAccum) -> None:
    acc.removed_q[...] = 0
    acc.ivals[...] = 0
    acc.ivals[IA_FIRST_TICK] = -1
    acc.ivals[IA_LAST_TICK] = -1
    acc.fvals[...] = 0.0
    acc.fvals[FA_ZREL_MIN] = np.inf
    acc.fvals[FA_ZREL_MAX] = -np.inf
    acc.fvals[FA_M_FROM_S:FA_M_FROM_S + 9] = np.eye(3).ravel()


def subtract_spec(axis_S, slab_c, slab_z, slab_dr, box_lo, box_hi, *, event_id: int = SEV_FIRST_CUT, tick: int = 0,
                  contact: int = 0, rapid: bool = False, plunge: bool = False, check_target: bool = True,
                  level_mask: int = -1, gouge_tol_m: float = 5e-6, r_dead_m: float = 0.0, z_dead_m: float = 0.0,
                  slack_m: float = 0.0) -> SubtractSpec:
    """A SubtractSpec from plain values (Python side; the hot loop fills the arrays in place)."""
    iv = np.zeros(N_SI, np.int64)
    iv[SI_EVENT_ID] = int(event_id)
    iv[SI_TICK] = int(tick)
    iv[SI_CONTACT] = int(contact)
    iv[SI_RAPID] = 1 if rapid else 0
    iv[SI_PLUNGE] = 1 if plunge else 0
    iv[SI_CHECK_TARGET] = 1 if check_target else 0
    iv[SI_LEVEL_MASK] = int(level_mask)
    fv = np.zeros(N_SF)
    fv[SF_GOUGE_TOL] = float(gouge_tol_m)
    fv[SF_R_DEAD] = float(r_dead_m)
    fv[SF_Z_DEAD] = float(z_dead_m)
    fv[SF_SLACK] = float(slack_m)
    c = np.ascontiguousarray(np.asarray(slab_c, dtype=np.float64).reshape(-1, 3))
    z = np.ascontiguousarray(np.asarray(slab_z, dtype=np.float64).reshape(-1))
    dr = np.ascontiguousarray(np.asarray(slab_dr, dtype=np.float64).reshape(-1))
    if len(z) != len(c) + 1 or len(dr) != len(c):
        raise ValueError("a SubtractSpec needs ns slab centres, ns + 1 slab heights and ns radius offsets")
    if np.any(np.diff(z) < 0.0):
        raise ValueError("slab heights must ascend")
    w = np.asarray(axis_S, dtype=np.float64).reshape(3)
    w = w / np.linalg.norm(w)
    return SubtractSpec(np.ascontiguousarray(w), c.copy(), z.copy(), dr.copy(),
                        np.ascontiguousarray(np.asarray(box_lo, dtype=np.float64).reshape(3)).copy(),
                        np.ascontiguousarray(np.asarray(box_hi, dtype=np.float64).reshape(3)).copy(), iv, fv)


def empty_profile() -> Profile:
    z = np.zeros(0)
    return Profile(z.copy(), z.copy(), np.zeros(0, np.int32), z.copy(), z.copy(), z.copy(), z.copy(),
                   np.zeros(0, np.int32), z.copy(), np.zeros(N_PI))


def empty_insert() -> InsertRegion:
    z = np.zeros(1)
    return InsertRegion(z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), np.zeros(3),
                        np.array([0.0, -1.0, 0.0, -1.0]))


def empty_helix() -> HelixSweep:
    return HelixSweep(np.zeros((0, 3, 2)), np.zeros(0, np.int32), np.zeros((0, 4)), np.zeros(1, np.int32),
                      np.zeros(0), np.zeros(4))


# --------------------------------------------------------------------------- domain bundle for the hot loop
class ToothSlabs(NamedTuple):
    """Slab heights and radius offsets of every tooth of a path's active tool (profile.tooth_slabs), rebuilt
    at L2 when the effective edge changes. Tooth j uses ``z[j, :n[j] + 1]`` and ``dr[j, :n[j]]``."""

    z: np.ndarray               # [nt, ns_max + 1] f8
    dr: np.ndarray              # [nt, ns_max] f8
    n: np.ndarray               # [nt] i4


# RemovalInputs.iparams layout
RI_NPATHS = 0
RI_NWORK = 1
RI_PATH0 = 2                    # per path p: RI_PATH0 + RI_PSTRIDE * p + RP_*
RI_PSTRIDE = 8
RP_WORKPIECE = 0
RP_SLOT = 1
RP_KIND = 2                     # TK_*
RP_CENTRE_CUTTING = 3           # 1 centre cutting, 0 not
RP_ROTATING = 4
TK_NONE = 0
TK_ENVELOPE = 1                 # rotating tool: revolved envelope per tooth pass
TK_TURNING = 2                  # non-rotating tool on a turning spindle: polar insert cut per L0 step
TK_HELIX = 3                    # tap: tooth profiles swept along the helical path
# after the paths: the next surface event id of every workpiece (the loop increments it per subtraction)
# at RI_PATH0 + RI_PSTRIDE * n_paths + w

# RemovalInputs.fparams layout
RF_GOUGE_TOL = 0
RF_COLLISION_TOL = 1
RF_GATE_SAMPLE = 2
RF_GATE_TOL = 3
RF_GATE_MARGIN = 4
RF_SECTOR_MARGIN = 5
RF_SWEEP_STEP = 6
RF_PATH0 = 8                    # per path p: RF_PATH0 + RF_PSTRIDE * p + RQ_*
RF_PSTRIDE = 4
RQ_R_DEAD = 0
RQ_Z_DEAD = 1
RQ_TAN_RAMP_MAX = 2
RQ_SLAB_H = 3


class RemovalInputs(NamedTuple):
    """What removal hands to the compiled loop for one macro step (frozen layout; arrays updated in place).

    ``fields``/``targets``/``polars`` per workpiece; ``cut``/``whole``/``inserts``/``helix``/``slabs`` per
    controller path (the path's active tool; placeholders of the same types when a path has no tool of that
    kind); ``accum`` per contact (contact index = path index: one cutting contact per path); one shared
    event buffer and scratch. The tuples are homogeneous (Numba UniTuple)."""

    fields: tuple
    targets: tuple
    polars: tuple
    cut: tuple
    whole: tuple
    inserts: tuple
    helix: tuple
    slabs: tuple
    accum: tuple
    events: object
    scratch: Scratch
    iparams: np.ndarray
    fparams: np.ndarray


def empty_slabs() -> ToothSlabs:
    return ToothSlabs(np.zeros((1, 2)), np.zeros((1, 1)), np.zeros(1, np.int32))
