"""Kernel layouts of the coolant domain (docs/models/coolant.md "Kernels", architecture 5.5).

Every layout is a NamedTuple of writable, C-contiguous float64/int64/int8 arrays. Frames: rays,
polylines, tool envelopes, boxes and zones are in M; the in-process workpiece bricks in S (with the
S pose in M per grid); the chip pile in its grid G fixed to S (``axes_S`` columns e_u, e_v, e_w).

* :class:`RaySet` - the flow-weighted rays of all jets of an occlusion job;
* :class:`Revolved` - rotating tool + holder + spindle-nose envelopes as radius tables R(z) in T0;
* :class:`Boxes` - oriented boxes (non-rotating tools);
* :class:`BrickGrid` - in-process workpieces as bricks (empty / full / partial with sub-cells), the
  layout the brick-level 3-D DDA walks (filled through the WorkpieceBricks protocol);
* :class:`HeightField` - the chip pile and the surface under it (bus chips record);
* :class:`Zones` - cutting-zone spheres;
* :class:`TraceOut` - per-ray first hits, stored polylines, zone entries and closest approaches.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------- TraceParams scalar slots
TP_GX = 0               # gravity in M (m/s2)
TP_GY = 1
TP_GZ = 2
TP_RHO_AIR = 3          # ambient air (kg/m3)
TP_MU_AIR = 4           # (Pa s)
TP_DS_MAX = 5           # polyline step bounds (m)
TP_DS_MIN = 6
TP_SAGITTA = 7          # chord-to-trajectory tolerance (m)
TP_S_MAX = 8            # longest occluded trace (m)
TP_CF_COLUMN = 9        # skin friction of a coherent liquid column
TP_SN_A = 10            # Schiller-Naumann a, n, Newton C_D, Re of the Newton regime
TP_SN_N = 11
TP_CD_NEWTON = 12
TP_RE_NEWTON = 13
TP_BOX_LO = 14          # enclosure interior in M: 14..16 lo, 17..19 hi
TP_BOX_HI = 17
N_TP = 20

# ---------------------------------------------------------------- hit kinds
HIT_NONE = 0
HIT_TOOL = 1            # rotating tool / holder / spindle envelope (hit_part: tooling PART_*)
HIT_BOX = 2             # oriented box (non-rotating tool body)
HIT_WORK = 3            # in-process workpiece
HIT_CHIPS = 4           # chip layer of the pile
HIT_BASE = 5            # surface under the pile grid (part top, fixture, table)
HIT_MESH = 6            # machine / fixture mesh (set by the mesh pass)
HIT_LOST = 7            # left the enclosure or the trace length
HIT_NAMES = ("none", "tool", "box", "workpiece", "chips", "surface", "mesh", "lost")

# ---------------------------------------------------------------- brick states
BRICK_EMPTY = 0
BRICK_FULL = 1
BRICK_PARTIAL = 2

# ---------------------------------------------------------------- impingement rows of the film map
IMP_U = 0               # position on the grid (G, m)
IMP_V = 1
IMP_FLOW = 2            # liquid (or gas) volume flow arriving (m3/s)
IMP_SPEED = 3           # impact speed (m/s)
IMP_D = 4               # jet diameter at impact (m)
IMP_RHO = 5             # density of the impinging fluid (kg/m3)
IMP_NU = 6              # kinematic viscosity (m2/s)
IMP_KIND = 7            # 0 liquid, 1 gas, 2 mist (gas carrying drops that wet)
IMP_RADIUS = 8          # mist: radius of the spray footprint (m)
IMP_WIDTH = 9

FP_WATSON_R0 = 0        # film map parameters
FP_WATSON_C = 1
FP_WALL_DECAY = 2
FP_STEP = 3             # film climbs at most this much between neighbouring cells (m)
N_FP = 4


class RaySet(NamedTuple):
    origin_M: np.ndarray        # [n, 3]
    vel_M: np.ndarray           # [n, 3] initial velocity (m/s)
    weight: np.ndarray          # [n] flow the ray carries (m3/s at the jet state)
    jet: np.ndarray             # [n] int64 jet index
    gas: np.ndarray             # [n] int64: 1 straight, decaying gas jet; 0 ballistic liquid
    breakup_s: np.ndarray       # [n] arc length of liquid break-up (inf: none)
    drop_d: np.ndarray          # [n] drop diameter after break-up (m)
    column_d: np.ndarray        # [n] coherent column diameter (m)
    rho_l: np.ndarray           # [n] liquid density (kg/m3; 0 for gas)
    core_s: np.ndarray          # [n] gas: potential-core length B d_e (m)
    s_aim: np.ndarray           # [n] arc length up to which zone entries are searched past the first hit
    skip_tool: np.ndarray       # [n] int64 Revolved row the ray ignores (its own tool: through-tool jets)


class Revolved(NamedTuple):
    tip_M: np.ndarray           # [nt, 3]
    R_M_T: np.ndarray           # [nt, 3, 3] columns x_T0, y_T0, z_T in M
    z0: np.ndarray              # [nt] first node (m along +z_T)
    dz: np.ndarray              # [nt] node spacing
    n_nodes: np.ndarray         # [nt] int64
    radius: np.ndarray          # [nt, n_max] envelope radius at the nodes
    part: np.ndarray            # [nt, n_max] int64 part of the interval node k .. k+1
    r_max: np.ndarray           # [nt]


class Boxes(NamedTuple):
    centre_M: np.ndarray        # [nb, 3]
    R_M_B: np.ndarray           # [nb, 3, 3] box axes in M (columns)
    half: np.ndarray            # [nb, 3] half extents
    owner: np.ndarray           # [nb] int64 tool slot


class BrickGrid(NamedTuple):
    lo_S: np.ndarray            # [ng, 3] corner of brick (0, 0, 0) in S
    brick_m: np.ndarray         # [ng] brick edge
    dims: np.ndarray            # [ng, 3] int64 bricks per axis
    offset: np.ndarray          # [ng] int64 first brick of grid g in state/sub (x fastest)
    R_M_S: np.ndarray           # [ng, 3, 3] S axes in M
    o_M_S: np.ndarray           # [ng, 3] S origin in M
    n_sub: np.ndarray           # [ng] int64 sub-cells per brick edge
    state: np.ndarray           # [nb_total] int8 BRICK_*
    sub: np.ndarray             # [nb_total] int64 first sub-cell of a partial brick in occ (-1 otherwise)
    occ: np.ndarray             # [n_occ] int8 sub-cell material flag (x fastest)
    workpiece: np.ndarray       # [ng] int64


class HeightField(NamedTuple):
    origin_S: np.ndarray        # [3] corner of cell (0, 0) at w = 0, in S
    axes_S: np.ndarray          # [3, 3] columns e_u, e_v, e_w in S
    cell: np.ndarray            # [1]
    base_w: np.ndarray          # [nu, nv] w of the surface under the cell (-inf: none)
    top_w: np.ndarray           # [nu, nv] top of the chips (= base_w without chips)
    R_M_S: np.ndarray           # [3, 3]
    o_M_S: np.ndarray           # [3]
    active: np.ndarray          # [1] int64: 0 = no pile grid


class Zones(NamedTuple):
    centre_M: np.ndarray        # [nz, 3]
    radius: np.ndarray          # [nz]


class TraceOut(NamedTuple):
    hit_kind: np.ndarray        # [n] int64 HIT_*
    hit_index: np.ndarray       # [n] int64 occluder row (tool, box, grid, mesh)
    hit_part: np.ndarray        # [n] int64 part of a tool hit
    hit_s: np.ndarray           # [n] arc length of the first hit (inf: none)
    hit_M: np.ndarray           # [n, 3]
    hit_vel_M: np.ndarray       # [n, 3] velocity at the hit
    npts: np.ndarray            # [n] int64 stored polyline points (up to the first hit)
    pts_M: np.ndarray           # [n, m, 3]
    pts_s: np.ndarray           # [n, m]
    pts_v: np.ndarray           # [n, m] speed
    zone_s: np.ndarray          # [n, nz] first entry along the unobstructed path (inf: none)
    zone_v: np.ndarray          # [n, nz] speed there
    zone_spray: np.ndarray      # [n, nz] int8: entry beyond break-up
    zone_dmin: np.ndarray       # [n, nz] closest approach to the zone centre
    zone_smin: np.ndarray       # [n, nz] arc length of the closest approach


def trace_out(n: int, n_zones: int, max_pts: int) -> TraceOut:
    return TraceOut(np.zeros(n, np.int64), np.full(n, -1, np.int64), np.full(n, -1, np.int64), np.full(n, np.inf),
                    np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n, np.int64), np.zeros((n, max_pts, 3)),
                    np.zeros((n, max_pts)), np.zeros((n, max_pts)), np.full((n, n_zones), np.inf),
                    np.zeros((n, n_zones)), np.zeros((n, n_zones), np.int8), np.full((n, n_zones), np.inf),
                    np.zeros((n, n_zones)))


def empty_revolved(n_max: int = 2) -> Revolved:
    return Revolved(np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros(0), np.ones(0), np.zeros(0, np.int64),
                    np.zeros((0, n_max)), np.zeros((0, n_max), np.int64), np.zeros(0))


def empty_boxes() -> Boxes:
    return Boxes(np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros((0, 3)), np.zeros(0, np.int64))


def empty_bricks() -> BrickGrid:
    return BrickGrid(np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3), np.int64), np.zeros(0, np.int64),
                     np.zeros((0, 3, 3)), np.zeros((0, 3)), np.zeros(0, np.int64), np.zeros(0, np.int8),
                     np.zeros(0, np.int64), np.zeros(0, np.int8), np.zeros(0, np.int64))


def empty_heightfield() -> HeightField:
    return HeightField(np.zeros(3), np.eye(3), np.ones(1), np.full((1, 1), -np.inf), np.full((1, 1), -np.inf),
                       np.eye(3), np.zeros(3), np.zeros(1, np.int64))


def empty_rays() -> RaySet:
    z = np.zeros(0)
    return RaySet(np.zeros((0, 3)), np.zeros((0, 3)), z.copy(), np.zeros(0, np.int64), np.zeros(0, np.int64), z.copy(),
                  z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), np.zeros(0, np.int64))


def contiguous(array, dtype=None) -> np.ndarray:
    out = np.ascontiguousarray(array, dtype=dtype)
    if not out.flags.writeable:
        out = out.copy()
    return out
