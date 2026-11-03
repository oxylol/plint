"""Kernel layouts owned by the surface domain (docs/models/surface.md "Kernels"; rules: architecture 5.5).

Everything a surface kernel reads or writes is a NamedTuple of writable, C-contiguous NumPy arrays
(float64 / int64), so every kernel has exactly one compiled specialisation.

* :class:`Passes` - the generating passes of one profile line, in processing (time) order: where the
  generating edge element was in the line frame (s along the profile, w along the line normal, m out of
  the profile plane), the line-frame components of its frame (radial and axial directions of the tool or
  work body the outline lives in, the cutting velocity, the outward surface normal), the path curvature,
  the minimum chip thickness, the template window and the built-up-edge overcut.
* :class:`Templates` - the generating outlines (2-D polylines) the passes refer to: for ``MODE_OUTLINE``
  the (radial, axial) points of the tool (rotating tools: meridian of the swept cutting edge; turning tools:
  the insert outline in the work's meridian frame), for ``MODE_DIRECT`` (u, w) points already in line
  coordinates relative to the pass origin (silhouettes, dents, fragments).
* :class:`Grid` - the uniform profile grid and the per-pass scratch buffers.

Line frame: unit vectors d (along the profile), n (outward surface normal of the line) and
m = d x n; a point P has line coordinates (P.d, P.n, P.m). Heights w grow away from the material, so
the generated surface is the lower envelope of the passes (architecture 4.1 material configuration).
"""

from __future__ import annotations

from typing import NamedTuple

import numba
import numpy as np

# ---------------------------------------------------------------- pass kinds and template modes
KIND_CUT = 0          # the pass cuts where it lies below the surface by at least h_min (Brammertz)
KIND_ADD = 1          # the template's heights are added to the surface (recut dents < 0, BUE fragments > 0)

MODE_DIRECT = 0       # template points are (u, w) in line coordinates relative to the pass origin
MODE_OUTLINE = 1      # template points are (r, z) of an outline; projected along the cutting velocity
MODE_PARABOLA = 2     # the element's own path: w = t v_n + curv t^2 / 2 (no outline)

# ---------------------------------------------------------------- columns of Passes.basis
B_RD, B_RN, B_RM = 0, 1, 2          # radial unit vector of the outline's body (line components)
B_AD, B_AN, B_AM = 3, 4, 5          # axial unit vector of the body
B_VD, B_VN, B_VM = 6, 7, 8          # unit cutting velocity at the element
B_ND, B_NN, B_NM = 9, 10, 11        # outward normal of the generated surface at the element
N_BASIS = 12

# ---------------------------------------------------------------- profile parameters (kernels.profile_params)
P_RA = 0
P_RQ = 1
P_RSK = 2
P_RKU = 3
P_RP = 4
P_RV = 5
P_RZ = 6
P_RT = 7
P_RSM = 8
P_RDQ = 9
P_N_SL = 10           # sampling lengths evaluated
P_N_ELEM = 11         # profile elements counted for RSm
N_P = 12
PARAM_NAMES = ("Ra_m", "Rq_m", "Rsk", "Rku", "Rp_m", "Rv_m", "Rz_m", "Rt_m", "RSm_m", "Rdq", "sampling_lengths",
               "profile_elements")


class Passes(NamedTuple):
    kind: np.ndarray        # [np] i8 KIND_*
    mode: np.ndarray        # [np] i8 MODE_*
    tpl: np.ndarray         # [np] i8 template index (-1: none)
    s: np.ndarray           # [np] f8 element position along the profile (m)
    w: np.ndarray           # [np] f8 element height along the line normal (m)
    m: np.ndarray           # [np] f8 element offset out of the profile plane (m)
    basis: np.ndarray       # [np, N_BASIS] f8 B_* columns
    elem_rz: np.ndarray     # [np, 2] f8 the element's (r, z) in its template's coordinates
    curv: np.ndarray        # [np] f8 normal curvature of the element path along the cutting velocity (1/m)
    hmin: np.ndarray        # [np] f8 minimum chip thickness of the pass (m)
    half: np.ndarray        # [np] f8 half-width of the window kept around the template's lowest point (m)
    t_max: np.ndarray       # [np] f8 longest projection along the cutting velocity (m)
    bue: np.ndarray         # [np, 2] f8 built-up edge: overcut depth (m), half-width along the outline (m)


class Templates(NamedTuple):
    off: np.ndarray         # [ntpl + 1] i8 start of each template in pts
    pts: np.ndarray         # [npts, 2] f8


class Grid(NamedTuple):
    u0: np.ndarray          # [1] f8 position of sample 0 along the profile (m)
    du: np.ndarray          # [1] f8 sample spacing (m)
    z: np.ndarray           # [n] f8 surface height (inf: not generated)
    hits: np.ndarray        # [n] i8 cutting passes that touched each sample
    tmp: np.ndarray         # [n] f8 scratch: the current pass rasterised (inf outside)
    tmp_slope: np.ndarray   # [n] f8 scratch: slope of the current pass
    ps: np.ndarray          # [cap] f8 scratch: polyline of the current pass (s)
    pw: np.ndarray          # [cap] f8 scratch: polyline of the current pass (w; nan = dropped point)


def new_passes(n: int) -> Passes:
    return Passes(np.zeros(n, np.int64), np.zeros(n, np.int64), np.full(n, -1, np.int64), np.zeros(n), np.zeros(n),
                  np.zeros(n), np.zeros((n, N_BASIS)), np.zeros((n, 2)), np.zeros(n), np.zeros(n), np.zeros(n),
                  np.zeros(n), np.zeros((n, 2)))


def new_templates(polylines: list) -> Templates:
    """Templates from a list of [m, 2] arrays (in order)."""
    off = np.zeros(len(polylines) + 1, np.int64)
    for i, p in enumerate(polylines):
        off[i + 1] = off[i] + len(p)
    pts = np.zeros((int(off[-1]), 2))
    for i, p in enumerate(polylines):
        if len(p):
            pts[off[i]:off[i + 1]] = np.asarray(p, dtype=float).reshape(-1, 2)
    return Templates(off, pts)


def new_grid(u0: float, du: float, n: int, cap: int) -> Grid:
    return Grid(np.array([float(u0)]), np.array([float(du)]), np.full(int(n), np.inf), np.zeros(int(n), np.int64),
                np.full(int(n), np.inf), np.zeros(int(n)), np.zeros(int(cap)), np.zeros(int(cap)))


PASSES_T = numba.typeof(new_passes(1))
TEMPLATES_T = numba.typeof(new_templates([np.zeros((2, 2))]))
GRID_T = numba.typeof(new_grid(0.0, 1.0, 2, 2))

__all__ = ["Passes", "Templates", "Grid", "new_passes", "new_templates", "new_grid", "PASSES_T", "TEMPLATES_T",
           "GRID_T", "KIND_CUT", "KIND_ADD", "MODE_DIRECT", "MODE_OUTLINE", "MODE_PARABOLA", "N_BASIS", "N_P",
           "PARAM_NAMES"] + [n for n in dir() if n.startswith(("B_", "P_"))]
