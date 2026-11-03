"""Kernel layouts of inspection (docs/models/inspect.md "Kernels"; rules: docs/architecture.md 5.5).

Every layout is a NamedTuple of writable, C-contiguous NumPy arrays, or a plain float64 row table whose
column indices are named here, so every kernel has exactly one compiled specialisation.

:class:`Columns` is the z family of a dexel lattice over the stock frame S: column ``c = i * ny + j`` runs
along +z through the node ``(x_i, y_j) = origin + (i, j) * cell`` and holds sorted, disjoint, non-touching
material intervals ``seg[start[c]:start[c + 1]]`` (metres, half-open ``[lo, hi)``). The column owns the cell
``[x_i - cx/2, x_i + cx/2] x [y_j - cy/2, y_j + cy/2]``, so a volume is ``cx * cy * sum(hi - lo)`` (the
midpoint rule removal uses for its own z family, architecture 5.4). The removal field and its CAD target
are read into this form through :class:`cnctwin.inspect.protocols.FieldColumns`; meshes are dexelised into
it by :func:`cnctwin.inspect.columns.columns_from_mesh`.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numba import typeof


class Columns(NamedTuple):
    origin_m: np.ndarray        # [2] f8 x, y of column (0, 0)
    cell_m: np.ndarray          # [2] f8 spacing along x and y
    dims: np.ndarray            # [2] i8 columns along x and y
    start: np.ndarray           # [nx * ny + 1] i8 CSR offsets
    seg: np.ndarray             # [m, 2] f8 z intervals [lo, hi)


# --------------------------------------------------------------------------- connected regions (component_stats_nb)
CS_VOLUME = 0               # m^3
CS_CX = 1                   # volume centroid
CS_CY = 2
CS_CZ = 3
CS_XMIN = 4                 # bounding box (cell extents in x, y; interval ends in z)
CS_YMIN = 5
CS_ZMIN = 6
CS_XMAX = 7
CS_YMAX = 8
CS_ZMAX = 9
CS_THICK = 10               # largest single-interval length (m)
CS_TX = 11                  # middle of that interval
CS_TY = 12
CS_TZ = 13
CS_COLUMNS = 14             # columns touched
CS_SEGMENTS = 15            # intervals
CS_FIRST = 16               # index of the first interval (ordering key)
N_CS = 17

# --------------------------------------------------------------------------- probe moves (probe_nb)
PR_T = 0                    # travel along the approach direction at contact (m)
PR_CX = 1                   # ball centre at contact
PR_CY = 2
PR_CZ = 3
PR_QX = 4                   # contact point on the surface (closest point to the centre)
PR_QY = 5
PR_QZ = 6
PR_FACE = 7                 # triangle of the contact
PR_STATUS = 8               # PROBE_* below
PR_ITER = 9                 # sphere-tracing steps used
N_PR = 10

PROBE_CONTACT = 0
PROBE_NO_CONTACT = 1        # nothing within the travel: material missing beyond the search distance
PROBE_START_IN_CONTACT = 2  # the ball already touches at the start point (prehit too short / excess material)
PROBE_NOT_CONVERGED = 3     # still approaching after the step budget (grazing approach)

# --------------------------------------------------------------------------- toolpath segments (chord_deviation_nb)
SG_KIND = 0                 # SEG_LINE | SEG_ARC
SG_P0 = 1                   # start point (3)
SG_P1 = 4                   # end point (3)
SG_C = 7                    # arc centre (3)
SG_N = 10                   # arc axis, unit, right-handed (3)
SG_SWEEP = 13               # signed sweep about SG_N (rad)
N_SG = 14
SEG_LINE = 0
SEG_ARC = 1

CH_D0 = 0                   # signed distance of the start point to the CAD (m, + outside the material)
CH_D1 = 1                   # of the end point
CH_EMIN = 2                 # most negative deviation from the end-point interpolation (toward / into the CAD)
CH_SMIN = 3                 # its segment parameter in [0, 1]
CH_EMAX = 4                 # most positive deviation (away from the CAD)
CH_SMAX = 5
CH_FMIN = 6                 # CAD triangle at CH_EMIN
CH_FMAX = 7
CH_LEN = 8                  # segment length (m)
N_CH = 9


def sample_columns() -> Columns:
    """A one-column field used to type the kernels' explicit signatures."""
    return Columns(np.zeros(2), np.ones(2), np.ones(2, np.int64), np.array([0, 1], np.int64),
                   np.array([[0.0, 1.0]]))


COLUMNS_TYPE = typeof(sample_columns())


def contiguous(array, dtype=None) -> np.ndarray:
    """Writable C-contiguous array (a read-only one would type differently and force a second compile)."""
    out = np.ascontiguousarray(array, dtype=dtype)
    if not out.flags.writeable:
        out = out.copy()
    return out


def make_columns(origin_m, cell_m, dims, start, seg) -> Columns:
    """A :class:`Columns` with every array in its kernel dtype and layout."""
    return Columns(contiguous(np.asarray(origin_m, dtype=np.float64).reshape(2)),
                   contiguous(np.asarray(cell_m, dtype=np.float64).reshape(2)),
                   contiguous(np.asarray(dims, dtype=np.int64).reshape(2)),
                   contiguous(np.asarray(start, dtype=np.int64).reshape(-1)),
                   contiguous(np.asarray(seg, dtype=np.float64).reshape(-1, 2)))


def empty_columns(origin_m, cell_m, dims) -> Columns:
    n = int(np.prod(np.asarray(dims, dtype=np.int64)))
    return make_columns(origin_m, cell_m, dims, np.zeros(n + 1, np.int64), np.zeros((0, 2)))


__all__ = ["Columns", "COLUMNS_TYPE", "sample_columns", "make_columns", "empty_columns", "contiguous",
           "CS_VOLUME", "CS_CX", "CS_CY", "CS_CZ", "CS_XMIN", "CS_YMIN", "CS_ZMIN", "CS_XMAX", "CS_YMAX", "CS_ZMAX",
           "CS_THICK", "CS_TX", "CS_TY", "CS_TZ", "CS_COLUMNS", "CS_SEGMENTS", "CS_FIRST", "N_CS",
           "PR_T", "PR_CX", "PR_CY", "PR_CZ", "PR_QX", "PR_QY", "PR_QZ", "PR_FACE", "PR_STATUS", "PR_ITER", "N_PR",
           "PROBE_CONTACT", "PROBE_NO_CONTACT", "PROBE_START_IN_CONTACT", "PROBE_NOT_CONVERGED",
           "SG_KIND", "SG_P0", "SG_P1", "SG_C", "SG_N", "SG_SWEEP", "N_SG", "SEG_LINE", "SEG_ARC",
           "CH_D0", "CH_D1", "CH_EMIN", "CH_SMIN", "CH_EMAX", "CH_SMAX", "CH_FMIN", "CH_FMAX", "CH_LEN", "N_CH"]
