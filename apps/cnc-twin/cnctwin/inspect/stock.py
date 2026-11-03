"""The next operation's stock from this operation's result (docs/models/inspect.md "Next stock").

The part goes to the next setup as it really is: the inspected geometry ``X + d_final`` (unclamped, spring-back and
machining distortion, 20 degC), turned over or transferred by ``from_previous`` and seated with a mislocation:

    p_new = M (R p_old + t)

* ``R, t`` follow removal's transfer conventions exactly (cnctwin.removal.transfer, mirrored here because domains
  do not import each other): ``{flip: {axis, angle_deg}}`` turns about an S axis; ``{transfer: sub_spindle}`` is a
  half turn about S y; ``main_spindle``, ``pallet``, ``manual`` keep the orientation; the turned stock box starts at
  the S origin (blocks), or keeps its turning axis on S z with z from 0 (cylinders and tubes whose axis stays on z).
  Pass the stock box of the field (removal's level-0 box) so both agree;
* ``M`` is the **mislocation** input: a small rigid pose in the new stock frame (a re-chucking error drawn by an
  uncertainty study, a measured seating error, a sub-spindle grip error). Do not pass holding's
  ``location_pose_delta`` here when motion already composes it into the workpiece chain: it would count twice.

The field of the next operation comes from :class:`cnctwin.inspect.protocols.StockFieldBuilder`: removal's exact
remap when the transfer is a lattice symmetry (a signed permutation), there is no mislocation and no distortion
(the material configuration then is the part); otherwise the transformed closed mesh is rasterised (removal's
mesh_field: the tri-dexel endpoints are exact to 10 nm along the rays, so a distortion well below one lattice cell
is still carried).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.errors import InputError
from ..geom.mesh import Mesh
from .result import INSPECTED, ResultPart

AXES = {"X": 0, "Y": 1, "Z": 2}
DISTORTION_FLOOR_M = 1e-9          # below the dexel quantum (10 nm): the material configuration is the part


def _get(obj, key):
    if obj is None:
        return None
    if hasattr(obj, "get"):
        try:
            return obj.get(key)
        except TypeError:
            pass
    return getattr(obj, key, None)


def rotation(axis: int, angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = np.eye(3)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


def transfer_rotation(from_previous) -> np.ndarray | None:
    """3x3 rotation of a ``from_previous`` entry (None: the part keeps its orientation)."""
    if not from_previous:
        return None
    flip = _get(from_previous, "flip")
    transfer = _get(from_previous, "transfer")
    if flip:
        axis = str(_get(flip, "axis")).upper()
        if axis not in AXES:
            raise InputError(f"from_previous.flip.axis {axis!r} is not X, Y or Z")
        angle = _get(flip, "angle_rad")
        angle = float(getattr(angle, "value", angle)) if angle is not None else math.radians(float(
            getattr(_get(flip, "angle_deg"), "value", _get(flip, "angle_deg"))))
        return rotation(AXES[axis], angle)
    if transfer == "sub_spindle":
        return rotation(1, math.pi)
    if transfer in ("main_spindle", "pallet", "manual", None):
        return None
    raise InputError(f"from_previous: unknown transfer {transfer!r}")


def signed_permutation(R: np.ndarray, tol: float = 1e-9):
    perm, sign = np.zeros(3, np.int64), np.zeros(3, np.int64)
    for i in range(3):
        j = int(np.argmax(np.abs(R[i])))
        if abs(abs(R[i, j]) - 1.0) > tol or np.sum(np.abs(R[i]) > tol) != 1:
            return None
        perm[i], sign[i] = j, 1 if R[i, j] > 0 else -1
    if len(set(perm.tolist())) != 3:
        return None
    return perm, sign


def new_box(lo, hi, R, kind: str):
    """(box lo, box hi, t) of the stock box turned by R and placed by the stock conventions (x_new = R x + t)."""
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    rot = corners @ R.T
    keeps_axis = abs(abs(R[2, 2]) - 1.0) < 1e-9 and abs(R[0, 2]) < 1e-9 and abs(R[1, 2]) < 1e-9
    if kind in ("cylinder", "tube") and keeps_axis:
        t = np.array([0.0, 0.0, -rot[:, 2].min()])
    else:
        t = -rot.min(axis=0)
    moved = rot + t
    return moved.min(axis=0), moved.max(axis=0), t


@dataclass
class NextStock:
    op: str
    workpiece: int
    transform_new_old: np.ndarray           # p_new = T p_old, mislocation included
    nominal_transform: np.ndarray           # without the mislocation
    mislocation: np.ndarray
    mesh: Mesh                              # the part in the new stock frame
    box_lo_m: np.ndarray
    box_hi_m: np.ndarray
    method: str                             # exact remap | rasterised | mesh
    stock_field: object = None              # what the StockFieldBuilder made (the next operation's field)
    notes: list = field(default_factory=list)

    def to_json(self) -> dict:
        return {"op": self.op, "workpiece": self.workpiece, "method": self.method,
                "transform_new_old": [[float(v) for v in row] for row in self.transform_new_old],
                "nominal_transform": [[float(v) for v in row] for row in self.nominal_transform],
                "mislocation": [[float(v) for v in row] for row in self.mislocation],
                "volume_m3": float(self.mesh.volume()), "box_lo_S_m": [float(v) for v in self.box_lo_m],
                "box_hi_S_m": [float(v) for v in self.box_hi_m], "notes": list(self.notes)}


def next_stock(result: ResultPart, from_previous, kind: str = "block", lattice_m: float = 2.5e-4, mislocation=None,
               builder=None, stock_box=None) -> NextStock:
    """The next operation's stock (module documentation)."""
    R = transfer_rotation(from_previous)
    Rm = np.eye(3) if R is None else R
    mesh = result.configuration(INSPECTED)
    if stock_box is None:
        lo, hi = result.material.bounds()
        notes = ["stock box taken from the result's material bounds (pass removal's field box to match it exactly)"]
    else:
        lo, hi = (np.asarray(v, dtype=np.float64) for v in stock_box)
        notes = []
    b_lo, b_hi, t = new_box(lo, hi, Rm, kind)
    T0 = np.eye(4)
    T0[:3, :3] = Rm
    T0[:3, 3] = t
    M = np.eye(4) if mislocation is None else np.asarray(mislocation, dtype=np.float64)
    if not np.allclose(M[:3, :3] @ M[:3, :3].T, np.eye(3), atol=1e-9) or abs(np.linalg.det(M[:3, :3]) - 1) > 1e-9:
        raise InputError("the mislocation must be a rigid pose (rotation and translation)")
    T = M @ T0
    moved = mesh.transformed(T)
    moved.name = f"stock_after_{result.op}"
    distorted = result.grid("final") is not None and result.grid("final").max_norm_m() > DISTORTION_FLOOR_M
    misplaced = not np.allclose(M, np.eye(4), rtol=0.0, atol=1e-15)
    method, fld = "mesh", None
    if builder is not None:
        if signed_permutation(Rm) is not None and not distorted and not misplaced:
            fld = builder.remap(result.workpiece, Rm, kind)
            method = "exact remap" if fld is not None else method
        if fld is None:
            m_lo, m_hi = moved.bounds()
            box_lo = np.minimum(b_lo, m_lo) - lattice_m
            box_hi = np.maximum(b_hi, m_hi) + lattice_m
            fld = builder.from_mesh(result.workpiece, moved.vertices, moved.faces, lattice_m, box_lo, box_hi)
            method = "rasterised"
    if distorted:
        notes.append(f"carries the spring-back / distortion of {result.op} (largest {result.grid('final').max_norm_m():.3g} m)")
    if misplaced:
        notes.append("includes the mislocation input")
    return NextStock(result.op, result.workpiece, T, T0, M, moved, b_lo, b_hi, method, fld, notes)


__all__ = ["NextStock", "next_stock", "transfer_rotation", "new_box", "signed_permutation", "rotation"]
