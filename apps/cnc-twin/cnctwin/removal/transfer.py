"""Carrying a workpiece to the next operation (architecture 5.3 end_operation, docs/models/removal.md
"Transfers").

``from_previous`` of an operation (db/_templates/project.yaml) is one of: nothing (the part stays: identity),
``{flip: {axis: X|Y|Z, angle_deg}}`` (turned over about an S axis through the stock box centre), or
``{transfer: sub_spindle | main_spindle | pallet | manual}`` (sub-spindle: gripped from the other end, a
half turn about S y; the others keep the part's orientation). The new stock frame keeps the conventions of
cnctwin.removal.stock: the material's box starts at the S origin (blocks) or keeps its turning axis on S z
with z from 0 (cylinders, tubes).

A rotation by a multiple of 90 degrees is a lattice symmetry: :func:`apply` remaps level 0 exactly
(k_transfer); any other rotation resamples it: the field is meshed (marching cubes with the exact crossings),
the mesh rotated and rasterised on a new lattice (error: the mesh's, about a^2 kappa / 8 on curved faces and up
to one cell at sharp edges, where marching cubes chamfers). Refine levels do not survive a transfer that
moves the material (they are resampled only for the identity); the note says so.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.errors import InputError
from . import abi as A
from .field import make_field, n_levels

AXES = {"X": 0, "Y": 1, "Z": 2}


def rotation(axis: int, angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = np.eye(3)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


def transform_of(from_previous, kind: str) -> np.ndarray | None:
    """3x3 rotation of a ``from_previous`` entry (None: identity)."""
    if not from_previous:
        return None
    flip = from_previous.get("flip") if hasattr(from_previous, "get") else getattr(from_previous, "flip", None)
    transfer = from_previous.get("transfer") if hasattr(from_previous, "get") else getattr(from_previous, "transfer",
                                                                                            None)
    if flip:
        axis = flip.get("axis") if hasattr(flip, "get") else getattr(flip, "axis")
        if "angle_rad" in flip:
            angle = float(flip["angle_rad"])
        else:
            angle = math.radians(float(flip.get("angle_deg")))
        return rotation(AXES[str(axis).upper()], angle)
    if transfer in ("sub_spindle",):
        return rotation(1, math.pi)
    if transfer in ("main_spindle", "pallet", "manual", None):
        return None
    raise InputError(f"from_previous: unknown transfer {transfer!r}")


def signed_permutation(R: np.ndarray, tol: float = 1e-9):
    """(perm, sign) with R[i, perm[i]] = sign[i] when R is a signed permutation, else None."""
    perm = np.zeros(3, np.int64)
    sign = np.zeros(3, np.int64)
    for i in range(3):
        j = int(np.argmax(np.abs(R[i])))
        if abs(abs(R[i, j]) - 1.0) > tol or np.sum(np.abs(R[i]) > tol) != 1:
            return None
        perm[i] = j
        sign[i] = 1 if R[i, j] > 0 else -1
    if len(set(perm.tolist())) != 3:
        return None
    return perm, sign


def new_box(lo, hi, R, kind: str):
    """(box lo, box hi, t) of the stock box rotated by R and placed by the stock conventions: x_new = R x + t.
    Blocks (and turned parts whose axis the rotation tilts): the rotated box starts at the S origin.
    Cylinders and tubes whose axis stays on S z (R maps z to +-z): the axis stays at x = y = 0 and z starts
    at 0."""
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    rot = corners @ R.T
    keeps_axis = abs(abs(R[2, 2]) - 1.0) < 1e-9 and abs(R[0, 2]) < 1e-9 and abs(R[1, 2]) < 1e-9
    if kind in ("cylinder", "tube") and keeps_axis:
        t = np.array([0.0, 0.0, -rot[:, 2].min()])
    else:
        t = -rot.min(axis=0)
    moved = rot + t
    return moved.min(axis=0), moved.max(axis=0), t


def apply(field: A.DexelField, R: np.ndarray | None, kind: str, lattice_m: float, margin_nodes: int = 1,
          refine=()) -> tuple[A.DexelField, np.ndarray, str]:
    """(new field, 4x4 transform new_S <- old_S, note)."""
    T = np.eye(4)
    if R is None:
        return field, T, ""
    lo, hi = field.lv_box_lo_m[0], field.lv_box_hi_m[0]
    b_lo, b_hi, t = new_box(lo, hi, R, kind)
    T[:3, :3] = R
    T[:3, 3] = t
    note = "" if n_levels(field) == 1 else "refine levels were dropped by the transfer (the material moved)"
    sp = signed_permutation(R)
    if sp is not None:
        perm, sign = sp
        from .field import make_field_lattice
        from .k_transfer import remap_level0

        old_dims = field.lv_dims[0]
        a = float(field.lv_cell_m[0])
        o = field.lv_origin_m[0]
        far = o + (old_dims - 1) * a
        corners = np.array([[x, y, z] for x in (o[0], far[0]) for y in (o[1], far[1]) for z in (o[2], far[2])])
        moved = corners @ R.T + t
        origin = moved.min(axis=0)
        dims = old_dims[perm]
        blocks = int(field.info[A.FI_N_BLOCKS])
        while True:
            new = make_field_lattice(origin, dims, int(field.lv_cell_q[0]), b_lo, b_hi, blocks,
                                     int(field.info[A.FI_WORKPIECE]))
            if remap_level0(field, new, perm, sign, A.make_scratch()) == 0:
                return new, T, note
            blocks *= 2
    return _resample(field, T, b_lo, b_hi, lattice_m, margin_nodes,
                     (note + "; " if note else "") + "the rotation is not a lattice symmetry: the field was resampled "
                     "through its marching-cubes mesh (error up to one cell at sharp edges)")


def _resample(field, T, b_lo, b_hi, lattice_m, margin_nodes, note):
    from .mesh import mesh_level
    from .stock import mesh_field

    m = mesh_level(field)
    if len(m.faces) == 0:
        return make_field(b_lo, b_hi, lattice_m, (), margin_nodes), T, note
    V = m.vertices @ T[:3, :3].T + T[:3, 3]
    new, _ = mesh_field(V, m.faces, lattice_m, (), margin_nodes, int(field.info[A.FI_WORKPIECE]), None,
                        box=(b_lo, b_hi))
    return new, T, note


def polar_transfer(pf: A.PolarField, R: np.ndarray, length_m: float) -> A.PolarField | None:
    """The polar field after a half turn about S y (sub-spindle transfer: theta -> pi - theta, z -> L - z),
    exactly; any other rotation drops it (rebuilt from the tri-dexel when the next operation turns)."""
    if R is None:
        return pf
    if not np.allclose(R, np.diag([-1.0, 1.0, -1.0]), atol=1e-12):
        return None
    from . import polar as PO

    nth = int(pf.geom[A.PG_NTHETA])
    nz = int(pf.geom[A.PG_NZ])
    nr = int(pf.geom[A.PG_NR])
    if nth % 2:
        return None
    z0 = float(pf.geom[A.PG_Z0])
    dz = float(pf.geom[A.PG_DZ])
    kp = int(pf.info[A.PI_KP])
    kpa = int(pf.info[A.PI_KPA])
    out = PO.make_polar(nth, length_m - (z0 + nz * dz), dz, nz, float(pf.geom[A.PG_DR]), nr, kp, kpa)
    b_new = (nth // 2 - 1 - np.arange(nth)) % nth
    rc = pf.rcnt.reshape(nth, nz)
    rs = pf.rseg.reshape(nth, nz, kp, 2)
    rv = pf.rsev.reshape(nth, nz, kp, 2)
    out.rcnt.reshape(nth, nz)[b_new] = rc[:, ::-1]
    out.rseg.reshape(nth, nz, kp, 2)[b_new] = rs[:, ::-1]
    out.rsev.reshape(nth, nz, kp, 2)[b_new] = rv[:, ::-1]
    # axial rays: z segments mirrored within the polar z range
    lz = int(round(nz * dz / A.QUANTUM_M))
    ac = pf.acnt.reshape(nth, nr)
    a_s = pf.aseg.reshape(nth, nr, kpa, 2)
    a_v = pf.asev.reshape(nth, nr, kpa, 2)
    oc = out.acnt.reshape(nth, nr)
    os_ = out.aseg.reshape(nth, nr, kpa, 2)
    ov = out.asev.reshape(nth, nr, kpa, 2)
    for b in range(nth):
        bn = b_new[b]
        for j in range(nr):
            n = int(ac[b, j])
            oc[bn, j] = n
            for k in range(n):
                src = n - 1 - k
                os_[bn, j, k, 0] = lz - a_s[b, j, src, 1]
                os_[bn, j, k, 1] = lz - a_s[b, j, src, 0]
                ov[bn, j, k, 0] = a_v[b, j, src, 1]
                ov[bn, j, k, 1] = a_v[b, j, src, 0]
    from .k_polar import refresh_zmax

    refresh_zmax(out, 0)
    return out
