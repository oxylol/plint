"""What holding asks of the workpiece geometry and of the other domains, as small Protocols the integration
wires to removal's queries (docs/models/holding.md "Geometry queries"), with defaults built from the stock.

* :class:`PartGeometry` - shape of the part for placing contacts (block or cylinder in S), its mass
  properties per unit density, and the material fraction of a contact patch (a through-cut under a vacuum
  seal or a gripped face machined away);
* :class:`TabSections` - the remaining material cross-section of a tab (area and centroid in S);
* :class:`BurrSource` - burr heights on locating patches from the surface domain's records.

Defaults: :class:`StockGeometry` (the project's stock: block, cylinder or tube) and :class:`MeshStockGeometry`
(a mesh stock: mass properties of the closed mesh, contacts on its bounding box, a cylinder when it is a turned
stock about S z); both reduce the volume by the removed volume booked from removal's contacts and keep
centroid and shape,
:class:`NominalTabs` (tabs keep their nominal section: tab release by cutting is then not seen - the domain
records that it runs without the query), :class:`NoBurrs` (no burr record: chips' locator debris, which
already includes removal's burr events, is the only lift under locators).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from ..core.errors import InputError
from . import params as P


@dataclass(frozen=True)
class PartShape:
    """Shape of a workpiece in S for placing contacts: a box [lo, hi], or a cylinder / tube about S z
    (radius, inner radius, axial extent [z0, z1])."""

    kind: str                       # "box" | "cylinder"
    lo_S_m: tuple = (0.0, 0.0, 0.0)
    hi_S_m: tuple = (0.0, 0.0, 0.0)
    radius_m: float = 0.0
    r_in_m: float = 0.0
    z0_m: float = 0.0
    z1_m: float = 0.0

    def corners_S(self) -> np.ndarray:
        lo, hi = np.asarray(self.lo_S_m, float), np.asarray(self.hi_S_m, float)
        return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])

    @property
    def size_m(self) -> np.ndarray:
        return np.asarray(self.hi_S_m, float) - np.asarray(self.lo_S_m, float)


@runtime_checkable
class PartGeometry(Protocol):
    def shape_S(self, workpiece: int) -> PartShape:
        """Shape of the workpiece for placing contacts (S frame, material configuration)."""

    def mass_properties_S(self, workpiece: int) -> tuple[float, np.ndarray, np.ndarray]:
        """(volume m3, centroid S [3], second moment of volume about the centroid [3, 3] m5) of the
        material now (multiply by the density for mass and inertia)."""

    def face_fraction(self, workpiece: int, centre_S, normal_S, axis1_S, axis2_S, half1_m: float,
                      half2_m: float) -> float:
        """Share of the rectangle (centre, axes, half sizes) on the part surface with outward normal_S that is
        still material (1: the whole patch; 0: machined away)."""


@runtime_checkable
class TabSections(Protocol):
    def section_S(self, workpiece: int, centre_S, normal_S, axis1_S, axis2_S, half1_m: float,
                  half2_m: float) -> tuple[float, np.ndarray]:
        """(area m2, centroid S [3]) of the material left in the tab's section rectangle."""


@runtime_checkable
class BurrSource(Protocol):
    def burr_heights_m(self, op_index: int, patches: list) -> dict:
        """patch id -> burr height (m) for the locating patches (id, centre_S, outward normal_S, half [2])
        of operation ``op_index``."""


# --------------------------------------------------------------------------- defaults
def stock_shape(stock) -> PartShape:
    """PartShape of the project's stock (block [0, size], cylinder/tube about S z; architecture: removal's
    stock conventions)."""
    kind = P.text(stock, "kind", "")
    size = P.numlist(stock, "size_m")
    if kind == "block":
        if not size or len(size) != 3:
            raise InputError("holding: a block stock needs size_mm [x, y, z]")
        return PartShape("box", (0.0, 0.0, 0.0), tuple(float(v) for v in size))
    if kind in ("cylinder", "tube"):
        if not size or len(size) < 2:
            raise InputError(f"holding: a {kind} stock needs its sizes")
        r = 0.5 * float(size[0])
        r_in = 0.5 * float(size[1]) if kind == "tube" else 0.0
        length = float(size[-1])
        return PartShape("cylinder", (-r, -r, 0.0), (r, r, length), r, r_in, 0.0, length)
    raise InputError(f"holding: stock kind {kind!r} needs a PartGeometry (removal's query) to place contacts")


def box_mass_properties(lo, hi) -> tuple[float, np.ndarray, np.ndarray]:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    a = hi - lo
    v = float(np.prod(a))
    J = v / 12.0 * np.diag([a[1] ** 2 + a[2] ** 2, a[0] ** 2 + a[2] ** 2, a[0] ** 2 + a[1] ** 2])
    return v, 0.5 * (lo + hi), J


def tube_mass_properties(r: float, r_in: float, z0: float, z1: float) -> tuple[float, np.ndarray, np.ndarray]:
    length = z1 - z0
    v = math.pi * (r * r - r_in * r_in) * length
    j_axial = 0.5 * v * (r * r + r_in * r_in)
    j_trans = v * (3.0 * (r * r + r_in * r_in) + length * length) / 12.0
    return v, np.array([0.0, 0.0, 0.5 * (z0 + z1)]), np.diag([j_trans, j_trans, j_axial])


@dataclass
class StockGeometry:
    """PartGeometry from the project stock (see the module doc); ``removed_m3`` per workpiece is booked by the
    domain from removal's contacts."""

    shapes: dict                                   # workpiece -> PartShape
    removed_m3: dict = field(default_factory=dict)

    @classmethod
    def from_stock(cls, stock, workpieces=(0,)) -> "StockGeometry":
        shape = stock_shape(stock)
        return cls({int(w): shape for w in workpieces})

    def shape_S(self, workpiece: int) -> PartShape:
        if workpiece in self.shapes:
            return self.shapes[workpiece]
        return self.shapes[min(self.shapes)]

    def mass_properties_S(self, workpiece: int):
        shape = self.shape_S(workpiece)
        if shape.kind == "box":
            v, c, J = box_mass_properties(shape.lo_S_m, shape.hi_S_m)
        else:
            v, c, J = tube_mass_properties(shape.radius_m, shape.r_in_m, shape.z0_m, shape.z1_m)
        removed = float(self.removed_m3.get(workpiece, 0.0))
        left = max(v - removed, 0.0)
        scale = left / v if v > 0.0 else 0.0
        return left, c, J * scale

    def face_fraction(self, workpiece, centre_S, normal_S, axis1_S, axis2_S, half1_m, half2_m) -> float:
        return 1.0

    def note_removed(self, workpiece: int, volume_m3: float) -> None:
        self.removed_m3[workpiece] = float(self.removed_m3.get(workpiece, 0.0)) + float(volume_m3)


def mesh_mass_properties(vertices: np.ndarray, faces: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """(volume, centroid, second moment of volume about the centroid [3, 3]) of a closed, outward-oriented
    triangle mesh: signed tetrahedra (origin, a, b, c), each with covariance det / 120 A [[2,1,1],[1,2,1],
    [1,1,2]] A^T (A = [a b c]); inertia per density J = tr(C) I - C about the centroid."""
    V = np.asarray(vertices, dtype=float)
    F = np.asarray(faces, dtype=np.int64)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    det = np.einsum("ij,ij->i", a, np.cross(b, c))
    volume = float(det.sum() / 6.0)
    if volume <= 0.0:
        raise InputError("holding: the stock mesh encloses no volume (open or inward-oriented)")
    centroid = (det[:, None] * (a + b + c)).sum(axis=0) / (24.0 * volume)
    A = np.stack([a, b, c], axis=2)                       # [n, 3 (xyz), 3 (vertex)]
    Mc = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]])
    C = np.einsum("n,nik,kl,njl->ij", det, A, Mc, A) / 120.0
    C = C - volume * np.outer(centroid, centroid)
    J = np.trace(C) * np.eye(3) - C
    return volume, centroid, J


def mesh_shape(lo, hi, rel_tol: float = 0.01) -> PartShape:
    """Contact shape of a mesh stock: a cylinder about S z when its bounding box is centred on the axis with
    equal x and y extents (a turned stock, removal's lathe convention), else its bounding box."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    size = hi - lo
    r = 0.5 * max(size[0], size[1])
    centred = abs(lo[0] + hi[0]) <= rel_tol * size[0] and abs(lo[1] + hi[1]) <= rel_tol * size[1]
    if centred and abs(size[0] - size[1]) <= rel_tol * max(size[0], size[1]):
        return PartShape("cylinder", (-r, -r, float(lo[2])), (r, r, float(hi[2])), r, 0.0, float(lo[2]), float(hi[2]))
    return PartShape("box", tuple(float(v) for v in lo), tuple(float(v) for v in hi))


@dataclass
class MeshStockGeometry(StockGeometry):
    """PartGeometry of a mesh stock (``stock.mesh``, read through cnctwin.geom): mass properties of the closed
    mesh, contacts on its bounding box (a cylinder for turned stock)."""

    props: dict = field(default_factory=dict)            # workpiece -> (volume, centroid, J)

    @classmethod
    def from_mesh(cls, path, workpieces=(0,)) -> "MeshStockGeometry":
        from ..geom.io import read_mesh
        mesh = read_mesh(path)
        lo, hi = mesh.bounds()
        props = mesh_mass_properties(mesh.vertices, mesh.faces)
        shape = mesh_shape(lo, hi)
        return cls({int(w): shape for w in workpieces}, props={int(w): props for w in workpieces})

    def mass_properties_S(self, workpiece: int):
        key = workpiece if workpiece in self.props else min(self.props)
        v, c, J = self.props[key]
        removed = float(self.removed_m3.get(workpiece, 0.0))
        left = max(v - removed, 0.0)
        scale = left / v if v > 0.0 else 0.0
        return left, np.asarray(c, float), np.asarray(J, float) * scale


def stock_geometry(stock, workpieces=(0,)) -> StockGeometry:
    """The default PartGeometry of the project's stock (block, cylinder, tube, or a mesh file)."""
    if P.text(stock, "kind", "") == "mesh":
        path = P.plain(P.field_of(stock, "mesh"))
        if not path:
            raise InputError("holding: a mesh stock needs its mesh file")
        return MeshStockGeometry.from_mesh(str(path), workpieces)
    return StockGeometry.from_stock(stock, workpieces)


class NominalTabs:
    """TabSections without removal's query: the nominal section (4 half1 half2) at the rectangle centre."""

    wired = False

    def section_S(self, workpiece, centre_S, normal_S, axis1_S, axis2_S, half1_m, half2_m):
        return 4.0 * float(half1_m) * float(half2_m), np.asarray(centre_S, dtype=float)


class NoBurrs:
    """BurrSource without the surface domain's records."""

    wired = False

    def burr_heights_m(self, op_index, patches) -> dict:
        return {}


__all__ = ["PartShape", "PartGeometry", "TabSections", "BurrSource", "StockGeometry", "MeshStockGeometry",
           "NominalTabs", "NoBurrs", "stock_shape", "stock_geometry", "mesh_shape", "mesh_mass_properties",
           "box_mass_properties", "tube_mass_properties"]
