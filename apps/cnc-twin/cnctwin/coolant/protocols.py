"""What the coolant domain needs from other parts, as small Protocols (wired by the integration).

* :class:`ToolShapes` - per tool slot the solid that occludes jets (revolved tool + holder + spindle
  nose envelope of rotating tools; a box for non-rotating tools), its coolant holes and tool class
  (tooling). Default: :class:`TooledShapes` over cnctwin.tooling ToolGeometry objects.
* :class:`WorkpieceBricks` - the in-process workpiece as a brick grid in S (removal's field, brick
  states and the sub-cell occupancy of partial bricks) for the brick-level 3-D DDA. Defaults:
  :class:`StockBricks` (the stock box), :class:`OccupancyBricks` (any occupancy function or voxel
  array, used by tests).
* :class:`SceneMeshes` - static machine and fixture meshes with the link each is attached to.
  Default: :class:`MachineMeshes` (the machine file's ``geometry`` primitives, cnctwin.geom meshes and
  trees).
* :class:`MountPoses` - the frame a nozzle's position and aim are written in (default
  :class:`cnctwin.coolant.kinematics.ChainMounts`).

All geometry is SI; workpiece bricks in the stock frame S (material configuration).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from . import abi


# ============================================================================ tools

@dataclass(frozen=True)
class ToolShape:
    """The occluding solid and the coolant holes of one tool slot (frames: T0 at the tip, +z_T toward
    the spindle)."""

    slot: int
    number: int
    type: str
    rotating: bool
    radius_m: float
    flute_length_m: float
    z_gauge_m: float | None                 # tip to the gauge line (spindle nose); None: no holder data
    holder_length_m: float                  # gauge line to holder nose (coolant passage length)
    tool_class: str                         # carbide | hss | cermet | ... (friction-factor lookup)
    profile_z0_m: float = 0.0               # rotating: radius table R(z) of the revolved envelope
    profile_dz_m: float = 1e-3
    profile_r_m: np.ndarray = field(default_factory=lambda: np.zeros(0))
    profile_part: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    box_centre_T_m: np.ndarray = field(default_factory=lambda: np.zeros(3))   # non-rotating: body box
    box_axes_T: np.ndarray = field(default_factory=lambda: np.eye(3))
    box_half_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    hole_count: int = 0
    hole_diameter_m: float = 0.0
    hole_length_m: float = 0.0
    hole_exit_kind: str = ""
    hole_pos_T_m: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))   # at spindle angle 0
    hole_dir_T: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    @property
    def diameter_m(self) -> float:
        return 2.0 * self.radius_m


@runtime_checkable
class ToolShapes(Protocol):
    def shape(self, slot: int) -> ToolShape | None:
        """The occluding geometry of a tool slot (None: unknown slot)."""


def _piece_radius(env, i: int, z: np.ndarray) -> np.ndarray:
    """Radius of envelope piece i at heights z (NaN outside its range), tooling's convention."""
    out = np.full(z.shape, np.nan)
    inside = (z >= env.z_lo_m[i] - 1e-15) & (z <= env.z_hi_m[i] + 1e-15)
    if not np.any(inside):
        return out
    zi = z[inside]
    if int(env.kind[i]) == 1:
        cos_a = np.clip((env.zc_m[i] - zi) / env.rho_m[i], -1.0, 1.0)
        out[inside] = env.rc_m[i] + env.rho_m[i] * np.sqrt(1.0 - cos_a * cos_a)
    elif env.z1_m[i] == env.z0_m[i]:
        out[inside] = max(env.r0_m[i], env.r1_m[i])
    else:
        t = (zi - env.z0_m[i]) / (env.z1_m[i] - env.z0_m[i])
        out[inside] = env.r0_m[i] + t * (env.r1_m[i] - env.r0_m[i])
    return out


def radius_table(env, dz_max_m: float, max_nodes: int) -> tuple[float, float, np.ndarray, np.ndarray]:
    """(z0, dz, R [n], part [n]) of a tooling Envelope: R is the union radius at the nodes, part the
    tooling PART_* of the widest piece at each interval's midpoint."""
    if env is None or len(env.kind) == 0:
        return 0.0, dz_max_m, np.zeros(0), np.zeros(0, np.int64)
    z_lo = float(np.min(env.z_lo_m))
    z_hi = float(np.max(env.z_hi_m))
    span = max(z_hi - z_lo, dz_max_m)
    n = int(min(max(math.ceil(span / dz_max_m) + 1, 2), max_nodes))
    dz = span / (n - 1)
    z = z_lo + dz * np.arange(n)
    mid = z[:-1] + 0.5 * dz
    radii = np.stack([_piece_radius(env, i, z) for i in range(len(env.kind))])
    mids = np.stack([_piece_radius(env, i, mid) for i in range(len(env.kind))])
    R = np.nan_to_num(np.nanmax(np.where(np.isnan(radii), -np.inf, radii), axis=0), neginf=0.0)
    R = np.maximum(R, 0.0)
    best = np.argmax(np.where(np.isnan(mids), -np.inf, mids), axis=0)
    part = np.zeros(n, np.int64)
    part[:-1] = np.asarray(env.part, dtype=np.int64)[best]
    part[-1] = part[-2] if n > 1 else 0
    return z_lo, dz, R, part


def shape_from_geometry(geo, dz_max_m: float = 2.5e-4, max_nodes: int = 2048) -> ToolShape:
    """A ToolShape from a cnctwin.tooling ToolGeometry."""
    props = geo.props
    layout = dict(geo.layout or {})
    info = dict(geo.info or {})
    radius = 0.5 * float(props.diameter_m)
    tool_class = str(getattr(geo.spec, "tool_material_class", "") or "carbide") if geo.spec is not None else "carbide"
    z_gauge = layout.get("z_gauge")
    holder_len = float(geo.holder.gauge_length_m) if getattr(geo, "holder", None) is not None else 0.0
    holes = props.coolant
    kwargs = {}
    if holes is not None:
        helix = info.get("helix")
        helix_rad = float(np.mean(np.abs(helix))) if helix else 0.0
        overall = float(info.get("overall", layout.get("overall", 0.0)) or layout.get("overall", 0.0) or 0.0)
        z_exit = float(np.mean(holes.exit_pos_T_m[:, 2])) if len(holes.exit_pos_T_m) else 0.0
        straight = max(overall - z_exit, 0.0) if overall > 0.0 else max(float(layout.get("z_flute_end", 0.0)), 0.0)
        length = straight / math.cos(helix_rad) if holes.exit in ("tip", "flute") and helix_rad < 1.2 else straight
        kwargs.update(hole_count=int(holes.count), hole_diameter_m=float(holes.diameter_m), hole_length_m=length,
                      hole_exit_kind=str(holes.exit), hole_pos_T_m=np.array(holes.exit_pos_T_m, dtype=float),
                      hole_dir_T=np.array(holes.exit_dir_T, dtype=float))
    if geo.rotating:
        z0, dz, R, part = radius_table(geo.envelope, dz_max_m, max_nodes)
        return ToolShape(slot=geo.slot, number=geo.number, type=geo.type, rotating=True, radius_m=radius,
                         flute_length_m=float(props.flute_length_m), z_gauge_m=z_gauge, holder_length_m=holder_len,
                         tool_class=tool_class.lower(), profile_z0_m=z0, profile_dz_m=dz, profile_r_m=R,
                         profile_part=part, **kwargs)
    hb = info.get("shank_hb") or [0.025, 0.025]
    h, b = float(hb[0]), float(hb[1])
    overhang = float(info.get("overhang") or 2.0 * max(h, b))
    shank_dir = np.asarray(info.get("shank_dir_T", (-1.0, 0.0, 0.0)), dtype=float)
    shank_dir = shank_dir / np.linalg.norm(shank_dir)
    ic = float(info.get("ic", 0.0) or 0.0)
    length = overhang + 2.0 * max(h, b)
    centre = shank_dir * (0.5 * length - 0.5 * ic) if ic > 0.0 else shank_dir * 0.5 * length
    other = np.array([0.0, 0.0, 1.0]) if abs(shank_dir[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e2 = np.cross(other, shank_dir)
    e2 = e2 / np.linalg.norm(e2)
    e3 = np.cross(shank_dir, e2)
    axes = np.stack([shank_dir, e2, e3], axis=1)
    return ToolShape(slot=geo.slot, number=geo.number, type=geo.type, rotating=False, radius_m=radius,
                     flute_length_m=float(props.flute_length_m), z_gauge_m=None, holder_length_m=holder_len,
                     tool_class=tool_class.lower(), box_centre_T_m=centre, box_axes_T=axes,
                     box_half_m=np.array([0.5 * length, 0.5 * b, 0.5 * h]), **kwargs)


class TooledShapes:
    """ToolShapes from the tooling library: a ToolingState (``geometry(slot)``) or a list of ToolGeometry."""

    def __init__(self, source, dz_max_m: float = 2.5e-4, max_nodes: int = 2048):
        self.source = source
        self.dz_max_m = dz_max_m
        self.max_nodes = max_nodes
        self._cache: dict[int, ToolShape] = {}

    def _geometry(self, slot: int):
        src = self.source
        if src is None:
            return None
        if hasattr(src, "geometry"):
            try:
                return src.geometry(slot) if 0 <= slot < len(src) else None
            except (IndexError, TypeError):
                return None
        return src[slot] if 0 <= slot < len(src) else None

    def shape(self, slot: int) -> ToolShape | None:
        if slot in self._cache:
            return self._cache[slot]
        geo = self._geometry(slot)
        if geo is None:
            return None
        shape = shape_from_geometry(geo, self.dz_max_m, self.max_nodes)
        self._cache[slot] = shape
        return shape


# ============================================================================ workpiece

@dataclass(frozen=True)
class BrickSet:
    """One workpiece as bricks in S (see abi.BrickGrid for the meaning of the arrays)."""

    lo_S_m: np.ndarray              # [3]
    brick_m: float
    dims: np.ndarray                # [3] int64
    state: np.ndarray               # [nx ny nz] int8 flattened x fastest
    sub: np.ndarray                 # [nx ny nz] int64 (-1 or first sub-cell in occ)
    n_sub: int
    occ: np.ndarray                 # int8
    workpiece: int = 0


@runtime_checkable
class WorkpieceBricks(Protocol):
    def bricks(self, workpiece: int) -> BrickSet | None:
        """The material of a workpiece as bricks in S at the end of the step (None: no material)."""

    def bounds_S(self, workpiece: int) -> tuple[np.ndarray, np.ndarray] | None:
        """(lo, hi) of the material's bounding box in S."""

    def version(self, workpiece: int) -> int:
        """Increments whenever the material changed (removal's field_version)."""


def bricks_from_occupancy(lo_S, voxel_m: float, occupied: np.ndarray, n_sub: int, workpiece: int = 0) -> BrickSet:
    """Bricks of n_sub^3 voxels from a voxel occupancy array [vx, vy, vz] (bool) whose voxel (0,0,0)
    starts at lo_S: bricks wholly occupied are full, wholly empty are empty, the rest partial."""
    occupied = np.asarray(occupied, dtype=bool)
    vx, vy, vz = occupied.shape
    dims = np.array([math.ceil(vx / n_sub), math.ceil(vy / n_sub), math.ceil(vz / n_sub)], np.int64)
    padded = np.zeros((dims[0] * n_sub, dims[1] * n_sub, dims[2] * n_sub), dtype=bool)
    padded[:vx, :vy, :vz] = occupied
    n_bricks = int(dims.prod())
    state = np.zeros(n_bricks, np.int8)
    sub = np.full(n_bricks, -1, np.int64)
    occ_chunks = []
    count = 0
    for k in range(dims[2]):
        for j in range(dims[1]):
            for i in range(dims[0]):
                block = padded[i * n_sub:(i + 1) * n_sub, j * n_sub:(j + 1) * n_sub, k * n_sub:(k + 1) * n_sub]
                idx = i + dims[0] * (j + dims[1] * k)
                if block.all():
                    state[idx] = abi.BRICK_FULL
                elif block.any():
                    state[idx] = abi.BRICK_PARTIAL
                    sub[idx] = count * n_sub ** 3
                    occ_chunks.append(np.transpose(block, (2, 1, 0)).reshape(-1).astype(np.int8))
                    count += 1
    occ = np.concatenate(occ_chunks) if occ_chunks else np.zeros(0, np.int8)
    return BrickSet(np.asarray(lo_S, dtype=float), float(voxel_m * n_sub), dims, state, sub, int(n_sub), occ,
                    workpiece)


class OccupancyBricks:
    """WorkpieceBricks from an inside test ``inside(points_S [n, 3]) -> bool [n]`` sampled at voxel
    centres on a grid over [lo, hi] (deterministic), or from a given voxel array."""

    def __init__(self, lo_S, hi_S, voxel_m: float, n_sub: int = 4, inside: Callable | None = None,
                 occupied: np.ndarray | None = None, workpiece: int = 0):
        self.lo = np.asarray(lo_S, dtype=float)
        self.hi = np.asarray(hi_S, dtype=float)
        self.voxel = float(voxel_m)
        self.n_sub = int(n_sub)
        self.workpiece = workpiece
        if occupied is None:
            shape = np.maximum(np.ceil((self.hi - self.lo) / self.voxel - 1e-9).astype(int), 1)
            axes = [self.lo[a] + self.voxel * (np.arange(shape[a]) + 0.5) for a in range(3)]
            X, Y, Z = np.meshgrid(*axes, indexing="ij")
            pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
            fn = inside if inside is not None else (lambda p: np.all((p >= self.lo) & (p <= self.hi), axis=1))
            occupied = np.asarray(fn(pts), dtype=bool).reshape(shape)
        self.occupied = np.asarray(occupied, dtype=bool)
        self._bricks = bricks_from_occupancy(self.lo, self.voxel, self.occupied, self.n_sub, workpiece)

    def bricks(self, workpiece: int) -> BrickSet | None:
        return self._bricks if workpiece == self.workpiece else None

    def bounds_S(self, workpiece: int):
        return (self.lo.copy(), self.hi.copy()) if workpiece == self.workpiece else None

    def version(self, workpiece: int) -> int:
        return 0


class StockBricks(OccupancyBricks):
    """The unmachined stock box [lo, hi] in S (exact bricks: the voxel edge divides the box)."""

    def __init__(self, lo_S, hi_S, voxel_m: float = 2e-3, n_sub: int = 4, workpiece: int = 0):
        lo = np.asarray(lo_S, dtype=float)
        hi = np.asarray(hi_S, dtype=float)
        size = hi - lo
        n = np.maximum(np.ceil(size / voxel_m - 1e-9).astype(int), 1)
        voxel = float(np.max(size / n))
        super().__init__(lo, lo + voxel * n, voxel, n_sub, occupied=np.ones(tuple(n), dtype=bool), workpiece=workpiece)
        self.hi = hi


def merge_bricks(sets: list[tuple[BrickSet, np.ndarray, np.ndarray]]) -> abi.BrickGrid:
    """One BrickGrid of several workpieces: (BrickSet, R_M_S [3, 3], o_M_S [3]) each."""
    if not sets:
        return abi.empty_bricks()
    lo, b, dims, offset, R, o, ns, wp = [], [], [], [], [], [], [], []
    states, subs, occs = [], [], []
    base = 0
    occ_base = 0
    for bs, R_M_S, o_M_S in sets:
        lo.append(bs.lo_S_m)
        b.append(bs.brick_m)
        dims.append(bs.dims)
        offset.append(base)
        R.append(R_M_S)
        o.append(o_M_S)
        ns.append(bs.n_sub)
        wp.append(bs.workpiece)
        states.append(np.asarray(bs.state, np.int8))
        sub = np.asarray(bs.sub, np.int64).copy()
        sub[sub >= 0] += occ_base
        subs.append(sub)
        occs.append(np.asarray(bs.occ, np.int8))
        base += len(bs.state)
        occ_base += len(bs.occ)
    c = abi.contiguous
    return abi.BrickGrid(c(np.array(lo, float)), c(np.array(b, float)), c(np.array(dims, np.int64)),
                         c(np.array(offset, np.int64)), c(np.array(R, float)), c(np.array(o, float)),
                         c(np.array(ns, np.int64)), c(np.concatenate(states)), c(np.concatenate(subs)),
                         c(np.concatenate(occs) if occs else np.zeros(0, np.int8)), c(np.array(wp, np.int64)))


# ============================================================================ scene meshes

@dataclass(frozen=True)
class SceneMesh:
    name: str                       # occluder id, e.g. "machine:table", "fixture:vise"
    tree: object                    # cnctwin.geom.abi.MeshTree in the link frame
    link: str                       # kinematic link the mesh is attached to ('' = machine base, M)


@runtime_checkable
class SceneMeshes(Protocol):
    def meshes(self) -> list[SceneMesh]:
        """Static meshes (machine components, fixtures), each in the frame of its link."""


class MachineMeshes:
    """The machine file's ``geometry`` components with ``collide: true`` (one tree per component)."""

    def __init__(self, machine, exclude_links: tuple = ()):
        from ..geom import bvh
        from ..geom.mesh import merge
        from ..geom.primitives import mesh_from_primitive
        self._meshes: list[SceneMesh] = []
        for comp in (getattr(machine, "geometry", None) or []) if machine is not None else []:
            collide = getattr(comp, "collide", True)
            link = str(getattr(comp, "attached_to", "") or "")
            if collide is False or link in exclude_links:
                continue
            parts = [mesh_from_primitive(p) for p in (getattr(comp, "primitives", None) or [])]
            if not parts:
                continue
            mesh = merge(parts, offset_groups=True, name=str(getattr(comp, "name", "") or link)) if len(parts) > 1 \
                else parts[0]
            self._meshes.append(SceneMesh(f"machine:{getattr(comp, 'name', link)}", bvh.static_tree(mesh), link))

    def meshes(self) -> list[SceneMesh]:
        return list(self._meshes)


class NoMeshes:
    def meshes(self) -> list[SceneMesh]:
        return []


# ============================================================================ mounts

@runtime_checkable
class MountPoses(Protocol):
    def pose_M(self, link: str, path, gauge_m: float | None, poses: dict) -> np.ndarray | None:
        """4x4 pose in M of the frame nozzle positions on ``link`` are written in (None: unknown link).
        ``path`` is the PathFrame of the tool chain, ``gauge_m`` the tool's tip-to-gauge-line length,
        ``poses`` the link poses of the machine chains at the step's axis values."""
