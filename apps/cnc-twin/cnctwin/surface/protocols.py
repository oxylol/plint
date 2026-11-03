"""What the surface domain needs from other parts, as small Protocols (wired by the integration).

* :class:`ToolInfo` - per tool slot the nominal EdgeModel (bus per-element arrays have its shape) and the
  generating :class:`ToolOutline`: the cutting outline as a dense 2-D polyline in the coordinates of the
  EdgeModel (rotating tools: meridian (r, z_T) of the swept cutting edges; turning tools: (x_T, z_T) of the
  insert's cutting chain), the finishing radius, teeth, hand and the thread profile of threading tools.
  :class:`GeometryToolInfo` wraps cnctwin.tooling ``ToolGeometry`` objects (the integration passes
  ``state.tooling.tools``); :class:`StaticToolInfo` takes the parts directly (tests, inline data) and the
  ``*_outline`` helpers build analytic outlines.
* :class:`WorkpieceQuery` - the final (or current) workpiece: ``point_inside`` and ``ray_cast`` with the
  signatures of cnctwin.removal.views.FieldGeometry, so the integration passes
  ``RemovalDomain.geometry(state)`` directly; :class:`StockQuery` is the analytic stand-in for block,
  cylinder and tube stock (conventions of cnctwin.removal.stock: block [0, size] in S, cylinder and tube
  about S z from z = 0).
* :class:`BrickGrid` - the brick numbering of the per-brick attribute arrays: :class:`LatticeBricks`
  reproduces removal's level-0 numbering from the lattice origin, spacing and node counts (the integration
  passes ``field.lv_origin_m[0], field.lv_cell_m[0], field.lv_dims[0]``).
* :class:`BrickMeshes` - the final surface vertices per brick for the attribute arrays: :class:`RunDirBricks`
  reads the run directory's last keyframe records (cnctwin.core.keyframes.KeyframeReader).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..core.abi import EdgeModel
from ..core.errors import InputError

BRICK_CELLS = 32                    # lattice cells per brick edge (architecture 16, removal BRICK)
OUTLINE_ARC_STEP_RAD = math.radians(0.25)
OUTLINE_LINE_STEP_M = 2.0e-6


@dataclass(frozen=True)
class ToolOutline:
    """The generating outline of one tool (see the module doc for the coordinates)."""

    rotating: bool
    pts: np.ndarray                 # [m, 2] dense polyline (radial, axial), along the cutting chain
    n_teeth: int
    finish_radius_m: float          # corner / nose / ball radius (0: sharp)
    tool_radius_m: float
    kind: str = ""
    hand: float = -1.0
    thread: object | None = None    # cnctwin.tooling.abi.ThreadProfile of threading tools
    kappa_major_rad: float = math.pi / 2.0   # entering angle of a turning insert's major edge (burr height)


@runtime_checkable
class ToolInfo(Protocol):
    def edge_model(self, slot: int) -> EdgeModel:
        """Nominal edge elements of a tool slot."""

    def outline(self, slot: int) -> ToolOutline:
        """Generating outline of a tool slot."""


class StaticToolInfo:
    """ToolInfo from explicit edge models and outlines (tests, inline tools)."""

    def __init__(self, edges: dict, outlines: dict):
        self._edges = dict(edges)
        self._outlines = dict(outlines)

    def edge_model(self, slot: int) -> EdgeModel:
        if slot not in self._edges:
            raise InputError(f"surface: no edge model for tool slot {slot}")
        return self._edges[slot]

    def outline(self, slot: int) -> ToolOutline:
        if slot not in self._outlines:
            raise InputError(f"surface: no tool outline for tool slot {slot}")
        return self._outlines[slot]


class GeometryToolInfo:
    """ToolInfo over cnctwin.tooling ``ToolGeometry`` objects indexed by slot."""

    def __init__(self, geometries):
        self._geos = {int(g.slot): g for g in geometries}
        self._cache: dict[int, ToolOutline] = {}

    def _geo(self, slot: int):
        if slot not in self._geos:
            raise InputError(f"surface: tooling has no geometry for tool slot {slot}")
        return self._geos[slot]

    def edge_model(self, slot: int) -> EdgeModel:
        return self._geo(slot).edge

    def outline(self, slot: int) -> ToolOutline:
        if slot in self._cache:
            return self._cache[slot]
        geo = self._geo(slot)
        edge = geo.edge
        valid = np.asarray(edge.elem_valid) != 0
        n_teeth = int(np.asarray(edge.tooth_phase_rad).shape[0])
        radius = 0.5 * float(geo.props.diameter_m) if getattr(geo.props, "diameter_m", None) else \
            float(np.max(np.abs(np.asarray(edge.elem_r_m)[valid]))) if valid.any() else 0.0
        if geo.rotating:
            pts = envelope_outline(geo.envelope)
        else:
            if geo.insert is None:
                raise InputError(f"surface: tool slot {slot} ({geo.type}) has no insert outline")
            pts = insert_outline(geo.insert)
        finish = _finish_radius(geo)
        kappa = _major_kappa(edge) if not geo.rotating else math.pi / 2.0
        out = ToolOutline(bool(geo.rotating), pts, n_teeth, finish, radius, str(geo.type), float(geo.hand_sign),
                          geo.thread, kappa)
        self._cache[slot] = out
        return out


def _finish_radius(geo) -> float:
    if geo.insert is not None and float(np.asarray(geo.insert.nose_radius_m)[0]) > 0.0:
        return float(np.asarray(geo.insert.nose_radius_m)[0])
    if geo.type == "ball_endmill":
        return 0.5 * float(geo.props.diameter_m)
    spec = geo.spec.geometry if geo.spec is not None else None
    corner = None
    if spec is not None:
        corner = spec.get("corner_radius_m") if hasattr(spec, "get") else getattr(spec, "corner_radius_m", None)
    corner = getattr(corner, "value", corner)
    return float(corner) if corner else 0.0


def _major_kappa(edge: EdgeModel) -> float:
    """Entering angle of the longest straight cutting element group of a turning tool (from the feed)."""
    kind = np.asarray(edge.elem_kind)
    valid = np.asarray(edge.elem_valid) != 0
    kappa = np.asarray(edge.elem_kappa_rad)
    length = np.asarray(edge.elem_len_m)
    straight = valid & ((kind == 8) | (kind == 0))         # EDGE_INSERT / EDGE_PERIPHERAL
    if not straight.any():
        return math.pi / 2.0
    k = float(kappa[straight][np.argmax(length[straight])])
    # elements carry kappa = 90 deg - kappa_r on the major edge (docs/models/tooling.md 2.2)
    return min(max(math.pi / 2.0 - k, 1.0e-3), math.pi - 1.0e-3)


def envelope_outline(envelope) -> np.ndarray:
    """Dense (r, z) polyline of the cutting pieces of chain 0 of a tooling Envelope, in profile order."""
    rows: list[tuple[float, float]] = []
    kind = np.asarray(envelope.kind)
    for i in range(len(kind)):
        if int(envelope.chain[i]) != 0 or int(envelope.cutting[i]) != 1:
            continue
        if int(kind[i]) == 1:          # ENV_ARC
            zc, rc, rho = float(envelope.zc_m[i]), float(envelope.rc_m[i]), float(envelope.rho_m[i])
            a0, a1 = float(envelope.a0_rad[i]), float(envelope.a1_rad[i])
            n = max(2, int(math.ceil(abs(a1 - a0) / OUTLINE_ARC_STEP_RAD)) + 1)
            for a in np.linspace(a0, a1, n):
                rows.append((rc + rho * math.sin(a), zc - rho * math.cos(a)))
        else:
            z0, r0, z1, r1 = (float(envelope.z0_m[i]), float(envelope.r0_m[i]), float(envelope.z1_m[i]),
                              float(envelope.r1_m[i]))
            n = max(2, int(math.ceil(math.hypot(z1 - z0, r1 - r0) / OUTLINE_LINE_STEP_M)) + 1)
            for u in np.linspace(0.0, 1.0, n):
                rows.append((r0 + u * (r1 - r0), z0 + u * (z1 - z0)))
    if len(rows) < 2:
        raise InputError("surface: the tool envelope has no cutting pieces")
    return _dedupe(np.array(rows))


def insert_outline(insert) -> np.ndarray:
    """Dense (x, z) polyline of the cutting chain of a tooling InsertProfile, the nose arc exact."""
    pts = np.asarray(insert.pts_zx_m, dtype=float)
    cut = np.asarray(insert.cutting) != 0
    n = len(pts)
    if n < 2 or not cut.any():
        raise InputError("surface: the insert outline has no cutting segments")
    start = 0
    for i in range(n):
        if cut[i] and not cut[i - 1]:
            start = i
            break
    centre = np.asarray(insert.nose_centre_zx_m, dtype=float)
    rho = float(np.asarray(insert.nose_radius_m)[0])
    rows: list[tuple[float, float]] = []
    i = start
    for _ in range(n):
        if not cut[i]:
            break
        a, b = pts[i], pts[(i + 1) % n]
        on_arc = rho > 0.0 and abs(np.hypot(*(a - centre)) - rho) < 1e-9 * max(rho, 1e-3) \
            and abs(np.hypot(*(b - centre)) - rho) < 1e-9 * max(rho, 1e-3)
        if on_arc:
            t0 = math.atan2(a[1] - centre[1], a[0] - centre[0])
            t1 = math.atan2(b[1] - centre[1], b[0] - centre[0])
            d = (t1 - t0 + math.pi) % (2.0 * math.pi) - math.pi
            m = max(2, int(math.ceil(abs(d) / OUTLINE_ARC_STEP_RAD)) + 1)
            for t in np.linspace(t0, t0 + d, m):
                rows.append((centre[1] + rho * math.sin(t), centre[0] + rho * math.cos(t)))
        else:
            m = max(2, int(math.ceil(float(np.hypot(*(b - a))) / OUTLINE_LINE_STEP_M)) + 1)
            for u in np.linspace(0.0, 1.0, m):
                p = a + u * (b - a)
                rows.append((float(p[1]), float(p[0])))
        i = (i + 1) % n
    return _dedupe(np.array(rows))


def _dedupe(rows: np.ndarray) -> np.ndarray:
    keep = np.ones(len(rows), bool)
    keep[1:] = np.any(np.abs(np.diff(rows, axis=0)) > 1e-12, axis=1)
    return np.ascontiguousarray(rows[keep])


# ----------------------------------------------------------------------------------- analytic outlines
def endmill_outline(radius_m: float, corner_m: float = 0.0, dish_rad: float = 0.0, length_m: float = 0.02,
                    core_m: float = 0.0) -> np.ndarray:
    """(r, z) of a flat / bull-nose end mill: dished end from the core radius, corner arc, periphery."""
    rows = []
    r_end = radius_m - corner_m
    z_corner = 0.0
    n_end = max(2, int(math.ceil((r_end - core_m) / OUTLINE_LINE_STEP_M)) + 1)
    for r in np.linspace(core_m, r_end, n_end):
        rows.append((r, (r_end - r) * math.tan(dish_rad)))
    if corner_m > 0.0:
        n = max(2, int(math.ceil((math.pi / 2.0 - dish_rad) / OUTLINE_ARC_STEP_RAD)) + 1)
        for a in np.linspace(dish_rad, math.pi / 2.0, n):
            rows.append((r_end + corner_m * math.sin(a), z_corner + corner_m * (1.0 - math.cos(a))))
    z0 = corner_m
    n_side = max(2, int(math.ceil((length_m - z0) / OUTLINE_LINE_STEP_M)) + 1)
    for z in np.linspace(z0, length_m, n_side):
        rows.append((radius_m, z))
    return _dedupe(np.array(rows))


def ball_outline(radius_m: float, length_m: float = 0.02) -> np.ndarray:
    """(r, z) of a ball-nose end mill: the hemisphere from the tip, then the periphery."""
    rows = []
    n = max(2, int(math.ceil((math.pi / 2.0) / OUTLINE_ARC_STEP_RAD)) + 1)
    for a in np.linspace(0.0, math.pi / 2.0, n):
        rows.append((radius_m * math.sin(a), radius_m * (1.0 - math.cos(a))))
    n_side = max(2, int(math.ceil((length_m - radius_m) / OUTLINE_LINE_STEP_M)) + 1)
    for z in np.linspace(radius_m, length_m, n_side):
        rows.append((radius_m, z))
    return _dedupe(np.array(rows))


def turning_insert_outline(nose_m: float, kappa_r_rad: float, kappa_minor_rad: float,
                           edge_m: float = 3.0e-3) -> np.ndarray:
    """(x_T, z_T) of a turning insert (tooling convention: x_T into the material, the major edge facing -z_T
    and leading when feeding along -z_T, origin at the imaginary tip): minor edge, nose arc, major edge.

    The nose centre is at (-nose, nose); the major edge leaves the arc at the angle kappa_r from the feed axis,
    the minor edge at kappa_r' on the other side."""
    cx, cz = -nose_m, nose_m
    rows = []
    # arc parameter a: point (cx + nose cos a, cz + nose sin a), outward normal (cos a, sin a); a = 0 is the
    # deepest point (x_T = 0). The minor (trailing) edge leaves the arc at a = kappa', the major at a = -kappa_r.
    a_minor = kappa_minor_rad
    a_major = -kappa_r_rad
    tm = (cx + nose_m * math.cos(a_minor), cz + nose_m * math.sin(a_minor))
    tM = (cx + nose_m * math.cos(a_major), cz + nose_m * math.sin(a_major))
    dir_minor = (-math.sin(kappa_minor_rad), math.cos(kappa_minor_rad))
    dir_major = (-math.sin(kappa_r_rad), -math.cos(kappa_r_rad))
    n_e = max(2, int(math.ceil(edge_m / OUTLINE_LINE_STEP_M)) + 1)
    for u in np.linspace(edge_m, 0.0, n_e):
        rows.append((tm[0] + u * dir_minor[0], tm[1] + u * dir_minor[1]))
    n_a = max(2, int(math.ceil(abs(a_minor - a_major) / OUTLINE_ARC_STEP_RAD)) + 1)
    for a in np.linspace(a_minor, a_major, n_a):
        rows.append((cx + nose_m * math.cos(a), cz + nose_m * math.sin(a)))
    for u in np.linspace(0.0, edge_m, n_e):
        rows.append((tM[0] + u * dir_major[0], tM[1] + u * dir_major[1]))
    return _dedupe(np.array(rows))


# ----------------------------------------------------------------------------------- workpiece queries
@runtime_checkable
class WorkpieceQuery(Protocol):
    def point_inside(self, workpiece: int, points_S_m) -> np.ndarray:
        """bool [n]: the points lie in material."""

    def ray_cast(self, workpiece: int, origin_S_m, direction, t_max: float, exit_: bool = False):
        """(hit, t, point, outward normal) of the first entry (or exit) along the ray."""


class StockQuery:
    """Analytic block / cylinder / tube stock in S (the default before removal's field is wired)."""

    def __init__(self, kind: str, size_m):
        self.kind = str(kind)
        self.size = np.asarray(size_m, dtype=float).reshape(-1)
        if self.kind not in ("block", "cylinder", "tube"):
            raise InputError(f"surface: no analytic stock query for {self.kind} stock")

    def _inside(self, p: np.ndarray) -> np.ndarray:
        if self.kind == "block":
            return np.all((p >= 0.0) & (p <= self.size[None, :3]), axis=1)
        rho = np.hypot(p[:, 0], p[:, 1])
        inner = 0.5 * self.size[1] if self.kind == "tube" else 0.0
        return (rho <= 0.5 * self.size[0]) & (rho >= inner) & (p[:, 2] >= 0.0) & (p[:, 2] <= self.size[-1])

    def point_inside(self, workpiece: int, points_S_m) -> np.ndarray:
        return self._inside(np.asarray(points_S_m, dtype=float).reshape(-1, 3))

    def ray_cast(self, workpiece: int, origin_S_m, direction, t_max: float, exit_: bool = False):
        o = np.asarray(origin_S_m, dtype=float)
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d)
        steps = 4096
        ts = np.linspace(0.0, float(t_max), steps + 1)
        inside = self._inside(o[None, :] + ts[:, None] * d[None, :])
        want = ~inside if exit_ else inside
        hits = np.nonzero(want[1:] & ~want[:-1])[0] if not want[0] else np.array([-1])
        if len(hits) == 0:
            return False, 0.0, o.copy(), np.zeros(3)
        if hits[0] < 0:
            return True, 0.0, o.copy(), self._normal(o)
        lo, hi = ts[hits[0]], ts[hits[0] + 1]
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if bool(self._inside((o + mid * d)[None, :])[0]) != exit_:
                hi = mid
            else:
                lo = mid
        t = 0.5 * (lo + hi)
        p = o + t * d
        return True, float(t), p, self._normal(p)

    def _normal(self, p: np.ndarray) -> np.ndarray:
        if self.kind == "block":
            lo = p
            hi = self.size[:3] - p
            dist = np.concatenate([lo, hi])
            k = int(np.argmin(np.abs(dist)))
            n = np.zeros(3)
            n[k % 3] = -1.0 if k < 3 else 1.0
            return n
        rho = math.hypot(p[0], p[1])
        cands = [(abs(0.5 * self.size[0] - rho), np.array([p[0], p[1], 0.0]) / max(rho, 1e-300)),
                 (abs(p[2]), np.array([0.0, 0.0, -1.0])), (abs(self.size[-1] - p[2]), np.array([0.0, 0.0, 1.0]))]
        if self.kind == "tube":
            cands.append((abs(rho - 0.5 * self.size[1]), -np.array([p[0], p[1], 0.0]) / max(rho, 1e-300)))
        return min(cands, key=lambda c: c[0])[1]


# ----------------------------------------------------------------------------------- bricks
@runtime_checkable
class BrickGrid(Protocol):
    def brick_of(self, workpiece: int, points_S_m) -> np.ndarray:
        """int64 [n] brick index of each point (removal's level-0 numbering)."""

    def brick_key(self, workpiece: int, brick: int) -> str:
        """The brick id of bricks.idx and the attribute files."""

    def brick_size_m(self, workpiece: int) -> float:
        """Edge length of a brick."""

    def coords(self, workpiece: int, bricks) -> np.ndarray:
        """int64 [n, 3] brick coordinates (bx, by, bz) of brick indices."""


def parse_brick_key(key: str) -> tuple[int, int]:
    """(workpiece, brick) of a brick id ``"<brick>"`` / ``"w<w>.<brick>"``."""
    text = str(key)
    if text.startswith("w") and "." in text:
        w, b = text[1:].split(".", 1)
        return int(w), int(b)
    return 0, int(text)


class LatticeBricks:
    """removal's level-0 brick numbering: brick (bx, by, bz) covers nodes 32 b .. 32 b + 32 (node index // 32,
    clamped to the last brick), index = (bx * nby + by) * nbz + bz."""

    def __init__(self, origin_S_m, cell_m: float, dims, per_workpiece: dict | None = None):
        self._grids = {0: (np.asarray(origin_S_m, dtype=float).reshape(3), float(cell_m),
                           np.asarray(dims, dtype=np.int64).reshape(3))}
        for w, (o, c, d) in (per_workpiece or {}).items():
            self._grids[int(w)] = (np.asarray(o, dtype=float).reshape(3), float(c), np.asarray(d, np.int64).reshape(3))

    def _grid(self, workpiece: int):
        return self._grids.get(int(workpiece), self._grids[0])

    def brick_of(self, workpiece: int, points_S_m) -> np.ndarray:
        origin, cell, dims = self._grid(workpiece)
        p = np.asarray(points_S_m, dtype=float).reshape(-1, 3)
        nodes = np.floor((p - origin[None, :]) / cell + 0.5).astype(np.int64)
        nodes = np.clip(nodes, 0, dims[None, :] - 1)
        bdims = np.maximum((dims - 1) // BRICK_CELLS + 1, 1)
        b = np.minimum(nodes // BRICK_CELLS, bdims[None, :] - 1)
        return (b[:, 0] * bdims[1] + b[:, 1]) * bdims[2] + b[:, 2]

    def brick_key(self, workpiece: int, brick: int) -> str:
        return str(int(brick)) if int(workpiece) == 0 else f"w{int(workpiece)}.{int(brick)}"

    def brick_size_m(self, workpiece: int) -> float:
        return BRICK_CELLS * self._grid(workpiece)[1]

    def coords(self, workpiece: int, bricks) -> np.ndarray:
        _, _, dims = self._grid(workpiece)
        bdims = np.maximum((dims - 1) // BRICK_CELLS + 1, 1)
        b = np.asarray(bricks, dtype=np.int64).reshape(-1)
        bz = b % bdims[2]
        by = (b // bdims[2]) % bdims[1]
        bx = b // (bdims[2] * bdims[1])
        return np.stack([bx, by, bz], axis=1)

    @classmethod
    def for_stock(cls, kind: str, size_m, lattice_m: float, margin_nodes: int = 1) -> "LatticeBricks":
        """The level-0 lattice removal builds for a block / cylinder / tube stock (cnctwin.removal.field:
        nodes at the cell centres of the stock box, ``margin_nodes`` empty layers outside): the default before
        the integration wires removal's own field."""
        size = np.asarray(size_m, dtype=float).reshape(-1)
        if kind == "block":
            lo, hi = np.zeros(3), size[:3].copy()
        else:
            r = 0.5 * float(size[0])
            lo, hi = np.array([-r, -r, 0.0]), np.array([r, r, float(size[-1])])
        cells = np.maximum(1, np.ceil((hi - lo) / lattice_m - 1e-9).astype(np.int64))
        dims = cells + 2 * int(margin_nodes)
        origin = lo + 0.5 * lattice_m - margin_nodes * lattice_m
        return cls(origin, lattice_m, dims)


@runtime_checkable
class BrickMeshes(Protocol):
    def final_bricks(self) -> dict:
        """brick key -> (vertices [n, 3] in S, faces [m, 3]) of the final part surface."""


class RunDirBricks:
    """The final keyframe scene of a run directory (the last record of every brick)."""

    def __init__(self, run_dir):
        self.run_dir = run_dir

    def final_bricks(self) -> dict:
        from ..core.keyframes import KeyframeReader
        return KeyframeReader(self.run_dir).scene_at()


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted outward vertex normals of a triangle mesh (counter-clockwise faces seen from outside)."""
    v = np.asarray(vertices, dtype=float).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    n = np.zeros_like(v)
    if len(f):
        fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        for c in range(3):
            np.add.at(n, f[:, c], fn)
    norm = np.linalg.norm(n, axis=1)
    norm[norm == 0.0] = 1.0
    return n / norm[:, None]


__all__ = ["ToolOutline", "ToolInfo", "StaticToolInfo", "GeometryToolInfo", "envelope_outline", "insert_outline",
           "endmill_outline", "ball_outline", "turning_insert_outline", "WorkpieceQuery", "StockQuery", "BrickGrid",
           "LatticeBricks", "BrickMeshes", "RunDirBricks", "vertex_normals", "parse_brick_key", "BRICK_CELLS"]
