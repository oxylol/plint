"""What the chips domain needs from other parts, as small Protocols (wired by the integration).

* :class:`FormationModel` - chip formation per contact (shear angle, chip ratio, strains, flow angle,
  form class, curl, single-chip size, colour, BUE): the mechanics library (cnctwin.mechanics).
* :class:`ChipGeometry` - the surface chips land on: the in-process workpiece (removal's dexel field),
  the fixture and the table, as the highest surface along an S axis; the stock bounds; whether the
  workpiece rotates (lathe) and its radius.
* :class:`ToolCatalog` - per tool slot the geometry chips meet (radius, flute length, flutes, helix,
  gullet area, holder, through-tool coolant, kind) - tooling / the project tool list.
* :class:`LocatorFaces` - the patches of the current part that the next setup clamps or locates on
  (holding / the project's next operation).

Complete implementations for analytic scenes (tests, parts that are not wired to a dexel field):
:class:`BoxScene` (a block in S from which boxes can be removed, on a table plane, with fixture
boxes), :class:`ProjectTools` (the tool list of a resolved project) and :class:`StockFaces` (the six
faces of the stock box). All geometry is in the stock lattice frame S, material configuration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from . import abi


# ------------------------------------------------------------------ formation

@dataclass(frozen=True)
class FormationQuery:
    """Mean cutting conditions of one contact over the step (mass-weighted over its emission rows)."""

    contact: int
    tool_slot: int
    h_m: float                  # uncut chip thickness
    b_m: float                  # chip width
    l_m: float                  # uncut chip length per tooth pass
    v_m_s: float                # cutting speed
    rake_rad: float
    incl_rad: float
    edge_kind: int              # cnctwin.core.abi EDGE_*
    temp_K: float               # chip temperature at separation (hot loop)
    shear_angle_rad: float      # from ProcessCoefficients (NaN when not available)
    chip_ratio: float           # h / h_c from ProcessCoefficients (NaN when not available)
    coolant_kind: str           # dry | flood | mist | mql | air | through_spindle | high_pressure
    continuous: bool            # True when the tooth never leaves the cut (turning, drilling)


@dataclass(frozen=True)
class Formation:
    shear_angle_rad: float
    chip_ratio: float           # h / h_c (< 1)
    shear_strain: float
    strain_rate_1_s: float
    chip_flow_angle_rad: float
    chip_temp_K: float
    form_code: int              # ISO 3685 code (cnctwin.core.bus.CHIP_FORMS)
    bue_active: bool
    segmented: bool
    thickness_m: float          # deformed chip thickness at h_m
    width_m: float
    length_m: float             # length of one separated chip (after breaking); <= 0: one chip per tooth pass
    curl_radius_m: float
    colour_rgb: tuple[int, int, int]


@runtime_checkable
class FormationModel(Protocol):
    def formation(self, query: FormationQuery) -> Formation:
        """Chip formation for the mean conditions of one contact."""


# ------------------------------------------------------------------ geometry

@runtime_checkable
class ChipGeometry(Protocol):
    def stock_bounds_S(self, workpiece: int) -> tuple[np.ndarray, np.ndarray]:
        """(lo [3], hi [3]) of the stock's bounding box in S (m)."""

    def surface_height(self, workpiece: int, axis: int, sign: int, a_S_m: np.ndarray, b_S_m: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
        """Highest surface along ``sign`` * S axis ``axis`` (0, 1, 2) above the points whose other two S
        coordinates (in increasing axis order) are ``a``, ``b`` ([n] each): (height [n] as the coordinate
        times ``sign`` (so larger is higher), -inf where nothing is below; kind [n] int64 abi.SURF_PART /
        SURF_FIXTURE / SURF_TABLE / SURF_NONE)."""

    def version(self, workpiece: int) -> int:
        """Increments whenever the surfaces changed (removal's field_version)."""

    def changed_box_S(self, workpiece: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Box in S containing every surface change since the previous call (None: nothing changed)."""

    def rotation_radius_S(self, workpiece: int) -> tuple[float, float, float]:
        """(largest radius of the material about the S z axis, z_lo, z_hi): the body a turning spindle
        spins (chips wrap on it)."""


@dataclass(frozen=True)
class ToolInfo:
    slot: int
    kind: str                   # tool type of the database (twist_drill, square_endmill, tap_cutting, ...)
    radius_m: float
    flute_length_m: float
    n_flutes: int
    helix_rad: float            # mean helix angle measured from the axis
    gullet_area_m2: float       # total flute cross-section of all flutes
    holder_radius_m: float
    holder_length_m: float      # shank + holder length beyond the flutes that chips can hit
    through_coolant: bool
    rotating: bool              # False for turning, boring-bar and other non-rotating tools
    spiral: str = ""            # taps: point | flute | straight

    @property
    def diameter_m(self) -> float:
        return 2.0 * self.radius_m

    @property
    def hole_making(self) -> bool:
        return self.kind in HOLE_MAKING


HOLE_MAKING = ("twist_drill", "indexable_drill", "gun_drill", "spot_drill", "centre_drill", "reamer", "tap_cutting",
               "tap_forming", "boring_head")
NON_ROTATING = ("turning_insert", "grooving_insert", "parting_insert", "threading_insert", "form_tool", "boring_bar")


@runtime_checkable
class ToolCatalog(Protocol):
    def tool(self, slot: int) -> ToolInfo | None:
        """Geometry of a tool slot (None when the slot is unknown)."""


@runtime_checkable
class LocatorFaces(Protocol):
    def faces_S(self, op_index: int) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
        """Patches of the current part the next setup clamps or locates on: (id, centre_S [3], outward
        normal_S [3], half extents [2] along the two other S axes in increasing order)."""


# ------------------------------------------------------------------ analytic implementations

@dataclass
class BoxScene:
    """A block stock [lo, hi] in S (boxes can be removed from it, exact), standing on a table plane
    (the face lo along the up axis unless ``table_S_m`` says otherwise), with fixture boxes.

    It answers surface heights along any S axis: highest of the remaining stock material (a column
    test against the removed boxes), the fixture boxes and the table plane (only along the up axis).
    """

    lo: np.ndarray
    hi: np.ndarray
    up_axis: int = 2
    up_sign: int = 1
    table_S_m: float | None = None          # coordinate of the table plane along up (S), default stock bottom
    table_half_size_m: float = math.inf     # half extent of the table around the stock centre (<= 0: no table)
    fixtures: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    removed: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    _version: int = 0
    _changed: tuple[np.ndarray, np.ndarray] | None = None

    def __post_init__(self):
        self.lo = np.asarray(self.lo, dtype=float)
        self.hi = np.asarray(self.hi, dtype=float)

    def stock_bounds_S(self, workpiece: int):
        return self.lo.copy(), self.hi.copy()

    def version(self, workpiece: int) -> int:
        return self._version

    def changed_box_S(self, workpiece: int):
        out = self._changed
        self._changed = None
        return out

    def rotation_radius_S(self, workpiece: int) -> tuple[float, float, float]:
        """(largest radius about the S z axis, z_lo, z_hi) of the stock (a bar turned on a lathe)."""
        corners = np.array([[x, y] for x in (self.lo[0], self.hi[0]) for y in (self.lo[1], self.hi[1])])
        return float(np.max(np.hypot(corners[:, 0], corners[:, 1]))), float(self.lo[2]), float(self.hi[2])

    def remove_box(self, lo_S_m, hi_S_m) -> None:
        lo = np.maximum(np.asarray(lo_S_m, dtype=float), self.lo)
        hi = np.minimum(np.asarray(hi_S_m, dtype=float), self.hi)
        if np.any(hi <= lo):
            return
        self.removed.append((lo, hi))
        self._version += 1
        if self._changed is None:
            self._changed = (lo.copy(), hi.copy())
        else:
            self._changed = (np.minimum(self._changed[0], lo), np.maximum(self._changed[1], hi))

    def _column_top(self, axis: int, sign: int, a: float, b: float, boxes_lo, boxes_hi):
        """Highest point (times sign) of the stock minus removed boxes along the column."""
        others = [k for k in range(3) if k != axis]
        p = {others[0]: a, others[1]: b}
        if not all(self.lo[k] <= p[k] < self.hi[k] for k in others):
            return -math.inf
        # material intervals along the axis: [lo, hi] minus removed boxes covering (a, b)
        intervals = [(self.lo[axis], self.hi[axis])]
        for rlo, rhi in zip(boxes_lo, boxes_hi):
            if not all(rlo[k] <= p[k] < rhi[k] for k in others):
                continue
            out = []
            for s, e in intervals:
                if rhi[axis] <= s or rlo[axis] >= e:
                    out.append((s, e))
                    continue
                if rlo[axis] > s:
                    out.append((s, rlo[axis]))
                if rhi[axis] < e:
                    out.append((rhi[axis], e))
            intervals = out
            if not intervals:
                return -math.inf
        if sign > 0:
            return max(e for _, e in intervals)
        return max(-s for s, _ in intervals)

    def surface_height(self, workpiece, axis, sign, a_S_m, b_S_m):
        a_S_m = np.asarray(a_S_m, dtype=float)
        b_S_m = np.asarray(b_S_m, dtype=float)
        n = len(a_S_m)
        height = np.full(n, -math.inf)
        kind = np.zeros(n, np.int64)
        rlo = [r[0] for r in self.removed]
        rhi = [r[1] for r in self.removed]
        others = [k for k in range(3) if k != axis]
        centre = 0.5 * (self.lo + self.hi)
        on_up = axis == self.up_axis and sign == self.up_sign
        table = (self.lo[axis] if sign > 0 else -self.hi[axis]) if self.table_S_m is None else sign * self.table_S_m
        for i in range(n):
            top = self._column_top(axis, sign, a_S_m[i], b_S_m[i], rlo, rhi)
            if top > -math.inf:
                height[i] = top
                kind[i] = abi.SURF_PART
            p = {others[0]: a_S_m[i], others[1]: b_S_m[i]}
            for flo, fhi in self.fixtures:
                if all(flo[k] <= p[k] < fhi[k] for k in others):
                    ftop = fhi[axis] if sign > 0 else -flo[axis]
                    if ftop > height[i]:
                        height[i] = ftop
                        kind[i] = abi.SURF_FIXTURE
            if on_up and height[i] == -math.inf and self.table_half_size_m > 0.0:
                if all(abs(p[k] - centre[k]) <= self.table_half_size_m for k in others):
                    height[i] = table
                    kind[i] = abi.SURF_TABLE
        return height, kind


class ProjectTools:
    """Tool geometry of a resolved project's tool list (db/_templates/tool.yaml items, SI views).

    A slot of the core ToolTable is matched by its T number and path. Values the tool file does not
    give are derived where the geometry allows: the gullet area from the chips-flute gullet area
    fraction, the holder radius from the holder nose diameter (else the tool radius).
    """

    def __init__(self, project, tool_table, gullet_area_fraction: float):
        self.project = project
        self.tool_table = tool_table
        self.fraction = float(gullet_area_fraction)
        self._cache: dict[int, ToolInfo | None] = {}

    def tool(self, slot: int) -> ToolInfo | None:
        if slot not in self._cache:
            self._cache[slot] = self._build(slot)
        return self._cache[slot]

    def _entry(self, slot: int):
        slots = getattr(self.tool_table, "slots", ()) or ()
        number = None
        for item in slots:
            if item.slot == slot:
                number = item.number
                break
        tools = getattr(self.project, "tools", None) if self.project is not None else None
        entries = getattr(tools, "list", None) if tools is not None else None
        for entry in entries or ():
            if number is not None and int(getattr(entry, "number", -1)) == int(number):
                return entry
        return None

    def _build(self, slot: int) -> ToolInfo | None:
        entry = self._entry(slot)
        if entry is None:
            return None
        tool = getattr(entry, "tool", None)
        geometry = getattr(tool, "geometry", None) if tool is not None else None
        if geometry is None:
            return None
        diameter = _f(getattr(geometry, "diameter_m", None))
        if diameter is None:
            return None
        kind = str(getattr(tool, "type", "") or "")
        flutes = int(_f(getattr(geometry, "flutes", None), 1.0))
        helix_rad = _mean(getattr(geometry, "helix_rad", None), 0.0)
        flute_len = _f(getattr(geometry, "cutting_length_m", None)) or _f(getattr(geometry, "flute_length_m", None))
        if flute_len is None:
            flute_len = diameter
        gullet = _f(getattr(geometry, "gullet_area_m2", None))
        if gullet is None or gullet <= 0.0:
            gullet = self.fraction * math.pi * diameter * diameter / 4.0
        holder = getattr(entry, "holder", None)
        nose = _f(getattr(holder, "nose_diameter_m", None)) if holder is not None else None
        gauge = _f(getattr(holder, "gauge_length_m", None), 0.0) if holder is not None else 0.0
        stickout = _f(getattr(entry, "stickout_m", None), flute_len)
        holder_len = max(stickout - flute_len, 0.0) + gauge
        spiral = str(getattr(geometry, "spiral", None) or "")
        return ToolInfo(slot=slot, kind=kind, radius_m=0.5 * diameter, flute_length_m=flute_len,
                        n_flutes=max(flutes, 1), helix_rad=helix_rad, gullet_area_m2=gullet,
                        holder_radius_m=0.5 * nose if nose else 0.5 * diameter, holder_length_m=holder_len,
                        through_coolant=getattr(geometry, "coolant_holes", None) is not None,
                        rotating=kind not in NON_ROTATING, spiral=spiral)


def _mean(x, default: float) -> float:
    if x is None:
        return default
    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return default
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return default
    return float(np.mean(arr))


def _f(x, default=None):
    if x is None:
        return default
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


@dataclass
class StockFaces:
    """The six faces of the stock box as locating patches (ids S+x, S-x, ... ): what the next setup
    may rest, clamp or locate on when holding has not wired its locators."""

    lo: np.ndarray
    hi: np.ndarray

    def faces_S(self, op_index: int):
        lo = np.asarray(self.lo, dtype=float)
        hi = np.asarray(self.hi, dtype=float)
        centre = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        out = []
        for axis in range(3):
            others = [k for k in range(3) if k != axis]
            for sign, name in ((1, "+"), (-1, "-")):
                c = centre.copy()
                c[axis] = hi[axis] if sign > 0 else lo[axis]
                n = np.zeros(3)
                n[axis] = sign
                out.append((f"S{name}{'xyz'[axis]}", c, n, half[others].copy()))
        return out


class MeshSurfaces:
    """A ChipGeometry of a workpiece provider plus static meshes in S (fixture, vise, pallet, table: everything
    that moves with the part) through cnctwin.geom.bvh: the highest mesh surface along an S axis by one
    downward ray per point, cast from one mesh height above the mesh's bounds. The part provider answers
    everything else (bounds, version, changes, rotation radius)."""

    def __init__(self, base: ChipGeometry, meshes: list):
        from ..geom.bvh import static_tree
        self.base = base
        self.items = []
        for mesh, kind in meshes:
            lo, hi = mesh.bounds()
            self.items.append((static_tree(mesh), np.asarray(lo, dtype=float), np.asarray(hi, dtype=float), int(kind)))

    def stock_bounds_S(self, workpiece: int):
        return self.base.stock_bounds_S(workpiece)

    def version(self, workpiece: int) -> int:
        return self.base.version(workpiece)

    def changed_box_S(self, workpiece: int):
        return self.base.changed_box_S(workpiece)

    def rotation_radius_S(self, workpiece: int):
        return self.base.rotation_radius_S(workpiece)

    def surface_height(self, workpiece, axis, sign, a_S_m, b_S_m):
        from ..geom.bvh import rays_first_hit_nb
        height, kind = self.base.surface_height(workpiece, axis, sign, a_S_m, b_S_m)
        height = np.array(height, dtype=float)
        kind = np.array(kind, dtype=np.int64)
        n = len(height)
        others = [k for k in range(3) if k != axis]
        for tree, lo, hi, mesh_kind in self.items:
            extent = float(hi[axis] - lo[axis]) or 1.0
            start = hi[axis] + extent if sign > 0 else lo[axis] - extent
            origins = np.zeros((n, 3))
            origins[:, others[0]] = a_S_m
            origins[:, others[1]] = b_S_m
            origins[:, axis] = start
            dirs = np.zeros((n, 3))
            dirs[:, axis] = -float(sign)
            t = np.zeros(n)
            face = np.zeros(n, np.int32)
            rays_first_hit_nb(tree, origins, dirs, np.zeros(n), np.full(n, 3.0 * extent), t, face)
            hit = face >= 0
            top = sign * (start - sign * t)
            better = hit & (top > height)
            height[better] = top[better]
            kind[better] = mesh_kind
        return height, kind
