"""Static engagement estimate: a height map of the stock carved by the program's moves.

For milling with the tool axis along +z of the program frame, the stock box is a grid of cells
(``zmap_cell_m``) holding the material top height. Each feed move is sampled along its path; at
each sample the tool's lower envelope (flat, torus, ball, or drill cone) is compared with the
heights in the front half of the tool disc (the half the tool moves into; the whole disc for a
plunge). Material above the envelope means the tool cuts: the depth of cut ``ap`` is the highest
engaged material above the tip, the width ``ae`` the extent of the engaged cells across the feed
direction. Samples of one move are evaluated against the heights before the move (exact for the
steady state of a straight pass, where the front half meets the uncut strip), then the move's
swept envelope is carved. Rapids are checked against the whole disc (``gap`` < 0: into material).

Error: ap and ae within about one cell plus the sampling step (min(cell, R/2)); the map is the
material configuration of the stock only (fixtures, previous operations' results and the part's
own CAD are not known here).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..controller.ir import Kind
from .context import Box, SanityContext, ToolFacts

EPS_CUT_M = 1.0e-6          # material thinner than 1 um over the envelope is numerical noise, not a cut
CHUNK_ELEMENTS = 2_000_000  # bound on samples x stencil cells evaluated at once (memory)


@dataclass
class Engagement:
    """Per command: whether it cuts and how (NaN/False where not evaluated). ``ap_m`` is the depth of the
    material the tool engages (from the highest engaged material down to the tip, or to the stock bottom when
    the tip is below it: a through cut); ``reach_m`` how far below that material top the tip reaches (the
    length of tool that must enter the cut, through or not)."""

    evaluated: np.ndarray
    cut: np.ndarray
    ap_m: np.ndarray
    ae_m: np.ndarray
    gap_m: np.ndarray                # rapids: smallest clearance between envelope and material (negative: into it)
    in_footprint: np.ndarray         # the tool disc overlaps the stock footprint during the move
    material_top_m: np.ndarray       # material top at the start point (centre cell), before the move
    reasons: dict[int, str] = field(default_factory=dict)   # command -> why it was not evaluated
    reach_m: np.ndarray | None = None

    def __post_init__(self):
        if self.reach_m is None:
            self.reach_m = np.full(len(self.cut), np.nan)

    @classmethod
    def empty(cls, n: int) -> "Engagement":
        return cls(np.zeros(n, bool), np.zeros(n, bool), np.full(n, np.nan), np.full(n, np.nan),
                   np.full(n, np.inf), np.zeros(n, bool), np.full(n, np.nan))


class HeightMap:
    def __init__(self, box: Box, cell_m: float, max_cells: int):
        size_x = box.hi[0] - box.lo[0]
        size_y = box.hi[1] - box.lo[1]
        cell = float(cell_m)
        while math.ceil(size_x / cell) * math.ceil(size_y / cell) > max_cells:
            cell *= 1.25
        self.cell = cell
        self.x0, self.y0 = float(box.lo[0]), float(box.lo[1])
        self.nx = max(1, int(math.ceil(size_x / cell)))
        self.ny = max(1, int(math.ceil(size_y / cell)))
        self.top = float(box.hi[2])
        self.bottom = float(box.lo[2])
        self.heights = np.full((self.nx, self.ny), self.top)

    def index(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (np.floor((x - self.x0) / self.cell).astype(np.int64),
                np.floor((y - self.y0) / self.cell).astype(np.int64))

    def centre(self, i: np.ndarray, j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.x0 + (i + 0.5) * self.cell, self.y0 + (j + 0.5) * self.cell


@dataclass
class Stencil:
    di: np.ndarray
    dj: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    lift: np.ndarray                 # envelope height above the tip at each cell
    radius: float


def tool_stencil(tool: ToolFacts, cell: float) -> Stencil | None:
    if tool.diameter_m is None or tool.diameter_m <= 0:
        return None
    radius = 0.5 * tool.diameter_m
    reach = int(math.ceil(radius / cell)) + 1
    grid = np.arange(-reach, reach + 1)
    di, dj = np.meshgrid(grid, grid, indexing="ij")
    di, dj = di.ravel(), dj.ravel()
    dx, dy = di * cell, dj * cell
    r = np.hypot(dx, dy)
    inside = r <= radius + 1e-12
    if not inside.any():
        inside = (di == 0) & (dj == 0)
    di, dj, dx, dy, r = di[inside], dj[inside], dx[inside], dy[inside], r[inside]
    return Stencil(di, dj, dx, dy, envelope_lift(tool, r), radius)


def envelope_lift(tool: ToolFacts, r: np.ndarray) -> np.ndarray:
    """Height of the tool's lower envelope above its tip at radius ``r`` (point, torus, ball, flat)."""
    radius = 0.5 * float(tool.diameter_m)
    if tool.is_drill and tool.point_angle_rad:
        return r / math.tan(0.5 * float(tool.point_angle_rad))
    corner = min(float(tool.corner_radius_m or 0.0), radius)
    if corner <= 0:
        return np.zeros_like(r)
    flat = radius - corner
    outer = np.clip(r - flat, 0.0, corner)
    return np.where(r <= flat, 0.0, corner - np.sqrt(np.maximum(corner * corner - outer * outer, 0.0)))


def estimate(program, ctx: SanityContext) -> Engagement:
    """Engagement of every command of ``program`` (mills: height map; lathes: stock envelope)."""
    n = len(program.commands)
    result = Engagement.empty(n)
    if program.machine_kind == "lathe":
        return _lathe(program, ctx, result)
    maps: dict[str, HeightMap] = {}
    stencils: dict[int, Stencil | None] = {}
    cell = float(ctx.p("zmap_cell_m"))
    max_cells = int(ctx.p("zmap_max_cells"))
    for index, command in enumerate(program.commands):
        if command.kind not in (Kind.LINEAR, Kind.ARC, Kind.RAPID, Kind.PROBE):
            continue
        modal = command.modal or {}
        offset = modal.get("work_offset", "G54")
        if offset not in maps:
            box = ctx.stock_box_W(offset)
            if box is None:
                maps[offset] = None
            else:
                maps[offset] = HeightMap(box, cell, max_cells)
        height_map = maps[offset]
        if height_map is None:
            result.reasons[index] = ("work offset not set" if ctx.stock_box_M is not None else
                                     "stock size or position unknown")
            continue
        tool_number = modal.get("tool")
        tool = ctx.tools.get(tool_number) if tool_number is not None else None
        if tool is not None and tool.is_probe and command.kind != Kind.RAPID:
            continue                    # a probe touches the part (skip moves stop at contact): it never cuts
        if tool is None or tool.diameter_m is None:
            result.reasons[index] = "tool geometry unknown"
            continue
        axis = command.tip_axis
        if axis is not None and float(axis[2]) < math.cos(float(ctx.p("max_tool_tilt_rad"))):
            result.reasons[index] = "tool axis tilted (five-axis move): the 3-axis engagement model does not apply"
            continue
        if tool_number not in stencils:
            stencils[tool_number] = tool_stencil(tool, height_map.cell)
        stencil = stencils[tool_number]
        start = program.tip_start_W_m[index]
        end = program.tip_end_W_m[index]
        if not (np.all(np.isfinite(start)) and np.all(np.isfinite(end))):
            result.reasons[index] = "position unknown"
            continue
        i0, j0 = height_map.index(np.array([start[0]]), np.array([start[1]]))
        if 0 <= i0[0] < height_map.nx and 0 <= j0[0] < height_map.ny:
            result.material_top_m[index] = height_map.heights[i0[0], j0[0]]
        result.evaluated[index] = True
        _evaluate_move(height_map, tool, stencil, command, start, end, index, result)
    return result


class _Accumulator:
    """Engagement of one command gathered over its pieces (segments, discs, samples)."""

    def __init__(self):
        self.cut = False
        self.ap = -np.inf
        self.reach = -np.inf
        self.ae = 0.0
        self.gap = np.inf
        self.in_footprint = False

    def store(self, index: int, rapid: bool, result: Engagement) -> None:
        result.in_footprint[index] = self.in_footprint
        if rapid:
            result.gap_m[index] = self.gap
            result.cut[index] = self.gap < -EPS_CUT_M
            return
        result.cut[index] = self.cut
        if self.cut:
            result.ap_m[index] = self.ap
            result.reach_m[index] = self.reach
            result.ae_m[index] = self.ae


def _material(heights: np.ndarray, bottom: float) -> np.ndarray:
    """Cells that still hold material (a through cut carves a cell below the stock bottom: none left)."""
    return heights > bottom + EPS_CUT_M


def _evaluate_move(height_map: HeightMap, tool: ToolFacts, stencil: Stencil, command, start: np.ndarray,
                   end: np.ndarray, index: int, result: Engagement) -> None:
    """Straight moves at constant Z (and arcs in XY, as chords) are evaluated exactly as swept
    segments; a vertical move by the disc at its lower end; anything else (ramps, helices) by
    samples along the path."""
    rapid = command.kind == Kind.RAPID
    acc = _Accumulator()
    arc = command.arc if command.kind == Kind.ARC else None
    flat_z = abs(end[2] - start[2]) < 1e-12
    if arc is not None and arc.plane == "XY" and flat_z and math.isfinite(arc.sweep_rad) and \
            all(math.isfinite(c) for c in arc.centre[:2]):
        points = _chords(arc, start, end, height_map.cell)
        previous = start
        cells: list[np.ndarray] = []
        for point in points:
            _segment(height_map, tool, stencil.radius, previous, point, start[2], rapid, acc, cells)
            previous = point
        if acc.cut and cells:
            radial = _arc_ae(arc, np.concatenate(cells), stencil.radius, height_map.cell)
            if radial is not None:
                acc.ae = radial
    elif math.hypot(end[0] - start[0], end[1] - start[1]) < 1e-12:
        low = end if end[2] <= start[2] else start
        if end[2] < start[2] or rapid:
            _segment(height_map, tool, stencil.radius, low, low, low[2], rapid, acc)
    elif flat_z and arc is None:
        _segment(height_map, tool, stencil.radius, start, end, start[2], rapid, acc)
    else:
        points = _samples(command, start, end, stencil.radius, height_map.cell)
        _evaluate(height_map, stencil, start, points, rapid, index, result)
        return
    acc.store(index, rapid, result)


def _chords(arc, start: np.ndarray, end: np.ndarray, cell: float) -> list[np.ndarray]:
    radius = max(arc.radius_start_m, arc.radius_end_m, 1e-9)
    sagitta = 0.25 * cell
    half_angle = math.acos(max(1.0 - sagitta / radius, -1.0)) if sagitta < radius else math.pi / 2
    count = max(1, int(math.ceil(abs(arc.sweep_rad) / (2.0 * half_angle))))
    angle0 = math.atan2(start[1] - arc.centre[1], start[0] - arc.centre[0])
    points = []
    for k in range(1, count + 1):
        t = k / count
        angle = angle0 + arc.sweep_rad * t
        r = arc.radius_start_m + (arc.radius_end_m - arc.radius_start_m) * t
        points.append(np.array([arc.centre[0] + r * math.cos(angle), arc.centre[1] + r * math.sin(angle), start[2]]))
    points[-1] = np.asarray(end, dtype=float).copy()
    return points


def _arc_ae(arc, cells: np.ndarray, radius: float, cell: float) -> float | None:
    """Radial depth of cut of an arc move in XY whose engagement is a crescent on one side of the path,
    measured from the arc's centre: outward (all engaged material beyond the centre path, a pocket wall or a
    hole) the tool's edge reaches rho + R and the material starts at the nearest engaged cell; inward (a
    boss inside the path) the edge reaches rho - R and the material ends at the farthest. Cell centres lie
    within half a cell of the material boundary. None when the engaged material lies on both sides of the
    path (the tool's whole face cuts, a floor clean-up turn): the swept-width estimate stands."""
    rho = 0.5 * (arc.radius_start_m + arc.radius_end_m)
    distance = np.hypot(cells[:, 0] - arc.centre[0], cells[:, 1] - arc.centre[1])
    half = 0.5 * cell
    if float(distance.min()) >= rho - half:
        ae = (rho + radius) - float(distance.min()) + half
    elif float(distance.max()) <= rho + half:
        ae = float(distance.max()) - (rho - radius) + half
    else:
        return None
    return min(max(ae, cell), 2.0 * radius)


def _segment(height_map: HeightMap, tool: ToolFacts, radius: float, a: np.ndarray, b: np.ndarray, z: float,
             rapid: bool, acc: _Accumulator, cells: list | None = None) -> None:
    """The tool swept from ``a`` to ``b`` at height ``z``: its lower envelope at a cell is the tool
    profile at the cell's distance from the segment (exact for a straight move at constant Z).
    ``cells`` collects the centres (x, y) of the engaged cells (arc moves measure ae from them)."""
    cell = height_map.cell
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    i0 = max(int(math.floor((min(ax, bx) - radius - height_map.x0) / cell)), 0)
    j0 = max(int(math.floor((min(ay, by) - radius - height_map.y0) / cell)), 0)
    i1 = min(int(math.floor((max(ax, bx) + radius - height_map.x0) / cell)), height_map.nx - 1)
    j1 = min(int(math.floor((max(ay, by) + radius - height_map.y0) / cell)), height_map.ny - 1)
    if i0 > i1 or j0 > j1:
        return
    xs = (height_map.x0 + (np.arange(i0, i1 + 1) + 0.5) * cell)[:, None]
    ys = (height_map.y0 + (np.arange(j0, j1 + 1) + 0.5) * cell)[None, :]
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    rx, ry = xs - ax, ys - ay
    if length2 > 0.0:
        t = np.clip((rx * dx + ry * dy) / length2, 0.0, 1.0)
        distance = np.hypot(rx - t * dx, ry - t * dy)
    else:
        distance = np.hypot(rx, ry)
    inside = distance <= radius + 1e-12
    if not inside.any():
        i = int(math.floor((ax - height_map.x0) / cell))
        j = int(math.floor((ay - height_map.y0) / cell))
        if length2 > 0.0 or not (i0 <= i <= i1 and j0 <= j <= j1):
            return
        inside = np.zeros(distance.shape, bool)
        inside[i - i0, j - j0] = True
    block = height_map.heights[i0:i1 + 1, j0:j1 + 1]
    heights = block[inside]
    distance = distance[inside]
    acc.in_footprint = True
    envelope = z + envelope_lift(tool, distance)
    material = _material(heights, height_map.bottom)
    if rapid:
        if material.any():
            acc.gap = min(acc.gap, float(np.min(envelope[material] - heights[material])))
    else:
        engaged = (heights - envelope > EPS_CUT_M) & material
        if engaged.any():
            acc.cut = True
            top = float(np.max(heights[engaged]))
            acc.ap = max(acc.ap, top - max(z, height_map.bottom))
            acc.reach = max(acc.reach, top - z)
            rx_cells = np.broadcast_to(rx, inside.shape)[inside][engaged]
            ry_cells = np.broadcast_to(ry, inside.shape)[inside][engaged]
            if cells is not None:
                cells.append(np.column_stack([rx_cells + ax, ry_cells + ay]))
            if length2 > 0.0:
                across = (-rx_cells * dy + ry_cells * dx) / math.sqrt(length2)
                width = float(np.ptp(across))
            else:
                width = max(float(np.ptp(rx_cells)), float(np.ptp(ry_cells)))
            acc.ae = max(acc.ae, min(width + cell, 2.0 * radius))
    block[inside] = np.minimum(heights, envelope)


def _samples(command, start: np.ndarray, end: np.ndarray, radius: float, cell: float) -> np.ndarray:
    step = max(min(cell, 0.5 * radius), 0.5 * cell)
    arc = command.arc if command.kind == Kind.ARC else None
    if arc is not None and arc.plane == "XY" and math.isfinite(arc.sweep_rad) and all(
            math.isfinite(c) for c in arc.centre[:2]):
        r0, r1 = arc.radius_start_m, arc.radius_end_m
        length = abs(arc.sweep_rad) * max(r0, r1, 1e-9)
        count = max(1, int(math.ceil(max(length, abs(end[2] - start[2])) / step)))
        t = np.arange(1, count + 1) / count
        angle0 = math.atan2(start[1] - arc.centre[1], start[0] - arc.centre[0])
        angles = angle0 + arc.sweep_rad * t
        radii = r0 + (r1 - r0) * t
        xs = arc.centre[0] + radii * np.cos(angles)
        ys = arc.centre[1] + radii * np.sin(angles)
        zs = start[2] + (end[2] - start[2]) * t
        return np.column_stack([xs, ys, zs])
    length = float(np.linalg.norm(end - start))
    count = max(1, int(math.ceil(length / step)))
    t = (np.arange(1, count + 1) / count)[:, None]
    return start[None, :] + (end - start)[None, :] * t


def _evaluate(height_map: HeightMap, stencil: Stencil, start: np.ndarray, points: np.ndarray, rapid: bool,
              index: int, result: Engagement) -> None:
    centre_i, centre_j = height_map.index(points[:, 0], points[:, 1])
    direction = np.zeros((len(points), 2))
    previous = np.vstack([start[None, :], points[:-1]])
    delta = points[:, :2] - previous[:, :2]
    norm = np.hypot(delta[:, 0], delta[:, 1])
    moving = norm > 1e-12
    direction[moving] = delta[moving] / norm[moving, None]
    chunk = max(1, CHUNK_ELEMENTS // max(len(stencil.di), 1))
    ap_max, reach_max, ae_max, gap_min = -np.inf, -np.inf, 0.0, np.inf
    cut = False
    in_footprint = False
    carve_rows, carve_cols, carve_z = [], [], []
    for begin in range(0, len(points), chunk):
        stop = min(begin + chunk, len(points))
        rows = centre_i[begin:stop, None] + stencil.di[None, :]
        cols = centre_j[begin:stop, None] + stencil.dj[None, :]
        valid = (rows >= 0) & (rows < height_map.nx) & (cols >= 0) & (cols < height_map.ny)
        if not valid.any():
            continue
        in_footprint = True
        rows_c = np.clip(rows, 0, height_map.nx - 1)
        cols_c = np.clip(cols, 0, height_map.ny - 1)
        heights = height_map.heights[rows_c, cols_c]
        tip_z = points[begin:stop, 2][:, None]
        envelope = tip_z + stencil.lift[None, :]
        valid = valid & _material(heights, height_map.bottom)
        excess = np.where(valid, heights - envelope, -np.inf)
        if rapid:
            gap_min = min(gap_min, float(np.min(np.where(valid, envelope - heights, np.inf))))
        else:
            dirs = direction[begin:stop]
            along = stencil.dx[None, :] * dirs[:, :1] + stencil.dy[None, :] * dirs[:, 1:2]
            plunge = ~moving[begin:stop]
            front = (along >= -1e-12) | plunge[:, None]
            engaged = (excess > EPS_CUT_M) & front
            if engaged.any():
                cut = True
                ap = np.where(engaged, heights - np.maximum(tip_z, height_map.bottom), -np.inf).max(axis=1)
                ap_max = max(ap_max, float(ap.max()))
                reach = np.where(engaged, heights - tip_z, -np.inf).max(axis=1)
                reach_max = max(reach_max, float(reach.max()))
                across = -stencil.dx[None, :] * dirs[:, 1:2] + stencil.dy[None, :] * dirs[:, :1]
                width = _extent(np.where(engaged, across, np.nan))
                plunge_width = np.maximum(_extent(np.where(engaged, stencil.dx[None, :] + 0 * across, np.nan)),
                                          _extent(np.where(engaged, stencil.dy[None, :] + 0 * across, np.nan)))
                width = np.where(plunge, plunge_width, width) + height_map.cell
                width = np.minimum(width, 2.0 * stencil.radius)
                ae_max = max(ae_max, float(np.nanmax(np.where(engaged.any(axis=1), width, np.nan))))
        carve_rows.append(rows_c[valid])
        carve_cols.append(cols_c[valid])
        carve_z.append(envelope[valid])
    if carve_rows:
        flat = np.concatenate(carve_rows) * height_map.ny + np.concatenate(carve_cols)
        values = np.concatenate(carve_z)
        np.minimum.at(height_map.heights.ravel(), flat, values)
    result.in_footprint[index] = in_footprint
    if rapid:
        result.gap_m[index] = gap_min
        result.cut[index] = gap_min < -EPS_CUT_M
        return
    result.cut[index] = cut
    if cut:
        result.ap_m[index] = ap_max
        result.reach_m[index] = reach_max
        result.ae_m[index] = ae_max


def _extent(values: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        hi = np.nanmax(np.where(np.isnan(values), -np.inf, values), axis=1)
        lo = np.nanmin(np.where(np.isnan(values), np.inf, values), axis=1)
    width = hi - lo
    return np.where(np.isfinite(width), width, 0.0)


def _lathe(program, ctx: SanityContext, result: Engagement) -> Engagement:
    radius = ctx.stock_radius_m
    for index, command in enumerate(program.commands):
        if command.kind not in (Kind.LINEAR, Kind.ARC, Kind.RAPID, Kind.PROBE):
            continue
        if radius is None:
            result.reasons[index] = "stock radius unknown"
            continue
        start = program.tip_start_W_m[index]
        end = program.tip_end_W_m[index]
        known = [abs(x) for x in (start[0], end[0]) if np.isfinite(x)]
        if not known:
            result.reasons[index] = "position unknown"
            continue
        closest = min(known)
        inside = closest < radius - EPS_CUT_M
        result.evaluated[index] = True
        result.in_footprint[index] = inside
        if command.kind == Kind.RAPID:
            result.gap_m[index] = closest - radius
            result.cut[index] = False
        else:
            result.cut[index] = inside
            result.ap_m[index] = max(radius - closest, 0.0) if inside else math.nan
            result.reach_m[index] = result.ap_m[index]
    return result
