"""Profile lines: generating events gathered along one profile direction and their kinematic profile
(docs/models/surface.md sections 2-3).

A line belongs to one face patch and one pass of one tool. Its frame is fixed when it opens:

* ``cartesian`` (milled walls and floors): origin O (first event), unit normal n (first event's outward
  normal), profile direction d (the feed direction projected on the surface), m = d x n;
* ``cylindrical`` (turned faces about S z; bored, drilled and reamed hole walls about the tool axis): the
  meridian half-plane of the axis, with (rho, z) components d2 and n2 and the circumferential direction as m,
  so a revolved surface has no sagitta in the line coordinates.

Each event becomes one generating pass row (:data:`ROW_COLUMNS`); :func:`evaluate_line` builds the kernel
layouts, runs the envelope and the ISO evaluation, and returns a :class:`LineResult`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import abi as A
from . import kernels as K
from .roughness import IsoRules, Roughness, evaluate, primary_params

ROW_COLUMNS = ("tick", "s", "w", "m", *(f"b{i}" for i in range(A.N_BASIS)), "r_e", "z_e", "curv", "hmin", "mode",
               "tooth", "elem", "bue_d", "bue_b", "kind", "add_h", "add_b", "u_n", "VB", "edge_r")
R = {name: i for i, name in enumerate(ROW_COLUMNS)}
N_ROW = len(ROW_COLUMNS)
R_BASIS = R["b0"]

CLASSES = ("wall", "floor", "turn", "bore")
PARABOLA_SIN = math.sin(math.radians(5.0))   # cutting velocity within 5 deg of the profile plane: element path
N_PARABOLA = 129                      # chord error curv (2 t_max / 128)^2 / 8 < 1 nm for t_max <= 0.5 mm, curv <= 500 /m
MAX_GRID = 2_000_000
MAX_TEMPLATE_POINTS = 4000


@dataclass
class LineBuffer:
    patch: int
    pass_id: int
    sub: int
    slot: int
    cls: str
    cylindrical: bool
    origin: np.ndarray                  # cartesian: O (S); cylindrical: a point on the axis (S)
    d: np.ndarray                       # cartesian: d (S); cylindrical: d2 (rho, z)
    n: np.ndarray                       # cartesian: n (S); cylindrical: n2 (rho, z)
    axis: np.ndarray                    # cylindrical: axis direction (S); cartesian: m
    rho0: float = 0.0
    z0: float = 0.0
    rows: np.ndarray = field(default_factory=lambda: np.zeros((0, N_ROW)))
    count: int = 0
    first_tick: int = -1
    last_tick: int = -1
    gap_sum: float = 0.0
    gap_n: int = 0
    block_first: int = -1
    block_last: int = -1
    op: int = 0
    hold_ticks: float = 0.0             # the line stays open this long without events (4 revolutions)

    def frame_dict(self) -> dict:
        return {"patch": self.patch, "pass_id": self.pass_id, "sub": self.sub, "slot": self.slot, "cls": self.cls,
                "cylindrical": self.cylindrical, "rho0": self.rho0, "z0": self.z0, "count": self.count,
                "first_tick": self.first_tick, "last_tick": self.last_tick, "gap_sum": self.gap_sum,
                "gap_n": self.gap_n, "block_first": self.block_first, "block_last": self.block_last, "op": self.op,
                "hold_ticks": self.hold_ticks}

    def mean_gap_ticks(self) -> float:
        return self.gap_sum / self.gap_n if self.gap_n else 0.0

    def append(self, rows: np.ndarray, cap: int) -> np.ndarray:
        """Append rows (in order); returns the rows that did not fit (the caller evaluates and reopens)."""
        free = cap - self.count
        take = rows[:free]
        if self.rows.shape[0] < self.count + len(take):
            grown = np.zeros((min(cap, max(2 * self.rows.shape[0], self.count + len(take), 16)), N_ROW))
            grown[:self.count] = self.rows[:self.count]
            self.rows = grown
        self.rows[self.count:self.count + len(take)] = take
        if len(take):
            ticks = take[:, R["tick"]]
            if self.last_tick >= 0 and int(ticks[0]) > self.last_tick:
                self.gap_sum += float(ticks[0]) - self.last_tick
                self.gap_n += 1
            if len(ticks) > 1:
                steps = np.diff(ticks)
                steps = steps[steps > 0]
                self.gap_sum += float(np.sum(steps))
                self.gap_n += int(len(steps))
            if self.first_tick < 0:
                self.first_tick = int(ticks[0])
            self.last_tick = max(self.last_tick, int(np.max(ticks)))
        self.count += len(take)
        return rows[free:]


# ------------------------------------------------------------------------------------------- coordinates
def cylinder_basis(X: np.ndarray, axis_point: np.ndarray, axis_dir: np.ndarray):
    """(rho, z, rho_hat [n, 3], theta_hat [n, 3]) of points X about an axis line."""
    rel = X - axis_point[None, :]
    z = rel @ axis_dir
    radial = rel - z[:, None] * axis_dir[None, :]
    rho = np.linalg.norm(radial, axis=1)
    safe = np.where(rho > 0.0, rho, 1.0)
    rho_hat = radial / safe[:, None]
    theta_hat = np.cross(axis_dir[None, :], rho_hat)
    return rho, z, rho_hat, theta_hat


# ------------------------------------------------------------------------------------------- evaluation
@dataclass
class LineResult:
    patch: int
    pass_id: int
    slot: int
    cls: str
    roughness: Roughness
    kinematic: Roughness                # unfiltered primary profile after form removal (Pt ~ f^2 / 8 r)
    spacing_m: float
    n_passes: int
    length_m: float
    direction_S: list
    mean_u_n_m: float
    mean_VB_m: float
    first_tick: int
    last_tick: int
    block_first: int
    block_last: int
    op: int
    profile: np.ndarray | None = None   # the evaluated profile (kept only when asked for, tests and plots)
    du_m: float = 0.0


def feed_spacing(rows: np.ndarray, cls: str) -> float:
    """Median spacing of consecutive passes of the same edge element along the profile."""
    cut = rows[rows[:, R["kind"]] == A.KIND_CUT]
    if len(cut) < 2:
        return 0.0
    order = np.argsort(cut[:, R["tick"]], kind="stable")
    cut = cut[order]
    gaps = []
    if cls == "floor":
        # the same element on the same side of the axis (leading-side twins carry the mirrored radial direction)
        side = np.sign(cut[:, R_BASIS + A.B_RD])
        for e in np.unique(cut[:, R["elem"]]):
            for sd in (-1.0, 0.0, 1.0):
                s = cut[(cut[:, R["elem"]] == e) & (side == sd), R["s"]]
                if len(s) > 1:
                    gaps.append(np.abs(np.diff(s)))
    else:
        gaps.append(np.abs(np.diff(cut[:, R["s"]])))
    if not gaps:
        return 0.0
    g = np.concatenate(gaps)
    g = g[g > 1e-9]
    return float(np.median(g)) if len(g) else 0.0


def outline_window(pts: np.ndarray, elems: np.ndarray, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
    """(points of the outline within radius of any element, their arc-length positions), resampled
    uniformly to at most MAX_TEMPLATE_POINTS points."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    lo, hi = math.inf, -math.inf
    for e in elems:
        d = np.hypot(pts[:, 0] - e[0], pts[:, 1] - e[1])
        near = np.nonzero(d <= radius_m)[0]
        if len(near) == 0:
            near = np.array([int(np.argmin(d))])
        lo = min(lo, float(arc[near[0]]))
        hi = max(hi, float(arc[near[-1]]))
    lo = max(lo - radius_m, 0.0)
    hi = min(hi + radius_m, float(arc[-1]))
    count = int(min(MAX_TEMPLATE_POINTS, max(16, math.ceil((hi - lo) / 1.0e-6) + 1)))
    s = np.linspace(lo, hi, count)
    return np.stack([np.interp(s, arc, pts[:, 0]), np.interp(s, arc, pts[:, 1])], axis=1), s


def worn_outline(window: np.ndarray, arc_s: np.ndarray, outline_pts: np.ndarray, elem_rz: np.ndarray,
                 radial: np.ndarray | None, axial: np.ndarray | None) -> np.ndarray:
    """The window with the generating tooth's wear (per element, interpolated along the arc length):
    r - radial wear (the edge recedes from the material), z + axial wear (tooling's composition signs)."""
    if radial is None and axial is None:
        return window
    seg = np.linalg.norm(np.diff(outline_pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    pos = np.array([arc[int(np.argmin(np.hypot(outline_pts[:, 0] - e[0], outline_pts[:, 1] - e[1])))]
                    for e in elem_rz])
    order = np.argsort(pos, kind="stable")
    out = window.copy()
    if radial is not None and len(radial):
        out[:, 0] -= np.interp(arc_s, pos[order], np.asarray(radial, dtype=float)[order])
    if axial is not None and len(axial):
        out[:, 1] += np.interp(arc_s, pos[order], np.asarray(axial, dtype=float)[order])
    return out


RISE_MARGIN = 1.5      # the window reaches 1.5 times the distance at which the generating curve rises by (1 + recovery) h_min


def window_half(rows: np.ndarray, spacing: float, half_spacings: float, finish_radius_m: float, du: float,
                recovery: float = 0.0) -> float:
    """Half-width of the template window: ``half_spacings`` feed spacings, and at least 1.5 times the distance
    sqrt(2 (1 + recovery) h_min rho) over which the generating curve (radius rho) rises by the thickest layer the
    minimum chip thickness can leave (so that layer stays inside the windows of the passes that remove it)."""
    cut = rows[rows[:, R["kind"]] == A.KIND_CUT]
    hmin = float(np.max(cut[:, R["hmin"]])) if len(cut) else 0.0
    curv = cut[:, R["curv"]]
    curv = curv[(curv > 0.0) & (cut[:, R["mode"]] == A.MODE_PARABOLA)]
    radius = max(float(1.0 / np.min(curv)) if len(curv) else 0.0, finish_radius_m)
    rise = RISE_MARGIN * math.sqrt(2.0 * (1.0 + recovery) * hmin * radius) if hmin > 0.0 and radius > 0.0 else 0.0
    return max(half_spacings * spacing, rise, 20.0 * du)


def build_kernel_inputs(line: LineBuffer, outline_pts: np.ndarray, edge, wear_radial, wear_axial, spacing: float,
                        half_spacings: float, rules: IsoRules, finish_radius_m: float = 0.0, recovery: float = 0.0):
    """(Passes, Templates, Grid, du, sorted rows, half) of a line."""
    rows = line.rows[:line.count]
    order = np.argsort(rows[:, R["tick"]] + 0.25 * rows[:, R["kind"]], kind="stable")
    rows = rows[order]
    du = rules.spacing_for(spacing) if spacing > 0.0 else rules.stylus_for(rules.by_ra[2][1])[1]
    half = window_half(rows, spacing, half_spacings, finish_radius_m, du, recovery)
    teeth = sorted({int(t) for t in rows[rows[:, R["mode"]] == A.MODE_OUTLINE, R["tooth"]]})
    templates = []
    tpl_of_tooth = {}
    direct = []
    if teeth:
        elems = np.unique(rows[rows[:, R["mode"]] == A.MODE_OUTLINE][:, [R["r_e"], R["z_e"]]], axis=0)
        window, arc_s = outline_window(outline_pts, elems, 2.0 * half + 1.0e-4)
        valid = np.asarray(edge.elem_valid) != 0 if edge is not None else None
        for t in teeth:
            rad = axi = None
            if edge is not None and wear_radial is not None and wear_radial.shape == np.shape(edge.elem_r_m):
                rad = wear_radial[t][valid[t]]
            if edge is not None and wear_axial is not None and wear_axial.shape == np.shape(edge.elem_r_m):
                axi = wear_axial[t][valid[t]]
            if (rad is not None or axi is not None) and edge is not None:
                nominal = np.stack([np.asarray(edge.elem_r_m)[t][valid[t]], np.asarray(edge.elem_z_m)[t][valid[t]]],
                                   axis=1)
                tpl_of_tooth[t] = len(templates)
                templates.append(worn_outline(window, arc_s, outline_pts, nominal, rad, axi))
            else:
                if -1 not in tpl_of_tooth:
                    tpl_of_tooth[-1] = len(templates)
                    templates.append(window)
                tpl_of_tooth[t] = tpl_of_tooth[-1]
    add_rows = np.nonzero(rows[:, R["kind"]] == A.KIND_ADD)[0]
    for i in add_rows:
        h, b = rows[i, R["add_h"]], max(rows[i, R["add_b"]], 2.0 * du)
        u = np.linspace(-b, b, 33)
        direct.append(np.stack([u, h * (1.0 - (u / b) ** 2)], axis=1))
    n = len(rows)
    P = A.new_passes(n)
    P.kind[:] = rows[:, R["kind"]].astype(np.int64)
    P.mode[:] = rows[:, R["mode"]].astype(np.int64)
    P.s[:] = rows[:, R["s"]]
    P.w[:] = rows[:, R["w"]]
    P.m[:] = rows[:, R["m"]]
    P.basis[:] = rows[:, R_BASIS:R_BASIS + A.N_BASIS]
    P.elem_rz[:, 0] = rows[:, R["r_e"]]
    P.elem_rz[:, 1] = rows[:, R["z_e"]]
    P.curv[:] = rows[:, R["curv"]]
    P.hmin[:] = rows[:, R["hmin"]]
    P.bue[:, 0] = rows[:, R["bue_d"]]
    P.bue[:, 1] = rows[:, R["bue_b"]]
    P.half[:] = np.where(P.kind == A.KIND_ADD, np.inf, half)
    vd = np.abs(rows[:, R_BASIS + A.B_VD])
    P.t_max[:] = np.where(P.mode == A.MODE_PARABOLA, half / np.maximum(vd, 0.05), 10.0 * half + 1.0e-3)
    tpls = list(templates)
    for q, i in enumerate(add_rows):
        P.tpl[i] = len(tpls)
        P.mode[i] = A.MODE_DIRECT
        P.w[i] = 0.0
        tpls.append(direct[q])
    for i in range(n):
        if P.mode[i] == A.MODE_OUTLINE:
            P.tpl[i] = tpl_of_tooth.get(int(rows[i, R["tooth"]]), tpl_of_tooth.get(-1, 0))
    T = A.new_templates(tpls if tpls else [np.zeros((2, 2))])
    s_lo = float(np.min(P.s)) - half - 1.0e-4
    s_hi = float(np.max(P.s)) + half + 1.0e-4
    count = int(math.ceil((s_hi - s_lo) / du)) + 1
    if count > MAX_GRID:
        du = (s_hi - s_lo) / (MAX_GRID - 1)
        count = MAX_GRID
    cap = max(int(np.max(np.diff(T.off))) if len(T.off) > 1 else 2, N_PARABOLA, 2)
    G = A.new_grid(s_lo, du, count, cap)
    return P, T, G, du, rows, half


def pass_valleys(P: A.Passes, T: A.Templates, G: A.Grid, side: float | None = None) -> np.ndarray:
    """Profile position of the lowest point of every cutting pass (kernels.pass_valleys); ``side`` keeps only
    the passes whose element lies on that side of the axis along the profile (sign of r_hat . d)."""
    out = np.zeros(len(P.s))
    n = K.pass_valleys(P, T, G, N_PARABOLA, 0.0 if side is None else float(side), out)
    return out[:n].copy()


def valley_range(line: LineBuffer, P: A.Passes, T: A.Templates, G: A.Grid) -> tuple[float, float] | None:
    """Stretch between the first and the last valley; a floor line cut by both sides of the tool (leading-side
    twins) is complete only where both sides' valleys reach (the line does not see the leading passes of the
    region behind its start, nor the trailing passes ahead of its end)."""
    if line.cls == "floor":
        trail = pass_valleys(P, T, G, -1.0)
        lead = pass_valleys(P, T, G, 1.0)
        if len(lead) >= 3 and len(trail) >= 3:
            lo, hi = max(float(np.min(trail)), float(np.min(lead))), min(float(np.max(trail)), float(np.max(lead)))
            return (lo, hi) if hi > lo else None
    valleys = pass_valleys(P, T, G)
    if len(valleys) < 3:
        return None
    return float(np.min(valleys)), float(np.max(valleys))


def evaluate_line(line: LineBuffer, outline_pts: np.ndarray, edge, wear_radial, wear_axial, rules: IsoRules,
                  half_spacings: float, keep_profile: bool = False, recovery: float = 1.0,
                  finish_radius_m: float = 0.0) -> LineResult | None:
    """The line's kinematic profile and its ISO parameters (None when too few passes generated it)."""
    rows = line.rows[:line.count]
    cut = rows[rows[:, R["kind"]] == A.KIND_CUT]
    if len(cut) < 3:
        return None
    spacing = feed_spacing(rows, line.cls)
    P, T, G, du, rows, half = build_kernel_inputs(line, outline_pts, edge, wear_radial, wear_axial, spacing,
                                                  half_spacings, rules, finish_radius_m, recovery)
    K.envelope_profile(P, T, G, N_PARABOLA, float(recovery))
    z = G.z
    finite = np.isfinite(z) & (G.hits > 0)
    if not finite.any():
        return None
    # the representative stretch lies between the lowest points of the first and the last cutting pass
    # (outside it the profile is the outer flank of a single pass); within it the longest generated run
    span = valley_range(line, P, T, G)
    if span is None:
        return None
    u0 = float(G.u0[0])
    # the first and the last window are start-up / run-out transients (the surface before the first pass is
    # not known, the minimum-chip layer of the last passes is not yet removed): one window is trimmed at each end
    i_lo = max(int(math.ceil((span[0] + half - u0) / du)), 0)
    i_hi = min(int(math.floor((span[1] - half - u0) / du)), len(z) - 1)
    if i_hi - i_lo < 16:
        return None
    inner = finite[i_lo:i_hi + 1]
    if not inner.all():
        idx = np.nonzero(inner)[0]
        if len(idx) < 16:
            return None
        breaks = np.nonzero(np.diff(idx) > 1)[0]
        starts = np.concatenate([[0], breaks + 1])
        ends = np.concatenate([breaks, [len(idx) - 1]])
        best = int(np.argmax(idx[ends] - idx[starts]))
        i_lo, i_hi = i_lo + int(idx[starts[best]]), i_lo + int(idx[ends[best]])
    if i_hi - i_lo < 16:
        return None
    seg = z[i_lo:i_hi + 1].copy()
    rough = evaluate(seg, du, rules, hint_rsm_m=spacing)
    kin = primary_params(seg, du, rules)
    d_S = line.d.tolist() if not line.cylindrical else list(line.d)
    return LineResult(patch=line.patch, pass_id=line.pass_id, slot=line.slot, cls=line.cls, roughness=rough,
                      kinematic=kin, spacing_m=spacing, n_passes=int(len(cut)), length_m=(i_hi - i_lo) * du,
                      direction_S=[float(v) for v in d_S], mean_u_n_m=float(np.mean(cut[:, R["u_n"]])),
                      mean_VB_m=float(np.mean(cut[:, R["VB"]])), first_tick=line.first_tick,
                      last_tick=line.last_tick, block_first=line.block_first, block_last=line.block_last,
                      op=line.op, profile=seg if keep_profile else None, du_m=du)


__all__ = ["LineBuffer", "LineResult", "ROW_COLUMNS", "R", "N_ROW", "R_BASIS", "CLASSES", "PARABOLA_SIN",
           "N_PARABOLA", "cylinder_basis", "feed_spacing", "outline_window",
           "worn_outline", "window_half", "build_kernel_inputs", "pass_valleys", "valley_range", "evaluate_line"]
