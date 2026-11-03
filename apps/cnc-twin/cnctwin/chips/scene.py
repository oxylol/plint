"""The window chips integrate over, built from the bus record they read (docs/models/chips.md "Frames").

* :func:`window_frames` - uniform samples of the motion record: S origin and rotation in M (the
  workpiece moves with the table axes), tool tip/axis/speed per path.
* :func:`pile_axes` / :func:`new_pile` / :func:`refresh_base` - the pile grid G fixed to S along the
  S axis closest to "up", its base surface from the geometry provider.
* :func:`build_scene` - enclosure, tool and body cylinders, nozzles and jets, air and coolant.
* :func:`coolant_on_pile` - wetted cells, flush pressure and film velocity on the pile grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..units import (AIR_CONDUCTIVITY_20C_W_MK, AIR_CP_20C_J_KGK, AIR_DENSITY_20C_KG_M3, AIR_VISCOSITY_20C_PA_S,
                     STANDARD_GRAVITY_M_S2)
from . import abi, kernels
from .params import value

LIQUID_KINDS = ("straight_oil", "emulsion", "semi_synthetic", "synthetic", "water", "mql_oil")
GAS_JETS = ("air", "mist", "mql")
TURNING_SPINDLES = ("turning", "sub")
LATHE_TYPES = ("lathe", "mill_turn", "swiss")


@dataclass
class PathSamples:
    path: int
    tool_slot: int
    workpiece: int
    spindle_id: str
    tip_M: np.ndarray           # [n_t + 1, 3]
    axis_M: np.ndarray          # [n_t + 1, 3]
    omega: np.ndarray           # [n_t + 1] rad/s right-handed about +z_T (tool) or the spindle axis
    feed_M: np.ndarray          # [n_t + 1, 3]
    tip_S: np.ndarray           # [n_t + 1, 3]
    events: list = field(default_factory=list)


@dataclass
class Frames:
    tick0: int
    tick_end: int
    t0_s: float
    t_end_s: float
    dt_tick_s: float
    n_t: int
    R_M_S: np.ndarray           # [3, 3] S -> M
    o_S_M: np.ndarray           # [n_t + 1, 3] S origin in M
    paths: list[PathSamples]

    def sample_times(self) -> np.ndarray:
        return self.t0_s + self.dt_tick_s * np.arange(self.n_t + 1)

    def index(self, tick: int) -> int:
        """Nearest sample of a base tick."""
        k = int(round((tick - self.tick0) / max((self.tick_end - self.tick0) / self.n_t, 1e-30)))
        return min(max(k, 0), self.n_t)


def _resample(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return np.ascontiguousarray(np.interp(grid, times, values))
    return np.ascontiguousarray(np.stack([np.interp(grid, times, values[:, k]) for k in range(values.shape[1])],
                                         axis=1))


def window_frames(motion, tick_s: float) -> Frames | None:
    """Frames of the window of a MotionFrame (None when it has no samples)."""
    paths = [p for p in getattr(motion, "paths", []) if len(p.tick)]
    if not paths:
        return None
    first = paths[0]
    ipo = int(first.ipo_ticks)
    n_t = len(first.tick)
    tick0 = int(first.tick0)
    tick_end = tick0 + n_t * ipo
    t0 = tick0 * tick_s
    dt = ipo * tick_s
    grid = t0 + dt * np.arange(n_t + 1)
    pose_M_T = np.asarray(first.pose_M_T, dtype=float)
    pose_S_T = np.asarray(first.pose_S_T, dtype=float)
    R_M_S = pose_M_T[:3, :3] @ pose_S_T[:3, :3].T
    times = np.asarray(first.tick, dtype=float) * tick_s
    if len(first.tip_M_m) == n_t and len(first.tip_S_m) == n_t:
        o = np.asarray(first.tip_M_m, dtype=float) - np.asarray(first.tip_S_m, dtype=float) @ R_M_S.T
    else:
        o = np.repeat((pose_M_T[:3, 3] - R_M_S @ pose_S_T[:3, 3])[None, :], n_t, axis=0)
    o_S_M = _resample(times, o, grid)
    samples = []
    for p in paths:
        ptimes = np.asarray(p.tick, dtype=float) * tick_s
        tip_M = _resample(ptimes, p.tip_M_m, grid) if len(p.tip_M_m) else np.zeros((n_t + 1, 3))
        tip_S = _resample(ptimes, p.tip_S_m, grid) if len(p.tip_S_m) else np.zeros((n_t + 1, 3))
        axis = _resample(ptimes, p.axis_T_M, grid) if len(p.axis_T_M) else np.tile([0.0, 0.0, 1.0], (n_t + 1, 1))
        norms = np.linalg.norm(axis, axis=1)
        axis = np.where(norms[:, None] > 0, axis / np.where(norms > 0, norms, 1.0)[:, None], [0.0, 0.0, 1.0])
        omega = _resample(ptimes, p.spindle_omega_rad_s, grid) if len(p.spindle_omega_rad_s) else np.zeros(n_t + 1)
        feed = _resample(ptimes, p.feed_vel_M_m_s, grid) if len(p.feed_vel_M_m_s) else np.zeros((n_t + 1, 3))
        samples.append(PathSamples(int(p.path), int(p.tool_slot), int(p.workpiece), str(p.spindle_id),
                                   np.ascontiguousarray(tip_M), np.ascontiguousarray(axis), omega,
                                   np.ascontiguousarray(feed), np.ascontiguousarray(tip_S), list(p.events)))
    return Frames(tick0, tick_end, t0, t0 + n_t * dt, dt, n_t, R_M_S, o_S_M, samples)


# ------------------------------------------------------------------ machine data

def gravity_M(machine) -> np.ndarray:
    """g in M: the machine's kinematics.gravity_direction, else -z for mills and -x for lathes (turret
    above the spindle), times standard gravity."""
    kin = getattr(machine, "kinematics", None) if machine is not None else None
    direction = getattr(kin, "gravity_direction", None) if kin is not None else None
    if direction is not None:
        d = np.asarray(direction, dtype=float)
    else:
        mtype = str(getattr(machine, "type", "") or "") if machine is not None else ""
        d = np.array([-1.0, 0.0, 0.0]) if mtype in LATHE_TYPES else np.array([0.0, 0.0, -1.0])
    return STANDARD_GRAVITY_M_S2 * d / np.linalg.norm(d)


def enclosure_M(machine, fallback_lo: np.ndarray, fallback_hi: np.ndarray, margin_m: float):
    """(lo, hi) of the enclosure interior in M (machine.enclosure.box_mm), else the given box + margin."""
    enc = getattr(machine, "enclosure", None) if machine is not None else None
    box = getattr(enc, "box_m", None) if enc is not None else None
    if box is not None:
        arr = np.asarray(box, dtype=float)
        if arr.shape == (2, 3):
            return np.minimum(arr[0], arr[1]), np.maximum(arr[0], arr[1]), True
    return fallback_lo - margin_m, fallback_hi + margin_m, False


def floor_face(gravity: np.ndarray) -> tuple[int, int]:
    """(axis, side) of the enclosure face chips fall onto."""
    axis = int(np.argmax(np.abs(gravity)))
    return axis, 0 if gravity[axis] < 0 else 1


def conveyor_rate(machine) -> tuple[float, str]:
    conv = getattr(machine, "chip_conveyor", None) if machine is not None else None
    if conv is None:
        return 0.0, "none"
    kind = str(getattr(conv, "type", "none") or "none")
    rate = value(getattr(conv, "rate_kg_s", None), 0.0)
    return (0.0 if kind == "none" else float(rate)), kind


def spindle_kinds(machine) -> dict[str, str]:
    out = {}
    for sp in (getattr(machine, "spindles", None) or []) if machine is not None else []:
        out[str(getattr(sp, "id", "main"))] = str(getattr(sp, "kind", "milling") or "milling")
    return out


# ------------------------------------------------------------------ pile grid

def pile_axes(up_S: np.ndarray) -> tuple[int, int, np.ndarray]:
    """(axis, sign, axes_S) of the grid G: e_w the signed S axis closest to up, (e_u, e_v, e_w)
    right-handed from the other two S axes in increasing order."""
    axis = int(np.argmax(np.abs(up_S)))
    sign = 1 if up_S[axis] >= 0 else -1
    a, b = (k for k in range(3) if k != axis)
    e = np.eye(3)
    e_w = sign * e[axis]
    e_u = e[a]
    e_v = e[b] if np.allclose(np.cross(e[a], e[b]), e_w) else -e[b]
    return axis, sign, np.ascontiguousarray(np.stack([e_u, e_v, e_w], axis=1))


def new_pile(lo_S: np.ndarray, hi_S: np.ndarray, axes_S: np.ndarray, cell_m: float, margin_m: float) -> abi.Pile:
    corners = np.array([[x, y, z] for x in (lo_S[0], hi_S[0]) for y in (lo_S[1], hi_S[1]) for z in (lo_S[2], hi_S[2])])
    g = corners @ axes_S                       # G coordinates (u, v, w) of the corners
    u0 = g[:, 0].min() - margin_m
    v0 = g[:, 1].min() - margin_m
    nu = int(math.ceil((g[:, 0].max() + margin_m - u0) / cell_m))
    nv = int(math.ceil((g[:, 1].max() + margin_m - v0) / cell_m))
    origin = axes_S @ np.array([u0, v0, 0.0])
    return abi.pile(nu, nv, origin, axes_S, cell_m)


def cell_points_S(H: abi.Pile, iu: np.ndarray, iv: np.ndarray, w: np.ndarray | float = 0.0) -> np.ndarray:
    cell = float(H.cell_m[0])
    g = np.stack([(np.asarray(iu) + 0.5) * cell, (np.asarray(iv) + 0.5) * cell,
                  np.broadcast_to(np.asarray(w, dtype=float), np.shape(iu))], axis=-1)
    return H.origin_G_m + g @ H.axes_S.T


def refresh_base(H: abi.Pile, geometry, workpiece: int, axis: int, sign: int, box_S=None) -> np.ndarray:
    """Re-query the base surface of the cells (all, or those whose footprint meets the S box) and return the
    flat indices of cells that lost their surface while carrying chips."""
    nu, nv = H.base_m.shape
    iu, iv = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
    iu = iu.ravel()
    iv = iv.ravel()
    pts = cell_points_S(H, iu, iv)
    a, b = (k for k in range(3) if k != axis)
    if box_S is not None:
        lo, hi = (np.asarray(x, dtype=float) for x in box_S)
        cell = float(H.cell_m[0])
        sel = ((pts[:, a] >= lo[a] - cell) & (pts[:, a] <= hi[a] + cell)
               & (pts[:, b] >= lo[b] - cell) & (pts[:, b] <= hi[b] + cell))
        iu, iv, pts = iu[sel], iv[sel], pts[sel]
    if len(iu) == 0:
        return np.zeros(0, np.int64)
    height, kind = geometry.surface_height(workpiece, axis, sign, pts[:, a], pts[:, b])
    height = np.asarray(height, dtype=float)
    kind = np.asarray(kind, dtype=np.int64)
    lost = (height == -np.inf) & (H.mass_kg[iu, iv] > 0.0)
    H.base_m[iu, iv] = height
    H.base_kind[iu, iv] = np.where(height == -np.inf, abi.SURF_NONE, kind)
    return np.ravel_multi_index((iu[lost], iv[lost]), (nu, nv)) if np.any(lost) else np.zeros(0, np.int64)


def surface_temperatures(H: abi.Pile, thermal, workpiece: int, ambient_K: float) -> None:
    """Temperature of the surface below each cell: the workpiece temperature grid of the thermal record at the
    base point (part cells), else ambient."""
    H.surf_temp_K[:] = ambient_K
    grids = getattr(thermal, "work_temp_K", {}) or {}
    grid = grids.get(workpiece) if isinstance(grids, dict) else None
    if grid is None or np.size(grid) == 0:
        return
    origin = np.asarray(thermal.work_temp_origin_S_m.get(workpiece, np.zeros(3)), dtype=float)
    cell = float(thermal.work_temp_cell_m.get(workpiece, 0.0))
    if cell <= 0.0:
        return
    part = np.nonzero((H.base_kind == abi.SURF_PART) & (H.base_m > -np.inf))
    if len(part[0]) == 0:
        return
    w = H.base_m[part] - 0.5 * cell
    pts = cell_points_S(H, part[0], part[1], w)
    idx = np.floor((pts - origin) / cell).astype(np.int64)
    shape = np.array(grid.shape[:3])
    ok = np.all((idx >= 0) & (idx < shape), axis=1)
    temps = np.full(len(w), ambient_K)
    vals = grid[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
    temps[ok] = np.where(np.isfinite(vals), vals, ambient_K)
    H.surf_temp_K[part] = temps


# ------------------------------------------------------------------ coolant on the pile grid

def _on_grid(H: abi.Pile, grid: np.ndarray, origin_S: np.ndarray, cell: float) -> np.ndarray | None:
    """A coolant grid on the pile grid: as is when it has the pile's shape, else sampled at the cell centres
    from its own origin and cell (axes along the first two S axes of the pile footprint)."""
    nu, nv = H.base_m.shape
    grid = np.asarray(grid)
    if grid.ndim != 2 or grid.size == 0:
        return None
    if grid.shape == (nu, nv):
        return grid
    if cell <= 0.0:
        return None
    iu, iv = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
    pts = cell_points_S(H, iu.ravel(), iv.ravel())
    axis = int(np.argmax(np.abs(H.axes_S[:, 2])))
    a, b = (k for k in range(3) if k != axis)
    ja = np.floor((pts[:, a] - origin_S[a]) / cell).astype(np.int64)
    jb = np.floor((pts[:, b] - origin_S[b]) / cell).astype(np.int64)
    ok = (ja >= 0) & (ja < grid.shape[0]) & (jb >= 0) & (jb < grid.shape[1])
    out = np.zeros(nu * nv, dtype=grid.dtype)
    out[ok] = grid[ja[ok], jb[ok]]
    return out.reshape(nu, nv)


def coolant_on_pile(H: abi.Pile, coolant, liquid_rho: float) -> None:
    """Wetted flags, flush pressure and film velocity (G, u v) of every pile cell. The film flows down the
    flush-pressure gradient (away from the jet impingement) with the speed sqrt(2 q / rho)."""
    if coolant is None or not getattr(coolant, "on", False):
        if H.wet.any() or H.flush_Pa.any():
            H.wet[:] = 0
            H.flush_Pa[:] = 0.0
            H.film_uv_m_s[:] = 0.0
        return
    H.wet[:] = 0
    H.flush_Pa[:] = 0.0
    H.film_uv_m_s[:] = 0.0
    origin = np.asarray(getattr(coolant, "wetted_origin_S_m", np.zeros(3)), dtype=float)
    cell = float(getattr(coolant, "wetted_cell_m", 0.0) or 0.0)
    wet = _on_grid(H, getattr(coolant, "wetted", np.zeros((0, 0))), origin, cell)
    if wet is not None:
        H.wet[:] = (np.asarray(wet) != 0).astype(np.int64)
    q = _on_grid(H, getattr(coolant, "flush_pressure_Pa", np.zeros((0, 0))), origin, cell)
    if q is None or liquid_rho <= 0.0:
        return
    np.maximum(np.asarray(q, dtype=float), 0.0, out=H.flush_Pa)
    kernels.film_field(H.flush_Pa, float(H.cell_m[0]), float(liquid_rho), H.film_uv_m_s)


# ------------------------------------------------------------------ fluids

def air_properties(ambient_K: float) -> np.ndarray:
    pr = AIR_CP_20C_J_KGK * AIR_VISCOSITY_20C_PA_S / AIR_CONDUCTIVITY_20C_W_MK
    return np.array([AIR_DENSITY_20C_KG_M3, AIR_VISCOSITY_20C_PA_S, AIR_CONDUCTIVITY_20C_W_MK, pr, ambient_K])


def liquid_properties(fluid_props: dict | None, fluid_temp_K: float, film_thickness_m: float) -> np.ndarray:
    """[rho, mu, k, Pr, T, film thickness, surface tension] of the coolant liquid (zeros: none)."""
    if not fluid_props:
        return np.array([0.0, 1.0, 0.0, 1.0, fluid_temp_K, 0.0, 0.0])
    return np.array([fluid_props["rho"], fluid_props["mu"], fluid_props["k"], fluid_props["pr"], fluid_temp_K,
                     film_thickness_m, fluid_props["gamma"]])


# ------------------------------------------------------------------ the scene

@dataclass
class SceneInfo:
    scene: abi.Scene
    cyl_slots: list[int]            # owner of each cylinder: tool slot, or -1 - workpiece (turning part)
    nozzle_ids: list[str]


def build_scene(frames: Frames, tools, box_lo: np.ndarray, box_hi: np.ndarray, gravity: np.ndarray, coolant,
                air: np.ndarray, liquid: np.ndarray, aero: dict, spindles: dict[str, str], geometry,
                nozzle_r0: dict[str, float], default_nozzle_r: float, rotating_threshold: float) -> SceneInfo:
    n_t = frames.n_t
    cyl = []          # (base [n,3], axis [n,3], omega [n], radius, length, kind, surf, owner, wrap, weld)
    for ps in frames.paths:
        kind = spindles.get(ps.spindle_id, "milling")
        if kind in TURNING_SPINDLES:
            if geometry is None or not hasattr(geometry, "rotation_radius_S"):
                continue
            radius, z_lo, z_hi = geometry.rotation_radius_S(ps.workpiece)
            if radius <= 0.0 or z_hi <= z_lo:
                continue
            axis = np.repeat((frames.R_M_S @ np.array([0.0, 0.0, 1.0]))[None, :], n_t + 1, axis=0)
            base = frames.o_S_M + z_lo * axis
            cyl.append((base, axis, ps.omega, radius, z_hi - z_lo, abi.CYL_SMOOTH, abi.SURF_BODY, -1 - ps.workpiece,
                        1, 0))
            continue
        info = tools.tool(ps.tool_slot) if (tools is not None and ps.tool_slot >= 0) else None
        if info is None or not info.rotating:
            continue
        cyl.append((ps.tip_M, ps.axis_M, ps.omega, info.radius_m, info.flute_length_m, abi.CYL_PADDLE, abi.SURF_TOOL,
                    ps.tool_slot, 1, 1))
        if info.holder_length_m > 0.0:
            base = ps.tip_M + info.flute_length_m * ps.axis_M
            cyl.append((base, ps.axis_M, ps.omega, info.holder_radius_m, info.holder_length_m, abi.CYL_SMOOTH,
                        abi.SURF_HOLDER, ps.tool_slot, 1, 0))
    jets = list(getattr(coolant, "jets", []) or []) if coolant is not None else []
    jets.sort(key=lambda j: str(j.nozzle_id))
    S = abi.scene(len(cyl), n_t, len(jets), len(jets))
    S.box_lo_M[:] = box_lo
    S.box_hi_M[:] = box_hi
    S.floor[:] = floor_face(gravity)
    for c, (base, axis, omega, radius, length, ckind, surf, owner, wrap, weld) in enumerate(cyl):
        S.cyl_base_M[c] = base
        S.cyl_axis_M[c] = axis
        S.cyl_omega[c] = omega
        S.cyl_radius[c] = radius
        S.cyl_length[c] = length
        S.cyl_kind[c] = ckind
        S.cyl_surf[c] = surf
        S.cyl_owner[c] = owner
        S.cyl_wrap[c] = wrap
        S.cyl_weld[c] = weld
    liquid_running = False
    for j, jet in enumerate(jets):
        speed = float(jet.speed_m_s)
        flow = float(jet.flow_m3_s)
        r0 = math.sqrt(flow / (math.pi * speed)) if speed > 0.0 and flow > 0.0 else nozzle_r0.get(
            str(jet.nozzle_id), default_nozzle_r)
        nozzle_r0[str(jet.nozzle_id)] = r0
        direction = np.asarray(jet.direction_M, dtype=float)
        norm = np.linalg.norm(direction)
        S.sph_center_M[j] = np.asarray(jet.origin_M_m, dtype=float)
        S.sph_radius[j] = r0
        S.sph_active[j] = 1 if speed > 0.0 else 0
        S.jet_origin_M[j] = jet.origin_M_m
        S.jet_dir_M[j] = direction / norm if norm > 0 else np.array([0.0, 0.0, -1.0])
        S.jet_speed[j] = speed if norm > 0 else 0.0
        S.jet_r0[j] = r0
        half = float(jet.half_angle_rad) if float(jet.half_angle_rad) > 0.0 else float(aero["spray_half_angle_rad"])
        S.jet_tan[j] = math.tan(half)
        gas = str(jet.kind) in GAS_JETS or liquid[0] <= 0.0
        S.jet_gas[j] = 1 if gas else 0
        S.jet_fluid[j] = air[:4] if gas else liquid[:4]
        if not gas and speed > 0.0:
            liquid_running = True
    S.walls_wet[0] = 1 if liquid_running else 0
    S.air[:] = air
    S.liquid[:] = liquid
    S.jet_const[:] = (aero["jet_centreline_B"], aero["jet_spreading_S"])
    return SceneInfo(S, [c[7] for c in cyl], [str(j.nozzle_id) for j in jets])


def window_layout(frames: Frames, H: abi.Pile, gravity: np.ndarray) -> abi.Window:
    W = abi.window(frames.n_t)
    W.t0_s[0] = frames.t0_s
    W.t_end_s[0] = frames.t_end_s
    W.dt_tick_s[0] = frames.dt_tick_s
    R_M_G = frames.R_M_S @ H.axes_S
    W.R_M_G[:] = R_M_G
    W.grid_origin_M[:] = frames.o_S_M + H.origin_G_m @ frames.R_M_S.T
    W.gravity_M[:] = gravity
    return W


def static_frames(R_M_S: np.ndarray, o_S_M: np.ndarray, t_s: float, dt_s: float, tick: int) -> Frames:
    """A window without motion (end of operation, finalize): the part where it was last."""
    o = np.repeat(np.asarray(o_S_M, dtype=float)[None, :], 2, axis=0)
    return Frames(tick, tick, t_s, t_s + dt_s, dt_s, 1, np.asarray(R_M_S, dtype=float), o, [])
