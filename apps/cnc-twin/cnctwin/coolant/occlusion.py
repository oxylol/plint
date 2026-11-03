"""Occlusion of the coolant jets (docs/models/coolant.md "Occlusion").

An occlusion job traces every ray of every jet once (kernels.trace_rays, then kernels.hit_meshes per
static mesh) and aggregates per cutting zone:

* **aimed** flow: rays whose unobstructed trajectory enters the zone sphere;
* **arriving** flow: aimed rays that enter the zone before (or at) their first hit;
* **shadows**: aimed but blocked flow by the occluder each ray hit first (holder of T3, shank of T3,
  tool T3, spindle nose, workpiece 0, chips, machine:table, ...);
* **reach** = arriving / aimed mass flow (0 when nothing is aimed at the zone);
* per jet and zone the arrival statistics the heat-transfer correlations need (impact speed,
  distance, offset of the impact from the zone centre, share beyond break-up);
* the **film map** on the pile grid: impingements of rays that hit the part, fixture, table or
  chips (and the film draining from tools the jets hit) -> wetted cells and flush pressure.

The result is checkpointable (:meth:`OcclusionResult.to_arrays`), so a pending job survives a resume.
"""

from __future__ import annotations

import functools
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from ..db.models import params as model_params
from ..tooling.abi import PART_BODY, PART_EDGE, PART_HOLDER, PART_INSERT, PART_NECK, PART_SHANK, PART_SPINDLE
from ..units import STANDARD_GRAVITY_M_S2
from . import abi
from . import kernels as K
from .jets import JetSpec, P as jet_params, make_rays
from .supply import LIQUID


@functools.lru_cache(maxsize=1)
def P() -> dict:
    return model_params("coolant-occlusion")


@dataclass
class Zone:
    key: int                        # contact index (>= 0), or -1 - tool slot for the zone of an idle tool
    tool_slot: int
    workpiece: int
    centre_M: np.ndarray
    radius_m: float
    heat_radius_m: float            # averaging radius on the work (Martin)
    in_cut: bool


@dataclass
class Arrival:
    """One jet at one zone."""

    jet_id: str
    phase: str
    kind: str
    aimed_kg_s: float = 0.0
    arrived_kg_s: float = 0.0
    arrived_m3_s: float = 0.0
    liquid_m3_s: float = 0.0        # liquid carried by the arriving flow (aerosols: droplets)
    speed_m_s: float = 0.0          # mean speed at the impact / closest approach
    distance_m: float = 0.0         # mean arc length from the outlet to the impact
    offset_m: float = 0.0           # mean distance of the impact from the zone centre
    spray_fraction: float = 0.0
    exit_diameter_m: float = 0.0
    exit_speed_m_s: float = 0.0
    jet_flow_m3_s: float = 0.0
    through_tool: bool = False


@dataclass
class ZoneResult:
    key: int
    tool_slot: int
    aimed_kg_s: float = 0.0
    arrived_kg_s: float = 0.0
    reach: float = 0.0
    aimed_m3_s: float = 0.0
    shadows: dict = field(default_factory=dict)          # occluder -> blocked aimed mass flow (kg/s)
    arrivals: dict = field(default_factory=dict)         # jet id -> Arrival
    nearest_jet: str = ""
    nearest_miss_m: float = math.inf


@dataclass
class OcclusionResult:
    tick: int = 0
    macro: int = 0
    apply_macro: int = 0
    zones: dict = field(default_factory=dict)            # key -> ZoneResult
    jet_reach: dict = field(default_factory=dict)        # jet id -> arriving / jet flow over all zones
    jet_blocked: dict = field(default_factory=dict)      # jet id -> occluder of most of its blocked flow
    jet_paths: dict = field(default_factory=dict)        # jet id -> axis polyline [n, 5]: x, y, z (M), s, speed
    jet_surface: dict = field(default_factory=dict)      # jet id -> impact on part / fixture / table / chips
    events: dict = field(default_factory=dict)           # zone key -> trace id of its occlusion event
    grid_origin_S: np.ndarray = field(default_factory=lambda: np.zeros(3))
    grid_axes_S: np.ndarray = field(default_factory=lambda: np.eye(3))
    grid_cell_m: float = 0.0
    wetted: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), bool))
    flush_Pa: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    n_rays: int = 0

    def to_arrays(self, prefix: str) -> tuple[dict, dict]:
        arrays = {prefix + "grid_origin_S": self.grid_origin_S, prefix + "grid_axes_S": self.grid_axes_S,
                  prefix + "wetted": self.wetted.astype(np.int8), prefix + "flush_Pa": self.flush_Pa}
        for i, (jid, path) in enumerate(sorted(self.jet_paths.items())):
            arrays[f"{prefix}path{i}"] = np.asarray(path, dtype=float)
        meta = {"tick": self.tick, "macro": self.macro, "apply_macro": self.apply_macro,
                "grid_cell_m": self.grid_cell_m, "n_rays": self.n_rays,
                "jet_reach": dict(sorted(self.jet_reach.items())), "jet_blocked": dict(sorted(self.jet_blocked.items())),
                "paths": [jid for jid, _ in sorted(self.jet_paths.items())],
                "jet_surface": {k: dict(v) for k, v in sorted(self.jet_surface.items())},
                "events": {str(k): int(v) for k, v in sorted(self.events.items())},
                "zones": [_zone_dict(z) for _, z in sorted(self.zones.items())]}
        return arrays, meta

    @classmethod
    def from_arrays(cls, arrays: dict, meta: dict, prefix: str) -> "OcclusionResult":
        out = cls(tick=int(meta["tick"]), macro=int(meta["macro"]), apply_macro=int(meta["apply_macro"]),
                  grid_cell_m=float(meta["grid_cell_m"]), n_rays=int(meta["n_rays"]))
        out.grid_origin_S = np.array(arrays[prefix + "grid_origin_S"], dtype=float)
        out.grid_axes_S = np.array(arrays[prefix + "grid_axes_S"], dtype=float)
        out.wetted = np.array(arrays[prefix + "wetted"]).astype(bool)
        out.flush_Pa = np.array(arrays[prefix + "flush_Pa"], dtype=float)
        out.jet_reach = {k: float(v) for k, v in meta["jet_reach"].items()}
        out.jet_blocked = {k: str(v) for k, v in meta["jet_blocked"].items()}
        out.jet_paths = {jid: np.array(arrays[f"{prefix}path{i}"], dtype=float) for i, jid in enumerate(meta["paths"])}
        out.jet_surface = {k: {kk: (float(vv) if not isinstance(vv, str) else vv) for kk, vv in v.items()}
                           for k, v in meta.get("jet_surface", {}).items()}
        out.events = {int(k): int(v) for k, v in meta.get("events", {}).items()}
        for zd in meta["zones"]:
            z = ZoneResult(int(zd["key"]), int(zd["tool_slot"]), float(zd["aimed_kg_s"]), float(zd["arrived_kg_s"]),
                           float(zd["reach"]), float(zd["aimed_m3_s"]), {k: float(v) for k, v in zd["shadows"].items()},
                           {k: Arrival(**v) for k, v in zd["arrivals"].items()}, str(zd["nearest_jet"]),
                           float(zd["nearest_miss_m"]) if zd["nearest_miss_m"] is not None else math.inf)
            out.zones[z.key] = z
        return out


def _zone_dict(z: ZoneResult) -> dict:
    return {"key": z.key, "tool_slot": z.tool_slot, "aimed_kg_s": z.aimed_kg_s, "arrived_kg_s": z.arrived_kg_s,
            "reach": z.reach, "aimed_m3_s": z.aimed_m3_s, "shadows": dict(sorted(z.shadows.items())),
            "arrivals": {k: asdict(v) for k, v in sorted(z.arrivals.items())}, "nearest_jet": z.nearest_jet,
            "nearest_miss_m": z.nearest_miss_m if math.isfinite(z.nearest_miss_m) else None}


@dataclass
class FilmGrid:
    """The grid the wetted and flush maps live on (the chips pile grid, or the coolant's own)."""

    origin_S: np.ndarray
    axes_S: np.ndarray
    cell_m: float
    base_w: np.ndarray
    top_w: np.ndarray


@dataclass
class Scene:
    """Everything an occlusion job traces against (built by the domain for the job's step)."""

    revolved: abi.Revolved
    revolved_names: list            # per row: (tool slot, number)
    boxes: abi.Boxes
    box_names: list                 # per row: "tool T5"
    bricks: abi.BrickGrid
    heightfield: abi.HeightField
    meshes: list                    # (name, tree, R_M_L, o_M_L)
    params: np.ndarray              # abi.TP_*
    grid: FilmGrid | None
    R_M_S: np.ndarray
    o_M_S: np.ndarray
    liquid_rho: float
    liquid_nu: float
    gas_rho: float
    gas_nu: float
    mist_wets: bool                 # mist jets carry a wetting liquid (water-based or oil) - MQL does not wet


PART_LABEL = {PART_EDGE: "tool", PART_BODY: "tool", PART_INSERT: "tool", PART_NECK: "neck", PART_SHANK: "shank",
              PART_HOLDER: "holder", PART_SPINDLE: "spindle nose"}


def occluder_name(scene: Scene, kind: int, index: int, part: int) -> str:
    if kind == abi.HIT_TOOL and 0 <= index < len(scene.revolved_names):
        slot, number = scene.revolved_names[index]
        label = PART_LABEL.get(int(part), "tool")
        if label == "spindle nose":
            return "spindle nose"
        if label == "tool":
            return f"tool T{number}"
        return f"{label} of T{number}"
    if kind == abi.HIT_BOX and 0 <= index < len(scene.box_names):
        return scene.box_names[index]
    if kind == abi.HIT_WORK:
        wp = int(scene.bricks.workpiece[index]) if 0 <= index < len(scene.bricks.workpiece) else 0
        return f"workpiece {wp}"
    if kind == abi.HIT_CHIPS:
        return "chips"
    if kind == abi.HIT_BASE:
        return "surface below (part, fixture or table)"
    if kind == abi.HIT_MESH and 0 <= index < len(scene.meshes):
        return str(scene.meshes[index][0])
    return "nothing (beyond the traced length)"


def zones_array(zones: list[Zone]) -> abi.Zones:
    if not zones:
        return abi.Zones(np.zeros((0, 3)), np.zeros(0))
    return abi.Zones(abi.contiguous(np.array([z.centre_M for z in zones], float)),
                     abi.contiguous(np.array([z.radius_m for z in zones], float)))


def aim_lengths(jets: list[JetSpec], zones: list[Zone], limit: float) -> np.ndarray:
    """How far each jet's unobstructed trajectory is followed for zone entries."""
    out = np.zeros(len(jets))
    for j, jet in enumerate(jets):
        best = 0.0
        for z in zones:
            best = max(best, 1.5 * (float(np.linalg.norm(z.centre_M - jet.origin_M)) + 2.0 * z.radius_m))
        out[j] = min(best, limit)
    return out


def trace(jets: list[JetSpec], zones: list[Zone], scene: Scene, settings) -> tuple[abi.RaySet, abi.TraceOut]:
    """Run the tracer and the mesh passes for one job."""
    s_aim = aim_lengths(jets, zones, settings.aim_length_m)
    rays = make_rays(jets, settings.rays_per_jet, settings.rays_through_tool, s_aim)
    Z = zones_array(zones)
    out = abi.trace_out(len(rays.weight), len(zones), settings.max_points)
    if len(rays.weight):
        K.trace_rays(rays, scene.params, scene.revolved, scene.boxes, scene.bricks, scene.heightfield, Z, out)
        limit = out.hit_s.copy()
        for m, (name, tree, R, o) in enumerate(scene.meshes):
            K.hit_meshes(tree, abi.contiguous(R, float), abi.contiguous(o, float), out.pts_M, out.pts_s, out.npts,
                         limit, out.hit_M, m, out.hit_kind, out.hit_index)
        changed = limit < out.hit_s
        if np.any(changed):
            _mesh_velocities(out, limit, changed)
            out.hit_s[:] = limit
    return rays, out


def _mesh_velocities(out: abi.TraceOut, limit: np.ndarray, changed: np.ndarray) -> None:
    """Velocity at a mesh hit from the stored polyline (speed interpolated, direction of the segment)."""
    for i in np.nonzero(changed)[0]:
        s = limit[i]
        n = int(out.npts[i])
        ks = np.searchsorted(out.pts_s[i, :n], s, side="right") - 1
        k = int(min(max(ks, 0), max(n - 2, 0)))
        seg = out.pts_M[i, k + 1] - out.pts_M[i, k]
        length = float(np.linalg.norm(seg))
        ds = out.pts_s[i, k + 1] - out.pts_s[i, k]
        t = (s - out.pts_s[i, k]) / ds if ds > 0 else 0.0
        speed = out.pts_v[i, k] + t * (out.pts_v[i, k + 1] - out.pts_v[i, k])
        out.hit_vel_M[i] = seg / length * speed if length > 0 else 0.0
        out.hit_part[i] = -1


def aggregate(jets: list[JetSpec], zones: list[Zone], scene: Scene, rays: abi.RaySet, out: abi.TraceOut,
              tick: int, macro: int, apply_macro: int) -> OcclusionResult:
    res = OcclusionResult(tick=tick, macro=macro, apply_macro=apply_macro, n_rays=len(rays.weight))
    n = len(rays.weight)
    ids = sorted({j.jet_id for j in jets})
    id_index = {jid: k for k, jid in enumerate(ids)}
    jet_code = np.array([id_index[jets[int(j)].jet_id] for j in rays.jet], np.int64) if n else np.zeros(0, np.int64)
    rho_jet = np.array([jets[int(j)].rho_kg_m3 for j in rays.jet]) if n else np.zeros(0)
    liquid_share = np.array([(jets[int(j)].liquid_m3_s / jets[int(j)].flow_m3_s) if jets[int(j)].flow_m3_s > 0 else 0.0
                             for j in rays.jet]) if n else np.zeros(0)
    w = rays.weight
    mass = w * rho_jet
    valid = w > 0.0
    names = [occluder_name(scene, int(out.hit_kind[i]), int(out.hit_index[i]), int(out.hit_part[i])) for i in range(n)]
    name_list = sorted(set(names))
    name_code = np.array([name_list.index(x) for x in names], np.int64) if n else np.zeros(0, np.int64)
    n_ids = len(ids)
    jet_info = {}
    for jet in jets:
        info = jet_info.setdefault(jet.jet_id, [jet, 0.0, 0.0])
        info[1] += jet.flow_m3_s
        info[2] += jet.flow_m3_s * jet.rho_kg_m3
    hit_speed = np.linalg.norm(out.hit_vel_M, axis=1) if n else np.zeros(0)
    arrived_any = np.zeros(n, bool)
    for zi, zone in enumerate(zones):
        zr = ZoneResult(zone.key, zone.tool_slot)
        entry = out.zone_s[:, zi]
        aimed = valid & np.isfinite(entry)
        arrived = aimed & (entry <= out.hit_s + 1e-12)
        blocked = aimed & ~arrived
        arrived_any |= arrived
        dist_hit = np.linalg.norm(out.hit_M - zone.centre_M, axis=1) if n else np.zeros(0)
        inside = dist_hit <= zone.radius_m
        dist = np.where(inside, out.hit_s, out.zone_smin[:, zi])
        off = np.where(inside, dist_hit, out.zone_dmin[:, zi])
        speed = np.where(inside, hit_speed, out.zone_v[:, zi])
        wa = np.where(arrived, mass, 0.0)
        sum_aimed = np.bincount(jet_code, np.where(aimed, mass, 0.0), n_ids)
        sum_arr = np.bincount(jet_code, wa, n_ids)
        sum_arr_v = np.bincount(jet_code, np.where(arrived, w, 0.0), n_ids)
        sum_liq = np.bincount(jet_code, np.where(arrived, w * liquid_share, 0.0), n_ids)
        sum_speed = np.bincount(jet_code, wa * np.where(arrived, speed, 0.0), n_ids)
        sum_dist = np.bincount(jet_code, wa * np.where(arrived, dist, 0.0), n_ids)
        sum_off = np.bincount(jet_code, wa * np.where(arrived, off, 0.0), n_ids)
        sum_spray = np.bincount(jet_code, wa * out.zone_spray[:, zi], n_ids)
        for k, jid in enumerate(ids):
            if sum_aimed[k] <= 0.0:
                continue
            jet, flow, _ = jet_info[jid]
            arr = Arrival(jid, jet.phase, jet.kind, aimed_kg_s=float(sum_aimed[k]), arrived_kg_s=float(sum_arr[k]),
                          arrived_m3_s=float(sum_arr_v[k]), liquid_m3_s=float(sum_liq[k]),
                          exit_diameter_m=jet.diameter_m, exit_speed_m_s=jet.speed_m_s, jet_flow_m3_s=flow,
                          through_tool=jet.through_tool)
            if sum_arr[k] > 0.0:
                arr.speed_m_s = float(sum_speed[k] / sum_arr[k])
                arr.distance_m = float(sum_dist[k] / sum_arr[k])
                arr.offset_m = float(sum_off[k] / sum_arr[k])
                arr.spray_fraction = float(sum_spray[k] / sum_arr[k])
            zr.arrivals[jid] = arr
        zr.aimed_kg_s = float(np.sum(np.where(aimed, mass, 0.0)))
        zr.aimed_m3_s = float(np.sum(np.where(aimed, w, 0.0)))
        zr.arrived_kg_s = float(np.sum(wa))
        if np.any(blocked):
            per = np.bincount(name_code, np.where(blocked, mass, 0.0), len(name_list))
            zr.shadows = {name_list[k]: float(per[k]) for k in range(len(name_list)) if per[k] > 0.0}
        zr.reach = zr.arrived_kg_s / zr.aimed_kg_s if zr.aimed_kg_s > 0.0 else 0.0
        if zr.aimed_kg_s <= 0.0 and n:
            miss = out.zone_dmin[:, zi] - zone.radius_m
            i = int(np.argmin(miss))
            zr.nearest_miss_m = float(miss[i])
            zr.nearest_jet = jets[int(rays.jet[i])].jet_id
        res.zones[zone.key] = zr
    arrived_mass = np.bincount(jet_code, np.where(arrived_any, mass, 0.0), n_ids) if n else np.zeros(n_ids)
    stopped = valid & ~arrived_any & (out.hit_kind != abi.HIT_LOST) & (out.hit_kind != abi.HIT_NONE)
    for k, jid in enumerate(ids):
        total = jet_info[jid][2]
        res.jet_reach[jid] = float(arrived_mass[k] / total) if total > 0.0 else 0.0
        sel = stopped & (jet_code == k)
        if np.any(sel):
            per = np.bincount(name_code[sel], mass[sel], len(name_list))
            res.jet_blocked[jid] = name_list[int(np.argmax(per))]
    for k, jid in enumerate(ids):
        axis_rays = np.nonzero((jet_code == k) & (w == 0.0))[0]
        if len(axis_rays):
            i = int(axis_rays[0])
            m = int(out.npts[i])
            res.jet_paths[jid] = np.column_stack([out.pts_M[i, :m], out.pts_s[i, :m], out.pts_v[i, :m]])
    _surface_impacts(res, jets, rays, out)
    if scene.grid is not None:
        _film(res, jets, scene, rays, out)
    return res


def _surface_impacts(res: OcclusionResult, jets, rays, out) -> None:
    """Per jet: flow, mean impact speed and distance of the rays that hit the part, fixture, table or
    chips (the far-field film)."""
    acc: dict[str, list] = {}
    for i in range(len(rays.weight)):
        if rays.weight[i] <= 0.0 or int(out.hit_kind[i]) not in (abi.HIT_WORK, abi.HIT_BASE, abi.HIT_CHIPS):
            continue
        jet = jets[int(rays.jet[i])]
        row = acc.setdefault(jet.jet_id, [0.0, 0.0, 0.0, jet])
        w = rays.weight[i]
        row[0] += w
        row[1] += w * float(np.linalg.norm(out.hit_vel_M[i]))
        row[2] += w * float(out.hit_s[i])
    for jid, (q, sv, sd, jet) in sorted(acc.items()):
        res.jet_surface[jid] = {"q_m3_s": q, "speed_m_s": sv / q, "distance_m": sd / q, "phase": jet.phase,
                                "diameter_m": jet.diameter_m, "exit_speed_m_s": jet.speed_m_s,
                                "jet_flow_m3_s": sum(j.flow_m3_s for j in jets if j.jet_id == jid),
                                "liquid_m3_s": q * (jet.liquid_m3_s / jet.flow_m3_s if jet.flow_m3_s > 0 else 0.0)}


def _film(res: OcclusionResult, jets, scene: Scene, rays, out) -> None:
    grid = scene.grid
    res.grid_origin_S = grid.origin_S.copy()
    res.grid_axes_S = grid.axes_S.copy()
    res.grid_cell_m = grid.cell_m
    nu, nv = grid.base_w.shape
    wet = np.zeros((nu, nv), np.int8)
    flush = np.zeros((nu, nv))
    rows = []
    jp = jet_params()
    surfaces = (abi.HIT_WORK, abi.HIT_BASE, abi.HIT_CHIPS, abi.HIT_MESH)
    groups: dict[tuple, list] = {}
    for i in range(len(rays.weight)):
        if rays.weight[i] <= 0.0:
            continue
        kind = int(out.hit_kind[i])
        jet = jets[int(rays.jet[i])]
        if kind in surfaces:
            groups.setdefault((jet.jet_id, 0), []).append(i)
        elif kind in (abi.HIT_TOOL, abi.HIT_BOX) and jet.phase == LIQUID:
            groups.setdefault((jet.jet_id, 1), []).append(i)
    jet_of = {}
    for jet in jets:
        jet_of.setdefault(jet.jet_id, jet)
    for (jid, drain), idx in sorted(groups.items()):
        jet = jet_of[jid]
        w = rays.weight[idx]
        q = float(w.sum())
        if q <= 0.0:
            continue
        pts_S = (out.hit_M[idx] - scene.o_M_S) @ scene.R_M_S
        g = (pts_S - grid.origin_S) @ grid.axes_S
        u = float(np.sum(g[:, 0] * w) / q)
        v = float(np.sum(g[:, 1] * w) / q)
        if drain:
            iu = int(math.floor(u / grid.cell_m))
            iv = int(math.floor(v / grid.cell_m))
            if not (0 <= iu < nu and 0 <= iv < nv) or grid.base_w[iu, iv] == -np.inf:
                continue
            fall = max(float(np.sum(g[:, 2] * w) / q) - grid.top_w[iu, iv], 0.0)
            speed = math.sqrt(2.0 * STANDARD_GRAVITY_M_S2 * fall)
            if speed <= 0.0:
                continue
            d = math.sqrt(4.0 * q / (math.pi * speed))
            rows.append([u, v, q, speed, d, scene.liquid_rho, scene.liquid_nu, 0.0, 0.0])
            continue
        speed = float(np.sum(np.linalg.norm(out.hit_vel_M[idx], axis=1) * w) / q)
        dist = float(np.sum(out.hit_s[idx] * w) / q)
        if jet.phase == LIQUID:
            d = math.sqrt(4.0 * q / (math.pi * speed)) if speed > 0.0 else jet.diameter_m
            rows.append([u, v, q, speed, d, scene.liquid_rho, scene.liquid_nu, 0.0, 0.0])
        else:
            d = max(jet.diameter_m, 2.0 * jp["gas_jet_spreading_rate"] * dist)
            kind = 2.0 if (jet.kind == "mist" and scene.mist_wets) else 1.0
            radius = dist * math.tan(jet.half_angle_rad) + 0.5 * jet.diameter_m
            rows.append([u, v, q, speed, d, scene.gas_rho, scene.gas_nu, kind, radius])
    if rows:
        imps = abi.contiguous(np.array(rows, float))
        fp = np.array([jp["watson_r0_coeff"], jp["watson_c"], jp["wall_jet_decay"], jp["film_step_m"]])
        queue = np.zeros(nu * nv, np.int64)
        stamp = np.zeros((nu, nv), np.int64)
        K.film_map(abi.contiguous(grid.base_w, float), float(grid.cell_m), imps, fp, wet, flush, queue, stamp)
    res.wetted = wet.astype(bool)
    res.flush_Pa = flush


def run(jets: list[JetSpec], zones: list[Zone], scene: Scene, settings, tick: int, macro: int,
        apply_macro: int) -> OcclusionResult:
    rays, out = trace(jets, zones, scene, settings)
    return aggregate(jets, zones, scene, rays, out, tick, macro, apply_macro)


def trace_params(gravity_M: np.ndarray, air_rho: float, air_mu: float, settings, box_lo: np.ndarray,
                 box_hi: np.ndarray) -> np.ndarray:
    jp = jet_params()
    prm = np.zeros(abi.N_TP)
    prm[abi.TP_GX:abi.TP_GX + 3] = gravity_M
    prm[abi.TP_RHO_AIR] = air_rho
    prm[abi.TP_MU_AIR] = air_mu
    prm[abi.TP_DS_MAX] = settings.step_max_m
    prm[abi.TP_DS_MIN] = settings.step_min_m
    prm[abi.TP_SAGITTA] = settings.sagitta_m
    prm[abi.TP_S_MAX] = settings.trace_length_m
    prm[abi.TP_CF_COLUMN] = jp["column_skin_friction"]
    prm[abi.TP_SN_A] = jp["drag_sn_a"]
    prm[abi.TP_SN_N] = jp["drag_sn_n"]
    prm[abi.TP_CD_NEWTON] = jp["drag_newton"]
    prm[abi.TP_RE_NEWTON] = jp["drag_re_newton"]
    prm[abi.TP_BOX_LO:abi.TP_BOX_LO + 3] = box_lo
    prm[abi.TP_BOX_HI:abi.TP_BOX_HI + 3] = box_hi
    return prm
