"""Coolant jets in the machine frame M (docs/models/coolant.md "Jets").

* Origin and aim of every outlet from its mount frame (kinematics.ChainMounts or the wired
  MountPoses): position and aim are written in the mount frame; a programmable nozzle at position
  k aims at the home aim tilted by (k - 1) x step toward -z of the mount frame (M34: away from home,
  lower, for longer tools), about the axis perpendicular to the aim and the mount's z axis.
* Through-tool jets leave the tool's coolant holes (tooling CoolantHoles, positions and directions in
  T at spindle angle 0) rotating with the spindle: the bus record carries their position at the end
  of the step, the occlusion samples ``through_tool_angles`` equally spaced spindle angles.
* Liquid jets: spread half-angle per nozzle type, intact (break-up) length of Grant & Middleman
  (1966) from the nozzle Weber, Ohnesorge and Reynolds numbers, drops of 1.89 d after break-up
  (Rayleigh 1878); gas jets (air blast, MQL, mist carrier): spreading rate and centreline decay of the
  self-similar round jet (Pope 2000), potential core B d.
* Rays: one axis ray (weight 0, the polyline drawn in the UI) and ``rays_per_jet`` rays per jet on a
  sunflower pattern of the exit disc (Vogel 1979: equal areas, deterministic), each tilted by
  (r/R) x half-angle away from the axis, carrying an equal share of the flow (top-hat jet).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np

from ..db.models import params as model_params
from . import abi
from .fluid import Props
from .supply import AEROSOL, GAS, LIQUID, Outlet

GOLDEN_ANGLE_RAD = math.pi * (3.0 - math.sqrt(5.0))


@functools.lru_cache(maxsize=1)
def P() -> dict:
    return model_params("coolant-jet")


@dataclass
class JetSpec:
    """One jet (or one angular sample of a rotating through-tool jet) in M."""

    jet_id: str
    nozzle_id: str
    kind: str
    phase: str
    origin_M: np.ndarray
    direction_M: np.ndarray
    speed_m_s: float
    flow_m3_s: float
    diameter_m: float
    rho_kg_m3: float
    liquid_m3_s: float
    half_angle_rad: float
    breakup_m: float                # arc length of break-up (inf: none / gas)
    drop_m: float                   # drop diameter after break-up
    core_m: float                   # gas: potential core
    through_tool: bool = False
    tool_row: int = -1              # Revolved row of the tool the jet leaves (through-tool), -1 none
    tool_slot: int = -1
    temp_K: float = 0.0
    weight_scale: float = 1.0       # angular samples carry flow / n_angles

    @property
    def mass_kg_s(self) -> float:
        return self.flow_m3_s * self.rho_kg_m3


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0.0 else np.array([0.0, 0.0, -1.0])


def rodrigues(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    a = unit(axis)
    c, s = math.cos(angle), math.sin(angle)
    return v * c + np.cross(a, v) * s + a * np.dot(a, v) * (1.0 - c)


def programmable_aim(home_aim, position, index: int, step_rad: float) -> np.ndarray:
    """Aim of a programmable nozzle at position ``index`` (1 = home) in its mount frame."""
    aim = unit(home_aim)
    if index <= 1 or step_rad == 0.0:
        return aim
    down = np.array([0.0, 0.0, -1.0])
    axis = np.cross(aim, down)
    if np.linalg.norm(axis) < 1e-9:
        radial = np.asarray(position, dtype=float)
        radial = np.array([radial[0], radial[1], 0.0])
        axis = np.cross(radial if np.linalg.norm(radial) > 0 else np.array([1.0, 0.0, 0.0]), down)
    return unit(rodrigues(aim, axis, (index - 1) * step_rad))


def breakup_length(speed: float, diameter: float, props: Props) -> float:
    """Intact length of a liquid jet (Grant & Middleman 1966)."""
    p = P()
    if speed <= 0.0 or diameter <= 0.0 or props.sigma_N_per_m <= 0.0:
        return math.inf
    we = props.rho_kg_m3 * speed * speed * diameter / props.sigma_N_per_m
    re = props.rho_kg_m3 * speed * diameter / props.mu_Pa_s
    if re >= p["jet_turbulent_re"]:
        return p["breakup_turbulent_c"] * we ** p["breakup_turbulent_n"] * diameter
    oh = props.mu_Pa_s / math.sqrt(props.rho_kg_m3 * props.sigma_N_per_m * diameter)
    return p["breakup_laminar_c"] * (we ** 0.5 * (1.0 + p["breakup_laminar_oh"] * oh)) ** p["breakup_laminar_n"] \
        * diameter


def gas_half_angle() -> float:
    p = P()
    return math.atan(p["gas_jet_edge_ratio"] * p["gas_jet_spreading_rate"])


def jet_spec(outlet: Outlet, origin_M, direction_M, liquid: Props | None, half_angle: float | None,
             tool_row: int = -1, tool_slot: int = -1, weight_scale: float = 1.0) -> JetSpec:
    """Physical description of one outlet's jet."""
    p = P()
    if outlet.phase == LIQUID:
        breakup = breakup_length(outlet.speed_m_s, outlet.diameter_m, liquid) if liquid is not None else math.inf
        half = half_angle if half_angle is not None else (
            p["through_tool_half_angle_rad"] if outlet.through_tool else p["liquid_jet_half_angle_rad"])
        drop = p["rayleigh_drop_ratio"] * outlet.diameter_m
        core = 0.0
    else:
        breakup = math.inf
        half = half_angle if half_angle is not None else gas_half_angle()
        drop = 0.0
        core = p["gas_jet_centreline_B"] * outlet.diameter_m
    return JetSpec(outlet.jet_id, outlet.nozzle_id, outlet.kind, outlet.phase, np.asarray(origin_M, dtype=float),
                   unit(direction_M), outlet.speed_m_s, outlet.flow_m3_s * weight_scale, outlet.diameter_m,
                   outlet.rho_kg_m3, outlet.liquid_m3_s * weight_scale, half, breakup, drop, core,
                   outlet.through_tool, tool_row, tool_slot, outlet.temp_K, weight_scale)


def through_tool_pose(path) -> tuple[np.ndarray, np.ndarray]:
    """(tip_M, R_M_T) of a path's tool at the end of the step (T rotating with the spindle)."""
    pose = np.asarray(getattr(path, "pose_M_T", np.eye(4)), dtype=float)
    tip = np.asarray(path.tip_M_m[-1], dtype=float) if len(getattr(path, "tip_M_m", [])) else pose[:3, 3]
    R = pose[:3, :3].copy()
    if len(getattr(path, "axis_T_M", [])):
        z = unit(path.axis_T_M[-1])
        if abs(np.dot(R[:, 2], z)) < 1.0 - 1e-9:
            from .kinematics import frame_from_z
            R = frame_from_z(z, R[:, 0])
    return tip, R


def hole_jets(outlet: Outlet, tool, tip_M: np.ndarray, R_M_T: np.ndarray, angles: np.ndarray, liquid: Props | None,
              tool_row: int) -> list[JetSpec]:
    """Jets of one coolant hole at spindle angles (rotation about +z_T)."""
    k = outlet.hole
    pos = tool.hole_pos_T_m[k] if 0 <= k < len(tool.hole_pos_T_m) else np.zeros(3)
    dirn = tool.hole_dir_T[k] if 0 <= k < len(tool.hole_dir_T) else np.array([0.0, 0.0, -1.0])
    out = []
    for theta in angles:
        c, s = math.cos(theta), math.sin(theta)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        origin = tip_M + R_M_T @ (Rz @ pos)
        direction = R_M_T @ (Rz @ dirn)
        out.append(jet_spec(outlet, origin, direction, liquid, None, tool_row, tool.slot, 1.0 / len(angles)))
    return out


def make_rays(jets: list[JetSpec], rays_per_jet: int, rays_through: int, s_aim: np.ndarray) -> abi.RaySet:
    """The flow-weighted rays of all jets (axis ray first, weight 0); ``s_aim`` per jet."""
    blocks = []
    for j, jet in enumerate(jets):
        n = max(int(rays_through if jet.through_tool else rays_per_jet), 1)
        a = jet.direction_M
        e1 = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(a, [0.0, 1.0, 0.0])
        e1 = unit(e1)
        e2 = np.cross(a, e1)
        k = np.arange(n, dtype=float)
        rk = np.concatenate([[0.0], np.sqrt((k + 0.5) / n)])
        phi = np.concatenate([[0.0], k * GOLDEN_ANGLE_RAD])
        weight = np.concatenate([[0.0], np.full(n, jet.flow_m3_s / n)])
        radial = np.cos(phi)[:, None] * e1[None, :] + np.sin(phi)[:, None] * e2[None, :]
        tilt = rk * jet.half_angle_rad
        direction = np.cos(tilt)[:, None] * a[None, :] + np.sin(tilt)[:, None] * radial
        is_gas = 1 if jet.phase in (GAS, AEROSOL) else 0
        m = n + 1
        blocks.append((jet.origin_M[None, :] + 0.5 * jet.diameter_m * rk[:, None] * radial,
                       jet.speed_m_s * direction, weight, np.full(m, j, np.int64), np.full(m, is_gas, np.int64),
                       np.full(m, jet.breakup_m), np.full(m, jet.drop_m), np.full(m, jet.diameter_m),
                       np.full(m, 0.0 if is_gas else jet.rho_kg_m3), np.full(m, jet.core_m),
                       np.full(m, float(s_aim[j])), np.full(m, jet.tool_row if jet.through_tool else -1, np.int64)))
    if not blocks:
        return abi.empty_rays()
    c = abi.contiguous
    cols = [np.concatenate([b[i] for b in blocks]) for i in range(12)]
    return abi.RaySet(c(cols[0], float), c(cols[1], float), c(cols[2], float), c(cols[3], np.int64),
                      c(cols[4], np.int64), c(cols[5], float), c(cols[6], float), c(cols[7], float),
                      c(cols[8], float), c(cols[9], float), c(cols[10], float), c(cols[11], np.int64))
