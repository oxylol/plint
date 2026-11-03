"""Coefficient tables for cutting's pipeline (docs/models/mechanics.md, "Coefficient tables").

Two layers:

* :func:`build_process_map` solves Oxley's theory (with the modified-Merchant fallback) on a
  grid of effective uncut chip thickness t1, cutting speed and tool rake temperature for one
  work material, tool, normal rake, workpiece temperature and friction state. It is the
  expensive step (about 2 ms per node) and is rebuilt only when that context changes.
* :func:`group_coefficients` turns a map into the rows of one coefficient group at the current
  slow state (speed, tool temperature, edge radius, VB, BUE height, friction factor) by
  trilinear lookups (Numba), the oblique transformation, the edge/flank models, the heat
  partition and, when the material has measured force data for the pairing, the calibration
  (:func:`build_calibration`).

Force convention of a group (what the hot loop evaluates per element):
F_j(h) = K_j(h) h db + Ke_j dS for j = t (along the cutting velocity), r (thrust, normal to the
machined surface), a (normal to both); db the chip width, dS the edge length of the element.
The chip forms only from the layer above the stagnation point: the shearing part uses
t1 = h - h_min, the layer below it gives the edge forces Ke.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from ..core.abi import EDGE_MARGIN, EDGE_PERIPHERAL, CoeffTable, contiguous
from ..core.bus import ProcessCoefficients
from ..units import REFERENCE_TEMP_K
from . import abi
from . import contact as C
from . import heat as H
from . import kernels as K
from . import oblique as O
from . import params as P
from .materials import KienzleEntry, MechanisticEntry, ToolSide, WorkMaterial, coating_matches
from .oxley import effective_mu, kernel_inputs, zorev_split

REGIME_NONE = 0
REGIME_OXLEY = 1
REGIME_MERCHANT = 2
REGIME_INDENTATION = 3
REGIME_PLOUGHING = 4
REGIME_DATA = 5

_SCALED = np.array(abi.SCALES_WITH_T1, dtype=np.int64)
_DISCRETE = np.array(abi.DISCRETE_SLOTS, dtype=np.int64)


# ============================================================================ inputs
@dataclass(frozen=True)
class EdgeGroup:
    """Geometry shared by the elements of one coefficient group (tooling EdgeModel columns)."""

    rake_rad: float                     # normal rake
    incl_rad: float = 0.0               # inclination (helix / lead)
    clearance_rad: float = 0.12
    edge_radius_m: float = 0.0
    kind: int = EDGE_PERIPHERAL         # core.abi.EDGE_*


@dataclass(frozen=True)
class SlowState:
    """Slow state of a group for the coming macro step (bus_prev values)."""

    speed_m_s: float                    # local cutting speed of the group
    tool_temp_K: float | None = None    # ToolHeat.rake_temp_mean_K (None: adiabatic tool, Oxley's assumption)
    tool_flank_temp_K: float | None = None   # ToolHeat flank temperature (None: the rake temperature)
    work_temp_K: float = REFERENCE_TEMP_K    # ThermalFeedback.work_preheat_K
    friction_factor: float = 1.0        # CoolantFeedback.friction_factor
    VB_m: float = 0.0                   # mean flank wear land of the group (ToolWear.VB_m)
    bue_height_m: float = 0.0           # ToolWear.bue_height_m
    contact_width_m: float | None = None    # engaged edge length (tool-side heat spreading)
    contact_time_s: float | None = None     # duration of one engagement (milling); None: continuous


# ============================================================================ process map
def default_t1_grid() -> np.ndarray:
    ox = P.get(P.OXLEY)
    return np.exp(np.linspace(math.log(ox["map_t1_lo_m"]), math.log(ox["map_t1_hi_m"]), int(ox["map_t1_n"])))


def default_speed_grid(speed_m_s: float) -> np.ndarray:
    ox = P.get(P.OXLEY)
    n = int(ox["map_speed_n"])
    span = float(ox["map_speed_span"])
    if n == 1:
        return np.array([float(speed_m_s)])
    return np.exp(np.linspace(math.log(speed_m_s / span), math.log(speed_m_s * span), n))


def default_tool_temp_grid(work_temp_K: float) -> np.ndarray:
    grid = np.array(P.get(P.OXLEY)["map_ttool_K"], dtype=float)
    grid = np.unique(np.concatenate([[work_temp_K], grid[grid > work_temp_K]]))
    return grid


def build_process_map(work: WorkMaterial, tool: ToolSide | None, rake_rad: float, *, speed_grid_m_s,
                      work_temp_K: float = REFERENCE_TEMP_K, friction_factor: float = 1.0, t1_grid_m=None,
                      tool_temp_grid_K=None, adiabatic_tool: bool = False, coulomb_limit: bool = True,
                      contact_width_m: float | None = None, contact_time_s: float | None = None) -> abi.ProcessMap:
    """Oxley outputs on a (tool temperature, speed, t1) grid for one context (see module doc).

    ``adiabatic_tool`` (or no tool) gives a single tool-temperature node and Oxley's original
    assumption that all rake heat enters the chip."""
    t1 = contiguous(default_t1_grid() if t1_grid_m is None else t1_grid_m, np.float64)
    speeds = contiguous(speed_grid_m_s, np.float64)
    adiabatic = adiabatic_tool or tool is None
    if adiabatic:
        ttool = np.array([float(work_temp_K)])
    else:
        ttool = contiguous(default_tool_temp_grid(work_temp_K) if tool_temp_grid_K is None else tool_temp_grid_K,
                           np.float64)
    mu = effective_mu(work, tool, friction_factor)
    inputs = kernel_inputs(work, tool, tool_temp_K=None if adiabatic else float(ttool[0]),
                           mu_limit=mu if coulomb_limit else -1.0, mu_fallback=mu, contact_width_m=contact_width_m,
                           contact_time_s=contact_time_s)
    values = np.zeros((ttool.shape[0], speeds.shape[0], t1.shape[0], abi.N_OUT))
    K.oxley_map(t1, speeds, ttool, float(rake_rad), float(work_temp_K), *inputs, values)
    exponent = P.number(P.CONTACT, "zorev_exponent")
    for idx in np.ndindex(values.shape[:3]):
        vec = values[idx]
        if vec[abi.OX_STATUS] in (abi.STATUS_OXLEY, abi.STATUS_MERCHANT):
            vec[abi.OX_STICK] = zorev_split(vec[abi.OX_TAU_INT], vec[abi.OX_SIGMA_N], mu, exponent)[0]
    context = np.zeros(abi.N_CTX)
    context[abi.CTX_RAKE_RAD] = rake_rad
    context[abi.CTX_WORK_TEMP_K] = work_temp_K
    context[abi.CTX_MU] = mu if coulomb_limit else -1.0
    context[abi.CTX_ADIABATIC] = 1.0 if adiabatic else 0.0
    context[abi.CTX_CONTACT_WIDTH_M] = contact_width_m or 0.0
    context[abi.CTX_CONTACT_TIME_S] = contact_time_s or 0.0
    return abi.ProcessMap(t1, speeds, ttool, values, context)


def map_matches(pmap: abi.ProcessMap, rake_rad: float, work_temp_K: float, mu: float, *, rake_tol_rad: float,
                temp_tol_K: float, mu_rel_tol: float) -> bool:
    """True if a map was built for this context within the tolerances (cutting's rebuild policy)."""
    ctx = pmap.context
    if abs(ctx[abi.CTX_RAKE_RAD] - rake_rad) > rake_tol_rad or abs(ctx[abi.CTX_WORK_TEMP_K] - work_temp_K) > temp_tol_K:
        return False
    if ctx[abi.CTX_MU] <= 0.0 or mu <= 0.0:
        return ctx[abi.CTX_MU] <= 0.0 and mu <= 0.0
    return abs(ctx[abi.CTX_MU] - mu) <= mu_rel_tol * mu


def lookup(pmap: abi.ProcessMap, t1_m, speed_m_s: float, tool_temp_K: float | None) -> np.ndarray:
    """Output vectors (abi.OX_*) at each t1 for one speed and tool temperature: [n, N_OUT]."""
    query = contiguous(np.atleast_1d(np.asarray(t1_m, dtype=float)), np.float64)
    out = np.zeros((query.shape[0], abi.N_OUT))
    t_tool = float(pmap.Ttool_grid_K[0] if tool_temp_K is None else tool_temp_K)
    K.map_lookup(pmap.t1_grid_m, pmap.speed_grid_m_s, pmap.Ttool_grid_K, pmap.values, _SCALED, _DISCRETE, query,
                 float(speed_m_s), t_tool, out)
    return out


# ============================================================================ calibration
@dataclass(frozen=True)
class Calibration:
    """Measured / predicted force factors of a pairing (see :func:`build_calibration`).

    ``scale[j, e, k]`` multiplies the model's total force component j (t, r, a) at the
    calibration chip thickness ``h_nodes_m[k]`` and the reference speed ``speeds_m_s[e]``
    (Kienzle); ``edge_scale[j, e]`` multiplies the edge part (mechanistic data: the shearing
    part takes ``scale[j, e, 0]``). Outside the measured h and speed ranges the end factors
    are held, so Oxley's prediction carries the dependence on speed, temperature, rake and wear
    away from the calibration conditions."""

    kind: str                            # kienzle | mechanistic | none
    speeds_m_s: np.ndarray               # [n_entry] ascending
    h_nodes_m: np.ndarray                # [n_h]
    scale: np.ndarray                    # [3, n_entry, n_h]
    edge_scale: np.ndarray               # [3, n_entry]
    clamped: bool = False
    sources: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def factors(self, h_m: np.ndarray, speed_m_s: float) -> tuple[np.ndarray, np.ndarray]:
        """(scale [3, n_h_query], edge scale [3]) at the query thicknesses and speed."""
        h = np.atleast_1d(np.asarray(h_m, dtype=float))
        n_e = self.speeds_m_s.shape[0]
        if n_e == 1:
            weights = [(0, 1.0)]
        else:
            ls = math.log(max(speed_m_s, 1e-12))
            grid = np.log(self.speeds_m_s)
            if ls <= grid[0]:
                weights = [(0, 1.0)]
            elif ls >= grid[-1]:
                weights = [(n_e - 1, 1.0)]
            else:
                e = int(np.searchsorted(grid, ls) - 1)
                w = (ls - grid[e]) / (grid[e + 1] - grid[e])
                weights = [(e, 1.0 - w), (e + 1, w)]
        lh = np.log(np.maximum(h, 1e-300))
        nodes = np.log(self.h_nodes_m)
        out = np.zeros((3, h.shape[0]))
        edge = np.zeros(3)
        for e, w in weights:
            for j in range(3):
                out[j] += w * np.interp(lh, nodes, self.scale[j, e])
                edge[j] += w * self.edge_scale[j, e]
        return out, edge


NO_CALIBRATION = Calibration("none", np.array([1.0]), np.array([1e-4]), np.ones((3, 1, 1)), np.ones((3, 1)))


def _entry_score(entry, tool: ToolSide | None, process: str) -> int:
    tool_class = tool.tool_class if tool is not None else "carbide"
    coating = tool.coating if tool is not None else "none"
    if entry.tool_class != tool_class:
        return -1
    score = 10 * coating_matches(entry.coating, coating)
    if entry.process in (process, ) and process:
        score += 3
    elif entry.process in ("any", "", "orthogonal"):
        score += 1
    if entry.src and entry.src != "est":
        score += 2
    return score


def select_entries(work: WorkMaterial, tool: ToolSide | None, process: str):
    """(kind, entries) with the best score: mechanistic data for milling when they score at least
    as well as the Kienzle data, otherwise Kienzle; entries of equal best score are kept (several
    reference speeds)."""
    def best(entries):
        scored = [(_entry_score(e, tool, process), e) for e in entries]
        scored = [s for s in scored if s[0] >= 0]
        if not scored:
            return -1, []
        top = max(s for s, _ in scored)
        return top, [e for s, e in scored if s == top]

    k_score, k_entries = best(work.kienzle)
    m_score, m_entries = best(work.mechanistic)
    if m_entries and (process == "milling" and m_score >= k_score or not k_entries):
        return "mechanistic", m_entries
    if k_entries:
        return "kienzle", k_entries
    return "none", []


def _ref_speed(entry) -> float | None:
    cal = entry.calibration
    if cal.speed_m_s is not None:
        return 0.5 * (cal.speed_m_s[0] + cal.speed_m_s[1])
    if isinstance(entry, KienzleEntry):
        return entry.vc_ref_m_s
    if entry.speed_m_s is not None:
        return 0.5 * (entry.speed_m_s[0] + entry.speed_m_s[1])
    return None


def _ref_h_range(entry) -> tuple[float, float]:
    cal = entry.calibration
    if cal.h_m is not None and cal.h_m[1] > cal.h_m[0]:
        return cal.h_m
    if isinstance(entry, MechanisticEntry) and entry.fz_m is not None and entry.fz_m[1] > entry.fz_m[0]:
        return entry.fz_m
    cp = P.get(P.CALIBRATION)
    return float(cp["default_cal_h_lo_m"]), float(cp["default_cal_h_hi_m"])


def _model_totals(work, tool, group: EdgeGroup, state: SlowState, h_nodes: np.ndarray) -> np.ndarray:
    """Uncalibrated total forces per unit width [3, n] (t, r, a) at h_nodes for one condition."""
    pmap = build_process_map(work, tool, group.rake_rad, speed_grid_m_s=np.array([state.speed_m_s]),
                             work_temp_K=state.work_temp_K, friction_factor=state.friction_factor,
                             t1_grid_m=_calibration_t1_grid(h_nodes, work, group),
                             tool_temp_grid_K=None if state.tool_temp_K is None else np.array([state.tool_temp_K]),
                             adiabatic_tool=state.tool_temp_K is None)
    gc = group_coefficients(work, tool, group, state, h_nodes, pmap)
    return np.vstack([gc.Kt_Pa * h_nodes + gc.Kte_N_per_m, gc.Kr_Pa * h_nodes + gc.Kre_N_per_m,
                      gc.Ka_Pa * h_nodes + gc.Kae_N_per_m])


def _calibration_t1_grid(h_nodes, work, group):
    h_min = C.min_chip_thickness(work, group.edge_radius_m)
    t1 = np.asarray(h_nodes) - h_min
    t1 = t1[t1 > 0.0]
    lo = default_t1_grid()[0]
    if t1.size == 0:
        return np.array([lo])
    return np.unique(np.concatenate([[min(lo, t1.min())], t1]))


def build_calibration(work: WorkMaterial, tool: ToolSide | None, group: EdgeGroup, process: str = "turning", *,
                      tool_temp_ref_K: float | None = None) -> Calibration:
    """Calibration of the Oxley prediction to the material's measured Kienzle or mechanistic data
    for this tool class / coating / process (see :class:`Calibration`).

    Reference conditions come from each entry's ``calibration`` block, else from its stated
    reference speed and rake, else the group's own rake, the model default edge radius and
    room temperature; the reference tool is taken at the workpiece temperature (short tests)."""
    if work.jc is None:
        return NO_CALIBRATION
    kind, entries = select_entries(work, tool, process)
    if kind == "none":
        return NO_CALIBRATION
    cp = P.get(P.CALIBRATION)
    lo_clamp = float(cp["min_cal_scale"])
    hi_clamp = float(cp["max_cal_scale"])
    n_nodes = int(cp["cal_h_nodes"])
    rows = []
    for entry in entries:
        cal = entry.calibration
        rake = cal.rake_rad if cal.rake_rad is not None else (
            entry.rake_ref_rad if isinstance(entry, KienzleEntry) else entry.rake_rad)
        if rake is None:
            rake = group.rake_rad
        radius = cal.edge_radius_m if cal.edge_radius_m is not None else float(cp["default_cal_edge_radius_m"])
        work_temp = cal.work_temp_K if cal.work_temp_K is not None else float(cp["default_cal_work_temp_K"])
        incl = (entry.helix_rad or 0.0) if isinstance(entry, MechanisticEntry) else 0.0
        speed = _ref_speed(entry)
        h_lo, h_hi = _ref_h_range(entry)
        nodes = np.exp(np.linspace(math.log(h_lo), math.log(h_hi), n_nodes))
        ref_group = replace(group, rake_rad=float(rake), incl_rad=float(incl), edge_radius_m=float(radius))
        state = SlowState(speed_m_s=float(speed if speed is not None else 1.0),
                          tool_temp_K=None if tool is None else (tool_temp_ref_K or work_temp), work_temp_K=work_temp,
                          friction_factor=1.0)
        model = _model_totals(work, tool, ref_group, state, nodes)
        if isinstance(entry, KienzleEntry):
            meas = np.zeros_like(model)
            meas[0] = entry.cutting_force_per_width(nodes)
            feed = entry.feed_force_per_width(nodes)
            passive = entry.passive_force_per_width(nodes)
            ratio_t = meas[0] / model[0]
            meas[1] = feed if feed is not None else model[1] * ratio_t
            meas[2] = passive if passive is not None else model[2] * ratio_t
            with np.errstate(divide="ignore", invalid="ignore"):
                scale = np.where(np.abs(model) > 0.0, meas / model, 1.0)
            edge = scale[:, 0]
        else:
            # fit the model's totals by F = Kc h + Ke over the calibration range, compare per part
            A = np.vstack([nodes, np.ones_like(nodes)]).T
            scale = np.ones((3, n_nodes))
            edge = np.ones(3)
            measured = ((entry.Ktc_Pa, entry.Kte_N_per_m), (entry.Krc_Pa, entry.Kre_N_per_m),
                        (entry.Kac_Pa, entry.Kae_N_per_m))
            for j in range(3):
                kc_m, ke_m = np.linalg.lstsq(A, model[j], rcond=None)[0]
                kc_meas, ke_meas = measured[j]
                s_c = kc_meas / kc_m if abs(kc_m) > 0.0 else 1.0
                s_e = ke_meas / ke_m if abs(ke_m) > 1e-9 * abs(kc_m) * h_hi else s_c
                if not (lo_clamp <= s_e <= hi_clamp):
                    s_e = s_c
                scale[j, :] = s_c
                edge[j] = s_e
        rows.append((float(speed) if speed is not None else math.nan, nodes, scale, edge, entry.src))
    clamped = any(np.any((r[2] < lo_clamp) | (r[2] > hi_clamp)) for r in rows)
    known = [r for r in rows if np.isfinite(r[0])]
    rows = sorted(known, key=lambda r: r[0]) if known else rows[:1]
    lo = min(r[1][0] for r in rows)
    hi = max(r[1][-1] for r in rows)
    nodes = np.exp(np.linspace(math.log(lo), math.log(hi), n_nodes))
    scale = np.stack([np.clip(np.vstack([np.interp(np.log(nodes), np.log(r[1]), r[2][j]) for j in range(3)]),
                              lo_clamp, hi_clamp) for r in rows], axis=1)
    edge = np.clip(np.stack([r[3] for r in rows], axis=1), lo_clamp, hi_clamp)
    speeds = np.array([r[0] if np.isfinite(r[0]) else 1.0 for r in rows])
    return Calibration(kind, speeds, nodes, scale, edge, clamped, tuple(r[4] for r in rows))


# ============================================================================ group coefficients
@dataclass
class GroupCoefficients:
    """Rows of one coefficient group over the h grid (fills cutting's ProcessCoefficients)."""

    h_grid_m: np.ndarray
    Kt_Pa: np.ndarray
    Kr_Pa: np.ndarray
    Ka_Pa: np.ndarray
    Kte_N_per_m: float
    Kre_N_per_m: float
    Kae_N_per_m: float
    shear_angle_rad: np.ndarray
    chip_ratio: np.ndarray
    friction_angle_rad: np.ndarray
    frac_shear_to_chip: np.ndarray
    frac_rake_to_tool: np.ndarray
    frac_flank_to_tool: np.ndarray
    power_frac_shear: np.ndarray        # share of the tangential force power dissipated in the primary zone
    power_frac_rake: np.ndarray         # ... at the rake face (secondary zone)
    power_frac_edge: np.ndarray         # ... by the edge (ploughing below the stagnation point); the three sum to 1
    T_shear_K: np.ndarray
    T_interface_K: np.ndarray
    T_chip_K: np.ndarray
    contact_length_m: np.ndarray
    sticking_fraction: np.ndarray
    interface_shear_Pa: np.ndarray
    interface_normal_Pa: np.ndarray
    shear_strain: np.ndarray
    shear_strain_rate_1_s: np.ndarray
    regime: np.ndarray                  # REGIME_* per h node
    h_min_m: float
    K_sp_N_per_m3: float
    K_sp_source: str
    v_min_m_s: float
    mu_eff: float
    flank_sigma0_Pa: float
    flank_tau0_Pa: float
    flank_land_star_m: float
    flank_sigma_mean_Pa: float          # at the group's VB
    flank_tau_mean_Pa: float
    plough_x: np.ndarray                # h / h_min nodes of the ploughing shape
    plough_shape_t: np.ndarray          # ploughing force / edge force below h_min
    plough_shape_r: np.ndarray
    calibration: str = "none"
    calibration_clamped: bool = False
    notes: list = field(default_factory=list)

    def forces_per_width(self, h_m) -> np.ndarray:
        """[3, n] total forces per unit width at h (table form, h-interpolation linear in log h)."""
        h = np.atleast_1d(np.asarray(h_m, dtype=float))
        out = np.zeros((3, h.shape[0]))
        for j, (row, ke) in enumerate(((self.Kt_Pa, self.Kte_N_per_m), (self.Kr_Pa, self.Kre_N_per_m),
                                       (self.Ka_Pa, self.Kae_N_per_m))):
            k = np.array([K.interp_log_h(self.h_grid_m, contiguous(row, np.float64), x) for x in h])
            out[j] = k * h + ke
        return out


def _defaults(h: np.ndarray, work_temp_K: float) -> dict:
    n = h.shape[0]
    zeros = np.zeros(n)
    return dict(h_grid_m=h, Kt_Pa=zeros.copy(), Kr_Pa=zeros.copy(), Ka_Pa=zeros.copy(), Kte_N_per_m=0.0,
                Kre_N_per_m=0.0, Kae_N_per_m=0.0, shear_angle_rad=zeros.copy(), chip_ratio=zeros.copy(),
                friction_angle_rad=zeros.copy(), frac_shear_to_chip=zeros.copy(), frac_rake_to_tool=zeros.copy(),
                frac_flank_to_tool=zeros.copy(), power_frac_shear=zeros.copy(), power_frac_rake=zeros.copy(),
                power_frac_edge=np.ones(n), T_shear_K=np.full(n, work_temp_K),
                T_interface_K=np.full(n, work_temp_K), T_chip_K=np.full(n, work_temp_K),
                contact_length_m=zeros.copy(), sticking_fraction=zeros.copy(), interface_shear_Pa=zeros.copy(),
                interface_normal_Pa=zeros.copy(), shear_strain=zeros.copy(), shear_strain_rate_1_s=zeros.copy(),
                regime=np.full(n, REGIME_NONE, np.int32), plough_x=np.linspace(0.0, 1.0, 2), plough_shape_t=np.ones(2),
                plough_shape_r=np.ones(2))


def _flank_partition(work, tool, state: SlowState, land_m: float, tau_land_Pa: float, work_surface_K) -> np.ndarray:
    """Share of flank / edge-rubbing heat entering the tool at each work-surface temperature."""
    surface = np.atleast_1d(np.asarray(work_surface_K, dtype=float))
    out = np.zeros(surface.shape[0])
    flank_temp = state.tool_flank_temp_K if state.tool_flank_temp_K is not None else state.tool_temp_K
    if land_m <= 0.0 or tool is None or flank_temp is None:
        return out
    width = state.contact_width_m or P.number(P.CONTACT, "default_contact_aspect") * land_m
    for k in range(surface.shape[0]):
        out[k] = H.flank_partition_to_tool(work, tool, land_m, width, state.speed_m_s, tau_land_Pa * state.speed_m_s,
                                           float(surface[k]), float(flank_temp), state.contact_time_s)
    return out


def _rubbing_group(work, tool, group, state, h, mu, h_min, k_ref, notes) -> GroupCoefficients:
    """Margins and other rubbing-only elements: no chip; the hot loop's indentation model with
    K_sp applies at every interference (h_min set to the top of the grid)."""
    depth = max(h_min, 1e-9)
    edge = C.edge_forces(group.edge_radius_m, depth, group.rake_rad, group.incl_rad, k_ref, mu)
    k_sp, src = C.indentation_coefficient(work, edge[abi.PL_FR], depth, group.edge_radius_m, group.rake_rad)
    sigma0, tau0 = C.flank_reference_stresses(k_ref, mu)
    s_mean, t_mean = C.flank_stresses(state.VB_m, sigma0, tau0)
    x, st, sr = C.ploughing_shape(group.edge_radius_m, depth, group.rake_rad, group.incl_rad, k_ref, mu)
    land = state.VB_m if state.VB_m > 0.0 else edge[abi.PL_CONTACT]
    tau_land = t_mean if state.VB_m > 0.0 else C.contact_stresses(k_ref, mu)[1]
    notes.append("rubbing-only group: forces from the indentation model (K_sp) in the hot loop")
    kw = _defaults(h, state.work_temp_K)
    kw.update(frac_flank_to_tool=_flank_partition(work, tool, state, land, tau_land,
                                                  np.full(h.shape[0], state.work_temp_K)),
              regime=np.full(h.shape[0], REGIME_PLOUGHING, np.int32), plough_x=x, plough_shape_t=st,
              plough_shape_r=sr)
    return GroupCoefficients(**kw, h_min_m=float(h[-1]), K_sp_N_per_m3=float(k_sp), K_sp_source=src,
                             v_min_m_s=C.min_shear_speed(), mu_eff=float(mu), flank_sigma0_Pa=float(sigma0),
                             flank_tau0_Pa=float(tau0), flank_land_star_m=P.number(P.CONTACT, "flank_plastic_land_m"),
                             flank_sigma_mean_Pa=float(s_mean), flank_tau_mean_Pa=float(t_mean), notes=notes)


def group_coefficients(work: WorkMaterial, tool: ToolSide | None, group: EdgeGroup, state: SlowState, h_grid_m,
                       pmap: abi.ProcessMap | None, *, bue_map: abi.ProcessMap | None = None,
                       calibration: Calibration | None = None) -> GroupCoefficients:
    """Coefficient rows of one group at the slow state (see the module doc for the convention).

    ``pmap`` must be built for the group's rake (``bue_map`` at the BUE rake, used in proportion
    to the BUE height); materials without a flow-stress law use the data-only model and need
    no map."""
    h = contiguous(np.asarray(h_grid_m, dtype=float), np.float64)
    n = h.shape[0]
    notes: list[str] = []
    mu = effective_mu(work, tool, state.friction_factor)
    h_min = C.min_chip_thickness(work, group.edge_radius_m)
    if work.jc is None:
        return _data_only(work, tool, group, state, h, mu, h_min, notes)
    if pmap is None:
        raise ValueError("a process map is required for a material with a flow-stress law")
    ref = lookup(pmap, max(h_min, pmap.t1_grid_m[0]), state.speed_m_s, state.tool_temp_K)[0]
    k_ref = ref[abi.OX_K_AB] if ref[abi.OX_K_AB] > 0.0 else _first_positive(pmap, abi.OX_K_AB)
    if group.kind == EDGE_MARGIN:
        return _rubbing_group(work, tool, group, state, h, mu, h_min, k_ref, notes)
    indentation_group = C.uses_indentation(group.kind, group.rake_rad)
    t1 = np.maximum(h - (0.0 if indentation_group else h_min), 0.0)
    rows = lookup(pmap, t1, state.speed_m_s, state.tool_temp_K)
    rake_rows = np.full(n, float(pmap.context[abi.CTX_RAKE_RAD]))
    if bue_map is not None and state.bue_height_m > 0.0:
        rows_b = lookup(bue_map, t1, state.speed_m_s, state.tool_temp_K)
        ratio = P.number(P.CONTACT, "bue_full_height_ratio")
        w = np.clip(state.bue_height_m / (ratio * np.maximum(h, 1e-300)), 0.0, 1.0)
        for q in range(abi.N_OUT):
            if q in abi.DISCRETE_SLOTS:
                rows[:, q] = np.where(w > 0.5, rows_b[:, q], rows[:, q])
            else:
                rows[:, q] = (1.0 - w) * rows[:, q] + w * rows_b[:, q]
        rake_rows = (1.0 - w) * rake_rows + w * float(bue_map.context[abi.CTX_RAKE_RAD])
        notes.append(f"BUE height {state.bue_height_m * 1e6:.1f} um: rake blended toward the BUE rake")
    status = rows[:, abi.OX_STATUS]
    shearing = (t1 > 0.0) & ((status == abi.STATUS_OXLEY) | (status == abi.STATUS_MERCHANT))
    f_sh = _oblique_shearing_forces(rows, t1, shearing, rake_rows, group.incl_rad)
    regime = np.where(shearing, np.where(status == abi.STATUS_OXLEY, REGIME_OXLEY, REGIME_MERCHANT),
                      REGIME_NONE).astype(np.int32)
    if indentation_group or np.any((t1 > 0.0) & ~shearing):
        for k in range(n):
            if indentation_group or (t1[k] > 0.0 and not shearing[k]):
                f = C.indentation_forces(h[k], group.edge_radius_m, h_min, group.rake_rad, group.incl_rad, k_ref, mu)
                f_sh[:, k] = (f[abi.PL_FT], f[abi.PL_FR], f[abi.PL_FA])
                regime[k] = REGIME_INDENTATION
        if indentation_group:
            notes.append("indentation/extrusion group (chisel edge or normal rake below the shear limit)")
    if indentation_group or h_min <= 0.0 or group.edge_radius_m <= 0.0:
        edge = np.zeros(abi.N_PL)
    else:
        edge = C.edge_forces(group.edge_radius_m, h_min, group.rake_rad, group.incl_rad, k_ref, mu)
    ke = np.array([edge[abi.PL_FT], edge[abi.PL_FR], edge[abi.PL_FA]])
    below = h <= h_min
    if np.any(below) and not indentation_group:
        for k in np.nonzero(below)[0]:
            f = C.ploughing_forces(h[k], group.edge_radius_m, h_min, group.rake_rad, group.incl_rad, k_ref, mu)
            f_sh[:, k] = (f[abi.PL_FT] - ke[0], f[abi.PL_FR] - ke[1], f[abi.PL_FA] - ke[2])
            regime[k] = REGIME_PLOUGHING
    total = f_sh + ke[:, None]
    edge_part = np.broadcast_to(ke[:, None], total.shape).copy()
    edge_part[:, h <= h_min] = total[:, h <= h_min]
    cal_name = "none"
    clamped = False
    if calibration is not None and calibration.kind != "none":
        scale, edge_scale = calibration.factors(h, state.speed_m_s)
        if calibration.kind == "kienzle":
            total = total * scale
            edge_part = edge_part * scale
            ke_cal = ke * scale[:, int(np.argmax(h > h_min))] if np.any(h > h_min) else ke * scale[:, -1]
        else:
            total = f_sh * scale + (ke * edge_scale)[:, None]
            edge_part = np.where(h > h_min, (ke * edge_scale)[:, None], total)
            ke_cal = ke * edge_scale
        ke = ke_cal
        cal_name = calibration.kind
        clamped = calibration.clamped
        notes.append(f"calibrated to {calibration.kind} data ({', '.join(calibration.sources)})")
    with np.errstate(divide="ignore", invalid="ignore"):
        kt = np.where(h > 0.0, (total[0] - ke[0]) / h, 0.0)
        kr = np.where(h > 0.0, (total[1] - ke[1]) / h, 0.0)
        ka = np.where(h > 0.0, (total[2] - ke[2]) / h, 0.0)
    # heat partition and temperatures
    beta = np.where(shearing, rows[:, abi.OX_BETA], 0.0)
    share_chip = np.where(shearing, rows[:, abi.OX_R_CHIP], 0.0)
    p_shear = np.where(shearing, rows[:, abi.OX_P_SHEAR], 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        edge_share = np.where(total[0] > 0.0, np.clip(edge_part[0] / total[0], 0.0, 1.0), 1.0)
    edge_share = np.where(shearing, edge_share, 1.0)
    t_shear = np.where(shearing, rows[:, abi.OX_T_AB], state.work_temp_K)
    t_int = np.where(shearing, rows[:, abi.OX_T_INT], state.work_temp_K)
    t_chip = np.where(shearing, rows[:, abi.OX_T_CHIP], state.work_temp_K)
    sigma0, tau0 = C.flank_reference_stresses(k_ref, mu)
    s_mean, t_mean = C.flank_stresses(state.VB_m, sigma0, tau0)
    land = state.VB_m if state.VB_m > 0.0 else edge[abi.PL_CONTACT]
    tau_land = t_mean if state.VB_m > 0.0 else C.contact_stresses(k_ref, mu)[1]
    flank_tool = _flank_partition(work, tool, state, land, tau_land, t_shear)
    k_sp, k_src = C.indentation_coefficient(work, ke[1] if ke[1] > 0 else edge[abi.PL_FR], max(h_min, 1e-12),
                                            group.edge_radius_m, group.rake_rad)
    if h_min > 0.0 and group.edge_radius_m > 0.0:
        px, pst, psr = C.ploughing_shape(group.edge_radius_m, h_min, group.rake_rad, group.incl_rad, k_ref, mu)
    else:
        px, pst, psr = np.linspace(0.0, 1.0, 2), np.ones(2), np.ones(2)
    return GroupCoefficients(
        h_grid_m=h, Kt_Pa=kt, Kr_Pa=kr, Ka_Pa=ka, Kte_N_per_m=float(ke[0]), Kre_N_per_m=float(ke[1]),
        Kae_N_per_m=float(ke[2]),
        shear_angle_rad=np.where(shearing, rows[:, abi.OX_PHI], 0.0),
        chip_ratio=np.where(shearing, rows[:, abi.OX_CHIP_RATIO], 0.0),
        friction_angle_rad=np.where(shearing, rows[:, abi.OX_LAMBDA], 0.0),
        frac_shear_to_chip=np.where(shearing, 1.0 - beta, 0.0), frac_rake_to_tool=np.where(shearing, 1.0 - share_chip,
                                                                                            0.0),
        frac_flank_to_tool=flank_tool, power_frac_shear=(1.0 - edge_share) * p_shear,
        power_frac_rake=(1.0 - edge_share) * (1.0 - p_shear), power_frac_edge=edge_share, T_shear_K=t_shear,
        T_interface_K=t_int,
        T_chip_K=t_chip, contact_length_m=np.where(shearing, rows[:, abi.OX_LC], 0.0),
        sticking_fraction=np.where(shearing, rows[:, abi.OX_STICK], 0.0),
        interface_shear_Pa=np.where(shearing, rows[:, abi.OX_TAU_INT], 0.0),
        interface_normal_Pa=np.where(shearing, rows[:, abi.OX_SIGMA_N], 0.0),
        shear_strain=np.where(shearing, 2.0 * math.sqrt(3.0) * rows[:, abi.OX_EPS_AB], 0.0),
        shear_strain_rate_1_s=np.where(shearing, math.sqrt(3.0) * rows[:, abi.OX_EPSDOT_AB], 0.0),
        regime=regime, h_min_m=float(h_min), K_sp_N_per_m3=float(k_sp), K_sp_source=k_src,
        v_min_m_s=C.min_shear_speed(), mu_eff=float(mu), flank_sigma0_Pa=float(sigma0), flank_tau0_Pa=float(tau0),
        flank_land_star_m=P.number(P.CONTACT, "flank_plastic_land_m"), flank_sigma_mean_Pa=float(s_mean),
        flank_tau_mean_Pa=float(t_mean), plough_x=px, plough_shape_t=pst, plough_shape_r=psr, calibration=cal_name,
        calibration_clamped=clamped, notes=notes)


def _oblique_shearing_forces(rows: np.ndarray, t1: np.ndarray, shearing: np.ndarray, rake_rows: np.ndarray,
                             incl_rad: float) -> np.ndarray:
    """Shearing forces per unit width [3, n] in the (t, r, a) frame: the interpolated orthogonal
    forces of the map times the ratio of the oblique to the orthogonal Budak-Altintas-Armarego
    coefficients at the row's (tau_s, phi, lambda, rake), so that i = 0 reproduces the map exactly."""
    tau = np.where(shearing, rows[:, abi.OX_K_AB], 1.0)
    phi = np.where(shearing, rows[:, abi.OX_PHI], 0.5)
    lam = np.where(shearing, rows[:, abi.OX_LAMBDA], 0.5)
    out = np.zeros((3, t1.shape[0]))
    for k in np.nonzero(shearing)[0]:
        kt0, kr0, _ = O.cutting_coefficients(tau[k], phi[k], lam[k], float(rake_rows[k]), 0.0)
        kti, kri, kai = O.cutting_coefficients(tau[k], phi[k], lam[k], float(rake_rows[k]), incl_rad)
        fc = rows[k, abi.OX_FC]
        ft = rows[k, abi.OX_FT]
        s_t = fc / (float(kt0) * t1[k]) if float(kt0) != 0.0 else 1.0
        s_r = ft / (float(kr0) * t1[k]) if abs(float(kr0)) > 1e-9 * abs(float(kt0)) else s_t
        out[0, k] = s_t * float(kti) * t1[k]
        out[1, k] = s_r * float(kri) * t1[k]
        out[2, k] = s_t * float(kai) * t1[k]
    return out


def _first_positive(pmap: abi.ProcessMap, slot: int) -> float:
    vals = pmap.values[..., slot]
    pos = vals[vals > 0.0]
    return float(pos[0]) if pos.size else 0.0


# ============================================================================ data-only model
def _data_only(work, tool, group, state, h, mu, h_min, notes) -> GroupCoefficients:
    """Materials without a flow-stress law (polymers, wood, foam, composites): forces from the
    best Kienzle/mechanistic entry; shear angle from Merchant with the friction angle atan(mu);
    temperatures from the shear-plane power with Boothroyd's partition and Oxley's chip
    temperature fit at the fallback delta; the flow stress for edge/flank pressure is the
    yield strength softened linearly to the material's melt/softening temperature."""
    from .materials import flow_stress_fallback
    n = h.shape[0]
    kind, entries = select_entries(work, tool, tool.process if tool is not None else "milling")
    notes.append(f"data-only model ({kind})")
    lam = math.atan(mu) if mu > 0.0 else 0.0
    alpha = group.rake_rad
    phi = max(C.shear_angle_merchant(lam, alpha), P.number(P.CONTACT, "min_merchant_phi_rad"))
    t1 = np.maximum(h - h_min, 0.0)
    if kind == "kienzle":
        entry = entries[0]
        ft = entry.cutting_force_per_width(np.maximum(t1, 1e-300))
        fr = entry.feed_force_per_width(np.maximum(t1, 1e-300))
        if fr is None:
            fr = ft * math.tan(lam - alpha)
        fa = np.zeros(n)
        ke = np.zeros(3)
    elif kind == "mechanistic":
        entry = entries[0]
        ft = entry.Ktc_Pa * t1
        fr = entry.Krc_Pa * t1
        fa = entry.Kac_Pa * t1
        ke = np.array([entry.Kte_N_per_m, entry.Kre_N_per_m, entry.Kae_N_per_m])
    else:
        raise ValueError(f"material {work.id!r} has neither a flow-stress law nor force data")
    ft = np.where(t1 > 0.0, ft, 0.0)
    fr = np.where(t1 > 0.0, fr, 0.0)
    total = np.vstack([ft, fr, fa]) + ke[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        kt = np.where(h > 0.0, (total[0] - ke[0]) / h, 0.0)
        kr = np.where(h > 0.0, (total[1] - ke[1]) / h, 0.0)
        ka = np.where(h > 0.0, (total[2] - ke[2]) / h, 0.0)
    tw = state.work_temp_K
    v = state.speed_m_s
    r_c = math.sin(phi) / math.cos(phi - alpha)
    prm = P.get(P.OXLEY)
    t_shear = np.full(n, tw)
    t_int = np.full(n, tw)
    t_chip = np.full(n, tw)
    beta_row = np.zeros(n)
    p_shear = np.zeros(n)
    contact = np.zeros(n)
    delta = P.number(P.CONTACT, "merchant_delta")
    for k in range(n):
        if t1[k] <= 0.0 or ft[k] <= 0.0:
            continue
        theta = phi + lam - alpha
        resultant = ft[k] / math.cos(lam - alpha)
        fs = resultant * math.cos(theta)
        vs = v * math.cos(alpha) / math.cos(phi - alpha)
        vc = v * r_c
        friction = resultant * math.sin(lam)
        p_shear[k] = fs * vs / (ft[k] * v)
        rho = work.rho_kg_m3
        cp = work.cp(tw)
        kth = work.conductivity(tw)
        rt = rho * cp * v * t1[k] / kth
        beta = float(K.beta_partition(rt * math.tan(phi), P.oxley_vector(-1.0, 0.5)))
        dtsz = (1.0 - beta) * fs * vs / (rho * v * t1[k] * cp)
        dtc = friction * vc / (rho * v * t1[k] * cp)
        t2 = t1[k] / r_c
        lc = t1[k] * math.sin(theta) / (math.cos(lam) * math.sin(phi))
        x = rt * t2 / lc
        dtm = max(dtc, dtc * 10.0 ** (prm["dTM_a"] - prm["dTM_b"] * delta * math.sqrt(x)) * x ** prm["dTM_c"])
        beta_row[k] = beta
        contact[k] = lc
        t_shear[k] = tw + prm["eta_shear_plane_temp"] * dtsz
        t_int[k] = tw + dtsz + prm["psi_interface_temp"] * dtm
        t_chip[k] = tw + dtsz + dtc
    k_ref = flow_stress_fallback(work, float(np.max(t_shear))) / math.sqrt(3.0)
    sigma0, tau0 = C.flank_reference_stresses(k_ref, mu)
    s_mean, t_mean = C.flank_stresses(state.VB_m, sigma0, tau0)
    edge = C.edge_forces(group.edge_radius_m, h_min, alpha, group.incl_rad, k_ref, mu) if h_min > 0 else np.zeros(
        abi.N_PL)
    k_sp, k_src = C.indentation_coefficient(work, edge[abi.PL_FR], max(h_min, 1e-12), group.edge_radius_m, alpha)
    shearing = t1 > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        edge_share = np.where(total[0] > 0.0, np.clip(ke[0] / total[0], 0.0, 1.0), 1.0)
    kw = _defaults(h, tw)
    kw.update(Kt_Pa=kt, Kr_Pa=kr, Ka_Pa=ka, Kte_N_per_m=float(ke[0]), Kre_N_per_m=float(ke[1]),
              Kae_N_per_m=float(ke[2]), shear_angle_rad=np.where(shearing, phi, 0.0),
              chip_ratio=np.where(shearing, r_c, 0.0), friction_angle_rad=np.where(shearing, lam, 0.0),
              frac_shear_to_chip=np.where(shearing, 1.0 - beta_row, 0.0),
              power_frac_shear=np.where(shearing, (1.0 - edge_share) * p_shear, 0.0),
              power_frac_rake=np.where(shearing, (1.0 - edge_share) * (1.0 - p_shear), 0.0),
              power_frac_edge=np.where(shearing, edge_share, 1.0),
              T_shear_K=t_shear, T_interface_K=t_int, T_chip_K=t_chip, contact_length_m=contact,
              regime=np.where(shearing, REGIME_DATA, REGIME_PLOUGHING).astype(np.int32))
    return GroupCoefficients(**kw, h_min_m=float(h_min), K_sp_N_per_m3=float(k_sp), K_sp_source=k_src,
                             v_min_m_s=C.min_shear_speed(), mu_eff=float(mu), flank_sigma0_Pa=float(sigma0),
                             flank_tau0_Pa=float(tau0), flank_land_star_m=P.number(P.CONTACT, "flank_plastic_land_m"),
                             flank_sigma_mean_Pa=float(s_mean), flank_tau_mean_Pa=float(t_mean), calibration=kind,
                             notes=notes)


# ============================================================================ assembly
def process_coefficients(groups: list[GroupCoefficients], *, tool_slot: int, version: int,
                         flank_coeff_N_per_m2: np.ndarray | None = None) -> ProcessCoefficients:
    """ProcessCoefficients (bus record of cutting) from the groups of one tool; all groups share
    the h grid. ``flank_coeff_N_per_m2`` [n_teeth, n_elem] comes from :func:`flank_coefficients`."""
    if not groups:
        raise ValueError("no coefficient groups")
    h = groups[0].h_grid_m
    for g in groups:
        if g.h_grid_m.shape != h.shape or not np.array_equal(g.h_grid_m, h):
            raise ValueError("coefficient groups must share the h grid")

    def stack(name):
        return np.vstack([getattr(g, name) for g in groups])

    return ProcessCoefficients(
        tool_slot=tool_slot, h_grid_m=h.copy(), Kt_Pa=stack("Kt_Pa"), Kr_Pa=stack("Kr_Pa"), Ka_Pa=stack("Ka_Pa"),
        Kte_N_per_m=np.array([g.Kte_N_per_m for g in groups]), Kre_N_per_m=np.array([g.Kre_N_per_m for g in groups]),
        Kae_N_per_m=np.array([g.Kae_N_per_m for g in groups]), shear_angle_rad=stack("shear_angle_rad"),
        chip_ratio=stack("chip_ratio"), frac_shear_to_chip=stack("frac_shear_to_chip"),
        frac_rake_to_tool=stack("frac_rake_to_tool"), frac_flank_to_tool=stack("frac_flank_to_tool"),
        T_shear_K=stack("T_shear_K"), T_interface_K=stack("T_interface_K"),
        h_min_m=np.array([g.h_min_m for g in groups]), v_min_m_s=float(min(g.v_min_m_s for g in groups)),
        K_sp_N_per_m3=np.array([g.K_sp_N_per_m3 for g in groups]),
        flank_coeff_N_per_m2=np.zeros((0, 0)) if flank_coeff_N_per_m2 is None else flank_coeff_N_per_m2,
        version=version)


def coeff_table(pc: ProcessCoefficients) -> CoeffTable:
    """Kernel form (core.abi.CoeffTable) of a ProcessCoefficients record."""
    return CoeffTable(
        h_grid_m=contiguous(pc.h_grid_m, np.float64),
        K_Pa=contiguous(np.stack([pc.Kt_Pa, pc.Kr_Pa, pc.Ka_Pa], axis=-1), np.float64),
        Ke_N_per_m=contiguous(np.stack([pc.Kte_N_per_m, pc.Kre_N_per_m, pc.Kae_N_per_m], axis=-1), np.float64),
        frac=contiguous(np.stack([pc.frac_shear_to_chip, pc.frac_rake_to_tool, pc.frac_flank_to_tool], axis=-1),
                        np.float64),
        h_min_m=contiguous(pc.h_min_m, np.float64), K_sp_N_per_m3=contiguous(pc.K_sp_N_per_m3, np.float64),
        flank_coeff_N_per_m2=contiguous(pc.flank_coeff_N_per_m2, np.float64),
        v_min_m_s=np.array([pc.v_min_m_s], dtype=np.float64))


def flank_coefficients(groups: list[GroupCoefficients], elem_group: np.ndarray, VB_m: np.ndarray) -> np.ndarray:
    """Mean wear-land normal stress (N/m2 = force per unit land area) per element [n_teeth, n_elem]
    from each element's VB and its group's flank reference stresses (Smithey et al. 2001)."""
    out = np.zeros(np.shape(VB_m))
    for g_index, g in enumerate(groups):
        mask = np.asarray(elem_group) == g_index
        if np.any(mask):
            sigma, _ = C.flank_stresses(np.asarray(VB_m)[mask], g.flank_sigma0_Pa, g.flank_tau0_Pa,
                                        g.flank_land_star_m)
            out[mask] = sigma
    return out


__all__ = ["EdgeGroup", "SlowState", "GroupCoefficients", "Calibration", "NO_CALIBRATION", "build_process_map",
           "map_matches", "lookup", "build_calibration", "select_entries", "group_coefficients",
           "process_coefficients", "coeff_table", "flank_coefficients", "default_t1_grid", "default_speed_grid",
           "default_tool_temp_grid", "REGIME_NONE", "REGIME_OXLEY", "REGIME_MERCHANT", "REGIME_INDENTATION",
           "REGIME_PLOUGHING", "REGIME_DATA"]
