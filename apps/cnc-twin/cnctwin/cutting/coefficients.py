"""Coefficient pipeline of the cutting domain (L2, once per macro step per tool; docs/models/cutting.md
"Coefficient pipeline").

For every table row of a tool (one per coefficient group of tooling's GroupTable, plus a low-speed
indentation variant of a group whose elements run below v_min, and the forming rows of thread-forming
lobes) the pipeline

1. collects the row's **slow state** from bus_prev: local cutting speed (spindle speed x mean element
   radius, or the work speed at the contact radius when turning), tool rake and flank temperature
   (ThermalFeedback.tools), workpiece preheat (contact / workpiece), coolant friction factor, VB and
   BUE height (WearFeedback.tools) and the worn edge radius (the effective EdgeModel);
2. asks a :class:`CoefficientModel` for the row's coefficients over a log-spaced h grid
   (:class:`MechanicsModel`: cnctwin.mechanics - Oxley / Johnson-Cook process maps, the oblique
   transformation, edge, ploughing, indentation and flank models, heat partition, calibration to the
   material's Kienzle or mechanistic data; :class:`LinearModel`: constant coefficients for the analytic
   checks), recomputing a row only when its slow state left the rebuild tolerances;
3. linearises the rows into :class:`cnctwin.cutting.abi.CutTable` (two uniform h grids per row,
   intercept and slope per segment) for the hot loop, and assembles the bus record
   ProcessCoefficients (cnctwin.mechanics.process_coefficients) and its core CoeffTable.

Force convention (cnctwin.mechanics): F_j(h) = K_j(h) h db + Ke_j dS for h >= h_min; below h_min the
row holds the ploughing force, which the table applies per unit edge length (dS).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np

from ..core.abi import EDGE_CHISEL, EDGE_MARGIN, EDGE_PERIPHERAL, CoeffTable
from ..core.bus import ProcessCoefficients
from ..units import REFERENCE_TEMP_K
from .abi import (COL_EA, COL_ER, COL_ET, COL_FA, COL_FR, COL_FT, COL_PR, COL_PS, COL_RC, COL_SIG_RAKE, COL_TCHIP,
                  COL_WF, COL_WS, N_COL, CutTable)


# ============================================================================ rows
@dataclass
class Rows:
    """Coefficients of one table row over a log-spaced h grid (cnctwin.mechanics GroupCoefficients form,
    with only the fields the cutting domain uses)."""

    h_grid_m: np.ndarray
    Kt_Pa: np.ndarray
    Kr_Pa: np.ndarray
    Ka_Pa: np.ndarray
    Kte_N_per_m: float
    Kre_N_per_m: float
    Kae_N_per_m: float
    h_min_m: float
    power_frac_shear: np.ndarray
    power_frac_rake: np.ndarray
    interface_normal_Pa: np.ndarray
    chip_ratio: np.ndarray
    T_chip_K: np.ndarray
    frac_shear_to_chip: np.ndarray
    frac_rake_to_tool: np.ndarray
    frac_flank_to_tool: np.ndarray
    shear_angle_rad: np.ndarray
    T_shear_K: np.ndarray
    T_interface_K: np.ndarray
    K_sp_N_per_m3: float
    v_min_m_s: float
    mu_eff: float
    flank_sigma0_Pa: float
    flank_tau0_Pa: float
    flank_land_star_m: float
    rubbing: bool = False                # margins: forces from K_sp x interference area at every h
    edge_radius_m: float = 0.0
    forming: bool = False
    plough_edge: np.ndarray | None = None   # [3, n] explicit per-length force below h_min (forming / rubbing rows)
    source: str = ""
    notes: list = field(default_factory=list)

    @classmethod
    def from_group(cls, g, edge_radius_m: float = 0.0, source: str = "mechanics") -> "Rows":
        """From a cnctwin.mechanics GroupCoefficients."""
        rubbing = bool(np.all(np.asarray(g.regime) == 4)) and g.h_min_m >= float(np.asarray(g.h_grid_m)[-1])
        return cls(h_grid_m=np.asarray(g.h_grid_m, float), Kt_Pa=np.asarray(g.Kt_Pa, float),
                   Kr_Pa=np.asarray(g.Kr_Pa, float), Ka_Pa=np.asarray(g.Ka_Pa, float),
                   Kte_N_per_m=float(g.Kte_N_per_m), Kre_N_per_m=float(g.Kre_N_per_m), Kae_N_per_m=float(g.Kae_N_per_m),
                   h_min_m=float(g.h_min_m), power_frac_shear=np.asarray(g.power_frac_shear, float),
                   power_frac_rake=np.asarray(g.power_frac_rake, float),
                   interface_normal_Pa=np.asarray(g.interface_normal_Pa, float), chip_ratio=np.asarray(g.chip_ratio, float),
                   T_chip_K=np.asarray(g.T_chip_K, float), frac_shear_to_chip=np.asarray(g.frac_shear_to_chip, float),
                   frac_rake_to_tool=np.asarray(g.frac_rake_to_tool, float),
                   frac_flank_to_tool=np.asarray(g.frac_flank_to_tool, float),
                   shear_angle_rad=np.asarray(g.shear_angle_rad, float), T_shear_K=np.asarray(g.T_shear_K, float),
                   T_interface_K=np.asarray(g.T_interface_K, float), K_sp_N_per_m3=float(g.K_sp_N_per_m3),
                   v_min_m_s=float(g.v_min_m_s), mu_eff=float(g.mu_eff), flank_sigma0_Pa=float(g.flank_sigma0_Pa),
                   flank_tau0_Pa=float(g.flank_tau0_Pa), flank_land_star_m=float(g.flank_land_star_m),
                   rubbing=rubbing, edge_radius_m=float(edge_radius_m), source=source, notes=list(g.notes))


@dataclass(frozen=True)
class RowGeometry:
    """What a model needs to know about a table row (tooling GroupTable columns + variant)."""

    kind: int
    rake_rad: float
    incl_rad: float
    clear_rad: float
    edge_radius_m: float
    forming: bool = False
    low_speed: bool = False          # elements below v_min: indentation rows
    radius_m: float = 0.0            # mean element radius (forming lobes)
    lobes: int = 0                   # forming taps: lobe count
    diameter_m: float = 0.0


@dataclass(frozen=True)
class RowState:
    """Slow state of a row for the coming step (bus_prev), SI."""

    speed_m_s: float
    tool_temp_K: float | None = None
    flank_temp_K: float | None = None
    work_temp_K: float = REFERENCE_TEMP_K
    friction_factor: float = 1.0
    VB_m: float = 0.0
    bue_m: float = 0.0
    contact_width_m: float | None = None
    contact_time_s: float | None = None

    def as_list(self) -> list:
        return [self.speed_m_s, -1.0 if self.tool_temp_K is None else self.tool_temp_K,
                -1.0 if self.flank_temp_K is None else self.flank_temp_K, self.work_temp_K, self.friction_factor,
                self.VB_m, self.bue_m, -1.0 if self.contact_width_m is None else self.contact_width_m,
                -1.0 if self.contact_time_s is None else self.contact_time_s]

    @classmethod
    def from_list(cls, v) -> "RowState":
        return cls(speed_m_s=float(v[0]), tool_temp_K=None if v[1] < 0 else float(v[1]),
                   flank_temp_K=None if v[2] < 0 else float(v[2]), work_temp_K=float(v[3]),
                   friction_factor=float(v[4]), VB_m=float(v[5]), bue_m=float(v[6]),
                   contact_width_m=None if v[7] < 0 else float(v[7]), contact_time_s=None if v[8] < 0 else float(v[8]))


@dataclass(frozen=True)
class Tolerances:
    """Rebuild policy: a row is recomputed when its slow state moved beyond these (physics.cutting)."""

    speed_rel: float = 0.02
    temp_K: float = 10.0
    work_temp_K: float = 5.0
    friction_rel: float = 0.02
    VB_m: float = 2e-6
    bue_m: float = 1e-6
    edge_radius_m: float = 0.5e-6
    map_temp_K: float = 25.0
    map_mu_rel: float = 0.05
    map_rake_rad: float = 1e-3


def _snap_log(value: float, rel: float) -> float:
    if rel <= 0.0 or value <= 0.0:
        return float(value)
    step = math.log1p(rel)
    return float(math.exp(round(math.log(value) / step) * step))


def _snap(value: float, step: float) -> float:
    if step <= 0.0:
        return float(value)
    return float(round(value / step) * step)


CONTACT_QUANTUM_REL = 0.05      # relative grid of the engaged edge length and engagement time in a row's slow state


def quantised_state(state: RowState, tol: Tolerances) -> RowState:
    """The slow state snapped to the rebuild grid (docs/models/cutting.md "Coefficient pipeline"): speed and
    friction factor on log grids of the relative tolerances, temperatures, VB and BUE height on uniform grids of the
    absolute ones. A row is computed at its quantised state, so its coefficients are a function of the grid cell
    (and the model's map): the cell changes exactly when the state moved beyond a threshold, and equal cells give
    equal rows (cached)."""
    return RowState(speed_m_s=_snap_log(state.speed_m_s, tol.speed_rel),
                    tool_temp_K=None if state.tool_temp_K is None else _snap(state.tool_temp_K, tol.temp_K),
                    flank_temp_K=None if state.flank_temp_K is None else _snap(state.flank_temp_K, tol.temp_K),
                    work_temp_K=_snap(state.work_temp_K, tol.work_temp_K),
                    friction_factor=_snap_log(state.friction_factor, tol.friction_rel),
                    VB_m=max(_snap(state.VB_m, tol.VB_m), 0.0), bue_m=max(_snap(state.bue_m, tol.bue_m), 0.0),
                    contact_width_m=None if state.contact_width_m is None else
                    _snap_log(state.contact_width_m, CONTACT_QUANTUM_REL),
                    contact_time_s=None if state.contact_time_s is None else
                    _snap_log(state.contact_time_s, CONTACT_QUANTUM_REL))


def quantised_radius(radius_m: float, tol: Tolerances) -> float:
    return max(_snap(radius_m, tol.edge_radius_m), 0.0)


def state_changed(old: RowState | None, new: RowState, old_radius: float, new_radius: float,
                  tol: Tolerances) -> bool:
    if old is None:
        return True
    if abs(new.speed_m_s - old.speed_m_s) > tol.speed_rel * max(abs(old.speed_m_s), 1e-9):
        return True
    for a, b in ((old.tool_temp_K, new.tool_temp_K), (old.flank_temp_K, new.flank_temp_K)):
        if (a is None) != (b is None) or (a is not None and abs(a - b) > tol.temp_K):
            return True
    if abs(new.work_temp_K - old.work_temp_K) > tol.work_temp_K:
        return True
    if abs(new.friction_factor - old.friction_factor) > tol.friction_rel * max(abs(old.friction_factor), 1e-9):
        return True
    if abs(new.VB_m - old.VB_m) > tol.VB_m or abs(new.bue_m - old.bue_m) > tol.bue_m:
        return True
    return abs(new_radius - old_radius) > tol.edge_radius_m


class CoefficientModel(Protocol):
    """Source of a row's coefficients (cnctwin.mechanics by default; fixtures in tests)."""

    name: str

    def rows(self, key: tuple, geom: RowGeometry, state: RowState, h_grid_m: np.ndarray,
             provisional: bool = False) -> Rows:
        """Coefficients of one row at the slow state over the log h grid. ``provisional``: a cheaper approximation
        is acceptable (the tool is not cutting and its spindle speed is not settled); the pipeline replaces the
        row once the speed settles or the tool cuts."""

    def checkpoint(self) -> dict:
        """JSON metadata that makes the model's caches reproducible after a resume."""

    def restore(self, meta: dict) -> None:
        """Rebuild the caches' contexts from checkpoint metadata."""


# ============================================================================ forming lobes
def forming_rows(h_grid_m: np.ndarray, flow_stress_Pa: float, mu: float, radius_m: float, lobes: int,
                 diameter_m: float, params: dict, base: Rows | None = None) -> Rows:
    """Thread-forming lobe rows (docs/models/cutting.md "Forming taps"; Chowdhary et al. 2002, 2003):
    contact pressure p = (2 + pi) sigma / sqrt(3) on the lobe-workpiece contact of length
    l(h) = (1 + c_er) sqrt(2 rho_eff h), rho_eff = 2 r^2 / (e N^2) (lobe of relief e on N lobes in the
    formed bore of radius r; Johnson 1985), and per unit edge length

        F_r = p l(h),   F_t = p h + min(mu p, k) l(h),   F_a = 0.

    The frontal term p h makes the forming work per revolution equal to p times the displaced volume
    (the forming torque grows with the thread engagement, Agapiou 1994); the friction term acts on the
    contact area. Heat partition fractions come from ``base`` (a rubbing row of the same material)."""
    h = np.asarray(h_grid_m, float)
    k = flow_stress_Pa / math.sqrt(3.0)
    p = float(params["forming_pressure_factor"]) * k
    relief = float(params["forming_lobe_relief_per_D"]) * max(diameter_m, 2.0 * radius_m, 1e-6)
    n_lobes = max(int(lobes), 1)
    r = max(radius_m, 1e-6)
    rho_eff = 2.0 * r * r / (relief * n_lobes * n_lobes)
    factor = 1.0 + float(params["forming_recovery_contact_factor"])
    contact = factor * np.sqrt(2.0 * rho_eff * np.maximum(h, 0.0))
    # Coulomb on the lobe contact, limited by the shear flow stress; a negative mu means unknown friction:
    # sticking (tau = k)
    tau = min(mu * p, k) if mu >= 0.0 else k
    fr = p * contact
    ft = p * h + tau * contact
    with np.errstate(divide="ignore", invalid="ignore"):
        kt = np.where(h > 0.0, ft / h, 0.0)
        kr = np.where(h > 0.0, fr / h, 0.0)
    n = h.shape[0]
    zero = np.zeros(n)
    flank_tool = base.frac_flank_to_tool if base is not None else zero
    rows = Rows(h_grid_m=h, Kt_Pa=kt, Kr_Pa=kr, Ka_Pa=zero.copy(), Kte_N_per_m=0.0, Kre_N_per_m=0.0, Kae_N_per_m=0.0,
                h_min_m=0.0, power_frac_shear=zero.copy(), power_frac_rake=zero.copy(), interface_normal_Pa=zero.copy(),
                chip_ratio=zero.copy(), T_chip_K=np.full(n, REFERENCE_TEMP_K), frac_shear_to_chip=zero.copy(),
                frac_rake_to_tool=zero.copy(), frac_flank_to_tool=np.asarray(flank_tool, float).copy(),
                shear_angle_rad=zero.copy(), T_shear_K=np.full(n, REFERENCE_TEMP_K),
                T_interface_K=np.full(n, REFERENCE_TEMP_K), K_sp_N_per_m3=0.0,
                v_min_m_s=base.v_min_m_s if base is not None else 0.0, mu_eff=float(mu), flank_sigma0_Pa=p,
                flank_tau0_Pa=tau, flank_land_star_m=base.flank_land_star_m if base is not None else 1e-4,
                forming=True, source="forming", notes=[f"forming lobes: p = {p / 1e6:.0f} MPa, rho_eff = "
                                                      f"{rho_eff * 1e3:.1f} mm, relief {relief * 1e6:.1f} um"])
    rows.plough_edge = np.vstack([ft, fr, np.zeros(n)])
    return rows


def flow_stress_for_forming(work, strain: float) -> float:
    """Flow stress of the forming pressure: the Johnson-Cook strain term A + B eps^n at room temperature
    (no rate or thermal factor: thread forming is slow), else the yield strength (data-only materials)."""
    from ..mechanics.abi import MAT_A_PA, MAT_B_PA, MAT_N
    if work is not None and getattr(work, "jc", None) is not None:
        jc = work.jc
        return float(jc[MAT_A_PA] + jc[MAT_B_PA] * max(strain, 0.0) ** jc[MAT_N])
    value = getattr(work, "yield_Pa", None) or getattr(work, "ultimate_Pa", None)
    if not value:
        raise ValueError("forming taps need a flow-stress law or a yield strength of the work material")
    return float(value)


# ============================================================================ mechanics model
class MechanicsModel:
    """cnctwin.mechanics as the coefficient source: Oxley / Johnson-Cook process maps per row (rebuilt when
    the rake, workpiece temperature or friction leave the map tolerances or the speed leaves the map's
    speed grid), BUE maps, calibration to the material's measured data, the forming model for forming
    rows and indentation rows for low-speed variants."""

    name = "mechanics"

    def __init__(self, work, tool_side, process: str, tolerances: Tolerances | None = None,
                 contact_params: dict | None = None):
        self.work = work
        self.tool = tool_side
        self.process = process
        self.tol = tolerances or Tolerances()
        self.contact = contact_params
        self._maps: dict = {}           # key -> (context list, ProcessMap)
        self._bue_maps: dict = {}
        self._cal: dict = {}
        self.contexts: dict = {}        # key -> [rake, work_T, friction, speed, adiabatic, width, time] of the map

    def _mech(self):
        from .. import mechanics as M
        return M

    def _group(self, geom: RowGeometry):
        M = self._mech()
        kind = EDGE_CHISEL if geom.low_speed else (EDGE_MARGIN if geom.forming else geom.kind)
        rake = 0.0 if geom.forming else geom.rake_rad
        return M.EdgeGroup(rake_rad=float(rake), incl_rad=float(geom.incl_rad), clearance_rad=float(geom.clear_rad),
                           edge_radius_m=float(geom.edge_radius_m), kind=int(kind))

    def _context(self, group, state: RowState) -> list:
        return [float(group.rake_rad), float(state.work_temp_K), float(state.friction_factor), float(state.speed_m_s),
                1.0 if state.tool_temp_K is None else 0.0,
                -1.0 if state.contact_width_m is None else float(state.contact_width_m),
                -1.0 if state.contact_time_s is None else float(state.contact_time_s)]

    def _map_ok(self, ctx: list, pmap, group, state: RowState) -> bool:
        M = self._mech()
        from ..mechanics.oxley import effective_mu
        mu = effective_mu(self.work, self.tool, state.friction_factor)
        if not M.map_matches(pmap, group.rake_rad, state.work_temp_K, mu, rake_tol_rad=self.tol.map_rake_rad,
                             temp_tol_K=self.tol.map_temp_K, mu_rel_tol=self.tol.map_mu_rel):
            return False
        if (ctx[4] > 0.5) != (state.tool_temp_K is None):
            return False
        grid = pmap.speed_grid_m_s
        return grid[0] * (1.0 + 1e-9) <= max(state.speed_m_s, 1e-9) <= grid[-1] * (1.0 - 1e-9) or grid.shape[0] == 1

    def _build_map(self, ctx: list, rake_rad: float):
        M = self._mech()
        speed = max(ctx[3], 1e-6)
        # lazy: the fallback refinement of a (tool temperature, speed) row runs when a lookup first reads it (the
        # pipeline reads the two speed and temperature rows around the slow state; mechanics build_process_map)
        return M.build_process_map(self.work, self.tool, rake_rad, speed_grid_m_s=M.coefficients.default_speed_grid(speed),
                                   work_temp_K=ctx[1], friction_factor=ctx[2], adiabatic_tool=ctx[4] > 0.5,
                                   contact_width_m=None if ctx[5] < 0 else ctx[5],
                                   contact_time_s=None if ctx[6] < 0 else ctx[6], lazy=True)

    def process_map(self, key, group, state: RowState):
        cached = self._maps.get(key)
        if cached is not None and self._map_ok(cached[0], cached[1], group, state):
            return cached[1]
        ctx = self._context(group, state)
        pmap = self._build_map(ctx, group.rake_rad)
        self._maps[key] = (ctx, pmap)
        self.contexts[key] = ctx
        self._bue_maps.pop(key, None)
        return pmap

    def bue_map(self, key, group, state: RowState):
        if state.bue_m <= 0.0 or self.work.jc is None:
            return None
        cached = self._bue_maps.get(key)
        if cached is not None:
            return cached
        from ..mechanics import params as MP
        rake = MP.number(MP.CONTACT, "bue_rake_rad")
        ctx = self.contexts.get(key) or self._context(group, state)
        pmap = self._build_map(ctx, rake)
        self._bue_maps[key] = pmap
        return pmap

    def calibration(self, key, group):
        """The material's calibration for this pairing and process (cnctwin.mechanics.build_calibration). It depends
        on the group only through the rake an entry without a stated reference rake takes, so it is cached by that."""
        from ..mechanics.coefficients import select_entries, stated_rake
        from ..mechanics.contact import uses_indentation
        rake = float(group.rake_rad)
        _, entries = select_entries(self.work, self.tool, self.process)
        if all(stated_rake(e) is not None for e in entries):
            ckey = None
        else:
            ckey = round(rake, 6) if not uses_indentation(EDGE_PERIPHERAL, rake) else 0.0
        cal = self._cal.get(ckey)
        if cal is None:
            M = self._mech()
            cal = M.build_calibration(self.work, self.tool, group, self.process)
            self._cal[ckey] = cal
        return cal

    def rows(self, key: tuple, geom: RowGeometry, state: RowState, h_grid_m: np.ndarray,
             provisional: bool = False) -> Rows:
        """Rows from the process map of the row's context. ``provisional`` (a tool not cutting while its spindle
        speed is not settled): the map's fallback nodes that are not refined yet keep the kernel's own fallback (no
        continuation); the pipeline recomputes the row in full once the speed settles or the tool cuts."""
        M = self._mech()
        group = self._group(geom)
        mstate = M.SlowState(speed_m_s=max(float(state.speed_m_s), 1e-6), tool_temp_K=state.tool_temp_K,
                             tool_flank_temp_K=state.flank_temp_K, work_temp_K=state.work_temp_K,
                             friction_factor=state.friction_factor, VB_m=state.VB_m, bue_height_m=state.bue_m,
                             contact_width_m=state.contact_width_m, contact_time_s=state.contact_time_s)
        pmap = None if self.work.jc is None else self.process_map(key, group, state)
        bue = None if geom.forming or geom.low_speed else self.bue_map(key, group, state)
        if provisional:
            pmap = pmap._replace(refine=None) if pmap is not None and pmap.refine is not None else pmap
            bue = bue._replace(refine=None) if bue is not None and bue.refine is not None else bue
        # the calibration is a material response (flow stress, interface friction): every row but the forming lobes
        cal = None if geom.forming else self.calibration(key, group)
        g = M.group_coefficients(self.work, self.tool, group, mstate, h_grid_m, pmap, bue_map=bue, calibration=cal)
        base = Rows.from_group(g, geom.edge_radius_m)
        if not geom.forming:
            return base
        from ..mechanics.oxley import effective_mu
        params = self.contact if self.contact is not None else _contact_params()
        sigma = flow_stress_for_forming(self.work, float(params["forming_strain"]))
        mu = effective_mu(self.work, self.tool, state.friction_factor)
        return forming_rows(h_grid_m, sigma, mu, geom.radius_m, geom.lobes, geom.diameter_m, params, base)

    def checkpoint(self) -> dict:
        return {"contexts": [[list(k), list(v)] for k, v in sorted(self.contexts.items(), key=lambda kv: kv[0])]}

    def restore(self, meta: dict) -> None:
        """Maps are rebuilt from the stored contexts (deterministic), not from the current state, so a
        resumed run uses exactly the maps the uninterrupted run had."""
        for key, ctx in meta.get("contexts", []):
            key = tuple(key)
            self.contexts[key] = list(ctx)
            self._maps[key] = (list(ctx), self._build_map(list(ctx), ctx[0]))


def _contact_params() -> dict:
    from ..db.models import params
    return dict(params("cutting-contact"))


# ============================================================================ linear model (fixtures, analytic checks)
class LinearModel:
    """Constant coefficients per row (Altintas' linear edge-force model F = Kc h b + Ke S, or Kienzle
    F_c = kc11 h_ref (h / h_ref)^(1 - mc) b with feed / passive ratios): the analytic reference of the
    force kernels and a fixture for tests. ``coeffs`` maps a row kind (or 'default') to
    {Ktc, Krc, Kac (Pa), Kte, Kre, Kae (N/m), optional kc11 (Pa), mc, ratio_r, ratio_a, h_min_m,
    K_sp (N/m3), mu, sigma0 (Pa), power shares ps, pr, T_chip_K}."""

    name = "linear"

    def __init__(self, coeffs: dict, flow_stress_Pa: float = 500e6, contact_params: dict | None = None):
        self.coeffs = coeffs
        self.flow_stress_Pa = flow_stress_Pa
        self.contact = contact_params
        self.calls = 0

    def _c(self, geom: RowGeometry) -> dict:
        if geom.forming and "forming" in self.coeffs:
            return self.coeffs["forming"]
        if geom.low_speed and "low_speed" in self.coeffs:
            return self.coeffs["low_speed"]
        return self.coeffs.get(int(geom.kind), self.coeffs.get("default", {}))

    def rows(self, key: tuple, geom: RowGeometry, state: RowState, h_grid_m: np.ndarray,
             provisional: bool = False) -> Rows:
        self.calls += 1
        c = self._c(geom)
        h = np.asarray(h_grid_m, float)
        n = h.shape[0]
        h_min = float(c.get("h_min_m", 0.0))
        t1 = np.maximum(h - h_min, 0.0)
        if "kc11" in c:
            h_ref = float(c.get("h_ref_m", 1e-3))
            ft = float(c["kc11"]) * h_ref * (np.maximum(t1, 1e-300) / h_ref) ** (1.0 - float(c["mc"]))
            ft = np.where(t1 > 0.0, ft, 0.0)
            fr = ft * float(c.get("ratio_r", 0.0))
            fa = ft * float(c.get("ratio_a", 0.0))
        else:
            ft = float(c.get("Ktc", 0.0)) * t1
            fr = float(c.get("Krc", 0.0)) * t1
            fa = float(c.get("Kac", 0.0)) * t1
        ke = np.array([float(c.get("Kte", 0.0)), float(c.get("Kre", 0.0)), float(c.get("Kae", 0.0))])
        with np.errstate(divide="ignore", invalid="ignore"):
            kt = np.where(h > 0.0, ft / h, 0.0)
            kr = np.where(h > 0.0, fr / h, 0.0)
            ka = np.where(h > 0.0, fa / h, 0.0)
        below = h < h_min
        plough = np.vstack([np.where(below, ke[0] * h / max(h_min, 1e-300), 0.0),
                            np.where(below, ke[1] * h / max(h_min, 1e-300), 0.0),
                            np.where(below, ke[2] * h / max(h_min, 1e-300), 0.0)])
        ps = float(c.get("ps", 0.7))
        pr = float(c.get("pr", 0.25))
        rows = Rows(h_grid_m=h, Kt_Pa=kt, Kr_Pa=kr, Ka_Pa=ka, Kte_N_per_m=float(ke[0]), Kre_N_per_m=float(ke[1]),
                    Kae_N_per_m=float(ke[2]), h_min_m=h_min, power_frac_shear=np.where(below, 0.0, ps),
                    power_frac_rake=np.where(below, 0.0, pr), interface_normal_Pa=np.full(n, float(c.get("sigma_rake", 0.0))),
                    chip_ratio=np.where(below, 0.0, float(c.get("chip_ratio", 0.4))),
                    T_chip_K=np.full(n, float(c.get("T_chip_K", 600.0))),
                    frac_shear_to_chip=np.full(n, float(c.get("frac_shear_to_chip", 0.8))),
                    frac_rake_to_tool=np.full(n, float(c.get("frac_rake_to_tool", 0.2))),
                    frac_flank_to_tool=np.full(n, float(c.get("frac_flank_to_tool", 0.3))),
                    shear_angle_rad=np.full(n, float(c.get("phi", 0.4))), T_shear_K=np.full(n, 500.0),
                    T_interface_K=np.full(n, 800.0), K_sp_N_per_m3=float(c.get("K_sp", 0.0)),
                    v_min_m_s=float(c.get("v_min", 0.0)), mu_eff=float(c.get("mu", 0.5)),
                    flank_sigma0_Pa=float(c.get("sigma0", 0.0)), flank_tau0_Pa=float(c.get("tau0", 0.0)),
                    flank_land_star_m=float(c.get("VB_star", 1e-4)), edge_radius_m=geom.edge_radius_m,
                    source="linear")
        rows.plough_edge = plough
        if geom.forming:
            params = self.contact if self.contact is not None else _contact_params()
            return forming_rows(h, float(c.get("flow_stress", self.flow_stress_Pa)), float(c.get("mu", 0.2)),
                                geom.radius_m, geom.lobes, geom.diameter_m, params, rows)
        return rows

    def checkpoint(self) -> dict:
        return {}

    def restore(self, meta: dict) -> None:
        return None


# ============================================================================ linearisation
def log_grid(h_lo_m: float, h_hi_m: float, n: int) -> np.ndarray:
    return np.exp(np.linspace(math.log(h_lo_m), math.log(h_hi_m), int(n)))


def _interp_log(h_nodes: np.ndarray, grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Values linear in log h between the grid nodes, held constant outside (mechanics interp_log_h)."""
    lg = np.log(grid)
    lh = np.log(np.maximum(h_nodes, grid[0] * 1e-6))
    return np.interp(np.clip(lh, lg[0], lg[-1]), lg, values)


def _rubbing_forces(rows: Rows, h: np.ndarray) -> np.ndarray:
    """Margins and pads: radial force K_sp A(h) per unit edge length from the interference area of the
    rounded edge (mechanics.contact), tangential mu times it (Coulomb on the land)."""
    from ..mechanics.contact import interference_area
    radius = max(rows.edge_radius_m, 1e-9)
    area = np.array([interference_area(float(x), radius) if x > 0.0 else 0.0 for x in h])
    fr = rows.K_sp_N_per_m3 * area
    return np.vstack([rows.mu_eff * fr, fr, np.zeros_like(fr)])


def node_values(rows: Rows, h: np.ndarray) -> np.ndarray:
    """[n, N_COL] column values of a row at chip thicknesses h (shearing part per unit chip width, edge /
    ploughing part per unit edge length, shares and stresses)."""
    out = np.zeros((h.shape[0], N_COL))
    grid = rows.h_grid_m
    above = h >= rows.h_min_m
    kt = _interp_log(h, grid, rows.Kt_Pa)
    kr = _interp_log(h, grid, rows.Kr_Pa)
    ka = _interp_log(h, grid, rows.Ka_Pa)
    if rows.forming and rows.plough_edge is not None:
        pe = np.vstack([np.interp(h, grid, rows.plough_edge[j], left=0.0) for j in range(3)])
        # forming: the force is per unit edge length at every h (no chip, no minimum)
        out[:, COL_ET] = pe[0]
        out[:, COL_ER] = pe[1]
        out[:, COL_EA] = pe[2]
    elif rows.rubbing:
        pe = _rubbing_forces(rows, h)
        out[:, COL_ET] = pe[0]
        out[:, COL_ER] = pe[1]
        out[:, COL_EA] = pe[2]
    else:
        out[:, COL_FT] = np.where(above, kt * h, 0.0)
        out[:, COL_FR] = np.where(above, kr * h, 0.0)
        out[:, COL_FA] = np.where(above, ka * h, 0.0)
        if rows.plough_edge is not None:
            below_t = np.interp(h, grid, rows.plough_edge[0], left=0.0)
            below_r = np.interp(h, grid, rows.plough_edge[1], left=0.0)
            below_a = np.interp(h, grid, rows.plough_edge[2], left=0.0)
        else:
            below_t = kt * h + rows.Kte_N_per_m
            below_r = kr * h + rows.Kre_N_per_m
            below_a = ka * h + rows.Kae_N_per_m
        out[:, COL_ET] = np.where(above, rows.Kte_N_per_m, below_t)
        out[:, COL_ER] = np.where(above, rows.Kre_N_per_m, below_r)
        out[:, COL_EA] = np.where(above, rows.Kae_N_per_m, below_a)
    out[:, COL_PS] = np.clip(_interp_log(h, grid, rows.power_frac_shear), 0.0, 1.0)
    out[:, COL_PR] = np.clip(_interp_log(h, grid, rows.power_frac_rake), 0.0, 1.0)
    total = out[:, COL_PS] + out[:, COL_PR]
    over = total > 1.0
    out[over, COL_PS] /= total[over]
    out[over, COL_PR] /= total[over]
    out[:, COL_SIG_RAKE] = np.maximum(_interp_log(h, grid, rows.interface_normal_Pa), 0.0)
    out[:, COL_RC] = np.maximum(_interp_log(h, grid, rows.chip_ratio), 0.0)
    out[:, COL_TCHIP] = _interp_log(h, grid, rows.T_chip_K)
    out[:, COL_WS] = np.clip(1.0 - _interp_log(h, grid, rows.frac_shear_to_chip), 0.0, 1.0)
    out[:, COL_WF] = np.clip(1.0 - _interp_log(h, grid, rows.frac_flank_to_tool), 0.0, 1.0)
    return out


def row_nodes(h_min_m: float, h_hi_m: float, n_lo: int, n_hi: int) -> tuple[np.ndarray, np.ndarray]:
    """Node positions of the low [0, h_min] and high [h_min, h_hi] uniform grids of one row."""
    lo = np.linspace(0.0, h_min_m, n_lo + 1) if h_min_m > 0.0 else np.zeros(n_lo + 1)
    hi = np.linspace(h_min_m, max(h_hi_m, h_min_m * 1.01 + 1e-9), n_hi + 1)
    return lo, hi


def linearise_row(rows: Rows, h_hi_m: float, n_lo: int, n_hi: int) -> tuple[np.ndarray, float, float, float]:
    """(coef [n_lo + n_hi, 2, N_COL], h_min, inv_dlo, inv_dhi) of one row: per segment the line through
    the node values (exact at the nodes)."""
    h_min = 0.0 if (rows.forming or rows.rubbing) else float(rows.h_min_m)
    lo, hi = row_nodes(h_min, h_hi_m, n_lo, n_hi)
    coef = np.zeros((n_lo + n_hi, 2, N_COL))
    v_hi = node_values(rows, hi)
    # just above h_min the row follows the shearing branch; the node at h_min of the high grid uses it
    for m in range(n_hi):
        x0, x1 = hi[m], hi[m + 1]
        slope = (v_hi[m + 1] - v_hi[m]) / (x1 - x0)
        coef[n_lo + m, 1] = slope
        coef[n_lo + m, 0] = v_hi[m] - slope * x0
    if h_min > 0.0:
        # below h_min: the ploughing branch; its last node is the limit from below (continuous for mechanics rows)
        v_lo = node_values(rows, lo[:-1])
        below = node_values(replace(rows, h_min_m=np.inf if not rows.rubbing else rows.h_min_m),
                            np.array([h_min * (1.0 - 1e-12)]))[0]
        v_lo = np.vstack([v_lo, below])
        for k in range(n_lo):
            x0, x1 = lo[k], lo[k + 1]
            slope = (v_lo[k + 1] - v_lo[k]) / (x1 - x0)
            coef[k, 1] = slope
            coef[k, 0] = v_lo[k] - slope * x0
        inv_dlo = n_lo / h_min
    else:
        coef[:n_lo] = coef[n_lo]
        inv_dlo = 0.0
    inv_dhi = n_hi / (hi[-1] - hi[0])
    return coef, h_min, inv_dlo, inv_dhi


def build_table(row_list: list[Rows], h_hi_m: float, n_lo: int, n_hi: int, v_min_m_s: float) -> CutTable:
    """CutTable of a tool from its rows (same h_hi for every row)."""
    nr = max(len(row_list), 1)
    coef = np.zeros((nr, n_lo + n_hi, 2, N_COL))
    h_min = np.zeros(nr)
    inv_dlo = np.zeros(nr)
    inv_dhi = np.zeros(nr)
    h_hi = np.zeros(nr)
    forming = np.zeros(nr, np.int8)
    h_write = np.zeros(nr)
    for r, rows in enumerate(row_list):
        c, hm, il, ih = linearise_row(rows, h_hi_m, n_lo, n_hi)
        coef[r] = c
        h_min[r] = hm
        inv_dlo[r] = il
        inv_dhi[r] = ih
        h_hi[r] = max(h_hi_m, hm * 1.01 + 1e-9)
        forming[r] = 1 if rows.forming else 0
        h_write[r] = np.inf if rows.rubbing else (0.0 if rows.forming else float(rows.h_min_m))
    return CutTable(np.ascontiguousarray(coef), h_min, inv_dlo, inv_dhi, h_hi, forming, h_write,
                    np.array([n_lo], np.int64), np.array([float(v_min_m_s)]))


def table_forces(table: CutTable, row: int, h: float) -> np.ndarray:
    """Plain-Python evaluation of a CutTable row: [N_COL] column values at h (the kernels' rule)."""
    nlo = int(table.n_lo[0])
    hm = table.h_min_m[row]
    if h < hm:
        k = min(int(h * table.inv_dlo[row]), nlo - 1)
    else:
        k = min(nlo + int((h - hm) * table.inv_dhi[row]), table.coef.shape[1] - 1)
    return table.coef[row, k, 0] + table.coef[row, k, 1] * h


# ============================================================================ bus record
def _gc_view(rows: Rows):
    """An object with the GroupCoefficients fields mechanics.process_coefficients reads."""
    return rows


def process_record(row_list: list[Rows], tool_slot: int, version: int, flank_coeff: np.ndarray) -> ProcessCoefficients:
    """ProcessCoefficients (bus) from the base rows of a tool (one per coefficient group, in group order),
    all on the same log h grid."""
    if not row_list:
        raise ValueError("no coefficient rows")
    h = row_list[0].h_grid_m
    for r in row_list:
        if r.h_grid_m.shape != h.shape or not np.array_equal(r.h_grid_m, h):
            raise ValueError("coefficient rows must share the h grid")

    def stack(name):
        return np.vstack([np.asarray(getattr(r, name), float) for r in row_list])

    return ProcessCoefficients(
        tool_slot=tool_slot, h_grid_m=h.copy(), Kt_Pa=stack("Kt_Pa"), Kr_Pa=stack("Kr_Pa"), Ka_Pa=stack("Ka_Pa"),
        Kte_N_per_m=np.array([r.Kte_N_per_m for r in row_list]), Kre_N_per_m=np.array([r.Kre_N_per_m for r in row_list]),
        Kae_N_per_m=np.array([r.Kae_N_per_m for r in row_list]), shear_angle_rad=stack("shear_angle_rad"),
        chip_ratio=stack("chip_ratio"), frac_shear_to_chip=stack("frac_shear_to_chip"),
        frac_rake_to_tool=stack("frac_rake_to_tool"), frac_flank_to_tool=stack("frac_flank_to_tool"),
        T_shear_K=stack("T_shear_K"), T_interface_K=stack("T_interface_K"),
        h_min_m=np.array([r.h_min_m for r in row_list]), v_min_m_s=float(min(r.v_min_m_s for r in row_list)),
        K_sp_N_per_m3=np.array([r.K_sp_N_per_m3 for r in row_list]),
        flank_coeff_N_per_m2=np.ascontiguousarray(flank_coeff, float), version=int(version))


def coeff_table(pc: ProcessCoefficients) -> CoeffTable:
    """core.abi.CoeffTable of the bus record (cnctwin.mechanics.coeff_table)."""
    from ..mechanics import coeff_table as mech_coeff_table
    return mech_coeff_table(pc)


def flank_stress(rows: Rows, VB_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean wear-land normal and shear stress (Waldorf 1999; Smithey, Kapoor & DeVor 2000, 2001; as
    cnctwin.mechanics.contact.flank_stresses): elastic contact below the critical land VB* (stresses falling with the
    square of the distance from the edge values to zero at the land end, mean one third); beyond it a plastic region
    of width VB - VB* at the edge values ahead of an elastic region of the fixed width VB*."""
    vb = np.asarray(VB_m, float)
    star = rows.flank_land_star_m if rows.flank_land_star_m > 0.0 else 1e-4
    plastic = np.maximum(vb - star, 0.0)
    elastic = np.minimum(vb, star)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(vb > 0.0, (plastic + elastic / 3.0) / np.where(vb > 0.0, vb, 1.0), 0.0)
    return rows.flank_sigma0_Pa * factor, rows.flank_tau0_Pa * factor


__all__ = ["Rows", "RowGeometry", "RowState", "Tolerances", "CoefficientModel", "MechanicsModel", "LinearModel",
           "forming_rows", "flow_stress_for_forming", "state_changed", "quantised_state", "quantised_radius",
           "log_grid", "node_values", "linearise_row",
           "build_table", "table_forces", "process_record", "coeff_table", "flank_stress", "row_nodes"]
