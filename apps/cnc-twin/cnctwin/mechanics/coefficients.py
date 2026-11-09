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
from .oxley import (continuation_reference, effective_mu, interface_coulomb, interface_inputs, merchant_from,
                    zorev_split)

REGIME_NONE = 0
REGIME_OXLEY = 1
REGIME_MERCHANT = 2
REGIME_INDENTATION = 3
REGIME_PLOUGHING = 4
REGIME_DATA = 5

_SCALED = np.array(abi.SCALES_WITH_T1, dtype=np.int64)
SHIFT_MARGIN_RAD = 0.05          # keeps a calibrated friction angle off the poles of the oblique force relations
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
                      tool_temp_grid_K=None, adiabatic_tool: bool = False, coulomb_limit: bool | None = None,
                      contact_width_m: float | None = None, contact_time_s: float | None = None,
                      lazy: bool = False) -> abi.ProcessMap:
    """Oxley outputs on a (tool temperature, speed, t1) grid for one context (see module doc).

    ``adiabatic_tool`` (or no tool) gives a single tool-temperature node and Oxley's original
    assumption that all rake heat enters the chip. The interface condition follows
    :func:`cnctwin.mechanics.oxley.interface_coulomb` (Oxley's condition unless
    ``coulomb_limit=True``).

    ``lazy``: the fallback refinement of the nodes without an admissible Oxley solution (the expensive part: a
    continuation per node in a row without any admissible node) is done per (tool temperature, speed) row when
    :func:`lookup` first reads the row (:class:`LazyRefinement`); every node read has the value the eager map
    gives it (same raw row, same edges), and rows never read cost nothing."""
    t1 = contiguous(default_t1_grid() if t1_grid_m is None else t1_grid_m, np.float64)
    speeds = contiguous(speed_grid_m_s, np.float64)
    adiabatic = adiabatic_tool or tool is None
    if adiabatic:
        ttool = np.array([float(work_temp_K)])
    else:
        ttool = contiguous(default_tool_temp_grid(work_temp_K) if tool_temp_grid_K is None else tool_temp_grid_K,
                           np.float64)
    mu = effective_mu(work, tool, friction_factor)
    coulomb = interface_coulomb(coulomb_limit)
    inputs = interface_inputs(work, tool, tool_temp_K=None if adiabatic else float(ttool[0]),
                              friction_factor=friction_factor, coulomb_limit=coulomb_limit,
                              contact_width_m=contact_width_m, contact_time_s=contact_time_s)
    values = np.zeros((ttool.shape[0], speeds.shape[0], t1.shape[0], abi.N_OUT))
    K.oxley_map(t1, speeds, ttool, float(rake_rad), float(work_temp_K), *inputs, values)
    exponent = P.number(P.CONTACT, "zorev_exponent")
    refine = None
    if lazy:
        refine = LazyRefinement(t1, speeds, ttool, float(rake_rad), float(work_temp_K), inputs, adiabatic, mu,
                                exponent, values)
        for idx in np.ndindex(values.shape[:3]):
            vec = values[idx]
            if vec[abi.OX_STATUS] == abi.STATUS_OXLEY:
                vec[abi.OX_STICK] = zorev_split(vec[abi.OX_TAU_INT], vec[abi.OX_SIGMA_N], mu, exponent)[0]
    else:
        _refine_fallbacks(values, t1, speeds, ttool, float(rake_rad), float(work_temp_K), inputs, adiabatic)
        for idx in np.ndindex(values.shape[:3]):
            vec = values[idx]
            if vec[abi.OX_STATUS] in (abi.STATUS_OXLEY, abi.STATUS_MERCHANT):
                vec[abi.OX_STICK] = zorev_split(vec[abi.OX_TAU_INT], vec[abi.OX_SIGMA_N], mu, exponent)[0]
    context = np.zeros(abi.N_CTX)
    context[abi.CTX_RAKE_RAD] = rake_rad
    context[abi.CTX_WORK_TEMP_K] = work_temp_K
    context[abi.CTX_MU] = mu           # friction state (database mu x friction factor) in either interface mode
    context[abi.CTX_ADIABATIC] = 1.0 if adiabatic else 0.0
    context[abi.CTX_CONTACT_WIDTH_M] = contact_width_m or 0.0
    context[abi.CTX_CONTACT_TIME_S] = contact_time_s or 0.0
    context[abi.CTX_COULOMB] = 1.0 if coulomb else 0.0
    return abi.ProcessMap(t1, speeds, ttool, values, context, refine)


class LazyRefinement:
    """Fallback refinement of a lazily built process map, row by row (see build_process_map ``lazy``): the raw
    admissibility of every row is kept, so a row refined later gets exactly the values of the eager refinement."""

    def __init__(self, t1, speeds, ttool, rake_rad, work_temp_K, inputs, adiabatic, mu, exponent, values):
        self.t1 = t1
        self.speeds = speeds
        self.ttool = ttool
        self.rake_rad = rake_rad
        self.work_temp_K = work_temp_K
        self.inputs = inputs
        self.adiabatic = adiabatic
        self.mu = mu
        self.exponent = exponent
        self.done = np.zeros(values.shape[:2], bool)
        self.done |= np.all(values[..., abi.OX_STATUS] == abi.STATUS_OXLEY, axis=2)

    def rows(self, values, it_: int, iv: int) -> None:
        """Refine row (it_, iv) of ``values`` (the map's own array) if not done yet."""
        if self.done[it_, iv]:
            return
        row = values[it_:it_ + 1, iv:iv + 1]
        _refine_fallbacks(row, self.t1, self.speeds[iv:iv + 1], self.ttool[it_:it_ + 1], self.rake_rad,
                          self.work_temp_K, self.inputs, self.adiabatic)
        for ih in range(row.shape[2]):
            vec = row[0, 0, ih]
            if vec[abi.OX_STATUS] in (abi.STATUS_OXLEY, abi.STATUS_MERCHANT):
                vec[abi.OX_STICK] = zorev_split(vec[abi.OX_TAU_INT], vec[abi.OX_SIGMA_N], self.mu, self.exponent)[0]
        self.done[it_, iv] = True


def _bracket_log(grid: np.ndarray, x: float) -> tuple[int, int]:
    """The two nodes map_lookup interpolates between in log x (its index rule)."""
    n = grid.shape[0]
    if n <= 1:
        return 0, 0
    lx = math.log(x) if x > 0.0 else -1e300
    if lx <= math.log(grid[0]):
        return 0, 1
    if lx >= math.log(grid[n - 1]):
        return n - 2, n - 1
    i = 0
    while i < n - 2 and math.log(grid[i + 1]) <= lx:
        i += 1
    return i, i + 1


def _bracket_lin(grid: np.ndarray, x: float) -> tuple[int, int]:
    n = grid.shape[0]
    if n <= 1:
        return 0, 0
    if x <= grid[0]:
        return 0, 1
    if x >= grid[n - 1]:
        return n - 2, n - 1
    i = 0
    while i < n - 2 and grid[i + 1] <= x:
        i += 1
    return i, i + 1


def ensure_refined(pmap: abi.ProcessMap, speed_m_s: float | None = None, tool_temp_K: float | None = None) -> None:
    """Refine the rows of a lazy map that a lookup at (speed, tool temperature) reads (all rows when None)."""
    lz = pmap.refine
    if lz is None:
        return
    if speed_m_s is None:
        for it_ in range(pmap.values.shape[0]):
            for iv in range(pmap.values.shape[1]):
                lz.rows(pmap.values, it_, iv)
        return
    t_tool = float(pmap.Ttool_grid_K[0] if tool_temp_K is None else tool_temp_K)
    for it_ in sorted(set(_bracket_lin(pmap.Ttool_grid_K, t_tool))):
        for iv in sorted(set(_bracket_log(pmap.speed_grid_m_s, float(speed_m_s)))):
            lz.rows(pmap.values, it_, iv)


def _refine_fallbacks(values, t1, speeds, ttool, rake_rad, work_temp_K, inputs, adiabatic) -> None:
    """Replace the kernel's nearest-node fallback of the map nodes without an admissible Oxley
    solution by the fallback continued from the edge of the admissible region, located by
    bisection, so a node's value does not depend on the grid (as oxley.solve): in a row of
    t1 nodes (one speed and tool temperature) a block of such nodes is continued from the edge
    between it and its nearer admissible neighbour; a row without an admissible node from the
    continuation of each node (oxley.continuation_reference)."""
    n_bisect = int(P.get(P.OXLEY)["fallback_bisection_steps"])
    tool = inputs.tool.copy()
    node_inputs = inputs._replace(tool=tool)
    case = np.zeros(4)
    case[2] = rake_rad
    case[3] = work_temp_K
    out = np.zeros(abi.N_OUT)
    n_h = t1.shape[0]
    for it_ in range(ttool.shape[0]):
        if not adiabatic:
            tool[abi.TOOL_T_RAKE_K] = ttool[it_]
        for iv in range(speeds.shape[0]):
            case[1] = speeds[iv]
            ok = values[it_, iv, :, abi.OX_STATUS] == abi.STATUS_OXLEY
            if ok.all():
                continue
            edges: dict[tuple[int, int], np.ndarray] = {}
            for ih in range(n_h):
                if ok[ih]:
                    continue
                case[0] = t1[ih]
                left = next((j for j in range(ih - 1, -1, -1) if ok[j]), None)
                right = next((j for j in range(ih + 1, n_h) if ok[j]), None)
                if left is None and right is None:
                    # the row's own nodes are solved and inadmissible: thickness trials next to them are not repeated
                    found = continuation_reference(case, node_inputs, inadmissible_t1=t1)
                    if found is not None:
                        values[it_, iv, ih] = merchant_from(case, node_inputs, found[0])
                    continue
                if right is None or (left is not None and ih - left <= right - ih):
                    pair = (left + 1, left)
                else:
                    pair = (right - 1, right)
                if pair not in edges:
                    lo, hi = math.log(t1[pair[0]]), math.log(t1[pair[1]])     # lo inadmissible, hi admissible
                    best = values[it_, iv, pair[1]].copy()
                    trial = case.copy()
                    for _ in range(n_bisect):
                        mid = 0.5 * (lo + hi)
                        trial[0] = math.exp(mid)
                        K.oxley_point(trial, *node_inputs, out)
                        if out[abi.OX_STATUS] == abi.STATUS_OXLEY:
                            hi = mid
                            best = out.copy()
                        else:
                            lo = mid
                    edges[pair] = best
                values[it_, iv, ih] = merchant_from(case, node_inputs, edges[pair])


def map_matches(pmap: abi.ProcessMap, rake_rad: float, work_temp_K: float, mu: float, *, rake_tol_rad: float,
                temp_tol_K: float, mu_rel_tol: float, coulomb_limit: bool | None = None) -> bool:
    """True if a map was built for this context within the tolerances (cutting's rebuild policy).
    ``mu`` is the effective friction coefficient (database x friction factor), which identifies
    the friction state of both interface modes; ``coulomb_limit`` (when given) must select the
    map's interface mode."""
    ctx = pmap.context
    if abs(ctx[abi.CTX_RAKE_RAD] - rake_rad) > rake_tol_rad or abs(ctx[abi.CTX_WORK_TEMP_K] - work_temp_K) > temp_tol_K:
        return False
    if coulomb_limit is not None and (ctx[abi.CTX_COULOMB] > 0.5) != interface_coulomb(coulomb_limit):
        return False
    if ctx[abi.CTX_MU] <= 0.0 or mu <= 0.0:
        return ctx[abi.CTX_MU] <= 0.0 and mu <= 0.0
    return abs(ctx[abi.CTX_MU] - mu) <= mu_rel_tol * mu


def lookup(pmap: abi.ProcessMap, t1_m, speed_m_s: float, tool_temp_K: float | None) -> np.ndarray:
    """Output vectors (abi.OX_*) at each t1 for one speed and tool temperature: [n, N_OUT]."""
    query = contiguous(np.atleast_1d(np.asarray(t1_m, dtype=float)), np.float64)
    out = np.zeros((query.shape[0], abi.N_OUT))
    ensure_refined(pmap, speed_m_s, tool_temp_K)
    t_tool = float(pmap.Ttool_grid_K[0] if tool_temp_K is None else tool_temp_K)
    K.map_lookup(pmap.t1_grid_m, pmap.speed_grid_m_s, pmap.Ttool_grid_K, pmap.values, _SCALED, _DISCRETE, query,
                 float(speed_m_s), t_tool, out)
    return out


# ============================================================================ calibration
@dataclass(frozen=True)
class Calibration:
    """Material response of a pairing fitted to its measured Kienzle or mechanistic data (see
    :func:`build_calibration`) and applied through the model at any geometry.

    ``scale[e, k]`` is the flow-stress factor at the calibration chip thickness ``h_nodes_m[k]`` and the reference
    speed ``speeds_m_s[e]``: measured / model cutting (tangential) force at the entry's reference geometry. It
    multiplies every force of the model (shearing, edge, ploughing, indentation), so the model's own dependence on
    rake, inclination, edge radius, speed, temperature and wear carries the measurement to other conditions.
    ``friction_shift_rad[e, k]`` is added to the model's mean rake-face friction angle before the oblique
    transformation: a measured thrust / cutting force ratio becomes an interface property (the friction angle), so
    the thrust follows the target's rake through the force relations instead of a component factor formed at the
    reference rake. ``edge_scale[j, e]`` (mechanistic data only: measured / model edge coefficient of component
    j = t, r, a) replaces the flow-stress factor on the edge and ploughing forces; ``None`` for Kienzle data, whose
    forces are totals. Outside the measured h and speed ranges the end values are held."""

    kind: str                            # kienzle | mechanistic | none
    speeds_m_s: np.ndarray               # [n_entry] ascending
    h_nodes_m: np.ndarray                # [n_h]
    scale: np.ndarray                    # [n_entry, n_h]
    friction_shift_rad: np.ndarray       # [n_entry, n_h]
    edge_scale: np.ndarray | None = None  # [3, n_entry]
    clamped: bool = False
    sources: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def _weights(self, speed_m_s: float) -> list[tuple[int, float]]:
        n_e = self.speeds_m_s.shape[0]
        if n_e == 1:
            return [(0, 1.0)]
        ls = math.log(max(speed_m_s, 1e-12))
        grid = np.log(self.speeds_m_s)
        if ls <= grid[0]:
            return [(0, 1.0)]
        if ls >= grid[-1]:
            return [(n_e - 1, 1.0)]
        e = int(np.searchsorted(grid, ls) - 1)
        w = (ls - grid[e]) / (grid[e + 1] - grid[e])
        return [(e, 1.0 - w), (e + 1, w)]

    def factors(self, h_m, speed_m_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(flow-stress factor [n], friction-angle shift [n] (rad), edge factor [3, n]) at the query thicknesses
        and speed (linear in log h and log V between the calibration nodes, held outside)."""
        h = np.atleast_1d(np.asarray(h_m, dtype=float))
        lh = np.log(np.maximum(h, 1e-300))
        nodes = np.log(self.h_nodes_m)
        scale = np.zeros(h.shape[0])
        shift = np.zeros(h.shape[0])
        edge = np.zeros((3, h.shape[0]))
        for e, w in self._weights(speed_m_s):
            s_e = np.interp(lh, nodes, self.scale[e])
            scale += w * s_e
            shift += w * np.interp(lh, nodes, self.friction_shift_rad[e])
            if self.edge_scale is None:
                edge += w * s_e[None, :]
            else:
                edge += w * self.edge_scale[:, e][:, None]
        return scale, shift, edge


NO_CALIBRATION = Calibration("none", np.array([1.0]), np.array([1e-4]), np.ones((1, 1)), np.zeros((1, 1)))


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


def stated_rake(entry) -> float | None:
    """Normal rake of an entry's test as the data state it (calibration block, else reference rake), or None."""
    cal = entry.calibration
    rake = cal.rake_rad if cal.rake_rad is not None else (
        entry.rake_ref_rad if isinstance(entry, KienzleEntry) else entry.rake_rad)
    return None if rake is None else float(rake)


def _ref_rake(entry, group: EdgeGroup) -> float:
    """Normal rake of an entry's test: the stated one, else the group's own rake when that is a chip-forming rake (a
    chisel's or other indentation rake falls back to 0, the neutral geometry of catalogue data: the reference test
    cut a chip)."""
    rake = stated_rake(entry)
    if rake is not None:
        return rake
    return float(group.rake_rad) if not C.uses_indentation(EDGE_PERIPHERAL, group.rake_rad) else 0.0


class _Reference:
    """The model at a calibration entry's reference conditions: its force parts at the calibration nodes, from
    which the total forces for any friction-angle shift follow (the reference test cut with an ordinary
    chip-forming edge, so the group is a peripheral edge whatever the calibrated group is)."""

    def __init__(self, work, tool, group: EdgeGroup, state: SlowState, h_nodes: np.ndarray):
        self.h = contiguous(np.asarray(h_nodes, dtype=float), np.float64)
        pmap = build_process_map(work, tool, group.rake_rad, speed_grid_m_s=np.array([state.speed_m_s]),
                                 work_temp_K=state.work_temp_K, friction_factor=state.friction_factor,
                                 t1_grid_m=_calibration_t1_grid(h_nodes, work, group),
                                 tool_temp_grid_K=None if state.tool_temp_K is None else np.array([state.tool_temp_K]),
                                 adiabatic_tool=state.tool_temp_K is None)
        mu = effective_mu(work, tool, state.friction_factor)
        h_min = C.min_chip_thickness(work, group.edge_radius_m)
        k_ref = _reference_shear_flow_stress(work, pmap, h_min, state)
        self.parts = _force_parts(work, group, state, self.h, pmap, None, mu, h_min, k_ref)
        self.incl = group.incl_rad

    def totals(self, shift_rad) -> np.ndarray:
        """[3, n] model forces per unit width (t, r, a) with the friction angle shifted by ``shift_rad`` (scalar or
        per node), uncalibrated otherwise."""
        shift = np.broadcast_to(np.asarray(shift_rad, dtype=float), self.h.shape)
        f_sh, ke, _ = _assembled_parts(self.parts, self.incl, shift)
        return f_sh + ke[:, None]


def _solve_shift(ratio_fn, target: float, bound: float) -> tuple[float, bool]:
    """The friction-angle shift d in [-bound, bound] with ratio_fn(d) = target (bisection; the thrust / cutting
    force ratio grows with the friction angle). Returns (d, clamped)."""
    lo, hi = -bound, bound
    r_lo, r_hi = ratio_fn(lo), ratio_fn(hi)
    if not (math.isfinite(r_lo) and math.isfinite(r_hi)) or r_hi <= r_lo:
        return 0.0, True
    if target <= r_lo:
        return lo, True
    if target >= r_hi:
        return hi, True
    for _ in range(int(P.number(P.CALIBRATION, "cal_shift_bisections"))):
        mid = 0.5 * (lo + hi)
        if ratio_fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), False


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
    """Material response of the pairing (tool class / coating / process) fitted to the material's measured Kienzle
    or mechanistic data (see :class:`Calibration`).

    For each entry the model is run at the entry's reference conditions (its ``calibration`` block, else its stated
    reference speed and rake (helix for mechanistic data), else the group's chip-forming rake, the default
    calibration edge radius and room temperature; the tool at the workpiece temperature, a short test) with an
    ordinary chip-forming edge. Kienzle data: a measured feed force (kf11) fixes the friction-angle shift per
    calibration node (the model's feed / cutting ratio meets the measured one), then the flow-stress factor is the
    measured / model cutting force; without one the shift is 0. Mechanistic data: the model's forces are fitted by
    F = Kc h + Ke over the calibration range (as the measured coefficients were identified); the shift makes the
    model's Krc / Ktc the measured one, the flow-stress factor is Ktc measured / model, and the edge factors are the
    measured / model edge coefficients (the axial edge force takes the tangential factor). Factors are clamped to
    [min_cal_scale, max_cal_scale], shifts to +-max_cal_friction_shift_rad; a clamp is reported."""
    if work.jc is None:
        return NO_CALIBRATION
    kind, entries = select_entries(work, tool, process)
    if kind == "none":
        return NO_CALIBRATION
    cp = P.get(P.CALIBRATION)
    lo_clamp = float(cp["min_cal_scale"])
    hi_clamp = float(cp["max_cal_scale"])
    bound = float(cp["max_cal_friction_shift_rad"])
    n_nodes = int(cp["cal_h_nodes"])
    rows = []
    clamped = False
    for entry in entries:
        cal = entry.calibration
        rake = _ref_rake(entry, group)
        radius = cal.edge_radius_m if cal.edge_radius_m is not None else float(cp["default_cal_edge_radius_m"])
        work_temp = cal.work_temp_K if cal.work_temp_K is not None else float(cp["default_cal_work_temp_K"])
        incl = (entry.helix_rad or 0.0) if isinstance(entry, MechanisticEntry) else 0.0
        speed = _ref_speed(entry)
        h_lo, h_hi = _ref_h_range(entry)
        nodes = np.exp(np.linspace(math.log(h_lo), math.log(h_hi), n_nodes))
        ref_group = EdgeGroup(rake_rad=float(rake), incl_rad=float(incl), clearance_rad=group.clearance_rad,
                              edge_radius_m=float(radius), kind=EDGE_PERIPHERAL)
        state = SlowState(speed_m_s=float(speed if speed is not None else 1.0),
                          tool_temp_K=None if tool is None else (tool_temp_ref_K or work_temp), work_temp_K=work_temp,
                          friction_factor=1.0)
        ref = _Reference(work, tool, ref_group, state, nodes)
        edge = None
        if isinstance(entry, KienzleEntry):
            fc = entry.cutting_force_per_width(nodes)
            ff = entry.feed_force_per_width(nodes)
            shift = np.zeros(n_nodes)
            if ff is not None:
                for k in range(n_nodes):
                    def ratio(d, k=k):
                        f = ref.totals(np.full(n_nodes, d))
                        return f[1, k] / f[0, k] if f[0, k] > 0.0 else math.nan
                    shift[k], c = _solve_shift(ratio, ff[k] / fc[k], bound)
                    clamped = clamped or c
            model_t = ref.totals(shift)[0]
            with np.errstate(divide="ignore", invalid="ignore"):
                scale = np.where(model_t > 0.0, fc / model_t, 1.0)
        else:
            A = np.vstack([nodes, np.ones_like(nodes)]).T

            def fit(d):
                f = ref.totals(np.full(n_nodes, d))
                return np.array([np.linalg.lstsq(A, f[j], rcond=None)[0] for j in range(3)])     # [3, (Kc, Ke)]

            d = 0.0
            if entry.Ktc_Pa > 0.0:
                def ratio(x):
                    kc = fit(x)
                    return kc[1, 0] / kc[0, 0] if kc[0, 0] > 0.0 else math.nan
                d, c = _solve_shift(ratio, entry.Krc_Pa / entry.Ktc_Pa, bound)
                clamped = clamped or c
            coeff = fit(d)
            s_c = entry.Ktc_Pa / coeff[0, 0] if coeff[0, 0] > 0.0 else 1.0
            scale = np.full(n_nodes, s_c)
            shift = np.full(n_nodes, d)
            edge = np.full(3, s_c)
            for j, ke_meas in ((0, entry.Kte_N_per_m), (1, entry.Kre_N_per_m)):
                ke_m = coeff[j, 1]
                if abs(ke_m) > 1e-9 * abs(coeff[0, 0]) * h_hi and ke_m > 0.0:
                    s_e = ke_meas / ke_m
                    if lo_clamp <= s_e <= hi_clamp:
                        edge[j] = s_e
            edge[2] = edge[0]
        clamped = clamped or bool(np.any((scale < lo_clamp) | (scale > hi_clamp)))
        rows.append((float(speed) if speed is not None else math.nan, nodes, np.clip(scale, lo_clamp, hi_clamp),
                     shift, None if edge is None else np.clip(edge, lo_clamp, hi_clamp), entry.src))
    known = [r for r in rows if np.isfinite(r[0])]
    rows = sorted(known, key=lambda r: r[0]) if known else rows[:1]
    lo = min(r[1][0] for r in rows)
    hi = max(r[1][-1] for r in rows)
    nodes = np.exp(np.linspace(math.log(lo), math.log(hi), n_nodes))
    scale = np.stack([np.interp(np.log(nodes), np.log(r[1]), r[2]) for r in rows])
    shift = np.stack([np.interp(np.log(nodes), np.log(r[1]), r[3]) for r in rows])
    edge = None if rows[0][4] is None else np.stack([r[4] for r in rows], axis=1)
    speeds = np.array([r[0] if np.isfinite(r[0]) else 1.0 for r in rows])
    return Calibration(kind, speeds, nodes, scale, shift, edge, clamped, tuple(r[5] for r in rows))


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
    # flank contact length per h node: wear land + the edge arc below the stagnation point (contact.stagnation_arc_length
    # at min(h, h_min)), the length frac_flank_to_tool was computed for and thermal deposits the tool's part over
    flank_contact_m: np.ndarray = field(default_factory=lambda: np.zeros(0))

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


def _flank_partition(work, tool, group: EdgeGroup, state: SlowState, h: np.ndarray, h_min: float, k_ref: float,
                     mu: float, edge_t_scale, tau_land_Pa: float, work_surface_K, *, indentation=False,
                     rubbing: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(share of the flank contact's heat entering the tool, flank contact length) at each h node.

    The heat the hot loop books as flank (wear land) and ploughing (edge) heat - thermal splits both with this share
    and deposits the tool's part over the contact length (docs/models/thermal.md 1) - is, per unit edge length: the
    friction on the wear land (tau_land VB V) and on the edge arc below the stagnation point (the friction part of
    the edge force, contact.contact_power_parts), and the plastic work of the material pressed under the edge (its
    pressure part). The share follows from equal mean temperatures of tool and work over the contact VB + arc
    (heat.flank_heat_to_tool: Blok 1937), the plastic work being a heat source inside the work, uniform to the depth
    of the ploughed zone (ploughing_zone_depth_ratio x arc). ``edge_t_scale``: the calibration's factor on the
    tangential edge force per node (1 without calibration); ``indentation`` (bool or per node): the node's forces are
    the indentation contact (the rake face in contact, depth h)."""
    h = np.atleast_1d(np.asarray(h, dtype=float))
    n = h.shape[0]
    surface = np.broadcast_to(np.atleast_1d(np.asarray(work_surface_K, dtype=float)), (n,))
    scale = np.broadcast_to(np.atleast_1d(np.asarray(edge_t_scale, dtype=float)), (n,))
    indent = np.broadcast_to(np.atleast_1d(np.asarray(indentation, dtype=bool)), (n,))
    frac = np.zeros(n)
    length = np.zeros(n)
    flank_temp = state.tool_flank_temp_K if state.tool_flank_temp_K is not None else state.tool_temp_K
    vb = max(float(state.VB_m), 0.0)
    v = float(state.speed_m_s)
    depth_ratio = P.number(P.CONTACT, "ploughing_zone_depth_ratio")
    cache: dict = {}
    for k in range(n):
        depth = h_min if (rubbing or h[k] >= h_min) and not indent[k] else float(h[k])
        key = (depth, bool(indent[k]), float(scale[k]), float(surface[k]))
        if key in cache:
            frac[k], length[k] = cache[key]
            continue
        f_p, f_tau, arc = C.contact_power_parts(depth, group.edge_radius_m, h_min, group.rake_rad, group.incl_rad,
                                                k_ref, mu, indentation=bool(indent[k]))
        l_f = vb + arc
        share = 0.0
        if l_f > 0.0 and tool is not None and flank_temp is not None and v > 0.0:
            q_fr = v * (float(scale[k]) * f_tau + tau_land_Pa * vb) / l_f
            q_d = v * float(scale[k]) * max(f_p, 0.0) / l_f
            width = state.contact_width_m or P.number(P.CONTACT, "default_contact_aspect") * l_f
            share, _ = H.flank_heat_to_tool(work, tool, l_f, width, v, q_fr, q_d, depth_ratio * arc,
                                            float(surface[k]), float(flank_temp), state.contact_time_s)
        frac[k], length[k] = share, l_f
        cache[key] = (share, l_f)
    return frac, length


def _rubbing_group(work, tool, group, state, h, mu, h_min, k_ref, notes) -> GroupCoefficients:
    """Margins and other rubbing-only elements: no chip; the hot loop's indentation model with
    K_sp applies at every interference (h_min set to the top of the grid)."""
    depth = max(h_min, 1e-9)
    edge = C.edge_forces(group.edge_radius_m, depth, group.rake_rad, group.incl_rad, k_ref, mu)
    k_sp, src = C.indentation_coefficient(work, edge[abi.PL_FR], depth, group.edge_radius_m, group.rake_rad)
    sigma0, tau0 = C.flank_reference_stresses(k_ref, mu)
    s_mean, t_mean = C.flank_stresses(state.VB_m, sigma0, tau0)
    x, st, sr = C.ploughing_shape(group.edge_radius_m, depth, group.rake_rad, group.incl_rad, k_ref, mu)
    notes.append("rubbing-only group: forces from the indentation model (K_sp) in the hot loop")
    kw = _defaults(h, state.work_temp_K)
    flank_tool, flank_len = _flank_partition(work, tool, group, state, h, depth, k_ref, mu, 1.0, t_mean,
                                             np.full(h.shape[0], state.work_temp_K), rubbing=True)
    kw.update(frac_flank_to_tool=flank_tool, regime=np.full(h.shape[0], REGIME_PLOUGHING, np.int32), plough_x=x,
              plough_shape_t=st, plough_shape_r=sr)
    return GroupCoefficients(**kw, h_min_m=float(h[-1]), K_sp_N_per_m3=float(k_sp), K_sp_source=src,
                             v_min_m_s=C.min_shear_speed(), mu_eff=float(mu), flank_sigma0_Pa=float(sigma0),
                             flank_tau0_Pa=float(tau0), flank_land_star_m=P.number(P.CONTACT, "flank_plastic_land_m"),
                             flank_sigma_mean_Pa=float(s_mean), flank_tau_mean_Pa=float(t_mean), notes=notes,
                             flank_contact_m=flank_len)


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
    k_ref = _reference_shear_flow_stress(work, pmap, h_min, state)
    if group.kind == EDGE_MARGIN:
        return _rubbing_group(work, tool, group, state, h, mu, h_min, k_ref, notes)
    parts = _force_parts(work, group, state, h, pmap, bue_map, mu, h_min, k_ref)
    rows, t1, shearing, regime, edge = parts.rows, parts.t1, parts.shearing, parts.regime, parts.edge
    if parts.bue_note:
        notes.append(parts.bue_note)
    if parts.indentation_group:
        notes.append("indentation/extrusion group (chisel edge or normal rake below the shear limit)")
    cal_name = "none"
    clamped = False
    if calibration is not None and calibration.kind != "none":
        scale, shift, edge_scale = calibration.factors(h, state.speed_m_s)
        f_sh, ke, below = _assembled_parts(parts, group.incl_rad, shift)
        above = ~below
        total = np.where(above[None, :], scale[None, :] * f_sh + edge_scale * ke[:, None],
                         edge_scale * (f_sh + ke[:, None]))
        edge_part = np.where(above[None, :], edge_scale * ke[:, None], total)
        edge_part[:, h <= h_min] = total[:, h <= h_min]
        first = int(np.argmax(above)) if np.any(above) else n - 1
        ke = edge_scale[:, first] * ke
        cal_name = calibration.kind
        clamped = calibration.clamped
        notes.append(f"calibrated to {calibration.kind} data ({', '.join(calibration.sources)})")
    else:
        f_sh, ke, _ = _assembled_parts(parts, group.incl_rad, None)
        total = f_sh + ke[:, None]
        edge_part = np.broadcast_to(ke[:, None], total.shape).copy()
        edge_part[:, h <= h_min] = total[:, h <= h_min]
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
    if cal_name != "none":
        # the land carries the ploughing stresses of the edge (Waldorf 1999): the edge factors of the calibration
        sigma0 *= float(edge_scale[1, first])
        tau0 *= float(edge_scale[0, first])
    s_mean, t_mean = C.flank_stresses(state.VB_m, sigma0, tau0)
    edge_t = np.ones(n)
    if cal_name != "none":
        edge_t = np.where(h <= h_min, edge_scale[0], float(edge_scale[0, first]))
    flank_tool, flank_len = _flank_partition(work, tool, group, state, h, h_min, k_ref, mu, edge_t, t_mean, t_shear,
                                             indentation=parts.indent_mask | parts.indentation_group)
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
        calibration_clamped=clamped, notes=notes, flank_contact_m=flank_len)


def _reference_shear_flow_stress(work: WorkMaterial, pmap: abi.ProcessMap, h_min_m: float, state: SlowState) -> float:
    """Shear flow stress of the material at the edge (the edge, ploughing, indentation and wear-land pressures):
    the Johnson-Cook stress of the primary-zone state at t1 = h_min, with the strain limited to the saturation strain
    of chip material (Nakayama 1962: the strength of chips does not rise further above a shear strain of about 2).
    Where the map holds no chip-forming solution at that thickness (very negative rakes: a chisel edge, where the
    fallback's shear angle of a few degrees means strains of 3-5), the Johnson-Cook extrapolation would otherwise
    set the contact pressure of the indentation and edge models."""
    ref = lookup(pmap, max(h_min_m, pmap.t1_grid_m[0]), state.speed_m_s, state.tool_temp_K)[0]
    k_ab = ref[abi.OX_K_AB] if ref[abi.OX_K_AB] > 0.0 else _first_positive(pmap, abi.OX_K_AB)
    jc = work.jc
    eps = float(ref[abi.OX_EPS_AB])
    cap = P.number(P.CONTACT, "edge_strain_cap")
    if jc is None or not ref[abi.OX_K_AB] > 0.0 or not eps > cap:
        return float(k_ab)
    a, b, n = float(jc[abi.MAT_A_PA]), float(jc[abi.MAT_B_PA]), float(jc[abi.MAT_N])
    return float(k_ab * (a + b * cap ** n) / (a + b * eps ** n))


@dataclass
class _Parts:
    """Force parts of a group at its h nodes before calibration (see group_coefficients)."""

    rows: np.ndarray
    t1: np.ndarray
    shearing: np.ndarray
    rake_rows: np.ndarray
    regime: np.ndarray
    indent: np.ndarray                   # [3, n] indentation forces where indent_mask
    indent_mask: np.ndarray
    edge: np.ndarray                     # abi.PL_* vector at h_min
    below: np.ndarray                    # ploughing nodes (h <= h_min, not an indentation group)
    plough: np.ndarray                   # [3, n] ploughing forces at the below nodes
    indentation_group: bool
    bue_note: str = ""


def _force_parts(work, group: EdgeGroup, state: SlowState, h: np.ndarray, pmap, bue_map, mu: float, h_min: float,
                 k_ref: float) -> _Parts:
    """Map rows at t1 = h - h_min (BUE-blended), the indentation, edge and ploughing forces of a group."""
    n = h.shape[0]
    indentation_group = C.uses_indentation(group.kind, group.rake_rad)
    t1 = np.maximum(h - (0.0 if indentation_group else h_min), 0.0)
    rows = lookup(pmap, t1, state.speed_m_s, state.tool_temp_K)
    rake_rows = np.full(n, float(pmap.context[abi.CTX_RAKE_RAD]))
    note = ""
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
        note = f"BUE height {state.bue_height_m * 1e6:.1f} um: rake blended toward the BUE rake"
    status = rows[:, abi.OX_STATUS]
    shearing = (t1 > 0.0) & ((status == abi.STATUS_OXLEY) | (status == abi.STATUS_MERCHANT))
    regime = np.where(shearing, np.where(status == abi.STATUS_OXLEY, REGIME_OXLEY, REGIME_MERCHANT),
                      REGIME_NONE).astype(np.int32)
    indent = np.zeros((3, n))
    indent_mask = np.zeros(n, bool)
    if indentation_group or np.any((t1 > 0.0) & ~shearing):
        for k in range(n):
            if indentation_group or (t1[k] > 0.0 and not shearing[k]):
                f = C.indentation_forces(h[k], group.edge_radius_m, h_min, group.rake_rad, group.incl_rad, k_ref, mu)
                indent[:, k] = (f[abi.PL_FT], f[abi.PL_FR], f[abi.PL_FA])
                indent_mask[k] = True
                regime[k] = REGIME_INDENTATION
    if indentation_group or h_min <= 0.0 or group.edge_radius_m <= 0.0:
        edge = np.zeros(abi.N_PL)
    else:
        edge = C.edge_forces(group.edge_radius_m, h_min, group.rake_rad, group.incl_rad, k_ref, mu)
    below = (h <= h_min) & (not indentation_group)
    plough = np.zeros((3, n))
    for k in np.nonzero(below)[0]:
        f = C.ploughing_forces(h[k], group.edge_radius_m, h_min, group.rake_rad, group.incl_rad, k_ref, mu)
        plough[:, k] = (f[abi.PL_FT], f[abi.PL_FR], f[abi.PL_FA])
        regime[k] = REGIME_PLOUGHING
    return _Parts(rows, t1, shearing, rake_rows, regime, indent, indent_mask, edge, below, plough, indentation_group,
                  note)


def _assembled_parts(parts: _Parts, incl_rad: float, shift_rad) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(shearing part f_sh [3, n], edge force ke [3], ploughing mask) with the model total f_sh + ke at every node:
    shearing forces (friction angle shifted by ``shift_rad`` when given) or indentation forces above h_min, the
    ploughing force minus ke below it."""
    f_sh = _oblique_shearing_forces(parts.rows, parts.t1, parts.shearing, parts.rake_rows, incl_rad, shift_rad)
    m = parts.indent_mask
    f_sh[:, m] = parts.indent[:, m]
    ke = np.array([parts.edge[abi.PL_FT], parts.edge[abi.PL_FR], parts.edge[abi.PL_FA]])
    b = parts.below
    f_sh[:, b] = parts.plough[:, b] - ke[:, None]
    return f_sh, ke, b


def _oblique_shearing_forces(rows: np.ndarray, t1: np.ndarray, shearing: np.ndarray, rake_rows: np.ndarray,
                             incl_rad: float, shift_rad=None) -> np.ndarray:
    """Shearing forces per unit width [3, n] in the (t, r, a) frame: the interpolated orthogonal
    forces of the map times the ratio of the oblique to the orthogonal Budak-Altintas-Armarego
    coefficients at the row's (tau_s, phi, lambda, rake), so that i = 0 reproduces the map exactly.
    ``shift_rad`` (calibration, per node) is added to the friction angle of the oblique coefficients (the map's
    orthogonal forces normalise them at the unshifted angle); the result stays within the force relations'
    validity (phi + lambda - alpha below 90 deg, lambda - alpha above -90 deg)."""
    tau = np.where(shearing, rows[:, abi.OX_K_AB], 1.0)
    phi = np.where(shearing, rows[:, abi.OX_PHI], 0.5)
    lam = np.where(shearing, rows[:, abi.OX_LAMBDA], 0.5)
    out = np.zeros((3, t1.shape[0]))
    for k in np.nonzero(shearing)[0]:
        kt0, kr0, _ = O.cutting_coefficients(tau[k], phi[k], lam[k], float(rake_rows[k]), 0.0)
        lam_k = lam[k]
        if shift_rad is not None:
            d = float(np.asarray(shift_rad)[k])
            if d != 0.0:
                rk = float(rake_rows[k])
                lam_k = min(max(lam[k] + d, rk - 0.5 * math.pi + SHIFT_MARGIN_RAD),
                            0.5 * math.pi - phi[k] + rk - SHIFT_MARGIN_RAD)
        kti, kri, kai = O.cutting_coefficients(tau[k], phi[k], lam_k, float(rake_rows[k]), incl_rad)
        fc = rows[k, abi.OX_FC]
        ft = rows[k, abi.OX_FT]
        s_t = fc / (float(kt0) * t1[k]) if float(kt0) != 0.0 else 1.0
        s_r = ft / (float(kr0) * t1[k]) if abs(float(kr0)) > 1e-9 * abs(float(kt0)) else s_t
        out[0, k] = s_t * float(kti) * t1[k]
        out[1, k] = s_r * float(kri) * t1[k]
        out[2, k] = s_t * float(kai) * t1[k]
    return out


def _first_positive(pmap: abi.ProcessMap, slot: int) -> float:
    ensure_refined(pmap)
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
