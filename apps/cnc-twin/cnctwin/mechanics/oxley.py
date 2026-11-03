"""Oxley's predictive machining theory for orthogonal cutting (docs/models/mechanics.md, "Oxley").

:func:`solve` runs the compiled point solver (:func:`cnctwin.mechanics.kernels.oxley_point`)
for one condition and returns an :class:`OrthogonalResult` with forces per unit width, the
shear-zone and interface state, temperatures, heat generation by zone and the heat partition.
:func:`solve_reference` is an independent plain-Python/SciPy solver of the same equations
(exact root finding everywhere, no interpolation on the phi grid) used to verify the kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy import optimize

from ..units import REFERENCE_TEMP_K
from . import abi
from . import kernels as K
from . import params as P
from .materials import ToolSide, WorkMaterial

STATUS_NAMES = {abi.STATUS_INVALID: "invalid", abi.STATUS_OXLEY: "oxley", abi.STATUS_MERCHANT: "merchant",
                abi.STATUS_INDENTATION: "indentation"}
REGIME_NAMES = {abi.REGIME_STICKING: "sticking", abi.REGIME_SLIDING: "sliding", 0.0: "none"}


class KernelInputs(NamedTuple):
    """Arrays the Oxley kernels take for one material / tool / friction / workpiece temperature."""

    mat: np.ndarray
    cp_T: np.ndarray
    cp_v: np.ndarray
    k_T: np.ndarray
    k_v: np.ndarray
    tool: np.ndarray
    kt_T: np.ndarray
    kt_v: np.ndarray
    ct_T: np.ndarray
    ct_v: np.ndarray
    prm: np.ndarray


def _c(array) -> np.ndarray:
    out = np.ascontiguousarray(array, dtype=np.float64)
    return out.copy() if not out.flags.writeable else out


def effective_mu(work: WorkMaterial, tool: ToolSide | None, friction_factor: float = 1.0) -> float:
    """Sliding friction coefficient of the pairing (database, best tool class / coating match)
    times the coolant friction factor (CoolantFeedback.friction_factor)."""
    tool_class = tool.tool_class if tool is not None else "carbide"
    coating = tool.coating if tool is not None else "none"
    return work.friction_for(tool_class, coating).mu * float(friction_factor)


def kernel_inputs(work: WorkMaterial, tool: ToolSide | None, *, tool_temp_K: float | None, mu_limit: float,
                  mu_fallback: float, contact_width_m: float | None = None,
                  contact_time_s: float | None = None) -> KernelInputs:
    """Pack material, tool and friction data for the kernels. ``tool_temp_K`` None (or no tool)
    keeps the tool adiabatic: all rake friction heat enters the chip, as in Oxley's theory."""
    if work.jc is None:
        raise ValueError(f"material {work.id!r} has no Johnson-Cook flow stress (use the data-only model)")
    vec = np.zeros(abi.N_TOOL)
    if tool is not None and tool_temp_K is not None:
        vec[abi.TOOL_T_RAKE_K] = float(tool_temp_K)
        vec[abi.TOOL_RHO_KG_M3] = tool.rho_kg_m3
        vec[abi.TOOL_R_COAT_M2K_W] = tool.coating_resistance_m2K_W
        vec[abi.TOOL_WIDTH_M] = float(contact_width_m) if contact_width_m else 0.0
        vec[abi.TOOL_TIME_S] = float(contact_time_s) if contact_time_s else 0.0
        kt_T, kt_v, ct_T, ct_v = tool.k_T_K, tool.k_W_mK, tool.cp_T_K, tool.cp_J_kgK
    else:
        vec[abi.TOOL_T_RAKE_K] = -1.0
        vec[abi.TOOL_RHO_KG_M3] = 1.0
        kt_T = ct_T = np.array([REFERENCE_TEMP_K])
        kt_v = ct_v = np.array([1.0])
    return KernelInputs(_c(work.jc), _c(work.cp_T_K), _c(work.cp_J_kgK), _c(work.k_T_K), _c(work.k_W_mK), _c(vec),
                        _c(kt_T), _c(kt_v), _c(ct_T), _c(ct_v), P.oxley_vector(mu_limit, mu_fallback))


def zorev_split(tau_int_Pa: float, sigma_n_Pa: float, mu: float, exponent: float) -> tuple[float, float, float]:
    """Sticking fraction s, sticking-zone shear stress and peak normal stress of Zorev's contact
    model consistent with the mean stresses: sigma(x) = sigma_max (1 - x/l_c)^a with
    sigma_max = (a + 1) sigma_N; tau = tau_p on [0, s l_c], mu sigma(x) beyond, continuous at
    s l_c and with mean tau_int, i.e. (1 - s)^a (1 + a s) = tau_int / (mu sigma_N)."""
    sigma_max = (exponent + 1.0) * sigma_n_Pa
    if mu <= 0.0 or sigma_n_Pa <= 0.0:
        return 1.0, tau_int_Pa, sigma_max
    ratio = tau_int_Pa / (mu * sigma_n_Pa)
    if ratio >= 1.0:
        return 0.0, mu * sigma_max, sigma_max
    if ratio <= 0.0:
        return 1.0, tau_int_Pa, sigma_max
    s = optimize.brentq(lambda x: (1.0 - x) ** exponent * (1.0 + exponent * x) - ratio, 0.0, 1.0, xtol=1e-14)
    return s, mu * sigma_max * (1.0 - s) ** exponent, sigma_max


@dataclass(frozen=True)
class OrthogonalResult:
    """One orthogonal cutting solution, per unit width of cut (SI)."""

    vector: np.ndarray              # the kernel output (abi.OX_*)
    t1_m: float
    speed_m_s: float
    rake_rad: float
    work_temp_K: float

    def __getitem__(self, slot: int) -> float:
        return float(self.vector[slot])

    @property
    def status(self) -> str:
        return STATUS_NAMES[float(self.vector[abi.OX_STATUS])]

    @property
    def regime(self) -> str:
        return REGIME_NAMES.get(float(self.vector[abi.OX_REGIME]), "none")

    @property
    def converged(self) -> bool:
        return self.vector[abi.OX_STATUS] == abi.STATUS_OXLEY

    cutting_force_N_per_m = property(lambda s: float(s.vector[abi.OX_FC]))
    thrust_force_N_per_m = property(lambda s: float(s.vector[abi.OX_FT]))
    shear_angle_rad = property(lambda s: float(s.vector[abi.OX_PHI]))
    friction_angle_rad = property(lambda s: float(s.vector[abi.OX_LAMBDA]))
    chip_ratio = property(lambda s: float(s.vector[abi.OX_CHIP_RATIO]))
    chip_thickness_m = property(lambda s: float(s.vector[abi.OX_T2]))
    contact_length_m = property(lambda s: float(s.vector[abi.OX_LC]))
    shear_flow_stress_Pa = property(lambda s: float(s.vector[abi.OX_K_AB]))
    interface_shear_Pa = property(lambda s: float(s.vector[abi.OX_TAU_INT]))
    interface_normal_Pa = property(lambda s: float(s.vector[abi.OX_SIGMA_N]))
    shear_strain = property(lambda s: 2.0 * math.sqrt(3.0) * float(s.vector[abi.OX_EPS_AB]))
    shear_strain_rate_1_s = property(lambda s: math.sqrt(3.0) * float(s.vector[abi.OX_EPSDOT_AB]))
    T_shear_K = property(lambda s: float(s.vector[abi.OX_T_AB]))
    T_interface_K = property(lambda s: float(s.vector[abi.OX_T_INT]))
    T_chip_K = property(lambda s: float(s.vector[abi.OX_T_CHIP]))
    beta_to_work = property(lambda s: float(s.vector[abi.OX_BETA]))
    rake_heat_to_chip = property(lambda s: float(s.vector[abi.OX_R_CHIP]))
    chip_speed_m_s = property(lambda s: float(s.vector[abi.OX_VC]))

    @property
    def heat_shear_W_per_m(self) -> float:
        """Primary-zone power F_S V_s per unit width."""
        return float(self.vector[abi.OX_FS] * self.vector[abi.OX_VS])

    @property
    def heat_rake_W_per_m(self) -> float:
        """Secondary-zone (rake friction) power F V_c per unit width."""
        return float(self.vector[abi.OX_F] * self.vector[abi.OX_VC])

    def sticking(self, mu: float, exponent: float | None = None) -> tuple[float, float, float]:
        """Zorev split (sticking fraction, sticking shear stress, peak normal stress) of the contact."""
        a = P.number(P.CONTACT, "zorev_exponent") if exponent is None else exponent
        return zorev_split(self.interface_shear_Pa, self.interface_normal_Pa, mu, a)


def solve(work: WorkMaterial, t1_m: float, speed_m_s: float, rake_rad: float, *, work_temp_K: float = REFERENCE_TEMP_K,
          tool: ToolSide | None = None, tool_temp_K: float | None = None, friction_factor: float = 1.0,
          coulomb_limit: bool = True, contact_width_m: float | None = None,
          contact_time_s: float | None = None) -> OrthogonalResult:
    """Orthogonal solution at uncut chip thickness ``t1_m``, cutting speed and normal rake.

    ``coulomb_limit`` caps the interface shear stress at mu sigma_N with mu = database sliding
    friction x ``friction_factor`` (False: Oxley's original sticking condition)."""
    mu = effective_mu(work, tool, friction_factor)
    inputs = kernel_inputs(work, tool, tool_temp_K=tool_temp_K, mu_limit=mu if coulomb_limit else -1.0,
                           mu_fallback=mu, contact_width_m=contact_width_m, contact_time_s=contact_time_s)
    out = np.zeros(abi.N_OUT)
    case = np.array([float(t1_m), float(speed_m_s), float(rake_rad), float(work_temp_K)])
    K.oxley_point(case, *inputs, out)
    if out[abi.OX_STATUS] in (abi.STATUS_OXLEY, abi.STATUS_MERCHANT):
        out[abi.OX_STICK] = zorev_split(out[abi.OX_TAU_INT], out[abi.OX_SIGMA_N], mu,
                                        P.number(P.CONTACT, "zorev_exponent"))[0]
    return OrthogonalResult(out, float(t1_m), float(speed_m_s), float(rake_rad), float(work_temp_K))


# ------------------------------------------------------------------------------ reference solver
def _case(t1, v, alpha, tw):
    return np.array([float(t1), float(v), float(alpha), float(tw)])


def reference_c0(phi: float, case: np.ndarray, inputs: KernelInputs, n_scan: int = 33):
    """C0(phi) by a fine scan + brentq (smallest cutting force among roots) or the bounded minimum
    of the mismatch; returns (state, residual) or None. Independent of the kernel's algorithm."""
    prm = inputs.prm
    st = np.zeros(K.N_ST)

    def g2(c0):
        if K.shear_plane_state(phi, c0, abi.NO_REF, case, inputs.mat, inputs.cp_T, inputs.cp_v, inputs.k_T,
                               inputs.k_v, prm, st) != 1:
            return math.nan
        return st[K.ST_G2]

    grid = np.linspace(prm[abi.P_C0_MIN], prm[abi.P_C0_MAX], n_scan)
    values = np.array([g2(c) for c in grid])
    roots = []
    for a, b, fa, fb in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb <= 0.0:
            roots.append(optimize.brentq(g2, a, b, xtol=1e-12) if fa != fb else a)
    if roots:
        best = None
        for c0 in roots:
            g2(c0)
            if best is None or st[K.ST_FC] < best[1]:
                best = (c0, st[K.ST_FC])
        g2(best[0])
        return st.copy(), abs(st[K.ST_G2]) / st[K.ST_KAB]
    finite = np.isfinite(values)
    if not finite.any():
        return None
    k_ab = []
    for c in grid:
        g2(c)
        k_ab.append(st[K.ST_KAB] if st[K.ST_KAB] > 0 else math.nan)
    rel = np.abs(values) / np.array(k_ab)
    j = int(np.nanargmin(rel))
    lo = grid[max(j - 1, 0)]
    hi = grid[min(j + 1, len(grid) - 1)]
    res = optimize.minimize_scalar(lambda c: abs(g2(c)) / st[K.ST_KAB] if np.isfinite(g2(c)) else 1e9,
                                   bounds=(lo, hi), method="bounded", options={"xatol": 1e-10})
    if res.fun > prm[abi.P_C0_RES_MAX]:
        return None
    g2(res.x)
    return st.copy(), abs(st[K.ST_G2]) / st[K.ST_KAB]


def solve_reference(work: WorkMaterial, t1_m: float, speed_m_s: float, rake_rad: float, *,
                    work_temp_K: float = REFERENCE_TEMP_K, tool: ToolSide | None = None,
                    tool_temp_K: float | None = None, friction_factor: float = 1.0, coulomb_limit: bool = True,
                    n_phi: int = 117, n_delta: int = 40) -> dict:
    """Plain-Python/SciPy solution of the same equations (verification of :func:`solve`):
    phi roots bracketed on a 0.5 deg grid and solved by brentq with exact C0(phi); delta by a
    40-node log grid and bounded scalar minimisation of the cutting force. Returns a dict of
    the main outputs (status 'oxley' or 'failed')."""
    mu = effective_mu(work, tool, friction_factor)
    inputs = kernel_inputs(work, tool, tool_temp_K=tool_temp_K, mu_limit=mu if coulomb_limit else -1.0,
                           mu_fallback=mu)
    prm = inputs.prm
    case = _case(t1_m, speed_m_s, rake_rad, work_temp_K)
    it = np.zeros(K.N_IT)
    phis = np.linspace(prm[abi.P_PHI_MIN], prm[abi.P_PHI_MAX], n_phi)
    c0_cache: dict[float, tuple | None] = {}

    def state(phi):
        if phi not in c0_cache:
            c0_cache[phi] = reference_c0(phi, case, inputs)
        return c0_cache[phi]

    def g1(phi, delta):
        s = state(phi)
        if s is None:
            return math.nan
        return K.interface_state(s[0], delta, case, inputs.mat, inputs.cp_T, inputs.cp_v, inputs.k_T, inputs.k_v,
                                 inputs.tool, inputs.kt_T, inputs.kt_v, inputs.ct_T, inputs.ct_v, prm, it)

    def best_root(delta):
        values = [g1(p, delta) for p in phis]
        best = None
        for a, b, fa, fb in zip(phis[:-1], phis[1:], values[:-1], values[1:]):
            if not (np.isfinite(fa) and np.isfinite(fb)) or fa * fb > 0.0:
                continue
            try:
                root = optimize.brentq(lambda p: g1(p, delta), a, b, xtol=1e-11)
            except ValueError:
                continue
            s = state(root)
            if s is None:
                continue
            fc = s[0][K.ST_FC]
            if best is None or fc < best[1] or (fc == best[1] and root > best[0]):
                best = (root, fc)
        return best

    deltas = np.exp(np.linspace(math.log(prm[abi.P_DELTA_MIN]), math.log(prm[abi.P_DELTA_MAX]), n_delta))
    found = [(d, best_root(d)) for d in deltas]
    valid = [(d, r) for d, r in found if r is not None]
    if not valid:
        return {"status": "failed"}
    j = min(range(len(found)), key=lambda k: found[k][1][1] if found[k][1] is not None else math.inf)
    lo = math.log(deltas[max(j - 1, 0)])
    hi = math.log(deltas[min(j + 1, len(deltas) - 1)])

    def fc_of(logd):
        r = best_root(math.exp(logd))
        return r[1] if r is not None else 1e30

    opt = optimize.minimize_scalar(fc_of, bounds=(lo, hi), method="bounded", options={"xatol": 1e-6})
    delta = math.exp(opt.x) if opt.fun <= found[j][1][1] else deltas[j]
    phi, _ = best_root(delta)
    st, c0res = state(phi)
    K.interface_state(st, delta, case, inputs.mat, inputs.cp_T, inputs.cp_v, inputs.k_T, inputs.k_v, inputs.tool,
                      inputs.kt_T, inputs.kt_v, inputs.ct_T, inputs.ct_v, prm, it)
    return {"status": "oxley", "phi": phi, "C0": st[K.ST_C0], "delta": delta, "Fc": st[K.ST_FC], "Ft": st[K.ST_FT],
            "T_AB": st[K.ST_TAB], "T_int": it[K.IT_TINT], "lc": st[K.ST_LC], "t2": st[K.ST_T2],
            "k_AB": st[K.ST_KAB], "k_chip": it[K.IT_KCHIP], "R": it[K.IT_R], "c0_residual": c0res}


__all__ = ["solve", "solve_reference", "OrthogonalResult", "kernel_inputs", "KernelInputs", "effective_mu",
           "zorev_split", "reference_c0", "STATUS_NAMES", "REGIME_NAMES"]
