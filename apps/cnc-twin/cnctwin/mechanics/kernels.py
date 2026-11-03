"""Numba kernels of the mechanics library (docs/models/mechanics.md, "Kernels").

Every function here is compiled with an explicit signature, ``cache=True, nogil=True,
error_model="numpy"``, without fast-math or parallel loops. Arguments are C-contiguous float64
arrays and scalars; slot names are in :mod:`cnctwin.mechanics.abi`. Root finding uses Brent's
method in reverse-communication form (no functions are passed around), and every grid and
tie-break is fixed, so a solution depends only on its inputs.

Contents
--------
* Johnson-Cook flow stress and its equivalent strain-hardening index (Lalwani et al. 2009).
* Oxley's orthogonal analysis: shear-plane state, interface state with the rake heat partition
  at the current tool temperature, the point solver (C0, phi, delta) and the modified-Merchant
  fallback; :func:`oxley_map` fills a :class:`~cnctwin.mechanics.abi.ProcessMap` grid.
* Table lookups for the coefficient pipeline (trilinear map lookup, log-h row interpolation).
* Edge contact (ploughing / indentation) forces, oxide exposure and built-up-edge dynamics.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from ..core.rng import hash_uniform
from .abi import (MAT_A_PA, MAT_B_PA, MAT_C, MAT_EPS0_1_S, MAT_M, MAT_N, MAT_RHO_KG_M3, MAT_TMELT_K, MAT_TREF_K,
                  N_OUT, NO_REF, OX_BETA, OX_C0, OX_C0_RES, OX_CHIP_RATIO, OX_DELTA, OX_DT_C, OX_DT_M, OX_DT_SZ,
                  OX_EPS_AB, OX_EPS_INT, OX_EPSDOT_AB, OX_EPSDOT_INT, OX_F, OX_FC, OX_FS, OX_FT, OX_G1_RES, OX_K_AB,
                  OX_K_CHIP, OX_LAMBDA, OX_LC, OX_N, OX_N_EQ, OX_P_SHEAR, OX_PHI, OX_R_CHIP, OX_REGIME,
                  OX_SIGMA_N, OX_SIGMA_NP, OX_STATUS, OX_T2, OX_T_AB, OX_T_CAPPED, OX_T_CHIP, OX_T_INT,
                  OX_TAU_INT, OX_THETA, OX_VC, OX_VS, P_BETA_HA, P_BETA_HB, P_BETA_LA, P_BETA_LB, P_BETA_RTMIN,
                  P_BETA_SWITCH, P_C0_MAX, P_C0_MIN, P_C0_N, P_C0_RES_MAX, P_CONTACT_ASPECT, P_DELTA_ITER,
                  P_DELTA_MAX, P_DELTA_MIN, P_DELTA_N, P_DTM_A, P_DTM_B, P_DTM_C, P_ETA, P_ITER_MAX,
                  P_MERCHANT_C0, P_MERCHANT_DELTA, P_MU, P_MU_FALLBACK, P_PHI_FLOOR, P_PHI_MAX, P_PHI_MIN,
                  P_PHI_N, P_PSI, P_RATE_MIN, P_REF_C0, P_REF_CM, P_REF_DELTA, P_REF_LAMBDA, P_TOL_C0,
                  P_TOL_PHI, P_TOL_T, P_TSTAR_MAX, PL_AREA, PL_CONTACT, PL_FA, PL_FR, PL_FT, REGIME_SLIDING,
                  REGIME_STICKING, STATUS_INDENTATION, STATUS_MERCHANT, STATUS_OXLEY, TOOL_R_COAT_M2K_W,
                  TOOL_RHO_KG_M3, TOOL_T_RAKE_K, TOOL_TIME_S, TOOL_WIDTH_M)

_OPTS = dict(cache=True, nogil=True, error_model="numpy")
SQRT3 = math.sqrt(3.0)
HALF_PI = 0.5 * math.pi
DBL_EPS = 2.220446049250313e-16      # IEEE-754 double machine epsilon (Brent's tolerance floor)
GOLDEN = 0.6180339887498949          # (sqrt(5) - 1) / 2

# case vector: the cutting condition of one orthogonal solve
CASE_T1_M = 0
CASE_V_M_S = 1
CASE_ALPHA_RAD = 2
CASE_TW_K = 3
N_CASE = 4

# shear-plane state vector (internal to the solver)
ST_PHI = 0
ST_C0 = 1
ST_TAB = 2
ST_KAB = 3
ST_NEQ = 4
ST_FS = 5
ST_THETA = 6
ST_LAMBDA = 7
ST_FC = 8
ST_FT = 9
ST_F = 10
ST_N = 11
ST_LC = 12
ST_SIGNP = 13
ST_SIGN = 14
ST_TAUINT = 15
ST_VS = 16
ST_VC = 17
ST_T2 = 18
ST_GAMMA = 19
ST_EPS = 20
ST_EPSDOT = 21
ST_DTSZ = 22
ST_BETA = 23
ST_CAPPED = 24
ST_G2 = 25
ST_C0RES = 26
N_ST = 27

# interface state vector (internal)
IT_TINT = 0
IT_TC = 1
IT_DTC = 2
IT_DTM = 3
IT_R = 4
IT_EPS = 5
IT_EPSDOT = 6
IT_KCHIP = 7
IT_TAULIM = 8
IT_G1 = 9
IT_REGIME = 10
N_IT = 11

# Brent state slots
_BA, _BB, _BC, _BFA, _BFB, _BFC, _BD, _BE, _BTOL = 0, 1, 2, 3, 4, 5, 6, 7, 8
N_BRENT = 10


# ============================================================================ small numerics
@njit("float64(float64[::1], float64[::1], float64)", **_OPTS)
def interp1(xs, ys, x):
    """Piecewise-linear table lookup; the end values are held outside the table."""
    n = xs.shape[0]
    if n == 1 or x <= xs[0]:
        return ys[0]
    if x >= xs[n - 1]:
        return ys[n - 1]
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    t = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + t * (ys[hi] - ys[lo])


@njit("void(float64[::1], float64, float64, float64, float64, float64)", **_OPTS)
def brent_init(bs, a, b, fa, fb, tol):
    """Start Brent's root search on [a, b] with f(a) fa and f(b) fb of opposite sign."""
    bs[_BA] = a
    bs[_BB] = b
    bs[_BC] = b
    bs[_BFA] = fa
    bs[_BFB] = fb
    bs[_BFC] = fb
    bs[_BD] = b - a
    bs[_BE] = b - a
    bs[_BTOL] = tol


@njit("int64(float64[::1])", **_OPTS)
def brent_step(bs):
    """One step of Brent's method (Brent 1973, as in Numerical Recipes' zbrent).

    Returns 1 when converged (the root is ``bs[1]``); otherwise writes the next abscissa to
    ``bs[1]`` and returns 0: the caller evaluates f there, stores it in ``bs[4]`` and calls again.
    """
    a = bs[_BA]
    b = bs[_BB]
    c = bs[_BC]
    fa = bs[_BFA]
    fb = bs[_BFB]
    fc = bs[_BFC]
    d = bs[_BD]
    e = bs[_BE]
    tol = bs[_BTOL]
    if (fb > 0.0 and fc > 0.0) or (fb < 0.0 and fc < 0.0):
        c = a
        fc = fa
        d = b - a
        e = d
    if abs(fc) < abs(fb):
        a = b
        b = c
        c = a
        fa = fb
        fb = fc
        fc = fa
    tol1 = 2.0 * DBL_EPS * abs(b) + 0.5 * tol
    xm = 0.5 * (c - b)
    if abs(xm) <= tol1 or fb == 0.0:
        bs[_BA] = a
        bs[_BB] = b
        bs[_BC] = c
        bs[_BFA] = fa
        bs[_BFB] = fb
        bs[_BFC] = fc
        bs[_BD] = d
        bs[_BE] = e
        return 1
    if abs(e) >= tol1 and abs(fa) > abs(fb):
        s = fb / fa
        if a == c:
            p = 2.0 * xm * s
            q = 1.0 - s
        else:
            qq = fa / fc
            r = fb / fc
            p = s * (2.0 * xm * qq * (qq - r) - (b - a) * (r - 1.0))
            q = (qq - 1.0) * (r - 1.0) * (s - 1.0)
        if p > 0.0:
            q = -q
        p = abs(p)
        min1 = 3.0 * xm * q - abs(tol1 * q)
        min2 = abs(e * q)
        if 2.0 * p < (min1 if min1 < min2 else min2):
            e = d
            d = p / q
        else:
            d = xm
            e = d
    else:
        d = xm
        e = d
    a = b
    fa = fb
    if abs(d) > tol1:
        b = b + d
    elif xm > 0.0:
        b = b + tol1
    else:
        b = b - tol1
    bs[_BA] = a
    bs[_BB] = b
    bs[_BC] = c
    bs[_BFA] = fa
    bs[_BFC] = fc
    bs[_BD] = d
    bs[_BE] = e
    return 0


# ============================================================================ material laws
@njit("float64(float64, float64, float64, float64[::1], float64, float64)", **_OPTS)
def jc_flow_stress(eps, epsdot_1_s, temp_K, mat, tstar_max, rate_min):
    """Johnson-Cook equivalent flow stress (Pa) with T* clamped to [0, tstar_max] and the rate
    ratio held at ``rate_min`` below; the rate factor is floored at 1 - tstar_max."""
    e = eps if eps > 0.0 else 0.0
    hard = mat[MAT_A_PA] + mat[MAT_B_PA] * e ** mat[MAT_N]
    ratio = epsdot_1_s / mat[MAT_EPS0_1_S]
    if not ratio >= rate_min:
        ratio = rate_min
    rate = 1.0 + mat[MAT_C] * math.log(ratio)
    floor = 1.0 - tstar_max
    if rate < floor:
        rate = floor
    ts = (temp_K - mat[MAT_TREF_K]) / (mat[MAT_TMELT_K] - mat[MAT_TREF_K])
    if ts < 0.0:
        ts = 0.0
    if ts > tstar_max:
        ts = tstar_max
    return hard * rate * (1.0 - ts ** mat[MAT_M])


@njit("float64(float64, float64[::1])", **_OPTS)
def jc_hardening_index(eps, mat):
    """Equivalent strain-hardening index n_eq = d ln(sigma) / d ln(eps) = n B eps^n / (A + B eps^n)."""
    if eps <= 0.0:
        return 0.0
    ben = mat[MAT_B_PA] * eps ** mat[MAT_N]
    return mat[MAT_N] * ben / (mat[MAT_A_PA] + ben)


@njit("float64(float64, float64[::1])", **_OPTS)
def beta_partition(rt_tanphi, prm):
    """Fraction of the shear-zone heat conducted into the work (Boothroyd's results, Oxley's fit)."""
    x = rt_tanphi
    if not x >= prm[P_BETA_RTMIN]:
        x = prm[P_BETA_RTMIN]
    if x <= prm[P_BETA_SWITCH]:
        b = prm[P_BETA_LA] - prm[P_BETA_LB] * math.log10(x)
    else:
        b = prm[P_BETA_HA] - prm[P_BETA_HB] * math.log10(x)
    if b < 0.0:
        return 0.0
    if b > 1.0:
        return 1.0
    return b


@njit("float64(float64)", **_OPTS)
def rect_area_factor(m):
    """Mean temperature rise of a uniformly heated a x b rectangle on a half-space in units of
    q a / k, for m = b / a (exact; Loewen & Shaw's area factor)."""
    if m <= 0.0:
        return 0.0
    if m > 1.0:
        excess = m * m * m * math.expm1(1.5 * math.log1p(1.0 / (m * m)))   # (1 + m^2)^1.5 - m^3
        term = (1.0 - excess) / (3.0 * m)
    else:
        term = (1.0 + m * m * m - (1.0 + m * m) ** 1.5) / (3.0 * m)
    return (math.asinh(m) + m * math.asinh(1.0 / m) + term) / math.pi


@njit("float64(float64, float64, float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1])",
      **_OPTS)
def tool_resistance(length_m, temp_K, tool, kt_T, kt_v, ct_T, ct_v, prm):
    """Mean temperature rise of the tool face per unit heat flux (m2 K / W) over a contact of
    ``length_m`` (perpendicular to the edge) adjacent to the cutting edge.

    Steady stationary source on a quarter-infinite tool with an adiabatic second face (image
    source: a 2 length x width rectangle on a half-space), capped by the transient
    one-dimensional rise averaged over one contact time, plus the coating resistance."""
    k = interp1(kt_T, kt_v, temp_K)
    width = tool[TOOL_WIDTH_M]
    if width <= 0.0:
        width = prm[P_CONTACT_ASPECT] * length_m
    resistance = 2.0 * length_m * rect_area_factor(width / (2.0 * length_m)) / k
    contact_time = tool[TOOL_TIME_S]
    if contact_time > 0.0:
        kappa = k / (tool[TOOL_RHO_KG_M3] * interp1(ct_T, ct_v, temp_K))
        transient = 4.0 / 3.0 * math.sqrt(kappa * contact_time / math.pi) / k
        if transient < resistance:
            resistance = transient
    return resistance + tool[TOOL_R_COAT_M2K_W]


# ============================================================================ Oxley: shear plane
@njit("float64(float64, float64, float64, float64, float64, float64[::1], float64[::1], float64[::1], float64[::1],"
      " float64[::1], float64[::1], float64[::1], float64[::1])", **_OPTS)
def shear_zone_rise(temp_K, tanphi, vs_over_v_sinphi, eps, epsdot, case, mat, cp_T, cp_v, k_T, k_v, prm, aux):
    """Temperature rise of the primary zone dT_SZ = (1 - beta) F_S V_s / (rho V t1 c_p) with the
    flow stress and properties at ``temp_K``; aux[0] = k_AB, aux[1] = beta."""
    rho = mat[MAT_RHO_KG_M3]
    cp = interp1(cp_T, cp_v, temp_K)
    kth = interp1(k_T, k_v, temp_K)
    k_ab = jc_flow_stress(eps, epsdot, temp_K, mat, prm[P_TSTAR_MAX], prm[P_RATE_MIN]) / SQRT3
    rt = rho * cp * case[CASE_V_M_S] * case[CASE_T1_M] / kth
    beta = beta_partition(rt * tanphi, prm)
    aux[0] = k_ab
    aux[1] = beta
    return (1.0 - beta) * k_ab * vs_over_v_sinphi / (rho * cp)


@njit("float64(float64, float64, float64, float64, float64[::1], float64[::1], float64[::1], float64[::1],"
      " float64[::1], float64[::1], float64[::1], float64[::1])", **_OPTS)
def solve_shear_plane_temp(tanphi, vs_over_v_sinphi, eps, epsdot, case, mat, cp_T, cp_v, k_T, k_v, prm, aux):
    """T_AB = T_w + eta dT_SZ(T_AB) by Brent's method on [T_w, T_cap]; aux[0] k_AB, aux[1] beta,
    aux[2] dT_SZ, aux[3] 1 if capped at the homologous-temperature clamp."""
    tw = case[CASE_TW_K]
    eta = prm[P_ETA]
    t_cap = mat[MAT_TREF_K] + prm[P_TSTAR_MAX] * (mat[MAT_TMELT_K] - mat[MAT_TREF_K])
    aux[3] = 0.0
    if t_cap <= tw:
        aux[2] = shear_zone_rise(tw, tanphi, vs_over_v_sinphi, eps, epsdot, case, mat, cp_T, cp_v, k_T, k_v, prm,
                                 aux)
        aux[3] = 1.0
        return tw
    fa = -eta * shear_zone_rise(tw, tanphi, vs_over_v_sinphi, eps, epsdot, case, mat, cp_T, cp_v, k_T, k_v, prm,
                                aux)
    if fa >= 0.0:
        aux[2] = -fa / eta
        return tw
    rise_cap = shear_zone_rise(t_cap, tanphi, vs_over_v_sinphi, eps, epsdot, case, mat, cp_T, cp_v, k_T, k_v, prm,
                               aux)
    fb = t_cap - tw - eta * rise_cap
    if fb <= 0.0:
        aux[2] = rise_cap
        aux[3] = 1.0
        return t_cap
    bs = np.zeros(N_BRENT)
    brent_init(bs, tw, t_cap, fa, fb, prm[P_TOL_T])
    for _ in range(int(prm[P_ITER_MAX])):
        if brent_step(bs) == 1:
            break
        t = bs[_BB]
        bs[_BFB] = t - tw - eta * shear_zone_rise(t, tanphi, vs_over_v_sinphi, eps, epsdot, case, mat, cp_T, cp_v,
                                                  k_T, k_v, prm, aux)
    t_ab = bs[_BB]
    aux[2] = shear_zone_rise(t_ab, tanphi, vs_over_v_sinphi, eps, epsdot, case, mat, cp_T, cp_v, k_T, k_v, prm, aux)
    return t_ab


@njit("int64(float64, float64, float64, float64[::1], float64[::1], float64[::1], float64[::1], float64[::1],"
      " float64[::1], float64[::1], float64[::1])", **_OPTS)
def shear_plane_state(phi, c0, lam_given, case, mat, cp_T, cp_v, k_T, k_v, prm, st):
    """Oxley's shear-plane analysis at shear angle ``phi`` and strain-rate constant ``c0``.

    With ``lam_given`` < -9 the angle theta of the resultant follows from the stress field,
    tan(theta) = 1 + 2 (pi/4 - phi) - C0 n_eq (Oxley); otherwise the friction angle is given and
    theta = phi + lambda - alpha (Merchant geometry, fallback). Returns 1 for a physical state."""
    for i in range(N_ST):
        st[i] = 0.0
    t1 = case[CASE_T1_M]
    v = case[CASE_V_M_S]
    alpha = case[CASE_ALPHA_RAD]
    sphi = math.sin(phi)
    cpa = math.cos(phi - alpha)
    if sphi <= 0.0 or cpa <= 0.0:
        return 0
    vs = v * math.cos(alpha) / cpa
    vc = v * sphi / cpa
    t2 = t1 * cpa / sphi
    gamma = 0.5 * math.cos(alpha) / (sphi * cpa)
    eps = gamma / SQRT3
    l_ab = t1 / sphi
    epsdot = c0 * vs / l_ab / SQRT3
    aux = np.zeros(4)
    t_ab = solve_shear_plane_temp(math.tan(phi), vs / (v * sphi), eps, epsdot, case, mat, cp_T, cp_v, k_T, k_v, prm,
                                  aux)
    k_ab = aux[0]
    if not k_ab > 0.0:
        return 0
    fs = k_ab * l_ab
    neq = jc_hardening_index(eps, mat)
    if lam_given < NO_REF + 1.0:
        tth = 1.0 + HALF_PI - 2.0 * phi - c0 * neq
        if tth <= 0.0:
            return 0
        theta = math.atan(tth)
        lam = theta + alpha - phi
    else:
        lam = lam_given
        theta = phi + lam - alpha
        if theta <= 0.0 or theta >= HALF_PI:
            return 0
        tth = math.tan(theta)
    if lam <= 0.0 or lam >= HALF_PI:
        return 0
    res = fs / math.cos(theta)
    lc = t1 * math.sin(theta) / (math.cos(lam) * sphi) * (1.0 + c0 * neq / (3.0 * tth))
    if not lc > 0.0:
        return 0
    st[ST_PHI] = phi
    st[ST_C0] = c0
    st[ST_TAB] = t_ab
    st[ST_KAB] = k_ab
    st[ST_NEQ] = neq
    st[ST_FS] = fs
    st[ST_THETA] = theta
    st[ST_LAMBDA] = lam
    st[ST_FC] = res * math.cos(lam - alpha)
    st[ST_FT] = res * math.sin(lam - alpha)
    st[ST_F] = res * math.sin(lam)
    st[ST_N] = res * math.cos(lam)
    st[ST_LC] = lc
    st[ST_SIGNP] = k_ab * (1.0 + HALF_PI - 2.0 * alpha - 2.0 * c0 * neq)
    st[ST_SIGN] = st[ST_N] / lc
    st[ST_TAUINT] = st[ST_F] / lc
    st[ST_VS] = vs
    st[ST_VC] = vc
    st[ST_T2] = t2
    st[ST_GAMMA] = gamma
    st[ST_EPS] = eps
    st[ST_EPSDOT] = epsdot
    st[ST_DTSZ] = aux[2]
    st[ST_BETA] = aux[1]
    st[ST_CAPPED] = aux[3]
    st[ST_G2] = st[ST_SIGNP] - st[ST_SIGN]
    return 1


@njit("int64(float64, float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1],"
      " float64[::1], float64[::1])", **_OPTS)
def solve_strain_rate_constant(phi, case, mat, cp_T, cp_v, k_T, k_v, prm, st):
    """C0 at shear angle ``phi`` from Oxley's condition sigma_N' = sigma_N (normal stress at B from
    the shear-plane field = mean rake normal stress). Roots in [C0_min, C0_max] are found by a
    fixed scan and Brent's method; several roots: the smallest cutting force. No root: the C0
    minimising the mismatch, accepted if below c0_residual_max x k_AB. Returns 1 when found."""
    n = int(prm[P_C0_N])
    lo = prm[P_C0_MIN]
    hi = prm[P_C0_MAX]
    tmp = np.zeros(N_ST)
    bs = np.zeros(N_BRENT)
    have_prev = False
    prev_c = 0.0
    prev_g = 0.0
    best_fc = np.inf
    best_c = -1.0
    best_res = np.inf
    best_res_c = -1.0
    for j in range(n):
        c = lo + (hi - lo) * j / (n - 1)
        if shear_plane_state(phi, c, NO_REF, case, mat, cp_T, cp_v, k_T, k_v, prm, tmp) == 1:
            g = tmp[ST_G2]
            res = abs(g) / tmp[ST_KAB]
            if res < best_res:
                best_res = res
                best_res_c = c
            if have_prev and ((g <= 0.0 <= prev_g) or (prev_g <= 0.0 <= g)):
                brent_init(bs, prev_c, c, prev_g, g, prm[P_TOL_C0])
                ok = True
                for _ in range(int(prm[P_ITER_MAX])):
                    if brent_step(bs) == 1:
                        break
                    if shear_plane_state(phi, bs[_BB], NO_REF, case, mat, cp_T, cp_v, k_T, k_v, prm, tmp) != 1:
                        ok = False
                        break
                    bs[_BFB] = tmp[ST_G2]
                if ok and shear_plane_state(phi, bs[_BB], NO_REF, case, mat, cp_T, cp_v, k_T, k_v, prm, tmp) == 1:
                    if tmp[ST_FC] < best_fc:
                        best_fc = tmp[ST_FC]
                        best_c = bs[_BB]
            have_prev = True
            prev_c = c
            prev_g = g
        else:
            have_prev = False
    if best_c > 0.0:
        shear_plane_state(phi, best_c, NO_REF, case, mat, cp_T, cp_v, k_T, k_v, prm, st)
        st[ST_C0RES] = abs(st[ST_G2]) / st[ST_KAB]
        return 1
    if best_res_c > 0.0 and best_res <= prm[P_C0_RES_MAX]:
        step = (hi - lo) / (n - 1)
        a = best_res_c - step if best_res_c - step > lo else lo
        b = best_res_c + step if best_res_c + step < hi else hi
        x1 = b - GOLDEN * (b - a)
        x2 = a + GOLDEN * (b - a)
        f1 = np.inf
        f2 = np.inf
        if shear_plane_state(phi, x1, NO_REF, case, mat, cp_T, cp_v, k_T, k_v, prm, tmp) == 1:
            f1 = abs(tmp[ST_G2]) / tmp[ST_KAB]
        if shear_plane_state(phi, x2, NO_REF, case, mat, cp_T, cp_v, k_T, k_v, prm, tmp) == 1:
            f2 = abs(tmp[ST_G2]) / tmp[ST_KAB]
        for _ in range(int(prm[P_ITER_MAX])):
            if b - a <= prm[P_TOL_C0]:
                break
            if f1 <= f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = b - GOLDEN * (b - a)
                f1 = np.inf
                if shear_plane_state(phi, x1, NO_REF, case, mat, cp_T, cp_v, k_T, k_v, prm, tmp) == 1:
                    f1 = abs(tmp[ST_G2]) / tmp[ST_KAB]
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + GOLDEN * (b - a)
                f2 = np.inf
                if shear_plane_state(phi, x2, NO_REF, case, mat, cp_T, cp_v, k_T, k_v, prm, tmp) == 1:
                    f2 = abs(tmp[ST_G2]) / tmp[ST_KAB]
        c_best = best_res_c
        f_best = best_res
        if f1 < f_best:
            c_best = x1
            f_best = f1
        if f2 < f_best:
            c_best = x2
            f_best = f2
        if shear_plane_state(phi, c_best, NO_REF, case, mat, cp_T, cp_v, k_T, k_v, prm, st) != 1:
            return 0
        st[ST_C0RES] = abs(st[ST_G2]) / st[ST_KAB]
        return 1
    return 0


# ============================================================================ Oxley: interface
@njit("float64(float64[::1], float64, float64[::1], float64[::1], float64[::1], float64[::1], float64[::1],"
      " float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1],"
      " float64[::1])", **_OPTS)
def interface_state(st, delta, case, mat, cp_T, cp_v, k_T, k_v, tool, kt_T, kt_v, ct_T, ct_v, prm, it):
    """Tool-chip interface at secondary-zone ratio ``delta`` for the shear-plane state ``st``.

    Chip temperatures after Boothroyd/Oxley with the share R of the rake friction heat that
    enters the chip; R equalises the mean interface temperature seen by the chip (moving
    source) and by the tool (stationary source at the tool rake temperature; Loewen & Shaw).
    The interface shear stress allowed is min(k_chip, mu sigma_N) (Coulomb-Tresca limit; mu <= 0
    gives Oxley's sticking condition). Returns g1 = tau_int - tau_lim."""
    t1 = case[CASE_T1_M]
    v = case[CASE_V_M_S]
    tw = case[CASE_TW_K]
    rho = mat[MAT_RHO_KG_M3]
    psi = prm[P_PSI]
    vc = st[ST_VC]
    t2 = st[ST_T2]
    lc = st[ST_LC]
    force = st[ST_F]
    dtsz = st[ST_DTSZ]
    eps_i = (2.0 * st[ST_GAMMA] + 0.5 * lc / (delta * t2)) / SQRT3
    epsdot_i = vc / (delta * t2) / SQRT3
    flux = force * vc / lc
    adiabatic = tool[TOOL_T_RAKE_K] <= 0.0
    share = 1.0
    tc = tw + dtsz
    dtc = 0.0
    dtm = 0.0
    tol = prm[P_TOL_T]
    for _ in range(int(prm[P_ITER_MAX])):
        cp = interp1(cp_T, cp_v, tc)
        dtc = force * vc / (rho * v * t1 * cp)
        tc_new = tw + dtsz + share * dtc
        cpn = interp1(cp_T, cp_v, tc_new)
        kth = interp1(k_T, k_v, tc_new)
        x = rho * cpn * v * t1 / kth * t2 / lc
        dtm = dtc * 10.0 ** (prm[P_DTM_A] - prm[P_DTM_B] * delta * math.sqrt(x)) * x ** prm[P_DTM_C]
        if dtm < dtc:
            dtm = dtc
        if adiabatic:
            share_new = 1.0
        else:
            guess = tw + dtsz + psi * share * dtm
            q_res = flux * tool_resistance(lc, guess, tool, kt_T, kt_v, ct_T, ct_v, prm)
            share_new = (tool[TOOL_T_RAKE_K] + q_res - tw - dtsz) / (psi * dtm + q_res)
            if share_new < 0.0:
                share_new = 0.0
            if share_new > 1.0:
                share_new = 1.0
        done = abs(tc_new - tc) <= tol and abs(share_new - share) <= tol * 1e-3
        tc = tc_new
        share = share_new
        if done:
            break
    cp = interp1(cp_T, cp_v, tc)
    dtc = force * vc / (rho * v * t1 * cp)
    tint = tw + dtsz + psi * share * dtm
    k_chip = jc_flow_stress(eps_i, epsdot_i, tint, mat, prm[P_TSTAR_MAX], prm[P_RATE_MIN]) / SQRT3
    tau_lim = k_chip
    regime = REGIME_STICKING
    if prm[P_MU] > 0.0:
        sliding = prm[P_MU] * st[ST_SIGN]
        if sliding < tau_lim:
            tau_lim = sliding
            regime = REGIME_SLIDING
    it[IT_TINT] = tint
    it[IT_TC] = tw + dtsz + share * dtc
    it[IT_DTC] = dtc
    it[IT_DTM] = dtm
    it[IT_R] = share
    it[IT_EPS] = eps_i
    it[IT_EPSDOT] = epsdot_i
    it[IT_KCHIP] = k_chip
    it[IT_TAULIM] = tau_lim
    it[IT_G1] = st[ST_TAUINT] - tau_lim
    it[IT_REGIME] = regime
    return it[IT_G1]


@njit("void(float64[::1], float64[::1], float64, float64[::1])", **_OPTS)
def blend_state(sa, sb, w, out):
    """Linear blend of two shear-plane states (w = 0: sa, w = 1: sb)."""
    for i in range(N_ST):
        out[i] = sa[i] + w * (sb[i] - sa[i])


@njit("void(float64[::1], float64[::1], float64, float64, float64[::1], float64[::1])", **_OPTS)
def fill_output(st, it, delta, status, case, out):
    """Write a solved state into the output vector (abi.OX_*)."""
    for i in range(N_OUT):
        out[i] = 0.0
    out[OX_STATUS] = status
    out[OX_REGIME] = it[IT_REGIME]
    out[OX_PHI] = st[ST_PHI]
    out[OX_C0] = st[ST_C0]
    out[OX_DELTA] = delta
    out[OX_FC] = st[ST_FC]
    out[OX_FT] = st[ST_FT]
    out[OX_F] = st[ST_F]
    out[OX_N] = st[ST_N]
    out[OX_FS] = st[ST_FS]
    out[OX_THETA] = st[ST_THETA]
    out[OX_LAMBDA] = st[ST_LAMBDA]
    out[OX_T2] = st[ST_T2]
    out[OX_CHIP_RATIO] = case[CASE_T1_M] / st[ST_T2]
    out[OX_LC] = st[ST_LC]
    out[OX_TAU_INT] = st[ST_TAUINT]
    out[OX_SIGMA_N] = st[ST_SIGN]
    out[OX_SIGMA_NP] = st[ST_SIGNP]
    out[OX_K_AB] = st[ST_KAB]
    out[OX_K_CHIP] = it[IT_KCHIP]
    out[OX_N_EQ] = st[ST_NEQ]
    out[OX_T_AB] = st[ST_TAB]
    out[OX_T_INT] = it[IT_TINT]
    out[OX_T_CHIP] = it[IT_TC]
    out[OX_DT_SZ] = st[ST_DTSZ]
    out[OX_DT_C] = it[IT_DTC]
    out[OX_DT_M] = it[IT_DTM]
    out[OX_BETA] = st[ST_BETA]
    out[OX_R_CHIP] = it[IT_R]
    out[OX_EPS_AB] = st[ST_EPS]
    out[OX_EPSDOT_AB] = st[ST_EPSDOT]
    out[OX_EPS_INT] = it[IT_EPS]
    out[OX_EPSDOT_INT] = it[IT_EPSDOT]
    out[OX_VS] = st[ST_VS]
    out[OX_VC] = st[ST_VC]
    out[OX_C0_RES] = st[ST_C0RES]
    out[OX_G1_RES] = abs(it[IT_G1]) / st[ST_KAB]
    out[OX_T_CAPPED] = st[ST_CAPPED]
    out[OX_P_SHEAR] = st[ST_FS] * st[ST_VS] / (st[ST_FC] * case[CASE_V_M_S])


# ============================================================================ Oxley: point solver
@njit("float64(float64, float64[::1], float64[:, ::1], int64[::1], float64[::1], float64[::1], float64[::1],"
      " float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1],"
      " float64[::1], float64[::1])", **_OPTS)
def phi_root_on_grid(delta, phis, states, valid, case, mat, cp_T, cp_v, k_T, k_v, tool, kt_T, kt_v, ct_T, ct_v, prm,
                     result):
    """Roots of the interface condition g1(phi) = 0 at ``delta`` using the shear-plane states on
    the phi grid (linear blend inside an interval). The root of smallest cutting force wins
    (larger phi on a tie). result = [phi, F_C, interval index]; returns F_C (inf: no root)."""
    n = phis.shape[0]
    g = np.zeros(n)
    it = np.zeros(N_IT)
    tmp = np.zeros(N_ST)
    bs = np.zeros(N_BRENT)
    for i in range(n):
        if valid[i] == 1:
            g[i] = interface_state(states[i], delta, case, mat, cp_T, cp_v, k_T, k_v, tool, kt_T, kt_v, ct_T, ct_v,
                                   prm, it)
    best_fc = np.inf
    best_phi = -1.0
    best_i = -1
    for i in range(1, n):
        if valid[i - 1] != 1 or valid[i] != 1:
            continue
        ga = g[i - 1]
        gb = g[i]
        if not ((ga <= 0.0 <= gb) or (gb <= 0.0 <= ga)):
            continue
        brent_init(bs, 0.0, 1.0, ga, gb, prm[P_TOL_PHI] / (phis[i] - phis[i - 1]))
        for _ in range(int(prm[P_ITER_MAX])):
            if brent_step(bs) == 1:
                break
            blend_state(states[i - 1], states[i], bs[_BB], tmp)
            bs[_BFB] = interface_state(tmp, delta, case, mat, cp_T, cp_v, k_T, k_v, tool, kt_T, kt_v, ct_T, ct_v, prm,
                                       it)
        w = bs[_BB]
        blend_state(states[i - 1], states[i], w, tmp)
        fc = tmp[ST_FC]
        phi = phis[i - 1] + w * (phis[i] - phis[i - 1])
        if fc < best_fc or (fc == best_fc and phi > best_phi):
            best_fc = fc
            best_phi = phi
            best_i = i
    result[0] = best_phi
    result[1] = best_fc
    result[2] = best_i
    return best_fc


@njit("void(float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1],"
      " float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1])", **_OPTS)
def merchant_point(case, mat, cp_T, cp_v, k_T, k_v, tool, kt_T, kt_v, ct_T, ct_v, prm, out):
    """Fallback when Oxley's conditions have no solution: modified Merchant relation
    2 phi + lambda - alpha = C_M with lambda and C_M of the nearest converged Oxley solution
    (prm P_REF_*), or Merchant's C_M = pi/2 with lambda = atan(mu) when there is none. The
    shear flow stress, temperatures and contact follow from the same shear-plane and interface
    models. A shear angle below the floor marks the condition as indentation."""
    alpha = case[CASE_ALPHA_RAD]
    if prm[P_REF_LAMBDA] > NO_REF + 1.0:
        lam = prm[P_REF_LAMBDA]
        c_m = prm[P_REF_CM]
    else:
        lam = math.atan(prm[P_MU_FALLBACK])
        c_m = HALF_PI
    c0 = prm[P_REF_C0] if prm[P_REF_C0] > 0.0 else prm[P_MERCHANT_C0]
    delta = prm[P_REF_DELTA] if prm[P_REF_DELTA] > 0.0 else prm[P_MERCHANT_DELTA]
    phi = 0.5 * (c_m - lam + alpha)
    for i in range(N_OUT):
        out[i] = 0.0
    out[OX_STATUS] = STATUS_INDENTATION
    if phi < prm[P_PHI_FLOOR]:
        return
    st = np.zeros(N_ST)
    if shear_plane_state(phi, c0, lam, case, mat, cp_T, cp_v, k_T, k_v, prm, st) != 1:
        return
    st[ST_C0RES] = 0.0
    it = np.zeros(N_IT)
    interface_state(st, delta, case, mat, cp_T, cp_v, k_T, k_v, tool, kt_T, kt_v, ct_T, ct_v, prm, it)
    fill_output(st, it, delta, STATUS_MERCHANT, case, out)
    out[OX_G1_RES] = 0.0


@njit("void(float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1],"
      " float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1])", **_OPTS)
def oxley_point(case, mat, cp_T, cp_v, k_T, k_v, tool, kt_T, kt_v, ct_T, ct_v, prm, out):
    """Solve Oxley's orthogonal theory at one cutting condition (case: t1, V, alpha, T_w).

    1. phi grid: C0(phi) from the hydrostatic-stress condition (shear-plane states kept).
    2. delta grid (log-spaced) and golden-section refinement: for each delta the phi root of
       the interface condition; delta minimises the cutting force (Oxley's criterion).
    3. Polish: exact Brent root in phi at the optimal delta with exact C0(phi).
    Falls back to :func:`merchant_point` when no solution exists. Results in ``out`` (OX_*)."""
    nphi = int(prm[P_PHI_N])
    phis = np.zeros(nphi)
    states = np.zeros((nphi, N_ST))
    valid = np.zeros(nphi, np.int64)
    st = np.zeros(N_ST)
    any_valid = False
    for i in range(nphi):
        phis[i] = prm[P_PHI_MIN] + (prm[P_PHI_MAX] - prm[P_PHI_MIN]) * i / (nphi - 1)
        if solve_strain_rate_constant(phis[i], case, mat, cp_T, cp_v, k_T, k_v, prm, st) == 1:
            valid[i] = 1
            any_valid = True
            for j in range(N_ST):
                states[i, j] = st[j]
    if not any_valid:
        merchant_point(case, mat, cp_T, cp_v, k_T, k_v, tool, kt_T, kt_v, ct_T, ct_v, prm, out)
        return
    nd = int(prm[P_DELTA_N])
    log_lo = math.log(prm[P_DELTA_MIN])
    log_hi = math.log(prm[P_DELTA_MAX])
    fcs = np.zeros(nd)
    res = np.zeros(3)
    best_j = -1
    best_fc = np.inf
    for j in range(nd):
        d = math.exp(log_lo + (log_hi - log_lo) * j / (nd - 1))
        fcs[j] = phi_root_on_grid(d, phis, states, valid, case, mat, cp_T, cp_v, k_T, k_v, tool, kt_T, kt_v, ct_T,
                                  ct_v, prm, res)
        if fcs[j] < best_fc:
            best_fc = fcs[j]
            best_j = j
    if best_j < 0:
        merchant_point(case, mat, cp_T, cp_v, k_T, k_v, tool, kt_T, kt_v, ct_T, ct_v, prm, out)
        return
    step = (log_hi - log_lo) / (nd - 1)
    a = log_lo + step * (best_j - 1) if best_j > 0 else log_lo
    b = log_lo + step * (best_j + 1) if best_j < nd - 1 else log_hi
    x_best = log_lo + step * best_j
    f_best = best_fc
    x1 = b - GOLDEN * (b - a)
    x2 = a + GOLDEN * (b - a)
    f1 = phi_root_on_grid(math.exp(x1), phis, states, valid, case, mat, cp_T, cp_v, k_T, k_v, tool, kt_T, kt_v, ct_T,
                          ct_v, prm, res)
    f2 = phi_root_on_grid(math.exp(x2), phis, states, valid, case, mat, cp_T, cp_v, k_T, k_v, tool, kt_T, kt_v, ct_T,
                          ct_v, prm, res)
    for _ in range(int(prm[P_DELTA_ITER])):
        if f1 <= f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - GOLDEN * (b - a)
            f1 = phi_root_on_grid(math.exp(x1), phis, states, valid, case, mat, cp_T, cp_v, k_T, k_v, tool, kt_T, kt_v,
                                  ct_T, ct_v, prm, res)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + GOLDEN * (b - a)
            f2 = phi_root_on_grid(math.exp(x2), phis, states, valid, case, mat, cp_T, cp_v, k_T, k_v, tool, kt_T, kt_v,
                                  ct_T, ct_v, prm, res)
    if f1 < f_best:
        x_best = x1
        f_best = f1
    if f2 < f_best:
        x_best = x2
        f_best = f2
    delta = math.exp(x_best)
    phi_root_on_grid(delta, phis, states, valid, case, mat, cp_T, cp_v, k_T, k_v, tool, kt_T, kt_v, ct_T, ct_v, prm,
                     res)
    phi_star = res[0]
    i = int(res[2])
    it = np.zeros(N_IT)
    # polish: exact root of g1 in phi with exact C0(phi), bracketed by the grid interval
    lo_phi = phis[i - 1]
    hi_phi = phis[i]
    polished = False
    stb = np.zeros(N_ST)
    if solve_strain_rate_constant(lo_phi, case, mat, cp_T, cp_v, k_T, k_v, prm, st) == 1 and \
            solve_strain_rate_constant(hi_phi, case, mat, cp_T, cp_v, k_T, k_v, prm, stb) == 1:
        ga = interface_state(st, delta, case, mat, cp_T, cp_v, k_T, k_v, tool, kt_T, kt_v, ct_T, ct_v, prm, it)
        gb = interface_state(stb, delta, case, mat, cp_T, cp_v, k_T, k_v, tool, kt_T, kt_v, ct_T, ct_v, prm, it)
        if (ga <= 0.0 <= gb) or (gb <= 0.0 <= ga):
            bs = np.zeros(N_BRENT)
            brent_init(bs, lo_phi, hi_phi, ga, gb, prm[P_TOL_PHI])
            polished = True
            for _ in range(int(prm[P_ITER_MAX])):
                if brent_step(bs) == 1:
                    break
                if solve_strain_rate_constant(bs[_BB], case, mat, cp_T, cp_v, k_T, k_v, prm, st) != 1:
                    polished = False
                    break
                bs[_BFB] = interface_state(st, delta, case, mat, cp_T, cp_v, k_T, k_v, tool, kt_T, kt_v, ct_T, ct_v,
                                           prm, it)
            if polished:
                phi_star = bs[_BB]
    if solve_strain_rate_constant(phi_star, case, mat, cp_T, cp_v, k_T, k_v, prm, st) != 1:
        blend_state(states[i - 1], states[i], (phi_star - lo_phi) / (hi_phi - lo_phi), st)
    interface_state(st, delta, case, mat, cp_T, cp_v, k_T, k_v, tool, kt_T, kt_v, ct_T, ct_v, prm, it)
    fill_output(st, it, delta, STATUS_OXLEY, case, out)


@njit("void(float64[::1], float64[::1], float64[::1], float64, float64, float64[::1], float64[::1], float64[::1],"
      " float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1],"
      " float64[::1], float64[:, :, :, ::1])", **_OPTS)
def oxley_map(t1_grid, v_grid, ttool_grid, alpha, work_temp_K, mat, cp_T, cp_v, k_T, k_v, tool, kt_T, kt_v, ct_T,
              ct_v, prm, values):
    """Fill values[iT, iv, ih, :] with :func:`oxley_point` at every grid node. Nodes without an
    Oxley solution are then solved by :func:`merchant_point` with the friction angle, Merchant
    constant, C0 and delta of the nearest converged node (same row first, then the whole grid,
    by index distance; ties to the lower index). ``tool[TOOL_T_RAKE_K] <= 0`` keeps the tool
    adiabatic (the T grid then has one node)."""
    n_t = ttool_grid.shape[0]
    n_v = v_grid.shape[0]
    n_h = t1_grid.shape[0]
    case = np.zeros(N_CASE)
    tl = tool.copy()
    pr = prm.copy()
    pr[P_REF_LAMBDA] = NO_REF
    pr[P_REF_CM] = 0.0
    pr[P_REF_C0] = 0.0
    pr[P_REF_DELTA] = 0.0
    out = np.zeros(N_OUT)
    case[CASE_ALPHA_RAD] = alpha
    case[CASE_TW_K] = work_temp_K
    for it_ in range(n_t):
        if tool[TOOL_T_RAKE_K] > 0.0:
            tl[TOOL_T_RAKE_K] = ttool_grid[it_]
        for iv in range(n_v):
            case[CASE_V_M_S] = v_grid[iv]
            for ih in range(n_h):
                case[CASE_T1_M] = t1_grid[ih]
                oxley_point(case, mat, cp_T, cp_v, k_T, k_v, tl, kt_T, kt_v, ct_T, ct_v, pr, out)
                for q in range(N_OUT):
                    values[it_, iv, ih, q] = out[q]
    for it_ in range(n_t):
        if tool[TOOL_T_RAKE_K] > 0.0:
            tl[TOOL_T_RAKE_K] = ttool_grid[it_]
        for iv in range(n_v):
            case[CASE_V_M_S] = v_grid[iv]
            for ih in range(n_h):
                if values[it_, iv, ih, OX_STATUS] == STATUS_OXLEY:
                    continue
                best = -1
                bt = 0
                bv = 0
                bh = 0
                for jt in range(n_t):
                    for jv in range(n_v):
                        for jh in range(n_h):
                            if values[jt, jv, jh, OX_STATUS] != STATUS_OXLEY:
                                continue
                            dist = abs(jh - ih) + n_h * (abs(jv - iv) + n_v * abs(jt - it_))
                            if best < 0 or dist < best:
                                best = dist
                                bt = jt
                                bv = jv
                                bh = jh
                if best >= 0:
                    lam = values[bt, bv, bh, OX_LAMBDA]
                    pr[P_REF_LAMBDA] = lam
                    pr[P_REF_CM] = 2.0 * values[bt, bv, bh, OX_PHI] + lam - alpha
                    pr[P_REF_C0] = values[bt, bv, bh, OX_C0]
                    pr[P_REF_DELTA] = values[bt, bv, bh, OX_DELTA]
                else:
                    pr[P_REF_LAMBDA] = NO_REF
                    pr[P_REF_CM] = 0.0
                    pr[P_REF_C0] = 0.0
                    pr[P_REF_DELTA] = 0.0
                case[CASE_T1_M] = t1_grid[ih]
                merchant_point(case, mat, cp_T, cp_v, k_T, k_v, tl, kt_T, kt_v, ct_T, ct_v, pr, out)
                for q in range(N_OUT):
                    values[it_, iv, ih, q] = out[q]


# ============================================================================ table lookups
@njit("void(float64[::1], float64[::1], float64[::1], float64[:, :, :, ::1], int64[::1], int64[::1], float64[::1],"
      " float64, float64, float64[:, ::1])", **_OPTS)
def map_lookup(t1_grid, v_grid, ttool_grid, values, scaled, discrete, t1_query, speed, ttool, out):
    """Trilinear interpolation of a process map in (log t1, log V, T_tool) at every t1 of
    ``t1_query`` for one speed and tool temperature; out[k, :] is an OX_* vector.

    Slots flagged in ``scaled`` (forces, lengths) are interpolated divided by t1 and multiplied
    back, so below the first node they scale linearly with t1; slots in ``discrete`` take the
    node of largest weight. Outside the grid the end nodes are held; t1 <= 0 gives zeros."""
    n_t = ttool_grid.shape[0]
    n_v = v_grid.shape[0]
    n_h = t1_grid.shape[0]
    nq = values.shape[3]
    # speed weights (log)
    iv0 = 0
    wv = 0.0
    if n_v > 1:
        ls = math.log(speed) if speed > 0.0 else -1e300
        if ls <= math.log(v_grid[0]):
            iv0 = 0
            wv = 0.0
        elif ls >= math.log(v_grid[n_v - 1]):
            iv0 = n_v - 2
            wv = 1.0
        else:
            iv0 = 0
            while iv0 < n_v - 2 and math.log(v_grid[iv0 + 1]) <= ls:
                iv0 += 1
            wv = (ls - math.log(v_grid[iv0])) / (math.log(v_grid[iv0 + 1]) - math.log(v_grid[iv0]))
    it0 = 0
    wt = 0.0
    if n_t > 1:
        if ttool <= ttool_grid[0]:
            it0 = 0
            wt = 0.0
        elif ttool >= ttool_grid[n_t - 1]:
            it0 = n_t - 2
            wt = 1.0
        else:
            it0 = 0
            while it0 < n_t - 2 and ttool_grid[it0 + 1] <= ttool:
                it0 += 1
            wt = (ttool - ttool_grid[it0]) / (ttool_grid[it0 + 1] - ttool_grid[it0])
    is_scaled = np.zeros(nq, np.int64)
    is_discrete = np.zeros(nq, np.int64)
    for s in range(scaled.shape[0]):
        is_scaled[scaled[s]] = 1
    for s in range(discrete.shape[0]):
        is_discrete[discrete[s]] = 1
    for k in range(t1_query.shape[0]):
        t1 = t1_query[k]
        for q in range(nq):
            out[k, q] = 0.0
        if not t1 > 0.0:
            continue
        ih0 = 0
        wh = 0.0
        if n_h > 1:
            lh = math.log(t1)
            if lh <= math.log(t1_grid[0]):
                ih0 = 0
                wh = 0.0
            elif lh >= math.log(t1_grid[n_h - 1]):
                ih0 = n_h - 2
                wh = 1.0
            else:
                ih0 = 0
                while ih0 < n_h - 2 and math.log(t1_grid[ih0 + 1]) <= lh:
                    ih0 += 1
                wh = (lh - math.log(t1_grid[ih0])) / (math.log(t1_grid[ih0 + 1]) - math.log(t1_grid[ih0]))
        best_w = -1.0
        bt = 0
        bv = 0
        bh = 0
        for dt in range(2 if n_t > 1 else 1):
            w_t = (wt if dt == 1 else 1.0 - wt) if n_t > 1 else 1.0
            jt = it0 + dt
            for dv in range(2 if n_v > 1 else 1):
                w_v = (wv if dv == 1 else 1.0 - wv) if n_v > 1 else 1.0
                jv = iv0 + dv
                for dh in range(2 if n_h > 1 else 1):
                    w_h = (wh if dh == 1 else 1.0 - wh) if n_h > 1 else 1.0
                    jh = ih0 + dh
                    w = w_t * w_v * w_h
                    if w > best_w:
                        best_w = w
                        bt = jt
                        bv = jv
                        bh = jh
                    if w == 0.0:
                        continue
                    for q in range(nq):
                        if is_discrete[q] == 1:
                            continue
                        val = values[jt, jv, jh, q]
                        if is_scaled[q] == 1:
                            val = val / t1_grid[jh]
                        out[k, q] += w * val
        for q in range(nq):
            if is_discrete[q] == 1:
                out[k, q] = values[bt, bv, bh, q]
            elif is_scaled[q] == 1:
                out[k, q] = out[k, q] * t1


@njit("float64(float64[::1], float64[::1], float64)", **_OPTS)
def interp_log_h(h_grid, row, h):
    """Value of a coefficient row at chip thickness h: linear in log h between nodes, end values
    held outside (the hot loop's own interpolation of ProcessCoefficients rows is the same)."""
    n = h_grid.shape[0]
    if n == 1 or h <= h_grid[0]:
        return row[0]
    if h >= h_grid[n - 1]:
        return row[n - 1]
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if h_grid[mid] <= h:
            lo = mid
        else:
            hi = mid
    t = (math.log(h) - math.log(h_grid[lo])) / (math.log(h_grid[hi]) - math.log(h_grid[lo]))
    return row[lo] + t * (row[hi] - row[lo])


@njit("float64(float64[::1], float64[::1], float64)", **_OPTS)
def interp_lin_h(h_grid, row, h):
    """Value of a coefficient row at chip thickness h, linear in h between nodes (no
    transcendental function: usable at L0), end values held outside."""
    n = h_grid.shape[0]
    if n == 1 or h <= h_grid[0]:
        return row[0]
    if h >= h_grid[n - 1]:
        return row[n - 1]
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if h_grid[mid] <= h:
            lo = mid
        else:
            hi = mid
    t = (h - h_grid[lo]) / (h_grid[hi] - h_grid[lo])
    return row[lo] + t * (row[hi] - row[lo])


# ============================================================================ edge contact
@njit("void(float64, float64, float64, float64, float64, float64, float64, float64[::1])", **_OPTS)
def edge_contact_forces(depth_m, radius_m, stag_m, rake_rad, incl_rad, pressure_Pa, tau_Pa, out):
    """Forces per unit edge length of a rounded edge (radius r, normal rake alpha, inclination i)
    pressed ``depth_m`` into the work (2-D normal-plane geometry; the arc runs from the lowest
    point, normal angle 0, to the rake-face tangency at pi/2 + alpha, then the rake face).

    Uniform pressure p and friction stress tau act on the contact. Material below the
    stagnation height ``stag_m`` flows down under the edge, above it up the rake face; the
    friction on the tool follows that flow in the normal plane (scaled by cos i) and the
    along-edge slip (sin i). out: PL_FT tangential (opposing the cutting velocity), PL_FR thrust
    (pushing the tool off the surface), PL_FA along the axis normal to the velocity in the cut
    surface, PL_CONTACT contact length, PL_AREA interference area below the uncut surface."""
    for i in range(out.shape[0]):
        out[i] = 0.0
    if not depth_m > 0.0:
        return
    r = radius_m
    w_r = HALF_PI + rake_rad                     # normal angle at the rake tangency
    y_r = r * (1.0 - math.cos(w_r))              # height of the tangency above the lowest point
    ci = math.cos(incl_rad)
    si = math.sin(incl_rad)
    # contact ends
    if r > 0.0 and depth_m < y_r:
        w_d = math.acos(1.0 - depth_m / r)
        face = 0.0
    else:
        w_d = w_r if r > 0.0 else 0.0
        face = (depth_m - y_r) / math.cos(rake_rad)
    if r > 0.0 and stag_m < y_r:
        w_s = math.acos(1.0 - stag_m / r) if stag_m > 0.0 else 0.0
        face_s = 0.0
    else:
        w_s = w_r if r > 0.0 else 0.0
        face_s = (stag_m - y_r) / math.cos(rake_rad)
    # arc below the stagnation point (downward flow), [0, min(w_d, w_s)]
    w1 = w_d if w_d < w_s else w_s
    s_sin = (1.0 - math.cos(w1)) * r             # integral of sin w ds
    s_cos = math.sin(w1) * r                     # integral of cos w ds
    fx = -pressure_Pa * s_sin - tau_Pa * ci * s_cos
    fy = pressure_Pa * s_cos - tau_Pa * ci * s_sin
    length = r * w1
    # arc above the stagnation point (upward flow), [w_s, w_d]
    if w_d > w_s:
        s_sin = (math.cos(w_s) - math.cos(w_d)) * r
        s_cos = (math.sin(w_d) - math.sin(w_s)) * r
        fx += -pressure_Pa * s_sin + tau_Pa * ci * s_cos
        fy += pressure_Pa * s_cos + tau_Pa * ci * s_sin
        length += r * (w_d - w_s)
    # rake face: below the stagnation height (downward) then above it (upward)
    if face > 0.0:
        sn = math.sin(w_r)
        cs = math.cos(w_r)
        down = face_s if face_s < face else face
        if down < 0.0:
            down = 0.0
        up = face - down
        fx += -pressure_Pa * sn * face - tau_Pa * ci * cs * down + tau_Pa * ci * cs * up
        fy += pressure_Pa * cs * face - tau_Pa * ci * sn * down + tau_Pa * ci * sn * up
        length += face
    f_along_edge = -tau_Pa * si * length         # slip along the edge opposes the work's along-edge motion
    out[PL_FT] = -(fx * ci + f_along_edge * si)
    out[PL_FR] = fy
    out[PL_FA] = -fx * si + f_along_edge * ci
    out[PL_CONTACT] = length
    # interference area: region between the uncut surface (y = depth) and the edge arc ahead of
    # the lowest point, plus (negative rake only) the triangle under a forward-leaning rake face
    area = 0.0
    if r > 0.0:
        area = (depth_m - r) * r * math.sin(w_d) + r * r * (0.5 * w_d + 0.25 * math.sin(2.0 * w_d))
    if face > 0.0 and rake_rad < 0.0:
        rise = depth_m - y_r
        area += 0.5 * rise * rise * abs(math.tan(rake_rad))
    out[PL_AREA] = area if area > 0.0 else 0.0


# ============================================================================ oxide and BUE
@njit("float64(float64[::1], float64, float64)", **_OPTS)
def oxide_exposure(temps_K, dt_s, activation_K):
    """Arrhenius exposure integral of exp(-Q/(R T)) dt (s) of a temperature history sampled
    every ``dt_s`` (trapezoidal rule); the parabolic oxide thickness grows as its square root."""
    n = temps_K.shape[0]
    if n == 0:
        return 0.0
    total = 0.0
    prev = math.exp(-activation_K / temps_K[0]) if temps_K[0] > 0.0 else 0.0
    for i in range(1, n):
        cur = math.exp(-activation_K / temps_K[i]) if temps_K[i] > 0.0 else 0.0
        total += 0.5 * (prev + cur) * dt_s
        prev = cur
    return total


@njit("void(float64[::1], float64[::1], float64[::1], float64, float64[::1], int64, int64, int64[::1], int64[::1],"
      " int64[::1])", **_OPTS)
def bue_step(height_m, in_window, uncut_m, dt_s, bue_prm, seed, stream, key_a, counter, broke):
    """Advance built-up-edge heights by dt (exact exponential relaxation): toward
    sat_ratio x uncut thickness inside the window (time constant growth_time), toward 0 outside
    (decay_time). Break-off hazard rate x height / saturation height inside the window: one
    counter-based draw per element and step (core.rng.hash_uniform(seed, stream, key_a, counter, 0),
    counter incremented); a break leaves residual x height. bue_prm = [sat_ratio, growth_time_s,
    decay_time_s, breakoff_rate_1_s, residual_fraction]."""
    sat_ratio = bue_prm[0]
    growth = bue_prm[1]
    decay = bue_prm[2]
    rate = bue_prm[3]
    residual = bue_prm[4]
    for i in range(height_m.shape[0]):
        broke[i] = 0
        h = height_m[i]
        if in_window[i] > 0.5 and uncut_m[i] > 0.0:
            sat = sat_ratio * uncut_m[i]
            hazard = rate * (h / sat if sat > 0.0 else 0.0)
            p_break = -math.expm1(-hazard * dt_s)
            u = hash_uniform(seed, stream, key_a[i], counter[i], 0)
            counter[i] += 1
            h = sat + (h - sat) * math.exp(-dt_s / growth)
            if u < p_break:
                h = residual * h
                broke[i] = 1
        else:
            h = h * math.exp(-dt_s / decay)
        height_m[i] = h
