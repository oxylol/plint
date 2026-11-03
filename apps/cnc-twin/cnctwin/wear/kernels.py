"""Numba kernels of the wear domain (docs/models/wear.md; rules: architecture 5.5).

Every kernel is compiled eagerly from one explicit signature (``cache=True, nogil=True,
error_model="numpy"``, no fastmath, no parallel) and has a plain-Python twin in
:mod:`cnctwin.wear.reference` that the tests compare it with.

* :func:`advance_elements` - one macro step of the per-element wear of one tool: flank wear depth by
  Usui's law evaluated at every in-cut quadrature temperature (never at the mean temperature) plus the
  Archard abrasive term, coating first and substrate when worn through, skin and BUE factors, BUE
  attrition, notch at the depth-of-cut line, crater depth on the rake, thermal-cycle fatigue (Basquin,
  Miner) and comb-crack depth.
* :func:`element_outputs` - the bus arrays (VB, notch land, radial/axial wear, worn hone radius) from
  the state.
* :func:`chip_trials` - per entry/exit shock row: Flamant wedge stress at the end of the contact,
  Weibull chipping probability with its effective stressed volume (plus the comb-crack term), a
  counter-based hash draw (stream ``wear.chipping``), and the coating spallation trial (stream
  ``wear.coating``).
* helpers: :func:`usui_rate`, :func:`table_at`, :func:`worn_edge_radius`, :func:`wedge_tension`,
  :func:`wedge_volume_factor`.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, types

from ..core.rng import hash_uniform
from . import abi
from .abi import (CL_BETA_MAX, CL_BETA_MIN, CL_CHIP_RATIO, CL_COAT_M, CL_COAT_SIGMA0_PA, CL_EXIT_FACTOR, CL_KIC,
                  CL_LC_RATIO, CL_M, CL_MU, CL_NTHETA, CL_ROUNDING, CL_SIGMA0_PA, CL_VREF_M3, CL_Y, CO_CHIPPED,
                  CO_DEPTH_M, CO_P, CO_P_COAT, CO_SIGMA_PA, CO_SPALLED, CO_VEFF_M3, IS_DT_S, KEY_SLOT, KEY_TOOTH,
                  LS_ABR, LS_BUE_PROTECT, LS_COAT_FACTOR, LS_COAT_TMAX_K, LS_CRACK_A0_M, LS_CRACK_GROWTH,
                  LS_DOC_RATIO, LS_FAT_B, LS_FAT_SIGMA_PA, LS_NOTCH_EXCESS, LS_PLUCK_M, LS_ROUNDING, LS_SKIN_FACTOR,
                  OS_COAT_THROUGH, OS_CRACK_INIT, OS_MAX_D, OS_MAX_RATE_M_S, OS_RAKE_THROUGH, OS_SUM_DVN, SH_ELEM,
                  SH_FORCE, SH_H, SH_KIND, SH_TOOTH, SHOCK_ENTRY)

OPTS = dict(cache=True, nogil=True, error_model="numpy")

f8 = types.float64
i8 = types.int64
F1 = types.float64[::1]
F2 = types.float64[:, ::1]
I1 = types.int64[::1]

GEO_T = abi.GEO_T
IN_T = abi.IN_T
LAW_T = abi.LAW_T
ST_T = abi.STATE_T
OUT_T = abi.OUT_T
CHIP_T = abi.CHIPLAW_T


# ====================================================================== helpers

@njit(f8(F1, F1, F1, f8), **OPTS)
def usui_rate(A, B, lo, temp_K):
    """A exp(-B / T) of the branch in force at T: branches sorted by the lower end of their range,
    the last one whose lower end is <= T (below every range the first). 0 without branches."""
    n = A.shape[0]
    if n == 0 or temp_K <= 0.0:
        return 0.0
    b = 0
    for i in range(n):
        if temp_K >= lo[i]:
            b = i
    return A[b] * math.exp(-B[b] / temp_K)


@njit(f8(F1, F1, f8), **OPTS)
def table_at(xs, ys, x):
    """Piecewise-linear table, end values held outside."""
    n = xs.shape[0]
    if n == 0:
        return 0.0
    if x <= xs[0]:
        return ys[0]
    if x >= xs[n - 1]:
        return ys[n - 1]
    i = 1
    while xs[i] < x:
        i += 1
    t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
    return ys[i - 1] + t * (ys[i] - ys[i - 1])


@njit(f8(f8, f8, f8, f8, i8, f8), **OPTS)
def worn_edge_radius(r_e0, vn, vn_notch, chip_d, chipped, rounding):
    """Hone radius grown by the edge recession; a chipped edge is at least half the chip depth blunt."""
    radius = r_e0 + rounding * (vn + vn_notch)
    if chipped != 0 and 0.5 * chip_d > radius:
        radius = 0.5 * chip_d
    return radius


@njit(types.UniTuple(f8, 3)(f8, f8, f8, f8, f8, f8), **OPTS)
def wedge_tension(force_per_m, contact_m, beta, rake, clear, mu):
    """Largest radial tensile stress of a wedge loaded at its edge (Flamant-Carothers, Timoshenko &
    Goodier 1970 section 36), at r = contact length, and the coefficients (c_x, c_y) of
    sigma_r(r, theta) = -(c_x cos theta + c_y sin theta) / r (theta from the bisector, + toward the rake).
    The force lies at lambda - rake above the cutting direction (lambda = atan mu)."""
    if force_per_m <= 0.0 or contact_m <= 0.0 or beta <= 0.0:
        return 0.0, 0.0, 0.0
    psi = math.atan(mu) - rake
    bisector = 0.5 * (clear + 0.5 * math.pi - rake)
    delta = psi - bisector
    p_x = force_per_m * math.cos(delta)
    p_y = force_per_m * math.sin(delta)
    c_x = 2.0 * p_x / (beta + math.sin(beta))
    c_y = 2.0 * p_y / (beta - math.sin(beta))
    a = 0.5 * beta
    s_rake = -(c_x * math.cos(a) + c_y * math.sin(a)) / contact_m
    s_flank = -(c_x * math.cos(a) - c_y * math.sin(a)) / contact_m
    sigma = s_rake if s_rake > s_flank else s_flank
    if sigma < 0.0:
        sigma = 0.0
    return sigma, c_x, c_y


@njit(f8(f8, f8, f8, f8, i8), **OPTS)
def wedge_volume_factor(c_x, c_y, beta, m, n):
    """Angular part of the Weibull effective volume: int over the wedge of (max(0, g) / g_max)^m dtheta,
    g = -(c_x cos theta + c_y sin theta), on a grid clustered at both faces (smoothstep map, midpoint
    rule, n nodes). The radial part is l_c^2 / (m - 2)."""
    a = 0.5 * beta
    g_lo = -(c_x * math.cos(-a) + c_y * math.sin(-a))
    g_hi = -(c_x * math.cos(a) + c_y * math.sin(a))
    g_max = g_lo if g_lo > g_hi else g_hi
    for i in range(n):
        u = (i + 0.5) / n
        theta = -a + 2.0 * a * (3.0 * u * u - 2.0 * u * u * u)
        g = -(c_x * math.cos(theta) + c_y * math.sin(theta))
        if g > g_max:
            g_max = g
    if g_max <= 0.0:
        return 0.0
    total = 0.0
    for i in range(n):
        u = (i + 0.5) / n
        theta = -a + 2.0 * a * (3.0 * u * u - 2.0 * u * u * u)
        jac = 2.0 * a * 6.0 * u * (1.0 - u)
        g = -(c_x * math.cos(theta) + c_y * math.sin(theta))
        if g > 0.0:
            total += (g / g_max) ** m * jac / n
    return total


# ====================================================================== wear of the elements

@njit(types.void(GEO_T, IN_T, LAW_T, ST_T, OUT_T), **OPTS)
def advance_elements(geo, inp, law, st, out):
    """One macro step of the per-element wear of one tool (see the module doc); updates ``st`` in
    place, writes the step increments and the depth-of-cut flags into ``out``."""
    nt = geo.valid.shape[0]
    ne = geo.valid.shape[1]
    nq = inp.w_q.shape[0]
    s = law.scal
    dt = inp.scal[IS_DT_S]
    for i in range(out.scal.shape[0]):
        out.scal[i] = 0.0
    for j in range(nt):
        # depth-of-cut line: ends of contiguous engaged runs where the edge continues into air
        for k in range(ne):
            out.doc[j, k] = 0
        k = 0
        while k < ne:
            if geo.valid[j, k] == 0 or inp.in_cut_s[j, k] <= 0.0:
                k += 1
                continue
            k0 = k
            h_top = 0.0
            while k < ne and geo.valid[j, k] != 0 and inp.in_cut_s[j, k] > 0.0:
                h_mean = inp.h_int[j, k] / inp.in_cut_s[j, k]
                if h_mean > h_top:
                    h_top = h_mean
                k += 1
            k1 = k - 1
            if h_top > 0.0:
                if k1 + 1 < ne and geo.valid[j, k1 + 1] != 0:
                    if inp.h_int[j, k1] / inp.in_cut_s[j, k1] >= s[LS_DOC_RATIO] * h_top:
                        out.doc[j, k1] = 1
                if k0 >= 1 and geo.valid[j, k0 - 1] != 0:
                    if inp.h_int[j, k0] / inp.in_cut_s[j, k0] >= s[LS_DOC_RATIO] * h_top:
                        out.doc[j, k0] = 1
        for k in range(ne):
            out.d_vn[j, k] = 0.0
            out.d_notch[j, k] = 0.0
            out.d_kt[j, k] = 0.0
            out.d_D[j, k] = 0.0
            if geo.valid[j, k] == 0:
                continue
            t_in = inp.in_cut_s[j, k]
            if t_in <= 0.0:
                continue
            h_mean = inp.h_int[j, k] / t_in
            m_skin = 1.0 + inp.skin_frac[j, k] * (s[LS_SKIN_FACTOR] - 1.0)
            m_bue = 1.0
            if st.bue_m[j, k] > 0.0 and h_mean > 0.0:
                cover = st.bue_m[j, k] / h_mean
                if cover > 1.0:
                    cover = 1.0
                m_bue = 1.0 - s[LS_BUE_PROTECT] * cover
            coated_f = st.coat_f[j, k] > 0.0
            coated_r = st.coat_r[j, k] > 0.0
            rate_f = 0.0
            rate_r = 0.0
            for q in range(nq):
                w = inp.w_q[q]
                t_f = inp.flank_T[j, k, q]
                t_r = inp.rake_T[j, k, q]
                u_f = usui_rate(law.fl_A, law.fl_B, law.fl_lo, t_f)
                u_r = usui_rate(law.rk_A, law.rk_B, law.rk_lo, t_r)
                if coated_f and t_f <= s[LS_COAT_TMAX_K]:
                    u_f *= s[LS_COAT_FACTOR]
                if coated_r and t_r <= s[LS_COAT_TMAX_K]:
                    u_r *= s[LS_COAT_FACTOR]
                h_f = table_at(law.hc_T, law.hc_Pa, t_f) if coated_f else table_at(law.hs_T, law.hs_Pa, t_f)
                h_r = table_at(law.hc_T, law.hc_Pa, t_r) if coated_r else table_at(law.hs_T, law.hs_Pa, t_r)
                a_f = s[LS_ABR] / h_f if h_f > 0.0 else 0.0
                a_r = s[LS_ABR] / h_r if h_r > 0.0 else 0.0
                rate_f += w * (u_f + a_f)
                rate_r += w * (u_r + a_r)
            d_vn = rate_f * inp.sv_flank[j, k] * m_skin * m_bue + inp.bue_broke[j, k] * s[LS_PLUCK_M]
            d_kt = rate_r * inp.sv_rake[j, k] * m_skin * m_bue
            d_notch = d_vn * s[LS_NOTCH_EXCESS] if out.doc[j, k] != 0 else 0.0
            # thermal-cycle fatigue: fully reversed surface stress alpha E dT / (2 (1 - nu)) per entry
            d_D = 0.0
            n_cyc = inp.entries[j, k]
            if n_cyc > 0.0 and inp.cycle_dT[j, k] > 0.0 and s[LS_FAT_SIGMA_PA] > 0.0 and s[LS_FAT_B] < 0.0:
                amp = inp.therm_coef[j, k] * inp.cycle_dT[j, k]
                if amp > 0.0:
                    n_fail = 0.5 * (amp / s[LS_FAT_SIGMA_PA]) ** (1.0 / s[LS_FAT_B])
                    d_D = n_cyc / n_fail
            # state
            st.vn[j, k] += d_vn
            st.vn_notch[j, k] += d_notch
            st.kt[j, k] += d_kt
            if coated_f:
                left = st.coat_f[j, k] - d_vn * geo.c_coat[j, k]
                if left <= 0.0:
                    left = 0.0
                    out.scal[OS_COAT_THROUGH] += 1.0
                st.coat_f[j, k] = left
            if coated_r:
                left = st.coat_r[j, k] - d_kt
                if left <= 0.0:
                    left = 0.0
                    out.scal[OS_RAKE_THROUGH] += 1.0
                st.coat_r[j, k] = left
            d_old = st.D_th[j, k]
            st.D_th[j, k] = d_old + d_D
            if st.D_th[j, k] >= 1.0:
                if d_old < 1.0:
                    out.scal[OS_CRACK_INIT] += 1.0
                a0 = s[LS_CRACK_A0_M]
                depth = a0
                if n_cyc > 0.0:
                    penetration = math.sqrt(inp.diffusivity[j, k] * (dt / n_cyc) / math.pi)
                    if penetration > a0:
                        depth = a0 + (penetration - a0) * (1.0 - math.exp(-(st.D_th[j, k] - 1.0) / s[LS_CRACK_GROWTH]))
                if depth > st.crack_m[j, k]:
                    st.crack_m[j, k] = depth
            st.cut_s[j, k] += t_in
            st.used[j, k] = 1
            if out.doc[j, k] != 0:
                st.doc_hit[j, k] = 1
            out.d_vn[j, k] = d_vn
            out.d_notch[j, k] = d_notch
            out.d_kt[j, k] = d_kt
            out.d_D[j, k] = d_D
            rate = d_vn / t_in
            if rate > out.scal[OS_MAX_RATE_M_S]:
                out.scal[OS_MAX_RATE_M_S] = rate
            if st.D_th[j, k] > out.scal[OS_MAX_D]:
                out.scal[OS_MAX_D] = st.D_th[j, k]
            out.scal[OS_SUM_DVN] += d_vn


@njit(types.void(GEO_T, LAW_T, ST_T, OUT_T), **OPTS)
def element_outputs(geo, law, st, out):
    """VB = (VN + chip) (cot a - tan g); notch land where the element has been at the depth-of-cut line;
    radial and axial wear = total recession x (sin kappa, cos kappa); worn hone radius."""
    nt = geo.valid.shape[0]
    ne = geo.valid.shape[1]
    rounding = law.scal[LS_ROUNDING]
    for j in range(nt):
        for k in range(ne):
            if geo.valid[j, k] == 0:
                out.VB[j, k] = 0.0
                out.VB_N[j, k] = 0.0
                out.radial[j, k] = 0.0
                out.axial[j, k] = 0.0
                out.edge_r[j, k] = geo.r_e0[j, k]
                continue
            g = geo.g_vb[j, k]
            total = st.vn[j, k] + st.vn_notch[j, k] + st.chip_d[j, k]
            out.VB[j, k] = (st.vn[j, k] + st.chip_d[j, k]) * g
            out.VB_N[j, k] = total * g if st.doc_hit[j, k] != 0 else 0.0
            out.radial[j, k] = total * geo.sin_k[j, k]
            out.axial[j, k] = total * geo.cos_k[j, k]
            out.edge_r[j, k] = worn_edge_radius(geo.r_e0[j, k], st.vn[j, k], st.vn_notch[j, k], st.chip_d[j, k],
                                                np.int64(st.chipped[j, k]), rounding)


# ====================================================================== chipping and coating spallation

@njit(types.void(GEO_T, ST_T, CHIP_T, I1, F2, F1, F2), **OPTS)
def chip_trials(geo, st, chip, ikeys, rows, b_eng, out):
    """Weibull chipping and coating-spallation trials of the shock rows (sorted by tick, tooth, elem).

    ikeys = [seed, stream key of wear.chipping, stream key of wear.coating, tool slot]. Per row: entry
    contact l_c = max(ratio h, hone radius), force per unit width F / b (b: the tooth's engaged chip
    width in the step), Flamant tension at r = l_c, effective volume b l_c^2 I_theta / (m - 2),
    P = 1 - exp(-(V_eff / V_ref)(sigma / sigma_0)^m - (sigma / sigma_crack)^m) with
    sigma_crack = K_IC / (Y sqrt(pi a)) for a comb crack of depth a. One draw per row and element counter."""
    c = chip.scal
    seed = ikeys[0]
    sid_chip = ikeys[1]
    sid_coat = ikeys[2]
    slot = ikeys[3]
    nt = geo.valid.shape[0]
    ne = geo.valid.shape[1]
    m = c[CL_M]
    n_theta = np.int64(c[CL_NTHETA])
    for i in range(rows.shape[0]):
        for col in range(out.shape[1]):
            out[i, col] = 0.0
        j = np.int64(rows[i, SH_TOOTH])
        k = np.int64(rows[i, SH_ELEM])
        if j < 0 or j >= nt or k < 0 or k >= ne:
            continue
        if geo.valid[j, k] == 0:
            continue
        force = rows[i, SH_FORCE]
        if force <= 0.0:
            continue
        width = b_eng[j] if b_eng[j] > 0.0 else geo.db[j, k]
        if width <= 0.0:
            continue
        radius = worn_edge_radius(geo.r_e0[j, k], st.vn[j, k], st.vn_notch[j, k], st.chip_d[j, k],
                                  np.int64(st.chipped[j, k]), c[CL_ROUNDING])
        contact = c[CL_LC_RATIO] * rows[i, SH_H]
        if contact < radius:
            contact = radius
        if contact <= 0.0:
            continue
        beta = geo.beta[j, k]
        if beta < c[CL_BETA_MIN]:
            beta = c[CL_BETA_MIN]
        if beta > c[CL_BETA_MAX]:
            beta = c[CL_BETA_MAX]
        per_m = force / width
        if np.int64(rows[i, SH_KIND]) != SHOCK_ENTRY:
            per_m *= c[CL_EXIT_FACTOR]
        sigma, c_x, c_y = wedge_tension(per_m, contact, beta, geo.rake[j, k], geo.clear[j, k], c[CL_MU])
        key = slot * KEY_SLOT + j * KEY_TOOTH + k
        out[i, CO_SIGMA_PA] = sigma
        p_chip = 0.0
        if sigma > 0.0:
            v_eff = width * contact * contact * wedge_volume_factor(c_x, c_y, beta, m, n_theta) / (m - 2.0)
            out[i, CO_VEFF_M3] = v_eff
            hazard = v_eff / c[CL_VREF_M3] * (sigma / c[CL_SIGMA0_PA]) ** m
            if c[CL_KIC] > 0.0 and st.crack_m[j, k] > 0.0:
                sigma_crack = c[CL_KIC] / (c[CL_Y] * math.sqrt(math.pi * st.crack_m[j, k]))
                hazard += (sigma / sigma_crack) ** m
            p_chip = -math.expm1(-hazard)
        out[i, CO_P] = p_chip
        draw = hash_uniform(seed, sid_chip, key, st.n_chip[j, k], 0)
        st.n_chip[j, k] += 1
        if draw < p_chip:
            depth = c[CL_CHIP_RATIO] * contact
            if depth > st.chip_d[j, k]:
                st.chip_d[j, k] = depth
            st.chipped[j, k] = 1
            st.coat_f[j, k] = 0.0
            st.coat_r[j, k] = 0.0
            out[i, CO_CHIPPED] = 1.0
            out[i, CO_DEPTH_M] = depth
            continue
        if st.coat_f[j, k] > 0.0 or st.coat_r[j, k] > 0.0:
            pressure = per_m / contact
            p_coat = -math.expm1(-(pressure / c[CL_COAT_SIGMA0_PA]) ** c[CL_COAT_M])
            out[i, CO_P_COAT] = p_coat
            draw_c = hash_uniform(seed, sid_coat, key, st.n_coat[j, k], 0)
            st.n_coat[j, k] += 1
            if draw_c < p_coat:
                st.coat_f[j, k] = 0.0
                st.coat_r[j, k] = 0.0
                out[i, CO_SPALLED] = 1.0


__all__ = ["usui_rate", "table_at", "worn_edge_radius", "wedge_tension", "wedge_volume_factor", "advance_elements",
           "element_outputs", "chip_trials"]
