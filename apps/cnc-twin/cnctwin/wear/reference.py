"""Plain-Python references of the wear kernels (architecture 5.5: every kernel has one, the tests
compare them). Written independently of :mod:`cnctwin.wear.kernels` (no shared code beyond the
layouts and core.rng's counter-based hash), element by element with ``math``."""

from __future__ import annotations

import math

import numpy as np

from ..core.rng import hash_uniform
from . import abi as A


def usui_rate_ref(A_, B_, lo, temp_K: float) -> float:
    if len(A_) == 0 or temp_K <= 0.0:
        return 0.0
    chosen = 0
    for i, start in enumerate(lo):
        if temp_K >= start:
            chosen = i
    return float(A_[chosen]) * math.exp(-float(B_[chosen]) / temp_K)


def table_at_ref(xs, ys, x: float) -> float:
    if len(xs) == 0:
        return 0.0
    return float(np.interp(x, np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)))


def worn_edge_radius_ref(r_e0, vn, vn_notch, chip_d, chipped, rounding) -> float:
    radius = r_e0 + rounding * (vn + vn_notch)
    return max(radius, 0.5 * chip_d) if chipped else radius


def wedge_tension_ref(force_per_m, contact_m, beta, rake, clear, mu):
    if force_per_m <= 0.0 or contact_m <= 0.0 or beta <= 0.0:
        return 0.0, 0.0, 0.0
    direction = math.atan(mu) - rake                 # resultant on the tool, from the cutting direction
    bisector = (clear + (math.pi / 2.0 - rake)) / 2.0
    axial = force_per_m * math.cos(direction - bisector)
    transverse = force_per_m * math.sin(direction - bisector)
    c_x = 2.0 * axial / (beta + math.sin(beta))
    c_y = 2.0 * transverse / (beta - math.sin(beta))
    stresses = [-(c_x * math.cos(t) + c_y * math.sin(t)) / contact_m for t in (beta / 2.0, -beta / 2.0)]
    return max(max(stresses), 0.0), c_x, c_y


def wedge_volume_factor_ref(c_x, c_y, beta, m, n) -> float:
    half = beta / 2.0
    thetas = []
    weights = []
    for i in range(int(n)):
        u = (i + 0.5) / n
        thetas.append(-half + beta * (3.0 * u * u - 2.0 * u ** 3))
        weights.append(beta * 6.0 * u * (1.0 - u) / n)
    g = [-(c_x * math.cos(t) + c_y * math.sin(t)) for t in thetas]
    ends = [-(c_x * math.cos(t) + c_y * math.sin(t)) for t in (-half, half)]
    top = max(max(ends), max(g) if g else -math.inf)
    if top <= 0.0:
        return 0.0
    return sum(w * (v / top) ** m for v, w in zip(g, weights) if v > 0.0)


def _doc_flags(geo, inp, ratio) -> np.ndarray:
    nt, ne = geo.valid.shape
    doc = np.zeros((nt, ne), np.int8)
    for j in range(nt):
        cutting = [bool(geo.valid[j, k]) and inp.in_cut_s[j, k] > 0.0 for k in range(ne)]
        runs = []
        k = 0
        while k < ne:
            if not cutting[k]:
                k += 1
                continue
            start = k
            while k < ne and cutting[k]:
                k += 1
            runs.append((start, k - 1))
        for start, end in runs:
            h = [inp.h_int[j, k] / inp.in_cut_s[j, k] for k in range(start, end + 1)]
            top = max(h)
            if top <= 0.0:
                continue
            if end + 1 < ne and geo.valid[j, end + 1] and h[-1] >= ratio * top:
                doc[j, end] = 1
            if start >= 1 and geo.valid[j, start - 1] and h[0] >= ratio * top:
                doc[j, start] = 1
    return doc


def advance_elements_ref(geo, inp, law, st) -> dict:
    """Returns the step increments and summary as plain values; updates ``st`` in place."""
    s = law.scal
    nt, ne = geo.valid.shape
    dt = float(inp.scal[A.IS_DT_S])
    doc = _doc_flags(geo, inp, s[A.LS_DOC_RATIO])
    inc = {name: np.zeros((nt, ne)) for name in ("d_vn", "d_notch", "d_kt", "d_D")}
    summary = np.zeros(A.N_OS)
    for j in range(nt):
        for k in range(ne):
            t_in = float(inp.in_cut_s[j, k])
            if not geo.valid[j, k] or t_in <= 0.0:
                continue
            h_mean = inp.h_int[j, k] / t_in
            skin = 1.0 + inp.skin_frac[j, k] * (s[A.LS_SKIN_FACTOR] - 1.0)
            bue = 1.0
            if st.bue_m[j, k] > 0.0 and h_mean > 0.0:
                bue = 1.0 - s[A.LS_BUE_PROTECT] * min(1.0, st.bue_m[j, k] / h_mean)
            coated_f, coated_r = st.coat_f[j, k] > 0.0, st.coat_r[j, k] > 0.0
            flank = rake = 0.0
            for q, w in enumerate(inp.w_q):
                tf, tr = float(inp.flank_T[j, k, q]), float(inp.rake_T[j, k, q])
                uf = usui_rate_ref(law.fl_A, law.fl_B, law.fl_lo, tf)
                ur = usui_rate_ref(law.rk_A, law.rk_B, law.rk_lo, tr)
                if coated_f and tf <= s[A.LS_COAT_TMAX_K]:
                    uf *= s[A.LS_COAT_FACTOR]
                if coated_r and tr <= s[A.LS_COAT_TMAX_K]:
                    ur *= s[A.LS_COAT_FACTOR]
                hf = table_at_ref(law.hc_T, law.hc_Pa, tf) if coated_f else table_at_ref(law.hs_T, law.hs_Pa, tf)
                hr = table_at_ref(law.hc_T, law.hc_Pa, tr) if coated_r else table_at_ref(law.hs_T, law.hs_Pa, tr)
                flank += w * (uf + (s[A.LS_ABR] / hf if hf > 0.0 else 0.0))
                rake += w * (ur + (s[A.LS_ABR] / hr if hr > 0.0 else 0.0))
            d_vn = flank * inp.sv_flank[j, k] * skin * bue + inp.bue_broke[j, k] * s[A.LS_PLUCK_M]
            d_kt = rake * inp.sv_rake[j, k] * skin * bue
            d_notch = d_vn * s[A.LS_NOTCH_EXCESS] if doc[j, k] else 0.0
            d_D = 0.0
            cycles = inp.entries[j, k]
            if cycles > 0.0 and inp.cycle_dT[j, k] > 0.0 and s[A.LS_FAT_SIGMA_PA] > 0.0 and s[A.LS_FAT_B] < 0.0:
                amp = inp.therm_coef[j, k] * inp.cycle_dT[j, k]
                if amp > 0.0:
                    d_D = cycles / (0.5 * (amp / s[A.LS_FAT_SIGMA_PA]) ** (1.0 / s[A.LS_FAT_B]))
            st.vn[j, k] += d_vn
            st.vn_notch[j, k] += d_notch
            st.kt[j, k] += d_kt
            if coated_f:
                st.coat_f[j, k] = max(0.0, st.coat_f[j, k] - d_vn * geo.c_coat[j, k])
                summary[A.OS_COAT_THROUGH] += st.coat_f[j, k] == 0.0
            if coated_r:
                st.coat_r[j, k] = max(0.0, st.coat_r[j, k] - d_kt)
                summary[A.OS_RAKE_THROUGH] += st.coat_r[j, k] == 0.0
            before = st.D_th[j, k]
            st.D_th[j, k] = before + d_D
            if st.D_th[j, k] >= 1.0:
                summary[A.OS_CRACK_INIT] += before < 1.0
                depth = s[A.LS_CRACK_A0_M]
                if cycles > 0.0:
                    reach = math.sqrt(inp.diffusivity[j, k] * dt / cycles / math.pi)
                    if reach > depth:
                        depth += (reach - depth) * (1.0 - math.exp(-(st.D_th[j, k] - 1.0) / s[A.LS_CRACK_GROWTH]))
                st.crack_m[j, k] = max(st.crack_m[j, k], depth)
            st.cut_s[j, k] += t_in
            st.used[j, k] = 1
            if doc[j, k]:
                st.doc_hit[j, k] = 1
            inc["d_vn"][j, k], inc["d_notch"][j, k], inc["d_kt"][j, k], inc["d_D"][j, k] = d_vn, d_notch, d_kt, d_D
            summary[A.OS_MAX_RATE_M_S] = max(summary[A.OS_MAX_RATE_M_S], d_vn / t_in)
            summary[A.OS_MAX_D] = max(summary[A.OS_MAX_D], st.D_th[j, k])
            summary[A.OS_SUM_DVN] += d_vn
    return {"doc": doc, "scal": summary, **inc}


def element_outputs_ref(geo, law, st) -> dict:
    nt, ne = geo.valid.shape
    out = {name: np.zeros((nt, ne)) for name in ("VB", "VB_N", "radial", "axial", "edge_r")}
    for j in range(nt):
        for k in range(ne):
            if not geo.valid[j, k]:
                out["edge_r"][j, k] = geo.r_e0[j, k]
                continue
            total = st.vn[j, k] + st.vn_notch[j, k] + st.chip_d[j, k]
            out["VB"][j, k] = (st.vn[j, k] + st.chip_d[j, k]) * geo.g_vb[j, k]
            out["VB_N"][j, k] = total * geo.g_vb[j, k] if st.doc_hit[j, k] else 0.0
            out["radial"][j, k] = total * geo.sin_k[j, k]
            out["axial"][j, k] = total * geo.cos_k[j, k]
            out["edge_r"][j, k] = worn_edge_radius_ref(geo.r_e0[j, k], st.vn[j, k], st.vn_notch[j, k],
                                                       st.chip_d[j, k], st.chipped[j, k], law.scal[A.LS_ROUNDING])
    return out


def chip_trials_ref(geo, st, chip, ikeys, rows, b_eng) -> np.ndarray:
    c = chip.scal
    seed, sid_chip, sid_coat, slot = (int(v) for v in ikeys)
    nt, ne = geo.valid.shape
    out = np.zeros((rows.shape[0], A.N_CO))
    m = float(c[A.CL_M])
    for i, row in enumerate(rows):
        j, k = int(row[A.SH_TOOTH]), int(row[A.SH_ELEM])
        if not (0 <= j < nt and 0 <= k < ne) or not geo.valid[j, k] or row[A.SH_FORCE] <= 0.0:
            continue
        width = b_eng[j] if b_eng[j] > 0.0 else geo.db[j, k]
        if width <= 0.0:
            continue
        radius = worn_edge_radius_ref(geo.r_e0[j, k], st.vn[j, k], st.vn_notch[j, k], st.chip_d[j, k],
                                      st.chipped[j, k], c[A.CL_ROUNDING])
        contact = max(c[A.CL_LC_RATIO] * row[A.SH_H], radius)
        if contact <= 0.0:
            continue
        beta = min(max(geo.beta[j, k], c[A.CL_BETA_MIN]), c[A.CL_BETA_MAX])
        per_m = row[A.SH_FORCE] / width * (1.0 if int(row[A.SH_KIND]) == A.SHOCK_ENTRY else c[A.CL_EXIT_FACTOR])
        sigma, c_x, c_y = wedge_tension_ref(per_m, contact, beta, geo.rake[j, k], geo.clear[j, k], c[A.CL_MU])
        key = slot * A.KEY_SLOT + j * A.KEY_TOOTH + k
        out[i, A.CO_SIGMA_PA] = sigma
        p = 0.0
        if sigma > 0.0:
            v_eff = width * contact ** 2 * wedge_volume_factor_ref(c_x, c_y, beta, m, int(c[A.CL_NTHETA])) / (m - 2.0)
            out[i, A.CO_VEFF_M3] = v_eff
            hazard = v_eff / c[A.CL_VREF_M3] * (sigma / c[A.CL_SIGMA0_PA]) ** m
            if c[A.CL_KIC] > 0.0 and st.crack_m[j, k] > 0.0:
                hazard += (sigma * c[A.CL_Y] * math.sqrt(math.pi * st.crack_m[j, k]) / c[A.CL_KIC]) ** m
            p = -math.expm1(-hazard)
        out[i, A.CO_P] = p
        u = hash_uniform(seed, sid_chip, key, int(st.n_chip[j, k]), 0)
        st.n_chip[j, k] += 1
        if u < p:
            depth = c[A.CL_CHIP_RATIO] * contact
            st.chip_d[j, k] = max(st.chip_d[j, k], depth)
            st.chipped[j, k] = 1
            st.coat_f[j, k] = 0.0
            st.coat_r[j, k] = 0.0
            out[i, A.CO_CHIPPED] = 1.0
            out[i, A.CO_DEPTH_M] = depth
            continue
        if st.coat_f[j, k] > 0.0 or st.coat_r[j, k] > 0.0:
            p_coat = -math.expm1(-((per_m / contact) / c[A.CL_COAT_SIGMA0_PA]) ** c[A.CL_COAT_M])
            out[i, A.CO_P_COAT] = p_coat
            u = hash_uniform(seed, sid_coat, key, int(st.n_coat[j, k]), 0)
            st.n_coat[j, k] += 1
            if u < p_coat:
                st.coat_f[j, k] = 0.0
                st.coat_r[j, k] = 0.0
                out[i, A.CO_SPALLED] = 1.0
    return out


__all__ = ["usui_rate_ref", "table_at_ref", "worn_edge_radius_ref", "wedge_tension_ref", "wedge_volume_factor_ref",
           "advance_elements_ref", "element_outputs_ref", "chip_trials_ref"]
