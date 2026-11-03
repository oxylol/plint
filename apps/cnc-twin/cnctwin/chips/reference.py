"""Plain-Python references of the chips kernels (architecture 5.5: every kernel has one, and a test
compares the two). Grid kernels are written vectorised with NumPy, the counter-based draws with
Python integers, and the particle flight with 3-vectors, so they are independent formulations of
the same equations as :mod:`cnctwin.chips.kernels`; they mutate the same layouts the same way.
"""

from __future__ import annotations

import math

import numpy as np

from ..units import STEFAN_BOLTZMANN_W_M2K4
from . import abi
from .abi import (CYL_PADDLE, EV_BOUNCE_LIMIT, EV_NOZZLE, EV_WELD, EV_WRAP, FLYING, FREE, MOB_FLUSH, MOB_RECUT,
                  MOB_SWEEP, ON_NOZZLE, ON_TOOL, OUT_DEPOSIT_KG, OUT_HEAT_AIR_J, OUT_HEAT_COOLANT_J, OUT_HITS,
                  OUT_PAN_BULK_M3, OUT_PAN_ENTHALPY_J, OUT_PAN_LONG_KG, OUT_PAN_MASS_KG, OUT_SUBSTEPS, SURF_ENCLOSURE,
                  SURF_NONE, SURF_NOZZLE, SURF_PILE, WRAPPED)

_SERIES_Z = 1e-3
_PARALLEL_EPS = 1e-12
_EXTENSIVE = ("bulk_m3", "mass_kg", "enthalpy_J", "size_mass", "thick_mass", "width_mass", "length_mass",
              "repose_mass", "long_kg", "fines_kg")


# ============================================================== numerics

def phi12(z: float) -> tuple[float, float]:
    if z < _SERIES_Z:
        return 1 - z / 2 + z * z / 6 - z ** 3 / 24, 0.5 - z / 6 + z * z / 24 - z ** 3 / 120
    em = math.expm1(-z)
    return -em / z, (z + em) / (z * z)


def propagate(x, v, u, g, k, dt):
    """x, v, u, g: 3-vectors (NumPy). Exact solution of dv/dt = g - k (v - u)."""
    p1, p2 = phi12(k * dt)
    x1 = x + u * dt + (v - u) * (dt * p1) + g * (dt * dt * p2)
    v1 = v + (u - v) * (k * dt * p1) + g * (dt * p1)
    return x1, v1


def drag_rate(speed, rho, mu, dv, area, chip_mass, stokes, a, b, c, d):
    if rho <= 0 or chip_mass <= 0 or dv <= 0:
        return 0.0
    re = rho * speed * dv / mu
    cd = stokes * mu / (rho * dv) * (1 + a * re ** b) + c * speed * re / (re + d)
    return 0.5 * rho * area * cd / chip_mass


def nusselt_h(speed, rho, mu, k, pr, dv, ra, rb, rm, rn):
    if dv <= 0 or rho <= 0:
        return 0.0
    return k / dv * (ra + rb * (rho * speed * dv / mu) ** rm * pr ** rn)


def cp_at(temp, cp_t, cp_v):
    return float(np.interp(temp, cp_t, cp_v))


def enthalpy(temp, cp_t, cp_v, h_t):
    if temp <= cp_t[0]:
        return h_t[0] + cp_v[0] * (temp - cp_t[0])
    if temp >= cp_t[-1]:
        return h_t[-1] + cp_v[-1] * (temp - cp_t[-1])
    i = int(np.searchsorted(cp_t, temp, side="left")) - 1
    i = min(max(i, 0), len(cp_t) - 2)
    dt = temp - cp_t[i]
    slope = (cp_v[i + 1] - cp_v[i]) / (cp_t[i + 1] - cp_t[i])
    return h_t[i] + cp_v[i] * dt + 0.5 * slope * dt * dt


def temperature(h, cp_t, cp_v, h_t):
    if h <= h_t[0]:
        return cp_t[0] + (h - h_t[0]) / cp_v[0]
    if h >= h_t[-1]:
        return cp_t[-1] + (h - h_t[-1]) / cp_v[-1]
    i = int(np.searchsorted(h_t, h, side="left")) - 1
    i = min(max(i, 0), len(cp_t) - 2)
    a = 0.5 * (cp_v[i + 1] - cp_v[i]) / (cp_t[i + 1] - cp_t[i])
    b = cp_v[i]
    c = h_t[i] - h
    return cp_t[i] + 2 * (-c) / (b + math.sqrt(max(b * b - 4 * a * c, 0.0)))


def impulse(v, s, n, e, mu):
    r = v - s
    vn = float(r @ n)
    if vn >= 0:
        return v.copy()
    t = r - vn * n
    tm = float(np.linalg.norm(t))
    if tm > 0:
        red = mu * (1 + e) * (-vn)
        t = t * (0.0 if red >= tm else 1 - red / tm)
    return s + t - e * vn * n


def tick_index(t, t0, dt, n_t):
    q = (t - t0) / dt
    if q <= 0:
        return 0, 0.0
    if q >= n_t:
        return n_t - 1, 1.0
    k = min(math.floor(q), n_t - 1)
    return k, q - k


def lerp(arr, k, frac):
    return arr[k] + frac * (arr[k + 1] - arr[k])


# ============================================================== geometry

def hf_hit(p0, p1, H):
    """Same contract as kernels.hf_hit, p0/p1 in G. Returns (s, n_G, kind, iu, iv)."""
    cell = float(H.cell_m[0])
    nu, nv = H.base_m.shape
    d = p1 - p0
    s_lo, s_hi, face = 0.0, 1.0, 0
    for axis, size in ((0, nu * cell), (1, nv * cell)):
        if abs(d[axis]) < _PARALLEL_EPS:
            if p0[axis] < 0 or p0[axis] >= size:
                return 2.0, np.zeros(3), 0, -1, -1
            continue
        sa, sb = sorted(((0 - p0[axis]) / d[axis], (size - p0[axis]) / d[axis]))
        if sa > s_lo:
            s_lo, face = sa, axis + 1
        s_hi = min(s_hi, sb)
    if s_lo > s_hi:
        return 2.0, np.zeros(3), 0, -1, -1
    s = s_lo
    iu = min(max(math.floor((p0[0] + s * d[0]) / cell), 0), nu - 1)
    iv = min(max(math.floor((p0[1] + s * d[1]) / cell), 0), nv - 1)
    step = [int(np.sign(d[0])), int(np.sign(d[1]))]
    tmax, tdel = [math.inf, math.inf], [math.inf, math.inf]
    idx = [iu, iv]
    for a in range(2):
        if step[a] != 0:
            boundary = (idx[a] + (1 if step[a] > 0 else 0)) * cell
            tmax[a] = (boundary - p0[a]) / d[a]
            tdel[a] = cell / abs(d[a])
    prev = (-1, -1)
    while True:
        s_out = min(tmax[0], tmax[1], s_hi)
        base = H.base_m[idx[0], idx[1]]
        if base > -math.inf:
            top = base + H.bulk_m3[idx[0], idx[1]] / (cell * cell)
            if p0[2] + s * d[2] < top:
                if face == 0:
                    return s, np.array([0.0, 0.0, 1.0]), 3, idx[0], idx[1]
                n = np.zeros(3)
                n[face - 1] = -float(step[face - 1])
                return s, n, 2, prev[0], prev[1]
            if p0[2] + s_out * d[2] < top and d[2] < 0:
                return (top - p0[2]) / d[2], np.array([0.0, 0.0, 1.0]), 1, idx[0], idx[1]
        if s_out >= s_hi:
            break
        prev = (idx[0], idx[1])
        a = 0 if tmax[0] <= tmax[1] else 1
        idx[a] += step[a]
        s = tmax[a]
        tmax[a] += tdel[a]
        face = a + 1
        if not (0 <= idx[0] < nu and 0 <= idx[1] < nv):
            break
    return 2.0, np.zeros(3), 0, -1, -1


def cyl_hit(p0, p1, b, a, radius, length):
    """(s, n, z_axial, flag) as kernels.cyl_hit."""
    d0, d1 = p0 - b, p1 - b
    za, zb = float(d0 @ a), float(d1 @ a)
    r0, r1 = d0 - za * a, d1 - zb * a
    c = float(r0 @ r0) - radius * radius
    if c <= 0 and 0 <= za <= length:
        rm = float(np.linalg.norm(r0))
        if rm > 0:
            return 0.0, r0 / rm, za, 2
        p = r1.copy()
        if np.linalg.norm(p) == 0:
            k = int(np.argmin(np.abs(a)))
            if k == 0:
                p = np.array([0.0, a[2], -a[1]])
            elif k == 1:
                p = np.array([-a[2], 0.0, a[0]])
            else:
                p = np.array([a[1], -a[0], 0.0])
        return 0.0, p / np.linalg.norm(p), za, 2
    dr = r1 - r0
    aa = float(dr @ dr)
    if aa <= 0:
        return 2.0, np.zeros(3), 0.0, 0
    bb = float(r0 @ dr)
    disc = bb * bb - aa * c
    if disc < 0:
        return 2.0, np.zeros(3), 0.0, 0
    s = (-bb - math.sqrt(disc)) / aa
    if s < 0 or s > 1:
        return 2.0, np.zeros(3), 0.0, 0
    zs = za + s * (zb - za)
    if zs < 0 or zs > length:
        return 2.0, np.zeros(3), 0.0, 0
    r = r0 + s * dr
    return s, r / np.linalg.norm(r), zs, 1


def sph_hit(p0, p1, centre, radius):
    d = p0 - centre
    c = float(d @ d) - radius * radius
    if c <= 0:
        m = float(np.linalg.norm(d))
        return (0.0, d / m, 2) if m > 0 else (0.0, np.array([0.0, 0.0, 1.0]), 2)
    e = p1 - p0
    aa = float(e @ e)
    if aa <= 0:
        return 2.0, np.zeros(3), 0
    bb = float(d @ e)
    disc = bb * bb - aa * c
    if disc < 0:
        return 2.0, np.zeros(3), 0
    s = (-bb - math.sqrt(disc)) / aa
    if s < 0 or s > 1:
        return 2.0, np.zeros(3), 0
    p = d + s * e
    return s, p / np.linalg.norm(p), 1


def box_exit(p0, p1, lo, hi):
    best, axis, side = 2.0, -1, -1
    for k in range(3):
        if p0[k] < lo[k]:
            return 0.0, k, 2
        if p0[k] > hi[k]:
            return 0.0, k, 3
        if p1[k] < lo[k]:
            s = (lo[k] - p0[k]) / (p1[k] - p0[k])
            if s < best:
                best, axis, side = s, k, 0
        elif p1[k] > hi[k]:
            s = (hi[k] - p0[k]) / (p1[k] - p0[k])
            if s < best:
                best, axis, side = s, k, 1
    return best, axis, side


# ============================================================== pile bookkeeping

def deposit(H, C, iu, iv, mass, enth, dv, thick, width, length, form):
    fi = min(max(int(form), 0), abi.N_FORM - 1)
    H.bulk_m3[iu, iv] += mass * C.form_R[fi] / C.rho_kg_m3[0]
    H.mass_kg[iu, iv] += mass
    H.enthalpy_J[iu, iv] += enth
    H.size_mass[iu, iv] += mass * dv
    H.thick_mass[iu, iv] += mass * thick
    H.width_mass[iu, iv] += mass * width
    H.length_mass[iu, iv] += mass * length
    H.repose_mass[iu, iv] += mass * C.form_repose_tan[fi]
    if C.form_long[fi] != 0:
        H.long_kg[iu, iv] += mass
    if thick < C.fines_thickness_m[0]:
        H.fines_kg[iu, iv] += mass


def take_fraction(H, iu, iv, frac):
    row = np.zeros(10)
    for k, name in enumerate(_EXTENSIVE):
        arr = getattr(H, name)
        d = arr[iu, iv] if frac >= 1.0 else arr[iu, iv] * frac
        arr[iu, iv] -= d
        row[k] = d
    return row


def write_mob(rec, n, kind, iu, iv, taken, vel, uvw, owner, time_s):
    i = int(n[0])
    if i >= rec.shape[0]:
        n[1] = 1
        return
    rec[i] = [kind, iu, iv, *taken[1:10], *vel, *uvw, owner, time_s]
    n[0] = i + 1


def write_event(ev, kind, particle, owner, time_s, mass, pos, temp, form):
    i = int(ev.n[0])
    if i >= ev.rec.shape[0]:
        ev.n[1] = 1
        return
    ev.rec[i] = [kind, particle, owner, time_s, mass, pos[0], pos[1], pos[2], temp, form]
    ev.n[0] = i + 1


def to_pan(P, C, i, out):
    m = P.mass_kg[i]
    fi = min(max(int(P.form[i]), 0), abi.N_FORM - 1)
    out[OUT_PAN_MASS_KG] += m
    out[OUT_PAN_ENTHALPY_J] += m * enthalpy(P.temp_K[i], C.cp_T_K, C.cp_J_kgK, C.h_J_kg)
    out[OUT_PAN_BULK_M3] += m * C.form_R[fi] / C.rho_kg_m3[0]
    if C.form_long[fi] != 0:
        out[OUT_PAN_LONG_KG] += m
    P.status[i] = FREE
    P.mass_kg[i] = 0.0


def settle(P, H, C, i, iu, iv, out):
    if iu < 0 or iv < 0 or H.base_m[iu, iv] == -math.inf:
        to_pan(P, C, i, out)
        return
    m = P.mass_kg[i]
    deposit(H, C, iu, iv, m, m * enthalpy(P.temp_K[i], C.cp_T_K, C.cp_J_kgK, C.h_J_kg), P.dv_m[i],
            P.thickness_m[i], P.width_m[i], P.length_m[i], P.form[i])
    out[OUT_DEPOSIT_KG] += m
    P.status[i] = FREE
    P.mass_kg[i] = 0.0


# ============================================================== fluids and heat

def jet_fluid(p, S):
    """(u, rho, mu, k, Pr, T, force_scale, liquid_fraction) as kernels.jet_fluid."""
    best = -1.0
    res = (np.zeros(3), *S.air[:5], 1.0, 0.0)
    for j in range(len(S.jet_speed)):
        sp = S.jet_speed[j]
        if sp <= 0:
            continue
        d = p - S.jet_origin_M[j]
        e = S.jet_dir_M[j]
        s = float(d @ e)
        if s <= 0:
            continue
        r = float(np.linalg.norm(d - s * e))
        r0 = S.jet_r0[j]
        if S.jet_gas[j] == 1:
            d0 = 2 * r0
            uc = sp if s <= S.jet_const[0] * d0 else sp * S.jet_const[0] * d0 / s
            rh = max(r0, S.jet_const[1] * s)
            u = uc * math.exp(-math.log(2.0) * (r / rh) ** 2)
            q = 0.5 * S.jet_fluid[j, 0] * u * u
            if q > best:
                best = q
                res = (u * e, *S.jet_fluid[j], S.air[4], 1.0, 0.0)
        else:
            rj = r0 + s * S.jet_tan[j]
            if r > rj:
                continue
            alpha = (r0 / rj) ** 2
            q = 0.5 * alpha * S.jet_fluid[j, 0] * sp * sp
            if q > best:
                best = q
                res = (sp * e, *S.jet_fluid[j], S.liquid[4], alpha, alpha)
    return res


def cool(P, C, S, i, dt, speed, rho, mu, kf, pr, tf, liq, t_rad, t_air):
    temp = P.temp_K[i]
    m_chip, area, dv = P.chip_mass_kg[i], P.surface_m2[i], P.dv_m[i]
    if m_chip <= 0 or area <= 0 or dt <= 0:
        return temp, 0.0, 0.0
    h_l = liq * nusselt_h(speed, rho, mu, kf, pr, dv, *C.rm) if liq > 0 else 0.0
    h_a = (1 - liq) * nusselt_h(speed, S.air[0], S.air[1], S.air[2], S.air[3], dv, *C.rm)
    h_r = C.emissivity[0] * STEFAN_BOLTZMANN_W_M2K4 * (temp ** 2 + t_rad ** 2) * (temp + t_rad)
    h_tot = h_l + h_a + h_r
    if h_tot <= 0:
        return temp, 0.0, 0.0
    t_inf = (h_l * tf + h_a * t_air + h_r * t_rad) / h_tot
    cp = cp_at(temp, C.cp_T_K, C.cp_J_kgK)
    t_new = t_inf + (temp - t_inf) * math.exp(-h_tot * area * dt / (m_chip * cp))
    q = P.mass_kg[i] * (enthalpy(temp, C.cp_T_K, C.cp_J_kgK, C.h_J_kg) - enthalpy(t_new, C.cp_T_K, C.cp_J_kgK, C.h_J_kg))
    fl = h_l * (temp - tf)
    tot = fl + h_a * (temp - t_air) + h_r * (temp - t_rad)
    q_l = q * fl / tot if tot != 0 else 0.0
    return t_new, q_l, q - q_l


def capillary_escape_speed(P, S, C, i):
    f_cap = C.wet[2] * S.liquid[6] * C.wet[1] * 2 * (P.width_m[i] + P.length_m[i])
    if P.chip_mass_kg[i] <= 0 or f_cap <= 0:
        return 0.0
    return math.sqrt(2 * f_cap * S.liquid[5] / P.chip_mass_kg[i])


def respond_hf(P, H, W, S, C, out, i, v, n, kind, iu, iv, s_vel):
    surf, wet = SURF_ENCLOSURE, False
    if iu >= 0 and iv >= 0:
        surf = SURF_PILE if H.bulk_m3[iu, iv] > 0 else int(H.base_kind[iu, iv])
        if surf == SURF_NONE:
            surf = SURF_ENCLOSURE
        wet = H.wet[iu, iv] != 0
    e, mu = C.e_pair[surf], C.mu_pair[surf]
    v_in = max(-float((v - s_vel) @ n), 0.0)
    g_n = -float(W.gravity_M @ n)
    dv = P.dv_m[i]
    v_reb = e * v_in
    stick = False
    if wet:
        v_esc = capillary_escape_speed(P, S, C, i)
        st_c = (1 + 1 / e) * C.wet[0] if e > 0 else math.inf
        v_visc = st_c * 9 * S.liquid[1] / (4 * C.rho_kg_m3[0] * dv)
        stick = v_reb < v_esc or v_in < v_visc
    if kind != 2:
        if g_n > 0 and (v_reb < math.sqrt(2 * g_n * C.settle_bounce[0] * dv) or stick):
            settle(P, H, C, i, iu, iv, out)
            return True
    elif stick:
        settle(P, H, C, i, iu, iv, out)
        return True
    P.vel_M_m_s[i] = impulse(v, s_vel, n, e, mu)
    return False


# ============================================================== K1

def advance_particles(P, H, W, S, C, out, ev):
    t0, t_end, dtk = W.t0_s[0], W.t_end_s[0], W.dt_tick_s[0]
    n_t = W.grid_origin_M.shape[0] - 1
    R = W.R_M_G
    g = W.gravity_M.copy()
    dt_max, drag_fac, offset = C.numerics[0], C.numerics[1], C.numerics[2]
    max_bounces, max_hits = int(C.numerics[3]), int(C.numerics[4])
    rho_p, t_air, cell = C.rho_kg_m3[0], S.air[4], H.cell_m[0]
    nu, nv = H.base_m.shape

    def to_G(p, t):
        k, f = tick_index(t, t0, dtk, n_t)
        return R.T @ (p - lerp(W.grid_origin_M, k, f))

    for i in range(int(P.n[0])):
        if P.status[i] == FREE:
            continue
        if P.status[i] != FLYING:
            dt_att = t_end - P.time_s[i]
            if dt_att > 0:
                u, rho, mu, kf, pr, tf, _, liq = jet_fluid(P.pos_M_m[i].copy(), S)
                tn, ql, qa = cool(P, C, S, i, dt_att, float(np.linalg.norm(u)), rho, mu, kf, pr, tf, liq, t_air, t_air)
                P.temp_K[i] = tn
                out[OUT_HEAT_COOLANT_J] += ql
                out[OUT_HEAT_AIR_J] += qa
                P.time_s[i] = t_end
            continue
        t = max(P.time_s[i], t0)
        zero_hits = 0
        while t < t_end:
            x = P.pos_M_m[i].copy()
            v = P.vel_M_m_s[i].copy()
            pg = to_G(x, t)
            u, rho, mu, kf, pr, tf, fscale, liq = jet_fluid(x, S)
            iu, iv = math.floor(pg[0] / cell), math.floor(pg[1] / cell)
            if 0 <= iu < nu and 0 <= iv < nv and H.wet[iu, iv] != 0 and H.base_m[iu, iv] > -math.inf:
                top = H.base_m[iu, iv] + H.bulk_m3[iu, iv] / cell ** 2
                film = S.liquid[5]
                if pg[2] - top <= max(P.dv_m[i], film):
                    fuv = H.film_uv_m_s[iu, iv]
                    immersed = min(1.0, film / max(P.dv_m[i], 1e-300))
                    q_film = 0.5 * immersed * S.liquid[0] * float(fuv @ fuv)
                    q_jet = 0.5 * fscale * rho * float(u @ u) if liq > 0 else 0.0
                    if q_film >= q_jet:
                        u = R[:, 0] * fuv[0] + R[:, 1] * fuv[1]
                        rho, mu, kf, pr, tf = S.liquid[:5]
                        fscale = liq = immersed
            srel = float(np.linalg.norm(v - u))
            kd = fscale * drag_rate(srel, rho, mu, P.dv_m[i], P.area_m2[i], P.chip_mass_kg[i], C.hl[0],
                                    P.hl_a[i], P.hl_b[i], P.hl_c[i], P.hl_d[i])
            ge = g * (1 - liq * S.liquid[0] / rho_p)
            dt = min(t_end - t, dt_max)
            if kd > 0 and drag_fac / kd < dt:
                dt = drag_fac / kd
            last = t + dt >= t_end or t + dt == t
            if last:
                dt = t_end - t
            if dt <= 0:
                break
            x1, v1 = propagate(x, v, u, ge, kd, dt)
            out[OUT_SUBSTEPS] += 1.0
            best, hit = 2.0, 0
            info = {}
            sb, ba, bs = box_exit(x, x1, S.box_lo_M, S.box_hi_M)
            if sb <= 1:
                best, hit, info = sb, 4, {"axis": ba, "side": bs}
            sh, n_g, hk, hiu, hiv = hf_hit(pg, to_G(x1, t + dt), H)
            if hk != 0 and sh < best:
                best, hit, info = sh, 1, {"kind": hk, "iu": hiu, "iv": hiv, "n": R @ n_g}
            km, fm = tick_index(t + 0.5 * dt, t0, dtk, n_t)
            for c in range(len(S.cyl_radius)):
                if S.cyl_radius[c] <= 0:
                    continue
                a = lerp(S.cyl_axis_M[c], km, fm)
                if np.linalg.norm(a) <= 0:
                    continue
                a = a / np.linalg.norm(a)
                sc, n_c, cz, flag = cyl_hit(x, x1, lerp(S.cyl_base_M[c], km, fm), a, S.cyl_radius[c], S.cyl_length[c])
                if flag != 0 and sc < best:
                    best, hit, info = sc, 2, {"c": c, "n": n_c, "z": cz}
            for s_i in range(len(S.sph_radius)):
                ss, n_s, flag = sph_hit(x, x1, S.sph_center_M[s_i], S.sph_radius[s_i])
                if flag != 0 and ss < best:
                    best, hit, info = ss, 3, {"c": s_i, "n": n_s}
            if hit == 0:
                srel1 = float(np.linalg.norm(v1 - u))
                tn, ql, qa = cool(P, C, S, i, dt, 0.5 * (srel + srel1), rho, mu, kf, pr, tf, liq, t_air, t_air)
                P.temp_K[i] = tn
                out[OUT_HEAT_COOLANT_J] += ql
                out[OUT_HEAT_AIR_J] += qa
                P.pos_M_m[i] = x1
                P.vel_M_m_s[i] = v1
                t = t_end if last else t + dt
                zero_hits = 0
                continue
            dth = best * dt
            hx = x + best * (x1 - x)
            if dth > 0:
                _, hv = propagate(x, v, u, ge, kd, dth)
                tn, ql, qa = cool(P, C, S, i, dth, srel, rho, mu, kf, pr, tf, liq, t_air, t_air)
                P.temp_K[i] = tn
                out[OUT_HEAT_COOLANT_J] += ql
                out[OUT_HEAT_AIR_J] += qa
                zero_hits = 0
            else:
                hv = v.copy()
                zero_hits += 1
            t += dth
            out[OUT_HITS] += 1.0
            P.bounces[i] += 1
            P.pos_M_m[i] = hx
            P.vel_M_m_s[i] = hv
            P.time_s[i] = t
            kh, fh = tick_index(t, t0, dtk, n_t)
            if P.bounces[i] > max_bounces or zero_hits > max_hits:
                q = R.T @ (hx - lerp(W.grid_origin_M, kh, fh))
                cu, cv = math.floor(q[0] / cell), math.floor(q[1] / cell)
                write_event(ev, EV_BOUNCE_LIMIT, i, P.slot[i], t, P.mass_kg[i], hx, P.temp_K[i], P.form[i])
                if 0 <= cu < nu and 0 <= cv < nv:
                    settle(P, H, C, i, cu, cv, out)
                else:
                    to_pan(P, C, i, out)
                break
            if hit == 4:
                if (info["axis"] == S.floor[0] and info["side"] == S.floor[1]) or S.walls_wet[0] != 0 \
                        or info["side"] >= 2:
                    to_pan(P, C, i, out)
                    break
                n = np.zeros(3)
                n[info["axis"]] = 1.0 if info["side"] == 0 else -1.0
                P.vel_M_m_s[i] = impulse(hv, np.zeros(3), n, C.e_pair[SURF_ENCLOSURE], C.mu_pair[SURF_ENCLOSURE])
                P.pos_M_m[i] = hx + offset * n
                continue
            if hit == 1:
                s_vel = (W.grid_origin_M[kh + 1] - W.grid_origin_M[kh]) / dtk
                n = info["n"]
                if info["kind"] == 3:
                    top = H.base_m[info["iu"], info["iv"]] + H.bulk_m3[info["iu"], info["iv"]] / cell ** 2
                    lift = top - float((R.T @ (hx - lerp(W.grid_origin_M, kh, fh)))[2])
                    hx = hx + lift * n
                if respond_hf(P, H, W, S, C, out, i, hv, n, info["kind"], info["iu"], info["iv"], s_vel):
                    break
                P.pos_M_m[i] = hx + offset * n
                continue
            if hit == 3:
                c, n = info["c"], info["n"]
                rn = S.sph_radius[c]
                hook = P.length_m[i] >= C.hook_radii[0] * rn
                if S.sph_active[c] != 0:
                    stuck = hook
                else:
                    stuck = hook or C.e_pair[SURF_NOZZLE] * max(-float(hv @ n), 0.0) < capillary_escape_speed(P, S, C, i)
                if stuck:
                    P.status[i] = ON_NOZZLE
                    P.owner[i] = c
                    P.vel_M_m_s[i] = 0.0
                    write_event(ev, EV_NOZZLE, i, c, t, P.mass_kg[i], hx, P.temp_K[i], P.form[i])
                    break
                P.vel_M_m_s[i] = impulse(hv, np.zeros(3), n, C.e_pair[SURF_NOZZLE], C.mu_pair[SURF_NOZZLE])
                P.pos_M_m[i] = S.sph_center_M[c] + (rn + offset + 0.5 * P.thickness_m[i]) * n
                continue
            c, n = info["c"], info["n"]
            b = lerp(S.cyl_base_M[c], kh, fh)
            a = lerp(S.cyl_axis_M[c], kh, fh)
            a = a / np.linalg.norm(a)
            omega = lerp(S.cyl_omega[c], kh, fh)
            radius = S.cyl_radius[c]
            s_vel = omega * np.cross(a, n * radius) + (S.cyl_base_M[c, kh + 1] - S.cyl_base_M[c, kh]) / dtk
            own_first = P.bounces[i] <= 1 and S.cyl_owner[c] == P.slot[i]
            if S.cyl_weld[c] != 0 and P.temp_K[i] > C.weld_temp_K[0] and not own_first:
                P.status[i] = ON_TOOL
                P.owner[i] = S.cyl_owner[c]
                P.vel_M_m_s[i] = s_vel
                write_event(ev, EV_WELD, i, S.cyl_owner[c], t, P.mass_kg[i], hx, P.temp_K[i], P.form[i])
                break
            fi = min(max(int(P.form[i]), 0), abi.N_FORM - 1)
            if (S.cyl_wrap[c] != 0 and C.form_wrap[fi] != 0 and omega != 0 and not own_first
                    and P.length_m[i] >= C.wrap_turns[0] * 2 * math.pi * radius):
                P.status[i] = WRAPPED
                P.owner[i] = S.cyl_owner[c]
                P.vel_M_m_s[i] = s_vel
                write_event(ev, EV_WRAP, i, S.cyl_owner[c], t, P.mass_kg[i], hx, P.temp_K[i], P.form[i])
                break
            e, mu = C.e_pair[S.cyl_surf[c]], C.mu_pair[S.cyl_surf[c]]
            if S.cyl_kind[c] == CYL_PADDLE and omega != 0:
                r = hv - s_vel
                tdir = (1.0 if omega > 0 else -1.0) * np.cross(a, n)
                vrn, vrt, vra = float(r @ n), float(r @ tdir), float(r @ a)
                dn = dtn = 0.0
                if vrn < 0:
                    dn, vrn = -(1 + e) * vrn, -e * vrn
                if vrt < 0:
                    dtn, vrt = -(1 + C.paddle_e[0]) * vrt, -C.paddle_e[0] * vrt
                red = mu * (dn + dtn)
                vra = 0.0 if abs(vra) <= red else vra - math.copysign(red, vra)
                P.vel_M_m_s[i] = s_vel + vrn * n + vrt * tdir + vra * a
            else:
                P.vel_M_m_s[i] = impulse(hv, s_vel, n, e, mu)
            P.pos_M_m[i] = b + info["z"] * a + (radius + offset + 0.5 * P.thickness_m[i]) * n
        if P.status[i] == FLYING:
            P.time_s[i] = t_end


# ============================================================== K2

def relax_pile(H, max_iter, tol_m3, out):
    cell = float(H.cell_m[0])
    area = cell * cell
    nu, nv = H.base_m.shape
    for _ in range(int(max_iter)):
        moved = 0.0
        for iu in range(nu):
            for iv in range(nv):
                if H.bulk_m3[iu, iv] <= 0 or H.mass_kg[iu, iv] <= 0 or H.base_m[iu, iv] == -math.inf:
                    continue
                tan_r = H.repose_mass[iu, iv] / H.mass_kg[iu, iv]
                for ju, jv in ((iu + 1, iv), (iu - 1, iv), (iu, iv + 1), (iu, iv - 1)):
                    if H.bulk_m3[iu, iv] <= 0:
                        break
                    if not (0 <= ju < nu and 0 <= jv < nv):
                        continue
                    h_i = H.bulk_m3[iu, iv] / area
                    if H.base_m[ju, jv] == -math.inf:
                        excess = h_i - 0.5 * tan_r * cell
                        if excess <= 0:
                            continue
                        taken = take_fraction(H, iu, iv, excess / h_i)
                        out[OUT_PAN_MASS_KG] += taken[1]
                        out[OUT_PAN_ENTHALPY_J] += taken[2]
                        out[OUT_PAN_BULK_M3] += taken[0]
                        out[OUT_PAN_LONG_KG] += taken[8]
                        moved += taken[0]
                        continue
                    excess = (H.base_m[iu, iv] + h_i) - (H.base_m[ju, jv] + H.bulk_m3[ju, jv] / area) - tan_r * cell
                    if excess <= 0:
                        continue
                    dvol = min(0.5 * excess * area, H.bulk_m3[iu, iv])
                    frac = dvol / H.bulk_m3[iu, iv]
                    for name in _EXTENSIVE:
                        arr = getattr(H, name)
                        d = arr[iu, iv] * frac
                        arr[iu, iv] -= d
                        arr[ju, jv] += d
                    moved += dvol
        if moved <= tol_m3:
            break


# ============================================================== K3

def fill_occupancy(H, P, R_M_S, o_S_M, v_slow, occ):
    occ.mass_kg[:] = 0.0
    occ.size_m[:] = 0.0
    occ.thickness_m[:] = 0.0
    cell = float(H.cell_m[0])
    oc = float(occ.cell_m[0])
    dims = np.array(occ.mass_kg.shape)
    kw = int(np.argmax(np.abs(H.axes_S[:, 2])))
    others = [k for k in range(3) if k != kw]
    iu, iv = np.nonzero((H.mass_kg > 0) & (H.base_m > -np.inf))
    for a, b in zip(iu, iv):
        m = H.mass_kg[a, b]
        h = H.bulk_m3[a, b] / (cell * cell)
        uv = np.array([(a + 0.5) * cell, (b + 0.5) * cell])
        pa = H.origin_G_m + H.axes_S @ np.array([uv[0], uv[1], H.base_m[a, b]])
        pb = H.origin_G_m + H.axes_S @ np.array([uv[0], uv[1], H.base_m[a, b] + h])
        idx = [0, 0, 0]
        ok = True
        for k in others:
            idx[k] = math.floor((pa[k] - occ.origin_S_m[k]) / oc)
            ok &= 0 <= idx[k] < dims[k]
        if not ok:
            continue
        lo, hi = sorted((pa[kw], pb[kw]))
        size, thick = H.size_mass[a, b] / m, H.thick_mass[a, b] / m
        if hi <= lo:
            j = math.floor((lo - occ.origin_S_m[kw]) / oc)
            if 0 <= j < dims[kw]:
                idx[kw] = j
                occ.mass_kg[tuple(idx)] += m
                occ.size_m[tuple(idx)] += m * size
                occ.thickness_m[tuple(idx)] += m * thick
            continue
        edges = occ.origin_S_m[kw] + oc * np.arange(dims[kw] + 1)
        overlap = np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo), 0.0, None)
        for j in np.nonzero(overlap > 0)[0]:
            dm = m * overlap[j] / (hi - lo)
            idx[kw] = int(j)
            occ.mass_kg[tuple(idx)] += dm
            occ.size_m[tuple(idx)] += dm * size
            occ.thickness_m[tuple(idx)] += dm * thick
    n = int(P.n[0])
    sel = np.nonzero((P.status[:n] == FLYING) & (np.linalg.norm(P.vel_M_m_s[:n], axis=1) < v_slow))[0]
    for i in sel:
        ps = R_M_S.T @ (P.pos_M_m[i] - o_S_M)
        j = np.floor((ps - occ.origin_S_m) / oc).astype(np.int64)
        if np.any(j < 0) or np.any(j >= dims):
            continue
        occ.mass_kg[tuple(j)] += P.mass_kg[i]
        occ.size_m[tuple(j)] += P.mass_kg[i] * P.dv_m[i]
        occ.thickness_m[tuple(j)] += P.mass_kg[i] * P.thickness_m[i]
    full = occ.mass_kg > 0
    occ.size_m[full] /= occ.mass_kg[full]
    occ.thickness_m[full] /= occ.mass_kg[full]


# ============================================================== K4

def flush_pile(H, fp, R_M_G, rho_f, g, dt, rec, n):
    cell = float(H.cell_m[0])
    c_f, theta_c0, a_wp, expo, cap, gamma, mu_bed, rho_s = (float(v) for v in fp)
    nu, nv = H.base_m.shape
    for iu in range(nu):
        for iv in range(nv):
            m, q = H.mass_kg[iu, iv], H.flush_Pa[iu, iv]
            if m <= 0 or q <= 0 or H.base_m[iu, iv] == -np.inf:
                continue
            d = H.size_mass[iu, iv] / m
            dens = (rho_s - rho_f) * g * d
            if d <= 0 or dens <= 0:
                continue
            theta = c_f * q / dens
            theta_c = theta_c0 + mu_bed * cap * gamma / d / dens
            if theta <= theta_c:
                continue
            if n[0] >= rec.shape[0]:
                n[1] = 1
                continue
            qs = a_wp * (theta - theta_c) ** expo * math.sqrt((rho_s / rho_f - 1) * g * d ** 3)
            frac = min(rho_s * qs * cell * dt / m, 1.0)
            taken = take_fraction(H, iu, iv, frac)
            vel = R_M_G[:, :2] @ H.film_uv_m_s[iu, iv]
            top = H.base_m[iu, iv] + H.bulk_m3[iu, iv] / cell ** 2
            write_mob(rec, n, MOB_FLUSH, iu, iv, taken, vel, ((iu + 0.5) * cell, (iv + 0.5) * cell, top + 0.5 * d),
                      -1.0, -1.0)


# ============================================================== K5

def pile_heat(H, C, hp, dt, q_surface, out):
    cell = float(H.cell_m[0])
    area = cell * cell
    h_bed, h_air, h_wet, t_air, t_cool = (float(v) for v in hp[:5])
    for iu, iv in zip(*np.nonzero(H.mass_kg > 0)):
        m = H.mass_kg[iu, iv]
        temp = temperature(H.enthalpy_J[iu, iv] / m, C.cp_T_K, C.cp_J_kgK, C.h_J_kg)
        h_b = h_bed if H.base_kind[iu, iv] != SURF_NONE else 0.0
        t_s = H.surf_temp_K[iu, iv]
        wet = H.wet[iu, iv] != 0
        if wet:
            h_t, t_t = h_wet, t_cool
        else:
            h_t = h_air + C.emissivity[0] * STEFAN_BOLTZMANN_W_M2K4 * (temp ** 2 + t_air ** 2) * (temp + t_air)
            t_t = t_air
        h_tot = h_b + h_t
        if h_tot <= 0:
            continue
        t_inf = (h_b * t_s + h_t * t_t) / h_tot
        cp = cp_at(temp, C.cp_T_K, C.cp_J_kgK)
        t_new = t_inf + (temp - t_inf) * math.exp(-h_tot * area * dt / (m * cp))
        q = H.enthalpy_J[iu, iv] - m * enthalpy(t_new, C.cp_T_K, C.cp_J_kgK, C.h_J_kg)
        fb, ft = h_b * (temp - t_s), h_t * (temp - t_t)
        q_b = q * fb / (fb + ft) if fb + ft != 0 else 0.0
        H.enthalpy_J[iu, iv] -= q
        q_surface[iu, iv] += q_b
        out[0 if wet else 1] += q - q_b


# ============================================================== K7

def column_in_cylinder(uc, vc, b, a, radius, length):
    d0 = np.array([uc - b[0], vc - b[1], -b[2]])
    z0 = float(d0 @ a)
    r0 = d0 - z0 * a
    r1 = np.array([0.0, 0.0, 1.0]) - a[2] * a
    aa = float(r1 @ r1)
    cc = float(r0 @ r0) - radius * radius
    if aa < _PARALLEL_EPS:
        if cc > 0:
            return 1.0, 0.0
        lo, hi = -math.inf, math.inf
    else:
        bb = float(r0 @ r1)
        disc = bb * bb - aa * cc
        if disc < 0:
            return 1.0, 0.0
        lo, hi = (-bb - math.sqrt(disc)) / aa, (-bb + math.sqrt(disc)) / aa
    if abs(a[2]) < _PARALLEL_EPS:
        if z0 < 0 or z0 > length:
            return 1.0, 0.0
    else:
        za, zb = sorted(((0 - z0) / a[2], (length - z0) / a[2]))
        lo, hi = max(lo, za), min(hi, zb)
    return (lo, hi) if hi > lo else (1.0, 0.0)


def sweep_cylinders(H, W, S, C, rec, n):
    cell = float(H.cell_m[0])
    area = cell * cell
    nu, nv = H.base_m.shape
    R = W.R_M_G
    n_t = W.grid_origin_M.shape[0] - 1
    for c in range(len(S.cyl_radius)):
        if S.cyl_owner[c] < 0 or S.cyl_radius[c] <= 0:
            continue
        radius, length = S.cyl_radius[c], S.cyl_length[c]
        for k in range(n_t + 1):
            bg = R.T @ (S.cyl_base_M[c, k] - W.grid_origin_M[k])
            am = S.cyl_axis_M[c, k]
            if np.linalg.norm(am) <= 0:
                continue
            am = am / np.linalg.norm(am)
            ag = R.T @ am
            end = bg + length * ag
            i0 = max(math.floor((min(bg[0], end[0]) - radius) / cell), 0)
            i1 = min(math.floor((max(bg[0], end[0]) + radius) / cell), nu - 1)
            j0 = max(math.floor((min(bg[1], end[1]) - radius) / cell), 0)
            j1 = min(math.floor((max(bg[1], end[1]) + radius) / cell), nv - 1)
            omega = S.cyl_omega[c, k]
            kk = k if k < n_t else n_t - 1
            tv = (S.cyl_base_M[c, kk + 1] - S.cyl_base_M[c, kk]) / W.dt_tick_s[0]
            t_k = W.t0_s[0] + k * W.dt_tick_s[0]
            for iu in range(i0, i1 + 1):
                for iv in range(j0, j1 + 1):
                    if H.mass_kg[iu, iv] <= 0 or H.base_m[iu, iv] == -np.inf:
                        continue
                    uc, vc = (iu + 0.5) * cell, (iv + 0.5) * cell
                    lo, hi = column_in_cylinder(uc, vc, bg, ag, radius, length)
                    if hi <= lo:
                        continue
                    base = H.base_m[iu, iv]
                    h = H.bulk_m3[iu, iv] / area
                    lo2, hi2 = max(lo, base), min(hi, base + h)
                    if hi2 <= lo2:
                        continue
                    if n[0] >= rec.shape[0]:
                        n[1] = 1
                        continue
                    d = np.array([uc, vc, 0.5 * (lo2 + hi2)]) - bg
                    za = float(d @ ag)
                    r = d - za * ag
                    rm = float(np.linalg.norm(r))
                    if rm > 0:
                        r = r * (radius / rm)
                    r_m = R @ r
                    sv = omega * np.cross(am, r_m)
                    if S.cyl_kind[c] == CYL_PADDLE:
                        vel = (1 + C.paddle_e[0]) * sv + tv
                    elif rm > 0:
                        surf = S.cyl_surf[c]
                        vel = impulse(np.zeros(3), sv + tv, r_m / radius, C.e_pair[surf], C.mu_pair[surf])
                    else:
                        vel = tv.copy()
                    size = H.size_mass[iu, iv] / H.mass_kg[iu, iv]
                    taken = take_fraction(H, iu, iv, (hi2 - lo2) / h)
                    out_r = (radius + 0.5 * size) / radius if rm > 0 else 1.0
                    release = bg + za * ag + r * out_r
                    write_mob(rec, n, MOB_SWEEP, iu, iv, taken, vel, release, float(S.cyl_owner[c]), t_k)


# ============================================================== K8

def recut_take(H, req, rec, n):
    nu, nv = H.base_m.shape
    order = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    missing = 0.0
    for row in req:
        want = row[2]
        for du, dv in order:
            if want <= 0:
                break
            iu, iv = int(row[0]) + du, int(row[1]) + dv
            if not (0 <= iu < nu and 0 <= iv < nv) or H.mass_kg[iu, iv] <= 0:
                continue
            if n[0] >= rec.shape[0]:
                n[1] = 1
                break
            taken = take_fraction(H, iu, iv, min(want / H.mass_kg[iu, iv], 1.0))
            want -= taken[1]
            write_mob(rec, n, MOB_RECUT, iu, iv, taken, row[3:6], row[8:11], row[6], row[7])
        if want > 0:
            missing += want
    return missing


# ============================================================== drop, draws, emission rows

def drop_particles(P, H, W, C, out):
    cell = float(H.cell_m[0])
    nu, nv = H.base_m.shape
    o = W.grid_origin_M[-1]
    for i in range(int(P.n[0])):
        if P.status[i] != FLYING:
            continue
        g = W.R_M_G.T @ (P.pos_M_m[i] - o)
        iu, iv = math.floor(g[0] / cell), math.floor(g[1] / cell)
        if 0 <= iu < nu and 0 <= iv < nv and H.base_m[iu, iv] > -np.inf and g[2] >= H.base_m[iu, iv]:
            settle(P, H, C, i, iu, iv, out)
        else:
            to_pan(P, C, i, out)


def film_field(q, cell, rho, film):
    film[:] = 0.0
    if q.shape[0] < 2 or q.shape[1] < 2:
        return
    gu, gv = np.gradient(q, cell)
    mag = np.hypot(gu, gv)
    speed = np.sqrt(2.0 * np.maximum(q, 0.0) / rho)
    ok = (q > 0) & (mag > 0)
    film[..., 0] = np.where(ok, -gu / np.where(ok, mag, 1.0) * speed, 0.0)
    film[..., 1] = np.where(ok, -gv / np.where(ok, mag, 1.0) * speed, 0.0)


_MASK = (1 << 64) - 1


def _splitmix64(x: int) -> int:
    z = (x + 0x9E3779B97F4A7C15) & _MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


def hash_uniform(seed: int, stream_id: int, a: int, b: int, c: int) -> float:
    h = _splitmix64(seed & _MASK)
    for value in (stream_id, a, b, c):
        h = _splitmix64(h ^ (value & _MASK))
    return (h >> 11) * (1.0 / 9007199254740992.0)


def hash_normal(seed, stream_id, a, b, c) -> float:
    u1 = max(hash_uniform(seed, stream_id, a, b, 2 * c), 1e-300)
    u2 = hash_uniform(seed, stream_id, a, b, 2 * c + 1)
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def hash_draws(seed, stream_id, keys, b, c0, uniform, normal):
    for i, key in enumerate(keys):
        for j in range(uniform.shape[1]):
            uniform[i, j] = hash_uniform(int(seed), int(stream_id), int(key), int(b), int(c0) + j)
        for j in range(normal.shape[1]):
            normal[i, j] = hash_normal(int(seed), int(stream_id), int(key), int(b), int(c0) + j)


def write_emission(rec, n, *values):
    i = int(n[0])
    if i >= rec.shape[0]:
        n[1] = 1
        return
    rec[i] = values
    n[0] = i + 1
