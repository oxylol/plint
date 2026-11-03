"""GPU twin of the chips flight kernel K1 ``cnctwin.chips.kernels.advance_particles`` (docs/models/gpu.md
"Particle step").

K1 advances every particle through the window (gravity, Haider-Levenspiel drag in air, jets and the wall
film with the exact exponential integrator, Ranz-Marshall cooling and radiation, contacts with the
enclosure, the pile height field, rotating cylinders and nozzles) one particle after the other. Its only
couplings between particles are shared state: the heat and count accumulators ``out``, the event buffer,
and the pile, which a settling particle raises under the particles after it.

The GPU path keeps K1's per-particle arithmetic and its order and moves the couplings to an ordered host
commit:

* :func:`flight` - one particle through the window, in K1's operation order, calling chips' own njit
  helpers (``propagate``, ``drag_rate``, ``cool``'s pieces, ``impulse``, ``cyl_hit``, ``sph_hit``,
  ``box_exit``, ``tick_index``, ``lerp3``) as twins. Instead of touching shared state it records: the
  heat addends of every sub-step (a log), its sub-step and hit counts, at most one event, the terminal
  action (settle into a pile cell, or the pan), and the box of pile cells it read. The pile it reads is
  the pile at the start of the call.
* :data:`SOURCES` ``k_flight`` - one GPU thread per particle runs :func:`flight` (compiled with
  ``cuda.jit`` by :mod:`cnctwin.gpu.twins`; the CPU twins with ``numba.njit``, cached).
* :func:`commit` (host, compiled from the same source for the CPU) - in particle order: writes the
  particle back, replays its heat addends into ``out`` (K1's summation order), writes its event and
  applies its settle / pan with chips' own ``settle`` / ``to_pan``. A particle whose box of read cells
  contains a cell an earlier particle of this call settled into, or whose log overflowed, is recomputed
  right there on the host by the CPU twin of :func:`flight` against the live pile: exactly K1's sequential
  semantics.

So the result equals K1's bit for bit wherever the device's transcendental functions (``exp``, ``expm1``,
``pow``) round like the host's libm (always in the CUDA simulator); on a real device libdevice differs
from glibc by up to 2 ulp in those functions (CUDA C Programming Guide, Table "Double-Precision
Mathematical Functions"), which propagates to the trajectory by about that relative amount per sub-step
until a contact decision flips (docs/models/gpu.md "Tolerances").
"""

from __future__ import annotations

import math

import numpy as np
from numba import types

from ..chips import abi as CA
from ..chips.abi import (CYL_PADDLE, EV_BOUNCE_LIMIT, EV_NOZZLE, EV_WELD, EV_WRAP, FLYING, FREE, ON_NOZZLE, ON_TOOL,
                         OUT_HEAT_AIR_J, OUT_HEAT_COOLANT_J, OUT_HITS, OUT_SUBSTEPS, SURF_ENCLOSURE, SURF_NONE,
                         SURF_NOZZLE, SURF_PILE, WRAPPED)
from ..chips.kernels import (_INF, _LN2, _PARALLEL_EPS, _TINY, box_exit, cp_at, cyl_hit, drag_rate, enthalpy, impulse,
                             lerp1, lerp3, nusselt_h, propagate, settle, sph_hit, tick_index, to_pan, write_event)
from ..units import STEFAN_BOLTZMANN_W_M2K4
from . import device as D
from .twins import builder, twin

N_FORM = CA.N_FORM

# ------------------------------------------------------------------ packed particle inputs pf [cap, NPF]
PF_X, PF_Y, PF_Z, PF_VX, PF_VY, PF_VZ, PF_TIME, PF_MASS, PF_TEMP = 0, 1, 2, 3, 4, 5, 6, 7, 8
PF_THICK, PF_WIDTH, PF_LENGTH, PF_DV, PF_CHIP_MASS, PF_AREA, PF_SURFACE = 9, 10, 11, 12, 13, 14, 15
PF_HLA, PF_HLB, PF_HLC, PF_HLD = 16, 17, 18, 19
NPF = 20
# pi [cap, NPI]
PI_FORM, PI_STATUS, PI_OWNER, PI_SLOT, PI_BOUNCES = 0, 1, 2, 3, 4
NPI = 5
# ------------------------------------------------------------------ results rf [cap, NRF], ri [cap, NRI]
RF_X, RF_Y, RF_Z, RF_VX, RF_VY, RF_VZ, RF_TIME, RF_TEMP = 0, 1, 2, 3, 4, 5, 6, 7
RF_EV_TIME, RF_EV_X, RF_EV_Y, RF_EV_Z, RF_EV_TEMP, RF_SUBSTEPS, RF_HITS = 8, 9, 10, 11, 12, 13, 14
NRF = 16
RI_DONE, RI_STATUS, RI_OWNER, RI_BOUNCES, RI_TERM, RI_TERM_IU, RI_TERM_IV = 0, 1, 2, 3, 4, 5, 6
RI_EV_KIND, RI_EV_OWNER, RI_NLOG, RI_BOX_U0, RI_BOX_U1, RI_BOX_V0, RI_BOX_V1 = 7, 8, 9, 10, 11, 12, 13
NRI = 16
TERM_NONE, TERM_SETTLE, TERM_PAN = 0, 1, 2
NO_BOX = 1 << 40               # box start when no pile cell was read
# ------------------------------------------------------------------ scalars fpar [NFP], ipar [NIP]
FP_T0, FP_TEND, FP_DTK, FP_GX, FP_GY, FP_GZ = 0, 1, 2, 3, 4, 5
FP_BOX_LO, FP_BOX_HI = 6, 9                  # 3 each
FP_AIR = 12                                  # 5: rho, mu, k, Pr, T
FP_LIQ = 17                                  # 7: rho, mu, k, Pr, T, film thickness, surface tension
FP_JC = 24                                   # 2: round-jet B, S
FP_RHO, FP_EMIS, FP_WELD, FP_HL0 = 26, 27, 28, 29
FP_RM = 30                                   # 4
FP_PADDLE = 34
FP_WET = 35                                  # 3
FP_SETTLE, FP_WRAP_TURNS, FP_HOOK = 38, 39, 40
FP_NUM = 41                                  # 5: dt_max, drag factor, offset, max bounces, max zero hits
FP_CELL = 46
NFP = 48
IP_FLOOR_AXIS, IP_FLOOR_SIDE, IP_WALLS_WET, IP_NCYL, IP_NSPH, IP_NJET = 0, 1, 2, 3, 4, 5
NIP = 8
# cylinder / sphere / jet packed columns
CF_RADIUS, CF_LENGTH = 0, 1
CI_KIND, CI_SURF, CI_OWNER, CI_WRAP, CI_WELD = 0, 1, 2, 3, 4
JF_SPEED, JF_R0, JF_TAN, JF_RHO, JF_MU, JF_K, JF_PR = 0, 1, 2, 3, 4, 5, 6
CAP_LOG = 128                  # heat addends a GPU thread can log (more: the host recomputes the particle)
HOST_LOG = 1 << 14             # first capacity of the host recompute log (doubled until it fits)


# ============================================================================ per-particle helpers (twins)
@twin
def p_jet_fluid(x, y, z, jet_o, jet_d, jet_f, jet_gas, n_jet, fpar):
    """chips.kernels.jet_fluid on packed jets (same operations, same order)."""
    best = -1.0
    ux = 0.0
    uy = 0.0
    uz = 0.0
    rho = fpar[FP_AIR + 0]
    mu = fpar[FP_AIR + 1]
    kf = fpar[FP_AIR + 2]
    pr = fpar[FP_AIR + 3]
    tf = fpar[FP_AIR + 4]
    fscale = 1.0
    liq = 0.0
    for j in range(n_jet):
        sp = jet_f[j, JF_SPEED]
        if sp <= 0.0:
            continue
        dx = x - jet_o[j, 0]
        dy = y - jet_o[j, 1]
        dz = z - jet_o[j, 2]
        ex = jet_d[j, 0]
        ey = jet_d[j, 1]
        ez = jet_d[j, 2]
        s = dx * ex + dy * ey + dz * ez
        if s <= 0.0:
            continue
        px = dx - s * ex
        py = dy - s * ey
        pz = dz - s * ez
        r = math.sqrt(px * px + py * py + pz * pz)
        r0 = jet_f[j, JF_R0]
        if jet_gas[j] == 1:
            d0 = 2.0 * r0
            uc = sp if s <= fpar[FP_JC + 0] * d0 else sp * fpar[FP_JC + 0] * d0 / s
            rh = max(r0, fpar[FP_JC + 1] * s)
            u = uc * math.exp(-_LN2 * (r / rh) * (r / rh))
            q = 0.5 * jet_f[j, JF_RHO] * u * u
            if q > best:
                best = q
                ux = u * ex
                uy = u * ey
                uz = u * ez
                rho = jet_f[j, JF_RHO]
                mu = jet_f[j, JF_MU]
                kf = jet_f[j, JF_K]
                pr = jet_f[j, JF_PR]
                tf = fpar[FP_AIR + 4]
                fscale = 1.0
                liq = 0.0
        else:
            rj = r0 + s * jet_f[j, JF_TAN]
            if r > rj:
                continue
            alpha = (r0 / rj) * (r0 / rj)
            q = 0.5 * alpha * jet_f[j, JF_RHO] * sp * sp
            if q > best:
                best = q
                ux = sp * ex
                uy = sp * ey
                uz = sp * ez
                rho = jet_f[j, JF_RHO]
                mu = jet_f[j, JF_MU]
                kf = jet_f[j, JF_K]
                pr = jet_f[j, JF_PR]
                tf = fpar[FP_LIQ + 4]
                fscale = alpha
                liq = alpha
    return ux, uy, uz, rho, mu, kf, pr, tf, fscale, liq


@twin
def p_cool(temp, m_chip, area, dv, mass, dt, speed, rho, mu, kf, pr, tf, liq, t_rad, t_air, cp_T, cp_v, h_t, fpar):
    """chips.kernels.cool for one particle's values: (T_new, Q_to_liquid, Q_to_air)."""
    if m_chip <= 0.0 or area <= 0.0 or dt <= 0.0:
        return temp, 0.0, 0.0
    h_l = 0.0
    if liq > 0.0:
        h_l = liq * nusselt_h(speed, rho, mu, kf, pr, dv, fpar[FP_RM + 0], fpar[FP_RM + 1], fpar[FP_RM + 2],
                              fpar[FP_RM + 3])
    h_a = (1.0 - liq) * nusselt_h(speed, fpar[FP_AIR + 0], fpar[FP_AIR + 1], fpar[FP_AIR + 2], fpar[FP_AIR + 3], dv,
                                  fpar[FP_RM + 0], fpar[FP_RM + 1], fpar[FP_RM + 2], fpar[FP_RM + 3])
    eps = fpar[FP_EMIS]
    h_r = eps * STEFAN_BOLTZMANN_W_M2K4 * (temp * temp + t_rad * t_rad) * (temp + t_rad)
    h_tot = h_l + h_a + h_r
    if h_tot <= 0.0:
        return temp, 0.0, 0.0
    t_inf = (h_l * tf + h_a * t_air + h_r * t_rad) / h_tot
    cp = cp_at(temp, cp_T, cp_v)
    t_new = t_inf + (temp - t_inf) * math.exp(-h_tot * area * dt / (m_chip * cp))
    q = mass * (enthalpy(temp, cp_T, cp_v, h_t) - enthalpy(t_new, cp_T, cp_v, h_t))
    fl = h_l * (temp - tf)
    fa = h_a * (temp - t_air) + h_r * (temp - t_rad)
    tot = fl + fa
    q_l = 0.0
    if tot != 0.0:
        q_l = q * fl / tot
    return t_new, q_l, q - q_l


@twin
def p_escape(width, length, m_chip, fpar):
    """chips.kernels.capillary_escape_speed."""
    perim = 2.0 * (width + length)
    f_cap = fpar[FP_WET + 2] * fpar[FP_LIQ + 6] * fpar[FP_WET + 1] * perim
    if m_chip <= 0.0 or f_cap <= 0.0:
        return 0.0
    return math.sqrt(2.0 * f_cap * fpar[FP_LIQ + 5] / m_chip)


@twin
def p_hf_hit(u0, v0, w0, u1, v1, w1, base_m, bulk_m3, cell):
    """chips.kernels.hf_hit plus the first and last pile cell it read: (s, n_u, n_v, n_w, kind, iu, iv,
    read_u0, read_v0, read_u1, read_v1); read_u0 = -1 when no cell was read."""
    nu = base_m.shape[0]
    nv = base_m.shape[1]
    du = u1 - u0
    dvv = v1 - v0
    dw = w1 - w0
    s_lo = 0.0
    s_hi = 1.0
    face = 0
    if abs(du) < _PARALLEL_EPS:
        if u0 < 0.0 or u0 >= nu * cell:
            return 2.0, 0.0, 0.0, 0.0, 0, -1, -1, -1, -1, -1, -1
    else:
        sa = (0.0 - u0) / du
        sb = (nu * cell - u0) / du
        if sa > sb:
            sa, sb = sb, sa
        if sa > s_lo:
            s_lo = sa
            face = 1
        if sb < s_hi:
            s_hi = sb
    if abs(dvv) < _PARALLEL_EPS:
        if v0 < 0.0 or v0 >= nv * cell:
            return 2.0, 0.0, 0.0, 0.0, 0, -1, -1, -1, -1, -1, -1
    else:
        sa = (0.0 - v0) / dvv
        sb = (nv * cell - v0) / dvv
        if sa > sb:
            sa, sb = sb, sa
        if sa > s_lo:
            s_lo = sa
            face = 2
        if sb < s_hi:
            s_hi = sb
    if s_lo > s_hi:
        return 2.0, 0.0, 0.0, 0.0, 0, -1, -1, -1, -1, -1, -1
    s = s_lo
    iu = int(np.floor((u0 + s * du) / cell))
    iv = int(np.floor((v0 + s * dvv) / cell))
    iu = min(max(iu, 0), nu - 1)
    iv = min(max(iv, 0), nv - 1)
    ru0 = iu
    rv0 = iv
    step_u = 1 if du > 0.0 else (-1 if du < 0.0 else 0)
    step_v = 1 if dvv > 0.0 else (-1 if dvv < 0.0 else 0)
    if step_u != 0:
        ub = (iu + (1 if step_u > 0 else 0)) * cell
        tmu = (ub - u0) / du
        tdu = cell / abs(du)
    else:
        tmu = _INF
        tdu = _INF
    if step_v != 0:
        vb = (iv + (1 if step_v > 0 else 0)) * cell
        tmv = (vb - v0) / dvv
        tdv = cell / abs(dvv)
    else:
        tmv = _INF
        tdv = _INF
    piu = -1
    piv = -1
    while True:
        s_out = min(tmu, tmv, s_hi)
        base = base_m[iu, iv]
        if base > -_INF:
            top = base + bulk_m3[iu, iv] / (cell * cell)
            w_in = w0 + s * dw
            if w_in < top:
                if face == 0:
                    return s, 0.0, 0.0, 1.0, 3, iu, iv, ru0, rv0, iu, iv
                if face == 1:
                    return s, -float(step_u), 0.0, 0.0, 2, piu, piv, ru0, rv0, iu, iv
                return s, 0.0, -float(step_v), 0.0, 2, piu, piv, ru0, rv0, iu, iv
            w_out = w0 + s_out * dw
            if w_out < top and dw < 0.0:
                return (top - w0) / dw, 0.0, 0.0, 1.0, 1, iu, iv, ru0, rv0, iu, iv
        if s_out >= s_hi:
            break
        piu = iu
        piv = iv
        if tmu <= tmv:
            iu += step_u
            s = tmu
            tmu += tdu
            face = 1
        else:
            iv += step_v
            s = tmv
            tmv += tdv
            face = 2
        if iu < 0 or iu >= nu or iv < 0 or iv >= nv:
            return 2.0, 0.0, 0.0, 0.0, 0, -1, -1, ru0, rv0, piu, piv
    return 2.0, 0.0, 0.0, 0.0, 0, -1, -1, ru0, rv0, iu, iv


@twin
def p_box(ri, i, iu, iv):
    """Grow particle i's box of read pile cells by cell (iu, iv)."""
    if iu < ri[i, RI_BOX_U0]:
        ri[i, RI_BOX_U0] = iu
    if iu > ri[i, RI_BOX_U1]:
        ri[i, RI_BOX_U1] = iu
    if iv < ri[i, RI_BOX_V0]:
        ri[i, RI_BOX_V0] = iv
    if iv > ri[i, RI_BOX_V1]:
        ri[i, RI_BOX_V1] = iv


@twin
def p_log(ri, i, log, ql, qa):
    """Append one heat addend pair; False (and RI_NLOG = -1) when the log is full."""
    n = ri[i, RI_NLOG]
    if n < 0:
        return False
    if n >= log.shape[0]:
        ri[i, RI_NLOG] = -1
        return False
    log[n, 0] = ql
    log[n, 1] = qa
    ri[i, RI_NLOG] = n + 1
    return True


@twin
def flight(i, pf, pi, base_m, base_kind, bulk_m3, wet, film_uv, origin, R, box, cyl_base, cyl_axis, cyl_omega, cyl_f,
           cyl_i, sph_c, sph_r, sph_a, jet_o, jet_d, jet_f, jet_gas, cp_T, cp_v, h_t, form_wrap, e_pair, mu_pair, fpar,
           ipar, rf, ri, log):
    """K1 for particle i (chips.kernels.advance_particles, its loop body), shared state deferred: see the
    module doc. Results in rf[i], ri[i] and log (heat addends)."""
    for q in range(rf.shape[1]):
        rf[i, q] = 0.0
    for q in range(ri.shape[1]):
        ri[i, q] = 0
    ri[i, RI_BOX_U0] = NO_BOX
    ri[i, RI_BOX_U1] = -NO_BOX
    ri[i, RI_BOX_V0] = NO_BOX
    ri[i, RI_BOX_V1] = -NO_BOX
    st = pi[i, PI_STATUS]
    owner = pi[i, PI_OWNER]
    bounces = pi[i, PI_BOUNCES]
    x = pf[i, PF_X]
    y = pf[i, PF_Y]
    z = pf[i, PF_Z]
    vx = pf[i, PF_VX]
    vy = pf[i, PF_VY]
    vz = pf[i, PF_VZ]
    time_out = pf[i, PF_TIME]
    temp = pf[i, PF_TEMP]
    if st == FREE:
        return
    ri[i, RI_DONE] = 1
    t0 = fpar[FP_T0]
    t_end = fpar[FP_TEND]
    dtk = fpar[FP_DTK]
    n_t = origin.shape[0] - 1
    gx = fpar[FP_GX]
    gy = fpar[FP_GY]
    gz = fpar[FP_GZ]
    dt_max = fpar[FP_NUM + 0]
    drag_fac = fpar[FP_NUM + 1]
    offset = fpar[FP_NUM + 2]
    max_bounces = int(fpar[FP_NUM + 3])
    max_hits = int(fpar[FP_NUM + 4])
    rho_p = fpar[FP_RHO]
    t_air = fpar[FP_AIR + 4]
    cell = fpar[FP_CELL]
    nu = base_m.shape[0]
    nv = base_m.shape[1]
    n_cyl = ipar[IP_NCYL]
    n_sph = ipar[IP_NSPH]
    mass = pf[i, PF_MASS]
    m_chip = pf[i, PF_CHIP_MASS]
    surface = pf[i, PF_SURFACE]
    dv = pf[i, PF_DV]
    form = pi[i, PI_FORM]
    slot = pi[i, PI_SLOT]
    term = TERM_NONE
    substeps = 0.0
    hits = 0.0
    if st != FLYING:
        dt_att = t_end - time_out
        if dt_att > 0.0:
            fx = p_jet_fluid(x, y, z, jet_o, jet_d, jet_f, jet_gas, ipar[IP_NJET], fpar)
            tn, ql, qa = p_cool(temp, m_chip, surface, dv, mass, dt_att, math.sqrt(fx[0] ** 2 + fx[1] ** 2 + fx[2] ** 2),
                                fx[3], fx[4], fx[5], fx[6], fx[7], fx[9], t_air, t_air, cp_T, cp_v, h_t, fpar)
            temp = tn
            if not p_log(ri, i, log, ql, qa):
                return
            time_out = t_end
    else:
        t = time_out
        if t < t0:
            t = t0
        zero_hits = 0
        while t < t_end:
            k0, f0 = tick_index(t, t0, dtk, n_t)
            ox, oy, oz = lerp3(origin, k0, f0)
            qx = x - ox
            qy = y - oy
            qz = z - oz
            pu = R[0, 0] * qx + R[1, 0] * qy + R[2, 0] * qz
            pv = R[0, 1] * qx + R[1, 1] * qy + R[2, 1] * qz
            pw = R[0, 2] * qx + R[1, 2] * qy + R[2, 2] * qz
            fl = p_jet_fluid(x, y, z, jet_o, jet_d, jet_f, jet_gas, ipar[IP_NJET], fpar)
            ux = fl[0]
            uy = fl[1]
            uz = fl[2]
            rho = fl[3]
            mu = fl[4]
            kf = fl[5]
            pr = fl[6]
            tf = fl[7]
            fscale = fl[8]
            liq = fl[9]
            iu = int(np.floor(pu / cell))
            iv = int(np.floor(pv / cell))
            if 0 <= iu < nu and 0 <= iv < nv and wet[iu, iv] != 0 and base_m[iu, iv] > -_INF:
                p_box(ri, i, iu, iv)
                top = base_m[iu, iv] + bulk_m3[iu, iv] / (cell * cell)
                film = fpar[FP_LIQ + 5]
                if pw - top <= max(dv, film):
                    fu = film_uv[iu, iv, 0]
                    fv = film_uv[iu, iv, 1]
                    immersed = min(1.0, film / max(dv, _TINY))
                    q_film = 0.5 * immersed * fpar[FP_LIQ + 0] * (fu * fu + fv * fv)
                    q_jet = 0.5 * fscale * rho * (ux * ux + uy * uy + uz * uz) if liq > 0.0 else 0.0
                    if q_film >= q_jet:
                        ux = R[0, 0] * fu + R[0, 1] * fv
                        uy = R[1, 0] * fu + R[1, 1] * fv
                        uz = R[2, 0] * fu + R[2, 1] * fv
                        rho = fpar[FP_LIQ + 0]
                        mu = fpar[FP_LIQ + 1]
                        kf = fpar[FP_LIQ + 2]
                        pr = fpar[FP_LIQ + 3]
                        tf = fpar[FP_LIQ + 4]
                        fscale = immersed
                        liq = immersed
            rvx = vx - ux
            rvy = vy - uy
            rvz = vz - uz
            srel = math.sqrt(rvx * rvx + rvy * rvy + rvz * rvz)
            kd = fscale * drag_rate(srel, rho, mu, dv, pf[i, PF_AREA], m_chip, fpar[FP_HL0], pf[i, PF_HLA],
                                    pf[i, PF_HLB], pf[i, PF_HLC], pf[i, PF_HLD])
            bfac = 1.0 - liq * fpar[FP_LIQ + 0] / rho_p
            dt = t_end - t
            if dt > dt_max:
                dt = dt_max
            if kd > 0.0 and drag_fac / kd < dt:
                dt = drag_fac / kd
            last = t + dt >= t_end or t + dt == t
            if last:
                dt = t_end - t
            if dt <= 0.0:
                break
            x1, y1, z1, vx1, vy1, vz1 = propagate(x, y, z, vx, vy, vz, ux, uy, uz, gx * bfac, gy * bfac, gz * bfac,
                                                  kd, dt)
            substeps += 1.0
            # ---- earliest contact along the chord
            best_s = 2.0
            hit = 0
            h_kind = 0
            h_iu = -1
            h_iv = -1
            hn0 = 0.0
            hn1 = 0.0
            hn2 = 0.0
            h_c = -1
            h_cz = 0.0
            b_axis = -1
            b_side = -1
            sb, ba, bs = box_exit(x, y, z, x1, y1, z1, box[0], box[1])
            if sb <= 1.0:
                best_s = sb
                hit = 4
                b_axis = ba
                b_side = bs
            k1, f1 = tick_index(t + dt, t0, dtk, n_t)
            o1x, o1y, o1z = lerp3(origin, k1, f1)
            q1x = x1 - o1x
            q1y = y1 - o1y
            q1z = z1 - o1z
            u1 = R[0, 0] * q1x + R[1, 0] * q1y + R[2, 0] * q1z
            v1 = R[0, 1] * q1x + R[1, 1] * q1y + R[2, 1] * q1z
            w1 = R[0, 2] * q1x + R[1, 2] * q1y + R[2, 2] * q1z
            hh = p_hf_hit(pu, pv, pw, u1, v1, w1, base_m, bulk_m3, cell)
            if hh[7] >= 0:
                p_box(ri, i, hh[7], hh[8])
                p_box(ri, i, hh[9], hh[10])
            if hh[4] != 0 and hh[0] < best_s:
                best_s = hh[0]
                hit = 1
                h_kind = hh[4]
                h_iu = hh[5]
                h_iv = hh[6]
                gu = hh[1]
                gv = hh[2]
                gw = hh[3]
                hn0 = R[0, 0] * gu + R[0, 1] * gv + R[0, 2] * gw
                hn1 = R[1, 0] * gu + R[1, 1] * gv + R[1, 2] * gw
                hn2 = R[2, 0] * gu + R[2, 1] * gv + R[2, 2] * gw
            km, fm = tick_index(t + 0.5 * dt, t0, dtk, n_t)
            for c in range(n_cyl):
                if cyl_f[c, CF_RADIUS] <= 0.0:
                    continue
                bx, by, bz = lerp3(cyl_base[c], km, fm)
                ax, ay, az = lerp3(cyl_axis[c], km, fm)
                am = math.sqrt(ax * ax + ay * ay + az * az)
                if am <= 0.0:
                    continue
                ax /= am
                ay /= am
                az /= am
                sc, cnx, cny, cnz, cz, cfl = cyl_hit(x, y, z, x1, y1, z1, bx, by, bz, ax, ay, az, cyl_f[c, CF_RADIUS],
                                                     cyl_f[c, CF_LENGTH])
                if cfl != 0 and sc < best_s:
                    best_s = sc
                    hit = 2
                    h_c = c
                    h_cz = cz
                    hn0 = cnx
                    hn1 = cny
                    hn2 = cnz
            for s_i in range(n_sph):
                ss, snx, sny, snz, sfl = sph_hit(x, y, z, x1, y1, z1, sph_c[s_i, 0], sph_c[s_i, 1], sph_c[s_i, 2],
                                                 sph_r[s_i])
                if sfl != 0 and ss < best_s:
                    best_s = ss
                    hit = 3
                    h_c = s_i
                    hn0 = snx
                    hn1 = sny
                    hn2 = snz
            if hit == 0:
                srel1 = math.sqrt((vx1 - ux) ** 2 + (vy1 - uy) ** 2 + (vz1 - uz) ** 2)
                tn, ql, qa = p_cool(temp, m_chip, surface, dv, mass, dt, 0.5 * (srel + srel1), rho, mu, kf, pr, tf,
                                    liq, t_air, t_air, cp_T, cp_v, h_t, fpar)
                temp = tn
                if not p_log(ri, i, log, ql, qa):
                    return
                x = x1
                y = y1
                z = z1
                vx = vx1
                vy = vy1
                vz = vz1
                t = t_end if last else t + dt
                zero_hits = 0
                continue
            # ---- advance to the contact
            dth = best_s * dt
            hx = x + best_s * (x1 - x)
            hy = y + best_s * (y1 - y)
            hz = z + best_s * (z1 - z)
            if dth > 0.0:
                pp = propagate(x, y, z, vx, vy, vz, ux, uy, uz, gx * bfac, gy * bfac, gz * bfac, kd, dth)
                hvx = pp[3]
                hvy = pp[4]
                hvz = pp[5]
                tn, ql, qa = p_cool(temp, m_chip, surface, dv, mass, dth, srel, rho, mu, kf, pr, tf, liq, t_air, t_air,
                                    cp_T, cp_v, h_t, fpar)
                temp = tn
                if not p_log(ri, i, log, ql, qa):
                    return
                zero_hits = 0
            else:
                hvx = vx
                hvy = vy
                hvz = vz
                zero_hits += 1
            t += dth
            hits += 1.0
            bounces += 1
            x = hx
            y = hy
            z = hz
            vx = hvx
            vy = hvy
            vz = hvz
            time_out = t
            kh, fh = tick_index(t, t0, dtk, n_t)
            if bounces > max_bounces or zero_hits > max_hits:
                ohx, ohy, ohz = lerp3(origin, kh, fh)
                qx = hx - ohx
                qy = hy - ohy
                qz = hz - ohz
                cu = int(np.floor((R[0, 0] * qx + R[1, 0] * qy + R[2, 0] * qz) / cell))
                cv = int(np.floor((R[0, 1] * qx + R[1, 1] * qy + R[2, 1] * qz) / cell))
                ri[i, RI_EV_KIND] = EV_BOUNCE_LIMIT
                ri[i, RI_EV_OWNER] = slot
                rf[i, RF_EV_TIME] = t
                rf[i, RF_EV_X] = hx
                rf[i, RF_EV_Y] = hy
                rf[i, RF_EV_Z] = hz
                rf[i, RF_EV_TEMP] = temp
                if 0 <= cu < nu and 0 <= cv < nv:
                    term = TERM_SETTLE
                    ri[i, RI_TERM_IU] = cu
                    ri[i, RI_TERM_IV] = cv
                else:
                    term = TERM_PAN
                break
            if hit == 4:
                floor_hit = b_axis == ipar[IP_FLOOR_AXIS] and b_side == ipar[IP_FLOOR_SIDE]
                if floor_hit or ipar[IP_WALLS_WET] != 0 or b_side >= 2:
                    term = TERM_PAN
                    break
                bnx = 0.0
                bny = 0.0
                bnz = 0.0
                sign = 1.0 if b_side == 0 else -1.0
                if b_axis == 0:
                    bnx = sign
                elif b_axis == 1:
                    bny = sign
                else:
                    bnz = sign
                nvx, nvy, nvz = impulse(hvx, hvy, hvz, 0.0, 0.0, 0.0, bnx, bny, bnz, e_pair[SURF_ENCLOSURE],
                                        mu_pair[SURF_ENCLOSURE])
                vx = nvx
                vy = nvy
                vz = nvz
                x = hx + offset * bnx
                y = hy + offset * bny
                z = hz + offset * bnz
                continue
            if hit == 1:
                sx = (origin[kh + 1, 0] - origin[kh, 0]) / dtk
                sy = (origin[kh + 1, 1] - origin[kh, 1]) / dtk
                sz = (origin[kh + 1, 2] - origin[kh, 2]) / dtk
                if h_kind == 3:
                    p_box(ri, i, h_iu, h_iv)
                    top = base_m[h_iu, h_iv] + bulk_m3[h_iu, h_iv] / (cell * cell)
                    ohx, ohy, ohz = lerp3(origin, kh, fh)
                    qx = hx - ohx
                    qy = hy - ohy
                    qz = hz - ohz
                    lift = top - (R[0, 2] * qx + R[1, 2] * qy + R[2, 2] * qz)
                    hx += lift * hn0
                    hy += lift * hn1
                    hz += lift * hn2
                # ---- respond_hf (settle deferred)
                surf = SURF_ENCLOSURE
                is_wet = False
                if h_iu >= 0 and h_iv >= 0:
                    p_box(ri, i, h_iu, h_iv)
                    surf = SURF_PILE if bulk_m3[h_iu, h_iv] > 0.0 else base_kind[h_iu, h_iv]
                    if surf == SURF_NONE:
                        surf = SURF_ENCLOSURE
                    is_wet = wet[h_iu, h_iv] != 0
                e = e_pair[surf]
                mu_c = mu_pair[surf]
                v_in = -((hvx - sx) * hn0 + (hvy - sy) * hn1 + (hvz - sz) * hn2)
                if v_in < 0.0:
                    v_in = 0.0
                g_n = -(gx * hn0 + gy * hn1 + gz * hn2)
                v_reb = e * v_in
                stick = False
                if is_wet:
                    v_esc = p_escape(pf[i, PF_WIDTH], pf[i, PF_LENGTH], m_chip, fpar)
                    st_c = (1.0 + 1.0 / e) * fpar[FP_WET + 0] if e > 0.0 else _INF
                    v_visc = st_c * 9.0 * fpar[FP_LIQ + 1] / (4.0 * fpar[FP_RHO] * dv)
                    if v_reb < v_esc or v_in < v_visc:
                        stick = True
                gone = False
                if h_kind != 2:
                    if g_n > 0.0:
                        v_set = math.sqrt(2.0 * g_n * fpar[FP_SETTLE] * dv)
                        if v_reb < v_set or stick:
                            gone = True
                else:
                    if stick:
                        gone = True
                if gone:
                    # K1 keeps the hit point from before the lift as the particle's last position
                    term = TERM_SETTLE
                    ri[i, RI_TERM_IU] = h_iu
                    ri[i, RI_TERM_IV] = h_iv
                    break
                nvx, nvy, nvz = impulse(hvx, hvy, hvz, sx, sy, sz, hn0, hn1, hn2, e, mu_c)
                vx = nvx
                vy = nvy
                vz = nvz
                x = hx + offset * hn0
                y = hy + offset * hn1
                z = hz + offset * hn2
                continue
            if hit == 3:
                rn = sph_r[h_c]
                hook = pf[i, PF_LENGTH] >= fpar[FP_HOOK] * rn
                stuck = False
                if sph_a[h_c] != 0:
                    stuck = hook
                else:
                    v_esc = p_escape(pf[i, PF_WIDTH], pf[i, PF_LENGTH], m_chip, fpar)
                    v_in = -(hvx * hn0 + hvy * hn1 + hvz * hn2)
                    stuck = hook or e_pair[SURF_NOZZLE] * max(v_in, 0.0) < v_esc
                if stuck:
                    st = ON_NOZZLE
                    owner = h_c
                    vx = 0.0
                    vy = 0.0
                    vz = 0.0
                    ri[i, RI_EV_KIND] = EV_NOZZLE
                    ri[i, RI_EV_OWNER] = h_c
                    rf[i, RF_EV_TIME] = t
                    rf[i, RF_EV_X] = hx
                    rf[i, RF_EV_Y] = hy
                    rf[i, RF_EV_Z] = hz
                    rf[i, RF_EV_TEMP] = temp
                    break
                nvx, nvy, nvz = impulse(hvx, hvy, hvz, 0.0, 0.0, 0.0, hn0, hn1, hn2, e_pair[SURF_NOZZLE],
                                        mu_pair[SURF_NOZZLE])
                vx = nvx
                vy = nvy
                vz = nvz
                gap = rn + offset + 0.5 * pf[i, PF_THICK]
                x = sph_c[h_c, 0] + gap * hn0
                y = sph_c[h_c, 1] + gap * hn1
                z = sph_c[h_c, 2] + gap * hn2
                continue
            # ---- hit == 2: rotating cylinder
            c = h_c
            bx, by, bz = lerp3(cyl_base[c], kh, fh)
            ax, ay, az = lerp3(cyl_axis[c], kh, fh)
            am = math.sqrt(ax * ax + ay * ay + az * az)
            ax /= am
            ay /= am
            az /= am
            omega = lerp1(cyl_omega[c], kh, fh)
            tvx = (cyl_base[c, kh + 1, 0] - cyl_base[c, kh, 0]) / dtk
            tvy = (cyl_base[c, kh + 1, 1] - cyl_base[c, kh, 1]) / dtk
            tvz = (cyl_base[c, kh + 1, 2] - cyl_base[c, kh, 2]) / dtk
            radius = cyl_f[c, CF_RADIUS]
            rx = hn0 * radius
            ry = hn1 * radius
            rz = hn2 * radius
            svx = omega * (ay * rz - az * ry) + tvx
            svy = omega * (az * rx - ax * rz) + tvy
            svz = omega * (ax * ry - ay * rx) + tvz
            own_first = bounces <= 1 and cyl_i[c, CI_OWNER] == slot
            if cyl_i[c, CI_WELD] != 0 and temp > fpar[FP_WELD] and not own_first:
                st = ON_TOOL
                owner = cyl_i[c, CI_OWNER]
                vx = svx
                vy = svy
                vz = svz
                ri[i, RI_EV_KIND] = EV_WELD
                ri[i, RI_EV_OWNER] = cyl_i[c, CI_OWNER]
                rf[i, RF_EV_TIME] = t
                rf[i, RF_EV_X] = hx
                rf[i, RF_EV_Y] = hy
                rf[i, RF_EV_Z] = hz
                rf[i, RF_EV_TEMP] = temp
                break
            fi = min(max(form, 0), N_FORM - 1)
            if (cyl_i[c, CI_WRAP] != 0 and form_wrap[fi] != 0 and omega != 0.0 and not own_first
                    and pf[i, PF_LENGTH] >= fpar[FP_WRAP_TURNS] * 2.0 * math.pi * radius):
                st = WRAPPED
                owner = cyl_i[c, CI_OWNER]
                vx = svx
                vy = svy
                vz = svz
                ri[i, RI_EV_KIND] = EV_WRAP
                ri[i, RI_EV_OWNER] = cyl_i[c, CI_OWNER]
                rf[i, RF_EV_TIME] = t
                rf[i, RF_EV_X] = hx
                rf[i, RF_EV_Y] = hy
                rf[i, RF_EV_Z] = hz
                rf[i, RF_EV_TEMP] = temp
                break
            surf = cyl_i[c, CI_SURF]
            e = e_pair[surf]
            mu_c = mu_pair[surf]
            if cyl_i[c, CI_KIND] == CYL_PADDLE and omega != 0.0:
                rvx = hvx - svx
                rvy = hvy - svy
                rvz = hvz - svz
                sgn = 1.0 if omega > 0.0 else -1.0
                tx = sgn * (ay * hn2 - az * hn1)
                ty = sgn * (az * hn0 - ax * hn2)
                tz = sgn * (ax * hn1 - ay * hn0)
                vrn = rvx * hn0 + rvy * hn1 + rvz * hn2
                vrt = rvx * tx + rvy * ty + rvz * tz
                vra = rvx * ax + rvy * ay + rvz * az
                dn = 0.0
                dtn = 0.0
                if vrn < 0.0:
                    dn = -(1.0 + e) * vrn
                    vrn = -e * vrn
                if vrt < 0.0:
                    dtn = -(1.0 + fpar[FP_PADDLE]) * vrt
                    vrt = -fpar[FP_PADDLE] * vrt
                red = mu_c * (dn + dtn)
                if abs(vra) <= red:
                    vra = 0.0
                elif vra > 0.0:
                    vra -= red
                else:
                    vra += red
                nvx = svx + vrn * hn0 + vrt * tx + vra * ax
                nvy = svy + vrn * hn1 + vrt * ty + vra * ay
                nvz = svz + vrn * hn2 + vrt * tz + vra * az
            else:
                nvx, nvy, nvz = impulse(hvx, hvy, hvz, svx, svy, svz, hn0, hn1, hn2, e, mu_c)
            vx = nvx
            vy = nvy
            vz = nvz
            gap = radius + offset + 0.5 * pf[i, PF_THICK]
            x = bx + h_cz * ax + gap * hn0
            y = by + h_cz * ay + gap * hn1
            z = bz + h_cz * az + gap * hn2
        if st == FLYING and term == TERM_NONE:
            time_out = t_end
    rf[i, RF_X] = x
    rf[i, RF_Y] = y
    rf[i, RF_Z] = z
    rf[i, RF_VX] = vx
    rf[i, RF_VY] = vy
    rf[i, RF_VZ] = vz
    rf[i, RF_TIME] = time_out
    rf[i, RF_TEMP] = temp
    rf[i, RF_SUBSTEPS] = substeps
    rf[i, RF_HITS] = hits
    ri[i, RI_STATUS] = st
    ri[i, RI_OWNER] = owner
    ri[i, RI_BOUNCES] = bounces
    ri[i, RI_TERM] = term



# ============================================================================ the CUDA kernel (source)
def k_flight(n, pf, pi, base_m, base_kind, bulk_m3, wet, film_uv, origin, R, box, cyl_base, cyl_axis, cyl_omega, cyl_f,
             cyl_i, sph_c, sph_r, sph_a, jet_o, jet_d, jet_f, jet_gas, cp_T, cp_v, h_t, form_wrap, e_pair, mu_pair,
             fpar, ipar, rf, ri, glog):
    """One thread per particle: :func:`flight` against the pile at the start of the call."""
    i = cuda.grid(1)
    if i >= n:
        return
    flight(i, pf, pi, base_m, base_kind, bulk_m3, wet, film_uv, origin, R, box, cyl_base, cyl_axis, cyl_omega, cyl_f, cyl_i,
           sph_c, sph_r, sph_a, jet_o, jet_d, jet_f, jet_gas, cp_T, cp_v, h_t, form_wrap, e_pair, mu_pair, fpar, ipar,
           rf, ri, glog[i])


_F = "float64"
_I = "int64"
SIGNATURES = {"k_flight": (f"void(int64, {_F}[:, ::1], {_I}[:, ::1], {_F}[:, ::1], {_I}[:, ::1], {_F}[:, ::1], "
                           f"{_I}[:, ::1], {_F}[:, :, ::1], {_F}[:, ::1], {_F}[:, ::1], {_F}[:, ::1], {_F}[:, :, ::1], "
                           f"{_F}[:, :, ::1], {_F}[:, ::1], {_F}[:, ::1], {_I}[:, ::1], {_F}[:, ::1], {_F}[::1], "
                           f"{_I}[::1], {_F}[:, ::1], {_F}[:, ::1], {_F}[:, ::1], {_I}[::1], {_F}[::1], {_F}[::1], "
                           f"{_F}[::1], {_I}[::1], {_F}[::1], {_F}[::1], {_F}[::1], {_I}[::1], {_F}[:, ::1], "
                           f"{_I}[:, ::1], {_F}[:, :, ::1])")}
SOURCES = {"k_flight": k_flight}


# ============================================================================ the ordered host commit (CPU)
@twin
def commit(n, P, H, C, out, ev, pf, pi, origin, R, box, cyl_base, cyl_axis, cyl_omega, cyl_f, cyl_i, sph_c, sph_r, sph_a,
           jet_o, jet_d, jet_f, jet_gas, cp_T, cp_v, h_t, form_wrap, e_pair, mu_pair, fpar, ipar, rf, ri, glog, hlog,
           dirty, stats):
    """K1's shared-state effects in particle order (see the module doc). ``stats``: [0] particles recomputed
    on the host (pile conflict or log overflow), [1] of them for a pile conflict."""
    nd = 0
    log = hlog
    for i in range(n):
        if ri[i, RI_DONE] == 0:
            continue
        redo = ri[i, RI_NLOG] < 0
        conflict = False
        if not redo:
            for d in range(nd):
                du = dirty[d, 0]
                dv = dirty[d, 1]
                if ri[i, RI_BOX_U0] <= du <= ri[i, RI_BOX_U1] and ri[i, RI_BOX_V0] <= dv <= ri[i, RI_BOX_V1]:
                    conflict = True
                    break
        if redo or conflict:
            while True:
                flight(i, pf, pi, H.base_m, H.base_kind, H.bulk_m3, H.wet, H.film_uv_m_s, origin, R, box, cyl_base,
                       cyl_axis, cyl_omega, cyl_f, cyl_i, sph_c, sph_r, sph_a, jet_o, jet_d, jet_f, jet_gas, cp_T, cp_v,
                       h_t, form_wrap, e_pair, mu_pair, fpar, ipar, rf, ri, hlog)
                if ri[i, RI_NLOG] >= 0:
                    break
                hlog = np.zeros((2 * hlog.shape[0], 2))
            log = hlog
            stats[0] += 1
            if conflict:
                stats[1] += 1
        else:
            log = glog[i]
        P.pos_M_m[i, 0] = rf[i, RF_X]
        P.pos_M_m[i, 1] = rf[i, RF_Y]
        P.pos_M_m[i, 2] = rf[i, RF_Z]
        P.vel_M_m_s[i, 0] = rf[i, RF_VX]
        P.vel_M_m_s[i, 1] = rf[i, RF_VY]
        P.vel_M_m_s[i, 2] = rf[i, RF_VZ]
        P.time_s[i] = rf[i, RF_TIME]
        P.temp_K[i] = rf[i, RF_TEMP]
        P.status[i] = ri[i, RI_STATUS]
        P.owner[i] = ri[i, RI_OWNER]
        P.bounces[i] = ri[i, RI_BOUNCES]
        for k in range(ri[i, RI_NLOG]):
            out[OUT_HEAT_COOLANT_J] += log[k, 0]
            out[OUT_HEAT_AIR_J] += log[k, 1]
        out[OUT_SUBSTEPS] += rf[i, RF_SUBSTEPS]
        out[OUT_HITS] += rf[i, RF_HITS]
        if ri[i, RI_EV_KIND] != 0:
            write_event(ev, ri[i, RI_EV_KIND], i, ri[i, RI_EV_OWNER], rf[i, RF_EV_TIME], P.mass_kg[i], rf[i, RF_EV_X],
                        rf[i, RF_EV_Y], rf[i, RF_EV_Z], rf[i, RF_EV_TEMP], P.form[i])
        term = ri[i, RI_TERM]
        if term == TERM_SETTLE:
            iu = ri[i, RI_TERM_IU]
            iv = ri[i, RI_TERM_IV]
            raises = not (iu < 0 or iv < 0 or H.base_m[iu, iv] == -_INF)
            settle(P, H, C, i, iu, iv, out)
            if raises:
                dirty[nd, 0] = iu
                dirty[nd, 1] = iv
                nd += 1
        elif term == TERM_PAN:
            to_pan(P, C, i, out)


@twin
def flight_all(n, pf, pi, base_m, base_kind, bulk_m3, wet, film_uv, origin, R, box, cyl_base, cyl_axis, cyl_omega, cyl_f,
               cyl_i, sph_c, sph_r, sph_a, jet_o, jet_d, jet_f, jet_gas, cp_T, cp_v, h_t, form_wrap, e_pair, mu_pair,
               fpar, ipar, rf, ri, glog):
    """What k_flight does, on the CPU (the reference of the GPU phase in the tests)."""
    for i in range(n):
        flight(i, pf, pi, base_m, base_kind, bulk_m3, wet, film_uv, origin, R, box, cyl_base, cyl_axis, cyl_omega, cyl_f,
               cyl_i, sph_c, sph_r, sph_a, jet_o, jet_d, jet_f, jet_gas, cp_T, cp_v, h_t, form_wrap, e_pair, mu_pair,
               fpar, ipar, rf, ri, glog[i])


# ============================================================================ packing (host)
def _rows(a: np.ndarray, shape_tail: tuple, dtype) -> np.ndarray:
    """At least one row (zero-size device arrays are avoided; the counts in ipar say what is real)."""
    a = np.asarray(a, dtype=dtype)
    if a.shape[0] == 0:
        return np.zeros((1,) + shape_tail, dtype)
    return np.ascontiguousarray(a)


def pack(P: CA.Particles, H: CA.Pile, W: CA.Window, S: CA.Scene, C: CA.ChipModel) -> dict:
    """The kernel arguments of one call (packed copies; P, H, W, S, C are not changed)."""
    n = int(P.n[0])
    pf = np.zeros((max(n, 1), NPF))
    pf[:n, PF_X:PF_Z + 1] = P.pos_M_m[:n]
    pf[:n, PF_VX:PF_VZ + 1] = P.vel_M_m_s[:n]
    for col, arr in ((PF_TIME, P.time_s), (PF_MASS, P.mass_kg), (PF_TEMP, P.temp_K), (PF_THICK, P.thickness_m),
                     (PF_WIDTH, P.width_m), (PF_LENGTH, P.length_m), (PF_DV, P.dv_m), (PF_CHIP_MASS, P.chip_mass_kg),
                     (PF_AREA, P.area_m2), (PF_SURFACE, P.surface_m2), (PF_HLA, P.hl_a), (PF_HLB, P.hl_b),
                     (PF_HLC, P.hl_c), (PF_HLD, P.hl_d)):
        pf[:n, col] = arr[:n]
    pi = np.zeros((max(n, 1), NPI), np.int64)
    for col, arr in ((PI_FORM, P.form), (PI_STATUS, P.status), (PI_OWNER, P.owner), (PI_SLOT, P.slot),
                     (PI_BOUNCES, P.bounces)):
        pi[:n, col] = arr[:n]
    nt1 = W.grid_origin_M.shape[0]
    n_cyl, n_sph, n_jet = int(S.cyl_radius.shape[0]), int(S.sph_radius.shape[0]), int(S.jet_speed.shape[0])
    fpar = np.zeros(NFP)
    fpar[FP_T0], fpar[FP_TEND], fpar[FP_DTK] = W.t0_s[0], W.t_end_s[0], W.dt_tick_s[0]
    fpar[FP_GX:FP_GZ + 1] = W.gravity_M
    fpar[FP_BOX_LO:FP_BOX_LO + 3] = S.box_lo_M
    fpar[FP_BOX_HI:FP_BOX_HI + 3] = S.box_hi_M
    fpar[FP_AIR:FP_AIR + 5] = S.air[:5]
    fpar[FP_LIQ:FP_LIQ + 7] = S.liquid[:7]
    fpar[FP_JC:FP_JC + 2] = S.jet_const[:2]
    fpar[FP_RHO], fpar[FP_EMIS], fpar[FP_WELD], fpar[FP_HL0] = C.rho_kg_m3[0], C.emissivity[0], C.weld_temp_K[0], C.hl[0]
    fpar[FP_RM:FP_RM + 4] = C.rm[:4]
    fpar[FP_PADDLE] = C.paddle_e[0]
    fpar[FP_WET:FP_WET + 3] = C.wet[:3]
    fpar[FP_SETTLE], fpar[FP_WRAP_TURNS], fpar[FP_HOOK] = C.settle_bounce[0], C.wrap_turns[0], C.hook_radii[0]
    fpar[FP_NUM:FP_NUM + 5] = C.numerics[:5]
    fpar[FP_CELL] = H.cell_m[0]
    ipar = np.zeros(NIP, np.int64)
    ipar[IP_FLOOR_AXIS], ipar[IP_FLOOR_SIDE] = S.floor[0], S.floor[1]
    ipar[IP_WALLS_WET] = S.walls_wet[0]
    ipar[IP_NCYL], ipar[IP_NSPH], ipar[IP_NJET] = n_cyl, n_sph, n_jet
    cyl_f = _rows(np.stack([S.cyl_radius, S.cyl_length], axis=1).reshape(n_cyl, 2), (2,), np.float64)
    cyl_i = _rows(np.stack([S.cyl_kind, S.cyl_surf, S.cyl_owner, S.cyl_wrap, S.cyl_weld], axis=1).reshape(n_cyl, 5),
                  (5,), np.int64)
    jet_f = np.zeros((n_jet, 7))
    if n_jet:
        jet_f[:, JF_SPEED], jet_f[:, JF_R0], jet_f[:, JF_TAN] = S.jet_speed, S.jet_r0, S.jet_tan
        jet_f[:, JF_RHO:JF_PR + 1] = S.jet_fluid[:, :4]
    return {
        "n": n, "pf": pf, "pi": pi,
        "base_m": np.ascontiguousarray(H.base_m), "base_kind": np.ascontiguousarray(H.base_kind, dtype=np.int64),
        "bulk_m3": np.ascontiguousarray(H.bulk_m3), "wet": np.ascontiguousarray(H.wet, dtype=np.int64),
        "film_uv": np.ascontiguousarray(H.film_uv_m_s),
        "origin": np.ascontiguousarray(W.grid_origin_M), "R": np.ascontiguousarray(W.R_M_G),
        "box": np.ascontiguousarray(np.stack([S.box_lo_M, S.box_hi_M]), dtype=np.float64),
        "cyl_base": _rows(S.cyl_base_M.reshape(n_cyl, nt1, 3), (nt1, 3), np.float64),
        "cyl_axis": _rows(S.cyl_axis_M.reshape(n_cyl, nt1, 3), (nt1, 3), np.float64),
        "cyl_omega": _rows(S.cyl_omega.reshape(n_cyl, nt1), (nt1,), np.float64), "cyl_f": cyl_f, "cyl_i": cyl_i,
        "sph_c": _rows(S.sph_center_M.reshape(n_sph, 3), (3,), np.float64),
        "sph_r": _rows(S.sph_radius, (), np.float64), "sph_a": _rows(S.sph_active, (), np.int64),
        "jet_o": _rows(S.jet_origin_M.reshape(n_jet, 3), (3,), np.float64),
        "jet_d": _rows(S.jet_dir_M.reshape(n_jet, 3), (3,), np.float64), "jet_f": _rows(jet_f, (7,), np.float64),
        "jet_gas": _rows(S.jet_gas, (), np.int64),
        "cp_T": np.ascontiguousarray(C.cp_T_K), "cp_v": np.ascontiguousarray(C.cp_J_kgK),
        "h_t": np.ascontiguousarray(C.h_J_kg), "form_wrap": np.ascontiguousarray(C.form_wrap, dtype=np.int64),
        "e_pair": np.ascontiguousarray(C.e_pair), "mu_pair": np.ascontiguousarray(C.mu_pair),
        "fpar": fpar, "ipar": ipar,
    }


SCENE_KEYS = ("origin", "R", "box", "cyl_base", "cyl_axis", "cyl_omega", "cyl_f", "cyl_i", "sph_c", "sph_r", "sph_a", "jet_o",
              "jet_d", "jet_f", "jet_gas", "cp_T", "cp_v", "h_t", "form_wrap", "e_pair", "mu_pair", "fpar", "ipar")
PILE_KEYS = ("base_m", "base_kind", "bulk_m3", "wet", "film_uv")


def results(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = max(int(n), 1)
    return np.zeros((rows, NRF)), np.zeros((rows, NRI), np.int64), np.zeros((rows, CAP_LOG, 2))


# ============================================================================ compiled entry points
_CPU: dict[str, object] = {}
_GPU: dict[str, object] = {}


def _signature(*args):
    import numba
    return types.void(*[numba.typeof(a) for a in args])


def _sample():
    """Small arguments of the exact types the entry points are compiled for."""
    P = CA.particles(1)
    H = CA.pile(1, 1)
    W = CA.window(1)
    S = CA.scene(1, 1, 1, 1)
    C = CA.chip_model()
    out, ev = CA.step_out(1)
    packed = pack(P, H, W, S, C)
    rf, ri, glog = results(1)
    return P, H, C, out, ev, packed, rf, ri, glog


def cpu_entries() -> dict:
    """The CPU twins: ``flight_all`` (the GPU phase on the CPU) and ``commit``."""
    with D.LAUNCH_LOCK:
        if _CPU:
            return _CPU
        b = builder("cpu")
        P, H, C, out, ev, k, rf, ri, glog = _sample()
        scene = [k[key] for key in SCENE_KEYS]
        pile_args = [k[key] for key in PILE_KEYS]
        sig_all = _signature(np.int64(1), k["pf"], k["pi"], *pile_args, *scene, rf, ri, glog)
        sig_commit = _signature(np.int64(1), P, H, C, out, ev, k["pf"], k["pi"], *scene, rf, ri, glog,
                                np.zeros((2, 2)), np.zeros((1, 2), np.int64), np.zeros(2, np.int64))
        _CPU["flight_all"] = b.kernel(flight_all, sig_all)
        _CPU["commit"] = b.kernel(commit, sig_commit)
        return _CPU


def gpu_kernel():
    with D.LAUNCH_LOCK:
        if "k_flight" not in _GPU:
            cuda = D.cuda_module()
            b = builder("cuda", cuda, D.simulator_mode())
            with D.exact_compile():
                _GPU["k_flight"] = b.kernel(k_flight, SIGNATURES["k_flight"])
        return _GPU["k_flight"]


def _commit(P, H, C, out, ev, k, rf, ri, glog) -> np.ndarray:
    n = k["n"]
    stats = np.zeros(2, np.int64)
    dirty = np.zeros((max(n, 1), 2), np.int64)
    hlog = np.zeros((HOST_LOG, 2))
    cpu_entries()["commit"](np.int64(n), P, H, C, out, ev, k["pf"], k["pi"], *[k[key] for key in SCENE_KEYS], rf, ri,
                            glog, hlog, dirty, stats)
    return stats


def advance_particles_cpu_twin(P, H, W, S, C, out, ev) -> np.ndarray:
    """K1 through the GPU path's two phases, both on the CPU (flight_all + commit). Returns the commit
    stats. Used by the tests and as the reference of the GPU phase."""
    k = pack(P, H, W, S, C)
    rf, ri, glog = results(k["n"])
    cpu_entries()["flight_all"](np.int64(k["n"]), k["pf"], k["pi"], *[k[key] for key in PILE_KEYS],
                                *[k[key] for key in SCENE_KEYS], rf, ri, glog)
    return _commit(P, H, C, out, ev, k, rf, ri, glog)


def advance_particles(P: CA.Particles, H: CA.Pile, W: CA.Window, S: CA.Scene, C: CA.ChipModel, out: np.ndarray,
                      ev) -> np.ndarray:
    """``cnctwin.chips.kernels.advance_particles`` with the flight on the GPU and the ordered commit on the
    host. Returns the commit stats [recomputed on the host, of them for a pile conflict]. P, H, out and ev
    change only in the commit, after the GPU phase succeeded; a CUDA error raises GpuError first."""
    k = pack(P, H, W, S, C)
    n = k["n"]
    if n == 0:
        return np.zeros(2, np.int64)
    try:
        kernel = gpu_kernel()
        cuda = D.cuda_module()
        with D.LAUNCH_LOCK:
            rf, ri, glog = results(n)
            d_rf, d_ri, d_log = cuda.to_device(rf), cuda.to_device(ri), cuda.device_array(glog.shape, np.float64)
            dev = {key: cuda.to_device(k[key]) for key in ("pf", "pi") + PILE_KEYS + SCENE_KEYS}
            blocks, tpb = D.grid(n)
            kernel[blocks, tpb](np.int64(n), dev["pf"], dev["pi"], *[dev[key] for key in PILE_KEYS],
                                *[dev[key] for key in SCENE_KEYS], d_rf, d_ri, d_log)
            cuda.synchronize()
            rf, ri, glog = d_rf.copy_to_host(), d_ri.copy_to_host(), d_log.copy_to_host()
    except D.GpuError:
        raise
    except Exception as exc:
        raise D.GpuError(f"chips advance_particles on the GPU failed ({type(exc).__name__}: {exc})") from exc
    return _commit(P, H, C, out, ev, k, rf, ri, glog)
