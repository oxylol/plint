"""Compiled kernels of the dynamics domain (docs/models/dynamics.md "Kernels"; architecture 5.4, 5.5).

Per L0 step in contact, for each system (the active tool of a path against its workpiece):

1. :func:`external_force` - unbalance (static + couple, ``U (W^2 - i alpha) e^(i theta)``) and base-excitation
   modal forces at the coming node;
2. :func:`modal_predict` - state at the coming node with the cutting force extrapolated from the last two nodes;
3. :func:`modal_deflection` - station deflections and rates (translations and rotations, T0) and the
   workpiece deflection and rate at the contact (S) at the predicted state, with the residual flexibilities
   (quasi-static, one-step lag);
4. the cutting kernel evaluates the forces with that deflection (cnctwin.cutting: StepOut.station_T0,
   StepOut.force_S_N);
5. :func:`modal_correct` - modal force projection of the evaluated wrenches (moments through the rotational
   shape components), relaxed correction, exact first-order-hold energy integrals (structural damping,
   cutting / unbalance / base work), vibration sample.

Out of contact, per L1 tick: :func:`modal_free` (exact propagation, harmonic unbalance + first-order-hold base
excitation, closed-form energy integrals). Per tick end: :func:`record_tick`. Helpers:
:func:`refresh_work_shapes`, :func:`relative_axis_acceleration`, :func:`process_damping_element`,
:func:`deflection_at_point`, :func:`relative_deflection`, :func:`ring_write`, :func:`free_interval`.

Rules (architecture 5.5): explicit signatures, cache, nogil, error_model numpy, no fastmath, no parallel.
"""

from __future__ import annotations

import cmath
import math

import numpy as np
from numba import njit, types

from ..core.abi import ACC_DAMPING_J, ACC_PROCESS_DAMPING_J, ACC_UNBALANCE_WORK_J
from .abi import (ACCUM_T, ACT_HI, ACT_LO, ACT_LOAD_HI, ACT_LOAD_LO, ACT_PREV_HI, ACT_PREV_LO, AXES_T, BASE_LINEAR, DA_DAMPING_J, DA_ENERGY_MAX_J, DA_FORCE_MAX_N, DA_N_FREE, DA_N_L0,
                  DA_UMAX_X, DA_UMIN_X, DA_W_BASE_J, DA_W_CUT_J, DA_W_CUT_QUAD_J, DA_W_UNB_J, DEFL_T, FL_CONTACT,
                  FL_HISTORY, FL_STEP_ERROR, FL_STEP_SEEN, GRID_T, KIND_WORK, MI_HAS_TOOL_RES, MI_HAS_WORK_RES,
                  MI_L0_TICKS, MI_REF_STATION, MI_RELAX, MI_RES_RELAX, MS_RES_RELAX, MS_TICK_S, N_TU, RING_T, SC_OMEGA, SYSTEM_T)

OPTS = dict(cache=True, nogil=True, error_model="numpy")
# helpers called inside the kernels are inlined at the Numba IR level (no call boundary, no per-array argument
# overhead); they are still compiled for their explicit signature as kernels of their own
INLINE = dict(OPTS, inline="always")
# the L0 kernels (modal_begin, modal_correct) allocate nothing and are compiled without reference counting (as the
# hot loop's and removal's hot kernels): otherwise every member array of the system tuple taken in the kernel costs
# an atomic incref / decref, about 1.2 us per L0 step for a DynSystem (docs/performance.md "Dynamics")
RAW = dict(OPTS, _nrt=False)
f8 = types.float64
i8 = types.int64
c16 = types.complex128
F1 = types.float64[::1]
F2 = types.float64[:, ::1]
F3 = types.float64[:, :, ::1]
I1B = types.int8[::1]
I4 = types.int32[::1]
I8V = types.int64[::1]

# Numerical constants of the closed forms (floating-point accuracy, not model choices; model doc section 2):
SERIES_RADIUS = 0.2     # |z| below which phi1 / psi1 use their Taylor series (14 terms reach 1e-17 there)
SERIES_TERMS = 14
TAYLOR_OMEGA_H = 0.1    # omega h below which a free interval is integrated by its Taylor series (the closed form
                        #   then cancels large particular solutions)
TAYLOR_SPIN_H = 4.0     # largest |W| h of one Taylor sub-interval
TAYLOR_TERMS = 44       # series terms of a sub-interval (4^44 / 44! < 1e-27)


# =============================================================================== closed-form helpers
@njit(c16(c16), **OPTS)
def phi1(z):
    """(e^z - 1) / z, accurate for every z."""
    if abs(z) < SERIES_RADIUS:
        total = 0j
        term = 1.0 + 0j
        fact = 1.0
        for n in range(SERIES_TERMS):
            fact *= n + 1
            total += term / fact
            term *= z
        return total
    x = z.real
    y = z.imag
    sh = math.sin(0.5 * y)
    em1 = complex(math.expm1(x) * math.cos(y) - 2.0 * sh * sh, math.exp(x) * math.sin(y))
    return em1 / z


@njit(c16(c16), **OPTS)
def psi1(z):
    """((z - 1) e^z + 1) / z^2 = sum_n z^n (n + 1) / (n + 2)!."""
    if abs(z) < SERIES_RADIUS:
        total = 0j
        term = 1.0 + 0j
        fact = 1.0
        for n in range(SERIES_TERMS):
            fact *= n + 2
            total += term * (n + 1) / fact
            term *= z
        return total
    return ((z - 1.0) * cmath.exp(z) + 1.0) / (z * z)


@njit(c16(c16, f8), **OPTS)
def j0(kappa, h):
    """int_0^h e^(kappa t) dt."""
    return h * phi1(kappa * h)


@njit(c16(c16, f8), **OPTS)
def j1(kappa, h):
    """int_0^h t e^(kappa t) dt."""
    return h * h * psi1(kappa * h)


@njit(types.void(f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, F1), **OPTS)
def _free_closed(w, zeta, h, q0, v0, Pr, Pi, W, g0, g1, out):
    """Closed-form free interval (discretize.free_interval): out = q1, v1, W_harmonic, W_linear, D."""
    P = complex(Pr, Pi)
    sig = zeta * w
    wd = w * math.sqrt(1.0 - zeta * zeta)
    lam = complex(-sig, wd)
    H = 1.0 / complex(w * w - W * W, 2.0 * sig * W)
    Q = P * H
    iW = complex(0.0, W)
    VP = iW * Q
    w2 = w * w
    qL0 = g0 / w2 - 2.0 * zeta * g1 / (w2 * w)
    vL = g1 / w2
    a = q0 - Q.real - qL0
    b = v0 - VP.real - vL
    Cq = complex(a, -(b + sig * a) / wd)
    Cv = lam * Cq
    El = cmath.exp(lam * h)
    EW = cmath.exp(iW * h)
    out[0] = (Cq * El).real + (Q * EW).real + (g0 + g1 * h) / w2 - 2.0 * zeta * g1 / (w2 * w)
    out[1] = (Cv * El).real + (VP * EW).real + vL
    lamc = lam.conjugate()
    Cvc = Cv.conjugate()
    VPc = VP.conjugate()
    j_iW = j0(iW, h)
    j_2iW = j0(2.0 * iW, h)
    j_lam = j0(lam, h)
    out[2] = 0.5 * (P * Cv * j0(iW + lam, h)).real + 0.5 * (P * Cvc * j0(iW + lamc, h)).real \
        + 0.5 * (P * VP * j_2iW).real + 0.5 * (P * VPc).real * h + vL * (P * j_iW).real
    out[3] = (Cv * (g0 * j_lam + g1 * j1(lam, h))).real + (VP * (g0 * j_iW + g1 * j1(iW, h))).real \
        + vL * (g0 * h + 0.5 * g1 * h * h)
    acc = 0.5 * (Cv * Cv * j0(2.0 * lam, h)).real + 0.5 * (Cv.real * Cv.real + Cv.imag * Cv.imag) \
        * j0(complex(-2.0 * sig, 0.0), h).real \
        + (Cv * VP * j0(lam + iW, h)).real + (Cv * VPc * j0(lam - iW, h)).real + 2.0 * vL * (Cv * j_lam).real \
        + 0.5 * (VP * VP * j_2iW).real + 0.5 * (VP.real * VP.real + VP.imag * VP.imag) * h \
        + 2.0 * vL * (VP * j_iW).real + vL * vL * h
    out[4] = 2.0 * sig * acc


@njit(types.void(f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, F1, F1, F1, F1, F1), **OPTS)
def _free_taylor_piece(w, zeta, h, q0, v0, Pr, Pi, W, g0, g1, out, C, D, FH, FL):
    """One Taylor sub-interval (omega h < TAYLOR_OMEGA_H, |W| h <= TAYLOR_SPIN_H), scaled coefficients."""
    n_t = C.shape[0]
    sig = zeta * w
    wh = w * h
    # forcing coefficients scaled by h^n: harmonic Re(P (i W h)^n / n!), linear g0, g1 h
    zr = 1.0
    zi = 0.0
    fact = 1.0
    for n in range(n_t):
        if n > 0:
            fact *= n
            tr = zr * 0.0 - zi * (W * h)
            zi = zr * (W * h) + zi * 0.0
            zr = tr
        FH[n] = (Pr * zr - Pi * zi) / fact
        FL[n] = 0.0
    FL[0] = g0
    if n_t > 1:
        FL[1] = g1 * h
    C[0] = q0
    C[1] = v0 * h
    for n in range(n_t - 2):
        C[n + 2] = ((FH[n] + FL[n]) * h * h - 2.0 * sig * h * (n + 1) * C[n + 1] - wh * wh * C[n]) / ((n + 1) * (n + 2))
    q1 = 0.0
    for n in range(n_t):
        q1 += C[n]
    for n in range(n_t - 1):
        D[n] = (n + 1) * C[n + 1] / h
    D[n_t - 1] = 0.0
    v1 = 0.0
    for n in range(n_t):
        v1 += D[n]
    wh_ = 0.0
    dd = 0.0
    wl = 0.0
    for m in range(n_t):
        wl += D[m] * (FL[0] / (m + 1) + FL[1] / (m + 2))
        for n in range(n_t):
            wh_ += FH[m] * D[n] / (m + n + 1)
            dd += D[m] * D[n] / (m + n + 1)
    out[0] = q1
    out[1] = v1
    out[2] = h * wh_
    out[3] = h * wl
    out[4] = 2.0 * sig * h * dd


@njit(types.void(f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, F1), **OPTS)
def free_interval(w, zeta, h, q0, v0, Pr, Pi, W, g0, g1, out):
    """Exact state after h and the work / dissipation integrals of one mode for
    f(t) = Re((Pr + i Pi) e^(i W t)) + g0 + g1 t: out = q1, v1, W_harmonic, W_linear, D."""
    if w * h >= TAYLOR_OMEGA_H:
        _free_closed(w, zeta, h, q0, v0, Pr, Pi, W, g0, g1, out)
        return
    pieces = int(math.ceil(abs(W) * h / TAYLOR_SPIN_H))
    if pieces < 1:
        pieces = 1
    hs = h / pieces
    C = np.zeros(TAYLOR_TERMS)
    D = np.zeros(TAYLOR_TERMS)
    FH = np.zeros(TAYLOR_TERMS)
    FL = np.zeros(TAYLOR_TERMS)
    piece = np.zeros(5)
    q = q0
    v = v0
    w_h = 0.0
    w_l = 0.0
    dis = 0.0
    for j in range(pieces):
        t0 = j * hs
        c = math.cos(W * t0)
        s = math.sin(W * t0)
        pr = Pr * c - Pi * s
        pi = Pr * s + Pi * c
        _free_taylor_piece(w, zeta, hs, q, v, pr, pi, W, g0 + g1 * t0, g1, piece, C, D, FH, FL)
        q = piece[0]
        v = piece[1]
        w_h += piece[2]
        w_l += piece[3]
        dis += piece[4]
    out[0] = q
    out[1] = v
    out[2] = w_h
    out[3] = w_l
    out[4] = dis


# Style note: kernels bind the individual arrays they use (``x = sys.state.x``), never a whole sub-tuple
# (``model = sys.model``): with reference counting (the kernels other than the RAW L0 pair), Numba reference-counts
# every member array of a tuple bound to a local, which costs ~0.5 us per call for a DynSystem.

# =============================================================================== external forces
@njit(types.void(SYSTEM_T, f8, f8, f8, F1), **INLINE)
def external_force(sys, theta_rad, omega_rad_s, alpha_rad_s2, base_in):
    """Unbalance and base-excitation modal forces at the coming node into sys.fe[0] and sys.fe[1].

    Unbalance (static vector U and couple C in T, rotating with the spindle angle theta about +z_T): the
    d'Alembert load on the rotor is F + i M_perp e_z = (W^2 - i alpha)(U + i C) e^(i theta) in complex form, so the
    modal force is Re[P_k (W^2 - i alpha) e^(i theta)] with P_k = conj(phi_c,k) U_c + i conj(psi_c,k) C_c at the
    unbalance height (DynModel.unb). Base: base_proj @ base_in (relative axis acceleration, rotary accelerations)."""
    unb = sys.dm.unb
    proj = sys.dm.base_proj
    fe = sys.fe
    nm = unb.shape[0]
    sys.scratch[SC_OMEGA] = abs(omega_rad_s)
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    w2 = omega_rad_s * omega_rad_s
    nb = proj.shape[1]
    for k in range(nm):
        pr = unb[k, 0]
        pi = unb[k, 1]
        ar = pr * w2 + pi * alpha_rad_s2
        ai = pi * w2 - pr * alpha_rad_s2
        fe[0, k] = ar * c - ai * s
        fb = 0.0
        for j in range(nb):
            fb += proj[k, j] * base_in[j]
        fe[1, k] = fb


# =============================================================================== L0 step: predictor
@njit(types.void(SYSTEM_T), **INLINE)
def modal_predict(sys):
    """x* = Phi x_n + G0 f_n + G1 (f*_cut + f_unb + f_base)_(n+1), f*_cut = 2 f_cut,n - f_cut,n-1 (or f_cut,n
    without history): the state the cutting kernel evaluates the force at (sys.xp)."""
    Phi = sys.model.Phi
    G0 = sys.model.G0
    G1 = sys.model.G1
    x = sys.state.x
    f_cut = sys.ds.f_cut
    f_prev = sys.ds.f_cut_prev
    f_unb = sys.ds.f_unb
    f_base = sys.ds.f_base
    fe = sys.fe
    xp = sys.xp
    hist = sys.ds.flags[FL_HISTORY] != 0
    nm = x.shape[0]
    for k in range(nm):
        fc = 2.0 * f_cut[k] - f_prev[k] if hist else f_cut[k]
        f0 = f_cut[k] + f_unb[k] + f_base[k]
        f1 = fc + fe[0, k] + fe[1, k]
        q = x[k, 0]
        v = x[k, 1]
        xp[k, 0] = Phi[k, 0, 0] * q + Phi[k, 0, 1] * v + G0[k, 0] * f0 + G1[k, 0] * f1
        xp[k, 1] = Phi[k, 1, 0] * q + Phi[k, 1, 1] * v + G0[k, 1] * f0 + G1[k, 1] * f1


# =============================================================================== deflections
@njit(types.UniTuple(i8, 2)(I8V, i8), **INLINE)
def station_range(act, nst):
    """First and last station (inclusive) of the step from DynSystem.act (every station when unset)."""
    lo = act[ACT_LO]
    hi = act[ACT_HI]
    if lo < 0:
        return 0, nst - 1
    if hi > nst - 1:
        hi = nst - 1
    return lo, hi


@njit(types.void(F2, F3, F2, I1B, I4, F2, F2, F2, F1, I8V, F2, F2, F1, F1, i8, i8), **INLINE)
def _deflection(x, tool_shape, work_shape, is_work, row, residual, st_force, res_work, work_force, iscal, u_st, v_st,
                u_work, v_work, lo, hi):
    """Modal + residual deflections of state x at stations lo..hi (tool stations from tool and machine modes, the
    workpiece at the contact from the work modes); the residual flexibilities use the last evaluated wrenches
    (non-zero only at the stations of the previous step). The stations lo..hi are one contiguous run of 6
    components each (u_st, v_st and a mode's tool_shape row are C-contiguous [station, 6]): the loops over it are
    vectorised, and every element still sums the modes in their order (the same results as a station loop)."""
    nst = u_st.shape[0]
    ns = tool_shape.shape[1]
    nm = x.shape[0]
    hi_s = hi if hi < nst - 1 else nst - 1
    uf = u_st.reshape(6 * nst)
    vf = v_st.reshape(6 * nst)
    if hi_s >= lo:
        u_run = uf[6 * lo:6 * (hi_s + 1)]
        v_run = vf[6 * lo:6 * (hi_s + 1)]
        for j in range(u_run.shape[0]):
            u_run[j] = 0.0
            v_run[j] = 0.0
    for c in range(3):
        u_work[c] = 0.0
        v_work[c] = 0.0
    hi_m = hi_s if hi_s < ns - 1 else ns - 1
    n_run = 6 * (hi_m - lo + 1) if hi_m >= lo else 0
    u_m = uf[6 * lo:6 * lo + n_run]
    v_m = vf[6 * lo:6 * lo + n_run]
    for k in range(nm):
        q = x[k, 0]
        v = x[k, 1]
        r = row[k]
        if is_work[k] == KIND_WORK:
            if 0 <= r < work_shape.shape[0]:
                for c in range(3):
                    u_work[c] += work_shape[r, c] * q
                    v_work[c] += work_shape[r, c] * v
            continue
        if r < 0 or r >= tool_shape.shape[0] or (q == 0.0 and v == 0.0) or n_run == 0:
            continue
        shape_k = tool_shape[r].reshape(6 * ns)[6 * lo:6 * lo + n_run]
        for j in range(n_run):
            phi = shape_k[j]
            u_m[j] += phi * q
            v_m[j] += phi * v
    if iscal[MI_HAS_TOOL_RES] != 0:
        nres = residual.shape[0] // 3
        lim = nres if nres < nst else nst
        a_hi = hi_s if hi_s < lim - 1 else lim - 1
        for b in range(lim):
            fb0 = st_force[b, 0]
            fb1 = st_force[b, 1]
            fb2 = st_force[b, 2]
            if fb0 == 0.0 and fb1 == 0.0 and fb2 == 0.0:
                continue
            for a in range(lo, a_hi + 1):
                for ca in range(3):
                    rr = 3 * a + ca
                    u_st[a, ca] += residual[rr, 3 * b] * fb0 + residual[rr, 3 * b + 1] * fb1 \
                        + residual[rr, 3 * b + 2] * fb2
    if iscal[MI_HAS_WORK_RES] != 0:
        for c in range(3):
            u_work[c] -= res_work[c, 0] * work_force[0] + res_work[c, 1] * work_force[1] + res_work[c, 2] * work_force[2]


@njit(types.void(SYSTEM_T), **INLINE)
def modal_deflection(sys):
    """Deflections at the predicted state sys.xp into sys.defl: tool stations u, v (dx dy dz rx ry rz, T0) from
    tool_shape rows (tool and machine modes) plus the tool residual flexibility times the last station forces;
    the workpiece at the contact (S) from the work modes plus its residual compliance times the reaction. Only
    the stations of sys.act (all when unset) are computed; the others keep their values."""
    lo, hi = station_range(sys.act, sys.defl.u_st_T0_m.shape[0])
    _deflection(sys.xp, sys.model.tool_shape, sys.model.work_shape_at_contact, sys.model.is_work, sys.dm.row,
                sys.model.static_residual, sys.ds.st_force, sys.dm.res_work_S, sys.ds.work_force_S, sys.dm.iscal,
                sys.defl.u_st_T0_m, sys.defl.v_st_T0_m_s, sys.defl.u_work_S_m, sys.defl.v_work_S_m_s, lo, hi)


@njit(types.void(SYSTEM_T, f8, f8, f8, F1), **RAW)
def modal_begin(sys, theta_rad, omega_rad_s, alpha_rad_s2, base_in):
    """external_force + modal_predict + modal_deflection in one call (the same results; one call boundary): the
    predicted state of the coming node and its deflections in sys.defl, ready for the force kernel."""
    external_force(sys, theta_rad, omega_rad_s, alpha_rad_s2, base_in)
    modal_predict(sys)
    modal_deflection(sys)


@njit(types.void(F2, F3, F2, I1B, I4, F2, F2, F2, F1, I8V, F2, F1), **INLINE)
def _relative(x, tool_shape, work_shape, is_work, row, residual, st_force, res_work, work_force, iscal, R, out):
    """Tool [0:3] (T0), workpiece [3:6] (S) and relative [6:9] (T0) deflection of state x at the reference station."""
    ref = iscal[MI_REF_STATION]
    nm = x.shape[0]
    for c in range(9):
        out[c] = 0.0
    ns = tool_shape.shape[1]
    for k in range(nm):
        q = x[k, 0]
        r = row[k]
        if is_work[k] == KIND_WORK:
            if 0 <= r < work_shape.shape[0]:
                for c in range(3):
                    out[3 + c] += work_shape[r, c] * q
        elif 0 <= r < tool_shape.shape[0] and 0 <= ref < ns:
            for c in range(3):
                out[c] += tool_shape[r, ref, c] * q
    if iscal[MI_HAS_TOOL_RES] != 0:
        nres = residual.shape[0] // 3
        if 0 <= ref < nres:
            for b in range(nres):
                for c in range(3):
                    rr = 3 * ref + c
                    out[c] += residual[rr, 3 * b] * st_force[b, 0] + residual[rr, 3 * b + 1] * st_force[b, 1] \
                        + residual[rr, 3 * b + 2] * st_force[b, 2]
    if iscal[MI_HAS_WORK_RES] != 0:
        for c in range(3):
            out[3 + c] -= res_work[c, 0] * work_force[0] + res_work[c, 1] * work_force[1] + res_work[c, 2] * work_force[2]
    for c in range(3):
        out[6 + c] = out[c] - (R[0, c] * out[3] + R[1, c] * out[4] + R[2, c] * out[5])


@njit(types.void(SYSTEM_T, F1), **OPTS)
def relative_deflection(sys, out):
    """Tool (T0) [0:3], workpiece (S) [3:6] and relative tool - work (T0) [6:9] deflection at the reference
    station at the current node (modal + residual with the last evaluated forces)."""
    _relative(sys.state.x, sys.model.tool_shape, sys.model.work_shape_at_contact, sys.model.is_work, sys.dm.row,
              sys.model.static_residual, sys.ds.st_force, sys.dm.res_work_S, sys.ds.work_force_S, sys.dm.iscal,
              sys.dm.R_S_T0, out)


# =============================================================================== L0 step: corrector
@njit(types.void(SYSTEM_T, F2, F1, i8, ACCUM_T), **RAW)
def modal_correct(sys, station_T0, force_S_N, tick, acc):
    """Correct the predicted state with the force evaluated there and advance the node to ``tick``.

    station_T0 [nst, 6]: force and moment on the tool at each station (T0, moment about the station's axis
    point); force_S_N [3]: force on the tool in S (the workpiece takes -F). Modal force f_x = sum_s phi_s . W_s
    (tool and machine modes), -phi_w . F_S (work modes); relaxed correction df = relax (f_x - f*_cut);
    x_(n+1) = x* + G1 df. Energy integrals of the step are exact for the first-order-hold force path
    (DynModel.gram_D / gram_W); structural damping goes to acc[ACC_DAMPING_J], the work of unbalance and base
    excitation to acc[ACC_UNBALANCE_WORK_J] (positive: into the modes). The wrenches are read for the stations
    of sys.act only (zero elsewhere: the caller's engaged stations with their neighbours)."""
    tool_shape = sys.model.tool_shape
    work_shape = sys.model.work_shape_at_contact
    is_work = sys.model.is_work
    G1 = sys.model.G1
    dt = sys.model.dt_s[0]
    row = sys.dm.row
    relax_m = sys.dm.relax
    gram_D = sys.dm.gram_D
    gram_W = sys.dm.gram_W
    omega = sys.dm.omega_rad_s
    iscal = sys.dm.iscal
    x = sys.state.x
    f_prev_tot = sys.state.f_prev
    f_cut = sys.ds.f_cut
    f_cut_prev = sys.ds.f_cut_prev
    f_unb = sys.ds.f_unb
    f_base = sys.ds.f_base
    flags = sys.ds.flags
    node = sys.ds.node_tick
    st_force = sys.ds.st_force
    work_force = sys.ds.work_force_S
    fe = sys.fe
    fx_arr = sys.fx
    xp = sys.xp
    da = sys.da.scal
    nm = x.shape[0]
    nst = station_T0.shape[0]
    step = tick - node[0]
    if step != iscal[MI_L0_TICKS]:
        flags[FL_STEP_ERROR] = 1
        flags[FL_STEP_SEEN] = step
    hist = flags[FL_HISTORY] != 0
    ns = tool_shape.shape[1]
    nsm = ns if ns < nst else nst
    act = sys.act
    s_lo, s_hi = station_range(act, nst)
    if act[ACT_LO] >= 0 and act[ACT_LOAD_LO] >= 0:
        # only the engaged stations can carry a load (the range's margins are for the deflections)
        s_lo = max(s_lo, act[ACT_LOAD_LO])
        s_hi = min(s_hi, act[ACT_LOAD_HI])
    if s_hi > nsm - 1:
        s_hi = nsm - 1
    # ---- modal projection of the evaluated wrenches; fx becomes the predictor miss f_x - f*_cut. The tool / machine
    # modes four at a time over the load stations' components as one contiguous run (C-contiguous [station, 6]):
    # four independent sums, each mode's sum in its order over the stations and components (the same result)
    inv_dt = 1.0 / dt
    w_quad = 0.0
    n_rows = tool_shape.shape[0]
    for k in range(nm):
        fx_arr[k] = 0.0
    n_run = 6 * (s_hi - s_lo + 1) if s_hi >= s_lo else 0
    if n_run > 0:
        w_run = station_T0.reshape(6 * nst)[6 * s_lo:6 * s_lo + n_run]
        kk = 0
        while kk < nm:
            n_g = 0
            ka = -1
            kb = -1
            kc = -1
            kd = -1
            ra = 0
            rb = 0
            rc = 0
            rd = 0
            while kk < nm and n_g < 4:
                r = row[kk]
                if is_work[kk] != KIND_WORK and 0 <= r < n_rows:
                    if n_g == 0:
                        ka = kk
                        ra = r
                    elif n_g == 1:
                        kb = kk
                        rb = r
                    elif n_g == 2:
                        kc = kk
                        rc = r
                    else:
                        kd = kk
                        rd = r
                    n_g += 1
                kk += 1
            if n_g == 0:
                break
            # a short group repeats its first mode in the free lanes (computed, not stored)
            if n_g < 2:
                rb = ra
            if n_g < 3:
                rc = ra
            if n_g < 4:
                rd = ra
            sh_a = tool_shape[ra].reshape(6 * ns)[6 * s_lo:6 * s_lo + n_run]
            sh_b = tool_shape[rb].reshape(6 * ns)[6 * s_lo:6 * s_lo + n_run]
            sh_c = tool_shape[rc].reshape(6 * ns)[6 * s_lo:6 * s_lo + n_run]
            sh_d = tool_shape[rd].reshape(6 * ns)[6 * s_lo:6 * s_lo + n_run]
            sum_a = 0.0
            sum_b = 0.0
            sum_c = 0.0
            sum_d = 0.0
            for j in range(n_run):
                w_j = w_run[j]
                sum_a += sh_a[j] * w_j
                sum_b += sh_b[j] * w_j
                sum_c += sh_c[j] * w_j
                sum_d += sh_d[j] * w_j
            fx_arr[ka] = sum_a
            if n_g > 1:
                fx_arr[kb] = sum_b
            if n_g > 2:
                fx_arr[kc] = sum_c
            if n_g > 3:
                fx_arr[kd] = sum_d
    for k in range(nm):
        r = row[k]
        fx = fx_arr[k]
        if is_work[k] == KIND_WORK:
            fx = 0.0
            if 0 <= r < work_shape.shape[0]:
                fx = -(work_shape[r, 0] * force_S_N[0] + work_shape[r, 1] * force_S_N[1]
                       + work_shape[r, 2] * force_S_N[2])
        w_quad += fx * xp[k, 1] * dt
        fstar = 2.0 * f_cut[k] - f_cut_prev[k] if hist else f_cut[k]
        fx_arr[k] = fx - fstar
    relax = iscal[MI_RELAX] != 0
    e_max = 0.0
    w_cut = 0.0
    w_unb = 0.0
    w_base = 0.0
    dis = 0.0
    for k in range(nm):
        fstar = 2.0 * f_cut[k] - f_cut_prev[k] if hist else f_cut[k]
        if relax:
            miss = 0.0
            for j in range(nm):
                miss += relax_m[k, j] * fx_arr[j]
        else:
            miss = fx_arr[k]
        fcn = fstar + miss
        f0c = f_cut[k]
        f0u = f_unb[k]
        f0b = f_base[k]
        f1u = fe[0, k]
        f1b = fe[1, k]
        f0 = f0c + f0u + f0b
        f1 = fcn + f1u + f1b
        z0 = x[k, 0]
        z1 = x[k, 1]
        z2 = f0
        z3 = (f1 - f0) * inv_dt
        dk = z0 * (gram_D[k, 0, 0] * z0 + gram_D[k, 0, 1] * z1 + gram_D[k, 0, 2] * z2 + gram_D[k, 0, 3] * z3) \
            + z1 * (gram_D[k, 1, 0] * z0 + gram_D[k, 1, 1] * z1 + gram_D[k, 1, 2] * z2 + gram_D[k, 1, 3] * z3) \
            + z2 * (gram_D[k, 2, 0] * z0 + gram_D[k, 2, 1] * z1 + gram_D[k, 2, 2] * z2 + gram_D[k, 2, 3] * z3) \
            + z3 * (gram_D[k, 3, 0] * z0 + gram_D[k, 3, 1] * z1 + gram_D[k, 3, 2] * z2 + gram_D[k, 3, 3] * z3)
        n0 = gram_W[k, 0, 0] * z0 + gram_W[k, 0, 1] * z1 + gram_W[k, 0, 2] * z2 + gram_W[k, 0, 3] * z3
        n1 = gram_W[k, 1, 0] * z0 + gram_W[k, 1, 1] * z1 + gram_W[k, 1, 2] * z2 + gram_W[k, 1, 3] * z3
        w_cut += f0c * n0 + (fcn - f0c) * inv_dt * n1
        w_unb += f0u * n0 + (f1u - f0u) * inv_dt * n1
        w_base += f0b * n0 + (f1b - f0b) * inv_dt * n1
        dis += dk
        # corrected state and the node's force samples
        qn = xp[k, 0] + G1[k, 0] * miss
        vn = xp[k, 1] + G1[k, 1] * miss
        x[k, 0] = qn
        x[k, 1] = vn
        f_cut_prev[k] = f0c
        f_cut[k] = fcn
        f_unb[k] = f1u
        f_base[k] = f1b
        f_prev_tot[k] = f1
        e_max += 0.5 * (vn * vn + omega[k] * omega[k] * qn * qn)
    flags[FL_HISTORY] = 1
    flags[FL_CONTACT] = 1
    node[0] = tick
    if act[ACT_LO] < 0 or act[ACT_PREV_LO] < 0:
        c_lo = 0
        c_hi = st_force.shape[0] - 1
    else:
        c_lo = min(act[ACT_LO], act[ACT_PREV_LO])
        c_hi = min(max(act[ACT_HI], act[ACT_PREV_HI]), st_force.shape[0] - 1)
    # the residual flexibilities' forces: this evaluation's, or with an estimate (MI_RES_RELAX) the fixed point
    # F_(n-1) + (I - K_c C)^-1 (F_n - F_(n-1)) at the load stations and for the workpiece (modal.residual_relaxation)
    res_relax = iscal[MI_RES_RELAX] != 0
    rr = sys.dm.scal
    for s in range(c_lo, c_hi + 1):
        if res_relax and s_lo <= s <= s_hi and s < nst:
            d0 = station_T0[s, 0] - st_force[s, 0]
            d1 = station_T0[s, 1] - st_force[s, 1]
            d2 = station_T0[s, 2] - st_force[s, 2]
            st_force[s, 0] += rr[MS_RES_RELAX] * d0 + rr[MS_RES_RELAX + 1] * d1 + rr[MS_RES_RELAX + 2] * d2
            st_force[s, 1] += rr[MS_RES_RELAX + 3] * d0 + rr[MS_RES_RELAX + 4] * d1 + rr[MS_RES_RELAX + 5] * d2
            st_force[s, 2] += rr[MS_RES_RELAX + 6] * d0 + rr[MS_RES_RELAX + 7] * d1 + rr[MS_RES_RELAX + 8] * d2
            for c in range(3, 6):
                st_force[s, c] = station_T0[s, c]
        else:
            for c in range(6):
                st_force[s, c] = station_T0[s, c] if s < nst else 0.0
    act[ACT_PREV_LO] = act[ACT_LO]
    act[ACT_PREV_HI] = act[ACT_HI]
    if res_relax:
        # in T0 (R^T F_S), relaxed, back to S
        Rw = sys.dm.R_S_T0
        e0 = force_S_N[0] - work_force[0]
        e1 = force_S_N[1] - work_force[1]
        e2 = force_S_N[2] - work_force[2]
        t0 = Rw[0, 0] * e0 + Rw[1, 0] * e1 + Rw[2, 0] * e2
        t1 = Rw[0, 1] * e0 + Rw[1, 1] * e1 + Rw[2, 1] * e2
        t2 = Rw[0, 2] * e0 + Rw[1, 2] * e1 + Rw[2, 2] * e2
        g0 = rr[MS_RES_RELAX] * t0 + rr[MS_RES_RELAX + 1] * t1 + rr[MS_RES_RELAX + 2] * t2
        g1 = rr[MS_RES_RELAX + 3] * t0 + rr[MS_RES_RELAX + 4] * t1 + rr[MS_RES_RELAX + 5] * t2
        g2 = rr[MS_RES_RELAX + 6] * t0 + rr[MS_RES_RELAX + 7] * t1 + rr[MS_RES_RELAX + 8] * t2
        for c in range(3):
            work_force[c] += Rw[c, 0] * g0 + Rw[c, 1] * g1 + Rw[c, 2] * g2
    else:
        for c in range(3):
            work_force[c] = force_S_N[c]
    da[DA_W_CUT_J] += w_cut
    da[DA_W_UNB_J] += w_unb
    da[DA_W_BASE_J] += w_base
    da[DA_DAMPING_J] += dis
    da[DA_W_CUT_QUAD_J] += w_quad
    da[DA_N_L0] += 1.0
    if e_max > da[DA_ENERGY_MAX_J]:
        da[DA_ENERGY_MAX_J] = e_max
    fmag = math.sqrt(force_S_N[0] * force_S_N[0] + force_S_N[1] * force_S_N[1] + force_S_N[2] * force_S_N[2])
    if fmag > da[DA_FORCE_MAX_N]:
        da[DA_FORCE_MAX_N] = fmag
    acc.scalars[ACC_DAMPING_J] += dis
    acc.scalars[ACC_UNBALANCE_WORK_J] += w_unb + w_base
    # ---- vibration sample at the reference station: relative modal shape there (DynModel.contact_shape) plus the
    # tool residual (block row of the reference station) and the workpiece residual with this step's forces as the
    # residuals hold them (st_force, work_force: relaxed with an estimate)
    cshape = sys.dm.contact_shape
    residual = sys.model.static_residual
    res_w = sys.dm.res_work_S
    R = sys.dm.R_S_T0
    ref = iscal[MI_REF_STATION]
    r0 = 0.0
    r1 = 0.0
    r2 = 0.0
    for k in range(nm):
        q = x[k, 0]
        r0 += cshape[k, 0] * q
        r1 += cshape[k, 1] * q
        r2 += cshape[k, 2] * q
    if iscal[MI_HAS_TOOL_RES] != 0:
        nres = residual.shape[0] // 3
        if 0 <= ref < nres:
            for b in range(s_lo, min(nres, nst, s_hi + 1)):
                fb0 = st_force[b, 0]
                fb1 = st_force[b, 1]
                fb2 = st_force[b, 2]
                r0 += residual[3 * ref, 3 * b] * fb0 + residual[3 * ref, 3 * b + 1] * fb1 + residual[3 * ref, 3 * b + 2] * fb2
                r1 += residual[3 * ref + 1, 3 * b] * fb0 + residual[3 * ref + 1, 3 * b + 1] * fb1 \
                    + residual[3 * ref + 1, 3 * b + 2] * fb2
                r2 += residual[3 * ref + 2, 3 * b] * fb0 + residual[3 * ref + 2, 3 * b + 1] * fb1 \
                    + residual[3 * ref + 2, 3 * b + 2] * fb2
    if iscal[MI_HAS_WORK_RES] != 0:
        # workpiece residual deflection -C F (S); the relative deflection subtracts it in T0: + R^T C F
        w0 = res_w[0, 0] * work_force[0] + res_w[0, 1] * work_force[1] + res_w[0, 2] * work_force[2]
        w1 = res_w[1, 0] * work_force[0] + res_w[1, 1] * work_force[1] + res_w[1, 2] * work_force[2]
        w2 = res_w[2, 0] * work_force[0] + res_w[2, 1] * work_force[1] + res_w[2, 2] * work_force[2]
        r0 += R[0, 0] * w0 + R[1, 0] * w1 + R[2, 0] * w2
        r1 += R[0, 1] * w0 + R[1, 1] * w1 + R[2, 1] * w2
        r2 += R[0, 2] * w0 + R[1, 2] * w1 + R[2, 2] * w2
    if r0 < da[DA_UMIN_X]:
        da[DA_UMIN_X] = r0
    if r0 > da[DA_UMAX_X]:
        da[DA_UMAX_X] = r0
    if r1 < da[DA_UMIN_X + 1]:
        da[DA_UMIN_X + 1] = r1
    if r1 > da[DA_UMAX_X + 1]:
        da[DA_UMAX_X + 1] = r1
    if r2 < da[DA_UMIN_X + 2]:
        da[DA_UMIN_X + 2] = r2
    if r2 > da[DA_UMAX_X + 2]:
        da[DA_UMAX_X + 2] = r2
    samp_n = sys.ds.samp_n
    n = samp_n[0]
    if n < sys.ds.samp_tick.shape[0]:
        samp_u = sys.ds.samp_u
        sys.ds.samp_tick[n] = tick
        samp_u[n, 0] = r0
        samp_u[n, 1] = r1
        samp_u[n, 2] = r2
        sys.ds.samp_w[n] = sys.scratch[SC_OMEGA]
        samp_n[0] = n + 1
    else:
        samp_n[1] += 1


# =============================================================================== out of contact
@njit(types.void(SYSTEM_T, f8, f8, f8, f8, F1, F1, i8), **OPTS)
def modal_free(sys, theta0_rad, theta1_rad, omega0_rad_s, omega1_rad_s, base0, base1, tick1):
    """Exact propagation from the current node to ``tick1`` without cutting forces: harmonic unbalance with the
    interval's mean speed W = (theta1 - theta0) / h (theta unwrapped) and alpha = (omega1 - omega0) / h, base
    excitation linear between base0 and base1 (first-order hold), closed-form work and dissipation."""
    unb = sys.dm.unb
    proj = sys.dm.base_proj
    omega = sys.dm.omega_rad_s
    zeta = sys.dm.zeta
    x = sys.state.x
    f_prev_tot = sys.state.f_prev
    f_cut = sys.ds.f_cut
    f_cut_prev = sys.ds.f_cut_prev
    f_unb = sys.ds.f_unb
    f_base = sys.ds.f_base
    flags = sys.ds.flags
    node = sys.ds.node_tick
    st_force = sys.ds.st_force
    work_force = sys.ds.work_force_S
    da = sys.da.scal
    scratch = sys.scratch
    nm = x.shape[0]
    h = (tick1 - node[0]) * sys.dm.scal[MS_TICK_S]
    if h <= 0.0:
        flags[FL_STEP_ERROR] = 1
        flags[FL_STEP_SEEN] = tick1 - node[0]
        return
    W = (theta1_rad - theta0_rad) / h
    alpha = (omega1_rad_s - omega0_rad_s) / h
    ar = W * W
    ai = -alpha
    c0 = math.cos(theta0_rad)
    s0 = math.sin(theta0_rad)
    cW = math.cos(W * h)
    sW = math.sin(W * h)
    nb = proj.shape[1]
    piece = scratch[9:14]
    w_h = 0.0
    w_l = 0.0
    dis = 0.0
    for k in range(nm):
        pr = unb[k, 0]
        pi = unb[k, 1]
        # P = (pr + i pi)(ar + i ai)(c0 + i s0)
        tr = pr * ar - pi * ai
        ti = pr * ai + pi * ar
        Pr = tr * c0 - ti * s0
        Pi = tr * s0 + ti * c0
        g0 = 0.0
        gb = 0.0
        for j in range(nb):
            g0 += proj[k, j] * base0[j]
            gb += proj[k, j] * base1[j]
        g1 = (gb - g0) / h
        free_interval(omega[k], zeta[k], h, x[k, 0], x[k, 1], Pr, Pi, W, g0, g1, piece)
        x[k, 0] = piece[0]
        x[k, 1] = piece[1]
        w_h += piece[2]
        w_l += piece[3]
        dis += piece[4]
        f_unb[k] = Pr * cW - Pi * sW
        f_base[k] = gb
        f_cut[k] = 0.0
        f_cut_prev[k] = 0.0
        f_prev_tot[k] = f_unb[k] + gb
    flags[FL_HISTORY] = 0
    flags[FL_CONTACT] = 0
    node[0] = tick1
    for s in range(st_force.shape[0]):
        for c in range(6):
            st_force[s, c] = 0.0
    for c in range(3):
        work_force[c] = 0.0
    da[DA_W_UNB_J] += w_h
    da[DA_W_BASE_J] += w_l
    da[DA_DAMPING_J] += dis
    da[DA_N_FREE] += 1.0
    e = 0.0
    for k in range(nm):
        e += 0.5 * (x[k, 1] * x[k, 1] + omega[k] * omega[k] * x[k, 0] * x[k, 0])
    if e > da[DA_ENERGY_MAX_J]:
        da[DA_ENERGY_MAX_J] = e
    rel = scratch
    _relative(x, sys.model.tool_shape, sys.model.work_shape_at_contact, sys.model.is_work, sys.dm.row,
              sys.model.static_residual, st_force, sys.dm.res_work_S, work_force, sys.dm.iscal, sys.dm.R_S_T0, rel)
    for c in range(3):
        if rel[6 + c] < da[DA_UMIN_X + c]:
            da[DA_UMIN_X + c] = rel[6 + c]
        if rel[6 + c] > da[DA_UMAX_X + c]:
            da[DA_UMAX_X + c] = rel[6 + c]


@njit(types.void(SYSTEM_T, i8), **OPTS)
def record_tick(sys, row):
    """Deflections at the current node into sys.da.tick_u[row] (call at every L1 tick end)."""
    tick_u = sys.da.tick_u
    if row < 0 or row >= tick_u.shape[0]:
        return
    rel = sys.scratch
    _relative(sys.state.x, sys.model.tool_shape, sys.model.work_shape_at_contact, sys.model.is_work, sys.dm.row,
              sys.model.static_residual, sys.ds.st_force, sys.dm.res_work_S, sys.ds.work_force_S, sys.dm.iscal,
              sys.dm.R_S_T0, rel)
    for c in range(N_TU):
        tick_u[row, c] = rel[c]
    tick_n = sys.da.tick_n
    if row + 1 > tick_n[0]:
        tick_n[0] = row + 1


# =============================================================================== refresh and helpers
@njit(types.void(SYSTEM_T, GRID_T, F1), **OPTS)
def refresh_work_shapes(sys, grid, point_S):
    """Workpiece mode shapes and residual compliance at ``point_S`` (S) by trilinear interpolation between
    cell centres (origin + (i + 1/2) cell, clamped to the outer centres); updates ModalModel.work_shape_at_contact,
    DynModel.res_work_S (+ grid.extra_S) and the work rows of DynModel.contact_shape (-R_S_T0^T phi, T0)."""
    wsc = sys.model.work_shape_at_contact
    is_work = sys.model.is_work
    res = sys.dm.res_work_S
    cshape = sys.dm.contact_shape
    row = sys.dm.row
    iscal = sys.dm.iscal
    gshape = grid.shape
    gcomp = grid.compliance
    origin = grid.origin_S_m
    extra = grid.extra_S
    R = grid.R_S_T0
    cell = grid.cell_m[0]
    nwm = gshape.shape[0]
    nx = gshape.shape[1]
    ny = gshape.shape[2]
    nz = gshape.shape[3]
    i0 = 0
    i1 = 0
    i2 = 0
    f0 = 0.0
    f1 = 0.0
    f2 = 0.0
    for a in range(3):
        dims = nx if a == 0 else (ny if a == 1 else nz)
        if cell > 0.0:
            u = (point_S[a] - origin[a]) / cell - 0.5
        else:
            u = 0.0
        if dims <= 1 or u <= 0.0:
            ia = 0
            fa = 0.0
        elif u >= dims - 1:
            ia = dims - 2
            fa = 1.0
        else:
            ia = int(math.floor(u))
            fa = u - ia
        if a == 0:
            i0 = ia
            f0 = fa
        elif a == 1:
            i1 = ia
            f1 = fa
        else:
            i2 = ia
            f2 = fa
    nres = wsc.shape[0]
    for r in range(nres):
        for c in range(3):
            wsc[r, c] = 0.0
    for c in range(3):
        for d in range(3):
            res[c, d] = extra[c, d]
    same = gcomp.shape[0] == nx and gcomp.shape[1] == ny and gcomp.shape[2] == nz
    for ii in range(2):
        wx = f0 if ii == 1 else 1.0 - f0
        ix = i0 + ii if nx > 1 else 0
        if wx == 0.0:
            continue
        for jj in range(2):
            wy = f1 if jj == 1 else 1.0 - f1
            iy = i1 + jj if ny > 1 else 0
            if wy == 0.0:
                continue
            for kk in range(2):
                wz = f2 if kk == 1 else 1.0 - f2
                iz = i2 + kk if nz > 1 else 0
                if wz == 0.0:
                    continue
                wgt = wx * wy * wz
                for r in range(min(nres, nwm)):
                    for c in range(3):
                        wsc[r, c] += wgt * gshape[r, ix, iy, iz, c]
                if same:
                    for c in range(3):
                        for d in range(3):
                            res[c, d] += wgt * gcomp[ix, iy, iz, c, d]
    nm = row.shape[0]
    for k in range(nm):
        if is_work[k] != KIND_WORK:
            continue
        r = row[k]
        if 0 <= r < nres:
            for c in range(3):
                cshape[k, c] = -(R[0, c] * wsc[r, 0] + R[1, c] * wsc[r, 1] + R[2, c] * wsc[r, 2])
    any_res = 0
    for c in range(3):
        for d in range(3):
            if res[c, d] != 0.0:
                any_res = 1
    iscal[MI_HAS_WORK_RES] = any_res


@njit(types.void(AXES_T, F1, F1), **OPTS)
def relative_axis_acceleration(axes, acc_axis, out):
    """Base-excitation input from the axis accelerations: out[0:3] = sum_a side_a acc_a dir_M[a] (linear axes, the
    tool side's rigid-body acceleration relative to the workpiece side, M); out[3 + column] = acc_a (rotary)."""
    side_a = axes.side
    rotary = axes.rotary
    column = axes.column
    dir_M = axes.dir_M
    for j in range(out.shape[0]):
        out[j] = 0.0
    na = side_a.shape[0]
    for a in range(min(na, acc_axis.shape[0])):
        side = side_a[a]
        if side == 0.0:
            continue
        if rotary[a] != 0:
            col = column[a]
            if col >= 0 and BASE_LINEAR + col < out.shape[0]:
                out[BASE_LINEAR + col] = acc_axis[a]
            continue
        for c in range(3):
            out[c] += side * acc_axis[a] * dir_M[a, c]


@njit(f8(f8, f8, f8, f8), **OPTS)
def interference_area(slope, land_m, tan_clear, max_len_m):
    """Area (m2 per m of edge) between the flank and the surface the edge has just generated for a path entering
    the material with slope = v_n / V (Wu 1989; Altintas, Eynian & Onozuka 2008; Ahmadi & Ismail 2011): a
    zero-clearance land of length L, then the clearance face (tan gamma); interference ends where
    slope x = (x - L) tan gamma, at most max_len behind the edge. A = int_0^x* (slope x - max(0, x - L) tan gamma) dx.
    (The same expression as cnctwin.cutting.kernels.flank_interference_area.)"""
    if slope <= 0.0:
        return 0.0
    L = land_m
    if L > max_len_m:
        L = max_len_m
    if L < 0.0:
        L = 0.0
    if slope >= tan_clear:
        xs = max_len_m
    else:
        xs = L * tan_clear / (tan_clear - slope)
        if xs > max_len_m:
            xs = max_len_m
    if xs < L:
        xs = L
    d = xs - L
    return 0.5 * slope * xs * xs - 0.5 * tan_clear * d * d


@njit(f8(f8, f8, f8, f8, f8, f8, f8, f8, ACCUM_T), **OPTS)
def process_damping_element(K_sp_N_per_m3, db_m, land_m, tan_clear, max_len_m, v_n_m_s, V_m_s, dt_s, acc):
    """Process-damping (flank indentation) force of one element (N, acting on the tool along -n: out of the
    material) F = K_sp db A(v_n / V), and its dissipation F v_n dt booked into acc[ACC_PROCESS_DAMPING_J].
    v_n is the element's velocity along its chip-thickness normal relative to the material (into the material
    positive), V its cutting speed. Only motion into the material indents (one-sided)."""
    if v_n_m_s <= 0.0 or V_m_s <= 0.0 or K_sp_N_per_m3 <= 0.0:
        return 0.0
    F = K_sp_N_per_m3 * db_m * interference_area(v_n_m_s / V_m_s, land_m, tan_clear, max_len_m)
    acc.scalars[ACC_PROCESS_DAMPING_J] += F * v_n_m_s * dt_s
    return F


@njit(types.void(DEFL_T, i8, F1, F1), **OPTS)
def deflection_at_point(defl, s, r_T0, out):
    """Deflection and rate of a point of station s at offset r (T0, from the station's axis point):
    u = u_s + theta_s x r, v = v_s + thetadot_s x r -> out[0:3], out[3:6]."""
    u = defl.u_st_T0_m
    v = defl.v_st_T0_m_s
    out[0] = u[s, 0] + u[s, 4] * r_T0[2] - u[s, 5] * r_T0[1]
    out[1] = u[s, 1] + u[s, 5] * r_T0[0] - u[s, 3] * r_T0[2]
    out[2] = u[s, 2] + u[s, 3] * r_T0[1] - u[s, 4] * r_T0[0]
    out[3] = v[s, 0] + v[s, 4] * r_T0[2] - v[s, 5] * r_T0[1]
    out[4] = v[s, 1] + v[s, 5] * r_T0[0] - v[s, 3] * r_T0[2]
    out[5] = v[s, 2] + v[s, 3] * r_T0[1] - v[s, 4] * r_T0[0]


@njit(types.void(SYSTEM_T, RING_T, i8, i8), **OPTS)
def ring_write(sys, ring, slot, col0):
    """Write the relative deflection at the reference station (T0, 3 columns) of the current node into the fast
    ring row ``slot`` from column ``col0`` (Recorder.fast_columns("dynamics.vibration_rel_m") + 3 path)."""
    data = ring.data
    if col0 < 0 or slot < 0 or slot >= data.shape[0] or col0 + 3 > data.shape[1]:
        return
    rel = sys.scratch
    _relative(sys.state.x, sys.model.tool_shape, sys.model.work_shape_at_contact, sys.model.is_work, sys.dm.row,
              sys.model.static_residual, sys.ds.st_force, sys.dm.res_work_S, sys.ds.work_force_S, sys.dm.iscal,
              sys.dm.R_S_T0, rel)
    for c in range(3):
        data[slot, col0 + c] = rel[6 + c]


KERNELS = (phi1, psi1, j0, j1, free_interval, external_force, modal_predict, _deflection, modal_deflection, modal_begin,
           _relative,
           relative_deflection, modal_correct, modal_free, record_tick, refresh_work_shapes, relative_axis_acceleration,
           interference_area, process_damping_element, deflection_at_point, ring_write)

__all__ = ["phi1", "psi1", "j0", "j1", "free_interval", "external_force", "modal_predict", "modal_deflection", "modal_begin",
           "relative_deflection", "modal_correct", "modal_free", "record_tick", "refresh_work_shapes",
           "relative_axis_acceleration", "interference_area", "process_damping_element", "deflection_at_point",
           "ring_write", "KERNELS"]
