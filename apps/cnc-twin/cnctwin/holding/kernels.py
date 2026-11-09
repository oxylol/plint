"""Compiled kernels of the holding domain (docs/models/holding.md "Kernels"), each with a plain-Python
reference in :mod:`cnctwin.holding.reference` that the tests compare against.

* :func:`contact_eval` - forces of every contact of a part at a rigid-body displacement, the residual of
  equilibrium and its consistent tangent (elastic contacts, unilateral normal law, Coulomb friction by
  return mapping on the tangential vector, bonded nodes).
* :func:`newton_path` / :func:`solve_equilibrium` - the displacement that balances an external wrench
  (Newton with backtracking line search and load increments from the reference configuration, a retry with
  more increments; deterministic: fixed tolerance, fixed iteration order, zero start).
* :func:`capacity` - the load factor at which the part starts to move under ``w0 + lambda wv`` (doubling
  bracket and bisection on "held"), the contact state there and the incipient motion (mechanism).
* :func:`mechanism` - direction of the incipient motion with the critical contacts released.
* :func:`ratchet_part` - stick-slip (or lift-off) of the part along its mechanism under the tick-wise mean
  and tooth-period pulse of the demand (1 DOF, rigid body, Coulomb resistance or constant restoring force; the
  slip no faster than the cut can push the part).
* :func:`ratchet_tool` - slip and axial walk of a tool in a friction holder (2 DOF: circumferential and
  axial sliding at the shank, isotropic Coulomb friction circle with an implicit friction impulse, back stop
  and walk limit; the tool stalls at the spindle speed, the cutting torque cannot turn it backwards).
* :func:`pulse_integral`, :func:`shape_mean` - the demand's tooth-period shape (abi.Wave) and its exact mean over
  a sub-step (the kernels apply every sub-step's impulse exactly).

Energy of a ratchet (out RP_IN_J / RT_IN_J, RP_LOSS_J / RT_WORK_J): the work the demand does on the motion equals
the dissipation plus the change of the kinetic (and, lifting, potential) energy of the moving body exactly for
the discrete scheme (constant acceleration within a sub-step for the part; the implicit friction impulse along
the new velocity and trapezoidal positions for the tool): the run's ledger books both sides.

Rules (architecture 5.5): explicit signatures, cache, nogil, error_model numpy, no fastmath, no parallel.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, typeof, types

from .abi import (CFG_K_REG, CFG_L_C, CFG_LAM_MAX, CFG_LAM_TOL, CFG_MAX_DISP, CFG_MAX_ITER, CFG_MAX_ROT, CFG_N_INC,
                  CFG_TOL, CFG_UTIL_EPS, INFO_ITER, INFO_RESIDUAL, INFO_SCALE_N, INFO_STATUS, KIND_BONDED, KIND_OFF,
                  MODE_FRICTION, MODE_SEPARATION, RP_DQ, RP_IN_J, RP_LOSS_J, RP_MOVING_S, RP_Q_MAX, RP_WORK_J, RT_DX1,
                  RT_DX2, RT_IN_J, RT_MOVING_S, RT_PEAK_RATIO, RT_WORK_J, ST_BONDED, ST_OFF, ST_OPEN, ST_SLIP, ST_STICK,
                  STATUS_HELD, STATUS_MOVES,
                  STATUS_NOT_CONVERGED, Contacts, Wave, empty_contacts, empty_wave)

F1 = types.float64[::1]
F2 = types.float64[:, ::1]
I1 = types.int32[::1]
F8 = types.float64
I4 = types.int32
I8 = types.int64


def example_contacts() -> Contacts:
    return empty_contacts(1)


def example_wave() -> Wave:
    return empty_wave(1, 1)


CONTACTS_T = typeof(example_contacts())
WAVE_T = typeof(example_wave())

LINE_SEARCH_STEPS = 10          # halvings of the Newton step before the smallest step is taken
LINE_SEARCH_DECREASE = 1e-4     # Armijo constant on the scaled residual
PIVOT_EPS = 1e-14               # relative pivot below which the tangent is singular
RETRY_FACTOR = 8                # load increments of the retry, relative to the configured number
RUNAWAY = 100.0                 # a Newton step this many times the displacement limit is a mechanism
LAMBDA_ITER = 40                # safeguarded Newton iterations of the implicit slip (unequal masses)
MECH_REG = 1e-8                 # regularisation of the mechanism solve, relative to the stiffest contact
COARSE_MARGIN = 2.0             # above this load factor the capacity is resolved to COARSE_TOL (relative) only
COARSE_TOL = 0.1


# --------------------------------------------------------------------------- small linear algebra
@njit(I4(F2, F1, F1), cache=True, nogil=True, error_model="numpy")
def solve6(A, b, x):
    """x = A^-1 b for a 6x6 system (Gaussian elimination with partial pivoting); 1 if singular."""
    n = 6
    M = np.empty((n, n))
    y = np.empty(n)
    scale = 0.0
    for i in range(n):
        y[i] = b[i]
        for j in range(n):
            M[i, j] = A[i, j]
            if abs(A[i, j]) > scale:
                scale = abs(A[i, j])
    if scale == 0.0:
        return 1
    for c in range(n):
        p = c
        best = abs(M[c, c])
        for r in range(c + 1, n):
            if abs(M[r, c]) > best:
                best = abs(M[r, c])
                p = r
        if best <= PIVOT_EPS * scale:
            return 1
        if p != c:
            for j in range(n):
                tmp = M[c, j]
                M[c, j] = M[p, j]
                M[p, j] = tmp
            tmp = y[c]
            y[c] = y[p]
            y[p] = tmp
        for r in range(c + 1, n):
            f = M[r, c] / M[c, c]
            if f != 0.0:
                for j in range(c, n):
                    M[r, j] -= f * M[c, j]
                y[r] -= f * y[c]
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= M[i, j] * x[j]
        x[i] = s / M[i, i]
    return 0


@njit(F8(F1, F8, F8), cache=True, nogil=True, error_model="numpy")
def scaled_norm(R, f_scale, l_c):
    """max(|R_F| / f_scale, |R_M| / (f_scale l_c))."""
    rf = math.sqrt(R[0] * R[0] + R[1] * R[1] + R[2] * R[2])
    rm = math.sqrt(R[3] * R[3] + R[4] * R[4] + R[5] * R[5])
    a = rf / f_scale
    b = rm / (f_scale * l_c)
    return a if a > b else b


# --------------------------------------------------------------------------- contact forces
@njit(F8(CONTACTS_T, F1, F1, F8, F8, I1, F2, I1, F1, F2), cache=True, nogil=True, error_model="numpy")
def contact_eval(cs, delta, w_ext, k_reg, l_c, release, f_out, st_out, R_out, J_out):
    """Contact forces on the part at the twist ``delta`` = (u, phi) about O, the residual
    R = w_ext + sum(G f) - K_reg delta and its tangent J = dR/d(delta); returns the force scale of R.

    Normal law f_n = pre_n + kn (gap - n.d), d = u + phi x r; unilateral contacts carry f_n >= 0 only (a
    contact exactly at f_n = 0 is closed: its stiffness is the generalized Jacobian's element at the kink).
    Tangential trial f_t = pre_t - kt (t.d), Coulomb return mapping |f_t| <= mu f_n. ``release[i]`` = 1
    treats a sticking contact as slipping (the slip direction of its trial force), 2 as open (mechanism)."""
    n = cs.r_m.shape[0]
    reg_r = k_reg * l_c * l_c
    f_scale = 0.0
    for a in range(6):
        R_out[a] = w_ext[a]
        for b in range(6):
            J_out[a, b] = 0.0
    for a in range(3):
        R_out[a] -= k_reg * delta[a]
        J_out[a, a] -= k_reg
        R_out[a + 3] -= reg_r * delta[a + 3]
        J_out[a + 3, a + 3] -= reg_r
    f_scale += math.sqrt(w_ext[0] ** 2 + w_ext[1] ** 2 + w_ext[2] ** 2)
    f_scale += math.sqrt(w_ext[3] ** 2 + w_ext[4] ** 2 + w_ext[5] ** 2) / l_c
    K = np.zeros((3, 3))
    for i in range(n):
        f_out[i, 0] = 0.0
        f_out[i, 1] = 0.0
        f_out[i, 2] = 0.0
        kind = cs.kind[i]
        if kind == KIND_OFF:
            st_out[i] = ST_OFF
            continue
        rx = cs.r_m[i, 0]
        ry = cs.r_m[i, 1]
        rz = cs.r_m[i, 2]
        # d = u + phi x r
        dx = delta[0] + delta[4] * rz - delta[5] * ry
        dy = delta[1] + delta[5] * rx - delta[3] * rz
        dz = delta[2] + delta[3] * ry - delta[4] * rx
        nx, ny, nz = cs.n[i, 0], cs.n[i, 1], cs.n[i, 2]
        ax, ay, az = cs.t1[i, 0], cs.t1[i, 1], cs.t1[i, 2]
        bx, by, bz = cs.t2[i, 0], cs.t2[i, 1], cs.t2[i, 2]
        kn = cs.kn_N_per_m[i]
        k1 = cs.kt_N_per_m[i, 0]
        k2 = cs.kt_N_per_m[i, 1]
        mu = cs.mu[i]
        f_scale += abs(cs.pre_N[i, 0]) + abs(cs.pre_N[i, 1]) + abs(cs.pre_N[i, 2]) + kn * abs(cs.gap_m[i])
        dn = nx * dx + ny * dy + nz * dz
        d1 = ax * dx + ay * dy + az * dz
        d2 = bx * dx + by * dy + bz * dz
        fn = cs.pre_N[i, 0] + kn * (cs.gap_m[i] - dn)
        for p in range(3):
            for q in range(3):
                K[p, q] = 0.0
        if kind == KIND_BONDED or (fn >= 0.0 and release[i] != 2):
            f1 = cs.pre_N[i, 1] - k1 * d1
            f2 = cs.pre_N[i, 2] - k2 * d2
            state = ST_BONDED if kind == KIND_BONDED else ST_STICK
            # stick tangent: -kn n n^T - k1 t1 t1^T - k2 t2 t2^T
            nv = (nx, ny, nz)
            av = (ax, ay, az)
            bv = (bx, by, bz)
            for p in range(3):
                for q in range(3):
                    K[p, q] = -kn * nv[p] * nv[q]
            if kind != KIND_BONDED and mu <= 0.0:
                f1 = 0.0
                f2 = 0.0
            else:
                ft = math.sqrt(f1 * f1 + f2 * f2)
                cap = mu * fn
                slip = kind != KIND_BONDED and (ft > cap or (release[i] == 1 and ft > 0.0))
                if not slip:
                    for p in range(3):
                        for q in range(3):
                            K[p, q] -= k1 * av[p] * av[q] + k2 * bv[p] * bv[q]
                else:
                    state = ST_SLIP
                    s1 = f1 / ft
                    s2 = f2 / ft
                    ratio = cap / ft
                    if ratio > 1.0:
                        ratio = 1.0
                    f1 = ratio * ft * s1
                    f2 = ratio * ft * s2
                    # d(f1, f2)/dd = mu s dfn/dd + ratio (I - s s^T) [-k1 t1^T; -k2 t2^T]
                    m11 = ratio * (1.0 - s1 * s1)
                    m12 = -ratio * s1 * s2
                    m22 = ratio * (1.0 - s2 * s2)
                    for p in range(3):
                        sv = s1 * av[p] + s2 * bv[p]
                        for q in range(3):
                            K[p, q] += -mu * kn * sv * nv[q]
                            g1 = -k1 * av[q]
                            g2 = -k2 * bv[q]
                            K[p, q] += av[p] * (m11 * g1 + m12 * g2) + bv[p] * (m12 * g1 + m22 * g2)
            fx = fn * nx + f1 * ax + f2 * bx
            fy = fn * ny + f1 * ay + f2 * by
            fz = fn * nz + f1 * az + f2 * bz
            st_out[i] = state
        else:
            st_out[i] = ST_OPEN
            continue
        f_out[i, 0] = fx
        f_out[i, 1] = fy
        f_out[i, 2] = fz
        R_out[0] += fx
        R_out[1] += fy
        R_out[2] += fz
        R_out[3] += ry * fz - rz * fy
        R_out[4] += rz * fx - rx * fz
        R_out[5] += rx * fy - ry * fx
        # dF/d(delta) = [K, -K [r]x];  dM/d(delta) = [r]x dF/d(delta);  [r]x = skew(r)
        for p in range(3):
            # column block for u
            for q in range(3):
                J_out[p, q] += K[p, q]
            # column block for phi: -K [r]x  -> -(K[:,0..2] @ skew(r))
            kr0 = -(K[p, 1] * rz - K[p, 2] * ry)
            kr1 = -(K[p, 2] * rx - K[p, 0] * rz)
            kr2 = -(K[p, 0] * ry - K[p, 1] * rx)
            J_out[p, 3] += kr0
            J_out[p, 4] += kr1
            J_out[p, 5] += kr2
        # rows of the moment: [r]x times the force-derivative columns of this contact
        for q in range(6):
            # force-derivative column q of this contact
            if q < 3:
                g0 = K[0, q]
                g1 = K[1, q]
                g2 = K[2, q]
            else:
                qq = q - 3
                if qq == 0:
                    g0 = -(K[0, 1] * rz - K[0, 2] * ry)
                    g1 = -(K[1, 1] * rz - K[1, 2] * ry)
                    g2 = -(K[2, 1] * rz - K[2, 2] * ry)
                elif qq == 1:
                    g0 = -(K[0, 2] * rx - K[0, 0] * rz)
                    g1 = -(K[1, 2] * rx - K[1, 0] * rz)
                    g2 = -(K[2, 2] * rx - K[2, 0] * rz)
                else:
                    g0 = -(K[0, 0] * ry - K[0, 1] * rx)
                    g1 = -(K[1, 0] * ry - K[1, 1] * rx)
                    g2 = -(K[2, 0] * ry - K[2, 1] * rx)
            J_out[3, q] += ry * g2 - rz * g1
            J_out[4, q] += rz * g0 - rx * g2
            J_out[5, q] += rx * g1 - ry * g0
    if f_scale <= 0.0:
        f_scale = 1.0
    return f_scale


# --------------------------------------------------------------------------- equilibrium
@njit(I4(CONTACTS_T, F1, F1, I8, F1, F2, I1, F1), cache=True, nogil=True, error_model="numpy")
def newton_path(cs, w_ext, cfg, n_inc, delta, f_out, st_out, info):
    """Newton with backtracking line search along n_inc load increments from the reference configuration
    (delta = 0, the external load the preloads balance there); the evaluation that ends a line search is the
    next iterate's (no second evaluation). Returns the status."""
    n = cs.r_m.shape[0]
    tol = cfg[CFG_TOL]
    max_iter = int(cfg[CFG_MAX_ITER])
    max_disp = cfg[CFG_MAX_DISP]
    max_rot = cfg[CFG_MAX_ROT]
    k_reg = cfg[CFG_K_REG]
    l_c = cfg[CFG_L_C]
    release = np.zeros(n, np.int32)
    R = np.zeros(6)
    J = np.zeros((6, 6))
    w = np.zeros(6)
    zero = np.zeros(6)
    step = np.zeros(6)
    rhs = np.zeros(6)
    trial = np.zeros(6)
    for a in range(6):
        delta[a] = 0.0
    contact_eval(cs, delta, zero, k_reg, l_c, release, f_out, st_out, R, J)
    w_ref = np.zeros(6)
    for a in range(6):
        w_ref[a] = -R[a]
    total_iter = 0
    res = 0.0
    scale = 1.0
    for inc in range(1, n_inc + 1):
        s = inc / n_inc
        for a in range(6):
            w[a] = w_ref[a] + s * (w_ext[a] - w_ref[a])
        scale = contact_eval(cs, delta, w, k_reg, l_c, release, f_out, st_out, R, J)
        res = scaled_norm(R, scale, l_c)
        converged = res <= tol
        it = 0
        while not converged and it < max_iter:
            it += 1
            total_iter += 1
            for a in range(6):
                rhs[a] = -R[a]
            if solve6(J, rhs, step) != 0:
                info[INFO_STATUS] = STATUS_NOT_CONVERGED
                info[INFO_ITER] = total_iter
                info[INFO_RESIDUAL] = res
                info[INFO_SCALE_N] = scale
                return STATUS_NOT_CONVERGED
            su = math.sqrt(step[0] ** 2 + step[1] ** 2 + step[2] ** 2)
            sp = math.sqrt(step[3] ** 2 + step[4] ** 2 + step[5] ** 2)
            if su > RUNAWAY * max_disp or sp > RUNAWAY * max_rot:
                # the tangent predicts a displacement far beyond the limits: a mechanism (no line search)
                for a in range(6):
                    delta[a] += step[a]
                info[INFO_STATUS] = STATUS_MOVES
                info[INFO_ITER] = total_iter
                info[INFO_RESIDUAL] = res
                info[INFO_SCALE_N] = scale
                return STATUS_MOVES
            alpha = 1.0
            rt = res
            for ls in range(LINE_SEARCH_STEPS):
                for a in range(6):
                    trial[a] = delta[a] + alpha * step[a]
                scale = contact_eval(cs, trial, w, k_reg, l_c, release, f_out, st_out, R, J)
                rt = scaled_norm(R, scale, l_c)
                if rt <= (1.0 - LINE_SEARCH_DECREASE * alpha) * res:
                    break
                alpha *= 0.5
            for a in range(6):
                delta[a] = trial[a]
            res = rt
            u = math.sqrt(delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2)
            ph = math.sqrt(delta[3] ** 2 + delta[4] ** 2 + delta[5] ** 2)
            if u > max_disp or ph > max_rot:
                info[INFO_STATUS] = STATUS_MOVES
                info[INFO_ITER] = total_iter
                info[INFO_RESIDUAL] = res
                info[INFO_SCALE_N] = scale
                return STATUS_MOVES
            converged = res <= tol
        if not converged:
            info[INFO_STATUS] = STATUS_NOT_CONVERGED
            info[INFO_ITER] = total_iter
            info[INFO_RESIDUAL] = res
            info[INFO_SCALE_N] = scale
            return STATUS_NOT_CONVERGED
    info[INFO_STATUS] = STATUS_HELD
    info[INFO_ITER] = total_iter
    info[INFO_RESIDUAL] = res
    info[INFO_SCALE_N] = scale
    return STATUS_HELD


@njit(I4(CONTACTS_T, F1, F1, F1, F2, I1, F1), cache=True, nogil=True, error_model="numpy")
def solve_equilibrium(cs, w_ext, cfg, delta, f_out, st_out, info):
    """Displacement ``delta`` (from zero) at which the contacts balance ``w_ext``: the configured number of
    load increments, and when Newton does not converge once more with RETRY_FACTOR times as many (a Newton
    path that left the basin at a kink of the contact law; a displacement beyond the limits is a mechanism and
    is final). Returns STATUS_HELD, STATUS_NOT_CONVERGED or STATUS_MOVES and writes forces, states and info."""
    n_inc = int(cfg[CFG_N_INC])
    if n_inc < 1:
        n_inc = 1
    status = newton_path(cs, w_ext, cfg, n_inc, delta, f_out, st_out, info)
    if status == STATUS_NOT_CONVERGED:
        status = newton_path(cs, w_ext, cfg, RETRY_FACTOR * n_inc, delta, f_out, st_out, info)
    return status


@njit(I4(CONTACTS_T, F1, F1, F1, F1), cache=True, nogil=True, error_model="numpy")
def mechanism(cs, delta, wv, cfg, m_out):
    """Incipient motion at a held state close to the capacity: the tangent with the critical contacts
    released (sticking with friction utilisation >= 1 - eps: slipping; normal force <= eps of the largest:
    open), m = -J*^-1 wv. Returns MODE_FRICTION when a frictional contact slips or is released by slip,
    else MODE_SEPARATION."""
    n = cs.r_m.shape[0]
    eps = cfg[CFG_UTIL_EPS]
    k_reg = cfg[CFG_K_REG]
    l_c = cfg[CFG_L_C]
    release = np.zeros(n, np.int32)
    f = np.zeros((n, 3))
    st = np.zeros(n, np.int32)
    R = np.zeros(6)
    J = np.zeros((6, 6))
    zero = np.zeros(6)
    contact_eval(cs, delta, zero, k_reg, l_c, release, f, st, R, J)
    fn_max = 0.0
    fn = np.zeros(n)
    for i in range(n):
        fn[i] = f[i, 0] * cs.n[i, 0] + f[i, 1] * cs.n[i, 1] + f[i, 2] * cs.n[i, 2]
        if st[i] == ST_STICK and fn[i] > fn_max:
            fn_max = fn[i]
    mode = MODE_SEPARATION
    for i in range(n):
        if cs.kind[i] == KIND_BONDED or cs.kind[i] == KIND_OFF:
            continue
        if st[i] == ST_SLIP:
            mode = MODE_FRICTION
            continue
        if st[i] != ST_STICK:
            continue
        if fn[i] <= eps * fn_max:
            release[i] = 2
            continue
        mu = cs.mu[i]
        if mu <= 0.0:
            continue
        t1 = f[i, 0] * cs.t1[i, 0] + f[i, 1] * cs.t1[i, 1] + f[i, 2] * cs.t1[i, 2]
        t2 = f[i, 0] * cs.t2[i, 0] + f[i, 1] * cs.t2[i, 1] + f[i, 2] * cs.t2[i, 2]
        util = math.sqrt(t1 * t1 + t2 * t2) / (mu * fn[i])
        if util >= 1.0 - eps:
            release[i] = 1
            mode = MODE_FRICTION
    # the released tangent is singular along the mechanism up to k_reg; a regularisation MECH_REG times the
    # stiffest contact (still negligible against every real stiffness) keeps the solve regular and returns
    # the null direction in the metric of the regularisation
    k_max = 0.0
    for i in range(n):
        if cs.kn_N_per_m[i] > k_max:
            k_max = cs.kn_N_per_m[i]
    k_mech = MECH_REG * k_max
    if k_mech < k_reg:
        k_mech = k_reg
    contact_eval(cs, delta, zero, k_mech, l_c, release, f, st, R, J)
    rhs = np.zeros(6)
    for a in range(6):
        rhs[a] = -wv[a]
    if solve6(J, rhs, m_out) != 0:
        for a in range(6):
            m_out[a] = 0.0
    return mode


@njit(F8(CONTACTS_T, F1, F1, F1, F1, F2, I1, F1, F1, F1), cache=True, nogil=True, error_model="numpy")
def capacity(cs, w0, wv, cfg, delta, f_out, st_out, info, m_out, mode_out):
    """Largest load factor lambda in [0, lam_max] with the part held under w0 + lambda wv (bracket by doubling
    from 1, bisection to the relative tolerance); leaves delta, forces and states at that factor and the
    incipient motion there in m_out (zero when the cap is reached), mode_out[0] = MODE_*."""
    lam_max = cfg[CFG_LAM_MAX]
    lam_tol = cfg[CFG_LAM_TOL]
    w = np.zeros(6)
    for a in range(6):
        m_out[a] = 0.0
    mode_out[0] = MODE_FRICTION
    lo = 0.0
    hi = 1.0
    for a in range(6):
        w[a] = w0[a] + hi * wv[a]
    if solve_equilibrium(cs, w, cfg, delta, f_out, st_out, info) == STATUS_HELD:
        lo = 1.0
        hi = 2.0
        while True:
            for a in range(6):
                w[a] = w0[a] + hi * wv[a]
            if solve_equilibrium(cs, w, cfg, delta, f_out, st_out, info) != STATUS_HELD:
                break
            lo = hi
            if hi >= lam_max:
                return lam_max
            hi = hi * 2.0
            if hi > lam_max:
                hi = lam_max
    else:
        for a in range(6):
            w[a] = w0[a]
        if solve_equilibrium(cs, w, cfg, delta, f_out, st_out, info) != STATUS_HELD:
            # not held under the base load alone: the incipient motion from the reference configuration
            for a in range(6):
                delta[a] = 0.0
            mode_out[0] = float(mechanism(cs, delta, w0, cfg, m_out))
            return 0.0
    while hi - lo > (lam_tol if lo < COARSE_MARGIN else max(lam_tol, COARSE_TOL)) * hi:
        mid = 0.5 * (lo + hi)
        for a in range(6):
            w[a] = w0[a] + mid * wv[a]
        if solve_equilibrium(cs, w, cfg, delta, f_out, st_out, info) == STATUS_HELD:
            lo = mid
        else:
            hi = mid
    for a in range(6):
        w[a] = w0[a] + lo * wv[a]
    solve_equilibrium(cs, w, cfg, delta, f_out, st_out, info)
    mode_out[0] = float(mechanism(cs, delta, wv, cfg, m_out))
    return lo


# --------------------------------------------------------------------------- slip integration
@njit(I8(F8, F8, I8, I8), cache=True, nogil=True, error_model="numpy")
def sub_steps(rate_rad_s, dt_s, per_tooth, max_sub):
    """Sub-steps of one tick: per_tooth per tooth period (2 pi / rate), at least 4, at most max_sub."""
    if rate_rad_s <= 0.0:
        return 4
    periods = rate_rad_s * dt_s / (2.0 * math.pi)
    n = int(math.ceil(per_tooth * periods))
    if n < 4:
        n = 4
    if n > max_sub:
        n = max_sub
    return n


@njit(F8(F8, F8), cache=True, nogil=True, error_model="numpy")
def pulse_integral(phi, alpha):
    """Integral of max(0, cos u - cos alpha) over u in [-pi, phi] (any real phi: the periodic continuation),
    alpha in (0, pi]."""
    c = math.cos(alpha)
    full = 2.0 * (math.sin(alpha) - alpha * c)
    k = math.floor((phi + math.pi) / (2.0 * math.pi))
    x = phi - 2.0 * math.pi * k
    if x < -alpha:
        part = 0.0
    elif x <= alpha:
        part = math.sin(x) + math.sin(alpha) - (x + alpha) * c
    else:
        part = full
    return k * full + part


@njit(F8(F8, F8, F8, F8, F8), cache=True, nogil=True, error_model="numpy")
def shape_mean(phi_a, phi_b, alpha, mean, amp):
    """Mean of mean + amp g(u; alpha) over u in [phi_a, phi_b] (abi.Wave: g = (s - <s>) / (1 - <s>), s the pulse
    of half-width alpha; alpha = pi gives g = cos u); the value at phi_a for an empty interval."""
    a = alpha
    if a > math.pi:
        a = math.pi
    if a < 1e-9:
        a = 1e-9
    c = math.cos(a)
    one_c = 1.0 - c
    ms = (math.sin(a) - a * c) / (math.pi * one_c)
    d = phi_b - phi_a
    if abs(d) > 1e-12:
        s_bar = (pulse_integral(phi_b, a) - pulse_integral(phi_a, a)) / (d * one_c)
    else:
        cu = math.cos(phi_a)
        s_bar = (cu - c) / one_c if cu > c else 0.0
    return mean + amp * (s_bar - ms) / (1.0 - ms)


@njit(types.void(WAVE_T, F8, F8, F8, I4, I8, I8, F1, F1), cache=True, nogil=True, error_model="numpy")
def ratchet_part(wave, dt_tick, q_cap, mass, mode, per_tooth, max_sub, state, out):
    """Generalized motion q of the part along its mechanism (mass ``mass``) under the demand of the wave (abi.Wave,
    component 0; its mean over each sub-step) against the resistance q_cap.

    MODE_FRICTION: Coulomb resistance, motion only forward (q grows), sticks when the velocity returns to
    zero with Q <= q_cap; the speed stays at most wave.v_max (the cut cannot push the part faster: the demand
    above q_cap is then not transmitted). MODE_SEPARATION: constant restoring force q_cap and a seat at q = 0
    (inelastic landing at its exact instant): lift-off and tipping that come back. Constant acceleration within a
    sub-step, exact stop and landing instants. state = [q, v] (persistent), out = RP_* of this call: RP_IN_J the
    demand's (transmitted) work, RP_LOSS_J friction work or landing losses; RP_IN_J = RP_LOSS_J + the change of
    m v^2 / 2 (+ q_cap q when lifting) exactly."""
    q = state[0]
    v = state[1]
    q_start = q
    q_max = q
    work = 0.0
    moving = 0.0
    w_in = 0.0
    loss = 0.0
    n_ticks = wave.mean.shape[0]
    for k in range(n_ticks):
        rate = wave.rate_rad_s[k]
        amp = wave.amp[k, 0]
        mean = wave.mean[k, 0]
        alpha = wave.alpha_rad[k, 0]
        vmax = wave.v_max[k, 0]
        n_sub = sub_steps(rate if amp != 0.0 else 0.0, dt_tick, per_tooth, max_sub)
        h = dt_tick / n_sub
        for j in range(n_sub):
            demand = mean
            if amp != 0.0:
                pa = wave.phase0_rad[k] + rate * j * h
                demand = shape_mean(pa, pa + rate * h, alpha, mean, amp)
            a = (demand - q_cap) / mass
            q_old = q
            if mode == MODE_FRICTION:
                if v <= 0.0:
                    v = 0.0
                    if a > 0.0:
                        t1 = h if a * h <= vmax else vmax / a
                        dq1 = 0.5 * a * t1 * t1
                        dq2 = vmax * (h - t1) if t1 < h else 0.0
                        q += dq1 + dq2
                        v = a * t1
                        w_in += demand * dq1 + q_cap * dq2
                        moving += h
                else:
                    v_new = v + a * h
                    if v_new <= 0.0:
                        t_stop = -v / a
                        dq = v * t_stop + 0.5 * a * t_stop * t_stop
                        q += dq
                        v = 0.0
                        moving += t_stop
                        w_in += demand * dq
                    elif v_new > vmax:
                        if v < vmax:
                            # accelerates to the largest speed the demand can drive, then keeps it
                            t1 = (vmax - v) / a
                            dq1 = v * t1 + 0.5 * a * t1 * t1
                            dq2 = vmax * (h - t1)
                            q += dq1 + dq2
                            w_in += demand * dq1 + q_cap * dq2
                        else:
                            # at or above that speed (it fell: the tool slowed): the constraint brings the part to
                            # it within the sub-step, its reaction rr is the demand not transmitted
                            dq = 0.5 * (v + vmax) * h
                            rr = demand - q_cap - mass * (vmax - v) / h
                            q += dq
                            w_in += (demand - rr) * dq
                        v = vmax
                        moving += h
                    else:
                        dq = v * h + 0.5 * a * h * h
                        q += dq
                        v = v_new
                        moving += h
                        w_in += demand * dq
                work += q_cap * (q - q_old)
                loss += q_cap * (q - q_old)
            else:
                if q <= 0.0 and v <= 0.0:
                    q = 0.0
                    v = 0.0
                    if a > 0.0:
                        q = 0.5 * a * h * h
                        v = a * h
                        moving += h
                        w_in += demand * q
                else:
                    q_new = q + v * h + 0.5 * a * h * h
                    v_new = v + a * h
                    moving += h
                    if q_new <= 0.0:
                        # lands within the sub-step: the first root of q + v t + a t^2 / 2 = 0
                        if a != 0.0:
                            disc = math.sqrt(max(v * v - 2.0 * a * q, 0.0))
                            t1 = (-v - disc) / a
                            t2 = (-v + disc) / a
                            t_l = h
                            if 0.0 <= t1 <= h:
                                t_l = t1
                            if 0.0 <= t2 <= h and t2 < t_l:
                                t_l = t2
                        else:
                            t_l = -q / v if v < 0.0 else h
                        v_l = v + a * t_l
                        w_in += demand * (-q)
                        loss += 0.5 * mass * v_l * v_l
                        q = 0.0
                        v = 0.0
                    else:
                        w_in += demand * (q_new - q)
                        q = q_new
                        v = v_new
                work += q_cap * abs(q - q_old)
            if q > q_max:
                q_max = q
    state[0] = q
    state[1] = v
    out[RP_DQ] = q - q_start
    out[RP_Q_MAX] = q_max
    out[RP_WORK_J] = work
    out[RP_MOVING_S] = moving
    out[RP_IN_J] = w_in
    out[RP_LOSS_J] = loss


@njit(F8(F8, F8, F8, F8, F8), cache=True, nogil=True, error_model="numpy")
def implicit_slip(p1, p2, m1, m2, impulse):
    """lambda of the implicit Coulomb slip v = (M + lambda I)^-1 p with lambda |v| = impulse (= h c), for
    |p| > impulse: the friction impulse along the new velocity. Safeguarded Newton from the equal-mass value,
    bracketed in [0, hi] (g(0) < 0 < g(hi)), LAMBDA_ITER iterations at most."""
    pm = math.sqrt(p1 * p1 + p2 * p2)
    m_bar = (m1 * p1 * p1 + m2 * p2 * p2) / (pm * pm)
    lam = impulse * m_bar / (pm - impulse)
    lo = 0.0
    hi = lam
    while True:
        v1 = p1 / (m1 + hi)
        v2 = p2 / (m2 + hi)
        if hi * math.sqrt(v1 * v1 + v2 * v2) > impulse:
            break
        hi = 2.0 * hi + m1 + m2
    for it in range(LAMBDA_ITER):
        a1 = m1 + lam
        a2 = m2 + lam
        v1 = p1 / a1
        v2 = p2 / a2
        vn = math.sqrt(v1 * v1 + v2 * v2)
        g = lam * vn - impulse
        if abs(g) <= 1e-15 * impulse:
            break
        if g > 0.0:
            hi = lam
        else:
            lo = lam
        dvn = -(p1 * p1 / (a1 * a1 * a1) + p2 * p2 / (a2 * a2 * a2)) / vn
        gp = vn + lam * dvn
        nxt = lam - g / gp if gp != 0.0 else 0.5 * (lo + hi)
        if not (lo < nxt < hi):
            nxt = 0.5 * (lo + hi)
        lam = nxt
    return lam


@njit(types.void(WAVE_T, F8, F8, F8, F8, F8, I4, F8, I8, I8, F1, F1), cache=True, nogil=True,
      error_model="numpy")
def ratchet_tool(wave, dt_tick, c_s, c_k, m1, m2, stop, x2_limit, per_tooth, max_sub, state, out):
    """Slip of a tool in a friction holder: x1 circumferential sliding at the shank radius (> 0: the tool lags
    the spindle), x2 axial walk out of the holder (> 0: the tool gets longer), equivalent masses m1 = J / r^2
    and m2 = m. The demand D = (torque / r, axial pull) of every tick is the wave's (abi.Wave; its mean over each
    sub-step); isotropic Coulomb friction with static capacity c_s and kinetic c_k (a circle in (D1, D2)).

    Each sub-step is implicit in the friction (backward Euler on the momentum p = M v + h D): the tool sticks
    when |p| <= h c (static when it was at rest, kinetic when it was sliding), else it slides with the friction
    impulse h c_k along its new velocity, v = (M + lambda I)^-1 p (:func:`implicit_slip`); positions by the
    trapezoidal rule (exact for a constant force), the stopping instant of a slide that sticks by the
    deceleration along its velocity (what is left of the velocity then is dissipated by the sticking contact).
    The slip speed v1 stays at most wave.v_max[:, 0] (the spindle speed at the shank radius): the cutting torque
    resists the tool's turning against the work and cannot turn it backwards, so a tool it overloads stalls (its
    excess demand is then not transmitted). ``stop`` = 1: a back stop at x2 = 0 takes inward axial demand;
    x2_limit: a positive stop on the walk (side-lock flats). state = [x1, x2, v1, v2] (persistent), out = RT_* of
    this call: RT_IN_J the demand's transmitted work, RT_WORK_J the dissipation; RT_IN_J = RT_WORK_J + the
    change of (m1 v1^2 + m2 v2^2) / 2 exactly (the stops' reactions excepted)."""
    x1 = state[0]
    x2 = state[1]
    v1 = state[2]
    v2 = state[3]
    x1_start = x1
    x2_start = x2
    work = 0.0
    moving = 0.0
    peak = 0.0
    w_in = 0.0
    n_ticks = wave.mean.shape[0]
    for k in range(n_ticks):
        rate = wave.rate_rad_s[k]
        has_amp = wave.amp[k, 0] != 0.0 or wave.amp[k, 1] != 0.0
        n_sub = sub_steps(rate if has_amp else 0.0, dt_tick, per_tooth, max_sub)
        h = dt_tick / n_sub
        vmax1 = wave.v_max[k, 0]
        for j in range(n_sub):
            d1 = wave.mean[k, 0]
            d2 = wave.mean[k, 1]
            if has_amp:
                pa = wave.phase0_rad[k] + rate * j * h
                pb = pa + rate * h
                d1 = shape_mean(pa, pb, wave.alpha_rad[k, 0], wave.mean[k, 0], wave.amp[k, 0])
                d2 = shape_mean(pa, pb, wave.alpha_rad[k, 1], wave.mean[k, 1], wave.amp[k, 1])
            if stop == 1 and x2 <= 0.0 and v2 <= 0.0 and d2 < 0.0:
                d2 = 0.0
            if x2 >= x2_limit and v2 >= 0.0 and d2 > 0.0:
                d2 = 0.0
            dm = math.sqrt(d1 * d1 + d2 * d2)
            if c_s > 0.0 and dm / c_s > peak:
                peak = dm / c_s
            speed = math.sqrt(v1 * v1 + v2 * v2)
            p1 = m1 * v1 + h * d1
            p2 = m2 * v2 + h * d2
            pm = math.sqrt(p1 * p1 + p2 * p2)
            cap = c_s if speed <= 0.0 else c_k
            if pm <= h * cap:
                if speed > 0.0:
                    e1 = v1 / speed
                    e2 = v2 / speed
                    a1 = (d1 - c_k * e1) / m1
                    a2 = (d2 - c_k * e2) / m2
                    along = a1 * e1 + a2 * e2
                    t_s = speed / (-along) if along < 0.0 else h
                    if t_s > h:
                        t_s = h
                    s1 = v1 * t_s + 0.5 * a1 * t_s * t_s
                    s2 = v2 * t_s + 0.5 * a2 * t_s * t_s
                    x1 += s1
                    x2 += s2
                    r1 = v1 + a1 * t_s
                    r2 = v2 + a2 * t_s
                    w_in += d1 * s1 + d2 * s2
                    work += c_k * (e1 * s1 + e2 * s2) + 0.5 * (m1 * r1 * r1 + m2 * r2 * r2)
                    moving += t_s
                v1 = 0.0
                v2 = 0.0
            else:
                lam = implicit_slip(p1, p2, m1, m2, h * c_k)
                n1 = p1 / (m1 + lam)
                n2 = p2 / (m2 + lam)
                vn = math.sqrt(n1 * n1 + n2 * n2)
                e1 = n1 / vn
                e2 = n2 / vn
                r1 = 0.0
                if n1 > vmax1:
                    # stalled: the excess of the demand is not transmitted (reaction r1 per unit time)
                    r1 = m1 * (n1 - vmax1) / h
                    n1 = vmax1
                s1 = 0.5 * (v1 + n1) * h
                s2 = 0.5 * (v2 + n2) * h
                x1 += s1
                x2 += s2
                v1 = n1
                v2 = n2
                w_in += (d1 - r1) * s1 + d2 * s2
                work += c_k * (e1 * s1 + e2 * s2)
                moving += h
            # the stops: a position correction dlt counts the demand's work over it on both sides (it does not
            # happen), the speed a stop takes is dissipated in it
            if stop == 1 and x2 < 0.0:
                dlt = -x2
                x2 = 0.0
                w_in += d2 * dlt
                work += d2 * dlt
                if v2 < 0.0:
                    work += 0.5 * m2 * v2 * v2
                    v2 = 0.0
            if x2 > x2_limit:
                dlt = x2_limit - x2
                x2 = x2_limit
                w_in += d2 * dlt
                work += d2 * dlt
                if v2 > 0.0:
                    work += 0.5 * m2 * v2 * v2
                    v2 = 0.0
    state[0] = x1
    state[1] = x2
    state[2] = v1
    state[3] = v2
    out[RT_DX1] = x1 - x1_start
    out[RT_DX2] = x2 - x2_start
    out[RT_WORK_J] = work
    out[RT_MOVING_S] = moving
    out[RT_PEAK_RATIO] = peak
    out[RT_IN_J] = w_in


KERNELS = (solve6, scaled_norm, contact_eval, newton_path, solve_equilibrium, mechanism, capacity, sub_steps,
           pulse_integral, shape_mean, ratchet_part, implicit_slip, ratchet_tool)

__all__ = ["solve6", "scaled_norm", "contact_eval", "newton_path", "solve_equilibrium", "mechanism", "capacity", "sub_steps",
           "pulse_integral", "shape_mean", "ratchet_part", "implicit_slip", "ratchet_tool", "KERNELS", "CONTACTS_T", "WAVE_T", "example_contacts", "example_wave"]
