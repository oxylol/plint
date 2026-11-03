"""Plain-Python references of the holding kernels (architecture 5.5: every kernel has one; the tests
compare them). Written with NumPy vector algebra, independently of the loop formulation in kernels.py."""

from __future__ import annotations

import math

import numpy as np

from .abi import (CFG_K_REG, CFG_L_C, CFG_LAM_MAX, CFG_LAM_TOL, CFG_MAX_DISP, CFG_MAX_ITER, CFG_MAX_ROT, CFG_N_INC,
                  CFG_TOL, CFG_UTIL_EPS, KIND_BONDED, KIND_OFF,
                  MODE_FRICTION, MODE_SEPARATION, RP_DQ, RP_MOVING_S, RP_Q_MAX, RP_WORK_J, RT_DX1, RT_DX2, RT_MOVING_S,
                  RT_PEAK_RATIO, RT_WORK_J, ST_BONDED, ST_OFF, ST_OPEN, ST_SLIP, ST_STICK, STATUS_HELD, STATUS_MOVES,
                  STATUS_NOT_CONVERGED)

LINE_SEARCH_STEPS = 10
RETRY_FACTOR = 8
RUNAWAY = 100.0
MECH_REG = 1e-8
COARSE_MARGIN = 2.0
COARSE_TOL = 0.1
LINE_SEARCH_DECREASE = 1e-4
PIVOT_EPS = 1e-14


def skew(r) -> np.ndarray:
    return np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])


def solve6_ref(A, b):
    """(x, singular) by partial pivoting (the kernel's algorithm)."""
    M = np.array(A, dtype=float)
    y = np.array(b, dtype=float)
    scale = float(np.max(np.abs(M))) if M.size else 0.0
    if scale == 0.0:
        return np.zeros(6), True
    for c in range(6):
        p = c + int(np.argmax(np.abs(M[c:, c])))
        if abs(M[p, c]) <= PIVOT_EPS * scale:
            return np.zeros(6), True
        if p != c:
            M[[c, p]] = M[[p, c]]
            y[[c, p]] = y[[p, c]]
        for r in range(c + 1, 6):
            f = M[r, c] / M[c, c]
            if f != 0.0:
                M[r, c:] -= f * M[c, c:]
                y[r] -= f * y[c]
    x = np.zeros(6)
    for i in range(5, -1, -1):
        x[i] = (y[i] - M[i, i + 1:] @ x[i + 1:]) / M[i, i]
    return x, False


def scaled_norm_ref(R, f_scale, l_c) -> float:
    return max(float(np.linalg.norm(R[:3])) / f_scale, float(np.linalg.norm(R[3:])) / (f_scale * l_c))


def contact_eval_ref(cs, delta, w_ext, k_reg, l_c, release=None):
    """(f [n, 3], states [n], R [6], J [6, 6], f_scale) at the twist delta (see kernels.contact_eval)."""
    n = len(cs.kind)
    release = np.zeros(n, np.int32) if release is None else release
    delta = np.asarray(delta, float)
    u, phi = delta[:3], delta[3:]
    R = np.array(w_ext, dtype=float)
    reg = np.array([k_reg] * 3 + [k_reg * l_c * l_c] * 3)
    R -= reg * delta
    J = -np.diag(reg)
    f_scale = float(np.linalg.norm(w_ext[:3]) + np.linalg.norm(w_ext[3:]) / l_c)
    f = np.zeros((n, 3))
    st = np.zeros(n, np.int32)
    for i in range(n):
        if cs.kind[i] == KIND_OFF:
            st[i] = ST_OFF
            continue
        r = cs.r_m[i]
        nv, a, b = cs.n[i], cs.t1[i], cs.t2[i]
        kn, k1, k2, mu = cs.kn_N_per_m[i], cs.kt_N_per_m[i, 0], cs.kt_N_per_m[i, 1], cs.mu[i]
        f_scale += float(np.sum(np.abs(cs.pre_N[i])) + kn * abs(cs.gap_m[i]))
        d = u + np.cross(phi, r)
        fn = cs.pre_N[i, 0] + kn * (cs.gap_m[i] - nv @ d)
        bonded = cs.kind[i] == KIND_BONDED
        if not bonded and (fn < 0.0 or release[i] == 2):
            st[i] = ST_OPEN
            continue
        ft = np.array([cs.pre_N[i, 1] - k1 * (a @ d), cs.pre_N[i, 2] - k2 * (b @ d)])
        K = -kn * np.outer(nv, nv)
        T = np.stack([a, b], axis=1)                       # 3 x 2
        dtrial = np.stack([-k1 * a, -k2 * b], axis=0)      # 2 x 3
        state = ST_BONDED if bonded else ST_STICK
        if not bonded and mu <= 0.0:
            ft = np.zeros(2)
        else:
            mag = float(np.hypot(ft[0], ft[1]))
            cap = mu * fn
            slip = (not bonded) and (mag > cap or (release[i] == 1 and mag > 0.0))
            if not slip:
                K = K + T @ dtrial
            else:
                state = ST_SLIP
                s = ft / mag
                ratio = min(cap / mag, 1.0)
                ft = ratio * mag * s
                dfn = -kn * nv
                dft = mu * np.outer(s, dfn) + ratio * (np.eye(2) - np.outer(s, s)) @ dtrial
                K = K + T @ dft
        fv = fn * nv + T @ ft
        f[i] = fv
        st[i] = state
        R[:3] += fv
        R[3:] += np.cross(r, fv)
        Gd = np.hstack([K, -K @ skew(r)])                # dF/d(delta)
        J[:3] += Gd
        J[3:] += skew(r) @ Gd
    return f, st, R, J, (f_scale if f_scale > 0.0 else 1.0)


def newton_path_ref(cs, w_ext, cfg, n_inc):
    """(status, delta, f, states, info) - kernels.newton_path."""
    tol = cfg[CFG_TOL]
    max_iter = int(cfg[CFG_MAX_ITER])
    k_reg, l_c = cfg[CFG_K_REG], cfg[CFG_L_C]
    delta = np.zeros(6)
    info = np.zeros(4)
    f, st, R, J, _ = contact_eval_ref(cs, delta, np.zeros(6), k_reg, l_c)
    w_ref = -R
    total = 0
    res = 0.0
    scale = 1.0
    for inc in range(1, n_inc + 1):
        w = w_ref + inc / n_inc * (np.asarray(w_ext, float) - w_ref)
        f, st, R, J, scale = contact_eval_ref(cs, delta, w, k_reg, l_c)
        res = scaled_norm_ref(R, scale, l_c)
        it = 0
        while res > tol and it < max_iter:
            it += 1
            total += 1
            step, singular = solve6_ref(J, -R)
            if singular:
                info[:] = (STATUS_NOT_CONVERGED, total, res, scale)
                return STATUS_NOT_CONVERGED, delta, f, st, info
            if np.linalg.norm(step[:3]) > RUNAWAY * cfg[CFG_MAX_DISP] or \
                    np.linalg.norm(step[3:]) > RUNAWAY * cfg[CFG_MAX_ROT]:
                info[:] = (STATUS_MOVES, total, res, scale)
                return STATUS_MOVES, delta + step, f, st, info
            alpha = 1.0
            for _ in range(LINE_SEARCH_STEPS):
                trial = delta + alpha * step
                f, st, R, J, scale = contact_eval_ref(cs, trial, w, k_reg, l_c)
                rt = scaled_norm_ref(R, scale, l_c)
                if rt <= (1.0 - LINE_SEARCH_DECREASE * alpha) * res:
                    break
                alpha *= 0.5
            delta = trial
            res = rt
            if np.linalg.norm(delta[:3]) > cfg[CFG_MAX_DISP] or np.linalg.norm(delta[3:]) > cfg[CFG_MAX_ROT]:
                info[:] = (STATUS_MOVES, total, res, scale)
                return STATUS_MOVES, delta, f, st, info
        if res > tol:
            info[:] = (STATUS_NOT_CONVERGED, total, res, scale)
            return STATUS_NOT_CONVERGED, delta, f, st, info
    info[:] = (STATUS_HELD, total, res, scale)
    return STATUS_HELD, delta, f, st, info


def solve_equilibrium_ref(cs, w_ext, cfg):
    """(status, delta, f, states, info) - kernels.solve_equilibrium (retry with RETRY_FACTOR x increments
    when Newton does not converge)."""
    n_inc = max(int(cfg[CFG_N_INC]), 1)
    out = newton_path_ref(cs, w_ext, cfg, n_inc)
    if out[0] == STATUS_NOT_CONVERGED:
        out = newton_path_ref(cs, w_ext, cfg, RETRY_FACTOR * n_inc)
    return out


def mechanism_ref(cs, delta, wv, cfg):
    """(m [6], mode) - kernels.mechanism."""
    n = len(cs.kind)
    eps, k_reg, l_c = cfg[CFG_UTIL_EPS], cfg[CFG_K_REG], cfg[CFG_L_C]
    f, st, _, _, _ = contact_eval_ref(cs, delta, np.zeros(6), k_reg, l_c)
    fn = np.einsum("ij,ij->i", f, cs.n)
    sticking = st == ST_STICK
    fn_max = float(fn[sticking].max()) if np.any(sticking) else 0.0
    release = np.zeros(n, np.int32)
    mode = MODE_SEPARATION
    for i in range(n):
        if cs.kind[i] in (KIND_BONDED, KIND_OFF):
            continue
        if st[i] == ST_SLIP:
            mode = MODE_FRICTION
            continue
        if st[i] != ST_STICK:
            continue
        if fn[i] <= eps * fn_max:
            release[i] = 2
            continue
        if cs.mu[i] <= 0.0:
            continue
        util = math.hypot(f[i] @ cs.t1[i], f[i] @ cs.t2[i]) / (cs.mu[i] * fn[i])
        if util >= 1.0 - eps:
            release[i] = 1
            mode = MODE_FRICTION
    k_mech = max(MECH_REG * float(np.max(cs.kn_N_per_m)) if len(cs.kind) else 0.0, k_reg)
    _, _, _, J, _ = contact_eval_ref(cs, delta, np.zeros(6), k_mech, l_c, release)
    m, singular = solve6_ref(J, -np.asarray(wv, float))
    return (np.zeros(6) if singular else m), mode


def capacity_ref(cs, w0, wv, cfg):
    """(lambda, delta, f, states, m, mode) - kernels.capacity."""
    w0 = np.asarray(w0, float)
    wv = np.asarray(wv, float)
    lam_max, lam_tol = cfg[CFG_LAM_MAX], cfg[CFG_LAM_TOL]

    def held(lam):
        return solve_equilibrium_ref(cs, w0 + lam * wv, cfg)

    status, delta, f, st, _ = held(1.0)
    if status == STATUS_HELD:
        lo, hi = 1.0, 2.0
        while True:
            status, delta, f, st, _ = held(hi)
            if status != STATUS_HELD:
                break
            lo = hi
            if hi >= lam_max:
                return lam_max, delta, f, st, np.zeros(6), MODE_FRICTION
            hi = min(hi * 2.0, lam_max)
    else:
        status, delta, f, st, _ = held(0.0)
        if status != STATUS_HELD:
            delta = np.zeros(6)
            m, mode = mechanism_ref(cs, delta, w0, cfg)
            return 0.0, delta, f, st, m, mode
        lo, hi = 0.0, 1.0
    while hi - lo > (lam_tol if lo < COARSE_MARGIN else max(lam_tol, COARSE_TOL)) * hi:
        mid = 0.5 * (lo + hi)
        if held(mid)[0] == STATUS_HELD:
            lo = mid
        else:
            hi = mid
    _, delta, f, st, _ = held(lo)
    m, mode = mechanism_ref(cs, delta, wv, cfg)
    return lo, delta, f, st, m, mode


def sub_steps_ref(rate, dt, per_tooth, max_sub) -> int:
    if rate <= 0.0:
        return 4
    return int(min(max(math.ceil(per_tooth * rate * dt / (2.0 * math.pi)), 4), max_sub))


def ratchet_part_ref(wave, dt_tick, q_cap, mass, mode, per_tooth, max_sub, state):
    """(state, out) - kernels.ratchet_part."""
    q, v = float(state[0]), float(state[1])
    q0 = q
    q_max = q
    work = moving = 0.0
    for k in range(wave.mean.shape[0]):
        amp, mean, rate = wave.amp[k, 0], wave.mean[k, 0], wave.rate_rad_s[k]
        n_sub = sub_steps_ref(rate if amp != 0.0 else 0.0, dt_tick, per_tooth, max_sub)
        h = dt_tick / n_sub
        for j in range(n_sub):
            a = (mean + amp * math.cos(wave.phase0_rad[k] + rate * (j + 0.5) * h) - q_cap) / mass
            q_old = q
            if mode == MODE_FRICTION:
                if v <= 0.0:
                    v = 0.0
                    if a > 0.0:
                        q, v, moving = q + 0.5 * a * h * h, a * h, moving + h
                elif v + a * h <= 0.0:
                    t = -v / a
                    q, v, moving = q + v * t + 0.5 * a * t * t, 0.0, moving + t
                else:
                    q, v, moving = q + v * h + 0.5 * a * h * h, v + a * h, moving + h
                work += q_cap * (q - q_old)
            else:
                if q <= 0.0 and v <= 0.0:
                    q, v = 0.0, 0.0
                    if a > 0.0:
                        q, v, moving = 0.5 * a * h * h, a * h, moving + h
                else:
                    q_new, v_new = q + v * h + 0.5 * a * h * h, v + a * h
                    moving += h
                    q, v = (0.0, 0.0) if q_new <= 0.0 else (q_new, v_new)
                work += q_cap * abs(q - q_old)
            q_max = max(q_max, q)
    out = np.zeros(4)
    out[RP_DQ], out[RP_Q_MAX], out[RP_WORK_J], out[RP_MOVING_S] = q - q0, q_max, work, moving
    return np.array([q, v]), out


def implicit_slip_ref(p, m, impulse) -> float:
    """lambda with lambda |(M + lambda I)^-1 p| = impulse, by bisection to round-off (an independent solver of
    the same equation kernels.implicit_slip solves by safeguarded Newton)."""
    p = np.asarray(p, float)
    m = np.asarray(m, float)

    def g(lam):
        return lam * float(np.linalg.norm(p / (m + lam))) - impulse

    lo, hi = 0.0, 1.0
    while g(hi) <= 0.0:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid in (lo, hi):
            break
        if g(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def ratchet_tool_ref(wave, dt_tick, c_s, c_k, m1, m2, stop, x2_limit, per_tooth, max_sub, state):
    """(state, out) - kernels.ratchet_tool, with position, velocity and momentum as vectors."""
    x = np.array(state[:2], dtype=float)
    v = np.array(state[2:], dtype=float)
    x0 = x.copy()
    m = np.array([m1, m2])
    work = moving = peak = 0.0
    for k in range(wave.mean.shape[0]):
        rate = wave.rate_rad_s[k]
        has_amp = wave.amp[k, 0] != 0.0 or wave.amp[k, 1] != 0.0
        n_sub = sub_steps_ref(rate if has_amp else 0.0, dt_tick, per_tooth, max_sub)
        h = dt_tick / n_sub
        for j in range(n_sub):
            d = np.array(wave.mean[k] + wave.amp[k] * math.cos(wave.phase0_rad[k] + rate * (j + 0.5) * h), float)
            if stop == 1 and x[1] <= 0.0 and v[1] <= 0.0 and d[1] < 0.0:
                d[1] = 0.0
            if x[1] >= x2_limit and v[1] >= 0.0 and d[1] > 0.0:
                d[1] = 0.0
            if c_s > 0.0:
                peak = max(peak, float(np.hypot(d[0], d[1])) / c_s)
            speed = float(np.hypot(v[0], v[1]))
            p = m * v + h * d
            pm = float(np.hypot(p[0], p[1]))
            cap = c_s if speed <= 0.0 else c_k
            if pm <= h * cap:
                if speed > 0.0:
                    e = v / speed
                    a = (d - c_k * e) / m
                    along = float(a @ e)
                    t_s = min(speed / -along, h) if along < 0.0 else h
                    s = v * t_s + 0.5 * a * t_s * t_s
                    x = x + s
                    work += c_k * float(np.hypot(s[0], s[1]))
                    moving += t_s
                v = np.zeros(2)
            else:
                v_new = p / (m + implicit_slip_ref(p, m, h * c_k))
                s = 0.5 * (v + v_new) * h
                x = x + s
                v = v_new
                work += c_k * float(np.hypot(s[0], s[1]))
                moving += h
            if stop == 1 and x[1] < 0.0:
                x[1] = 0.0
                v[1] = max(v[1], 0.0)
            if x[1] > x2_limit:
                x[1] = x2_limit
                v[1] = min(v[1], 0.0)
    out = np.zeros(5)
    out[RT_DX1], out[RT_DX2] = x - x0
    out[RT_WORK_J], out[RT_MOVING_S], out[RT_PEAK_RATIO] = work, moving, peak
    return np.concatenate([x, v]), out


__all__ = ["skew", "solve6_ref", "scaled_norm_ref", "contact_eval_ref", "newton_path_ref", "solve_equilibrium_ref", "mechanism_ref",
           "capacity_ref", "sub_steps_ref", "ratchet_part_ref", "implicit_slip_ref", "ratchet_tool_ref"]
