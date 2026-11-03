"""Plain-Python references of the dynamics kernels (architecture 5.5: every kernel has one and a test compares).

They operate on the same NamedTuples (NumPy arrays) and write the same outputs, written for clarity with
matrix algebra (complex numbers, numpy dot products) instead of the kernels' scalar loops.
"""

from __future__ import annotations

import cmath
import math

import numpy as np

from ..core.abi import ACC_DAMPING_J, ACC_PROCESS_DAMPING_J, ACC_UNBALANCE_WORK_J
from . import discretize as DZ
from .abi import (BASE_LINEAR, DA_DAMPING_J, DA_ENERGY_MAX_J, DA_FORCE_MAX_N, DA_N_FREE, DA_N_L0, DA_UMAX_X, DA_UMIN_X,
                  DA_W_BASE_J, DA_W_CUT_J, DA_W_CUT_QUAD_J, DA_W_UNB_J, FL_CONTACT, FL_HISTORY, FL_STEP_ERROR,
                  FL_STEP_SEEN, KIND_WORK, MI_HAS_TOOL_RES, MI_HAS_WORK_RES, MI_L0_TICKS, MI_REF_STATION, MS_TICK_S,
                  SC_OMEGA)


def free_interval_ref(w, zeta, h, q0, v0, Pr, Pi, W, g0, g1):
    """(q1, v1, W_harmonic, W_linear, D) independently of the kernel's closed forms: the state from the matrix
    exponential of the exosystem-augmented mode, the integrals by Gauss-Legendre quadrature of it."""
    return _free_by_quadrature(w, zeta, h, q0, v0, complex(Pr, Pi), W, g0, g1)


def _free_by_quadrature(w, zeta, h, q0, v0, P, W, g0, g1, pieces: int = 96):
    """Reference for small omega h: the state from the augmented matrix exponential (harmonic forcing as an
    exosystem) and the integrals by 8-point Gauss-Legendre per piece."""
    from scipy.linalg import expm
    A = np.zeros((6, 6))
    A[0, 1] = 1.0
    A[1, 0] = -w * w
    A[1, 1] = -2.0 * zeta * w
    A[1, 2] = 1.0          # c = Re(P e^(iWt)) part
    A[1, 4] = 1.0          # linear part g
    A[2, 3] = -W           # c' = -W s
    A[3, 2] = W            # s' = W c
    A[4, 5] = 1.0          # g' = g1
    y0 = np.array([q0, v0, P.real, P.imag, g0, g1])
    # Re(P e^(iWt)) = Pr cos - Pi sin = c(t) with c(0) = Pr, s(0) = Pi
    nodes, weights = np.polynomial.legendre.leggauss(8)
    hs = h / pieces
    wh = wl = dis = 0.0
    for j in range(pieces):
        for t_n, w_n in zip(nodes, weights):
            t = hs * (j + 0.5 * (t_n + 1.0))
            y = expm(A * t) @ y0
            v = y[1]
            wh += 0.5 * hs * w_n * y[2] * v
            wl += 0.5 * hs * w_n * y[4] * v
            dis += 0.5 * hs * w_n * 2.0 * zeta * w * v * v
    y1 = expm(A * h) @ y0
    return float(y1[0]), float(y1[1]), float(wh), float(wl), float(dis)


def external_force_ref(sys, theta_rad, omega_rad_s, alpha_rad_s2, base_in) -> None:
    dm = sys.dm
    sys.scratch[SC_OMEGA] = abs(omega_rad_s)
    P = dm.unb[:, 0] + 1j * dm.unb[:, 1]
    sys.fe[0, :] = np.real(P * (omega_rad_s ** 2 - 1j * alpha_rad_s2) * cmath.exp(1j * theta_rad))
    sys.fe[1, :] = dm.base_proj @ np.asarray(base_in)


def _fstar(ds) -> np.ndarray:
    return 2.0 * ds.f_cut - ds.f_cut_prev if ds.flags[FL_HISTORY] else ds.f_cut.copy()


def modal_predict_ref(sys) -> None:
    m = sys.model
    ds = sys.ds
    f0 = ds.f_cut + ds.f_unb + ds.f_base
    f1 = _fstar(ds) + sys.fe[0] + sys.fe[1]
    for k in range(len(f0)):
        sys.xp[k] = m.Phi[k] @ sys.state.x[k] + m.G0[k] * f0[k] + m.G1[k] * f1[k]


def _tool_rows(sys):
    m = sys.model
    return [k for k in range(len(sys.dm.row)) if m.is_work[k] != KIND_WORK]


def _work_rows(sys):
    m = sys.model
    return [k for k in range(len(sys.dm.row)) if m.is_work[k] == KIND_WORK]


def _deflection_at(sys, x) -> None:
    m = sys.model
    dm = sys.dm
    ds = sys.ds
    defl = sys.defl
    nst = defl.u_st_T0_m.shape[0]
    u = np.zeros((nst, 6))
    v = np.zeros((nst, 6))
    for k in _tool_rows(sys):
        r = int(dm.row[k])
        shape = m.tool_shape[r, :nst, :]
        u[:shape.shape[0]] += shape * x[k, 0]
        v[:shape.shape[0]] += shape * x[k, 1]
    if dm.iscal[MI_HAS_TOOL_RES]:
        nres = m.static_residual.shape[0] // 3
        lim = min(nres, nst)
        F = ds.st_force[:lim, :3].reshape(-1)
        u[:lim, :3] += (m.static_residual[:3 * lim, :3 * lim] @ F).reshape(lim, 3)
    uw = np.zeros(3)
    vw = np.zeros(3)
    for k in _work_rows(sys):
        r = int(dm.row[k])
        uw += m.work_shape_at_contact[r] * x[k, 0]
        vw += m.work_shape_at_contact[r] * x[k, 1]
    if dm.iscal[MI_HAS_WORK_RES]:
        uw -= dm.res_work_S @ ds.work_force_S
    defl.u_st_T0_m[:] = u
    defl.v_st_T0_m_s[:] = v
    defl.u_work_S_m[:] = uw
    defl.v_work_S_m_s[:] = vw


def modal_deflection_ref(sys) -> None:
    _deflection_at(sys, sys.xp)


def relative_deflection_ref(sys) -> np.ndarray:
    m = sys.model
    dm = sys.dm
    ds = sys.ds
    x = sys.state.x
    ref = int(dm.iscal[MI_REF_STATION])
    tool = np.zeros(3)
    work = np.zeros(3)
    for k in _tool_rows(sys):
        tool += m.tool_shape[int(dm.row[k]), ref, :3] * x[k, 0]
    for k in _work_rows(sys):
        work += m.work_shape_at_contact[int(dm.row[k])] * x[k, 0]
    if dm.iscal[MI_HAS_TOOL_RES]:
        nres = m.static_residual.shape[0] // 3
        tool += m.static_residual[3 * ref:3 * ref + 3, :3 * nres] @ ds.st_force[:nres, :3].reshape(-1)
    if dm.iscal[MI_HAS_WORK_RES]:
        work -= dm.res_work_S @ ds.work_force_S
    rel = tool - dm.R_S_T0.T @ work
    return np.concatenate([tool, work, rel])


def modal_correct_ref(sys, station_T0, force_S_N, tick, acc) -> None:
    m = sys.model
    dm = sys.dm
    ds = sys.ds
    da = sys.da
    x = sys.state.x
    nm = x.shape[0]
    if tick - ds.node_tick[0] != dm.iscal[MI_L0_TICKS]:
        ds.flags[FL_STEP_ERROR] = 1
        ds.flags[FL_STEP_SEEN] = tick - ds.node_tick[0]
    nst = station_T0.shape[0]
    fx = np.zeros(nm)
    for k in range(nm):
        r = int(dm.row[k])
        if m.is_work[k] == KIND_WORK:
            fx[k] = -m.work_shape_at_contact[r] @ force_S_N
        else:
            ns = min(m.tool_shape.shape[1], nst)
            fx[k] = float(np.sum(m.tool_shape[r, :ns, :] * station_T0[:ns, :]))
    dt = float(m.dt_s[0])
    fstar = _fstar(ds)
    miss = dm.relax @ (fx - fstar)
    fcn = fstar + miss
    f0 = ds.f_cut + ds.f_unb + ds.f_base
    f1 = fcn + sys.fe[0] + sys.fe[1]
    dis = w_ext = 0.0
    for k in range(nm):
        z = np.array([x[k, 0], x[k, 1], f0[k], (f1[k] - f0[k]) / dt])
        d_k = float(z @ dm.gram_D[k] @ z)
        nz = dm.gram_W[k] @ z
        w_u = ds.f_unb[k] * nz[0] + (sys.fe[0, k] - ds.f_unb[k]) / dt * nz[1]
        w_b = ds.f_base[k] * nz[0] + (sys.fe[1, k] - ds.f_base[k]) / dt * nz[1]
        da.scal[DA_DAMPING_J] += d_k
        da.scal[DA_W_CUT_J] += ds.f_cut[k] * nz[0] + (fcn[k] - ds.f_cut[k]) / dt * nz[1]
        da.scal[DA_W_UNB_J] += w_u
        da.scal[DA_W_BASE_J] += w_b
        dis += d_k
        w_ext += w_u + w_b
    acc.scalars[ACC_DAMPING_J] += dis
    acc.scalars[ACC_UNBALANCE_WORK_J] += w_ext
    da.scal[DA_W_CUT_QUAD_J] += float(np.sum(fx * sys.xp[:, 1])) * dt
    xn = sys.xp + m.G1 * miss[:, None]
    x[:] = xn
    ds.f_cut_prev[:] = ds.f_cut
    ds.f_cut[:] = fcn
    ds.f_unb[:] = sys.fe[0]
    ds.f_base[:] = sys.fe[1]
    sys.state.f_prev[:] = f1
    ds.flags[FL_HISTORY] = 1
    ds.flags[FL_CONTACT] = 1
    ds.node_tick[0] = tick
    ds.st_force[:] = 0.0
    ds.st_force[:min(nst, ds.st_force.shape[0])] = station_T0[:min(nst, ds.st_force.shape[0])]
    ds.work_force_S[:] = force_S_N
    da.scal[DA_N_L0] += 1.0
    e = DZ.modal_energy(x, dm.omega_rad_s)
    da.scal[DA_ENERGY_MAX_J] = max(da.scal[DA_ENERGY_MAX_J], e)
    da.scal[DA_FORCE_MAX_N] = max(da.scal[DA_FORCE_MAX_N], float(np.linalg.norm(force_S_N)))
    # the vibration sample uses this step's wrenches for the residual parts (the kernel's form)
    rel = relative_deflection_ref(sys)
    _extremes(da, rel[6:9])
    n = ds.samp_n[0]
    if n < len(ds.samp_tick):
        ds.samp_tick[n] = tick
        ds.samp_u[n] = rel[6:9]
        ds.samp_w[n] = sys.scratch[SC_OMEGA]
        ds.samp_n[0] = n + 1
    else:
        ds.samp_n[1] += 1


def _extremes(da, rel3) -> None:
    for c in range(3):
        da.scal[DA_UMIN_X + c] = min(da.scal[DA_UMIN_X + c], rel3[c])
        da.scal[DA_UMAX_X + c] = max(da.scal[DA_UMAX_X + c], rel3[c])


def modal_free_ref(sys, theta0, theta1, omega0, omega1, base0, base1, tick1) -> None:
    dm = sys.dm
    ds = sys.ds
    da = sys.da
    x = sys.state.x
    h = (tick1 - ds.node_tick[0]) * dm.scal[MS_TICK_S]
    if h <= 0.0:
        ds.flags[FL_STEP_ERROR] = 1
        ds.flags[FL_STEP_SEEN] = tick1 - ds.node_tick[0]
        return
    W = (theta1 - theta0) / h
    alpha = (omega1 - omega0) / h
    Pk = (dm.unb[:, 0] + 1j * dm.unb[:, 1]) * (W * W - 1j * alpha) * cmath.exp(1j * theta0)
    g0 = dm.base_proj @ np.asarray(base0)
    gb = dm.base_proj @ np.asarray(base1)
    for k in range(x.shape[0]):
        q1, v1, wh, wl, dis = free_interval_ref(float(dm.omega_rad_s[k]), float(dm.zeta[k]), h, float(x[k, 0]),
                                                float(x[k, 1]), Pk[k].real, Pk[k].imag, W, float(g0[k]),
                                                float((gb[k] - g0[k]) / h))
        x[k] = (q1, v1)
        da.scal[DA_W_UNB_J] += wh
        da.scal[DA_W_BASE_J] += wl
        da.scal[DA_DAMPING_J] += dis
    ds.f_unb[:] = np.real(Pk * cmath.exp(1j * W * h))
    ds.f_base[:] = gb
    ds.f_cut[:] = 0.0
    ds.f_cut_prev[:] = 0.0
    sys.state.f_prev[:] = ds.f_unb + gb
    ds.flags[FL_HISTORY] = 0
    ds.flags[FL_CONTACT] = 0
    ds.node_tick[0] = tick1
    ds.st_force[:] = 0.0
    ds.work_force_S[:] = 0.0
    da.scal[DA_N_FREE] += 1.0
    da.scal[DA_ENERGY_MAX_J] = max(da.scal[DA_ENERGY_MAX_J], DZ.modal_energy(x, dm.omega_rad_s))
    _extremes(da, relative_deflection_ref(sys)[6:9])


def refresh_work_shapes_ref(sys, grid, point_S) -> None:
    m = sys.model
    dm = sys.dm
    cell = float(grid.cell_m[0])
    dims = grid.shape.shape[1:4]
    lo = []
    for a in range(3):
        u = (point_S[a] - grid.origin_S_m[a]) / cell - 0.5 if cell > 0.0 else 0.0
        n = dims[a]
        if n <= 1:
            lo.append((0, 0, 0.0))
            continue
        u = min(max(u, 0.0), n - 1.0)
        i0 = min(int(math.floor(u)), n - 2)
        lo.append((i0, i0 + 1, u - i0))
    nres = m.work_shape_at_contact.shape[0]
    shape = np.zeros((nres, 3))
    comp = np.array(grid.extra_S, dtype=float)
    same = grid.compliance.shape[:3] == tuple(dims)
    for ii, wx in ((lo[0][0], 1.0 - lo[0][2]), (lo[0][1], lo[0][2])):
        for jj, wy in ((lo[1][0], 1.0 - lo[1][2]), (lo[1][1], lo[1][2])):
            for kk, wz in ((lo[2][0], 1.0 - lo[2][2]), (lo[2][1], lo[2][2])):
                wgt = wx * wy * wz
                if wgt == 0.0:
                    continue
                n = min(nres, grid.shape.shape[0])
                shape[:n] += wgt * grid.shape[:n, ii, jj, kk, :]
                if same:
                    comp += wgt * grid.compliance[ii, jj, kk]
    m.work_shape_at_contact[:] = shape
    dm.res_work_S[:] = comp
    for k in _work_rows(sys):
        dm.contact_shape[k] = -(grid.R_S_T0.T @ shape[int(dm.row[k])])
    dm.iscal[MI_HAS_WORK_RES] = int(np.any(comp != 0.0))


def relative_axis_acceleration_ref(axes, acc_axis, out) -> None:
    out[:] = 0.0
    for a in range(min(len(axes.side), len(acc_axis))):
        if axes.side[a] == 0.0:
            continue
        if axes.rotary[a]:
            col = int(axes.column[a])
            if col >= 0 and BASE_LINEAR + col < len(out):
                out[BASE_LINEAR + col] = acc_axis[a]
            continue
        out[:3] += axes.side[a] * acc_axis[a] * axes.dir_M[a]


def interference_area_ref(slope, land_m, tan_clear, max_len_m) -> float:
    """Direct quadrature of delta(x) = slope x - max(0, x - L) tan(gamma) over the interference [0, x*]."""
    if slope <= 0.0:
        return 0.0
    L = min(max(land_m, 0.0), max_len_m)
    xs = max_len_m if slope >= tan_clear else min(L * tan_clear / (tan_clear - slope), max_len_m)
    xs = max(xs, L)
    x = np.linspace(0.0, xs, 20001)
    delta = slope * x - np.maximum(0.0, x - L) * tan_clear
    return float(np.trapezoid(delta, x))


def process_damping_element_ref(K_sp, db, land, tan_clear, max_len, v_n, V, dt, acc) -> float:
    if v_n <= 0.0 or V <= 0.0 or K_sp <= 0.0:
        return 0.0
    F = K_sp * db * interference_area_ref(v_n / V, land, tan_clear, max_len)
    acc.scalars[ACC_PROCESS_DAMPING_J] += F * v_n * dt
    return F


def deflection_at_point_ref(defl, s, r_T0) -> np.ndarray:
    u = defl.u_st_T0_m[s]
    v = defl.v_st_T0_m_s[s]
    r = np.asarray(r_T0, float)
    return np.concatenate([u[:3] + np.cross(u[3:], r), v[:3] + np.cross(v[3:], r)])


__all__ = ["free_interval_ref", "external_force_ref", "modal_predict_ref", "modal_deflection_ref",
           "relative_deflection_ref", "modal_correct_ref", "modal_free_ref", "refresh_work_shapes_ref",
           "relative_axis_acceleration_ref", "interference_area_ref", "process_damping_element_ref",
           "deflection_at_point_ref"]
