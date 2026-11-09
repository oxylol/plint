"""Plain-Python references of the thermal kernels (docs/architecture.md 5.5): independent
implementations of the same equations (dense linear algebra, adaptive quadrature, brute-force
superposition) that the tests compare the compiled kernels against. Not used in a run.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate
from scipy.special import erf, erfc

from .abi import (B_ALL_WET, B_COOLANT_ON, B_EPS, B_FMIN, B_H_CONTACT, B_H_WET, B_SIGMA, B_T_AIR, B_T_FIX,
                  B_T_FLUID, B_T_RAD, EDGE_INTERRUPTED, EDGE_NONE)

FULL_CELL = 1.0 - 1e-12


def interp(xs, vs, x):
    """Linear interpolation with end values held, in the kernel's operation order (the half-cell
    resistances are stored in float32, where a last-bit difference of k would show)."""
    n = len(xs)
    if n == 1 or x <= xs[0]:
        return float(vs[0])
    if x >= xs[-1]:
        return float(vs[-1])
    i = int(np.searchsorted(xs, x, side="right") - 1)
    t = (x - xs[i]) / (xs[i + 1] - xs[i])
    return float(vs[i] + t * (vs[i + 1] - vs[i]))


# ------------------------------------------------------------------ properties
def enthalpy_ref(mat, temp_K: float) -> float:
    """Integral of the piecewise-linear cp (end values held) from 20 degC (adaptive quadrature)."""
    def cp(t):
        return float(np.interp(t, mat.T_K, mat.cp_J_kgK))
    pts = [p for p in mat.T_K if min(293.15, temp_K) < p < max(293.15, temp_K)]
    value, _ = integrate.quad(cp, 293.15, temp_K, points=pts or None, limit=200, epsabs=1e-12, epsrel=1e-13)
    return value


def temperature_ref(mat, h: float) -> float:
    from scipy.optimize import brentq
    return brentq(lambda t: enthalpy_ref(mat, t) - h, 1.0, 1.0e4, xtol=1e-12, rtol=1e-14, maxiter=200)


# ------------------------------------------------------------------ workpiece
def _exposure(f, i, j, k, fmin):
    nx, ny, nz = f.shape
    bits = 0
    nbrs = ((i - 1, j, k), (i + 1, j, k), (i, j - 1, k), (i, j + 1, k), (i, j, k - 1), (i, j, k + 1))
    for b, (a, c, d) in enumerate(nbrs):
        if not (0 <= a < nx and 0 <= c < ny and 0 <= d < nz) or f[a, c, d] <= fmin:
            bits |= 1 << b
    return bits


def _weights(f, i, j, k, bits):
    fi = f[i, j, k]
    if fi >= FULL_CELL:
        return np.full(3, 1.0 / 3.0)
    e = np.array([((bits >> 0) & 1) + ((bits >> 1) & 1), ((bits >> 2) & 1) + ((bits >> 3) & 1),
                  ((bits >> 4) & 1) + ((bits >> 5) & 1)], dtype=float)
    if (e * e).sum() > 0:
        return e * e / (e * e).sum()
    nx, ny, nz = f.shape

    def val(a, c, d):
        return f[a, c, d] if 0 <= a < nx and 0 <= c < ny and 0 <= d < nz else 0.0
    g = np.array([val(i + 1, j, k) - val(i - 1, j, k), val(i, j + 1, k) - val(i, j - 1, k),
                  val(i, j, k + 1) - val(i, j, k - 1)])
    if (g * g).sum() > 0:
        return g * g / (g * g).sum()
    return np.full(3, 1.0 / 3.0)


def _coolant_h(bnd, x, y, z):
    sc = bnd.scal
    if sc[B_COOLANT_ON] <= 0:
        return 0.0
    best = 0.0
    for row in bnd.near:
        if (x - row[0]) ** 2 + (y - row[1]) ** 2 + (z - row[2]) ** 2 <= row[3] ** 2 and row[4] > best:
            best = row[4]
    if best > 0:
        return best
    if sc[B_ALL_WET] > 0:
        return sc[B_H_WET]
    if bnd.wet.size == 0:
        return 0.0
    iw = math.floor((x - bnd.wet_geom[0]) / bnd.wet_geom[2])
    jw = math.floor((y - bnd.wet_geom[1]) / bnd.wet_geom[2])
    if 0 <= iw < bnd.wet.shape[0] and 0 <= jw < bnd.wet.shape[1] and bnd.wet[iw, jw]:
        return sc[B_H_WET]
    return 0.0


def _face_h(bnd, x, y, z, temp, area, face, contact):
    """(air, coolant, radiation, contact) conductances and their temperatures."""
    sc = bnd.scal
    if contact:
        return np.array([0.0, 0.0, 0.0, sc[B_H_CONTACT] * area])
    hc = _coolant_h(bnd, x, y, z)
    if hc > 0:
        return np.array([0.0, hc * area, 0.0, 0.0])
    tr = sc[B_T_RAD]
    hr = sc[B_EPS] * sc[B_SIGMA] * (temp ** 2 + tr ** 2) * (temp + tr)
    return np.array([bnd.h_air[face] * area, 0.0, hr * area, 0.0])


def _prepare(grid, mat, bnd):
    """Capacities, half-cell resistances (float32 like the kernel), exposure bits of the step."""
    f = grid.f
    h = grid.h.copy()
    temp = grid.T.copy()
    origin = grid.geom[:3]
    dx = grid.geom[3]
    fmin = bnd.scal[B_FMIN]
    rho_v = mat.rho_kg_m3[0] * dx ** 3
    t_env = np.array([bnd.scal[B_T_AIR], bnd.scal[B_T_FLUID], bnd.scal[B_T_RAD], bnd.scal[B_T_FIX]])
    material = f > fmin
    cap = np.zeros(f.shape)
    r_half = np.zeros(f.shape + (3,))
    bits = np.zeros(f.shape, dtype=int)
    for i, j, k in zip(*np.nonzero(material)):
        fi = f[i, j, k]
        t = temp[i, j, k]
        cap[i, j, k] = rho_v * fi * interp(mat.T_K, mat.cp_J_kgK, t)
        kk = interp(mat.T_K, mat.k_W_mK, t)
        b = _exposure(f, i, j, k, fmin)
        bits[i, j, k] = b
        w = _weights(f, i, j, k, b)
        base = 1.0 / (2.0 * kk * dx)             # same operation order as the kernel (float32 storage)
        r_half[i, j, k] = [np.float32(base * (w[a] * fi + (1.0 - w[a]) / fi)) for a in range(3)]
    return cap, r_half, bits, material, h, temp, t_env, origin, dx, rho_v, fmin


def work_step_reference(grid, src, mat, bnd, dt):
    """Reference of kernels.work_step (locally one-dimensional backward Euler, each sub-step a dense
    solve per line): returns (h, T, terms) with terms = {src, air, cool, rad, contact, du}."""
    cap, r_half, bits, material, h, temp, t_env, origin, dx, rho_v, fmin = _prepare(grid, mat, bnd)
    f = grid.f
    area = dx * dx
    losses = np.zeros(4)

    def face_terms(i, j, k, face):
        b = bits[i, j, k]
        if not (b >> face) & 1:
            return np.zeros(4)
        e = [((b >> 0) & 1) + ((b >> 1) & 1), ((b >> 2) & 1) + ((b >> 3) & 1), ((b >> 4) & 1) + ((b >> 5) & 1)]
        tot = sum(e)
        if f[i, j, k] >= FULL_CELL or tot == 0:
            af = 1.0
        elif sum(1 for v in e if v == 2) == 1:            # a slab thinner than the cell: shell faces
            af = 1.0 if e[face // 2] == 2 else f[i, j, k]
        else:
            af = math.sqrt(sum(v * v for v in e)) / tot
        c = origin + (np.array([i, j, k]) + 0.5) * dx
        return _face_h(bnd, c[0], c[1], c[2], temp[i, j, k], af * area, face,
                       bool((grid.contact[i, j, k] >> face) & 1))

    def g_between(a, b, axis):
        return 1.0 / (float(np.float32(r_half[a + (axis,)])) + float(np.float32(r_half[b + (axis,)])))

    current = np.where(material, temp, 0.0)
    for axis in range(3):
        new = np.zeros(f.shape)
        other = [a for a in range(3) if a != axis]
        for p in range(f.shape[other[0]]):
            for q in range(f.shape[other[1]]):
                idx = []
                for s_ in range(f.shape[axis]):
                    cell = [0, 0, 0]
                    cell[axis] = s_
                    cell[other[0]] = p
                    cell[other[1]] = q
                    idx.append(tuple(cell))
                n = len(idx)
                a_mat = np.zeros((n, n))
                b_vec = np.zeros(n)
                for s_, cell in enumerate(idx):
                    if not material[cell]:
                        a_mat[s_, s_] = 1.0
                        continue
                    hs = sum(face_terms(*cell, face) for face in (2 * axis, 2 * axis + 1))
                    a_mat[s_, s_] = cap[cell] + dt * hs.sum()
                    b_vec[s_] = cap[cell] * current[cell] + dt * float(np.sum(hs * t_env))
                    if axis == 0:
                        b_vec[s_] += src[cell]
                    for t_s in (s_ - 1, s_ + 1):
                        if 0 <= t_s < n and material[idx[t_s]]:
                            g = g_between(cell, idx[t_s], axis)
                            a_mat[s_, s_] += dt * g
                            a_mat[s_, t_s] -= dt * g
                sol = np.linalg.solve(a_mat, b_vec)
                for s_, cell in enumerate(idx):
                    if material[cell]:
                        new[cell] = sol[s_]
                        for face in (2 * axis, 2 * axis + 1):
                            losses += dt * face_terms(*cell, face) * (sol[s_] - t_env)
        current = new
    delta = np.where(material, current - temp, 0.0)
    du = float(np.sum(cap * delta))
    h_new = h.copy()
    t_new = temp.copy()
    for cell in zip(*np.nonzero(material)):
        h_new[cell] += cap[cell] * delta[cell] / (rho_v * f[cell])
        t_new[cell] = temperature_ref_fast(mat, h_new[cell])
    return h_new, t_new, {"src": float(np.sum(np.where(material, src, 0.0))), "air": losses[0], "cool": losses[1],
                          "rad": losses[2], "contact": losses[3], "du": du}


def temperature_ref_fast(mat, h: float) -> float:
    """Inverse enthalpy by bisection on the exact piecewise-quadratic enthalpy (reference)."""
    lo, hi = 1.0, 1.0e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _enthalpy_exact(mat, mid) < h:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _enthalpy_exact(mat, t):
    tg, cp, hh = mat.T_K, mat.cp_J_kgK, mat.H_J_kg
    if t <= tg[0]:
        return hh[0] + cp[0] * (t - tg[0])
    if t >= tg[-1]:
        return hh[-1] + cp[-1] * (t - tg[-1])
    i = int(np.searchsorted(tg, t, side="right") - 1)
    d = t - tg[i]
    return hh[i] + cp[i] * d + 0.5 * (cp[i + 1] - cp[i]) / (tg[i + 1] - tg[i]) * d * d


def fill_update_ref(f, h, fnew_full, rho_v, fmin, h_amb):
    """Reference of kernels.fill_update on a full grid: returns (f, h, removed, added)."""
    f = f.copy()
    h = h.copy()
    removed = 0.0
    added = 0.0
    nx, ny, nz = f.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                fo = f[i, j, k]
                fn = fnew_full[i, j, k]
                fn = 0.0 if fn <= fmin else min(fn, 1.0)
                if fn == fo:
                    continue
                if fn < fo:
                    removed += rho_v * (fo - fn) * h[i, j, k]
                    f[i, j, k] = fn
                    if fn == 0.0:
                        h[i, j, k] = 0.0
                    continue
                if fo > fmin:
                    h_add = h[i, j, k]
                else:
                    vals = [h[a, b, c] for a, b, c in ((i - 1, j, k), (i + 1, j, k), (i, j - 1, k), (i, j + 1, k),
                                                       (i, j, k - 1), (i, j, k + 1))
                            if 0 <= a < nx and 0 <= b < ny and 0 <= c < nz and f[a, b, c] > fmin]
                    h_add = sum(vals) / len(vals) if vals else h_amb
                    fo = 0.0
                added += rho_v * (fn - fo) * h_add
                h[i, j, k] = (fo * h[i, j, k] + (fn - fo) * h_add) / fn
                f[i, j, k] = fn
    return f, h, removed, added


def deposit_point_ref(f, geom, point, normal, energy, fmin, search):
    """Reference of one point deposit: returns the src grid and the unplaced energy."""
    src = np.zeros(f.shape)
    dx = geom[3]
    n = np.asarray(normal, dtype=float)
    norm = np.linalg.norm(n)
    trials = [np.asarray(point) - (0.5 + s) * dx * n / norm for s in range(search + 1)] if norm > 0 else \
        [np.asarray(point, dtype=float)]
    for p in trials:
        u = (p - geom[:3]) / dx - 0.5
        base = np.floor(u).astype(int)
        t = u - base
        weights = {}
        for corner in np.ndindex(2, 2, 2):
            cell = tuple(base + np.array(corner))
            if all(0 <= cell[a] < f.shape[a] for a in range(3)) and f[cell] > fmin:
                w = np.prod([t[a] if corner[a] else 1 - t[a] for a in range(3)]) * f[cell]
                weights[cell] = w
        total = sum(weights.values())
        if total > 1e-12:
            for cell, w in weights.items():
                src[cell] += energy * w / total
            return src, 0.0
    centre = np.floor((np.asarray(point) - geom[:3]) / dx).astype(int)
    best = None
    best_d = np.inf
    for off in np.ndindex(2 * search + 1, 2 * search + 1, 2 * search + 1):
        cell = tuple(centre + np.array(off) - search)
        if all(0 <= cell[a] < f.shape[a] for a in range(3)) and f[cell] > fmin:
            d = np.sum((geom[:3] + (np.array(cell) + 0.5) * dx - point) ** 2)
            if d < best_d:
                best_d = d
                best = cell
    if best is not None:
        src[best] += energy
        return src, 0.0
    return src, energy


# ------------------------------------------------------------------ ladder
def ladder_step_ref(lad, mat_tool, mat_holder, dt):
    """Dense backward-Euler reference of kernels.ladder_step: returns (T_new, terms)."""
    n = lad.m.size
    cap = np.zeros(n)
    kk = np.zeros(n)
    for p in range(n):
        mat = mat_tool if lad.mat[p] == 0 else mat_holder
        cap[p] = lad.m[p] * np.interp(lad.T[p], mat.T_K, mat.cp_J_kgK)
        kk[p] = np.interp(lad.T[p], mat.T_K, mat.k_W_mK)
    a_mat = np.diag(cap + dt * (lad.g_air + lad.g_cool + lad.g_sp))
    rhs = lad.src + dt * (lad.g_air * (lad.scal[0] - lad.T) + lad.g_cool * (lad.scal[1] - lad.T)
                          + lad.g_sp * (lad.scal[2] - lad.T))
    for a, b, kind, g_fixed in zip(lad.link_a, lad.link_b, lad.link_kind, lad.link_g):
        g = g_fixed if kind == 1 else 1.0 / (lad.dz[a] / (2 * kk[a] * lad.area[a]) + lad.dz[b] / (2 * kk[b] * lad.area[b]))
        a_mat[a, a] += dt * g
        a_mat[b, b] += dt * g
        a_mat[a, b] -= dt * g
        a_mat[b, a] -= dt * g
        rhs[a] += dt * g * (lad.T[b] - lad.T[a])
        rhs[b] += dt * g * (lad.T[a] - lad.T[b])
    delta = np.linalg.solve(a_mat, rhs)
    t_new = lad.T + delta
    terms = {"air": float(dt * np.sum(lad.g_air * (t_new - lad.scal[0]))),
             "cool": float(dt * np.sum(lad.g_cool * (t_new - lad.scal[1]))),
             "sp": float(dt * np.sum(lad.g_sp * (t_new - lad.scal[2]))), "du": float(np.sum(cap * delta))}
    return delta, terms


# ------------------------------------------------------------------ edge
def unit_response_ref(t, a, b, k, alpha, which):
    """Rise per unit flux after t (K per W/m2) at the centre (which 0) or the area mean (1) of a
    rectangle 2a x 2b, by adaptive quadrature of the defining integral in u = sqrt(tau)."""
    if t <= 0:
        return 0.0
    rho_c = k / alpha

    def strip_mean(c, s4):                  # mean over the strip of the erf pair, s4 = 4 alpha tau
        root = math.sqrt(s4)
        return math.erf(2 * c / root) - root / (2 * c * math.sqrt(math.pi)) * (1 - math.exp(-4 * c * c / s4))

    def integrand(u):
        tau = u * u
        if tau <= 0:
            return 2.0 / math.sqrt(math.pi * alpha)
        if which == 0:
            sig = 2.0 * math.sqrt(alpha * tau)
            g = math.erf(a / sig) * math.erf(b / sig)
        else:
            g = strip_mean(a, 4 * alpha * tau) * strip_mean(b, 4 * alpha * tau)
        return 2.0 / math.sqrt(math.pi * alpha) * g

    value, _ = integrate.quad(integrand, 0.0, math.sqrt(t), limit=400, epsabs=0.0, epsrel=1e-11)
    return value / rho_c


def periodic_ref(tau, th, period, ncyc, a, b, k, alpha, which):
    """Brute-force superposition of ncyc heating cycles (every cycle term by term)."""
    total = 0.0
    for n in range(int(ncyc)):
        t1 = tau + n * period
        total += unit_response_ref(t1, a, b, k, alpha, which) - unit_response_ref(t1 - th, a, b, k, alpha, which)
    return total


def disc_depth_factor_ref(depth, radius):
    if depth <= 0 or radius <= 0:
        return 1.0
    value, _ = integrate.quad(lambda z: (math.sqrt(radius ** 2 + z * z) - z) / radius, 0.0, depth, epsrel=1e-12)
    return value / depth


def edge_element_ref(mode, qr, ar, qf, af, b, th, period, ncyc, t_eng, k, alpha, wedge, rfar, tb, xi):
    """Reference temperatures (rake[q], peak[q], flank[q], amp) of one element (brute force)."""
    if mode == EDGE_NONE:
        return np.full(len(xi), tb), np.full(len(xi), tb), np.full(len(xi), tb), 0.0
    if mode == EDGE_INTERRUPTED:
        duty = th / period
        t_total = ncyc * period
    else:
        duty = 1.0
        t_total = t_eng
    far_scale = duty * erfc(rfar / (2 * math.sqrt(alpha * t_total))) / (math.pi * k * rfar)
    far_r = qr * ar * 2 * b * far_scale
    far_f = qf * af * 2 * b * far_scale
    s_rf = disc_depth_factor_ref(ar, math.sqrt(4 * af * b / math.pi))
    s_fr = disc_depth_factor_ref(af, math.sqrt(4 * ar * b / math.pi))
    rake = []
    peak = []
    flank = []

    def sums(tau, a, which):
        if mode == EDGE_INTERRUPTED:
            return periodic_ref(tau, th, period, ncyc, a, b, k, alpha, which)
        return unit_response_ref(t_eng, a, b, k, alpha, which)
    for x in xi:
        tau = x * th
        rc, rm = sums(tau, ar, 0), sums(tau, ar, 1)
        fc, fm = sums(tau, af, 0), sums(tau, af, 1)
        er = qr * rc - far_r
        ef = qf * fc - far_f
        rake.append(tb + wedge * max(qr * rm - far_r + s_rf * ef, 0.0))
        flank.append(tb + wedge * max(qf * fm - far_f + s_fr * er, 0.0))
        peak.append(tb + wedge * max(er + ef, 0.0))
    amp = 0.0
    if mode == EDGE_INTERRUPTED:
        hot = qr * sums(th, ar, 0) + qf * sums(th, af, 0)
        cold = qr * sums(period, ar, 0) + qf * sums(period, af, 0)
        amp = wedge * (hot - cold)
    return np.array(rake), np.array(peak), np.array(flank), amp


# ------------------------------------------------------------------ moving source
def moving_rect_ref(q, l, w, v, k, alpha, t, x, y, z):
    """Reference of kernels.moving_rect_rise by adaptive quadrature in tau (w <= 0: infinite width)."""
    if t <= 0:
        return 0.0

    def integrand(tau):
        if tau <= 0:
            return 0.0
        sig = 2 * math.sqrt(alpha * tau)
        xf = 0.5 * (erf((x + v * tau + l) / sig) - erf((x + v * tau - l) / sig))
        yf = 0.5 * (erf((w - y) / sig) + erf((w + y) / sig)) if w > 0 else 1.0
        return xf * yf * math.exp(-z * z / (sig * sig)) / math.sqrt(math.pi * alpha * tau)
    pts = sorted({p for p in ((-l - x) / v, (l - x) / v) if 0 < p < t}) if v > 0 else []
    edges = [0.0] + pts + [t]
    total = 0.0
    for lo, hi in zip(edges, edges[1:]):
        # tau = s^2 removes the 1/sqrt(tau) singularity at 0
        value, _ = integrate.quad(lambda s: 2 * s * integrand(s * s), math.sqrt(lo), math.sqrt(hi), limit=500,
                                  epsabs=0.0, epsrel=1e-10)
        total += value
    return q * alpha / k * total


# ------------------------------------------------------------------ cutting-region field
def field_step_ref(fm, fs, lad, mat_tool, mat_holder, dt):
    """Dense backward-Euler reference of kernels.field_step (direct solve of the same couplings).

    Returns (dT of the cells, dT of the ladder nodes, terms) without changing the inputs."""
    n = fs.T.size
    nn = lad.m.size
    big = n + nn
    t_air, t_cool, t_sp = float(lad.scal[0]), float(lad.scal[1]), float(lad.scal[2])
    temp = np.concatenate([fs.T, lad.T])
    cap = np.zeros(big)
    kk = np.ones(big)
    for i in range(n):
        cap[i] = fm.m[i] * np.interp(fs.T[i], mat_tool.T_K, mat_tool.cp_J_kgK)
        kk[i] = np.interp(fs.T[i], mat_tool.T_K, mat_tool.k_W_mK)
    active = np.asarray(fs.active) != 0
    for p in range(nn):
        if active[p]:
            mat = mat_tool if lad.mat[p] == 0 else mat_holder
            cap[n + p] = lad.m[p] * np.interp(lad.T[p], mat.T_K, mat.cp_J_kgK)
            kk[n + p] = np.interp(lad.T[p], mat.T_K, mat.k_W_mK)
        else:
            cap[n + p] = 1.0
    a_mat = np.diag(cap.copy())
    rhs = np.zeros(big)
    rhs[:n] = fs.src
    sinks = []                  # (row, conductance, reference temperature, kind)

    def couple(i, j, g):
        a_mat[i, i] += dt * g
        a_mat[j, j] += dt * g
        a_mat[i, j] -= dt * g
        a_mat[j, i] -= dt * g
        rhs[i] += dt * g * (temp[j] - temp[i])
        rhs[j] += dt * g * (temp[i] - temp[j])

    for a, c, area, da, db in zip(fm.link_a, fm.link_b, fm.link_area, fm.link_da, fm.link_db):
        couple(int(a), int(c), area / (da / kk[a] + db / kk[c]))
    couples = []
    for f in range(fm.face_cell.size):
        c = int(fm.face_cell[f])
        node = int(fm.face_node[f])
        if node >= 0:
            if not active[node]:
                continue
            g = fm.face_area[f] / (fm.face_d[f] / kk[c] + fm.face_rc[f] + fm.face_lk[f] / kk[n + node])
            couple(c, n + node, g)
            couples.append((c, n + node, g))
            continue
        if fs.fa_w[f] > 0 and fs.fa_h[f] > 0:
            sinks.append((c, fs.fa_w[f] / (fm.face_d[f] / kk[c] + 1.0 / fs.fa_h[f]), t_air, "air"))
        if fs.fc_w[f] > 0 and fs.fc_h[f] > 0:
            sinks.append((c, fs.fc_w[f] / (fm.face_d[f] / kk[c] + 1.0 / fs.fc_h[f]), t_cool, "cool"))
    for p in range(nn):
        if not active[p]:
            continue
        rhs[n + p] += lad.src[p]
        for g, ref, kind in ((lad.g_air[p], t_air, "air"), (lad.g_cool[p], t_cool, "cool"), (lad.g_sp[p], t_sp, "sp")):
            if g > 0:
                sinks.append((n + p, g, ref, kind))
    for row, g, ref, _ in sinks:
        a_mat[row, row] += dt * g
        rhs[row] += dt * g * (ref - temp[row])
    for a, c, kind, g_fixed in zip(lad.link_a, lad.link_b, lad.link_kind, lad.link_g):
        a, c = int(a), int(c)
        if not (active[a] and active[c]):
            continue
        if kind == 0:
            g = 1.0 / (lad.dz[a] / (2 * kk[n + a] * lad.area[a]) + lad.dz[c] / (2 * kk[n + c] * lad.area[c]))
        else:
            g = g_fixed
        couple(n + a, n + c, g)
    delta = np.linalg.solve(a_mat, rhs)
    new = temp + delta
    terms = {kind: float(sum(dt * g * (new[row] - ref) for row, g, ref, k in sinks if k == kind))
             for kind in ("air", "cool", "sp")}
    terms["couple"] = float(sum(dt * g * (new[i] - new[j]) for i, j, g in couples))
    terms["du"] = float(np.sum(cap[:n] * delta[:n]) + np.sum(cap[n:][active] * delta[n:][active]))
    terms["src"] = float(np.sum(fs.src) + np.sum(np.asarray(lad.src)[active]))
    return delta[:n], delta[n:], terms


def edge_deviation_ref(qr, ar, qf, af, b, th, period, ncyc, k, alpha, wedge, xi):
    """Reference of the edge kernel's deviation mode for an interrupted element: the brute-force
    superposition at the phases minus its cycle mean, the mean by adaptive quadrature over the period."""
    s_rf = disc_depth_factor_ref(ar, math.sqrt(4 * af * b / math.pi))
    s_fr = disc_depth_factor_ref(af, math.sqrt(4 * ar * b / math.pi))

    def theta(tau, a, which):
        return periodic_ref(tau, th, period, ncyc, a, b, k, alpha, which)

    def mean(a, which):
        pts = [th] if 0.0 < th < period else None
        value, _ = integrate.quad(lambda s: theta(s, a, which), 0.0, period, points=pts, limit=200, epsrel=1e-9)
        return value / period

    m_rc, m_rm, m_fc, m_fm = mean(ar, 0), mean(ar, 1), mean(af, 0), mean(af, 1)
    rake, peak, flank = [], [], []
    for x in xi:
        tau = x * th
        er = qr * (theta(tau, ar, 0) - m_rc)
        ef = qf * (theta(tau, af, 0) - m_fc)
        rake.append(wedge * (qr * (theta(tau, ar, 1) - m_rm) + s_rf * ef))
        flank.append(wedge * (qf * (theta(tau, af, 1) - m_fm) + s_fr * er))
        peak.append(wedge * (er + ef))
    return np.array(rake), np.array(peak), np.array(flank)
