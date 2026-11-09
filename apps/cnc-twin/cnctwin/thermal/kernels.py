"""Compiled kernels of the thermal domain (docs/models/thermal.md; rules: docs/architecture.md 5.5).

* Material properties: piecewise-linear tables with exact enthalpy (``enthalpy``/``temperature``).
* Workpiece: finite volumes on the stock lattice with cut cells, locally one-dimensional
  backward-Euler splitting (monotone, unconditionally stable, exactly conservative: ``work_step``;
  Douglas-Gunn ADI was rejected: unstable with cut cells), removal of cell fractions with
  their enthalpy (``fill_update``), heat deposits into the remaining material (``deposit``).
* Tool and holder: implicit 1-D ladder, banded Cholesky (``ladder_step``).
* Cutting edge: periodic heating / cooling per revolution (Stephenson & Ali 1992) on a quarter
  space with mirrored rectangular contacts (``edge_temperatures``).
* Two-scale workpiece surface: finite rectangle source moving over a half space (Jaeger 1942)
  for a finite time (``moving_rect_rise``).
* Cutting region of the tool: 3-D finite volumes of the wedge at the edge solved together with the
  ladder, backward Euler, conjugate gradients with a block symmetric Gauss-Seidel preconditioner
  (exact banded solves of the slabs and of the ladder; ``field_step``); the edge kernel's deviation
  mode gives the periodic part around the cycle mean.

Every kernel has an explicit signature (one compiled specialisation), ``cache=True``,
``nogil=True``, ``error_model="numpy"``; plain-Python references are in
:mod:`cnctwin.thermal.reference`. Index constants are repeated here (checked against
:mod:`cnctwin.thermal.abi` at import) so a cached kernel can never see other values.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, typeof, types

from . import abi as _abi

# ------------------------------------------------------------------ layout constants (= abi)
B_T_AIR, B_T_FLUID, B_H_WET, B_ALL_WET, B_EPS, B_T_RAD = 0, 1, 2, 3, 4, 5
B_H_CONTACT, B_T_FIX, B_FMIN, B_COOLANT_ON, B_SIGMA = 6, 7, 8, 9, 10
L_SRC, L_AIR, L_COOL, L_RAD, L_CONTACT, L_DU, L_U0, L_U1 = 0, 1, 2, 3, 4, 5, 6, 7
L_AREA, L_TSURF, L_TMAX, L_MASS, L_TMEAN, L_LOST = 8, 9, 10, 11, 12, 13
LD_SRC, LD_AIR, LD_COOL, LD_SP, LD_DU, LD_U0, LD_U1 = 0, 1, 2, 3, 4, 5, 6
EDGE_NONE, EDGE_INTERRUPTED, EDGE_CONTINUOUS = 0, 1, 2
EDGE_OUT_TEMPERATURE, EDGE_OUT_DEVIATION = 0, 1
FS_TOL, FS_MAX_ITER = 0, 1
FD_SRC, FD_AIR, FD_COOL, FD_SP, FD_DU, FD_U0, FD_U1, FD_ITER, FD_RES, FD_COUPLE = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
FULL_CELL = 1.0 - 1e-12          # fractions above count as full cells (no orientation weighting)

for _name in ("B_T_AIR", "B_T_FLUID", "B_H_WET", "B_ALL_WET", "B_EPS", "B_T_RAD", "B_H_CONTACT", "B_T_FIX",
              "B_FMIN", "B_COOLANT_ON", "B_SIGMA", "L_SRC", "L_AIR", "L_COOL", "L_RAD", "L_CONTACT", "L_DU",
              "L_U0", "L_U1", "L_AREA", "L_TSURF", "L_TMAX", "L_MASS", "L_TMEAN", "L_LOST", "LD_SRC", "LD_AIR",
              "LD_COOL", "LD_SP", "LD_DU", "LD_U0", "LD_U1", "EDGE_NONE", "EDGE_INTERRUPTED",
              "EDGE_CONTINUOUS", "EDGE_OUT_TEMPERATURE", "EDGE_OUT_DEVIATION", "FS_TOL", "FS_MAX_ITER", "FD_SRC",
              "FD_AIR", "FD_COOL", "FD_SP", "FD_DU", "FD_U0", "FD_U1", "FD_ITER", "FD_RES", "FD_COUPLE"):
    if getattr(_abi, _name) != globals()[_name]:
        raise ImportError(f"cnctwin.thermal.kernels: {_name} differs from cnctwin.thermal.abi")
if _abi.N_L <= L_LOST or _abi.N_BSCAL <= B_SIGMA or _abi.N_LD <= LD_U1 or _abi.N_FD <= FD_COUPLE \
        or _abi.N_FS <= FS_MAX_ITER:
    raise ImportError("cnctwin.thermal.kernels: ledger/scalar arrays of cnctwin.thermal.abi are too short")

MAT_T, GRID_T, SCR_T, BND_T, LAD_T, RESP_T, EIN_T, EOUT_T = (typeof(x) for x in _abi.sample_types())
FMESH_T, FSYS_T = (typeof(x) for x in _abi.field_sample_types())

f8 = types.float64
i8 = types.int64
b1 = types.boolean
F1 = types.float64[::1]
I1 = types.int32[::1]
F2 = types.float64[:, ::1]
F3 = types.float64[:, :, ::1]
OPTS = dict(cache=True, nogil=True, error_model="numpy")
INLINE = dict(OPTS, inline="always")      # small helpers of the edge chain: inlined (no call overhead)
SQRT_PI = math.sqrt(math.pi)
LN10 = math.log(10.0)


# ================================================================== tables and enthalpy
@njit(i8(F1, f8), **OPTS)
def seg_index(xs, x):
    """Index i with xs[i] <= x < xs[i+1], clamped to [0, n-2] (0 for a single point)."""
    n = xs.shape[0]
    if n < 2 or x <= xs[0]:
        return 0
    if x >= xs[n - 1]:
        return n - 2
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    return lo


@njit(f8(F1, F1, f8), **OPTS)
def prop_interp(xs, vs, x):
    """Linear interpolation, end values held outside (the database table rule)."""
    n = xs.shape[0]
    if n == 1 or x <= xs[0]:
        return vs[0]
    if x >= xs[n - 1]:
        return vs[n - 1]
    i = seg_index(xs, x)
    t = (x - xs[i]) / (xs[i + 1] - xs[i])
    return vs[i] + t * (vs[i + 1] - vs[i])


@njit(f8(MAT_T, f8), **OPTS)
def enthalpy(mat, T):
    """Specific enthalpy (J/kg, rel. 20 degC) of the piecewise-linear cp, exact."""
    tg = mat.T_K
    cp = mat.cp_J_kgK
    hh = mat.H_J_kg
    n = tg.shape[0]
    if T <= tg[0]:
        return hh[0] + cp[0] * (T - tg[0])
    if T >= tg[n - 1]:
        return hh[n - 1] + cp[n - 1] * (T - tg[n - 1])
    i = seg_index(tg, T)
    d = T - tg[i]
    s = (cp[i + 1] - cp[i]) / (tg[i + 1] - tg[i])
    return hh[i] + cp[i] * d + 0.5 * s * d * d


@njit(f8(MAT_T, f8), **OPTS)
def temperature(mat, h):
    """Inverse of :func:`enthalpy` (stable root of the segment's quadratic)."""
    tg = mat.T_K
    cp = mat.cp_J_kgK
    hh = mat.H_J_kg
    n = tg.shape[0]
    if h <= hh[0]:
        return tg[0] + (h - hh[0]) / cp[0]
    if h >= hh[n - 1]:
        return tg[n - 1] + (h - hh[n - 1]) / cp[n - 1]
    i = seg_index(hh, h)
    dh = h - hh[i]
    s = (cp[i + 1] - cp[i]) / (tg[i + 1] - tg[i])
    disc = cp[i] * cp[i] + 2.0 * s * dh
    if disc < 0.0:
        disc = 0.0
    return tg[i] + 2.0 * dh / (cp[i] + math.sqrt(disc))


@njit(types.UniTuple(f8, 2)(f8, f8, f8), **OPTS)
def neumaier(total, comp, x):
    """One step of Neumaier's compensated summation: returns (total, compensation)."""
    t = total + x
    if abs(total) >= abs(x):
        comp += (total - t) + x
    else:
        comp += (x - t) + total
    return t, comp


# ================================================================== workpiece grid
@njit(types.void(GRID_T, MAT_T, f8), **OPTS)
def grid_temperature(g, mat, fmin):
    """T from h in every material cell, NaN in empty cells."""
    nx, ny, nz = g.f.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if g.f[i, j, k] > fmin:
                    g.T[i, j, k] = temperature(mat, g.h[i, j, k])
                else:
                    g.T[i, j, k] = np.nan


@njit(types.void(GRID_T, MAT_T, f8, F1), **OPTS)
def work_energy(g, mat, fmin, out):
    """out: [0] internal energy (J, rel. 20 degC), [1] mass (kg), [2] mass-weighted mean T (K),
    [3] max T (K) (compensated sums, fixed order)."""
    nx, ny, nz = g.f.shape
    dx = g.geom[3]
    rho_v = mat.rho_kg_m3[0] * dx * dx * dx
    u_s = 0.0
    u_c = 0.0
    m_s = 0.0
    m_c = 0.0
    t_s = 0.0
    t_c = 0.0
    t_max = -np.inf
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                fi = g.f[i, j, k]
                if fi <= fmin:
                    continue
                m = rho_v * fi
                u_s, u_c = neumaier(u_s, u_c, m * g.h[i, j, k])
                m_s, m_c = neumaier(m_s, m_c, m)
                t_s, t_c = neumaier(t_s, t_c, m * g.T[i, j, k])
                if g.T[i, j, k] > t_max:
                    t_max = g.T[i, j, k]
    mass = m_s + m_c
    out[0] = u_s + u_c
    out[1] = mass
    out[2] = (t_s + t_c) / mass if mass > 0.0 else np.nan
    out[3] = t_max if mass > 0.0 else np.nan


@njit(types.void(GRID_T, MAT_T, F3, i8, i8, i8, f8, f8, F1), **OPTS)
def fill_update(g, mat, fnew, i0, j0, k0, fmin, h_amb, out):
    """Apply new material fractions on the sub-box starting at cell (i0, j0, k0).

    Removed material leaves with the specific enthalpy of its cell; re-added material (side flow,
    burrs, forming) enters with the cell's enthalpy, or the mean of its material neighbours, or
    ``h_amb`` (ambient) when it has none. out: [0] removed enthalpy (J), [1] added enthalpy (J),
    [2] removed mass (kg), [3] added mass (kg), [4] cells changed."""
    f = g.f
    h = g.h
    nx, ny, nz = f.shape
    bx, by, bz = fnew.shape
    dx = g.geom[3]
    rho_v = mat.rho_kg_m3[0] * dx * dx * dx
    rem_s = 0.0
    rem_c = 0.0
    add_s = 0.0
    add_c = 0.0
    rem_m = 0.0
    add_m = 0.0
    changed = 0
    for a in range(bx):
        i = i0 + a
        if i < 0 or i >= nx:
            continue
        for b in range(by):
            j = j0 + b
            if j < 0 or j >= ny:
                continue
            for c in range(bz):
                k = k0 + c
                if k < 0 or k >= nz:
                    continue
                fo = f[i, j, k]
                fn = fnew[a, b, c]
                if fn <= fmin:
                    fn = 0.0
                if fn > 1.0:
                    fn = 1.0
                if fn == fo:
                    continue
                changed += 1
                if fn < fo:
                    dm = rho_v * (fo - fn)
                    rem_s, rem_c = neumaier(rem_s, rem_c, dm * h[i, j, k])
                    rem_m += dm
                    f[i, j, k] = fn
                    if fn == 0.0:
                        h[i, j, k] = 0.0
                        g.T[i, j, k] = np.nan
                    continue
                dm = rho_v * (fn - fo)
                if fo > fmin:
                    h_add = h[i, j, k]
                else:
                    total = 0.0
                    count = 0
                    for d in range(6):
                        ii = i
                        jj = j
                        kk = k
                        if d == 0:
                            ii = i - 1
                        elif d == 1:
                            ii = i + 1
                        elif d == 2:
                            jj = j - 1
                        elif d == 3:
                            jj = j + 1
                        elif d == 4:
                            kk = k - 1
                        else:
                            kk = k + 1
                        if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz and f[ii, jj, kk] > fmin:
                            total += h[ii, jj, kk]
                            count += 1
                    h_add = total / count if count > 0 else h_amb
                    fo = 0.0
                add_s, add_c = neumaier(add_s, add_c, dm * h_add)
                add_m += dm
                h[i, j, k] = (fo * h[i, j, k] + (fn - fo) * h_add) / fn
                f[i, j, k] = fn
                g.T[i, j, k] = temperature(mat, h[i, j, k])
    out[0] = rem_s + rem_c
    out[1] = add_s + add_c
    out[2] = rem_m
    out[3] = add_m
    out[4] = changed


@njit(b1(F3, F3, F1, f8, f8, f8, f8, f8), **OPTS)
def place_trilinear(f, src, geom, x, y, z, energy, fmin):
    """Spread ``energy`` over the 8 cells around (x, y, z) with trilinear weights times the
    material fraction; False when none of them holds material."""
    nx, ny, nz = f.shape
    dx = geom[3]
    u = (x - geom[0]) / dx - 0.5
    v = (y - geom[1]) / dx - 0.5
    w = (z - geom[2]) / dx - 0.5
    iu = int(math.floor(u))
    iv = int(math.floor(v))
    iw = int(math.floor(w))
    tu = u - iu
    tv = v - iv
    tw = w - iw
    total = 0.0
    for a in range(2):
        i = iu + a
        if i < 0 or i >= nx:
            continue
        wa = tu if a == 1 else 1.0 - tu
        for b in range(2):
            j = iv + b
            if j < 0 or j >= ny:
                continue
            wb = tv if b == 1 else 1.0 - tv
            for c in range(2):
                k = iw + c
                if k < 0 or k >= nz:
                    continue
                if f[i, j, k] <= fmin:
                    continue
                wc = tw if c == 1 else 1.0 - tw
                total += wa * wb * wc * f[i, j, k]
    if total <= 1e-12:
        return False
    for a in range(2):
        i = iu + a
        if i < 0 or i >= nx:
            continue
        wa = tu if a == 1 else 1.0 - tu
        for b in range(2):
            j = iv + b
            if j < 0 or j >= ny:
                continue
            wb = tv if b == 1 else 1.0 - tv
            for c in range(2):
                k = iw + c
                if k < 0 or k >= nz:
                    continue
                if f[i, j, k] <= fmin:
                    continue
                wc = tw if c == 1 else 1.0 - tw
                src[i, j, k] += energy * (wa * wb * wc * f[i, j, k]) / total
    return True


@njit(types.void(F3, F3, F1, f8, f8, f8, f8, f8, f8, f8, f8, i8, F1), **OPTS)
def place_point(f, src, geom, px, py, pz, ux, uy, uz, energy, fmin, search, out):
    """Deposit at a surface point with outward normal u: half a cell into the material, then
    marching inward, then the nearest material cell within ``search`` cells; else unplaced.
    out[0] += unplaced J, out[1] += placed J."""
    dx = geom[3]
    norm = math.sqrt(ux * ux + uy * uy + uz * uz)
    if norm > 0.0:
        ux /= norm
        uy /= norm
        uz /= norm
        for step in range(search + 1):
            depth = (0.5 + step) * dx
            if place_trilinear(f, src, geom, px - depth * ux, py - depth * uy, pz - depth * uz, energy, fmin):
                out[1] += energy
                return
    elif place_trilinear(f, src, geom, px, py, pz, energy, fmin):
        out[1] += energy
        return
    nx, ny, nz = f.shape
    ic = int(math.floor((px - geom[0]) / dx))
    jc = int(math.floor((py - geom[1]) / dx))
    kc = int(math.floor((pz - geom[2]) / dx))
    best_i = -1
    best_j = -1
    best_k = -1
    best_d = np.inf
    for i in range(ic - search, ic + search + 1):
        if i < 0 or i >= nx:
            continue
        for j in range(jc - search, jc + search + 1):
            if j < 0 or j >= ny:
                continue
            for k in range(kc - search, kc + search + 1):
                if k < 0 or k >= nz or f[i, j, k] <= fmin:
                    continue
                cx = geom[0] + (i + 0.5) * dx - px
                cy = geom[1] + (j + 0.5) * dx - py
                cz = geom[2] + (k + 0.5) * dx - pz
                d2 = cx * cx + cy * cy + cz * cz
                if d2 < best_d:
                    best_d = d2
                    best_i = i
                    best_j = j
                    best_k = k
    if best_i >= 0:
        src[best_i, best_j, best_k] += energy
        out[1] += energy
    else:
        out[0] += energy


@njit(types.void(F3, F3, F1, F2, f8, i8, F1), **OPTS)
def deposit(f, src, geom, rows, fmin, search, out):
    """Deposit heat rows [x, y, z, nx, ny, nz, J, arc] (S, material configuration) into ``src``.

    ``arc`` != 0 spreads the row over the circular arc of that signed angle about the z_S axis
    behind the point (turning: the band the workpiece carried past the tool), in sub-points one
    cell apart. out[0]: unplaced J, out[1]: placed J (accumulated)."""
    dx = geom[3]
    for r in range(rows.shape[0]):
        energy = rows[r, 6]
        if energy == 0.0:
            continue
        arc = rows[r, 7]
        x = rows[r, 0]
        y = rows[r, 1]
        z = rows[r, 2]
        if arc == 0.0:
            place_point(f, src, geom, x, y, z, rows[r, 3], rows[r, 4], rows[r, 5], energy, fmin, search, out)
            continue
        radius = math.sqrt(x * x + y * y)
        nsub = int(math.ceil(abs(arc) * radius / dx))
        if nsub < 1:
            nsub = 1
        if nsub > 8192:
            nsub = 8192
        part = energy / nsub
        for s in range(nsub):
            ang = -arc * (s + 0.5) / nsub
            c = math.cos(ang)
            sn = math.sin(ang)
            place_point(f, src, geom, c * x - sn * y, sn * x + c * y, z,
                        c * rows[r, 3] - sn * rows[r, 4], sn * rows[r, 3] + c * rows[r, 4], rows[r, 5],
                        part, fmin, search, out)


@njit(f8(BND_T, f8, f8, f8), **OPTS)
def coolant_h(bnd, x, y, z):
    """Coolant film coefficient at a point of the surface (0 when dry)."""
    sc = bnd.scal
    if sc[B_COOLANT_ON] <= 0.0:
        return 0.0
    best = 0.0
    for n in range(bnd.near.shape[0]):
        dxp = x - bnd.near[n, 0]
        dyp = y - bnd.near[n, 1]
        dzp = z - bnd.near[n, 2]
        if dxp * dxp + dyp * dyp + dzp * dzp <= bnd.near[n, 3] * bnd.near[n, 3] and bnd.near[n, 4] > best:
            best = bnd.near[n, 4]
    if best > 0.0:
        return best
    if sc[B_ALL_WET] > 0.0:
        return sc[B_H_WET]
    nwx = bnd.wet.shape[0]
    nwy = bnd.wet.shape[1]
    if nwx == 0 or nwy == 0:
        return 0.0
    cell = bnd.wet_geom[2]
    iw = int(math.floor((x - bnd.wet_geom[0]) / cell))
    jw = int(math.floor((y - bnd.wet_geom[1]) / cell))
    if 0 <= iw < nwx and 0 <= jw < nwy and bnd.wet[iw, jw] != 0:
        return sc[B_H_WET]
    return 0.0


@njit(types.UniTuple(f8, 4)(BND_T, f8, f8, f8, f8, f8, i8, b1), **OPTS)
def face_conductances(bnd, x, y, z, temp, area, face, contact):
    """(air, coolant, radiation, contact) conductances (W/K) of one exposed face."""
    sc = bnd.scal
    if contact:
        return 0.0, 0.0, 0.0, sc[B_H_CONTACT] * area
    hc = coolant_h(bnd, x, y, z)
    if hc > 0.0:
        return 0.0, hc * area, 0.0, 0.0
    tr = sc[B_T_RAD]
    h_rad = sc[B_EPS] * sc[B_SIGMA] * (temp * temp + tr * tr) * (temp + tr)
    return bnd.h_air[face] * area, 0.0, h_rad * area, 0.0


@njit(f8(types.uint8, f8, i8), **OPTS)
def face_fraction(bits, fi, face):
    """Area of an exposed face in dx^2 (docs/models/thermal.md "Cut cells"): 1 on full cells; on a
    cut cell with exactly one opposite pair exposed (a wall or floor thinner than the cell: the shell
    model) 1 for that pair and f for its edge faces; otherwise one oblique facet sqrt(sum e^2) / sum e
    shared by the exposed faces."""
    if fi >= FULL_CELL:
        return 1.0
    ex = ((bits >> 0) & 1) + ((bits >> 1) & 1)
    ey = ((bits >> 2) & 1) + ((bits >> 3) & 1)
    ez = ((bits >> 4) & 1) + ((bits >> 5) & 1)
    pairs = (1 if ex == 2 else 0) + (1 if ey == 2 else 0) + (1 if ez == 2 else 0)
    if pairs == 1:
        axis = face // 2
        e_axis = ex if axis == 0 else (ey if axis == 1 else ez)
        return 1.0 if e_axis == 2 else fi
    total = ex + ey + ez
    if total == 0:
        return 1.0
    return math.sqrt(float(ex * ex + ey * ey + ez * ez)) / total


@njit(types.void(GRID_T, SCR_T, MAT_T, BND_T, f8, F1), **OPTS)
def work_step(g, s, mat, bnd, dt, out):
    """One conservative, monotone step of the cut-cell finite volumes: locally one-dimensional
    backward-Euler splitting (Yanenko), the x, y and z sub-steps each an implicit tridiagonal solve
    (C + dt (A_d + H_d)) T_d = C T_(d-1) + dt H_d T_env (+ the deposited energy in the x sub-step).
    Each factor is an M-matrix solve with unit row sums, so the step obeys the maximum principle for
    any cut-cell fraction and any dt; exposed faces (convection to air and coolant, linearised
    radiation, fixture contact) are implicit in the sub-step of their axis.

    Exact identity (to round-off): sum C (T_z - T^n) = sum src - sum over faces of dt H (T_axis - T_env),
    T_axis the temperature after the face's sub-step. h is updated by C dT / m and T = T(h).
    ``out`` (length >= 14) receives the L_* ledger terms."""
    f = g.f
    h = g.h
    temp = g.T
    nx, ny, nz = f.shape
    ox = g.geom[0]
    oy = g.geom[1]
    oz = g.geom[2]
    dx = g.geom[3]
    face_area = dx * dx
    rho_v = mat.rho_kg_m3[0] * dx * dx * dx
    sc = bnd.scal
    fmin = sc[B_FMIN]
    t_air = sc[B_T_AIR]
    t_fl = sc[B_T_FLUID]
    t_rad = sc[B_T_RAD]
    t_fix = sc[B_T_FIX]
    tg = mat.T_K
    cp_tab = mat.cp_J_kgK
    k_tab = mat.k_W_mK
    for q in range(out.shape[0]):
        out[q] = 0.0
    src = s.src
    ta = s.a
    cap = s.cap
    rx = s.rx
    ry = s.ry
    rz = s.rz
    bits = s.bits
    lc = s.line_c
    ld = s.line_d
    gprev = s.gprev

    u0_s = 0.0
    u0_c = 0.0
    e_s = 0.0
    e_c = 0.0
    lost = 0.0
    # ---- pass 0: capacity, half-cell resistances, exposure, sums
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                fi = f[i, j, k]
                if fi <= fmin:
                    cap[i, j, k] = 0.0
                    rx[i, j, k] = 0.0
                    ry[i, j, k] = 0.0
                    rz[i, j, k] = 0.0
                    bits[i, j, k] = 0
                    lost += src[i, j, k]
                    src[i, j, k] = 0.0
                    continue
                ti = temp[i, j, k]
                m = rho_v * fi
                cap[i, j, k] = m * prop_interp(tg, cp_tab, ti)
                ki = prop_interp(tg, k_tab, ti)
                bb = 0
                if i == 0 or f[i - 1, j, k] <= fmin:
                    bb |= 1
                if i == nx - 1 or f[i + 1, j, k] <= fmin:
                    bb |= 2
                if j == 0 or f[i, j - 1, k] <= fmin:
                    bb |= 4
                if j == ny - 1 or f[i, j + 1, k] <= fmin:
                    bb |= 8
                if k == 0 or f[i, j, k - 1] <= fmin:
                    bb |= 16
                if k == nz - 1 or f[i, j, k + 1] <= fmin:
                    bb |= 32
                bits[i, j, k] = bb
                wx = 1.0 / 3.0
                wy = 1.0 / 3.0
                wz = 1.0 / 3.0
                if fi < FULL_CELL:
                    ex = float((bb & 1) + ((bb >> 1) & 1))
                    ey = float(((bb >> 2) & 1) + ((bb >> 3) & 1))
                    ez = float(((bb >> 4) & 1) + ((bb >> 5) & 1))
                    e2 = ex * ex + ey * ey + ez * ez
                    if e2 > 0.0:
                        wx = ex * ex / e2
                        wy = ey * ey / e2
                        wz = ez * ez / e2
                    else:
                        gx = (f[i + 1, j, k] if i < nx - 1 else 0.0) - (f[i - 1, j, k] if i > 0 else 0.0)
                        gy = (f[i, j + 1, k] if j < ny - 1 else 0.0) - (f[i, j - 1, k] if j > 0 else 0.0)
                        gz = (f[i, j, k + 1] if k < nz - 1 else 0.0) - (f[i, j, k - 1] if k > 0 else 0.0)
                        g2 = gx * gx + gy * gy + gz * gz
                        if g2 > 0.0:
                            wx = gx * gx / g2
                            wy = gy * gy / g2
                            wz = gz * gz / g2
                base = 1.0 / (2.0 * ki * dx)
                rx[i, j, k] = base * (wx * fi + (1.0 - wx) / fi)
                ry[i, j, k] = base * (wy * fi + (1.0 - wy) / fi)
                rz[i, j, k] = base * (wz * fi + (1.0 - wz) / fi)
                u0_s, u0_c = neumaier(u0_s, u0_c, m * h[i, j, k])
                e_s, e_c = neumaier(e_s, e_c, src[i, j, k])

    la_s = 0.0
    lc_s = 0.0
    lr_s = 0.0
    lk_s = 0.0
    area_s = 0.0
    tsurf_s = 0.0
    # ---- x sub-step: T^n (+ src) -> ta
    for j in range(ny):
        y = oy + (j + 0.5) * dx
        for k in range(nz):
            gprev[k] = 0.0
        for i in range(nx):
            x = ox + (i + 0.5) * dx
            for k in range(nz):
                fi = f[i, j, k]
                if fi <= fmin:
                    lc[i, k] = 0.0
                    ld[i, k] = 0.0
                    gprev[k] = 0.0
                    continue
                gl = gprev[k]
                gr = 0.0
                if i < nx - 1 and f[i + 1, j, k] > fmin:
                    gr = 1.0 / (np.float64(rx[i, j, k]) + np.float64(rx[i + 1, j, k]))
                ti = temp[i, j, k]
                hsum = 0.0
                rhs = cap[i, j, k] * ti + src[i, j, k]
                bb = bits[i, j, k]
                if bb != 0:
                    z = oz + (k + 0.5) * dx
                    cbits = g.contact[i, j, k]
                    for face in range(6):
                        if (bb >> face) & 1:
                            afc = face_fraction(bb, fi, face) * face_area
                            is_contact = ((cbits >> face) & 1) == 1
                            ha, hc, hr, hk = face_conductances(bnd, x, y, z, ti, afc, face, is_contact)
                            if not is_contact:
                                area_s += afc
                                tsurf_s += afc * ti
                            if face < 2:
                                hsum += ha + hc + hr + hk
                                rhs += dt * (ha * t_air + hc * t_fl + hr * t_rad + hk * t_fix)
                aa = -dt * gl
                bdiag = cap[i, j, k] + dt * (gl + gr + hsum)
                cc = -dt * gr
                if i > 0:
                    denom = bdiag - aa * lc[i - 1, k]
                    ld[i, k] = (rhs - aa * ld[i - 1, k]) / denom
                else:
                    denom = bdiag
                    ld[i, k] = rhs / denom
                lc[i, k] = cc / denom
                gprev[k] = gr
        for i in range(nx - 1, -1, -1):
            x = ox + (i + 0.5) * dx
            for k in range(nz):
                fi = f[i, j, k]
                if fi <= fmin:
                    ta[i, j, k] = 0.0
                    continue
                val = ld[i, k]
                if i < nx - 1:
                    val -= lc[i, k] * ta[i + 1, j, k]
                ta[i, j, k] = val
                bb = bits[i, j, k]
                if bb & 3:
                    z = oz + (k + 0.5) * dx
                    cbits = g.contact[i, j, k]
                    for face in range(2):
                        if (bb >> face) & 1:
                            afc = face_fraction(bb, fi, face) * face_area
                            ha, hc, hr, hk = face_conductances(bnd, x, y, z, temp[i, j, k], afc, face,
                                                               ((cbits >> face) & 1) == 1)
                            la_s += dt * ha * (val - t_air)
                            lc_s += dt * hc * (val - t_fl)
                            lr_s += dt * hr * (val - t_rad)
                            lk_s += dt * hk * (val - t_fix)

    # ---- y sub-step: ta -> src
    for i in range(nx):
        x = ox + (i + 0.5) * dx
        for k in range(nz):
            gprev[k] = 0.0
        for j in range(ny):
            y = oy + (j + 0.5) * dx
            for k in range(nz):
                fi = f[i, j, k]
                if fi <= fmin:
                    lc[j, k] = 0.0
                    ld[j, k] = 0.0
                    gprev[k] = 0.0
                    continue
                gl = gprev[k]
                gr = 0.0
                if j < ny - 1 and f[i, j + 1, k] > fmin:
                    gr = 1.0 / (np.float64(ry[i, j, k]) + np.float64(ry[i, j + 1, k]))
                hsum = 0.0
                rhs = cap[i, j, k] * ta[i, j, k]
                bb = bits[i, j, k]
                if bb & 12:
                    z = oz + (k + 0.5) * dx
                    cbits = g.contact[i, j, k]
                    for face in range(2, 4):
                        if (bb >> face) & 1:
                            afc = face_fraction(bb, fi, face) * face_area
                            ha, hc, hr, hk = face_conductances(bnd, x, y, z, temp[i, j, k], afc, face,
                                                               ((cbits >> face) & 1) == 1)
                            hsum += ha + hc + hr + hk
                            rhs += dt * (ha * t_air + hc * t_fl + hr * t_rad + hk * t_fix)
                aa = -dt * gl
                bdiag = cap[i, j, k] + dt * (gl + gr + hsum)
                cc = -dt * gr
                if j > 0:
                    denom = bdiag - aa * lc[j - 1, k]
                    ld[j, k] = (rhs - aa * ld[j - 1, k]) / denom
                else:
                    denom = bdiag
                    ld[j, k] = rhs / denom
                lc[j, k] = cc / denom
                gprev[k] = gr
        for j in range(ny - 1, -1, -1):
            y = oy + (j + 0.5) * dx
            for k in range(nz):
                fi = f[i, j, k]
                if fi <= fmin:
                    src[i, j, k] = 0.0
                    continue
                val = ld[j, k]
                if j < ny - 1:
                    val -= lc[j, k] * src[i, j + 1, k]
                src[i, j, k] = val
                bb = bits[i, j, k]
                if bb & 12:
                    z = oz + (k + 0.5) * dx
                    cbits = g.contact[i, j, k]
                    for face in range(2, 4):
                        if (bb >> face) & 1:
                            afc = face_fraction(bb, fi, face) * face_area
                            ha, hc, hr, hk = face_conductances(bnd, x, y, z, temp[i, j, k], afc, face,
                                                               ((cbits >> face) & 1) == 1)
                            la_s += dt * ha * (val - t_air)
                            lc_s += dt * hc * (val - t_fl)
                            lr_s += dt * hr * (val - t_rad)
                            lk_s += dt * hk * (val - t_fix)

    # ---- z sub-step: src -> ta
    for i in range(nx):
        x = ox + (i + 0.5) * dx
        for j in range(ny):
            y = oy + (j + 0.5) * dx
            gl = 0.0
            for k in range(nz):
                fi = f[i, j, k]
                if fi <= fmin:
                    lc[0, k] = 0.0
                    ld[0, k] = 0.0
                    gl = 0.0
                    continue
                gr = 0.0
                if k < nz - 1 and f[i, j, k + 1] > fmin:
                    gr = 1.0 / (np.float64(rz[i, j, k]) + np.float64(rz[i, j, k + 1]))
                hsum = 0.0
                rhs = cap[i, j, k] * src[i, j, k]
                bb = bits[i, j, k]
                if bb & 48:
                    z = oz + (k + 0.5) * dx
                    cbits = g.contact[i, j, k]
                    for face in range(4, 6):
                        if (bb >> face) & 1:
                            afc = face_fraction(bb, fi, face) * face_area
                            ha, hc, hr, hk = face_conductances(bnd, x, y, z, temp[i, j, k], afc, face,
                                                               ((cbits >> face) & 1) == 1)
                            hsum += ha + hc + hr + hk
                            rhs += dt * (ha * t_air + hc * t_fl + hr * t_rad + hk * t_fix)
                aa = -dt * gl
                bdiag = cap[i, j, k] + dt * (gl + gr + hsum)
                cc = -dt * gr
                if k > 0:
                    denom = bdiag - aa * lc[0, k - 1]
                    ld[0, k] = (rhs - aa * ld[0, k - 1]) / denom
                else:
                    denom = bdiag
                    ld[0, k] = rhs / denom
                lc[0, k] = cc / denom
                gl = gr
            for k in range(nz - 1, -1, -1):
                fi = f[i, j, k]
                if fi <= fmin:
                    ta[i, j, k] = 0.0
                    continue
                val = ld[0, k]
                if k < nz - 1:
                    val -= lc[0, k] * ta[i, j, k + 1]
                ta[i, j, k] = val
                bb = bits[i, j, k]
                if bb & 48:
                    z = oz + (k + 0.5) * dx
                    cbits = g.contact[i, j, k]
                    for face in range(4, 6):
                        if (bb >> face) & 1:
                            afc = face_fraction(bb, fi, face) * face_area
                            ha, hc, hr, hk = face_conductances(bnd, x, y, z, temp[i, j, k], afc, face,
                                                               ((cbits >> face) & 1) == 1)
                            la_s += dt * ha * (val - t_air)
                            lc_s += dt * hc * (val - t_fl)
                            lr_s += dt * hr * (val - t_rad)
                            lk_s += dt * hk * (val - t_fix)

    # ---- update enthalpy and temperature
    du_s = 0.0
    du_c = 0.0
    u1_s = 0.0
    u1_c = 0.0
    m_s = 0.0
    m_c = 0.0
    tm_s = 0.0
    tm_c = 0.0
    t_max = -np.inf
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                fi = f[i, j, k]
                if fi <= fmin:
                    continue
                m = rho_v * fi
                de = cap[i, j, k] * (ta[i, j, k] - temp[i, j, k])
                du_s, du_c = neumaier(du_s, du_c, de)
                h[i, j, k] = h[i, j, k] + de / m
                t_new = temperature(mat, h[i, j, k])
                temp[i, j, k] = t_new
                u1_s, u1_c = neumaier(u1_s, u1_c, m * h[i, j, k])
                m_s, m_c = neumaier(m_s, m_c, m)
                tm_s, tm_c = neumaier(tm_s, tm_c, m * t_new)
                if t_new > t_max:
                    t_max = t_new
    mass = m_s + m_c
    out[L_SRC] = e_s + e_c
    out[L_AIR] = la_s
    out[L_COOL] = lc_s
    out[L_RAD] = lr_s
    out[L_CONTACT] = lk_s
    out[L_DU] = du_s + du_c
    out[L_U0] = u0_s + u0_c
    out[L_U1] = u1_s + u1_c
    out[L_AREA] = area_s
    out[L_TSURF] = tsurf_s / area_s if area_s > 0.0 else np.nan
    out[L_TMAX] = t_max if mass > 0.0 else np.nan
    out[L_MASS] = mass
    out[L_TMEAN] = (tm_s + tm_c) / mass if mass > 0.0 else np.nan
    out[L_LOST] = lost


# ================================================================== tool / holder ladder
@njit(f8(LAD_T), **OPTS)
def ladder_energy(lad):
    total = 0.0
    comp = 0.0
    for p in range(lad.m.shape[0]):
        total, comp = neumaier(total, comp, lad.m[p] * lad.h[p])
    return total + comp


@njit(types.void(LAD_T, MAT_T, MAT_T, f8, F1), **OPTS)
def ladder_step(lad, mat_tool, mat_holder, dt, out):
    """One backward-Euler step of the tool/holder ladder (delta form, banded Cholesky).

    Exact identity: sum C dT = sum src - dt sum_c g_c (T + dT - T_c) over air, coolant and spindle
    conductances. out: LD_* terms."""
    n = lad.m.shape[0]
    bw = lad.band.shape[1] - 1
    t_air = lad.scal[0]
    t_cool = lad.scal[1]
    t_sp = lad.scal[2]
    band = lad.band
    rhs = lad.rhs
    cap = np.empty(n)
    kk = np.empty(n)
    for q in range(out.shape[0]):
        out[q] = 0.0
    u0 = ladder_energy(lad)
    e_s = 0.0
    e_c = 0.0
    for p in range(n):
        mat = mat_tool if lad.mat[p] == 0 else mat_holder
        cap[p] = lad.m[p] * prop_interp(mat.T_K, mat.cp_J_kgK, lad.T[p])
        kk[p] = prop_interp(mat.T_K, mat.k_W_mK, lad.T[p])
        for d in range(bw + 1):
            band[p, d] = 0.0
        tp = lad.T[p]
        band[p, 0] = cap[p] + dt * (lad.g_air[p] + lad.g_cool[p] + lad.g_sp[p])
        rhs[p] = lad.src[p] + dt * (lad.g_air[p] * (t_air - tp) + lad.g_cool[p] * (t_cool - tp)
                                    + lad.g_sp[p] * (t_sp - tp))
        e_s, e_c = neumaier(e_s, e_c, lad.src[p])
    for li in range(lad.link_a.shape[0]):
        a = lad.link_a[li]
        b = lad.link_b[li]
        if a > b:
            a, b = b, a
        if lad.link_kind[li] == 0:
            g = 1.0 / (lad.dz[a] / (2.0 * kk[a] * lad.area[a]) + lad.dz[b] / (2.0 * kk[b] * lad.area[b]))
        else:
            g = lad.link_g[li]
        band[a, 0] += dt * g
        band[b, 0] += dt * g
        band[b, b - a] -= dt * g
        flow = g * (lad.T[b] - lad.T[a])
        rhs[a] += dt * flow
        rhs[b] -= dt * flow
    # banded Cholesky, lower band storage band[i, d] = A[i, i - d]
    for i in range(n):
        for d in range(bw, -1, -1):
            j = i - d
            if j < 0:
                continue
            acc = band[i, d]
            lo = i - bw
            if lo < 0:
                lo = 0
            for p in range(lo, j):
                acc -= band[i, i - p] * band[j, j - p]
            if d == 0:
                band[i, 0] = math.sqrt(acc)
            else:
                band[i, d] = acc / band[j, 0]
    for i in range(n):
        acc = rhs[i]
        for d in range(1, bw + 1):
            if i - d >= 0:
                acc -= band[i, d] * rhs[i - d]
        rhs[i] = acc / band[i, 0]
    for i in range(n - 1, -1, -1):
        acc = rhs[i]
        for d in range(1, bw + 1):
            if i + d < n:
                acc -= band[i + d, d] * rhs[i + d]
        rhs[i] = acc / band[i, 0]
    la = 0.0
    lco = 0.0
    lsp = 0.0
    du_s = 0.0
    du_c = 0.0
    for p in range(n):
        dtp = rhs[p]
        tn = lad.T[p] + dtp
        la += dt * lad.g_air[p] * (tn - t_air)
        lco += dt * lad.g_cool[p] * (tn - t_cool)
        lsp += dt * lad.g_sp[p] * (tn - t_sp)
        de = cap[p] * dtp
        du_s, du_c = neumaier(du_s, du_c, de)
        lad.h[p] = lad.h[p] + de / lad.m[p]
        mat = mat_tool if lad.mat[p] == 0 else mat_holder
        lad.T[p] = temperature(mat, lad.h[p])
    out[LD_SRC] = e_s + e_c
    out[LD_AIR] = la
    out[LD_COOL] = lco
    out[LD_SP] = lsp
    out[LD_DU] = du_s + du_c
    out[LD_U0] = u0
    out[LD_U1] = ladder_energy(lad)


# ================================================================== cutting-edge field
@njit(types.UniTuple(f8, 2)(RESP_T, f8), **INLINE)
def m_index(resp, m):
    """(index, weight) of the aspect ratio m = b / a on the table's log10 m grid (clamped)."""
    lm = resp.log_m
    nm = lm.shape[0]
    x = math.log10(m)
    if x <= lm[0]:
        return 0.0, 0.0
    if x >= lm[nm - 1]:
        return float(nm - 2), 1.0
    u = (x - lm[0]) / (lm[1] - lm[0])
    i = int(u)
    if i > nm - 2:
        i = nm - 2
    return float(i), u - i


@njit(types.UniTuple(f8, 2)(RESP_T, f8, i8, f8), **INLINE)
def phi_pair(resp, fo, im, wm):
    """Dimensionless step response (centre, area mean) Phi(Fo, m); 1-D limit 2 sqrt(Fo/pi) below
    the table, the asymptote Phi_inf - 2 m / (pi^1.5 sqrt Fo) above it, log-log interpolation."""
    if fo <= 0.0:
        return 0.0, 0.0
    lf = resp.log_fo
    nf = lf.shape[0]
    x = math.log10(fo)
    if x <= lf[0]:
        one_d = 2.0 * math.sqrt(fo / math.pi)
        return one_d, one_d
    if x >= lf[nf - 1]:
        inf_c = (1.0 - wm) * resp.phi_inf_c[im] + wm * resp.phi_inf_c[im + 1]
        inf_m = (1.0 - wm) * resp.phi_inf_m[im] + wm * resp.phi_inf_m[im + 1]
        lm = (1.0 - wm) * resp.log_m[im] + wm * resp.log_m[im + 1]
        tail = 2.0 * math.exp(LN10 * lm) / (math.pi * SQRT_PI * math.sqrt(fo))
        return inf_c - tail, inf_m - tail
    u = (x - lf[0]) / (lf[1] - lf[0])
    i = int(u)
    if i > nf - 2:
        i = nf - 2
    t = u - i
    w0 = 1.0 - wm
    c0 = w0 * resp.log_phi_c[im, i] + wm * resp.log_phi_c[im + 1, i]
    c1 = w0 * resp.log_phi_c[im, i + 1] + wm * resp.log_phi_c[im + 1, i + 1]
    m0 = w0 * resp.log_phi_m[im, i] + wm * resp.log_phi_m[im + 1, i]
    m1 = w0 * resp.log_phi_m[im, i + 1] + wm * resp.log_phi_m[im + 1, i + 1]
    return math.exp(LN10 * (c0 + t * (c1 - c0))), math.exp(LN10 * (m0 + t * (m1 - m0)))


@njit(types.UniTuple(f8, 2)(RESP_T, f8, f8, f8, f8, i8, f8), **INLINE)
def unit_pair(resp, t, a, k, alpha, im, wm):
    """(centre, mean) rise in K per W/m2 after t of a unit flux on the mirrored contact (half length a)."""
    if t <= 0.0:
        return 0.0, 0.0
    pc, pm = phi_pair(resp, alpha * t / (a * a), im, wm)
    scale = a / k
    return scale * pc, scale * pm


@njit(types.UniTuple(f8, 2)(RESP_T, f8, f8, f8, f8, f8, i8, f8, F1, F1), **INLINE)
def window_pair(resp, t, th, a, k, alpha, im, wm, glx, glw):
    """Integrals of the (centre, mean) unit responses over [t - th, t] (Gauss-Legendre; 0 below t = 0)."""
    lo = t - th
    if lo < 0.0:
        lo = 0.0
    if t <= lo:
        return 0.0, 0.0
    half = 0.5 * (t - lo)
    mid = 0.5 * (t + lo)
    acc_c = 0.0
    acc_m = 0.0
    for q in range(glx.shape[0]):
        rc, rm = unit_pair(resp, mid + half * glx[q], a, k, alpha, im, wm)
        acc_c += glw[q] * rc
        acc_m += glw[q] * rm
    return acc_c * half, acc_m * half


@njit(types.UniTuple(f8, 2)(RESP_T, f8, f8, f8, f8, f8, f8, f8, i8, f8, i8, F1, F1), **INLINE)
def periodic_pair(resp, tau, th, period, ncyc, a, k, alpha, im, wm, ndirect, glx, glw):
    """(centre, mean) rise per unit flux at phase tau of the current cycle after ncyc cycles of
    heating for th in every period (Stephenson & Ali 1992 superposition); the cycles beyond
    ``ndirect`` by the midpoint integral of the response (error O(n^-2) of the tail)."""
    tot_c = 0.0
    tot_m = 0.0
    nd = ndirect
    if ncyc < nd:
        nd = int(ncyc)
        if nd < 1:
            nd = 1
    for n in range(nd):
        t1 = tau + n * period
        hc, hm = unit_pair(resp, t1, a, k, alpha, im, wm)
        oc, om = unit_pair(resp, t1 - th, a, k, alpha, im, wm)
        tot_c += hc - oc
        tot_m += hm - om
    if ncyc > nd:
        lo_c, lo_m = window_pair(resp, tau + (nd - 0.5) * period, th, a, k, alpha, im, wm, glx, glw)
        hi_c, hi_m = window_pair(resp, tau + (ncyc - 0.5) * period, th, a, k, alpha, im, wm, glx, glw)
        tot_c += (hi_c - lo_c) / period
        tot_m += (hi_m - lo_m) / period
    return tot_c, tot_m


@njit(f8(f8, f8), **INLINE)
def disc_depth_factor(depth, radius):
    """Mean over [0, depth] of the axial field of a uniformly heated disc of ``radius`` relative to
    its centre value: (sqrt(R^2 + z^2) - z) / R (Carslaw & Jaeger 1959, 10.5)."""
    if depth <= 0.0 or radius <= 0.0:
        return 1.0
    r2 = radius * radius
    integral = 0.5 * depth * math.sqrt(r2 + depth * depth) + 0.5 * r2 * math.asinh(depth / radius) \
        - 0.5 * depth * depth
    return integral / (depth * radius)


@njit(types.void(EIN_T, RESP_T, F1, F1, EOUT_T), **OPTS)
def edge_temperatures(inp, resp, glx, glw, out):
    """Edge, rake-mean and flank-mean temperatures at the in-cut quadrature points and the cycle
    amplitude at the edge, per element: body temperature + wedge * local rise (periodic
    superposition on the quarter space minus the far field the 1-D body already holds).

    ``rake_entry`` / ``flank_entry``: the contact means when the edge enters the cut (the end of the
    cooling interval of an interrupted element: what is left of the earlier engagements; a continuous
    element's own steady rise is excluded, so its entry value is the body temperature): the background
    temperature of the stationary-source heat partition (cnctwin.mechanics), which adds the rise of the
    engagement's own heat itself.

    With ``scal[2] = EDGE_OUT_DEVIATION`` the outputs are instead the periodic deviation from the
    cycle mean, wedge * (theta(tau) - <theta>), <theta> = q W(N T) / T with W(t) the integral of the
    unit response over [t - t_h, t] (the exact cycle mean of the superposition): the 3-D field of the
    cutting region holds the mean, this adds the oscillation. Continuous elements then deviate by 0."""
    nt, ne = inp.mode.shape
    nq = inp.xi.shape[0]
    ndirect = int(inp.scal[0])
    deviation = inp.scal.shape[0] > 2 and int(inp.scal[2]) == EDGE_OUT_DEVIATION
    for j in range(nt):
        for e in range(ne):
            tb = inp.T_body[j, e]
            mode = inp.mode[j, e]
            out.amp[j, e] = 0.0
            base = 0.0 if deviation else tb
            out.rake_entry[j, e] = base
            out.flank_entry[j, e] = base
            if mode == EDGE_NONE or (deviation and mode == EDGE_CONTINUOUS):
                for q in range(nq):
                    out.rake_q[j, e, q] = base
                    out.peak_q[j, e, q] = base
                    out.flank_q[j, e, q] = base
                continue
            k = inp.k[j, e]
            alpha = inp.alpha[j, e]
            b = inp.b[j, e]
            ar = inp.a_r[j, e]
            af = inp.a_f[j, e]
            qr = inp.q_r[j, e]
            qf = inp.q_f[j, e]
            wedge = inp.wedge[j, e]
            rfar = inp.r_far[j, e]
            imr_f, wmr = m_index(resp, b / ar)
            imf_f, wmf = m_index(resp, b / af)
            imr = int(imr_f)
            imf = int(imf_f)
            if mode == EDGE_INTERRUPTED:
                th = inp.t_h[j, e]
                period = inp.period[j, e]
                ncyc = inp.n_cyc[j, e]
                duty = th / period
                t_total = ncyc * period
            else:
                th = inp.t_eng[j, e]
                period = th
                ncyc = 1.0
                duty = 1.0
                t_total = th
            far_scale = duty * math.erfc(rfar / (2.0 * math.sqrt(alpha * t_total))) / (math.pi * k * rfar)
            far_r = qr * ar * 2.0 * b * far_scale
            far_f = qf * af * 2.0 * b * far_scale
            mean_rc = 0.0
            mean_rm = 0.0
            mean_fc = 0.0
            mean_fm = 0.0
            if deviation:
                # cycle means of the periodic solution replace the far-field subtraction
                far_r = 0.0
                far_f = 0.0
                wc, wm_ = window_pair(resp, t_total, th, ar, k, alpha, imr, wmr, glx, glw)
                mean_rc = wc / period
                mean_rm = wm_ / period
                wc, wm_ = window_pair(resp, t_total, th, af, k, alpha, imf, wmf, glx, glw)
                mean_fc = wc / period
                mean_fm = wm_ / period
            s_rf = disc_depth_factor(ar, math.sqrt(4.0 * af * b / math.pi))
            s_fr = disc_depth_factor(af, math.sqrt(4.0 * ar * b / math.pi))
            hot = 0.0
            n_phase = nq + 2 if mode == EDGE_INTERRUPTED else nq
            for qq in range(n_phase):
                # phases: the nq Gauss points of the heating interval, then its end (hottest) and the
                # end of the cooling interval (coldest) for the cycle amplitude
                if mode == EDGE_INTERRUPTED:
                    if qq < nq:
                        tau = inp.xi[qq] * th
                    elif qq == nq:
                        tau = th
                    else:
                        tau = period
                    rc, rm = periodic_pair(resp, tau, th, period, ncyc, ar, k, alpha, imr, wmr, ndirect, glx, glw)
                    fc, fm = periodic_pair(resp, tau, th, period, ncyc, af, k, alpha, imf, wmf, ndirect, glx, glw)
                else:
                    rc, rm = unit_pair(resp, th, ar, k, alpha, imr, wmr)
                    fc, fm = unit_pair(resp, th, af, k, alpha, imf, wmf)
                if qq >= nq:
                    edge_now = qr * rc + qf * fc
                    if qq == nq:
                        hot = edge_now
                    else:
                        out.amp[j, e] = wedge * (hot - edge_now)
                        # the end of the cooling interval: the contact means when the edge enters again
                        if deviation:
                            ent_r = qr * (rc - mean_rc)
                            ent_f = qf * (fc - mean_fc)
                            out.rake_entry[j, e] = wedge * (qr * (rm - mean_rm) + s_rf * ent_f)
                            out.flank_entry[j, e] = wedge * (qf * (fm - mean_fm) + s_fr * ent_r)
                        else:
                            ent_r = qr * rc - far_r
                            ent_f = qf * fc - far_f
                            rake_in = qr * rm - far_r + s_rf * ent_f
                            flank_in = qf * fm - far_f + s_fr * ent_r
                            out.rake_entry[j, e] = tb + wedge * (rake_in if rake_in > 0.0 else 0.0)
                            out.flank_entry[j, e] = tb + wedge * (flank_in if flank_in > 0.0 else 0.0)
                    continue
                if deviation:
                    edge_r = qr * (rc - mean_rc)
                    edge_f = qf * (fc - mean_fc)
                    out.rake_q[j, e, qq] = wedge * (qr * (rm - mean_rm) + s_rf * edge_f)
                    out.flank_q[j, e, qq] = wedge * (qf * (fm - mean_fm) + s_fr * edge_r)
                    out.peak_q[j, e, qq] = wedge * (edge_r + edge_f)
                    continue
                edge_r = qr * rc - far_r
                edge_f = qf * fc - far_f
                rake = qr * rm - far_r + s_rf * edge_f
                flank = qf * fm - far_f + s_fr * edge_r
                peak = edge_r + edge_f
                out.rake_q[j, e, qq] = tb + wedge * (rake if rake > 0.0 else 0.0)
                out.flank_q[j, e, qq] = tb + wedge * (flank if flank > 0.0 else 0.0)
                out.peak_q[j, e, qq] = tb + wedge * (peak if peak > 0.0 else 0.0)


# ================================================================== moving rectangle source
@njit(f8(f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, F1, F1), **OPTS)
def moving_rect_rise(q, l, w, v, k, alpha, t, x, y, z, glx, glw):
    """Rise at (x, y, z) of a half space under a uniform flux q on |x'| < l, |y'| < w moving at v
    along +x, switched on t ago (x relative to the current source centre, z depth):

        theta = (q alpha / k) int_0^t (pi alpha tau)^-1/2 exp(-z^2 / 4 alpha tau) X(tau) Y(tau) dtau

    with X, Y the erf factors of the strip (Jaeger 1942; Carslaw & Jaeger 1959). The integral is
    split at the times the point entered and left the footprint, at the diffusion cut-off and on
    a geometric ladder of l^2 / alpha, each piece by Gauss-Legendre (the first with tau = s^2)."""
    if t <= 0.0 or q == 0.0:
        return 0.0
    cut = t
    pts = np.empty(48)
    npts = 0
    pts[npts] = 0.0
    npts += 1
    if v > 0.0:
        disc = 64.0 * alpha - 4.0 * v * (x - l)
        if disc < 0.0:
            return 0.0
        root = (8.0 * math.sqrt(alpha) + math.sqrt(disc)) / (2.0 * v)
        if root * root < cut:
            cut = root * root
        for cand in ((-l - x) / v, (l - x) / v):
            if 0.0 < cand < cut:
                pts[npts] = cand
                npts += 1
    scale = min(l, w) if w > 0.0 else l
    tau_s = 0.25 * scale * scale / alpha
    while tau_s < cut and npts < 46:
        pts[npts] = tau_s
        npts += 1
        tau_s *= 4.0
    pts[npts] = cut
    npts += 1
    for a in range(1, npts):          # insertion sort (few points)
        key = pts[a]
        b = a - 1
        while b >= 0 and pts[b] > key:
            pts[b + 1] = pts[b]
            b -= 1
        pts[b + 1] = key
    sq_pa = math.sqrt(math.pi * alpha)
    total = 0.0
    for seg in range(npts - 1):
        lo = pts[seg]
        hi = pts[seg + 1]
        if hi <= lo:
            continue
        acc = 0.0
        for g in range(glx.shape[0]):
            if lo == 0.0:
                u = 0.5 * (glx[g] + 1.0)
                tau = hi * u * u
                jac = hi * u
            else:
                tau = 0.5 * (lo + hi) + 0.5 * (hi - lo) * glx[g]
                jac = 0.5 * (hi - lo)
            if tau <= 0.0:
                continue
            sig = 2.0 * math.sqrt(alpha * tau)
            xf = 0.5 * (math.erf((x + v * tau + l) / sig) - math.erf((x + v * tau - l) / sig))
            if w > 0.0:
                yf = 0.5 * (math.erf((w - y) / sig) + math.erf((w + y) / sig))
            else:
                yf = 1.0
            zf = math.exp(-z * z / (sig * sig)) if z != 0.0 else 1.0
            acc += glw[g] * jac * xf * yf * zf / (sq_pa * math.sqrt(tau))
        total += acc
    return q * alpha / k * total


# ================================================================== cutting-region field (3-D)
@njit(types.void(I1, I1, F1, F1, F1, F1), **INLINE)
def csr_sym_matvec(row_ptr, col, val, dia, x, y):
    """y = (D + L + L^T) x, L the strictly lower CSR matrix (row_ptr, col, val), D = diag(dia)."""
    n = dia.shape[0]
    for i in range(n):
        y[i] = dia[i] * x[i]
    for i in range(n):
        xi = x[i]
        acc = 0.0
        for idx in range(row_ptr[i], row_ptr[i + 1]):
            j = col[idx]
            v = val[idx]
            acc += v * x[j]
            y[j] += v * xi
        y[i] += acc


@njit(types.void(FMESH_T, FSYS_T), **INLINE)
def block_factor(fm, fs):
    """Banded Cholesky of every diagonal block (band[i, d] = A[i, i - d] on entry, L on exit; the
    blocks are contiguous index ranges fm.blk_ptr, banded within band.shape[1] - 1)."""
    band = fs.band
    bw = band.shape[1] - 1
    for blk in range(fm.blk_ptr.shape[0] - 1):
        s0 = fm.blk_ptr[blk]
        s1 = fm.blk_ptr[blk + 1]
        for i in range(s0, s1):
            for d in range(bw, -1, -1):
                j = i - d
                if j < s0:
                    continue
                acc = band[i, d]
                lo = i - bw
                if lo < s0:
                    lo = s0
                for p in range(lo, j):
                    acc -= band[i, i - p] * band[j, j - p]
                if d == 0:
                    band[i, 0] = math.sqrt(acc) if acc > 0.0 else math.sqrt(fs.dia[i])
                else:
                    band[i, d] = acc / band[j, 0]


@njit(types.void(F2, i8, i8, F1), **INLINE)
def band_solve(band, s0, s1, x):
    """x[s0:s1] = (L L^T)^-1 x[s0:s1] with the banded factor of one block."""
    bw = band.shape[1] - 1
    for i in range(s0, s1):
        acc = x[i]
        for d in range(1, bw + 1):
            if i - d >= s0:
                acc -= band[i, d] * x[i - d]
        x[i] = acc / band[i, 0]
    for i in range(s1 - 1, s0 - 1, -1):
        acc = x[i]
        for d in range(1, bw + 1):
            if i + d < s1:
                acc -= band[i + d, d] * x[i + d]
        x[i] = acc / band[i, 0]


@njit(types.void(FMESH_T, FSYS_T, F1, F1), **INLINE)
def block_sgs(fm, fs, r, z):
    """z = M^-1 r for the block symmetric Gauss-Seidel preconditioner M = (D + L) D^-1 (D + L^T), D the
    exactly factored diagonal blocks (symmetric positive definite: valid for conjugate gradients)."""
    row_ptr = fm.row_ptr
    col = fm.col
    val = fs.val
    w = fs.acc
    nb = fm.blk_ptr.shape[0] - 1
    for blk in range(nb):
        s0 = fm.blk_ptr[blk]
        s1 = fm.blk_ptr[blk + 1]
        for i in range(s0, s1):
            acc = r[i]
            for idx in range(row_ptr[i], row_ptr[i + 1]):
                j = col[idx]
                if j < s0:
                    acc -= val[idx] * z[j]
            z[i] = acc
        band_solve(fs.band, s0, s1, z)
    for i in range(w.shape[0]):
        w[i] = 0.0
    for blk in range(nb - 1, -1, -1):
        s0 = fm.blk_ptr[blk]
        s1 = fm.blk_ptr[blk + 1]
        band_solve(fs.band, s0, s1, w)
        for i in range(s0, s1):
            z[i] -= w[i]
        for i in range(s0, s1):
            zi = z[i]
            for idx in range(row_ptr[i], row_ptr[i + 1]):
                j = col[idx]
                if j < s0:
                    w[j] += val[idx] * zi


@njit(f8(F1, F1), **INLINE)
def dot(a, b):
    acc = 0.0
    for i in range(a.shape[0]):
        acc += a[i] * b[i]
    return acc


@njit(types.UniTuple(f8, 2)(FMESH_T, FSYS_T, f8, i8), **OPTS)
def field_pcg(fm, fs, tol, max_iter):
    """Solve (D + L + L^T) x = b (fs.dia, fs.val, fs.b) by conjugate gradients with the block symmetric
    Gauss-Seidel preconditioner until |b - A x| <= tol |b| (the residual is recomputed from x at the end,
    and the iteration restarted from x when the recursive residual drifted). The start is the x on
    entry (the previous step's increment) when its residual is below |b|, else 0. Returns (iterations,
    relative residual). Fixed operation order: deterministic (x is part of the field's state)."""
    x = fs.x
    r = fs.r
    z = fs.z
    p = fs.p
    q = fs.q
    b = fs.b
    n = b.shape[0]
    bn = math.sqrt(dot(b, b))
    if bn == 0.0:
        for i in range(n):
            x[i] = 0.0
        return 0.0, 0.0
    band = fs.band
    bw = band.shape[1] - 1
    for blk in range(fm.blk_ptr.shape[0] - 1):
        s0 = fm.blk_ptr[blk]
        for i in range(s0, fm.blk_ptr[blk + 1]):
            for d in range(bw + 1):
                band[i, d] = 0.0
            band[i, 0] = fs.dia[i]
            for idx in range(fm.row_ptr[i], fm.row_ptr[i + 1]):
                j = fm.col[idx]
                if j >= s0:
                    band[i, i - j] = fs.val[idx]
    block_factor(fm, fs)
    # warm start from the previous increment (on entry in x; zero after a reset) when it is the better guess
    csr_sym_matvec(fm.row_ptr, fm.col, fs.val, fs.dia, x, q)
    for i in range(n):
        r[i] = b[i] - q[i]
    if dot(r, r) >= bn * bn:
        for i in range(n):
            x[i] = 0.0
            r[i] = b[i]
    it = 0
    rel = 1.0
    for _restart in range(4):
        block_sgs(fm, fs, r, z)
        rz = dot(r, z)
        for i in range(n):
            p[i] = z[i]
        while it < max_iter:
            csr_sym_matvec(fm.row_ptr, fm.col, fs.val, fs.dia, p, q)
            pq = dot(p, q)
            if pq <= 0.0:
                break
            alpha = rz / pq
            for i in range(n):
                x[i] += alpha * p[i]
                r[i] -= alpha * q[i]
            it += 1
            if math.sqrt(dot(r, r)) <= tol * bn:
                break
            block_sgs(fm, fs, r, z)
            rz_new = dot(r, z)
            beta = rz_new / rz
            rz = rz_new
            for i in range(n):
                p[i] = z[i] + beta * p[i]
        csr_sym_matvec(fm.row_ptr, fm.col, fs.val, fs.dia, x, q)
        for i in range(n):
            r[i] = b[i] - q[i]
        rel = math.sqrt(dot(r, r)) / bn
        if rel <= 4.0 * tol or it >= max_iter:
            break
    return float(it), rel


@njit(types.void(FMESH_T, FSYS_T, LAD_T, MAT_T, MAT_T, f8, F1), **OPTS)
def field_step(fm, fs, lad, mat_tool, mat_holder, dt, out):
    """One backward-Euler step of the cutting-region field and the tool ladder as one system.

    Unknowns: the temperature change of every field cell and of every ladder node with
    ``fs.active`` (ladder nodes that are views of the field keep an identity row). Couplings:
    internal faces A / (d_a / k_a + d_b / k_b), exposed faces to air and coolant
    A_w / (d / k + 1 / h) each, coupled faces to their ladder node A / (d / k + r_c + l_k / k_node),
    the ladder's own links and its air, coolant and spindle/turret conductances (as ladder_step);
    properties at the start of the step. The symmetric M-matrix is solved by preconditioned conjugate
    gradients (field_pcg, block symmetric Gauss-Seidel) to the relative residual ``fs.scal[FS_TOL]``.

    Identity: sum C dT = sources - dt sum_c g_c (T + dT - T_c) - sum(b - A dT) (the last term is the
    solver's residual, booked by the ledger with the rest). out: FD_* terms."""
    n = fs.T.shape[0]
    nn = lad.m.shape[0]
    t_air = lad.scal[0]
    t_cool = lad.scal[1]
    t_sp = lad.scal[2]
    cap = fs.cap
    kk = fs.kk
    dia = fs.dia
    val = fs.val
    b = fs.b
    x = fs.x
    for q in range(out.shape[0]):
        out[q] = 0.0
    for idx in range(val.shape[0]):
        val[idx] = 0.0
    u_s = 0.0
    u_c = 0.0
    e_s = 0.0
    e_c = 0.0
    tg = mat_tool.T_K
    for i in range(n):
        ti = fs.T[i]
        cap[i] = fm.m[i] * prop_interp(tg, mat_tool.cp_J_kgK, ti)
        kk[i] = prop_interp(tg, mat_tool.k_W_mK, ti)
        dia[i] = cap[i]
        b[i] = fs.src[i]
        u_s, u_c = neumaier(u_s, u_c, fm.m[i] * fs.h[i])
        e_s, e_c = neumaier(e_s, e_c, fs.src[i])
    for p in range(nn):
        g_i = n + p
        if fs.active[p] == 0:
            cap[g_i] = 1.0
            kk[g_i] = 1.0
            dia[g_i] = 1.0
            b[g_i] = 0.0
            continue
        mat = mat_tool if lad.mat[p] == 0 else mat_holder
        tp = lad.T[p]
        cap[g_i] = lad.m[p] * prop_interp(mat.T_K, mat.cp_J_kgK, tp)
        kk[g_i] = prop_interp(mat.T_K, mat.k_W_mK, tp)
        dia[g_i] = cap[g_i] + dt * (lad.g_air[p] + lad.g_cool[p] + lad.g_sp[p])
        b[g_i] = lad.src[p] + dt * (lad.g_air[p] * (t_air - tp) + lad.g_cool[p] * (t_cool - tp)
                                    + lad.g_sp[p] * (t_sp - tp))
        u_s, u_c = neumaier(u_s, u_c, lad.m[p] * lad.h[p])
        e_s, e_c = neumaier(e_s, e_c, lad.src[p])
    for li in range(fm.link_a.shape[0]):
        a = fm.link_a[li]
        c = fm.link_b[li]
        g = fm.link_area[li] / (fm.link_da[li] / kk[a] + fm.link_db[li] / kk[c])
        dia[a] += dt * g
        dia[c] += dt * g
        val[fm.link_slot[li]] -= dt * g
        flow = dt * g * (fs.T[c] - fs.T[a])
        b[a] += flow
        b[c] -= flow
    nf = fm.face_cell.shape[0]
    for f in range(nf):
        c = fm.face_cell[f]
        node = fm.face_node[f]
        if node >= 0:
            if fs.active[node] == 0:
                continue
            g_i = n + node
            g = fm.face_area[f] / (fm.face_d[f] / kk[c] + fm.face_rc[f] + fm.face_lk[f] / kk[g_i])
            dia[c] += dt * g
            dia[g_i] += dt * g
            val[fm.face_slot[f]] -= dt * g
            flow = dt * g * (lad.T[node] - fs.T[c])
            b[c] += flow
            b[g_i] -= flow
            continue
        ga = 0.0
        if fs.fa_w[f] > 0.0 and fs.fa_h[f] > 0.0:
            ga = fs.fa_w[f] / (fm.face_d[f] / kk[c] + 1.0 / fs.fa_h[f])
        gc = 0.0
        if fs.fc_w[f] > 0.0 and fs.fc_h[f] > 0.0:
            gc = fs.fc_w[f] / (fm.face_d[f] / kk[c] + 1.0 / fs.fc_h[f])
        dia[c] += dt * (ga + gc)
        b[c] += dt * (ga * (t_air - fs.T[c]) + gc * (t_cool - fs.T[c]))
    for li in range(lad.link_a.shape[0]):
        slot = fm.lad_slot[li]
        if slot < 0:
            continue
        a = lad.link_a[li]
        c = lad.link_b[li]
        if a > c:
            a, c = c, a
        if lad.link_kind[li] == 0:
            g = 1.0 / (lad.dz[a] / (2.0 * kk[n + a] * lad.area[a]) + lad.dz[c] / (2.0 * kk[n + c] * lad.area[c]))
        else:
            g = lad.link_g[li]
        dia[n + a] += dt * g
        dia[n + c] += dt * g
        val[slot] -= dt * g
        flow = dt * g * (lad.T[c] - lad.T[a])
        b[n + a] += flow
        b[n + c] -= flow
    iters, rel = field_pcg(fm, fs, fs.scal[FS_TOL], int(fs.scal[FS_MAX_ITER]))
    # losses with the temperatures at the end of the step (T + dT), before the state is updated
    la = 0.0
    lco = 0.0
    lsp = 0.0
    lcp = 0.0
    for f in range(nf):
        c = fm.face_cell[f]
        tc = fs.T[c] + x[c]
        node = fm.face_node[f]
        if node >= 0:
            if fs.active[node] == 0:
                continue
            g_i = n + node
            g = fm.face_area[f] / (fm.face_d[f] / kk[c] + fm.face_rc[f] + fm.face_lk[f] / kk[g_i])
            lcp += dt * g * (tc - (lad.T[node] + x[g_i]))
            continue
        if fs.fa_w[f] > 0.0 and fs.fa_h[f] > 0.0:
            la += dt * fs.fa_w[f] / (fm.face_d[f] / kk[c] + 1.0 / fs.fa_h[f]) * (tc - t_air)
        if fs.fc_w[f] > 0.0 and fs.fc_h[f] > 0.0:
            lco += dt * fs.fc_w[f] / (fm.face_d[f] / kk[c] + 1.0 / fs.fc_h[f]) * (tc - t_cool)
    for p in range(nn):
        if fs.active[p] == 0:
            continue
        tn = lad.T[p] + x[n + p]
        la += dt * lad.g_air[p] * (tn - t_air)
        lco += dt * lad.g_cool[p] * (tn - t_cool)
        lsp += dt * lad.g_sp[p] * (tn - t_sp)
    du_s = 0.0
    du_c = 0.0
    v_s = 0.0
    v_c = 0.0
    for i in range(n):
        de = cap[i] * x[i]
        du_s, du_c = neumaier(du_s, du_c, de)
        fs.h[i] = fs.h[i] + de / fm.m[i]
        fs.T[i] = temperature(mat_tool, fs.h[i])
        v_s, v_c = neumaier(v_s, v_c, fm.m[i] * fs.h[i])
    for p in range(nn):
        if fs.active[p] == 0:
            continue
        de = cap[n + p] * x[n + p]
        du_s, du_c = neumaier(du_s, du_c, de)
        lad.h[p] = lad.h[p] + de / lad.m[p]
        mat = mat_tool if lad.mat[p] == 0 else mat_holder
        lad.T[p] = temperature(mat, lad.h[p])
        v_s, v_c = neumaier(v_s, v_c, lad.m[p] * lad.h[p])
    out[FD_SRC] = e_s + e_c
    out[FD_AIR] = la
    out[FD_COOL] = lco
    out[FD_SP] = lsp
    out[FD_DU] = du_s + du_c
    out[FD_U0] = u_s + u_c
    out[FD_U1] = v_s + v_c
    out[FD_ITER] = iters
    out[FD_RES] = rel
    out[FD_COUPLE] = lcp
