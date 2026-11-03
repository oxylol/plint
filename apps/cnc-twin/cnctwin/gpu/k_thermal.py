"""GPU twin of the workpiece finite-volume step ``cnctwin.thermal.kernels.work_step`` (docs/models/gpu.md
"Thermal step").

The CPU kernel is a locally one-dimensional backward-Euler splitting: a per-cell pass (capacity, half-cell
resistances, exposure bits), then x, y and z sub-steps, each a set of independent tridiagonal (Thomas)
solves along the lattice lines, then a per-cell enthalpy update. The GPU runs the same work with one
thread per cell (passes 0 and 4) and one thread per line (sub-steps), every line in the CPU's own
operation order, and calls thermal's own njit helpers (``prop_interp``, ``face_fraction``,
``face_conductances``, ``temperature``) compiled as device twins. Cell values are therefore bit-identical
to the CPU kernel (FMA contraction off; division and square root correctly rounded).

The ledger terms (``out``, L_*) are sums over cells in a fixed order; a parallel reduction would change
their rounding. They are recomputed on the host by :func:`ledger` from the GPU's per-cell results in
exactly the CPU kernel's order (and with the CPU's Neumaier compensation), so ``out`` is bit-identical
too. The scratch rows ``line_c``, ``line_d`` and ``gprev`` of WorkScratch are not written by the GPU path
(the CPU kernel leaves the last line's coefficients there; nothing reads them).
"""

from __future__ import annotations

import numpy as np
from numba import njit, types

from ..thermal import abi as TA
from ..thermal.abi import MatTable, WorkBoundary
from ..thermal.kernels import (BND_T, FULL_CELL, GRID_T, MAT_T, SCR_T, face_conductances, face_fraction, neumaier,
                               prop_interp, temperature)
from . import device as D
from .twins import builder

B_T_AIR, B_T_FLUID, B_T_RAD, B_T_FIX, B_FMIN = TA.B_T_AIR, TA.B_T_FLUID, TA.B_T_RAD, TA.B_T_FIX, TA.B_FMIN
L_SRC, L_AIR, L_COOL, L_RAD, L_CONTACT, L_DU = TA.L_SRC, TA.L_AIR, TA.L_COOL, TA.L_RAD, TA.L_CONTACT, TA.L_DU
L_U0, L_U1, L_AREA, L_TSURF, L_TMAX, L_MASS = TA.L_U0, TA.L_U1, TA.L_AREA, TA.L_TSURF, TA.L_TMAX, TA.L_MASS
L_TMEAN, L_LOST = TA.L_TMEAN, TA.L_LOST
ONE_THIRD = 1.0 / 3.0


# ============================================================================ device kernels (CUDA source)
def k_cells(f, temp, T_K, k_W, cp, H, alpha, rho, geom, scal, cap, rx, ry, rz, bits):
    """Pass 0 of work_step, one thread per cell: capacity, half-cell resistances, exposure bits."""
    t = cuda.grid(1)
    nx, ny, nz = f.shape
    if t >= nx * ny * nz:
        return
    i = t // (ny * nz)
    j = (t // nz) % ny
    k = t % nz
    dx = geom[3]
    rho_v = rho[0] * dx * dx * dx
    fmin = scal[B_FMIN]
    fi = f[i, j, k]
    if fi <= fmin:
        cap[i, j, k] = 0.0
        rx[i, j, k] = 0.0
        ry[i, j, k] = 0.0
        rz[i, j, k] = 0.0
        bits[i, j, k] = 0
        return
    ti = temp[i, j, k]
    m = rho_v * fi
    cap[i, j, k] = m * prop_interp(T_K, cp, ti)
    ki = prop_interp(T_K, k_W, ti)
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
    wx = ONE_THIRD
    wy = ONE_THIRD
    wz = ONE_THIRD
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


def k_sweep_x(f, temp, contact, geom, h_air, near, wet, wet_geom, scal, src, cap, rx, bits, lc, ld, out, dt):
    """x sub-step of work_step, one thread per line (j, k): T^n (+ src) -> out."""
    t = cuda.grid(1)
    nx, ny, nz = f.shape
    if t >= ny * nz:
        return
    j = t // nz
    k = t % nz
    bnd = WorkBoundary(h_air, near, wet, wet_geom, scal)
    ox = geom[0]
    oy = geom[1]
    oz = geom[2]
    dx = geom[3]
    face_area = dx * dx
    fmin = scal[B_FMIN]
    t_air = scal[B_T_AIR]
    t_fl = scal[B_T_FLUID]
    t_rad = scal[B_T_RAD]
    t_fix = scal[B_T_FIX]
    y = oy + (j + 0.5) * dx
    gprev = 0.0
    for i in range(nx):
        x = ox + (i + 0.5) * dx
        fi = f[i, j, k]
        if fi <= fmin:
            lc[i, j, k] = 0.0
            ld[i, j, k] = 0.0
            gprev = 0.0
            continue
        gl = gprev
        gr = 0.0
        if i < nx - 1 and f[i + 1, j, k] > fmin:
            gr = 1.0 / (np.float64(rx[i, j, k]) + np.float64(rx[i + 1, j, k]))
        ti = temp[i, j, k]
        hsum = 0.0
        rhs = cap[i, j, k] * ti + src[i, j, k]
        bb = bits[i, j, k]
        if bb & 3:
            z = oz + (k + 0.5) * dx
            cbits = contact[i, j, k]
            for face in range(2):
                if (bb >> face) & 1:
                    afc = face_fraction(bb, fi, face) * face_area
                    is_contact = ((cbits >> face) & 1) == 1
                    ha, hc, hr, hk = face_conductances(bnd, x, y, z, ti, afc, face, is_contact)
                    hsum += ha + hc + hr + hk
                    rhs += dt * (ha * t_air + hc * t_fl + hr * t_rad + hk * t_fix)
        aa = -dt * gl
        bdiag = cap[i, j, k] + dt * (gl + gr + hsum)
        cc = -dt * gr
        if i > 0:
            denom = bdiag - aa * lc[i - 1, j, k]
            ld[i, j, k] = (rhs - aa * ld[i - 1, j, k]) / denom
        else:
            denom = bdiag
            ld[i, j, k] = rhs / denom
        lc[i, j, k] = cc / denom
        gprev = gr
    for i in range(nx - 1, -1, -1):
        fi = f[i, j, k]
        if fi <= fmin:
            out[i, j, k] = 0.0
            continue
        val = ld[i, j, k]
        if i < nx - 1:
            val -= lc[i, j, k] * out[i + 1, j, k]
        out[i, j, k] = val


def k_sweep_y(f, temp, contact, geom, h_air, near, wet, wet_geom, scal, inp, cap, ry, bits, lc, ld, out, dt):
    """y sub-step, one thread per line (i, k): inp (the x result) -> out."""
    t = cuda.grid(1)
    nx, ny, nz = f.shape
    if t >= nx * nz:
        return
    i = t // nz
    k = t % nz
    bnd = WorkBoundary(h_air, near, wet, wet_geom, scal)
    ox = geom[0]
    oy = geom[1]
    oz = geom[2]
    dx = geom[3]
    face_area = dx * dx
    fmin = scal[B_FMIN]
    t_air = scal[B_T_AIR]
    t_fl = scal[B_T_FLUID]
    t_rad = scal[B_T_RAD]
    t_fix = scal[B_T_FIX]
    x = ox + (i + 0.5) * dx
    gprev = 0.0
    for j in range(ny):
        y = oy + (j + 0.5) * dx
        fi = f[i, j, k]
        if fi <= fmin:
            lc[i, j, k] = 0.0
            ld[i, j, k] = 0.0
            gprev = 0.0
            continue
        gl = gprev
        gr = 0.0
        if j < ny - 1 and f[i, j + 1, k] > fmin:
            gr = 1.0 / (np.float64(ry[i, j, k]) + np.float64(ry[i, j + 1, k]))
        hsum = 0.0
        rhs = cap[i, j, k] * inp[i, j, k]
        bb = bits[i, j, k]
        if bb & 12:
            z = oz + (k + 0.5) * dx
            cbits = contact[i, j, k]
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
            denom = bdiag - aa * lc[i, j - 1, k]
            ld[i, j, k] = (rhs - aa * ld[i, j - 1, k]) / denom
        else:
            denom = bdiag
            ld[i, j, k] = rhs / denom
        lc[i, j, k] = cc / denom
        gprev = gr
    for j in range(ny - 1, -1, -1):
        fi = f[i, j, k]
        if fi <= fmin:
            out[i, j, k] = 0.0
            continue
        val = ld[i, j, k]
        if j < ny - 1:
            val -= lc[i, j, k] * out[i, j + 1, k]
        out[i, j, k] = val


def k_sweep_z(f, temp, contact, geom, h_air, near, wet, wet_geom, scal, inp, cap, rz, bits, lc, ld, out, dt):
    """z sub-step, one thread per line (i, j): inp (the y result) -> out."""
    t = cuda.grid(1)
    nx, ny, nz = f.shape
    if t >= nx * ny:
        return
    i = t // ny
    j = t % ny
    bnd = WorkBoundary(h_air, near, wet, wet_geom, scal)
    ox = geom[0]
    oy = geom[1]
    oz = geom[2]
    dx = geom[3]
    face_area = dx * dx
    fmin = scal[B_FMIN]
    t_air = scal[B_T_AIR]
    t_fl = scal[B_T_FLUID]
    t_rad = scal[B_T_RAD]
    t_fix = scal[B_T_FIX]
    x = ox + (i + 0.5) * dx
    y = oy + (j + 0.5) * dx
    gl = 0.0
    for k in range(nz):
        fi = f[i, j, k]
        if fi <= fmin:
            lc[i, j, k] = 0.0
            ld[i, j, k] = 0.0
            gl = 0.0
            continue
        gr = 0.0
        if k < nz - 1 and f[i, j, k + 1] > fmin:
            gr = 1.0 / (np.float64(rz[i, j, k]) + np.float64(rz[i, j, k + 1]))
        hsum = 0.0
        rhs = cap[i, j, k] * inp[i, j, k]
        bb = bits[i, j, k]
        if bb & 48:
            z = oz + (k + 0.5) * dx
            cbits = contact[i, j, k]
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
            denom = bdiag - aa * lc[i, j, k - 1]
            ld[i, j, k] = (rhs - aa * ld[i, j, k - 1]) / denom
        else:
            denom = bdiag
            ld[i, j, k] = rhs / denom
        lc[i, j, k] = cc / denom
        gl = gr
    for k in range(nz - 1, -1, -1):
        fi = f[i, j, k]
        if fi <= fmin:
            out[i, j, k] = 0.0
            continue
        val = ld[i, j, k]
        if k < nz - 1:
            val -= lc[i, j, k] * out[i, j, k + 1]
        out[i, j, k] = val


def k_update(f, h, temp, cap, tz, T_K, k_W, cp, H, alpha, rho, geom, scal):
    """Last pass, one thread per cell: h += C dT / m, T = T(h)."""
    t = cuda.grid(1)
    nx, ny, nz = f.shape
    if t >= nx * ny * nz:
        return
    i = t // (ny * nz)
    j = (t // nz) % ny
    k = t % nz
    fi = f[i, j, k]
    if fi <= scal[B_FMIN]:
        return
    mat = MatTable(T_K, k_W, cp, H, alpha, rho)
    dx = geom[3]
    m = rho[0] * dx * dx * dx * fi
    de = cap[i, j, k] * (tz[i, j, k] - temp[i, j, k])
    h[i, j, k] = h[i, j, k] + de / m
    temp[i, j, k] = temperature(mat, h[i, j, k])


F1 = "float64[::1]"
F2 = "float64[:, ::1]"
F3 = "float64[:, :, ::1]"
R3 = "float32[:, :, ::1]"
U3 = "uint8[:, :, ::1]"
I8_2 = "int8[:, ::1]"
MAT = ", ".join([F1] * 6)
BND = f"{F1}, {F2}, {I8_2}, {F1}, {F1}"
SIGNATURES = {
    "k_cells": f"void({F3}, {F3}, {MAT}, {F1}, {F1}, {F3}, {R3}, {R3}, {R3}, {U3})",
    "k_sweep_x": f"void({F3}, {F3}, {U3}, {F1}, {BND}, {F3}, {F3}, {R3}, {U3}, {F3}, {F3}, {F3}, float64)",
    "k_sweep_y": f"void({F3}, {F3}, {U3}, {F1}, {BND}, {F3}, {F3}, {R3}, {U3}, {F3}, {F3}, {F3}, float64)",
    "k_sweep_z": f"void({F3}, {F3}, {U3}, {F1}, {BND}, {F3}, {F3}, {R3}, {U3}, {F3}, {F3}, {F3}, float64)",
    "k_update": f"void({F3}, {F3}, {F3}, {F3}, {F3}, {MAT}, {F1}, {F1})",
}
SOURCES = {"k_cells": k_cells, "k_sweep_x": k_sweep_x, "k_sweep_y": k_sweep_y, "k_sweep_z": k_sweep_z,
           "k_update": k_update}


# ============================================================================ host ledger (CPU order)
@njit(types.void(GRID_T, SCR_T, MAT_T, BND_T, types.float64, types.float64[:, :, ::1], types.float64[:, :, ::1],
                 types.float64[:, :, ::1], types.float64[:, :, ::1], types.float64[:, :, ::1], types.float64[:, :, ::1],
                 types.float64[:, :, ::1], types.float64[::1]),
      cache=True, nogil=True, error_model="numpy")
def ledger(g, s, mat, bnd, dt, src0, tx, ty, tz, h1, t1, cap, out):
    """The L_* terms of work_step from the GPU's per-cell results, summed in the CPU kernel's order.

    ``g`` holds the state before the step (f, h, T, contact), ``s.bits`` the exposure bits, ``src0`` the
    deposited energy on entry, ``tx``/``ty``/``tz`` the x/y/z sub-step results, ``h1``/``t1`` the new
    enthalpy and temperature, ``cap`` the cell capacities."""
    f = g.f
    h0 = g.h
    temp = g.T
    bits = s.bits
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
    for q in range(out.shape[0]):
        out[q] = 0.0
    u0_s = 0.0
    u0_c = 0.0
    e_s = 0.0
    e_c = 0.0
    lost = 0.0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                fi = f[i, j, k]
                if fi <= fmin:
                    lost += src0[i, j, k]
                    continue
                m = rho_v * fi
                u0_s, u0_c = neumaier(u0_s, u0_c, m * h0[i, j, k])
                e_s, e_c = neumaier(e_s, e_c, src0[i, j, k])
    la_s = 0.0
    lc_s = 0.0
    lr_s = 0.0
    lk_s = 0.0
    area_s = 0.0
    tsurf_s = 0.0
    # x sub-step: exposed area and mean surface temperature (forward order), then the x-face losses
    for j in range(ny):
        y = oy + (j + 0.5) * dx
        for i in range(nx):
            for k in range(nz):
                fi = f[i, j, k]
                if fi <= fmin:
                    continue
                bb = bits[i, j, k]
                if bb != 0:
                    ti = temp[i, j, k]
                    cbits = g.contact[i, j, k]
                    for face in range(6):
                        if (bb >> face) & 1:
                            afc = face_fraction(bb, fi, face) * face_area
                            if not ((cbits >> face) & 1) == 1:
                                area_s += afc
                                tsurf_s += afc * ti
        for i in range(nx - 1, -1, -1):
            x = ox + (i + 0.5) * dx
            for k in range(nz):
                fi = f[i, j, k]
                if fi <= fmin:
                    continue
                val = tx[i, j, k]
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
    for i in range(nx):
        x = ox + (i + 0.5) * dx
        for j in range(ny - 1, -1, -1):
            y = oy + (j + 0.5) * dx
            for k in range(nz):
                fi = f[i, j, k]
                if fi <= fmin:
                    continue
                val = ty[i, j, k]
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
    for i in range(nx):
        x = ox + (i + 0.5) * dx
        for j in range(ny):
            y = oy + (j + 0.5) * dx
            for k in range(nz - 1, -1, -1):
                fi = f[i, j, k]
                if fi <= fmin:
                    continue
                val = tz[i, j, k]
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
                de = cap[i, j, k] * (tz[i, j, k] - temp[i, j, k])
                du_s, du_c = neumaier(du_s, du_c, de)
                t_new = t1[i, j, k]
                u1_s, u1_c = neumaier(u1_s, u1_c, m * h1[i, j, k])
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


# ============================================================================ the GPU step
_KERNELS: dict[str, object] = {}


def kernels() -> dict:
    """The five compiled kernels (once per process)."""
    with D.LAUNCH_LOCK:
        if _KERNELS:
            return _KERNELS
        cuda = D.cuda_module()
        b = builder("cuda", cuda, D.simulator_mode())
        with D.exact_compile():
            built = {name: b.kernel(SOURCES[name], SIGNATURES[name]) for name in SOURCES}
        _KERNELS.update(built)
        return _KERNELS


def work_step(g: TA.WorkGrid, s: TA.WorkScratch, mat: TA.MatTable, bnd: TA.WorkBoundary, dt: float,
              out: np.ndarray) -> None:
    """``cnctwin.thermal.kernels.work_step`` on the GPU (same arguments, same results). The caller's inputs
    (g.h, g.T, s.src) change only after every kernel and transfer succeeded; any CUDA error is raised as
    GpuError with them unchanged (the scratch outputs the CPU kernel overwrites anyway may already hold the
    GPU's values)."""
    shape = g.f.shape
    nx, ny, nz = shape
    if min(nx, ny, nz) <= 0:
        raise D.GpuError("empty workpiece grid")
    try:
        ks = kernels()
        cuda = D.cuda_module()
        with D.LAUNCH_LOCK:
            dev = {name: cuda.to_device(np.ascontiguousarray(a)) for name, a in
                   (("f", g.f), ("h", g.h), ("T", g.T), ("contact", g.contact), ("geom", g.geom), ("src", s.src))}
            dmat = [cuda.to_device(np.ascontiguousarray(a)) for a in mat]
            dbnd = [cuda.to_device(np.ascontiguousarray(a)) for a in bnd]
            cap = cuda.device_array(shape, np.float64)
            rx = cuda.device_array(shape, np.float32)
            ry = cuda.device_array(shape, np.float32)
            rz = cuda.device_array(shape, np.float32)
            bits = cuda.device_array(shape, np.uint8)
            lc = cuda.device_array(shape, np.float64)
            ld = cuda.device_array(shape, np.float64)
            tx = cuda.device_array(shape, np.float64)
            ty = cuda.device_array(shape, np.float64)
            tz = cuda.device_array(shape, np.float64)
            dt = float(dt)
            blocks, tpb = D.grid(nx * ny * nz)
            ks["k_cells"][blocks, tpb](dev["f"], dev["T"], *dmat, dev["geom"], dbnd[4], cap, rx, ry, rz, bits)
            blocks, tpb = D.grid(ny * nz)
            ks["k_sweep_x"][blocks, tpb](dev["f"], dev["T"], dev["contact"], dev["geom"], *dbnd, dev["src"], cap, rx,
                                         bits, lc, ld, tx, dt)
            blocks, tpb = D.grid(nx * nz)
            ks["k_sweep_y"][blocks, tpb](dev["f"], dev["T"], dev["contact"], dev["geom"], *dbnd, tx, cap, ry, bits, lc,
                                         ld, ty, dt)
            blocks, tpb = D.grid(nx * ny)
            ks["k_sweep_z"][blocks, tpb](dev["f"], dev["T"], dev["contact"], dev["geom"], *dbnd, ty, cap, rz, bits, lc,
                                         ld, tz, dt)
            blocks, tpb = D.grid(nx * ny * nz)
            ks["k_update"][blocks, tpb](dev["f"], dev["h"], dev["T"], cap, tz, *dmat, dev["geom"], dbnd[4])
            cuda.synchronize()
            # pure outputs of the CPU kernel (it never reads them first): straight into the caller's arrays
            for d_arr, h_arr in ((cap, s.cap), (rx, s.rx), (ry, s.ry), (rz, s.rz), (bits, s.bits), (tz, s.a)):
                d_arr.copy_to_host(h_arr)
            # inputs of the CPU kernel (a fallback would need them): temporaries until nothing can fail
            tx_h = tx.copy_to_host()
            ty_h = ty.copy_to_host()
            h_new = dev["h"].copy_to_host()
            t_new = dev["T"].copy_to_host()
    except D.GpuError:
        raise
    except Exception as exc:          # any CUDA / NVVM / driver error: the caller runs the CPU kernel
        raise D.GpuError(f"thermal work_step on the GPU failed ({type(exc).__name__}: {exc})") from exc
    src0 = np.array(s.src, copy=True)
    ledger(g, s, mat, bnd, dt, src0, tx_h, ty_h, s.a, h_new, t_new, s.cap, out)
    s.src[...] = ty_h
    g.h[...] = h_new
    g.T[...] = t_new
