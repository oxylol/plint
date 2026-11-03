"""Numba kernels of the chips domain (docs/models/chips.md; rules: architecture 5.5).

Every function here is compiled eagerly from an explicit signature (``cache=True, nogil=True,
error_model="numpy"``, no fastmath, no parallel) and has a plain-Python twin in
:mod:`cnctwin.chips.reference` that the tests compare it with.

Top-level kernels (called once per macro step from :mod:`cnctwin.chips.domain`):

* :func:`advance_particles` (K1) - flight of every representative particle over the window:
  gravity, shape-dependent drag (Haider & Levenspiel 1989) in air, in coolant jets and in the
  wall film, exact exponential integration with frozen drag rate, contacts with the enclosure,
  the pile height field (in-process workpiece, fixture, table, chips), rotating tool and body
  cylinders (flute-face tangential impulse) and nozzles; restitution/friction by surface kind,
  wet sticking (Ennis et al. 1991 viscous Stokes number; Wilhelmy capillary escape), welding to
  a hot cutting edge, wrapping on rotating bodies, Ranz-Marshall cooling plus radiation, and
  settling into the height field.
* :func:`relax_pile` (K2) - avalanches down to the angle of repose, spill over edges.
* :func:`fill_occupancy` (K3) - the frozen chip-occupancy grid in S for recutting.
* :func:`flush_pile` (K4) - bed-load transport by the coolant flow (Shields criterion, Wong &
  Parker 2006 bed-load rate).
* :func:`pile_heat` (K5) - heat of resting chips to the surface below, the coolant and the air.
* :func:`sweep_cylinders` (K7) - chips displaced by tools moving through the pile.
* :func:`recut_take` (K8) - removes the chips the hot loop recut.
* :func:`drop_particles` - lays every flying particle down (end of an operation).
* :func:`hash_draws` - counter-based uniform and normal draws (core.rng) for many keys.
* :func:`write_emission` - the hot loop's row writer for the chips EventBuffer.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, types

from ..core.rng import hash_normal, hash_uniform
from ..units import STEFAN_BOLTZMANN_W_M2K4
from . import abi
from .abi import (CYL_PADDLE, EV_BOUNCE_LIMIT, EV_NOZZLE, EV_WELD, EV_WRAP, FLYING, FREE, MOB_FLUSH, MOB_RECUT,
                  MOB_SWEEP, ON_NOZZLE, ON_TOOL, OUT_DEPOSIT_KG, OUT_HEAT_AIR_J, OUT_HEAT_COOLANT_J,
                  OUT_HITS, OUT_PAN_BULK_M3, OUT_PAN_ENTHALPY_J, OUT_PAN_LONG_KG, OUT_PAN_MASS_KG, OUT_SUBSTEPS,
                  SURF_ENCLOSURE, SURF_NONE, SURF_NOZZLE, SURF_PILE, WRAPPED)

_OPTS = dict(cache=True, nogil=True, error_model="numpy")

f8 = types.float64
i8 = types.int64
b1 = types.boolean
F1 = types.float64[::1]
F2 = types.float64[:, ::1]
F3 = types.float64[:, :, ::1]
I1 = types.int64[::1]
I2 = types.int64[:, ::1]
U2 = types.UniTuple(f8, 2)
U3 = types.UniTuple(f8, 3)
U6 = types.UniTuple(f8, 6)
U10 = types.UniTuple(f8, 10)

P_T = abi.PARTICLES_T
H_T = abi.PILE_T
W_T = abi.WINDOW_T
S_T = abi.SCENE_T
C_T = abi.MODEL_T
E_T = abi.EVENTS_T
O_T = abi.OCCUPANCY_T

_INF = np.inf
_LN2 = math.log(2.0)
_TINY = 1e-300
_SERIES_Z = 1e-3          # below: Taylor series of the exponential-integrator weights (truncation < 2e-15)
_PARALLEL_EPS = 1e-12     # |direction component| below this is treated as parallel


# ============================================================== small numerics

@njit(U2(f8), **_OPTS)
def phi12(z):
    """phi1 = (1 - e^-z)/z and phi2 = (z - 1 + e^-z)/z^2, stable for z -> 0."""
    if z < _SERIES_Z:
        z2 = z * z
        return 1.0 - 0.5 * z + z2 / 6.0 - z2 * z / 24.0, 0.5 - z / 6.0 + z2 / 24.0 - z2 * z / 120.0
    em = math.expm1(-z)
    return -em / z, (z + em) / (z * z)


@njit(U6(f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8), **_OPTS)
def propagate(x, y, z, vx, vy, vz, ux, uy, uz, gx, gy, gz, k, dt):
    """Exact solution of dv/dt = g - k (v - u) over dt (k, u, g frozen): exponential integrator."""
    zz = k * dt
    p1, p2 = phi12(zz)
    a = dt * p1
    b = dt * dt * p2
    c = zz * p1
    return (x + ux * dt + (vx - ux) * a + gx * b,
            y + uy * dt + (vy - uy) * a + gy * b,
            z + uz * dt + (vz - uz) * a + gz * b,
            vx + (ux - vx) * c + gx * a,
            vy + (uy - vy) * c + gy * a,
            vz + (uz - vz) * c + gz * a)


@njit(f8(f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8), **_OPTS)
def drag_rate(speed, rho, mu, dv, area, chip_mass, stokes, a, b, c, d):
    """k (1/s) of F = m k (u - v): 1/2 rho Cd A |v_rel| / m with the Haider-Levenspiel Cd, written
    as Cd |v| = stokes mu/(rho dv) (1 + a Re^b) + c |v| Re/(Re + d) so that Re -> 0 stays finite."""
    if rho <= 0.0 or chip_mass <= 0.0 or dv <= 0.0:
        return 0.0
    re = rho * speed * dv / mu
    cdv = stokes * mu / (rho * dv) * (1.0 + a * re ** b) + c * speed * re / (re + d)
    return 0.5 * rho * area * cdv / chip_mass


@njit(f8(f8, f8, f8, f8, f8, f8, f8, f8, f8, f8), **_OPTS)
def nusselt_h(speed, rho, mu, k, pr, dv, ra, rb, rm, rn):
    """Ranz-Marshall h = k/dv (a + b Re^m Pr^n)."""
    if dv <= 0.0 or rho <= 0.0:
        return 0.0
    re = rho * speed * dv / mu
    return k / dv * (ra + rb * re ** rm * pr ** rn)


@njit(f8(f8, F1, F1), **_OPTS)
def cp_at(temp, cp_t, cp_v):
    n = cp_t.shape[0]
    if temp <= cp_t[0]:
        return cp_v[0]
    if temp >= cp_t[n - 1]:
        return cp_v[n - 1]
    i = 0
    while i < n - 2 and temp > cp_t[i + 1]:
        i += 1
    f = (temp - cp_t[i]) / (cp_t[i + 1] - cp_t[i])
    return cp_v[i] + f * (cp_v[i + 1] - cp_v[i])


@njit(f8(f8, F1, F1, F1), **_OPTS)
def enthalpy(temp, cp_t, cp_v, h_t):
    """Specific enthalpy (J/kg, 0 at 20 degC) of the piecewise-linear cp table, held at the ends."""
    n = cp_t.shape[0]
    if temp <= cp_t[0]:
        return h_t[0] + cp_v[0] * (temp - cp_t[0])
    if temp >= cp_t[n - 1]:
        return h_t[n - 1] + cp_v[n - 1] * (temp - cp_t[n - 1])
    i = 0
    while i < n - 2 and temp > cp_t[i + 1]:
        i += 1
    dt = temp - cp_t[i]
    slope = (cp_v[i + 1] - cp_v[i]) / (cp_t[i + 1] - cp_t[i])
    return h_t[i] + cp_v[i] * dt + 0.5 * slope * dt * dt


@njit(f8(f8, F1, F1, F1), **_OPTS)
def temperature(h, cp_t, cp_v, h_t):
    """Inverse of :func:`enthalpy` (exact on each quadratic segment)."""
    n = cp_t.shape[0]
    if h <= h_t[0]:
        return cp_t[0] + (h - h_t[0]) / cp_v[0]
    if h >= h_t[n - 1]:
        return cp_t[n - 1] + (h - h_t[n - 1]) / cp_v[n - 1]
    i = 0
    while i < n - 2 and h > h_t[i + 1]:
        i += 1
    a = 0.5 * (cp_v[i + 1] - cp_v[i]) / (cp_t[i + 1] - cp_t[i])
    b = cp_v[i]
    c = h_t[i] - h
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        disc = 0.0
    return cp_t[i] + 2.0 * (-c) / (b + math.sqrt(disc))


@njit(U3(f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8), **_OPTS)
def impulse(vx, vy, vz, sx, sy, sz, nx, ny, nz, e, mu):
    """Velocity after an impact on a surface moving with s, outward normal n: normal restitution e,
    Coulomb friction impulse mu (1 + e)|v_n| capped at sticking (Brach 1991)."""
    rx = vx - sx
    ry = vy - sy
    rz = vz - sz
    vn = rx * nx + ry * ny + rz * nz
    if vn >= 0.0:
        return vx, vy, vz
    tx = rx - vn * nx
    ty = ry - vn * ny
    tz = rz - vn * nz
    tm = math.sqrt(tx * tx + ty * ty + tz * tz)
    if tm > 0.0:
        red = mu * (1.0 + e) * (-vn)
        fac = 0.0 if red >= tm else 1.0 - red / tm
        tx *= fac
        ty *= fac
        tz *= fac
    return sx + tx - e * vn * nx, sy + ty - e * vn * ny, sz + tz - e * vn * nz


@njit(types.Tuple((i8, f8))(f8, f8, f8, i8), **_OPTS)
def tick_index(t, t0, dt, n_t):
    """Sample interval k (0 <= k < n_t) and fraction of time t on the window's uniform samples."""
    q = (t - t0) / dt
    if q <= 0.0:
        return 0, 0.0
    if q >= n_t:
        return n_t - 1, 1.0
    k = int(np.floor(q))
    if k > n_t - 1:
        k = n_t - 1
    return k, q - k


@njit(U3(F2, i8, f8), **_OPTS)
def lerp3(arr, k, frac):
    return (arr[k, 0] + frac * (arr[k + 1, 0] - arr[k, 0]),
            arr[k, 1] + frac * (arr[k + 1, 1] - arr[k, 1]),
            arr[k, 2] + frac * (arr[k + 1, 2] - arr[k, 2]))


@njit(f8(F1, i8, f8), **_OPTS)
def lerp1(arr, k, frac):
    return arr[k] + frac * (arr[k + 1] - arr[k])


# ============================================================== geometry queries

@njit(types.Tuple((f8, f8, f8, f8, i8, i8, i8))(f8, f8, f8, f8, f8, f8, H_T), **_OPTS)
def hf_hit(u0, v0, w0, u1, v1, w1, H):
    """First contact of the segment (G frame) with the height field (base + pile).

    Returns (s, n_u, n_v, n_w, kind, iu, iv): kind 0 none, 1 column top (cell entered), 2 column
    side wall (iu, iv = cell the segment is in before the wall, -1 if outside the grid),
    3 the start point is below the top of its own cell."""
    cell = H.cell_m[0]
    nu = H.base_m.shape[0]
    nv = H.base_m.shape[1]
    du = u1 - u0
    dvv = v1 - v0
    dw = w1 - w0
    s_lo = 0.0
    s_hi = 1.0
    face = 0
    if abs(du) < _PARALLEL_EPS:
        if u0 < 0.0 or u0 >= nu * cell:
            return 2.0, 0.0, 0.0, 0.0, 0, -1, -1
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
            return 2.0, 0.0, 0.0, 0.0, 0, -1, -1
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
        return 2.0, 0.0, 0.0, 0.0, 0, -1, -1
    s = s_lo
    iu = int(np.floor((u0 + s * du) / cell))
    iv = int(np.floor((v0 + s * dvv) / cell))
    iu = min(max(iu, 0), nu - 1)
    iv = min(max(iv, 0), nv - 1)
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
        base = H.base_m[iu, iv]
        if base > -_INF:
            top = base + H.bulk_m3[iu, iv] / (cell * cell)
            w_in = w0 + s * dw
            if w_in < top:
                if face == 0:
                    return s, 0.0, 0.0, 1.0, 3, iu, iv
                if face == 1:
                    return s, -float(step_u), 0.0, 0.0, 2, piu, piv
                return s, 0.0, -float(step_v), 0.0, 2, piu, piv
            w_out = w0 + s_out * dw
            if w_out < top and dw < 0.0:
                return (top - w0) / dw, 0.0, 0.0, 1.0, 1, iu, iv
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
            break
    return 2.0, 0.0, 0.0, 0.0, 0, -1, -1


@njit(types.Tuple((f8, f8, f8, f8, f8, i8))(f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8), **_OPTS)
def cyl_hit(x0, y0, z0, x1, y1, z1, bx, by, bz, ax, ay, az, radius, length):
    """First entry of the segment into the side of a finite cylinder (axis point b at z = 0, unit axis
    a, 0 <= z <= length). Returns (s, n_x, n_y, n_z, z_axial, flag): flag 0 none, 1 hit, 2 start inside."""
    d0x = x0 - bx
    d0y = y0 - by
    d0z = z0 - bz
    d1x = x1 - bx
    d1y = y1 - by
    d1z = z1 - bz
    za = d0x * ax + d0y * ay + d0z * az
    zb = d1x * ax + d1y * ay + d1z * az
    r0x = d0x - za * ax
    r0y = d0y - za * ay
    r0z = d0z - za * az
    r1x = d1x - zb * ax
    r1y = d1y - zb * ay
    r1z = d1z - zb * az
    c = r0x * r0x + r0y * r0y + r0z * r0z - radius * radius
    if c <= 0.0 and 0.0 <= za <= length:
        rm = math.sqrt(r0x * r0x + r0y * r0y + r0z * r0z)
        if rm > 0.0:
            return 0.0, r0x / rm, r0y / rm, r0z / rm, za, 2
        # on the axis: push out along the axis-normal direction closest to the motion
        px = r1x
        py = r1y
        pz = r1z
        pm = math.sqrt(px * px + py * py + pz * pz)
        if pm == 0.0:
            # any normal: a x e_k with the smallest |a_k|
            if abs(ax) <= abs(ay) and abs(ax) <= abs(az):
                px, py, pz = 0.0, az, -ay
            elif abs(ay) <= abs(az):
                px, py, pz = -az, 0.0, ax
            else:
                px, py, pz = ay, -ax, 0.0
            pm = math.sqrt(px * px + py * py + pz * pz)
        return 0.0, px / pm, py / pm, pz / pm, za, 2
    drx = r1x - r0x
    dry = r1y - r0y
    drz = r1z - r0z
    aa = drx * drx + dry * dry + drz * drz
    if aa <= 0.0:
        return 2.0, 0.0, 0.0, 0.0, 0.0, 0
    bb = r0x * drx + r0y * dry + r0z * drz
    disc = bb * bb - aa * c
    if disc < 0.0:
        return 2.0, 0.0, 0.0, 0.0, 0.0, 0
    s = (-bb - math.sqrt(disc)) / aa
    if s < 0.0 or s > 1.0:
        return 2.0, 0.0, 0.0, 0.0, 0.0, 0
    zs = za + s * (zb - za)
    if zs < 0.0 or zs > length:
        return 2.0, 0.0, 0.0, 0.0, 0.0, 0
    rx = r0x + s * drx
    ry = r0y + s * dry
    rz = r0z + s * drz
    rm = math.sqrt(rx * rx + ry * ry + rz * rz)
    return s, rx / rm, ry / rm, rz / rm, zs, 1


@njit(types.Tuple((f8, f8, f8, f8, i8))(f8, f8, f8, f8, f8, f8, f8, f8, f8, f8), **_OPTS)
def sph_hit(x0, y0, z0, x1, y1, z1, cx, cy, cz, radius):
    """First entry of the segment into a sphere: (s, n, flag) flag 0 none, 1 hit, 2 start inside."""
    dx = x0 - cx
    dy = y0 - cy
    dz = z0 - cz
    c = dx * dx + dy * dy + dz * dz - radius * radius
    if c <= 0.0:
        m = math.sqrt(dx * dx + dy * dy + dz * dz)
        if m > 0.0:
            return 0.0, dx / m, dy / m, dz / m, 2
        return 0.0, 0.0, 0.0, 1.0, 2
    ex = x1 - x0
    ey = y1 - y0
    ez = z1 - z0
    aa = ex * ex + ey * ey + ez * ez
    if aa <= 0.0:
        return 2.0, 0.0, 0.0, 0.0, 0
    bb = dx * ex + dy * ey + dz * ez
    disc = bb * bb - aa * c
    if disc < 0.0:
        return 2.0, 0.0, 0.0, 0.0, 0
    s = (-bb - math.sqrt(disc)) / aa
    if s < 0.0 or s > 1.0:
        return 2.0, 0.0, 0.0, 0.0, 0
    px = dx + s * ex
    py = dy + s * ey
    pz = dz + s * ez
    m = math.sqrt(px * px + py * py + pz * pz)
    return s, px / m, py / m, pz / m, 1


@njit(types.Tuple((f8, i8, i8))(f8, f8, f8, f8, f8, f8, F1, F1), **_OPTS)
def box_exit(x0, y0, z0, x1, y1, z1, lo, hi):
    """First crossing of the segment out of the box interior: (s, axis, side 0 lo / 1 hi) or (2, -1, -1);
    side 2 / 3 when the start point is already outside (below lo / above hi)."""
    p0 = (x0, y0, z0)
    p1 = (x1, y1, z1)
    best = 2.0
    axis = -1
    side = -1
    for k in range(3):
        a = p0[k]
        b = p1[k]
        if a < lo[k]:
            return 0.0, k, 2
        if a > hi[k]:
            return 0.0, k, 3
        if b < lo[k]:
            s = (lo[k] - a) / (b - a)
            if s < best:
                best = s
                axis = k
                side = 0
        elif b > hi[k]:
            s = (hi[k] - a) / (b - a)
            if s < best:
                best = s
                axis = k
                side = 1
    return best, axis, side


# ============================================================== pile bookkeeping

@njit(types.void(H_T, C_T, i8, i8, f8, f8, f8, f8, f8, f8, i8), **_OPTS)
def deposit(H, C, iu, iv, mass, enth, dv, thick, width, length, form):
    """Add one settled parcel to pile cell (iu, iv): bulk volume by the form's bulk volume ratio."""
    fi = min(max(form, 0), abi.N_FORM - 1)
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


@njit(types.void(H_T, i8, i8, i8, i8, f8), **_OPTS)
def move_fraction(H, iu, iv, ju, jv, frac):
    """Move ``frac`` of every extensive field of cell (iu, iv) to cell (ju, jv)."""
    for arr in (H.bulk_m3, H.mass_kg, H.enthalpy_J, H.size_mass, H.thick_mass, H.width_mass, H.length_mass,
                H.repose_mass, H.long_kg, H.fines_kg):
        d = arr[iu, iv] * frac
        arr[iu, iv] -= d
        arr[ju, jv] += d


@njit(types.void(H_T, i8, i8, f8, F1), **_OPTS)
def take_fraction(H, iu, iv, frac, row):
    """Remove ``frac`` of cell (iu, iv) into row[0:10] = bulk, mass, enthalpy, size_mass, thick_mass,
    width_mass, length_mass, repose_mass, long_kg, fines_kg (the cell keeps the rest)."""
    k = 0
    for arr in (H.bulk_m3, H.mass_kg, H.enthalpy_J, H.size_mass, H.thick_mass, H.width_mass, H.length_mass,
                H.repose_mass, H.long_kg, H.fines_kg):
        d = arr[iu, iv] * frac
        if frac >= 1.0:
            d = arr[iu, iv]
        arr[iu, iv] -= d
        row[k] = d
        k += 1


@njit(types.void(F2, I1, i8, f8, f8, F1, f8, f8, f8, f8, f8, f8, f8, f8), **_OPTS)
def write_mob(rec, n, kind, iu, iv, taken, vx, vy, vz, u, v, w, owner, time_s):
    """One mobilised row (abi.MOB_COLUMNS) from a take_fraction() result."""
    i = n[0]
    if i >= rec.shape[0]:
        n[1] = 1
        return
    rec[i, 0] = kind
    rec[i, 1] = iu
    rec[i, 2] = iv
    rec[i, 3] = taken[1]
    rec[i, 4] = taken[2]
    rec[i, 5] = taken[3]
    rec[i, 6] = taken[4]
    rec[i, 7] = taken[5]
    rec[i, 8] = taken[6]
    rec[i, 9] = taken[7]
    rec[i, 10] = taken[8]
    rec[i, 11] = taken[9]
    rec[i, 12] = vx
    rec[i, 13] = vy
    rec[i, 14] = vz
    rec[i, 15] = u
    rec[i, 16] = v
    rec[i, 17] = w
    rec[i, 18] = owner
    rec[i, 19] = time_s
    n[0] = i + 1


@njit(types.void(E_T, i8, i8, i8, f8, f8, f8, f8, f8, f8, i8), **_OPTS)
def write_event(ev, kind, particle, owner, time_s, mass, x, y, z, temp, form):
    i = ev.n[0]
    if i >= ev.rec.shape[0]:
        ev.n[1] = 1
        return
    ev.rec[i, 0] = kind
    ev.rec[i, 1] = particle
    ev.rec[i, 2] = owner
    ev.rec[i, 3] = time_s
    ev.rec[i, 4] = mass
    ev.rec[i, 5] = x
    ev.rec[i, 6] = y
    ev.rec[i, 7] = z
    ev.rec[i, 8] = temp
    ev.rec[i, 9] = form
    ev.n[0] = i + 1


@njit(types.void(P_T, C_T, i8, F1), **_OPTS)
def to_pan(P, C, i, out):
    m = P.mass_kg[i]
    fi = min(max(P.form[i], 0), abi.N_FORM - 1)
    out[OUT_PAN_MASS_KG] += m
    out[OUT_PAN_ENTHALPY_J] += m * enthalpy(P.temp_K[i], C.cp_T_K, C.cp_J_kgK, C.h_J_kg)
    out[OUT_PAN_BULK_M3] += m * C.form_R[fi] / C.rho_kg_m3[0]
    if C.form_long[fi] != 0:
        out[OUT_PAN_LONG_KG] += m
    P.status[i] = FREE
    P.mass_kg[i] = 0.0


@njit(types.void(P_T, H_T, C_T, i8, i8, i8, F1), **_OPTS)
def settle(P, H, C, i, iu, iv, out):
    """Particle i comes to rest in pile cell (iu, iv) (or the pan when there is none)."""
    if iu < 0 or iv < 0 or H.base_m[iu, iv] == -_INF:
        to_pan(P, C, i, out)
        return
    m = P.mass_kg[i]
    deposit(H, C, iu, iv, m, m * enthalpy(P.temp_K[i], C.cp_T_K, C.cp_J_kgK, C.h_J_kg), P.dv_m[i],
            P.thickness_m[i], P.width_m[i], P.length_m[i], P.form[i])
    out[OUT_DEPOSIT_KG] += m
    P.status[i] = FREE
    P.mass_kg[i] = 0.0


# ============================================================== fluids and heat

@njit(U10(f8, f8, f8, S_T), **_OPTS)
def jet_fluid(x, y, z, S):
    """The fluid at a point: the jet with the largest dynamic pressure there, else still air.

    Returns (u_x, u_y, u_z, rho, mu, k, Pr, T, force_scale, liquid_fraction). Liquid jets keep the
    exit velocity inside the spray cone; their momentum flux falls with the cone area (force scale
    = liquid volume fraction (r0 / r_j)^2). Gas jets follow the self-similar round jet: centreline
    u_c = u_0 min(1, B d_0 / s), Gaussian profile with half width S s."""
    best = -1.0
    ux = 0.0
    uy = 0.0
    uz = 0.0
    rho = S.air[0]
    mu = S.air[1]
    kf = S.air[2]
    pr = S.air[3]
    tf = S.air[4]
    fscale = 1.0
    liq = 0.0
    for j in range(S.jet_speed.shape[0]):
        sp = S.jet_speed[j]
        if sp <= 0.0:
            continue
        dx = x - S.jet_origin_M[j, 0]
        dy = y - S.jet_origin_M[j, 1]
        dz = z - S.jet_origin_M[j, 2]
        ex = S.jet_dir_M[j, 0]
        ey = S.jet_dir_M[j, 1]
        ez = S.jet_dir_M[j, 2]
        s = dx * ex + dy * ey + dz * ez
        if s <= 0.0:
            continue
        px = dx - s * ex
        py = dy - s * ey
        pz = dz - s * ez
        r = math.sqrt(px * px + py * py + pz * pz)
        r0 = S.jet_r0[j]
        if S.jet_gas[j] == 1:
            d0 = 2.0 * r0
            uc = sp if s <= S.jet_const[0] * d0 else sp * S.jet_const[0] * d0 / s
            rh = max(r0, S.jet_const[1] * s)
            u = uc * math.exp(-_LN2 * (r / rh) * (r / rh))
            q = 0.5 * S.jet_fluid[j, 0] * u * u
            if q > best:
                best = q
                ux = u * ex
                uy = u * ey
                uz = u * ez
                rho = S.jet_fluid[j, 0]
                mu = S.jet_fluid[j, 1]
                kf = S.jet_fluid[j, 2]
                pr = S.jet_fluid[j, 3]
                tf = S.air[4]
                fscale = 1.0
                liq = 0.0
        else:
            rj = r0 + s * S.jet_tan[j]
            if r > rj:
                continue
            alpha = (r0 / rj) * (r0 / rj)
            q = 0.5 * alpha * S.jet_fluid[j, 0] * sp * sp
            if q > best:
                best = q
                ux = sp * ex
                uy = sp * ey
                uz = sp * ez
                rho = S.jet_fluid[j, 0]
                mu = S.jet_fluid[j, 1]
                kf = S.jet_fluid[j, 2]
                pr = S.jet_fluid[j, 3]
                tf = S.liquid[4]
                fscale = alpha
                liq = alpha
    return ux, uy, uz, rho, mu, kf, pr, tf, fscale, liq


@njit(U3(P_T, C_T, S_T, i8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8), **_OPTS)
def cool(P, C, S, i, dt, speed, rho, mu, kf, pr, tf, liq, t_rad, t_air):
    """Lumped cooling of particle i over dt: Ranz-Marshall convection (liquid share ``liq``, air the
    rest) plus grey radiation to the enclosure. Returns (T_new, Q_to_liquid, Q_to_air), Q exact from
    the enthalpy table."""
    temp = P.temp_K[i]
    m_chip = P.chip_mass_kg[i]
    area = P.surface_m2[i]
    dv = P.dv_m[i]
    if m_chip <= 0.0 or area <= 0.0 or dt <= 0.0:
        return temp, 0.0, 0.0
    h_l = 0.0
    if liq > 0.0:
        h_l = liq * nusselt_h(speed, rho, mu, kf, pr, dv, C.rm[0], C.rm[1], C.rm[2], C.rm[3])
    h_a = (1.0 - liq) * nusselt_h(speed, S.air[0], S.air[1], S.air[2], S.air[3], dv,
                                  C.rm[0], C.rm[1], C.rm[2], C.rm[3])
    eps = C.emissivity[0]
    h_r = eps * STEFAN_BOLTZMANN_W_M2K4 * (temp * temp + t_rad * t_rad) * (temp + t_rad)
    h_tot = h_l + h_a + h_r
    if h_tot <= 0.0:
        return temp, 0.0, 0.0
    t_inf = (h_l * tf + h_a * t_air + h_r * t_rad) / h_tot
    cp = cp_at(temp, C.cp_T_K, C.cp_J_kgK)
    t_new = t_inf + (temp - t_inf) * math.exp(-h_tot * area * dt / (m_chip * cp))
    mass = P.mass_kg[i]
    q = mass * (enthalpy(temp, C.cp_T_K, C.cp_J_kgK, C.h_J_kg) - enthalpy(t_new, C.cp_T_K, C.cp_J_kgK, C.h_J_kg))
    fl = h_l * (temp - tf)
    fa = h_a * (temp - t_air) + h_r * (temp - t_rad)
    tot = fl + fa
    q_l = 0.0
    if tot != 0.0:
        q_l = q * fl / tot
    return t_new, q_l, q - q_l


# ============================================================== K1: particle flight

@njit(f8(P_T, S_T, C_T, i8), **_OPTS)
def capillary_escape_speed(P, S, C, i):
    """Rebound speed below which a chip stays on a wetted face: the Wilhelmy perimeter force
    gamma cos(theta) P times the film thickness must exceed the rebound energy (P = 2 (w + L))."""
    perim = 2.0 * (P.width_m[i] + P.length_m[i])
    f_cap = C.wet[2] * S.liquid[6] * C.wet[1] * perim
    m = P.chip_mass_kg[i]
    if m <= 0.0 or f_cap <= 0.0:
        return 0.0
    return math.sqrt(2.0 * f_cap * S.liquid[5] / m)


@njit(types.boolean(P_T, H_T, W_T, S_T, C_T, F1, i8, f8, f8, f8, f8, f8, f8, i8, i8, i8, f8, f8, f8), **_OPTS)
def respond_hf(P, H, W, S, C, out, i, vx, vy, vz, nxm, nym, nzm, kind, iu, iv, sx, sy, sz):
    """Contact with the height field (kind 1 top, 2 side wall, 3 start below the top). Returns True when
    the particle came to rest (deposited); otherwise its velocity in P is the rebound."""
    surf = SURF_ENCLOSURE
    wet = False
    if iu >= 0 and iv >= 0:
        surf = SURF_PILE if H.bulk_m3[iu, iv] > 0.0 else H.base_kind[iu, iv]
        if surf == SURF_NONE:
            surf = SURF_ENCLOSURE
        wet = H.wet[iu, iv] != 0
    e = C.e_pair[surf]
    mu = C.mu_pair[surf]
    v_in = -((vx - sx) * nxm + (vy - sy) * nym + (vz - sz) * nzm)
    if v_in < 0.0:
        v_in = 0.0
    g_n = -(W.gravity_M[0] * nxm + W.gravity_M[1] * nym + W.gravity_M[2] * nzm)
    dv = P.dv_m[i]
    v_reb = e * v_in
    stick = False
    if wet:
        v_esc = capillary_escape_speed(P, S, C, i)
        # viscous Stokes number (Ennis et al. 1991): St = 4 rho v dv / (9 mu) < (1 + 1/e) ln(h/h_a)
        st_c = (1.0 + 1.0 / e) * C.wet[0] if e > 0.0 else _INF
        v_visc = st_c * 9.0 * S.liquid[1] / (4.0 * C.rho_kg_m3[0] * dv)
        if v_reb < v_esc or v_in < v_visc:
            stick = True
    if kind != 2:
        if g_n > 0.0:
            v_set = math.sqrt(2.0 * g_n * C.settle_bounce[0] * dv)
            if v_reb < v_set or stick:
                settle(P, H, C, i, iu, iv, out)
                return True
    else:
        if stick:
            settle(P, H, C, i, iu, iv, out)
            return True
    nvx, nvy, nvz = impulse(vx, vy, vz, sx, sy, sz, nxm, nym, nzm, e, mu)
    P.vel_M_m_s[i, 0] = nvx
    P.vel_M_m_s[i, 1] = nvy
    P.vel_M_m_s[i, 2] = nvz
    return False


@njit(types.void(P_T, H_T, W_T, S_T, C_T, F1, E_T), **_OPTS)
def advance_particles(P, H, W, S, C, out, ev):
    """K1: advance every FLYING particle to the end of the window (see module doc)."""
    t0 = W.t0_s[0]
    t_end = W.t_end_s[0]
    dtk = W.dt_tick_s[0]
    n_t = W.grid_origin_M.shape[0] - 1
    R = W.R_M_G
    gx = W.gravity_M[0]
    gy = W.gravity_M[1]
    gz = W.gravity_M[2]
    dt_max = C.numerics[0]
    drag_fac = C.numerics[1]
    offset = C.numerics[2]
    max_bounces = int(C.numerics[3])
    max_hits = int(C.numerics[4])
    rho_p = C.rho_kg_m3[0]
    t_air = S.air[4]
    cell = H.cell_m[0]
    nu = H.base_m.shape[0]
    nv = H.base_m.shape[1]
    n_cyl = S.cyl_radius.shape[0]
    n_sph = S.sph_radius.shape[0]
    for i in range(P.n[0]):
        st = P.status[i]
        if st == FREE:
            continue
        if st != FLYING:
            # attached chips only cool (in air, or in a jet at their position)
            dt_att = t_end - P.time_s[i]
            if dt_att > 0.0:
                fx = jet_fluid(P.pos_M_m[i, 0], P.pos_M_m[i, 1], P.pos_M_m[i, 2], S)
                tn, ql, qa = cool(P, C, S, i, dt_att, math.sqrt(fx[0] ** 2 + fx[1] ** 2 + fx[2] ** 2), fx[3], fx[4],
                                  fx[5], fx[6], fx[7], fx[9], t_air, t_air)
                P.temp_K[i] = tn
                out[OUT_HEAT_COOLANT_J] += ql
                out[OUT_HEAT_AIR_J] += qa
                P.time_s[i] = t_end
            continue
        t = P.time_s[i]
        if t < t0:
            t = t0
        zero_hits = 0
        while t < t_end:
            x = P.pos_M_m[i, 0]
            y = P.pos_M_m[i, 1]
            z = P.pos_M_m[i, 2]
            vx = P.vel_M_m_s[i, 0]
            vy = P.vel_M_m_s[i, 1]
            vz = P.vel_M_m_s[i, 2]
            k0, f0 = tick_index(t, t0, dtk, n_t)
            ox, oy, oz = lerp3(W.grid_origin_M, k0, f0)
            # particle in G
            qx = x - ox
            qy = y - oy
            qz = z - oz
            pu = R[0, 0] * qx + R[1, 0] * qy + R[2, 0] * qz
            pv = R[0, 1] * qx + R[1, 1] * qy + R[2, 1] * qz
            pw = R[0, 2] * qx + R[1, 2] * qy + R[2, 2] * qz
            # fluid: jets, else the wall film over a wet cell, else air
            fl = jet_fluid(x, y, z, S)
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
            if 0 <= iu < nu and 0 <= iv < nv and H.wet[iu, iv] != 0 and H.base_m[iu, iv] > -_INF:
                top = H.base_m[iu, iv] + H.bulk_m3[iu, iv] / (cell * cell)
                film = S.liquid[5]
                if pw - top <= max(P.dv_m[i], film):
                    fu = H.film_uv_m_s[iu, iv, 0]
                    fv = H.film_uv_m_s[iu, iv, 1]
                    immersed = min(1.0, film / max(P.dv_m[i], _TINY))
                    q_film = 0.5 * immersed * S.liquid[0] * (fu * fu + fv * fv)
                    q_jet = 0.5 * fscale * rho * (ux * ux + uy * uy + uz * uz) if liq > 0.0 else 0.0
                    if q_film >= q_jet:
                        ux = R[0, 0] * fu + R[0, 1] * fv
                        uy = R[1, 0] * fu + R[1, 1] * fv
                        uz = R[2, 0] * fu + R[2, 1] * fv
                        rho = S.liquid[0]
                        mu = S.liquid[1]
                        kf = S.liquid[2]
                        pr = S.liquid[3]
                        tf = S.liquid[4]
                        fscale = immersed
                        liq = immersed
            rvx = vx - ux
            rvy = vy - uy
            rvz = vz - uz
            srel = math.sqrt(rvx * rvx + rvy * rvy + rvz * rvz)
            kd = fscale * drag_rate(srel, rho, mu, P.dv_m[i], P.area_m2[i], P.chip_mass_kg[i], C.hl[0],
                                    P.hl_a[i], P.hl_b[i], P.hl_c[i], P.hl_d[i])
            # buoyancy of the liquid share
            bfac = 1.0 - liq * S.liquid[0] / rho_p
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
            x1, y1, z1, vx1, vy1, vz1 = propagate(x, y, z, vx, vy, vz, ux, uy, uz, gx * bfac, gy * bfac,
                                                  gz * bfac, kd, dt)
            out[OUT_SUBSTEPS] += 1.0
            # ---- earliest contact along the chord
            best_s = 2.0
            hit = 0          # 1 height field, 2 cylinder, 3 sphere, 4 box
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
            sb, ba, bs = box_exit(x, y, z, x1, y1, z1, S.box_lo_M, S.box_hi_M)
            if sb <= 1.0:
                best_s = sb
                hit = 4
                b_axis = ba
                b_side = bs
            k1, f1 = tick_index(t + dt, t0, dtk, n_t)
            o1x, o1y, o1z = lerp3(W.grid_origin_M, k1, f1)
            q1x = x1 - o1x
            q1y = y1 - o1y
            q1z = z1 - o1z
            u1 = R[0, 0] * q1x + R[1, 0] * q1y + R[2, 0] * q1z
            v1 = R[0, 1] * q1x + R[1, 1] * q1y + R[2, 1] * q1z
            w1 = R[0, 2] * q1x + R[1, 2] * q1y + R[2, 2] * q1z
            sh, gu, gv, gw, hk, hiu, hiv = hf_hit(pu, pv, pw, u1, v1, w1, H)
            if hk != 0 and sh < best_s:
                best_s = sh
                hit = 1
                h_kind = hk
                h_iu = hiu
                h_iv = hiv
                hn0 = R[0, 0] * gu + R[0, 1] * gv + R[0, 2] * gw
                hn1 = R[1, 0] * gu + R[1, 1] * gv + R[1, 2] * gw
                hn2 = R[2, 0] * gu + R[2, 1] * gv + R[2, 2] * gw
            km, fm = tick_index(t + 0.5 * dt, t0, dtk, n_t)
            for c in range(n_cyl):
                if S.cyl_radius[c] <= 0.0:
                    continue
                bx, by, bz = lerp3(S.cyl_base_M[c], km, fm)
                ax, ay, az = lerp3(S.cyl_axis_M[c], km, fm)
                am = math.sqrt(ax * ax + ay * ay + az * az)
                if am <= 0.0:
                    continue
                ax /= am
                ay /= am
                az /= am
                sc, cnx, cny, cnz, cz, cfl = cyl_hit(x, y, z, x1, y1, z1, bx, by, bz, ax, ay, az, S.cyl_radius[c],
                                                     S.cyl_length[c])
                if cfl != 0 and sc < best_s:
                    best_s = sc
                    hit = 2
                    h_c = c
                    h_cz = cz
                    hn0 = cnx
                    hn1 = cny
                    hn2 = cnz
            for s_i in range(n_sph):
                ss, snx, sny, snz, sfl = sph_hit(x, y, z, x1, y1, z1, S.sph_center_M[s_i, 0], S.sph_center_M[s_i, 1],
                                                 S.sph_center_M[s_i, 2], S.sph_radius[s_i])
                if sfl != 0 and ss < best_s:
                    best_s = ss
                    hit = 3
                    h_c = s_i
                    hn0 = snx
                    hn1 = sny
                    hn2 = snz
            if hit == 0:
                srel1 = math.sqrt((vx1 - ux) ** 2 + (vy1 - uy) ** 2 + (vz1 - uz) ** 2)
                tn, ql, qa = cool(P, C, S, i, dt, 0.5 * (srel + srel1), rho, mu, kf, pr, tf, liq, t_air, t_air)
                P.temp_K[i] = tn
                out[OUT_HEAT_COOLANT_J] += ql
                out[OUT_HEAT_AIR_J] += qa
                P.pos_M_m[i, 0] = x1
                P.pos_M_m[i, 1] = y1
                P.pos_M_m[i, 2] = z1
                P.vel_M_m_s[i, 0] = vx1
                P.vel_M_m_s[i, 1] = vy1
                P.vel_M_m_s[i, 2] = vz1
                t = t_end if last else t + dt
                zero_hits = 0
                continue
            # ---- advance to the contact
            dth = best_s * dt
            hx = x + best_s * (x1 - x)
            hy = y + best_s * (y1 - y)
            hz = z + best_s * (z1 - z)
            if dth > 0.0:
                _, _, _, hvx, hvy, hvz = propagate(x, y, z, vx, vy, vz, ux, uy, uz, gx * bfac, gy * bfac, gz * bfac,
                                                   kd, dth)
                tn, ql, qa = cool(P, C, S, i, dth, srel, rho, mu, kf, pr, tf, liq, t_air, t_air)
                P.temp_K[i] = tn
                out[OUT_HEAT_COOLANT_J] += ql
                out[OUT_HEAT_AIR_J] += qa
                zero_hits = 0
            else:
                hvx = vx
                hvy = vy
                hvz = vz
                zero_hits += 1
            t += dth
            out[OUT_HITS] += 1.0
            P.bounces[i] += 1
            P.pos_M_m[i, 0] = hx
            P.pos_M_m[i, 1] = hy
            P.pos_M_m[i, 2] = hz
            P.vel_M_m_s[i, 0] = hvx
            P.vel_M_m_s[i, 1] = hvy
            P.vel_M_m_s[i, 2] = hvz
            P.time_s[i] = t
            kh, fh = tick_index(t, t0, dtk, n_t)
            if P.bounces[i] > max_bounces or zero_hits > max_hits:
                # cannot resolve the contacts: lay it down where it is
                ohx, ohy, ohz = lerp3(W.grid_origin_M, kh, fh)
                qx = hx - ohx
                qy = hy - ohy
                qz = hz - ohz
                cu = int(np.floor((R[0, 0] * qx + R[1, 0] * qy + R[2, 0] * qz) / cell))
                cv = int(np.floor((R[0, 1] * qx + R[1, 1] * qy + R[2, 1] * qz) / cell))
                write_event(ev, EV_BOUNCE_LIMIT, i, P.slot[i], t, P.mass_kg[i], hx, hy, hz, P.temp_K[i], P.form[i])
                if 0 <= cu < nu and 0 <= cv < nv:
                    settle(P, H, C, i, cu, cv, out)
                else:
                    to_pan(P, C, i, out)
                break
            if hit == 4:
                floor_hit = b_axis == S.floor[0] and b_side == S.floor[1]
                if floor_hit or S.walls_wet[0] != 0 or b_side >= 2:
                    to_pan(P, C, i, out)
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
                nvx, nvy, nvz = impulse(hvx, hvy, hvz, 0.0, 0.0, 0.0, bnx, bny, bnz, C.e_pair[SURF_ENCLOSURE],
                                        C.mu_pair[SURF_ENCLOSURE])
                P.vel_M_m_s[i, 0] = nvx
                P.vel_M_m_s[i, 1] = nvy
                P.vel_M_m_s[i, 2] = nvz
                P.pos_M_m[i, 0] = hx + offset * bnx
                P.pos_M_m[i, 1] = hy + offset * bny
                P.pos_M_m[i, 2] = hz + offset * bnz
                continue
            if hit == 1:
                # moving surface: velocity of the grid frame (table axes)
                sx = (W.grid_origin_M[kh + 1, 0] - W.grid_origin_M[kh, 0]) / dtk
                sy = (W.grid_origin_M[kh + 1, 1] - W.grid_origin_M[kh, 1]) / dtk
                sz = (W.grid_origin_M[kh + 1, 2] - W.grid_origin_M[kh, 2]) / dtk
                if h_kind == 3:
                    # below the top of its own cell (the pile grew under it): lift it onto the top
                    top = H.base_m[h_iu, h_iv] + H.bulk_m3[h_iu, h_iv] / (cell * cell)
                    ohx, ohy, ohz = lerp3(W.grid_origin_M, kh, fh)
                    qx = hx - ohx
                    qy = hy - ohy
                    qz = hz - ohz
                    lift = top - (R[0, 2] * qx + R[1, 2] * qy + R[2, 2] * qz)
                    hx += lift * hn0
                    hy += lift * hn1
                    hz += lift * hn2
                gone = respond_hf(P, H, W, S, C, out, i, hvx, hvy, hvz, hn0, hn1, hn2, h_kind, h_iu, h_iv,
                                  sx, sy, sz)
                if gone:
                    break
                P.pos_M_m[i, 0] = hx + offset * hn0
                P.pos_M_m[i, 1] = hy + offset * hn1
                P.pos_M_m[i, 2] = hz + offset * hn2
                continue
            if hit == 3:
                rn = S.sph_radius[h_c]
                hook = P.length_m[i] >= C.hook_radii[0] * rn
                stuck = False
                if S.sph_active[h_c] != 0:
                    stuck = hook
                else:
                    v_esc = capillary_escape_speed(P, S, C, i)
                    v_in = -(hvx * hn0 + hvy * hn1 + hvz * hn2)
                    stuck = hook or C.e_pair[SURF_NOZZLE] * max(v_in, 0.0) < v_esc
                if stuck:
                    P.status[i] = ON_NOZZLE
                    P.owner[i] = h_c
                    P.vel_M_m_s[i, 0] = 0.0
                    P.vel_M_m_s[i, 1] = 0.0
                    P.vel_M_m_s[i, 2] = 0.0
                    write_event(ev, EV_NOZZLE, i, h_c, t, P.mass_kg[i], hx, hy, hz, P.temp_K[i], P.form[i])
                    break
                nvx, nvy, nvz = impulse(hvx, hvy, hvz, 0.0, 0.0, 0.0, hn0, hn1, hn2, C.e_pair[SURF_NOZZLE],
                                        C.mu_pair[SURF_NOZZLE])
                P.vel_M_m_s[i, 0] = nvx
                P.vel_M_m_s[i, 1] = nvy
                P.vel_M_m_s[i, 2] = nvz
                gap = rn + offset + 0.5 * P.thickness_m[i]
                P.pos_M_m[i, 0] = S.sph_center_M[h_c, 0] + gap * hn0
                P.pos_M_m[i, 1] = S.sph_center_M[h_c, 1] + gap * hn1
                P.pos_M_m[i, 2] = S.sph_center_M[h_c, 2] + gap * hn2
                continue
            # ---- hit == 2: rotating cylinder
            c = h_c
            bx, by, bz = lerp3(S.cyl_base_M[c], kh, fh)
            ax, ay, az = lerp3(S.cyl_axis_M[c], kh, fh)
            am = math.sqrt(ax * ax + ay * ay + az * az)
            ax /= am
            ay /= am
            az /= am
            omega = lerp1(S.cyl_omega[c], kh, fh)
            tvx = (S.cyl_base_M[c, kh + 1, 0] - S.cyl_base_M[c, kh, 0]) / dtk
            tvy = (S.cyl_base_M[c, kh + 1, 1] - S.cyl_base_M[c, kh, 1]) / dtk
            tvz = (S.cyl_base_M[c, kh + 1, 2] - S.cyl_base_M[c, kh, 2]) / dtk
            radius = S.cyl_radius[c]
            # contact point on the surface and its velocity omega a x r + translation
            rx = hn0 * radius
            ry = hn1 * radius
            rz = hn2 * radius
            svx = omega * (ay * rz - az * ry) + tvx
            svy = omega * (az * rx - ax * rz) + tvy
            svz = omega * (ax * ry - ay * rx) + tvz
            # a fresh chip leaving its own tool's flute (first contact) neither welds nor wraps there
            own_first = P.bounces[i] <= 1 and S.cyl_owner[c] == P.slot[i]
            if S.cyl_weld[c] != 0 and P.temp_K[i] > C.weld_temp_K[0] and not own_first:
                P.status[i] = ON_TOOL
                P.owner[i] = S.cyl_owner[c]
                P.vel_M_m_s[i, 0] = svx
                P.vel_M_m_s[i, 1] = svy
                P.vel_M_m_s[i, 2] = svz
                write_event(ev, EV_WELD, i, S.cyl_owner[c], t, P.mass_kg[i], hx, hy, hz, P.temp_K[i], P.form[i])
                break
            fi = min(max(P.form[i], 0), abi.N_FORM - 1)
            if (S.cyl_wrap[c] != 0 and C.form_wrap[fi] != 0 and omega != 0.0 and not own_first
                    and P.length_m[i] >= C.wrap_turns[0] * 2.0 * math.pi * radius):
                P.status[i] = WRAPPED
                P.owner[i] = S.cyl_owner[c]
                P.vel_M_m_s[i, 0] = svx
                P.vel_M_m_s[i, 1] = svy
                P.vel_M_m_s[i, 2] = svz
                write_event(ev, EV_WRAP, i, S.cyl_owner[c], t, P.mass_kg[i], hx, hy, hz, P.temp_K[i], P.form[i])
                break
            surf = S.cyl_surf[c]
            e = C.e_pair[surf]
            mu = C.mu_pair[surf]
            if S.cyl_kind[c] == CYL_PADDLE and omega != 0.0:
                # flute face strikes the chip: restitution along the surface motion, radial reflection,
                # Coulomb friction on the axial component
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
                    dtn = -(1.0 + C.paddle_e[0]) * vrt
                    vrt = -C.paddle_e[0] * vrt
                red = mu * (dn + dtn)
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
                nvx, nvy, nvz = impulse(hvx, hvy, hvz, svx, svy, svz, hn0, hn1, hn2, e, mu)
            P.vel_M_m_s[i, 0] = nvx
            P.vel_M_m_s[i, 1] = nvy
            P.vel_M_m_s[i, 2] = nvz
            # leave it just outside the surface at the contact's axial position
            gap = radius + offset + 0.5 * P.thickness_m[i]
            P.pos_M_m[i, 0] = bx + h_cz * ax + gap * hn0
            P.pos_M_m[i, 1] = by + h_cz * ay + gap * hn1
            P.pos_M_m[i, 2] = bz + h_cz * az + gap * hn2
        if P.status[i] == FLYING:
            P.time_s[i] = t_end


# ============================================================== K2: avalanches

@njit(types.void(H_T, i8, f8, F1), **_OPTS)
def relax_pile(H, max_iter, tol_m3, out):
    """Move bulk volume down every slope steeper than the cells' angle of repose (Gauss-Seidel in fixed
    order), and spill over edges with nothing below. out[2:6] += spilled mass, enthalpy, bulk, long."""
    cell = H.cell_m[0]
    area = cell * cell
    nu = H.base_m.shape[0]
    nv = H.base_m.shape[1]
    du = (1, -1, 0, 0)
    dvv = (0, 0, 1, -1)
    taken = np.zeros(10)
    for _ in range(max_iter):
        moved = 0.0
        for iu in range(nu):
            for iv in range(nv):
                if H.bulk_m3[iu, iv] <= 0.0 or H.mass_kg[iu, iv] <= 0.0 or H.base_m[iu, iv] == -_INF:
                    continue
                tan_r = H.repose_mass[iu, iv] / H.mass_kg[iu, iv]
                for d in range(4):
                    if H.bulk_m3[iu, iv] <= 0.0:
                        break
                    ju = iu + du[d]
                    jv = iv + dvv[d]
                    if ju < 0 or ju >= nu or jv < 0 or jv >= nv:
                        continue
                    h_i = H.bulk_m3[iu, iv] / area
                    top_i = H.base_m[iu, iv] + h_i
                    if H.base_m[ju, jv] == -_INF:
                        # an edge with nothing below: the pile may reach half a cell out at the repose slope
                        excess = h_i - 0.5 * tan_r * cell
                        if excess <= 0.0:
                            continue
                        frac = excess / h_i
                        take_fraction(H, iu, iv, frac, taken)
                        out[OUT_PAN_MASS_KG] += taken[1]
                        out[OUT_PAN_ENTHALPY_J] += taken[2]
                        out[OUT_PAN_BULK_M3] += taken[0]
                        out[OUT_PAN_LONG_KG] += taken[8]
                        moved += taken[0]
                        continue
                    top_j = H.base_m[ju, jv] + H.bulk_m3[ju, jv] / area
                    excess = (top_i - top_j) - tan_r * cell
                    if excess <= 0.0:
                        continue
                    dvol = 0.5 * excess * area
                    if dvol > H.bulk_m3[iu, iv]:
                        dvol = H.bulk_m3[iu, iv]
                    frac = dvol / H.bulk_m3[iu, iv]
                    move_fraction(H, iu, iv, ju, jv, frac)
                    moved += dvol
        if moved <= tol_m3:
            break


# ============================================================== K3: occupancy grid

@njit(types.void(H_T, P_T, F2, F1, f8, O_T), **_OPTS)
def fill_occupancy(H, P, R_M_S, o_S_M, v_slow, occ):
    """Chip mass per voxel of the occupancy grid (S, aligned with the pile cells): every pile column
    spread uniformly over its height, plus flying particles slower than ``v_slow``. size_m and
    thickness_m are mass-weighted means."""
    nx = occ.mass_kg.shape[0]
    ny = occ.mass_kg.shape[1]
    nz = occ.mass_kg.shape[2]
    dims = (nx, ny, nz)
    oc = occ.cell_m[0]
    cell = H.cell_m[0]
    area = cell * cell
    occ.mass_kg[:] = 0.0
    occ.size_m[:] = 0.0
    occ.thickness_m[:] = 0.0
    A = H.axes_S
    nu = H.base_m.shape[0]
    nv = H.base_m.shape[1]
    for iu in range(nu):
        for iv in range(nv):
            m = H.mass_kg[iu, iv]
            if m <= 0.0 or H.base_m[iu, iv] == -_INF:
                continue
            h = H.bulk_m3[iu, iv] / area
            uc = (iu + 0.5) * cell
            vc = (iv + 0.5) * cell
            wa = H.base_m[iu, iv]
            wb = wa + h
            pa = (H.origin_G_m[0] + A[0, 0] * uc + A[0, 1] * vc + A[0, 2] * wa,
                  H.origin_G_m[1] + A[1, 0] * uc + A[1, 1] * vc + A[1, 2] * wa,
                  H.origin_G_m[2] + A[2, 0] * uc + A[2, 1] * vc + A[2, 2] * wa)
            pb = (H.origin_G_m[0] + A[0, 0] * uc + A[0, 1] * vc + A[0, 2] * wb,
                  H.origin_G_m[1] + A[1, 0] * uc + A[1, 1] * vc + A[1, 2] * wb,
                  H.origin_G_m[2] + A[2, 0] * uc + A[2, 1] * vc + A[2, 2] * wb)
            # axis of S along e_w
            kw = 0
            if abs(A[1, 2]) > abs(A[kw, 2]):
                kw = 1
            if abs(A[2, 2]) > abs(A[kw, 2]):
                kw = 2
            idx = np.zeros(3, np.int64)
            ok = True
            for k in range(3):
                if k == kw:
                    continue
                j = int(np.floor((pa[k] - occ.origin_S_m[k]) / oc))
                if j < 0 or j >= dims[k]:
                    ok = False
                idx[k] = j
            if not ok:
                continue
            lo = min(pa[kw], pb[kw])
            hi = max(pa[kw], pb[kw])
            size = H.size_mass[iu, iv] / m
            thick = H.thick_mass[iu, iv] / m
            if hi <= lo:
                j = int(np.floor((lo - occ.origin_S_m[kw]) / oc))
                if 0 <= j < dims[kw]:
                    idx[kw] = j
                    occ.mass_kg[idx[0], idx[1], idx[2]] += m
                    occ.size_m[idx[0], idx[1], idx[2]] += m * size
                    occ.thickness_m[idx[0], idx[1], idx[2]] += m * thick
                continue
            j0 = int(np.floor((lo - occ.origin_S_m[kw]) / oc))
            j1 = int(np.floor((hi - occ.origin_S_m[kw]) / oc))
            for j in range(max(j0, 0), min(j1, dims[kw] - 1) + 1):
                a = max(lo, occ.origin_S_m[kw] + j * oc)
                b = min(hi, occ.origin_S_m[kw] + (j + 1) * oc)
                if b <= a:
                    continue
                dm = m * (b - a) / (hi - lo)
                idx[kw] = j
                occ.mass_kg[idx[0], idx[1], idx[2]] += dm
                occ.size_m[idx[0], idx[1], idx[2]] += dm * size
                occ.thickness_m[idx[0], idx[1], idx[2]] += dm * thick
    for i in range(P.n[0]):
        if P.status[i] != FLYING:
            continue
        vx = P.vel_M_m_s[i, 0]
        vy = P.vel_M_m_s[i, 1]
        vz = P.vel_M_m_s[i, 2]
        if math.sqrt(vx * vx + vy * vy + vz * vz) >= v_slow:
            continue
        qx = P.pos_M_m[i, 0] - o_S_M[0]
        qy = P.pos_M_m[i, 1] - o_S_M[1]
        qz = P.pos_M_m[i, 2] - o_S_M[2]
        ps = (R_M_S[0, 0] * qx + R_M_S[1, 0] * qy + R_M_S[2, 0] * qz,
              R_M_S[0, 1] * qx + R_M_S[1, 1] * qy + R_M_S[2, 1] * qz,
              R_M_S[0, 2] * qx + R_M_S[1, 2] * qy + R_M_S[2, 2] * qz)
        jx = int(np.floor((ps[0] - occ.origin_S_m[0]) / oc))
        jy = int(np.floor((ps[1] - occ.origin_S_m[1]) / oc))
        jz = int(np.floor((ps[2] - occ.origin_S_m[2]) / oc))
        if jx < 0 or jx >= nx or jy < 0 or jy >= ny or jz < 0 or jz >= nz:
            continue
        m = P.mass_kg[i]
        occ.mass_kg[jx, jy, jz] += m
        occ.size_m[jx, jy, jz] += m * P.dv_m[i]
        occ.thickness_m[jx, jy, jz] += m * P.thickness_m[i]
    for jx in range(nx):
        for jy in range(ny):
            for jz in range(nz):
                m = occ.mass_kg[jx, jy, jz]
                if m > 0.0:
                    occ.size_m[jx, jy, jz] /= m
                    occ.thickness_m[jx, jy, jz] /= m


# ============================================================== K4: flushing

@njit(types.void(H_T, F1, F2, f8, f8, f8, F2, I1), **_OPTS)
def flush_pile(H, fp, R_M_G, rho_f, g, dt, rec, n):
    """Bed-load transport of settled chips by the coolant flow over the pile.

    fp = [C_f, theta_c, a_wp, exponent, capillary coefficient c (sigma_cap = c gamma / d; 0 when the bed is
    submerged), gamma, bed friction mu (resists the capillary stress), rho_s of the chips].
    Shields number theta = C_f q / ((rho_s - rho_f) g d), d the mass-weighted chip size; above
    theta_c (raised by capillary adhesion when the bed is wet but not submerged) the cell loses
    rho_s q_s cell dt with q_s = a (theta - theta_c)^n sqrt((rho_s/rho_f - 1) g d^3) (Wong & Parker 2006)."""
    cell = H.cell_m[0]
    nu = H.base_m.shape[0]
    nv = H.base_m.shape[1]
    taken = np.zeros(10)
    for iu in range(nu):
        for iv in range(nv):
            m = H.mass_kg[iu, iv]
            q = H.flush_Pa[iu, iv]
            if m <= 0.0 or q <= 0.0 or H.base_m[iu, iv] == -_INF:
                continue
            d = H.size_mass[iu, iv] / m
            if d <= 0.0:
                continue
            rho_s = fp[7]
            dens = (rho_s - rho_f) * g * d
            if dens <= 0.0:
                continue
            theta = fp[0] * q / dens
            theta_c = fp[1] + fp[6] * fp[4] * fp[5] / d / dens
            if theta <= theta_c:
                continue
            if n[0] >= rec.shape[0]:
                n[1] = 1                  # buffer full: the chips stay in place this step
                continue
            qstar = fp[2] * (theta - theta_c) ** fp[3]
            qs = qstar * math.sqrt((rho_s / rho_f - 1.0) * g * d * d * d)
            dm = rho_s * qs * cell * dt
            frac = dm / m
            if frac > 1.0:
                frac = 1.0
            take_fraction(H, iu, iv, frac, taken)
            fu = H.film_uv_m_s[iu, iv, 0]
            fv = H.film_uv_m_s[iu, iv, 1]
            vx = R_M_G[0, 0] * fu + R_M_G[0, 1] * fv
            vy = R_M_G[1, 0] * fu + R_M_G[1, 1] * fv
            vz = R_M_G[2, 0] * fu + R_M_G[2, 1] * fv
            top = H.base_m[iu, iv] + H.bulk_m3[iu, iv] / (cell * cell)
            write_mob(rec, n, MOB_FLUSH, float(iu), float(iv), taken, vx, vy, vz, (iu + 0.5) * cell,
                      (iv + 0.5) * cell, top + 0.5 * d, -1.0, -1.0)


# ============================================================== K5: heat of resting chips

@njit(types.void(H_T, C_T, F1, f8, F2, F1), **_OPTS)
def pile_heat(H, C, hp, dt, q_surface, out):
    """Heat of every pile cell to the surface below (contact conductance), to the coolant film (wet top)
    or to the air (convection + radiation), exact enthalpy bookkeeping.

    hp = [h_bed, h_top_air, h_top_wet, T_air, T_coolant]. q_surface[iu, iv] += J to the surface,
    out[0] += J to coolant, out[1] += J to air."""
    cell = H.cell_m[0]
    area = cell * cell
    nu = H.base_m.shape[0]
    nv = H.base_m.shape[1]
    eps = C.emissivity[0]
    t_air = hp[3]
    t_cool = hp[4]
    for iu in range(nu):
        for iv in range(nv):
            m = H.mass_kg[iu, iv]
            if m <= 0.0:
                continue
            hm = H.enthalpy_J[iu, iv] / m
            temp = temperature(hm, C.cp_T_K, C.cp_J_kgK, C.h_J_kg)
            h_b = hp[0] if H.base_kind[iu, iv] != SURF_NONE else 0.0
            t_s = H.surf_temp_K[iu, iv]
            wet = H.wet[iu, iv] != 0
            if wet:
                h_t = hp[2]
                t_t = t_cool
            else:
                h_t = hp[1] + eps * STEFAN_BOLTZMANN_W_M2K4 * (temp * temp + t_air * t_air) * (temp + t_air)
                t_t = t_air
            h_tot = h_b + h_t
            if h_tot <= 0.0:
                continue
            t_inf = (h_b * t_s + h_t * t_t) / h_tot
            cp = cp_at(temp, C.cp_T_K, C.cp_J_kgK)
            t_new = t_inf + (temp - t_inf) * math.exp(-h_tot * area * dt / (m * cp))
            q = H.enthalpy_J[iu, iv] - m * enthalpy(t_new, C.cp_T_K, C.cp_J_kgK, C.h_J_kg)
            fb = h_b * (temp - t_s)
            ft = h_t * (temp - t_t)
            tot = fb + ft
            q_b = q * fb / tot if tot != 0.0 else 0.0
            q_t = q - q_b
            H.enthalpy_J[iu, iv] -= q
            q_surface[iu, iv] += q_b
            if wet:
                out[0] += q_t
            else:
                out[1] += q_t


# ============================================================== K7: tools moving through the pile

@njit(types.Tuple((f8, f8))(f8, f8, f8, f8, f8, f8, f8, f8, f8, f8), **_OPTS)
def column_in_cylinder(uc, vc, bu, bv, bw, au, av, aw, radius, length):
    """w-interval of the vertical line (uc, vc, w) of G inside the finite cylinder (axis point b at
    z = 0, unit axis a, radius, 0 <= z <= length), or (1, 0) when empty."""
    d0u = uc - bu
    d0v = vc - bv
    d0w = -bw
    z0 = d0u * au + d0v * av + d0w * aw
    r0u = d0u - z0 * au
    r0v = d0v - z0 * av
    r0w = d0w - z0 * aw
    r1u = -aw * au
    r1v = -aw * av
    r1w = 1.0 - aw * aw
    aa = r1u * r1u + r1v * r1v + r1w * r1w
    cc = r0u * r0u + r0v * r0v + r0w * r0w - radius * radius
    if aa < _PARALLEL_EPS:
        if cc > 0.0:
            return 1.0, 0.0
        lo = -_INF
        hi = _INF
    else:
        bb = r0u * r1u + r0v * r1v + r0w * r1w
        disc = bb * bb - aa * cc
        if disc < 0.0:
            return 1.0, 0.0
        sq = math.sqrt(disc)
        lo = (-bb - sq) / aa
        hi = (-bb + sq) / aa
    if abs(aw) < _PARALLEL_EPS:
        if z0 < 0.0 or z0 > length:
            return 1.0, 0.0
    else:
        za = (0.0 - z0) / aw
        zb = (length - z0) / aw
        if za > zb:
            za, zb = zb, za
        if za > lo:
            lo = za
        if zb < hi:
            hi = zb
    if hi <= lo:
        return 1.0, 0.0
    return lo, hi


@njit(types.void(H_T, W_T, S_T, C_T, F2, I1), **_OPTS)
def sweep_cylinders(H, W, S, C, rec, n):
    """Chips in the path of a tool (cylinders owned by a tool slot) at each window sample are thrown
    out: the overlap of the pile column with the cylinder leaves the cell with the velocity a chip at
    rest gets from that surface (flute face: (1 + e_paddle) omega a x r; smooth: friction impulse)."""
    cell = H.cell_m[0]
    area = cell * cell
    nu = H.base_m.shape[0]
    nv = H.base_m.shape[1]
    R = W.R_M_G
    n_t = W.grid_origin_M.shape[0] - 1
    taken = np.zeros(10)
    for c in range(S.cyl_radius.shape[0]):
        if S.cyl_owner[c] < 0 or S.cyl_radius[c] <= 0.0:
            continue
        radius = S.cyl_radius[c]
        length = S.cyl_length[c]
        for k in range(n_t + 1):
            bxm = S.cyl_base_M[c, k, 0] - W.grid_origin_M[k, 0]
            bym = S.cyl_base_M[c, k, 1] - W.grid_origin_M[k, 1]
            bzm = S.cyl_base_M[c, k, 2] - W.grid_origin_M[k, 2]
            bu = R[0, 0] * bxm + R[1, 0] * bym + R[2, 0] * bzm
            bv = R[0, 1] * bxm + R[1, 1] * bym + R[2, 1] * bzm
            bw = R[0, 2] * bxm + R[1, 2] * bym + R[2, 2] * bzm
            axm = S.cyl_axis_M[c, k, 0]
            aym = S.cyl_axis_M[c, k, 1]
            azm = S.cyl_axis_M[c, k, 2]
            am = math.sqrt(axm * axm + aym * aym + azm * azm)
            if am <= 0.0:
                continue
            axm /= am
            aym /= am
            azm /= am
            au = R[0, 0] * axm + R[1, 0] * aym + R[2, 0] * azm
            av = R[0, 1] * axm + R[1, 1] * aym + R[2, 1] * azm
            aw = R[0, 2] * axm + R[1, 2] * aym + R[2, 2] * azm
            eu = bu + length * au
            ev = bv + length * av
            u_lo = min(bu, eu) - radius
            u_hi = max(bu, eu) + radius
            v_lo = min(bv, ev) - radius
            v_hi = max(bv, ev) + radius
            i0 = max(int(np.floor(u_lo / cell)), 0)
            i1 = min(int(np.floor(u_hi / cell)), nu - 1)
            j0 = max(int(np.floor(v_lo / cell)), 0)
            j1 = min(int(np.floor(v_hi / cell)), nv - 1)
            omega = S.cyl_omega[c, k]
            kk = k if k < n_t else n_t - 1
            tvx = (S.cyl_base_M[c, kk + 1, 0] - S.cyl_base_M[c, kk, 0]) / W.dt_tick_s[0]
            tvy = (S.cyl_base_M[c, kk + 1, 1] - S.cyl_base_M[c, kk, 1]) / W.dt_tick_s[0]
            tvz = (S.cyl_base_M[c, kk + 1, 2] - S.cyl_base_M[c, kk, 2]) / W.dt_tick_s[0]
            t_k = W.t0_s[0] + k * W.dt_tick_s[0]
            for iu in range(i0, i1 + 1):
                for iv in range(j0, j1 + 1):
                    if H.mass_kg[iu, iv] <= 0.0 or H.base_m[iu, iv] == -_INF:
                        continue
                    uc = (iu + 0.5) * cell
                    vc = (iv + 0.5) * cell
                    lo, hi = column_in_cylinder(uc, vc, bu, bv, bw, au, av, aw, radius, length)
                    if hi <= lo:
                        continue
                    base = H.base_m[iu, iv]
                    h = H.bulk_m3[iu, iv] / area
                    a = max(lo, base)
                    b = min(hi, base + h)
                    if b <= a:
                        continue
                    if n[0] >= rec.shape[0]:
                        n[1] = 1
                        continue
                    frac = (b - a) / h
                    wm = 0.5 * (a + b)
                    # radial vector at the contact (G -> M)
                    du_ = uc - bu
                    dv_ = vc - bv
                    dw_ = wm - bw
                    za = du_ * au + dv_ * av + dw_ * aw
                    ru = du_ - za * au
                    rv = dv_ - za * av
                    rw = dw_ - za * aw
                    rm = math.sqrt(ru * ru + rv * rv + rw * rw)
                    if rm > 0.0:
                        ru *= radius / rm
                        rv *= radius / rm
                        rw *= radius / rm
                    rxm = R[0, 0] * ru + R[0, 1] * rv + R[0, 2] * rw
                    rym = R[1, 0] * ru + R[1, 1] * rv + R[1, 2] * rw
                    rzm = R[2, 0] * ru + R[2, 1] * rv + R[2, 2] * rw
                    svx = omega * (aym * rzm - azm * rym)
                    svy = omega * (azm * rxm - axm * rzm)
                    svz = omega * (axm * rym - aym * rxm)
                    surf = S.cyl_surf[c]
                    if S.cyl_kind[c] == CYL_PADDLE:
                        f = 1.0 + C.paddle_e[0]
                        vx = f * svx + tvx
                        vy = f * svy + tvy
                        vz = f * svz + tvz
                    elif rm > 0.0:
                        vx, vy, vz = impulse(0.0, 0.0, 0.0, svx + tvx, svy + tvy, svz + tvz, rxm / radius,
                                             rym / radius, rzm / radius, C.e_pair[surf], C.mu_pair[surf])
                    else:
                        vx = tvx
                        vy = tvy
                        vz = tvz
                    size = H.size_mass[iu, iv] / H.mass_kg[iu, iv]
                    take_fraction(H, iu, iv, frac, taken)
                    # released just outside the cylinder surface
                    out_r = (radius + 0.5 * size) / radius if rm > 0.0 else 1.0
                    pu = bu + za * au + ru * out_r
                    pv = bv + za * av + rv * out_r
                    pw = bw + za * aw + rw * out_r
                    write_mob(rec, n, MOB_SWEEP, float(iu), float(iv), taken, vx, vy, vz, pu, pv, pw,
                              float(S.cyl_owner[c]), t_k)


# ============================================================== K8: recut chips

@njit(f8(H_T, F2, F2, I1), **_OPTS)
def recut_take(H, req, rec, n):
    """Remove the chips the hot loop recut. req rows: [iu, iv, mass_kg, vx, vy, vz, owner, time_s,
    u, v, w]; mass is taken from cell (iu, iv), then its 8 neighbours in fixed order. Returns the
    requested mass that was not found (the chips had moved since the frozen grid was published)."""
    nu = H.base_m.shape[0]
    nv = H.base_m.shape[1]
    taken = np.zeros(10)
    nbr_u = (0, 1, -1, 0, 0, 1, 1, -1, -1)
    nbr_v = (0, 0, 0, 1, -1, 1, -1, 1, -1)
    missing = 0.0
    for r in range(req.shape[0]):
        want = req[r, 2]
        iu0 = int(req[r, 0])
        iv0 = int(req[r, 1])
        for d in range(9):
            if want <= 0.0:
                break
            iu = iu0 + nbr_u[d]
            iv = iv0 + nbr_v[d]
            if iu < 0 or iu >= nu or iv < 0 or iv >= nv:
                continue
            m = H.mass_kg[iu, iv]
            if m <= 0.0:
                continue
            if n[0] >= rec.shape[0]:
                n[1] = 1
                break
            frac = want / m
            if frac > 1.0:
                frac = 1.0
            take_fraction(H, iu, iv, frac, taken)
            want -= taken[1]
            write_mob(rec, n, MOB_RECUT, float(iu), float(iv), taken, req[r, 3], req[r, 4], req[r, 5],
                      req[r, 8], req[r, 9], req[r, 10], req[r, 6], req[r, 7])
        if want > 0.0:
            missing += want
    return missing


# ============================================================== end of operation

@njit(types.void(P_T, H_T, W_T, C_T, F1), **_OPTS)
def drop_particles(P, H, W, C, out):
    """Every FLYING particle comes to rest in the pile column below it (or the pan)."""
    cell = H.cell_m[0]
    nu = H.base_m.shape[0]
    nv = H.base_m.shape[1]
    R = W.R_M_G
    n_t = W.grid_origin_M.shape[0] - 1
    for i in range(P.n[0]):
        if P.status[i] != FLYING:
            continue
        qx = P.pos_M_m[i, 0] - W.grid_origin_M[n_t, 0]
        qy = P.pos_M_m[i, 1] - W.grid_origin_M[n_t, 1]
        qz = P.pos_M_m[i, 2] - W.grid_origin_M[n_t, 2]
        iu = int(np.floor((R[0, 0] * qx + R[1, 0] * qy + R[2, 0] * qz) / cell))
        iv = int(np.floor((R[0, 1] * qx + R[1, 1] * qy + R[2, 1] * qz) / cell))
        pw = R[0, 2] * qx + R[1, 2] * qy + R[2, 2] * qz
        if 0 <= iu < nu and 0 <= iv < nv and H.base_m[iu, iv] > -_INF and pw >= H.base_m[iu, iv]:
            settle(P, H, C, i, iu, iv, out)
        else:
            to_pan(P, C, i, out)


# ============================================================== coolant film on the pile grid

@njit(types.void(F2, f8, f8, F3), **_OPTS)
def film_field(q, cell, rho, film):
    """Film velocity (u, v) of every cell: down the gradient of the flush (dynamic) pressure q with the speed
    sqrt(2 q / rho); central differences inside, one-sided at the edges (as numpy.gradient, edge order 1)."""
    nu = q.shape[0]
    nv = q.shape[1]
    for iu in range(nu):
        for iv in range(nv):
            if nu < 2 or nv < 2 or q[iu, iv] <= 0.0:
                film[iu, iv, 0] = 0.0
                film[iu, iv, 1] = 0.0
                continue
            if iu == 0:
                gu = (q[1, iv] - q[0, iv]) / cell
            elif iu == nu - 1:
                gu = (q[nu - 1, iv] - q[nu - 2, iv]) / cell
            else:
                gu = (q[iu + 1, iv] - q[iu - 1, iv]) / (2.0 * cell)
            if iv == 0:
                gv = (q[iu, 1] - q[iu, 0]) / cell
            elif iv == nv - 1:
                gv = (q[iu, nv - 1] - q[iu, nv - 2]) / cell
            else:
                gv = (q[iu, iv + 1] - q[iu, iv - 1]) / (2.0 * cell)
            mag = math.sqrt(gu * gu + gv * gv)
            if mag <= 0.0:
                film[iu, iv, 0] = 0.0
                film[iu, iv, 1] = 0.0
                continue
            speed = math.sqrt(2.0 * q[iu, iv] / rho)
            film[iu, iv, 0] = -gu / mag * speed
            film[iu, iv, 1] = -gv / mag * speed


# ============================================================== counter-based draws

@njit(types.void(i8, i8, I1, i8, i8, F2, F2), **_OPTS)
def hash_draws(seed, stream_id, keys, b, c0, uniform, normal):
    """uniform[i, j] = hash_uniform(seed, stream, keys[i], b, c0 + j); normal[i, j] = hash_normal(...)."""
    for i in range(keys.shape[0]):
        for j in range(uniform.shape[1]):
            uniform[i, j] = hash_uniform(seed, stream_id, keys[i], b, c0 + j)
        for j in range(normal.shape[1]):
            normal[i, j] = hash_normal(seed, stream_id, keys[i], b, c0 + j)


# ============================================================== hot-loop row writer

@njit(types.void(F2, I1, i8, i8, i8, i8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, i8, i8),
      **_OPTS)
def write_emission(rec, n, tick, contact, tool_slot, tooth, mass_kg, x, y, z, vx, vy, vz, temp_K, h_m, b_m, l_m,
                   v_cut_m_s, rake_rad, incl_rad, edge_kind, group):
    """Append one chip-emission row (abi.EMISSION_COLUMNS) to the chips EventBuffer (rec, n); sets
    n[1] on overflow (the loop then replays the step with a larger buffer)."""
    i = n[0]
    if i >= rec.shape[0]:
        n[1] = 1
        return
    rec[i, 0] = tick
    rec[i, 1] = contact
    rec[i, 2] = tool_slot
    rec[i, 3] = tooth
    rec[i, 4] = mass_kg
    rec[i, 5] = x
    rec[i, 6] = y
    rec[i, 7] = z
    rec[i, 8] = vx
    rec[i, 9] = vy
    rec[i, 10] = vz
    rec[i, 11] = temp_K
    rec[i, 12] = h_m
    rec[i, 13] = b_m
    rec[i, 14] = l_m
    rec[i, 15] = v_cut_m_s
    rec[i, 16] = rake_rad
    rec[i, 17] = incl_rad
    rec[i, 18] = edge_kind
    rec[i, 19] = group
    n[0] = i + 1


KERNELS = (advance_particles, relax_pile, fill_occupancy, flush_pile, pile_heat, sweep_cylinders, recut_take,
           drop_particles, film_field, hash_draws, write_emission)
