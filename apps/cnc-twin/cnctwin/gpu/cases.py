"""Synthetic, seeded inputs of the three kernels for the benchmark and the parity tests (docs/models/gpu.md
"Benchmark").

These are test cases, not models: geometry and states are arbitrary but plausible (a block of Al 6061-T6
cut by a tilted bull-nose or ball end mill along a path; a workpiece grid with a pocket, fixture contact
and coolant zones; a cloud of aluminium chips over a pile with a tool, a holder, a nozzle and two jets).
Material data come from the databases (db/materials/al6061-t6.yaml); every random draw comes from
``numpy.random.default_rng(seed)``, so a case is the same for the same arguments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MATERIAL_ID = "al6061-t6"


# ============================================================================ dexel subtraction
@dataclass
class DexelCase:
    field: object               # removal.abi.DexelField
    profile: object             # removal.abi.Profile
    specs: list                 # removal.abi.SubtractSpec, in order
    target: object              # removal.abi.TargetField
    levels: int


def _target_below(field, z_top_m: float):
    """Target field: the stock below the plane z = z_top (a floor the cuts must not go below)."""
    from ..removal import abi as A

    starts = [0]
    segs = []
    for lvl in range(int(field.info[A.FI_NLEVELS])):
        a = float(field.lv_cell_m[lvl])
        dims = field.lv_dims[lvl]
        lo = field.lv_box_lo_m[lvl]
        hi = field.lv_box_hi_m[lvl]
        for fam in range(3):
            u, v = A.TRANS_U[fam], A.TRANS_V[fam]
            for iu in range(int(dims[u])):
                for iv in range(int(dims[v])):
                    p = field.lv_origin_m[lvl].copy()
                    p[u] += iu * a
                    p[v] += iv * a
                    inside = lo[u] <= p[u] <= hi[u] and lo[v] <= p[v] <= hi[v]
                    t0, t1 = lo[fam] - p[fam], hi[fam] - p[fam]
                    if fam == 2:
                        t1 = min(t1, z_top_m - p[2])
                    elif p[2] > z_top_m:
                        inside = False
                    q0 = int(math.floor(max(t0, 0.0) / A.QUANTUM_M + 0.5))
                    q1 = int(math.floor(t1 / A.QUANTUM_M + 0.5))
                    if inside and q1 > q0:
                        segs.append((q0, q1))
                    starts.append(len(segs))
    seg = np.array(segs, np.int32).reshape(-1, 2)
    return A.TargetField(np.array(starts, np.int64), seg, np.array([1, len(segs), 0, 0], np.int64))


def dexel_case(size_mm=(6.0, 5.0, 4.0), lattice_mm=0.25, tool: str = "torus", cuts: int = 6, refine: bool = True,
               target: bool = True, pool_blocks: int | None = None, radius_frac: float = 0.3, slabs: int = 3,
               seed: int = 0) -> DexelCase:
    """A block cut by a tilted end mill (radius ``radius_frac`` x the block's y size) along x, ``cuts`` subtractions
    stepping over the block, the axis tilted by up to 0.3 rad, ``slabs`` slabs with radius offsets; with
    ``target`` the part is the stock below 0.35 of its height, and (cuts >= 4) the last two cuts go into it
    and through the bottom (gouge and breakthrough events)."""
    from ..removal import abi as A
    from ..removal.profile import profile_ball, profile_torus
    from ..removal.stock import block_field

    rng = np.random.default_rng(seed)
    size = np.asarray(size_mm, dtype=float) * 1e-3
    lattice = float(lattice_mm) * 1e-3
    boxes = []
    if refine:
        boxes.append((size * [0.2, 0.2, 0.5], size * [0.6, 0.6, 1.0], 2))
    field = block_field(size, lattice, refine=boxes, pool_blocks=pool_blocks)
    radius = float(radius_frac) * size[1]
    length = 3.0 * size[2]
    prof = profile_torus(radius, 0.3 * radius, length) if tool == "torus" else profile_ball(radius, length)
    tilt = rng.uniform(-0.3, 0.3, 2)
    axis = np.array([math.sin(tilt[0]), math.sin(tilt[1]), 1.0])
    specs = []
    # tip heights: above the target floor, except one cut into it (gouge) and one through the bottom
    # (breakthrough) when there are enough cuts
    depth = np.full(int(cuts), 0.55)
    if cuts >= 4:
        depth[cuts - 2] = 0.25
        depth[cuts - 1] = -0.1
    for k in range(int(cuts)):
        x = size[0] * (0.1 + 0.8 * k / max(1, cuts - 1))
        tip = np.array([x, 0.5 * size[1] + rng.uniform(-0.1, 0.1) * radius, depth[k] * size[2]])
        jitter = rng.uniform(-2e-5, 2e-5, (slabs, 3))
        unit = axis / np.linalg.norm(axis)
        jitter -= np.outer(jitter @ unit, unit)          # deflections across the axis: slab ends stay aligned
        slab_c = tip + jitter
        # slabs over the cutting third of the length, the last one up to the end of the envelope
        slab_z = list(np.linspace(0.0, length / 3.0, slabs)) + [length]
        slab_dr = rng.uniform(-2e-5, 2e-5, slabs)
        specs.append(A.subtract_spec(axis, slab_c, slab_z, slab_dr, -np.ones(3), np.ones(3), event_id=1 + k,
                                     tick=1000 * (k + 1), contact=0, rapid=(k == cuts // 2), plunge=(k == 1),
                                     r_dead_m=0.1 * radius, z_dead_m=0.2 * radius))
    tgt = _target_below(field, 0.35 * size[2]) if target else A.empty_target(len(field.cnt))
    return DexelCase(field, prof, specs, tgt, int(field.info[A.FI_NLEVELS]))


def dexel_run(case: DexelCase, call, events: int = 64):
    """Apply every subtraction of the case with ``call(f, prof, spec, tgt, acc, ev, sc) -> status`` on a copy
    of the field; returns (field, accum, events, statuses)."""
    from ..core.abi import event_buffer
    from ..removal import abi as A
    from ..removal.field import copy_field, grow_pool

    f = copy_field(case.field)
    acc = A.new_accum(case.levels)
    ev = event_buffer(events, A.EVENT_WIDTH)
    sc = A.make_scratch()
    statuses = []
    for spec in case.specs:
        status = call(f, case.profile, spec, case.target, acc, ev, sc)
        statuses.append(int(status))
        if status:
            f = grow_pool(f, max(1024, int(f.info[A.FI_N_BLOCKS])))
            statuses.append(int(call(f, case.profile, spec, case.target, acc, ev, sc)))
    return f, acc, ev, statuses


# ============================================================================ thermal step
def thermal_case(shape=(10, 8, 6), cell_mm: float = 2.0, coolant: bool = True, seed: int = 0):
    """(grid, scratch, mat, bnd, dt): Al 6061-T6 cells with a pocket of cut cells in the top layers, the
    bottom on the fixture, random temperatures 293-493 K, deposited energy in a few cells."""
    from ..thermal import abi as T
    from ..thermal import kernels as K
    from ..thermal.props import material_table_by_id
    from ..units import REFERENCE_TEMP_K, STEFAN_BOLTZMANN_W_M2K4

    rng = np.random.default_rng(seed)
    shape = tuple(int(s) for s in shape)
    nx, ny, nz = shape
    dx = float(cell_mm) * 1e-3
    mat = material_table_by_id(MATERIAL_ID)
    grid = T.new_work_grid(shape, (0.0, 0.0, 0.0), dx)
    f = np.ones(shape)
    px0, px1 = nx // 4, max(nx // 4 + 1, 3 * nx // 4)
    py0, py1 = ny // 4, max(ny // 4 + 1, 3 * ny // 4)
    depth = max(1, nz // 3)
    f[px0:px1, py0:py1, nz - depth:] = 0.0
    f[px0:px1, py0:py1, nz - depth - 1] = rng.choice([0.2, 0.5, 0.8], size=(px1 - px0, py1 - py0))
    f[rng.random(shape) < 0.05] = 0.4
    f[0, 0, :] = 0.0
    grid.f[:] = f
    temps = REFERENCE_TEMP_K + 200.0 * rng.random(shape)
    for idx in zip(*np.nonzero(f > 0.0)):
        grid.h[idx] = K.enthalpy(mat, float(temps[idx]))
        grid.T[idx] = temps[idx]
    grid.contact[:, :, 0] = np.uint8(1 << 4)
    bnd = T.new_work_boundary(2, (nx, ny))
    bnd.h_air[:] = [5.0, 6.0, 7.0, 8.0, 3.0, 9.0]
    bnd.near[0] = [0.5 * nx * dx, 0.5 * ny * dx, nz * dx, 2.5 * dx, 9000.0]
    bnd.near[1] = [0.2 * nx * dx, 0.7 * ny * dx, nz * dx, 1.5 * dx, 4000.0]
    bnd.wet[:, ny // 2:] = 1
    bnd.wet_geom[:] = [0.0, 0.0, dx]
    sc = bnd.scal
    sc[T.B_T_AIR] = REFERENCE_TEMP_K
    sc[T.B_T_FLUID] = 300.0
    sc[T.B_H_WET] = 2500.0
    sc[T.B_ALL_WET] = 0.0
    sc[T.B_EPS] = 0.35
    sc[T.B_T_RAD] = REFERENCE_TEMP_K
    sc[T.B_H_CONTACT] = 3000.0
    sc[T.B_T_FIX] = 305.0
    sc[T.B_FMIN] = 1e-9
    sc[T.B_COOLANT_ON] = 1.0 if coolant else 0.0
    sc[T.B_SIGMA] = STEFAN_BOLTZMANN_W_M2K4
    scratch = T.new_work_scratch(shape)
    hot = rng.integers(0, nx * ny * nz, size=max(1, nx * ny * nz // 50))
    scratch.src.reshape(-1)[hot] = rng.uniform(0.1, 3.0, hot.size)
    return grid, scratch, mat, bnd, 0.05


# ============================================================================ particle flight
def _material():
    from ..db import Database

    return Database.load().get("material", MATERIAL_ID)


def chips_model():
    from ..chips import params
    from ..chips.schema import settings as chips_settings

    return params.chip_model(_material(), chips_settings(None), params.load())


def chips_case(n: int = 32, seed: int = 0, liquid: bool = True, model=None, n_t: int = 10, dt_tick_s: float = 1e-3):
    """(P, H, W, S, C): ``n`` aluminium chip parcels over a 48 x 40 mm pile with a pocket, a rotating tool and
    holder, a nozzle and two jets (liquid or gas), a moving table; a few aimed particles weld, wrap, hook on
    the nozzle, reach the pan and settle in the pocket."""
    from ..chips import abi, params

    rng = np.random.default_rng(seed)
    C = model if model is not None else chips_model()
    pr = params.load()
    # pile
    nu, nv, cell = 24, 20, 2e-3
    H = abi.pile(nu, nv, (0.0, 0.0, 0.0), np.eye(3), cell)
    H.base_m[:] = 0.03
    H.base_m[8:14, 6:12] = 0.02
    H.base_m[0, :] = -np.inf
    H.base_kind[:] = abi.SURF_PART
    H.base_kind[0, :] = abi.SURF_NONE
    H.base_kind[-3:, :] = abi.SURF_TABLE
    mass = rng.uniform(0, 2e-6, (nu, nv)) * (rng.uniform(size=(nu, nv)) > 0.5)
    mass[0, :] = 0.0
    rho = float(C.rho_kg_m3[0])
    H.mass_kg[:] = mass
    H.bulk_m3[:] = mass * 15.0 / rho
    H.enthalpy_J[:] = mass * 1.5e5
    H.size_mass[:] = mass * 1.2e-3
    H.thick_mass[:] = mass * 1.5e-4
    H.width_mass[:] = mass * 2e-3
    H.length_mass[:] = mass * 4e-3
    H.repose_mass[:] = mass * math.tan(math.radians(40))
    H.long_kg[:] = mass * 0.3
    H.fines_kg[:] = mass * 0.1
    H.wet[:, 10:] = 1
    H.flush_Pa[:, 10:] = rng.uniform(0, 400, (nu, nv - 10))
    H.film_uv_m_s[:, 10:, 0] = 0.3
    H.surf_temp_K[:] = 300.0
    # window
    W = abi.window(n_t)
    W.t0_s[0] = 0.0
    W.t_end_s[0] = n_t * dt_tick_s
    W.dt_tick_s[0] = dt_tick_s
    for k in range(n_t + 1):
        W.grid_origin_M[k] = (1e-4 * k, 0.0, 0.0)
    # scene
    S = abi.scene(2, n_t, 1, 2)
    S.box_lo_M[:] = (-0.2, -0.2, -0.1)
    S.box_hi_M[:] = (0.3, 0.3, 0.4)
    S.floor[:] = (2, 0)
    for k in range(n_t + 1):
        S.cyl_base_M[0, k] = (0.024 + 2e-4 * k, 0.02, 0.025)
        S.cyl_base_M[1, k] = (0.024 + 2e-4 * k, 0.02, 0.047)
    S.cyl_axis_M[:, :, 2] = 1.0
    S.cyl_omega[:] = -1047.0
    S.cyl_radius[:] = (5e-3, 8e-3)
    S.cyl_length[:] = (22e-3, 60e-3)
    S.cyl_kind[:] = (abi.CYL_PADDLE, abi.CYL_SMOOTH)
    S.cyl_surf[:] = (abi.SURF_TOOL, abi.SURF_HOLDER)
    S.cyl_owner[:] = (0, 0)
    S.cyl_wrap[:] = (1, 1)
    S.cyl_weld[:] = (1, 0)
    S.sph_center_M[0] = (0.06, 0.02, 0.09)
    S.sph_radius[0] = 2e-3
    S.jet_origin_M[0] = (0.06, 0.02, 0.09)
    S.jet_dir_M[0] = (-0.5, 0.0, -math.sqrt(0.75))
    S.jet_speed[0] = 12.0
    S.jet_r0[0] = 1.5e-3
    S.jet_tan[0] = math.tan(math.radians(5))
    S.jet_gas[0] = 0 if liquid else 1
    S.jet_fluid[0] = (992.0, 1.2e-3, 0.55, 8.8) if liquid else (1.204, 1.81e-5, 0.0257, 0.71)
    S.jet_origin_M[1] = (-0.05, 0.02, 0.06)
    S.jet_dir_M[1] = (1.0, 0.0, 0.0)
    S.jet_speed[1] = 80.0
    S.jet_r0[1] = 1e-3
    S.jet_gas[1] = 1
    S.jet_fluid[1] = (1.204, 1.81e-5, 0.0257, 0.71)
    S.air[:] = (1.204, 1.81e-5, 0.0257, 0.71, 293.15)
    S.liquid[:] = (992.0, 1.2e-3, 0.55, 8.8, 300.0, 5e-5, 0.035)
    S.jet_const[:] = (5.8, 0.094)
    # particles
    P = abi.particles(n + 4)
    t = rng.uniform(5e-5, 3e-4, n)
    w = rng.uniform(1e-3, 3e-3, n)
    length = rng.uniform(2e-3, 60e-3, n)
    m_chip, dv, area, surface, psi = params.chip_shape(t, w, length, rho)
    a, b, c, d = params.hl_coefficients(pr.aero, psi)
    P.pos_M_m[:n, 0] = rng.uniform(0.0, 0.045, n)
    P.pos_M_m[:n, 1] = rng.uniform(0.0, 0.04, n)
    P.pos_M_m[:n, 2] = rng.uniform(0.032, 0.06, n)
    P.vel_M_m_s[:n] = rng.normal(0, 4.0, (n, 3))
    P.time_s[:n] = rng.uniform(0, 5e-3, n)
    P.mass_kg[:n] = rng.uniform(1e-7, 1e-6, n)
    P.temp_K[:n] = rng.uniform(320, 700, n)
    P.thickness_m[:n] = t
    P.width_m[:n] = w
    P.length_m[:n] = length
    P.dv_m[:n] = dv
    P.chip_mass_kg[:n] = m_chip
    P.area_m2[:n] = area
    P.surface_m2[:n] = surface
    P.hl_a[:n], P.hl_b[:n], P.hl_c[:n], P.hl_d[:n] = a, b, c, d
    P.form[:n] = rng.choice([11, 21, 42, 62, 70], n)
    P.status[:n] = abi.FLYING
    aims = {0: ((0.024, 0.032, 0.035), (0.0, -6.0, 0.0), 700.0, 62, 4e-3),
            1: ((0.024, 0.032, 0.036), (0.0, -6.0, 0.0), 320.0, 11, 60e-3),
            2: ((0.012, 0.02, 0.040), (6.0, 0.0, 0.0), 320.0, 62, 4e-3),
            4: ((0.06, 0.02, 0.080), (0.0, 0.0, 5.0), 320.0, 62, 12e-3),
            6: ((0.25, 0.25, -0.090), (0.0, 0.0, -5.0), 320.0, 70, 2e-3),
            7: ((0.019, 0.015, 0.0201), (0.0, 0.0, 0.0), 320.0, 70, 2e-3),
            8: ((0.29, 0.10, 0.10), (8.0, 0.0, 0.0), 320.0, 70, 2e-3)}
    for i, (pos, vel, temp, form, length_i) in aims.items():
        if i >= n:
            continue
        P.pos_M_m[i] = pos
        P.vel_M_m_s[i] = vel
        P.temp_K[i] = temp
        P.form[i] = form
        P.time_s[i] = 0.0
        P.length_m[i] = length_i
        m_i, dv_i, a_i, s_i, psi_i = params.chip_shape(t[i], w[i], length_i, rho)
        P.chip_mass_kg[i], P.dv_m[i], P.area_m2[i], P.surface_m2[i] = m_i, dv_i, a_i, s_i
        ha, hb, hc, hd = params.hl_coefficients(pr.aero, psi_i)
        P.hl_a[i], P.hl_b[i], P.hl_c[i], P.hl_d[i] = ha, hb, hc, hd
        P.bounces[i] = 3
    if n > 5:
        P.status[3] = abi.ON_TOOL
        P.status[5] = abi.FREE
    P.slot[:n] = 0
    P.pid[:n] = np.arange(n)
    P.n[0] = n
    return P, H, W, S, C


def stack_case(n: int = 6, model=None):
    """Particles dropped slowly onto one pile cell, one above the other: each settles on the chips of the
    ones before it (the pile coupling the GPU path must resolve on the host)."""
    from ..chips import abi, params

    P, H, W, S, C = chips_case(max(n, 8), seed=5, liquid=False, model=model)
    S.cyl_radius[:] = 0.0                      # no tool, no holder
    S.sph_radius[:] = 0.0
    S.jet_speed[:] = 0.0
    H.wet[:] = 0
    P.n[0] = n
    P.status[:] = abi.FREE
    P.status[:n] = abi.FLYING
    rho = float(C.rho_kg_m3[0])
    pr = params.load()
    for i in range(n):
        P.pos_M_m[i] = (0.0105, 0.0145, 0.0302 + 2.5e-4 * i)
        P.vel_M_m_s[i] = (0.0, 0.0, -0.05)
        P.time_s[i] = 0.0
        P.mass_kg[i] = 2e-5
        P.length_m[i] = 2e-3
        P.width_m[i] = 2e-3
        P.thickness_m[i] = 1e-4
        P.form[i] = 70
        P.bounces[i] = 0
        m_i, dv_i, a_i, s_i, psi_i = params.chip_shape(1e-4, 2e-3, 2e-3, rho)
        P.chip_mass_kg[i], P.dv_m[i], P.area_m2[i], P.surface_m2[i] = m_i, dv_i, a_i, s_i
        ha, hb, hc, hd = params.hl_coefficients(pr.aero, psi_i)
        P.hl_a[i], P.hl_b[i], P.hl_c[i], P.hl_d[i] = ha, hb, hc, hd
    return P, H, W, S, C


def clone(nt):
    """A deep copy of a NamedTuple of arrays."""
    return type(nt)(*(np.array(a, copy=True) for a in nt))
