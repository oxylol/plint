"""Settling into the height field, bulk packing, avalanches to the angle of repose, spill over edges."""

from __future__ import annotations

import math

import numpy as np

from cnctwin.chips import abi, kernels, params
from cnctwin.chips.schema import settings as chips_settings

import chipsfix as fx


def _model():
    return params.chip_model(fx.project().material, chips_settings(None), params.load())


def _drop_set(n, cell, nu, nv, seed=5):
    pr = params.load()
    r = np.random.default_rng(seed)
    P = abi.particles(n)
    t = np.full(n, 1.5e-4)
    w = np.full(n, 2e-3)
    length = np.full(n, 3e-3)
    m, dv, area, surface, psi = params.chip_shape(t, w, length, 2700.0)
    a, b, c, d = params.hl_coefficients(pr.aero, psi)
    P.pos_M_m[:, 0] = r.uniform(0.2, nu - 0.2, n) * cell
    P.pos_M_m[:, 1] = r.uniform(0.2, nv - 0.2, n) * cell
    P.pos_M_m[:, 2] = r.uniform(0.005, 0.02, n)
    P.mass_kg[:] = r.uniform(1e-7, 1e-6, n)
    P.temp_K[:] = 400.0
    P.thickness_m[:], P.width_m[:], P.length_m[:] = t, w, length
    P.dv_m[:], P.chip_mass_kg[:], P.area_m2[:], P.surface_m2[:] = dv, m, area, surface
    P.hl_a[:], P.hl_b[:], P.hl_c[:], P.hl_d[:] = a, b, c, d
    P.form[:] = np.where(np.arange(n) % 2 == 0, 62, 42)
    P.status[:] = abi.FLYING
    P.n[0] = n
    return P


def _world(nu, nv, cell, t_end=0.5):
    H = abi.pile(nu, nv, (0.0, 0.0, 0.0), np.eye(3), cell)
    H.base_m[:] = 0.0
    H.base_kind[:] = abi.SURF_TABLE
    W = abi.window(10)
    W.t_end_s[0] = t_end
    W.dt_tick_s[0] = t_end / 10
    S = abi.scene(0, 10, 0, 0)
    S.box_lo_M[:] = (-1.0, -1.0, -1.0)
    S.box_hi_M[:] = (1.0, 1.0, 1.0)
    S.air[:] = (1.204, 1.81e-5, 0.0257, 0.71, 293.15)
    return H, W, S


def test_dropped_chips_settle_and_build_the_bulk_height():
    C = _model()
    cell, nu, nv = 2e-3, 12, 12
    P = _drop_set(200, cell, nu, nv)
    mass0 = float(P.mass_kg.sum())
    expected_bulk = float(np.sum(P.mass_kg * C.form_R[P.form] / C.rho_kg_m3[0]))
    H, W, S = _world(nu, nv, cell)
    out, ev = abi.step_out(16)
    kernels.advance_particles(P, H, W, S, C, out, ev)
    assert np.all(P.status == abi.FREE)                                   # every chip came to rest
    assert math.isclose(H.mass_kg.sum(), mass0, rel_tol=1e-13)
    assert math.isclose(H.bulk_m3.sum(), expected_bulk, rel_tol=1e-13)
    # bulk volume ratio of the forms (R = 3 for loose arcs 62, 30 for short washer helices 42)
    assert C.form_R[62] == 3 and C.form_R[42] == 30
    height = H.bulk_m3 / cell ** 2
    assert math.isclose(height.sum() * cell ** 2, expected_bulk, rel_tol=1e-13)
    assert height.max() > 0
    # energy: the chips left at 400 K; what they lost to the air on the way down plus what the pile holds
    h400 = kernels.enthalpy(400.0, C.cp_T_K, C.cp_J_kgK, C.h_J_kg)
    assert out[abi.OUT_HEAT_AIR_J] > 0
    assert math.isclose(H.enthalpy_J.sum() + out[abi.OUT_HEAT_AIR_J], mass0 * h400, rel_tol=1e-12)


def test_heap_relaxes_to_the_angle_of_repose_conserving_mass():
    C = _model()
    cell, nu, nv = 2e-3, 21, 21
    H = abi.pile(nu, nv, (0.0, 0.0, 0.0), np.eye(3), cell)
    H.base_m[:] = 0.0
    H.base_kind[:] = abi.SURF_TABLE
    mass = 4e-4
    kernels.deposit(H, C, 10, 10, mass, mass * 1e5, 1.2e-3, 1.5e-4, 2e-3, 3e-3, 62)
    total0 = (H.mass_kg.sum(), H.bulk_m3.sum(), H.enthalpy_J.sum())
    out = np.zeros(abi.N_OUT)
    kernels.relax_pile(H, 5000, 1e-16, out)
    assert math.isclose(H.mass_kg.sum(), total0[0], rel_tol=1e-12)
    assert math.isclose(H.bulk_m3.sum(), total0[1], rel_tol=1e-12)
    assert math.isclose(H.enthalpy_J.sum(), total0[2], rel_tol=1e-12)
    top = H.base_m + H.bulk_m3 / cell ** 2
    tan_r = C.form_repose_tan[62]
    slope_u = np.abs(np.diff(top, axis=0)) / cell
    slope_v = np.abs(np.diff(top, axis=1)) / cell
    assert max(slope_u.max(), slope_v.max()) <= tan_r * (1 + 1e-6) + 1e-9
    assert np.count_nonzero(H.mass_kg > 0) > 9                       # it spread
    assert out[abi.OUT_PAN_MASS_KG] == 0.0


def test_chips_spill_over_an_edge_into_the_pan():
    C = _model()
    cell = 2e-3
    H = abi.pile(5, 1, (0.0, 0.0, 0.0), np.eye(3), cell)
    H.base_m[:] = 0.0
    H.base_m[4, 0] = -np.inf                                         # the part ends here
    H.base_kind[:] = abi.SURF_PART
    kernels.deposit(H, C, 3, 0, 2e-5, 2e-5 * 5e4, 1.2e-3, 1.5e-4, 2e-3, 3e-3, 70)
    m0 = H.mass_kg.sum()
    out = np.zeros(abi.N_OUT)
    kernels.relax_pile(H, 200, 1e-18, out)
    assert out[abi.OUT_PAN_MASS_KG] > 0
    assert math.isclose(H.mass_kg.sum() + out[abi.OUT_PAN_MASS_KG], m0, rel_tol=1e-12)
    # what stays at the edge cell is at most half a cell at the repose slope
    assert H.bulk_m3[3, 0] / cell ** 2 <= 0.5 * C.form_repose_tan[70] * cell * (1 + 1e-9) or \
        H.bulk_m3[3, 0] / cell ** 2 - H.bulk_m3[2, 0] / cell ** 2 <= C.form_repose_tan[70] * cell * (1 + 1e-9)


def test_chips_fill_a_pocket_before_the_floor_around_it():
    """Chips landing next to a pocket avalanche into it: hole/pocket fill."""
    C = _model()
    cell, nu, nv = 2e-3, 15, 15
    H = abi.pile(nu, nv, (0.0, 0.0, 0.0), np.eye(3), cell)
    H.base_m[:] = 0.03
    H.base_m[5:10, 5:10] = 0.02                                       # 10 mm deep pocket
    H.base_kind[:] = abi.SURF_PART
    mass = 3e-5
    for iu, iv in ((4, 7), (7, 4), (10, 7), (7, 10)):                 # on the rim
        kernels.deposit(H, C, iu, iv, mass, 0.0, 1.2e-3, 1.5e-4, 2e-3, 3e-3, 70)
    out = np.zeros(abi.N_OUT)
    kernels.relax_pile(H, 2000, 1e-18, out)
    pocket = H.mass_kg[5:10, 5:10].sum()
    assert pocket > 0.5 * H.mass_kg.sum()
    assert math.isclose(H.mass_kg.sum(), 4 * mass, rel_tol=1e-12)
