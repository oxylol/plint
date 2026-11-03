"""Workpiece finite volumes against analytic transients (Carslaw & Jaeger 1959)."""

import math

import numpy as np
import pytest
from scipy.special import erfc

from cnctwin.thermal import kernels as K
from cnctwin.thermal import props
from cnctwin.thermal.abi import (B_ALL_WET, B_COOLANT_ON, B_FMIN, B_H_WET, B_SIGMA, B_T_AIR, B_T_FLUID, B_T_RAD, L_U1,
                                 N_L, new_work_boundary, new_work_grid, new_work_scratch)
from cnctwin.units import STEFAN_BOLTZMANN_W_M2K4

K_COND = 40.0
CP = 500.0
RHO = 8000.0
ALPHA = K_COND / (RHO * CP)


def constant_material():
    return props.mat_table({"value": RHO}, {"value": K_COND}, {"value": CP}, {"value": 1.2e-5}, "const")


def column(nz, dx, temp0):
    grid = new_work_grid((1, 1, nz), (0.0, 0.0, 0.0), dx)
    mat = constant_material()
    grid.f[:] = 1.0
    grid.h[:] = K.enthalpy(mat, temp0)
    grid.T[:] = temp0
    return grid, mat


def boundary(h_top, t_fluid, temp_air):
    bnd = new_work_boundary(1, (0, 0))
    bnd.near[0] = [0.0, 0.0, 1e9, 0.0, 0.0]        # no near zone
    sc = bnd.scal
    sc[B_T_AIR] = temp_air
    sc[B_T_FLUID] = t_fluid
    sc[B_T_RAD] = temp_air
    sc[B_FMIN] = 1e-9
    sc[B_SIGMA] = STEFAN_BOLTZMANN_W_M2K4
    sc[B_COOLANT_ON] = 1.0 if h_top > 0 else 0.0
    sc[B_ALL_WET] = 1.0 if h_top > 0 else 0.0
    sc[B_H_WET] = h_top
    return bnd


def test_semi_infinite_solid_with_surface_convection():
    """A deep column (side faces adiabatic) cooled on its top face by a fluid with large h:
    theta/theta_inf = erfc(eta) - exp(h x / k + h^2 alpha t / k^2) erfc(eta + h sqrt(alpha t) / k)."""
    dx = 0.5e-3
    nz = 200
    grid, mat = column(nz, dx, 293.15)
    grid.f[:] = 1.0
    bnd = boundary(5000.0, 393.15, 293.15)
    # only the top face convects: mark it as the single wetted face by giving every other face zero h
    bnd.h_air[:] = 0.0
    scratch = new_work_scratch(grid.f.shape)
    out = np.zeros(N_L)
    dt = 0.01
    t_end = 4.0
    # the column's side and bottom faces are exposed too; make them adiabatic with the contact class
    grid.contact[:] = np.uint8(0b001111)          # +-x, +-y faces: contact with h_contact = 0
    grid.contact[0, 0, 0] |= np.uint8(1 << 4)    # bottom face
    for _ in range(int(round(t_end / dt))):
        scratch.src[:] = 0.0
        K.work_step(grid, scratch, mat, bnd, dt, out)
    h = 5000.0
    depth = (nz - np.arange(nz) - 0.5) * dx          # depth of each cell centre below the top face
    eta = depth / (2.0 * math.sqrt(ALPHA * t_end))
    beta = h * math.sqrt(ALPHA * t_end) / K_COND
    analytic = erfc(eta) - np.exp(h * depth / K_COND + beta * beta) * erfc(eta + beta)
    numeric = (grid.T[0, 0, :] - 293.15) / 100.0
    near = depth < 6e-3
    assert np.max(np.abs(numeric[near] - analytic[near])) < 0.01


def test_lumped_cooling_of_a_small_part():
    """Bi << 1: T - T_f = (T0 - T_f) exp(-h A t / (m c)) (lumped capacity)."""
    dx = 2e-3
    n = 5
    grid = new_work_grid((n, n, n), (0.0, 0.0, 0.0), dx)
    mat = constant_material()
    grid.f[:] = 1.0
    grid.h[:] = K.enthalpy(mat, 393.15)
    grid.T[:] = 393.15
    bnd = boundary(30.0, 293.15, 293.15)           # every exposed face wetted with h = 30
    scratch = new_work_scratch(grid.f.shape)
    out = np.zeros(N_L)
    side = n * dx
    tau = RHO * CP * side ** 3 / (30.0 * 6.0 * side * side)
    dt = tau / 200.0
    for _ in range(200):
        scratch.src[:] = 0.0
        K.work_step(grid, scratch, mat, bnd, dt, out)
    mean = float(np.mean(grid.T))
    expected = 293.15 + 100.0 * math.exp(-1.0)
    assert mean == pytest.approx(expected, abs=0.25)       # Bi = 30 * 5 mm / 40 = 0.004
    assert out[L_U1] == pytest.approx(RHO * side ** 3 * CP * (mean - 293.15), rel=2e-3)


def test_constant_surface_flux_on_a_half_space():
    """Uniform flux q into the top: theta = 2 q sqrt(alpha t / pi) / k exp(-x^2/4 alpha t) - q x / k erfc(.)."""
    dx = 0.5e-3
    nz = 160
    grid, mat = column(nz, dx, 293.15)
    bnd = boundary(0.0, 293.15, 293.15)
    bnd.h_air[:] = 0.0
    bnd.scal[4] = 0.0
    grid.contact[:] = np.uint8(0b111111)          # every exposed face adiabatic (contact class, h = 0)
    scratch = new_work_scratch(grid.f.shape)
    out = np.zeros(N_L)
    q = 2.0e5
    dt = 0.005
    t_end = 2.0
    area = dx * dx
    for _ in range(int(round(t_end / dt))):
        scratch.src[:] = 0.0
        scratch.src[0, 0, nz - 1] = q * area * dt
        K.work_step(grid, scratch, mat, bnd, dt, out)
    depth = (nz - np.arange(nz) - 0.5) * dx
    s = math.sqrt(ALPHA * t_end)
    analytic = 2 * q * s / (K_COND * math.sqrt(math.pi)) * np.exp(-depth ** 2 / (4 * s * s)) - \
        q * depth / K_COND * erfc(depth / (2 * s))
    numeric = grid.T[0, 0, :] - 293.15
    near = depth < 5e-3
    surface = 2 * q * s / (K_COND * math.sqrt(math.pi))
    assert np.max(np.abs(numeric[near] - analytic[near])) < 0.02 * surface
    assert float(np.sum(numeric)) * RHO * CP * dx ** 3 == pytest.approx(q * area * t_end, rel=1e-9)


def test_thin_wall_inside_one_cell_cools_through_both_faces():
    """A 0.5 mm wall in 2 mm cells (f = 0.25, the shell model of docs/models/thermal.md): lumped
    plate cooling from both faces, tau = rho c t / (2 h), conducting in its own plane."""
    dx = 2e-3
    shape = (3, 6, 6)
    grid = new_work_grid(shape, (0.0, 0.0, 0.0), dx)
    mat = constant_material()
    grid.f[1, :, :] = 0.25
    grid.h[1, :, :] = K.enthalpy(mat, 393.15)
    grid.T[:] = np.nan
    grid.T[1, :, :] = 393.15
    bnd = boundary(50.0, 293.15, 293.15)
    grid.contact[1, :, :] = np.uint8(0b111100)       # wall edges (+-y, +-z) adiabatic: only the two faces cool
    scratch = new_work_scratch(shape)
    out = np.zeros(N_L)
    thickness = 0.25 * dx
    tau = RHO * CP * thickness / (2.0 * 50.0)
    steps = 100
    for _ in range(steps):
        scratch.src[:] = 0.0
        K.work_step(grid, scratch, mat, bnd, tau / steps, out)
    mean = float(np.nanmean(grid.T))
    assert mean == pytest.approx(293.15 + 100.0 * math.exp(-1.0), abs=0.4)
