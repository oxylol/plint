"""Displacement grids: residual-stress release against the Treuting-Read layer-removal relation, thermo-elastic
expansion and bow of a free bar, the uncut stock (d = 0), clamping, the profile balance and the temperature
sampling."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.unit.structure import structfix as fx
from cnctwin.structure import displacement as DS
from cnctwin.structure import voxel as VX
from cnctwin.structure import work as WK
from cnctwin.structure.lathe import LatheSupports
from cnctwin.structure.materials import work_material
from cnctwin.structure.protocols import BoxGeometry
from cnctwin.structure.schema import settings

E, NU = 70e9, 0.33
SIGMA = 100e6
PROFILE = [[z, SIGMA / 1e6 * math.cos(2 * math.pi * z)] for z in np.linspace(0.0, 1.0, 41)]


def strip(cut_layers=2, cell=0.5e-3):
    """30 x 3 x 4 mm strip (cells 0.5 mm), the top layers removed."""
    size = np.array([0.03, 0.003, 0.004])
    grid = VX.VoxelGrid((0.0, 0.0, 0.0), cell, tuple(int(round(s / cell)) for s in size))
    mat = work_material(fx.material(profile=PROFILE, thickness_mm=4.0))
    active = np.ones(grid.shape, bool)
    active[:, :, grid.shape[2] - cut_layers:] = False
    return grid, mat, active, size


def curvature_x(model, u):
    """-d2 uz / dx2 of the bottom centre line over the middle 60 % (a parabola fit)."""
    ijk = model.node_ijk
    jm = int(np.median(ijk[:, 1]))
    line = np.nonzero((ijk[:, 1] == jm) & (ijk[:, 2] == 0))[0]
    x = model.node_xyz()[line, 0]
    w = u.reshape(-1, 3)[line, 2]
    L = x.max()
    mid = (x > 0.2 * L) & (x < 0.8 * L)
    a = np.polyfit(x[mid], w[mid], 2)[0]
    return -2.0 * a


def test_layer_removal_against_treuting_read():
    grid, mat, active, size = strip()
    model = VX.VoxelModel(grid, active, mat.E_Pa, mat.nu, mat.rho_kg_m3)
    sigma, layers = DS.plate_stress(mat, "rolled_plate", grid, model.cells, np.zeros(3), size)
    assert np.all(sigma[:, 1:] == 0.0)                 # L along x (the longer side); LT absent
    lay = layers["L"]
    assert abs(np.sum(lay)) < 1e-6 * SIGMA and abs(np.sum(lay * (np.arange(8) + 0.5))) < 1e-5 * SIGMA
    system = VX.System(model, None, VX.minimal_constraints(model), settings({}))
    u = system.solve(model.initial_stress_load(sigma))
    kappa = curvature_x(model, u)
    z_edges = np.arange(9) * grid.cell_m
    h1 = 0.003
    expected = DS.treuting_read_curvature(z_edges, lay, np.zeros(8), h1, mat.E_Pa, mat.nu, plate=False)[0]
    assert abs(expected) > 0.5                          # a curvature radius of about 1 m
    assert kappa == pytest.approx(expected, rel=0.03)


def test_treuting_read_relation_forms():
    z = np.linspace(0.0, 0.004, 9)
    s = np.array([10, 5, 0, -5, -5, 0, 5, 10.0]) * 1e6
    beam = DS.treuting_read_curvature(z, s, np.zeros(8), 0.003, E, NU, plate=False)
    plate = DS.treuting_read_curvature(z, s, np.zeros(8), 0.003, E, NU, plate=True)
    assert plate[0] == pytest.approx(beam[0]) and plate[1] == pytest.approx(-NU * beam[0])
    equi = DS.treuting_read_curvature(z, s, s, 0.003, E, NU, plate=True)
    # equi-biaxial: the plate's biaxial bending stiffness D (1 + nu) = E h^3 / (12 (1 - nu))
    assert equi[0] == pytest.approx(equi[1]) and equi[0] == pytest.approx(beam[0] * (1 - NU))


def free_bar_inputs(thermal, geo=None):
    proj = fx.project(stock={"kind": "block", "size_mm": [40, 4, 4], "form": "rolled_plate"})
    mat = work_material(proj.material)
    geo = geo or BoxGeometry([0, 0, 0], [0.04, 0.004, 0.004])
    grid = VX.VoxelGrid((0.0, 0.0, 0.0), 1e-3, (40, 4, 4))
    return WK.WorkInputs(0, grid, mat, settings({}), geo, BoxGeometry([0, 0, 0], [0.04, 0.004, 0.004]),
                         "rolled_plate", np.zeros(3), np.array([0.04, 0.004, 0.004]), "voxel", VX.Supports.empty(),
                         LatheSupports(), False, thermal)


def test_thermo_elastic_bar_expansion_and_bow():
    dT = 30.0
    th = fx.thermal_grid([0, 0, 0], [0.04, 0.004, 0.004], 0.5e-3, lambda c: 293.15 + dT + 0.0 * c[..., 0])
    sol = WK.compute(free_bar_inputs(th), None, None)
    d = sol.displacement.thermal_m
    # free expansion relative to the 3-2-1 point at the origin corner: exact (patch test)
    assert d[40, 0, 0, 0] == pytest.approx(23e-6 * dT * 0.04, rel=1e-9)
    assert d[40, 4, 4] == pytest.approx(23e-6 * dT * np.array([0.04, 0.004, 0.004]), rel=1e-9)
    assert np.allclose(sol.displacement.total_m, d) and not np.any(sol.displacement.clamp_m)
    # a linear gradient through the thickness: free bow kappa = alpha g (layer-wise constant cells)
    g = 5000.0                                         # K/m: 20 K over the 4 mm
    th = fx.thermal_grid([0, 0, 0], [0.04, 0.004, 0.004], 0.5e-3, lambda c: 293.15 + g * (c[..., 2] - 0.002))
    sol = WK.compute(free_bar_inputs(th), None, None)
    grid = VX.VoxelGrid((0.0, 0.0, 0.0), 1e-3, (40, 4, 4))
    model = VX.VoxelModel(grid, np.ones(grid.shape, bool), E, NU, 2700.0)
    u = model.from_lattice(sol.displacement.thermal_m)
    kappa = curvature_x(model, u)
    T_layers = g * ((np.arange(4) + 0.5) * 1e-3 - 0.002)
    eps = 23e-6 * T_layers
    expected = 12.0 / 0.004 ** 3 * np.sum(eps * ((np.arange(4) + 0.5) * 1e-3 - 0.002) * 1e-3)
    # the hotter top grows longer: the bar bows with its ends down (kappa = -w'' > 0), kappa = alpha g
    assert kappa == pytest.approx(expected, rel=0.02) and kappa > 0
    assert expected == pytest.approx(23e-6 * g * (1 - 1 / 16), rel=1e-9)   # layer-wise constant: (1 - 1/n^2)


def test_uncut_stock_is_undistorted_and_cutting_releases_stress():
    proj = fx.project(stock={"kind": "block", "size_mm": [24, 12, 6], "form": "rolled_plate"},
                      mat=fx.material(profile=PROFILE, thickness_mm=6.0))
    mat = work_material(proj.material)
    geo = BoxGeometry([0, 0, 0], [0.024, 0.012, 0.006])
    grid = VX.VoxelGrid((0.0, 0.0, 0.0), 1e-3, (24, 12, 6))
    base = dict(workpiece=0, grid=grid, material=mat, settings=settings({}), geometry=geo,
                stock_geometry=BoxGeometry([0, 0, 0], [0.024, 0.012, 0.006]), form="rolled_plate",
                stock_lo=np.zeros(3), stock_hi=np.array([0.024, 0.012, 0.006]), kind="voxel",
                supports=VX.Supports.empty(), lathe=LatheSupports(), clamped=False, thermal=None)
    inputs = WK.WorkInputs(**base)
    ref = WK.residual_reference(inputs)
    sol = WK.compute(inputs, None, ref)
    assert np.max(np.abs(sol.displacement.residual_m)) < 1e-15
    geo.remove_box([0.002, 0.002, 0.003], [0.022, 0.010, 0.007])
    sol = WK.compute(WK.WorkInputs(**base), None, ref)
    assert np.max(np.abs(sol.displacement.residual_m)) > 1e-7
    # the 3-2-1 points stay put in the released state
    model = VX.VoxelModel(grid, sol.active, mat.E_Pa, mat.nu, mat.rho_kg_m3)
    fixed = VX.minimal_constraints(model)
    u = model.from_lattice(sol.displacement.residual_m)
    assert np.max(np.abs(u[fixed])) < 1e-15


def test_clamp_distortion_and_spring_back():
    proj = fx.project(stock={"kind": "block", "size_mm": [40, 30, 12]})
    mat = work_material(proj.material)
    geo = BoxGeometry([0, 0, 0], [0.04, 0.03, 0.012])
    grid = VX.VoxelGrid((0.0, 0.0, 0.0), 2e-3, (20, 15, 6))
    sup = VX.supports_from_holding(fx.vise_supports(clamp_N=5000.0))
    common = dict(workpiece=0, grid=grid, material=mat, settings=settings({}), geometry=geo, stock_geometry=geo,
                  form="", stock_lo=np.zeros(3), stock_hi=np.array([0.04, 0.03, 0.012]), kind="voxel",
                  supports=sup, lathe=LatheSupports(), thermal=None)
    held = WK.compute(WK.WorkInputs(clamped=True, **common), None, None)
    c = held.displacement.clamp_m
    # the jaws squeeze the part along y: the moving-jaw side moves toward the fixed jaw (-y)
    assert c[10, 15, 2, 1] < 0.0 and abs(c[10, 15, 2, 1]) > 1e-7
    assert held.displacement.clamped
    released = WK.compute(WK.WorkInputs(clamped=False, **common), None, None)
    assert not np.any(released.displacement.clamp_m) and not released.displacement.clamped


def test_profile_balance_and_temperature_sampling():
    w = np.ones(10)
    zeta = (np.arange(10) + 0.5) / 10
    vals = 3.0 + 2.0 * zeta + np.cos(4 * zeta)
    b = DS.balance(vals, zeta, w)
    assert abs(np.sum(b)) < 1e-12 and abs(np.sum(b * (zeta - zeta.mean()))) < 1e-12
    grid = VX.VoxelGrid((0.0, 0.0, 0.0), 2e-3, (5, 4, 3))
    cells = np.argwhere(np.ones(grid.shape, bool))
    lo, cell, T = fx.thermal_grid([0, 0, 0], [0.01, 0.008, 0.006], 1e-3,
                                  lambda c: 300.0 + 1000.0 * c[..., 0] + 500.0 * c[..., 2])
    T = T.copy()
    T[0, 0, 0] = np.nan                                 # a cell outside material
    got = DS.cell_temperatures(grid, cells, lo, cell, T)
    centres = grid.origin + (cells + 0.5) * grid.cell_m
    exact = 300.0 + 1000.0 * centres[:, 0] + 500.0 * centres[:, 2]
    far = np.all(cells > 0, axis=1)
    assert np.allclose(got[far], exact[far], rtol=1e-12)
    assert np.allclose(DS.cell_temperatures(grid, cells, lo, cell, None), 293.15)
