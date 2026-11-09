"""Voxel finite elements: a cantilever against beam theory, a clamped plate against plate theory (with the
documented thick-plate term), multigrid against the direct solver, Rayleigh-Ritz against a full eigen solve,
support springs, the free thermal expansion patch, the grid size rule and connectivity."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cnctwin.structure import voxel as VX
from cnctwin.structure.schema import settings

E, NU, RHO = 70e9, 0.33, 2700.0


def cantilever(n_len=40, n_sec=4, cell=1e-3):
    grid = VX.VoxelGrid((0.0, 0.0, 0.0), cell, (n_len, n_sec, n_sec))
    model = VX.VoxelModel(grid, np.ones(grid.shape, bool), E, NU, RHO)
    fixed = np.nonzero(np.repeat(model.node_ijk[:, 0] == 0, 3))[0]
    return grid, model, fixed


def test_cantilever_against_beam_theory():
    grid, model, fixed = cantilever()
    system = VX.System(model, None, fixed, settings({}))
    lam, V = system.eigen(4, 1e-10)
    f = np.sqrt(lam) / (2 * math.pi)
    L, b = 0.04, 0.004
    I = b ** 4 / 12.0
    A = b * b
    f_eb = 1.875104069 ** 2 / (2 * math.pi * L * L) * math.sqrt(E * I / (RHO * A))
    assert f[0] == pytest.approx(f_eb, rel=0.01) and f[1] == pytest.approx(f[0], rel=1e-8)
    # canonical pair: mode 0 along y (bending in the x-y plane), mode 1 along z, each sign-normalised
    tip = np.nonzero(model.node_ijk[:, 0] == 40)[0]
    u0 = V[:, 0].reshape(-1, 3)[tip].mean(axis=0)
    u1 = V[:, 1].reshape(-1, 3)[tip].mean(axis=0)
    assert abs(u0[1]) > 100 * abs(u0[2]) and abs(u1[2]) > 100 * abs(u1[1])
    P = 10.0
    F = np.zeros(model.n_dof)
    F[3 * tip + 2] = P / len(tip)
    w = system.solve(F)[3 * tip + 2].mean()
    G = E / (2 * (1 + NU))
    k = 10 * (1 + NU) / (12 + 11 * NU)
    assert w == pytest.approx(P * L ** 3 / (3 * E * I) + P * L / (k * G * A), rel=0.02)


@pytest.mark.parametrize("n,nt", [(40, 2), (40, 1)])
def test_clamped_plate_against_plate_theory(n, nt):
    """Clamped square plate under uniform pressure: Kirchhoff w = 0.00126 q a^4 / D (Timoshenko & Woinowsky-Krieger
    Table 35) times 1 + c (t / a)^2 for transverse shear (c = 14.5 measured, independent of the cell count; the
    FSDT value of the literature is 15-19): the voxel model within 1.5 % of it."""
    a = 0.04
    c = a / n * (2 if nt == 1 else 1)
    n_cells = int(round(a / c))
    t = nt * c
    grid = VX.VoxelGrid((0.0, 0.0, 0.0), c, (n_cells, n_cells, nt))
    model = VX.VoxelModel(grid, np.ones(grid.shape, bool), E, 0.3, RHO)
    ijk = model.node_ijk
    edge = (ijk[:, 0] == 0) | (ijk[:, 0] == n_cells) | (ijk[:, 1] == 0) | (ijk[:, 1] == n_cells)
    system = VX.System(model, None, np.nonzero(np.repeat(edge, 3))[0], settings({}))
    q = 1e5
    top = np.nonzero(ijk[:, 2] == nt)[0]
    w = np.full(len(top), c * c)
    w[(ijk[top, 0] == 0) | (ijk[top, 0] == n_cells)] *= 0.5
    w[(ijk[top, 1] == 0) | (ijk[top, 1] == n_cells)] *= 0.5
    f = np.zeros(model.n_dof)
    f[3 * top + 2] = -q * w
    u = system.solve(f)
    centre = np.nonzero((ijk[:, 0] == n_cells // 2) & (ijk[:, 1] == n_cells // 2))[0]
    w_max = -float(u[3 * centre + 2].mean())
    D = E * t ** 3 / (12 * (1 - 0.3 ** 2))
    ratio = w_max / (0.00126 * q * a ** 4 / D)
    assert ratio == pytest.approx(1.0 + 14.5 * (t / a) ** 2, rel=0.015)


def test_multigrid_equals_direct():
    grid = VX.VoxelGrid((0.0, 0.0, 0.0), 1e-3, (12, 9, 6))
    active = np.ones(grid.shape, bool)
    active[3:9, 2:7, 3:] = False                      # a pocket
    model = VX.VoxelModel(grid, active, E, NU, RHO)
    fixed = VX.minimal_constraints(model)
    springs = model.spring_data(model.node_xyz()[model.node_ijk[:, 2] == 0][:12],
                                np.tile(np.eye(3) * 1e8, (12, 1, 1)))
    direct = VX.System(model, springs, None, settings({"direct_max_dof": 100000}))
    mg = VX.System(model, springs, None, settings({"direct_max_dof": 0, "gmg_coarse_dof": 200}))
    assert direct.solver.kind == "direct" and mg.solver.kind == "multigrid" and mg.solver._mg.n_levels >= 2
    f = np.random.default_rng(0).standard_normal(model.n_dof)
    xd, xm = direct.solve(f), mg.solve(f)
    assert np.allclose(xm, xd, rtol=1e-7, atol=1e-9 * np.abs(xd).max())
    ld, Vd = direct.eigen(6, 1e-12)
    lm, Vm = mg.eigen(6, 1e-12)
    assert np.allclose(lm, ld, rtol=1e-8)
    assert np.allclose(np.abs(np.einsum("ij,ij->j", Vd, model.M @ Vm)), 1.0, atol=1e-6)
    assert len(fixed) == 6


def test_ritz_refresh_matches_a_full_solve():
    grid = VX.VoxelGrid((0.0, 0.0, 0.0), 1e-3, (14, 10, 6))
    active = np.ones(grid.shape, bool)
    model0 = VX.VoxelModel(grid, active, E, NU, RHO)
    pts = model0.node_xyz()[model0.node_ijk[:, 2] == 0]
    k = np.tile(np.eye(3) * 5e7, (len(pts), 1, 1))
    st = settings({})
    sys0 = VX.System(model0, model0.spring_data(pts, k), None, st)
    lam0, V0 = sys0.eigen(8, 1e-12)
    removed = active.copy()
    removed[4:10, 3:7, 5] = False                     # a shallow pocket: a small change
    model1 = VX.VoxelModel(grid, removed, E, NU, RHO)
    sys1 = VX.System(model1, model1.spring_data(pts, k), None, st)
    guess = np.stack([model1.from_lattice(model0.to_lattice(V0[:, j])) for j in range(8)], axis=1)
    lam_r, V_r, res, ok = sys1.ritz(guess, lam0, 8, 2e-3, 6)
    lam_f, V_f = sys1.eigen(8, 1e-12)
    assert ok and res <= 2e-3
    # energy-norm residual eps bounds the eigenvalue error by about eps^2
    assert np.allclose(lam_r[:6], lam_f[:6], rtol=1e-5)
    assert np.all(lam_r >= lam_f * (1 - 1e-9))        # Ritz values are upper bounds
    mac = np.einsum("ij,ij->j", V_r[:, :6], sys1.M @ V_f[:, :6]) ** 2
    assert np.all(mac > 1 - 1e-5)


def test_ritz_accepts_a_guess_the_change_hardly_moved_after_one_block_solve():
    """The first residual check is on the guess's own Rayleigh-Ritz pairs: converged modes of the same model pass it
    with one block solve (the check needs K^-1 r), by the same energy-norm criterion as later iterations."""
    grid = VX.VoxelGrid((0.0, 0.0, 0.0), 1e-3, (14, 10, 6))
    model = VX.VoxelModel(grid, np.ones(grid.shape, bool), E, NU, RHO)
    pts = model.node_xyz()[model.node_ijk[:, 2] == 0]
    k = np.tile(np.eye(3) * 5e7, (len(pts), 1, 1))
    system = VX.System(model, model.spring_data(pts, k), None, settings({}))
    lam, V = system.eigen(8, 1e-12)
    calls = []
    solve = system.solve
    system.solve = lambda f: (calls.append(1), solve(f))[1]
    lam_r, V_r, res, ok = system.ritz(V, lam, 8, 2e-3, 6)
    assert ok and res <= 1e-6 and len(calls) == 1
    assert np.allclose(lam_r, lam, rtol=1e-10)


def test_block_solves_do_not_depend_on_the_block_width():
    """Every column of a block CG solve is its own recurrence: one block of 20 columns gives bit-identical solutions
    to blocks of at most 6 (voxel.BLOCK)."""
    grid = VX.VoxelGrid((0.0, 0.0, 0.0), 1e-3, (24, 16, 10))
    model = VX.VoxelModel(grid, np.ones(grid.shape, bool), E, NU, RHO)
    pts = model.node_xyz()[model.node_ijk[:, 2] == 0]
    k = np.tile(np.eye(3) * 5e7, (len(pts), 1, 1))
    st = settings({"direct_max_dof": 100})                 # the multigrid path
    system = VX.System(model, model.spring_data(pts, k), None, st)
    assert system.solver.kind == "multigrid"
    F = np.random.default_rng(3).standard_normal((model.n_dof, 20))
    wide = system.solve(F)
    narrow = np.hstack([system.solve(F[:, a:a + 6]) for a in range(0, 20, 6)])
    assert np.array_equal(wide, narrow)


def test_springs_and_mass_give_the_bounce_frequency():
    """A stiff block on four soft vertical springs: f = sqrt(4 k / m) / 2 pi."""
    grid = VX.VoxelGrid((0.0, 0.0, 0.0), 2e-3, (5, 5, 3))
    model = VX.VoxelModel(grid, np.ones(grid.shape, bool), 200e9, 0.3, 7800.0)
    corners = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0.01, 0.01, 0]], float)
    kz = 1e4                                          # 1e-5 of the block's own corner stiffness
    kmat = np.zeros((4, 3, 3))
    kmat[:, 2, 2] = kz
    kmat[:, 0, 0] = kmat[:, 1, 1] = 4e4
    system = VX.System(model, model.spring_data(corners, kmat), None, settings({}))
    lam, _ = system.eigen(6, 1e-12)
    m = 7800.0 * 0.01 * 0.01 * 0.006
    f = np.sqrt(lam) / (2 * math.pi)
    f_b = math.sqrt(4 * kz / m) / (2 * math.pi)
    assert np.min(np.abs(f - f_b)) / f_b < 2e-4


def test_free_thermal_expansion_is_exact_with_a_3_2_1_support():
    grid = VX.VoxelGrid((0.0, 0.0, 0.0), 2e-3, (8, 5, 3))
    active = np.ones(grid.shape, bool)
    active[2:6, 1:4, 2] = False
    model = VX.VoxelModel(grid, active, E, NU, RHO)
    fixed = VX.minimal_constraints(model)
    system = VX.System(model, None, fixed, settings({}))
    eps = 23e-6 * 40.0
    u = system.solve(model.thermal_load(np.full(len(model.cells), eps)))
    x = model.node_xyz()
    A = model.node_xyz()[fixed[0] // 3]
    assert np.allclose(u.reshape(-1, 3), eps * (x - A), rtol=1e-9, atol=1e-15)


def test_grid_rule_and_connectivity():
    grid = VX.choose_grid([0, 0, 0], [0.3, 0.2, 0.1], 50000, 1e-3)
    assert 3 * np.prod(np.array(grid.shape) + 1) <= 50000 and grid.cell_m > 1e-3
    small = VX.choose_grid([0, 0, 0], [0.01, 0.01, 0.01], 50000, 1e-3)
    assert small.cell_m == 1e-3 and small.shape == (10, 10, 10)
    active = np.zeros((6, 6, 6), bool)
    active[:3, :, :] = True
    active[5, 5, 5] = True                            # an island
    kept, dropped = VX.keep_connected(active, None)
    assert dropped == 1 and not kept[5, 5, 5] and kept[0, 0, 0]
    with pytest.raises(Exception):
        VX.choose_grid([0, 0, 0], [0.1, 0.1, 0.1], 10, 1e-3)
