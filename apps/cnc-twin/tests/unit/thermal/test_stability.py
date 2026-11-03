"""Stability of the cut-cell ADI at the cadence's step with 20 % (and 1 %) cut cells."""

import numpy as np
import pytest

from cnctwin.thermal import kernels as K
from cnctwin.thermal import props
from cnctwin.thermal.abi import (B_FMIN, B_SIGMA, B_T_AIR, B_T_RAD, L_U0, L_U1, N_L, new_work_boundary, new_work_grid,
                                 new_work_scratch)

ALU = (2700.0, 167.0, 900.0)      # aluminium: the fastest diffusion of the database


def adiabatic():
    bnd = new_work_boundary(0, (0, 0))
    bnd.scal[B_T_AIR] = 293.15
    bnd.scal[B_T_RAD] = 293.15
    bnd.scal[B_FMIN] = 1e-9
    bnd.scal[B_SIGMA] = 0.0
    return bnd


@pytest.mark.parametrize("fraction", [0.2, 0.01])
@pytest.mark.parametrize("dt", [0.05, 0.5, 50.0])
def test_cut_cells_stay_bounded_and_conservative(fraction, dt):
    rho, k, cp = ALU
    mat = props.mat_table({"value": rho}, {"value": k}, {"value": cp}, {"value": 2.3e-5}, "al")
    shape = (12, 10, 8)
    dx = 2e-3
    grid = new_work_grid(shape, (0.0, 0.0, 0.0), dx)
    rng = np.random.default_rng(11)
    f = np.ones(shape)
    f[rng.random(shape) < 0.3] = fraction             # 30 % of the cells are thin cut cells
    f[:, :, -1] = fraction                           # a floor of slivers
    f[3, :, :] = fraction                            # a thin wall of cut cells
    grid.f[:] = f
    temps = 293.15 + 300.0 * rng.random(shape)
    for idx in np.ndindex(shape):
        grid.h[idx] = K.enthalpy(mat, temps[idx])
        grid.T[idx] = temps[idx]
    grid.contact[:] = np.uint8(0b111111)             # every exposed face adiabatic
    t_min, t_max = temps.min(), temps.max()
    scratch = new_work_scratch(shape)
    out = np.zeros(N_L)
    explicit_limit = rho * cp * fraction * dx * dx / (6.0 * k * 2.0)
    assert dt > 10 * explicit_limit
    u0 = None
    spread = []
    for _ in range(40):
        scratch.src[:] = 0.0
        K.work_step(grid, scratch, mat, adiabatic(), dt, out)
        u0 = out[L_U0] if u0 is None else u0
        assert out[L_U1] == pytest.approx(u0, rel=1e-9)        # the ledger tolerance (internal flows >> U)
        assert np.all(np.isfinite(grid.T))
        assert grid.T.min() >= t_min - 1e-6 * (t_max - t_min)
        assert grid.T.max() <= t_max + 1e-6 * (t_max - t_min)
        spread.append(grid.T.max() - grid.T.min())
    assert spread[-1] <= spread[0] + 1e-9
