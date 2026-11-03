"""Measured kernel cost (model doc section 7) and the physics.dynamics settings."""

from __future__ import annotations

import time

import numpy as np
import pytest

from cnctwin.core.errors import InputError
from cnctwin.dynamics.abi import KIND_MACHINE, core_accum
from cnctwin.dynamics.drive import cost_contact, cost_free
from cnctwin.dynamics.schema import PHYSICS_SCHEMA, settings
from tests.unit.dynamics.dynfix import cantilever_mode, system, tool_mode, work_grid


def _system(n_tool, nst):
    z = np.linspace(0.0, 0.02, nst)
    modes = [cantilever_mode(1500.0 + 300 * k, 0.02, z + 0.05, 0.07, 0.1, axis=k % 2, key=("b", k)) for k in range(n_tool)]
    modes.append(tool_mode(50.0, 0.05, 400.0, (1, 0, 0), nst=nst, kind=KIND_MACHINE, base=[-20, 0, 0, 0, 0, 0, 0]))
    wm, grid = work_grid(2, (900.0, 1300.0))
    sys, _ = system(modes + wm, nst=nst, l0_ticks=8, residual=np.eye(3 * nst) * 1e-9, grid=grid,
                    point_S=[0.01, 0.01, 0.01], unbalance=(np.array([1e-6, 0, 0]), np.zeros(3)),
                    K_c=np.diag([-1e6, -1e6, 0.0]))
    return sys


@pytest.mark.parametrize("n_tool,nst,budget_us", [(2, 1, 20.0), (10, 20, 60.0)])
def test_kernel_cost(n_tool, nst, budget_us):
    """Per L0 step (modal_begin + modal_correct, the force kernel excluded) and per free tick, from a compiled loop.
    The budgets are generous ceilings (a loaded machine); the measured values are in the model doc."""
    sys = _system(n_tool, nst)
    st = np.random.default_rng(1).normal(size=(nst, 6))
    acc = core_accum()
    cost_contact(sys, 10, st, np.ones(3), acc)
    n = 20000
    t0 = time.perf_counter()
    cost_contact(sys, n, st, np.ones(3), acc)
    per_step = (time.perf_counter() - t0) / n * 1e6
    cost_free(sys, 10, 1000)
    m = 2000
    t0 = time.perf_counter()
    cost_free(sys, m, 1000)
    per_tick = (time.perf_counter() - t0) / m * 1e6
    print(f"modes {n_tool + 3}, stations {nst}: {per_step:.2f} us per L0 step, {per_tick:.2f} us per free tick")
    assert per_step < budget_us and per_tick < 3.0 * budget_us


def test_settings():
    s = settings(None)
    assert s.sd_m == 40 and s.relax_estimate == 1
    s = settings({"sd_m": 60, "chatter_window_revs": 30})
    assert s.sd_m == 60 and s.chatter_window_revs == 30
    with pytest.raises(InputError):
        settings({"sd_m": 2})
    assert "sld_n_freq" in PHYSICS_SCHEMA.fields
