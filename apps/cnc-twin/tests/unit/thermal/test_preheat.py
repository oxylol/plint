"""The preheat cutting's coefficients read is a physical quantity, not a grid one (docs/models/thermal.md 3.3;
docs/convergence.md 6): the temperature of the material entering the cut at the leading edge of the contact's moving
heat source converges as the workpiece cell shrinks, where the cell one feed half length plus one cell ahead doubled
with every halving of the cell (it sits inside the source's near field)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cnctwin.thermal import props
from cnctwin.thermal.protocols import BoxStock
from cnctwin.thermal.schema import ThermalSettings
from cnctwin.thermal.workpiece import ContactSource, WorkEnvironment, WorkThermal

SIZE = (0.06, 0.03, 0.02)
SPEED = 0.02            # feed (m/s)
POWER = 15.0            # work-bound heat of the contact (W)
HALF = 1.0e-3           # feed half length of the engagement (m)
WIDTH = 2.0e-3          # across the feed (ap)
MACRO = 0.01
FV_EVERY = 5


def aluminium():
    return props.mat_table({"value": 2700.0}, {"value": 167.0}, {"value": 896.0}, {"value": 23.6e-6}, "al")


def preheat_after(cell_m: float, t_end: float = 1.5):
    work = WorkThermal(0, BoxStock([0.0, 0.0, 0.0], list(SIZE)), aluminium(), ThermalSettings(cell_m=cell_m),
                       293.15, "aluminium", 0.0)
    env = WorkEnvironment(T_air_K=293.15, T_table_K=293.15)
    x0, y, z = 0.005, 0.5 * SIZE[1], SIZE[2]
    n = int(round(t_end / MACRO))
    since_fv = 0
    old = None
    for i in range(n):
        t = (i + 1) * MACRO
        x = x0 + SPEED * t
        # the step's work heat on the generated surface under the band (cutting's work_heat_S rows)
        xs = np.linspace(x - HALF, x + HALF, 5)
        rows = np.zeros((xs.size, 8))
        rows[:, 0] = xs
        rows[:, 1] = y
        rows[:, 2] = z
        rows[:, 5] = 1.0
        rows[:, 6] = POWER * MACRO / xs.size
        work.add_rows(rows)
        since_fv += 1
        fv_now = since_fv >= FV_EVERY
        if fv_now:
            work.fv_step(since_fv * MACRO, env, t)
        window = since_fv * MACRO
        src = ContactSource(point_S_m=np.array([x, y, z]), feed_dir_S=np.array([1.0, 0.0, 0.0]), power_W=POWER,
                            feed_speed_m_s=SPEED, feed_half_length_m=HALF, width_m=WIDTH, window_s=window,
                            flash_flux_W_m2=0.0, flash_half_length_m=0.0, flash_speed_m_s=0.0, flash_duration_s=1e-4,
                            engaged_s=t, pending_s=0.0 if fv_now else window, ahead_m=0.01)
        surface = work.contact_surface(src, fv_now)
        if fv_now:
            since_fv = 0
        if i == n - 1:
            # the former definition: the cell one feed half length plus one cell ahead
            ahead = work.cell_of(np.array([x + HALF + cell_m, y, z]))
            old = work.cell_temp(ahead)
            return surface.preheat_K, old
    raise AssertionError("no step")


def test_preheat_converges_with_the_cell():
    values = {cell: preheat_after(cell) for cell in (2e-3, 1e-3, 0.5e-3)}
    rise = {cell: v[0] - 293.15 for cell, v in values.items()}
    old = {cell: v[1] - 293.15 for cell, v in values.items()}
    assert rise[0.5e-3] > 5.0
    # within the convergence study's 1.5 % (a tenth of the tool-temperature budget) from 1 mm, 5 % at 2 mm
    assert rise[1e-3] == pytest.approx(rise[0.5e-3], rel=0.015)
    assert rise[2e-3] == pytest.approx(rise[0.5e-3], rel=0.05)
    # the cell ahead was a grid quantity: it grows by a quarter or more with each halving of the cell (1.5, 3.6, 4.6,
    # 6.2 K at 4, 2, 1, 0.5 mm, against 14.1 K of the converged preheat at every cell)
    assert old[0.5e-3] > 1.25 * old[1e-3] and old[1e-3] > 1.25 * old[2e-3]
