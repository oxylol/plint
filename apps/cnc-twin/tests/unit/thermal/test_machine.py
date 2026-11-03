"""Spindle growth curve, ballscrew drift, structure drift, cold and warm starts."""

import math

import numpy as np
import pytest

from cnctwin.thermal import machine as M

import fixtures as F


def spindle(start="cold"):
    return M.spindle_models(F.MACHINE_RAW, 293.15, start, 1e-9, 0.0)["main"]


def test_spindle_growth_follows_first_order_curve_at_max_speed():
    model = spindle()
    omega_max = 12000 * 2 * math.pi / 60
    tau = 25 * 60.0
    growth = []
    for _ in range(60):
        model.step(60.0, np.full(10, omega_max), 0.0, 1e-9, 0.0)
        growth.append(model.growth_abs_m())
    times = 60.0 * np.arange(1, 61)
    expected = 25e-6 * (1.0 - np.exp(-times / tau))
    assert np.allclose(growth, expected, rtol=1e-9, atol=1e-15)
    assert model.growth_rel_m() == pytest.approx(0.3 * growth[-1], rel=1e-12)      # compensation residual
    peak = model.growth_abs_m()
    model.step(tau, np.zeros(1), 0.0, 1e-9, 0.0)                                   # stopped: cools
    assert model.growth_abs_m() == pytest.approx(peak * math.exp(-1.0), rel=1e-9)
    assert model.body.ledger.worst_rel < 1e-12


def test_bearing_heat_scales_with_speed_to_five_thirds():
    model = spindle()
    omega_max = 12000 * 2 * math.pi / 60
    assert model.bearing_heat(np.array([omega_max])) == pytest.approx(450.0)
    assert model.bearing_heat(np.array([0.5 * omega_max])) == pytest.approx(450.0 * 0.5 ** (5.0 / 3.0))


def test_warm_start_and_explicit_start_states():
    warm = spindle("warm")
    cold = spindle("cold")
    assert warm.body.T_K > cold.body.T_K
    power = 450.0 * 0.5 ** (5.0 / 3.0)
    rise = power / warm.body.g_W_K * (1.0 - math.exp(-3600.0 / (25 * 60.0)))
    assert warm.body.T_K - 293.15 == pytest.approx(rise, rel=1e-9)
    assert warm.growth_rel_m() == 0.0                       # growth counts from the start state
    given = spindle({"spindle_K": 303.15, "axes_K": None, "structure_K": None})
    assert given.body.T_K == pytest.approx(303.15)


def test_ballscrew_drift_from_travel_and_position():
    axes = M.axis_models(F.MACHINE_RAW, 293.15, "cold", 1e-9, 0.0)
    x = axes["X"]
    assert x.direction_M == pytest.approx([-1.0, 0.0, 0.0])
    for _ in range(100):
        x.step(10.0, 0.5, 293.15, 1e-9, 0.0)              # 0.05 m/s for 1000 s
    assert x.body.T_K > 293.15
    strain = 11.5e-6 * (x.body.T_K - 293.15) * 0.3       # electronic compensation residual
    assert x.strain() == pytest.approx(strain, rel=1e-9)
    assert x.error_at(x.fixed_end_m) == 0.0
    assert x.error_at(x.fixed_end_m - 0.5) == pytest.approx(-0.5 * strain, rel=1e-12)
    assert x.body.ledger.worst_rel < 1e-12


def test_structure_relaxes_to_ambient_with_drift():
    model = M.structure_model(F.MACHINE_RAW, 295.15, {"spindle_K": None, "axes_K": None, "structure_K": 291.15})
    assert model.drift_m() == 0.0
    model.body.step(4 * 3600.0, 0.0, 295.15, 0.0, 1e-9, 0.0)
    assert model.body.T_K == pytest.approx(295.15 - 4.0 * math.exp(-1.0), rel=1e-9)
    assert model.drift_m() == pytest.approx(3e-6 * (model.body.T_K - 291.15), rel=1e-12)
