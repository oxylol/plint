"""The least-squares engine: estimates, intervals from the Jacobian, identifiability, bounds, reweighting."""

import numpy as np
import pytest
from scipy.stats import t as student_t

from cnctwin.mdata.fit import FitError, ObservationSet, Param, fit


def _line_problem(noise=0.05, n=40, seed=0):
    x = np.linspace(0.0, 10.0, n)
    y = 1.5 + 0.8 * x + np.random.default_rng(seed).normal(0.0, noise, n)
    params = [Param("a", 0.0, -10.0, 10.0), Param("b", 0.0, -10.0, 10.0)]
    return x, y, params, (lambda v: {"y": v["a"] + v["b"] * x})


def test_linear_model_matches_ordinary_least_squares():
    x, y, params, simulate = _line_problem()
    result = fit(params, simulate, [ObservationSet("y", y)])
    A = np.column_stack([np.ones_like(x), x])
    coef, residual, *_ = np.linalg.lstsq(A, y, rcond=None)
    s2 = residual[0] / (len(x) - 2)
    cov = s2 * np.linalg.inv(A.T @ A)
    t = student_t.ppf(0.975, len(x) - 2)
    by_name = {p["name"]: p for p in result.params}
    for k, name in enumerate(("a", "b")):
        assert by_name[name]["value"] == pytest.approx(coef[k], abs=1e-8)
        assert by_name[name]["std_error"] == pytest.approx(np.sqrt(cov[k, k]), rel=1e-4)
        lo, hi = by_name[name]["ci"]
        assert lo == pytest.approx(coef[k] - t * np.sqrt(cov[k, k]), rel=1e-4)
        assert hi == pytest.approx(coef[k] + t * np.sqrt(cov[k, k]), rel=1e-4)
        assert not by_name[name]["ci_check"]["approximate"]           # linear: the check ratio is 1
    assert result.dof == len(x) - 2 and result.metrics["y"]["after"]["rmse"] < result.metrics["y"]["before"]["rmse"]


def test_confounded_and_non_identifiable_parameters():
    x = np.linspace(0, 1, 30)
    y = 3.0 * x + np.random.default_rng(1).normal(0, 0.01, 30)
    params = [Param("k1", 1.0, 0.0, 10.0), Param("k2", 1.0, 0.0, 10.0)]
    result = fit(params, lambda v: {"y": v["k1"] * v["k2"] * x}, [ObservationSet("y", y)])
    by_name = {p["name"]: p for p in result.params}
    assert by_name["k1"]["value"] * by_name["k2"]["value"] == pytest.approx(3.0, rel=1e-3)
    assert not by_name["k1"]["identifiable"] and not by_name["k2"]["identifiable"]
    assert by_name["k1"]["ci"] is None
    near = [Param("k1", 1.0, 0.0, 10.0), Param("k2", 1.0, 0.0, 10.0)]
    result = fit(near, lambda v: {"y": v["k1"] * x + v["k2"] * (x + 1e-3 * x ** 2)}, [ObservationSet("y", y)])
    assert result.confounded and abs(result.confounded[0]["correlation"]) > 0.95


def test_bounds_log_parameters_and_fixed_values():
    t = np.linspace(300.0, 900.0, 25)
    A_true, B_true = 3e-9, 4000.0
    y = np.log(A_true * np.exp(-B_true / t)) + np.random.default_rng(2).normal(0, 0.01, 25)
    params = [Param("A", 1e-6, 1e-20, 1e-3, log=True), Param("B", 1000.0, 10.0, 1e5)]
    result = fit(params, lambda v: {"ln_rate": np.log(v["A"]) - v["B"] / t}, [ObservationSet("ln_rate", y)])
    by_name = {p["name"]: p for p in result.params}
    assert by_name["A"]["value"] == pytest.approx(A_true, rel=0.1)
    assert by_name["B"]["value"] == pytest.approx(B_true, rel=0.02)
    lo, hi = by_name["A"]["ci"]
    assert lo < A_true < hi and hi / by_name["A"]["value"] == pytest.approx(by_name["A"]["value"] / lo, rel=1e-6)
    capped = [Param("a", 0.0, -10.0, 1.0), Param("b", 0.5, 0.0, 1.0, free=False)]
    x, y2, _, _ = _line_problem()
    result = fit(capped, lambda v: {"y": v["a"] + v["b"] * x}, [ObservationSet("y", y2)])
    by_name = {p["name"]: p for p in result.params}
    assert by_name["a"]["at_bound"] == "upper" and by_name["a"]["value"] == pytest.approx(1.0)
    assert by_name["b"]["value"] == 0.5 and by_name["b"]["free"] is False


def test_reweighting_balances_sets_with_different_scatter():
    x = np.linspace(0, 1, 50)
    rng = np.random.default_rng(4)
    precise = 2.0 * x + rng.normal(0, 1e-3, 50)
    noisy = 2.0 * x * 1000.0 + rng.normal(0, 50.0, 50)               # same parameter, another unit and noise
    params = [Param("k", 1.0, 0.0, 10.0)]
    result = fit(params, lambda v: {"p": v["k"] * x, "q": v["k"] * x * 1000.0},
                 [ObservationSet("p", precise), ObservationSet("q", noisy)])
    assert result.params[0]["value"] == pytest.approx(2.0, abs=5e-4)
    assert result.sigma["p"] == pytest.approx(1e-3, rel=0.3) and result.sigma["q"] == pytest.approx(50.0, rel=0.3)


def test_setup_errors():
    x, y, params, simulate = _line_problem(n=2)
    with pytest.raises(FitError, match="cannot determine"):
        fit(params, simulate, [ObservationSet("y", y)])
    x, y, params, simulate = _line_problem()
    with pytest.raises(FitError, match="returned no"):
        fit(params, lambda v: {"z": x}, [ObservationSet("y", y)])
    with pytest.raises(FitError, match="not finite"):
        fit(params, lambda v: {"y": np.full_like(x, np.nan)}, [ObservationSet("y", y)])
    with pytest.raises(FitError, match="outside its bounds"):
        fit([Param("a", 20.0, 0.0, 10.0)], simulate, [ObservationSet("y", y)])
