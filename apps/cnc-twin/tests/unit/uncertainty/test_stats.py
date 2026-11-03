"""Aggregation statistics: quantiles and mean interval, Wilson intervals, capability indices (two- and
one-sided, with intervals), rank and partial rank correlations."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as sps

from cnctwin.core.rng import stream
from cnctwin.uncertainty import stats

LEVELS = [0.00135, 0.025, 0.5, 0.975, 0.99865]


def test_describe():
    data = np.arange(1.0, 11.0)
    record = stats.describe(list(data) + [None, float("nan")], LEVELS, 0.95)
    assert record["n"] == 10 and record["mean"] == 5.5 and record["std"] == pytest.approx(np.std(data, ddof=1))
    assert record["quantiles"]["0.5"] == 5.5 and record["quantiles"]["0.025"] == pytest.approx(1.225)
    half = sps.t.ppf(0.975, 9) * np.std(data, ddof=1) / math.sqrt(10)
    assert record["mean_ci"] == pytest.approx([5.5 - half, 5.5 + half])
    assert stats.describe([], LEVELS, 0.95)["mean"] is None


def test_wilson_interval():
    record = stats.wilson(5, 10, 0.95)
    assert record["p"] == 0.5 and record["low"] == pytest.approx(0.2366, abs=1e-4) and record["high"] == pytest.approx(
        0.7634, abs=1e-4)
    zero = stats.wilson(0, 10, 0.95)
    assert zero["low"] == 0.0 and zero["high"] == pytest.approx(0.2775, abs=1e-4)
    assert stats.wilson(10, 10, 0.95)["high"] == 1.0 and stats.wilson(0, 0, 0.95)["p"] is None


def test_capability_two_sided():
    rng = stream(3, "test.capability")
    data = rng.normal(10.0, 0.1, size=400)
    record = stats.capability(data, 9.7, 10.4, one_sided=False, confidence=0.95, target=1.33, min_samples=10,
                              levels=LEVELS)
    s, m = float(np.std(data, ddof=1)), float(np.mean(data))
    assert record["Cp"] == pytest.approx(0.7 / (6 * s)) and record["Cpk"] == pytest.approx(min(10.4 - m, m - 9.7) / (3 * s))
    assert record["Cpk"] == record["Cpl"]                                   # nearer the lower limit
    low, high = record["Cpk_ci"]
    half = 1.959964 * math.sqrt(1 / (9 * 400) + record["Cpk"] ** 2 / (2 * 399))
    assert low == pytest.approx(record["Cpk"] - half, rel=1e-5) and high == pytest.approx(record["Cpk"] + half, rel=1e-5)
    assert low < 1.0 < high                                                  # true Cpk is 0.3 / 0.3 = 1
    assert record["expected_out_frac"] == pytest.approx(sps.norm.cdf((9.7 - m) / s) + sps.norm.sf((10.4 - m) / s))
    assert record["capable"] is False and record["Cp_ci"][0] < record["Cp"] < record["Cp_ci"][1]


def test_capability_one_sided_and_degenerate():
    data = np.linspace(4e-6, 12e-6, 50)
    record = stats.capability(data, 0.0, 2e-5, one_sided=True, confidence=0.95, target=1.33, min_samples=10,
                              levels=LEVELS)
    s, m = float(np.std(data, ddof=1)), float(np.mean(data))
    assert record["Cp"] is None and record["lower"] is None
    assert record["Cpk"] == record["Cpu"] == pytest.approx((2e-5 - m) / (3 * s))
    constant = stats.capability([1.0] * 20, 0.5, 1.5, one_sided=False, confidence=0.95, target=1.33, min_samples=10,
                                levels=LEVELS)
    assert constant["Cpk"] is None and constant["expected_out_frac"] == 0.0
    few = stats.capability([1.0, 1.1, 1.2], 0.5, 1.5, one_sided=False, confidence=0.95, target=1.33, min_samples=10,
                           levels=LEVELS)
    assert few["reported"] is False


def test_rank_correlations():
    rng = stream(8, "test.rank")
    x1 = rng.uniform(size=200)
    x2 = 0.8 * x1 + 0.2 * rng.uniform(size=200)                          # correlated with x1, no effect of its own
    noise = rng.uniform(size=200)
    y = np.exp(3.0 * x1) + 0.01 * noise                                    # monotone in x1 only
    assert stats.spearman(x1, y)["rho"] == pytest.approx(sps.spearmanr(x1, y).statistic)
    assert stats.spearman(x1, y)["rho"] > 0.99 and stats.spearman(x1, y)["p"] < 1e-10
    assert stats.spearman(x2, y)["rho"] > 0.8                              # spurious through x1
    partial = stats.prcc(np.column_stack([x1, x2, noise]), y, min_dof=3)
    assert partial[0]["prcc"] > 0.95 and abs(partial[1]["prcc"]) < 0.3 and abs(partial[2]["prcc"]) < 0.3
    assert partial[0]["dof"] == 200 - 2 - 2
    assert stats.spearman(np.ones(10), np.arange(10))["rho"] is None
    assert all(p["prcc"] is None for p in stats.prcc(np.column_stack([x1[:5], x2[:5], noise[:5]]), y[:5], min_dof=3))
