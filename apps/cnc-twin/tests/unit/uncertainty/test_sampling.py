"""Sampling plans: Latin hypercube strata, open unit draws, per-factor seeded streams, distributions."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from cnctwin.core.rng import stream
from cnctwin.uncertainty.sampling import TruncatedNormal, Uniform, latin_hypercube, monte_carlo, plan, unit_column


@pytest.mark.parametrize("n", [1, 2, 7, 64, 1000])
def test_lhs_one_draw_per_stratum(n):
    u = latin_hypercube(n, stream(5, "test.lhs"))
    assert u.shape == (n,) and np.all((u > 0.0) & (u < 1.0))
    assert sorted(np.floor(u * n).astype(int).tolist()) == list(range(n))


def test_lhs_marginal_is_nearly_exact():
    n = 200
    columns = np.column_stack([latin_hypercube(n, stream(seed, "test.lhs")) for seed in range(20)])
    # the mean of a stratified uniform sample deviates from 1/2 by at most 1/(2n), far less than random sampling
    assert np.max(np.abs(columns.mean(axis=0) - 0.5)) < 1.0 / (2 * n)
    random = np.column_stack([monte_carlo(n, stream(seed, "test.mc")) for seed in range(20)])
    assert np.std(columns.mean(axis=0)) < 0.2 * np.std(random.mean(axis=0))


def test_monte_carlo_is_uniform_and_open():
    u = monte_carlo(4000, stream(11, "test.mc"))
    assert np.all((u > 0.0) & (u < 1.0))
    assert sps.kstest(u, "uniform").pvalue > 1e-3


def test_plans_are_seeded_per_factor():
    a = unit_column("lhs", 50, 3, "stock_size[0]")
    assert np.array_equal(a, unit_column("lhs", 50, 3, "stock_size[0]"))
    assert not np.array_equal(a, unit_column("lhs", 50, 4, "stock_size[0]"))
    assert not np.array_equal(a, unit_column("lhs", 50, 3, "stock_size[1]"))
    names = ["kc_scale", "runout[T1]"]
    dists = [TruncatedNormal(1.0, 0.04, 0.88, 1.12), Uniform(2e-6, 8e-6)]
    units, values = plan(names, dists, 30, "lhs", 9)
    more_units, _ = plan(names + ["stock_size[2]"], dists + [Uniform(0.0149, 0.0151)], 30, "lhs", 9)
    assert np.array_equal(units, more_units[:, :2])       # adding a factor leaves the others' draws alone
    assert np.all((values[:, 1] >= 2e-6) & (values[:, 1] <= 8e-6))
    with pytest.raises(ValueError):
        plan(names, dists, 0, "lhs", 1)
    with pytest.raises(ValueError):
        unit_column("sobol", 4, 1, "x")


def test_distributions():
    uniform = Uniform(-5e-6, 5e-6)
    assert uniform.ppf(np.array([0.0, 0.5, 1.0])).tolist() == pytest.approx([-5e-6, 0.0, 5e-6])
    assert uniform.std == pytest.approx(1e-5 / np.sqrt(12.0))
    normal = TruncatedNormal(1.0, 0.04, 0.88, 1.12)
    u = latin_hypercube(20000, stream(1, "test.tn"))
    x = normal.ppf(u)
    assert np.all((x >= 0.88) & (x <= 1.12)) and np.all(np.diff(normal.ppf(np.sort(u))) >= 0.0)
    expected = sps.truncnorm(-3.0, 3.0, loc=1.0, scale=0.04)
    assert normal.mean == pytest.approx(expected.mean()) and normal.std == pytest.approx(expected.std())
    assert x.mean() == pytest.approx(1.0, abs=1e-4) and x.std() == pytest.approx(expected.std(), rel=0.01)
    assert normal.std == pytest.approx(0.04 * 0.9866, rel=1e-3)            # truncation at 3 sigma: -1.3 %
    clipped = TruncatedNormal(1.0, 0.5, 0.1, 2.5)
    assert clipped.ppf(np.array([1e-12]))[0] >= 0.1
    assert TruncatedNormal(1.0, 0.0, 1.0, 1.0).ppf(np.array([0.3, 0.7])).tolist() == [1.0, 1.0]
