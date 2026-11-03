import math

import numpy as np

from cnctwin import units
from cnctwin.core import rng


def test_to_si_converts_suffixed_keys():
    assert units.to_si("travel_mm", [-762, 0]) == ("travel_m", [-0.762, 0.0])
    key, value = units.to_si("max_rpm", 12000)
    assert key == "max_rad_s" and math.isclose(value, 12000 * 2 * math.pi / 60)
    assert units.to_si("ambient_degC", 20) == ("ambient_K", 293.15)
    assert units.to_si("hysteresis_dK", 2) == ("hysteresis_dK", 2.0)
    assert units.to_si("kc11_N_mm2", 700) == ("kc11_Pa", 7e8)
    assert units.to_si("hardness_HRC", 62) == ("hardness_HRC", 62)
    assert units.to_si("flutes", 4) == ("flutes", 4)
    assert units.to_si("density_per_mm", 2) == ("density_per_m", 2000.0)


def test_hash_uniform_is_order_independent_and_uniform():
    key = rng.stream_key("wear.chipping")
    values = np.array([rng.hash_uniform(42, key, i, 7, 3) for i in range(20000)])
    again = np.array([rng.hash_uniform(42, key, i, 7, 3) for i in reversed(range(20000))])[::-1]
    assert np.array_equal(values, again)
    assert 0.0 <= values.min() and values.max() < 1.0
    assert abs(values.mean() - 0.5) < 0.01


def test_streams_separate_counter_draws():
    a = rng.hash_uniform(42, rng.stream_key("wear.chipping"), 1, 2, 3)
    b = rng.hash_uniform(42, rng.stream_key("chips.scatter"), 1, 2, 3)
    assert a != b


def test_init_streams_are_independent_and_repeatable():
    a1 = rng.stream(7, "uncertainty.plan").random(5)
    a2 = rng.stream(7, "uncertainty.plan").random(5)
    b = rng.stream(7, "chips.init").random(5)
    assert np.array_equal(a1, a2)
    assert not np.array_equal(a1, b)
