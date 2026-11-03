"""M4 downsampling: per bucket the first, last, minimum and maximum sample, streaming = one shot."""

from __future__ import annotations

import numpy as np
import pytest

from cnctwin.api.downsample import M4, m4


def reference(ticks, values, buckets, tick0, tick1):
    """Plain-Python M4: bucket by integer arithmetic, keep first/min/max/last (NaN left out)."""
    span = tick1 - tick0 + 1
    groups = {}
    for t, v in zip(ticks.tolist(), values.tolist()):
        groups.setdefault(min(buckets - 1, max(0, (t - tick0) * buckets // span)), []).append((t, v))
    out = []
    for b in sorted(groups):
        finite = [(t, v) for t, v in groups[b] if np.isfinite(v)]
        if not finite:
            out.append((groups[b][0][0], float("nan")))
            continue
        lo = min(finite, key=lambda p: (p[1], p[0]))
        hi = max(finite, key=lambda p: (p[1], -p[0]))
        chosen = {finite[0], lo, hi, finite[-1]}
        out.extend(sorted(chosen))
    return np.array([t for t, _ in out]), np.array([v for _, v in out])


def signal(n=50000, seed=0):
    rng = np.random.default_rng(seed)
    ticks = np.arange(n, dtype=np.int64) * 8
    values = np.sin(ticks * 1e-3) + 0.1 * rng.normal(size=n)
    values[int(0.25 * n)] = 25.0                             # a spike
    values[int(0.8 * n)] = -30.0
    return ticks, values


def test_matches_plain_reference():
    ticks, values = signal(5000)
    values[100:180] = np.nan
    got_t, got_v = m4(ticks, values, 64)
    ref_t, ref_v = reference(ticks, values, 64, int(ticks[0]), int(ticks[-1]))
    np.testing.assert_array_equal(got_t, ref_t)
    np.testing.assert_array_equal(got_v, ref_v)


def test_extremes_and_their_times_are_kept():
    ticks, values = signal()
    t, v = m4(ticks, values, 250)
    assert len(t) <= 1000
    assert v.max() == 25.0 and t[np.argmax(v)] == ticks[12500]
    assert v.min() == -30.0 and t[np.argmin(v)] == ticks[40000]
    assert np.all(np.diff(t) > 0)
    for tick, value in zip(t, v):                            # every point is a real sample
        assert values[tick // 8] == value


@pytest.mark.parametrize("chunks", [2, 7, 33])
def test_streaming_equals_one_shot(chunks):
    ticks, values = signal(20000, seed=3)
    one_t, one_v = m4(ticks, values, 100, int(ticks[0]), int(ticks[-1]))
    acc = M4(int(ticks[0]), int(ticks[-1]), 100)
    for part_t, part_v in zip(np.array_split(ticks, chunks), np.array_split(values, chunks)):
        acc.add(part_t, part_v)
    t, v = acc.result()
    np.testing.assert_array_equal(t, one_t)
    np.testing.assert_array_equal(v, one_v)


def test_gaps_become_one_nan_point_per_bucket():
    ticks = np.arange(1000, dtype=np.int64)
    values = np.ones(1000)
    values[400:600] = np.nan
    t, v = m4(ticks, values, 10)
    gap = np.isnan(v)
    assert gap.sum() == 2 and set(t[gap]) == {400, 500}
    assert np.all(v[~gap] == 1.0)


def test_empty_buckets_produce_nothing_and_single_samples_once():
    ticks = np.array([0, 1, 1000, 1001], np.int64)
    t, v = m4(ticks, np.array([1.0, 2.0, 3.0, 4.0]), 50)
    np.testing.assert_array_equal(t, ticks)
    one_t, one_v = m4(np.array([5], np.int64), np.array([7.0]), 4)
    assert one_t.tolist() == [5] and one_v.tolist() == [7.0]
    with pytest.raises(ValueError):
        M4(0, 10, 0)


def test_out_of_range_samples_fall_into_the_edge_buckets():
    acc = M4(100, 199, 4)
    acc.add(np.array([50, 150, 250], np.int64), np.array([1.0, 2.0, 3.0]))
    t, v = acc.result()
    assert t.tolist() == [50, 150, 250] and v.tolist() == [1.0, 2.0, 3.0]
