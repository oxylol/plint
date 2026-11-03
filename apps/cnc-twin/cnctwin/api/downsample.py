"""Min/max-preserving downsampling of time series for plots (M4).

M4 (Jugel, Jerzak, Hackenbroich and Markl, "M4: A Visualization-Oriented Time Series Data
Aggregation", PVLDB 7(10), 797-808, 2014, doi:10.14778/2732951.2732953) splits the time range into
equal-width buckets and keeps, per bucket, the first, last, minimum and maximum sample (with their
own times). Drawn as a line chart with one bucket per pixel column, the result rasterises exactly
like the full series (the paper's error-free line rendering); with fewer buckets than pixel columns
the line keeps every bucket's true extremes, so a force peak or a temperature spike is never
smoothed away; every point returned is a sample as recorded (its own time and value), so the
only error is in the samples left out between a bucket's extremes, bounded by the bucket's range.

The accumulator is streaming: chunks (one Parquet part at a time) arrive in time order and only
O(buckets) state is kept, so a query over an hour of 1 kHz data reads one part at a time.

NaN samples (channel padding, or a channel undefined out of cut) are left out of the extremes; a
bucket with samples but none finite yields one NaN point at its first sample, so a plot breaks
the line there instead of joining across the gap.
"""

from __future__ import annotations

import numpy as np

_BIG = np.iinfo(np.int64).max
_SMALL = np.iinfo(np.int64).min


class M4:
    """Streaming M4 over integer ticks in [tick0, tick1] with ``buckets`` equal-width buckets."""

    def __init__(self, tick0: int, tick1: int, buckets: int) -> None:
        if buckets < 1:
            raise ValueError("at least one bucket")
        self.tick0 = int(tick0)
        self.span = max(1, int(tick1) - int(tick0) + 1)
        self.n = int(buckets)
        n = self.n
        self.count = np.zeros(n, np.int64)
        self.finite = np.zeros(n, np.int64)
        self.gap_t = np.full(n, _BIG, np.int64)
        self.first_t = np.full(n, _BIG, np.int64)
        self.first_v = np.full(n, np.nan)
        self.last_t = np.full(n, _SMALL, np.int64)
        self.last_v = np.full(n, np.nan)
        self.min_t = np.full(n, _BIG, np.int64)
        self.min_v = np.full(n, np.inf)
        self.max_t = np.full(n, _BIG, np.int64)
        self.max_v = np.full(n, -np.inf)

    def bucket(self, ticks: np.ndarray) -> np.ndarray:
        offset = np.asarray(ticks, np.int64) - self.tick0
        return np.clip((offset * self.n) // self.span, 0, self.n - 1)

    def add(self, ticks: np.ndarray, values: np.ndarray) -> None:
        ticks = np.asarray(ticks, np.int64)
        values = np.asarray(values, np.float64)
        if ticks.size == 0:
            return
        b = self.bucket(ticks)
        self.count += np.bincount(b, minlength=self.n)
        np.minimum.at(self.gap_t, b, ticks)
        finite = np.isfinite(values)
        if not finite.any():
            return
        bf, tf, vf = b[finite], ticks[finite], values[finite]
        self.finite += np.bincount(bf, minlength=self.n)
        order = np.lexsort((tf, bf))                     # by bucket, then time
        bs, ts, vs = bf[order], tf[order], vf[order]
        groups, starts = np.unique(bs, return_index=True)
        ends = np.append(starts[1:], len(bs)) - 1
        better = ts[starts] < self.first_t[groups]
        self.first_t[groups[better]] = ts[starts][better]
        self.first_v[groups[better]] = vs[starts][better]
        better = ts[ends] > self.last_t[groups]
        self.last_t[groups[better]] = ts[ends][better]
        self.last_v[groups[better]] = vs[ends][better]
        by_value = np.lexsort((tf, vf, bf))              # by bucket, then value, then time
        bv = bf[by_value]
        lo = np.searchsorted(bv, groups, "left")
        hi = np.searchsorted(bv, groups, "right") - 1
        mins, maxs = by_value[lo], by_value[hi]
        better = (vf[mins] < self.min_v[groups]) | ((vf[mins] == self.min_v[groups]) & (tf[mins] < self.min_t[groups]))
        self.min_v[groups[better]] = vf[mins][better]
        self.min_t[groups[better]] = tf[mins][better]
        better = (vf[maxs] > self.max_v[groups]) | ((vf[maxs] == self.max_v[groups]) & (tf[maxs] < self.max_t[groups]))
        self.max_v[groups[better]] = vf[maxs][better]
        self.max_t[groups[better]] = tf[maxs][better]

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        """(ticks, values) in time order: up to 4 points per bucket, NaN markers for gaps."""
        have = np.flatnonzero(self.finite > 0)
        gaps = np.flatnonzero((self.count > 0) & (self.finite == 0))
        t = np.stack([self.first_t[have], self.min_t[have], self.max_t[have], self.last_t[have]], axis=1)
        v = np.stack([self.first_v[have], self.min_v[have], self.max_v[have], self.last_v[have]], axis=1)
        order = np.argsort(t, axis=1, kind="stable")
        t = np.take_along_axis(t, order, axis=1)
        v = np.take_along_axis(v, order, axis=1)
        keep = np.ones_like(t, bool)
        keep[:, 1:] = t[:, 1:] != t[:, :-1]
        ticks = np.concatenate([t[keep], self.gap_t[gaps]])
        values = np.concatenate([v[keep], np.full(len(gaps), np.nan)])
        order = np.argsort(ticks, kind="stable")
        return ticks[order], values[order]


def m4(ticks: np.ndarray, values: np.ndarray, buckets: int, tick0: int | None = None,
       tick1: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """M4 of one in-memory series (see :class:`M4`)."""
    ticks = np.asarray(ticks, np.int64)
    if ticks.size == 0:
        return ticks, np.asarray(values, np.float64)
    acc = M4(int(ticks.min()) if tick0 is None else tick0, int(ticks.max()) if tick1 is None else tick1, buckets)
    acc.add(ticks, values)
    return acc.result()
