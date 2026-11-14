"""What the learned models read about a stock on an evening: 56 figures from its own prices and volume,
each turned into the stock's place among that day's universe (-1 lowest, +1 highest), plus a short
description of the market that day.

Every figure on row t is computed from rows up to t. The same code runs on ten years for training and on
the last LOOKBACK days each evening; tests/test_features.py checks that the two agree on the last row,
which is what makes the fitted model and the live model the same thing."""
from __future__ import annotations

from datetime import date
from typing import Callable, Iterator

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .cal import next_trading_day, prev_trading_day
from .panel import Panel
from .util import quiet

LOOKBACK = 320          # rows a figure may reach back (the longest: a 21-day window a year ago), with room
_REG: list[tuple[str, Callable]] = []


def _feat(name: str):
    def deco(fn):
        _REG.append((name, fn))
        return fn
    return deco


# ── rolling helpers over [T, N] tables (pandas does the windows; NaN is "no bar that day") ─────────
def _roll(a: np.ndarray, w: int, how: str, minp: int | None = None) -> np.ndarray:
    minp = minp or max(2, int(np.ceil(w * 0.7)))
    return getattr(pd.DataFrame(a).rolling(w, min_periods=min(minp, w)), how)().to_numpy(dtype=np.float32)


def rsum(a, w, minp=None):
    return a if w == 1 else _roll(a, w, "sum", minp)


def rmean(a, w, minp=None):
    return _roll(a, w, "mean", minp)


def rstd(a, w, minp=None):
    return _roll(a, w, "std", minp)


def rmax(a, w, minp=None):
    return _roll(a, w, "max", minp)


def rmin(a, w, minp=None):
    return _roll(a, w, "min", minp)


def lag(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(a, np.nan)
    if k < len(a):
        out[k:] = a[:-k]
    return out


class Base:
    """The few tables several figures share."""

    def __init__(self, p: Panel, index_symbol: str):
        self.p = p
        self.valid = np.isfinite(p.c)
        self.r = p.ret
        with quiet():
            self.lr = np.log1p(np.clip(p.ret, -0.95, None))
            self.lov = np.log1p(np.clip(p.ovn, -0.95, None))
            self.lit = np.log1p(np.clip(p.intr, -0.95, None))
            # a total-return price line: only ratios of it are ever used, so its starting level is free
            line = np.exp(np.cumsum(np.nan_to_num(self.lr), axis=0, dtype=np.float64))
            line[~np.maximum.accumulate(self.valid, axis=0)] = np.nan
            self.P = line.astype(np.float32)
            member = p.member if p.member is not None else self.valid
            self.mkt = np.nanmean(np.where(member, self.r, np.nan), axis=1)          # the universe's day, equal weight
        self.idx = self.r[:, p.ix[index_symbol]] if index_symbol in p.ix else self.mkt
        self._beta: dict[int, tuple] = {}

    def beta(self, w: int):
        """Slope, leftover volatility and correlation of each stock's day against the index over w days."""
        if w not in self._beta:
            x = np.where(np.isfinite(self.r), self.idx[:, None], np.nan).astype(np.float32)   # the index on the stock's own days
            mx, my = rmean(x, w), rmean(self.r, w)
            with quiet():
                cov = rmean(x * self.r, w) - mx * my
                vx = rmean(x * x, w) - mx * mx
                vy = rmean(self.r * self.r, w) - my * my
                b = cov / np.where(vx > 1e-10, vx, np.nan)
                resid = np.sqrt(np.clip(vy - b * b * vx, 0, None))
                corr = cov / np.sqrt(np.where((vx > 1e-10) & (vy > 1e-10), vx * vy, np.nan))
            self._beta[w] = (b.astype(np.float32), resid.astype(np.float32), corr.astype(np.float32))
        return self._beta[w]


# ── the figures ──────────────────────────────────────────────────────────────────────────────────
for _k in (1, 2, 3, 5, 10, 21, 42, 63, 126, 252):                                    # how far it has moved
    _feat(f"mom_{_k}")(lambda b, k=_k: rsum(b.lr, k))
_feat("mom_12_1")(lambda b: lag(rsum(b.lr, 231), 21))                                 # the year, without the last month
_feat("mom_6_1")(lambda b: lag(rsum(b.lr, 105), 21))
for _k in (5, 21, 63):                                                                # overnight and in-session, apart
    _feat(f"ovn_{_k}")(lambda b, k=_k: rsum(b.lov, k))
    _feat(f"intr_{_k}")(lambda b, k=_k: rsum(b.lit, k))
for _k in (5, 21, 63, 252):                                                           # how wildly
    _feat(f"vol_{_k}")(lambda b, k=_k: rstd(b.r, k))


@_feat("vol_ratio")
def _vol_ratio(b):
    with quiet():
        return rstd(b.r, 5) / rstd(b.r, 63)


@_feat("dvol_63")
def _dvol(b):
    return np.sqrt(rmean(np.minimum(b.r, 0) ** 2, 63))


_feat("skew_63")(lambda b: _roll(b.r, 63, "skew"))
_feat("max_21")(lambda b: rmax(b.r, 21))
_feat("min_21")(lambda b: rmin(b.r, 21))


def _range(b):
    with quiet():
        return (b.p.h - b.p.l) / b.p.c


_feat("range_21")(lambda b: rmean(_range(b), 21))


@_feat("range_1")
def _range_1(b):
    with quiet():
        return _range(b) / rmean(_range(b), 21)


_feat("beta_63")(lambda b: b.beta(63)[0])
_feat("beta_252")(lambda b: b.beta(252)[0])
_feat("ivol_63")(lambda b: b.beta(63)[1])
_feat("corr_63")(lambda b: b.beta(63)[2])


@_feat("ldv_63")                                                                      # how much money trades in it
def _ldv(b):
    with quiet():
        return np.log(_roll(b.p.dv, 63, "median"))


@_feat("surge_1")
def _surge_1(b):
    with quiet():
        return b.p.v / rmean(b.p.v, 21)


@_feat("surge_5")
def _surge_5(b):
    with quiet():
        return rmean(b.p.v, 5) / rmean(b.p.v, 63)


@_feat("amihud_21")
def _amihud(b):
    with quiet():
        return np.log(rmean(np.abs(b.r) / np.where(b.p.dv > 0, b.p.dv, np.nan), 21) + 1e-15)


@_feat("tsize")
def _tsize(b):
    with quiet():
        return np.log(b.p.dv / np.where(b.p.n > 0, b.p.n, np.nan))


@_feat("nsurge")
def _nsurge(b):
    with quiet():
        return b.p.n / rmean(b.p.n, 21)


@_feat("vwap_dev")
def _vwap_dev(b):
    with quiet():
        return b.p.c / np.where(b.p.vw > 0, b.p.vw, np.nan) - 1.0


def _clv(b):
    with quiet():
        span = b.p.h - b.p.l
        return ((b.p.c - b.p.l) - (b.p.h - b.p.c)) / np.where(span > 0, span, np.nan)


_feat("clv_1")(_clv)                                                                  # where in the day's range it closed
_feat("clv_5")(lambda b: rmean(_clv(b), 5))
for _k, _how in ((252, "hi"), (252, "lo"), (21, "hi"), (21, "lo")):                   # distance to its highs and lows
    _feat(f"{_how}_{_k}")(lambda b, k=_k, how=_how: b.P / (rmax if how == "hi" else rmin)(b.P, k) - 1.0)
for _k in (5, 21, 63, 200):                                                           # against its own averages
    _feat(f"sma_{_k}")(lambda b, k=_k: b.P / rmean(b.P, k) - 1.0)
_feat("gap_1")(lambda b: b.lov)
_feat("gapabs_21")(lambda b: rmean(np.abs(b.lov), 21))
_feat("it_1")(lambda b: b.lit)
_feat("seas_252")(lambda b: lag(rsum(b.lr, 21), 231))                                 # the month that followed, a year ago
_feat("age")(lambda b: np.minimum(np.cumsum(b.valid, axis=0), 252).astype(np.float32) / 252)


@_feat("lprice")
def _lprice(b):
    with quiet():
        return np.log(b.p.c)


NAMES = [n for n, _ in _REG]
F = len(NAMES)


def raw(panel: Panel, index_symbol: str) -> Iterator[tuple[str, np.ndarray]]:
    """Each figure as a [T, N] table, one at a time (ten years of all of them at once would not fit)."""
    b = Base(panel, index_symbol)
    for name, fn in _REG:
        a = np.array(fn(b), dtype=np.float32)       # a copy: some figures are tables the others share
        a[~b.valid] = np.nan                         # no bar today: nothing to say about today
        yield name, a


def xs_rank(a: np.ndarray, member: np.ndarray) -> np.ndarray:
    """Each row's values as places among that day's members: -1 lowest to +1 highest, NaN elsewhere."""
    out = np.full(a.shape, np.nan, dtype=np.float32)
    for t in range(a.shape[0]):
        m = member[t] & np.isfinite(a[t])
        n = int(m.sum())
        if n >= 2:
            out[t, m] = (rankdata(a[t, m]) - 0.5) / n * 2.0 - 1.0
    return out


# ── the market that day: the same row for every stock ───────────────────────────────────────────
CTX_NAMES = ["idx_1", "idx_5", "idx_21", "idx_63", "idxvol_21", "idxvol_63", "breadth_50", "disp_1", "disp_21",
             "small_21", "tight_21", "dow_0", "dow_1", "dow_2", "dow_3", "dow_4", "month_end", "month_start"]
C = len(CTX_NAMES)


def context(panel: Panel, index_symbol: str) -> np.ndarray:
    """[T, C], scaled by fixed constants (a running average would depend on where the table starts)."""
    b = Base(panel, index_symbol)
    T = panel.T
    member = panel.member if panel.member is not None else b.valid
    li = np.log1p(np.nan_to_num(b.idx))[:, None].astype(np.float32)
    idx_r = b.idx[:, None].astype(np.float32)
    out = np.zeros((T, C), dtype=np.float32)
    with quiet():
        for j, k in enumerate((1, 5, 21, 63)):
            out[:, j] = rsum(li, k)[:, 0] / (0.011 * np.sqrt(k))
        out[:, 4] = rstd(idx_r, 21)[:, 0] / 0.011 - 1.0
        out[:, 5] = rstd(idx_r, 63)[:, 0] / 0.011 - 1.0
        above = np.where(member & np.isfinite(b.P), b.P > rmean(b.P, 50), np.nan)
        out[:, 6] = (np.nanmean(above, axis=1) - 0.5) * 4.0
        disp = np.nanstd(np.where(member, b.r, np.nan), axis=1)
        out[:, 7] = disp / 0.02 - 1.0
        out[:, 8] = rmean(disp[:, None].astype(np.float32), 21)[:, 0] / 0.02 - 1.0
        out[:, 9] = rsum((np.log1p(np.nan_to_num(b.mkt))[:, None] - li).astype(np.float32), 21)[:, 0] / 0.02
        var_each = np.nanmean(np.where(member, rstd(b.r, 21) ** 2, np.nan), axis=1)
        out[:, 10] = rstd(b.mkt[:, None].astype(np.float32), 21)[:, 0] ** 2 / var_each * 4.0 - 1.0   # how much moves together
    for t, d in enumerate(panel.days):                       # the calendar, by rule: known in advance, no table needed
        day = date(int(d) // 10000, int(d) // 100 % 100, int(d) % 100)
        out[t, 11 + min(day.weekday(), 4)] = 1.0
        n1 = next_trading_day(day)
        out[t, 16] = n1.month != day.month or next_trading_day(n1).month != day.month
        p1 = prev_trading_day(day)
        out[t, 17] = p1.month != day.month or prev_trading_day(p1).month != day.month
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -5.0, 5.0)


def today(panel: Panel, index_symbol: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The newest day's inputs: (X [n, F] in -1..1, the members' column numbers, the market row [C]).
    `panel` should hold at least LOOKBACK days."""
    t = panel.T - 1
    cols = panel.members(t)
    cols = cols[np.isfinite(panel.c[t, cols])]
    member = np.zeros((1, panel.N), dtype=bool)
    member[0, cols] = True
    X = np.zeros((len(cols), F), dtype=np.float32)
    for f, (_, a) in enumerate(raw(panel, index_symbol)):
        X[:, f] = np.nan_to_num(xs_rank(a[t:t + 1], member)[0, cols], nan=0.0)
    ctx = context(panel, index_symbol)[t]
    return X, cols, ctx
