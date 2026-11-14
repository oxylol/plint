"""Prices as day-by-stock tables in memory: row t is a trading day, column j a stock.

Everything a model computes comes from a Panel. `upto(t)` is the only way a decision sees the market: it
returns the rows up to and including day t and nothing after, so what a model can know on an evening is a
matter of which rows it was handed, not of care taken inside the model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bars import BarStore
from .util import quiet

FIELDS = ("o", "h", "l", "c", "v", "vw", "n", "ret")


class Panel:
    def __init__(self, days: np.ndarray, syms: list[str], data: dict[str, np.ndarray],
                 member: np.ndarray | None = None):
        self.days = np.asarray(days, dtype=np.int32)
        self.syms = list(syms)
        self.ix = {s: j for j, s in enumerate(self.syms)}
        self.data = data
        self.member = member                    # [T, N] bool: in that month's universe
        self._cache: dict[str, np.ndarray] = {}

    @property
    def T(self) -> int:
        return len(self.days)

    @property
    def N(self) -> int:
        return len(self.syms)

    def __getattr__(self, name: str) -> np.ndarray:
        if name in FIELDS:
            return self.data[name]
        raise AttributeError(name)

    # ── derived, each row from that row alone ───────────────────────────────────────────────────
    def _derived(self, name: str, make) -> np.ndarray:
        if name not in self._cache:
            with quiet():
                self._cache[name] = make().astype(np.float32)
        return self._cache[name]

    @property
    def intr(self) -> np.ndarray:
        """Open to close, the same day."""
        return self._derived("intr", lambda: np.where(self.o > 0, self.c / self.o - 1.0, np.nan))

    @property
    def ovn(self) -> np.ndarray:
        """Previous close to this open, dividends and splits included (the total return less the day's own move)."""
        return self._derived("ovn", lambda: (1.0 + self.ret) / (1.0 + self.intr) - 1.0)

    @property
    def dv(self) -> np.ndarray:
        """Dollars traded."""
        return self._derived("dv", lambda: np.where(self.vw > 0, self.vw, self.c) * self.v)

    # ── views ───────────────────────────────────────────────────────────────────────────────────
    def _slice(self, s: slice) -> "Panel":
        p = Panel(self.days[s], self.syms, {k: v[s] for k, v in self.data.items()},
                  None if self.member is None else self.member[s])
        p._cache = {k: v[s] for k, v in self._cache.items()}
        return p

    def upto(self, t: int) -> "Panel":
        """Days 0..t. What an evening may see."""
        return self._slice(slice(0, t + 1))

    def tail(self, k: int) -> "Panel":
        return self._slice(slice(max(0, self.T - k), self.T))

    def window(self, t: int, k: int) -> "Panel":
        """The k days ending at t."""
        return self._slice(slice(max(0, t + 1 - k), t + 1))

    def t_of(self, d: int) -> int:
        i = int(np.searchsorted(self.days, d))
        if i >= self.T or self.days[i] != d:
            raise KeyError(f"{d} is not a trading day in this panel")
        return i

    def members(self, t: int) -> np.ndarray:
        """Column numbers of the universe on day t."""
        return np.flatnonzero(self.member[t]) if self.member is not None else np.arange(self.N)

    def set_members(self, universe) -> None:
        member = np.zeros((self.T, self.N), dtype=bool)
        months = self.days // 100
        for m in np.unique(months):
            cols = [self.ix[s] for s in universe.members(int(m) * 100 + 1) if s in self.ix]
            rows = np.flatnonzero(months == m)
            member[np.ix_(rows, cols)] = True
        self.member = member

    @classmethod
    def load(cls, store: BarStore, syms: list[str], calendar_symbol: str, start: int | None = None,
             end: int | None = None) -> "Panel":
        """Read the price table into memory, a year at a time so the long form never sits there whole."""
        syms = sorted(set(syms) | {calendar_symbol})
        days = store.days(calendar_symbol, start, end)
        day_ix = {int(d): i for i, d in enumerate(days)}
        sym_ix = {s: j for j, s in enumerate(syms)}
        data = {f: np.full((len(days), len(syms)), np.nan, dtype=np.float32) for f in FIELDS}
        for y in sorted({int(d) // 10000 for d in days}):
            lo, hi = max(y * 10000 + 101, int(days[0])), min(y * 10000 + 1231, int(days[-1]))
            df = store.read(lo, hi, syms)
            df = df[df["d"].isin(day_ix)]
            if df.empty:
                continue
            ti = df["d"].map(day_ix).to_numpy()
            ni = df["sym"].map(sym_ix).to_numpy()
            for f in FIELDS:
                data[f][ti, ni] = df[f].to_numpy(dtype=np.float32)
        return cls(days, syms, data)

    @classmethod
    def from_frame(cls, df: pd.DataFrame, calendar_symbol: str | None = None) -> "Panel":
        """From a long table already in memory (tests, small jobs)."""
        days = np.sort(df.loc[df["sym"] == calendar_symbol, "d"].unique() if calendar_symbol else df["d"].unique())
        syms = sorted(df["sym"].unique())
        day_ix = {int(d): i for i, d in enumerate(days)}
        sym_ix = {s: j for j, s in enumerate(syms)}
        df = df[df["d"].isin(day_ix)]
        ti, ni = df["d"].map(day_ix).to_numpy(), df["sym"].map(sym_ix).to_numpy()
        data = {f: np.full((len(days), len(syms)), np.nan, dtype=np.float32) for f in FIELDS}
        for f in FIELDS:
            data[f][ti, ni] = df[f].to_numpy(dtype=np.float32)
        return cls(days, syms, data)
