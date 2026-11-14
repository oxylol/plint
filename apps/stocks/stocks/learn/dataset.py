"""The training table: one row per stock per day it was in the universe.

    X    [R, F]  the stock's place among that day's stocks on each figure (-1..1)
    Y    [R, H]  what to learn: its place among that day's stocks on the return that followed
    R    [R, H]  that return itself, over the universe's average (for scoring, never for fitting)
    ctx  [T, C]  the market that day
    ptr  [T+1]   rows ptr[t]:ptr[t+1] are day t;  col [R] is each row's column in the panel

The return that followed is the one a decision made that evening could earn: in at the next open, out at
the open H days later. Y ranks it after dividing by the stock's own volatility, so the target is "did
better than its peers for its temperament", not "is a wild stock"."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import features
from ..features import rstd, xs_rank
from ..panel import Panel
from ..util import quiet

HORIZONS = (1, 5, 10, 20)
MAIN = 1                    # the 5-day column: what the book trades on and the ledger scores
H_MAX = max(HORIZONS)
WARMUP = 252                # days of history before a row is used


def forward(panel: Panel, h: int) -> np.ndarray:
    """[T, N]: return from the open after day t to the open h days later. NaN where it could not be entered
    (no bar the next day) or is not known yet. A day without a bar inside the window counts as no move."""
    T = panel.T
    with quiet():
        it = np.nan_to_num(np.log1p(np.clip(panel.intr, -0.95, None)))
        ov = np.nan_to_num(np.log1p(np.clip(panel.ovn, -0.95, None)))
    oo = np.zeros_like(it, dtype=np.float64)
    oo[:-1] = it[:-1] + ov[1:]                       # open of d to open of d+1
    cs = np.cumsum(oo, axis=0)
    out = np.full(it.shape, np.nan, dtype=np.float32)
    last = T - h - 2                                 # needs the open of day t+h+1
    if last >= 0:
        out[:last + 1] = np.expm1(cs[h:h + last + 1] - cs[:last + 1])
        enterable = np.isfinite(panel.c[1:last + 2]) & (panel.o[1:last + 2] > 0)
        out[:last + 1][~enterable] = np.nan
    return out


def relative(fwd: np.ndarray, member: np.ndarray) -> np.ndarray:
    """Over the average of that day's members."""
    with quiet():
        mean = np.nanmean(np.where(member, fwd, np.nan), axis=1, keepdims=True)
    return (fwd - mean).astype(np.float32)


@dataclass
class Table:
    days: np.ndarray
    syms: list[str]
    ptr: np.ndarray
    col: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    R: np.ndarray
    ctx: np.ndarray

    @property
    def T(self) -> int:
        return len(self.days)

    def rows(self, t: int) -> slice:
        return slice(int(self.ptr[t]), int(self.ptr[t + 1]))

    def span(self, t0: int, t1: int) -> slice:
        """Rows of days t0..t1-1."""
        return slice(int(self.ptr[t0]), int(self.ptr[t1]))

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        for name in ("ptr", "col", "Y", "R", "ctx", "days"):
            np.save(out / f"{name}.npy", getattr(self, name))
        if not isinstance(self.X, np.memmap):
            np.save(out / "X.npy", self.X)
        (out / "meta.json").write_text(json.dumps({"syms": self.syms, "features": features.NAMES,
                                                   "context": features.CTX_NAMES, "horizons": HORIZONS}))

    @classmethod
    def load(cls, src: Path) -> "Table":
        meta = json.loads((src / "meta.json").read_text())
        if meta["features"] != features.NAMES or tuple(meta["horizons"]) != HORIZONS:
            raise ValueError("the stored training table was built from other figures: rebuild it")
        get = lambda n, mm=None: np.load(src / f"{n}.npy", mmap_mode=mm)
        return cls(get("days"), meta["syms"], get("ptr"), get("col"), get("X", "r"), get("Y"), get("R"), get("ctx"))


def build(panel: Panel, index_symbol: str, out: Path | None = None) -> Table:
    """From a panel with its members set. With `out`, X is written straight to disk (ten years do not fit
    in memory twice)."""
    member = panel.member & np.isfinite(panel.c)
    counts = member.sum(axis=1)
    ptr = np.r_[0, np.cumsum(counts)].astype(np.int64)
    col = np.nonzero(member)[1].astype(np.int32)
    n_rows = int(ptr[-1])
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        X = np.lib.format.open_memmap(out / "X.npy", mode="w+", dtype=np.float16, shape=(n_rows, features.F))
    else:
        X = np.zeros((n_rows, features.F), dtype=np.float16)
    for f, (_, a) in enumerate(features.raw(panel, index_symbol)):
        X[:, f] = np.nan_to_num(xs_rank(a, member)[member], nan=0.0)
    Y = np.full((n_rows, len(HORIZONS)), np.nan, dtype=np.float32)
    R = np.full_like(Y, np.nan)
    vol = rstd(panel.ret, 63)
    for i, h in enumerate(HORIZONS):
        rel = relative(forward(panel, h), member)
        R[:, i] = rel[member]
        with quiet():
            scaled = rel / np.where(vol > 1e-4, vol, np.nan)
        Y[:, i] = xs_rank(scaled, member)[member]
    table = Table(panel.days.copy(), list(panel.syms), ptr, col, X, Y, R, features.context(panel, index_symbol))
    if out is not None:
        X.flush()
        table.save(out)
    return table
