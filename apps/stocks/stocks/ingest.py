"""Fetching prices into the table: the long history once, then a short tail every evening.

Each pull asks for the same window twice, as it traded and with splits and dividends applied. The ratio of
consecutive adjusted closes is the day's total return; the raw bar is stored beside it. The tail is pulled
again every evening for a couple of weeks back, so a correction on Alpaca's side (a late split, a busted
trade) reaches the table and is logged."""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Callable

import numpy as np
import pandas as pd

from .alpaca import Alpaca, AlpacaError
from .bars import BarStore, d2s

log = logging.getLogger("stocks.ingest")

# what an unadjusted split looks like in the raw close-to-close move (new/old share counts)
SPLIT_RATIOS = [2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30, 40, 50, 1.5, 1.25, 4 / 3,
                1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 6, 1 / 8, 1 / 10, 1 / 12, 1 / 15, 1 / 20, 1 / 25, 1 / 30, 1 / 40, 1 / 50]


def _frame(pairs) -> pd.DataFrame:
    rows = [(int(b["t"][:4]) * 10000 + int(b["t"][5:7]) * 100 + int(b["t"][8:10]), s,
             b["o"], b["h"], b["l"], b["c"], b["v"], b.get("vw", b["c"]), b.get("n", 0)) for s, b in pairs]
    return pd.DataFrame(rows, columns=["d", "sym", "o", "h", "l", "c", "v", "vw", "n"])


def pull(api: Alpaca, symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """Rows for the symbols between two dates, with `ret` computed inside the pull (so both closes of every
    ratio are on the same adjustment basis). The first row of each symbol has no return."""
    raw = _frame(api.bars(symbols, start, end, "raw"))
    if raw.empty:
        return raw.assign(ret=pd.Series(dtype="float32"))
    adj = _frame(api.bars(symbols, start, end, "all"))[["d", "sym", "c"]].rename(columns={"c": "ac"})
    df = raw.merge(adj, on=["d", "sym"], how="left").sort_values(["sym", "d"], kind="stable")
    g = df.groupby("sym", sort=False)
    df["ret"] = df["ac"] / g["ac"].shift(1) - 1.0
    df["raw_move"] = df["c"] / g["c"].shift(1) - 1.0
    return df.drop(columns=["ac"]).reset_index(drop=True)


def split_suspects(df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose raw move is the size of a split and whose adjusted return is the same move: the adjustment
    has not reached Alpaca's series yet. Only a suspicion; `apply_splits` asks the corporate-action list."""
    if df.empty:
        return df
    big = df[(df["raw_move"].abs() >= 0.19) & ((df["ret"] - df["raw_move"]).abs() < 1e-3)]
    if big.empty:
        return big
    ratio = 1.0 + big["raw_move"].to_numpy()                 # new price / old price = old shares / new shares
    near = np.zeros(len(big), dtype=bool)
    for r in SPLIT_RATIOS:
        near |= np.abs(ratio * r - 1.0) < 0.04
    return big[near]


def apply_splits(api: Alpaca, df: pd.DataFrame, note: Callable[[str], None] = log.warning) -> pd.DataFrame:
    """Where the corporate-action list confirms a split on a suspected day, compute the return ourselves.
    A halving that no split explains stays what it looks like: a real loss."""
    sus = split_suspects(df)
    if sus.empty:
        return df
    lo, hi = d2s(sus["d"].min()), d2s(sus["d"].max())
    try:
        acts = api.corporate_actions(sorted(set(sus["sym"])), lo, hi, "forward_split,reverse_split")
    except AlpacaError as e:
        note(f"could not check {len(sus)} possible splits: {e}")
        return df
    factor = {}
    for row in acts.get("forward_splits", []) + acts.get("reverse_splits", []):
        try:
            factor[(row["symbol"], int(row["ex_date"].replace("-", "")))] = float(row["new_rate"]) / float(row["old_rate"])
        except (KeyError, ValueError, ZeroDivisionError):
            continue
    df = df.copy()
    for i, r in sus.iterrows():
        f = factor.get((r["sym"], int(r["d"])))
        if f:
            fixed = (1.0 + r["raw_move"]) * f - 1.0
            df.loc[i, "ret"] = fixed
            note(f"split not yet in the adjusted prices: {r['sym']} {d2s(r['d'])}, return set to {fixed:+.2%} (was {r['ret']:+.2%})")
    return df


def backfill(api: Alpaca, store: BarStore, symbols: list[str], start: str, end: str, chunk: int = 50,
             progress: Callable[[int, int, int], None] | None = None) -> int:
    """The long history, a handful of symbols at a time (a page holds 10,000 bars). Returns rows written."""
    total = 0
    for i in range(0, len(symbols), chunk):
        part = symbols[i:i + chunk]
        try:
            df = apply_splits(api, pull(api, part, start, end))
        except AlpacaError as e:
            if e.status in (400, 404, 422) and len(part) > 1:     # one bad symbol must not lose the other 49
                for s in part:
                    try:
                        one = apply_splits(api, pull(api, [s], start, end))
                        store.upsert(one.drop(columns=["raw_move"], errors="ignore"))
                        total += len(one)
                    except AlpacaError as e1:
                        log.warning("no history for %s: %s", s, e1)
                continue
            raise
        store.upsert(df.drop(columns=["raw_move"], errors="ignore"))
        total += len(df)
        if progress:
            progress(min(i + chunk, len(symbols)), len(symbols), total)
    return total


def update(api: Alpaca, store: BarStore, symbols: list[str], today: date, back_days: int = 16, chunk: int = 200,
           note: Callable[[str], None] = log.warning) -> dict:
    """The evening tail: the last couple of weeks again, for every symbol. Returns what changed."""
    start, end = (today - timedelta(days=back_days)).isoformat(), today.isoformat()
    before = store.last_day()
    rows, corrections = 0, []
    for i in range(0, len(symbols), chunk):
        df = apply_splits(api, pull(api, symbols[i:i + chunk], start, end), note)
        if df.empty:
            continue
        first = df.groupby("sym")["d"].transform("min")
        df = df[(df["d"] > first) | df["ret"].notna()]          # a window's first row only if it carries a return
        corrections += store.upsert(df.drop(columns=["raw_move"], errors="ignore"))
        rows += len(df)
    for c in corrections:
        note(f"data corrected: {c['sym']} {d2s(c['d'])} return {c['was']:+.2%} → {c['now']:+.2%}")
    return {"rows": rows, "corrections": corrections, "before": before, "after": store.last_day()}
