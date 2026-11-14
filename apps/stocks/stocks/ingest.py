"""Fetching prices into the table: the long history once, then a short tail every evening.

Each pull asks for the same window twice, as it traded and with splits and dividends applied. The ratio of
consecutive adjusted closes is the day's total return; the raw bar is stored beside it. The tail is pulled
again every evening for a couple of weeks back, so a correction on Alpaca's side (a late split, a busted
trade) reaches the table and is logged."""
from __future__ import annotations

import logging
import re
import zlib
from datetime import date, timedelta
from typing import Callable

import pandas as pd
import pyarrow.parquet as pq

from .alpaca import Alpaca, AlpacaError
from .bars import COLS, BarStore, d2s, tidy

log = logging.getLogger("stocks.ingest")
LONG_GAP_DAYS = 45
TICKER = re.compile(r"^[A-Z]{1,5}(\.[A-TVX-Z])?$")        # a share class after the dot; .U .W .R are units, warrants, rights
PAPER = re.compile(r"^[A-Z]{4}[WURYFQ]$")          # a fifth letter: warrants, units, rights, foreign and bankrupt paper

def _frame(pairs) -> pd.DataFrame:
    # Alpaca fills a halted day with a bar that repeats the last price on zero volume. Nothing traded: no bar.
    rows = [(int(b["t"][:4]) * 10000 + int(b["t"][5:7]) * 100 + int(b["t"][8:10]), s,
             b["o"], b["h"], b["l"], b["c"], b["v"], b.get("vw", b["c"]), b.get("n", 0)) for s, b in pairs if b["v"] > 0]
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
    # A ticker is sometimes handed to another company after the first has gone. Bars that resume after a long
    # silence are a new listing, not a return on the old price.
    when = pd.to_datetime(df["d"].astype(str), format="%Y%m%d")
    silent = (when - when.groupby(df["sym"], sort=False).shift(1)).dt.days > LONG_GAP_DAYS
    df.loc[silent, ["ret", "raw_move"]] = float("nan")
    return df.drop(columns=["ac"]).reset_index(drop=True)


def gone_symbols(api: Alpaca, start: str, end: str, companies: set[str]) -> dict[str, str]:
    """Companies that were bought out, from the corporate-action list: {symbol: the day it happened}. The asset
    list forgets most of them, and a backtest that never held a company that was later taken over is a
    backtest of survivors. A buy-out paid in shares counts only when the buyer is itself a listed company
    (`companies`): funds merge into funds the same way. (The list is thin before 2020: what is still missing
    is said in every report.)"""
    acts = api.corporate_actions(None, start, end, "cash_merger,stock_and_cash_merger,stock_merger")
    out: dict[str, str] = {}
    for kind, rows in acts.items():
        for r in rows:
            sym, buyer = r.get("acquiree_symbol") or "", r.get("acquirer_symbol")
            if not TICKER.match(sym) or PAPER.search(sym) or sym == buyer:
                continue
            if kind != "cash_mergers" and buyer not in companies:
                continue
            out.setdefault(sym, r.get("effective_date") or r.get("process_date") or "")
    return out


def split_suspects(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a large raw move that the adjusted series repeats exactly: either a real jump, or a split
    whose adjustment has not reached Alpaca's series yet. Only a suspicion; `apply_splits` asks the
    corporate-action list, and only a split on record for that very day changes anything."""
    if df.empty:
        return df
    return df[(df["raw_move"].abs() >= 0.19) & ((df["ret"] - df["raw_move"]).abs() < 1e-3)]


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


def _history(api: Alpaca, part: list[str], start: str, end: str) -> pd.DataFrame:
    try:
        return apply_splits(api, pull(api, part, start, end))
    except AlpacaError as e:
        if e.status not in (400, 404, 422) or len(part) == 1:
            raise
    frames = []                                      # one bad symbol must not lose the other 49
    for s in part:
        try:
            frames.append(apply_splits(api, pull(api, [s], start, end)))
        except AlpacaError as e1:
            log.warning("no history for %s: %s", s, e1)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def backfill(api: Alpaca, store: BarStore, symbols: list[str], start: str, end: str, chunk: int = 50,
             progress: Callable[[int, int, int], None] | None = None) -> int:
    """The long history, a handful of symbols at a time (a page holds 10,000 bars). Each handful is parked
    in a staging file, so a stop halfway loses nothing; the year files are written once at the end.
    Returns rows written."""
    staging = store.dir / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    total = 0
    for i in range(0, len(symbols), chunk):
        part = symbols[i:i + chunk]
        path = staging / f"{i // chunk:05d}-{zlib.crc32(','.join(part + [start, end]).encode()):08x}.parquet"
        if path.exists():
            total += pq.ParquetFile(path).metadata.num_rows
        else:
            df = _history(api, part, start, end)
            df = tidy(df.drop(columns=["raw_move"], errors="ignore")) if len(df) else pd.DataFrame(columns=COLS)
            tmp = path.with_suffix(".tmp")
            df.to_parquet(tmp, index=False)
            tmp.replace(path)
            total += len(df)
        if progress:
            progress(min(i + chunk, len(symbols)), len(symbols), total)
    parts = sorted(staging.glob("*.parquet"))
    y0, y1 = int(start[:4]), int(end[:4])
    for year in range(y0, y1 + 1):                   # a year at a time: the whole history does not fit in memory
        frames = [pd.read_parquet(p, filters=[("d", ">=", year * 10000), ("d", "<", (year + 1) * 10000)]) for p in parts]
        frames = [f for f in frames if len(f)]
        if frames:
            store.upsert(pd.concat(frames, ignore_index=True))
    for p in parts:
        p.unlink()
    staging.rmdir()
    return total


def update(api: Alpaca, store: BarStore, symbols: list[str], today: date, back_days: int = 16, chunk: int = 200,
           note: Callable[[str], None] = log.warning) -> dict:
    """The evening tail: the last couple of weeks again, for every symbol. Returns what changed."""
    start, end = (today - timedelta(days=back_days)).isoformat(), today.isoformat()
    before = store.last_day()
    frames = []
    for i in range(0, len(symbols), chunk):
        df = apply_splits(api, pull(api, symbols[i:i + chunk], start, end), note)
        if df.empty:
            continue
        first = df.groupby("sym")["d"].transform("min")
        frames.append(df[(df["d"] > first) | df["ret"].notna()])          # a window's first row only if it carries a return
    rows = sum(len(f) for f in frames)
    corrections = store.upsert(pd.concat(frames, ignore_index=True).drop(columns=["raw_move"], errors="ignore")) if frames else []
    for c in corrections:
        note(f"data corrected: {c['sym']} {d2s(c['d'])} return {c['was']:+.2%} → {c['now']:+.2%}")
    return {"rows": rows, "corrections": corrections, "before": before, "after": store.last_day()}
