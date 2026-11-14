"""Dated facts about companies from their SEC filings: when results came out, when insiders bought.

Every fact carries `day`: the first trading day whose close comes after the filing was public. A filing
accepted at 16:05 New York time belongs to the next trading day; one accepted at 09:10 to the same day.
That day is the first evening a model may know it, and for results it is also the day the price reacts.

Two tables, each a parquet file under <cache>/events/:

    earnings   sym, day, at, form, accession
               one row per results announcement (Form 8-K with item 2.02)
    insiders   sym, day, at, owner, officer, director, ten_pct, title, shares, price, usd, accession
               one row per open-market purchase (Form 4, transaction code P, shares acquired)

`at` is the acceptance time as an ISO string in New York time. Symbols are today's tickers: a company that
has left the market since has no ticker to look its filings up by, so its history is missing here (the
event models' backtests see fewer companies in the early years than really existed, and say so)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .panel import Panel
from .types import Event

EARNINGS_COLS = ["sym", "day", "at", "form", "accession"]
INSIDER_COLS = ["sym", "day", "at", "owner", "officer", "director", "ten_pct", "title", "shares", "price", "usd", "accession"]
FORMS_FOR_AGENT = ("8-K", "10-Q", "10-K", "4", "SC 13D", "SC 13D/A", "SC 13G", "6-K", "20-F")


class Events:
    def __init__(self, cache_dir: Path):
        self.dir = Path(cache_dir) / "events"
        self._earnings: pd.DataFrame | None = None
        self._insiders: pd.DataFrame | None = None

    # ── the tables ──────────────────────────────────────────────────────────────────────────────
    @property
    def earnings(self) -> pd.DataFrame:
        if self._earnings is None:
            self._earnings = self._read("earnings", EARNINGS_COLS)
        return self._earnings

    @property
    def insiders(self) -> pd.DataFrame:
        if self._insiders is None:
            self._insiders = self._read("insiders", INSIDER_COLS)
        return self._insiders

    def _read(self, name: str, cols: list[str]) -> pd.DataFrame:
        path = self.dir / f"{name}.parquet"
        return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=cols)

    def add(self, name: str, rows: pd.DataFrame) -> int:
        """Append rows to a table, dropping any already there (same accession, symbol and owner). Returns
        how many were new."""
        cols = EARNINGS_COLS if name == "earnings" else INSIDER_COLS
        key = ["accession", "sym"] + (["owner", "shares", "price"] if name == "insiders" else [])
        old = self._read(name, cols)
        both = pd.concat([old, rows[cols]], ignore_index=True).drop_duplicates(key, keep="first")
        self.dir.mkdir(parents=True, exist_ok=True)
        both.to_parquet(self.dir / f"{name}.parquet", index=False)
        self._earnings = self._insiders = None
        return len(both) - len(old)

    def empty(self) -> bool:
        return self.earnings.empty and self.insiders.empty

    # ── as day-by-stock tables for the models ───────────────────────────────────────────────────
    def _mark(self, panel: Panel, df: pd.DataFrame, value=True, dtype=bool) -> np.ndarray:
        out = np.zeros((panel.T, panel.N), dtype=dtype)
        if df.empty:
            return out
        df = df[df["sym"].isin(panel.ix) & (df["day"] >= int(panel.days[0])) & (df["day"] <= int(panel.days[-1]))]
        if df.empty:
            return out
        t = np.searchsorted(panel.days, df["day"].to_numpy())           # a day the market was shut maps forward
        t = np.clip(t, 0, panel.T - 1)
        j = df["sym"].map(panel.ix).to_numpy()
        if dtype is bool:
            out[t, j] = True
        else:
            np.add.at(out, (t, j), np.asarray(value, dtype=dtype))
        return out

    def earnings_table(self, panel: Panel) -> np.ndarray:
        """[T, N] bool: True on the day a stock's results could first move its price."""
        return self._mark(panel, self.earnings)

    def earnings_within(self, panel: Panel, k: int) -> np.ndarray:
        """[T, N] bool: results came out within the last k trading days, today included."""
        hit = self.earnings_table(panel).astype(np.int32)
        cs = np.cumsum(hit, axis=0)
        prev = np.zeros_like(cs)
        prev[k:] = cs[:-k]
        return (cs - prev) > 0

    def insider_clusters(self, panel: Panel, window: int = 21, min_buyers: int = 2, min_usd: float = 100_000.0,
                         officers_only: bool = True) -> np.ndarray:
        """[T, N] float32: on the day a cluster of insider buying first forms (at least `min_buyers` different
        officers or directors bought on the open market within `window` trading days, at least `min_usd`
        together), the dollars bought in that window; 0 elsewhere. A cluster fires once: not again until the
        window has emptied."""
        out = np.zeros((panel.T, panel.N), dtype=np.float32)
        df = self.insiders
        if df.empty:
            return out
        if officers_only:
            df = df[df["officer"] | df["director"]]
        df = df[df["sym"].isin(panel.ix) & (df["day"] <= int(panel.days[-1]))]
        for sym, g in df.groupby("sym"):
            j = panel.ix[sym]
            t = np.clip(np.searchsorted(panel.days, g["day"].to_numpy()), 0, panel.T - 1)
            owners, usd = g["owner"].to_numpy(), g["usd"].to_numpy(dtype=np.float64)
            active = False
            for day in np.unique(t):
                inside = (t <= day) & (t > day - window)
                formed = len(set(owners[inside])) >= min_buyers and usd[inside].sum() >= min_usd
                if formed and not active:
                    out[day, j] = usd[inside].sum()
                active = formed
        return out

    # ── for the agent ───────────────────────────────────────────────────────────────────────────
    def recent(self, since: str, symbols: set[str]) -> list[Event]:
        """Filings about these companies accepted after `since` (ISO, New York time), oldest first. Filled by
        `update`; kept in <cache>/events/recent.parquet for a month."""
        path = self.dir / "recent.parquet"
        if not path.exists():
            return []
        df = pd.read_parquet(path)
        df = df[(df["at"] > since) & df["sym"].isin(symbols)].sort_values("at")
        return [Event(id=r.accession, kind=r.form, symbol=r.sym, at=r.at, title=r.title, url=r.url,
                      facts={"items": r.items} if r.items else {}) for r in df.itertuples()]
