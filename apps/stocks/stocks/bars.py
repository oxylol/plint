"""The price table: one row per stock per day, as it traded, plus that day's total return.

    d    the day, as the integer 20261113
    sym  the ticker
    o h l c v vw n   the bar as it traded (never adjusted)
    ret  total return from the previous close to this one, splits and dividends included

Raw prices never change after the fact; `ret` is the only adjusted figure, and it is a ratio, so a later
split leaves old rows alone. That is what lets the table be appended to day by day and still be the truth
(an adjusted price series is rewritten from its first row at every dividend). One parquet file a year."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

COLS = ["d", "sym", "o", "h", "l", "c", "v", "vw", "n", "ret"]
DTYPES = {"d": "int32", "o": "float32", "h": "float32", "l": "float32", "c": "float32", "v": "float32",
          "vw": "float32", "n": "int32", "ret": "float32"}
CORRECTION = 0.005        # a stored return that moves by more than this is logged as a correction


def d2s(d: int) -> str:
    d = int(d)
    return f"{d // 10000:04d}-{d // 100 % 100:02d}-{d % 100:02d}"


def s2d(s: str) -> int:
    return int(s[:4]) * 10000 + int(s[5:7]) * 100 + int(s[8:10])


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    df = df[COLS].copy()
    df["sym"] = df["sym"].astype(str)
    for c, t in DTYPES.items():
        df[c] = df[c].astype(t)
    return df.sort_values(["sym", "d"], kind="stable").reset_index(drop=True)


class BarStore:
    def __init__(self, cache_dir: Path):
        self.dir = Path(cache_dir) / "bars"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, year: int) -> Path:
        return self.dir / f"{year}.parquet"

    def years(self) -> list[int]:
        return sorted(int(p.stem) for p in self.dir.glob("[0-9][0-9][0-9][0-9].parquet"))

    def read(self, start: int | None = None, end: int | None = None, symbols: Iterable[str] | None = None,
             columns: list[str] | None = None) -> pd.DataFrame:
        """Rows with start <= d <= end, for the symbols given (all when None)."""
        want = columns or COLS
        cols = sorted(set(want) | {"d", "sym"})
        syms = None if symbols is None else list(symbols)
        parts = []
        for y in self.years():
            if (start and y < start // 10000) or (end and y > end // 10000):
                continue
            filters = [("sym", "in", syms)] if syms is not None else None
            df = pd.read_parquet(self._path(y), columns=cols, filters=filters)
            if start:
                df = df[df["d"] >= start]
            if end:
                df = df[df["d"] <= end]
            parts.append(df)
        if not parts:
            return pd.DataFrame({c: pd.Series(dtype=DTYPES.get(c, "object")) for c in want})
        return pd.concat(parts, ignore_index=True)[want]

    def symbols(self) -> set[str]:
        """Every symbol that has at least one bar."""
        out: set[str] = set()
        for y in self.years():
            out.update(pd.read_parquet(self._path(y), columns=["sym"])["sym"].unique())
        return out

    def last_day(self, symbol: str | None = None) -> int | None:
        for y in reversed(self.years()):
            df = pd.read_parquet(self._path(y), columns=["d"], filters=[("sym", "==", symbol)] if symbol else None)
            if len(df):
                return int(df["d"].max())
        return None

    def days(self, symbol: str, start: int | None = None, end: int | None = None) -> np.ndarray:
        """The days a symbol traded: for the index fund, that is the calendar."""
        return np.sort(self.read(start, end, [symbol], ["d"])["d"].to_numpy())

    def upsert(self, new: pd.DataFrame) -> list[dict]:
        """Add rows, replacing any with the same day and symbol. A new row without a return (the first row
        of a pull has no previous close to compare with) keeps the return already stored. Returns the
        corrections: stored returns that changed by more than half a percent."""
        if new.empty:
            return []
        new = tidy(new)
        corrections: list[dict] = []
        for year, part in new.groupby(new["d"] // 10000):
            path = self._path(int(year))
            if path.exists():
                old = pd.read_parquet(path)
                key = ["d", "sym"]
                both = part.merge(old[key + ["ret"]], on=key, how="left", suffixes=("", "_old"))
                keep_old = both["ret"].isna() & both["ret_old"].notna()
                both.loc[keep_old, "ret"] = both.loc[keep_old, "ret_old"]
                moved = both[both["ret"].notna() & both["ret_old"].notna()
                             & ((both["ret"] - both["ret_old"]).abs() > CORRECTION)]
                corrections += [{"sym": r.sym, "d": int(r.d), "was": float(r.ret_old), "now": float(r.ret)}
                                for r in moved.itertuples()]
                part = both[COLS]
                merged = pd.concat([old, part], ignore_index=True).drop_duplicates(key, keep="last")
            else:
                merged = part
            tmp = path.with_suffix(".tmp")
            tidy(merged).to_parquet(tmp, index=False, compression="zstd")
            tmp.replace(path)
        return corrections
