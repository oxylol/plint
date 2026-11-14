"""Which stocks the models may pick from, month by month, decided only from what was known then.

At each month end the common stocks are ranked by how much money changed hands in them (the median day of
the last three months); the top N at or above a minimum price are the universe for the following month.
A stock that later fell out, was bought or went bust is in the months it earned, which keeps most of the
survivor's flattery out of a backtest (most: free data has little on shares that stopped trading)."""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .bars import BarStore
from .cal import next_trading_day
from .util import quiet

EXCHANGES = {"NYSE", "NASDAQ", "AMEX"}          # ARCA and BATS list funds, not companies
FALLEN = "OTC"
WINDOW, MIN_BARS = 63, 40

# A fund, a note or a piece of paper on top of a share, by its listed name. Not perfect; what slips through
# is listed by `python -m stocks universe` and can be named in universe_overrides.json.
_FUND_HOUSE = re.compile(
    r"^(ishares|proshares|spdr|invesco (?!ltd)|vanguard|direxion|wisdomtree (?!,? ?inc)|vaneck|powershares|"
    r"graniteshares|yieldmax|roundhill|defiance|global x|first trust|grayscale|innovator|simplify|t-rex|"
    r"tradr|leverage shares|microsectors|ark |amplify|pacer|schwab|jpmorgan (equity premium|nasdaq)|"
    r"fidelity|franklin (ftse|templeton)|xtrackers|sprott|abrdn|nuveen|blackrock (?!,? ?inc)|pimco|eaton vance)\b", re.I)
_NOT_COMMON = re.compile(
    r"\b(etf|etn|etp|fund|trust,? series|ultrapro|ultrashort|[0-9](\.[0-9]+)?x (long|short|bull|bear|inverse)|"
    r"warrants?|rights?|preferred (stock|shares?)|preference shares?|pfd|"
    r"notes?( due)?|debentures?|[0-9](\.[0-9]+)? ?%)\b"
    r"|(?<!common )(?<!partner )\bunits?\b(?! representing)", re.I)
_ADR = re.compile(r"american depositary|\bads\b|\badr\b", re.I)
_PAPER_SYMBOL = re.compile(r"[.\-](WS|W|U|R|RT|P[A-Z]?|PR[A-Z]?)$|^[A-Z]{4}[WUR]$")


def is_common(symbol: str, name: str, exchange: str) -> bool:
    """A company's ordinary share (or its depositary receipt), listed on a main exchange, or one that was:
    a company pushed off its exchange is quoted over the counter afterwards, and its exchange years count
    (the price table only ever holds exchange trades, so its over-the-counter life is not in it)."""
    if exchange not in EXCHANGES and exchange != FALLEN:
        return False
    if _PAPER_SYMBOL.search(symbol or ""):
        return False
    if exchange == FALLEN and (not name or re.search(r"^[A-Z]{4}[YF]$", symbol or "")):
        return False                                 # five letters ending Y or F: foreign shares that never listed here
    name = name or ""
    if _FUND_HOUSE.search(name):
        return False
    if _ADR.search(name) and "preferred" not in name.lower():
        return True
    return not _NOT_COMMON.search(name)


class Assets:
    """The asset list as last fetched: symbol, name, exchange, whether still listed, whether a common stock."""

    def __init__(self, cache_dir: Path):
        self.path = Path(cache_dir) / "assets.parquet"
        self.overrides_path = Path(cache_dir) / "universe_overrides.json"
        self.gone_path = Path(cache_dir) / "gone.json"          # companies bought out: {symbol: day}
        self._df: pd.DataFrame | None = None

    def save_gone(self, gone: dict[str, str]) -> None:
        old = self.gone()
        old.update(gone)
        self.gone_path.write_text(json.dumps(old))
        self._df = None

    def gone(self) -> dict[str, str]:
        try:
            return json.loads(self.gone_path.read_text())
        except (FileNotFoundError, ValueError):
            return {}

    def save(self, assets: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame([{"sym": a["symbol"], "name": a.get("name") or "", "exchange": a.get("exchange") or "",
                            "active": a.get("status") == "active", "tradable": bool(a.get("tradable"))}
                           for a in assets if a.get("symbol")]).drop_duplicates("sym", keep="first")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.path, index=False)
        self._df = None
        return self.frame()

    def frame(self) -> pd.DataFrame:
        if self._df is None:
            if not self.path.exists():
                return pd.DataFrame(columns=["sym", "name", "exchange", "active", "tradable", "common"])
            df = pd.read_parquet(self.path)
            extra = [s for s in self.gone() if s not in set(df["sym"])]
            if extra:                               # no name, no exchange: all that is known is that it was bought
                df = pd.concat([df, pd.DataFrame({"sym": extra, "name": "", "exchange": "GONE", "active": False,
                                                  "tradable": False})], ignore_index=True)
            ov = self.overrides()
            common = [is_common(s, n, e) for s, n, e in zip(df["sym"], df["name"], df["exchange"])]
            df["common"] = (pd.Series(common, index=df.index) | (df["exchange"] == "GONE") | df["sym"].isin(ov["include"])) \
                & ~df["sym"].isin(ov["exclude"])
            self._df = df
        return self._df

    def overrides(self) -> dict:
        try:
            ov = json.loads(self.overrides_path.read_text())
        except (FileNotFoundError, ValueError):
            ov = {}
        return {"include": set(ov.get("include", [])), "exclude": set(ov.get("exclude", []))}

    def common(self, active_only: bool = False) -> list[str]:
        """Common stocks. `active_only`: those listed on a main exchange today (what is fetched every evening);
        otherwise also those that have left or fallen off (what the history is fetched for)."""
        df = self.frame()
        if df.empty:
            return []
        if active_only:
            df = df[df["common"] & df["active"] & df["exchange"].isin(EXCHANGES)]
        else:
            df = df[df["common"]]
        return sorted(df["sym"])

    def names(self) -> dict[str, str]:
        df = self.frame()
        return dict(zip(df["sym"], df["name"].map(short_name)))


_SUFFIX = re.compile(r"\s+(class [a-z]\s+)?(common stock|common shares?|ordinary shares?|american depositary shares?.*|"
                     r"capital stock|subordinate voting shares?|new common stock).*$", re.I)


def short_name(name: str) -> str:
    return _SUFFIX.sub("", name or "").strip(" ,")


def month_of(d: int) -> int:
    return d // 100


def next_month(m: int) -> int:
    return m + 1 if m % 100 < 12 else (m // 100 + 1) * 100 + 1


def rank_at(days: np.ndarray, syms: list[str], dv: np.ndarray, close: np.ndarray, t: int, size: int,
            min_price: float) -> list[tuple[str, int, float]]:
    """The top `size` at day index t: (symbol, rank, median dollar volume), from rows up to t only."""
    win = dv[max(0, t - WINDOW + 1):t + 1]
    n = np.isfinite(win).sum(axis=0)
    with quiet():
        med = np.nanmedian(win, axis=0)
    last = close[t]
    ok = (n >= MIN_BARS) & np.isfinite(med) & np.isfinite(last) & (last >= min_price)
    order = np.argsort(-np.where(ok, med, -np.inf), kind="stable")[:size]
    return [(syms[j], r + 1, float(med[j])) for r, j in enumerate(order) if ok[j]]


class Universe:
    """The membership table: (month, sym, rank, dv). `month` 202611 holds the stocks tradable in November 2026,
    picked at October's last close."""

    def __init__(self, cache_dir: Path):
        self.path = Path(cache_dir) / "universe.parquet"
        self._df: pd.DataFrame | None = None
        self._by_month: dict[int, list[str]] = {}

    def frame(self) -> pd.DataFrame:
        if self._df is None:
            self._df = (pd.read_parquet(self.path) if self.path.exists()
                        else pd.DataFrame(columns=["month", "sym", "rank", "dv"]))
            self._by_month = {}
        return self._df

    def months(self) -> list[int]:
        return sorted(int(m) for m in self.frame()["month"].unique())

    def members(self, d: int) -> list[str]:
        """The universe on day d, best-traded first. A month not built yet falls back to the latest one."""
        m = month_of(d)
        if m not in self._by_month:
            df = self.frame()
            have = df[df["month"] == m]
            if have.empty:
                earlier = [x for x in self.months() if x <= m]
                have = df[df["month"] == earlier[-1]] if earlier else have
            self._by_month[m] = list(have.sort_values("rank")["sym"])
        return self._by_month[m]

    def ranks(self, d: int) -> dict[str, int]:
        return {s: i + 1 for i, s in enumerate(self.members(d))}

    def ever(self) -> list[str]:
        return sorted(self.frame()["sym"].unique())

    def build(self, store: BarStore, common: list[str], index_symbol: str, size: int, min_price: float,
              since_month: int | None = None) -> int:
        """(Re)build membership for every month from `since_month` on (all history when None). Returns the
        number of months written."""
        start = None if since_month is None else (since_month // 100 * 10000 + since_month % 100 * 100 + 1) - 10000
        days = store.days(index_symbol, start)
        if len(days) < MIN_BARS:
            return 0
        day_ix = {int(d): i for i, d in enumerate(days)}
        have = store.symbols()
        common = [s for s in common if s in have]        # the asset list is several times longer than the price table
        sym_ix = {s: j for j, s in enumerate(common)}
        dv = np.full((len(days), len(common)), np.nan, dtype=np.float32)
        close = np.full_like(dv, np.nan)
        for y in sorted({int(d) // 10000 for d in days}):
            df = store.read(y * 10000 + 101, y * 10000 + 1231, None, ["d", "sym", "c", "v", "vw"])
            df = df[df["sym"].isin(sym_ix) & df["d"].isin(day_ix)]
            ti = df["d"].map(day_ix).to_numpy()
            ni = df["sym"].map(sym_ix).to_numpy()
            dv[ti, ni] = (df["vw"].where(df["vw"] > 0, df["c"]) * df["v"]).to_numpy(dtype=np.float32)
            close[ti, ni] = df["c"].to_numpy(dtype=np.float32)
        rows = []
        months = [month_of(int(d)) for d in days]
        for t in range(len(days)):
            if t + 1 < len(days):
                last_of_month = months[t + 1] != months[t]
            else:                                   # the newest day closes a month only if tomorrow is another one
                d = int(days[t])
                nxt = next_trading_day(date(d // 10000, d // 100 % 100, d % 100))
                last_of_month = nxt.year * 100 + nxt.month != months[t]
            if not last_of_month or t < MIN_BARS:
                continue
            target = next_month(months[t])
            if since_month is not None and target < since_month:
                continue
            rows += [(target, s, r, v) for s, r, v in rank_at(days, common, dv, close, t, size, min_price)]
        if not rows:
            return 0
        new = pd.DataFrame(rows, columns=["month", "sym", "rank", "dv"]).astype({"month": "int32", "rank": "int32", "dv": "float32"})
        old = self.frame()
        if len(old):
            new = pd.concat([old[~old["month"].isin(new["month"].unique())], new], ignore_index=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new.sort_values(["month", "rank"]).to_parquet(self.path, index=False)
        self._df = None
        return int(new["month"].nunique())
