"""Stands in for Alpaca's API in the tests: serves the made-up market up to a day it is told is today."""
from __future__ import annotations

import numpy as np
import pandas as pd

NAMES = {"SPY": ("SPDR S&P 500 ETF Trust", "ARCA"), "BIL": ("SPDR Bloomberg 1-3 Month T-Bill ETF", "ARCA")}
DECOYS = [("TQQQ", "ProShares UltraPro QQQ", "NASDAQ"), ("ZZZW", "Zeta Acquisition Corp. Warrant", "NASDAQ"),
          ("BAC.PRB", "Bank of America Corporation Depositary Shares each representing Preferred Stock", "NYSE")]


def _iso(d: int) -> str:
    return f"{d // 10000:04d}-{d // 100 % 100:02d}-{d % 100:02d}"


class FakeAlpaca:
    def __init__(self, frame: pd.DataFrame, today: int):
        self.today = today
        self.calls = 0
        self.late_splits: set[tuple[str, int]] = set()      # (symbol, day): the adjustment has not arrived yet
        self.splits: dict[tuple[str, int], tuple[float, float]] = {}   # (symbol, day) → (new_rate, old_rate)
        self.by_sym = {}
        for sym, g in frame.sort_values(["sym", "d"]).groupby("sym"):
            g = g.reset_index(drop=True)
            ac = g["c"].to_numpy(dtype=float).copy()
            ret = g["ret"].to_numpy(dtype=float)
            for i in range(len(g) - 2, -1, -1):              # back-adjust: ratios of neighbours are the total return
                ac[i] = ac[i + 1] / (1.0 + ret[i + 1])
            g["ac"] = ac
            self.by_sym[sym] = g

    def assets(self, status=None):
        out = []
        for sym, g in self.by_sym.items():
            name, exch = NAMES.get(sym, (f"{sym} Holdings Inc. Common Stock", "NASDAQ"))
            active = int(g["d"].iloc[-1]) >= self.today or int(g["d"].max()) >= int(max(x["d"].max() for x in self.by_sym.values()))
            out.append({"symbol": sym, "name": name, "exchange": exch, "status": "active" if active else "inactive", "tradable": active})
        out += [{"symbol": s, "name": n, "exchange": e, "status": "active", "tradable": True} for s, n, e in DECOYS]
        return [a for a in out if status is None or a["status"] == status]

    def calendar(self, start, end):
        return []

    def bars(self, symbols, start, end, adjustment="raw"):
        self.calls += 1
        lo, hi = int(start[:10].replace("-", "")), min(int(end[:10].replace("-", "")), self.today)
        for sym in symbols:
            g = self.by_sym.get(sym)
            if g is None:
                continue
            ac = g["ac"].to_numpy().copy()
            if adjustment != "raw":
                for (s, d) in self.late_splits:
                    if s == sym and d <= self.today:
                        i = int(np.searchsorted(g["d"].to_numpy(), d))
                        raw_move = g["c"].iloc[i] / g["c"].iloc[i - 1]
                        ac[:i] *= (ac[i] / ac[i - 1]) / raw_move      # as if the split had not been applied
            for i, r in enumerate(g.itertuples()):
                if lo <= r.d <= hi:
                    c = r.c if adjustment == "raw" else float(ac[i])
                    k = c / r.c
                    yield sym, {"t": f"{_iso(int(r.d))}T05:00:00Z", "o": r.o * k, "h": r.h * k, "l": r.l * k, "c": c,
                                "v": r.v, "n": int(r.n), "vw": r.vw * k}

    def corporate_actions(self, symbols, start, end, types=""):
        self.calls += 1
        rows = [{"symbol": s, "ex_date": _iso(d), "new_rate": n, "old_rate": o}
                for (s, d), (n, o) in self.splits.items() if s in symbols and start <= _iso(d) <= end]
        return {"forward_splits": rows} if rows else {}

    def news(self, start, end, symbols=None, include_content=False, max_items=5000):
        return iter(())
