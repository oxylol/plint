"""Tonight's view of the market for the desk, built from the same panel the other models read.

Everything here stops at the panel's last row, which is tonight's close. Filings come from the events
tables (SEC, when a contact is set), headlines from Alpaca's news feed (when the plan allows it)."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

import numpy as np

from ..bars import d2s
from ..cal import NY, days_in_month, days_left_in_month
from ..features import Base, lag, rmean, rstd, rsum
from ..panel import Panel
from ..types import Event
from ..util import quiet

log = logging.getLogger("stocks.agent.data")
SPANS = (1, 5, 21, 63, 252)


class LiveDeskData:
    def __init__(self, panel: Panel, index_symbol: str, names: dict[str, str], ranks: dict[str, int],
                 events=None, sec=None, alpaca=None, exchanges: dict[str, str] | None = None):
        self.panel, self.index_symbol, self.names, self.ranks = panel, index_symbol, names, ranks
        self._events, self._sec, self._alpaca = events, sec, alpaca
        self._exchanges = exchanges or {}
        t = panel.T - 1
        self.date = d2s(panel.days[t])
        d = int(panel.days[t])
        self.days_left = days_left_in_month(date(d // 10000, d // 100 % 100, d % 100))
        self.days_in_month = days_in_month(date(d // 10000, d // 100 % 100, d % 100))
        cols = panel.members(t)
        self.universe = [panel.syms[j] for j in cols if np.isfinite(panel.c[t, j])]
        b = Base(panel, index_symbol)
        self._b = b
        with quiet():
            self._ret = {k: np.expm1(rsum(b.lr, k))[t] for k in SPANS}
            self._vol = rstd(b.r, 21)[t]
            sig = lag(rstd(b.r, 63), 1)[t]
            self._rel = b.r[t] - b.mkt[t]
            self._sigmas = self._rel / sig
            self._surge = (panel.v / lag(rmean(panel.v, 21), 1))[t]
        self._idx = panel.ix[index_symbol]
        self._since_earn, self._to_earn = self._earnings_distance()

    def _earnings_distance(self) -> tuple[dict[str, int], dict[str, int]]:
        """Trading days since each stock's last results, and until the next, guessed as a quarter after the last."""
        since: dict[str, int] = {}
        to: dict[str, int] = {}
        if self._events is None or self._events.earnings.empty:
            return since, to
        hit = self._events.earnings_table(self.panel)
        T = self.panel.T
        for j in np.flatnonzero(hit.any(axis=0)):
            last = int(np.flatnonzero(hit[:, j])[-1])
            sym = self.panel.syms[j]
            since[sym] = T - 1 - last
            to[sym] = max(0, 63 - since[sym])
        return since, to

    # ── the Protocol ────────────────────────────────────────────────────────────────────────────
    def profile(self, symbol: str) -> dict:
        p, t = self.panel, self.panel.T - 1
        j = p.ix.get(symbol)
        if j is None:
            return {"symbol": symbol, "name": self.names.get(symbol, symbol)}
        f = lambda x: None if not np.isfinite(x) else round(float(x), 4)
        return {
            "symbol": symbol, "name": self.names.get(symbol, symbol), "exchange": self._exchanges.get(symbol),
            "price": f(p.c[t, j]),
            "ret": {str(k): f(self._ret[k][j]) for k in SPANS},
            "vs_index": {str(k): f(self._ret[k][j] - self._ret[k][self._idx]) for k in SPANS},
            "vol21": f(self._vol[j]), "dollar_volume_rank": self.ranks.get(symbol),
            "days_since_earnings": self._since_earn.get(symbol), "days_to_earnings": self._to_earn.get(symbol),
        }

    def history(self, symbol: str, days: int = 120) -> list[dict]:
        p = self.panel
        j = p.ix.get(symbol)
        if j is None:
            return []
        out = []
        for t in range(max(0, p.T - days), p.T):
            if np.isfinite(p.c[t, j]):
                out.append({"d": d2s(p.days[t]), "o": round(float(p.o[t, j]), 2), "h": round(float(p.h[t, j]), 2),
                            "l": round(float(p.l[t, j]), 2), "c": round(float(p.c[t, j]), 2), "v": int(p.v[t, j])})
        return out

    def movers(self, n: int = 40) -> list[dict]:
        p, t = self.panel, self.panel.T - 1
        cols = np.array([p.ix[s] for s in self.universe], dtype=int)
        if not len(cols):
            return []
        s = np.abs(np.nan_to_num(self._sigmas[cols]))
        out = []
        for j in cols[np.argsort(-s)][:n]:
            if not np.isfinite(self._sigmas[j]):
                continue
            sym = p.syms[j]
            out.append({"symbol": sym, "name": self.names.get(sym, sym), "ret": round(float(p.ret[t, j]), 4),
                        "vs_index": round(float(self._rel[j]), 4), "sigmas": round(float(self._sigmas[j]), 2),
                        "volume_ratio": round(float(self._surge[j]), 2) if np.isfinite(self._surge[j]) else None})
        return out

    def market(self) -> dict:
        p, t, b = self.panel, self.panel.T - 1, self._b
        member = p.member[t] if p.member is not None else np.isfinite(p.c[t])
        with quiet():
            above = (b.P > rmean(b.P, 50))[t]
            breadth = float(np.nanmean(np.where(member & np.isfinite(b.P[t]), above, np.nan)))
            disp = float(np.nanstd(np.where(member, b.r[t], np.nan)))
        f = lambda x: None if not np.isfinite(x) else round(float(x), 4)
        return {"index": {"symbol": self.index_symbol, "ret": {str(k): f(self._ret[k][self._idx]) for k in SPANS},
                          "vol21": f(self._vol[self._idx])},
                "breadth_50d": f(breadth), "dispersion": f(disp)}

    def events(self, since: str) -> list[Event]:
        out: list[Event] = []
        uni = set(self.universe)
        if self._events is not None:
            try:
                out += self._events.recent(_to_ny(since), uni)
            except Exception as e:                       # the desk still has the movers and the news
                log.warning("filings unavailable: %s", e)
        if self._alpaca is not None:
            try:
                end = datetime.now(timezone.utc) - timedelta(minutes=16)
                for a in self._alpaca.news(since, end.strftime("%Y-%m-%dT%H:%M:%SZ"), include_content=False, max_items=4000):
                    for sym in a.get("symbols") or []:
                        if sym in uni:
                            out.append(Event(id=f"news-{a['id']}-{sym}", kind="news", symbol=sym, at=a.get("created_at", ""),
                                             title=a.get("headline", ""), url=a.get("url"),
                                             text=(a.get("summary") or None), facts={"source": a.get("source", "")}))
            except Exception as e:
                log.warning("news unavailable: %s", e)
        return sorted(out, key=lambda e: e.at)

    def fetch(self, url: str, max_chars: int = 60_000) -> str:
        if self._sec is None:
            raise ValueError("filings are switched off (no SEC contact set)")
        return self._sec.text(url, max_chars)


def _to_ny(iso: str) -> str:
    """The events tables keep New York time; the desk's clock is UTC."""
    try:
        t = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return iso
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(NY).strftime("%Y-%m-%dT%H:%M:%S")
