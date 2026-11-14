"""Alpaca's REST API, the parts used here: daily bars, the asset list, corporate actions, news.

Shapes are from docs.alpaca.markets/us/reference (read 2026-11-14). The free plan gives the full
consolidated tape for history as long as the window ends at least 15 minutes ago, 200 calls a minute.
Nothing here places an order: there is no trading call in this file, on purpose."""
from __future__ import annotations

import gzip
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterator

log = logging.getLogger("stocks.alpaca")

DATA = "https://data.alpaca.markets"
PAPER = "https://paper-api.alpaca.markets"     # read-only use: the asset list and the calendar
HELD_BACK = timedelta(minutes=16)              # free data may not be newer than 15 minutes


class AlpacaError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"Alpaca answered {status}: {message}")
        self.status = status


def _http(url: str, headers: dict, timeout: float = 60.0) -> tuple[int, dict, bytes]:
    req = urllib.request.Request(url, headers={**headers, "Accept-Encoding": "gzip"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body, hdrs, status = r.read(), dict(r.headers), r.status
    except urllib.error.HTTPError as e:
        body, hdrs, status = e.read(), dict(e.headers), e.code
    if hdrs.get("Content-Encoding", "").lower() == "gzip":
        body = gzip.decompress(body)
    return status, hdrs, body


class Alpaca:
    def __init__(self, key: str, secret: str, fetch: Callable = _http, sleep: Callable = time.sleep,
                 now: Callable = lambda: datetime.now(timezone.utc), per_minute: int = 180):
        self._headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret, "Accept": "application/json"}
        self._fetch, self._sleep, self._now = fetch, sleep, now
        self._gap = 60.0 / per_minute
        self._last = 0.0
        self.calls = 0

    def _get(self, base: str, path: str, params: dict | None = None):
        q = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v not in (None, "")})
        url = f"{base}{path}" + (f"?{q}" if q else "")
        for attempt in range(6):
            wait = self._last + self._gap - time.monotonic()
            if wait > 0:
                self._sleep(wait)
            self._last = time.monotonic()
            self.calls += 1
            try:
                status, hdrs, body = self._fetch(url, self._headers)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                log.warning("alpaca: %s (try %d)", e, attempt + 1)
                self._sleep(min(60, 2 ** attempt))
                continue
            if status == 200:
                return json.loads(body)
            if status == 429 or status >= 500:
                reset = hdrs.get("X-RateLimit-Reset") or hdrs.get("x-ratelimit-reset")
                pause = max(1.0, float(reset) - time.time()) if status == 429 and reset else 2 ** attempt
                log.warning("alpaca: %d, waiting %.0f s", status, min(pause, 65))
                self._sleep(min(pause, 65))
                continue
            try:
                message = json.loads(body).get("message", "")
            except (ValueError, AttributeError):
                message = body[:200].decode("utf-8", "replace")
            raise AlpacaError(status, message)
        raise AlpacaError(0, f"no answer after 6 tries: {path}")

    # ── reference ───────────────────────────────────────────────────────────────────────────────
    def assets(self, status: str | None = None) -> list[dict]:
        """Every US equity Alpaca knows, listed and delisted: {symbol, name, exchange, status, tradable, …}."""
        return self._get(PAPER, "/v2/assets", {"asset_class": "us_equity", "status": status})

    def calendar(self, start: str, end: str) -> list[dict]:
        return self._get(PAPER, "/v2/calendar", {"start": start, "end": end})

    # ── prices ──────────────────────────────────────────────────────────────────────────────────
    def _end(self, end: str) -> str:
        """A window may not reach into the last 15 minutes: an end date of today or later becomes a time."""
        limit = self._now() - HELD_BACK
        if end >= limit.strftime("%Y-%m-%d"):
            return limit.strftime("%Y-%m-%dT%H:%M:%SZ")
        return end

    def bars(self, symbols: list[str], start: str, end: str, adjustment: str = "raw") -> Iterator[tuple[str, dict]]:
        """Daily bars for the symbols between two dates (inclusive), as (symbol, bar) with bar fields
        t, o, h, l, c, v, n, vw. `adjustment` is "raw" (what traded) or "all" (splits and dividends applied)."""
        params = {"symbols": ",".join(symbols), "timeframe": "1Day", "start": start, "end": self._end(end),
                  "limit": 10000, "adjustment": adjustment, "feed": "sip", "sort": "asc"}
        while True:
            page = self._get(DATA, "/v2/stocks/bars", params)
            for sym, rows in (page.get("bars") or {}).items():
                for bar in rows:
                    yield sym, bar
            token = page.get("next_page_token")
            if not token:
                return
            params["page_token"] = token

    def corporate_actions(self, symbols: list[str], start: str, end: str,
                          types: str = "forward_split,reverse_split,cash_dividend,stock_dividend") -> dict[str, list]:
        out: dict[str, list] = {}
        params = {"symbols": ",".join(symbols), "start": start, "end": end, "types": types, "limit": 1000}
        while True:
            page = self._get(DATA, "/v1/corporate-actions", params)
            for kind, rows in (page.get("corporate_actions") or {}).items():
                out.setdefault(kind, []).extend(rows)
            token = page.get("next_page_token")
            if not token:
                return out
            params["page_token"] = token

    # ── news ────────────────────────────────────────────────────────────────────────────────────
    def news(self, start: str, end: str, symbols: list[str] | None = None, include_content: bool = False,
             max_items: int = 5000) -> Iterator[dict]:
        """Articles between two times, oldest first: {id, headline, summary, content, created_at, url, symbols, source}."""
        params = {"start": start, "end": end, "sort": "asc", "limit": 50,
                  "include_content": "true" if include_content else "false",
                  "symbols": ",".join(symbols) if symbols else None}
        n = 0
        while True:
            page = self._get(DATA, "/v1beta1/news", params)
            for item in page.get("news") or []:
                yield item
                n += 1
                if n >= max_items:
                    return
            token = page.get("next_page_token")
            if not token:
                return
            params["page_token"] = token
