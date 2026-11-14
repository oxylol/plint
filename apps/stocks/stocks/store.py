"""The record: what each model decided, what it predicted, what the agent spent. SQLite in DATA_DIR.

Books are not stored. A book is the replay of a model's decisions over the price table (book.py), so a
late price correction or a bug fix reaches every figure on the page without a second copy to repair."""
from __future__ import annotations

import io
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .types import Decision, Order, Signal

SCHEMA = """
create table if not exists kv (k text primary key, v text);
create table if not exists decisions (
  model text, date text, orders text, note text, meta text, made_at text,
  primary key (model, date));
create table if not exists scores (            -- a model's score for every stock that evening
  model text, date text, data blob, primary key (model, date));
create table if not exists ic (                -- settled: rank correlation of those scores with what followed
  model text, date text, horizon integer, ic real, n integer, primary key (model, date, horizon));
create table if not exists signals (           -- forecasts without a book: the agent's, the event models'
  model text, date text, symbol text, horizon integer, p real, note text,
  excess real, settled_on text, primary key (model, date, symbol, horizon));
create table if not exists spend (
  id integer primary key, ts text, day text, month text, stage text, model text,
  in_tok integer, out_tok integer, cache_read integer, cache_write integer, searches integer, usd real);
create table if not exists memos (model text, date text, memo text, primary key (model, date));
create table if not exists reads (date text, symbol text, payload text, primary key (date, symbol));
create table if not exists seen (id text primary key, at text);   -- events already read by the agent
create table if not exists log (id integer primary key, ts text, level text, text text);
create table if not exists since (model text primary key, date text);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Store:
    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.dir / "stocks.db", check_same_thread=False, isolation_level=None)
        self.db.execute("pragma journal_mode=wal")
        self.db.execute("pragma synchronous=normal")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self.db.close()

    def _q(self, sql: str, args: tuple = ()) -> list[tuple]:
        with self._lock:
            return self.db.execute(sql, args).fetchall()

    def _x(self, sql: str, args: tuple = ()) -> None:
        with self._lock:
            self.db.execute(sql, args)

    # ── small values ────────────────────────────────────────────────────────────────────────────
    def get(self, key: str, default=None):
        r = self._q("select v from kv where k=?", (key,))
        return json.loads(r[0][0]) if r else default

    def put(self, key: str, value) -> None:
        self._x("insert into kv (k, v) values (?, ?) on conflict(k) do update set v=excluded.v",
                (key, json.dumps(value)))

    def log(self, text: str, level: str = "info") -> None:
        self._x("insert into log (ts, level, text) values (?, ?, ?)", (now_iso(), level, text))
        self._x("delete from log where id < (select max(id) from log) - 2000")

    def recent_log(self, n: int = 40) -> list[dict]:
        return [{"t": t, "level": lv, "text": tx}
                for t, lv, tx in self._q("select ts, level, text from log order by id desc limit ?", (n,))]

    # ── decisions ───────────────────────────────────────────────────────────────────────────────
    def save_decision(self, d: Decision, overwrite: bool = False) -> bool:
        """Returns False when that model already decided that evening (the first decision stands)."""
        row = (d.model, d.date, json.dumps([o.to_json() for o in d.orders]), d.note, json.dumps(d.meta), now_iso())
        with self._lock:
            if overwrite:
                self.db.execute("insert or replace into decisions values (?, ?, ?, ?, ?, ?)", row)
            else:
                cur = self.db.execute("insert or ignore into decisions values (?, ?, ?, ?, ?, ?)", row)
                if cur.rowcount == 0:
                    return False
            if d.scores:
                self.save_scores(d.model, d.date, d.scores)
        return True

    def decisions(self, model: str, since: str | None = None) -> list[Decision]:
        rows = self._q("select date, orders, note, meta from decisions where model=? and date>=? order by date",
                       (model, since or ""))
        return [Decision(model, date, [Order(**o) for o in json.loads(orders)], note, None, json.loads(meta or "{}"))
                for date, orders, note, meta in rows]

    def last_decision_date(self, model: str) -> str | None:
        r = self._q("select max(date) from decisions where model=?", (model,))
        return r[0][0] if r else None

    def since(self, model: str) -> str | None:
        """The day a model's record starts: after a rule change it starts over (`reset`)."""
        r = self._q("select date from since where model=?", (model,))
        return r[0][0] if r else None

    def set_since(self, model: str, date: str) -> None:
        self._x("insert into since values (?, ?) on conflict(model) do update set date=excluded.date", (model, date))

    def reset(self, model: str) -> None:
        """Start a model's record over: its decisions, scores and forecasts go; the spend log stays."""
        with self._lock:
            for t in ("decisions", "scores", "ic", "signals", "memos", "since"):
                self.db.execute(f"delete from {t} where model=?", (model,))

    # ── the ledger: scores and their settlement ─────────────────────────────────────────────────
    def save_scores(self, model: str, date: str, scores: dict[str, float]) -> None:
        syms = sorted(scores)
        buf = io.BytesIO()
        np.savez_compressed(buf, s=np.array(syms), v=np.array([scores[s] for s in syms], dtype=np.float32))
        self._x("insert or replace into scores values (?, ?, ?)", (model, date, buf.getvalue()))

    def load_scores(self, model: str, date: str) -> dict[str, float] | None:
        r = self._q("select data from scores where model=? and date=?", (model, date))
        if not r:
            return None
        z = np.load(io.BytesIO(r[0][0]))
        return dict(zip(z["s"].tolist(), z["v"].tolist()))

    def score_dates(self, model: str) -> list[str]:
        return [d for (d,) in self._q("select date from scores where model=? order by date", (model,))]

    def save_ic(self, model: str, date: str, horizon: int, ic: float, n: int) -> None:
        self._x("insert or replace into ic values (?, ?, ?, ?, ?)", (model, date, horizon, ic, n))

    def ic_series(self, model: str, horizon: int) -> list[tuple[str, float, int]]:
        return self._q("select date, ic, n from ic where model=? and horizon=? order by date", (model, horizon))

    def save_signal(self, s: Signal) -> None:
        self._x("insert or ignore into signals (model, date, symbol, horizon, p, note) values (?, ?, ?, ?, ?, ?)",
                (s.model, s.date, s.symbol, s.horizon, s.p, s.note))

    def open_signals(self) -> list[tuple]:
        return self._q("select model, date, symbol, horizon, p from signals where settled_on is null order by date")

    def settle_signal(self, model: str, date: str, symbol: str, horizon: int, excess: float, on: str) -> None:
        self._x("update signals set excess=?, settled_on=? where model=? and date=? and symbol=? and horizon=?",
                (excess, on, model, date, symbol, horizon))

    def signals(self, model: str, settled: bool | None = None, limit: int = 100000) -> list[dict]:
        cond = "" if settled is None else (" and settled_on is not null" if settled else " and settled_on is null")
        rows = self._q("select date, symbol, horizon, p, note, excess, settled_on from signals where model=?"
                       + cond + " order by date desc, symbol limit ?", (model, limit))
        return [dict(zip(("date", "symbol", "horizon", "p", "note", "excess", "settled_on"), r)) for r in rows]

    # ── the agent: spend, memos, reads ──────────────────────────────────────────────────────────
    def add_spend(self, *, stage: str, model: str, in_tok: int = 0, out_tok: int = 0, cache_read: int = 0,
                  cache_write: int = 0, searches: int = 0, usd: float, ts: str | None = None) -> None:
        ts = ts or now_iso()
        self._x("insert into spend (ts, day, month, stage, model, in_tok, out_tok, cache_read, cache_write, searches, usd)"
                " values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ts, ts[:10], ts[:7], stage, model, in_tok, out_tok, cache_read, cache_write, searches, usd))

    def spend_month(self, month: str) -> float:
        return float(self._q("select coalesce(sum(usd), 0) from spend where month=?", (month,))[0][0])

    def spend_day(self, day: str) -> float:
        return float(self._q("select coalesce(sum(usd), 0) from spend where day=?", (day,))[0][0])

    def spend_summary(self, month: str) -> dict:
        calls, usd = self._q("select count(*), coalesce(sum(usd), 0) from spend where month=?", (month,))[0]
        by_stage = {s: round(u, 4) for s, u in
                    self._q("select stage, sum(usd) from spend where month=? group by stage", (month,))}
        by_day = [[d, round(u, 4)] for d, u in
                  self._q("select day, sum(usd) from spend where month=? group by day order by day", (month,))]
        return {"month": month, "used": round(float(usd), 4), "calls": int(calls), "by_stage": by_stage, "by_day": by_day}

    def save_memo(self, model: str, date: str, memo: str) -> None:
        self._x("insert or replace into memos values (?, ?, ?)", (model, date, memo))

    def memo(self, model: str, date: str | None = None) -> tuple[str, str] | None:
        r = (self._q("select date, memo from memos where model=? and date=?", (model, date)) if date else
             self._q("select date, memo from memos where model=? order by date desc limit 1", (model,)))
        return (r[0][0], r[0][1]) if r else None

    def save_read(self, date: str, symbol: str, payload: dict) -> None:
        self._x("insert or replace into reads values (?, ?, ?)", (date, symbol, json.dumps(payload)))

    def reads(self, date: str | None = None, symbol: str | None = None, limit: int = 200) -> list[dict]:
        sql, args = "select date, symbol, payload from reads where 1=1", []
        if date:
            sql += " and date=?"; args.append(date)
        if symbol:
            sql += " and symbol=?"; args.append(symbol)
        rows = self._q(sql + " order by date desc, symbol limit ?", (*args, limit))
        return [{"date": d, "symbol": s, **json.loads(p)} for d, s, p in rows]

    def mark_seen(self, ids: list[str]) -> None:
        with self._lock:
            self.db.executemany("insert or ignore into seen values (?, ?)", [(i, now_iso()) for i in ids])

    def unseen(self, ids: list[str]) -> set[str]:
        out = set(ids)
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            got = self._q(f"select id from seen where id in ({','.join('?' * len(chunk))})", tuple(chunk))
            out -= {g for (g,) in got}
        return out

    # ── what the page reads ─────────────────────────────────────────────────────────────────────
    def write_json(self, name: str, obj) -> None:
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False), encoding="utf-8")
        tmp.replace(path)

    def read_json(self, name: str, default=None):
        try:
            return json.loads((self.dir / name).read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return default
