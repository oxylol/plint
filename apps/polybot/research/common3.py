"""The research side's pure pieces, shared by collect3.py and build3.py and importable without numpy, so the bot's
test suite can prove that research and bot compute the same model inputs (tests/test_features.py)."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from polybot import features as F  # noqa: E402
from polybot.categories import category  # noqa: E402

U = dt.timezone.utc
LOOKBACK = 48 * 3600     # the bot's lookback for the learned setups
MIN_USD = 10.0           # their min_leader_usd


def ts(s):
    if not s:
        return None
    d = dt.datetime.fromisoformat(s)
    return (d if d.tzinfo else d.replace(tzinfo=U)).timestamp()


def pos_row(p) -> dict:
    return {"cid": str(p.condition_id), "asset": str(p.asset_id), "oi": p.outcome_index, "outcome": p.outcome,
            "avg": float(p.avg_price or 0), "size": float(p.total_size or 0), "cur_size": float(p.current_size or 0),
            "cur": float(p.current_price or 0), "rpnl": float(p.realized_pnl or 0), "tpnl": float(p.total_pnl or 0),
            "first": p.first_entry_at.isoformat() if p.first_entry_at else None,
            "last": p.last_event_at.isoformat() if p.last_event_at else None,
            "end": p.end_date.isoformat() if p.end_date else None, "nr": bool(p.negative_risk),
            "title": (p.title or "")[:90], "slug": p.slug, "ev": p.event_slug}


class M:
    """The market fields the builder needs, compact (hundreds of thousands of these live in every worker)."""
    __slots__ = ("closed", "prices", "resolved", "end", "nr", "ev", "fee", "game_start")

    def __init__(self, m):
        self.closed = bool(m.get("closed"))
        self.prices = tuple(m.get("prices") or (None, None))
        t = ts(m.get("closed_time"))
        if t is None and m.get("end"):
            t = ts(m.get("end")) + 3600
        self.resolved = t if self.closed else None
        self.end = ts(m.get("end"))
        self.nr = bool(m.get("nr"))
        self.ev = m.get("ev")
        self.fee = float(m.get("fee_rate") or 0.0) if m.get("fee_on") else 0.0
        self.game_start = ts(m.get("game_start"))


def market_final(m, oi):
    if m is None or oi not in (0, 1) or not m.closed:
        return None
    return F.final_from_price(m.prices[oi])


def market_resolved_at(m):
    return m.resolved if m is not None else None


def positions_of(w, markets) -> list[F.Pos]:
    seen = set()
    out = []
    for key in ("closed", "redeemable", "open"):
        for r in w.get(key, []):
            if r["asset"] in seen:
                continue
            seen.add(r["asset"])
            m = markets.get(r["cid"])
            oi = r.get("oi")
            final = market_final(m, oi)
            if final is None and m is None:
                # no market data: a row's own price is the final price once its market has resolved
                end = ts(r.get("end"))
                if end is not None and end < ts(w["at"]) - 2 * 86400:
                    final = F.final_from_price(r.get("cur"))
                resolved = (end + 86400) if final is not None else None
            else:
                resolved = market_resolved_at(m) if final is not None else None
            out.append(F.Pos(cost=r["size"] * r["avg"], avg=r["avg"], final=final,
                             category=category(r.get("title"), r.get("ev") or r.get("slug")),
                             first=ts(r.get("first")), last=ts(r.get("last")), resolved=resolved,
                             is_no=str(r.get("outcome") or "").lower() == "no", neg_risk=bool(r.get("nr"))))
    return out


def signal_context(rec, fills, asset, td, m, title, slug, ev_slug, outcome, joined, taker, fill_price):
    """The model's non-price inputs for one signal, as the bot computes them (features.leader_stats + features.context).
    fills: the leader's (time, asset, side, size, price) in this market; m: the market (M) or None."""
    ls = F.leader_stats(fills, asset, td, LOOKBACK, MIN_USD)
    lpx = ls["leader_price"] if ls["leader_price"] is not None else fill_price
    cat = category(title, ev_slug or slug)
    end = m.end if m else None
    gs = m.game_start if m else None
    ctx = F.context(rec, leader_price=lpx, stance_usd=ls["stance"], hours_to_end=(end - td) / 3600 if end else None, category=cat,
                    is_no=str(outcome or "").lower() == "no", neg_risk=m.nr if m else False, n_fills=ls["n_fills"],
                    minutes_since_first=ls["minutes_since_first"], game_minutes=(td - gs) / 60 if gs else None,
                    age_days=(td - joined) / 86400 if joined else None, taker_share=taker)
    return ctx, ls, cat


def gz_lines(path, chunk=1 << 20):
    """Lines of an append-only .gz file (one gzip member per append), streamed, skipping a member damaged by an
    interrupted write: on a decompression error, resume at the next gzip header."""
    import zlib
    data = Path(path).read_bytes()          # compressed bytes only (tens of MB); the text is streamed
    n = len(data)
    pos = 0
    d = zlib.decompressobj(31)
    tail = b""
    while pos < n:
        piece = data[pos: pos + chunk]
        try:
            out = d.decompress(piece)
        except zlib.error:
            nxt = data.find(b"\x1f\x8b\x08", pos + 1)
            if nxt < 0:
                return
            pos, d, tail = nxt, zlib.decompressobj(31), b""
            continue
        if out:
            lines = (tail + out).split(b"\n")
            tail = lines.pop()
            for ln in lines:
                if ln:
                    yield ln.decode("utf-8", "replace")
        if d.eof:                            # end of a member: the rest of this piece starts the next one
            rest = d.unused_data
            pos += len(piece) - len(rest)
            d = zlib.decompressobj(31)
        else:
            pos += len(piece)
    if tail:
        yield tail.decode("utf-8", "replace")
