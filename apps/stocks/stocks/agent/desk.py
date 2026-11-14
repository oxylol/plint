"""The research desk: the one model in the lab that reads.

An evening has three stages. The small model screens every new filing, headline and unusual move. The
main model reads in depth the companies that screening flagged, and every holding that had news, and
states for each the chance that it beats the index over 5 and 20 trading days: those forecasts are how
the desk is judged long before its book can be. Then the main model sees the book and its own notes and
proposes orders, and the code here cuts them to the limits whatever it proposed.

Two things hold throughout. Nothing a document says reaches the book except through the main model's
decision, and a new name must have been read that evening, so a line in one company's filing cannot buy
another. And no call is made that the month's cap has no room for: see budget.py."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from ..config import Config
from ..store import Store, now_iso
from ..types import BookView, Event, Order, Signal
from . import prompts
from .api import DeskData, DeskLimits, DeskResult
from .budget import Budget, tokens, worst_case
from .llm import LLM, Ask, LLMError, Reply, make_client

MODEL = "agent"                  # the desk's name in the store
SINCE = "agent:since"            # kv: the latest time among the items the feed has returned
SEARCHES = "agent:searches"      # kv: {"date", "n"}: web searches made on that market date
PLAYBOOK, PLAYBOOK_PREV = "playbook.md", "playbook.prev.md"

TRIAGE_BATCH = 25                # items per screening request: far under the small model's 100,000-token price step
TRIAGE_EFFORT = "low"            # sorting headlines is the small model's easy case
READ_MAX_TOKENS = 16_000         # thinking counts against it; a read cut off mid-answer is money lost
DECIDE_MAX_TOKENS = 16_000
REVIEW_MAX_TOKENS = 12_000
DECIDE_GUESS_TOKENS = 15_000     # the decision's prompt, guessed, to keep its cost back while reading
SEARCHES_PER_READ = 2
SEARCH_SHARE = 0.6               # searches stop when this share of the day's allowance is gone; reading goes on to its end
DOC_CHARS = 24_000               # of one document
READ_CHARS = 48_000              # of all the documents in one read
ITEMS_PER_READ = 8
MAX_RESUMES = 2                  # continuations of a search turn the API paused
MAX_FAILS = 3                    # failed calls in a row before a stage stops asking
FIRST_LOOKBACK_H = 24
OVERLAP_H = 6                    # the feed is asked again for the hours before the last item seen: a filing can be indexed late
MAX_LOOKBACK_D = 5               # news older than this is not worth a call
PLAYBOOK_WORDS = 800             # the hard ceiling; the prompt asks for about 600
MEMO_CHARS = 4000
FATAL = ("auth", "permission")   # failures that every further call tonight would repeat
RANK = {"day allowance": 1, "error": 2, "month cap": 3, "refusal": 4}

_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
_SHORT = {"inc", "corp", "co", "ltd", "llc", "plc", "u.s", "vs", "no", "mr", "ms", "mrs", "dr", "st", "jr", "sr", "e.g", "i.e", "etc"}


@dataclass(frozen=True)
class Verdict:
    """The small model's view of one item. It chooses what gets read and is never turned into an order."""
    material: bool = False
    direction: int = 0
    confidence: float = 0.0
    why: str = ""


class _Cap(Exception):
    """The month's cap has no room for the call that was about to be made."""


class Desk:
    def __init__(self, cfg: Config, store: Store, client=None, limits: DeskLimits = DeskLimits()):
        self.cfg, self.store, self.limits = cfg, store, limits
        self._client = client       # a test's stand-in; otherwise made from the key on first use
        self._model: LLM | None = None
        self._begin("")

    # ── the evening ─────────────────────────────────────────────────────────────────────────────
    def run(self, data: DeskData, book: BookView) -> DeskResult:
        if not self._ready():
            return DeskResult(stopped="no key")
        res = DeskResult()
        budget = Budget(self.store, self.cfg.agent_cap_usd, data.days_left)
        self._begin(data.date)
        try:
            items = self._triage(data, budget, res)
            picks = self._pick(items, data, book)
            brief = prompts.brief(self.limits, data.index_symbol)
            night = prompts.tonight(data.date, data.index_symbol, self._market(data), self._playbook())
            cut = self._read_all(data, picks, budget, res, (brief + "\n\n" + prompts.READ_TASK, night))
            flagged = sum(1 for _, v in items if v and v.material)
            self._decide(data, book, budget, res, (brief + "\n\n" + prompts.decide_task(self.limits, data.index_symbol), night),
                         flagged, len(picks), cut)
        except LLMError as e:
            self._hold(data, res, "error", f"the API turned the desk away ({e.kind})")
        res.usd = round(budget.spent, 6)
        self.store.log(f"agent {data.date}: screened {res.triaged}, read {len(res.reads)}, {len(res.orders)} orders, "
                       f"${res.usd:.2f}" + (f", stopped: {res.stopped}" if res.stopped else ""))
        return res

    def weekly_review(self, data: DeskData, closed: list[dict], scored: list[dict]) -> str:
        path = self.store.dir / PLAYBOOK
        old = path.read_text(encoding="utf-8").strip() if path.exists() else ""
        if not self._ready() or not (closed or scored):
            return old
        self._begin(data.date)
        budget = Budget(self.store, self.cfg.agent_cap_usd, data.days_left)
        ask = Ask(model=self.cfg.agent_model, system=(prompts.REVIEW,), user=prompts.review_user(data.date, old, closed, scored),
                  schema=prompts.REVIEW_SCHEMA, max_tokens=REVIEW_MAX_TOKENS, effort=self.cfg.agent_effort, fallback=True)
        try:
            reply = self._exchange("review", ask, budget)
        except _Cap:
            self.store.log("agent: no weekly review, the month's budget is used up", "warn")
            return old
        except LLMError as e:
            self.store.log(f"agent: the weekly review failed ({e})"[:300], "warn")
            return old
        new = _cap_words(_text((reply.data or {}).get("playbook"), 20_000), PLAYBOOK_WORDS)
        if not new:
            self.store.log(f"agent: the weekly review gave no playbook ({reply.refusal or reply.stop})", "warn")
            return old
        if old:
            (self.store.dir / PLAYBOOK_PREV).write_text(old + "\n", encoding="utf-8")
        tmp = path.with_name(PLAYBOOK + ".tmp")
        tmp.write_text(new + "\n", encoding="utf-8")
        tmp.replace(path)
        self.store.log(f"agent: playbook rewritten from {len(closed)} closed trades and {len(scored)} settled forecasts, ${reply.usd:.2f}")
        return new

    # ── stage 1: screening ──────────────────────────────────────────────────────────────────────
    def _triage(self, data: DeskData, budget: Budget, res: DeskResult) -> list[tuple[Event, Verdict | None]]:
        """Every new item past the small model, in batches. Returns each item with its verdict, or None
        where screening could not be done; those stay unseen and come back tomorrow."""
        events, newest = self._fresh(data)
        system = (prompts.triage_system(data.date),)
        verdicts: dict[str, Verdict] = {}
        done = True
        for i in range(0, len(events), TRIAGE_BATCH):
            try:
                self._screen(events[i:i + TRIAGE_BATCH], system, budget, verdicts)
            except _Cap:
                done = False
                self._stop(res, "month cap")
                break
            except LLMError as e:
                if e.kind in FATAL:
                    raise
                done = False
                if self._trouble("screening", e):
                    self._stop(res, "error")
                    break
            finally:
                res.triaged = len(verdicts)
        last = _when(self.store.get(SINCE))
        if done and newest and (last is None or _when(newest) > last):
            self.store.put(SINCE, newest)       # only once everything up to it has been screened
        return [(e, verdicts.get(e.id)) for e in events]

    def _fresh(self, data: DeskData) -> tuple[list[Event], str | None]:
        """Tonight's unread items, oldest first, and the latest time among what the feed returned. That
        time, not the clock, is where the next evening starts: a release made between the close and
        this run is not in tonight's view, and must not fall behind the mark."""
        feed = list(data.events(_since(self.store.get(SINCE), now_iso())))
        newest = max((e.at for e in feed if _when(e.at)), key=_when, default=None)
        moves = [_move(data.date, m) for m in data.movers()]
        unseen = self.store.unseen([e.id for e in feed + moves])
        out, taken = [], set()
        for e in feed + moves:
            if e.id in unseen and e.id not in taken:
                taken.add(e.id)
                out.append(e)
        return out, newest

    def _screen(self, chunk: list[Event], system: tuple[str, ...], budget: Budget, out: dict[str, Verdict]) -> None:
        """One batch. A batch that is refused, cut off or unreadable is halved and asked again, so one bad
        item costs its neighbours a second request and nothing more."""
        ask = Ask(model=self.cfg.agent_small_model, system=system, user=prompts.triage_user(chunk), schema=prompts.TRIAGE_SCHEMA,
                  max_tokens=1500 + 100 * len(chunk), effort=TRIAGE_EFFORT, cache_system=True)
        reply = self._exchange("triage", ask, budget)
        self._fails = 0
        items = (reply.data or {}).get("items")
        if not isinstance(items, list):
            if len(chunk) > 1:
                mid = len(chunk) // 2
                self._screen(chunk[:mid], system, budget, out)
                self._screen(chunk[mid:], system, budget, out)
                return
            self.store.log(f"agent: no screening answer for {chunk[0].id} ({reply.refusal or reply.stop}); passed over", "warn")
            items = []
        out.update(_verdicts(chunk, items))
        self.store.mark_seen([e.id for e in chunk])

    def _pick(self, items: list[tuple[Event, Verdict | None]], data: DeskData, book: BookView) -> list[tuple[str, list]]:
        """Who is read, in the order they are read: every holding that had any item tonight, then the
        flagged companies the book could buy, surest first, up to the evening's number."""
        by: dict[str, list] = {}
        for e, v in items:
            by.setdefault(e.symbol, []).append((e, v))
        sure = {s: max((v.confidence for _, v in evs if v and v.material), default=-1.0) for s, evs in by.items()}
        universe = set(data.universe)
        held = [s for s in by if s in book.holdings and s != data.index_symbol]
        new = [s for s in by if s not in book.holdings and s in universe and s != data.index_symbol and sure[s] >= 0]
        order = sorted(held, key=lambda s: (-sure[s], s)) + sorted(new, key=lambda s: (-sure[s], s))
        return [(s, by[s]) for s in order[:max(0, self.cfg.agent_deep_reads)]]

    # ── stage 2: reading ────────────────────────────────────────────────────────────────────────
    def _read_all(self, data: DeskData, picks: list[tuple[str, list]], budget: Budget, res: DeskResult,
                  system: tuple[str, ...]) -> str:
        """Reads the picks in order until they are done or the money says stop. Returns a sentence about
        an early stop, for the decision and the memo, or "". The decision's own cost is kept back from
        the day's allowance, so reading gives way before deciding does."""
        reserve = worst_case(self.cfg.agent_model, DECIDE_GUESS_TOKENS, DECIDE_MAX_TOKENS)
        for symbol, evs in picks:
            why = ""
            if budget.tonight + reserve >= budget.allowance:
                self._stop(res, "day allowance")
                why = f"tonight's allowance of ${budget.allowance:.2f} was used up"
            else:
                try:
                    read = self._read(data, symbol, evs, budget, system)
                except _Cap:
                    self._stop(res, "month cap")
                    why = f"the month's budget of ${budget.cap:.2f} is used up"
                except LLMError as e:
                    if e.kind in FATAL:
                        raise
                    if not self._trouble(f"the read of {symbol}", e):
                        continue
                    self._stop(res, "error")
                    why = f"the API kept failing ({e.kind})"
                else:
                    if read:
                        res.reads.append(read)
                        res.signals += self._signals(data.date, read)
                    continue
            return f"Reading stopped early, after {len(res.reads)} of the {len(picks)} companies chosen: {why}."
        return ""

    def _read(self, data: DeskData, symbol: str, evs: list, budget: Budget, system: tuple[str, ...]) -> dict | None:
        try:
            profile, bars = data.profile(symbol), data.history(symbol)
        except Exception as e:      # the lab's own data is missing for this company: nothing to read against
            self.store.log(f"agent: no read of {symbol}, its profile or prices could not be had ({type(e).__name__})", "warn")
            return None
        docs, left_out = self._documents(data, evs)
        left = self.cfg.agent_searches - self._searched(data.date)
        searches = min(SEARCHES_PER_READ, left) if self._search_ok and left > 0 and budget.tonight < SEARCH_SHARE * budget.allowance else 0

        def ask(n: int) -> Ask:
            return Ask(model=self.cfg.agent_model, system=system, user=prompts.read_user(symbol, profile, bars, docs, n, left_out),
                       schema=prompts.READ_SCHEMA, max_tokens=READ_MAX_TOKENS, effort=self.cfg.agent_effort, fallback=True,
                       cache_system=True, cache_user=n > 0, searches=n)
        try:
            reply = self._exchange("read", ask(searches), budget)
        except _Cap:
            if not searches:
                raise
            searches, reply = 0, self._exchange("read", ask(0), budget)     # the search is the first thing to give up
        except LLMError as e:
            if not searches or e.kind != "bad request":
                raise
            self._no_search(f"the API would not take the request: {e}")
            searches, reply = 0, self._exchange("read", ask(0), budget)
        if searches and reply.stop == "end_turn" and reply.data is None:
            self._no_search(f"the read of {symbol} came back without the answer in its form")
            paid, reply = reply.usd, self._exchange("read", ask(0), budget)
            reply.usd += paid
        self._fails = 0
        known = {_url(e.url): e.title for e, _, _ in docs if _url(e.url)}
        known.update({_url(u): t for u, t in reply.urls.items() if _url(u)})
        read = _payload(symbol, reply.data, known, reply.usd) if reply.data else None
        if read is None:
            self.store.log(f"agent: no read of {symbol} ({reply.refusal or reply.stop})", "warn")
            return None
        self.store.save_read(data.date, symbol, read)
        return read

    def _no_search(self, why: str) -> None:
        """Gives up web search for the rest of the evening. Search and a JSON answer in one request is the
        one combination the desk could not try before it had a key; if it fails for one read it would
        fail, and be paid for, at every read after."""
        self._search_ok = False
        self.store.log(f"agent: reading without web search for the rest of tonight ({why})"[:300], "warn")

    def _documents(self, data: DeskData, evs: list) -> tuple[list[tuple[Event, str, str]], int]:
        """A company's items as (item, its text made safe to quote, the desk's note on that text), the
        ones screening thought most of first, and how many were left out. A filing's body is fetched
        here, where it will be read, not at screening. Long documents are cut, and the note says so."""
        ranked = sorted(evs, key=lambda ev: (not (ev[1] and ev[1].material), -(ev[1].confidence if ev[1] else 0.0),
                                             -(_when(ev[0].at) or _EPOCH).timestamp()))
        docs, room = [], READ_CHARS
        for e, _ in ranked[:ITEMS_PER_READ]:
            limit, text, note = min(DOC_CHARS, room), e.text or "", ""
            if limit <= 0:
                text, note = "", "" if e.kind == "move" else "left out to keep this read within its budget; only the title is shown"
            elif not text.strip() and e.url and _is_sec(e.url):
                try:
                    text = data.fetch(e.url, max_chars=limit) or ""
                    if len(text) >= limit:
                        note = f"only the first {limit:,} characters of this document are shown"
                except Exception as err:    # a bad address, or the SEC not answering: read on with the title
                    note = "the text of this document could not be fetched; only its title is known"
                    self._unfetched += 1
                    if self._unfetched <= 3:    # an evening the SEC is down would otherwise fill the log
                        self.store.log(f"agent: could not fetch {e.id} ({type(err).__name__})", "warn")
            text = prompts.clean(text)
            if len(text) > max(limit, 0):
                note = note or f"only the first {limit:,} of this document's {len(text):,} characters are shown"
                text = text[:limit]
            elif not text and not note and e.kind != "move":
                note = "no text came with this item; only its title is known"
            room -= len(text)
            docs.append((e, text, note))
        return docs, max(0, len(evs) - ITEMS_PER_READ)

    def _signals(self, date: str, read: dict) -> list[Signal]:
        note = _first_sentence(read["thesis"])
        out = [Signal(MODEL, date, read["symbol"], h, read[f"p{h}"], note) for h in (5, 20)]
        for s in out:
            self.store.save_signal(s)
        return out

    # ── stage 3: deciding ───────────────────────────────────────────────────────────────────────
    def _decide(self, data: DeskData, book: BookView, budget: Budget, res: DeskResult, system: tuple[str, ...],
                flagged: int, planned: int, cut: str) -> None:
        mine = {r["symbol"] for r in res.reads}     # with what an earlier attempt this evening already paid to read
        reads = res.reads + [r for r in self.store.reads(date=data.date) if r.get("symbol") not in mine]
        ask = Ask(model=self.cfg.agent_model, system=system, user=prompts.decide_user(book, reads, res.triaged, flagged, planned, cut),
                  schema=prompts.DECIDE_SCHEMA, max_tokens=DECIDE_MAX_TOKENS, effort=self.cfg.agent_effort, fallback=True)
        try:
            reply = self._exchange("decide", ask, budget)
        except _Cap:
            return self._hold(data, res, "month cap", f"this month's budget for the model is used up (${budget.month:.2f} of "
                                                      f"${budget.cap:.2f}), so tonight's decision could not be made")
        except LLMError as e:
            self.store.log(f"agent: the decision failed ({e})"[:300], "warn")
            return self._hold(data, res, "error", f"the API did not answer tonight's decision ({e.kind})")
        if reply.stop == "refusal":
            return self._hold(data, res, "refusal", "the model declined to make tonight's decision, and so did the fallback "
                                                    f"model where one was tried (refusal: {reply.refusal})")
        if reply.data is None:
            return self._hold(data, res, "error", f"tonight's decision came back unusable ({reply.stop})")
        proposed = reply.data.get("orders")
        res.orders, notes = enforce_limits(proposed if isinstance(proposed, list) else [], book, set(data.universe),
                                           data.index_symbol, self.limits, {r.get("symbol") for r in reads})
        memo = _text(reply.data.get("memo"), MEMO_CHARS) or "No memo came with tonight's decision."
        if notes:
            memo += "\n\nLimits applied by the desk's code: " + "; ".join(notes) + "."
        if cut:
            memo += "\n\n" + cut
        res.memo = memo
        self.store.save_memo(MODEL, data.date, memo)

    def _hold(self, data: DeskData, res: DeskResult, reason: str, why: str) -> None:
        """No orders: every position is left as it is, and the memo says why in plain words."""
        self._stop(res, reason)
        res.orders = []
        n = len(res.reads)
        res.memo = f"**Held everything tonight.** Nothing was traded: {why}."
        if n:
            res.memo += f" {n} compan{'y was' if n == 1 else 'ies were'} read before that, and the forecasts are on record."
        self.store.save_memo(MODEL, data.date, res.memo)

    # ── the plumbing every stage shares ─────────────────────────────────────────────────────────
    def _exchange(self, stage: str, ask: Ask, budget: Budget) -> Reply:
        """One question, gated before and booked after. Raises _Cap rather than make a call the month
        has no room for. A search turn the API paused is carried on with the same request plus what
        came back, and each continuation is gated and booked as the call it is."""
        usd, urls = 0.0, {}
        for _ in range(1 + MAX_RESUMES):
            if not budget.fits(worst_case(ask.model, _size(ask), ask.max_tokens, ask.searches)):
                raise _Cap()
            reply = self._llm().call(ask)
            cost = budget.record(stage, reply.top, reply.parts)
            usd += cost
            urls.update(reply.urls)
            if reply.searches:
                self._searched(self._date, reply.searches)
            self._keep(stage, ask, reply, cost)
            if reply.stop != "pause_turn":
                break
            ask = replace(ask, paused=ask.paused + (reply.content,), container=reply.container or ask.container)
        reply.usd, reply.urls = usd, urls
        return reply

    def _keep(self, stage: str, ask: Ask, reply: Reply, cost: float) -> None:
        """Every prompt and reply, a line each, in one gzipped file per evening: what the desk was shown
        and what it said, for the day a trade needs explaining. A system prompt is written once and
        named by its hash after that."""
        mark = hashlib.sha1("\x1e".join(ask.system).encode()).hexdigest()[:12]
        lines = [] if mark in self._kept else [{"system": mark, "text": list(ask.system)}]
        lines.append({"t": now_iso(), "stage": stage, "model": ask.model, "served": reply.top.model, "system": mark, "user": ask.user,
                      "searches": reply.searches, "resumed": len(ask.paused), "stop": reply.stop, "refusal": reply.refusal,
                      "answer": reply.data, "text": None if reply.data is not None else _said(reply.content),
                      "usd": round(cost, 6)})
        try:
            path = self.store.dir / MODEL / f"{self._date}.jsonl.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "at", encoding="utf-8") as f:
                f.writelines(json.dumps(line, default=str) + "\n" for line in lines)
            self._kept.add(mark)
        except OSError as e:
            if "!" not in self._kept:       # said once an evening, not once a call
                self._kept.add("!")
                self.store.log(f"agent: cannot write the evening's transcript ({type(e).__name__})", "warn")

    def _searched(self, date: str, add: int = 0) -> int:
        """Web searches made on this market date. Kept in the store, so a second run the same evening
        draws on the same day's number."""
        kept = self.store.get(SEARCHES) or {}
        n = (int(kept.get("n", 0)) if kept.get("date") == date else 0) + add
        if add:
            self.store.put(SEARCHES, {"date": date, "n": n})
        return n

    def _trouble(self, what: str, e: LLMError) -> bool:
        """Notes a call that failed. True when the stage should stop asking: the same thing has failed
        three times running, or the model does not exist for this key."""
        self._fails += 1
        self.store.log(f"agent: {what} failed ({e})"[:300], "warn")
        return self._fails >= MAX_FAILS or e.kind == "not found"

    def _stop(self, res: DeskResult, reason: str) -> None:
        if RANK[reason] > RANK.get(res.stopped or "", 0):
            res.stopped = reason

    def _begin(self, date: str) -> None:
        self._date, self._fails, self._unfetched, self._search_ok, self._kept = date, 0, 0, True, set()

    def _ready(self) -> bool:
        return self._client is not None or bool(self.cfg.anthropic_key)

    def _llm(self) -> LLM:
        if self._model is None:
            self._model = LLM(self._client or make_client(self.cfg.anthropic_key))
        return self._model

    def _market(self, data: DeskData) -> dict:
        try:
            return data.market()
        except Exception as e:      # the evening can be read without the market's summary
            self.store.log(f"agent: no market summary tonight ({type(e).__name__})", "warn")
            return {}

    def _playbook(self) -> str:
        path = self.store.dir / PLAYBOOK
        return path.read_text(encoding="utf-8") if path.exists() else ""


# ── the limits ──────────────────────────────────────────────────────────────────────────────────
def enforce_limits(proposed: list, book: BookView, universe: set[str], index: str, limits: DeskLimits,
                   read: set[str]) -> tuple[list[Order], list[str]]:
    """What the model asked for, cut to what the book allows, and a plain note for each cut.

    The index fund, when the book may use it, is where idle money waits and not a bet: it needs no read,
    is not one of the names and may be larger than one position; only the ceiling on the whole book
    binds it. Everything else is as DeskLimits says. Weights are rounded down, so a cut never rounds
    back over its limit."""
    held = {s: h.weight for s, h in book.holdings.items()}
    notes: list[str] = []
    want: dict[str, dict] = {}

    def parked(sym: str) -> bool:
        return limits.allow_index and sym == index

    for p in proposed:
        if not isinstance(p, dict):
            continue
        sym = str(p.get("symbol") or "").strip().upper()
        if not _SYMBOL.match(sym):
            notes.append("an order with an unreadable symbol was dropped")
            continue
        if sym in want:
            continue
        w = _number(p.get("weight"))
        if w is None:
            notes.append(f"{sym} dropped (no usable weight)")
            continue
        if w <= 0:
            if sym in held:
                want[sym] = {"w": 0.0, "c": 0.0, "why": _line(p.get("why"), 160)}
            continue
        if not parked(sym) and sym not in universe and w > held.get(sym, 0.0):
            notes.append(f"{sym} dropped (not in the lab's universe)")
            continue
        if not parked(sym) and w > limits.max_weight:
            notes.append(f"{sym} cut from {w:.0%} to {limits.max_weight:.0%} (the most one position may be)")
            w = limits.max_weight
        if w < limits.min_weight:
            notes.append(f"{sym} dropped ({w:.1%} is under the {limits.min_weight:.0%} minimum)")
            continue
        want[sym] = {"w": w, "c": _unit(p.get("conviction")), "why": _line(p.get("why"), 160)}

    new = [s for s, o in want.items() if s not in held and o["w"] > 0 and not parked(s)]
    if limits.new_needs_read:
        for s in [s for s in new if s not in read]:
            notes.append(f"{s} dropped (a new name must have been read tonight)")
            del want[s]
        new = [s for s in new if s in want]
    new.sort(key=lambda s: (-want[s]["c"], -want[s]["w"], s))
    for s in new[limits.max_new_per_day:]:
        notes.append(f"{s} dropped (at most {limits.max_new_per_day} new names an evening, and it had the least conviction)")
        del want[s]
    new = new[:limits.max_new_per_day]
    staying = sum(1 for s in held if not parked(s) and want.get(s, {"w": 1.0})["w"] > 0)
    while new and staying + len(new) > limits.max_positions:
        s = new.pop()
        notes.append(f"{s} dropped (the book may hold {limits.max_positions} names)")
        del want[s]

    buys = {s: o["w"] - held.get(s, 0.0) for s, o in want.items() if o["w"] > held.get(s, 0.0)}
    over = sum({**held, **{s: o["w"] for s, o in want.items()}}.values()) - limits.max_invested
    if over > 1e-9 and buys:
        keep = max(0.0, 1.0 - over / sum(buys.values()))
        notes.append(f"buys scaled to {keep:.0%} of what was asked (the book may be at most {limits.max_invested:.0%} invested)")
        for s, add in buys.items():
            w = held.get(s, 0.0) + keep * add
            if w < limits.min_weight or keep * add < 5e-4:
                notes.append(f"{s} dropped (too little of it was left after scaling)")
                del want[s]
            else:
                want[s]["w"] = w
    return [Order(s, math.floor(o["w"] * 1e4 + 1e-6) / 1e4, o["why"]) for s, o in want.items()], notes


# ── small things ────────────────────────────────────────────────────────────────────────────────
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _when(iso) -> datetime | None:
    try:
        t = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _since(stored: str | None, now: str) -> str:
    """Where to start asking the feed: a day back on the first evening; after that a few hours before
    the last item seen, since the seen-list makes asking twice free and asking too late loses filings."""
    t = _when(now) or datetime.now(timezone.utc)
    last = _when(stored) if stored else None
    start = t - timedelta(hours=FIRST_LOOKBACK_H) if last is None else max(last - timedelta(hours=OVERLAP_H),
                                                                          t - timedelta(days=MAX_LOOKBACK_D))
    return start.strftime("%Y-%m-%dT%H:%M:%SZ")


def _move(date: str, m: dict) -> Event:
    """One of the day's large moves as an item to screen. The id holds the date, so it is screened once."""
    sym = str(m.get("symbol", ""))

    def of(key: str, fmt: str) -> str:
        v = m.get(key)
        return format(v, fmt) if isinstance(v, (int, float)) else "n/a"
    title = (f"{m.get('name') or sym} moved {of('ret', '+.1%')} today, {of('vs_index', '+.1%')} against the index: "
             f"{of('sigmas', '+.1f')} times its usual daily move, on {of('volume_ratio', '.1f')} times its usual volume")
    return Event(id=f"move:{date}:{sym}", kind="move", symbol=sym, at=f"{date}T21:00:00Z", title=title,
                 facts={k: m[k] for k in ("ret", "vs_index", "sigmas", "volume_ratio") if k in m})


def _verdicts(chunk: list[Event], items: list) -> dict[str, Verdict]:
    """The small model's answers matched to the items by number. An item it skipped, or answered
    twice, or numbered out of range, counts as not material: its word is taken only where it is clear."""
    got: dict[int, Verdict] = {}
    for it in items:
        n = it.get("n") if isinstance(it, dict) else None
        if isinstance(n, bool) or not isinstance(n, int) or not 0 <= n < len(chunk) or n in got:
            continue
        d = it.get("direction")
        got[n] = Verdict(material=it.get("material") is True, direction=d if d in (-1, 0, 1) and not isinstance(d, bool) else 0,
                         confidence=_unit(it.get("confidence")), why=_line(it.get("why"), 240))
    return {e.id: got.get(n, Verdict()) for n, e in enumerate(chunk)}


def _payload(symbol: str, d: dict, known: dict[str, str], usd: float) -> dict | None:
    """A read as stored and shown, or None when it lacks what it is for: the two chances and a thesis.
    A source is kept only if its address is one the desk gave the model or the search returned."""
    p5, p20, thesis = _prob(d.get("p5")), _prob(d.get("p20")), _text(d.get("thesis"), 700)
    if p5 is None or p20 is None or not thesis:
        return None
    sources: list[dict] = []
    for s in d.get("sources") or []:
        url = _url(s.get("url")) if isinstance(s, dict) else None
        if url in known and all(url != x["url"] for x in sources):
            sources.append({"title": _line(s.get("title"), 160) or known[url] or url, "url": url})
    if not sources:
        sources = [{"title": _line(t, 160) or u, "url": u} for u, t in list(known.items())[:3]]
    facts = [f for f in (_line(x, 240) for x in d.get("facts") or [] if isinstance(x, str)) if f]
    return {"symbol": symbol, "thesis": thesis, "facts": facts[:8], "priced_in": _text(d.get("priced_in"), 700), "p5": p5,
            "p20": p20, "conviction": _unit(d.get("conviction")), "risks": _text(d.get("risks"), 500), "sources": sources[:8],
            "usd": round(usd, 4)}


def _size(ask: Ask) -> int:
    return tokens(*ask.system, ask.user, json.dumps(ask.schema), *(repr(c) for c in ask.paused))


def _said(content: list) -> str:
    return "".join(getattr(b, "text", "") or "" for b in content if getattr(b, "type", "") == "text")[:4000]


def _number(x) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        return None
    return float(x)


def _unit(x) -> float:
    n = _number(x)
    return 0.0 if n is None else round(min(1.0, max(0.0, n)), 4)


def _prob(x) -> float | None:
    """A stated chance, kept off 0 and 1 so that no single forecast can be infinitely wrong."""
    n = _number(x)
    return None if n is None or not 0.0 <= n <= 1.0 else round(min(0.99, max(0.01, n)), 4)


def _line(x, limit: int) -> str:
    s = " ".join(str(x).split()) if isinstance(x, str) else ""
    return s if len(s) <= limit else s[:limit - 1].rstrip() + "…"


def _text(x, limit: int) -> str:
    s = x.strip() if isinstance(x, str) else ""
    return s if len(s) <= limit else s[:limit].rsplit(None, 1)[0] + " …"


def _first_sentence(text: str, limit: int = 200) -> str:
    for m in re.finditer(r"[.!?](?=\s+[A-Z0-9\"'(])", text):
        words = text[:m.start()].split()
        last = words[-1].lower() if words else ""
        if m.group() == "." and (last in _SHORT or len(last) == 1):
            continue
        return _line(text[:m.end()], limit)
    return _line(text, limit)


def _cap_words(text: str, limit: int) -> str:
    """A playbook that ran long, cut at the last whole line within the limit: every evening's prompts pay for each word."""
    if len(text.split()) <= limit:
        return text
    kept, n = [], 0
    for line in text.splitlines():
        n += len(line.split())
        if n > limit:
            break
        kept.append(line)
    return "\n".join(kept).rstrip()


def _url(url) -> str:
    """An address in the one spelling the desk compares by; "" unless it is a plain web address."""
    u = url.strip().split("#")[0].rstrip("/") if isinstance(url, str) else ""
    return u if urlparse(u).scheme in ("http", "https") else ""


def _is_sec(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host == "sec.gov" or host.endswith(".sec.gov")
