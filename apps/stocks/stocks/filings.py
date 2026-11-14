"""From the SEC's files to the dated tables in events.py: the days results came out, the purchases insiders
made, and the month of filings the desk reads.

`update` is the evening pass over what was accepted since the last one; `backfill` is the history, once.
Every fact gets `day`, the first trading day whose 16:00 New York close comes after the filing was accepted
(`reaction_day`). Where only a filing date is known, the day is the next trading day after it (`day_after`):
never before the truth, at most one day late.

A company that cannot be read goes on the list of failures and the rest are read. When the SEC turns the
run away, or stops answering, or its times stop looking like UTC, the run ends with `Stopped` after saving
what was sound: nothing more is asked, and the caller's `since` should stay where it was. Reading the same
stretch again is cheap: a Form 4 that was read before is not fetched a second time."""
from __future__ import annotations

import logging
import os
import re
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from .cal import NY, is_trading_day, next_trading_day
from .events import EARNINGS_COLS, FORMS_FOR_AGENT, INSIDER_COLS, Events
from .sec import ISSUER_COLS, SET_COLS, Sec, SecError, read_insider_set, read_set_issuers

_log = logging.getLogger("stocks.filings")

CLOSE = time(16, 0)
KEEP_DAYS = 30                  # how long a filing stays in recent.parquet
RECENT_COLS = ["sym", "at", "form", "items", "title", "url", "accession"]
IN_A_ROW = 8                    # this many requests failing one after another, and the run stops
TAIL_DAYS = 200                 # the stretch after the last quarterly set that `backfill` will read Form 4 by Form 4
BATCH = 50                      # companies between saves in `backfill`
MIN_FILINGS = 3                 # insider filings under a ticker before the sets are believed about whose it was
REACH_BACK = 400                # days before the first of them that the company's link to the ticker reaches,
REACH_ON = 120                  # and days after the last: results are not filed on the days insiders' forms are
EVER, ALWAYS = "0000-00-00", "9999-99-99"
_BUY = ("owner", "officer", "director", "ten_pct", "title", "shares", "price", "usd")

# Since December 2024 EDGAR lists these under their full names; events.FORMS_FOR_AGENT has the old ones.
ALIASES = {"SCHEDULE 13D": "SC 13D", "SCHEDULE 13D/A": "SC 13D/A", "SCHEDULE 13G": "SC 13G", "SCHEDULE 13G/A": "SC 13G/A"}

# Form 8-K's items (sec.gov/files/form8-k.pdf) in plain words.
ITEMS = {
    "1.01": "material agreement", "1.02": "material agreement ended", "1.03": "bankruptcy", "1.04": "mine safety",
    "1.05": "cybersecurity incident", "2.01": "acquisition or sale completed", "2.02": "results of operations",
    "2.03": "new debt", "2.04": "debt called early", "2.05": "restructuring costs", "2.06": "impairment",
    "3.01": "delisting notice", "3.02": "unregistered share sale", "3.03": "holders' rights changed",
    "4.01": "auditor changed", "4.02": "past statements not to be relied on", "5.01": "change of control",
    "5.02": "officers or directors changed", "5.03": "bylaws or fiscal year changed",
    "5.04": "trading in benefit plans suspended", "5.05": "code of ethics changed", "5.06": "no longer a shell company",
    "5.07": "shareholder vote", "5.08": "director nominations", "6.01": "asset-backed: computational material",
    "6.02": "asset-backed: servicer or trustee changed", "6.03": "asset-backed: credit support changed",
    "6.04": "asset-backed: missed distribution", "6.05": "asset-backed: updating disclosure",
    "6.06": "asset-backed: static pool", "7.01": "Regulation FD disclosure", "8.01": "other events", "9.01": "exhibits",
}
FORMS = {
    "8-K": "current report", "10-Q": "quarterly report", "10-K": "annual report",
    "20-F": "annual report of a foreign company", "6-K": "report from a foreign company",
    "SC 13D": "a holder of more than 5% states its intentions",
    "SC 13D/A": "a holder of more than 5% changes its stake or its intentions",
    "SC 13G": "a passive holder reports more than 5%", "SC 13G/A": "a passive holder's stake has changed",
}


class Stopped(Exception):
    """A run that ended early. What was sound is saved; `result` holds the counts so far."""

    def __init__(self, why: str, result: dict):
        super().__init__(why)
        self.result = result


# ── days ────────────────────────────────────────────────────────────────────────────────────────
def _when(at) -> datetime:
    """A time as a moment. A string that names no zone is New York time, as in this app's tables and clocks."""
    t = at if isinstance(at, datetime) else datetime.fromisoformat(re.sub(r"[Zz]$", "+00:00", str(at).strip()))
    return t if t.tzinfo else t.replace(tzinfo=NY)


def _d(d: date) -> int:
    return d.year * 10000 + d.month * 100 + d.day


def reaction_day(at) -> int:
    """YYYYMMDD of the first trading day whose 16:00 New York close is after `at`: the same day for a filing
    accepted before the close on a trading day, the next trading day for one at 16:00:00 or later, on a
    weekend or on a holiday."""
    t = _when(at).astimezone(NY)
    d = t.date()
    if not (is_trading_day(d) and t.time() < CLOSE):
        d = next_trading_day(d)
    return _d(d)


def day_after(filed: str) -> int:
    """YYYYMMDD of the next trading day after a filing date: the rule when the time of day is not known.
    Forms 3, 4 and 5 are taken until 22:00, so one dated today may have come after the close."""
    return _d(next_trading_day(date.fromisoformat(str(filed)[:10])))


# ── words ───────────────────────────────────────────────────────────────────────────────────────
def _items(items: str) -> list[str]:
    return [x.strip() for x in str(items or "").split(",") if x.strip()]


def _trade(word: str, rows: list[dict]) -> str:
    shares, usd = sum(r["shares"] for r in rows), sum(r["usd"] for r in rows)
    return f"{word} {shares:,.0f} shares at ${usd / shares:,.2f} (${usd:,.0f})"


def title_of(form: str, items: str = "", doc: dict | None = None) -> str:
    """One line a person can read, which is all the desk's screening sees of a filing:
    "8-K: results of operations (2.02), exhibits (9.01)". For a Form 4, `doc` is what was read in it."""
    if form == "4":
        if doc is None:
            return "4: an insider reports trades"
        buys, sells = doc.get("purchases") or [], doc.get("sales") or []
        if not buys and not sells:
            return "4: an insider reports awards, exercises or gifts, no open-market trade"
        one = (buys or sells)[0]
        who = one["owner"] + (f" ({one['title']})" if one.get("title") else "")
        return f"4: {who} " + " and ".join(t for t in (buys and _trade("bought", buys), sells and _trade("sold", sells)) if t)
    codes = _items(items)
    if form == "8-K" and codes:
        return "8-K: " + ", ".join(f"{ITEMS.get(c, 'item')} ({c})" for c in codes)
    return f"{form}: {FORMS.get(form, 'filing')}"


def _lines(buys: list[dict]) -> list[dict]:
    """A filing's purchases with those at one price by one person added together. events.add tells rows
    apart by filing, symbol, owner, shares and price, so two equal fills in one Form 4 would be taken for one."""
    out: dict[tuple, dict] = {}
    for b in buys:
        key = (b["owner"], float(b["price"]))
        if key in out:
            o = out[key]
            o["shares"], o["usd"], o["date"] = o["shares"] + b["shares"], round(o["usd"] + b["usd"], 2), min(o["date"], b["date"])
        else:
            out[key] = dict(b)
    return list(out.values())


# ── a run ───────────────────────────────────────────────────────────────────────────────────────
class _Run:
    """What `update` and `backfill` share: reading a company and a Form 4, keeping count, stopping, saving."""

    def __init__(self, sec: Sec, events: Events, out: dict, say: Callable[[str], None], hours_from: str,
                 now: datetime | None = None):
        self.sec, self.events, self.out, self.say, self.now = sec, events, out, say, now
        self.have = set(events.insiders["accession"])       # Form 4s whose purchases are stored, or about to be
        self.earn: list[dict] = []
        self.buys: list[dict] = []
        self.news: list[dict] = []
        self.in_a_row = 0
        self.hours_from, self.seen, self.odd = hours_from, 0, 0

    def current(self, symbols: list[str]) -> dict[str, int]:
        """Symbol → CIK for the companies the SEC's ticker list knows: those listed today."""
        try:
            return self.sec.ciks(symbols)
        except SecError as e:
            raise Stopped(f"the SEC's ticker list could not be read: {e}", self.out) from e

    def companies(self, symbols: Iterable[str]) -> dict[int, list[str]]:
        """CIK → the symbols it trades under. Symbols the SEC's ticker list does not know are counted."""
        symbols = list(dict.fromkeys(symbols))
        ciks = self.current(symbols)
        by_cik: dict[int, list[str]] = {}
        for s in symbols:
            if s in ciks:
                by_cik.setdefault(ciks[s], []).append(s)
        self.out["no_cik"] = len(symbols) - len(ciks)
        return by_cik

    def fail(self, e: Exception, **where) -> None:
        self.out["failures"].append({**where, "error": f"{type(e).__name__}: {e}"[:240]})
        self.in_a_row += 1
        if isinstance(e, SecError) and e.refused:
            self.stop(f"the SEC is refusing requests ({e.status}); nothing more was asked of it")
        if self.in_a_row >= IN_A_ROW:
            self.stop(f"{IN_A_ROW} requests in a row failed, the last with {type(e).__name__}: {e}")

    def stop(self, why: str) -> None:
        self.save()
        raise Stopped(why, self.out)

    def read(self, cik: int, syms: list[str], since: str) -> dict | None:
        try:
            sub = self.sec.submissions(cik, all_pages=True, since=since)
        except Exception as e:
            self.fail(e, sym=",".join(syms), cik=cik)
            return None
        self.in_a_row = 0
        self.out["companies"] += 1
        for f in sub["filings"]:                            # the check on the clock, see save()
            if f["at"] and f["filing_date"] >= self.hours_from:
                self.seen += 1
                self.odd += not ("05:55" <= f["at"][11:16] <= "22:30")
        return sub

    def form4(self, cik: int, syms: list[str], f: dict) -> dict | None:
        """Reads one Form 4 and queues its purchases. None when it could not be read (a failure is noted)
        or when this company is not the issuer: its own list also holds the Form 4s in which it is the buyer."""
        try:
            doc = self.sec.ownership(cik, f["accession"], f["primary"])
        except Exception as e:
            self.fail(e, sym=",".join(syms), cik=cik, accession=f["accession"])
            return None
        self.in_a_row = 0
        self.out["form4"] += 1
        if doc["issuer_cik"] != cik:
            return None
        day = reaction_day(f["at"]) if f["at"] else day_after(f["filing_date"])
        for b in _lines(doc["purchases"]):
            self.buys += [{"sym": s, "day": day, "at": f["at"] or f["filing_date"], **{k: b[k] for k in _BUY},
                           "accession": f["accession"]} for s in syms]
        self.have.add(f["accession"])
        return doc

    def stored(self, accession: str, syms: list[str]) -> dict | None:
        """A Form 4 already in the table, as it was read then. None if its rows are another company's."""
        t = self.events.insiders
        rows = t[(t["accession"] == accession) & (t["sym"] == syms[0])]
        return {"purchases": rows.to_dict("records"), "sales": []} if len(rows) else None

    def save(self) -> None:
        """Appends what has been gathered. First the check on the clock: EDGAR takes filings from 06:00 to
        22:00 New York time, so times read rightly all fall inside those hours. If the SEC ever wrote New
        York digits under the Z (or stopped), every morning filing would land before 06:00, and every
        after-hours results day would be a day early: then nothing is saved."""
        if self.seen >= 200 and self.odd > 0.02 * self.seen:
            self.earn, self.buys, self.news = [], [], []
            raise Stopped(f"{self.odd} of {self.seen} acceptance times fall outside EDGAR's hours (06:00 to 22:00 New "
                          "York): they no longer look like UTC. Nothing was saved; see ny_time in sec.py", self.out)
        if self.earn:
            self.out["earnings"] += self.events.add("earnings", pd.DataFrame(self.earn, columns=EARNINGS_COLS))
        if self.buys:
            self.out["insiders"] += self.events.add("insiders", pd.DataFrame(self.buys, columns=INSIDER_COLS))
        if self.now is not None:
            self.out["recent"] += _save_recent(self.events.dir, self.news, self.now)
        self.earn, self.buys, self.news = [], [], []


def _recent(folder: Path) -> pd.DataFrame:
    path = Path(folder) / "recent.parquet"
    return pd.read_parquet(path)[RECENT_COLS] if path.exists() else pd.DataFrame(columns=RECENT_COLS)


def _save_recent(folder: Path, rows: list[dict], now: datetime) -> int:
    """Adds filings to recent.parquet, drops those accepted more than KEEP_DAYS ago, and returns how many
    of the new ones are in it. A filing already there stays as it was."""
    old = _recent(folder)
    new = pd.DataFrame(rows, columns=RECENT_COLS)
    both = pd.concat([old.assign(new=False), new.assign(new=True)], ignore_index=True) if len(old) else new.assign(new=True)
    both = both.drop_duplicates(["accession", "sym"], keep="first")
    both = both.assign(when=pd.to_datetime(both["at"], utc=True, format="ISO8601"))     # before the cut: a column given to
    both = both[both["when"] >= pd.Timestamp(now - timedelta(days=KEEP_DAYS))].sort_values("when", kind="stable")   # an empty table brings its rows back
    if len(both) or len(old):
        Path(folder).mkdir(parents=True, exist_ok=True)
        tmp = Path(folder) / "recent.parquet.tmp"
        both[RECENT_COLS].to_parquet(tmp, index=False)
        os.replace(tmp, Path(folder) / "recent.parquet")       # whole or not at all: the page may be reading it
    return int(both["new"].sum())


# ── the evening ─────────────────────────────────────────────────────────────────────────────────
def update(sec: Sec, events: Events, symbols: Iterable[str], since: str, log: Callable[[str], None] | None = None,
           now: datetime | None = None) -> dict:
    """Reads what each company has had accepted after `since` (ISO; New York time if it names no zone):
    an `earnings` row for each 8-K with item 2.02, the purchases of each Form 4 into `insiders`, and every
    filing of the forms in events.FORMS_FOR_AGENT into recent.parquet with a title and its document's address.

    Returns {companies, earnings, insiders, recent, form4, no_cik, failures}: companies read, rows that were
    new in each table, Form 4s fetched, symbols without a CIK, and one entry per company or Form 4 that could
    not be read. Raises Stopped when the run had to end early."""
    out = {"companies": 0, "earnings": 0, "insiders": 0, "recent": 0, "form4": 0, "no_cik": 0, "failures": []}
    say = log or _log.info
    now = (now or datetime.now(NY)).astimezone(NY)
    start = _when(since)
    floor = start.astimezone(NY).date().isoformat()         # no filing dated before this was accepted after `since`
    run = _Run(sec, events, out, say, hours_from=(now - timedelta(days=365)).date().isoformat(), now=now)
    by_cik = run.companies(symbols)
    old = _recent(events.dir)
    shown = set(zip(old["accession"], old["sym"]))          # read on an earlier evening: not fetched again
    for n, (cik, syms) in enumerate(sorted(by_cik.items()), 1):
        sub = run.read(cik, syms, floor)
        try:
            for f in (sub["filings"] if sub else []):
                if f["filing_date"] < floor or not f["at"] or _when(f["at"]) <= start:
                    continue
                form, doc = ALIASES.get(f["form"], f["form"]), None
                if form == "8-K" and "2.02" in _items(f["items"]):
                    run.earn += [{"sym": s, "day": reaction_day(f["at"]), "at": f["at"], "form": form,
                                  "accession": f["accession"]} for s in syms]
                if form == "4":
                    if (f["accession"], syms[0]) in shown:
                        continue
                    doc = run.stored(f["accession"], syms) if f["accession"] in run.have else run.form4(cik, syms, f)
                    if doc is None:
                        continue
                if form not in FORMS_FOR_AGENT or (form.startswith("SC 13") and not f["file_number"]):
                    continue                                # a Schedule 13 with no file number is this company's own
                try:                                        # report of a stake in another: that one's news, not this one's
                    url = sec.filing_url(cik, f["accession"], f["primary"])
                except ValueError:
                    url = sec.filing_url(cik, f["accession"], "")
                run.news += [{"sym": s, "at": f["at"], "form": form, "items": f["items"], "title": title_of(form, f["items"], doc),
                              "url": url, "accession": f["accession"]} for s in syms]
        except Stopped:
            raise
        except Exception as e:
            run.fail(e, sym=",".join(syms), cik=cik)
        if n % 100 == 0:
            say(f"filings: {n} of {len(by_cik)} companies read")
    run.save()
    return out


# ── the history ─────────────────────────────────────────────────────────────────────────────────
def _quarter_end(year: int, quarter: int) -> str:
    return f"{year}-{3 * quarter:02d}-{(31, 30, 30, 31)[quarter - 1]}"


def _sets(run: _Run, start_year: int, today: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every quarterly set from `start_year` on: the purchases in them (SET_COLS), and who filed under which
    ticker from when to when (ISSUER_COLS, the quarters added together)."""
    try:
        listed = sorted(run.sec.insider_quarters())
        if not listed:
            raise ValueError("no set is linked from it")
    except Exception as e:                                  # the page has moved or changed: every finished quarter
        run.fail(e, quarter="the page that lists the sets")  # is then tried at its known address
        listed = [(y, q) for y in range(start_year, today.year + 1) for q in (1, 2, 3, 4) if _quarter_end(y, q) < today.isoformat()]
    parts, names = [], []
    for year, quarter in listed:
        if year < start_year:
            continue
        try:
            body = run.sec.insider_set(year, quarter)
            df, who = read_insider_set(body), read_set_issuers(body)
        except Exception as e:
            run.fail(e, quarter=f"{year}q{quarter}")
            continue
        run.in_a_row = 0
        run.out["quarters"] += 1
        run.out["insiders_through"] = max(run.out["insiders_through"] or "", _quarter_end(year, quarter))
        parts.append(df)
        names.append(who)
        run.say(f"insider set {year} Q{quarter}: {len(df)} purchases, {who['cik'].nunique() if len(who) else 0} companies")
    buys = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SET_COLS)
    names = [w for w in names if len(w)]
    issuers = (pd.concat(names, ignore_index=True).groupby(["cik", "sym"], as_index=False)
               .agg(first=("first", "min"), last=("last", "max"), n=("n", "sum"))[ISSUER_COLS]
               if names else pd.DataFrame(columns=ISSUER_COLS))
    return buys, issuers


def _shift(day: str, days: int) -> str:
    return (date.fromisoformat(day) + timedelta(days=days)).isoformat()


def links(symbols: Iterable[str], current: dict[str, int], issuers: pd.DataFrame, start: str) -> dict[int, list[tuple[str, str, str]]]:
    """Which company each symbol was, and from which day to which: CIK → [(symbol, first day, last day)].

    The SEC's ticker list knows only who holds a ticker today. A backtest that stopped there would know the
    results days and the insiders of the companies that survived, and of no company since bought, merged or
    delisted. The insider sets say who filed under a ticker in each quarter, so:

    - a company listed today is its symbol for all time: Alpaca's bars follow a company through a change of
      ticker (META's reach back through the years it was FB), and so do its filings;
    - a symbol nobody holds today belongs to whoever filed under it, from a little before the first of
      those filings to a little after the last (from the start, if they were filing when the history begins);
    - a ticker that passed from a company now gone to the one that holds it today gives the old company its
      own years too (Dow Chemical until 2017, Dow Inc. from 2019), unless today's holder was already filing
      then under another ticker: in that case the bars of those years are its own, and the old company's
      are out of reach."""
    seen = issuers[issuers["n"] >= MIN_FILINGS] if len(issuers) else issuers
    born = seen.groupby("cik")["first"].min().to_dict() if len(seen) else {}
    by_sym = {s: g.sort_values("first").to_dict("records") for s, g in seen.groupby("sym")} if len(seen) else {}
    early = _shift(start, 365)
    out: dict[int, list[tuple[str, str, str]]] = {}
    for s in dict.fromkeys(symbols):
        cur = current.get(s)
        if cur is not None:
            out.setdefault(int(cur), []).append((s, EVER, ALWAYS))
        for i, r in enumerate(by_sym.get(s, [])):
            cik = int(r["cik"])
            if cik == cur or (cur is not None and born.get(cur, ALWAYS) <= r["last"]):
                continue
            lo = EVER if cur is None and i == 0 and r["first"] <= early else _shift(r["first"], -REACH_BACK)
            out.setdefault(cik, []).append((s, lo, _shift(r["last"], REACH_ON)))
    return out


def backfill(sec: Sec, events: Events, symbols: Iterable[str], start_year: int, log: Callable[[str], None] | None = None,
             tail: bool | Iterable[str] = True, today: date | None = None) -> dict:
    """The history from `start_year`: results days from every page of each company's filings, insider
    purchases from the SEC's quarterly sets.

    The sets carry a filing date only. A purchase is dated by the acceptance time in the company's own list
    of filings where the Form 4 is found there, and otherwise by `day_after`. The sets also end with the last
    finished quarter that has been published, so with `tail` the Form 4s filed since are read one by one
    (a request each; skipped when that stretch is longer than TAIL_DAYS). `tail` may be a list of symbols:
    then only theirs are read, which for the stocks a model can buy today is a third of the requests.

    Companies listed today are found through the SEC's ticker list; those since bought, merged or delisted
    through the tickers their insiders filed under (`links`), so the history is not one of survivors only.

    Returns {companies, earnings, insiders, quarters, insiders_through, form4, no_cik, former, failures}:
    `insiders_through` is the last day the sets cover, `former` the symbols found only through the sets,
    `no_cik` those not found at all. Raises Stopped when the run had to end early; run again and it picks
    up, adding nothing twice."""
    out = {"companies": 0, "earnings": 0, "insiders": 0, "quarters": 0, "insiders_through": None, "form4": 0,
           "no_cik": 0, "former": 0, "failures": []}
    say = log or _log.info
    today = today or datetime.now(NY).date()
    start = f"{int(start_year)}-01-01"
    symbols = list(dict.fromkeys(symbols))
    run = _Run(sec, events, out, say, hours_from=start)
    current = run.current(symbols)
    sets, issuers = _sets(run, int(start_year), today)
    spans = links(symbols, current, issuers, start)
    linked = {s for rows in spans.values() for s, _, _ in rows}
    out["no_cik"], out["former"] = len(symbols) - len(linked), len(linked - set(current))
    if out["former"]:
        say(f"{out['former']} symbols no longer listed were found through the tickers their insiders filed under")
    by_issuer = {int(c): g for c, g in sets[sets["cik"].isin(set(spans))].groupby("cik")}
    through, tail_from = out["insiders_through"], None
    wanted = None if isinstance(tail, bool) else set(tail)      # None: every symbol's
    if (tail is True or wanted) and through and (today - date.fromisoformat(through)).days <= TAIL_DAYS:
        tail_from = (date.fromisoformat(through) - timedelta(days=4)).isoformat()    # a Form 4 of the quarter's last evening is in the next set
        say(f"insider purchases since {through}: reading each Form 4 filed since, one request apiece")
    for n, (cik, rows) in enumerate(sorted(spans.items()), 1):
        names = [s for s, _, _ in rows]

        def on(day: str) -> list[str]:                      # the symbols this company was on that day
            return [s for s, lo, hi in rows if lo <= day[:10] <= hi]
        sub = run.read(cik, names, start)
        filings = [f for f in (sub["filings"] if sub else []) if f["filing_date"] >= start]
        try:
            for f in filings:
                if f["form"] == "8-K" and "2.02" in _items(f["items"]):
                    day = reaction_day(f["at"]) if f["at"] else day_after(f["filing_date"])
                    run.earn += [{"sym": s, "day": day, "at": f["at"] or f["filing_date"], "form": "8-K",
                                  "accession": f["accession"]} for s in on(f["filing_date"])]
            accepted = {f["accession"]: f["at"] for f in filings if f["at"]}
            for accession, g in (by_issuer[cik].groupby("accession", sort=False) if cik in by_issuer else ()):
                if accession in run.have:
                    continue
                at, filed = accepted.get(accession, ""), str(g["filed"].iloc[0])
                day = reaction_day(at) if at else day_after(filed)
                for b in _lines(g.to_dict("records")):
                    run.buys += [{"sym": s, "day": day, "at": at or filed, **{k: b[k] for k in _BUY}, "accession": accession}
                                 for s in on(filed)]
                run.have.add(accession)
            for f in filings if tail_from else ():
                if f["form"] == "4" and f["filing_date"] >= tail_from and f["accession"] not in run.have:
                    mine = [s for s in on(f["filing_date"]) if wanted is None or s in wanted]
                    if mine:
                        run.form4(cik, mine, f)
        except Stopped:
            raise
        except Exception as e:
            run.fail(e, sym=",".join(names), cik=cik)
        if n % BATCH == 0:
            run.save()
            say(f"filings history: {n} of {len(spans)} companies read")
    run.save()
    return out


def sectors(sec: Sec, symbols: Iterable[str], log: Callable[[str], None] | None = None) -> dict[str, int]:
    """The SIC code (the SEC's four-digit industry) of each symbol that has one: a request per company.
    A company that cannot be read is left out and said so; a refusal by the SEC, or IN_A_ROW failures one
    after another, raises."""
    say = log or _log.info
    symbols = list(dict.fromkeys(symbols))
    ciks = sec.ciks(symbols)
    by_cik: dict[int, list[str]] = {}
    for s in symbols:
        if s in ciks:
            by_cik.setdefault(ciks[s], []).append(s)
    out: dict[str, int] = {}
    in_a_row = 0
    for cik, syms in sorted(by_cik.items()):
        try:
            sic = sec.submissions(cik)["sic"]
        except Exception as e:
            in_a_row += 1
            if (isinstance(e, SecError) and e.refused) or in_a_row >= IN_A_ROW:
                raise
            say(f"no industry for {','.join(syms)}: {e}")
            continue
        in_a_row = 0
        if sic is not None:
            out.update({s: sic for s in syms})
    return out
