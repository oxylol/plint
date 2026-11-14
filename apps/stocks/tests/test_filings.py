"""From filings to the dated tables: which trading day a filing belongs to, what the evening pass and the
history write, and what happens when the SEC does not answer. No network: the same stand-in and fixtures as
tests/test_sec.py (the companies are made up; what each fixture row is there for is said where it is used).

    docker run --rm -v "$PWD":/app -w /app plint/stocks:dev python -m unittest tests.test_sec tests.test_filings -v
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from stocks import filings
from stocks.cal import NY, trading_days
from stocks.events import EARNINGS_COLS, INSIDER_COLS, Events
from stocks.filings import Stopped, day_after, reaction_day, title_of
from stocks.panel import Panel
from stocks.sec import SETS_PAGE, WWW, SecError, sec_url

try:
    from tests.test_sec import ARCH, COVER, SETS, SUBS, TICKERS, Clock, Web, client, fixture, insider_zip, site
except ImportError:                                     # run with tests/ itself on the path
    from test_sec import ARCH, COVER, SETS, SUBS, TICKERS, Clock, Web, client, fixture, insider_zip, site

SYMS = ["NWT", "RVR.A", "RVR.B", "GONE", "NOPE"]        # GONE has a CIK and no file at the SEC; NOPE has no CIK
SINCE = "2026-11-04T00:00:00"                           # New York, no zone named: as pipeline.py passes it
NOW = datetime(2026, 11, 13, 18, 0, tzinfo=NY)          # a Friday evening
A = "0001999990-26-000"
RESULTS, OFFICER_8K, AMENDED_8K = "0009900001-26-000031", "0009900001-26-000030", "0009900001-26-000027"
FORM4_MIXED, FORM4_PAIR, FORM4_SALE, FORM4_AUGUST = A + "412", A + "409", A + "398", A + "301"


def form4_url(cik: int, accession: str) -> str:
    return f"{ARCH}{cik}/{accession.replace('-', '')}/form4.xml"


class Days(unittest.TestCase):
    def test_both_sides_of_the_close(self):
        self.assertEqual(reaction_day("2026-11-12T15:59:59-05:00"), 20261112)
        self.assertEqual(reaction_day("2026-11-12T16:00:00-05:00"), 20261113)          # the close is not after it
        self.assertEqual(reaction_day("2026-11-12T16:05:11-05:00"), 20261113)
        self.assertEqual(reaction_day("2026-11-12T09:10:00-05:00"), 20261112)
        self.assertEqual(reaction_day("2026-11-12T06:00:01-05:00"), 20261112)
        self.assertEqual(reaction_day("2026-11-12T21:59:00-05:00"), 20261113)

    def test_the_hour_is_new_yorks_whatever_zone_the_time_came_in(self):
        self.assertEqual(reaction_day("2026-07-30T19:59:59Z"), 20260730)               # 15:59:59 in summer
        self.assertEqual(reaction_day("2026-07-30T20:00:00Z"), 20260731)
        self.assertEqual(reaction_day("2026-01-29T20:59:59Z"), 20260129)               # 15:59:59 in winter
        self.assertEqual(reaction_day("2026-01-29T21:00:00Z"), 20260130)
        # the same hour by UTC on either side of the night the clocks go back (1 November 2026)
        self.assertEqual(reaction_day("2026-10-30T20:30:00Z"), 20261102)               # Friday 16:30: Monday's
        self.assertEqual(reaction_day("2026-11-02T20:30:00Z"), 20261102)               # Monday 15:30: still Monday's
        self.assertEqual(reaction_day("2026-11-12T22:05:11+01:00"), 20261113)          # 16:05 New York, written in Stockholm
        self.assertEqual(reaction_day(datetime(2026, 11, 12, 21, 5, 11, tzinfo=timezone.utc)), 20261113)

    def test_a_time_that_names_no_zone_is_new_york_time(self):
        self.assertEqual(reaction_day("2026-11-12T16:05:11"), 20261113)
        self.assertEqual(reaction_day("2026-11-12T15:05:11"), 20261112)

    def test_across_a_weekend(self):
        self.assertEqual(reaction_day("2026-11-06T15:30:00-05:00"), 20261106)          # Friday before the close
        self.assertEqual(reaction_day("2026-11-06T16:17:40-05:00"), 20261109)          # Friday after it
        self.assertEqual(reaction_day("2026-11-07T10:00:00-05:00"), 20261109)          # Saturday
        self.assertEqual(reaction_day("2026-11-08T23:00:00-05:00"), 20261109)          # Sunday night
        self.assertEqual(reaction_day("2026-11-09T08:30:02-05:00"), 20261109)          # Monday before the open

    def test_across_a_holiday(self):
        self.assertEqual(reaction_day("2026-11-25T16:30:00-05:00"), 20261127)          # the eve of Thanksgiving
        self.assertEqual(reaction_day("2026-11-26T10:00:00-05:00"), 20261127)          # Thanksgiving morning
        self.assertEqual(reaction_day("2026-04-02T17:00:00-04:00"), 20260406)          # Thursday before Good Friday: Monday
        self.assertEqual(reaction_day("2026-05-22T16:45:00-04:00"), 20260526)          # Friday before Memorial Day: Tuesday
        self.assertEqual(reaction_day("2026-06-18T16:30:00-04:00"), 20260622)          # Juneteenth on a Friday
        self.assertEqual(reaction_day("2026-12-24T16:30:00-05:00"), 20261228)          # Christmas on a Friday
        self.assertEqual(reaction_day("2026-12-31T16:30:00-05:00"), 20270104)          # into the new year
        self.assertEqual(reaction_day("2026-11-10T16:30:00-05:00"), 20261111)          # Veterans Day: the market is open

    def test_with_only_a_filing_date_it_is_the_next_trading_day(self):
        self.assertEqual(day_after("2026-11-12"), 20261113)
        self.assertEqual(day_after("2026-08-14"), 20260817)                            # Friday
        self.assertEqual(day_after("2026-09-04"), 20260908)                            # Friday before Labor Day
        self.assertEqual(day_after("2026-11-07"), 20261109)                            # a Saturday


class Titles(unittest.TestCase):
    def test_an_8k_says_what_it_is_about(self):
        self.assertEqual(title_of("8-K", "2.02,9.01"), "8-K: results of operations (2.02), exhibits (9.01)")
        self.assertEqual(title_of("8-K", "5.02"), "8-K: officers or directors changed (5.02)")
        self.assertEqual(title_of("8-K", " 1.05 , 7.01"), "8-K: cybersecurity incident (1.05), Regulation FD disclosure (7.01)")
        self.assertEqual(title_of("8-K", "9.99"), "8-K: item (9.99)")
        self.assertEqual(title_of("8-K", ""), "8-K: current report")

    def test_the_other_forms(self):
        self.assertEqual(title_of("10-Q"), "10-Q: quarterly report")
        self.assertEqual(title_of("10-K"), "10-K: annual report")
        self.assertEqual(title_of("SC 13D"), "SC 13D: a holder of more than 5% states its intentions")
        self.assertEqual(title_of("DEF 14A"), "DEF 14A: filing")
        for form in filings.FORMS_FOR_AGENT:
            self.assertNotIn(": filing", title_of(form), form)                         # every form the desk is shown has its words

    def test_a_form_4_says_who_did_what(self):
        doe = {"owner": "DOE JANE", "title": "Chief Financial Officer"}
        doc = {"purchases": [{**doe, "shares": 5000.0, "usd": 200500.0}], "sales": [{**doe, "shares": 1200.0, "usd": 49500.0}]}
        self.assertEqual(title_of("4", "", doc), "4: DOE JANE (Chief Financial Officer) bought 5,000 shares at $40.10 ($200,500) "
                                                 "and sold 1,200 shares at $41.25 ($49,500)")
        self.assertEqual(title_of("4", "", {"purchases": [], "sales": doc["sales"]}), "4: DOE JANE (Chief Financial Officer) sold 1,200 shares at $41.25 ($49,500)")
        self.assertEqual(title_of("4", "", {"purchases": [], "sales": []}), "4: an insider reports awards, exercises or gifts, no open-market trade")
        self.assertEqual(title_of("4"), "4: an insider reports trades")


class Lab(unittest.TestCase):
    """A cache folder of its own, the made-up SEC, and a clock that never really sleeps."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="stocks-filings-"))
        self.ev = Events(self.tmp)
        self.clock = Clock()
        self.web = Web(site(), self.clock)
        self.sec = client(self.web)
        self.said: list[str] = []

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def update(self, since=SINCE, now=NOW, symbols=SYMS):
        return filings.update(self.sec, self.ev, symbols, since, log=self.said.append, now=now)

    def recent_file(self) -> pd.DataFrame:
        return pd.read_parquet(self.tmp / "events" / "recent.parquet")

    def shown(self, since="2026-01-01T00:00:00"):
        return self.ev.recent(since, set(SYMS))


class Update(Lab):
    def test_the_count_and_the_company_that_could_not_be_read(self):
        got = self.update()
        self.assertEqual({k: v for k, v in got.items() if k != "failures"},
                         {"companies": 2, "earnings": 3, "insiders": 3, "recent": 11, "form4": 3, "no_cik": 1})
        self.assertEqual(len(got["failures"]), 1)
        gone = got["failures"][0]
        self.assertEqual((gone["sym"], gone["cik"]), ("GONE", 9900002))
        self.assertIn("404", gone["error"])
        self.assertEqual(self.web.asked.index(SUBS + "CIK0009900002.json") + 1, self.web.asked.index(SUBS + "CIK0009900003.json"))  # and on it went

    def test_results_days(self):
        self.update()
        e = self.ev.earnings.sort_values("sym").reset_index(drop=True)
        self.assertEqual(list(e.columns), EARNINGS_COLS)
        self.assertEqual(e.to_dict("records"), [
            # accepted 16:05:11 on Thursday the 12th: Friday's close is the first after it
            {"sym": "NWT", "day": 20261113, "at": "2026-11-12T16:05:11-05:00", "form": "8-K", "accession": RESULTS},
            # accepted 16:30 on Tuesday the 10th, for both of Riverbend's share classes
            {"sym": "RVR.A", "day": 20261111, "at": "2026-11-10T16:30:00-05:00", "form": "8-K", "accession": "0009900003-26-000044"},
            {"sym": "RVR.B", "day": 20261111, "at": "2026-11-10T16:30:00-05:00", "form": "8-K", "accession": "0009900003-26-000044"},
        ])                                                  # not the 8-K about a new officer (item 5.02), not the amended 8-K/A
        self.assertEqual(str(e["day"].dtype), "int64")

    def test_insider_purchases(self):
        self.update()
        i = self.ev.insiders.sort_values(["day", "price"]).reset_index(drop=True)
        self.assertEqual(list(i.columns), INSIDER_COLS)
        roe = {"sym": "NWT", "day": 20261110, "at": "2026-11-10T10:12:44-05:00", "owner": "ROE RICHARD", "officer": False,
               "director": True, "ten_pct": True, "title": "Director", "accession": FORM4_PAIR}
        self.assertEqual(i.to_dict("records"), [
            {**roe, "shares": 16000.0, "price": 39.8, "usd": 636800.0},      # two fills at 39.80 as one line, or events.add would drop one
            {**roe, "shares": 4000.0, "price": 39.95, "usd": 159800.0},
            {"sym": "NWT", "day": 20261113, "at": "2026-11-12T17:41:09-05:00", "owner": "DOE JANE", "officer": True, "director": False,
             "ten_pct": False, "title": "Chief Financial Officer", "shares": 5000.0, "price": 40.1, "usd": 200500.0, "accession": FORM4_MIXED},
        ])                                                  # nothing for Riverbend: in the Form 4 on its list it is the buyer
        self.assertEqual([str(i[c].dtype) for c in ("day", "officer", "director", "ten_pct", "usd")], ["int64", "bool", "bool", "bool", "float64"])

    def test_the_models_can_read_what_was_written(self):
        self.update()
        panel = Panel(np.array([20261109, 20261110, 20261111, 20261112, 20261113, 20261116]), ["NWT", "RVR.A"], {})
        self.assertEqual(self.ev.earnings_table(panel).tolist(), [[False, False], [False, False], [False, True], [False, False], [True, False], [False, False]])
        fire = self.ev.insider_clusters(panel)              # Roe on the 10th, Doe on the 13th: two buyers, $997,100
        self.assertEqual(fire[:, 0].tolist(), [0.0, 0.0, 0.0, 0.0, 997100.0, 0.0])

    def test_what_the_desk_is_shown(self):
        self.update()
        evs = self.shown()
        self.assertEqual(len(evs), 11)
        self.assertEqual([e.at for e in evs], sorted(e.at for e in evs))               # oldest first
        by = {(e.symbol, e.id): e for e in evs}
        results = by["NWT", RESULTS]
        self.assertEqual((results.kind, results.at, results.title, results.url, results.facts),
                         ("8-K", "2026-11-12T16:05:11-05:00", "8-K: results of operations (2.02), exhibits (9.01)", COVER, {"items": "2.02,9.01"}))
        self.assertEqual(by["NWT", OFFICER_8K].title, "8-K: officers or directors changed (5.02)")
        quarterly = by["NWT", "0009900001-26-000029"]
        self.assertEqual((quarterly.kind, quarterly.title, quarterly.facts), ("10-Q", "10-Q: quarterly report", {}))
        stake = by["NWT", "0002000001-26-000077"]           # EDGAR's "SCHEDULE 13G" under the name events.py knows
        self.assertEqual((stake.kind, stake.title), ("SC 13G", "SC 13G: a passive holder reports more than 5%"))
        self.assertEqual(stake.url, ARCH + "9900001/000200000126000077/xslSCHEDULE_13G_X01/primary_doc.xml")
        self.assertEqual(by["RVR.B", "0002000002-26-000015"].kind, "SC 13D/A")
        self.assertEqual(by["NWT", FORM4_MIXED].title, "4: DOE JANE (Chief Financial Officer) bought 5,000 shares at $40.10 ($200,500) "
                                                       "and sold 1,200 shares at $41.25 ($49,500)")
        self.assertEqual(by["NWT", FORM4_MIXED].url, ARCH + "9900001/000199999026000412/xslF345X06/form4.xml")     # the page for people
        self.assertEqual(by["NWT", FORM4_PAIR].title, "4: ROE RICHARD (Director) bought 20,000 shares at $39.83 ($796,600)")
        self.assertEqual(by["NWT", FORM4_SALE].title, "4: DOE JANE (Chief Financial Officer) sold 2,000 shares at $42.00 ($84,000)")
        # not shown: the prospectus (424B2), the amended 8-K, Riverbend's own Schedule 13G about another company,
        # and the Form 4 in which Riverbend is the buyer
        self.assertEqual({(e.symbol, e.kind) for e in evs if e.symbol != "NWT"},
                         {("RVR.A", "8-K"), ("RVR.A", "SC 13D/A"), ("RVR.B", "8-K"), ("RVR.B", "SC 13D/A")})
        self.assertEqual({e.kind for e in evs}, {"8-K", "10-Q", "4", "SC 13G", "SC 13D/A"})
        for e in evs:
            sec_url(e.url)
        # and events.recent picks by time as pipeline.py and the desk ask it to: after the close on the 12th, one Form 4
        self.assertEqual([e.id for e in self.ev.recent("2026-11-12T16:30:00", {"NWT"})], [FORM4_MIXED])

    def test_only_what_was_accepted_after_since(self):
        got = self.update(since="2026-11-12T16:05:11")       # the second the results were accepted is not after it
        self.assertEqual((got["earnings"], got["insiders"], got["recent"], got["form4"]), (0, 1, 1, 1))
        got = self.update(since="2026-11-12T21:05:10Z")      # one second earlier, said in UTC
        self.assertEqual((got["earnings"], got["recent"]), (1, 1))
        self.assertEqual(self.ev.earnings["accession"].tolist(), [RESULTS])

    def test_twice_adds_nothing_and_fetches_no_form_4_again(self):
        self.update()
        before = (len(self.ev.earnings), len(self.ev.insiders), len(self.recent_file()))
        self.assertEqual(before, (3, 3, 11))
        asked = len(self.web.asked)
        again = self.update()
        self.assertEqual((again["earnings"], again["insiders"], again["recent"]), (0, 0, 0))
        self.assertEqual((len(self.ev.earnings), len(self.ev.insiders), len(self.recent_file())), before)
        self.assertEqual(self.ev.insiders.duplicated(["accession", "sym", "owner", "price"]).sum(), 0)
        # the three Form 4s read the first time are not fetched again, and neither is the one on Riverbend's
        # list in which it is the buyer: its purchases are stored under Northwind, so it is known not to be Riverbend's
        self.assertEqual([u for u in self.web.asked[asked:] if u.endswith("form4.xml")], [])

    def test_a_month_of_filings_is_kept(self):
        self.update()
        filings.update(self.sec, self.ev, SYMS, "2026-11-13T18:00:00", now=datetime(2026, 12, 8, 18, 0, tzinfo=NY))
        self.assertEqual({e.id for e in self.shown()}, {FORM4_MIXED, RESULTS, FORM4_PAIR, OFFICER_8K, "0009900003-26-000044"})   # from 8 November 18:00
        filings.update(self.sec, self.ev, SYMS, "2026-12-08T18:00:00", now=datetime(2026, 12, 20, 18, 0, tzinfo=NY))
        self.assertEqual(self.shown(), [])
        self.assertEqual(len(self.recent_file()), 0)
        self.assertEqual(len(self.ev.earnings), 3)           # the tables the models read are not trimmed

    def test_the_whole_evening_keeps_the_pace(self):
        self.update()
        self.assertEqual(len(self.web.times), 7)             # the ticker list, three companies, three Form 4s
        self.assertNotIn(form4_url(9900003, FORM4_PAIR), self.web.asked)               # read once, under the company it is about
        self.assertGreaterEqual(min(b - a for a, b in zip(self.web.times, self.web.times[1:])), 0.13)

    def test_a_refusal_stops_the_evening_and_keeps_what_was_read(self):
        self.web.routes[SUBS + "CIK0009900003.json"] = (403, b"<title>Request Rate Threshold Exceeded</title>")
        self.web.routes[SUBS + "CIK0009900009.json"] = fixture("CIK0009900003.json")
        with self.assertRaises(Stopped) as c:
            self.update(symbols=SYMS + ["ELSW"])
        self.assertIn("403", str(c.exception))
        self.assertEqual(self.web.asked[-1], SUBS + "CIK0009900003.json")              # asked once, and nothing after it
        self.assertNotIn(SUBS + "CIK0009900009.json", self.web.asked)
        self.assertEqual((c.exception.result["companies"], c.exception.result["earnings"], c.exception.result["insiders"]), (1, 1, 3))
        self.assertEqual(self.ev.earnings["sym"].tolist(), ["NWT"])
        self.assertEqual(len(self.recent_file()), 7)

    def test_a_refusal_in_the_middle_of_a_companys_form_4s_stops_it_too(self):
        self.web.routes[form4_url(9900001, FORM4_PAIR)] = (429, b"Too Many Requests")
        with self.assertRaises(Stopped):
            self.update()
        self.assertEqual(self.web.asked[-1], form4_url(9900001, FORM4_PAIR))
        self.assertEqual(self.ev.insiders["owner"].tolist(), ["DOE JANE"])             # the Form 4 read before it is kept

    def test_one_form_4_that_cannot_be_read_is_a_failure_and_no_more(self):
        self.web.routes[form4_url(9900001, FORM4_PAIR)] = b"<html><body>temporarily unavailable</body></html>"
        got = self.update()
        self.assertEqual((got["earnings"], got["insiders"], got["recent"], got["form4"]), (3, 1, 10, 3))
        self.assertEqual([(f["sym"], f.get("accession")) for f in got["failures"]], [("NWT", FORM4_PAIR), ("GONE", None)])
        del self.web.routes[form4_url(9900001, FORM4_PAIR)]
        self.web.routes[form4_url(9900001, FORM4_PAIR)] = fixture("form4_two_owners.xml")
        again = self.update()                                # the next evening over the same stretch picks it up
        self.assertEqual((again["earnings"], again["insiders"], again["recent"]), (0, 2, 1))

    def test_when_nothing_answers_the_evening_gives_up_after_eight(self):
        names = {str(n): {"cik_str": 9900100 + n, "ticker": f"D{n:02d}", "title": f"Dead Line {n}"} for n in range(12)}
        web = Web({TICKERS: json.dumps(names).encode()}, self.clock)
        with self.assertRaises(Stopped) as c:
            filings.update(client(web), self.ev, [f"D{n:02d}" for n in range(12)], SINCE, log=self.said.append, now=NOW)
        self.assertIn("8 requests in a row failed", str(c.exception))
        self.assertEqual(len(web.asked), 1 + 8)
        self.assertEqual(len(c.exception.result["failures"]), 8)

    def test_without_the_ticker_list_nothing_is_tried(self):
        web = Web({TICKERS: (403, b"Forbidden")}, self.clock)
        with self.assertRaises(Stopped):
            filings.update(client(web), self.ev, SYMS, SINCE, now=NOW)
        self.assertEqual(web.asked, [TICKERS])


def busy_company(stamp) -> bytes:
    """A company that files twice every trading day for seven months, in the real shape: something at 08:00 New
    York time and its results at 16:30. `stamp` writes a New York wall-clock time the way the SEC is to have done."""
    rows = []
    for n, d in enumerate(reversed(trading_days(date(2026, 4, 1), date(2026, 11, 12)))):
        rows.append((f"0009900007-26-{2 * n + 1:06d}", d.isoformat(), stamp(datetime(d.year, d.month, d.day, 16, 30)), "2.02,9.01"))
        rows.append((f"0009900007-26-{2 * n:06d}", d.isoformat(), stamp(datetime(d.year, d.month, d.day, 8, 0)), "7.01"))
    cols = {"accessionNumber": [r[0] for r in rows], "filingDate": [r[1] for r in rows], "reportDate": [r[1] for r in rows],
            "acceptanceDateTime": [r[2] for r in rows], "act": ["34"] * len(rows), "form": ["8-K"] * len(rows),
            "fileNumber": ["001-39907"] * len(rows), "filmNumber": [""] * len(rows), "items": [r[3] for r in rows],
            "core_type": ["8-K"] * len(rows), "size": [1000] * len(rows), "isXBRL": [0] * len(rows), "isInlineXBRL": [0] * len(rows),
            "isXBRLNumeric": [0] * len(rows), "primaryDocument": ["busy8k.htm"] * len(rows), "primaryDocDescription": ["8-K"] * len(rows)}
    return json.dumps({"cik": "0009900007", "sic": "7372", "name": "Busy Corp", "tickers": ["BUSY"],
                       "filings": {"recent": cols, "files": []}}).encode()


class TheClock(Lab):
    """The tripwire for the day the SEC's times stop being UTC."""

    def busy(self, stamp) -> dict:
        web = Web({TICKERS: b'{"0":{"cik_str":9900007,"ticker":"BUSY","title":"Busy Corp"}}', SUBS + "CIK0009900007.json": busy_company(stamp)}, self.clock)
        return filings.update(client(web), self.ev, ["BUSY"], "2026-11-01T00:00:00", log=self.said.append, now=NOW)

    def test_times_that_are_utc_pass(self):
        got = self.busy(lambda t: t.replace(tzinfo=NY).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"))
        self.assertEqual(got["earnings"], 9)                 # the trading days from 2 to 12 November
        e = self.ev.earnings.sort_values("day")
        self.assertEqual(e["day"].tolist(), [20261103, 20261104, 20261105, 20261106, 20261109, 20261110, 20261111, 20261112, 20261113])
        self.assertEqual(e["at"].iloc[-1], "2026-11-12T16:30:00-05:00")                # each on the day after its 16:30

    def test_new_york_digits_under_a_z_stop_the_run_before_anything_is_saved(self):
        with self.assertRaises(Stopped) as c:                # the same filings, the wall-clock time written as if it were UTC
            self.busy(lambda t: t.strftime("%Y-%m-%dT%H:%M:%S.000Z"))
        self.assertIn("no longer look like UTC", str(c.exception))
        self.assertTrue(self.ev.earnings.empty)              # read as UTC, 16:30 would be 11:30 and every results day a day early
        self.assertFalse((self.tmp / "events" / "recent.parquet").exists())
        self.assertFalse((self.tmp / "events" / "earnings.parquet").exists())


class Backfill(Lab):
    TODAY = date(2026, 11, 13)
    Q3 = f"{WWW}/files/datastandardsinnovation{SETS}2026q3_form345.zip"

    def backfill(self, **more):
        return filings.backfill(self.sec, self.ev, SYMS, 2016, log=self.said.append, today=self.TODAY, **more)

    def test_results_days_back_to_the_start_year(self):
        got = self.backfill(tail=False)
        self.assertEqual((got["companies"], got["earnings"], got["no_cik"]), (2, 9, 1))
        e = self.ev.earnings
        nwt = e[e["sym"] == "NWT"].sort_values("day")
        self.assertEqual(nwt["day"].tolist(), [
            20190422,       # Thursday 18 April 2019, 16:10: Good Friday and a weekend later
            20250731,       # 07:45 on a Thursday: the same day
            20251107,       # 16:10 on a Thursday: Friday
            20260213,       # 16:00:00 exactly: the close was not after it
            20260526,       # Friday 22 May, 16:45: Tuesday, after Memorial Day
            20260806,       # 07:30 in summer
            20261113,
        ])
        self.assertEqual(nwt["at"].iloc[0], "2019-04-18T16:10:00-04:00")
        self.assertEqual(e[e["sym"] != "NWT"].sort_values("sym")[["sym", "day"]].values.tolist(), [["RVR.A", 20261111], ["RVR.B", 20261111]])
        self.assertIn(SUBS + "CIK0009900001-submissions-001.json", self.web.asked)
        self.assertNotIn(SUBS + "CIK0009900001-submissions-002.json", self.web.asked)   # it ends in 2015: not fetched, and its 2015 results not there

    def test_insider_purchases_from_the_quarterly_sets(self):
        got = self.backfill(tail=False)
        self.assertEqual((got["quarters"], got["insiders_through"], got["insiders"], got["form4"]), (2, "2026-09-30", 6, 0))
        self.assertEqual([f.get("quarter") or f.get("sym") for f in got["failures"]], ["2026q2", "GONE"])      # a broken zip, a company with no file
        self.assertNotIn(f"{WWW}/files/structureddata{SETS}2015q4_form345.zip", self.web.asked)                # before the start year
        i = self.ev.insiders.sort_values(["day", "sym", "price"]).reset_index(drop=True)
        doe = {"sym": "NWT", "day": 20260814, "at": "2026-08-14T10:20:00-04:00", "owner": "DOE JANE", "officer": True, "director": False,
               "ten_pct": False, "title": "Chief Financial Officer", "accession": FORM4_AUGUST}
        self.assertEqual(i.to_dict("records"), [
            # the set gives a filing date only; this Form 4 is in Northwind's own list, accepted 10:20 that Friday: the same day
            {**doe, "shares": 2000.0, "price": 35.5, "usd": 71000.0},
            {**doe, "shares": 1000.0, "price": 35.75, "usd": 35750.0},
            # this one is not in the list: dated Monday the 17th, so Tuesday. Two reporting owners, one buyer
            {"sym": "NWT", "day": 20260818, "at": "2026-08-17", "owner": "ROE RICHARD", "officer": False, "director": True, "ten_pct": True,
             "title": "Director", "shares": 10000.0, "price": 36.0, "usd": 360000.0, "accession": A + "305"},
            # Gone Corp's own list cannot be read: dated Friday 4 September, so Tuesday, after Labor Day
            {"sym": "GONE", "day": 20260908, "at": "2026-09-04", "owner": "LOW SAM", "officer": True, "director": True, "ten_pct": False,
             "title": "Chief Executive Officer", "shares": 4000.0, "price": 2.5, "usd": 10000.0, "accession": A + "330"},
            # filed under the ticker "RVR.A, RVR.B": matched by the issuer's CIK, for both share classes
            *({"sym": s, "day": 20260910, "at": "2026-09-09", "owner": "VANCE IDA", "officer": False, "director": True, "ten_pct": False,
               "title": "Director", "shares": 250.0, "price": 410.0, "usd": 102500.0, "accession": A + "335"} for s in ("RVR.A", "RVR.B")),
        ])                  # not there: the amendment, the Form 5, the sale, the award, the company that is not ours
        self.assertEqual([str(i[c].dtype) for c in ("day", "officer", "director", "ten_pct", "shares")], ["int64", "bool", "bool", "bool", "float64"])

    def test_the_weeks_since_the_last_set_are_read_form_4_by_form_4(self):
        got = self.backfill()
        self.assertEqual((got["insiders"], got["form4"]), (9, 3))                      # the pair's Form 4 once, not again for its buyer
        i = self.ev.insiders
        self.assertEqual(sorted(set(i["accession"])), [FORM4_AUGUST, A + "305", A + "330", A + "335", FORM4_PAIR, FORM4_MIXED])
        self.assertEqual(i[i["accession"] == FORM4_MIXED][["sym", "day", "owner", "usd"]].values.tolist(), [["NWT", 20261113, "DOE JANE", 200500.0]])
        self.assertEqual(set(i[i["accession"] == FORM4_PAIR]["sym"]), {"NWT"})         # Riverbend bought; Northwind was bought
        self.assertNotIn(form4_url(9900001, FORM4_AUGUST), self.web.asked)             # the set already had it
        self.assertTrue(any("since 2026-09-30" in line for line in self.said))

    def test_no_tail_when_the_sets_are_long_out_of_date(self):
        got = filings.backfill(self.sec, self.ev, SYMS, 2016, log=self.said.append, today=date(2027, 6, 1))
        self.assertEqual((got["insiders"], got["form4"]), (6, 0))

    def test_twice_adds_nothing(self):
        self.backfill()
        before = (len(self.ev.earnings), len(self.ev.insiders))
        self.assertEqual(before, (9, 9))
        again = self.backfill()
        self.assertEqual((again["earnings"], again["insiders"]), (0, 0))
        self.assertEqual((len(self.ev.earnings), len(self.ev.insiders)), before)

    def test_the_evening_after_the_history_adds_only_what_is_new(self):
        self.backfill()
        asked = len(self.web.asked)
        got = self.update(since="2026-08-01T00:00:00")
        self.assertEqual((got["earnings"], got["insiders"], got["recent"], got["form4"]), (0, 0, 11, 1))
        self.assertEqual(len(self.ev.insiders), 9)
        self.assertEqual([u for u in self.web.asked[asked:] if u.endswith("form4.xml")], [form4_url(9900001, FORM4_SALE)])   # the one not yet read
        by = {(e.symbol, e.id): e for e in self.shown()}
        self.assertEqual(by["NWT", FORM4_PAIR].title, "4: ROE RICHARD (Director) bought 20,000 shares at $39.83 ($796,600)")   # from the table
        self.assertNotIn(("RVR.A", FORM4_PAIR), by)

    def test_when_the_page_that_lists_the_sets_is_gone_each_quarter_is_tried_by_name(self):
        del self.web.routes[SETS_PAGE]
        got = filings.backfill(self.sec, self.ev, SYMS, 2026, log=self.said.append, today=self.TODAY, tail=False)
        self.assertEqual((got["quarters"], got["insiders"], got["insiders_through"]), (2, 6, "2026-09-30"))
        self.assertEqual([f["quarter"] for f in got["failures"] if "quarter" in f], ["the page that lists the sets", "2026q2"])
        self.assertIn(self.Q3, self.web.asked)

    def test_a_refusal_stops_the_history(self):
        self.web.routes[self.Q3] = (403, b"Request Rate Threshold Exceeded")
        with self.assertRaises(Stopped):
            self.backfill()
        self.assertEqual(self.web.asked[-1], self.Q3)
        self.assertNotIn(SUBS + "CIK0009900001.json", self.web.asked)
        self.assertTrue(self.ev.empty())


def gone_company() -> dict:
    """Oldcastle Works (CIK 9900005, ticker OLDC): bought out in the spring of 2026, so it is not in the SEC's
    ticker list. Its insiders filed three Form 4s under OLDC in the first quarter, one of them a purchase;
    its own list holds results from 2024, February 2026 and (a successor's paperwork) August 2026."""
    head = lambda t: fixture(f"form345/{t}.tsv").split(b"\n", 1)[0] + b"\n"       # noqa: E731
    sub = "".join(f"{A}{n}\t{d}\t{d}\t\t\t0\t\t\t4\t0009900005\tOldcastle Works Inc\tNYSE: OLDC\t\t0\n"
                  for n, d in (("101", "12-JAN-2026"), ("102", "20-JAN-2026"), ("103", "02-FEB-2026")))
    own = "".join(f"{A}{n}\t0001999005\tHALE MARY\tOfficer\tChief Executive Officer\t\t1 EXAMPLE WAY\t\tSPRINGFIELD\tIL\t62701\tILLINOIS\t\n"
                  for n in ("101", "102", "103"))
    buy = f"{A}102\t880101\tCommon Stock\t\t16-JAN-2026\t\t\t\t4\tP\t0\t\t\t\t3000\t\t50\t\tA\t\t9000\t\t\t\tD\t\t\t\n"
    quarter = insider_zip({"SUBMISSION": head("SUBMISSION") + sub.encode(), "REPORTINGOWNER": head("REPORTINGOWNER") + own.encode(),
                           "NONDERIV_TRANS": head("NONDERIV_TRANS") + buy.encode()})
    rows = [("0009900005-26-000009", "2026-08-05", "2026-08-05T20:10:00.000Z"),      # after its last insider filing by more than 120 days
            ("0009900005-26-000002", "2026-02-10", "2026-02-10T21:10:00.000Z"),      # 16:10 on a Tuesday
            ("0009900005-24-000007", "2024-06-03", "2024-06-03T20:10:00.000Z")]      # more than 400 days before the first
    cols = {"accessionNumber": [r[0] for r in rows], "filingDate": [r[1] for r in rows], "reportDate": [r[1] for r in rows],
            "acceptanceDateTime": [r[2] for r in rows], "act": ["34"] * 3, "form": ["8-K"] * 3, "fileNumber": ["001-39905"] * 3,
            "filmNumber": [""] * 3, "items": ["2.02,9.01"] * 3, "core_type": ["8-K"] * 3, "size": [1000] * 3, "isXBRL": [0] * 3,
            "isInlineXBRL": [0] * 3, "isXBRLNumeric": [0] * 3, "primaryDocument": ["oldc8k.htm"] * 3, "primaryDocDescription": ["8-K"] * 3}
    company = json.dumps({"cik": "0009900005", "sic": "3420", "name": "Oldcastle Works Inc", "tickers": [],
                          "filings": {"recent": cols, "files": []}}).encode()
    return {f"{WWW}/files/structureddata{SETS}2026q1_form345.zip": quarter, SUBS + "CIK0009900005.json": company}


class NoLongerListed(Lab):
    """The history is not one of survivors only: a company the ticker list has forgotten is found through
    the ticker its insiders filed under, and is that symbol only for the years it filed under it."""

    def test_a_company_since_bought_out_is_found_through_its_insiders_filings(self):
        self.web.routes.update(gone_company())
        got = filings.backfill(self.sec, self.ev, SYMS + ["OLDC"], 2016, log=self.said.append, today=date(2026, 11, 13), tail=False)
        self.assertEqual((got["former"], got["no_cik"], got["companies"]), (1, 1, 3))      # NOPE is still nobody
        e = self.ev.earnings
        self.assertEqual(e[e["sym"] == "OLDC"][["day", "accession"]].values.tolist(), [[20260211, "0009900005-26-000002"]])
        i = self.ev.insiders
        self.assertEqual(i[i["sym"] == "OLDC"][["day", "owner", "shares", "price", "accession"]].values.tolist(),
                         [[20260121, "HALE MARY", 3000.0, 50.0, A + "102"]])       # filed Tuesday the 20th, no time known: Wednesday
        self.assertTrue(any("no longer listed" in line for line in self.said))

    def test_links_follow_who_held_a_ticker_and_when(self):
        rows = [
            # cik, sym, first, last, n
            (11, "AAA", "2016-01-20", "2026-10-30", 400),      # listed today, always under AAA
            (22, "GONE", "2016-02-01", "2019-05-10", 90),      # bought out in 2019
            (33, "LATE", "2021-03-01", "2023-06-30", 40),      # listed 2021, delisted 2023
            (44, "TYPO", "2020-01-01", "2020-01-01", 2),       # two filings: someone's slip of the pen
            (55, "DOW", "2016-01-15", "2017-08-31", 200),      # the old holder of a ticker …
            (56, "DOW", "2019-03-20", "2026-11-02", 300),      # … and today's, a company that did not exist before
            (66, "OLDT", "2016-01-10", "2021-05-30", 250),     # renamed in 2021: the same company …
            (66, "NEWT", "2021-06-02", "2026-11-05", 220),     # … under its new ticker
            (77, "NEWT", "2016-03-01", "2018-02-01", 30),      # someone else had NEWT while that company was OLDT
        ]
        issuers = pd.DataFrame(rows, columns=["cik", "sym", "first", "last", "n"])
        current = {"AAA": 11, "DOW": 56, "NEWT": 66, "FPI": 88}
        got = filings.links(["AAA", "GONE", "LATE", "TYPO", "DOW", "NEWT", "OLDT", "FPI", "NOPE"], current, issuers, "2016-01-01")
        ever, always = filings.EVER, filings.ALWAYS
        self.assertEqual(got, {
            11: [("AAA", ever, always)],
            22: [("GONE", ever, "2019-09-07")],                # filing when the history begins: from the start; 120 days past the last
            33: [("LATE", "2020-01-26", "2023-10-28")],        # 400 days before its first
            55: [("DOW", "2014-12-11", "2017-12-29")],         # the old company keeps its own years
            56: [("DOW", ever, always)],
            66: [("NEWT", ever, always), ("OLDT", ever, "2021-09-27")],   # today's company is NEWT all the way back
            88: [("FPI", ever, always)],                       # never in the sets (a foreign company): the ticker list is all there is
        })                                                     # not 44 (too few filings), not 77 (the bars of those years are company 66's)


class Sectors(Lab):
    def test_the_industry_code_of_each_symbol(self):
        self.assertEqual(filings.sectors(self.sec, SYMS, log=self.said.append), {"NWT": 3540, "RVR.A": 6331, "RVR.B": 6331})
        self.assertEqual(len(self.said), 1)
        self.assertIn("GONE", self.said[0])
        self.assertNotIn(SUBS + "CIK0009900001-submissions-001.json", self.web.asked)   # one request a company

    def test_a_refusal_is_not_swallowed(self):
        self.web.routes[SUBS + "CIK0009900003.json"] = (403, b"Forbidden")
        with self.assertRaises(SecError):
            filings.sectors(self.sec, SYMS, log=self.said.append)


if __name__ == "__main__":
    unittest.main()
