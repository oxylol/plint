"""The two tables from filings as the models see them, and the two models that trade on them.

    docker run --rm -v "$PWD":/app -w /app -e PYTHONPATH=/app:/app/tests plint/stocks:dev python -m unittest tests.test_events -v
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import synth
from stocks import backtest
from stocks.events import EARNINGS_COLS, INSIDER_COLS, Events
from stocks.models import Env
from stocks.models.events import Drift, Insiders
from stocks.models.rules import Index, Reversal
from stocks.panel import FIELDS, Panel
from stocks.types import BookView


def buy(sym, day, owner, usd, officer=True, director=False, acc=None):
    return {"sym": sym, "day": day, "at": "", "owner": owner, "officer": officer, "director": director, "ten_pct": False,
            "title": "CFO", "shares": usd / 50.0, "price": 50.0, "usd": usd, "accession": acc or f"{sym}-{day}-{owner}"}


class Tables(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = synth.panel(30, 420, seed=9)
        cls.d = [int(x) for x in cls.p.days]

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="stocks-events-"))
        self.ev = Events(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_results_days_and_the_days_after(self):
        rows = pd.DataFrame([{"sym": "SAAB", "day": self.d[300], "at": "", "form": "8-K", "accession": "a1"},
                             {"sym": "SAAC", "day": self.d[310], "at": "", "form": "8-K", "accession": "a2"},
                             {"sym": "NOPE", "day": self.d[310], "at": "", "form": "8-K", "accession": "a3"}], columns=EARNINGS_COLS)
        self.assertEqual(self.ev.add("earnings", rows), 3)
        self.assertEqual(self.ev.add("earnings", rows), 0)                     # the same filings again add nothing
        table = self.ev.earnings_table(self.p)
        self.assertEqual(int(table.sum()), 2)                                   # a symbol the panel does not hold is ignored
        self.assertTrue(table[300, self.p.ix["SAAB"]])
        within = self.ev.earnings_within(self.p, 5)
        j = self.p.ix["SAAB"]
        self.assertEqual([bool(within[t, j]) for t in (299, 300, 304, 305)], [False, True, True, False])

    def test_a_cluster_needs_two_people_and_enough_money_and_fires_once(self):
        d = self.d
        rows = pd.DataFrame([
            buy("SAAB", d[200], "Ann", 80_000), buy("SAAB", d[205], "Ann", 80_000),          # one person twice: no cluster
            buy("SAAB", d[210], "Bo", 30_000),                                               # second person: now it forms
            buy("SAAB", d[212], "Cy", 30_000),                                               # still formed: no second firing
            buy("SAAC", d[200], "Dee", 20_000), buy("SAAC", d[201], "Eli", 20_000),          # two people, too little money
            buy("SAAD", d[200], "Fay", 300_000, officer=False), buy("SAAD", d[201], "Gus", 300_000, officer=False),  # ten-percent owners only
            buy("SAAE", d[200], "Hal", 60_000), buy("SAAE", d[260], "Ida", 60_000),          # two people, months apart
        ], columns=INSIDER_COLS)
        self.ev.add("insiders", rows)
        fire = self.ev.insider_clusters(self.p, window=21, min_buyers=2, min_usd=100_000)
        hits = {(self.p.syms[j], t) for t, j in zip(*np.nonzero(fire))}
        self.assertEqual(hits, {("SAAB", 210)})
        self.assertAlmostEqual(float(fire[210, self.p.ix["SAAB"]]), 190_000.0)
        loose = self.ev.insider_clusters(self.p, officers_only=False)
        self.assertGreater(loose[201, self.p.ix["SAAD"]], 0)

    def test_a_filing_dated_on_a_closed_day_counts_from_the_next_open_one(self):
        saturday = 20190105
        self.assertNotIn(saturday, self.d)
        self.ev.add("earnings", pd.DataFrame([{"sym": "SAAB", "day": saturday, "at": "", "form": "8-K", "accession": "w"}], columns=EARNINGS_COLS))
        t = int(np.flatnonzero(self.ev.earnings_table(self.p)[:, self.p.ix["SAAB"]])[0])
        self.assertEqual(self.d[t], 20190107)


class EventModels(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="stocks-events-"))
        base = synth.panel(40, 480, seed=12)
        data = {f: base.data[f].copy() for f in FIELDS}
        self.jump_t, self.jump_j = 400, base.ix["SABA"]
        # SABA reports on day 400: +9 % on four times its usual volume
        for f in ("o", "h", "l", "c"):
            data[f][self.jump_t:, self.jump_j] *= 1.09
        data["ret"][self.jump_t, self.jump_j] = (1 + base.ret[self.jump_t, self.jump_j]) * 1.09 - 1
        data["v"][self.jump_t, self.jump_j] *= 4.0
        # SABB reports the same day and does nothing; SABC jumps on day 405 with no results out
        j12 = base.ix["SABC"]
        for f in ("o", "h", "l", "c"):
            data[f][405:, j12] *= 1.09
        data["ret"][405, j12] = (1 + base.ret[405, j12]) * 1.09 - 1
        data["v"][405, j12] *= 4.0
        self.p = Panel(base.days, base.syms, data, base.member)
        self.ev = Events(self.tmp)
        d = [int(x) for x in self.p.days]
        self.ev.add("earnings", pd.DataFrame([{"sym": "SABA", "day": d[400], "at": "", "form": "8-K", "accession": "e1"},
                                              {"sym": "SABB", "day": d[400], "at": "", "form": "8-K", "accession": "e2"}], columns=EARNINGS_COLS))
        self.ev.add("insiders", pd.DataFrame([buy("SACA", d[380], "Ann", 90_000), buy("SACA", d[384], "Bo", 90_000)], columns=INSIDER_COLS))
        self.env = Env(index="SPY", cash="BIL", events=self.ev)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_drift_buys_the_strong_reaction_and_only_that(self):
        m = Drift()
        m.prepare(self.p, self.env)
        hits = {(self.p.syms[j], t) for t, j in zip(*np.nonzero(m.fire))}
        self.assertEqual(hits, {("SABA", 400)})
        d = m.decide(400, BookView(10_000.0, 1.0))
        self.assertEqual([(o.symbol, o.weight) for o in d.orders], [("SABA", 1 / 40)])
        self.assertIn("on results", d.orders[0].why)
        self.assertEqual(d.meta["signals"][0]["symbol"], "SABA")
        self.assertEqual(d.meta["signals"][0]["horizon"], 20)
        self.assertEqual(m.decide(399, BookView(10_000.0, 1.0)).orders, [])       # the evening before: nothing known yet

    def test_the_book_holds_forty_days_then_sells(self):
        out = backtest.run(self.p, [Index(), Drift(), Insiders()], self.env, 370, self.p.T - 1)
        drift = out["drift"].record.book
        buys = [f for f in drift.fills if f.side == "buy"]
        sells = [f for f in drift.fills if f.side == "sell"]
        self.assertEqual([(f.sym, f.d) for f in buys], [("SABA", int(self.p.days[401]))])     # filled at the next open
        self.assertEqual(len(sells), 1)
        held = int(np.searchsorted(self.p.days, sells[0].d)) - 401
        self.assertEqual(held, 40)
        ins = out["insiders"].record.book
        self.assertEqual([(f.sym, f.d) for f in ins.fills if f.side == "buy"], [("SACA", int(self.p.days[385]))])
        self.assertEqual([(f.sym, f.d) for f in ins.fills if f.side == "sell"], [("SACA", int(self.p.days[445]))])   # sixty sessions on
        self.assertIn("insiders bought $180,000", ins.fills[0].why)

    def test_without_filings_they_wait_and_say_why(self):
        m = Drift()
        m.prepare(self.p, Env(index="SPY", cash="BIL"))
        self.assertIn("SEC_CONTACT", m.waiting())
        self.assertEqual(m.decide(400, BookView(10_000.0, 1.0)).orders, [])
        empty = Events(self.tmp / "none")
        i = Insiders()
        i.prepare(self.p, Env(index="SPY", cash="BIL", events=empty))
        self.assertIn("first insider filings", i.waiting())

    def test_reversal_leaves_fresh_results_alone(self):
        with_ev, without = Reversal(), Reversal()
        with_ev.prepare(self.p, self.env)
        without.prepare(self.p, Env(index="SPY", cash="BIL"))
        j = self.p.ix["SABB"]
        self.assertTrue(np.isnan(with_ev.S[402, j]))
        self.assertTrue(np.isfinite(without.S[402, j]))
        self.assertTrue(np.isfinite(with_ev.S[410, j]))


if __name__ == "__main__":
    unittest.main()
