"""The paper book against cases worked by hand.

    docker run --rm -v "$PWD":/app -w /app -e PYTHONPATH=/app:/app/tests plint/stocks:dev python -m unittest tests.test_book -v
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from stocks.book import BAND, STALE_DAYS, Book, replay
from stocks.panel import Panel
from stocks.types import Order


class Flat:
    """Ten basis points on everything, so the sums can be done by hand."""

    def rate(self, t, j, usd):
        return 0.001


def panel(rows: dict[str, list[tuple]]) -> Panel:
    """rows[sym] = [(day, open, close, total return), ...]; a missing day is a day without a bar."""
    out = []
    for sym, bars in rows.items():
        for d, o, c, ret in bars:
            out.append((d, sym, o, max(o, c), min(o, c), c, 1e6, (o + c) / 2, 1000, ret))
    return Panel.from_frame(pd.DataFrame(out, columns=["d", "sym", "o", "h", "l", "c", "v", "vw", "n", "ret"]), "SPY")


DAYS = [20240102, 20240103, 20240104, 20240105, 20240108]
SPY = [(d, 100.0, 100.0, 0.0) for d in DAYS]


class BookArithmetic(unittest.TestCase):
    def test_buy_at_the_open_then_the_session(self):
        p = panel({"SPY": SPY, "A": [(DAYS[0], 100, 100, np.nan), (DAYS[1], 101, 102, 0.02)]})
        b = Book(10_000.0)
        fills = b.step(p, 1, [Order("A", 0.5, "test")], Flat())
        self.assertEqual(len(fills), 1)
        self.assertAlmostEqual(fills[0].usd, 5000.0)
        self.assertAlmostEqual(fills[0].price, 101.0)
        self.assertAlmostEqual(fills[0].cost, 5.0)
        self.assertAlmostEqual(b.cash, 10_000 - 5000 - 5.0)
        self.assertAlmostEqual(b.pos["A"].value, 5000 * 102 / 101, delta=0.01)      # bought at 101, closed at 102
        self.assertAlmostEqual(b.pos["A"].basis, 5005.0)

    def test_a_split_is_not_a_loss(self):
        # day 2: 2-for-1. The raw price halves; the total return says the stock rose 1 %.
        a = [(DAYS[0], 100, 100, np.nan), (DAYS[1], 100, 100, 0.0), (DAYS[2], 50.25, 50.5, 0.01)]
        p = panel({"SPY": SPY, "A": a})
        b = Book(10_000.0)
        b.step(p, 1, [Order("A", 1.0, "")], Flat())
        before = b.pos["A"].value
        b.step(p, 2, [], Flat())
        self.assertAlmostEqual(b.pos["A"].value / before, 1.01, places=5)

    def test_a_holding_follows_its_company_to_a_new_ticker(self):
        # OLD trades on days 0-2 and is NEW from day 3: NEW's bars are the company's whole history, as Alpaca gives them
        old = [(DAYS[0], 100, 100, np.nan), (DAYS[1], 100, 100, 0.0), (DAYS[2], 100, 102, 0.02)]
        new = old + [(DAYS[3], 103, 104, 104 / 102 - 1), (DAYS[4], 104, 105, 105 / 104 - 1)]
        p = panel({"SPY": SPY, "OLD": old, "NEW": new})
        b = Book(10_000.0, renamed={"OLD": "NEW"})
        b.step(p, 1, [Order("OLD", 0.5, "")], Flat())
        b.step(p, 2, [], Flat())
        value, basis = b.pos["OLD"].value, b.pos["OLD"].basis
        fills = b.step(p, 3, [], Flat())
        self.assertEqual((fills, list(b.pos)), ([], ["NEW"]))                      # nothing bought or sold: the same shares
        self.assertAlmostEqual(b.pos["NEW"].value, value * 104 / 102, delta=0.01)  # and the day's move is the company's
        self.assertAlmostEqual(b.pos["NEW"].basis, basis)
        self.assertEqual(b.pos["NEW"].days, 2)                                     # two nights held: the count goes on
        self.assertTrue(any("OLD is now NEW" in n for n in b.notes))
        # without the new name on record the old one goes quiet, as before
        c = Book(10_000.0)
        c.step(p, 1, [Order("OLD", 0.5, "")], Flat())
        c.step(p, 2, [], Flat())
        c.step(p, 3, [], Flat())
        self.assertEqual((list(c.pos), c.pos["OLD"].stale), (["OLD"], 1))

    def test_a_dividend_is_kept(self):
        # day 2: goes ex a 2 % dividend. Raw close falls 2 %; total return is flat.
        a = [(DAYS[0], 100, 100, np.nan), (DAYS[1], 100, 100, 0.0), (DAYS[2], 98, 98, 0.0)]
        p = panel({"SPY": SPY, "A": a})
        b = Book(10_000.0)
        b.step(p, 1, [Order("A", 1.0, "")], Flat())
        before = b.equity()
        b.step(p, 2, [], Flat())
        self.assertAlmostEqual(b.equity(), before, places=4)

    def test_sell_all_and_what_it_made(self):
        a = [(DAYS[0], 100, 100, np.nan), (DAYS[1], 100, 100, 0.0), (DAYS[2], 110, 110, 0.10)]
        p = panel({"SPY": SPY, "A": a})
        b = Book(10_000.0)
        b.step(p, 1, [Order("A", 0.5, "in")], Flat())
        fills = b.step(p, 2, [Order("A", 0.0, "out")], Flat())
        self.assertNotIn("A", b.pos)
        f = fills[0]
        self.assertEqual(f.side, "sell")
        self.assertAlmostEqual(f.usd, 5500.0, delta=0.01)
        self.assertAlmostEqual(f.ret, (5500 - 5.5) / 5005 - 1, places=5)        # proceeds less costs over what was paid
        self.assertAlmostEqual(b.cash, 4995 + 5500 - 5.5, delta=0.01)
        self.assertAlmostEqual(b.costs, 5.0 + 5.5, places=4)

    def test_buys_never_spend_more_than_the_cash(self):
        a = [(DAYS[0], 100, 100, np.nan), (DAYS[1], 100, 100, 0.0)]
        p = panel({"SPY": SPY, "A": a, "B": a})
        b = Book(10_000.0)
        b.step(p, 1, [Order("A", 0.8, ""), Order("B", 0.7, "")], Flat())
        self.assertGreaterEqual(b.cash, -1e-6)
        self.assertAlmostEqual(b.pos["A"].value / b.pos["B"].value, 0.8 / 0.7, places=6)   # scaled alike
        self.assertAlmostEqual(b.equity(), 10_000 - b.costs, places=4)

    def test_sells_fund_buys_at_the_same_open(self):
        a = [(DAYS[0], 100, 100, np.nan), (DAYS[1], 100, 100, 0.0), (DAYS[2], 100, 100, 0.0)]
        p = panel({"SPY": SPY, "A": a, "B": a})
        b = Book(10_000.0)
        b.step(p, 1, [Order("A", 1.0, "")], Flat())
        b.step(p, 2, [Order("B", 1.0, ""), Order("A", 0.0, "")], Flat())      # order given buy first: sells still go first
        self.assertNotIn("A", b.pos)
        self.assertGreater(b.pos["B"].value, 9900)

    def test_no_bar_no_fill(self):
        p = panel({"SPY": SPY, "A": [(DAYS[0], 100, 100, np.nan), (DAYS[2], 100, 100, 0.0)]})
        b = Book(10_000.0)
        fills = b.step(p, 1, [Order("A", 0.5, "")], Flat())
        self.assertEqual(fills, [])
        self.assertEqual(b.cash, 10_000.0)
        self.assertTrue(any("not filled" in n for n in b.notes))

    def test_a_halt_holds_the_value_and_the_gap_arrives_on_return(self):
        a = [(DAYS[0], 100, 100, np.nan), (DAYS[1], 100, 100, 0.0), (DAYS[3], 80, 80, -0.20)]      # no bar on day 2
        p = panel({"SPY": SPY, "A": a})
        b = Book(10_000.0)
        b.step(p, 1, [Order("A", 0.5, "")], Flat())
        v = b.pos["A"].value
        b.step(p, 2, [], Flat())
        self.assertAlmostEqual(b.pos["A"].value, v)
        self.assertEqual(b.pos["A"].stale, 1)
        b.step(p, 3, [], Flat())
        self.assertAlmostEqual(b.pos["A"].value, v * 0.8, delta=0.01)
        self.assertEqual(b.pos["A"].stale, 0)

    def test_a_stock_that_stops_trading_is_closed_at_its_last_value(self):
        days = pd.bdate_range("2024-01-02", periods=STALE_DAYS + 3).strftime("%Y%m%d").astype(int).tolist()
        p = panel({"SPY": [(d, 100.0, 100.0, 0.0) for d in days],
                   "A": [(days[0], 100, 100, np.nan), (days[1], 100, 100, 0.0)]})
        b = Book(10_000.0)
        b.step(p, 1, [Order("A", 0.5, "")], Flat())
        for t in range(2, len(days)):
            b.step(p, t, [], Flat())
        self.assertNotIn("A", b.pos)
        self.assertAlmostEqual(b.cash, 10_000 - 5.0, places=4)
        self.assertIn("no price", b.fills[-1].why)

    def test_a_collapse_then_silence_is_written_off_unless_it_was_a_buy_out(self):
        days = pd.bdate_range("2024-01-02", periods=STALE_DAYS + 6).strftime("%Y%m%d").astype(int).tolist()
        crash = [(days[0], 100, 100, np.nan), (days[1], 100, 100, 0.0), (days[2], 100, 100, 0.0), (days[3], 60, 55, -0.45)]
        calm = [(days[0], 100, 100, np.nan), (days[1], 100, 100, 0.0), (days[2], 100, 100, 0.0), (days[3], 100, 101, 0.01)]
        p = panel({"SPY": [(d, 100.0, 100.0, 0.0) for d in days], "BANK": crash, "DEAL": calm, "TOOK": crash})
        b = Book(30_000.0, bought_out=frozenset({"TOOK"}))
        b.step(p, 1, [Order("BANK", 1 / 3, ""), Order("DEAL", 1 / 3, ""), Order("TOOK", 1 / 3, "")], Flat())
        for t in range(2, len(days)):
            b.step(p, t, [], Flat())
        self.assertEqual(b.pos, {})
        by = {f.sym: f for f in b.fills if f.side == "sell"}
        self.assertEqual(by["BANK"].usd, 0.0)                      # fell 45 % and went silent: nothing came back
        self.assertEqual(by["BANK"].ret, -1.0)
        self.assertIn("written off", by["BANK"].why)
        each = 30_000 / 3 / 1.001                                  # three equal buys, scaled so the costs fit in the cash
        self.assertAlmostEqual(by["DEAL"].usd, each * 1.01, delta=1.0)       # quiet without a fall: its last price
        self.assertAlmostEqual(by["TOOK"].usd, each * 0.55, delta=1.0)       # fell, but known to be a buy-out: the deal price
        self.assertAlmostEqual(b.cash, by["DEAL"].usd + by["TOOK"].usd, delta=1.0)

    def test_a_change_inside_the_band_is_left_alone(self):
        a = [(DAYS[0], 100, 100, np.nan), (DAYS[1], 100, 100, 0.0), (DAYS[2], 100, 100, 0.0)]
        p = panel({"SPY": SPY, "A": a})
        b = Book(10_000.0)
        b.step(p, 1, [Order("A", 0.5, "")], Flat())
        fills = b.step(p, 2, [Order("A", 0.5 + BAND / 2, "")], Flat())
        self.assertEqual(fills, [])

    def test_replay_starts_in_cash_and_fills_the_day_after_the_decision(self):
        a = [(d, 100.0, 100.0, 0.0 if i else np.nan) for i, d in enumerate(DAYS)]
        p = panel({"SPY": SPY, "A": a})
        rec = replay(p, {DAYS[1]: [Order("A", 1.0, "")]}, 10_000.0, Flat(), DAYS[1])
        self.assertEqual(rec.days, DAYS[1:])
        self.assertEqual(rec.equity[0], 10_000.0)                  # the evening it decided: still cash
        self.assertEqual(rec.n_pos, [0, 1, 1, 1])
        self.assertEqual(rec.book.fills[0].d, DAYS[2])
        again = replay(p, {DAYS[1]: [Order("A", 1.0, "")]}, 10_000.0, Flat(), DAYS[1])
        self.assertEqual(rec.equity, again.equity)


if __name__ == "__main__":
    unittest.main()
