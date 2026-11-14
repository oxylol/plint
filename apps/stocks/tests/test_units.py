"""The small parts: the calendar, which names are companies, the holding rule, the ledger's lamp, the fitters.

    docker run --rm -v "$PWD":/app -w /app -e PYTHONPATH=/app:/app/tests plint/stocks:dev python -m unittest tests.test_units -v
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

import synth
from stocks import cal, ledger, metrics
from stocks.agent.prices import cost_usd
from stocks.learn import dataset, fit, net, ridge, trees
from stocks.portfolio import TopK
from stocks.store import Store
from stocks.types import BookView, Decision, Holding, Order, Signal
from stocks.universe import is_common, short_name


class Calendar(unittest.TestCase):
    def test_known_days(self):
        closed = [date(2024, 3, 29), date(2023, 1, 2), date(2022, 6, 20), date(2021, 7, 5), date(2025, 1, 9),
                  date(2024, 11, 28), date(2026, 12, 25), date(2026, 7, 3), date(2027, 1, 1), date(2024, 1, 15)]
        opened = [date(2021, 12, 31), date(2024, 11, 29), date(2026, 11, 13), date(2021, 6, 18), date(2026, 12, 24)]
        for d in closed:
            self.assertFalse(cal.is_trading_day(d), d)
        for d in opened:
            self.assertTrue(cal.is_trading_day(d), d)
        self.assertEqual([len(cal.trading_days(date(y, 1, 1), date(y, 12, 31))) for y in (2019, 2020, 2021, 2022, 2023, 2024)],
                         [252, 253, 252, 251, 250, 252])

    def test_the_run_clock(self):
        fri_evening = datetime(2026, 11, 13, 21, 30, tzinfo=timezone.utc)       # 16:30 New York, a Friday
        self.assertEqual(cal.last_close(fri_evening, "16:25"), date(2026, 11, 13))
        self.assertEqual(cal.last_close(datetime(2026, 11, 13, 21, 0, tzinfo=timezone.utc), "16:25"), date(2026, 11, 12))
        self.assertEqual(cal.last_close(datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc), "16:25"), date(2026, 11, 13))
        nxt = cal.next_run(fri_evening, "16:25")
        self.assertEqual((nxt.date(), nxt.hour, nxt.minute), (date(2026, 11, 16), 16, 25))
        self.assertEqual(cal.days_left_in_month(date(2026, 11, 13)), 11)
        self.assertEqual(cal.days_in_month(date(2026, 11, 13)), 20)          # 21 weekdays less Thanksgiving
        self.assertEqual(cal.prev_trading_day(date(2026, 11, 16)), date(2026, 11, 13))


class Names(unittest.TestCase):
    def test_companies_in_funds_and_paper_out(self):
        yes = [("AAPL", "Apple Inc. Common Stock", "NASDAQ"), ("XOM", "Exxon Mobil Corporation", "NYSE"),
               ("BRK.B", "Berkshire Hathaway Inc. Class B", "NYSE"), ("TSM", "Taiwan Semiconductor Manufacturing Company Ltd. American Depositary Shares", "NYSE"),
               ("IVZ", "Invesco Ltd Common Stock", "NYSE"), ("EPD", "Enterprise Products Partners L.P. Common Units", "NYSE"),
               ("VNO", "Vornado Realty Trust Common Stock", "NYSE"), ("BLK", "BlackRock, Inc. Common Stock", "NYSE"),
               ("WT", "WisdomTree, Inc. Common Stock", "NYSE"), ("UCTT", "Ultra Clean Holdings, Inc. Common Stock", "NASDAQ"),
               ("PFBC", "Preferred Bank Common Stock", "NASDAQ"),
               ("SBNY", "Signature Bank Common Stock", "OTC")]       # pushed off its exchange: its exchange years count
        no = [("SPY", "SPDR S&P 500 ETF Trust", "ARCA"), ("QQQ", "Invesco QQQ Trust, Series 1", "NASDAQ"),
              ("TQQQ", "ProShares UltraPro QQQ", "NASDAQ"), ("TLT", "iShares 20+ Year Treasury Bond ETF", "NASDAQ"),
              ("IBIT", "iShares Bitcoin Trust ETF", "NASDAQ"), ("NVDL", "GraniteShares 2x Long NVDA Daily ETF", "NASDAQ"),
              ("ABCDW", "Abcd Acquisition Corp. Warrant", "NASDAQ"), ("ABCDU", "Abcd Acquisition Corp. Unit", "NASDAQ"),
              ("BAC.PRB", "Bank of America Corporation Depositary Shares each representing a 1/1000th interest in a share of Preferred Stock", "NYSE"),
              ("XYZ", "XYZ Corp 6.50% Notes due 2029", "NYSE"), ("AAPL", "Apple Inc. Common Stock", "ARCA"),
              ("TCEHY", "Tencent Holdings Ltd Unsponsored ADR", "OTC"), ("NONAME", "", "OTC"),
              ("JEPQ", "JPMorgan Nasdaq Equity Premium Income ETF", "NASDAQ"), ("NUV", "Nuveen Municipal Value Fund, Inc.", "NYSE")]
        for sym, name, ex in yes:
            self.assertTrue(is_common(sym, name, ex), name)
        for sym, name, ex in no:
            self.assertFalse(is_common(sym, name, ex), name)
        self.assertEqual(short_name("Alphabet Inc. Class A Common Stock"), "Alphabet Inc.")
        self.assertEqual(short_name("Exxon Mobil Corporation"), "Exxon Mobil Corporation")


def view(holdings: dict[str, tuple[float, int]]) -> BookView:
    return BookView(10_000.0, 0.0, {s: Holding(s, w, d, "2026-01-02", 0.0) for s, (w, d) in holdings.items()})


class HoldingRule(unittest.TestCase):
    ranked = [f"S{i:03d}" for i in range(100)]

    def test_an_empty_book_is_filled_at_once(self):
        orders = TopK(k=10).orders(self.ranked, BookView(10_000.0, 1.0))
        self.assertEqual([o.symbol for o in orders], self.ranked[:10])
        self.assertTrue(all(abs(o.weight - 0.1) < 1e-9 for o in orders))
        self.assertEqual(orders[0].why, "rank 1 of 100")

    def test_a_holding_stays_while_it_ranks_well_and_few_change_a_day(self):
        top = TopK(k=10, hold_frac=0.3, max_new=2)
        held = {s: (0.1, 5) for s in ["S000", "S005", "S020", "S029", "S031", "S050", "S070", "S090", "S002", "S003"]}
        orders = top.orders(self.ranked, view(held))
        sells = [o.symbol for o in orders if o.weight == 0]
        buys = [o.symbol for o in orders if o.weight > 0]
        self.assertEqual(sells, ["S090", "S070"])                     # the two worst of the four below rank 30
        self.assertEqual(buys, ["S001", "S004"])                      # the best it does not hold
        self.assertIn("fell to rank 91 of 100", [o.why for o in orders])

    def test_a_stock_that_is_no_longer_ranked_goes_first(self):
        top = TopK(k=4, hold_frac=0.5, max_new=1)
        orders = top.orders(self.ranked, view({"GONE": (0.25, 3), "S080": (0.25, 3), "S001": (0.25, 3), "S002": (0.25, 3)}))
        self.assertEqual([o.symbol for o in orders if o.weight == 0], ["GONE"])      # the one swap goes to the forced sale

    def test_held_too_long_and_grown_too_big(self):
        top = TopK(k=10, hold_frac=0.5, max_new=3, max_hold=7)
        orders = top.orders(self.ranked, view({"S015": (0.1, 6), "S001": (0.1, 9), "S002": (0.25, 2), "S003": (0.1, 1)}))
        by = {o.symbol: o for o in orders}
        self.assertEqual(by["S015"].weight, 0.0)                      # seven sessions, and ranked 16th of a book of 10
        self.assertIn("held 7 days", by["S015"].why)
        self.assertNotIn("S001", by)                                  # held longer still, but ranked 2nd: it stays
        self.assertAlmostEqual(by["S002"].weight, 0.1)                # cut back to its share
        self.assertNotIn("S003", by)

    def test_a_book_bought_on_one_day_does_not_leave_on_one_day(self):
        top = TopK(k=10, hold_frac=0.5, max_new=2, max_hold=5)
        held = {f"S{i:03d}": (0.1, 4) for i in range(20, 30)}         # ten names, all five sessions old, none a top pick now
        orders = top.orders(self.ranked, view(held))
        self.assertEqual([o.symbol for o in orders if o.weight == 0], ["S029", "S028"])     # the two worst go; the rest wait their turn
        self.assertEqual([o.symbol for o in orders if o.weight > 0], ["S000", "S001"])      # and the book stays full

    def test_monthly_models_wait_for_the_month(self):
        top = TopK(every="month")
        self.assertFalse(top.due(20261113, 20261112, empty=False))
        self.assertTrue(top.due(20261201, 20261130, empty=False))
        self.assertTrue(top.due(20261113, 20261112, empty=True))
        week = TopK(every="week")
        self.assertTrue(week.due(20261116, 20261113, empty=False))
        self.assertFalse(week.due(20261117, 20261116, empty=False))


class Ledger(unittest.TestCase):
    def test_rank_correlation(self):
        a = np.arange(50, dtype=float)
        self.assertAlmostEqual(ledger.spearman(a, a ** 3)[0], 1.0)
        self.assertAlmostEqual(ledger.spearman(a, -a)[0], -1.0)
        b = a.copy()
        b[:40] = np.nan
        self.assertTrue(np.isnan(ledger.spearman(a, b)[0]))           # ten pairs: too few to mean anything

    def test_overlapping_outcomes_widen_the_error(self):
        rng = np.random.default_rng(0)
        iid = rng.normal(0.02, 0.1, 400)
        smooth = np.convolve(rng.normal(0.02, 0.1, 420), np.ones(5) / 5, "valid")[:400] * np.sqrt(5)
        _, se0, _, _ = ledger.nw_t(smooth, 0)
        _, se4, _, _ = ledger.nw_t(smooth, 4)
        self.assertGreater(se4, se0 * 1.5)
        _, a, _, _ = ledger.nw_t(iid, 0)
        _, b, _, _ = ledger.nw_t(iid, 4)
        self.assertLess(abs(b / a - 1), 0.25)

    def test_the_lamp(self):
        rng = np.random.default_rng(1)
        self.assertEqual(ledger.verdict(rng.normal(0.03, 0.08, 20), 1)["lamp"], "none")             # too few days, whatever they say
        self.assertEqual(ledger.verdict(rng.normal(0.03, 0.08, 200), 1)["lamp"], "likely")
        self.assertEqual(ledger.verdict(rng.normal(-0.01, 0.08, 400), 1)["lamp"], "unlikely")
        mid = ledger.verdict(rng.normal(0.006, 0.08, 120), 1)
        self.assertEqual(mid["lamp"], "none")
        self.assertIn("inside the noise", mid["text"])
        empty = ledger.verdict(np.array([]), 5)
        self.assertEqual((empty["lamp"], empty["n"], empty["ic"]), ("none", 0, None))

    def test_forecasts_are_settled_against_the_index_and_judged_by_evening(self):
        tmp = Path(tempfile.mkdtemp(prefix="stocks-ledger-"))
        try:
            store = Store(tmp)
            p = synth.panel(20, 200, seed=4)
            day = f"{p.days[150] // 10000}-{p.days[150] // 100 % 100:02d}-{p.days[150] % 100:02d}"
            store.save_signal(Signal("agent", day, "SAAB", 5, 0.62, "up"))
            store.save_signal(Signal("agent", day, "SAAC", 5, 0.40, "down"))
            late = f"{p.days[198] // 10000}-{p.days[198] // 100 % 100:02d}-{p.days[198] % 100:02d}"
            store.save_signal(Signal("agent", late, "SAAD", 5, 0.6, "too new"))
            self.assertEqual(ledger.settle_signals(store, p, "SPY"), 2)
            got = {s["symbol"]: s for s in store.signals("agent")}
            fwd = dataset.forward(p, 5)
            want = float(fwd[150, p.ix["SAAB"]] - fwd[150, p.ix["SPY"]])
            self.assertAlmostEqual(got["SAAB"]["excess"], want, places=6)
            self.assertIsNone(got["SAAD"]["excess"])
            self.assertEqual(len(store.open_signals()), 1)
            v = ledger.signal_verdict(store.signals("agent", settled=True), 5)
            sign = (got["SAAB"]["excess"] - got["SAAC"]["excess"]) / 2          # the call against counts with its sign turned
            self.assertAlmostEqual(v["ic"], round(sign, 4), places=4)
            self.assertEqual(v["n"], 1)
            self.assertEqual(ledger.brier(store.signals("agent"), 5)[1], 2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_the_first_decision_of_an_evening_stands(self):
        tmp = Path(tempfile.mkdtemp(prefix="stocks-store-"))
        try:
            store = Store(tmp)
            self.assertTrue(store.save_decision(Decision("net", "2026-11-13", [Order("A", 0.1, "x")], scores={"A": 1.0, "B": -1.0})))
            self.assertFalse(store.save_decision(Decision("net", "2026-11-13", [Order("B", 0.1, "y")])))
            self.assertEqual([o.symbol for o in store.decisions("net")[0].orders], ["A"])
            self.assertEqual(store.load_scores("net", "2026-11-13"), {"A": 1.0, "B": -1.0})
            store.reset("net")
            self.assertEqual(store.decisions("net"), [])
            self.assertIsNone(store.load_scores("net", "2026-11-13"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class Fitting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = synth.panel(90, 760, seed=11, reversal=0.25)
        cls.table = dataset.build(cls.p, "SPY")

    def test_no_training_outcome_reaches_the_validation_span_or_the_fit_day(self):
        t_fit = 700
        train, val = fit.split(self.table, t_fit)
        self.assertLess(max(train) + dataset.H_MAX + 1, min(val))          # a training row's 20-day outcome ends before validation begins
        self.assertLess(max(val) + dataset.H_MAX + 1, t_fit)               # and validation's ends before the fit takes over
        self.assertEqual(min(train), dataset.WARMUP)
        with self.assertRaises(fit.NotEnoughHistory):
            fit.split(self.table, 500)

    def test_refits_fall_on_january_and_july(self):
        rows = fit.refit_rows(self.p.days, 300, 759)
        months = [int(self.p.days[t]) // 100 % 100 for t in rows[1:]]
        self.assertTrue(months and set(months) <= {1, 7})
        self.assertEqual(rows[0], 300)

    def test_each_fitter_finds_what_was_planted(self):
        train, val = fit.split(self.table, 700)
        test_days = list(range(700, 735))
        fits = {"ridge": ridge.fit(self.table, train, val), "trees": trees.fit(self.table, train, val),
                "net": net.fit(self.table, train, val, cfg=net.NetCfg(epochs=4, patience=2))}
        for algo, params in fits.items():
            ic = ridge.day_ic(self.table, test_days, lambda X, c, a=algo, p=params: fit.ALGOS[a].predict(p, X, c))
            self.assertGreater(ic, 0.03, algo)

    def test_fits_are_kept_and_found_again(self):
        tmp = Path(tempfile.mkdtemp(prefix="stocks-fits-"))
        try:
            store = fit.Fits(tmp)
            train, val = fit.split(self.table, 700)
            params = ridge.fit(self.table, train, val)
            store.save("ridge", 20201001, [params])
            store.save("ridge", 20201101, [params])
            self.assertEqual(store.latest("ridge")["day"], 20201101)
            self.assertEqual(store.latest("ridge", 20201015)["day"], 20201001)
            self.assertIsNone(store.latest("net"))
            X = np.asarray(self.table.X[self.table.rows(720)], dtype=np.float32)
            a = fit.ensemble("ridge", store.latest("ridge")["fits"], X, self.table.ctx[720])
            self.assertAlmostEqual(float(a.mean()), 0.0, places=4)        # each seed is put on a common scale
            self.assertAlmostEqual(float(a.std()), 1.0, places=3)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class Figures(unittest.TestCase):
    def test_summary_of_a_book(self):
        idx = 10_000 * np.cumprod(np.r_[1.0, np.full(252, 1.0004)])
        eq = 10_000 * np.cumprod(np.r_[1.0, np.full(252, 1.0008)])
        s = metrics.summarize(eq, idx, np.full(253, 100.0), np.full(253, 0.05))
        self.assertAlmostEqual(s["ret_ann"], 1.0008 ** 252 - 1, places=6)
        self.assertAlmostEqual(s["vs_index_ann"], 1.0008 ** 252 - 1.0004 ** 252, places=6)
        self.assertAlmostEqual(s["max_dd"], 0.0)
        self.assertAlmostEqual(s["turnover"], 253 * 100 / eq.mean(), places=6)
        self.assertEqual(metrics.max_drawdown(np.array([100, 120, 90, 130])), -0.25)
        self.assertEqual(metrics.clean({"a": float("nan"), "b": [np.float32(1.5), np.inf], "c": np.int64(3)}), {"a": None, "b": [1.5, None], "c": 3})
        years = metrics.by_year(np.array([20231229, 20240102, 20240103, 20250102]), np.array([100, 110, 121, 133.1]), np.array([100, 100, 100, 100.0]))
        self.assertEqual([y["year"] for y in years], [2024, 2025])
        self.assertAlmostEqual(years[0]["ret"], 0.21, places=6)

    def test_what_a_call_costs(self):
        self.assertAlmostEqual(cost_usd("claude-opus-5-5", 1_000_000, 100_000), 4.0 + 2.0)
        self.assertAlmostEqual(cost_usd("claude-haiku-5-5", 50_000, 10_000), 0.005 + 0.005)
        self.assertAlmostEqual(cost_usd("claude-haiku-5-5", 150_000, 0), 0.075)                 # the long-prompt rate
        self.assertAlmostEqual(cost_usd("claude-opus-5-5", 0, 0, searches=30), 0.30)
        self.assertAlmostEqual(cost_usd("claude-opus-5-5", 1_000_000, 0, batch=True), 2.0)
        self.assertAlmostEqual(cost_usd("no-such-model", 1_000_000, 0), 10.0)                   # unknown reads high, never low


if __name__ == "__main__":
    unittest.main()
