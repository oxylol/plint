"""Nothing may see tomorrow.

Three checks, each on the made-up market:
1. A figure on row t does not change when everything after t is replaced with noise.
2. The figures computed each evening from the last LOOKBACK days equal the ones the models were trained on.
3. Every model decides the same on a table that ends at t as on one that runs on past it.

    docker run --rm -v "$PWD":/app -w /app -e PYTHONPATH=/app:/app/tests plint/stocks:dev python -m unittest tests.test_leak -v
"""
from __future__ import annotations

import unittest

import numpy as np

import synth
from stocks import features
from stocks.learn import dataset, fit, ridge
from stocks.models import Env
from stocks.models.learned import Ridge
from stocks.models.rules import Index, Momentum, Reversal, Trend
from stocks.panel import FIELDS, Panel
from stocks.types import BookView, Holding
from stocks.universe import rank_at


def scrambled_after(p: Panel, t: int, seed: int = 1) -> Panel:
    """The same panel with every row after t replaced by something else entirely."""
    rng = np.random.default_rng(seed)
    data = {f: p.data[f].copy() for f in FIELDS}
    for f in FIELDS:
        tail = data[f][t + 1:]
        data[f][t + 1:] = np.where(np.isfinite(tail), tail * rng.uniform(0.5, 1.5, tail.shape), tail)
    data["ret"][t + 1:] = rng.normal(0, 0.05, data["ret"][t + 1:].shape)
    return Panel(p.days, p.syms, data, p.member)


class Causality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = synth.panel(60, 520, seed=5)

    def test_figures_do_not_move_when_the_future_does(self):
        t = 400
        other = scrambled_after(self.p, t)
        a = dict(features.raw(self.p, "SPY"))
        b = dict(features.raw(other, "SPY"))
        for name in features.NAMES:
            np.testing.assert_array_equal(np.nan_to_num(a[name][: t + 1]), np.nan_to_num(b[name][: t + 1]), err_msg=name)
        np.testing.assert_array_equal(features.context(self.p, "SPY")[: t + 1], features.context(other, "SPY")[: t + 1])

    def test_the_outcome_table_is_the_only_thing_that_looks_ahead(self):
        t = 400
        other = scrambled_after(self.p, t)
        for h in dataset.HORIZONS:
            a, b = dataset.forward(self.p, h), dataset.forward(other, h)
            known = t - h - 1                       # the last evening whose outcome was complete by t
            np.testing.assert_array_equal(np.nan_to_num(a[: known + 1]), np.nan_to_num(b[: known + 1]))
            self.assertFalse(np.allclose(np.nan_to_num(a[known + 1: t + 1]), np.nan_to_num(b[known + 1: t + 1])))

    def test_evening_figures_equal_training_figures(self):
        table = dataset.build(self.p, "SPY")
        for t in (features.LOOKBACK + 3, 430, 519):
            X, cols, ctx = features.today(self.p.window(t, features.LOOKBACK), "SPY")
            sl = table.rows(t)
            np.testing.assert_array_equal(table.col[sl], cols)
            full = np.asarray(table.X[sl], dtype=np.float32)
            self.assertLess(np.abs(full - X).max(), 2.5 / len(cols), f"row {t}")     # at most one place apart (ties)
            np.testing.assert_allclose(table.ctx[t], ctx, atol=1e-5)

    def test_the_universe_is_picked_from_the_past(self):
        p = self.p
        dv = np.where(np.isfinite(p.dv), p.dv, np.nan)
        a = rank_at(p.days, p.syms, dv, p.c, 300, 20, 1.0)
        other = scrambled_after(p, 300)
        b = rank_at(other.days, other.syms, np.where(np.isfinite(other.dv), other.dv, np.nan), other.c, 300, 20, 1.0)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 20)
        self.assertEqual([r for _, r, _ in a], list(range(1, 21)))


class Decisions(unittest.TestCase):
    """A model prepared on days 0..t and one prepared on all days must give the same orders at t."""

    @classmethod
    def setUpClass(cls):
        cls.p = synth.panel(80, 560, seed=7)
        cls.env = Env(index="SPY", cash="BIL")

    def view(self, syms):
        return BookView(10_000.0, 0.1, {s: Holding(s, 0.9 / len(syms), 3, "2019-01-02", 0.01) for s in syms})

    def both(self, cls_, t: int, view: BookView):
        whole, upto = cls_(), cls_()
        whole.prepare(scrambled_after(self.p, t), Env(index="SPY", cash="BIL"))
        cut = self.p.upto(t)
        upto.prepare(cut, Env(index="SPY", cash="BIL"))
        return whole.decide(t, view), upto.decide(cut.T - 1, view)

    def test_rule_models(self):
        held = [self.p.syms[j] for j in (3, 9, 14, 22)]
        for cls_ in (Index, Trend, Momentum, Reversal):
            for t in (340, 471, 559):
                for view in (BookView(10_000.0, 1.0), self.view(held)):
                    a, b = self.both(cls_, t, view)
                    self.assertEqual([(o.symbol, round(o.weight, 6), o.why) for o in a.orders],
                                     [(o.symbol, round(o.weight, 6), o.why) for o in b.orders], f"{cls_.id} at {t}")
                    self.assertEqual(a.date, b.date)

    def test_a_learned_model_scores_tonight_as_it_was_trained(self):
        p = self.p
        table = dataset.build(p, "SPY")
        t = 540
        train, val = list(range(252, 380)), list(range(402, 460))
        params = ridge.fit(table, train, val)
        expect = fit.ensemble("ridge", [params], np.asarray(table.X[table.rows(t)], dtype=np.float32), table.ctx[t])

        class OneFit:
            def latest(self, algo, on_or_before=None):
                return {"algo": "ridge", "day": 20190101, "fits": [params]}

        m = Ridge()
        cut = p.upto(t)
        m.prepare(cut, Env(index="SPY", cash="BIL", fits=OneFit()))
        cols = table.col[table.rows(t)]
        got = m.S[-1, cols]
        self.assertTrue(np.isfinite(got).all())
        self.assertGreater(np.corrcoef(got, expect)[0, 1], 0.999)
        self.assertEqual(m.waiting(), "")
        ranked, scores = m.ranked(cut.T - 1)
        self.assertEqual(len(ranked), len(cols))
        self.assertEqual(ranked[0], max(scores, key=scores.get))

    def test_a_learned_model_without_a_fit_waits(self):
        m = Ridge()

        class NoFit:
            def latest(self, algo, on_or_before=None):
                return None

        m.prepare(self.p.upto(400), Env(index="SPY", cash="BIL", fits=NoFit()))
        self.assertIn("first fit", m.waiting())
        self.assertEqual(m.decide(400, BookView(10_000.0, 1.0)).orders, [])


if __name__ == "__main__":
    unittest.main()
