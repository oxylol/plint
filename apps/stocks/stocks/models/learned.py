"""The models that are fitted: the straight line, the boosted trees, the neural network, and their blend.

All three read the same fifty-six figures and hold their books by the same rule, so the page compares what
each one learned and nothing else."""
from __future__ import annotations

import numpy as np

from .. import features
from ..learn.dataset import MAIN
from ..learn.fit import ensemble
from ..portfolio import TopK
from .base import ScoreModel
from ..util import quiet


class Learned(ScoreModel):
    kind = "learned"
    algo = ""
    head = MAIN                 # which of the network's horizons it trades on (the others have only the 5-day one)
    # Five days is what the ranking speaks about, so five days is how long a holding stays unless it is still a
    # top pick. (Filip, 2026-11-14: "the network should probably not hold for long". The same rule goes for the
    # trees, the line and the blend, so the four still differ only in what they learned.)
    top = TopK(k=40, hold_frac=0.20, max_new=8, every="day", max_hold=5)
    horizon = 5

    def score_table(self) -> np.ndarray:
        p = self.panel
        S = np.full((p.T, p.N), np.nan, dtype=np.float32)
        self._wait, self.fit_day = "", None
        fit = self.env.fits.latest(self.algo, int(p.days[-1])) if self.env.fits is not None else None
        if fit is None:
            self._wait = "waiting for its first fit"
            return S
        if p.T < features.LOOKBACK:
            self._wait = f"waiting for {features.LOOKBACK} days of prices"
            return S
        if "today" not in self.env.cache:
            self.env.cache["today"] = features.today(p.tail(features.LOOKBACK), self.env.index)
        X, cols, ctx = self.env.cache["today"]
        if len(cols):
            S[-1, cols] = ensemble(self.algo, fit["fits"], X, ctx, self.head)
        self.fit_day = fit["day"]
        return S

    def waiting(self) -> str:
        return getattr(self, "_wait", "")


class Ridge(Learned):
    id, title, algo = "ridge", "Straight line", "ridge"
    what = "The network's inputs through a plain linear fit. If the network cannot beat this, it has learned nothing."


class Trees(Learned):
    id, title, algo = "trees", "Boosted trees", "trees"
    what = "The network's inputs read by gradient-boosted trees: the usual winner on tables of figures, here as the rival."


class Net(Learned):
    id, title, algo = "net", "Neural network", "net"
    what = "A network trained on ten years of daily prices ranks 1,000 stocks each evening and holds the top 40, a week at most each."


class NetFast(Learned):
    """The same fitted network read at its shortest horizon. Filip, 2026-11-14: "the neural network can trade
    more often if that is more profitable". Whether it is, is for the backtest and the paper record to say:
    a one-day view is sharper and decays in a day, so this book turns over many times faster and pays for it."""
    id, title, algo = "netfast", "Network, fast", "net"
    what = "The same network trading on its one-day view: holdings last a day or two. Here to see whether trading more often pays after costs."
    head = 0
    horizon = 1
    top = TopK(k=40, hold_frac=0.08, max_new=20, every="day", max_hold=2)


class Blend(ScoreModel):
    id, title, kind = "blend", "Blend", "learned"
    what = "The average of the three learned rankings: often steadier than any one of them."
    top = Learned.top
    horizon = 5
    PARTS = ("ridge", "trees", "net")

    def score_table(self) -> np.ndarray:
        tables = [self.env.peers[i].S for i in self.PARTS if i in self.env.peers and getattr(self.env.peers[i], "S", None) is not None]
        self._wait = ""
        p = self.panel
        if len(tables) < 2:
            self._wait = "waiting for at least two of the learned models"
            return np.full((p.T, p.N), np.nan, dtype=np.float32)
        with quiet():
            zs = []
            for S in tables:
                mu, sd = np.nanmean(S, axis=1, keepdims=True), np.nanstd(S, axis=1, keepdims=True)
                zs.append((S - mu) / np.where(sd > 0, sd, np.nan))
            stack = np.stack(zs)
            enough = np.isfinite(stack).sum(axis=0) >= 2
            out = np.where(enough, np.nanmean(stack, axis=0), np.nan).astype(np.float32)
        if not np.isfinite(out[-1]).any():
            self._wait = "waiting for at least two of the learned models"
        return out

    def waiting(self) -> str:
        return getattr(self, "_wait", "")
