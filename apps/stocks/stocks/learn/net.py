"""The neural network.

What it is asked: on an evening, given every stock's place among its peers on fifty-six figures, order the
stocks by how they will do against each other over the next days. It is trained on the ordering itself
(the correlation between its scores and the ranked outcomes, day by day), at four horizons at once; the
5-day head is the one that trades.

What makes it a network rather than trees: it reads the whole day at once. Each stock is encoded alone;
sixteen learned "slots" then attend over all thousand encodings (and the market's own row), and each stock
reads the slots back. So a stock's score can depend on what the rest of the market looks like that day:
which kinds of stock are being bought, how crowded its corner is. Trees cannot see that. Whether it helps
is measured, not assumed: `mode` switches the cross-section off ("mlp") or down to a plain average
("pool"), and the backtest report shows all three beside the ridge and the trees.

Sizes are small on purpose (about 100k weights): the signal in daily returns is faint and Plint has four
cores. Several seeds are averaged, which does more for a faint signal than any one larger network."""
from __future__ import annotations

import io
from dataclasses import asdict, dataclass

import numpy as np

from ..ledger import spearman
from .dataset import HORIZONS, MAIN, Table

HEAD_WEIGHTS = (0.15, 0.40, 0.25, 0.20)      # how much each horizon counts in the loss


@dataclass(frozen=True)
class NetCfg:
    mode: str = "set"           # "mlp": each stock alone | "pool": + the day's average | "set": + attention slots
    d: int = 64
    hidden: int = 160
    slots: int = 16
    heads: int = 4
    drop: float = 0.10
    lr: float = 2e-3
    wd: float = 1e-4
    epochs: int = 16
    patience: int = 4
    batch_days: int = 8
    day_frac: float = 0.6       # share of the training days seen per epoch
    mse: float = 0.02           # a touch of squared error so the scores keep a scale
    threads: int = 3


def _torch():
    import torch
    return torch


def build_net(F: int, C: int, cfg: NetCfg):
    torch = _torch()
    nn = torch.nn

    class MAB(nn.Module):
        """One attention block: `q` reads `k`."""

        def __init__(self, d: int, heads: int, drop: float):
            super().__init__()
            self.att = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
            self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

        def forward(self, q, k, ignore=None):
            h = self.n1(q + self.att(q, k, k, key_padding_mask=ignore, need_weights=False)[0])
            return self.n2(h + self.ff(h))

    class SetNet(nn.Module):
        def __init__(self):
            super().__init__()
            d = cfg.d
            self.enc = nn.Sequential(nn.Linear(F, cfg.hidden), nn.GELU(), nn.Dropout(cfg.drop),
                                     nn.Linear(cfg.hidden, d), nn.GELU(), nn.LayerNorm(d))
            if cfg.mode != "mlp":
                self.market = nn.Sequential(nn.Linear(C, 32), nn.GELU(), nn.Linear(32, d))
            if cfg.mode == "set":
                self.slots = nn.Parameter(torch.randn(1, cfg.slots, d) * 0.1)
                self.gather = MAB(d, cfg.heads, cfg.drop)       # the slots read the day's stocks
                self.spread = MAB(d, cfg.heads, cfg.drop)       # each stock reads the slots
            wide = d if cfg.mode == "mlp" else 2 * d
            self.head = nn.Sequential(nn.Linear(wide, 64), nn.GELU(), nn.Dropout(cfg.drop), nn.Linear(64, len(HORIZONS)))

        def forward(self, x, ctx, real):
            """x [B, n, F], ctx [B, C], real [B, n] (False = padding) → [B, n, H]"""
            e = self.enc(x)
            if cfg.mode == "mlp":
                return self.head(e)
            c = self.market(ctx)
            if cfg.mode == "pool":
                w = real.unsqueeze(-1).to(e.dtype)
                g = ((e * w).sum(1) / w.sum(1).clamp(min=1.0) + c).unsqueeze(1).expand_as(e)
            else:
                s = self.gather(self.slots.expand(len(x), -1, -1) + c.unsqueeze(1), e, ~real)
                g = self.spread(e, s)
            return self.head(torch.cat([e, g], dim=-1))

    return SetNet()


def _batch(table: Table, days: list[int]):
    torch = _torch()
    n = max(int(table.ptr[t + 1] - table.ptr[t]) for t in days)
    x = np.zeros((len(days), n, table.X.shape[1]), dtype=np.float32)
    y = np.full((len(days), n, len(HORIZONS)), np.nan, dtype=np.float32)
    real = np.zeros((len(days), n), dtype=bool)
    for i, t in enumerate(days):
        sl = table.rows(t)
        k = sl.stop - sl.start
        x[i, :k], y[i, :k], real[i, :k] = table.X[sl], table.Y[sl], True
    return torch.from_numpy(x), torch.from_numpy(table.ctx[days].astype(np.float32)), torch.from_numpy(y), torch.from_numpy(real)


def _loss(pred, y, real, mse: float):
    """Minus the day-by-day correlation between score and ranked outcome, per horizon, plus a little MSE."""
    torch = _torch()
    v = (real.unsqueeze(-1) & torch.isfinite(y)).to(pred.dtype)
    y0 = torch.nan_to_num(y)
    cnt = v.sum(1)
    pm = (pred * v).sum(1) / cnt.clamp(min=1.0)
    ym = (y0 * v).sum(1) / cnt.clamp(min=1.0)
    pc, yc = (pred - pm.unsqueeze(1)) * v, (y0 - ym.unsqueeze(1)) * v
    corr = (pc * yc).sum(1) / torch.sqrt((pc * pc).sum(1) * (yc * yc).sum(1) + 1e-8)      # [B, H]
    use = (cnt >= 20).to(pred.dtype)
    w = torch.tensor(HEAD_WEIGHTS, dtype=pred.dtype)
    ic = (corr * use * w).sum() / (use * w).sum().clamp(min=1e-6)
    sq = (((pred - y0) ** 2) * v).sum() / v.sum().clamp(min=1.0)
    return -ic + mse * sq


def _scores(net, x, ctx, real, head: int = MAIN) -> np.ndarray:
    torch = _torch()
    net.eval()
    with torch.no_grad():
        return net(x, ctx, real)[..., head].numpy()


def val_ic(net, table: Table, days: list[int], batch_days: int = 16) -> float:
    ics = []
    for i in range(0, len(days), batch_days):
        part = days[i:i + batch_days]
        x, ctx, _, real = _batch(table, part)
        out = _scores(net, x, ctx, real)
        for b, t in enumerate(part):
            sl = table.rows(t)
            ic, _ = spearman(out[b, :sl.stop - sl.start], table.R[sl, MAIN])
            if np.isfinite(ic):
                ics.append(ic)
    return float(np.mean(ics)) if ics else float("nan")


def _run(table: Table, days: list[int], cfg: NetCfg, seed: int, epochs: int, val: list[int] | None, log=None):
    """Train for `epochs` (stopping early on `val` when given). Returns (net, best epoch, best val IC)."""
    torch = _torch()
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = build_net(table.X.shape[1], table.ctx.shape[1], cfg)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    per_epoch = max(1, int(len(days) * cfg.day_frac) // cfg.batch_days)
    # warm up, then hold: a rate that does not depend on the epoch count, so "train again for E epochs" means the same path
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1.0, (step + 1) / 100.0))
    best, best_ic, best_epoch, bad = None, -np.inf, epochs, 0
    for epoch in range(1, epochs + 1):
        net.train()
        order = rng.permutation(days)[: per_epoch * cfg.batch_days]
        total = 0.0
        for i in range(0, len(order), cfg.batch_days):
            x, ctx, y, real = _batch(table, sorted(int(t) for t in order[i:i + cfg.batch_days]))
            loss = _loss(net(x, ctx, real), y, real, cfg.mse)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            total += float(loss.detach())
        if val:
            ic = val_ic(net, table, val)
            if log:
                log(f"seed {seed} epoch {epoch}: loss {total / per_epoch:+.4f}, validation IC {ic:+.4f}")
            if np.isfinite(ic) and ic > best_ic + 1e-5:
                best_ic, best_epoch, bad = ic, epoch, 0
                best = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    if val and best is not None:
        net.load_state_dict(best)
    return net, best_epoch, best_ic


def fit(table: Table, train: list[int], val: list[int], seed: int = 0, cfg: NetCfg = NetCfg(), log=None) -> dict:
    """Pick the number of epochs on the validation span, then train again on everything for that many."""
    torch = _torch()
    _, epochs, ic = _run(table, train, cfg, seed, cfg.epochs, val, log)
    net, _, _ = _run(table, sorted(train + val), cfg, seed, epochs, None, log)
    buf = io.BytesIO()
    torch.save(net.state_dict(), buf)
    return {"state": buf.getvalue(), "cfg": asdict(cfg), "epochs": epochs, "val_ic": ic,
            "F": table.X.shape[1], "C": table.ctx.shape[1]}


def _net_of(params: dict):
    net = params.get("_net")
    if net is None:
        torch = _torch()
        net = build_net(params["F"], params["C"], NetCfg(**params["cfg"]))
        net.load_state_dict(torch.load(io.BytesIO(params["state"]), map_location="cpu"))
        params["_net"] = net
    return net


def predict(params: dict, X: np.ndarray, ctx: np.ndarray, head: int = MAIN) -> np.ndarray:
    """Scores for one day's stocks, from the head for one horizon (HORIZONS[head] days ahead)."""
    torch = _torch()
    x = torch.from_numpy(np.asarray(X, dtype=np.float32)).unsqueeze(0)
    c = torch.from_numpy(np.asarray(ctx, dtype=np.float32)).unsqueeze(0)
    return _scores(_net_of(params), x, c, torch.ones(1, x.shape[1], dtype=torch.bool), head)[0]
