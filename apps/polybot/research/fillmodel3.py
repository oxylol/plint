"""What a follower really pays: cost above the midpoint by market type, price and time to the end, from book snapshots.

    python fillmodel3.py DATA_DIR

Reads DATA_DIR/books/*.jsonl.gz (collect3_books.py) and writes DATA_DIR/fill_model.json:
for each (type group, price band, hours-to-end band) the median and 75th percentile of
(VWAP of a $5 / $10 / $20 / $50 market buy − midpoint), the share of books too thin
to take the order at all, and the dollars resting within 1¢, 2¢ and 5¢ of the best ask
(the capacity of a trade). A band with too few snapshots falls back to its type group,
then to everything.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA = Path(sys.argv[1])
SIZES = ("5", "10", "20", "50")
GROUP = {"sports": "sports", "weather": "weather", "crypto": "crypto", "politics": "politics", "geopolitics": "politics"}


def band(mid):
    return "lt10" if mid < 0.10 else "10_30" if mid < 0.30 else "30_70" if mid < 0.70 else "70_90" if mid < 0.90 else "ge90"


def hband(h):
    return "lt6" if h < 6 else "6_48" if h < 48 else "2_14d" if h < 336 else "gt14d"


def keys(cat, mid, hours, in_game):
    g = GROUP.get(cat, "other")
    if g == "sports" and in_game:
        g = "sports_live"
    return [(g, band(mid), hband(hours)), (g, band(mid)), (g,), ()]


def main():
    acc = defaultdict(lambda: {s: [] for s in SIZES} | {"thin": [], "d1": [], "d2": [], "d5": [], "spread": []})
    n = 0
    for f in sorted((DATA / "books").glob("*.jsonl.gz")):
        try:
            fh = gzip.open(f, "rt")
            lines = list(fh)
        except Exception:
            continue
        for line in lines:
            try:
                b = json.loads(line)
            except Exception:
                continue
            if b.get("bid") is None or b.get("ask") is None or b["ask"] <= b["bid"]:
                continue
            mid = (b["bid"] + b["ask"]) / 2
            if not (0.02 <= mid <= 0.98):
                continue
            t = dt.datetime.fromisoformat(b["t"])
            hours = (dt.datetime.fromisoformat(b["end"]) - t).total_seconds() / 3600 if b.get("end") else 24 * 30
            gs = dt.datetime.fromisoformat(b["game_start"]) if b.get("game_start") else None
            in_game = gs is not None and t >= gs
            asks = b.get("asks") or []
            best = b["ask"]
            depth = {k: sum(p * s for p, s in asks if p <= best + k / 100 + 1e-9) for k in (1, 2, 5)}
            for k in keys(b.get("cat") or "other", mid, max(0.0, hours), in_game):
                a = acc[k]
                for s in SIZES:
                    v = b["buy"].get(s)
                    if v is not None:
                        a[s].append(v - mid)
                a["thin"].append(1.0 if b["buy"].get("5") is None else 0.0)
                a["d1"].append(depth[1])
                a["d2"].append(depth[2])
                a["d5"].append(depth[5])
                a["spread"].append(b["ask"] - b["bid"])
            n += 1
    out = {"books": n, "cells": {}}
    for k, a in acc.items():
        if len(a["thin"]) < 40:
            continue
        cell = {"n": len(a["thin"]), "thin": float(np.mean(a["thin"])), "spread_med": float(np.median(a["spread"])),
                "depth1_med": float(np.median(a["d1"])), "depth2_med": float(np.median(a["d2"])), "depth5_med": float(np.median(a["d5"]))}
        for s in SIZES:
            if len(a[s]) >= 20:
                cell[f"c{s}_med"] = float(np.median(a[s]))
                cell[f"c{s}_p75"] = float(np.quantile(a[s], 0.75))
        out["cells"]["|".join(k)] = cell
    (DATA / "fill_model.json").write_text(json.dumps(out, indent=1))
    print(f"{n} books → {len(out['cells'])} cells")
    for k in sorted(out["cells"], key=lambda k: (k.count("|"), k)):
        c = out["cells"][k]
        if k.count("|") <= 1:
            print(f"  {k or 'all':22s} n {c['n']:6d}  $5 +{100 * c.get('c5_med', float('nan')):.1f}¢ (p75 +{100 * c.get('c5_p75', float('nan')):.1f}¢)  "
                  f"$20 +{100 * c.get('c20_med', float('nan')):.1f}¢  thin {100 * c['thin']:.0f}%  spread {100 * c['spread_med']:.1f}¢  "
                  f"$ within 1¢/2¢/5¢ of ask {c['depth1_med']:.0f}/{c['depth2_med']:.0f}/{c['depth5_med']:.0f}")


class FillModel:
    """Look up the cost above mid for a stake; falls back to coarser cells."""
    def __init__(self, path: Path):
        self.cells = json.loads(Path(path).read_text())["cells"]

    def cell(self, cat, mid, hours, in_game):
        for k in keys(cat, mid, hours, in_game):
            c = self.cells.get("|".join(k))
            if c is not None and "c5_med" in c:
                return c
        return None

    def cost(self, cat, mid, hours, in_game, stake="5", q="med"):
        c = self.cell(cat, mid, hours, in_game)
        if c is None:
            return 0.01
        return max(0.0, c.get(f"c{stake}_{q}", c["c5_med"]))


if __name__ == "__main__":
    main()
