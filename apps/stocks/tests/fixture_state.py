"""Writes example answers for the page's two endpoints into tests/fixtures/ (shapes in PAGE.md).

    python tests/fixture_state.py

Three states: `running` (every model with a record), `keys` (first day: nothing but missing keys), `stopped`
(the agent hit its cap, a data fault). The numbers are made up and seeded, so the files do not change between
runs. The real page JSON is built by stocks/publish.py and is checked against the same field lists."""
from __future__ import annotations

import json
import math
import random
from datetime import date, timedelta
from pathlib import Path

OUT = Path(__file__).parent / "fixtures"
NAMES = {
    "NVDA": "NVIDIA Corporation", "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "XOM": "Exxon Mobil Corporation",
    "AMD": "Advanced Micro Devices, Inc.", "LLY": "Eli Lilly and Company", "JPM": "JPMorgan Chase & Co.",
    "COST": "Costco Wholesale Corporation", "PLTR": "Palantir Technologies Inc.", "UBER": "Uber Technologies, Inc.",
    "CAT": "Caterpillar Inc.", "DE": "Deere & Company", "MRK": "Merck & Co., Inc.", "TSLA": "Tesla, Inc.",
    "ANET": "Arista Networks, Inc.", "CRWD": "CrowdStrike Holdings, Inc.", "GE": "GE Aerospace", "WMT": "Walmart Inc.",
    "ABNB": "Airbnb, Inc.", "SHOP": "Shopify Inc.", "ZS": "Zscaler, Inc.", "FSLR": "First Solar, Inc.",
    "DAL": "Delta Air Lines, Inc.", "NEM": "Newmont Corporation", "HOOD": "Robinhood Markets, Inc.",
    "VRT": "Vertiv Holdings Co", "TTD": "The Trade Desk, Inc.", "MU": "Micron Technology, Inc.",
    "BKNG": "Booking Holdings Inc.", "CVNA": "Carvana Co.", "APP": "AppLovin Corporation", "RKLB": "Rocket Lab Corporation",
}
SYMS = sorted(NAMES)

MODELS = [
    ("index", "Index fund", "index", "Buys the S&P 500 fund once and holds it. The bar the others must beat.", 0.0, None),
    ("net", "Neural network", "learned", "A network trained on ten years of daily prices ranks 1,000 stocks each evening and holds the top 40.", 0.9, 5),
    ("netfast", "Neural network, fast", "learned", "The same network on its one-day view: it holds a stock for a day or two.", 0.5, 1),
    ("trees", "Boosted trees", "learned", "The same inputs as the network, read by gradient-boosted trees: the classic rival.", 0.6, 5),
    ("ridge", "Straight line", "learned", "The same inputs through a plain linear fit. If the network cannot beat this, it has learned nothing.", 0.2, 5),
    ("blend", "Blend", "learned", "The average of the three learned rankings.", 0.7, 5),
    ("agent", "API agent", "agent", "Claude reads the day's filings and news, writes a memo and trades at most four names a day.", 0.3, 20),
    ("momentum", "Momentum", "rule", "Holds the stocks that rose most over the past year, skipping the last month; re-picked monthly.", 0.4, 20),
    ("reversal", "Reversal", "rule", "Buys last week's hardest-hit stocks and holds them about a week.", -0.5, 5),
    ("drift", "After-earnings drift", "rule", "Buys a stock the day after a strong earnings reaction and holds it eight weeks.", 0.5, 20),
    ("insiders", "Insider buying", "rule", "Buys when two or more of a company's own officers buy its shares on the open market.", -0.1, 20),
    ("trend", "Trend filter", "rule", "Holds the index fund while it is above its 200-day average, otherwise Treasury bills.", -0.2, None),
]


def bdays(end: date, n: int) -> list[str]:
    out, d = [], end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d -= timedelta(days=1)
    return out[::-1]


def walk(rng: random.Random, n: int, drift: float, vol: float) -> list[float]:
    x, out = 1.0, []
    for _ in range(n):
        out.append(x)
        x *= 1 + rng.gauss(drift, vol)
    return out


def thin(xs: list[float], k: int = 60) -> list[float]:
    if len(xs) <= k:
        return [round(x, 5) for x in xs]
    step = (len(xs) - 1) / (k - 1)
    return [round(xs[round(i * step)], 5) for i in range(k)]


def backtest(rng: random.Random, edge: float, has_ic: bool) -> dict:
    idx = 0.128
    vs = edge * 0.035 + rng.gauss(0, 0.004)
    exam_vs = vs * 0.4 + rng.gauss(0, 0.01)
    ic = round(0.004 + edge * 0.012, 4) if has_ic else None
    return {
        "from": "2020-01-02", "to": "2025-05-13",
        "ret_ann": round(idx + vs, 4), "index_ann": idx, "vs_index_ann": round(vs, 4),
        "ci": [round(vs - 0.045, 4), round(vs + 0.045, 4)],
        "sharpe": round(0.62 + edge * 0.3, 2), "max_dd": round(-0.34 - rng.random() * 0.08, 3),
        "turnover": round(8 + rng.random() * 6, 1), "cost_drag": round(0.012 + rng.random() * 0.02, 4),
        "ic": ic, "ic_t": round(ic / 0.0045, 1) if ic is not None else None,
        "exam": {"from": "2025-05-14", "to": "2026-11-13", "ret_ann": round(0.171 + exam_vs, 4), "index_ann": 0.171,
                 "vs_index_ann": round(exam_vs, 4), "ci": [round(exam_vs - 0.09, 4), round(exam_vs + 0.09, 4)],
                 "ic": round(ic * 0.6, 4) if ic is not None else None,
                 "ic_t": round(ic * 0.6 / 0.008, 1) if ic is not None else None},
    }


def build(seed: int = 7, ndays: int = 46):
    rng = random.Random(seed)
    days = bdays(date(2026, 11, 13), ndays)
    index_curve = walk(rng, ndays, 0.0006, 0.008)
    summaries, details = [], {}
    for mid, title, kind, what, edge, horizon in MODELS:
        r = random.Random(f"{seed}-{mid}")
        if kind == "index":
            curve = index_curve[:]
        else:
            noise = walk(r, ndays, edge * 0.0004, 0.0035 if kind != "agent" else 0.005)
            curve = [a * b for a, b in zip(index_curve, noise)]
        eq = [10000 * x for x in curve]
        ix = [10000 * x for x in index_curve]
        peak, dd = 0.0, 0.0
        for v in eq:
            peak = max(peak, v)
            dd = min(dd, v / peak - 1)
        ret, iret = curve[-1] - 1, index_curve[-1] - 1
        scored = max(0, ndays - (horizon or 5) - 1)
        if kind in ("index",) or mid == "trend":
            verdict = None
        else:
            ic = 0.003 + edge * 0.011 + r.gauss(0, 0.003)
            t = ic / (0.085 / math.sqrt(max(scored, 1)))
            lamp = "likely" if t > 2 else "unlikely" if t < -1 else "none"
            text = {"likely": f"Skill likely: {scored} days scored", "unlikely": f"Skill unlikely: {scored} days scored",
                    "none": f"No verdict yet: {scored} of about 120 days scored"}[lamp]
            verdict = {"lamp": lamp, "text": text, "ic": round(ic, 4), "t": round(t, 2), "n": scored, "horizon": horizon}
        n_pos = 1 if mid in ("index", "trend") else 11 if kind == "agent" else 40
        held = r.sample(SYMS, min(len(SYMS), 14 if n_pos > 14 else n_pos))
        positions = []
        for s in held:
            d0 = r.randrange(1, 25)
            positions.append({"symbol": s, "name": NAMES[s], "weight": round(1 / max(n_pos, 1) * r.uniform(0.8, 1.3), 4),
                              "value": 0.0, "since": days[-d0], "days": d0 - 1, "ret": round(r.gauss(0.004, 0.05), 4),
                              "why": f"rank {r.randrange(1, 40)} of 1000" if kind != "agent" else "guidance raised; drift not yet in the price"})
        if mid in ("index", "trend"):
            positions = [{"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "weight": 0.999, "value": 0.0,
                          "since": days[0], "days": ndays - 1, "ret": round(iret, 4), "why": "the index, held"}]
        for p in positions:
            p["value"] = round(p["weight"] * eq[-1], 2)
        trades = []
        ntr = 2 if mid in ("index", "trend") else 22 if kind == "agent" else 160
        for i in range(min(ntr, 300)):
            s = r.choice(SYMS)
            side = r.choice(("buy", "sell"))
            px = round(r.uniform(20, 900), 2)
            usd = round(eq[-1] / max(n_pos, 1) * r.uniform(0.7, 1.2), 2)
            trades.append({"date": days[-1 - min(i // 6, ndays - 2)], "symbol": s, "name": NAMES[s], "side": side, "usd": usd,
                           "price": px, "shares": round(usd / px, 4), "cost": round(usd * r.uniform(1.5, 4.5) / 1e4, 3),
                           "why": (f"rank {r.randrange(1, 40)} of 1000" if side == "buy" else f"fell to rank {r.randrange(210, 700)}")
                           if kind != "agent" else ("8-K: guidance raised" if side == "buy" else "thesis played out"),
                           "ret": round(r.gauss(0.006, 0.06), 4) if side == "sell" else None})
        costs = round(sum(t["cost"] for t in trades), 2)
        summary = {
            "id": mid, "title": title, "kind": kind, "what": what, "state": "running", "state_text": "",
            "since": days[0], "days": ndays - 1, "equity": round(eq[-1], 2), "ret": round(ret, 5),
            "index_ret": round(iret, 5), "vs_index": round(ret - iret, 5),
            "day_ret": round(curve[-1] / curve[-2] - 1, 5),
            "day_vs_index": round(curve[-1] / curve[-2] - index_curve[-1] / index_curve[-2], 5),
            "max_dd": round(dd, 4), "n_pos": n_pos, "cash": round(r.uniform(0.0, 0.06), 4) if kind != "index" else 0.001,
            "trades": len(trades) if kind != "learned" else 160 + r.randrange(0, 90), "costs": costs,
            "turnover": 0.0 if mid == "index" else round(r.uniform(3, 14), 1),
            "verdict": verdict, "spark": thin(curve), "spark_index": thin(index_curve),
            "backtest": None if kind == "agent" else backtest(r, edge, kind in ("learned", "rule") and mid != "trend"),
        }
        top = [{"symbol": s, "name": NAMES[s], "score": round(1 - i * 0.03, 3), "rank": i + 1, "held": s in held}
               for i, s in enumerate(r.sample(SYMS, 12))]
        bottom = [{"symbol": s, "name": NAMES[s], "score": round(-1 + i * 0.03, 3), "rank": 1000 - i, "held": False}
                  for i, s in enumerate(r.sample(SYMS, 8))]
        scores_all = kind == "learned" or mid in ("momentum", "reversal")
        detail = dict(summary)
        detail.update({
            "curve": [[d, round(a, 2), round(b, 2)] for d, a, b in zip(days, eq, ix)],
            "positions": sorted(positions, key=lambda p: -p["weight"]),
            "pending": [] if mid in ("index", "trend") else
            [{"symbol": s, "name": NAMES[s], "side": sd, "weight": 0.025 if sd == "buy" else 0.0,
              "why": "rank 2 of 1000" if sd == "buy" else "fell to rank 312"}
             for s, sd in zip(r.sample(SYMS, 3), ("buy", "buy", "sell"))],
            "trades": trades,
            "today": {"date": days[-1], "note": "998 stocks ranked; 3 changes for tomorrow's open", "top": top, "bottom": bottom}
            if scores_all else None,
            "ranking_history": [{"date": d, "ic": round(r.gauss(0.01 + edge * 0.01, 0.08), 4), "n": 990 + r.randrange(0, 10)}
                                for d in days[:scored]] if scores_all else [],
            "signals": [], "memo": None, "reads": [],
            "report": None if kind == "agent" else (
                f"# {title}: backtest\n\nWalk-forward from 2020-01-02 to 2025-05-13, costs included. "
                "Each refit saw only what was known on its day.\n\n"
                "| year | model | index |\n|---|---|---|\n| 2020 | +21.3 % | +18.4 % |\n| 2021 | +24.0 % | +28.7 % |\n\n"
                "## What it does not show\n\nCompanies that left the market are missing from free data, "
                "so the history flatters every stock picker a little."),
            "by_year": None if kind == "agent" else
            [{"year": y, "ret": round(r.gauss(0.13, 0.15), 4), "index": i} for y, i in
             ((2020, 0.184), (2021, 0.287), (2022, -0.181), (2023, 0.262), (2024, 0.249), (2025, 0.071))],
        })
        if mid in ("drift", "insiders", "agent"):
            detail["signals"] = [{"date": days[-1 - (i // 3)], "symbol": s, "horizon": 20 if i % 2 else 5,
                                  "p": round(r.uniform(0.5, 0.72), 2),
                                  "note": "two officers bought $410k" if mid == "insiders" else
                                  "+7.2 % on results, 2.4x volume" if mid == "drift" else
                                  # the desk's own notes run long: the first sentence of a read, as it wrote it
                                  ("What changed tonight is a discretionary, sizeable open-market buy by the founder-CEO after a "
                                   "roughly 30% two-month drawdown and a fresh earnings reset." if i % 3 == 0 else "raised guidance, quiet reaction"),
                                  "excess": round(r.gauss(0.004, 0.04), 4) if i > 18 else None}
                                 for i, s in enumerate(r.choices(SYMS, k=45))]
        if kind == "agent":
            detail["memo"] = {"date": days[-1], "text": (
                "**Tonight.** 212 filings and 1,480 headlines triaged, 27 companies read in depth.\n\n"
                "- **Bought VRT (6 %)**: the 8-K lifts full-year guidance by 9 % and the stock closed up only 2.1 %. "
                "I put the chance it beats the index over 20 days at 0.61.\n"
                "- **Sold DAL**: the fuel-cost thesis is in the price; +4.2 % over the index since entry.\n"
                "- **Held the rest.** Nothing in today's reading changes them.\n\n"
                "*Not acted on:* CVNA's beat looked strong, but the move was 14 % by the close.")}
            detail["reads"] = [{"date": days[-1], "symbol": s, "name": NAMES[s],
                                "thesis": "Guidance raised on data-centre orders; the reaction was small against the size of the change.",
                                "facts": ["FY revenue guide +9 %", "backlog $8.5bn, +30 % y/y", "closed +2.1 % on 1.8x volume"],
                                "priced_in": "About a third: the stock had run 12 % into the print.",
                                "p5": round(r.uniform(0.48, 0.62), 2), "p20": round(r.uniform(0.5, 0.66), 2),
                                "conviction": round(r.uniform(0.2, 0.7), 2),
                                "risks": "One customer is 22 % of backlog.",
                                "sources": [{"title": "Form 8-K, 13 Nov 2026", "url": "https://www.sec.gov/Archives/edgar/data/0000000/example.htm"}]}
                               for s in r.sample(SYMS, 6)]
        summaries.append(summary)
        details[mid] = detail
    spend = {"month": "2026-11", "used": 61.37, "cap": 200.0, "today": 5.84, "calls": 4312, "stopped": None,
             "by_day": [[d, round(random.Random(d).uniform(3.5, 7.5), 2)] for d in days if d >= "2026-11-01"]}
    state = {
        "app": "stocks", "generated": "2026-11-13T22:41:12+01:00", "asof": days[-1], "phase": "running",
        "phase_text": "Running", "keys": {"alpaca": True, "anthropic": True, "sec": True, "ntfy": True},
        "notices": [], "clock": {"last_run": "2026-11-13T22:25:04+01:00", "next_run": "2026-11-16T22:25:00+01:00", "busy": None},
        "start_cash": 10000, "index": {"symbol": "SPY", "title": "S&P 500 (SPY)"}, "spend": spend, "models": summaries,
        "log": [{"t": "2026-11-13T21:41:12Z", "level": "info", "text": "evening run done: 11 models, 31 orders for the next open"},
                {"t": "2026-11-13T21:26:40Z", "level": "warn", "text": "data corrected: ANET 2026-11-12 return -74.9 % → +0.6 % (split)"},
                {"t": "2026-11-12T21:40:02Z", "level": "info", "text": "evening run done: 11 models, 27 orders for the next open"}],
    }
    return state, details


def keys_state(state: dict) -> dict:
    s = json.loads(json.dumps(state))
    s.update({"asof": None, "phase": "keys", "phase_text": "Waiting for the Alpaca keys",
              "keys": {"alpaca": False, "anthropic": False, "sec": False, "ntfy": True},
              "notices": [{"level": "warn", "text": "No market data yet: add ALPACA_API_KEY and ALPACA_API_SECRET to .env, then `plint up stocks`."},
                          {"level": "info", "text": "The agent is off until STOCKS_ANTHROPIC_API_KEY is in .env."},
                          {"level": "info", "text": "Filings are off until SEC_CONTACT (an email address, the SEC asks for one) is in .env."}],
              "clock": {"last_run": None, "next_run": None, "busy": None}, "log": [],
              "spend": {"month": "2026-11", "used": 0.0, "cap": 200.0, "today": 0.0, "calls": 0, "stopped": None, "by_day": []}})
    for m in s["models"]:
        for k in ("since", "equity", "ret", "index_ret", "vs_index", "day_ret", "day_vs_index", "max_dd", "verdict", "backtest", "turnover"):
            m[k] = None
        m.update({"state": "waiting", "state_text": "waiting for market data", "days": 0, "n_pos": 0, "cash": None,
                  "trades": 0, "costs": 0.0, "spark": [], "spark_index": []})
    return s


def stopped_state(state: dict) -> dict:
    s = json.loads(json.dumps(state))
    s["spend"].update({"used": 200.0, "today": 0.0, "stopped": "month cap"})
    s["notices"] = [{"level": "fault", "text": "No new prices since 2026-11-11: Alpaca answered 403. The books stand at that close."},
                    {"level": "warn", "text": "The agent reached its $200 for November and holds what it has until 1 December."}]
    s["clock"]["busy"] = "fitting the network (seed 3 of 5)"
    for m in s["models"]:
        if m["id"] == "agent":
            m.update({"state": "off", "state_text": "month cap reached"})
        if m["id"] == "insiders":
            m.update({"state": "waiting", "state_text": "waiting for SEC_CONTACT"})
    return s


def waiting_detail(summary: dict) -> dict:
    d = dict(summary)
    d.update({"curve": [], "positions": [], "pending": [], "trades": [], "today": None, "ranking_history": [],
              "signals": [], "memo": None, "reads": [], "report": None, "by_year": None})
    return d


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    state, details = build()
    dump = lambda name, obj: (OUT / name).write_text(json.dumps(obj, indent=1), encoding="utf-8")
    dump("state.json", state)
    for mid, d in details.items():
        dump(f"model_{mid}.json", d)
    ks = keys_state(state)
    dump("state_keys.json", ks)
    for m in ks["models"]:
        dump(f"model_{m['id']}_keys.json", waiting_detail(m))
    dump("state_stopped.json", stopped_state(state))
    print(f"wrote {len(list(OUT.glob('*.json')))} files to {OUT}")


if __name__ == "__main__":
    main()
