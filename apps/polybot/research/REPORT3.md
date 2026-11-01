# Round 3: the model a follower can actually trade

Collected and run 2026-10-31. Code: `collect3*.py`, `build3.py`, `prep3.py`, `fillmodel3.py`, `analyze3.py`,
`analyze3_weather.py`, `posthoc3.py`, `validate3_fills.py`, `inventory3.py`. Full tables: `TOURNAMENT3.md`, `WEATHER3.md`.

## Bottom line

1. **Copying leaders five minutes late, at the price a $5 buy really pays, has no edge we can detect.** Across 51,290
   signals from the bot's pool, the follower makes −7.7% per trade. The same signals make −2.8% at the leader's own
   price and −15.1% half an hour late.
2. **No model predicts outcomes much better than the price.** The best gain in log loss over the market alone is
   0.0016 (0.5638 → 0.5622), and AUC moves from 0.772 to 0.774. Every model's top 10% of "edges" loses money: −2% to
   −12% for the round-3 families, −26% for the round-2 model.
3. **The shipped model** is a market-anchored, heavily regularised logistic regression, with one position per event and
   at most 25% of the book behind one leader. Out of sample it made **+1.1% per trade (90% CI −4.1 to +6.5) on 940
   signals**: +1.4% in the selection weeks and +0.4% in the holdout, which includes the live paper period. Through the
   bot's budget that is **≈ +$1 to +$4 a week at $5 a trade, CI about −$14 to +$23**.
4. It beats naive copying by +6.9 points per trade (P 98%) and the day board's sports bets at 40–60¢ by +5.8 (P 94%).
   It **does not beat two live setups on point estimate**:
   - The current `learned` rules made +10.2% on 162 signals (CI −7 to +28, P(new better) 21%).
   - The live `day4` rules (four day-board leaders agree), replayed on the same Sep 15–Oct 27 days, made +2.4% a trade
     and +$8.3 a week against the new model's +0.9% and +$2.0. The gap is +$6.7 a week (90% CI −$26 to +$39, P(day4
     better) 63%).

   Neither result is significant, but day4 is the front-runner on the evidence.
5. **Weather:** at the bot's own timing, the market prices temperature buckets better than the bot's five forecast
   models (log loss 0.344 against 0.396), and nothing forecast-based beats it. The +2°C rule is +4.0% per trade when it
   buys at 12:00 UTC the day before (CI +1.3 to +6.7) and +1.3% when it buys just before the city's day starts.

6. **No model of the price alone works either (added 2026-11-01).** On a random sample of every sports market Gamma
   lists, kick-off underdogs win 0.2–2.5 points *less* often than their price says, and at the prices people actually
   paid, outcomes are calibrated to about a point, which is less than the taker fee. An apparent +36%-a-bet price
   model came from three backtest traps (market list built from wallets, a look-ahead volume filter, stale in-play
   history). See "A model of the price itself" below. After this and the day4 searches, no setup has a demonstrated
   edge; live paper weeks are the only remaining test.

## Data

| dataset | what | dates | size |
|---|---|---|---|
| leaderboards | 58 boards (4 windows × 11 categories, by profit and by volume), 72,000 rows, 25,871 distinct wallets | snapshot 2026-10-31 | 11 MB |
| control takers | 94,754 wallets trading in 1,368 randomly drawn resolved markets; 3,000 seen in 2+ markets join the universe | markets ending 06-01 → 10-30 | - |
| wallets | 10,325 collected of 10,332: 6,541 on a profit board, 3,784 not picked by profit (3,000 controls + volume boards) | 2026-10-31 | 3.2 GB |
| positions | 9.09 M CLOSED + 2.32 M REDEEMABLE + 2.60 M OPEN rows (≤3,000 per status per wallet) | full history | in wallets |
| fills | 21.45 M fills, maker and taker; newest 6,000 per wallet (2,263 wallets truncated: their signals start where their fills do) | 06-01 → 10-31 | in wallets |
| markets | 1.62 M touched by positions and fills; 1.40 M closed with an outcome; 174,857 not found (their signals are dropped) | - | 313 MB |
| order books | 203,626 snapshots of ~6,000 markets every 20 min, top 12 levels a side, $1–$100 walk costs | 10-31 14:00 → 20:00 UTC | 44 MB |
| price paths | 289,488 midpoint spans, signal to resolution, ≤15 days each (176,525 at 5-minute buckets, 112,963 hourly) | 5-minute buckets only exist from ~09-01 | 255 MB |
| thin-market tapes | 40,078 markets (lifetime volume < $250k), every taker trade back to the first signal; all markets with a pool signal | signals since ~08-31 | 81 MB |
| temperature markets | 150,861 (102,109 highest, 48,752 lowest), 54 cities, 53 resolving stations | 05-01 → 11-02 | 25 MB |
| METAR / forecasts | every report of 53 stations (IEM); Open-Meteo previous-day and two-days-before runs of the bot's 5 models | 04-29 → 11-02 | 34 MB |
| temperature tapes | 42,537 highest-temperature markets, every taker trade | days 08-15 → 10-31 | 79 MB |
| signals (panel) | 986,895 resolved, priced signals from 6,698 wallets in 117,363 markets; 840,310 priced 2–98¢ go into the models | 08-01 → 10-31 | 168 MB |

Gaps:
- Only resolved markets can be scored. October's long-dated signals are mostly unresolved, so the last folds lean
  short-horizon.
- Prices before 1 September are hourly; Polymarket keeps 5-minute buckets for ~60 days.
- 85% of the non-pool, profit-board signals were left unpriced to fit the API budget (pool and unbiased wallets were
  priced first).
- Book depth is a 6-hour sample, not a history.

## How the test works

- **Unit = what the bot sees.** A signal is the moment a wallet's net buy of an outcome over the 48-hour lookback first
  reaches $10, at most once per outcome per 72 hours. "Net" means money into this outcome minus money into the other.
  The follower acts 5 minutes later.
- **The follower's price.** The CLOB midpoint at the next 5-minute bucket, plus what a $5 market buy pays above the mid
  in that kind of market. That cost comes from the book snapshots by type, price and time to the end, in play or not.
  - In thin markets the price is the first real taker trade after the signal. 255,247 signals are priced this way; a
    further 53,361 with no taker buy within 2 hours are dropped as unfillable.
  - Checked against the bot's own 244 paper buys, which paid above mid: +0.5¢ in sports (model +0.5¢), +0.6¢ median
    elsewhere (model +0.5¢), +1.5¢ in weather (model +2.5¢).
- **Scored at resolution**, after the taker fee (shares × rate × p × (1 − p)).
- **Features come from the bot's own code.** `polybot/features.py` builds the leader's record and the signal's inputs:
  - The record uses CLOSED + REDEEMABLE positions scored at resolution, the newest 300, as of the last 6-hourly pack
    refresh. Negative-risk "No" rows are dropped.
  - `tests/test_features.py` proves the research builder and the bot produce identical vectors.
- **Walk-forward by time.** Weekly folds from 1 September; each trains only on signals resolved before the fold. Every
  family gets the same tuning budget: 6 configurations, picked by log loss on the last 14 days, re-tuned every third fold.
- **Every qualifying pool signal counts.** The pool is the 30-day profit board's top 200, rebuilt day by day from
  resolved P&L.
  - Policy is chosen per fold on earlier folds only, by mean return × the buys a day the budget allows: threshold,
    price band, horizon, in play.
  - One position per event and a 25% leader cap are design rules.
  - The family is chosen on Sep 15 → Oct 19. **Oct 20 → 31 is a holdout nobody chose on.**
  - The bot's real book (10 a day, 20 open, 3 per cycle, carried across weeks) is run alongside as a check.
  - Confidence intervals resample days.

## What predicts

| model (out of sample, 725,248 signals) | log loss | vs market | AUC | calibration slope | top 10% of pool by expected return: predicted → realised |
|---|---|---|---|---|---|
| market price only (isotonic) | 0.5638 | – | 0.772 | 1.01 | +0.9% → −6.2% |
| **logistic, anchored on the price (shipped)** | 0.5631 | −0.0007 | 0.773 | **1.00** | +2.7% → −10.6% |
| boosted trees, anchored | 0.5623 | −0.0015 | 0.774 | 1.00 | +5.5% → −11.7% |
| boosted trees, raw (round-2 style, retrained) | 0.5627 | −0.0011 | 0.774 | 1.00 | +6.2% → −10.9% |
| boosted trees + trader embedding | 0.5623 | −0.0016 | 0.774 | 1.00 | +5.2% → −9.1% |
| trader skill with shrinkage (empirical Bayes) | 0.5635 | −0.0003 | 0.773 | 0.99 | +0.1% → −8.7% |
| per-category trees | 0.5649 | +0.0010 | 0.771 | 0.96 | +11.6% → −8.3% |
| ranking objective (lambdarank by day) | 0.5666 | +0.0028 | 0.772 | 0.90 | −0.9% → −2.4% |
| boosted trees, time-decayed (30-day half-life) | 0.5623 | −0.0015 | 0.774 | 0.99 | +6.0% → −9.3% |
| **round-2 model.json, fed what the live bot feeds it** | 0.6035 | **+0.0396** | 0.732 | **0.83** | **+105.5% → −26.0%** |

- **Price decides; leaders add almost nothing.** Dropping any feature group changes log loss by at most 0.0002; dropping
  the market group moves P&L most (−4.9% per trade).
- **The tails are adverse selection.** Where a model disagrees most with the market, the market is right.
- **The round-2 model's failure is partly train/serve skew.** It was trained on win rates that count resolved losses, but
  the live bot feeds it CLOSED positions only: the average leader's win rate is 0.71 live against 0.52 with resolved losses.
- **Lag is the cost.** Pool signals make −2.6% if bought before the leader, −2.8% at the leader's price, −7.7% five
  minutes later, −15.1% after 30 minutes and −18.7% after 60.
- **Selection bias is visible.** Wallets no profit board picked make −11.1% per signal against −7.7% for the pool.
  Under the shipped policy, unbiased wallets make −1.6% [−4.6, +1.6] and profit-board wallets +2.6% [−0.2, +5.3].

## The bot's portfolio, out of sample (every qualifying pool signal; $/week at $5, ≤10 a day)

| | signals | mean / trade [90% CI] | select (Sep 15–Oct 19) | holdout (Oct 20–31) [90% CI] | $/week | bot's book check |
|---|---|---|---|---|---|---|
| **logistic, anchored (shipped)** | 940 | **+1.1% [−4.1, +6.5]** | +1.4% | +0.4% [−10.4, +11.7] | +3.94 | 301 trades, +0.5% |
| trader skill (shrinkage) | 369 | +4.7% [−8.6, +22.3] | −2.6% | +11.6% [−10.7, +45.4] | +13.34 | 156, −18.4% |
| boosted trees, anchored | 978 | −0.9% [−5.7, +3.6] | +0.2% | −2.6% | −3.04 | 354, +1.3% |
| boosted trees + trader embedding | 852 | −0.5% [−6.5, +5.9] | **+3.3% (picked)** | **−6.0% [−16.5, +6.0]** | −1.59 | 341, −4.1% |
| boosted trees, raw | 812 | −4.1% [−11.0, +2.5] | −5.8% | −0.9% | −14.26 | 347, +0.1% |
| market price only | 448 | −1.4% [−8.3, +6.1] | −0.9% | −2.8% | −4.72 | 307, −0.4% |
| round-2 model, new policy | 468 | +3.9% [−4.9, +12.7] | −1.5% | +15.3% [+4.5, +24.8] | +13.55 | 270, +4.9% |
| **current `learned` rules** (round-2 model, 2 pool leaders, ≤5¢ above them, 5 a day) | 162 | +10.2% [−7.0, +28.1] (capped +7.5%) | +9.6% | +11.6% [−20.9, +50.3] | +12.27 | 130, +16.9% |
| **day board's sports bets at 40–60¢** | 1,730 | **−4.5% [−8.2, −0.7]** | −5.9% | −2.3% | −15.91 | 409, −3.0% |
| copy every pool signal, 5–95¢ | 22,914 | **−5.8% [−6.8, −4.8]** | −5.9% | −5.6% | −20.33 | 464, −15.4% |

- **Live paper period** (Oct 28–31, inside the holdout): the shipped model 38 trades, +1.0%; day-board sports 208, +1.3%;
  `learned` rules 10, +118.2% (8 of 10 won, bought at 20–69¢: a hot streak on 10 trades).
- **The live run's +33% on in-game day-board sports (22 markets) was a hot week.** Over the seven scored weeks the rule is
  −4.5%.
- **Post hoc** (`posthoc3.py`, the idea came from seeing the `learned` result): letting the grid also require a second
  pool leader and ≤5¢ above the leaders gives the shipped model +2.4% [−3.3, +7.8] on 910 signals. That is not
  enough to change what ships.

### The live `day4` setup, replayed (added 2026-10-31, `collect3_group.py` + `replay_day4.py`)

day4 (four day-board leaders agree) made +30% a trade on its first 22 paper trades. Replaying its exact rules on Sep 15 → Oct 27, weeks the
paper run never saw, gives a much smaller number. The replay uses the complete fills of the 29 wallets it copied (3.2 M fills), 5-minute
prices plus book cost, follows their sells and holds the rest to resolution.

| replay | trades | mean / trade [90% CI] |
|---|---|---|
| day4's rules, the 29 wallets, Sep 15–Oct 27 | 424 | +2.4% [−3.8, +8.8] |
| same, bought before the game (subset) | 311 | +9.3% [+0.8, +18.1] |
| same, bought in play (subset) | 107 | −13.5% [−30.3, +4.9] |
| rule changed to skip games in play | 413 | +3.1% [−4.6, +11.0] |
| 36 wallets (adds today's day-board top 25) | 424 | +1.7% [−4.7, +8.2] |
| replay of the live days, Oct 28–31 (real paper record: 22 trades, +30%) | 33 | +18.3% [−14.5, +49.9] |

- Weekly results swing from −8.2% to +16.7%.
- The replay matches the live days in direction but not trade for trade: 8 of 26 outcomes are the same, because the
  live board changes daily.
- day4's long-run edge looks like about +2% a trade. At ten buys a day that is ≈ +$460 a year at $5 a trade
  (90% range −$780 to +$1,630).

### A model on day4's own signals (`consensus_model.py`, `export_consensus.py` → `model_day4.json`, setup `day4model`)

Every moment a consensus forms or strengthens among the 29 wallets is a candidate: 73,219 moments, 20,805 of them priced
and resolved. A market-anchored logistic regression scores each candidate on `features.consensus_context`'s inputs:
- the score and the agree / against counts, and the money behind the agreement;
- the price against the leaders' own price;
- the hours to the end and to kick-off, and whether the game is in play;
- the league, and whether it is a spread, a total or a moneyline.

The policy is chosen per week on the weeks before it, and day4's book decides what is bought. Scored Sep 29 → Oct 31:

| | trades | mean / trade [90% CI] | $/week at $5 | holdout Oct 20–31 |
|---|---|---|---|---|
| day4's rule (hold to the end) | 313 | +7.6% [−1.7, +17.1] | +25.3 | +1.3% [−16.0, +21.3] |
| day4's rule, settings re-tuned weekly | 309 | +10.8% [+2.9, +18.6] | +35.5 | +6.6% |
| **consensus model, λ = 1000** | 295 | **+12.0% [+4.5, +19.4]** | **+37.7** | **+16.9% [+4.4, +28.9]** |
| same model, λ = 100 / λ = 10 / boosted trees | 291 / 288 / 236 | +6.3% / +4.0% / +0.9% | +19.3 / +12.2 / +2.2 | +3.0% / −3.6% / −0.7% |

- **It beats day4's rule** by +$12.7 a week (90% CI −$18 to +$42, P(better) 75%), with the best holdout of all.
  **Later, more powerful checks undo this** (next section): over every qualifying signal, its filter does worse than
  taking them all.
- **Its lead is not proven.** It was the best of four model variants, and a weekly re-tuning of day4's own settings
  comes close.
- **What it changes:** it keeps day4's signals, drops those it expects to lose on, and buys the best first when the
  budget binds.
- **What it weighs:** more money behind the consensus and favourites count for a bet; NFL, spreads, dissenting leaders
  and a long time to the end count against.
- **Testing:** `tests/test_features.py` checks that the bot computes the same inputs as the research. It runs as the
  `day4model` paper setup beside `day4` and `day4pre`.

### Looking for a better day4 model (2026-11-01): what did not work, and why day4's edge is doubtful

Checks run on the same replay data (`consensus_filter.py`, `day4_checks/*.py`, inside the research image with `/data` = `data/polybot/research3`):

| idea | result |
|---|---|
| sell at kick-off (closing-line value) | −5.5% a trade (1,449 pre-game buys, sd 12%): the price does not move toward the consensus before games (−0.4¢) |
| who agrees (leaders' records), price momentum | worse than day4model by $30–46 a week; more inputs fit noise on 20k candidates |
| more consensus levels / a kick-off window in the policy | worse (nested choice picks up noise): +9.7% vs +12.0% |
| **every qualifying signal, not just the budgeted 10 a day** | **day4's rule +3.8% a signal (1,339; CI −1.6 to +9.2); the day4model filter keeps 714 at +0.7%. No model filter beats taking them all; the one picked on the earlier weeks lost 7% on the holdout. day4model's book advantage came from which 10 a day it happened to buy.** |
| 144 top sports traders instead of the 29 | rule +2.9% a signal (2,188); filters move it ±2 points; no robust winner |
| maker entry (rest a bid, no fee) | −2.6% vs +3.8% taker: bids fill when the price moves against the consensus (filled bets −13%) |
| faster entry | 5 min sooner −2.5 to −3.3 points, 5/15/60 min later −1 to −1.6 / −4 to −5 / −9 to −14: the 5-minute poll is already near the best moment |
| rule tweaks across both groups and both periods | only "skip 30–50¢ underdogs" improves all four cells (29: +12.5% / +3.6% vs +8.3% / −6.9%; 144: +6.2% / +1.0% vs +4.9% / −1.4%); 11 rules were tried and the groups overlap, so this is weak |

**The point-in-time consensus test.** In the 986k-signal panel, the monthly top-200 pool is rebuilt from resolved
P&L day by day, so there is no hindsight. Pool signals on outcomes other pool members also bought in the previous 48
hours lose, and lose more the bigger the crowd (Sep 15 → Oct 31, 5–90¢, follower price):

| other pool members on the same outcome | signals | mean / signal [90% CI] |
|---|---|---|
| 0 or more (all pool signals) | 36,137 | −4.4% [−5.5, −3.2] |
| 1+ | 14,364 | −1.8% [−4.0, +0.2] |
| 3+ (4+ agreeing) | 4,653 | −4.6% [−8.5, −1.0] |
| 5+ (6+ agreeing) | 1,876 | −10.6% [−16.5, −5.2] |
| sports, 3+ / 5+ | 4,464 / 1,816 | −3.3% / −9.5% |

Counting each outcome once (`fade_crowd.py`, the first moment the crowd reaches a size; follower price, taker fee):

| first moment … pool traders agree | outcomes | follow it | select / holdout | fade it (buy the other side) |
|---|---|---|---|---|
| 4+ | 1,738 | +3.2% [−1.7, +7.9] | −2.8% / +8.6% | −6.9% [−11.3, −2.5] |
| 5+ | 1,047 | −1.9% [−6.5, +2.2] | −6.3% / +1.6% | — |
| 6+ | 668 | −7.8% [−12.7, −3.3] | −12.3% / −4.7% | −2.2% [−8.7, +4.6] |

Consensus among top traders, picked without hindsight, has no edge; crowded consensus loses, and fading it loses too. day4's positive replays
use groups taken from today's leaderboards (which reflect their recent wins), and its live +30% is 22 trades. The
safest reading: none of the copy setups, day4 and day4model included, has a demonstrated edge. Keep them on paper,
and judge them on weeks of live record, not on these replays.

### A model of the price itself (2026-11-01): three backtest traps, and no edge once they are removed

Scripts: `collect3_kickoff.py`, `collect3_universe.py`, `kickoff_model.py`, `kickoff_vs_real.py`, `kickoff_liquid.py`,
`kickoff_unbiased.py`, `kickoff_books_now.py`, `calibration_real.py`, `day4_checks/real_prices.py`. No leaders
involved: buy one side of a sports market (moneyline, child moneyline, spread, total) at kick-off or in play and hold it
to resolution, paying the taker fee (5% × (1 − price) per dollar; 3% on NFL/CFB).

**The first version looked like a goldmine.** 48,190 resolved markets from `markets.jsonl.gz` that traded ≥ $5k, game
start Sep 3 → Oct 31, with 5-minute price history. Underdogs priced 2–10¢ five minutes before kick-off won 14%, and
10–20¢ ones won 24%. A weekly-refitted price model "made" **+36% a bet [+31, +41] on 23,837 bets, $150+ a week**,
+33% in the selection weeks and +46% in the holdout. Every variant agreed. Three traps produced all of it:

1. **The volume filter looks ahead.** An upset brings in-play trading, so thin markets pass "≥ $5k traded" more
   often when the underdog wins. In a random 9,000 of the 68,900 markets under $5k, 2–20¢ underdogs win only +1.0
   point above their price, against +8.1 above $5k.
2. **The market list depends on the wallets.** `markets.jsonl.gz` was built from the collected wallets' positions and
   fills, and the positions API leaves out resolved losing positions. Gamma lists **262,825** resolved markets in the
   window against the 117k on file. **In a random 12,000 of all of them, underdogs win less than their price, the
   usual favourite–longshot bias:**

   | 5 min before kick-off, underdog at | markets | won vs price | buy it (+0.5¢, fee) [90% CI] | select / holdout |
   |---|---|---|---|---|
   | 2–10¢ | 1,826 | 4.9% vs 5.7¢ | −32.6% [−45.8, −19.2] | −33% / −30% |
   | 10–20¢ | 1,589 | 14.6% vs 14.8¢ | −8.6% [−18.3, +1.3] | −10% / −2% |
   | 20–30¢ | 1,692 | 23.2% vs 25.1¢ | −12.8% [−19.0, −5.8] | −13% / −12% |
   | 30–40¢ | 1,545 | 33.7% vs 34.8¢ | −7.0% [−13.7, −0.9] | −8% / −2% |
   | moneylines 20–30¢ / spreads + totals 2–10¢ | 676 / 1,683 | −0.9 / −0.8 pts | −9.3% / −32.6% | |

   (The 40–50¢ row is mostly untraded markets still showing their opening 50¢. Those are not real prices.)
3. **In play, the 5-minute history is stale.** On 430,151 real buys in these markets, the history before kick-off
   matches the price paid to 0.3¢. In play, a history of 6.7¢ meant buyers paid 14.8¢, and 15.5¢ meant 22.2¢. Measured
   against the price paid, the in-play "edge" disappears (+8.0 → −0.04 points at 2–10¢).

Where real buys confirm the price (≥ 3 buys in the hour before kick-off, history within 0.5¢ of paid; 6,121 markets),
pre-game underdogs win within ±1.2 points of their price and buying them loses 1.7–13.6%. Live books on 276 upcoming
games are tight (best ask = history + 0.5¢, even below $1k traded). So the pre-game history is a buyable price, and
the edge was in which markets were counted. **At the prices people actually paid (`calibration_real.py`, first buy
per outcome), sports outcomes are calibrated to about a point,** smaller than the 2.5–4.7-point fee on a 10–50¢ buy:

| price paid | before the game (last hour): won − paid, after fee | in play, random control wallets: won − paid, after fee |
|---|---|---|
| 2–10¢ | +0.05 pts, −6.5% | −1.09 pts, −21.5% |
| 10–20¢ | +0.34, −3.2% | +0.21, −4.8% |
| 20–30¢ | +0.44, −1.3% | +0.30, −2.1% |
| 30–40¢ | −0.82, −5.6% | −0.81, −5.6% |
| 40–50¢ | +0.88, −0.3% | +0.74, −0.9% |
| 50–98¢ | within ±0.9, −0.4% to −3.6% | −0.58 / −0.80, −2.9% / −1.8% |

Profit-board wallets' in-play underdog buys look good (+11% at 20–30¢). They were picked from today's board,
though, and the random controls lose on the same buys.

**Does the stale-price trap touch the day4 replay?** It priced the follower at the history plus the fill model's cost.
Re-priced at the panel's real buys of the same outcome within ±3 minutes (414 of 1,339 signals): **+3.3% → +2.4% a
trade** (underdogs +6.7% → +4.4%, in play +2.1% → −0.9%). The 144-trader group goes from +5.4% to +3.6%. A 20-minute
window says −5.1%, but that window lets game events in after the signal. The replay is a little optimistic, and its
conclusion stands: positive, not significant.

**Verdict:** there is no price-only model to build. Sports prices are right to within the fee, with the usual mild
longshot bias. A sports backtest needs three things: a market list taken from the exchange (not from traders'
histories), no filter on what a market later traded, and prices confirmed by real trades in play. The underdog
strategy's own backtest (+1–5% on 349 games, from price history) has the stale-price problem, and it is −25% over 40
live paper trades.

## Why this is the best way to build this model

The evidence does not support "this model makes money". It supports a narrower claim: of everything tried, this is
the way to build the model that does not lose. Its estimate is honest, and it fixes every failure of the previous
model that the data exposed.

- **It is calibrated where the others are not.** Slope 1.00 and intercept +0.02, against the round-2 model's 0.83, which
  predicted +105% for its top decile and got −26%.
  - The policy trades on probability minus price, so calibration is the whole game. The round-2 model is worse than the
    price itself (log loss +0.040).
  - Anchoring on the follower's price (logit(price) + corrections) makes "no information" mean "no trade".
  - It is not immune to the tail: its own top 2% by expected return lost −25.6%, like every family's (−20% to −30%).
    That is why its policy, chosen walk-forward, trades a broad band (30–70¢, threshold 0, ≤14 days, no games in play)
    and not its most extreme "edges".
- **It is the only family positive in both the selection weeks and the holdout** with more than 900 signals (+1.4% and
  +0.4%), and it has the tightest interval of any positive estimate (±5.3 points). The family the selection rule
  preferred, trees plus a trader embedding, fell from +3.3% to −6.0% on the holdout: exactly the overfitting the
  holdout exists to catch.
- **It beats the rules people actually run.**
  - Naive copying of the pool: +6.9 points per trade (5th percentile +1.3, P 98%).
  - The day board's sports bets at 40–60¢: +5.8 points (5th percentile −0.1, P 94%).
  - Price alone: +2.6 points (P 69%).
  - Trees, raw / anchored / time-decayed: +5.3 / +2.0 / +1.3 points.
- **It fixes the live run's structural flaws.**
  - Events held on both sides at once: 0 (279 without the per-event rule; 27 in the budgeted book).
  - The busiest leader is behind 5% of its trades, the top five 19% (live: one wallet behind 45% of open positions).
  - It skips games in play (−5.1% per trade in the book when allowed).
  - It only acts on bets under 10 minutes old.
- **Its numbers are realistic.**
  - 475 of its 940 trades are priced off the real tape.
  - At the 75th-percentile book cost it is −0.0%, not a collapse.
  - Its features are computed by the bot's code and tested identical.
  - Its record counts resolved losses.

### Where it would lose

- **Against the live `day4` rules** (replayed with complete fills): +$8.3 against +$2.0 a week over the same 43 days, a gap
  of +$6.7 (90% CI −$26 to +$39). The replay picks its 29 wallets with hindsight (the ones day4 copied this week), which
  flatters it somewhat. day4 was not in the original tournament; it should have been.
- **Against the current `learned` rules**, on point estimate: −9.2 points per trade, P(new better) 21%. That comparison
  rests on 162 signals; two weeks of side-by-side paper trading will say more than another backtest.
- **Against the round-2 model run under the new policy:** −2.8 points (P 33%), mostly from that setup's holdout (+15.3%).
- **If the follower is slower.** The same picks make −1.8% at 30 minutes and −5.7% at 4 hours.
- **On wallets no leaderboard picked** (−1.6%). Part of its small positive number is the October boards' selection of
  their own winners.
- **With five trades a day instead of ten:** the book check falls to −7.8% per trade.

## Expected live return and capacity

- **Per trade:** +1.1% (90% CI −4.1 to +6.5) on qualifying signals; +0.5% in the budgeted book check.
- **Per week at $5 a trade and ≤10 a day:** ≈ +$4 (CI about −$14 to +$23); the book check gives +$1.
- **Capacity is not the constraint; edge is.** A $5 buy costs +0.5¢ over mid in sports before kick-off, with
  ~$12,000 resting within 1¢ of the ask. On 94% of its trades a $50 stake costs no more extra than half the
  expected edge, so ~24 qualifying signals a day could absorb ~$1,100 a day. That only matters if the edge turns out
  to be real.

## Weather

Full tables: `WEATHER3.md`.

- **The forecast adds nothing at real prices.** At every timing the market's last trade beats the per-city ridge on
  the five models (log loss 0.332–0.344 against 0.371–0.396), and blending never beats the market alone.
- **The +24% forecast "edge" was look-ahead.** It paired a 12:00 UTC price with day-before runs issued later that day
  for the Americas. With runs certainly out in time it is −5.3%; at the bot's timing it is +2.7% ± 15.
- **Mids are worse than useless here.** Buckets the midpoint history priced at 35–50% resolved 18% of the time
  (empty books), which made a mid-based backtest show +60% per trade.
- **The +2°C rule is small, positive and timing-dependent:** +4.0% [+1.3, +6.7] (544 trades) and +4.7% [+1.8, +7.4]
  with older runs when buying at 12:00 UTC the day before; +1.3% [−1.7, +4.4] just before the city's day starts.
- **Books are thin:** a $5 buy costs 2–5¢ above the mid; $15–140 rests within 1¢ of the ask.
- **Re-check 2026-11-01: the rule's +4.7% also looks ahead.** The backtest priced each buy at the *next* taker trade
  on that side within 3 hours, and skipped markets where nobody traded after the decision. That trades only when
  someone else later bought the same side. Priced at what was known at the decision (`--entry-last`: last trade
  before it + the book's typical $5 cost), with two-days-before runs:

  | +2°C rule (No at max forecast + 2°C, No 50–95¢) | trades | mean [90% CI] | Aug 15 → Sep 14 | Sep 15 → Oct 19 | Oct 20 → 31 |
  |---|---|---|---|---|---|
  | 12:00 UTC the day before, next taker price (as reported) | 505 | +4.7% [+1.8, +7.4] | +4.0% | +3.5% | +10.3% |
  | 12:00 UTC the day before, known at decision | 540 | **+2.2% [−0.7, +4.9]** | +0.4% | +1.0% | +9.1% |
  | just before the city's day (bot timing), known at decision | 537 | **+0.1% [−2.6, +2.8]** | −0.5% | +1.4% | −2.9% |

  So the forecast strategy has no demonstrated edge either. Its best honest version is about where day4 is (+2%,
  not significant), and most of that came in the last 12 days.

## What changed in the bot

- **`polybot/features.py`** (new): the record and signal inputs, shared with research.
- **`pack.py`** also fetches REDEEMABLE positions, so the record counts resolved losses (`Record.rec3`).
- **`model.py`** reads round-3 files (`version: 3`: kinds `lgbm`, `linear`, `skill`, anchored on the follower's price;
  the score is expected return per dollar after the fee). Old files still load.
- **`model.json`** is the shipped logistic. `model_r2.json` is the old file, used by `learned`, `weather5` and `weather10`.
- **New setup `edge` (Edge model, round 3)**: single-leader signals, fresh bets only, scored at the stake's real price,
  one position per event, a 25% leader cap, no games in play, 30–70¢, ≤14 days to the end.
- **The poly stack's memory goes from 256 MB to 512 MB.** It was at the limit, with ~190 MB in swap, before this change.
