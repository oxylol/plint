# Round 4: who gets paid (a model of my own, and why it stays on paper)

Run 2026-11-10. Filip's brief: look at polybot, draft a model of your own, and since what is in there mostly failed,
look elsewhere. Code: `collect4_*.py`, `build4.py`, `r4_*.py`. Data: `data/polybot/research4` (2.5 GB).

Rounds 1–3 asked who is right (leaders, forecasts, a price model) and bought from them at the ask. This round asks
who is paid: every Polymarket trade has a taker who pays the spread and the fee, and a maker who does not. I tested
whether a small, slow bot can sit on the paid side, in five places. It cannot, and the one rule that survived every
correction earns about what a savings account does.

## Bottom line

1. **No new family has an edge.** Weather makers lose to informed takers. A volatility anchor does not beat the
   price in crypto ladders. Resting bids lose in every market family. Favourites are priced right.
2. **The draft model is "carry"**: buy geopolitics favourites that have sat quietly above 90¢ and still have 10–45
   days to run, hold to resolution. On every such market Gamma lists since January 2024 it makes
   **+0.48% a position (90% CI −0.39 to +1.31), 1,818 positions, 96.1% wins at 95.7¢**. Through a $5 book that is
   +$20 in 2.4 years on $223 of capital in use: **about +4% a year**, with a worst month of −$24.
3. **I would not put real money on it**, and I have not added it to the bot. It is the least bad thing tested, not a
   good one. `r4_carry_live.py` lists what it would hold today (56 markets on 2026-11-10).
4. **Three numbers looked like an edge on the way and were not.** Each died when the sample was made honest. They
   are the useful part of this round: traps 5–7 below.

## What outside evidence says

Read before testing (sources in the session notes; the two largest are Becker's 72M Kalshi trades and Bürgi, Deng &
Whelan's Kalshi paper):

- Takers lose and makers gain on Kalshi, by about 1.1% of stake each way; the gap is largest in world events and
  entertainment (5–7 points) and smallest in finance. Before 2024 makers lost.
- On Polymarket the same longshot bias is measured as weaker: about +1¢ a share for buying at 80–90¢, before fees.
- About $40M of arbitrage was taken on Polymarket in a year, by fast accounts.
- No study shows a small account earning from liquidity rewards. The smallest makers earn the least per fill.

So the only replicated edge is "be the maker against people buying longshots", worth about a point. Everything
below tests whether that point exists where this bot could reach it.

## 1. Weather: the takers are the informed side

41,690 highest-temperature markets, 78 days, 5.75M taker trades (`r4_weather_trades.py`, `r4_weather_makers.py`).
Each trade seen from the maker's side, held to resolution, no fee:

| | maker return, $5 of each trade | dollar-weighted |
|---|---|---|
| all trades | −0.15% [−0.78, +0.59] | −0.75% [−0.91, −0.61] |
| before the city's day | −1.95% [−2.69, −1.19] | −1.78% |
| in play | +0.87% [−0.13, +1.75] | −0.41% |
| before the day, takers buying No | −3.78% [−4.99, −2.59] | −4.19% |

A bid that rests below the market is worse off than the average maker, because it is filled when the price moves
through it (`r4_weather_resting.py`: a $5 bid each hour at the last two-sided prints minus d, filled only by a print
through the level):

| bid | before the day | in play |
|---|---|---|
| Yes, at the mid to −3¢ | −7.5% to −11.6% | −11.6% to −15.8% |
| No, at the mid to −3¢ | −1.9% to −3.1% | −6.2% to −6.7% |
| 8¢ under the mid | +3% to +5% on 1.5% of orders, CI touching 0 | −6% to −8% |

The 25% maker rebate is worth 0.4–0.9% of a fill. It does not cover this.

## 2. Crypto ladders: an anchor from Binance does not beat the price

Daily "Bitcoin/Ethereum/Solana/XRP above K" markets, 46 expiry days, 2,011 markets, 313k trades, $30.9M
(`r4_crypto_anchor.py`, `r4_crypto_build.py`, `r4_crypto_report.py`). The anchor is the chance of finishing above
the strike from spot one minute stale and the distribution of volatility-scaled moves in five years of Binance
history before the test window (2-sigma moves happen 3.5% of the time against the normal's 2.3%, 3-sigma 1.0%
against 0.13%).

- Log loss: trade price 0.2636, anchor 0.2822. In a regression the price takes all the weight (1.03) and the anchor
  none (−0.06).
- A maker bidding the favourite side at the anchor minus 0–5¢, filled by through-prints: −3.3% to −4.1% a fill.
- In these seven weeks the cheap side won more often than its price (10–20¢ won 18.3%): no longshot bias to sell.

## 3. Every other family: resting bids lose, favourites looked underpriced

15,517 markets ending 2026-09-20 → 11-08 outside sports, weather and the short crypto markets, listed from Gamma
day by day (`collect4_universe.py`, `build4.py`, `r4_families.py`).

- Bids in the middle of the book (20–80¢) lose 1–20% a fill in nine of ten families (tweet counts: about zero).
  Cheap-side bids lose 20–68% in most.
- Favourite-side bids were positive only in geopolitics (+4.6% [+1.6, +7.2]) and culture (+3.3% [+0.1, +6.0]).
- Counting every trade, a maker buying a favourite at 95–99.5¢ made +0.9% to +1.6%. Across 90–99.5¢ without the
  five biggest series, buying favourites made −0.23%.

That pointed at geopolitics, where there is no fee, and at a longer history.

## 4. Carry, on 2.4 years of geopolitics

Every closed event under the tags geopolitics, world and middle-east since January 2024: 2,584 events, 9,708 traded
markets, 9.56M trades (`collect4_events.py`, `collect4_tapes.py`, `build4.py`).

The rule, one position per market: the first hour at which the favourite's ask is 90–98.5¢, its mid has stayed at
or above 90¢ for 72 hours, 10–45 days remain, and both sides traded in the last two hours. Buy at the ask (the last
print on that side), hold to resolution, no fee.

| sample | positions | paid | won | return a position [90% CI by series] |
|---|---|---|---|---|
| markets with a complete tape, replayed on prints | 1,488 | 95.7¢ | 97.0% | +1.41% [+0.62, +2.16] |
| the 1,055 busiest markets, replayed on 12-hour mids | 330 | 95.7¢ | 92.1% | −3.73% [−6.71, −0.98] |
| **all** (`r4_carry.py`) | **1,818** | 95.7¢ | 96.1% | **+0.48% [−0.39, +1.31]** |

- Median hold 25 days. Half-years: +1.0%, +1.4%, +0.3%, +0.5%, −1.3% (2026 H2, 101 positions).
- $5 a position, at most 3 open in a series and 10 new a day: 1,390 positions, +$20 on $223 in use (peak $900),
  18 of 29 months up, worst month −$24, largest drawdown $33.
- The stability filter is what matters: without it the prints sample makes −1.1%.
- A stop does not help (`r4_carry_stop.py`): selling when the bid touches 80/85/88¢ gives −1.4% / −1.8% / −2.0%,
  because 77–88% of stopped positions would have won.
- Pre-registered guess that quiet markets carry the premium: wrong. By dollars traded in the 72 hours before entry,
  the quiet third made −0.15%, the middle +1.56%, the busy third +2.81%.
- Check of the mids replay: on 308 busy markets fetched in full, prints give −0.81% and mids −0.91%.

The favourite's price is right to within half a point. The quiet 1.4% is the premium for the loud 3.7%.

## Traps 5–7 (1–4 are in REPORT3)

5. **Nineteen days is not a sample.** The crypto maker, bidding 2–5¢ under the anchor, made +2.6% to +6.2% a fill
   on the first 19 expiry days fetched and −3.5% on 46. One calm stretch of Bitcoin looks like an edge for selling tails.
6. **Count positions, not trades.** Most trades in a favourite happen after the outcome is nearly known, when it
   wins. Per trade, a maker buying geopolitics favourites at 90–98¢ "made" +2.6% to +4.6%; one entry per market at the
   first qualifying hour made −1.1% before the stability filter.
7. **A capped tape drops the upsets.** Tapes were fetched newest-first to 3,000 trades. An upset brings a surge of
   trading that pushes the calm weeks before it off the tape, so the capped 11% of markets held 26 of the 70 losses
   and turned +1.4% into +0.5%. Fetch busy markets in full, or replay them on the exchange's price history.

## If it were built

Not built. The spec, should a paper line be wanted for comparison:

- Setup `carry`, `CA_*`: tags geopolitics, world, middle-east; favourite ask 0.90–0.985; hourly mid ≥ 0.90 for 72 h;
  10–45 days left; spread ≤ 3¢; $5; one position per market; ≤ 3 open per series (the event slug with dates and
  numbers removed); ≤ 10 new a day; hold to resolution; no stop.
- Expect about 55 open positions ($275), one total loss in 25, and +0.5% a position. A first verdict needs about
  300 closed positions, four to five months.
- Stop it if 300 positions make less than −1%.

## What I would do instead

- Keep real money out. Four rounds, more than a dozen families, no edge above costs at this size and speed.
- Consider switching off the setups that are clearly losing on paper (underdog, credible, smart, month) so the
  page shows what is still an open question.
- A real edge here needs something this bot does not have: quotes that react in under a second with capital behind
  them, or information before the market has it.
