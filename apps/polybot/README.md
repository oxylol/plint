# Polybot

Two Polymarket strategies, paper-traded side by side so they can be compared,
with the option to go live. Runs on Plint as the `poly` stack with a page at
`https://poly.<domain>`.

## Strategy 1 · Consensus

Follow the **leaderboard**. Every five minutes the bot

1. takes the top *N* traders by profit (day / week / month / all-time board),
2. pulls every fill they made in the last *L* hours,
3. works out each leader's **stance** per market: money into one outcome minus
   money into the other, so market makers who buy both sides count as flat.
   A leader *agrees* with an outcome when the stance is at least
   `POLY_MIN_LEADER_USD`, and is *against* it below minus that,
4. **buys** when `agree − against ≥ POLY_MIN_AGREE`, with a market order of
   `POLY_STAKE_USD`, if the price is inside the band and not more than
   `POLY_MAX_ABOVE_LEADERS` above what the leaders paid,
5. **sells** a position when the same number of leaders net-sell it
   (`POLY_FOLLOW_SELLS`), and books resolved markets at 1 or 0.

### Variants: comparing consensus setups

`POLY_VARIANTS` runs extra copies of the consensus strategy with some settings
changed, each with its own paper book, so the page can compare them. The
syntax is `name:setting=value,setting=value;name2:...` where the settings are
the `POLY_*` names without the prefix. The default compares the four
leaderboard windows, each at the base threshold and at four agreeing:

```
POLY_VARIANTS=week4:min_agree=4;month:window=month;month4:window=month,min_agree=4;day:window=day;day4:window=day,min_agree=4;all:window=all;all4:window=all,min_agree=4
```

Other ideas: `week4:min_agree=4` (a lower bar), `top50:top_n=50` (a bigger
pack), `politics:category=politics` (one category's board),
`fast:lookback_hours=6,min_agree=3`. Leaders and fills are fetched once and
shared, so variants are cheap. The base setup is always called `consensus`.

## Strategy 2 · Underdog

Filip's observation: the underdog gets a boost once a match begins. Every
30 seconds the bot

1. keeps a schedule of moneyline (match winner) markets starting in the next
   `UD_LOOKAHEAD_HOURS`, with the cheaper side priced between `UD_MIN_PRICE`
   and `UD_MAX_PRICE` (team markets use the cheaper side; soccer-style
   "Will X win?" markets use Yes only),
2. in the window from `UD_BUY_BEFORE_MIN` before kickoff to `UD_BUY_UNTIL_MIN`
   after, **buys** the underdog for `UD_STAKE_USD` if the market is liquid
   enough and the spread is tight enough,
3. **sells** at `+UD_TAKE_PROFIT_PCT` (live: a resting sell order at that
   price; paper: the first trade at or above it), at `−UD_STOP_LOSS_PCT`, or
   `UD_MAX_HOLD_MIN` minutes after kickoff, whichever comes first.

A quick backtest on four days of games (349 underdogs priced 10–45¢, minute
price history): from the first in-game print the underdog was up on average
about 3% after 15 minutes and 5% after 30, and the "+15% target, out after 30
minutes" rule averaged about +1% to +5% per trade before fees, winning about
70% of the time. Sports markets charge a taker fee of about that size, which
is why paper trades charge the fee too. The page will show whether anything is
left.

## Comparing them

The page's **Strategies** table lists every setup: paper (or live) P&L,
number of closed trades, win rate and average return per trade, with what is
open underneath. The consensus section has a picker to look at any variant's
leaders and signals. Positions and trades carry a strategy tag. Everything is
tracked per strategy in `data/polybot/state.json` and `trades.jsonl`.

## Paper first

Without a key the bot runs in **paper mode**: the same decisions, no orders,
fills at the prices the order book shows at that moment, minus Polymarket's
taker fee. Leave it there for a week or two before going live.

## Going live

1. On polymarket.com → profile menu → copy the **wallet address**.
2. Export the account's **private key** (Settings → Export private key for
   email/Google logins; for a browser wallet it is that wallet's key).
3. Optional, for claiming winnings automatically: Settings → API Keys →
   **Relayer API Key** (copy the key and its signer address).
4. In `/opt/plint/.env`:
   ```
   POLY_PRIVATE_KEY=0x…
   POLY_WALLET=0x…
   POLY_RELAYER_API_KEY=…            # optional
   POLY_RELAYER_API_KEY_ADDRESS=0x…  # optional
   POLY_LIVE=1
   ```
5. `plint restart poly`, then `docker exec polybot python -m polybot check`
   should show the wallet type, the pUSD balance and `approved: true`.

The bot only spends pUSD that is already in the account; it cannot withdraw.
Going live applies to both strategies; set `UD_ENABLED=0` to run only the
consensus one.

## Settings (`.env`, all optional)

Consensus:

| variable | default | meaning |
|---|---|---|
| `POLY_WINDOW` | `week` | leaderboard window: `day`, `week`, `month`, `all` |
| `POLY_CATEGORY` | `overall` | or a Polymarket category such as `politics`, `sports`, `crypto` |
| `POLY_TOP_N` | `25` | how many leaders to watch |
| `POLY_MIN_AGREE` | `5` | agree minus against needed to trade |
| `POLY_LOOKBACK_HOURS` | `24` | how far back a leader's fills count |
| `POLY_MIN_LEADER_USD` | `25` | a leader's stance below this is noise |
| `POLY_STAKE_USD` | `5` | per trade (Polymarket's minimum order is about $5) |
| `POLY_MAX_PRICE` / `POLY_MIN_PRICE` | `0.90` / `0.05` | only buy inside this band |
| `POLY_MAX_ABOVE_LEADERS` | `0.05` | never pay more than this above the leaders' average |
| `POLY_MAX_OPEN` | `20` | open positions at most |
| `POLY_DAILY_BUDGET_USD` | `50` | buys per rolling 24 h at most |
| `POLY_COOLDOWN_HOURS` | `72` | don't re-enter the same outcome sooner |
| `POLY_MIN_HOURS_TO_END` | `1` | skip markets that end sooner |
| `POLY_FOLLOW_SELLS` | `1` | sell when the leaders sell |
| `POLY_AUTO_REDEEM` | `1` | claim winnings (needs the Relayer key) |
| `POLY_POLL_SECONDS` | `300` | how often to look |
| `POLY_VARIANTS` | windows × thresholds, see above | extra setups to compare |

Underdog:

| variable | default | meaning |
|---|---|---|
| `UD_ENABLED` | `1` | run the strategy at all |
| `UD_SPORTS` | *(all)* | comma-separated slug prefixes to allow, e.g. `nba,nhl,epl,wta` |
| `UD_STAKE_USD` | `5` | per game |
| `UD_TAKE_PROFIT_PCT` | `15` | sell this much above entry |
| `UD_STOP_LOSS_PCT` | `30` | or this much below |
| `UD_MAX_HOLD_MIN` | `30` | or this many minutes after kickoff |
| `UD_MIN_PRICE` / `UD_MAX_PRICE` | `0.10` / `0.45` | what counts as an underdog |
| `UD_MAX_SPREAD` | `0.03` | skip wider books |
| `UD_MIN_LIQUIDITY` | `5000` | skip thinner markets |
| `UD_BUY_BEFORE_MIN` / `UD_BUY_UNTIL_MIN` | `3` / `2` | the buying window around kickoff |
| `UD_MAX_OPEN` | `6` | open games at most |
| `UD_DAILY_BUDGET_USD` | `50` | buys per rolling 24 h at most |
| `UD_LOOKAHEAD_HOURS` | `3` | games this far ahead are watched |
| `UD_TICK_SECONDS` | `30` | how often to look |

Change a value, then `plint restart poly`.

## Files

- `polybot/core.py` — Polymarket calls, paper and live orders, resting sells, settlement, fees
- `polybot/consensus.py` + `strategy_consensus.py` — the agreement logic (pure, tested) and its strategy
- `polybot/underdog.py` + `strategy_underdog.py` — the underdog rules (pure, tested) and its strategy
- `polybot/runner.py` — runs both on their own clocks, writes the page's JSON
- `polybot/store.py` — `data/polybot/state.json`, `trades.jsonl`, `status.json`
- `static/index.html` — the page (Plint Soft UI tokens)

Commands: `docker exec polybot python -m polybot once` runs one cycle of both
strategies and prints their reasoning; `… check` shows the account state.
Tests: `docker run --rm -v $PWD:/app -w /app python:3.12-slim sh -c "pip install -q -r requirements.txt && python -m unittest discover -s tests"`.

Uses the official [`polymarket-client`](https://pypi.org/project/polymarket-client/)
SDK; data from Gamma and `data-api.polymarket.com`, orders through the CLOB.
