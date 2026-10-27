# Polybot

A Polymarket bot that watches the **leaderboard** and trades where the top
traders agree. It runs on Plint as the `poly` stack and has a page at
`https://poly.<domain>`.

## What it does

Every few minutes it

1. takes the top *N* traders by profit (day / week / month / all-time board),
2. pulls every fill they made in the last *L* hours,
3. works out each leader's **net flow** per outcome (buys minus sells, in USD) —
   a leader counts as *agreeing* when they net-bought an outcome for at least
   `POLY_MIN_LEADER_USD`, and as *against* when they net-sold it or net-bought the
   other side of the same market,
4. **buys** an outcome when `agree − against ≥ POLY_MIN_AGREE`, with a market
   order of `POLY_STAKE_USD`, if the price is still inside the band and not more
   than `POLY_MAX_ABOVE_LEADERS` above what the leaders paid,
5. **sells** a position it holds when the same number of leaders net-sold it
   (`POLY_FOLLOW_SELLS`), and books resolved markets at 1 or 0.

Each check, each trade and each problem is pushed to the phone through ntfy.

## Paper first

Without a key the bot runs in **paper mode**: the same decisions, no orders,
and a paper P&L on the page. Leave it there for a week or two before going live;
the leaderboard is full of high-frequency sports and market-making bots whose
fills are hard to copy five minutes late.

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
Without a Relayer key, winnings from resolved markets are claimed on
polymarket.com as usual.

## Settings (`.env`, all optional)

| variable | default | meaning |
|---|---|---|
| `POLY_WINDOW` | `week` | leaderboard window: `day`, `week`, `month`, `all` |
| `POLY_CATEGORY` | `overall` | or a Polymarket category such as `politics`, `sports`, `crypto` |
| `POLY_TOP_N` | `25` | how many leaders to watch |
| `POLY_MIN_AGREE` | `5` | agree minus against needed to trade |
| `POLY_LOOKBACK_HOURS` | `24` | how far back a leader's fills count |
| `POLY_MIN_LEADER_USD` | `25` | a leader's net flow below this is noise |
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

Change a value, then `plint restart poly`.

## Files

- `polybot/consensus.py` — the agreement logic (pure; `tests/`)
- `polybot/engine.py` — Polymarket calls, trading, paper trading, the page's JSON
- `polybot/store.py` — `data/polybot/state.json`, `trades.jsonl`, `status.json`
- `static/index.html` — the page (Plint Soft UI tokens)

Commands: `docker exec polybot python -m polybot once` runs one cycle and prints
the signals; `… check` shows the account state. Tests:
`docker run --rm -v $PWD:/app -w /app python:3.12-slim sh -c "pip install -q -r requirements.txt && python -m unittest discover -s tests"`.

Uses the official [`polymarket-client`](https://pypi.org/project/polymarket-client/)
SDK; data from `data-api.polymarket.com`, orders through the CLOB.
