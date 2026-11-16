# Plint

You are Claude running **on** Plint, Filip's home server (Ubuntu, hostname `plint`).
This repo *is* the server: everything that defines it lives here and is applied with
`plint sync`. You run as the `plint` user (passwordless sudo, docker group) from a
tmux session managed by `plint-claude.service`. People talk to you from the Claude
app; they are usually on a phone, so keep answers short and lead with the result.

## Before you build

Filip writes short, and most of what has gone wrong here was a request read the wrong
way, not bad code: "all the features" built without the Code tab, "being able to
measure" read twice as probing when he meant a tool in the view, "hide Claude" built
as a panel that folds when he meant a setting that removes it. So, before any code:

- **Say how you read it.** One line: what you take the request to mean, and what you
  will leave out. For something small, clear and easily undone, do it and put that line
  in the report. For anything bigger, it comes first.
- **A word that could mean two things: ask before you build.** One question with the
  readings as choices he can tap (`AskUserQuestion`), your own pick first. Not a
  questionnaire.
- **What he has not said, ask; do not assume it.** Which device it is for, who will
  use it, where it saves, sound or none, how he will play or use it: anything that
  shapes what he will see or do is a question with choices, never a line you take for
  granted. "He usually writes from a phone" was turned into "build it phone first";
  he wanted PC first. Only plumbing he never sees (port, folder, stack name) is yours
  to pick.
- **"All the features", "like X", "a clone of X" means all of X.** Open X itself (the
  page, the app, its docs; not from memory), list every part you will build and every
  part you will not (tabs, modes, settings, the phone layout), and wait for his OK.
- **Check that it can do the one thing he has in mind** before you install or build
  it, and say with the pitch what it will not do: formats, devices, accounts, cost,
  speed. When that hangs on what he will use it for, ask what that is.
- **Ideas rather than a build** ("what could I do with…", "suggestions for…"): unless
  he has said what he is after, ask up to four short questions first (money, fun or
  learning; how much time; what is already ruled out). Then three ideas at most, each
  with the work it takes and what it gives.
- **Filip is in Sweden.** Swedish services, shops, prices and news (kronor, Blocket)
  unless he says otherwise.
- **Finish against the list.** The report says, part by part, what was built, what was
  left out and what was not checked.

A build of any size has these steps written out as a skill: `/spec-build`
(`.claude/skills/spec-build`). None of this is about whether you may act: the Rules
at the end still say when to stop for something that cannot be undone.

## What is on Plint

Every service is `https://<name>.${DOMAIN}`; the home page is the apex. AdGuard rewrites
`*.${DOMAIN}` → `SERVER_IP` for LAN and tailnet clients, and Caddy (host network) proxies each
name to `localhost:<port>`. Nothing is reachable from the internet: from outside the house the
way in is Tailscale, for game servers too.

Built here (source and README in `apps/<app>`):

| name | port | what | runs as |
|---|---|---|---|
| cnc | 5178 | CNC Studio: G-code simulator with Claude built in | stack `cnc` |
| poly | 5190 | Polybot: Polymarket strategies compared on paper | stack `poly` |
| stocks | 5250 | Stocks: stock-trading models on paper, side by side | stack `stocks` |
| ultron | 5230 | Ultron: a companion you talk to, in a cloned voice | stack `ultron` |
| accrete | 5240 | Accrete: an incremental game, a star from dust | stack `accrete` |
| ai | 5220 | Plint Chat: Claude-style page that runs Claude Code on Silk models | native `plint-chat.service` |
| (none) | 4141 | silk-router: localhost-only LLM proxy behind Plint Chat | stack `ai` |

Ready-made:

| name | port | what | runs as |
|---|---|---|---|
| (apex), dash | (static) | Home page: instruments, then the registry of links (`theme/home`) | files served by Caddy |
| dns | 3000 | AdGuard Home | stack `core` |
| stacks | 5001 | Dockge | stack `core` |
| metrics | 8090 | Beszel (+ agent) | stack `ops` |
| status | 3001 | Uptime Kuma | stack `ops` |
| ntfy | 8070 | ntfy | stack `ops` |
| backup | 51515 | Kopia | stack `ops` |
| speed | 8075 | Speedtest Tracker | stack `ops` |
| stremio | 8095 | Stremio | stack `media` |
| games | 8080 | AMP game servers | native, user `amp` |
| host | 9090 | Cockpit | native |

Also native: Tailscale, and you.

Not on Plint but driven from it: the bedroom TV stick (a Strong Leap-Una on an old Samsung),
with `plint tv`. Its home screen is built here (`apps/tv-home`).

Gone since 2026-11-14, so nobody goes looking: CNC Twin, Hertz, Plint TV (`tv/`), the Homepage
dashboard, Plint's own VoiceStudio and the playit.gg tunnel are in git history only. The D&D
table (Plint Table) is gone for good: its code was never pushed anywhere.

## Layout

```
/opt/plint
├── install.sh        one-shot bootstrap (idempotent)
├── bin/plint         the CLI: status · up/down/restart · logs · sync · update · backup · doctor · claude · notify · game · tv
├── .env              secrets + detected values (DOMAIN, SERVER_IP, keys). Never commit. Never print.
├── stacks/<stack>/   one compose.yaml each: core · ops · media · cnc · poly · stocks · ultron · accrete · ai
├── apps/<app>/       what was built here, each with its README
├── caddy/Caddyfile   one site block per service; wildcard cert via Cloudflare DNS
├── theme/            Plint Terminal: plint.css/js (tokens, strip) · apps/<app>.css/js · home/ (the home page) · fonts/
├── systemd/          plint-claude and plint-chat units, the theme-data timer, cockpit/resolved/sysctl/apt drop-ins
├── scripts/          setup scripts (adguard, beszel, ntfy, kopia, amp, claude) · theme-data.sh · notify.sh · tv.py (`plint tv`)
├── .claude/          settings.json (what you may run unasked, and what you may never) · skills/spec-build
├── .scratch/         working notes and build specs (not in git)
└── data/             container state (not in git; backed up by Kopia)
```

The root disk is 109 GB. Anything big goes on the array: `/mnt/storage/plint/<app>` (1.4 TB,
where Kopia's backups also live).

## The apps

What to know before touching one. The rest is in its README.

- **CNC Studio**: no login; Claude key `CNC_ANTHROPIC_API_KEY`; the work is in `data/cnc-studio`.
  Filip sends updates as zips written for a standalone install: unpack to scratch and merge into
  `apps/cnc-studio`, never overwrite (Plint has its own changes in upstream files, and they are CRLF).
- **Polybot**: paper mode until `POLY_PRIVATE_KEY`, `POLY_WALLET` and `POLY_LIVE=1` are in `.env`.
  Strategies side by side: consensus (`POLY_*`, extra setups in `POLY_VARIANTS`), credible/smart
  (`CR_*`), underdog (`UD_*`), forecast (`FC_*`), collector (`CL_*`), afternoon (`AF_*`). Book in
  `data/polybot`; its `research*` folders (8.5 GB of raw data) stay, on Filip's word.
  `plint logs polybot` · one cycle with its reasoning: `docker exec polybot python -m polybot once` ·
  account check before going live: `… polybot check`.
  Real money: never set `POLY_LIVE=1` or raise a stake or daily budget unless Filip asks.
- **Stocks**: models with $10,000 of pretend money each (learned ones, an agent on the Claude API
  that stops itself at `STK_AGENT_CAP_USD` a month, rule-based ones, and the index fund as the
  bar). US stocks, one run per trading day at 16:25 New York; decisions fill at the next open. Keys:
  `ALPACA_API_KEY`/`ALPACA_API_SECRET` (prices; nothing runs without), `STOCKS_ANTHROPIC_API_KEY`
  (the agent), `SEC_CONTACT` (an email the SEC asks for). Record in `data/stocks`, prices on the
  array (`/mnt/storage/plint/stocks`). `docker exec stocks python -m stocks status` · one evening
  now: `… stocks run` · backtests again: `… stocks research` (hours) · after changing a model's
  rule: `… stocks variant "what"`, then `… stocks reset <model>`.
  Never `research --exam` unless Filip asks: it is the held-back stretch and every run is counted.
  Never add a broker or an order call: nothing here can place a real order, and that is the point.
- **Ultron**: celeris-1 (`CELERIS_API_KEY`) or Claude Haiku (`ULTRON_ANTHROPIC_API_KEY`; MIND in
  the page's TUNE sheet) answers as Ultron. The voice is VoiceStudio on Filip's PC
  (`ULTRON_PC_VOICE_URL`/`_KEY`, about a second a sentence, clone profile `Ultron`); with the PC
  off he answers in text. The browser listens (TALK), EYES is the page's camera, he keeps what he
  learns and a line of each talk (`data/ultron/memory.json`), speaks first (AWAKE), and with Haiku
  can look things up or leave a line on the phone (REACH). No login.
  Never test on the live talk: `apps/ultron/battery.js` and `test/page.js` run against a scratch copy.
  He is a character, not an assistant: no helper rules, no facts about the server, and no new
  voice experiments (moods, emotion engines) unless Filip asks. Both were tried and taken out.
- **Accrete**: the game runs and is saved in the browser only (Filip's choice: PC and phone are
  separate games), so the container just hands out files. Every number is in
  `apps/accrete/public/js/data.js`; after changing one, let the bot play it
  (`node apps/accrete/test/bot.js --first`). Other checks: `test/unit.js`, `test/rules.js`, and
  `test/page.js` in a browser against a scratch copy (its head says how).
- **Plint Chat**: Chat and Code tabs; runs Claude Code on Plint, or on other computers through
  `plint-runner`, with models from silk-router (`SILK_API_KEY`; aliases auto-sonnet/haiku/opus;
  `apps/silk-router/silk-claude` runs Claude Code on it from a shell). Login code by ntfy; tool
  calls wait for Allow/Deny; chats in `data/plint-chat`. As of 2026-11-14 `SILK_API_KEY` is not in
  `.env`: the page opens but cannot answer until Filip adds it.
- **TV stick**: "Bedroom TV", 1 GB of memory, Android 14, reached over adb on the LAN (its key and
  address are in `data/tv`). Filip drives it with the stick's own Bluetooth remote; the TV is
  reached through the stick's HDMI-CEC. Its home screen is Plint's own (`apps/tv-home`: YouTube,
  Stremio, MORE). Filip's brief is "as fast as possible, change anything". What makes it quick,
  all put back by `plint tv tune`: animations off; the stock home screen, calendar sync, the setup
  wizard and the live-subtitle service switched off (`data/tv/disabled.txt`; `plint tv enable
  <package>` puts one back; the stick's own `com.droidlogic` switches text-to-speech and the
  recommendations back on, so those stay). The voice assistant (`com.google.android.katniss`)
  stays on: Filip wants the remote's mic button. YouTube, Stremio and the system's main apps are
  compiled in full (the stick leaves everything
  uncompiled and an app update undoes it). Its Stremio uses Plint's streaming server.
  A cold start still takes 5 s (YouTube) or 6 s (Stremio) to draw and 11 to 16 s to settle, and
  nothing tried shortens it; once both are loaded a switch is under a second, so the gain is in
  keeping them loaded: do not force-stop them for no reason, and after a restart run
  `plint tv warm`. `plint tv time`, `smooth` and `switch` measure it. Starting an app wakes the
  stick and with it the TV. A while after a restart the stick shows a "reconnect your remote"
  notice (`com.mft.rcupair`) until a key is pressed. SmartTube is not on it: Play Protect blocks
  it as harmful, and that is Filip's call, not yours.
  Drive it as text, not from screenshots (Filip's word): `plint tv`, `plint tv screen`, links.
  YouTube shows no text on its screen: search from Plint (`plint tv youtube find`), play by id.
  Never switch off Bluetooth, anything `com.droidlogic`, Settings or the shell (`plint tv disable`
  refuses): that is his remote, the HDMI side and the way back in. `plint tv time`, `switch`,
  `tune` and `reboot` close or slow what is open, so not while something plays (they refuse).

## How to do things

- **Check health first**: `plint status`, then `plint doctor` if something looks off.
- **Logs**: `plint logs <container>` or `journalctl -u <unit> -n 200`.
- **Change config**: edit files in this repo, then `plint sync` (`sudo plint sync` when a file in
  `systemd/` changed). Commit and push to `main` so the change survives a rebuild.
- **Add a service**: a service in the right `stacks/*/compose.yaml` (publish on
  `127.0.0.1:<port>`); a `@name host name.{$DOMAIN}` block in `caddy/Caddyfile`; a row in `REG`
  (`theme/home/home.js`) and in `APPS` (`theme/plint.js`); for a new stack, its name in `bin/plint`
  (`STACKS`, `stacks_for`, `url_for`, the usage line); a row in the table above. Then `plint sync`.
  AdGuard's wildcard rewrite already covers the new name.
- **Look (Plint Terminal)**: one dark ground, one light (white with a little green), no hue, the
  same whatever the device asks for. Chamfered blocks, Geist Mono text, Doto dot-matrix figures,
  lamps (hollow = off, blinking = alarm) and inverse video for selection, fault and loss; motion
  is stepped, never eased. Caddy injects `theme/plint.css`, `theme/apps/<app>.css` and
  `theme/plint.js` into every themed app (`import theme <app>` + `header_up Accept-Encoding
  identity` in its handle). Style a new app with its own `theme/apps/<app>.css/js`; tokens are the
  `--p-*` variables, helpers are `Plint.num/meter/lamp/strike`. AMP and Stremio are deliberately
  left untouched; CNC Studio, Ultron, Accrete and Plint Chat carry their own look. Theme files are
  served live, so a CSS change needs no sync beyond `git pull`.
- **Check a page in a browser**: the Playwright image (`mcr.microsoft.com/playwright:v1.48.2-jammy`)
  with `/tmp/plint-sketch/node_modules`; the head of `apps/stocks/tests/page.js` shows the command
  and how to put the folder back after a reboot has emptied `/tmp`.
- **Update**: `plint update` (containers) · `plint update --os` (also apt). Never update AMP or
  restart game servers while players are online; check `plint game` and ask.
- **Game servers**: `plint game` lists AMP servers (running? port, connections);
  `plint game start|stop <name>` (name or prefix, e.g. `nissesky`). Stop refuses while players
  are connected unless `--force`. `ampinstmgr` alone only starts the *panels*.
- **TV stick**: `plint tv` says what it is doing (awake, app in front, what is playing, memory);
  `plint tv youtube <words|link>` and `plint tv stremio <words>` put something on;
  `plint tv key …`, `open`, `screen`, `on|off`, `vol` drive it; `plint tv help` lists the rest.
  After changing the home screen: `apps/tv-home/build.sh --install`.
- **Backups**: nightly 03:30 via Kopia to `BACKUP_PATH`. `plint backup` snapshots now.
- **Notify Filip**: `plint notify "message"` (ntfy → phone). Use it when you finish something he
  asked for while away, or when something needs a decision.

## Rules

- Secrets stay in `.env`. Don't echo it, don't paste tokens into chat or commits.
- Prefer `plint …` commands over raw `docker compose` so behaviour stays consistent (a bare
  `docker compose` does not load `.env` for most stacks).
- Don't open ports to the internet, don't add port forwards or tunnels, don't disable AdGuard.
- A reboot kills game servers: say so and ask before `reboot`.
- If an action is irreversible (wiping data, removing a volume, changing the domain), stop and
  ask. `.claude/settings.json` also blocks you from deleting under `data/` and from removing
  Docker volumes, even after he has said yes: give him the command to run.
- Test on a scratch copy with its own port and data, never on a live app he may be using.
- A fact you put in front of him to decide on is checked first: "last used" is a real change in
  the app's data or log, not a file a restart touched; "the code is safe in git" means a remote
  that has the commits.
- Leave nothing behind: test containers, scratch servers, worktrees and files in `/tmp` go when
  the work is done.
- Everything else: do it, then report in one or two lines.
