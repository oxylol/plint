# Plint

> On a Mac (`uname` says Darwin, hostname `tv`)? You're on **Plint TV**: read `tv/CLAUDE.md`
> instead; only *Before you build* and the Rules below carry over.

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

## Layout

```
/opt/plint
├── install.sh          one-shot bootstrap (idempotent)
├── .claude/            settings.json (what Claude may run unasked) · skills/spec-build (a request read back as a list before it is built)
├── bin/plint           the CLI: status · up/down/restart · logs · sync · update · backup · doctor · claude · notify · game
├── .env                secrets + detected values (DOMAIN, SERVER_IP, tokens). Never commit. Never print.
├── stacks/
│   ├── core/           caddy · adguard · homepage · socket-proxy · dockge
│   ├── ops/            beszel (+agent) · uptime-kuma · ntfy · kopia
│   ├── media/          stremio
│   ├── table/          includes /opt/plint-table/deploy/compose.yaml (D&D table, separate repo)
│   ├── cnc/            cnc-studio (built from apps/cnc-studio)
│   ├── poly/           polybot (built from apps/polybot)
│   ├── music/          hertz (built from apps/hertz)
│   ├── twin/           cnc-twin (built from apps/cnc-twin)
│   ├── ai/             silk-router (built from apps/silk-router)
│   ├── accrete/        accrete (built from apps/accrete)
│   └── ultron/         ultron (built from apps/ultron); talks to voicestudio in the table stack
├── apps/cnc-studio/    CNC Studio source (G-code sim + Claude; upstream zip, plus our Dockerfile)
├── apps/polybot/       Polybot: Polymarket leaderboard-consensus trader (Python; README there)
├── apps/hertz/         Hertz: music maker made of waves (Web Audio in the browser; README there)
├── apps/cnc-twin/      CNC Twin: physics twin of machine, controller, tool, part, chips, coolant (README there)
├── apps/silk-router/   silk-router: Silk Compute key behind OpenAI/Anthropic APIs, refusal → -uncensored (README there)
├── apps/ultron/        Ultron: voice companion; celeris-1 thinks, VoiceStudio speaks in the cloned voice (README there)
├── apps/accrete/       Accrete: incremental game, a star from dust (all in the browser; README there)
├── apps/plint-chat/    Plint Chat: Claude-style phone page driving Claude Code on Silk models (native unit; README there)
├── caddy/Caddyfile     one site block per service; wildcard cert via Cloudflare DNS
├── theme/              Plint Terminal: plint.css/js (tokens, strip) + apps/<app>.css/js + home/ (the home page) + fonts/
├── dashboard/          Homepage config (services/widgets/settings/images); no longer the home screen
├── systemd/            plint-claude.service, cockpit.conf, resolved + sysctl + apt drop-ins
├── scripts/            setup scripts (adguard, beszel, ntfy, kopia, amp, claude) + notify.sh
└── data/               container state (gitignored; backed up by Kopia)
```

Native (not in Docker): **AMP** (user `amp`, web :8080, `ampinstmgr`), **Cockpit** (:9090),
**Tailscale** (`tailscale`), **Claude Code** (you), **Plint Chat** (`plint-chat.service`).

**Plint Table** (D&D 5e, human DM + voice AI DM) is its own repo at `/opt/plint-table`
(README there). Its data lives in `/srv/dnd/{adventures,library,data}`; `plint sync` pulls that
repo too and rebuilds the `table` image. Secrets for it live in `/opt/plint-table/.env`.

**Plint TV** is the Mac on the TV (`tv/` in this repo, its own `tv/CLAUDE.md`). It has its
own Claude session ("Plint TV"). From here: `plint tv` shows its state, `plint tv register
<ip> <key>` wires `tv.${DOMAIN}` to it (`TV_IP`/`TV_KEY` in `.env`). To change it, edit
`tv/`, push, then ask Plint TV to `tv sync`.

## Names

Every service is `https://<name>.${DOMAIN}`; the home screen is the apex `https://${DOMAIN}`.
AdGuard rewrites `*.${DOMAIN}` → `SERVER_IP` for LAN and tailnet clients. Caddy runs on the
host network and proxies to `localhost:<port>`:

| name | port | what |
|---|---|---|
| (apex), dash | (static) | Home page: `theme/home`, instruments on `data/theme/*.json` (written by `scripts/theme-data.sh`), then the registry of links. Homepage still runs on 3100, unused |
| dns | 3000 | AdGuard Home |
| games | 8080 | AMP |
| stremio | 8095 | Stremio |
| host | 9090 | Cockpit |
| stacks | 5001 | Dockge |
| metrics | 8090 | Beszel |
| status | 3001 | Uptime Kuma |
| ntfy | 8070 | ntfy |
| backup | 51515 | Kopia |
| speed | 8075 | Speedtest Tracker |
| table | 3200 | Plint Table (D&D), own repo `/opt/plint-table` |
| cnc | 5178 | CNC Studio (G-code simulator + Claude), no login, key `CNC_ANTHROPIC_API_KEY`; work in `data/cnc-studio` |
| poly | 5190 | Polybot: Polymarket strategies compared on one page. "Consensus" buys what the top traders agree on (`POLY_*`; `POLY_VARIANTS` runs extra setups such as other leaderboard windows), "credible"/"smart" copy only traders who pass credibility rules (`CR_*`), "underdog" buys the cheaper side of a game at kickoff and sells into the bounce (`UD_*`), "forecast" bets No on tomorrow's high at the forecast +2°C (`FC_*`), "collector" buys what a finished day's METAR reports have already decided and redeems at $1 (`CL_*`), "afternoon" buys Yes on the bucket the day's running high is in at each city's own hour (`AF_*`). Paper mode until `POLY_PRIVATE_KEY`, `POLY_WALLET` and `POLY_LIVE=1` are in `.env` (apps/polybot/README.md); book in `data/polybot` |
| music | 5200 | Hertz: a music maker with no keys. Each voice fires when its pulse wave crosses a threshold; a second sine (drift) bends timing and pitch; tones are FM sines. Sound is made in the browser; saved songs in `data/hertz` |
| twin | 5210 | CNC Twin: runs a CNC program on a physical twin before it goes to the machine (controller in Fanuc/Haas/Siemens 840D/Heidenhain dialects, motion, cutting forces, chips, heat, coolant, wear, vibration, workholding, finish, inspection vs CAD) and says whether it is safe and why. Projects and runs in `data/cnc-twin`; headless CLI `apps/cnc-twin/bin/twin`; one simulation at a time |
| ai | 5220 | Plint Chat: claude.ai-style app (Chat + Code tabs) that runs Claude Code on Plint, or on other computers via `plint-runner`, through silk-router; native `plint-chat.service` as `plint`; login code via ntfy; tool calls wait for Allow/Deny; chats in `data/plint-chat` |
| ultron | 5230 | Ultron: a companion you talk to. celeris-1 (`CELERIS_API_KEY`) answers as Ultron (or Claude Haiku, `ULTRON_ANTHROPIC_API_KEY`, when MIND in the page's TUNE sheet says so), VoiceStudio (container `voicestudio` in the table stack, `COMPOSE_PROFILES=speech`, key `OMNIVOICE_API_KEY`) speaks each sentence with the clone profile `Ultron`; the browser does the listening (TALK keeps listening and, after a while, answers to his name). No login. He can see through the page's camera (EYES), keeps a graph of what he learns and a line or two of each talk (`data/ultron/memory.json`, MEMORY on the page), speaks first when you come in or go quiet (AWAKE), and with Haiku he can reach out (REACH in TUNE): look things up on the web, and on WEB + HOUSE leave a line on the phone (ntfy) or work the TV (`TV_IP`/`TV_KEY`) when asked. Talk, memory and the reference clip in `data/ultron` (test on a scratch copy with `apps/ultron/battery.js`, never on the live talk); VoiceStudio's models on the array (`/mnt/storage/plint/voicestudio`); voice upkeep with `scripts/ultron-voice.sh`. The fast voice is VoiceStudio on Filip's PC (RX 9070 XT, `ULTRON_PC_VOICE_URL`/`_KEY`, about 1 s a sentence), asked first; Plint's own CPU voice (about 20 s a sentence) is the fallback when the PC is off (apps/ultron/README.md) |
| accrete | 5240 | Accrete: an incremental game, a star from dust (bodies orbit a core and feed it; three resets: ignition, collapse, bang; an end). The game runs and is saved in the browser only (Filip's choice: no save on Plint, PC and phone are separate games), so the container just hands out files. Every number is in `apps/accrete/public/js/data.js`; after changing one, let the bot play it (`node apps/accrete/test/bot.js --first`). The `?` in the page lists each hook with its evidence |
| (none) | 4141 | silk-router: localhost-only LLM proxy on `SILK_API_KEY`; aliases auto-sonnet/haiku/opus; `apps/silk-router/silk-claude` runs Claude Code on it |
| tv | `TV_IP`:7700 | the Mac on the TV (remote page + API), key-gated |

## How to do things

- **Check health first**: `plint status`, then `plint doctor` if something looks off.
- **Logs**: `plint logs <container>` or `journalctl -u <unit> -n 200`.
- **Change config**: edit files in this repo, then `plint sync`. Commit and push so the
  change survives a rebuild (`git add -A && git commit -m "..." && git push`).
- **Add a service**: service in the right `stacks/*/compose.yaml` (publish on
  `127.0.0.1:<port>`), a `@name host name.{$DOMAIN}` block in `caddy/Caddyfile`, a row in `REG`
  (`theme/home/home.js`) and in `APPS` (`theme/plint.js`). Then `plint sync`. AdGuard's wildcard rewrite already covers
  the new name.
- **Look (Plint Terminal)**: one dark ground, one light (white with a little green), no hue, the same
  whatever the device asks for. Chamfered blocks, Geist Mono text, Doto dot-matrix figures, lamps (hollow =
  off, blinking = alarm) and inverse video for selection, fault and loss; motion is stepped, never eased.
  Caddy injects `theme/plint.css`, `theme/apps/<app>.css` and `theme/plint.js` into every themed app
  (`import theme <app>` + `header_up Accept-Encoding identity` in its handle). Style a new app with its
  own `theme/apps/<app>.css/js`; tokens are the `--p-*` variables, helpers are `Plint.num/meter/lamp/strike`.
  A new app also goes in `APPS` in `theme/plint.js` and `REG` in `theme/home/home.js`. AMP and Stremio are
  deliberately left untouched. Theme files are served live, so a CSS change needs no sync beyond `git pull`.
- **Update**: `plint update` (containers) · `plint update --os` (also apt). Never update
  AMP or restart game servers while players are online; check `plint game` and ask.
- **Game servers**: `plint game` lists AMP servers (running? port, connections);
  `plint game start|stop <name>` (name or prefix, e.g. `nissesky`). Stop refuses while
  players are connected unless `--force`. `ampinstmgr` alone only starts the *panels*.
- **Backups**: nightly 03:30 via Kopia to `BACKUP_PATH`. `plint backup` snapshots now.
- **Polybot**: `plint logs polybot` · one cycle of both strategies with their reasoning: `docker exec polybot
  python -m polybot once` · account check before going live: `docker exec polybot python -m polybot check`.
  Real money: never set `POLY_LIVE=1` or raise the stakes / daily budgets (`POLY_*`, `UD_*`) unless Filip asks.
- **CNC Twin**: `apps/cnc-twin/bin/twin run <project.yaml>` (headless) · `bin/twin check` · tests `apps/cnc-twin/bin/test`
  (each package in its own process, memory-capped via `bin/tsafe`; `--all` adds regression and validation). Run any
  heavy Python for it through `apps/cnc-twin/bin/tsafe` so a runaway job cannot push the box into swap.
- **Accrete**: no state on the server. Rebuild after edits with `plint sync` (files are baked into the image). Checks:
  `node apps/accrete/test/unit.js`, `test/rules.js`, `test/bot.js`; the browser checks in `test/page.js` run against
  a scratch copy on its own port (its head says how).
- **Notify Filip**: `plint notify "message"` (ntfy → phone). Use it when you finish
  something he asked for while away, or when something needs a decision.

## Rules

- Secrets stay in `.env`. Don't echo it, don't paste tokens into chat or commits.
- Prefer `plint …` commands over raw `docker compose` so behaviour stays consistent.
- Don't open ports to the internet, don't add port forwards, don't disable AdGuard.
- A reboot kills game servers: say so and ask before `reboot`.
- If an action is irreversible (wiping data, removing a volume, changing the domain),
  stop and ask. Everything else: do it, then report in one or two lines.
