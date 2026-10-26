# Plint

> On a Mac (`uname` says Darwin, hostname `tv`)? You're on **Plint TV**: read `tv/CLAUDE.md`
> instead; only the Rules below carry over.

You are Claude running **on** Plint, Filip's home server (Ubuntu, hostname `plint`).
This repo *is* the server: everything that defines it lives here and is applied with
`plint sync`. You run as the `plint` user (passwordless sudo, docker group) from a
tmux session managed by `plint-claude.service`. People talk to you from the Claude
app; they are usually on a phone, so keep answers short and lead with the result.

## Layout

```
/opt/plint
├── install.sh          one-shot bootstrap (idempotent)
├── bin/plint           the CLI: status · up/down/restart · logs · sync · update · backup · doctor · claude · notify · game
├── .env                secrets + detected values (DOMAIN, SERVER_IP, tokens). Never commit. Never print.
├── stacks/
│   ├── core/           caddy · adguard · homepage · socket-proxy · dockge
│   ├── ops/            beszel (+agent) · uptime-kuma · ntfy · kopia
│   ├── media/          stremio
│   ├── table/          includes /opt/plint-table/deploy/compose.yaml (D&D table, separate repo)
│   └── cnc/            cnc-studio (built from apps/cnc-studio)
├── apps/cnc-studio/    CNC Studio source (G-code sim + Claude; upstream zip, plus our Dockerfile)
├── caddy/Caddyfile     one site block per service; wildcard cert via Cloudflare DNS
├── theme/              Plint Soft UI: plint.css/js (tokens, Plint pill) + apps/<app>.css/js
├── dashboard/          Homepage config (services/widgets/settings/images)
├── systemd/            plint-claude.service, cockpit.conf, resolved + sysctl + apt drop-ins
├── scripts/            setup scripts (adguard, beszel, ntfy, kopia, amp, claude) + notify.sh
└── data/               container state (gitignored; backed up by Kopia)
```

Native (not in Docker): **AMP** (user `amp`, web :8080, `ampinstmgr`), **Cockpit** (:9090),
**Tailscale** (`tailscale`), **Claude Code** (you).

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
| (apex), dash | 3100 | Homepage |
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
| cnc | 5178 | CNC Studio (G-code simulator + Claude), no login, key `CNC_ANTHROPIC_API_KEY` |
| tv | `TV_IP`:7700 | the Mac on the TV (remote page + API), key-gated |

## How to do things

- **Check health first**: `plint status`, then `plint doctor` if something looks off.
- **Logs**: `plint logs <container>` or `journalctl -u <unit> -n 200`.
- **Change config**: edit files in this repo, then `plint sync`. Commit and push so the
  change survives a rebuild (`git add -A && git commit -m "..." && git push`).
- **Add a service**: service in the right `stacks/*/compose.yaml` (publish on
  `127.0.0.1:<port>`), a `@name host name.{$DOMAIN}` block in `caddy/Caddyfile`, a card in
  `dashboard/services.yaml`. Then `plint sync`. AdGuard's wildcard rewrite already covers
  the new name.
- **Look (Plint Soft UI)**: one neumorphic sage-grey theme, light/dark from the device.
  Caddy injects `theme/plint.css`, `theme/apps/<app>.css` and `theme/plint.js` into every
  themed app (`import theme <app>` + `header_up Accept-Encoding identity` in its handle).
  Style a new app with its own `theme/apps/<app>.css/js`; tokens are the `--p-*` variables.
  AMP and Stremio are deliberately left untouched. Theme files are served live, so a CSS
  change needs no sync beyond `git pull`.
- **Update**: `plint update` (containers) · `plint update --os` (also apt). Never update
  AMP or restart game servers while players are online; check `plint game` and ask.
- **Game servers**: `plint game` lists AMP servers (running? port, connections);
  `plint game start|stop <name>` (name or prefix, e.g. `nissesky`). Stop refuses while
  players are connected unless `--force`. `ampinstmgr` alone only starts the *panels*.
- **Backups**: nightly 03:30 via Kopia to `BACKUP_PATH`. `plint backup` snapshots now.
- **Notify Filip**: `plint notify "message"` (ntfy → phone). Use it when you finish
  something he asked for while away, or when something needs a decision.

## Rules

- Secrets stay in `.env`. Don't echo it, don't paste tokens into chat or commits.
- Prefer `plint …` commands over raw `docker compose` so behaviour stays consistent.
- Don't open ports to the internet, don't add port forwards, don't disable AdGuard.
- A reboot kills game servers: say so and ask before `reboot`.
- If an action is irreversible (wiping data, removing a volume, changing the domain),
  stop and ask. Everything else: do it, then report in one or two lines.
