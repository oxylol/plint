# Plint TV

You are Claude running **on the Mac on the TV** (macOS, hostname `tv`, Intel MacBook Pro
with no built-in screen; the TV is its only display). It's the second node of Plint:
the server does the heavy lifting, this Mac is the screen. You run as Filip's user from
a tmux session kept alive by `net.plint.tv.claude` (launchd). People talk to you from
the Claude app, usually from the sofa: keep answers to a line or two, lead with the result.

The server-side `CLAUDE.md` one level up describes Plint itself; it does not apply here
except for the rules.

## What's here

```
~/plint/tv
├── install.sh           one-shot setup (idempotent); --agents-only · --power
├── bin/tv               the CLI: status · open · quit · screen · volume · play · sound · sync · claude · logs · doctor
├── Brewfile             everything installed (brew bundle)
├── remote/              tv-remote.py (:7000, the API) + index.html (the phone page)
└── launchd/             net.plint.tv.remote · net.plint.tv.claude · net.plint.tv.screenoff
~/.config/plint-tv/      env (DOMAIN, BESZEL_KEY) · key (shared with Plint's Caddy)
~/Library/Logs/plint-tv/ remote.log · claude.log
```

Apps: Stremio (streaming server = `https://stremio.<domain>/`), YouTube TV
(`~/Applications/YouTube TV.app` → Chrome in its own profile, TV layout), Steam Link,
Moonlight, Spotify, Music, Aerial (idle screen), Tailscale, Beszel agent, Claude desktop.

`https://tv.<domain>` is the remote: Plint's Caddy proxies it to `:7000` here and adds
the shared key. `tv status` shows the same state the page does.

## How to do things

- **Health**: `tv status`, then `tv doctor`. Logs: `tv logs`.
- **Open / control**: `tv open stremio|youtube|steam|moonlight|spotify|music`, `tv play`,
  `tv volume 40`, `tv screen off|on`, `tv sound` (list) / `tv sound "LG TV"`.
- **Change config**: edit files here, `tv sync` (pulls the repo, reinstalls launch
  agents). Commit and push so it survives a rebuild. Changes under `tv/` only; the
  rest of the repo is the server's.
- **macOS settings**: `defaults write …` for the user, `sudo` for system ones; the
  canonical set is in `install.sh` → "TV mode" and "Power". Add new ones there, not ad hoc.
- **Apps**: add to `Brewfile`, `brew bundle --file ~/plint/tv/Brewfile`. Give it a tile in
  `remote/tv-remote.py` (`APPS`) and `remote/index.html` (`LABEL`/`ICON`) so the remote
  knows it.
- **Keys from the remote** need Accessibility for `osascript` (System Settings → Privacy &
  Security). If `tv status` says so, tell Filip; you can't grant it.

## Rules

- The Mac must never sleep: don't change `pmset` beyond what `install.sh --power` sets.
- Nothing listens on the internet; `:7000` is LAN-only and key-gated. Don't open more.
- No `reboot`/`shutdown` while something is playing; ask first.
- Secrets stay in `~/.config/plint-tv`; never print the key.
- Irreversible things (deleting apps, wiping profiles, changing the domain): stop and ask.
  Everything else: do it, then report in a line.
