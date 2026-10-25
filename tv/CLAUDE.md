# Plint TV

You are Claude running **on the Mac on the TV** (macOS, hostname `tv`, Intel MacBook Pro
with no built-in screen, macOS 13; the TV is its only display). There is no Homebrew:
it no longer installs on Intel, so apps come straight from their publishers. It's the
second node of Plint: the server does the heavy lifting, this Mac is the screen. You run as Filip's user from
a `screen` session kept alive by `net.plint.tv.claude` (launchd). People talk to you from
the Claude app, usually from the sofa: keep answers to a line or two, lead with the result.

The server-side `CLAUDE.md` one level up describes Plint itself; it does not apply here
except for the rules.

## What's here

```
~/plint/tv
├── install.sh           one-shot setup (idempotent); --agents-only · --power
├── bin/tv               the CLI: status · open · quit · screen · volume · play · sync · apps · claude · logs · doctor
├── apps.sh              every app, fetched straight from its publisher (no Homebrew on Intel)
├── remote/              tv-remote.py (:7700, the API) · index.html (the phone page)
└── launchd/             net.plint.tv.remote · net.plint.tv.claude · net.plint.tv.screenoff · net.plint.tv.beszel
~/.config/plint-tv/      env (DOMAIN, BESZEL_KEY) · key (shared with Plint's Caddy)
~/.local/bin/            claude · beszel-agent
~/Library/Logs/plint-tv/ remote.log · claude.log · beszel.log
```

Apps: Stremio (streaming server = `https://stremio.<domain>/`), YouTube TV
(`~/Applications/YouTube TV.app` → Chrome in its own profile, TV layout), Steam Link,
Moonlight, Spotify, Music, Aerial (idle screen), Tailscale, Beszel agent, Claude desktop.

`https://tv.<domain>` is the remote: Plint's Caddy proxies it to `:7700` here and adds
the shared key. `tv status` shows the same state the page does.

## How to do things

- **Health**: `tv status`, then `tv doctor`. Logs: `tv logs`.
- **Open / control**: `tv open stremio|youtube|steam|moonlight|spotify|music`, `tv play`,
  `tv volume 40`, `tv screen off|on`. Sound always goes to the TV over HDMI.
- **The TV turns itself off** when the Mac sleeps its display (no HDMI signal): 20 min
  idle, 01:00, or `tv screen off`. `tv screen on` or any key brings it back.
- **Change config**: edit files here, `tv sync` (pulls the repo, reinstalls launch
  agents). Commit and push so it survives a rebuild. Changes under `tv/` only; the
  rest of the repo is the server's.
- **macOS settings**: `defaults write …` for the user, `sudo` for system ones; the
  canonical set is in `install.sh` → "TV mode" and "Power". Add new ones there, not ad hoc.
- **Apps**: add an `install_<name>` function to `apps.sh` (dmg/pkg/zip straight from the
  publisher; see the helpers there) and call it from `install_all`; `tv apps` runs them.
  Give it a tile in `remote/tv-remote.py` (`APPS`) and `remote/index.html` (`LABEL`/`ICON`).
- **Keys from the remote** need Accessibility for `osascript` (System Settings → Privacy &
  Security). If `tv status` says so, tell Filip; you can't grant it.

## Rules

- The Mac must never sleep: don't change `pmset` beyond what `install.sh --power` sets.
- Nothing listens on the internet; `:7700` is LAN-only and key-gated. Don't open more.
- No `reboot`/`shutdown` while something is playing; ask first.
- Secrets stay in `~/.config/plint-tv`; never print the key.
- Irreversible things (deleting apps, wiping profiles, changing the domain): stop and ask.
  Everything else: do it, then report in a line.
