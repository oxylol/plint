<p align="center">
  <img src="dashboard/images/logo.svg" width="96" alt="">
</p>
<h1 align="center">Plint</h1>
<p align="center">A home server that feels like one thing.<br>
Ad blocking · game servers · streaming · remote access · and Claude living on the box.</p>

---

Every service is a name under one domain, with a real certificate, reachable the same
way at home and away. One command installs it, one command runs it, and the whole
machine is this repository.

| | |
|---|---|
| **plint.net** | Home screen (Homepage) |
| **dns.plint.net** | AdGuard Home — network-wide ad blocking, encrypted upstream DNS |
| **games.plint.net** | AMP — game servers |
| **stremio.plint.net** | Stremio streaming server + web player |
| **host.plint.net** | Cockpit — updates, disks, logs, terminal |
| **stacks.plint.net** | Dockge — containers |
| **metrics.plint.net** | Beszel — CPU, memory, disk, per-container |
| **status.plint.net** | Uptime Kuma — status page |
| **ntfy.plint.net** | ntfy — push notifications to your phone |
| **backup.plint.net** | Kopia — nightly snapshots |
| **speed.plint.net** | Speedtest Tracker — internet speed, four tests a day |
| *Tailscale* | remote access, Tailscale SSH, LAN subnet routing |
| *Claude* | Claude Code on the server, controlled from the Claude app |

Every site except AMP and Stremio wears one theme, Plint Soft UI (`theme/`): neumorphic,
sage-grey, light or dark with your device, numbers drawn as rings, bars and strips.

Nothing is exposed to the internet. The domain's public DNS has no records pointing at the
server; it exists so Caddy can get a wildcard certificate via Cloudflare's API.

## Install

Fresh Ubuntu Server 24.04 or newer, with a static IP (or a DHCP reservation) on your LAN.

**1. Before you wipe the old box**

```bash
# on the old server: keep AMP's data and your licence key
sudo -u amp ampinstmgr backup        # or copy /home/amp/.ampdata somewhere safe
```

**2. On the new box**

```bash
sudo -i
apt install -y git
ssh-keygen -t ed25519 -N "" -C plint -f /root/.ssh/id_ed25519
cat /root/.ssh/id_ed25519.pub
```

Add that public key as a **deploy key** at
<https://github.com/oxylol/plint/settings/keys> (read-only is fine). Then:

```bash
ssh-keyscan -t ed25519 github.com >> /root/.ssh/known_hosts
git clone git@github.com:oxylol/plint.git /opt/plint
/opt/plint/install.sh
```

The installer asks for the domain, your Cloudflare API token, an admin password and a
backup folder, then does everything else. Three steps are interactive because they
have to be: the Tailscale login link, the AMP installer, and signing Claude in.

**3. Three things only you can do**

1. **Router** → DHCP → DNS server = the server's LAN IP. Every device now gets ad
   blocking and resolves `*.plint.net`.
2. **Tailscale admin** → Machines → *plint* → approve the subnet route.
   → DNS → add the server's Tailscale IP as a nameserver, enable *Override DNS servers*.
   Now your phone gets ad blocking and the same names on 5G.
3. Open <https://plint.net>.

**4. First ten minutes in the browser**

- **metrics.plint.net** — create your account, *Add System* → name `plint`,
  host `/beszel_socket/beszel.sock`.
- **status.plint.net** — create your account, add monitors (or let Claude do it),
  add ntfy as the notification channel (`https://ntfy.plint.net`, topic `plint`,
  your admin login).
- **games.plint.net** — finish AMP setup with your licence key, restore the backup.
- **Phone** — install *ntfy*, add server `https://ntfy.plint.net`, log in, subscribe to
  `plint`. Install *Tailscale*, sign in. In *Stremio* settings set the streaming server
  to `https://stremio.plint.net/`.

## Daily

```
plint status          what's running
plint doctor          DNS, certs, ports, tailscale — all green?
plint logs stremio    follow a container
plint update          refresh containers (add --os for apt as well)
plint backup          snapshot now
plint game            AMP game servers · plint game start|stop nissesky
plint claude          attach to Claude on the box (Ctrl+B then D to leave)
```

Or open the Claude app, pick **Plint**, and ask.

## Claude on the box

`plint-claude.service` keeps `claude remote-control` running as the `plint` user inside
tmux. The server appears as a session called **Plint** at claude.ai/code and in the
Claude app. It reads `CLAUDE.md` for the layout and rules, can run every `plint`
command, and asks you on the phone before doing anything it can't undo.

Needs a Pro or Max plan. Sign-in happens once during install (`claude auth login`);
the two one-time prompts (trust the folder, enable Remote Control) are answered with
`plint claude`.

## Change something

Edit the file, `plint sync`, commit, push. That's the whole workflow. A rebuilt box is
`git clone` + `install.sh` + restoring `data/` from Kopia.

## Not included, on purpose

Port forwarding, Portainer, auto-updating containers (Watchtower), and a public
status page. Phase two, if you want them: Pocket ID (passkey single sign-on in front of
the admin tools) and Immich or Jellyfin if a media library appears.
