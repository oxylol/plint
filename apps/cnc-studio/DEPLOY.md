# Deploying CNC Studio on your own server

CNC Studio is one small Node.js process (`server.js`) that serves the web app and forwards chat requests to the
Claude API. Put it behind your web server for the domain and HTTPS. These steps are for Linux with systemd; the
example files are in `deploy/`.

## 1. Requirements

- Node.js 20 or newer (22 LTS recommended) and npm.
- A reverse proxy for the domain: Caddy or nginx (whichever the server already uses).
- An Anthropic API key, unless every user enters their own key in the app's Settings.

## 2. Install

```bash
sudo useradd --system --home /opt/cnc-studio --shell /usr/sbin/nologin cncstudio
sudo unzip cnc-studio.zip -d /opt            # creates /opt/cnc-studio
cd /opt/cnc-studio && sudo npm ci --omit=dev
sudo chown -R cncstudio:cncstudio /opt/cnc-studio
```

## 3. Configure

Copy `deploy/cnc-studio.env.example` to `/etc/cnc-studio.env`, fill it in, and make it readable by root only:

```bash
sudo cp deploy/cnc-studio.env.example /etc/cnc-studio.env
sudo chmod 600 /etc/cnc-studio.env
```

| Variable | Meaning |
|---|---|
| `PORT` / `HOST` | Where the app listens. Keep `HOST=127.0.0.1` and let the reverse proxy face the network. |
| `ANTHROPIC_API_KEY` | The server's key. Every Claude request from the app is billed to it. |
| `APP_PASSWORD`, `APP_USER` | Password-protects the whole site (browser login, user defaults to `cnc`). **Set it whenever the site is reachable from the internet**, otherwise anyone who finds it can use your API key. |
| `CNCSIM_TOOLS_DIR` | Optional folder with `TurnTools.ini` / `MillTools.millreg` copied from CncSimulator5, offered by *Tools → Import from CncSimulator5*. |

## 4. Run as a service

```bash
sudo cp deploy/cnc-studio.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now cnc-studio
journalctl -u cnc-studio -n 20     # should say "CNC Studio running at http://127.0.0.1:5178 (password protected …)"
```

## 5. Domain and HTTPS

Use `deploy/Caddyfile` (Caddy gets the certificate itself) or `deploy/nginx.conf` (then `certbot --nginx -d your.domain`).
Whatever proxy is used needs three things:

- **no response buffering**: Claude's answer is streamed (the app also sends `X-Accel-Buffering: no`);
- **long timeouts**: one answer can take several minutes (15 min is safe);
- **large request bodies**: drawings and PDFs sent to Claude go in the request (allow 60 MB).

## 6. Check

```bash
cd /opt/cnc-studio && npm run check                       # simulator + API-path tests, no key needed
curl -s -u cnc:YOUR_PASSWORD https://your.domain/api/status    # {"ok":true,...,"serverKey":true}
```

Open `https://your.domain`, log in, load *Examples → Mill job — plate machined on both sides* and ask Claude something.

## Notes

- Programs, jobs, versions, the tool library and the target model are kept **in each user's browser** (local storage),
  per site address. Work made on `http://localhost:5178` does not show up on the domain: save it with *+ → Save job*
  there and open it on the server site. Re-import the tool registers with *Tools → Import register file…*
  (or put them in `CNCSIM_TOOLS_DIR`).
- Updating: replace the files (keep `/etc/cnc-studio.env`), run `npm ci --omit=dev`, `sudo systemctl restart cnc-studio`.
- Windows server: run `node server.js` as a service (for example with NSSM) with the same environment variables,
  and put Caddy for Windows in front of it with the same Caddyfile.
