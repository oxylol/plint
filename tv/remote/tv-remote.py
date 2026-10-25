#!/usr/bin/env python3
"""
Plint TV · the remote.

A tiny HTTP service on the Mac (port 7000). Plint's Caddy proxies it as
https://tv.<domain> and adds the shared key to every request, so the page you
open on your phone can drive the TV while nobody else on the LAN can.

  GET  /                 the remote (phone-first page)
  GET  /health           {"ok": true}            – no key needed (dashboard dot)
  GET  /api/state        what's on, volume, now playing, which apps run
  POST /api/open         {"app": "stremio|youtube|steam|moonlight|spotify|music"}
  POST /api/quit         {"app": ...}
  POST /api/volume       {"level": 0-100} | {"delta": ±n} | {"mute": true|false}
  POST /api/playpause
  POST /api/key          {"key": "left|right|up|down|enter|back|space|f|k|esc"}
  POST /api/display      {"action": "off|on"}

Everything is stdlib; nothing to install. Actions shell out to macOS tools
(open, osascript, pmset, caffeinate). Display sleep is what turns the TV off.
"""
import json
import os
import re
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOME = Path(os.environ.get("HOME", str(Path.home())))
HERE = Path(__file__).resolve().parent
CONF = HOME / ".config" / "plint-tv"
KEY = (CONF / "key").read_text().strip() if (CONF / "key").exists() else ""
PORT = 7000
STARTED = time.time()

CHROME_UA = ("Mozilla/5.0 (SMART-TV; Linux; Tizen 5.0) AppleWebKit/538.1 "
             "(KHTML, like Gecko) Version/5.0 NativeTVAds Safari/538.1")
YOUTUBE_PROFILE = HOME / "Library" / "Application Support" / "PlintTV" / "youtube"

# name → (how to open, process name for pgrep, label)
APPS = {
    "stremio":   (["open", "-a", "Stremio"], "Stremio", "Stremio"),
    "youtube":   (["open", "-na", "Google Chrome", "--args",
                   "--app=https://www.youtube.com/tv",
                   f"--user-agent={CHROME_UA}",
                   f"--user-data-dir={YOUTUBE_PROFILE}",
                   "--start-fullscreen", "--no-first-run", "--no-default-browser-check"],
                  "Google Chrome", "YouTube"),
    "steam":     (["open", "-a", "Steam Link"], "Steam Link", "Steam Link"),
    "moonlight": (["open", "-a", "Moonlight"], "Moonlight", "Moonlight"),
    "spotify":   (["open", "-a", "Spotify"], "Spotify", "Spotify"),
    "music":     (["open", "-a", "Music"], "Music", "Music"),
}

KEYCODES = {"left": 123, "right": 124, "up": 126, "down": 125, "enter": 36,
            "back": 51, "esc": 53, "space": 49, "f": 3, "k": 40, "tab": 48}


def run(cmd, timeout=8):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()
    except Exception as e:  # noqa: BLE001
        return 1, "", str(e)


def osa(script):
    return run(["osascript", "-e", script])


def running(proc):
    return run(["pgrep", "-x", proc])[0] == 0


def front_app():
    rc, out, _ = run(["lsappinfo", "front"])
    if rc or not out:
        return ""
    rc, info, _ = run(["lsappinfo", "info", "-only", "name", out])
    m = re.search(r'"LSDisplayName"="?([^"\n]+)"?', info)
    return (m.group(1) if m else "").strip()


def volume():
    rc, out, _ = osa("get volume settings")
    lvl = re.search(r"output volume:(\d+)", out)
    mut = re.search(r"output muted:(true|false)", out)
    return (int(lvl.group(1)) if lvl else None, mut.group(1) == "true" if mut else None)


def now_playing():
    for app in ("Spotify", "Music"):
        if not running(app):
            continue
        rc, out, _ = osa(
            f'tell application "{app}"\n'
            'set s to player state as string\n'
            'try\n set t to name of current track\n set a to artist of current track\n'
            'on error\n set t to ""\n set a to ""\nend try\n'
            'return s & "|" & t & "|" & a\nend tell')
        if rc == 0 and out:
            s, t, a = (out.split("|") + ["", "", ""])[:3]
            if s in ("playing", "paused") and t:
                return {"app": app, "state": s, "title": t, "artist": a}
    return None


def state():
    lvl, muted = volume()
    return {
        "host": os.uname().nodename,
        "uptime": int(time.time() - STARTED),
        "front": front_app(),
        "volume": lvl, "muted": muted,
        "playing": now_playing(),
        "apps": {k: running(v[1]) for k, v in APPS.items()},
        "accessibility": run(["osascript", "-e",
                              'tell application "System Events" to return (exists process "Finder")'])[0] == 0,
    }


# ── actions ─────────────────────────────────────────────────
def do_open(app):
    spec = APPS.get(app)
    if not spec:
        return False, f"unknown app {app!r}"
    if app == "youtube":
        YOUTUBE_PROFILE.mkdir(parents=True, exist_ok=True)
    rc, _, err = run(spec[0])
    if rc == 0 and app != "youtube":
        # bring it to the front even if it was already running
        osa(f'tell application "{spec[1]}" to activate')
    return rc == 0, err


def do_quit(app):
    spec = APPS.get(app)
    if not spec:
        return False, f"unknown app {app!r}"
    rc, _, err = run(["pkill", "-x", spec[1]])
    return rc in (0, 1), err


def do_volume(body):
    if "mute" in body:
        osa(f'set volume output muted {"true" if body["mute"] else "false"}')
    if "level" in body:
        osa(f"set volume output volume {max(0, min(100, int(body['level'])))}")
    if "delta" in body:
        lvl, _ = volume()
        if lvl is not None:
            osa(f"set volume output volume {max(0, min(100, lvl + int(body['delta'])))}")
    return True, ""


def do_playpause():
    np_ = now_playing()
    if np_:
        rc, _, err = osa(f'tell application "{np_["app"]}" to playpause')
        return rc == 0, err
    # anything else: space to whatever is in front (needs Accessibility once)
    return do_key({"key": "space"})


def do_key(body):
    k = str(body.get("key", "")).lower()
    if k not in KEYCODES:
        return False, f"unknown key {k!r}"
    rc, _, err = osa(f'tell application "System Events" to key code {KEYCODES[k]}')
    if rc != 0 and "not allowed assistive access" in err:
        err = "macOS needs Accessibility permission for osascript (System Settings → Privacy & Security → Accessibility)"
    return rc == 0, err


def do_display(body):
    a = body.get("action")
    if a == "off":
        rc, _, err = run(["pmset", "displaysleepnow"])
    elif a == "on":
        rc, _, err = run(["caffeinate", "-u", "-t", "2"])
    else:
        return False, "action must be on|off"
    return rc == 0, err




ACTIONS = {
    "/api/open": lambda b: do_open(str(b.get("app", ""))),
    "/api/quit": lambda b: do_quit(str(b.get("app", ""))),
    "/api/volume": do_volume,
    "/api/playpause": lambda b: do_playpause(),
    "/api/key": do_key,
    "/api/display": do_display,
}


# ── http ────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    server_version = "PlintTV/1"

    def log_message(self, fmt, *args):  # quieter log
        if self.path not in ("/health", "/api/state"):
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _authed(self):
        return bool(KEY) and self.headers.get("X-Plint-Key", "") == KEY

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True, "host": os.uname().nodename})
        if self.path in ("/", "/index.html"):
            page = (HERE / "index.html").read_bytes()
            return self._send(200, page, "text/html; charset=utf-8")
        if self.path == "/api/state":
            if not self._authed():
                return self._send(401, {"error": "key"})
            return self._send(200, state())
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._authed():
            return self._send(401, {"error": "key"})
        fn = ACTIONS.get(self.path)
        if not fn:
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        except json.JSONDecodeError:
            return self._send(400, {"error": "bad json"})
        ok, err = fn(body)
        return self._send(200 if ok else 500, {"ok": ok, "error": err or None, "state": state()})


def main():
    if not KEY:
        sys.stderr.write(f"no key at {CONF / 'key'} — run tv/install.sh\n")
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    sys.stderr.write(f"Plint TV remote on :{PORT}\n")
    srv.serve_forever()


if __name__ == "__main__":
    main()
