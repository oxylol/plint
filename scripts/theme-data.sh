#!/usr/bin/env bash
# Plint · data for the Soft UI infographics that apps can't serve themselves.
# Writes small JSON files to data/theme/, which Caddy serves read-only as
# /_plint/data/<file>. Run every 2 minutes by plint-theme-data.timer.
#   games.json          AMP game servers: running, players now (open connections), max
#   games-history.json  players per server over the last 24 h
#   tailscale.json      tailnet devices: name, OS, online, last seen (no addresses)
#   speedtest.json      the last 60 Speedtest Tracker results
#   backups.json        Kopia snapshot times and sizes, no contents (hourly)
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

out="${PLINT_DIR}/data/theme"
install -d -m 0755 "$out"

OUT="$out" PLINT_DIR="$PLINT_DIR" python3 - <<'PY'
import json, os, pwd, re, sqlite3, subprocess, time

OUT = os.environ["OUT"]
PLINT = os.environ["PLINT_DIR"]
now = time.time()


def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return ""


def write(name, data):
    tmp = os.path.join(OUT, "." + name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    os.chmod(tmp, 0o644)
    os.replace(tmp, os.path.join(OUT, name))


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def fresh(name, secs):
    p = os.path.join(OUT, name)
    return os.path.exists(p) and now - os.path.getmtime(p) < secs


# ── AMP game servers ──────────────────────────────────────────
# Same idea as `plint game list`: a game server is the child of its instance's
# AMP process; players are established non-loopback connections to its ports.
def games():
    try:
        home = pwd.getpwnam("amp").pw_dir
    except KeyError:
        return
    try:
        insts = json.load(open(os.path.join(home, ".ampdata", "instances.json")))
    except Exception:
        return
    procs = {}
    for line in sh("ps", "-u", "amp", "-o", "pid=,args=").splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[1].endswith("/AMP_Linux_x86_64"):
            procs[parts[1].split()[0]] = parts[0]
    listen = sh("ss", "-Hltnp")
    est = [l.split() for l in sh("ss", "-Htn", "state", "established").splitlines()]
    servers = []
    for i in insts:
        if i.get("Module") == "ADS":
            continue
        name = i.get("InstanceName", "")
        base = os.path.join(home, ".ampdata", "instances", name)
        amp_pid = procs.get(os.path.join(base, "AMP_Linux_x86_64"))
        child = sh("pgrep", "-P", amp_pid).split()[:1] if amp_pid else []
        ports, players = [], 0
        if child:
            for l in listen.splitlines():
                if "pid=%s," % child[0] in l:
                    ports.append(l.split()[3].rsplit(":", 1)[-1])
            ports = sorted(set(ports), key=int)
            for f in est:
                if len(f) >= 4 and f[2].rsplit(":", 1)[-1] in ports and not re.match(r"^(127\.|\[::1\]|\[::ffff:127\.)", f[3]):
                    players += 1
        maxp = None
        try:
            for l in open(os.path.join(base, i.get("Module", ""), "server.properties")):
                if l.startswith("max-players="):
                    maxp = int(l.split("=", 1)[1])
        except Exception:
            pass
        servers.append({"id": name, "name": i.get("FriendlyName") or name, "module": i.get("Module", ""),
                        "running": bool(child), "players": players, "max": maxp})
    write("games.json", {"generated": stamp(), "servers": servers})

    # players per server over 24 h (-1 = server stopped)
    hist_path = os.path.join(OUT, "games-history.json")
    try:
        hist = json.load(open(hist_path)).get("samples", [])
    except Exception:
        hist = []
    hist.append([int(now), {s["id"]: (s["players"] if s["running"] else -1) for s in servers}])
    hist = [h for h in hist if h[0] >= now - 86400]
    write("games-history.json", {"generated": stamp(), "samples": hist})


# ── Tailscale ─────────────────────────────────────────────────
def tailscale():
    raw = sh("tailscale", "status", "--json")
    if not raw:
        return
    try:
        d = json.loads(raw)
    except ValueError:
        return

    def dev(p, me=False):
        name = p.get("HostName") or ""
        if not name or name == "localhost":
            name = (p.get("DNSName") or "").split(".")[0] or name or "device"
        seen = p.get("LastSeen") or ""
        return {"name": name, "os": p.get("OS", ""), "online": bool(me or p.get("Online")),
                "self": me, "lastSeen": "" if seen.startswith("0001") else seen}

    devices = [dev(d.get("Self", {}), True)] + [dev(p) for p in (d.get("Peer") or {}).values()]
    devices.sort(key=lambda x: (not x["self"], not x["online"], x["name"].lower()))
    write("tailscale.json", {"generated": stamp(), "devices": devices})


# ── Speedtest Tracker ─────────────────────────────────────────
def speedtest():
    db = os.path.join(PLINT, "data", "speedtest", "database.sqlite")
    if not os.path.exists(db):
        return
    try:
        c = sqlite3.connect("file:%s?mode=ro" % db, uri=True, timeout=5)
        rows = c.execute("select created_at, status, ping, download, upload from results "
                         "order by id desc limit 60").fetchall()
    except sqlite3.Error:
        return
    res = [{"at": r[0].replace(" ", "T") + "Z", "ok": r[1] == "completed", "ping": r[2],
            # bytes/s in the database → Mbit/s
            "down": round((r[3] or 0) * 8 / 1e6, 1), "up": round((r[4] or 0) * 8 / 1e6, 1)} for r in rows]
    write("speedtest.json", {"generated": stamp(), "results": list(reversed(res))})


# ── Kopia (hourly is plenty) ──────────────────────────────────
def backups():
    if fresh("backups.json", 55 * 60):
        return
    raw = sh("docker", "exec", "kopia", "kopia", "snapshot", "list", "--json", "--all")
    try:
        snaps = json.loads(raw)
    except ValueError:
        return
    rows = []
    for s in snaps:
        st, start = s.get("stats", {}), s.get("startTime", "")
        try:
            t = time.mktime(time.strptime(start[:19], "%Y-%m-%dT%H:%M:%S"))
        except ValueError:
            continue
        if t < now - 60 * 86400:
            continue
        rows.append({"path": s.get("source", {}).get("path", ""), "start": start, "end": s.get("endTime", ""),
                     "size": st.get("totalSize", 0), "files": st.get("fileCount", 0), "errors": st.get("errorCount", 0)})
    rows.sort(key=lambda r: r["start"])
    write("backups.json", {"generated": stamp(), "snapshots": rows})


for job in (games, tailscale, speedtest, backups):
    try:
        job()
    except Exception as e:  # one broken source must not stop the others
        print("theme-data: %s failed: %s" % (job.__name__, e))
PY
