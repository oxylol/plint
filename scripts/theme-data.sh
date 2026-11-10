#!/usr/bin/env bash
# Plint · data for the Soft UI infographics that apps can't serve themselves.
# Writes small JSON files to data/theme/, which Caddy serves read-only as
# /_plint/data/<file>. Run every 2 minutes by plint-theme-data.timer.
#   games.json          AMP game servers: running, players now (open connections), max
#   games-history.json  players per server over the last 24 h
#   tailscale.json      tailnet devices: name, OS, online, last seen (no addresses)
#   speedtest.json      the last 60 Speedtest Tracker results
#   backups.json        Kopia snapshot times and sizes, no contents (hourly)
#   system.json         uptime, load, CPU per core (2 min mean, 1 h peak and a 10 s recording at 1 s),
#                       memory by tenant (containers, Claude, AMP), disks, containers and stacks
#   system-history.json CPU per core and memory used over the last 24 h
#   dns.json            AdGuard: queries and blocked per hour (counted from the query log, 48 h)
#                       and per day (30 d); no names, no clients
#   uptime.json         Uptime Kuma: every monitor, up now, and the share of each of the last 30 days it was down
#   storage.json        what fills each disk, by folder (du, every 6 h)
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

out="${PLINT_DIR}/data/theme"
install -d -m 0755 "$out"
[[ -f "$PLINT_ENV" ]] && load_env

OUT="$out" PLINT_DIR="$PLINT_DIR" AG_USER="${ADMIN_USER:-}" AG_PASS="${ADMIN_PASSWORD:-}" BACKUP_PATH="${BACKUP_PATH:-}" python3 - <<'PY'
import base64, json, os, pwd, re, sqlite3, subprocess, time, urllib.request

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


def load(name, default):
    try:
        return json.load(open(os.path.join(OUT, name)))
    except Exception:
        return default


# counters carried from one run to the next (CPU ticks, query log position); kept outside the served folder
STATE_FILE = os.path.join(PLINT, "data", ".theme-state.json")
try:
    STATE = json.load(open(STATE_FILE))
except Exception:
    STATE = {}


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

    def ping(p):   # one ping: how far away is it, and is the path direct or through a relay
        ip = (p.get("TailscaleIPs") or [""])[0]
        m = re.search(r"\bin ([\d.]+)(ms|s)\b", sh("tailscale", "ping", "--c", "1", "--timeout", "3s", ip)) if ip else None
        if not m:
            return {}
        out = m.string[:m.start()]
        return {"ms": round(float(m.group(1)) * (1000 if m.group(2) == "s" else 1), 1), "direct": "DERP" not in out}

    devices = [dev(d.get("Self", {}), True)]
    for p in (d.get("Peer") or {}).values():
        devices.append(dict(dev(p), **(ping(p) if p.get("Online") else {})))
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


# ── The host: CPU, memory, disks ──────────────────────────────
def cpu_ticks():   # per core: (all ticks, idle ticks)
    rows = {}
    for l in open("/proc/stat"):
        f = l.split()
        if f[0].startswith("cpu") and f[0][3:].isdigit():
            v = [int(x) for x in f[1:9]]
            rows[int(f[0][3:])] = [sum(v), v[3] + v[4]]
    return [rows[k] for k in sorted(rows)]


def busy(a, b):
    return [max(0, min(100, round(100 * (1 - (y[1] - x[1]) / max(1, y[0] - x[0]))))) for x, y in zip(a, b)]


def mib(s):   # "176.8MiB" → MB
    m = re.match(r"([\d.]+)\s*([KMGT]?)i?B", s.strip())
    return float(m.group(1)) * {"": 1 / 1048576, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1048576}[m.group(2)] if m else 0


def system():
    # ten seconds recorded at one-second steps, before this script does anything heavy itself
    first = cpu_ticks()
    frames, a = [], first
    for _ in range(10):
        time.sleep(1)
        b = cpu_ticks()
        frames.append(busy(a, b))
        a = b
    prev = STATE.get("cpu")
    if prev and len(prev) == len(first) and now - STATE.get("cpu_t", 0) < 900:
        cores = busy(prev, first)                       # the mean since the last run
    else:
        cores = [round(sum(f[i] for f in frames) / len(frames)) for i in range(len(first))]
    STATE["cpu"], STATE["cpu_t"] = a, int(time.time())

    mem = {}
    for l in open("/proc/meminfo"):
        k, v = l.split(":")
        mem[k] = int(v.split()[0]) // 1024
    used = mem["MemTotal"] - mem["MemAvailable"]

    hist = load("system-history.json", {}).get("samples", [])
    hist.append([int(now), cores, used])
    hist = [h for h in hist if h[0] >= now - 86400]
    write("system-history.json", {"generated": stamp(), "samples": hist})
    hour = [h[1] for h in hist if h[0] >= now - 3600 and len(h[1]) == len(cores)] + frames
    peak = [max(h[i] for h in hour) for i in range(len(cores))]

    # who holds the memory: every container, then what runs on the host itself
    tenants = []
    for l in sh("docker", "stats", "--no-stream", "--format", "{{.Name}}|{{.MemUsage}}").splitlines():
        name, _, use = l.partition("|")
        tenants.append({"name": name, "mb": round(mib(use.split("/")[0])), "kind": "container"})
    host = {}
    try:
        amp_uid = pwd.getpwnam("amp").pw_uid
    except KeyError:
        amp_uid = -1
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            if "docker" in open("/proc/%s/cgroup" % pid).read():
                continue
            who = "AMP" if os.stat("/proc/" + pid).st_uid == amp_uid else \
                "Claude" if b"claude" in open("/proc/%s/cmdline" % pid, "rb").read() else None
            if who:
                for l in open("/proc/%s/smaps_rollup" % pid):
                    if l.startswith("Pss:"):
                        host[who] = host.get(who, 0) + int(l.split()[1]) / 1024
        except Exception:
            continue
    tenants += [{"name": k, "mb": round(v), "kind": "host"} for k, v in host.items()]
    tenants.sort(key=lambda t: -t["mb"])

    disks = []
    for name, mount in (("SSD", "/"), ("ARRAY", storage_mount())):
        if mount and os.path.ismount(mount):
            st = os.statvfs(mount)
            disks.append({"name": name, "total_gb": round(st.f_blocks * st.f_frsize / 1e9, 1),
                          "used_gb": round((st.f_blocks - st.f_bfree) * st.f_frsize / 1e9, 1)})

    cont = [l.split("|") for l in sh("docker", "ps", "-a", "--format", '{{.State}}|{{.Label "com.docker.compose.project"}}').splitlines()]
    write("system.json", {
        "generated": stamp(), "uptime_s": int(float(open("/proc/uptime").read().split()[0])),
        "load": [float(x) for x in open("/proc/loadavg").read().split()[:3]],
        "cores": cores, "peak": peak, "loop": {"at": stamp(), "frames": frames},
        "mem": {"total_mb": mem["MemTotal"], "used_mb": used, "tenants": tenants}, "disks": disks,
        "containers": {"running": sum(c[0] == "running" for c in cont), "total": len(cont)},
        "stacks": len({c[1] for c in cont if len(c) > 1 and c[1]})})


def storage_mount():   # the disk the backups live on
    p = os.environ.get("BACKUP_PATH") or ""
    while p and p != "/" and not os.path.ismount(p):
        p = os.path.dirname(p)
    return p if p and p != "/" else ""


# ── What fills the disks (du is slow: every 6 h) ──────────────
def storage():
    if fresh("storage.json", 6 * 3600 - 120):
        return

    def du(path):
        try:
            return int(subprocess.run(["du", "-sxk", path], capture_output=True, text=True, timeout=900).stdout.split()[0]) / 1e6
        except Exception:
            return 0

    def disk(name, mount, parts):
        st = os.statvfs(mount)
        total, used = st.f_blocks * st.f_frsize / 1e9, (st.f_blocks - st.f_bfree) * st.f_frsize / 1e9
        layers = [{"name": n, "gb": round(du(p), 1)} for n, p in parts if os.path.exists(p)]
        rest = used - sum(l["gb"] for l in layers)
        if rest > 0.5:
            layers.append({"name": "SYSTEM" if mount == "/" else "OTHER", "gb": round(rest, 1)})
        return {"name": name, "total_gb": round(total, 1), "used_gb": round(used, 1),
                "layers": sorted((l for l in layers if l["gb"] >= 0.1), key=lambda l: -l["gb"])}

    disks = [disk("SSD", "/", [("DOCKER", "/var/lib/docker"), ("APP DATA", PLINT + "/data"), ("HOME", "/home"), ("TMP", "/tmp")])]
    mount = storage_mount()
    if mount:
        names = {"plint-backups": "BACKUPS", ".ampdata": "GAMES"}
        parts = [(names.get(e, e.lstrip(".").upper()), os.path.join(mount, e)) for e in sorted(os.listdir(mount)) if e != "lost+found"]
        disks.append(disk("ARRAY", mount, parts))
    write("storage.json", {"generated": stamp(), "disks": disks})


# ── AdGuard ───────────────────────────────────────────────────
# Per day comes from its API. Per hour it does not keep, so that is counted from the query
# log: only the lines written since the last run are read (the first run finds where
# 26 hours ago starts). Only the hour and whether it was blocked are taken from a line.
def dns():
    out = load("dns.json", {})
    hours = out.get("hours", {})
    path = os.path.join(PLINT, "data", "adguard", "work", "data", "querylog.json")
    try:
        st = os.stat(path)
        pos = STATE.get("ql") or {}
        with open(path, "rb") as f:
            if pos.get("ino") == st.st_ino and 0 <= pos.get("off", -1) <= st.st_size:
                off = pos["off"]
            elif st.st_size < 64 << 20:
                off = 0                                  # a new log after a rotation
            else:
                cut = time.strftime("%Y-%m-%dT%H", time.localtime(now - 26 * 3600)).encode()
                lo, hi = 0, st.st_size
                while hi - lo > 1 << 20:
                    mid = (lo + hi) // 2
                    f.seek(mid)
                    f.readline()
                    l = f.readline()
                    i = l.find(b'"T":"')
                    if i >= 0 and l[i + 5:i + 18] < cut:
                        lo = mid
                    else:
                        hi = mid
                f.seek(lo)
                f.readline()
                off = f.tell()
            f.seek(off)
            for l in f:
                if not l.endswith(b"\n"):
                    break                                # half-written last line: next time
                off += len(l)
                i = l.find(b'"T":"')
                if i < 0:
                    continue
                h = hours.setdefault(l[i + 5:i + 18].decode(), [0, 0])
                h[0] += 1
                if b'"IsFiltered":true' in l:
                    h[1] += 1
        STATE["ql"] = {"ino": st.st_ino, "off": off}
    except OSError:
        pass
    hours = {k: hours[k] for k in sorted(hours)[-48:]}

    days = out.get("days", {})
    if os.environ.get("AG_USER"):
        try:
            req = urllib.request.Request("http://127.0.0.1:3000/control/stats")
            req.add_header("Authorization", "Basic " + base64.b64encode(
                (os.environ["AG_USER"] + ":" + os.environ.get("AG_PASS", "")).encode()).decode())
            s = json.load(urllib.request.urlopen(req, timeout=10))
            if s.get("time_units") == "days":
                days = {"queries": s.get("dns_queries", []), "blocked": s.get("blocked_filtering", []),
                        "avg_ms": round(s.get("avg_processing_time", 0) * 1000, 1)}
        except Exception:
            pass
    write("dns.json", {"generated": stamp(), "hours": hours, "days": days})


# ── Uptime Kuma ───────────────────────────────────────────────
def uptime():
    db = os.path.join(PLINT, "data", "uptime-kuma", "kuma.db")
    if not os.path.exists(db):
        return
    try:
        c = sqlite3.connect("file:%s?mode=ro" % db, uri=True, timeout=5)
        mons = c.execute("select id, name, type from monitor where active = 1 order by id").fetchall()
        today = int(now) // 86400 * 86400            # Kuma's days are UTC days
        days = [today - 86400 * i for i in range(29, -1, -1)]
        stat = {(r[0], r[1]): (r[2], r[3]) for r in c.execute(
            "select monitor_id, timestamp, up, down from stat_daily where timestamp >= ?", (days[0],))}
        last = {r[0]: r for r in c.execute("select monitor_id, status, time, ping from heartbeat "
                                           "where id in (select max(id) from heartbeat group by monitor_id)")}
    except sqlite3.Error:
        return
    rows = []
    for mid, name, kind in mons:
        d = []
        for t in days:
            u = stat.get((mid, t))
            d.append(None if not u or not (u[0] + u[1]) else round(u[1] / (u[0] + u[1]), 4))
        hb = last.get(mid)
        rows.append({"name": name, "type": kind, "up": bool(hb and hb[1] == 1), "ping": hb[3] if hb else None, "down": d})
    write("uptime.json", {"generated": stamp(), "days": days, "monitors": rows})


for job in (system, games, tailscale, speedtest, backups, dns, uptime, storage):
    try:
        job()
    except Exception as e:  # one broken source must not stop the others
        print("theme-data: %s failed: %s" % (job.__name__, e))
try:
    with open(STATE_FILE + ".tmp", "w") as f:
        json.dump(STATE, f)
    os.replace(STATE_FILE + ".tmp", STATE_FILE)
except Exception as e:
    print("theme-data: state not saved: %s" % e)
PY
