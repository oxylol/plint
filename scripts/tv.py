#!/usr/bin/env python3
"""plint tv · the TV stick, driven as text (no screenshots)

  plint tv                    what the stick is doing: awake, app in front, what is playing, memory
  plint tv find [ip]          look for the stick on the LAN and remember where it is
  plint tv connect            open the debugging link (the TV asks once: Always allow)
  plint tv on | off           wake or sleep the stick; the TV follows over HDMI-CEC
  plint tv key [-g sec] <k>   up down left right ok back home menu play pause playpause stop next prev
                              rew ff · several at once: down*3 ok · ok! holds · -g waits between keys
  plint tv vol up|down|mute [n]
  plint tv open <app>         youtube · stremio · settings · home, or part of a package name
  plint tv youtube <what>     a link or video id plays; words play the top result · find <words> only lists
  plint tv stremio <what>     words open that film or series' page · find <words> only lists · a stremio:// link
  plint tv type "text"        type into the field in focus
  plint tv screen [-v]        the screen as text: one line per label, > marks what is in focus (-v: where)
  plint tv apps [-a]          apps on the stick (-a: system ones too); off = switched off
  plint tv disable|enable <package>   switch an app off or on again (list in data/tv/disabled.txt)
  plint tv install <file.apk>
  plint tv tune               put the stick back in its quick state (animations, apps switched off, apps compiled)
  plint tv warm               load YouTube and Stremio and return home, so a tile is a quick switch (after a restart)
  plint tv time [app...]      seconds to open home and each app from cold: first picture, and calm (stops what is open)
  plint tv smooth <app> [key...]   how fast an app answers the remote: ms to the next frame after each key
  plint tv switch [app...]    go round the apps: is each still loaded on return, and how long the switch takes
  plint tv reboot             restart the stick and say how long it took
  plint tv sh <command>       a shell command on the stick

  PLINT_TV_DRY=1 prints the adb commands instead of running them.
"""
import json, os, re, shlex, socket, subprocess, sys, time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

PLINT_DIR = Path(os.environ.get("PLINT_DIR", "/opt/plint"))
DATA = PLINT_DIR / "data" / "tv"          # the stick's address, adb's key, the list of switched-off apps
STICK = DATA / "stick.json"
OFF_LIST = DATA / "disabled.txt"
ADB_PORT = "5038"                         # our own adb server, so its key is the one in data/tv
DRY = bool(os.environ.get("PLINT_TV_DRY"))

APPS = {"youtube": "com.google.android.youtube.tv", "stremio": "com.stremio.one",
        "settings": "com.android.tv.settings"}
HOME_APP = "se.plint.tvhome"              # Plint's own home screen (apps/tv-home)
KEYS = {"up": 19, "down": 20, "left": 21, "right": 22, "ok": 23, "enter": 23, "back": 4, "home": 3,
        "menu": 82, "play": 126, "pause": 127, "playpause": 85, "stop": 86, "next": 87, "prev": 88,
        "rew": 89, "ff": 90, "volup": 24, "voldown": 25, "mute": 164, "power": 26, "sleep": 223,
        "wake": 224, "search": 84, "info": 165, "del": 67, "tab": 61, "space": 62}
# Losing one of these locks Filip out: the stick's remote is Bluetooth, the TV is reached through
# the system's HDMI-CEC, and adb's shell is the way back in.
KEEP = {"android", "com.android.systemui", "com.android.tv.settings", "com.android.shell",
        "com.android.providers.settings", "com.google.android.gms", "com.android.bluetooth"}
HOW_TO_DEBUG = "Settings → System → Developer options → USB debugging"


def die(msg):
    sys.exit(f"✗ {msg}")


# ── where the stick is ───────────────────────────────────────
def cast_info(ip, timeout=2.0):
    """Name, maker and id of a cast device, from the description it serves on :8008."""
    try:
        xml = urlopen(f"http://{ip}:8008/ssdp/device-desc.xml", timeout=timeout).read().decode("utf-8", "replace")
    except Exception:
        return None
    tag = lambda t: (re.search(rf"<{t}>(.*?)</{t}>", xml) or [None, ""])[1]
    return {"host": ip, "name": tag("friendlyName"), "maker": tag("manufacturer"), "udn": tag("UDN")}


def lan_hosts():
    out = subprocess.run(["ip", "-4", "route", "get", "1.1.1.1"], capture_output=True, text=True).stdout
    src = re.search(r"src (\d+\.\d+\.\d+)\.\d+", out) or die("no LAN address on this box")
    return [f"{src[1]}.{n}" for n in range(1, 255)]


def port_open(ip, port, timeout=0.5):
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def saved():
    if STICK.exists():
        return json.loads(STICK.read_text())
    if os.environ.get("TV_HOST"):
        return {"host": os.environ["TV_HOST"]}
    return None


def stick():
    return saved() or cmd_find([])


def cmd_find(args):
    if args:
        found = [cast_info(args[0]) or die(f"nothing like a TV stick answers at {args[0]}")]
    else:
        with ThreadPoolExecutor(64) as pool:
            hosts = lan_hosts()
            up = [ip for ip, yes in zip(hosts, pool.map(lambda ip: port_open(ip, 8008), hosts)) if yes]
            found = [d for d in pool.map(cast_info, up) if d]
    known = (saved() or {}).get("udn")   # Strong's sticks are made by Skyworth
    pick = found if args else ([d for d in found if d["udn"] == known]
                               or [d for d in found if "skyworth" in d["maker"].lower()])
    if len(pick) != 1:
        for d in found:
            print(f"  {d['host']:<16}{d['name']} ({d['maker']})")
        die("the stick is not on the network (off?)" if not found else "which one is it: plint tv find <ip>")
    d = pick[0]
    DATA.mkdir(parents=True, exist_ok=True)
    STICK.write_text(json.dumps(d, indent=1) + "\n")
    print(f"✓ {d['name']} ({d['maker']}) at {d['host']}")
    return d


# ── adb ──────────────────────────────────────────────────────
def adb(*args, timeout=40):
    if DRY:
        print("adb", *args)
        return subprocess.CompletedProcess(args, 0, "", "")
    DATA.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HOME=str(DATA), ANDROID_ADB_SERVER_PORT=ADB_PORT)
    try:
        return subprocess.run(["adb", *args], capture_output=True, text=True, errors="replace", timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        die(f"the stick did not answer within {timeout} s")


def serial():
    return f"{stick()['host']}:5555"


def connect():
    host = stick()["host"]
    if not DRY and not port_open(host, 5555, 2):
        if not port_open(host, 8008, 2):
            die(f"the stick is not answering at {host} (off, or its address changed: plint tv find)")
        die(f"debugging is off on the stick: {HOW_TO_DEBUG}")
    r = adb("connect", serial())
    if DRY:
        return
    asked = False
    for _ in range(30):                           # the TV asks once; give him a minute to press OK
        if adb("-s", serial(), "get-state").stdout.strip() == "device":
            return
        if "unauthorized" not in adb("devices").stdout and "authenticate" not in r.stdout:
            die(f"could not connect: {(r.stdout + r.stderr).strip()}")
        if not asked:
            print("the TV is asking whether to allow debugging: tick Always allow and press OK", flush=True)
            asked = True
        time.sleep(2)
        r = adb("connect", serial())
    die("nobody pressed OK on the TV: run this again when you are in front of it")


def sh(cmd, timeout=40):
    """Run a shell command on the stick; connect first if the link is down."""
    r = adb("-s", serial(), "shell", cmd, timeout=timeout)
    if r.returncode and re.match(r"(adb|error):", r.stderr):
        connect()
        r = adb("-s", serial(), "shell", cmd, timeout=timeout)
        if r.returncode and re.match(r"(adb|error):", r.stderr):
            die(r.stderr.strip())
    return r.stdout or r.stderr


# ── what it is doing ─────────────────────────────────────────
PLAYBACK = {"0": "idle", "1": "stopped", "2": "paused", "3": "playing", "6": "loading", "7": "error"}


def sessions(dump):
    """The media sessions in `dumpsys media_session`: package, state, title."""
    found, cur = [], None
    for line in dump.splitlines():
        line = line.strip()
        if line.startswith("package="):
            cur = {"package": line[8:], "state": "", "title": ""}
            found.append(cur)
        elif cur and line.startswith("state=PlaybackState"):
            m = re.search(r"state=(?:[A-Z_]+\()?(\d+)", line[6:])
            cur["state"] = PLAYBACK.get(m[1], m[1]) if m else ""
        elif cur and line.startswith("metadata:"):
            m = re.search(r"description=(.*)", line)
            cur["title"] = re.sub(r"^null$|(, null)+$", "", m[1]) if m else ""
    order = {"playing": 0, "loading": 1, "paused": 2}   # the rest (idle, stopped, error) is nothing on screen
    return sorted((s for s in found if s["state"] in order), key=lambda s: order[s["state"]])


def front():
    out = sh("dumpsys activity activities | grep -m1 -E 'topResumedActivity|mResumedActivity'")
    m = re.search(r" u\d+ ([\w.]+)/([\w.$]+)", out)
    return (m[1], m[2]) if m else ("", "")


def name_of(package):
    return "home screen" if package == HOME_APP else next((k for k, v in APPS.items() if v == package), package)


def cmd_status(args):
    s = stick()
    if not DRY and not port_open(s["host"], 5555, 2):   # no way in: the cast side can still say it is there
        try:
            info = json.load(urlopen(f"http://{s['host']}:8008/setup/eureka_info", timeout=3))
        except Exception:
            die(f"the stick is not answering at {s['host']} (off, or its address changed: plint tv find)")
        print(f"{info.get('name', 'TV stick')} · {s['host']}")
        print(f"debugging  off: {HOW_TO_DEBUG}")
        return
    out = sh("dumpsys power | grep -m1 mWakefulness=; echo @@;"
             "dumpsys activity activities | grep -m1 -E 'topResumedActivity|mResumedActivity'; echo @@;"
             "dumpsys media_session; echo @@; cat /proc/meminfo; echo @@; cat /proc/uptime")
    if DRY:
        return
    power, top, media, mem, up = (out.split("@@") + [""] * 5)[:5]
    print(f"{s.get('name', 'TV stick')} · {s['host']} · on for {int(float((up.split() or ['0'])[0])) // 60} min")
    m = re.search(r" u\d+ ([\w.]+)/([\w.$]+)", top)
    print(f"screen     {(re.search(r'mWakefulness=(\w+)', power) or [0, '?'])[1].lower()}")
    print(f"in front   {name_of(m[1]) if m else '?'}" + (f"  ({m[1]})" if m and name_of(m[1]) != m[1] else ""))
    now = sessions(media)
    print("playing    " + (f"{now[0]['title'] or '(no title)'} · {now[0]['state']} · {name_of(now[0]['package'])}"
                           if now else "nothing"))
    kb = {k: int(v) for k, v in re.findall(r"(MemTotal|MemAvailable):\s+(\d+)", mem)}
    if kb:
        print(f"memory     {kb.get('MemAvailable', 0) // 1024} MB free of {kb.get('MemTotal', 0) // 1024} MB")


def cmd_screen(args):
    raw = sh("uiautomator dump /data/local/tmp/ui.xml 2>&1; cat /data/local/tmp/ui.xml; rm -f /data/local/tmp/ui.xml",
             timeout=60)
    i, j = raw.find("<?xml"), raw.rfind("</hierarchy>")
    if i < 0 or j < 0:
        die(f"{name_of(front()[0])} does not give its screen as text ({raw.strip().splitlines()[-1] if raw.strip() else 'no answer'}); "
            "plint tv says what is in front and what is playing")
    rows, spot = [], []                           # [y, x, in focus, label, right, bottom] · the box in focus

    def box(node):
        return [int(n) for n in (re.findall(r"\d+", node.get("bounds", "")) + ["0"] * 4)[:4]]

    def walk(node, focus):
        if node.get("focused") == "true":
            focus = True
            spot[:] = box(node)
        label = (node.get("text") or node.get("content-desc") or "").strip()
        if label:
            x, y, r, b = box(node)
            rows.append([y, x, focus, " ".join(label.split()), r, b])
        for child in node:
            walk(child, focus)

    root = ET.fromstring(raw[i:j + 12])
    walk(root, False)
    if spot and not any(r[2] for r in rows):      # the box in focus holds no label of its own: take what lies in it
        for r in rows:
            r[2] = spot[0] <= (r[1] + r[4]) / 2 <= spot[2] and spot[1] <= (r[0] + r[5]) / 2 <= spot[3]
        if not any(r[2] for r in rows):
            rows.append([spot[1], spot[0], True, "(something with no name)", spot[2], spot[3]])
    print(name_of(next((n.get("package") for n in root.iter("node") if n.get("package")), "?")))
    if not rows:
        print("  (shows no text: YouTube draws its own screen; plint tv says what is playing)")
    last = None
    for y, x, focus, label, r, b in sorted(rows):
        if label != last:
            print(("> " if focus else "  ") + label + (f"   [{x},{y} {r - x}x{b - y}]" if "-v" in args else ""))
        last = label


# ── driving it ───────────────────────────────────────────────
def keycode(name):
    if name.isdigit() or name.upper().startswith("KEYCODE_"):
        return name.upper()
    return str(KEYS.get(name.lower()) or die(f"no key called {name}: {' '.join(KEYS)}"))


def cmd_key(args):
    gap = 0.0
    if args[:1] == ["-g"]:
        gap, args = float(args[1]), args[2:]
    args or die("plint tv key <key>...")
    presses = []                                  # (code, held)
    for a in args:
        name, _, n = a.partition("*")
        held = name.endswith("!")
        presses += [(keycode(name.rstrip("!")), held)] * int(n or 1)
    while presses:
        if presses[0][1] or gap:
            (code, held), presses = presses[0], presses[1:]
            sh(f"input keyevent {'--longpress ' if held else ''}{code}")
            time.sleep(gap)
        else:                                     # a run of plain presses goes in one call
            n = next((i for i, p in enumerate(presses) if p[1]), len(presses))
            sh("input keyevent " + " ".join(p[0] for p in presses[:n]))
            presses = presses[n:]


def cmd_on(args):
    sh("input keyevent 224; cmd hdmi_control onetouchplay")   # wake, then tell the TV to switch on and over to us


def cmd_off(args):
    sh("input keyevent 223")


def cmd_vol(args):
    key = {"up": "24", "down": "25", "mute": "164"}.get(args[0] if args else "") or die("plint tv vol up|down|mute [n]")
    sh("input keyevent " + " ".join([key] * int(args[1] if len(args) > 1 else 1)))


def packages(flag=""):
    return sorted(p[8:].strip() for p in sh(f"pm list packages {flag}").splitlines() if p.startswith("package:"))


def resolve(app):
    """An app's package from its short name, its full name or a part of it."""
    if app.lower() in APPS:
        return APPS[app.lower()]
    if DRY:
        return app
    have = packages()
    hits = [app] if app in have else [p for p in have if app.lower() in p.lower()]
    if len(hits) != 1:
        die(f"no app like {app} on the stick" if not hits else f"{app} could be: {' '.join(hits)}")
    return hits[0]


def launcher_of(package):
    for cat in ("LEANBACK_LAUNCHER", "LAUNCHER"):
        out = sh(f"cmd package resolve-activity --brief -c android.intent.category.{cat} {package}")
        last = (out.strip().splitlines() or [""])[-1].strip()
        if "/" in last:
            return last
    return package if DRY else die(f"{package} is not on the stick, or has nothing to open")


def cmd_open(args):
    args or die("plint tv open <app>")
    if args[0] == "home":
        return sh("input keyevent 3")
    print(sh(f"am start -n {launcher_of(resolve(args[0]))}").strip())


def view(uri, package):
    print(sh(f"am start -a android.intent.action.VIEW -d {shlex.quote(uri)} -p {package}").strip())


def fetch(url, body=None):
    req = Request(url, json.dumps(body).encode() if body else None,
                  {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
    try:
        return json.load(urlopen(req, timeout=15))
    except Exception as e:
        die(f"no answer from {url.split('/')[2]}: {e}")


def youtube_search(words, n=5):
    """YouTube's own search, asked from Plint: its app shows no text, so the results are read here."""
    data = fetch("https://www.youtube.com/youtubei/v1/search?prettyPrint=false",
                 {"context": {"client": {"clientName": "WEB", "clientVersion": "2.20240101.00.00", "hl": "en", "gl": "SE"}},
                  "query": words})
    found = []

    def walk(o):
        if isinstance(o, dict):
            if "videoRenderer" in o:
                found.append(o["videoRenderer"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(data)
    text = lambda v, k: "".join(r["text"] for r in v.get(k, {}).get("runs", []))
    return [(v["videoId"], v.get("lengthText", {}).get("simpleText", "live"), text(v, "title"), text(v, "ownerText"))
            for v in found[:n]]


def cmd_youtube(args):
    if not args:
        return cmd_open(["youtube"])
    find = args[0] == "find"
    what = " ".join(args[find:]) or die("plint tv youtube find <words>")
    m = re.search(r"(?:v=|youtu\.be/|shorts/|live/)([\w-]{11})", what)
    vid = m[1] if m else what if re.fullmatch(r"[\w-]{11}", what) and not what.islower() else None
    if not vid:
        hits = youtube_search(what) or die(f"YouTube finds nothing for {what}")
        for i, (v, length, title, who) in enumerate(hits):
            print(f"{'>' if i == 0 and not find else ' '} {v}  {length:>8}  {title} · {who}")
        if find:
            return
        vid = hits[0][0]
    view(f"https://www.youtube.com/watch?v={vid}", APPS["youtube"])


def stremio_search(words, n=3):
    """Stremio's catalogue (Cinemeta), asked from Plint: films and series by name."""
    hits = []
    for kind in ("movie", "series"):
        metas = fetch(f"https://v3-cinemeta.strem.io/catalog/{kind}/top/search={quote(words)}.json").get("metas", [])
        hits += [(kind, m["id"], m.get("name", "?"), m.get("releaseInfo") or "") for m in metas[:n]]
    return hits


def cmd_stremio(args):
    if not args:
        return cmd_open(["stremio"])
    if args[0].startswith("stremio:"):
        return view(args[0], APPS["stremio"])
    find = args[0] == "find"
    what = " ".join(args[find:]) or die("plint tv stremio find <words>")
    hits = stremio_search(what) or die(f"Stremio's catalogue has nothing for {what}")
    # the closest name wins; films before series on a tie
    best = min(range(len(hits)), key=lambda i: (what.lower() not in hits[i][2].lower(), abs(len(hits[i][2]) - len(what)), i))
    for i, (kind, id_, name, year) in enumerate(hits):
        print(f"{'>' if i == best and not find else ' '} {kind:<7} {id_:<11} {name} {year}")
    if not find:
        view(f"stremio:///detail/{hits[best][0]}/{hits[best][1]}", APPS["stremio"])


def cmd_type(args):
    args or die('plint tv type "text"')
    sh("input text " + shlex.quote(" ".join(args).replace(" ", "%s")))


# ── apps ─────────────────────────────────────────────────────
def cmd_apps(args):
    off = set(packages("-d"))
    for p in packages() if "-a" in args else sorted(set(packages("-3")) | off):
        print(f"{'off' if p in off else '   '}  {p}")


def home_apps():
    out = sh("cmd package query-activities --brief -a android.intent.action.MAIN -c android.intent.category.HOME")
    return {l.split("/")[0].strip() for l in out.splitlines() if "/" in l}


def cmd_disable(args):
    force = "--force" in args
    args = [a for a in args if a != "--force"] or die("plint tv disable <package>")
    package = args[0]
    if not force:
        if package in KEEP or package.startswith("com.droidlogic"):   # droidlogic is Amlogic's HDMI side
            die(f"{package} stays on: without it the remote or the way back in is lost")
        homes = home_apps()
        if package in homes and not homes - {package, "com.android.tv.settings", "com.android.settings"}:
            die(f"{package} is the only home screen on the stick: put another one on first")
    out = sh(f"pm disable-user --user 0 {shlex.quote(package)}").strip()
    print(out)
    if "disabled" in out:
        with OFF_LIST.open("a") as f:
            f.write(f"{time.strftime('%Y-%m-%d')} {package}\n")


def cmd_enable(args):
    args or die("plint tv enable <package>")
    out = sh(f"pm enable {shlex.quote(args[0])}").strip()
    print(out)
    if "enabled" in out and OFF_LIST.exists():
        OFF_LIST.write_text("".join(l for l in OFF_LIST.read_text().splitlines(True) if l.split()[-1:] != [args[0]]))


# What makes this stick quick, in the order it is put back by `plint tv tune`.
ANIMATIONS = {"window_animation_scale": "0", "transition_animation_scale": "0", "animator_duration_scale": "0.5"}
COMPILED = ["com.google.android.youtube.tv", "com.stremio.one", "com.android.systemui",
            "com.android.tv.settings", "com.google.android.gms", "com.android.vending", "com.google.android.webview"]


def cmd_tune(args):
    """Put the stick back in its quick state: animations off, the apps in data/tv/disabled.txt off, and the
    apps that matter compiled in full (the stick leaves everything uncompiled, and an update undoes it).
    Safe to run again; compiling takes minutes and keeps the stick busy, so not while something plays."""
    if busy() and "--force" not in args:
        die("something is playing: tuning keeps the stick busy for minutes (--force to do it anyway)")
    for key, want in ANIMATIONS.items():
        if sh(f"settings get global {key}").strip() != want:
            sh(f"settings put global {key} {want}")
            print(f"animations  {key} → {want}")
    wanted = [l.split()[-1] for l in OFF_LIST.read_text().splitlines() if l.strip()] if OFF_LIST.exists() else []
    off = set(packages("-d"))
    for package in wanted:
        if package not in off:
            out = sh(f"pm disable-user --user 0 {shlex.quote(package)}").strip()
            print(f"off again   {package}" if "disabled" in out else f"stays on    {package}: {out}")
    status = dict(re.findall(r"\[([\w.]+)\]\s+path: \S+\s+\w+: \[status=([\w-]+)\]", sh("dumpsys package dexopt", timeout=90)))
    have = set(packages())
    for package in COMPILED:
        if package in have and status.get(package) != "speed":
            start = time.time()
            out = sh(f"cmd package compile -m speed -f {package}", timeout=1800).strip()
            print(f"compiled    {package} ({status.get(package, '?')} → speed, {time.time() - start:.0f} s): {out}", flush=True)
    print("✓ tuned")


def cmd_install(args):
    args and Path(args[0]).is_file() or die("plint tv install <file.apk>")
    sh("true")
    env = dict(os.environ, HOME=str(DATA), ANDROID_ADB_SERVER_PORT=ADB_PORT)
    try:
        r = subprocess.run(["adb", "-s", serial(), "install", "-r", args[0]], capture_output=True, text=True, timeout=240, env=env)
    except subprocess.TimeoutExpired:           # Play Protect puts a notice on the TV and the install waits for it
        die(f"not installed after 4 min: the stick is asking something on the TV, read it with plint tv screen ({name_of(front()[0])} is in front)")
    print((r.stdout + r.stderr).strip())


def busy():
    return any(s["state"] == "playing" for s in sessions(sh("dumpsys media_session")))


def cmd_time(args):
    force = "--force" in args
    args = [a for a in args if a != "--force"]
    if busy() and not force:
        die("something is playing: this closes the apps it times, so not now (--force to do it anyway)")
    out = sh("cmd package resolve-activity --brief -a android.intent.action.MAIN -c android.intent.category.HOME")
    home = (out.strip().splitlines() or [""])[-1].strip()
    "/" in home or die(f"the stick names no home screen: {out.strip()}")
    have = set(packages())
    targets = [("home", home)] if not args else []
    for app in args or [a for a in ("youtube", "stremio") if APPS[a] in have]:
        targets.append((app, launcher_of(resolve(app))))
    for label, comp in targets:
        shown, drawn, calm = cold(comp)
        sec = lambda v, word: f"{word} {v:.1f} s" if v else f"no {word}"
        print(f"{label:<10} {sec(shown, 'picture')} · {sec(drawn, 'drawn') + ' · ' if drawn else ''}"
              f"{sec(calm, 'calm') if calm else 'still working after a minute'}", flush=True)
        sh("input keyevent 3")
        time.sleep(8)                             # let the stick settle, or the next one pays for this one


def cold(comp):
    """A cold start, in seconds: until the first picture and until the app says it is fully drawn (the system's
    own lines; `am start -W` gives up after 10 s), and until its process uses less than one core, which is
    when it stops getting in its own way. The loop runs on the stick, so the link does not blur the times."""
    package = comp.split("/")[0]
    out = sh(f"p={package}; f=/data/local/tmp/plint-atm.log; logcat -c; logcat -s ActivityTaskManager:I > $f & lp=$!; "
             "am force-stop $p; sleep 1; read a _ < /proc/uptime; t0=${a/./}; "
             f"am start -n {comp} >/dev/null 2>&1; i=0; last=0; tl=$t0; calm=0; busy=0; "
             "while [ $i -lt 110 ] && [ $calm -lt 4 ]; do "
             "pid=$(pidof $p); pid=${pid%% *}; ticks=$last; "
             "if [ -n \"$pid\" ]; then set -- $(cat /proc/$pid/stat 2>/dev/null); ticks=$((${14:-0}+${15:-0})); fi; "
             "read a _ < /proc/uptime; t=${a/./}; rate=$(( (ticks-last)*100/(t-tl+1) )); echo \"S $t $rate\"; "
             "last=$ticks; tl=$t; "                 # rate: 100 is one core
             "if [ $rate -gt 85 ]; then busy=1; calm=0; elif [ $busy -eq 1 ]; then calm=$((calm+1)); fi; "
             "sleep 0.5; i=$((i+1)); done; kill $lp; echo \"T $t0 $calm\"; "
             "grep -E 'Displayed|Fully drawn' $f; rm -f $f", timeout=150)
    took = lambda word: next((int(m[1] or 0) * 60 + int(m[2] or 0) + int(m[3]) / 1000 for m in re.finditer(
        rf"{word} {re.escape(package)}/[^\s:]+(?: for user \d+)?: \+(?:(\d+)m)?(?:(\d+)s)?(\d+)ms", out)), None)
    samples = [int(m[1]) for m in re.finditer(r"^S (\d+) -?\d+", out, re.M)]
    end = re.search(r"^T (\d+) (\d+)", out, re.M)
    calm = (samples[-4] - int(end[1])) / 100 if end and int(end[2]) >= 4 and len(samples) >= 4 else None
    return took("Displayed"), took("Fully drawn"), calm


def layer(package):
    """The surface the app draws its menus on, as SurfaceFlinger names it: its window (video goes on a
    SurfaceView of its own, which says nothing about the menus)."""
    names = [l.strip() for l in sh("dumpsys SurfaceFlinger --list").splitlines() if package + "/" in l]
    names = [n for n in names if not re.match(r"[0-9a-f]{5,8} |ActivityRecord|Bounds for|Background for", n)]
    return ([n for n in names if not n.startswith("SurfaceView")] or [n for n in names if "(BLAST)" in n] or [None])[-1]


def cmd_smooth(args):
    """How quickly and evenly an app answers the remote: for each key, the time until the next new frame and
    the frames that follow, from SurfaceFlinger's own times. Sending a key costs the same for every app, so
    compare with `plint tv smooth home`, which does next to nothing."""
    args or die("plint tv smooth <app> [key...]")
    package = HOME_APP if args[0] == "home" else resolve(args[0])
    keys = [keycode(k) for k in args[1:]] or ["22"] * 4 + ["21"] * 4 + ["20"] * 2 + ["19"] * 2
    if front()[0] != package:
        sh(f"am start -n {launcher_of(package)} >/dev/null 2>&1")
        time.sleep(10)
    surface = layer(package) or die(f"{name_of(package)} draws on nothing SurfaceFlinger names")
    out = sh("L=" + shlex.quote(surface) + "; for k in " + " ".join(keys) + "; do dumpsys SurfaceFlinger --latency-clear; "
             "read a _ < /proc/uptime; input keyevent $k; sleep 0.9; echo \"KEY ${a/./}\"; "
             "dumpsys SurfaceFlinger --latency \"$L\"; done", timeout=180)
    waits, gaps, counts = [], [], []
    for burst in out.split("KEY ")[1:]:
        sent = int(burst.split()[0]) * 10_000_000     # /proc/uptime is in hundredths; frames are in nanoseconds
        shown = sorted(int(r[1]) for r in re.findall(r"^(\d+)\t(\d+)\t(\d+)$", burst, re.M) if sent < int(r[1]) < 1 << 62)
        counts.append(len(shown))
        if shown:
            waits.append((shown[0] - sent) / 1e6)
        gaps += [(b - a) / 1e6 for a, b in zip(shown, shown[1:]) if b - a < 300e6]   # a longer gap is a pause, not a frame
    if not waits:
        return print(f"{name_of(package):<12} drew nothing after {len(keys)} keys")
    waits.sort()
    gaps.sort()
    print(f"{name_of(package):<12} answers in {waits[len(waits) // 2]:.0f} ms (slowest {waits[-1]:.0f}) · "
          f"{sum(counts) / len(counts):.0f} frames a key"
          + (f" · late {100 * sum(g > 25 for g in gaps) // len(gaps)}%" if gaps else "")
          + (f" · no answer to {len(counts) - len(waits)} of {len(counts)} keys" if len(waits) < len(counts) else ""))


def cmd_switch(args):
    """Go round the apps: is each still loaded when you come back to it, and how long does the switch take.
    A stick short of memory closes the app you left, and then every switch is a cold start."""
    if busy() and "--force" not in args:
        die("something is playing: this goes from app to app, so not now (--force to do it anyway)")
    apps = [a for a in args if a != "--force"] or ["youtube", "stremio"]
    comps = [(a, launcher_of(resolve(a))) for a in apps]
    for lap in range(3):
        for app, comp in comps:
            res = sh(f"am start -W -n {comp}", timeout=60)
            state = (re.search(r"LaunchState: (\w+)", res) or [0, "?"])[1]
            took = re.search(r"TotalTime: (\d+)", res)
            if lap:                               # the first lap only loads them
                was = {"HOT": "still loaded", "WARM": "still loaded", "COLD": "had been closed"}.get(state, "slow to come")
                print(f"{app:<10} {was} · {int(took[1]) / 1000:.1f} s" if took else f"{app:<10} {was} · over 10 s", flush=True)
            time.sleep(4 if lap else 22)
            sh("input keyevent 3")
            time.sleep(2)
    kb = {k: int(v) for k, v in re.findall(r"(MemAvailable|SwapFree):\s+(\d+)", sh("cat /proc/meminfo"))}
    print(f"memory     {kb.get('MemAvailable', 0) // 1024} MB free, swap {kb.get('SwapFree', 0) // 1024} MB free")


def cmd_warm(args):
    """Load YouTube and Stremio and go back to the home screen, so that a tile is a 0.2 s switch and not a cold
    start. It shows on the TV for half a minute, and starting an app wakes the stick and with it the TV, so
    only from the home screen, awake, with nothing playing. For after a restart."""
    if busy() or front()[0] != HOME_APP or "Awake" not in sh("dumpsys power | grep -m1 mWakefulness="):
        die("only from the home screen, with the TV on and nothing playing")
    for app, wait in (("youtube", 18), ("stremio", 12)):
        if not sh(f"pidof {APPS[app]}").strip():
            sh(f"am start -n {launcher_of(APPS[app])} >/dev/null 2>&1")
            time.sleep(wait)
            print(f"loaded     {app}", flush=True)
    sh("input keyevent 3")


def cmd_reboot(args):
    if busy() and "--force" not in args:
        die("something is playing: not restarting now (--force to do it anyway)")
    start = time.time()
    adb("-s", serial(), "reboot")
    adb("disconnect", serial())
    time.sleep(10)
    while time.time() - start < 300:
        if port_open(stick()["host"], 5555, 2):
            adb("connect", serial())
            if adb("-s", serial(), "shell", "getprop sys.boot_completed").stdout.strip() == "1":
                return print(f"✓ back after {int(time.time() - start)} s")
        time.sleep(3)
    die("the stick has not come back after 5 min")


def cmd_sh(args):
    args or die("plint tv sh <command>")
    print(sh(" ".join(args)), end="")


def cmd_connect(args):
    connect()
    print(f"✓ connected to {serial()}")


COMMANDS = {"status": cmd_status, "find": cmd_find, "connect": cmd_connect, "on": cmd_on, "off": cmd_off,
            "key": cmd_key, "vol": cmd_vol, "open": cmd_open, "youtube": cmd_youtube, "stremio": cmd_stremio,
            "type": cmd_type, "screen": cmd_screen, "apps": cmd_apps, "disable": cmd_disable,
            "enable": cmd_enable, "install": cmd_install, "tune": cmd_tune, "warm": cmd_warm, "time": cmd_time, "smooth": cmd_smooth, "switch": cmd_switch, "reboot": cmd_reboot, "sh": cmd_sh}

if __name__ == "__main__":
    name, rest = (sys.argv[1:2] or ["status"])[0], sys.argv[2:]
    if name in ("-h", "--help", "help"):
        sys.exit(print(__doc__.strip()))
    COMMANDS.get(name, lambda _: die(f"plint tv {name}? see: plint tv help"))(rest)
