/* ─────────────────────────────────────────────────────────────
   Plint Soft UI · Homepage extras
   - System card: CPU / memory / disk / storage as rings
   - Status card: one strip per Uptime Kuma check
   - AdGuard card: blocked share as a ring
   - line icons instead of the colourful app logos
   Everything is redrawn idempotently, because Homepage re-renders.
   ───────────────────────────────────────────────────────────── */
(function () {
  "use strict";
  var P = window.Plint;
  if (!P) return;

  var ICON = {
    Games: "games", Stremio: "stremio", Notifications: "ntfy", Status: "status",
    AdGuard: "dns", Backups: "backup", Host: "host", Stacks: "stacks",
    Metrics: "metrics", Tailscale: "tailscale", Internet: "speed"
  };
  // Cards Soft UI fills with its own infographics (full width for the first two).
  var SPAN = { Games: 1, Tailscale: 1 }, WIDE = { Games: 1, Tailscale: 1, Internet: 1 };

  var MARK = { Cloudflare: "cloud", "Tailscale DNS": "tailscale", Repo: "code", Claude: "chat" };

  // Compare with what we last wrote: the browser re-serialises innerHTML
  // (<circle/> → <circle></circle>), so comparing against it never matches.
  function set(node, html) { if (node && node.__plintHtml !== html) { node.__plintHtml = html; node.innerHTML = html; } }
  function q(sel, root) { return (root || document).querySelector(sel); }
  function qa(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }
  function nameOf(li) {
    var t = q(".service-title .text-sm", li);
    return t && t.firstChild ? String(t.firstChild.textContent).trim() : "";
  }
  function block(li, label) {
    var hit = qa(".service-block-label", li).filter(function (n) { return n.textContent.trim().toLowerCase() === label; })[0];
    return hit ? hit.parentNode.querySelector(".service-block-value").textContent.trim() : "";
  }
  function num(s) { return parseFloat(String(s).replace(/[^0-9.]/g, "")) || 0; }

  // ── data ────────────────────────────────────────────────────
  var res = {}, kuma = { names: null, beats: null }, backups = null;
  var games = null, gamesHist = null, tail = null, speed = null;

  function get(u) { return fetch(u, { cache: "no-store" }).then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; }); }
  function loadResources() {
    Promise.all([
      get("/api/widgets/resources?type=cpu"),
      get("/api/widgets/resources?type=memory"),
      get("/api/widgets/resources?type=disk&target=/"),
      get("/api/widgets/resources?type=disk&target=/storage"),
      get("/api/widgets/resources?type=uptime")
    ]).then(function (r) {
      res = { cpu: r[0] && r[0].cpu, mem: r[1] && r[1].memory, disk: r[2] && r[2].drive, storage: r[3] && r[3].drive, uptime: r[4] && r[4].uptime };
      decorate();
    });
  }
  function loadKuma() {
    Promise.all([get("/_plint/monitors"), get("/_plint/health")]).then(function (r) {
      if (r[0] && r[0].publicGroupList) {
        kuma.names = [];
        r[0].publicGroupList.forEach(function (g) { g.monitorList.forEach(function (m) { kuma.names.push({ id: String(m.id), name: m.name }); }); });
        kuma.names.sort(function (a, b) { return a.name.localeCompare(b.name); });
      }
      if (r[1] && r[1].heartbeatList) kuma.beats = r[1].heartbeatList;
      decorate();
    });
  }

  // Kopia snapshot history, exported hourly by scripts/theme-data.sh.
  function loadBackups() {
    get("/_plint/data/backups.json").then(function (d) {
      if (d && d.snapshots) { backups = d.snapshots; decorate(); }
    });
  }

  // Game servers, tailnet devices and speed tests, exported every 2 min by theme-data.sh.
  function loadLive() {
    Promise.all([get("/_plint/data/games.json"), get("/_plint/data/games-history.json"),
      get("/_plint/data/tailscale.json"), get("/_plint/data/speedtest.json")]).then(function (r) {
      games = r[0] && r[0].servers; gamesHist = r[1] && r[1].samples;
      tail = r[2] && r[2].devices; speed = r[3] && r[3].results;
      decorate();
    });
  }

  function gib(b) { return (b / 1073741824).toFixed(1).replace(/\.0$/, "") + " GB"; }
  function size(b) { return b >= 1e12 ? (b / 1e12).toFixed(2).replace(/0$/, "") + " TB" : Math.round(b / 1e9) + " GB"; }
  function uptime(s) {
    var d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600);
    if (d >= 14) return Math.floor(d / 7) + " w " + (d % 7) + " d";
    if (d >= 1) return d + " d " + h + " h";
    return h + " h";
  }

  // ── header ──────────────────────────────────────────────────
  var LOGO = '<svg viewBox="0 0 128 128" aria-hidden="true"><rect x="44" y="26" width="40" height="34" rx="7" opacity=".55"/>' +
    '<rect x="32" y="62" width="64" height="18" rx="6" opacity=".78"/><rect x="18" y="84" width="92" height="20" rx="7"/></svg>';

  function header() {
    var img = q('#widgets-wrap img[src$="logo.svg"]');
    if (img && !q(".p-logo", img.parentNode)) img.parentNode.appendChild(P.el("span", "p-logo", LOGO));
    var title = q("#widgets-wrap .text-2xl");
    if (title && !q(".p-greet", title)) title.appendChild(P.el("span", "p-greet"));
    var g = q(".p-greet");
    if (g) {
      var h = new Date().getHours();
      var word = h < 5 ? "Good night." : h < 12 ? "Good morning." : h < 18 ? "Good afternoon." : "Good evening.";
      var st = health();
      var tail = !st ? "" : st.down ? (st.down === 1 ? " One thing needs a look." : " " + st.down + " things need a look.") : " Everything's running.";
      if (g.textContent !== word + tail) g.textContent = word + tail;
    }
    var right = q("#information-widgets-right");
    if (right) {
      var pill = q(".p-healthpill", right);
      if (!pill) {
        pill = P.el("a", "p-healthpill", "<i></i><span></span>");
        pill.href = P.url("status");
        right.insertBefore(pill, right.firstChild);
      }
      var s = health();
      var cls = "p-healthpill " + (!s ? "" : s.down ? "bad" : s.pending ? "warn" : "ok");
      if (pill.className !== cls) pill.className = cls;
      var txt = s ? s.up + " / " + s.total + " up" : "Status";
      if (pill.lastChild.textContent !== txt) pill.lastChild.textContent = txt;
    }
  }
  function health() {
    if (!kuma.beats) return null;
    var ids = Object.keys(kuma.beats), up = 0, pending = 0;
    ids.forEach(function (id) {
      var l = kuma.beats[id], last = l && l[l.length - 1];
      if (last && (last.status === 1 || last.status === 3)) up++;
      else if (last && last.status === 2) pending++;
    });
    return { up: up, pending: pending, down: ids.length - up - pending, total: ids.length };
  }

  // ── System card ─────────────────────────────────────────────
  function gauge(label, pct, sub, tone) {
    return '<div class="p-gauge"><div class="p-dial">' + P.ring(pct, { size: 112, width: 11, tone: tone, label: label + " " + Math.round(pct) + " percent" }) +
      "<b>" + Math.round(pct) + "%</b></div><div><strong>" + P.esc(label) + "</strong><small>" + P.esc(sub) + "</small></div></div>";
  }
  function system() {
    var groups = q("#layout-groups");
    if (!groups || !res.cpu) return;
    var card = q("#p-system");
    if (!card) {
      card = P.el("section");
      card.id = "p-system";
      card.setAttribute("aria-label", "System");
      groups.parentNode.insertBefore(card, groups);
    }
    var m = res.mem, d = res.disk, s = res.storage;
    var memPct = m ? (m.total - m.available) / m.total * 100 : 0;
    var html = '<div class="p-rings">' +
      gauge("CPU", res.cpu.usage, "load " + (res.cpu.load != null ? res.cpu.load.toFixed(2) : "–")) +
      (m ? gauge("Memory", memPct, gib(m.total - m.available) + " of " + gib(m.total)) : "") +
      (d ? gauge("System disk", d.use, size(d.used) + " of " + size(d.size)) : "") +
      (s ? gauge("Storage", s.use, size(s.available) + " free", "info") : "") +
      '</div><div class="p-facts">' +
      (res.uptime ? '<div class="p-fact"><span>Up</span><b>' + uptime(res.uptime) + "</b></div>" : "") +
      (m ? '<div class="p-fact"><span>Memory free</span><b>' + gib(m.available) + "</b></div>" : "") +
      (s ? '<div class="p-fact"><span>Backups disk</span><b>' + size(s.size) + "</b></div>" : "") +
      "</div>";
    set(card, html);
  }

  // ── service cards ───────────────────────────────────────────
  function cards() {
    qa("li.service").forEach(function (li) {
      var name = nameOf(li);
      var a = q(".service-title > a:first-child", li);
      if (a && !q(".p-well", a) && ICON[name]) a.appendChild(P.el("span", "p-well", P.icon(ICON[name], "")));
      var wide = !!q(".relative.flex.flex-row.w-full", li) || !!WIDE[name];
      if (wide !== li.classList.contains("p-wide")) li.classList.toggle("p-wide", wide);
      if (!!SPAN[name] !== li.classList.contains("p-span")) li.classList.toggle("p-span", !!SPAN[name]);
      if (name === "Status") checks(li);
      if (name === "AdGuard") adguard(li);
      if (name === "Backups") nights(li);
      if (name === "Games") gamesCard(li);
      if (name === "Tailscale") tailCard(li);
      if (name === "Internet") speedCard(li);
    });
  }

  function bookmarks() {
    qa("li.bookmark").forEach(function (li) {
      var n = q(".bookmark-name", li), box = q(".shrink-0.flex.items-center.justify-center", li);
      var key = n && MARK[n.textContent.trim()];
      if (key && box && !q(".p-mark", box)) box.appendChild(P.el("span", "p-mark", P.icon(key, "")));
    });
  }

  function checks(li) {
    if (!kuma.names || !kuma.beats) return;
    var box = q(".p-checks", li);
    if (!box) { box = P.el("div", "p-checks"); li.firstElementChild.appendChild(box); li.classList.add("p-has-checks"); }
    var N = 24;
    var html = kuma.names.map(function (m) {
      var l = (kuma.beats[m.id] || []).slice(-N), up = l.filter(function (b) { return b.status === 1 || b.status === 3; }).length;
      var cells = "";
      for (var i = 0; i < N - l.length; i++) cells += '<i class="sx"></i>';
      l.forEach(function (b) { cells += '<i class="s' + (b.status === 3 ? 1 : b.status) + '"></i>'; });
      return '<div class="p-check"><span title="' + P.esc(m.name) + '">' + P.esc(m.name) + '</span><span class="p-strip" role="img" aria-label="' +
        P.esc(m.name + ": " + up + " of last " + l.length + " checks up") + '">' + cells + "</span></div>";
    }).join("");
    set(box, html);
  }

  function adguard(li) {
    var queries = num(block(li, "queries")), blocked = num(block(li, "blocked")), latency = block(li, "latency");
    if (!queries) return;
    var box = q(".p-split", li);
    if (!box) { box = P.el("div", "p-split"); li.firstElementChild.appendChild(box); li.classList.add("p-has-split"); }
    var pct = blocked / queries * 100;
    set(box, '<div class="p-dial">' + P.ring(pct, { size: 96, width: 15, butt: true, tone: "warn", label: Math.round(pct) + " percent of queries blocked" }) +
      "<b>" + Math.round(pct) + "%<small>BLOCKED</small></b></div>" +
      "<dl><dt>Queries</dt><dd>" + P.fmt(queries) + "</dd><dt>Blocked</dt><dd>" + P.fmt(blocked) + "</dd><dt>Answer time</dt><dd>" + P.esc(latency.replace(/(\.\d)\d+/, "$1")) + "</dd></dl>");
  }

  // One dot per night for the last four weeks: sage = a snapshot ran,
  // amber = missed, hollow = tonight (not run yet), faint = before the first one.
  function nights(li) {
    if (!backups) return;
    var box = q(".p-nights", li);
    if (!box) { box = P.el("div", "p-nights"); li.firstElementChild.appendChild(box); }
    var ran = {}, first = null;
    backups.forEach(function (s) {
      var d = new Date(s.start);
      if (isNaN(d)) return;
      ran[d.toDateString()] = true;
      if (!first || d < first) first = d;
    });
    var today = new Date(); today.setHours(0, 0, 0, 0);
    var firstDay = first ? new Date(first.getFullYear(), first.getMonth(), first.getDate()) : null;
    var ok = 0, miss = 0, cells = "";
    for (var i = 27; i >= 0; i--) {
      var d = new Date(today); d.setDate(today.getDate() - i);
      var cls;
      if (ran[d.toDateString()]) { cls = "ok"; ok++; }
      else if (i === 0) cls = "next";
      else if (firstDay && d >= firstDay) { cls = "miss"; miss++; }
      else cls = "pre";
      cells += '<i class="' + cls + '" title="' + d.toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short" }) + '"></i>';
    }
    var label = "Backups, last 28 nights: " + ok + " ran" + (miss ? ", " + miss + " missed" : "") + (ran[today.toDateString()] ? "" : ", tonight still to run");
    set(box, '<span class="p-nights-cap">Last 4 weeks</span><div class="p-nights-grid" role="img" aria-label="' + P.esc(label) + '">' + cells + "</div>");
  }

  function box(li, cls) {
    var b = q("." + cls, li);
    if (!b) { b = P.el("div", cls); li.firstElementChild.appendChild(b); }
    return b;
  }
  function ago(iso) {
    var s = (Date.now() - new Date(iso)) / 1000;
    if (!iso || isNaN(s)) return "";
    if (s < 3600) return Math.max(1, Math.round(s / 60)) + " min ago";
    if (s < 86400) return Math.round(s / 3600) + " h ago";
    return Math.round(s / 86400) + " d ago";
  }

  // Each AMP server: players now as a ring of max players, and the last 24 h
  // as 48 half-hour cells (faint = no data yet, grey = stopped, sage = online,
  // deeper sage = people playing).
  function gamesCard(li) {
    if (!games) return;
    var B = 48, nowS = Date.now() / 1000, t0 = nowS - 86400, w = 86400 / B;
    var running = games.filter(function (g) { return g.running; }), online = 0;
    running.forEach(function (g) { online += g.players; });
    var tiles = games.map(function (g) {
      var cells = [];
      for (var i = 0; i < B; i++) cells.push(null);
      (gamesHist || []).forEach(function (h) {
        var v = h[1][g.id], b = Math.floor((h[0] - t0) / w);
        if (v === undefined || b < 0 || b >= B) return;
        cells[b] = cells[b] === null ? v : Math.max(cells[b], v);
      });
      var peak = 0, up = 0, seen = 0;
      var strip = cells.map(function (v) {
        if (v === null) return '<i class="x"></i>';
        seen++;
        if (v < 0) return '<i class="off"></i>';
        up++; peak = Math.max(peak, v);
        return '<i class="' + (v === 0 ? "idle" : "on") + '"></i>';
      }).join("");
      var max = g.max || 0, pct = g.running && max ? g.players / max * 100 : 0;
      var dial = g.running
        ? P.ring(pct, { size: 60, width: 7, label: g.players + " of " + (max || "?") + " players" }) + "<b>" + g.players + "</b>"
        : '<span class="p-off">' + P.icon("games", "") + "</span>";
      var state = !g.running ? "Stopped" : g.players ? g.players + (g.players === 1 ? " player" : " players") + (max ? " of " + max : "") : "Online · empty";
      var label = g.name + ", last 24 hours: " + (seen ? "online " + Math.round(up / seen * 100) + "% of the time" + (peak ? ", peak " + peak + " players" : "") : "no data yet");
      return '<div class="p-game' + (g.running ? " up" : "") + '"><div class="p-game-top"><div class="p-dial">' + dial + "</div><div><strong>" +
        P.esc(g.name) + "</strong><small>" + P.esc(state) + '</small></div></div><span class="p-strip p-strip-24" role="img" aria-label="' + P.esc(label) + '">' + strip + "</span></div>";
    }).join("");
    var sum = "<b>" + online + "</b> " + (online === 1 ? "player" : "players") + " online · " + running.length + " of " + games.length + " servers running";
    set(box(li, "p-games"), '<div class="p-sumline"><span>' + sum + '</span><span class="p-cap">last 24 h →</span></div><div class="p-games-grid">' + tiles + "</div>");
  }

  // Tailnet devices: online first, offline with when they were last seen.
  function tailCard(li) {
    if (!tail) return;
    var on = tail.filter(function (d) { return d.online; }).length;
    var OS = { iOS: "iPhone/iPad", android: "Android", windows: "Windows", linux: "Linux", macOS: "Mac" };
    var devs = tail.map(function (d) {
      var sub = d.self ? "this server" : (OS[d.os] || d.os) + (d.online ? "" : " · seen " + ago(d.lastSeen));
      return '<div class="p-dev ' + (d.online ? "on" : "off") + '"><i></i><span><b>' + P.esc(d.name) + "</b><small>" + P.esc(sub) + "</small></span></div>";
    }).join("");
    set(box(li, "p-tail"), '<div class="p-sumline"><span><b>' + on + "</b> of " + tail.length + ' devices online</span></div><div class="p-devs">' + devs + "</div>");
  }

  // Latest speed test as numbers, the last week (28 tests) as a line.
  function speedCard(li) {
    var ok = (speed || []).filter(function (r) { return r.ok; });
    if (!ok.length) return;
    var last = ok[ok.length - 1], pts = ok.slice(-28);
    var nums = '<div class="p-speed-nums"><div><span>Down</span><b>' + Math.round(last.down) + "<small> Mbit/s</small></b></div>" +
      "<div><span>Up</span><b>" + Math.round(last.up) + "<small> Mbit/s</small></b></div>" +
      "<div><span>Ping</span><b>" + (Math.round(last.ping * 10) / 10) + "<small> ms</small></b></div></div>";
    var chart;
    if (pts.length < 2) {
      chart = '<p class="p-speed-note">The graph fills in as tests run, four a day.</p>';
    } else {
      var top = Math.max.apply(null, pts.map(function (r) { return Math.max(r.down, r.up); })) * 1.1 || 1;
      var xy = function (key) {
        return pts.map(function (r, i) { return (i / (pts.length - 1) * 300).toFixed(1) + " " + (80 - r[key] / top * 76).toFixed(1); });
      };
      var dl = xy("down"), ul = xy("up");
      chart = '<div class="p-speed-chart"><svg viewBox="0 0 300 80" preserveAspectRatio="none" role="img" aria-label="Last ' + pts.length +
        ' speed tests: download ' + Math.round(Math.min.apply(null, pts.map(function (r) { return r.down; }))) + " to " +
        Math.round(Math.max.apply(null, pts.map(function (r) { return r.down; }))) + ' Mbit/s"><path class="area" d="M' + dl.join(" L") + " L300 80 L0 80 Z\"/>" +
        '<path class="dl" d="M' + dl.join(" L") + '"/><path class="ul" d="M' + ul.join(" L") + '"/></svg></div>';
    }
    var at = new Date(last.at);
    var when = isNaN(at) ? "" : at.toLocaleString("en-GB", { weekday: "short", hour: "2-digit", minute: "2-digit" });
    set(box(li, "p-speed"), nums + chart + '<div class="p-speed-cap"><span>Last test ' + P.esc(when) + "</span>" +
      (pts.length > 1 ? '<span><i class="k dl"></i>down <i class="k ul"></i>up</span>' : "") + "</div>");
  }

  // ── run ─────────────────────────────────────────────────────
  var busy = false;
  function decorate() {
    if (busy) return;
    busy = true;
    try { header(); system(); cards(); bookmarks(); } finally { busy = false; }
  }
  var queued = false;
  function schedule() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(function () { queued = false; decorate(); });
  }
  function start() {
    new MutationObserver(schedule).observe(document.body, { childList: true, subtree: true, characterData: true });
    loadResources(); loadKuma(); loadBackups(); loadLive();
    setInterval(loadLive, 60000);
    setInterval(loadBackups, 600000);
    setInterval(loadResources, 15000);
    setInterval(loadKuma, 60000);
    decorate();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
