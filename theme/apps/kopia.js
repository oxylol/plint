/* ─────────────────────────────────────────────────────────────
   Plint Terminal · Kopia extras
   - the Plint strip goes into Kopia's navbar; the brand becomes
     BACKUPS / АРХИВ; every page gets a title
   - Snapshots: one instrument per source: last / size / next as
     dot-matrix figures, a strip of 28 night lamps (lit = kept,
     hollow = missed, dim = still to run) and a one-line readout;
     a lamp read by tap or pointer says its own night
   - a source's history: latest size / files / duration, and growth
     per snapshot (size vs the one before, last 14) as cell columns
   - snapshot browser: a dot meter under each entry's size
   Data comes from read-only GETs on Kopia's own API (same origin, the
   browser is already signed in). Everything is redrawn idempotently
   under a MutationObserver, and only written when it changed.
   ───────────────────────────────────────────────────────────── */
(function () {
  "use strict";
  var P = window.Plint;
  if (!P) return;
  P.mount = ["#root > nav.navbar", "nav.navbar"];

  var DAY = 864e5, HOUR = 36e5, SLACK = 3 * HOUR, NIGHTS = 28, BARS = 14;

  // friendly names for the sources Plint sets up; anything else uses its folder name
  var NAMES = { "/sources/amp": "Game servers", "/sources/plint-data": "App data", "/sources/plint.env": "Secrets" };
  var TITLES = {
    snapshots: "Snapshots", source: "Snapshots", dir: "Snapshots", restore: "Restore", "new": "New snapshot",
    policies: "Policies", policy: "Policies", tasks: "Tasks", task: "Tasks", repo: "Repository", prefs: "Preferences"
  };

  // ── small helpers ───────────────────────────────────────────
  function q(sel, root) { return (root || document).querySelector(sel); }
  function qa(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }
  // write only when our own last write differs (innerHTML round-trips aren't exact)
  function set(node, html) { if (!node || node._p === html) return false; node._p = html; node.innerHTML = html; return true; }
  function attr(node, k, v) { if (node && node.getAttribute(k) !== v) node.setAttribute(k, v); }
  function text(node, v) { if (node && node.textContent !== v) node.textContent = v; }

  function page() {
    var p = location.pathname;
    if (/^\/snapshots\/single-source/.test(p)) return "source";
    if (/^\/snapshots\/dir\/.+\/restore/.test(p)) return "restore";
    if (/^\/snapshots\/dir\//.test(p)) return "dir";
    if (/^\/snapshots\/new/.test(p)) return "new";
    if (/^\/snapshots/.test(p) || p === "/") return "snapshots";
    if (/^\/policies\/edit/.test(p)) return "policy";
    if (/^\/policies/.test(p)) return "policies";
    if (/^\/tasks\/./.test(p)) return "task";
    if (/^\/tasks/.test(p)) return "tasks";
    if (/^\/repo/.test(p)) return "repo";
    if (/^\/preferences/.test(p)) return "prefs";
    return "other";
  }

  function nameOf(path) {
    if (NAMES[path]) return NAMES[path];
    var b = String(path || "").replace(/\/+$/, "").split(/[\\/]/).pop() || path || "Source";
    return b.charAt(0).toUpperCase() + b.slice(1);
  }
  function sourceKey(s) { return s.userName + "@" + s.host + ":" + s.path; }
  function historyHref(s) {
    return "/snapshots/single-source?userName=" + encodeURIComponent(s.userName) + "&host=" + encodeURIComponent(s.host) + "&path=" + encodeURIComponent(s.path);
  }

  // base-10, one decimal, like Kopia's default byte setting (140.9 GB)
  function bytes(b) {
    if (b == null || !isFinite(b)) return "–";
    var u = ["B", "KB", "MB", "GB", "TB", "PB"], i = 0;
    while (Math.abs(b) >= 1000 && i < u.length - 1) { b /= 1000; i++; }
    return (i ? b.toFixed(1).replace(/\.0$/, "") : String(Math.round(b))) + " " + u[i];
  }
  function parseBytes(s) {
    var m = String(s).replace(/,/g, "").match(/(-?[\d.]+)\s*([KMGTP]?)(i?)B\b/i);
    if (!m) return null;
    var k = m[3] ? 1024 : 1000, e = " KMGTP".indexOf(m[2].toUpperCase() || " ");
    return parseFloat(m[1]) * Math.pow(k, Math.max(0, e));
  }
  function dur(ms) {
    if (!(ms >= 0)) return "";
    var s = ms / 1000;
    if (s < 10) return s.toFixed(1).replace(/\.0$/, "") + " s";
    if (s < 60) return Math.round(s) + " s";
    var m = Math.floor(s / 60), r = Math.round(s % 60);
    if (m < 60) return m + " min" + (m < 10 && r ? " " + r + " s" : "");
    var h = Math.floor(m / 60);
    return h + " h" + (m % 60 ? " " + (m % 60) + " min" : "");
  }
  var TIME = { hour: "2-digit", minute: "2-digit", hour12: false };
  function hm(t) { return new Date(t).toLocaleTimeString("en-GB", TIME); }
  function dm(t) { return new Date(t).toLocaleDateString("en-GB", { day: "numeric", month: "short" }); }
  function wdm(t) { return new Date(t).toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short" }); }
  function midnight(t) { var d = new Date(t); d.setHours(0, 0, 0, 0); return d.getTime(); }
  // "tonight" · "last night" · "today" · "yesterday" · "tomorrow" · "Mon 19 Oct"
  function when(t, now) {
    var d = Math.round((midnight(t) - midnight(now)) / DAY), h = new Date(t).getHours(), night = h < 7 || h >= 20;
    if (night && t > now && t - now < 16 * HOUR) return "tonight";
    if (night && t <= now && now - t < 16 * HOUR) return "last night";
    if (d === 0) return "today";
    if (d === -1) return "yesterday";
    if (d === 1) return "tomorrow";
    return wdm(t);
  }
  // "140.9 GB" → ["140.9", "GB"]: the figure and its unit
  function split(s) {
    var m = String(s).match(/^(-?[\d.,:]+)\s+(\S+)$/);
    return m ? [m[1], m[2]] : [String(s), ""];
  }
  // a duration as a figure: 0.4 S · 42 S · 4:12 MIN · 1:05 H
  function durFig(ms) {
    if (!(ms >= 0)) return ["--", ""];
    var s = ms / 1000;
    if (s < 10) return [s.toFixed(1), "s"];
    if (s < 60) return [String(Math.round(s)), "s"];
    var m = Math.floor(s / 60), r = Math.floor(s % 60);
    if (m < 60) return [m + ":" + (r < 10 ? "0" : "") + r, "min"];
    return [Math.floor(m / 60) + ":" + (m % 60 < 10 ? "0" : "") + (m % 60), "h"];
  }
  // one cell of a row of figures; the digits are lit by fill() once the markup is in
  function fig(id, v, label, sub) {
    return '<div><b data-id="' + P.esc(id) + '" data-n="' + P.esc(v[0]) + '" data-u="' + P.esc(v[1] || "") + '"></b>' +
      '<span class="p-lbl">' + P.esc(label) + "</span><small>" + P.esc(sub || "") + "</small></div>";
  }
  function fill(box) {
    var olds = box._v || {}, vals = {};
    qa("b[data-n]", box).forEach(function (b) {
      var id = b.getAttribute("data-id"), n = b.getAttribute("data-n"), u = b.getAttribute("data-u"), o = olds[id];
      vals[id] = [n, u];
      try {
        if (o && (o[0] !== n || o[1] !== u)) P.num(b, o[0], o[1]);   // a changed figure spins into place
        P.num(b, n, u);
      } catch (e) { b.textContent = n + (u ? " " + u : ""); }
    });
    box._v = vals;
  }
  // lamps come on one after another, once per instrument
  function light(box, sel) {
    if (box._lit) return;
    box._lit = 1;
    try { qa(sel, box).forEach(function (l, i) { P.strike(l, (i % NIGHTS) * 34); }); } catch (e) {}
  }
  function sec(title, ru, right) {
    return '<header class="p-sec"><h2>' + P.esc(title) + ' <span class="p-ru">/ ' + ru + "</span></h2><span>" + P.esc(right || "") + "</span></header>";
  }

  // ── data: read-only GETs on Kopia's API ─────────────────────
  var data = { sources: null, local: null, snaps: {}, at: 0, busy: false };
  function get(u) {
    var meta = q('meta[name="kopia-csrf-token"]'), h = { Accept: "application/json" };
    if (meta && meta.content) h["X-Kopia-Csrf-Token"] = meta.content;
    return fetch(u, { credentials: "same-origin", cache: "no-store", headers: h })
      .then(function (r) { return r.ok ? r.json() : null; })
      .catch(function () { return null; });
  }
  function snapsUrl(s) {
    return "/api/v1/snapshots?userName=" + encodeURIComponent(s.userName) + "&host=" + encodeURIComponent(s.host) + "&path=" + encodeURIComponent(s.path) + "&all=1";
  }
  function load() {
    if (data.busy || Date.now() - data.at < 55e3) return;
    data.busy = true;
    get("/api/v1/sources").then(function (r) {
      if (!r || !r.sources) { data.busy = false; return; }
      return Promise.all(r.sources.map(function (s) { return get(snapsUrl(s.source)); })).then(function (all) {
        var snaps = {};
        r.sources.forEach(function (s, i) {
          var l = (all[i] && all[i].snapshots) || [];
          snaps[sourceKey(s.source)] = l.slice().sort(function (a, b) { return Date.parse(a.startTime) - Date.parse(b.startTime); });
        });
        data.sources = r.sources;
        data.local = r.sources.filter(function (s) { return s.source.userName === r.localUsername && s.source.host === r.localHost; });
        data.snaps = snaps;
        data.at = Date.now();
        data.busy = false;
        decorate();
      });
    }).catch(function () { data.busy = false; });
  }

  // ── the nightly calendar ────────────────────────────────────
  // One cell per scheduled run, newest last. A run "happened" when a
  // snapshot started between 3 h before it and 3 h before the next one
  // (manual and catch-up runs count). Nights before the first snapshot,
  // and nights older than the densest retention rule keeps (latest /
  // hourly / daily), are "no record" rather than missed: Kopia prunes
  // those snapshots on purpose.
  function nights(src, snaps, now) {
    var sched = (src && src.schedule) || {};
    var nightly = !!((sched.timeOfDay && sched.timeOfDay.length) || (sched.cron && sched.cron.length));
    var next = src && src.nextSnapshotTime ? Date.parse(src.nextSnapshotTime) : NaN;
    var starts = snaps.map(function (s) { return Date.parse(s.startTime); }).filter(isFinite);
    var first = starts.length ? Math.min.apply(null, starts) : Infinity;
    var reasons = snaps.some(function (s) { return s.retention && s.retention.length; });
    var dense = reasons ? Infinity : first;
    snaps.forEach(function (s) {
      if ((s.retention || []).some(function (r) { return /^(latest|hourly|daily)-/.test(r); })) dense = Math.min(dense, Date.parse(s.startTime));
    });
    var anchor, lo, hi;
    if (nightly && isFinite(next)) {
      anchor = next;
      while (anchor - DAY > now) anchor -= DAY;
      while (anchor <= now - HOUR * 20) anchor += DAY;
      lo = function (k) { return anchor - k * DAY - SLACK; };
      hi = function (k) { return anchor - (k - 1) * DAY - SLACK; };
    } else {
      anchor = midnight(now) + DAY;
      lo = function (k) { return anchor - (k + 1) * DAY; };
      hi = function (k) { return anchor - k * DAY; };
    }
    var running = src && (src.status === "UPLOADING" || src.status === "PENDING");
    var cells = [];
    for (var k = NIGHTS - 1; k >= 0; k--) {
      var a = lo(k), b = hi(k), at = nightly ? anchor - k * DAY : a;
      var hit = snaps.filter(function (s) { var t = Date.parse(s.startTime); return t >= a && t < b; });
      var st;
      if (hit.length) st = "ok";
      else if (nightly && at > now) st = "next";
      else if (now >= a && now < b && (running || (nightly && now < at + HOUR))) st = "run";
      else if (!nightly) st = k === 0 ? "next" : "none";
      else if (b <= first || a < dense) st = "none";
      else st = "miss";
      if (st === "next" && running && k === 0 && now >= a) st = "run";
      cells.push({ st: st, at: at, hit: hit[hit.length - 1] });
    }
    return { cells: cells, nightly: nightly, first: first };
  }

  // the strip of lamps, its two end labels and the readout under it
  function strip(n, took, now) {
    var c = n.cells;
    var count = { ok: 0, miss: 0, none: 0, next: 0, run: 0 };
    c.forEach(function (x) { count[x.st]++; });
    var lamps = c.map(function (x) {
      var d = wdm(x.at), t;
      if (x.st === "ok") t = d + " · kept · started " + hm(Date.parse(x.hit.startTime)) +
        (x.hit.summary && x.hit.summary.size != null ? " · " + bytes(x.hit.summary.size) : "");
      else if (x.st === "miss") t = d + " · missed";
      else if (x.st === "next") t = d + " · runs " + (n.nightly ? "at " + hm(x.at) : "later");
      else if (x.st === "run") t = d + " · running now";
      else t = d + " · " + (x.at < n.first ? "before the first backup" : "thinned out by retention");
      return '<i class="' + x.st + '" data-t="' + P.esc(t) + '"></i>';
    }).join("");
    var ran = count.ok + count.miss;
    var word = n.nightly ? "nights" : "days";
    var label = "Last " + (NIGHTS - 1) + " " + word + ": " + count.ok + " backed up, " + count.miss + " missed" +
      (count.none ? ", " + count.none + " with no snapshot on record" : "") + ". " +
      (count.run ? "A snapshot is running now. " : "") +
      (count.next && n.nightly ? "Tonight's runs at " + hm(c[c.length - 1].at) + "." : "");
    var rd = ran ? count.ok + " of " + ran + " " + (ran === 1 ? word.slice(0, -1) : word) + " kept" + (count.miss ? " · " + count.miss + " missed" : "") +
        (took ? " · last run took " + took : "") + (isFinite(n.first) ? " · on record since " + dm(n.first) : "")
      : count.run ? "first snapshot running now"
      : n.nightly ? "no snapshot yet · first run " + when(c[c.length - 1].at, now) + " at " + hm(c[c.length - 1].at)
      : "no snapshot yet · manual only";
    return '<div class="p-nights" role="img" aria-label="' + P.esc(label) + '">' + lamps + "</div>" +
      '<div class="p-foot" aria-hidden="true"><span>' + P.esc(dm(c[0].at)) + "</span><span>" + P.esc(n.nightly ? when(c[c.length - 1].at, now) : "today") + "</span></div>" +
      '<p class="p-rd" data-d="' + P.esc(rd) + '">' + P.esc(rd) + "</p>";
  }

  // the chip in each card's header: running > errors > never ran > missed > paused > ok
  function chip(src, cal) {
    var last = src.lastSnapshot, errs = last && last.stats ? (last.stats.errorCount || 0) : 0;
    if (src.status === "UPLOADING") {
      var u = src.upload || {}, pct = u.estimatedBytes ? Math.min(99, Math.round((u.hashedBytes + u.cachedBytes) / u.estimatedBytes * 100)) : null;
      return ["warn", "Running" + (pct != null && isFinite(pct) ? " · " + pct + "%" : "")];
    }
    if (errs) return ["bad", errs + (errs === 1 ? " error" : " errors")];
    if (!last) return ["warn", "No snapshot yet"];
    var c = cal.cells, prev = c.filter(function (x) { return x.st !== "next" && x.st !== "run"; }).pop();
    if (prev && prev.st === "miss") return ["warn", "Missed last night"];
    if (src.status === "PAUSED") return ["warn", "Paused"];
    return ["ok", "Up to date"];
  }

  function signed(d) { return (d > 0 ? "+" : d < 0 ? "−" : "±") + bytes(Math.abs(d)); }

  function sourceCard(src, now, i) {
    var s = src.source, snaps = data.snaps[sourceKey(s)] || [];
    var last = src.lastSnapshot, cal = nights(src, snaps, now), ch = chip(src, cal);
    var size = last && last.stats ? last.stats.totalSize : null;
    var files = last && last.rootEntry && last.rootEntry.summ ? last.rootEntry.summ.files : last && last.stats ? last.stats.fileCount : null;
    var lt = last ? Date.parse(last.startTime) : NaN, le = last ? Date.parse(last.endTime) : NaN;
    var next = src.nextSnapshotTime ? Date.parse(src.nextSnapshotTime) : NaN;
    var figs =
      fig(i + "l", [isFinite(lt) ? hm(lt) : "--", ""], "Last", isFinite(lt) ? when(lt, now) : "no snapshot yet") +
      fig(i + "s", size != null ? split(bytes(size)) : ["--", ""], "Size", files != null ? P.fmt(files) + (files === 1 ? " file" : " files") : "") +
      fig(i + "n", [isFinite(next) ? hm(next) : "--", ""], "Next", isFinite(next) ? when(next, now) : "manual only");
    return '<article class="p-src" aria-label="' + P.esc(nameOf(s.path)) + '">' +
      '<header class="p-head"><div class="p-name"><h3><a href="' + P.esc(historyHref(s)) + '" data-p-nav>' + P.esc(nameOf(s.path)) + "</a></h3>" +
      '<span class="p-path">' + P.esc(s.path) + "</span></div>" +
      '<span class="p-state ' + ch[0] + '"><i class="p-blk' + (ch[0] === "bad" ? " alarm" : "") + '"></i>' + P.esc(ch[1]) + "</span></header>" +
      '<div class="p-big">' + figs + "</div>" + strip(cal, isFinite(le - lt) ? (le - lt < 100 ? "under 0.1 s" : dur(le - lt)) : "", now) + "</article>";
  }

  // ── pages ───────────────────────────────────────────────────
  function frame() { return q("#root > .container-fluid"); }

  function chrome(pg) {
    attr(document.documentElement, "data-p-page", pg);
    var brand = q("#root > nav.navbar .navbar-brand");
    if (brand && !q(".p-label", brand)) {
      brand.appendChild(P.el("span", "p-label", "Backups"));
      var ru = P.el("span", "p-ru", "/ Архив");
      ru.setAttribute("aria-hidden", "true");
      brand.appendChild(ru);
      brand.setAttribute("aria-label", "Backups home");
    }
    navHint();
    var f = frame();
    if (!f) return;
    var t = q(":scope > h1.p-title", f);
    var active = q("#root > nav.navbar .nav-link.active");
    var name = TITLES[pg] || (active && active.getAttribute("data-title")) || "Backups";
    if (!t) { t = P.el("h1", "p-title"); f.insertBefore(t, f.firstChild); }
    else if (f.firstChild !== t) f.insertBefore(t, f.firstChild);
    text(t, name);
    if (document.title !== name + " · Backups") document.title = name + " · Backups";
  }

  // phones: the tabs scroll sideways; fade the edge that has more, keep the active tab in view
  var navPage = null;
  function navHint() {
    var c = q("#root > nav.navbar .navbar-collapse");
    if (!c) return;
    if (!c._p) {
      c._p = 1;
      c.addEventListener("scroll", function () { navEdge(c); }, { passive: true });
      window.addEventListener("resize", function () { navEdge(c); });
    }
    if (navPage !== location.pathname && c.scrollWidth > c.clientWidth + 2 && reveal(c)) navPage = location.pathname;
    navEdge(c);
  }
  function reveal(c) {
    var act = q(".nav-link.active", c);
    if (!act) return false;
    var a = act.getBoundingClientRect(), b = c.getBoundingClientRect();
    if (a.right + 12 > b.right) c.scrollLeft += a.right + 12 - b.right;
    else if (a.left - 12 < b.left) c.scrollLeft -= b.left - a.left + 12;
    return true;
  }
  function navEdge(c) {
    var more = c.scrollWidth - c.clientWidth;
    var v = more <= 2 ? "" : c.scrollLeft >= more - 2 ? "start" : c.scrollLeft <= 2 ? "end" : "both";
    if ((c.getAttribute("data-p-more") || "") !== v) { if (v) c.setAttribute("data-p-more", v); else c.removeAttribute("data-p-more"); }
  }

  function sourcesPage(now) {
    var f = frame(), anchor = f && (q(":scope > .list-actions", f) || q(":scope > .p-2", f));
    if (!anchor || !data.sources) return;
    var box = q("#p-sources");
    if (!box) { box = P.el("section", "p-sources"); box.id = "p-sources"; box.setAttribute("aria-label", "Backup sources"); }
    if (box.parentNode !== f || box.nextElementSibling !== anchor) f.insertBefore(box, anchor);
    // this server's own sources (Kopia's "Local Snapshots"); every source if none match
    var list = data.local && data.local.length ? data.local : data.sources;
    var at = list.map(function (s) { return s.nextSnapshotTime ? Date.parse(s.nextSnapshotTime) : NaN; }).filter(isFinite)[0];
    var html = sec("Sources", "Источники", NIGHTS + " nights" + (at ? " · " + hm(at) : "")) +
      '<div class="p-grid">' + list.map(function (s, i) { return sourceCard(s, now, i); }).join("") + "</div>";
    if (set(box, html)) { fill(box); light(box, ".p-nights i"); }
  }

  function historyPage(now) {
    var f = frame(), tableRow = f && q(".p-2", f);
    tableRow = tableRow && (tableRow.closest(".row") || tableRow);
    if (!tableRow || tableRow.parentNode !== f || !data.sources) return;
    var u = new URLSearchParams(location.search);
    var key = u.get("userName") + "@" + u.get("host") + ":" + u.get("path");
    var src = data.sources.filter(function (s) { return sourceKey(s.source) === key; })[0];
    var snaps = data.snaps[key];
    if (!src || !snaps) return;
    var box = q("#p-history");
    if (!box) { box = P.el("section", "p-hist"); box.id = "p-history"; box.setAttribute("aria-label", "Snapshot history"); }
    if (box.parentNode !== f || box.nextElementSibling !== tableRow) f.insertBefore(box, tableRow);

    // growth = total size minus the snapshot before it (Kopia's list has no "bytes added")
    var win = snaps.slice(-(BARS + 1)), recent = win.slice(-BARS), off = win.length - recent.length;
    var rows = recent.map(function (x, j) {
      var p = win[off + j - 1], sz = x.summary ? x.summary.size : null;
      return { x: x, t: Date.parse(x.startTime), size: sz, d: p && p.summary && sz != null ? sz - p.summary.size : null, failed: x.summary ? x.summary.numFailed : 0 };
    });
    var ds = rows.map(function (r) { return r.d; }).filter(function (d) { return d != null; });
    // each column is CELLS cells; the biggest change of the window fills it, any change lights at least one
    var CELLS = 8, top = Math.max.apply(null, ds.map(Math.abs).concat([0]));
    var bars = "", days = "", prevLabel = null;
    for (var i = 0; i < BARS - rows.length; i++) { bars += '<i class="void"></i>'; days += "<span></span>"; }
    rows.forEach(function (r) {
      var tip = wdm(r.t) + " " + hm(r.t) + " · " + bytes(r.size) + (r.d != null ? " · " + signed(r.d) : " · first snapshot") +
        (r.failed ? " · " + r.failed + " failed" : "");
      var cls = r.d != null && r.d < 0 ? "neg" : "";
      if (r.failed) cls += " err";
      var lit = r.d && top ? Math.max(1, Math.round(Math.abs(r.d) / top * CELLS)) : 0;
      bars += "<i" + (cls ? ' class="' + cls.trim() + '"' : "") + ' style="--v:' + (lit / CELLS * 100).toFixed(1) + '%" data-t="' + P.esc(tip) + '"></i>';
      var day = String(new Date(r.t).getDate()), mo = new Date(r.t).toLocaleDateString("en-GB", { month: "short" });
      var sub = prevLabel === null || prevLabel.mo !== mo ? mo : prevLabel.day === day ? hm(r.t) : "&nbsp;";
      days += "<span>" + day + "<small>" + sub + "</small></span>";
      prevLabel = { day: day, mo: mo };
    });
    var last = rows[rows.length - 1], sm = last && last.x.summary;
    var lt = last ? last.t : NaN, le = last ? Date.parse(last.x.endTime) : NaN;
    var took = rows.map(function (r) { return Date.parse(r.x.endTime) - r.t; }).filter(function (v) { return v >= 0; });
    var avg = took.length ? took.reduce(function (a, b) { return a + b; }, 0) / took.length : NaN;
    var label = !rows.length ? "No snapshots yet." : !ds.length ? "One snapshot so far, " + bytes(last.size) + "; growth shows from the second snapshot on." :
      "Growth per snapshot over the last " + rows.length + ", between " + signed(Math.min.apply(null, ds)) + " and " + signed(Math.max.apply(null, ds)) +
      (last.d != null ? "; latest " + signed(last.d) : "") + ".";
    var head = last && last.d != null ? signed(last.d) + " last time" : "";
    var rd = !rows.length ? "no snapshots yet" : !ds.length ? "one snapshot so far · " + bytes(last.size) + " · growth shows from the second one on" :
      !top ? "no change in size over the last " + rows.length + " snapshots · " + bytes(last.size) :
      "one cell = " + bytes(top / CELLS) + (last.d != null ? " · latest " + signed(last.d) : "") +
      " · between " + signed(Math.min.apply(null, ds)) + " and " + signed(Math.max.apply(null, ds));
    var html = sec("History", "История", P.fmt(snaps.length) + (snaps.length === 1 ? " snapshot" : " snapshots")) +
      '<div class="p-hist-body"><div>' +
      '<header class="p-head"><div class="p-name"><h3>' + P.esc(nameOf(src.source.path)) + '</h3><span class="p-path">' + P.esc(src.source.path) + "</span></div></header>" +
      '<div class="p-big">' +
      fig("hl", sm ? split(bytes(sm.size)) : ["--", ""], "Latest", isFinite(lt) ? when(lt, now) + " " + hm(lt) : "") +
      fig("hf", [sm ? P.fmt(sm.files) : "--", ""], "Files", sm ? P.fmt(sm.dirs) + " folders" : "") +
      fig("ht", durFig(le - lt), "Took", took.length > 1 ? "average " + dur(avg) : took.length ? "first snapshot" : "") +
      "</div></div><div>" +
      '<div class="p-sub"><span>Growth per snapshot</span><b>' + P.esc(head) + "</b></div>" +
      '<div class="p-cols" role="img" aria-label="' + P.esc(label) + '">' + bars + "</div>" +
      '<div class="p-days" aria-hidden="true">' + days + "</div></div></div>" +
      '<p class="p-rd" data-d="' + P.esc(rd) + '">' + P.esc(rd) + "</p>";
    if (set(box, html)) { fill(box); light(box, ".p-cols i"); }
  }

  // snapshot browser: a bar under each size, scaled to the biggest entry on the page
  function dirPage() {
    qa("#root .p-2 > .table").forEach(function (table) {
      var heads = qa("thead th", table), col = -1;
      heads.forEach(function (th, i) { if (/^\s*size\s*$/i.test(th.textContent)) col = i; });
      if (col < 0) return;
      var cells = qa("tbody > tr > td:nth-child(" + (col + 1) + ")", table);
      var vals = cells.map(function (td) { var s = td.querySelector("span"); return s ? parseBytes(s.textContent) : null; });
      var max = Math.max.apply(null, vals.filter(function (v) { return v != null; }).concat([1]));
      cells.forEach(function (td, i) {
        var bar = q(":scope > .p-sizebar", td);
        if (vals[i] == null) { if (bar) bar.remove(); return; }
        if (!bar) { bar = P.el("i", "p-sizebar"); bar.setAttribute("aria-hidden", "true"); }
        if (td.lastChild !== bar) td.appendChild(bar);   // React may re-add the value after us
        var w = Math.max(3, vals[i] / max * 100).toFixed(1) + "%";
        if (bar.style.getPropertyValue("--v") !== w) bar.style.setProperty("--v", w);
      });
    });
  }

  // Kopia spaces some pages with <div class="row"><div class="col">&nbsp;</div></div>
  function spacers() {
    qa("#root > .container-fluid .row").forEach(function (r) {
      if (r.hasAttribute("data-p-spacer") || r.textContent.trim() || r.querySelector("input, button, select, textarea, table, a, img, svg")) return;
      r.setAttribute("data-p-spacer", "");
    });
  }

  // Preferences: Kopia's theme and font-size pickers no longer change the look here
  function prefsPage() {
    var sel = q("#themeSelector");
    if (!sel || q(".p-note-theme", sel.parentNode)) return;
    sel.parentNode.appendChild(P.el("small", "p-note-theme", "The Plint look is fixed: theme and font size here do not change it."));
  }

  // a lamp or a column, tapped or pointed at, says its own night in the readout; then the summary returns
  var rdNode = null, rdTimer = null;
  function unread() {
    clearTimeout(rdTimer); rdTimer = null;
    if (!rdNode) return;
    var host = rdNode.closest(".p-src, .p-hist"), rd = host && q(".p-rd", host);
    rdNode.classList.remove("sel");
    rdNode = null;
    if (rd) text(rd, rd.getAttribute("data-d") || "");
  }
  function read(e) {
    var i = e.target && e.target.closest && e.target.closest("#p-sources i[data-t], #p-history i[data-t]");
    if (!i || i === rdNode) return;
    var host = i.closest(".p-src, .p-hist"), rd = host && q(".p-rd", host);
    if (!rd) return;
    unread();
    rdNode = i;
    i.classList.add("sel");
    text(rd, i.getAttribute("data-t"));
    rdTimer = setTimeout(unread, 6000);
  }

  function cleanup(pg) {
    if (pg !== "snapshots") { var a = q("#p-sources"); if (a) a.remove(); }
    if (pg !== "source") { var b = q("#p-history"); if (b) b.remove(); }
  }

  // ── run ─────────────────────────────────────────────────────
  var busy = false;
  function decorate() {
    if (busy) return;
    busy = true;
    try {
      var pg = page(), now = Date.now();
      chrome(pg);
      cleanup(pg);
      if (pg === "snapshots" || pg === "source") load();
      if (pg === "snapshots") sourcesPage(now);
      if (pg === "source") historyPage(now);
      if (pg === "dir") dirPage();
      if (pg === "prefs") prefsPage();
      spacers();
    } finally { busy = false; }
  }
  var queued = false;
  function schedule() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(function () { queued = false; decorate(); });
  }

  // card titles navigate through Kopia's own router link when there is one
  document.addEventListener("click", function (e) {
    var a = e.target.closest && e.target.closest("a[data-p-nav]");
    if (!a || e.defaultPrevented || e.button || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    var twin = qa("#root a[href]").filter(function (x) { return !x.hasAttribute("data-p-nav") && x.getAttribute("href") === a.getAttribute("href"); })[0];
    if (twin) { e.preventDefault(); twin.click(); }
  });
  document.addEventListener("click", read);
  document.addEventListener("mouseover", read, { passive: true });

  function start() {
    new MutationObserver(schedule).observe(document.body, { childList: true, subtree: true, characterData: true });
    // refresh data (pages that need it) and relative times once a minute, and on return to the tab
    setInterval(function () { if (!document.hidden) decorate(); }, 60000);
    document.addEventListener("visibilitychange", function () { if (!document.hidden) decorate(); });
    decorate();
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(function () { navPage = null; decorate(); });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
