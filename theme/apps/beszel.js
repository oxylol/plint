/* ─────────────────────────────────────────────────────────────
   Plint Terminal · Beszel extras
   - the Plint strip goes into Beszel's header bar
   - an instrument panel ("NOW") on All Systems and on a system page:
     six dot-matrix figures with a last-hour tape each, read from the
     hub's own API (same origin, the token Beszel stores)
   - the current header link is marked so it can sit inverse
   - chart series are numbered per chart so the sheet can tell them
     apart by line and fill pattern (never hue); legend and tooltip
     swatches get the same number
   Everything is redrawn idempotently, because React re-renders.
   ───────────────────────────────────────────────────────────── */
(function () {
  "use strict";
  var P = window.Plint;
  if (!P) return;

  var HEADER = "#app > div > .container:first-child > div";
  P.mount = [HEADER];
  // Beszel can live under a sub-path (BESZEL.BASE_PATH); it is "/" here.
  var BASE = String((window.BESZEL && window.BESZEL.BASE_PATH) || "/").replace(/\/?$/, "/");

  function q(sel, root) { return (root || document).querySelector(sel); }
  function qa(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }
  // Compare with what we last wrote, not innerHTML (the browser re-serialises markup).
  function set(node, html) { if (node && node.__plintHtml !== html) { node.innerHTML = html; node.__plintHtml = html; } }
  function setAttr(node, k, v) { if (node.getAttribute(k) !== v) node.setAttribute(k, v); }
  function cls(node, name, on) { if (node.classList.contains(name) !== !!on) node.classList.toggle(name, !!on); }

  // ── API (PocketBase, same origin) ───────────────────────────
  function token() {
    try { return (JSON.parse(localStorage.getItem("pocketbase_auth") || "{}") || {}).token || ""; } catch (e) { return ""; }
  }
  function api(path) {
    var t = token();
    if (!t) return Promise.resolve(null);
    return fetch(BASE + path, { headers: { Authorization: t }, cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .catch(function () { return null; });
  }
  function pbq(s) { return "'" + String(s).replace(/\\/g, "\\\\").replace(/'/g, "\\'") + "'"; }

  var prefs = { warn: 65, crit: 90, fahrenheit: false };
  function loadPrefs() {
    return api("api/collections/user_settings/records?perPage=1").then(function (d) {
      var s = d && d.items && d.items[0] && d.items[0].settings;
      if (!s) return;
      if (s.colorWarn > 0) prefs.warn = Number(s.colorWarn);
      if (s.colorCrit > 0) prefs.crit = Number(s.colorCrit);
      prefs.fahrenheit = Number(s.unitTemp) === 3;
    });
  }

  function route() { return "/" + location.pathname.slice(BASE.length).replace(/^\/+/, ""); }
  function systemKey() {
    var m = route().match(/^\/system\/([^/]+)/);
    try { return m ? decodeURIComponent(m[1]) : ""; } catch (e) { return ""; }
  }
  function wanted() { return route() === "/" || !!systemKey(); }

  // ── data: the systems list, and one system's last hour ──────
  var st = { systems: null, loading: false, tried: 0, ticks: 0, hist: { id: "", list: [], full: false, busy: false } };
  function findSystem(key) {
    var l = st.systems || [], i;
    for (i = 0; i < l.length; i++) if (l[i].id === key || l[i].name === key) return l[i];
    return null;
  }
  // the system whose history the page shows: the open one, or the only one
  function target() {
    if (!st.systems) return null;
    var key = systemKey();
    if (key) return findSystem(key);
    return route() === "/" && st.systems.length === 1 ? st.systems[0] : null;
  }
  function stamp(s) { var t = Date.parse(String(s || "").replace(" ", "T")); return isNaN(t) ? 0 : t; }
  function loadHist(id) {
    var h = st.hist;
    if (h.busy) return;
    if (h.id !== id) { h.id = id; h.list = []; h.full = false; }
    h.busy = true;
    api("api/collections/system_stats/records?perPage=" + (h.full ? 3 : 60) + "&skipTotal=1&sort=-created&fields=created,stats&filter=" +
      encodeURIComponent("system=" + pbq(id) + " && type='1m'")).then(function (d) {
      h.busy = false;
      if (!d || !d.items || h.id !== id) return;
      h.full = true;
      var have = {};
      h.list.forEach(function (it) { have[it.t] = 1; });
      d.items.forEach(function (it) {
        var t = stamp(it.created);
        if (t && it.stats && !have[t]) { have[t] = 1; h.list.push({ t: t, s: it.stats }); }
      });
      h.list.sort(function (a, b) { return a.t - b.t; });
      if (h.list.length > 64) h.list = h.list.slice(h.list.length - 64);
      decorate();
    });
  }
  function load(withHist) {
    if (!wanted() || st.loading || !token()) return;
    st.loading = true;
    st.tried = Date.now();
    api("api/collections/systems/records?perPage=50&skipTotal=1&sort=name&fields=id,name,status,info,updated").then(function (d) {
      st.loading = false;
      if (d && d.items) st.systems = d.items;
      var t = target();
      if (t && t.status === "up" && (withHist || !st.hist.full || st.hist.id !== t.id)) loadHist(t.id);
      decorate();
    });
  }

  // ── formatting ──────────────────────────────────────────────
  function gb(v) { // Beszel stores memory and disk in GB
    if (v == null || isNaN(v)) return "–";
    if (v >= 1000) return (v / 1000).toFixed(v >= 10000 ? 0 : 1).replace(/\.0$/, "") + " TB";
    return (v >= 100 ? Math.round(v) : v >= 10 ? v.toFixed(0) : v.toFixed(1).replace(/\.0$/, "")) + " GB";
  }
  function rateParts(b) { // bytes per second → [figure, unit]
    if (b == null || isNaN(b)) return ["–", ""];
    var u = ["B/s", "KB/s", "MB/s", "GB/s"], i = 0;
    while (b >= 1000 && i < u.length - 1) { b /= 1024; i++; }
    return [b >= 100 || i === 0 ? String(Math.round(b)) : b.toFixed(1), u[i]];
  }
  function rateShort(b) { var r = rateParts(b); return r[0] + r[1].charAt(0); }
  function pair(used, total) { // "4.9 / 7.7 GB"
    var a = gb(used), b = gb(total);
    return a.slice(-2) === b.slice(-2) ? a.slice(0, -3) + " / " + b : a + " / " + b;
  }
  function pct(v) { v = Number(v) || 0; return v > 0 && v < 10 ? v.toFixed(1) : String(Math.round(v)); }
  function tone(p) { return p >= prefs.crit ? "crit" : p >= prefs.warn ? "warn" : ""; }
  function clock(t) {
    if (!t) return "";
    var d = new Date(t), h = d.getHours(), m = d.getMinutes();
    return (h < 10 ? "0" : "") + h + ":" + (m < 10 ? "0" : "") + m;
  }

  // what a system says about itself right now
  function reading(rec) {
    var i = (rec && rec.info) || {};
    return {
      cpu: Number(i.cpu) || 0, mem: Number(i.mp) || 0, disk: Number(i.dp) || 0,
      la: i.la && i.la.length ? i.la : (i.l1 != null ? [i.l1, i.l5, i.l15] : null),
      temp: i.dt ? Number(i.dt) : null,
      net: i.bb != null ? Number(i.bb) : (i.b != null ? Number(i.b) * 1048576 : null),
      threads: Number(i.t) || 0
    };
  }
  // the same six series out of one stored sample
  var SER = {
    cpu: function (s) { return s.cpu; },
    mem: function (s) { return s.mp; },
    disk: function (s) { return s.dp; },
    load: function (s) { return s.la && s.la.length ? s.la[0] : s.l1; },
    temp: function (s) {
      var m = null, k;
      if (s.t && typeof s.t === "object") for (k in s.t) if (typeof s.t[k] === "number" && (m == null || s.t[k] > m)) m = s.t[k];
      return m;
    },
    net: function (s) {
      if (s.b && s.b.length === 2) return Number(s.b[0]) + Number(s.b[1]);
      return s.ns != null || s.nr != null ? ((Number(s.ns) || 0) + (Number(s.nr) || 0)) * 1048576 : null;
    }
  };
  var tapeOf = ""; // the system the figures being drawn belong to
  // last hour as 20 columns of 3 minutes, six cells high; scale 0 = the series' own peak
  function tape(key, scale) {
    var list = st.hist.list, n = list.length;
    if (!n || st.hist.id !== tapeOf) return "";
    var now = Date.now(), end = list[n - 1].t, b = [], any = false, j, max = scale || 0;
    if (Math.abs(now - end) > 300000) end = now;
    for (j = 0; j < 20; j++) b.push(null);
    list.forEach(function (it) {
      var idx = 19 - Math.floor((end - it.t) / 180000), v = SER[key](it.s);
      if (idx < 0 || idx > 19 || v == null || isNaN(v)) return;
      v = Number(v);
      if (b[idx] == null || v > b[idx]) b[idx] = v;
      any = true;
    });
    if (!any) return "";
    if (!scale) b.forEach(function (v) { if (v > max) max = v; });
    if (!(max > 0)) max = 1;
    return '<div class="p-bz-tape" aria-hidden="true">' + b.map(function (v) {
      if (v == null) return '<i class="z" style="height:4px"></i>';
      return '<i style="height:' + Math.max(1, Math.min(6, Math.round(v / max * 6))) * 4 + 'px"></i>';
    }).join("") + "</div>";
  }
  function latest() { var l = st.hist.list; return l.length ? l[l.length - 1].s || {} : {}; }

  // ── the panel ───────────────────────────────────────────────
  // model: { title, note, figs: [{ k, label, value, unit, tone, lamp, graph, sub }], rd }
  function render(panel, m) {
    var sig = m.figs.map(function (f) { return f.k; }).join(",");
    if (panel.__sig !== sig) {
      panel.__sig = sig;
      panel.innerHTML = '<header class="p-bz-h"><span></span><i></i></header><div class="p-bz-figs">' +
        m.figs.map(function (f) {
          return '<div class="p-bz-fig p-fl" data-f="' + f.k + '"><strong></strong><span class="p-bz-l"></span><div class="p-bz-g"></div><small></small></div>';
        }).join("") + '</div><p class="p-bz-rd p-fl"></p>';
      qa(".p-fl", panel).forEach(function (n, i) {
        try { P.strike(n, i * 60); } catch (e) { n.classList.add("on"); }
      });
    }
    set(q(".p-bz-h span", panel), m.title);
    set(q(".p-bz-h i", panel), P.esc(m.note || ""));
    m.figs.forEach(function (f) {
      var n = q('[data-f="' + f.k + '"]', panel);
      if (!n) return;
      cls(n, "warn", f.tone === "warn");
      cls(n, "crit", f.tone === "crit");
      try { P.num(n.children[0], f.value, f.unit || ""); } catch (e) { n.children[0].textContent = f.value + " " + (f.unit || ""); }
      set(n.children[1], (f.lamp != null || f.tone === "warn" ? P.lamp(f.lamp == null ? true : f.lamp) : "") + "<b>" + P.esc(f.label) + "</b>");
      set(n.children[2], f.graph || "");
      set(n.children[3], P.esc(f.sub || ""));
    });
    set(q(".p-bz-rd", panel), m.rd || "");
  }
  function place(id, parent, extra) {
    var n = document.getElementById(id);
    if (!parent) { if (n) n.remove(); return null; }
    if (!n) {
      n = P.el("section", "p-bz-panel" + (extra ? " " + extra : ""));
      n.id = id;
      n.setAttribute("aria-label", "Current readings");
    }
    if (n.parentNode !== parent || parent.lastElementChild !== n) parent.appendChild(n);
    return n;
  }
  function tempFig(c, graph, sub) {
    if (c == null) return { k: "temp", label: "Temp", value: "–", unit: "", graph: "", sub: "No sensor" };
    return {
      k: "temp", label: "Temp", value: String(Math.round(prefs.fahrenheit ? c * 1.8 + 32 : c)), unit: prefs.fahrenheit ? "°F" : "°C",
      tone: c >= 90 ? "crit" : c >= 80 ? "warn" : "", graph: graph, sub: sub
    };
  }
  function netFig(b, graph, sub) {
    var r = rateParts(b);
    return { k: "net", label: "Net", value: r[0], unit: r[1], graph: graph, sub: sub };
  }
  function overLine(figs) {
    var crit = [], warn = [];
    figs.forEach(function (f) {
      var t = P.esc(f.label + " " + f.value + (f.unit || ""));
      if (f.tone === "crit") crit.push(t); else if (f.tone === "warn") warn.push(t);
    });
    if (!crit.length && !warn.length) return "All readings under the warning line <b>" + prefs.warn + "%</b>";
    return (crit.length ? '<b class="p-inv">Critical</b> <b>' + crit.join(" · ") + "</b>" : "") +
      (crit.length && warn.length ? " · " : "") + (warn.length ? "Warning <b>" + warn.join(" · ") + "</b>" : "");
  }
  // five figures for one system; `s` is its newest stored sample
  function oneSystem(rec, s) {
    tapeOf = rec.id;
    var r = reading(rec), net = s.b && s.b.length === 2 ? "Dn " + rateShort(s.b[1]) + " Up " + rateShort(s.b[0]) : "Down + up";
    return {
      cpu: { k: "cpu", label: "CPU", value: pct(r.cpu), unit: "%", tone: tone(r.cpu), graph: tape("cpu", 100) || P.meter(r.cpu), sub: r.threads ? r.threads + " threads" : "Last hour" },
      mem: { k: "mem", label: "Memory", value: pct(r.mem), unit: "%", tone: tone(r.mem), graph: tape("mem", 100) || P.meter(r.mem), sub: s.m ? pair(s.mu, s.m) : "" },
      disk: { k: "disk", label: "Disk", value: pct(r.disk), unit: "%", tone: tone(r.disk), graph: tape("disk", 100) || P.meter(r.disk), sub: s.d ? pair(s.du, s.d) : "" },
      temp: tempFig(r.temp, tape("temp", 100), "Hottest sensor"),
      net: netFig(r.net, tape("net", 0), net),
      r: r
    };
  }

  // All Systems: units up, then the highest reading among them
  function homePanel() {
    var bar = q(HEADER), ok = route() === "/" && bar && st.systems && st.systems.length && token();
    var panel = place("p-bz-home", ok ? bar.parentNode : null);
    if (!panel) return;
    var all = st.systems, up = all.filter(function (x) { return x.status === "up"; });
    var down = all.filter(function (x) { return x.status === "down"; });
    var figs = [{
      k: "units", label: "Units up", value: String(up.length), unit: "of " + all.length,
      tone: down.length ? "crit" : "", lamp: down.length ? null : up.length === all.length ? true : false,
      graph: P.meter(all.length ? up.length / all.length * 100 : 0),
      sub: down.length ? "Down: " + down.map(function (x) { return x.name; }).join(", ") : up.length === all.length ? "All reporting" : (all.length - up.length) + " paused or pending"
    }];
    if (all.length === 1 && up.length === 1) {
      var o = oneSystem(all[0], st.hist.id === all[0].id ? latest() : {});
      figs.push(o.cpu, o.mem, o.disk, o.temp, o.net);
    } else {
      var top = function (key) {
        var best = null, bv = -1;
        up.forEach(function (x) { var v = reading(x)[key]; if (v != null && v > bv) { bv = v; best = x; } });
        return best ? { v: bv, name: best.name } : { v: null, name: "" };
      };
      var c = top("cpu"), m = top("mem"), d = top("disk"), t = top("temp"), net = 0;
      up.forEach(function (x) { net += reading(x).net || 0; });
      figs.push(
        { k: "cpu", label: "CPU peak", value: c.v == null ? "–" : pct(c.v), unit: c.v == null ? "" : "%", tone: tone(c.v || 0), graph: P.meter(c.v || 0), sub: c.name },
        { k: "mem", label: "Memory peak", value: m.v == null ? "–" : pct(m.v), unit: m.v == null ? "" : "%", tone: tone(m.v || 0), graph: P.meter(m.v || 0), sub: m.name },
        { k: "disk", label: "Disk peak", value: d.v == null ? "–" : pct(d.v), unit: d.v == null ? "" : "%", tone: tone(d.v || 0), graph: P.meter(d.v || 0), sub: d.name },
        tempFig(t.v, t.v == null ? "" : P.meter(t.v), t.name),
        netFig(up.length ? net : null, "", "All units")
      );
    }
    var newest = 0;
    all.forEach(function (x) { var u = stamp(x.updated); if (u > newest) newest = u; });
    render(panel, {
      title: 'Now <span class="p-ru">/ Сейчас</span>', note: newest ? "Read " + clock(newest) : "", figs: figs,
      rd: down.length ? '<b class="p-inv">Down</b> <b>' + P.esc(down.map(function (x) { return x.name; }).join(", ")) + "</b>" : overLine(figs)
    });
  }

  // a system page: the six readings of that system, under its title
  function systemPanel() {
    var key = systemKey(), rec = key && findSystem(key);
    var card = key && q("#app .grid.overflow-x-clip > .rounded-lg:first-child");
    var panel = place("p-bz-sys", rec && card && rec.status === "up" && rec.info ? card : null, "p-bz-stats");
    if (!panel) return;
    var s = st.hist.id === rec.id ? latest() : {}, o = oneSystem(rec, s), r = o.r, figs = [o.cpu, o.mem, o.disk];
    if (r.la) {
      var l1 = Number(r.la[0]) || 0;
      figs.push({
        k: "load", label: "Load", value: l1.toFixed(2), unit: "", tone: r.threads ? tone(l1 / r.threads * 100) : "",
        graph: tape("load", 0), sub: "5/15m " + (Number(r.la[1]) || 0).toFixed(2) + " " + (Number(r.la[2]) || 0).toFixed(2)
      });
    } else figs.push({ k: "load", label: "Load", value: "–", unit: "", graph: "", sub: "" });
    figs.push(o.temp, o.net);
    var n = st.hist.list.length, at = st.hist.id === rec.id && n ? st.hist.list[n - 1].t : stamp(rec.updated);
    render(panel, { title: 'Now <span class="p-ru">/ Сейчас</span>', note: at ? "Read " + clock(at) + " · tape 1 h" : "", figs: figs, rd: overLine(figs) });
  }

  // ── the strip: in the header bar, or floating where there is none ──
  function pill(bar) {
    var p = document.getElementById("plint-pill");
    if (!p) return;
    if (bar) {
      if (p.parentNode !== bar || bar.firstChild !== p) { p.classList.remove("plint-float"); bar.insertBefore(p, bar.firstChild); }
    } else if (!p.isConnected || p.closest("#plint-menu") || p.closest("#app")) {
      p.classList.add("plint-float");
      if (p.parentNode !== document.body) document.body.appendChild(p);
    }
  }

  // ── header: mark the link for the current page ──────────────
  function header() {
    var bar = q(HEADER);
    pill(bar);
    if (!bar) return;
    var here = route().replace(/\/+$/, "") || "/";
    qa("a[href]", bar).forEach(function (a) {
      if (a.getAttribute("aria-label") === "Home" || a.closest("#plint-pill")) return;
      var p = ("/" + a.pathname.slice(BASE.length)).replace(/\/+$/, "");
      var on = !!p && p !== "/" && (here === p || here.indexOf(p.replace(/\/general$/, "") + "/") === 0);
      cls(a, "p-on", on);
      if (on) setAttr(a, "aria-current", "page"); else if (a.hasAttribute("aria-current")) a.removeAttribute("aria-current");
    });
  }

  // ── charts: number the series, so the sheet can pattern them ──
  // 0 solid · 1 dashed · 2 dotted · 3 dim · 4 dash-dot · 5–7 the dim versions.
  // Area fills: 0 and 3 are flat (the sheet), the others are hatch patterns,
  // which need an in-document <pattern> and an inline url().
  var FILL = { 1: 1, 2: 1, 4: 1, 5: 1, 6: 1, 7: 1 };
  function defs() {
    if (document.getElementById("p-bz-defs")) return;
    var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("id", "p-bz-defs");
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("width", "0");
    svg.setAttribute("height", "0");
    var pat = function (n, size, rot, rects) {
      return '<pattern id="p-bz-f' + n + '" width="' + size + '" height="' + size + '" patternUnits="userSpaceOnUse"' +
        (rot ? ' patternTransform="rotate(' + rot + ')"' : "") + ">" + rects + "</pattern>";
    };
    svg.innerHTML = "<defs>" +
      pat(1, 6, 45, '<rect width="1.2" height="6"/>') +
      pat(2, 5, 0, '<rect width="1.5" height="1.5"/>') +
      pat(4, 4, 0, '<rect width="1" height="4"/>') +
      pat(5, 6, -45, '<rect width="1.2" height="6"/>') +
      pat(6, 4, 0, '<rect width="4" height="1"/>') +
      pat(7, 6, 45, '<rect width="1" height="6"/><rect width="6" height="1"/>') + "</defs>";
    document.body.appendChild(svg);
  }
  function charts() {
    var list = qa("[data-chart]");
    if (!list.length) return;
    defs();
    list.forEach(function (chart) {
      var idx = {}, n = 0;
      qa("path.recharts-curve[name]", chart).forEach(function (p) {
        var k = "$" + p.getAttribute("name");
        if (idx[k] == null) idx[k] = n++;
        var s = String(idx[k] % 8);
        setAttr(p, "data-p-s", s);
        if (p.classList.contains("recharts-area-area")) {
          var want = FILL[s] ? "url(#p-bz-f" + s + ")" : "";
          if (p.__plintFill !== want) {
            p.__plintFill = want;
            p.style.fill = want;
            p.style.fillOpacity = want ? "1" : "";
          }
        }
      });
      if (!n) return;
      qa(".recharts-legend-wrapper .h-2.w-2", chart).forEach(function (sw) {
        var k = "$" + (sw.parentElement ? sw.parentElement.textContent : "").trim();
        if (idx[k] != null) setAttr(sw, "data-p-s", String(idx[k] % 8)); else if (sw.hasAttribute("data-p-s")) sw.removeAttribute("data-p-s");
      });
      qa('.recharts-tooltip-wrapper [style*="--color-bg"]', chart).forEach(function (sw) {
        var row = sw.parentElement, name = row && q("span", row);
        var k = "$" + (name ? name.textContent : "").trim();
        if (idx[k] != null) setAttr(sw, "data-p-s", String(idx[k] % 8)); else if (sw.hasAttribute("data-p-s")) sw.removeAttribute("data-p-s");
      });
    });
  }

  // ── run ─────────────────────────────────────────────────────
  var busy = false, lastPath = null;
  function decorate() {
    if (busy) return;
    busy = true;
    try {
      if (location.pathname !== lastPath) {
        lastPath = location.pathname;
        load(true);
      } else if (!st.systems && wanted() && Date.now() - st.tried > 5000) {
        load(true); // just logged in: the path did not change
      }
      header();
      homePanel();
      systemPanel();
      charts();
    } catch (e) { /* the app's DOM moved: leave it alone */ } finally { busy = false; }
  }
  var queued = false;
  function schedule() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(function () { queued = false; decorate(); });
  }
  var OWN = ".p-bz-panel, #plint-pill, #plint-menu, #p-bz-defs";
  function seen(records) {
    for (var i = 0; i < records.length; i++) {
      var t = records[i].target;
      if (t && t.nodeType === 1 && !t.closest(OWN)) { schedule(); return; }
    }
  }
  function start() {
    var mo = new MutationObserver(seen), opts = { childList: true, subtree: true };
    mo.observe(document.body, opts);
    document.addEventListener("visibilitychange", function () {
      if (document.hidden) mo.disconnect();
      else { mo.observe(document.body, opts); load(true); schedule(); }
    });
    loadPrefs().then(decorate);
    // one slow tick: readings every 30 s, the tape every minute, nothing while hidden
    setInterval(function () { if (!document.hidden) { st.ticks++; load(st.ticks % 2 === 0); } }, 30000);
    decorate();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
