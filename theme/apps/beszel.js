/* ─────────────────────────────────────────────────────────────
   Plint Soft UI · Beszel extras
   - the Plint pill goes into Beszel's header bar
   - system page: CPU / memory / disk / swap as rings, plus a few facts,
     from the hub's own API (same origin, the token Beszel stores)
   - the current header link is marked so it can sit pressed
   - chart series that Beszel spreads around the hue wheel (containers,
     cores, sensors, load) are recoloured onto the Plint palette
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
  // Compare with what we last wrote, not innerHTML: the browser re-serialises
  // markup (e.g. <circle/> → <circle></circle>), which would redraw every frame.
  function set(node, html) { if (node && node.__plintHtml !== html) { node.innerHTML = html; node.__plintHtml = html; } }
  function setAttr(node, k, v) { if (node.getAttribute(k) !== v) node.setAttribute(k, v); }

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

  var prefs = { warn: 65, crit: 90, fahrenheit: false, loaded: false };
  function loadPrefs() {
    return api("api/collections/user_settings/records?perPage=1").then(function (d) {
      var s = d && d.items && d.items[0] && d.items[0].settings;
      if (s) {
        if (s.colorWarn > 0) prefs.warn = Number(s.colorWarn);
        if (s.colorCrit > 0) prefs.crit = Number(s.colorCrit);
        prefs.fahrenheit = Number(s.unitTemp) === 3;
      }
      prefs.loaded = true;
    });
  }

  var sys = { key: "", rec: null, stats: null, inflight: "" };
  function route() { return "/" + location.pathname.slice(BASE.length).replace(/^\/+/, ""); }
  function systemKey() {
    var m = route().match(/^\/system\/([^/]+)/);
    try { return m ? decodeURIComponent(m[1]) : ""; } catch (e) { return ""; }
  }
  function loadSystem() {
    var key = systemKey();
    if (!key || sys.inflight === key) return;
    if (key !== sys.key) { sys.key = key; sys.rec = null; sys.stats = null; }
    sys.inflight = key;
    api("api/collections/systems/records?perPage=1&filter=" + encodeURIComponent("id=" + pbq(key) + " || name=" + pbq(key)))
      .then(function (d) {
        var rec = d && d.items && d.items[0];
        if (!rec || sys.key !== key) return null;
        sys.rec = rec;
        return api("api/collections/system_stats/records?perPage=1&sort=-created&fields=stats,created&filter=" +
          encodeURIComponent("system=" + pbq(rec.id) + " && type='1m'"));
      })
      .then(function (d) {
        var it = d && d.items && d.items[0];
        if (it && sys.key === key) sys.stats = it.stats || null;
      })
      .then(function () { if (sys.inflight === key) sys.inflight = ""; decorate(); },
            function () { if (sys.inflight === key) sys.inflight = ""; });
  }

  // ── formatting ──────────────────────────────────────────────
  function gb(v) { // Beszel stores memory and disk in GB
    if (v == null || isNaN(v)) return "–";
    if (v >= 1000) return (v / 1000).toFixed(v >= 10000 ? 0 : 1).replace(/\.0$/, "") + " TB";
    return (v >= 100 ? Math.round(v) : v >= 10 ? v.toFixed(0) : v.toFixed(1).replace(/\.0$/, "")) + " GB";
  }
  function rate(b) { // bytes per second
    if (b == null || isNaN(b)) return "–";
    var u = ["B/s", "KB/s", "MB/s", "GB/s"], i = 0;
    while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
    return (b >= 100 || i === 0 ? Math.round(b) : b.toFixed(1)) + " " + u[i];
  }
  function tone(pct) { return pct >= prefs.crit ? "bad" : pct >= prefs.warn ? "warn" : ""; }
  function pctOf(used, total) { return total > 0 ? Math.max(0, Math.min(100, used / total * 100)) : 0; }

  // ── system page: rings + facts in the header card ───────────
  function gauge(label, pct, sub) {
    var t = tone(pct), n = Math.round(pct);
    return '<div class="p-bz-gauge"><div class="p-bz-dial">' +
      P.ring(pct, { size: 108, width: 11, tone: t, label: label + " " + n + " percent" }) +
      "<b>" + n + '%</b></div><div><strong>' + P.esc(label) + "</strong>" +
      (sub ? "<small>" + P.esc(sub) + "</small>" : "") + "</div></div>";
  }
  function systemStats() {
    var key = systemKey();
    var card = q("#app .grid.overflow-x-clip > .rounded-lg:first-child");
    var box = card && q(":scope > .p-bz-stats", card);
    if (!key || !card || !sys.rec || sys.key !== key || !sys.rec.info) {
      if (box) box.remove();
      return;
    }
    var i = sys.rec.info, s = sys.stats || {};
    if (sys.rec.status !== "up") { if (box) box.remove(); return; }
    var memPct = i.mp != null ? i.mp : pctOf(s.mu, s.m);
    var diskPct = i.dp != null ? i.dp : pctOf(s.du, s.d);
    var rings =
      gauge("CPU", i.cpu || 0, i.t ? i.t + " threads" : "") +
      gauge("Memory", memPct, s.m ? gb(s.mu) + " of " + gb(s.m) : "") +
      gauge("Disk", diskPct, s.d ? gb(s.du) + " of " + gb(s.d) : "") +
      (s.s > 0 ? gauge("Swap", pctOf(s.su, s.s), gb(s.su) + " of " + gb(s.s)) : "");
    var facts = "";
    if (i.la && i.la.length) {
      var cores = i.t || 0, hi = Math.max.apply(null, i.la);
      var lt = cores ? tone(hi / cores * 100) : "";
      facts += '<div class="p-bz-fact"><span>Load</span><b' + (lt ? ' class="' + lt + '"' : "") + ">" +
        i.la.map(function (x) { return Number(x).toFixed(2); }).join(" · ") + "</b></div>";
    }
    if (i.dt) {
      var c = Number(i.dt), shown = prefs.fahrenheit ? Math.round(c * 1.8 + 32) + " °F" : Math.round(c) + " °C";
      var tt = c >= 90 ? "bad" : c >= 80 ? "warn" : "";
      facts += '<div class="p-bz-fact"><span>Hottest sensor</span><b' + (tt ? ' class="' + tt + '"' : "") + ">" + shown + "</b></div>";
    }
    if (s.b && s.b.length === 2) {
      facts += '<div class="p-bz-fact"><span>Network</span><b>↓ ' + rate(s.b[1]) + " · ↑ " + rate(s.b[0]) + "</b></div>";
    } else if (i.bb != null) {
      facts += '<div class="p-bz-fact"><span>Network</span><b>' + rate(i.bb) + "</b></div>";
    }
    var html = '<div class="p-bz-rings">' + rings + "</div>" + (facts ? '<div class="p-bz-facts">' + facts + "</div>" : "");
    if (!box) {
      box = P.el("section", "p-bz-stats");
      box.setAttribute("aria-label", "Current usage");
      card.appendChild(box);
    } else if (box !== card.lastElementChild) {
      card.appendChild(box);
    }
    set(box, html);
  }

  // ── the pill: in the header bar, or floating where there is none ──
  // (plint.js falls back to a bare "nav", which can be its own hidden
  // #plint-menu on pages without a header, like the login page)
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
      if (on !== a.classList.contains("p-on")) a.classList.toggle("p-on", on);
      if (on) setAttr(a, "aria-current", "page"); else if (a.hasAttribute("aria-current")) a.removeAttribute("aria-current");
    });
  }

  // ── charts: hue-spread series onto the Plint palette ────────
  // Known fixed colours first; anything else on the hue wheel is placed
  // along sage → slate → amber → lichen by its hue.
  var FIXED = {
    "hsl(271, 81%, 60%)": "var(--p-accent)",      // load 1 min
    "hsl(217, 91%, 60%)": "var(--p-info)",        // load 5 min
    "hsl(25, 95%, 53%)": "var(--p-warn)",         // load 15 min
    "hsl(80, 65%, 52%)": "var(--p-ink2)",         // CPU: other
    "hsla(160 60% 45% / 0.5)": "color-mix(in oklab, var(--p-info) 55%, transparent)", // memory: cache / buffers
    "hsla(175 60% 45% / 0.8)": "var(--p-ink2)"    // memory: ZFS ARC
  };
  var STOPS = ["--p-accent", "--p-info", "--p-warn", "--p-ink2"];
  var HSL = /^hsla?\(\s*(-?[\d.]+)/;
  function remap(c) {
    if (!c) return null;
    c = String(c).trim();
    if (FIXED[c]) return FIXED[c];
    var m = c.match(HSL);
    if (!m) return null;
    var h = ((parseFloat(m[1]) % 360) + 360) % 360;
    var t = h / 360 * STOPS.length, k = Math.floor(t), f = t - k;
    var a = STOPS[k % STOPS.length], b = STOPS[(k + 1) % STOPS.length];
    var w = Math.round((1 - f) * 100);
    if (w >= 99) return "var(" + a + ")";
    if (w <= 1) return "var(" + b + ")";
    return "color-mix(in oklab, var(" + a + ") " + w + "%, var(" + b + "))";
  }
  // Swatches get the same colours as inline styles, which the browser
  // re-serialises as rgb(); remember each colour's rgb() form to match them.
  var probe = document.createElement("i"), seen = {};
  function rgb(c) { probe.style.color = ""; probe.style.color = c; return probe.style.color; }
  Object.keys(FIXED).forEach(function (k) { var n = rgb(k); if (n) seen[n] = FIXED[k]; });
  var SWATCH_PROPS = ["--color-bg", "--color-border", "background-color", "border-color", "color"];
  // CPU breakdown series that beszel.css recolours by name (Beszel keeps
  // these labels untranslated); their swatches follow by label.
  var NAMED = { IOWait: "var(--p-warn)", Steal: "var(--p-bad)", Idle: "var(--p-ink2)" };
  function named(n) {
    var t = n.parentElement ? n.parentElement.textContent.trim() : "";
    for (var k in NAMED) if (t.indexOf(k) === 0 && !/^[A-Za-z]/.test(t.charAt(k.length))) return NAMED[k];
    return null;
  }
  function charts() {
    qa('[data-chart] [fill^="hsl"], [data-chart] [stroke^="hsl"]').forEach(function (n) {
      ["fill", "stroke"].forEach(function (k) {
        var v = n.getAttribute(k), r = v && v.indexOf("hsl") === 0 && remap(v);
        if (!r) return;
        if (v.indexOf("var(") < 0) { var c = rgb(v); if (c) seen[c] = r; }
        n.setAttribute(k, r);
      });
    });
    qa('.recharts-legend-wrapper [style], .recharts-tooltip-wrapper [style]').forEach(function (n) {
      SWATCH_PROPS.forEach(function (k) {
        var v = n.style.getPropertyValue(k);
        if (!v) return;
        var r = named(n) || (v.indexOf("hsl") >= 0 ? remap(v) : seen[v]);
        if (r && r !== v) n.style.setProperty(k, r);
      });
    });
  }

  // ── run ─────────────────────────────────────────────────────
  var busy = false, lastPath = "";
  function decorate() {
    if (busy) return;
    busy = true;
    try {
      if (location.pathname !== lastPath) {
        lastPath = location.pathname;
        if (systemKey()) loadSystem();
      }
      header();
      systemStats();
      charts();
    } finally { busy = false; }
  }
  var queued = false;
  function schedule() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(function () { queued = false; decorate(); });
  }
  function start() {
    new MutationObserver(schedule).observe(document.body, {
      childList: true, subtree: true, attributes: true, attributeFilter: ["fill", "stroke", "style"]
    });
    loadPrefs().then(decorate);
    setInterval(function () { if (systemKey() && !document.hidden) loadSystem(); }, 30000);
    decorate();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
