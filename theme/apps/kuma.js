/* Plint Terminal · Uptime Kuma extras
   - the strip sits in Kuma's header (or on top of a public status page)
   - Quick Stats gets a tape (one cell per monitor) and a one-line readout
   - a monitor's uptime figures become dot-matrix figures with a meter
   - the heartbeat canvases are drawn as cells: up lit, down hollow,
     pending hatched, maintenance dim hatch (Kuma paints them from JS)
   - the response-time chart (Chart.js) is repainted in one light
   Everything is best effort: when Kuma's markup or internals move, the
   sheet alone still carries the look. */
(function () {
  "use strict";
  var P = window.Plint;
  if (!P) return;
  P.mount = ["#app header", "#app .main"];

  // Kuma writes its theme colour into <meta id="theme-color"> whenever its
  // theme changes; the strip owns the real meta, so Kuma gets an inert one
  // (no name) to write to instead of throwing.
  function meta() {
    if (document.getElementById("theme-color") || !document.head) return;
    var m = document.createElement("meta");
    m.id = "theme-color";
    document.head.appendChild(m);
  }
  try { meta(); } catch (e) { /* skip */ }

  function q(s, r) { try { return (r || document).querySelector(s); } catch (e) { return null; } }
  function qa(s, r) { try { return Array.prototype.slice.call((r || document).querySelectorAll(s)); } catch (e) { return []; } }
  function num(t) { var m = String(t == null ? "" : t).replace(",", ".").match(/-?\d+(\.\d+)?/); return m ? parseFloat(m[0]) : NaN; }
  function flag(node, name, on) {
    if (!node) return;
    if (on) { if (!node.hasAttribute(name)) node.setAttribute(name, ""); }
    else if (node.hasAttribute(name)) node.removeAttribute(name);
  }
  var PH = "228, 248, 234";
  try { PH = getComputedStyle(document.documentElement).getPropertyValue("--p-ph").trim() || PH; } catch (e) { /* default */ }
  function ink(a) { return "rgba(" + PH + "," + (a == null ? 1 : a) + ")"; }
  var FONT = "'Geist Mono', ui-monospace, monospace";

  // ── Quick Stats: a tape of all monitors and a readout ────────
  function quick() {
    var cols = qa("#app .shadow-box > .row > .col").filter(function (c) { return q(":scope > h3 + .num", c); });
    if (cols.length < 2) return;
    var box = cols[0].parentNode.parentNode;
    var v = cols.map(function (c) { var n = num(q(".num", c).textContent); return n > 0 ? Math.round(n) : 0; });
    cols.forEach(function (c, i) { flag(c, "data-p-zero", !v[i]); flag(c, "data-p-alarm", i === 1 && v[i] > 0); });
    var host = q(":scope > .p-qs", box), first = !host;
    if (!host) { host = P.el("div", "p-qs"); box.appendChild(host); }
    // pending has no column of its own; Kuma keeps the count on its root
    var pend = 0;
    try { pend = Math.round(document.getElementById("app")._vnode.component.proxy.stats.pending) || 0; } catch (e) { pend = 0; }
    if (pend < 0 || pend > 120) pend = 0;
    var key = v.join(",") + "/" + pend;
    if (host.getAttribute("data-k") === key) return;
    host.setAttribute("data-k", key);
    // Kuma's order: up, down, maintenance, unknown, paused
    var up = v[0], down = v[1] || 0, total = v.reduce(function (a, b) { return a + b; }, 0) + pend;
    var kinds = ["", "dn", "mt", "off", "off"], cells = "";
    v.forEach(function (n, i) {
      for (var k = 0; k < n && k < 120; k++) cells += '<i class="' + (kinds[i] == null ? "off" : kinds[i]) + '"></i>';
      if (i === 1) for (k = 0; k < pend; k++) cells += '<i class="pd"></i>';
    });
    var rd = up + " of " + total + " up" + (down ? ' · <b>' + down + " down</b>" : total && up === total ? " · all checks answer" : "");
    if (pend) rd += " · " + pend + " pending";
    if (v[2]) rd += " · " + v[2] + " in maintenance";
    if (v[4]) rd += " · " + v[4] + " paused";
    host.innerHTML = '<div class="p-grp"><span>Monitors <span class="p-ru">/ МОНИТОРЫ</span></span><span>' + total + "</span></div>" +
      '<div class="p-tape">' + cells + '</div><div class="p-rd">' + rd + "</div>";
    if (first && P.strike) { host.classList.add("p-fl"); P.strike(host, 60); }
  }

  // ── one monitor: figures and meters ──────────────────────────
  function stats() {
    qa("#app main .shadow-box.stats > .row > div").forEach(function (tile) {
      var n = q(":scope > .num", tile);
      if (!n) return;
      var txt = (n.textContent || "").trim(), link = q("a", n);
      var fig = q(":scope > .p-fig", tile), meter = q(":scope > .p-meter", tile);
      var pct = /%\s*$/.test(txt) ? num(txt) : NaN;
      if (!isNaN(pct) && !link) {
        // an uptime figure: digits in the matrix, the unit small
        if (!fig) { fig = P.el("span", "p-fig"); tile.insertBefore(fig, n.nextSibling); }
        P.num(fig, txt.replace(/\s*%\s*$/, ""), "%");
        flag(tile, "data-p-fig", true);
        flag(tile, "data-p-low", pct < 99);
        if (!meter) { meter = P.el("div", "p-meter"); tile.appendChild(meter); }
        meter.style.setProperty("--v", Math.max(0, Math.min(100, pct)) + "%");
        return;
      }
      if (fig) { fig.parentNode.removeChild(fig); flag(tile, "data-p-fig", false); flag(tile, "data-p-low", false); }
      var days = link && /\d/.test(txt) && !/ms\s*$/i.test(txt) && q(":scope > p", tile) && /\d{4}-\d\d-\d\d/.test(q(":scope > p", tile).textContent) ? num(txt) : NaN;
      if (!isNaN(days)) {
        // the certificate: days left, of ninety
        if (!meter) { meter = P.el("div", "p-meter"); tile.appendChild(meter); }
        meter.style.setProperty("--v", Math.max(0, Math.min(100, days / 90 * 100)) + "%");
        flag(tile, "data-p-low", days <= 7);
        flag(tile, "data-p-soon", days > 7 && days <= 21);
      } else if (meter) { meter.parentNode.removeChild(meter); }
    });
  }

  // ── uptime badges: inverse when the figure has slipped ───────
  function badges() {
    qa("#app .monitor-list .badge").forEach(function (b) {
      var t = b.textContent || "";
      flag(b, "data-p-low", /%/.test(t) && num(t) < 99);
    });
  }

  // ── the stock Kuma logo becomes the Plint mark ───────────────
  function logo() {
    qa("#app .main > h1 img.logo, #app a.item > img.logo").forEach(function (img) {
      var stock = /\/icon\.svg(\?|$)/.test(img.getAttribute("src") || "");
      var mark = img.previousElementSibling && img.previousElementSibling.classList.contains("p-logo") ? img.previousElementSibling : null;
      if (stock && !mark) img.insertAdjacentHTML("beforebegin", P.mark("p-logo"));
      if (!stock && mark) mark.parentNode.removeChild(mark);
      flag(img, "data-p-swapped", stock);
    });
  }

  // ── heartbeat canvases ───────────────────────────────────────
  function cell(x, x0, y0, w, h, c) {
    x.beginPath();
    x.moveTo(x0 + c, y0); x.lineTo(x0 + w, y0); x.lineTo(x0 + w, y0 + h - c);
    x.lineTo(x0 + w - c, y0 + h); x.lineTo(x0, y0 + h); x.lineTo(x0, y0 + c);
    x.closePath();
  }
  function beatDraw(c, orig) {
    return function () {
      try {
        var vm = c.proxy, cv = vm.$refs && vm.$refs.canvas;
        if (!cv) return;
        var list = vm.shortBeatList, W = vm.canvasWidth, H = vm.canvasHeight, bw = vm.beatWidth, bh = vm.beatHeight, pad = vm.beatHoverAreaPadding;
        if (!Array.isArray(list) || !(bw > 0) || !(bh > 0) || !(pad >= 0) || !(H > 0) || !(W >= 0)) return orig.apply(this, arguments);
        var x = cv.getContext("2d"), dpr = window.devicePixelRatio || 1;
        cv.width = W * dpr; cv.height = H * dpr; cv.style.width = W + "px"; cv.style.height = H + "px";
        x.scale(dpr, dpr); x.clearRect(0, 0, W, H);
        var pitch = bw + 2 * pad, big = bw >= 8, gap = big ? 4 : 2, w = pitch - gap, cut = big ? 4 : 2, mid = H / 2;
        for (var i = 0; i < list.length; i++) {
          var b = list[i], h = i === vm.hoveredBeatIndex ? H : bh, x0 = i * pitch + gap / 2, y0 = mid - h / 2;
          var st = (b === 0 || b == null || b.status == null) ? -1 : b.status;
          cell(x, x0, y0, w, h, cut);
          if (st === -1) { x.fillStyle = ink(0.11); x.fill(); }
          else if (st === 1) { x.fillStyle = ink(1); x.fill(); }
          else if (st === 0) { x.save(); x.clip(); x.lineWidth = 2; x.strokeStyle = ink(1); x.stroke(); x.restore(); }
          else {
            // pending: hatch; maintenance: the same hatch, dim
            x.save(); x.clip(); x.lineWidth = 1; x.strokeStyle = ink(st === 2 ? 1 : 0.45); x.beginPath();
            for (var k = -h; k < w; k += 3) { x.moveTo(x0 + k, y0 + h); x.lineTo(x0 + k + h, y0); }
            x.stroke(); x.restore();
          }
        }
        if (!cv.hasAttribute("data-p-beat")) cv.setAttribute("data-p-beat", "");
      } catch (e) { try { return orig.apply(this, arguments); } catch (e2) { /* nothing to draw */ } }
    };
  }

  // ── the response-time chart ──────────────────────────────────
  var pats = {};
  function hatch(alpha, step) {
    var key = alpha + "/" + step;
    if (pats[key]) return pats[key];
    var c = document.createElement("canvas"); c.width = c.height = step;
    var x = c.getContext("2d");
    x.strokeStyle = ink(alpha); x.lineWidth = 1; x.lineCap = "square";
    x.beginPath(); x.moveTo(0, step); x.lineTo(step, 0); x.moveTo(-1, 1); x.lineTo(1, -1); x.moveTo(step - 1, step + 1); x.lineTo(step + 1, step - 1); x.stroke();
    return (pats[key] = x.createPattern(c, "repeat"));
  }
  function paint(c) {
    var ds = (c.data && c.data.datasets) || [];
    var lines = ds.filter(function (d) { return d.type !== "bar"; });
    lines.forEach(function (d, i) {
      // one line: the average. three: min, avg, max
      var role = lines.length === 1 ? "avg" : i === 0 ? "min" : i === 1 ? "avg" : "max";
      var col = role === "avg" ? ink(1) : ink(0.58);
      d.borderColor = col; d.backgroundColor = "rgba(0,0,0,0)"; d.fill = false;
      d.borderWidth = role === "avg" ? 1.5 : 1;
      d.borderDash = role === "avg" ? [] : role === "min" ? [1, 3] : [5, 4];
      d.pointBackgroundColor = col; d.pointBorderColor = col; d.pointHoverBackgroundColor = col;
    });
    ds.filter(function (d) { return d.type === "bar" && Array.isArray(d.backgroundColor); }).forEach(function (d) {
      d.backgroundColor = d.backgroundColor.map(function (v) {
        if (typeof v !== "string") return v;
        var s = v.replace(/\s/g, "");
        if (s.indexOf("220,53,69") >= 0) return hatch(0.9, 4);      // down
        if (s.indexOf("245,182,23") >= 0) return hatch(0.5, 6);     // pending
        if (s.indexOf("23,71,245") >= 0) return hatch(0.28, 8);     // maintenance
        return v;
      });
    });
    var o = c.config && c.config.options;
    if (!o) return;
    var f = { family: FONT, size: 9, weight: "400" };
    ["x", "y"].forEach(function (k) {
      var s = o.scales && o.scales[k];
      if (!s) return;
      s.grid = s.grid || {}; s.grid.color = ink(0.14); s.grid.borderColor = ink(0.28);
      s.border = s.border || {}; s.border.color = ink(0.28);
      s.ticks = s.ticks || {}; s.ticks.color = ink(0.58); s.ticks.font = f;
      if (s.title) { s.title.color = ink(0.58); s.title.font = f; if (typeof s.title.text === "string") s.title.text = s.title.text.toUpperCase(); }
    });
    var pl = o.plugins || (o.plugins = {});
    var tip = pl.tooltip || (pl.tooltip = {});
    tip.backgroundColor = "#0f0e11"; tip.titleColor = ink(0.58); tip.bodyColor = ink(1);
    tip.borderColor = ink(0.28); tip.borderWidth = 1; tip.cornerRadius = 0; tip.padding = 10; tip.displayColors = false;
    tip.titleFont = { family: FONT, size: 10, weight: "400" }; tip.bodyFont = { family: FONT, size: 12, weight: "600" };
    var lg = pl.legend || (pl.legend = {});
    lg.labels = lg.labels || {};
    lg.labels.color = ink(0.58); lg.labels.font = { family: FONT, size: 9, weight: "400" };
    lg.labels.boxWidth = 14; lg.labels.boxHeight = 1;
  }
  var TINT = { id: "plintTint", beforeUpdate: function (c) { try { paint(c); } catch (e) { /* keep Kuma's colours; the sheet greys them */ } } };

  // ── one walk over Vue's tree finds both kinds of canvas owner ─
  function walk(each) {
    var app = document.getElementById("app"), stack = [app && app._vnode], seen = 0;
    while (stack.length && seen++ < 20000) {
      var v = stack.pop();
      if (!v || typeof v !== "object") continue;
      var c = v.component;
      if (c) { try { each(c); } catch (e) { /* next */ } if (c.subTree) stack.push(c.subTree); }
      if (v.suspense && v.suspense.activeBranch) stack.push(v.suspense.activeBranch);
      var ch = v.children;
      if (Array.isArray(ch)) for (var i = 0; i < ch.length; i++) stack.push(ch[i]);
    }
  }
  var tries = 0, lastWalk = 0, lastWant = -1, lastPath = "", retry = null;
  function canvases() {
    // Unpainted canvases get a few tries (at most one walk a second) and are
    // then left to the sheet's fallback, so a changed Kuma costs nothing.
    var want = qa("#app canvas.heartbeat-canvas:not([data-p-beat]), #app .chart-wrapper canvas:not([data-p-tint])").length;
    if (!want) { tries = 0; lastWant = 0; return; }
    if (want !== lastWant || location.pathname !== lastPath) { tries = 0; lastWant = want; lastPath = location.pathname; }
    if (tries >= 12) return;
    var now = Date.now();
    if (now - lastWalk < 1000) { if (!retry) retry = setTimeout(function () { retry = null; schedule(); }, 1050); return; }
    lastWalk = now; tries++;
    var fresh = [];
    try {
      walk(function (c) {
        var ctx = c.ctx;
        if (ctx && typeof ctx.drawCanvas === "function" && !ctx.$plintBeat && c.proxy && Array.isArray(c.proxy.shortBeatList)) {
          ctx.$plintBeat = true;
          ctx.drawCanvas = beatDraw(c, ctx.drawCanvas);
          ctx.drawCanvas();
        }
        var ex = c.exposed && c.exposed.chart, chart = ex && (ex.value !== undefined ? ex.value : ex);
        if (chart && chart.canvas && chart.config && !chart.$plintTint) fresh.push(chart);
      });
    } catch (e) { /* internals moved */ }
    fresh.forEach(function (chart) {
      chart.$plintTint = true;
      try {
        var list = chart.config.plugins;
        if (!Array.isArray(list)) return;
        if (list.indexOf(TINT) < 0) list.push(TINT);
        // stop the intro animation first: when it ends it would put Kuma's options back
        chart.stop();
        chart.update("none");
        chart.canvas.setAttribute("data-p-tint", "");
        if (document.fonts && document.fonts.ready) document.fonts.ready.then(function () { try { if (chart.canvas) chart.draw(); } catch (e) { /* gone */ } });
      } catch (e) { /* leave it grey */ }
    });
    if (!retry && qa("#app canvas.heartbeat-canvas:not([data-p-beat]), #app .chart-wrapper canvas:not([data-p-tint])").length)
      retry = setTimeout(function () { retry = null; schedule(); }, 1100);
  }

  // ── run ─────────────────────────────────────────────────────
  var busy = false;
  function decorate() {
    busy = false;
    try { meta(); } catch (e) { /* skip */ }
    try { quick(); } catch (e) { /* skip */ }
    try { stats(); } catch (e) { /* skip */ }
    try { badges(); } catch (e) { /* skip */ }
    try { logo(); } catch (e) { /* skip */ }
    try { canvases(); } catch (e) { /* skip */ }
  }
  function schedule() {
    if (busy) return;
    busy = true;
    if (window.requestAnimationFrame) window.requestAnimationFrame(decorate); else setTimeout(decorate, 60);
  }
  function start() {
    schedule();
    try { new MutationObserver(schedule).observe(document.body, { childList: true, subtree: true, characterData: true }); } catch (e) { /* static page */ }
  }
  if (document.body) start(); else document.addEventListener("DOMContentLoaded", start);
})();
