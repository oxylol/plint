/* ─────────────────────────────────────────────────────────────
   Plint Soft UI · Uptime Kuma extras
   - Quick Stats: the up share as a ring around Kuma's own Up number
   - monitor page: 24 h / 30 d uptime and certificate days as rings
   - response-time chart (Chart.js): line, fill, grid and tooltip
     repainted in our tokens through a small Chart.js plugin
   The heartbeat strips need no JS: kuma.css feeds the canvas colours.
   Everything is redrawn idempotently, because Vue re-renders.
   ───────────────────────────────────────────────────────────── */
(function () {
  "use strict";
  var P = window.Plint;
  if (!P) return;

  // The public status page has no <header>; the pill goes above its title.
  P.mount = ["#app header", "#app .main"];

  function q(sel, root) { return (root || document).querySelector(sel); }
  function qa(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }
  function num(s) { var n = parseFloat(String(s).replace(/[^0-9.\-]/g, "")); return isNaN(n) ? null : n; }
  function token(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

  // A dial (raised disc + ring) that is only rebuilt when its value changes.
  function dial(host, before, pct, opts) {
    var d = q(":scope > .p-dial", host);
    if (!d) { d = P.el("div", "p-dial"); host.insertBefore(d, before || host.firstChild); }
    var key = Math.round(pct * 10) + "|" + (opts.tone || "") + "|" + opts.label;
    if (d.getAttribute("data-p") !== key) {
      d.setAttribute("data-p", key);
      d.innerHTML = P.ring(pct, { size: 112, width: 11, tone: opts.tone, label: opts.label });
    }
    return d;
  }
  function setText(node, txt) { if (node && node.textContent !== txt) node.textContent = txt; }
  function setClass(node, cls, on) { if (node && node.classList.contains(cls) !== on) node.classList.toggle(cls, on); }

  // ── dashboard home: Quick Stats ─────────────────────────────
  function quick() {
    var cols = qa("#app .shadow-box > .row > .col").filter(function (c) { return q(":scope > h3 + .num", c); });
    if (cols.length < 2) return;
    var vals = cols.map(function (c) { return num(q(".num", c).textContent) || 0; });
    var up = vals[0], down = vals[1] || 0, total = vals.reduce(function (a, b) { return a + b; }, 0);
    var first = cols[0];
    setClass(first, "p-quick-up", true);
    var pct = total ? up / total * 100 : 0;
    dial(first, first.firstChild, pct, { label: up + " of " + total + " monitors up" });
    var box = q(":scope > .p-up", first);
    if (!box) { box = P.el("div", "p-up", '<span class="p-lbl"></span><strong></strong><small></small>'); first.appendChild(box); }
    setText(box.children[0], "Up");
    setText(box.children[1], !total ? "No monitors yet" : up === total ? "All checks up" : down ? down + " down" : (total - up) + " not up");
    setClass(box.children[1], "p-bad", down > 0);
    setText(box.children[2], up + " of " + total + " · " + Math.round(pct) + "%");
  }

  // ── monitor page: rings for uptime and certificate ──────────
  function stats() {
    var tiles = qa("#app main .shadow-box.stats > .row > div");
    if (!tiles.length) return;
    var uptimeSeen = 0, afterUptime = false;
    tiles.forEach(function (t) {
      var n = q(".num", t);
      if (!n) return;
      var isUptime = !!q(":scope > span[title]", n);
      var kind = null;
      if (isUptime) {
        kind = uptimeSeen === 0 ? "24" : uptimeSeen === 1 ? "720" : null;
        uptimeSeen++;
        afterUptime = true;
      } else if (afterUptime && q(":scope > a", n)) {
        kind = "cert";
      }
      setClass(t, "p-ringtile", !!kind);
      setClass(t, "p-cert", kind === "cert");
      if (!kind) { var old = q(":scope > .p-dial", t); if (old) old.remove(); return; }
      var v = num(n.textContent);
      if (kind === "cert") {
        var days = v == null ? 0 : v;
        var tone = days <= 7 ? "bad" : days <= 21 ? "warn" : "info";
        dial(t, t.firstChild, Math.min(100, days / 90 * 100), { tone: tone, label: "Certificate valid for " + days + " more days" });
      } else {
        var pct = v == null ? 0 : v;
        var tone2 = v == null ? null : pct < 95 ? "bad" : pct < 99.5 ? "warn" : null;
        dial(t, t.firstChild, pct, { tone: tone2, label: "Uptime " + (kind === "24" ? "24 hours" : "30 days") + ": " + n.textContent.trim() });
      }
    });
  }

  // ── public status page: Plint mark instead of Kuma's default logo ──
  var LOGO = '<svg viewBox="0 0 128 128" aria-hidden="true"><rect x="44" y="26" width="40" height="34" rx="7" opacity=".55"/>' +
    '<rect x="32" y="62" width="64" height="18" rx="6" opacity=".78"/><rect x="18" y="84" width="92" height="20" rx="7"/></svg>';
  function logo() {
    qa("#app .main > h1 img.logo, #app a.item > img.logo").forEach(function (img) {
      var stock = /\/icon\.svg(\?|$)/.test(img.getAttribute("src") || "");
      setClass(img, "p-swapped", stock);
      var wrap = img.parentNode, mark = q(":scope > .p-logo", wrap);
      if (stock && !mark) wrap.insertBefore(P.el("span", "p-logo", LOGO), img);
      if (!stock && mark) mark.remove();
    });
  }

  // ── response-time chart (vue-chartjs → Chart.js) ────────────
  // Kuma hard-codes its greens in the datasets, so a tiny plugin repaints
  // them before every update. If the internals ever change, the walk
  // finds nothing and the chart simply keeps Kuma's colours.
  function rgba(hex, a) {
    var m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})/i.exec(hex || "");
    return m ? "rgba(" + parseInt(m[1], 16) + "," + parseInt(m[2], 16) + "," + parseInt(m[3], 16) + "," + a + ")" : hex;
  }
  function paint(c) {
    var t = {
      ink: token("--p-ink"), ink2: token("--p-ink2"), line: token("--p-line"), bg: token("--p-bg"),
      accent: token("--p-accent"), info: token("--p-info"), warn: token("--p-warn"), bad: token("--p-bad"),
      font: token("--p-font") || "sans-serif"
    };
    if (!t.accent) return;
    var ds = (c.data && c.data.datasets) || [];
    var lines = ds.filter(function (d) { return d.type !== "bar"; });
    lines.forEach(function (d, i) {
      var role = lines.length === 1 ? "avg" : i === 0 ? "min" : i === 1 ? "avg" : "max";
      var col = role === "avg" ? t.accent : role === "min" ? t.info : t.ink2;
      d.borderColor = col;
      d.backgroundColor = role === "avg" ? rgba(t.accent, 0.14) : "rgba(0,0,0,0)";
      d.borderWidth = role === "avg" ? 2.5 : 1.5;
      d.pointBackgroundColor = col;
      d.pointBorderColor = col;
      d.pointHoverBackgroundColor = col;
    });
    ds.filter(function (d) { return d.type === "bar" && Array.isArray(d.backgroundColor); }).forEach(function (d) {
      d.backgroundColor = d.backgroundColor.map(function (v) {
        var s = String(v).replace(/\s/g, "");
        if (s.indexOf("220,53,69") >= 0) return rgba(t.bad, 0.3);
        if (s.indexOf("245,182,23") >= 0) return rgba(t.warn, 0.3);
        if (s.indexOf("23,71,245") >= 0) return rgba(t.info, 0.3);
        return v;
      });
    });
    var o = c.config && c.config.options;
    if (!o) return;
    var f = { family: t.font, size: 11, weight: "600" };
    ["x", "y"].forEach(function (k) {
      var s = o.scales && o.scales[k];
      if (!s) return;
      s.grid = s.grid || {}; s.grid.color = t.line;
      s.border = s.border || {}; s.border.color = t.line;
      s.ticks = s.ticks || {}; s.ticks.color = t.ink2; s.ticks.font = f;
      if (s.title) { s.title.color = t.ink2; s.title.font = { family: t.font, size: 11, weight: "700" }; }
    });
    var pl = o.plugins || (o.plugins = {});
    var tip = pl.tooltip || (pl.tooltip = {});
    tip.backgroundColor = t.ink; tip.titleColor = t.bg; tip.bodyColor = t.bg;
    tip.cornerRadius = 12; tip.padding = 12; tip.displayColors = false;
    tip.titleFont = { family: t.font, size: 12, weight: "700" }; tip.bodyFont = { family: t.font, size: 13, weight: "800" };
    var lg = pl.legend || (pl.legend = {});
    lg.labels = lg.labels || {};
    lg.labels.color = t.ink; lg.labels.font = { family: t.font, size: 12, weight: "700" };
    lg.labels.boxWidth = 14; lg.labels.boxHeight = 4;
  }
  var retry = null;
  var TINT = { id: "plintTint", beforeUpdate: function (c) { try { paint(c); } catch (e) { /* keep Kuma's colours */ } } };

  function findCharts() {
    var root = document.getElementById("app");
    var out = [], seen = 0;
    function walk(v, depth) {
      if (!v || depth > 400 || seen > 20000) return;
      seen++;
      var c = v.component;
      if (c) {
        var ch = c.exposed && c.exposed.chart;
        var chart = ch && (ch.value !== undefined ? ch.value : ch);
        if (chart && chart.canvas && chart.config) out.push(chart);
        walk(c.subTree, depth + 1);
      }
      if (Array.isArray(v.children)) for (var i = 0; i < v.children.length; i++) walk(v.children[i], depth + 1);
    }
    try { walk(root && root._vnode, 0); } catch (e) { /* internals moved: nothing to tint */ }
    return out;
  }
  function charts() {
    // Untinted chart canvases; each gets a few tries (at most one walk a
    // second) and is then left alone, so a changed Kuma costs nothing.
    var now = Date.now(), waiting = false;
    var canv = qa("#app .chart-wrapper canvas").filter(function (c) {
      if (c.hasAttribute("data-p-tint")) return false;
      var t = c.$plintTry || (c.$plintTry = { n: 0, at: 0 });
      if (t.n >= 15) return false;
      waiting = true;
      if (now - t.at < 1000) return false;
      t.n++; t.at = now;
      return true;
    });
    if (waiting && !retry) retry = setTimeout(function () { retry = null; schedule(); }, 1100);
    if (!canv.length) return;
    findCharts().forEach(function (chart) {
      if (canv.indexOf(chart.canvas) < 0 || chart.$plintTint) return;
      chart.$plintTint = true;
      chart.canvas.setAttribute("data-p-tint", "");
      try {
        var list = chart.config.plugins;
        if (Array.isArray(list) && list.indexOf(TINT) < 0) list.push(TINT);
        // Stop Chart.js's intro animation first: when it finishes it would
        // put Kuma's original options back on the line.
        chart.stop();
        chart.update("none");
        if (document.fonts && document.fonts.ready) document.fonts.ready.then(function () { if (chart.canvas) chart.draw(); });
      } catch (e) { /* leave it */ }
    });
  }

  // ── run ─────────────────────────────────────────────────────
  var busy = false;
  function decorate() {
    if (busy) return;
    busy = true;
    try { quick(); stats(); logo(); charts(); } catch (e) { /* never break Kuma */ } finally { busy = false; }
  }
  var queued = false;
  function schedule() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(function () { queued = false; decorate(); });
  }
  function start() {
    new MutationObserver(schedule).observe(document.body, { childList: true, subtree: true, characterData: true });
    decorate();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
