/* ─────────────────────────────────────────────────────────────
   Plint Terminal · Speedtest Tracker extras
   - one mode: the app keeps its own light/dark choice (a key per
     page, a `dark` class on <html>); both are pinned to dark
   - the Chart.js series are redrawn in the one light: first series
     solid, the next dashed, then dotted; the "Average" reference a
     dim dashed line; a lone series gets a dot-matrix fill
   ───────────────────────────────────────────────────────────── */
(function () {
  "use strict";
  var P = window.Plint;
  if (!P) return;
  P.mount = ["main > header > div:first-child", ".fi-topbar-start"];

  var root = document.documentElement;

  // ── always dark ─────────────────────────────────────────────
  function pin() {
    try {
      if (localStorage.getItem("filament:theme:mode") !== "dark") localStorage.setItem("filament:theme:mode", "dark");
      if (localStorage.getItem("theme") !== "dark") localStorage.setItem("theme", "dark");
    } catch (e) {}
    if (!root.classList.contains("dark")) root.classList.add("dark");
  }
  pin();
  try {
    // the app takes the class off when its switcher or the device says "light"
    new MutationObserver(pin).observe(root, { attributes: true, attributeFilter: ["class"] });
  } catch (e) {}

  // ── charts ──────────────────────────────────────────────────
  function token(name) {
    try { return getComputedStyle(root).getPropertyValue(name).trim(); } catch (e) { return ""; }
  }
  function ink(a) { return "rgba(" + (token("--p-ph") || "228, 248, 234") + "," + a + ")"; }

  // a dot-matrix fill: one lit dot per 4px cell, sharp at any pixel ratio
  var fills = {};
  function dots(ctx) {
    var dpr = Math.max(1, Math.round(window.devicePixelRatio || 1));
    if (fills[dpr]) return fills[dpr];
    try {
      var c = document.createElement("canvas"), cell = 4 * dpr;
      c.width = c.height = cell;
      var x = c.getContext("2d");
      x.fillStyle = ink(0.3);
      x.fillRect(0, 0, dpr, dpr);
      var p = ctx.createPattern(c, "repeat");
      if (p.setTransform && window.DOMMatrix) p.setTransform(new DOMMatrix().scale(1 / dpr));
      fills[dpr] = p;
    } catch (e) { fills[dpr] = ink(0.08); }
    return fills[dpr];
  }

  var DASH = [[], [6, 4], [1, 4]], MARK = ["rect", "rectRot", "crossRot"];
  function isRef(ds) { return /^(average|avg|mean)$/i.test(String(ds.label || "").trim()) || (ds.fill === false && ds.pointRadius === 0); }

  function restyle(chart) {
    var sets = chart && chart.data && chart.data.datasets;
    if (!sets || !sets.length) return;
    var type = chart.config && chart.config.type, series = 0, i, n = 0;
    for (i = 0; i < sets.length; i++) if (!isRef(sets[i])) series++;
    for (i = 0; i < sets.length; i++) {
      var ds = sets[i], c;
      ds.tension = 0;
      ds.cubicInterpolationMode = "default";
      ds.borderCapStyle = "butt";
      ds.pointBorderWidth = 0;
      ds.pointHoverBorderWidth = 0;
      if (type === "bar") {
        c = ink(n % 2 ? 0.45 : 1);
        ds.backgroundColor = c; ds.borderColor = c; ds.hoverBackgroundColor = c; ds.hoverBorderColor = c; ds.borderWidth = 0; ds.borderRadius = 0;
        n++;
        continue;
      }
      if (isRef(ds)) {
        c = ink(0.58);
        ds.borderDash = [2, 3]; ds.borderWidth = 1; ds.pointRadius = 0; ds.pointHoverRadius = 0; ds.fill = false;
        ds.backgroundColor = "rgba(0,0,0,0)";
      } else {
        c = ink(n > 1 ? 0.58 : 1);
        ds.borderDash = DASH[n % DASH.length];
        ds.borderWidth = 1.5;
        ds.pointStyle = MARK[n % MARK.length];
        ds.pointRadius = sets[i].data && sets[i].data.length > 48 ? 0 : 2.5;
        ds.pointHoverRadius = 4;
        ds.fill = series === 1 ? "origin" : false;
        ds.backgroundColor = series === 1 ? dots(chart.ctx) : "rgba(0,0,0,0)";
        n++;
      }
      ds.borderColor = c; ds.pointBackgroundColor = c; ds.pointBorderColor = c;
      ds.pointHoverBackgroundColor = ink(1); ds.pointHoverBorderColor = ink(1);
    }
  }

  function defaults(Chart) {
    try {
      var d = Chart.defaults;
      d.font.size = 9;
      d.font.family = token("--p-font") || d.font.family;
      d.plugins.legend.labels.boxHeight = 1;
      d.plugins.legend.labels.padding = 14;
      d.plugins.legend.labels.font = { size: 9 };
      d.plugins.tooltip.displayColors = false;
      d.plugins.tooltip.titleFont = { size: 9, weight: "600" };
      d.plugins.tooltip.bodyFont = { size: 11 };
      d.plugins.tooltip.padding = 9;
      d.plugins.tooltip.caretSize = 0;
    } catch (e) {}
  }

  function caps(v) {
    if (typeof v === "string") return v.toUpperCase();
    if (v && v.map) return v.map(caps);
    return v;
  }

  var plugin = {
    id: "plintTerminal",
    beforeInit: function (chart) { defaults(chart.constructor); restyle(chart); },
    beforeUpdate: function (chart) { restyle(chart); },
    // legend entries: a short sample of each line, not a filled box
    beforeDraw: function (chart) {
      try {
        var items = chart.legend && chart.legend.legendItems, i, k, t;
        for (i = 0; items && i < items.length; i++) { items[i].fillStyle = "rgba(0,0,0,0)"; items[i].lineWidth = 1.5; items[i].text = caps(items[i].text); }
        // axis text in caps like every other label (one cell per glyph, so the layout holds)
        for (k in chart.scales) {
          t = chart.scales[k].ticks;
          for (i = 0; t && i < t.length; i++) t[i].label = caps(t[i].label);
        }
      } catch (e) {}
    }
  };

  // The app builds its charts with its own bundled Chart.js. It registers window.filamentChartJsGlobalPlugins when that
  // bundle loads; if it loaded before us, take the class from the first chart we can reach and register there.
  try { (window.filamentChartJsGlobalPlugins = window.filamentChartJsGlobalPlugins || []).push(plugin); } catch (e) {}

  var Lib = null, tries = 0, timer = null;
  function chartOf(frame) {
    try {
      var data = frame._x_dataStack && frame._x_dataStack[0];
      return data && data.getChart ? data.getChart() : null;
    } catch (e) { return null; }
  }
  function adopt() {
    timer = null;
    var frames = document.querySelectorAll(".fi-wi-chart-frame"), waiting = 0, i, chart;
    for (i = 0; i < frames.length; i++) {
      chart = chartOf(frames[i]);
      if (!chart) { waiting++; continue; }
      try {
        if (!Lib) {
          Lib = chart.constructor;
          defaults(Lib);
          if (!(Lib.registry && Lib.registry.plugins && Lib.registry.plugins.items && Lib.registry.plugins.items[plugin.id])) Lib.register(plugin);
        }
        if (!chart.$plint) { chart.$plint = true; chart.update(); }
      } catch (e) {}
    }
    // charts are created a tick after Alpine starts: look a few more times, then stop
    if (waiting && tries++ < 25) timer = setTimeout(adopt, 400);
  }
  function scan() { tries = 0; if (!timer) timer = setTimeout(adopt, 60); }

  // ── the name: same as its card on the home screen ───────────
  function title() {
    var h = document.querySelector("main > header h1");
    if (h && /speedtest tracker/i.test(h.textContent) && h.textContent !== "Internet") h.textContent = "Internet";
  }

  function start() {
    title();
    scan();
    // a web font that arrives after the first draw: draw the axis text again in it
    try {
      if (document.fonts && document.fonts.ready) document.fonts.ready.then(function () {
        var frames = document.querySelectorAll(".fi-wi-chart-frame");
        for (var i = 0; i < frames.length; i++) { var c = chartOf(frames[i]); if (c) try { c.update(); } catch (e) {} }
      });
    } catch (e) {}
    // Livewire page changes (if the panel runs as one page) bring new headers and charts
    document.addEventListener("livewire:navigated", function () { pin(); title(); scan(); });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
