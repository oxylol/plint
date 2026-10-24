/* ─────────────────────────────────────────────────────────────
   Plint Soft UI · Speedtest Tracker extras
   - light/dark follows the device (Filament keeps its own choice in
     localStorage and toggles html.dark; we pin it to "system")
   - recolours the Chart.js datasets: download sage, upload slate,
     averages amber, the rest down the same scale
   ───────────────────────────────────────────────────────────── */
(function () {
  "use strict";
  var P = window.Plint;
  if (!P) return;
  P.mount = ["main > header > div:first-child", ".fi-topbar-start"];

  // ── device light/dark ───────────────────────────────────────
  var mq = window.matchMedia("(prefers-color-scheme: dark)");
  function syncTheme() {
    try { if (localStorage.getItem("theme") !== "system") localStorage.setItem("theme", "system"); } catch (e) {}
    document.documentElement.classList.toggle("dark", mq.matches);
  }
  syncTheme();
  if (mq.addEventListener) mq.addEventListener("change", function () { syncTheme(); recolourAll(true); });

  // ── charts ──────────────────────────────────────────────────
  function token(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function alpha(hex, a) {
    var m = /^#?([0-9a-f]{6})$/i.exec(hex);
    if (!m) return hex;
    var n = parseInt(m[1], 16);
    return "rgba(" + (n >> 16) + "," + ((n >> 8) & 255) + "," + (n & 255) + "," + a + ")";
  }
  function colourFor(label, i) {
    var l = String(label || "").toLowerCase();
    if (/average|avg|mean/.test(l)) return token("--p-warn");
    if (/upload|jitter/.test(l)) return token("--p-info");
    if (/download|ping|latency/.test(l)) return token("--p-accent");
    return [token("--p-accent"), token("--p-info"), token("--p-warn"), token("--p-ink2")][i % 4];
  }
  function recolour(frame, force) {
    var data = frame._x_dataStack && frame._x_dataStack[0], chart;
    try { chart = data && data.getChart && data.getChart(); } catch (e) { return; }
    if (!chart || (chart.$plint && !force)) return;
    chart.$plint = true;
    chart.data.datasets.forEach(function (ds, i) {
      var c = colourFor(ds.label, i), avg = /average|avg|mean/i.test(ds.label || "");
      ds.borderColor = c;
      ds.pointBackgroundColor = c;
      ds.pointBorderColor = c;
      ds.backgroundColor = chart.config.type === "bar" ? alpha(c, 0.75) : alpha(c, avg ? 0 : 0.14);
      ds.borderWidth = avg ? 1.5 : 2.25;
      if (avg) ds.borderDash = [6, 5];
    });
    // "none" keeps Chart.js's cached point styles; a full update re-resolves them.
    chart.update();
  }
  function recolourAll(force) {
    Array.prototype.forEach.call(document.querySelectorAll(".fi-wi-chart-frame"), function (f) { recolour(f, force); });
  }

  var queued = false;
  function schedule() {
    if (queued) return;
    queued = true;
    setTimeout(function () { queued = false; title(); recolourAll(false); }, 150);
  }
  // Same name as its card on the home screen.
  function title() {
    var h = document.querySelector("main > header h1");
    if (h && /speedtest tracker/i.test(h.textContent) && h.textContent !== "Internet") h.textContent = "Internet";
  }

  function start() {
    title();
    recolourAll(false);
    // Livewire swaps charts on refresh and filter changes; new instances lack $plint.
    new MutationObserver(schedule).observe(document.body, { childList: true, subtree: true });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
