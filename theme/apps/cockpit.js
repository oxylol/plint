/* ─────────────────────────────────────────────────────────────
   Plint Soft UI · Cockpit extras
   plint.js loads this in the shell and in every page iframe.
   - shell: the Plint pill goes into the masthead (#topnav), which
     cockpit.css keeps on top on desktop and on phones
   - Overview frame: the Usage card's CPU and memory bars become rings
   - login page: the Plint mark above the form
   Everything is redrawn idempotently, because Cockpit re-renders.
   Cockpit's CSP forbids inline style attributes, so this only sets
   classes and SVG attributes, never style="…".
   ───────────────────────────────────────────────────────────── */
(function () {
  "use strict";
  var P = window.Plint;
  if (!P) return;

  var LOGIN = !!document.querySelector("body.login-pf");
  // The login page has no masthead, so there the pill simply floats top-left.
  if (window.top === window && !LOGIN) P.mount = ["#topnav"];

  function q(sel, root) { return (root || document).querySelector(sel); }
  function set(node, html) { if (node && node.innerHTML !== html) node.innerHTML = html; }
  function toggle(node, cls, on) { if (node && node.classList.contains(cls) !== on) node.classList.toggle(cls, on); }

  // ── Overview · Usage card ───────────────────────────────────
  function tone(pct) { return pct >= 95 ? "bad" : pct >= 85 ? "warn" : ""; }
  function gauge(label, pct, sub) {
    var r = Math.round(pct);
    return '<div class="p-gauge"><div class="p-dial">' +
      P.ring(pct, { size: 112, width: 11, tone: tone(pct), label: label + " " + r + " percent" }) +
      "<b>" + r + "%</b></div><div><strong>" + P.esc(label) + "</strong><small>" + P.esc(sub) + "</small></div></div>";
  }
  // Reads one row of the Usage table: the progressbar's aria values and its measure text.
  function reading(th) {
    var row = th && th.closest("tr");
    var bar = row && q('[role="progressbar"]', row);
    if (!bar) return null;
    var now = parseFloat(bar.getAttribute("aria-valuenow")) || 0;
    var min = parseFloat(bar.getAttribute("aria-valuemin")) || 0;
    var max = parseFloat(bar.getAttribute("aria-valuemax")) || 100;
    var m = q(".pf-v5-c-progress__status .pf-v5-c-progress__measure", row);
    return { pct: max > min ? Math.max(0, Math.min(100, (now - min) / (max - min) * 100)) : 0, text: m ? m.textContent.trim() : "" };
  }
  function usage() {
    var card = q(".system-usage");
    var body = card && q(".pf-v5-c-card__body", card);
    if (!body) return;
    var cpu = reading(q("#system-usage-cpu-progress", card));
    var mem = reading(q("#system-usage-memory-progress", card));
    if (!cpu && !mem) { toggle(card, "p-has-rings", false); return; }
    var box = q(".p-usage", body);
    if (!box) {
      box = P.el("div", "p-usage");
      box.setAttribute("role", "group");
      box.setAttribute("aria-label", "Usage");
      body.insertBefore(box, body.firstChild);
    }
    toggle(card, "p-has-rings", true);
    set(box,
      (cpu ? gauge("CPU", cpu.pct, cpu.text.replace(/^\s*[\d.]+\s*%\s*/, "") || "usage") : "") +
      (mem ? gauge("Memory", mem.pct, mem.text) : ""));
  }

  // ── login page: the Plint mark above the form ──────────────
  var LOGO = '<svg viewBox="0 0 128 128" aria-hidden="true"><rect x="44" y="26" width="40" height="34" rx="7" opacity=".55"/>' +
    '<rect x="32" y="62" width="64" height="18" rx="6" opacity=".78"/><rect x="18" y="84" width="92" height="20" rx="7"/></svg>';
  function login() {
    if (!LOGIN) return;
    var brand = document.getElementById("brand");
    if (brand && !document.querySelector(".p-login-head")) {
      brand.parentNode.insertBefore(P.el("div", "p-login-head", '<span class="p-logo">' + LOGO + "</span><b>Plint</b>"), brand);
      brand.classList.add("p-under-head");
    }
  }

  // ── run ─────────────────────────────────────────────────────
  var busy = false;
  function decorate() {
    if (busy) return;
    busy = true;
    try { login(); usage(); } finally { busy = false; }
  }
  var queued = false;
  function schedule() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(function () { queued = false; decorate(); });
  }
  function start() {
    if (!document.body) return;
    new MutationObserver(schedule).observe(document.body, {
      childList: true, subtree: true, characterData: true,
      attributes: true, attributeFilter: ["aria-valuenow"]
    });
    decorate();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
