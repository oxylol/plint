/* ─────────────────────────────────────────────────────────────
   Plint Soft UI · Dockge extras
   - home: the active share of stacks as a ring around Dockge's own
     "active" number
   - stack page: one chip per container (status dot + name) under the
     title, read from the container cards Dockge renders
   Everything is redrawn idempotently, because Vue re-renders.
   ───────────────────────────────────────────────────────────── */
(function () {
  "use strict";
  var P = window.Plint;
  if (!P) return;

  // Dockge has no header on phones; the pill then opens <main>.
  P.mount = ["#app header", "#app main"];

  function q(sel, root) { return (root || document).querySelector(sel); }
  function qa(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }
  function num(s) { var n = parseFloat(String(s).replace(/[^0-9.\-]/g, "")); return isNaN(n) ? 0 : n; }
  function setText(node, txt) { if (node && node.textContent !== txt) node.textContent = txt; }
  function setClass(node, cls, on) { if (node && node.classList.contains(cls) !== on) node.classList.toggle(cls, on); }
  function setHTML(node, html) { if (node && node.innerHTML !== html) node.innerHTML = html; }

  // ── home: active / exited / inactive ────────────────────────
  function quick() {
    var cols = qa("#app .shadow-box > .row > .col").filter(function (c) { return q(":scope > h3 + .num", c); });
    if (cols.length < 2) return;
    var vals = cols.map(function (c) { return num(q(".num", c).textContent); });
    var active = vals[0], exited = vals[1] || 0, total = vals.reduce(function (a, b) { return a + b; }, 0);
    var first = cols[0];
    setClass(first, "p-quick-up", true);
    var pct = total ? active / total * 100 : 0;
    var d = q(":scope > .p-dial", first);
    if (!d) { d = P.el("div", "p-dial"); first.insertBefore(d, first.firstChild); }
    var key = Math.round(pct * 10) + "|" + total;
    if (d.getAttribute("data-p") !== key) {
      d.setAttribute("data-p", key);
      d.innerHTML = P.ring(pct, { size: 112, width: 11, label: active + " of " + total + " stacks running" });
    }
    var box = q(":scope > .p-up", first);
    if (!box) { box = P.el("div", "p-up", '<span class="p-lbl"></span><strong></strong><small></small>'); first.appendChild(box); }
    setText(box.children[0], "Running");
    setText(box.children[1], !total ? "No stacks yet" : active === total ? "All stacks running" : exited ? exited + " exited" : (total - active) + " not running");
    setClass(box.children[1], "p-bad", exited > 0);
    setText(box.children[2], active + " of " + total + " · " + Math.round(pct) + "%");
  }

  // ── stack page: container chips ─────────────────────────────
  function tone(s) {
    s = String(s || "").toLowerCase();
    if (/^(running|healthy|up)/.test(s)) return "ok";
    if (/(unhealthy|exited|dead|error|stopped)/.test(s)) return "bad";
    if (/(starting|restarting|created|paused|removing)/.test(s)) return "warn";
    return "";
  }
  function chips() {
    var h1 = q("#app main h1");
    var root = h1 && h1.parentNode;
    var bar = root && q(":scope > .p-stackbar", root);
    var cards = root ? qa(".shadow-box.container", root) : [];
    var items = cards.map(function (card) {
      var name = q("h4", card), st = q(":scope > .row > :first-child > div:not(.image) > .badge", card);
      return name && st ? { name: name.textContent.trim(), status: st.textContent.trim() } : null;
    }).filter(Boolean);
    if (!items.length) { if (bar) bar.remove(); return; }
    if (!bar) {
      bar = P.el("div", "p-stackbar");
      bar.setAttribute("role", "list");
      bar.setAttribute("aria-label", "Containers");
      root.insertBefore(bar, h1.nextSibling);
    }
    var key = JSON.stringify(items);
    if (bar.getAttribute("data-p") === key) return;
    bar.setAttribute("data-p", key);
    var up = items.filter(function (i) { return tone(i.status) === "ok"; }).length;
    setHTML(bar, '<span class="p-sum">' + up + " of " + items.length + " running</span>" + items.map(function (i) {
      return '<span class="p-chip ' + tone(i.status) + '" role="listitem" title="' + P.esc(i.name + ": " + i.status) + '"><i></i>' +
        P.esc(i.name) + "<small>" + P.esc(i.status) + "</small></span>";
    }).join(""));
  }

  // ── run ─────────────────────────────────────────────────────
  var busy = false;
  function decorate() {
    if (busy) return;
    busy = true;
    try { quick(); chips(); } catch (e) { /* never break Dockge */ } finally { busy = false; }
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
