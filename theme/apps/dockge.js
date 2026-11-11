/* ─────────────────────────────────────────────────────────────
   Plint Terminal · Dockge extras
   - home: the stack counts become one instrument: a heading row,
     a segmented dial for the active share, and one readout line.
     Dockge's own figures stay Dockge's; a zero goes dim and an
     exited count goes inverse.
   - stack page: one line under the title that says how many
     containers run, then one lamp + name + state per container,
     read from the cards Dockge renders.
   - phone: the folded stack row is scrolled to the open stack.
   Everything is redrawn idempotently, because Vue re-renders.
   ───────────────────────────────────────────────────────────── */
(function () {
  "use strict";
  var P = window.Plint;
  if (!P) return;

  // The strip opens the header; if a version has none, it opens <main>.
  P.mount = ["#app header", "#app main"];

  var OWN = ".xterm, .prism-editor-wrapper, .p-stackbar, .p-dial, .p-hd, .p-rd, #plint-pill, #plint-menu";

  function q(sel, root) { return (root || document).querySelector(sel); }
  function qa(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }
  function txt(n) { return n ? String(n.textContent || "").replace(/\s+/g, " ").trim() : ""; }
  function int(n) { var v = parseInt(txt(n).replace(/[^0-9]/g, ""), 10); return isNaN(v) ? 0 : v; }
  function cls(n, c, on) { if (n && n.classList.contains(c) !== on) n.classList.toggle(c, on); }
  function drop(sel) { qa(sel).forEach(function (n) { if (n.parentNode) n.parentNode.removeChild(n); }); }

  // ── home: active / exited / inactive ──
  function stats() {
    var nums = qa("#app .shadow-box > .row > .col > .num");
    if (!nums.length) { drop("#app .p-hd, #app .p-dial, #app .p-rd"); return; }
    var row = nums[0].parentNode.parentNode, box = row.parentNode;
    var c = { active: 0, exited: 0, inactive: 0 }, seen = 0;
    nums.forEach(function (n) {
      var k = n.classList.contains("inactive") ? "inactive" : n.classList.contains("exited") ? "exited" : n.classList.contains("active") ? "active" : "";
      if (!k) return;
      seen++;
      c[k] = int(n);
      cls(n.parentNode, "p-zero", c[k] === 0);
      cls(n.parentNode, "p-alarm", k === "exited" && c[k] > 0);
    });
    if (!seen) return;
    var total = c.active + c.exited + c.inactive;
    var pct = total ? Math.round(c.active / total * 100) : 0;
    var key = [c.active, c.exited, c.inactive].join("/");
    var hd = q(":scope > .p-hd", box), dial = q(":scope > .p-dial", box), rd = q(":scope > .p-rd", box);
    var fresh = !hd && !dial && !rd;
    if (box.getAttribute("data-p") === key && hd && dial && rd) return;
    box.setAttribute("data-p", key);

    if (!hd) { hd = P.el("div", "p-hd" + (fresh ? " p-fl" : "")); box.insertBefore(hd, box.firstChild); }
    hd.innerHTML = '<span>Stacks <span class="p-ru">/ СТЕКИ</span></span><span class="p-dim">' + total + " total</span>";

    if (!dial) { dial = P.el("div", "p-dial" + (fresh ? " p-fl" : "")); box.insertBefore(dial, row); }
    dial.innerHTML = P.ring(pct, { size: 76, width: 9, label: pct + "% of stacks active" }) + "<b>" + pct + "</b>";

    if (!rd) { rd = P.el("div", "p-rd" + (fresh ? " p-fl" : "")); box.appendChild(rd); }
    var line = total ? "<b>" + c.active + " of " + total + "</b> stack" + (total === 1 ? "" : "s") + " active" : "no stacks yet";
    if (c.exited) line += " · <b>" + c.exited + " exited</b>";
    if (c.inactive) line += " · " + c.inactive + " inactive";
    rd.innerHTML = line;

    if (fresh && P.strike) { P.strike(hd, 0); P.strike(dial, 60); P.strike(rd, 120); }
  }

  // ── stack page: the containers at a glance ──
  function tone(s) {
    s = s.toLowerCase();
    if (/(unhealthy|exited|dead|error|stopped)/.test(s)) return "bad";
    if (/^(running|healthy|up)/.test(s)) return "ok";
    if (/(starting|restarting|paused|removing)/.test(s)) return "warn";
    return "off";
  }
  function strip() {
    var bar = q("#app main .p-stackbar");
    var h1 = q("#app main h1");
    var cards = /^\/compose\/.+/.test(location.pathname) && h1 ? qa("#app main .shadow-box.container") : [];
    if (!cards.length) { if (bar && bar.parentNode) bar.parentNode.removeChild(bar); return; }
    var items = [], up = 0, bad = 0, known = 0;
    cards.forEach(function (card) {
      var name = txt(q("h4", card)), state = "";
      try { state = txt(q(":scope > .row > :first-child > div:not(.image) > .badge", card)); } catch (e) {}
      if (!name) return;
      var t = state ? tone(state) : "off";
      if (state) known++;
      if (t === "ok") up++;
      if (t === "bad") bad++;
      items.push({ name: name, state: state, tone: t });
    });
    if (!items.length) { if (bar && bar.parentNode) bar.parentNode.removeChild(bar); return; }
    var key = location.pathname + "|" + items.map(function (i) { return i.name + ":" + i.state; }).join(",");
    if (bar && bar.getAttribute("data-p") === key && bar.previousElementSibling === h1) return;
    var fresh = !bar;
    if (!bar) { bar = P.el("div", "p-stackbar p-fl"); bar.setAttribute("role", "list"); }
    bar.setAttribute("data-p", key);
    var n = items.length, sum = known
      ? "<b>" + up + " of " + n + "</b> container" + (n === 1 ? "" : "s") + " running" + (bad ? " · <b>" + bad + " down</b>" : "")
      : n + " container" + (n === 1 ? "" : "s");
    bar.innerHTML = '<span class="p-sum">' + sum + "</span>" + items.map(function (i) {
      return '<span class="p-chip ' + i.tone + '" role="listitem">' + (i.tone === "bad" ? "" : P.lamp(i.tone !== "off")) +
        "<b>" + P.esc(i.name) + "</b>" + (i.state ? "<small>" + P.esc(i.state) + "</small>" : "") + "</span>";
    }).join("");
    if (bar.previousElementSibling !== h1) h1.parentNode.insertBefore(bar, h1.nextSibling);
    if (fresh && P.strike) P.strike(bar, 40); else bar.classList.add("on");
  }

  // ── phone: the stack row scrolls sideways; show the open stack ──
  var shown = "";
  function reveal() {
    if (shown === location.pathname) return;
    var list = q("#app .stack-list"), act = list && q(".item.active", list);
    if (!list || !act) return;
    shown = location.pathname;
    if (list.scrollWidth <= list.clientWidth + 2) return;
    var a = act.getBoundingClientRect(), l = list.getBoundingClientRect();
    if (a.left < l.left || a.right > l.right) list.scrollLeft += a.left - l.left - 12;
  }

  function decorate() {
    try { stats(); } catch (e) {}
    try { strip(); } catch (e) {}
    try { reveal(); } catch (e) {}
  }

  // One observer, deaf to the terminal, the editors and our own pieces.
  var timer = 0, mo = null;
  function later() { if (!timer) timer = setTimeout(function () { timer = 0; decorate(); }, 150); }
  function mine(node) {
    if (!node) return false;
    if (node.nodeType !== 1) node = node.parentNode;
    return !!(node && node.closest && node.closest(OWN));
  }
  function watch() {
    if (mo || !window.MutationObserver) return;
    mo = new MutationObserver(function (recs) {
      for (var i = 0; i < recs.length; i++) if (!mine(recs[i].target)) { later(); return; }
    });
    mo.observe(q("#app") || document.body, { childList: true, subtree: true, characterData: true });
  }
  function unwatch() { if (mo) { mo.disconnect(); mo = null; } }

  document.addEventListener("visibilitychange", function () {
    if (document.hidden) unwatch(); else { watch(); later(); }
  });
  function start() { decorate(); if (!document.hidden) watch(); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start); else start();
})();
