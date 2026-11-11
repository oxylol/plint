/* ─────────────────────────────────────────────────────────────
   Plint Terminal · ntfy extras
   - the Plint strip goes into ntfy's own toolbar (Plint.mount)
   - the week: seven columns of cells above the notification list,
     one cell = k messages, with a one-line readout
   - notifications become a log: day headings, priority labels and
     tags as flat labels (all via data attributes that apps/ntfy.css draws)
   - a "no signal" line above ntfy's empty-state text
   - small hooks for the sheet: data-p-view on <html>, data-p-prof on the
     account button, data-p-raw on a dialog title that holds an address
   Data comes from the web app's own IndexedDB (read-only; the same
   store the list is rendered from), so nothing extra hits the server.
   Everything is redrawn idempotently, because React re-renders.
   ───────────────────────────────────────────────────────────── */
(function () {
  "use strict";
  var P = window.Plint;
  if (!P) return;
  P.mount = ["header.MuiAppBar-root > .MuiToolbar-root", "header .MuiToolbar-root"];

  function q(sel, root) { return (root || document).querySelector(sel); }
  function qa(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }
  function set(node, html) { if (node && node.innerHTML !== html) node.innerHTML = html; }
  function attr(node, name, value) {
    if (!node) return;
    if (value == null) { if (node.hasAttribute(name)) node.removeAttribute(name); }
    else if (node.getAttribute(name) !== value) node.setAttribute(name, value);
  }
  function norm(s) { return String(s || "").toLowerCase().replace(/[^a-z0-9]/g, ""); }

  // ── which page are we on ────────────────────────────────────
  var RESERVED = /^(settings|account|login|signup|reset-password|docs|static|file|v1)$/;
  function view() {
    var parts = location.pathname.split("/").filter(Boolean);
    try { parts = parts.map(decodeURIComponent); } catch (e) {}
    if (!parts.length || (parts.length === 1 && parts[0] === "app")) return { kind: "all" };
    if (parts.length === 1 && !RESERVED.test(parts[0])) return { kind: "topic", topic: parts[0] };
    if (parts.length === 2 && parts[0].indexOf(".") > 0) return { kind: "topic", topic: parts[1], host: parts[0] };
    return { kind: "other" };
  }

  // ── data: the web app's IndexedDB, read-only ────────────────
  // ntfy keeps one database per signed-in user ("ntfy-<user>") or "ntfy".
  var data = null, loading = false, loadedSig = null, reloadTimer = null;
  function signedIn() {
    try { return !!(localStorage.getItem("user") && localStorage.getItem("token")); } catch (e) { return false; }
  }
  function dbName() {
    try { return signedIn() ? "ntfy-" + localStorage.getItem("user") : "ntfy"; } catch (e) { return "ntfy"; }
  }
  function load() {
    if (loading || !window.indexedDB || document.hidden) return;
    loading = true;
    function done(d) { loading = false; data = d; decorate(); }
    var req;
    try { req = indexedDB.open(dbName()); } catch (e) { return done(null); }
    // No database yet: abort, so we never create one under the app's feet.
    req.onupgradeneeded = function () { try { req.transaction.abort(); } catch (e) {} };
    req.onerror = function (e) { if (e && e.preventDefault) e.preventDefault(); done(null); };
    req.onsuccess = function () {
      var db = req.result, out = {};
      if (!db.objectStoreNames.contains("notifications") || !db.objectStoreNames.contains("subscriptions")) { db.close(); return done(null); }
      var tx;
      try { tx = db.transaction(["subscriptions", "notifications"], "readonly"); } catch (e) { db.close(); return done(null); }
      tx.objectStore("subscriptions").getAll().onsuccess = function (e) { out.subs = e.target.result || []; };
      tx.objectStore("notifications").getAll().onsuccess = function (e) { out.notes = e.target.result || []; };
      tx.oncomplete = function () { db.close(); done(out); };
      tx.onerror = tx.onabort = function () { db.close(); done(null); };
    };
  }
  function reloadSoon() {
    clearTimeout(reloadTimer);
    reloadTimer = setTimeout(load, 250);
  }

  // The notifications this page lists, newest first (as ntfy sorts them).
  function listed(v) {
    if (!data || !data.notes || (v.kind !== "all" && v.kind !== "topic")) return null;
    var notes = data.notes;
    if (v.kind === "topic") {
      var base = (window.config && window.config.base_url) || location.origin;
      var subs = (data.subs || []).filter(function (s) {
        return s.topic === v.topic && (!v.host || String(s.baseUrl).replace(/^https?:\/\//, "") === v.host);
      });
      if (!v.host) {
        var own = subs.filter(function (s) { return s.baseUrl === base; });
        if (own.length) subs = own;
      }
      var ids = subs.map(function (s) { return s.id; });
      notes = notes.filter(function (n) { return ids.indexOf(n.subscriptionId) >= 0; });
    }
    return notes.slice().sort(function (a, b) { return b.time - a.time || (a.id < b.id ? 1 : a.id > b.id ? -1 : 0); });
  }

  // ── days ────────────────────────────────────────────────────
  function dayStart(d) { return new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime(); }
  function dayKey(t) { return dayStart(new Date(t * 1000)); }
  var DAY = 86400000;
  function dayLabel(t) {
    var d = new Date(t * 1000), today = dayStart(new Date());
    var diff = Math.round((today - dayStart(d)) / DAY);
    if (diff === 0) return "Today";
    if (diff === 1) return "Yesterday";
    var o = { weekday: "short", day: "numeric", month: "short" };
    if (d.getFullYear() !== new Date().getFullYear()) o.year = "numeric";
    return d.toLocaleDateString("en-GB", o).replace(",", "");
  }

  // ── hooks for the sheet ─────────────────────────────────────
  // The account button has a translated label, so it is found by place:
  // signed in, it is the last thing in the toolbar.
  function hooks() {
    var tb = q("header.MuiAppBar-root > .MuiToolbar-root");
    if (tb) {
      var last = tb.lastElementChild, on = signedIn();
      qa(":scope > [data-p-prof]", tb).forEach(function (b) { if (b !== last || !on) b.removeAttribute("data-p-prof"); });
      if (on && last && last.classList.contains("MuiIconButton-root")) attr(last, "data-p-prof", "");
    }
    // "Publish to host/topic": an address keeps its own case
    qa(".MuiDialogTitle-root").forEach(function (t) {
      attr(t, "data-p-raw", t.textContent.indexOf("/") >= 0 ? "" : null);
    });
  }

  // ── empty states: a "no signal" line above ntfy's own words ──
  function nosignal() {
    qa("#main > .MuiContainer-root > h5.MuiTypography-alignCenter").forEach(function (h5) {
      var box = h5.parentNode;
      if (q(":scope > .p-ntfy-nosig", box)) return;
      var line = P.el("div", "p-ntfy-nosig", '<i class="p-blk off"></i><span>No signal <span class="p-ru">/ Нет сигнала</span></span>');
      line.setAttribute("aria-hidden", "true");
      box.insertBefore(line, h5);
    });
  }

  // ── notification cards ──────────────────────────────────────
  function cardKey(card) {
    var h = q('[role="rowheader"]', card), m = q(":scope > .MuiCardContent-root > p:nth-of-type(2)", card);
    return norm((h ? h.textContent : "") + (m ? m.textContent : ""));
  }
  // ntfy shortens links in the text it shows ("https://x.net" → "x.net")
  function noteKey(n) { return norm(String(n.title || n.message || "").replace(/https?:\/\//g, "")).slice(0, 12); }
  function match(card, list, i) {
    var text = cardKey(card);
    for (var d = 0; d <= 3; d++) {
      var cand = [list[i + d], list[i - d]];
      for (var k = 0; k < 2; k++) {
        var n = cand[k];
        if (n && (!noteKey(n) || text.indexOf(noteKey(n)) >= 0)) return n;
      }
    }
    return null;
  }
  function prioFromImg(card) {
    var img = q(":scope > .MuiCardContent-root > p:first-of-type > img", card);
    var m = img && /([1-5])/.exec(img.getAttribute("alt") || "");
    return m ? m[1] : "3";
  }
  function tags(card) {
    var content = q(":scope > .MuiCardContent-root", card);
    var src = content && q(":scope > p:nth-of-type(3)", content);
    var chips = content && q(":scope > .p-ntfy-tags", content);
    var txt = src ? src.textContent : "";
    var at = txt.indexOf(": ");
    if (!src || at < 0) { if (chips) chips.remove(); return; }
    attr(src, "data-p-tags-src", "");
    var list = txt.slice(at + 2).split(/,\s*/).filter(Boolean);
    if (!chips) {
      chips = P.el("div", "p-ntfy-tags");
      chips.setAttribute("role", "list");
      chips.setAttribute("aria-label", txt.slice(0, at));
      src.insertAdjacentElement("afterend", chips);
    }
    set(chips, list.map(function (t) { return '<span role="listitem">' + P.esc(t) + "</span>"; }).join(""));
  }
  function cards(list) {
    var prev = null;
    qa('[role="list"] .MuiCard-root[role="listitem"]').forEach(function (card, i) {
      var n = list ? match(card, list, i) : null;
      attr(card, "data-p-prio", n ? String(n.priority || 3) : prioFromImg(card));
      var day = n ? dayKey(n.time) : null;
      attr(card, "data-p-day", n && day !== prev ? dayLabel(n.time) : null);
      prev = day;
      tags(card);
    });
  }

  // ── the week: messages per day, last 7 days, as cells ───────
  // Each day is a 4 x 6 field; cells light from the bottom:
  // low/min hollow, normal dim, high/urgent full.
  var COLS = 4, ROWS = 6, CELLS = COLS * ROWS;
  function plural(n, word) { return P.fmt(n) + " " + word + (n === 1 ? "" : "s"); }
  function week(list) {
    var scroller = q("#main > .infinite-scroll-component__outerdiv");
    var box = q("#main > .p-ntfy-week");
    var today = dayStart(new Date()), days = [];
    for (var i = 6; i >= 0; i--) days.push({ at: today - i * DAY, lo: 0, n: 0, hi: 0 });
    var total = 0, high = 0, urgent = 0;
    (list || []).forEach(function (n) {
      var idx = 6 - Math.round((today - dayKey(n.time)) / DAY);
      if (idx < 0 || idx > 6) return;
      var p = n.priority || 3, d = days[idx];
      if (p >= 4) { d.hi++; if (p === 5) urgent++; else high++; } else if (p <= 2) d.lo++; else d.n++;
      total++;
    });
    if (!scroller || !list || !total) { if (box) box.remove(); return; }
    var fresh = !box;
    if (fresh) {
      box = P.el("section", "p-ntfy-week p-fl");
      box.setAttribute("aria-label", "Messages per day");
    }
    if (box.nextElementSibling !== scroller) scroller.parentNode.insertBefore(box, scroller);
    var top = Math.max.apply(null, days.map(function (d) { return d.lo + d.n + d.hi; })) || 1;
    var per = Math.max(1, Math.ceil(top / CELLS));
    var said = [], cols = days.map(function (d, k) {
      var sum = d.lo + d.n + d.hi;
      var name = k === 6 ? "Today" : new Date(d.at).toLocaleDateString("en-GB", { weekday: "short" });
      said.push(name + " " + sum);
      // cells, counted from the bottom left: low first, then normal, high on top
      var all = Math.min(CELLS, Math.ceil(sum / per));
      var hi = d.hi ? Math.min(all, Math.max(1, Math.round(d.hi / per))) : 0;
      var lo = d.lo ? Math.min(all - hi, Math.max(1, Math.round(d.lo / per))) : 0;
      var cells = "";
      for (var r = ROWS - 1; r >= 0; r--) {
        for (var c = 0; c < COLS; c++) {
          var at = r * COLS + c;
          cells += at >= all ? "<i></i>" : at < lo ? '<i class="lo"></i>' : at < all - hi ? '<i class="n"></i>' : '<i class="hi"></i>';
        }
      }
      return '<div class="p-ntfy-day' + (k === 6 ? " today" : "") + '"><span class="p-ntfy-n' + (sum ? "" : " zero") + '">' + sum + "</span>" +
        '<span class="p-ntfy-col">' + cells + "</span>" +
        '<span class="p-ntfy-d">' + P.esc(name) + "</span></div>";
    }).join("");
    var now = days[6].lo + days[6].n + days[6].hi;
    var read = "&gt; " + plural(total, "message") + " in 7 days · " + P.fmt(now) + " today" +
      (urgent ? " · " + P.fmt(urgent) + " urgent" : "") + (high ? " · " + P.fmt(high) + " high" : "");
    set(box,
      '<div class="p-ntfy-head"><h2 class="p-ntfy-h">Last 7 days <span class="p-ru">/ 7 дней</span></h2>' +
      '<span class="p-ntfy-note">One cell = ' + plural(per, "message") + "</span></div>" +
      '<div class="p-ntfy-body"><div class="p-ntfy-sum"><span class="p-num">' + P.fmt(total) + "<sub>MSG</sub></span></div>" +
      '<div class="p-ntfy-grid" role="img" aria-label="' + P.esc("Messages per day, last 7 days: " + said.join(", ")) + '">' + cols + "</div></div>" +
      '<div class="p-ntfy-key" aria-hidden="true"><span><i class="hi"></i>High / urgent</span><span><i class="n"></i>Normal</span><span><i class="lo"></i>Low</span></div>' +
      '<p class="p-ntfy-read">' + read + "</p>");
    if (fresh) { if (P.strike) P.strike(box, 60); else box.classList.add("on"); }
  }

  // ── run ─────────────────────────────────────────────────────
  var busy = false;
  function signature(v) {
    var c = qa('[role="list"] .MuiCard-root[role="listitem"]');
    return location.pathname + "|" + v.kind + "|" + c.length + "|" + (c[0] ? cardKey(c[0]).slice(0, 40) : "");
  }
  function decorate() {
    if (busy) return;
    busy = true;
    try {
      var v = view();
      attr(document.documentElement, "data-p-view", v.kind);
      var sig = signature(v);
      if (sig !== loadedSig) { loadedSig = sig; reloadSoon(); }
      var list = listed(v);
      hooks();
      nosignal();
      cards(list);
      week(list);
    } catch (e) {
      /* a redraw that fails leaves ntfy as it is */
    } finally { busy = false; }
  }
  var queued = false;
  function schedule() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(function () { queued = false; decorate(); });
  }
  function start() {
    new MutationObserver(schedule).observe(document.body, { childList: true, subtree: true, characterData: true });
    load();
    setInterval(load, 30000);
    document.addEventListener("visibilitychange", function () { if (!document.hidden) load(); });
    decorate();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
