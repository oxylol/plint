/* ─────────────────────────────────────────────────────────────
   Plint Soft UI · ntfy extras
   - the Plint pill goes into ntfy's own toolbar (Plint.mount)
   - messages per day, last 7 days, above the notification list
   - notifications become a timeline: day headings, priority chips,
     tags as chips (all via data attributes that apps/ntfy.css draws)
   - line icons instead of ntfy's teal logo
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
  function dbName() {
    try {
      var u = localStorage.getItem("user"), t = localStorage.getItem("token");
      return u && t ? "ntfy-" + u : "ntfy";
    } catch (e) { return "ntfy"; }
  }
  function load() {
    if (loading || !window.indexedDB) return;
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

  // ── header: a bell in a pressed well before the title ───────
  function header() {
    var tb = q("header.MuiAppBar-root > .MuiToolbar-root");
    if (!tb) return;
    var title = q(":scope > .MuiTypography-h6", tb);
    var mark = q(":scope > .p-ntfy-mark", tb);
    if (title && !mark) {
      mark = P.el("span", "p-ntfy-mark", P.icon("ntfy", ""));
      mark.setAttribute("aria-hidden", "true");
      tb.insertBefore(mark, title);
    } else if (title && mark && mark.nextElementSibling !== title && mark.nextElementSibling && !mark.nextElementSibling.matches("img")) {
      tb.insertBefore(mark, title);
    }
  }

  // ── empty states and sign-in: our bell instead of ntfy's logo ─
  function marks() {
    qa("#main > .MuiContainer-root > h5.MuiTypography-alignCenter").forEach(function (h5) {
      var box = h5.parentNode;
      if (!q(":scope > .p-ntfy-hero", box)) {
        var hero = P.el("div", "p-ntfy-hero", P.icon("ntfy", ""));
        hero.setAttribute("aria-hidden", "true");
        box.insertBefore(hero, h5);
      }
    });
    qa("#root > .MuiBox-root > a > .MuiAvatar-root").forEach(function (av) {
      if (!q("svg", av)) av.insertAdjacentHTML("beforeend", P.icon("ntfy", ""));
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

  // ── messages per day, last 7 days ───────────────────────────
  function week(list) {
    var scroller = q("#main > .infinite-scroll-component__outerdiv");
    var box = q("#main > .p-ntfy-week");
    var today = dayStart(new Date()), days = [];
    for (var i = 6; i >= 0; i--) days.push({ at: today - i * DAY, n: 0, hi: 0, max: 0 });
    var total = 0, hi = 0, max = 0;
    (list || []).forEach(function (n) {
      var idx = 6 - Math.round((today - dayKey(n.time)) / DAY);
      if (idx < 0 || idx > 6) return;
      var p = n.priority || 3, d = days[idx];
      if (p === 5) { d.max++; max++; } else if (p === 4) { d.hi++; hi++; } else d.n++;
      total++;
    });
    if (!scroller || !list || !total) { if (box) box.remove(); return; }
    if (!box) {
      box = P.el("section", "p-ntfy-week");
      box.setAttribute("aria-label", "Messages per day");
    }
    if (box.nextElementSibling !== scroller) scroller.parentNode.insertBefore(box, scroller);
    var top = Math.max.apply(null, days.map(function (d) { return d.n + d.hi + d.max; })) || 1;
    var said = [], cols = days.map(function (d, k) {
      var sum = d.n + d.hi + d.max;
      var name = k === 6 ? "Today" : new Date(d.at).toLocaleDateString("en-GB", { weekday: "short" });
      said.push(name + " " + sum);
      function seg(v, cls) { return v ? '<i class="' + cls + '" style="height:' + (v / top * 100).toFixed(1) + '%"></i>' : ""; }
      return '<div class="p-ntfy-day' + (k === 6 ? " today" : "") + '"><span class="p-ntfy-n' + (sum ? "" : " zero") + '">' + sum + "</span>" +
        '<span class="p-ntfy-col">' + seg(d.n, "ok") + seg(d.hi, "warn") + seg(d.max, "bad") + "</span>" +
        '<span class="p-ntfy-d">' + P.esc(name) + "</span></div>";
    }).join("");
    var facts = "<span><b>" + (days[6].n + days[6].hi + days[6].max) + "</b>today</span>" +
      (hi ? '<span class="warn"><b>' + hi + "</b>high</span>" : "") +
      (max ? '<span class="bad"><b>' + max + "</b>urgent</span>" : "");
    set(box, "<div>" +
      '<div class="p-ntfy-sum"><span class="p-ntfy-label">Last 7 days</span>' +
      '<span class="p-ntfy-big">' + P.fmt(total) + "<small>" + (total === 1 ? "message" : "messages") + "</small></span>" +
      '<div class="p-ntfy-facts">' + facts + "</div></div>" +
      '<div class="p-ntfy-well" role="img" aria-label="' + P.esc("Messages per day, last 7 days: " + said.join(", ")) + '">' + cols + "</div>" +
      "</div>");
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
      header();
      marks();
      cards(list);
      week(list);
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
    decorate();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
