/* ─────────────────────────────────────────────────────────────
   Plint Terminal · AdGuard Home extras
   - the Plint strip goes first in AdGuard's own header row
   - three instruments above the Dashboard, from the read-only GET
     control/stats and control/status the browser is already logged
     in for:
       1 FILTER   the blocked share on a tick dial, with a legend
       2 QUERIES  queries over time: allowed solid, blocked hatched
       3 ANSWER   average answer time on a dot meter (+ fastest /
                  slowest upstream)
     each ends in one plain line that says what the drawing shows
   - tags the Top blocked domains card so its bars read as ticks
   No colour: on = a lit lamp, a warning blinks, a fault is inverse.
   The frame of each instrument is built once and only its slots
   are rewritten (AdGuard is React and re-renders whole pages), so
   the figures can spin when they change. The page is watched only
   while the Dashboard is the route.
   ───────────────────────────────────────────────────────────── */
(function () {
  "use strict";
  var P = window.Plint;
  if (!P) return;

  P.mount = [".header .header__row", ".header .header__container"];

  function q(sel, root) { return (root || document).querySelector(sel); }
  function qa(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }
  // set-only-if-changed; compares with what we last wrote, because SVG
  // never serialises back to the same string (<circle/> → <circle></circle>)
  function set(node, html) { if (node && node._plint !== html) { node._plint = html; node.innerHTML = html; } }
  function sum(a) { var s = 0; for (var i = 0; i < (a || []).length; i++) s += +a[i] || 0; return s; }
  function pct(v, t) { return t ? v / t * 100 : 0; }
  function pctTxt(p) { return p > 0 && p < 0.1 ? "<0.1%" : (p > 0 && p < 10 ? p.toFixed(1) : Math.round(p)) + "%"; }
  function compact(n) {
    if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1).replace(/\.0$/, "") + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(n >= 1e4 ? 0 : 1).replace(/\.0$/, "") + "k";
    return String(Math.round(n));
  }
  function ms(sec) {
    var v = (+sec || 0) * 1000;
    return v >= 100 ? String(Math.round(v)) : v >= 10 ? v.toFixed(0) : v.toFixed(1).replace(/\.0$/, "");
  }
  function host(u) { return String(u).replace(/^[a-z]+:\/\//i, "").replace(/(:\d+)?(\/.*)?$/, ""); }
  function pairs(list) {
    return (list || []).map(function (o) { var k = Object.keys(o)[0]; return { name: k, v: +o[k] || 0 }; });
  }

  // ── data ────────────────────────────────────────────────────
  var S = { stats: null, status: null, at: 0, sel: null };

  function get(u) {
    return fetch(u, { cache: "no-store", credentials: "same-origin" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .catch(function () { return null; });
  }
  function load(force) {
    if (!q(".dashboard")) return;
    if (!force && Date.now() - S.at < 25000) return;
    S.at = Date.now();
    Promise.all([get("control/stats"), get("control/status")]).then(function (r) {
      if (r[0]) S.stats = r[0];
      if (r[1]) S.status = r[1];
      decorate();
    });
  }

  // ── time ────────────────────────────────────────────────────
  function niceMax(n) {
    if (n <= 0) return 1;
    var e = Math.pow(10, Math.floor(Math.log10(n))), m = n / e;
    return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * e;
  }
  // when bucket i (of n) started: the last bucket is the current day/hour
  function when(i, n, hours) {
    var d = new Date();
    if (hours) { d.setMinutes(0, 0, 0); d.setHours(d.getHours() - (n - 1 - i)); }
    else { d.setHours(0, 0, 0, 0); d.setDate(d.getDate() - (n - 1 - i)); }
    return d;
  }
  var MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  function dm(d) { return d.getDate() + " " + MON[d.getMonth()]; }
  function hm(d) { return ("0" + d.getHours()).slice(-2) + ":00"; }
  function dayName(d, i, n, hours) {
    if (hours) return i === n - 1 ? "This hour" : hm(d);
    if (i === n - 1) return "Today";
    if (i === n - 2) return "Yesterday";
    return dm(d);
  }
  function tickName(d, i, n, hours) {
    if (hours) return i === n - 1 ? "Now" : hm(d);
    return i === n - 1 ? "Today" : dm(d);
  }

  // ── the instruments ─────────────────────────────────────────
  // state at the right of a heading: "on" lit lamp, "warn" blinking lamp, "bad" inverse video, "" plain
  function state(node, kind, text) {
    set(node, !text ? "" : kind === "bad" ? '<span class="p-inv">' + P.esc(text) + "</span>" :
      (kind ? P.lamp(kind === "warn" ? "alarm" : true) : "") + "<span>" + P.esc(text) + "</span>");
  }
  function read(card, text) { set(q(".p-ag-read", card), '<span class="p-dim">&gt;</span> ' + text); }

  function shareCard(c, st) {
    var total = +st.num_dns_queries || 0;
    var blk = +st.num_blocked_filtering || 0, sb = +st.num_replaced_safebrowsing || 0, par = +st.num_replaced_parental || 0;
    var stopped = blk + sb + par, allowed = Math.max(total - stopped, 0), share = pct(stopped, total);
    var s = S.status;
    if (!s) state(q(".p-ag-state", c), "", "");
    else if (s.protection_enabled) state(q(".p-ag-state", c), "on", "Protection on");
    else if (s.protection_disabled_duration > 0) state(q(".p-ag-state", c), "warn", "Paused · " + Math.max(1, Math.round(s.protection_disabled_duration / 60000)) + " min");
    else state(q(".p-ag-state", c), "bad", "Protection off");
    set(q(".p-ag-ring", c), P.ring(share, { size: 120, width: 12, label: pctTxt(share) + " of " + P.fmt(total) + " queries blocked" }));
    P.num(q(".p-ag-fig", c), share > 0 && share < 0.1 ? "<0.1" : share > 0 && share < 10 ? share.toFixed(1) : Math.round(share), "%");
    // long name, and a short one for a narrow screen (the sheet shows one of them)
    var rows = [["", "Allowed", "Allowed", allowed], ["on", "Blocked by filters", "Filters", blk], ["on", "Malware / phishing", "Malware", sb], ["on", "Adult sites", "Adult", par]];
    set(q(".p-ag-legend", c), rows.map(function (r) {
      return '<li><i class="' + r[0] + '"></i><span title="' + r[1] + '"><span class="w">' + r[1] + '</span><span class="n">' + r[2] + "</span></span><b>" + P.fmt(r[3]) + "</b><small>" + pctTxt(pct(r[3], total)) + "</small></li>";
    }).join(""));
    read(c, total ? P.fmt(stopped) + " of " + P.fmt(total) + " queries stopped" : "No queries counted yet");
  }

  function blockedAt(st, i) {
    return Math.min(+(st.dns_queries || [])[i] || 0,
      (+(st.blocked_filtering || [])[i] || 0) + (+(st.replaced_safebrowsing || [])[i] || 0) + (+(st.replaced_parental || [])[i] || 0));
  }
  function queriesCard(c, st) {
    var qs = st.dns_queries || [], n = qs.length, hours = st.time_units === "hours";
    var total = +st.num_dns_queries || sum(qs);
    state(q(".p-ag-state", c), "", n ? "Last " + n + (hours ? " hours" : " days") : "");
    P.num(q(".p-ag-fig", c), P.fmt(total));
    if (!n) { set(q(".p-ag-sub", c), "queries"); set(q(".p-ag-chart", c), ""); read(c, "No statistics yet"); return; }

    var max = 0, first = -1, i;
    for (i = 0; i < n; i++) { if (qs[i] > max) max = qs[i]; if (first < 0 && qs[i] > 0) first = i; }
    set(q(".p-ag-sub", c), "queries" + (first > 0 && !hours ? " · counting since " + dm(when(first, n, false)) : ""));
    var top = niceMax(max), cols = "";
    for (i = 0; i < n; i++) {
      var t = +qs[i] || 0, b = blockedAt(st, i), a = t - b;
      cols += '<span class="p-ag-col" data-i="' + i + '">' +
        (a ? '<i class="a" style="height:' + (a / top * 100).toFixed(2) + '%"></i>' : "") +
        (b ? '<i class="b" style="height:' + (b / top * 100).toFixed(2) + '%"></i>' : "") + "</span>";
    }
    // ticks: the current bucket and every week (or every 6 hours) back from it
    var step = hours ? 6 : 7, ticks = "";
    for (i = n - 1; i >= 0; i -= step) {
      var x = (i + 0.5) / n * 100, cls = i === n - 1 ? "r" : x < 8 ? "l" : "";
      if (i !== n - 1 && x > 80) continue;
      ticks += '<span class="' + cls + '" style="left:' + (cls === "l" ? 0 : x.toFixed(2)) + '%">' + P.esc(tickName(when(i, n, hours), i, n, hours)) + "</span>";
    }
    set(q(".p-ag-chart", c),
      '<div class="p-ag-well" role="img" aria-label="' + P.esc(P.fmt(total) + " queries over the last " + n + (hours ? " hours" : " days")) + '">' +
      '<span class="p-ag-grid t"><em>' + compact(top) + '</em></span><span class="p-ag-grid m"><em>' + compact(top / 2) + "</em></span>" +
      '<div class="p-ag-bars">' + cols + "</div></div>" +
      '<div class="p-ag-ticks">' + ticks + "</div>");
    pick(c, st);
  }
  // the picked column (hover / tap), or the current bucket: lit in the chart, read out below
  function pick(c, st) {
    c = c || q("#p-ag-hero .p-ag-queries");
    st = st || S.stats;
    if (!c || !st) return;
    var qs = st.dns_queries || [], n = qs.length, hours = st.time_units === "hours";
    if (!n) return;
    if (S.sel != null && S.sel >= n) S.sel = null;
    qa(".p-ag-col", c).forEach(function (col, i) {
      var on = S.sel === i;
      if (col.classList.contains("on") !== on) col.classList.toggle("on", on);
    });
    var k = S.sel != null ? S.sel : n - 1, kt = +qs[k] || 0, kb = blockedAt(st, k);
    read(c, "<b>" + P.esc(dayName(when(k, n, hours), k, n, hours)) + "</b> · " + P.fmt(kt) + " queries · " + P.fmt(kb) + " blocked" + (kt ? " (" + pctTxt(pct(kb, kt)) + ")" : ""));
  }

  function speedCard(c, st) {
    var sec = +st.avg_processing_time || 0, v = sec * 1000, SCALE = 50;
    if (!v) state(q(".p-ag-state", c), "", "No data");
    else if (v < 20) state(q(".p-ag-state", c), "on", "Fast");
    else if (v < SCALE) state(q(".p-ag-state", c), "warn", "Fair");
    else state(q(".p-ag-state", c), "bad", "Slow");
    P.num(q(".p-ag-fig", c), ms(sec), "MS");
    set(q(".p-ag-m", c), P.meter(v / SCALE * 100));
    var up = pairs(st.top_upstreams_avg_time).filter(function (u) { return u.v > 0; }).sort(function (a, b) { return a.v - b.v; });
    function fact(label, u) {
      return '<div class="p-ag-fact"><small>' + label + '</small><em title="' + P.esc(u.name) + '">' + P.esc(host(u.name)) + "</em><b>" + ms(u.v) + " ms</b></div>";
    }
    set(q(".p-ag-facts", c), up.length ? fact(up.length > 1 ? "Fastest upstream" : "Upstream", up[0]) + (up.length > 1 ? fact("Slowest upstream", up[up.length - 1]) : "") : "");
    read(c, !v ? "No answers timed yet" : v < 20 ? "Fast · under 20 ms on average" : v < SCALE ? "Fair · 20 to " + SCALE + " ms on average" : "Slow · over " + SCALE + " ms on average");
  }

  function frame(cls, en, ru, body) {
    return '<div class="p-ag-card p-ag-' + cls + ' p-fl"><div class="p-ag-head"><span class="p-ag-label">' + en + ' <span class="p-ru">/ ' + ru + '</span></span><span class="p-ag-state"></span></div>' +
      body + '<div class="p-ag-read"></div></div>';
  }
  function hero() {
    var dash = q(".row-cards.dashboard, .dashboard");
    var box = q("#p-ag-hero");
    if (!dash) { if (box) box.remove(); return; }
    if (!S.stats) return;
    var fresh = !box;
    if (fresh) {
      box = P.el("section", "p-ag-hero");
      box.id = "p-ag-hero";
      box.setAttribute("aria-label", "DNS at a glance");
      box.innerHTML =
        frame("share", "Filter", "фильтр", '<div class="p-ag-split"><div class="p-ag-dial"><span class="p-ag-ring"></span><b><span class="p-ag-fig"></span><small>Blocked</small></b></div><ul class="p-ag-legend"></ul></div>') +
        frame("queries", "Queries", "запросы", '<div class="p-ag-total"><span class="p-ag-fig"></span><span class="p-ag-sub"></span><span class="p-ag-keys"><span><i class="a"></i>Allowed</span><span><i class="b"></i>Blocked</span></span></div><div class="p-ag-chart"></div>') +
        frame("time", "Answer", "ответ", '<div class="p-ag-speed"><span class="p-ag-fig"></span><div class="p-ag-gauge"><span class="p-ag-m"></span><div class="p-ag-scale"><span>0</span><span>25</span><span>50 ms</span></div></div></div><div class="p-ag-facts"></div>');
      bind(box);
    }
    if (box.nextElementSibling !== dash) dash.parentNode.insertBefore(box, dash);
    shareCard(q(".p-ag-share", box), S.stats);
    queriesCard(q(".p-ag-queries", box), S.stats);
    speedCard(q(".p-ag-time", box), S.stats);
    if (fresh) qa(".p-ag-card", box).forEach(function (card, i) { P.strike(card, i * 60); });
  }

  // hover / tap a bar to read it; leaving goes back to the current bucket
  function bind(box) {
    function over(e) {
      var c = e.target.closest && e.target.closest(".p-ag-col");
      var i = c ? +c.getAttribute("data-i") : null;
      if (i !== S.sel) { S.sel = i; pick(); }
    }
    box.addEventListener("mouseover", over);
    box.addEventListener("click", over);
    box.addEventListener("mouseleave", function () { if (S.sel != null) { S.sel = null; pick(); } });
  }

  // ── tables ──────────────────────────────────────────────────
  // Top blocked domains is found by its contents, not its (translated) title.
  function tables() {
    if (!S.stats) return;
    var blocked = pairs(S.stats.top_blocked_domains).slice(0, 3).map(function (d) { return d.name; });
    qa(".dashboard .card").forEach(function (card) {
      var names = qa(".rt-tbody .rt-tr-group", card).slice(0, 3).map(function (g) {
        var t = q(".logs__text", g);
        return t ? (t.getAttribute("title") || t.textContent).trim() : "";
      });
      var hit = !!blocked.length && names.length === Math.min(3, blocked.length) && names.every(function (n, i) { return n === blocked[i]; });
      if (hit !== card.classList.contains("p-ag-blocked")) card.classList.toggle("p-ag-blocked", hit);
    });
  }

  // ── run ─────────────────────────────────────────────────────
  var busy = false;
  function decorate() {
    if (busy) return;
    busy = true;
    try { hero(); tables(); } catch (e) { /* the app's DOM moved: stay quiet */ } finally { busy = false; }
  }
  var queued = false;
  function schedule() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(function () { queued = false; decorate(); load(false); });
  }
  // the page is watched only while the Dashboard is the route; our own redraws are not news
  var mo = null, watching = false;
  function home() { return /^#?\/?$/.test(location.hash) || !!q(".dashboard"); }
  function watch() {
    var want = home();
    if (want === watching) return;
    watching = want;
    if (!mo) mo = new MutationObserver(function (list) {
      for (var i = 0; i < list.length; i++) {
        var t = list[i].target;
        if (!t.closest || !t.closest("#p-ag-hero")) { schedule(); return; }
      }
    });
    if (want) mo.observe(document.body, { childList: true, subtree: true });
    else mo.disconnect();
  }
  function start() {
    watch();
    // AdGuard's own refresh / protection buttons: re-read once they've acted
    document.addEventListener("click", function (e) {
      var b = e.target.closest && e.target.closest(".page-header .btn, .card-options .btn, .dropdown-menu--protection .dropdown-item");
      if (b && watching) setTimeout(function () { load(true); }, 1200);
    }, true);
    window.addEventListener("hashchange", function () {
      S.sel = null;
      [50, 400, 1500].forEach(function (t) { setTimeout(function () { watch(); decorate(); load(false); }, t); });
    });
    setInterval(function () { if (!document.hidden && watching) load(true); }, 60000);
    load(true);
    decorate();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
