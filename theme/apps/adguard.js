/* ─────────────────────────────────────────────────────────────
   Plint Soft UI · AdGuard Home extras
   - the Plint pill goes first in AdGuard's own header row
   - Dashboard hero, from the read-only GET control/stats and
     control/status the browser is already logged in for:
       1 blocked-share ring with a legend
       2 queries over time, allowed vs blocked, stacked bars
       3 average answer time gauge (+ fastest / slowest upstream)
   - tags the Top blocked domains card so its bars turn amber
   Colours: allowed = sage, blocked by filters = amber (warn),
   malware/phishing = rust (bad), adult = slate (info).
   Everything is redrawn idempotently: AdGuard is React and
   re-renders whole pages on navigation and refresh.
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
    return v >= 100 ? Math.round(v) : v >= 10 ? v.toFixed(0) : v.toFixed(1).replace(/\.0$/, "");
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

  // ── drawing ─────────────────────────────────────────────────
  // a ring split into parts, same classes as Plint.ring
  function donut(parts, size, w, label) {
    var r = (size - w) / 2 - 1, c = 2 * Math.PI * r, cx = size / 2;
    var total = 0;
    parts.forEach(function (p) { total += p.v; });
    var live = parts.filter(function (p) { return p.v > 0; }).length;
    var gap = live > 1 ? 3 : 0, off = 0;
    var svg = '<svg class="p-ring" width="' + size + '" height="' + size + '" viewBox="0 0 ' + size + " " + size + '" role="img" aria-label="' + P.esc(label) + '">' +
      '<circle class="p-ring-track" cx="' + cx + '" cy="' + cx + '" r="' + r.toFixed(2) + '" stroke-width="' + w + '"/>';
    if (total) parts.forEach(function (p) {
      if (!p.v) return;
      var len = p.v / total * c, seg = Math.max(len - gap, 2);
      svg += '<circle class="p-ring-val' + (p.tone ? " p-" + p.tone : "") + '" cx="' + cx + '" cy="' + cx + '" r="' + r.toFixed(2) + '" stroke-width="' + w +
        '" stroke-dasharray="' + seg.toFixed(2) + " " + c.toFixed(2) + '" stroke-dashoffset="' + (-off).toFixed(2) + '" transform="rotate(-90 ' + cx + " " + cx + ')"/>';
      off += len;
    });
    return svg + "</svg>";
  }

  // a 270° arc gauge
  function arc(frac, size, w, tone, label) {
    var r = (size - w) / 2 - 1, c = 2 * Math.PI * r, cx = size / 2, a = 0.75 * c;
    var v = Math.max(0, Math.min(1, frac)) * a;
    var rot = ' transform="rotate(135 ' + cx + " " + cx + ')" stroke-linecap="round"';
    return '<svg class="p-ring" width="' + size + '" height="' + size + '" viewBox="0 0 ' + size + " " + size + '" role="img" aria-label="' + P.esc(label) + '">' +
      '<circle class="p-ring-track" cx="' + cx + '" cy="' + cx + '" r="' + r.toFixed(2) + '" stroke-width="' + w + '" stroke-dasharray="' + a.toFixed(2) + " " + c.toFixed(2) + '"' + rot + "/>" +
      (v > 0 ? '<circle class="p-ring-val' + (tone ? " p-" + tone : "") + '" cx="' + cx + '" cy="' + cx + '" r="' + r.toFixed(2) + '" stroke-width="' + w + '" stroke-dasharray="' + Math.max(v, 0.5).toFixed(2) + " " + c.toFixed(2) + '"' + rot + "/>" : "") +
      "</svg>";
  }

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

  // ── hero cards ──────────────────────────────────────────────
  function shareCard(st) {
    var total = +st.num_dns_queries || 0;
    var blk = +st.num_blocked_filtering || 0, sb = +st.num_replaced_safebrowsing || 0, par = +st.num_replaced_parental || 0;
    var allowed = Math.max(total - blk - sb - par, 0);
    var share = pct(blk + sb + par, total);
    var s = S.status, chip = "";
    if (s) {
      if (s.protection_enabled) chip = '<span class="p-ag-chip">Protection on</span>';
      else if (s.protection_disabled_duration > 0) chip = '<span class="p-ag-chip warn">Paused · ' + Math.max(1, Math.round(s.protection_disabled_duration / 60000)) + " min</span>";
      else chip = '<span class="p-ag-chip bad">Protection off</span>';
    }
    var rows = [
      ["", "Allowed", allowed], ["warn", "Blocked by filters", blk],
      ["bad", "Malware / phishing", sb], ["info", "Adult sites", par]
    ];
    return '<div class="p-ag-head"><span class="p-ag-label">Blocked share</span>' + chip + "</div>" +
      '<div class="p-ag-split"><div class="p-ag-dial">' +
      donut([{ v: allowed }, { v: blk, tone: "warn" }, { v: sb, tone: "bad" }, { v: par, tone: "info" }], 120, 16,
        pctTxt(share) + " of " + P.fmt(total) + " queries blocked") +
      "<b>" + pctTxt(share) + "<small>BLOCKED</small></b></div>" +
      '<ul class="p-ag-legend">' + rows.map(function (r) {
        return '<li><i class="' + r[0] + '"></i><span>' + r[1] + "</span><b>" + P.fmt(r[2]) + "<small>" + pctTxt(pct(r[2], total)) + "</small></b></li>";
      }).join("") + "</ul></div>";
  }

  function queriesCard(st) {
    var qs = st.dns_queries || [], n = qs.length, hours = st.time_units === "hours";
    var bf = st.blocked_filtering || [], sb = st.replaced_safebrowsing || [], pa = st.replaced_parental || [];
    var total = +st.num_dns_queries || sum(qs);
    var span = hours ? "last " + n + " hours" : "last " + n + " days";
    var head = '<div class="p-ag-head"><span class="p-ag-label">Queries · ' + span + '</span><span class="p-ag-keys"><span><i></i>Allowed</span><span><i class="warn"></i>Blocked</span></span></div>';
    if (!n) return head + '<div class="p-ag-read">No statistics yet.</div>';

    var max = 0, first = -1;
    for (var i = 0; i < n; i++) { if (qs[i] > max) max = qs[i]; if (first < 0 && qs[i] > 0) first = i; }
    var top = niceMax(max);
    var cols = "";
    for (i = 0; i < n; i++) {
      var t = +qs[i] || 0, b = Math.min(t, (+bf[i] || 0) + (+sb[i] || 0) + (+pa[i] || 0)), a = t - b;
      var d = when(i, n, hours);
      var tip = dayName(d, i, n, hours) + ": " + P.fmt(t) + " queries, " + P.fmt(b) + " blocked";
      cols += '<span class="p-ag-col' + (t ? "" : " nil") + (S.sel === i ? " on" : "") + '" data-i="' + i + '" title="' + P.esc(tip) + '">' +
        '<i class="b" style="height:' + (b / top * 100).toFixed(2) + '%"></i><i class="a" style="height:' + (a / top * 100).toFixed(2) + '%"></i></span>';
    }
    // ticks: the current bucket and every week (or every 6 hours) back from it
    var step = hours ? 6 : 7, ticks = "";
    for (i = n - 1; i >= 0; i -= step) {
      var x = (i + 0.5) / n * 100, cls = i === n - 1 ? "r" : x < 8 ? "l" : "";
      if (i !== n - 1 && x > 82) continue;
      ticks += '<span class="' + cls + '" style="left:' + (cls === "l" ? 0 : x.toFixed(2)) + '%">' + P.esc(tickName(when(i, n, hours), i, n, hours)) + "</span>";
    }
    var k = S.sel != null && S.sel < n ? S.sel : n - 1;
    var kt = +qs[k] || 0, kb = Math.min(kt, (+bf[k] || 0) + (+sb[k] || 0) + (+pa[k] || 0));
    var since = first > 0 && !hours ? " · counting since " + dm(when(first, n, false)) : "";
    return head +
      '<div class="p-ag-total"><strong>' + P.fmt(total) + "</strong><span>queries" + since + "</span></div>" +
      '<div class="p-ag-chart"><div class="p-ag-well" role="img" aria-label="' + P.esc(P.fmt(total) + " queries over the " + span) + '">' +
      '<span class="p-ag-grid" style="top:14px"><em>' + compact(top) + '</em></span><span class="p-ag-grid" style="top:calc(14px + (100% - 24px) / 2)"><em>' + compact(top / 2) + "</em></span>" +
      '<div class="p-ag-bars">' + cols + "</div></div>" +
      '<div class="p-ag-ticks">' + ticks + "</div></div>" +
      '<div class="p-ag-read"><b>' + P.esc(dayName(when(k, n, hours), k, n, hours)) + "</b> · " + P.fmt(kt) + " queries · " + P.fmt(kb) + " blocked" + (kt ? " (" + pctTxt(pct(kb, kt)) + ")" : "") + "</div>";
  }

  function speedCard(st) {
    var sec = +st.avg_processing_time || 0, v = sec * 1000, SCALE = 50;
    var tone = v < 20 ? "" : v < SCALE ? "warn" : "bad";
    var word = !v ? "No data" : v < 20 ? "Fast" : v < SCALE ? "Fair" : "Slow";
    var up = pairs(st.top_upstreams_avg_time).filter(function (u) { return u.v > 0; }).sort(function (a, b) { return a.v - b.v; });
    function fact(label, u) {
      return '<div class="p-ag-fact"><span><small>' + label + '</small><em title="' + P.esc(u.name) + '">' + P.esc(host(u.name)) + "</em></span><b>" + ms(u.v) + " ms</b></div>";
    }
    var facts = up.length ? fact(up.length > 1 ? "Fastest upstream" : "Upstream", up[0]) + (up.length > 1 ? fact("Slowest upstream", up[up.length - 1]) : "") : "";
    return '<div class="p-ag-head"><span class="p-ag-label">Average answer</span><span class="p-ag-chip' + (tone ? " " + tone : "") + '">' + word + "</span></div>" +
      '<div class="p-ag-speed"><div class="p-ag-gauge">' + arc(v / SCALE, 136, 13, tone, "Average answer time " + ms(sec) + " milliseconds") +
      "<b>" + ms(sec) + "<small>MS</small></b>" +
      '<span class="p-ag-scale"><span>0</span><span>' + SCALE + "</span></span></div>" +
      (facts ? '<div class="p-ag-facts">' + facts + "</div>" : "") + "</div>";
  }

  function hero() {
    var dash = q(".row-cards.dashboard, .dashboard");
    var box = q("#p-ag-hero");
    if (!dash) { if (box) box.remove(); return; }
    if (!S.stats) return;
    if (!box) {
      box = P.el("section", "p-ag-hero");
      box.id = "p-ag-hero";
      box.setAttribute("aria-label", "DNS at a glance");
      box.innerHTML = '<div class="p-ag-card p-ag-share"></div><div class="p-ag-card p-ag-queries"></div><div class="p-ag-card p-ag-time"></div>';
      bind(box);
    }
    if (box.nextElementSibling !== dash) dash.parentNode.insertBefore(box, dash);
    set(q(".p-ag-share", box), shareCard(S.stats));
    set(q(".p-ag-queries", box), queriesCard(S.stats));
    set(q(".p-ag-time", box), speedCard(S.stats));
  }

  // hover / tap a bar to read it; leaving goes back to the current bucket
  function bind(box) {
    function pick(e) {
      var c = e.target.closest && e.target.closest(".p-ag-col");
      var i = c ? +c.getAttribute("data-i") : null;
      if (i !== S.sel) { S.sel = i; hero(); }
    }
    box.addEventListener("mouseover", pick);
    box.addEventListener("click", pick);
    box.addEventListener("mouseleave", function () { if (S.sel != null) { S.sel = null; hero(); } });
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
    try { hero(); tables(); } finally { busy = false; }
  }
  var queued = false;
  function schedule() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(function () { queued = false; decorate(); load(false); });
  }
  function start() {
    new MutationObserver(schedule).observe(document.body, { childList: true, subtree: true });
    // AdGuard's own refresh / protection buttons: re-read once they've acted
    document.addEventListener("click", function (e) {
      var b = e.target.closest && e.target.closest(".page-header .btn, .card-options .btn, .dropdown-menu--protection .dropdown-item");
      if (b) setTimeout(function () { load(true); }, 1200);
    }, true);
    window.addEventListener("hashchange", function () { S.sel = null; setTimeout(function () { load(false); }, 50); });
    setInterval(function () { if (!document.hidden) load(true); }, 60000);
    load(true);
    decorate();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
