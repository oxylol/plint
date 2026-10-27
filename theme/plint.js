/* ─────────────────────────────────────────────────────────────
   Plint Soft UI · the Plint pill and per-app extras
   Injected by Caddy as <script src="/_plint/plint.js" data-app="…">.
   - Puts a small pill (home · all apps · health) into the app's own
     header, or floats it top-left when the header can't be found.
   - Loads /_plint/apps/<app>.js for that app's infographics.
   - Shares a few drawing helpers on window.Plint.
   ───────────────────────────────────────────────────────────── */
(function () {
  "use strict";
  var me = document.currentScript;
  var app = (me && me.dataset.app) || "";

  // dns.example.net → example.net; the apex stays as is.
  var host = location.hostname;
  var parts = host.split(".");
  var base = parts.length > 2 ? parts.slice(1).join(".") : host;
  function url(sub) { return location.protocol + "//" + (sub ? sub + "." : "") + base + "/"; }

  var ICONS = {
    home: "M3 10.5 12 3l9 7.5V20a1 1 0 0 1-1 1h-5v-6h-6v6H4a1 1 0 0 1-1-1z",
    dns: "M12 3 4 6v6c0 4.5 3.4 8.3 8 9 4.6-.7 8-4.5 8-9V6z",
    status: "M3 12h4l3-8 4 16 3-8h4",
    metrics: "M3 3v18h18M7 15l4-4 3 3 5-6",
    stacks: "m12 3 9 5-9 5-9-5zM3 13l9 5 9-5",
    host: "M5 4h14a2 2 0 0 1 2 2v3a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2zM5 13h14a2 2 0 0 1 2 2v3a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-3a2 2 0 0 1 2-2zM7 7.5h.01M7 16.5h.01",
    backup: "M4 4h16a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1zM5 9v10a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V9M10 13h4",
    ntfy: "M6 8a6 6 0 1 1 12 0c0 7 3 9 3 9H3s3-2 3-9M10.3 21a1.94 1.94 0 0 0 3.4 0",
    games: "M6 6h12a4 4 0 0 1 4 4v4a4 4 0 0 1-4 4H6a4 4 0 0 1-4-4v-4a4 4 0 0 1 4-4zM6 12h4M8 10v4M15 11h.01M18 13h.01",
    stremio: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM10 8.5l5.5 3.5-5.5 3.5z",
    tailscale: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18",
    speed: "M12 14l4-4M3.3 19a9 9 0 1 1 17.4 0",
    tv: "M4 5h16a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1zM8 21h8M12 17v4M10 9l4 2.5-4 2.5z",
    poly: "M3 17l6-6 4 4 8-9M15 6h6v6",
    cloud: "M17.5 19H7a5 5 0 1 1 1.1-9.9A6 6 0 0 1 19.5 11a4 4 0 0 1-2 8z",
    code: "M8 6l-6 6 6 6M16 6l6 6-6 6",
    chat: "M21 12a8 8 0 0 1-11.6 7.1L3 21l1.9-6.4A8 8 0 1 1 21 12z",
    grid: "M4 3h5a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1zM15 3h5a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1h-5a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1zM4 14h5a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1v-5a1 1 0 0 1 1-1zM15 14h5a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1h-5a1 1 0 0 1-1-1v-5a1 1 0 0 1 1-1z"
  };
  function icon(name, cls) {
    return '<svg class="' + (cls || "plint-ico") + '" viewBox="0 0 24 24" aria-hidden="true"><path d="' + ICONS[name] + '"/></svg>';
  }
  var LOGO = '<svg class="plint-logo" viewBox="0 0 128 128" aria-hidden="true">' +
    '<rect x="44" y="26" width="40" height="34" rx="7" opacity=".55"/>' +
    '<rect x="32" y="62" width="64" height="18" rx="6" opacity=".78"/>' +
    '<rect x="18" y="84" width="92" height="20" rx="7"/></svg>';

  // [injection name, subdomain, label, icon]
  var APPS = [
    ["homepage", "", "Home", "home"],
    ["adguard", "dns", "DNS", "dns"],
    ["kuma", "status", "Status", "status"],
    ["beszel", "metrics", "Metrics", "metrics"],
    ["dockge", "stacks", "Stacks", "stacks"],
    ["cockpit", "host", "Host", "host"],
    ["kopia", "backup", "Backups", "backup"],
    ["ntfy", "ntfy", "Notifications", "ntfy"],
    ["speedtest", "speed", "Internet", "speed"],
    ["amp", "games", "Games", "games"],
    ["stremio", "stremio", "Stremio", "stremio"],
    ["tv", "tv", "TV", "tv"],
    ["poly", "poly", "Polybot", "poly"]
  ];

  // Where the pill goes in each app's own header. First match wins.
  var MOUNT = {
    adguard: [".header .header__container", ".header"],
    kuma: ["header"],
    dockge: ["header"],
    beszel: ["#app > div > .container:first-child > div"],
    kopia: [".navbar", "nav"],
    ntfy: ["header .MuiToolbar-root"],
    cockpit: [".masthead-toolbar", "#topnav", "nav"]
  };

  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }

  // ── shared drawing helpers for apps/*.js ────────────────────
  function ring(pct, o) {
    o = o || {};
    var size = o.size || 112, w = o.width || 11, r = (size - w) / 2 - 1, c = 2 * Math.PI * r;
    var v = Math.max(0, Math.min(100, pct || 0)) / 100 * c;
    return '<svg class="p-ring" width="' + size + '" height="' + size + '" viewBox="0 0 ' + size + " " + size + '" role="img" aria-label="' + esc(o.label || Math.round(pct) + "%") + '">' +
      '<circle class="p-ring-track" cx="' + size / 2 + '" cy="' + size / 2 + '" r="' + r + '" stroke-width="' + w + '"/>' +
      '<circle class="p-ring-val' + (o.tone ? " p-" + o.tone : "") + '" cx="' + size / 2 + '" cy="' + size / 2 + '" r="' + r + '" stroke-width="' + w + '" stroke-dasharray="' + v.toFixed(1) + " " + c.toFixed(1) + '" transform="rotate(-90 ' + size / 2 + " " + size / 2 + ')"' + (o.butt ? "" : ' stroke-linecap="round"') + "/></svg>";
  }
  function el(tag, cls, html) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  }
  function fmt(n) { return Number(n).toLocaleString("en-GB"); }

  window.Plint = { app: app, base: base, url: url, icon: icon, ring: ring, el: el, esc: esc, fmt: fmt, health: null, mount: null };

  // ── per-app extras (also inside Cockpit's iframes) ──────────
  if (app) {
    var s = document.createElement("script");
    s.src = "/_plint/apps/" + app + ".js";
    s.defer = true;
    document.head.appendChild(s);
  }

  // The pill belongs to the top window only, and Home is home already.
  if (window.top !== window || app === "homepage") return;

  var pill, menu;
  function build() {
    pill = el("div");
    pill.id = "plint-pill";
    pill.innerHTML =
      '<a class="plint-home" href="' + url("") + '" title="Plint home">' + LOGO + "<span>Plint</span></a>" +
      '<button type="button" class="plint-apps" aria-label="All apps" aria-expanded="false" aria-controls="plint-menu">' + icon("grid") + "</button>" +
      '<a class="plint-status" href="' + url("status") + '" title="Status"><i class="plint-dot"></i><span class="plint-health">…</span></a>';
    menu = el("nav");
    menu.id = "plint-menu";
    menu.setAttribute("aria-label", "Plint apps");
    menu.hidden = true;
    menu.innerHTML = APPS.map(function (a) {
      return '<a href="' + url(a[1]) + '"' + (a[0] === app ? ' aria-current="page"' : "") + '><span class="plint-well">' + icon(a[3]) + "</span>" + esc(a[2]) + "</a>";
    }).join("");
    var btn = pill.querySelector(".plint-apps");
    btn.addEventListener("click", function (e) {
      e.preventDefault();
      e.stopPropagation();
      toggle(menu.hidden);
    });
    document.addEventListener("click", function (e) { if (!menu.hidden && !menu.contains(e.target)) toggle(false); });
    document.addEventListener("keydown", function (e) { if (e.key === "Escape" && !menu.hidden) { toggle(false); btn.focus(); } });
    window.addEventListener("resize", function () { if (!menu.hidden) toggle(false); });
  }
  function toggle(open) {
    var btn = pill.querySelector(".plint-apps");
    if (open) {
      var r = pill.getBoundingClientRect();
      menu.style.top = Math.round(r.bottom + 10) + "px";
      menu.style.left = Math.max(12, Math.min(Math.round(r.left), window.innerWidth - 352)) + "px";
    }
    menu.hidden = !open;
    btn.setAttribute("aria-expanded", open ? "true" : "false");
  }

  function place() {
    if (!document.body) return;
    if (!menu.isConnected) document.body.appendChild(menu);
    // apps/<app>.js may set Plint.mount = [selectors] to override the table.
    var sels = window.Plint.mount || MOUNT[app] || [];
    for (var i = 0; i < sels.length; i++) {
      // Never mount inside our own menu (it is a <nav> too).
      var t = Array.prototype.filter.call(document.querySelectorAll(sels[i]), function (n) {
        return n !== menu && !menu.contains(n) && !pill.contains(n);
      })[0];
      if (t) {
        if (pill.parentNode !== t || t.firstChild !== pill) {
          pill.classList.remove("plint-float");
          t.insertBefore(pill, t.firstChild);
        }
        return;
      }
    }
    if (!pill.isConnected) {
      pill.classList.add("plint-float");
      document.body.appendChild(pill);
    }
  }

  // Up/total from Uptime Kuma's public status page (proxied as /_plint/health).
  function health() {
    fetch("/_plint/health", { credentials: "omit", cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d || !d.heartbeatList) return;
        var ids = Object.keys(d.heartbeatList), up = 0, pending = 0;
        ids.forEach(function (id) {
          var l = d.heartbeatList[id], last = l && l[l.length - 1];
          if (last && (last.status === 1 || last.status === 3)) up++;
          else if (last && last.status === 2) pending++;
        });
        window.Plint.health = d;
        var dot = pill.querySelector(".plint-dot");
        dot.className = "plint-dot " + (up === ids.length ? "ok" : up + pending === ids.length ? "warn" : "bad");
        pill.querySelector(".plint-health").textContent = up + "/" + ids.length;
        pill.querySelector(".plint-status").title = up + " of " + ids.length + " up";
        document.dispatchEvent(new CustomEvent("plint:health", { detail: d }));
      })
      .catch(function () {});
  }

  function start() {
    build();
    place();
    // SPAs re-render their headers; keep the pill where it belongs.
    var pending = false;
    new MutationObserver(function () {
      if (pending) return;
      pending = true;
      requestAnimationFrame(function () { pending = false; place(); });
    }).observe(document.body, { childList: true, subtree: true });
    health();
    setInterval(health, 60000);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
