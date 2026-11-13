/* ─────────────────────────────────────────────────────────────
   Plint Terminal · the Plint strip and per-app extras
   Injected by Caddy as <script src="/_plint/plint.js" data-app="…">.
   - Puts a small strip (home · all apps · health) into the app's own
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
    music: "M2 12c1.7-6 3.3-6 5 0s3.3 6 5 0 3.3-6 5 0 3.3 6 5 0",
    cloud: "M17.5 19H7a5 5 0 1 1 1.1-9.9A6 6 0 0 1 19.5 11a4 4 0 0 1-2 8z",
    code: "M8 6l-6 6 6 6M16 6l6 6-6 6",
    chat: "M21 12a8 8 0 0 1-11.6 7.1L3 21l1.9-6.4A8 8 0 1 1 21 12z",
    grid: "M4 3h5a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1zM15 3h5a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1h-5a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1zM4 14h5a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1v-5a1 1 0 0 1 1-1zM15 14h5a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1h-5a1 1 0 0 1-1-1v-5a1 1 0 0 1 1-1z"
  };
  function icon(name, cls) {
    return '<svg class="' + (cls || "plint-ico") + '" viewBox="0 0 24 24" aria-hidden="true"><path d="' + ICONS[name] + '"/></svg>';
  }
  // The mark: a plinth and two stones, each cut top-left and bottom-right.
  var STONES = [[6, 54, 122, 82, 8], [20, 30, 98, 48, 6], [44, 8, 106, 24, 5]];
  function mark(cls) {
    return '<svg class="' + (cls || "plint-logo") + '" viewBox="0 0 128 88" aria-hidden="true">' + STONES.map(function (s) {
      var x0 = s[0], y0 = s[1], x1 = s[2], y1 = s[3], c = s[4];
      return '<polygon points="' + [x0 + c, y0, x1, y0, x1, y1 - c, x1 - c, y1, x0, y1, x0, y0 + c].join(" ") + '"/>';
    }).join("") + "</svg>";
  }

  // The units, as on the home page: a heading, or [injection name, subdomain, label, port]
  var APPS = [
    ["homepage", "", "Home", ""],
    "EVERYDAY",
    ["amp", "games", "Games", 8080],
    ["table", "table", "D&D Table", 3200],
    ["ai", "ai", "Plint Chat", 5220],
    ["ultron", "ultron", "Ultron", 5230],
    ["cnc", "cnc", "CNC Studio", 5178],
    ["twin", "twin", "CNC Twin", 5210],
    ["poly", "poly", "Polybot", 5190],
    ["hertz", "music", "Hertz", 5200],
    ["accrete", "accrete", "Accrete", 5240],
    ["stremio", "stremio", "Stremio", 8095],
    ["tv", "tv", "TV", 7700],
    "HEALTH",
    ["kuma", "status", "Status", 3001],
    ["adguard", "dns", "AdGuard", 3000],
    ["kopia", "backup", "Backups", 51515],
    ["speedtest", "speed", "Internet", 8075],
    ["ntfy", "ntfy", "Notifications", 8070],
    "SYSTEM",
    ["cockpit", "host", "Host", 9090],
    ["dockge", "stacks", "Stacks", 5001],
    ["beszel", "metrics", "Metrics", 8090]
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
  // A dial of ticks; the lit ones are the value. o: size, width, label, ticks
  function ring(pct, o) {
    o = o || {};
    var size = o.size || 112, w = o.width || 11, r = (size - w) / 2 - 1, c = 2 * Math.PI * r;
    var n = o.ticks || Math.max(12, Math.round(c / 7)), step = c / n, on = step * 0.56, off = step - on;
    var p = Math.max(0, Math.min(100, pct || 0)), k = p > 0 ? Math.max(1, Math.round(p / 100 * n)) : 0, lit = "";
    for (var i = 0; i < k; i++) lit += on.toFixed(2) + " " + off.toFixed(2) + " ";
    var at = ' cx="' + size / 2 + '" cy="' + size / 2 + '" r="' + r + '" stroke-width="' + w + '" transform="rotate(-90 ' + size / 2 + " " + size / 2 + ')"';
    return '<svg class="p-ring" width="' + size + '" height="' + size + '" viewBox="0 0 ' + size + " " + size + '" role="img" aria-label="' + esc(o.label || Math.round(pct) + "%") + '">' +
      '<circle class="p-ring-track"' + at + ' stroke-dasharray="' + on.toFixed(2) + " " + off.toFixed(2) + '"/>' +
      '<circle class="p-ring-val' + (o.tone ? " p-" + o.tone : "") + '"' + at + ' stroke-dasharray="' + lit + "0 " + (c + 1).toFixed(1) + '"/></svg>';
  }
  function el(tag, cls, html) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  }
  function fmt(n) { return Number(n).toLocaleString("en-GB"); }

  // ── light: every lamp strikes on by itself, with a stutter; nothing fades ──
  function live() { return !(window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches); }
  function rnd(a, b) { return a + Math.random() * (b - a); }
  // Switch one element on (give it class p-fl first to keep it dark until then).
  function strike(node, delay) {
    node.classList.add("on");
    if (!live() || !node.animate) return;
    var ev = [[0, 1]], t = 0;
    for (var k = Math.floor(rnd(1, 3.5)); k > 0; k--) { t += rnd(30, 85); ev.push([t, 0]); t += rnd(25, 95); ev.push([t, 1]); }
    var T = t + 40;
    node.animate([{ opacity: 0, offset: 0, easing: "step-end" }].concat(ev.map(function (e, i) {
      return { opacity: e[1] ? (i < ev.length - 1 ? rnd(0.45, 1) : 1) : rnd(0, 0.16), offset: (e[0] + 20) / T, easing: "step-end" };
    })), { duration: T, delay: delay || 0, fill: "backwards" });
  }
  // A dot-matrix figure in `node`: each digit is its own light and spins when the figure changes.
  function num(node, value, unit) {
    var v = String(value), key = v + "|" + (unit || "");
    if (node.dataset.k === key) return;
    var first = !node.dataset.k;
    node.dataset.k = key;
    node.classList.add("p-num");
    node.innerHTML = v.split("").map(function (ch) { return '<span data-v="' + esc(ch) + '">' + esc(ch) + "</span>"; }).join("") + (unit ? "<sub>" + esc(unit) + "</sub>" : "");
    if (first || !live()) return;
    Array.prototype.forEach.call(node.querySelectorAll("span"), function (sp) {
      if (!/\d/.test(sp.dataset.v)) return;
      var id = setInterval(function () { sp.textContent = Math.floor(Math.random() * 10); }, 50);
      setTimeout(function () { clearInterval(id); sp.textContent = sp.dataset.v; }, rnd(160, 420));
    });
  }
  function meter(pct) { return '<div class="p-meter" style="--v:' + Math.max(0, Math.min(100, pct || 0)).toFixed(1) + '%"></div>'; }
  // state: true lit · false hollow · "alarm" blinking
  function lamp(state) { return '<i class="p-blk' + (state === "alarm" ? " alarm" : state && state !== "off" ? "" : " off") + '"></i>'; }

  window.Plint = { app: app, base: base, url: url, icon: icon, mark: mark, ring: ring, el: el, esc: esc, fmt: fmt, live: live, strike: strike, num: num, meter: meter, lamp: lamp, health: null, mount: null };

  // One ground colour for the browser chrome, whatever the app asks for.
  // The app's own tags lose their name but stay in the page: some apps (Uptime Kuma) keep writing to theirs.
  function chrome() {
    Array.prototype.forEach.call(document.querySelectorAll('meta[name="theme-color"]'), function (m) { if (!m.dataset.plint) m.removeAttribute("name"); });
    if (!document.querySelector("meta[data-plint]")) {
      var m = document.createElement("meta");
      m.name = "theme-color"; m.content = "#0f0e11"; m.dataset.plint = "1";
      document.head.appendChild(m);
    }
  }
  chrome();

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
      '<a class="plint-home" href="' + url("") + '" title="Plint home">' + mark() + "<span>Plint</span></a>" +
      '<button type="button" class="plint-apps" aria-label="All apps" aria-expanded="false" aria-controls="plint-menu">Apps</button>' +
      '<a class="plint-status" href="' + url("status") + '" title="Status"><i class="plint-dot"></i><span class="plint-health">…</span></a>';
    menu = el("nav");
    menu.id = "plint-menu";
    menu.setAttribute("aria-label", "Plint apps");
    menu.hidden = true;
    menu.innerHTML = APPS.map(function (a) {
      if (typeof a === "string") return "<h4>" + a + "</h4>";
      return '<a href="' + url(a[1]) + '"' + (a[0] === app ? ' aria-current="page"' : "") + "><span>" + esc(a[2]) + "</span><small>" + (a[3] ? ":" + a[3] : "") + "</small></a>";
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
      menu.style.top = Math.round(r.bottom + 8) + "px";
      menu.style.left = Math.max(10, Math.min(Math.round(r.left), window.innerWidth - 310)) + "px";
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
        var st = pill.querySelector(".plint-status");
        st.title = up + " of " + ids.length + " up";
        st.classList.toggle("bad", up + pending < ids.length);
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
    setTimeout(chrome, 2500);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
