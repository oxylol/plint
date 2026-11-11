/* ─────────────────────────────────────────────────────────────
   Plint Terminal · Cockpit extras
   plint.js loads this in the shell and in every page iframe.
   - shell: the strip goes into the masthead (#topnav)
   - Overview frame: the Usage card's CPU and memory bars become two
     tick dials with dot-matrix figures and one plain readout line
   - Terminal frame: the xterm canvas takes ground, ink and typeface
     from the sheet (CSS cannot reach a canvas)
   - login page: the Plint mark above the form
   Every other frame gets nothing: no observer, no timers.
   Cockpit's CSP forbids inline style attributes, so this only sets
   classes and SVG attributes, never style="…".
   ───────────────────────────────────────────────────────────── */
(function () {
  "use strict";
  var P = window.Plint;
  if (!P) return;

  var LOGIN = !!document.querySelector("body.login-pf");
  var PATH = "";
  try { PATH = location.pathname; } catch (e) {}
  // The login page has no masthead, so there the strip keeps its default place.
  if (window.top === window && !LOGIN) P.mount = ["#topnav"];

  function q(sel, root) { return (root || document).querySelector(sel); }
  function text(node, s) { if (node && node.textContent !== s) node.textContent = s; }
  function toggle(node, cls, on) { if (node && node.classList.contains(cls) !== on) node.classList.toggle(cls, on); }

  // One rAF-debounced observer per document, pointed at as little as will do.
  // on(key, [[node, options], …]) re-points it only when the key node changes.
  function watcher(fn) {
    var mo = null, at = null, queued = false;
    function run() { queued = false; try { fn(); } catch (e) {} }
    function kick() { if (!queued) { queued = true; requestAnimationFrame(run); } }
    return {
      on: function (key, pairs) {
        if (!key || at === key) return;
        if (mo) mo.disconnect();
        mo = new MutationObserver(kick);
        for (var i = 0; i < pairs.length; i++) if (pairs[i][0]) mo.observe(pairs[i][0], pairs[i][1]);
        at = key;
      }
    };
  }

  // ── Overview · Usage card ───────────────────────────────────
  // Reads one row of the Usage table: the progressbar's aria values and its measure text.
  function reading(th) {
    var row = th && th.closest("tr");
    var bar = row && q('[role="progressbar"]', row);
    if (!bar) return null;
    var now = parseFloat(bar.getAttribute("aria-valuenow")) || 0;
    var min = parseFloat(bar.getAttribute("aria-valuemin")) || 0;
    var max = parseFloat(bar.getAttribute("aria-valuemax")) || 100;
    var m = q(".pf-v5-c-progress__status .pf-v5-c-progress__measure", row);
    return { pct: max > min ? Math.max(0, Math.min(100, (now - min) / (max - min) * 100)) : 0, text: m ? m.textContent.replace(/\s+/g, " ").trim() : "" };
  }
  function gauge(box, key, label, r) {
    var g = q('.p-gauge[data-g="' + key + '"]', box);
    if (!r) { if (g) g.parentNode.removeChild(g); return; }
    var fresh = !g;
    if (fresh) {
      g = P.el("div", "p-gauge p-fl", '<div class="p-dial"><i></i><b></b></div><strong>' + P.lamp(true) + "<span>" + P.esc(label) + "</span></strong>");
      g.setAttribute("data-g", key);
      box.insertBefore(g, q(".p-rd", box));
    }
    var pct = Math.round(r.pct);
    if (g.getAttribute("data-v") !== String(pct)) {
      g.setAttribute("data-v", String(pct));
      q(".p-dial > i", g).innerHTML = P.ring(pct, { size: 112, width: 10, label: label + " " + pct + " percent" });
      P.num(q(".p-dial > b", g), pct, "%");
    }
    toggle(g, "p-hot", pct >= 90);   // nearly full: the lamp blinks
    if (fresh) P.strike(g, key === "cpu" ? 0 : 70);
  }
  var usageWatch = watcher(usage);
  var TREE = { childList: true, subtree: true };
  var VALUES = { subtree: true, characterData: true, attributes: true, attributeFilter: ["aria-valuenow"] };
  function usage() {
    var card = q(".system-usage");
    var body = card && q(".pf-v5-c-card__body", card);
    if (!body) { usageWatch.on(document.body, [[document.body, TREE]]); return; }
    // nodes coming and going anywhere (React may redraw the card), values only inside the card
    usageWatch.on(card, [[document.body, TREE], [card, VALUES]]);
    var cpu = reading(q("#system-usage-cpu-progress", card));
    var mem = reading(q("#system-usage-memory-progress", card));
    var box = q(".p-usage", body);
    if (!cpu && !mem) { toggle(card, "p-has-rings", false); if (box) box.parentNode.removeChild(box); return; }
    if (!box) {
      box = P.el("div", "p-usage", '<p class="p-rd"></p>');
      box.setAttribute("role", "group");
      box.setAttribute("aria-label", "Usage");
      body.insertBefore(box, body.firstChild);
    }
    toggle(card, "p-has-rings", true);
    gauge(box, "cpu", "CPU", cpu);
    gauge(box, "mem", "Memory", mem);
    var line = [];
    if (cpu) line.push("CPU " + (cpu.text || Math.round(cpu.pct) + "%"));
    if (mem) line.push("memory " + (mem.text || Math.round(mem.pct) + "%"));
    text(q(".p-rd", box), line.join(" · "));
  }

  // ── Terminal: xterm draws on a canvas, so it gets its colours here ──
  function rgb(s) { var m = /rgba?\(\s*(\d+)[,\s]+(\d+)[,\s]+(\d+)/.exec(s || ""); return m ? [+m[1], +m[2], +m[3]] : null; }
  function palette() {
    var cs = getComputedStyle(document.body), g = rgb(cs.backgroundColor), i = rgb(cs.color);
    if (!g || !i) return null;
    function mix(a) { return "rgb(" + [0, 1, 2].map(function (k) { return Math.round(g[k] + (i[k] - g[k]) * a); }).join(",") + ")"; }
    var ink = mix(1), mid = mix(0.78), dim = mix(0.58), low = mix(0.36);
    return {
      background: mix(0), foreground: ink, cursor: ink, cursorAccent: mix(0),
      selectionBackground: "rgba(" + i.join(",") + ",0.28)", selectionInactiveBackground: "rgba(" + i.join(",") + ",0.14)",
      black: low, red: ink, green: mid, yellow: ink, blue: mid, magenta: dim, cyan: mid, white: mid,
      brightBlack: dim, brightRed: ink, brightGreen: ink, brightYellow: ink, brightBlue: ink, brightMagenta: mid, brightCyan: ink, brightWhite: ink
    };
  }
  // Cockpit keeps the xterm instance on its React component; reach it from the mount node.
  function xterm(host) {
    var k, f = null, n = 0;
    for (k in host) if (k.indexOf("__reactFiber$") === 0) { f = host[k]; break; }
    while (f && n++ < 40) {
      var s = f.stateNode;
      if (s && s.terminal && s.terminal.options) return s.terminal;
      f = f.return;
    }
    return null;
  }
  var termWatch = watcher(term), termFont = false;
  function term() {
    var host = q(".console-ct"), body = q(".terminal-body");
    var t = host && xterm(host), pal = t && palette();
    if (!t || !pal) { termWatch.on(document.body, [[document.body, TREE]]); return; }
    // Cockpit sets its own theme again when the Appearance menu changes: only that class is watched now
    termWatch.on(body || host, [[body || host, { attributes: true, attributeFilter: ["class"] }]]);
    if (!t.options.theme || t.options.theme.foreground !== pal.foreground || t.options.theme.background !== pal.background) t.options.theme = pal;
    if (termFont) return;
    termFont = true;
    var face = '"Geist Mono"';
    var set = function () {
      try {
        t.options.fontFamily = face + ", ui-monospace, Menlo, Consolas, monospace";
        window.dispatchEvent(new Event("resize"));   // Cockpit refits rows and columns to the new cell
      } catch (e) {}
    };
    if (document.fonts && document.fonts.load) {
      Promise.all([document.fonts.load("400 14px " + face), document.fonts.load("600 14px " + face)]).then(set, function () {});
    }
  }

  // ── login page: the Plint mark above the form ──────────────
  function login() {
    var brand = document.getElementById("brand");
    if (!brand || q(".p-login-head")) return;
    var head = P.el("div", "p-login-head", P.mark("plint-logo p-fl") + '<b class="p-fl">Plint<span class="p-ru">/ ВХОД</span></b>');
    brand.parentNode.insertBefore(head, brand);
    Array.prototype.forEach.call(head.children, function (n, i) { P.strike(n, i * 70); });
  }

  // ── run ─────────────────────────────────────────────────────
  function start() {
    if (!document.body) return;
    try {
      if (LOGIN) login();
      else if (/\/system\/index\.html$/.test(PATH)) usage();
      else if (/\/system\/terminal\.html$/.test(PATH)) term();
    } catch (e) {}
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
