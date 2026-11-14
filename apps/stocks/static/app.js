/* Stocks · the page. It reads /api/state once a minute and /api/model/<id> when a model is opened (PAGE.md),
   and nothing else. Every string from the API reaches the page as a text node, never as HTML: the agent's memo
   and reads are model output. Structure and behaviour are here and in app.css; the Plint Terminal look is
   theme/apps/stocks.css, which Caddy slips in together with the Plint strip. */
(function () {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";
  var NBSP = "\u00a0", MINUS = "\u2212", DASH = "\u2013", TIMES = "\u00d7";
  var MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  var MONTH = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
  var DOW = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  var EVERY = 60000;           // how often /api/state is read
  var SHOW = 12;               // rows of a long table before "show all"

  var S = {
    state: null, raw: "",      // the last good /api/state, and its text (to see whether anything changed)
    fail: null,                // { since } while the engine cannot be reached
    busy: false, timer: 0, tried: 0,
    broken: null,              // what went wrong if the page itself could not draw what it was sent
    open: null,                // id of the open model
    book: Object.create(null), // id -> { data, raw, err, busy }: the last good /api/model/<id>
    ui: { fold: Object.create(null), more: Object.create(null) },   // folds that are open, lists shown in full: kept across refreshes
    boardY: 0,                 // where the overview was scrolled to when a model was opened
    charts: [],                // redraws of the charts now on the page (they follow the width)
    painted: { board: false, book: null }, fresh: 0
  };

  /* ── small tools ─────────────────────────────────────────────────────────────────────────────────────── */

  function $(id) { return document.getElementById(id); }
  function isNum(x) { return typeof x === "number" && isFinite(x); }
  function arr(x) { return Array.isArray(x) ? x : []; }
  function obj(x) { return x && typeof x === "object" && !Array.isArray(x) ? x : {}; }
  /* text from the API: a string or a number, anything else is nothing (never "null" or "undefined") */
  function txt(x) { return typeof x === "string" ? x : isNum(x) ? String(x) : ""; }
  /* a word from one of our own lists, by a key that came from the API */
  function pick(map, k) { return typeof k === "string" && Object.prototype.hasOwnProperty.call(map, k) ? map[k] : null; }

  /* picture characters (the agent may write them) get a span of their own, so the sheet can take their colour away */
  var PICTURE = null;
  try { PICTURE = new RegExp("((?:\\p{Extended_Pictographic}|[\\u{1F1E6}-\\u{1F1FF}])(?:\\uFE0F|\\u200D|\\p{Extended_Pictographic}|[\\u{1F3FB}-\\u{1F3FF}\\u{1F1E6}-\\u{1F1FF}])*)", "u"); } catch (e) { /* an old browser: they keep their colours */ }
  function words(el, t) {
    if (!PICTURE || el.namespaceURI === SVGNS || !PICTURE.test(t)) { el.appendChild(document.createTextNode(t)); return; }
    t.split(PICTURE).forEach(function (part, i) {
      if (!part) return;
      if (i % 2) { var sp = document.createElement("span"); sp.className = "emo"; sp.textContent = part; el.appendChild(sp); }
      else el.appendChild(document.createTextNode(part));
    });
  }
  function put(el, c) {
    if (c == null || c === false || c === true) return;
    if (Array.isArray(c)) { for (var i = 0; i < c.length; i++) put(el, c[i]); return; }
    if (c.nodeType) el.appendChild(c); else words(el, String(c));
  }
  function attrs(el, a) {
    if (!a) return;
    for (var k in a) {
      var v = a[k];
      if (v == null || v === false) continue;
      if (k === "css") { for (var p in v) el.style.setProperty(p, v[p]); }
      else el.setAttribute(k, v === true ? "" : String(v));
    }
  }
  /* h("p", {class: "x"}, "text", node, [more]): children are text nodes or nodes, nothing is parsed as HTML */
  function h(tag, a) {
    var el = document.createElement(tag);
    attrs(el, a);
    for (var i = 2; i < arguments.length; i++) put(el, arguments[i]);
    return el;
  }
  function s(tag, a) {
    var el = document.createElementNS(SVGNS, tag);
    attrs(el, a);
    for (var i = 2; i < arguments.length; i++) put(el, arguments[i]);
    return el;
  }
  function fill(box, kids) {
    while (box.firstChild) box.removeChild(box.firstChild);
    put(box, kids);
  }

  /* ── figures ─────────────────────────────────────────────────────────────────────────────────────────── */

  function round(v, dp) { var f = Math.pow(10, dp), r = Math.round(v * f) / f; return r === 0 ? 0 : r; }
  /* thousands separators; a figure too long for any cell is shortened (12,300,000 becomes 12.3M) */
  function nf(v, dp) {
    var a = Math.abs(v);
    if (a >= 1e7) { try { return a.toLocaleString("en-US", { notation: "compact", maximumFractionDigits: 1 }); } catch (e) { return a.toExponential(1); } }
    return a.toLocaleString("en-US", { minimumFractionDigits: dp, maximumFractionDigits: dp });
  }
  function signed(v, dp) { var r = round(v, dp); return (r > 0 ? "+" : r < 0 ? MINUS : "") + nf(r, dp); }
  function plain(v, dp) { var r = round(v, dp); return (r < 0 ? MINUS : "") + nf(r, dp); }
  function tone(r) { return r > 0 ? "pos" : r < 0 ? "neg" : "zero"; }

  /* the digits of a figure. The point and the comma get a span of their own: the dot-matrix face borrows them
     from the text face, where they sit too wide, and the sheet pulls them in. */
  function digits(t) {
    return t.split(/([.,])/).map(function (x) { return x === "." || x === "," ? h("span", { class: "dp" }, x) : x || null; });
  }

  /* nothing yet: a dim dash, never a word */
  function nil() { return h("span", { class: "nil" }, h("span", { "aria-hidden": "true" }, DASH), h("span", { class: "sr" }, "nothing yet")); }

  /* a fraction as a percent. o.tone: a loss is marked, a win is lit (the money convention); o.sign: false for shares */
  function pct(x, o) {
    o = o || {};
    if (!isNum(x)) return nil();
    var dp = o.dp == null ? 2 : o.dp, r = round(x * 100, dp);
    return h("span", { class: "n " + (o.tone ? tone(r) : "flat") }, digits(o.sign === false ? plain(r, dp) : signed(r, dp)), h("span", { class: "u" }, NBSP + "%"));
  }
  function usd(x, dp) {
    if (!isNum(x)) return nil();
    return h("span", { class: "n flat" }, (x < 0 ? MINUS : "") + "$" + nf(x, dp == null ? 2 : dp));
  }
  function count(x) { return isNum(x) ? h("span", { class: "n flat" }, plain(x, 0)) : nil(); }
  function fig(x, dp, sign) { return isNum(x) ? h("span", { class: "n flat" }, sign ? signed(x, dp) : plain(x, dp)) : nil(); }
  function usdText(x, dp) { return (x < 0 ? MINUS : "") + "$" + nf(x, dp == null ? 2 : dp); }

  /* a dot-matrix figure: the digits in the dot face, the unit beside it in the text face (the dot face has no usable %) */
  function dot(text, unit, cls) { return h("b", { class: "p-num" + (cls ? " " + cls : "") }, digits(text), unit ? h("sub", null, unit) : null); }
  function dotPct(x, toned) {
    if (!isNum(x)) return h("b", { class: "p-num nil" }, h("span", { "aria-hidden": "true" }, DASH), h("span", { class: "sr" }, "nothing yet"));
    var r = round(x * 100, 2);
    return dot(signed(r, 2), "%", toned ? tone(r) : "flat");
  }

  /* ── dates ───────────────────────────────────────────────────────────────────────────────────────────── */

  function ymd(v) {
    var m = /^(\d{4})-(\d{2})-(\d{2})/.exec(txt(v));
    if (!m || +m[2] < 1 || +m[2] > 12 || +m[3] < 1 || +m[3] > 31) return null;
    return { y: +m[1], m: +m[2], d: +m[3] };
  }
  function thisYear() { var a = S.state && ymd(S.state.asof); return a ? a.y : new Date().getFullYear(); }
  /* "13 Nov"; the year is added when it is not this one */
  function dayText(v, year) {
    var d = ymd(v);
    if (!d) return "";
    return d.d + NBSP + MON[d.m - 1] + (year || d.y !== thisYear() ? NBSP + d.y : "");
  }
  function day(v, year) { var t = dayText(v, year); return t ? h("span", { class: "dt" }, t) : nil(); }
  function span(a, b, year) { var x = dayText(a, year), y = dayText(b, year); return x && y ? x + " " + DASH + " " + y : x || y || ""; }
  function two(n) { return (n < 10 ? "0" : "") + n; }
  /* a moment in this device's own time: "13 Nov 22:41", or "Mon 16 Nov 22:25" */
  function when(iso, dow) {
    if (!txt(iso)) return "";
    var d = new Date(iso);
    if (isNaN(d.getTime())) return "";
    return (dow ? DOW[d.getDay()] + " " : "") + d.getDate() + NBSP + MON[d.getMonth()] + (d.getFullYear() !== thisYear() ? NBSP + d.getFullYear() : "") + " " + two(d.getHours()) + ":" + two(d.getMinutes());
  }
  function clock(ms) { var d = new Date(ms); return two(d.getHours()) + ":" + two(d.getMinutes()); }
  function plural(n, one, many) { return n + " " + (n === 1 ? one : many || one + "s"); }

  /* ── text with marks: a small markdown, built as nodes ───────────────────────────────────────────────── */

  function safeUrl(u) {
    try { var x = new URL(String(u)); return x.protocol === "http:" || x.protocol === "https:" ? x.href : null; } catch (e) { return null; }
  }
  function link(url, label) { return h("a", { href: url, target: "_blank", rel: "noopener noreferrer", class: "out" }, label); }

  var LINK = /^\[((?:[^\[\]\\]|\\.|\[[^\[\]]*\])*)\]\(\s*<?([^\s<>()]*(?:\([^\s<>()]*\)[^\s<>()]*)*)>?(?:\s+(?:"[^"]*"|'[^']*'))?\s*\)/;

  /* bold, italic, code, links (http and https only); everything else stays as it was typed */
  function inl(parent, text, depth, nolink) {
    var i = 0, buf = "", m, n = text.length;
    depth = depth || 0;
    function flush() { if (buf) { words(parent, buf); buf = ""; } }
    function closer(from, mark) {
      for (var k = text.indexOf(mark, from); k >= 0; k = text.indexOf(mark, k + 1)) {
        if (k === from || /\s/.test(text[k - 1]) || text[k - 1] === "\\") continue;
        if (mark.length === 1 && (text[k + 1] === mark || text[k - 1] === mark)) continue;
        if (mark[0] === "_" && k + mark.length < n && /\w/.test(text[k + mark.length])) continue;
        return k;
      }
      return -1;
    }
    while (i < n) {
      var ch = text[i], rest;
      if (ch === "\\" && i + 1 < n && /[\\`*_{}\[\]()#+\-.!|<>~]/.test(text[i + 1])) { buf += text[i + 1]; i += 2; continue; }
      if (ch === "`") {
        m = /^(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/.exec(text.slice(i));
        if (m) { flush(); parent.appendChild(h("code", null, m[2].replace(/^ ([\s\S]*) $/, "$1"))); i += m[0].length; continue; }
      }
      if (ch === "!" && text[i + 1] === "[") {            /* a picture: only its words, nothing is fetched */
        m = LINK.exec(text.slice(i + 1));
        if (m) { buf += m[1]; i += 1 + m[0].length; continue; }
      }
      if (ch === "[") {
        m = LINK.exec(text.slice(i));
        if (m) {
          flush();
          var url = nolink ? null : safeUrl(m[2]);
          if (url && depth < 8) { var a = link(url); inl(a, m[1], depth + 1, true); parent.appendChild(a); }
          else if (depth < 8) inl(parent, m[1], depth + 1, nolink);
          else buf += m[1];
          i += m[0].length; continue;
        }
      }
      if (ch === "<" && !nolink) {
        m = /^<(https?:\/\/[^\s<>]+)>/.exec(text.slice(i));
        if (m && safeUrl(m[1])) { flush(); parent.appendChild(link(safeUrl(m[1]), m[1])); i += m[0].length; continue; }
      }
      if ((ch === "*" || ch === "_") && depth < 8 && i + 1 < n && !/\s/.test(text[i + 1]) && !(ch === "_" && i > 0 && /\w/.test(text[i - 1]))) {
        var mark = text[i + 1] === ch ? ch + ch : ch, end = closer(i + mark.length, mark);
        if (end > i + mark.length - 1 && end > i + mark.length) {
          flush();
          rest = h(mark.length === 2 ? "strong" : "em");
          inl(rest, text.slice(i + mark.length, end), depth + 1, nolink);
          parent.appendChild(rest);
          i = end + mark.length; continue;
        }
      }
      buf += ch; i++;
    }
    flush();
    return parent;
  }

  var HEAD = [null, "h4", "h5", "h6", "h6", "h6", "h6"];   /* the page owns h1 to h3 */

  /* headings, paragraphs, lists, tables, quotes, rules, code blocks. Unknown marks are shown as typed. */
  function md(src, depth) {
    var root = document.createDocumentFragment();
    var lines = txt(src).replace(/\r\n?/g, "\n").split("\n"), i = 0, m;
    depth = depth || 0;
    function blank(l) { return /^\s*$/.test(l); }
    function head(l) { return /^ {0,3}(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$/.exec(l); }
    function rule(l) { return /^ {0,3}([-*_])( *\1){2,} *$/.test(l); }
    function fence(l) { return /^ {0,3}(```|~~~)/.exec(l); }
    function quote(l) { return /^ {0,3}>/.test(l); }
    function item(l) { return /^( *)([-*+]|\d{1,9}[.)])[ \t]+(.*)$/.exec(l); }
    function sep(l) { return l.indexOf("-") >= 0 && /^ *\|? *:?-+:? *(\| *:?-+:? *)*\|? *$/.test(l); }
    function tableAt(k) { return lines[k].indexOf("|") >= 0 && k + 1 < lines.length && lines[k + 1].indexOf("|") >= 0 && sep(lines[k + 1]); }
    function block(k) { return head(lines[k]) || rule(lines[k]) || fence(lines[k]) || quote(lines[k]) || tableAt(k); }
    function cells(l) {
      var t = l.trim(), out = [], cur = "";
      if (t[0] === "|") t = t.slice(1);
      for (var k = 0; k < t.length; k++) {
        if (t[k] === "\\" && t[k + 1] === "|") { cur += "|"; k++; }
        else if (t[k] === "|") { out.push(cur.trim()); cur = ""; }
        else cur += t[k];
      }
      if (cur.trim() !== "") out.push(cur.trim());
      return out;
    }
    function list(parent, base, deep) {
      var first = item(lines[i]), ordered = /\d/.test(first[2]), L = h(ordered ? "ol" : "ul"), it = null, mm, j;
      function flush() { if (it && it.t.length) { inl(it.el, it.t.join(" ")); it.t = []; } }
      if (ordered && parseInt(first[2], 10) !== 1) L.setAttribute("start", String(parseInt(first[2], 10)));
      parent.appendChild(L);
      while (i < lines.length) {
        var l = lines[i];
        if (blank(l)) {
          for (j = i + 1; j < lines.length && blank(lines[j]); j++) { /* skip */ }
          mm = j < lines.length ? item(lines[j]) : null;
          if (mm && mm[1].length >= base) { i = j; continue; }
          break;
        }
        mm = item(l);
        if (mm && !rule(l)) {
          if (mm[1].length < base) break;
          if (mm[1].length >= base + 2 && it && deep < 6) { flush(); list(it.el, mm[1].length, deep + 1); continue; }
          if (/\d/.test(mm[2]) !== ordered) break;
          flush();
          it = { el: h("li"), t: [mm[3]] };
          L.appendChild(it.el);
          i++; continue;
        }
        if (!it || block(i)) break;
        it.t.push(l.trim()); i++;                          /* a wrapped line of the same item */
      }
      flush();
    }
    while (i < lines.length) {
      var l = lines[i];
      if (blank(l)) { i++; continue; }
      if ((m = fence(l))) {
        var code = [];
        for (i++; i < lines.length && lines[i].trim().slice(0, 3) !== m[1]; i++) code.push(lines[i]);
        i++;
        root.appendChild(h("pre", null, h("code", null, code.join("\n"))));
        continue;
      }
      if ((m = head(l))) { root.appendChild(inl(h(HEAD[m[1].length]), m[2])); i++; continue; }
      if (rule(l)) { root.appendChild(h("hr")); i++; continue; }
      if (tableAt(i)) {
        var heads = cells(l), body = h("tbody"), tr;
        i += 2;
        while (i < lines.length && !blank(lines[i]) && lines[i].indexOf("|") >= 0) {
          var cs = cells(lines[i++]);
          tr = h("tr");
          for (var c = 0; c < heads.length; c++) tr.appendChild(inl(h("td"), cs[c] || ""));
          body.appendChild(tr);
        }
        tr = h("tr");
        heads.forEach(function (x) { tr.appendChild(inl(h("th", { scope: "col" }), x)); });
        root.appendChild(h("div", { class: "md-table" }, h("table", null, h("thead", null, tr), body)));
        continue;
      }
      if (quote(l)) {
        var q = [];
        while (i < lines.length && quote(lines[i])) q.push(lines[i++].replace(/^ {0,3}> ?/, ""));
        if (depth < 4) root.appendChild(h("blockquote", null, md(q.join("\n"), depth + 1)));
        else root.appendChild(h("p", null, q.join(" ")));
        continue;
      }
      if (item(l)) { list(root, item(l)[1].length, 0); continue; }
      var para = [l.trim()];
      for (i++; i < lines.length && !blank(lines[i]) && !block(i) && !item(lines[i]); i++) para.push(lines[i].trim());
      root.appendChild(inl(h("p"), para.join(" ")));
    }
    return root;
  }

  /* one line of plain text in which `this` is a code word (the engine's notices name files and commands) */
  function withCode(v) {
    var t = txt(v), out = [], re = /`([^`\n]+)`/g, last = 0, m;
    while ((m = re.exec(t))) {
      if (m.index > last) out.push(t.slice(last, m.index));
      out.push(h("code", null, m[1]));
      last = re.lastIndex;
    }
    if (last < t.length) out.push(t.slice(last));
    return out;
  }

  /* ── lamps, bars, folds ──────────────────────────────────────────────────────────────────────────────── */

  /* a lamp: "on" lit · "off" hollow · "alarm" blinking · "no" struck through */
  function lamp(state) {
    return h("i", { class: "p-blk" + (state === "on" ? "" : state === "alarm" ? " alarm" : state === "no" ? " no" : " off"), "aria-hidden": "true" });
  }
  var VERDICT = {
    likely: { lamp: "on", word: "skill likely" },
    none: { lamp: "off", word: "too early" },
    unlikely: { lamp: "no", word: "skill unlikely" }
  };
  function lampOf(v) { v = obj(v); return v.lamp === "likely" || v.lamp === "unlikely" ? v.lamp : "none"; }   /* anything else is "too early" */

  /* a bar either side of the index: cells to the right when ahead, to the left when behind, on one scale */
  function sides(v, scale) {
    var p = isNum(v) && scale > 0 ? Math.min(100, Math.abs(v) / scale * 100) : 0;
    return h("span", { class: "dv", "aria-hidden": "true" },
      h("i", { class: "neg", css: { "--v": (v < 0 ? Math.max(4, p) : 0).toFixed(1) + "%" } }),
      h("b"),
      h("i", { class: "pos", css: { "--v": (v > 0 ? Math.max(4, p) : 0).toFixed(1) + "%" } }));
  }
  function meter(frac, label) {
    var p = isNum(frac) ? Math.max(0, Math.min(1, frac)) * 100 : 0;
    return h("div", { class: "p-meter", role: "img", "aria-label": label || null, css: { "--v": p.toFixed(1) + "%" } });
  }
  function fold(key, title, body, note) {
    var d = h("details", { class: "fold", "data-fold": key },
      h("summary", { "data-k": "fold:" + key }, h("span", { class: "ft" }, title), note ? h("span", { class: "note" }, note) : null),
      h("div", { class: "fold-b" }, body));
    if (S.ui.fold[key]) d.open = true;
    d.addEventListener("toggle", function () { S.ui.fold[key] = d.open; });
    return d;
  }
  function sec(id, title, ru, note, body, cls) {
    return h("section", { class: "sec" + (cls ? " " + cls : ""), id: "sec-" + id },
      h("h3", { class: "grp" }, h("span", null, title, ru ? h("span", { class: "p-ru", "aria-hidden": "true" }, " / " + ru) : null), note ? h("span", { class: "note" }, note) : null),
      body);
  }
  function empty(text) { return h("p", { class: "empty" }, text); }
  function key(cls, label) { return h("span", { class: "lg" }, h("i", { class: "sw sw-" + cls, "aria-hidden": "true" }), label); }

  var LAMP_TEXT = "Profit alone needs years to tell skill from luck. So every evening each model's ranking or forecast " +
    "is written down, and later scored against what the stocks went on to do. The lamp reads that score: lit when " +
    "skill is likely, hollow while it is too early to tell, struck through when skill is unlikely. It can speak " +
    "long before the profit column means anything.";
  function lampFold(k) {
    return fold(k, "What the lamp means", [
      h("p", null, LAMP_TEXT),
      h("p", { class: "lamps" }, h("span", { class: "lg" }, lamp("on"), "skill likely"), h("span", { class: "lg" }, lamp("off"), "too early"), h("span", { class: "lg" }, lamp("no"), "skill unlikely"))
    ]);
  }

  /* ── the models, in order ────────────────────────────────────────────────────────────────────────────── */

  function okId(id) { return id.length > 0 && id.length <= 200; }
  function href(id) { return "#/" + encodeURIComponent(id); }
  function hasRecord(m) { return isNum(m.equity) || isNum(m.ret) || !!ymd(m.since); }
  function byVs(a, b) { return (isNum(b.vs_index) ? b.vs_index : -Infinity) - (isNum(a.vs_index) ? a.vs_index : -Infinity); }
  /* the index first (it is the bar), then the running models best to worst against it, then those not running */
  function order() {
    var all = arr(S.state && S.state.models).filter(function (m) { return m && typeof m === "object"; });
    var o = { index: [], run: [], out: [] };
    all.forEach(function (m) { (m.kind === "index" ? o.index : m.state === "running" ? o.run : o.out).push(m); });
    o.run.sort(byVs); o.out.sort(byVs);
    o.all = o.index.concat(o.run, o.out);
    o.ids = o.all.map(function (m) { return txt(m.id); });
    return o;
  }
  function model(id) { var a = order().all; for (var i = 0; i < a.length; i++) if (txt(a[i].id) === id) return a[i]; return null; }
  var KIND = { index: "the bar", rule: "rule", learned: "learned", agent: "agent" };
  function kindWord(m) { return pick(KIND, m.kind) || txt(m.kind); }
  /* why a model is not running, in the engine's own words */
  function stateWords(m) {
    var st = txt(m.state) || "not running", why = txt(m.state_text);
    if (!why) return [h("b", { class: "chip" }, st)];
    if (why.toLowerCase().indexOf(st.toLowerCase()) === 0) return [h("span", { class: "why" }, why)];
    return [h("b", { class: "chip" }, st), " ", h("span", { class: "why" }, why)];
  }

  /* ── reading the engine ──────────────────────────────────────────────────────────────────────────────── */

  /* when figures first appear their blocks strike on one after another, each with its own stutter. The theme's
     own helper does it (Plint.strike); without the theme, or with reduced motion, they are simply there. */
  function strike(list, gap) {
    var P = window.Plint;
    if (!P || typeof P.strike !== "function" || typeof P.live !== "function" || !P.live()) return;
    for (var i = 0; i < list.length; i++) { try { P.strike(list[i], Math.min(i * gap, 900)); } catch (e) { return; } }
  }
  /* repaint without moving the reader: the scroll position and the focused control stay where they were */
  function keep(fn) {
    var y = window.scrollY, a = document.activeElement, k = a && a.getAttribute ? a.getAttribute("data-k") : null;
    fn();
    if (Math.abs(window.scrollY - y) > 1) window.scrollTo(0, y);
    if (k && (!document.activeElement || document.activeElement.getAttribute("data-k") !== k)) {
      var n = document.querySelectorAll("[data-k]");
      for (var i = 0; i < n.length; i++) if (n[i].getAttribute("data-k") === k) { try { n[i].focus({ preventScroll: true }); } catch (e) { /* gone */ } break; }
    }
  }
  /* if the page itself trips over what it was sent, it says so: that is a fault here, not a silent engine */
  function draw(fn) {
    var was = S.broken;
    S.broken = null;
    try { fn(); } catch (e) {
      S.broken = String(e && e.message || e);
      if (window.console && console.error) console.error(e);
    }
    if (S.broken || was) { try { paintStatus(); } catch (e2) { /* nothing more to say it with */ } }
  }
  function plan() { clearTimeout(S.timer); S.timer = setTimeout(getState, S.state ? EVERY : 10000); }
  function getState() {
    if (S.busy) return;
    S.busy = true; S.tried = Date.now();
    fetch("api/state", { cache: "no-store" })
      .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.text(); })
      .then(function (raw) {
        var d = JSON.parse(raw);
        if (!d || typeof d !== "object" || !Array.isArray(d.models)) throw new Error("not a state");
        return { raw: raw, d: d };
      })
      .then(function (x) {                                  /* read: paint it if anything changed */
        var failed = !!S.fail;
        S.fail = null;
        if (x.raw !== S.raw) { S.raw = x.raw; S.state = x.d; draw(paint); }
        else if (failed) keep(paintStatus);
      }, function () {                                      /* not read: keep what there is, and say it is stale */
        if (!S.fail) S.fail = { since: Date.now() };
        keep(paintStatus);
      })
      .then(next, next);
    function next() {
      S.busy = false;
      plan();
      if (S.open) getBook(S.open, true);
    }
  }
  function getBook(id, quiet) {
    var b = S.book[id] || (S.book[id] = { data: null, raw: "", err: false, busy: false });
    if (b.busy) return;
    b.busy = true;
    if (!quiet && S.open === id) paintBook(true);
    fetch("api/model/" + encodeURIComponent(id), { cache: "no-store" })
      .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.text(); })
      .then(function (raw) {
        var d = JSON.parse(raw);
        if (!d || typeof d !== "object" || Array.isArray(d)) throw new Error("not a model");
        var changed = raw !== b.raw || b.err;
        b.raw = raw; b.data = d; b.err = false; b.busy = false;
        return changed;
      })
      .catch(function () { var changed = !b.err; b.err = true; b.busy = false; return changed || !quiet; })
      .then(function (changed) { if (S.open === id && (changed || !quiet)) draw(function () { keep(function () { paintBook(); }); }); });
  }

  /* ── the first thing on the page: the engine's state, and what is missing ────────────────────────────── */

  var PHASE = { keys: "Waiting for keys", backfill: "Fetching the history", research: "Testing the models on history", running: "Running" };
  var KEYS = { alpaca: ["Alpaca", "market data"], anthropic: ["Anthropic", "the agent"], sec: ["SEC", "filings"], ntfy: ["ntfy", "phone notices"] };
  var LEVEL = { fault: "alarm", warn: "on", info: "off" };

  function stItem(k, v) { return v ? [" ", h("span", { class: "st-i" }, h("span", { class: "k" }, k), " ", h("span", { class: "v" }, v))] : null; }
  function paintStatus() {
    var st = S.state, box = $("status"), out = [];
    if (!st) {
      out.push(S.fail
        ? h("p", { class: "st-fault", role: "alert" }, lamp("alarm"), h("span", null, h("b", null, "Can't reach the engine."), " Nothing has been read yet. Trying again every ten seconds."))
        : h("p", { class: "st-line" }, h("span", { class: "st-phase" }, h("span", { class: "p-cursor", "aria-hidden": "true" }), h("b", null, "Reading the books"))));
      box.setAttribute("data-phase", S.fail ? "fault" : "reading");
      fill(box, out);
      return;
    }
    var ck = obj(st.clock), phase = txt(st.phase), running = phase === "running";
    var keys = obj(st.keys), names = Object.keys(keys), missing = names.filter(function (k) { return keys[k] === false; });
    out.push(h("p", { class: "st-line" },
      h("span", { class: "st-phase" + (running ? " is-running" : " is-held") }, lamp(running ? "on" : "off"), h("b", null, txt(st.phase_text) || pick(PHASE, phase) || phase || "Unknown")),
      stItem("last close", dayText(st.asof)),
      stItem("figures read", when(st.generated)),
      stItem("next run", when(ck.next_run, true))));
    if (txt(ck.busy)) out.push(h("p", { class: "st-busy" }, h("span", { class: "k" }, "now"), " ", h("span", { class: "v" }, txt(ck.busy)), h("span", { class: "p-cursor", "aria-hidden": "true" })));
    if (S.fail) {
      out.push(h("p", { class: "st-fault", role: "alert" }, lamp("alarm"),
        h("span", null, h("b", null, "Stale."), " Can't reach the engine since " + clock(S.fail.since) + ". These are the figures read " + (when(st.generated) || "earlier") + "; trying again every minute.")));
    }
    if (S.broken) {
      out.push(h("p", { class: "st-fault", role: "alert" }, lamp("alarm"),
        h("span", null, h("b", null, "The page has a fault."), " It could not draw the figures it was sent (" + S.broken + "). The engine is not to blame; what shows may be incomplete.")));
    }
    if (missing.length) {
      out.push(h("ul", { class: "st-keys", "aria-label": "Keys" }, names.map(function (k) {
        var w = pick(KEYS, k) || [k, ""], set = keys[k] !== false;
        return h("li", { class: set ? "is-set" : "is-missing", "data-key": k }, lamp(set ? "on" : "off"), h("b", null, w[0]), " ", w[1] ? [h("span", { class: "k" }, w[1]), " "] : null, h("span", { class: "v" }, set ? "set" : "missing"));
      })));
    }
    var notes = arr(st.notices).filter(function (n) { return n && txt(n.text); });
    if (notes.length) {
      out.push(h("ul", { class: "st-notes", "aria-label": "Notices" }, notes.map(function (n) {
        var lv = pick(LEVEL, n.level) ? n.level : "info";
        return h("li", { class: "lv-" + lv }, lamp(LEVEL[lv]), h("span", null, withCode(n.text)));
      })));
    }
    box.setAttribute("data-phase", running ? "running" : "held");
    box.classList.toggle("is-stale", !!S.fail);
    fill(box, out);
  }

  /* ── four figures ────────────────────────────────────────────────────────────────────────────────────── */

  function tile(cls, label, figure, extra, note) {
    return h("div", { class: "tile tile-" + cls }, h("span", { class: "tl" }, label), figure, extra, note ? h("span", { class: "tn" }, note) : null);
  }
  function stopWords(v) {
    return v === "month cap" ? "at the month's cap" : v === "day allowance" ? "for today: the day's allowance is used" : txt(v);
  }
  function paintTiles() {
    var st = S.state, box = $("tiles"), o = order(), out = [];
    var rec = o.all.filter(hasRecord);
    box.hidden = !rec.length;
    if (!rec.length) { fill(box, []); return; }
    var ix = o.index[0], rivals = o.run.concat(o.out).filter(function (m) { return isNum(m.vs_index); });
    if (ix && hasRecord(ix)) {
      out.push(tile("index", "The index", dotPct(ix.ret, true), null,
        [txt(obj(st.index).title) || txt(ix.title), dayText(ix.since) ? " " + DASH + " since " + dayText(ix.since) : ""]));
    }
    if (rivals.length) {
      var ahead = rivals.filter(function (m) { return m.vs_index > 0; }).length;
      out.push(tile("ahead", "Ahead of the index", dot(String(ahead), "of " + rivals.length),
        h("span", { class: "marks", "aria-hidden": "true" }, rivals.map(function (m) { return h("i", { class: "mk" + (m.vs_index > 0 ? " lit" : "") }); })),
        plural(rivals.length - ahead, "model") + " behind"));
    }
    var scored = o.run.concat(o.out).filter(function (m) { return m.verdict && typeof m.verdict === "object"; });
    if (scored.length) {
      var n = { likely: 0, none: 0, unlikely: 0 };
      scored.forEach(function (m) { n[lampOf(m.verdict)]++; });
      out.push(tile("lamps", "Skill likely", dot(String(n.likely), "of " + scored.length),
        h("span", { class: "marks", "aria-hidden": "true" }, scored.map(function (m) { return lamp(VERDICT[lampOf(m.verdict)].lamp); })),
        n.none + " too early, " + n.unlikely + " unlikely"));
    }
    var sp = st.spend && typeof st.spend === "object" ? st.spend : null;
    if (sp && isNum(sp.used) && isNum(sp.cap)) {
      var mo = /^(\d{4})-(\d{2})$/.exec(txt(sp.month));
      out.push(tile("spend" + (txt(sp.stopped) ? " is-stopped" : ""), "Agent spend" + (mo && MON[+mo[2] - 1] ? ", " + MON[+mo[2] - 1] : ""),
        dot("$" + nf(sp.used, 0), "of $" + nf(sp.cap, 0)),
        meter(sp.cap > 0 ? sp.used / sp.cap : 0),
        txt(sp.stopped) ? [h("b", { class: "p-inv" }, "Stopped"), " " + stopWords(sp.stopped)] : isNum(sp.today) ? "today " + usdText(sp.today) : null));
    }
    fill(box, out);
  }

  /* ── every model at once ─────────────────────────────────────────────────────────────────────────────── */

  function nums(xs) { return arr(xs).filter(isNum); }
  /* the model's line and the index's, both as value / start. Every row is drawn on the same scale. */
  function spark(m, lo, hi) {
    var a = nums(m.spark), b = m.kind === "index" ? a : nums(m.spark_index), W = 120, H = 28, P = 2.5;
    if (a.length < 2 && b.length < 2) return nil();
    function y(v) { return H - P - (v - lo) / ((hi - lo) || 1) * (H - 2 * P); }
    function pts(xs) { return xs.map(function (v, i) { return (P + i / (xs.length - 1) * (W - 2 * P)).toFixed(1) + "," + y(v).toFixed(1); }).join(" "); }
    return s("svg", { class: "spark", viewBox: "0 0 " + W + " " + H, preserveAspectRatio: "none", "aria-hidden": "true", focusable: "false" },
      lo <= 1 && hi >= 1 ? s("line", { class: "ax", x1: 0, x2: W, y1: y(1).toFixed(1), y2: y(1).toFixed(1) }) : null,
      b.length > 1 ? s("polyline", { class: "ln-index", points: pts(b) }) : null,
      a.length > 1 && m.kind !== "index" ? s("polyline", { class: "ln-model", points: pts(a) }) : null);
  }
  function cell(cls, label, value) { return h("span", { class: "c c-" + cls }, h("span", { class: "k" }, label), h("span", { class: "v" }, value)); }

  function rowFor(m, o, rank) {
    var id = txt(m.id), isIx = m.kind === "index", run = m.state === "running", rec = hasRecord(m), v = m.verdict && typeof m.verdict === "object" ? m.verdict : null;
    var name = h("span", { class: "c c-name" },
      h("span", { class: "rk", "aria-hidden": "true" }, rank ? String(rank) : ""),
      h("b", { class: "t" }, txt(m.title) || id || "Unnamed"),
      h("span", { class: "tag" }, kindWord(m)),
      run ? null : h("span", { class: "st" }, lamp("off"), stateWords(m)));
    var kids;
    if (!rec) {
      kids = [name, h("span", { class: "c c-what" }, txt(m.what))];
    } else {
      var lw = v ? VERDICT[lampOf(v)] : null;
      kids = [name,
        h("span", { class: "c c-spark" }, spark(m, o.lo, o.hi)),
        cell("ret", "return", pct(m.ret, { tone: true })),
        cell("idx", "index", pct(m.index_ret)),
        h("span", { class: "c c-vs" }, h("span", { class: "k" }, "vs index"),
          h("span", { class: "v" }, isIx ? h("span", { class: "bar-word" }, "the bar") : pct(m.vs_index, { tone: true })), sides(isIx ? 0 : m.vs_index, o.scale)),
        cell("day", "today", pct(m.day_ret, { tone: true })),
        cell("dd", "worst drop", pct(m.max_dd)),
        cell("tr", "trades", count(m.trades)),
        cell("cost", "costs", usd(m.costs)),
        h("span", { class: "c c-lamp" + (lw ? " is-" + lampOf(v) + (txt(v.text) ? "" : " no-text") : " is-null") }, h("span", { class: "k" }, "verdict"),
          h("span", { class: "v" }, lw ? [lamp(lw.lamp), h("span", { class: "lw" }, lw.word), isNum(v.n) ? h("span", { class: "ln" }, plain(v.n, 0) + " scored") : null, h("span", { class: "lt" }, txt(v.text))]
            : [nil(), isIx ? null : h("span", { class: "lt" }, "not scored")])),
        cell("since", "since", day(m.since))];
    }
    var cls = "row" + (isIx ? " is-index" : "") + (run ? "" : " is-out") + (rec ? "" : " is-bare");
    var a = okId(id)
      ? h("a", { class: "row-a", href: href(id), "data-k": "row:" + id, "aria-current": S.open === id ? "true" : null }, kids)
      : h("div", { class: "row-a" }, kids);
    return h("li", { class: cls, "data-id": id, "data-state": run ? "running" : txt(m.state) === "off" ? "off" : "waiting" }, a);
  }

  var COLS = [["name", "Model"], ["spark", "Curve"], ["ret", "Return"], ["idx", "Index"], ["vs", "vs index"], ["day", "Today"], ["dd", "Worst drop"],
    ["tr", "Trades"], ["cost", "Costs"], ["lamp", "Verdict"], ["since", "Since"]];

  function paintBoard() {
    var box = $("board"), o = order(), all = o.all, lo = Infinity, hi = -Infinity, scale = 0;
    all.forEach(function (m) {
      nums(m.spark).concat(nums(m.spark_index)).forEach(function (v) { if (v < lo) lo = v; if (v > hi) hi = v; });
      if (m.kind !== "index" && isNum(m.vs_index)) scale = Math.max(scale, Math.abs(m.vs_index));
    });
    if (!(hi > lo)) { lo = (isFinite(lo) ? lo : 1) - 0.01; hi = lo + 0.02; }
    o.lo = lo; o.hi = hi; o.scale = scale;
    var rec = all.filter(hasRecord).length, rivals = o.run.length + o.out.length, rows = [], rank = 0;
    o.index.forEach(function (m) { rows.push(rowFor(m, o, 0)); });
    o.run.forEach(function (m) { rows.push(rowFor(m, o, ++rank)); });
    if (o.out.length && (o.run.length || o.index.some(hasRecord))) rows.push(h("li", { class: "row-gap" }, h("span", null, "Not running")));
    o.out.forEach(function (m) { rows.push(rowFor(m, o, 0)); });
    var note = !all.length ? "" : !rec ? plural(all.length, "model") + ", none has a record yet"
      : plural(rivals, "model") + (o.index.length ? " and the index" : "") + ", best first against it";
    fill(box, [
      h("h2", { class: "grp" }, h("span", null, "Models", h("span", { class: "p-ru", "aria-hidden": "true" }, " / МОДЕЛИ")), note ? h("span", { class: "note" }, note) : null),
      rec ? h("div", { class: "legend" }, key("model", "model"), key("index", "index"),
        h("span", { class: "lg lg-lamp" }, lamp("on"), "skill likely"), h("span", { class: "lg lg-lamp" }, lamp("off"), "too early"), h("span", { class: "lg lg-lamp" }, lamp("no"), "skill unlikely"),
        lampFold("lamp")) : null,
      rec ? h("div", { class: "cols", "aria-hidden": "true" }, COLS.map(function (c) { return h("span", { class: "c c-" + c[0] }, c[1], c[0] === "vs" ? h("span", { class: "srt" }, " ↓") : null); })) : null,
      all.length ? h("ol", { class: "rows" + (rec ? "" : " is-roster") }, rows) : empty("The engine lists no models yet.")
    ]);
  }

  function paintLog() {
    var box = $("log"), log = arr(S.state.log).filter(function (l) { return l && txt(l.text); });
    box.hidden = !log.length;
    if (!log.length) { fill(box, []); return; }
    fill(box, fold("log", "Log", h("ol", { class: "log" }, log.map(function (l) {
      return h("li", { class: l.level === "warn" ? "lv-warn" : "lv-info" }, h("time", null, when(l.t)), lamp(l.level === "warn" ? "on" : "off"), h("span", null, withCode(l.text)));
    })), plural(log.length, "line") + (when(log[0].t) ? ", the last at " + when(log[0].t) : "")));
  }

  function paintFoot() {
    var st = S.state, ck = obj(st.clock), cash = isNum(st.start_cash) ? usdText(st.start_cash, 0) : "";
    fill($("foot"), [
      h("p", { class: "paper" }, h("b", null, "Paper only."), " Every book here is pretend money" + (cash ? ", " + cash + " a model" : "") + ". Nothing on this page can place a real order."),
      h("p", { class: "read", id: "read" }, [when(ck.last_run) ? "Last run " + when(ck.last_run) : "", when(st.generated) ? "figures read " + when(st.generated) : "", "the page looks again every minute"]
        .filter(Boolean).join(" · "))
    ]);
  }

  function paint() {
    var first = !S.painted.board;
    keep(function () {
      paintStatus(); paintTiles(); paintBoard(); paintLog(); paintFoot();
      if (S.open) paintBook();
    });
    name();
    S.painted.board = true;
    if (first) strike(document.querySelectorAll("#status > *, #tiles > .tile, #board > .grp, #board > .legend, #board > .cols, #board .rows > li, #log, #foot > *"), 45);
  }

  /* ── charts: drawn here, plain polylines and cells ───────────────────────────────────────────────────── */

  var uid = 0;
  function steps(lo, hi, want) {
    var raw = ((hi - lo) || Math.abs(hi) || 1) / want, mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10)), n = raw / mag;
    var step = (n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10) * mag, out = [];
    for (var v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(round(v, 8));
    return out;
  }
  /* which day the pointer is over; arrows walk the days, so a keyboard reads the chart as a finger does */
  function reader(host, n, show) {
    var cur = n - 1, geo = null;
    function at(e) {
      if (!geo) return cur;
      var r = host.getBoundingClientRect(), x = e.clientX - r.left;
      return Math.max(0, Math.min(n - 1, Math.round((x - geo.l) / (geo.w || 1) * (n - 1))));
    }
    function go(i) { cur = Math.max(0, Math.min(n - 1, i)); show(cur); }
    host.addEventListener("pointermove", function (e) { go(at(e)); });
    host.addEventListener("pointerdown", function (e) { go(at(e)); });
    host.addEventListener("pointerleave", function (e) { if (e.pointerType === "mouse") go(n - 1); });
    host.addEventListener("keydown", function (e) {
      var k = e.key, to = k === "ArrowLeft" ? cur - 1 : k === "ArrowRight" ? cur + 1 : k === "Home" ? 0 : k === "End" ? n - 1 : null;
      if (to == null) return;
      e.preventDefault(); go(to);
    });
    return { set: function (g) { geo = g; show(cur); }, cur: function () { return cur; } };
  }

  /* the model against the index from the same start: two lines on one axis, the gap between them shaded */
  function curveChart(rows, o) {
    var host = h("div", { class: "chart chart-curve", tabindex: "0", role: "group", "data-k": "chart:curve", "aria-label": o.label + ". Arrow keys read it day by day." });
    var rd = h("p", { class: "rd", "aria-live": "polite" }), n = rows.length, cross, pa, pb, X, Y;
    var read = reader(host, n, function (i) {
      var r = rows[i], gap = isNum(r[1]) && isNum(r[2]) && o.start ? (r[1] - r[2]) / o.start : null;
      if (cross) {
        cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i));
        [[pa, r[1]], [pb, o.single ? null : r[2]]].forEach(function (p) {
          if (!p[0]) return;
          if (isNum(p[1])) { p[0].setAttribute("x", X(i) - 3); p[0].setAttribute("y", Y(p[1]) - 3); p[0].removeAttribute("display"); }
          else p[0].setAttribute("display", "none");
        });
      }
      fill(rd, [h("b", null, dayText(r[0]) || DASH), " ", h("span", { class: "rd-i" }, o.nameA + " ", usd(r[1]), o.start && isNum(r[1]) ? [" (", pct(r[1] / o.start - 1, { tone: true }), ")"] : null),
        o.single ? null : [" ", h("span", { class: "rd-i" }, o.nameB + " ", usd(r[2]), o.start && isNum(r[2]) ? [" (", pct(r[2] / o.start - 1), ")"] : null)],
        o.single || gap == null ? null : [" ", h("span", { class: "rd-i" }, round(gap * 100, 2) < 0 ? "behind by " : "ahead by ", pct(gap, { tone: true }))]]);
    });
    function draw() {
      var W = Math.max(260, Math.round(host.clientWidth || 0)), big = W >= 620, H = big ? 290 : 210, L = big ? 62 : 54, R = 10, T = 12, B = 26;
      var lo = Infinity, hi = -Infinity, pw = W - L - R, ph = H - T - B, id = "g" + (++uid);
      rows.forEach(function (r) { [r[1], o.single ? null : r[2]].forEach(function (v) { if (isNum(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } }); });
      if (o.start) { lo = Math.min(lo, o.start); hi = Math.max(hi, o.start); }
      if (!(hi > lo)) { lo = (isFinite(lo) ? lo : 0) - 1; hi = lo + 2; }
      var pad = (hi - lo) * 0.07; lo -= pad; hi += pad;
      X = function (i) { return round(L + (n > 1 ? i / (n - 1) : 0.5) * pw, 1); };
      Y = function (v) { return round(T + (hi - v) / (hi - lo) * ph, 1); };
      function line(k) {
        var d = "", pen = false;
        rows.forEach(function (r, i) { if (isNum(r[k])) { d += (pen ? "L" : "M") + X(i) + " " + Y(r[k]); pen = true; } else pen = false; });
        return d;
      }
      /* the gap to the index, cut where the lines cross: dots while ahead, hatching while behind */
      var up = "", down = "";
      if (!o.single) {
        for (var i = 0; i + 1 < n; i++) {
          var a0 = rows[i][1], b0 = rows[i][2], a1 = rows[i + 1][1], b1 = rows[i + 1][2];
          if (!isNum(a0) || !isNum(b0) || !isNum(a1) || !isNum(b1)) continue;
          var d0 = a0 - b0, d1 = a1 - b1, x0 = X(i), x1 = X(i + 1), q;
          if (d0 * d1 >= 0) {
            q = "M" + x0 + " " + Y(a0) + "L" + x1 + " " + Y(a1) + "L" + x1 + " " + Y(b1) + "L" + x0 + " " + Y(b0) + "Z";
            if (d0 + d1 > 0) up += q; else if (d0 + d1 < 0) down += q;
          } else {
            var t = d0 / (d0 - d1), xc = round(x0 + t * (x1 - x0), 1), yc = Y(a0 + t * (a1 - a0));
            q = "M" + x0 + " " + Y(a0) + "L" + xc + " " + yc + "L" + x0 + " " + Y(b0) + "Z";
            if (d0 > 0) up += q; else down += q;
            q = "M" + xc + " " + yc + "L" + x1 + " " + Y(a1) + "L" + x1 + " " + Y(b1) + "Z";
            if (d1 > 0) up += q; else down += q;
          }
        }
      }
      var ys = steps(lo, hi, big ? 5 : 4), dp = ys.length > 1 && Math.abs(ys[1] - ys[0]) < 1 ? 2 : 0;
      var want = Math.max(2, Math.min(n, Math.floor(pw / 92))), xs = [];
      for (var j = 0; j < want; j++) { var ix = n > 1 ? Math.round(j * (n - 1) / (want - 1)) : 0; if (xs.indexOf(ix) < 0) xs.push(ix); }
      cross = s("line", { class: "cross", y1: T, y2: T + ph });
      pa = s("rect", { class: "pt pt-model", width: 6, height: 6 });
      pb = o.single ? null : s("rect", { class: "pt pt-index", width: 6, height: 6 });
      fill(host, s("svg", { class: "plot", width: W, height: H, viewBox: "0 0 " + W + " " + H, "aria-hidden": "true", focusable: "false" },
        s("defs", null,
          s("pattern", { id: id + "a", width: 4, height: 4, patternUnits: "userSpaceOnUse" }, s("rect", { class: "pat", width: 1.4, height: 1.4 })),
          s("pattern", { id: id + "b", width: 6, height: 6, patternUnits: "userSpaceOnUse" }, s("path", { class: "pat-l", d: "M-1 7L7 -1M-1 1L1 -1M5 7L7 5" }))),
        ys.map(function (v) { return [s("line", { class: "grid", x1: L, x2: W - R, y1: Y(v), y2: Y(v) }), s("text", { class: "tick", x: L - 8, y: Y(v) + 3, "text-anchor": "end" }, usdText(v, dp))]; }),
        o.start ? s("line", { class: "base", x1: L, x2: W - R, y1: Y(o.start), y2: Y(o.start) }) : null,
        xs.map(function (ix, k) { return s("text", { class: "tick", x: X(ix), y: H - 7, "text-anchor": k === 0 ? "start" : k === xs.length - 1 ? "end" : "middle" }, dayText(rows[ix][0])); }),
        up ? s("path", { class: "band band-up", d: up, fill: "url(#" + id + "a)" }) : null,
        down ? s("path", { class: "band band-down", d: down, fill: "url(#" + id + "b)" }) : null,
        o.single ? null : s("path", { class: "ln-index", d: line(2) }),
        s("path", { class: "ln-model", d: line(1) }),
        cross, pb, pa));
      read.set({ l: L, w: pw });
    }
    S.charts.push(draw);
    return h("div", { class: "chartbox" }, host, rd);
  }

  /* one column of cells for each item, up from the line or down from it; exact figures are in the readout */
  function barsChart(items, o) {
    var host = h("div", { class: "chart chart-bars", tabindex: "0", role: "group", "data-k": "chart:" + o.key, "aria-label": o.label + ". Arrow keys read it one by one." });
    var rd = h("p", { class: "rd", "aria-live": "polite" }), n = items.length, hl, geo;
    var read = reader(host, n, function (i) {
      if (hl && geo) { hl.setAttribute("x", round(geo.l + i * geo.slot, 1)); hl.setAttribute("width", Math.max(1, round(geo.slot, 1))); }
      fill(rd, o.read(items[i]));
    });
    function draw() {
      var W = Math.max(260, Math.round(host.clientWidth || 0)), H = o.h || 150, L = 50, R = 8, T = 8, B = 22, pw = W - L - R, ph = H - T - B;
      var max = 0;
      items.forEach(function (it) { if (isNum(it.v)) max = Math.max(max, Math.abs(it.v)); });
      if (isNum(o.avg)) max = Math.max(max, Math.abs(o.avg));
      if (!max) max = 1;
      var slot = pw / n, p = slot >= 4 ? 4 : slot >= 3 ? 3 : slot >= 2 ? 2 : 1, cols = slot >= 7 ? Math.max(1, Math.min(4, Math.floor((slot - 2) / 4))) : 1, cellPx = p > 1 ? p - 1 : 1;
      var zero = o.signed ? T + Math.round(ph / 2 / p) * p : T + Math.floor(ph / p) * p, room = o.signed ? Math.floor(ph / 2 / p) : Math.floor(ph / p);
      var up = "", down = "";
      items.forEach(function (it, i) {
        if (!isNum(it.v) || !it.v) return;
        var k = Math.max(1, Math.round(Math.abs(it.v) / max * room)), x0 = Math.round(L + i * slot + (slot - (cols * p - (p > 1 ? 1 : 0))) / 2), d = "";
        for (var c = 0; c < cols; c++) for (var r = 0; r < k; r++) {
          var y = it.v > 0 ? zero - (r + 1) * p : zero + r * p + (p > 1 ? 1 : 0);
          d += "M" + (x0 + c * p) + " " + y + "h" + cellPx + "v" + cellPx + "h-" + cellPx + "z";
        }
        if (it.v > 0) up += d; else down += d;
      });
      var ya = isNum(o.avg) ? round(zero - o.avg / max * room * p, 1) : null, f = o.axis || function (v) { return signed(v, 2); };
      var ticks = [[zero - room * p, f(max)], [zero, o.signed ? "0" : f(0)]];
      if (o.signed) ticks.push([zero + room * p, f(-max)]);
      hl = s("rect", { class: "hl", y: T, height: (o.signed ? 2 : 1) * room * p });
      fill(host, s("svg", { class: "plot", width: W, height: H, viewBox: "0 0 " + W + " " + H, "aria-hidden": "true", focusable: "false" },
        hl,
        ticks.map(function (t) { return [s("line", { class: t[0] === zero ? "base" : "grid", x1: L, x2: W - R, y1: t[0], y2: t[0] }), s("text", { class: "tick", x: L - 8, y: t[0] + 3, "text-anchor": "end" }, t[1])]; }),
        ya == null ? null : s("line", { class: "avg", x1: L, x2: W - R, y1: ya, y2: ya }),
        s("text", { class: "tick", x: L, y: H - 6, "text-anchor": "start" }, items[0].x),
        n > 1 ? s("text", { class: "tick", x: W - R, y: H - 6, "text-anchor": "end" }, items[n - 1].x) : null,
        up ? s("path", { class: "cells cells-up", d: up }) : null,
        down ? s("path", { class: "cells cells-down", d: down }) : null));
      geo = { l: L, w: pw - slot, slot: slot };
      read.set({ l: L + slot / 2, w: pw - slot });
    }
    S.charts.push(draw);
    return h("div", { class: "chartbox" }, host, rd);
  }

  /* where the truth could lie: the range as a band, the best guess as a block, zero as a tick */
  function rangeBar(lo, mid, hi, scale) {
    function at(v) { return 50 + Math.max(-1, Math.min(1, v / (scale || 1))) * 50; }
    return h("span", { class: "rg", "aria-hidden": "true" },
      h("i", { class: "rg-band", css: { left: at(lo).toFixed(1) + "%", width: Math.max(1, at(hi) - at(lo)).toFixed(1) + "%" } }),
      h("i", { class: "rg-zero" }),
      isNum(mid) ? h("i", { class: "rg-mid", css: { left: at(mid).toFixed(1) + "%" } }) : null);
  }

  /* ── tables ──────────────────────────────────────────────────────────────────────────────────────────── */

  /* cols: [key, label, right-aligned]; rows: arrays of cells (a cell may be { c: class, v: content }).
     Past `SHOW` rows the rest waits behind a button. */
  function table(name, cols, rows, o) {
    o = o || {};
    var k = o.more, all = !k || rows.length <= SHOW + 3 || !!S.ui.more[k];
    var trs = rows.map(function (r, i) {
      return h("tr", { hidden: !all && i >= SHOW }, r.map(function (v, j) {
        var own = v && !v.nodeType && !Array.isArray(v) && typeof v === "object" ? v : null;
        return h("td", { class: "t-" + cols[j][0] + (cols[j][2] ? " r" : "") + (own ? " " + own.c : ""), "data-l": cols[j][1] }, own ? own.v : v);
      }));
    });
    var t = h("table", { class: "tb tb-" + name },
      h("thead", null, h("tr", null, cols.map(function (c) { return h("th", { scope: "col", class: "t-" + c[0] + (c[2] ? " r" : "") }, c[1]); }))),
      h("tbody", null, trs));
    if (!k || rows.length <= SHOW + 3) return t;
    var btn = h("button", { type: "button", class: "more", "data-k": "more:" + k, "aria-expanded": all ? "true" : "false" });
    function label() { btn.textContent = S.ui.more[k] ? "Show the first " + SHOW + " only" : "Show all " + rows.length + " " + o.noun; }
    btn.addEventListener("click", function () {
      S.ui.more[k] = !S.ui.more[k];
      trs.forEach(function (tr, i) { tr.hidden = !S.ui.more[k] && i >= SHOW; });
      btn.setAttribute("aria-expanded", S.ui.more[k] ? "true" : "false");
      label();
      if (!S.ui.more[k] && t.getBoundingClientRect().top < 0) t.scrollIntoView({ block: "start" });
    });
    label();
    return [t, btn];
  }
  function side(v) { var sell = v === "sell"; return h("span", { class: "side " + (sell ? "side-sell" : "side-buy") }, sell ? "sell" : v === "buy" ? "buy" : txt(v) || DASH); }
  /* a stock: its symbol, its name, and (for an order or a trade) which way it went */
  function stock(sym, name, way) { return [way === undefined ? null : [side(way), " "], h("b", { class: "sym" }, txt(sym) || DASH), txt(name) ? h("span", { class: "nm" }, " " + txt(name)) : null]; }
  function why(v) { return txt(v) ? h("span", { class: "why" }, txt(v)) : nil(); }

  /* ── the open model ──────────────────────────────────────────────────────────────────────────────────── */

  function figure(label, value, cls, sub) {
    return h("div", { class: "fg" + (cls ? " " + cls : "") }, value, h("span", { class: "fl" }, label), h("span", { class: "fs" }, sub || null));
  }
  function fact(label, value) { return h("div", null, h("dt", null, label), h("dd", null, value)); }

  function bookHead(m, o) {
    var id = txt(m.id), at = o.ids.indexOf(id), prev = at > 0 ? o.all[at - 1] : null, next = at >= 0 && at < o.all.length - 1 ? o.all[at + 1] : null;
    var isIx = m.kind === "index", run = m.state === "running", rec = hasRecord(m), v = m.verdict && typeof m.verdict === "object" ? m.verdict : null;
    function step(to, word, sign) {
      return to && okId(txt(to.id))
        ? h("a", { class: "pg", href: href(txt(to.id)), "data-k": "pg:" + word, "aria-label": word + " model: " + (txt(to.title) || txt(to.id)) }, sign)
        : h("span", { class: "pg is-end", "aria-hidden": "true" }, sign);
    }
    return h("header", { class: "bk-head" },
      h("nav", { class: "bk-nav", "aria-label": "Models" },
        h("a", { class: "back", href: "#/", "data-k": "back" }, h("span", { "aria-hidden": "true" }, "‹ "), "All models"),
        at >= 0 ? h("span", { class: "pager" }, step(prev, "Previous", "‹"), h("span", { class: "pg-n" }, (at + 1) + " of " + o.all.length), step(next, "Next", "›")) : null),
      h("div", { class: "bk-title" },
        h("h2", { id: "book-title", tabindex: "-1", "data-k": "title" }, txt(m.title) || id),
        h("span", { class: "tag" }, kindWord(m)),
        v ? h("span", { class: "tag-lamp is-" + lampOf(v) }, lamp(VERDICT[lampOf(v)].lamp), VERDICT[lampOf(v)].word) : null),
      txt(m.what) ? h("p", { class: "what" }, txt(m.what)) : null,
      run ? null : h("p", { class: "bk-state", "data-state": txt(m.state) === "off" ? "off" : "waiting" }, lamp("off"), h("span", null, stateWords(m))),
      !rec ? null : h("div", { class: "figs" + (isIx ? " is-index" : "") },
        isIx ? figure("return", dotPct(m.ret, true), "hero") : figure("vs index", dotPct(m.vs_index, true), "hero"),
        isIx ? null : figure("return", dotPct(m.ret, true)),
        isIx ? null : figure("index", dotPct(m.index_ret)),
        figure("today", dotPct(m.day_ret, true), null, !isIx && isNum(m.day_vs_index) ? ["vs index ", pct(m.day_vs_index, { tone: true })] : null),
        figure("worst drop", dotPct(m.max_dd))),
      !rec ? null : h("dl", { class: "facts" },
        fact("Value", usd(m.equity)),
        fact("Cash", pct(m.cash, { dp: 1, sign: false })),
        fact("Holds", isNum(m.n_pos) ? plural(m.n_pos, "stock") : nil()),
        fact("Trades", count(m.trades)),
        fact("Costs paid", usd(m.costs)),
        fact("Turns over", isNum(m.turnover) ? plain(m.turnover, 1) + TIMES + " a year" : nil()),
        fact("Since", [day(m.since), isNum(m.days) ? " · " + plural(m.days, "trading day") : null])));
  }

  function curveSec(m, d) {
    var rows = arr(d.curve).filter(function (r) { return Array.isArray(r) && ymd(r[0]) && (isNum(r[1]) || isNum(r[2])); });
    if (rows.length < 2) return sec("curve", "Curve", "КРИВАЯ", null, empty("The curve starts with the second close."), "w");
    var isIx = m.kind === "index", st = S.state || {}, name = txt(m.title) || "model", ixName = txt(obj(st.index).title) || "index";
    var start = isNum(st.start_cash) && st.start_cash > 0 ? st.start_cash : isNum(rows[0][1]) ? rows[0][1] : null;   /* both lines start from the same money */
    return sec("curve", "Curve", "КРИВАЯ", span(rows[0][0], rows[rows.length - 1][0]), [
      h("div", { class: "legend" }, key("model", name), isIx ? null : key("index", ixName), isIx ? null : key("up", "ahead"), isIx ? null : key("down", "behind")),
      curveChart(rows, { label: "Value of " + name + (isIx ? "" : " against " + ixName), start: start, nameA: isIx ? "value" : "model", nameB: "index", single: isIx })
    ], "w");
  }

  function verdictSec(m) {
    var v = m.verdict && typeof m.verdict === "object" ? m.verdict : null, body;
    if (!v) {
      body = [h("p", { class: "vd-line" }, lamp("off"), h("b", null, "Not scored")),
        h("p", { class: "vd-say" }, m.kind === "index" ? "This is the bar itself: there is nothing to score." : "This model makes no ranking or forecast that could be checked later, so only its record against the index speaks.")];
    } else {
      var l = lampOf(v), n = isNum(v.n) ? plain(v.n, 0) : null, hz = isNum(v.horizon) ? plural(v.horizon, "trading day") : "days";
      var lead = l === "likely" ? "Skill is likely" : l === "unlikely" ? "Skill is unlikely" : "Too early to tell";
      body = [h("p", { class: "vd-line is-" + l }, lamp(VERDICT[l].lamp), h("b", null, txt(v.text) || VERDICT[l].word)),
        h("p", { class: "vd-say" }, lead + ": what it said each evening is checked against what the stocks did over the next " + hz +
          (n ? ", and over the " + n + " scored so far" : ", and so far") + " the match is ",
          fig(v.ic, 3, true), " (0 is chance, 1 is perfect)", isNum(v.t) ? [" with the evidence at ", fig(v.t, 2, false), ", where about 2 or more rules out luck."] : ".")];
    }
    body.push(lampFold("lamp-" + txt(m.id)));
    return sec("verdict", "Verdict", "ВЕРДИКТ", v && isNum(v.horizon) ? plural(v.horizon, "day") + " ahead" : null, body);
  }

  function spendSec() {
    var sp = S.state && S.state.spend && typeof S.state.spend === "object" ? S.state.spend : null;
    if (!sp || !isNum(sp.used) || !isNum(sp.cap)) return null;
    var mo = /^(\d{4})-(\d{2})$/.exec(txt(sp.month)), days = arr(sp.by_day).filter(function (x) { return Array.isArray(x) && ymd(x[0]) && isNum(x[1]); });
    var left = Math.max(0, sp.cap - sp.used), frac = sp.cap > 0 ? sp.used / sp.cap : 0;
    return sec("spend", "Spend", "РАСХОД", mo && MONTH[+mo[2] - 1] ? MONTH[+mo[2] - 1] + " " + mo[1] : null, [
      txt(sp.stopped) ? h("p", { class: "stop", role: "status" }, h("b", { class: "p-inv" }, "Stopped"), " " + stopWords(sp.stopped) + ". It holds what it has and makes no more calls until the allowance is back.") : null,
      h("div", { class: "spend-fig" }, dot("$" + nf(sp.used, 2)), h("span", { class: "of" }, "of ", usd(sp.cap, 2), " this month")),
      meter(frac, "Spent " + usdText(sp.used) + " of " + usdText(sp.cap)),
      h("dl", { class: "facts" }, fact("Left", usd(left)), fact("Today", usd(sp.today)), fact("Calls", count(sp.calls)), fact("Cap", usd(sp.cap))),
      days.length > 1 ? barsChart(days.map(function (x) { return { v: x[1], x: dayText(x[0]) }; }), {
        key: "spend", h: 110, label: "Spend a day this month", axis: function (v) { return usdText(v, v % 1 ? 2 : 0); },
        read: function (it) { return [h("b", null, it.x), " ", h("span", { class: "rd-i" }, "spent ", usd(it.v))]; }
      }) : null
    ]);
  }

  function memoSec(d) {
    var memo = d.memo && typeof d.memo === "object" ? d.memo : null;
    if (!memo || !txt(memo.text)) return null;
    return sec("memo", "Memo", "ЗАПИСКА", dayText(memo.date) || null, h("div", { class: "md" }, md(memo.text)));
  }

  function pendingSec(m, d) {
    var p = arr(d.pending).filter(function (x) { return x && typeof x === "object"; });
    return sec("pending", "Orders for the next open", "ЗАЯВКИ", p.length ? plural(p.length, "order") + ", filled at the open" : null,
      p.length ? table("pending", [["stock", "Stock"], ["w", "To", true], ["why", "Why"]], p.map(function (x) {
        return [stock(x.symbol, x.name, x.side), isNum(x.weight) ? (x.weight > 0 ? pct(x.weight, { dp: 1, sign: false }) : h("span", { class: "n flat" }, "none")) : nil(), why(x.why)];
      })) : empty("No orders are waiting for the next open."));
  }

  function rankRows(list, max) {
    return list.map(function (x) {
      var sc = isNum(x.score) ? x.score : null;
      return [isNum(x.rank) ? plain(x.rank, 0) : nil(), stock(x.symbol, x.name),
        [h("span", { class: "scb", "aria-hidden": "true", css: { "--v": (sc == null ? 0 : Math.min(100, Math.abs(sc) / (max || 1) * 100)).toFixed(1) + "%" } }), fig(sc, 2, true)],
        x.held ? h("span", { class: "held" }, lamp("on"), "held") : h("span", { class: "nil" }, h("span", { "aria-hidden": "true" }, DASH), h("span", { class: "sr" }, "not held"))];
    });
  }
  function todaySec(d) {
    var t = d.today && typeof d.today === "object" ? d.today : null;
    if (!t) return null;
    var top = arr(t.top).filter(Boolean), bot = arr(t.bottom).filter(Boolean), max = 0;
    if (!top.length && !bot.length) return null;
    top.concat(bot).forEach(function (x) { if (isNum(x.score)) max = Math.max(max, Math.abs(x.score)); });
    var cols = [["rank", "#", true], ["stock", "Stock"], ["score", "Score", true], ["held", "Held"]];
    return sec("today", "Today's ranking", "РЕЙТИНГ", dayText(t.date) || null, [
      txt(t.note) ? h("p", { class: "cap" }, txt(t.note)) : null,
      h("div", { class: "duo" },
        top.length ? h("div", null, h("h4", { class: "sub" }, "Top"), table("rank", cols, rankRows(top, max))) : null,
        bot.length ? h("div", null, h("h4", { class: "sub" }, "Bottom"), table("rank", cols, rankRows(bot, max))) : null)
    ], "w");
  }

  function historySec(m, d) {
    var hs = arr(d.ranking_history).filter(function (x) { return x && isNum(x.ic); });
    if (!hs.length) return null;
    var sum = 0, up = 0, hz = m.verdict && isNum(m.verdict.horizon) ? m.verdict.horizon : null;
    hs.forEach(function (x) { sum += x.ic; if (x.ic > 0) up++; });
    return sec("history", "Past rankings", "ПРОВЕРКА", plural(hs.length, "evening") + " scored", [
      h("p", { class: "cap" }, "Each column is one evening: how well that ranking matched what the stocks did" + (hz ? " over the next " + plural(hz, "trading day") : " next") +
        ". Above the line it was more right than wrong; 0 is chance. ", h("b", null, up + " of " + hs.length), " were above, the average is ", fig(sum / hs.length, 3, true), "."),
      h("div", { class: "legend" }, key("cells-up", "more right than wrong"), key("cells-down", "more wrong"), key("avg", "average")),
      barsChart(hs.map(function (x) { return { v: x.ic, x: dayText(x.date), n: x.n }; }), {
        key: "history", h: 170, signed: true, avg: sum / hs.length, label: "How each evening's ranking held up",
        read: function (it) { return [h("b", null, it.x || DASH), " ", h("span", { class: "rd-i" }, "score ", fig(it.v, 3, true)), isNum(it.n) ? [" ", h("span", { class: "rd-i" }, plain(it.n, 0) + " stocks ranked")] : null]; }
      })
    ], "w");
  }

  function positionsSec(m, d) {
    var ps = arr(d.positions).filter(function (x) { return x && typeof x === "object"; }), max = 0;
    ps.forEach(function (x) { if (isNum(x.weight)) max = Math.max(max, x.weight); });
    var note = ps.length ? (isNum(m.n_pos) && m.n_pos !== ps.length ? ps.length + " listed of " + m.n_pos : plural(ps.length, "stock")) + (isNum(m.cash) ? ", cash " + plain(m.cash * 100, 1) + NBSP + "%" : "") : null;
    return sec("positions", "Holds", "ПОРТФЕЛЬ", note,
      ps.length ? table("positions", [["stock", "Stock"], ["w", "Share", true], ["val", "Value", true], ["held", "Held since"], ["ret", "Return", true], ["why", "Why"]], ps.map(function (x) {
        return [stock(x.symbol, x.name),
          [h("span", { class: "scb", "aria-hidden": "true", css: { "--v": (isNum(x.weight) && max ? Math.min(100, x.weight / max * 100) : 0).toFixed(1) + "%" } }), pct(x.weight, { dp: 1, sign: false })],
          usd(x.value), [day(x.since), isNum(x.days) ? h("span", { class: "days" }, " · " + plural(x.days, "day")) : null], pct(x.ret, { tone: true }), why(x.why)];
      }), { more: "positions:" + txt(m.id), noun: "holdings" }) : empty("It holds nothing: the book is all cash."), "w");
  }

  function tradesSec(m, d) {
    var ts = arr(d.trades).filter(function (x) { return x && typeof x === "object"; });
    return sec("trades", "Trades", "СДЕЛКИ", ts.length ? plural(ts.length, "trade") + ", newest first" : null,
      ts.length ? table("trades", [["date", "Day"], ["stock", "Stock"], ["usd", "Amount", true], ["sh", "Shares", true], ["px", "Price", true], ["cost", "Cost", true], ["made", "Made", true], ["why", "Why"]],
        ts.map(function (x) {
          return [day(x.date), stock(x.symbol, x.name, x.side), usd(x.usd), fig(x.shares, 3), usd(x.price), usd(x.cost),
            x.side === "sell" || isNum(x.ret) ? pct(x.ret, { tone: true }) : { c: "is-na", v: h("span", { class: "nil" }, h("span", { "aria-hidden": "true" }, DASH), h("span", { class: "sr" }, "a buy: nothing made yet")) }, why(x.why)];
        }), { more: "trades:" + txt(m.id), noun: "trades" }) : empty("No trades yet."), "w");
  }

  function signalsSec(m, d) {
    var sg = arr(d.signals).filter(function (x) { return x && typeof x === "object"; });
    if (!sg.length) return null;
    var settled = sg.filter(function (x) { return isNum(x.excess); }), won = settled.filter(function (x) { return x.excess > 0; }).length;
    return sec("signals", "Forecasts", "ПРОГНОЗЫ", plural(sg.length, "forecast") + ", " + settled.length + " settled" + (settled.length ? ", " + won + " beat the index" : ""), [
      h("p", { class: "cap" }, "Each forecast is a chance that the stock beats the index over the days ahead. Once those days have passed, the result is what it made over the index."),
      table("signals", [["date", "Day"], ["stock", "Stock"], ["hz", "Days ahead", true], ["p", "Chance", true], ["why", "Why"], ["res", "Result", true]], sg.map(function (x) {
        return [day(x.date), stock(x.symbol), count(x.horizon), pct(x.p, { dp: 0, sign: false }), why(x.note),
          isNum(x.excess) ? pct(x.excess, { tone: true }) : h("span", { class: "open" }, "open")];
      }), { more: "signals:" + txt(m.id), noun: "forecasts" })
    ], "w");
  }

  function readsSec(m, d) {
    var rs = arr(d.reads).filter(function (x) { return x && typeof x === "object"; });
    if (!rs.length) return null;
    var k = "reads:" + txt(m.id), all = rs.length <= 5 || !!S.ui.more[k];
    var cards = rs.map(function (r, i) {
      var facts = arr(r.facts).map(txt).filter(Boolean), src = arr(r.sources).filter(function (x) { return x && typeof x === "object"; });
      function line(label, v) { return txt(v) ? h("p", { class: "rl" }, h("span", { class: "k" }, label), " ", txt(v)) : null; }
      function odds(label, p) { return isNum(p) ? h("span", { class: "od" }, h("span", { class: "k" }, label), " ", pct(p, { dp: 0, sign: false })) : null; }
      return h("article", { class: "readc", hidden: !all && i >= 4 },
        h("header", null, h("b", { class: "sym" }, txt(r.symbol) || DASH), h("span", { class: "nm" }, txt(r.name)), h("span", { class: "dt" }, dayText(r.date))),
        txt(r.thesis) ? h("p", { class: "thesis" }, txt(r.thesis)) : null,
        facts.length ? h("ul", { class: "factl" }, facts.map(function (f) { return h("li", null, f); })) : null,
        line("In the price", r.priced_in),
        line("Risks", r.risks),
        isNum(r.p5) || isNum(r.p20) || isNum(r.conviction) ? h("p", { class: "odds" }, h("span", { class: "k" }, "Chance to beat the index"),
          odds("in 5 days", r.p5), odds("in 20 days", r.p20),
          isNum(r.conviction) ? h("span", { class: "od od-c" }, h("span", { class: "k" }, "conviction"), " ", fig(r.conviction, 2), meter(r.conviction)) : null) : null,
        src.length ? h("ul", { class: "srcl", "aria-label": "Sources" }, src.map(function (x) {
          var u = safeUrl(x.url), t = txt(x.title) || (u ? new URL(u).hostname : "source");
          return h("li", null, u ? [link(u, t), h("span", { class: "host" }, " " + new URL(u).hostname.replace(/^www\./, ""))] : t);
        })) : null);
    });
    var out = [h("div", { class: "reads" }, cards)];
    if (rs.length > 5) {
      var btn = h("button", { type: "button", class: "more", "data-k": "more:" + k });
      var label = function () { btn.textContent = S.ui.more[k] ? "Show the first 4 only" : "Show all " + rs.length + " reads"; btn.setAttribute("aria-expanded", S.ui.more[k] ? "true" : "false"); };
      btn.addEventListener("click", function () { S.ui.more[k] = !S.ui.more[k]; cards.forEach(function (c, i) { c.hidden = !S.ui.more[k] && i >= 4; }); label(); });
      label();
      out.push(btn);
    }
    return sec("reads", "Reads", "ЧТЕНИЕ", plural(rs.length, "company", "companies") + " read in depth", out, "w");
  }

  function backtestSec(m, d) {
    var b = m.backtest && typeof m.backtest === "object" ? m.backtest : null;
    if (!b) {
      return sec("backtest", "Backtest", "ИСПЫТАНИЕ", null, empty(m.kind === "agent"
        ? "No backtest: the model has already read the past, so only live paper counts."
        : hasRecord(m) ? "This model has no backtest." : "No backtest yet."));
    }
    var e = b.exam && typeof b.exam === "object" ? b.exam : null, scale = 0, rows = [];
    [b.ci, e && e.ci].forEach(function (ci) { arr(ci).forEach(function (v) { if (isNum(v)) scale = Math.max(scale, Math.abs(v)); }); });
    function ci(x) {
      var c = arr(x && x.ci);
      if (!isNum(c[0]) || !isNum(c[1])) return nil();
      return [h("span", { class: "ci" }, pct(c[0]), " to ", pct(c[1])), rangeBar(c[0], x.vs_index_ann, c[1], scale),
        c[0] <= 0 && c[1] >= 0 ? h("span", { class: "ci-n" }, "spans zero: could be luck") : h("span", { class: "ci-n" }, c[0] > 0 ? "clear of zero" : "below zero")];
    }
    function add(label, f) { rows.push([label, f(b), e ? f(e) : null]); }
    add("Years", function (x) { return span(x.from, x.to, true) || nil(); });
    add("Return a year", function (x) { return pct(x.ret_ann, { tone: true }); });
    add("Index a year", function (x) { return pct(x.index_ann); });
    add("vs index a year", function (x) { return pct(x.vs_index_ann, { tone: true }); });
    add("Likely range", ci);
    if (isNum(b.ic) || (e && isNum(e.ic))) add("Ranking score", function (x) { return fig(x.ic, 3, true); });
    if (isNum(b.ic_t) || (e && isNum(e.ic_t))) add("Evidence", function (x) { return fig(x.ic_t, 1); });
    var years = arr(d && d.by_year).filter(function (y) { return y && (isNum(y.ret) || isNum(y.index)); }), ys = 0;
    years.forEach(function (y) { if (isNum(y.ret) && isNum(y.index)) ys = Math.max(ys, Math.abs(y.ret - y.index)); });
    return sec("backtest", "Backtest", "ИСПЫТАНИЕ", "history replayed with costs", [
      h("table", { class: "tb tb-bt" },
        h("thead", null, h("tr", null, h("td"), h("th", { scope: "col" }, "Backtest"), e ? h("th", { scope: "col" }, "Exam", h("span", { class: "thn" }, " held back until the design was frozen")) : null)),
        h("tbody", null, rows.map(function (r) { return h("tr", null, h("th", { scope: "row" }, r[0]), h("td", { "data-l": "Backtest" }, r[1]), e ? h("td", { "data-l": "Exam" }, r[2]) : null); }))),
      e ? null : empty("The exam has not been sat: its stretch stays locked until the design is frozen."),
      h("dl", { class: "facts" },
        fact("Worst drop", pct(b.max_dd)),
        fact("Return for its risk", [fig(b.sharpe, 2), h("span", { class: "u" }, " Sharpe")]),
        fact("Turns over", isNum(b.turnover) ? plain(b.turnover, 1) + TIMES + " a year" : nil()),
        fact("Costs took", isNum(b.cost_drag) ? [pct(b.cost_drag, { sign: false }), " a year"] : nil())),
      years.length ? [h("h4", { class: "sub" }, "Year by year"),
        table("years", [["year", "Year"], ["ret", "Model", true], ["idx", "Index", true], ["vs", "vs index", true]], years.map(function (y) {
          var gap = isNum(y.ret) && isNum(y.index) ? y.ret - y.index : null;
          return [txt(y.year) || DASH, pct(y.ret, { tone: true }), pct(y.index), [pct(gap, { tone: true }), sides(gap, ys)]];
        }))] : null,
      txt(d && d.report) ? fold("report:" + txt(m.id), "Read the full report", h("div", { class: "md" }, md(d.report))) : null
    ], "w");
  }

  function paintBook(reading) {
    var box = $("detail"), id = S.open;
    S.charts = [];
    if (!id) { box.hidden = true; fill(box, []); return; }
    var b = S.book[id] || {}, d = b.data, sum = S.state ? model(id) : null, o = order(), kids;
    box.hidden = false;
    box.setAttribute("data-id", id);
    if (!sum && !d) {
      var gone = S.state && b.err;
      kids = [h("header", { class: "bk-head" },
        h("nav", { class: "bk-nav", "aria-label": "Models" }, h("a", { class: "back", href: "#/", "data-k": "back" }, h("span", { "aria-hidden": "true" }, "‹ "), "All models")),
        h("div", { class: "bk-title" }, h("h2", { id: "book-title", tabindex: "-1", "data-k": "title" }, gone ? "No such model" : "Reading the book")),
        gone ? h("p", { class: "what" }, "The engine has no model called “", id, "”.") : h("p", { class: "what" }, h("span", { class: "p-cursor", "aria-hidden": "true" })))];
    } else {
      var m = {}, k;
      for (k in obj(d)) m[k] = d[k];
      for (k in obj(sum)) m[k] = sum[k];                    /* the minute-old state is the newer of the two */
      var rec = hasRecord(m), isAgent = m.kind === "agent";
      kids = [bookHead(m, o)];
      if (b.err && d) kids.push(h("p", { class: "st-fault", role: "alert" }, lamp("alarm"), h("span", null, h("b", null, "Stale."), " This book could not be read again; these are the last figures that came through.")));
      if (!d) {
        kids.push(b.err
          ? h("p", { class: "st-fault", role: "alert" }, lamp("alarm"), h("span", null, h("b", null, "Can't read this book."), " The page tries again with the next refresh."))
          : h("p", { class: "empty" }, "Reading the book", h("span", { class: "p-cursor", "aria-hidden": "true" })));
      } else {
        var body = [];
        /* the order is the order of reading; on a PC the narrow blocks pair up: verdict beside the orders,
           and for the agent its spend beside the verdict, its memo beside the orders */
        if (rec) {
          if (isAgent) body.push(spendSec(), verdictSec(m), curveSec(m, d), memoSec(d));
          else body.push(curveSec(m, d), verdictSec(m));
          body.push(pendingSec(m, d), todaySec(d), historySec(m, d), positionsSec(m, d), tradesSec(m, d));
          if (isAgent) body.push(readsSec(m, d));
          body.push(signalsSec(m, d), backtestSec(m, d));
        } else {
          if (isAgent) body.push(spendSec());
          body.push(sec("record", "Record", "ЗАПИСЬ", null, empty("No record yet. Its book opens when it starts to trade.")));
          if (isAgent || m.backtest) body.push(backtestSec(m, d));
        }
        kids.push(h("div", { class: "bk-body" }, body));
      }
    }
    fill(box, kids);
    S.charts.forEach(function (f) { f(); });
    if (!reading && d && S.painted.book !== id) {          /* just opened: the sheet may plot the curve once (.is-fresh) */
      S.painted.book = id;
      strike(box.querySelectorAll(".bk-head > *, .bk-body > .sec"), 55);
      box.classList.add("is-fresh");
      clearTimeout(S.fresh);
      S.fresh = setTimeout(function () { box.classList.remove("is-fresh"); }, 1500);
    }
  }

  /* ── which model is open: the address says it, so a reload and the back button keep it ───────────────── */

  /* the tab says which model is open, so the browser's history list reads as names */
  function name() {
    var m = S.open && S.state ? model(S.open) : null;
    document.title = (m && txt(m.title) ? txt(m.title) + " \u00b7 " : "") + "Stocks";
  }
  function marks() {
    var rows = document.querySelectorAll("#board .row-a[data-k]");
    for (var i = 0; i < rows.length; i++) {
      if (rows[i].getAttribute("data-k") === "row:" + S.open) rows[i].setAttribute("aria-current", "true"); else rows[i].removeAttribute("aria-current");
    }
  }
  function route(first) {
    var id = null, was = S.open;
    if ((location.hash || "").slice(0, 2) === "#/") { try { id = decodeURIComponent(location.hash.slice(2)); } catch (e) { id = null; } }
    if (id != null && !okId(id)) id = null;
    if (id === was && !first) return;
    S.open = id;
    S.painted.book = null;
    document.body.setAttribute("data-view", id ? "book" : "board");
    marks();
    name();
    paintBook();
    if (id) {
      getBook(id, !!(S.book[id] && S.book[id].data));
      window.scrollTo(0, 0);
      if (!first) { var t = $("book-title"); if (t) try { t.focus({ preventScroll: true }); } catch (e) { /* no focus */ } }
    } else if (!first) {
      var at = history.state && isNum(history.state.boardY) ? history.state.boardY : S.boardY;
      window.scrollTo(0, at);
      var rows = document.querySelectorAll("#board .row-a[data-k]");
      for (var i = 0; was && i < rows.length; i++) {
        if (rows[i].getAttribute("data-k") === "row:" + was) { try { rows[i].focus({ preventScroll: true }); } catch (e) { /* no focus */ } break; }
      }
    }
  }

  window.addEventListener("hashchange", function () { route(false); });
  /* leaving the list for a model: note where the list was, in the list's own history entry */
  document.addEventListener("click", function (e) {
    if (S.open || !e.target || !e.target.closest || !e.target.closest("#board a.row-a")) return;
    S.boardY = window.scrollY;
    try { history.replaceState({ boardY: window.scrollY }, ""); } catch (err) { /* the address still works */ }
  }, true);
  document.addEventListener("keydown", function (e) {
    if (e.key !== "Escape" || !S.open || e.defaultPrevented) return;
    var menu = $("plint-menu");
    if (menu && !menu.hidden) return;                       /* the strip's own list closes first */
    location.hash = "#/";
  });
  document.addEventListener("visibilitychange", function () { if (!document.hidden && Date.now() - S.tried > 15000) getState(); });
  var rz = 0;
  window.addEventListener("resize", function () { clearTimeout(rz); rz = setTimeout(function () { S.charts.forEach(function (f) { f(); }); }, 120); });

  try { history.scrollRestoration = "manual"; } catch (e) { /* older browsers */ }
  route(true);
  getState();
})();
