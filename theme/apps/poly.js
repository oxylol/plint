/* Plint Terminal · Polybot. The page is ours but is served from the bot's container, so the few things CSS cannot
   do are done here, from the rendered DOM only (no requests): the strip in the header, the dimmed second half of
   the section headings, the setup picker kept where it was on a phone, and a profit/loss cell bar under each
   strategy with a one-line readout. Everything fails silently; the page re-renders itself every 30 s by innerHTML,
   so the two observers below watch just the two containers they repair. */
(function () {
  "use strict";
  if (window.Plint) window.Plint.mount = ["header.poly-head"];

  function ready(fn) { if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn); else fn(); }
  function num(s) { var n = parseFloat(String(s).replace(/−/g, "-").replace(/[^0-9.\-]/g, "")); return isFinite(n) ? n : null; }

  /* "Forecast · tomorrow's temperatures" → the name lit, the rest dim (static text, written once) */
  function headings() {
    var hs = document.querySelectorAll("section > h2"), i, t, k, s;
    for (i = 0; i < hs.length; i++) {
      t = hs[i].firstChild;
      if (!t || t.nodeType !== 3) continue;
      k = t.nodeValue.indexOf(" · ");
      if (k < 1) continue;
      s = document.createElement("span");
      s.className = "p-dim";
      s.textContent = t.nodeValue.slice(k + 3).replace(/\s+$/, "");
      t.nodeValue = t.nodeValue.slice(0, k);
      hs[i].insertBefore(s, t.nextSibling);
    }
  }

  /* the picker is one swipeable row on a phone; a re-render empties it and the row jumps back to the start */
  function picker() {
    var p = document.getElementById("picker"), left = 0, cur = null;
    if (!p || !window.MutationObserver) return;
    p.addEventListener("scroll", function () { left = p.scrollLeft; }, { passive: true });
    function fix() {
      var b = p.querySelector('button[aria-pressed="true"]'), v = b && b.getAttribute("data-v");
      if (p.scrollWidth <= p.clientWidth) return;
      if (b && v !== cur && (b.offsetLeft < p.scrollLeft + 16 || b.offsetLeft + b.offsetWidth > p.scrollLeft + p.clientWidth - 16)) {
        p.scrollLeft = Math.max(0, b.offsetLeft - p.offsetLeft - 40);
      } else if (v === cur) {
        p.scrollLeft = left;
      }
      cur = v; left = p.scrollLeft;
    }
    new MutationObserver(function () { try { fix(); } catch (e) {} }).observe(p, { childList: true });
    try { fix(); } catch (e) {}
  }

  /* profit and loss as cells either side of a zero tick, all on one scale, then the tally in a line */
  function bars() {
    var box = document.getElementById("compare"), t, rows, list = [], max = 0, i, r, v, c, tr, td, ahead = 0, behind = 0, best = null, worst = null, cols;
    if (!box || box.querySelector(".pl-read")) return;
    t = box.querySelector("table.compare");
    if (!t) return;
    rows = t.querySelectorAll("tr");
    for (i = 0; i < rows.length; i++) {
      c = rows[i].querySelector("td.big");
      if (!c) continue;
      v = num(c.textContent);
      if (v == null) continue;
      list.push({ tr: rows[i], v: v, name: (rows[i].cells[0].firstChild && rows[i].cells[0].firstChild.nodeValue || "").replace(/\s+$/, ""), text: c.textContent });
      if (Math.abs(v) > max) max = Math.abs(v);
    }
    if (!list.length || !max) return;
    cols = list[0].tr.cells.length;
    for (i = 0; i < list.length; i++) {
      r = list[i];
      if (r.v > 0) ahead++; else if (r.v < 0) behind++;
      if (!best || r.v > best.v) best = r;
      if (!worst || r.v < worst.v) worst = r;
      tr = document.createElement("tr");
      tr.className = "pl-row" + (r.tr.className ? " " + r.tr.className : "");
      td = document.createElement("td");
      td.colSpan = cols;
      td.innerHTML = '<div class="pl-bar" aria-hidden="true"><i class="neg" style="--v:' + (r.v < 0 ? Math.max(2, Math.round(-r.v / max * 100)) : 0) + '%"></i><b></b><i class="pos" style="--v:' + (r.v > 0 ? Math.max(2, Math.round(r.v / max * 100)) : 0) + '%"></i></div>';
      tr.appendChild(td);
      r.tr.parentNode.insertBefore(tr, r.tr.nextSibling);
    }
    var esc = window.Plint && window.Plint.esc || function (s) { return String(s).replace(/[&<>"]/g, function (ch) { return "&#" + ch.charCodeAt(0) + ";"; }); };
    var d = document.createElement("div");
    d.className = "pl-read";
    d.innerHTML = "<b>" + ahead + "</b> ahead · <b>" + behind + "</b> behind · <b>" + (list.length - ahead - behind) + "</b> flat" +
      (best && best.v > 0 ? " · best " + esc(best.name) + " <b>" + esc(best.text) + "</b>" : "") +
      (worst && worst.v < 0 ? " · worst " + esc(worst.name) + ' <b><span class="down">' + esc(worst.text) + "</span></b>" : "");
    box.appendChild(d);
  }

  ready(function () {
    try { headings(); } catch (e) {}
    try { picker(); } catch (e) {}
    try {
      var box = document.getElementById("compare");
      if (box && window.MutationObserver) new MutationObserver(function () { try { bars(); } catch (e) {} }).observe(box, { childList: true });
      bars();
    } catch (e) {}
  });
})();
