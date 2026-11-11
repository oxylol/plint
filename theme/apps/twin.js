/* Plint Terminal · CNC Twin extras. Loaded by plint.js after the app's own module.
   1. the strip goes into the header slot the app keeps for it;
   2. one scheme: the app picks its chart palettes with matchMedia('(prefers-color-scheme: dark)')
      (js/plot/colormap.js mode()). The look is always dark, so that question is answered "dark" on
      any device; nothing else about matchMedia changes;
   3. short numeric tile values are tagged so the sheet can set them in the dot face.
   No polling: one observer on <main>, coalesced. Everything fails silently. */
(function () {
  "use strict";
  var P = window.Plint;
  try { if (P && typeof P === "object") P.mount = [".app-head .pill-slot"]; } catch (e) {}

  /* ── one scheme ── */
  var wasLight = false;
  try {
    var real = window.matchMedia;
    if (typeof real === "function" && !real.__plint) {
      wasLight = !real.call(window, "(prefers-color-scheme: dark)").matches;
      var locked = function (q) {
        var s = String(q);
        if (/prefers-color-scheme/i.test(s)) {
          s = s.replace(/\(\s*prefers-color-scheme\s*:\s*dark\s*\)/gi, "(min-width: 0px)")
               .replace(/\(\s*prefers-color-scheme\s*:\s*light\s*\)/gi, "(max-width: 0px)");
        }
        return real.call(window, s);
      };
      locked.__plint = true;
      window.matchMedia = locked;
    }
  } catch (e) {}

  /* a chart drawn before this file arrived on a light device used the light palette: draw the
     route once more (the app re-renders on hashchange; GET only). Not on the compare page. */
  try {
    if (wasLight && document.querySelector("main canvas") && !/^#\/compare/.test(location.hash)) {
      setTimeout(function () {
        try { window.dispatchEvent(new HashChangeEvent("hashchange")); } catch (e) {}
      }, 0);
    }
  } catch (e) {}

  /* ── figures ── */
  var NUM = /^[+\-\u2212]?\d[\d.,:]*(\s?[^\s\d]{1,4})?$/;
  function tag() {
    pending = 0;
    try {
      var list = document.querySelectorAll(".tile-value:not([data-tw])");
      for (var i = 0; i < list.length; i++) {
        var el = list[i], t = (el.textContent || "").trim();
        el.setAttribute("data-tw", "1");
        if (!el.firstElementChild && t.length <= 11 && NUM.test(t)) el.classList.add("tw-num");
      }
    } catch (e) {}
  }
  var pending = 0;
  function soon() { if (!pending) pending = setTimeout(tag, 120); }
  try {
    var main = document.getElementById("app") || document.body;
    if (main && window.MutationObserver) {
      new MutationObserver(function (recs) {
        // playback rewrites readouts every frame: only look when elements were added
        for (var i = 0; i < recs.length; i++) {
          var a = recs[i].addedNodes;
          for (var j = 0; j < a.length; j++) if (a[j].nodeType === 1) { soon(); return; }
        }
      }).observe(main, { childList: true, subtree: true });
    }
    soon();
  } catch (e) {}
})();
