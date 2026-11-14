/* Plint Terminal · Stocks. The page is ours and draws itself (apps/stocks/static), and it calls the shared
   helpers itself (Plint.strike when its figures first appear). All that is left to do from here is to say where
   the strip goes: into the page's own header. No requests, no polling; fails silently. */
(function () {
  "use strict";
  var P = window.Plint;
  if (!P || typeof P !== "object") return;
  try {
    P.mount = ["header.stk-head"];
    /* the strip is put in place when the page changes: now that it knows where to go, give it one change */
    var nudge = document.createComment("");
    document.body.appendChild(nudge);
    document.body.removeChild(nudge);
  } catch (e) {}
})();
