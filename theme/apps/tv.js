/* Plint Terminal · TV remote: the strip in the page's own header, and a key that stays lit (inverse video, class
   p-hit) for a moment after a tap, since a thumb leaves before :active can be seen. One delegated listener, no
   polling; fails silently. */
(function () {
  "use strict";
  if (window.Plint) window.Plint.mount = ["header.tv-head"];
  try {
    var held = null, timer = 0;
    var off = function () { if (held) { held.classList.remove("p-hit"); held = null; } };
    var up = function () { clearTimeout(timer); timer = setTimeout(off, 130); };
    document.addEventListener("pointerdown", function (e) {
      var b = e.target && e.target.closest && e.target.closest("button.btn, button.tile");
      clearTimeout(timer); off();
      if (b) { held = b; b.classList.add("p-hit"); }
    }, { passive: true });
    document.addEventListener("pointerup", up, { passive: true });
    document.addEventListener("pointercancel", up, { passive: true });
    document.addEventListener("touchstart", function () {}, { passive: true });   /* lets :active apply on iOS */
  } catch (e) {}
})();
