// Accrete · how numbers and times are written. Never more than four digits; big numbers get a name or an exponent.
(function (root) {
  "use strict";
  const node = typeof module !== "undefined" && module.exports;
  const Big = node ? require("./big.js") : root.Big;
  const NAMES = ["", "K", "M", "B", "T", "Qa", "Qi", "Sx", "Sp", "Oc", "No", "Dc"];
  let notation = "names";

  const trim = (x, d) => { const s = x.toFixed(d); return s.indexOf(".") < 0 ? s : s.replace(/0+$/, "").replace(/\.$/, ""); };
  // three significant digits of a mantissa that has been scaled to 1 ≤ x < 1000
  const sig3 = (x) => (x >= 99.95 ? String(Math.round(x)) : x >= 9.995 ? trim(x, 1) : trim(x, 2));

  function num(v) {
    const b = Big.from(v);
    if (b.isZero()) return "0";
    if (b.m < 0) return "-" + num(new Big(-b.m, b.e));
    if (b.e < 3 && b.toNumber() < 999.5) {
      const x = b.toNumber();
      if (x >= 99.95) return String(Math.round(x));
      if (x >= 9.995) return trim(x, 1);
      if (x >= 0.995) return trim(x, 2);
      return x < 0.0005 ? "0" : trim(x, 3);
    }
    let m = b.m, e = b.e;
    if (e < 3) { m = 1; e = 3; }                             // 999.6 reads as 1K
    if (m >= 9.995) { m /= 10; e += 1; }                     // 9.999e5 reads as 1.00e6, not 10.00e5
    const g = Math.floor(e / 3);
    // "names": K M B T, then powers of ten · "long": names as far as they go (Qa Qi Sx Sp Oc No Dc) · "sci": powers only
    const named = notation === "long" ? NAMES.length - 1 : notation === "names" ? 4 : 0;
    if (g <= named) { const x = m * Math.pow(10, e - g * 3); if (x < 999.5) return sig3(x) + NAMES[g]; if (g < named) return sig3(x / 1000) + NAMES[g + 1]; }
    return trim(m, 2) + "e" + e;
  }
  // whole things: bodies owned, stars, light
  function int(v) {
    const b = Big.from(v);
    if (b.e < 4) return String(Math.round(b.toNumber())).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    return num(b);
  }
  function mult(x) { return "x" + (x >= 100 ? num(x) : trim(x, x < 10 ? 2 : 1)); }
  function pct(x) { const p = x * 100; return (p >= 99.95 ? "100" : p >= 10 ? String(Math.round(p)) : trim(p, 1)) + "%"; }
  // how long: "42 s", "3 min", "2 h 5 min", "3 d 4 h"
  function time(sec) {
    if (!isFinite(sec)) return "a long time";
    sec = Math.round(Math.max(0, sec));
    if (sec < 1) return "now";
    if (sec < 60) return sec + " s";
    if (sec < 600) { const m = Math.floor(sec / 60), s = sec % 60; return s ? m + " min " + s + " s" : m + " min"; }
    const min = Math.round(sec / 60);
    if (min < 60) return min + " min";
    if (min < 1440) { const h = Math.floor(min / 60), m = min % 60; return m ? h + " h " + m + " min" : h + " h"; }
    const hrs = Math.round(sec / 3600);
    if (hrs < 24 * 365) { const d = Math.floor(hrs / 24), h = hrs % 24; return h ? d + " d " + h + " h" : d + " d"; }
    return "years";
  }
  const F = { num, int, mult, pct, time, set(n) { notation = n; }, get() { return notation; } };
  if (node) module.exports = F; else root.Fmt = F;
})(typeof self !== "undefined" ? self : this);
