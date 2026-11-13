// Big: numbers kept as mantissa × 10^exponent, so the game can count far past 1.8e308.
// Immutable; every operation returns a new Big. Works in the browser (window.Big) and in node.
(function (root) {
  "use strict";

  const P10 = [];
  for (let i = 0; i <= 308; i++) P10.push(Number("1e" + i));
  const pow10 = (n) => (n >= 0 && n <= 308 && Number.isInteger(n) ? P10[n] : Math.pow(10, n));

  class Big {
    constructor(m, e) { this.m = m; this.e = e; }

    static norm(m, e) {
      if (!m) return ZERO;
      if (!Number.isFinite(m) || Number.isNaN(e)) throw new Error("Big: not a number");
      const a = Math.abs(m);
      if (a >= 1 && a < 10) return new Big(m, e);
      let k = Math.floor(Math.log10(a));
      // scale by a whole power of ten each way, so 0.7 becomes 7 and not 6.999…
      m = k < -300 ? (m * 1e300) * pow10(-k - 300) : k < 0 ? m * pow10(-k) : m / pow10(k);
      const b = Math.abs(m);
      if (b >= 10) { m /= 10; k++; } else if (b < 1) { m *= 10; k--; }
      return new Big(m, e + k);
    }

    static from(x) {
      if (x instanceof Big) return x;
      if (typeof x === "number") return Big.norm(x, 0);
      if (typeof x === "string") {
        const i = x.indexOf("e");
        return i < 0 ? Big.norm(Number(x), 0) : Big.norm(Number(x.slice(0, i)), Number(x.slice(i + 1)));
      }
      if (x && typeof x.m === "number") return Big.norm(x.m, x.e);
      throw new Error("Big: cannot read " + x);
    }

    // 10^x for any real x
    static exp10(x) {
      const e = Math.floor(x);
      return Big.norm(Math.pow(10, x - e), e);
    }

    add(o) {
      o = Big.from(o);
      if (!this.m) return o;
      if (!o.m) return this;
      const hi = this.e >= o.e ? this : o, lo = hi === this ? o : this;
      const d = hi.e - lo.e;
      if (d > 17) return hi;
      // on the smaller one's scale while that stays exact: whole numbers add up to whole numbers
      if (d <= 15) return Big.norm(hi.m * pow10(d) + lo.m, lo.e);
      return Big.norm(hi.m + lo.m / pow10(d), hi.e);
    }
    sub(o) { o = Big.from(o); return this.add(new Big(-o.m, o.e)); }
    mul(o) { o = Big.from(o); return Big.norm(this.m * o.m, this.e + o.e); }
    div(o) {
      o = Big.from(o);
      if (!o.m) throw new Error("Big: divide by zero");
      return Big.norm(this.m / o.m, this.e - o.e);
    }
    // this^p for a plain number p; the base must not be negative
    pow(p) {
      if (!this.m) return p === 0 ? ONE : ZERO;
      if (this.m < 0) throw new Error("Big: negative base");
      return Big.exp10(this.log10() * p);
    }
    sqrt() { return this.pow(0.5); }
    log10() { return this.m > 0 ? Math.log10(this.m) + this.e : -Infinity; }
    ln() { return this.log10() * Math.LN10; }

    cmp(o) {
      o = Big.from(o);
      const sa = Math.sign(this.m), sb = Math.sign(o.m);
      if (sa !== sb) return sa < sb ? -1 : 1;
      if (!sa) return 0;
      if (this.e !== o.e) return (this.e < o.e ? -1 : 1) * sa;
      return this.m === o.m ? 0 : this.m < o.m ? -1 : 1;
    }
    gte(o) { return this.cmp(o) >= 0; }
    gt(o) { return this.cmp(o) > 0; }
    lte(o) { return this.cmp(o) <= 0; }
    lt(o) { return this.cmp(o) < 0; }
    eq(o) { return this.cmp(o) === 0; }
    max(o) { o = Big.from(o); return this.cmp(o) >= 0 ? this : o; }
    min(o) { o = Big.from(o); return this.cmp(o) <= 0 ? this : o; }
    isZero() { return !this.m; }

    toNumber() { return this.e > 308 ? (this.m < 0 ? -Infinity : Infinity) : this.e < -320 ? 0 : this.m * pow10(this.e); }
    floor() {
      if (this.e >= 15) return this;
      return Big.norm(Math.floor(this.toNumber() + 1e-9 * Math.max(1, Math.abs(this.toNumber()))), 0);
    }
    toJSON() { return this.m + "e" + this.e; }
    toString() { return this.toJSON(); }
  }

  const ZERO = new Big(0, 0), ONE = new Big(1, 0);
  Big.ZERO = ZERO; Big.ONE = ONE;
  Big.pow10 = pow10;

  // Sum of a geometric run of prices: base·r^n + … for k items starting at item n.
  Big.geomSum = function (base, r, n, k) {
    base = Big.from(base);
    if (k <= 0) return ZERO;
    if (r === 1) return base.mul(k);
    const first = base.mul(Big.from(r).pow(n));
    return first.mul(Big.from(r).pow(k).sub(1)).div(r - 1);
  };
  // How many items of that run `money` pays for.
  Big.geomMax = function (money, base, r, n) {
    money = Big.from(money); base = Big.from(base);
    const first = base.mul(Big.from(r).pow(n));
    if (money.lt(first)) return 0;
    if (r === 1) return Math.floor(money.div(first).toNumber());
    // k = floor(log_r(money·(r−1)/first + 1))
    const k = Math.floor(money.mul(r - 1).div(first).add(1).log10() / Math.log10(r) + 1e-9);
    // guard the edge: never claim one more than is affordable
    return Big.geomSum(base, r, n, k).lte(money) ? k : Math.max(0, k - 1);
  };

  if (typeof module !== "undefined" && module.exports) module.exports = Big;
  else root.Big = Big;
})(typeof self !== "undefined" ? self : this);
