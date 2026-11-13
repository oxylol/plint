// Accrete · unit checks of the arithmetic the game counts with.   node test/unit.js   (the rules: test/rules.js)
const Big = require(require("path").join(__dirname, "../public/js/big.js"));
let bad = 0, n = 0;
const ok = (name, cond, detail) => { n++; if (!cond) { bad++; console.log("FAIL  " + name + (detail ? "  — " + detail : "")); } };
const near = (a, b, rel = 1e-12) => Math.abs(a - b) <= rel * Math.max(1, Math.abs(a), Math.abs(b));
const B = (x) => Big.from(x);

// small exact arithmetic
ok("7+3=10", B(7).add(3).eq(10) && B(7).add(3).toNumber() === 10);
ok("10-3-7=0", B(10).sub(3).sub(7).isZero());
ok("0.1*3", near(B(0.1).mul(3).toNumber(), 0.3));
ok("norm 999.5", B(999.5).e === 2 && near(B(999.5).m, 9.995));
ok("norm 1e-5", B(1e-5).e === -5 && near(B(1e-5).m, 1));
ok("norm 1e308*10", B(1e308).mul(10).e === 309);
ok("tiny", B(5e-320).e === -320);
ok("zero", B(0).isZero() && B(0).add(0).isZero() && B(0).mul(1e300).isZero());

// random against doubles
let seed = 12345; const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
for (let i = 0; i < 20000; i++) {
  const a = Math.pow(10, rnd() * 40 - 20) * (rnd() < 0.2 ? -1 : 1), b = Math.pow(10, rnd() * 40 - 20) * (rnd() < 0.2 ? -1 : 1);
  ok("add", near(B(a).add(b).toNumber(), a + b, 1e-10), `${a} + ${b}`);
  ok("sub", near(B(a).sub(b).toNumber(), a - b, 1e-10), `${a} - ${b}`);
  ok("mul", near(B(a).mul(b).toNumber(), a * b), `${a} * ${b}`);
  ok("div", near(B(a).div(b).toNumber(), a / b), `${a} / ${b}`);
  ok("cmp", B(a).cmp(b) === (a < b ? -1 : a > b ? 1 : 0), `${a} ? ${b}`);
  if (a > 0) { const p = rnd() * 6 - 3; ok("pow", near(B(a).pow(p).toNumber(), Math.pow(a, p), 1e-9), `${a}^${p}`); }
  if (a > 0) ok("log10", near(B(a).log10(), Math.log10(a), 1e-12));
}

// past doubles
const huge = B("1e400").mul("3e500");
ok("1e400*3e500", huge.e === 900 && near(huge.m, 3));
ok("huge add small", huge.add(1).eq(huge));
ok("huge cmp", huge.gt("2.9e900") && huge.lt("3.1e900") && B("-1e900").lt(huge) && B("-1e900").lt("-1e899"));
ok("huge pow", near(B("1e1000").pow(2.5).log10(), 2500));
ok("exp10", near(Big.exp10(1234.5).log10(), 1234.5));
ok("toNumber inf", huge.toNumber() === Infinity);
ok("json", Big.from(JSON.parse(JSON.stringify({ x: huge })).x).eq(huge));
ok("json small", Big.from(JSON.parse(JSON.stringify({ x: B(1234.5678) })).x).eq(1234.5678));

// floor
ok("floor 9.99", B(9.99).floor().eq(9));
ok("floor 10", B(10).floor().eq(10));
ok("floor 7+3", B(7).add(3).floor().eq(10));
ok("floor 0.3", B(0.3).floor().isZero());
ok("floor 1e20", B(1e20).floor().eq(1e20));

// geometric prices against a plain loop
for (let i = 0; i < 3000; i++) {
  const base = 1 + Math.floor(rnd() * 1000), r = 1.05 + rnd() * 0.3, own = Math.floor(rnd() * 60);
  const money = Math.pow(10, rnd() * 12);
  let k = 0, spent = 0;
  for (;;) { const c = base * Math.pow(r, own + k); if (spent + c > money) break; spent += c; k++; }
  const got = Big.geomMax(money, base, r, own);
  ok("geomMax", got === k || (got === k - 1 && near(spent, money, 1e-9)), `base ${base} r ${r} own ${own} money ${money}: ${got} vs ${k}`);
  ok("geomSum", near(Big.geomSum(base, r, own, k).toNumber(), spent, 1e-9));
  ok("geomSum affordable", Big.geomSum(base, r, own, got).lte(money));
}
ok("geomMax exact", Big.geomMax(10, 10, 1.15, 0) === 1);
ok("geomMax short", Big.geomMax(9.999, 10, 1.15, 0) === 0);

console.log(bad ? `${bad} of ${n} FAILED` : `all ${n} passed`);
process.exit(bad ? 1 : 0);
