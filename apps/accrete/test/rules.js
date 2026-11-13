// Accrete · checks of the rules: prices, time away, saving, the resets, how numbers are written.   node test/rules.js
const path = require("path");
const Big = require(path.join(__dirname, "../public/js/big.js"));
const G = require(path.join(__dirname, "../public/js/game.js"));
const D = require(path.join(__dirname, "../public/js/data.js"));
const F = require(path.join(__dirname, "../public/js/format.js"));
let bad = 0, n = 0;
const ok = (name, cond, detail) => { n++; if (!cond) bad++; console.log((cond ? "PASS  " : "FAIL  ") + name + (detail !== undefined && !cond ? "  — " + detail : "")); };
const near = (a, b, rel = 1e-12) => Math.abs(a - b) <= rel * Math.max(1, Math.abs(a), Math.abs(b));

// a state some way in, made by playing: buy what is cheapest, for some simulated minutes
function played(seed, secs) {
  const s = G.fresh(0, seed);
  for (let t = 0; t < secs; t++) {
    G.tick(s, 1, { hold: 1 });
    if (s.comet.live) G.catchComet(s);
    for (let i = 0; i < 6; i++) { const c = G.calc(s); const o = G.offers(s, c).filter((x) => x.what !== "shepherd" && G.afford(s.mass, x.cost)).sort((a, b) => a.cost.cmp(b.cost))[0]; if (!o) break; if (o.what === "kind") G.buy(s, o.k, 1); else G.buyUpgrade(s, o.id); }
  }
  G.drain();
  return s;
}
const copy = (s) => G.revive(JSON.parse(JSON.stringify(s)));

// prices
{
  const a = played(3, 600), b = copy(a);
  a.mass = Big.from(1e12); b.mass = Big.from(1e12);
  const price = G.cost(a, G.calc(a), 1, 10);
  G.buy(a, 1, 10); for (let i = 0; i < 10; i++) G.buy(b, 1, 1);
  ok("ten at once cost what ten singles cost", a.own[1] === b.own[1] && near(a.mass.toNumber(), b.mass.toNumber(), 1e-9) && near(1e12 - a.mass.toNumber(), price.toNumber(), 1e-9));
  const m = copy(a); m.mass = G.cost(m, G.calc(m), 2, 7).mul(1.0001); const before = m.own[2];
  G.buy(m, 2, "max"); ok("max buys what the mass pays for, no more", m.own[2] === before + 7, m.own[2] - before);
  const nb = copy(a); nb.mass = Big.from(1e15); const want = G.nextBump(nb.own[0]); G.buy(nb, 0, "bump"); ok("“to next x2” stops at the bump", nb.own[0] === want);
  const poor = copy(a); poor.mass = Big.from(1); ok("what cannot be paid for is not bought", G.buy(poor, 3, 1) === 0 && poor.mass.eq(1));
  ok("bumps at 10, 25, 50, 100, 150, 200 and every 50 after", G.bumps(9) === 0 && G.bumps(10) === 1 && G.bumps(25) === 2 && G.bumps(200) === 6 && G.bumps(249) === 6 && G.bumps(250) === 7 && G.nextBump(200) === 250 && G.prevBump(260) === 250 && G.nextBump(0) === 10);
}
// away = open
{
  const a = played(5, 900); a.comet.wait = 1e9; a.comet.live = null; a.surge = { mult: 1, left: 0 };
  const b = copy(a), start = a.life.mass;
  for (let i = 0; i < 3600; i++) G.tick(a, 1, {});
  const sum = G.away(b, 3600);
  ok("an hour away earns what an hour open earns, to within 0.2%", near(a.life.mass.toNumber(), b.life.mass.toNumber(), 2e-3), a.life.mass.toJSON() + " vs " + b.life.mass.toJSON());
  const d1 = copy(b), d2 = copy(b); for (let i = 0; i < 86400; i += 10) G.tick(d1, 10, {}); G.away(d2, 86400);
  ok("a day away earns what a day open earns, to within 1%", near(d1.life.mass.toNumber(), d2.life.mass.toNumber(), 1e-2), d1.life.mass.toJSON() + " vs " + d2.life.mass.toJSON());
  ok("the summary says what was gained", near(sum.mass.toNumber(), b.life.mass.sub(start).toNumber(), 1e-9) && sum.secs === 3600);
  const c1 = copy(b); c1.surge = { mult: 3, left: 30 };
  const r = G.calc(b).perSec.toNumber();
  G.away(c1, 100);
  ok("a surge that runs out while away counts for as long as it lasted", near(c1.life.mass.sub(b.life.mass).toNumber(), r * 3 * 30 + r * 70, 1e-6) && c1.surge.left === 0);
  // with shepherds the time away is done in steps: it may differ from open play, but not by much
  const h1 = played(5, 900), h2 = copy(h1); for (const x of [h1, h2]) { for (let k = 0; k < 3; k++) x.shep[k] = true; x.comet.wait = 1e9; }
  for (let i = 0; i < 3600; i++) G.tick(h1, 1, {});
  G.away(h2, 3600);
  const ratio = h2.life.mass.div(h1.life.mass).toNumber();
  ok("with shepherds, an hour away is within 15% of an hour open", ratio > 0.85 && ratio < 1.15, "away/open = " + ratio.toFixed(3));
  console.log("      with shepherds: away / open = " + ratio.toFixed(3));
}
// nothing lower after time away
{
  let fine = true;
  for (const secs of [0, 1, 59, 3600, 86400, 30 * 86400]) for (const herd of [false, true]) {
    const a = played(7, 1200); if (herd) for (let k = 0; k < 4; k++) a.shep[k] = true;
    const b = copy(a); G.away(b, secs);
    if (b.life.mass.lt(a.life.mass) || b.own.some((x, i) => x < a.own[i]) || b.life.bought < a.life.bought || G.count(b.ms) < G.count(a.ms) || G.count(b.up) < G.count(a.up) || (!herd && b.mass.lt(a.mass))) { fine = false; console.log("      lower after " + secs + " s" + (herd ? " herded" : "")); }
  }
  ok("nothing is lower after 0 s to 30 days away, herded or not", fine);
}
// saving
{
  const a = played(11, 1500), b = copy(a);
  ok("a save reads back as the same game", JSON.stringify(a) === JSON.stringify(b));
  for (let i = 0; i < 300; i++) { G.tick(a, 1, {}); G.tick(b, 1, {}); }
  ok("and plays on the same", JSON.stringify(a) === JSON.stringify(b));
  const old = JSON.parse(JSON.stringify(a)); delete old.wells; delete old.set; delete old.life.away;
  const r = G.revive(old); ok("an older save with parts missing is filled in", r.wells.hole === 0 && r.set.notation === "names" && r.life.away === 0 && r.own.length === D.KINDS.length);
}
// the resets
{
  const s = played(13, 1200); s.cloud.mass = Big.from(D.IGNITE.at).mul(150); s.shep[0] = true; const ms = G.count(s.ms), life = s.life.mass;
  G.tick(s, 60, {});
  ok("no reset is forced: an open ignition waits to be asked", s.life.ignitions === 0);
  const r = G.ignite(s);
  ok("ignition gives light and the class of the star", r && r.light.gte(1) && r.cls === "K" && s.classes.M && s.classes.K && s.stars === 1);
  ok("ignition keeps shepherds, milestones and the record; the cloud starts again", s.shep[0] && G.count(s.ms) >= ms && s.life.mass.gte(life) && s.own.every((x) => x === 0) && G.count(s.up) === 0);
  s.lightGal = Big.from(D.COLLAPSE.at).mul(100); s.light = Big.from(50); G.buyNode(s, "kindling");
  const d = G.collapse(s, "thin");
  ok("collapse: one dark matter, and one more for every tenfold of light", d && d.dark.eq(3) && s.law === "thin" && s.light.isZero() && !s.nodes.kindling && s.shep[0], d && d.dark.toJSON());
  ok("a law changes the rule while its galaxy lasts", G.calc(s).growth === 1.25);
  s.lightGal = Big.from(D.COLLAPSE.at); G.collapse(s, null);
  ok("a law collapsed is kept for good", s.laws.thin === true && G.calc(s).growth === 1.14);
  s.darkUni = G.bangAt(s); s.wells.hole = 3;
  const b = G.bang(s, "G");
  ok("bang: a constant up, laws kept, wells and dark matter gone", b && s.consts.G === 1 && s.laws.thin && s.wells.hole === 0 && s.darkUni.isZero() && s.shep[0] && !s.ended);
  for (const k of ["c", "h", "L", "O", "a"]) { s.darkUni = G.bangAt(s); G.bang(s, k); }
  ok("the end comes when every constant has been raised", s.ended === true && s.life.bangs === 6);
}
// sets, and what things say
{
  const sizes = (list, key) => { const o = {}; for (const x of list) o[x[key]] = (o[x[key]] || 0) + 1; return o; };
  const ms = sizes(D.MILESTONES, "set"), ups = sizes(D.UPGRADES, "set"), cons = sizes(D.NODES, "con");
  ok("every set has three to six in it", D.MILESTONE_SETS.every((x) => ms[x.id] >= 3 && ms[x.id] <= 6) && D.UPGRADE_SETS.every((x) => ups[x.id] >= 3 && ups[x.id] <= 6) && D.CONSTELLATIONS.every((x) => cons[x.id] >= 3 && cons[x.id] <= 6) && D.WELLS.length >= 3 && D.LAWS.length <= 6 && D.CONSTANTS.length <= 6);
  const all = D.UPGRADES.concat(D.NODES, D.WELLS, D.CONSTANTS), plain = /bought for you|ignites by itself|lit for you|stay|caught for you|kind of body|at once/;
  ok("every purchase says what it does, with a number where there is one", all.every((u) => /\d/.test(u.text) || plain.test(u.text)), all.filter((u) => !(/\d/.test(u.text) || plain.test(u.text))).map((u) => u.name).join(", "));
}
// numbers on the page
{
  F.set("names");
  let worst = 0, sd = 99; const rr = () => (sd = (sd * 1103515245 + 12345) % 2147483648) / 2147483648;
  for (let i = 0; i < 20000; i++) { const t = F.num(Big.exp10(rr() * 60 - 2)); const digits = t.replace(/e\d+$/, "").replace(/[^0-9]/g, "").length; if (digits > worst) worst = digits; }
  ok("a number is never shown with more than four digits", worst <= 4, worst);
  ok("numbers at the edges", F.num(999.6) === "1K" && F.num(999950) === "1M" && F.num(0) === "0" && F.num(1e15) === "1e15" && F.num("1e400") === "1e400" && F.int(1234) === "1,234");
  ok("times at the edges", F.time(59.6) === "1 min" && F.time(3599) === "1 h" && F.time(0.2) === "now" && F.time(Infinity) === "a long time");
}
console.log(bad ? `\n${bad} of ${n} FAILED` : `\nall ${n} passed`);
process.exit(bad ? 1 : 0);
