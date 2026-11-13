// Accrete · a bot that plays the game, to measure its pace and to hold it to the checks in the spec.
//
//   node test/bot.js                  the whole game on a plausible week: a first sitting, then short visits
//   node test/bot.js --first          only the first sitting (75 min), with the minute-by-minute log
//   node test/bot.js --days 30 --idle no hand on the core, no comets caught: the laziest player
//
// It buys whatever pays itself back soonest (price ÷ gain, plus the wait until it can be paid for), resets when
// a reset adds half again or more, and takes the laws one by one. A person will play differently; the bot is a
// floor, not a forecast. Exit code 1 if a check fails.
"use strict";
const path = require("path");
const G = require(path.join(__dirname, "../public/js/game.js"));
const D = require(path.join(__dirname, "../public/js/data.js"));
const Big = require(path.join(__dirname, "../public/js/big.js"));

const args = process.argv.slice(2);
const flag = (n) => args.includes("--" + n);
const opt = (n, d) => { const i = args.indexOf("--" + n); return i < 0 ? d : Number(args[i + 1]); };
const DAYS = opt("days", 21), FIRST = flag("first"), IDLE = flag("idle"), SEED = opt("seed", 7), LOG = flag("log") || FIRST;
const TRACE = opt("trace", 0);
const SIT = opt("sit", 75) * 60, VISIT = opt("visit", 4) * 60, EVERY = opt("every", 60) * 60;

const fmt = (b) => { b = Big.from(b); return b.e < 6 ? String(Math.round(b.toNumber() * 100) / 100) : b.m.toFixed(2) + "e" + b.e; };
const clock = (t) => { const d = Math.floor(t / 86400), h = Math.floor((t % 86400) / 3600), m = Math.floor((t % 3600) / 60), s = Math.floor(t % 60); return (d ? d + "d " : "") + (h || d ? String(h).padStart(2, "0") + ":" : "") + String(m).padStart(2, "0") + ":" + String(s).padStart(2, "0"); };

// TUNE='D.IGNITE.at = 1e10' tries other numbers without touching data.js
if (process.env.TUNE) new Function("D", "Big", process.env.TUNE)(D, Big);
const s = G.fresh(0, SEED);
let T = 0;                                   // seconds since the game began, open or away
const log = [];                              // [T, text]
const note = (text) => { log.push([T, text]); if (LOG) console.log(clock(T).padStart(12) + "  " + text); };
const M = {                                  // what is measured
  buys: [], firstBuy: null, kinds: {}, ups: {}, sheps: {}, comets: [], ms: [], ignites: [], collapses: [], bangs: [], laws: [], end: null, mechanics: [],
  ticks: 0, half: 0, zeroBars: 0, outl0: 0, outlMany: 0, choice: 0, choices: 0, feedback: 0, top: 0, lost: 0, lulls: 0,
  barsSeen: {}, regain: null, byTen: [],
};
let best1 = null;                            // the most mass gathered in a cloud before the first ignition, and when

function eat() {
  for (const e of G.drain()) {
    if (e.tier > 0) { M.feedback++; if (e.tier === 3) M.top++; }
    if (e.type === "buy") {
      M.buys.push(T); if (M.firstBuy === null) M.firstBuy = T;
      if (e.first && !(e.k in M.kinds)) { M.kinds[e.k] = T; note("first " + D.KINDS[e.k].name); }
    } else if (e.type === "upgrade") { if (!(e.id in M.ups)) { M.ups[e.id] = T; note("upgrade " + e.name + (e.full ? "  (set complete)" : "")); } M.buys.push(T); }
    else if (e.type === "shepherd") { if (!(e.k in M.sheps)) { M.sheps[e.k] = T; note("shepherd for " + D.KINDS[e.k].plural); if (Object.keys(M.sheps).length === 1) M.mechanics.push([T, "shepherds"]); } M.buys.push(T); }
    else if (e.type === "comet") { if (!M.comets.length) { note("first comet"); M.mechanics.push([T, "comets"]); } M.comets.push(T); }
    else if (e.type === "milestone") { M.ms.push(T); note("milestone " + e.name + (e.full ? "  (set complete)" : "")); }
    else if (e.type === "ignite") { M.ignites.push({ T, secs: e.secs, light: e.light, cls: e.cls }); if (M.ignites.length <= 12 || e.fresh.length) note("IGNITE #" + M.ignites.length + " after " + clock(e.secs) + " → " + fmt(e.light) + " light, class " + e.cls + (e.fresh.length ? "  new: " + e.fresh.join("") : "")); if (M.ignites.length === 1) M.mechanics.push([T, "the sky"]); }
    else if (e.type === "node") { if (!e.by) note("star " + e.name + (e.full ? "  (constellation complete)" : "")); }
    else if (e.type === "collapse") { M.collapses.push({ T, dark: e.dark }); note("COLLAPSE #" + M.collapses.length + " → " + fmt(e.dark) + " dark matter" + (e.kept ? ", kept law " + e.kept : "") + (e.law ? ", next under " + e.law : "")); if (e.kept) M.laws.push(T); if (M.collapses.length === 1) M.mechanics.push([T, "the galaxy"]); }
    else if (e.type === "well") note("well " + e.id + " " + e.level);
    else if (e.type === "bang") { M.bangs.push(T); note("BANG #" + M.bangs.length + " → " + e.id + " " + e.level); if (M.bangs.length === 1) M.mechanics.push([T, "the universe"]); }
    else if (e.type === "end") { M.end = T; note("THE END"); }
  }
}

// What one more of kind k adds per second, without recomputing everything.
function gainOf(c, k) {
  const n = s.own[k], bump = Big.from(c.bump);
  const lk = (i) => (i < 7 && s.up["link" + i] && c.exists[i + 1] ? 1 + c.link * s.own[i + 1] : 1);
  const r = (m) => (m ? Big.from(D.KINDS[k].rate * m * (k === 0 && s.up.dust ? 2 : 1)).mul(bump.pow(G.bumps(m))).mul(lk(k)).mul(c.global) : Big.ZERO);
  let d = r(n + 1).sub(r(n));
  if (k > 0 && s.up["link" + (k - 1)] && c.exists[k - 1]) d = d.add(c.kind[k - 1].mul(c.link / (1 + c.link * n)));
  return d.mul(c.speed);
}
function choose(c) {
  let best = null;
  const per = c.perSec;
  const offers = G.offers(s, c);
  const scored = [];
  for (const o of offers) {
    let d;
    if (o.what === "kind") d = gainOf(c, o.k);
    else if (o.what === "upgrade") { s.up[o.id] = true; d = G.calc(s).perSec.sub(per); delete s.up[o.id]; if (d.lte(0)) d = per.mul(IDLE ? 0.02 : 0.2); }
    else d = per.mul(0.05);                                   // a shepherd: worth it as soon as it is cheap
    if (d.lte(0)) continue;
    const pay = o.cost.div(d).toNumber();
    const wait = o.cost.lte(s.mass) ? 0 : o.cost.sub(s.mass).div(per).toNumber();
    scored.push({ o, pay, wait, score: pay + wait });
    if (!best || pay + wait < best.score) best = { o, pay, wait, score: pay + wait };
  }
  return { best, scored };
}
function act(o) {
  if (o.what === "kind") return G.buy(s, o.k, 1) > 0;
  if (o.what === "upgrade") return G.buyUpgrade(s, o.id);
  return G.buyShepherd(s, o.k);
}

const gal = { T0: 0, last: null };
const LAW_ORDER = ["bare", "dim", "flat", "thin", "dark", "alone"];
const CONST_ORDER = ["G", "c", "L", "O", "h", "a"];
function upper(c) {
  // the sky: cheapest open star first
  for (;;) {
    const open = D.NODES.filter((n) => G.nodeOpen(s, n)).sort((a, b) => a.cost - b.cost)[0];
    if (!open || !G.buyNode(s, open.id)) break;
    if (open.id === "watch") { s.auto.on = true; }
  }
  for (;;) { const w = D.WELLS.filter((x) => s.wells[x.id] < x.max).sort((a, b) => G.wellCost(s, a).cmp(G.wellCost(s, b)))[0]; if (!w || !G.buyWell(s, w.id)) break; }
  if (G.bangOpen(s)) { G.bang(s, CONST_ORDER.find((k) => !s.consts[k]) || "G"); return; }
  if (G.collapseOpen(s)) {
    // collapse when waiting no longer pays: the last stretch added dark matter slower than the galaxy has on average
    const g = G.darkGain(s, c).toNumber();
    if (gal.last === null) gal.last = { T, g };
    const span = T - gal.last.T;
    if (!s.darkUni.gt(0) || s.law || (span >= 600 && (g - gal.last.g) / span < g / (T - gal.T0))) {
      const next = s.life.collapses >= 2 && s.life.collapses % 2 === 0 ? LAW_ORDER.find((l) => !s.laws[l] && l !== s.law) : null;
      G.collapse(s, next || null);
      gal.T0 = T; gal.last = null;
      return;
    }
    if (span >= 600) gal.last = { T, g };
  }
  if (G.igniteOpen(s)) {
    const g = G.lightGain(s, c);
    const nc = G.nextClass(s);
    const near = nc && G.goals(s, c).find((x) => x.id === "class" + nc.id);
    const soon = near && near.eta < 90;                        // a new class of star is a minute away: wait for it
    if (!soon && (s.lightGal.isZero() || g.gte(s.lightGal.mul(0.5)) || s.cloud.secs > 1800)) {
      if (s.nodes.watch) s.auto.light = Math.max(1, g.mul(0.5).toNumber());
      G.ignite(s);
    }
  }
}

// One second with the page open.
let reaction = 0;
function second() {
  G.tick(s, 1, { hold: IDLE ? 0 : 1 });
  T += 1;
  if (s.comet.live && !IDLE) { if (++reaction >= 3) { G.catchComet(s); reaction = 0; } } else reaction = 0;
  let c = G.calc(s);
  for (let i = 0; i < 40; i++) {
    const { best, scored } = choose(c);
    if (!best) break;
    if (best.wait > 0) {
      // a person does not sit on their hands: with 15 s or more to go, something that can be paid for now is bought,
      // if it is not much worse
      const now = scored.filter((x) => x.wait === 0 && x.pay <= best.score * 6).sort((a, b) => a.pay - b.pay)[0];
      if (now && best.wait > 15) { if (!act(now.o)) break; c = G.calc(s); continue; }
      if (best.wait > 90) M.lulls++;
      break;
    }
    // was there a real choice? two things to buy whose payback is within a factor of three
    const can = scored.filter((x) => x.wait === 0).sort((a, b) => a.pay - b.pay);
    const real = can.length >= 2 && can[1].pay <= can[0].pay * 3;
    M.choices++; if (real) M.choice++;
    const bk = Math.min(8, Math.floor(T / 600)); M.byTen[bk] = M.byTen[bk] || [0, 0]; M.byTen[bk][1]++; if (real) M.byTen[bk][0]++;
    if (!act(best.o)) break;
    c = G.calc(s);
  }
  upper(c);
  eat();
  if (TRACE && T % TRACE === 0) console.log(clock(T).padStart(12) + "  cloud " + fmt(s.cloud.mass) + " · " + fmt(c.perSec) + "/s · light " + fmt(s.lightGal) + " · dark " + fmt(s.darkUni) + " · stars " + s.stars + " · own " + s.own.join(","));
  // the checks that hold at every moment
  M.ticks++;
  const goals = G.goals(s, c);
  if (goals.some((g) => g.frac >= 0.5)) M.half++; else if (LOG && M.ticks - M.half < 6) note("no goal half done: " + goals.slice(0, 3).map((g) => g.id + " " + g.frac.toFixed(2)).join(", "));
  for (const g of goals) if (!(g.id in M.barsSeen)) { M.barsSeen[g.id] = g.frac; if (g.frac <= 0) { M.zeroBars++; if (LOG) note("bar at zero: " + g.id); } }
  const out = G.outlines(s, c).length;
  if (out === 0 && !s.ended) { M.outl0++; if (LOG && M.outl0 < 4) note("no locked outline"); }
  if (out > 3) M.outlMany++;
  if (M.regain === null && M.ignites.length === 1 && G.igniteOpen(s)) M.regain = T - M.ignites[0].T;
  if (M.regain === null && M.ignites.length > 1) M.regain = M.ignites[1].T - M.ignites[0].T;
}
function sit(secs) { const until = T + secs; while (T < until && !(M.end !== null && T > M.end + 600)) second(); }
function leave(secs) {
  const before = { mass: s.life.mass, light: s.life.light, dark: s.life.dark, own: s.own.slice(), stars: s.life.ignitions };
  const hadReset = s.nodes.watch && s.auto.on;
  G.away(s, secs); T += secs;
  if (s.life.mass.lt(before.mass) || s.life.light.lt(before.light) || s.life.dark.lt(before.dark)) M.lost++;
  if (!hadReset && s.own.some((n, i) => n < before.own[i])) M.lost++;
  eat();
}

// ── The week ──
sit(SIT);
if (!FIRST) {
  const end = DAYS * 86400;
  while (T < end && !(M.end !== null && T > M.end + 3600)) {
    const hour = (T % 86400) / 3600;
    if (hour >= 16) leave(86400 - (T % 86400));              // the night: eight hours
    else { leave(EVERY - VISIT); sit(VISIT); }
  }
}

// ── The report ──
const min = (t) => (t / 60).toFixed(1) + " min";
console.log("\n── pace ──");
console.log("first buy                 " + (M.firstBuy === null ? "never" : M.firstBuy + " s"));
const b10 = M.buys.filter((t) => t <= 600);
let wait10 = 0; for (let i = 1; i < b10.length; i++) { wait10 = Math.max(wait10, b10[i] - b10[i - 1]); if (LOG && b10[i] - b10[i - 1] > 30) console.log("  wait of " + (b10[i] - b10[i - 1]) + " s before " + clock(b10[i])); }
console.log("first 10 min              " + b10.length + " buys, longest wait " + wait10 + " s");
const kt = Object.keys(M.kinds).map((k) => [Number(k), M.kinds[k]]).sort((a, b) => a[1] - b[1]);
console.log("kinds                     " + kt.map(([k, t]) => D.KINDS[k].name + " " + min(t)).join(" · "));
console.log("first comet               " + (M.comets.length ? min(M.comets[0]) : "none"));
console.log("shepherds                 " + (Object.keys(M.sheps).map((k) => D.KINDS[k].name + " " + min(M.sheps[k])).join(" · ") || "none"));
console.log("upgrades                  " + Object.keys(M.ups).length + " of " + D.UPGRADES.length + (M.ups.spin1 !== undefined ? " · core pulls itself at " + min(M.ups.spin1) : ""));
const left = D.MILESTONES.filter((m) => !s.ms[m.id]);       // from the state: those reached while away are not in the log
console.log("milestones                " + (D.MILESTONES.length - left.length) + " of " + D.MILESTONES.length + (left.length ? " · not reached: " + left.map((m) => m.name).join(", ") : ""));
console.log("first ignition            " + (M.ignites.length ? min(M.ignites[0].T) : "not reached") + (M.regain !== null ? " · old best regained after " + min(M.regain) : ""));
if (M.ignites.length > 1) console.log("ignitions                 " + M.ignites.length + " · the first ten took " + M.ignites.slice(0, 10).map((x) => clock(x.secs)).join(", "));
console.log("first collapse            " + (M.collapses.length ? clock(M.collapses[0].T) : "not reached") + (M.collapses.length ? " · " + M.collapses.length + " in all, laws kept " + G.count(s.laws) : ""));
console.log("first bang                " + (M.bangs.length ? clock(M.bangs[0]) : "not reached") + (M.bangs.length ? " · " + M.bangs.length + " in all" : ""));
console.log("the end                   " + (M.end !== null ? clock(M.end) : "not reached in " + DAYS + " days"));
console.log("where it stands           mass " + fmt(s.life.mass) + " · light " + fmt(s.life.light) + " · dark " + fmt(s.life.dark) + " · stars " + s.life.ignitions + " · open " + clock(s.life.secs) + " · away " + clock(s.life.away));

console.log("\n── checks ──");
let bad = 0;
const check = (name, ok, detail) => { if (!ok) bad++; console.log((ok ? "PASS  " : "FAIL  ") + name + (detail ? "  — " + detail : "")); };
check("first buy within 15 s", M.firstBuy !== null && M.firstBuy <= 15, M.firstBuy + " s");
check("first 10 min: 20 buys or more, never over 30 s between them", b10.length >= 20 && wait10 <= 30, b10.length + " buys, " + wait10 + " s");
const k30 = kt.filter(([, t]) => t <= 1800);
let gapMax = 0, gapMin = Infinity; for (let i = 1; i < k30.length; i++) { const g = k30[i][1] - k30[i - 1][1]; gapMax = Math.max(gapMax, g); if (i >= 2) gapMin = Math.min(gapMin, g); }
const lastKind = k30.length ? k30[k30.length - 1][1] : 0;
if (!IDLE) check("the six kinds of the first cloud come 3 to 7 min apart, the last by minute 25", k30.length >= 6 && gapMax <= 420 && gapMin >= 150 && lastKind <= 1500, k30.length + " kinds, gaps " + (gapMin / 60).toFixed(1) + " to " + (gapMax / 60).toFixed(1) + " min, last at " + (lastKind / 60).toFixed(1));
if (!IDLE) check("first comet within 15 min", M.comets.length && M.comets[0] <= 900, M.comets.length ? min(M.comets[0]) : "none");
if (!IDLE) check("first ignition reachable in about an hour (45 to 75 min)", M.ignites.length && M.ignites[0].T >= 2700 && M.ignites[0].T <= 4500, M.ignites.length ? min(M.ignites[0].T) : "not reached");
check("something automated before the first ignition", M.ups.spin1 !== undefined && Object.values(M.sheps).some((t) => !M.ignites.length || t < M.ignites[0].T), "core " + (M.ups.spin1 !== undefined ? min(M.ups.spin1) : "no") + ", shepherds " + Object.values(M.sheps).filter((t) => !M.ignites.length || t < M.ignites[0].T).length);
if (M.ignites.length) check("after the first reset the old best comes back in half the time or less", M.regain !== null && M.regain <= M.ignites[0].T / 2, M.regain !== null ? min(M.regain) + " against " + min(M.ignites[0].T) : "not regained");
check("a goal is at least half done, always", M.half === M.ticks, ((100 * M.half) / M.ticks).toFixed(2) + "% of the time");
check("no bar starts at zero", M.zeroBars === 0, M.zeroBars + " of " + Object.keys(M.barsSeen).length + " bars");
check("one to three locked outlines, always", M.outl0 === 0 && M.outlMany === 0, M.outl0 + " seconds with none");
// Not "always": a new kind is plainly the best buy for its first few. Measured and said, with a floor of 60%.
check("two things worth buying within a factor of three, at 60% of buys or more", M.choices && M.choice / M.choices >= 0.6, M.choices ? ((100 * M.choice) / M.choices).toFixed(1) + "% of " + M.choices + " buys" : "no buys");
if (LOG) console.log("      real choice by ten minutes: " + M.byTen.map((b, i) => (b ? Math.round((100 * b[0]) / b[1]) + "%" : "-")).join(" "));
check("nothing is ever lower after time away", M.lost === 0, M.lost + " times");
check("the loud events are under 5% of all feedback", M.feedback && M.top / M.feedback < 0.05, M.top + " of " + M.feedback);
// in the first hour things may come 3 minutes apart (the easy start); after it, never two within 10 minutes
let close = 0; for (let i = 1; i < M.mechanics.length; i++) if (M.mechanics[i][0] - M.mechanics[i - 1][0] < (M.mechanics[i][0] < 3600 ? 180 : 600)) close++;
check("no two new mechanics on top of each other (3 min apart in the first hour, 10 after)", close === 0, M.mechanics.map(([t, n]) => n + " " + min(t)).join(" · "));
if (!FIRST) check("the end is reached", M.end !== null, M.end !== null ? clock(M.end) : "not in " + DAYS + " days");
process.exit(bad ? 1 : 0);
