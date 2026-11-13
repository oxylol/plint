// Accrete · the rules. A state is plain data; every function here reads or changes one. No page, no clock of
// its own: the caller says how many seconds went by. The browser, the tests and the bot all run this same file.
(function (root) {
  "use strict";
  const node = typeof module !== "undefined" && module.exports;
  const Big = node ? require("./big.js") : root.Big;
  const D = node ? require("./data.js") : root.AccreteData;
  const F = node ? require("./format.js") : root.Fmt;
  const ZERO = Big.ZERO, ONE = Big.ONE;
  const NK = D.KINDS.length;

  // ── What happened, for the page to show and sound. tier 1 a buy · 2 an unlock, bump, milestone · 3 the big ones ──
  let events = [];
  let quiet = null;                       // while away: a summary is kept and nothing is announced
  function emit(type, tier, data) {
    const e = Object.assign({ type, tier }, data || {});
    if (quiet) { (quiet.events = quiet.events || []).push(e); return; }
    events.push(e);
  }
  function drain() { const e = events; events = []; return e; }

  // ── A new game ──
  function fresh(now, seed) {
    return {
      v: 1, seed: (seed === undefined ? Math.floor(Math.random() * 4294967296) : seed) >>> 0, at: now || 0, born: now || 0,
      // one cloud
      mass: Big.from(D.CORE.start), cloud: { mass: Big.from(D.CORE.start), secs: 0 }, own: new Array(NK).fill(0), up: {},
      comet: { wait: null, live: null, n: 0 }, surge: { mult: 1, left: 0 },
      // kept through ignition
      shep: {}, shepOff: {}, light: ZERO, lightGal: ZERO, nodes: {}, classes: {}, stars: 0, auto: { on: false, light: 1 },
      // kept through collapse
      dark: ZERO, darkUni: ZERO, wells: { hole: 0, nursery: 0, gazer: 0, memory: 0 }, law: null, galaxies: 0,
      // kept through a bang
      laws: {}, consts: { G: 0, c: 0, h: 0, L: 0, O: 0, a: 0 }, echo: 0, ended: false,
      ms: {}, seen: {},
      life: { mass: Big.from(D.CORE.start), light: ZERO, dark: ZERO, bought: 0, pulls: 0, comets: 0, most: 0, kinds: 0, ignitions: 0, fastest: null,
        cons: 0, collapses: 0, bangs: 0, classes: {}, secs: 0, away: 0 },
      set: { notation: "names", motion: 1, volume: 0.7, mute: false, buy: 1 },
    };
  }
  const BIGS = [["mass"], ["cloud", "mass"], ["light"], ["lightGal"], ["dark"], ["darkUni"], ["life", "mass"], ["life", "light"], ["life", "dark"]];
  // A saved game back into a state. Anything missing is filled from a new game, so old saves keep working.
  function revive(raw) {
    const s = fresh(0, 1);
    const merge = (into, from) => {
      for (const k of Object.keys(from || {})) {
        if (from[k] && typeof from[k] === "object" && !Array.isArray(from[k]) && into[k] && typeof into[k] === "object" && !(into[k] instanceof Big)) merge(into[k], from[k]);
        else into[k] = from[k];
      }
    };
    merge(s, raw);
    for (const p of BIGS) { const o = p.length === 2 ? s[p[0]] : s; const k = p[p.length - 1]; o[k] = Big.from(o[k] === undefined || o[k] === null ? 0 : o[k]); }
    s.own = D.KINDS.map((_, i) => Math.max(0, Math.floor(Number((raw.own || [])[i]) || 0)));
    return s;
  }

  function rnd(s) {
    s.seed = (s.seed + 0x6D2B79F5) >>> 0;
    let t = s.seed;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  const between = (s, [lo, hi]) => lo + rnd(s) * (hi - lo);
  const count = (o) => { let n = 0; for (const k in o) if (o[k]) n++; return n; };

  // ── Bumps ──
  function bumps(n) {
    let b = 0;
    for (const t of D.BUMPS) if (n >= t) b++;
    const last = D.BUMPS[D.BUMPS.length - 1];
    if (n > last) b += Math.floor((n - last) / D.BUMP_STEP);
    return b;
  }
  function nextBump(n) {
    for (const t of D.BUMPS) if (n < t) return t;
    const last = D.BUMPS[D.BUMPS.length - 1];
    return last + (Math.floor((n - last) / D.BUMP_STEP) + 1) * D.BUMP_STEP;
  }
  function prevBump(n) {
    let p = 0;
    for (const t of D.BUMPS) if (n >= t) p = t;
    const last = D.BUMPS[D.BUMPS.length - 1];
    if (n > last) p = last + Math.floor((n - last) / D.BUMP_STEP) * D.BUMP_STEP;
    return p;
  }

  // ── Milestones ──
  function stat(s, name) {
    const L = s.life;
    switch (name) {
      case "mass": return L.mass;
      case "dark": return L.dark;
      case "shepherds": return count(s.shep);
      case "laws": return count(s.laws);
      case "consts": return count(s.consts);
      case "fastest": return L.fastest === null ? Infinity : L.fastest;
      default:
        if (name.startsWith("class")) return L.classes[name.slice(5)] ? 1 : 0;
        return L[name] || 0;
    }
  }
  function msFrac(s, m) {
    const v = stat(s, m.stat);
    if (m.lower) return v === Infinity ? 0 : Math.min(1, m.at / Math.max(v, 1e-9));
    if (v instanceof Big) { const t = Big.from(m.at); return v.gte(t) ? 1 : Math.max(0, Math.min(1, Math.log10(1 + v.toNumber()) / t.log10())); }
    return Math.max(0, Math.min(1, v / m.at));
  }
  function msDone(s, m) {
    const v = stat(s, m.stat);
    if (m.lower) return v <= m.at;
    return v instanceof Big ? v.gte(m.at) : v >= m.at;
  }
  const setOf = {};
  for (const m of D.MILESTONES) (setOf[m.set] = setOf[m.set] || []).push(m);
  function msCounts(s) {
    let got = 0, sets = 0;
    for (const id in setOf) { let n = 0; for (const m of setOf[id]) if (s.ms[m.id]) n++; got += n; if (n === setOf[id].length) sets++; }
    return { got, sets };
  }
  function checkMilestones(s) {
    for (const m of D.MILESTONES) {
      if (s.ms[m.id] || !msDone(s, m)) continue;
      s.ms[m.id] = true;
      const full = setOf[m.set].every((x) => s.ms[x.id]);
      emit("milestone", full ? 3 : 2, { id: m.id, name: m.name, set: m.set, full });
    }
  }
  // How far the game has unfolded: 0 the cloud · 1 the sky · 2 the galaxy · 3 the universe
  function layer(s) { return s.life.bangs || s.seen.bang ? 3 : s.life.collapses || s.seen.collapse ? 2 : s.life.ignitions ? 1 : 0; }

  const conOf = {};
  for (const n of D.NODES) (conOf[n.con] = conOf[n.con] || []).push(n);
  function consDone(s) { let n = 0; for (const id in conOf) if (conOf[id].every((x) => s.nodes[x.id])) n++; return n; }
  const upSetOf = {};
  for (const u of D.UPGRADES) (upSetOf[u.set] = upSetOf[u.set] || []).push(u);
  function upSetsDone(s) { let n = 0; for (const id in upSetOf) if (upSetOf[id].every((x) => s.up[x.id])) n++; return n; }

  // ── Everything that follows from a state: rates, prices, multipliers ──
  function calc(s) {
    const L = s.laws, law = s.law, N = s.nodes, U = s.up, K = s.consts, c = {};
    c.growth = Math.max(1.05, (law === "thin" ? 1.25 : L.thin ? 1.14 : D.GROWTH) - 0.01 * K.h);
    c.bump = law === "flat" ? 1 : (N.crucible ? 2.5 : D.BUMP_MULT) * (L.flat ? 1.5 : 1);
    c.link = N.reach ? 2 * D.LINK : D.LINK;
    const ms = msCounts(s);
    let m = Math.pow(D.MS_MULT, ms.got) * Math.pow(D.MS_SET, ms.sets);
    if (N.kindling) m *= 2;
    if (N.bellows) m *= 2;
    if (N.whiteheat) m *= 1 + 0.01 * Math.min(500, s.stars);
    m *= Math.pow(D.CON_MULT, consDone(s)) * Math.pow(D.IGNITE.classMult, count(s.classes)) * Math.pow(D.SET_MULT, upSetsDone(s));
    if (L.alone) m *= 10;
    let g = s.lightGal.mul(D.IGNITE.passive).add(1).mul(m).mul(s.darkUni.add(1).pow(D.COLLAPSE.mass));
    if (K.G) g = g.mul(Big.exp10(2 * K.G));
    c.calm = g;                                         // without a surge
    c.surge = s.surge.left > 0 ? s.surge.mult : 1;
    g = g.mul(c.surge);
    c.global = g;
    c.exists = D.KINDS.map((k, i) => !(law === "alone" && i % 2 === 1) && (!k.node || !!N[k.node]));
    c.kind = [];
    let sum = ZERO;
    const bump = Big.from(c.bump);
    for (let i = 0; i < NK; i++) {
      const n = s.own[i];
      if (!n || !c.exists[i]) { c.kind.push(ZERO); continue; }
      let r = Big.from(D.KINDS[i].rate * n).mul(bump.pow(bumps(n)));
      if (i === 0 && U.dust) r = r.mul(2);
      if (i < NK - 1 && U["link" + i] && c.exists[i + 1]) r = r.mul(1 + c.link * s.own[i + 1]);
      r = r.mul(g);
      c.kind.push(r); sum = sum.add(r);
    }
    c.rate = sum.add(g.mul(D.CORE.trickle));
    c.speed = Math.pow(1.3, s.wells.hole) * Math.pow(2, K.c);
    const grip = U.grip3 ? 0.10 : U.grip2 ? 0.05 : U.grip1 ? 0.02 : 0;
    c.pull = c.rate.mul(grip).add(g.mul(D.CORE.pull));
    c.auto = U.spin2 ? 5 : U.spin1 ? 1 : 0;
    c.hand = law === "dark" ? 0 : 1;
    c.perSec = c.rate.add(c.pull.mul(c.auto)).mul(c.speed);
    c.comet = L.dark ? 3 : 1;
    c.cometWait = (U.tail2 ? 0.7 : 1) * (N.widenet ? 0.7 : 1);
    c.share = N.openhand ? 0.25 : D.SHEPHERD.share;
    let lm = s.darkUni.add(1).pow(D.COLLAPSE.light);
    let l = 1;
    if (N.lens) l *= 2;
    if (N.mirror) l *= 2;
    if (N.prism) l *= Math.pow(1.5, count(s.classes));
    l *= Math.pow(2, s.wells.nursery);
    if (L.dim) l *= 5;
    if (law === "dim") l *= 0.1;
    lm = lm.mul(l);
    if (K.L) lm = lm.mul(Big.exp10(K.L));
    c.light = lm;
    c.dark = 1 + K.O;
    c.nodeCost = (law === "bare" ? 5 : 1) * (L.bare ? 0.5 : 1);
    return c;
  }

  // ── Bodies: price, how many, buying ──
  const afford = (have, price) => have.gte(price.mul(1 - 1e-12));
  function cost(s, c, k, n) { return Big.geomSum(D.KINDS[k].cost, c.growth, s.own[k], n === undefined ? 1 : n); }
  function maxBuy(s, c, k) { return Big.geomMax(s.mass.mul(1 + 1e-12), D.KINDS[k].cost, c.growth, s.own[k]); }
  // mode: a number, "max" (at least 1, as many as the mass pays for) or "bump" (up to the next bump)
  function amount(s, c, k, mode) {
    if (mode === "max") return Math.max(1, maxBuy(s, c, k));
    if (mode === "bump") return nextBump(s.own[k]) - s.own[k];
    return mode;
  }
  function shown(s, c, k) { return k === 0 || (c.exists[k] && !!s.seen["k" + k]); }
  function buy(s, k, mode, o) {
    const c = (o && o.c) || calc(s);
    if (!c.exists[k] || !shown(s, c, k)) return 0;
    const n = amount(s, c, k, mode === undefined ? 1 : mode);
    const price = cost(s, c, k, n);
    if (n < 1 || !afford(s.mass, price)) return 0;
    const before = s.own[k], b0 = bumps(before);
    s.mass = s.mass.sub(price).max(0);
    s.own[k] += n;
    const by = o && o.by;
    if (!by) s.life.bought += n;
    noteOwned(s);
    const first = before === 0 && !s.seen["o" + k];
    if (first) s.seen["o" + k] = true;
    const bumped = bumps(s.own[k]) > b0 && s.law !== "flat";
    if (by) emit("auto", 0, { k, n });
    else emit("buy", first || bumped ? 2 : 1, { k, n, first, bump: bumped ? prevBump(s.own[k]) : 0, mult: bumped ? c.bump : 0 });
    return n;
  }
  function noteOwned(s) {
    let kinds = 0, most = 0;
    for (const n of s.own) { if (n > 0) kinds++; if (n > most) most = n; }
    if (kinds > s.life.kinds) s.life.kinds = kinds;
    if (most > s.life.most) s.life.most = most;
  }

  // ── Upgrades and shepherds ──
  function needMet(s, c, need) {
    if (!need) return true;
    if (need.mass !== undefined && s.cloud.mass.lt(need.mass)) return false;
    if (need.own && s.own[need.own[0]] < need.own[1]) return false;
    if (need.comets !== undefined && s.life.comets < need.comets) return false;
    return true;
  }
  function needText(need) {
    const p = [];
    if (need.own) p.push("own " + need.own[1] + " " + D.KINDS[need.own[0]].plural);
    if (need.comets !== undefined) p.push(need.comets === 1 ? "catch a comet" : "catch " + need.comets + " comets");
    if (need.mass !== undefined) p.push("gather " + F.num(need.mass) + " mass");
    return p;
  }
  function upgradeOpen(s, c, u) {
    if (s.up[u.id]) return false;
    if (u.link !== undefined && !(c.exists[u.link] && c.exists[u.link + 1])) return false;
    if (u.set === "comet" && s.law === "dark") return false;
    return needMet(s, c, u.need);
  }
  function buyUpgrade(s, id, o) {
    const u = D.UPGRADES.find((x) => x.id === id), c = (o && o.c) || calc(s);
    if (!u || !upgradeOpen(s, c, u) || !afford(s.mass, Big.from(u.cost))) return false;
    s.mass = s.mass.sub(u.cost).max(0);
    s.up[id] = true;
    const full = upSetOf[u.set].every((x) => s.up[x.id]);
    emit(o && o.by ? "auto" : "upgrade", o && o.by ? 0 : full ? 3 : 2, { id, name: u.name, set: u.set, full });
    return true;
  }
  function shepherdCost(k) { return Big.from(D.KINDS[k].cost).mul(D.SHEPHERD.costFactor); }
  function shepherdOpen(s, c, k) { return !s.shep[k] && c.exists[k] && (s.own[k] >= D.SHEPHERD.own || !!s.seen["s" + k]); }
  function buyShepherd(s, k) {
    const c = calc(s);
    if (!shepherdOpen(s, c, k) || !afford(s.mass, shepherdCost(k))) return false;
    s.mass = s.mass.sub(shepherdCost(k)).max(0);
    s.shep[k] = true;
    emit("shepherd", 2, { k });
    return true;
  }
  // What the shepherds and the steward do in one step. Each buys only what costs no more than its share of the mass.
  function automate(s, c) {
    let did = false;
    for (let k = NK - 1; k >= 0; k--) {
      if (!s.shep[k] || s.shepOff[k] || !c.exists[k] || !shown(s, c, k)) continue;
      const purse = s.mass.mul(c.share);
      let n = Big.geomMax(purse, D.KINDS[k].cost, c.growth, s.own[k]);
      if (n < 1) continue;
      if (!s.nodes.herd) n = 1;
      if (buy(s, k, n, { c, by: "shepherd" })) { did = true; if (quiet) quiet.bought = (quiet.bought || 0) + n; }
    }
    if (s.wells.gazer) {
      for (const n of D.NODES) if (nodeOpen(s, n) && s.light.gte(nodeCost(c, n)) && buyNode(s, n.id, { by: "gazer" })) did = true;
    }
    if (s.nodes.steward) {
      for (const u of D.UPGRADES) {
        if (s.up[u.id] || !upgradeOpen(s, c, u) || s.mass.mul(c.share).lt(u.cost)) continue;
        if (buyUpgrade(s, u.id, { c, by: "steward" })) did = true;
      }
    }
    return did;
  }

  // ── The comet ──
  function cometBand(s, c, type) {
    if (type === "surge") { const k = s.up.tail1 ? 2 : 1; return [D.COMET.surge.secs[0] * k, D.COMET.surge.secs[1] * k]; }
    const k = (s.up.tail3 ? 2 : 1) * c.comet;
    return [D.COMET.windfall[0] * k, D.COMET.windfall[1] * k];
  }
  function spawnComet(s, c) {
    const type = rnd(s) < D.COMET.surgeChance ? "surge" : "windfall";
    const [lo, hi] = cometBand(s, c, type);
    s.comet.n++;
    s.comet.live = { type, ttl: D.COMET.ttl, lo, hi, id: s.comet.n, path: rnd(s), mult: type === "surge" ? D.COMET.surge.mult * c.comet : 0 };
    emit("comet", 1, { comet: s.comet.live });
  }
  function catchComet(s, o) {
    const live = s.comet.live;
    if (!live) return null;
    const c = calc(s), share = (o && o.share) || 1;
    const secs = live.lo + rnd(s) * (live.hi - live.lo);
    let got;
    if (live.type === "surge") {
      s.surge = { mult: live.mult, left: secs * share };
      got = { type: "surge", secs: secs * share, mult: live.mult };
    } else {
      const mass = c.rate.div(c.surge).mul(c.speed).mul(secs * share);
      gain(s, mass);
      got = { type: "windfall", secs: secs * share, mass };
    }
    s.comet.live = null;
    s.comet.wait = between(s, D.COMET.wait) * c.cometWait;
    s.life.comets++;
    emit(o && o.by ? "comet-auto" : "comet-catch", o && o.by ? 0 : 2, got);
    return got;
  }

  function gain(s, m) { s.mass = s.mass.add(m); s.cloud.mass = s.cloud.mass.add(m); s.life.mass = s.life.mass.add(m); }
  // One pull of the hand on the core.
  function pull(s) {
    const c = calc(s);
    if (!c.hand) return ZERO;
    const m = c.pull.mul(c.speed);
    gain(s, m);
    s.life.pulls++;
    emit("pull", 1, { mass: m });
    return m;
  }

  // ── Time ──
  // dt seconds with the page open (o.hold: how much of that time the hand was held on the core, 0 to 1).
  // For time away use away().
  function tick(s, dt, o) {
    if (!(dt > 0)) return;
    const awayNow = !!(o && o.away);
    let left = dt, c = null;
    for (let i = 0; i < 3 && left > 0; i++) {
      c = calc(s);
      const part = s.surge.left > 0 ? Math.min(left, s.surge.left) : left;
      const held = o && o.hold && c.hand ? Math.min(1, Number(o.hold)) : 0;
      const pulls = c.auto + held * Math.max(0, D.CORE.holdRate - c.auto);
      gain(s, c.rate.add(c.pull.mul(pulls)).mul(c.speed * part));
      if (s.surge.left > 0) { s.surge.left -= part; if (s.surge.left <= 1e-9) s.surge = { mult: 1, left: 0 }; }
      left -= part;
    }
    s.cloud.secs += dt;
    if (awayNow) s.life.away += dt; else s.life.secs += dt;
    c = calc(s);
    if (!awayNow && s.law !== "dark") {
      if (s.comet.wait === null) s.comet.wait = between(s, D.COMET.first);
      if (s.comet.live) {
        s.comet.live.ttl -= dt;
        if (s.comet.live.ttl <= 0) {
          if (s.nodes.driftnet) catchComet(s, { by: "net", share: s.nodes.fullnet ? 1 : 0.5 });
          else { emit("comet-gone", 0, {}); s.comet.live = null; s.comet.wait = between(s, D.COMET.wait) * c.cometWait; }
        }
      } else {
        s.comet.wait -= dt;
        if (s.comet.wait <= 0) spawnComet(s, c);
      }
    }
    reveal(s, c);
    if (count(s.shep) || s.nodes.steward || s.wells.gazer) { if (automate(s, c)) c = calc(s); }
    if (s.nodes.watch && s.auto.on && igniteOpen(s) && lightGain(s, c).gte(Math.max(1, s.auto.light))) {
      const r = ignite(s, { by: "watch" });
      if (quiet && r) { quiet.stars = (quiet.stars || 0) + 1; quiet.light = (quiet.light || ZERO).add(r.light); }
    }
    checkMilestones(s);
  }
  // Things that come into view as the cloud grows.
  function reveal(s, c) {
    for (let k = 1; k < NK; k++) {
      if (s.seen["k" + k] || !c.exists[k]) continue;
      if (D.KINDS[k].node || s.cloud.mass.gte(D.KINDS[k].cost * D.REVEAL)) { s.seen["k" + k] = true; emit("reveal", 2, { what: "kind", k }); }
    }
    for (const u of D.UPGRADES) {
      if (s.seen["u" + u.id] || !upgradeOpen(s, c, u)) continue;
      s.seen["u" + u.id] = true; emit("reveal", 1, { what: "upgrade", id: u.id });
    }
    for (let k = 0; k < NK; k++) {
      if (s.seen["s" + k] || !c.exists[k] || s.own[k] < D.SHEPHERD.own) continue;
      s.seen["s" + k] = true; emit("reveal", 1, { what: "shepherd", k });
    }
    if (!s.seen.ignite && igniteOpen(s)) { s.seen.ignite = true; emit("reveal", 3, { what: "ignite" }); }
    if (!s.seen.collapse && collapseOpen(s)) { s.seen.collapse = true; emit("reveal", 3, { what: "collapse" }); }
    if (!s.seen.bang && bangOpen(s)) { s.seen.bang = true; emit("reveal", 3, { what: "bang" }); }
  }
  // Time with the page closed. Earns at the full rate, with no cap; what the shepherds would have done is done
  // in steps. Returns what there is to tell on the way back in.
  function away(s, secs) {
    const sum = { secs, mass: ZERO, bought: 0, stars: 0, light: ZERO, events: [] };
    if (!(secs > 0)) return sum;
    const before = s.life.mass;
    // in steps, so that what changes on the way (a milestone reached, a shepherd buying) counts from when it happens
    const steps = Math.max(1, Math.min(4000, Math.ceil(secs / 2)));
    quiet = sum;
    try { for (let i = 0; i < steps; i++) tick(s, secs / steps, { away: true }); } finally { quiet = null; }
    sum.mass = s.life.mass.sub(before);
    return sum;
  }

  // ── Reset 1 · ignition ──
  function igniteOpen(s) { return s.cloud.mass.gte(D.IGNITE.at); }
  function lightGain(s, c) {
    if (!igniteOpen(s)) return ZERO;
    return s.cloud.mass.div(D.IGNITE.at).pow(1 / D.IGNITE.root).mul((c || calc(s)).light).floor().max(1);
  }
  function starClass(mass) {
    let got = null;
    for (const k of D.IGNITE.classes) if (Big.from(mass).gte(Big.from(D.IGNITE.at).mul(k.at))) got = k;
    return got;
  }
  function nextClass(s) { return D.IGNITE.classes.find((k) => !s.classes[k.id]) || null; }
  function newCloud(s, keepUp) {
    s.mass = ZERO; s.cloud = { mass: ZERO, secs: 0 }; s.own = new Array(NK).fill(0);
    if (!keepUp) s.up = {};
    else if (keepUp === "core") { const u = {}; for (const x of D.UPGRADES) if (x.set === "core" && s.up[x.id]) u[x.id] = true; s.up = u; }
    s.comet.live = null; s.surge = { mult: 1, left: 0 };
    const c = calc(s);
    if (s.nodes.seed && c.exists[0]) s.own[0] = 25;
    if (s.nodes.sapling) for (let k = 0; k < 4; k++) if (c.exists[k]) { s.own[k] = Math.max(s.own[k], 10); s.seen["k" + k] = true; }
    noteOwned(s);
  }
  function ignite(s, o) {
    if (!igniteOpen(s)) return null;
    const c = calc(s), light = lightGain(s, c), cls = starClass(s.cloud.mass), secs = s.cloud.secs, mass = s.cloud.mass;
    s.light = s.light.add(light); s.lightGal = s.lightGal.add(light); s.life.light = s.life.light.add(light);
    s.stars++; s.life.ignitions++;
    if (s.life.fastest === null || secs < s.life.fastest) s.life.fastest = secs;
    // every class up to this one is now in the sky
    const fresh = [];
    for (const k of D.IGNITE.classes) {
      if (k.at > cls.at) break;
      if (!s.classes[k.id]) { s.classes[k.id] = true; fresh.push(k.id); }
      s.life.classes[k.id] = true;
    }
    newCloud(s, s.nodes.oldwood ? "all" : s.nodes.roots ? "core" : null);
    const r = { light, cls: cls.id, fresh, secs, mass, by: o && o.by };
    emit("ignite", o && o.by ? 0 : 3, r);
    return r;
  }
  function nodeCost(c, n) { return Math.max(1, Math.ceil(n.cost * c.nodeCost)); }
  function nodeOpen(s, n) { return !s.nodes[n.id] && (!n.after || !!s.nodes[n.after]); }
  function buyNode(s, id, o) {
    const n = D.NODES.find((x) => x.id === id), c = calc(s);
    if (!n || !nodeOpen(s, n) || !afford(s.light, Big.from(nodeCost(c, n)))) return false;
    s.light = s.light.sub(nodeCost(c, n)).max(0);
    s.nodes[id] = true;
    const full = conOf[n.con].every((x) => s.nodes[x.id]);
    const done = consDone(s);
    if (done > s.life.cons) s.life.cons = done;
    emit("node", o && o.by ? 0 : full ? 3 : 2, { id, name: n.name, con: n.con, full, by: o && o.by });
    reveal(s, calc(s));
    return true;
  }

  // ── Reset 2 · collapse ──
  function collapseOpen(s) { return s.lightGal.gte(D.COLLAPSE.at); }
  function darkGain(s, c) {
    if (!collapseOpen(s)) return ZERO;
    return Big.from(Math.floor(s.lightGal.div(D.COLLAPSE.at).log10() + 1 + 1e-9) * (c || calc(s)).dark);
  }
  function newGalaxy(s) {
    s.light = ZERO; s.lightGal = ZERO; s.nodes = {}; s.classes = {}; s.stars = 0;
    for (let i = 0; i < Math.min(s.wells.memory, D.MEMORY.length); i++) for (const n of conOf[D.MEMORY[i]]) s.nodes[n.id] = true;
    const done = consDone(s);
    if (done > s.life.cons) s.life.cons = done;
    newCloud(s, null);
    reveal(s, calc(s));
  }
  // law: the id of a law to put the next galaxy under, or nothing
  function collapse(s, law) {
    if (!collapseOpen(s)) return null;
    const dark = darkGain(s), kept = s.law;
    s.dark = s.dark.add(dark); s.darkUni = s.darkUni.add(dark); s.life.dark = s.life.dark.add(dark);
    s.galaxies++; s.life.collapses++;
    if (kept) s.laws[kept] = true;
    s.law = law && !s.laws[law] && D.LAWS.some((x) => x.id === law) ? law : null;
    newGalaxy(s);
    const r = { dark, kept, law: s.law };
    emit("collapse", 3, r);
    return r;
  }
  function wellCost(s, w) { return Big.from(Math.round(w.cost * Math.pow(w.step, s.wells[w.id]))); }
  function buyWell(s, id) {
    const w = D.WELLS.find((x) => x.id === id);
    if (!w || s.wells[id] >= w.max || !afford(s.dark, wellCost(s, w))) return false;
    s.dark = s.dark.sub(wellCost(s, w)).max(0);
    s.wells[id]++;
    emit("well", 2, { id, level: s.wells[id] });
    return true;
  }

  // ── Reset 3 · bang ──
  function bangAt(s) { return Big.from(D.BANG.at + D.BANG.add * s.life.bangs); }
  function bangOpen(s) { return s.darkUni.gte(bangAt(s)); }
  function bang(s, id) {
    if (!bangOpen(s) || !(id in s.consts)) return null;
    s.consts[id]++; s.echo++; s.life.bangs++;
    s.dark = ZERO; s.darkUni = ZERO; s.law = null; s.galaxies = 0;
    for (const w of D.WELLS) s.wells[w.id] = Math.min(s.wells[w.id], s.consts.a);
    newGalaxy(s);
    const end = !s.ended && D.CONSTANTS.every((k) => s.consts[k.id] > 0);
    if (end) s.ended = true;
    const r = { id, level: s.consts[id], end };
    emit("bang", 3, r);
    if (end) emit("end", 3, {});
    return r;
  }

  // ── What the page says about where you are ──
  const logFrac = (v, t) => { v = Big.from(v); t = Big.from(t); return v.gte(t) ? 1 : Math.max(0, Math.min(1, Math.log10(1 + v.toNumber()) / Math.log10(1 + t.toNumber()))); };
  const bigFrac = (v, t) => { v = Big.from(v); t = Big.from(t); if (v.gte(t)) return 1; if (t.e < 300) return logFrac(v, t); return Math.max(0, Math.min(1, Math.max(0, v.log10()) / t.log10())); };
  const secsFor = (need, have, per) => { const d = Big.from(need).sub(have); return d.lte(0) ? 0 : per.isZero() ? Infinity : d.div(per).toNumber(); };
  // Goals with their distance, nearest first. Every one is a number with a target; none can go backwards.
  function goals(s, c) {
    c = c || calc(s);
    const g = [];
    const L = layer(s);
    if (!s.life.bought && !s.own[0]) { const p = cost(s, c, 0, 1); g.push({ id: "first", kind: "buy", k: 0, label: "Your first Mote", frac: Math.min(1, s.mass.div(p).toNumber()), eta: secsFor(p, s.mass, c.perSec), cur: s.mass, at: p, unit: "mass" }); }
    for (let k = 1; k < NK; k++) {
      if (s.seen["k" + k] || !c.exists[k]) continue;
      const t = D.KINDS[k].cost * D.REVEAL;
      g.push({ id: "kind" + k, kind: "reveal", label: "A new kind of body", frac: logFrac(s.cloud.mass, t), eta: secsFor(t, s.cloud.mass, c.perSec), cur: s.cloud.mass, at: t, unit: "mass gathered" });
      break;
    }
    for (let k = 0; k < NK; k++) {
      if (!s.own[k] || !c.exists[k] || s.law === "flat") continue;
      const nb = nextBump(s.own[k]), pb = prevBump(s.own[k]);
      g.push({ id: "bump" + k, kind: "bump", k, label: nb + " " + D.KINDS[k].plural, reward: "x" + c.bump, frac: s.own[k] / nb, eta: secsFor(cost(s, c, k, nb - s.own[k]), s.mass, c.perSec), cur: s.own[k], at: nb, from: pb });
    }
    if (!igniteOpen(s) && (L >= 1 || s.seen.k3)) g.push({ id: "ignite", kind: "layer", label: "Ignition", frac: bigFrac(s.cloud.mass, D.IGNITE.at), eta: secsFor(D.IGNITE.at, s.cloud.mass, c.perSec), cur: s.cloud.mass, at: D.IGNITE.at, unit: "mass gathered" });
    if (L >= 1) {
      const nc = nextClass(s);
      if (nc) { const t = Big.from(D.IGNITE.at).mul(nc.at); if (s.cloud.mass.lt(t)) g.push({ id: "class" + nc.id, kind: "class", label: nc.name.charAt(0).toUpperCase() + nc.name.slice(1), reward: "x" + D.IGNITE.classMult, frac: bigFrac(s.cloud.mass, t), eta: secsFor(t, s.cloud.mass, c.perSec), cur: s.cloud.mass, at: t, unit: "mass gathered" }); }
      if (!collapseOpen(s) && (L >= 2 || s.lightGal.gte(D.COLLAPSE.at / 50))) g.push({ id: "collapse", kind: "layer", label: "Collapse", frac: bigFrac(s.lightGal, D.COLLAPSE.at), eta: Infinity, cur: s.lightGal, at: D.COLLAPSE.at, unit: "light made" });
    }
    if (L >= 2 && !bangOpen(s) && s.darkUni.gte(bangAt(s).div(30))) g.push({ id: "bang", kind: "layer", label: "Bang", frac: bigFrac(s.darkUni, bangAt(s)), eta: Infinity, cur: s.darkUni, at: bangAt(s), unit: "dark matter made" });
    for (const set of D.MILESTONE_SETS) {
      if ((set.layer || 0) > L) continue;
      const m = setOf[set.id].find((x) => !s.ms[x.id]);
      if (!m || (m.layer || 0) > L) continue;
      const v = stat(s, m.stat);
      let eta = Infinity;
      if (m.stat === "mass") eta = secsFor(m.at, v, c.perSec);
      g.push({ id: "ms" + m.id, kind: "milestone", label: m.name, set: set.name, frac: msFrac(s, m), eta, cur: v, at: m.at, lower: !!m.lower });
    }
    // a goal not yet begun is not shown as a bar: no bar starts at zero
    const live = g.filter((x) => x.frac > 0);
    live.sort((a, b) => b.frac - a.frac);
    return live;
  }
  // The next things that are still shut, each with what opens it. Never more than three.
  function outlines(s, c) {
    c = c || calc(s);
    const out = [], L = layer(s);
    for (let k = 1; k < NK; k++) {
      if (s.seen["k" + k]) continue;
      if (!c.exists[k] && s.law === "alone" && k % 2 === 1) continue;
      const kd = D.KINDS[k];
      if (kd.node) { if (L >= 1) out.push({ what: "kind", k, cond: "light the star “" + D.NODES.find((n) => n.id === kd.node).name + "” in the sky" }); }
      else out.push({ what: "kind", k, cond: "gather " + F.num(kd.cost * D.REVEAL) + " mass", at: kd.cost * D.REVEAL });
      break;
    }
    if (L === 0 && !s.seen.ignite) out.push({ what: "ignite", cond: "gather " + F.num(D.IGNITE.at) + " mass", at: D.IGNITE.at });
    else if (L === 1 && !s.seen.collapse) out.push({ what: "collapse", cond: "make " + F.num(D.COLLAPSE.at) + " light in this galaxy", at: D.COLLAPSE.at });
    else if (L === 2 && !s.seen.bang) out.push({ what: "bang", cond: "make " + F.num(bangAt(s)) + " dark matter in this universe", at: bangAt(s) });
    const shut = D.UPGRADES.filter((u) => !s.up[u.id] && !upgradeOpen(s, c, u) && !(u.link !== undefined && !(c.exists[u.link] && c.exists[u.link + 1])) && !(u.set === "comet" && s.law === "dark"));
    shut.sort((a, b) => a.cost - b.cost);
    if (shut.length) out.push({ what: "upgrade", id: shut[0].id, cond: needText(shut[0].need).join(", "), need: shut[0].need });
    for (let k = 0; k < NK && out.length < 3; k++) {
      if (s.shep[k] || s.seen["s" + k] || !c.exists[k]) continue;
      out.push({ what: "shepherd", k, cond: "own " + D.SHEPHERD.own + " " + D.KINDS[k].plural });
      break;
    }
    if (out.length < 3 && shut.length > 1) out.push({ what: "upgrade", id: shut[1].id, cond: needText(shut[1].need).join(", "), need: shut[1].need });
    // further up: a star behind another star, a law not yet kept, a constant not yet raised
    if (L >= 1 && out.length < 3) {
      const n = D.NODES.filter((x) => !s.nodes[x.id] && x.after && !s.nodes[x.after]).sort((a, b) => a.cost - b.cost)[0];
      if (n) out.push({ what: "node", id: n.id, cond: "light “" + D.NODES.find((x) => x.id === n.after).name + "” first" });
    }
    if (L >= 2 && out.length < 3) {
      const l = D.LAWS.find((x) => !s.laws[x.id] && x.id !== s.law);
      if (l) out.push({ what: "law", id: l.id, cond: "collapse a galaxy under it" });
    }
    if (L >= 3 && out.length < 3 && !s.ended) {
      const k = D.CONSTANTS.find((x) => !s.consts[x.id]);
      if (k) out.push({ what: "constant", id: k.id, cond: "raise it at a bang" });
    }
    return out.slice(0, 3);
  }
  // Everything that can be bought with mass right now, with its price. For the page, the bot and the lull.
  function offers(s, c) {
    c = c || calc(s);
    const o = [];
    for (let k = 0; k < NK; k++) if (shown(s, c, k)) o.push({ what: "kind", k, cost: cost(s, c, k, 1) });
    for (const u of D.UPGRADES) if (upgradeOpen(s, c, u)) o.push({ what: "upgrade", id: u.id, cost: Big.from(u.cost) });
    for (let k = 0; k < NK; k++) if (shepherdOpen(s, c, k)) o.push({ what: "shepherd", k, cost: shepherdCost(k) });
    return o;
  }
  // When nothing can be bought for a while: how long, and what comes then.
  function lull(s, c) {
    c = c || calc(s);
    let best = null;
    for (const o of offers(s, c)) { const t = secsFor(o.cost, s.mass, c.perSec); if (!best || t < best.secs) best = { secs: t, offer: o }; }
    return best;
  }

  const G = {
    fresh, revive, calc, tick, away, pull, buy, cost, maxBuy, amount, shown, afford, bumps, nextBump, prevBump,
    upgradeOpen, buyUpgrade, shepherdCost, shepherdOpen, buyShepherd, needText,
    catchComet, cometBand, igniteOpen, lightGain, starClass, nextClass, ignite, nodeCost, nodeOpen, buyNode,
    collapseOpen, darkGain, collapse, wellCost, buyWell, bangOpen, bangAt, bang,
    goals, outlines, offers, lull, layer, stat, msFrac, msCounts, consDone, upSetsDone, count, drain, rnd,
    sets: { milestones: setOf, constellations: conOf, upgrades: upSetOf },
  };
  if (node) module.exports = G; else root.Accrete = G;
})(typeof self !== "undefined" ? self : this);
