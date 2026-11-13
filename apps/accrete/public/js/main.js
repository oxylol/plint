// Accrete · the part that runs: keeps the game in this browser, moves time on, takes the hand on the screen,
// and turns what happens in the game into something to see and to hear.
(function () {
  "use strict";
  const G = window.Accrete, D = window.AccreteData, F = window.Fmt, Big = window.Big;
  const KEY = "accrete:save", BACKUP = "accrete:backup";
  const $ = (id) => document.getElementById(id);
  const phone = window.matchMedia("(max-width: 1000px)");
  const KINDS = window.AccreteSceneKinds, CLASS = window.AccreteSceneClass;
  const rgb = (c) => c.map((x) => Math.round(x * 255)).join(", ");
  const WARM = [1.0, 0.72, 0.42];

  // ── the saved game ──
  function load() {
    for (const key of [KEY, BACKUP]) {
      try { const raw = localStorage.getItem(key); if (!raw) continue; const o = JSON.parse(raw); if (o && o.v === 1) return G.revive(o); } catch (e) { /* try the older copy */ }
    }
    const s = G.fresh(Date.now());
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) s.set.motion = 0;
    return s;
  }
  let lastBackup = 0, dirty = false;
  function save() {
    try {
      const now = Date.now();
      // an older copy is kept beside the newest, so one bad write cannot lose the game
      if (now - lastBackup > 120000) { const prev = localStorage.getItem(KEY); if (prev) localStorage.setItem(BACKUP, prev); lastBackup = now; }
      localStorage.setItem(KEY, JSON.stringify(s));
      dirty = false;
    } catch (e) { /* storage full or shut: the game goes on, unsaved */ }
  }
  const encode = (text) => btoa(unescape(encodeURIComponent(text)));
  const decode = (text) => decodeURIComponent(escape(atob(text.replace(/\s+/g, ""))));

  let s = load();
  F.set(s.set.notation);
  const canvas = $("scene");
  const scene = new window.AccreteScene(canvas, { quality: phone.matches ? 1 : 2 });
  const sound = new window.AccreteSound();
  sound.set(s.set.volume, s.set.mute);
  scene.setMotion(s.set.motion);
  const fx = [];                                     // every piece of feedback given, for the checks
  let holding = false, busy = false, lastFloat = 0, cometShown = null, holdFx = 0;
  let heldFrom = null, heldFor = 0;                  // the hand is timed by the clock, not by frames

  // ── what the page can ask for ──
  const act = {
    state: () => s,
    buy(k) { sound.wake(); G.buy(s, k, s.set.buy); dirty = true; },
    buyMode(m) { s.set.buy = m; },
    upgrade(id) { sound.wake(); G.buyUpgrade(s, id); dirty = true; },
    shepherd(k) { sound.wake(); G.buyShepherd(s, k); dirty = true; },
    toggleShepherd(k) { if (s.shepOff[k]) delete s.shepOff[k]; else s.shepOff[k] = true; },
    node(id) { sound.wake(); G.buyNode(s, id); dirty = true; },
    well(id) { sound.wake(); G.buyWell(s, id); dirty = true; },
    watch(on, light) { if (on !== null) s.auto.on = on; if (light !== null && light >= 1) s.auto.light = Math.floor(light); },
    view(v) { if (busy) return; scene.setView(v); $("app").dataset.tab = v === "cloud" ? ($("app").dataset.tab === "bodies" ? "bodies" : "cloud") : v; },
    tab(id) { $("app").dataset.tab = id; if (!busy) scene.setView(id === "bodies" ? "cloud" : id); },
    mute() { sound.wake(); s.set.mute = !s.set.mute; sound.set(s.set.volume, s.set.mute); },
    setting(k, v) { s.set[k] = v; if (k === "notation") F.set(v); if (k === "volume") sound.set(s.set.volume, s.set.mute); if (k === "motion") scene.setMotion(v); },
    exportSave() { save(); return encode(JSON.stringify(s)); },
    importSave(text) {
      let o;
      try { o = JSON.parse(decode(text)); } catch (e) { return "That text is not a saved game."; }
      if (!o || o.v !== 1 || !Array.isArray(o.own)) return "That text is not a saved game.";
      s = G.revive(o); s.at = Date.now(); F.set(s.set.notation); sound.set(s.set.volume, s.set.mute); scene.setMotion(s.set.motion); scene.setView("cloud"); scene.setComet(null);
      save();
      return true;
    },
    wipe() { try { localStorage.removeItem(KEY); localStorage.removeItem(BACKUP); } catch (e) { /* nothing to remove */ } s = G.fresh(Date.now()); scene.setView("cloud"); scene.setComet(null); save(); },
    go() {
      sound.wake();
      const v = scene.view(), c = G.calc(s);
      if ((v === "cloud" || v === "sky") && G.igniteOpen(s)) { if (s.life.ignitions === 0) ui.cardReset("ignite", s, c, ignite); else ignite(); }
      else if (v === "galaxy" && G.collapseOpen(s)) ui.cardReset("collapse", s, c, collapse);
      else if (v === "universe" && G.bangOpen(s)) ui.cardReset("bang", s, c, bang);
    },
  };
  const ui = new window.AccreteUI(act);

  // ── the resets, as things that are seen ──
  const wait = (ms) => new Promise((ok) => setTimeout(ok, ms));
  async function ignite() {
    if (busy || !G.igniteOpen(s)) return;
    busy = true; dirty = true;
    const first = s.life.ignitions === 0, quick = s.set.motion < 0.5 || s.life.ignitions >= 5;
    scene.setView("cloud");
    sound.play(quick ? "node" : "ignite");
    const dur = quick ? 450 : 1700, t0 = performance.now();
    while (performance.now() - t0 < dur) { scene.fall(Math.pow((performance.now() - t0) / dur, 1.6)); await wait(16); }
    const cls = G.starClass(s.cloud.mass), col = CLASS[cls.id];
    const r = G.ignite(s);
    scene.fall(0);
    if (r) {
      scene.flash(quick ? 0.5 : 1.3); scene.shake(quick ? 4 : 16);
      scene.coreRing(col, { to: 900, speed: 900, width: 16, alpha: 1 }); scene.coreRing([1, 1, 1], { to: 600, speed: 520, width: 8, alpha: 0.6 });
      const p = scene.centre(); scene.spark(p.x, p.y, col, quick ? 40 : 140, 420, 4, 1.4);
      ui.toast(cls.name.charAt(0).toUpperCase() + cls.name.slice(1), "+" + F.int(r.light) + " light" + (r.fresh.length ? " · a new class: " + F.mult(D.IGNITE.classMult) + " to all bodies" : ""), rgb(col), true);
      note("ignite", 3);
    }
    await wait(quick ? 250 : 1300);
    busy = false;
    if (first) scene.setView("sky");
    save();
  }
  async function collapse(law) {
    if (busy || !G.collapseOpen(s)) return;
    busy = true;
    sound.play("collapse");
    const t0 = performance.now(), dur = s.set.motion < 0.5 ? 500 : 2200;
    while (performance.now() - t0 < dur) { scene.dim(Math.pow((performance.now() - t0) / dur, 2) * 0.95); await wait(16); }
    const r = G.collapse(s, law);
    scene.dim(0); scene.flash(0.9); scene.shake(12);
    const p = scene.centre(); scene.ring(p.x, p.y, [0.72, 0.62, 1.0], { to: 1100, speed: 800, width: 18, alpha: 1 });
    if (r) { ui.toast("A galaxy", "+" + F.int(r.dark) + " dark matter" + (r.kept ? " · law kept: " + D.LAWS.find((l) => l.id === r.kept).name : ""), "185, 162, 255", true); note("collapse", 3); }
    scene.setView("galaxy"); $("app").dataset.tab = "galaxy";
    await wait(600);
    busy = false; save();
  }
  async function bang(id) {
    if (busy || !G.bangOpen(s)) return;
    busy = true;
    sound.play("bang");
    const t0 = performance.now(), dur = s.set.motion < 0.5 ? 400 : 500;
    while (performance.now() - t0 < dur) { scene.dim((performance.now() - t0) / dur); await wait(16); }
    const r = G.bang(s, id);
    scene.dim(0); scene.flash(2.2); scene.shake(24);
    const p = scene.centre();
    for (let i = 0; i < 3; i++) scene.ring(p.x, p.y, [0.5 + 0.2 * i, 0.89, 1.0], { to: 1600, speed: 1300 - i * 300, width: 22 - i * 5, alpha: 1 });
    scene.spark(p.x, p.y, [0.6, 0.9, 1.0], 220, 700, 4, 1.8);
    if (r) { const k = D.CONSTANTS.find((x) => x.id === r.id); ui.toast("A new universe", k.name + " is now " + r.level, "127, 227, 255", true); note("bang", 3); }
    scene.setView("universe"); $("app").dataset.tab = "universe";
    await wait(1200);
    busy = false; save();
    if (r && r.end) { sound.play("end"); ui.cardEnd(s); }
  }

  // ── what happened in the game, shown and sounded: one of each, and only for what was done ──
  function note(type, tier) { fx.push({ type, tier, t: Date.now() }); if (fx.length > 600) fx.shift(); }
  function feedback(events) {
    for (const e of events) {
      switch (e.type) {
        case "buy": {
          const col = KINDS[e.k].col, own = s.own[e.k];
          scene.pop(e.k, own - e.n, true); ui.bought(e.k);
          const p = scene.bodyAt(e.k, own - 1);
          scene.spark(p.x, p.y, col, Math.min(22, 6 + e.n * 2), 70 + e.k * 12, 2.4 + e.k * 0.3, 0.6);
          if (e.bump) {
            scene.coreRing(col, { to: 520, speed: 560, width: 10, alpha: 0.9 }); scene.pulse(0.8);
            const c0 = scene.centre(); ui.float(c0.x, c0.y - c0.r - 26, F.mult(e.mult) + " " + D.KINDS[e.k].plural, rgb(col), true);
            sound.play("bump", { k: e.k });
          } else if (e.first) { scene.coreRing(col, { to: 420, speed: 420, width: 7, alpha: 0.7 }); sound.play("first", { k: e.k }); }
          else sound.play("buy", { k: e.k });
          note("buy", e.tier); break;
        }
        case "auto": if (e.k !== undefined) scene.pop(e.k, s.own[e.k] - e.n); break;
        case "pull": {
          scene.inward(WARM, 5); scene.pulse(0.55); sound.play("pull");
          const now = performance.now();
          if (now - lastFloat > 260) { lastFloat = now; const c0 = scene.centre(); ui.float(c0.x + (Math.random() - 0.5) * 50, c0.y - c0.r - 12, "+" + F.num(e.mass), rgb(WARM)); }
          note("pull", 1); break;
        }
        case "upgrade": {
          scene.pulse(1); scene.coreRing(WARM, { to: 300, speed: 380, width: 7, alpha: 0.6 });
          if (e.full) { sound.play("set"); ui.toast(D.UPGRADE_SETS.find((x) => x.id === e.set).name + ": whole", F.mult(D.SET_MULT) + " to all bodies", "255, 190, 120", true); }
          else sound.play("upgrade");
          note("upgrade", e.tier); break;
        }
        case "shepherd": sound.play("shepherd"); ui.toast("A shepherd of " + D.KINDS[e.k].plural, "It buys them for you from now on, through every reset.", rgb(KINDS[e.k].col)); note("shepherd", 2); break;
        case "reveal":
          if (e.what === "kind") { sound.play("reveal"); ui.toast("A new kind of body", "The " + D.KINDS[e.k].name + ": " + F.num(D.KINDS[e.k].cost) + " mass", rgb(KINDS[e.k].col)); note("reveal", 2); }
          else if (e.what === "ignite" || e.what === "collapse" || e.what === "bang") {
            sound.play("ready"); scene.coreRing([1, 0.9, 0.6], { to: 700, speed: 500, width: 12, alpha: 0.9 });
            ui.toast(e.what === "ignite" ? "The cloud can ignite" : e.what === "collapse" ? "The galaxy can collapse" : "A bang is in reach", e.what === "ignite" ? "When you choose. Waiting gives more light." : e.what === "collapse" ? "Look at the galaxy." : "Look at the universe.", "255, 220, 150", true);
            note("ready", 3);
          } else { sound.play("open"); note("open", 1); }
          break;
        case "milestone":
          if (e.full) { sound.play("set"); ui.toast("Set complete: " + D.MILESTONE_SETS.find((x) => x.id === e.set).name, e.name + " · " + F.mult(D.MS_MULT * D.MS_SET) + " to all bodies", "125, 240, 180", true); }
          else { sound.play("milestone"); ui.toast("Milestone: " + e.name, F.mult(D.MS_MULT) + " to all bodies", "125, 240, 180"); }
          note("milestone", e.tier); break;
        case "comet": scene.setComet(e.comet); cometShown = e.comet.id; sound.play("comet"); note("comet", 1); break;
        case "comet-catch": {
          const p = scene.cometAt() || scene.centre(), col = e.type === "surge" ? [0.55, 0.85, 1.0] : [1.0, 0.86, 0.52];
          scene.spark(p.x, p.y, col, 60, 260, 3.5, 1.0); scene.ring(p.x, p.y, col, { to: 260, speed: 420, width: 8 });
          ui.float(p.x, p.y, e.type === "surge" ? "surge " + F.mult(e.mult) + " · " + F.time(e.secs) : "+" + F.num(e.mass), rgb(col), true);
          sound.play("catch", { surge: e.type === "surge" });
          scene.setComet(null); note("catch", 2); break;
        }
        case "comet-gone": case "comet-auto": scene.setComet(null); break;
        case "node":
          if (e.by) break;
          if (e.full) { sound.play("set"); ui.toast(D.CONSTELLATIONS.find((x) => x.id === e.con).name + ": whole", F.mult(D.CON_MULT) + " to all bodies", "255, 231, 163", true); }
          else sound.play("node");
          scene.flash(0.12); note("node", e.tier); break;
        case "well": sound.play("well"); scene.flash(0.1); note("well", 2); break;
        default: break;                               // ignite, collapse, bang and end are shown by their own sequences
      }
    }
  }

  // ── what the picture is told each frame ──
  const TINT = {
    sky: [[0.10, 0.14, 0.34], [0.30, 0.20, 0.10], [0.06, 0.20, 0.30]],
    galaxy: [[0.22, 0.12, 0.40], [0.10, 0.12, 0.36], [0.36, 0.12, 0.26]],
    universe: [[0.06, 0.20, 0.34], [0.20, 0.10, 0.36], [0.05, 0.26, 0.28]],
  };
  const igLog = Math.log10(D.IGNITE.at);
  function model(c) {
    const v = scene.view();
    const own = s.own.map((n, k) => (c.exists[k] ? n : 0));
    const flow = c.kind.map((r) => (r.isZero() ? 0 : 14 + 60 * Math.max(0, r.mul(c.speed).log10() + 1)));
    const grown = Math.log10(1 + Math.min(1e300, s.cloud.mass.toNumber())) / igLog;
    const owned = []; for (let k = 7; k >= 0; k--) if (own[k] > 0) owned.push(KINDS[k].col);
    const tint = v === "cloud" ? [owned[0] || [0.3, 0.36, 0.5], owned[1] || owned[0] || [0.22, 0.28, 0.44], owned[2] || owned[0] || [0.2, 0.24, 0.4]] : TINT[v];
    const cons = {};
    for (const con of D.CONSTELLATIONS) cons[con.id] = G.sets.constellations[con.id].map((n) => (s.nodes[n.id] ? 2 : G.nodeOpen(s, n) ? 1 : 0));
    // the picture reaches as far as the outermost kind in view, and pulls back when a new one arrives
    let reach = 0.37; for (let k = 0; k < 8; k++) if (c.exists[k] && G.shown(s, c, k)) reach = Math.max(reach, KINDS[k].r[1] + 0.035);
    return {
      reach, own, flow, grown, heat: Math.pow(Math.max(0, Math.min(1, (grown - 0.3) / 0.7)), 2),
      ign: Math.min(1, grown), ignShow: G.layer(s) >= 1 || s.seen.k3 ? 1 : 0, surge: s.surge.left > 0,
      tint, tintAmt: v === "cloud" ? owned.length / 8 : 0.75,
      stars: s.stars, classes: s.classes, cons, hole: s.wells.hole, galaxiesEver: s.life.collapses + 1,
    };
  }

  // ── size ──
  // If the machine cannot keep up, the picture is drawn with fewer pixels rather than fewer frames.
  let sharp = 1, slowFrames = 0, frames = 0;
  function pace(dt) {
    frames++;
    if (frames < 20 || document.hidden) return;
    slowFrames = dt > 0.040 ? slowFrames + 1 : Math.max(0, slowFrames - 2);
    if (slowFrames > 45 && sharp > 0.45) { sharp = Math.max(0.45, sharp * 0.75); slowFrames = 0; resize(); }
  }
  function resize() {
    const w = window.innerWidth, h = window.innerHeight;
    const cs = getComputedStyle(document.documentElement);
    const pad = phone.matches ? { left: 0, right: 0, top: 60, bottom: Math.round(h * 0.54) } : { left: parseInt(cs.getPropertyValue("--left"), 10) + 32, right: parseInt(cs.getPropertyValue("--right"), 10) + 32, top: 0, bottom: 0 };
    scene.resize(w, h, (window.devicePixelRatio || 1) * sharp, pad);
  }
  window.addEventListener("resize", resize);
  resize();

  // ── the hand ──
  const stage = $("stage");
  function press(x, y) {
    sound.wake();
    if (busy || ui.cardOpen() || scene.view() !== "cloud") return;
    const at = s.comet.live ? scene.cometAt() : null;
    if (at && Math.hypot(at.x - x, at.y - y) < 56) { G.catchComet(s); dirty = true; return; }
    G.pull(s); holding = true; heldFrom = Date.now();
  }
  stage.addEventListener("pointerdown", (e) => { if (e.target.closest("button")) return; e.preventDefault(); try { stage.setPointerCapture(e.pointerId); } catch (err) { /* no capture, no matter */ } press(e.clientX, e.clientY); });
  const release = () => { if (heldFrom !== null) { heldFor += Date.now() - heldFrom; heldFrom = null; } holding = false; };
  stage.addEventListener("pointerup", release); stage.addEventListener("pointercancel", release); stage.addEventListener("lostpointercapture", release);
  window.addEventListener("blur", release);
  window.addEventListener("keydown", (e) => {
    if (e.target.matches("input, textarea") || ui.cardOpen()) { if (e.key === "Escape") ui.close(); return; }
    if (e.code === "Space") { e.preventDefault(); if (!e.repeat) { const c0 = scene.centre(); press(c0.x, c0.y); } }
    else if (e.key >= "1" && e.key <= "8") act.buy(Number(e.key) - 1);
    else if (e.key === "m" || e.key === "M") act.mute();
  });
  window.addEventListener("keyup", (e) => { if (e.code === "Space") release(); });
  document.addEventListener("contextmenu", (e) => { if (!e.target.matches("input, textarea")) e.preventDefault(); });

  // ── time ──
  // Back after a while: the time away is worked out in one go and told on a card.
  function comeBack(gap) {
    const sum = G.away(s, gap);
    G.drain();
    scene.setComet(null);
    if (gap >= 300) { ui.cardAway(sum, s, G.calc(s)); for (const e of sum.events || []) if (e.type === "milestone" || e.type === "reveal") note("away-" + e.type, 0); }
  }
  {
    const gap = (Date.now() - s.at) / 1000;
    if (s.life.secs + s.life.away > 0 && gap > 1) comeBack(gap);
    s.at = Date.now();
    if (s.comet.live) { scene.setComet(s.comet.live); cometShown = s.comet.live.id; }
  }
  let lastT = performance.now(), slowAt = 0, saveAt = performance.now();
  function frame(t) {
    const dt = Math.min(0.25, (t - lastT) / 1000); lastT = t;
    pace(dt);
    const wall = Date.now(), gap = (wall - s.at) / 1000;
    if (gap > 30) comeBack(gap);
    else if (gap > 0) {
      let held = heldFor; heldFor = 0;
      if (heldFrom !== null) { held += wall - heldFrom; heldFrom = wall; }
      G.tick(s, gap, { hold: busy || scene.view() !== "cloud" ? 0 : Math.min(1, held / 1000 / gap) });
    }
    s.at = wall;
    feedback(G.drain());
    const c = G.calc(s);
    if (holding && c.hand && c.auto < D.CORE.holdRate && !busy) { holdFx += dt; if (holdFx > 0.2) { holdFx = 0; scene.inward(WARM, 3); scene.pulse(0.35); sound.play("pull"); s.life.pulls++; } }
    if (s.comet.live && cometShown !== s.comet.live.id) { scene.setComet(s.comet.live); cometShown = s.comet.live.id; }
    ui.fast(s, c);
    if (t - slowAt > 180) { slowAt = t; ui.slow(s, c, scene); sound.hum(s.own.filter((n) => n > 0).length, Math.min(1, Math.log10(1 + Math.min(1e300, s.cloud.mass.toNumber())) / igLog), s.surge.left > 0); }
    else ui.cometTagMove(scene, s);
    scene.frame(dt, model(c));
    if (t - saveAt > (dirty ? 1500 : 10000)) { saveAt = t; save(); }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  document.addEventListener("visibilitychange", () => { if (document.hidden) { release(); save(); } });
  window.addEventListener("pagehide", save);

  // for the checks in test/page.js, and for looking under the bonnet
  window.__accrete = { get s() { return s; }, set s(v) { s = v; }, G, D, scene, sound, fx, save, act, ui };
})();
