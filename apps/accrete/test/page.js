// Accrete · browser checks of the page, against a SCRATCH copy of the server (its own port; the game keeps itself
// in the browser, so nothing here can touch a game somebody is playing).
//
//   PORT=5941 node server.js &
//   docker run --rm --network host -v /opt/plint/apps/accrete/test:/t:ro -v /tmp/plint-sketch/node_modules:/nm:ro \
//     -e NODE_PATH=/nm -v /tmp/accrete-shots:/shots mcr.microsoft.com/playwright:v1.48.2-jammy node /t/page.js
//
// The browser in that image draws WebGL in software, so frames are slow; the game counts time by the clock, not by
// frames, and the waits below are generous. Pictures of each layer go to /shots (pc-*.png, phone-*.png).
const { chromium, devices } = require("playwright");
const B = process.env.BASE || "http://127.0.0.1:5941";
const SHOTS = process.env.SHOTS || "/shots";
const sleep = (ms) => new Promise((ok) => setTimeout(ok, ms));
const results = [];
const check = (name, ok, detail) => { results.push(ok); console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail !== undefined ? "  — " + detail : ""}`); };
const until = async (f, ms = 15000) => { const t0 = Date.now(); for (;;) { try { if (await f()) return true; } catch (e) { /* not yet */ } if (Date.now() - t0 > ms) return false; await sleep(150); } };

// states far into the game, written into the running page
const READY = () => { const a = window.__accrete, s = a.s, Big = window.Big; s.own = [142, 108, 89, 70, 62, 48, 0, 0]; for (let k = 1; k < 6; k++) { s.seen["k" + k] = true; s.seen["o" + k] = true; } s.mass = Big.from(4.4e9); s.cloud.mass = Big.from(7e11); s.life.mass = Big.from(7e11); s.life.bought = 519; s.life.pulls = 300; s.life.comets = 11; for (const u of a.D.UPGRADES) if (!/link5|link6/.test(u.id)) s.up[u.id] = true; for (let k = 0; k < 6; k++) { s.shep[k] = true; s.seen["s" + k] = true; } s.life.most = 142; s.life.kinds = 6; s.comet.wait = 9999; };
const GALAXY = () => { const a = window.__accrete, s = a.s, Big = window.Big; s.life.ignitions = 900; s.stars = 900; s.lightGal = Big.from(3.2e7); s.light = Big.from(100); s.life.light = Big.from(3.2e7); for (const n of a.D.NODES) s.nodes[n.id] = true; s.classes = { M: 1, K: 1, G: 1, F: 1 }; s.seen.ignite = true; s.seen.collapse = true; s.life.collapses = 1; s.dark = Big.from(3); s.darkUni = Big.from(3); s.life.dark = Big.from(3); };
const UNIVERSE = () => { const a = window.__accrete, s = a.s, Big = window.Big; s.darkUni = Big.from(400); s.dark = Big.from(40); s.life.dark = Big.from(400); s.life.collapses = 40; s.seen.bang = true; };

(async () => {
  const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--autoplay-policy=no-user-gesture-required"] });
  const errors = [];
  const open = async (ctx) => { const p = await ctx.newPage(); p.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); }); p.on("pageerror", (e) => errors.push(e.message)); await p.goto(B + "/", { waitUntil: "load" }); await until(() => p.evaluate(() => !!window.__accrete)); await sleep(600); return p; };
  const S = (p, f, arg) => p.evaluate(f, arg);

  // ── 1 · a new game on a PC: the first minute ──
  let ctx = await browser.newContext({ viewport: { width: 1920, height: 1080 } });
  let p = await open(ctx);
  check("the page draws: WebGL is up, nothing in the console", (await S(p, () => !window.__accrete.scene.dead)) && errors.length === 0, errors.join(" | ") || "clean");
  check("a new game starts with a little dust and a goal already full", await S(p, () => { const a = window.__accrete; return a.s.mass.toNumber() >= 30 && a.G.goals(a.s).some((g) => g.frac >= 0.5) && a.G.goals(a.s).every((g) => g.frac > 0); }));
  check("a locked outline is shown, with what opens it", (await p.locator("#bodies .outline:visible .cond").innerText()).includes("480"));
  await p.screenshot({ path: SHOTS + "/pc-0-start.png" });

  // nothing happens by itself: three quiet seconds give no feedback at all
  // (a thing coming into reach is said once, quietly; that is a goal met, not a celebration of nothing)
  const fx0 = await S(p, () => window.__accrete.fx.length);
  await sleep(3000);
  const idle = await S(p, (n) => window.__accrete.fx.slice(n).map((e) => e.type + e.tier), fx0);
  check("nothing celebrates by itself: three idle seconds give at most the quiet note of something opening", idle.every((t) => t === "open1"), idle.join(",") || "nothing at all");

  // one buy: one thing seen, one thing heard
  const before = await S(p, () => ({ fx: window.__accrete.fx.length, snd: window.__accrete.sound.log.length }));
  await p.locator("#bodies .body:visible .buy").first().click();
  await until(() => S(p, (n) => window.__accrete.fx.slice(n).some((e) => e.type === "milestone"), before.fx), 8000);
  await sleep(300);
  const after = await S(p, () => { const a = window.__accrete; return { own: a.s.own[0], fx: a.fx.slice(), snd: a.sound.log.slice(), ready: a.sound.ready() }; });
  const fxNew = after.fx.slice(before.fx), sndNew = after.snd.slice(before.snd);
  check("the first Mote is bought with one press", after.own === 1);
  check("a buy gives one thing to see and one to hear (and the milestone it earned, one each)", fxNew.filter((e) => e.type === "buy").length === 1 && sndNew.filter((n) => n === "buy" || n === "first" || n === "bump").length === 1 && fxNew.filter((e) => e.type === "milestone").length === sndNew.filter((n) => n === "milestone" || n === "set").length, fxNew.map((e) => e.type).join(",") + " · " + sndNew.join(","));
  check("the milestone it earned is told", await until(() => p.locator(".toast").first().isVisible(), 3000));
  check("sound is made in the page (an audio context is running)", after.ready && (await S(p, () => window.__accrete.sound.context().state)) === "running");

  // the hand: holding pulls mass in
  // (the page moves the game on once a frame, and frames are slow here: read well after the hand is lifted)
  const m0 = await S(p, () => ({ m: window.__accrete.s.life.mass.toNumber(), t: Date.now(), per: window.__accrete.G.calc(window.__accrete.s).perSec.toNumber() }));
  const box = await p.locator("#stage").boundingBox();
  await p.mouse.move(box.x + box.width / 2, box.y + box.height * 0.3);
  await p.mouse.down(); await sleep(2000); await p.mouse.up();
  await sleep(2500);
  const m1 = await S(p, () => ({ m: window.__accrete.s.life.mass.toNumber(), t: window.__accrete.s.at, pulls: window.__accrete.s.life.pulls }));
  const alone = m0.per * (m1.t - m0.t) / 1000, byHand = m1.m - m0.m - alone;
  check("holding on the picture pulls mass in: 5 pulls a second on top of what comes by itself", byHand > 9 && byHand < 13 && m1.pulls > 0, byHand.toFixed(1) + " mass by hand in 2 s (10 expected, and 1 for the press)");

  // what is next: bars with distances
  check("Next shows goals with a bar and what is left", (await p.locator("#goals .goal:visible").count()) >= 2 && /\d/.test(await p.locator("#goals .goal:visible .of").first().innerText()));

  // ── 2 · it keeps itself ──
  await S(p, () => window.__accrete.save());
  const kept = await S(p, () => ({ own: window.__accrete.s.own[0], bought: window.__accrete.s.life.bought, mass: window.__accrete.s.life.mass.toNumber() }));
  await p.reload({ waitUntil: "load" }); await until(() => p.evaluate(() => !!window.__accrete)); await sleep(500);
  const back = await S(p, () => ({ own: window.__accrete.s.own[0], bought: window.__accrete.s.life.bought, mass: window.__accrete.s.life.mass.toNumber() }));
  check("reload keeps the game", back.own === kept.own && back.bought === kept.bought && back.mass >= kept.mass, JSON.stringify(back));

  // export, wipe, import
  const text = await S(p, () => window.__accrete.act.exportSave());
  await S(p, () => window.__accrete.act.wipe());
  const wiped = await S(p, () => window.__accrete.s.life.bought);
  const imp = await S(p, (t) => window.__accrete.act.importSave(t), text);
  const again = await S(p, () => ({ own: window.__accrete.s.own[0], bought: window.__accrete.s.life.bought }));
  check("export, wipe, import gives the same game", wiped === 0 && imp === true && again.own === kept.own && again.bought === kept.bought);
  check("text that is not a game is refused", (await S(p, () => window.__accrete.act.importSave("hello"))) !== true);

  // mute stays muted
  await p.locator("#tools button").first().click();
  await S(p, () => window.__accrete.save());
  await p.reload({ waitUntil: "load" }); await until(() => p.evaluate(() => !!window.__accrete)); await sleep(400);
  check("mute silences it and stays after a reload", await S(p, () => window.__accrete.s.set.mute === true && window.__accrete.sound.muted() === true));
  await p.locator("#tools button").first().click();

  // ── 3 · away: it earns at the full rate, and says so first ──
  await S(p, () => { const a = window.__accrete, s = a.s; s.own[0] = 40; s.at = Date.now() - 2 * 3600 * 1000; a.save(); });
  // what two hours with the page open would give: the same game, played forward a second at a time, comets apart
  const rate = await S(p, () => { const a = window.__accrete, b = a.G.revive(JSON.parse(JSON.stringify(a.s))); b.comet.wait = 1e9; for (let i = 0; i < 7200; i++) a.G.tick(b, 1, {}); a.G.drain(); return { open: b.life.mass.toNumber() - a.s.life.mass.toNumber(), mass: a.s.life.mass.toNumber() }; });
  await p.reload({ waitUntil: "load" }); await until(() => p.evaluate(() => !!window.__accrete)); await sleep(700);
  const aw = await S(p, () => ({ mass: window.__accrete.s.life.mass.toNumber(), away: window.__accrete.s.life.away }));
  const gained = aw.mass - rate.mass, expect = rate.open;
  check("two hours away earn what two hours open would", Math.abs(gained - expect) / expect < 0.01, gained.toFixed(0) + " against " + expect.toFixed(0));
  check("the card on return opens with what was gained", (await p.locator("#card h1").innerText()).includes("away") && /\+/.test(await p.locator("#card .gain").innerText()) && /where you left off/i.test(await p.locator("#card").innerText()));
  await p.screenshot({ path: SHOTS + "/pc-away.png" });
  await p.locator("#card .btn.hot").click();

  // ── 4 · the evidence, the settings ──
  await p.locator("#tools button").nth(1).click();
  const why = await S(p, () => ({ n: document.querySelectorAll("#card .why").length, links: document.querySelectorAll("#card .why a[href^='http']").length, proven: document.querySelectorAll("#card .why .g.proven").length, list: window.AccreteWhy.list.length }));
  check("the WHY page lists every hook with a grade and a source", why.n === why.list && why.links === why.n && why.proven === 2, why.n + " hooks, " + why.proven + " proven");
  await p.screenshot({ path: SHOTS + "/pc-why.png" });
  await p.keyboard.press("Escape");
  await p.locator("#tools button").nth(2).click();
  // set the mass, press a way of writing, wait for the number on the page to read that way
  const reads = async (label, v, re) => { await p.locator("#card .seg button", { hasText: label }).click(); await S(p, (x) => { const a = window.__accrete; a.s.own = a.s.own.map(() => 0); a.s.mass = window.Big.from(x); }, v); let t = ""; await until(async () => { t = await p.locator("#mass").innerText(); return re.test(t); }, 8000); return t; };
  const sci = await reads("1.2e12", 2.5e12, /^2\.5\d?e12$/), long = await reads("3Qa", 3.38e28, /^33\.\dOc$/), names = await reads("3e15", 2.5e13, /^25(\.\d)?T$/);
  check("the way numbers are written can be changed", /e12$/.test(sci) && /Oc$/.test(long) && /T$/.test(names), sci + " / " + long + " / " + names);
  await p.keyboard.press("Escape");
  await ctx.close();

  // ── 5 · the layers ──
  ctx = await browser.newContext({ viewport: { width: 1920, height: 1080 } });
  p = await open(ctx);
  await S(p, READY); await sleep(1200);
  await p.screenshot({ path: SHOTS + "/pc-1-cloud.png" });
  check("when the cloud can ignite, the button is there and says what it gives", (await p.locator("#go").isVisible()) && /light/.test(await p.locator("#go").innerText()));
  await p.locator("#go").click();
  check("the first ignition asks first, and says what goes and what stays", /what goes[\s\S]*what stays/i.test(await p.locator("#card").innerText()));
  await p.locator("#card .btn.hot").click();
  check("ignition: a star, light, the sky", await until(() => S(p, () => { const a = window.__accrete; return a.s.life.ignitions === 1 && a.s.light.toNumber() >= 1 && a.scene.view() === "sky"; }), 12000));
  const ig = await S(p, () => { const a = window.__accrete; return { sheps: a.G.count(a.s.shep), ms: a.G.msCounts(a.s).got, own: a.s.own.reduce((x, y) => x + y, 0), cls: Object.keys(a.s.classes).join("") }; });
  check("what stays, stays: shepherds and milestones; the cloud starts again", ig.sheps === 6 && ig.ms >= 3 && ig.own < 20 && ig.cls === "M", JSON.stringify(ig));
  await sleep(1500);
  await p.locator('#pane-sky .cardrow:has(.t:text-is("Kindling"))').click(); await sleep(300);
  check("a star in a constellation is lit with light", await S(p, () => !!window.__accrete.s.nodes.kindling));
  await p.screenshot({ path: SHOTS + "/pc-2-sky.png" });
  // the next cloud is faster: the same bodies make more
  check("the next cloud is faster", await S(p, () => { const a = window.__accrete; return a.G.calc(a.s).global.toNumber() >= 4; }), "all bodies x" + (await S(p, () => window.__accrete.G.calc(window.__accrete.s).global.toNumber().toFixed(2))));

  await S(p, GALAXY); await sleep(400);
  await p.locator("#views button", { hasText: "Galaxy" }).click(); await sleep(1200);
  await p.screenshot({ path: SHOTS + "/pc-3-galaxy.png" });
  await p.locator("#go").click();
  check("collapse asks first and offers a law", (await p.locator("#card .choice").count()) === 7);
  await p.locator("#card .choice", { hasText: "Thin dust" }).click();
  await p.locator("#card .btn.hot").click();
  check("collapse: dark matter, a galaxy under the chosen law", await until(() => S(p, () => { const a = window.__accrete; return a.s.life.collapses === 2 && a.s.law === "thin" && a.s.darkUni.toNumber() >= 5 && a.s.light.isZero(); }), 12000));
  await p.locator("#pane-galaxy .cardrow", { hasText: "Black hole" }).click(); await sleep(300);
  check("a well is dug with dark matter", await S(p, () => window.__accrete.s.wells.hole === 1));

  await S(p, UNIVERSE); await sleep(400);
  await p.locator("#views button", { hasText: "Universe" }).click(); await sleep(1200);
  await p.locator("#go").click();
  check("a bang asks first and offers the six constants", (await p.locator("#card .choice").count()) === 6);
  await p.locator("#card .choice", { hasText: "Speed" }).click();
  await p.locator("#card .btn.hot").click();
  check("bang: a constant raised, laws kept, everything below begun again", await until(() => S(p, () => { const a = window.__accrete; return a.s.life.bangs === 1 && a.s.consts.c === 1 && a.s.darkUni.isZero() && a.G.count(a.s.shep) === 6; }), 12000));
  await sleep(1500);
  await p.screenshot({ path: SHOTS + "/pc-4-universe.png" });
  // the end
  await S(p, () => { const a = window.__accrete, s = a.s; s.consts = { G: 1, c: 1, h: 1, L: 1, O: 1, a: 0 }; s.life.bangs = 5; s.darkUni = window.Big.from(1e6); });
  await sleep(500);
  await p.locator("#go").click(); await p.locator("#card .choice", { hasText: "Memory" }).click(); await p.locator("#card .btn.hot").click();
  check("the end: said plainly when every constant has been raised", await until(async () => (await p.locator("#card h1").innerText()).includes("complete"), 12000));
  await p.screenshot({ path: SHOTS + "/pc-5-end.png" });
  await ctx.close();

  // ── 6 · a phone ──
  ctx = await browser.newContext({ ...devices["iPhone 14"] });
  p = await open(ctx);
  await p.locator("#bodies .body:visible .buy").first().tap(); await sleep(400);
  check("on a phone a tap buys", await S(p, () => window.__accrete.s.own[0] === 1));
  await p.screenshot({ path: SHOTS + "/phone-0-start.png" });
  await p.locator("#tabbar button", { hasText: "Upgrades" }).tap(); await sleep(400);
  check("on a phone the sheet switches between bodies and upgrades", (await p.locator("#right").isVisible()) && !(await p.locator("#left").isVisible()) && (await p.locator("#goals .goal:visible").count()) >= 1);
  await p.screenshot({ path: SHOTS + "/phone-1-upgrades.png" });
  await S(p, READY); await S(p, () => { const a = window.__accrete; a.s.life.ignitions = 3; a.s.stars = 3; a.s.light = window.Big.from(4); a.s.lightGal = window.Big.from(4); a.s.classes = { M: 1 }; a.s.seen.ignite = true; });
  await sleep(600);
  await p.locator("#tabbar button", { hasText: "Sky" }).tap(); await sleep(1500);
  check("on a phone the sky has its own tab", (await S(p, () => window.__accrete.scene.view())) === "sky" && (await p.locator("#pane-sky").isVisible()));
  await p.screenshot({ path: SHOTS + "/phone-2-sky.png" });
  await p.locator("#tabbar button", { hasText: "Bodies" }).tap(); await sleep(1200);
  await p.screenshot({ path: SHOTS + "/phone-3-cloud.png" });
  check("on a phone nothing runs off the side", await S(p, () => document.documentElement.scrollWidth <= window.innerWidth + 1));
  await ctx.close();

  await browser.close();
  check("no errors in the console through all of it", errors.length === 0, [...new Set(errors)].slice(0, 4).join(" | ") || "clean");
  const bad = results.filter((r) => !r).length;
  console.log(bad ? `\n${bad} of ${results.length} FAILED` : `\nall ${results.length} passed`);
  process.exit(bad ? 1 : 0);
})();
