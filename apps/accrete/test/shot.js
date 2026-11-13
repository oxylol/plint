// Accrete · pictures of the page in set-up states, to look at. Against a scratch copy of the server:
//   PORT=5941 node server.js &
//   docker run --rm --network host -v /opt/plint/apps/accrete/test:/t:ro -v /tmp/plint-sketch/node_modules:/nm:ro \
//     -e NODE_PATH=/nm -v /tmp/accrete-shots:/shots mcr.microsoft.com/playwright:v1.48.2-jammy node /t/shot.js [names…]
const { chromium, devices } = require("playwright");
const B = process.env.BASE || "http://127.0.0.1:5941";
const OUT = process.env.SHOTS || "/shots";
const want = process.argv.slice(2);
const sleep = (ms) => new Promise((ok) => setTimeout(ok, ms));

// Each state is made by writing into the running game, then letting it draw.
const STATES = {
  start: () => {},
  early: () => { const a = window.__accrete, s = a.s, Big = window.Big; s.own = [14, 0, 0, 0, 0, 0, 0, 0]; s.mass = Big.from(86); s.cloud.mass = Big.from(410); s.life.mass = Big.from(410); s.life.bought = 14; s.up.grip1 = true; s.life.pulls = 20; },
  mid: () => { const a = window.__accrete, s = a.s, Big = window.Big; s.own = [72, 41, 30, 17, 6, 0, 0, 0]; for (let k = 1; k < 5; k++) { s.seen["k" + k] = true; s.seen["o" + k] = true; } s.mass = Big.from(2.1e6); s.cloud.mass = Big.from(3.4e7); s.life.mass = Big.from(3.4e7); s.life.bought = 166; s.life.comets = 4; s.life.pulls = 200; Object.assign(s.up, { grip1: 1, spin1: 1, grip2: 1, spin2: 1, dust: 1, link0: 1, link1: 1, tail1: 1 }); s.shep[0] = true; s.shep[1] = true; s.seen.s0 = s.seen.s1 = s.seen.s2 = true; ["sp1", "sp2", "sp3", "sp4", "sp5", "ga1", "ga2", "cr1", "cr2", "fa1", "sh1"].forEach((m) => (s.ms[m] = true)); s.life.most = 72; s.life.kinds = 5; },
  comet: () => { STATES_mid(); const a = window.__accrete, s = a.s; s.comet.wait = 0.05; s.surge = { mult: 3, left: 40 }; },
  ready: () => { const a = window.__accrete, s = a.s, Big = window.Big; s.own = [142, 108, 89, 70, 62, 48, 0, 0]; for (let k = 1; k < 6; k++) { s.seen["k" + k] = true; s.seen["o" + k] = true; } s.seen.k3 = true; s.mass = Big.from(4.4e9); s.cloud.mass = Big.from(6.6e11); s.life.mass = Big.from(6.6e11); s.life.bought = 519; s.life.pulls = 300; s.life.comets = 11; for (const u of a.D.UPGRADES) if (!/link5|link6/.test(u.id)) s.up[u.id] = true; for (let k = 0; k < 6; k++) { s.shep[k] = true; s.seen["s" + k] = true; } s.life.most = 142; s.life.kinds = 6; },
  sky: () => { const a = window.__accrete, s = a.s, Big = window.Big; STATES_ready(); s.life.ignitions = 37; s.stars = 37; s.light = Big.from(64); s.lightGal = Big.from(412); s.life.light = Big.from(412); s.classes = { M: true, K: true, G: true }; s.life.classes = { M: true, K: true, G: true }; s.life.fastest = 190; ["kindling", "seed", "widenet", "openhand", "giants", "sapling", "bellows", "steward", "lens"].forEach((n) => (s.nodes[n] = true)); s.seen.ignite = true; s.seen.k6 = true; s.own[6] = 12; a.scene.setView("sky"); },
  galaxy: () => { const a = window.__accrete, s = a.s, Big = window.Big; STATES_sky(); s.stars = 2400; s.life.ignitions = 9000; s.lightGal = Big.from(3.2e7); s.light = Big.from(1.1e7); for (const n of a.D.NODES) s.nodes[n.id] = true; s.life.collapses = 7; s.galaxies = 7; s.dark = Big.from(5); s.darkUni = Big.from(19); s.life.dark = Big.from(19); s.wells = { hole: 2, nursery: 2, gazer: 1, memory: 1 }; s.laws = { bare: true, dim: true }; s.law = "flat"; s.seen.collapse = true; s.classes = { M: 1, K: 1, G: 1, F: 1, A: 1 }; a.scene.setView("galaxy"); },
  universe: () => { const a = window.__accrete, s = a.s, Big = window.Big; STATES_galaxy(); s.life.bangs = 3; s.life.collapses = 96; s.consts = { G: 1, c: 1, h: 0, L: 1, O: 0, a: 0 }; s.echo = 3; s.seen.bang = true; s.darkUni = Big.from(240); a.scene.setView("universe"); },
};
const src = "(" + Object.entries(STATES).map(([k, f]) => `window.STATES_${k} = ${f.toString()}`).join(", ") + ")";

(async () => {
  const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
  const errors = [];
  const list = want.length ? want : ["start", "early", "mid", "ready"];
  for (const name of list) {
    const [state, dev] = name.split("@");
    const ctx = dev === "phone" ? await browser.newContext({ ...devices["iPhone 14"] }) : await browser.newContext({ viewport: { width: Number(process.env.W || 1920), height: Number(process.env.H || 1080) }, deviceScaleFactor: 1 });
    const page = await ctx.newPage();
    page.on("console", (m) => { if (m.type() === "error") errors.push(name + ": " + m.text()); });
    page.on("pageerror", (e) => errors.push(name + ": " + e.message));
    await page.goto(B + "/", { waitUntil: "load" });
    await sleep(400);
    await page.evaluate(src);
    await page.evaluate((n) => { const f = window["STATES_" + n]; f(); }, state);
    await sleep(Number(process.env.WAIT || 2600));
    if (process.env.EVAL) await page.evaluate(process.env.EVAL);
    if (process.env.EVAL) await sleep(900);
    await page.screenshot({ path: `${OUT}/${name.replace("@", "-")}.png` });
    console.log("shot " + name);
    await ctx.close();
  }
  await browser.close();
  if (errors.length) { console.log("ERRORS\n" + [...new Set(errors)].join("\n")); process.exit(1); }
})();
