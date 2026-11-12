// Ultron · browser checks of the page. They run against a SCRATCH copy of the server (never the live one:
// he is spoken to and his talk is forgotten), whose two voices are the stand-ins in mockvoice.js, with the
// real mind behind it. A stand-in speech recogniser and a drawn camera view take the place of a room.
//
//   node test/mockvoice.js &                                   (two pretend VoiceStudios on 5940 and 5941)
//   PORT=5933 DATA_DIR=/tmp/ultron-scratch GREET_AFTER_MIN=0.05 VOICE_URL=http://127.0.0.1:5940 VOICE_KEY=x \
//     VOICE_FORMAT=wav VOICE_FAST_URL=http://127.0.0.1:5941 VOICE_FAST_KEY=x node server.js     (node 22, keys in the environment)
//   docker run --rm --network host -v /opt/plint/apps/ultron/test:/t:ro -v /tmp/plint-sketch/node_modules:/nm:ro \
//     -e NODE_PATH=/nm mcr.microsoft.com/playwright:v1.48.2-jammy node /t/page.js
//
// ONLY=1,2,3 picks the parts: 1 he speaks first, layout, TUNE and MEMORY · 2 TALK · 3 eyes (takes two minutes).
const { chromium, devices } = require('playwright');
const B = 'http://127.0.0.1:5933';
const api = (p, body) => fetch(B + p, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body || {}) });
const sleep = (ms) => new Promise((ok) => setTimeout(ok, ms));
const SHOTS = process.env.SHOTS || '/tmp';
const results = [];
const ONLY = (process.env.ONLY || '1,2,3').split(',');
const check = (name, ok, detail) => { results.push([ok, name]); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`); };
const lamp = (p) => p.textContent('#state');
const until = async (p, f, ms = 15000) => { const t0 = Date.now(); for (;;) { if (await f()) return true; if (Date.now() - t0 > ms) return false; await sleep(200); } };

const FAKE_SR = () => {
  window.__sr = { cur: null, made: 0, auto: 0 };
  class FakeSR {
    constructor() { window.__sr.made++; this.on = false; }
    start() { this.on = true; window.__sr.cur = this; if (window.__sr.auto) setTimeout(() => this._end(), window.__sr.auto); }
    abort() { this.on = false; }
    stop() { this._end(); }
    _end() { if (!this.on) return; this.on = false; if (this.onend) this.onend(); }
    _say(text) { if (this.onresult) this.onresult({ results: [Object.assign([{ transcript: text }], { isFinal: true })] }); this._end(); }
  }
  window.SpeechRecognition = FakeSR; window.webkitSpeechRecognition = FakeSR;
  window.__say = (t) => (window.__sr.cur && window.__sr.cur.on ? (window.__sr.cur._say(t), true) : false);
  window.__silence = () => (window.__sr.cur && window.__sr.cur.on ? (window.__sr.cur._end(), true) : false);
};
const FAKE_CAM = () => {
  const c = document.createElement('canvas'); c.width = 640; c.height = 480;
  const g = c.getContext('2d');
  window.__scene = (n) => {
    g.fillStyle = n ? '#f4f4f4' : '#101418'; g.fillRect(0, 0, 640, 480);
    if (n) { g.fillStyle = '#c0281c'; g.fillRect(60, 80, 300, 320); g.fillStyle = '#111'; g.font = 'bold 90px sans-serif'; g.fillText('MUG', 90, 270); }
    else { g.fillStyle = '#2a6'; g.beginPath(); g.arc(470, 240, 90, 0, 7); g.fill(); }
  };
  window.__scene(0);
  setInterval(() => { g.fillStyle = 'rgba(0,0,0,0.002)'; g.fillRect(0, 0, 1, 1); }, 200);     // keeps frames coming
  navigator.mediaDevices.getUserMedia = async () => c.captureStream(5);
};

(async () => {
  const browser = await chromium.launch({ args: ['--autoplay-policy=document-user-activation-required'] });
  const errors = [];
  const open = async (init, opts = {}) => {
    const ctx = await browser.newContext({ ...devices['iPhone 15 Pro'], ...opts });
    for (const f of init) await ctx.addInitScript(f);
    const p = await ctx.newPage();
    p.on('console', (m) => { if (m.type() === 'error' && !/AudioContext|autoplay/i.test(m.text())) errors.push(m.text()); });
    p.on('pageerror', (e) => errors.push('pageerror ' + e.message));
    return [ctx, p];
  };

  // ── 1. he speaks first when you come in; the first touch lets him be heard ──
  if (ONLY.includes('1')) {
    await api('/api/forget'); await sleep(4500);
    const [ctx, p] = await open([]);
    await p.goto(B + '/', { waitUntil: 'load' });
    const spoke = await until(p, async () => (await p.locator('.ln.he p .s').count()) > 0, 20000);
    check('he says something on his own when the page opens', spoke, spoke ? (await p.locator('.ln.he p').last().textContent()).slice(0, 90) : '');
    check('no line of yours was added for it', (await p.locator('.ln.me').count()) === 0);
    const held = await until(p, async () => /tap anywhere/i.test(await p.textContent('#notice')), 12000);
    check('with no touch yet he waits to be heard, and says so', held, await p.textContent('#notice'));
    await p.screenshot({ path: SHOTS + '/t1-greeting-held.png' });
    await p.locator('#log').tap();
    const speaking = await until(p, async () => (await lamp(p)) === 'SPEAKING', 5000);
    check('the first touch plays him', speaking, await lamp(p));
    const done = await until(p, async () => (await lamp(p)) === 'READY', 20000);
    check('and he comes to rest', done && !(await p.isVisible('#notice')), await lamp(p));

    // layout
    const m = await p.evaluate(() => {
      const tops = (sel) => [...document.querySelectorAll(sel)].filter((e) => !e.hidden).map((e) => Math.round(e.getBoundingClientRect().top));
      return { sw: tops('.deck .sw > *'), acts: tops('.acts > *'), log: document.getElementById('log').clientHeight, deck: document.querySelector('.deck').offsetHeight, over: document.documentElement.scrollWidth > innerWidth, bar: document.querySelector('.grp.bar').getBoundingClientRect().height };
    });
    check('the switches sit on one row', new Set(m.sw).size === 1, JSON.stringify(m));
    check('the transcript has more room than before (was 151 px)', m.log > 195, m.log + ' px');
    check('nothing overflows sideways', !m.over);
    await p.screenshot({ path: SHOTS + '/t2-layout.png' });

    // a second talk, forgotten, so there is a talk to remember
    await p.fill('#input', 'I am repainting the hallway green this week.'); await p.tap('#send');
    await until(p, async () => (await lamp(p)) === 'READY' && (await p.locator('.ln.he').count()) >= 2, 25000);
    await p.fill('#input', 'My sister Ingrid thinks green is a mistake.'); await p.tap('#send');
    await sleep(800);
    await until(p, async () => (await lamp(p)) === 'READY', 25000);
    check('typed turns still work and are spoken', (await p.locator('.ln.me').count()) === 2 && (await p.locator('.ln.he').count()) >= 3);
    await p.tap('#swForget'); await p.tap('#swForget'); await sleep(6000);

    // TUNE
    await p.tap('#swTune'); await sleep(700);
    const rows = await p.locator('#tuneRows .trow b').allTextContents();
    check('TUNE on Haiku: REACH is there, WILDNESS is not', rows.includes('REACH') && !rows.includes('WILDNESS'), rows.join(' · '));
    check('TUNE names the mind in use', /HAIKU/.test(await p.textContent('#tuneWho')), await p.textContent('#tuneWho'));
    await p.screenshot({ path: SHOTS + '/t3-tune.png' });
    await p.tap('#tuneClose');
    // MEMORY
    await p.tap('#swMem'); await sleep(1500);
    const talks = await p.locator('#memTalks .fact').count();
    check('MEMORY lists the talk he now remembers', talks >= 1, talks ? (await p.locator('#memTalks .fact span').first().textContent()).slice(0, 140) : await p.textContent('#memCount'));
    await p.evaluate(() => document.getElementById('memTalks').scrollIntoView());
    await p.screenshot({ path: SHOTS + '/t4-memory.png' });
    await p.tap('#memClose');
    await ctx.close();
  }

  // ── 2. TALK stays on; after a while he answers to his name ──
  if (ONLY.includes('2')) {
    const [ctx, p] = await open([FAKE_SR, () => { try { localStorage.setItem('ultron.awake', '0'); } catch {} }]);
    await p.goto(B + '/', { waitUntil: 'load' }); await sleep(1200);
    await p.tap('#swLoop');
    check('TALK on: he listens', (await lamp(p)) === 'LISTENING', await lamp(p));
    const users = () => p.locator('.ln.me:not(.old)').count();
    await p.evaluate(() => window.__say('what do you make of rain'));
    check('what you say right away is taken', await until(p, async () => (await users()) === 1, 4000));
    await until(p, async () => (await lamp(p)) === 'LISTENING', 30000);
    check('after his answer he listens again by himself', (await lamp(p)) === 'LISTENING', await lamp(p));
    // three silences in a row: the old page gave up after two
    for (let i = 0; i < 3; i++) { await sleep(1500); await p.evaluate(() => window.__silence()); await sleep(600); }
    check('silences do not switch TALK off', (await p.getAttribute('#swLoop', 'aria-pressed')) === 'true' && /LISTENING|STANDBY/.test(await lamp(p)), await lamp(p));
    const standby = await until(p, async () => (await lamp(p)) === 'STANDBY', 32000);
    check('after 25 s without a word the lamp says STANDBY', standby, await lamp(p));
    await p.evaluate(() => window.__say('did you put the bins out'));
    await sleep(1500);
    check('talk in the room that does not name him is left alone', (await users()) === 1 && (await p.inputValue('#input')) === '', `${await users()} lines`);
    await until(p, async () => p.evaluate(() => !!(window.__sr.cur && window.__sr.cur.on)), 3000);
    await p.evaluate(() => window.__say('Ultron what do you make of that'));
    check('his name gets his attention', await until(p, async () => (await users()) === 2, 4000));
    await until(p, async () => /LISTENING/.test(await lamp(p)), 30000);
    // a recogniser that dies at once, over and over
    await p.evaluate(() => { window.__sr.auto = 5; window.__silence(); });
    const off = await until(p, async () => (await p.getAttribute('#swLoop', 'aria-pressed')) === 'false', 45000);
    check('a recogniser that keeps dying at once switches TALK off, with a word why', off && /cannot keep listening/i.test(await p.textContent('#notice')), await p.textContent('#notice'));
    await ctx.close();
  }

  // ── 3. eyes: a first look, then a look when the view changes ──
  if (ONLY.includes('3')) {
    const [ctx, p] = await open([FAKE_CAM], { permissions: ['camera'] });
    const cues = [];
    p.on('request', (r) => { if (r.url().endsWith('/api/say') && r.method() === 'POST') { const b = JSON.parse(r.postData() || '{}'); cues.push({ cue: b.cue || null, image: b.image ? b.image.length : 0, at: Date.now() }); } });
    await p.goto(B + '/', { waitUntil: 'load' }); await sleep(1500);
    await p.tap('#swEyes');
    const first = await until(p, async () => cues.some((c) => c.cue === 'glance' && c.image > 1000), 40000);
    check('eyes on: he takes a first look, with a frame', first, JSON.stringify(cues.map((c) => [c.cue, c.image])));
    await until(p, async () => (await lamp(p)) === 'READY' && !(await p.evaluate(() => false)), 25000);
    await sleep(3000);
    const said = await p.locator('.ln.he:not(.old) p').allTextContents();
    console.log('      what he said on looking:', JSON.stringify(said.map((s) => s.slice(0, 120))));
    const before = cues.length;
    console.log('      … waiting out the 90 s between looks, then changing the view');
    await sleep(93000);
    check('an unchanged view is left alone', cues.length === before, `${cues.length - before} more`);
    await p.evaluate(() => window.__scene(1));
    const second = await until(p, async () => cues.length > before && cues[cues.length - 1].cue === 'glance', 15000);
    check('when the view changes and settles, he looks again', second, JSON.stringify(cues.slice(before).map((c) => [c.cue, c.image])));
    await until(p, async () => (await lamp(p)) === 'READY', 25000); await sleep(2500);
    console.log('      and then:', JSON.stringify((await p.locator('.ln.he:not(.old) p').allTextContents()).slice(said.length).map((s) => s.slice(0, 140))));
    await p.screenshot({ path: SHOTS + '/t5-eyes.png' });
    await ctx.close();
  }

  check('no errors in the page', errors.length === 0, errors.slice(0, 3).join(' | '));
  await browser.close();
  const bad = results.filter(([ok]) => !ok).length;
  console.log(`\n${results.length - bad}/${results.length} passed`);
})().catch((e) => { console.error('test crashed:', e.message); process.exit(1); });
