// Stocks · browser checks of the page, at two sizes (PC 1440x900, phone 390x844), against the fixture server:
// never a live site. One check or more for each of the seven points in PAGE.md, in each of the three fixture
// states (running, keys, stopped), plus the page's own rules (escaping, refresh, look). Every check compares
// what is on the page with what is in the fixture files, so it fails when the page is wrong.
//
//   cd /opt/plint/apps/stocks                                    (or the worktree's apps/stocks)
//   python3 tests/serve_fixture.py 5258 &                        (the page over the fixtures, theme slipped in as Caddy does)
//   mkdir -p /opt/plint/.scratch/stocks-shots
//   docker run --rm --network host -v "$PWD/tests":/t:ro -v /tmp/plint-sketch/node_modules:/nm:ro \
//     -v /opt/plint/.scratch/stocks-shots:/shots -e NODE_PATH=/nm \
//     mcr.microsoft.com/playwright:v1.48.2-jammy node /t/page.js
//
// /tmp/plint-sketch/node_modules holds playwright 1.48.2 (if /tmp was emptied: npm i playwright@1.48.2 in any
// folder and mount its node_modules instead). ONLY=board,books,keys,refresh,hostile,look,touch picks groups;
// SIZES=pc or SIZES=phone picks one size; BASE and SHOTS move the server and the screenshots.
// The one request that is expected to fail is /_plint/health: the Plint strip asks Uptime Kuma for it through
// Caddy, and the fixture server has no Kuma behind it.
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const BASE = process.env.BASE || 'http://127.0.0.1:5258';
const SHOTS = process.env.SHOTS || '/shots';
const ONLY = (process.env.ONLY || 'board,books,keys,refresh,hostile,look,touch').split(',');
const SIZES = (process.env.SIZES || 'pc,phone').split(',');
const VIEW = {
  pc: { viewport: { width: 1440, height: 900 } },
  phone: { viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true },
};
const fx = (name) => JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', name), 'utf8'));
const stateOf = (fixture) => fx(fixture === 'running' ? 'state.json' : `state_${fixture}.json`);
const sleep = (ms) => new Promise((ok) => setTimeout(ok, ms));
const until = async (f, ms = 8000) => { const t0 = Date.now(); for (;;) { if (await f()) return true; if (Date.now() - t0 > ms) return false; await sleep(100); } };

const results = [];
const check = (name, ok, detail) => { results.push([!!ok, name]); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + String(detail).slice(0, 300) : ''}`); };

// ── what the page should say, worked out here from the fixture, not taken from the page ──
const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const norm = (s) => (s == null ? s : s.replace(/\u00a0/g, ' ').replace(/\u2212/g, '-').replace(/\s+/g, ' ').trim());
const fixed = (v, dp) => Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: dp, maximumFractionDigits: dp });
const rnd = (v, dp) => { const f = 10 ** dp, r = Math.round(v * f) / f; return r === 0 ? 0 : r; };
const pct = (x, dp = 2, sign = true) => { const r = rnd(x * 100, dp); return (r > 0 && sign ? '+' : r < 0 ? '-' : '') + fixed(r, dp) + ' %'; };
const usd = (x, dp = 2) => (x < 0 ? '-' : '') + '$' + fixed(x, dp);
const day = (s) => { const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(s); return `${+m[3]} ${MON[+m[2] - 1]}`; };
const vsKey = (m) => (typeof m.vs_index === 'number' ? m.vs_index : -Infinity);
const order = (st) => {
  const idx = st.models.filter((m) => m.kind === 'index'), rest = st.models.filter((m) => m.kind !== 'index');
  const run = rest.filter((m) => m.state === 'running').sort((a, b) => vsKey(b) - vsKey(a));
  const out = rest.filter((m) => m.state !== 'running').sort((a, b) => vsKey(b) - vsKey(a));
  return { idx, run, out, all: [...idx, ...run, ...out] };
};
const BAD_WORDS = /\b(NaN|null|undefined|Infinity)\b|\[object/;
const isHealth = (u) => /\/_plint\/health/.test(u || '');

async function open(browser, size, o = {}) {
  const ctx = await browser.newContext({ ...VIEW[size], timezoneId: 'Europe/Stockholm', locale: 'en-GB', reducedMotion: o.motion ? 'no-preference' : 'reduce' });
  const page = await ctx.newPage();
  const seen = { errors: [], requests: [], bad: [] };
  page.on('console', (m) => { if (m.type() === 'error' && !isHealth((m.location() || {}).url) && !(o.failing && /Failed to load resource|net::ERR/.test(m.text()))) seen.errors.push(m.text()); });
  page.on('pageerror', (e) => seen.errors.push('pageerror: ' + e.message));
  page.on('request', (r) => seen.requests.push(r.url()));
  page.on('response', (r) => { if (r.status() >= 400 && !isHealth(r.url()) && !o.failing) seen.bad.push(r.status() + ' ' + r.url()); });
  page.on('requestfailed', (r) => { if (!o.failing) seen.bad.push('failed ' + r.url()); });
  return { ctx, page, seen };
}
const url = (fixture, hash = '') => `${BASE}/${fixture === 'running' ? '' : '?fixture=' + fixture}${hash}`;
const boardReady = (page) => page.waitForSelector('#board .rows > li', { timeout: 10000 });
const bookReady = (page) => page.waitForSelector('#detail .bk-body', { timeout: 10000 });
const text = (page, sel) => page.evaluate((s) => { const e = document.querySelector(s); return e ? e.textContent : null; }, sel);
const visibleText = (page) => page.evaluate(() => document.body.innerText);
const sideways = (page) => page.evaluate(() => ({ sw: document.documentElement.scrollWidth, iw: innerWidth, cw: document.documentElement.clientWidth }));
// what sticks out past the right edge, widest first (said when a sideways check fails)
const wide = (page, w) => page.evaluate((w) => [...document.querySelectorAll('body *')].map((e) => [e, e.getBoundingClientRect()]).filter(([e, r]) => r.width && r.right > w + 0.5 && getComputedStyle(e).position !== 'fixed')
  .sort((a, b) => b[1].right - a[1].right).slice(0, 4).map(([e, r]) => `${e.tagName.toLowerCase()}.${String(e.getAttribute('class') || '').replace(/\s+/g, '.')} to ${Math.round(r.right)} "${(e.textContent || '').trim().slice(0, 24)}"`).join(' | '), w);
const shot = (page, name, full = true) => page.screenshot({ path: path.join(SHOTS, name + '.png'), fullPage: full });
const quiet = (seen, tag) => {
  check(`${tag} no script error, no failed request (bar the strip's health call)`, !seen.errors.length && !seen.bad.length, [...seen.errors, ...seen.bad].join(' | '));
  const away = seen.requests.filter((u) => !u.startsWith(BASE + '/') && !u.startsWith('data:'));
  check(`${tag} every request went to the app's own host`, !away.length, away.join(' '));
};
// the rows of the overview as the page shows them
const rowsOf = (page) => page.evaluate(() => [...document.querySelectorAll('#board .rows > li.row')].map((li) => {
  const g = (sel) => { const e = li.querySelector(sel); return e ? e.textContent : null; };
  const r = li.getBoundingClientRect(), lamp = li.querySelector('.c-lamp .p-blk');
  return {
    id: li.dataset.id, state: li.dataset.state, cls: li.className, title: g('.t'), tag: g('.tag'), st: g('.st'), what: g('.c-what'),
    ret: g('.c-ret .v'), idx: g('.c-idx .v'), vs: g('.c-vs .v'), day: g('.c-day .v'), dd: g('.c-dd .v'), tr: g('.c-tr .v'), cost: g('.c-cost .v'), since: g('.c-since .v'),
    lamp: lamp ? lamp.className : null, lampText: g('.c-lamp .v'), lines: li.querySelectorAll('.spark polyline').length,
    solid: li.querySelectorAll('.spark .ln-model').length, dashed: li.querySelectorAll('.spark .ln-index').length,
    top: r.top + scrollY, bottom: r.bottom + scrollY, left: r.left, right: r.right, height: r.height, isLink: !!li.querySelector('a.row-a[href]'),
  };
}));

// ── 1 · 6 · 7 on the overview, for one size and one fixture ─────────────────────────────────────────────
async function board(browser, size, fixture) {
  const tag = `[${size} ${fixture}]`, st = stateOf(fixture), exp = order(st);
  const { ctx, page, seen } = await open(browser, size);
  await page.goto(url(fixture)); await boardReady(page); await page.waitForTimeout(400);
  await shot(page, `${size}-${fixture}-board`);
  const rows = await rowsOf(page), by = Object.fromEntries(rows.map((r) => [r.id, r]));

  // 1. every model at once; the index is the bar; the rest best to worst against it; not running last, saying why
  check(`${tag} 1 every model is listed (${st.models.length})`, rows.length === st.models.length && rows.every((r) => r.isLink), rows.map((r) => r.id).join(' '));
  check(`${tag} 1 the index comes first and is marked as the bar, not ranked`, rows[0].id === exp.idx[0].id && /is-index/.test(rows[0].cls) && /the bar/i.test(rows[0].tag || ''), `${rows[0].id} ${rows[0].cls} ${rows[0].tag}`);
  check(`${tag} 1 then the running models best to worst against the index, then those not running`, rows.map((r) => r.id).join() === exp.all.map((m) => m.id).join(), rows.map((r) => r.id).join());
  const wrong = [];
  for (const m of st.models) {
    const r = by[m.id];
    if (typeof m.ret !== 'number') { if (r.ret != null || r.vs != null) wrong.push(`${m.id} shows figures it does not have`); continue; }
    const want = { ret: pct(m.ret), idx: pct(m.index_ret), vs: m.kind === 'index' ? 'the bar' : pct(m.vs_index), day: pct(m.day_ret), dd: pct(m.max_dd), tr: fixed(m.trades, 0), cost: usd(m.costs), since: day(m.since) };
    for (const k of Object.keys(want)) if (norm(r[k]) !== want[k]) wrong.push(`${m.id}.${k}: page "${norm(r[k])}" fixture "${want[k]}"`);
  }
  check(`${tag} 1 ${st.models.some((m) => typeof m.ret === 'number') ? 'each row shows return, index, vs index, today, worst drop, trades, costs and since as in the fixture' : 'no row shows a figure, since no model has one yet'}`, !wrong.length, wrong.join('; '));
  const withRec = st.models.filter((m) => typeof m.ret === 'number');
  const curves = withRec.filter((m) => { const r = by[m.id]; return m.kind === 'index' ? r.dashed !== 1 || r.solid !== 0 : r.solid !== 1 || r.dashed !== 1; });
  if (withRec.length) check(`${tag} 1 each row has its small curve: the model solid, the index dashed`, !curves.length, curves.map((m) => m.id).join());
  const lampWrong = withRec.filter((m) => {
    const r = by[m.id];
    if (!m.verdict) return r.lamp != null;
    const want = m.verdict.lamp === 'likely' ? 'p-blk' : m.verdict.lamp === 'unlikely' ? 'p-blk no' : 'p-blk off';
    return r.lamp !== want || !norm(r.lampText).includes(norm(m.verdict.text));
  });
  if (withRec.length) check(`${tag} 1 each scored model has its verdict lamp and the engine's sentence; the unscored have none`, !lampWrong.length, lampWrong.map((m) => m.id + ':' + by[m.id].lamp).join());
  const off = st.models.filter((m) => m.state !== 'running' && m.kind !== 'index');
  const offWrong = off.filter((m) => { const r = by[m.id]; return r.state === 'running' || !norm(r.st || '').toLowerCase().includes(m.state_text.toLowerCase()); });
  if (off.length) check(`${tag} 1 a model that is not running says why (${off.length} such)`, !offWrong.length, offWrong.map((m) => m.id + ': ' + by[m.id].st).join('; '));

  // the two layouts
  const lay = await page.evaluate(() => {
    const vis = (e) => { if (!e) return false; const r = e.getBoundingClientRect(), s = getComputedStyle(e); return s.display !== 'none' && s.visibility !== 'hidden' && r.width > 2 && r.height > 2; };
    const row = document.querySelector('#board .row:not(.is-bare):not(.is-index)');
    const cols = document.querySelector('#board .cols');
    const edge = (root, c) => { const e = root && root.querySelector('.c-' + c); return e ? Math.round(e.getBoundingClientRect().right * 10) / 10 : null; };
    return {
      cols: vis(cols), labels: row ? ['ret', 'idx', 'day', 'dd', 'tr', 'cost', 'since'].filter((c) => vis(row.querySelector('.c-' + c + ' > .k'))) : null,
      vsLabel: row ? vis(row.querySelector('.c-vs > .k')) : null,
      head: cols ? ['ret', 'idx', 'day', 'dd', 'tr', 'cost', 'since'].map((c) => [c, edge(cols, c), edge(row, c)]) : null,
      rowH: row ? row.getBoundingClientRect().height : null, inner: innerHeight, pointer: matchMedia('(pointer: coarse)').matches,
    };
  });
  const sw = await sideways(page);
  if (withRec.length) {
    if (size === 'pc') {
      const askew = (lay.head || []).filter((h) => h[1] == null || Math.abs(h[1] - h[2]) > 1.5);
      check(`${tag} 1 PC: one table, a line a model, under one row of headings that line up with their columns`, lay.cols && lay.rowH < 70 && lay.labels.length === 0 && !askew.length, JSON.stringify({ cols: lay.cols, rowH: lay.rowH, labels: lay.labels, askew }));
      if (fixture === 'running') check(`${tag} 1 PC: the whole field fits one screen without scrolling`, rows[rows.length - 1].bottom <= lay.inner, `last row ends at ${Math.round(rows[rows.length - 1].bottom)} of ${lay.inner}`);
    } else {
      check(`${tag} 1 phone: each model is a block with every figure under its own label, no table headings`, !lay.cols && lay.labels.length === 7 && lay.vsLabel && lay.rowH > 150, JSON.stringify({ cols: lay.cols, rowH: lay.rowH, labels: lay.labels }));
    }
  }
  check(`${tag} 1 nothing scrolls sideways`, sw.sw <= sw.iw && sw.iw === VIEW[size].viewport.width && rows.every((r) => r.left >= 0 && r.right <= sw.iw + 0.5), JSON.stringify(sw));

  // 6. the first thing on the page says what is missing
  const status = await page.evaluate(() => {
    const box = document.getElementById('status'), main = document.getElementById('main');
    const top = (e) => e.getBoundingClientRect().top + scrollY;
    return {
      first: main.firstElementChild === box, above: top(box) < top(document.getElementById('board')), phase: (box.querySelector('.st-phase') || {}).textContent, held: !!box.querySelector('.st-phase.is-held'),
      line: (box.querySelector('.st-line') || {}).textContent, busy: (box.querySelector('.st-busy') || {}).textContent || null,
      keys: [...box.querySelectorAll('.st-keys li')].map((li) => [li.dataset.key, li.className, li.textContent]),
      notes: [...box.querySelectorAll('.st-notes li')].map((li) => [li.className, li.textContent, li.querySelector('.p-blk').className]),
    };
  });
  check(`${tag} 6 the engine's state is the first thing on the page and names the phase ("${st.phase_text}")`, status.first && status.above && norm(status.phase) === st.phase_text && status.held === (st.phase !== 'running'), JSON.stringify([status.first, status.phase, status.held]));
  const missing = Object.keys(st.keys).filter((k) => st.keys[k] === false);
  const keysOk = missing.length
    ? Object.keys(st.keys).every((k) => { const li = status.keys.find((x) => x[0] === k); return li && (st.keys[k] ? /is-set/.test(li[1]) && /set$/.test(norm(li[2])) : /is-missing/.test(li[1]) && /missing$/.test(norm(li[2]))); })
    : status.keys.length === 0;
  check(`${tag} 6 keys: ${missing.length ? missing.join(', ') + ' shown as missing' : 'none missing, so no key list'}`, keysOk, JSON.stringify(status.keys));
  const LAMP = { fault: 'p-blk alarm', warn: 'p-blk', info: 'p-blk off' };
  const notesOk = status.notes.length === st.notices.length && st.notices.every((n, i) => status.notes[i][0] === 'lv-' + n.level && norm(status.notes[i][1]) === norm(n.text.replace(/`/g, '')) && status.notes[i][2] === LAMP[n.level]);
  check(`${tag} 6 notices: ${st.notices.length} shown in the engine's words, each with the lamp of its level`, notesOk, JSON.stringify(status.notes));
  check(`${tag} 6 the clock: ${st.clock.busy ? 'says what the engine is doing now' : 'no work under way, so no such line'}`, st.clock.busy ? norm(status.busy || '').includes(st.clock.busy) : status.busy == null, status.busy);
  const dates = norm(status.line);
  check(`${tag} 6 it says when the figures were read${st.asof ? ' and which close they stand at' : ''}`, /figures read 13 Nov 22:41/.test(dates) && (st.asof ? dates.includes('last close ' + day(st.asof)) : !/last close/.test(dates)), dates);

  // 7. paper, plainly
  const paper = await page.evaluate(() => {
    const m = document.getElementById('mode'), r = m.getBoundingClientRect();
    return { badge: m.textContent, shown: r.top >= 0 && r.bottom <= 200 && r.width > 40, foot: document.querySelector('#foot .paper').textContent, fields: document.querySelectorAll('input, select, textarea, form').length };
  });
  const all = await visibleText(page);
  check(`${tag} 7 it says PAPER at the top, and in words at the foot; nothing on the page takes input`, norm(paper.badge) === 'Paper' && paper.shown && /Paper only\./.test(paper.foot) && /real order/.test(paper.foot) && paper.fields === 0 && !/\blive\b/i.test(all), JSON.stringify(paper));

  check(`${tag} no "NaN", "null" or "undefined" anywhere on the overview`, !BAD_WORDS.test(all), (BAD_WORDS.exec(all) || [])[0]);
  const hollow = await page.evaluate(() => [...document.querySelectorAll('table')].filter((t) => !t.querySelector('tbody tr')).length + (document.querySelector('#board .cols') && !document.querySelector('#board .row:not(.is-bare)') ? 1 : 0));
  check(`${tag} no table of headings with nothing under them`, hollow === 0, hollow);
  quiet(seen, tag);
  await ctx.close();
}

// ── 6 in the keys state: a first day must still look meant ───────────────────────────────────────────────
async function keys(browser, size) {
  const tag = `[${size} keys]`, st = stateOf('keys');
  const { ctx, page, seen } = await open(browser, size);
  await page.goto(url('keys')); await boardReady(page); await page.waitForTimeout(300);
  const rows = await rowsOf(page);
  check(`${tag} 6 every model is listed as waiting, with what it will do and why it waits`, rows.length === st.models.length &&
    rows.every((r, i) => { const m = st.models.find((x) => x.id === r.id); return r.state === 'waiting' && norm(r.st).toLowerCase() === m.state_text && norm(r.what) === m.what && r.ret == null; }), rows.map((r) => `${r.id}:${r.state}`).join(' '));
  const bare = await page.evaluate(() => ({ cols: !!document.querySelector('#board .cols'), tiles: !document.getElementById('tiles').hidden, legend: !!document.querySelector('#board .legend'), nils: document.querySelectorAll('#board .nil').length, tables: document.querySelectorAll('#main table').length }));
  check(`${tag} 6 no empty table, no row of dashes, no figures at a glance to be empty`, !bare.cols && !bare.tiles && !bare.legend && bare.nils === 0 && bare.tables === 0, JSON.stringify(bare));
  // every model's own page in this state
  const bad = [];
  for (const m of st.models) {
    await page.goto(url('keys', '#/' + m.id)); await bookReady(page);
    const t = await visibleText(page), d = await page.evaluate(() => ({ title: document.getElementById('book-title').textContent, state: (document.querySelector('.bk-state') || {}).textContent, tables: document.querySelectorAll('#detail table').length, hollow: [...document.querySelectorAll('#detail table')].filter((x) => !x.querySelector('tbody tr')).length, secs: [...document.querySelectorAll('#detail .sec')].map((x) => x.id), charts: document.querySelectorAll('#detail .chart').length }));
    if (BAD_WORDS.test(t)) bad.push(`${m.id}: "${BAD_WORDS.exec(t)[0]}"`);
    if (d.title !== m.title || !norm(d.state || '').toLowerCase().includes(m.state_text) || d.hollow || !d.secs.includes('sec-record')) bad.push(`${m.id}: ${JSON.stringify(d)}`);
    if (m.kind === 'agent' && (!d.secs.includes('sec-spend') || !/No backtest: the model has already read the past, so only live paper counts\./i.test(t))) bad.push('agent: spend or the backtest line is missing');
    const sw = await sideways(page);
    if (sw.sw > sw.iw) bad.push(`${m.id}: scrolls sideways`);
  }
  check(`${tag} 6 each model opens to a page that says it waits and why: no NaN, no null, no empty table (${st.models.length} pages)`, !bad.length, bad.join('; '));
  await page.goto(url('keys', '#/agent')); await bookReady(page); await shot(page, `${size}-keys-agent`);
  quiet(seen, tag);
  await ctx.close();
}

// ── 2 · 3 · 4 · 5: the open model ────────────────────────────────────────────────────────────────────────
async function books(browser, size, fixture) {
  const tag = `[${size} ${fixture}]`, st = stateOf(fixture), tap = (sel) => (size === 'phone' ? page.tap(sel) : page.click(sel));
  const { ctx, page, seen } = await open(browser, size);
  await page.goto(url(fixture)); await boardReady(page);

  // 2. open a model by a click or a tap; the address keeps it
  const net = fx('model_net.json');
  await page.evaluate(() => window.scrollTo(0, 120));
  await page.locator('#board a.row-a[data-k="row:net"]').scrollIntoViewIfNeeded();
  const y0 = await page.evaluate(() => scrollY);            // where the list stands when the row is touched
  await tap('#board a.row-a[data-k="row:net"]');
  await bookReady(page); await page.waitForTimeout(300);
  const opened = await page.evaluate(() => ({ hash: location.hash, title: document.getElementById('book-title').textContent, y: scrollY, view: document.body.dataset.view, over: getComputedStyle(document.getElementById('over')).display, sel: [...document.querySelectorAll('#board [aria-current="true"]')].map((a) => a.getAttribute('data-k')).join() }));
  check(`${tag} 2 a ${size === 'phone' ? 'tap' : 'click'} on a row opens that model, and the address says which`, opened.hash === '#/net' && opened.title === net.title && opened.y === 0, JSON.stringify(opened));
  if (size === 'pc') check(`${tag} 2 PC: the list stays beside the open model, with that model marked`, opened.over !== 'none' && opened.sel === 'row:net', JSON.stringify(opened));
  else check(`${tag} 2 phone: the open model takes the screen, the list is put away`, opened.over === 'none', opened.over);
  await shot(page, `${size}-${fixture}-net`);

  const book = await page.evaluate(() => {
    const cells = (tr) => Object.fromEntries([...tr.children].map((td) => [td.className.split(' ')[0].slice(2), td.textContent]));
    const rows = (sel) => [...document.querySelectorAll(sel + ' tbody tr')].map((tr) => ({ shown: !tr.hidden, ...cells(tr) }));
    return { positions: rows('.tb-positions'), trades: rows('.tb-trades'), pending: rows('.tb-pending'), what: (document.querySelector('.what') || {}).textContent,
      figs: [...document.querySelectorAll('.figs .fg')].map((f) => [f.querySelector('.fl').textContent, f.querySelector('.p-num').textContent]), facts: [...document.querySelectorAll('.bk-head .facts > div')].map((d) => [d.querySelector('dt').textContent, d.querySelector('dd').textContent]) };
  });
  const sm = st.models.find((m) => m.id === 'net');
  const figs = Object.fromEntries(book.figs.map((f) => [f[0], norm(f[1])]));
  check(`${tag} 2 its page leads with its figures`, figs['vs index'] === pct(sm.vs_index).replace(' %', '%') && figs['return'] === pct(sm.ret).replace(' %', '%') && figs['index'] === pct(sm.index_ret).replace(' %', '%') && figs['today'] === pct(sm.day_ret).replace(' %', '%') && figs['worst drop'] === pct(sm.max_dd).replace(' %', '%') && book.what === sm.what, JSON.stringify(figs));
  const posWrong = net.positions.map((p, i) => {
    const r = book.positions[i] || {};
    return norm(r.stock) === `${p.symbol} ${p.name}` && norm(r.w) === pct(p.weight, 1, false) && norm(r.val) === usd(p.value) && norm(r.held).startsWith(day(p.since)) && norm(r.ret) === pct(p.ret) && norm(r.why) === p.why ? null : `${p.symbol}: ${JSON.stringify(r)}`;
  }).filter(Boolean);
  check(`${tag} 2 what it holds: ${net.positions.length} holdings with share, value, since, return and why`, book.positions.length === net.positions.length && !posWrong.length, posWrong.join('; '));
  const trWrong = net.trades.map((t, i) => {
    const r = book.trades[i] || {};
    const made = t.side === 'sell' ? pct(t.ret) : null;
    return norm(r.date) === day(t.date) && norm(r.stock) === `${t.side} ${t.symbol} ${t.name}` && norm(r.usd) === usd(t.usd) && norm(r.px) === usd(t.price) && norm(r.cost) === usd(t.cost) && norm(r.sh) === fixed(rnd(t.shares, 3), 3) && norm(r.why) === t.why && (made ? norm(r.made) === made : /^–/.test(norm(r.made))) ? null : `#${i} ${t.symbol}: ${JSON.stringify(r)}`;
  }).filter(Boolean);
  check(`${tag} 2 each trade with its day, side, amount, shares, price, cost, what a sale made, and why (${net.trades.length} trades)`, book.trades.length === net.trades.length && !trWrong.length, trWrong.slice(0, 3).join('; '));
  const shown = book.trades.filter((t) => t.shown).length;
  await tap('button.more[data-k="more:trades:net"]');
  const after = await page.evaluate(() => [...document.querySelectorAll('.tb-trades tbody tr')].filter((tr) => !tr.hidden && tr.getBoundingClientRect().height > 0).length);
  check(`${tag} 2 the first 12 trades show, and one button shows all ${net.trades.length}`, shown === 12 && after === net.trades.length, `${shown} then ${after}`);
  const pendWrong = net.pending.map((p, i) => { const r = book.pending[i] || {}; return norm(r.stock) === `${p.side} ${p.symbol} ${p.name}` && norm(r.why) === p.why ? null : JSON.stringify(r); }).filter(Boolean);
  check(`${tag} 2 the orders waiting for the next open (${net.pending.length}), each with its side and why`, book.pending.length === net.pending.length && !pendWrong.length, pendWrong.join('; '));

  // the curve against the index
  const curve = await page.evaluate(() => {
    const c = document.querySelector('#sec-curve .chart-curve'), cs = (e) => getComputedStyle(e), m = c.querySelector('.ln-model'), i = c.querySelector('.ln-index'), box = c.getBoundingClientRect();
    return { model: m.getAttribute('d').split('L').length, index: i.getAttribute('d').split('L').length, mDash: cs(m).strokeDasharray, iDash: cs(i).strokeDasharray, mStroke: cs(m).stroke, iStroke: cs(i).stroke,
      rd: c.parentNode.querySelector('.rd').textContent, legend: document.querySelector('#sec-curve .legend').textContent, box: { x: box.left, y: box.top + scrollY, w: box.width, h: box.height } };
  });
  const last = net.curve[net.curve.length - 1];
  check(`${tag} 2 the curve against the index: ${net.curve.length} closes each, the model a solid lit line, the index a dimmer dashed one`, curve.model === net.curve.length && curve.index === net.curve.length && curve.mDash === 'none' && curve.iDash !== 'none' && curve.mStroke === 'rgb(228, 248, 234)' && /rgba\(228, 248, 234, 0\.\d+\)/.test(curve.iStroke), JSON.stringify(curve));
  check(`${tag} 2 under it, in words, the last close: model, index and the gap`, norm(curve.rd).includes(day(last[0])) && norm(curve.rd).includes(usd(last[1])) && norm(curve.rd).includes(usd(last[2])) && norm(curve.rd).includes('ahead by ' + pct(sm.vs_index)), norm(curve.rd));
  // a pointer (or a finger) reads any day
  await page.evaluate((y) => window.scrollTo(0, y - 100), curve.box.y);
  const cy = await page.evaluate(() => { const b = document.querySelector('#sec-curve .chart-curve').getBoundingClientRect(); return { x: b.left, y: b.top, w: b.width, h: b.height }; });
  if (size === 'phone') await page.touchscreen.tap(cy.x + 55, cy.y + cy.h / 2); else await page.mouse.move(cy.x + 63, cy.y + cy.h / 2);
  await page.waitForTimeout(150);
  const first = norm(await text(page, '#sec-curve .rd'));
  check(`${tag} 2 a ${size === 'phone' ? 'finger' : 'pointer'} on the curve reads that day (here the first close, ${day(net.curve[0][0])})`, first.includes(day(net.curve[0][0])) && first.includes('model ' + usd(net.curve[0][1])) && first.includes('index ' + usd(net.curve[0][2])), first);

  // the verdict in a sentence, the lamp explained once in a fold
  const verdict = await page.evaluate(() => ({ line: document.querySelector('#sec-verdict .vd-line').textContent, lamp: document.querySelector('#sec-verdict .vd-line .p-blk').className, say: document.querySelector('#sec-verdict .vd-say').textContent, fold: document.querySelector('#sec-verdict details').open }));
  const v = sm.verdict;
  check(`${tag} 2 the verdict: the engine's line, then one plain sentence with the days ahead, the count, the score and the evidence`, norm(verdict.line) === v.text && verdict.lamp === 'p-blk off' && norm(verdict.say).includes(`next ${v.horizon} trading days`) && norm(verdict.say).includes(`${v.n} scored`) && norm(verdict.say).includes((v.ic > 0 ? '+' : '-') + Math.abs(v.ic).toFixed(3)) && norm(verdict.say).includes(v.t.toFixed(2)) && (norm(verdict.say).match(/\./g) || []).length <= 4 && !verdict.fold, norm(verdict.say));
  await tap('#sec-verdict details > summary');
  const lampText = norm(await text(page, '#sec-verdict details .fold-b'));
  check(`${tag} 2 the lamp is explained in a fold: profit needs years, so each evening is scored against what followed`, /Profit alone needs years to tell skill from luck/.test(lampText) && /scored against what the stocks went on to do/.test(lampText) && /skill likely/.test(lampText) && /too early/.test(lampText) && /skill unlikely/.test(lampText), lampText.slice(0, 120));

  // 3. today's ranking and how past rankings held up
  const rank = await page.evaluate(() => {
    const t = [...document.querySelectorAll('#sec-today .tb-rank')].map((tb) => [...tb.querySelectorAll('tbody tr')].map((tr) => ({ rank: tr.children[0].textContent, stock: tr.children[1].textContent, score: tr.children[2].textContent, held: !!tr.children[3].querySelector('.held') })));
    const h = document.querySelector('#sec-history');
    return { tables: t, note: document.querySelector('#sec-today .cap').textContent, heads: [...document.querySelectorAll('#sec-today h4')].map((x) => x.textContent), date: document.querySelector('#sec-today .grp .note').textContent,
      hist: h ? { note: h.querySelector('.grp .note').textContent, cap: h.querySelector('.cap').textContent, cells: h.querySelectorAll('.plot .cells').length, up: (h.querySelector('.cells-up') || { getAttribute: () => '' }).getAttribute('d').split('M').length - 1, rd: h.querySelector('.rd').textContent } : null };
  });
  const sideOk = (list, rows) => list.length === rows.length && list.every((x, i) => norm(rows[i].rank) === fixed(x.rank, 0) && norm(rows[i].stock) === `${x.symbol} ${x.name}` && norm(rows[i].score) === (x.score > 0 ? '+' : x.score < 0 ? '-' : '') + Math.abs(x.score).toFixed(2) && rows[i].held === !!x.held);
  check(`${tag} 3 today's ranking: the top ${net.today.top.length} and the bottom ${net.today.bottom.length} with rank, score and which are held, and the engine's note`, rank.tables.length === 2 && sideOk(net.today.top, rank.tables[0]) && sideOk(net.today.bottom, rank.tables[1]) && rank.note === net.today.note && rank.heads.join() === 'Top,Bottom' && norm(rank.date) === day(net.today.date), JSON.stringify(rank.tables[0] && rank.tables[0][0]));
  const hs = net.ranking_history, upN = hs.filter((x) => x.ic > 0).length, mean = hs.reduce((a, x) => a + x.ic, 0) / hs.length, lastH = hs[hs.length - 1];
  check(`${tag} 3 how past rankings held up: ${hs.length} evenings as columns, ${upN} above the line, the average in words`, rank.hist && norm(rank.hist.note) === `${hs.length} evenings scored` && norm(rank.hist.cap).includes(`${upN} of ${hs.length} were above`) && norm(rank.hist.cap).includes((mean > 0 ? '+' : '-') + Math.abs(mean).toFixed(3)) && rank.hist.cells === 2 && rank.hist.up > 0, JSON.stringify(rank.hist && { note: rank.hist.note, up: rank.hist.up }));
  check(`${tag} 3 under the columns, the last evening in words`, rank.hist && norm(rank.hist.rd).includes(day(lastH.date)) && norm(rank.hist.rd).includes((lastH.ic > 0 ? '+' : '-') + Math.abs(lastH.ic).toFixed(3)) && norm(rank.hist.rd).includes(`${lastH.n} stocks ranked`), rank.hist && norm(rank.hist.rd));
  await page.focus('#sec-history .chart-bars'); await page.keyboard.press('ArrowLeft'); await page.waitForTimeout(100);
  const prev = norm(await text(page, '#sec-history .rd')), before = hs[hs.length - 2];
  check(`${tag} 3 the arrow keys walk the evenings (one back: ${day(before.date)})`, prev.includes(day(before.date)) && prev.includes((before.ic > 0 ? '+' : '-') + Math.abs(before.ic).toFixed(3)), prev);

  // 5. the backtest in a few figures, the exam beside it, the report in full
  const bt = await page.evaluate(() => {
    const t = document.querySelector('#sec-backtest .tb-bt');
    return { heads: [...t.querySelectorAll('thead th')].map((x) => x.textContent), rows: Object.fromEntries([...t.querySelectorAll('tbody tr')].map((tr) => [tr.children[0].textContent, [tr.children[1].textContent, tr.children[2].textContent]])),
      facts: Object.fromEntries([...document.querySelectorAll('#sec-backtest .facts > div')].map((d) => [d.querySelector('dt').textContent, d.querySelector('dd').textContent])),
      years: [...document.querySelectorAll('#sec-backtest .tb-years tbody tr')].map((tr) => [...tr.children].map((td) => td.textContent)), fold: document.querySelector('#sec-backtest details.fold').open, md: document.querySelector('#sec-backtest .md').checkVisibility() };
  });
  const b = sm.backtest, e = b.exam, two = (f) => [f(b), f(e)], is = (label, want) => bt.rows[label] && norm(bt.rows[label][0]).startsWith(want[0]) && norm(bt.rows[label][1]).startsWith(want[1]);
  const btOk = /^Backtest/.test(bt.heads[0]) && /^Exam/.test(bt.heads[1]) && /held back until the design was frozen/.test(bt.heads[1]) &&
    is('Years', ['2 Jan 2020 – 13 May 2025', '14 May 2025 – 13 Nov 2026']) && is('Return a year', two((x) => pct(x.ret_ann))) && is('Index a year', two((x) => pct(x.index_ann))) && is('vs index a year', two((x) => pct(x.vs_index_ann))) &&
    is('Likely range', two((x) => `${pct(x.ci[0])} to ${pct(x.ci[1])}`)) && is('Ranking score', two((x) => (x.ic > 0 ? '+' : '-') + Math.abs(x.ic).toFixed(3))) && is('Evidence', two((x) => x.ic_t.toFixed(1)));
  check(`${tag} 5 the backtest in figures with the exam beside it: years, return, index, vs index, likely range, ranking score, evidence`, btOk, JSON.stringify(bt.rows));
  check(`${tag} 5 and the backtest's own: worst drop, return for its risk, turnover, costs a year`, norm(bt.facts['Worst drop']) === pct(b.max_dd) && norm(bt.facts['Return for its risk']).startsWith(b.sharpe.toFixed(2)) && norm(bt.facts['Turns over']).startsWith(b.turnover.toFixed(1) + '×') && norm(bt.facts['Costs took']).startsWith(pct(b.cost_drag, 2, false)), JSON.stringify(bt.facts));
  const yWrong = net.by_year.filter((y, i) => { const r = bt.years[i] || []; return !(norm(r[0]) === String(y.year) && norm(r[1]) === pct(y.ret) && norm(r[2]) === pct(y.index) && norm(r[3]) === pct(y.ret - y.index)); });
  check(`${tag} 5 year by year against the index (${net.by_year.length} years)`, bt.years.length === net.by_year.length && !yWrong.length, JSON.stringify(yWrong.map((y) => y.year)));
  await tap('#sec-backtest details.fold > summary'); await page.waitForTimeout(100);
  const rep = await page.evaluate(() => { const m = document.querySelector('#sec-backtest .md'); return { open: m.closest('details').open, seen: m.checkVisibility(), h4: [...m.querySelectorAll('h4, h5')].map((x) => x.tagName + ' ' + x.textContent), p: m.querySelectorAll('p').length, th: [...m.querySelectorAll('table th')].map((x) => x.textContent), td: [...m.querySelectorAll('table tbody tr')].map((tr) => [...tr.children].map((x) => x.textContent)), text: m.textContent }; });
  check(`${tag} 5 the full report waits in a fold and opens as text: headings, paragraphs and its table, no markdown signs left`, !bt.fold && bt.md === false && rep.open && rep.seen === true && rep.h4.join('|') === 'H4 Neural network: backtest|H5 What it does not show' && rep.p === 2 && rep.th.join() === 'year,model,index' && rep.td.length === 2 && rep.td[0].join() === '2020,+21.3 %,+18.4 %' && !/[#|*]|---/.test(rep.text), JSON.stringify({ h4: rep.h4, td: rep.td }));
  let all = await visibleText(page);
  check(`${tag} no "NaN", "null" or "undefined" on the network's page`, !BAD_WORDS.test(all), (BAD_WORDS.exec(all) || [])[0]);
  const sw1 = await sideways(page);
  check(`${tag} the network's page does not scroll sideways`, sw1.sw <= sw1.iw && sw1.iw === VIEW[size].viewport.width, JSON.stringify(sw1));

  // a reload keeps the open model; the back button goes back to the list, where it was
  await page.reload(); await bookReady(page);
  check(`${tag} 2 a reload keeps the model open`, (await text(page, '#book-title')) === net.title && (await page.evaluate(() => location.hash)) === '#/net');
  await page.goBack(); await page.waitForFunction(() => document.getElementById('detail').hidden);
  const back = await page.evaluate(() => ({ hash: location.hash, view: document.body.dataset.view, rows: document.querySelectorAll('#board .row').length, y: scrollY }));
  check(`${tag} 2 the back button returns to the list, at the place it was left`, back.hash === '' && back.view === 'board' && back.rows === st.models.length && y0 >= 100 && Math.abs(back.y - y0) <= 2, JSON.stringify({ ...back, left: y0 }));

  // 4. the agent: memo, reads, spend meter with the cap, and the word that it stopped
  const agent = fx('model_agent.json'), am = st.models.find((m) => m.id === 'agent'), sp = st.spend;
  await page.goto(url(fixture, '#/agent')); await bookReady(page); await page.waitForTimeout(300);
  await shot(page, `${size}-${fixture}-agent`);
  const ag = await page.evaluate(() => {
    const md = document.querySelector('#sec-memo .md'), sp = document.querySelector('#sec-spend');
    return { secs: [...document.querySelectorAll('#detail .sec')].map((x) => x.id), memoDate: document.querySelector('#sec-memo .grp .note').textContent, strong: [...md.querySelectorAll('strong')].map((x) => x.textContent), li: md.querySelectorAll('li').length, em: [...md.querySelectorAll('em')].map((x) => x.textContent), memo: md.textContent,
      spend: { fig: sp.querySelector('.spend-fig').textContent, v: sp.querySelector('.p-meter').style.getPropertyValue('--v'), facts: Object.fromEntries([...sp.querySelectorAll('.facts > div')].map((d) => [d.querySelector('dt').textContent, d.querySelector('dd').textContent])), stop: (sp.querySelector('.stop') || {}).textContent || null, month: sp.querySelector('.grp .note').textContent, days: sp.querySelectorAll('.chart-bars').length },
      state: (document.querySelector('.bk-state') || {}).textContent || null, bt: document.querySelector('#sec-backtest').textContent, btTables: document.querySelectorAll('#sec-backtest table').length,
      reads: [...document.querySelectorAll('#sec-reads .readc')].map((c) => ({ shown: !c.hidden, sym: c.querySelector('.sym').textContent, name: c.querySelector('.nm').textContent, thesis: (c.querySelector('.thesis') || {}).textContent, facts: [...c.querySelectorAll('.factl li')].map((x) => x.textContent), text: c.textContent, links: [...c.querySelectorAll('.srcl a')].map((a) => ({ href: a.getAttribute('href'), rel: a.rel, target: a.target, text: a.textContent })) })),
      signals: document.querySelectorAll('#sec-signals .tb-signals tbody tr').length, sigNote: document.querySelector('#sec-signals .grp .note').textContent };
  });
  check(`${tag} 4 the memo is there, dated, and set as text: bold, a list, emphasis; no asterisks left`, norm(ag.memoDate) === day(agent.memo.date) && ag.strong.join('|') === 'Tonight.|Bought VRT (6 %)|Sold DAL|Held the rest.' && ag.li === 3 && ag.em.join() === 'Not acted on:' && !/\*/.test(ag.memo) && ag.memo.includes('212 filings and 1,480 headlines triaged'), JSON.stringify({ strong: ag.strong, li: ag.li }));
  const r0 = agent.reads[0], c0 = ag.reads[0] || {};
  check(`${tag} 4 its reads: ${agent.reads.length} companies, each with thesis, facts, what is in the price, the chances, conviction, risks and sources`, ag.reads.length === agent.reads.length && ag.reads.filter((r) => r.shown).length === 4 && c0.sym === r0.symbol && norm(c0.name) === r0.name && c0.thesis === r0.thesis && c0.facts.join('|') === r0.facts.join('|') &&
    norm(c0.text).includes(r0.priced_in) && norm(c0.text).includes(r0.risks) && norm(c0.text).includes(`in 5 days ${Math.round(r0.p5 * 100)} %`) && norm(c0.text).includes(`in 20 days ${Math.round(r0.p20 * 100)} %`) && norm(c0.text).includes(`conviction ${r0.conviction.toFixed(2)}`), JSON.stringify({ n: ag.reads.length, sym: c0.sym }));
  const l0 = (c0.links || [])[0] || {};
  check(`${tag} 4 a source is a link only when it is http(s), and opens apart with rel="noopener"`, l0.href === r0.sources[0].url && /noopener/.test(l0.rel) && l0.target === '_blank' && l0.text === r0.sources[0].title, JSON.stringify(l0));
  await tap('button.more[data-k="more:reads:agent"]');
  check(`${tag} 4 one button shows all ${agent.reads.length} reads`, (await page.evaluate(() => [...document.querySelectorAll('#sec-reads .readc')].filter((c) => !c.hidden && c.getBoundingClientRect().height > 0).length)) === agent.reads.length);
  const frac = (sp.used / sp.cap * 100).toFixed(1) + '%';
  check(`${tag} 4 the spend meter: ${usd(sp.used)} of the ${usd(sp.cap)} cap for the month, the meter at ${frac}, what is left, today, calls`, norm(ag.spend.fig).startsWith(usd(sp.used)) && norm(ag.spend.fig).includes('of ' + usd(sp.cap)) && ag.spend.v === frac && norm(ag.spend.facts.Left) === usd(sp.cap - sp.used) && norm(ag.spend.facts.Today) === usd(sp.today) && norm(ag.spend.facts.Calls) === fixed(sp.calls, 0) && norm(ag.spend.facts.Cap) === usd(sp.cap) && norm(ag.spend.month) === 'November 2026', JSON.stringify(ag.spend));
  if (sp.stopped) {
    check(`${tag} 4 it says the agent has stopped, and at which limit ("${sp.stopped}")`, /^Stopped at the month's cap\./.test(norm(ag.spend.stop || '')) && norm(ag.state || '').toLowerCase().includes(am.state_text), `${ag.spend.stop} | ${ag.state}`);
  } else {
    check(`${tag} 4 nothing says "stopped" while it has not`, ag.spend.stop == null && ag.state == null && !/stopped/i.test(await visibleText(page)));
  }
  check(`${tag} 4/5 where the backtest would be, one line says why the agent has none`, /No backtest: the model has already read the past, so only live paper counts\./.test(ag.bt) && ag.btTables === 0, ag.bt);
  const sg = agent.signals, settled = sg.filter((x) => typeof x.excess === 'number');
  check(`${tag} 4 its forecasts (${sg.length}), with how many are settled and how many of those beat the index`, ag.signals === sg.length && norm(ag.sigNote) === `${sg.length} forecasts, ${settled.length} settled, ${settled.filter((x) => x.excess > 0).length} beat the index`, ag.sigNote);
  check(`${tag} 3 the agent ranks no stocks, so its page has no ranking and no empty place for one`, !ag.secs.includes('sec-today') && !ag.secs.includes('sec-history'), ag.secs.join());
  all = await visibleText(page);
  check(`${tag} no "NaN", "null" or "undefined" on the agent's page`, !BAD_WORDS.test(all), (BAD_WORDS.exec(all) || [])[0]);
  const sw2 = await sideways(page);
  check(`${tag} the agent's page does not scroll sideways`, sw2.sw <= sw2.iw && sw2.iw === VIEW[size].viewport.width, JSON.stringify(sw2));

  // every other model opens clean too
  const bad = [];
  for (const m of st.models) {
    if (m.id === 'net' || m.id === 'agent') continue;
    const d = fx(`model_${m.id}.json`);
    await page.goto(url(fixture, '#/' + m.id)); await bookReady(page);
    const t = await visibleText(page), s = await sideways(page);
    const got = await page.evaluate(() => ({ title: document.getElementById('book-title').textContent, secs: [...document.querySelectorAll('#detail .sec')].map((x) => x.id), pos: document.querySelectorAll('.tb-positions tbody tr').length, tr: document.querySelectorAll('.tb-trades tbody tr').length, sig: document.querySelectorAll('.tb-signals tbody tr').length, hollow: [...document.querySelectorAll('#detail table')].filter((x) => !x.querySelector('tbody tr')).length, state: (document.querySelector('.bk-state') || {}).textContent || null, vd: (document.querySelector('#sec-verdict .vd-line') || {}).textContent }));
    const want = ['sec-curve', 'sec-verdict', 'sec-pending', ...(d.today ? ['sec-today'] : []), ...(d.ranking_history.length ? ['sec-history'] : []), 'sec-positions', 'sec-trades', ...(d.signals.length ? ['sec-signals'] : []), 'sec-backtest'];
    if (BAD_WORDS.test(t)) bad.push(`${m.id}: "${BAD_WORDS.exec(t)[0]}"`);
    if (s.sw > s.iw) bad.push(`${m.id}: scrolls sideways`);
    if (got.title !== m.title || got.secs.join() !== want.join() || got.pos !== d.positions.length || got.tr !== d.trades.length || got.sig !== d.signals.length || got.hollow) bad.push(`${m.id}: ${JSON.stringify(got)}`);
    if (m.state !== 'running' && !norm(got.state || '').toLowerCase().includes(m.state_text.toLowerCase())) bad.push(`${m.id}: does not say why it is not running`);
    if (!m.verdict && !/Not scored/.test(got.vd || '')) bad.push(`${m.id}: no verdict, and the page does not say so`);
  }
  check(`${tag} 2 3 5 every other model opens with the blocks its data has and no others (${st.models.length - 2} pages)`, !bad.length, bad.join('; '));
  await page.goto(url(fixture, '#/drift')); await bookReady(page); await shot(page, `${size}-${fixture}-drift`);

  // an address for a model that is not there
  await page.goto(url(fixture, '#/no-such-model')); await page.waitForFunction(() => /No such model/.test(document.getElementById('detail').textContent), null, { timeout: 8000 }).catch(() => {});
  const gone = await page.evaluate(() => ({ t: document.getElementById('detail').textContent, back: !!document.querySelector('#detail a.back[href="#/"]') }));
  check(`${tag} 2 an address for a model that does not exist says so and offers the way back`, /No such model/.test(gone.t) && gone.t.includes('no-such-model') && gone.back, gone.t.slice(0, 120));
  check(`${tag} books: no script error; the only failed request is the one for the model that is not there`, !seen.errors.filter((x) => !/404/.test(x)).length && seen.bad.every((x) => /no-such-model/.test(x)), [...seen.errors, ...seen.bad].join(' | '));
  await ctx.close();
}

// ── the keyboard, on a PC ───────────────────────────────────────────────────────────────────────────────
async function keyboard(browser) {
  const tag = '[pc running]';
  const { ctx, page } = await open(browser, 'pc');
  await page.goto(url('running')); await boardReady(page);
  let reached = null, tabs = 0;
  for (; tabs < 12 && !reached; tabs++) { await page.keyboard.press('Tab'); reached = await page.evaluate(() => { const a = document.activeElement; return a && a.dataset.k && a.dataset.k.startsWith('row:') ? a.dataset.k : null; }); }
  const ring = await page.evaluate(() => { const s = getComputedStyle(document.activeElement); return { style: s.outlineStyle, w: parseFloat(s.outlineWidth), color: s.outlineColor }; });
  check(`${tag} keyboard: Tab reaches the first model within a few presses, and the focused row is outlined`, reached === 'row:index' && tabs <= 8 && ring.style === 'solid' && ring.w >= 1 && ring.color === 'rgb(228, 248, 234)', `${reached} after ${tabs}; ${JSON.stringify(ring)}`);
  await page.keyboard.press('Tab'); await page.keyboard.press('Enter'); await bookReady(page);
  const o = await page.evaluate(() => ({ hash: location.hash, focus: document.activeElement.id }));
  check(`${tag} keyboard: Enter opens the model and puts the reader at its title`, o.hash === '#/momentum' && o.focus === 'book-title', JSON.stringify(o));
  const stops = [];
  for (let i = 0; i < 12; i++) { await page.keyboard.press('Tab'); const k = await page.evaluate(() => { const a = document.activeElement, s = getComputedStyle(a); return a.dataset.k ? { k: a.dataset.k, ring: s.outlineStyle === 'solid' && parseFloat(s.outlineWidth) >= 1 } : null; }); if (k) stops.push(k); if (k && k.k.startsWith('more:trades')) break; }
  const kinds = stops.map((s) => s.k.replace(/:momentum$/, ''));
  check(`${tag} keyboard: from there Tab walks the page in order: the curve, the lamp's fold, the past rankings, show-all; each outlined when focused`, kinds.join(' ') === 'chart:curve fold:lamp-momentum chart:history more:trades' && stops.every((s) => s.ring), kinds.join(' '));
  await page.keyboard.press('Escape'); await page.waitForFunction(() => document.getElementById('detail').hidden);
  const c = await page.evaluate(() => ({ hash: location.hash, focus: document.activeElement.dataset.k, view: document.body.dataset.view }));
  check(`${tag} keyboard: Escape closes it and focus goes back to its row`, c.view === 'board' && c.focus === 'row:momentum' && (c.hash === '#/' || c.hash === ''), JSON.stringify(c));
  // the rail: another model in one click, no way back needed
  await page.click('#board a.row-a[data-k="row:net"]'); await bookReady(page);
  await page.click('#board a.row-a[data-k="row:trees"]'); await page.waitForFunction(() => document.getElementById('book-title').textContent === 'Boosted trees');
  const rail = await page.evaluate(() => ({ cur: [...document.querySelectorAll('#board [aria-current="true"]')].map((a) => a.dataset.k), bg: getComputedStyle(document.querySelector('#board [aria-current="true"]')).backgroundColor, w: document.getElementById('over').getBoundingClientRect().width, left: document.getElementById('detail').getBoundingClientRect().left }));
  check(`${tag} PC: with a model open the list is a rail beside it; one click there opens another, shown in inverse video`, rail.cur.join() === 'row:trees' && rail.bg === 'rgb(228, 248, 234)' && rail.w < 360 && rail.left > 300, JSON.stringify(rail));
  await ctx.close();
}

// ── the minute's refresh, and a fetch that fails ────────────────────────────────────────────────────────
async function refresh(browser, size) {
  const tag = `[${size} running]`;
  const { ctx, page, seen } = await open(browser, size, { failing: true });
  let calls = 0, books = 0, mode = 'pass', bump = false;
  await page.clock.install({ time: new Date('2026-11-13T22:45:00+01:00') });
  await page.addInitScript(() => {                        // a way to make the page itself trip: a model called "boom"
    const real = Element.prototype.setAttribute;
    Element.prototype.setAttribute = function (k, v) { if (k === 'data-id' && v === 'boom') throw new Error('boom went the page'); return real.call(this, k, v); };
  });
  await page.route('**/api/state', async (route) => {
    calls++;
    if (mode === 'fail') return route.abort();
    if (mode === 'junk') return route.fulfill({ status: 200, contentType: 'text/html', body: '<html>502 Bad Gateway</html>' });
    if (mode === 'empty') return route.fulfill({ json: { app: 'stocks' } });
    const r = await route.fetch(), j = await r.json();
    if (bump) { const m = j.models.find((x) => x.id === 'net'); m.ret = 0.1234; m.vs_index = 0.1111; j.generated = '2026-11-13T23:07:00+01:00'; }
    if (mode === 'boom') j.models.push({ id: 'boom', kind: 'rule', title: 'Boom', state: 'running', ret: 0.01 });
    await route.fulfill({ response: r, json: j });
  });
  await page.route('**/api/model/net', async (route) => { books++; if (mode === 'fail') return route.abort(); await route.continue(); });
  await page.goto(url('running', '#/net')); await bookReady(page);
  await page.click('#sec-backtest details.fold > summary');
  await page.click('button.more[data-k="more:trades:net"]');
  await page.evaluate(() => window.scrollTo(0, 1500)); await page.waitForTimeout(200);
  const snap = () => page.evaluate(() => ({ hash: location.hash, title: document.getElementById('book-title').textContent, y: scrollY, fold: document.querySelector('#sec-backtest details.fold').open, trades: [...document.querySelectorAll('.tb-trades tbody tr')].filter((t) => !t.hidden).length, vs: document.querySelector('.figs .hero .p-num').textContent, stale: (document.querySelector('#status .st-fault') || {}).textContent || null, read: document.querySelector('#status .st-line').textContent, book: (document.querySelector('#detail .st-fault') || {}).textContent || null, rows: document.querySelectorAll('#board .row').length }));
  const a = await snap(), c0 = calls;
  await page.clock.fastForward(60500); await until(() => calls > c0); await page.waitForTimeout(400);
  const b = await snap();
  check(`${tag} after a minute the page reads /api/state again (and the open model's own figures)`, calls === c0 + 1 && books >= 2, `${c0} -> ${calls} state reads, ${books} of the model`);
  check(`${tag} the refresh leaves the open model, the scroll position, the open fold and the full list as they were`, b.hash === '#/net' && b.title === a.title && b.y === a.y && b.y === 1500 && b.fold && b.trades === a.trades && a.trades > 12, JSON.stringify(b));
  bump = true; const c1 = calls;
  await page.clock.fastForward(60500); await until(() => calls > c1); await page.waitForFunction(() => /11\.11/.test(document.querySelector('.figs .hero .p-num').textContent), null, { timeout: 5000 }).catch(() => {});
  const c = await snap();
  check(`${tag} new figures arrive in place: the page shows them, and still nothing has moved`, norm(c.vs) === '+11.11%' && /23:07/.test(c.read) && c.hash === '#/net' && Math.abs(c.y - 1500) <= 2 && c.fold && c.trades === a.trades, JSON.stringify(c));
  mode = 'fail'; const c2 = calls;
  await page.clock.fastForward(60500); await until(() => calls > c2); await page.waitForSelector('#status .st-fault', { timeout: 5000 }).catch(() => {});
  const d = await snap();
  check(`${tag} when the engine cannot be reached the page says "Stale", says since when, and keeps the last good figures`, /^Stale\./.test(norm(d.stale || '')) && /Can't reach the engine since \d\d:\d\d/.test(d.stale) && /23:07/.test(d.stale) && norm(d.vs) === '+11.11%' && d.title === a.title && Math.abs(d.y - 1500) <= 2, d.stale);
  check(`${tag} the open model says the same of its own figures, and keeps them`, /^Stale\./.test(norm(d.book || '')) && d.trades === a.trades, d.book);
  mode = 'pass'; const c3 = calls;
  await page.clock.fastForward(60500); await until(() => calls > c3); await page.waitForFunction(() => !document.querySelector('#status .st-fault'), null, { timeout: 5000 }).catch(() => {});
  const e = await snap();
  check(`${tag} and takes the word back once the engine answers again`, e.stale == null && e.book == null && norm(e.vs) === '+11.11%', `${e.stale} ${e.book}`);
  // an answer that is not the engine's (a proxy's error page, a JSON without models) is stale too, and does not stop the reading
  for (const junk of ['junk', 'empty']) {
    mode = junk; const cj = calls;
    await page.clock.fastForward(60500); await until(() => calls > cj); await page.waitForSelector('#status .st-fault', { timeout: 5000 }).catch(() => {});
    const j1 = await snap();
    mode = 'pass'; const ck = calls;
    await page.clock.fastForward(60500); await until(() => calls > ck); await page.waitForFunction(() => !document.querySelector('#status .st-fault'), null, { timeout: 5000 }).catch(() => {});
    const j2 = await snap();
    check(`${tag} an answer that is not a state (${junk === 'junk' ? "a proxy's error page" : 'JSON without models'}) counts as stale, keeps the figures, and the next good one clears it`, /^Stale\./.test(norm(j1.stale || '')) && norm(j1.vs) === '+11.11%' && j1.rows === 11 && j2.stale == null && calls === ck + 1, `${j1.stale} -> ${j2.stale}`);
  }
  // the page tripping over what it was sent is said as the page's fault, and the reading goes on
  mode = 'boom'; const cb = calls;
  await page.clock.fastForward(60500); await until(() => calls > cb); await page.waitForSelector('#status .st-fault', { timeout: 5000 }).catch(() => {});
  const f1 = await snap();
  mode = 'pass'; const cc = calls;
  await page.clock.fastForward(60500); await until(() => calls > cc); await page.waitForFunction(() => !document.querySelector('#status .st-fault'), null, { timeout: 5000 }).catch(() => {});
  const f2 = await snap();
  check(`${tag} if the page itself cannot draw what it was sent it says "the page has a fault", not "stale", and mends itself on the next good read`, /^The page has a fault\./.test(norm(f1.stale || '')) && /boom went the page/.test(f1.stale) && !/Stale/.test(f1.stale) && f2.stale == null && f2.rows === 11 && norm(f2.vs) === '+11.11%', `${f1.stale} -> ${f2.stale}`);
  // coming back to the tab reads at once
  const cv = calls;
  await page.clock.fastForward(20000);
  await page.evaluate(() => document.dispatchEvent(new Event('visibilitychange')));
  check(`${tag} coming back to the tab reads again at once, not a minute later`, await until(() => calls > cv, 3000), `${cv} -> ${calls}`);
  check(`${tag} refresh: no script error on the way (bar the one that was provoked)`, seen.errors.every((x) => /boom went the page/.test(x)), seen.errors.join(' | '));
  await ctx.close();

  // nothing read at all yet
  const f = await open(browser, size, { failing: true });
  let dead = true;
  await f.page.clock.install({ time: new Date('2026-11-13T22:45:00+01:00') });
  await f.page.route('**/api/state', (route) => (dead ? route.abort() : route.continue()));
  await f.page.goto(url('running')); await f.page.waitForSelector('#status .st-fault', { timeout: 8000 }).catch(() => {});
  const g = await f.page.evaluate(() => ({ s: document.getElementById('status').textContent, rows: document.querySelectorAll('#board .row').length, paper: document.getElementById('mode').textContent }));
  check(`${tag} if the engine cannot be reached on the first load the page says so (and still says PAPER)`, /Can't reach the engine\./.test(g.s) && g.rows === 0 && norm(g.paper) === 'Paper' && !BAD_WORDS.test(await visibleText(f.page)), g.s);
  dead = false; await f.page.clock.fastForward(10500);
  check(`${tag} and it fills in by itself when the engine is back`, await f.page.waitForSelector('#board .rows > li', { timeout: 8000 }).then(() => true, () => false));
  await f.ctx.close();
}

// ── text from the API is never markup; the worst data does not break the page ───────────────────────────
const HTML = '<img src=x onerror="window.__xss=1">';
function hostile(j) {
  const m = j.models.find((x) => x.id === 'net'), a = j.models.find((x) => x.id === 'agent'), w = j.models.find((x) => x.id === 'reversal'), t = j.models.find((x) => x.id === 'trend'), bl = j.models.find((x) => x.id === 'blend');
  m.title = HTML; m.what = '<script>window.__xss=2</script>what'; m.verdict.text = '<b onmouseover="window.__xss=3">verdict</b>'; m.verdict.lamp = 'likely';
  a.title = 'Averyveryveryveryveryveryveryveryveryveryveryveryveryveryveryveryverylongnamewithnospacesatall'; bl.kind = '<i>kind</i>'; a.state = 'off'; a.state_text = '<u>because</u>';
  w.verdict.lamp = 'unlikely'; w.ret = -5.5; w.vs_index = -1234567.891; w.costs = 1e12; w.trades = 123456789; w.max_dd = -0.999999; w.day_ret = null; w.since = 'not a date';
  t.id = 'odd id/"<>&?#'; t.title = 'Ünïcødé 模型 🙂 مرحبا'; delete t.spark; t.verdict = { lamp: 'constructor' };
  j.phase_text = '<svg onload="window.__xss=4">phase'; j.notices = [{ level: 'fault', text: '<iframe src="https://evil.example/"></iframe> `<b>code</b>`' }, { level: '__proto__', text: 'odd level' }, { level: 'warn' }];
  j.keys = { '<a href=x>k</a>': false, constructor: false, alpaca: true }; j.clock.busy = '<marquee>busy</marquee>'; j.index.title = '<h1>index</h1>';
  j.log = [{ t: 'never', level: 'warn', text: '<script>window.__xss=5</script>log' }, null, 7]; j.spend.stopped = '<b>cap</b>'; j.unknown_field = { a: 1 };
  j.models.push({ id: 'bare' }, null, 'text', { id: 'zero', kind: 'rule', title: 'Zero', state: 'running', ret: 0, vs_index: 0, index_ret: 0, day_ret: -0.000001, max_dd: 0, trades: 0, costs: 0, since: '2025-12-31', spark: [1], spark_index: [] });
  return j;
}
function hostileBook(j) {
  j.memo.text = 'Hello ' + HTML + ' **bold** [bad](javascript:window.__xss=6) [ok](https://example.org/a?b=1&c=2) ![pic](https://evil.example/p.png) <script>window.__xss=7</script>\n\n<iframe src="https://evil.example/"></iframe>\n\n| a | b |\n|---|---|\n| <b>x</b> | [j](data:text/html,<script>window.__xss=8</script>) |\n\n1. one\n2. two\n   - deep `code <b>`\n\n> quoted **text**\n\n```\n<script>window.__xss=9</script>\n```\n';
  j.reads[0].sources = [{ title: 'js', url: 'javascript:window.__xss=10' }, { title: '<b>t</b>', url: 'https://example.org/' }, { title: 'data', url: 'data:text/html,<script>window.__xss=11</script>' }, { url: 'ftp://example.org/x' }, null];
  j.reads[0].thesis = '<style>*{display:none}</style>thesis'; j.reads[0].facts = [HTML, 5, null]; j.reads[0].symbol = '<S>'; j.positions[0].why = '<style>body{display:none}</style>why'; j.positions[0].symbol = 'SUPERCALIFRAGILISTICEXPIALIDOCIOUSSYMBOL';
  j.positions[0].value = 1e15; j.positions[0].weight = 'x'; j.trades[0].price = null; j.trades[0].why = HTML; j.pending = [{ symbol: HTML, side: '<b>', weight: -1 }, null]; j.curve[3] = ['bad', 'x', null]; j.curve[4][2] = null;
  j.signals[0].note = HTML; j.signals[0].p = 7; j.memo.date = null;
  return j;
}
async function hostileRun(browser, size) {
  const tag = `[${size} hostile]`;
  const { ctx, page, seen } = await open(browser, size, { failing: true });
  await page.route('**/api/state', async (route) => { const r = await route.fetch(); await route.fulfill({ response: r, json: hostile(await r.json()) }); });
  await page.route('**/api/model/agent', async (route) => { const r = await route.fetch(); await route.fulfill({ response: r, json: hostileBook(await r.json()) }); });
  await page.route('**/api/model/odd*', async (route) => { await route.fulfill({ json: { ...fx('model_trend.json'), id: 'odd id/"<>&?#' } }); });
  const probe = () => page.evaluate(() => ({ xss: window.__xss, tags: [...document.querySelectorAll('#main img, #main script, #main iframe, #main style, #main marquee, #main object, #main embed, #main u, #main h1, #main svg[onload], header img, footer img')].map((e) => e.tagName),
    on: [...document.querySelectorAll('body *')].filter((e) => [...e.attributes].some((a) => /^on/i.test(a.name))).length, hrefs: [...document.querySelectorAll('#main a[href]')].map((a) => a.getAttribute('href')).filter((h) => !/^#\/|^https?:\/\//.test(h)), sw: document.documentElement.scrollWidth, iw: innerWidth }));
  await page.goto(url('running')); await boardReady(page); await page.waitForTimeout(400);
  await shot(page, `${size}-hostile-board`);
  let p = await probe(), t = await visibleText(page);
  check(`${tag} markup in the API's text is shown as the text it is: nothing runs, no element is made of it`, p.xss === undefined && !p.tags.length && p.on === 0 && !p.hrefs.length && t.includes(HTML.toUpperCase()) && t.toLowerCase().includes('<svg onload="window.__xss=4">phase') && t.toLowerCase().includes('<i>kind</i>') && t.toLowerCase().includes('<u>because</u>'), JSON.stringify(p));
  check(`${tag} the worst data (a 90-letter name, absurd figures, missing fields, rows that are not models) does not push the page sideways`, p.sw <= p.iw && p.iw === VIEW[size].viewport.width, `${p.sw} of ${p.iw} ${await wide(page, VIEW[size].viewport.width)}`);
  check(`${tag} and still no "NaN", "null" or "undefined"`, !BAD_WORDS.test(t), (BAD_WORDS.exec(t) || [])[0]);
  const lamps = await page.evaluate(() => Object.fromEntries([...document.querySelectorAll('#board .rows > li.row')].map((li) => [li.dataset.id, [(li.querySelector('.c-lamp') || {}).className, (li.querySelector('.c-lamp .p-blk') || {}).className, (li.querySelector('.c-vs .v') || {}).textContent]])));
  check(`${tag} lamps: lit for "likely", struck through for "unlikely", hollow for anything else`, /is-likely/.test(lamps.net[0]) && lamps.net[1] === 'p-blk' && /is-unlikely/.test(lamps.reversal[0]) && lamps.reversal[1] === 'p-blk no' && lamps['odd id/"<>&?#'][1] === 'p-blk off', JSON.stringify([lamps.net, lamps.reversal]));
  check(`${tag} a figure too long for its cell is shortened, not spilled`, norm(lamps.reversal[2]) === '-123.5M %', lamps.reversal[2]);
  // a model whose id is anything at all still opens
  await (size === 'phone' ? page.tap('#board li[data-id^="odd id"] a.row-a') : page.click('#board li[data-id^="odd id"] a.row-a'));
  await bookReady(page); await page.reload(); await bookReady(page);
  const odd = await page.evaluate(() => ({ hash: location.hash, title: document.getElementById('book-title').textContent }));
  check(`${tag} a model whose id is full of signs still opens, and its address holds after a reload`, odd.hash === '#/' + encodeURIComponent('odd id/"<>&?#') && odd.title === 'Ünïcødé 模型 🙂 مرحبا', JSON.stringify(odd));
  const emo = await page.evaluate(() => [...document.querySelectorAll('#book-title .emo')].map((e) => [e.textContent, getComputedStyle(e).filter]));
  check(`${tag} a picture character in the API's text is drained of its colour (no hue, even there)`, emo.length === 1 && emo[0][0] === '🙂' && emo[0][1] === 'grayscale(1)', JSON.stringify(emo));
  await page.goto(url('running', '#/agent')); await bookReady(page); await page.waitForTimeout(300);
  await shot(page, `${size}-hostile-agent`);
  p = await probe(); t = await visibleText(page);
  const md = await page.evaluate(() => { const m = document.querySelector('#sec-memo .md'); return { links: [...m.querySelectorAll('a')].map((a) => [a.getAttribute('href'), a.rel, a.target, a.textContent]), strong: m.querySelectorAll('strong').length, td: m.querySelectorAll('td').length, ol: m.querySelectorAll('ol > li').length, deep: m.querySelectorAll('ol ul li code').length, quote: m.querySelectorAll('blockquote strong').length, pre: (m.querySelector('pre') || {}).textContent, text: m.textContent,
    src: [...document.querySelectorAll('#sec-reads .readc')][0].querySelector('.srcl').innerHTML.includes('<a') ? [...[...document.querySelectorAll('#sec-reads .readc')][0].querySelectorAll('.srcl a')].map((a) => a.getAttribute('href')) : [] }; });
  check(`${tag} the memo's markdown: the http(s) link is a link (rel=noopener, new tab), "javascript:" and "data:" are plain words, a picture is only its words`, md.links.length === 1 && md.links[0][0] === 'https://example.org/a?b=1&c=2' && /noopener/.test(md.links[0][1]) && md.links[0][2] === '_blank' && md.links[0][3] === 'ok' && md.text.includes('bad') && md.text.includes('pic') && md.src.join() === 'https://example.org/', JSON.stringify([md.links, md.src]));
  check(`${tag} and its blocks are built: bold, a table, a numbered list with a list inside, a quote, a code block, with their markup inert`, md.strong >= 1 && md.td === 2 && md.ol === 2 && md.deep === 1 && md.quote === 1 && /<script>window\.__xss=9<\/script>/.test(md.pre || '') && p.xss === undefined && !p.tags.length && p.on === 0 && !p.hrefs.length, JSON.stringify({ td: md.td, ol: md.ol, deep: md.deep, quote: md.quote, tags: p.tags, xss: p.xss }));
  check(`${tag} the agent's page with the worst data: not sideways, no "NaN", "null" or "undefined"`, p.sw <= p.iw && p.iw === VIEW[size].viewport.width && !BAD_WORDS.test(t), `${p.sw} of ${p.iw} ${(BAD_WORDS.exec(t) || [])[0] || ''} ${await wide(page, VIEW[size].viewport.width)}`);
  const away = seen.requests.filter((u) => !u.startsWith(BASE + '/') && !u.startsWith('data:'));
  check(`${tag} nothing was asked of any other host (the picture and the frames were never fetched), and no script error`, !away.length && !seen.errors.length, [...away, ...seen.errors].join(' '));
  await ctx.close();
}

// ── the look: Plint Terminal as the theme defines it ────────────────────────────────────────────────────
async function look(browser, size) {
  const tag = `[${size} look]`;
  const { ctx, page } = await open(browser, size, { motion: true });
  await page.goto(url('stopped')); await boardReady(page);
  await page.waitForTimeout(2600);                       // the lights have struck on by now
  const scan = () => page.evaluate(() => {
    const INK = '228,248,234', BG = '15,14,17', bad = [], dark = [], moves = [];
    const cols = (v) => (String(v).match(/rgba?\([^)]+\)/g) || []).map((c) => { const p = c.replace(/rgba?\(|\)/g, '').split(/[\s,\/]+/).filter(Boolean).map(Number); return { rgb: p.slice(0, 3).join(','), a: p.length > 3 ? p[3] : 1 }; });
    const ok = (c, shade) => c.a === 0 || c.rgb === INK || c.rgb === BG || (shade && c.rgb === '0,0,0');
    const name = (e) => e.tagName.toLowerCase() + (e.id ? '#' + e.id : '') + (e.getAttribute('class') ? '.' + e.getAttribute('class').split(/\s+/).join('.') : '');
    for (const e of document.querySelectorAll('body, body *')) {
      const s = getComputedStyle(e), svg = e instanceof SVGElement;
      if (s.display === 'none') continue;
      const paints = svg ? (/^(path|rect|polyline|polygon|circle|text)$/.test(e.tagName) ? [['fill', s.fill], ['stroke', s.stroke]] : e.tagName === 'line' ? [['stroke', s.stroke]] : [])
        : [['color', s.color], ['background', s.backgroundColor], ['backgroundImage', s.backgroundImage], ...(parseFloat(s.borderTopWidth) ? [['border', s.borderTopColor]] : []), ...(parseFloat(s.borderBottomWidth) ? [['border', s.borderBottomColor]] : []), ...(parseFloat(s.borderLeftWidth) ? [['border', s.borderLeftColor]] : []), ['outline', s.outlineStyle !== 'none' ? s.outlineColor : '']];
      for (const [k, v] of paints) for (const c of cols(v)) if (!ok(c, false)) bad.push(`${name(e)} ${k} ${c.rgb}`);
      for (const [k, v] of [['boxShadow', s.boxShadow], ['textShadow', s.textShadow], ['filter', s.filter]]) for (const c of cols(v)) if (!ok(c, true)) bad.push(`${name(e)} ${k} ${c.rgb}`);
      if (s.animationName !== 'none' && !/^steps\(/.test(s.animationTimingFunction)) moves.push(`${name(e)} animation ${s.animationName} ${s.animationTimingFunction}`);
      if (s.transitionDuration.split(',').some((d) => parseFloat(d) > 0) && !/^steps\(/.test(s.transitionTimingFunction)) moves.push(`${name(e)} transition ${s.transitionDuration} ${s.transitionTimingFunction}`);
      if (!svg && e.closest('#main') && parseFloat(s.opacity) === 0 && s.animationName === 'none') dark.push(name(e));
    }
    return { bad: [...new Set(bad)].slice(0, 8), moves: [...new Set(moves)].slice(0, 8), dark: dark.slice(0, 8) };
  });
  const cs = (sel, props) => page.evaluate(([s, ps]) => { const e = document.querySelector(s); if (!e) return null; const c = getComputedStyle(e); return Object.fromEntries(ps.map((p) => [p, c[p]])); }, [sel, props]);
  let s = await scan();
  check(`${tag} one light on one ground: every colour on the overview is the ink (228,248,234) or the ground (15,14,17), at whatever strength`, !s.bad.length, s.bad.join(' | '));
  check(`${tag} motion is stepped: every animation runs in steps, nothing eases, and nothing is left dark after the lights strike on`, !s.moves.length && !s.dark.length, [...s.moves, ...s.dark].join(' | '));
  const fonts = await page.evaluate(async () => { await document.fonts.ready; return { body: getComputedStyle(document.body).fontFamily, num: getComputedStyle(document.querySelector('.tile .p-num')).fontFamily, gm: document.fonts.check('400 12px "Geist Mono"'), doto: document.fonts.check('900 36px "Doto"'), bg: getComputedStyle(document.body).backgroundColor, ink: getComputedStyle(document.body).color }; });
  check(`${tag} the faces: Geist Mono for text, Doto dot-matrix for the big figures, both loaded from the theme`, /^"Geist Mono"/.test(fonts.body) && /^"?Doto/.test(fonts.num) && fonts.gm && fonts.doto && fonts.bg === 'rgb(15, 14, 17)' && fonts.ink === 'rgb(228, 248, 234)', JSON.stringify(fonts));
  const neg = await cs('#board .row .n.neg', ['backgroundColor', 'color']), pos = await cs('#board .row:not(.is-index) .n.pos', ['backgroundColor', 'color', 'fontWeight']);
  const negText = await text(page, '#board .row .n.neg'), posText = await text(page, '#board .row:not(.is-index) .n.pos');
  check(`${tag} money: a loss is the figure in inverse video with its minus, a win is lit ink with its plus`, neg.backgroundColor === 'rgb(228, 248, 234)' && neg.color === 'rgb(15, 14, 17)' && /^\u2212/.test(negText) && pos.backgroundColor === 'rgba(0, 0, 0, 0)' && pos.color === 'rgb(228, 248, 234)' && /^\+/.test(posText), JSON.stringify([neg, negText, pos, posText]));
  const cut = await page.evaluate(() => ['#mode', '#status .lv-fault', '.tile-spend .p-num'].map((s) => getComputedStyle(document.querySelector(s)).clipPath.slice(0, 7)));
  const fault = await cs('#status .lv-fault', ['backgroundColor', 'color']), alarm = await cs('#status .lv-fault .p-blk', ['animationName', 'animationTimingFunction', 'animationIterationCount']);
  check(`${tag} blocks are chamfered; a fault is inverse video and its lamp blinks`, cut.every((c) => c === 'polygon') && fault.backgroundColor === 'rgb(228, 248, 234)' && fault.color === 'rgb(15, 14, 17)' && alarm.animationName === 'p-blink' && /^steps\(1/.test(alarm.animationTimingFunction) && alarm.animationIterationCount === 'infinite', JSON.stringify([cut, fault, alarm]));
  const hollow = await cs('#board .c-lamp .p-blk.off', ['backgroundColor', 'boxShadow', 'animationName']), lit = await cs('#status .lv-warn .p-blk', ['backgroundColor', 'animationName']);
  check(`${tag} lamps: hollow is off (an outline, no fill), lit is filled and steady`, hollow.backgroundColor === 'rgba(0, 0, 0, 0)' && /inset/.test(hollow.boxShadow) && hollow.animationName === 'none' && lit.backgroundColor === 'rgb(228, 248, 234)' && lit.animationName === 'none', JSON.stringify([hollow, lit]));
  const paper = await cs('#mode .p-blk', ['backgroundColor']);
  check(`${tag} the PAPER badge is an outlined block with a hollow lamp: nothing is at stake`, paper.backgroundColor === 'rgba(0, 0, 0, 0)' && (await cs('#mode', ['textTransform'])).textTransform === 'uppercase', JSON.stringify(paper));
  await page.goto(url('stopped', '#/net')); await bookReady(page); await page.waitForTimeout(2600);
  s = await scan();
  check(`${tag} the same on a model's page, charts included: no hue, stepped motion, nothing left dark`, !s.bad.length && !s.moves.length && !s.dark.length, [...s.bad, ...s.moves, ...s.dark].join(' | '));
  const hero = await cs('.figs .hero .p-num', ['fontFamily', 'fontSize']);
  check(`${tag} the model's leading figure is in the dot-matrix face, large`, /^"?Doto/.test(hero.fontFamily) && parseFloat(hero.fontSize) >= 48, JSON.stringify(hero));
  const titles = await page.evaluate(() => document.querySelectorAll('#main [title], header [title]:not(#plint-pill *), footer [title]').length);
  check(`${tag} nothing is said only in a hover tip`, titles === 0, titles);
  await ctx.close();
}

// ── a finger: 44 px targets, safe areas ─────────────────────────────────────────────────────────────────
async function touch(browser) {
  const tag = '[phone touch]';
  const { ctx, page } = await open(browser, 'phone');
  const small = async () => page.evaluate(() => {
    const out = [];
    for (const e of document.querySelectorAll('a[href], button, summary, [tabindex="0"]')) {
      const r = e.getBoundingClientRect(), s = getComputedStyle(e);
      if (s.display === 'none' || s.visibility === 'hidden' || !r.width || !r.height || e.closest('[hidden]') || (e.closest('details:not([open])') && e.tagName !== 'SUMMARY')) continue;
      if (r.height < 43.5 || r.width < 43.5) out.push(`${e.tagName.toLowerCase()}.${e.className} "${e.textContent.trim().slice(0, 24)}" ${Math.round(r.width)}x${Math.round(r.height)}`);
    }
    return { out, coarse: matchMedia('(pointer: coarse)').matches, n: document.querySelectorAll('a[href], button, summary, [tabindex="0"]').length };
  });
  await page.goto(url('stopped')); await boardReady(page);
  await page.tap('#log details > summary'); await page.tap('#board details > summary');
  let a = await small();
  check(`${tag} the test runs with a coarse pointer, as a phone has`, a.coarse === true);
  check(`${tag} on the overview everything that can be tapped is at least 44 px each way (${a.n} things)`, !a.out.length && a.n > 12, a.out.join(' | '));
  for (const id of ['net', 'agent']) {
    await page.goto(url('stopped', '#/' + id)); await bookReady(page);
    for (const sum of await page.$$('#detail details > summary')) await sum.tap();
    for (const btn of await page.$$('#detail button.more')) await btn.tap();
    a = await small();
    check(`${tag} on the ${id === 'net' ? "network's" : "agent's"} page too, links in the text included (${a.n} things)`, !a.out.length && a.n > 8, a.out.slice(0, 6).join(' | '));
  }
  const meta = await page.evaluate(() => ({ vp: document.querySelector('meta[name=viewport]').content, pad: getComputedStyle(document.body).paddingLeft, css: [...document.styleSheets].filter((s) => /app\.css/.test(s.href || '')).map((s) => [...s.cssRules].map((r) => r.cssText).join(' ')).join(' ') }));
  check(`${tag} the page reaches under the notch and pads itself back by the safe-area insets; zoom is left alone`, /viewport-fit=cover/.test(meta.vp) && !/user-scalable|maximum-scale/.test(meta.vp) && /env\(safe-area-inset-top\)/.test(meta.css) && /env\(safe-area-inset-bottom\)/.test(meta.css) && /env\(safe-area-inset-left\)/.test(meta.css) && /env\(safe-area-inset-right\)/.test(meta.css), meta.vp);
  await ctx.close();
}

(async () => {
  fs.mkdirSync(SHOTS, { recursive: true });
  const browser = await chromium.launch();
  try {
    for (const size of SIZES) {
      if (ONLY.includes('board')) for (const f of ['running', 'keys', 'stopped']) await board(browser, size, f);
      if (ONLY.includes('keys')) await keys(browser, size);
      if (ONLY.includes('books')) for (const f of ['running', 'stopped']) await books(browser, size, f);
      if (ONLY.includes('refresh')) await refresh(browser, size);
      if (ONLY.includes('hostile')) await hostileRun(browser, size);
      if (ONLY.includes('look')) await look(browser, size);
    }
    if (ONLY.includes('books') && SIZES.includes('pc')) await keyboard(browser);
    if (ONLY.includes('touch') && SIZES.includes('phone')) await touch(browser);
  } catch (e) {
    check('the checks ran to the end', false, e.stack || e.message);
  }
  await browser.close();
  const failed = results.filter((r) => !r[0]);
  console.log(`\n${results.length - failed.length} of ${results.length} passed${failed.length ? '; FAILED:\n  ' + failed.map((r) => r[1]).join('\n  ') : ''}`);
  console.log(`screenshots in ${SHOTS}`);
  process.exit(failed.length ? 1 : 0);
})();
