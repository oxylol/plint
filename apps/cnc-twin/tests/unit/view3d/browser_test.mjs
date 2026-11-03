// Browser test of the 3D view (web/js/view3d) in Chromium (Playwright 1.48.2, the
// mcr.microsoft.com/playwright:v1.48.2-jammy image), against serve_view3d.py: the real API on synthetic
// runs and the test page view3d-harness.html. Every call of the contract, the workpiece versions per
// time, playback, the tool pose against frames.bin, every layer (and its colour range against the
// shell's own computation), capped sections, picking, the three measurements, follow and reset,
// visibility, exaggeration, destroy, the contract's minimal api on a running run, a phone with touch, and
// no console errors. Screens and view3d-results.json go to OUT.
//
// Environment: V3D_BASE (http://127.0.0.1:5996), PW_MODULE (playwright's index.mjs), OUT (/out), SUITE
// (desktop: the contract on the rich run; runs: the default run, a running run through the minimal api,
// a phone; all: both, the default).

import fs from 'node:fs';
import zlib from 'node:zlib';

const BASE = (process.env.V3D_BASE || 'http://127.0.0.1:5996').replace(/\/$/, '');
const PW = process.env.PW_MODULE || '/deps/node_modules/playwright/index.mjs';
const OUT = process.env.OUT || '/out';
const SUITE = process.env.SUITE || 'all';
const RICH = '00002-v3d-rich';
const DEFAULT = '00001-v3d-default';
const PARTIAL = '00003-v3d-partial';
const results = [];
const problems = [];
const started = Date.now();

function ok(name, cond, detail = '') {
  results.push({ name, pass: !!cond, detail: String(detail).slice(0, 400) });
  console.log(`${cond ? 'ok  ' : 'FAIL'} ${name}${detail ? ` (${String(detail).slice(0, 200)})` : ''}`);
}

/** {colours, drawn, reddish, at(x, y)} of a PNG screenshot. */
function png(buffer) {
  let pos = 8;
  let width = 0; let height = 0; let colorType = 6;
  const idat = [];
  while (pos < buffer.length) {
    const length = buffer.readUInt32BE(pos);
    const type = buffer.toString('ascii', pos + 4, pos + 8);
    const data = buffer.subarray(pos + 8, pos + 8 + length);
    if (type === 'IHDR') { width = data.readUInt32BE(0); height = data.readUInt32BE(4); colorType = data[9]; }
    else if (type === 'IDAT') idat.push(data);
    else if (type === 'IEND') break;
    pos += 12 + length;
  }
  const raw = zlib.inflateSync(Buffer.concat(idat));
  const bpp = { 6: 4, 2: 3 }[colorType] || 4;
  const stride = width * bpp;
  const pixels = Buffer.alloc(height * stride);
  let prev = Buffer.alloc(stride);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)];
    const line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    const cur = pixels.subarray(y * stride, (y + 1) * stride);
    for (let x = 0; x < stride; x++) {
      const a = x >= bpp ? cur[x - bpp] : 0;
      const b = prev[x];
      const c = x >= bpp ? prev[x - bpp] : 0;
      let v = line[x];
      if (filter === 1) v += a; else if (filter === 2) v += b; else if (filter === 3) v += (a + b) >> 1;
      else if (filter === 4) { const p = a + b - c; const pa = Math.abs(p - a); const pb = Math.abs(p - b); const pc = Math.abs(p - c); v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c; }
      cur[x] = v & 255;
    }
    prev = cur;
  }
  const counts = new Map();
  let reddish = 0;
  let purple = 0;
  for (let i = 0; i < width * height; i++) {
    const r = pixels[i * bpp]; const g = pixels[i * bpp + 1]; const b = pixels[i * bpp + 2];
    const key = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3);
    counts.set(key, (counts.get(key) || 0) + 1);
    if (r > 110 && r - g > 50 && r - b > 50) reddish++;
    if (b > 90 && r > 70 && g < r - 10 && b > g + 25) purple++;
  }
  let top = 0;
  for (const v of counts.values()) top = Math.max(top, v);
  const n = width * height || 1;
  return { width, height, colours: counts.size, drawn: 1 - top / n, reddish: reddish / n, purple: purple / n, pixels, bpp };
}

function difference(a, b) {
  let changed = 0;
  const n = Math.min(a.pixels.length, b.pixels.length) / a.bpp;
  for (let i = 0; i < n; i++) {
    const d = Math.abs(a.pixels[i * a.bpp] - b.pixels[i * b.bpp]) + Math.abs(a.pixels[i * a.bpp + 1] - b.pixels[i * b.bpp + 1]) + Math.abs(a.pixels[i * a.bpp + 2] - b.pixels[i * b.bpp + 2]);
    if (d > 24) changed++;
  }
  return changed / n;
}

const { chromium } = await import(PW);
const browser = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });

const W = 720;
const H = 480;
async function open(run, { viewport = { width: W, height: H }, query = '', mobile = false, t = 0 } = {}) {
  const context = await browser.newContext(mobile ? { viewport, isMobile: true, hasTouch: true, deviceScaleFactor: 3 } : { viewport });
  const page = await context.newPage();
  page.setDefaultTimeout(30000);
  page.on('console', (m) => { if (m.type() === 'error') problems.push(`${run}: ${m.text()}`); });
  page.on('pageerror', (e) => problems.push(`${run}: uncaught ${e.message}`));
  await page.goto(`${BASE}/view3d-harness.html?run=${run}&t=${t}${query}`);
  await page.waitForFunction(() => window.__ready, null, { timeout: 60000 });
  const error = await page.evaluate(() => window.__error || null);
  return { context, page, error };
}

const debug = (page) => page.evaluate(() => window.view3d.debug());
async function settle(page, extra = 150) {
  await page.waitForFunction(() => { const d = window.view3d.debug(); return d.updated && !d.loading && d.ipw.pending === 0 && d.ipw.ready; }, null, { timeout: 30000 }).catch(() => {});
  await page.waitForTimeout(extra);
}
async function shot(page, name) {
  const buffer = await page.locator('#view canvas.view3d-canvas').screenshot({ path: `${OUT}/${name}.png` });
  return png(buffer);
}

/** Expected brick versions at t with the API's rule, from the run's own index files. */
async function expectedVersions(page, run, times) {
  return page.evaluate(async ({ run, times }) => {
    const idx = await (await fetch(`/api/runs/${run}/files/keyframes/bricks.idx`)).json();
    const kfs = (await (await fetch(`/api/runs/${run}/keyframes`)).json()).keyframes;
    const time = new Map(kfs.map((e) => [Number(e.k), Number(e.time_s)]));
    const out = {};
    for (const t of times) {
      const chosen = {};
      for (const [key, entries] of Object.entries(idx.bricks)) {
        let k = null;
        for (const e of entries) if (time.has(e[0]) && time.get(e[0]) <= t + 1e-12) k = e[0];
        if (k !== null) chosen[key] = k;
      }
      out[t] = chosen;
    }
    return out;
  }, { run, times });
}

/** The shell's colour-bar range of a layer (web/js/panels/layers.js layerLegend) for the same run. */
async function shellLegend(page, run, layer) {
  return page.evaluate(async ({ run, layer }) => {
    const { createApi } = await import('/js/api.js');
    const { layerAvailability, layerLegend } = await import('/js/panels/layers.js');
    const api = createApi({ base: '' });
    const keyframes = (await api.keyframes(run)).keyframes;
    const files = new Map((await api.files(run)).files.map((f) => [f.path, f.bytes]));
    const channels = (await api.channels(run)).channels;
    const data = { id: run, keyframes, files, channels };
    const available = await layerAvailability(api, data);
    if (!available[layer]) return null;
    const legend = await layerLegend(api, data, layer, available[layer]);
    return legend && legend.domain ? { min: legend.domain[0], max: legend.domain[1], unit: legend.unit } : legend;
  }, { run, layer });
}

// ===================================================================== the rich run, desktop
if (SUITE === 'all' || SUITE === 'desktop') {
  const { page, error, context } = await open(RICH, { t: 0 });
  ok('rich run: createView3D resolves', !error, error || '');
  const mountMs = await page.evaluate(() => window.__mountMs);
  ok('rich run: mounts within 10 s', mountMs < 10000, `${Math.round(mountMs)} ms`);
  await settle(page, 500);
  const home = await shot(page, 'rich-0-home');
  ok('rich run: the canvas renders', home.colours > 60 && home.drawn > 0.3, `${home.colours} colours, ${(home.drawn * 100).toFixed(1)} % drawn`);
  let d = await debug(page);
  ok('rich run: workpiece shown, stock hidden', d.ipw.count === 6 && d.ipw.ready && !d.stockVisible, JSON.stringify(d.ipw));
  ok('rich run: machine components placed', d.components.length >= 5 && d.components.every((c) => c.visible), JSON.stringify(d.components));
  ok('rich run: no notes', d.notes.length === 0, d.notes.join(' | '));

  // the workpiece at each time: every brick at its newest version at or before t
  const times = [0, 0.6, 1.0, 1.2, 1.6, 2.3];
  const expected = await expectedVersions(page, RICH, times);
  for (const t of times) {
    await page.evaluate((t) => window.view3d.setTime(t), t);
    await settle(page, 100);
    d = await debug(page);
    ok(`workpiece at t = ${t} s: brick versions as the run records them`, JSON.stringify(d.bricks) === JSON.stringify(expected[t]), `${JSON.stringify(d.bricks)} vs ${JSON.stringify(expected[t])}`);
  }
  const late = await shot(page, 'rich-1-t2.3');
  ok('the workpiece changes between keyframes (incremental replacement)', difference(home, late) > 0.002, `${(difference(home, late) * 100).toFixed(2)} % of pixels`);

  // playback at 5x for 0.6 s of wall time: versions only move forward, frames keep coming
  const playback = await page.evaluate(async () => {
    const seen = [];
    const t0 = performance.now();
    window.view3d.setTime(0.3);
    const framesBefore = window.view3d.debug().frames;
    await new Promise((resolve) => {
      const step = () => {
        const wall = (performance.now() - t0) / 1000;
        const t = 0.3 + 5 * wall;
        window.view3d.setTime(t);
        const d = window.view3d.debug();
        seen.push({ t, k: d.keyframe, bricks: { ...d.bricks } });
        if (wall < 0.6) requestAnimationFrame(step); else resolve();
      };
      requestAnimationFrame(step);
    });
    return { seen, frames: window.view3d.debug().frames - framesBefore };
  });
  const monotonic = playback.seen.every((s, i) => i === 0 || Object.keys(s.bricks).every((key) => s.bricks[key] >= (playback.seen[i - 1].bricks[key] ?? -1)));
  ok('playback: brick versions only move forward', monotonic);
  ok('playback: frames are drawn while playing', playback.frames >= 2, `${playback.frames} frames (software GL, one CPU)`);
  ok('playback: keyframes advance', playback.seen[playback.seen.length - 1].k > playback.seen[0].k, `${playback.seen[0].k} -> ${playback.seen[playback.seen.length - 1].k}`);

  // the tool where frames.bin has it
  await page.evaluate(() => window.view3d.setTime(1.2345));
  await page.waitForFunction(() => { const d = window.view3d.debug(); return d.sample && d.sample.exact; }, null, { timeout: 15000 }).catch(() => {});
  d = await debug(page);
  const frame = await page.evaluate(async (run) => (await (await fetch(`/api/runs/${run}/frames?t0=1.234&t1=1.235&format=json`)).json()), RICH);
  const tipA = frame.fields.tip_S_m[0][0];
  const tipB = frame.fields.tip_S_m[1][0];
  const w = (1.2345 - 1.234) / 0.001;
  const want = [0, 1, 2].map((c) => tipA[c] + (tipB[c] - tipA[c]) * w);
  ok('tool: tip interpolated between the two L1 ticks of frames.bin', d.sample && want.every((v, c) => Math.abs(v - d.sample.tip[c]) < 1e-7), `${JSON.stringify(d.sample)} vs ${JSON.stringify(want)}`);
  ok('tool: shown with its number from the timeline', d.tools[0].visible && d.tools[0].shown && d.tools[0].tool === 1, JSON.stringify(d.tools[0]));
  const tipOnScreen = await page.evaluate((p) => window.view3d.project(p), d.sample.tip);
  ok('tool: tip projects into the view', tipOnScreen.visible && tipOnScreen.x > 0 && tipOnScreen.x < W && tipOnScreen.y > 0 && tipOnScreen.y < H, JSON.stringify(tipOnScreen));

  // layers: mode, legend, the same range as the shell's colour bar, a changed picture
  await page.evaluate(() => window.view3d.setTime(1.6));
  await settle(page, 200);
  const plain = await shot(page, 'rich-2-plain');
  const modes = { temperature: 2, chips: 3, coolant: 4, finish: 1, deviation: 1, deflection: 2, wear: 1 };
  for (const [layer, mode] of Object.entries(modes)) {
    if (['finish', 'deviation', 'wear'].includes(layer)) await page.evaluate(() => window.view3d.setTime(2.3));
    else await page.evaluate(() => window.view3d.setTime(1.6));
    await page.evaluate((l) => window.view3d.setLayer(l), layer);
    await page.waitForFunction((l) => window.view3d.legend(l) !== null, layer, { timeout: 20000 }).catch(() => {});
    await settle(page, 400);
    d = await debug(page);
    ok(`layer ${layer}: shader mode ${mode}`, d.layerMode === mode, `${d.layerMode}`);
    const legend = await page.evaluate((l) => window.view3d.legend(l), layer);
    if (layer === 'coolant') {
      ok('layer coolant: categories as the shell legend', legend && legend.categories && legend.categories.map((c) => c[1]).join(',') === 'wetted,dry,jet,shadowed flow');
    } else {
      const shell = await shellLegend(page, RICH, layer);
      const same = legend && shell && Math.abs(legend.min - shell.min) <= 1e-9 * Math.max(1, Math.abs(shell.min)) && Math.abs(legend.max - shell.max) <= 1e-9 * Math.max(1, Math.abs(shell.max)) && legend.unit === shell.unit;
      ok(`layer ${layer}: colour range equals the shell's colour bar`, same, `${JSON.stringify(legend)} vs ${JSON.stringify(shell)}`);
    }
    const stats = await shot(page, `rich-3-layer-${layer}`);
    ok(`layer ${layer}: the picture changes`, difference(plain, stats) > 0.003, `${(difference(plain, stats) * 100).toFixed(2)} % of pixels`);
    const overlay = await page.locator('.v3d-legend').isVisible();
    ok(`layer ${layer}: colour bar shown in the view`, overlay);
  }
  await page.evaluate(() => window.view3d.setLayer(null));
  d = await debug(page);
  ok('layer none: material colours', d.layerMode === 0 && !(await page.locator('.v3d-legend').isVisible()));

  // sections with caps: the part red, the fixture its own colour; off again
  await page.evaluate(() => { window.view3d.setTime(2.3); window.view3d.setSection({ axis: 'y', offset_m: 0.0152, flip: true }); });
  await settle(page, 300);
  const cut = await shot(page, 'rich-4-section-y');
  d = await debug(page);
  ok('section: state', d.section.active && d.section.axis === 'y' && Math.abs(d.section.offset_m - 0.0152) < 1e-12 && d.section.flip === true);
  ok('section: the part cap shows (red)', cut.reddish > 0.004, `${(cut.reddish * 100).toFixed(2)} % reddish`);
  ok('section: the fixture cap shows (its own colour)', cut.purple > 0.004, `${(cut.purple * 100).toFixed(2)} % purple`);
  await page.evaluate(() => window.view3d.setSection({ axis: 'x', offset_m: 0.02, flip: false }));
  await settle(page, 200);
  const cutX = await shot(page, 'rich-4-section-x');
  ok('section: normal to x keeps the material before the offset', cutX.reddish > 0.002, `${(cutX.reddish * 100).toFixed(2)} % reddish`);
  await page.evaluate(() => window.view3d.setSection({ axis: null, offset_m: 0, flip: false }));
  await settle(page, 200);
  const whole = await shot(page, 'rich-4-section-off');
  ok('section: off again', whole.reddish < 0.0015 && !(await debug(page)).section.active, `${(whole.reddish * 100).toFixed(2)} % reddish`);

  // picking: a point on the uncut top and on the pocket floor, with the layer value
  await page.evaluate(() => { window.view3d.resetCamera(); window.view3d.setLayer('temperature'); });
  await page.waitForFunction(() => window.view3d.legend('temperature') !== null).catch(() => {});
  await settle(page, 300);
  const clickAt = async (point) => {
    const s = await page.evaluate((p) => window.view3d.project(p), point);
    const box = await page.locator('#view canvas.view3d-canvas').boundingBox();
    await page.mouse.click(box.x + s.x, box.y + s.y);
    await page.waitForTimeout(60);
    return page.evaluate(() => window.__events.pick[window.__events.pick.length - 1] || null);
  };
  const top = await clickAt([0.036, 0.027, 0.015]);
  console.log(`  ${((Date.now() - started) / 1000).toFixed(1)} s`);
  ok('pick: a point on the top face', top && Math.abs(top.point_m[2] - 0.015) < 1e-5 && Math.abs(top.point_m[0] - 0.036) < 5e-4 && Math.abs(top.point_m[1] - 0.027) < 5e-4, JSON.stringify(top));
  ok('pick: the normal of the top face', top && Math.abs(top.normal[2] - 1) < 1e-6, top && top.normal.join(','));
  ok('pick: the temperature there (K)', top && Number.isFinite(top.layerValue) && top.layerValue > 280 && top.layerValue < 500 && top.object === 'part', top && `${top.layerValue} ${top.object}`);
  const floor = await clickAt([0.018, 0.015, 0.013]);
  ok('pick: a point on the pocket floor', floor && Math.abs(floor.point_m[2] - 0.013) < 1e-5, JSON.stringify(floor));

  // measuring: distance, to a plane, hole diameter
  await page.evaluate(() => { window.view3d.setLayer(null); window.view3d.setMeasure(true); });
  ok('measure: toolbar shown', await page.locator('.v3d-measure').isVisible());
  const measureCount = () => page.evaluate(() => window.__events.measure.length);
  const lastMeasure = () => page.evaluate(() => window.__events.measure[window.__events.measure.length - 1]);
  let before = await measureCount();
  const a = await clickAt([0.034, 0.004, 0.015]);
  const b = await clickAt([0.034, 0.026, 0.015]);
  let m = await lastMeasure();
  const dab = a && b ? Math.hypot(a.point_m[0] - b.point_m[0], a.point_m[1] - b.point_m[1], a.point_m[2] - b.point_m[2]) : NaN;
  ok('measure: point to point', (await measureCount()) === before + 1 && m.kind === 'point' && Math.abs(m.distance_m - dab) < 1e-12 && Math.abs(m.distance_m - 0.022) < 1e-3, JSON.stringify(m));
  await page.locator('.v3d-measure button:has-text("To plane")').click();
  before = await measureCount();
  await clickAt([0.036, 0.020, 0.015]);
  await clickAt([0.018, 0.015, 0.013]);
  m = await lastMeasure();
  ok('measure: point to plane (pocket depth 2 mm)', (await measureCount()) === before + 1 && m.kind === 'plane' && Math.abs(m.distance_m - 0.002) < 1e-5, JSON.stringify(m));
  await page.locator('.v3d-measure button:has-text("Hole")').click();
  before = await measureCount();
  const centre = [0.020, 0.015];
  const r = 0.006;
  const picks = [];
  for (const angle of [0.2, 2.3, 4.2]) picks.push(await clickAt([centre[0] + r * Math.cos(angle), centre[1] + r * Math.sin(angle), 0.013]));
  m = await lastMeasure();
  ok('measure: hole diameter from three picks', (await measureCount()) === before + 1 && m.kind === 'hole' && Math.abs(m.diameter_m - 2 * r) < 1e-3, JSON.stringify(m));
  ok('measure: labels drawn', (await page.locator('.v3d-label:not([hidden])').count()) >= 1);
  await shot(page, 'rich-5-measure-hole');
  await page.evaluate(() => window.view3d.setMeasure(false));
  ok('measure: off hides the toolbar', !(await page.locator('.v3d-measure').isVisible()));

  // follow the tool, then reset
  await page.evaluate(() => { window.view3d.setTime(1.0); window.view3d.followTool(true); });
  await page.waitForTimeout(400);
  for (const t of [1.1, 1.3, 1.5]) {
    await page.evaluate((t) => window.view3d.setTime(t), t);
    await page.waitForTimeout(350);
    d = await debug(page);
    const s = await page.evaluate((p) => window.view3d.project(p), d.sample.tip);
    ok(`follow: the tip stays near the centre at t = ${t} s`, d.following && Math.abs(s.x - W / 2) < 0.2 * W && Math.abs(s.y - H / 2) < 0.2 * H, JSON.stringify(s));
  }
  await shot(page, 'rich-6-follow');
  await page.evaluate(() => window.view3d.resetCamera());
  await page.waitForTimeout(400);
  const stockCentre = await page.evaluate(() => window.view3d.project([0.02, 0.015, 0.0075]));
  ok('reset camera: the stock is framed again', !(await debug(page)).following && Math.abs(stockCentre.x - W / 2) < 0.1 * W && Math.abs(stockCentre.y - H / 2) < 0.1 * H, JSON.stringify(stockCentre));

  // visibility, deflected tool and its exaggeration
  await page.evaluate(() => window.view3d.setVisible({ machine: false, fixture: false, chips: false, coolant: false, shadows: false, tool: false, deflectedTool: true }));
  await page.evaluate(() => window.view3d.setTime(1.4));
  await page.waitForTimeout(500);
  d = await debug(page);
  ok('visibility: machine and fixture hidden', d.components.every((c) => !c.visible));
  ok('visibility: tool hidden, deflected tool shown', !d.tools[0].shown && d.tools[0].ghost && d.tools[0].deflection && d.tools[0].deflection.vector_m.length === 3, JSON.stringify(d.tools[0]));
  ok('visibility: chips and coolant hidden', !d.chipsShown && !d.coolantShown);
  ok('deflection: read from the tool displacement channel', d.tools[0].deflection && d.tools[0].deflection.source === 'dynamics.tool_displacement_m', JSON.stringify(d.tools[0].deflection));
  ok('deflection: exaggeration control shown', await page.locator('.v3d-defl').isVisible());
  await page.evaluate(() => window.view3d.setExaggeration(10));
  ok('deflection: exaggeration set', (await debug(page)).exaggeration === 10);
  await page.locator('.v3d-defl button[aria-label="More exaggeration"]').click();
  ok('deflection: + steps the exaggeration up', (await debug(page)).exaggeration === 20, `${(await debug(page)).exaggeration}`);
  await shot(page, 'rich-7-deflected-only');
  await page.evaluate(() => window.view3d.setVisible({ machine: true, fixture: true, chips: true, coolant: true, shadows: true, tool: true, deflectedTool: false }));
  await page.evaluate(() => window.view3d.setTime(1.0));
  await page.waitForTimeout(800);
  d = await debug(page);
  ok('coolant: the blocked jet has its shadow', d.jets.some((j) => j.id === 'n2' && j.shadowed && j.blockedBy === 'workpiece 0'), JSON.stringify(d.jets));
  ok('chips: in flight and piled', d.chips > 0 && d.pile && d.chipsShown);
  await shot(page, 'rich-8-coolant-shadow');

  // unknown layer refused; destroy
  const refused = await page.evaluate(() => { try { window.view3d.setLayer('nonsense'); return false; } catch { return true; } });
  ok('setLayer refuses an unknown layer', refused);
  await page.evaluate(() => window.view3d.destroy());
  const left = await page.evaluate(() => ({ canvases: document.querySelectorAll('#view canvas').length, overlays: document.querySelectorAll('#view .v3d-root').length }));
  ok('destroy: the view leaves nothing behind', left.canvases === 0 && left.overlays === 0, JSON.stringify(left));
  await page.evaluate(() => window.view3d.destroy());
  await context.close();
}

// ===================================================================== the default synthetic run
if (SUITE === 'all' || SUITE === 'runs') {
  const { page, error, context } = await open(DEFAULT, { t: 1.2 });
  ok('default run: createView3D resolves', !error, error || '');
  await settle(page, 400);
  const stats = await shot(page, 'default-0');
  ok('default run: the canvas renders', stats.colours > 40 && stats.drawn > 0.3, `${stats.colours} colours`);
  const expected = await expectedVersions(page, DEFAULT, [1.2]);
  const d = await debug(page);
  ok('default run: brick versions at 1.2 s', JSON.stringify(d.bricks) === JSON.stringify(expected[1.2]), JSON.stringify(d.bricks));
  for (const [layer, mode] of Object.entries({ temperature: 3, chips: 3, deviation: 1, finish: 1, wear: 1, deflection: 0 })) {
    await page.evaluate((l) => { window.view3d.setTime(2.3); window.view3d.setLayer(l); }, layer);
    await page.waitForFunction((l) => window.view3d.legend(l) !== null, layer, { timeout: 20000 }).catch(() => {});
    await settle(page, 200);
    const mine = await page.evaluate((l) => window.view3d.legend(l), layer);
    const shell = await shellLegend(page, DEFAULT, layer);
    ok(`default run: layer ${layer} range equals the shell's`, mine && shell && Math.abs(mine.min - shell.min) <= 1e-9 * Math.max(1, Math.abs(shell.min)) && Math.abs(mine.max - shell.max) <= 1e-9 * Math.max(1, Math.abs(shell.max)),
      `${JSON.stringify(mine)} vs ${JSON.stringify(shell)}`);
    ok(`default run: layer ${layer} shader mode ${mode}`, (await debug(page)).layerMode === mode, `${(await debug(page)).layerMode}`);
  }
  await shot(page, 'default-1-wear');
  await context.close();
}

// ===================================================================== a running run through the contract's minimal api
if (SUITE === 'all' || SUITE === 'runs') {
  const { page, error, context } = await open(PARTIAL, { t: 1.1, query: '&api=minimal' });
  ok('running run, minimal api {get, getBinary}: createView3D resolves', !error, error || '');
  await settle(page, 400);
  const expected = await expectedVersions(page, PARTIAL, [1.1]);
  const d = await debug(page);
  ok('running run: brick versions at 1.1 s', JSON.stringify(d.bricks) === JSON.stringify(expected[1.1]), JSON.stringify(d.bricks));
  await page.evaluate(() => window.view3d.setLayer('finish'));
  await page.waitForFunction(() => /records no values/.test(document.querySelector('.v3d-legend')?.textContent || ''), null, { timeout: 10000 }).catch(() => {});
  const legend = await page.evaluate(() => window.view3d.legend('finish'));
  ok('running run: no final attributes yet, the layer says so', legend === null && (await page.locator('.v3d-legend').textContent()).includes('records no values'), JSON.stringify(legend));
  await page.waitForTimeout(3300);
  const later = await debug(page);
  ok('running run: refreshing the index files keeps the view going', later.ipw.ready && later.notes.length === 0, later.notes.join(' | '));
  await shot(page, 'partial-0');
  await context.close();
}

// ===================================================================== a phone with touch
if (SUITE === 'all' || SUITE === 'runs') {
  const { page, error, context } = await open(RICH, { viewport: { width: 390, height: 520 }, mobile: true, t: 2.3 });
  ok('phone: createView3D resolves', !error, error || '');
  await settle(page, 500);
  const d = await debug(page);
  ok('phone: pixel ratio capped for coarse pointers (3x screen -> 1.5)', d.renderer.pixelRatio <= 1.5 + 1e-9, `${d.renderer.pixelRatio}`);
  const stats = await shot(page, 'phone-0');
  ok('phone: the canvas renders', stats.colours > 40 && stats.drawn > 0.3, `${stats.colours} colours`);
  const s = await page.evaluate(() => window.view3d.project([0.036, 0.027, 0.015]));
  const box = await page.locator('#view canvas.view3d-canvas').boundingBox();
  await page.touchscreen.tap(box.x + s.x, box.y + s.y);
  await page.waitForTimeout(300);
  const pick = await page.evaluate(() => window.__events.pick[window.__events.pick.length - 1] || null);
  ok('phone: a tap picks the part', pick && Math.abs(pick.point_m[2] - 0.015) < 1e-5, JSON.stringify(pick));
  const buttons = await page.evaluate(() => { window.view3d.setMeasure(true); return [...document.querySelectorAll('.v3d-measure button')].map((b) => b.getBoundingClientRect().height); });
  ok('phone: measuring buttons are 44 px touch targets', buttons.length >= 4 && buttons.every((h) => h >= 44), buttons.join(','));
  await context.close();
}

ok('no console errors', problems.length === 0, problems.slice(0, 6).join(' | '));
await browser.close();
const failed = results.filter((r) => !r.pass);
const seconds = (Date.now() - started) / 1000;
fs.writeFileSync(`${OUT}/view3d-${SUITE}-results.json`, JSON.stringify({ passed: results.length - failed.length, failed: failed.length, seconds, results }, null, 1));
console.log(`\nview3d ${SUITE}: ${results.length - failed.length} passed, ${failed.length} failed in ${seconds.toFixed(1)} s`);
process.exit(failed.length ? 1 : 0);
