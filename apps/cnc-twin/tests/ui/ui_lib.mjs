// Shared pieces of the CNC Twin UI tests (ui_test.mjs, ui_phone_test.mjs): settings from the
// environment, the result log, screenshot analysis (a PNG decoder: a canvas "renders" when its
// screenshot has at least three colours and more than 0.3 % of its pixels off the background, which
// works for 2-D and WebGL canvases alike), page helpers and the console error collector.
//
// Environment: UI_BASE (http://127.0.0.1:5995), PW_MODULE (playwright's index.mjs), UI_OUT (screens
// and results), UPLOAD_ZIP (a zipped project), VIEW3D_STUB (1 when the 3D view test double is served).

import zlib from 'node:zlib';
import fs from 'node:fs';

export const BASE = (process.env.UI_BASE || 'http://127.0.0.1:5995').replace(/\/$/, '');
export const PW = process.env.PW_MODULE || '/deps/node_modules/playwright/index.mjs';
export const OUT = process.env.UI_OUT || '/out';
export const UPLOAD = process.env.UPLOAD_ZIP || '/out/upload.zip';
export const STUB = process.env.VIEW3D_STUB === '1';
export const MAIN = '00001-pocket-unsafe';
export const SAFE = '00002-pocket-safe';
export const PARTIAL = '00003-pocket-partial';
export const TABS = ['overview', 'playback', 'inspection', 'surface', 'tools', 'production', 'stability', 'coolant', 'chips',
  'thermal', 'checks', 'uncertainty', 'calibration', 'files', 'info'];

export const results = [];
export const consoleProblems = [];
const started = Date.now();

export function ok(name, cond, detail = '') {
  results.push({ name, pass: !!cond, detail });
  console.log(`${cond ? 'ok  ' : 'FAIL'} ${name}${detail ? ` (${detail})` : ''}`);
}

export function pngStats(buffer) {
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
  const bpp = { 6: 4, 2: 3, 4: 2, 0: 1 }[colorType] || 4;
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
      if (filter === 1) v += a;
      else if (filter === 2) v += b;
      else if (filter === 3) v += (a + b) >> 1;
      else if (filter === 4) { const p = a + b - c; const pa = Math.abs(p - a); const pb = Math.abs(p - b); const pc = Math.abs(p - c); v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c; }
      cur[x] = v & 255;
    }
    prev = cur;
  }
  const counts = new Map();
  for (let i = 0; i < width * height; i++) {
    const key = ((pixels[i * bpp] >> 3) << 10) | ((pixels[i * bpp + 1] >> 3) << 5) | (pixels[i * bpp + 2] >> 3);
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  let top = 0;
  for (const v of counts.values()) top = Math.max(top, v);
  return { width, height, colours: counts.size, nonBackground: 1 - top / (width * height || 1) };
}

export async function rendered(locator, name, { min = 0.003 } = {}) {
  try {
    await locator.scrollIntoViewIfNeeded();
    const stats = pngStats(await locator.screenshot());
    ok(`${name} renders`, stats.colours >= 3 && stats.nonBackground > min, `${stats.colours} colours, ${(stats.nonBackground * 100).toFixed(1)} % drawn`);
    return stats;
  } catch (error) {
    ok(`${name} renders`, false, error.message.split('\n')[0]);
    return null;
  }
}

export async function launch() {
  const { chromium } = await import(PW);
  return chromium.launch({ args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
}

export async function open(browser, options) {
  const context = await browser.newContext(options);
  const page = await context.newPage();
  page.setDefaultTimeout(20000);
  page.on('console', (m) => { if (m.type() === 'error') consoleProblems.push(`${m.text()} at ${page.url()}`); });
  page.on('pageerror', (e) => consoleProblems.push(`uncaught: ${e.message} at ${page.url()}`));
  return { context, page };
}

export async function go(page, hash) {
  await page.goto(`${BASE}/${hash}`);
}

/** Wait until nothing under `scope` shows a loading spinner. */
export async function settle(page, scope = 'body') {
  await page.waitForFunction((sel) => {
    const root = document.querySelector(sel);
    return root && !root.querySelector('.loading');
  }, scope, { timeout: 30000 }).catch(() => {});
}

export async function readTime(page) {
  const text = await page.textContent('.time-readout');
  const m = /^([\d.]+) s/.exec(text.trim());
  return m ? Number(m[1]) : NaN;
}

export async function openTab(page, tab) {
  await page.click(`.tabs a[data-tab="${tab}"]`);
  await page.waitForSelector(`.panel-${tab}:not([hidden])`);
  await settle(page, `.panel-${tab}`);
}

/** Check the console, write <name>-results.json, exit 1 on any failure. */
export async function finish(browser, name) {
  ok('no console errors', consoleProblems.length === 0, consoleProblems.slice(0, 6).join(' | '));
  await browser.close();
  const failed = results.filter((r) => !r.pass);
  const seconds = (Date.now() - started) / 1000;
  fs.writeFileSync(`${OUT}/${name}-results.json`, JSON.stringify({ passed: results.length - failed.length, failed: failed.length, seconds, results }, null, 1));
  console.log(`\n${name}: ${results.length - failed.length} passed, ${failed.length} failed in ${seconds.toFixed(1)} s`);
  process.exit(failed.length ? 1 : 0);
}
