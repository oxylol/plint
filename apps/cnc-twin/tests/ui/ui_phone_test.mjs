// CNC Twin UI test, phone (Playwright, Chromium, 390 x 844, touch): the projects, a project, the
// runs, the comparison and every tab of a run with no sideways overflow and 16 px fields, touch
// scrubbing and touch play/pause, 44 px transport buttons; no console errors.
// Run by tests/ui/run_ui_tests.sh (tests/ui/README.md).

import { MAIN, SAFE, TABS, OUT, ok, launch, open, go, settle, readTime, finish } from './ui_lib.mjs';

const browser = await launch();
const { context: phone, page: mobile } = await open(browser, { viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
async function phoneChecks(label) {
  const res = await mobile.evaluate(() => {
    const W = window.innerWidth;
    const inScroller = (el) => { for (let p = el.parentElement; p && p !== document.body; p = p.parentElement) { if (['auto', 'scroll', 'hidden'].includes(getComputedStyle(p).overflowX)) return true; } return false; };
    const wide = [];
    for (const el of document.querySelectorAll('body *')) {
      const r = el.getBoundingClientRect();
      if (r.width && r.right > W + 1 && !inScroller(el)) wide.push(`${el.tagName.toLowerCase()}.${String(el.className).slice(0, 30)}`);
    }
    const small = [];
    for (const el of document.querySelectorAll('input:not([type=checkbox]):not([type=range]):not([type=radio]):not([type=file]), select, textarea')) {
      if (el.offsetParent && parseFloat(getComputedStyle(el).fontSize) < 16) small.push(el.getAttribute('aria-label') || el.tagName);
    }
    return { scroll: document.documentElement.scrollWidth, W, wide: wide.slice(0, 5), small: small.slice(0, 5) };
  });
  ok(`phone ${label}: no sideways overflow`, res.scroll <= res.W && !res.wide.length, `${res.scroll} px; ${res.wide.join(', ')}`);
  ok(`phone ${label}: fields are 16 px`, !res.small.length, res.small.join(', '));
}
for (const hash of ['#/', '#/projects/example-pocket', '#/runs', `#/compare?a=${MAIN}&b=${SAFE}`]) {
  await go(mobile, hash);
  await mobile.waitForTimeout(400);
  await settle(mobile, '#app');
  await phoneChecks(hash);
}
await go(mobile, `#/runs/${MAIN}/overview`);
await mobile.waitForSelector('.verdict');
for (const tab of TABS) {
  await mobile.locator(`.tabs a[data-tab="${tab}"]`).scrollIntoViewIfNeeded();
  await mobile.click(`.tabs a[data-tab="${tab}"]`);
  await mobile.waitForSelector(`.panel-${tab}:not([hidden])`);
  await settle(mobile, `.panel-${tab}`);
  await mobile.waitForTimeout(250);
  await phoneChecks(`tab ${tab}`);
  await mobile.screenshot({ path: `${OUT}/phone-${tab}.png`, fullPage: true });
}
await mobile.click('.tabs a[data-tab="playback"]');
await mobile.waitForSelector('.scrubber');
const scrubber = mobile.locator('.scrubber');
const sbox = await scrubber.boundingBox();
const before = await readTime(mobile);
await scrubber.dispatchEvent('pointerdown', { pointerId: 7, pointerType: 'touch', isPrimary: true, clientX: sbox.x + sbox.width * 0.2, clientY: sbox.y + sbox.height / 2, bubbles: true });
await scrubber.dispatchEvent('pointermove', { pointerId: 7, pointerType: 'touch', isPrimary: true, clientX: sbox.x + sbox.width * 0.6, clientY: sbox.y + sbox.height / 2, bubbles: true });
await scrubber.dispatchEvent('pointerup', { pointerId: 7, pointerType: 'touch', isPrimary: true, clientX: sbox.x + sbox.width * 0.6, clientY: sbox.y + sbox.height / 2, bubbles: true });
await mobile.waitForTimeout(200);
const after = await readTime(mobile);
ok('touch scrubbing moves the time', Math.abs(after - before) > 0.3 && after > 1.0 && after < 1.8, `${before} -> ${after} s`);
await mobile.tap('.transport .play');
await mobile.waitForTimeout(700);
await mobile.tap('.transport .play');
ok('touch play and pause', (await readTime(mobile)) > after);
const touchTargets = await mobile.$$eval('.transport button', (els) => els.map((e) => Math.min(e.getBoundingClientRect().width, e.getBoundingClientRect().height)));
ok('transport buttons are touch-sized (44 px)', touchTargets.every((s) => s >= 44), touchTargets.join(','));
await phone.close();
await finish(browser, 'phone');

