// CNC Twin UI test, desktop (Playwright, Chromium, 1280 x 900): every project (with the upload
// and the check), a run started here and followed live to its verdict, every run, every tab of a
// run with every canvas checked on a screenshot, the playback (DRO, program, transport, plots of
// every channel, layers and the 3D view contract with the real web/js/view3d or the test double
// tests/ui/view3d_stub), findings jumping to their line and time, the chain of cause, the files and
// exports, the comparison, an uncertainty study and a calibration; no console errors.
// Run by tests/ui/run_ui_tests.sh (tests/ui/README.md).

import { MAIN, SAFE, TABS, OUT, UPLOAD, STUB, ok, rendered, launch, open, go, settle, readTime, openTab, finish } from './ui_lib.mjs';

const browser = await launch();
const { context: desktop, page } = await open(browser, { viewport: { width: 1280, height: 900 } });

// projects, upload, every project, check
await go(page, '#/');
await page.waitForSelector('.project-card');
const exampleCount = await page.locator('.project-card').count();
ok('projects list shows the examples', exampleCount >= 2, `${exampleCount} projects`);
await page.setInputFiles('input[type=file]', UPLOAD);
await page.fill('input[aria-label="Project id"]', 'ui-upload');
await page.click('form button:has-text("Upload")');
await page.waitForURL(/#\/projects\/ui-upload/, { timeout: 20000 }).catch(() => {});
ok('a zipped project uploads', /#\/projects\/ui-upload/.test(page.url()), page.url());
await go(page, '#/');
await page.waitForSelector('.project-card');
const projectLinks = await page.$$eval('.project-card .project-name', (els) => els.map((e) => e.getAttribute('href')));
ok('the upload is listed', projectLinks.some((href) => href.includes('ui-upload')), projectLinks.join(' '));
for (const href of projectLinks) {
  await go(page, href);
  await page.waitForSelector('.page-head h1');
  await page.waitForFunction(() => /cnctwin-project/.test(document.querySelector('.code-view')?.textContent || ''), null, { timeout: 15000 }).catch(() => {});
  const yaml = await page.textContent('.code-view').catch(() => '');
  ok(`project ${href} shows project.yaml`, /format: cnctwin-project/.test(yaml || ''));
  const files = await page.locator('table.data tbody tr').count();
  ok(`project ${href} lists its files`, files >= 2, `${files} rows`);
  await page.click('button:has-text("Check the project")');
  await page.waitForSelector('.check-result', { timeout: 30000 }).catch(() => {});
  ok(`project ${href} check result shown`, /ok|problems|done/.test(await page.textContent('.check-result').catch(() => '')));
}

// start a run and follow it live
await go(page, '#/projects/example-pocket');
await page.waitForSelector('textarea[aria-label="Run options (JSON)"]');
await page.fill('textarea[aria-label="Run options (JSON)"]', '{"step_delay_s": 0.01}');
await page.click('button:has-text("Start a run")');
await page.waitForURL(/#\/runs\/[^/]+\/overview/, { timeout: 20000 });
const newRun = decodeURIComponent(/#\/runs\/([^/]+)\//.exec(page.url())[1]);
ok('a run starts from the project page', !!newRun, newRun);
const sawProgress = await page.waitForFunction(() => /simulated/.test(document.querySelector('.progress-box')?.textContent || ''), null, { timeout: 30000 }).then(() => true).catch(() => false);
ok('the progress of the running run is shown', sawProgress);
const sawEta = await page.waitForFunction(() => /% · .* of .* simulated/.test(document.querySelector('.progress-box')?.textContent || ''), null, { timeout: 30000 }).then(() => true).catch(() => false);
const etaText = await page.textContent('.progress-box').catch(() => '');
ok('progress against the expected total while running', sawEta, (etaText || '').slice(0, 90));
ok('time remaining shown while running', /left/.test(etaText || '') || /100 %/.test(etaText || ''), (etaText || '').slice(0, 90));
const finished = await page.waitForSelector('.verdict.unsafe .verdict-main strong, .verdict.safe .verdict-main strong', { timeout: 90000 }).then(() => true).catch(() => false);
await page.waitForFunction(() => /safe to run/i.test(document.querySelector('.verdict')?.textContent || ''), null, { timeout: 30000 }).catch(() => {});
ok('the verdict appears when the run is done', finished && /safe to run/i.test(await page.textContent('.verdict')));
const liveFindings = await page.locator('.panel-overview .finding').count();
ok('findings arrived with the run', liveFindings === 5, `${liveFindings} findings`);

// runs list
await go(page, '#/runs');
await page.waitForSelector('table.data tbody tr');
const runLinks = await page.$$eval('table.data tbody tr td a', (els) => els.map((e) => e.getAttribute('href')).filter((h) => h.startsWith('#/runs/')));
ok('runs list shows every run', runLinks.length >= 4, `${runLinks.length} runs`);

// every run opens
for (const href of runLinks) {
  await go(page, href);
  await page.waitForSelector('.verdict');
  await settle(page, '.run-page');
  const banner = await page.textContent('.verdict');
  ok(`run ${href} shows a verdict banner`, /safe to run|Running|Queued|Cancelled|failed/i.test(banner), banner.trim().slice(0, 60));
}

// the main run: every tab
await go(page, `#/runs/${MAIN}/overview`);
await page.waitForSelector('.verdict.unsafe');
ok('the unsafe run says why', /out_of_tolerance/.test(await page.textContent('.verdict-reasons')));
for (const tab of TABS) {
  await openTab(page, tab);
  const errors = await page.locator(`.panel-${tab} .errorbox`).allTextContents();
  ok(`tab ${tab} renders without errors`, errors.length === 0, errors.join(' | ').slice(0, 200));
  await page.waitForFunction((sel) => {
    const root = document.querySelector(sel);
    return root && ![...root.querySelectorAll('.chart')].some((c) => c.offsetParent && !c.dataset.rendered)
      && ![...root.querySelectorAll('.tplot')].some((p) => p.offsetParent && p.dataset.points === undefined);
  }, `.panel-${tab}`, { timeout: 15000 }).catch(() => {});
  await page.waitForTimeout(300);
  const canvases = page.locator(`.panel-${tab} canvas:visible`);
  const count = await canvases.count();
  for (let i = 0; i < count; i++) {
    const canvas = canvases.nth(i);
    const box = await canvas.boundingBox();
    if (!box || box.width < 20 || box.height < 20) continue;
    await rendered(canvas, `tab ${tab} canvas ${i + 1}`);
  }
  await page.screenshot({ path: `${OUT}/desktop-${tab}.png`, fullPage: true });
}

// playback: DRO, program, plots, transport, the 3D view contract
await openTab(page, 'playback');
await page.waitForSelector('.dro-cell.axis .dro-v');
const droX = await page.locator('.dro-cell.axis .dro-v').first().textContent();
ok('DRO shows machine coordinates', /\d/.test(droX), droX);
ok('program viewer lists the program', (await page.locator('.program-row').count()) >= 10);
await page.waitForFunction(() => [...document.querySelectorAll('.panel-playback .tplot')].every((p) => Number(p.dataset.points) > 0), null, { timeout: 20000 }).catch(() => {});
const plotPoints = await page.$$eval('.panel-playback .tplot', (els) => els.map((e) => Number(e.dataset.points || 0)));
ok('default plots have data', plotPoints.length >= 3 && plotPoints.every((n) => n > 0), plotPoints.join(','));
const rate = (ms) => page.evaluate(async (wait) => {
  const read = () => parseFloat(document.querySelector('.time-readout').textContent);
  await new Promise((r) => requestAnimationFrame(() => r()));
  const a = read();
  const w0 = performance.now();
  await new Promise((r) => setTimeout(r, wait));
  await new Promise((r) => requestAnimationFrame(() => r()));
  return { sim: read() - a, wall: (performance.now() - w0) / 1000 };
}, ms);
const isPlaying = async () => (await page.getAttribute('.transport .play', 'aria-label')) === 'Pause';
await page.click('.transport .play');
await page.waitForFunction(() => document.querySelector('.transport .play').getAttribute('aria-label') === 'Pause', null, { timeout: 5000 }).catch(() => {});
const real = await rate(500);
if (await isPlaying()) await page.click('.transport .play');
const t1 = await readTime(page);
ok('playback runs at real speed (1x)', real.sim / real.wall > 0.6 && real.sim / real.wall < 1.25, `${real.sim.toFixed(3)} s simulated in ${real.wall.toFixed(3)} s`);
await page.click('button[aria-label="Next block"]');
await page.waitForTimeout(150);
const t2 = await readTime(page);
ok('next block steps the time', t2 > t1, `${t1} -> ${t2} s`);
await page.click('button[aria-label="Previous block"]');
await page.waitForTimeout(150);
ok('previous block steps back', (await readTime(page)) < t2);
await page.focus('.scrubber');
await page.keyboard.press('Home');
await page.waitForFunction(() => /^0\.000 s/.test(document.querySelector('.time-readout').textContent));
await page.selectOption('.transport select.speed', '5');
await page.click('.transport .play');
await page.waitForFunction(() => document.querySelector('.transport .play').getAttribute('aria-label') === 'Pause', null, { timeout: 5000 }).catch(() => {});
const fast = await rate(250);
if (await isPlaying()) await page.click('.transport .play');
ok('5x plays five times real speed', fast.sim / fast.wall > 3.0 && fast.sim / fast.wall < 6.25, `${fast.sim.toFixed(3)} s simulated in ${fast.wall.toFixed(3)} s`);
await page.selectOption('.transport select.speed', '1');
const active = await page.locator('.program-row.active').count();
ok('the active line follows playback', active === 1);
const droBlock = await page.locator('.dro-cell.wide .dro-v').last().textContent();
ok('DRO shows the executed block', /line \d+/.test(droBlock), droBlock);
// plots: every channel
await page.click('.picker > summary');
await page.click('.picker-actions button:has-text("All")');
const channelCount = await page.locator('.pick input[type=checkbox]').count();
await page.waitForFunction((n) => document.querySelectorAll('.panel-playback .plot-box').length === n, channelCount, { timeout: 20000 }).catch(() => {});
await page.waitForFunction(() => [...document.querySelectorAll('.panel-playback .tplot')].every((p) => p.dataset.points !== undefined && !p.classList.contains('is-loading')), null, { timeout: 30000 }).catch(() => {});
const allPoints = await page.$$eval('.panel-playback .tplot', (els) => els.map((e) => [e.querySelector('.tplot-title').textContent, Number(e.dataset.points || 0)]));
ok('every channel is plotted', allPoints.length === channelCount, `${allPoints.length} of ${channelCount}`);
ok('every plot has data', allPoints.every(([, n]) => n > 0), allPoints.filter(([, n]) => !n).map(([t]) => t).join(', '));
const plotsBox = page.locator('.panel-playback .tplot-body canvas');
for (let i = 0; i < Math.min(await plotsBox.count(), 30); i++) await rendered(plotsBox.nth(i), `plot ${allPoints[i] ? allPoints[i][0] : i}`);
const label0 = await plotsBox.first().getAttribute('aria-label');
await page.click('.plots-toolbar button:has-text("Zoom in")');
await page.waitForTimeout(600);
ok('zoom changes the plotted range', (await plotsBox.first().getAttribute('aria-label')) !== label0);
await page.click('.plots-toolbar button:has-text("Whole run")');
const tablePlot = page.locator('.panel-playback .plot-box').first();
await tablePlot.locator('button:has-text("Table")').click();
ok('a plot has a table view', (await tablePlot.locator('.figure-table table.data tbody tr').count()) >= 1);
await page.click('.picker-actions button:has-text("Defaults")');
// layers
const chips = page.locator('.chip-row .chip:not([disabled])');
await page.waitForFunction(() => document.querySelectorAll('.chip-row .chip').length >= 8, null, { timeout: 15000 }).catch(() => {});
const chipNames = await chips.allTextContents();
ok('every layer is offered', chipNames.length >= 8, chipNames.join(', '));
for (let i = 1; i < chipNames.length; i++) {
  await chips.nth(i).click();
  await page.waitForFunction(() => { const box = document.querySelector('.legend-box'); return box && !/Computing/.test(box.textContent); }, null, { timeout: 20000 }).catch(() => {});
  const bar = page.locator('.legend-box canvas');
  if (await bar.count()) await rendered(bar.first(), `colour bar of layer ${chipNames[i]}`);
  else ok(`layer ${chipNames[i]} legend`, /records no values/.test(await page.textContent('.legend-box')), await page.textContent('.legend-box'));
  if (STUB) {
    const last = await page.evaluate(() => (window.__view3d.calls.filter((c) => c[0] === 'setLayer').pop() || [])[1]);
    ok(`the 3D view gets layer ${chipNames[i]}`, typeof last === 'string' && last.length > 0, String(last));
  }
  const view = page.locator('.pb-view canvas');
  if (await view.count()) await rendered(view.first(), `3D view with layer ${chipNames[i]}`);
}
await chips.first().click();
await page.selectOption('select[aria-label="Section plane"]', 'x');
await page.locator('.layers details.disclosure > summary').click();
await page.locator('.vis-row label:has-text("Coolant shadows") input').check();
await page.locator('.layers label:has-text("Follow the tool") input').check();
await page.click('button:has-text("Reset camera")');
if (STUB) {
  const calls = await page.evaluate(() => window.__view3d.calls.map((c) => c[0]));
  for (const name of ['setTime', 'setLayer', 'setSection', 'setVisible', 'followTool', 'resetCamera', 'on']) ok(`the shell calls view.${name}`, calls.includes(name));
  const section = await page.evaluate(() => (window.__view3d.calls.filter((c) => c[0] === 'setSection').pop() || [])[1]);
  ok('the section goes to the view in metres', section && section.axis === 'x' && section.offset_m > 0 && section.offset_m < 0.1, JSON.stringify(section));
  const visible = await page.evaluate(() => window.__view3d.state.visible);
  ok('visibility reaches the view', visible.shadows === true && visible.machine === true, JSON.stringify(visible));
  const times = await page.evaluate(() => window.__view3d.calls.filter((c) => c[0] === 'setTime').map((c) => c[1]));
  ok('playback drives view.setTime', times.length > 20 && Math.max(...times) > 1, `${times.length} calls`);
  await page.locator('.layers label:has-text("Measure") input').check();
  const stubCanvas = page.locator('.pb-view canvas').first();
  await stubCanvas.click({ position: { x: 60, y: 60 } });
  await stubCanvas.click({ position: { x: 160, y: 120 } });
  ok('a pick shows the point', /Point .* mm/.test(await page.textContent('.pick-box')));
  ok('a measurement shows the distance', /Distance [\d.]+ mm/.test(await page.locator('.pick-box').nth(1).textContent()));
}
// findings jump to time and line
await openTab(page, 'overview');
await page.locator('.panel-overview .finding .loc').last().click();
await page.waitForSelector('.panel-playback:not([hidden]) .program-row.selected', { timeout: 15000 }).catch(() => {});
ok('a finding jumps to its program line', (await page.locator('.program-row.selected').count()) === 1);
ok('a finding jumps to its time', Math.abs((await readTime(page)) - 2.161) < 0.002, String(await readTime(page)));
// cause chain
await openTab(page, 'overview');
await page.locator('.panel-overview .finding details.causes > summary').last().click();
await page.waitForSelector('.panel-overview .chain li', { timeout: 15000 }).catch(() => {});
ok('the chain of cause expands', (await page.locator('.panel-overview .chain li').count()) >= 3);
// inspection details
await openTab(page, 'inspection');
ok('the CMM table lists the characteristics', (await page.locator('.panel-inspection table.data').first().locator('tbody tr').count()) === 5);
await page.locator('.panel-inspection details.disclosure > summary').first().click();
ok('probed points and error budget shown', /Probed points/.test(await page.textContent('.panel-inspection')) && /Error budget/.test(await page.textContent('.panel-inspection')));
// files
await openTab(page, 'files');
ok('report links', (await page.locator('.panel-files a[href*="report.html"]').count()) >= 1 && (await page.locator('.panel-files a[href*="setup_sheet.pdf"]').count()) >= 1);
ok('video and sound players', (await page.locator('.panel-files video').count()) === 1 && (await page.locator('.panel-files audio').count()) === 1);
ok('export links', (await page.locator('.panel-files a[href*="/export?format=csv"]').count()) >= 2);
const exportUrl = await page.locator('.panel-files a[href*="/export?format=csv"]').first().getAttribute('href');
const exported = await page.evaluate(async (url) => { const r = await fetch(url); return [r.status, (await r.text()).split('\n')[0]]; }, exportUrl);
ok('a CSV export downloads', exported[0] === 200 && /tick/.test(exported[1]), exported.join(' '));

// comparison
await go(page, `#/compare?a=${MAIN}&b=${SAFE}`);
await page.waitForSelector('.page-compare .headlines li', { timeout: 60000 }).catch(() => {});
const headlines = await page.locator('.page-compare .headlines li').allTextContents();
ok('the comparison shows its headlines', headlines.some((h) => /verdict changed/.test(h)), headlines.join(' | '));
ok('the comparison has diff tables', (await page.locator('.page-compare table.data').count()) >= 8);
await page.screenshot({ path: `${OUT}/desktop-compare.png`, fullPage: true });

// uncertainty study and calibration on the run started here
await go(page, `#/runs/${newRun}/uncertainty`);
await page.waitForSelector('button:has-text("Start the study")', { timeout: 20000 });
await page.fill('input[aria-label="Samples"]', '6');
await page.click('button:has-text("Start the study")');
const studied = await page.waitForFunction(() => /P\(not safe to run\)/.test(document.querySelector('.panel-uncertainty')?.textContent || ''), null, { timeout: 90000 }).then(() => true).catch(() => false);
ok('an uncertainty study runs and shows its results', studied);
await settle(page, '.panel-uncertainty');
for (const [i, canvas] of (await page.locator('.panel-uncertainty canvas:visible').all()).entries()) await rendered(canvas, `uncertainty chart ${i + 1}`);
await go(page, `#/runs/${newRun}/calibration`);
await page.waitForSelector('button:has-text("Start the calibration")', { timeout: 20000 });
await page.click('button:has-text("Start the calibration")');
const calibrated = await page.waitForFunction(() => /Measured against simulated/.test(document.querySelector('.panel-calibration')?.textContent || '') && document.querySelector('.panel-calibration canvas'), null, { timeout: 90000 }).then(() => true).catch(() => false);
ok('a calibration runs and shows the overlay', calibrated);
await page.waitForTimeout(500);
const overlay = page.locator('.panel-calibration .tplot-body canvas');
if (await overlay.count()) await rendered(overlay.first(), 'calibration overlay plot');
ok('calibration parameters listed', /kc_scale/.test(await page.textContent('.panel-calibration')));
await page.screenshot({ path: `${OUT}/desktop-calibration.png`, fullPage: true });
await desktop.close();
await finish(browser, 'desktop');

