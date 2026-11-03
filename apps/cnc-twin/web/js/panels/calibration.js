/* Calibration overlay (cnctwin.mdata, docs/models/calibration.md): recorded machine data against
 * the simulation, before (simulated, prior parameters) and after (calibrated) the fit, on one time
 * axis (simulated time after the alignment), with the error metrics of each series, the fitted
 * parameters with their intervals and flags, and what goes into the calibration overlay. Results
 * are read from the calibration jobs of this run (overlay.json, calibration.json); a new
 * calibration can be started from a machine data log. */

import { h, clear, section, table, notice, errorBox, figure, kv, button, toast, loading, humanize, jsonView, lazyDetails } from './dom.js';
import { PlotGroup, TimePlot, medianStep } from '../plot/timeplot.js';
import { XYPlot } from '../plot/charts.js';
import { chrome } from '../plot/colormap.js';
import { displayUnit, magnitudeOf, num, fixed, duration, DASH, asNumber } from '../plot/units.js';

export const TARGETS = ['spindle', 'servo', 'axis_friction', 'block_time', 'thermal_growth', 'tool_life', 'usui', 'deflection', 'kienzle', 'mechanistic'];
const COLUMNS = [['measured', 0], ['simulated', 1], ['calibrated', 2]];

function seriesFigure(s, data, page) {
  const c = chrome();
  const mag = magnitudeOf(s.measured, s.simulated, s.calibrated);
  const du = displayUnit(s.unit, mag);
  const conv = (arr) => Float64Array.from(arr || [], (v) => (v === null ? NaN : v * du.factor + du.offset));
  const t = Float64Array.from(s.t_s || [], (v) => Number(v));
  const present = COLUMNS.filter(([key]) => Array.isArray(s[key]));
  return figure({
    title: `${s.target ? `${s.target}: ` : ''}${s.name}`, subtitle: `${du.label || s.unit || ''}${s.reduced === 'm4' ? ' · M4-reduced' : ''}`,
    chart: (box) => {
      if (s.x === 'index') {
        new XYPlot(box, { xLabel: 'observation', yLabel: du.label, height: 220 }).setData({
          lines: present.map(([key, slot]) => ({ x: Array.from(t), y: Array.from(conv(s[key])), color: c.series[slot], label: key })),
        });
        return;
      }
      const end = t.length ? t[t.length - 1] : 1;
      const group = new PlotGroup({ duration: Math.max(end, data.duration(), 1e-3), onSeek: (x) => page.seekTo(x) });
      const plot = new TimePlot(box, { group, height: 190, title: s.name, unit: du.label });
      plot.setNote(s.doc || '');
      const gap = 3 * (medianStep(t) || 0) || Infinity;
      plot.setSeries(present.map(([key, slot]) => ({ label: key, color: c.series[slot], t, v: conv(s[key]), gap })));
    },
    tableView: () => metricsTable(s.metrics || {}, s.unit),
  });
}

function metricsTable(metrics, unit) {
  const rows = Object.entries(metrics).map(([name, m]) => ({ name, ...m }));
  const value = (v) => (asNumber(v) === null ? DASH : num(v));
  return table({
    columns: [{ key: 'name', label: 'prediction', render: (r) => h('b', null, r.name) }, { key: 'n', label: 'n', num: true },
      { key: 'bias', label: `bias${unit ? ` (${unit})` : ''}`, num: true, render: (r) => value(r.bias) }, { key: 'rmse', label: 'RMSE', num: true, render: (r) => value(r.rmse) },
      { key: 'mae', label: 'MAE', num: true, render: (r) => value(r.mae) }, { key: 'max_abs', label: 'max |error|', num: true, render: (r) => value(r.max_abs) },
      { key: 'nrmse', label: 'NRMSE', num: true, render: (r) => value(r.nrmse) }, { key: 'r2', label: 'R²', num: true, render: (r) => value(r.r2) }, { key: 'r', label: 'r', num: true, render: (r) => value(r.r) }],
    rows, empty: 'No metrics.', caption: 'predicted minus measured, SI units',
  });
}

export function renderOverlay(root, overlay, report, { data, page }) {
  const alignment = (overlay && overlay.alignment) || (report && report.alignment) || null;
  if (alignment) {
    root.append(section('Time alignment', kv([['method', humanize(alignment.method)], ['offset', `${num(alignment.offset_s)} s`], ['scale', fixed(alignment.scale, 8)],
      ['meaning', 't_sim = offset + scale × t_measured']])));
  }
  const series = (overlay && overlay.series) || [];
  const box = section('Measured against simulated');
  if (!series.length) box.append(notice('This calibration wrote no overlay series.'));
  for (const s of series) box.append(seriesFigure(s, data, page), metricsTable(s.metrics || {}, s.unit));
  root.append(box);
  if (overlay && overlay.cycle_time) {
    const ct = overlay.cycle_time;
    root.append(section('Cycle time', kv([['measured', duration(ct.measured_s)], ['simulated', duration(ct.simulated_s)], ['difference', ct.error_s === null || ct.error_s === undefined ? DASH : `${num(ct.error_s)} s (${num((ct.error_frac || 0) * 100, 3)} %)`], ['method', ct.method || DASH]])));
  }
  if (overlay && Array.isArray(overlay.blocks) && overlay.blocks.length) {
    root.append(section('Blocks', table({
      columns: [{ key: 'line', label: 'line', num: true }, { key: 'block', label: 'block', render: (b) => h('code', null, b.block || '') },
        { key: 'sim_duration_s', label: 'simulated', num: true, render: (b) => duration(b.sim_duration_s) }, { key: 'measured_duration_s', label: 'measured', num: true, render: (b) => duration(b.measured_duration_s) },
        { key: 'error_s', label: 'difference', num: true, render: (b) => `${num(b.error_s * 1e3)} ms` }],
      rows: overlay.blocks, maxRows: 100,
    })));
  }
  const targets = (report && report.targets) || [];
  for (const target of targets) {
    const params = target.params || [];
    const tbox = section(`Target ${humanize(target.target)} (${target.status || 'fitted'}${target.simulator ? `, ${target.simulator}` : ''})`);
    if (target.reason) tbox.append(notice(target.reason));
    if (params.length) {
      tbox.append(table({
        columns: [{ key: 'name', label: 'parameter', render: (p) => h('b', null, p.name) }, { key: 'unit', label: 'unit' },
          { key: 'prior', label: 'prior', num: true, render: (p) => num(p.prior) }, { key: 'value', label: 'calibrated', num: true, render: (p) => num(p.value) },
          { key: 'ci', label: '95 % interval', sortable: false, render: (p) => (Array.isArray(p.ci) ? `${num(p.ci[0])} … ${num(p.ci[1])}` : DASH) },
          { key: 'std_error', label: 'std. error', num: true, render: (p) => num(p.std_error) },
          { key: 'flags', label: 'flags', sortable: false, render: (p) => [p.free ? 'free' : 'fixed', p.identifiable === false ? 'not identifiable' : null, p.at_bound ? 'at a bound' : null].filter(Boolean).join(', ') },
          { key: 'written', label: 'written', render: (p) => (p.written ? h('span', { class: 'badge tone-ok' }, 'yes') : h('span', { class: 'badge tone-muted' }, 'no')) },
          { key: 'remark', label: 'remark' }],
        rows: params,
      }));
    }
    if (target.confounded && target.confounded.length) tbox.append(notice(`Confounded: ${target.confounded.map((x) => (Array.isArray(x) ? x.join(' with ') : JSON.stringify(x))).join('; ')}`, 'warning'));
    if (target.notes && target.notes.length) tbox.append(h('ul', { class: 'plain small' }, target.notes.map((n) => h('li', null, String(n)))));
    root.append(tbox);
  }
  if (report && report.set && Object.keys(report.set).length) {
    root.append(section('Written to the calibration overlay', table({
      columns: [{ key: 'path', label: 'path', render: (r) => h('code', null, r.path) }, { key: 'value', label: 'value', num: true, render: (r) => num(r.value) },
        { key: 'ci', label: 'interval', render: (r) => (Array.isArray(r.ci) ? `${num(r.ci[0])} … ${num(r.ci[1])}` : DASH) }, { key: 'parameter', label: 'parameter' }],
      rows: Object.entries(report.set).map(([path, v]) => ({ path, ...v })),
    })));
  }
}

function startForm(api, data, onStarted) {
  const kind = h('select', { class: 'input', 'aria-label': 'Machine data kind' }, ['csv', 'mtconnect', 'opcua', 'focas', 'log'].map((k) => h('option', { value: k }, k)));
  const path = h('input', { class: 'input', type: 'text', placeholder: 'machine_log.csv (in the project folder) or a URL', 'aria-label': 'Machine data path' });
  const mapping = h('textarea', { class: 'input mono', rows: '3', spellcheck: 'false', placeholder: '{"spindle.power_W": "SpindlePower kW"}', 'aria-label': 'Mapping (JSON)' });
  const simulator = h('select', { class: 'input', 'aria-label': 'Simulator' }, ['auto', 'surrogate', 'loop'].map((k) => h('option', { value: k }, k)));
  const boxes = TARGETS.map((t) => ({ t, box: h('input', { type: 'checkbox' }) }));
  const extra = h('textarea', { class: 'input mono', rows: '3', spellcheck: 'false', placeholder: '{"alignment": {"method": "markers"}}', 'aria-label': 'More options (JSON)' });
  const message = h('div');
  const go = button('Start the calibration', async () => {
    clear(message);
    let options;
    let map = null;
    try {
      options = extra.value.trim() ? JSON.parse(extra.value) : {};
      map = mapping.value.trim() ? JSON.parse(mapping.value) : null;
    } catch (error) { message.append(errorBox(`JSON: ${error.message}`)); return; }
    const targets = boxes.filter((b) => b.box.checked).map((b) => b.t);
    if (targets.length) options.targets = targets;
    options.simulator = simulator.value;
    const machineData = path.value.trim() ? { kind: kind.value, path: path.value.trim(), ...(map ? { mapping: map } : {}) } : null;
    go.disabled = true;
    try {
      const started = await api.calibrate({ projectId: data.projectId, runId: data.id, machineData, options });
      toast(`Calibration ${started.job.id} queued`, 'info');
      onStarted(started.job);
    } catch (error) { message.append(errorBox(error)); } finally { go.disabled = false; }
  }, { kind: 'primary' });
  return h('form', { class: 'form', on: { submit: (e) => e.preventDefault() } },
    h('div', { class: 'form-row' }, h('label', { class: 'field' }, h('span', null, 'Machine data'), kind), h('label', { class: 'field grow' }, h('span', null, 'Path or address'), path)),
    h('label', { class: 'field' }, h('span', null, 'Channel mapping (JSON: canonical channel to "<column or node id> [unit]"; without a path the project’s machine_data section is used)'), mapping),
    h('fieldset', { class: 'picker-group' }, h('legend', null, 'Targets (none ticked: every target the data allow)'), h('div', { class: 'vis-row' }, boxes.map((b) => h('label', { class: 'check' }, b.box, h('span', null, humanize(b.t)))))),
    h('div', { class: 'form-row' }, h('label', { class: 'field' }, h('span', null, 'Simulator'), simulator)),
    h('label', { class: 'field' }, h('span', null, 'More options (JSON)'), extra), go, message);
}

export async function renderCalibration(root, { api, data, page }) {
  const holder = h('div');
  root.append(holder);
  const draw = async (selected = null) => {
    clear(holder);
    holder.append(loading('Looking for calibrations…'));
    let jobs = [];
    try { jobs = ((await api.jobs({ kind: 'calibrate', limit: 500 })).jobs || []).filter((j) => j.run_id === data.id || (data.projectId && j.project_id === data.projectId)); } catch (error) { clear(holder); holder.append(errorBox(error)); return; }
    clear(holder);
    const done = jobs.filter((j) => j.state === 'done');
    const listBox = section('Calibrations of this run and project');
    holder.append(listBox);
    if (!jobs.length) listBox.append(notice('No calibration has been made against this run or its project yet.'));
    else {
      listBox.append(table({
        columns: [{ key: 'id', label: 'job', render: (j) => h('b', null, j.id) }, { key: 'state', label: 'state' }, { key: 'run_id', label: 'run' },
          { key: 'error', label: 'error', render: (j) => (j.error ? j.error.message : '') }],
        rows: jobs, onRow: (j) => draw(j.id),
      }));
    }
    for (const j of jobs.filter((x) => ['queued', 'running', 'cancelling'].includes(x.state))) follow(api, j, listBox, () => draw(j.id));
    const chosen = selected ? jobs.find((j) => j.id === selected) : done[0];
    if (chosen && chosen.state === 'done') {
      const resultBox = h('div');
      holder.append(h('h2', { class: 'card-title' }, `Calibration ${chosen.id}`), resultBox);
      try {
        const files = new Set(((await api.jobFiles(chosen.id)).files || []).map((f) => f.path));
        const overlay = files.has('overlay.json') ? await api.jobFileJson(chosen.id, 'overlay.json') : null;
        const report = files.has('calibration.json') ? await api.jobFileJson(chosen.id, 'calibration.json') : (await api.job(chosen.id)).result;
        renderOverlay(resultBox, overlay, report, { data, page });
        const links = [...files].filter((f) => /\.(yaml|json|parquet)$/.test(f));
        if (links.length) resultBox.append(section('Files of the calibration', h('div', { class: 'link-grid' }, links.map((f) => h('a', { class: 'btn tile-link', href: api.jobFileUrl(chosen.id, f), target: '_blank', rel: 'noopener' }, f)))));
      } catch (error) { resultBox.append(errorBox(error)); }
    } else if (chosen) {
      holder.append(notice(`Calibration ${chosen.id} is ${chosen.state}.`));
      if (chosen.error) holder.append(lazyDetails('Error', (b) => b.append(jsonView(chosen.error))));
    }
    const startBox = section('New calibration');
    startBox.append(startForm(api, data, (job) => follow(api, job, startBox, () => draw(job.id))));
    holder.append(startBox);
  };
  await draw();
}

function follow(api, job, box, onDone) {
  const line = h('p', { class: 'muted' }, `Job ${job.id}: ${job.state}`);
  box.append(line);
  api.waitJob(job.id, {
    onProgress: (record) => { const p = record.progress || {}; line.textContent = `Job ${record.id}: ${record.state}${p.phase ? ` · ${p.phase}` : ''}${p.message ? ` · ${p.message}` : ''}`; },
  }).then((result) => {
    line.textContent = `Job ${result.job.id}: ${result.job.state}${result.job.error ? ` · ${result.job.error.message}` : ''}`;
    onDone(result);
  }).catch((error) => box.append(errorBox(error)));
}
