/* Uncertainty results of a run (cnctwin.uncertainty, docs/models/uncertainty.md section 10): the
 * probability that the run is not safe, the factors and their distributions, every inspected
 * dimension's distribution and capability (Cp, Cpk with intervals), the scalar results, the
 * probability of each finding category and program location, and the sensitivity ranking with a
 * scatter of the strongest factor. A study is read from the run (uncertainty/study.json, attached by
 * the study) or from an uncertainty job of this run; a new study can be started here. */

import { h, clear, section, table, notice, errorBox, figure, kv, tiles, button, lazyDetails, humanize, toast, loading } from './dom.js';
import { Histogram, XYPlot } from '../plot/charts.js';
import { barList } from '../plot/bars.js';
import { histogram } from '../plot/scale.js';
import { chrome } from '../plot/colormap.js';
import { DASH, num, pct, fmt, fixed, quantity, asNumber } from '../plot/units.js';

function probability(p) {
  if (!p || asNumber(p.p) === null) return DASH;
  return `${pct(p.p, 1)} (${pct(p.low, 1)} … ${pct(p.high, 1)}, ${p.k} of ${p.n})`;
}

function valueOf(name, v) {
  if (v === null || v === undefined) return DASH;
  return quantity(name, v).text;
}

function lengthText(v) {
  if (asNumber(v) === null) return DASH;
  return Math.abs(v) < 1e-3 ? fmt.um(v, 2) : fmt.mm(v, 4);
}

async function findStudy(api, data) {
  if (data.has('uncertainty/study.json')) {
    return { doc: await api.fileJson(data.id, 'uncertainty/study.json'), source: 'uncertainty/study.json of this run' };
  }
  const jobs = await api.jobs({ kind: 'uncertainty', limit: 200 });
  const mine = (jobs.jobs || []).filter((j) => j.run_id === data.id);
  const done = mine.find((j) => j.state === 'done');
  if (done) {
    const files = await api.jobFiles(done.id);
    if ((files.files || []).some((f) => f.path === 'study.json')) return { doc: await api.jobFileJson(done.id, 'study.json'), source: `job ${done.id}` };
  }
  return { doc: null, jobs: mine };
}

export function renderStudy(root, study, { page, data }) {
  const results = study.results || {};
  const settings = study.settings || {};
  const verdict = results.verdict || {};
  root.append(section(null, tiles([
    { label: 'P(not safe to run)', value: verdict.unsafe ? pct(verdict.unsafe.p, 1) : DASH, sub: verdict.unsafe ? `95 % interval ${pct(verdict.unsafe.low, 1)} … ${pct(verdict.unsafe.high, 1)}` : null,
      tone: verdict.unsafe && verdict.unsafe.p > 0 ? 'error' : 'ok' },
    { label: 'Samples', value: `${num(results.samples_ok)} ok`, sub: `${num(results.samples_failed)} failed`, tone: results.samples_failed ? 'warning' : null },
    { label: 'Plan', value: `${settings.runs ?? DASH} × ${settings.method || DASH}`, sub: `seed ${settings.seed ?? DASH}${settings.vary_seed ? ', own seed per sample' : ''}` },
  ])));
  const factors = study.factors || [];
  root.append(section('Factors', table({
    columns: [{ key: 'name', label: 'factor', render: (f) => h('b', null, f.name) }, { key: 'label', label: 'what' },
      { key: 'distribution', label: 'distribution', sortable: false, render: (f) => {
        const d = f.distribution || {};
        const unit = f.unit && f.unit !== '-' ? f.unit : '';
        if (d.kind === 'uniform') return `uniform ${num(d.low)} … ${num(d.high)}${unit ? ` ${unit}` : ''}`;
        return `${humanize(d.kind)} mean ${num(d.mean)}, sd ${num(d.std)}${d.low !== undefined ? ` on ${num(d.low)} … ${num(d.high)}` : ''}${unit ? ` ${unit}` : ''}`;
      } },
      { key: 'nominal', label: 'nominal', num: true, render: (f) => num(f.nominal) }, { key: 'stated_by', label: 'stated by' }],
    rows: factors, empty: 'No factors.',
  }), (study.not_varied || []).length ? h('p', { class: 'small muted' }, `Not varied: ${(study.not_varied || []).map((x) => `${x.key} (${x.reason})`).join('; ')}`) : null,
  (study.options_only || []).length ? h('p', { class: 'small muted' }, `Reach the simulation as options only: ${study.options_only.join(', ')}`) : null));
  const dims = results.dimensions || [];
  root.append(section('Dimensions', table({
    columns: [
      { key: 'id', label: 'characteristic', render: (d) => h('b', null, d.id) }, { key: 'kind', label: 'kind', render: (d) => humanize(d.kind) },
      { key: 'mean', label: 'mean', num: true, value: (d) => d.measured && d.measured.mean, render: (d) => lengthText(d.measured && d.measured.mean) },
      { key: 'std', label: 'sd', num: true, value: (d) => d.measured && d.measured.std, render: (d) => (d.measured && asNumber(d.measured.std) !== null ? fmt.um(d.measured.std, 2) : DASH) },
      { key: 'cpk', label: 'Cpk', num: true, value: (d) => d.capability && d.capability.Cpk, render: (d) => (d.capability && d.capability.reported ? `${fixed(d.capability.Cpk, 2)}${d.capability.Cpk_ci ? ` (${fixed(d.capability.Cpk_ci[0], 2)} … ${fixed(d.capability.Cpk_ci[1], 2)})` : ''}` : 'not reported') },
      { key: 'capable', label: 'capable', render: (d) => (d.capability && d.capability.reported ? (d.capability.capable ? h('span', { class: 'badge tone-ok' }, `≥ ${d.capability.target}`) : h('span', { class: 'badge tone-error' }, `< ${d.capability.target}`)) : DASH) },
      { key: 'out', label: 'P(out of tolerance)', num: true, value: (d) => d.out_of_tolerance && d.out_of_tolerance.p, render: (d) => probability(d.out_of_tolerance) },
    ],
    rows: dims, empty: 'The samples give no dimensions (no inspection.json in the sample runs).',
  })));
  const c = chrome();
  for (const d of dims) {
    root.append(lazyDetails(h('span', null, h('b', null, d.id), ` distribution of ${humanize(d.kind)}`), (body) => {
      const values = (d.samples || []).filter((v) => asNumber(v) !== null);
      const small = values.every((v) => Math.abs(v) < 1e-3);
      const scale = small ? 1e6 : 1e3;
      const unit = small ? 'µm' : 'mm';
      const marks = [];
      if (asNumber(d.lower_m) !== null) marks.push({ value: d.lower_m * scale, label: 'lower limit', color: c.error });
      if (asNumber(d.upper_m) !== null) marks.push({ value: d.upper_m * scale, label: 'upper limit', color: c.error });
      if (asNumber(d.nominal_m) !== null) marks.push({ value: d.nominal_m * scale, label: 'nominal', color: c.ink2 });
      if (asNumber(d.nominal_run_m) !== null) marks.push({ value: d.nominal_run_m * scale, label: 'this run', color: c.series[1] });
      body.append(figure({
        title: `${d.id}: measured over ${values.length} samples`, subtitle: unit,
        chart: (b) => new Histogram(b, { xLabel: `measured (${unit})`, unit }).setData({ ...histogram(values.map((v) => v * scale), { bins: Math.max(6, Math.min(30, Math.round(Math.sqrt(values.length) * 2))) }), marks }),
        tableView: () => table({ columns: [{ key: 'i', label: '#', num: true }, { key: 'v', label: `measured (${unit})`, num: true }], rows: values.map((v, i) => ({ i: i + 1, v: num(v * scale, 5) })) }),
      }));
      const m = d.measured || {};
      const cap = d.capability || {};
      body.append(kv([
        ['mean', `${lengthText(m.mean)}${m.mean_ci ? ` (95 %: ${lengthText(m.mean_ci[0])} … ${lengthText(m.mean_ci[1])})` : ''}`], ['standard deviation', m.std !== undefined ? fmt.um(m.std, 3) : DASH],
        ['lowest', lengthText(m.min)], ['highest', lengthText(m.max)],
        ...Object.entries(m.quantiles || {}).map(([q, v]) => [`quantile ${pct(Number(q), 3)}`, lengthText(v)]),
        ['Cp', cap.Cp !== null && cap.Cp !== undefined ? `${fixed(cap.Cp, 3)}${cap.Cp_ci ? ` (${fixed(cap.Cp_ci[0], 2)} … ${fixed(cap.Cp_ci[1], 2)})` : ''}` : DASH],
        ['Cpk', cap.Cpk !== null && cap.Cpk !== undefined ? fixed(cap.Cpk, 3) : DASH], ['Cpk (quantiles)', cap.Cpk_q !== null && cap.Cpk_q !== undefined ? fixed(cap.Cpk_q, 3) : DASH],
        ['expected out of tolerance (normal)', cap.expected_out_frac !== undefined && cap.expected_out_frac !== null ? num(cap.expected_out_frac, 3) : DASH],
      ]));
    }));
  }
  const scalars = results.scalars || [];
  root.append(section('Results per sample', table({
    columns: [{ key: 'name', label: 'result', render: (s) => h('b', null, s.name) }, { key: 'mean', label: 'mean', num: true, render: (s) => valueOf(s.name.replace(/\[.*\]$/, ''), s.mean) },
      { key: 'std', label: 'sd', num: true, render: (s) => valueOf(s.name.replace(/\[.*\]$/, ''), s.std) },
      { key: 'min', label: 'lowest', num: true, render: (s) => valueOf(s.name.replace(/\[.*\]$/, ''), s.min) },
      { key: 'max', label: 'highest', num: true, render: (s) => valueOf(s.name.replace(/\[.*\]$/, ''), s.max) }],
    rows: scalars, empty: 'No scalar results.',
  })));
  const findings = results.findings || {};
  root.append(section('Findings over the samples', table({
    columns: [{ key: 'category', label: 'category', render: (f) => humanize(f.category) }, { key: 'worst', label: 'worst' },
      { key: 'p', label: 'probability', num: true, value: (f) => f.probability && f.probability.p, render: (f) => probability(f.probability) },
      { key: 'mean_count', label: 'mean count', num: true, render: (f) => num(f.mean_count, 3) }],
    rows: findings.by_category || [], empty: 'No sample has a finding.', caption: 'by category',
  }), table({
    columns: [{ key: 'location', label: 'where', render: (f) => {
      const m = /:(\d+)$/.exec(f.location || '');
      return m ? h('button', { type: 'button', class: 'linkish', on: { click: () => { const line = Number(m[1]); const stat = [...data.timeline.lines.values()].find((s) => s.line === line); page.seekTo(stat ? stat.first * data.tickS : 0, { line, program: stat ? stat.program : null }); } } }, f.location) : (f.location || DASH);
    } }, { key: 'category', label: 'category', render: (f) => humanize(f.category) },
    { key: 'p', label: 'probability', num: true, value: (f) => f.probability && f.probability.p, render: (f) => probability(f.probability) }],
    rows: findings.by_location || [], empty: 'No locations.', caption: 'by program location',
  })));
  const sensitivity = results.sensitivity || [];
  const sensBox = section('Sensitivity (rank correlation)');
  root.append(sensBox);
  if (!sensitivity.length) { sensBox.append(notice('No output varies between the samples.')); } else {
    const select = h('select', { class: 'input', 'aria-label': 'Output' }, sensitivity.map((s, i) => h('option', { value: String(i) }, s.output)));
    const body = h('div');
    const show = () => {
      clear(body);
      const s = sensitivity[Number(select.value)];
      const ranking = s.ranking || [];
      body.append(figure({
        title: `What moves ${s.output}`, subtitle: 'Spearman rank correlation; bold: significant',
        chart: (b) => b.append(barList(ranking.map((r) => ({ label: `${r.significant ? '● ' : ''}${r.factor}`, value: r.rho, text: `${fixed(r.rho, 2)}${r.prcc !== null && r.prcc !== undefined ? ` (partial ${fixed(r.prcc, 2)})` : ''}`, color: r.significant ? c.series[0] : c.ink2 })), { signed: true, max: 1, label: 'sensitivity ranking' })),
        tableView: () => table({ columns: [{ key: 'factor', label: 'factor' }, { key: 'rho', label: 'rho', num: true, render: (r) => fixed(r.rho, 3) }, { key: 'p', label: 'p', num: true, render: (r) => num(r.p, 3) },
          { key: 'significant', label: 'significant', render: (r) => (r.significant ? 'yes' : 'no') }, { key: 'prcc', label: 'partial', num: true, render: (r) => (r.prcc === null || r.prcc === undefined ? DASH : fixed(r.prcc, 3)) }], rows: ranking }),
        note: s.prcc_dof !== undefined ? `Partial rank correlations need ${s.prcc_dof} degrees of freedom or more to be reported.` : null,
      }));
      const top = ranking[0];
      const fs = results.factor_samples;
      if (top && fs && Array.isArray(fs.names) && Array.isArray(study.sample_results)) {
        const fi = fs.names.indexOf(top.factor);
        const byIndex = new Map((fs.indexes || []).map((idx, row) => [idx, fs.values[row]]));
        const points = [];
        const dimId = s.kind === 'dimension' ? s.output.replace(/^dimension /, '') : null;
        for (const r of study.sample_results) {
          if (!r.ok) continue;
          const xRow = byIndex.get(r.index);
          if (!xRow || fi < 0) continue;
          let y = null;
          if (dimId && r.dimensions && r.dimensions[dimId]) y = r.dimensions[dimId].measured_m;
          else if (r.scalars && r.scalars[s.output] !== undefined) y = r.scalars[s.output];
          else if (s.kind === 'verdict') y = r.safe ? 0 : 1;
          if (asNumber(y) === null) continue;
          points.push({ x: xRow[fi], y, color: c.series[0], r: 4, tip: [[c.series[0], `${num(xRow[fi])} → ${num(y)}`, `sample ${r.index}`]] });
        }
        if (points.length > 1) {
          body.append(figure({
            title: `${s.output} over ${top.factor}`, subtitle: `${points.length} samples (SI values)`,
            chart: (b) => new XYPlot(b, { xLabel: top.factor, yLabel: s.output, height: 230 }).setData({ points }),
            tableView: () => table({ columns: [{ key: 'x', label: top.factor, num: true, render: (p) => num(p.x) }, { key: 'y', label: s.output, num: true, render: (p) => num(p.y) }], rows: points }),
          }));
        }
      }
    };
    select.addEventListener('change', show);
    sensBox.append(select, body);
    show();
  }
  const failed = results.failed || [];
  if (failed.length) {
    root.append(section('Failed samples', table({ columns: [{ key: 'index', label: 'sample', num: true }, { key: 'error', label: 'error', render: (f) => (typeof f.error === 'string' ? f.error : JSON.stringify(f.error)) }], rows: failed })));
  }
}

function startForm(api, data, onStarted) {
  const runs = h('input', { class: 'input', type: 'number', min: '2', max: '1000', value: '20', 'aria-label': 'Samples', inputmode: 'numeric' });
  const method = h('select', { class: 'input', 'aria-label': 'Sampling plan' }, h('option', { value: 'lhs' }, 'Latin hypercube'), h('option', { value: 'monte_carlo' }, 'Monte Carlo'));
  const seed = h('input', { class: 'input', type: 'number', placeholder: 'project seed', 'aria-label': 'Seed', inputmode: 'numeric' });
  const varySeed = h('input', { type: 'checkbox' });
  const extra = h('textarea', { class: 'input mono', rows: '3', spellcheck: 'false', 'aria-label': 'More options (JSON)', placeholder: '{"vary": {"runout_um": 3}}' });
  const message = h('div');
  const go = button('Start the study', async () => {
    clear(message);
    let options = {};
    try { options = extra.value.trim() ? JSON.parse(extra.value) : {}; } catch (error) { message.append(errorBox(`options: ${error.message}`)); return; }
    options.runs = Number(runs.value) || 20;
    options.method = method.value;
    if (seed.value.trim()) options.seed = Number(seed.value);
    if (varySeed.checked) options.vary_seed = true;
    go.disabled = true;
    try {
      const started = await api.startUncertainty(data.id, options);
      toast(`Study ${started.job.id} queued`, 'info');
      onStarted(started.job);
    } catch (error) {
      message.append(errorBox(error));
    } finally { go.disabled = false; }
  }, { kind: 'primary' });
  return h('form', { class: 'form', on: { submit: (e) => e.preventDefault() } },
    h('div', { class: 'form-row' }, h('label', { class: 'field' }, h('span', null, 'Samples'), runs), h('label', { class: 'field' }, h('span', null, 'Plan'), method),
      h('label', { class: 'field' }, h('span', null, 'Seed'), seed)),
    h('label', { class: 'check' }, varySeed, h('span', null, 'A different simulation seed in every sample (adds the model’s own scatter)')),
    h('label', { class: 'field' }, h('span', null, 'More options (JSON, merged over the project’s uncertainty section)'), extra), go, message);
}

function followJob(api, job, box, onDone) {
  const line = h('p', { class: 'muted' }, `Job ${job.id}: ${job.state}`);
  box.append(line);
  api.waitJob(job.id, {
    onProgress: (record) => {
      const p = record.progress || {};
      line.textContent = `Job ${record.id}: ${record.state}${p.phase ? ` · ${p.phase}` : ''}${p.done !== undefined && p.total ? ` · ${p.done} of ${p.total}` : ''}${p.message ? ` · ${p.message}` : ''}`;
    },
  }).then((result) => {
    line.textContent = `Job ${result.job.id}: ${result.job.state}${result.job.error ? ` · ${result.job.error.message}` : ''}`;
    onDone(result);
  }).catch((error) => box.append(errorBox(error)));
}

export async function renderUncertainty(root, { api, data, page }) {
  const holder = h('div');
  root.append(holder);
  const draw = async () => {
    clear(holder);
    holder.append(loading('Looking for a study…'));
    let found;
    try { found = await findStudy(api, data); } catch (error) { clear(holder); holder.append(errorBox(error)); return; }
    clear(holder);
    if (found.doc) {
      holder.append(h('p', { class: 'muted small' }, `From ${found.source}. ${found.doc.project && found.doc.project.name ? `Project ${found.doc.project.name}.` : ''}`));
      renderStudy(holder, found.doc, { page, data });
    } else {
      holder.append(section('Uncertainty', notice('No uncertainty study has been made around this run yet.')));
    }
    const active = (found.jobs || []).filter((j) => ['queued', 'running', 'cancelling'].includes(j.state));
    const startBox = section('New study');
    for (const job of active) followJob(api, job, startBox, () => draw());
    const simulation = data.record && data.record.job;
    if (!simulation) {
      startBox.append(notice('This run was not started by this server, so its project is not known here: start a study from a run of a project on this server, or with python -m cnctwin.uncertainty.'));
    } else {
      startBox.append(startForm(api, data, (job) => followJob(api, job, startBox, () => draw())));
    }
    holder.append(startBox);
  };
  await draw();
}
