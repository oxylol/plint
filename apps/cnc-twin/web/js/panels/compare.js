/* Comparison of two finished runs (GET /api/compare, cnctwin.report.compare; docs/report.md
 * section 8): verdicts, the headline changes, findings matched / resolved / new, every
 * characteristic, the cycle time by category, line and tool, the tools, energy and cost, surface
 * finish and the input files that differ. Run A is the reference, B the candidate. */

import { h, clear, section, table, notice, errorBox, kv, button, loading, tiles, badge, humanize, verdictBadge, link } from './dom.js';
import { DASH, num, fmt, duration, pct, asNumber } from '../plot/units.js';

const CHANGE = { worse: 'tone-error', better: 'tone-ok', same: 'tone-muted' };

function change(text) {
  return text ? h('span', { class: `badge ${CHANGE[text] || 'tone-info'}` }, text) : DASH;
}

function signedValue(v, format) {
  const n = asNumber(v);
  if (n === null) return DASH;
  return `${n > 0 ? '+' : ''}${format(n)}`;
}

function deltaTable(rows, format, caption, { extra = [] } = {}) {
  return table({
    columns: [{ key: 'name', label: 'what', render: (r) => h('b', null, humanize(r.name)) }, ...extra,
      { key: 'a', label: 'A', num: true, render: (r) => format(r.a) }, { key: 'b', label: 'B', num: true, render: (r) => format(r.b) },
      { key: 'delta', label: 'B − A', num: true, render: (r) => signedValue(r.delta, format) },
      { key: 'relative', label: 'relative', num: true, render: (r) => (asNumber(r.relative) === null ? DASH : `${r.relative > 0 ? '+' : ''}${pct(r.relative, 1)}`) },
      { key: 'change', label: 'change', render: (r) => change(r.change) }],
    rows, caption, empty: 'Nothing to compare.',
  });
}

export function renderComparison(root, doc) {
  const a = doc.a || {};
  const b = doc.b || {};
  root.append(section(null, h('div', { class: 'compare-head' },
    h('div', null, h('span', { class: 'muted' }, 'A '), link(`#/runs/${encodeURIComponent(a.run)}/overview`, a.run || DASH), ' ', verdictBadge({ safe: a.safe }), h('div', { class: 'small muted' }, `${a.name || ''}${a.seed !== undefined ? ` · seed ${a.seed}` : ''} · ${duration(a.cycle_time_s)}`)),
    h('div', null, h('span', { class: 'muted' }, 'B '), link(`#/runs/${encodeURIComponent(b.run)}/overview`, b.run || DASH), ' ', verdictBadge({ safe: b.safe }), h('div', { class: 'small muted' }, `${b.name || ''}${b.seed !== undefined ? ` · seed ${b.seed}` : ''} · ${duration(b.cycle_time_s)}`))),
  doc.verdict && doc.verdict.changed ? notice('The verdict changed.', 'warning') : null,
  h('ul', { class: 'plain headlines' }, (doc.summary || []).map((s) => h('li', null, s)))));
  const findings = doc.findings || {};
  const counts = findings.counts || {};
  root.append(section('Findings', tiles([
    { label: 'Matched', value: num(counts.matched), sub: `${num(counts.worse)} worse, ${num(counts.better)} better` },
    { label: 'Resolved in B', value: num(counts.resolved), tone: counts.resolved ? 'ok' : null },
    { label: 'New in B', value: num(counts.new), tone: counts.new ? 'error' : null },
    { label: 'Total', value: `${num(counts.a)} → ${num(counts.b)}` },
  ]), table({
    columns: [{ key: 'category', label: 'category', render: (m) => humanize(m.category) }, { key: 'line', label: 'where', render: (m) => `${m.op || ''} ${m.program || ''}${m.line ? `:${m.line}` : ''}${m.feature ? ` ${m.feature}` : ''}` },
      { key: 'sev', label: 'severity', sortable: false, render: (m) => h('span', null, badge(m.a && m.a.severity), ' → ', badge(m.b && m.b.severity)) },
      { key: 'severity_change', label: 'change', render: (m) => change(m.severity_change) },
      { key: 'dt_s', label: 'time shift', num: true, render: (m) => signedValue(m.dt_s, (v) => `${num(v)} s`) },
      { key: 'q', label: 'quantities', sortable: false, render: (m) => Object.entries(m.quantities || {}).map(([k, v]) => `${humanize(k)} ${num(v.a)} → ${num(v.b)}`).join('; ') },
      { key: 'matched_by', label: 'matched by' }],
    rows: findings.matched || [], caption: 'In both runs', empty: 'No finding is in both runs.',
  }), table({
    columns: [{ key: 'severity', label: 'severity', render: (f) => badge(f.severity) }, { key: 'category', label: 'category', render: (f) => humanize(f.category) },
      { key: 'line', label: 'where', render: (f) => `${f.op || ''} ${f.program || ''}${f.line ? `:${f.line}` : ''}` }, { key: 'message', label: 'message' }],
    rows: findings.only_a || [], caption: 'Only in A (resolved in B)', empty: 'None.',
  }), table({
    columns: [{ key: 'severity', label: 'severity', render: (f) => badge(f.severity) }, { key: 'category', label: 'category', render: (f) => humanize(f.category) },
      { key: 'line', label: 'where', render: (f) => `${f.op || ''} ${f.program || ''}${f.line ? `:${f.line}` : ''}` }, { key: 'message', label: 'message' }],
    rows: findings.only_b || [], caption: 'Only in B (new)', empty: 'None.',
  })));
  const mmOrUm = (v) => (asNumber(v) === null ? DASH : Math.abs(v) < 1e-3 ? fmt.um(v, 2) : fmt.mm(v, 4));
  root.append(section('Characteristics', table({
    columns: [{ key: 'id', label: 'characteristic', render: (d) => h('b', null, d.id) }, { key: 'kind', label: 'kind', render: (d) => humanize(d.kind) },
      { key: 'a_measured_m', label: 'A', num: true, render: (d) => mmOrUm(d.a_measured_m) }, { key: 'b_measured_m', label: 'B', num: true, render: (d) => mmOrUm(d.b_measured_m) },
      { key: 'delta_m', label: 'B − A', num: true, render: (d) => (asNumber(d.delta_m) === null ? DASH : fmt.signedUm(d.delta_m, 2)) },
      { key: 'use', label: 'tolerance used', num: true, sortable: false, render: (d) => `${pct(d.a_utilisation, 0)} → ${pct(d.b_utilisation, 0)}` },
      { key: 'pass', label: 'result', sortable: false, render: (d) => `${d.a_pass === undefined ? DASH : d.a_pass ? 'pass' : 'fail'} → ${d.b_pass === undefined ? DASH : d.b_pass ? 'pass' : 'fail'}` },
      { key: 'change', label: 'change', render: (d) => change(d.change) }],
    rows: doc.dimensions || [], empty: 'No characteristics in either run.',
  })));
  const cycle = doc.cycle || {};
  const seconds = (v) => (asNumber(v) === null ? DASH : `${num(v)} s`);
  const lines = (cycle.lines || []).slice().sort((x, y) => Math.abs(y.delta || 0) - Math.abs(x.delta || 0)).slice(0, 30);
  root.append(section('Cycle time', cycle.total ? deltaTable([cycle.total], seconds, 'Machine time') : null,
    deltaTable(cycle.categories || [], seconds, 'By category'),
    deltaTable(lines, seconds, 'By program line (largest changes)', { extra: [{ key: 'block', label: 'block', render: (r) => h('code', null, r.block || '') }] }),
    deltaTable(cycle.tools || [], seconds, 'By tool')));
  const toolRows = [];
  for (const t of doc.tools || []) {
    for (const [key, value] of Object.entries(t)) {
      if (value && typeof value === 'object' && 'a' in value && key !== 'broken') toolRows.push({ ...value, name: `T${t.number} ${value.name || humanize(key)}`, key });
    }
    if (t.broken) toolRows.push({ name: `T${t.number} broken`, a: t.broken.a, b: t.broken.b, change: t.broken.a === t.broken.b ? 'same' : t.broken.b ? 'worse' : 'better', key: 'broken' });
  }
  root.append(section('Tools', table({
    columns: [{ key: 'name', label: 'what', render: (r) => h('b', null, r.name) },
      { key: 'a', label: 'A', num: true, render: (r) => (typeof r.a === 'boolean' ? (r.a ? 'yes' : 'no') : num(r.a)) },
      { key: 'b', label: 'B', num: true, render: (r) => (typeof r.b === 'boolean' ? (r.b ? 'yes' : 'no') : num(r.b)) },
      { key: 'delta', label: 'B − A', num: true, render: (r) => signedValue(r.delta, num) }, { key: 'change', label: 'change', render: (r) => change(r.change) }],
    rows: toolRows, empty: 'No tools to compare.', caption: 'SI values (VB in m, times in s, temperatures in K, cost in EUR)',
  })));
  const energy = doc.energy || {};
  root.append(section('Energy and cost per part', deltaTable([energy.total_J, energy.tool_cost_eur, energy.coolant_cost_eur, energy.takt_s, energy.parts_per_hour].filter(Boolean), num),
    deltaTable(energy.consumers || [], (v) => fmt.kJ(v), 'By consumer')));
  root.append(section('Surface finish', table({
    columns: [{ key: 'name', label: 'feature', render: (r) => h('b', null, r.name) }, { key: 'a', label: 'Ra A', num: true, render: (r) => fmt.um(r.a, 2) }, { key: 'b', label: 'Ra B', num: true, render: (r) => fmt.um(r.b, 2) },
      { key: 'delta', label: 'B − A', num: true, render: (r) => (asNumber(r.delta) === null ? DASH : fmt.signedUm(r.delta, 2)) }, { key: 'spec', label: 'allowed', num: true, render: (r) => fmt.um(r.spec_Ra_m, 2) },
      { key: 'pass', label: 'result', render: (r) => `${r.a_pass ? 'pass' : 'fail'} → ${r.b_pass ? 'pass' : 'fail'}` }, { key: 'change', label: 'change', render: (r) => change(r.change) }],
    rows: doc.surface || [], empty: 'No surface features.',
  })));
  root.append(section('Inputs', table({
    columns: [{ key: 'file', label: 'file' }, { key: 'state', label: 'state', render: (r) => (r.state === 'same' ? h('span', { class: 'badge tone-muted' }, 'same') : h('span', { class: 'badge tone-warning' }, r.state)) },
      { key: 'a_sha256', label: 'sha256 A', render: (r) => h('code', { class: 'hash' }, r.a_sha256 || DASH) }, { key: 'b_sha256', label: 'sha256 B', render: (r) => h('code', { class: 'hash' }, r.b_sha256 || DASH) }],
    rows: doc.inputs || [], empty: 'No inputs listed.',
  })));
  const missing = doc.missing || {};
  if ((missing.a || []).length || (missing.b || []).length) {
    root.append(section('Not in the run directories', kv([['A', (missing.a || []).map((m) => (typeof m === 'string' ? m : `${m.item}: ${m.reason}`)).join('; ') || 'nothing'],
      ['B', (missing.b || []).map((m) => (typeof m === 'string' ? m : `${m.item}: ${m.reason}`)).join('; ') || 'nothing']])));
  }
  if (doc.parameters) root.append(section('Thresholds', kv(Object.entries(doc.parameters).map(([k, v]) => [humanize(k), num(v)]))));
}

export async function renderCompare(root, { api, query, navigate }) {
  root.append(h('div', { class: 'page-head' }, h('h1', null, 'Compare two runs')));
  const pickA = h('select', { class: 'input', 'aria-label': 'Run A (reference)' });
  const pickB = h('select', { class: 'input', 'aria-label': 'Run B (candidate)' });
  const out = h('div');
  let runs = [];
  try { runs = ((await api.runs({ limit: 1000 })).runs || []).filter((r) => r.verdict); } catch (error) { root.append(errorBox(error)); return; }
  for (const select of [pickA, pickB]) {
    select.append(h('option', { value: '' }, 'choose a finished run'));
    for (const r of runs) select.append(h('option', { value: r.id }, `${r.id}${r.name ? ` · ${r.name}` : ''}${r.verdict ? (r.verdict.safe ? ' · safe' : ' · not safe') : ''}`));
  }
  pickA.value = query.a || '';
  pickB.value = query.b || '';
  const go = () => {
    if (!pickA.value || !pickB.value) return;
    navigate(`#/compare?a=${encodeURIComponent(pickA.value)}&b=${encodeURIComponent(pickB.value)}`);
  };
  root.append(h('div', { class: 'filters' }, h('label', { class: 'field' }, h('span', null, 'A (reference)'), pickA), h('label', { class: 'field' }, h('span', null, 'B (candidate)'), pickB),
    button('Swap', () => { const v = pickA.value; pickA.value = pickB.value; pickB.value = v; go(); }, { kind: 'small' }),
    button('Compare', go, { kind: 'primary', iconName: 'compare' })), out);
  if (!runs.length) { out.append(notice('There are no finished runs yet.')); return; }
  if (!query.a || !query.b) { out.append(notice('Pick two finished runs.')); return; }
  out.append(loading('Comparing…'));
  try {
    let result = await api.compare(query.a, query.b, { waitS: 20 });
    if (result.status === 202) {
      clear(out);
      const line = h('p', { class: 'muted' }, `Comparison job ${result.job.id}…`);
      out.append(line);
      await api.waitJob(result.job.id, { onProgress: (j) => { line.textContent = `Comparison job ${j.id}: ${j.state}`; } });
      result = await api.compare(query.a, query.b, { waitS: 5 });
    }
    clear(out);
    if (!result.result) { out.append(errorBox(result.detail || 'the comparison gave no result')); return; }
    renderComparison(out, result.result);
  } catch (error) {
    clear(out);
    out.append(errorBox(error, 'Compare: '));
  }
}
