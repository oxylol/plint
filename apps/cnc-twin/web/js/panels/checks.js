/* Checks and identity of a run: the conservation balances (mass, energy, time, thermal solvers,
 * heat partition: largest and final residual against the tolerance), the time check of every path
 * from timeline.json in integer ticks, the ledger channels; and the run identity (project, seed,
 * clock and L0 strides, versions, thread settings, every input file with its sha256, the estimated
 * database values and the model approximations the run lists). */

import { h, section, table, notice, kv, link, humanize, jsonView, lazyDetails } from './dom.js';
import { PlotGroup, TimePlot } from '../plot/timeplot.js';
import { buildSeries } from './plots.js';
import { DASH, num, bytes, duration, quantity } from '../plot/units.js';

function leaf(key, value) {
  if (value === null || value === undefined) return DASH;
  if (Array.isArray(value)) return value.every((v) => v === null || typeof v !== 'object') ? quantity(key, value).text : JSON.stringify(value);
  if (typeof value === 'object') return JSON.stringify(value);
  return quantity(key, value).text;
}

/** Any object of run.json in display units: scalars by their key's unit, nested objects as
 * disclosures, lists of objects as tables. */
export function objectView(value, { depth = 0, skip = [] } = {}) {
  if (value === null || typeof value !== 'object') return h('span', null, String(value));
  if (Array.isArray(value)) {
    if (!value.length) return notice('none');
    if (value.every((v) => v && typeof v === 'object' && !Array.isArray(v))) {
      const keys = [...new Set(value.flatMap((v) => Object.keys(v)))].filter((k) => !skip.includes(k));
      return table({ columns: keys.map((k) => ({ key: k, label: quantity(k, 0).label, render: (row) => leaf(k, row[k]) })), rows: value, maxRows: 200 });
    }
    return h('span', null, leaf('', value));
  }
  const pairs = [];
  const nested = [];
  for (const [key, v] of Object.entries(value)) {
    if (skip.includes(key)) continue;
    if (v && typeof v === 'object' && !(Array.isArray(v) && v.every((x) => x === null || typeof x !== 'object'))) nested.push([key, v]);
    else pairs.push([quantity(key, 0).label, leaf(key, v)]);
  }
  return h('div', { class: 'object-view' }, pairs.length ? kv(pairs) : null,
    nested.map(([key, v]) => (depth < 4 ? lazyDetails(humanize(key), (body) => body.append(objectView(v, { depth: depth + 1, skip })), { open: depth === 0 && nested.length <= 3 }) : h('pre', { class: 'json' }, JSON.stringify(v, null, 1)))));
}

export async function renderChecks(root, { api, data, page }) {
  const run = data.run || {};
  const conservation = (run.summary && run.summary.conservation) || run.conservation || null;
  if (conservation) {
    const rows = Object.entries(conservation).filter(([, v]) => v && typeof v === 'object').map(([name, v]) => ({ name, ...v }));
    root.append(section('Conservation (architecture 12)', table({
      columns: [
        { key: 'name', label: 'balance', render: (r) => h('b', null, humanize(r.name)) },
        { key: 'max_residual_rel', label: 'largest residual', num: true, render: (r) => (r.residual_ticks !== undefined && r.max_residual_rel === undefined ? `${num(r.residual_ticks)} ticks` : num(r.max_residual_rel, 3)) },
        { key: 'final_residual_rel', label: 'final residual', num: true, render: (r) => num(r.final_residual_rel, 3) },
        { key: 'tolerance_rel', label: 'tolerance', num: true, render: (r) => num(r.tolerance_rel, 3) },
        { key: 'checked_steps', label: 'steps checked', num: true, render: (r) => num(r.checked_steps) },
        { key: 'ok', label: 'result', render: (r) => (r.ok === true ? h('span', { class: 'badge tone-ok' }, 'holds') : r.ok === false ? h('span', { class: 'badge tone-error' }, 'violated') : h('span', { class: 'badge tone-muted' }, r.enforced === false ? 'reported only' : 'not checked')) },
        { key: 'note', label: 'note', render: (r) => r.note || '' },
      ],
      rows,
    })));
  } else {
    root.append(section('Conservation', notice(data.finished ? 'run.json records no conservation summary.' : 'The conservation summary is written when the run ends.')));
  }
  const rows = [];
  for (const [path, entry] of data.timeline.paths) {
    const blocks = entry.blocks;
    if (!blocks.length) continue;
    const first = blocks[0].tick0;
    const last = blocks.reduce((m, b) => Math.max(m, b.tick1), 0);
    let busy = 0;
    let booked = 0;
    let over = 0;
    for (const b of blocks) {
      const d = b.tick1 - b.tick0;
      busy += d;
      const ticks = b.ticks ? Object.values(b.ticks).reduce((a, x) => a + x, 0) : d;
      booked += ticks;
      if (ticks > d) over++;
    }
    rows.push({ path, blocks: blocks.length, span: last - first, busy, gaps: last - first - busy, booked, over });
  }
  root.append(section('Time (from timeline.json, integer ticks)', table({
    columns: [{ key: 'path', label: 'path', render: (r) => `path ${r.path + 1}` }, { key: 'blocks', label: 'blocks', num: true },
      { key: 'span', label: 'span (ticks)', num: true, render: (r) => String(r.span) }, { key: 'busy', label: 'block time (ticks)', num: true, render: (r) => String(r.busy) },
      { key: 'gaps', label: 'gaps (ticks)', num: true, render: (r) => String(r.gaps) },
      { key: 'booked', label: 'booked by category', num: true, render: (r) => String(r.booked) },
      { key: 'over', label: 'over-booked blocks', num: true, render: (r) => String(r.over) },
      { key: 'span_s', label: 'span', num: true, render: (r) => duration(r.span * data.tickS) }],
    rows, empty: 'No timeline.',
  }), notice('A path’s block times plus its gaps (waits) equal its span exactly; categories never book more than a block lasts.')));
  const ledger = data.channels.filter((c) => /^ledger\.|consistency|residual/.test(c.name));
  const box = section('Balance channels');
  root.append(box);
  if (!ledger.length) { box.append(notice('No ledger.*, *consistency* or *residual* channels are recorded.')); return; }
  const group = new PlotGroup({ duration: Math.max(data.duration(), 1e-3), onSeek: (t) => page.seekTo(t) });
  for (const channel of ledger) {
    const plot = new TimePlot(box, { group, height: 120, title: channel.name });
    try {
      const doc = await api.series(data.id, { tier: channel.tier === 'fast' ? 'tick' : channel.tier, channels: [channel.name], maxPoints: 2400 });
      const built = buildSeries(doc, channel, { buckets: 600 });
      plot.setTitle(channel.name, built.unit.label);
      plot.setNote(channel.doc || '');
      plot.setSeries(built.series);
    } catch (error) {
      plot.setNote(`Could not load: ${error.detail || error.message}`);
      plot.setSeries([]);
    }
  }
}

export async function renderInfo(root, { data }) {
  const run = data.run || {};
  const record = data.record || {};
  const job = record.job || {};
  const clock = run.clock || {};
  const project = data.project;
  root.append(section('Run', kv([
    ['run', data.id], ['project', data.projectId ? link(`#/projects/${encodeURIComponent(data.projectId)}`, `${project.name || data.projectId}`) : (project.name || DASH)],
    ['description', project.description || DASH], ['seed', run.seed !== undefined ? String(run.seed) : DASH],
    ['format', run.format ? `${run.format} ${run.version}` : DASH], ['complete', run.complete === undefined ? DASH : run.complete ? 'yes' : 'no'],
    ['project file', run.project_file || DASH], ['project folder', run.project_dir || DASH], ['project hash', run.project_hash || DASH],
    ['numba cache key', run.numba_cache_key || DASH], ['status', data.state], ['job', job.id ? `${job.id} (attempt ${job.attempt}, ${job.memory_mode || 'no'} memory cap${job.memory_bytes ? ` ${bytes(job.memory_bytes)}` : ''})` : 'not started by this server'],
  ])));
  root.append(section('Clock', kv([
    ['base tick', clock.tick_s !== undefined ? `${num(clock.tick_s)} s` : DASH], ['L1 tick', clock.ipo_ticks !== undefined ? `${clock.ipo_ticks} base ticks` : DASH],
    ['macro step', clock.macro_ipo !== undefined ? `${clock.macro_ipo} L1 ticks` : DASH],
    ['L0 strides', clock.l0_ticks !== undefined ? (typeof clock.l0_ticks === 'object' ? Object.entries(clock.l0_ticks).map(([op, v]) => `${op}: ${typeof v === 'object' ? JSON.stringify(v) : `${v} base ticks`}`).join('; ') : String(clock.l0_ticks)) : DASH],
  ])));
  root.append(section('Versions and threads', kv([
    ...Object.entries(run.versions || {}).map(([k, v]) => [k, String(v)]),
    ...Object.entries(run.threads || {}).map(([k, v]) => [k, String(v)]),
    ...Object.entries(run.platform || {}).map(([k, v]) => [k, String(v)]),
  ])));
  const inputs = Object.entries(run.inputs || {}).map(([file, v]) => ({ file, bytes: v.bytes, sha256: v.sha256 }));
  root.append(section('Inputs', table({
    columns: [{ key: 'file', label: 'file' }, { key: 'bytes', label: 'size', num: true, render: (r) => bytes(r.bytes) }, { key: 'sha256', label: 'sha256', render: (r) => h('code', { class: 'hash' }, r.sha256 || DASH) }],
    rows: inputs, empty: data.finished ? 'run.json lists no inputs.' : 'Listed in run.json when the run ends.',
  })));
  if (run.options && Object.keys(run.options).length) root.append(section('Options', jsonView(run.options)));
  const estimates = run.estimates || run.db_estimates || [];
  root.append(section('Estimated database values used', table({
    columns: [{ key: 'obj', label: 'object' }, { key: 'file', label: 'file' }, { key: 'line', label: 'line', num: true }, { key: 'path', label: 'key' },
      { key: 'value', label: 'value', num: true, render: (r) => (typeof r.value === 'number' ? num(r.value) : String(r.value)) }, { key: 'note', label: 'how it was estimated', render: (r) => r.note || '(no note)' }],
    rows: estimates, empty: 'run.json lists no estimated values.',
  })));
  const approximations = run.approximations || [];
  root.append(section('Model approximations', table({
    columns: [{ key: 'domain', label: 'domain' }, { key: 'approximation', label: 'approximation' }, { key: 'error', label: 'error' }],
    rows: approximations, empty: 'run.json lists none here: the report lists the approximations of docs/approximations.md that concern this run.',
  })));
  const tools = (project.tools && project.tools.list) || [];
  root.append(section('Setup', project.machine ? h('div', null,
    h('h3', null, 'Machine'), objectView(project.machine, { depth: 1 }),
    h('h3', null, 'Stock and material'), objectView({ stock: project.stock || null, material: project.material || null }, { depth: 1 }),
    h('h3', null, `Tools (${tools.length})`), objectView(tools, { depth: 1 }),
    h('h3', null, 'Operations'), objectView(project.operations || [], { depth: 1, skip: ['program_text', 'program_texts'] }),
    h('h3', null, 'Coolant'), objectView(project.coolant || {}, { depth: 1 }))
    : notice('The resolved project is written into run.json when the run ends.')));
  const known = new Set(['machine', 'stock', 'material', 'tools', 'operations', 'coolant', 'name', 'description']);
  const rest = Object.fromEntries(Object.entries(project).filter(([k]) => !known.has(k)));
  if (Object.keys(rest).length) root.append(section('Rest of the resolved project', objectView(rest, { depth: 1 })));
  root.append(section(`Channels (${data.channels.length})`, table({
    columns: [{ key: 'name', label: 'channel', render: (c) => h('code', null, c.name) }, { key: 'tier', label: 'tier' }, { key: 'unit', label: 'SI unit' },
      { key: 'max_shape', label: 'columns', render: (c) => (Array.isArray(c.max_shape) && c.max_shape.length ? c.max_shape.join(' × ') : '1') },
      { key: 'domain', label: 'owner' }, { key: 'dtype', label: 'type' }, { key: 'doc', label: 'what it is' }],
    rows: data.channels, empty: 'channels.json lists no channels.', caption: 'Plotted on the Playback tab (statistics in each plot’s table)',
  })));
  root.append(section('Raw', lazyDetails('run.json and status.json as written', (body) => body.append(jsonView({ run: data.run, status: data.status })))));
}
