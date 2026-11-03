/* Result panels of the finished part: inspection (the CMM table of every characteristic, probed
 * points, error budget, surface deviation, threads, part temperature at unload), surface finish
 * and burrs, and the tools (wear, life, cost, wear channels). */

import { h, append, section, table, notice, errorBox, figure, kv, lazyDetails, badge, humanize, tiles } from './dom.js';
import { attributeValues, stats } from './fields.js';
import { Histogram } from '../plot/charts.js';
import { barList, meter } from '../plot/bars.js';
import { histogram } from '../plot/scale.js';
import { PlotGroup, TimePlot } from '../plot/timeplot.js';
import { buildSeries, componentLabels } from './plots.js';
import { categorical } from '../plot/colormap.js';
import { DASH, num, fmt, fixed, duration, pct, quantity, asNumber } from '../plot/units.js';

async function load(fn, data, file) {
  if (!data.has(file)) return { missing: true };
  try { return { doc: await fn() }; } catch (error) { return { error }; }
}

function absent(name, result) {
  if (result.error) return errorBox(result.error, `${name}: `);
  return notice(`This run directory has no ${name} (not recorded, or the run has not finished).`);
}

/** Buttons that jump to the first execution of each program line. */
export function lineLinks(page, data, program, lines) {
  const list = (Array.isArray(lines) ? lines : [lines]).filter((l) => Number.isFinite(Number(l)) && Number(l) > 0);
  if (!list.length) return DASH;
  return h('span', { class: 'line-links' }, list.map((line) => h('button', {
    type: 'button', class: 'linkish',
    on: { click: () => {
      const stat = data.lineStat(program, Number(line)) || [...data.timeline.lines.values()].find((s) => s.line === Number(line));
      page.seekTo(stat ? stat.first * data.tickS : 0, { program: stat ? stat.program : program, line: Number(line) });
    } },
  }, `line ${line}`)));
}

// ============================================================================ inspection

function limits(t) {
  const nominal = asNumber(t.nominal_m);
  if (asNumber(t.lower_m) !== null || asNumber(t.upper_m) !== null) return { kind: 'size', nominal, lower: asNumber(t.lower_m), upper: asNumber(t.upper_m) };
  if (nominal !== null && (asNumber(t.plus_m) !== null || asNumber(t.minus_m) !== null)) {
    return { kind: 'size', nominal, lower: nominal - Math.abs(asNumber(t.minus_m) || 0), upper: nominal + Math.abs(asNumber(t.plus_m) || 0) };
  }
  if (asNumber(t.tolerance_m) !== null) return { kind: 'zone', nominal: null, lower: 0, upper: asNumber(t.tolerance_m) };
  return { kind: 'unknown', nominal, lower: null, upper: null };
}

export function judge(t) {
  const lim = limits(t);
  const measured = asNumber(t.measured_m);
  let deviation = asNumber(t.deviation_m);
  if (deviation === null && measured !== null && lim.nominal !== null) deviation = measured - lim.nominal;
  let used = null;
  let out = null;
  if (measured !== null && lim.kind === 'size' && lim.nominal !== null) {
    const allowed = deviation >= 0 ? lim.upper - lim.nominal : lim.nominal - lim.lower;
    used = allowed > 0 ? Math.abs(deviation) / allowed : null;
    out = Math.max(0, measured - lim.upper, lim.lower - measured);
  } else if (measured !== null && lim.kind === 'size') {
    out = Math.max(0, lim.upper !== null ? measured - lim.upper : 0, lim.lower !== null ? lim.lower - measured : 0);
  } else if (measured !== null && lim.kind === 'zone') {
    used = lim.upper > 0 ? measured / lim.upper : null;
    out = Math.max(0, measured - lim.upper);
  }
  const pass = typeof t.pass === 'boolean' ? t.pass : (out === null ? null : out <= 0);
  return { ...lim, measured, deviation, used, out, pass };
}

function passBadge(pass) {
  if (pass === null || pass === undefined) return h('span', { class: 'badge tone-muted' }, 'not judged');
  return pass ? h('span', { class: 'badge tone-ok' }, 'pass') : h('span', { class: 'badge tone-error' }, 'fail');
}

function threadDetail(t) {
  const part = (name) => {
    const v = t[`${name}_m`];
    if (v && typeof v === 'object') return `${fmt.mm(v.measured, 4)} (limits ${fmt.mm(v.min, 4)} … ${fmt.mm(v.max, 4)})`;
    return fmt.mm(v, 4);
  };
  return kv([['designation', t.designation || DASH], ['class', t.class || DASH], ['pitch diameter', part('pitch_diameter')],
    ['major diameter', part('major_diameter')], ['minor diameter', part('minor_diameter')],
    ['gauges', t.gauges ? Object.entries(t.gauges).map(([g, ok]) => `${g}: ${ok ? 'accepts' : 'rejects'}`).join(', ') : DASH]]);
}

function characteristicDetail(t, r) {
  const body = h('div', { class: 'char-detail' });
  const known = new Set(['id', 'kind', 'nominal_m', 'plus_m', 'minus_m', 'lower_m', 'upper_m', 'tolerance_m', 'measured_m', 'deviation_m', 'pass',
    'feature', 'op', 'program', 'lines', 'line', 'tools', 'datums', 'fit', 'size', 'method', 'points', 'contributions_m']);
  body.append(kv([['feature', t.feature || DASH], ['method', t.method || DASH], ['datums', (t.datums || []).join(', ') || DASH],
    ['tools', (t.tools || []).map((n) => `T${n}`).join(', ') || DASH], ['fit', t.fit || (t.size && t.size.fit) || DASH]]));
  if (t.kind === 'thread') body.append(threadDetail(t));
  const points = Array.isArray(t.points) ? t.points : [];
  if (points.length) {
    const devs = points.map((p) => asNumber(p.deviation_m)).filter((v) => v !== null);
    const s = stats(devs);
    append(body, [h('h4', null, `Probed points (${points.length})`),
      s ? kv([['mean', fmt.signedUm(s.mean, 2)], ['standard deviation', fmt.um(s.std, 2)], ['lowest', fmt.signedUm(s.min, 2)],
        ['highest', fmt.signedUm(s.max, 2)], ['range (form seen by the probe)', fmt.um(s.range, 2)]]) : null,
      table({
        columns: [{ key: 'i', label: '#', num: true }, { key: 'x', label: 'x (mm)', num: true }, { key: 'y', label: 'y (mm)', num: true },
          { key: 'z', label: 'z (mm)', num: true }, { key: 'd', label: 'deviation (µm)', num: true }],
        rows: points.map((p, i) => ({ i: i + 1, x: fixed(p.x_S_m * 1e3, 3), y: fixed(p.y_S_m * 1e3, 3), z: fixed(p.z_S_m * 1e3, 3), d: fixed(p.deviation_m * 1e6, 2) })),
        maxRows: 50,
      })]);
  }
  const budget = Array.isArray(t.contributions_m) ? t.contributions_m : [];
  if (budget.length) {
    const total = budget.reduce((a, b) => a + (asNumber(b.value_m) || 0), 0);
    const reference = r.kind === 'zone' ? r.measured : r.deviation;
    body.append(h('h4', null, 'Error budget'),
      barList(budget.map((b, i) => ({ label: `${humanize(b.source)}${b.tool ? ` (T${b.tool})` : ''}`, value: (asNumber(b.value_m) || 0) * 1e6,
        text: fmt.signedUm(b.value_m, 2), color: categorical(i) })), { signed: true, label: 'error budget' }),
      h('p', { class: 'small' }, `Sum of the contributions ${fmt.signedUm(total, 2)}; ${r.kind === 'zone' ? 'measured' : 'deviation'} ${fmt.signedUm(reference, 2)}.`));
  }
  const rest = Object.entries(t).filter(([key]) => !known.has(key));
  if (rest.length) body.append(h('h4', null, 'Other fields'), kv(rest.map(([key, value]) => [quantity(key, value).label, typeof value === 'object' ? JSON.stringify(value) : quantity(key, value).text])));
  return body;
}

export async function renderInspection(root, { api, data, page }) {
  const result = await load(() => api.inspection(data.id), data, 'inspection.json');
  if (!result.doc) { root.append(section('Inspection', absent('inspection.json', result))); return; }
  const doc = result.doc;
  const items = Array.isArray(doc.tolerances) ? doc.tolerances : [];
  const rows = items.map((t) => ({ t, r: judge(t) }));
  const failed = rows.filter((x) => x.r.pass === false).length;
  root.append(section(null, tiles([
    { label: 'Characteristics', value: String(items.length) },
    { label: 'Failed', value: String(failed), tone: failed ? 'error' : 'ok' },
    { label: 'Inspected at', value: doc.temperature_K !== undefined ? fmt.degC(doc.temperature_K) : DASH, sub: 'part temperature of the measurement' },
  ])));
  const mmOrUm = (v) => (v === null ? DASH : Math.abs(v) < 1e-3 ? fmt.um(v, 2) : fmt.mm(v, 4));
  root.append(section('Characteristics (CMM table)', table({
    columns: [
      { key: 'id', label: 'characteristic', value: (x) => x.t.id, render: (x) => h('b', null, x.t.id || DASH) },
      { key: 'kind', label: 'kind', value: (x) => x.t.kind, render: (x) => `${humanize(x.t.kind)}${x.t.fit || (x.t.size && x.t.size.fit) ? ` ${x.t.fit || x.t.size.fit}` : ''}${(x.t.datums || []).length ? ` | ${x.t.datums.join(' | ')}` : ''}` },
      { key: 'nominal', label: 'nominal', num: true, value: (x) => x.r.nominal, render: (x) => (x.r.kind === 'zone' ? 'zone' : mmOrUm(x.r.nominal)) },
      { key: 'limits', label: 'limits', num: true, sortable: false, render: (x) => (x.r.kind === 'zone' ? `≤ ${fmt.um(x.r.upper, 1)}` : x.r.lower !== null ? `${mmOrUm(x.r.lower)} … ${mmOrUm(x.r.upper)}` : DASH) },
      { key: 'measured', label: 'measured', num: true, value: (x) => x.r.measured, render: (x) => mmOrUm(x.r.measured) },
      { key: 'deviation', label: 'deviation', num: true, value: (x) => x.r.deviation, render: (x) => (x.r.deviation === null ? DASH : fmt.signedUm(x.r.deviation, 1)) },
      { key: 'out', label: 'out of tolerance', num: true, value: (x) => x.r.out, render: (x) => (x.r.out ? fmt.um(x.r.out, 1) : x.r.out === 0 ? '0' : DASH) },
      { key: 'used', label: 'tolerance used', num: true, value: (x) => x.r.used, render: (x) => (x.r.used === null ? DASH : meter(x.r.used, { text: pct(x.r.used, 0), tone: x.r.used > 1 ? 'bad' : x.r.used > 0.8 ? 'warn' : 'ok', label: 'tolerance used' })) },
      { key: 'pass', label: 'result', value: (x) => x.r.pass, render: (x) => passBadge(x.r.pass) },
      { key: 'lines', label: 'made by', sortable: false, render: (x) => lineLinks(page, data, x.t.program, x.t.lines || x.t.line) },
    ],
    rows,
    empty: 'inspection.json lists no characteristics.',
    rowClass: (x) => (x.r.pass === false ? 'row-bad' : null),
  })));
  const detailBox = section('Each characteristic');
  for (const { t, r } of rows) detailBox.append(lazyDetails(h('span', null, passBadge(r.pass), ' ', h('b', null, t.id || DASH), ` ${t.feature || humanize(t.kind)}`), (body) => body.append(characteristicDetail(t, r))));
  if (!rows.length) detailBox.append(notice('Nothing to show.'));
  root.append(detailBox);
  const deviation = section('Deviation of the machined surface');
  root.append(deviation);
  if ([...data.files.keys()].some((p) => p.startsWith('keyframes/attr/'))) {
    deviation.append(lazyDetails('Per-vertex deviation (final surface)', async (body) => {
      const { values, files, of } = await attributeValues(api, data, 'deviation_m', 64);
      if (!values.length) { body.append(notice('The final attributes record no deviation_m.')); return; }
      const um = values.map((v) => v * 1e6);
      const s = stats(um);
      body.append(figure({
        title: 'Deviation per vertex', subtitle: `${num(s.n)} vertices of ${files} of ${of} bricks`,
        chart: (box) => new Histogram(box, { xLabel: 'deviation (µm)', unit: 'µm' }).setData({ ...histogram(um), marks: [{ value: 0, label: 'nominal' }] }),
        tableView: () => kv([['vertices', num(s.n)], ['mean', `${num(s.mean)} µm`], ['standard deviation', `${num(s.std)} µm`], ['lowest', `${num(s.min)} µm`], ['highest', `${num(s.max)} µm`]]),
        note: 'Per mesh vertex, not weighted by area.',
      }));
    }, { open: true }));
  } else {
    deviation.append(notice('The run directory has no final per-vertex attributes (keyframes/attr/).'));
  }
  const unload = Array.isArray(doc.unload) ? doc.unload : [];
  root.append(section('Part temperature at unload', table({
    columns: [{ key: 'op', label: 'operation' }, { key: 'time_s', label: 'time', num: true, render: (u) => `${num(u.time_s)} s` },
      { key: 'mean_temp_K', label: 'mean', num: true, render: (u) => fmt.degC(u.mean_temp_K) }, { key: 'max_temp_K', label: 'highest', num: true, render: (u) => fmt.degC(u.max_temp_K) }],
    rows: unload, empty: 'inspection.json records no unload temperatures (see the Thermal tab for the work temperature channels).',
  })));
}

// ============================================================================ surface

export async function renderSurface(root, { api, data, page }) {
  const result = await load(() => api.surface(data.id), data, 'surface.json');
  if (!result.doc) { root.append(section('Surface finish and burrs', absent('surface.json', result))); return; }
  const doc = result.doc;
  const features = Array.isArray(doc.features) ? doc.features : [];
  root.append(section('Surface finish (ISO 4287)', table({
    columns: [
      { key: 'id', label: 'feature', render: (f) => h('b', null, f.id || DASH) },
      { key: 'op', label: 'operation' },
      { key: 'tool', label: 'tool', render: (f) => (f.tool !== undefined ? `T${f.tool}` : DASH) },
      { key: 'Ra_m', label: 'Ra', num: true, render: (f) => fmt.um(f.Ra_m, 2) },
      { key: 'spec_Ra_m', label: 'Ra allowed', num: true, render: (f) => fmt.um(f.spec_Ra_m, 2) },
      { key: 'Rz_m', label: 'Rz', num: true, render: (f) => fmt.um(f.Rz_m, 2) },
      { key: 'spec_Rz_m', label: 'Rz allowed', num: true, render: (f) => fmt.um(f.spec_Rz_m, 2) },
      { key: 'pass', label: 'result', render: (f) => passBadge(typeof f.pass === 'boolean' ? f.pass : null) },
      { key: 'lines', label: 'made by', sortable: false, render: (f) => lineLinks(page, data, f.program, f.lines || f.line) },
    ],
    rows: features, empty: 'surface.json lists no features.',
  })));
  const dist = section('Ra over the surface');
  root.append(dist);
  dist.append(lazyDetails('Per-vertex Ra (final surface)', async (body) => {
    const { values, files, of } = await attributeValues(api, data, 'Ra_m', 64);
    if (!values.length) { body.append(notice('The final attributes record no Ra_m.')); return; }
    const um = values.map((v) => v * 1e6);
    const s = stats(um);
    const marks = features.filter((f) => asNumber(f.spec_Ra_m) !== null).slice(0, 2).map((f) => ({ value: f.spec_Ra_m * 1e6, label: `${f.id} limit` }));
    body.append(figure({
      title: 'Ra per vertex', subtitle: `${num(s.n)} vertices of ${files} of ${of} bricks`,
      chart: (box) => new Histogram(box, { xLabel: 'Ra (µm)', unit: 'µm' }).setData({ ...histogram(um), marks }),
      tableView: () => kv([['vertices', num(s.n)], ['mean', `${num(s.mean)} µm`], ['standard deviation', `${num(s.std)} µm`], ['lowest', `${num(s.min)} µm`], ['highest', `${num(s.max)} µm`]]),
    }));
  }, { open: true }));
  const burrs = Array.isArray(doc.burrs) ? doc.burrs : [];
  root.append(section('Burrs', table({
    columns: [
      { key: 'edge', label: 'edge', render: (b) => h('b', null, b.edge || DASH) },
      { key: 'kind', label: 'kind', render: (b) => humanize(b.kind) },
      { key: 'height_m', label: 'height', num: true, render: (b) => fmt.um(b.height_m, 1) },
      { key: 'root_thickness_m', label: 'root thickness', num: true, render: (b) => fmt.um(b.root_thickness_m, 1) },
      { key: 'op', label: 'operation' },
      { key: 'line', label: 'made by', sortable: false, render: (b) => lineLinks(page, data, b.program, b.line) },
      { key: 'location_S_m', label: 'where (mm, stock frame)', sortable: false, render: (b) => (Array.isArray(b.location_S_m) ? b.location_S_m.map((v) => fixed(v * 1e3, 2)).join(', ') : DASH) },
    ],
    rows: burrs, empty: 'No burrs recorded.',
  })));
}

// ============================================================================ tools

const TOOL_CATEGORIES = new Set(['tool_breakage', 'tool_end_of_life', 'tool_pullout', 'unbalance', 'overspeed', 'spindle_overload']);

/** Peak deflection per tool and component from the dynamics channels in metres whose name says
 * deflection or displacement (docs/report.md section 4): every sample is given to the tool of the
 * path-0 timeline block it falls in; the bound sqrt(sum of the component peaks squared) is an upper
 * bound of the vector magnitude (the peaks need not coincide). */
export async function deflectionByTool(api, data) {
  const channels = data.channels.filter((c) => (c.domain === 'dynamics' || String(c.name).startsWith('dynamics.'))
    && c.unit === 'm' && /deflection|displacement/.test(c.name));
  const rows = [];
  for (const channel of channels) {
    const doc = await api.series(data.id, { tier: channel.tier === 'fast' ? 'tick' : channel.tier, channels: [channel.name], maxPoints: 20000 });
    const peaks = new Map();
    let width = 0;
    for (const s of doc.series || []) {
      const m = /^(.*?)(?:\.(min|mean|max))?$/.exec(s.column);
      if (m[2] === 'mean') continue;
      const index = /\[(\d+)\]$/.exec(m[1]);
      const component = index ? Number(index[1]) : 0;
      width = Math.max(width, component + 1);
      const t = s.t_s || [];
      const v = s.v || [];
      for (let i = 0; i < t.length; i++) {
        if (v[i] === null || !Number.isFinite(v[i])) continue;
        const block = data.blockAt(t[i], data.pathList[0] || 0);
        if (!block || block.tool === null || block.tool === undefined) continue;
        if (!peaks.has(block.tool)) peaks.set(block.tool, []);
        const list = peaks.get(block.tool);
        list[component] = Math.max(list[component] || 0, Math.abs(v[i]));
      }
    }
    const labels = componentLabels(channel, width) || Array.from({ length: width }, (_, i) => `[${i}]`);
    for (const [tool, list] of [...peaks].sort((a, b) => a[0] - b[0])) {
      const values = Array.from({ length: width }, (_, i) => list[i] || 0);
      rows.push({ tool, channel: channel.name, labels, values, bound: Math.sqrt(values.reduce((a, x) => a + x * x, 0)) });
    }
  }
  return { channels, rows };
}

export async function renderTools(root, { api, data, page }) {
  const [tools, production] = await Promise.all([
    load(() => api.tools(data.id), data, 'tools.json'),
    load(() => api.production(data.id), data, 'production.json'),
  ]);
  if (!tools.doc) { root.append(section('Tools', absent('tools.json', tools))); }
  const list = tools.doc && Array.isArray(tools.doc.tools) ? tools.doc.tools : [];
  const costs = new Map();
  const part = production.doc && Array.isArray(production.doc.per_part) ? production.doc.per_part[0] : null;
  for (const t of (part && part.tools) || []) costs.set(t.number, t);
  for (const t of list) {
    const vbFrac = asNumber(t.VB_max_m) !== null && asNumber(t.VB_limit_m) ? t.VB_max_m / t.VB_limit_m : null;
    const life = asNumber(t.life_used_fraction);
    const cost = costs.get(t.number);
    const card = section(`T${t.number} ${t.tool_id || ''}`);
    card.append(h('p', { class: 'muted' }, `${t.holder_id ? `in ${t.holder_id}` : ''}${t.stickout_m ? `, stick-out ${fmt.mm(t.stickout_m, 1)}` : ''}${t.broken ? ' · BROKEN' : ''}`));
    card.append(h('div', { class: 'meters' },
      h('div', null, h('span', { class: 'label' }, 'Flank wear VB max'), meter(vbFrac, { text: `${fmt.um(t.VB_max_m, 1)} of ${fmt.um(t.VB_limit_m, 0)}`, tone: vbFrac > 1 ? 'bad' : vbFrac > 0.8 ? 'warn' : 'ok', label: 'VB max of the limit' })),
      h('div', null, h('span', { class: 'label' }, 'Life used'), meter(life, { text: life === null ? DASH : pct(life, 2), tone: life > 1 ? 'bad' : life > 0.8 ? 'warn' : 'ok', label: 'tool life used' }))));
    card.append(tiles([
      { label: 'Cutting time', value: duration(t.cut_time_s) },
      { label: 'Life left', value: asNumber(t.remaining_life_s) !== null ? duration(t.remaining_life_s) : DASH, sub: t.life_limit_s ? `of ${duration(t.life_limit_s)}` : null },
      { label: 'Hottest edge', value: fmt.degC(t.max_edge_temp_K) },
      { label: 'Broken', value: t.broken ? 'yes' : 'no', tone: t.broken ? 'error' : 'ok' },
      cost && cost.cost ? { label: 'Cost per part', value: fmt.eur(cost.cost.cost_per_part_eur), sub: `${num(cost.cost.parts_per_edge, 4)} parts per edge` } : null,
    ].filter(Boolean)));
    const vb = Array.isArray(t.VB_m) ? t.VB_m : [];
    if (vb.length) {
      card.append(figure({
        title: 'Flank wear per tooth', subtitle: 'VB (µm)',
        chart: (box) => box.append(barList(vb.map((v, i) => ({ label: `tooth ${i + 1}`, value: v * 1e6, text: fmt.um(v, 1), color: categorical(0) })),
          { max: asNumber(t.VB_limit_m) ? Math.max(t.VB_limit_m, ...vb) * 1e6 : null, label: 'VB per tooth' })),
        tableView: () => table({ columns: [{ key: 'tooth', label: 'tooth', num: true }, { key: 'vb', label: 'VB (µm)', num: true }],
          rows: vb.map((v, i) => ({ tooth: i + 1, vb: fixed(v * 1e6, 2) })) }),
      }));
    }
    if (cost && cost.cost) {
      card.append(kv([['price', fmt.eur(cost.cost.price_eur)], ['regrinds', num(cost.cost.regrinds)], ['edge lives', num(cost.cost.edge_lives)],
        ['cost per edge', fmt.eur(cost.cost.cost_per_edge_eur)], ['parts per edge', num(cost.cost.parts_per_edge, 4)], ['life used per part', pct(cost.life_used_part_frac, 4)],
        ['tool changes', num(cost.changes)], ['missing for the cost', (cost.cost.missing || []).join(', ') || 'nothing']]));
    }
    root.append(card);
  }
  const wearChannels = data.channels.filter((c) => c.domain === 'wear' || String(c.name).startsWith('wear.'));
  if (wearChannels.length) {
    const box = section('Wear over time');
    root.append(box);
    const group = new PlotGroup({ duration: Math.max(data.duration(), 1e-3), onSeek: (t) => page.seekTo(t) });
    for (const channel of wearChannels) {
      const plot = new TimePlot(box, { group, height: 140, title: channel.name });
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
  const deflection = section('Tool deflection and its share of the errors');
  root.append(deflection);
  try {
    const { channels, rows } = await deflectionByTool(api, data);
    if (!channels.length) deflection.append(notice('No dynamics channel in metres names a deflection or displacement.'));
    else {
      deflection.append(table({
        columns: [{ key: 'tool', label: 'tool', render: (r) => `T${r.tool}` }, { key: 'channel', label: 'channel', render: (r) => h('code', null, r.channel) },
          { key: 'peaks', label: 'peak per component', sortable: false, render: (r) => r.values.map((v, i) => `${r.labels[i]} ${fmt.um(v, 2)}`).join(' · ') },
          { key: 'bound', label: 'combined bound', num: true, render: (r) => fmt.um(r.bound, 2) }],
        rows, empty: 'The deflection channels have no samples while a tool is in the spindle.',
        caption: 'Peaks over the run; the bound combines the component peaks (an upper bound of the magnitude)',
      }));
    }
  } catch (error) { deflection.append(errorBox(error, 'Deflection: ')); }
  if (data.has('inspection.json')) {
    try {
      const inspection = await api.inspection(data.id);
      const shares = [];
      for (const t of inspection.tolerances || []) {
        const r = judge(t);
        for (const c of t.contributions_m || []) {
          if (!/deflection/.test(String(c.source))) continue;
          const reference = r.kind === 'zone' ? r.measured : r.deviation;
          shares.push({ id: t.id, tool: c.tool, value: c.value_m, share: reference ? c.value_m / reference : null });
        }
      }
      deflection.append(table({
        columns: [{ key: 'id', label: 'characteristic' }, { key: 'tool', label: 'tool', render: (x) => (x.tool ? `T${x.tool}` : DASH) },
          { key: 'value', label: 'deflection contribution', num: true, render: (x) => fmt.signedUm(x.value, 2) },
          { key: 'share', label: 'share of the deviation', num: true, render: (x) => (x.share === null ? DASH : pct(x.share, 0)) }],
        rows: shares, empty: 'The error budgets of inspection.json name no deflection contribution.', caption: 'In the error budget of each characteristic',
      }));
    } catch (error) { deflection.append(errorBox(error, 'Inspection: ')); }
  }
  const flagged = data.findings.filter((f) => TOOL_CATEGORIES.has(f.category));
  root.append(section('Tool findings', flagged.length ? h('div', null, flagged.map((f) => h('button', { type: 'button', class: 'finding-mini', on: { click: () => page.focusFinding(f) } },
    badge(f.severity), h('span', null, `${humanize(f.category)} at ${num(f.time_s)} s: ${f.message}`)))) : notice('No tool findings.')));
  if (!list.length && tools.doc) root.append(notice('tools.json lists no tools.'));
}
