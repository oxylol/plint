/* Process panels: coolant (reach and its share of the run, nozzles and jets, the wetted map,
 * shadows), chips (mass per part, chips in flight and piled, chip forms, the accumulation map) and
 * thermal state (temperature map, maximum per keyframe, part temperature at unload, hottest edges).
 * Maps come from keyframe fields at a chosen keyframe (default: the one at the playback time). */

import { h, clear, section, table, notice, errorBox, figure, kv, tiles, badge, humanize, lazyDetails, loading } from './dom.js';
import { Heatmap, XYPlot } from '../plot/charts.js';
import { barList } from '../plot/bars.js';
import { PlotGroup, TimePlot } from '../plot/timeplot.js';
import { buildSeries } from './plots.js';
import { gridFromArrays, sampleEvenly } from './fields.js';
import { categorical, chrome } from '../plot/colormap.js';
import { decodeArrayDoc } from '../api.js';
import { DASH, num, fmt, fixed, pct, asNumber, quantity } from '../plot/units.js';

/** ISO 3685:1993 Annex G chip forms (cnctwin.core.bus.CHIP_FORMS). */
export const CHIP_FORMS = {
  11: 'ribbon, long', 12: 'ribbon, short', 13: 'ribbon, snarled', 21: 'tubular, long', 22: 'tubular, short', 23: 'tubular, snarled',
  31: 'spiral, flat', 32: 'spiral, conical', 41: 'washer-type helical, long', 42: 'washer-type helical, short', 43: 'washer-type helical, snarled',
  51: 'conical helical, long', 52: 'conical helical, short', 61: 'arc, connected', 62: 'arc, loose', 70: 'elemental', 80: 'needle', 90: 'dust',
  95: 'molten / smeared',
};

function stockSize(data) {
  const s = data.scene && data.scene.stock && data.scene.stock.size_m;
  return Array.isArray(s) ? s.map(Number) : null;
}

/** A keyframe picker (range slider + "playback time" button): onPick(keyframe) after a change; `box`
 * shows a spinner from the change until onPick has drawn into it. */
function keyframePicker(data, time, box, onPick) {
  const frames = data.keyframes;
  const label = h('span', { class: 'muted' });
  const start = (time.t > 0 && data.keyframeAt(time.t)) || frames[frames.length - 1];
  const slider = h('input', { type: 'range', class: 'range', min: '0', max: String(Math.max(0, frames.length - 1)), value: String(Math.max(0, frames.indexOf(start))), 'aria-label': 'Keyframe' });
  let timer = null;
  const update = () => {
    const kf = frames[Number(slider.value)];
    label.textContent = kf ? `keyframe ${kf.k} at ${num(kf.time_s)} s` : 'no keyframes';
    clearTimeout(timer);
    if (!kf) return;
    clear(box);
    box.append(loading());
    timer = setTimeout(() => onPick(kf), 150);
  };
  slider.addEventListener('input', update);
  const now = h('button', { type: 'button', class: 'btn small', on: { click: () => { const kf = data.keyframeAt(time.t); if (kf) { slider.value = String(frames.indexOf(kf)); update(); } } } }, 'At playback time');
  const last = h('button', { type: 'button', class: 'btn small', on: { click: () => { slider.value = String(frames.length - 1); update(); } } }, 'Last');
  queueMicrotask(update);
  return h('div', { class: 'kf-picker' }, slider, h('div', { class: 'kf-picker-row' }, label, now, last));
}

async function channelPlots(box, api, data, page, channels) {
  if (!channels.length) { box.append(notice('No channels of this domain are recorded.')); return; }
  const group = new PlotGroup({ duration: Math.max(data.duration(), 1e-3), onSeek: (t) => page.seekTo(t) });
  for (const channel of channels) {
    const plot = new TimePlot(box, { group, height: 130, title: channel.name });
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

function findingsOf(data, page, categories) {
  const flagged = data.findings.filter((f) => categories.includes(f.category));
  return flagged.length ? h('div', null, flagged.map((f) => h('button', { type: 'button', class: 'finding-mini', on: { click: () => page.focusFinding(f) } },
    badge(f.severity), h('span', null, `${humanize(f.category)} at ${num(f.time_s)} s${f.line ? `, line ${f.line}` : ''}: ${f.message}`)))) : notice('None.');
}

function domainChannels(data, domain) {
  return data.channels.filter((c) => (c.domain || String(c.name).split('.')[0]) === domain);
}

// ============================================================================ coolant

async function macroSeries(api, data, name) {
  if (!data.channelsByName.has(name)) return null;
  const channel = data.channelsByName.get(name);
  try {
    const doc = await api.series(data.id, { tier: channel.tier === 'fast' ? 'tick' : channel.tier, channels: [name], maxPoints: 200000 });
    const s = (doc.series || [])[0];
    return s ? { t: s.t_s || [], v: (s.v || []).map((x) => (x === null ? NaN : Number(x))) } : null;
  } catch { return null; }
}

export async function renderCoolant(root, { api, data, time, page }) {
  const reach = await macroSeries(api, data, 'coolant.reach_fraction');
  if (reach && reach.t.length > 1) {
    let on = 0; let total = 0; let sum = 0; let lowest = Infinity;
    for (let i = 1; i < reach.t.length; i++) {
      const dt = reach.t[i] - reach.t[i - 1];
      total += dt;
      const v = reach.v[i];
      if (Number.isFinite(v) && v > 0) { on += dt; sum += v * dt; lowest = Math.min(lowest, v); }
    }
    root.append(section(null, tiles([
      { label: 'Coolant at the cut', value: total ? pct(on / total, 1) : DASH, sub: 'share of the run with a reach above zero' },
      { label: 'Reach while on', value: on ? pct(sum / on, 1) : DASH, sub: 'flow-weighted, time mean' },
      { label: 'Lowest reach while on', value: Number.isFinite(lowest) ? pct(lowest, 1) : DASH },
    ])));
  } else {
    root.append(section(null, notice('The run records no coolant.reach_fraction channel: the share of the run with coolant at the cut is not known.')));
  }
  const nozzles = (data.scene && data.scene.nozzles) || [];
  const perNozzle = new Map();
  const frames = sampleEvenly(data.keyframes, 48);
  const shadows = [];
  for (const kf of frames) {
    try {
      const doc = await api.keyframeFields(data.id, kf.k);
      const arrays = doc.arrays || {};
      const ids = arrays.jet_nozzle_id ? decodeArrayDoc(arrays.jet_nozzle_id).data : [];
      const get = (name, i) => (arrays[name] ? decodeArrayDoc(arrays[name]).data[i] : null);
      ids.forEach((id, i) => {
        if (!perNozzle.has(id)) perNozzle.set(id, { id, kind: get('jet_kind', i), on: 0, flow: 0, speed: 0, reach: 0 });
        const n = perNozzle.get(id);
        n.on++;
        n.flow += Number(get('jet_flow_m3_s', i)) || 0;
        n.speed += Number(get('jet_speed_m_s', i)) || 0;
        n.reach += Number(get('jet_reach_fraction', i)) || 0;
      });
      if (arrays.shadow_id) {
        const sid = decodeArrayDoc(arrays.shadow_id).data;
        const frac = arrays.shadow_fraction ? decodeArrayDoc(arrays.shadow_fraction).data : [];
        sid.forEach((s, i) => shadows.push({ k: kf.k, time_s: kf.time_s, occluder: s, fraction: frac[i] }));
      }
    } catch { /* keyframe without fields */ }
  }
  root.append(section('Nozzles', table({
    columns: [{ key: 'id', label: 'nozzle' }, { key: 'kind', label: 'kind' },
      { key: 'on', label: 'on at keyframes', num: true, render: (n) => `${n.on} of ${frames.length}` },
      { key: 'flow', label: 'mean flow', num: true, render: (n) => (n.on ? `${num(n.flow / n.on * 60000, 4)} l/min` : DASH) },
      { key: 'speed', label: 'mean jet speed', num: true, render: (n) => (n.on ? `${num(n.speed / n.on, 4)} m/s` : DASH) },
      { key: 'reach', label: 'mean reach', num: true, render: (n) => (n.on ? pct(n.reach / n.on, 1) : DASH) },
      { key: 'where', label: 'origin (mm, machine)', sortable: false, render: (n) => { const z = nozzles.find((x) => x.id === n.id); return z && z.origin_M_m ? z.origin_M_m.map((v) => fixed(v * 1e3, 1)).join(', ') : DASH; } }],
    rows: [...perNozzle.values(), ...nozzles.filter((z) => !perNozzle.has(z.id)).map((z) => ({ id: z.id, kind: z.kind, on: 0, flow: 0, speed: 0, reach: 0 }))],
    empty: 'The scene lists no nozzles and no keyframe records a jet.',
    caption: `${frames.length} of ${data.keyframes.length} keyframes read`,
  })));
  const jetsBox = section('Jets at the playback time');
  root.append(jetsBox);
  try {
    const jets = await api.jets(data.id, time.t);
    jetsBox.append(table({
      columns: [{ key: 'nozzle_id', label: 'nozzle' }, { key: 'kind', label: 'kind' }, { key: 'flow_m3_s', label: 'flow', num: true, render: (j) => `${num(j.flow_m3_s * 60000, 4)} l/min` },
        { key: 'speed_m_s', label: 'speed', num: true, render: (j) => `${num(j.speed_m_s, 4)} m/s` }, { key: 'reach_fraction', label: 'reach', num: true, render: (j) => pct(j.reach_fraction, 1) },
        { key: 'half_angle_rad', label: 'half angle', num: true, render: (j) => `${num(j.half_angle_rad * 180 / Math.PI, 3)}°` }],
      rows: jets.jets || [], empty: 'No jet is on at this keyframe.', caption: jets.k !== null ? `keyframe ${jets.k} at ${num(jets.time_s)} s` : null,
    }));
  } catch (error) { jetsBox.append(errorBox(error)); }
  const wettedBox = section('Wetted surface');
  root.append(wettedBox);
  if (data.keyframes.length) {
    const mapBox = h('div');
    wettedBox.append(keyframePicker(data, time, mapBox, async (kf) => {
      clear(mapBox);
      try {
        const doc = await api.keyframeFields(data.id, kf.k);
        const grid = gridFromArrays(doc.arrays || {}, 'wetted', ['wetted', 'surface'], stockSize(data));
        if (!grid) { mapBox.append(notice(`Keyframe ${kf.k} records no wetted field.`)); return; }
        const c = chrome();
        mapBox.append(figure({
          title: `Wetted cells at ${num(kf.time_s)} s`, subtitle: grid.cellsOnly ? 'grid origin not recorded: axes in cells' : 'stock frame, mm',
          chart: (b) => new Heatmap(b, { label: 'wetted map' }).setData({ ...grid, map: 'binary', categories: [[c.series[0], 'wetted'], [c.ink2, 'dry']] }),
          tableView: () => { let wet = 0; let n = 0; for (const v of grid.values) if (Number.isFinite(v)) { n++; if (v) wet++; } return kv([['cells', num(n)], ['wetted', `${num(wet)} (${pct(wet / (n || 1), 1)})`]]); },
          note: 'Dry cells near the cut are coolant shadows.',
        }));
      } catch (error) { mapBox.append(errorBox(error)); }
    }), mapBox);
  } else {
    wettedBox.append(notice('The run has no keyframes.'));
  }
  root.append(section('Shadows', shadows.length ? table({
    columns: [{ key: 'time_s', label: 'time', num: true, render: (s) => `${num(s.time_s)} s` }, { key: 'occluder', label: 'occluder' },
      { key: 'fraction', label: 'flow blocked', num: true, render: (s) => pct(s.fraction, 1) }],
    rows: shadows,
  }) : notice('No keyframe records a shadow (shadow_id, shadow_fraction).'), h('h3', null, 'Coolant findings'), findingsOf(data, page, ['coolant_shadowing'])));
  const plots = section('Coolant channels');
  root.append(plots);
  await channelPlots(plots, api, data, page, domainChannels(data, 'coolant'));
}

// ============================================================================ chips

export async function renderChips(root, { api, data, time, page }) {
  const summary = (data.run && data.run.summary) || {};
  let production = null;
  if (data.has('production.json')) { try { production = await api.production(data.id); } catch { production = null; } }
  const parts = production && Array.isArray(production.per_part) ? production.per_part : [];
  root.append(section(null, tiles([
    { label: 'Chip mass per part', value: parts.length ? fmt.g(parts[0].removed_mass_kg) : DASH, sub: 'removed material (mass balance)' },
    { label: 'Removed in this run', value: summary.removed_mass_kg !== undefined ? fmt.g(summary.removed_mass_kg) : DASH,
      sub: summary.removed_volume_m3 !== undefined ? `${num(summary.removed_volume_m3 * 1e9, 5)} mm³` : null },
  ])));
  if (parts.length > 1) {
    root.append(section('Chip mass per part', table({ columns: [{ key: 'part', label: 'part', num: true, render: (p) => String(p.part + 1) }, { key: 'm', label: 'chip mass', num: true, render: (p) => fmt.g(p.removed_mass_kg) }], rows: parts })));
  }
  const kfBox = section('Chips at a keyframe');
  root.append(kfBox);
  if (!data.keyframes.length) { kfBox.append(notice('The run has no keyframes.')); } else {
    const body = h('div');
    kfBox.append(keyframePicker(data, time, body, async (kf) => {
      clear(body);
      try {
        const doc = await api.chips(data.id, { t: kf.time_s });
        const arrays = Object.fromEntries(Object.entries(doc.arrays || {}).map(([k, v]) => [k, decodeArrayDoc(v)]));
        const mass = arrays.mass_kg ? Array.from(arrays.mass_kg.data) : [];
        const total = mass.reduce((a, b) => a + (Number.isFinite(b) ? b : 0), 0);
        const scalars = Object.entries(arrays).filter(([k, v]) => v && v.shape.length === 0 && k !== 'tick' && k !== 'time_s');
        body.append(kv([['keyframe', `${doc.k} at ${num(doc.time_s)} s`], ['chips in flight', `${num(mass.length)} particles, ${fmt.g(total)}`],
          ...scalars.map(([k, v]) => { const q = quantity(k, v.data[0]); return [q.label, q.text]; })]));
        const codes = arrays.form_code ? Array.from(arrays.form_code.data) : [];
        if (codes.length && codes.length === mass.length) {
          const byForm = new Map();
          codes.forEach((code, i) => byForm.set(code, (byForm.get(code) || 0) + (mass[i] || 0)));
          const rows = [...byForm].sort((a, b) => b[1] - a[1]).map(([code, m]) => ({ code, name: CHIP_FORMS[code] || 'unknown code', m }));
          body.append(figure({
            title: 'Chip forms by mass in flight', subtitle: 'ISO 3685 codes',
            chart: (b) => b.append(barList(rows.map((r) => ({ label: `${r.code} ${r.name}`, value: r.m, text: `${fmt.g(r.m)} · ${pct(r.m / (total || 1), 1)}`, color: categorical(0) })), { label: 'chip forms' })),
            tableView: () => table({ columns: [{ key: 'code', label: 'code', num: true }, { key: 'name', label: 'chip form' }, { key: 'm', label: 'mass', num: true, render: (r) => fmt.g(r.m) }], rows }),
          }));
        }
        const fields = await api.keyframeFields(data.id, kf.k);
        const grid = gridFromArrays(fields.arrays || {}, 'pile_height_m', ['pile', 'surface'], stockSize(data));
        if (grid) {
          for (let i = 0; i < grid.values.length; i++) grid.values[i] *= 1e3;
          body.append(figure({
            title: `Chip accumulation at ${num(kf.time_s)} s`, subtitle: grid.cellsOnly ? 'grid origin not recorded: axes in cells' : 'settled chip height, stock frame',
            chart: (b) => new Heatmap(b, { label: 'chip accumulation map' }).setData({ ...grid, map: 'sequential', unit: 'mm' }),
            tableView: () => { let max = -Infinity; let sum = 0; let n = 0; for (const v of grid.values) if (Number.isFinite(v)) { n++; sum += v; max = Math.max(max, v); } return kv([['cells', num(n)], ['highest pile', `${num(max)} mm`], ['mean height', `${num(sum / (n || 1))} mm`]]); },
          }));
        } else {
          body.append(notice(`Keyframe ${kf.k} records no chip pile height field.`));
        }
      } catch (error) { body.append(errorBox(error)); }
    }), body);
  }
  root.append(section('Chip findings', findingsOf(data, page, ['chip_packing', 'chip_recutting', 'chip_hazard'])));
  const plots = section('Chip channels');
  root.append(plots);
  await channelPlots(plots, api, data, page, domainChannels(data, 'chips'));
}

// ============================================================================ thermal

function temperatureField(arrays) {
  const names = Object.keys(arrays).filter((n) => /temp_K$/.test(n) && !/^jet_/.test(n) && arrays[n] && (arrays[n].shape || []).length >= 2).sort();
  return names[0] || null;
}

export async function renderThermal(root, { api, data, time, page }) {
  const mapBox = section('Temperature of the part');
  root.append(mapBox);
  if (data.keyframes.length) {
    const body = h('div');
    mapBox.append(keyframePicker(data, time, body, async (kf) => {
      clear(body);
      try {
        const doc = await api.keyframeFields(data.id, kf.k);
        const arrays = doc.arrays || {};
        const name = temperatureField(arrays);
        if (!name) { body.append(notice(`Keyframe ${kf.k} records no temperature field.`)); return; }
        const prefix = name.replace(/temp_K$/, '').replace(/_$/, '');
        const grid = gridFromArrays(arrays, name, [prefix, 'surface', 'thermal', 'work_temp'], stockSize(data));
        for (let i = 0; i < grid.values.length; i++) grid.values[i] -= 273.15;
        body.append(figure({
          title: `${name} at ${num(kf.time_s)} s`, subtitle: (arrays[name].shape || []).length === 3 ? 'maximum through z' : 'stock frame',
          chart: (b) => new Heatmap(b, { label: 'temperature map' }).setData({ ...grid, map: 'sequential-orange', unit: '°C' }),
          tableView: () => { let max = -Infinity; let min = Infinity; for (const v of grid.values) if (Number.isFinite(v)) { max = Math.max(max, v); min = Math.min(min, v); } return kv([['lowest', `${fixed(min, 1)} °C`], ['highest', `${fixed(max, 1)} °C`], ['cells', `${grid.nx} × ${grid.ny}`]]); },
        }));
      } catch (error) { body.append(errorBox(error)); }
    }), body);
    mapBox.append(lazyDetails('Highest temperature per keyframe', async (b) => {
      const frames = sampleEvenly(data.keyframes, 40);
      const rows = [];
      for (const kf of frames) {
        try {
          const doc = await api.keyframeFields(data.id, kf.k);
          const name = temperatureField(doc.arrays || {});
          if (!name) continue;
          const arr = decodeArrayDoc(doc.arrays[name]);
          let max = -Infinity;
          for (const v of arr.data) if (Number.isFinite(v) && v > max) max = v;
          if (Number.isFinite(max)) rows.push({ k: kf.k, t: Number(kf.time_s), max: max - 273.15 });
        } catch { /* skip */ }
      }
      if (!rows.length) { b.append(notice('No keyframe records a temperature field.')); return; }
      const c = chrome();
      b.append(figure({
        title: 'Highest part temperature', subtitle: `${rows.length} of ${data.keyframes.length} keyframes`,
        chart: (x) => new XYPlot(x, { xLabel: 'time (s)', yLabel: 'temperature (°C)', height: 220 }).setData({
          lines: [{ x: rows.map((r) => r.t), y: rows.map((r) => r.max), color: c.series[1], label: 'highest' }],
          points: rows.map((r) => ({ x: r.t, y: r.max, color: c.series[1], r: 3, tip: [[c.series[1], `${fixed(r.max, 1)} °C`, `keyframe ${r.k} at ${num(r.t)} s`]] })),
        }),
        tableView: () => table({ columns: [{ key: 'k', label: 'keyframe', num: true }, { key: 't', label: 'time (s)', num: true, render: (r) => num(r.t) }, { key: 'max', label: 'highest (°C)', num: true, render: (r) => fixed(r.max, 1) }], rows }),
      }));
    }));
  } else {
    mapBox.append(notice('The run has no keyframes.'));
  }
  const unloadBox = section('Part temperature at unload');
  root.append(unloadBox);
  let unload = null;
  if (data.has('inspection.json')) {
    try { const doc = await api.inspection(data.id); unload = Array.isArray(doc.unload) && doc.unload.length ? doc.unload : null; } catch { unload = null; }
  }
  if (unload) {
    unloadBox.append(table({ columns: [{ key: 'op', label: 'operation' }, { key: 'time_s', label: 'time', num: true, render: (u) => `${num(u.time_s)} s` },
      { key: 'mean_temp_K', label: 'mean', num: true, render: (u) => fmt.degC(u.mean_temp_K) }, { key: 'max_temp_K', label: 'highest', num: true, render: (u) => fmt.degC(u.max_temp_K) }], rows: unload, caption: 'inspection.json' }));
  } else {
    const name = ['thermal.work_unload_mean_K', 'thermal.work_bulk_mean_K'].find((n) => data.channelsByName.has(n));
    const series = name ? await macroSeries(api, data, name) : null;
    const last = series ? [...series.v].reverse().find(Number.isFinite) : null;
    unloadBox.append(last !== undefined && last !== null ? kv([[name, `${fmt.degC(last)} at the end of the run`]]) : notice('Not recorded (no inspection.json unload and no work temperature channel).'));
  }
  if (data.has('tools.json')) {
    try {
      const tools = await api.tools(data.id);
      root.append(section('Hottest cutting edges', table({ columns: [{ key: 'number', label: 'tool', render: (t) => `T${t.number} ${t.tool_id || ''}` },
        { key: 'max_edge_temp_K', label: 'hottest edge', num: true, render: (t) => fmt.degC(t.max_edge_temp_K) }], rows: (tools.tools || []).filter((t) => asNumber(t.max_edge_temp_K) !== null), empty: 'tools.json gives no edge temperatures.' })));
    } catch { /* tools unreadable: the Tools tab says why */ }
  }
  root.append(section('Thermal findings', findingsOf(data, page, ['thermal_damage', 'thermal_error'])));
  const plots = section('Thermal channels');
  root.append(plots);
  await channelPlots(plots, api, data, page, domainChannels(data, 'thermal'));
}
