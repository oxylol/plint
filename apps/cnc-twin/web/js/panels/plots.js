/* Plots: every channel of channels.json, picked by domain, drawn as stacked plots that share one
 * time range and the playback cursor (and the 3D view's time). Data come from the series API with
 * M4 downsampling at four points per pixel column, re-fetched for the visible range after a zoom or
 * pan, so peaks are never smoothed away; fast (L0) channels are drawn from their tick-tier min/max
 * envelopes, with the fast samples of the recording windows on top when the view spans 1 s or
 * less. More than eight columns fold into one band (lowest minimum to highest maximum, with the
 * mean), as in the report. Each plot has a table view (minimum, mean, maximum, value at the cursor). */

import { h, clear, button, notice, errorBox, humanize, table } from './dom.js';
import { PlotGroup, TimePlot, bucketize, medianStep } from '../plot/timeplot.js';
import { displayUnit, magnitudeOf, num } from '../plot/units.js';
import { categorical } from '../plot/colormap.js';
import { toArrays } from './rundata.js';

export const DEFAULT_CHANNELS = ['cutting.spindle_power_W', 'cutting.force_N', 'thermal.edge_temp_K', 'wear.VB_m'];
const STORAGE = 'cnctwin.plots.v1';
const MAX_COLUMNS = 8;
const FAST_SPAN_S = 1.0;
const REFRESH_S = 3;

function columnIndex(column) {
  const m = /\[(\d+)\]$/.exec(column);
  return m ? Number(m[1]) : -1;
}

export function componentLabels(channel, count) {
  const doc = String(channel.doc || '');
  const m = /\(([^()]+)\)\s*$/.exec(doc);
  if (m) {
    const parts = m[1].split(',').map((s) => s.trim()).filter(Boolean);
    if (parts.length >= count && parts.every((p) => p.length <= 12)) return parts;
  }
  if (/tooth/i.test(doc) || /tooth|teeth/i.test(channel.name)) return Array.from({ length: count }, (_, i) => `tooth ${i + 1}`);
  return null;
}

function unitFor(channel, magnitude) {
  const name = String(channel.name);
  if ((channel.unit === '-' || channel.unit === '' || channel.unit === '1') && /(_fraction|_frac)$/.test(name)) {
    return { label: '%', factor: 100, offset: 0 };
  }
  return displayUnit(channel.unit, magnitude);
}

function convert(array, du) {
  if (!array) return null;
  const out = new Float64Array(array.length);
  for (let i = 0; i < array.length; i++) out[i] = array[i] * du.factor + du.offset;
  return out;
}

function sameTimes(arrays) {
  const list = arrays.filter(Boolean);
  if (list.length < 2) return true;
  const n = list[0].t.length;
  return list.every((a) => a.t.length === n && a.t[0] === list[0].t[0] && a.t[n - 1] === list[0].t[n - 1]);
}

/** Series of one channel from a series response: bands for envelopes, lines otherwise. */
export function buildSeries(doc, channel, { buckets = 600 } = {}) {
  const groups = new Map();
  for (const s of doc.series || []) {
    const m = /^(.*)\.(min|mean|max)$/.exec(s.column);
    const base = m ? m[1] : s.column;
    if (!groups.has(base)) groups.set(base, {});
    groups.get(base)[m ? m[2] : 'plain'] = toArrays(s);
  }
  let bases = [...groups.keys()].sort((a, b) => columnIndex(a) - columnIndex(b));
  const hasData = (g) => Object.values(g).some((a) => a.v.some(Number.isFinite));
  bases = bases.filter((b) => hasData(groups.get(b)));
  const magnitude = magnitudeOf(...bases.flatMap((b) => Object.values(groups.get(b)).map((a) => a.v)));
  const du = unitFor(channel, magnitude);
  const t0 = Number(doc.t0_s);
  const t1 = Number(doc.t1_s);
  const nb = Math.max(1, Math.round(Number(doc.buckets) || buckets));
  const downsampled = !!doc.downsampled;
  const labels = componentLabels(channel, bases.length && columnIndex(bases[bases.length - 1]) + 1);
  const gapOf = (t) => (downsampled ? 2.5 * ((t1 - t0) / nb) : 3 * (medianStep(t) || 0)) || Infinity;
  const items = bases.map((base) => {
    const g = groups.get(base);
    const index = columnIndex(base);
    const label = index < 0 ? channel.name.split('.').slice(1).join('.') || channel.name : (labels && labels[index]) || `[${index}]`;
    if (g.min || g.max) {
      const parts = [g.min, g.mean, g.max];
      if (!downsampled && sameTimes(parts)) {
        const t = (g.mean || g.min || g.max).t;
        return { index, label, t, v: g.mean ? g.mean.v : null, lo: g.min ? g.min.v : null, hi: g.max ? g.max.v : null, gap: gapOf(t) };
      }
      const lo = g.min ? bucketize(g.min.t, g.min.v, t0, t1, nb) : null;
      const hi = g.max ? bucketize(g.max.t, g.max.v, t0, t1, nb) : null;
      const mean = g.mean ? bucketize(g.mean.t, g.mean.v, t0, t1, nb) : null;
      const ref = mean || lo || hi;
      return { index, label, t: ref.t, v: mean ? mean.mean : null, lo: lo ? lo.lo : null, hi: hi ? hi.hi : null, gap: 2.5 * ref.width };
    }
    const plain = g.plain;
    return { index, label, t: plain.t, v: plain.v, gap: gapOf(plain.t), dense: downsampled };
  });
  let series;
  if (items.length > MAX_COLUMNS) {
    const lo = new Float64Array(nb).fill(NaN);
    const hi = new Float64Array(nb).fill(NaN);
    const sum = new Float64Array(nb);
    const count = new Uint32Array(nb);
    let grid = null;
    for (const item of items) {
      const lower = bucketize(item.t, item.lo || item.v, t0, t1, nb);
      const upper = bucketize(item.t, item.hi || item.v, t0, t1, nb);
      const mid = bucketize(item.t, item.v || item.lo, t0, t1, nb);
      grid = lower.t;
      for (let k = 0; k < nb; k++) {
        if (Number.isFinite(lower.lo[k]) && !(lo[k] <= lower.lo[k])) lo[k] = lower.lo[k];
        if (Number.isFinite(upper.hi[k]) && !(hi[k] >= upper.hi[k])) hi[k] = upper.hi[k];
        if (Number.isFinite(mid.mean[k])) { sum[k] += mid.mean[k]; count[k]++; }
      }
    }
    const mean = new Float64Array(nb);
    for (let k = 0; k < nb; k++) mean[k] = count[k] ? sum[k] / count[k] : NaN;
    series = [{ label: `${items.length} columns`, color: categorical(0), t: grid, v: convert(mean, du), lo: convert(lo, du), hi: convert(hi, du), gap: 2.5 * (t1 - t0) / nb }];
  } else {
    series = items.map((item) => ({
      label: item.label, color: categorical(Math.max(0, item.index)), t: item.t, v: convert(item.v, du),
      lo: convert(item.lo, du), hi: convert(item.hi, du), gap: item.gap, dense: item.dense,
    }));
  }
  return { series, unit: du, folded: items.length > MAX_COLUMNS, columns: items.length, downsampled, rows: doc.rows, buckets: doc.buckets };
}

export class PlotsPanel {
  constructor(parent, { api, data, time, height = 150 }) {
    this.api = api;
    this.data = data;
    this.time = time;
    this.height = height;
    this.entries = new Map();
    this.el = h('div', { class: 'plots' });
    parent.append(this.el);
    this.group = new PlotGroup({ duration: Math.max(time.duration, 1e-3), onSeek: (t) => time.seek(t, 'plot'), onRange: () => this.refetchAll() });
    this.group.setCursor(time.t);
    time.addEventListener('time', (e) => this.group.setCursor(e.detail.t));
    this._lastGrowth = 0;
    time.addEventListener('duration', (e) => {
      // a running run grows: re-fetch at most every REFRESH_S, and once more when it stops growing
      const now = performance.now();
      const due = now - this._lastGrowth > REFRESH_S * 1000;
      if (due) this._lastGrowth = now;
      this.group.setDuration(Math.max(e.detail, 1e-3), { emit: due });
      clearTimeout(this._growthTimer);
      if (!due) this._growthTimer = setTimeout(() => { this._lastGrowth = performance.now(); this.refetchAll(); }, REFRESH_S * 1000);
    });
    this.selected = this._initialSelection();
    this.pickerBody = h('div', { class: 'picker-body' });
    this.picker = h('details', { class: 'disclosure picker' }, h('summary', null, 'Channels'), this.pickerBody);
    this.follow = h('input', { type: 'checkbox', checked: true, on: { change: () => { this.group.follow = this.follow.checked; } } });
    const toolbar = h('div', { class: 'plots-toolbar' },
      button('Zoom in', () => this.group.zoom(2, this.time.t), { kind: 'small', ariaLabel: 'Zoom in' }),
      button('Zoom out', () => this.group.zoom(0.5, this.time.t), { kind: 'small', ariaLabel: 'Zoom out' }),
      button('Whole run', () => this.group.reset(), { kind: 'small' }),
      h('label', { class: 'check' }, this.follow, h('span', null, 'Keep the cursor in view')),
      h('span', { class: 'muted small' }, 'Tap or click to set the time · drag to pan · pinch or ctrl + wheel to zoom'));
    this.stack = h('div', { class: 'plot-stack' });
    this.el.append(this.picker, toolbar, this.stack);
    this.renderPicker();
    this.sync();
  }

  _initialSelection() {
    const names = new Set(this.data.channels.map((c) => c.name));
    try {
      const stored = JSON.parse(localStorage.getItem(STORAGE) || 'null');
      if (Array.isArray(stored)) {
        const kept = stored.filter((n) => names.has(n));
        if (kept.length) return kept;
      }
    } catch { /* no storage: defaults */ }
    const defaults = DEFAULT_CHANNELS.filter((n) => names.has(n));
    return defaults.length ? defaults : this.data.channels.slice(0, 4).map((c) => c.name);
  }

  _store() {
    try { localStorage.setItem(STORAGE, JSON.stringify(this.selected)); } catch { /* private mode */ }
  }

  renderPicker() {
    clear(this.pickerBody);
    if (!this.data.channels.length) {
      this.pickerBody.append(this.data.errors.channels ? errorBox(this.data.errors.channels) : notice('This run records no channels.'));
      return;
    }
    const domains = new Map();
    for (const c of this.data.channels) {
      const domain = c.domain || String(c.name).split('.')[0];
      if (!domains.has(domain)) domains.set(domain, []);
      domains.get(domain).push(c);
    }
    this.pickerBody.append(h('div', { class: 'picker-actions' },
      button('Defaults', () => { this.selected = this._initialDefaults(); this._store(); this.renderPicker(); this.sync(); }, { kind: 'small' }),
      button('All', () => { this.selected = this.data.channels.map((c) => c.name); this._store(); this.renderPicker(); this.sync(); }, { kind: 'small' }),
      button('None', () => { this.selected = []; this._store(); this.renderPicker(); this.sync(); }, { kind: 'small' })));
    for (const [domain, channels] of [...domains].sort((a, b) => a[0].localeCompare(b[0]))) {
      const group = h('fieldset', { class: 'picker-group' }, h('legend', null, humanize(domain)));
      for (const c of channels) {
        const box = h('input', {
          type: 'checkbox', checked: this.selected.includes(c.name),
          on: { change: () => {
            if (box.checked) { if (!this.selected.includes(c.name)) this.selected.push(c.name); } else { this.selected = this.selected.filter((n) => n !== c.name); }
            this._store();
            this.sync();
          } },
        });
        const shape = Array.isArray(c.max_shape) && c.max_shape.length ? ` [${c.max_shape.join('×')}]` : '';
        group.append(h('label', { class: 'pick', title: c.doc || '' }, box,
          h('span', null, h('b', null, String(c.name).split('.').slice(1).join('.') || c.name), ` ${c.unit && c.unit !== '-' ? c.unit : ''}${shape}`,
            h('span', { class: `tier tier-${c.tier}` }, c.tier), c.doc ? h('small', null, c.doc) : null)));
      }
      this.pickerBody.append(group);
    }
  }

  _initialDefaults() {
    const names = new Set(this.data.channels.map((c) => c.name));
    const defaults = DEFAULT_CHANNELS.filter((n) => names.has(n));
    return defaults.length ? defaults : this.data.channels.slice(0, 4).map((c) => c.name);
  }

  /** Make the stack match the selection (in channels.json order). */
  sync() {
    const wanted = this.data.channels.filter((c) => this.selected.includes(c.name));
    for (const [name, entry] of this.entries) {
      if (!wanted.some((c) => c.name === name)) { entry.plot.destroy(); entry.box.remove(); this.entries.delete(name); }
    }
    if (!wanted.length) {
      if (!this.emptyNote) { this.emptyNote = notice('Pick channels above to plot them.'); this.stack.append(this.emptyNote); }
      return;
    }
    if (this.emptyNote) { this.emptyNote.remove(); this.emptyNote = null; }
    for (const channel of wanted) {
      if (this.entries.has(channel.name)) { this.stack.append(this.entries.get(channel.name).box); continue; }
      const box = h('div', { class: 'plot-box' });
      const plot = new TimePlot(box, { group: this.group, height: this.height, title: channel.name, unit: '' });
      const tableBox = h('div', { class: 'figure-table', hidden: true });
      const toggle = button('Table', () => {
        tableBox.hidden = !tableBox.hidden;
        toggle.setAttribute('aria-pressed', String(!tableBox.hidden));
        if (!tableBox.hidden) this._fillTable(entry);
      }, { kind: 'small ghost', title: 'Minimum, mean, maximum and the value at the cursor' });
      plot.toolsEl.append(toggle);
      box.append(tableBox);
      const entry = { channel, plot, box, tableBox, request: 0 };
      this.entries.set(channel.name, entry);
      this.stack.append(box);
      this.fetch(entry);
    }
  }

  _fillTable(entry) {
    clear(entry.tableBox);
    const unit = entry.unitLabel || '';
    entry.tableBox.append(table({
      columns: [
        { key: 'label', label: 'column' },
        { key: 'min', label: `minimum${unit ? ` (${unit})` : ''}`, num: true, render: (r) => num(r.min) },
        { key: 'mean', label: `mean${unit ? ` (${unit})` : ''}`, num: true, render: (r) => num(r.mean) },
        { key: 'max', label: `maximum${unit ? ` (${unit})` : ''}`, num: true, render: (r) => num(r.max) },
        { key: 'atCursor', label: `at ${num(this.time.t)} s`, num: true, render: (r) => num(r.atCursor) },
      ],
      rows: entry.plot.stats(),
      caption: `${entry.channel.name}, ${num(this.group.range[0])} s to ${num(this.group.range[1])} s`,
    }));
  }

  async fetch(entry) {
    const request = ++entry.request;
    const channel = entry.channel;
    const [a, b] = this.group.range;
    const span = Math.max(b - a, 1e-6);
    const t0 = Math.max(0, a - span * 0.02);
    const t1 = b + span * 0.02;
    const width = entry.plot.width || this.stack.clientWidth || 600;
    const maxPoints = Math.min(20000, Math.max(400, Math.round(width * 4)));
    entry.plot.setLoading(true);
    try {
      const tier = channel.tier === 'fast' ? 'tick' : (channel.tier || 'tick');
      const doc = await this.api.series(this.data.id, { tier, channels: [channel.name], t0, t1, maxPoints });
      if (request !== entry.request) return;
      const built = buildSeries(doc, channel, { buckets: maxPoints / 4 });
      let series = built.series;
      const notes = [`${channel.doc || ''}`.trim(), `${channel.tier} tier`];
      if (channel.tier === 'fast') notes.push('band: min to max of the L0 samples in each L1 tick, line: their mean');
      if (built.downsampled) notes.push(`M4 downsampled from ${num(built.rows)} rows`);
      if (built.folded) notes.push(`${built.columns} columns folded into one band`);
      if (channel.tier === 'fast' && span <= FAST_SPAN_S && !built.folded) {
        try {
          const fast = await this.api.series(this.data.id, { tier: 'fast', channels: [channel.name], t0, t1, maxPoints });
          if (request !== entry.request) return;
          if ((fast.rows || 0) > 0) {
            const extra = buildSeries(fast, channel, { buckets: maxPoints / 4 });
            const factor = built.unit.factor / (extra.unit.factor || 1);
            for (const s of extra.series) {
              if (factor !== 1 && s.v) for (let i = 0; i < s.v.length; i++) s.v[i] = (s.v[i] - extra.unit.offset) * factor + built.unit.offset;
              series.push({ ...s, label: `${s.label} (L0)`, lo: null, hi: null, dense: true });
            }
            notes.push(`L0 samples of ${num(fast.rows)} rows in the recording windows`);
          } else {
            notes.push('no L0 samples recorded in this range (outside the recording windows)');
          }
        } catch (error) {
          notes.push(`fast samples unavailable (${error.detail || error.message})`);
        }
      }
      entry.unitLabel = built.unit.label;
      entry.plot.setTitle(channel.name, built.unit.label);
      entry.plot.setNote(notes.filter(Boolean).join(' · '));
      entry.plot.setLoading(false);
      entry.plot.setSeries(series, { emptyText: doc.rows ? 'no finite values in this range' : 'no samples in this range' });
      if (!entry.tableBox.hidden) this._fillTable(entry);
    } catch (error) {
      if (request !== entry.request) return;
      entry.plot.setLoading(false);
      entry.plot.setNote(`Could not load: ${error.detail || error.message}`);
      entry.plot.setSeries([], { emptyText: 'could not load this channel' });
    }
  }

  refetchAll() {
    for (const entry of this.entries.values()) this.fetch(entry);
  }

  destroy() {
    for (const entry of this.entries.values()) entry.plot.destroy();
    this.entries.clear();
  }
}
