/* Layer controls of the 3D view: which result is painted on the part (temperature, chips,
 * coolant, finish, deviation, deflection, wear), its colour bar and legend, the section plane,
 * what is shown (machine, fixture, chips, coolant, shadows, tool, deflected tool), measuring,
 * following the tool, the camera reset, and the picked point's value.
 *
 * Colour bar ranges come from the run's own data: keyframe fields (sampled at up to 12 keyframes
 * over the run) for temperature, chip piles and tool deflection; the final per-vertex attributes
 * (keyframes/attr/*.npz, up to 24 bricks) for finish, deviation and wear. When the 3D view
 * provides legend(layer) -> {min, max, unit, map | categories}, that is shown instead. */

import { h, clear, button, notice } from './dom.js';
import { ColorBar } from '../plot/charts.js';
import { LAYER_MAPS, chrome, withAlpha } from '../plot/colormap.js';
import { num, fixed } from '../plot/units.js';
import { decodeArrayDoc } from '../api.js';

export const LAYERS = [
  { name: 'temperature', label: 'Temperature', unit: '°C' },
  { name: 'chips', label: 'Chips', unit: 'mm' },
  { name: 'coolant', label: 'Coolant', unit: '' },
  { name: 'finish', label: 'Finish (Ra)', unit: 'µm' },
  { name: 'deviation', label: 'Deviation', unit: 'µm' },
  { name: 'deflection', label: 'Deflection', unit: 'µm' },
  { name: 'wear', label: 'Wear (VB)', unit: 'µm' },
];
const VISIBLE = [['machine', 'Machine', true], ['fixture', 'Fixture', true], ['tool', 'Tool', true], ['chips', 'Chips', true],
  ['coolant', 'Coolant', true], ['shadows', 'Coolant shadows', false], ['deflectedTool', 'Deflected tool', false]];

function sample(list, count) {
  if (list.length <= count) return list.slice();
  const out = [];
  for (let i = 0; i < count; i++) out.push(list[Math.round(i * (list.length - 1) / (count - 1))]);
  return [...new Set(out)];
}

function finiteRange(values, transform = (v) => v) {
  let lo = Infinity; let hi = -Infinity;
  for (const raw of values) {
    if (raw === null || !Number.isFinite(raw)) continue;
    const v = transform(raw);
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return lo <= hi ? [lo, hi] : null;
}

/** What the run records for each layer (fields of one keyframe, attribute files). */
export async function layerAvailability(api, data) {
  const out = {};
  const last = data.keyframes[data.keyframes.length - 1];
  let names = [];
  if (last) {
    try { names = Object.keys((await api.keyframeFields(data.id, last.k)).arrays || {}); } catch { names = []; }
  }
  const attrFiles = [...data.files.keys()].filter((p) => /^keyframes\/attr\/.+\.npz$/.test(p));
  let attrNames = [];
  if (attrFiles.length) {
    try { attrNames = Object.keys(await api.npz(data.id, attrFiles[0])); } catch { attrNames = []; }
  }
  const temp = names.filter((n) => /temp_K$/.test(n) && !/^jet_/.test(n));
  out.temperature = temp.length ? { fields: temp } : null;
  out.chips = names.includes('pile_height_m') || data.keyframes.some((k) => k.chips) ? { field: names.includes('pile_height_m') ? 'pile_height_m' : null } : null;
  out.coolant = names.includes('wetted') || names.some((n) => n.startsWith('jet_')) ? {} : null;
  const pick = (list) => list.find((n) => attrNames.includes(n)) || null;
  out.finish = pick(['Ra_m', 'finish_Ra_m']) ? { attr: pick(['Ra_m', 'finish_Ra_m']), files: attrFiles } : null;
  out.deviation = pick(['deviation_m']) ? { attr: 'deviation_m', files: attrFiles } : null;
  out.wear = pick(['generating_VB_m', 'VB_m', 'wear_m']) ? { attr: pick(['generating_VB_m', 'VB_m', 'wear_m']), files: attrFiles } : null;
  const deflectionChannels = data.channels.filter((c) => /deflection|displacement/.test(c.name) && c.unit === 'm');
  out.deflection = names.includes('tool_deflection_m') || deflectionChannels.length ? { field: names.includes('tool_deflection_m') ? 'tool_deflection_m' : null, channels: deflectionChannels } : null;
  return out;
}

/** Colour-bar data of a layer from the run's data: {map, domain, unit, source} or {categories, source}. */
export async function layerLegend(api, data, name, info) {
  const c = chrome();
  if (name === 'coolant') {
    return { categories: [[c.series[0], 'wetted'], [withAlpha(c.ink2, 0.35), 'dry'], [c.series[2], 'jet'], [c.series[1], 'shadowed flow']],
      source: 'keyframe fields: wetted cells, jets, shadows' };
  }
  if (['temperature', 'chips', 'deflection'].includes(name)) {
    const frames = sample(data.keyframes, 12);
    const fieldNames = name === 'temperature' ? info.fields : [info.field].filter(Boolean);
    const values = [];
    for (const kf of frames) {
      if (!fieldNames.length) break;
      try {
        const doc = await api.keyframeFields(data.id, kf.k, fieldNames);
        for (const field of fieldNames) {
          const arr = decodeArrayDoc((doc.arrays || {})[field]);
          if (!arr) continue;
          if (name === 'deflection') {
            const d = arr.data;
            if (d.length && d.length % 3 === 0) for (let i = 0; i < d.length; i += 3) values.push(Math.hypot(d[i], d[i + 1], d[i + 2]));
            else for (const v of d) values.push(Math.abs(v));
          } else {
            for (const v of arr.data) values.push(v);
          }
        }
      } catch { /* that keyframe has no such field */ }
    }
    if (name === 'deflection' && !values.length && info.channels && info.channels.length) {
      for (const channel of info.channels.slice(0, 2)) {
        try {
          const doc = await api.series(data.id, { tier: channel.tier === 'fast' ? 'tick' : channel.tier, channels: [channel.name], maxPoints: 2000 });
          for (const s of doc.series || []) for (const v of s.v || []) if (v !== null) values.push(Math.abs(v));
        } catch { /* skip */ }
      }
    }
    const transform = name === 'temperature' ? (v) => v - 273.15 : (v) => v * (name === 'chips' ? 1e3 : 1e6);
    const range = finiteRange(values, transform);
    if (!range) return null;
    const unit = name === 'temperature' ? '°C' : name === 'chips' ? 'mm' : 'µm';
    return { map: LAYER_MAPS[name], domain: range[1] > range[0] ? range : [range[0], range[0] + 1e-9], unit,
      source: `${fieldNames.join(', ') || 'deflection channels'} over ${frames.length} keyframe${frames.length === 1 ? '' : 's'}` };
  }
  const files = sample(info.files || [], 24);
  const values = [];
  for (const file of files) {
    try {
      const arrays = await api.npz(data.id, file);
      const arr = arrays[info.attr];
      if (arr) for (const v of arr.data) values.push(v);
    } catch { /* unreadable brick */ }
  }
  let range = finiteRange(values, (v) => v * 1e6);
  if (!range) return null;
  if (name === 'deviation') {
    const bound = Math.max(Math.abs(range[0]), Math.abs(range[1]));
    range = [-bound, bound];
  }
  return { map: LAYER_MAPS[name], domain: range[1] > range[0] ? range : [range[0], range[0] + 1e-9], unit: 'µm',
    source: `${info.attr} of ${files.length} of ${(info.files || []).length} bricks (final surface)` };
}

export class LayerControls {
  constructor(parent, { api, data, getView }) {
    this.api = api;
    this.data = data;
    this.getView = getView;
    this.layer = null;
    this.legends = new Map();
    this.visible = Object.fromEntries(VISIBLE.map(([key, , on]) => [key, on]));
    this.el = h('div', { class: 'layers' });
    parent.append(this.el);
    this.chips = h('div', { class: 'chip-row', role: 'radiogroup', 'aria-label': '3D layer' });
    this.legendBox = h('div', { class: 'legend-box' });
    this.colorBar = null;
    this.pickBox = h('div', { class: 'pick-box muted small' }, 'Tap the part to read the layer value there.');
    this.measureBox = h('div', { class: 'pick-box muted small' });
    const stock = (data.scene && data.scene.stock && data.scene.stock.size_m) || [0.1, 0.1, 0.1];
    this.sectionAxis = h('select', { class: 'input', 'aria-label': 'Section plane', on: { change: () => this.applySection() } },
      h('option', { value: '' }, 'no section'), ['x', 'y', 'z'].map((a) => h('option', { value: a }, `section normal to ${a}`)));
    this.sectionOffset = h('input', { type: 'range', class: 'range', min: '0', max: '1000', value: '500', 'aria-label': 'Section offset',
      on: { input: () => this.applySection() } });
    this.sectionFlip = h('input', { type: 'checkbox', on: { change: () => this.applySection() } });
    this.sectionText = h('span', { class: 'muted small' });
    this.stock = stock.map(Number);
    this.measure = h('input', { type: 'checkbox', on: { change: () => { const v = this.getView(); if (v) v.setMeasure(this.measure.checked); this.measureBox.textContent = this.measure.checked ? 'Pick two points on the part.' : ''; } } });
    this.followTool = h('input', { type: 'checkbox', on: { change: () => { const v = this.getView(); if (v) v.followTool(this.followTool.checked); } } });
    const visibility = h('div', { class: 'vis-row' }, VISIBLE.map(([key, label]) => {
      const box = h('input', { type: 'checkbox', checked: this.visible[key], on: { change: () => { this.visible[key] = box.checked; this.applyVisible(); } } });
      return h('label', { class: 'check' }, box, h('span', null, label));
    }));
    this.el.append(
      h('div', { class: 'layers-row' }, h('span', { class: 'label' }, 'Layer'), this.chips),
      this.legendBox,
      h('div', { class: 'layers-row wrap' },
        h('label', { class: 'check' }, this.measure, h('span', null, 'Measure')),
        h('label', { class: 'check' }, this.followTool, h('span', null, 'Follow the tool')),
        button('Reset camera', () => { const v = this.getView(); if (v) v.resetCamera(); }, { kind: 'small', iconName: 'camera' })),
      h('div', { class: 'layers-row wrap' }, this.sectionAxis, this.sectionOffset, h('label', { class: 'check' }, this.sectionFlip, h('span', null, 'Flip')), this.sectionText),
      h('details', { class: 'disclosure' }, h('summary', null, 'Show'), visibility),
      this.pickBox, this.measureBox);
    this._chipsPending();
    const find = () => layerAvailability(api, data).then((available) => { this.available = available; this._chips(); })
      .catch(() => { this.available = this.available || {}; this._chips(); });
    find();
    // a run that finishes writes the final attributes (finish, deviation, wear): look again
    data.addEventListener('refresh', () => { if (!data.active) { this.legends.clear(); find(); } });
  }

  _chipsPending() {
    clear(this.chips);
    this.chips.append(h('span', { class: 'muted small' }, 'Finding what the run recorded…'));
  }

  _chips() {
    clear(this.chips);
    const make = (name, label, enabled) => {
      const pressed = this.layer === name;
      const chip = h('button', {
        type: 'button', class: ['chip', pressed ? 'on' : ''], role: 'radio', 'aria-checked': String(pressed), disabled: !enabled,
        title: enabled ? '' : 'not recorded by this run',
        on: { click: () => this.setLayer(name) },
      }, label);
      this.chips.append(chip);
    };
    make(null, 'None', true);
    for (const layer of LAYERS) make(layer.name, layer.label, !!(this.available && this.available[layer.name]));
  }

  async setLayer(name) {
    this.layer = name;
    this._chips();
    const view = this.getView();
    if (view) view.setLayer(name);
    clear(this.legendBox);
    if (!name) return;
    const fromView = view ? view.legend(name) : null;
    let legend = null;
    if (fromView && (Number.isFinite(fromView.min) || fromView.categories)) {
      legend = fromView.categories ? { categories: fromView.categories, source: '3D view' }
        : { map: fromView.map || LAYER_MAPS[name], domain: [fromView.min, fromView.max], unit: fromView.unit || '', source: '3D view' };
    } else {
      if (!this.legends.has(name)) this.legends.set(name, layerLegend(this.api, this.data, name, this.available[name]));
      this.legendBox.append(h('span', { class: 'muted small' }, 'Computing the colour scale…'));
      legend = await this.legends.get(name);
      if (this.layer !== name) return;
    }
    clear(this.legendBox);
    if (!legend) { this.legendBox.append(notice('The run records no values for this layer.')); return; }
    const meta = LAYERS.find((l) => l.name === name);
    this.legendBox.append(h('div', { class: 'legend-title' }, `${meta.label}${legend.unit ? ` (${legend.unit})` : ''}`));
    this.colorBar = new ColorBar(this.legendBox, { label: `${meta.label} colour scale` });
    this.colorBar.setData(legend);
    this.legendBox.append(h('div', { class: 'muted small' },
      legend.domain ? `from ${num(legend.domain[0])} to ${num(legend.domain[1])} ${legend.unit} · ` : '', legend.source));
  }

  applySection() {
    const axis = this.sectionAxis.value || null;
    const view = this.getView();
    if (!axis) { this.sectionText.textContent = ''; if (view) view.setSection({ axis: null, offset_m: 0, flip: false }); return; }
    const k = { x: 0, y: 1, z: 2 }[axis];
    const offset = this.stock[k] * Number(this.sectionOffset.value) / 1000;
    this.sectionText.textContent = `${axis} = ${fixed(offset * 1e3, 2)} mm (stock frame)`;
    if (view) view.setSection({ axis, offset_m: offset, flip: this.sectionFlip.checked });
  }

  applyVisible() {
    const view = this.getView();
    if (view) view.setVisible({ ...this.visible });
  }

  /** Apply the current state to a (new) view. */
  attach(view) {
    if (!view) return;
    view.setVisible({ ...this.visible });
    if (this.layer) view.setLayer(this.layer);
    this.applySection();
  }

  showPick(event) {
    if (!event) return;
    const p = event.point_m || [];
    const meta = LAYERS.find((l) => l.name === this.layer);
    let value = '';
    if (event.layerValue !== undefined && event.layerValue !== null && this.layer) {
      const v = Number(event.layerValue);
      if (Number.isFinite(v)) {
        value = this.layer === 'temperature' ? `${fixed(v - 273.15, 1)} °C`
          : this.layer === 'chips' ? `${num(v * 1e3)} mm`
            : this.layer === 'coolant' ? (v ? 'wetted' : 'dry')
              : `${num(v * 1e6)} µm`;
      } else {
        value = String(event.layerValue);
      }
    }
    this.pickBox.textContent = `Point ${p.map((x) => fixed(Number(x) * 1e3, 3)).join(', ')} mm`
      + (event.normal ? ` · normal ${event.normal.map((x) => fixed(Number(x), 2)).join(', ')}` : '')
      + (value ? ` · ${meta ? meta.label : 'value'} ${value}` : '');
  }

  showMeasure(event) {
    if (!event) return;
    this.measureBox.textContent = Number.isFinite(Number(event.distance_m)) ? `Distance ${fixed(Number(event.distance_m) * 1e3, 3)} mm` : '';
  }
}

