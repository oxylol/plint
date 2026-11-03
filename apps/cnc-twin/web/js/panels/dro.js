/* DRO: the machine as the control's position display shows it at the playback time: machine
 * coordinates of every axis, the tool tip in work coordinates, the tool, spindle speed and load,
 * feed and the overrides. Positions come from frames.bin (the UI never re-implements
 * kinematics); spindle speed and feed are differences of consecutive frames; load, power and
 * overrides are tick-tier channels when the run records them ("not recorded" otherwise). */

import { h, clear } from './dom.js';
import { DASH, fixed, num } from '../plot/units.js';

const LOAD = 'cutting.spindle_load_fraction';
const POWER = 'cutting.spindle_power_W';
const FEED = 'motion.feed_m_s';

export function droColumns(channels) {
  const names = new Set(channels.filter((c) => c.tier === 'tick').map((c) => c.name));
  const overrides = channels.filter((c) => /override/i.test(c.name) && c.tier !== 'fast');
  return {
    load: names.has(LOAD) ? LOAD : null,
    power: names.has(POWER) ? POWER : null,
    feed: names.has(FEED) ? FEED : null,
    overrides: overrides.map((c) => ({ name: c.name, tier: c.tier, kind: /rapid/i.test(c.name) ? 'rapid' : /spindle|speed/i.test(c.name) ? 'spindle' : 'feed' })),
  };
}

function mat3(pose) {
  if (!Array.isArray(pose) || pose.length < 3) return null;
  return [0, 1, 2].map((i) => [0, 1, 2].map((j) => Number(pose[i][j])));
}

export class Dro {
  constructor(parent, { data, frames, sampler, columns }) {
    this.data = data;
    this.frames = frames;
    this.sampler = sampler;
    this.columns = columns;
    this.path = 0;
    this.el = h('div', { class: 'dro', 'aria-live': 'off' });
    parent.append(this.el);
    const scene = data.scene || {};
    this.zero = Array.isArray(scene.work_zero_S_m) ? scene.work_zero_S_m.map(Number) : null;
    this.R = mat3(scene.stock && scene.stock.pose_W_S);
    const op = (data.project.operations || [])[0];
    const offsets = op && op.setup && op.setup.work_offsets_m;
    this.offsetName = offsets && typeof offsets === 'object' ? Object.keys(offsets)[0] : null;
    this._build();
  }

  _cell(label, cls = '') {
    const value = h('span', { class: 'dro-v' }, DASH);
    const cell = h('div', { class: ['dro-cell', cls] }, h('span', { class: 'dro-k' }, label), value);
    return { cell, value };
  }

  _build() {
    clear(this.el);
    const axes = this.frames ? this.frames.axes : [];
    this.machine = axes.map((a) => this._cell(a.name, 'axis'));
    const relative = axes.some((a) => a.kind !== 'rotary' && !Number.isFinite(Number(a.origin)));
    this.work = ['X', 'Y', 'Z'].map((n) => this._cell(n, 'axis'));
    this.tool = this._cell('Tool', 'wide');
    this.spindle = this._cell('Spindle', 'wide');
    this.load = this._cell('Load');
    this.feed = this._cell('Feed');
    this.override = this._cell('Override', 'wide');
    this.block = this._cell('Block', 'wide');
    this.el.append(
      h('div', { class: 'dro-group' }, h('div', { class: 'dro-title' }, relative ? 'Machine (relative to the stock origin, mm/°)' : 'Machine (mm/°)'),
        h('div', { class: 'dro-axes' }, this.machine.map((c) => c.cell)), axes.length ? null : h('p', { class: 'muted small' }, 'frames.bin not recorded')),
      h('div', { class: 'dro-group' }, h('div', { class: 'dro-title' }, `Work${this.offsetName ? ` ${this.offsetName}` : ''} (tool tip, mm)`),
        h('div', { class: 'dro-axes' }, this.work.map((c) => c.cell))),
      h('div', { class: 'dro-grid' }, this.tool.cell, this.spindle.cell, this.load.cell, this.feed.cell, this.override.cell, this.block.cell));
  }

  update(t) {
    const state = this.frames ? this.frames.at(t, this.path) : null;
    if (this.frames) this.frames.request(t);
    if (state) {
      state.axes.forEach((axis, i) => {
        const cell = this.machine[i];
        if (!cell) return;
        cell.value.textContent = axis.kind === 'rotary' ? fixed(axis.absolute * 180 / Math.PI, 3) : fixed(axis.absolute * 1e3, 3);
      });
      if (state.tip) {
        let d = this.zero ? state.tip.map((v, k) => v - this.zero[k]) : state.tip.slice();
        if (this.R) d = this.R.map((row) => row[0] * d[0] + row[1] * d[1] + row[2] * d[2]);
        this.work.forEach((cell, k) => { cell.value.textContent = fixed(d[k] * 1e3, 3); });
      }
    }
    const block = this.data.blockAt(t, this.path);
    const number = block && block.tool !== undefined ? block.tool : null;
    if (number !== null && number !== undefined) {
      const info = this.data.toolInfo(number);
      const id = (info.scene && info.scene.tool_id) || (info.project && info.project.tool && info.project.tool.id) || '';
      const dia = info.scene && info.scene.diameter_m ? ` · Ø${num(info.scene.diameter_m * 1e3, 4)} mm` : '';
      this.tool.value.textContent = `T${number}${id ? ` ${id}` : ''}${dia}`;
    } else {
      this.tool.value.textContent = 'none';
    }
    let omega = state ? state.omega : NaN;
    if (Number.isFinite(omega) && Math.abs(omega) < 1e-3) omega = 0;
    const rpm = Number.isFinite(omega) ? Math.abs(omega) * 60 / (2 * Math.PI) : null;
    const dir = !Number.isFinite(omega) || omega === 0 ? '' : omega < 0 ? ' CW (M03)' : ' CCW (M04)';
    const power = this.columns.power && this.sampler ? this.sampler.value(this.columns.power, t) : null;
    this.spindle.value.textContent = rpm === null ? DASH : `${fixed(rpm, 0)} rpm${dir}${power !== null ? ` · ${num(power / 1000, 3)} kW` : ''}`;
    const load = this.columns.load && this.sampler ? this.sampler.value(this.columns.load, t) : null;
    this.load.value.textContent = load === null ? (this.columns.load ? DASH : 'not recorded') : `${fixed(load * 100, 1)} %`;
    const feedChannel = this.columns.feed && this.sampler ? this.sampler.value(this.columns.feed, t) : null;
    const feed = feedChannel !== null ? feedChannel : (state ? state.feed : null);
    this.feed.value.textContent = Number.isFinite(feed) ? `${fixed(feed * 60000, 0)} mm/min` : DASH;
    if (this.columns.overrides.length && this.sampler) {
      this.override.value.textContent = this.columns.overrides.map((o) => {
        const v = this.sampler.value(o.name, t);
        return `${o.kind} ${v === null ? DASH : `${fixed(v * 100, 0)} %`}`;
      }).join(' · ');
    } else {
      this.override.value.textContent = 'not recorded by this run';
    }
    this.block.value.textContent = block ? `${block.op ? `${block.op} · ` : ''}line ${block.line}: ${block.block || ''} (${String(block.kind || '').replace(/_/g, ' ')})` : DASH;
  }
}
