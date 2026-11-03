/* Transport: play/pause at real speed or a multiple, previous/next block, the time readout and
 * the scrubber. The scrubber shows the timeline's blocks by time category (cutting, air cutting,
 * rapid, ...), the findings by severity and the keyframes; drag it (mouse, pen or finger) to scrub,
 * or use the arrow keys (shift: larger steps). */

import { h, icon, clear } from './dom.js';
import { SPEEDS } from './timebase.js';
import { fitCanvas, observeWidth, clamp } from '../plot/canvas.js';
import { chrome, withAlpha, onSchemeChange } from '../plot/colormap.js';
import { clock, num } from '../plot/units.js';

/** Fixed colour slots of the time categories (colour follows the category, never its rank). */
export const CATEGORY_SLOT = { cutting: 0, air_cutting: 1, rapid: 2, feed: 3, tool_change: 4, acceleration: 5, spindle_ramp: 6 };
export const KIND_CATEGORY = { rapid: 'rapid', dwell: 'dwell', tool_change: 'tool_change', turret_index: 'tool_change', spindle: 'spindle_ramp',
  coolant: 'coolant_delay', probe: 'probing', tool_setter: 'probing', stop: 'program_stop', sync: 'path_wait', linear: 'feed', arc: 'feed' };

export function blockCategory(block) {
  const ticks = block.ticks;
  if (ticks && typeof ticks === 'object') {
    let best = null;
    let most = -1;
    for (const [name, value] of Object.entries(ticks)) if (value > most) { most = value; best = name; }
    if (best) return best;
  }
  return KIND_CATEGORY[block.kind] || 'auxiliary';
}

export function categoryColor(category, c = chrome()) {
  const slot = CATEGORY_SLOT[category];
  return slot === undefined ? withAlpha(c.ink2, 0.45) : c.series[slot];
}

export class Transport {
  constructor(parent, { time, data, path = () => 0 }) {
    this.time = time;
    this.data = data;
    this.path = path;
    this.el = h('div', { class: 'transport', role: 'group', 'aria-label': 'Playback' });
    this.playBtn = h('button', { type: 'button', class: 'btn round play', 'aria-label': 'Play', on: { click: () => time.toggle() } }, icon('play'));
    const prev = h('button', { type: 'button', class: 'btn round', 'aria-label': 'Previous block', title: 'Previous block',
      on: { click: () => { time.pause(); time.seek(data.stepBlock(time.t, -1, path()), 'step'); } } }, icon('prev'));
    const next = h('button', { type: 'button', class: 'btn round', 'aria-label': 'Next block', title: 'Next block',
      on: { click: () => { time.pause(); time.seek(data.stepBlock(time.t, 1, path()), 'step'); } } }, icon('next'));
    this.speed = h('select', { class: 'input speed', 'aria-label': 'Playback speed', on: { change: () => time.setSpeed(Number(this.speed.value)) } },
      SPEEDS.map((s) => h('option', { value: String(s), selected: s === 1 }, `${s}×`)));
    this.readout = h('span', { class: 'time-readout', 'aria-live': 'off' });
    this.canvas = h('canvas', { class: 'scrubber', tabindex: '0', role: 'slider', 'aria-label': 'Time', 'aria-valuemin': '0' });
    this.scrubBox = h('div', { class: 'scrub-box' }, this.canvas);
    this.hover = h('div', { class: 'tplot-tip scrub-tip', hidden: true });
    this.scrubBox.append(this.hover);
    this.el.append(h('div', { class: 'transport-buttons' }, prev, this.playBtn, next, this.speed, this.readout), this.scrubBox);
    parent.append(this.el);
    this.base = document.createElement('canvas');
    this.width = 0;
    this._unobserve = observeWidth(this.scrubBox, (w) => { this.width = w; this.drawBase(); });
    this._unscheme = onSchemeChange(() => this.drawBase());
    time.addEventListener('time', () => this.present());
    time.addEventListener('state', (e) => this._state(e.detail.playing));
    time.addEventListener('duration', () => this.drawBase());
    data.addEventListener('findings', () => this.drawBase());
    data.addEventListener('timeline', () => this.drawBase());
    this._bind();
  }

  _state(playing) {
    clear(this.playBtn);
    this.playBtn.append(icon(playing ? 'pause' : 'play'));
    this.playBtn.setAttribute('aria-label', playing ? 'Pause' : 'Play');
  }

  _x(t) {
    const d = this.time.duration || 1;
    return 8 + (this.width - 16) * clamp(t / d, 0, 1);
  }

  _t(x) {
    return clamp((x - 8) / Math.max(1, this.width - 16), 0, 1) * this.time.duration;
  }

  drawBase() {
    const width = this.width || this.scrubBox.clientWidth || 300;
    this.width = width;
    const height = 40;
    const { ctx } = fitCanvas(this.base, width, height);
    fitCanvas(this.canvas, width, height);
    const c = chrome();
    this._c = c;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = c.grid;
    ctx.fillRect(8, 14, width - 16, 12);
    const tickS = this.data.tickS;
    const entry = this.data.timeline.paths.get(this.path()) || this.data.timeline.paths.values().next().value;
    if (entry) {
      for (const block of entry.blocks) {
        const x0 = this._x(block.tick0 * tickS);
        const x1 = this._x(block.tick1 * tickS);
        ctx.fillStyle = categoryColor(blockCategory(block), c);
        ctx.fillRect(x0, 14, Math.max(0.6, x1 - x0), 12);
      }
    }
    for (const kf of this.data.keyframes) {
      ctx.fillStyle = c.muted;
      ctx.fillRect(Math.round(this._x(Number(kf.time_s))), 9, 1, 4);
    }
    for (const f of this.data.findings) {
      const x = this._x(f.time_s);
      ctx.fillStyle = c[f.severity] || c.info;
      ctx.beginPath();
      ctx.moveTo(x, 28); ctx.lineTo(x - 4, 36); ctx.lineTo(x + 4, 36); ctx.closePath();
      ctx.fill();
    }
    this.canvas.setAttribute('aria-valuemax', String(this.time.duration));
    this.present();
  }

  present() {
    const c = this._c || chrome();
    const ctx = this.canvas.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.drawImage(this.base, 0, 0);
    const ratio = this.base.width / (this.width || 1);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    const x = this._x(this.time.t);
    ctx.fillStyle = c.cursor;
    ctx.fillRect(x - 1, 6, 2, 28);
    ctx.beginPath();
    ctx.arc(x, 20, 7, 0, 2 * Math.PI);
    ctx.fillStyle = c.surface;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(x, 20, 5, 0, 2 * Math.PI);
    ctx.fillStyle = c.cursor;
    ctx.fill();
    this.readout.textContent = `${clock(this.time.t)} / ${clock(this.time.duration)}`;
    this.canvas.setAttribute('aria-valuenow', String(this.time.t));
    this.canvas.setAttribute('aria-valuetext', `${num(this.time.t)} seconds of ${num(this.time.duration)}`);
  }

  _tipAt(x) {
    const t = this._t(x);
    const block = this.data.blockAt(t, this.path());
    const near = this.data.findings.filter((f) => Math.abs(this._x(f.time_s) - x) < 6);
    const lines = [`${num(t)} s`];
    if (block) lines.push(`line ${block.line}: ${block.block || ''}`);
    for (const f of near.slice(0, 3)) lines.push(`${f.severity}: ${f.message}`);
    this.hover.replaceChildren(...lines.map((l, i) => h('div', { class: i === 0 ? 'tplot-tip-t' : null }, l)));
    this.hover.hidden = false;
    const w = this.hover.offsetWidth || 160;
    this.hover.style.left = `${clamp(x - w / 2, 0, Math.max(0, this.width - w))}px`;
    this.hover.style.bottom = '44px';
  }

  _bind() {
    const canvas = this.canvas;
    let dragging = false;
    const local = (e) => e.clientX - canvas.getBoundingClientRect().left;
    canvas.addEventListener('pointerdown', (e) => {
      dragging = true;
      try { canvas.setPointerCapture(e.pointerId); } catch { /* pointer already gone: plain events still arrive */ }
      this.wasPlaying = this.time.playing;
      this.time.pause();
      this.time.seek(this._t(local(e)), 'scrub');
    });
    canvas.addEventListener('pointermove', (e) => {
      const x = local(e);
      if (dragging) { this.time.seek(this._t(x), 'scrub'); if (e.pointerType !== 'mouse') this._tipAt(x); return; }
      if (e.pointerType === 'mouse') this._tipAt(x);
    });
    const end = () => {
      if (!dragging) return;
      dragging = false;
      this.hover.hidden = true;
      if (this.wasPlaying) this.time.play();
    };
    canvas.addEventListener('pointerup', end);
    canvas.addEventListener('pointercancel', end);
    canvas.addEventListener('pointerleave', () => { if (!dragging) this.hover.hidden = true; });
    canvas.addEventListener('keydown', (e) => {
      const d = this.time.duration;
      const step = e.shiftKey ? d / 10 : d / 100;
      if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') { this.time.seek(this.time.t - step, 'key'); e.preventDefault(); }
      else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') { this.time.seek(this.time.t + step, 'key'); e.preventDefault(); }
      else if (e.key === 'Home') { this.time.seek(0, 'key'); e.preventDefault(); }
      else if (e.key === 'End') { this.time.seek(d, 'key'); e.preventDefault(); }
    });
  }

  destroy() { this._unobserve(); this._unscheme(); }
}

