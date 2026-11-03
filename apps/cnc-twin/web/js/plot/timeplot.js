/* Stacked time-series plots on canvas, sharing one time range and one time cursor (PlotGroup).
 *
 * A series is {label, color, t, v, lo?, hi?, gap?}: times in seconds (simulated), values already
 * in display units; `lo`/`hi` draw the min/max envelope as a band (tick-tier envelopes of fast
 * channels, or buckets of M4-downsampled data), `v` the line. Lines break at NaN and where the time
 * step exceeds `gap` (fast recording windows, missing parts). Interaction, the same on every plot of
 * a group: click or tap sets the time (onSeek), drag or one-finger swipe pans, ctrl/cmd + wheel and
 * two-finger pinch zoom, shift + wheel pans, double-click resets; keyboard: arrows pan, + / - zoom,
 * 0 resets. Hovering (or tapping) shows every series' value at that time. Data are drawn once into
 * a cache canvas; cursor moves only blit it, so playback at 60 frames per second stays cheap.
 */

import { fitCanvas, observeWidth, setFont, clamp, fitText } from './canvas.js';
import { niceTicks, niceStep, tickLabel, extent, paddedDomain, linear, bisectRight, nearestIndex } from './scale.js';
import { chrome, withAlpha, onSchemeChange } from './colormap.js';
import { num, DASH } from './units.js';

const MARGIN = { left: 58, right: 12, top: 8, bottom: 22 };

export class PlotGroup {
  constructor({ duration = 1, onSeek = null, onRange = null, minSpan = 1e-4 } = {}) {
    this.plots = new Set();
    this.full = [0, Math.max(duration, 1e-6)];
    this.range = this.full.slice();
    this.cursor = null;
    this.follow = true;
    this.onSeek = onSeek;
    this.onRange = onRange;
    this.minSpan = minSpan;
    this._timer = null;
  }

  get zoomed() {
    return this.range[0] > this.full[0] + 1e-12 || this.range[1] < this.full[1] - 1e-12;
  }

  /** A longer run (still running): the full range grows; `emit` asks for the data to be re-fetched. */
  setDuration(duration, { emit = true } = {}) {
    const wasFull = !this.zoomed;
    this.full = [0, Math.max(duration, 1e-6)];
    if (wasFull) this.range = this.full.slice();
    this.setRange(this.range, { emit: emit && wasFull });
  }

  add(plot) { this.plots.add(plot); plot.setRange(this.range); plot.setCursor(this.cursor); }
  remove(plot) { this.plots.delete(plot); }

  setRange([a, b], { emit = true } = {}) {
    const span = Math.max(this.minSpan, Math.min(b - a, this.full[1] - this.full[0]));
    let lo = a;
    if (lo < this.full[0]) lo = this.full[0];
    if (lo + span > this.full[1]) lo = this.full[1] - span;
    this.range = [lo, lo + span];
    for (const plot of this.plots) plot.setRange(this.range);
    if (emit && this.onRange) {
      clearTimeout(this._timer);
      this._timer = setTimeout(() => this.onRange(this.range.slice()), 220);
    }
  }

  zoom(factor, center = null) {
    const [a, b] = this.range;
    const c = center === null ? (a + b) / 2 : center;
    const span = (b - a) / factor;
    const f = (c - a) / (b - a || 1);
    this.setRange([c - f * span, c - f * span + span]);
  }

  pan(dt) { this.setRange([this.range[0] + dt, this.range[1] + dt]); }
  reset() { this.setRange(this.full.slice()); }

  setCursor(t) {
    this.cursor = t;
    if (this.follow && this.zoomed && t !== null && Number.isFinite(t)) {
      const [a, b] = this.range;
      if (t < a || t > b) {
        const span = b - a;
        this.setRange([t - 0.1 * span, t + 0.9 * span]);
      }
    }
    for (const plot of this.plots) plot.setCursor(t);
  }

  seek(t) { if (this.onSeek) this.onSeek(clamp(t, this.full[0], this.full[1])); }
}

export class TimePlot {
  constructor(parent, { group, height = 140, title = '', unit = '', note = '' } = {}) {
    this.group = group;
    this.height = height;
    this.series = [];
    this.cursor = null;
    this.range = group ? group.range.slice() : [0, 1];
    this.hoverX = null;
    this.width = 0;
    this.loading = false;
    this.emptyText = 'no data in this range';
    this.el = document.createElement('div');
    this.el.className = 'tplot';
    const head = document.createElement('div');
    head.className = 'tplot-head';
    this.titleEl = document.createElement('span');
    this.titleEl.className = 'tplot-title';
    this.unitEl = document.createElement('span');
    this.unitEl.className = 'tplot-unit';
    this.legendEl = document.createElement('span');
    this.legendEl.className = 'tplot-legend';
    this.toolsEl = document.createElement('span');
    this.toolsEl.className = 'tplot-tools';
    head.append(this.titleEl, this.unitEl, this.legendEl, this.toolsEl);
    this.noteEl = document.createElement('div');
    this.noteEl.className = 'tplot-note';
    this.body = document.createElement('div');
    this.body.className = 'tplot-body';
    this.canvas = document.createElement('canvas');
    this.canvas.tabIndex = 0;
    this.canvas.setAttribute('role', 'img');
    this.tip = document.createElement('div');
    this.tip.className = 'tplot-tip';
    this.tip.hidden = true;
    this.body.append(this.canvas, this.tip);
    this.el.append(head, this.noteEl, this.body);
    this.cache = document.createElement('canvas');
    parent.append(this.el);
    this.setTitle(title, unit);
    this.setNote(note);
    this._unobserve = observeWidth(this.body, (w) => { this.width = w; this.draw(); });
    this._unscheme = onSchemeChange(() => this.draw());
    this._bindPointer();
    if (group) group.add(this);
  }

  setTitle(title, unit = '') {
    this.title = title;
    this.unit = unit;
    this.titleEl.textContent = title;
    this.unitEl.textContent = unit ? `(${unit})` : '';
    this.canvas.setAttribute('aria-label', `${title}${unit ? ` in ${unit}` : ''} over time`);
  }

  setNote(text) {
    this.noteEl.textContent = text || '';
    this.noteEl.hidden = !text;
  }

  setLoading(on) {
    this.loading = on;
    this.el.classList.toggle('is-loading', !!on);
  }

  /** Replace the series (see the module comment). */
  setSeries(series, { emptyText } = {}) {
    this.series = series || [];
    if (emptyText) this.emptyText = emptyText;
    this.legendEl.replaceChildren();
    if (this.series.length > 1) {
      for (const s of this.series) {
        const item = document.createElement('span');
        item.className = 'tplot-key';
        const key = document.createElement('i');
        key.style.background = s.color;
        if (s.lo && s.hi && !s.v) key.className = 'band';
        item.append(key, document.createTextNode(s.label));
        this.legendEl.append(item);
      }
    }
    this.draw();
  }

  setRange(range) {
    this.range = range.slice();
    this.draw();
  }

  setCursor(t) {
    this.cursor = t;
    this.present();
  }

  /** Statistics of every series over the visible range: [{label, min, mean, max, atCursor}]. */
  stats() {
    const [a, b] = this.range;
    return this.series.map((s) => {
      const i0 = Math.max(0, bisectRight(s.t, a));
      const i1 = Math.min(s.t.length - 1, bisectRight(s.t, b));
      let min = Infinity; let max = -Infinity; let sum = 0; let n = 0;
      for (let i = i0; i <= i1; i++) {
        const lo = s.lo ? s.lo[i] : s.v[i];
        const hi = s.hi ? s.hi[i] : s.v[i];
        const mid = s.v ? s.v[i] : (lo + hi) / 2;
        if (Number.isFinite(lo) && lo < min) min = lo;
        if (Number.isFinite(hi) && hi > max) max = hi;
        if (Number.isFinite(mid)) { sum += mid; n++; }
      }
      let atCursor = null;
      if (this.cursor !== null && s.t.length) {
        const j = nearestIndex(s.t, this.cursor);
        const value = s.v ? s.v[j] : (s.lo[j] + s.hi[j]) / 2;
        atCursor = Number.isFinite(value) ? value : null;
      }
      return { label: s.label, min: n ? min : null, mean: n ? sum / n : null, max: n ? max : null, atCursor };
    });
  }

  destroy() {
    if (this.group) this.group.remove(this);
    this._unobserve();
    this._unscheme();
    this.el.remove();
  }

  // ------------------------------------------------------------------ drawing
  _layout() {
    const width = this.width || this.body.clientWidth || 300;
    return { width, height: this.height, x0: MARGIN.left, x1: width - MARGIN.right, y0: MARGIN.top, y1: this.height - MARGIN.bottom };
  }

  _domain() {
    const [a, b] = this.range;
    const arrays = [];
    for (const s of this.series) {
      const i0 = Math.max(0, bisectRight(s.t, a));
      const i1 = Math.min(s.t.length, bisectRight(s.t, b) + 2);
      for (const key of ['v', 'lo', 'hi']) if (s[key]) arrays.push(s[key].subarray ? s[key].subarray(i0, i1) : s[key].slice(i0, i1));
    }
    return paddedDomain(extent(...arrays));
  }

  draw() {
    const L = this._layout();
    if (!this.cache) return;
    const { ctx } = fitCanvas(this.cache, L.width, L.height);
    fitCanvas(this.canvas, L.width, L.height);
    const c = chrome();
    this._chrome = c;
    ctx.clearRect(0, 0, L.width, L.height);
    const [a, b] = this.range;
    const x = linear([a, b], [L.x0, L.x1]);
    const yDomain = this._domain();
    const y = linear(yDomain, [L.y1, L.y0]);
    this._scales = { x, y, L };
    setFont(ctx, 11, 500, c.font);
    ctx.lineWidth = 1;
    // grid and y ticks
    const yTicks = niceTicks(yDomain[0], yDomain[1], Math.max(2, Math.floor((L.y1 - L.y0) / 32)));
    const yStep = yTicks.length > 1 ? yTicks[1] - yTicks[0] : niceStep(yDomain[1] - yDomain[0]);
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (const value of yTicks) {
      const py = Math.round(y(value)) + 0.5;
      ctx.strokeStyle = c.grid;
      ctx.beginPath(); ctx.moveTo(L.x0, py); ctx.lineTo(L.x1, py); ctx.stroke();
      ctx.fillStyle = c.muted;
      ctx.fillText(fitText(ctx, tickLabel(value, yStep), MARGIN.left - 8), L.x0 - 6, py);
    }
    // x ticks (seconds)
    const xTicks = niceTicks(a, b, Math.max(2, Math.floor((L.x1 - L.x0) / 70)));
    const xStep = xTicks.length > 1 ? xTicks[1] - xTicks[0] : niceStep(b - a);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (const value of xTicks) {
      const px = Math.round(x(value)) + 0.5;
      ctx.strokeStyle = c.grid;
      ctx.beginPath(); ctx.moveTo(px, L.y0); ctx.lineTo(px, L.y1); ctx.stroke();
      ctx.fillStyle = c.muted;
      const label = tickLabel(value, xStep);
      const half = ctx.measureText(label).width / 2;
      ctx.fillText(label, clamp(px, L.x0 + half, L.width - half - 1), L.y1 + 5);
    }
    ctx.fillStyle = c.muted;
    ctx.textAlign = 'right';
    ctx.fillText('t (s)', L.x0 - 6, L.y1 + 5);
    ctx.strokeStyle = c.axis;
    ctx.beginPath(); ctx.moveTo(L.x0, L.y1 + 0.5); ctx.lineTo(L.x1, L.y1 + 0.5); ctx.stroke();
    // data
    ctx.save();
    ctx.beginPath();
    ctx.rect(L.x0, L.y0 - 1, L.x1 - L.x0, L.y1 - L.y0 + 2);
    ctx.clip();
    let drawn = 0;
    for (const s of this.series) drawn += this._drawSeries(ctx, s, x, y, a, b, L);
    ctx.restore();
    this.el.dataset.points = String(drawn);
    if (!drawn) {
      ctx.fillStyle = c.muted;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      setFont(ctx, 12, 500, c.font);
      ctx.fillText(this.loading ? 'loading…' : this.emptyText, (L.x0 + L.x1) / 2, (L.y0 + L.y1) / 2);
    }
    const stats = this.series.length ? this.stats() : [];
    const summary = stats.filter((s) => s.min !== null).map((s) => `${s.label} from ${num(s.min)} to ${num(s.max)}`).join('; ');
    this.canvas.setAttribute('aria-label', `${this.title}${this.unit ? ` in ${this.unit}` : ''} from ${num(a)} s to ${num(b)} s${summary ? `: ${summary}` : ''}`);
    this.present();
  }

  _drawSeries(ctx, s, x, y, a, b, L) {
    const n = s.t.length;
    if (!n) return 0;
    const i0 = Math.max(0, bisectRight(s.t, a) - 1);
    const i1 = Math.min(n - 1, bisectRight(s.t, b) + 1);
    const gap = s.gap || Infinity;
    let drawn = 0;
    if (s.lo && s.hi) {
      ctx.fillStyle = withAlpha(s.color, s.v ? 0.2 : 0.42);
      let start = -1;
      const flush = (end) => {
        if (start < 0) return;
        ctx.beginPath();
        for (let i = start; i <= end; i++) ctx.lineTo(x(s.t[i]), y(s.hi[i]));
        for (let i = end; i >= start; i--) ctx.lineTo(x(s.t[i]), y(s.lo[i]));
        ctx.closePath();
        ctx.fill();
        if (end === start) ctx.fillRect(x(s.t[start]) - 0.5, y(s.hi[start]), 1, Math.max(1, y(s.lo[start]) - y(s.hi[start])));
        drawn++;
        start = -1;
      };
      for (let i = i0; i <= i1; i++) {
        const ok = Number.isFinite(s.lo[i]) && Number.isFinite(s.hi[i]);
        if (!ok) { flush(i - 1); continue; }
        if (start >= 0 && s.t[i] - s.t[i - 1] > gap) flush(i - 1);
        if (start < 0) start = i;
      }
      flush(i1);
    }
    if (s.v) {
      const visible = i1 - i0 + 1;
      const pixels = L.x1 - L.x0;
      ctx.strokeStyle = s.color;
      ctx.lineWidth = s.dense || visible > pixels * 1.5 ? 1 : 2;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.beginPath();
      let pen = false;
      let lone = null;
      for (let i = i0; i <= i1; i++) {
        const value = s.v[i];
        if (!Number.isFinite(value)) { pen = false; continue; }
        const px = x(s.t[i]);
        const py = y(value);
        if (pen && s.t[i] - s.t[i - 1] <= gap) {
          ctx.lineTo(px, py);
        } else {
          ctx.moveTo(px, py);
          lone = [px, py];
        }
        pen = true;
        drawn++;
      }
      ctx.stroke();
      if (visible <= 40 || (drawn === 1 && lone)) {
        ctx.fillStyle = s.color;
        for (let i = i0; i <= i1; i++) {
          if (!Number.isFinite(s.v[i])) continue;
          ctx.beginPath();
          ctx.arc(x(s.t[i]), y(s.v[i]), 3, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
    }
    return drawn;
  }

  /** Blit the cached data layer and draw the cursor and the hover crosshair. */
  present() {
    if (!this._scales) return;
    const { x, L } = this._scales;
    const ratio = this.cache.width / L.width || 1;
    const ctx = this.canvas.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.drawImage(this.cache, 0, 0);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    const c = this._chrome || chrome();
    if (this.hoverX !== null) {
      ctx.strokeStyle = c.axis;
      ctx.lineWidth = 1;
      const px = Math.round(this.hoverX) + 0.5;
      ctx.beginPath(); ctx.moveTo(px, L.y0); ctx.lineTo(px, L.y1); ctx.stroke();
    }
    if (this.cursor !== null && Number.isFinite(this.cursor)) {
      const px = x(this.cursor);
      if (px >= L.x0 - 1 && px <= L.x1 + 1) {
        ctx.strokeStyle = c.cursor;
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(px, L.y0 - 2); ctx.lineTo(px, L.y1); ctx.stroke();
      }
    }
  }

  _showTip(px) {
    if (!this._scales || !this.series.length) { this.tip.hidden = true; return; }
    const { x, L } = this._scales;
    const t = x.invert(clamp(px, L.x0, L.x1));
    const rows = [];
    for (const s of this.series) {
      if (!s.t.length) continue;
      const j = nearestIndex(s.t, t);
      const value = s.v ? s.v[j] : null;
      const lo = s.lo ? s.lo[j] : null;
      const hi = s.hi ? s.hi[j] : null;
      let text = DASH;
      if (Number.isFinite(value)) text = num(value);
      if (Number.isFinite(lo) && Number.isFinite(hi) && (lo !== hi || !Number.isFinite(value))) {
        text = Number.isFinite(value) ? `${num(value)} (${num(lo)} … ${num(hi)})` : `${num(lo)} … ${num(hi)}`;
      }
      rows.push([s.color, text, s.label]);
    }
    this.tip.replaceChildren();
    const head = document.createElement('div');
    head.className = 'tplot-tip-t';
    head.textContent = `${num(t)} s`;
    this.tip.append(head);
    for (const [color, text, label] of rows) {
      const row = document.createElement('div');
      const key = document.createElement('i');
      key.style.background = color;
      const value = document.createElement('b');
      value.textContent = `${text}${this.unit ? ` ${this.unit}` : ''}`;
      const name = document.createElement('span');
      name.textContent = label;
      row.append(key, value, name);
      this.tip.append(row);
    }
    this.tip.hidden = false;
    const width = this.tip.offsetWidth || 160;
    const left = px + 14 + width > L.width ? px - 14 - width : px + 14;
    this.tip.style.left = `${Math.max(0, left)}px`;
    this.tip.style.top = `${L.y0 + 4}px`;
  }

  _bindPointer() {
    const canvas = this.canvas;
    const pointers = new Map();
    let drag = null;
    let pinch = null;
    let tipTimer = null;
    const local = (e) => {
      const rect = canvas.getBoundingClientRect();
      return { x: e.clientX - rect.left, y: e.clientY - rect.top };
    };
    const plotWidth = () => (this._scales ? this._scales.L.x1 - this._scales.L.x0 : 1);
    canvas.addEventListener('pointerdown', (e) => {
      if (e.button !== 0 && e.pointerType === 'mouse') return;
      try { canvas.setPointerCapture(e.pointerId); } catch { /* pointer already gone: plain events still arrive */ }
      pointers.set(e.pointerId, local(e));
      if (pointers.size === 1) {
        const p = local(e);
        drag = { x: p.x, range: this.group ? this.group.range.slice() : this.range.slice(), moved: false };
        pinch = null;
      } else if (pointers.size === 2 && this.group) {
        const [p, q] = [...pointers.values()];
        pinch = { dist: Math.max(10, Math.abs(p.x - q.x)), mid: (p.x + q.x) / 2, range: this.group.range.slice() };
        drag = null;
      }
    });
    canvas.addEventListener('pointermove', (e) => {
      const p = local(e);
      if (pointers.has(e.pointerId)) pointers.set(e.pointerId, p);
      if (pinch && pointers.size === 2 && this.group) {
        const [u, v] = [...pointers.values()];
        const factor = Math.max(10, Math.abs(u.x - v.x)) / pinch.dist;
        const [a, b] = pinch.range;
        const center = linear([this._scales.L.x0, this._scales.L.x1], [a, b])(pinch.mid);
        const span = (b - a) / factor;
        const f = (center - a) / (b - a);
        this.group.setRange([center - f * span, center - f * span + span]);
        return;
      }
      if (drag && pointers.size === 1) {
        const dx = p.x - drag.x;
        if (Math.abs(dx) > 5) drag.moved = true;
        if (drag.moved && this.group) {
          const [a, b] = drag.range;
          const dt = -dx * (b - a) / plotWidth();
          this.group.setRange([a + dt, b + dt]);
        }
        return;
      }
      if (e.pointerType === 'mouse') {
        this.hoverX = p.x;
        this._showTip(p.x);
        this.present();
      }
    });
    const end = (e, cancelled) => {
      const p = local(e);
      const wasDrag = drag;
      pointers.delete(e.pointerId);
      if (pointers.size < 2) pinch = null;
      if (!cancelled && wasDrag && !wasDrag.moved && this._scales && pointers.size === 0) {
        const t = this._scales.x.invert(clamp(p.x, this._scales.L.x0, this._scales.L.x1));
        if (this.group) this.group.seek(t);
        if (e.pointerType !== 'mouse') {
          this.hoverX = p.x;
          this._showTip(p.x);
          this.present();
          clearTimeout(tipTimer);
          tipTimer = setTimeout(() => { this.hoverX = null; this.tip.hidden = true; this.present(); }, 2500);
        }
      }
      if (pointers.size === 0) drag = null;
    };
    canvas.addEventListener('pointerup', (e) => end(e, false));
    canvas.addEventListener('pointercancel', (e) => end(e, true));
    canvas.addEventListener('pointerleave', (e) => {
      if (e.pointerType === 'mouse') { this.hoverX = null; this.tip.hidden = true; this.present(); }
    });
    canvas.addEventListener('wheel', (e) => {
      if (!this.group || !this._scales) return;
      const { L, x } = this._scales;
      const p = local(e);
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        this.group.zoom(Math.exp(-e.deltaY * 0.0025), x.invert(clamp(p.x, L.x0, L.x1)));
      } else if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
        e.preventDefault();
        const [a, b] = this.group.range;
        const delta = e.shiftKey && !e.deltaX ? e.deltaY : e.deltaX;
        this.group.pan(delta * (b - a) / plotWidth());
      }
    }, { passive: false });
    canvas.addEventListener('dblclick', () => { if (this.group) this.group.reset(); });
    canvas.addEventListener('keydown', (e) => {
      if (!this.group) return;
      const [a, b] = this.group.range;
      if (e.key === 'ArrowLeft') { this.group.pan(-(b - a) * 0.1); e.preventDefault(); }
      else if (e.key === 'ArrowRight') { this.group.pan((b - a) * 0.1); e.preventDefault(); }
      else if (e.key === '+' || e.key === '=') { this.group.zoom(1.6, this.group.cursor); e.preventDefault(); }
      else if (e.key === '-' || e.key === '_') { this.group.zoom(1 / 1.6, this.group.cursor); e.preventDefault(); }
      else if (e.key === '0' || e.key === 'Home') { this.group.reset(); e.preventDefault(); }
    });
  }
}

/** Per-bucket min / max / mean of (t, v) on `buckets` equal intervals of [t0, t1] (NaN when empty). */
export function bucketize(t, v, t0, t1, buckets) {
  const centre = new Float64Array(buckets);
  const lo = new Float64Array(buckets).fill(NaN);
  const hi = new Float64Array(buckets).fill(NaN);
  const sum = new Float64Array(buckets);
  const count = new Uint32Array(buckets);
  const width = (t1 - t0) / buckets || 1;
  for (let k = 0; k < buckets; k++) centre[k] = t0 + (k + 0.5) * width;
  for (let i = 0; i < t.length; i++) {
    const value = v[i];
    if (!Number.isFinite(value)) continue;
    const k = Math.floor((t[i] - t0) / width);
    if (k < 0 || k >= buckets) continue;
    if (!(lo[k] <= value)) lo[k] = value;
    if (!(hi[k] >= value)) hi[k] = value;
    sum[k] += value;
    count[k]++;
  }
  const mean = new Float64Array(buckets);
  for (let k = 0; k < buckets; k++) mean[k] = count[k] ? sum[k] / count[k] : NaN;
  return { t: centre, lo, hi, mean, count, width };
}

/** Median spacing of a sorted time array (0 when fewer than two samples). */
export function medianStep(t) {
  if (t.length < 2) return 0;
  const steps = [];
  const stride = Math.max(1, Math.floor(t.length / 2000));
  for (let i = stride; i < t.length; i += stride) steps.push((t[i] - t[i - stride]) / stride);
  steps.sort((p, q) => p - q);
  return steps[steps.length >> 1];
}
