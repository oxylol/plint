/* Canvas charts besides the time plots: XY plots (stability lobes, scatter), histograms, heat
 * maps of 2-D fields with their colour bar, and colour bars for the 3D layers.
 *
 * Every chart redraws on resize and on a light/dark switch, shows the values under the pointer
 * (or finger) in a tooltip, and leaves its numbers to a table next to it (panels/dom.js figure()).
 */

import { fitCanvas, observeWidth, setFont, clamp, fitText } from './canvas.js';
import { niceTicks, niceStep, tickLabel, extent, paddedDomain, linear } from './scale.js';
import { chrome, withAlpha, onSchemeChange, colormap, lut, stops } from './colormap.js';
import { num } from './units.js';

class CanvasChart {
  constructor(parent, { height = 220, label = 'chart' } = {}) {
    this.height = height;
    this.el = document.createElement('div');
    this.el.className = 'chart';
    this.canvas = document.createElement('canvas');
    this.canvas.setAttribute('role', 'img');
    this.canvas.setAttribute('aria-label', label);
    this.canvas.tabIndex = 0;
    this.tip = document.createElement('div');
    this.tip.className = 'tplot-tip';
    this.tip.hidden = true;
    this.el.append(this.canvas, this.tip);
    parent.append(this.el);
    this.width = 0;
    this._unobserve = observeWidth(this.el, (w) => { this.width = w; this.render(); });
    this._unscheme = onSchemeChange(() => this.render());
    const move = (e) => {
      const rect = this.canvas.getBoundingClientRect();
      this.hover(e.clientX - rect.left, e.clientY - rect.top, e.pointerType);
    };
    this.canvas.addEventListener('pointermove', (e) => { if (e.pointerType === 'mouse') move(e); });
    this.canvas.addEventListener('pointerdown', (e) => { if (e.pointerType !== 'mouse') move(e); });
    this.canvas.addEventListener('pointerleave', () => { this.tip.hidden = true; this._hoverIndex = null; this.render(); });
  }

  render() {
    const width = this.width || this.el.clientWidth || 300;
    const { ctx } = fitCanvas(this.canvas, width, this.height);
    this.ctx = ctx;
    this.c = chrome();
    ctx.clearRect(0, 0, width, this.height);
    this.draw(ctx, width, this.height);
    this.el.dataset.rendered = '1';
  }

  draw() {}
  hover() {}

  showTip(x, y, lines) {
    this.tip.replaceChildren();
    for (const [color, text, label] of lines) {
      const row = document.createElement('div');
      if (color) { const key = document.createElement('i'); key.style.background = color; row.append(key); }
      const strong = document.createElement('b');
      strong.textContent = text;
      row.append(strong);
      if (label) { const span = document.createElement('span'); span.textContent = label; row.append(span); }
      this.tip.append(row);
    }
    this.tip.hidden = false;
    const w = this.tip.offsetWidth || 140;
    this.tip.style.left = `${Math.max(0, x + 14 + w > this.width ? x - 14 - w : x + 14)}px`;
    this.tip.style.top = `${clamp(y - 10, 0, this.height - 30)}px`;
  }

  destroy() { this._unobserve(); this._unscheme(); this.el.remove(); }
}

function axes(ctx, c, L, x, y, { xLabel, yLabel, xTicks, yTicks }) {
  setFont(ctx, 11, 500, c.font);
  ctx.lineWidth = 1;
  const yStep = yTicks.length > 1 ? yTicks[1] - yTicks[0] : 1;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const v of yTicks) {
    const py = Math.round(y(v)) + 0.5;
    ctx.strokeStyle = c.grid;
    ctx.beginPath(); ctx.moveTo(L.x0, py); ctx.lineTo(L.x1, py); ctx.stroke();
    ctx.fillStyle = c.muted;
    ctx.fillText(fitText(ctx, tickLabel(v, yStep), L.x0 - 8), L.x0 - 6, py);
  }
  const xStep = xTicks.length > 1 ? xTicks[1] - xTicks[0] : 1;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (const v of xTicks) {
    const px = Math.round(x(v)) + 0.5;
    ctx.strokeStyle = c.grid;
    ctx.beginPath(); ctx.moveTo(px, L.y0); ctx.lineTo(px, L.y1); ctx.stroke();
    ctx.fillStyle = c.muted;
    const label = tickLabel(v, xStep);
    const half = ctx.measureText(label).width / 2;
    ctx.fillText(label, clamp(px, L.x0 + half - 8, L.x1 + 10 - half), L.y1 + 5);
  }
  ctx.strokeStyle = c.axis;
  ctx.beginPath(); ctx.moveTo(L.x0, L.y1 + 0.5); ctx.lineTo(L.x1, L.y1 + 0.5); ctx.stroke();
  ctx.fillStyle = c.ink2;
  if (xLabel) { ctx.textAlign = 'right'; ctx.fillText(xLabel, L.x1, L.y1 + 20); }
  if (yLabel) {
    ctx.save();
    ctx.translate(12, (L.y0 + L.y1) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
  }
}

/** Lines (optionally filled to the x axis) and points on linear x and y axes. */
export class XYPlot extends CanvasChart {
  constructor(parent, opts = {}) {
    super(parent, { height: opts.height || 240, label: opts.label || 'xy plot' });
    this.opts = opts;
    this.data = { lines: [], points: [] };
  }

  setData({ lines = [], points = [], xDomain = null, yDomain = null, includeZeroY = false } = {}) {
    this.data = { lines, points, xDomain, yDomain, includeZeroY };
    this.render();
  }

  draw(ctx, width, height) {
    const { lines, points } = this.data;
    const c = this.c;
    const L = { x0: 64, x1: width - 14, y0: 12, y1: height - 40 };
    const xs = [...lines.map((l) => l.x), points.map((p) => p.x)];
    const ys = [...lines.map((l) => l.y), points.map((p) => p.y)];
    let xDomain = this.data.xDomain || paddedDomain(extent(...xs));
    let yDomain = this.data.yDomain || paddedDomain(extent(...ys), { includeZero: this.data.includeZeroY });
    if (this.opts.equalAspect) {
      // one data unit is the same length on both axes (plans and elevations of the tool path)
      const perX = (xDomain[1] - xDomain[0]) / (L.x1 - L.x0);
      const perY = (yDomain[1] - yDomain[0]) / (L.y1 - L.y0);
      const per = Math.max(perX, perY);
      const cx = (xDomain[0] + xDomain[1]) / 2;
      const cy = (yDomain[0] + yDomain[1]) / 2;
      xDomain = [cx - per * (L.x1 - L.x0) / 2, cx + per * (L.x1 - L.x0) / 2];
      yDomain = [cy - per * (L.y1 - L.y0) / 2, cy + per * (L.y1 - L.y0) / 2];
    }
    const x = linear(xDomain, [L.x0, L.x1]);
    const y = linear(yDomain, [L.y1, L.y0]);
    this.scales = { x, y, L };
    axes(ctx, c, L, x, y, {
      xLabel: this.opts.xLabel, yLabel: this.opts.yLabel,
      xTicks: niceTicks(xDomain[0], xDomain[1], Math.max(2, Math.floor((L.x1 - L.x0) / 80))),
      yTicks: niceTicks(yDomain[0], yDomain[1], Math.max(2, Math.floor((L.y1 - L.y0) / 36))),
    });
    ctx.save();
    ctx.beginPath(); ctx.rect(L.x0, L.y0, L.x1 - L.x0, L.y1 - L.y0); ctx.clip();
    for (const line of lines) {
      if (line.fill) {
        ctx.fillStyle = withAlpha(line.color, 0.12);
        ctx.beginPath();
        ctx.moveTo(x(line.x[0]), y(Math.max(yDomain[0], 0)));
        for (let i = 0; i < line.x.length; i++) if (Number.isFinite(line.y[i])) ctx.lineTo(x(line.x[i]), y(line.y[i]));
        ctx.lineTo(x(line.x[line.x.length - 1]), y(Math.max(yDomain[0], 0)));
        ctx.closePath();
        ctx.fill();
      }
      ctx.strokeStyle = line.color;
      ctx.lineWidth = line.width || 2;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      let pen = false;
      for (let i = 0; i < line.x.length; i++) {
        if (!Number.isFinite(line.y[i]) || !Number.isFinite(line.x[i])) { pen = false; continue; }
        if (pen) ctx.lineTo(x(line.x[i]), y(line.y[i])); else ctx.moveTo(x(line.x[i]), y(line.y[i]));
        pen = true;
      }
      ctx.stroke();
    }
    points.forEach((p, i) => {
      if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) return;
      const px = x(p.x);
      const py = y(p.y);
      const r = this._hoverIndex === i ? 6 : (p.r || 5);
      ctx.beginPath();
      ctx.arc(px, py, r + 2, 0, 2 * Math.PI);
      ctx.fillStyle = c.surface;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(px, py, r, 0, 2 * Math.PI);
      if (p.hollow) {
        ctx.strokeStyle = p.color; ctx.lineWidth = 2.5; ctx.stroke();
      } else {
        ctx.fillStyle = p.color; ctx.fill();
      }
      if (p.text) {
        setFont(ctx, 11, 600, c.font);
        ctx.fillStyle = c.ink;
        ctx.textAlign = px > (L.x0 + L.x1) / 2 ? 'right' : 'left';
        ctx.textBaseline = 'bottom';
        ctx.fillText(p.text, px + (ctx.textAlign === 'left' ? 8 : -8), py - 6);
      }
    });
    ctx.restore();
  }

  hover(px, py) {
    if (!this.scales) return;
    const { x, y } = this.scales;
    let best = null;
    let bestD = 24 * 24;
    this.data.points.forEach((p, i) => {
      const d = (x(p.x) - px) ** 2 + (y(p.y) - py) ** 2;
      if (d < bestD) { bestD = d; best = i; }
    });
    if (best !== null) {
      const p = this.data.points[best];
      this._hoverIndex = best;
      this.render();
      this.showTip(px, py, (p.tip || [[p.color, `${num(p.x)}, ${num(p.y)}`, p.label || '']]));
      return;
    }
    this._hoverIndex = null;
    const lines = [];
    for (const line of this.data.lines) {
      let j = -1; let dx = Infinity;
      for (let i = 0; i < line.x.length; i++) {
        const d = Math.abs(x(line.x[i]) - px);
        if (d < dx) { dx = d; j = i; }
      }
      if (j >= 0 && dx < 30) lines.push([line.color, `${num(line.y[j])} at ${num(line.x[j])}`, line.label || '']);
    }
    this.render();
    if (lines.length) this.showTip(px, py, lines); else this.tip.hidden = true;
  }
}

/** Counts of values in equal bins, with labelled vertical marks (limits, nominal, this run). */
export class Histogram extends CanvasChart {
  constructor(parent, opts = {}) {
    super(parent, { height: opts.height || 190, label: opts.label || 'histogram' });
    this.opts = opts;
    this.data = { edges: [], counts: [], marks: [] };
  }

  setData({ edges, counts, marks = [], color = null } = {}) {
    this.data = { edges, counts, marks, color };
    this.render();
  }

  draw(ctx, width, height) {
    const { edges, counts, marks } = this.data;
    const c = this.c;
    const L = { x0: 50, x1: width - 14, y0: 26, y1: height - 40 };
    if (!edges.length) {
      setFont(ctx, 12, 500, c.font);
      ctx.fillStyle = c.muted; ctx.textAlign = 'center';
      ctx.fillText('no values', width / 2, height / 2);
      return;
    }
    const markValues = marks.map((m) => m.value).filter(Number.isFinite);
    const xr = extent(edges, markValues);
    const xDomain = [xr[0] - (xr[1] - xr[0]) * 0.03, xr[1] + (xr[1] - xr[0]) * 0.03];
    const top = Math.max(1, ...counts);
    const x = linear(xDomain, [L.x0, L.x1]);
    const y = linear([0, top * 1.08], [L.y1, L.y0]);
    this.scales = { x, y, L };
    axes(ctx, c, L, x, y, {
      xLabel: this.opts.xLabel, yLabel: 'count',
      xTicks: niceTicks(xDomain[0], xDomain[1], Math.max(2, Math.floor((L.x1 - L.x0) / 80))),
      yTicks: niceTicks(0, top * 1.08, 4).filter((v) => Number.isInteger(v)),
    });
    const fill = this.data.color || c.series[0];
    for (let i = 0; i < counts.length; i++) {
      if (!counts[i]) continue;
      const left = x(edges[i]) + 1;
      const right = x(edges[i + 1]) - 1;
      ctx.fillStyle = this._hoverIndex === i ? withAlpha(fill, 0.75) : fill;
      const w = Math.max(1, right - left);
      const h = L.y1 - y(counts[i]);
      ctx.beginPath();
      const r = Math.min(4, w / 2, h);
      ctx.moveTo(left, L.y1);
      ctx.lineTo(left, L.y1 - h + r); ctx.arcTo(left, L.y1 - h, left + r, L.y1 - h, r);
      ctx.lineTo(left + w - r, L.y1 - h); ctx.arcTo(left + w, L.y1 - h, left + w, L.y1 - h + r, r);
      ctx.lineTo(left + w, L.y1);
      ctx.closePath();
      ctx.fill();
    }
    setFont(ctx, 11, 600, c.font);
    marks.forEach((m, k) => {
      if (!Number.isFinite(m.value)) return;
      const px = Math.round(x(m.value)) + 0.5;
      ctx.strokeStyle = m.color || c.ink2;
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(px, L.y0 - 4); ctx.lineTo(px, L.y1); ctx.stroke();
      ctx.fillStyle = c.ink;
      ctx.textAlign = px > (L.x0 + L.x1) / 2 ? 'right' : 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillText(m.label, px + (ctx.textAlign === 'left' ? 4 : -4), L.y0 - 4 - (k % 2) * 12);
    });
  }

  hover(px, py) {
    if (!this.scales) return;
    const { x } = this.scales;
    const { edges, counts } = this.data;
    const value = x.invert(px);
    const i = edges.findIndex((e, k) => k < counts.length && value >= e && value < edges[k + 1]);
    this._hoverIndex = i >= 0 ? i : null;
    this.render();
    if (i >= 0) this.showTip(px, py, [[null, `${counts[i]}`, `from ${num(edges[i])} to ${num(edges[i + 1])}${this.opts.unit ? ` ${this.opts.unit}` : ''}`]]);
    else this.tip.hidden = true;
  }
}

/** A 2-D field: values[iy * nx + ix] (cell-centred) over x0 + ix dx, y0 + iy dy (display units),
 * NaN cells left as the surface. map: 'sequential' | 'sequential-orange' | 'diverging' | 'binary'. */
export class Heatmap extends CanvasChart {
  constructor(parent, opts = {}) {
    super(parent, { height: opts.height || 260, label: opts.label || 'map' });
    this.opts = opts;
    this.data = null;
  }

  setData(data) {
    this.data = data;
    this.render();
  }

  domain() {
    const d = this.data;
    if (d.domain) return d.domain;
    const range = extent(d.values) || [0, 1];
    if (d.map === 'diverging') {
      const bound = Math.max(Math.abs(range[0]), Math.abs(range[1]), 1e-30);
      return [-bound, bound];
    }
    return range[1] > range[0] ? range : [range[0], range[0] + (Math.abs(range[0]) * 1e-6 || 1e-12)];
  }

  draw(ctx, width, height) {
    const d = this.data;
    const c = this.c;
    if (!d || !d.nx || !d.ny) {
      setFont(ctx, 12, 500, c.font); ctx.fillStyle = c.muted; ctx.textAlign = 'center';
      ctx.fillText('no field', width / 2, height / 2);
      return;
    }
    const barH = 44;
    const L = { x0: 54, x1: width - 12, y0: 8, y1: height - 30 - barH };
    const spanX = d.nx * d.dx;
    const spanY = d.ny * d.dy;
    const scale = Math.min((L.x1 - L.x0) / spanX, (L.y1 - L.y0) / spanY);
    const w = spanX * scale;
    const h = spanY * scale;
    const ox = L.x0 + ((L.x1 - L.x0) - w) / 2;
    const oy = L.y0 + ((L.y1 - L.y0) - h) / 2;
    const x = linear([d.x0, d.x0 + spanX], [ox, ox + w]);
    const y = linear([d.y0, d.y0 + spanY], [oy + h, oy]);
    this.scales = { x, y, L: { x0: ox, x1: ox + w, y0: oy, y1: oy + h } };
    const [lo, hi] = this.domain();
    const binary = d.map === 'binary';
    const table = binary ? null : lut(d.map || 'sequential');
    const image = document.createElement('canvas');
    image.width = d.nx;
    image.height = d.ny;
    const ictx = image.getContext('2d');
    const pixels = ictx.createImageData(d.nx, d.ny);
    const on = binary ? colourOf(c.series[0]) : null;
    const off = binary ? colourOf(c.ink2) : null;
    for (let iy = 0; iy < d.ny; iy++) {
      for (let ix = 0; ix < d.nx; ix++) {
        const v = d.values[iy * d.nx + ix];
        const p = ((d.ny - 1 - iy) * d.nx + ix) * 4;
        if (!Number.isFinite(v)) { pixels.data[p + 3] = 0; continue; }
        if (binary) {
          const rgb = v ? on : off;
          pixels.data.set(rgb, p);
          pixels.data[p + 3] = v ? 255 : 70;
          continue;
        }
        const k = Math.round(clamp((v - lo) / (hi - lo || 1), 0, 1) * 255) * 3;
        pixels.data[p] = table[k]; pixels.data[p + 1] = table[k + 1]; pixels.data[p + 2] = table[k + 2]; pixels.data[p + 3] = 255;
      }
    }
    ictx.putImageData(pixels, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(image, ox, oy, w, h);
    ctx.strokeStyle = c.axis;
    ctx.lineWidth = 1;
    ctx.strokeRect(ox + 0.5, oy + 0.5, w - 1, h - 1);
    setFont(ctx, 11, 500, c.font);
    ctx.fillStyle = c.muted;
    const xt = niceTicks(d.x0, d.x0 + spanX, Math.max(2, Math.floor(w / 70)));
    const yt = niceTicks(d.y0, d.y0 + spanY, Math.max(2, Math.floor(h / 40)));
    const xs = xt.length > 1 ? xt[1] - xt[0] : niceStep(spanX);
    const ys = yt.length > 1 ? yt[1] - yt[0] : niceStep(spanY);
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (const v of xt) ctx.fillText(tickLabel(v, xs), x(v), oy + h + 4);
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (const v of yt) ctx.fillText(tickLabel(v, ys), ox - 5, y(v));
    ctx.fillStyle = c.ink2;
    ctx.textAlign = 'right'; ctx.textBaseline = 'top';
    ctx.fillText(this.opts.xLabel || 'x (mm)', ox + w, oy + h + 17);
    ctx.save(); ctx.translate(Math.max(12, ox - 40), oy + h / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center';
    ctx.fillText(this.opts.yLabel || 'y (mm)', 0, 0); ctx.restore();
    const top = height - barH + 4;
    if (binary) {
      drawCategories(ctx, c, 54, top, width - 66, d.categories || [[c.series[0], 'yes'], [withAlpha(c.ink2, 0.3), 'no']]);
    } else {
      drawColorBar(ctx, c, 54, top, width - 66, d.map || 'sequential', [lo, hi], d.unit || '');
    }
  }

  hover(px, py) {
    if (!this.scales || !this.data) return;
    const { x, y, L } = this.scales;
    if (px < L.x0 || px > L.x1 || py < L.y0 || py > L.y1) { this.tip.hidden = true; return; }
    const d = this.data;
    const ix = Math.floor((x.invert(px) - d.x0) / d.dx);
    const iy = Math.floor((y.invert(py) - d.y0) / d.dy);
    if (ix < 0 || iy < 0 || ix >= d.nx || iy >= d.ny) { this.tip.hidden = true; return; }
    const v = d.values[iy * d.nx + ix];
    const where = `at ${num(d.x0 + (ix + 0.5) * d.dx)}, ${num(d.y0 + (iy + 0.5) * d.dy)} mm`;
    const text = !Number.isFinite(v) ? 'no material' : d.map === 'binary' ? (v ? 'yes' : 'no') : `${num(v)}${d.unit ? ` ${d.unit}` : ''}`;
    this.showTip(px, py, [[null, text, where]]);
  }
}

function colourOf(css) {
  if (css.startsWith('#')) {
    const h = css.slice(1);
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  }
  const m = /rgba?\(([^)]+)\)/.exec(css);
  return m ? m[1].split(',').slice(0, 3).map((v) => Number(v.trim())) : [128, 128, 128];
}

function drawColorBar(ctx, c, x0, top, width, map, [lo, hi], unit) {
  const fn = colormap(map);
  const steps = Math.max(2, Math.floor(width));
  for (let i = 0; i < steps; i++) {
    const rgb = fn(i / (steps - 1));
    ctx.fillStyle = `rgb(${rgb[0] | 0},${rgb[1] | 0},${rgb[2] | 0})`;
    ctx.fillRect(x0 + i * width / steps, top, width / steps + 0.6, 12);
  }
  ctx.strokeStyle = c.axis;
  ctx.strokeRect(x0 + 0.5, top + 0.5, width - 1, 11);
  setFont(ctx, 11, 500, c.font);
  ctx.fillStyle = c.ink2;
  ctx.textBaseline = 'top';
  const ticks = niceTicks(lo, hi, Math.max(2, Math.floor(width / 80)));
  const step = ticks.length > 1 ? ticks[1] - ticks[0] : niceStep(hi - lo);
  const x = linear([lo, hi], [x0, x0 + width]);
  ctx.textAlign = 'center';
  for (const v of ticks) {
    const px = x(v);
    ctx.fillRect(px - 0.5, top + 12, 1, 3);
    ctx.fillText(tickLabel(v, step), clamp(px, x0 + 12, x0 + width - 12), top + 16);
  }
  if (unit) { ctx.textAlign = 'right'; ctx.fillText(unit, x0 + width, top - 14); }
}

function drawCategories(ctx, c, x0, top, width, categories) {
  setFont(ctx, 12, 500, c.font);
  let x = x0;
  for (const [color, label] of categories) {
    ctx.fillStyle = color;
    ctx.fillRect(x, top, 14, 12);
    ctx.fillStyle = c.ink2;
    ctx.textBaseline = 'top';
    ctx.textAlign = 'left';
    ctx.fillText(label, x + 20, top);
    x += 20 + ctx.measureText(label).width + 18;
    if (x > x0 + width) break;
  }
}

/** A standalone colour bar (layers): {map, domain: [lo, hi] in display units, unit} or
 * {categories: [[color, label], ...]}. */
export class ColorBar extends CanvasChart {
  constructor(parent, opts = {}) {
    super(parent, { height: opts.height || 46, label: opts.label || 'colour scale' });
    this.data = null;
  }

  setData(data) { this.data = data; this.render(); }

  draw(ctx, width) {
    if (!this.data) return;
    const c = this.c;
    if (this.data.categories) drawCategories(ctx, c, 2, 14, width - 4, this.data.categories);
    else drawColorBar(ctx, c, 2, 16, width - 4, this.data.map, this.data.domain, this.data.unit || '');
    this.canvas.setAttribute('aria-label', this.data.categories
      ? `legend: ${this.data.categories.map((x) => x[1]).join(', ')}`
      : `colour scale from ${num(this.data.domain[0])} to ${num(this.data.domain[1])} ${this.data.unit || ''}`);
  }
}

export { stops };
