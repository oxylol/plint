/* Scales and ticks for the canvas plots: linear maps, "nice" tick steps (1, 2, 5 x 10^k, the
 * Heckbert rule), data extents that skip NaN, and tick label precision. */

import { num, MINUS } from './units.js';

/** The step of about `count` ticks over `span`: 1, 2 or 5 times a power of ten. */
export function niceStep(span, count = 5) {
  if (!(span > 0) || !Number.isFinite(span)) return 1;
  const raw = span / Math.max(1, count);
  const power = 10 ** Math.floor(Math.log10(raw));
  const fraction = raw / power;
  const nice = fraction < 1.5 ? 1 : fraction < 3 ? 2 : fraction < 7 ? 5 : 10;
  return nice * power;
}

/** Tick values inside [min, max] at a nice step. */
export function niceTicks(min, max, count = 5) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (max < min) [min, max] = [max, min];
  if (max === min) return [min];
  const step = niceStep(max - min, count);
  const first = Math.ceil(min / step - 1e-9) * step;
  const ticks = [];
  for (let value = first; value <= max + step * 1e-9; value += step) {
    ticks.push(Math.abs(value) < step * 1e-9 ? 0 : value);
    if (ticks.length > 200) break;
  }
  return ticks;
}

/** Decimals that show the difference between ticks `step` apart. */
export function stepDecimals(step) {
  if (!(step > 0)) return 0;
  return Math.max(0, Math.min(12, -Math.floor(Math.log10(step) + 1e-9)));
}

export function tickLabel(value, step) {
  if (Math.abs(value) >= 1e6 || (Math.abs(value) > 0 && Math.abs(value) < 1e-4)) return num(value, 3);
  const text = value.toFixed(stepDecimals(step));
  return text.startsWith('-') && /[1-9]/.test(text) ? MINUS + text.slice(1) : text.replace(/^-/, '');
}

/** [min, max] of finite values of any number of arrays (null when there are none). */
export function extent(...arrays) {
  let min = Infinity;
  let max = -Infinity;
  for (const array of arrays) {
    if (!array) continue;
    for (let i = 0; i < array.length; i++) {
      const v = array[i];
      if (v === null || !Number.isFinite(v)) continue;
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  return min <= max ? [min, max] : null;
}

/** A padded y-domain for plotting: 6 % margin, a flat series opened around its value. */
export function paddedDomain(range, { includeZero = false } = {}) {
  if (!range) return [0, 1];
  let [min, max] = range;
  if (includeZero) { min = Math.min(min, 0); max = Math.max(max, 0); }
  if (max === min) {
    const pad = Math.abs(min) > 0 ? Math.abs(min) * 0.1 : 1;
    return [min - pad, max + pad];
  }
  const pad = (max - min) * 0.06;
  return [min - pad, max + pad];
}

/** Linear map domain -> range, with invert. */
export function linear(domain, range) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
  const map = (v) => r0 + (v - d0) * k;
  map.invert = (r) => (k === 0 ? d0 : d0 + (r - r0) / k);
  map.domain = domain;
  map.range = range;
  return map;
}

/** Index of the last element <= value in a sorted array (-1 when all are larger). */
export function bisectRight(sorted, value) {
  let lo = 0;
  let hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (sorted[mid] <= value) lo = mid + 1; else hi = mid;
  }
  return lo - 1;
}

/** Index of the sample nearest to value in a sorted array. */
export function nearestIndex(sorted, value) {
  if (!sorted.length) return -1;
  const i = bisectRight(sorted, value);
  if (i < 0) return 0;
  if (i >= sorted.length - 1) return sorted.length - 1;
  return value - sorted[i] <= sorted[i + 1] - value ? i : i + 1;
}

/** Equal-width histogram: {edges, counts}. `bins` default: Freedman-Diaconis, clamped to 8..60. */
export function histogram(values, { bins = null, range = null } = {}) {
  const finite = [];
  for (const v of values) if (v !== null && Number.isFinite(v)) finite.push(v);
  if (!finite.length) return { edges: [], counts: [] };
  finite.sort((a, b) => a - b);
  let [min, max] = range || [finite[0], finite[finite.length - 1]];
  if (max === min) { const pad = Math.abs(min) * 0.01 || 1e-12; min -= pad; max += pad; }
  let n = bins;
  if (!n) {
    const q = (p) => finite[Math.min(finite.length - 1, Math.floor(p * (finite.length - 1)))];
    const iqr = q(0.75) - q(0.25);
    const width = iqr > 0 ? 2 * iqr / Math.cbrt(finite.length) : (max - min) / 10;
    n = Math.max(8, Math.min(60, Math.round((max - min) / width) || 10));
  }
  const edges = Array.from({ length: n + 1 }, (_, i) => min + (max - min) * i / n);
  const counts = new Array(n).fill(0);
  for (const v of finite) {
    if (v < min || v > max) continue;
    counts[Math.min(n - 1, Math.floor((v - min) / (max - min) * n))]++;
  }
  return { edges, counts };
}
