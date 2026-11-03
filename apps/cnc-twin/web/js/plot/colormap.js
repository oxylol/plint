/* Colour for the plots, maps and the 3D layers' colour bars.
 *
 * Categorical: the eight hues of the report (cnctwin/report/plots.py PALETTE) in their fixed,
 * colour-vision-validated order, stepped for the dark surface in dark mode (validated against the
 * Plint surfaces #e4e9e5 / #1e2421: adjacent CVD Delta E >= 8.4, normal-vision >= 19.3; four light
 * slots are below 3:1 contrast, so every chart has a table view and a legend). Series beyond eight
 * fold into one band. Sequential: one hue, light -> dark (blue; temperature takes the second slot's
 * orange as its own one-hue ramp at the blue ramp's lightness steps), anchored at the surface in
 * dark mode. Diverging: blue <-> red with a neutral grey midpoint. Severity colours are Plint's
 * status tokens and always appear next to a text label.
 *
 * The 3D view should colour its layers with `layerColormap(layer, mode)` so the colour bars of the
 * shell match it (docs/ui.md "Layers").
 */

export const CATEGORICAL = {
  light: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'],
  dark: ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'],
};
const BLUE = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf',
  '#1c5cab', '#184f95', '#104281', '#0d366b'];
const MIDPOINT = { light: '#f0efec', dark: '#383835' };

export function mode() {
  return typeof matchMedia === 'function' && matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export function categorical(index, m = mode()) {
  const list = CATEGORICAL[m];
  return list[((index % list.length) + list.length) % list.length];
}

// ------------------------------------------------------------------ OKLab (Ottosson 2020)

export function hexToRgb(hex) {
  const h = hex.replace('#', '');
  const full = h.length === 3 ? h.split('').map((c) => c + c).join('') : h;
  return [parseInt(full.slice(0, 2), 16), parseInt(full.slice(2, 4), 16), parseInt(full.slice(4, 6), 16)];
}
export function rgbToHex([r, g, b]) {
  return `#${[r, g, b].map((v) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, '0')).join('')}`;
}
const toLinear = (c) => { const v = c / 255; return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
const fromLinear = (c) => 255 * (c <= 0.0031308 ? 12.92 * c : 1.055 * Math.sign(c) * Math.abs(c) ** (1 / 2.4) - 0.055);

export function rgbToOklab([r, g, b]) {
  const lr = toLinear(r); const lg = toLinear(g); const lb = toLinear(b);
  const l = Math.cbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb);
  const m = Math.cbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb);
  const s = Math.cbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb);
  return [0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s];
}
function oklabToLinear([L, a, b]) {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3;
  return [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s];
}
export function oklabToRgb(lab) {
  return oklabToLinear(lab).map(fromLinear).map((v) => Math.max(0, Math.min(255, v)));
}

/** The colour at OKLCH (L, C, h), with the chroma reduced until it is inside sRGB. */
function oklch(L, C, h) {
  let lo = 0;
  let hi = C;
  const inside = (c) => oklabToLinear([L, c * Math.cos(h), c * Math.sin(h)]).every((v) => v >= -1e-4 && v <= 1 + 1e-4);
  if (inside(C)) return rgbToHex(oklabToRgb([L, C * Math.cos(h), C * Math.sin(h)]));
  for (let i = 0; i < 24; i++) { const mid = (lo + hi) / 2; if (inside(mid)) lo = mid; else hi = mid; }
  return rgbToHex(oklabToRgb([L, lo * Math.cos(h), lo * Math.sin(h)]));
}

/** A one-hue ramp at the blue ramp's lightness and chroma steps, in the hue of `hex`. */
function rampLike(hex) {
  const [, a, b] = rgbToOklab(hexToRgb(hex));
  const hue = Math.atan2(b, a);
  return BLUE.map((step) => {
    const [L, sa, sb] = rgbToOklab(hexToRgb(step));
    return oklch(L, Math.hypot(sa, sb), hue);
  });
}

const ORANGE = rampLike(CATEGORICAL.light[1]);
const RED = rampLike(CATEGORICAL.light[7]);

/** Colour stops (hex, low -> high) of a named map for a page mode. */
export function stops(name, m = mode()) {
  if (name === 'diverging') {
    return [BLUE[9], BLUE[6], BLUE[2], MIDPOINT[m], RED[2], RED[6], RED[9]];
  }
  const ramp = name === 'sequential-orange' ? ORANGE : BLUE;
  const chosen = ramp.slice(0, 12);
  return m === 'dark' ? chosen.slice().reverse() : chosen;
}

/** A function t in [0, 1] -> [r, g, b] interpolating the stops in OKLab. */
export function colormap(name, m = mode()) {
  const labs = stops(name, m).map((hex) => rgbToOklab(hexToRgb(hex)));
  const n = labs.length - 1;
  return (t) => {
    if (!Number.isFinite(t)) return null;
    const x = Math.max(0, Math.min(1, t)) * n;
    const i = Math.min(n - 1, Math.floor(x));
    const f = x - i;
    const p = labs[i];
    const q = labs[i + 1];
    return oklabToRgb([p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f, p[2] + (q[2] - p[2]) * f]);
  };
}

/** A 256-entry RGB lookup table (Uint8ClampedArray, 3 per entry). */
export function lut(name, m = mode(), size = 256) {
  const map = colormap(name, m);
  const table = new Uint8ClampedArray(size * 3);
  for (let i = 0; i < size; i++) table.set(map(i / (size - 1)), i * 3);
  return table;
}

/** The colour map of each 3D layer (what the colour bars show). */
export const LAYER_MAPS = {
  temperature: 'sequential-orange',
  chips: 'sequential',
  finish: 'sequential',
  deviation: 'diverging',
  deflection: 'sequential',
  wear: 'sequential',
  coolant: 'categorical',
};
export function layerColormap(layer, m = mode()) {
  const name = LAYER_MAPS[layer] || 'sequential';
  return name === 'categorical' ? null : colormap(name, m);
}

// ------------------------------------------------------------------ chart chrome from CSS

const FALLBACK = {
  light: { surface: '#e4e9e5', ink: '#22302a', ink2: '#56655d', muted: '#6f7d75', grid: 'rgba(34,48,42,0.10)',
    axis: 'rgba(34,48,42,0.28)', cursor: '#3f6a53', info: '#4a6585', warning: '#8a5010', error: '#a63d32', fatal: '#a63d32' },
  dark: { surface: '#1e2421', ink: '#dde6e0', ink2: '#9aa8a0', muted: '#87958d', grid: 'rgba(221,230,224,0.09)',
    axis: 'rgba(221,230,224,0.26)', cursor: '#8ec3a4', info: '#93b0d6', warning: '#e0a458', error: '#e5877b', fatal: '#e5877b' },
};

/** Colours of the chart furniture, read from the page's CSS custom properties (css/tokens.css). */
export function chrome() {
  const m = mode();
  const out = { ...FALLBACK[m], mode: m, series: CATEGORICAL[m] };
  if (typeof document === 'undefined' || typeof getComputedStyle !== 'function') return out;
  const style = getComputedStyle(document.documentElement);
  const read = (name, key) => { const v = style.getPropertyValue(name).trim(); if (v) out[key] = v; };
  read('--chart-surface', 'surface'); read('--ink', 'ink'); read('--ink2', 'ink2'); read('--chart-muted', 'muted');
  read('--chart-grid', 'grid'); read('--chart-axis', 'axis'); read('--accent', 'cursor'); read('--info', 'info');
  read('--warn', 'warning'); read('--bad', 'error'); read('--bad', 'fatal');
  out.font = style.getPropertyValue('--font').trim() || 'system-ui, sans-serif';
  out.mono = style.getPropertyValue('--mono').trim() || 'ui-monospace, monospace';
  return out;
}

export function severityColor(severity, c = chrome()) {
  return c[severity] || c.info;
}

/** Re-render on a light/dark switch: callback() after the scheme changes. Returns an unsubscribe. */
export function onSchemeChange(callback) {
  if (typeof matchMedia !== 'function') return () => {};
  const query = matchMedia('(prefers-color-scheme: dark)');
  const handler = () => callback(mode());
  query.addEventListener('change', handler);
  return () => query.removeEventListener('change', handler);
}

export function rgba([r, g, b], alpha = 1) {
  return `rgba(${r | 0},${g | 0},${b | 0},${alpha})`;
}
export function withAlpha(color, alpha) {
  if (color.startsWith('#')) return rgba(hexToRgb(color), alpha);
  const m = /rgba?\(([^)]+)\)/.exec(color);
  if (!m) return color;
  const [r, g, b] = m[1].split(',').map((s) => Number(s.trim()));
  return rgba([r, g, b], alpha);
}
