/* Keyframe fields (keyframes/kf_<k>.npz, docs/formats.md "Keyframes") as grids the view can sample:
 * 3-D grids (thermal, displacement) and 2-D grids over the part (surface temperature, chip pile
 * heights, coolant wetted cells), their textures for the part shader, CPU sampling for picking, which
 * layers a run records, and the layers' colour ranges.
 *
 * Grids are cell-centred: cell (i, j, k) has its lower corner at origin + (i, j, k) cell. A grid's
 * origin and cell size are `<name>_origin_S_m` / `<name>_cell_m`, else those of its name with the
 * last `_token`s dropped (work_temp_K -> work_temp -> work), else (2-D grids) of the pile grid, then
 * of the surface grid. Axis order: 3-D grids are values[nx, ny, nz] (cnctwin.core.keyframes.FieldGrid);
 * a 2-D grid is read [nx, ny] unless only the [ny, nx] reading covers the stock with less than one cell
 * of padding (the rule of the UI shell, web/js/panels/fields.js, so both read a grid the same way). A
 * 2-D grid lies in the plane of its origin with axes e_u, e_v (`<prefix>_axes_S` columns; default x_S,
 * y_S) and is projected along e_w onto the part.
 *
 * Interpolation (picking and the shader alike) is normalised convolution (H. Knutsson and C.-F.
 * Westin, "Normalized and differential convolution", CVPR 1993): tri/bilinear weights times a validity
 * mask, divided by the masked weight sum, so cells without material (NaN) never bleed into the part;
 * the error inside a field is that of linear interpolation, (cell^2 / 8) |f''| per axis. */

import * as THREE from './three.js';

/** Evenly spaced members of a list (at most `count`, ends kept, no repeats): the shell's sampling. */
export function sampleEvenly(list, count) {
  if (list.length <= count) return list.slice();
  const out = [];
  for (let i = 0; i < count; i++) out.push(list[Math.round(i * (list.length - 1) / (count - 1))]);
  return [...new Set(out)];
}

function arrayNumbers(arr) {
  if (!arr) return null;
  return Array.from(arr.data, Number);
}

function scalar(arr) {
  const v = arr && arr.data && arr.data.length ? Number(arr.data[0]) : NaN;
  return Number.isFinite(v) ? v : null;
}

/** Prefixes under which a grid's origin and cell size are looked for. */
export function gridPrefixes(name, dims) {
  const out = [name];
  const parts = name.split('_');
  for (let n = parts.length - 1; n >= 1; n--) out.push(parts.slice(0, n).join('_'));
  if (dims === 2) out.push('pile', 'surface');
  return [...new Set(out)];
}

/** A grid from npz arrays: {name, dims, n: [nx, ny, nz], origin, cell, axes (3x3 rows e_u, e_v, e_w),
 * values: Float32Array (x fastest), orientation}; vector grids ([.., 3] last) become their magnitude.
 * Null when the array is not a grid or has no origin and cell. */
export function gridFromArrays(arrays, name, stockSizeM = null) {
  const arr = arrays[name];
  if (!arr || Array.isArray(arr.data)) return null;
  let shape = arr.shape.slice();
  let data = arr.data;
  if (shape.length >= 3 && shape[shape.length - 1] === 3 && (shape.length === 4 || /displacement|deflection/.test(name))) {
    const count = data.length / 3;
    const magnitude = new Float64Array(count);
    for (let i = 0; i < count; i++) magnitude[i] = Math.hypot(data[i * 3], data[i * 3 + 1], data[i * 3 + 2]);
    data = magnitude;
    shape = shape.slice(0, -1);
  }
  if (shape.length !== 2 && shape.length !== 3) return null;
  const dims = shape.length;
  let origin = null;
  let cell = null;
  let axes = null;
  for (const prefix of gridPrefixes(name, dims)) {
    if (arrays[`${prefix}_origin_S_m`] && arrays[`${prefix}_cell_m`]) {
      origin = arrayNumbers(arrays[`${prefix}_origin_S_m`]);
      cell = scalar(arrays[`${prefix}_cell_m`]);
      const axesArr = arrays[`${prefix}_axes_S`];
      if (axesArr && axesArr.data.length === 9) {
        const m = Array.from(axesArr.data, Number);
        axes = [[m[0], m[3], m[6]], [m[1], m[4], m[7]], [m[2], m[5], m[8]]];
      }
      break;
    }
  }
  if (!origin || !(cell > 0)) return null;
  while (origin.length < 3) origin.push(dims === 2 ? NaN : 0);
  if (!axes) axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  let nx;
  let ny;
  let nz = 1;
  const values = new Float32Array(data.length);
  let orientation;
  if (dims === 3) {
    [nx, ny, nz] = shape;
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) for (let k = 0; k < nz; k++) {
      values[i + nx * (j + ny * k)] = data[(i * ny + j) * nz + k];
    }
    orientation = 'values[x, y, z]';
  } else {
    const [n0, n1] = shape;
    let xFirst = true;
    if (stockSizeM) {
      const covers = (n, size) => n * cell >= size * (1 - 1e-9) && n * cell < size + cell * (1 - 1e-9);
      const fits = (a, b) => covers(a, stockSizeM[0]) && covers(b, stockSizeM[1]);
      if (!fits(n0, n1) && fits(n1, n0)) xFirst = false;
    }
    nx = xFirst ? n0 : n1;
    ny = xFirst ? n1 : n0;
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
      values[i + nx * j] = xFirst ? data[i * n1 + j] : data[j * n1 + i];
    }
    orientation = xFirst ? 'values[x, y]' : 'values[y, x]';
  }
  return { name, dims, n: [nx, ny, nz], origin, cell, axes, values, orientation };
}

/** Grid coordinates (in cells) of an S point. */
function gridCoords(grid, x, y, z) {
  const dx = x - grid.origin[0];
  const dy = y - grid.origin[1];
  const dz = z - (Number.isFinite(grid.origin[2]) ? grid.origin[2] : 0);
  const a = grid.axes;
  return [
    (a[0][0] * dx + a[0][1] * dy + a[0][2] * dz) / grid.cell - 0.5,
    (a[1][0] * dx + a[1][1] * dy + a[1][2] * dz) / grid.cell - 0.5,
    (a[2][0] * dx + a[2][1] * dy + a[2][2] * dz) / grid.cell - 0.5,
  ];
}

/** Value of a grid at an S point by normalised convolution (null where no valid cell is near). */
export function sampleGrid(grid, x, y, z) {
  const [u, v, w] = gridCoords(grid, x, y, z);
  const [nx, ny, nz] = grid.n;
  const clamp = (i, n) => (i < 0 ? 0 : i >= n ? n - 1 : i);
  const i0 = Math.floor(u); const j0 = Math.floor(v); const k0 = grid.dims === 3 ? Math.floor(w) : 0;
  const fu = u - i0; const fv = v - j0; const fw = grid.dims === 3 ? w - k0 : 0;
  let sum = 0;
  let weight = 0;
  for (let dk = 0; dk < (grid.dims === 3 ? 2 : 1); dk++) {
    for (let dj = 0; dj < 2; dj++) {
      for (let di = 0; di < 2; di++) {
        const wgt = (di ? fu : 1 - fu) * (dj ? fv : 1 - fv) * (grid.dims === 3 ? (dk ? fw : 1 - fw) : 1);
        if (wgt <= 0) continue;
        const value = grid.values[clamp(i0 + di, nx) + nx * (clamp(j0 + dj, ny) + ny * clamp(k0 + dk, nz))];
        if (!Number.isFinite(value)) continue;
        sum += wgt * value;
        weight += wgt;
      }
    }
  }
  return weight > 1e-6 ? sum / weight : null;
}

/** RG half-float texture of a grid: R the value mapped to [0, 1] over the grid's own range (so half
 * precision keeps 11 bits of that range whatever the unit: micrometre deflections included), G the
 * validity. Returns {texture, min, max}: value = min + R (max - min). */
export function gridTexture(grid) {
  const count = grid.values.length;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < count; i++) {
    const v = grid.values[i];
    if (Number.isFinite(v)) { if (v < min) min = v; if (v > max) max = v; }
  }
  if (!(min <= max)) { min = 0; max = 1; }
  const span = max > min ? max - min : 1;
  const data = new Uint16Array(count * 2);
  const toHalf = THREE.DataUtils.toHalfFloat;
  const one = toHalf(1);
  for (let i = 0; i < count; i++) {
    const v = grid.values[i];
    if (Number.isFinite(v)) { data[i * 2] = toHalf(roundHalf((v - min) / span)); data[i * 2 + 1] = one; }
  }
  const [nx, ny, nz] = grid.n;
  const texture = grid.dims === 3 ? new THREE.Data3DTexture(data, nx, ny, nz) : new THREE.DataTexture(data, nx, ny);
  texture.format = THREE.RGFormat;
  texture.type = THREE.HalfFloatType;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  if (grid.dims === 3) texture.wrapR = THREE.ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.unpackAlignment = 1;
  texture.needsUpdate = true;
  return { texture, min, max: min + span };
}

const isJet = (name) => name.startsWith('jet_');
const isOriginEtc = (name) => /_(origin_S_m|cell_m|axes_S)$/.test(name);

/** The grids of a keyframe's arrays that a layer can use: {temperature, chips, coolant, deflection}. */
export function layerGrids(arrays, stockSizeM) {
  const out = { temperature: null, chips: null, coolant: null, deflection: null };
  const names = Object.keys(arrays).filter((n) => !isJet(n) && !isOriginEtc(n)).sort();
  const grids = new Map();
  for (const name of names) {
    const grid = gridFromArrays(arrays, name, stockSizeM);
    if (grid) grids.set(name, grid);
  }
  const temps = [...grids.values()].filter((g) => /temp_K$/.test(g.name));
  out.temperature = temps.find((g) => g.dims === 3) || temps.find((g) => g.dims === 2) || null;
  out.chips = grids.get('pile_height_m') || null;
  out.coolant = grids.get('wetted') || null;
  out.deflection = [...grids.values()].find((g) => /displacement|deflection/.test(g.name) && g.name !== 'tool_deflection_m') || null;
  return out;
}

/** The tool deflection of a keyframe (T0 components, metres) for a path, or null. */
export function toolDeflection(arrays, path = 0) {
  const arr = arrays.tool_deflection_m || arrays.tool_deflection_T0_m || arrays.tool_deflection_S_m;
  if (!arr || Array.isArray(arr.data) || !arr.data.length) return null;
  const d = arr.data;
  let row = 0;
  if (arr.shape.length === 2 && arr.shape[0] > path) row = path;
  const width = arr.shape.length === 2 ? arr.shape[1] : d.length;
  const v = [Number(d[row * width]) || 0, width > 1 ? Number(d[row * width + 1]) || 0 : 0, width > 2 ? Number(d[row * width + 2]) || 0 : 0];
  return { vector_m: v, frame: arrays.tool_deflection_S_m && !arrays.tool_deflection_m ? 'S' : 'T0' };
}

// ------------------------------------------------------------------ colour ranges (the shell's rules)

/** `value` rounded to the nearest IEEE 754 binary16 number (ties to even, as numpy's float16; three.js's
 * DataUtils.toHalfFloat truncates). Overflow gives +-Infinity. */
export function roundHalf(value) {
  if (!Number.isFinite(value) || value === 0) return value;
  const a = Math.abs(value);
  if (a >= 65520) return Math.sign(value) * Infinity;
  let e = Math.floor(Math.log2(a));
  if (2 ** e > a) e--;
  if (2 ** (e + 1) <= a) e++;
  const spacing = 2 ** (Math.max(e, -14) - 10);
  const q = a / spacing;
  const base = Math.floor(q);
  const frac = q - base;
  const r = frac > 0.5 || (frac === 0.5 && base % 2 === 1) ? base + 1 : base;
  return Math.sign(value) * r * spacing;
}

/** The shortest decimal that reads back to the same float16 / float32: what the API's JSON carries (numpy's
 * str(): Dragon4 shortest digits, the even last digit when two are equally near), so ranges computed here
 * and by the shell from /keyframes/{k}/fields JSON are the same numbers. */
export function shortestRepr(value, dtype) {
  if (!Number.isFinite(value) || value === 0 || (dtype !== 'float16' && dtype !== 'float32')) return value;
  const round = dtype === 'float32' ? Math.fround : roundHalf;
  const target = round(value);
  for (let p = 1; p <= 17; p++) {
    const up = Number(value.toPrecision(p));
    const magnitude = Math.floor(Math.log10(Math.abs(up || value)));
    const step = 10 ** (magnitude - p + 1);
    const candidates = [up, Number((up - step).toPrecision(p)), Number((up + step).toPrecision(p))]
      .filter((c) => Number.isFinite(c) && round(c) === target);
    if (!candidates.length) continue;
    candidates.sort((a, b) => {
      const da = Math.abs(a - value);
      const db = Math.abs(b - value);
      if (Math.abs(da - db) > Math.abs(value) * 1e-15) return da - db;
      const even = (c) => Math.round(Math.abs(c) / step) % 2 === 0;
      return (even(a) ? 0 : 1) - (even(b) ? 0 : 1);
    });
    return candidates[0];
  }
  return value;
}

/** [min, max] of finite values after `transform`, null without any. */
export function finiteRange(values, transform = (v) => v) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const raw of values) {
    if (raw === null || !Number.isFinite(raw)) continue;
    const v = transform(raw);
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return lo <= hi ? [lo, hi] : null;
}

/** Finite extremes of an array document's data (shortest-repr rounded), pushed into `into`. */
export function pushExtremes(arr, into, magnitude3 = false) {
  if (!arr || Array.isArray(arr.data)) return;
  const d = arr.data;
  let lo = Infinity;
  let hi = -Infinity;
  if (magnitude3 && d.length && d.length % 3 === 0) {
    for (let i = 0; i < d.length; i += 3) {
      const v = Math.hypot(shortestRepr(d[i], arr.dtype), shortestRepr(d[i + 1], arr.dtype), shortestRepr(d[i + 2], arr.dtype));
      if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
    }
  } else {
    for (let i = 0; i < d.length; i++) {
      const v = magnitude3 ? Math.abs(d[i]) : d[i];
      if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
    }
    if (lo <= hi) { lo = shortestRepr(lo, arr.dtype); hi = shortestRepr(hi, arr.dtype); }
  }
  if (lo <= hi) into.push(lo, hi);
}
