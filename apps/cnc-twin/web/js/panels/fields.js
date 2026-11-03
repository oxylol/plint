/* Keyframe fields and per-vertex attributes as the panels need them: a 2-D grid ready for a heat
 * map (rows = y, columns = x, axes in mm), values of an attribute over the final surface, and
 * evenly sampled keyframes.
 *
 * Grid axis order: cnctwin.core.keyframes.FieldGrid documents values[nx, ny(, nz)] (first index
 * x); the report and the synthetic run read 2-D fields as rows = y. A grid is taken as [nx, ny]
 * unless only the [ny, nx] reading covers the stock with less than one cell of padding; a 3-D grid
 * is reduced to its maximum through z (the report does the same). */

import { decodeArrayDoc } from '../api.js';

export function sampleEvenly(list, count) {
  if (list.length <= count) return list.slice();
  const out = [];
  for (let i = 0; i < count; i++) out.push(list[Math.round(i * (list.length - 1) / (count - 1))]);
  return [...new Set(out)];
}

function scalar(doc) {
  const arr = decodeArrayDoc(doc);
  return arr && arr.data.length ? Number(arr.data[0]) : null;
}

/** {values (Float64Array, [iy * nx + ix]), nx, ny, x0, y0, dx, dy (mm), orientation} or null. A grid over
 * the stock covers it with less than one cell of padding (cnctwin.core.keyframes.coarsen); the reading
 * whose extents do so is taken, [x, y] when both or neither do. */
export function gridFromArrays(arrays, name, prefixes, stockSizeM = null) {
  const doc = arrays[name];
  if (!doc) return null;
  const arr = decodeArrayDoc(doc);
  if (!arr || arr.shape.length < 2) return null;
  let [n0, n1] = arr.shape;
  let values = arr.data;
  if (arr.shape.length === 3) {
    const n2 = arr.shape[2];
    const reduced = new Float64Array(n0 * n1).fill(NaN);
    for (let i = 0; i < n0; i++) {
      for (let j = 0; j < n1; j++) {
        let best = NaN;
        for (let k = 0; k < n2; k++) {
          const v = values[(i * n1 + j) * n2 + k];
          if (Number.isFinite(v) && !(best >= v)) best = v;
        }
        reduced[i * n1 + j] = best;
      }
    }
    values = reduced;
  }
  let origin = null;
  let cell = null;
  for (const prefix of prefixes) {
    if (arrays[`${prefix}_origin_S_m`] && arrays[`${prefix}_cell_m`]) {
      origin = decodeArrayDoc(arrays[`${prefix}_origin_S_m`]).data;
      cell = scalar(arrays[`${prefix}_cell_m`]);
      break;
    }
  }
  // decide the axis order
  let xFirst = true;
  if (cell && stockSizeM) {
    // a grid over the stock covers it with at most one cell of padding: size <= n cell < size + cell
    const covers = (n, size) => n * cell >= size * (1 - 1e-9) && n * cell < size + cell * (1 - 1e-9);
    const fits = (nx, ny) => covers(nx, stockSizeM[0]) && covers(ny, stockSizeM[1]);
    if (!fits(n0, n1) && fits(n1, n0)) xFirst = false;
  }
  const nx = xFirst ? n0 : n1;
  const ny = xFirst ? n1 : n0;
  const out = new Float64Array(nx * ny);
  for (let ix = 0; ix < nx; ix++) {
    for (let iy = 0; iy < ny; iy++) {
      out[iy * nx + ix] = xFirst ? values[ix * n1 + iy] : values[iy * n1 + ix];
    }
  }
  const dx = cell ? cell * 1e3 : 1;
  return {
    values: out, nx, ny, x0: origin ? origin[0] * 1e3 : 0, y0: origin ? origin[1] * 1e3 : 0, dx, dy: dx,
    orientation: xFirst ? 'values[x, y]' : 'values[y, x]', cellsOnly: !cell,
  };
}

/** Values of a per-vertex attribute over (a sample of) keyframes/attr/*.npz: {values, files, of}. */
export async function attributeValues(api, data, name, maxFiles = 64) {
  const files = [...data.files.keys()].filter((p) => /^keyframes\/attr\/.+\.npz$/.test(p));
  const chosen = sampleEvenly(files, maxFiles);
  const values = [];
  for (const file of chosen) {
    try {
      const arrays = await api.npz(data.id, file);
      const arr = arrays[name];
      if (arr) for (const v of arr.data) if (Number.isFinite(v)) values.push(v);
    } catch { /* an unreadable brick is skipped (counted in `files`) */ }
  }
  return { values, files: chosen.length, of: files.length };
}

export function stats(values) {
  const finite = values.filter(Number.isFinite);
  if (!finite.length) return null;
  let sum = 0;
  let min = Infinity;
  let max = -Infinity;
  for (const v of finite) { sum += v; if (v < min) min = v; if (v > max) max = v; }
  const mean = sum / finite.length;
  let ss = 0;
  for (const v of finite) ss += (v - mean) ** 2;
  return { n: finite.length, mean, std: finite.length > 1 ? Math.sqrt(ss / (finite.length - 1)) : 0, min, max, range: max - min };
}

export { decodeArrayDoc };
