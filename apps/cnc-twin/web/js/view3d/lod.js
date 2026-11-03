/* Coarser versions of the workpiece mesh for far views (level of detail): vertex clustering
 * (J. Rossignac and P. Borrel, "Multi-resolution 3D approximations for rendering complex scenes",
 * Modeling in Computer Graphics, Springer 1993, pp. 455-465). Every vertex falls into a cube of edge
 * `cell` of a grid anchored at `anchor`; the vertices of one cube merge into their mean, triangles
 * with two corners in the same cube vanish. Error bound: every vertex moves by less than the cube
 * diagonal, sqrt(3) cell; the view picks the level whose cell projects to at most a set number of
 * pixels (params max_lod_cluster_px), so the shape changes by at most that on screen. Values carried
 * per vertex (layer attributes) are averaged over the cube's vertices that have one. */

export const NO_VALUE = -1e30;
export const HAS_VALUE = -1e29;

/** Cluster a triangle mesh. positions Float32Array [n * 3], indices Uint32Array [m * 3]; values
 * Float32Array [n] (NO_VALUE where none) or null. Returns {positions, indices, values, vertexMap}. */
export function clusterMesh(positions, indices, cell, anchor = [0, 0, 0], values = null) {
  const n = positions.length / 3;
  const vertexMap = new Uint32Array(n);
  const clusters = new Map();
  const sum = new Float64Array(n * 3);
  const count = new Uint32Array(n);
  const valueSum = values ? new Float64Array(n) : null;
  const valueCount = values ? new Uint32Array(n) : null;
  const inv = 1 / cell;
  const K = 1 << 17; // keys stay below 2^51: exact as doubles
  let next = 0;
  for (let i = 0; i < n; i++) {
    const x = positions[i * 3];
    const y = positions[i * 3 + 1];
    const z = positions[i * 3 + 2];
    const ix = Math.floor((x - anchor[0]) * inv) + (K >> 1);
    const iy = Math.floor((y - anchor[1]) * inv) + (K >> 1);
    const iz = Math.floor((z - anchor[2]) * inv) + (K >> 1);
    const key = (iz * K + iy) * K + ix;
    let c = clusters.get(key);
    if (c === undefined) { c = next++; clusters.set(key, c); }
    vertexMap[i] = c;
    sum[c * 3] += x;
    sum[c * 3 + 1] += y;
    sum[c * 3 + 2] += z;
    count[c]++;
    if (values && values[i] > HAS_VALUE) { valueSum[c] += values[i]; valueCount[c]++; }
  }
  const outPositions = new Float32Array(next * 3);
  for (let c = 0; c < next; c++) {
    outPositions[c * 3] = sum[c * 3] / count[c];
    outPositions[c * 3 + 1] = sum[c * 3 + 1] / count[c];
    outPositions[c * 3 + 2] = sum[c * 3 + 2] / count[c];
  }
  let outValues = null;
  if (values) {
    outValues = new Float32Array(next);
    for (let c = 0; c < next; c++) outValues[c] = valueCount[c] ? valueSum[c] / valueCount[c] : NO_VALUE;
  }
  const kept = new Uint32Array(indices.length);
  let m = 0;
  for (let t = 0; t + 2 < indices.length; t += 3) {
    const a = vertexMap[indices[t]];
    const b = vertexMap[indices[t + 1]];
    const c = vertexMap[indices[t + 2]];
    if (a === b || b === c || a === c) continue;
    kept[m++] = a;
    kept[m++] = b;
    kept[m++] = c;
  }
  return { positions: outPositions, indices: kept.slice(0, m), values: outValues, vertexMap };
}

/** Per-cluster values of a level from the full-resolution values and its vertexMap. */
export function clusterValues(values, vertexMap, clusterCount) {
  const sum = new Float64Array(clusterCount);
  const count = new Uint32Array(clusterCount);
  for (let i = 0; i < vertexMap.length; i++) {
    if (values[i] > HAS_VALUE) { sum[vertexMap[i]] += values[i]; count[vertexMap[i]]++; }
  }
  const out = new Float32Array(clusterCount);
  for (let c = 0; c < clusterCount; c++) out[c] = count[c] ? sum[c] / count[c] : NO_VALUE;
  return out;
}

/** The level to draw: the coarsest whose cube projects to <= maxPx pixels at `distance` (perspective
 * camera with vertical field of view `fovRad` over `viewportPx` pixel rows). cells[0] is level 0. */
export function chooseLevel(cells, distance, fovRad, viewportPx, maxPx) {
  const pxPerMetre = viewportPx / (2 * Math.max(distance, 1e-9) * Math.tan(fovRad / 2));
  let level = 0;
  for (let i = 1; i < cells.length; i++) if (cells[i] * pxPerMetre <= maxPx) level = i;
  return level;
}
