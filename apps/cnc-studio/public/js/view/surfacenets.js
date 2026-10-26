// Naive surface nets over one chunk of a VoxelSim distance field.
// Returns typed arrays (part-frame mm) ready for a THREE.BufferGeometry, or null if empty.

export function meshChunk(sim, chunkIndex, colorTable) {
  const { nx, ny, nz, h, sdf, toolIdx, chunk: C, cnx, cny } = sim;
  const [ox, oy, oz] = sim.origin;
  const ci = chunkIndex % cnx, cj = Math.floor(chunkIndex / cnx) % cny, ck = Math.floor(chunkIndex / (cnx * cny));
  // cells (base sample i..i+1) owned by this chunk
  const i0 = ci * C, i1 = Math.min(nx - 1, i0 + C);
  const j0 = cj * C, j1 = Math.min(ny - 1, j0 + C);
  const k0 = ck * C, k1 = Math.min(nz - 1, k0 + C);
  // vertex map covers one extra cell on the low side (quads of owned edges need it)
  const a0 = Math.max(0, i0 - 1), b0 = Math.max(0, j0 - 1), c0 = Math.max(0, k0 - 1);
  const W = i1 - a0, H = j1 - b0, D = k1 - c0;
  if (W <= 0 || H <= 0 || D <= 0) return null;
  const vid = new Int32Array(W * H * D).fill(-1);
  const pos = [], nor = [], col = [], idx = [];
  const sNy = nx, sNz = nx * ny;
  const corner = new Float32Array(8);
  const cornerOff = [0, 1, sNy, sNy + 1, sNz, sNz + 1, sNz + sNy, sNz + sNy + 1];
  // edges as pairs of corner indices (bit0 = +x, bit1 = +y, bit2 = +z)
  const EDGES = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];

  for (let k = c0; k < k1; k++) {
    for (let j = b0; j < j1; j++) {
      let base = k * sNz + j * sNy + a0;
      for (let i = a0; i < i1; i++, base++) {
        let mask = 0;
        for (let c = 0; c < 8; c++) {
          const v = sdf[base + cornerOff[c]];
          corner[c] = v;
          if (v < 0) mask |= 1 << c;
        }
        if (mask === 0 || mask === 255) continue;
        let sx = 0, sy = 0, sz = 0, n = 0;
        for (const [e0, e1] of EDGES) {
          const v0 = corner[e0], v1 = corner[e1];
          if ((v0 < 0) === (v1 < 0)) continue;
          const t = v0 / (v0 - v1);
          sx += (e0 & 1) + ((e1 & 1) - (e0 & 1)) * t;
          sy += ((e0 >> 1) & 1) + (((e1 >> 1) & 1) - ((e0 >> 1) & 1)) * t;
          sz += ((e0 >> 2) & 1) + (((e1 >> 2) & 1) - ((e0 >> 2) & 1)) * t;
          n++;
        }
        const vx = ox + (i + sx / n) * h, vy = oy + (j + sy / n) * h, vz = oz + (k + sz / n) * h;
        // gradient of the field = outward normal
        let gx = (corner[1] + corner[3] + corner[5] + corner[7]) - (corner[0] + corner[2] + corner[4] + corner[6]);
        let gy = (corner[2] + corner[3] + corner[6] + corner[7]) - (corner[0] + corner[1] + corner[4] + corner[5]);
        let gz = (corner[4] + corner[5] + corner[6] + corner[7]) - (corner[0] + corner[1] + corner[2] + corner[3]);
        const gl = Math.hypot(gx, gy, gz) || 1;
        // colour: tool of the material corner nearest the surface
        let best = 0, bv = -1e9;
        for (let c = 0; c < 8; c++) if (corner[c] < 0 && corner[c] > bv) { bv = corner[c]; best = c; }
        const t = toolIdx[base + cornerOff[best]];
        const vIndex = pos.length / 3;
        pos.push(vx, vy, vz);
        nor.push(gx / gl, gy / gl, gz / gl);
        col.push(colorTable[t * 3], colorTable[t * 3 + 1], colorTable[t * 3 + 2]);
        vid[((k - c0) * H + (j - b0)) * W + (i - a0)] = vIndex;
      }
    }
  }
  if (!pos.length) return null;
  const V = (i, j, k) => (i < a0 || j < b0 || k < c0 || i >= i1 || j >= j1 || k >= k1 ? -1 : vid[((k - c0) * H + (j - b0)) * W + (i - a0)]);
  const quad = (a, b, c, d, flip) => {
    if (a < 0 || b < 0 || c < 0 || d < 0) return;
    if (flip) idx.push(a, c, b, a, d, c); else idx.push(a, b, c, a, c, d);
  };
  // owned edges: base sample inside the chunk's cell range
  for (let k = k0; k < k1; k++) {
    for (let j = j0; j < j1; j++) {
      for (let i = i0; i < i1; i++) {
        const s = k * sNz + j * sNy + i;
        const inside = sdf[s] < 0;
        // +x edge
        if (i + 1 < nx && (sdf[s + 1] < 0) !== inside && j > 0 && k > 0) {
          quad(V(i, j - 1, k - 1), V(i, j, k - 1), V(i, j, k), V(i, j - 1, k), !inside);
        }
        // +y edge
        if (j + 1 < ny && (sdf[s + sNy] < 0) !== inside && i > 0 && k > 0) {
          quad(V(i - 1, j, k - 1), V(i - 1, j, k), V(i, j, k), V(i, j, k - 1), !inside);
        }
        // +z edge
        if (k + 1 < nz && (sdf[s + sNz] < 0) !== inside && i > 0 && j > 0) {
          quad(V(i - 1, j - 1, k), V(i, j - 1, k), V(i, j, k), V(i - 1, j, k), !inside);
        }
      }
    }
  }
  if (!idx.length) return null;
  return {
    positions: new Float32Array(pos),
    normals: new Float32Array(nor),
    colors: new Float32Array(col),
    index: pos.length / 3 > 65535 ? new Uint32Array(idx) : new Uint16Array(idx),
  };
}
