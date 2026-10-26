// Full 3D material model for multi-setup / rotary milling.
// The part lives in its own fixed "part frame" (the first operation's work coordinates).
// Material is a narrow-band signed distance field on a voxel grid (negative inside,
// clamped to ±2 cells, one Int8 per voxel). A cut maps voxels into the current setup's
// work frame (p = R·q + t, optionally rotated by A/B) and subtracts the tool's swept volume:
// m' = max(m, -tool). Surfaces are extracted with surface nets (view/surfacenets.js).

const Q = 127;

export const IDENTITY = Object.freeze({ R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [0, 0, 0] });

export function mulXform(A, B) {
  // (A∘B)(q) = A(B(q))
  const a = A.R, b = B.R;
  const R = [
    a[0] * b[0] + a[1] * b[3] + a[2] * b[6], a[0] * b[1] + a[1] * b[4] + a[2] * b[7], a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
    a[3] * b[0] + a[4] * b[3] + a[5] * b[6], a[3] * b[1] + a[4] * b[4] + a[5] * b[7], a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
    a[6] * b[0] + a[7] * b[3] + a[8] * b[6], a[6] * b[1] + a[7] * b[4] + a[8] * b[7], a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
  ];
  const t = applyR(A.R, B.t);
  return { R, t: [t[0] + A.t[0], t[1] + A.t[1], t[2] + A.t[2]] };
}
export const applyR = (R, v) => [R[0] * v[0] + R[1] * v[1] + R[2] * v[2], R[3] * v[0] + R[4] * v[1] + R[5] * v[2], R[6] * v[0] + R[7] * v[1] + R[8] * v[2]];
export const applyX = (X, q) => { const p = applyR(X.R, q); return [p[0] + X.t[0], p[1] + X.t[1], p[2] + X.t[2]]; };
export function invertX(X) {
  const R = X.R;
  const Rt = [R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]];
  const t = applyR(Rt, X.t);
  return { R: Rt, t: [-t[0], -t[1], -t[2]] };
}
export function rotX(deg) { const c = Math.cos(deg * Math.PI / 180), s = Math.sin(deg * Math.PI / 180); return { R: [1, 0, 0, 0, c, -s, 0, s, c], t: [0, 0, 0] }; }
export function rotY(deg) { const c = Math.cos(deg * Math.PI / 180), s = Math.sin(deg * Math.PI / 180); return { R: [c, 0, s, 0, 1, 0, -s, 0, c], t: [0, 0, 0] }; }
export function rotZ(deg) { const c = Math.cos(deg * Math.PI / 180), s = Math.sin(deg * Math.PI / 180); return { R: [c, -s, 0, s, c, 0, 0, 0, 1], t: [0, 0, 0] }; }
export function translate(v) { return { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: v.slice() }; }
// clean up rounding so 90° steps stay exact
export function snapX(X) {
  return { R: X.R.map((v) => (Math.abs(v - Math.round(v)) < 1e-9 ? Math.round(v) : v)), t: X.t.map((v) => (Math.abs(v) < 1e-9 ? 0 : v)) };
}

// radius of the tool at height w above its tip
function rhoFn(tool) {
  const r = tool.diameter / 2;
  switch (tool.type) {
    case 'ball': return (w) => (w >= r ? r : Math.sqrt(Math.max(0, r * r - (r - w) * (r - w))));
    case 'bull': {
      const cr = Math.min(tool.cornerRadius || 0, r);
      return (w) => (w >= cr ? r : (r - cr) + Math.sqrt(Math.max(0, cr * cr - (cr - w) * (cr - w))));
    }
    case 'drill': case 'spot': {
      const k = Math.tan(((tool.tipAngle || 118) * Math.PI) / 360);
      return (w) => Math.min(r, w * k);
    }
    case 'chamfer': case 'engrave': {
      const k = Math.tan(((tool.tipAngle || 90) * Math.PI) / 360);
      const t0 = (tool.tipDiameter || 0) / 2;
      return (w) => Math.min(r, t0 + w * k);
    }
    default: return () => r;
  }
}

export class VoxelSim {
  // bounds = { min, max } in the part frame; the grid covers it with a margin
  constructor(bounds, tools, { cell, budget = 6e6 } = {}) {
    this.tools = tools;
    const size = [0, 1, 2].map((k) => Math.max(1, bounds.max[k] - bounds.min[k]));
    let h = cell || Math.cbrt((size[0] * size[1] * size[2]) / budget);
    h = Math.max(0.15, Math.min(2, h));
    const pad = 3;
    this.h = h;
    this.S = 2 * h;
    this.origin = [0, 1, 2].map((k) => bounds.min[k] - pad * h);
    this.nx = Math.ceil(size[0] / h) + 1 + 2 * pad;
    this.ny = Math.ceil(size[1] / h) + 1 + 2 * pad;
    this.nz = Math.ceil(size[2] / h) + 1 + 2 * pad;
    const N = this.nx * this.ny * this.nz;
    this.sdf = new Int8Array(N).fill(Q);
    this.toolIdx = new Uint8Array(N);
    this.bounds = { min: bounds.min.slice(), max: bounds.max.slice() };
    this.chunk = 32;
    this.cnx = Math.ceil((this.nx - 1) / this.chunk);
    this.cny = Math.ceil((this.ny - 1) / this.chunk);
    this.cnz = Math.ceil((this.nz - 1) / this.chunk);
    this.dirty = new Uint8Array(this.cnx * this.cny * this.cnz).fill(1);
    this.anyDirty = true;
    this.xform = IDENTITY;
    this.kind = 'voxel';
    this.version = 1;
  }

  // ---------------------------------------------------------------- construction
  static fromBox(min, max, tools, opts) {
    const s = new VoxelSim({ min, max }, tools, opts);
    const c = [0, 1, 2].map((k) => (min[k] + max[k]) / 2), hb = [0, 1, 2].map((k) => (max[k] - min[k]) / 2);
    s.fill((x, y, z) => {
      const dx = Math.abs(x - c[0]) - hb[0], dy = Math.abs(y - c[1]) - hb[1], dz = Math.abs(z - c[2]) - hb[2];
      const out = Math.hypot(Math.max(dx, 0), Math.max(dy, 0), Math.max(dz, 0));
      return out + Math.min(Math.max(dx, dy, dz), 0);
    });
    s.initialVolume = s.materialVolume();
    return s;
  }

  // a turned part (LatheSim profile) as a solid: lathe axis = part Z, radial X at angle 0
  static fromLathe(lsim, tools, opts) {
    let R = 0;
    for (let i = 0; i < lsim.n; i++) R = Math.max(R, lsim.outer[i]);
    const zLo = lsim.zPart ?? lsim.zMin, zHi = lsim.zMax;
    const s = new VoxelSim({ min: [-R, -R, zLo], max: [R, R, zHi] }, tools, opts);
    const at = (arr, z) => {
      const f = (z - lsim.zMin) / lsim.dz;
      const i = Math.max(0, Math.min(lsim.n - 1, Math.round(f)));
      return arr[i];
    };
    s.fill((x, y, z) => {
      if (z < lsim.zMin - 1e-9 || z > zHi + 1e-9) return Math.max(lsim.zMin - z, z - zHi);
      const r = Math.hypot(x, y);
      const o = at(lsim.outer, z), inn = at(lsim.inner, z);
      if (o - inn < 1e-4) return s.S; // nothing left in this column
      return Math.max(r - o, inn - r, lsim.zMin - z, z - zHi);
    });
    s.initialVolume = s.materialVolume();
    return s;
  }

  // a heightmap MillSim result as a solid (single-setup part handed to a later operation)
  static fromHeightmap(ms, tools, opts) {
    const st = ms.stock;
    const s = new VoxelSim({ min: st.min, max: st.max }, tools, opts);
    const c = [0, 1, 2].map((k) => (st.min[k] + st.max[k]) / 2), hb = [0, 1, 2].map((k) => (st.max[k] - st.min[k]) / 2);
    const H = (x, y) => {
      const i = Math.max(0, Math.min(ms.nx - 1, Math.round((x - ms.minX) / ms.cx)));
      const j = Math.max(0, Math.min(ms.ny - 1, Math.round((y - ms.minY) / ms.cy)));
      const v = ms.heights[j * ms.nx + i];
      return v < ms.bottom ? -1e9 : v;
    };
    s.fill((x, y, z) => {
      const dx = Math.abs(x - c[0]) - hb[0], dy = Math.abs(y - c[1]) - hb[1], dz = Math.abs(z - c[2]) - hb[2];
      const box = Math.hypot(Math.max(dx, 0), Math.max(dy, 0), Math.max(dz, 0)) + Math.min(Math.max(dx, dy, dz), 0);
      return Math.max(box, z - H(x, y));
    });
    s.initialVolume = s.materialVolume();
    return s;
  }

  // closed triangle mesh (Float32Array of xyz triplets per vertex, 9 per triangle) as a solid
  static fromMesh(tris, tools, opts) {
    let mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < tris.length; i += 3) for (let k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], tris[i + k]); mx[k] = Math.max(mx[k], tris[i + k]); }
    const s = new VoxelSim({ min: mn, max: mx }, tools, opts);
    s.fillFromMesh(tris);
    s.initialVolume = s.materialVolume();
    return s;
  }

  clone() {
    const c = Object.create(VoxelSim.prototype);
    Object.assign(c, this);
    c.sdf = this.sdf.slice();
    c.toolIdx = this.toolIdx.slice();
    c.dirty = new Uint8Array(this.dirty.length).fill(1);
    c.touched = new Uint8Array(this.dirty.length);
    c.anyDirty = true;
    c.bounds = { min: this.bounds.min.slice(), max: this.bounds.max.slice() };
    return c;
  }

  // copy material state from another sim of the same grid
  copyFrom(o) {
    this.sdf.set(o.sdf);
    this.toolIdx.set(o.toolIdx);
    this.dirty.fill(1);
    this.anyDirty = true;
  }

  fill(fn) {
    const { nx, ny, nz, h, S, sdf } = this;
    const [ox, oy, oz] = this.origin;
    let idx = 0;
    for (let k = 0; k < nz; k++) {
      const z = oz + k * h;
      for (let j = 0; j < ny; j++) {
        const y = oy + j * h;
        for (let i = 0; i < nx; i++, idx++) {
          const m = fn(ox + i * h, y, z);
          sdf[idx] = Math.max(-Q, Math.min(Q, Math.round((Math.max(-S, Math.min(S, m)) / S) * Q)));
        }
      }
    }
    this.dirty.fill(1);
    this.anyDirty = true;
  }

  // voxelize a closed mesh: ray-crossing parity along X, Y and Z through every voxel row.
  // Inside = majority of the three (a crack or a ray grazing a shared edge only spoils one);
  // the distance is the nearest crossing on any axis, so walls in every direction are sub-voxel.
  fillFromMesh(tris) {
    const { nx, ny, nz, h, S, sdf } = this;
    const n = [nx, ny, nz];
    const dist = new Float32Array(sdf.length).fill(S);
    const votes = new Uint8Array(sdf.length);
    const stride = [1, nx, nx * ny];
    for (let ax = 0; ax < 3; ax++) {
      const a1 = (ax + 1) % 3, a2 = (ax + 2) % 3; // the row is along ax, rows indexed by (a1, a2)
      const o = this.origin;
      // rays are nudged off the grid so they miss shared edges; bin triangles at the nudged
      // position too, or a ray just past a mesh row at a grid plane falls between two triangles
      const jp = 1e-7 * (ax + 1), jq = 1.3e-7 * (ax + 1);
      const rows = new Map();
      for (let t = 0; t < tris.length; t += 9) {
        let p0 = Infinity, p1 = -Infinity, q0 = Infinity, q1 = -Infinity;
        for (let v = 0; v < 3; v++) { const p = tris[t + v * 3 + a1], q = tris[t + v * 3 + a2]; p0 = Math.min(p0, p); p1 = Math.max(p1, p); q0 = Math.min(q0, q); q1 = Math.max(q1, q); }
        const i0 = Math.max(0, Math.ceil((p0 - o[a1] - jp) / h)), i1 = Math.min(n[a1] - 1, Math.floor((p1 - o[a1] - jp) / h));
        const j0 = Math.max(0, Math.ceil((q0 - o[a2] - jq) / h)), j1 = Math.min(n[a2] - 1, Math.floor((q1 - o[a2] - jq) / h));
        for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
          const key = j * n[a1] + i;
          let list = rows.get(key);
          if (!list) { list = []; rows.set(key, list); }
          list.push(t);
        }
      }
      for (const [key, list] of rows) {
        const i = key % n[a1], j = (key - i) / n[a1];
        const p = o[a1] + i * h + jp, q = o[a2] + j * h + jq;
        const cs = [];
        for (const t of list) {
          const ap = tris[t + a1], aq = tris[t + a2], aa = tris[t + ax];
          const bp = tris[t + 3 + a1], bq = tris[t + 3 + a2], ba = tris[t + 3 + ax];
          const cp = tris[t + 6 + a1], cq = tris[t + 6 + a2], ca = tris[t + 6 + ax];
          const d = (bq - cq) * (ap - cp) + (cp - bp) * (aq - cq);
          if (Math.abs(d) < 1e-14) continue;
          const l1 = ((bq - cq) * (p - cp) + (cp - bp) * (q - cq)) / d;
          const l2 = ((cq - aq) * (p - cp) + (ap - cp) * (q - cq)) / d;
          const l3 = 1 - l1 - l2;
          if (l1 < 0 || l2 < 0 || l3 < 0) continue;
          cs.push(l1 * aa + l2 * ba + l3 * ca);
        }
        if (cs.length < 2) continue;
        cs.sort((x, y) => x - y);
        // the same crossing reported by two triangles (a ray through a shared edge) counts once
        const zs = [cs[0]];
        for (let k = 1; k < cs.length; k++) if (cs[k] - zs[zs.length - 1] > 1e-6) zs.push(cs[k]);
        const base = i * stride[a1] + j * stride[a2];
        let c = 0;
        for (let k = 0; k < n[ax]; k++) {
          const z = o[ax] + k * h;
          while (c < zs.length && zs[c] < z) c++;
          const idx = base + k * stride[ax];
          if (c % 2 === 1) votes[idx]++;
          const dn = Math.min(c < zs.length ? zs[c] - z : Infinity, c > 0 ? z - zs[c - 1] : Infinity);
          if (dn < dist[idx]) dist[idx] = dn;
        }
      }
    }
    for (let idx = 0; idx < sdf.length; idx++) {
      const m = (votes[idx] >= 2 ? -1 : 1) * Math.min(dist[idx], S);
      sdf[idx] = Math.max(-Q, Math.min(Q, Math.round((m / S) * Q)));
    }
    this.dirty.fill(1);
    this.anyDirty = true;
  }

  reset() {
    // VoxelSim has no implicit initial state; callers keep a pristine copy (see SimRunner).
    // Only chunks cut since the last reset change (keeps re-meshing after a rewind cheap).
    if (this.pristine) {
      this.sdf.set(this.pristine.sdf);
      this.toolIdx.set(this.pristine.toolIdx);
      if (this.touched) {
        for (let c = 0; c < this.touched.length; c++) if (this.touched[c]) { this.dirty[c] = 1; this.touched[c] = 0; this.anyDirty = true; }
      } else { this.dirty.fill(1); this.anyDirty = true; }
    }
    this.version++;
  }

  // ---------------------------------------------------------------- queries
  val(i, j, k) {
    if (i < 0 || j < 0 || k < 0 || i >= this.nx || j >= this.ny || k >= this.nz) return this.S;
    return (this.sdf[(k * this.ny + j) * this.nx + i] / Q) * this.S;
  }
  // trilinear SDF in mm at a part-frame point (positive = air)
  sample(x, y, z) {
    const fx = (x - this.origin[0]) / this.h, fy = (y - this.origin[1]) / this.h, fz = (z - this.origin[2]) / this.h;
    const i = Math.floor(fx), j = Math.floor(fy), k = Math.floor(fz);
    const u = fx - i, v = fy - j, w = fz - k;
    const nx = this.nx, ny = this.ny;
    if (i >= 0 && j >= 0 && k >= 0 && i < nx - 1 && j < ny - 1 && k < this.nz - 1) {
      // interior: read the 8 corners directly
      const d = this.sdf, sz = nx * ny, b = (k * ny + j) * nx + i;
      const x00 = d[b] + (d[b + 1] - d[b]) * u, x10 = d[b + nx] + (d[b + nx + 1] - d[b + nx]) * u;
      const x01 = d[b + sz] + (d[b + sz + 1] - d[b + sz]) * u, x11 = d[b + sz + nx] + (d[b + sz + nx + 1] - d[b + sz + nx]) * u;
      const y0 = x00 + (x10 - x00) * v, y1 = x01 + (x11 - x01) * v;
      return ((y0 + (y1 - y0) * w) / Q) * this.S;
    }
    const c = (a, b, t) => a + (b - a) * t;
    return c(
      c(c(this.val(i, j, k), this.val(i + 1, j, k), u), c(this.val(i, j + 1, k), this.val(i + 1, j + 1, k), u), v),
      c(c(this.val(i, j, k + 1), this.val(i + 1, j, k + 1), u), c(this.val(i, j + 1, k + 1), this.val(i + 1, j + 1, k + 1), u), v),
      w,
    );
  }
  isMaterial(x, y, z) { return this.sample(x, y, z) < 0; }

  // fractional occupancy from the distance field: a voxel whose centre is d from the surface
  // is about (0.5 - d/h) full, which is exact for axis-aligned faces
  materialVolume() {
    const k = this.S / Q / this.h;
    let n = 0;
    for (let i = 0; i < this.sdf.length; i++) {
      const v = this.sdf[i];
      if (v >= Q) continue;
      const f = 0.5 - v * k;
      n += f >= 1 ? 1 : f <= 0 ? 0 : f;
    }
    return n * this.h * this.h * this.h;
  }
  removedVolume() { return Math.max(0, (this.initialVolume ?? this.materialVolume()) - this.materialVolume()); }

  // actual bounds of the remaining material (part frame)
  materialBounds() {
    const { nx, ny, nz, h, sdf } = this;
    let i0 = nx, i1 = -1, j0 = ny, j1 = -1, k0 = nz, k1 = -1;
    let idx = 0;
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++, idx++) {
      if (sdf[idx] < 0) { if (i < i0) i0 = i; if (i > i1) i1 = i; if (j < j0) j0 = j; if (j > j1) j1 = j; if (k < k0) k0 = k; if (k > k1) k1 = k; }
    }
    if (i1 < 0) return null;
    // sub-voxel: where the distance field crosses zero beyond the outermost inside voxels
    const sNy = nx, sNz = nx * ny;
    const ext = (axis, layer, dir) => {
      const [na, nb, sa, sb, sl] = axis === 0 ? [ny, nz, sNy, sNz, 1] : axis === 1 ? [nx, nz, 1, sNz, sNy] : [nx, ny, 1, sNy, sNz];
      const lim = axis === 0 ? nx : axis === 1 ? ny : nz;
      if (layer + dir < 0 || layer + dir >= lim) return layer;
      let best = 0;
      for (let b = 0; b < nb; b++) for (let a = 0; a < na; a++) {
        const q = a * sa + b * sb + layer * sl;
        const v = sdf[q];
        if (v >= 0) continue;
        const w = sdf[q + dir * sl];
        const t = w > v ? v / (v - w) : 0;
        if (t > best) best = t;
      }
      return layer + dir * best;
    };
    const o = this.origin;
    return {
      min: [o[0] + ext(0, i0, -1) * h, o[1] + ext(1, j0, -1) * h, o[2] + ext(2, k0, -1) * h],
      max: [o[0] + ext(0, i1, 1) * h, o[1] + ext(1, j1, 1) * h, o[2] + ext(2, k1, 1) * h],
    };
  }

  // work-frame AABB of the part's grid bounds under a transform
  workBounds(X) {
    const b = this.bounds;
    let mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
    for (const cx of [b.min[0], b.max[0]]) for (const cy of [b.min[1], b.max[1]]) for (const cz of [b.min[2], b.max[2]]) {
      const p = applyX(X, [cx, cy, cz]);
      for (let k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], p[k]); mx[k] = Math.max(mx[k], p[k]); }
    }
    return { min: mn, max: mx };
  }

  takeDirty() {
    if (!this.anyDirty) return null;
    this.anyDirty = false;
    const list = [];
    for (let c = 0; c < this.dirty.length; c++) if (this.dirty[c]) { list.push(c); this.dirty[c] = 0; }
    return list;
  }

  markDirty(i0, i1, j0, j1, k0, k1) {
    const C = this.chunk;
    const a0 = Math.max(0, Math.floor((i0 - 1) / C)), a1 = Math.min(this.cnx - 1, Math.floor(i1 / C));
    const b0 = Math.max(0, Math.floor((j0 - 1) / C)), b1 = Math.min(this.cny - 1, Math.floor(j1 / C));
    const c0 = Math.max(0, Math.floor((k0 - 1) / C)), c1 = Math.min(this.cnz - 1, Math.floor(k1 / C));
    const touched = this.touched || (this.touched = new Uint8Array(this.dirty.length));
    for (let c = c0; c <= c1; c++) for (let b = b0; b <= b1; b++) for (let a = a0; a <= a1; a++) { const q = (c * this.cny + b) * this.cnx + a; this.dirty[q] = 1; touched[q] = 1; }
    this.anyDirty = true;
  }

  // ---------------------------------------------------------------- cutting
  // a, b: tool tip in the work frame; X: part -> work transform for this move
  cutSegment(a, b, move, X = this.xform) {
    const tool = this.tools[move.tool];
    if (!tool || !(tool.diameter > 0)) return null;
    const r = tool.diameter / 2;
    const rho = rhoFn(tool);
    const wb = this.workBounds(X);
    const zLow = Math.min(a[2], b[2]);
    if (zLow >= wb.max[2] + this.h) return null;
    const { h, S, nx, ny, nz, sdf, toolIdx } = this;
    const [ox, oy, oz] = this.origin;
    // work-frame box swept by the cutting part of the tool
    const wmin = [Math.min(a[0], b[0]) - r - h, Math.min(a[1], b[1]) - r - h, zLow - h];
    const wmax = [Math.max(a[0], b[0]) + r + h, Math.max(a[1], b[1]) + r + h, wb.max[2] + h];
    if (wmin[0] > wb.max[0] || wmax[0] < wb.min[0] || wmin[1] > wb.max[1] || wmax[1] < wb.min[1]) return null;
    const inv = invertX(X);
    let pmn = [Infinity, Infinity, Infinity], pmx = [-Infinity, -Infinity, -Infinity];
    for (const cx of [wmin[0], wmax[0]]) for (const cy of [wmin[1], wmax[1]]) for (const cz of [wmin[2], wmax[2]]) {
      const q = applyX(inv, [cx, cy, cz]);
      for (let k = 0; k < 3; k++) { pmn[k] = Math.min(pmn[k], q[k]); pmx[k] = Math.max(pmx[k], q[k]); }
    }
    const i0 = Math.max(0, Math.floor((pmn[0] - ox) / h)), i1 = Math.min(nx - 1, Math.ceil((pmx[0] - ox) / h));
    const j0 = Math.max(0, Math.floor((pmn[1] - oy) / h)), j1 = Math.min(ny - 1, Math.ceil((pmx[1] - oy) / h));
    const k0 = Math.max(0, Math.floor((pmn[2] - oz) / h)), k1 = Math.min(nz - 1, Math.ceil((pmx[2] - oz) / h));
    if (i0 > i1 || j0 > j1 || k0 > k1) return null;
    const R = X.R, T = X.t;
    const dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
    const L2 = dx * dx + dy * dy;
    const flute = tool.fluteLength || 1e9;
    const tnum = Math.min(255, tool.num | 0);
    const sQ = Q / S;
    let removedN = 0, changed = false, shank = 0;
    let ci0 = nx, ci1 = -1, cj0 = ny, cj1 = -1, ck0 = nz, ck1 = -1;
    for (let k = k0; k <= k1; k++) {
      const qz = oz + k * h;
      for (let j = j0; j <= j1; j++) {
        const qy = oy + j * h;
        const qx0 = ox + i0 * h;
        // work-frame point of the row start, then step by the first column of R
        let px = R[0] * qx0 + R[1] * qy + R[2] * qz + T[0];
        let py = R[3] * qx0 + R[4] * qy + R[5] * qz + T[1];
        let pz = R[6] * qx0 + R[7] * qy + R[8] * qz + T[2];
        const sx = R[0] * h, sy = R[3] * h, sz = R[6] * h;
        let idx = (k * ny + j) * nx + i0;
        for (let i = i0; i <= i1; i++, idx++, px += sx, py += sy, pz += sz) {
          const v = sdf[idx];
          if (v >= Q) continue; // already air
          if (pz < zLow - S) continue;
          let t = 0;
          if (L2 > 1e-12) { t = ((px - a[0]) * dx + (py - a[1]) * dy) / L2; t = t < 0 ? 0 : t > 1 ? 1 : t; }
          const ex = a[0] + t * dx - px, ey = a[1] + t * dy - py;
          const d2 = Math.sqrt(ex * ex + ey * ey);
          if (d2 > r + S) continue;
          const zt = L2 > 1e-12 ? a[2] + t * dz : zLow;
          const w = pz - zt;
          const st = Math.max(d2 - rho(w < 0 ? 0 : w), -w); // tool SDF, negative inside
          const nv = Math.round(Math.min(S, -st) * sQ);
          if (nv <= v) continue;
          if (v < 0 && nv >= 0) {
            removedN++;
            if (w > flute + h) shank = Math.max(shank, w - flute);
          }
          sdf[idx] = nv > Q ? Q : nv;
          toolIdx[idx] = tnum;
          changed = true;
          if (i < ci0) ci0 = i; if (i > ci1) ci1 = i;
          if (j < cj0) cj0 = j; if (j > cj1) cj1 = j;
          if (k < ck0) ck0 = k; if (k > ck1) ck1 = k;
        }
      }
    }
    if (changed) this.markDirty(ci0, ci1 + 1, cj0, cj1 + 1, ck0, ck1 + 1);
    const removedVol = removedN * h * h * h;
    // shank wider than the cutter / holder against remaining material
    let holder = 0;
    const sr = (tool.shankDiameter || tool.diameter) / 2;
    if (sr > r + 0.01 && zLow + flute < wb.max[2]) shank = Math.max(shank, this.interference(a, b, sr, flute, X));
    if (tool.overallLength && zLow + tool.overallLength < wb.max[2]) holder = this.interference(a, b, (tool.holderDiameter || 32) / 2, tool.overallLength, X);
    return { removed: removedVol > 0 ? Math.max(h, removedVol / (Math.PI * r * r)) : 0, removedVol, shank, holder };
  }

  // radial penetration of remaining material into a cylinder (radius R) that starts h0 above the tip
  interference(a, b, Rc, h0, X) {
    const { h, nx, ny, nz, sdf } = this;
    const [ox, oy, oz] = this.origin;
    const wb = this.workBounds(X);
    const zLow = Math.min(a[2], b[2]) + h0;
    if (zLow >= wb.max[2]) return 0;
    const wmin = [Math.min(a[0], b[0]) - Rc, Math.min(a[1], b[1]) - Rc, zLow];
    const wmax = [Math.max(a[0], b[0]) + Rc, Math.max(a[1], b[1]) + Rc, wb.max[2]];
    const inv = invertX(X);
    let pmn = [Infinity, Infinity, Infinity], pmx = [-Infinity, -Infinity, -Infinity];
    for (const cx of [wmin[0], wmax[0]]) for (const cy of [wmin[1], wmax[1]]) for (const cz of [wmin[2], wmax[2]]) {
      const q = applyX(inv, [cx, cy, cz]);
      for (let k = 0; k < 3; k++) { pmn[k] = Math.min(pmn[k], q[k]); pmx[k] = Math.max(pmx[k], q[k]); }
    }
    const st = Math.max(1, Math.round(0.5 / h)); // coarse scan is enough for a yes/no + depth
    const i0 = Math.max(0, Math.floor((pmn[0] - ox) / h)), i1 = Math.min(nx - 1, Math.ceil((pmx[0] - ox) / h));
    const j0 = Math.max(0, Math.floor((pmn[1] - oy) / h)), j1 = Math.min(ny - 1, Math.ceil((pmx[1] - oy) / h));
    const k0 = Math.max(0, Math.floor((pmn[2] - oz) / h)), k1 = Math.min(nz - 1, Math.ceil((pmx[2] - oz) / h));
    const R = X.R, T = X.t;
    const dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
    const L2 = dx * dx + dy * dy;
    let worst = 0;
    for (let k = k0; k <= k1; k += st) for (let j = j0; j <= j1; j += st) for (let i = i0; i <= i1; i += st) {
      if (sdf[(k * ny + j) * nx + i] >= 0) continue;
      const qx = ox + i * h, qy = oy + j * h, qz = oz + k * h;
      const px = R[0] * qx + R[1] * qy + R[2] * qz + T[0], py = R[3] * qx + R[4] * qy + R[5] * qz + T[1], pz = R[6] * qx + R[7] * qy + R[8] * qz + T[2];
      let t = 0;
      if (L2 > 1e-12) { t = ((px - a[0]) * dx + (py - a[1]) * dy) / L2; t = t < 0 ? 0 : t > 1 ? 1 : t; }
      const zt = (L2 > 1e-12 ? a[2] + t * dz : Math.min(a[2], b[2])) + h0;
      if (pz < zt) continue;
      const ex = a[0] + t * dx - px, ey = a[1] + t * dy - py;
      const pen = Rc - Math.sqrt(ex * ex + ey * ey);
      if (pen > worst) worst = pen;
    }
    return worst;
  }
}
