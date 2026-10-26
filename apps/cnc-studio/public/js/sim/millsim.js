// 3-axis material removal on a heightmap (one height per grid node).
// Each feed/rapid segment is swept analytically: for every grid node within
// reach of the segment the lowest tool position covering it is found, then the
// tool's profile (flat / ball / bull / cone) is applied.

import { millProfile } from '../gcode/tools.js';

export class MillSim {
  constructor(stock, tools, { resolution = 420 } = {}) {
    this.stock = stock;
    this.tools = tools;
    const [x0, y0, z0] = stock.min;
    const [x1, y1, z1] = stock.max;
    this.minX = x0; this.minY = y0; this.bottom = z0; this.top = z1;
    const sx = Math.max(1e-3, x1 - x0), sy = Math.max(1e-3, y1 - y0);
    const cell = Math.max(sx, sy) / resolution;
    this.nx = Math.max(2, Math.round(sx / cell) + 1);
    this.ny = Math.max(2, Math.round(sy / cell) + 1);
    this.cx = sx / (this.nx - 1);
    this.cy = sy / (this.ny - 1);
    this.heights = new Float32Array(this.nx * this.ny);
    this.toolIdx = new Uint8Array(this.nx * this.ny);
    this.profiles = new Map();
    this.reset();
  }

  reset() {
    this.heights.fill(this.top);
    this.toolIdx.fill(0);
    this.dirty = { i0: 0, i1: this.nx - 1, j0: 0, j1: this.ny - 1 };
    this.events = [];
    this.version = (this.version || 0) + 1;
  }

  clone() {
    const c = Object.create(MillSim.prototype);
    Object.assign(c, this);
    c.heights = this.heights.slice();
    c.toolIdx = this.toolIdx.slice();
    c.profiles = new Map();
    c.dirty = { i0: 0, i1: this.nx - 1, j0: 0, j1: this.ny - 1 };
    c.events = [];
    return c;
  }

  profile(tool) {
    let p = this.profiles.get(tool);
    if (!p) { p = millProfile(tool); this.profiles.set(tool, p); }
    return p;
  }

  markDirty(i0, i1, j0, j1) {
    const d = this.dirty;
    if (!d) this.dirty = { i0, i1, j0, j1 };
    else { d.i0 = Math.min(d.i0, i0); d.i1 = Math.max(d.i1, i1); d.j0 = Math.min(d.j0, j0); d.j1 = Math.max(d.j1, j1); }
  }
  takeDirty() { const d = this.dirty; this.dirty = null; return d; }

  // a, b: [x,y,z] tool tip; returns { removed, shank } (max depth removed / shank interference)
  cutSegment(a, b, move) {
    const tool = this.tools[move.tool];
    if (!tool) return null;
    const r = tool.diameter / 2;
    const prof = this.profile(tool);
    const minZ = Math.min(a[2], b[2]);
    if (minZ >= this.top + (prof(r) > 0 ? 0 : 0)) return null; // entirely above the stock
    const { nx, ny, cx, cy, minX, minY, heights, toolIdx } = this;
    const i0 = Math.max(0, Math.floor((Math.min(a[0], b[0]) - r - minX) / cx));
    const i1 = Math.min(nx - 1, Math.ceil((Math.max(a[0], b[0]) + r - minX) / cx));
    const j0 = Math.max(0, Math.floor((Math.min(a[1], b[1]) - r - minY) / cy));
    const j1 = Math.min(ny - 1, Math.ceil((Math.max(a[1], b[1]) + r - minY) / cy));
    if (i0 > i1 || j0 > j1) return null;
    const dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
    const L2 = dx * dx + dy * dy;
    const r2 = r * r;
    const flatBottom = ['flat', 'face', 'tap', 'threadmill'].includes(tool.type) || (tool.type === 'bull' && !(tool.cornerRadius > 0));
    const shankTop = tool.fluteLength || 1e9;
    const tnum = Math.min(255, tool.num | 0);
    const throughZ = this.bottom - 1e-3;
    let removed = 0, shank = 0, changed = false;
    let ci0 = nx, ci1 = -1, cj0 = ny, cj1 = -1;
    for (let j = j0; j <= j1; j++) {
      const y = minY + j * cy;
      const row = j * nx;
      for (let i = i0; i <= i1; i++) {
        const x = minX + i * cx;
        // zt = tool tip height, zc = height of the cutting surface over this node
        let zt, zc;
        if (L2 < 1e-12) {
          const ex = x - a[0], ey = y - a[1];
          const d2 = ex * ex + ey * ey;
          if (d2 > r2) continue;
          zt = minZ;
          zc = zt + prof(Math.sqrt(d2));
        } else {
          const t = ((x - a[0]) * dx + (y - a[1]) * dy) / L2;
          const px = a[0] + t * dx - x, py = a[1] + t * dy - y;
          const perp2 = px * px + py * py;
          if (perp2 > r2) continue;
          const h = Math.sqrt((r2 - perp2) / L2);
          const tlo = Math.max(0, t - h), thi = Math.min(1, t + h);
          if (tlo > thi) continue;
          // deepest cutting surface over the reach: at the closest point (where a ball or
          // cone reaches lowest) or at the reach end where a ramping tip is lowest
          if (flatBottom) {
            // flat end: only the lowest tip position in reach matters
            zt = zc = a[2] + (dz > 0 ? tlo : thi) * dz;
          } else {
            const tc = Math.max(tlo, Math.min(thi, t));
            zc = Infinity;
            for (const tt of dz === 0 ? [tc] : [tc, dz > 0 ? tlo : thi]) {
              const qx = a[0] + tt * dx - x, qy = a[1] + tt * dy - y;
              const tip = a[2] + tt * dz;
              const z = tip + prof(Math.min(r, Math.sqrt(qx * qx + qy * qy)));
              if (z < zc) { zc = z; zt = tip; }
            }
          }
        }
        if (zc < this.bottom) zc = throughZ; // below the stock: nothing more to remove there
        const k = row + i;
        const hOld = heights[k];
        if (zc < hOld - 1e-9) {
          if (hOld - zt > shankTop + 0.01 && hOld - zc > 0.01) shank = Math.max(shank, hOld - zt - shankTop);
          if (hOld - zc > removed) removed = hOld - zc;
          heights[k] = zc;
          toolIdx[k] = tnum;
          changed = true;
          if (i < ci0) ci0 = i; if (i > ci1) ci1 = i;
          if (j < cj0) cj0 = j; if (j > cj1) cj1 = j;
        }
      }
    }
    if (changed) this.markDirty(ci0, ci1, cj0, cj1);
    // shank / holder against the material that is left (only when the cut is deep enough to matter)
    let holder = 0;
    const sr = (tool.shankDiameter || tool.diameter) / 2;
    if (sr > r + 0.01 && minZ + tool.fluteLength < this.top) shank = Math.max(shank, this.interference(a, b, sr, tool.fluteLength));
    if (tool.overallLength && minZ + tool.overallLength < this.top) holder = this.interference(a, b, (tool.holderDiameter || 32) / 2, tool.overallLength);
    return { removed, shank, holder };
  }

  // how far material reaches into a cylinder of radius R whose bottom is h above the tool tip
  interference(a, b, R, h) {
    const { nx, ny, cx, cy, minX, minY, heights } = this;
    const i0 = Math.max(0, Math.floor((Math.min(a[0], b[0]) - R - minX) / cx));
    const i1 = Math.min(nx - 1, Math.ceil((Math.max(a[0], b[0]) + R - minX) / cx));
    const j0 = Math.max(0, Math.floor((Math.min(a[1], b[1]) - R - minY) / cy));
    const j1 = Math.min(ny - 1, Math.ceil((Math.max(a[1], b[1]) + R - minY) / cy));
    const dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
    const L2 = dx * dx + dy * dy, R2 = R * R;
    const zLow = Math.min(a[2], b[2]);
    let worst = 0;
    for (let j = j0; j <= j1; j++) {
      const y = minY + j * cy;
      for (let i = i0; i <= i1; i++) {
        const x = minX + i * cx;
        let t = L2 < 1e-12 ? 0 : ((x - a[0]) * dx + (y - a[1]) * dy) / L2;
        t = Math.max(0, Math.min(1, t));
        const px = a[0] + t * dx - x, py = a[1] + t * dy - y;
        if (px * px + py * py > R2) continue;
        const bottom = (L2 < 1e-12 ? zLow : a[2] + t * dz) + h;
        const over = heights[j * nx + i] - bottom;
        if (over > worst) worst = over;
      }
    }
    return worst;
  }

  removedVolume() {
    let v = 0;
    const area = this.cx * this.cy;
    for (let k = 0; k < this.heights.length; k++) v += this.top - Math.max(this.bottom, this.heights[k]);
    return v * area;
  }
}
