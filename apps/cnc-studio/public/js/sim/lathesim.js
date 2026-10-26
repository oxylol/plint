// 2-axis turning: the part is a revolved profile (outer and inner radius per Z sample).
// Tools are polygons (cutter + holder) in (dz, dx) around their reference point; see
// gcode/lathetools.js. A straight move sweeps the cutter as the convex hull of its start
// and end placement, threads are stamped once per pitch so the crests show, and the
// holder is checked against the remaining material.

import { convexHull } from '../gcode/lathetools.js';

const EPS = 1e-6;

// radial coverage of a (possibly axis-crossing) interval
function radial(lo, hi) {
  if (hi <= 0) return [-hi, -lo];
  if (lo < 0) return [0, Math.max(-lo, hi)];
  return [lo, hi];
}

// all [lo, hi] intervals of a polygon on the vertical line z (even-odd)
function polyIntervals(poly, z) {
  const xs = [];
  for (let i = 0, n = poly.length; i < n; i++) {
    const [z0, x0] = poly[i], [z1, x1] = poly[(i + 1) % n];
    if ((z0 <= z && z < z1) || (z1 <= z && z < z0)) xs.push(x0 + ((z - z0) * (x1 - x0)) / (z1 - z0));
  }
  xs.sort((a, b) => a - b);
  const out = [];
  for (let i = 0; i + 1 < xs.length; i += 2) out.push([xs[i], xs[i + 1]]);
  return out;
}

export class LatheSim {
  // stock: { diameter, length (out of the chuck), faceStock, innerDiameter, grip (inside the jaws) }
  constructor(stock, tools) {
    this.stock = stock;
    this.tools = tools;
    this.kind = 'lathe';
    if (!stock) return; // fromProfile fills the rest
    this.r0 = stock.diameter / 2;
    this.ri0 = (stock.innerDiameter || 0) / 2;
    this.grip = stock.grip ?? 15;
    this.zMax = stock.faceStock || 0;
    this.zChuck = -stock.length;
    this.zMin = this.zChuck - this.grip; // the bar continues into the chuck
    this.alloc();
    this.reset();
  }

  alloc() {
    const len = this.zMax - this.zMin;
    this.dz = Math.min(0.05, len / 1500);
    this.n = Math.max(2, Math.round(len / this.dz) + 1);
    this.dz = len / (this.n - 1);
    this.outer = new Float32Array(this.n);
    this.inner = new Float32Array(this.n);
    this.toolOuter = new Uint8Array(this.n);
    this.toolInner = new Uint8Array(this.n);
  }

  // part from a previous lathe operation, re-chucked. X maps the source program's
  // coordinates to this one's (only a flip about X and a Z shift make sense on a lathe).
  static fromProfile(src, X, grip = 15, tools = src.tools) {
    const s = new LatheSim(null, tools);
    const sz = X.R[8] < 0 ? -1 : 1, tz = X.t[2];
    // the new bar spans the material only (air faced off earlier is not part of it)
    const mb = src.materialBounds() || { min: [0, 0, src.zMin], max: [0, 0, src.zMax] };
    const a = sz * mb.min[2] + tz, b = sz * mb.max[2] + tz;
    s.zMin = Math.min(a, b);
    s.zMax = Math.max(a, b);
    s.grip = grip;
    s.zChuck = s.zMin + grip;
    s.alloc();
    for (let i = 0; i < s.n; i++) {
      const zp = (s.zMin + i * s.dz - tz) / sz;
      const j = Math.max(0, Math.min(src.n - 1, Math.round((zp - src.zMin) / src.dz)));
      s.outer[i] = src.outer[j];
      s.inner[i] = src.inner[j];
      s.toolOuter[i] = src.toolOuter[j];
      s.toolInner[i] = src.toolInner[j];
    }
    let R = 0;
    for (let i = 0; i < s.n; i++) R = Math.max(R, s.outer[i]);
    s.r0 = R;
    s.ri0 = 0;
    s.stock = { kind: 'lathe', diameter: 2 * R, length: s.zMax - s.zChuck - Math.max(0, s.zMax), faceStock: Math.max(0, s.zMax), innerDiameter: 0, grip };
    s.pristine = { outer: s.outer.slice(), inner: s.inner.slice(), toolOuter: s.toolOuter.slice(), toolInner: s.toolInner.slice() };
    s.dirty = true;
    s.partedOff = null;
    s.version = 1;
    return s;
  }

  clone() {
    const c = Object.create(LatheSim.prototype);
    Object.assign(c, this);
    for (const k of ['outer', 'inner', 'toolOuter', 'toolInner']) c[k] = this[k].slice();
    return c;
  }

  reset() {
    if (this.pristine) {
      this.outer.set(this.pristine.outer);
      this.inner.set(this.pristine.inner);
      this.toolOuter.set(this.pristine.toolOuter);
      this.toolInner.set(this.pristine.toolInner);
    } else {
      this.outer.fill(this.r0);
      this.inner.fill(this.ri0);
      this.toolOuter.fill(0);
      this.toolInner.fill(0);
    }
    this.dirty = true;
    this.partedOff = null;
    this.version = (this.version || 0) + 1;
  }

  // part-frame bounds for hand-off to other operations (lathe axis = Z)
  materialBounds() {
    let R = 0, z0 = Infinity, z1 = -Infinity;
    for (let i = 0; i < this.n; i++) {
      if (this.outer[i] - this.inner[i] < 1e-4) continue;
      R = Math.max(R, this.outer[i]);
      const z = this.zMin + i * this.dz;
      z0 = Math.min(z0, z); z1 = Math.max(z1, z);
    }
    if (!(R > 0)) return null;
    return { min: [-R, -R, z0], max: [R, R, z1] };
  }

  sample(x, y, z) {
    const i = Math.round((z - this.zMin) / this.dz);
    if (i < 0 || i >= this.n) return 1;
    const r = Math.hypot(x, y);
    return Math.max(r - this.outer[i], this.inner[i] - r);
  }
  isMaterial(x, y, z) { return this.sample(x, y, z) < 0; }

  takeDirty() { const d = this.dirty; this.dirty = false; return d; }

  // reference point of the tool for a programmed position (G41/G42 paths are nose centres)
  refPoint(tool, p, isCenter) {
    if (isCenter && tool.noseOffset) return [p[2] - tool.noseOffset[0], p[0] - tool.noseOffset[1]];
    return [p[2], p[0]];
  }

  cutSegment(a, b, move) {
    const tool = this.tools[move.tool];
    if (!tool || !tool.cutter || tool.cutter.length < 3) return null;
    const [za, ra] = this.refPoint(tool, a, move.center);
    const [zb, rb] = this.refPoint(tool, b, move.center);
    const tnum = Math.min(255, tool.num | 0);
    const bodyUp = tool.type !== 'bore';
    this.cutThrough = -1;
    let removed = 0;
    if (move.thread > 0 && Math.abs(zb - za) > EPS) {
      // one stamp per pitch, phase anchored at the pass start so every pass lines up
      const lead = move.thread;
      const z0 = move.pts[0][2];
      const dir = Math.sign(zb - za);
      let k0 = Math.ceil((dir > 0 ? za - z0 : z0 - za) / lead - 1e-9);
      for (let k = Math.max(0, k0); k < 100000; k++) {
        const z = z0 + dir * k * lead;
        if (dir > 0 ? z > zb + EPS : z < zb - EPS) break;
        const t = (z - za) / (zb - za);
        const r = ra + (rb - ra) * t;
        removed = Math.max(removed, this.stampHull(convexHull(tool.cutter.map(([dz, dx]) => [z + dz, r + dx])), tnum, bodyUp));
      }
    } else {
      const pts = [];
      for (const [dz, dx] of tool.cutter) { pts.push([za + dz, ra + dx]); pts.push([zb + dz, rb + dx]); }
      removed = this.stampHull(convexHull(pts), tnum, bodyUp);
    }
    if (this.cutThrough >= 0) this.dropFreeEnd();
    if (removed > 0) this.dirty = true;
    // chuck: anything below the chuck face (other than a drill on the axis)
    const zLow = Math.min(za, zb) + Math.min(...tool.cutter.map((p) => p[0]));
    const chuckR = Math.max(this.r0 * 1.9, this.r0 + 35); // same chuck the viewer draws
    const chuck = zLow < this.zChuck - 0.01 && Math.min(ra, rb) < chuckR && tool.type !== 'drill';
    const holder = tool.holder && tool.holder.length >= 3 ? this.holderInterference(tool, za, ra, zb, rb) : 0;
    return { removed, chuck, holder };
  }

  stampHull(hull, tnum, bodyUp) {
    const { lower, upper } = hull;
    if (!lower || lower.length < 2) return 0;
    const { zMin, dz, n } = this;
    const z0 = lower[0][0], z1 = lower[lower.length - 1][0];
    // a column exactly on the outline's end only touches it (a face the tool's side is flush with)
    let i0 = Math.max(0, Math.ceil((z0 + 1e-3 - zMin) / dz)), i1 = Math.min(n - 1, Math.floor((z1 - 1e-3 - zMin) / dz));
    if (i0 > i1) return 0;
    let li = 0, ui = 0, removed = 0;
    for (let i = i0; i <= i1; i++) {
      const z = zMin + i * dz;
      while (li < lower.length - 2 && lower[li + 1][0] < z) li++;
      while (ui < upper.length - 2 && upper[ui + 1][0] < z) ui++;
      const lerp = (c, j) => {
        const [za, xa] = c[j], [zb, xb] = c[j + 1];
        return zb - za < 1e-12 ? Math.min(xa, xb) : xa + ((z - za) * (xb - xa)) / (zb - za);
      };
      const lo = lerp(lower, li), hi = lerp(upper, ui);
      const r = this.applyInterval(i, Math.min(lo, hi), Math.max(lo, hi), tnum, bodyUp);
      if (r > removed) removed = r;
    }
    return removed;
  }

  applyInterval(i, lo, hi, tnum, bodyUp) {
    const [a, b] = radial(lo, hi);
    const inr = this.inner[i], out = this.outer[i];
    if (b <= inr + EPS || a >= out - EPS) return 0;
    let removed;
    if (b >= out - EPS) {
      // cut right through: an empty column collapses to the axis (nothing left to draw)
      if (a <= inr + EPS) { removed = out - inr; this.outer[i] = 0; this.inner[i] = 0; this.cutThrough = Math.max(this.cutThrough, i); }
      else { removed = out - a; this.outer[i] = a; }
      this.toolOuter[i] = tnum;
    } else if (a <= inr + EPS) {
      removed = b - inr;
      this.inner[i] = b;
      this.toolInner[i] = tnum;
    } else if (bodyUp) {
      removed = out - a;
      this.outer[i] = a;
      this.toolOuter[i] = tnum;
    } else {
      removed = b - inr;
      this.inner[i] = b;
      this.toolInner[i] = tnum;
    }
    return removed;
  }

  // a column cut right through separates everything on the tailstock side of it
  dropFreeEnd() {
    const { outer, inner, n } = this;
    let cut = -1;
    for (let i = this.cutThrough; i >= 0; i--) if (outer[i] - inner[i] < 1e-4) { cut = i; break; }
    if (cut < 0 || cut >= n - 1) return;
    let any = false;
    for (let j = cut + 1; j < n; j++) {
      if (outer[j] - inner[j] > 1e-4) any = true;
      outer[j] = 0;
      inner[j] = 0;
    }
    if (any) {
      this.partedOff = this.zMin + cut * this.dz;
      this.dirty = true;
    }
  }

  // max radial overlap of the holder with remaining material along the move (mm)
  holderInterference(tool, za, ra, zb, rb) {
    const len = Math.hypot(zb - za, rb - ra);
    const steps = Math.max(1, Math.ceil(len / 2));
    let worst = 0;
    for (let s = 0; s <= steps; s++) {
      const t = s / steps;
      const z = za + (zb - za) * t, r = ra + (rb - ra) * t;
      worst = Math.max(worst, this.holderAt(tool, z, r));
    }
    return worst;
  }

  holderAt(tool, z, r) {
    const { zMin, dz, n, outer, inner } = this;
    let hz0 = Infinity, hz1 = -Infinity;
    for (const [hz] of tool.holder) { if (hz < hz0) hz0 = hz; if (hz > hz1) hz1 = hz; }
    const za = Math.max(z + hz0, zMin), zb = Math.min(z + hz1, this.zMax);
    if (za > zb) return 0;
    let worst = 0;
    const step = Math.max(dz, 0.5);
    for (let zz = za; zz <= zb + 1e-9; zz += step) {
      const i = Math.round((zz - zMin) / dz);
      if (i < 0 || i >= n || outer[i] - inner[i] < 1e-4) continue;
      const hs = polyIntervals(tool.holder, zz - z);
      if (!hs.length) continue;
      const cs = polyIntervals(tool.cutter, zz - z);
      for (const [hlo, hhi] of hs) {
        const [a, b] = radial(hlo + r, hhi + r);
        let ov = Math.min(b, outer[i]) - Math.max(a, inner[i]);
        if (ov <= 0.02) continue;
        // ignore the part of the holder that sits on the insert itself
        for (const [clo, chi] of cs) {
          const [ca, cb] = radial(clo + r, chi + r);
          ov -= Math.max(0, Math.min(b, outer[i], cb) - Math.max(a, inner[i], ca));
        }
        if (ov > worst) worst = ov;
      }
    }
    return worst;
  }

  removedVolume() {
    let v = 0;
    for (let i = 0; i < this.n; i++) {
      const o0 = this.pristine ? this.pristine.outer[i] : this.r0, i0 = this.pristine ? this.pristine.inner[i] : this.ri0;
      v += (o0 * o0 - i0 * i0) - (this.outer[i] * this.outer[i] - this.inner[i] * this.inner[i]);
    }
    return Math.PI * v * this.dz;
  }

  materialVolume() {
    let v = 0;
    for (let i = 0; i < this.n; i++) v += this.outer[i] * this.outer[i] - this.inner[i] * this.inner[i];
    return Math.PI * v * this.dz;
  }

  // simplified outline for rendering: [[r, z, toolNum], ...] outer from +Z to -Z then inner back
  outline(maxPts = 900) {
    const { outer, inner, n, zMin, dz } = this;
    const simplify = (arr, tools) => {
      const pts = [];
      for (let i = n - 1; i >= 0; i--) {
        const r = arr[i];
        if (pts.length >= 2) {
          const p = pts[pts.length - 1], q = pts[pts.length - 2];
          if (Math.abs(p[0] - r) < 1e-4 && Math.abs(q[0] - r) < 1e-4 && p[2] === tools[i]) { p[1] = zMin + i * dz; continue; }
        }
        pts.push([r, zMin + i * dz, tools[i]]);
      }
      return pts;
    };
    let o = simplify(outer, this.toolOuter);
    let inn = simplify(inner, this.toolInner).reverse();
    const thin = (pts) => {
      if (pts.length <= maxPts) return pts;
      const k = Math.ceil(pts.length / maxPts);
      return pts.filter((_, i) => i % k === 0 || i === pts.length - 1);
    };
    o = thin(o); inn = thin(inn);
    return { outer: o, inner: inn };
  }
}
