// Touch-probe geometry against the simulated material: where the ball touches, what the stylus
// or body hits. Pure, shared by the interpreter's oracle and the simulation runner.
import { invertX, applyX } from './voxelsim.js';

// 26 directions on the sphere: axes, edges, corners
const DIRS = [];
for (const x of [-1, 0, 1]) for (const y of [-1, 0, 1]) for (const z of [-1, 0, 1]) {
  if (!x && !y && !z) continue;
  const n = Math.hypot(x, y, z);
  DIRS.push([x / n, y / n, z / n]);
}
const RING = Array.from({ length: 12 }, (_, i) => [Math.cos((i * Math.PI) / 6), Math.sin((i * Math.PI) / 6)]);

// is there material at a work point?
export function materialTest(sim, X) {
  if (sim.kind === 'voxel') {
    const inv = invertX(X || sim.xform || { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [0, 0, 0] });
    return (p) => { const q = applyX(inv, p); return sim.sample(q[0], q[1], q[2]) < 0; };
  }
  const { nx, ny, cx, cy, minX, minY, heights, bottom } = sim;
  return (p) => {
    const u = (p[0] - minX) / cx, v = (p[1] - minY) / cy;
    if (u < 0 || v < 0 || u > nx - 1 || v > ny - 1) return false;
    const i = Math.min(nx - 2, Math.floor(u)), j = Math.min(ny - 2, Math.floor(v));
    const a = u - i, b = v - j, k = j * nx + i;
    const h = (q) => Math.max(heights[q], bottom - 1e-3);
    const top = (h(k) * (1 - a) + h(k + 1) * a) * (1 - b) + (h(k + nx) * (1 - a) + h(k + nx + 1) * a) * b;
    return top >= bottom && p[2] <= top && p[2] >= bottom - 1e-6;
  };
}

// the probe's geometry at ball centre c: 'ball' (the ball touches), 'stylus', 'body', or null
export function probeContact(inside, c, tool) {
  const r = tool.diameter / 2;
  const tip = c[2] - r;
  const rs = (tool.shankDiameter || r) / 2, Rb = (tool.holderDiameter || 40) / 2;
  const zBody = tip + (tool.overallLength || 50);
  let hit = null;
  if (inside(c)) return 'ball';
  for (const d of DIRS) if (inside([c[0] + d[0] * r, c[1] + d[1] * r, c[2] + d[2] * r])) { hit = 'ball'; break; }
  // stylus: rings from above the ball up to the body
  for (let z = c[2] + r; z < zBody - 1e-6; z += Math.max(2, r)) for (const [a, b] of RING) if (inside([c[0] + a * rs, c[1] + b * rs, z])) return 'stylus';
  // body: its bottom face (rings at three radii) and its sides up to 30 mm
  for (const rr of [0, Rb * 0.5, Rb]) for (const [a, b] of RING) if (inside([c[0] + a * rr, c[1] + b * rr, zBody + 0.02])) return 'body';
  for (let z = zBody + 5; z <= zBody + 30; z += 5) for (const [a, b] of RING) if (inside([c[0] + a * Rb, c[1] + b * Rb, z])) return 'body';
  return hit;
}

// march the ball centre from a to b: where does the probe first touch (or the stylus/body hit)?
export function probeAlong(sim, a, b, tool, X) {
  const inside = materialTest(sim, X);
  const r = tool.diameter / 2;
  const L = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
  const at = (t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  const first = probeContact(inside, a, tool);
  if (first) return { t: 0, point: a.slice(), hit: first };
  if (L < 1e-9) return { t: 1, point: b.slice(), hit: null };
  const n = Math.max(1, Math.ceil(L / Math.max(0.05, Math.min(0.3, r / 4))));
  let prevT = 0;
  for (let i = 1; i <= n; i++) {
    const t = i / n;
    const hit = probeContact(inside, at(t), tool);
    if (!hit) { prevT = t; continue; }
    if (hit !== 'ball') return { t, point: at(t), hit };
    // the ball: pin the touch down between the last free step and this one
    let lo = prevT, hi = t;
    for (let k = 0; k < 10; k++) {
      const mid = (lo + hi) / 2;
      if (probeContact(inside, at(mid), tool) === 'ball') hi = mid; else lo = mid;
    }
    const p = at(hi);
    const body = probeContact(inside, p, tool);
    return { t: hi, point: p, hit: body === 'stylus' || body === 'body' ? body : 'ball' };
  }
  return { t: 1, point: b.slice(), hit: null };
}

