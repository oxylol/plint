// Geometry facts about a simulated part, as text for Claude (instead of screenshots).
//
// Everything works on a "material field": inside(p) for a point in the operation's work
// frame. Features come from depth grids seen along an axis (top, bottom, front …) and are
// then measured precisely with rays at mid-depth (bisection on the material boundary).
import { applyX, invertX } from './voxelsim.js';

const fx = (v, d = 3) => { const s = (Math.abs(v) < 0.5 * 10 ** -d ? 0 : v).toFixed(d); return s.includes('.') ? s.replace(/\.?0+$/, '') || '0' : s; };
// measurements are printed to what the engine resolves: 0.01 mm on the mill models, 0.001 on the lathe profile
let PREC = 3;
const usePrecision = (kind) => { PREC = kind === 'lathe' ? 3 : 2; };
const f3 = (v) => fx(v, PREC);
const f2 = (v) => fx(v, 2);

// ---------------------------------------------------------------- material fields
export function materialField(out) {
  const sim = out.final;
  if (out.engine === 'voxel') {
    const inv = invertX(out.X);
    const R = inv.R, t = inv.t;
    const at = (p) => sim.sample(R[0] * p[0] + R[1] * p[1] + R[2] * p[2] + t[0], R[3] * p[0] + R[4] * p[1] + R[5] * p[2] + t[1], R[6] * p[0] + R[7] * p[1] + R[8] * p[2] + t[2]);
    const mb = sim.materialBounds();
    return {
      kind: 'voxel', step: sim.h,
      inside: (p) => at(p) < 0,
      distance: at,
      bounds: mb ? boundsUnder(out.X, mb) : null,
    };
  }
  if (out.engine === 'height') {
    const { nx, ny, cx, cy, minX, minY, heights, bottom } = sim;
    const hAt = (x, y) => {
      const u = (x - minX) / cx, v = (y - minY) / cy;
      if (u < 0 || v < 0 || u > nx - 1 || v > ny - 1) return -Infinity;
      const i = Math.min(nx - 2, Math.floor(u)), j = Math.min(ny - 2, Math.floor(v));
      const a = u - i, b = v - j, k = j * nx + i;
      const h = (q) => Math.max(heights[q], bottom - 1e-3);
      return (h(k) * (1 - a) + h(k + 1) * a) * (1 - b) + (h(k + nx) * (1 - a) + h(k + nx + 1) * a) * b;
    };
    let top = -Infinity;
    for (let k = 0; k < heights.length; k++) if (heights[k] > top) top = heights[k];
    return {
      kind: 'height', step: Math.min(cx, cy), hAt,
      inside: (p) => { const h = hAt(p[0], p[1]); return h >= bottom && p[2] <= h && p[2] >= bottom; },
      bounds: { min: [minX, minY, bottom], max: [minX + (nx - 1) * cx, minY + (ny - 1) * cy, Math.min(sim.top, top)] },
    };
  }
  // lathe: work frame Z = spindle axis, radius = hypot(x, y)
  const s = sim;
  const at = (arr, z) => {
    const u = (z - s.zMin) / s.dz;
    if (u < 0 || u > s.n - 1) return null;
    const i = Math.min(s.n - 2, Math.floor(u)), a = u - i;
    return arr[i] * (1 - a) + arr[i + 1] * a;
  };
  const mb = s.materialBounds();
  return {
    kind: 'lathe', step: s.dz,
    inside: (p) => {
      const o = at(s.outer, p[2]);
      if (o === null) return false;
      const r = Math.hypot(p[0], p[1]);
      return r <= o && r >= at(s.inner, p[2]) && o > 1e-6;
    },
    bounds: mb,
  };
}

export function boundsUnder(X, b) {
  const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  for (const cx of [b.min[0], b.max[0]]) for (const cy of [b.min[1], b.max[1]]) for (const cz of [b.min[2], b.max[2]]) {
    const p = applyX(X, [cx, cy, cz]);
    for (let k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], p[k]); mx[k] = Math.max(mx[k], p[k]); }
  }
  return { min: mn, max: mx };
}

const add = (p, d, t) => [p[0] + d[0] * t, p[1] + d[1] * t, p[2] + d[2] * t];

// distance along dir to where inside() changes from `from` (bisection), or null within maxT
function rayTo(field, p0, dir, maxT, step, from = false) {
  let a = 0;
  for (let t = step; t <= maxT + step * 0.5; t += step) {
    const tt = Math.min(t, maxT);
    if (field.inside(add(p0, dir, tt)) !== from) {
      let lo = a, hi = tt;
      while (hi - lo > 2e-4) { const m = (lo + hi) / 2; if (field.inside(add(p0, dir, m)) !== from) hi = m; else lo = m; }
      return (lo + hi) / 2;
    }
    a = tt;
  }
  return null;
}

// ---------------------------------------------------------------- views (depth grids)
// ray = viewing direction; u, v = in-plane axes; d = depth axis
export const VIEWS = {
  top: { ray: [0, 0, -1], u: 0, v: 1, d: 2, label: 'from above (looking down −Z)' },
  bottom: { ray: [0, 0, 1], u: 0, v: 1, d: 2, label: 'from below (looking up +Z)' },
  front: { ray: [0, 1, 0], u: 0, v: 2, d: 1, label: 'from the front (looking +Y)' },
  back: { ray: [0, -1, 0], u: 0, v: 2, d: 1, label: 'from the back (looking −Y)' },
  left: { ray: [1, 0, 0], u: 1, v: 2, d: 0, label: 'from the left (looking +X)' },
  right: { ray: [-1, 0, 0], u: 1, v: 2, d: 0, label: 'from the right (looking −X)' },
};
const AX = 'XYZ';

export function depthGrid(field, viewName, maxCells = 200) {
  const view = VIEWS[viewName];
  const b = field.bounds;
  const su = b.max[view.u] - b.min[view.u], sv = b.max[view.v] - b.min[view.v];
  const g = Math.max(field.step * 0.9, Math.max(su, sv) / maxCells, 0.05);
  const nu = Math.max(2, Math.ceil(su / g) + 1), nv = Math.max(2, Math.ceil(sv / g) + 1);
  const gu = su / (nu - 1), gv = sv / (nv - 1);
  const sign = view.ray[view.d]; // -1: starting at max, +1: at min
  const start = sign < 0 ? b.max[view.d] + 0.5 : b.min[view.d] - 0.5;
  const span = b.max[view.d] - b.min[view.d] + 1;
  const t = new Float32Array(nu * nv).fill(NaN);
  const march = Math.max(field.step * 0.6, 0.02);
  for (let j = 0; j < nv; j++) {
    for (let i = 0; i < nu; i++) {
      const p = [0, 0, 0];
      p[view.u] = b.min[view.u] + i * gu;
      p[view.v] = b.min[view.v] + j * gv;
      p[view.d] = start;
      if (field.kind === 'height' && viewName === 'top') {
        const h = field.hAt(p[0], p[1]);
        if (h >= b.min[2]) t[j * nu + i] = start - h;
        continue;
      }
      const hit = rayTo(field, p, view.ray, span, march);
      if (hit !== null) t[j * nu + i] = hit;
    }
  }
  // coordinate of the surface along the depth axis = start + sign * t
  return { view: viewName, t, nu, nv, gu, gv, u0: b.min[view.u], v0: b.min[view.v], start, sign, bounds: b };
}

// ---------------------------------------------------------------- features of one view
function clusterLevels(t, tol) {
  const counts = new Map();
  for (const v of t) if (!Number.isNaN(v)) { const k = Math.round(v / 0.01); counts.set(k, (counts.get(k) || 0) + 1); }
  const keys = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  const centers = [];
  for (const [k, c] of keys) {
    const v = k * 0.01;
    const near = centers.find((cl) => Math.abs(cl.v - v) <= tol);
    if (near) { near.sum += v * c; near.n += c; } else centers.push({ v, sum: v * c, n: c });
  }
  for (const c of centers) c.v = c.sum / c.n;
  return centers.sort((a, b) => a.v - b.v);
}

function components(label, nu, nv, want) {
  const seen = new Uint8Array(nu * nv);
  const comps = [];
  for (let s = 0; s < label.length; s++) {
    if (seen[s] || !want(label[s])) continue;
    const L = label[s];
    const cells = [];
    const stack = [s];
    seen[s] = 1;
    while (stack.length) {
      const q = stack.pop();
      cells.push(q);
      const i = q % nu, j = (q - i) / nu;
      for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const a = i + di, bb = j + dj;
        if (a < 0 || bb < 0 || a >= nu || bb >= nv) continue;
        const r = bb * nu + a;
        if (!seen[r] && label[r] === L) { seen[r] = 1; stack.push(r); }
      }
    }
    comps.push({ label: L, cells });
  }
  return comps;
}

function fitCircle(pts) {
  // Kasa: x² + y² + D x + E y + F = 0
  let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, sxz = 0, syz = 0, sz = 0;
  const n = pts.length;
  for (const [x, y] of pts) { const z = x * x + y * y; sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y; sxz += x * z; syz += y * z; sz += z; }
  const A = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n]], B = [-sxz, -syz, -sz];
  const det = (m) => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const d = det(A);
  if (Math.abs(d) < 1e-12) return null;
  const sol = [0, 1, 2].map((c) => det(A.map((row, r) => row.map((v, k) => (k === c ? B[r] : v)))) / d);
  const cx = -sol[0] / 2, cy = -sol[1] / 2;
  const r = Math.sqrt(Math.max(0, cx * cx + cy * cy - sol[2]));
  let e = 0;
  for (const [x, y] of pts) e += (Math.hypot(x - cx, y - cy) - r) ** 2;
  return { cx, cy, r, rms: Math.sqrt(e / n) };
}

// distance from the centre of a rounded rectangle (half sizes a, b, corner radius rc) to its edge along angle th
function roundedRectDist(a, b, rc, th) {
  const ux = Math.abs(Math.cos(th)), uy = Math.abs(Math.sin(th));
  const tx = ux > 1e-12 ? a / ux : Infinity, ty = uy > 1e-12 ? b / uy : Infinity;
  const t = Math.min(tx, ty);
  const px = ux * t, py = uy * t;
  if (rc <= 0 || px <= a - rc || py <= b - rc) return t;
  const cx = a - rc, cy = b - rc;
  const ud = ux * cx + uy * cy;
  return ud + Math.sqrt(Math.max(0, ud * ud - (cx * cx + cy * cy) + rc * rc));
}

// outline of a feature in its plane at depth coordinate dval: 32 rays plus one toward each
// corner of what they found (so corner radii are measured). fromOutside: rays come in from
// radius R around (cu, cv) (islands, the part outline), else they go out from (cu, cv) (pockets, holes).
function outlinePts(field, view, cu, cv, dval, R, fromOutside, N = 32) {
  const step = Math.max(field.step * 0.5, 0.02);
  const shoot = (th) => {
    const c = Math.cos(th), sn = Math.sin(th);
    const p = [0, 0, 0], dir = [0, 0, 0];
    p[view.d] = dval;
    if (fromOutside) { p[view.u] = cu + c * R; p[view.v] = cv + sn * R; dir[view.u] = -c; dir[view.v] = -sn; if (field.inside(p)) return null; }
    else { p[view.u] = cu; p[view.v] = cv; dir[view.u] = c; dir[view.v] = sn; }
    const t = rayTo(field, p, dir, R, step, false);
    return t === null ? null : [p[view.u] + dir[view.u] * t, p[view.v] + dir[view.v] * t];
  };
  const pts = [];
  for (let k = 0; k < N; k++) { const q = shoot((k / N) * 2 * Math.PI); if (!q) return null; pts.push(q); }
  let u0 = Infinity, u1 = -Infinity, v0 = Infinity, v1 = -Infinity;
  for (const [u, v] of pts) { u0 = Math.min(u0, u); u1 = Math.max(u1, u); v0 = Math.min(v0, v); v1 = Math.max(v1, v); }
  for (const [u, v] of [[u0, v0], [u1, v0], [u1, v1], [u0, v1]]) { const q = shoot(Math.atan2(v - cv, u - cu)); if (q) pts.push(q); }
  return { pts };
}

// diameter of a round hole at depth coordinate dval (null if not round / centre in material)
function diameterAt(field, view, cu, cv, dval, R) {
  const p = [0, 0, 0];
  p[view.u] = cu; p[view.v] = cv; p[view.d] = dval;
  if (field.inside(p)) return null;
  const m = outlinePts(field, view, cu, cv, dval, R, false, 16);
  if (!m) return null;
  const c = fitCircle(m.pts);
  return c && c.rms < Math.max(0.05, c.r * 0.02) ? 2 * c.r : null;
}

// how a round hole's diameter changes with depth: [{ from, to, d0, d1 }] in depth coordinates
function holeProfile(field, view, cu, cv, coord, tFrom, tTo, R) {
  const N = 28;
  const smp = [];
  for (let k = 0; k <= N; k++) {
    const t = tFrom + ((tTo - tFrom) * (k + 0.5)) / (N + 1);
    smp.push({ t, d: diameterAt(field, view, cu, cv, coord(t), R) });
  }
  const runs = [];
  for (const sm of smp) {
    const r = runs[runs.length - 1];
    if (r && sm.d !== null && r.d1 !== null && Math.abs(sm.d - r.d1) < 0.06 && (r.kind === 'flat' || r.n === 1)) { r.t1 = sm.t; r.n++; r.sum += sm.d; r.d1 = sm.d; r.kind = 'flat'; continue; }
    // a steadily changing diameter is a chamfer / countersink / cone
    if (r && sm.d !== null && r.d1 !== null && r.n >= 1 && (r.kind === 'taper' || r.n === 1) && Math.abs(sm.d - r.d1) < 1.5 && (r.kind !== 'taper' || Math.sign(sm.d - r.d1) === r.dir)) {
      r.kind = 'taper'; r.dir = Math.sign(sm.d - r.d1); r.t1 = sm.t; r.d1 = sm.d; r.n++; continue;
    }
    runs.push({ t0: sm.t, t1: sm.t, d0: sm.d, d1: sm.d, sum: sm.d ?? 0, n: 1, kind: 'flat' });
  }
  return runs.filter((r) => r.d0 !== null).map((r) => ({ from: coord(r.t0), to: coord(r.t1), d0: r.kind === 'flat' ? r.sum / r.n : r.d0, d1: r.kind === 'flat' ? r.sum / r.n : r.d1, kind: r.kind }));
}

function shapeOf(m, tolAbs) {
  const c = fitCircle(m.pts);
  if (c && c.rms < Math.max(tolAbs, c.r * 0.003)) return { kind: 'circle', cx: c.cx, cy: c.cy, d: 2 * c.r, rms: c.rms };
  // axis-aligned rounded rectangle: centre from the ±u / ±v rays, corner radius from the diagonals
  let minU = Infinity, maxU = -Infinity, minV = Infinity, maxV = -Infinity;
  for (const [u, v] of m.pts) { minU = Math.min(minU, u); maxU = Math.max(maxU, u); minV = Math.min(minV, v); maxV = Math.max(maxV, v); }
  const cu = (minU + maxU) / 2, cv = (minV + maxV) / 2, a = (maxU - minU) / 2, b = (maxV - minV) / 2;
  const rel = m.pts.map(([u, v]) => [Math.atan2(v - cv, u - cu), Math.hypot(u - cu, v - cv)]);
  const err = (rc) => { let e = 0; for (const [th, d] of rel) e += (roundedRectDist(a, b, rc, th) - d) ** 2; return Math.sqrt(e / rel.length); };
  let best = 0, bestE = err(0);
  const rmax = Math.min(a, b);
  for (let k = 1; k <= 60; k++) { const rc = (rmax * k) / 60; const e = err(rc); if (e < bestE) { bestE = e; best = rc; } }
  for (let span = rmax / 60; span > 1e-4; span /= 2) for (const rc of [best - span, best + span]) if (rc >= 0 && rc <= rmax) { const e = err(rc); if (e < bestE) { bestE = e; best = rc; } }
  if (bestE < Math.max(tolAbs * 1.5, 0.03)) return { kind: best > rmax - 0.02 && Math.abs(a - b) > 0.05 ? 'slot' : 'rect', cu, cv, w: 2 * a, h: 2 * b, rc: best < tolAbs * 4 ? 0 : best, rms: bestE };
  return { kind: 'irregular', cu, cv, w: 2 * a, h: 2 * b };
}

function describeShape(sh, view, through) {
  const U = AX[view.u], V = AX[view.v];
  if (sh.kind === 'circle') return `round Ø${f3(sh.d)} at ${U}${f3(sh.cx)} ${V}${f3(sh.cy)}`;
  if (sh.kind === 'slot') return `slot ${f3(Math.max(sh.w, sh.h))} long × ${f3(Math.min(sh.w, sh.h))} wide (full-radius ends) along ${sh.w > sh.h ? U : V}, centre ${U}${f3(sh.cu)} ${V}${f3(sh.cv)}`;
  if (sh.kind === 'rect') return `${through ? 'opening' : 'rectangular'} ${f3(sh.w)} (${U}) × ${f3(sh.h)} (${V})${sh.rc > 0.02 ? `, corner R${f2(sh.rc)}` : ', sharp corners'}, centre ${U}${f3(sh.cu)} ${V}${f3(sh.cv)}`;
  return `irregular, about ${f2(sh.w)} × ${f2(sh.h)} around ${U}${f2(sh.cu)} ${V}${f2(sh.cv)}`;
}

// features seen in one view: levels, pockets / holes / islands / steps with measured shapes
export function viewFeatures(field, grid) {
  const view = VIEWS[grid.view];
  const { t, nu, nv, gu, gv, u0, v0, start, sign } = grid;
  const coord = (tv) => start + sign * tv;
  const levels = clusterLevels(t, Math.max(0.04, field.step * 0.25));
  const label = new Int32Array(nu * nv);
  for (let q = 0; q < t.length; q++) {
    if (Number.isNaN(t[q])) { label[q] = -2; continue; }
    let best = -1, bd = Infinity;
    levels.forEach((l, k) => { const d = Math.abs(l.v - t[q]); if (d < bd) { bd = d; best = k; } });
    label[q] = bd <= Math.max(0.05, field.step * 0.3) ? best : -1;
  }
  const cellA = gu * gv;
  // edges of a voxel part are rounded over about a cell: "touching the border" allows a margin
  const margin = Math.max(2, Math.ceil((field.step * 1.5) / Math.min(gu, gv)));
  const comps = components(label, nu, nv, (L) => L !== -1);
  for (const l of levels) l.interior = 0;
  const feats = [];
  const b = field.bounds;
  const farSide = sign < 0 ? b.min[view.d] : b.max[view.d];
  for (const c of comps) {
    let interior = 0, su = 0, sv = 0;
    let i0 = nu, i1 = -1, j0 = nv, j1 = -1;
    const set = new Set(c.cells);
    for (const q of c.cells) {
      const i = q % nu, j = (q - i) / nu;
      if (i > 0 && j > 0 && i < nu - 1 && j < nv - 1 && set.has(q + 1) && set.has(q - 1) && set.has(q + nu) && set.has(q - nu)) interior++;
      su += i; sv += j;
      i0 = Math.min(i0, i); i1 = Math.max(i1, i); j0 = Math.min(j0, j); j1 = Math.max(j1, j);
    }
    if (c.label >= 0) levels[c.label].interior += interior;
    if (interior < 2) continue;
    // pockets / steps smaller than ~3 mm² are facets of curved surfaces, not features
    if (c.label !== -2 && (interior * cellA < 3 || c.cells.length < 9)) continue;
    const sides = { u0: i0 < margin, u1: i1 > nu - 1 - margin, v0: j0 < margin, v1: j1 > nv - 1 - margin };
    const border = sides.u0 || sides.u1 || sides.v0 || sides.v1;
    const through = c.label === -2;
    if (through && border) continue; // outside the part
    // the surrounding surface: most common level just outside the component
    const votes = new Map();
    // look a bit further out around larger features (past chamfers / counterbores at their rim)
    const ring = margin + Math.round(Math.min(3, 0.25 * Math.min((i1 - i0) * gu, (j1 - j0) * gv)) / Math.min(gu, gv));
    for (const q of c.cells) {
      const i = q % nu, j = (q - i) / nu;
      for (const [di, dj] of [[ring, 0], [-ring, 0], [0, ring], [0, -ring]]) {
        const a = i + di, bb = j + dj;
        if (a < 0 || bb < 0 || a >= nu || bb >= nv) continue;
        const r = bb * nu + a;
        if (!set.has(r) && label[r] >= 0 && label[r] !== c.label) votes.set(label[r], (votes.get(label[r]) || 0) + 1);
      }
    }
    let surroundL = -1, bestN = 0;
    for (const [L, n] of votes) if (n > bestN) { bestN = n; surroundL = L; }
    const surroundT = surroundL >= 0 ? levels[surroundL].v : (levels[0]?.v ?? 0);
    const depthT = through ? null : levels[c.label].v;
    let kind;
    if (through) kind = 'hole';
    else if (surroundL < 0 || Math.abs(depthT - surroundT) < 0.02) continue;
    else if (depthT > surroundT) kind = border ? 'open' : 'pocket';
    else if (!border) kind = 'island';
    else continue; // a raised area reaching the edge: described by the lower region next to it
    const ci = Math.round(su / c.cells.length), cj = Math.round(sv / c.cells.length);
    feats.push({
      kind, through, border, sides, area: c.cells.length * cellA, cells: c.cells.length, set,
      depthT, surroundT, floor: through ? null : coord(depthT), wallFrom: coord(surroundT), wallTo: through ? farSide : coord(depthT),
      cu: u0 + (su / c.cells.length) * gu, cv: v0 + (sv / c.cells.length) * gv,
      centroidIn: set.has(cj * nu + ci),
      bbox: { u0: u0 + i0 * gu, u1: u0 + i1 * gu, v0: v0 + j0 * gv, v1: v0 + j1 * gv },
    });
  }
  const step = Math.max(field.step * 0.5, 0.02);
  const tol = Math.max(0.01, field.step * (field.kind === 'height' ? 0.4 : 0.2));
  for (const f of feats) {
    const maxR = Math.hypot(f.bbox.u1 - f.bbox.u0, f.bbox.v1 - f.bbox.v0) + 4 * Math.max(gu, gv);
    const at = (dval) => { const p = [0, 0, 0]; p[view.u] = f.cu; p[view.v] = f.cv; p[view.d] = dval; return p; };
    if (f.kind === 'open') {
      const alongU = f.bbox.u1 - f.bbox.u0 >= f.bbox.v1 - f.bbox.v0;
      f.open = { along: alongU ? AX[view.u] : AX[view.v], from: alongU ? f.bbox.u0 : f.bbox.v0, to: alongU ? f.bbox.u1 : f.bbox.v1, all: f.sides.u0 && f.sides.u1 && f.sides.v0 && f.sides.v1 };
      const p = at(coord((f.surroundT + f.depthT) / 2));
      if (f.centroidIn && !field.inside(p)) {
        const d1 = [0, 0, 0], d2 = [0, 0, 0];
        d1[alongU ? view.v : view.u] = 1; d2[alongU ? view.v : view.u] = -1;
        const a = rayTo(field, p, d1, maxR, step, false), c = rayTo(field, p, d2, maxR, step, false);
        if (a !== null && c !== null) { f.open.width = a + c; f.open.center = (alongU ? f.cv : f.cu) + (a - c) / 2; }
      }
      continue;
    }
    if (f.kind === 'island') {
      // come in from just outside the island at half its height
      const mid = coord((f.surroundT + f.depthT) / 2);
      const pad = 3 * Math.max(gu, gv) + field.step;
      const R = Math.hypot(f.bbox.u1 - f.bbox.u0, f.bbox.v1 - f.bbox.v0) / 2 + pad;
      const cu = (f.bbox.u0 + f.bbox.u1) / 2, cv = (f.bbox.v0 + f.bbox.v1) / 2;
      const m = outlinePts(field, view, cu, cv, mid, R, true);
      f.shape = m ? shapeOf(m, tol) : { kind: 'irregular', cu, cv, w: f.bbox.u1 - f.bbox.u0, h: f.bbox.v1 - f.bbox.v0 };
      continue;
    }
    // pockets and holes: measured at mid-depth (holes: the narrowest of a few depths)
    const range = [f.surroundT, f.through ? Math.abs(farSide - start) : f.depthT];
    const fracs = f.through ? [0.2, 0.5, 0.8] : [0.5];
    const shapes = [];
    for (const fr of fracs) {
      const p = at(coord(range[0] + (range[1] - range[0]) * fr));
      if (field.inside(p)) continue;
      const m = outlinePts(field, view, f.cu, f.cv, p[view.d], maxR, false);
      if (m) shapes.push(shapeOf(m, tol));
    }
    const size = (sh) => (sh.kind === 'circle' ? sh.d : Math.min(sh.w, sh.h));
    shapes.sort((x, y) => size(x) - size(y));
    f.shape = shapes[0] || { kind: 'irregular', cu: f.cu, cv: f.cv, w: f.bbox.u1 - f.bbox.u0, h: f.bbox.v1 - f.bbox.v0 };
    // round holes / blind holes: diameter along the depth (counterbores, tapped tops, chamfers, drill points)
    if (f.shape.kind === 'circle' || shapes.some((x) => x.kind === 'circle')) {
      const c = shapes.find((x) => x.kind === 'circle');
      // a cell in from the rim and the floor: those edges are rounded by the model, not by the tool
      const trim = Math.min(field.kind === 'voxel' ? field.step * 1.2 : field.step * 0.5, Math.abs(range[1] - range[0]) / 4);
      const prof = holeProfile(field, view, c.cx, c.cy, coord, range[0] + trim, range[1] - trim, maxR + 6);
      if (prof.length > 1) {
        f.profile = prof;
        // what fits through a hole is its narrowest straight section
        const flats = prof.filter((r) => r.kind === 'flat');
        if (f.through && flats.length) f.shape = { ...c, d: Math.min(...flats.map((r) => r.d0)) };
      }
    }
  }
  return { levels, outer: levels[0] || null, feats, cellA };
}

const sameProfile = (a, b) => (!a && !b) || (a && b && a.length === b.length && a.every((r, k) => Math.abs(r.d0 - b[k].d0) < 0.05 && Math.abs(r.from - b[k].from) < 0.3));
const sameShape = (a, b, tol) => {
  if (!a || !b || a.kind !== b.kind) return false;
  if (a.kind === 'circle') return Math.abs(a.d - b.d) <= tol;
  return Math.abs(a.w - b.w) <= tol && Math.abs(a.h - b.h) <= tol && Math.abs((a.rc || 0) - (b.rc || 0)) <= tol * 2;
};
const centreOf = (f) => (f.shape && 'cx' in f.shape ? [f.shape.cx, f.shape.cy] : f.shape ? [f.shape.cu, f.shape.cv] : [f.cu, f.cv]);

function describeView(field, grid, res, { maxFeatures = 24 } = {}) {
  const view = VIEWS[grid.view];
  const D = AX[view.d], U = AX[view.u], V = AX[view.v];
  const coord = (tv) => grid.start + grid.sign * tv;
  const out = [];
  if (!res.feats.length) return null;
  out.push(`Seen ${view.label}: face at ${D}${f3(coord(res.outer.v))}`);
  const tol = Math.max(0.02, field.step * 0.1);
  const groups = [];
  for (const f of res.feats.filter((x) => x.kind !== 'open').sort((a, b) => b.area - a.area)) {
    const g = groups.find((x) => x.f.kind === f.kind && sameShape(x.f.shape, f.shape, tol) && sameProfile(x.f.profile, f.profile) && Math.abs((x.f.floor ?? 0) - (f.floor ?? 0)) <= tol && Math.abs(x.f.wallFrom - f.wallFrom) <= tol);
    if (g) g.items.push(f); else groups.push({ f, items: [f] });
  }
  let shown = 0;
  for (const g of groups) {
    if (shown >= maxFeatures) { out.push(`  … ${groups.length - shown} more feature groups`); break; }
    shown++;
    const f = g.f;
    const n = g.items.length;
    const sh = f.shape;
    let what, depth;
    if (f.kind === 'hole') {
      what = sh?.kind === 'circle' ? 'through hole' : 'through opening';
      depth = `through from ${D}${f3(f.wallFrom)} to ${D}${f3(f.wallTo)}`;
    } else if (f.kind === 'island') {
      what = 'island/boss';
      depth = `top ${D}${f3(f.floor)}, standing ${f3(Math.abs(f.wallFrom - f.floor))} above the ${D}${f3(f.wallFrom)} floor around it`;
    } else {
      what = sh?.kind === 'circle' ? 'round pocket/blind hole' : 'pocket';
      depth = `floor ${D}${f3(f.floor)}, ${f3(Math.abs(f.floor - f.wallFrom))} deep below ${D}${f3(f.wallFrom)}`;
    }
    if (f.profile) depth += `; along the depth: ${f.profile.map((r) => (r.kind === 'taper' ? `Ø${f2(r.d0)}→Ø${f2(r.d1)} (cone/chamfer)` : `Ø${f3(r.d0)}`) + ` ${D}${f2(r.from)}..${f2(r.to)}`).join(', ')}`;
    if (n === 1) {
      out.push(`  ${what}: ${sh ? describeShape(sh, view, f.kind === 'hole') : 'unmeasured'}; ${depth}; area ${f2(f.area)} mm²`);
    } else {
      const size = sh?.kind === 'circle' ? `Ø${f3(sh.d)}` : sh ? `${f2(sh.w)} × ${f2(sh.h)}${sh.rc > 0.02 ? ` R${f2(sh.rc)}` : ''}` : '';
      const at = g.items.map((x) => { const [a, b] = centreOf(x); return `(${f3(a)}, ${f3(b)})`; }).join(' ');
      out.push(`  ${n}× ${what} ${size}; ${depth}; centres ${U},${V}: ${at}`);
    }
  }
  for (const f of res.feats.filter((x) => x.kind === 'open').sort((a, b) => b.area - a.area).slice(0, 12)) {
    const o = f.open;
    const below = `floor ${D}${f3(f.floor)}, ${f3(Math.abs(f.floor - f.wallFrom))} below ${D}${f3(f.wallFrom)}`;
    if (o.all && !f.centroidIn) out.push(`  cut down all around the outside (outside contour / frame): ${below}; area ${f2(f.area)} mm²`);
    else if (o.width) out.push(`  open slot/channel along ${o.along} ${f2(o.from)}..${f2(o.to)}, ${f3(o.width)} wide (centre ${o.along === U ? V : U}${f3(o.center)}): ${below}; area ${f2(f.area)} mm²`);
    else out.push(`  step/shoulder at the edge, ${U}${f2(f.bbox.u0)}..${f2(f.bbox.u1)} ${V}${f2(f.bbox.v0)}..${f2(f.bbox.v1)}: ${below}; area ${f2(f.area)} mm²`);
  }
  const lv = res.levels.filter((l) => l !== res.outer && l.interior * res.cellA >= 2).sort((a, b) => a.v - b.v).slice(0, 12);
  if (lv.length) out.push(`  flat levels (area): ${lv.map((l) => `${D}${f2(coord(l.v))} ${f2(l.n * res.cellA)} mm²`).join(', ')}`);
  return out.join('\n');
}

// coarse picture of a view: one character per cell (digits = depth bands)
export function depthMap(grid, maxCols = 64) {
  const { t, nu, nv } = grid;
  const step = Math.max(1, Math.ceil(nu / maxCols), Math.ceil(nv / (maxCols / 2)));
  let tmin = Infinity, tmax = -Infinity;
  for (const v of t) if (!Number.isNaN(v)) { tmin = Math.min(tmin, v); tmax = Math.max(tmax, v); }
  const span = Math.max(1, tmax - tmin); // shallower than 1 mm stays '#' (edge rounding, not features)
  const rows = [];
  for (let j = nv - 1; j >= 0; j -= step) {
    let row = '';
    for (let i = 0; i < nu; i += step) {
      const v = t[j * nu + i];
      if (Number.isNaN(v)) { row += ' '; continue; }
      const k = Math.round(((v - tmin) / span) * 9);
      row += k === 0 ? '#' : String(10 - k > 9 ? 9 : k);
    }
    rows.push(row.replace(/\s+$/, ''));
  }
  const coord = (tv) => grid.start + grid.sign * tv;
  const view = VIEWS[grid.view];
  return {
    text: rows.join('\n'),
    legend: `'#' = the face at ${AX[view.d]}${f2(coord(tmin))}, digits 1–9 = deeper in steps of ${f2(span / 9)} mm down to ${AX[view.d]}${f2(coord(tmax))}, ' ' = no material (through/outside). One character ≈ ${f2(grid.gu * step)} mm; ${AX[view.u]} to the right${view.u === 0 && grid.view === 'back' ? ' (mirrored)' : ''}, ${AX[view.v]} up.`,
  };
}

// ---------------------------------------------------------------- outline (footprint) of the part
function footprint(field, view) {
  const b = field.bounds;
  const cu = (b.min[view.u] + b.max[view.u]) / 2, cv = (b.min[view.v] + b.max[view.v]) / 2;
  const R = Math.hypot(b.max[view.u] - b.min[view.u], b.max[view.v] - b.min[view.v]);
  let best = null;
  for (const fr of [0.2, 0.5, 0.8]) {
    const dval = b.min[view.d] + (b.max[view.d] - b.min[view.d]) * fr;
    const m = outlinePts(field, view, cu, cv, dval, R, true);
    if (!m) continue;
    const pts = m.pts.slice(0, 32);
    let A = 0;
    for (let k = 0; k < pts.length; k++) { const p = pts[k], q = pts[(k + 1) % pts.length]; A += p[0] * q[1] - q[0] * p[1]; }
    if (!best || A > best.A + 1e-6) best = { A, m, dval };
  }
  if (!best) return null;
  return { ...shapeOf(best.m, Math.max(0.01, field.step * 0.2)), at: best.dval };
}

// ---------------------------------------------------------------- lathe profile
function rdp(pts, tol) {
  if (pts.length < 3) return pts.slice();
  const keep = new Uint8Array(pts.length);
  keep[0] = keep[pts.length - 1] = 1;
  const stack = [[0, pts.length - 1]];
  while (stack.length) {
    const [a, b] = stack.pop();
    const [za, ra] = pts[a], [zb, rb] = pts[b];
    const dz = zb - za, dr = rb - ra, L = Math.hypot(dz, dr) || 1e-12;
    let md = 0, mi = -1;
    for (let i = a + 1; i < b; i++) { const d = Math.abs((pts[i][0] - za) * dr - (pts[i][1] - ra) * dz) / L; if (d > md) { md = d; mi = i; } }
    if (md > tol) { keep[mi] = 1; stack.push([a, mi], [mi, b]); }
  }
  return pts.filter((_, i) => keep[i]);
}

// threads: diameter oscillating with a constant pitch
function findThreads(z, d) {
  const ext = [];
  let dir = 0, lastI = 0;
  for (let i = 1; i < d.length; i++) {
    const dd = d[i] - d[lastI];
    if (Math.abs(dd) < 0.02) continue;
    const nd = Math.sign(dd);
    if (dir && nd !== dir) ext.push({ i: lastI, max: dir > 0 });
    dir = nd; lastI = i;
  }
  const maxima = ext.filter((e) => e.max);
  const runs = [];
  let run = [];
  for (let k = 0; k < maxima.length; k++) {
    if (run.length >= 1) {
      const p = Math.abs(z[maxima[k].i] - z[run[run.length - 1].i]);
      const p0 = run.length >= 2 ? Math.abs(z[run[1].i] - z[run[0].i]) : p;
      if (Math.abs(p - p0) > 0.08 * p0 || p > 8) { if (run.length >= 4) runs.push(run); run = []; }
    }
    run.push(maxima[k]);
  }
  if (run.length >= 4) runs.push(run);
  return runs.map((r) => {
    const i0 = r[0].i, i1 = r[r.length - 1].i;
    let lo = Infinity, hi = -Infinity;
    for (let i = Math.min(i0, i1); i <= Math.max(i0, i1); i++) { lo = Math.min(lo, d[i]); hi = Math.max(hi, d[i]); }
    return { i0: Math.min(i0, i1), i1: Math.max(i0, i1), pitch: Math.abs(z[i1] - z[i0]) / (r.length - 1), major: hi, minor: lo };
  }).filter((t) => t.major - t.minor > 0.15);
}

function profileRows(sim, arr, isBore) {
  const zs = [], ds = [];
  for (let i = sim.n - 1; i >= 0; i--) {
    if (sim.outer[i] - sim.inner[i] < 1e-4) continue;
    zs.push(sim.zMin + i * sim.dz);
    ds.push(arr[i] * 2);
  }
  if (!zs.length) return [];
  const threads = findThreads(zs, ds);
  // a thread's first/last partial turns belong to it
  for (const th of threads) {
    const p = Math.max(1, Math.round(th.pitch / sim.dz));
    let a = th.i0, b = th.i1;
    for (let k = 0; k < p && a > 0 && ds[a - 1] >= th.minor - 0.02 && ds[a - 1] <= th.major + 0.02; k++) a--;
    for (let k = 0; k < p && b < zs.length - 1 && ds[b + 1] >= th.minor - 0.02 && ds[b + 1] <= th.major + 0.02; k++) b++;
    th.i0 = a; th.i1 = b;
  }
  const rows = [];
  if (!isBore) rows.push({ z0: zs[0], front: true, text: `  Z${f2(zs[0])}: front face (Ø0 → Ø${f3(ds[0])})` });
  const pieces = [];
  let s0 = 0;
  for (const th of threads) { pieces.push([s0, th.i0]); pieces.push({ th }); s0 = th.i1; }
  pieces.push([s0, zs.length - 1]);
  for (const pc of pieces) {
    if (pc.th) {
      const t = pc.th;
      rows.push({ text: `  Z${f2(zs[t.i0])} → Z${f2(zs[t.i1])}: ${isBore ? 'internal ' : ''}thread, pitch ${f3(t.pitch)}, ${isBore ? 'minor (bore) Ø' + f3(t.minor) + ', major Ø' + f3(t.major) : 'major Ø' + f3(t.major) + ', minor Ø' + f3(t.minor)} (depth ${f3((t.major - t.minor) / 2)})` });
      continue;
    }
    const [a, b] = pc;
    if (b - a < 1) continue;
    const pts = [];
    for (let i = a; i <= b; i++) pts.push([zs[i], ds[i] / 2]);
    const v = rdp(pts, 0.004);
    const segs = [];
    for (let k = 0; k + 1 < v.length; k++) segs.push([v[k], v[k + 1]]);
    let k = 0;
    while (k < segs.length) {
      let m = k;
      const short = (s) => Math.hypot(s[1][0] - s[0][0], s[1][1] - s[0][1]) < 1.6 && Math.abs(s[1][0] - s[0][0]) > 1e-3 && Math.abs(s[1][1] - s[0][1]) > 1e-3;
      while (m < segs.length && short(segs[m])) m++;
      if (m - k >= 3) {
        const P = [segs[k][0], ...segs.slice(k, m).map((s) => s[1])];
        const c = fitCircle(P);
        if (c && c.rms < 0.01 && c.r < 60) {
          const [za, ra] = P[0], [zb, rb] = P[P.length - 1];
          const convex = c.cy < (ra + rb) / 2;
          rows.push({ text: `  Z${f2(za)} → Z${f2(zb)}: R${f2(c.r)} ${convex !== isBore ? 'rounded edge' : 'fillet'} Ø${f3(2 * ra)} → Ø${f3(2 * rb)}` });
          k = m;
          continue;
        }
      }
      // a leftover sliver (< 0.2 mm, from the grid or a nose radius) joins the next piece
      if (k + 1 < segs.length && Math.hypot(segs[k][1][0] - segs[k][0][0], segs[k][1][1] - segs[k][0][1]) < 0.2) { segs[k + 1] = [segs[k][0], segs[k + 1][1]]; k++; continue; }
      const [[za, ra], [zb, rb]] = segs[k];
      const dz = Math.abs(zb - za), dr = rb - ra;
      if (Math.abs(dr) < 0.004) rows.push({ text: `  Z${f2(za)} → Z${f2(zb)}: Ø${f3(2 * ra)} (${f2(dz)} long)` });
      else if (dz < sim.dz * 2.5) {
        // consecutive face pieces (a face cut in two passes, or a nose-radius foot) are one face
        const prev = rows[rows.length - 1];
        if (prev?.front && Math.abs(prev.z0 - za) < 0.3 && rb > ra) { prev.dOut = 2 * rb; prev.text = `  Z${f2(prev.z0)}: front face (Ø0 → Ø${f3(prev.dOut)})`; }
        else if (prev?.face && prev.d1 !== undefined && Math.abs(prev.z - za) < 0.3) { prev.d1 = 2 * rb; prev.text = `  Z${f2(prev.z)}: face/shoulder Ø${f3(prev.d0)} → Ø${f3(prev.d1)}`; }
        else rows.push({ face: true, z: za, d0: 2 * ra, d1: 2 * rb, text: `  Z${f2(za)}: face/shoulder Ø${f3(2 * ra)} → Ø${f3(2 * rb)}` });
      } else {
        const ang = (Math.atan2(Math.abs(dr), dz) * 180) / Math.PI;
        rows.push({ text: `  Z${f2(za)} → Z${f2(zb)}: ${Math.hypot(dz, dr) < 4 && Math.abs(ang - 45) < 2 ? 'chamfer' : 'taper'} Ø${f3(2 * ra)} → Ø${f3(2 * rb)} (${f2(ang)}° to the axis)` });
      }
      k++;
    }
  }
  return rows.map((r) => r.text);
}

export function profileSummary(sim) {
  const mb = sim.materialBounds();
  if (!mb) return 'Nothing is left of the part.';
  const out = [];
  out.push(`Part: Z${f2(mb.max[2])} (front face) to Z${f2(mb.min[2])}, max Ø${f3(mb.max[0] * 2)}; the chuck jaws hold it from Z${f2(sim.zChuck ?? mb.min[2])} back.`);
  if (sim.partedOff !== null && sim.partedOff !== undefined) out.push(`Parted off at Z${f2(sim.partedOff)}.`);
  out.push('Outside profile, front face → chuck:');
  out.push(...profileRows(sim, sim.outer, false).slice(0, 60));
  if (sim.inner.some((v) => v > 0.01)) {
    out.push('Bore profile, front face → chuck:');
    out.push(...profileRows(sim, sim.inner, true).slice(0, 40));
  }
  return out.join('\n');
}

// ---------------------------------------------------------------- summaries
function millSummary(out, field, { views = null } = {}) {
  const b = field.bounds;
  const o = [];
  o.push(`Part after this operation (work coordinates of this operation): X${f3(b.min[0])}..${f3(b.max[0])} Y${f3(b.min[1])}..${f3(b.max[1])} Z${f3(b.min[2])}..${f3(b.max[2])} — ${f3(b.max[0] - b.min[0])} × ${f3(b.max[1] - b.min[1])} × ${f3(b.max[2] - b.min[2])} mm`);
  o.push(field.kind === 'voxel'
    ? `Measured on the 3D material model (${f2(field.step)} mm cells): sizes ±${f2(Math.max(0.01, field.step * 0.1))} mm, sharp edges look rounded by about one cell.`
    : `Measured on the height map (${f2(field.step)} mm grid): Z levels are exact, wall positions ±${f2(field.step / 2)} mm.`);
  const fp = footprint(field, VIEWS.top);
  if (fp) o.push(`Outline seen from above (widest, at Z${f2(fp.at)}): ${fp.kind === 'circle' ? `round Ø${f3(fp.d)} centred X${f3(fp.cx)} Y${f3(fp.cy)}` : fp.kind === 'rect' || fp.kind === 'slot' ? `${f3(fp.w)} × ${f3(fp.h)}${fp.rc > 0.02 ? `, corners R${f2(fp.rc)}` : ', square corners'}, centre X${f3(fp.cu)} Y${f3(fp.cv)}` : 'not a plain rectangle/circle (profiled shape)'}`);
  const list = views || (field.kind === 'voxel' ? ['top', 'bottom', 'front', 'back', 'left', 'right'] : ['top']);
  for (const v of list) {
    const coarse = v !== 'top' && v !== 'bottom';
    const g = depthGrid(field, v, coarse ? 90 : 200);
    const res = viewFeatures(field, g);
    const txt = describeView(field, g, res);
    if (txt) o.push(txt);
    else if (v === 'top' || v === 'bottom') o.push(`Seen ${VIEWS[v].label}: plain face at ${AX[VIEWS[v].d]}${f3(g.start + g.sign * (res.outer?.v ?? 0))}, no features.`);
  }
  return o.join('\n');
}

export function partSummary(out) {
  if (out.summaryText) return out.summaryText;
  usePrecision(out.engine);
  let txt;
  try {
    if (out.engine === 'lathe') txt = profileSummary(out.final);
    else {
      const field = materialField(out);
      txt = field.bounds ? millSummary(out, field) : 'Nothing is left of the part.';
    }
  } catch (e) {
    txt = `(part analysis failed: ${e.message})`;
  }
  out.summaryText = txt;
  return txt;
}

// ---------------------------------------------------------------- sections & probes
function marchingSquares(field, plane, at, b, maxCells = 260) {
  const [ua, va] = plane; // in-plane axes
  const da = 3 - ua - va;
  const su = b.max[ua] - b.min[ua] + 2, sv = b.max[va] - b.min[va] + 2;
  const g = Math.max(field.step * 0.7, Math.max(su, sv) / maxCells);
  const nu = Math.ceil(su / g) + 1, nv = Math.ceil(sv / g) + 1;
  const u0 = b.min[ua] - 1, v0 = b.min[va] - 1;
  const P = (u, v) => { const p = [0, 0, 0]; p[ua] = u; p[va] = v; p[da] = at; return p; };
  const ins = new Uint8Array(nu * nv);
  for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) ins[j * nu + i] = field.inside(P(u0 + i * g, v0 + j * g)) ? 1 : 0;
  // crossing point on a grid edge, refined by bisection
  const cache = new Map();
  const cross = (i, j, horiz) => {
    const key = `${i},${j},${horiz}`;
    if (cache.has(key)) return cache.get(key);
    let a = [u0 + i * g, v0 + j * g], c = horiz ? [a[0] + g, a[1]] : [a[0], a[1] + g];
    const inA = field.inside(P(...a));
    for (let k = 0; k < 16; k++) {
      const m = [(a[0] + c[0]) / 2, (a[1] + c[1]) / 2];
      if (field.inside(P(...m)) === inA) a = m; else c = m;
    }
    const r = [(a[0] + c[0]) / 2, (a[1] + c[1]) / 2];
    cache.set(key, r);
    return r;
  };
  const segs = [];
  for (let j = 0; j < nv - 1; j++) for (let i = 0; i < nu - 1; i++) {
    const a = ins[j * nu + i], bq = ins[j * nu + i + 1], c = ins[(j + 1) * nu + i + 1], d = ins[(j + 1) * nu + i];
    const idx = a | (bq << 1) | (c << 2) | (d << 3);
    if (idx === 0 || idx === 15) continue;
    const E = { b: () => cross(i, j, true), r: () => cross(i + 1, j, false), t: () => cross(i, j + 1, true), l: () => cross(i, j, false) };
    const table = { 1: [['l', 'b']], 2: [['b', 'r']], 3: [['l', 'r']], 4: [['r', 't']], 5: [['l', 't'], ['r', 'b']], 6: [['b', 't']], 7: [['l', 't']], 8: [['t', 'l']], 9: [['t', 'b']], 10: [['t', 'r'], ['b', 'l']], 11: [['t', 'r']], 12: [['r', 'l']], 13: [['r', 'b']], 14: [['b', 'l']] };
    for (const [e1, e2] of table[idx]) segs.push([E[e1](), E[e2]()]);
  }
  // chain into loops
  const k2 = (p) => `${p[0].toFixed(5)},${p[1].toFixed(5)}`;
  const byStart = new Map();
  for (const s of segs) byStart.set(k2(s[0]), s);
  const used = new Set();
  const loops = [];
  for (const s of segs) {
    if (used.has(s)) continue;
    const loop = [s[0]];
    let cur = s;
    while (cur && !used.has(cur)) { used.add(cur); loop.push(cur[1]); cur = byStart.get(k2(cur[1])); }
    loops.push(loop);
  }
  return loops;
}

export function section(out, axis, at) {
  usePrecision(out.engine);
  if (out.engine === 'lathe') return profileSummary(out.final);
  const field = materialField(out);
  if (!field.bounds) return 'Nothing is left of the part.';
  const a = 'xyz'.indexOf(String(axis).toLowerCase());
  if (a < 0) return 'axis must be x, y or z';
  const plane = [0, 1, 2].filter((k) => k !== a);
  const loops = marchingSquares(field, plane, at, field.bounds);
  if (!loops.length) return `Section ${AX[a]}=${f3(at)}: no material in this plane.`;
  const U = AX[plane[0]], V = AX[plane[1]];
  const o = [`Section ${AX[a]}=${f3(at)} (${U}${V} plane, work coordinates), ${loops.length} loop${loops.length > 1 ? 's' : ''}; each is a closed outline — of material (clockwise, ${U} right / ${V} up) or of a hole/cavity in the material (counter-clockwise):`];
  let budget = 700;
  loops
    .map((lp) => {
      let A = 0;
      for (let i = 0; i + 1 < lp.length; i++) A += lp[i][0] * lp[i + 1][1] - lp[i + 1][0] * lp[i][1];
      return { lp, A: A / 2 };
    })
    .sort((x, y) => Math.abs(y.A) - Math.abs(x.A))
    .forEach(({ lp, A }, n) => {
      // closed loop: simplify the two halves split at the point farthest from the start
      let far = 0, fd = -1;
      lp.forEach((p, k) => { const d = Math.hypot(p[0] - lp[0][0], p[1] - lp[0][1]); if (d > fd) { fd = d; far = k; } });
      const what = A < 0 ? 'material' : 'hole/cavity';
      const c = lp.length >= 8 ? fitCircle(lp) : null;
      if (c && c.rms < Math.max(0.02, field.step * (field.kind === 'height' ? 0.35 : 0.15))) {
        o.push(`loop ${n + 1}: ${what}, circle Ø${f3(2 * c.r)} centred ${U}${f3(c.cx)} ${V}${f3(c.cy)}`);
        return;
      }
      const tolS = Math.max(0.01, field.step * (field.kind === 'height' ? 0.5 : 0.1));
      const simp = [...rdp(lp.slice(0, far + 1), tolS).slice(0, -1), ...rdp(lp.slice(far), tolS)];
      const pts = simp.length > budget ? simp.filter((_, i) => i % Math.ceil(simp.length / budget) === 0) : simp;
      budget = Math.max(40, budget - pts.length);
      o.push(`loop ${n + 1}: ${what}, area ${f2(Math.abs(A))} mm², ${pts.length} points:`);
      o.push('  ' + pts.map((p) => `${f2(p[0])},${f2(p[1])}`).join(' '));
    });
  return o.join('\n');
}

export function probe(out, points) {
  usePrecision(out.engine);
  const field = materialField(out);
  return points.slice(0, 200).map((p) => {
    const q = [Number(p[0]) || 0, Number(p[1]) || 0, Number(p[2]) || 0];
    const ins = field.inside(q);
    const d = field.distance ? Math.abs(field.distance(q)) : null;
    return `X${f3(q[0])} Y${f3(q[1])} Z${f3(q[2])}: ${ins ? 'material' : 'air'}${d !== null && d < field.step * 4 ? ` (≈${f2(d)} mm from the surface)` : ''}`;
  }).join('\n');
}

// one view in detail: features + a character depth map
export function viewDetail(out, viewName) {
  usePrecision(out.engine);
  if (out.engine === 'lathe') return profileSummary(out.final);
  const field = materialField(out);
  if (!field.bounds) return 'Nothing is left of the part.';
  if (!VIEWS[viewName]) return `view must be one of ${Object.keys(VIEWS).join(', ')}`;
  const g = depthGrid(field, viewName, 200);
  const res = viewFeatures(field, g);
  const txt = describeView(field, g, res, { maxFeatures: 60 }) || `Seen ${VIEWS[viewName].label}: plain face, no features.`;
  const m = depthMap(g);
  return `${txt}\nDepth map (${m.legend}):\n${m.text}`;
}
