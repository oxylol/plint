// Measuring the machined part in the view (Iteration 2): a clicked point snaps to what the
// simulator already knows — hole and boss centres and rims, pocket and part edges and corners,
// floor and face levels, lathe shoulders — and comes back with its coordinates, what it is, the
// deviation from the target model there, and distances between points. Pure: runs in Node.
import { materialField, depthGrid, viewFeatures, VIEWS, footprint } from './inspect.js';
import { invertX, applyX } from './voxelsim.js';

const f2 = (v) => (Math.abs(v) < 0.005 ? 0 : v).toFixed(2);
const f3 = (v) => (Math.abs(v) < 0.0005 ? 0 : v).toFixed(3);

// everything a session of clicks needs for one operation's result
export function measureContext(out, target) {
  if (out.engine === 'lathe') {
    const sim = out.final;
    const shoulders = [];
    for (let i = 1; i < sim.n; i++) {
      if (sim.outer[i] - sim.inner[i] < 1e-4 && sim.outer[i - 1] - sim.inner[i - 1] < 1e-4) continue;
      if (Math.abs(sim.outer[i] - sim.outer[i - 1]) > 0.05 || Math.abs(sim.inner[i] - sim.inner[i - 1]) > 0.05) shoulders.push(sim.zMin + i * sim.dz - sim.dz / 2);
    }
    const mb = sim.materialBounds();
    if (mb) shoulders.push(mb.max[2], mb.min[2]);
    return { lathe: true, sim, shoulders, out, target, snap: Math.max(0.3, sim.dz * 3) };
  }
  const field = materialField(out);
  const ctx = { field, feats: [], levels: [], outline: null, out, target, snap: Math.max(0.6, field.step * 2.5) };
  if (!field.bounds) return ctx;
  const grid = depthGrid(field, 'top', 200);
  const res = viewFeatures(field, grid);
  const coord = (t) => grid.start + grid.sign * t;
  ctx.feats = res.feats;
  ctx.levels = res.levels.map((l) => coord(l.v));
  ctx.top = ctx.levels[0] ?? field.bounds.max[2];
  ctx.outline = footprint(field, VIEWS.top);
  if (target?.sim) {
    const inv = invertX(out.workFromPart || { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [0, 0, 0] });
    ctx.deviation = (p) => target.sim.sample(...applyX(inv, p));
  }
  return ctx;
}

const featName = (f) => {
  const sh = f.shape || {};
  if (f.kind === 'hole') return `Ø${f2(sh.d)} hole`;
  const size = sh.kind === 'circle' ? `Ø${f2(sh.d)}` : sh.w ? `${f2(sh.w)}×${f2(sh.h)}` : '';
  return `${f.kind === 'island' ? 'island' : f.kind === 'open' ? 'open pocket' : 'pocket'}${size ? ` ${size}` : ''}`;
};

// snap a work point to feature geometry within ctx.snap; returns { p, what: [...], feat }
export function snapMill(ctx, p0) {
  const sd = ctx.snap;
  const p = p0.slice();
  const what = [];
  let best = null, feat = null;
  const consider = (x, y, text, pri, f) => {
    const d = Math.hypot(x - p0[0], y - p0[1]);
    if (d <= sd && (!best || d < best.d - 1e-9 || (Math.abs(d - best.d) < 1e-9 && pri > best.pri))) best = { d, x, y, text, pri, f };
  };
  const edges = [];
  for (const f of ctx.feats) {
    const sh = f.shape;
    if (!sh) continue;
    const name = featName(f);
    if (sh.kind === 'circle') {
      consider(sh.cx, sh.cy, `centre of ${name}`, 3, f);
      const d = Math.hypot(p0[0] - sh.cx, p0[1] - sh.cy);
      if (d > 1e-6 && Math.abs(d - sh.d / 2) <= sd) { const k = sh.d / 2 / d; consider(sh.cx + (p0[0] - sh.cx) * k, sh.cy + (p0[1] - sh.cy) * k, `rim of ${name}`, 1, f); }
    } else if (sh.kind === 'rect' || sh.kind === 'slot') {
      const x0 = sh.cu - sh.w / 2, x1 = sh.cu + sh.w / 2, y0 = sh.cv - sh.h / 2, y1 = sh.cv + sh.h / 2;
      if (!(sh.rc > 0.02) || sh.kind === 'rect') for (const x of [x0, x1]) for (const y of [y0, y1]) consider(x, y, `corner of ${name}`, 2, f);
      consider(sh.cu, sh.cv, `centre of ${name}`, 2, f);
      edges.push({ axis: 0, v: x0, lo: y0, hi: y1, text: `X${f2(x0)} wall of ${name}`, f }, { axis: 0, v: x1, lo: y0, hi: y1, text: `X${f2(x1)} wall of ${name}`, f });
      edges.push({ axis: 1, v: y0, lo: x0, hi: x1, text: `Y${f2(y0)} wall of ${name}`, f }, { axis: 1, v: y1, lo: x0, hi: x1, text: `Y${f2(y1)} wall of ${name}`, f });
    }
  }
  const o = ctx.outline;
  if (o && (o.kind === 'rect' || o.kind === 'slot')) {
    const x0 = o.cu - o.w / 2, x1 = o.cu + o.w / 2, y0 = o.cv - o.h / 2, y1 = o.cv + o.h / 2;
    if (!(o.rc > 0.02)) for (const x of [x0, x1]) for (const y of [y0, y1]) consider(x, y, `corner of the part (${f2(o.w)}×${f2(o.h)})`, 2, null);
    edges.push({ axis: 0, v: x0, lo: y0, hi: y1, text: `X${f2(x0)} side of the part` }, { axis: 0, v: x1, lo: y0, hi: y1, text: `X${f2(x1)} side of the part` });
    edges.push({ axis: 1, v: y0, lo: x0, hi: x1, text: `Y${f2(y0)} side of the part` }, { axis: 1, v: y1, lo: x0, hi: x1, text: `Y${f2(y1)} side of the part` });
  } else if (o && o.kind === 'circle') {
    consider(o.cx, o.cy, `centre of the round part Ø${f2(o.d)}`, 2, null);
  }
  if (best) { p[0] = best.x; p[1] = best.y; what.push(best.text); feat = best.f; }
  else {
    for (const axis of [0, 1]) {
      let e = null;
      for (const ed of edges) {
        if (ed.axis !== axis) continue;
        const d = Math.abs(ed.v - p0[axis]);
        const along = p0[1 - axis];
        if (d <= sd && along >= ed.lo - sd && along <= ed.hi + sd && (!e || d < e.d)) e = { d, ed };
      }
      if (e) { p[axis] = e.ed.v; what.push(e.ed.text); feat = feat || e.ed.f || null; }
    }
  }
  let bz = null;
  for (const z of ctx.levels) { const d = Math.abs(z - p0[2]); if (d <= sd && (!bz || d < bz.d)) bz = { d, z }; }
  if (bz) p[2] = bz.z;
  return { p, what, feat, level: bz ? bz.z : null };
}

// which face the point sits on, from the surface normal
function surfaceKind(ctx, p, n) {
  if (!n) return '';
  const [nx, ny, nz] = n;
  if (Math.abs(nz) > 0.8) {
    const f = ctx.feats.find((x) => x.floor !== null && x.floor !== undefined && Math.abs(x.floor - p[2]) < 0.05 && inFeat(x, p));
    if (f) return `floor of ${featName(f)}`;
    if (Math.abs(p[2] - ctx.top) < 0.05) return nz > 0 ? 'top face' : 'underside';
    return nz > 0 ? `face at Z${f2(p[2])}` : `underside at Z${f2(p[2])}`;
  }
  if (Math.abs(nz) < 0.2) {
    const f = ctx.feats.find((x) => x.shape && inFeat(x, p, ctx.snap));
    if (f && f.shape.kind === 'circle') return `wall of ${featName(f)}`;
    const ax = Math.abs(nx) > Math.abs(ny) ? 'X' : 'Y';
    const v = ax === 'X' ? p[0] : p[1];
    return `${f ? `wall of ${featName(f)}` : 'wall'} at ${ax}${f2(v)}`;
  }
  return 'sloped surface';
}
function inFeat(f, p, pad = 0) {
  const sh = f.shape;
  if (!sh) return false;
  if (sh.kind === 'circle') return Math.hypot(p[0] - sh.cx, p[1] - sh.cy) <= sh.d / 2 + pad;
  const b = f.bbox;
  return p[0] >= b.u0 - pad && p[0] <= b.u1 + pad && p[1] >= b.v0 - pad && p[1] <= b.v1 + pad;
}

export function deviationText(ctx, p) {
  if (!ctx.deviation) return '';
  const d = ctx.deviation(p);
  if (!Number.isFinite(d) || Math.abs(d) > 50) return '';
  if (Math.abs(d) < 0.01) return 'on size vs model';
  return d > 0 ? `+${f2(d)} mm material left vs model` : `${f2(d)} mm gouge vs model`;
}

// one click on the mill part (or its target ghost): p work point, n work normal, on 'part' | 'target'
export function pointItem(ctx, p0, n, on = 'part') {
  if (ctx.lathe) return lathePoint(ctx, p0);
  const s = snapMill(ctx, p0);
  const p = s.p;
  const parts = [];
  const kind = on === 'target' ? 'on the target model' : surfaceKind(ctx, p, n);
  if (s.what.length) parts.push(s.what.join(', '));
  else if (kind) parts.push(kind);
  if (s.what.length && kind && !/floor|wall|face/.test(s.what.join())) parts.push(kind);
  const item = { kind: 'point', p, points: [p], text: `X${f3(p[0])} Y${f3(p[1])} Z${f3(p[2])}`, detail: parts.join(' · '), labels: [{ p, text: `X${f2(p[0])} Y${f2(p[1])} Z${f2(p[2])}` }] };
  if (on === 'part') { const dev = deviationText(ctx, p); if (dev) item.detail += `${item.detail ? ' · ' : ''}${dev}`; }
  const f = s.feat;
  if (f?.shape?.kind === 'circle' && f.kind !== 'island') {
    const sh = f.shape;
    const depth = f.through ? null : Math.abs(f.wallFrom - f.wallTo);
    item.feature = { kind: f.kind, d: sh.d, cx: sh.cx, cy: sh.cy, top: f.wallFrom, depth };
    item.detail += ` · Ø${f3(sh.d)} centre X${f2(sh.cx)} Y${f2(sh.cy)}, ${f.through ? 'through' : `${f2(depth)} deep`}`;
    item.circle = { c: [sh.cx, sh.cy, f.wallFrom], r: sh.d / 2, axis: 'z' };
    if (f.profile?.length > 1) item.detail += ` (${f.profile.map((r) => `${r.kind === 'flat' ? `Ø${f2(r.d0)}` : `Ø${f2(r.d0)}→${f2(r.d1)}`} to Z${f2(r.to)}`).join(', ')})`;
  } else if (f?.shape && (f.shape.kind === 'rect' || f.shape.kind === 'slot')) {
    const sh = f.shape;
    const depth = f.through ? null : Math.abs(f.wallFrom - f.wallTo);
    item.detail += ` · ${featName(f)} centre X${f2(sh.cu)} Y${f2(sh.cv)}${sh.rc > 0.02 ? `, corners R${f2(sh.rc)}` : ''}, ${f.through ? 'through' : `${f2(depth)} deep`}`;
  }
  return item;
}

function lathePoint(ctx, p0) {
  const sd = ctx.snap;
  let r = p0[0], z = p0[2];
  const what = [];
  let bz = null;
  for (const s of ctx.shoulders) { const d = Math.abs(s - z); if (d <= sd && (!bz || d < bz.d)) bz = { d, s }; }
  if (bz) { z = bz.s; what.push('shoulder'); }
  const sim = ctx.sim;
  const i = Math.round((z - sim.zMin) / sim.dz);
  if (i >= 0 && i < sim.n) {
    // at a shoulder the radius is read on the plateaus beside it, not in the transition cell
    const k = Math.max(2, Math.ceil(0.6 / sim.dz)); // past the nose-radius corner
    const idx = bz ? [i - k, i + k] : [i];
    let best = null;
    for (const j of idx) {
      const jj = Math.min(sim.n - 1, Math.max(0, j));
      for (const [val, name] of [[sim.outer[jj], 'outside'], [sim.inner[jj], 'bore']]) {
        if (name === 'bore' && !(val > 0.01)) continue;
        const d = Math.abs(r - val);
        if (d <= sd && (!best || d < best.d)) best = { d, val, name };
      }
    }
    if (best) { r = best.val; what.push(`${best.name} Ø${f3(2 * best.val)}`); }
  }
  const p = [r, 0, z];
  return { kind: 'point', p, points: [p], text: `Ø${f3(2 * r)} Z${f3(z)}`, detail: what.join(', '), labels: [{ p, text: `Ø${f2(2 * r)} Z${f2(z)}` }] };
}

export function distanceItem(ctx, a, b) {
  if (ctx.lathe) {
    const dz = b.p[2] - a.p[2], dd = 2 * (b.p[0] - a.p[0]);
    const mid = [(a.p[0] + b.p[0]) / 2, 0, (a.p[2] + b.p[2]) / 2];
    return { kind: 'distance', points: [a.p, b.p], line: [a.p, b.p], text: `${f3(Math.hypot(dz, dd / 2))} mm`, detail: `ΔZ ${f3(Math.abs(dz))} · ΔØ ${f3(Math.abs(dd))} (${a.text} → ${b.text})`, labels: [{ p: mid, text: `${f2(Math.abs(dz))} along Z` }] };
  }
  const d = [0, 1, 2].map((k) => b.p[k] - a.p[k]);
  const L = Math.hypot(...d);
  const mid = [0, 1, 2].map((k) => (a.p[k] + b.p[k]) / 2);
  return { kind: 'distance', points: [a.p, b.p], line: [a.p, b.p], text: `${f3(L)} mm`, detail: `ΔX ${f3(Math.abs(d[0]))} · ΔY ${f3(Math.abs(d[1]))} · ΔZ ${f3(Math.abs(d[2]))}`, labels: [{ p: mid, text: f2(L) }] };
}
