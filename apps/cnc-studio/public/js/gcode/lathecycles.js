// Fanuc lathe cycles (Group A): G70 G71 G72 G73 G74 G75 G76, G90 G92 G94.
// All geometry in (radius, z); api.rapid/feed take (r, z).

import { tessellate } from './contour.js';

function polyline(prims) {
  const pts = [];
  for (const p of prims) {
    const t = tessellate(p, 0.01);
    if (!pts.length) pts.push([t[0][0], t[0][2]]);
    for (let i = 1; i < t.length; i++) pts.push([t[i][0], t[i][2]]);
  }
  return pts;
}

const sgn = (v) => (v > 1e-9 ? 1 : v < -1e-9 ? -1 : 0);

// G71 (turning, levels in X) / G72 (facing, levels in Z) / G73 (pattern repeat)
export function latheRoughing(api, code, start, prims, o) {
  if (!prims.length) { api.error('Empty profile between P and Q'); return; }
  const S = [start[0], start[2]];
  const extra = { ...o.extra, keepCycleFeed: true };

  if (code === 73) {
    const n = o.div;
    for (let i = 1; i <= n; i++) {
      const k = n === 1 ? 0 : (n - i) / (n - 1);
      const sh = [o.du + o.relief[0] * k, o.dw + o.relief[1] * k];
      const A = prims[0].b;
      api.rapid(A[0] + sh[0], A[2] + sh[1], extra);
      api.emitPrims(prims.slice(1), sh, extra, 'feed');
      const p = api.pos();
      api.rapid(S[0], p[2], extra);
      api.rapid(S[0], S[1], extra);
    }
    return;
  }

  // shifted contour without the approach move (starts at A')
  const raw = polyline(prims);
  const contour = raw.slice(1).map(([r, z]) => [r + o.du, z + o.dw]);
  if (contour.length < 2) { api.error('Profile is too short for roughing'); return; }
  const toLC = code === 71 ? (p) => [p[0], p[1]] : (p) => [p[1], p[0]];
  const fromLC = code === 71 ? (lv, ct) => [lv, ct] : (lv, ct) => [ct, lv];
  const C = contour.map(toLC);
  const s = toLC(S);
  const A = C[0];
  const sL = sgn(A[0] - s[0]);
  if (!sL) {
    api.error(`${code === 71 ? 'G71' : 'G72'}: the start point must be outside the profile start (${code === 71 ? 'X' : 'Z'})`);
    return;
  }
  let meanCt = 0;
  for (const p of C) meanCt += p[1];
  meanCt /= C.length;
  const sC = sgn(meanCt - s[1]) || -1;

  // monotonic check (type I)
  let reversals = 0;
  for (let i = 1; i < C.length; i++) if (sL * (C[i][0] - C[i - 1][0]) > 1e-6) reversals++;
  if (reversals) api.warn(`${code === 71 ? 'G71' : 'G72'}: profile is not monotonic — pockets (type II) are left for the finishing pass`);

  const go = (kind, lv, ct) => { const [r, z] = fromLC(lv, ct); kind === 'rapid' ? api.rapid(r, z, extra) : api.feed(r, z, extra); };
  const d = o.depth, e = o.retract;
  let lastRetract = null;
  for (let k = 1; k < 5000; k++) {
    const L = s[0] + sL * d * k;
    if (sL * (A[0] - L) <= 1e-6) break;
    let ctEnd = C[C.length - 1][1];
    for (let i = 0; i < C.length - 1; i++) {
      const dp = sL * (C[i][0] - L), dq = sL * (C[i + 1][0] - L);
      if (dp > 0 && dq <= 0) {
        const t = dp / (dp - dq);
        ctEnd = C[i][1] + (C[i + 1][1] - C[i][1]) * t;
        break;
      }
    }
    if (sC * (ctEnd - s[1]) <= 1e-6) continue;
    go('rapid', L, s[1]);
    go('feed', L, ctEnd);
    go('feed', L - sL * e, ctEnd - sC * e);
    go('rapid', L - sL * e, s[1]);
    lastRetract = L - sL * e;
  }
  // final pass along the shifted contour
  const [ar, az] = fromLC(A[0], A[1]);
  if (lastRetract !== null) go('rapid', A[0], s[1]);
  api.rapid(ar, az, extra);
  api.emitPrims(prims.slice(1), [o.du, o.dw], extra, 'feed');
  const p = toLC([api.pos()[0], api.pos()[2]]);
  go('rapid', s[0], p[1]);
  go('rapid', s[0], s[1]);
}

// G70 finishing along the profile, then back to the start point
export function latheFinish(api, start, prims, extra) {
  api.emitPrims(prims, [0, 0], { ...extra, profileLines: true, withComp: true }, null);
  const p = api.pos();
  if (Math.abs(p[0] - start[0]) > 1e-6) api.rapid(start[0], p[2], extra);
  api.rapid(start[0], start[2], extra);
}

// G90 turning / G94 facing box cycles (modal)
export function latheBoxCycle(api, code, start, target, R) {
  const [r0, z0] = [start[0], start[2]];
  const [rx, zz] = [target[0], target[2]];
  const extra = { cycle: code === 90 ? 'G90 turning cycle' : 'G94 facing cycle' };
  if (code === 90) {
    api.rapid(rx + R, z0, extra);
    api.feed(rx, zz, extra);
    api.feed(r0, zz, extra);
    api.rapid(r0, z0, extra);
  } else {
    api.rapid(r0, zz + R, extra);
    api.feed(rx, zz, extra);
    api.feed(rx, z0, extra);
    api.rapid(r0, z0, extra);
  }
}

// G92 threading cycle (modal): F is the lead
export function latheThreadCycle(api, start, target, R) {
  const [r0, z0] = [start[0], start[2]];
  const [rx, zz] = [target[0], target[2]];
  const extra = { cycle: 'G92 thread cycle', feedPerRev: true, thread: true };
  api.rapid(rx + R, z0, extra);
  api.feed(rx, zz, extra);
  api.rapid(r0, zz, extra);
  api.rapid(r0, z0, extra);
}

// G74 face peck drilling / face grooving, G75 OD grooving
export function lathePeck(api, code, start, target, o) {
  const [r0, z0] = [start[0], start[2]];
  const [rx, zz] = [target[0], target[2]];
  const extra = o.extra;
  const peck = (axis, from, to, step, fixed) => {
    // axis 'z': move in z at fixed r ; axis 'r': move in r at fixed z
    const dir = sgn(to - from);
    if (!dir) return;
    const at = (v) => (axis === 'z' ? [fixed, v] : [v, fixed]);
    let cur = from;
    let guard = 0;
    while (dir * (to - cur) > 1e-9 && guard++ < 2000) {
      const next = step > 0 ? (dir > 0 ? Math.min(to, cur + step) : Math.max(to, cur - step)) : to;
      const [r1, z1] = at(next);
      api.feed(r1, z1, extra);
      cur = next;
      if (dir * (to - cur) > 1e-9 && o.e > 0) {
        const [r2, z2] = at(cur - dir * o.e);
        api.rapid(r2, z2, extra);
        const [r3, z3] = at(cur);
        api.rapid(r3, z3, extra);
      }
    }
    const [rb, zb] = at(from);
    api.rapid(rb, zb, extra);
  };
  if (code === 74) {
    const cols = [];
    if (Math.abs(rx - r0) < 1e-6 || !(o.di > 0)) cols.push(Math.abs(rx - r0) < 1e-6 ? r0 : rx);
    else {
      const dir = sgn(rx - r0);
      for (let r = r0; dir * (rx - r) > -1e-9; r += dir * o.di) cols.push(dir * (rx - r) < o.di ? r : r);
      if (Math.abs(cols[cols.length - 1] - rx) > 1e-6) cols.push(rx);
    }
    for (const r of cols) {
      api.rapid(r, z0, extra);
      peck('z', z0, zz, o.dk, r);
    }
  } else {
    const rows = [];
    if (Math.abs(zz - z0) < 1e-6 || !(o.dk > 0)) rows.push(Math.abs(zz - z0) < 1e-6 ? z0 : zz);
    else {
      const dir = sgn(zz - z0);
      for (let z = z0; dir * (zz - z) > -1e-9; z += dir * o.dk) rows.push(z);
      if (Math.abs(rows[rows.length - 1] - zz) > 1e-6) rows.push(zz);
    }
    for (const z of rows) {
      api.rapid(r0, z, extra);
      peck('r', r0, rx, o.di, z);
    }
  }
  api.rapid(r0, z0, extra);
}

// G76 multi-pass threading
export function latheG76(api, start, target, o) {
  const [r0, z0] = [start[0], start[2]];
  const [rx, zz] = [target[0], target[2]];
  const extra = { ...o.extra, feedPerRev: true, thread: true };
  if (!(o.height > 0) || !(o.firstDepth > 0)) { api.error('G76: thread height (P) and first depth (Q) must be > 0'); return; }
  if (!(o.lead > 0)) { api.error('G76: no lead (F)'); return; }
  const side = r0 > rx ? 1 : -1; // +1 external thread
  const crest = rx + side * o.height;
  const rough = Math.max(0, o.height - o.finishAllow);
  const depths = [];
  let prev = 0;
  for (let n = 1; n < 200; n++) {
    let dn = o.firstDepth * Math.sqrt(n);
    if (dn - prev < o.minDepth) dn = prev + o.minDepth;
    if (dn >= rough - 1e-9) { depths.push(rough); break; }
    depths.push(dn);
    prev = dn;
  }
  for (let i = 0; i < Math.max(1, o.finishPasses); i++) depths.push(o.height);
  const oldF = api.setFeed(o.lead);
  for (const dpt of depths) {
    const r = crest - side * dpt;
    api.rapid(r + o.taper, z0, extra);
    api.feed(r, zz, extra);
    api.rapid(r0, zz, extra);
    api.rapid(r0, z0, extra);
  }
  api.setFeed(oldF);
}
