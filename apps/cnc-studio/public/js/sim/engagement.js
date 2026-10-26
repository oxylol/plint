// How much of the tool is in the material while it cuts, measured along every feed move on the
// material as it is just before that piece is cut. A tool sweeping a straight piece only meets new
// material with its leading side: on the mill the half of the cutter ahead of the motion, on the
// lathe what the insert covers that it did not cover a moment earlier. That is what is sampled;
// the rest of the tool is inside its own cut.
//   mill:  ae = radial width of cut (mm, % of Ø), angle = arc of the cutter in contact, ap = axial depth
//   lathe: ap = depth of cut across the motion, f = feed per rev
//   both:  q = material removal rate (cm³/min)
import { millProfile } from '../gcode/tools.js';
import { convexHull } from '../gcode/lathetools.js';
import { invertX } from './voxelsim.js';

const DRILLING = new Set(['drill', 'spot', 'tap']);
const D2R = Math.PI / 180;
const PHI = Array.from({ length: 19 }, (_, i) => -90 + 10 * i); // leading half, from the motion direction

export class EngagementRecorder {
  constructor(result, sim) {
    this.result = result;
    this.sim = sim;
    this.lathe = result.machine === 'lathe';
    this.cols = { t: [], move: [], len: [], ae: [], angle: [], ap: [], f: [], q: [], plunge: [] };
    this.index = new Map(result.moves.map((m, i) => [m, i]));
    this.shapes = new Map();
  }

  // a → b: one straight piece of move m about to be cut, s0 = its distance from the move start
  probe(a, b, m, X, s0) {
    if (m.kind !== 'feed') return;
    const tool = this.result.tools[m.tool];
    if (!tool) return;
    if (this.lathe) this.probeLathe(a, b, m, tool, s0);
    else this.probeMill(a, b, m, tool, X, s0);
  }

  push(m, s, len, v) {
    const c = this.cols;
    c.t.push(m.len > 0 ? m.t0 + ((m.t1 - m.t0) * s) / m.len : m.t0);
    c.move.push(this.index.get(m));
    c.len.push(len);
    c.ae.push(v.ae); c.angle.push(v.angle); c.ap.push(v.ap); c.f.push(v.f || 0); c.q.push(v.q); c.plunge.push(v.plunge ? 1 : 0);
  }

  // ---------------------------------------------------------------- mill
  probeMill(a, b, m, tool, X, s0) {
    if (DRILLING.has(tool.type) || !(tool.diameter > 0)) return;
    const dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
    const L = Math.hypot(dx, dy, dz), Lxy = Math.hypot(dx, dy);
    if (L < 1e-9) return;
    const plunge = Lxy < 0.05 * L;
    if (plunge && dz > 0) return; // straight up: nothing ahead of the tool
    const shape = this.millShape(tool);
    const look = this.materialLookup(X);
    const ux = Lxy > 1e-12 ? dx / Lxy : 1, uy = Lxy > 1e-12 ? dy / Lxy : 0;
    const n = Math.max(1, Math.ceil(L / Math.max(0.25, shape.R / 4)));
    for (let k = 0; k < n; k++) {
      const f = (k + 1) / n; // the end of each sub-piece: a plunge is deepest there
      const p = [a[0] + dx * f, a[1] + dy * f, a[2] + dz * f];
      const v = plunge ? this.plungeAt(p, shape, look) : this.sideAt(p, ux, uy, dz < -0.05 * L, shape, look);
      v.q = ((plunge ? (Math.PI * shape.R * shape.R * v.angle) / 360 : v.ae * v.ap) * (m.rate || 0)) / 1000;
      this.push(m, s0 + f * L, L / n, v);
    }
  }

  // the cutter's leading surface: points (rho from the axis, z above the tip). A flat end mill
  // leads with its rim only; balls, bull noses and cones also with the curved part of the bottom.
  millShape(tool) {
    let s = this.shapes.get(tool);
    if (s) return s;
    const R = tool.diameter / 2, prof = millProfile(tool);
    const radii = [{ rho: R * 0.999, z: prof(R * 0.999) }];
    for (const k of [0.85, 0.7, 0.5, 0.3]) if (prof(R * k) > 1e-6) radii.push({ rho: R * k, z: prof(R * k) });
    // going down, the bottom leads as well
    const withBottom = [...radii, ...[0.75, 0.5, 0.25].filter((k) => !(prof(R * k) > 1e-6)).map((k) => ({ rho: R * k, z: 0 }))];
    s = { R, prof, radii, withBottom, flute: tool.fluteLength || 4 * R };
    this.shapes.set(tool, s);
    return s;
  }

  // highest material at (x, y) between z0 and z1 (work frame), or -Infinity
  materialLookup(X) {
    const sim = this.sim;
    if (sim.kind === 'voxel') {
      const inv = invertX(X || sim.xform);
      const R = inv.R, T = inv.t;
      const solid = (x, y, z) => sim.sample(R[0] * x + R[1] * y + R[2] * z + T[0], R[3] * x + R[4] * y + R[5] * z + T[1], R[6] * x + R[7] * y + R[8] * z + T[2]) < 0;
      const zTop = sim.workBounds(X || sim.xform).max[2];
      const look = (x, y, z0, z1) => {
        z1 = Math.min(z1, zTop);
        if (z1 <= z0) return -Infinity;
        const st = Math.max(sim.h * 0.8, (z1 - z0) / 24);
        let best = -Infinity;
        for (let z = z0; z <= z1 + 1e-9; z += st) if (solid(x, y, z)) best = z;
        if (best === -Infinity) return best;
        let lo = best, hi = Math.min(z1, best + st);
        if (hi <= lo || solid(x, y, hi)) return hi;
        for (let k = 0; k < 5; k++) { const mid = (lo + hi) / 2; if (solid(x, y, mid)) lo = mid; else hi = mid; }
        return lo;
      };
      look.fine = false;
      return look;
    }
    const { nx, ny, cx, cy, minX, minY, heights, bottom } = sim;
    const look = (x, y, z0, z1) => {
      const u = (x - minX) / cx, v = (y - minY) / cy;
      if (u < 0 || v < 0 || u > nx - 1 || v > ny - 1) return -Infinity;
      const i = Math.min(nx - 2, Math.floor(u)), j = Math.min(ny - 2, Math.floor(v));
      const fu = u - i, fv = v - j, k = j * nx + i;
      const h = (q) => Math.max(heights[q], bottom - 1e-3);
      const top = (h(k) * (1 - fu) + h(k + 1) * fu) * (1 - fv) + (h(k + nx) * (1 - fu) + h(k + nx + 1) * fu) * fv;
      return top < bottom || top <= z0 || bottom >= z1 ? -Infinity : Math.min(top, z1);
    };
    look.fine = true; // cheap enough to pin the edges of the contact arc down by bisection
    return look;
  }

  sideAt(p, ux, uy, descending, shape, look) {
    const tip = p[2];
    const pts = descending ? shape.withBottom : shape.radii;
    const top = (deg) => {
      const c = Math.cos(deg * D2R), s = Math.sin(deg * D2R);
      const ex = ux * c - uy * s, ey = ux * s + uy * c;
      let best = -Infinity;
      for (const { rho, z } of pts) best = Math.max(best, look(p[0] + ex * rho, p[1] + ey * rho, tip + z + 0.005, tip + shape.flute));
      return best;
    };
    const tops = PHI.map(top);
    const on = tops.map((h) => h > -Infinity);
    if (!on.some(Boolean)) return { ae: 0, angle: 0, ap: 0 };
    const edge = (i, j) => {
      if (!look.fine) return (PHI[i] + PHI[j]) / 2;
      let lo = PHI[i], hi = PHI[j];
      for (let k = 0; k < 5; k++) { const mid = (lo + hi) / 2; if ((top(mid) > -Infinity) === on[i]) lo = mid; else hi = mid; }
      return (lo + hi) / 2;
    };
    let angle = 0, from = Infinity, to = -Infinity, ap = 0;
    for (let i = 0; i < PHI.length; i++) {
      if (!on[i]) continue;
      ap = Math.max(ap, tops[i] - tip);
      if (i > 0 && on[i - 1]) continue;
      let j = i;
      while (j + 1 < PHI.length && on[j + 1]) j++;
      const a = i === 0 ? -90 : edge(i - 1, i), b = j === PHI.length - 1 ? 90 : edge(j, j + 1);
      angle += b - a;
      from = Math.min(from, a);
      to = Math.max(to, b);
    }
    return { ae: shape.R * (Math.sin(to * D2R) - Math.sin(from * D2R)), angle, ap };
  }

  // straight down into material: the whole bottom leads
  plungeAt(p, shape, look) {
    const tip = p[2];
    let hit = 0, n = 0, ap = 0;
    const at = (x, y, z) => { n++; const h = look(x, y, tip + z + 0.005, tip + shape.flute); if (h > -Infinity) { hit++; ap = Math.max(ap, h - tip); } };
    at(p[0], p[1], shape.prof(0));
    for (let d = 0; d < 360; d += 30) for (const k of [0.5, 0.999]) at(p[0] + Math.cos(d * D2R) * shape.R * k, p[1] + Math.sin(d * D2R) * shape.R * k, shape.prof(shape.R * k));
    return hit ? { ae: 2 * shape.R, angle: (360 * hit) / n, ap, plunge: true } : { ae: 0, angle: 0, ap: 0, plunge: true };
  }

  // ---------------------------------------------------------------- lathe
  probeLathe(a, b, m, tool, s0) {
    if (m.thread > 0 || !tool.cutter || tool.cutter.length < 3) return; // threading passes are not chips of this kind
    const s = this.sim;
    const [za, ra] = s.refPoint(tool, a, m.center), [zb, rb] = s.refPoint(tool, b, m.center);
    const dz = zb - za, dr = rb - ra, L = Math.hypot(dz, dr);
    if (L < 1e-9) return;
    let hull = this.shapes.get(tool);
    if (!hull) { hull = hullSpans(convexHull(tool.cutter)); this.shapes.set(tool, hull); }
    const uz = dz / L, ur = dr / L;
    const feed = m.feedPerRev ? m.F : m.rpm > 0 ? (m.rate || 0) / m.rpm : 0;
    const n = Math.max(1, Math.ceil(L / 0.4));
    for (let k = 0; k < n; k++) {
      const f = (k + 1) / n;
      const z = za + dz * f, r = ra + dr * f;
      const ap = this.latheDepth(hull, z, r, uz, ur);
      const vc = (2 * Math.PI * Math.max(0, r) * (m.rpm || 0)) / 1000; // m/min at the tool point
      this.push(m, s0 + f * L, L / n, { ae: 0, angle: 0, ap, f: feed, q: ap * feed * vc });
    }
  }

  // what the insert at (z, r) covers that it did not 0.05 mm back along the motion, and that is
  // material: its extent across the motion is the depth of cut
  latheDepth(hull, z, r, uz, ur) {
    const s = this.sim;
    const back = 0.05;
    const zb = z - uz * back, rb = r - ur * back;
    const i0 = Math.max(0, Math.ceil((z + hull.z0 - s.zMin) / s.dz)), i1 = Math.min(s.n - 1, Math.floor((z + hull.z1 - s.zMin) / s.dz));
    let lo = Infinity, hi = -Infinity;
    const mark = (zc, x) => { const q = zc * -ur + x * uz; lo = Math.min(lo, q); hi = Math.max(hi, q); };
    for (let i = i0; i <= i1; i++) {
      if (s.outer[i] - s.inner[i] < 1e-4) continue;
      const zc = s.zMin + i * s.dz;
      const a = hull.at(zc - z, r);
      if (!a) continue;
      const m0 = Math.max(a[0], s.inner[i]), m1 = Math.min(a[1], s.outer[i]);
      if (m1 - m0 < 1e-4) continue;
      const b = hull.at(zc - zb, rb);
      for (const [p0, p1] of b ? [[m0, Math.min(m1, b[0])], [Math.max(m0, b[1]), m1]] : [[m0, m1]]) {
        if (p1 - p0 > 1e-4) { mark(zc, p0); mark(zc, p1); }
      }
    }
    return hi > lo ? hi - lo : 0;
  }

  // ---------------------------------------------------------------- results
  finish() {
    const c = this.cols, moves = this.result.moves, tools = this.result.tools;
    const perTool = {}, perLine = new Map();
    for (let k = 0; k < c.t.length; k++) {
      const m = moves[c.move[k]];
      const tool = tools[m.tool];
      const on = c.ap[k] > 1e-3 || c.angle[k] > 0;
      const T = (perTool[m.tool] ||= { tool: m.tool, len: 0, cut: 0, slot: 0, plunge: 0, max: {}, ae: [], ap: [] });
      T.len += c.len[k];
      const keyL = `${m.callLine}|${m.file}|${m.line}`;
      let Lr = perLine.get(keyL);
      if (!Lr) { Lr = { callLine: m.callLine, file: m.file, line: m.line, tool: m.tool, len: 0, cut: 0, ae: 0, angle: 0, ap: 0, f: 0, q: 0, plunge: false, first: k }; perLine.set(keyL, Lr); }
      Lr.len += c.len[k];
      if (!on) continue;
      T.cut += c.len[k];
      Lr.cut += c.len[k];
      if (c.plunge[k]) { T.plunge += c.len[k]; Lr.plunge = true; }
      else if (tool?.diameter && c.ae[k] >= 0.95 * tool.diameter) T.slot += c.len[k];
      T.ae.push([c.ae[k], c.len[k]]);
      T.ap.push([c.ap[k], c.len[k]]);
      for (const f of ['ae', 'angle', 'ap', 'f', 'q']) {
        if (!(T.max[f]?.v >= c[f][k])) T.max[f] = { v: c[f][k], k };
        Lr[f] = Math.max(Lr[f], c[f][k]);
      }
    }
    for (const T of Object.values(perTool)) { T.aeTyp = weightedMedian(T.ae); T.apTyp = weightedMedian(T.ap); delete T.ae; delete T.ap; }
    return { machine: this.result.machine, samples: c, perTool, lines: [...perLine.values()] };
  }
}

// a convex insert outline as its lower and upper chains (both by increasing z): at(z, r) is the
// [lo, hi] radius it covers on the line z when its reference point is at radius r (a tool that
// crosses the axis covers from 0)
function hullSpans({ lower, upper }) {
  const along = (chain, z) => {
    let lo = 0, hi = chain.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (chain[mid][0] <= z) lo = mid; else hi = mid; }
    const [za, xa] = chain[lo], [zb, xb] = chain[hi];
    return zb > za ? xa + ((z - za) * (xb - xa)) / (zb - za) : Math.min(xa, xb);
  };
  const z0 = lower[0][0], z1 = lower[lower.length - 1][0];
  return {
    z0, z1,
    at(z, r) {
      if (z < z0 || z > z1 || lower.length < 2) return null;
      const lo = along(lower, z) + r, hi = along(upper, z) + r;
      return lo < 0 ? [0, Math.max(-lo, hi)] : [lo, hi];
    },
  };
}

function weightedMedian(pairs) {
  if (!pairs.length) return 0;
  pairs.sort((a, b) => a[0] - b[0]);
  const half = pairs.reduce((s, p) => s + p[1], 0) / 2;
  let acc = 0;
  for (const [v, w] of pairs) { acc += w; if (acc >= half) return v; }
  return pairs[pairs.length - 1][0];
}

// ---------------------------------------------------------------- what Claude (and the Check tab) get
const fx = (v, d = 2) => { const s = (Math.abs(v) < 0.5 * 10 ** -d ? 0 : v).toFixed(d); return s.includes('.') ? s.replace(/\.?0+$/, '') || '0' : s; };
const toolLabel = (t, n) => `T${n}${t?.diameter ? ` Ø${fx(t.diameter)}` : ''}${t?.type ? ` ${t.type}` : ''}`;
const where = (L) => (L.file && L.file !== 'main' ? `line ${L.callLine + 1} (in ${L.file} line ${L.line + 1})` : `line ${L.callLine + 1}`);

function header(lathe) {
  return lathe
    ? 'Tool engagement (the insert\'s leading edge on the material just before each cut; ap = depth of cut across the motion, f = feed per rev; threading passes not included):'
    : 'Tool engagement (the cutter\'s leading side on the material just before each cut; ae = radial width of cut, contact = arc of the cutter in the material, ap = axial depth; drills and taps not included):';
}

// per-line lookups for the maxima, built once
function indexLines(eng) {
  if (eng.lineOf) return;
  eng.lineOf = new Map();
  const m2l = new Map();
  for (const L of eng.lines) m2l.set(`${L.callLine}|${L.file}|${L.line}`, L);
  eng.sampleLine = (k, moves) => { const m = moves[eng.samples.move[k]]; return m2l.get(`${m.callLine}|${m.file}|${m.line}`); };
}

export function engagementReport(out, { detail = false } = {}) {
  const eng = out.engagement;
  if (!eng) return '';
  indexLines(eng);
  const moves = out.result.moves, tools = out.result.tools;
  const lathe = eng.machine === 'lathe';
  const src = out.op.text.split('\n');
  const code = (L) => (L.file === 'main' ? ` \`${(src[L.callLine] || '').trim().slice(0, 48)}\`` : '');
  const at = (mx) => (mx ? where(eng.sampleLine(mx.k, moves)) : '?');
  const o = [header(lathe)];
  const T = Object.values(eng.perTool).filter((t) => t.len > 0).sort((a, b) => a.tool - b.tool);
  if (!T.length) return '';
  for (const t of T) {
    const tool = tools[t.tool], D = tool?.diameter || 0;
    const air = t.len > 0 ? Math.round((100 * (t.len - t.cut)) / t.len) : 0;
    const feedShare = `cutting ${fx(t.cut, 0)} of ${fx(t.len, 0)} mm of feed${air ? ` (${air}% in air)` : ''}`;
    if (!t.cut) { o.push(`  ${toolLabel(tool, t.tool)}: never in material (${fx(t.len, 0)} mm of feed in air)`); continue; }
    if (lathe) {
      const chip = t.max.ap.v * (t.max.f?.v || 0);
      o.push(`  ${toolLabel(tool, t.tool)}: ap up to ${fx(t.max.ap.v)} mm at ${at(t.max.ap)}, typical ${fx(t.apTyp)} mm; f up to ${fx(t.max.f?.v || 0, 3)} mm/rev (chip area up to ${fx(chip, 3)} mm²); removal up to ${fx(t.max.q.v, 1)} cm³/min (${at(t.max.q)}); ${feedShare}`);
      continue;
    }
    const pct = (v) => (D ? `${Math.round((100 * v) / D)}%` : '');
    const extra = [t.slot > 0.5 ? `full-width slotting ${fx(t.slot, 0)} mm` : '', t.plunge > 0.2 ? `plunging ${fx(t.plunge, 1)} mm` : ''].filter(Boolean).join(', ');
    o.push(`  ${toolLabel(tool, t.tool)}: ae up to ${fx(t.max.ae.v)} mm (${pct(t.max.ae.v)}${t.max.ae.v >= 0.95 * D ? ', full slot' : ''}, contact ${Math.round(t.max.angle.v)}°) at ${at(t.max.ae)}; ap up to ${fx(t.max.ap.v)} mm${D ? ` (${fx(t.max.ap.v / D)}×Ø)` : ''} at ${at(t.max.ap)}; typical ae ${pct(t.aeTyp)} / ap ${fx(t.apTyp)} mm${extra ? `; ${extra}` : ''}; removal up to ${fx(t.max.q.v, 1)} cm³/min (${at(t.max.q)}); ${feedShare}`);
    // the heaviest lines: the largest cross-section of cut (ae × ap)
    const heavy = eng.lines.filter((L) => L.tool === t.tool && L.cut > 0 && !L.plunge).sort((a, b) => b.ae * b.ap - a.ae * a.ap).slice(0, detail ? 0 : 3);
    if (heavy.length) o.push(`    heaviest: ${heavy.map((L) => `${where(L)}${code(L)} ae ${pct(L.ae)} ap ${fx(L.ap)}`).join(' · ')}`);
  }
  if (!detail) return o.join('\n');
  // every line with feed moves, in program order
  o.push('Per program line (maxima over the line):');
  const rows = eng.lines.filter((L) => L.len > 0).sort((a, b) => a.callLine - b.callLine || a.first - b.first);
  for (const L of rows.slice(0, 120)) {
    const tool = tools[L.tool], D = tool?.diameter || 0;
    if (!L.cut) { o.push(`  ${where(L)} T${L.tool}${code(L)}: in air (${fx(L.len, 1)} mm)`); continue; }
    const share = L.cut < L.len - 0.05 ? ` — ${fx(L.cut, 1)} of ${fx(L.len, 1)} mm in material` : ` — ${fx(L.len, 1)} mm`;
    o.push(lathe
      ? `  ${where(L)} T${L.tool}${code(L)}: ap ${fx(L.ap)} mm, f ${fx(L.f, 3)}, up to ${fx(L.q, 1)} cm³/min${share}`
      : `  ${where(L)} T${L.tool}${code(L)}: ${L.plunge ? `plunge, ap ${fx(L.ap)} mm` : `ae ${fx(L.ae)} mm (${D ? Math.round((100 * L.ae) / D) : '?'}%, ${Math.round(L.angle)}°), ap ${fx(L.ap)} mm${D ? ` (${fx(L.ap / D)}×Ø)` : ''}`}, up to ${fx(L.q, 1)} cm³/min${share}`);
  }
  if (rows.length > 120) o.push(`  … ${rows.length - 120} more lines`);
  return o.join('\n');
}

// engagement at playback time t (the sample just before t on the move being run), or null
export function engagementAt(eng, t, moveIndex) {
  const s = eng?.samples;
  if (!s || !s.t.length) return null;
  let lo = 0, hi = s.t.length - 1;
  if (s.t[0] > t) return null;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (s.t[mid] <= t) lo = mid; else hi = mid - 1; }
  if (s.move[lo] !== moveIndex) return s.move[lo + 1] === moveIndex ? { k: lo + 1, ...pick(s, lo + 1) } : null;
  return { k: lo, ...pick(s, lo) };
}
const pick = (s, k) => ({ ae: s.ae[k], angle: s.angle[k], ap: s.ap[k], f: s.f[k], q: s.q[k], plunge: !!s.plunge[k] });
