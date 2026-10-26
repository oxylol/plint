// Drives material removal along the timed move list and answers
// "where is the tool at time t". Also runs a full-program analysis.

import { MillSim } from './millsim.js';
import { LatheSim } from './lathesim.js';
import { IDENTITY, mulXform, rotX, rotY } from './voxelsim.js';

// part rotation for 4th/5th-axis words, through the work zero. ISO sense: +A moves the tool
// right-hand around +X relative to the part, so the table turns the part by -A (B likewise
// about +Y). reverse = machines wired the other way.
export function rotaryX(a, b, reverse = false) {
  const k = reverse ? 1 : -1;
  let X = IDENTITY;
  if (a) X = mulXform(rotX(k * a), X);
  if (b) X = mulXform(rotY(k * b), X);
  return X;
}

export function createSim(result, opts = {}) {
  return result.machine === 'lathe'
    ? new LatheSim(result.stock, result.tools)
    : new MillSim(result.stock, result.tools, { resolution: opts.resolution || 420 });
}

function cumLengths(m) {
  if (m.cum) return m.cum;
  const c = new Float64Array(m.pts.length);
  for (let i = 1; i < m.pts.length; i++) {
    const a = m.pts[i - 1], b = m.pts[i];
    c[i] = c[i - 1] + Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
  }
  m.cum = c;
  return c;
}

const lerp3 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

// XY distance between segment ab and an axis-aligned box's footprint (0 if they overlap)
function segRectDist(a, b, f) {
  const inside = (x, y) => x >= f.min[0] && x <= f.max[0] && y >= f.min[1] && y <= f.max[1];
  if (inside(a[0], a[1]) || inside(b[0], b[1])) return 0;
  const ptRect = (x, y) => Math.hypot(Math.max(f.min[0] - x, 0, x - f.max[0]), Math.max(f.min[1] - y, 0, y - f.max[1]));
  const ptSeg = (x, y) => {
    const dx = b[0] - a[0], dy = b[1] - a[1], L2 = dx * dx + dy * dy;
    const t = L2 > 1e-12 ? Math.max(0, Math.min(1, ((x - a[0]) * dx + (y - a[1]) * dy) / L2)) : 0;
    return Math.hypot(a[0] + t * dx - x, a[1] + t * dy - y);
  };
  // segment crossing the rectangle (Liang–Barsky)
  let t0 = 0, t1 = 1;
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const clip = (p, q) => { if (Math.abs(p) < 1e-12) return q >= 0; const r = q / p; if (p < 0) { if (r > t1) return false; if (r > t0) t0 = r; } else { if (r < t0) return false; if (r < t1) t1 = r; } return true; };
  if (clip(-dx, a[0] - f.min[0]) && clip(dx, f.max[0] - a[0]) && clip(-dy, a[1] - f.min[1]) && clip(dy, f.max[1] - a[1]) && t0 <= t1) return 0;
  return Math.min(ptRect(a[0], a[1]), ptRect(b[0], b[1]), ptSeg(f.min[0], f.min[1]), ptSeg(f.max[0], f.min[1]), ptSeg(f.min[0], f.max[1]), ptSeg(f.max[0], f.max[1]));
}

export class SimRunner {
  // opts.xform: part -> work transform of this setup (VoxelSim only)
  // opts.atEnd: the sim already holds the final state (skip the initial reset)
  constructor(result, sim, opts = {}) {
    this.result = result;
    this.moves = result.moves;
    this.sim = sim;
    this.xform = opts.xform || IDENTITY;
    this.rotReverse = !!opts.rotReverse;
    this.issues = [];
    if (opts.atEnd) {
      this.issueKeys = new Set();
      this.issueGroups = new Map();
      this.idx = this.moves.length;
      this.done = 0;
      this.time = this.moves.length ? this.moves[this.moves.length - 1].t1 + 1 : 0;
    } else {
      this.reset();
    }
  }

  // transform for a move at a given fraction (rotary axes interpolate)
  xformAt(m, f) {
    if (!m.rot) return this.xform;
    const a = m.rot[0] + (m.rot[1] - m.rot[0]) * f, b = m.rot[2] + (m.rot[3] - m.rot[2]) * f;
    return mulXform(rotaryX(a, b, this.rotReverse), this.xform);
  }

  reset() {
    this.sim.reset();
    this.idx = 0;
    this.done = 0;
    this.time = 0;
    this.issueKeys = new Set();
    this.issueGroups = new Map();
    this.issues = [];
  }

  // group: issues of one kind for one tool collapse into a single entry with a block count
  addIssue(level, text, m, key, group) {
    if (group) {
      const g = this.issueGroups.get(group);
      if (g) {
        if (!g.lines.has(m.callLine)) { g.lines.add(m.callLine); g.issue.count = g.lines.size; }
        if (g.worst !== undefined && g.value > g.worst) { g.worst = g.value; }
        return;
      }
      const issue = { level, text, line: m.line, file: m.file, callLine: m.callLine, source: 'sim', count: 1 };
      this.issueGroups.set(group, { issue, lines: new Set([m.callLine]) });
      this.issues.push(issue);
      return;
    }
    const k = key || `${text}|${m.callLine}`;
    if (this.issueKeys.has(k)) return;
    this.issueKeys.add(k);
    this.issues.push({ level, text, line: m.line, file: m.file, callLine: m.callLine, source: 'sim' });
  }

  // issues with "(N blocks)" suffixes for grouped entries
  finalIssues() {
    return this.issues.map((i) => (i.count > 1 ? { ...i, text: `${i.text} — first here, ${i.count} blocks in total` } : i));
  }

  applyRange(m, from, to) {
    if (m.kind !== 'rapid' && m.kind !== 'feed') return;
    if (to <= from) return;
    const voxel = this.sim.kind === 'voxel';
    // rotary moves: split so the part turns at most ~1° per piece
    const turning = voxel && m.rot && (m.rot[0] !== m.rot[1] || m.rot[2] !== m.rot[3]);
    if (turning || (voxel && m.rotOnly)) {
      const span = Math.max(Math.abs(m.rot[1] - m.rot[0]), Math.abs(m.rot[3] - m.rot[2]));
      const L = m.len || 1;
      const f0 = from / L, f1 = Math.min(1, to / L);
      const n = Math.max(1, Math.ceil(span * (f1 - f0)), Math.ceil(((f1 - f0) * L) / 2));
      for (let s = 0; s < n; s++) {
        const fa = f0 + ((f1 - f0) * s) / n, fb = f0 + ((f1 - f0) * (s + 1)) / n;
        const a = this.pointAt(m, fa), b = this.pointAt(m, fb);
        const res = this.sim.cutSegment(a, b, m, this.xformAt(m, (fa + fb) / 2));
        if (res) this.check(m, res, a, b);
        this.checkFixtures(m, a, b);
      }
      return;
    }
    const X = voxel ? this.xformAt(m, 0) : undefined;
    const cum = cumLengths(m);
    const pts = m.pts;
    for (let i = 1; i < pts.length; i++) {
      const s0 = cum[i - 1], s1 = cum[i];
      if (s1 <= from || s0 >= to) continue;
      const seg = s1 - s0 || 1;
      const a = s0 >= from ? pts[i - 1] : lerp3(pts[i - 1], pts[i], (from - s0) / seg);
      const b = s1 <= to ? pts[i] : lerp3(pts[i - 1], pts[i], (to - s0) / seg);
      const res = this.sim.cutSegment(a, b, m, X);
      this.checkFixtures(m, a, b);
      if (!res) continue;
      this.check(m, res, a, b);
    }
  }

  // tool vs fixtures (vise jaws): boxes in the work frame
  checkFixtures(m, a, b) {
    const fx = this.fixtures;
    if (!fx || !fx.length) return;
    const tool = this.result.tools[m.tool];
    if (!tool) return;
    const r = (tool.diameter || 0) / 2, Rh = (tool.holderDiameter || 32) / 2, Lo = tool.overallLength || 1e9;
    const zLow = Math.min(a[2], b[2]);
    for (const f of fx) {
      if (zLow >= f.max[2]) continue;
      const d = segRectDist(a, b, f);
      if (d < r - 0.01 || (zLow + Lo < f.max[2] && d < Rh - 0.01)) {
        this.addIssue('error', `T${m.tool} hits the ${f.name} — ${d < r ? 'the cutter' : 'the holder'} reaches below the jaw top (Z${f.max[2].toFixed(2)})`, m, null, `fix|${f.name}|${m.tool}`);
      }
    }
  }

  pointAt(m, f) {
    const cum = cumLengths(m);
    const L = cum[cum.length - 1];
    if (!(L > 0)) return m.pts[0].slice();
    const target = f * L;
    let k = 1;
    while (k < cum.length - 1 && cum[k] < target) k++;
    const s0 = cum[k - 1], s1 = cum[k];
    return lerp3(m.pts[k - 1], m.pts[k], s1 > s0 ? (target - s0) / (s1 - s0) : 1);
  }

  check(m, res, a, b) {
    const lathe = this.result.machine === 'lathe';
    const fmt = (p) => lathe ? `X${(p[0] * 2).toFixed(2)} Z${p[2].toFixed(2)}` : `X${p[0].toFixed(2)} Y${p[1].toFixed(2)} Z${p[2].toFixed(2)}`;
    if (m.kind === 'rapid' && (res.removedVol !== undefined ? res.removedVol > 0.5 : res.removed > 0.02)) {
      const amount = res.removedVol !== undefined ? `${res.removedVol.toFixed(1)} mm³ cut` : `${res.removed.toFixed(2)} mm cut`;
      this.addIssue('error', `Rapid move (G00) into material — ${amount} at rapid near ${fmt(b)}`, m);
    }
    if (res.shank > 0.05) {
      const t = this.result.tools[m.tool];
      this.addIssue('warning', `Tool T${m.tool} goes deeper than its ${t?.fluteLength ?? '?'} mm flutes — the shank rubs the part`, m, null, `shank|${m.tool}`);
    }
    if (res.holder > 0.1) {
      const t = this.result.tools[m.tool];
      const hint = lathe
        ? t?.type === 'groove' ? ` (blade is ${t.height ?? 16} mm long)` : t?.type === 'drill' ? ` (drill is ${t.length ?? '?'} mm long)` : ''
        : ` (stick-out ${t?.overallLength} mm)`;
      this.addIssue('error', `The holder of T${m.tool} hits the part — ${res.holder.toFixed(2)} mm interference${hint}`, m, null, `holder|${m.tool}`);
    }
    if (res.chuck) this.addIssue('error', `Tool goes past the chuck face (Z${(-this.result.stock.length).toFixed(1)}) — chuck collision`, m, 'chuck');
    if (!lathe && m.kind === 'feed' && res.removed > 0.05) {
      const tool = this.result.tools[m.tool];
      if (tool && ['drill', 'spot', 'tap'].includes(tool.type) && Math.hypot(b[0] - a[0], b[1] - a[1]) > 0.05) {
        this.addIssue('warning', `${tool.type === 'tap' ? 'Tap' : 'Drill'} T${m.tool} moves sideways while in material (is it really a ${tool.type}?)`, m, null, `side|${m.tool}`);
      }
    }
    if (m.kind === 'feed' && res.removed > 0.02 && m.spindle === 'off') {
      this.addIssue('error', 'Cutting with the spindle stopped', m, `spoff|${m.callLine}`);
    }
  }

  advanceTo(t) {
    if (t < this.time - 1e-9) this.reset();
    const moves = this.moves;
    while (this.idx < moves.length) {
      const m = moves[this.idx];
      if (m.t1 <= t) {
        this.applyRange(m, this.done, m.len);
        this.idx++;
        this.done = 0;
        continue;
      }
      if (m.t0 < t && m.t1 > m.t0) {
        const target = ((t - m.t0) / (m.t1 - m.t0)) * m.len;
        this.applyRange(m, this.done, target);
        this.done = Math.max(this.done, target);
      }
      break;
    }
    this.time = t;
  }

  runAll() {
    const end = this.moves.length ? this.moves[this.moves.length - 1].t1 : 0;
    this.advanceTo(end + 1);
  }

  moveIndexAt(t) {
    const mv = this.moves;
    if (!mv.length) return -1;
    let lo = 0, hi = mv.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (mv[mid].t1 < t) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  // tool position + active move at time t
  stateAt(t) {
    const i = this.moveIndexAt(t);
    if (i < 0) return { pos: this.result.home.slice(), move: null, index: -1, rot: null };
    const m = this.moves[i];
    if (m.kind !== 'rapid' && m.kind !== 'feed') return { pos: m.pts[0].slice(), move: m, index: i, rot: m.rot ? [m.rot[1], m.rot[3]] : null };
    const f = m.t1 > m.t0 ? Math.min(1, Math.max(0, (t - m.t0) / (m.t1 - m.t0))) : 1;
    const rot = m.rot ? [m.rot[0] + (m.rot[1] - m.rot[0]) * f, m.rot[2] + (m.rot[3] - m.rot[2]) * f] : null;
    const target = f * m.len;
    const cum = cumLengths(m);
    let k = 1;
    while (k < cum.length - 1 && cum[k] < target) k++;
    const s0 = cum[k - 1], s1 = cum[k];
    const pos = lerp3(m.pts[k - 1], m.pts[k], s1 > s0 ? (target - s0) / (s1 - s0) : 1);
    return { pos, move: m, index: i, rot };
  }
}

// Full analysis for the check panel and for Claude's tool results.
export function analyzeProgram(result, opts = {}) {
  const sim = createSim(result, { resolution: opts.resolution || 300 });
  const runner = new SimRunner(result, sim);
  runner.runAll();
  const lathe = result.machine === 'lathe';
  const issues = runner.finalIssues();
  if (lathe && sim.partedOff !== null) {
    issues.push({ level: 'info', text: `The part is cut off at Z${sim.partedOff.toFixed(2)} — everything beyond falls away`, line: null, file: 'main', callLine: null, source: 'sim' });
  }
  if (!lathe) {
    let through = 0;
    for (let k = 0; k < sim.heights.length; k++) if (sim.heights[k] < sim.bottom) through++;
    if (through) {
      issues.push({ level: 'info', text: `Cuts through the stock bottom (${((through * sim.cx * sim.cy) / 100).toFixed(1)} cm² area) — make sure the part is on parallels / a sacrificial plate`, line: null, file: 'main', callLine: null, source: 'sim' });
    }
  }
  // per-tool envelopes
  const perTool = {};
  for (const m of result.moves) {
    if (m.kind !== 'feed') continue;
    const t = (perTool[m.tool] ||= { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity], length: 0, time: 0 });
    for (const p of m.pts) for (let k = 0; k < 3; k++) { if (p[k] < t.min[k]) t.min[k] = p[k]; if (p[k] > t.max[k]) t.max[k] = p[k]; }
    t.length += m.len;
    t.time += m.t1 - m.t0;
  }
  return { sim, runner, issues, removedVolume: sim.removedVolume(), perTool };
}
