// Threads an operation cuts, read from its moves (work coordinates): lathe thread passes
// (G76 / G92 / G32), taps (G84 / G74) and internal thread milling (helical moves).
// The simulator cuts them as plain shapes — rings at the pitch on the lathe, a plain bore of
// the major diameter on the mill — so comparisons and reports need to know where they are.

const fx = (v, d = 3) => { const s = (Math.abs(v) < 0.5 * 10 ** -d ? 0 : v).toFixed(d); return s.includes('.') ? s.replace(/\.?0+$/, '') || '0' : s; };

// ISO metric coarse series: nominal Ø, pitch
const ISO_COARSE = [[1, 0.25], [1.2, 0.25], [1.6, 0.35], [2, 0.4], [2.5, 0.45], [3, 0.5], [4, 0.7], [5, 0.8], [6, 1], [8, 1.25], [10, 1.5], [12, 1.75], [14, 2], [16, 2], [18, 2.5], [20, 2.5], [22, 2.5], [24, 3], [27, 3], [30, 3.5], [33, 3.5], [36, 4], [39, 4], [42, 4.5], [45, 4.5], [48, 5], [52, 5], [56, 5.5], [60, 5.5], [64, 6]];

// metric nominal size for a measured major diameter (a turned major sits a little under nominal)
export function metricNominal(major, { under = 0.5, over = 0.05 } = {}) {
  const hit = ISO_COARSE.find(([d]) => major >= d - under && major <= d + over);
  return hit ? hit[0] : null;
}

export const designation = (nominal, major, pitch) => (nominal ? `M${fx(nominal)}×${fx(pitch)}` : `Ø${fx(major)}×${fx(pitch)}`);
export const tapDrill = (d, pitch) => Math.round((d - pitch) * 10) / 10;
export const minorInternal = (d, pitch) => d - 1.0825 * pitch; // ISO basic minor (D1)

// "tap drill for M6" when a hole diameter is the usual coarse tap drill of some size
export function tapDrillOf(d, tol = 0.06) {
  const hit = ISO_COARSE.find(([D, P]) => Math.abs(tapDrill(D, P) - d) <= tol);
  return hit ? `M${fx(hit[0])}` : null;
}

const cycleCode = (m, fallback) => /G\d+/.exec(m.cycle || '')?.[0] || fallback;

export function programThreads(out, field = null) {
  if (out.threads) return out.threads;
  if (!out.result) return []; // a target model described as a part: no program
  const moves = out.result.moves || [];
  out.threads = out.engine === 'lathe' ? latheThreads(out, moves) : [...tappedHoles(out, moves), ...milledThreads(out, moves, field)];
  return out.threads;
}

// ---------------------------------------------------------------- lathe
// passes of one thread: same tool and lead, overlapping Z ranges
function latheThreads(out, moves) {
  const groups = [];
  moves.forEach((m, k) => {
    if (m.kind !== 'feed' || !(m.thread > 0)) return;
    const a = m.pts[0], b = m.pts[m.pts.length - 1];
    const lo = Math.min(a[2], b[2]), hi = Math.max(a[2], b[2]);
    if (hi - lo < 1e-6) return;
    let g = groups.find((x) => x.tool === m.tool && Math.abs(x.lead - m.thread) < 1e-9 && lo <= x.hi + 1e-6 && hi >= x.lo - 1e-6);
    if (!g) { g = { tool: m.tool, lead: m.thread, cycle: cycleCode(m, 'G32'), lo, hi, radii: [], out: 0 }; groups.push(g); }
    g.lo = Math.min(g.lo, lo); g.hi = Math.max(g.hi, hi);
    g.radii.push((a[0] + b[0]) / 2);
    // the retract after a pass goes outward on an external thread, toward the axis in a bore
    const next = moves[k + 1];
    if (next) g.out += Math.sign(next.pts[next.pts.length - 1][0] - b[0]);
  });
  return groups.map((g) => latheThread(out, g)).filter(Boolean);
}

function latheThread(out, g) {
  const s = out.final;
  const tool = out.result.tools?.[g.tool];
  const internal = g.out < 0;
  // the tip of the insert relative to the tool's reference point (radial)
  const tip = tool?.cutter?.length ? (internal ? Math.max(...tool.cutter.map((c) => c[1])) : Math.min(...tool.cutter.map((c) => c[1]))) : 0;
  const rootR = (internal ? Math.max(...g.radii) : Math.min(...g.radii)) + tip;
  const arr = internal ? s.inner : s.outer;
  // where the tool really engaged material: inside the pass range, material beyond the root
  const zs = [], rs = [];
  for (let i = 0; i < s.n; i++) {
    const z = s.zMin + i * s.dz;
    if (z < g.lo - 1e-9 || z > g.hi + 1e-9 || s.outer[i] - s.inner[i] < 1e-4) continue;
    if (internal ? arr[i] < rootR - 0.01 : arr[i] > rootR + 0.01) { zs.push(z); rs.push(arr[i]); }
  }
  if (zs.length < 3) return null;
  // crest: the median of the per-lead extremes (a shoulder in the range does not count)
  const peaks = [];
  for (let z0 = zs[0]; z0 <= zs[zs.length - 1]; z0 += g.lead) {
    const w = rs.filter((_, k) => zs[k] >= z0 && zs[k] < z0 + g.lead);
    if (w.length) peaks.push(internal ? Math.min(...w) : Math.max(...w));
  }
  peaks.sort((a, b) => a - b);
  const crestR = peaks[Math.floor(peaks.length / 2)];
  const full = zs.filter((_, k) => Math.abs(rs[k] - crestR) <= 0.02);
  const major = 2 * (internal ? rootR : crestR), minor = 2 * (internal ? crestR : rootR);
  const nominal = metricNominal(major, internal ? { under: 0.05, over: 0.5 } : {});
  return {
    kind: 'lathe', internal, tool: g.tool, cycle: g.cycle, pitch: g.lead,
    rootR, crestR, major, minor, nominal, name: designation(nominal, major, g.lead),
    zone: [zs[0], zs[zs.length - 1]], // engaged range (low, high)
    full: full.length ? [full[0], full[full.length - 1]] : [zs[0], zs[zs.length - 1]],
  };
}

// ---------------------------------------------------------------- mill
// taps: vertical feed moves of a tap tool, one thread per hole position
function tappedHoles(out, moves) {
  const tools = out.result.tools || {};
  const holes = new Map();
  for (const m of moves) {
    const tool = tools[m.tool];
    if (m.kind !== 'feed' || tool?.type !== 'tap') continue;
    const a = m.pts[0], b = m.pts[m.pts.length - 1];
    if (Math.hypot(b[0] - a[0], b[1] - a[1]) > 0.01) continue;
    const key = `${m.tool}|${a[0].toFixed(2)}|${a[1].toFixed(2)}`;
    let h = holes.get(key);
    const pitch = tool.pitch > 0 ? tool.pitch : m.feedPerRev ? m.F : m.rpm > 0 ? m.F / m.rpm : 0;
    if (!h) { h = { kind: 'tap', tool: m.tool, cycle: cycleCode(m, 'tap'), x: a[0], y: a[1], d: tool.diameter, pitch, zTip: Infinity, zTop: -Infinity }; holes.set(key, h); }
    h.zTip = Math.min(h.zTip, a[2], b[2]);
    h.zTop = Math.max(h.zTop, a[2], b[2]);
  }
  return [...holes.values()].filter((h) => h.d > 0 && h.pitch > 0).map(holeThread);
}

// thread mills: helical arcs of a threadmill tool around one centre (internal threads only:
// a helix whose centre is in the material is an external thread, not handled here)
function milledThreads(out, moves, field) {
  const tools = out.result.tools || {};
  const helices = new Map();
  for (const m of moves) {
    const tool = tools[m.tool];
    if (m.kind !== 'feed' || !m.arc || tool?.type !== 'threadmill' || m.pts.length < 3) continue;
    const a = m.pts[0], b = m.pts[m.pts.length - 1];
    if (Math.abs(b[2] - a[2]) < 1e-6) continue;
    const c = circle3(a, m.pts[Math.floor(m.pts.length / 3)], m.pts[Math.floor((2 * m.pts.length) / 3)]); // a full circle ends where it starts
    if (!c) continue;
    let sweep = 0;
    for (let k = 1; k < m.pts.length; k++) {
      let d = Math.atan2(m.pts[k][1] - c.y, m.pts[k][0] - c.x) - Math.atan2(m.pts[k - 1][1] - c.y, m.pts[k - 1][0] - c.x);
      d -= 2 * Math.PI * Math.round(d / (2 * Math.PI));
      sweep += Math.abs(d);
    }
    const key = `${m.tool}|${c.x.toFixed(2)}|${c.y.toFixed(2)}`;
    let h = helices.get(key);
    if (!h) { h = { kind: 'threadmill', tool: m.tool, cycle: 'helix', x: c.x, y: c.y, rPath: c.r, dz: 0, sweep: 0, zTip: Infinity, zTop: -Infinity }; helices.set(key, h); }
    h.rPath = Math.max(h.rPath, c.r);
    h.dz += Math.abs(b[2] - a[2]);
    h.sweep += sweep;
    h.zTip = Math.min(h.zTip, a[2], b[2]);
    h.zTop = Math.max(h.zTop, a[2], b[2]);
  }
  const out2 = [];
  for (const h of helices.values()) {
    const D = tools[h.tool].diameter;
    if (!(h.sweep > 0) || !(D > 0)) continue;
    if (field?.inside([h.x, h.y, (h.zTip + h.zTop) / 2])) continue;
    out2.push(holeThread({ ...h, d: 2 * h.rPath + D, pitch: h.dz / (h.sweep / (2 * Math.PI)) }));
  }
  return out2;
}

function holeThread(h) {
  const nominal = metricNominal(h.d, { under: 0.1, over: 0.1 });
  return { ...h, nominal, name: designation(nominal, h.d, h.pitch), minor: minorInternal(h.d, h.pitch), drill: tapDrill(h.d, h.pitch) };
}

// circle through three points (XY)
function circle3(a, b, c) {
  const d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
  if (Math.abs(d) < 1e-12) return null;
  const s = (p) => p[0] * p[0] + p[1] * p[1];
  const x = (s(a) * (b[1] - c[1]) + s(b) * (c[1] - a[1]) + s(c) * (a[1] - b[1])) / d;
  const y = (s(a) * (c[0] - b[0]) + s(b) * (a[0] - c[0]) + s(c) * (b[0] - a[0])) / d;
  return { x, y, r: Math.hypot(a[0] - x, a[1] - y) };
}

// how a thread is named in reports: "M8×1.25 by T2 (G84)"
export const threadSource = (th) => `T${th.tool}, ${th.cycle === 'helix' ? 'helical thread milling' : th.cycle}`;
export const holeAt = (th) => `X${fx(th.x, 2)} Y${fx(th.y, 2)}`;
