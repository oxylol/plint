// Craft checks on the measured engagement: what a machinist checks by habit and the geometry
// report cannot show — a face mill run down the centreline, an edge rubbing on a chip too thin, a
// face mill plunged, a ball mill finishing a floor with its tip, a lathe insert pushed instead of
// cutting. Each finding is an issue with a line, so it reaches the Check tab, the gutter and Claude.

export const CRAFT = {
  faceCentredTol: 20,   // °: |lo + hi| below this = the contact arc is symmetric about the motion
  faceMinAe: 0.5, faceMaxAe: 0.98, // of Ø: one pass over a surface narrower than the cutter
  faceMinLen: 2,        // × Ø of such path before it counts
  hmPerD: 0.0015, hmMin: 0.005, hmMax: 0.02, // mean chip thickness floor = clamp(0.0015·Ø, 0.005, 0.02) mm
  rubMinAe: 0.04,       // × Ø: a cutter merely kissing the material (roll-in, roll-out) is not judged
  rubLen: 10,           // mm of cut below the floor before it counts (at least Ø)
  ballTipAp: 0.15,      // × R: a ball mill cutting a floor with its tip only
  ballTipLen: 20,
  facePlunge: 0.2,      // mm: a face mill going straight down into material
  latheFeedMin: 0.05,   // mm/rev
  latheApRatio: 0.25,   // × nose radius
  latheLen: 10,
};

const fx = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '?');
// the typical value over the flagged length, not the entry transient
function weightedMedian(pairs) {
  const s = pairs.slice().sort((a, b) => a[0] - b[0]);
  const total = s.reduce((n, p) => n + p[1], 0);
  let run = 0;
  for (const [v, w] of s) { run += w; if (run >= total / 2) return v; }
  return s.length ? s[s.length - 1][0] : 0;
}
const chipFloor = (D) => Math.min(CRAFT.hmMax, Math.max(CRAFT.hmMin, CRAFT.hmPerD * D));

export function craftIssues({ result, engagement: eng }) {
  if (!eng?.samples || !result?.moves?.length) return [];
  const c = eng.samples, moves = result.moves, tools = result.tools;
  const lathe = result.machine === 'lathe';
  // one finding per tool and kind: the length it covers and its worst sample
  const acc = new Map();
  const bump = (kind, tool, k, len, v, worse, extra) => {
    const key = `${kind}|${tool}`;
    let a = acc.get(key);
    if (!a) { a = { kind, tool, len: 0, k, v, extra, pairs: [] }; acc.set(key, a); }
    a.len += len;
    a.pairs.push([v, len]);
    if (worse(v, a.v)) { a.v = v; a.k = k; a.extra = extra; }
  };
  const lower = (a, b) => a < b, higher = (a, b) => a > b;
  for (let k = 0; k < c.t.length; k++) {
    const m = moves[c.move[k]];
    const tool = tools[m.tool];
    if (!tool) continue;
    const on = c.ap[k] > 1e-3 || c.angle[k] > 0;
    if (!on) continue;
    const len = c.len[k];
    if (lathe) {
      if (['groove', 'drill', 'thread'].includes(tool.type)) continue;
      const f = c.f[k], ap = c.ap[k];
      if (ap > 0.02 && f > 0 && f < CRAFT.latheFeedMin) bump('feed', m.tool, k, len, f, lower);
      if (ap > 0.02 && tool.noseRadius > 0 && ap < CRAFT.latheApRatio * tool.noseRadius) bump('ap', m.tool, k, len, ap, lower);
      continue;
    }
    const D = tool.diameter || 0;
    if (c.plunge[k]) {
      if (tool.type === 'face' && c.ap[k] > CRAFT.facePlunge) bump('faceplunge', m.tool, k, len, c.ap[k], higher);
      continue;
    }
    const z = tool.flutes || 0;
    if (z > 0 && m.rpm > 0 && m.rate > 0 && D > 0 && c.angle[k] >= 5 && c.ae[k] >= CRAFT.rubMinAe * D) {
      const fz = m.rate / (m.rpm * z);
      // mean chip thickness from the measured contact arc: a full slot gives 2/π·fz, 10 % ae about a third of fz
      const hm = (fz * 360 * c.ae[k]) / (Math.PI * D * c.angle[k]);
      if (hm < chipFloor(D)) bump('rub', m.tool, k, len, hm, lower, { fz, rate: m.rate });
    }
    if (tool.type === 'face' && c.ae[k] >= CRAFT.faceMinAe * D && c.ae[k] <= CRAFT.faceMaxAe * D && Math.abs(c.lo[k] + c.hi[k]) < CRAFT.faceCentredTol) {
      bump('facecentre', m.tool, k, len, c.ae[k], higher, { lo: c.lo[k], hi: c.hi[k] });
    }
    if (tool.type === 'ball') {
      const dz = Math.abs(m.pts[m.pts.length - 1][2] - m.pts[0][2]);
      if (dz < 0.05 * (m.len || 1) && c.ap[k] > 0.01 && c.ap[k] < CRAFT.ballTipAp * (D / 2)) bump('balltip', m.tool, k, len, c.ap[k], lower);
    }
  }
  const issues = [];
  for (const a of acc.values()) {
    const m = moves[c.move[a.k]];
    const tool = tools[a.tool];
    const D = tool?.diameter || 0;
    const at = { line: m.line, file: m.file, callLine: m.callLine, source: 'craft' };
    const L = `${fx(a.len, 0)} mm`;
    const typ = weightedMedian(a.pairs);
    switch (a.kind) {
      case 'facecentre':
        if (a.len < CRAFT.faceMinLen * D) break;
        issues.push({ level: 'warning', ...at, text: `T${a.tool} Ø${D} face mill runs centred over the ${fx(a.v, 0)} mm width (contact ${fx(a.extra.lo, 0)}°..${fx(a.extra.hi, 0)}°, ${L}) — offset the path about ${fx(D / 4, 0)} mm so the inserts leave the part on one side; centred, the radial force flips at every insert and the cut vibrates` });
        break;
      case 'rub': {
        if (a.len < Math.max(CRAFT.rubLen, D)) break;
        const floor = chipFloor(D);
        const want = Math.max(50, Math.round((a.extra.rate * (floor * 1.3)) / Math.max(typ, 1e-4) / 50) * 50);
        issues.push({ level: 'warning', ...at, text: `T${a.tool} Ø${D}: chip thickness ${fx(typ, 3)} mm (fz ${fx(a.extra.fz, 3)} mm, z${tool?.flutes || '?'}) over ${L} — rubbing below ${fx(floor, 3)} mm; feed about F${want} there, or take more width of cut` });
        break;
      }
      case 'faceplunge':
        issues.push({ level: 'error', ...at, text: `T${a.tool} face mill plunges ${fx(a.v)} mm straight into material — a face mill cannot cut on its centre; ramp in from outside the stock or start the pass beside it` });
        break;
      case 'balltip':
        if (a.len < CRAFT.ballTipLen) break;
        issues.push({ level: 'warning', ...at, text: `T${a.tool} Ø${D} ball mill cuts a flat area with its tip (ap ${fx(a.v)} mm, no surface speed at the centre) over ${L} — use a flat or bull-nose end mill for floors, or tilt the cut` });
        break;
      case 'feed':
        if (a.len < CRAFT.latheLen) break;
        issues.push({ level: 'warning', ...at, text: `T${a.tool}${tool?.noseRadius ? ` R${tool.noseRadius}` : ''}: feed ${fx(typ, 3)} mm/rev while cutting (${L}) — below ${CRAFT.latheFeedMin} mm/rev the insert rubs; use 0.08–0.15 mm/rev for finishing` });
        break;
      case 'ap':
        if (a.len < CRAFT.latheLen) break;
        issues.push({ level: 'info', ...at, text: `T${a.tool} R${tool?.noseRadius}: depth of cut ${fx(typ)} mm over ${L} — below a quarter of the nose radius the insert pushes instead of cutting; take ${fx(CRAFT.latheApRatio * tool.noseRadius)} mm or more, or a smaller radius` });
        break;
    }
  }
  return issues;
}
