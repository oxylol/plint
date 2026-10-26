// Lathe tool geometry: every lathe tool is a cutter polygon + holder polygon in
// tool-local (dz, dx) millimetres around its reference point (the programmed point;
// for turning inserts that is the imaginary tip), plus the nose-centre offset used by
// G41/G42. dz points to +Z (tailstock), dx to +X (radially outward / up in the view).
//
// Imported CncSimulator5 tools (TurnTools.ini) carry their own polygons. Their
// conventions were taken from the simulator's register loader:
//  * CutterN / HolderN = "dz dx" (first value along Z, second radially)
//  * NoseOffsetX/Y = nose centre (dz, dx) relative to the reference point
//  * Groov*: blade W x H rectangle built from ToolOrigin 1..8 (1 bottom-left,
//    2 bottom-middle, 3 bottom-right, 4 middle-right, 5 top-right, 6 top-middle,
//    7 top-left, 8 middle-left); GroovVertDown holders hang from the blade's top
//    centre, GroovHoriLeft holders from the blade's far end
//  * Drill*: Width = length, Height = diameter, apex at the reference point,
//    holder starts at the back end

const DEG = Math.PI / 180;

function insertPolygon(R, alphaDeg, size) {
  // rhombic insert, nose radius R, nose centre at (R, R) so (0,0) is the imaginary tip,
  // body toward +Z/+X (OD tool cutting toward the chuck)
  const a = Math.max(20, Math.min(100, alphaDeg)) * DEG;
  const th1 = 45 * DEG - a / 2, th2 = 45 * DEG + a / 2;
  const e1 = [Math.cos(th1), Math.sin(th1)], e2 = [Math.cos(th2), Math.sin(th2)];
  const c = [R, R];
  const bis = [Math.SQRT1_2, Math.SQRT1_2];
  const k = R / Math.sin(a / 2);
  const S = [c[0] - bis[0] * k, c[1] - bis[1] * k]; // sharp corner of the insert
  const T1 = [c[0] + R * Math.sin(th1), c[1] - R * Math.cos(th1)];
  const A = [S[0] + e1[0] * size, S[1] + e1[1] * size];
  const C = [A[0] + e2[0] * size, A[1] + e2[1] * size];
  const B = [S[0] + e2[0] * size, S[1] + e2[1] * size];
  const pts = [T1, A, C, B];
  if (R > 0.001) {
    const a0 = th2 + Math.PI / 2, a1 = th1 - Math.PI / 2 + 2 * Math.PI;
    const n = 10;
    for (let i = 0; i <= n; i++) {
      const t = a0 + ((a1 - a0) * i) / n;
      pts.push([c[0] + R * Math.cos(t), c[1] + R * Math.sin(t)]);
    }
  } else {
    pts.push(S);
  }
  return pts;
}

const box = (z0, x0, z1, x1) => [[z0, x0], [z1, x0], [z1, x1], [z0, x1]];
const mirrorZ = (p) => p.map(([z, x]) => [-z, x]).reverse();
const mirrorX = (p) => p.map(([z, x]) => [z, -x]).reverse();
const shift = (p, dz, dx) => p.map(([z, x]) => [z + dz, x + dx]);

// corner of a W x H blade (screen-style origin numbers) relative to its bottom-left corner
function originOffset(origin, W, H) {
  return ({ 1: [0, 0], 2: [W / 2, 0], 3: [W, 0], 4: [W, H / 2], 5: [W, H], 6: [W / 2, H], 7: [0, H], 8: [0, H / 2] })[origin] || [0, 0];
}

// Fill cutter / holder / noseOffset for a lathe tool that has none (built-in or typed by hand).
export function buildLatheGeometry(t) {
  if (t.cutter && t.cutter.length >= 3) return t;
  const R = t.noseRadius ?? 0.4;
  switch (t.type) {
    case 'groove': {
      const W = t.width || 3, H = t.height || 16;
      const [oz, ox] = originOffset(t.origin || 1, W, H);
      t.cutter = shift(box(0, 0, W, H), -oz, -ox);
      const cz = W / 2 - oz, top = H - ox;
      t.holder = shift([[-6, -1], [6, -1], [6, 24], [9, 27], [9, 108], [-9, 108], [-9, 27], [-6, 24]], cz, top);
      t.noseOffset = [0, 0];
      break;
    }
    case 'thread': {
      const h = 5, half = h * Math.tan(((t.tipAngle || 60) / 2) * DEG);
      t.cutter = [[0, 0], [half, h], [half, h + 3], [-half, h + 3], [-half, h]];
      t.holder = box(-7, h + 2, 7, 110);
      t.noseOffset = [0, 0];
      break;
    }
    case 'drill': {
      const D = t.diameter || 10, L = t.length || Math.max(60, D * 6);
      const xc = (D / 2) / Math.tan(((t.tipAngle || 118) / 2) * DEG);
      t.cutter = [[0, 0], [xc, D / 2], [L, D / 2], [L, -D / 2], [xc, -D / 2]];
      const hr = Math.max(D / 2 + 8, 15);
      t.holder = box(L - 1, -hr, L + 28, hr);
      if (t.hand === 'right') { t.cutter = mirrorZ(t.cutter); t.holder = mirrorZ(t.holder); }
      t.noseOffset = [0, 0];
      break;
    }
    case 'bore': {
      const size = 9;
      t.cutter = mirrorX(insertPolygon(R, t.insertAngle || 55, size));
      const bar = t.barDiameter || 16;
      t.holder = box(size * 0.45, -bar - 1.5, 140, -1.5);
      t.noseOffset = [R, -R];
      break;
    }
    default: { // turn
      const size = t.insertAngle && t.insertAngle < 50 ? 13 : 12;
      t.cutter = insertPolygon(R, t.insertAngle || 80, size);
      t.holder = box(size * 0.55, size * 0.8, size * 0.55 + 22, 112);
      t.noseOffset = [R, R];
      if (t.hand === 'right') { t.cutter = mirrorZ(t.cutter); t.holder = mirrorZ(t.holder); t.noseOffset = [-R, R]; }
    }
  }
  t.generated = true;
  return t;
}

// ---------------------------------------------------------------------------
// CncSimulator5 TurnTools.ini
// ---------------------------------------------------------------------------

function iniSections(text) {
  const out = [];
  let cur = null;
  for (const raw of text.replace(/\r/g, '').split('\n')) {
    const line = raw.trim();
    const m = /^\[(.+)\]$/.exec(line);
    if (m) { cur = { name: m[1], kv: {} }; out.push(cur); continue; }
    if (!cur) continue;
    const eq = line.indexOf('=');
    if (eq > 0) cur.kv[line.slice(0, eq).trim().toLowerCase()] = line.slice(eq + 1).trim();
  }
  return out;
}

function points(kv, prefix, count) {
  const pts = [];
  const n = Number(count) || 200;
  for (let i = 1; i <= n; i++) {
    const v = kv[`${prefix}${i}`];
    if (v === undefined) { if (count) continue; break; }
    const [a, b] = v.split(/\s+/).map(Number);
    if (Number.isFinite(a) && Number.isFinite(b)) pts.push([a, b]);
  }
  // CncSimulator closes polygons by repeating the first point
  if (pts.length > 2) {
    const f = pts[0], l = pts[pts.length - 1];
    if (Math.abs(f[0] - l[0]) < 1e-6 && Math.abs(f[1] - l[1]) < 1e-6) pts.pop();
  }
  return pts;
}

const num = (v, d = 0) => { const n = parseFloat(v); return Number.isFinite(n) ? n : d; };
const centroid = (p) => p.reduce((s, q) => [s[0] + q[0] / p.length, s[1] + q[1] / p.length], [0, 0]);

export function parseTurnToolsIni(text) {
  const tools = {};
  const skipped = [];
  for (const sec of iniSections(text)) {
    if (!/^TOOL\d+$/i.test(sec.name)) continue;
    const kv = sec.kv;
    const n = parseInt(kv.toolnr ?? '', 10);
    if (!Number.isFinite(n) || n <= 0) continue;
    if (tools[n]) { skipped.push(`T${n} (${sec.name}) — duplicate number, CncSimulator uses the first`); continue; }
    const type = kv.tooltype || 'Normal';
    const R = num(kv.noseradius);
    const t = {
      num: n,
      name: kv.toolname && !/^TOOL\d+$/i.test(kv.toolname) ? kv.toolname : `${type} T${n}`,
      noseRadius: R,
      feed: num(kv.feed, undefined),
      speed: num(kv.speed, undefined),
      source: 'cncsim',
      registerType: type,
      geometryName: kv.toolgeometry || '',
    };
    const W = num(kv.width, 1), H = num(kv.height, 1);
    const holderFile = points(kv, 'holder', kv.nrofholderpos);
    if (type === 'Normal') {
      t.cutter = points(kv, 'cutter', kv.nrofcutterpos);
      t.holder = holderFile;
      t.noseOffset = [num(kv.noseoffsetx), num(kv.noseoffsety)];
      const [cz, cx] = centroid(t.cutter.length ? t.cutter : [[1, 1]]);
      t.type = cx >= 0 ? 'turn' : 'bore';
      t.hand = cz >= 0 ? 'left' : 'right';
      // the nose centre lies on the insert's side of the tip; some factory entries
      // (T26–T28 in TurnTools.ini) contradict their own polygon, so trust the polygon
      if (R > 0 && t.cutter.length >= 3) {
        const want = [Math.sign(cz) * R, Math.sign(cx) * R];
        if (Math.sign(t.noseOffset[0]) !== Math.sign(want[0]) || Math.sign(t.noseOffset[1]) !== Math.sign(want[1])) {
          t.noseOffsetFixed = t.noseOffset;
          t.noseOffset = want;
        }
      }
      const a = estimateInsertAngle(t.cutter);
      if (a) t.insertAngle = a;
      if (t.cutter.length < 3) { delete t.cutter; buildLatheGeometry(t); }
    } else if (/^Drill(Left|Right)$/i.test(type)) {
      const right = /Right/i.test(type);
      t.type = 'drill';
      t.hand = right ? 'right' : 'left';
      t.diameter = H;
      t.length = W;
      t.tipAngle = num(kv.tipangledrill, 118) || 118;
      const xc = (H / 2) / Math.tan((t.tipAngle / 2) * DEG);
      t.cutter = [[0, 0], [xc, H / 2], [W, H / 2], [W, -H / 2], [xc, -H / 2]];
      t.holder = shift(holderFile, W, 0).map(([z, x]) => [Math.max(z, W), x]);
      if (right) { t.cutter = mirrorZ(t.cutter); t.holder = holderFile.map(([z, x]) => [Math.min(z - W, -W), x]); }
      t.noseOffset = [0, 0];
    } else if (/^Groov/i.test(type)) {
      t.type = 'groove';
      t.width = W;
      t.height = H;
      t.origin = parseInt(kv.toolorigin, 10) || 1;
      const [oz, ox] = originOffset(t.origin, W, H);
      t.cutter = shift(box(0, 0, W, H), -oz, -ox);
      // register holders overlap the blade by 1 mm for drawing; clip that so the holder
      // only starts where the blade ends (otherwise every full-depth groove "collides")
      if (/^GroovVertDown$/i.test(type)) {
        // holder hangs from the blade's top centre
        t.holder = shift(holderFile, W / 2 - oz, H - ox).map(([z, x]) => [z, Math.max(x, H - ox)]);
      } else if (/^GroovHoriRight$/i.test(type)) {
        t.holder = shift(holderFile, -oz, H / 2 - ox).map(([z, x]) => [Math.min(z, -oz), x]);
      } else {
        // GroovHoriLeft / GroovHoriLeftUp: holder continues from the blade's far end
        t.holder = shift(holderFile, W - oz, H / 2 - ox).map(([z, x]) => [Math.max(z, W - oz), x]);
      }
      t.noseOffset = [0, 0];
      t.faceGroove = /^GroovHori(Left|Right)$/i.test(type);
      t.boreGroove = /^GroovHoriLeftUp$/i.test(type);
    } else {
      skipped.push(`T${n} — unknown tool type ${type}`);
      continue;
    }
    if (/^(TOOL\s*\d+|NAME)?$/i.test(kv.toolname || '')) t.name = genericName(t);
    tools[n] = t;
  }
  return { tools, skipped };
}

function genericName(t) {
  switch (t.type) {
    case 'groove': return `${t.faceGroove ? 'Face groove' : t.boreGroove ? 'Groove bar' : 'Groove'} ${t.width} mm`;
    case 'drill': return `Drill Ø${t.diameter}${t.hand === 'right' ? ' (right)' : ''}`;
    case 'bore': return `Boring bar R${t.noseRadius}`;
    default: return `Turning ${t.hand === 'right' ? 'right' : 'left'} R${t.noseRadius}${t.insertAngle ? ` ${t.insertAngle}°` : ''}`;
  }
}

// angle between the two insert edges that meet at the nose (for display / Claude)
function estimateInsertAngle(poly) {
  if (!poly || poly.length < 4) return null;
  let best = 0, bd = Infinity;
  poly.forEach((p, i) => { const d = Math.hypot(p[0], p[1]); if (d < bd) { bd = d; best = i; } });
  const n = poly.length;
  // walk away from the nose until the edge direction settles (skips the nose arc)
  const far = (dir) => {
    let i = best;
    for (let k = 0; k < n; k++) { i = (i + dir + n) % n; if (Math.hypot(poly[i][0], poly[i][1]) > 4) return poly[i]; }
    return null;
  };
  const a = far(1), b = far(-1);
  if (!a || !b) return null;
  const nose = poly[best];
  const ang = Math.abs(Math.atan2(a[1] - nose[1], a[0] - nose[0]) - Math.atan2(b[1] - nose[1], b[0] - nose[0])) / DEG;
  const v = ang > 180 ? 360 - ang : ang;
  return Math.round(v / 5) * 5;
}

// ---------------------------------------------------------------------------
// CncSimulator5 MillTools.millreg  ("%d %lf %lf %lf %lf %d %lf %s")
// slot  diameter  cuttingLength  feed  spindle  toolNumber  overallLength  name
// ---------------------------------------------------------------------------
export function parseMillReg(text) {
  const tools = {};
  const skipped = [];
  const lines = text.replace(/\r/g, '').split('\n');
  for (const line of lines) {
    const m = /^\s*(\d+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(\d+)\s+(-?[\d.]+)\s*(.*)$/.exec(line);
    if (!m) continue;
    const [, , dia, len, feed, spindle, tnum, overall, nameRaw] = m;
    const D = parseFloat(dia);
    const n = parseInt(tnum, 10);
    const name = nameRaw.trim();
    if (!(D > 0) || !n) continue; // empty slot
    if (tools[n]) { skipped.push(`T${n} — duplicate number`); continue; }
    const up = name.toUpperCase();
    let type = 'flat';
    if (/BORR|DRILL/.test(up)) type = 'drill';
    else if (/KUL|BALL/.test(up)) type = 'ball';
    else if (/FAS|CHAMF|VSINK|FÖRSÄNK/.test(up)) type = 'chamfer';
    else if (/CENTR|PUNKT|SPOT/.test(up)) type = 'spot';
    else if (/GÄNGTAPP|GANGTAPP|TAPP|\bTAP\b/.test(up)) type = 'tap';
    else if (/GÄNGFR|GANGFR|THREAD/.test(up)) type = 'threadmill';
    else if (/PLAN|FACE|PLANFR/.test(up)) type = 'face';
    else if (/GRAV|ENGRAV/.test(up)) type = 'engrave';
    tools[n] = {
      num: n,
      type,
      diameter: D,
      fluteLength: parseFloat(len) > 0 ? parseFloat(len) : undefined,
      overallLength: parseFloat(overall) > 0 ? parseFloat(overall) : undefined,
      feed: parseFloat(feed) || undefined,
      speed: parseFloat(spindle) || undefined,
      name: name && name !== 'NAME' ? name : `T${n}`,
      source: 'cncsim',
      ...(type === 'drill' ? { tipAngle: 118 } : {}),
    };
  }
  return { tools, skipped };
}

export function detectRegister(text) {
  if (/\[TURNTOOLREGESTRY\]|\[TOOL\d+\][\s\S]*ToolType=/i.test(text)) return 'lathe';
  if (/^\s*\d+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\d+\s+[\d.]+/m.test(text)) return 'mill';
  return null;
}

// ---------------------------------------------------------------------------
// polygon helpers shared by the simulator and the views
// ---------------------------------------------------------------------------
export function convexHull(pts) {
  const p = pts.map((q) => [q[0], q[1]]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return { lower: p, upper: p };
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [];
  for (const q of p) { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop(); lower.push(q); }
  const upper = [];
  for (let i = p.length - 1; i >= 0; i--) { const q = p[i]; while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop(); upper.push(q); }
  return { lower, upper: upper.reverse() }; // both sorted by increasing z
}

export function polygonBounds(p) {
  let z0 = Infinity, z1 = -Infinity, x0 = Infinity, x1 = -Infinity;
  for (const [z, x] of p) { if (z < z0) z0 = z; if (z > z1) z1 = z; if (x < x0) x0 = x; if (x > x1) x1 = x; }
  return { z0, z1, x0, x1 };
}

// short human description for prompts / tables
export function describeLatheGeometry(t) {
  if (!t.cutter) return '';
  const b = polygonBounds(t.cutter);
  const hb = t.holder && t.holder.length ? polygonBounds(t.holder) : null;
  const parts = [];
  if (t.type === 'groove') {
    parts.push(`blade ${t.width}×${t.height ?? 16} mm, reference = ${originName(t.origin || 1)} corner${t.faceGroove ? ' (face grooving, axial)' : t.boreGroove ? ' (horizontal grooving bar, edge facing +X)' : ''}`);
    if (!t.faceGroove && !t.boreGroove) parts.push(`max plunge depth ≈ ${(t.height ?? 16).toFixed(0)} mm before the holder touches`);
  } else if (t.type === 'drill') {
    parts.push(`Ø${t.diameter} × ${t.length ?? Math.round(b.z1 - b.z0)} mm, ${t.tipAngle}° point, on centre (program X0)`);
  } else if (t.type === 'bore') {
    const minBore = hb ? 2 * Math.max(-hb.x0, -b.x0) : null;
    parts.push(`boring bar, nose R${t.noseRadius}${t.insertAngle ? `, ${t.insertAngle}° insert` : ''}${minBore ? `, needs a bore ≥ Ø${minBore.toFixed(0)}` : ''}`);
  } else {
    parts.push(`${t.hand === 'right' ? 'right-hand (cuts toward +Z)' : 'left-hand (cuts toward the chuck)'} turning, nose R${t.noseRadius}${t.insertAngle ? `, ${t.insertAngle}° insert` : ''}`);
  }
  return parts.join('; ');
}

function originName(o) {
  return ({ 1: 'bottom-left (−Z)', 2: 'bottom-centre', 3: 'bottom-right (+Z)', 4: 'right-middle', 5: 'top-right', 6: 'top-centre', 7: 'top-left', 8: 'left-middle' })[o] || `origin ${o}`;
}
