// Setup (operation) directives read from program comments.
//
//   (SETUP OP20)                 operation name (default OP10, OP20 … by position)
//   (STOCK FROM OP10)            start from the result of another operation (name or file name)
//   (ORIENT FLIP X)              re-clamp: FLIP X | FLIP Y | ROT X90 | ROT Y-90 | ROT Z180 … (applied in order)
//   (ZERO TOP CENTER)            work zero: SAME | TOP CENTER | TOP CORNER | PART X.. Y.. Z.. | FACE (lathe)
//   (VISE GRIP 8 WIDTH 150)      machine vise: jaws hold the part GRIP mm up from its bottom
//   (CHUCK GRIP 20)              lathe: length of the part inside the chuck jaws
//   (ROTARY A)                   4th axis about X (B about Y) through the work zero; ISO sense: +A moves the
//                                tool right-hand around +X relative to the part. (ROTARY A REVERSE) flips it
//
// The part frame is the coordinate system the part was first described in (the first
// operation's work coordinates). Each operation has X = part -> work transform.

import { IDENTITY, mulXform, rotX, rotY, rotZ, translate, applyX, snapX } from '../sim/voxelsim.js';

export function parseSetup(comments) {
  const s = { name: null, stockFrom: null, orient: [], zero: null, vise: null, chuck: null, rotary: null };
  for (const c of comments) {
    const t = c.text.toUpperCase().replace(/\s+/g, ' ').trim();
    let m;
    if (!s.name && (m = /^(?:SETUP|OPERATION)\s*[:=]?\s*(\S+)/.exec(t))) s.name = m[1];
    if (!s.name && (m = /^OP\s*(\d+)$/.exec(t))) s.name = `OP${m[1]}`;
    if ((m = /^STOCK\s+FROM\s+(.+)$/.exec(t))) s.stockFrom = c.text.replace(/^\s*stock\s+from\s+/i, '').replace(/["']/g, '').trim();
    if ((m = /^ORIENT(?:ATION)?\s+(.+)$/.exec(t))) {
      const re = /(FLIP|ROT)\s*([XYZ])\s*(-?\d+(?:\.\d+)?)?/g;
      let r;
      while ((r = re.exec(m[1]))) {
        const deg = r[1] === 'FLIP' ? 180 : parseFloat(r[3] ?? '90');
        s.orient.push({ axis: r[2].toLowerCase(), deg });
      }
    }
    if ((m = /^ZERO\s+(.+)$/.exec(t))) {
      const z = m[1];
      if (/^SAME/.test(z)) s.zero = { mode: 'same' };
      else if (/TOP\s+(CORNER|LEFT|FRONT)/.test(z)) s.zero = { mode: 'top-corner' };
      else if (/TOP/.test(z)) s.zero = { mode: 'top-center' };
      else if (/^FACE/.test(z)) s.zero = { mode: 'face' };
      else if (/^PART/.test(z)) {
        const g = (L) => { const mm = new RegExp(`${L}\\s*(-?\\d*\\.?\\d+)`).exec(z); return mm ? parseFloat(mm[1]) : 0; };
        s.zero = { mode: 'part', point: [g('X'), g('Y'), g('Z')] };
      }
    }
    if ((m = /^VISE\b(.*)$/.exec(t))) {
      const g = (L, d) => { const mm = new RegExp(`${L}\\s*=?\\s*(\\d*\\.?\\d+)`).exec(m[1]); return mm ? parseFloat(mm[1]) : d; };
      s.vise = { grip: g('GRIP', 8), width: g('WIDTH', 150), jaw: g('JAW', 45) };
    }
    if ((m = /^CHUCK\b(.*)$/.exec(t))) {
      const mm = /GRIP\s*=?\s*(\d*\.?\d+)/.exec(m[1]);
      s.chuck = { grip: mm ? parseFloat(mm[1]) : 15 };
    }
    if ((m = /^ROTARY\s+([AB])(.*)$/.exec(t))) s.rotary = { axis: m[1], reverse: /REV/.test(m[2]) };
  }
  return s;
}

export function orientationX(orient) {
  let X = IDENTITY;
  for (const o of orient) X = mulXform(o.axis === 'x' ? rotX(o.deg) : o.axis === 'y' ? rotY(o.deg) : rotZ(o.deg), X);
  return snapX(X);
}

function boundsUnder(X, b) {
  let mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  for (const cx of [b.min[0], b.max[0]]) for (const cy of [b.min[1], b.max[1]]) for (const cz of [b.min[2], b.max[2]]) {
    const p = applyX(X, [cx, cy, cz]);
    for (let k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], p[k]); mx[k] = Math.max(mx[k], p[k]); }
  }
  return { min: mn, max: mx };
}

// part -> work transform for an operation, given the part's current bounds (part frame)
export function setupTransform(setup, partBounds, machine) {
  const R = orientationX(setup.orient);
  const reoriented = setup.orient.length > 0;
  const mode = setup.zero?.mode || (reoriented ? (machine === 'lathe' ? 'face' : 'top-center') : 'same');
  if (mode === 'same') return R;
  if (mode === 'part') return snapX(mulXform(translate(applyX(R, setup.zero.point).map((v) => -v)), R));
  const b = boundsUnder(R, partBounds);
  let zero;
  if (mode === 'face') zero = [0, 0, b.max[2]];
  else if (mode === 'top-corner') zero = [b.min[0], b.min[1], b.max[2]];
  else zero = [(b.min[0] + b.max[0]) / 2, (b.min[1] + b.max[1]) / 2, b.max[2]];
  if (machine === 'lathe') zero = [0, 0, zero[2]]; // lathe zero stays on the spindle axis
  return snapX(mulXform(translate(zero.map((v) => -v)), R));
}

export function describeSetup(setup, X, machine) {
  const parts = [];
  if (setup.stockFrom) parts.push(`stock = result of ${setup.stockFrom}`);
  if (setup.orient.length) parts.push(`orientation ${setup.orient.map((o) => (o.deg === 180 ? `FLIP ${o.axis.toUpperCase()}` : `ROT ${o.axis.toUpperCase()}${o.deg}`)).join(' ')}`);
  if (setup.zero) parts.push(`zero ${setup.zero.mode === 'part' ? `at part X${setup.zero.point[0]} Y${setup.zero.point[1]} Z${setup.zero.point[2]}` : setup.zero.mode}`);
  if (setup.vise) parts.push(`vise grip ${setup.vise.grip} mm`);
  if (setup.chuck) parts.push(`chuck grip ${setup.chuck.grip} mm`);
  if (setup.rotary) parts.push(`rotary ${setup.rotary.axis} axis${setup.rotary.reverse ? ' (reversed)' : ''}`);
  if (X && X !== IDENTITY && !isIdentity(X)) {
    const o = applyX(X, [0, 0, 0]);
    parts.push(`part origin is at work X${fmt(o[0])} Y${fmt(o[1])} Z${fmt(o[2])}`);
  }
  return parts.join(', ') || (machine === 'lathe' ? 'single chucking' : 'single setup');
}

export function isIdentity(X) {
  return X.R.every((v, i) => Math.abs(v - [1, 0, 0, 0, 1, 0, 0, 0, 1][i]) < 1e-9) && X.t.every((v) => Math.abs(v) < 1e-9);
}
const fmt = (v) => (Math.abs(v) < 5e-4 ? '0' : (+v.toFixed(3)).toString());
