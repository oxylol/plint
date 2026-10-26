// Path geometry: arcs, tessellation, corner rounding (,R / ,C) and
// cutter / nose radius compensation (G41 / G42) with one-element lookahead.
//
// Prim = { type: 'line'|'arc', kind: 'rapid'|'feed', a:[x,y,z], b:[x,y,z], meta,
//          arc only: c:[x,y,z] (center), plane: 17|18|19, sweep (rad, +CCW), full }

export const PLANES = { 17: [0, 1, 2], 18: [2, 0, 1], 19: [1, 2, 0] };
const EPS = 1e-9;

export const uv = (p, plane) => { const ax = PLANES[plane]; return [p[ax[0]], p[ax[1]]]; };
export const wOf = (p, plane) => p[PLANES[plane][2]];
export function fromUV(u, v, w, plane) {
  const ax = PLANES[plane];
  const p = [0, 0, 0];
  p[ax[0]] = u; p[ax[1]] = v; p[ax[2]] = w;
  return p;
}
const sub2 = (a, b) => [a[0] - b[0], a[1] - b[1]];
const add2 = (a, b) => [a[0] + b[0], a[1] + b[1]];
const mul2 = (a, k) => [a[0] * k, a[1] * k];
const len2 = (a) => Math.hypot(a[0], a[1]);
const dot2 = (a, b) => a[0] * b[0] + a[1] * b[1];
const cross2 = (a, b) => a[0] * b[1] - a[1] * b[0];
const norm2 = (a) => { const l = len2(a); return l < EPS ? null : [a[0] / l, a[1] / l]; };
const left2 = (d) => [-d[1], d[0]];
const dist3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

// --------------------------------------------------------------------------
// Arcs
// --------------------------------------------------------------------------

// Build an arc prim. cw = G02. Returns { prim, error } where error is a radius mismatch message.
export function buildArc(a, b, centerUV, cw, plane, kind, meta, tol = 0.01) {
  const A = uv(a, plane), B = uv(b, plane);
  const r0 = len2(sub2(A, centerUV));
  const r1 = len2(sub2(B, centerUV));
  let error = null;
  if (r0 < 1e-6) return { prim: null, error: 'Arc radius is zero (centre on start point)' };
  if (Math.abs(r0 - r1) > Math.max(tol, 0.0005 * r0)) {
    error = `Arc end point is not on the circle (start radius ${r0.toFixed(3)}, end radius ${r1.toFixed(3)})`;
  }
  const a0 = Math.atan2(A[1] - centerUV[1], A[0] - centerUV[0]);
  const a1 = Math.atan2(B[1] - centerUV[1], B[0] - centerUV[0]);
  let sweep = a1 - a0;
  const same = len2(sub2(A, B)) < 1e-6;
  if (cw) {
    while (sweep >= -1e-12) sweep -= 2 * Math.PI;
    while (sweep < -2 * Math.PI - 1e-12) sweep += 2 * Math.PI;
    if (same) sweep = -2 * Math.PI;
  } else {
    while (sweep <= 1e-12) sweep += 2 * Math.PI;
    while (sweep > 2 * Math.PI + 1e-12) sweep -= 2 * Math.PI;
    if (same) sweep = 2 * Math.PI;
  }
  const c = fromUV(centerUV[0], centerUV[1], wOf(a, plane), plane);
  return { prim: { type: 'arc', kind, a, b, c, plane, sweep, radius: r0, full: same, meta }, error };
}

// Centre from radius word (R>0: <=180°, R<0: >180°)
export function centerFromRadius(a, b, R, cw, plane) {
  const A = uv(a, plane), B = uv(b, plane);
  const chord = sub2(B, A);
  const d = len2(chord);
  if (d < 1e-9) return { error: 'R-arc with identical start and end point (use I/J/K for a full circle)' };
  const r = Math.abs(R);
  let error = null;
  let h2 = r * r - (d * d) / 4;
  if (h2 < 0) {
    if (-h2 > Math.max(1e-4, 0.002 * r * r)) error = `Arc radius R${r} is too small for chord length ${d.toFixed(3)}`;
    h2 = 0;
  }
  const h = Math.sqrt(h2);
  const mid = mul2(add2(A, B), 0.5);
  const left = left2(mul2(chord, 1 / d));
  const sign = (cw ? -1 : 1) * (R > 0 ? 1 : -1);
  return { center: add2(mid, mul2(left, sign * h)), error };
}

export function primLength(p) {
  if (p.type === 'line') return dist3(p.a, p.b);
  const planar = Math.abs(p.sweep) * p.radius;
  const dw = wOf(p.b, p.plane) - wOf(p.a, p.plane);
  return Math.hypot(planar, dw);
}

export function tessellate(p, tol = 0.005) {
  if (p.type === 'line') return [p.a, p.b];
  const C = uv(p.c, p.plane);
  const A = uv(p.a, p.plane);
  const a0 = Math.atan2(A[1] - C[1], A[0] - C[0]);
  const w0 = wOf(p.a, p.plane), w1 = wOf(p.b, p.plane);
  const r = p.radius;
  const maxStep = r > tol ? Math.min(Math.PI / 36, 2 * Math.acos(Math.max(-1, 1 - tol / r))) : Math.PI / 8;
  const n = Math.max(2, Math.ceil(Math.abs(p.sweep) / maxStep));
  const pts = [p.a];
  for (let i = 1; i < n; i++) {
    const t = i / n;
    const ang = a0 + p.sweep * t;
    pts.push(fromUV(C[0] + r * Math.cos(ang), C[1] + r * Math.sin(ang), w0 + (w1 - w0) * t, p.plane));
  }
  pts.push(p.b);
  return pts;
}

// Direction (unit 2D, in plane) at start / end. null when the move has no in-plane component.
export function dirAt(p, plane, atEnd) {
  if (p.type === 'line') return norm2(sub2(uv(p.b, plane), uv(p.a, plane)));
  if (p.plane !== plane) return null;
  const C = uv(p.c, plane);
  const P = uv(atEnd ? p.b : p.a, plane);
  const rad = norm2(sub2(P, C));
  if (!rad) return null;
  return p.sweep > 0 ? left2(rad) : mul2(left2(rad), -1);
}

// --------------------------------------------------------------------------
// Corner stage (,R fillet and ,C chamfer between two lines)
// --------------------------------------------------------------------------

export class CornerStage {
  constructor(next, warn) {
    this.next = next;
    this.warn = warn;
    this.pending = null;
  }
  push(prim, opt = {}) {
    if (this.pending) {
      const p = this.pending;
      this.pending = null;
      this._resolve(p.prim, p.opt, prim, opt);
      return;
    }
    this._accept(prim, opt);
  }
  _accept(prim, opt) {
    if (opt.corner && prim.type === 'line') {
      this.pending = { prim, opt };
      return;
    }
    if (opt.corner) this.warn(prim.meta, 'Corner rounding/chamfer (,R / ,C) is only supported between two straight moves');
    this.next.push(prim, opt);
  }
  // emit a pending corner move unchanged (tool change, cycle start …)
  flushPending() {
    if (this.pending) {
      const p = this.pending;
      this.pending = null;
      this.warn(p.prim.meta, 'Corner rounding/chamfer (,R / ,C) needs a following straight move');
      this.next.push(p.prim, p.opt);
    }
  }
  flush() {
    this.flushPending();
    this.next.flush?.();
  }
  _resolve(A, optA, B, optB) {
    const corner = optA.corner;
    const plane = A.meta.plane || 17;
    if (B.type !== 'line') {
      if (B.type !== 'event') this.warn(A.meta, 'Corner rounding/chamfer (,R / ,C) is only supported between two straight moves');
      else this.warn(A.meta, 'Corner rounding/chamfer (,R / ,C) needs a following straight move');
      this.next.push(A, optA);
      this._accept(B, optB);
      return;
    }
    const P0 = uv(A.a, plane), P1 = uv(A.b, plane), P2 = uv(B.b, plane);
    const dA = norm2(sub2(P1, P0)), dB = norm2(sub2(P2, P1));
    if (!dA || !dB) { this.next.push(A, optA); this._accept(B, optB); return; }
    const cosPhi = Math.max(-1, Math.min(1, dot2(dA, dB)));
    const phi = Math.acos(cosPhi);
    if (phi < 1e-4 || phi > Math.PI - 1e-4) {
      this.warn(A.meta, 'Corner rounding/chamfer ignored: the two moves are collinear');
      this.next.push(A, optA); this._accept(B, optB);
      return;
    }
    const lenA = len2(sub2(P1, P0)), lenB = len2(sub2(P2, P1));
    const t = corner.R !== undefined ? corner.R * Math.tan(phi / 2) : corner.C;
    if (t > lenA + 1e-6 || t > lenB + 1e-6) {
      this.warn(A.meta, `Corner ${corner.R !== undefined ? 'radius ,R' + corner.R : 'chamfer ,C' + corner.C} is too large for the adjacent moves`);
      this.next.push(A, optA); this._accept(B, optB);
      return;
    }
    const wA0 = wOf(A.a, plane), wA1 = wOf(A.b, plane), wB1 = wOf(B.b, plane);
    const T1uv = sub2(P1, mul2(dA, t));
    const T2uv = add2(P1, mul2(dB, t));
    const T1 = fromUV(T1uv[0], T1uv[1], wA1 - (wA1 - wA0) * (t / lenA), plane);
    const T2 = fromUV(T2uv[0], T2uv[1], wA1 + (wB1 - wA1) * (t / lenB), plane);
    const { corner: _c, ...restA } = optA;
    this.next.push({ ...A, b: T1 }, restA);
    if (corner.R !== undefined) {
      const turnLeft = cross2(dA, dB) > 0;
      const center = add2(T1uv, mul2(left2(dA), (turnLeft ? 1 : -1) * corner.R));
      const { prim } = buildArc(T1, T2, center, !turnLeft, plane, A.kind, A.meta, 0.05);
      if (prim) this.next.push(prim, {});
    } else {
      this.next.push({ type: 'line', kind: A.kind, a: T1, b: T2, meta: A.meta }, {});
    }
    this._accept({ ...B, a: T2 }, optB);
  }
}

// --------------------------------------------------------------------------
// Cutter / nose radius compensation stage
// --------------------------------------------------------------------------

function lineLine(p1, d1, p2, d2) {
  const den = cross2(d1, d2);
  if (Math.abs(den) < 1e-12) return [];
  const t = cross2(sub2(p2, p1), d2) / den;
  return [add2(p1, mul2(d1, t))];
}
function lineCircle(p, d, c, r) {
  const f = sub2(p, c);
  const b = dot2(f, d);
  const cc = dot2(f, f) - r * r;
  const disc = b * b - cc;
  if (disc < -1e-9) return [];
  const s = Math.sqrt(Math.max(0, disc));
  return [add2(p, mul2(d, -b - s)), add2(p, mul2(d, -b + s))];
}
function circleCircle(c1, r1, c2, r2) {
  const d = len2(sub2(c2, c1));
  if (d < 1e-12 || d > r1 + r2 + 1e-9 || d < Math.abs(r1 - r2) - 1e-9) return [];
  const a = (r1 * r1 - r2 * r2 + d * d) / (2 * d);
  const h = Math.sqrt(Math.max(0, r1 * r1 - a * a));
  const ex = mul2(sub2(c2, c1), 1 / d);
  const m = add2(c1, mul2(ex, a));
  const ey = left2(ex);
  return [add2(m, mul2(ey, h)), sub2(m, mul2(ey, h))];
}

export class CompStage {
  // next: sink with push(prim) ; warn(meta,msg) ; error(meta,msg)
  constructor(next, warn, error) {
    this.next = next;
    this.warn = warn;
    this.error = error;
    this.side = 0;      // +1 = G41 (left), -1 = G42 (right)
    this.r = 0;
    this.plane = 17;
    this.prev = null;   // { prim, startOff:uv, neutrals: [] }
    this.startup = null; // { prim }
    this.pos = null;    // current actual tool position (3D)
  }

  setPosition(p) { this.pos = p.slice(); }

  // opt: { side, radius, plane, start: bool, cancel: bool }
  push(prim, opt = {}) {
    if (!this.pos) this.pos = prim.a.slice();
    const plane = opt.plane ?? this.plane;
    if (opt.start && opt.side) {
      this.flushChain();
      this.side = opt.side; this.r = opt.radius; this.plane = plane;
      if (prim.type === 'arc') {
        this.error(prim.meta, 'Cutter compensation cannot start on an arc (start G41/G42 on a straight move)');
        this.side = 0;
        this._emit(prim);
        return;
      }
      if (!dirAt(prim, plane, false)) {
        // e.g. "G41 D1" on a Z-only move: activate, nothing to offset yet
        this._emit(prim);
        this.startupPending = true;
        return;
      }
      this.startup = { prim };
      return;
    }
    if (!this.side) { this._emit(prim); return; }
    if (opt.side !== undefined && opt.side !== this.side && opt.side !== 0 && !opt.cancel) {
      // switching G41 <-> G42 mid-contour: finish the chain, restart
      this.flushChain();
      this.side = opt.side; this.r = opt.radius;
    }
    if (opt.radius !== undefined) this.r = opt.radius;
    if (opt.cancel) {
      this._cancel(prim);
      return;
    }
    const inPlane = prim.type === 'arc' ? prim.plane === this.plane : !!dirAt(prim, this.plane, false);
    if (!inPlane) {
      // Z-only (neutral) move: keep it, it happens at the corner point
      if (this.prev) this.prev.neutrals.push(prim);
      else if (this.startup) { this.startup.neutrals = this.startup.neutrals || []; this.startup.neutrals.push(prim); }
      else this._emitShifted(prim);
      return;
    }
    if (this.startup) {
      const S = this.startup;
      this.startup = null;
      const d = dirAt(prim, this.plane, false);
      const Puv = add2(uv(S.prim.b, this.plane), mul2(left2(d), this.side * this.r));
      const target = fromUV(Puv[0], Puv[1], wOf(S.prim.b, this.plane), this.plane);
      this._emit({ ...S.prim, a: this.pos.slice(), b: target });
      for (const n of S.neutrals || []) this._emitShifted(n);
      this.prev = { prim, startOff: Puv, neutrals: [] };
      return;
    }
    if (this.startupPending) {
      this.startupPending = false;
      // first in-plane move after compensation was switched on along Z: treat as startup move
      const d = dirAt(prim, this.plane, true);
      const Puv = add2(uv(prim.b, this.plane), mul2(left2(d || [1, 0]), this.side * this.r));
      this._emit({ ...prim, type: 'line', a: this.pos.slice(), b: fromUV(Puv[0], Puv[1], wOf(prim.b, this.plane), this.plane) });
      this.warn(prim.meta, 'Compensation start-up on this move (G41/G42 was programmed without an XY move)');
      return;
    }
    if (!this.prev) {
      // re-entering after a flush: start perpendicular to the element
      const d = dirAt(prim, this.plane, false);
      const Puv = add2(uv(prim.a, this.plane), mul2(left2(d), this.side * this.r));
      if (len2(sub2(Puv, uv(this.pos, this.plane))) > 1e-6) {
        this._emit({ type: 'line', kind: prim.kind, a: this.pos.slice(), b: fromUV(Puv[0], Puv[1], wOf(this.pos, this.plane), this.plane), meta: prim.meta });
      }
      this.prev = { prim, startOff: Puv, neutrals: [] };
      return;
    }
    this._join(prim);
  }

  _offsetGeom(prim) {
    // returns {type, p, d} for lines (through offset start, direction d) or {type, c, r} for arcs
    const s = this.side * this.r;
    if (prim.type === 'line') {
      const d = dirAt(prim, this.plane, false);
      return { type: 'line', p: add2(uv(prim.a, this.plane), mul2(left2(d), s)), d };
    }
    const ccw = prim.sweep > 0 ? 1 : -1;
    return { type: 'arc', c: uv(prim.c, this.plane), r: prim.radius - s * ccw };
  }

  _endPerp(prim) {
    const d = dirAt(prim, this.plane, true);
    return add2(uv(prim.b, this.plane), mul2(left2(d), this.side * this.r));
  }
  _startPerp(prim) {
    const d = dirAt(prim, this.plane, false);
    return add2(uv(prim.a, this.plane), mul2(left2(d), this.side * this.r));
  }

  _join(E) {
    const P = this.prev;
    const A = P.prim;
    const dA = dirAt(A, this.plane, true);
    const dB = dirAt(E, this.plane, false);
    const corner = uv(A.b, this.plane);
    const turn = cross2(dA, dB);
    const dot = dot2(dA, dB);
    const endA = this._endPerp(A);
    const startB = this._startPerp(E);
    let joint = null;
    let arcAround = false;
    if (Math.abs(turn) < 1e-7 && dot > 0) {
      joint = endA;
    } else if (this.side * turn < 0 || (Math.abs(turn) < 1e-7 && dot < 0)) {
      arcAround = true;
    } else {
      const gA = this._offsetGeom(A), gB = this._offsetGeom(E);
      let cands = [];
      if (gA.type === 'line' && gB.type === 'line') cands = lineLine(gA.p, gA.d, gB.p, gB.d);
      else if (gA.type === 'line') cands = lineCircle(gA.p, gA.d, gB.c, gB.r);
      else if (gB.type === 'line') cands = lineCircle(gB.p, gB.d, gA.c, gA.r);
      else cands = circleCircle(gA.c, gA.r, gB.c, gB.r);
      let best = null, bd = Infinity;
      for (const c of cands) {
        const dd = len2(sub2(c, corner));
        if (dd < bd) { bd = dd; best = c; }
      }
      if (best && bd < Math.max(10 * this.r, 1)) joint = best;
      else {
        this.warn(E.meta, 'Cutter compensation: inside corner could not be resolved (tool radius too large?) — path may gouge');
        joint = null;
      }
    }
    if (joint) {
      this._emitOffset(A, P.startOff, joint);
      for (const n of P.neutrals) this._emitShifted(n);
      this.prev = { prim: E, startOff: joint, neutrals: [] };
      return;
    }
    // outside corner (or unresolved inside corner): go to perpendicular end, arc around the corner
    this._emitOffset(A, P.startOff, endA);
    for (const n of P.neutrals) this._emitShifted(n);
    if (len2(sub2(endA, startB)) > 1e-6) {
      const w = wOf(this.pos, this.plane);
      const from = fromUV(endA[0], endA[1], w, this.plane);
      const to = fromUV(startB[0], startB[1], w, this.plane);
      if (arcAround) {
        const { prim } = buildArc(from, to, corner, turn < 0 || (Math.abs(turn) < 1e-7 && this.side > 0), this.plane, E.kind, E.meta, 0.05);
        if (prim) this._emit(prim);
      } else {
        this._emit({ type: 'line', kind: E.kind, a: from, b: to, meta: E.meta });
      }
    }
    this.prev = { prim: E, startOff: startB, neutrals: [] };
  }

  _emitOffset(prim, startUV, endUV) {
    const a = fromUV(startUV[0], startUV[1], wOf(prim.a, this.plane), this.plane);
    const b = fromUV(endUV[0], endUV[1], wOf(prim.b, this.plane), this.plane);
    if (prim.type === 'line') {
      this._emit({ ...prim, a, b });
      return;
    }
    const g = this._offsetGeom(prim);
    if (g.r <= 1e-6) {
      this.error(prim.meta, `Tool radius ${this.r} is too large for arc radius ${prim.radius.toFixed(3)} (gouge)`);
      this._emit({ type: 'line', kind: prim.kind, a, b, meta: prim.meta });
      return;
    }
    const { prim: arc } = buildArc(a, b, g.c, prim.sweep < 0, this.plane, prim.kind, prim.meta, 0.05);
    if (!arc) return;
    if (!prim.full && Math.abs(arc.sweep) > Math.abs(prim.sweep) + Math.PI / 2 && Math.abs(prim.sweep) < 1.5 * Math.PI) {
      this.error(prim.meta, 'Cutter compensation reversed this arc (tool too large for the contour) — gouge');
      this._emit({ type: 'line', kind: prim.kind, a, b, meta: prim.meta });
      return;
    }
    if (prim.full && len2(sub2(startUV, endUV)) < 1e-6) { arc.full = true; arc.sweep = prim.sweep; }
    this._emit(arc);
  }

  _cancel(M) {
    // finish chain at perpendicular end, then go from there to the programmed end point
    if (this.prev) {
      const endA = this._endPerp(this.prev.prim);
      this._emitOffset(this.prev.prim, this.prev.startOff, endA);
      for (const n of this.prev.neutrals) this._emitShifted(n);
      this.prev = null;
    } else if (this.startup) {
      this._emit(this.startup.prim);
      this.startup = null;
    }
    this.side = 0;
    this.startupPending = false;
    this._emit({ ...M, type: 'line', a: this.pos.slice() });
  }

  flushChain() {
    if (this.prev) {
      const endA = this._endPerp(this.prev.prim);
      this._emitOffset(this.prev.prim, this.prev.startOff, endA);
      for (const n of this.prev.neutrals) this._emitShifted(n);
      this.prev = null;
    }
    if (this.startup) {
      this._emit(this.startup.prim);
      for (const n of this.startup.neutrals || []) this._emitShifted(n);
      this.startup = null;
    }
  }

  // end of program or mode change
  flush() {
    if (this.side && (this.prev || this.startup)) {
      const meta = (this.prev || this.startup).prim.meta;
      this.warn(meta, 'Cutter compensation still active at end of program (missing G40)');
    }
    this.flushChain();
    this.next.flush?.();
  }

  // emit a move whose in-plane position follows the current (offset) position
  _emitShifted(prim) {
    if (prim.type === 'event') {
      this.next.push({ ...prim, a: this.pos.slice(), b: this.pos.slice() });
      return;
    }
    const cur = uv(this.pos, this.plane);
    const a = this.pos.slice();
    const b = fromUV(cur[0], cur[1], wOf(prim.b, this.plane), this.plane);
    this._emit({ ...prim, type: 'line', a, b });
  }

  _emit(prim) {
    if (prim.type === 'event') {
      this.next.push({ ...prim, a: (this.pos || prim.a).slice(), b: (this.pos || prim.a).slice() });
      return;
    }
    // keep the actual start point continuous
    if (this.pos && dist3(prim.a, this.pos) > 1e-6 && prim.type === 'line') prim = { ...prim, a: this.pos.slice() };
    prim.center = this.side !== 0;
    this.next.push(prim);
    this.pos = prim.b.slice();
  }
}
