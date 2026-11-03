/* Frames and poses of the 3D view (docs/architecture.md 4; conventions of cnctwin/geom/transform.py):
 * a pose is a 4 x 4 matrix acting on column vectors, pose_A_B maps B coordinates to A
 * (p_A = R p_B + t), pose_A_C = pose_A_B pose_B_C.
 *
 * The view draws in the stock frame S of workpiece 0 (metres). The tool comes from frames.bin: T0 at
 * the tip with +z_T0 along the recorded tool axis and x_T0 the hint made perpendicular to it (the
 * machine's x axis carried into S, which is T0 exactly whenever the tool chain does not rotate the
 * spindle: every three-axis mill and table-tilting five-axis machine); T is T0 turned by the spindle
 * angle about +z_T (right-handed, so M03 turns it clockwise seen from the spindle). Components in M
 * are placed with pose_S_M = inv(pose_M_S): from the work chain when the scene gives the stock in the
 * chain-end frame (stock.pose_F_S), else between the keyframes' recorded pose_M_S (interpolated, or
 * moved with the work chain between two keyframes when the chain reproduces the second one), else the
 * scene's static stock.pose_M_S. Machine links follow the chains of scene.json from the axis values:
 * T_i = T_{i-1} J_i(q) X_i (docs/models/motion.md 5, nominal: without the error transforms, which
 * only the tool and part poses carry). */

import * as THREE from './three.js';

const AXES = { x: new THREE.Vector3(1, 0, 0), y: new THREE.Vector3(0, 1, 0), z: new THREE.Vector3(0, 0, 1) };

function numbers(list, n) {
  if (!Array.isArray(list) || list.length !== n) return null;
  const out = list.map(Number);
  return out.every(Number.isFinite) ? out : null;
}

/** THREE.Matrix4 of a pose written as 4 x 4 rows, 16 numbers (row-major), {matrix}, or
 * {translate_m|translate_mm, rotate_rad|rotate_deg, order} (angles by axis, intrinsic in `order`).
 * Null when it is none of these. */
export function matrixFromPose(value) {
  if (value === null || value === undefined) return null;
  if (value instanceof THREE.Matrix4) return value.clone();
  if (Array.isArray(value)) {
    let flat = null;
    if (value.length === 4 && value.every((row) => Array.isArray(row) && row.length === 4)) flat = numbers(value.flat(), 16);
    else if (value.length === 16) flat = numbers(value, 16);
    if (!flat) return null;
    return new THREE.Matrix4().set(...flat);
  }
  if (typeof value === 'object') {
    if (value.matrix !== undefined) return matrixFromPose(value.matrix);
    const scaleT = value.translate_m !== undefined ? 1 : value.translate_mm !== undefined ? 1e-3 : null;
    const scaleR = value.rotate_rad !== undefined ? 1 : value.rotate_deg !== undefined ? Math.PI / 180 : null;
    if (scaleT === null && scaleR === null) return null;
    const t = scaleT === null ? [0, 0, 0] : numbers(value.translate_m ?? value.translate_mm, 3);
    const r = scaleR === null ? [0, 0, 0] : numbers(value.rotate_rad ?? value.rotate_deg, 3);
    if (!t || !r) return null;
    const order = String(value.order || 'xyz').toUpperCase();
    if (!/^[XYZ]{3}$/.test(order) || new Set(order).size !== 3) return null;
    const euler = new THREE.Euler(r[0] * (scaleR || 0), r[1] * (scaleR || 0), r[2] * (scaleR || 0), order);
    const m = new THREE.Matrix4().makeRotationFromEuler(euler);
    m.setPosition(t[0] * (scaleT || 0), t[1] * (scaleT || 0), t[2] * (scaleT || 0));
    return m;
  }
  return null;
}

/** A unit vector perpendicular to `z`: z crossed with the coordinate axis it is least aligned with
 * (lowest index on ties), as cnctwin.geom.transform.perpendicular. */
export function perpendicular(z, out = new THREE.Vector3()) {
  const a = [Math.abs(z.x), Math.abs(z.y), Math.abs(z.z)];
  let k = 0;
  if (a[1] < a[k]) k = 1;
  if (a[2] < a[k]) k = 2;
  const e = [AXES.x, AXES.y, AXES.z][k];
  return out.crossVectors(z, e).normalize();
}

const _x = new THREE.Vector3();
const _y = new THREE.Vector3();
const _z = new THREE.Vector3();
const _h = new THREE.Vector3();

/** pose_S_T: origin `tip`, z along `axis`, x the hint made perpendicular (or perpendicular(z)),
 * turned by `spinRad` about z. Writes `out`. */
export function toolMatrix(out, tip, axis, xHint, spinRad = 0) {
  _z.set(axis[0], axis[1], axis[2]);
  if (_z.lengthSq() < 1e-24) _z.set(0, 0, 1);
  _z.normalize();
  let ok = false;
  if (xHint) {
    _h.copy(xHint);
    _h.addScaledVector(_z, -_h.dot(_z));
    const n = _h.length();
    if (n > 1e-6 * Math.max(1, xHint.length())) { _x.copy(_h).divideScalar(n); ok = true; }
  }
  if (!ok) perpendicular(_z, _x);
  _y.crossVectors(_z, _x);
  if (spinRad) {
    const c = Math.cos(spinRad);
    const s = Math.sin(spinRad);
    _h.copy(_x).multiplyScalar(c).addScaledVector(_y, s);
    _y.copy(_x).multiplyScalar(-s).addScaledVector(_y, c);
    _x.copy(_h);
  }
  out.set(
    _x.x, _y.x, _z.x, tip[0],
    _x.y, _y.y, _z.y, tip[1],
    _x.z, _y.z, _z.z, tip[2],
    0, 0, 0, 1,
  );
  return out;
}

const _p0 = new THREE.Vector3();
const _p1 = new THREE.Vector3();
const _q0 = new THREE.Quaternion();
const _q1 = new THREE.Quaternion();
const _s = new THREE.Vector3();

/** Rigid interpolation of two poses: translation linear, rotation by slerp (Shoemake 1985). */
export function interpolatePose(a, b, w, out = new THREE.Matrix4()) {
  a.decompose(_p0, _q0, _s);
  b.decompose(_p1, _q1, _s);
  _p0.lerp(_p1, w);
  _q0.slerp(_q1, w);
  return out.compose(_p0, _q0, _s.set(1, 1, 1));
}

/** Translation distance (m) and rotation angle (rad) between two poses. */
export function poseDifference(a, b) {
  a.decompose(_p0, _q0, _s);
  b.decompose(_p1, _q1, _s);
  return { distance_m: _p0.distanceTo(_p1), angle_rad: _q0.angleTo(_q1) };
}

function unitVector(v) {
  const n = numbers(v, 3);
  if (!n) return null;
  const l = Math.hypot(n[0], n[1], n[2]);
  return l > 0 ? new THREE.Vector3(n[0] / l, n[1] / l, n[2] / l) : null;
}

/** One kinematic chain of scene.json: a list of link names, axis names, or link objects
 * {link, axis, direction, point_m | axis_point_m, fixed | matrix | pose}. */
export class Chain {
  constructor(entries, axes) {
    this.links = [];
    this.complete = Array.isArray(entries) && entries.length > 0;
    for (const raw of entries || []) {
      const entry = typeof raw === 'string' ? { name: raw } : { ...raw, name: raw.link ?? raw.name ?? raw.axis };
      const axisName = typeof raw === 'string' ? (axes.has(raw) ? raw : null) : (raw.axis ?? null);
      const axis = axisName !== null ? axes.get(axisName) : null;
      const link = { name: entry.name === undefined ? null : String(entry.name), axis: axisName, kind: null, direction: null, point: null, fixed: null };
      if (axisName !== null) {
        link.kind = (raw.kind || (axis && axis.kind)) === 'rotary' ? 'rotary' : 'linear';
        link.direction = unitVector(raw.direction ?? (axis && axis.direction));
        const point = numbers(raw.point_m ?? raw.axis_point_m ?? (axis && (axis.point_m ?? axis.axis_point_m)) ?? [0, 0, 0], 3);
        link.point = point ? new THREE.Vector3(...point) : new THREE.Vector3();
        if (!link.direction) this.complete = false;
      }
      if (typeof raw === 'object') {
        const fixed = raw.fixed ?? raw.matrix ?? raw.pose;
        if (fixed !== undefined) {
          link.fixed = matrixFromPose(fixed);
          if (!link.fixed) this.complete = false;
        }
      }
      this.links.push(link);
    }
    this.names = new Set(this.links.map((l) => l.name).filter((n) => n !== null));
    this.axisNames = new Set(this.links.map((l) => l.axis).filter((n) => n !== null));
  }

  has(name) { return this.names.has(name) || this.axisNames.has(name); }

  /** pose_M_<link> for every link name and the chain end, from absolute axis values {name: value}. */
  fk(values) {
    const poses = new Map();
    const T = new THREE.Matrix4();
    const J = new THREE.Matrix4();
    const R = new THREE.Matrix4();
    for (const link of this.links) {
      if (link.axis !== null && link.direction) {
        const q = Number(values[link.axis]) || 0;
        if (link.kind === 'rotary') {
          R.makeRotationAxis(link.direction, q);
          J.makeTranslation(link.point.x, link.point.y, link.point.z).multiply(R)
            .multiply(new THREE.Matrix4().makeTranslation(-link.point.x, -link.point.y, -link.point.z));
        } else {
          J.makeTranslation(link.direction.x * q, link.direction.y * q, link.direction.z * q);
        }
        T.multiply(J);
      }
      if (link.fixed) T.multiply(link.fixed);
      if (link.name !== null) poses.set(link.name, T.clone());
      if (link.axis !== null) poses.set(link.axis, T.clone());
    }
    return { poses, end: T.clone() };
  }
}

/** The chains of a scene: {work, tool, axes} (Chain or null). Accepts chain.{work, tool} or
 * kinematics.{workpiece_chain, tool_chain}. */
export function sceneChains(scene) {
  const axes = new Map();
  for (const axis of scene.axes || []) if (axis && axis.name !== undefined) axes.set(String(axis.name), axis);
  const chain = scene.chain || {};
  const kin = scene.kinematics || {};
  const work = chain.work ?? chain.workpiece ?? kin.workpiece_chain ?? null;
  const tool = chain.tool ?? kin.tool_chain ?? null;
  return { axes, work: work ? new Chain(work, axes) : null, tool: tool ? new Chain(tool, axes) : null };
}

/** pose_M_S over time (see the module comment). Keyframe poses arrive through setKeyframe as the
 * keyframe fields are read; validate() decides whether the work chain reproduces an interval. */
export class StockMotion {
  constructor({ staticPose = null, poseFS = null, workChain = null, keyframeTimes = [], toleranceM = 1e-4, toleranceRad = 1e-3 }) {
    this.staticPose = staticPose;
    this.poseFS = poseFS;
    this.chain = workChain && workChain.complete ? workChain : null;
    this.times = keyframeTimes.slice();
    this.keyed = new Map();
    this.anchors = new Map();
    this.chainOk = new Map();
    this.toleranceM = toleranceM;
    this.toleranceRad = toleranceRad;
  }

  setTimes(times) { this.times = times.slice(); }

  setKeyframe(k, matrix) { if (matrix) this.keyed.set(k, matrix); }

  /** Record the exact axis values at keyframes k and k+1 and test the chain on that interval. */
  validate(k, axesK, axesK1) {
    if (!this.chain || !this.keyed.has(k) || !this.keyed.has(k + 1)) return null;
    const wk = this.chain.fk(axesK).end;
    const wk1 = this.chain.fk(axesK1).end;
    const predicted = wk1.clone().multiply(wk.clone().invert()).multiply(this.keyed.get(k));
    const diff = poseDifference(predicted, this.keyed.get(k + 1));
    const ok = diff.distance_m <= this.toleranceM && diff.angle_rad <= this.toleranceRad;
    this.anchors.set(k, wk);
    this.chainOk.set(k, ok);
    return ok;
  }

  /** How the pose at keyframe k is known: 'chain', 'keyframes', 'static' or null. */
  source(k) {
    if (this.poseFS && this.chain) return 'chain';
    if (this.keyed.has(k)) return 'keyframes';
    if (this.staticPose) return 'static';
    return null;
  }

  /** pose_M_S at time t (k: newest keyframe at or before t, or -1; axes: absolute values at t). */
  at(t, k, axes, out = new THREE.Matrix4()) {
    if (this.poseFS && this.chain) return out.copy(this.chain.fk(axes).end).multiply(this.poseFS);
    if (k >= 0 && this.keyed.has(k)) {
      const pk = this.keyed.get(k);
      if (this.chain && this.chainOk.get(k) === true && this.anchors.has(k)) {
        return out.copy(this.chain.fk(axes).end).multiply(this.anchors.get(k).clone().invert()).multiply(pk);
      }
      if (this.keyed.has(k + 1) && k + 1 < this.times.length) {
        const t0 = this.times[k];
        const t1 = this.times[k + 1];
        const w = t1 > t0 ? Math.max(0, Math.min(1, (t - t0) / (t1 - t0))) : 0;
        return interpolatePose(pk, this.keyed.get(k + 1), w, out);
      }
      return out.copy(pk);
    }
    if (k < 0 && this.keyed.has(0)) return out.copy(this.keyed.get(0));
    if (this.staticPose) return out.copy(this.staticPose);
    return null;
  }
}
