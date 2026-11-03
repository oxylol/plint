/* Measuring on the part (docs/ui-3d.md "Measuring"): point to point (two picks), point to plane (the
 * first pick gives a plane through the point normal to the surface there, the second the point whose
 * distance to it is wanted) and hole diameter (three picks on a circle of the hole: its rim or the
 * wall at one height). Picks are surface points in S (the ray hits of pick.js), so a measurement is
 * as exact as the recorded mesh: within half a lattice cell of the removal field for points on the
 * machined surface, plus the brick quantisation (step / 2, 0.06 um for 8 mm bricks).
 *
 * Circle through three points (circumcircle): with a = p2 - p1, b = p3 - p1, n = a x b,
 * centre = p1 + (|b|^2 (n x a) + |a|^2 (b x n)) / (2 |n|^2) — the point in their plane equidistant from
 * all three. Three points closer to a line than min_circle_sine (the sine of the smallest angle of the
 * triangle) are refused: the radius would be ill-conditioned. */

import * as THREE from './three.js';

/** {center, radius_m, normal} of the circle through three points, or null when they are (nearly) collinear. */
export function circleThrough(p1, p2, p3, minSine = 1e-3) {
  const a = new THREE.Vector3().subVectors(p2, p1);
  const b = new THREE.Vector3().subVectors(p3, p1);
  const c = new THREE.Vector3().subVectors(p3, p2);
  const n = new THREE.Vector3().crossVectors(a, b);
  const n2 = n.lengthSq();
  const la = a.length(); const lb = b.length(); const lc = c.length();
  if (!(n2 > 0) || !(la > 0) || !(lb > 0) || !(lc > 0)) return null;
  const area2 = Math.sqrt(n2);
  const sines = [area2 / (la * lb), area2 / (la * lc), area2 / (lb * lc)];
  if (Math.min(...sines) < minSine) return null;
  const term1 = new THREE.Vector3().crossVectors(n, a).multiplyScalar(b.lengthSq());
  const term2 = new THREE.Vector3().crossVectors(b, n).multiplyScalar(a.lengthSq());
  const center = p1.clone().add(term1.add(term2).divideScalar(2 * n2));
  return { center, radius_m: center.distanceTo(p1), normal: n.normalize() };
}

/** Signed distance of p from the plane through `origin` with unit `normal`, and the foot point. */
export function pointPlane(p, origin, normal) {
  const distance = new THREE.Vector3().subVectors(p, origin).dot(normal);
  return { distance_m: distance, foot: p.clone().addScaledVector(normal, -distance) };
}

export const MODES = [
  { id: 'point', label: 'Distance', picks: 2 },
  { id: 'plane', label: 'To plane', picks: 2 },
  { id: 'hole', label: 'Hole Ø', picks: 3 },
];

export class MeasureTool {
  constructor({ root, params, format, onResult }) {
    this.root = root;
    this.params = params;
    this.format = format;
    this.onResult = onResult;
    this.group = new THREE.Group();
    this.group.name = 'measurements';
    root.add(this.group);
    this.on = false;
    this.mode = 'point';
    this.picks = [];
    this.result = null;
    this.labels = [];
    const colour = new THREE.Color().setRGB(...params.measure_srgb, THREE.SRGBColorSpace);
    this.markerMaterial = new THREE.MeshBasicMaterial({ color: colour, depthTest: false, transparent: true, opacity: 0.95 });
    this.lineMaterial = new THREE.LineBasicMaterial({ color: colour, depthTest: false, transparent: true, opacity: 0.95 });
    this.planeMaterial = new THREE.MeshBasicMaterial({ color: colour, transparent: true, opacity: 0.18, depthWrite: false, side: THREE.DoubleSide });
    this.markerGeometry = new THREE.SphereGeometry(1, 12, 8);
  }

  setOn(on) {
    this.on = !!on;
    if (!this.on) this.clear();
  }

  setMode(mode) {
    if (!MODES.some((m) => m.id === mode)) return;
    this.mode = mode;
    this.clear();
  }

  clear() {
    this.picks = [];
    this.result = null;
    this.labels = [];
    for (const child of [...this.group.children]) {
      child.removeFromParent();
      if (child.geometry && child.geometry !== this.markerGeometry) child.geometry.dispose();
    }
  }

  get needed() { return MODES.find((m) => m.id === this.mode).picks; }

  /** Add a pick {point, normal}; returns the finished measurement or null. */
  add(hit) {
    if (!this.on || !hit) return null;
    if (this.picks.length >= this.needed) this.clear();
    this.picks.push({ point: hit.point.clone(), normal: hit.normal ? hit.normal.clone().normalize() : new THREE.Vector3(0, 0, 1) });
    this._marker(hit.point);
    if (this.picks.length < this.needed) return null;
    const result = this._measure();
    this.result = result;
    if (result && this.onResult) this.onResult(result);
    return result;
  }

  _marker(point) {
    const marker = new THREE.Mesh(this.markerGeometry, this.markerMaterial);
    marker.position.copy(point);
    marker.renderOrder = 20;
    marker.userData.marker = true;
    this.group.add(marker);
  }

  _line(points) {
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const line = new THREE.Line(geometry, this.lineMaterial);
    line.renderOrder = 20;
    this.group.add(line);
    return line;
  }

  _measure() {
    const [p1, p2, p3] = this.picks.map((p) => p.point);
    const asList = (v) => [v.x, v.y, v.z];
    if (this.mode === 'point') {
      const d = p1.distanceTo(p2);
      this._line([p1, p2]);
      this.labels = [{ at: p1.clone().lerp(p2, 0.5), text: this.format(d) }];
      return { kind: 'point', distance_m: d, points: [asList(p1), asList(p2)], delta_m: asList(p2.clone().sub(p1)) };
    }
    if (this.mode === 'plane') {
      const normal = this.picks[0].normal;
      const { distance_m, foot } = pointPlane(p2, p1, normal);
      const size = Math.max(Math.abs(distance_m), p1.distanceTo(p2), 1e-4) * 0.8;
      const patch = new THREE.Mesh(new THREE.PlaneGeometry(size, size), this.planeMaterial);
      patch.position.copy(p1);
      patch.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal);
      patch.renderOrder = 19;
      this.group.add(patch);
      this._line([p2, foot]);
      this._marker(foot);
      this.labels = [{ at: p2.clone().lerp(foot, 0.5), text: this.format(Math.abs(distance_m)) }];
      return { kind: 'plane', distance_m: Math.abs(distance_m), signed_distance_m: distance_m, points: [asList(p1), asList(p2)], foot_m: asList(foot), normal: asList(normal) };
    }
    const circle = circleThrough(p1, p2, p3, this.params.min_circle_sine);
    if (!circle) {
      this.labels = [{ at: p3.clone(), text: 'the three points lie on a line' }];
      return null;
    }
    const ring = [];
    const u = new THREE.Vector3().subVectors(p1, circle.center).normalize();
    const v = new THREE.Vector3().crossVectors(circle.normal, u);
    for (let i = 0; i <= 64; i++) {
      const a = (2 * Math.PI * i) / 64;
      ring.push(circle.center.clone().addScaledVector(u, Math.cos(a) * circle.radius_m).addScaledVector(v, Math.sin(a) * circle.radius_m));
    }
    this._line(ring);
    this._marker(circle.center);
    const diameter = 2 * circle.radius_m;
    this.labels = [{ at: circle.center.clone(), text: `Ø ${this.format(diameter)}` }];
    return { kind: 'hole', distance_m: diameter, diameter_m: diameter, radius_m: circle.radius_m, center_m: asList(circle.center), normal: asList(circle.normal), points: [asList(p1), asList(p2), asList(p3)] };
  }

  /** Keep markers at a constant size on screen. */
  update(camera, viewportPx) {
    const tan = 2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
    for (const child of this.group.children) {
      if (!child.userData.marker) continue;
      const metresPerPx = (camera.position.distanceTo(child.position) * tan) / Math.max(1, viewportPx);
      child.scale.setScalar((this.params.measure_marker_px / 2) * metresPerPx);
    }
  }

  dispose() {
    this.clear();
    this.group.removeFromParent();
    this.markerGeometry.dispose();
    this.markerMaterial.dispose();
    this.lineMaterial.dispose();
    this.planeMaterial.dispose();
  }
}
