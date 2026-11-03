/* Coolant: the nozzles of the scene, the jets of the newest keyframe (keyframe fields jet_*), their
 * shadows and spray.
 *
 * A jet is drawn along the coolant model's own path (jet_path_M_m) when the run records it; else along
 * its free trajectory from the nozzle: straight for gas and mist jets, for liquid jets the ballistic
 * curve p(s) = p0 + v0 t + g t^2 / 2 with t = s / |v0| (a coherent jet in still air; drag on a jet of a
 * few millimetres is below 1 % of its momentum over a metre, so the curve is the jet's centreline until
 * it breaks up), cut at the first obstacle the centreline meets (part, fixture, machine, tool: a ray
 * cast per segment) or at jet_max_length_m. Its radius starts at the continuity radius sqrt(Q / (pi v))
 * (or jet_diameter_m / 2) and widens with jet_half_angle_rad, as cnctwin.render draws it.
 *
 * Shadows (blocked ray segments, docs/models/coolant.md 4.2): a jet whose centreline is stopped before it
 * reaches the cutting zone (the coolant model's tool zone: a sphere of zone_radius_factor x the tool
 * radius, centred tool_zone_height_ratio tool radii above the tip on the tool axis) or that the run
 * marks blocked (jet_blocked_by) is continued past the obstacle
 * along its unobstructed trajectory up to the zone, in the "shadowed flow" colour of the coolant legend:
 * the part of the jet the obstacle takes away from the cut.
 *
 * Spray: drops leaving the impact point, each on a short ballistic arc; drop i of a jet has a phase from
 * hash(stream "view3d.spray", jet, i) and restarts every spray_life_s of simulated time with new
 * directions, so the same time always shows the same drops and playback animates them. A visual cue,
 * not a predicted drop field. Jets of head-mounted nozzles (scene nozzle attached_to a tool-chain link)
 * ride with the tool between keyframes; the others stay where the keyframe has them. */

import * as THREE from './three.js';
import { hashUniform } from './hash.js';
import { coolantColors } from './layers.js';

const GAS = new Set(['air', 'air_blast', 'gas', 'mist', 'mql', 'dry', 'cold_air']);

function columns(arrays, name, n, width) {
  const arr = arrays[`jet_${name}`];
  if (!arr || !arr.data || arr.data.length < n * width) return null;
  return arr.data;
}

/** Jets of a keyframe's arrays: [{id, kind, origin, direction, speed, flow, halfAngle, reach, path, diameter, blockedBy}] in M. */
export function jetsFromArrays(arrays) {
  if (!arrays) return [];
  const originArr = arrays.jet_origin_M_m;
  if (!originArr) return [];
  const n = originArr.shape.length === 2 ? originArr.shape[0] : Math.floor(originArr.data.length / 3);
  const ids = columns(arrays, 'nozzle_id', n, 1);
  const kinds = columns(arrays, 'kind', n, 1);
  const dirs = columns(arrays, 'direction_M', n, 3);
  const speeds = columns(arrays, 'speed_m_s', n, 1);
  const flows = columns(arrays, 'flow_m3_s', n, 1);
  const halves = columns(arrays, 'half_angle_rad', n, 1);
  const reach = columns(arrays, 'reach_fraction', n, 1);
  const diameters = columns(arrays, 'diameter_m', n, 1);
  const blocked = columns(arrays, 'blocked_by', n, 1);
  const pathArr = arrays.jet_path_M_m;
  const out = [];
  for (let i = 0; i < n; i++) {
    const o = originArr.data;
    const jet = {
      index: i, id: ids ? String(ids[i]) : `jet${i}`, kind: kinds ? String(kinds[i]) : 'flood',
      origin: new THREE.Vector3(o[i * 3], o[i * 3 + 1], o[i * 3 + 2]),
      direction: dirs ? new THREE.Vector3(dirs[i * 3], dirs[i * 3 + 1], dirs[i * 3 + 2]) : new THREE.Vector3(0, 0, -1),
      speed: speeds ? Number(speeds[i]) : null, flow: flows ? Number(flows[i]) : null,
      halfAngle: halves ? Number(halves[i]) : 0, reach: reach ? Number(reach[i]) : null,
      diameter: diameters ? Number(diameters[i]) : null, blockedBy: blocked ? String(blocked[i] || '') : '', path: null,
    };
    if (jet.direction.lengthSq() > 0) jet.direction.normalize();
    if (pathArr && pathArr.shape.length === 3 && pathArr.shape[0] === n && pathArr.shape[2] === 3) {
      const P = pathArr.shape[1];
      const points = [];
      for (let k = 0; k < P; k++) {
        const base = (i * P + k) * 3;
        const p = new THREE.Vector3(pathArr.data[base], pathArr.data[base + 1], pathArr.data[base + 2]);
        if ([p.x, p.y, p.z].every(Number.isFinite)) points.push(p);
      }
      if (points.length >= 2) jet.path = points;
    }
    if ([jet.origin.x, jet.origin.y, jet.origin.z].every(Number.isFinite)) out.push(jet);
  }
  return out;
}

/** Ballistic or straight centreline in S: points every `step` up to `length`. */
export function centreline(origin, direction, speed, gravity, liquid, length, samples) {
  const points = [];
  const v0 = direction.clone().multiplyScalar(speed && speed > 0 ? speed : 1);
  const vNorm = v0.length();
  for (let k = 0; k <= samples; k++) {
    const s = (length * k) / samples;
    const t = s / vNorm;
    const p = origin.clone().addScaledVector(v0, t);
    if (liquid && speed > 0) p.addScaledVector(gravity, 0.5 * t * t);
    points.push(p);
  }
  return points;
}

/** A tube along points with radius r(s) and alpha(s) (vertex colours with alpha). */
function tube(points, radiusAt, alphaAt, colour, sides) {
  const n = points.length;
  const positions = new Float32Array(n * sides * 3);
  const colours = new Float32Array(n * sides * 4);
  const index = [];
  const tangent = new THREE.Vector3();
  const normal = new THREE.Vector3();
  const binormal = new THREE.Vector3();
  let s = 0;
  const lengths = [0];
  for (let i = 1; i < n; i++) { s += points[i].distanceTo(points[i - 1]); lengths.push(s); }
  const total = s || 1;
  let previousNormal = null;
  for (let i = 0; i < n; i++) {
    tangent.subVectors(points[Math.min(n - 1, i + 1)], points[Math.max(0, i - 1)]).normalize();
    if (!previousNormal) {
      const helper = Math.abs(tangent.z) < 0.9 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(1, 0, 0);
      normal.crossVectors(tangent, helper).normalize();
    } else {
      normal.copy(previousNormal).addScaledVector(tangent, -previousNormal.dot(tangent)).normalize();
    }
    previousNormal = normal.clone();
    binormal.crossVectors(tangent, normal);
    const r = radiusAt(lengths[i], lengths[i] / total);
    const alpha = alphaAt(lengths[i] / total);
    for (let k = 0; k < sides; k++) {
      const a = (2 * Math.PI * k) / sides;
      const off = normal.clone().multiplyScalar(Math.cos(a) * r).addScaledVector(binormal, Math.sin(a) * r);
      const j = i * sides + k;
      positions[j * 3] = points[i].x + off.x;
      positions[j * 3 + 1] = points[i].y + off.y;
      positions[j * 3 + 2] = points[i].z + off.z;
      colours[j * 4] = colour.r; colours[j * 4 + 1] = colour.g; colours[j * 4 + 2] = colour.b; colours[j * 4 + 3] = alpha;
      if (i < n - 1) {
        const k1 = (k + 1) % sides;
        index.push(i * sides + k, (i + 1) * sides + k, (i + 1) * sides + k1, i * sides + k, (i + 1) * sides + k1, i * sides + k1);
      }
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colours, 4));
  geometry.setIndex(index);
  geometry.computeBoundingSphere();
  return geometry;
}

export class CoolantView {
  constructor({ root, params, nozzles, toolChainLinks }) {
    this.root = root;
    this.params = params;
    this.group = new THREE.Group();
    this.group.name = 'coolant';
    root.add(this.group);
    this.jetGroup = new THREE.Group();
    this.jetGroup.matrixAutoUpdate = false;
    this.group.add(this.jetGroup);
    this.headGroup = new THREE.Group();
    this.headGroup.matrixAutoUpdate = false;
    this.group.add(this.headGroup);
    this.nozzleGroup = new THREE.Group();
    this.nozzleGroup.matrixAutoUpdate = false;
    this.group.add(this.nozzleGroup);
    this.material = new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false, side: THREE.DoubleSide });
    this.shadowMaterial = new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false, side: THREE.DoubleSide });
    this.sprayMaterial = new THREE.PointsMaterial({ size: params.spray_drop_size_m, sizeAttenuation: true, transparent: true, opacity: params.jet_opacity, depthWrite: false });
    this.nozzleMaterial = new THREE.MeshStandardMaterial({ color: new THREE.Color().setRGB(...params.machine_srgb, THREE.SRGBColorSpace), metalness: 0.3, roughness: 0.5 });
    this.nozzles = Array.isArray(nozzles) ? nozzles : [];
    this.toolLinks = toolChainLinks || new Set();
    this.jets = [];
    this.spray = null;
    this.visible = { coolant: true, shadows: false };
    this.nozzleMeshes = [];
    this.headIds = new Set(this.nozzles.filter((n) => n && n.attached_to && (this.toolLinks.has(String(n.attached_to)) || /head|spindle|tool|^T0?$/i.test(String(n.attached_to)))).map((n) => String(n.id)));
    this._buildNozzles();
  }

  _buildNozzles() {
    const geometry = new THREE.CylinderGeometry(this.params.nozzle_radius_m * 0.6, this.params.nozzle_radius_m, this.params.nozzle_length_m, 12);
    geometry.translate(0, -this.params.nozzle_length_m / 2, 0);
    this.nozzleGeometry = geometry;
    for (const nozzle of this.nozzles) {
      const o = nozzle && nozzle.origin_M_m;
      const d = nozzle && nozzle.direction_M;
      if (!Array.isArray(o) || !Array.isArray(d)) continue;
      // a head-mounted nozzle moves with the tool: it is drawn at its jets' origins (setJets), not here
      if (this.headIds.has(String(nozzle.id))) continue;
      const mesh = this._nozzleMesh(new THREE.Vector3(...o.map(Number)), new THREE.Vector3(...d.map(Number)));
      mesh.name = `nozzle ${nozzle.id}`;
      this.nozzleGroup.add(mesh);
      this.nozzleMeshes.push({ mesh, id: String(nozzle.id) });
    }
  }

  /** A nozzle body behind `origin`, its orifice at the origin, aimed along `direction`. */
  _nozzleMesh(origin, direction) {
    const mesh = new THREE.Mesh(this.nozzleGeometry, this.nozzleMaterial);
    mesh.position.copy(origin);
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.clone().normalize());
    mesh.updateMatrix();
    return mesh;
  }

  _clear() {
    for (const child of [...this.jetGroup.children, ...this.headGroup.children]) {
      child.removeFromParent();
      if (child.geometry && child.geometry !== this.nozzleGeometry) child.geometry.dispose();
    }
    this.spray = null;
  }

  /** Jets of keyframe k. ctx: {poseSM (at the keyframe), gravityS, raycast(origin, dir, maxT) -> t|null,
   * zoneS (centre of the cutting zone at the keyframe), toolRadius, toolT0 (pose_S_T0 at the keyframe)}. */
  setJets(jets, ctx) {
    this._clear();
    this.jets = [];
    if (!jets.length || !ctx.poseSM) return;
    const colors = coolantColors();
    const jetColour = new THREE.Color().setStyle(colors.jet, THREE.SRGBColorSpace);
    const shadowColour = new THREE.Color().setStyle(colors.shadow, THREE.SRGBColorSpace);
    const p = this.params;
    const rotation = new THREE.Matrix3().setFromMatrix4(ctx.poseSM);
    const zoneRadius = p.zone_radius_factor * Math.max(ctx.toolRadius || 0, p.min_zone_radius_m);
    const toHead = ctx.toolT0 ? ctx.toolT0.clone().invert() : null;
    for (const jet of jets) {
      const liquid = !GAS.has(jet.kind);
      const origin = jet.origin.clone().applyMatrix4(ctx.poseSM);
      const direction = jet.direction.clone().applyMatrix3(rotation).normalize();
      let points;
      let stoppedAt = null;
      if (jet.path) {
        points = jet.path.map((q) => q.clone().applyMatrix4(ctx.poseSM));
      } else {
        const full = centreline(origin, direction, jet.speed, ctx.gravityS, liquid, p.jet_max_length_m, p.jet_samples);
        points = [full[0]];
        for (let i = 1; i < full.length; i++) {
          const seg = new THREE.Vector3().subVectors(full[i], full[i - 1]);
          const len = seg.length();
          const t = ctx.raycast(full[i - 1], seg.clone().divideScalar(len), len);
          if (t !== null) { stoppedAt = full[i - 1].clone().addScaledVector(seg, t / len); points.push(stoppedAt); break; }
          points.push(full[i]);
        }
        jet.fullPath = full;
      }
      const speed = jet.speed && jet.speed > 0 ? jet.speed : 1;
      const r0 = jet.diameter && jet.diameter > 0 ? jet.diameter / 2 : jet.flow && jet.flow > 0 ? Math.sqrt(jet.flow / (Math.PI * speed)) : p.nozzle_radius_m * 0.5;
      const widen = Math.tan(Math.max(0, jet.halfAngle || 0));
      const radiusAt = (s) => r0 + s * widen;
      const alphaAt = (f) => p.jet_opacity * (1 - (1 - p.jet_end_opacity_fraction) * f);
      const geometry = tube(points, radiusAt, alphaAt, jetColour, p.jet_sides);
      const mesh = new THREE.Mesh(geometry, this.material);
      mesh.name = `jet ${jet.id}`;
      mesh.renderOrder = 9;
      const end = points[points.length - 1];
      const head = this.headIds.has(jet.id) && toHead;
      (head ? this.headGroup : this.jetGroup).add(mesh);
      if (head) {
        mesh.applyMatrix4(toHead);
        const body = this._nozzleMesh(origin, direction);
        body.name = `nozzle ${jet.id}`;
        body.applyMatrix4(toHead);
        this.headGroup.add(body);
      }
      const record = { jet, mesh, end: end.clone(), direction, head, shadow: null, liquid, stopped: !!stoppedAt };
      // shadow: the jet's unobstructed continuation, when the zone lies beyond the obstacle
      if (ctx.zoneS) {
        const zone = ctx.zoneS.clone();
        const reachedZone = points.some((q) => q.distanceTo(zone) <= zoneRadius);
        const blocked = jet.blockedBy !== '' || (stoppedAt && !reachedZone);
        if (blocked) {
          let continuation;
          if (jet.fullPath) {
            const startIndex = Math.max(0, points.length - 1);
            continuation = [end, ...jet.fullPath.slice(startIndex)];
          } else {
            const last = points[points.length - 1];
            const before = points[Math.max(0, points.length - 2)];
            const dir = new THREE.Vector3().subVectors(last, before).normalize();
            continuation = centreline(last, dir, speed, ctx.gravityS, liquid, p.jet_max_length_m, p.jet_samples);
          }
          let closest = 0;
          let best = Infinity;
          for (let i = 0; i < continuation.length; i++) {
            const d = continuation[i].distanceTo(zone);
            if (d < best) { best = d; closest = i; }
          }
          if (closest >= 1 && best <= Math.max(zoneRadius * 4, p.max_shadow_miss_m)) {
            const shadowPoints = continuation.slice(0, closest + 1);
            const shadowGeometry = tube(shadowPoints, (s) => radiusAt(0) + s * widen, () => p.shadow_opacity, shadowColour, p.jet_sides);
            const shadow = new THREE.Mesh(shadowGeometry, this.shadowMaterial);
            shadow.name = `shadow of jet ${jet.id}${jet.blockedBy ? ` (${jet.blockedBy})` : ''}`;
            shadow.renderOrder = 9;
            shadow.visible = this.visible.shadows;
            (head ? this.headGroup : this.jetGroup).add(shadow);
            if (head) shadow.applyMatrix4(toHead);
            record.shadow = shadow;
            record.blockedBy = jet.blockedBy || 'obstacle';
          }
        }
      }
      this.jets.push(record);
    }
    this._buildSpray(ctx.gravityS);
    this.setVisible(this.visible);
  }

  _buildSpray(gravity) {
    const count = this.jets.length * this.params.spray_drops_per_jet;
    if (!count) return;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(count * 3), 3));
    const colors = coolantColors();
    this.sprayMaterial.color.setStyle(colors.jet, THREE.SRGBColorSpace);
    this.spray = new THREE.Points(geometry, this.sprayMaterial);
    this.spray.name = 'coolant spray';
    this.spray.frustumCulled = false;
    this.spray.renderOrder = 10;
    this.jetGroup.add(this.spray);
    this.gravity = gravity.clone();
  }

  /** Spray drops at simulated time t (head-mounted jets carried with headMatrix = pose_S_T0(t)). */
  update(t, headMatrix) {
    if (headMatrix) { this.headGroup.matrix.copy(headMatrix); this.headGroup.matrixWorldNeedsUpdate = true; }
    if (!this.spray) return;
    const p = this.params;
    const attr = this.spray.geometry.getAttribute('position');
    const life = p.spray_life_s;
    const tmpDir = new THREE.Vector3();
    const side1 = new THREE.Vector3();
    const side2 = new THREE.Vector3();
    let j = 0;
    for (let jetIndex = 0; jetIndex < this.jets.length; jetIndex++) {
      const record = this.jets[jetIndex];
      const end = record.head && headMatrix ? record.end.clone().applyMatrix4(record.mesh.matrix).applyMatrix4(headMatrix) : record.end;
      const back = record.direction.clone().negate();
      side1.crossVectors(back, Math.abs(back.z) < 0.9 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(1, 0, 0)).normalize();
      side2.crossVectors(back, side1);
      const speed = (record.jet.speed > 0 ? record.jet.speed : 1) * p.spray_speed_fraction;
      for (let i = 0; i < p.spray_drops_per_jet; i++, j++) {
        const phase = hashUniform('view3d.spray', jetIndex, i, 0);
        const cycles = t / life + phase;
        const cycle = Math.floor(cycles);
        const age = (cycles - cycle) * life;
        const a = 2 * Math.PI * hashUniform('view3d.spray', jetIndex, i, cycle * 4 + 1);
        const spread = hashUniform('view3d.spray', jetIndex, i, cycle * 4 + 2);
        const lift = hashUniform('view3d.spray', jetIndex, i, cycle * 4 + 3);
        tmpDir.copy(side1).multiplyScalar(Math.cos(a) * spread).addScaledVector(side2, Math.sin(a) * spread).addScaledVector(back, 0.35 + 0.65 * lift).normalize();
        const v = speed * (0.4 + 0.6 * hashUniform('view3d.spray', jetIndex, i, cycle * 4 + 4));
        attr.setXYZ(j, end.x + tmpDir.x * v * age + 0.5 * this.gravity.x * age * age,
          end.y + tmpDir.y * v * age + 0.5 * this.gravity.y * age * age,
          end.z + tmpDir.z * v * age + 0.5 * this.gravity.z * age * age);
      }
    }
    attr.needsUpdate = true;
  }

  /** Nozzles (M) with pose_S_M at the time shown. */
  placeNozzles(poseSM) {
    this.nozzleGroup.visible = !!poseSM && this.visible.coolant;
    if (poseSM) { this.nozzleGroup.matrix.copy(poseSM); this.nozzleGroup.matrixWorldNeedsUpdate = true; }
  }

  setVisible({ coolant, shadows }) {
    this.visible = { coolant: coolant !== false, shadows: shadows === true };
    this.jetGroup.visible = this.visible.coolant;
    this.headGroup.visible = this.visible.coolant;
    for (const record of this.jets) if (record.shadow) record.shadow.visible = this.visible.shadows;
  }

  /** What the jets of the keyframe do, for the status line. */
  summary() {
    return this.jets.map((r) => ({ id: r.jet.id, kind: r.jet.kind, reach: r.jet.reach, shadowed: !!r.shadow, blockedBy: r.blockedBy || null }));
  }

  dispose() {
    this._clear();
    this.group.removeFromParent();
    this.nozzleGeometry.dispose();
    for (const m of [this.material, this.shadowMaterial, this.sprayMaterial, this.nozzleMaterial]) m.dispose();
  }
}
