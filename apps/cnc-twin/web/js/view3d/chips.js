/* Chips (chips/kf_<k>.npz): the particles in flight as one instanced mesh, each chip in its own colour
 * (colour_rgb, the temper colour of the chips model) or, with the temperature layer, by temp_K; and
 * the chip piles as a height field.
 *
 * A chip is drawn as a curled ribbon of thickness : width : length = chip_aspect (an arc chip), its
 * length along the chip's velocity (else along a direction hashed from its index: stream
 * "view3d.chip_orientation"), its length `length_m`, else from its mass and the material density at
 * that aspect (volume = mass / density = t w l), else chip_default_length_m; never smaller on screen
 * than min_chip_size_px. Chips are given in M and drawn with the pose of S in M of their own keyframe,
 * so the snapshot is exact at the keyframe time; between keyframes the snapshot is held (the chips
 * model's trajectories include drag, collisions and coolant forces the view does not repeat).
 *
 * Piles: the pile grid's cells higher than min_pile_height_m, standing on the grid plane (or on
 * pile_base_m when the run gives the base height per cell), corner heights the mean of the adjacent
 * occupied cells (as cnctwin.render draws them). */

import * as THREE from './three.js';
import { hashUniform } from './hash.js';
import { gridFromArrays } from './fields.js';
import { lutColor } from './layers.js';

function ribbonGeometry(aspect, curlRad, segments) {
  const length = 1;
  const width = aspect[1] / aspect[2];
  const thick = aspect[0] / aspect[2];
  const radius = length / curlRad;
  const ring = [];
  for (let i = 0; i <= segments; i++) {
    const a = -curlRad / 2 + (curlRad * i) / segments;
    const out = [];
    for (const r of [radius + thick / 2, radius - thick / 2]) {
      for (const y of [-width / 2, width / 2]) out.push(new THREE.Vector3(r * Math.sin(a), y, radius - r * Math.cos(a)));
    }
    ring.push(out);
  }
  const positions = [];
  const quad = (a, b, c, d) => { for (const p of [a, b, c, a, c, d]) positions.push(p.x, p.y, p.z); };
  for (let i = 0; i < segments; i++) {
    const [o0, o1, n0, n1] = ring[i];
    const [p0, p1, m0, m1] = ring[i + 1];
    quad(o0, p0, p1, o1);
    quad(n1, m1, m0, n0);
    quad(o0, n0, m0, p0);
    quad(o1, p1, m1, n1);
  }
  const first = ring[0];
  const last = ring[segments];
  quad(first[0], first[1], first[3], first[2]);
  quad(last[0], last[2], last[3], last[1]);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  return geometry;
}

export class ChipsView {
  constructor({ root, params, density, section }) {
    this.root = root;
    this.params = params;
    this.density = density;
    this.geometry = ribbonGeometry(params.chip_aspect, params.chip_curl_rad, 6);
    this.material = new THREE.MeshStandardMaterial({ metalness: 0.1, roughness: 0.5 });
    section.clip(this.material);
    this.capacity = 0;
    this.mesh = null;
    this.chips = null;
    this.visible = true;
    this.pileMaterial = new THREE.MeshStandardMaterial({ vertexColors: true, metalness: 0.2, roughness: 0.8, side: THREE.DoubleSide, flatShading: true });
    section.clip(this.pileMaterial);
    this.pile = null;
    this.pileGrid = null;
    this.lastCamera = null;
  }

  _ensure(count) {
    if (this.mesh && count <= this.capacity) return;
    if (this.mesh) { this.mesh.removeFromParent(); this.mesh.dispose(); }
    this.capacity = Math.max(64, 2 ** Math.ceil(Math.log2(Math.max(1, count))));
    this.mesh = new THREE.InstancedMesh(this.geometry, this.material, this.capacity);
    this.mesh.name = 'chips';
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.mesh.frustumCulled = false;
    this.mesh.count = 0;
    this.root.add(this.mesh);
  }

  /** The chips of a keyframe (npz arrays) with pose_S_M of that keyframe (null: not drawable). */
  set(arrays, poseSM) {
    this.chips = null;
    if (!arrays || !arrays.pos_M_m || !poseSM) { if (this.mesh) this.mesh.count = 0; return; }
    const pos = arrays.pos_M_m.data;
    const n = Math.floor(pos.length / 3);
    const vel = arrays.vel_M_m_s && arrays.vel_M_m_s.data.length === n * 3 ? arrays.vel_M_m_s.data : null;
    const mass = arrays.mass_kg && arrays.mass_kg.data.length === n ? arrays.mass_kg.data : null;
    const len = arrays.length_m && arrays.length_m.data.length === n ? arrays.length_m.data : null;
    const temp = arrays.temp_K && arrays.temp_K.data.length === n ? arrays.temp_K.data : null;
    const rgb = arrays.colour_rgb && arrays.colour_rgb.data.length === n * 3 ? arrays.colour_rgb.data : null;
    const a = this.params.chip_aspect;
    const volumeFactor = (a[0] * a[1]) / (a[2] * a[2]);
    const rotation = new THREE.Matrix3().setFromMatrix4(poseSM);
    const chips = [];
    for (let i = 0; i < n; i++) {
      const p = new THREE.Vector3(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]).applyMatrix4(poseSM);
      if (![p.x, p.y, p.z].every(Number.isFinite)) continue;
      let dir = vel ? new THREE.Vector3(vel[i * 3], vel[i * 3 + 1], vel[i * 3 + 2]).applyMatrix3(rotation) : null;
      if (!dir || !(dir.lengthSq() > 1e-24)) {
        const u = hashUniform('view3d.chip_orientation', i, 0);
        const v = hashUniform('view3d.chip_orientation', i, 1);
        const z = 2 * u - 1;
        const r = Math.sqrt(Math.max(0, 1 - z * z));
        dir = new THREE.Vector3(r * Math.cos(2 * Math.PI * v), r * Math.sin(2 * Math.PI * v), z);
      }
      dir.normalize();
      let length = len && len[i] > 0 ? len[i] : null;
      if (!length && mass && mass[i] > 0 && this.density) length = Math.cbrt(mass[i] / this.density / volumeFactor);
      if (!length) length = this.params.chip_default_length_m;
      const roll = 2 * Math.PI * hashUniform('view3d.chip_orientation', i, 2);
      chips.push({ p, dir, length, roll, temp: temp ? temp[i] : null, rgb: rgb ? [rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]] : null });
    }
    this.chips = chips;
    this._ensure(chips.length);
    this.mesh.count = chips.length;
    this.lastCamera = null;
  }

  /** Colour the chips: their own colour, or by temperature on the layer's map. */
  colour(table = null, lo = 0, hi = 1) {
    if (!this.chips || !this.mesh) return;
    const c = new THREE.Color();
    const fallback = new THREE.Color().setRGB(...this.params.chip_default_srgb, THREE.SRGBColorSpace);
    for (let i = 0; i < this.chips.length; i++) {
      const chip = this.chips[i];
      if (table && chip.temp !== null && Number.isFinite(chip.temp)) lutColor(table, chip.temp, lo, hi, c);
      else if (chip.rgb) c.setRGB(chip.rgb[0] / 255, chip.rgb[1] / 255, chip.rgb[2] / 255, THREE.SRGBColorSpace);
      else c.copy(fallback);
      this.mesh.setColorAt(i, c);
    }
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
  }

  /** Instance matrices with the on-screen minimum size; only when the camera or the chips changed. */
  update(camera, viewportPx) {
    if (!this.mesh || !this.chips) return;
    this.mesh.visible = this.visible && this.chips.length > 0;
    const key = `${camera.position.toArray().join(',')}|${camera.fov}|${viewportPx}|${this.chips.length}`;
    if (key === this.lastCamera) return;
    this.lastCamera = key;
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const qRoll = new THREE.Quaternion();
    const s = new THREE.Vector3();
    const x = new THREE.Vector3(1, 0, 0);
    const tan = 2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
    for (let i = 0; i < this.chips.length; i++) {
      const chip = this.chips[i];
      const metresPerPx = (camera.position.distanceTo(chip.p) * tan) / Math.max(1, viewportPx);
      const size = Math.max(chip.length, this.params.min_chip_size_px * metresPerPx);
      q.setFromUnitVectors(x, chip.dir);
      qRoll.setFromAxisAngle(x, chip.roll);
      q.multiply(qRoll);
      m.compose(chip.p, q, s.set(size, size, size));
      this.mesh.setMatrixAt(i, m);
    }
    this.mesh.instanceMatrix.needsUpdate = true;
    this.mesh.computeBoundingSphere();
  }

  /** The pile grid of a keyframe: from the chips file (pile_*), else the fields' pile_height_m. */
  setPile(chipArrays, fieldArrays, stockSizeM, colourSrgb) {
    const source = chipArrays && chipArrays.pile_height_m && chipArrays.pile_height_m.shape.length === 2 ? chipArrays : fieldArrays;
    const grid = source ? gridFromArrays(source, 'pile_height_m', stockSizeM) : null;
    if (this.pile) { this.pile.removeFromParent(); this.pile.geometry.dispose(); this.pile = null; }
    this.pileGrid = grid;
    this.pileBase = null;
    if (!grid) return;
    const base = source.pile_base_m ? gridFromArrays({ ...source, pile_base_m: source.pile_base_m }, 'pile_base_m', stockSizeM) : null;
    const [nx, ny] = grid.n;
    const minH = this.params.min_pile_height_m;
    const occupied = (i, j) => i >= 0 && j >= 0 && i < nx && j < ny && grid.values[i + nx * j] > minH;
    const height = (i, j) => grid.values[i + nx * j] + (base && Number.isFinite(base.values[i + nx * j]) ? base.values[i + nx * j] : 0);
    const corner = (ci, cj) => {
      let sum = 0;
      let count = 0;
      for (const [i, j] of [[ci - 1, cj - 1], [ci, cj - 1], [ci - 1, cj], [ci, cj]]) if (occupied(i, j)) { sum += height(i, j); count++; }
      return count ? sum / count : 0;
    };
    const a = grid.axes;
    const o = [grid.origin[0], grid.origin[1], Number.isFinite(grid.origin[2]) ? grid.origin[2] : 0];
    const point = (u, v, w) => [0, 1, 2].map((c) => o[c] + a[0][c] * u + a[1][c] * v + a[2][c] * w);
    const positions = [];
    const values = [];
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        if (!occupied(i, j)) continue;
        const h00 = corner(i, j); const h10 = corner(i + 1, j); const h11 = corner(i + 1, j + 1); const h01 = corner(i, j + 1);
        const c = grid.cell;
        const p00 = point(i * c, j * c, h00); const p10 = point((i + 1) * c, j * c, h10);
        const p11 = point((i + 1) * c, (j + 1) * c, h11); const p01 = point(i * c, (j + 1) * c, h01);
        for (const p of [p00, p10, p11, p00, p11, p01]) positions.push(...p);
        for (const h of [h00, h10, h11, h00, h11, h01]) values.push(h);
      }
    }
    if (!positions.length) return;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(positions.length), 3));
    geometry.computeBoundingSphere();
    this.pileHeights = Float32Array.from(values);
    this.pile = new THREE.Mesh(geometry, this.pileMaterial);
    this.pile.name = 'chip pile';
    this.pile.renderOrder = 6;
    this.pile.visible = this.visible;
    this.root.add(this.pile);
    this.colourPile(null, 0, 1, colourSrgb);
  }

  /** Pile colour: chips' colour, or by height on the chips layer's map. */
  colourPile(table, lo, hi, colourSrgb) {
    if (!this.pile) return;
    const attr = this.pile.geometry.getAttribute('color');
    const c = new THREE.Color();
    const plain = new THREE.Color().setRGB(...(colourSrgb || this.params.chip_default_srgb), THREE.SRGBColorSpace);
    for (let i = 0; i < this.pileHeights.length; i++) {
      if (table) lutColor(table, this.pileHeights[i], lo, hi, c); else c.copy(plain);
      attr.setXYZ(i, c.r, c.g, c.b);
    }
    attr.needsUpdate = true;
  }

  /** Mean chip colour (sRGB [0, 1]) of the snapshot, or null. */
  meanColour() {
    if (!this.chips) return null;
    const list = this.chips.filter((c) => c.rgb);
    if (!list.length) return null;
    const sum = [0, 0, 0];
    for (const c of list) for (let k = 0; k < 3; k++) sum[k] += c.rgb[k] / 255;
    return sum.map((v) => v / list.length);
  }

  setVisible(on) {
    this.visible = !!on;
    if (this.mesh) this.mesh.visible = this.visible && !!this.chips && this.chips.length > 0;
    if (this.pile) this.pile.visible = this.visible;
  }

  dispose() {
    if (this.mesh) { this.mesh.removeFromParent(); this.mesh.dispose(); }
    if (this.pile) { this.pile.removeFromParent(); this.pile.geometry.dispose(); }
    this.geometry.dispose();
    this.material.dispose();
    this.pileMaterial.dispose();
  }
}
