/* The in-process workpiece on screen: the bricks the loader shows, merged into chunks of
 * chunk_bricks^3 bricks (one draw call per chunk instead of one per brick), each chunk at the level of
 * detail its distance asks for (lod.js: level 0 is the recorded mesh, levels 1, 2 vertex clusters of
 * lod_cell_factors lattice cells), built lazily within a time budget per frame; a chunk keeps showing
 * its previous meshes until the rebuilt ones are ready, so incremental replacement never flickers.
 * Frustum culling is per chunk (three.js, bounding spheres). Flat shading from screen-space
 * derivatives, so no normals are stored: the recorded facets are shown as they are.
 *
 * Picking casts the ray against the bricks' own triangles (Moeller and Trumbore, "Fast, minimum storage
 * ray-triangle intersection", J. Graphics Tools 2(1), 1997), never against a coarser level, with the
 * section plane respected (hits on the removed side are skipped; a ray that enters the kept side
 * inside material hits the section cap). */

import * as THREE from './three.js';
import { brickPositions, workpieceOf } from './bricks.js';
import { clusterMesh, clusterValues, chooseLevel, NO_VALUE } from './lod.js';

function nowMs() { return typeof performance !== 'undefined' ? performance.now() : Date.now(); }

function releaseArray() { this.array = null; }

export class IPWRenderer {
  constructor({ parent, material, sectionMaterials, params, layout, workpiecePose }) {
    this.parent = parent;
    this.material = material;
    this.sectionMaterials = sectionMaterials;
    this.params = params;
    this.workpiecePose = workpiecePose;
    this.groups = new Map();
    this.chunks = new Map();
    this.brickChunk = new Map();
    this.bricks = new Map();
    this.attributeProvider = null;
    this.attributeKey = null;
    this.section = false;
    this.lattice = Number(layout.lattice_m) || null;
    this.cells = Number(layout.cells) || 32;
    this.grid = Array.isArray(layout.grid) && layout.grid.length === 3 ? layout.grid.map(Number) : null;
    this.gridOrigin = Array.isArray(layout.origin_S_m) ? layout.origin_S_m.map(Number) : [0, 0, 0];
    this.brickM = this.lattice ? this.lattice * this.cells : null;
    this.stats = { chunks: 0, built: 0, triangles: 0, pending: 0 };
    this.version = 0;
  }

  get count() { return this.bricks.size; }

  _group(workpiece) {
    if (this.groups.has(workpiece)) return this.groups.get(workpiece);
    const group = new THREE.Group();
    group.name = `ipw-w${workpiece}`;
    group.matrixAutoUpdate = false;
    const pose = workpiece === 0 ? new THREE.Matrix4() : this.workpiecePose(workpiece);
    if (!pose) { group.visible = false; } else { group.matrix.copy(pose); }
    group.userData.hasPose = !!pose;
    this.parent.add(group);
    this.groups.set(workpiece, group);
    return group;
  }

  _chunkOf(key, brick) {
    if (this.brickChunk.has(key)) return this.brickChunk.get(key);
    const per = this.params.chunk_bricks;
    const workpiece = workpieceOf(key);
    let coord;
    const id = Number(/(\d+)$/.exec(key) ? /(\d+)$/.exec(key)[1] : NaN);
    if (this.grid && Number.isInteger(id)) {
      const [nx, ny] = this.grid;
      coord = [id % nx, Math.floor(id / nx) % ny, Math.floor(id / (nx * ny))].map((c) => Math.floor(c / per));
    } else {
      if (!this.brickM) {
        const b = brick.bbox;
        this.brickM = Math.max(b.hi[0] - b.lo[0], b.hi[1] - b.lo[1], b.hi[2] - b.lo[2]) || 1e-3;
        this.lattice = this.lattice || this.brickM / this.cells;
      }
      const size = this.brickM * per;
      coord = [0, 1, 2].map((c) => Math.floor(((brick.bbox.lo[c] + brick.bbox.hi[c]) / 2 - this.gridOrigin[c]) / size));
    }
    const chunkId = `w${workpiece}:${coord.join(',')}`;
    this.brickChunk.set(key, chunkId);
    if (!this.chunks.has(chunkId)) {
      const group = new THREE.Group();
      group.name = `chunk ${chunkId}`;
      this._group(workpiece).add(group);
      this.chunks.set(chunkId, { id: chunkId, workpiece, coord, bricks: new Map(), dirty: true, levels: [], stale: null, shown: null, group, sphere: null });
    }
    return chunkId;
  }

  /** Show `brick` for `key` (null: the brick is gone). */
  setBrick(key, brick) {
    if (!brick) {
      if (!this.bricks.has(key)) return;
      this.bricks.delete(key);
      const chunk = this.chunks.get(this.brickChunk.get(key));
      if (chunk) { chunk.bricks.delete(key); chunk.dirty = true; }
      this.version++;
      return;
    }
    if (!brick.bbox) return;
    const chunk = this.chunks.get(this._chunkOf(key, brick));
    this.bricks.set(key, brick);
    chunk.bricks.set(key, brick);
    chunk.dirty = true;
    this.version++;
  }

  /** Per-vertex values for an attribute layer: provider(key, brick) -> Float32Array [n] or null. */
  setAttributes(provider, key) {
    if (this.attributeKey === key && this.attributeProvider === provider) return;
    this.attributeProvider = provider;
    this.attributeKey = key;
    for (const chunk of this.chunks.values()) chunk.dirty = true;
  }

  /** Rebuild every chunk from the bricks (after a lost and restored WebGL context: the GPU buffers are
   * gone and their CPU copies were released after upload). */
  rebuildAll() {
    for (const chunk of this.chunks.values()) {
      for (const entry of chunk.levels) if (entry) this._disposeLevel(entry);
      if (chunk.stale) for (const entry of chunk.stale) if (entry) this._disposeLevel(entry);
      chunk.levels = [];
      chunk.stale = null;
      chunk.shown = null;
      chunk.dirty = true;
      for (const child of [...chunk.group.children]) chunk.group.remove(child);
    }
  }

  setSection(on) {
    this.section = !!on;
    for (const chunk of this.chunks.values()) this._attach(chunk, chunk.shown);
  }

  /** Levels' cluster cells (m): level 0 is the lattice. */
  levelCells() {
    const lattice = this.lattice || (this.brickM ? this.brickM / this.cells : 1e-3);
    return [lattice, ...this.params.lod_cell_factors.map((f) => f * lattice)];
  }

  _merge(chunk) {
    let vertices = 0;
    let indices = 0;
    const list = [...chunk.bricks.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1));
    for (const [, brick] of list) { vertices += brick.nVertices; indices += brick.nIndices; }
    const positions = new Float32Array(vertices * 3);
    const index = new Uint32Array(indices);
    // every chunk carries the attribute (NO_VALUE without an attribute layer), so a chunk still shown from
    // before a layer change reads as "no value" (material colour), never as a value of 0
    const values = new Float32Array(vertices).fill(NO_VALUE);
    let v = 0;
    let i = 0;
    for (const [key, brick] of list) {
      positions.set(brickPositions(brick), v * 3);
      const src = brick.indices;
      for (let j = 0; j < src.length; j++) index[i + j] = src[j] + v;
      if (this.attributeProvider) {
        const attr = this.attributeProvider(key, brick);
        if (attr && attr.length === brick.nVertices) values.set(attr, v);
      }
      v += brick.nVertices;
      i += src.length;
    }
    return { positions, index, values };
  }

  _anchor(chunk) {
    const per = this.params.chunk_bricks;
    const size = (this.brickM || 1e-3) * per;
    return chunk.coord.map((c, k) => this.gridOrigin[k] + c * size);
  }

  _geometry(positions, index, values) {
    const geometry = new THREE.BufferGeometry();
    const count = positions.length / 3;
    const pos = new THREE.BufferAttribute(positions, 3);
    geometry.setAttribute('position', pos);
    const idx = count < 65536 ? new THREE.BufferAttribute(Uint16Array.from(index), 1) : new THREE.BufferAttribute(index, 1);
    geometry.setIndex(idx);
    if (values) geometry.setAttribute('aLayer', new THREE.BufferAttribute(values, 1));
    geometry.computeBoundingSphere();
    geometry.computeBoundingBox();
    pos.onUpload(releaseArray);
    idx.onUpload(releaseArray);
    if (values) geometry.getAttribute('aLayer').onUpload(releaseArray);
    return geometry;
  }

  _build(chunk, level) {
    const merged = this._merge(chunk);
    let { positions, index, values } = merged;
    if (level > 0) {
      const cells = this.levelCells();
      const clustered = clusterMesh(positions, index, cells[level], this._anchor(chunk), values);
      positions = clustered.positions;
      index = clustered.indices;
      values = clustered.values;
    }
    const geometry = this._geometry(positions, index, values);
    const mesh = new THREE.Mesh(geometry, this.material);
    mesh.name = `ipw ${chunk.id} L${level}`;
    mesh.renderOrder = 6;
    mesh.userData.triangles = index.length / 3;
    return { mesh, geometry, triangles: index.length / 3, stencil: null };
  }

  _disposeLevel(entry) {
    if (!entry) return;
    entry.mesh.removeFromParent();
    if (entry.stencil) for (const m of entry.stencil) m.removeFromParent();
    entry.geometry.dispose();
  }

  _stencil(entry) {
    if (entry.stencil) return entry.stencil;
    const back = new THREE.Mesh(entry.geometry, this.sectionMaterials.back);
    const front = new THREE.Mesh(entry.geometry, this.sectionMaterials.front);
    back.renderOrder = 1;
    front.renderOrder = 1;
    back.name = `${entry.mesh.name} stencil back`;
    front.name = `${entry.mesh.name} stencil front`;
    entry.stencil = [back, front];
    return entry.stencil;
  }

  _attach(chunk, entry) {
    for (const child of [...chunk.group.children]) chunk.group.remove(child);
    chunk.shown = entry;
    if (!entry) return;
    chunk.group.add(entry.mesh);
    if (this.section) for (const m of this._stencil(entry)) chunk.group.add(m);
  }

  _pick(list, want) {
    if (!list) return null;
    if (list[want]) return list[want];
    for (let d = 1; d < 4; d++) {
      if (list[want - d]) return list[want - d];
      if (list[want + d]) return list[want + d];
    }
    return null;
  }

  _present(chunk, want) {
    const current = this._pick(chunk.levels, want);
    const entry = current || this._pick(chunk.stale, want);
    if (entry !== chunk.shown) this._attach(chunk, entry);
    if (current && chunk.stale) {
      for (const old of chunk.stale) if (old) this._disposeLevel(old);
      chunk.stale = null;
    }
    return entry;
  }

  /** Rebuild and choose levels; called once per frame. Returns true while work remains. */
  update(camera, viewportPx) {
    const budget = this.params.rebuild_budget_s * 1000;
    const start = nowMs();
    const cells = this.levelCells();
    const fov = THREE.MathUtils.degToRad(camera.fov);
    const maxPx = this.params.lodMaxPx;
    const work = [];
    const wantOf = new Map();
    for (const chunk of this.chunks.values()) {
      if (chunk.dirty) {
        if (chunk.stale) for (const old of chunk.stale) if (old) this._disposeLevel(old);
        chunk.stale = chunk.levels.some(Boolean) ? chunk.levels : null;
        chunk.levels = [];
        chunk.dirty = false;
        chunk.sphere = null;
        if (!chunk.bricks.size) {
          if (chunk.stale) for (const old of chunk.stale) if (old) this._disposeLevel(old);
          chunk.stale = null;
          this._attach(chunk, null);
          continue;
        }
      }
      if (!chunk.bricks.size) continue;
      if (!chunk.sphere) {
        const lo = [Infinity, Infinity, Infinity];
        const hi = [-Infinity, -Infinity, -Infinity];
        for (const brick of chunk.bricks.values()) {
          for (let c = 0; c < 3; c++) { lo[c] = Math.min(lo[c], brick.bbox.lo[c]); hi[c] = Math.max(hi[c], brick.bbox.hi[c]); }
        }
        const center = new THREE.Vector3((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2);
        chunk.sphere = { center, radius: Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2 };
      }
      const worldCenter = chunk.sphere.center.clone().applyMatrix4(chunk.group.matrixWorld);
      const distance = Math.max(worldCenter.distanceTo(camera.position) - chunk.sphere.radius, chunk.sphere.radius * 0.1, 1e-6);
      const want = this.section ? 0 : chooseLevel(cells, distance, fov, viewportPx, maxPx);
      wantOf.set(chunk, want);
      chunk.want = want;
      chunk.distance = distance;
      if (!chunk.levels[want]) work.push({ chunk, level: want, priority: distance });
      this._present(chunk, want);
    }
    work.sort((a, b) => a.priority - b.priority);
    let done = 0;
    for (const item of work) {
      if (done > 0 && nowMs() - start > budget) break;
      item.chunk.levels[item.level] = this._build(item.chunk, item.level);
      this._present(item.chunk, wantOf.get(item.chunk));
      done++;
    }
    let triangles = 0;
    let built = 0;
    for (const chunk of this.chunks.values()) if (chunk.shown) { triangles += chunk.shown.triangles; built++; }
    this.stats = { chunks: this.chunks.size, built, triangles, pending: work.length - done,
      levels: [...this.chunks.values()].map((c) => ({ id: c.id, want: c.want, shown: c.shown ? c.levels.indexOf(c.shown) : null, distance: c.distance, bricks: c.bricks.size })) };
    return work.length > done;
  }

  /** True when every shown brick is on screen in some built level (the stock can be hidden). */
  ready() {
    if (!this.bricks.size) return false;
    for (const chunk of this.chunks.values()) if (chunk.bricks.size && !chunk.shown) return false;
    return true;
  }

  /** Nearest hit of a ray (S of workpiece 0) on the shown bricks: {t, point, normal, key, workpiece,
   * triangle, bary, front, cap} or null. `plane` {normal, constant} keeps n.p + c >= 0. */
  raycast(origin, direction, plane = null) {
    let best = null;
    const local = new Map();
    for (const [workpiece, group] of this.groups) {
      if (!group.visible || !group.userData.hasPose) continue;
      const inverse = group.matrix.clone().invert();
      const o = origin.clone().applyMatrix4(inverse);
      const d = direction.clone().transformDirection(inverse);
      local.set(workpiece, { o, d, matrix: group.matrix });
    }
    const keptSide = (p) => !plane || plane.normal.dot(p) + plane.constant >= -1e-12;
    const hitsFrom = (tMin) => {
      let nearest = null;
      for (const [key, brick] of this.bricks) {
        const frame = local.get(workpieceOf(key));
        if (!frame) continue;
        const tBox = rayBox(frame.o, frame.d, brick.bbox);
        if (tBox === null || (nearest && tBox > nearest.t)) continue;
        const hit = rayBrick(frame.o, frame.d, brick, tMin, nearest ? nearest.t : Infinity, (p) => keptSide(p.clone().applyMatrix4(frame.matrix)));
        if (hit) { hit.key = key; hit.workpiece = workpieceOf(key); hit.matrix = frame.matrix; nearest = hit; }
      }
      return nearest;
    };
    best = hitsFrom(1e-9);
    if (plane) {
      const denom = plane.normal.dot(direction);
      if (Math.abs(denom) > 1e-12) {
        const tPlane = -(plane.normal.dot(origin) + plane.constant) / denom;
        const entering = denom > 0;
        if (tPlane > 0 && entering && (!best || tPlane < best.t)) {
          const beyond = hitsFrom(tPlane);
          if (beyond && !beyond.front) {
            const point = origin.clone().addScaledVector(direction, tPlane);
            best = { t: tPlane, point, normal: plane.normal.clone().negate(), key: beyond.key, workpiece: beyond.workpiece, cap: true, front: true };
          }
        }
      }
    }
    if (best && !best.cap) {
      best.point.applyMatrix4(best.matrix);
      best.normal.transformDirection(best.matrix);
    }
    return best;
  }

  dispose() {
    for (const chunk of this.chunks.values()) {
      for (const entry of chunk.levels) this._disposeLevel(entry);
      for (const entry of chunk.stale || []) this._disposeLevel(entry);
    }
    for (const group of this.groups.values()) group.removeFromParent();
    this.chunks.clear();
    this.bricks.clear();
  }
}

/** Entry distance of a ray into an axis-aligned box {lo, hi} (0 when inside), or null. */
export function rayBox(o, d, box) {
  let t0 = 0;
  let t1 = Infinity;
  const oo = [o.x, o.y, o.z];
  const dd = [d.x, d.y, d.z];
  for (let c = 0; c < 3; c++) {
    const pad = 1e-9;
    if (Math.abs(dd[c]) < 1e-300) {
      if (oo[c] < box.lo[c] - pad || oo[c] > box.hi[c] + pad) return null;
      continue;
    }
    let a = (box.lo[c] - pad - oo[c]) / dd[c];
    let b = (box.hi[c] + pad - oo[c]) / dd[c];
    if (a > b) { const s = a; a = b; b = s; }
    if (a > t0) t0 = a;
    if (b < t1) t1 = b;
    if (t0 > t1) return null;
  }
  return t0;
}

/** Nearest triangle of a brick hit with tMin < t < tMax whose point passes `accept`. */
export function rayBrick(o, d, brick, tMin, tMax, accept) {
  const p = brickPositions(brick);
  const idx = brick.indices;
  let best = null;
  const e1 = new THREE.Vector3();
  const e2 = new THREE.Vector3();
  const pv = new THREE.Vector3();
  const tv = new THREE.Vector3();
  const qv = new THREE.Vector3();
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  for (let i = 0; i + 2 < idx.length; i += 3) {
    a.fromArray(p, idx[i] * 3);
    b.fromArray(p, idx[i + 1] * 3);
    c.fromArray(p, idx[i + 2] * 3);
    e1.subVectors(b, a);
    e2.subVectors(c, a);
    pv.crossVectors(d, e2);
    const det = e1.dot(pv);
    if (Math.abs(det) < 1e-30) continue;
    const inv = 1 / det;
    tv.subVectors(o, a);
    const u = tv.dot(pv) * inv;
    if (u < 0 || u > 1) continue;
    qv.crossVectors(tv, e1);
    const v = d.dot(qv) * inv;
    if (v < 0 || u + v > 1) continue;
    const t = e2.dot(qv) * inv;
    if (t <= tMin || t >= tMax || (best && t >= best.t)) continue;
    const point = o.clone().addScaledVector(d, t);
    if (accept && !accept(point)) continue;
    const normal = new THREE.Vector3().crossVectors(e1, e2).normalize();
    best = { t, point, normal, triangle: i / 3, bary: [1 - u - v, u, v], front: normal.dot(d) < 0, vertices: [idx[i], idx[i + 1], idx[i + 2]] };
  }
  return best;
}

export { clusterValues };
