/* The static scene of a run (scene.json, docs/formats.md "Run directory"): machine components,
 * fixture, enclosure and stock, their meshes (scene/<name>.glb, read with three.js's GLTFLoader and
 * baked into one geometry per component in the component's own frame), materials by kind, and their
 * placement in the view frame S at each time.
 *
 * Placement of a component (docs/ui-3d.md "Scene"): `frame` S (its `pose` in S), W (pose_S_W pose),
 * M (pose_S_M(t) pose), T0 (with the non-rotating tool frame of its path), T (with the tool, turning
 * with the spindle; housings of kind spindle/head/machine are drawn in T0: they do not turn), or
 * `link`: the chain's pose of that link from the axis values, pose_S_M(t) pose_M_link(q(t)) pose.
 * Without a frame: a link in the tool chain means T0 (the meshes are written about the tool tip),
 * another link the chain, no link M. A component whose placement is unknown at a time (M without any
 * pose_M_S, a chain with an axis direction missing) is not drawn, and the view says so. */

import * as THREE from './three.js';
import { GLTFLoader, mergeGeometries, toCreasedNormals } from './three.js';
import { matrixFromPose } from './poses.js';
import { NO_VALUE } from './lod.js';

const TOOL_SIDE_KINDS = new Set(['spindle', 'head', 'machine', 'housing']);

/** One geometry (positions + creased normals) of a GLB's meshes in its root frame. */
export function bakeGltf(gltf, creaseRad) {
  const parts = [];
  gltf.scene.updateMatrixWorld(true);
  gltf.scene.traverse((node) => {
    if (!node.isMesh || !node.geometry || !node.geometry.attributes.position) return;
    const source = node.geometry.index ? node.geometry.toNonIndexed() : node.geometry.clone();
    const geometry = new THREE.BufferGeometry();
    const position = source.getAttribute('position');
    const array = new Float32Array(position.count * 3);
    for (let i = 0; i < position.count; i++) {
      array[i * 3] = position.getX(i);
      array[i * 3 + 1] = position.getY(i);
      array[i * 3 + 2] = position.getZ(i);
    }
    geometry.setAttribute('position', new THREE.BufferAttribute(array, 3));
    geometry.applyMatrix4(node.matrixWorld);
    parts.push(geometry);
    source.dispose();
  });
  if (!parts.length) return null;
  const merged = parts.length === 1 ? parts[0] : mergeGeometries(parts, false);
  const creased = toCreasedNormals(merged, creaseRad);
  creased.computeBoundingBox();
  creased.computeBoundingSphere();
  return creased;
}

export function parseGlb(buffer) {
  return new Promise((resolve, reject) => {
    new GLTFLoader().parse(buffer, '', resolve, (error) => reject(error instanceof Error ? error : new Error(String(error && error.message ? error.message : error))));
  });
}

function srgbColor(rgb) {
  return new THREE.Color().setRGB(rgb[0], rgb[1], rgb[2], THREE.SRGBColorSpace);
}

/** The materials of the scene by kind (appearance colours of db/models/view3d.yaml). */
export function sceneMaterials(params, family) {
  const family_ = family && params.material_srgb[family] ? family : 'default';
  const standard = (rgb, extra = {}) => new THREE.MeshStandardMaterial({ color: srgbColor(rgb), metalness: 0.05, roughness: 0.62, ...extra });
  return {
    part: standard(params.material_srgb[family_], { metalness: 0.1, roughness: 0.5, flatShading: true }),
    machine: standard(params.machine_srgb),
    spindle: standard(params.machine_srgb),
    fixture: standard(params.fixture_srgb),
    tool: standard(params.tool_srgb, { metalness: 0.1, roughness: 0.45 }),
    holder: standard(params.holder_srgb),
    enclosure: new THREE.MeshStandardMaterial({ color: srgbColor(params.enclosure_srgb), transparent: true, opacity: params.enclosure_opacity, depthWrite: false, side: THREE.DoubleSide, roughness: 0.3 }),
    door: new THREE.MeshStandardMaterial({ color: srgbColor(params.enclosure_srgb), transparent: true, opacity: params.door_opacity, depthWrite: false, side: THREE.DoubleSide, roughness: 0.3 }),
  };
}

function categoryOf(kind, list) {
  if (list === 'fixture') return 'fixture';
  if (list === 'enclosure' || kind === 'enclosure' || kind === 'door' || kind === 'window') return 'enclosure';
  return 'machine';
}

export class SceneModel {
  constructor({ run, params, root, materials, chains, notes }) {
    this.run = run;
    this.params = params;
    this.root = root;
    this.materials = materials;
    this.chains = chains;
    this.notes = notes;
    this.components = [];
    this.stock = null;
    this.bounds = new THREE.Box3();
    this.visible = { machine: true, fixture: true };
  }

  _placement(item) {
    const pose = matrixFromPose(item.pose) || new THREE.Matrix4();
    let frame = item.frame ? String(item.frame) : null;
    if (frame !== null) {
      const upper = frame.toUpperCase();
      frame = ['S', 'M', 'W', 'T', 'T0'].includes(upper) ? upper : upper === 'LINK' ? 'link' : 'unknown';
    }
    const link = item.link === undefined || item.link === null ? null : String(item.link);
    const kind = item.kind ? String(item.kind) : null;
    if (frame === 'T' && TOOL_SIDE_KINDS.has(kind || '')) frame = 'T0';
    if (!frame) {
      if (link === null) frame = 'M';
      else if (this.chains.tool && this.chains.tool.has(link)) frame = 'T0';
      else if (this.chains.work && this.chains.work.has(link)) frame = 'link';
      else frame = 'M';
    }
    if (frame === 'link' && !(link !== null && ((this.chains.work && this.chains.work.has(link) && this.chains.work.complete) || (this.chains.tool && this.chains.tool.has(link) && this.chains.tool.complete)))) {
      frame = 'unknown';
    }
    return { frame, pose, link, path: Number(item.path ?? 0) || 0 };
  }

  async _load(path) {
    const buffer = await this.run.file(path);
    if (!buffer) throw new Error(`${path} is not in the run directory`);
    return bakeGltf(await parseGlb(buffer), this.params.crease_angle_rad);
  }

  async load(section) {
    const scene = this.run.scene;
    const items = [];
    for (const comp of (scene.machine && scene.machine.components) || []) items.push({ item: comp, list: 'machine' });
    for (const comp of scene.fixture || []) items.push({ item: comp, list: 'fixture' });
    for (const comp of scene.enclosure || []) items.push({ item: comp, list: 'enclosure' });
    const tasks = items.map(async ({ item, list }) => {
      if (!item.mesh) return;
      let geometry;
      try {
        geometry = await this._load(item.mesh);
      } catch (error) {
        this.notes.add(`${item.id || item.mesh}: mesh not readable (${error.message})`);
        return;
      }
      if (!geometry) return;
      const kind = item.kind || (list === 'fixture' ? 'fixture' : 'machine');
      const category = categoryOf(kind, list);
      const material = category === 'fixture' ? this.materials.fixture
        : category === 'enclosure' ? (kind === 'door' || kind === 'window' ? this.materials.door : this.materials.enclosure)
          : kind === 'spindle' ? this.materials.spindle : this.materials.machine;
      const mesh = new THREE.Mesh(geometry, material);
      mesh.name = String(item.id || item.mesh);
      mesh.matrixAutoUpdate = false;
      mesh.userData.kind = kind;
      mesh.userData.category = category;
      const placement = this._placement(item);
      if (placement.frame === 'unknown') {
        this.notes.add(placement.link !== null ? `${mesh.name}: no complete kinematic chain to place link ${placement.link}`
          : `${mesh.name}: frame ${item.frame} is not S, M, W, T, T0 or link`);
      }
      if (category === 'fixture' && section) section.addSolid(mesh);
      this.root.add(mesh);
      this.components.push({ id: mesh.name, kind, category, mesh, ...placement });
    });
    const stock = scene.stock || {};
    if (stock.mesh) {
      tasks.push((async () => {
        try {
          const geometry = await this._load(stock.mesh);
          if (!geometry) return;
          geometry.deleteAttribute('normal');
          geometry.setAttribute('aLayer', new THREE.BufferAttribute(new Float32Array(geometry.getAttribute('position').count).fill(NO_VALUE), 1));
          const mesh = new THREE.Mesh(geometry, this.materials.part);
          mesh.name = 'stock';
          mesh.renderOrder = 6;
          mesh.matrixAutoUpdate = false;
          mesh.matrix.identity();
          this.root.add(mesh);
          this.stock = mesh;
        } catch (error) {
          this.notes.add(`stock: mesh not readable (${error.message})`);
        }
      })());
    }
    await Promise.all(tasks);
    this.components.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  }

  /** Place every component for this frame. ctx: {poseSM (Matrix4|null), toolT0 [per path Matrix4],
   * linkPoses: {work: Map, tool: Map} (pose_M_link) or null, poseSW}. */
  update(ctx) {
    const missingM = [];
    for (const c of this.components) {
      const category = c.category === 'fixture' ? this.visible.fixture : this.visible.machine;
      let placed = true;
      const m = c.mesh.matrix;
      switch (c.frame) {
        case 'S': m.copy(c.pose); break;
        case 'W': m.copy(ctx.poseSW).multiply(c.pose); break;
        case 'M':
          if (ctx.poseSM) m.copy(ctx.poseSM).multiply(c.pose); else { placed = false; missingM.push(c.id); }
          break;
        case 'T0':
        case 'T': {
          const frame = (c.frame === 'T' ? ctx.toolT : ctx.toolT0)[c.path] || (c.frame === 'T' ? ctx.toolT : ctx.toolT0)[0];
          if (frame) m.copy(frame).multiply(c.pose); else placed = false;
          break;
        }
        case 'link': {
          const poses = ctx.linkPoses;
          const pose = poses && (poses.work.get(c.link) || poses.tool.get(c.link));
          if (pose && ctx.poseSM) m.copy(ctx.poseSM).multiply(pose).multiply(c.pose); else { placed = false; if (!ctx.poseSM) missingM.push(c.id); }
          break;
        }
        default: placed = false;
      }
      c.mesh.visible = placed && category;
      c.mesh.matrixWorldNeedsUpdate = true;
    }
    return missingM;
  }

  setVisible({ machine, fixture }) {
    this.visible.machine = machine !== false;
    this.visible.fixture = fixture !== false;
  }

  /** Bounding box of what is drawn in S (for sections and framing). */
  partBounds() {
    const box = new THREE.Box3();
    const size = this.run.scene && this.run.scene.stock && this.run.scene.stock.size_m;
    if (Array.isArray(size) && size.length === 3) box.expandByPoint(new THREE.Vector3(0, 0, 0)).expandByPoint(new THREE.Vector3(...size.map(Number)));
    if (this.stock) {
      this.stock.geometry.computeBoundingBox();
      box.union(this.stock.geometry.boundingBox.clone().applyMatrix4(this.stock.matrix));
    }
    return box;
  }

  /** Meshes for picking (visible machine and fixture parts). */
  pickables() {
    return this.components.filter((c) => c.mesh.visible && c.category !== 'enclosure').map((c) => c.mesh);
  }

  dispose() {
    for (const c of this.components) { c.mesh.removeFromParent(); c.mesh.geometry.dispose(); }
    if (this.stock) { this.stock.removeFromParent(); this.stock.geometry.dispose(); }
  }
}
