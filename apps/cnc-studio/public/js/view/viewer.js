// three.js scene: stock / part, toolpath, tool, fixtures.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { meshChunk } from './surfacenets.js';
import { IDENTITY, applyX, invertX } from '../sim/voxelsim.js';
import { rotaryX } from '../sim/simrunner.js';

const STOCK_HEX = '#b9c2cc';

function toolColorTable(tools) {
  const table = new Float32Array(256 * 3);
  const base = new THREE.Color(STOCK_HEX);
  for (let i = 0; i < 256; i++) {
    let c = base;
    if (i > 0) {
      const t = tools[i];
      c = base.clone().lerp(new THREE.Color(t ? t.color : '#4cc9f0'), 0.62);
    }
    table[i * 3] = c.r; table[i * 3 + 1] = c.g; table[i * 3 + 2] = c.b;
  }
  return table;
}

// Ramer–Douglas–Peucker on [r, z, tool] points (keeps tool boundaries)
function simplify(pts, tol) {
  if (pts.length < 3) return pts;
  const keep = new Uint8Array(pts.length);
  keep[0] = keep[pts.length - 1] = 1;
  for (let i = 1; i < pts.length; i++) if (pts[i][2] !== pts[i - 1][2]) { keep[i] = 1; keep[i - 1] = 1; }
  const stack = [[0, pts.length - 1]];
  while (stack.length) {
    const [a, b] = stack.pop();
    let maxD = 0, idx = -1;
    const [ra, za] = pts[a], [rb, zb] = pts[b];
    const dz = zb - za, dr = rb - ra;
    const L = Math.hypot(dz, dr) || 1e-12;
    for (let i = a + 1; i < b; i++) {
      const d = Math.abs((pts[i][1] - za) * dr - (pts[i][0] - ra) * dz) / L;
      if (d > maxD) { maxD = d; idx = i; }
    }
    if (maxD > tol && idx > 0) { keep[idx] = 1; stack.push([a, idx], [idx, b]); }
  }
  return pts.filter((_, i) => keep[i]);
}

// {R (row-major 3x3), t} -> THREE.Matrix4
function toMatrix4(X) {
  const R = X.R, t = X.t;
  return new THREE.Matrix4().set(R[0], R[1], R[2], t[0], R[3], R[4], R[5], t[1], R[6], R[7], R[8], t[2], 0, 0, 0, 1);
}

const PATH_VERT = `
attribute float aTime; attribute float aDist; attribute vec3 color;
varying vec3 vColor; varying float vT; varying float vD;
void main() { vColor = color; vT = aTime; vD = aDist; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
const PATH_FRAG = `
uniform float uTime; uniform float uFuture; uniform float uDash; uniform float uOpacity;
varying vec3 vColor; varying float vT; varying float vD;
void main() {
  if (uDash > 0.0 && mod(vD, uDash) > uDash * 0.55) discard;
  float done = vT <= uTime ? 1.0 : uFuture;
  gl_FragColor = vec4(vColor * (vT <= uTime ? 1.0 : 0.8), done * uOpacity);
}`;

export class Viewer {
  constructor(el) {
    this.el = el;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.localClippingEnabled = true;
    el.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(32, 1, 0.5, 20000);
    this.groups = {};
    // the part and its toolpath sit on the rotary table (rotGroup = current A/B position);
    // tool, fixtures and helpers stay in the work frame
    this.rotGroup = new THREE.Group();
    this.rotGroup.matrixAutoUpdate = false;
    this.scene.add(this.rotGroup);
    for (const g of ['stock', 'path', 'tool', 'fixture', 'helpers', 'target']) {
      this.groups[g] = new THREE.Group();
      (g === 'stock' || g === 'path' || g === 'target' ? this.rotGroup : this.scene).add(this.groups[g]);
    }
    this.groups.stock.matrixAutoUpdate = false;
    this.groups.target.matrixAutoUpdate = false;
    this.pending = new Set();
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x3a4552, 1.25));
    this.key = new THREE.DirectionalLight(0xffffff, 1.9);
    this.fill = new THREE.DirectionalLight(0xbcd4ff, 0.7);
    this.scene.add(this.key, this.fill);
    this.uniforms = { uTime: { value: 0 }, uBottom: { value: -1e9 } };
    this.visible = { path: true, stock: true, tool: true, rapids: true, target: true };
    this.section = false;
    this.machine = 'mill';
    this.needsRender = true;
    this.makeControls();
    new ResizeObserver(() => this.resize()).observe(el);
    this.resize();
    const loop = () => {
      requestAnimationFrame(loop);
      const moved = this.controls.update();
      if (moved || this.needsRender) { this.needsRender = false; this.renderer.render(this.scene, this.camera); }
    };
    loop();
  }

  makeControls() {
    if (this.controls) this.controls.dispose();
    this.camera.up.set(...(this.machine === 'mill' ? [0, 0, 1] : [0, 1, 0]));
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.screenSpacePanning = true;
    this.controls.addEventListener('change', () => { this.needsRender = true; });
  }

  resize() {
    const w = this.el.clientWidth || 1, h = this.el.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.needsRender = true;
  }

  clearGroup(name) {
    const g = this.groups[name];
    for (const c of [...g.children]) {
      g.remove(c);
      c.traverse?.((o) => { o.geometry?.dispose(); if (o.material) [].concat(o.material).forEach((m) => m.dispose()); });
    }
  }

  // ---------------------------------------------------------------- load
  // extra: { engine: 'height' | 'voxel' | 'lathe', X: part -> work transform (voxel),
  //          fixtures: [{name, min, max}] work-frame boxes, rotReverse }
  load(result, sim, extra = {}) {
    const machineChanged = result.machine !== this.machine;
    this.machine = result.machine;
    if (machineChanged) this.makeControls();
    this.result = result;
    this.sim = sim;
    this.engine = extra.engine || (result.machine === 'lathe' ? 'lathe' : 'height');
    this.partX = extra.X || IDENTITY;
    this.fixtures = extra.fixtures || [];
    this.rotReverse = !!extra.rotReverse;
    this.workFromPart = extra.workFromPart || IDENTITY;
    this.colorTable = toolColorTable(result.tools);
    for (const g of ['stock', 'path', 'tool', 'fixture', 'helpers']) this.clearGroup(g);
    this.pending.clear();
    this.groups.stock.matrix.copy(this.engine === 'voxel' ? toMatrix4(this.partX) : new THREE.Matrix4());
    this.groups.stock.matrixWorldNeedsUpdate = true;
    this.rotKey = null;
    this.setRotary(0, 0);
    if (this.engine === 'lathe') this.buildLathe(); else if (this.engine === 'voxel') this.buildVoxel(); else this.buildMill();
    this.buildFixtures();
    this.buildPath();
    this.placeTarget();
    this.toolKey = null;
    this.updateLights();
    if (machineChanged || !this.fitted || this.stockKey !== JSON.stringify(result.stock)) { this.setView(this.lastView || 'iso'); this.fitted = true; }
    this.stockKey = JSON.stringify(result.stock);
    this.applyVisibility();
    this.needsRender = true;
  }

  updateLights() {
    const b = this.bounds();
    const c = b.getCenter(new THREE.Vector3());
    const s = b.getSize(new THREE.Vector3()).length() || 100;
    if (this.machine === 'mill') {
      this.key.position.set(c.x + s * 0.6, c.y - s * 0.9, c.z + s * 1.4);
      this.fill.position.set(c.x - s, c.y + s * 0.6, c.z + s * 0.5);
    } else {
      this.key.position.set(c.x + s * 0.4, c.y + s * 1.2, c.z + s * 1.0);
      this.fill.position.set(c.x - s, c.y - s * 0.4, c.z + s * 0.6);
    }
    this.key.target.position.copy(c); this.fill.target.position.copy(c);
    this.scene.add(this.key.target, this.fill.target);
  }

  // ---------------------------------------------------------------- mill stock
  buildMill() {
    const sim = this.sim;
    const { nx, ny, cx, cy, minX, minY } = sim;
    const N = nx * ny;
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(N * 3), nor = new Float32Array(N * 3), col = new Float32Array(N * 3);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const k = j * nx + i;
      pos[k * 3] = minX + i * cx; pos[k * 3 + 1] = minY + j * cy;
    }
    const idx = new Uint32Array((nx - 1) * (ny - 1) * 6);
    let q = 0;
    for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
      const a = j * nx + i, b = a + 1, c = a + nx, d = c + 1;
      idx[q++] = a; idx[q++] = b; idx[q++] = d;
      idx[q++] = a; idx[q++] = d; idx[q++] = c;
    }
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3).setUsage(THREE.DynamicDrawUsage));
    geo.setAttribute('normal', new THREE.BufferAttribute(nor, 3).setUsage(THREE.DynamicDrawUsage));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3).setUsage(THREE.DynamicDrawUsage));
    this.uniforms.uBottom.value = sim.bottom;
    const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.42, metalness: 0.38, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 });
    mat.onBeforeCompile = (shader) => {
      shader.uniforms.uBottom = this.uniforms.uBottom;
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', '#include <common>\nvarying float vHz;')
        .replace('#include <begin_vertex>', '#include <begin_vertex>\nvHz = position.z;');
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', '#include <common>\nvarying float vHz;\nuniform float uBottom;')
        .replace('void main() {', 'void main() {\n  if (vHz < uBottom) discard;');
    };
    this.millMesh = new THREE.Mesh(geo, mat);
    this.groups.stock.add(this.millMesh);
    // walls
    const wallMat = new THREE.MeshStandardMaterial({ color: new THREE.Color(STOCK_HEX).multiplyScalar(0.86), roughness: 0.55, metalness: 0.3 });
    this.walls = [];
    const sides = [
      { nodes: [...Array(nx).keys()].map((i) => i), n: [0, -1, 0] },
      { nodes: [...Array(ny).keys()].map((j) => j * nx + nx - 1), n: [1, 0, 0] },
      { nodes: [...Array(nx).keys()].map((i) => (ny - 1) * nx + nx - 1 - i), n: [0, 1, 0] },
      { nodes: [...Array(ny).keys()].map((j) => (ny - 1 - j) * nx), n: [-1, 0, 0] },
    ];
    for (const s of sides) {
      const m = s.nodes.length;
      const g = new THREE.BufferGeometry();
      const p = new Float32Array(m * 2 * 3), nn = new Float32Array(m * 2 * 3);
      for (let i = 0; i < m * 2; i++) nn.set(s.n, i * 3);
      const id = [];
      for (let i = 0; i < m - 1; i++) {
        const t0 = i * 2, b0 = t0 + 1, t1 = t0 + 2, b1 = t0 + 3;
        id.push(t0, b0, b1, t0, b1, t1);
      }
      g.setIndex(id);
      g.setAttribute('position', new THREE.BufferAttribute(p, 3).setUsage(THREE.DynamicDrawUsage));
      g.setAttribute('normal', new THREE.BufferAttribute(nn, 3));
      const mesh = new THREE.Mesh(g, wallMat);
      this.walls.push({ mesh, nodes: s.nodes });
      this.groups.stock.add(mesh);
    }
    const sx = (nx - 1) * cx, sy = (ny - 1) * cy;
    const bottom = new THREE.Mesh(new THREE.PlaneGeometry(sx, sy), wallMat);
    bottom.rotation.x = Math.PI;
    bottom.position.set(minX + sx / 2, minY + sy / 2, sim.bottom);
    this.groups.stock.add(bottom);
    // grid + origin
    const size = Math.max(sx, sy) * 2.2;
    const step = size > 400 ? 50 : size > 150 ? 10 : 5;
    const grid = new THREE.GridHelper(Math.ceil(size / step) * step, Math.ceil(size / step), 0x5b6b7c, 0x2f3a46);
    grid.rotation.x = Math.PI / 2;
    grid.position.set(minX + sx / 2, minY + sy / 2, sim.bottom - 0.02);
    grid.material.transparent = true; grid.material.opacity = 0.55;
    this.groups.helpers.add(grid);
    this.addOrigin(Math.max(8, Math.min(30, Math.max(sx, sy) * 0.12)));
    this.updateMillStock(true);
  }

  addOrigin(len) {
    const axes = new THREE.Group();
    const mk = (dir, color) => {
      const arrow = new THREE.ArrowHelper(new THREE.Vector3(...dir), new THREE.Vector3(0, 0, 0), len, color, len * 0.22, len * 0.12);
      arrow.line.material.depthTest = false; arrow.cone.material.depthTest = false;
      arrow.renderOrder = 10;
      axes.add(arrow);
    };
    if (this.machine === 'mill') { mk([1, 0, 0], 0xff5d5d); mk([0, 1, 0], 0x5dff8a); mk([0, 0, 1], 0x5da8ff); }
    else { mk([1, 0, 0], 0x5da8ff); mk([0, 1, 0], 0xff5d5d); }
    this.groups.helpers.add(axes);
  }

  updateMillStock(full = false) {
    const sim = this.sim;
    let d = sim.takeDirty();
    if (full) d = { i0: 0, i1: sim.nx - 1, j0: 0, j1: sim.ny - 1 };
    if (!d) return false;
    const { nx, ny, cx, cy, heights, toolIdx } = sim;
    const geo = this.millMesh.geometry;
    const pos = geo.attributes.position.array, nor = geo.attributes.normal.array, col = geo.attributes.color.array;
    const i0 = Math.max(0, d.i0 - 1), i1 = Math.min(nx - 1, d.i1 + 1);
    const j0 = Math.max(0, d.j0 - 1), j1 = Math.min(ny - 1, d.j1 + 1);
    const ct = this.colorTable;
    const shade = sim.top - sim.bottom || 1;
    for (let j = j0; j <= j1; j++) {
      for (let i = i0; i <= i1; i++) {
        const k = j * nx + i;
        const h = heights[k];
        pos[k * 3 + 2] = h;
        const hl = heights[i > 0 ? k - 1 : k], hr = heights[i < nx - 1 ? k + 1 : k];
        const hd = heights[j > 0 ? k - nx : k], hu = heights[j < ny - 1 ? k + nx : k];
        let gx = -(Math.max(hr, sim.bottom) - Math.max(hl, sim.bottom)) / ((i > 0 && i < nx - 1 ? 2 : 1) * cx);
        let gy = -(Math.max(hu, sim.bottom) - Math.max(hd, sim.bottom)) / ((j > 0 && j < ny - 1 ? 2 : 1) * cy);
        const inv = 1 / Math.hypot(gx, gy, 1);
        nor[k * 3] = gx * inv; nor[k * 3 + 1] = gy * inv; nor[k * 3 + 2] = inv;
        const t = toolIdx[k];
        // deeper cuts slightly darker for depth perception
        const f = t ? 1 - 0.28 * Math.min(1, (sim.top - Math.max(h, sim.bottom)) / shade) : 1;
        col[k * 3] = ct[t * 3] * f; col[k * 3 + 1] = ct[t * 3 + 1] * f; col[k * 3 + 2] = ct[t * 3 + 2] * f;
      }
    }
    const start = j0 * nx * 3, count = (j1 - j0 + 1) * nx * 3;
    for (const a of [geo.attributes.position, geo.attributes.normal, geo.attributes.color]) {
      a.clearUpdateRanges?.();
      a.addUpdateRange(start, count);
      a.needsUpdate = true;
    }
    if (full || i0 === 0 || j0 === 0 || i1 === nx - 1 || j1 === ny - 1) {
      for (const w of this.walls) {
        const p = w.mesh.geometry.attributes.position;
        const arr = p.array;
        w.nodes.forEach((k, idx) => {
          const x = sim.minX + (k % nx) * cx, y = sim.minY + Math.floor(k / nx) * cy;
          arr.set([x, y, Math.max(sim.bottom, heights[k]), x, y, sim.bottom], idx * 6);
        });
        p.needsUpdate = true;
      }
    }
    geo.computeBoundingSphere();
    this.needsRender = true;
    return true;
  }

  // ---------------------------------------------------------------- lathe stock
  buildLathe() {
    const s = this.result.stock;
    const r0 = s.diameter / 2;
    this.latheMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.38, metalness: 0.45, side: THREE.FrontSide });
    this.capMat = new THREE.MeshStandardMaterial({ color: '#d9c9a3', roughness: 0.8, metalness: 0.05, side: THREE.DoubleSide });
    this.latheMesh = new THREE.Mesh(new THREE.BufferGeometry(), this.latheMat);
    this.groups.stock.add(this.latheMesh);
    this.capGroup = new THREE.Group();
    this.groups.stock.add(this.capGroup);
    // chuck body behind the jaws; the jaws hold the bar from zChuck - grip to zChuck
    const sim = this.sim;
    const zChuck = sim.zChuck ?? -s.length, grip = sim.grip ?? 15;
    const zMin = zChuck - grip;
    let rGrip = 0;
    for (let i = 0; i < sim.n && sim.zMin + i * sim.dz <= zChuck; i++) rGrip = Math.max(rGrip, sim.outer[i]);
    if (!rGrip) rGrip = r0;
    const chuckR = Math.max(rGrip * 1.9, rGrip + 35);
    const dark = new THREE.MeshStandardMaterial({ color: '#3a4350', roughness: 0.6, metalness: 0.6 });
    const jawMat = new THREE.MeshStandardMaterial({ color: '#59636f', roughness: 0.5, metalness: 0.7 });
    const body = new THREE.Mesh(new THREE.CylinderGeometry(chuckR, chuckR, 45, 64), dark);
    body.rotation.z = Math.PI / 2;
    body.position.set(zMin - 22.5, 0, 0);
    this.groups.fixture.add(body);
    const jl = Math.max(4, grip);
    for (let k = 0; k < 3; k++) {
      const a = Math.PI / 2 + (k * 2 * Math.PI) / 3;
      const jh = Math.max(12, rGrip * 0.5);
      const jaw = new THREE.Mesh(new THREE.BoxGeometry(jl, jh, Math.max(14, rGrip * 0.45)), jawMat);
      jaw.position.set(zChuck - jl / 2, Math.cos(a) * (rGrip + jh / 2), Math.sin(a) * (rGrip + jh / 2));
      jaw.rotation.x = -(a - Math.PI / 2);
      this.groups.fixture.add(jaw);
    }
    // centre line
    const cl = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(zMin - 50, 0, 0), new THREE.Vector3((s.faceStock || 0) + 40, 0, 0)]),
      new THREE.LineDashedMaterial({ color: 0x6b7a8a, dashSize: 4, gapSize: 3 }),
    );
    cl.computeLineDistances();
    this.groups.helpers.add(cl);
    this.addOrigin(Math.max(8, Math.min(25, r0 * 0.6)));
    this.updateLatheStock(true);
  }

  setSection(on) {
    this.section = on;
    if (this.machine === 'lathe' && this.sim) { this.updateLatheStock(true); this.needsRender = true; }
  }

  updateLatheStock(force = false) {
    const sim = this.sim;
    if (!sim.takeDirty() && !force) return false;
    const now = performance.now();
    if (!force && this.lastLatheBuild && now - this.lastLatheBuild < 70) { sim.dirty = true; return false; }
    this.lastLatheBuild = now;
    const { outer, inner } = sim.outline(4000);
    const o = simplify(outer, 0.004), inn = simplify(inner, 0.004);
    const loop = o.concat(inn.length ? inn : [[0, sim.zMin, 0], [0, sim.zMax, 0]]);
    const segs = this.section ? 54 : 72;
    const a0 = this.section ? Math.PI / 2 : 0, a1 = 2 * Math.PI;
    const ct = this.colorTable;
    const positions = [], normals = [], colors = [], index = [];
    const cos = [], sin = [];
    for (let s = 0; s <= segs; s++) { const a = a0 + ((a1 - a0) * s) / segs; cos.push(Math.cos(a)); sin.push(Math.sin(a)); }
    let v = 0;
    for (let i = 0; i < loop.length; i++) {
      const p = loop[i], q = loop[(i + 1) % loop.length];
      const tz = q[1] - p[1], tr = q[0] - p[0];
      const L = Math.hypot(tz, tr);
      if (L < 1e-6) continue;
      if (p[0] < 1e-6 && q[0] < 1e-6) continue; // on the axis
      const nz = tr / L, nr = -tz / L;
      const t = p[2] || q[2] || 0;
      const cr = ct[t * 3], cg = ct[t * 3 + 1], cb = ct[t * 3 + 2];
      for (let s = 0; s <= segs; s++) {
        positions.push(p[1], p[0] * cos[s], p[0] * sin[s], q[1], q[0] * cos[s], q[0] * sin[s]);
        normals.push(nz, nr * cos[s], nr * sin[s], nz, nr * cos[s], nr * sin[s]);
        colors.push(cr, cg, cb, cr, cg, cb);
      }
      for (let s = 0; s < segs; s++) {
        const a = v + s * 2, b = a + 1, c = a + 2, d = a + 3;
        index.push(a, b, d, a, d, c);
      }
      v += (segs + 1) * 2;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    g.setIndex(v > 65535 ? new THREE.Uint32BufferAttribute(index, 1) : new THREE.Uint16BufferAttribute(index, 1));
    this.latheMesh.geometry.dispose();
    this.latheMesh.geometry = g;
    // section caps
    for (const c of [...this.capGroup.children]) { this.capGroup.remove(c); c.geometry.dispose(); }
    if (this.section && o.length > 2) {
      const shape = new THREE.Shape(loop.map((p) => new THREE.Vector2(p[1], p[0])));
      const capGeo = new THREE.ShapeGeometry(shape);
      const c1 = new THREE.Mesh(capGeo, this.capMat);
      const c2 = new THREE.Mesh(capGeo.clone(), this.capMat);
      c2.rotation.x = Math.PI / 2;
      this.capGroup.add(c1, c2);
    }
    this.needsRender = true;
    return true;
  }

  // ---------------------------------------------------------------- 3D (voxel) part
  buildVoxel() {
    const sim = this.sim;
    this.voxMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.42, metalness: 0.38 });
    this.chunkMeshes = new Array(sim.cnx * sim.cny * sim.cnz).fill(null);
    const s = this.result.stock;
    const sx = s.max[0] - s.min[0], sy = s.max[1] - s.min[1];
    const size = Math.max(sx, sy) * 2.2;
    const step = size > 400 ? 50 : size > 150 ? 10 : 5;
    const grid = new THREE.GridHelper(Math.ceil(size / step) * step, Math.ceil(size / step), 0x5b6b7c, 0x2f3a46);
    grid.rotation.x = Math.PI / 2;
    const floor = this.result.usesRotary ? s.min[2] - Math.max(sx, sy, s.max[2] - s.min[2]) * 0.3 : s.min[2] - 0.02;
    grid.position.set((s.min[0] + s.max[0]) / 2, (s.min[1] + s.max[1]) / 2, floor);
    grid.material.transparent = true; grid.material.opacity = 0.55;
    this.groups.helpers.add(grid);
    this.addOrigin(Math.max(8, Math.min(30, Math.max(sx, sy) * 0.12)));
    this.updateVoxelStock(true);
  }

  // re-mesh dirty chunks, ~24 ms per frame so rewinds stay responsive (force = everything now)
  updateVoxelStock(force = false) {
    const sim = this.sim;
    const list = sim.takeDirty();
    if (force) for (let c = 0; c < this.chunkMeshes.length; c++) this.pending.add(c);
    else if (list) for (const c of list) this.pending.add(c);
    if (!this.pending.size) return false;
    const t0 = performance.now();
    for (const c of this.pending) {
      this.pending.delete(c);
      this.remeshChunk(c);
      if (!force && performance.now() - t0 > 24) break;
    }
    if (this.pending.size && !this.pendingRaf) {
      this.pendingRaf = requestAnimationFrame(() => { this.pendingRaf = 0; if (this.sim === sim) this.updateVoxelStock(); });
    }
    this.needsRender = true;
    return true;
  }

  remeshChunk(c) {
    const m = meshChunk(this.sim, c, this.colorTable);
    let mesh = this.chunkMeshes[c];
    if (!m) {
      if (mesh) { this.groups.stock.remove(mesh); mesh.geometry.dispose(); this.chunkMeshes[c] = null; }
      return;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(m.positions, 3));
    g.setAttribute('normal', new THREE.BufferAttribute(m.normals, 3));
    g.setAttribute('color', new THREE.BufferAttribute(m.colors, 3));
    g.setIndex(new THREE.BufferAttribute(m.index, 1));
    g.computeBoundingSphere();
    if (mesh) { mesh.geometry.dispose(); mesh.geometry = g; }
    else { mesh = new THREE.Mesh(g, this.voxMat); this.chunkMeshes[c] = mesh; this.groups.stock.add(mesh); }
  }

  // finish any queued re-meshing (before a capture)
  flush() {
    if (this.engine === 'voxel' && this.sim) this.updateVoxelStock(true);
  }

  // ---------------------------------------------------------------- fixtures (vise jaws)
  buildFixtures() {
    if (!this.fixtures.length) return;
    const mat = new THREE.MeshStandardMaterial({ color: '#4a5562', roughness: 0.55, metalness: 0.6, transparent: true, opacity: 0.88 });
    for (const f of this.fixtures) {
      const size = [0, 1, 2].map((k) => f.max[k] - f.min[k]);
      const box = new THREE.Mesh(new THREE.BoxGeometry(...size), mat);
      box.position.set(...[0, 1, 2].map((k) => (f.min[k] + f.max[k]) / 2));
      this.groups.fixture.add(box);
    }
  }

  // ---------------------------------------------------------------- target model (ghost)
  // tris: part-frame triangles (Float32Array, 9 per triangle) or null
  setTarget(tris) {
    for (const c of [...this.groups.target.children]) { this.groups.target.remove(c); c.geometry.dispose(); c.material.dispose(); }
    if (tris) {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(tris, 3));
      g.computeVertexNormals();
      const mesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: '#58d6a4', emissive: '#103a2a', roughness: 0.6, metalness: 0.1, transparent: true, opacity: 0.16, depthWrite: false, side: THREE.DoubleSide }));
      mesh.renderOrder = 4;
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(g, 40), new THREE.LineBasicMaterial({ color: '#58d6a4', transparent: true, opacity: 0.4, depthWrite: false }));
      edges.renderOrder = 4;
      this.groups.target.add(mesh, edges);
    }
    this.placeTarget();
  }

  placeTarget() {
    // part frame -> this operation's work frame (the lathe scene shows work Z along scene X)
    let m = toMatrix4(this.workFromPart || IDENTITY);
    if (this.machine === 'lathe') m = new THREE.Matrix4().set(0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1).multiply(m);
    this.groups.target.matrix.copy(m);
    this.groups.target.matrixWorldNeedsUpdate = true;
    this.groups.target.visible = this.visible.target && this.groups.target.children.length > 0;
    this.needsRender = true;
  }

  // 4th/5th axis position (degrees): turns the part and its toolpath
  setRotary(a, b) {
    const key = `${a}|${b}`;
    if (key === this.rotKey) return;
    this.rotKey = key;
    this.rotGroup.matrix.copy(a || b ? toMatrix4(rotaryX(a, b, this.rotReverse)) : new THREE.Matrix4());
    this.rotGroup.matrixWorldNeedsUpdate = true;
    this.needsRender = true;
  }

  updateStock(force = false) {
    if (!this.sim) return;
    if (this.engine === 'lathe') this.updateLatheStock(force);
    else if (this.engine === 'voxel') this.updateVoxelStock(force);
    else this.updateMillStock(force);
  }

  // ---------------------------------------------------------------- toolpath
  toScene(p) { return this.machine === 'lathe' ? [p[2], p[0], 0] : p; }

  buildPath() {
    const res = this.result;
    const feed = { pos: [], col: [], time: [], dist: [] };
    const rapid = { pos: [], col: [], time: [], dist: [] };
    const rc = new THREE.Color('#ff6b6b');
    let dist = 0;
    // with rotary axes the path is drawn on the table (where it lands on the part), so it turns with it
    const unrot = (p, m, f) => {
      const a = m.rot[0] + (m.rot[1] - m.rot[0]) * f, b = m.rot[2] + (m.rot[3] - m.rot[2]) * f;
      return a || b ? applyX(invertX(rotaryX(a, b, this.rotReverse)), p) : p;
    };
    for (const m0 of res.moves) {
      if (m0.kind !== 'feed' && m0.kind !== 'rapid') continue;
      if (m0.rotOnly) continue;
      let m = m0;
      if (m0.rot) {
        // turning while moving: subdivide (2 degree steps) so the path wraps around the part
        const turn = Math.max(Math.abs(m0.rot[1] - m0.rot[0]), Math.abs(m0.rot[3] - m0.rot[2]));
        const n = Math.max(1, Math.ceil(turn / 2));
        const total = m0.len || 1;
        const pts = [unrot(m0.pts[0], m0, 0)];
        let s = 0;
        for (let i = 1; i < m0.pts.length; i++) {
          const p = m0.pts[i - 1], q = m0.pts[i];
          const l = Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]);
          const k = turn > 1e-9 ? Math.max(1, Math.ceil((n * l) / total)) : 1;
          for (let j = 1; j <= k; j++) {
            const u = j / k;
            pts.push(unrot([p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u, p[2] + (q[2] - p[2]) * u], m0, (s + l * u) / total));
          }
          s += l;
        }
        m = { ...m0, pts };
      }
      const tgt = m.kind === 'rapid' ? rapid : feed;
      const c = m.kind === 'rapid' ? rc : new THREE.Color(res.tools[m.tool]?.color || '#4cc9f0');
      let acc = 0;
      for (let i = 1; i < m.pts.length; i++) {
        const a = this.toScene(m.pts[i - 1]), b = this.toScene(m.pts[i]);
        const l = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
        const ta = m.t0 + (m.t1 - m.t0) * (m.len ? acc / m.len : 0);
        acc += l;
        const tb = m.t0 + (m.t1 - m.t0) * (m.len ? acc / m.len : 1);
        tgt.pos.push(...a, ...b);
        tgt.col.push(c.r, c.g, c.b, c.r, c.g, c.b);
        tgt.time.push(ta, tb);
        tgt.dist.push(dist, dist + l);
        dist += l;
      }
    }
    const mk = (d, dash, opacity) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(d.pos, 3));
      g.setAttribute('color', new THREE.Float32BufferAttribute(d.col, 3));
      g.setAttribute('aTime', new THREE.Float32BufferAttribute(d.time, 1));
      g.setAttribute('aDist', new THREE.Float32BufferAttribute(d.dist, 1));
      const mat = new THREE.ShaderMaterial({
        vertexShader: PATH_VERT, fragmentShader: PATH_FRAG, transparent: true, depthWrite: false,
        depthTest: this.machine !== 'lathe',
        uniforms: { uTime: this.uniforms.uTime, uFuture: { value: 0.32 }, uDash: { value: dash }, uOpacity: { value: opacity } },
      });
      const l = new THREE.LineSegments(g, mat);
      l.renderOrder = 5;
      return l;
    };
    this.feedLines = mk(feed, 0, 1);
    this.rapidLines = mk(rapid, this.pathScale() * 0.025, 0.75);
    this.groups.path.add(this.feedLines, this.rapidLines);
  }

  pathScale() {
    return this.bounds().getSize(new THREE.Vector3()).length() || 100;
  }

  // ---------------------------------------------------------------- tool
  buildTool(tool) {
    const g = new THREE.Group();
    const col = new THREE.Color(tool?.color || '#4cc9f0');
    const flute = new THREE.MeshStandardMaterial({ color: col, roughness: 0.3, metalness: 0.7, emissive: col.clone().multiplyScalar(0.12) });
    const steel = new THREE.MeshStandardMaterial({ color: '#9aa4ae', roughness: 0.35, metalness: 0.85 });
    const dark = new THREE.MeshStandardMaterial({ color: '#2b323b', roughness: 0.55, metalness: 0.6 });
    const zCyl = (rTop, rBot, h, z0, mat, seg = 40) => {
      const m = new THREE.Mesh(new THREE.CylinderGeometry(rTop, rBot, h, seg), mat);
      m.rotation.x = Math.PI / 2;
      m.position.z = z0 + h / 2;
      g.add(m);
      return m;
    };
    if (this.machine === 'mill') {
      const t = tool || { type: 'flat', diameter: 10, fluteLength: 30 };
      const r = t.diameter / 2;
      const L = t.fluteLength || 30;
      let z = 0;
      switch (t.type) {
        case 'ball': {
          const s = new THREE.Mesh(new THREE.SphereGeometry(r, 32, 16), flute);
          s.position.z = r; g.add(s);
          zCyl(r, r, L - r, r, flute); z = L;
          break;
        }
        case 'drill': case 'spot': {
          const h = r / Math.tan(((t.tipAngle || 118) * Math.PI) / 360);
          const cone = new THREE.Mesh(new THREE.ConeGeometry(r, h, 40), flute);
          cone.rotation.x = -Math.PI / 2; cone.position.z = h / 2; g.add(cone);
          zCyl(r, r, Math.max(0, L - h), h, flute); z = Math.max(L, h);
          break;
        }
        case 'chamfer': case 'engrave': {
          const tr = (t.tipDiameter || 0) / 2;
          const h = (r - tr) / Math.tan(((t.tipAngle || 90) * Math.PI) / 360);
          zCyl(r, tr, h, 0, flute); z = h;
          zCyl(Math.min(r, 6), Math.min(r, 6), 8, z, steel); z += 8;
          break;
        }
        case 'face': {
          zCyl(r, r, Math.min(L, 8), 0, flute); z = Math.min(L, 8);
          zCyl(r * 0.6, r * 0.8, 12, z, steel); z += 12;
          break;
        }
        default:
          zCyl(r, r, L, 0, flute); z = L;
      }
      // shank up to the stick-out, then the holder (sizes drive the collision checks too)
      const shankR = (t.shankDiameter || t.diameter) / 2;
      const stick = Math.max(t.overallLength || z + 15, z + 2);
      zCyl(shankR, shankR, stick - z, z, steel); z = stick;
      const hr = (t.holderDiameter || 32) / 2;
      zCyl(hr, Math.max(shankR + 2, hr * 0.55), 22, z, dark); z += 22;
      zCyl(hr, hr, 18, z, dark, 48); z += 18;
      zCyl(hr * 1.25, hr, 10, z, dark, 48);
      return g;
    }
    // lathe tools (scene: x = Z axis, y = radial)
    const t = tool || { type: 'turn', noseRadius: 0.4, insertAngle: 80 };
    const extrude = (pts, depth = 4) => {
      const shape = new THREE.Shape(pts.map((p) => new THREE.Vector2(p[0], p[1])));
      const geo = new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: false });
      geo.translate(0, 0, -depth / 2);
      return geo;
    };
    const holder = (x, y, w, h) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, 14), dark);
      m.position.set(x, y + h / 2, 0);
      g.add(m);
    };
    // real outlines (imported CncSimulator registers and generated built-ins)
    if (t.cutter && t.cutter.length >= 3) {
      if (t.type === 'drill') {
        const r = (t.diameter || 10) / 2, L = t.length || 80, dir = t.hand === 'right' ? -1 : 1;
        const h = r / Math.tan(((t.tipAngle || 118) * Math.PI) / 360);
        const cone = new THREE.Mesh(new THREE.ConeGeometry(r, h, 32), flute);
        cone.rotation.z = dir * Math.PI / 2; cone.position.x = dir * h / 2; g.add(cone);
        const body = new THREE.Mesh(new THREE.CylinderGeometry(r, r, Math.max(1, L - h), 32), flute);
        body.rotation.z = Math.PI / 2; body.position.x = dir * (h + (L - h) / 2); g.add(body);
      } else {
        g.add(new THREE.Mesh(extrude(t.cutter, t.type === 'groove' ? Math.min(4, (t.width || 3) + 1) : 4.8), flute));
      }
      if (t.holder && t.holder.length >= 3) g.add(new THREE.Mesh(extrude(t.holder, t.type === 'drill' ? 30 : 18), dark));
      return g;
    }
    switch (t.type) {
      case 'groove': {
        const w = t.width || 3;
        g.add(new THREE.Mesh(extrude([[0, 0], [w, 0], [w, 16], [0, 16]], 3), flute));
        holder(w / 2, 16, 18, 45);
        break;
      }
      case 'thread': {
        const h = 6, half = h * Math.tan(((t.tipAngle || 60) * Math.PI) / 360);
        g.add(new THREE.Mesh(extrude([[0, 0], [half, h], [-half, h]], 3), flute));
        holder(0, h, 16, 45);
        break;
      }
      case 'drill': {
        const r = (t.diameter || 10) / 2;
        const h = r / Math.tan(((t.tipAngle || 118) * Math.PI) / 360);
        const cone = new THREE.Mesh(new THREE.ConeGeometry(r, h, 32), flute);
        cone.rotation.z = Math.PI / 2; cone.position.x = h / 2; g.add(cone);
        const body = new THREE.Mesh(new THREE.CylinderGeometry(r, r, 70, 32), flute);
        body.rotation.z = Math.PI / 2; body.position.x = h + 35; g.add(body);
        const sh = new THREE.Mesh(new THREE.CylinderGeometry(r + 6, r + 6, 30, 32), dark);
        sh.rotation.z = Math.PI / 2; sh.position.x = h + 85; g.add(sh);
        break;
      }
      case 'bore': {
        const a = ((t.insertAngle || 55) * Math.PI) / 180, s = 9;
        const dir = Math.PI / 4; // body toward +Z and -X
        const v1 = [s * Math.cos(dir + a / 2), -s * Math.sin(dir + a / 2)];
        const v2 = [s * Math.cos(dir - a / 2), -s * Math.sin(dir - a / 2)];
        g.add(new THREE.Mesh(extrude([[0, 0], v1, [v1[0] + v2[0], v1[1] + v2[1]], v2]), flute));
        const bar = new THREE.Mesh(new THREE.CylinderGeometry(4.5, 4.5, 90, 24), dark);
        bar.rotation.z = Math.PI / 2; bar.position.set(50, -6, 0); g.add(bar);
        break;
      }
      default: {
        const a = ((t.insertAngle || 80) * Math.PI) / 180, s = 12;
        const dir = Math.PI / 4; // insert body extends toward +Z and +X
        const v1 = [s * Math.cos(dir + a / 2), s * Math.sin(dir + a / 2)];
        const v2 = [s * Math.cos(dir - a / 2), s * Math.sin(dir - a / 2)];
        const far = [v1[0] + v2[0], v1[1] + v2[1]];
        g.add(new THREE.Mesh(extrude([[0, 0], v2, far, v1]), flute));
        holder(far[0] * 0.6, far[1] * 0.55, 22, 55);
      }
    }
    return g;
  }

  setTool(pos, move) {
    if (!this.result) return;
    const num = move ? move.tool : 0;
    const tool = this.result.tools[num];
    const key = num + '|' + (tool ? JSON.stringify([tool.type, tool.diameter, tool.fluteLength, tool.overallLength, tool.holderDiameter, tool.width, tool.insertAngle, tool.color, tool.cutter?.length, tool.cutter?.[1], tool.holder?.[1]]) : '');
    if (key !== this.toolKey) {
      this.clearGroup('tool');
      this.toolMesh = this.buildTool(tool);
      this.groups.tool.add(this.toolMesh);
      this.toolKey = key;
    }
    const p = this.toScene(pos);
    this.toolMesh.position.set(p[0], p[1], p[2]);
    // G41/G42 paths are nose centres: draw the tool at its reference point
    if (this.machine === 'lathe' && move && move.center && tool?.noseOffset) {
      this.toolMesh.position.x -= tool.noseOffset[0];
      this.toolMesh.position.y -= tool.noseOffset[1];
    }
    this.groups.tool.visible = this.visible.tool && num > 0;
    this.needsRender = true;
  }

  setTime(t) {
    this.uniforms.uTime.value = t;
    this.needsRender = true;
  }

  // ---------------------------------------------------------------- views
  bounds() {
    const b = new THREE.Box3();
    const s = this.result?.stock;
    if (!s) return b.set(new THREE.Vector3(-50, -50, -20), new THREE.Vector3(50, 50, 0));
    if (s.kind === 'lathe') {
      const r = s.diameter / 2;
      const zMin = this.sim?.zMin ?? -s.length - 15;
      b.set(new THREE.Vector3(zMin - 15, -r, -r), new THREE.Vector3((this.sim?.zMax ?? s.faceStock ?? 0) + 10, r, r));
    } else if (this.result?.usesRotary) {
      // the part turns about X (A) / Y (B) through the work zero: fit the swept volume
      const r = Math.max(...[s.min, s.max].flatMap((v) => [Math.abs(v[1]), Math.abs(v[2])]));
      b.set(new THREE.Vector3(s.min[0], -r, -r), new THREE.Vector3(s.max[0], r, r));
    } else {
      b.set(new THREE.Vector3(...s.min), new THREE.Vector3(...s.max));
    }
    return b;
  }

  setView(name) {
    this.lastView = name;
    const b = this.bounds();
    const c = b.getCenter(new THREE.Vector3());
    const size = b.getSize(new THREE.Vector3());
    const radius = Math.max(size.length() / 2, 10);
    // fit to the narrower field of view (portrait panes are limited horizontally)
    const vHalf = (this.camera.fov * Math.PI) / 360;
    const hHalf = Math.atan(Math.tan(vHalf) * this.camera.aspect);
    const dist = (radius / Math.sin(Math.min(vHalf, hHalf))) * 1.02;
    let dir;
    if (this.machine === 'mill') {
      dir = { iso: [0.85, -1.25, 1.05], top: [0, -0.0001, 1], front: [0, -1, 0.0001], side: [1, 0, 0.0001] }[name] || [0.85, -1.25, 1.05];
    } else {
      dir = { iso: [0.55, 0.75, 1.15], top: [0, 1, 0.0001], front: [0, 0.0001, 1], side: [1, 0.3, 0.2] }[name] || [0.55, 0.75, 1.15];
    }
    const d = new THREE.Vector3(...dir).normalize();
    this.camera.position.copy(c).addScaledVector(d, dist);
    this.camera.near = Math.max(0.1, dist / 200);
    this.camera.far = dist * 20;
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(c);
    this.controls.update();
    this.needsRender = true;
  }

  setVisible(key, v) {
    this.visible[key] = v;
    this.applyVisibility();
  }

  applyVisibility() {
    this.groups.path.visible = this.visible.path;
    if (this.rapidLines) this.rapidLines.visible = this.visible.rapids;
    this.groups.stock.visible = this.visible.stock;
    this.groups.target.visible = this.visible.target && this.groups.target.children.length > 0;
    this.groups.tool.visible = this.visible.tool && !!this.toolKey && !this.toolKey.startsWith('0|');
    this.needsRender = true;
  }

  // JPEG snapshots of named views (restores the camera afterwards)
  capture(views = ['iso', 'top'], quality = 0.82) {
    this.flush();
    const saved = { pos: this.camera.position.clone(), target: this.controls.target.clone(), view: this.lastView, near: this.camera.near, far: this.camera.far };
    const out = [];
    this.renderer.setClearColor(0x161b22, 1); // JPEG has no alpha
    for (const v of views) {
      this.setView(v);
      this.renderer.render(this.scene, this.camera);
      out.push({ view: v, dataUrl: this.renderer.domElement.toDataURL('image/jpeg', quality) });
    }
    this.renderer.setClearColor(0x000000, 0);
    this.camera.position.copy(saved.pos);
    this.controls.target.copy(saved.target);
    this.camera.near = saved.near; this.camera.far = saved.far;
    this.camera.updateProjectionMatrix();
    this.lastView = saved.view;
    this.controls.update();
    this.needsRender = true;
    return out;
  }
}
