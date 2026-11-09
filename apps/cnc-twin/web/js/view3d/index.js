/* CNC Twin 3D view (docs/ui-3d.md): the machine, fixture, in-process workpiece, tool and deflected
 * tool, chips and coolant of a run, in real units at the time the shell gives, with result layers,
 * sections, measuring, picking and a tool-following camera.
 *
 *   import { createView3D } from './view3d/index.js'
 *   const view = await createView3D(containerEl, { api, runId })
 *   view.setTime(t_s); view.setLayer(name | null); view.setSection({ axis, offset_m, flip });
 *   view.setMeasure(on); view.followTool(on); view.resetCamera(); view.setVisible({...});
 *   view.on('pick', cb); view.on('measure', cb); view.legend(layer); view.destroy()
 *
 * It reads only the run's API (api.get / api.getBinary, apiio.js) and draws in the stock frame S of
 * workpiece 0 (metres; for a spinning stock, scene.stock.spin, in the non-rotating W0). Rendering is on
 * demand: a frame is drawn when the time, the camera, the data or an animation changes, never in an
 * idle loop. Every pick and measurement is in S, metres; layer values in SI. */

import * as THREE from './three.js';
import { PARAMS } from './params.js';
import { adaptApi, RunData } from './apiio.js';
import { framesLayout, FrameTrack, emptySample } from './frames.js';
import { BrickLoader } from './bricks.js';
import { IPWRenderer } from './ipw.js';
import { NO_VALUE, HAS_VALUE } from './lod.js';
import { sceneChains, matrixFromPose, StockMotion, toolMatrix } from './poses.js';
import { SceneModel, sceneMaterials } from './scene3d.js';
import { ToolRig } from './tool.js';
import { ChipsView } from './chips.js';
import { CoolantView, jetsFromArrays } from './coolant.js';
import { Section } from './section.js';
import { MeasureTool, MODES } from './measure.js';
import { CameraRig } from './camera.js';
import { Overlay } from './overlay.js';
import { LAYERS, ATTRIBUTE_NAMES, DISPLAY, PartShading, coolantColors } from './layers.js';
import { layerGrids, gridTexture, sampleGrid, toolDeflection, sampleEvenly, pushExtremes, finiteRange } from './fields.js';
import { lut, LAYER_MAPS, mode as schemeMode, onSchemeChange, chrome, withAlpha } from '../plot/colormap.js';
import { formatSI, num } from '../plot/units.js';

const VISIBLE_KEYS = ['machine', 'fixture', 'chips', 'coolant', 'shadows', 'tool', 'deflectedTool'];
const EXAGGERATION_STEPS = [1, 2, 5];

function coarsePointer() {
  return typeof matchMedia === 'function' && matchMedia('(pointer: coarse)').matches;
}

/** The parameters for this device (phones: smaller caches, coarser LOD, lower pixel ratio). */
function deviceParams() {
  const touch = coarsePointer();
  return {
    ...PARAMS,
    touch,
    pixelRatioMax: touch ? PARAMS.max_pixel_ratio_touch : PARAMS.max_pixel_ratio,
    lodMaxPx: touch ? PARAMS.max_lod_cluster_px_touch : PARAMS.max_lod_cluster_px,
    brickCacheBytes: touch ? PARAMS.brick_cache_bytes_touch : PARAMS.brick_cache_bytes,
  };
}

function exaggerationLadder(min, max) {
  const out = [];
  for (let e = -3; e <= 7; e++) for (const s of EXAGGERATION_STEPS) {
    const v = s * 10 ** e;
    if (v >= min * (1 - 1e-9) && v <= max * (1 + 1e-9)) out.push(Number(v.toPrecision(3)));
  }
  return out;
}

class View3D {
  constructor(container, run, params) {
    this.container = container;
    this.run = run;
    this.params = params;
    this.notes = new Set();
    this.handlers = { pick: new Set(), measure: new Set() };
    this.t = 0;
    this.layer = null;
    this.visible = { machine: true, fixture: true, chips: true, coolant: true, shadows: false, tool: true, deflectedTool: false };
    this.destroyed = false;
    this.raf = 0;
    this.lastFrame = 0;
    this.keyframeIndex = -2;
    this.keyframeData = new Map();
    this.layerRanges = new Map();
    this.attributes = new Map();
    this.attributeLoads = new Map();
    this.deflectionSeries = new Map();
    this.timers = [];
    this.lastSetTime = { t: null, wall: 0, forward: false };
    this.stats = { frames: 0 };
  }

  // ------------------------------------------------------------------ set-up
  async start() {
    const p = this.params;
    const scene = this.run.scene;
    const stockSize = scene.stock && Array.isArray(scene.stock.size_m) ? scene.stock.size_m.map(Number) : null;
    this.stockSize = stockSize;
    let renderer;
    try {
      const ratio = Math.min(window.devicePixelRatio || 1, p.pixelRatioMax);
      renderer = new THREE.WebGLRenderer({ antialias: ratio <= p.max_antialias_pixel_ratio, alpha: false, stencil: true, powerPreference: 'high-performance' });
      renderer.setPixelRatio(ratio);
    } catch (error) {
      throw new Error(`WebGL 2 is not available in this browser (${error && error.message ? error.message : error})`);
    }
    this.renderer = renderer;
    renderer.localClippingEnabled = true;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.domElement.className = 'view3d-canvas';
    renderer.domElement.style.display = 'block';
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    renderer.domElement.style.touchAction = 'none';
    renderer.domElement.setAttribute('aria-label', '3D view of the machine and the part');
    this.container.append(renderer.domElement);
    this._background();

    this.scene = new THREE.Scene();
    this.root = new THREE.Group();
    this.root.name = 'view frame (S, or W0 for a spinning stock)';
    this.root.matrixAutoUpdate = false;
    this.scene.add(this.root);
    this.partRoot = new THREE.Group();
    this.partRoot.name = 'workpiece';
    this.root.add(this.partRoot);

    this.cameraRig = new CameraRig(renderer.domElement, p, () => this.invalidate());
    this.camera = this.cameraRig.camera;
    this.scene.add(this.camera);
    const hemi = new THREE.HemisphereLight(0xffffff, 0x8a8f8c, p.fill_light_intensity);
    hemi.position.set(0, 0, 1);
    this.scene.add(hemi);
    const key = new THREE.DirectionalLight(0xffffff, p.key_light_intensity);
    const az = p.key_light_azimuth_rad;
    const el = p.key_light_elevation_rad;
    key.position.set(-Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el));
    key.target.position.set(0, 0, 0);
    this.camera.add(key);
    this.camera.add(key.target);

    this.materials = sceneMaterials(p, this.run.material.family);
    this.shading = new PartShading({ unlitFraction: p.layer_unlit_fraction, faceCos: p.min_surface_field_cos });
    this.shading.patch(this.materials.part);
    this.section = new Section(this.root, { capSrgb: p.cap_srgb, fixtureCapSrgb: p.fixture_cap_srgb });
    this.section.clip(this.materials.part);
    this.section.clip(this.materials.fixture);

    this.chains = sceneChains(scene);
    this.sceneModel = new SceneModel({ run: this.run, params: p, root: this.root, materials: this.materials, chains: this.chains, notes: this.notes });
    const frames = scene.frames;
    this.layout = framesLayout(frames);
    this.sample = emptySample(this.layout);
    this.tools = new ToolRig({ run: this.run, params: p, root: this.root, materials: this.materials, notes: this.notes, nPaths: this.layout.nPaths });
    this.chips = new ChipsView({ root: this.root, params: p, density: this.run.material.density_kg_m3, section: this.section });
    const toolLinks = new Set(this.chains.tool ? [...this.chains.tool.names, ...this.chains.tool.axisNames] : []);
    this.coolant = new CoolantView({ root: this.root, params: p, nozzles: scene.nozzles, toolChainLinks: toolLinks });
    this.measure = new MeasureTool({ root: this.root, params: p, format: (v) => formatSI(v, 'm'), onResult: (r) => this._emit('measure', r) });
    this.overlay = new Overlay(this.container, {
      modes: MODES,
      onMode: (mode) => { this.measure.setMode(mode); this._measureText(); this.invalidate(); },
      onClear: () => { this.measure.clear(); this._measureText(); this.invalidate(); },
      onExaggeration: (dir) => this._stepExaggeration(dir),
    });

    const layoutDoc = { ...(scene.stock && scene.stock.bricks), ...(scene.bricks || {}), ...(this.run.bricks ? this.run.bricks.layout : {}) };
    if (!layoutDoc.lattice_m && scene.stock && scene.stock.lattice_m) layoutDoc.lattice_m = scene.stock.lattice_m;
    this.ipw = new IPWRenderer({
      parent: this.partRoot, material: this.materials.part, sectionMaterials: { back: this.section.back, front: this.section.front },
      params: p, layout: layoutDoc, workpiecePose: (w) => this._workpiecePose(w),
    });
    if (this.run.bricks) this._makeLoader();

    this.stockMotion = new StockMotion({
      staticPose: matrixFromPose(scene.stock && scene.stock.pose_M_S),
      poseFS: matrixFromPose(scene.stock && scene.stock.pose_F_S),
      workChain: this.chains.work,
      keyframeTimes: this._keyframeTimes(),
      toleranceM: p.pose_chain_tolerance_m,
      toleranceRad: p.pose_chain_tolerance_rad,
    });
    this.poseSW = new THREE.Matrix4();
    const poseWS = matrixFromPose(scene.stock && scene.stock.pose_W_S);
    if (poseWS) this.poseSW.copy(poseWS).invert();
    const g = Array.isArray(scene.gravity_M) ? new THREE.Vector3(...scene.gravity_M.map(Number)) : new THREE.Vector3(0, 0, -1);
    this.gravityM = (g.lengthSq() > 0 ? g.normalize() : new THREE.Vector3(0, 0, -1)).multiplyScalar(p.standard_gravity_m_s2);

    if (this.run.has('frames.bin') && this.run.files.get('frames.bin') > 0) {
      this.frames = new FrameTrack(this.layout, (t0, t1, max) => this.run.frames(t0, t1, max), {
        overviewMax: p.max_overview_frames, chunkRecords: p.frames_chunk_records, maxChunks: p.frames_chunks_kept,
        onLoad: () => this.invalidate(),
      });
      try { await this.frames.loadOverview(); } catch (error) { this.notes.add(`frames.bin: ${error.message}`); this.frames = null; }
    } else {
      this.notes.add('the run has written no frames yet: the tool is not shown');
    }

    await Promise.all([
      this.sceneModel.load(this.section).catch((error) => this.notes.add(`scene: ${error.message}`)),
      this.tools.load().catch((error) => this.notes.add(`tools: ${error.message}`)),
    ]);

    this._bindPointer();
    this.onContextRestored = () => { this.ipw.rebuildAll(); this.invalidate(); };
    renderer.domElement.addEventListener('webglcontextrestored', this.onContextRestored);
    this.resizeObserver = new ResizeObserver(() => this._resize());
    this.resizeObserver.observe(this.container);
    this._resize();
    this.homeBox = this._setupBox();
    this._update(0);
    this.cameraRig.sceneRadius = this._sceneRadius();
    this.cameraRig.frame(this.homeBox);
    this.unsubscribeScheme = onSchemeChange(() => { this._background(); this._applyLayer(); this.invalidate(); });
    if (this.run.active) this._startRefresh();
    this.invalidate();
  }

  _makeLoader() {
    const p = this.params;
    this.loader = new BrickLoader({
      index: this.run.bricks,
      fetchFile: (file) => this.run.keyframeFile(file),
      fetchBrick: (key, t) => this.run.brick(key, t),
      apply: (key, brick) => { this.ipw.setBrick(key, brick); if (this.layerKind() === 'attribute') this._needAttributes(); this.invalidate(); },
      onIdle: () => this.invalidate(),
      cacheBytes: p.brickCacheBytes, maxParallel: p.max_parallel_requests, perBrickMax: p.max_bricks_per_file_read,
    });
  }

  _keyframeTimes() {
    const times = [];
    for (const entry of this.run.keyframes) times[Number(entry.k)] = this.run.keyframeTime(entry);
    return times;
  }

  _workpiecePose(w) {
    const list = Array.isArray(this.run.scene.workpieces) ? this.run.scene.workpieces : [];
    const entry = list.find((e, i) => Number(e.workpiece ?? e.index ?? i) === w);
    const pose = entry && (matrixFromPose(entry.pose_S0_S) || null);
    if (!pose) this.notes.add(`workpiece ${w}: no pose relative to workpiece 0 in scene.json (workpieces[].pose_S0_S): not shown`);
    return pose;
  }

  _background() {
    const c = chrome();
    this.renderer.setClearColor(new THREE.Color().setStyle(c.surface || '#e4e9e5', THREE.SRGBColorSpace), 1);
  }

  /** The setup view frames the stock (the part is what the view is for; the machine is around it). */
  _setupBox() {
    const box = this.sceneModel.partBounds();
    if (box.isEmpty() && this.frames && this.frames.overview && this.frames.overview.count) {
      const span = this.frames.overview;
      const P = this.layout.nPaths;
      for (let i = 0; i < span.count; i += Math.max(1, Math.floor(span.count / 400))) {
        const v = new THREE.Vector3(span.tip[i * P * 3], span.tip[i * P * 3 + 1], span.tip[i * P * 3 + 2]);
        if ([v.x, v.y, v.z].every(Number.isFinite)) box.expandByPoint(v);
      }
    }
    if (box.isEmpty()) box.setFromCenterAndSize(new THREE.Vector3(), new THREE.Vector3(0.1, 0.1, 0.1));
    return box;
  }

  _sceneRadius() {
    const box = new THREE.Box3();
    for (const c of this.sceneModel.components) { c.mesh.geometry.computeBoundingBox(); box.union(c.mesh.geometry.boundingBox.clone().applyMatrix4(c.mesh.matrix)); }
    box.union(this.homeBox);
    return Math.max(box.getSize(new THREE.Vector3()).length(), 0.1);
  }

  _resize() {
    if (this.destroyed) return;
    const width = Math.max(1, this.container.clientWidth || 1);
    const height = Math.max(1, this.container.clientHeight || 1);
    this.renderer.setSize(width, height, false);
    this.cameraRig.resize(width, height);
    this.size = { width, height };
    this.invalidate();
  }

  _startRefresh() {
    const period = this.params.refresh_period_s * 1000;
    const tick = async () => {
      if (this.destroyed) return;
      try {
        const changed = await this.run.refresh();
        if (changed.keyframes && this.run.bricks) {
          if (!this.loader) this._makeLoader(); else this.loader.setIndex(this.run.bricks);
          this.stockMotion.setTimes(this._keyframeTimes());
          this.keyframeIndex = -2;
        }
        if (changed.frames) {
          if (this.frames) await this.frames.loadOverview();
          else if (this.run.has('frames.bin')) {
            this.frames = new FrameTrack(this.layout, (t0, t1, max) => this.run.frames(t0, t1, max), {
              overviewMax: this.params.max_overview_frames, chunkRecords: this.params.frames_chunk_records,
              maxChunks: this.params.frames_chunks_kept, onLoad: () => this.invalidate(),
            });
            await this.frames.loadOverview();
          }
        }
        if (!this.run.active) {
          this.layerRanges.clear();
          this.attributes.clear();
          this.attributeLoads.clear();
          this._applyLayer();
        }
        this.invalidate();
      } catch (error) {
        this.notes.add(`refresh: ${error.message}`);
      }
      if (!this.destroyed && this.run.active) this.timers.push(setTimeout(tick, period));
    };
    this.timers.push(setTimeout(tick, period));
  }

  // ------------------------------------------------------------------ keyframe data
  keyframeAt(t) {
    const list = this.run.keyframes;
    let lo = 0;
    let hi = list.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.run.keyframeTime(list[mid]) <= t + 1e-12) lo = mid + 1; else hi = mid;
    }
    return lo - 1;
  }

  _data(index) {
    const entry = this.run.keyframes[index];
    if (!entry) return null;
    const k = Number(entry.k);
    if (this.keyframeData.has(k)) {
      const value = this.keyframeData.get(k);
      this.keyframeData.delete(k);
      this.keyframeData.set(k, value);
      return value;
    }
    const record = { k, entry, fields: null, chips: null, ready: false };
    record.fieldsPromise = this.run.fields(entry).then((arrays) => { record.fields = arrays; return arrays; })
      .catch((error) => { this.notes.add(`keyframe ${k} fields: ${error.message}`); return null; });
    record.chipsPromise = this.run.chips(entry).then((arrays) => { record.chips = arrays; return arrays; })
      .catch((error) => { this.notes.add(`keyframe ${k} chips: ${error.message}`); return null; });
    record.promise = Promise.all([record.fieldsPromise, record.chipsPromise]).then(() => {
      record.ready = true;
      if (record.fields && record.fields.pose_M_S) this.stockMotion.setKeyframe(k, matrixFromPose(Array.from(record.fields.pose_M_S.data.slice(0, 16))));
      this.invalidate();
      return record;
    });
    this.keyframeData.set(k, record);
    while (this.keyframeData.size > this.params.keyframes_cached) this.keyframeData.delete(this.keyframeData.keys().next().value);
    return record;
  }

  async _onKeyframe(index) {
    const record = this._data(index);
    for (let d = 1; d <= this.params.prefetch_keyframes; d++) if (this.lastSetTime.forward) this._data(index + d);
    if (!record) {
      this.current = null;
      this.chips.set(null, null);
      this.coolant.setJets([], {});
      this.chips.setPile(null, null, this.stockSize, null);
      this._applyLayer();
      return;
    }
    await record.promise;
    if (this.destroyed || this.keyframeIndex !== index) return;
    // the next keyframe's pose of S in M: interpolation (or the chain test) between the two keyframes must
    // not depend on whether playback happened to read it ahead
    if (this.run.keyframes[index + 1] && ((record.fields && record.fields.pose_M_S) || this.stockMotion.chain)) {
      const next = this._data(index + 1);
      if (next && this.stockMotion.chain) next.promise.then(() => this._validateChain(index)).catch(() => {});
    }
    this.current = record;
    const tk = this.run.keyframeTime(record.entry);
    const atK = this.frames ? this.frames.sample(tk, { prefetch: false }) : null;
    const axesK = atK && atK.valid ? this._axesObject(atK) : {};
    const poseMS = this.stockMotion.at(tk, Number(record.entry.k), axesK);
    const poseSM = poseMS ? poseMS.clone().invert() : null;
    const tipK = atK && atK.valid ? new THREE.Vector3(atK.tip[0], atK.tip[1], atK.tip[2]) : null;
    const radius = this.tools.radius(0);
    const zoneS = tipK && radius ? tipK.clone().addScaledVector(new THREE.Vector3(atK.toolAxis[0], atK.toolAxis[1], atK.toolAxis[2]), this.params.tool_zone_height_ratio * radius) : tipK;
    const toolT0 = atK && atK.valid ? this._toolT0At(atK) : null;
    this.chips.set(record.chips, poseSM);
    this.chips.setPile(record.chips, record.fields, this.stockSize, this.chips.meanColour());
    if (!poseSM && record.chips && record.chips.pos_M_m && record.chips.pos_M_m.data.length) this.notes.add('chips and coolant are in M and the run gives no pose of S in M: not shown');
    const gravityS = poseSM ? this.gravityM.clone().applyMatrix3(new THREE.Matrix3().setFromMatrix4(poseSM)) : new THREE.Vector3(0, 0, -this.params.standard_gravity_m_s2);
    this.coolant.setJets(jetsFromArrays(record.fields), {
      poseSM, gravityS, zoneS, toolRadius: radius, toolT0,
      raycast: (origin, dir, maxT) => this._obstacle(origin, dir, maxT),
    });
    this._applyLayer();
    this.invalidate();
  }

  async _validateChain(index) {
    if (!this.frames) return;
    const a = this.run.keyframes[index];
    const b = this.run.keyframes[index + 1];
    if (!a || !b) return;
    const [sa, sb] = await Promise.all([this.frames.exactAt(Number(a.tick)), this.frames.exactAt(Number(b.tick))]);
    if (!sa || !sb) return;
    this.stockMotion.validate(Number(a.k), this._axesObject(sa), this._axesObject(sb));
    this.invalidate();
  }

  _axesObject(sample) {
    const out = {};
    this.layout.axes.forEach((axis, i) => { out[axis.name] = sample.axesAbs[i]; });
    return out;
  }

  _toolT0At(sample) {
    const tip = [sample.tip[0], sample.tip[1], sample.tip[2]];
    const axis = [sample.toolAxis[0], sample.toolAxis[1], sample.toolAxis[2]];
    return toolMatrix(new THREE.Matrix4(), tip, axis, this.xHint || new THREE.Vector3(1, 0, 0), 0);
  }

  /** Distance along a ray (S) to the first solid (part, fixture, machine, tool), or null. */
  _obstacle(origin, dir, maxT) {
    let best = null;
    const hit = this.ipw.raycast(origin, dir, null);
    if (hit && hit.t <= maxT) best = hit.t;
    const raycaster = new THREE.Raycaster(origin.clone().applyMatrix4(this.root.matrixWorld), dir.clone().transformDirection(this.root.matrixWorld), 0, best !== null ? best : maxT);
    const objects = [...this.sceneModel.components.filter((c) => c.mesh.visible && c.category !== 'enclosure').map((c) => c.mesh), ...this.tools.pickables()];
    const hits = raycaster.intersectObjects(objects, false);
    if (hits.length && (best === null || hits[0].distance < best)) best = hits[0].distance;
    return best;
  }

  // ------------------------------------------------------------------ layers
  layerKind(layer = this.layer) {
    if (!layer) return null;
    if (ATTRIBUTE_NAMES[layer]) return 'attribute';
    return 'field';
  }

  _finalVersion(key) {
    const list = this.run.bricks && this.run.bricks.entries.get(key);
    return list && list.length ? list[list.length - 1].k : null;
  }

  _attributeName(arrays) {
    for (const name of ATTRIBUTE_NAMES[this.layer] || []) if (arrays && arrays[name]) return name;
    return null;
  }

  /** Per-vertex values of the attribute layer for a shown brick (its final version only). */
  _attributeValues(key, brick) {
    if (this._finalVersion(key) !== brick.k) return null;
    const arrays = this.attributes.get(key);
    if (!arrays) return null;
    const name = this._attributeName(arrays);
    if (!name) return null;
    const data = arrays[name].data;
    if (data.length !== brick.nVertices) return null;
    const out = Float32Array.from(data);
    for (let i = 0; i < out.length; i++) if (!Number.isFinite(out[i])) out[i] = NO_VALUE;
    return out;
  }

  _needAttributes() {
    if (this.run.active) return;
    const files = new Set(this.run.attributeFiles());
    for (const key of this.ipw.bricks.keys()) {
      if (this.attributes.has(key) || this.attributeLoads.has(key)) continue;
      const path = `keyframes/attr/brick_${key}.npz`;
      if (!files.has(path)) continue;
      const load = this.run.attributes(path).then((arrays) => {
        this.attributes.set(key, arrays || {});
        const brick = this.ipw.bricks.get(key);
        if (brick && this.layerKind() === 'attribute') this.ipw.setBrick(key, brick);
        this.invalidate();
      }).catch((error) => { this.notes.add(`${path}: ${error.message}`); this.attributes.set(key, {}); });
      this.attributeLoads.set(key, load);
    }
  }

  /** The layer's colour range in SI with the shell's rules (12 keyframes / 24 attribute files, min-max,
   * deviation symmetric about 0), computed once per layer. */
  _range(layer) {
    if (this.layerRanges.has(layer)) return this.layerRanges.get(layer);
    const p = this.params;
    const promise = (async () => {
      const values = [];
      let source = '';
      if (layer === 'temperature' || layer === 'chips' || layer === 'deflection') {
        const frames = sampleEvenly(this.run.keyframes, p.field_sample_keyframes);
        for (const entry of frames) {
          const index = this.run.keyframes.indexOf(entry);
          const record = this._data(index);
          if (!record) continue;
          await record.promise;
          const arrays = record.fields;
          if (!arrays) continue;
          if (layer === 'temperature') {
            for (const name of Object.keys(arrays).filter((n) => /temp_K$/.test(n) && !n.startsWith('jet_')).sort()) pushExtremes(arrays[name], values);
          } else if (layer === 'chips') {
            pushExtremes(arrays.pile_height_m, values);
          } else {
            pushExtremes(arrays.tool_deflection_m, values, true);
          }
        }
        source = `${layer === 'temperature' ? 'temperature fields' : layer === 'chips' ? 'pile_height_m' : 'tool_deflection_m'} over ${frames.length} keyframe${frames.length === 1 ? '' : 's'}`;
        if (layer === 'deflection' && !values.length) {
          const channel = this._deflectionChannel();
          if (channel) {
            const doc = await this.run.series(channel.tier === 'fast' ? 'tick' : channel.tier, [channel.name], { maxPoints: 2000 });
            for (const s of (doc && doc.series) || []) for (const v of s.v || []) if (v !== null) values.push(Math.abs(v));
            source = `${channel.name} (series)`;
          }
        }
      } else if (ATTRIBUTE_NAMES[layer]) {
        const files = sampleEvenly(this.run.attributeFiles(), p.attribute_sample_bricks);
        for (const file of files) {
          try {
            const arrays = await this.run.attributes(file);
            const name = (ATTRIBUTE_NAMES[layer] || []).find((n) => arrays && arrays[n]);
            if (name) for (const v of arrays[name].data) values.push(v);
          } catch { /* an unreadable brick does not change the range */ }
        }
        source = `final surface attributes of ${files.length} of ${this.run.attributeFiles().length} bricks`;
      }
      let range = finiteRange(values);
      if (!range) return null;
      if (layer === 'deviation') { const b = Math.max(Math.abs(range[0]), Math.abs(range[1])); range = [-b, b]; }
      return { lo: range[0], hi: range[1], source };
    })().catch((error) => { this.notes.add(`${layer} colour range: ${error.message}`); return null; });
    const entry = { promise, value: undefined };
    this.layerRanges.set(layer, entry);
    promise.then((value) => {
      entry.value = value;
      if (this.destroyed || this.layerRanges.get(layer) !== entry) return;
      if (this.layer === layer) this._applyLayer();
      this.invalidate();
    });
    return entry;
  }

  _applyLayer() {
    const layer = this.layer;
    const kind = this.layerKind(layer);
    if (!layer) {
      this.shading.off();
      this.ipw.setAttributes(null, null);
      this.chips.colour(null);
      this.chips.colourPile(null, 0, 1, this.chips.meanColour());
      this.overlay.legend(null);
      return;
    }
    const display = DISPLAY[layer];
    if (layer === 'coolant') {
      const c = coolantColors();
      const grids = this.current && this.current.fields ? layerGrids(this.current.fields, this.stockSize) : {};
      this._fieldGrid = grids.coolant || null;
      if (grids.coolant) this.shading.field('coolant', grids.coolant, this._texture(grids.coolant), 0, 1, c); else this.shading.off();
      this.ipw.setAttributes(null, null);
      this.overlay.legend({ title: 'Coolant', categories: [[c.wetted, 'wetted'], [withAlpha(c.dry, c.dryAlpha), 'dry'], [c.jet, 'jet'], [c.shadow, 'shadowed flow']] });
      return;
    }
    const value = this._range(layer).value;
    const table = lut(LAYER_MAPS[layer], schemeMode(), 256);
    if (value === undefined) {
      this.shading.off();
      this.overlay.legend({ title: `${layer[0].toUpperCase()}${layer.slice(1)}`, pending: 'computing the colour scale…' });
      return;
    }
    if (value === null) {
      this.shading.off();
      this.overlay.legend({ title: `${layer[0].toUpperCase()}${layer.slice(1)}`, pending: 'the run records no values for this layer' });
      return;
    }
    const { lo, hi } = value;
    const title = { temperature: 'Temperature', chips: 'Chip pile height', finish: 'Finish (Ra)', deviation: 'Deviation', deflection: 'Deflection', wear: 'Wear (VB)' }[layer];
    this.overlay.legend({
      title: `${title} (${display.unit})`, table,
      minText: num(lo * display.scale + display.offset), maxText: num(hi * display.scale + display.offset),
      note: value.source,
    });
    this.chips.colour(layer === 'temperature' ? table : null, lo, hi);
    this.chips.colourPile(layer === 'chips' ? table : null, lo, hi, this.chips.meanColour());
    if (kind === 'attribute') {
      this._fieldGrid = null;
      // one provider for the view's life and the layer as key: chunks are rebuilt when the layer changes, and
      // per brick (setBrick) as its attribute file arrives, never all of them at every keyframe
      if (!this._attributeProvider) this._attributeProvider = (key, brick) => this._attributeValues(key, brick);
      this.ipw.setAttributes(this._attributeProvider, layer);
      this.shading.attribute(layer, lo, hi);
      this._needAttributes();
    } else {
      this.ipw.setAttributes(null, null);
      const grids = this.current && this.current.fields ? layerGrids(this.current.fields, this.stockSize) : {};
      const grid = layer === 'temperature' ? grids.temperature : layer === 'chips' ? grids.chips : grids.deflection;
      this._fieldGrid = grid || null;
      const floor = layer === 'chips' ? this.params.min_pile_height_m : null;
      if (grid) this.shading.field(layer, grid, this._texture(grid), lo, hi, null, floor); else this.shading.off();
    }
    this.toolTable = table;
    this.toolRange = [lo, hi];
  }

  _texture(grid) {
    if (this._tex && this._tex.grid === grid) return this._tex.value;
    if (this._tex) this._tex.value.texture.dispose();
    this._tex = { grid, value: gridTexture(grid) };
    return this._tex.value;
  }

  // ------------------------------------------------------------------ deflection
  _deflectionChannel() {
    if (this._deflChannel !== undefined && this._deflChannelFiles === this.run.files.size) return this._deflChannel;
    this._deflChannelFiles = this.run.files.size;
    this._deflChannel = this._findDeflectionChannel();
    return this._deflChannel;
  }

  _findDeflectionChannel() {
    const list = this.run.channels.filter((c) => c && c.unit === 'm' && /deflection|displacement/.test(c.name) && /tool/.test(c.name) && (c.domain === 'dynamics' || /^dynamics\./.test(c.name)));
    const shaped = list.find((c) => Array.isArray(c.max_shape) && c.max_shape.length === 1 && c.max_shape[0] >= 2 && c.max_shape[0] <= 3);
    if (!shaped) return null;
    const tier = shaped.tier === 'fast' ? 'tick' : shaped.tier;
    if (!this.run.hasSeries(tier)) return null;
    return { name: shaped.name, tier: shaped.tier, width: shaped.max_shape[0] };
  }

  /** Tool deflection at t for path 0 (T0 components, m): the channel's tick values when recorded,
   * else the newest keyframe's tool_deflection_m. */
  _deflection(t) {
    const out = [];
    const channel = this._deflectionChannel();
    let fromChannel = null;
    if (channel && (this.visible.deflectedTool || this.layer === 'deflection')) {
      const window = this.params.deflection_window_s;
      const w = Math.floor(t / window);
      const key = `${channel.name}:${w}`;
      if (!this.deflectionSeries.has(key)) {
        const tier = channel.tier === 'fast' ? 'tick' : channel.tier;
        const columns = [];
        for (let i = 0; i < channel.width; i++) columns.push(channel.tier === 'fast' ? `${channel.name}[${i}].mean` : `${channel.name}[${i}]`);
        const entry = { ready: false, series: null };
        this.deflectionSeries.set(key, entry);
        this.run.series(tier, columns, { t0: w * window, t1: (w + 1) * window, maxPoints: Math.ceil(window / (this.layout.ipoTicks * this.layout.tickS)) + 16 })
          .then((doc) => { entry.series = doc && doc.series; entry.ready = true; this.invalidate(); })
          .catch((error) => { entry.ready = true; this.notes.add(`${channel.name}: ${error.message}`); });
        while (this.deflectionSeries.size > 6) this.deflectionSeries.delete(this.deflectionSeries.keys().next().value);
      }
      const entry = this.deflectionSeries.get(key);
      if (entry && entry.ready && entry.series && entry.series.length) {
        const v = [0, 0, 0];
        let ok = false;
        entry.series.forEach((s, i) => {
          const ts = s.t_s || [];
          let lo = 0; let hi = ts.length;
          while (lo < hi) { const mid = (lo + hi) >> 1; if (ts[mid] <= t) lo = mid + 1; else hi = mid; }
          const j = lo - 1;
          if (j >= 0 && s.v[j] !== null) {
            let value = s.v[j];
            if (j + 1 < ts.length && s.v[j + 1] !== null && ts[j + 1] > ts[j]) value += (s.v[j + 1] - s.v[j]) * (t - ts[j]) / (ts[j + 1] - ts[j]);
            const component = Number(/\[(\d+)\]/.exec(s.column)?.[1] ?? i);
            if (component < 3) { v[component] = value; ok = true; }
          }
        });
        if (ok) fromChannel = { vector_m: v, frame: 'T0', source: channel.name };
      }
    }
    for (let p = 0; p < this.layout.nPaths; p++) {
      if (p === 0 && fromChannel) { out.push(fromChannel); continue; }
      const arrays = this.current && this.current.fields;
      out.push(arrays ? toolDeflection(arrays, p) : null);
    }
    return out;
  }

  _stepExaggeration(direction) {
    const ladder = exaggerationLadder(this.params.min_deflection_exaggeration, this.params.max_deflection_exaggeration);
    const now = this.tools.exaggeration;
    let index = ladder.findIndex((v) => v >= now * (1 - 1e-9));
    if (index < 0) index = ladder.length - 1;
    index = Math.max(0, Math.min(ladder.length - 1, index + direction));
    this.tools.setExaggeration(ladder[index]);
    this.invalidate();
  }

  // ------------------------------------------------------------------ frame update
  invalidate() {
    if (this.destroyed || this.raf) return;
    this.raf = requestAnimationFrame((now) => { this.raf = 0; this._frame(now); });
  }

  _frame(now) {
    if (this.destroyed) return;
    const dt = this.lastFrame ? Math.min(0.25, (now - this.lastFrame) / 1000) : 0;
    this.lastFrame = now;
    const busy = this._update(dt);
    this.renderer.render(this.scene, this.camera);
    this.stats.frames++;
    this._labels();
    if (busy) this.invalidate(); else this.lastFrame = 0;
  }

  /** Bring everything to time this.t. Returns true while something still moves or loads. */
  _update(dt) {
    let busy = false;
    this.updatedT = this.t;
    const sample = this.frames ? this.frames.sample(this.t) : null;
    const valid = !!(sample && sample.valid);
    if (valid) this._copySample(sample);
    else this.sample.valid = false;
    if (this.frames && valid && !sample.exact) busy = true;
    const index = this.keyframeAt(this.t);
    if (index !== this.keyframeIndex) {
      this.keyframeIndex = index;
      this._onKeyframe(index);
    }
    const axes = valid ? this._axesObject(this.sample) : {};
    const poseMS = this.stockMotion.at(this.t, index >= 0 ? Number(this.run.keyframes[index].k) : -1, axes);
    const poseSM = poseMS ? poseMS.clone().invert() : null;
    this.poseSM = poseSM;
    this.xHint = poseSM ? new THREE.Vector3(1, 0, 0).applyMatrix3(new THREE.Matrix3().setFromMatrix4(poseSM)) : new THREE.Vector3(1, 0, 0);
    // spinning stock (lathes): the view frame is W0, everything in S turns by the work spindle angle
    const spin = this.run.scene.stock && this.run.scene.stock.spin;
    if (spin && valid) {
      const path = Number(spin.path ?? 0);
      const axis = new THREE.Vector3(...(Array.isArray(spin.axis_S) ? spin.axis_S.map(Number) : [0, 0, 1])).normalize();
      const point = new THREE.Vector3(...(Array.isArray(spin.point_S_m) ? spin.point_S_m.map(Number) : [0, 0, 0]));
      const r = new THREE.Matrix4().makeRotationAxis(axis, this.sample.spindle[path] || 0);
      this.root.matrix.makeTranslation(point.x, point.y, point.z).multiply(r).multiply(new THREE.Matrix4().makeTranslation(-point.x, -point.y, -point.z));
      this.root.matrixWorldNeedsUpdate = true;
    }
    this.root.updateMatrixWorld(true);
    this.shading.uniforms.uSFromWorld.value.copy(this.root.matrixWorld).invert();
    // tools, machine, coolant
    const deflections = this._deflection(this.t);
    const colours = this._toolColours(deflections);
    const frames = valid ? this.tools.update(this.sample, this.layout, this.xHint, deflections, colours) : { T0: [], T: [] };
    if (!valid) for (const state of this.tools.paths) state.group.visible = false;
    this.toolFrames = frames;
    const linkPoses = this._linkPoses(axes);
    const missing = this.sceneModel.update({ poseSM, toolT0: frames.T0, toolT: frames.T, linkPoses, poseSW: this.poseSW });
    if (missing.length) this.notes.add(`no pose of S in M at some times: ${missing.join(', ')} not shown then`);
    this.coolant.placeNozzles(poseSM);
    this.coolant.update(this.t, frames.T0[0] || null);
    this.chips.setVisible(this.visible.chips);
    // workpiece
    if (this.loader) {
      this.loader.setTarget(this.t);
      if (this.lastSetTime.forward) {
        const next = this.run.keyframes[index + 1];
        if (next) this.loader.prefetch(this.run.keyframeTime(next));
      }
      if (this.loader.busy) busy = true;
    }
    if (this.ipw.update(this.camera, this.size ? this.size.height * this.renderer.getPixelRatio() : 600)) busy = true;
    if (this.sceneModel.stock) this.sceneModel.stock.visible = !this.ipw.ready();
    // camera
    const tip = valid ? new THREE.Vector3(this.sample.tip[0], this.sample.tip[1], this.sample.tip[2]).applyMatrix4(this.root.matrixWorld) : null;
    if (this.cameraRig.track(tip, dt)) busy = true;
    if (this.cameraRig.update()) busy = true;
    const px = this.size ? this.size.height * this.renderer.getPixelRatio() : 600;
    this.chips.update(this.camera, px);
    this.measure.update(this.camera, px);
    this._status();
    const deflection = deflections[0];
    const showDefl = this.visible.deflectedTool;
    this.overlay.deflection(showDefl, num(this.tools.exaggeration),
      deflection ? `${formatSI(Math.hypot(...deflection.vector_m), 'm')} at the tip` : 'not recorded');
    return busy;
  }

  _copySample(sample) {
    this.sample.valid = true;
    this.sample.exact = sample.exact;
    this.sample.tick = sample.tick;
    this.sample.time_s = sample.time_s;
    this.sample.axes.set(sample.axes);
    this.sample.axesAbs.set(sample.axesAbs);
    this.sample.spindle.set(sample.spindle);
    this.sample.tip.set(sample.tip);
    this.sample.toolAxis.set(sample.toolAxis);
    this.sample.tool.set(sample.tool);
    this.sample.hasTool = sample.hasTool;
  }

  _linkPoses(axes) {
    if (!this.sceneModel.components.some((c) => c.frame === 'link')) return null;
    return {
      work: this.chains.work && this.chains.work.complete ? this.chains.work.fk(axes).poses : new Map(),
      tool: this.chains.tool && this.chains.tool.complete ? this.chains.tool.fk(axes).poses : new Map(),
    };
  }

  _toolColours(deflections) {
    if (!this.layer || !this.toolTable || !this.toolRange) return null;
    const [lo, hi] = this.toolRange;
    const out = [];
    for (let p = 0; p < this.layout.nPaths; p++) {
      let value = null;
      if (this.layer === 'deflection' && deflections[p]) value = Math.hypot(...deflections[p].vector_m);
      else if (this.layer === 'temperature') value = this._macroValue(['thermal.tool_edge_max_K', 'thermal.edge_temp_K'], 'max');
      else if (this.layer === 'wear') value = this._macroValue(['wear.VB_m'], 'max');
      if (value === null || !Number.isFinite(value)) { out.push(null); continue; }
      const c = new THREE.Color();
      const t = hi > lo ? Math.max(0, Math.min(1, (value - lo) / (hi - lo))) : 0;
      const i = Math.min(254, Math.floor(t * 255));
      const f = t * 255 - i;
      const table = this.toolTable;
      c.setRGB((table[i * 3] + (table[(i + 1) * 3] - table[i * 3]) * f) / 255, (table[i * 3 + 1] + (table[(i + 1) * 3 + 1] - table[i * 3 + 1]) * f) / 255,
        (table[i * 3 + 2] + (table[(i + 1) * 3 + 2] - table[i * 3 + 2]) * f) / 255, THREE.SRGBColorSpace);
      out.push(c);
    }
    return out;
  }

  /** The value at t of the first macro channel recorded among `names` (max over its columns). */
  _macroValue(names, reduce) {
    const channel = this.run.channels.find((c) => names.includes(c.name));
    if (!channel || !this.run.hasSeries(channel.tier)) return null;
    const key = `macro:${channel.name}`;
    if (!this.deflectionSeries.has(key)) {
      const entry = { ready: false, series: null };
      this.deflectionSeries.set(key, entry);
      this.run.series(channel.tier, [channel.name], { maxPoints: 4000 })
        .then((doc) => { entry.series = doc && doc.series; entry.ready = true; this.invalidate(); })
        .catch(() => { entry.ready = true; });
    }
    const entry = this.deflectionSeries.get(key);
    if (!entry.ready || !entry.series) return null;
    let best = null;
    for (const s of entry.series) {
      const ts = s.t_s || [];
      let lo = 0; let hi = ts.length;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (ts[mid] <= this.t) lo = mid + 1; else hi = mid; }
      const v = lo > 0 ? s.v[lo - 1] : null;
      if (v !== null && Number.isFinite(v)) best = best === null ? v : reduce === 'max' ? Math.max(best, v) : best;
    }
    return best;
  }

  _status() {
    const parts = [];
    const entry = this.run.keyframes[this.keyframeIndex];
    if (entry) parts.push(`keyframe ${entry.k} at ${num(this.run.keyframeTime(entry))} s`);
    else if (this.run.keyframes.length) parts.push('before the first keyframe: stock');
    if (this.ipw.count) parts.push(`${this.ipw.count} bricks`);
    if (this.loader && this.loader.busy) parts.push('loading the workpiece…');
    if (this.frames && this.sample.valid && !this.sample.exact) parts.push('tool from the overview, loading full resolution…');
    if (this.loader && this.loader.errors.length) for (const e of this.loader.errors.slice(-3)) this.notes.add(e);
    this.overlay.status(parts.join(' · '), [...this.notes]);
  }

  _labels() {
    const labels = [];
    if (this.measure.labels.length && this.size) {
      for (const label of this.measure.labels) {
        const p = label.at.clone().applyMatrix4(this.root.matrixWorld).project(this.camera);
        if (p.z < -1 || p.z > 1) { labels.push(null); continue; }
        labels.push({ x: (p.x + 1) / 2 * this.size.width, y: (1 - p.y) / 2 * this.size.height, text: label.text });
      }
    }
    this.overlay.place(labels);
  }

  _measureText() {
    const mode = MODES.find((m) => m.id === this.measure.mode);
    const done = this.measure.result;
    let text;
    if (done) text = done.kind === 'hole' ? `Ø ${formatSI(done.diameter_m, 'm')}` : `${done.kind === 'plane' ? 'to plane' : 'distance'} ${formatSI(done.distance_m, 'm')}`;
    else if (this.measure.labels.length && !done) text = this.measure.labels[0].text;
    else text = `pick ${mode.picks - this.measure.picks.length} point${mode.picks - this.measure.picks.length === 1 ? '' : 's'}`;
    this.overlay.measure(this.measure.on, this.measure.mode, text);
  }

  // ------------------------------------------------------------------ picking
  _bindPointer() {
    const dom = this.renderer.domElement;
    const down = new Map();
    this.onPointerDown = (e) => { down.set(e.pointerId, { x: e.clientX, y: e.clientY, time: performance.now(), count: down.size + 1 }); };
    this.onPointerUp = (e) => {
      const start = down.get(e.pointerId);
      down.delete(e.pointerId);
      if (!start || start.count > 1) return;
      const moved = Math.hypot(e.clientX - start.x, e.clientY - start.y);
      if (moved > this.params.pick_tolerance_px || (performance.now() - start.time) / 1000 > this.params.max_tap_s) return;
      const rect = dom.getBoundingClientRect();
      this.pickAt(e.clientX - rect.left, e.clientY - rect.top);
    };
    this.onPointerCancel = (e) => down.delete(e.pointerId);
    dom.addEventListener('pointerdown', this.onPointerDown);
    dom.addEventListener('pointerup', this.onPointerUp);
    dom.addEventListener('pointercancel', this.onPointerCancel);
  }

  /** Pick at container pixel (x, y): emits 'pick' (and adds to a measurement). Returns the event. */
  pickAt(x, y) {
    if (!this.size) return null;
    const ndc = new THREE.Vector2((x / this.size.width) * 2 - 1, -(y / this.size.height) * 2 + 1);
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(ndc, this.camera);
    const toS = this.root.matrixWorld.clone().invert();
    const originS = raycaster.ray.origin.clone().applyMatrix4(toS);
    const dirS = raycaster.ray.direction.clone().transformDirection(toS);
    const plane = this.section.pickPlane();
    const planeS = plane ? (() => { const pl = new THREE.Plane(plane.normal, plane.constant).applyMatrix4(toS); return { normal: pl.normal, constant: pl.constant }; })() : null;
    let best = null;
    const part = this.ipw.count ? this.ipw.raycast(originS, dirS, planeS) : null;
    if (part) best = { ...part, object: 'part' };
    const meshes = [...this.sceneModel.pickables(), ...this.tools.pickables()];
    if (this.sceneModel.stock && this.sceneModel.stock.visible) meshes.push(this.sceneModel.stock);
    for (const hit of raycaster.intersectObjects(meshes, false)) {
      const pointS = hit.point.clone().applyMatrix4(toS);
      const category = hit.object.userData.category || hit.object.userData.kind || (hit.object === this.sceneModel.stock ? 'part' : 'machine');
      if (planeS && (category === 'fixture' || category === 'part') && planeS.normal.dot(pointS) + planeS.constant < 0) continue;
      const tS = pointS.distanceTo(originS);
      if (best && best.t <= tS) break;
      const normal = hit.face ? hit.face.normal.clone().transformDirection(hit.object.matrixWorld).transformDirection(toS) : new THREE.Vector3(0, 0, 1);
      best = { t: tS, point: pointS, normal, object: category === 'tool' || category === 'holder' ? 'tool' : category, mesh: hit.object };
      break;
    }
    if (!best) return null;
    const event = {
      point_m: [best.point.x, best.point.y, best.point.z],
      normal: [best.normal.x, best.normal.y, best.normal.z],
      layerValue: this._layerValueAt(best),
      object: best.object, cap: !!best.cap, brick: best.key ?? null, layer: this.layer,
    };
    this._emit('pick', event);
    if (this.measure.on) {
      this.measure.add(best);
      this._measureText();
    }
    this.invalidate();
    return event;
  }

  _layerValueAt(hit) {
    const layer = this.layer;
    if (!layer) return null;
    if (hit.object === 'tool') {
      if (layer === 'deflection') { const d = this._deflection(this.t)[0]; return d ? Math.hypot(...d.vector_m) : null; }
      if (layer === 'temperature') return this._macroValue(['thermal.tool_edge_max_K', 'thermal.edge_temp_K'], 'max');
      if (layer === 'wear') return this._macroValue(['wear.VB_m'], 'max');
      return null;
    }
    if (hit.object !== 'part') return null;
    if (this.layerKind(layer) === 'attribute') {
      if (!hit.key || !hit.vertices) return null;
      const brick = this.ipw.bricks.get(hit.key);
      const values = brick ? this._attributeValues(hit.key, brick) : null;
      if (!values) return null;
      let v = 0;
      for (let i = 0; i < 3; i++) {
        const value = values[hit.vertices[i]];
        if (!(value > HAS_VALUE)) return null;
        v += hit.bary[i] * value;
      }
      return v;
    }
    const grid = this._fieldGrid;
    if (!grid) return null;
    const value = sampleGrid(grid, hit.point.x, hit.point.y, hit.point.z);
    if (value === null) return null;
    return layer === 'coolant' ? (value >= 0.5 ? 1 : 0) : value;
  }

  _emit(name, event) {
    for (const cb of this.handlers[name] || []) {
      try { cb(event); } catch (error) { this.notes.add(`${name} handler: ${error.message}`); }
    }
  }

  // ------------------------------------------------------------------ the contract
  setTime(t) {
    const value = Number(t);
    if (!Number.isFinite(value)) return;
    const wall = performance.now();
    const last = this.lastSetTime;
    last.forward = last.t !== null && value > last.t && wall - last.wall < 500;
    last.t = value;
    last.wall = wall;
    this.t = value;
    if (this.loader) this.loader.setTarget(value);
    this.invalidate();
  }

  setLayer(name) {
    const layer = name === null || name === undefined || name === '' ? null : String(name);
    if (layer !== null && !LAYERS.includes(layer)) throw new Error(`unknown layer ${layer} (one of ${LAYERS.join(', ')})`);
    this.layer = layer;
    this._applyLayer();
    this.invalidate();
  }

  /** What a section can cut: the stock and the fixture (the caps must cover both). */
  _sectionBounds() {
    const box = this.homeBox.clone();
    for (const c of this.sceneModel.components) {
      if (c.category !== 'fixture' || !c.mesh.visible) continue;
      c.mesh.geometry.computeBoundingBox();
      box.union(c.mesh.geometry.boundingBox.clone().applyMatrix4(c.mesh.matrix));
    }
    return box;
  }

  setSection(section) {
    const active = this.section.set(section || {}, this._sectionBounds());
    this.ipw.setSection(active);
    this.invalidate();
  }

  setMeasure(on) {
    this.measure.setOn(on);
    this._measureText();
    this.invalidate();
  }

  followTool(on) {
    const tip = this.sample.valid ? new THREE.Vector3(this.sample.tip[0], this.sample.tip[1], this.sample.tip[2]).applyMatrix4(this.root.matrixWorld) : null;
    this.cameraRig.follow(on, tip);
    this.invalidate();
  }

  resetCamera() {
    this.cameraRig.follow(false);
    this.cameraRig.frame(this.homeBox);
    this.invalidate();
  }

  setVisible(visible = {}) {
    for (const key of VISIBLE_KEYS) if (visible[key] !== undefined) this.visible[key] = !!visible[key];
    this.sceneModel.setVisible(this.visible);
    this.tools.setVisible(this.visible);
    this.chips.setVisible(this.visible.chips);
    this.coolant.setVisible({ coolant: this.visible.coolant, shadows: this.visible.shadows });
    this.invalidate();
  }

  on(event, cb) {
    if (!this.handlers[event] || typeof cb !== 'function') return () => {};
    this.handlers[event].add(cb);
    return () => this.handlers[event].delete(cb);
  }

  /** {min, max, unit, map} in display units, {categories}, or null while the range is computed. */
  legend(layer) {
    if (!layer) return null;
    if (layer === 'coolant') {
      const c = coolantColors();
      return { categories: [[c.wetted, 'wetted'], [withAlpha(c.dry, c.dryAlpha), 'dry'], [c.jet, 'jet'], [c.shadow, 'shadowed flow']] };
    }
    const entry = this._range(layer);
    if (!entry.value) return null;
    const d = DISPLAY[layer];
    const min = entry.value.lo * d.scale + d.offset;
    let max = entry.value.hi * d.scale + d.offset;
    if (!(max > min)) max = min + 1e-9;
    return { min, max, unit: d.unit, map: LAYER_MAPS[layer], source: entry.value.source };
  }

  /** State for tests and diagnostics (not part of the contract). */
  debug() {
    const shown = {};
    if (this.loader) for (const [key, k] of this.loader.displayed) shown[key] = k;
    return {
      t: this.t, updated: this.updatedT === this.t, keyframe: this.run.keyframes[this.keyframeIndex] ? Number(this.run.keyframes[this.keyframeIndex].k) : null,
      bricks: shown, loading: this.loader ? this.loader.busy : false, ipw: { ...this.ipw.stats, count: this.ipw.count, ready: this.ipw.ready() },
      stockVisible: !!(this.sceneModel.stock && this.sceneModel.stock.visible),
      sample: this.sample.valid ? { exact: this.sample.exact, tip: Array.from(this.sample.tip.slice(0, 3)), axis: Array.from(this.sample.toolAxis.slice(0, 3)), spindle: this.sample.spindle[0] } : null,
      layer: this.layer, layerMode: this.shading.uniforms.uLayerMode.value, legend: this.layer ? this.legend(this.layer) : null,
      section: { ...this.section.state, active: this.section.active }, measure: { on: this.measure.on, mode: this.measure.mode, picks: this.measure.picks.length, result: this.measure.result },
      following: this.cameraRig.following, visible: { ...this.visible },
      chips: this.chips.chips ? this.chips.chips.length : 0, pile: !!this.chips.pile, jets: this.coolant.summary(),
      tools: this.tools.paths.map((s) => ({ visible: s.group.visible, shown: !!(s.toolMesh && s.toolMesh.visible), tool: s.current ? s.current.number : null, ghost: !!(s.ghost && s.ghost.visible), deflection: s.deflection })),
      chipsShown: !!(this.chips.mesh && this.chips.mesh.visible), coolantShown: this.coolant.jetGroup.visible,
      components: this.sceneModel.components.map((c) => ({ id: c.id, frame: c.frame, visible: c.mesh.visible, position: [c.mesh.matrix.elements[12], c.mesh.matrix.elements[13], c.mesh.matrix.elements[14]] })),
      root: Array.from(this.root.matrix.elements),
      poseSM: this.poseSM ? this.poseSM.toArray() : null, notes: [...this.notes],
      renderer: { calls: this.renderer.info.render.calls, triangles: this.renderer.info.render.triangles, pixelRatio: this.renderer.getPixelRatio() },
      frames: this.stats.frames, exaggeration: this.tools.exaggeration,
    };
  }

  /** Container pixel of an S point (tests click at known points). */
  project(point) {
    const p = new THREE.Vector3(...point).applyMatrix4(this.root.matrixWorld).project(this.camera);
    return this.size ? { x: (p.x + 1) / 2 * this.size.width, y: (1 - p.y) / 2 * this.size.height, visible: p.z >= -1 && p.z <= 1 } : null;
  }

  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    if (this.raf) cancelAnimationFrame(this.raf);
    for (const timer of this.timers) clearTimeout(timer);
    if (this.resizeObserver) this.resizeObserver.disconnect();
    if (this.unsubscribeScheme) this.unsubscribeScheme();
    const dom = this.renderer && this.renderer.domElement;
    if (dom && this.onContextRestored) dom.removeEventListener('webglcontextrestored', this.onContextRestored);
    if (dom && this.onPointerDown) {
      dom.removeEventListener('pointerdown', this.onPointerDown);
      dom.removeEventListener('pointerup', this.onPointerUp);
      dom.removeEventListener('pointercancel', this.onPointerCancel);
    }
    if (this.loader) this.loader.destroy();
    for (const part of [this.ipw, this.sceneModel, this.tools, this.chips, this.coolant, this.measure, this.section, this.shading, this.cameraRig]) {
      try { if (part) part.dispose(); } catch { /* disposing what is half built */ }
    }
    if (this._tex) this._tex.value.texture.dispose();
    for (const material of Object.values(this.materials || {})) material.dispose();
    if (this.overlay) this.overlay.destroy();
    if (this.renderer) {
      this.renderer.dispose();
      if (dom) dom.remove();
    }
    this.handlers.pick.clear();
    this.handlers.measure.clear();
  }

  publicApi() {
    const view = this;
    return {
      setTime: (t) => view.setTime(t),
      setLayer: (name) => view.setLayer(name),
      setSection: (section) => view.setSection(section),
      setMeasure: (on) => view.setMeasure(on),
      followTool: (on) => view.followTool(on),
      resetCamera: () => view.resetCamera(),
      setVisible: (visible) => view.setVisible(visible),
      on: (event, cb) => view.on(event, cb),
      legend: (layer) => view.legend(layer),
      setExaggeration: (factor) => { view.tools.setExaggeration(factor); view.invalidate(); },
      pickAt: (x, y) => view.pickAt(x, y),
      project: (point) => view.project(point),
      debug: () => view.debug(),
      destroy: () => view.destroy(),
    };
  }
}

/** Mount the 3D view of run `runId` in `container` (docs/ui-3d.md). */
export async function createView3D(container, { api, runId } = {}) {
  if (!container || typeof container.append !== 'function') throw new Error('createView3D needs a container element');
  if (runId === undefined || runId === null || runId === '') throw new Error('createView3D needs a runId');
  const run = new RunData(adaptApi(api), runId);
  await run.init();
  const view = new View3D(container, run, deviceParams());
  try {
    await view.start();
  } catch (error) {
    view.destroy();
    throw error;
  }
  return view.publicApi();
}

export { LAYERS };
