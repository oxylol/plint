/* The 3D view's access to a run (docs/api.md, architecture 17), through the `api` object it is given:
 * `get(path) -> JSON` and `getBinary(path) -> ArrayBuffer` (the UI shell's client also answers to its
 * other spellings, used when these two are absent). Every request names a file the run's listing
 * shows (or an endpoint that always answers), so a run that is still being written causes no 404s.
 *
 * RunData keeps what the view reads once per run: the run record, its file list, scene.json, the
 * keyframe index, bricks.idx, channels.json, timeline.json and the material's appearance family and
 * density; refresh() reads the growing parts again while the run is active. */

import { parseNpz } from './npz.js';
import { BrickIndex } from './bricks.js';

const ACTIVE = new Set(['queued', 'running', 'cancelling']);

/** {json(path), binary(path)} over the api object of the contract or of the UI shell. */
export function adaptApi(api) {
  if (!api) throw new Error('createView3D needs an api object');
  const json = ['get', 'getJSON', 'getJson', 'json', 'fetchJSON'].map((n) => api[n]).find((f) => typeof f === 'function');
  let binary = ['getBinary', 'getArrayBuffer', 'arrayBuffer', 'buffer', 'fetchArrayBuffer'].map((n) => api[n]).find((f) => typeof f === 'function');
  if (!binary && typeof api.binary === 'function') binary = async (path) => (await api.binary(path)).buffer;
  if (!json || !binary) throw new Error('the api object has no get(path) and getBinary(path)');
  return {
    json: (path) => json.call(api, path),
    binary: async (path) => {
      const value = await binary.call(api, path);
      if (value instanceof ArrayBuffer) return value;
      if (ArrayBuffer.isView(value)) return value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength);
      if (value && value.buffer instanceof ArrayBuffer) return value.buffer;
      throw new Error(`${path}: the api returned no binary data`);
    },
  };
}

export function query(params) {
  const parts = [];
  for (const [key, value] of Object.entries(params)) {
    if (value === null || value === undefined || value === '') continue;
    parts.push(`${encodeURIComponent(key)}=${encodeURIComponent(Array.isArray(value) ? value.join(',') : String(value))}`);
  }
  return parts.length ? `?${parts.join('&')}` : '';
}

function encodePath(path) {
  return String(path).split('/').map(encodeURIComponent).join('/');
}

export class RunData {
  constructor(io, runId) {
    this.io = io;
    this.id = String(runId);
    this.base = `/api/runs/${encodeURIComponent(this.id)}`;
    this.files = new Map();
    this.scene = null;
    this.keyframes = [];
    this.bricks = null;
    this.channels = [];
    this.timeline = [];
    this.toolChanges = new Map();
    this.record = null;
    this.material = { family: null, density_kg_m3: null };
    this.tickS = 1e-6;
  }

  get status() { return this.record && this.record.status ? this.record.status.status || null : null; }
  get active() { return ACTIVE.has(this.status); }
  has(path) { return this.files.has(path); }

  async init() {
    await this.readRecord();
    await this.readFiles();
    if (!this.has('scene.json')) throw new Error(`run ${this.id} has not written its scene (scene.json) yet`);
    this.scene = await this.io.json(`${this.base}/scene`);
    const frames = this.scene && this.scene.frames;
    if (frames && Number(frames.tick_s) > 0) this.tickS = Number(frames.tick_s);
    const run = this.record && this.record.run;
    if (run && run.clock && Number(run.clock.tick_s) > 0 && !(frames && frames.tick_s)) this.tickS = Number(run.clock.tick_s);
    await Promise.all([this.readKeyframes(), this.readChannels(), this.readTimeline(), this.readMaterial()]);
    return this;
  }

  async readRecord() {
    this.record = await this.io.json(this.base);
  }

  async readFiles() {
    const listing = await this.io.json(`${this.base}/files`);
    const files = new Map();
    for (const item of (listing && listing.files) || []) files.set(item.path, Number(item.bytes) || 0);
    this.files = files;
  }

  async readKeyframes() {
    if (this.has('keyframes/index.jsonl')) {
      const doc = await this.io.json(`${this.base}/keyframes`);
      this.keyframes = ((doc && doc.keyframes) || []).slice().sort((a, b) => Number(a.k) - Number(b.k));
    } else {
      this.keyframes = [];
    }
    if (this.has('keyframes/bricks.idx')) {
      const text = new TextDecoder('utf-8').decode(await this.io.binary(`${this.base}/files/keyframes/bricks.idx`));
      this.bricks = BrickIndex.parse(text, this.keyframes, this.tickS);
    } else {
      this.bricks = null;
    }
  }

  async readChannels() {
    if (!this.has('channels.json')) { this.channels = []; return; }
    const doc = await this.io.json(`${this.base}/channels`);
    this.channels = Array.isArray(doc) ? doc : (doc && doc.channels) || [];
  }

  async readTimeline() {
    if (!this.has('timeline.json')) { this.timeline = []; this.toolChanges = new Map(); return; }
    const doc = await this.io.json(`${this.base}/timeline`);
    this.timeline = Array.isArray(doc) ? doc : (doc && doc.blocks) || [];
    const changes = new Map();
    for (const block of this.timeline) {
      if (block.tool === null || block.tool === undefined || !Number.isFinite(Number(block.tick0))) continue;
      const path = Number(block.path ?? 0);
      if (!changes.has(path)) changes.set(path, []);
      changes.get(path).push([Number(block.tick0), Number(block.tool)]);
    }
    for (const list of changes.values()) list.sort((a, b) => a[0] - b[0]);
    this.toolChanges = changes;
  }

  async readMaterial() {
    const run = this.record && this.record.run;
    const material = run && run.project && run.project.material;
    if (!material) return;
    const physical = material.physical || {};
    this.material = {
      id: material.id || null,
      family: material.family || null,
      density_kg_m3: Number(physical.density_kg_m3 ?? material.density_kg_m3) || null,
    };
    if ((this.material.family && this.material.density_kg_m3) || !this.material.id) return;
    try {
      const items = await this.io.json('/api/db/materials');
      const list = Array.isArray(items) ? items : (items && (items.items || items.objects)) || [];
      if (!list.some((item) => item && item.id === this.material.id)) return;
      const doc = await this.io.json(`/api/db/materials/${encodeURIComponent(this.material.id)}`);
      const data = (doc && (doc.si || doc.data)) || {};
      const raw = (doc && doc.data) || {};
      this.material.family = this.material.family || data.family || raw.family || null;
      const density = data.physical && (data.physical.density_kg_m3 ?? data.physical.density);
      const rawDensity = raw.physical && raw.physical.density_kg_m3;
      const value = Number(density && typeof density === 'object' ? density.value : density) ||
        Number(rawDensity && typeof rawDensity === 'object' ? rawDensity.value : rawDensity);
      this.material.density_kg_m3 = this.material.density_kg_m3 || (Number.isFinite(value) && value > 0 ? value : null);
    } catch {
      // the database is a convenience for colours and chip sizes: without it the defaults apply
    }
  }

  /** Re-read the parts a running run extends. Returns {keyframes, files} change flags. */
  async refresh() {
    const before = { keyframes: this.keyframes.length, frames: this.files.get('frames.bin') || 0, files: this.files.size };
    await this.readRecord();
    await this.readFiles();
    await this.readKeyframes();
    if (this.has('timeline.json')) await this.readTimeline();
    return {
      keyframes: this.keyframes.length !== before.keyframes,
      frames: (this.files.get('frames.bin') || 0) !== before.frames,
      files: this.files.size !== before.files,
    };
  }

  keyframeTime(entry) {
    return Number.isFinite(Number(entry.time_s)) ? Number(entry.time_s) : Number(entry.tick) * this.tickS;
  }

  /** frames.bin records between t0 and t1 (s; null for the whole run), every n-th below maxFrames. */
  frames(t0, t1, maxFrames) {
    return this.io.binary(`${this.base}/frames${query({ format: 'bin', t0, t1, max_frames: maxFrames })}`);
  }

  keyframeFile(file) {
    return this.io.binary(`${this.base}/files/keyframes/${encodePath(file)}`);
  }

  brick(key, t) {
    return this.io.binary(`${this.base}/bricks/${encodeURIComponent(key)}${query({ t })}`);
  }

  async fields(entry) {
    const path = entry.fields || `keyframes/kf_${entry.k}.npz`;
    if (!this.has(path)) return null;
    return parseNpz(await this.io.binary(`${this.base}/keyframes/${encodeURIComponent(entry.k)}/fields${query({ format: 'npz' })}`));
  }

  /** The chips of a keyframe (/chips?t=<its time>&format=npz: the keyframe at or before t). */
  async chips(entry) {
    const path = entry.chips;
    if (!path || !this.has(path)) return null;
    return parseNpz(await this.io.binary(`${this.base}/chips${query({ t: this.keyframeTime(entry), format: 'npz' })}`));
  }

  attributeFiles() {
    return [...this.files.keys()].filter((p) => /^keyframes\/attr\/brick_.+\.npz$/.test(p)).sort();
  }

  async attributes(path) {
    if (!this.has(path)) return null;
    return parseNpz(await this.io.binary(`${this.base}/files/${encodePath(path)}`));
  }

  async file(path) {
    if (!this.has(path)) return null;
    return this.io.binary(`${this.base}/files/${encodePath(path)}`);
  }

  hasSeries(tier) {
    for (const path of this.files.keys()) if (path.startsWith(`series/${tier}/`) && path.endsWith('.parquet')) return true;
    return false;
  }

  series(tier, channels, { t0, t1, maxPoints } = {}) {
    return this.io.json(`${this.base}/series${query({ tier, channels, t0, t1, max_points: maxPoints })}`);
  }

  /** The tool number active on `path` at base tick `tick` from timeline.json (null if none). */
  toolAt(tick, path = 0) {
    const list = this.toolChanges && this.toolChanges.get(path);
    if (!list || !list.length || list[0][0] > tick) return null;
    let lo = 0;
    let hi = list.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (list[mid][0] <= tick) lo = mid + 1; else hi = mid;
    }
    return list[lo - 1][1];
  }
}
