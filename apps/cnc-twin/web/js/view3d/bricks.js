/* The in-process workpiece (IPW) as the run records it (docs/formats.md "Keyframes"): brick records,
 * keyframe GLB files, the brick index, and the loader that keeps every brick at its newest version at
 * or before the time shown.
 *
 * Brick record: header `<4sIIffff` (b"BRK1", n_vertices, n_indices, origin_S_m[3], step_m), uint16
 * positions [n_vertices, 4] (x, y, z, 0; p = origin + q step, metres in S) and uint32 indices
 * (triangles, counter-clockwise seen from outside); a record without vertices is an emptied brick.
 * A keyframe GLB holds one node per brick written at that keyframe; its BIN chunk is the records one
 * after the other (4-byte aligned, in node order), so a file is read without its glTF accessors. A
 * file whose records are not BRK1 is read through the glTF accessors of its brick_<b> nodes instead.
 *
 * bricks.idx: {bricks: {"<brick>": [[k, file, offset, length], ...]}} sorted by k (or JSON lines
 * {brick, k, file, offset, length}); the state of brick b at time t is its newest entry whose
 * keyframe time is <= t (+1e-12 s, the API's rule for /bricks/{b}?t=). */

export const RECORD_HEADER_BYTES = 28;
const MAGIC = 0x314b5242; // "BRK1" little-endian
const GLB_MAGIC = 0x46546c67;
const CHUNK_JSON = 0x4e4f534a;
const CHUNK_BIN = 0x004e4942;
const TIME_EPS_S = 1e-12;

/** Is there a BRK1 record at `offset`? */
export function isRecord(buffer, offset = 0) {
  if (offset + RECORD_HEADER_BYTES > buffer.byteLength) return false;
  return new DataView(buffer, offset, 4).getUint32(0, true) === MAGIC;
}

/** Byte length of the record at `offset` (from its header). */
export function recordLength(buffer, offset = 0) {
  const view = new DataView(buffer, offset, RECORD_HEADER_BYTES);
  return RECORD_HEADER_BYTES + view.getUint32(4, true) * 8 + view.getUint32(8, true) * 4;
}

/** One brick record -> {key, k, nVertices, nIndices, origin, step, q, indices, positions, bbox, bytes}.
 * The record bytes are copied once (an aligned buffer the typed arrays view). */
export function decodeRecord(buffer, offset = 0, length = null, key = null, k = null) {
  const total = length === null ? recordLength(buffer, offset) : length;
  if (total < RECORD_HEADER_BYTES || offset + total > buffer.byteLength) throw new Error(`brick ${key}: record outside its file`);
  const own = buffer.slice(offset, offset + total);
  const view = new DataView(own);
  if (view.getUint32(0, true) !== MAGIC) throw new Error(`brick ${key}: not a BRK1 record`);
  const nVertices = view.getUint32(4, true);
  const nIndices = view.getUint32(8, true);
  const need = RECORD_HEADER_BYTES + nVertices * 8 + nIndices * 4;
  if (need > total) throw new Error(`brick ${key}: record of ${total} bytes, its header needs ${need}`);
  const origin = [view.getFloat32(12, true), view.getFloat32(16, true), view.getFloat32(20, true)];
  const step = view.getFloat32(24, true);
  const q = new Uint16Array(own, RECORD_HEADER_BYTES, nVertices * 4);
  const indices = new Uint32Array(own, RECORD_HEADER_BYTES + nVertices * 8, nIndices);
  return withBounds({ key, k, nVertices, nIndices, origin, step, q, indices, positions: null, bytes: total });
}

function withBounds(brick) {
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  if (brick.q) {
    const q = brick.q;
    let qlo0 = 65535; let qlo1 = 65535; let qlo2 = 65535; let qhi0 = 0; let qhi1 = 0; let qhi2 = 0;
    for (let i = 0; i < brick.nVertices; i++) {
      const x = q[i * 4]; const y = q[i * 4 + 1]; const z = q[i * 4 + 2];
      if (x < qlo0) qlo0 = x; if (x > qhi0) qhi0 = x;
      if (y < qlo1) qlo1 = y; if (y > qhi1) qhi1 = y;
      if (z < qlo2) qlo2 = z; if (z > qhi2) qhi2 = z;
    }
    if (brick.nVertices) {
      lo[0] = brick.origin[0] + qlo0 * brick.step; hi[0] = brick.origin[0] + qhi0 * brick.step;
      lo[1] = brick.origin[1] + qlo1 * brick.step; hi[1] = brick.origin[1] + qhi1 * brick.step;
      lo[2] = brick.origin[2] + qlo2 * brick.step; hi[2] = brick.origin[2] + qhi2 * brick.step;
    }
  } else if (brick.positions) {
    const p = brick.positions;
    for (let i = 0; i < brick.nVertices; i++) {
      for (let c = 0; c < 3; c++) {
        const v = p[i * 3 + c];
        if (v < lo[c]) lo[c] = v;
        if (v > hi[c]) hi[c] = v;
      }
    }
  }
  brick.bbox = brick.nVertices ? { lo, hi } : null;
  return brick;
}

/** Vertex positions of a brick in S (Float32Array [n * 3], dequantised on first use and kept). */
export function brickPositions(brick) {
  if (brick.positions) return brick.positions;
  const n = brick.nVertices;
  const out = new Float32Array(n * 3);
  const q = brick.q;
  const [ox, oy, oz] = brick.origin;
  const s = brick.step;
  for (let i = 0; i < n; i++) {
    out[i * 3] = ox + q[i * 4] * s;
    out[i * 3 + 1] = oy + q[i * 4 + 1] * s;
    out[i * 3 + 2] = oz + q[i * 4 + 2] * s;
  }
  return out;
}

/** {json, binOffset, binLength} of a GLB file. */
export function parseGlb(buffer) {
  const view = new DataView(buffer);
  if (buffer.byteLength < 20 || view.getUint32(0, true) !== GLB_MAGIC) throw new Error('not a GLB file');
  let offset = 12;
  let json = null;
  let binOffset = -1;
  let binLength = 0;
  while (offset + 8 <= buffer.byteLength) {
    const length = view.getUint32(offset, true);
    const type = view.getUint32(offset + 4, true);
    if (type === CHUNK_JSON) json = JSON.parse(new TextDecoder('utf-8').decode(new Uint8Array(buffer, offset + 8, length)));
    else if (type === CHUNK_BIN) { binOffset = offset + 8; binLength = length; }
    offset += 8 + length;
  }
  if (!json) throw new Error('GLB file without a JSON chunk');
  return { json, binOffset, binLength };
}

export function brickKeyOfNode(node, index) {
  if (node && node.extras && node.extras.brick !== undefined && node.extras.brick !== null) return String(node.extras.brick);
  const name = node && typeof node.name === 'string' ? node.name : '';
  return name.startsWith('brick_') ? name.slice(6) : String(index);
}

const COMPONENT = { 5120: [1, 'getInt8'], 5121: [1, 'getUint8'], 5122: [2, 'getInt16'], 5123: [2, 'getUint16'], 5125: [4, 'getUint32'], 5126: [4, 'getFloat32'] };
const NORMALISE = { 5120: 127, 5121: 255, 5122: 32767, 5123: 65535 };

function accessorValues(json, buffer, binOffset, index, columns) {
  const accessor = json.accessors[index];
  const viewDef = json.bufferViews[accessor.bufferView];
  const [size, getter] = COMPONENT[accessor.componentType];
  const stride = viewDef.byteStride || size * columns;
  const start = binOffset + (viewDef.byteOffset || 0) + (accessor.byteOffset || 0);
  const view = new DataView(buffer);
  const out = new Float64Array(accessor.count * columns);
  const scale = accessor.normalized ? 1 / NORMALISE[accessor.componentType] : 1;
  for (let i = 0; i < accessor.count; i++) {
    for (let c = 0; c < columns; c++) out[i * columns + c] = view[getter](start + i * stride + c * size, true) * scale;
  }
  return out;
}

/** A brick from the glTF accessors of a node (files that do not hold BRK1 records). */
function nodeBrick(json, buffer, binOffset, node, key, k) {
  if (node.mesh === undefined) return { key, k, nVertices: 0, nIndices: 0, origin: [0, 0, 0], step: 0, q: null, indices: new Uint32Array(0), positions: new Float32Array(0), bbox: null, bytes: RECORD_HEADER_BYTES };
  const primitive = json.meshes[node.mesh].primitives[0];
  const raw = accessorValues(json, buffer, binOffset, primitive.attributes.POSITION, 3);
  const n = raw.length / 3;
  const t = node.translation || [0, 0, 0];
  const s = node.scale || [1, 1, 1];
  const positions = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) positions[i * 3 + c] = t[c] + raw[i * 3 + c] * s[c];
  let indices;
  if (primitive.indices !== undefined) {
    indices = Uint32Array.from(accessorValues(json, buffer, binOffset, primitive.indices, 1));
  } else {
    indices = new Uint32Array(n);
    for (let i = 0; i < n; i++) indices[i] = i;
  }
  return withBounds({ key, k, nVertices: n, nIndices: indices.length, origin: [0, 0, 0], step: 0, q: null, indices, positions, bytes: positions.byteLength + indices.byteLength });
}

/** Every brick written in a keyframe GLB: [{key, offset, brick}] in node order. */
export function keyframeBricks(buffer, k = null) {
  const { json, binOffset } = parseGlb(buffer);
  const nodes = json.nodes || [];
  const out = [];
  let position = binOffset;
  let sequential = binOffset >= 0;
  for (let i = 0; i < nodes.length && sequential; i++) {
    if (!isRecord(buffer, position)) { sequential = false; break; }
    const length = recordLength(buffer, position);
    out.push({ key: brickKeyOfNode(nodes[i], i), offset: position, brick: decodeRecord(buffer, position, length, brickKeyOfNode(nodes[i], i), k) });
    position += length;
    while ((position - binOffset) % 4) position++;
  }
  if (sequential) return out;
  return nodes.map((node, i) => ({ key: brickKeyOfNode(node, i), offset: -1, brick: nodeBrick(json, buffer, binOffset, node, brickKeyOfNode(node, i), k) }));
}

/** Natural order of brick keys ("12" < "w1.3"). */
export function keyOrder(a, b) {
  const pa = /^(?:w(\d+)\.)?(\d+)$/.exec(a);
  const pb = /^(?:w(\d+)\.)?(\d+)$/.exec(b);
  if (pa && pb) {
    const wa = Number(pa[1] || 0); const wb = Number(pb[1] || 0);
    if (wa !== wb) return wa - wb;
    return Number(pa[2]) - Number(pb[2]);
  }
  return a < b ? -1 : a > b ? 1 : 0;
}

/** Workpiece number of a brick key (0 for "<n>", w for "w<w>.<n>"). */
export function workpieceOf(key) {
  const match = /^w(\d+)\./.exec(key);
  return match ? Number(match[1]) : 0;
}

/** bricks.idx with the keyframe times: versionAt(key, t). */
export class BrickIndex {
  constructor(document, keyframes, tickS = 1e-6) {
    this.layout = {};
    this.entries = new Map();
    this.times = new Map();
    for (const entry of keyframes || []) {
      const time = Number.isFinite(Number(entry.time_s)) ? Number(entry.time_s) : Number(entry.tick) * tickS;
      this.times.set(Number(entry.k), time);
    }
    const add = (key, k, file, offset, length) => {
      const name = String(key);
      if (!this.entries.has(name)) this.entries.set(name, []);
      this.entries.get(name).push({ k: Number(k), file: String(file).replace(/^keyframes\//, ''), offset: Number(offset), length: Number(length) });
    };
    if (document && typeof document === 'object' && !Array.isArray(document) && document.bricks) {
      for (const name of ['cells', 'lattice_m', 'grid', 'origin_S_m']) if (document[name] !== undefined) this.layout[name] = document[name];
      for (const [key, list] of Object.entries(document.bricks)) {
        for (const e of list) {
          if (Array.isArray(e)) add(key, e[0], e[1], e[2], e[3]);
          else add(key, e.k, e.file, e.offset, e.length);
        }
      }
    } else if (Array.isArray(document)) {
      for (const record of document) add(record.brick, record.k, record.file, record.offset, record.length);
    }
    for (const list of this.entries.values()) {
      list.sort((a, b) => a.k - b.k);
      for (const e of list) e.time = this.times.has(e.k) ? this.times.get(e.k) : Infinity;
    }
    this.keys = [...this.entries.keys()].sort(keyOrder);
    this.fileBytes = new Map();
    for (const list of this.entries.values()) {
      for (const e of list) this.fileBytes.set(e.file, Math.max(this.fileBytes.get(e.file) || 0, e.offset + e.length));
    }
    this.sortedTimes = [...this.times.values()].sort((a, b) => a - b);
  }

  /** Number of keyframes at or before t: the IPW can only change when this does. */
  position(t) {
    const times = this.sortedTimes;
    let lo = 0;
    let hi = times.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (times[mid] <= t + TIME_EPS_S) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  static parse(text, keyframes, tickS) {
    let document;
    try {
      document = JSON.parse(text);
    } catch {
      document = text.split('\n').filter((line) => line.trim()).map((line) => JSON.parse(line));
    }
    return new BrickIndex(document, keyframes, tickS);
  }

  /** The newest entry of `key` whose keyframe time is <= t, or null (none yet, or unknown keyframes). */
  versionAt(key, t) {
    const list = this.entries.get(key);
    if (!list || !list.length) return null;
    let lo = 0;
    let hi = list.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (list[mid].time <= t + TIME_EPS_S) lo = mid + 1; else hi = mid;
    }
    return lo > 0 ? list[lo - 1] : null;
  }

  /** Entries of every brick stored in `file` (for reading a whole keyframe file once). */
  entriesInFile(file) {
    const out = [];
    for (const [key, list] of this.entries) for (const e of list) if (e.file === file) out.push({ key, ...e });
    return out;
  }

  /** Time of the first keyframe with any brick (null if none). */
  firstTime() {
    let best = null;
    for (const list of this.entries.values()) {
      for (const e of list) {
        const time = this.times.get(e.k);
        if (time !== undefined && (best === null || time < best)) best = time;
      }
    }
    return best;
  }
}

/** Keeps the bricks shown at their version for the target time: cached records are applied at once;
 * missing ones are read per brick (/bricks/{b}?t=, one ranged read on the server) when few are needed
 * from a file, else the whole keyframe file is read once and every record in it is cached. A
 * byte-budgeted LRU keeps records for scrubbing back and forth; prefetch() warms it for playback. */
export class BrickLoader {
  constructor({ index, fetchFile, fetchBrick, apply, onIdle = null, cacheBytes = 64 << 20, maxParallel = 4, perBrickMax = 6 }) {
    this.index = index;
    this.fetchFile = fetchFile;
    this.fetchBrick = fetchBrick;
    this.apply = apply;
    this.onIdle = onIdle;
    this.cacheBytes = cacheBytes;
    this.maxParallel = maxParallel;
    this.perBrickMax = perBrickMax;
    this.cache = new Map();
    this.cachedBytes = 0;
    this.displayed = new Map();
    this.shownBricks = new Map();
    this.failed = new Set();
    this.inflightFiles = new Map();
    this.inflightBricks = new Map();
    this.queue = [];
    this.running = 0;
    this.target = null;
    this.targetPosition = -1;
    this.prefetchTarget = null;
    this.errors = [];
    this.destroyed = false;
  }

  setIndex(index) {
    this.index = index;
    this.targetPosition = -1;
    this.failed.clear();
    if (this.target !== null) this.setTarget(this.target);
  }

  get busy() { return this.running > 0 || this.queue.length > 0; }

  _key(key, k) { return `${key}@${k}`; }

  _remember(brick) {
    const id = this._key(brick.key, brick.k);
    if (this.cache.has(id)) { this.cache.delete(id); this.cache.set(id, brick); return; }
    this.cache.set(id, brick);
    this.cachedBytes += brick.bytes;
    if (this.cachedBytes <= this.cacheBytes) return;
    for (const [oldest, value] of this.cache) {
      if (this.cachedBytes <= this.cacheBytes) break;
      if (this.displayed.get(value.key) === value.k) continue;
      this.cache.delete(oldest);
      this.cachedBytes -= value.bytes;
    }
  }

  _cached(key, k) {
    const id = this._key(key, k);
    const brick = this.cache.get(id);
    if (brick) { this.cache.delete(id); this.cache.set(id, brick); }
    return brick || null;
  }

  /** The brick shown for `key` (decoded record) or null. */
  shown(key) {
    return this.shownBricks.get(key) || null;
  }

  setTarget(t) {
    const position = this.index ? this.index.position(t) : -1;
    if (position === this.targetPosition && this.target !== null && !this.busy) { this.target = t; return; }
    this.target = t;
    this.targetPosition = position;
    this._plan(t, true);
  }

  prefetch(t) {
    this.prefetchTarget = t;
    if (!this.busy) this._plan(t, false);
  }

  _plan(t, show) {
    if (this.destroyed || !this.index) return;
    const missing = new Map();
    for (const key of this.index.keys) {
      const version = this.index.versionAt(key, t);
      if (!version) {
        if (show && this.displayed.has(key)) { this.displayed.delete(key); this.shownBricks.delete(key); this.apply(key, null); }
        continue;
      }
      if (show && this.displayed.get(key) === version.k) continue;
      const brick = this._cached(key, version.k);
      if (brick) {
        if (show) this._show(key, brick);
        continue;
      }
      if (!missing.has(version.file)) missing.set(version.file, []);
      missing.get(version.file).push({ key, ...version });
    }
    for (const [file, list] of missing) {
      if (this.inflightFiles.has(file)) continue;
      const wanted = list.filter((e) => !this.inflightBricks.has(this._key(e.key, e.k)));
      if (!wanted.length) continue;
      const fileBytes = this.index.fileBytes.get(file) || Infinity;
      const needBytes = wanted.reduce((sum, e) => sum + e.length, 0);
      if (wanted.length <= this.perBrickMax && needBytes * 4 < fileBytes) {
        for (const e of wanted) this._enqueue({ kind: 'brick', entry: e, t, priority: show ? 0 : 1 });
      } else {
        this._enqueue({ kind: 'file', file, priority: show ? 0 : 1 });
      }
    }
    this._run();
    if (show && !this.busy && this.onIdle) this.onIdle();
  }

  _show(key, brick) {
    this.displayed.set(key, brick.k);
    if (brick.nVertices) this.shownBricks.set(key, brick); else this.shownBricks.delete(key);
    this.apply(key, brick.nVertices ? brick : null);
  }

  _enqueue(task) {
    const id = task.kind === 'file' ? `f:${task.file}` : `b:${this._key(task.entry.key, task.entry.k)}`;
    if (this.failed.has(id) || this.queue.some((q) => q.id === id)) return;
    task.id = id;
    if (task.kind === 'file') this.inflightFiles.set(task.file, true);
    else this.inflightBricks.set(this._key(task.entry.key, task.entry.k), true);
    this.queue.push(task);
    this.queue.sort((a, b) => a.priority - b.priority);
  }

  _run() {
    while (!this.destroyed && this.running < this.maxParallel && this.queue.length) {
      const task = this.queue.shift();
      this.running++;
      const done = () => {
        this.running--;
        if (task.kind === 'file') this.inflightFiles.delete(task.file);
        else this.inflightBricks.delete(this._key(task.entry.key, task.entry.k));
        if (this.destroyed) return;
        if (this.target !== null) this._plan(this.target, true);
        if (!this.busy && this.prefetchTarget !== null && this.prefetchTarget !== this.target) this._plan(this.prefetchTarget, false);
        if (!this.busy && this.onIdle) this.onIdle();
      };
      const work = task.kind === 'file'
        ? this.fetchFile(task.file).then((buffer) => this._absorbFile(task.file, buffer))
        : this.fetchBrick(task.entry.key, task.t).then((buffer) => {
          this._remember(decodeRecord(buffer, 0, buffer.byteLength, task.entry.key, task.entry.k));
        });
      work.then(done, (error) => {
        this.errors.push(`${task.kind === 'file' ? task.file : `brick ${task.entry.key}`}: ${error && error.message ? error.message : error}`);
        if (this.errors.length > 20) this.errors.shift();
        this.failed.add(task.id);
        done();
      });
    }
  }

  _absorbFile(file, buffer) {
    const entries = this.index.entriesInFile(file);
    if (entries.length && entries.every((e) => isRecord(buffer, e.offset))) {
      for (const e of entries) {
        if (this.cache.has(this._key(e.key, e.k))) continue;
        this._remember(decodeRecord(buffer, e.offset, e.length, e.key, e.k));
      }
      return;
    }
    const kOf = new Map(entries.map((e) => [e.key, e.k]));
    for (const { key, brick } of keyframeBricks(buffer)) {
      if (!kOf.has(key)) continue;
      brick.k = kOf.get(key);
      this._remember(brick);
    }
  }

  destroy() {
    this.destroyed = true;
    this.queue = [];
    this.cache.clear();
    this.shownBricks.clear();
  }
}
