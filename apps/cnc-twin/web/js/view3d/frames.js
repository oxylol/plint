/* frames.bin for the 3D view: the record layout from scene.json["frames"] (docs/formats.md
 * "frames.bin"; this module is the only place that knows it), decoding of records, and FrameTrack,
 * which gives the machine state at any time t by interpolating between the two L1 ticks around it.
 *
 * Record (little-endian, record_bytes long, fields at their header offsets): tick (int64), axis
 * [n_axes] (float32, linear axes relative to axes[a].origin, rotary wrapped to [0, 2 pi)), axis_rev
 * [n_axes] (int32, optional), spindle_angle_rad [n_paths] (float32, wrapped) + spindle_rev [n_paths]
 * (int32), tip_S_m [n_paths, 3], axis_T_S [n_paths, 3] (float32), and optionally tool_number or
 * tool_slot [n_paths]. Any other field is decoded too (FrameBlock.extra) and left to the caller.
 *
 * Interpolation between ticks i and i+1 (w in [0, 1]): axis values linearly in axis space (rotary
 * axes unwrapped with their revolution count, so a turn through 2 pi interpolates correctly), the
 * spindle angle linearly in its unwrapped value (exact at constant speed), the tool tip linearly in
 * S, the tool axis by normalised linear interpolation (its angle error over one 1 ms tick is below
 * 1e-9 rad for any real tilt rate), the tool number of tick i. The tool pose is read from the record:
 * the view never re-implements the machine's kinematics. */

const DTYPES = {
  f8: [8, 'getFloat64'], f4: [4, 'getFloat32'], i8: [8, 'getBigInt64'], u8: [8, 'getBigUint64'],
  i4: [4, 'getInt32'], u4: [4, 'getUint32'], i2: [2, 'getInt16'], u2: [2, 'getUint16'], i1: [1, 'getInt8'], u1: [1, 'getUint8'],
  b1: [1, 'getUint8'],
};
const TWO_PI = 2 * Math.PI;

function fieldType(dtype) {
  const match = /^([<>|=]?)([a-z])(\d+)$/.exec(String(dtype));
  if (!match) throw new Error(`frames.bin: unsupported dtype ${dtype}`);
  const type = DTYPES[`${match[2]}${match[3]}`];
  if (!type) throw new Error(`frames.bin: unsupported dtype ${dtype}`);
  return { size: type[0], getter: type[1], little: match[1] !== '>', big: type[1].startsWith('getBig') };
}

/** The record layout of a frames header: {recordBytes, tickS, ipoTicks, axes, nAxes, nPaths, fields}. */
export function framesLayout(header) {
  if (!header || typeof header !== 'object') throw new Error('scene.json has no frames header');
  const recordBytes = Number(header.record_bytes);
  if (!(recordBytes > 0)) throw new Error('frames header: record_bytes is missing');
  const little = (header.byte_order || 'little') !== 'big';
  const fields = {};
  for (const field of header.fields || []) {
    const type = fieldType(field.dtype);
    const shape = Array.isArray(field.shape) ? field.shape.map(Number) : [];
    const count = shape.reduce((a, b) => a * b, 1);
    fields[field.name] = { ...type, little: field.dtype && /^[<>]/.test(field.dtype) ? type.little : little, offset: Number(field.offset), shape, count };
  }
  if (!fields.tick) throw new Error('frames header has no tick field');
  const axes = (header.axes || []).map((axis, index) => ({
    name: String(axis.name ?? `axis${index}`),
    kind: axis.kind === 'rotary' ? 'rotary' : 'linear',
    unit: axis.unit || (axis.kind === 'rotary' ? 'rad' : 'm'),
    origin: Number.isFinite(Number(axis.origin)) ? Number(axis.origin) : 0,
    hasOrigin: Number.isFinite(Number(axis.origin)),
  }));
  const nAxes = fields.axis ? fields.axis.count : axes.length;
  while (axes.length < nAxes) axes.push({ name: `axis${axes.length}`, kind: 'linear', unit: 'm', origin: 0, hasOrigin: false });
  let nPaths = Array.isArray(header.paths) && header.paths.length ? header.paths.length : 0;
  if (fields.tip_S_m) nPaths = Math.max(nPaths, Math.floor(fields.tip_S_m.count / 3));
  if (fields.spindle_angle_rad) nPaths = Math.max(nPaths, fields.spindle_angle_rad.count);
  nPaths = Math.max(1, nPaths);
  const paths = [];
  for (let p = 0; p < nPaths; p++) {
    const entry = (header.paths || [])[p] || {};
    paths.push({ path: entry.path ?? p, spindle: entry.spindle ?? null });
  }
  const toolField = fields.tool_number ? 'tool_number' : fields.tool_slot ? 'tool_slot' : null;
  return {
    recordBytes, little, tickS: Number(header.tick_s) || 1e-6, ipoTicks: Number(header.ipo_ticks) || 1000,
    axes, nAxes, nPaths, paths, fields, toolField,
    relative: axes.length > 0 && !axes.some((a) => a.hasOrigin),
  };
}

function reader(view, field, record, index) {
  const at = record + field.offset + index * field.size;
  const value = view[field.getter](at, field.little);
  return field.big ? Number(value) : value;
}

/** Decode whole records of `buffer` -> FrameBlock {count, tick, axis, spindle, tip, toolAxis, tool, extra}. */
export function decodeFrames(buffer, layout) {
  const bytes = buffer instanceof ArrayBuffer ? buffer : buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
  const count = Math.floor(bytes.byteLength / layout.recordBytes);
  const view = new DataView(bytes);
  const A = layout.nAxes;
  const P = layout.nPaths;
  const f = layout.fields;
  const tick = new Float64Array(count);
  const axis = new Float64Array(count * A);
  const spindle = new Float64Array(count * P);
  const tip = new Float64Array(count * P * 3);
  const toolAxis = new Float64Array(count * P * 3);
  const tool = layout.toolField ? new Int32Array(count * P) : null;
  const known = new Set(['tick', 'axis', 'axis_rev', 'spindle_angle_rad', 'spindle_rev', 'tip_S_m', 'axis_T_S', 'tool_number', 'tool_slot']);
  const extraFields = Object.keys(f).filter((name) => !known.has(name));
  const extra = {};
  for (const name of extraFields) extra[name] = new Float64Array(count * f[name].count);
  for (let r = 0; r < count; r++) {
    const base = r * layout.recordBytes;
    tick[r] = reader(view, f.tick, base, 0);
    for (let a = 0; a < A; a++) {
      let value = f.axis ? reader(view, f.axis, base, a) : 0;
      if (layout.axes[a].kind === 'rotary' && f.axis_rev) value += TWO_PI * reader(view, f.axis_rev, base, a);
      axis[r * A + a] = value;
    }
    for (let p = 0; p < P; p++) {
      let angle = f.spindle_angle_rad ? reader(view, f.spindle_angle_rad, base, p) : 0;
      if (f.spindle_rev) angle += TWO_PI * reader(view, f.spindle_rev, base, p);
      spindle[r * P + p] = angle;
      for (let c = 0; c < 3; c++) {
        tip[(r * P + p) * 3 + c] = f.tip_S_m ? reader(view, f.tip_S_m, base, p * 3 + c) : NaN;
        toolAxis[(r * P + p) * 3 + c] = f.axis_T_S ? reader(view, f.axis_T_S, base, p * 3 + c) : (c === 2 ? 1 : 0);
      }
      if (tool) tool[r * P + p] = reader(view, f[layout.toolField], base, p);
    }
    for (const name of extraFields) {
      const field = f[name];
      for (let j = 0; j < field.count; j++) extra[name][r * field.count + j] = reader(view, field, base, j);
    }
  }
  return { count, tick, axis, spindle, tip, toolAxis, tool, extra };
}

/** Index i with tick[i] <= x < tick[i + 1] (clamped to 0 .. count - 2; -1 when count is 0). */
export function bracket(tick, count, x) {
  if (count === 0) return -1;
  if (count === 1 || x <= tick[0]) return 0;
  if (x >= tick[count - 1]) return count - 2;
  let lo = 0;
  let hi = count - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (tick[mid] <= x) lo = mid; else hi = mid;
  }
  return lo;
}

/** A reusable machine state at one time. */
export function emptySample(layout) {
  return {
    valid: false, exact: false, time_s: 0, tick: 0,
    axes: new Float64Array(layout.nAxes), axesAbs: new Float64Array(layout.nAxes),
    spindle: new Float64Array(layout.nPaths), tip: new Float64Array(layout.nPaths * 3),
    toolAxis: new Float64Array(layout.nPaths * 3), tool: new Int32Array(layout.nPaths).fill(-1),
    hasTool: !!layout.toolField,
  };
}

/** Interpolate block records around base tick x into `out`. */
export function interpolate(block, layout, x, out) {
  const i = bracket(block.tick, block.count, x);
  if (i < 0) { out.valid = false; return out; }
  const j = Math.min(block.count - 1, i + 1);
  const t0 = block.tick[i];
  const t1 = block.tick[j];
  let w = t1 > t0 ? (x - t0) / (t1 - t0) : 0;
  w = Math.max(0, Math.min(1, w));
  const A = layout.nAxes;
  const P = layout.nPaths;
  for (let a = 0; a < A; a++) {
    const v = block.axis[i * A + a] + (block.axis[j * A + a] - block.axis[i * A + a]) * w;
    out.axes[a] = v;
    out.axesAbs[a] = v + layout.axes[a].origin;
  }
  for (let p = 0; p < P; p++) {
    out.spindle[p] = block.spindle[i * P + p] + (block.spindle[j * P + p] - block.spindle[i * P + p]) * w;
    let n = 0;
    for (let c = 0; c < 3; c++) {
      const k0 = (i * P + p) * 3 + c;
      const k1 = (j * P + p) * 3 + c;
      out.tip[p * 3 + c] = block.tip[k0] + (block.tip[k1] - block.tip[k0]) * w;
      const v = block.toolAxis[k0] + (block.toolAxis[k1] - block.toolAxis[k0]) * w;
      out.toolAxis[p * 3 + c] = v;
      n += v * v;
    }
    n = Math.sqrt(n) || 1;
    for (let c = 0; c < 3; c++) out.toolAxis[p * 3 + c] /= n;
    out.tool[p] = block.tool ? block.tool[i * P + p] : -1;
  }
  out.tick = x;
  out.time_s = x * layout.tickS;
  out.valid = true;
  return out;
}

/** Frames of a run at any time: an overview (every n-th record) plus full-resolution chunks loaded
 * around the times asked for. `fetch(t0_s, t1_s, maxFrames)` resolves to the records' bytes. */
export class FrameTrack {
  constructor(layout, fetch, { overviewMax = 20000, chunkRecords = 4096, maxChunks = 8, onLoad = null } = {}) {
    this.layout = layout;
    this.fetch = fetch;
    this.overviewMax = overviewMax;
    this.chunkRecords = chunkRecords;
    this.maxChunks = maxChunks;
    this.onLoad = onLoad;
    this.overview = null;
    this.overviewFull = false;
    this.chunks = new Map();
    this.pending = new Set();
    this.failed = new Set();
    this.generation = 0;
    this.sampleOut = emptySample(layout);
  }

  get chunkTicks() { return this.chunkRecords * this.layout.ipoTicks; }

  async loadOverview() {
    const buffer = await this.fetch(null, null, this.overviewMax);
    this.overview = decodeFrames(buffer, this.layout);
    let widest = 0;
    for (let i = 1; i < this.overview.count; i++) widest = Math.max(widest, this.overview.tick[i] - this.overview.tick[i - 1]);
    this.overviewFull = widest <= this.layout.ipoTicks * 1.5;
    this.generation++;
    for (const [c, chunk] of this.chunks) if (!chunk.complete) this.chunks.delete(c);
    this.failed.clear();
    return this.overview;
  }

  /** First and last time recorded (s), or null before the overview arrived. */
  span() {
    const o = this.overview;
    if (!o || !o.count) return null;
    return [o.tick[0] * this.layout.tickS, o.tick[o.count - 1] * this.layout.tickS];
  }

  lastTick() {
    const o = this.overview;
    return o && o.count ? o.tick[o.count - 1] : null;
  }

  _ensure(c) {
    if (c < 0 || this.chunks.has(c) || this.pending.has(c) || this.failed.has(c)) return;
    const last = this.lastTick();
    if (last !== null && c * this.chunkTicks > last) return;
    this.pending.add(c);
    const generation = this.generation;
    const tick0 = c * this.chunkTicks;
    const tick1 = (c + 1) * this.chunkTicks;
    const tickS = this.layout.tickS;
    this.fetch(tick0 * tickS, tick1 * tickS, null).then((buffer) => {
      this.pending.delete(c);
      const block = decodeFrames(buffer, this.layout);
      block.complete = block.count > 0 && block.tick[block.count - 1] >= tick1;
      block.used = performance.now();
      if (generation === this.generation || block.complete) this.chunks.set(c, block);
      while (this.chunks.size > this.maxChunks) {
        let oldest = null;
        for (const [key, value] of this.chunks) if (oldest === null || value.used < this.chunks.get(oldest).used) oldest = key;
        this.chunks.delete(oldest);
      }
      if (this.onLoad) this.onLoad(c);
    }).catch(() => {
      this.pending.delete(c);
      this.failed.add(c);
    });
  }

  /** The state at time t (s): exact when the full-resolution chunk around t is loaded, else from the
   * overview (exact = false) while the chunk loads. Returns a shared object; copy what you keep. */
  sample(t, { prefetch = true } = {}) {
    const out = this.sampleOut;
    const x = t / this.layout.tickS;
    const c = Math.max(0, Math.floor(x / this.chunkTicks));
    let block = this.chunks.get(c);
    if (block && block.count && (x > block.tick[block.count - 1] && !block.complete)) block = null;
    if (block && block.count) {
      block.used = performance.now();
      interpolate(block, this.layout, x, out);
      out.exact = true;
    } else if (this.overview && this.overview.count) {
      interpolate(this.overview, this.layout, x, out);
      out.exact = this.overviewFull;
      if (!out.exact) this._ensure(c);
    } else {
      out.valid = false;
    }
    if (prefetch && out.valid && this.overview && !this.overviewFull) {
      const into = (x - c * this.chunkTicks) / this.chunkTicks;
      if (into > 0.6) this._ensure(c + 1);
      if (!this.chunks.has(c)) this._ensure(c);
    }
    return out;
  }

  /** Exact record at a tick (for keyframe-aligned checks): resolves to a sample copy. */
  async exactAt(tick) {
    const tickS = this.layout.tickS;
    const buffer = await this.fetch(tick * tickS, tick * tickS, null);
    const block = decodeFrames(buffer, this.layout);
    if (!block.count) return null;
    const out = emptySample(this.layout);
    interpolate(block, this.layout, tick, out);
    out.exact = true;
    return out;
  }
}
