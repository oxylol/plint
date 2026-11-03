/* CNC Twin UI: client of the HTTP API (docs/api.md; methods listed in docs/ui.md "api.js").
 *
 * One object per server (`createApi({base})`; `api` is the same-origin default). Every method
 * returns a Promise and rejects with ApiError {status, detail, url, body} on an HTTP error. SI
 * throughout: names carry units exactly as the API's do. The same object is handed to the 3D
 * view (`createView3D(container, {api, runId})`), which reads the scene, frames, keyframes,
 * bricks, chips and jets through it.
 *
 * Also here, because every consumer of run files needs them: the frames.bin decoder (header in
 * scene.json, docs/formats.md "frames.bin"), an .npy/.npz reader (keyframe fields, chips,
 * keyframes/attr/*.npz) and a raw-DEFLATE inflater used when the browser has no
 * DecompressionStream('deflate-raw').
 */

export class ApiError extends Error {
  constructor(status, detail, url, body = null) {
    super(status ? `${status}: ${detail}` : String(detail));
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
    this.url = url;
    this.body = body;
  }
}

const TERMINAL = new Set(['done', 'failed', 'cancelled']);

/** "?a=1&b=x" from an object; null/undefined values are left out, arrays joined by commas. */
export function queryString(params) {
  if (!params) return '';
  const parts = [];
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null) continue;
    const text = Array.isArray(value) ? value.join(',') : String(value);
    parts.push(`${encodeURIComponent(key)}=${encodeURIComponent(text)}`);
  }
  return parts.length ? `?${parts.join('&')}` : '';
}

/** A relative file path with every segment encoded and the slashes kept. */
export function encodePath(path) {
  return String(path).split('/').map(encodeURIComponent).join('/');
}

const enc = encodeURIComponent;

export function createApi(options = {}) {
  const base = (options.base || '').replace(/\/+$/, '');
  const fetchImpl = options.fetch || ((...args) => globalThis.fetch(...args));
  const EventSourceImpl = options.EventSource || globalThis.EventSource;
  const memo = new Map();
  const api = { base };

  api.url = (path, params) => `${base}${path}${queryString(params)}`;

  async function request(path, { method = 'GET', params, body, headers = {}, signal } = {}) {
    const url = api.url(path, params);
    const init = { method, headers: { ...headers }, signal };
    if (body !== undefined) {
      if (body instanceof Blob || body instanceof FormData || body instanceof ArrayBuffer) {
        init.body = body;
      } else {
        init.body = JSON.stringify(body);
        init.headers['Content-Type'] = 'application/json';
      }
    }
    let response;
    try {
      response = await fetchImpl(url, init);
    } catch (error) {
      if (error && error.name === 'AbortError') throw error;
      throw new ApiError(0, `the server cannot be reached (${error && error.message ? error.message : error})`, url);
    }
    if (!response.ok) {
      let detail = response.statusText || 'request failed';
      let parsed = null;
      try {
        const text = await response.text();
        try {
          parsed = JSON.parse(text);
          if (parsed && parsed.detail !== undefined) detail = typeof parsed.detail === 'string' ? parsed.detail : JSON.stringify(parsed.detail);
        } catch {
          if (text) detail = text.slice(0, 300);
        }
      } catch { /* body unreadable: keep the status text */ }
      throw new ApiError(response.status, detail, url, parsed);
    }
    return response;
  }
  api.request = request;

  api.json = async (path, opts = {}) => {
    if (opts.cache) {
      const key = api.url(path, opts.params);
      if (!memo.has(key)) memo.set(key, request(path, opts).then((r) => r.json()).catch((error) => { memo.delete(key); throw error; }));
      return memo.get(key);
    }
    return (await request(path, opts)).json();
  };
  api.text = async (path, opts) => (await request(path, opts)).text();
  api.arrayBuffer = async (path, opts) => (await request(path, opts)).arrayBuffer();
  /** GET with the response headers: {buffer, headers}. */
  api.binary = async (path, opts) => {
    const response = await request(path, opts);
    return { buffer: await response.arrayBuffer(), headers: response.headers };
  };
  api.post = async (path, body, opts = {}) => {
    const response = await request(path, { ...opts, method: 'POST', body });
    const status = response.status;
    const data = status === 204 ? null : await response.json();
    return opts.withStatus ? { status, data } : data;
  };
  api.clearCache = () => memo.clear();

  // Aliases for generic callers (the 3D view may use any of these spellings).
  api.get = api.json;
  api.getJSON = api.json;
  api.getJson = api.json;
  api.fetchJSON = api.json;
  api.getText = api.text;
  api.buffer = api.arrayBuffer;
  api.getArrayBuffer = api.arrayBuffer;
  api.fetchArrayBuffer = api.arrayBuffer;

  // ------------------------------------------------------------------ server-sent events
  /** EventSource on `path` with JSON-parsed handlers {open, progress, finding, keyframe, done, error}.
   * Closed after `done`. Returns {close(), source}. */
  api.events = (path, handlers = {}, { lastEventId } = {}) => {
    if (!EventSourceImpl) throw new ApiError(0, 'this browser has no EventSource', path);
    const source = new EventSourceImpl(api.url(path, lastEventId ? { last_event_id: lastEventId } : undefined));
    let closed = false;
    const close = () => { closed = true; source.close(); };
    for (const name of ['progress', 'finding', 'keyframe', 'done']) {
      source.addEventListener(name, (event) => {
        let data = null;
        try { data = JSON.parse(event.data); } catch { data = event.data; }
        if (name === 'done') close();
        if (handlers[name]) handlers[name](data, event);
      });
    }
    source.onopen = (event) => { if (handlers.open) handlers.open(event); };
    source.onerror = (event) => {
      if (closed) return;
      if (handlers.error) handlers.error(event, source.readyState);
    };
    return { close, source };
  };

  // ------------------------------------------------------------------ health, projects
  api.health = () => api.json('/api/health');
  api.projects = () => api.json('/api/projects');
  api.project = (id) => api.json(`/api/projects/${enc(id)}`);
  api.projectFileUrl = (id, path) => api.url(`/api/projects/${enc(id)}/files/${encodePath(path)}`);
  api.projectFile = (id, path) => api.text(`/api/projects/${enc(id)}/files/${encodePath(path)}`);
  /** Upload a zipped project (multipart field `file`, optional id `name`). */
  api.uploadProject = (file, name) => {
    const form = new FormData();
    form.append('file', file, file.name || 'project.zip');
    if (name) form.append('name', name);
    return request('/api/projects', { method: 'POST', body: form }).then((r) => r.json());
  };
  /** twin check as a job: {status: 200|202, job, result}. */
  api.checkProject = async (id, options = {}, { waitS = 20 } = {}) => {
    const { status, data } = await api.post(`/api/projects/${enc(id)}/check`, { options },
      { params: { wait_s: waitS }, withStatus: true });
    return { status, ...data };
  };

  // ------------------------------------------------------------------ runs
  api.runs = ({ projectId, status, limit, offset } = {}) =>
    api.json('/api/runs', { params: { project_id: projectId, status, limit, offset } });
  api.run = (id) => api.json(`/api/runs/${enc(id)}`);
  api.createRun = (projectId, options = {}) => api.post('/api/runs', { project_id: projectId, options });
  api.cancelRun = (id) => api.post(`/api/runs/${enc(id)}/cancel`);
  api.resumeRun = (id) => api.post(`/api/runs/${enc(id)}/resume`);
  api.runEvents = (id, handlers, opts) => api.events(`/api/runs/${enc(id)}/events`, handlers, opts);

  // ------------------------------------------------------------------ results
  api.findings = (id, { severity, category, op } = {}) =>
    api.json(`/api/runs/${enc(id)}/findings`, { params: { severity, category, op } });
  api.trace = (id, eventId, { maxEvents } = {}) =>
    api.json(`/api/runs/${enc(id)}/trace/${enc(eventId)}`, { params: { max_events: maxEvents } });
  api.channels = (id, tier) => api.json(`/api/runs/${enc(id)}/channels`, { params: { tier } });
  /** Series of one tier: {tier, tick_s, t0_s, t1_s, rows, downsampled, method, buckets, series:[{column, channel, unit, n, t_s, v}]}. */
  api.series = (id, { tier = 'tick', channels = [], t0, t1, maxPoints, signal } = {}) =>
    api.json(`/api/runs/${enc(id)}/series`, {
      params: { tier, channels: Array.isArray(channels) ? channels.join(',') : channels, t0, t1, max_points: maxPoints },
      signal,
    });
  api.exportUrl = (id, { format = 'csv', tier = 'tick', channels, t0, t1 } = {}) =>
    api.url(`/api/runs/${enc(id)}/export`, { format, tier, channels: channels && channels.length ? channels : undefined, t0, t1 });
  api.program = (id, op) => api.json(`/api/runs/${enc(id)}/program/${enc(op)}`);
  api.timeline = (id, { op, t0, t1 } = {}) => api.json(`/api/runs/${enc(id)}/timeline`, { params: { op, t0, t1 } });
  api.scene = (id) => api.json(`/api/runs/${enc(id)}/scene`, { cache: true });
  /** frames.bin records between t0 and t1 (every n-th to stay below maxFrames), decoded with the
   * scene's frames header: see decodeFrames. */
  api.frames = async (id, { t0, t1, maxFrames, signal } = {}) => {
    const scene = await api.scene(id);
    const header = scene && scene.frames;
    if (!header) throw new ApiError(404, `run ${id} has no frames header in scene.json`, '');
    const { buffer, headers } = await api.binary(`/api/runs/${enc(id)}/frames`,
      { params: { t0, t1, format: 'bin', max_frames: maxFrames }, signal });
    const frames = decodeFrames(buffer, header);
    frames.stride = Number(headers.get('X-Frame-Stride') || 1);
    return frames;
  };
  api.framesJson = (id, { t0, t1, maxFrames } = {}) =>
    api.json(`/api/runs/${enc(id)}/frames`, { params: { t0, t1, format: 'json', max_frames: maxFrames } });
  api.keyframes = (id) => api.json(`/api/runs/${enc(id)}/keyframes`);
  api.keyframeGlbUrl = (id, k) => api.url(`/api/runs/${enc(id)}/keyframes/${enc(k)}.glb`);
  api.keyframeGlb = (id, k) => api.arrayBuffer(`/api/runs/${enc(id)}/keyframes/${enc(k)}.glb`);
  /** Coarse fields of keyframe k as JSON arrays {k, time_s, file, arrays: {name: {dtype, shape, data}}};
   * `decodeArrayDoc` turns one into a flat typed array. */
  api.keyframeFields = (id, k, names) =>
    api.json(`/api/runs/${enc(id)}/keyframes/${enc(k)}/fields`, { params: { names: names && names.length ? names : undefined } });
  /** One brick record at time t (one ranged read on the server): {buffer, keyframe, timeS, sourceFile}. */
  api.brick = async (id, brick, t) => {
    const { buffer, headers } = await api.binary(`/api/runs/${enc(id)}/bricks/${enc(brick)}`, { params: { t } });
    return {
      buffer,
      keyframe: Number(headers.get('X-Keyframe')),
      timeS: headers.get('X-Keyframe-Time-S') === null ? null : Number(headers.get('X-Keyframe-Time-S')),
      sourceFile: headers.get('X-Source-File'),
    };
  };
  api.chips = (id, { t, names } = {}) =>
    api.json(`/api/runs/${enc(id)}/chips`, { params: { t, names: names && names.length ? names : undefined } });
  api.jets = (id, t) => api.json(`/api/runs/${enc(id)}/jets`, { params: { t } });
  api.sldList = (id) => api.json(`/api/runs/${enc(id)}/sld`);
  api.sld = (id, tool) => api.json(`/api/runs/${enc(id)}/sld/${enc(tool)}`);
  api.inspection = (id) => api.json(`/api/runs/${enc(id)}/inspection`);
  api.surface = (id) => api.json(`/api/runs/${enc(id)}/surface`);
  api.tools = (id) => api.json(`/api/runs/${enc(id)}/tools`);
  api.production = (id) => api.json(`/api/runs/${enc(id)}/production`);
  api.files = (id) => api.json(`/api/runs/${enc(id)}/files`);
  api.fileUrl = (id, path) => api.url(`/api/runs/${enc(id)}/files/${encodePath(path)}`);
  api.file = (id, path) => api.arrayBuffer(`/api/runs/${enc(id)}/files/${encodePath(path)}`);
  api.fileText = (id, path) => api.text(`/api/runs/${enc(id)}/files/${encodePath(path)}`);
  api.fileJson = (id, path) => api.json(`/api/runs/${enc(id)}/files/${encodePath(path)}`);
  /** A run's .npz file (keyframes/attr/brick_<b>.npz ...) parsed: {name: {dtype, shape, data}}. */
  api.npz = async (id, path) => parseNpz(await api.file(id, path));
  // aliases for the 3D view
  api.getScene = api.scene;
  api.getFrames = api.frames;
  api.getKeyframes = api.keyframes;
  api.fields = api.keyframeFields;
  api.glb = api.keyframeGlb;
  api.runFileUrl = api.fileUrl;

  // ------------------------------------------------------------------ studies and jobs
  /** Compare two finished runs: {status: 200|202, job, result}. */
  api.compare = async (a, b, { waitS = 20, retry = false } = {}) => {
    const response = await request('/api/compare', { params: { a, b, wait_s: waitS, retry: retry ? 'true' : undefined } });
    return { status: response.status, ...(await response.json()) };
  };
  api.startUncertainty = (id, options = {}) => api.post(`/api/runs/${enc(id)}/uncertainty`, { options });
  api.calibrate = ({ projectId, runId, machineData, options = {} } = {}) =>
    api.post('/api/calibrate', { project_id: projectId || null, run_id: runId || null, machine_data: machineData || null, options });
  api.jobs = ({ kind, state, limit } = {}) => api.json('/api/jobs', { params: { kind, state, limit } });
  api.job = (id) => api.json(`/api/jobs/${enc(id)}`);
  api.cancelJob = (id) => api.post(`/api/jobs/${enc(id)}/cancel`);
  api.jobEvents = (id, handlers) => api.events(`/api/jobs/${enc(id)}/events`, handlers);
  api.jobFiles = (id) => api.json(`/api/jobs/${enc(id)}/files`);
  api.jobFileUrl = (id, path) => api.url(`/api/jobs/${enc(id)}/files/${encodePath(path)}`);
  api.jobFileJson = (id, path) => api.json(`/api/jobs/${enc(id)}/files/${encodePath(path)}`);
  api.jobLogUrl = (id) => api.url(`/api/jobs/${enc(id)}/log`);
  /** Follow a job to its end (server-sent events, polling when they fail): resolves {job, result}.
   * onProgress(jobRecord) on every change. */
  api.waitJob = (id, { onProgress, signal, pollS = 1.5 } = {}) => new Promise((resolve, reject) => {
    let stream = null;
    let timer = null;
    let finished = false;
    const finish = async () => {
      if (finished) return;
      finished = true;
      if (stream) stream.close();
      if (timer) clearTimeout(timer);
      try { resolve(await api.job(id)); } catch (error) { reject(error); }
    };
    const poll = async () => {
      if (finished) return;
      try {
        const record = await api.job(id);
        if (onProgress) onProgress(record.job);
        if (TERMINAL.has(record.job.state)) { finished = true; resolve(record); return; }
      } catch (error) { finished = true; reject(error); return; }
      timer = setTimeout(poll, pollS * 1000);
    };
    if (signal) signal.addEventListener('abort', () => { finished = true; if (stream) stream.close(); if (timer) clearTimeout(timer); reject(new DOMException('aborted', 'AbortError')); });
    if (!EventSourceImpl) { poll(); return; }
    stream = api.jobEvents(id, {
      progress: (job) => { if (onProgress && job && job.id) onProgress(job); },
      done: () => { finish(); },
      error: (_event, readyState) => { if (readyState === 2 && !finished) { stream.close(); poll(); } },
    });
  });

  // ------------------------------------------------------------------ databases
  api.dbKinds = () => api.json('/api/db');
  api.dbItems = (kind) => api.json(`/api/db/${enc(kind)}`);
  api.dbItem = (kind, id) => api.json(`/api/db/${enc(kind)}/${enc(id)}`);

  return api;
}

/** The API's prefix seen from the page: '' when the UI is at '/', '/twin' when it is at '/twin/'. */
export function pageBase() {
  if (typeof location === 'undefined' || !location.pathname) return '';
  return location.pathname.replace(/\/index\.html$/, '').replace(/\/+$/, '');
}

export const api = createApi({ base: pageBase() });
export default api;

// ============================================================================ frames.bin

const DTYPES = {
  f8: { size: 8, read: (v, o, le) => v.getFloat64(o, le), array: Float64Array, name: 'float64' },
  f4: { size: 4, read: (v, o, le) => v.getFloat32(o, le), array: Float32Array, name: 'float32' },
  f2: { size: 2, read: (v, o, le) => halfToFloat(v.getUint16(o, le)), array: Float32Array, name: 'float16' },
  i8: { size: 8, read: (v, o, le) => Number(v.getBigInt64(o, le)), array: Float64Array, name: 'int64' },
  u8: { size: 8, read: (v, o, le) => Number(v.getBigUint64(o, le)), array: Float64Array, name: 'uint64' },
  i4: { size: 4, read: (v, o, le) => v.getInt32(o, le), array: Int32Array, name: 'int32' },
  u4: { size: 4, read: (v, o, le) => v.getUint32(o, le), array: Uint32Array, name: 'uint32' },
  i2: { size: 2, read: (v, o, le) => v.getInt16(o, le), array: Int16Array, name: 'int16' },
  u2: { size: 2, read: (v, o, le) => v.getUint16(o, le), array: Uint16Array, name: 'uint16' },
  i1: { size: 1, read: (v, o) => v.getInt8(o), array: Int8Array, name: 'int8' },
  u1: { size: 1, read: (v, o) => v.getUint8(o), array: Uint8Array, name: 'uint8' },
  b1: { size: 1, read: (v, o) => v.getUint8(o), array: Uint8Array, name: 'bool' },
};

function parseDescr(descr) {
  const match = /^([<>|=]?)([a-zA-Z])(\d+)$/.exec(String(descr));
  if (!match) throw new Error(`unsupported dtype ${descr}`);
  const order = match[1];
  const kind = match[2];
  const size = Number(match[3]);
  return { little: order !== '>', kind, size, key: `${kind}${size}` };
}

/** Decode frames.bin bytes with the scene.json frames header ({record_bytes, fields:[{name, dtype,
 * shape, offset}], axes, paths, tick_s}). Returns {count, header, tick: Float64Array (base ticks),
 * time_s: Float64Array, fields: {name: {dtype, shape, width, data}}} with `data` flat, row-major,
 * `width` values per record. */
export function decodeFrames(buffer, header) {
  const recordBytes = Number(header.record_bytes);
  const view = new DataView(buffer);
  const count = recordBytes > 0 ? Math.floor(buffer.byteLength / recordBytes) : 0;
  const little = (header.byte_order || 'little') !== 'big';
  const tickS = Number(header.tick_s || 1e-6);
  const fields = {};
  for (const field of header.fields || []) {
    const { key } = parseDescr(field.dtype);
    const type = DTYPES[key];
    if (!type) throw new Error(`frames field ${field.name}: unsupported dtype ${field.dtype}`);
    const shape = field.shape || [];
    const width = shape.reduce((a, b) => a * b, 1);
    const data = new type.array(count * width);
    for (let r = 0; r < count; r++) {
      const base = r * recordBytes + Number(field.offset);
      for (let j = 0; j < width; j++) data[r * width + j] = type.read(view, base + j * type.size, little);
    }
    fields[field.name] = { dtype: type.name, shape, width, data };
  }
  const tick = fields.tick ? Float64Array.from(fields.tick.data) : new Float64Array(count);
  const time = new Float64Array(count);
  for (let i = 0; i < count; i++) time[i] = tick[i] * tickS;
  return { count, header, tick, time_s: time, fields };
}

// ============================================================================ .npy / .npz

export function halfToFloat(h) {
  const sign = (h & 0x8000) ? -1 : 1;
  const exponent = (h >> 10) & 0x1f;
  const fraction = h & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function parseNpyHeader(text) {
  const descr = /'descr'\s*:\s*'([^']+)'/.exec(text);
  const fortran = /'fortran_order'\s*:\s*(True|False)/.exec(text);
  const shape = /'shape'\s*:\s*\(([^)]*)\)/.exec(text);
  if (!descr || !fortran || !shape) throw new Error(`not an .npy header: ${text.slice(0, 80)}`);
  const dims = shape[1].split(',').map((s) => s.trim()).filter((s) => s.length).map(Number);
  return { descr: descr[1], fortran: fortran[1] === 'True', shape: dims };
}

function toCOrder(data, shape) {
  if (shape.length < 2) return data;
  const total = data.length;
  const out = new data.constructor(total);
  const nd = shape.length;
  const fStride = new Array(nd);
  fStride[0] = 1;
  for (let d = 1; d < nd; d++) fStride[d] = fStride[d - 1] * shape[d - 1];
  const index = new Array(nd).fill(0);
  for (let c = 0; c < total; c++) {
    let f = 0;
    for (let d = 0; d < nd; d++) f += index[d] * fStride[d];
    out[c] = data[f];
    for (let d = nd - 1; d >= 0; d--) {
      index[d]++;
      if (index[d] < shape[d]) break;
      index[d] = 0;
    }
  }
  return out;
}

/** Parse one .npy file (Uint8Array or ArrayBuffer): {dtype, shape, data}. Integers of 8 bytes come
 * back as Float64Array, float16 as Float32Array, booleans as Uint8Array, strings as an Array. */
export function parseNpy(input) {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  if (bytes[0] !== 0x93 || String.fromCharCode(...bytes.subarray(1, 6)) !== 'NUMPY') throw new Error('not an .npy file');
  const major = bytes[6];
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLength = major >= 2 ? view.getUint32(8, true) : view.getUint16(8, true);
  const headerStart = major >= 2 ? 12 : 10;
  const headerText = new TextDecoder('latin1').decode(bytes.subarray(headerStart, headerStart + headerLength));
  const { descr, fortran, shape } = parseNpyHeader(headerText);
  const dataStart = headerStart + headerLength;
  const count = shape.reduce((a, b) => a * b, 1);
  const parsed = parseDescr(descr);
  let data;
  let dtype;
  if (parsed.kind === 'U') {
    const chars = parsed.size;
    data = new Array(count);
    for (let i = 0; i < count; i++) {
      let s = '';
      for (let c = 0; c < chars; c++) {
        const code = view.getUint32(dataStart + (i * chars + c) * 4, parsed.little);
        if (code === 0) break;
        s += String.fromCodePoint(code);
      }
      data[i] = s;
    }
    dtype = 'str';
  } else if (parsed.kind === 'S') {
    const size = parsed.size;
    const decoder = new TextDecoder('utf-8');
    data = new Array(count);
    for (let i = 0; i < count; i++) {
      const raw = bytes.subarray(dataStart + i * size, dataStart + (i + 1) * size);
      const end = raw.indexOf(0);
      data[i] = decoder.decode(end >= 0 ? raw.subarray(0, end) : raw);
    }
    dtype = 'bytes';
  } else {
    const type = DTYPES[parsed.key];
    if (!type) throw new Error(`unsupported dtype ${descr}`);
    data = new type.array(count);
    for (let i = 0; i < count; i++) data[i] = type.read(view, dataStart + i * type.size, parsed.little);
    dtype = type.name;
  }
  if (fortran && !Array.isArray(data)) data = toCOrder(data, shape);
  return { dtype, shape, data };
}

/** Parse an .npz (zip of .npy members, stored or deflated): {name: {dtype, shape, data}}. */
export async function parseNpz(input) {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  const out = {};
  for (const entry of zipEntries(bytes)) {
    if (!entry.name.endsWith('.npy')) continue;
    out[entry.name.slice(0, -4)] = parseNpy(await zipData(bytes, entry));
  }
  return out;
}

function zipEntries(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let eocd = -1;
  for (let i = bytes.length - 22; i >= Math.max(0, bytes.length - 22 - 65535); i--) {
    if (view.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  }
  if (eocd < 0) throw new Error('not a zip file (no end of central directory)');
  let total = view.getUint16(eocd + 10, true);
  let offset = view.getUint32(eocd + 16, true);
  if (offset === 0xffffffff || total === 0xffff) {
    const locator = eocd - 20;
    if (locator >= 0 && view.getUint32(locator, true) === 0x07064b50) {
      const zip64 = Number(view.getBigUint64(locator + 8, true));
      total = Number(view.getBigUint64(zip64 + 32, true));
      offset = Number(view.getBigUint64(zip64 + 48, true));
    }
  }
  const decoder = new TextDecoder('utf-8');
  const entries = [];
  for (let n = 0; n < total; n++) {
    if (view.getUint32(offset, true) !== 0x02014b50) throw new Error('zip: bad central directory entry');
    const method = view.getUint16(offset + 10, true);
    let compressed = view.getUint32(offset + 20, true);
    let size = view.getUint32(offset + 24, true);
    const nameLength = view.getUint16(offset + 28, true);
    const extraLength = view.getUint16(offset + 30, true);
    const commentLength = view.getUint16(offset + 32, true);
    let local = view.getUint32(offset + 42, true);
    const name = decoder.decode(bytes.subarray(offset + 46, offset + 46 + nameLength));
    let extra = offset + 46 + nameLength;
    const extraEnd = extra + extraLength;
    while (extra + 4 <= extraEnd) {
      const id = view.getUint16(extra, true);
      const length = view.getUint16(extra + 2, true);
      if (id === 0x0001) {
        let p = extra + 4;
        if (size === 0xffffffff) { size = Number(view.getBigUint64(p, true)); p += 8; }
        if (compressed === 0xffffffff) { compressed = Number(view.getBigUint64(p, true)); p += 8; }
        if (local === 0xffffffff) { local = Number(view.getBigUint64(p, true)); }
      }
      extra += 4 + length;
    }
    entries.push({ name, method, compressed, size, local });
    offset = extraEnd + commentLength;
  }
  return entries;
}

async function zipData(bytes, entry) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(entry.local, true) !== 0x04034b50) throw new Error(`zip: bad local header of ${entry.name}`);
  const start = entry.local + 30 + view.getUint16(entry.local + 26, true) + view.getUint16(entry.local + 28, true);
  const raw = bytes.subarray(start, start + entry.compressed);
  if (entry.method === 0) return raw;
  if (entry.method === 8) return inflateRaw(raw);
  throw new Error(`zip: ${entry.name} uses compression method ${entry.method}`);
}

/** Raw DEFLATE (RFC 1951) to bytes: the browser's DecompressionStream when it has 'deflate-raw',
 * else inflateRawSync. */
export async function inflateRaw(bytes) {
  if (typeof DecompressionStream === 'function') {
    let stream = null;
    try { stream = new DecompressionStream('deflate-raw'); } catch { stream = null; }
    if (stream) {
      const output = new Response(new Blob([bytes]).stream().pipeThrough(stream));
      return new Uint8Array(await output.arrayBuffer());
    }
  }
  return inflateRawSync(bytes);
}

const LEN_BASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEN_EXTRA = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073,
  4097, 6145, 8193, 12289, 16385, 24577];
const DIST_EXTRA = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const CODE_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

function huffman(lengths) {
  const counts = new Uint16Array(16);
  for (const length of lengths) counts[length]++;
  counts[0] = 0;
  const offsets = new Uint16Array(16);
  for (let len = 1; len < 16; len++) offsets[len] = offsets[len - 1] + counts[len - 1];
  const symbols = new Uint16Array(lengths.length);
  for (let symbol = 0; symbol < lengths.length; symbol++) {
    if (lengths[symbol]) symbols[offsets[lengths[symbol]]++] = symbol;
  }
  return { counts, symbols };
}

let FIXED = null;
function fixedTables() {
  if (!FIXED) {
    const lit = new Uint8Array(288);
    lit.fill(8, 0, 144); lit.fill(9, 144, 256); lit.fill(7, 256, 280); lit.fill(8, 280, 288);
    FIXED = { lit: huffman(lit), dist: huffman(new Uint8Array(30).fill(5)) };
  }
  return FIXED;
}

/** Raw DEFLATE decoder (canonical Huffman decoding as in Mark Adler's puff.c). */
export function inflateRawSync(input) {
  const src = input instanceof Uint8Array ? input : new Uint8Array(input);
  let pos = 0;
  let bitBuffer = 0;
  let bitCount = 0;
  let out = new Uint8Array(Math.max(1024, src.length * 4));
  let length = 0;
  const need = (n) => {
    if (length + n <= out.length) return;
    let size = out.length * 2;
    while (size < length + n) size *= 2;
    const grown = new Uint8Array(size);
    grown.set(out.subarray(0, length));
    out = grown;
  };
  const bits = (n) => {
    while (bitCount < n) {
      if (pos >= src.length) throw new Error('inflate: the data ends early');
      bitBuffer |= src[pos++] << bitCount;
      bitCount += 8;
    }
    const value = bitBuffer & ((1 << n) - 1);
    bitBuffer >>>= n;
    bitCount -= n;
    return value;
  };
  const decode = (table) => {
    let code = 0;
    let first = 0;
    let index = 0;
    for (let len = 1; len < 16; len++) {
      code |= bits(1);
      const count = table.counts[len];
      if (code - first < count) return table.symbols[index + code - first];
      index += count;
      first = (first + count) << 1;
      code <<= 1;
    }
    throw new Error('inflate: bad Huffman code');
  };
  const codes = (lit, dist) => {
    for (;;) {
      const symbol = decode(lit);
      if (symbol < 256) { need(1); out[length++] = symbol; continue; }
      if (symbol === 256) return;
      const li = symbol - 257;
      if (li >= 29) throw new Error('inflate: bad length code');
      const len = LEN_BASE[li] + bits(LEN_EXTRA[li]);
      const di = decode(dist);
      if (di >= 30) throw new Error('inflate: bad distance code');
      const distance = DIST_BASE[di] + bits(DIST_EXTRA[di]);
      if (distance > length) throw new Error('inflate: distance too far back');
      need(len);
      for (let i = 0; i < len; i++) { out[length] = out[length - distance]; length++; }
    }
  };
  let last = 0;
  while (!last) {
    last = bits(1);
    const type = bits(2);
    if (type === 0) {
      bitBuffer = 0;
      bitCount = 0;
      if (pos + 4 > src.length) throw new Error('inflate: the data ends early');
      const len = src[pos] | (src[pos + 1] << 8);
      const nlen = src[pos + 2] | (src[pos + 3] << 8);
      pos += 4;
      if ((len ^ 0xffff) !== nlen) throw new Error('inflate: stored block length check failed');
      if (pos + len > src.length) throw new Error('inflate: the data ends early');
      need(len);
      out.set(src.subarray(pos, pos + len), length);
      length += len;
      pos += len;
    } else if (type === 1) {
      const { lit, dist } = fixedTables();
      codes(lit, dist);
    } else if (type === 2) {
      const nlen = bits(5) + 257;
      const ndist = bits(5) + 1;
      const ncode = bits(4) + 4;
      const order = new Uint8Array(19);
      for (let i = 0; i < ncode; i++) order[CODE_ORDER[i]] = bits(3);
      const lencode = huffman(order);
      const lengths = new Uint8Array(nlen + ndist);
      for (let i = 0; i < nlen + ndist;) {
        const symbol = decode(lencode);
        if (symbol < 16) { lengths[i++] = symbol; continue; }
        let repeat;
        let value = 0;
        if (symbol === 16) {
          if (i === 0) throw new Error('inflate: repeat without a previous length');
          value = lengths[i - 1];
          repeat = 3 + bits(2);
        } else if (symbol === 17) {
          repeat = 3 + bits(3);
        } else {
          repeat = 11 + bits(7);
        }
        if (i + repeat > nlen + ndist) throw new Error('inflate: too many lengths');
        while (repeat--) lengths[i++] = value;
      }
      if (!lengths[256]) throw new Error('inflate: no end-of-block code');
      codes(huffman(lengths.subarray(0, nlen)), huffman(lengths.subarray(nlen)));
    } else {
      throw new Error('inflate: invalid block type');
    }
  }
  return out.slice(0, length);
}

// ============================================================================ JSON arrays

/** The API's JSON array form {dtype, shape, data: nested lists} as {dtype, shape, data: flat}. A
 * numeric array becomes a Float64Array with null as NaN; strings and booleans stay plain arrays. */
export function decodeArrayDoc(doc) {
  if (!doc) return null;
  const flat = [];
  const walk = (value) => {
    if (Array.isArray(value)) { for (const item of value) walk(item); } else { flat.push(value); }
  };
  walk(doc.data);
  const numeric = doc.dtype !== 'str' && doc.dtype !== 'bool' && doc.dtype !== 'bytes';
  if (!numeric) return { dtype: doc.dtype, shape: doc.shape || [], data: flat };
  const data = new Float64Array(flat.length);
  for (let i = 0; i < flat.length; i++) data[i] = flat[i] === null || flat[i] === undefined ? NaN : Number(flat[i]);
  return { dtype: doc.dtype, shape: doc.shape || [], data };
}
