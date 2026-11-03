/* .npz / .npy reader for the run files the view reads as binary (keyframe fields, chips, final
 * brick attributes): a zip of .npy members, stored or deflated (numpy.savez, the run's
 * deterministic writer), zip64 sizes and offsets understood. Arrays come back in C order as
 * {dtype, shape, data}: float64 -> Float64Array, float32 and float16 -> Float32Array, 8-byte integers
 * -> Float64Array (exact to 2^53), smaller integers in their own typed array, bool -> Uint8Array,
 * unicode and byte strings -> Array of strings.
 *
 * Deflated members are inflated with the platform's DecompressionStream ('deflate-raw', or the member
 * framed as gzip where only 'gzip' is offered: Chromium 80, Safari 16.4, Firefox 113, node 18 and
 * later); a browser without DecompressionStream gets an error naming the member. */

const TYPES = {
  f8: [8, (v, o, le) => v.getFloat64(o, le), Float64Array, 'float64'],
  f4: [4, (v, o, le) => v.getFloat32(o, le), Float32Array, 'float32'],
  f2: [2, (v, o, le) => halfToFloat(v.getUint16(o, le)), Float32Array, 'float16'],
  i8: [8, (v, o, le) => Number(v.getBigInt64(o, le)), Float64Array, 'int64'],
  u8: [8, (v, o, le) => Number(v.getBigUint64(o, le)), Float64Array, 'uint64'],
  i4: [4, (v, o, le) => v.getInt32(o, le), Int32Array, 'int32'],
  u4: [4, (v, o, le) => v.getUint32(o, le), Uint32Array, 'uint32'],
  i2: [2, (v, o, le) => v.getInt16(o, le), Int16Array, 'int16'],
  u2: [2, (v, o, le) => v.getUint16(o, le), Uint16Array, 'uint16'],
  i1: [1, (v, o) => v.getInt8(o), Int8Array, 'int8'],
  u1: [1, (v, o) => v.getUint8(o), Uint8Array, 'uint8'],
  b1: [1, (v, o) => v.getUint8(o), Uint8Array, 'bool'],
};

/** IEEE 754 binary16 to a JavaScript number. */
export function halfToFloat(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function parseHeader(text) {
  const descr = /'descr'\s*:\s*'([^']+)'/.exec(text);
  const fortran = /'fortran_order'\s*:\s*(True|False)/.exec(text);
  const shape = /'shape'\s*:\s*\(([^)]*)\)/.exec(text);
  if (!descr || !fortran || !shape) throw new Error(`not an .npy header: ${text.slice(0, 60)}`);
  const dims = shape[1].split(',').map((s) => s.trim()).filter((s) => s.length).map(Number);
  return { descr: descr[1], fortran: fortran[1] === 'True', shape: dims };
}

function toCOrder(data, shape) {
  if (shape.length < 2) return data;
  const out = new data.constructor(data.length);
  const nd = shape.length;
  const stride = new Array(nd);
  stride[0] = 1;
  for (let d = 1; d < nd; d++) stride[d] = stride[d - 1] * shape[d - 1];
  const index = new Array(nd).fill(0);
  for (let c = 0; c < data.length; c++) {
    let f = 0;
    for (let d = 0; d < nd; d++) f += index[d] * stride[d];
    out[c] = data[f];
    for (let d = nd - 1; d >= 0; d--) {
      index[d]++;
      if (index[d] < shape[d]) break;
      index[d] = 0;
    }
  }
  return out;
}

/** One .npy file (Uint8Array) -> {dtype, shape, data}. */
export function parseNpy(input) {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  if (bytes[0] !== 0x93 || String.fromCharCode(...bytes.subarray(1, 6)) !== 'NUMPY') throw new Error('not an .npy file');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const major = bytes[6];
  const headerLength = major >= 2 ? view.getUint32(8, true) : view.getUint16(8, true);
  const headerStart = major >= 2 ? 12 : 10;
  const header = parseHeader(new TextDecoder('latin1').decode(bytes.subarray(headerStart, headerStart + headerLength)));
  const start = headerStart + headerLength;
  const count = header.shape.reduce((a, b) => a * b, 1);
  const match = /^([<>|=]?)([a-zA-Z])(\d+)$/.exec(header.descr);
  if (!match) throw new Error(`unsupported dtype ${header.descr}`);
  const little = match[1] !== '>';
  const kind = match[2];
  const size = Number(match[3]);
  let data;
  let dtype;
  if (kind === 'U') {
    data = new Array(count);
    for (let i = 0; i < count; i++) {
      let s = '';
      for (let c = 0; c < size; c++) {
        const code = view.getUint32(start + (i * size + c) * 4, little);
        if (code === 0) break;
        s += String.fromCodePoint(code);
      }
      data[i] = s;
    }
    dtype = 'str';
  } else if (kind === 'S') {
    const decoder = new TextDecoder('utf-8');
    data = new Array(count);
    for (let i = 0; i < count; i++) {
      const raw = bytes.subarray(start + i * size, start + (i + 1) * size);
      const end = raw.indexOf(0);
      data[i] = decoder.decode(end >= 0 ? raw.subarray(0, end) : raw);
    }
    dtype = 'bytes';
  } else {
    const type = TYPES[`${kind}${size}`];
    if (!type) throw new Error(`unsupported dtype ${header.descr}`);
    const [itemSize, read, ArrayType, name] = type;
    data = new ArrayType(count);
    for (let i = 0; i < count; i++) data[i] = read(view, start + i * itemSize, little);
    dtype = name;
  }
  if (header.fortran && !Array.isArray(data)) data = toCOrder(data, header.shape);
  return { dtype, shape: header.shape, data };
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
    const crc32 = view.getUint32(offset + 16, true);
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
        if (local === 0xffffffff) local = Number(view.getBigUint64(p, true));
      }
      extra += 4 + length;
    }
    entries.push({ name, method, crc32, compressed, size, local });
    offset = extraEnd + commentLength;
  }
  return entries;
}

/** Inflate a deflated zip member: DecompressionStream('deflate-raw') where the platform has it, else the
 * same bytes framed as a gzip member (RFC 1952: a 10-byte header, the raw DEFLATE data, then the CRC-32
 * and size the zip entry records), which every DecompressionStream accepts. */
async function inflateRaw(raw, entry) {
  if (typeof DecompressionStream !== 'function') {
    throw new Error(`${entry.name} is deflated and this browser has no DecompressionStream to inflate it`);
  }
  let stream = null;
  try { stream = new DecompressionStream('deflate-raw'); } catch { stream = null; }
  let input = raw;
  if (!stream) {
    const framed = new Uint8Array(10 + raw.length + 8);
    framed.set([0x1f, 0x8b, 0x08, 0, 0, 0, 0, 0, 0, 0xff], 0);
    framed.set(raw, 10);
    const tail = new DataView(framed.buffer, 10 + raw.length, 8);
    tail.setUint32(0, entry.crc32 >>> 0, true);
    tail.setUint32(4, entry.size >>> 0, true);
    input = framed;
    stream = new DecompressionStream('gzip');
  }
  const out = new Response(new Blob([input]).stream().pipeThrough(stream));
  return new Uint8Array(await out.arrayBuffer());
}

async function memberBytes(bytes, entry) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(entry.local, true) !== 0x04034b50) throw new Error(`zip: bad local header of ${entry.name}`);
  const start = entry.local + 30 + view.getUint16(entry.local + 26, true) + view.getUint16(entry.local + 28, true);
  const raw = bytes.subarray(start, start + entry.compressed);
  if (entry.method === 0) return raw;
  if (entry.method === 8) return inflateRaw(raw, entry);
  throw new Error(`zip: ${entry.name} uses compression method ${entry.method}`);
}

/** An .npz (ArrayBuffer or Uint8Array) -> {name: {dtype, shape, data}}. `names` limits the members read. */
export async function parseNpz(input, names = null) {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  const wanted = names ? new Set(names) : null;
  const out = {};
  for (const entry of zipEntries(bytes)) {
    if (!entry.name.endsWith('.npy')) continue;
    const name = entry.name.slice(0, -4);
    if (wanted && !wanted.has(name)) continue;
    out[name] = parseNpy(await memberBytes(bytes, entry));
  }
  return out;
}

/** A scalar of an array document (first element), or null. */
export function scalarOf(array) {
  if (!array || !array.data || !array.data.length) return null;
  const value = Number(array.data[0]);
  return Number.isFinite(value) ? value : null;
}
