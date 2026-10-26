// 3D models (STL / OBJ / 3MF / STEP / IGES) as the job's target part: reading, placement in
// the part frame, and what Claude gets about it — measured features and how far each
// operation's result is from it — as text.
import { VoxelSim, IDENTITY, mulXform, rotX, rotY, rotZ, translate, applyX, invertX, snapX } from '../sim/voxelsim.js';
import { partSummary, materialField, profileSummary, boundsUnder } from '../sim/inspect.js';
import { programThreads, tapDrillOf, threadSource, holeAt } from '../sim/threads.js';

const fx = (v, d = 2) => { const s = (Math.abs(v) < 0.5 * 10 ** -d ? 0 : v).toFixed(d); return s.includes('.') ? s.replace(/\.?0+$/, '') || '0' : s; };

export const MODEL_EXTS = ['stl', 'obj', '3mf', 'step', 'stp', 'iges', 'igs', 'brep'];
export const isModelFile = (name) => MODEL_EXTS.includes(String(name).split('.').pop().toLowerCase());

// ---------------------------------------------------------------- readers
export async function readModel(name, bytes) {
  const ext = String(name).split('.').pop().toLowerCase();
  let tris, warnings = [];
  if (ext === 'stl') tris = parseSTL(bytes);
  else if (ext === 'obj') tris = parseOBJ(new TextDecoder().decode(bytes));
  else if (ext === '3mf') ({ tris, warnings } = await parse3MF(bytes));
  else if (['step', 'stp', 'iges', 'igs', 'brep'].includes(ext)) tris = meshesToTris(await occtRead(ext === 'stp' ? 'step' : ext === 'igs' ? 'iges' : ext, bytes));
  else throw new Error(`Unsupported model format .${ext}`);
  if (!tris || tris.length < 9) throw new Error('No triangles found in the model');
  return { name, format: ext === 'stp' ? 'step' : ext === 'igs' ? 'iges' : ext, tris, triCount: tris.length / 9, bounds: trisBounds(tris), warnings };
}

export function parseSTL(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.length >= 84) {
    const n = dv.getUint32(80, true);
    if (84 + n * 50 === bytes.length) {
      const out = new Float32Array(n * 9);
      for (let t = 0; t < n; t++) for (let k = 0; k < 9; k++) out[t * 9 + k] = dv.getFloat32(84 + t * 50 + 12 + k * 4, true);
      return out;
    }
  }
  const text = new TextDecoder().decode(bytes);
  const v = [];
  const re = /vertex\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)/g;
  let m;
  while ((m = re.exec(text))) v.push(+m[1], +m[2], +m[3]);
  return new Float32Array(v.length - (v.length % 9)).map((_, i) => v[i]);
}

export function parseOBJ(text) {
  const vs = [], out = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (line.startsWith('v ')) { const p = line.split(/\s+/); vs.push([+p[1], +p[2], +p[3]]); }
    else if (line.startsWith('f ')) {
      const idx = line.split(/\s+/).slice(1).map((tok) => { const i = parseInt(tok.split('/')[0], 10); return i < 0 ? vs.length + i : i - 1; });
      for (let k = 1; k + 1 < idx.length; k++) for (const q of [idx[0], idx[k], idx[k + 1]]) out.push(...(vs[q] || [0, 0, 0]));
    }
  }
  return new Float32Array(out);
}

// minimal zip reader (stored / deflate entries)
export async function unzip(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let eocd = -1;
  for (let i = bytes.length - 22; i >= Math.max(0, bytes.length - 65557); i--) if (dv.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  if (eocd < 0) throw new Error('Not a zip / 3MF file');
  const count = dv.getUint16(eocd + 10, true);
  let p = dv.getUint32(eocd + 16, true);
  const files = new Map();
  for (let e = 0; e < count; e++) {
    if (dv.getUint32(p, true) !== 0x02014b50) break;
    const method = dv.getUint16(p + 10, true), csize = dv.getUint32(p + 20, true);
    const nlen = dv.getUint16(p + 28, true), xlen = dv.getUint16(p + 30, true), clen = dv.getUint16(p + 32, true);
    const local = dv.getUint32(p + 42, true);
    const name = new TextDecoder().decode(bytes.subarray(p + 46, p + 46 + nlen));
    files.set(name, { method, csize, local });
    p += 46 + nlen + xlen + clen;
  }
  const read = async (name) => {
    const f = files.get(name);
    if (!f) return null;
    const start = f.local + 30 + dv.getUint16(f.local + 26, true) + dv.getUint16(f.local + 28, true);
    const data = bytes.subarray(start, start + f.csize);
    if (f.method === 0) return data;
    if (f.method !== 8) throw new Error(`Unsupported zip compression ${f.method}`);
    const ds = new Blob([data]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
    return new Uint8Array(await new Response(ds).arrayBuffer());
  };
  return { names: [...files.keys()], read };
}

export async function parse3MF(bytes) {
  const z = await unzip(bytes);
  const name = z.names.find((n) => /^3D\/.*\.model$/i.test(n)) || z.names.find((n) => /\.model$/i.test(n));
  if (!name) throw new Error('No 3D model inside the 3MF file');
  const xml = new TextDecoder().decode(await z.read(name));
  const unit = /<model[^>]*\bunit="(\w+)"/.exec(xml)?.[1] || 'millimeter';
  const scale = { micron: 0.001, millimeter: 1, centimeter: 10, inch: 25.4, foot: 304.8, meter: 1000 }[unit] ?? 1;
  const attr = (tag, a) => { const m = new RegExp(`\\b${a}="([^"]*)"`).exec(tag); return m ? m[1] : null; };
  const objects = new Map();
  for (const om of xml.matchAll(/<object\b([^>]*)>([\s\S]*?)<\/object>/g)) {
    const id = attr(om[1], 'id');
    const body = om[2];
    const verts = [...body.matchAll(/<vertex\b([^>]*)\/>/g)].map((m) => [+attr(m[1], 'x'), +attr(m[1], 'y'), +attr(m[1], 'z')]);
    const trisI = [...body.matchAll(/<triangle\b([^>]*)\/>/g)].map((m) => [+attr(m[1], 'v1'), +attr(m[1], 'v2'), +attr(m[1], 'v3')]);
    const comps = [...body.matchAll(/<component\b([^>]*)\/>/g)].map((m) => ({ id: attr(m[1], 'objectid'), T: parseT(attr(m[1], 'transform')) }));
    objects.set(id, { verts, tris: trisI, comps });
  }
  const out = [];
  const emit = (id, T, depth = 0) => {
    const o = objects.get(id);
    if (!o || depth > 8) return;
    for (const [a, b, c] of o.tris) for (const q of [a, b, c]) { const v = o.verts[q] || [0, 0, 0]; const p = applyT(T, v); out.push(p[0] * scale, p[1] * scale, p[2] * scale); }
    for (const c of o.comps) emit(c.id, mulT(T, c.T), depth + 1);
  };
  const items = [...xml.matchAll(/<item\b([^>]*)\/>/g)].map((m) => ({ id: attr(m[1], 'objectid'), T: parseT(attr(m[1], 'transform')) }));
  for (const it of items.length ? items : [...objects.keys()].map((id) => ({ id, T: null }))) emit(it.id, it.T);
  return { tris: new Float32Array(out), warnings: unit !== 'millimeter' ? [`3MF unit ${unit} converted to mm`] : [] };
}
// 3MF transforms: "m00 m01 m02 m10 m11 m12 m20 m21 m22 m30 m31 m32" (row vectors)
function parseT(s) { if (!s) return null; const v = s.trim().split(/\s+/).map(Number); return v.length === 12 ? v : null; }
function applyT(T, p) { if (!T) return p; return [p[0] * T[0] + p[1] * T[3] + p[2] * T[6] + T[9], p[0] * T[1] + p[1] * T[4] + p[2] * T[7] + T[10], p[0] * T[2] + p[1] * T[5] + p[2] * T[8] + T[11]]; }
function mulT(A, B) {
  if (!A) return B; if (!B) return A;
  // child B applied first, then A
  const o = new Array(12);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 3; c++) {
    const row = r < 3 ? [B[r * 3], B[r * 3 + 1], B[r * 3 + 2], 0] : [B[9], B[10], B[11], 1];
    o[r * 3 + c] = row[0] * A[c] + row[1] * A[3 + c] + row[2] * A[6 + c] + row[3] * (r === 3 ? A[9 + c] : 0);
  }
  return o;
}

// STEP / IGES / BREP through OpenCascade (occt-import-js, WebAssembly) in a worker
let occtWorker = null, occtSeq = 0;
const occtWaiting = new Map();
function occtRead(format, bytes) {
  if (typeof Worker === 'undefined') throw new Error('STEP/IGES import needs the browser (WebAssembly worker)');
  if (!occtWorker) {
    occtWorker = new Worker('/js/model/occt-worker.js');
    occtWorker.onmessage = (e) => { const w = occtWaiting.get(e.data.id); occtWaiting.delete(e.data.id); if (e.data.error) w?.reject(new Error(e.data.error)); else w?.resolve(e.data.result); };
    occtWorker.onerror = (e) => { for (const w of occtWaiting.values()) w.reject(new Error(e.message || 'STEP reader failed to load')); occtWaiting.clear(); occtWorker = null; };
  }
  const id = ++occtSeq;
  return new Promise((resolve, reject) => {
    occtWaiting.set(id, { resolve, reject });
    const buf = bytes.slice().buffer;
    occtWorker.postMessage({ id, format, buffer: buf, params: { linearUnit: 'millimeter', linearDeflectionType: 'absolute_value', linearDeflection: 0.01, angularDeflection: 0.15 } }, [buf]);
  });
}

export function meshesToTris(result) {
  if (!result?.success) throw new Error('The CAD file could not be read');
  let n = 0;
  for (const m of result.meshes) n += m.index.array.length;
  const out = new Float32Array(n * 3);
  let o = 0;
  for (const m of result.meshes) {
    const P = m.attributes.position.array, I = m.index.array;
    for (let i = 0; i < I.length; i++) { const q = I[i] * 3; out[o++] = P[q]; out[o++] = P[q + 1]; out[o++] = P[q + 2]; }
  }
  return out;
}

export function trisBounds(tris) {
  const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < tris.length; i += 3) for (let k = 0; k < 3; k++) { const v = tris[i + k]; if (v < mn[k]) mn[k] = v; if (v > mx[k]) mx[k] = v; }
  return { min: mn, max: mx };
}

export function meshStats(tris) {
  let vol = 0, area = 0;
  for (let t = 0; t < tris.length; t += 9) {
    const ax = tris[t], ay = tris[t + 1], az = tris[t + 2], bx = tris[t + 3], by = tris[t + 4], bz = tris[t + 5], cx = tris[t + 6], cy = tris[t + 7], cz = tris[t + 8];
    vol += (ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)) / 6;
    const ux = bx - ax, uy = by - ay, uz = bz - az, vx = cx - ax, vy = cy - ay, vz = cz - az;
    area += Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx) / 2;
  }
  return { volume: Math.abs(vol), area };
}

// ---------------------------------------------------------------- placement
// the model's axis that becomes part +Z (mill: up; lathe: the spindle axis, front face toward +Z)
const UP = { '+Z': IDENTITY, '-Z': rotX(180), '+Y': rotX(-90), '-Y': rotX(90), '+X': rotY(90), '-X': rotY(-90) };

// round symmetric about which bbox axis? (lathe parts)
export function guessTurnAxis(b) {
  const s = [0, 1, 2].map((k) => b.max[k] - b.min[k]);
  const cand = [0, 1, 2].filter((k) => { const o = [0, 1, 2].filter((q) => q !== k); return Math.abs(s[o[0]] - s[o[1]]) <= 0.01 * Math.max(s[o[0]], s[o[1]]) + 1e-6; });
  if (!cand.length) return null;
  return 'XYZ'[cand.sort((a, b) => s[b] - s[a])[0]];
}

export function defaultPlacement(model, machine) {
  if (machine === 'lathe') return { up: `+${guessTurnAxis(model.bounds) || 'Z'}`, rotZ: 0, align: 'lathe-front', offset: [0, 0, 0], scale: 1 };
  return { up: '+Z', rotZ: 0, align: 'top-center', offset: [0, 0, 0], scale: 1 };
}

// model -> part frame (the first operation's work coordinates)
export function placementX(model, pl) {
  let R = mulXform(rotZ(pl.rotZ || 0), UP[pl.up] || IDENTITY);
  const sc = pl.scale || 1;
  const b = model.bounds;
  const sb = { min: b.min.map((v) => v * sc), max: b.max.map((v) => v * sc) };
  const rb = boundsUnder(R, sb);
  let zero = [0, 0, 0];
  if (pl.align === 'top-center') zero = [(rb.min[0] + rb.max[0]) / 2, (rb.min[1] + rb.max[1]) / 2, rb.max[2]];
  else if (pl.align === 'top-corner') zero = [rb.min[0], rb.min[1], rb.max[2]];
  else if (pl.align === 'lathe-front') zero = [(rb.min[0] + rb.max[0]) / 2, (rb.min[1] + rb.max[1]) / 2, rb.max[2]];
  const off = pl.offset || [0, 0, 0];
  return snapX(mulXform(translate([off[0] - zero[0], off[1] - zero[1], off[2] - zero[2]]), R));
}

export function describePlacement(pl) {
  const al = { 'top-center': 'top face centred on X0 Y0 Z0', 'top-corner': 'top face, −X−Y corner at X0 Y0 Z0', 'lathe-front': 'turning axis on the spindle axis, front face at Z0', model: 'model coordinates kept' }[pl.align] || pl.align;
  const off = pl.offset && pl.offset.some((v) => v) ? `, then moved X${fx(pl.offset[0])} Y${fx(pl.offset[1])} Z${fx(pl.offset[2])}` : '';
  return `model ${pl.up} ${pl.align === 'lathe-front' ? 'along the spindle axis' : 'up'}${pl.rotZ ? `, turned ${pl.rotZ}° about Z` : ''}, ${al}${off}${pl.scale && pl.scale !== 1 ? `, scaled ×${pl.scale}` : ''}`;
}

// target = the placed model as a solid in the part frame
export function buildTarget(model, pl, { budget = 4e6 } = {}) {
  const X = placementX(model, pl);
  const sc = pl.scale || 1;
  const src = model.tris;
  const tris = new Float32Array(src.length);
  for (let i = 0; i < src.length; i += 3) {
    const p = applyX(X, [src[i] * sc, src[i + 1] * sc, src[i + 2] * sc]);
    tris[i] = p[0]; tris[i + 1] = p[1]; tris[i + 2] = p[2];
  }
  const sim = VoxelSim.fromMesh(tris, {}, { budget });
  const st = meshStats(tris);
  return { name: model.name, format: model.format, placement: pl, X, tris, bounds: trisBounds(tris), sim, volume: st.volume, area: st.area, triCount: model.triCount };
}

// ---------------------------------------------------------------- what Claude gets
// out: the operation whose work frame to describe the target in (its workFromPart)
export function targetSummary(target, out) {
  const key = out ? `${out.key}|${out.op.machine}` : 'part';
  target.summaries ||= new Map();
  if (target.summaries.has(key)) return target.summaries.get(key);
  const W = out?.workFromPart || IDENTITY;
  const wb = boundsUnder(W, target.bounds);
  const o = [];
  o.push(`Target part: ${target.name} (${target.format.toUpperCase()}, ${target.triCount} triangles) — ${fx(target.bounds.max[0] - target.bounds.min[0])} × ${fx(target.bounds.max[1] - target.bounds.min[1])} × ${fx(target.bounds.max[2] - target.bounds.min[2])} mm in the part frame, volume ${fx(target.volume / 1000)} cm³.`);
  o.push(`Placement: ${describePlacement(target.placement)}.`);
  if (out) o.push(`In ${out.name}'s work coordinates the target spans X${fx(wb.min[0])}..${fx(wb.max[0])} Y${fx(wb.min[1])}..${fx(wb.max[1])} Z${fx(wb.min[2])}..${fx(wb.max[2])}.`);
  if (out?.op.machine === 'lathe') o.push(latheTargetProfile(target, W));
  else {
    const pseudo = { engine: 'voxel', final: target.sim, X: W, key: `target:${key}` };
    o.push(partSummary(pseudo).replace(/^Part after this operation \(work coordinates of this operation\)/, 'Target'));
  }
  const txt = o.join('\n');
  target.summaries.set(key, txt);
  return txt;
}

// turned profile of a target along the work Z axis (W = part -> work), exact from the mesh:
// cut it with the half-plane Y=0, X>0 and read where the outline crosses each Z
function latheProfileArrays(target, W) {
  const src = target.tris;
  const segs = [];
  const P = (k) => applyX(W, [src[k], src[k + 1], src[k + 2]]);
  for (let t = 0; t < src.length; t += 9) {
    const a = P(t), b = P(t + 3), c = P(t + 6);
    const pts = [];
    for (const [u, v] of [[a, b], [b, c], [c, a]]) {
      if ((u[1] > 0) === (v[1] > 0)) continue;
      const f = u[1] / (u[1] - v[1]);
      const x = u[0] + (v[0] - u[0]) * f, z = u[2] + (v[2] - u[2]) * f;
      if (x >= -1e-9) pts.push([z, Math.max(0, x)]);
    }
    if (pts.length === 2) segs.push(pts);
  }
  const wb = boundsUnder(W, target.bounds);
  const zMin = wb.min[2], zMax = wb.max[2];
  const n = Math.max(2, Math.min(6000, Math.round((zMax - zMin) / 0.025) + 1));
  const dz = (zMax - zMin) / (n - 1);
  const cols = Array.from({ length: n }, () => []);
  for (const [[za, xa], [zb, xb]] of segs) {
    if (Math.abs(zb - za) < 1e-12) continue;
    const i0 = Math.max(0, Math.ceil((Math.min(za, zb) - zMin) / dz - 1e-9)), i1 = Math.min(n - 1, Math.floor((Math.max(za, zb) - zMin) / dz + 1e-9));
    for (let i = i0; i <= i1; i++) {
      const z = zMin + i * dz;
      cols[i].push(xa + ((xb - xa) * (z - za)) / (zb - za));
    }
  }
  const outer = new Float32Array(n), inner = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const xs = cols[i].sort((p, q) => p - q);
    const u = [];
    for (const x of xs) if (!u.length || x - u[u.length - 1] > 1e-6) u.push(x);
    if (!u.length) continue;
    outer[i] = u[u.length - 1];
    inner[i] = u.length % 2 === 1 ? 0 : u[0]; // odd: the axis is inside the material
  }
  return { n, zMin, zMax, dz, outer, inner };
}

function profileFor(target, W) {
  target.profiles ||= new Map();
  const key = JSON.stringify([W.R, W.t]);
  let p = target.profiles.get(key);
  if (!p) { p = latheProfileArrays(target, W); target.profiles.set(key, p); }
  return p;
}

function latheTargetProfile(target, W) {
  const p = profileFor(target, W);
  const stub = {
    ...p, zChuck: null, partedOff: null,
    materialBounds() {
      let R = 0, z0 = Infinity, z1 = -Infinity;
      for (let i = 0; i < p.n; i++) if (p.outer[i] - p.inner[i] > 1e-4) { R = Math.max(R, p.outer[i]); const z = p.zMin + i * p.dz; z0 = Math.min(z0, z); z1 = Math.max(z1, z); }
      return R > 0 ? { min: [-R, -R, z0], max: [R, R, z1] } : null;
    },
  };
  return profileSummary(stub).replace(/; the chuck jaws hold it from Z[^.]*\./, '.').replace(/^Part:/, 'Target (turned profile):');
}

// how far an operation's result is from the target: excess (still to remove) and gouges
export function compareTarget(out, target) {
  target.compares ||= new Map();
  if (target.compares.has(out.key)) return target.compares.get(out.key);
  const txt = out.engine === 'lathe' ? compareLathe(out, target) : compareMill(out, target);
  target.compares.set(out.key, txt);
  return txt;
}

function compareMill(out, target) {
  const field = materialField(out);
  const W = out.workFromPart || IDENTITY;
  const inv = invertX(W);
  const tsim = target.sim;
  const tAt = (p) => tsim.sample(...applyX(inv, p)); // target distance (mm, + outside), work point
  const tb = boundsUnder(W, target.bounds);
  const sb = field.bounds || tb;
  const b = { min: [0, 1, 2].map((k) => Math.min(tb.min[k], sb.min[k])), max: [0, 1, 2].map((k) => Math.max(tb.max[k], sb.max[k])) };
  const size = [0, 1, 2].map((k) => b.max[k] - b.min[k]);
  const g = Math.max(tsim.h * 0.9, field.step * 0.9, Math.max(...size) / 240);
  const n = size.map((s) => Math.max(2, Math.ceil(s / g) + 1));
  const tol = Math.max(0.05, tsim.h * 0.6, field.step * (field.kind === 'height' ? 0.6 : 0.6));
  // tapped / thread-milled holes are judged as threads below; the simulation cuts a plain bore
  const threads = programThreads(out, field);
  const zones = threads.map((th) => ({ x: th.x, y: th.y, r2: (th.d / 2 + tol + g) ** 2, z0: th.zTip - g, z1: th.zTop + g }));
  const inZone = (p) => zones.some((zn) => p[2] >= zn.z0 && p[2] <= zn.z1 && (p[0] - zn.x) ** 2 + (p[1] - zn.y) ** 2 <= zn.r2);
  const lab = new Int8Array(n[0] * n[1] * n[2]);
  const sIn = new Uint8Array(lab.length), tIn = new Uint8Array(lab.length);
  // fill fractions of each sample cell: stock minus target = excess, target minus stock = gouge
  const exF = new Float32Array(lab.length), goF = new Float32Array(lab.length);
  const sd = field.distance;
  const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
  let ex = 0, go = 0;
  for (let k = 0; k < n[2]; k++) for (let j = 0; j < n[1]; j++) for (let i = 0; i < n[0]; i++) {
    const p = [b.min[0] + i * g, b.min[1] + j * g, b.min[2] + k * g];
    const td = tAt(p);
    const s = field.inside(p);
    const q = (k * n[1] + j) * n[0] + i;
    sIn[q] = s ? 1 : 0; tIn[q] = td < 0 ? 1 : 0;
    if (zones.length && inZone(p)) continue;
    const fs = sd ? clamp01(0.5 - sd(p) / g) : s ? 1 : 0, ft = clamp01(0.5 - td / g);
    exF[q] = Math.max(0, fs - ft); goF[q] = Math.max(0, ft - fs);
    if (s && td > tol) { lab[q] = 1; ex++; } else if (!s && td < -tol) { lab[q] = 2; go++; }
  }
  // a sample on a convex edge of the material it is in (air next to it along two axes): both
  // distance fields round such edges by about a cell, so regions made only of these are slivers
  const onEdge = (c, inside) => {
    const i = c % n[0], j = ((c - i) / n[0]) % n[1], k = Math.floor(c / (n[0] * n[1]));
    const idx = [i, j, k], st = [1, n[0], n[0] * n[1]];
    let axes = 0;
    for (let a = 0; a < 3; a++) {
      const lo = idx[a] === 0 || !inside[c - st[a]], hi = idx[a] === n[a] - 1 || !inside[c + st[a]];
      if (lo || hi) axes++;
    }
    return axes >= 2;
  };
  const regions = [];
  const seen = new Uint8Array(lab.length);
  for (let q = 0; q < lab.length; q++) {
    if (!lab[q] || seen[q]) continue;
    const L = lab[q];
    const stack = [q];
    seen[q] = 1;
    let cnt = 0, deep = 0, solid = 0;
    const F = L === 1 ? exF : goF;
    const own = L === 1 ? sIn : tIn;
    const ring = new Set();
    const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
    while (stack.length) {
      const c = stack.pop();
      cnt++;
      if (!onEdge(c, own)) solid++;
      const i = c % n[0], j = ((c - i) / n[0]) % n[1], k = Math.floor(c / (n[0] * n[1]));
      const p = [b.min[0] + i * g, b.min[1] + j * g, b.min[2] + k * g];
      for (let a = 0; a < 3; a++) { mn[a] = Math.min(mn[a], p[a]); mx[a] = Math.max(mx[a], p[a]); }
      deep = Math.max(deep, Math.abs(tAt(p)));
      for (const [di, dj, dk] of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) {
        const a = i + di, bb = j + dj, cc = k + dk;
        if (a < 0 || bb < 0 || cc < 0 || a >= n[0] || bb >= n[1] || cc >= n[2]) continue;
        const r = (cc * n[1] + bb) * n[0] + a;
        if (!seen[r] && lab[r] === L) { seen[r] = 1; stack.push(r); } else if (lab[r] !== L) ring.add(r);
      }
    }
    if (cnt < 2 || solid < Math.max(2, 0.25 * cnt)) continue;
    // volume: the region's cells plus the partly-off cells around it (inside the tolerance band)
    let w = 0;
    stack.length = 0;
    // (re-walk the region: cells with this label reachable from q)
    const walk = [q], vis = new Set([q]);
    while (walk.length) {
      const c = walk.pop();
      w += F[c];
      const i = c % n[0], j = ((c - i) / n[0]) % n[1], k = Math.floor(c / (n[0] * n[1]));
      for (const [di, dj, dk] of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) {
        const a = i + di, bb = j + dj, cc = k + dk;
        if (a < 0 || bb < 0 || cc < 0 || a >= n[0] || bb >= n[1] || cc >= n[2]) continue;
        const r = (cc * n[1] + bb) * n[0] + a;
        if (lab[r] === L && !vis.has(r)) { vis.add(r); walk.push(r); }
      }
    }
    for (const r of ring) w += F[r];
    regions.push({ kind: L === 1 ? 'excess' : 'gouge', vol: w * g ** 3, mn, mx, deep, clipped: deep >= tsim.S * 0.98 });
  }
  regions.sort((a, b) => b.vol - a.vol);
  const exR = regions.filter((r) => r.kind === 'excess'), goR = regions.filter((r) => r.kind === 'gouge');
  const o = [];
  const box = (r) => `X${fx(r.mn[0] - g / 2)}..${fx(r.mx[0] + g / 2)} Y${fx(r.mn[1] - g / 2)}..${fx(r.mx[1] + g / 2)} Z${fx(r.mn[2] - g / 2)}..${fx(r.mx[2] + g / 2)}`;
  const checks = threads.length ? ['Threads, checked as threads (the simulation cuts a plain bore, models usually show the tap drill or a plain hole):', ...threads.map((th) => `  ${holeThreadCheck(th, target, W, tol)}`)] : [];
  if (!exR.length && !goR.length) {
    o.push(`Compared with the target model: matches within ±${fx(tol)} mm everywhere${zones.length ? ' outside the thread zones' : ''} (sampled every ${fx(g)} mm).`, ...checks);
    return o.join('\n');
  }
  o.push(`Compared with the target model (sampled every ${fx(g)} mm, tolerance ±${fx(tol)} mm, this operation's work coordinates${zones.length ? ', thread zones checked separately' : ''}):`);
  const vsum = (rs) => rs.reduce((s, r) => s + r.vol, 0);
  o.push(exR.length ? `  Material still to remove: ${exR.length} region${exR.length > 1 ? 's' : ''}, ${fx(vsum(exR) / 1000, 2)} cm³ in total` : '  No excess material — everything that should be removed is gone.');
  for (const r of exR.slice(0, 10)) o.push(`    ${box(r)}: ${fx(r.vol)} mm³, up to ${r.clipped ? 'more than ' : ''}${fx(r.deep)} mm beyond the target surface`);
  if (exR.length > 10) o.push(`    … ${exR.length - 10} smaller regions`);
  o.push(goR.length ? `  GOUGES — cut into the finished part: ${goR.length} region${goR.length > 1 ? 's' : ''}, ${fx(vsum(goR), 1)} mm³` : '  No gouges (nothing cut into the finished part).');
  for (const r of goR.slice(0, 10)) o.push(`    ${box(r)}: ${fx(r.vol)} mm³, up to ${r.clipped ? 'more than ' : ''}${fx(r.deep)} mm too deep`);
  o.push(...checks);
  return o.join('\n');
}

function compareLathe(out, target) {
  const s = out.final;
  const W = out.workFromPart || IDENTITY;
  const p = profileFor(target, W);
  const tol = 0.02;
  // thread zones are judged as threads below, not as an outline (the simulation cuts rings)
  const threads = programThreads(out);
  const zones = threads.filter((th) => !th.internal).map((th) => ({ lo: th.zone[0] - th.pitch, hi: th.zone[1] + th.pitch, r0: th.rootR - 0.05, r1: Math.max(th.crestR, (th.nominal || 0) / 2) + 0.05 }));
  const inZone = (z, so, to) => zones.some((zn) => z >= zn.lo && z <= zn.hi && so >= zn.r0 && so <= zn.r1 && to >= zn.r0 && to <= zn.r1);
  // target outline as (z, r) points (including the vertical faces between samples)
  const tz = (i) => p.zMin + i * p.dz;
  const at = (z) => { const u = (z - p.zMin) / p.dz; if (u < 0 || u > p.n - 1) return 0; const i = Math.min(p.n - 2, Math.floor(u)), a = u - i; return p.outer[i] * (1 - a) + p.outer[i + 1] * a; };
  const dist = (z, r) => {
    let best = Infinity;
    const w = Math.max(2, Math.ceil(1.5 / p.dz));
    const c = Math.round((z - p.zMin) / p.dz);
    for (let i = Math.max(0, c - w); i < Math.min(p.n - 1, c + w); i++) {
      const az = tz(i), ar = p.outer[i], bz = tz(i + 1), br = p.outer[i + 1];
      const dzz = bz - az, drr = br - ar, L2 = dzz * dzz + drr * drr;
      const f = L2 > 0 ? Math.max(0, Math.min(1, ((z - az) * dzz + (r - ar) * drr) / L2)) : 0;
      best = Math.min(best, Math.hypot(z - az - f * dzz, r - ar - f * drr));
    }
    return best;
  };
  const runs = [];
  let cur = null;
  const zEnd = s.zChuck ?? s.zMin;
  for (let i = s.n - 1; i >= 0; i--) {
    const z = s.zMin + i * s.dz;
    if (z < zEnd) break; // inside the chuck: not machined in this operation
    const so = s.outer[i] - s.inner[i] > 1e-4 ? s.outer[i] : 0;
    const to = at(z);
    if (so <= tol && to <= tol) { cur = null; continue; } // neither has material here
    if (inZone(z, so, to)) { cur = null; continue; }
    const d = dist(z, so);
    const kind = d <= tol ? null : so > to ? 'excess' : 'gouge';
    if (kind && cur && cur.kind === kind && cur.z1 - z < s.dz * 1.5) { cur.z1 = z; cur.max = Math.max(cur.max, d); }
    else if (kind) { cur = { kind, z0: z, z1: z, max: d, d0: 2 * to }; runs.push(cur); }
    else cur = null;
  }
  const big = runs.filter((r) => r.z0 - r.z1 >= 0.1 || r.max > 0.1);
  const besides = zones.length ? ' outside the thread zones' : '';
  const o = [];
  if (!big.length) o.push(`Compared with the target model: the outside profile from the front face to the chuck (Z${fx(zEnd)}) is within ±${tol} mm of the target outline${besides}.`);
  else {
    o.push(`Compared with the target model (outside profile${besides}, ±${tol} mm from the target outline, up to the chuck at Z${fx(zEnd)}):`);
    for (const r of big.slice(0, 14)) o.push(`  Z${fx(r.z0)} → Z${fx(r.z1)}: ${r.kind === 'excess' ? 'material still to remove' : 'GOUGE — cut under size'}, up to ${fx(r.max, 3)} mm from the target outline (target Ø${fx(r.d0, 3)} at Z${fx(r.z0)})`);
  }
  if (threads.length) o.push('Threads, checked as threads (the simulation cuts them as rings, models usually show them as a plain diameter):', ...threads.map((th) => `  ${latheThreadCheck(th, p)}`));
  return o.join('\n');
}

// ---------------------------------------------------------------- threads vs the target
// a target radius along a thread zone: plain (a cosmetic thread), a modeled thread (the radius
// rises through its middle once per pitch), or neither
function readThreadSeries(zs, rs, depth) {
  const live = rs.filter((r) => r > 1e-3);
  if (live.length < Math.max(3, rs.length / 2)) return { kind: 'none' };
  const srt = live.slice().sort((a, b) => a - b);
  const q = (f) => srt[Math.min(srt.length - 1, Math.floor(f * srt.length))];
  const lo = q(0.05), med = q(0.5), hi = q(0.95);
  if (live.filter((r) => Math.abs(r - med) <= Math.max(0.03, 0.15 * depth)).length >= 0.8 * live.length) return { kind: 'plain', r: med };
  const mid = (lo + hi) / 2, hys = 0.2 * (hi - lo);
  const ups = [];
  let state = 0;
  zs.forEach((z, k) => {
    if (rs[k] > mid + hys) { if (state < 0) ups.push(z); state = 1; } else if (rs[k] < mid - hys) state = -1;
  });
  if (ups.length >= 3) {
    const steps = ups.slice(1).map((z, k) => Math.abs(z - ups[k]));
    const pitch = Math.abs(ups[ups.length - 1] - ups[0]) / steps.length;
    if (steps.every((s) => Math.abs(s - pitch) <= 0.15 * pitch)) return { kind: 'thread', pitch, rMin: lo, rMax: hi };
  }
  return { kind: 'irregular', rMin: lo, rMax: hi };
}

const pitchesMatch = (a, b) => Math.abs(a - b) <= Math.max(0.03, 0.04 * b);
const fp = (v) => { const t = fx(v, 2); return t.includes('.') ? t : `${t}.0`; }; // pitch 1.0, not 1

function latheThreadCheck(th, p) {
  const D = (r) => `Ø${fx(2 * r, 3)}`;
  const head = `Thread ${th.name} ${th.internal ? 'internal' : 'external'} (${threadSource(th)}) Z${fx(th.full[1])}..Z${fx(th.full[0])}, ${th.internal ? `bore ${D(th.crestR)} / root ${D(th.rootR)}` : `${D(th.crestR)} / root ${D(th.rootR)}`}`;
  // judged half a lead in from both ends of the full-depth part
  const lo = th.full[1] - th.full[0] > 2 * th.pitch ? th.full[0] + th.pitch / 2 : th.full[0], hi = th.full[1] - th.full[0] > 2 * th.pitch ? th.full[1] - th.pitch / 2 : th.full[1];
  const arr = th.internal ? p.inner : p.outer;
  const zs = [], rs = [];
  for (let i = 0; i < p.n; i++) { const z = p.zMin + i * p.dz; if (z >= lo && z <= hi) { zs.push(z); rs.push(arr[i]); } }
  const depth = Math.abs(th.crestR - th.rootR);
  const tolD = Math.max(0.1, 0.1 * th.pitch);
  const m = readThreadSeries(zs, rs, depth);
  if (m.kind === 'none') return `${head}: the model has no ${th.internal ? 'bore' : 'material'} there — MISMATCH: the program cuts a thread.`;
  if (m.kind === 'irregular') return `${head}: the model's diameter varies there (Ø${fx(2 * m.rMin)}..Ø${fx(2 * m.rMax)}) without a regular pitch — MISMATCH: check the model against the program's ${th.name}.`;
  if (m.kind === 'thread') {
    const what = `the model has a modeled thread of pitch ${fp(m.pitch)} (Ø${fx(2 * m.rMax)} / ${th.internal ? 'bore' : 'root'} Ø${fx(2 * m.rMin)})`;
    if (!pitchesMatch(m.pitch, th.pitch)) return `${head}: ${what} — MISMATCH: the program cuts pitch ${fp(th.pitch)}.`;
    const crest = th.internal ? m.rMin : m.rMax;
    if (Math.abs(crest - th.crestR) * 2 > tolD && !(th.nominal && !th.internal && Math.abs(2 * crest - th.nominal) <= tolD)) return `${head}: ${what} — MISMATCH: the program's ${th.internal ? 'bore' : 'crests'} are at ${D(th.crestR)}.`;
    return `${head}: ${what} — consistent.`;
  }
  const d = 2 * m.r;
  const shape = th.internal ? 'bore' : 'cylinder';
  if (Math.abs(d - 2 * th.crestR) <= tolD) return `${head}: the model has a plain Ø${fx(d)} ${shape} there (cosmetic thread) — consistent.`;
  if (th.nominal && Math.abs(d - th.nominal) <= tolD) return `${head}: the model has a plain Ø${fx(d)} ${shape} there (the nominal Ø of ${th.name} — cosmetic thread) — consistent.`;
  if (th.internal && Math.abs(d - 2 * th.rootR) <= tolD) return `${head}: the model has a plain Ø${fx(d)} ${shape} there (the thread's major Ø — cosmetic thread) — consistent.`;
  return `${head}: the model has a plain Ø${fx(d)} ${shape} there — MISMATCH: the program's thread ${th.internal ? 'bore is' : 'crests are'} at ${D(th.crestR)} (root ${D(th.rootR)}).`;
}

// a tapped / thread-milled hole: the model is cut exactly with half-planes from the hole axis
// toward +X, +Y, −X, −Y, which gives the hole radius along the depth in each direction
function holeThreadCheck(th, target, W, tol) {
  const head = `${th.kind === 'tap' ? 'Tapped hole' : 'Thread-milled hole'} ${th.name} (${threadSource(th)}) at ${holeAt(th)} to Z${fx(th.zTip)}`;
  const series = [0, 90, 180, 270].map((deg) => {
    const p = profileFor(target, mulXform(rotZ(-deg), mulXform(translate([-th.x, -th.y, 0]), W)));
    return { p, at: (z) => Math.max(0, Math.min(p.n - 1, Math.round((z - p.zMin) / p.dz))) };
  });
  // depth range: the tapped length inside the model's material, a pitch in from both ends
  const { p: p0 } = series[0];
  let zLoM = Infinity, zHiM = -Infinity;
  for (let i = 0; i < p0.n; i++) if (p0.outer[i] > 0) { const z = p0.zMin + i * p0.dz; zLoM = Math.min(zLoM, z); zHiM = Math.max(zHiM, z); }
  if (zHiM < zLoM) return `${head}: the model has no material there — MISMATCH: the program taps ${th.name}.`;
  let lo = Math.max(th.zTip, zLoM) + th.pitch, hi = Math.min(th.zTop, zHiM) - th.pitch;
  if (hi < lo) lo = hi = (Math.max(th.zTip, zLoM) + Math.min(th.zTop, zHiM)) / 2;
  const reads = series.map(({ p, at }) => {
    const zs = [], rs = [];
    let solid = 0;
    for (let i = at(lo); i <= at(hi); i++) { zs.push(p.zMin + i * p.dz); rs.push(p.inner[i]); if (p.outer[i] > 0 && !(p.inner[i] > 0)) solid++; }
    return { ...readThreadSeries(zs, rs, (th.d - th.minor) / 2), solid: solid > zs.length / 2 };
  });
  if (reads.filter((r) => r.solid).length >= 2) return `${head}: the model has no hole there — MISMATCH: the program taps ${th.name}.`;
  const tolD = Math.max(0.15, 0.1 * th.pitch, tol);
  const threads = reads.filter((r) => r.kind === 'thread');
  if (threads.length >= 3) {
    const pitch = threads.map((r) => r.pitch).sort((a, b) => a - b)[threads.length >> 1];
    const major = 2 * Math.max(...threads.map((r) => r.rMax)), minor = 2 * Math.min(...threads.map((r) => r.rMin));
    const what = `the model has a modeled thread of pitch ${fp(pitch)} (Ø${fx(major)} / minor Ø${fx(minor)})`;
    if (!pitchesMatch(pitch, th.pitch)) return `${head}: ${what} — MISMATCH: the program cuts pitch ${fp(th.pitch)}.`;
    if (Math.abs(major - th.d) > tolD) return `${head}: ${what} — MISMATCH: the program taps ${th.name}.`;
    return `${head}: ${what} — consistent.`;
  }
  if (!reads.every((r) => r.kind === 'plain')) return `${head}: the model's hole there is not round or not plain (no regular thread either) — MISMATCH: check it against ${th.name}.`;
  const [r0, r90, r180, r270] = reads.map((r) => r.r);
  const d = (r0 + r90 + r180 + r270) / 2, off = [(r0 - r180) / 2, (r90 - r270) / 2];
  if (Math.hypot(...off) > Math.max(0.1, tol)) return `${head}: the model's Ø${fx(d)} hole is centred X${fx(th.x + off[0])} Y${fx(th.y + off[1])} — MISMATCH: the program taps ${th.name} at ${holeAt(th)}.`;
  const size = th.nominal ? `M${fx(th.nominal)}` : th.name;
  const plain = `the model has a plain Ø${fx(d)} hole there`;
  if (Math.abs(d - th.drill) <= tolD) return `${head}: ${plain} (tap drill for ${size} — cosmetic thread) — consistent.`;
  if (Math.abs(d - th.minor) <= tolD) return `${head}: ${plain} (the minor Ø of ${th.name} — cosmetic thread) — consistent.`;
  if (Math.abs(d - th.d) <= tolD) return `${head}: ${plain} (the nominal Ø of ${size} — cosmetic thread) — consistent.`;
  if (d > th.minor && d < th.d) return `${head}: ${plain} (between the minor Ø${fx(th.minor)} and the nominal Ø${fx(th.d)} — a simplified thread) — consistent.`;
  const other = tapDrillOf(d);
  return `${head}: ${plain}${other ? ` (tap drill for ${other})` : ''} — MISMATCH: the program taps ${th.name} (tap drill Ø${fx(th.drill)}, nominal Ø${fx(th.d)}).`;
}
