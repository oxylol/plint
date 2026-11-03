// Unit tests of the 3D view's pure modules (web/js/view3d), in node 18+ without a browser:
//
//   node tests/unit/view3d/view3d_unit.mjs <web folder> <expect.json>
//
// expect.json comes from view3d_expect.py (a synthetic run directory read with cnctwin.core.keyframes,
// cnctwin.core.rundir.read_frames, numpy and cnctwin.geom.transform), so frames.bin, brick records,
// keyframe GLBs, bricks.idx, .npz files and poses are checked against the Python side of each format.
// The browser modules are copied with the vendored three.js into a temporary folder with a package.json
// of type "module" (node then loads them as ES modules). Prints "N passed, M failed"; exit 1 on failure.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const webDir = path.resolve(process.argv[2]);
const expect = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const copy = fs.mkdtempSync(path.join(os.tmpdir(), 'cnctwin-v3d-unit-'));
fs.cpSync(webDir, path.join(copy, 'web'), { recursive: true });
fs.writeFileSync(path.join(copy, 'package.json'), '{"type": "module"}\n');
const load = (rel) => import(pathToFileURL(path.join(copy, 'web', 'js', 'view3d', rel)).href);

let passed = 0;
let failed = 0;
function check(name, cond, detail = '') {
  if (cond) passed++; else { failed++; console.log(`FAIL ${name}${detail ? ` (${detail})` : ''}`); }
}
const near = (a, b, tol) => Math.abs(a - b) <= tol;
const nearAll = (a, b, tol) => a.length === b.length && a.every((v, i) => near(v, b[i], tol));
const readBuffer = (file) => { const b = fs.readFileSync(file); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };

const THREE = await load('three.js');
const frames = await load('frames.js');
const bricks = await load('bricks.js');
const lod = await load('lod.js');
const poses = await load('poses.js');
const fields = await load('fields.js');
const measure = await load('measure.js');
const npz = await load('npz.js');
const hash = await load('hash.js');
const layers = await load('layers.js');
const coolant = await load('coolant.js');
const tool = await load('tool.js');
const apiio = await load('apiio.js');
const { PARAMS } = await load('params.js');
const runDir = expect.run_dir;

// ------------------------------------------------------------------ frames.bin
{
  const layout = frames.framesLayout(expect.frames_header);
  check('frames layout: record bytes', layout.recordBytes === expect.frames_header.record_bytes);
  check('frames layout: axes and paths', layout.nAxes === 3 && layout.nPaths === 1 && layout.relative === true);
  const block = frames.decodeFrames(readBuffer(path.join(runDir, 'frames.bin')), layout);
  check('frames: record count', block.count === expect.frames.count, `${block.count} vs ${expect.frames.count}`);
  for (const pick of expect.frames.picks) {
    const i = pick.index;
    check(`frames: tick of record ${i}`, block.tick[i] === pick.tick);
    check(`frames: axes of record ${i}`, nearAll(Array.from(block.axis.slice(i * 3, i * 3 + 3)), pick.axis, 1e-12));
    check(`frames: tip of record ${i}`, nearAll(Array.from(block.tip.slice(i * 3, i * 3 + 3)), pick.tip, 1e-12));
    check(`frames: tool axis of record ${i}`, nearAll(Array.from(block.toolAxis.slice(i * 3, i * 3 + 3)), pick.axis_T, 1e-12));
    check(`frames: unwrapped spindle of record ${i}`, near(block.spindle[i], pick.spindle_unwrapped, 1e-4), `${block.spindle[i]} vs ${pick.spindle_unwrapped}`);
  }
  const out = frames.emptySample(layout);
  const a = expect.frames.picks[2];
  const b = expect.frames.picks[2].index + 1;
  frames.interpolate(block, layout, a.tick + layout.ipoTicks / 2, out);
  const mid = [0, 1, 2].map((c) => (block.tip[a.index * 3 + c] + block.tip[b * 3 + c]) / 2);
  check('frames: interpolation halfway between ticks', nearAll(Array.from(out.tip), mid, 1e-12));
  check('frames: spindle interpolates its unwrapped angle', near(out.spindle[0], (block.spindle[a.index] + block.spindle[b]) / 2, 1e-9));
  check('frames: bracket below, above and empty', frames.bracket(block.tick, block.count, -5) === 0
    && frames.bracket(block.tick, block.count, 1e12) === block.count - 2 && frames.bracket(block.tick, 0, 1) === -1);
  // a rotary axis with its revolution count: interpolation through 2 pi
  const header = {
    record_bytes: 8 + 4 + 4 + 4 + 4 + 12 + 12, tick_s: 1e-6, ipo_ticks: 1000,
    axes: [{ name: 'C', kind: 'rotary', unit: 'rad' }], paths: [{ path: 0 }],
    fields: [{ name: 'tick', dtype: '<i8', shape: [], offset: 0 }, { name: 'axis', dtype: '<f4', shape: [1], offset: 8 },
      { name: 'axis_rev', dtype: '<i4', shape: [1], offset: 12 }, { name: 'spindle_angle_rad', dtype: '<f4', shape: [1], offset: 16 },
      { name: 'spindle_rev', dtype: '<i4', shape: [1], offset: 20 }, { name: 'tip_S_m', dtype: '<f4', shape: [1, 3], offset: 24 },
      { name: 'axis_T_S', dtype: '<f4', shape: [1, 3], offset: 36 }],
  };
  const rl = frames.framesLayout(header);
  const buf = new ArrayBuffer(rl.recordBytes * 2);
  const v = new DataView(buf);
  const put = (r, tick, angle, rev, spin, srev) => {
    const o = r * rl.recordBytes;
    v.setBigInt64(o, BigInt(tick), true); v.setFloat32(o + 8, angle, true); v.setInt32(o + 12, rev, true);
    v.setFloat32(o + 16, spin, true); v.setInt32(o + 20, srev, true);
    v.setFloat32(o + 36 + 8, 1, true);
  };
  put(0, 0, 6.2, 0, 6.0, 3);
  put(1, 1000, 0.1, 1, 0.2, 4);
  const rb = frames.decodeFrames(buf, rl);
  const ro = frames.emptySample(rl);
  frames.interpolate(rb, rl, 500, ro);
  const expected = (Math.fround(6.2) + Math.fround(0.1) + 2 * Math.PI) / 2;
  check('frames: rotary axis interpolates across 2 pi with its revolution', near(ro.axes[0], expected, 1e-6), `${ro.axes[0]} vs ${expected}`);
  check('frames: spindle revolutions unwrap', near(rb.spindle[1] - rb.spindle[0], Math.fround(0.2) + 8 * Math.PI - Math.fround(6.0) - 6 * Math.PI, 1e-6));
}

// ------------------------------------------------------------------ brick records, GLB, bricks.idx
{
  const index = bricks.BrickIndex.parse(fs.readFileSync(path.join(runDir, 'keyframes', 'bricks.idx'), 'utf8'),
    fs.readFileSync(path.join(runDir, 'keyframes', 'index.jsonl'), 'utf8').trim().split('\n').map(JSON.parse));
  for (const [t, chosen] of Object.entries(expect.versions)) {
    const mine = Object.fromEntries(index.keys.map((key) => [key, index.versionAt(key, Number(t)) ? index.versionAt(key, Number(t)).k : null]));
    check(`bricks.idx: versions at t = ${t} as the API picks them`, JSON.stringify(mine) === JSON.stringify(chosen), `${JSON.stringify(mine)} vs ${JSON.stringify(chosen)}`);
  }
  check('bricks.idx: position counts keyframes at or before t', index.position(0) === 1 && index.position(0.75) === 2 && index.position(1e9) === index.sortedTimes.length);
  for (const [key, want] of Object.entries(expect.bricks)) {
    const version = index.versionAt(key, 2.3);
    const buffer = readBuffer(path.join(runDir, 'keyframes', version.file));
    const brick = bricks.decodeRecord(buffer, version.offset, version.length, key, version.k);
    const p = bricks.brickPositions(brick);
    const sum = [0, 0, 0];
    for (let i = 0; i < brick.nVertices; i++) for (let c = 0; c < 3; c++) sum[c] += p[i * 3 + c];
    check(`brick ${key}: vertex and triangle counts`, brick.nVertices === want.n && brick.nIndices === want.m * 3);
    check(`brick ${key}: dequantised positions (float32 of the reader's float64)`, nearAll(sum, want.sum, 1e-6 * want.n) && nearAll(Array.from(p.slice(0, 3)), want.first, 1e-7));
    check(`brick ${key}: first triangle`, JSON.stringify(Array.from(brick.indices.slice(0, 3))) === JSON.stringify(want.faces0));
    const all = bricks.keyframeBricks(buffer, version.k);
    const found = all.find((b) => b.key === key);
    check(`brick ${key}: the GLB read alone gives the same record`, found && found.offset === version.offset && found.brick.nVertices === want.n);
    check(`brick ${key}: bounding box holds the vertices`, (() => { for (let i = 0; i < brick.nVertices; i++) for (let c = 0; c < 3; c++) if (p[i * 3 + c] < brick.bbox.lo[c] - 1e-9 || p[i * 3 + c] > brick.bbox.hi[c] + 1e-9) return false; return true; })());
  }
  check('brick keys order naturally', ['10', '2', 'w1.3', '1'].sort(bricks.keyOrder).join(',') === '1,2,10,w1.3' && bricks.workpieceOf('w2.7') === 2 && bricks.workpieceOf('7') === 0);
  // the loader: fetches per file or per brick, applies the versions of the target time, reuses its cache
  const fetched = { files: [], bricks: [] };
  const shown = new Map();
  let idle = 0;
  const loader = new bricks.BrickLoader({
    index,
    fetchFile: async (file) => { fetched.files.push(file); return readBuffer(path.join(runDir, 'keyframes', file)); },
    fetchBrick: async (key, t) => {
      fetched.bricks.push(key);
      const v = index.versionAt(key, t);
      const buffer = readBuffer(path.join(runDir, 'keyframes', v.file));
      return buffer.slice(v.offset, v.offset + v.length);
    },
    apply: (key, brick) => { shown.set(key, brick ? brick.k : null); },
    onIdle: () => { idle++; }, cacheBytes: 64 << 20, maxParallel: 2, perBrickMax: 6,
  });
  const settle = async () => { for (let i = 0; i < 200 && loader.busy; i++) await new Promise((r) => setTimeout(r, 5)); };
  for (const t of [1.2, 2.3, 0.0, 1.6]) {
    loader.setTarget(t);
    await settle();
    const want = Object.fromEntries(index.keys.map((key) => [key, index.versionAt(key, t).k]));
    check(`loader: every brick at its version for t = ${t}`, JSON.stringify(Object.fromEntries(shown)) === JSON.stringify(want), JSON.stringify(Object.fromEntries(shown)));
  }
  const before = fetched.files.length + fetched.bricks.length;
  loader.setTarget(1.2);
  await settle();
  check('loader: scrubbing back is served from the cache', fetched.files.length + fetched.bricks.length === before);
  check('loader: idle callbacks', idle > 0);
  check('loader: no errors', loader.errors.length === 0, loader.errors.join(' | '));
}

// ------------------------------------------------------------------ level of detail
{
  const n = 41;
  const positions = new Float32Array(n * n * 3);
  const index = [];
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) positions.set([i * 0.5e-3, j * 0.5e-3, 0.015], (j * n + i) * 3);
  for (let j = 0; j + 1 < n; j++) for (let i = 0; i + 1 < n; i++) {
    const a = j * n + i;
    index.push(a, a + 1, a + n + 1, a, a + n + 1, a + n);
  }
  const values = new Float32Array(n * n).map((_, i) => (i % 2 ? 1 : 3));
  const cell = 2e-3;
  const anchor = [-cell / 2, -cell / 2, 0];
  const out = lod.clusterMesh(positions, Uint32Array.from(index), cell, anchor, values);
  check('lod: vertices merge into cubes', out.positions.length / 3 === 11 * 11, `${out.positions.length / 3}`);
  let worst = 0;
  for (let v = 0; v < n * n; v++) {
    const c = out.vertexMap[v];
    worst = Math.max(worst, Math.hypot(positions[v * 3] - out.positions[c * 3], positions[v * 3 + 1] - out.positions[c * 3 + 1], positions[v * 3 + 2] - out.positions[c * 3 + 2]));
  }
  check('lod: no vertex moves more than the cube diagonal', worst <= Math.sqrt(3) * cell + 1e-9, `${worst}`);
  check('lod: triangles with two corners in one cube are removed', out.indices.length < index.length && out.indices.length % 3 === 0);
  let degenerate = 0;
  for (let t = 0; t < out.indices.length; t += 3) if (out.indices[t] === out.indices[t + 1] || out.indices[t + 1] === out.indices[t + 2] || out.indices[t] === out.indices[t + 2]) degenerate++;
  check('lod: no degenerate triangle left', degenerate === 0);
  check('lod: values are averaged per cube', out.values.every((v) => v > 1 && v < 3));
  const missing = new Float32Array(n * n).fill(lod.NO_VALUE);
  check('lod: no value stays no value', lod.clusterMesh(positions, Uint32Array.from(index), cell, anchor, missing).values.every((v) => v < lod.HAS_VALUE));
  check('lod: level choice by projected cube size', lod.chooseLevel([0.5e-3, 2e-3, 8e-3], 0.2, Math.PI / 6, 600, 1.5) === 0
    && lod.chooseLevel([0.5e-3, 2e-3, 8e-3], 20, Math.PI / 6, 600, 1.5) === 2);
}

// ------------------------------------------------------------------ poses, chains, stock motion
{
  for (const { spec, matrix } of expect.poses) {
    const m = poses.matrixFromPose(spec);
    const rows = [0, 1, 2, 3].map((r) => [0, 1, 2, 3].map((c) => m.elements[c * 4 + r]));
    check(`pose ${JSON.stringify(spec)} as cnctwin.geom.transform`, rows.every((row, r) => nearAll(row, matrix[r], 1e-12)));
  }
  const rows = [[1, 0, 0, 0.1], [0, 0, -1, 0.2], [0, 1, 0, 0.3], [0, 0, 0, 1]];
  const a = poses.matrixFromPose(rows);
  check('pose from 4 x 4 rows', near(a.elements[12], 0.1, 0) && near(a.elements[6], 1, 0) && near(a.elements[9], -1, 0));
  check('pose from 16 numbers equals rows', poses.matrixFromPose(rows.flat()).equals(a));
  check('pose: bad input is null', poses.matrixFromPose([1, 2, 3]) === null && poses.matrixFromPose({ foo: 1 }) === null && poses.matrixFromPose({ translate_m: [0, 0, 0], order: 'xxy' }) === null);
  for (const { z, x } of expect.perpendiculars) {
    const zv = new THREE.Vector3(...z).normalize();
    const p = poses.perpendicular(zv);
    check(`perpendicular(${z}) as cnctwin.geom.transform`, nearAll(p.toArray(), x, 1e-12), p.toArray().join(','));
  }
  const m = poses.toolMatrix(new THREE.Matrix4(), [0.01, 0.02, 0.03], [0, 0, 1], new THREE.Vector3(1, 0, 0), Math.PI / 2);
  const e = m.elements;
  check('tool frame: x turned by the spindle angle about z', near(e[0], 0, 1e-12) && near(e[1], 1, 1e-12) && near(e[8], 0, 1e-12) && near(e[10], 1, 1e-12) && near(e[12], 0.01, 0));
  const tilted = poses.toolMatrix(new THREE.Matrix4(), [0, 0, 0], [0, Math.SQRT1_2, Math.SQRT1_2], new THREE.Vector3(1, 0, 0), 0);
  check('tool frame: z along the tool axis, x from the hint', near(tilted.elements[9], Math.SQRT1_2, 1e-12) && near(tilted.elements[0], 1, 1e-12));
  const p0 = new THREE.Matrix4().makeTranslation(0, 0, 0);
  const p1 = new THREE.Matrix4().makeRotationZ(Math.PI / 2).setPosition(1, 0, 0);
  const half = poses.interpolatePose(p0, p1, 0.5);
  const d = poses.poseDifference(half, new THREE.Matrix4().makeRotationZ(Math.PI / 4).setPosition(0.5, 0, 0));
  check('pose interpolation: lerp and slerp', d.distance_m < 1e-12 && d.angle_rad < 1e-7, JSON.stringify(d));
  const axes = new Map([['X', { name: 'X', kind: 'linear', direction: [1, 0, 0] }], ['C', { name: 'C', kind: 'rotary', direction: [0, 0, 1], point_m: [0.1, 0, 0] }]]);
  const chain = new poses.Chain(['base', 'X', { link: 'table', axis: 'C' }, { link: 'top', fixed: { translate_mm: [0, 0, 50] } }], axes);
  check('chain: complete when every joint has a direction', chain.complete && chain.has('table') && chain.has('X'));
  const fk = chain.fk({ X: 0.2, C: Math.PI / 2 });
  const end = new THREE.Vector3().setFromMatrixPosition(fk.end);
  check('chain: linear then rotary about a line through a point, then a fixed offset', nearAll(end.toArray(), [0.3, -0.1, 0.05], 1e-12), end.toArray().join(','));
  check('chain: incomplete without a direction', !new poses.Chain(['Y'], new Map([['Y', { name: 'Y', kind: 'linear' }]])).complete);
  const keyed = new poses.StockMotion({ staticPose: null, keyframeTimes: [0, 0.5, 1.0] });
  keyed.setKeyframe(0, new THREE.Matrix4().makeTranslation(0, 0, 0));
  keyed.setKeyframe(1, new THREE.Matrix4().makeTranslation(0.1, 0, 0));
  check('stock motion: interpolated between keyframe poses', near(new THREE.Vector3().setFromMatrixPosition(keyed.at(0.25, 0, {})).x, 0.05, 1e-12));
  check('stock motion: held after the last keyframe pose', near(new THREE.Vector3().setFromMatrixPosition(keyed.at(0.75, 1, {})).x, 0.1, 1e-12));
  check('stock motion: unknown without any pose', new poses.StockMotion({}).at(0, -1, {}) === null);
  const linear = new poses.Chain(['X'], new Map([['X', { name: 'X', kind: 'linear', direction: [-1, 0, 0] }]]));
  const moving = new poses.StockMotion({ workChain: linear, keyframeTimes: [0, 1] });
  moving.setKeyframe(0, new THREE.Matrix4().makeTranslation(-0.3, 0, 0));
  moving.setKeyframe(1, new THREE.Matrix4().makeTranslation(-0.4, 0, 0));
  check('stock motion: the work chain reproduces the next keyframe', moving.validate(0, { X: 0 }, { X: 0.1 }) === true);
  check('stock motion: between keyframes it follows the chain', near(new THREE.Vector3().setFromMatrixPosition(moving.at(0.3, 0, { X: 0.07 })).x, -0.37, 1e-12));
  const still = new poses.StockMotion({ workChain: linear, keyframeTimes: [0, 1] });
  still.setKeyframe(0, new THREE.Matrix4());
  still.setKeyframe(1, new THREE.Matrix4());
  check('stock motion: a chain that contradicts the keyframes is not used', still.validate(0, { X: 0 }, { X: 0.1 }) === false
    && near(new THREE.Vector3().setFromMatrixPosition(still.at(0.5, 0, { X: 0.05 })).x, 0, 1e-12));
}

// ------------------------------------------------------------------ .npz and keyframe fields
{
  const arrays = await npz.parseNpz(readBuffer(expect.extra_npz));
  check('npz: float16', nearAll(Array.from(arrays.h.data), expect.extra.h.data, 1e-9));
  check('npz: bool', arrays.b.dtype === 'bool' && Array.from(arrays.b.data).join() === '1,0,1');
  check('npz: unicode, stored member', arrays.u.dtype === 'str' && arrays.u.data.join('|') === 'n1|jet two|');
  check('npz: int64 exact', arrays.i8.data[0] === 2 ** 40 && arrays.i8.data[1] === -3);
  check('npz: Fortran order returned in C order', JSON.stringify(arrays.f.shape) === '[3,4]' && nearAll(Array.from(arrays.f.data), expect.extra.f.data, 0));
  check('npz: big-endian float64', nearAll(Array.from(arrays.big.data), [1.5, 2.5], 0));
  const kf = await npz.parseNpz(readBuffer(path.join(runDir, 'keyframes', 'kf_2.npz')));
  for (const [name, doc] of Object.entries(expect.fields_kf2)) {
    const got = kf[name];
    if (!got) { check(`npz field ${name} present`, false); continue; }
    const same = Array.isArray(got.data) ? got.data.join('|') === doc.data.join('|')
      : got.data.length === doc.data.length && doc.data.every((v, i) => (v === null ? !Number.isFinite(got.data[i]) : near(got.data[i], v, Math.abs(v) * 1e-6 + 1e-12)));
    check(`npz field ${name} (${doc.dtype} ${JSON.stringify(doc.shape)})`, same && JSON.stringify(got.shape) === JSON.stringify(doc.shape));
  }
  const size = expect.stock_size_m;
  const grids = fields.layerGrids(kf, size);
  check('fields: the 3-D temperature grid is preferred', grids.temperature && grids.temperature.dims === 3 && grids.temperature.name === 'work_temp_K');
  check('fields: the pile grid has its own origin', grids.chips && near(grids.chips.origin[0], -0.010, 1e-12) && grids.chips.n[0] === 35 && grids.chips.orientation === 'values[x, y]');
  check('fields: the displacement grid is its magnitude', grids.deflection && grids.deflection.name === 'work_displacement_m' && grids.deflection.dims === 3);
  const surface = fields.gridFromArrays(kf, 'surface_temp_K', size);
  check('fields: a [ny, nx] surface grid is read rows = y (the shell rule)', surface && surface.n[0] === 10 && surface.n[1] === 8 && surface.orientation === 'values[y, x]');
  const raw = kf.surface_temp_K;
  const want = raw.data[3 * 10 + 7];
  const at = fields.sampleGrid(surface, (7 + 0.5) * 0.004, (3 + 0.5) * 0.004, 0.015);
  check('fields: sampling at a cell centre gives the cell', near(at, want, 1e-9), `${at} vs ${want}`);
  const lin = { name: 'lin', dims: 3, n: [4, 3, 2], origin: [0, 0, 0], cell: 1, axes: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], values: new Float32Array(24) };
  for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 2; k++) lin.values[i + 4 * (j + 3 * k)] = 2 * i + 3 * j - k;
  const f = (x, y, z) => 2 * (x - 0.5) + 3 * (y - 0.5) - (z - 0.5);
  check('fields: trilinear interpolation is exact for a linear field', near(fields.sampleGrid(lin, 1.3, 1.1, 0.9), f(1.3, 1.1, 0.9), 1e-6));
  lin.values[1 + 4 * (1 + 3 * 0)] = NaN;
  const masked = fields.sampleGrid(lin, 1.9, 1.5, 0.5);
  check('fields: cells without material do not bleed (normalised convolution)', near(masked, 7, 1e-6), `${masked}`);
  check('fields: at the centre of an empty cell there is no value', fields.sampleGrid(lin, 1.5, 1.5, 0.5) === null);
  check('fields: nothing near is null', fields.sampleGrid({ ...lin, values: new Float32Array(24).fill(NaN) }, 1, 1, 1) === null);
  check('fields: tool deflection of a keyframe', fields.toolDeflection(kf).vector_m.length === 3 && fields.toolDeflection(kf).frame === 'T0');
  check('fields: float16 rounding to nearest even as numpy', fields.roundHalf(310.2) === 310.25 && fields.roundHalf(310.13) === 310.25
    && fields.roundHalf(1e-5) === 1.0013580322265625e-05 && fields.roundHalf(65520) === Infinity && fields.roundHalf(2049) === 2048 && fields.roundHalf(2051) === 2052);
  check('fields: shortest decimal of float16 and float32', fields.shortestRepr(310.25, 'float16') === 310.2
    && fields.shortestRepr(Math.fround(0.1), 'float32') === 0.1 && fields.shortestRepr(0.123456789, 'float64') === 0.123456789);
  check('fields: sampleEvenly as the shell', JSON.stringify(fields.sampleEvenly([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 4)) === '[0,3,6,9]');
  check('fields: finite range', JSON.stringify(fields.finiteRange([3, NaN, -1, null, 2])) === '[-1,3]' && fields.finiteRange([NaN]) === null);
  const attr = await npz.parseNpz(readBuffer(path.join(runDir, 'keyframes', 'attr', 'brick_0.npz')));
  check('attributes: names the layers read', layers.ATTRIBUTE_NAMES.deviation.includes('deviation_m') && layers.ATTRIBUTE_NAMES.finish.includes('Ra_m')
    && layers.ATTRIBUTE_NAMES.wear.includes('generating_VB_m') && attr.deviation_m.data.length === expect.attr_brick0.deviation_m.data.length);
  const jets = coolant.jetsFromArrays(kf);
  check('coolant: jets of a keyframe with paths, diameters, blocked', jets.length === 2 && jets[1].blockedBy === 'workpiece 0' && jets[1].path && jets[1].path.length >= 2 && near(jets[1].diameter, 0.003, 1e-12));
}

// ------------------------------------------------------------------ measuring, coolant curve, tool bend, hash, layers, api
{
  const c = new THREE.Vector3(0.01, 0.02, 0.005);
  const r = 0.00412;
  const onCircle = (a) => new THREE.Vector3(c.x + r * Math.cos(a), c.y + r * Math.sin(a), c.z);
  const circle = measure.circleThrough(onCircle(0.3), onCircle(2.1), onCircle(4.4));
  check('measure: circle through three points', circle && near(circle.radius_m, r, 1e-12) && circle.center.distanceTo(c) < 1e-12);
  check('measure: collinear points are refused', measure.circleThrough(new THREE.Vector3(0, 0, 0), new THREE.Vector3(1, 0, 0), new THREE.Vector3(2, 0, 1e-9)) === null);
  const pp = measure.pointPlane(new THREE.Vector3(1, 2, 3.5), new THREE.Vector3(0, 0, 3), new THREE.Vector3(0, 0, 1));
  check('measure: point to plane', near(pp.distance_m, 0.5, 1e-12) && pp.foot.equals(new THREE.Vector3(1, 2, 3)));
  const g = new THREE.Vector3(0, 0, -PARAMS.standard_gravity_m_s2);
  const line = coolant.centreline(new THREE.Vector3(0, 0, 0), new THREE.Vector3(1, 0, 0), 3, g, true, 0.6, 6);
  const t = 0.6 / 3;
  check('coolant: liquid jets fall ballistically', near(line[6].x, 0.6, 1e-12) && near(line[6].z, -0.5 * PARAMS.standard_gravity_m_s2 * t * t, 1e-12));
  check('coolant: gas jets go straight', coolant.centreline(new THREE.Vector3(), new THREE.Vector3(0, 1, 0), 30, g, false, 0.3, 3)[3].z === 0);
  const tip = tool.bendOffset(0, 0.035, [1e-5, 0, 2e-6]);
  const clamp = tool.bendOffset(0.035, 0.035, [1e-5, 0, 2e-6]);
  const middle = tool.bendOffset(0.0175, 0.035, [1e-5, 0, 0]);
  check('tool: cantilever bend, full at the tip, none at the clamp', near(tip[0], 1e-5, 1e-18) && near(tip[2], 2e-6, 1e-18) && clamp.every((v) => v === 0));
  check('tool: cantilever shape at mid-length (3/2 xi^2 - xi^3/2 = 5/16)', near(middle[0], 1e-5 * 5 / 16, 1e-18));
  const u = Array.from({ length: 2000 }, (_, i) => hash.hashUniform('view3d.test', i, 7));
  check('hash: deterministic and in [0, 1)', hash.hashUniform('view3d.test', 3, 7) === u[3] && u.every((v) => v >= 0 && v < 1));
  const mean = u.reduce((s, v) => s + v, 0) / u.length;
  check('hash: uniform mean', Math.abs(mean - 0.5) < 0.02, `${mean}`);
  check('hash: streams differ', hash.hashUniform('view3d.a', 1) !== hash.hashUniform('view3d.b', 1));
  const colormap = await import(pathToFileURL(path.join(copy, 'web', 'js', 'plot', 'colormap.js')).href);
  const table = colormap.lut('sequential-orange', 'light', 256);
  const lo = layers.lutColor(table, 0, 0, 1).getHexString(THREE.SRGBColorSpace);
  const hi = layers.lutColor(table, 1, 0, 1).getHexString(THREE.SRGBColorSpace);
  const hex = (i) => [0, 1, 2].map((k) => table[i * 3 + k].toString(16).padStart(2, '0')).join('');
  check('layers: colour lookup matches the shell colour map ends', lo === hex(0) && hi === hex(255), `${lo} ${hi}`);
  check('layers: display units', layers.DISPLAY.temperature.offset === -273.15 && layers.DISPLAY.deviation.scale === 1e6 && layers.DISPLAY.chips.unit === 'mm');
  check('layers: the seven layers', layers.LAYERS.join(',') === 'temperature,chips,coolant,finish,deviation,deflection,wear');
  const calls = [];
  const minimal = apiio.adaptApi({ get: async (p) => { calls.push(['get', p]); return { ok: 1 }; }, getBinary: async (p) => { calls.push(['bin', p]); return new Uint8Array([1, 2, 3]); } });
  const json = await minimal.json('/api/x');
  const bin = await minimal.binary('/api/y');
  check('api: the contract api (get, getBinary; a typed array becomes an ArrayBuffer)', json.ok === 1 && bin instanceof ArrayBuffer && bin.byteLength === 3 && calls.length === 2);
  const shell = apiio.adaptApi({ json: async () => 1, binary: async () => ({ buffer: new ArrayBuffer(2), headers: null }) });
  check('api: the shell client spellings', (await shell.binary('/a')).byteLength === 2 && (await shell.json('/b')) === 1);
  let refused = false;
  try { apiio.adaptApi({}); } catch { refused = true; }
  check('api: an object without get/getBinary is refused', refused);
  check('api: query strings', apiio.query({ t0: 0, t1: 1.5, max_frames: null, names: ['a', 'b'] }) === '?t0=0&t1=1.5&names=a%2Cb');
  check('params: SI values from db/models/view3d.yaml', near(PARAMS.fov_rad, Math.PI / 6, 1e-12) && near(PARAMS.jet_max_length_m, 0.6, 1e-12) && PARAMS.chunk_bricks === 4);
}

fs.rmSync(copy, { recursive: true, force: true });
console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
