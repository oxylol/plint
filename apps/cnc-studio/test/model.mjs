// Target models: readers, placement, target features, stock-vs-target comparison. node test/model.mjs
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { EXAMPLES } from '../public/js/examples.js';
import { simulateJob } from '../public/js/sim/job.js';
import { meshChunk } from '../public/js/view/surfacenets.js';
import { parseSTL, parseOBJ, meshesToTris, trisBounds, meshStats, buildTarget, defaultPlacement, targetSummary, compareTarget, guessTurnAxis, unzip, parse3MF } from '../public/js/model/model.js';

let fails = 0;
const ok = (c, m) => { console.log(c ? 'ok  ' : 'FAIL', m); if (!c) fails++; };
const near = (a, b, tol) => Math.abs(a - b) <= tol;
const require = createRequire(import.meta.url);
const FILES = new URL('../node_modules/occt-import-js/test/testfiles/', import.meta.url);

// readers
{
  const stl = parseSTL(new Uint8Array(readFileSync(new URL('cube-10x10mm/Cube 10x10.stl', FILES))));
  const b = trisBounds(stl);
  ok(stl.length === 12 * 9 && near(b.max[0] - b.min[0], 10, 1e-4) && near(meshStats(stl).volume, 1000, 0.01), 'binary STL cube: 12 triangles, 10 mm, 1000 mm³');
  const ascii = new TextEncoder().encode('solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid t\n');
  ok(parseSTL(ascii).length === 9, 'ASCII STL');
  const obj = parseOBJ('v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n');
  ok(obj.length === 18, 'OBJ quad → 2 triangles');
  const occt = await require('occt-import-js')();
  const r = occt.ReadStepFile(new Uint8Array(readFileSync(new URL('cube-10x10mm/Cube 10x10.stp', FILES))), { linearUnit: 'millimeter', linearDeflectionType: 'absolute_value', linearDeflection: 0.01, angularDeflection: 0.15 });
  const t = meshesToTris(r);
  ok(near(meshStats(t).volume, 1000, 0.5), `STEP cube through OpenCascade: ${meshStats(t).volume.toFixed(1)} mm³`);
  // 3MF = zip; build one with a stored (uncompressed) entry
  const xml = '<?xml version="1.0"?><model unit="millimeter"><resources><object id="1"><mesh><vertices><vertex x="0" y="0" z="0"/><vertex x="10" y="0" z="0"/><vertex x="0" y="10" z="0"/><vertex x="0" y="0" z="10"/></vertices><triangles><triangle v1="0" v2="2" v3="1"/><triangle v1="0" v2="1" v3="3"/><triangle v1="0" v2="3" v3="2"/><triangle v1="1" v2="2" v3="3"/></triangles></mesh></object></resources><build><item objectid="1" transform="1 0 0 0 1 0 0 0 1 5 0 0"/></build></model>';
  const zip = storeZip('3D/3dmodel.model', new TextEncoder().encode(xml));
  ok((await unzip(zip)).names.includes('3D/3dmodel.model'), 'zip reader');
  const m3 = await parse3MF(zip);
  const b3 = trisBounds(m3.tris);
  ok(m3.tris.length === 36 && near(b3.min[0], 5, 1e-6) && near(meshStats(m3.tris).volume, 1000 / 6, 0.01), '3MF tetrahedron with its build transform');
  ok(guessTurnAxis({ min: [-10, -10, -50], max: [10, 10, 0] }) === 'Z' && guessTurnAxis({ min: [0, -5, -5], max: [80, 5, 5] }) === 'X' && guessTurnAxis({ min: [0, 0, 0], max: [10, 20, 30] }) === null, 'turning-axis guess from the bounding box');
}

// a mill job against its own finished part
const job = (id, upto) => {
  const ex = EXAMPLES.find((e) => e.id === id);
  const ops = ex.ops.map((o, i) => ({ id: id + i, ...o }));
  return simulateJob(ops, upto ?? ops.length - 1, { interp: (m) => ({ machine: m }) });
};
function voxelMesh(sim) {
  const out = [];
  for (let c = 0; c < sim.cnx * sim.cny * sim.cnz; c++) {
    const m = meshChunk(sim, c, new Float32Array(256 * 3));
    if (!m) continue;
    for (let i = 0; i < m.index.length; i++) { const q = m.index[i] * 3; out.push(m.positions[q], m.positions[q + 1], m.positions[q + 2]); }
  }
  return new Float32Array(out);
}
{
  const [op10, op20] = job('plate2sides');
  // the finished part (after OP20) as a model in the part frame (= OP10 work coordinates)
  const tris = voxelMesh(op20.final);
  const model = { name: 'plate.stl', format: 'stl', tris, triCount: tris.length / 9, bounds: trisBounds(tris) };
  const t0 = performance.now();
  const target = buildTarget(model, { up: '+Z', rotZ: 0, align: 'model', offset: [0, 0, 0], scale: 1 });
  console.log(`     target voxelised in ${(performance.now() - t0).toFixed(0)} ms (${target.sim.nx}×${target.sim.ny}×${target.sim.nz})`);
  ok(near(target.volume, op20.final.materialVolume(), op20.final.materialVolume() * 0.01), `target volume ≈ part volume (${(target.volume / 1000).toFixed(2)} cm³)`);
  const c20 = compareTarget(op20, target);
  ok(/matches within/.test(c20), 'OP20 result matches the target: ' + c20.split('\n')[0]);
  const c10 = compareTarget(op10, target);
  const m = /Material still to remove: (\d+) regions?, ([\d.]+) cm³/.exec(c10);
  ok(m && +m[1] <= 2 && near(+m[2], 10.65, 0.8), `after OP10: OP20's facing + slot still to remove (${m?.[1]} region, ${m?.[2]} cm³)`);
  ok(/No gouges/.test(c10), 'after OP10: no gouges');
  const box = /X(-?[\d.]+)\.\.(-?[\d.]+) Y(-?[\d.]+)\.\.(-?[\d.]+) Z(-?[\d.]+)\.\.(-?[\d.]+)/.exec(c10.split('\n')[2] || '');
  ok(box && near(+box[5], -25, 0.6) && near(+box[6], -20, 0.6), `the excess is on the bottom (OP10 work Z${box?.[5]}..${box?.[6]})`);
  // the target described in OP20's work frame shows the slot on top and the pocket below
  const s20 = targetSummary(target, op20);
  ok(/open slot\/channel along X[^,]*, 12(\.0\d)? wide/.test(s20) && /Seen from below[\s\S]*pocket: rectangular 50/.test(s20), 'target features in OP20 coordinates: slot on top, pocket below');
  const s10 = targetSummary(target, op10);
  ok(/Seen from above[\s\S]*pocket: rectangular 50/.test(s10) && /Seen from below[\s\S]*open slot/.test(s10), 'target features in OP10 coordinates: pocket on top, slot below');
  // a gouge: OP20 slot 1 mm too deep
  const ex = EXAMPLES.find((e) => e.id === 'plate2sides');
  const deep = ex.ops.map((o, i) => ({ id: 'deep' + i, ...o, text: i === 1 ? o.text.replace('Z-5. F300', 'Z-6. F300') : o.text }));
  const [, d20] = simulateJob(deep, 1, { interp: (mm) => ({ machine: mm }) });
  const cd = compareTarget(d20, target);
  const gm = /GOUGES[^:]*: (\d+) regions?, ([\d.]+) mm³/.exec(cd);
  ok(gm && near(+gm[2], 1200, 200), `slot 1 mm too deep shows as a gouge (${gm?.[2]} mm³ ≈ 100 × 12 × 1)`);
}

// lathe: the turned shaft as a revolved mesh
{
  const [s10, s20] = job('shaft2ends');
  const sim = s20.final;
  const pts = [];
  for (let i = sim.n - 1; i >= 0; i--) if (sim.outer[i] > 1e-4) pts.push([sim.zMin + i * sim.dz, sim.outer[i]]);
  const tris = revolve(pts, 96);
  const model = { name: 'shaft.stl', format: 'stl', tris, triCount: tris.length / 9, bounds: trisBounds(tris) };
  // the model in OP20's frame is the part frame of OP10 flipped: place it in OP10 coordinates
  const pl = defaultPlacement(model, 'lathe');
  ok(pl.up === '+Z', 'lathe placement: turning axis Z');
  // OP20 frame -> part frame
  const W = s20.workFromPart;
  const inv = { R: [W.R[0], W.R[3], W.R[6], W.R[1], W.R[4], W.R[7], W.R[2], W.R[5], W.R[8]], t: [0, 0, 0] };
  const Rt = (p) => [inv.R[0] * p[0] + inv.R[1] * p[1] + inv.R[2] * p[2], inv.R[3] * p[0] + inv.R[4] * p[1] + inv.R[5] * p[2], inv.R[6] * p[0] + inv.R[7] * p[1] + inv.R[8] * p[2]];
  const tw = W.t;
  const partTris = new Float32Array(tris.length);
  for (let i = 0; i < tris.length; i += 3) { const p = Rt([tris[i] - tw[0], tris[i + 1] - tw[1], tris[i + 2] - tw[2]]); partTris.set(p, i); }
  const pm = { ...model, tris: partTris, bounds: trisBounds(partTris) };
  const target = buildTarget(pm, { up: '+Z', rotZ: 0, align: 'model', offset: [0, 0, 0], scale: 1 });
  const c20 = compareTarget(s20, target);
  ok(/within ±/.test(c20), 'flipped OP20 shaft matches the revolved target: ' + c20.split('\n')[0]);
  const ts = targetSummary(target, s20);
  ok(/Target \(turned profile\)/.test(ts) && /Ø24/.test(ts) && /Ø32/.test(ts), 'target profile in OP20 coordinates lists Ø24 and Ø32');
  const c10 = compareTarget(s10, target);
  ok(/Compared with the target model \(outside profile/.test(c10) || /matches/.test(c10), 'OP10 vs target: ' + c10.split('\n').slice(0, 3).join(' | '));
}
process.exitCode = fails ? 1 : 0;

function revolve(pts, segs) {
  const out = [];
  const ring = (r, z) => [...Array(segs).keys()].map((k) => [r * Math.cos((2 * Math.PI * k) / segs), r * Math.sin((2 * Math.PI * k) / segs), z]);
  const P = [[0, pts[0][0]], ...pts.map(([z, r]) => [r, z]), [0, pts[pts.length - 1][0]]];
  for (let i = 0; i + 1 < P.length; i++) {
    const a = ring(P[i][0], P[i][1]), b = ring(P[i + 1][0], P[i + 1][1]);
    for (let k = 0; k < segs; k++) {
      const k2 = (k + 1) % segs;
      out.push(...a[k], ...b[k], ...b[k2], ...a[k], ...b[k2], ...a[k2]);
    }
  }
  return new Float32Array(out);
}
function storeZip(name, data) {
  const enc = new TextEncoder().encode(name);
  const crc = crc32(data);
  const local = new Uint8Array(30 + enc.length);
  const dv = new DataView(local.buffer);
  dv.setUint32(0, 0x04034b50, true); dv.setUint16(8, 0, true); dv.setUint32(14, crc, true); dv.setUint32(18, data.length, true); dv.setUint32(22, data.length, true); dv.setUint16(26, enc.length, true);
  local.set(enc, 30);
  const cen = new Uint8Array(46 + enc.length);
  const cv = new DataView(cen.buffer);
  cv.setUint32(0, 0x02014b50, true); cv.setUint16(10, 0, true); cv.setUint32(16, crc, true); cv.setUint32(20, data.length, true); cv.setUint32(24, data.length, true); cv.setUint16(28, enc.length, true); cv.setUint32(42, 0, true);
  cen.set(enc, 46);
  const end = new Uint8Array(22);
  const ev = new DataView(end.buffer);
  ev.setUint32(0, 0x06054b50, true); ev.setUint16(8, 1, true); ev.setUint16(10, 1, true); ev.setUint32(12, cen.length, true); ev.setUint32(16, local.length + data.length, true);
  const all = new Uint8Array(local.length + data.length + cen.length + end.length);
  all.set(local, 0); all.set(data, local.length); all.set(cen, local.length + data.length); all.set(end, local.length + data.length + cen.length);
  return all;
}
function crc32(d) { let c, crc = 0xffffffff; for (let n = 0; n < d.length; n++) { c = (crc ^ d[n]) & 0xff; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; crc = (crc >>> 8) ^ c; } return (crc ^ 0xffffffff) >>> 0; }
