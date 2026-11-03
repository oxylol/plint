// Unit tests of the UI's pure modules (no browser): number and unit formatting, scales and ticks,
// colour maps, the API client's helpers and decoders (query strings, frames.bin, .npy/.npz, the
// raw-DEFLATE inflater against zlib), series building for the plots (envelopes, M4 buckets, folding
// past eight columns), the run data index (blocks by time, stepping, time per line), the
// inspection judgement and the field-grid orientation. Node 18 or later:
//
//   node tests/ui/unit_test.mjs [web/js folder]
//
// The modules are plain browser ES modules; they are copied to a temporary folder with a
// package.json of type "module" so Node loads them as ES modules.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import zlib from 'node:zlib';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const source = path.resolve(process.argv[2] || process.env.WEB_JS || path.join(here, '..', '..', 'web', 'js'));
const copy = fs.mkdtempSync(path.join(os.tmpdir(), 'cnctwin-ui-unit-'));
fs.cpSync(source, path.join(copy, 'js'), { recursive: true });
fs.writeFileSync(path.join(copy, 'js', 'package.json'), '{"type": "module"}\n');
const load = (rel) => import(pathToFileURL(path.join(copy, 'js', rel)).href);

let passed = 0;
let failed = 0;
function check(name, cond, detail = '') {
  if (cond) { passed++; } else { failed++; console.log(`FAIL ${name}${detail ? ` (${detail})` : ''}`); }
}
function near(a, b, tol = 1e-9) { return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b)); }

const units = await load('plot/units.js');
const scale = await load('plot/scale.js');
const colormap = await load('plot/colormap.js');
const api = await load('api.js');
const timeplot = await load('plot/timeplot.js');
const plots = await load('panels/plots.js');
const rundata = await load('panels/rundata.js');
const results = await load('panels/results.js');
const fields = await load('panels/fields.js');
const transport = await load('panels/transport.js');

// ------------------------------------------------------------------ units
const M = units.MINUS;
check('num: 4 significant figures', units.num(2.06217) === '2.062', units.num(2.06217));
check('num: integers as they are', units.num(2320) === '2320' && units.num(-3) === `${M}3`);
check('num: large values rounded plainly', units.num(177699.168) === '177699', units.num(177699.168));
check('num: tiny values in e notation', units.num(6.2e-6) === `6.200e${M}6`, units.num(6.2e-6));
check('num: rounding into the next decade', units.num(0.99999) === '1.000', units.num(0.99999));
check('num: missing is a dash', units.num(null) === units.DASH && units.num(NaN) === units.DASH && units.num('x') === units.DASH);
check('fixed: minus sign U+2212, no negative zero', units.fixed(-0.0001, 2) === '0.00' && units.fixed(-1.5, 1) === `${M}1.5`);
check('formatSI: lengths by size', units.formatSI(6.2e-5, 'm') === '62.00 µm' && units.formatSI(0.002062, 'm') === '2.062 mm' && units.formatSI(1.5, 'm') === '1.500 m',
  `${units.formatSI(6.2e-5, 'm')} ${units.formatSI(0.002062, 'm')}`);
check('formatSI: temperature in °C', units.formatSI(293.15, 'K') === '20.00 °C', units.formatSI(293.15, 'K'));
check('formatSI: power, energy, speed, volume', units.formatSI(1500, 'W') === '1.500 kW' && units.formatSI(177699, 'J') === '177.7 kJ'
  && units.formatSI(1e8, 'J') === '27.78 kWh' && units.formatSI(986.17, 'rad/s') === '9417 rpm' && units.formatSI(9.44e-7, 'm3') === '944.0 mm³',
  [units.formatSI(1500, 'W'), units.formatSI(177699, 'J'), units.formatSI(986.17, 'rad/s'), units.formatSI(9.44e-7, 'm3')].join(' '));
check('formatSI: feeds in mm/min, cutting speeds in m/min', units.formatSI(0.05, 'm/s') === '3000 mm/min' && units.formatSI(5, 'm/s') === '300.0 m/min');
check('quantity: key units', units.quantity('deviation_m', 6.2e-5).text === '62.00 µm' && units.quantity('deviation_m', 6.2e-5).label === 'deviation');
check('quantity: hertz, fractions, plain', units.quantity('chatter_freq_Hz', 1669).text === '1669 Hz' && units.quantity('life_used_fraction', 0.25).text === '25.00 %'
  && units.quantity('flute_fill', 0.72).text === '0.7200', `${units.quantity('life_used_fraction', 0.25).text}`);
check('quantity: non-SI interface keys converted', units.quantity('travel_mm', 762).text === '762.0 mm' && units.quantity('ambient_degC', 20).text === '20.00 °C',
  `${units.quantity('travel_mm', 762).text} ${units.quantity('ambient_degC', 20).text}`);
check('parseKey: longest suffix first', units.parseKey('flow_m3_s').si === 'm3/s' && units.parseKey('omega_rad_s').si === 'rad/s' && units.parseKey('hardness_HB') === null);
check('duration and clock', units.duration(2.32) === '2.320 s' && units.duration(192.4) === '3 min 12.4 s' && units.duration(7500) === '2 h 05 min'
  && units.clock(75.5) === '1:15.500', `${units.duration(192.4)} ${units.clock(75.5)}`);
check('bytes', units.bytes(512) === '512 B' && units.bytes(1536) === '1.50 KiB', units.bytes(1536));

// ------------------------------------------------------------------ scales
check('niceStep 1-2-5', scale.niceStep(10, 5) === 2 && scale.niceStep(0.7, 5) === 0.1 && scale.niceStep(3.2, 4) === 1);
const ticks = scale.niceTicks(0, 2.32, 6);
check('niceTicks inside the range', ticks[0] === 0 && ticks[ticks.length - 1] <= 2.32 && ticks.length >= 4, ticks.join(','));
check('extent skips NaN and null', JSON.stringify(scale.extent([1, NaN, 3], [null, -2])) === '[-2,3]');
check('bisectRight and nearestIndex', scale.bisectRight([0, 1, 2, 3], 2.5) === 2 && scale.bisectRight([1, 2], 0) === -1 && scale.nearestIndex([0, 1, 2], 1.6) === 2);
const hist = scale.histogram([1, 2, 2, 3, NaN, 4], { bins: 3 });
check('histogram counts every finite value once', hist.counts.reduce((a, b) => a + b, 0) === 5 && hist.edges.length === 4);
check('tickLabel precision', scale.tickLabel(0.25, 0.05) === '0.25' && scale.tickLabel(-1, 0.5) === `${M}1.0`);

// ------------------------------------------------------------------ colours
check('categorical palette in its fixed order', colormap.categorical(0, 'light') === '#2a78d6' && colormap.categorical(9, 'light') === '#eb6834'
  && colormap.categorical(0, 'dark') === '#3987e5');
const lab = colormap.rgbToOklab([42, 120, 214]);
const back = colormap.oklabToRgb(lab);
check('OKLab round trip', back.every((v, i) => Math.abs(v - [42, 120, 214][i]) < 0.6), back.join(','));
const seq = colormap.colormap('sequential', 'light');
check('sequential light runs light to dark', colormap.rgbToOklab(seq(0))[0] > colormap.rgbToOklab(seq(1))[0]);
const seqDark = colormap.colormap('sequential', 'dark');
check('sequential dark is anchored at the surface', colormap.rgbToOklab(seqDark(0))[0] < colormap.rgbToOklab(seqDark(1))[0]);
const orange = colormap.stops('sequential-orange', 'light');
check('the orange ramp is monotone in lightness', orange.every((hex, i) => i === 0 || colormap.rgbToOklab(colormap.hexToRgb(hex))[0] < colormap.rgbToOklab(colormap.hexToRgb(orange[i - 1]))[0] + 1e-9));
const div = colormap.stops('diverging', 'light');
check('diverging has a grey midpoint', div[3] === '#f0efec' && div.length === 7);
check('every layer has a map', ['temperature', 'chips', 'finish', 'deviation', 'deflection', 'wear', 'coolant'].every((l) => colormap.LAYER_MAPS[l]));

// ------------------------------------------------------------------ API helpers and decoders
check('queryString skips null and joins lists', api.queryString({ a: 1, b: null, c: ['x', 'y'], d: undefined }) === '?a=1&c=x%2Cy');
check('encodePath keeps slashes', api.encodePath('keyframes/attr/brick 0.npz') === 'keyframes/attr/brick%200.npz');
const client = api.createApi({ base: '/twin/', fetch: async () => { throw new Error('no network in unit tests'); } });
check('createApi url', client.url('/api/runs', { limit: 5 }) === '/twin/api/runs?limit=5' && client.fileUrl('r1', 'images/a b.png') === '/twin/api/runs/r1/files/images/a%20b.png');
for (const name of ['scene', 'frames', 'keyframes', 'keyframeGlb', 'keyframeFields', 'brick', 'chips', 'jets', 'json', 'get', 'getJSON', 'arrayBuffer', 'npz', 'fileUrl']) {
  check(`api.${name} exists for the 3D view`, typeof client[name] === 'function');
}
let caught = null;
try { await client.run('x'); } catch (error) { caught = error; }
check('network failure is an ApiError with status 0', caught instanceof api.ApiError && caught.status === 0);
let inflateOk = true;
for (let n = 0; n < 40; n++) {
  const size = (n * 7919) % 150000;
  const data = Buffer.alloc(size);
  let x = n + 1;
  for (let i = 0; i < size; i++) { x = (x * 1103515245 + 12345) & 0x7fffffff; data[i] = i % 3 === 0 ? (x >> 16) & 255 : i % 251; }
  for (const level of [0, 1, 9]) {
    const out = api.inflateRawSync(new Uint8Array(zlib.deflateRawSync(data, { level })));
    if (Buffer.compare(Buffer.from(out), data) !== 0) inflateOk = false;
  }
}
check('inflateRawSync equals zlib (stored, fixed and dynamic blocks)', inflateOk);
check('half floats', api.halfToFloat(0x3c00) === 1 && api.halfToFloat(0xc000) === -2 && Number.isNaN(api.halfToFloat(0x7e00)) && api.halfToFloat(0x0001) > 0);

function npy(descr, shape, body, fortran = false) {
  let header = `{'descr': '${descr}', 'fortran_order': ${fortran ? 'True' : 'False'}, 'shape': (${shape.join(', ')}${shape.length === 1 ? ',' : ''}), }`;
  while ((10 + header.length + 1) % 64) header += ' ';
  header += '\n';
  const head = Buffer.alloc(10);
  head.write('\x93NUMPY', 0, 'latin1');
  head[6] = 1; head[7] = 0;
  head.writeUInt16LE(header.length, 8);
  return Buffer.concat([head, Buffer.from(header, 'latin1'), body]);
}
const f8 = Buffer.alloc(48); [1, 2, 3, 4, 5, 6].forEach((v, i) => f8.writeDoubleLE(v, i * 8));
const parsed = api.parseNpy(npy('<f8', [2, 3], f8));
check('parseNpy float64 2-D', parsed.dtype === 'float64' && parsed.shape.join() === '2,3' && parsed.data[5] === 6);
const fortran = api.parseNpy(npy('<f8', [2, 3], f8, true));
check('parseNpy Fortran order to C order', Array.from(fortran.data).join() === '1,3,5,2,4,6', Array.from(fortran.data).join());
const i8 = Buffer.alloc(16); i8.writeBigInt64LE(2161000n, 0); i8.writeBigInt64LE(-5n, 8);
check('parseNpy int64', Array.from(api.parseNpy(npy('<i8', [2], i8)).data).join() === '2161000,-5');
const u = Buffer.alloc(16); 'n1'.split('').forEach((ch, i) => u.writeUInt32LE(ch.charCodeAt(0), i * 4)); 'ab'.split('').forEach((ch, i) => u.writeUInt32LE(ch.charCodeAt(0), 8 + i * 4));
check('parseNpy unicode strings', api.parseNpy(npy('<U2', [2], u)).data.join() === 'n1,ab');
const b1 = Buffer.from([1, 0, 1]);
check('parseNpy booleans', Array.from(api.parseNpy(npy('|b1', [3], b1)).data).join() === '1,0,1');
const f2 = Buffer.alloc(4); f2.writeUInt16LE(0x3c00, 0); f2.writeUInt16LE(0x5bd0, 2);
check('parseNpy float16', api.parseNpy(npy('<f2', [2], f2)).data[1] === 250, String(api.parseNpy(npy('<f2', [2], f2)).data[1]));
const be = Buffer.alloc(8); be.writeDoubleBE(2.5, 0);
check('parseNpy big endian', api.parseNpy(npy('>f8', [1], be)).data[0] === 2.5);

function crc32(buf) {
  let c = ~0;
  for (const byte of buf) { c ^= byte; for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1)); }
  return ~c >>> 0;
}
function zip(entries, deflate) {
  const locals = []; const centrals = []; let offset = 0;
  for (const [name, data] of entries) {
    const body = deflate ? zlib.deflateRawSync(data) : data;
    const nameBuf = Buffer.from(name);
    const local = Buffer.alloc(30); local.writeUInt32LE(0x04034b50, 0); local.writeUInt16LE(20, 4); local.writeUInt16LE(deflate ? 8 : 0, 8);
    local.writeUInt32LE(crc32(data), 14); local.writeUInt32LE(body.length, 18); local.writeUInt32LE(data.length, 22); local.writeUInt16LE(nameBuf.length, 26);
    const central = Buffer.alloc(46); central.writeUInt32LE(0x02014b50, 0); central.writeUInt16LE(20, 4); central.writeUInt16LE(20, 6); central.writeUInt16LE(deflate ? 8 : 0, 10);
    central.writeUInt32LE(crc32(data), 16); central.writeUInt32LE(body.length, 20); central.writeUInt32LE(data.length, 24); central.writeUInt16LE(nameBuf.length, 28); central.writeUInt32LE(offset, 42);
    locals.push(local, nameBuf, body); centrals.push(central, nameBuf);
    offset += 30 + nameBuf.length + body.length;
  }
  const cd = Buffer.concat(centrals);
  const end = Buffer.alloc(22); end.writeUInt32LE(0x06054b50, 0); end.writeUInt16LE(entries.length, 8); end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(cd.length, 12); end.writeUInt32LE(offset, 16);
  return Buffer.concat([...locals, cd, end]);
}
for (const deflate of [false, true]) {
  const archive = zip([['deviation_m.npy', npy('<f8', [2, 3], f8)], ['ids.npy', npy('<U2', [2], u)]], deflate);
  const npz = await api.parseNpz(new Uint8Array(archive));
  check(`parseNpz ${deflate ? 'deflated' : 'stored'}`, npz.deviation_m && npz.deviation_m.data[4] === 5 && npz.ids.data[0] === 'n1');
}
const header = { record_bytes: 8 + 12 + 4 + 4 + 12 + 12, byte_order: 'little', tick_s: 1e-6, axes: [{ name: 'X' }, { name: 'Y' }, { name: 'Z' }],
  fields: [{ name: 'tick', dtype: '<i8', shape: [], offset: 0 }, { name: 'axis', dtype: '<f4', shape: [3], offset: 8 },
    { name: 'spindle_angle_rad', dtype: '<f4', shape: [1], offset: 20 }, { name: 'spindle_rev', dtype: '<i4', shape: [1], offset: 24 },
    { name: 'tip_S_m', dtype: '<f4', shape: [1, 3], offset: 28 }, { name: 'axis_T_S', dtype: '<f4', shape: [1, 3], offset: 40 }] };
const records = Buffer.alloc(header.record_bytes * 2);
for (let r = 0; r < 2; r++) {
  const o = r * header.record_bytes;
  records.writeBigInt64LE(BigInt(1000 * r), o);
  [0.01, 0.02, 0.03].forEach((v, i) => records.writeFloatLE(v + r, o + 8 + 4 * i));
  records.writeFloatLE(1.5, o + 20); records.writeInt32LE(-r, o + 24);
  [0.1, 0.2, 0.3].forEach((v, i) => records.writeFloatLE(v, o + 28 + 4 * i));
  [0, 0, 1].forEach((v, i) => records.writeFloatLE(v, o + 40 + 4 * i));
}
const frames = api.decodeFrames(records.buffer.slice(records.byteOffset, records.byteOffset + records.length), header);
check('decodeFrames', frames.count === 2 && frames.tick[1] === 1000 && near(frames.time_s[1], 1e-3) && near(frames.fields.axis.data[3], 1.01, 1e-6)
  && frames.fields.tip_S_m.width === 3 && frames.fields.spindle_rev.data[1] === -1);
const doc = api.decodeArrayDoc({ dtype: 'float32', shape: [2, 2], data: [[1, null], [3, 4]] });
check('decodeArrayDoc flattens and maps null to NaN', doc.data.length === 4 && Number.isNaN(doc.data[1]) && doc.data[3] === 4);

// ------------------------------------------------------------------ plots: series building
const bucket = timeplot.bucketize([0, 0.1, 0.2, 0.9], [1, 5, 3, 2], 0, 1, 2);
check('bucketize min/max/mean', bucket.lo[0] === 1 && bucket.hi[0] === 5 && bucket.mean[0] === 3 && bucket.mean[1] === 2);
check('medianStep', timeplot.medianStep([0, 1, 2, 3, 5]) === 1);
const raw = {
  t0_s: 0, t1_s: 0.002, rows: 3, downsampled: false, buckets: null,
  series: ['cutting.force_N[0]', 'cutting.force_N[1]'].flatMap((col, k) => ['min', 'mean', 'max'].map((s, j) => ({
    column: `${col}.${s}`, channel: 'cutting.force_N', unit: 'N', n: 3, t_s: [0, 0.001, 0.002], v: [10 * k + j, 10 * k + j + 1, null] }))),
};
const forceChannel = { name: 'cutting.force_N', unit: 'N', tier: 'fast', doc: 'cutting force on the tool in T0 (x, y, z)', max_shape: [3] };
const built = plots.buildSeries(raw, forceChannel);
check('envelopes become bands with a mean line', built.series.length === 2 && built.series[0].lo && built.series[0].hi && built.series[0].v);
check('component labels from the channel doc', built.series[0].label === 'x' && built.series[1].label === 'y', built.series.map((s) => s.label).join());
check('colour follows the column', built.series[1].color === colormap.categorical(1));
const wide = { t0_s: 0, t1_s: 1, rows: 2, downsampled: false, series: Array.from({ length: 12 }, (_, i) => ({ column: `wear.VB_m[${i}]`, unit: 'm', t_s: [0, 1], v: [1e-5 * i, 2e-5 * i + 1e-6] })) };
const folded = plots.buildSeries(wide, { name: 'wear.VB_m', unit: 'm', tier: 'macro', doc: 'flank wear land width per tooth' }, { buckets: 4 });
check('more than eight columns fold into one band', folded.folded && folded.series.length === 1 && folded.columns === 12);
check('display unit chosen by size (µm)', folded.unit.label === 'µm');
const padded = { t0_s: 0, t1_s: 1, rows: 2, downsampled: false, series: [{ column: 'wear.VB_m[0]', t_s: [0, 1], v: [1e-5, 2e-5] }, { column: 'wear.VB_m[1]', t_s: [0, 1], v: [null, null] }] };
check('padding columns (all NaN) are left out', plots.buildSeries(padded, { name: 'wear.VB_m', unit: 'm', tier: 'macro', doc: 'per tooth' }).series.length === 1);
const m4 = { t0_s: 0, t1_s: 1, rows: 1000, downsampled: true, buckets: 2, series: [
  { column: 'x.y_N.min', t_s: [0, 0.2, 0.6, 0.9], v: [1, 0, 2, 3] }, { column: 'x.y_N.max', t_s: [0.1, 0.3, 0.7], v: [5, 6, 4] },
  { column: 'x.y_N.mean', t_s: [0, 0.6], v: [2, 3] }] };
const fromM4 = plots.buildSeries(m4, { name: 'x.y_N', unit: 'N', tier: 'fast' });
check('M4 envelopes are re-bucketed on a common grid', fromM4.series[0].t.length === 2 && fromM4.series[0].lo[0] === 0 && fromM4.series[0].hi[0] === 6);
check('fractions shown in %', plots.buildSeries({ t0_s: 0, t1_s: 1, rows: 1, series: [{ column: 'cutting.spindle_load_fraction', t_s: [0], v: [0.5] }] },
  { name: 'cutting.spindle_load_fraction', unit: '-', tier: 'tick' }).series[0].v[0] === 50);

// ------------------------------------------------------------------ run data index
const data = new rundata.RunData(client, 'r1');
data.setTimeline({ tick_s: 1e-6, blocks: [
  { index: 0, op: 'OP10', program: 'p.nc', line: 2, block: 'G0', path: 0, kind: 'rapid', tick0: 0, tick1: 1000 },
  { index: 1, op: 'OP10', program: 'p.nc', line: 3, block: 'G1', path: 0, kind: 'linear', tick0: 1000, tick1: 5000, ticks: { cutting: 3000, air_cutting: 1000 } },
  { index: 2, op: 'OP10', program: 'p.nc', line: 2, block: 'G0', path: 0, kind: 'rapid', tick0: 5000, tick1: 6000 },
  { index: 3, op: 'OP10', program: 'q.nc', line: 1, block: 'M30', path: 1, kind: 'stop', tick0: 0, tick1: 6000 }] });
check('blockAt by time and path', data.blockAt(0.0015).line === 3 && data.blockAt(0.0055).index === 2 && data.blockAt(0.001, 1).program === 'q.nc');
check('stepBlock forward and back', near(data.stepBlock(0.0015, 1), 0.005) && near(data.stepBlock(0.0015, -1), 0.001) && near(data.stepBlock(0.001, -1), 0));
const stat = data.lineStat('p.nc', 2);
check('time per line sums its executions', stat.ticks === 2000 && stat.executions === 2 && stat.starts.join() === '0,5000');
check('duration is the timeline end', near(data.duration(), 0.006));
check('paths listed', data.pathList.join() === '0,1');
data.setFindings([{ id: 1, time_s: 0.002, severity: 'error', program: 'p.nc', line: 3 }, { id: 0, time_s: 0.001, severity: 'info', program: 'p.nc', line: 3 }, { id: 1, time_s: 0.002, severity: 'error', program: 'p.nc', line: 3 }]);
check('findings are kept once each, in time order, by line', data.findings.length === 2 && data.findings[0].id === 0 && data.findingsByLine.get(rundata.lineKey('p.nc', 3)).length === 2);
check('worst severity', data.worstSeverity() === 'error');
check('block category from its ticks, else its kind', transport.blockCategory({ ticks: { cutting: 3, air_cutting: 1 } }) === 'cutting'
  && transport.blockCategory({ kind: 'rapid' }) === 'rapid' && transport.blockCategory({ kind: 'spindle' }) === 'spindle_ramp');

// ------------------------------------------------------------------ inspection judgement and grids
const size = results.judge({ nominal_m: 0.002, plus_m: 5e-5, minus_m: 5e-5, measured_m: 0.002062 });
check('a size out of tolerance', size.pass === false && near(size.out, 1.2e-5, 1e-6) && near(size.used, 1.24, 1e-6) && near(size.deviation, 6.2e-5, 1e-6));
const zone = results.judge({ tolerance_m: 2e-5, measured_m: 8e-6, pass: true });
check('a tolerance zone', zone.kind === 'zone' && zone.pass && near(zone.used, 0.4));
check('limits as lower/upper', results.judge({ lower_m: 1, upper_m: 2, measured_m: 2.5 }).out === 0.5);
const arrays = { surface_temp_K: { dtype: 'float32', shape: [2, 3], data: [[1, 2, 3], [4, 5, 6]] }, surface_origin_S_m: { dtype: 'float64', shape: [3], data: [0, 0, 0] },
  surface_cell_m: { dtype: 'float64', shape: [], data: 0.01 } };
const asYX = fields.gridFromArrays(arrays, 'surface_temp_K', ['surface'], [0.03, 0.02, 0.01]);
check('grid read as [y, x] when only that fits the stock', asYX.nx === 3 && asYX.ny === 2 && asYX.values[1] === 2 && asYX.orientation === 'values[y, x]');
const asXY = fields.gridFromArrays(arrays, 'surface_temp_K', ['surface'], [0.02, 0.03, 0.01]);
check('grid read as [x, y] by the core contract', asXY.nx === 2 && asXY.ny === 3 && asXY.values[1] === 4 && asXY.orientation === 'values[x, y]');
const cube = { t_K: { dtype: 'float32', shape: [2, 1, 2], data: [[[1, 7]], [[3, 2]]] }, t_origin_S_m: { dtype: 'float64', shape: [3], data: [0, 0, 0] }, t_cell_m: { dtype: 'float64', shape: [], data: 1 } };
const reduced = fields.gridFromArrays(cube, 't_K', ['t'], null);
check('3-D grids reduce to the maximum through z', reduced.values[0] === 7 && reduced.values[1] === 3, Array.from(reduced.values).join());
check('stats', fields.stats([1, 2, 3, NaN]).mean === 2 && fields.stats([1, 3]).range === 2);

fs.rmSync(copy, { recursive: true, force: true });
console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
