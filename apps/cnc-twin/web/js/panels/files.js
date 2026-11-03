/* Reports and files of a run: the report and setup sheet (HTML, PDF), the video and the sound of
 * the cut, the pictures, the machined part (GLB, STL, 3MF, unloaded STL), the full-resolution
 * series exports (CSV, Parquet; every column or chosen channels) and every file of the run. */

import { h, section, table, notice, link, button, figure, errorBox, kv } from './dom.js';
import { XYPlot } from '../plot/charts.js';
import { chrome, withAlpha } from '../plot/colormap.js';
import { bytes, fixed, num } from '../plot/units.js';

/** The tool path from frames.bin over the stock, seen from above (x, y) and from the front (x, z), in
 * the stock frame (mm): feed moves in the first colour, rapids grey (as the report draws it). */
async function toolPath(api, data) {
  const box = h('div');
  if (!data.scene || !data.scene.frames) { box.append(notice('No frames header in scene.json: no tool path.')); return box; }
  let frames;
  try { frames = await api.frames(data.id, { maxFrames: 6000 }); } catch (error) { box.append(errorBox(error, 'Frames: ')); return box; }
  if (!frames.count || !frames.fields.tip_S_m) { box.append(notice('frames.bin holds no records yet.')); return box; }
  const tip = frames.fields.tip_S_m;
  const c = chrome();
  const view = (a, b) => {
    const feed = { x: [], y: [], color: c.series[0], label: 'feed', width: 1.5 };
    const rapid = { x: [], y: [], color: withAlpha(c.ink2, 0.55), label: 'rapid', width: 1 };
    let last = null;
    for (let i = 0; i < frames.count; i++) {
      const block = data.blockAt(frames.time_s[i], data.pathList[0] || 0);
      const kind = block && block.kind === 'rapid' ? 'rapid' : 'feed';
      const pa = tip.data[i * tip.width + a] * 1e3;
      const pb = tip.data[i * tip.width + b] * 1e3;
      const own = kind === 'rapid' ? rapid : feed;
      const other = kind === 'rapid' ? feed : rapid;
      if (last && last.kind !== kind) { own.x.push(last.a); own.y.push(last.b); other.x.push(NaN); other.y.push(NaN); }
      own.x.push(pa); own.y.push(pb);
      last = { kind, a: pa, b: pb };
    }
    const size = (data.scene.stock && data.scene.stock.size_m) || null;
    const lines = [rapid, feed];
    if (size) {
      const [w, d] = [size[a] * 1e3, size[b] * 1e3];
      lines.unshift({ x: [0, w, w, 0, 0], y: [0, 0, d, d, 0], color: withAlpha(c.ink, 0.45), label: 'stock', width: 1 });
    }
    const zero = Array.isArray(data.scene.work_zero_S_m) ? data.scene.work_zero_S_m : null;
    const points = zero ? [{ x: zero[a] * 1e3, y: zero[b] * 1e3, color: c.series[1], text: 'work zero', r: 4 }] : [];
    return { lines, points };
  };
  const axes = ['x', 'y', 'z'];
  for (const [a, b, title] of [[0, 1, 'from above'], [0, 2, 'from the front']]) {
    box.append(figure({
      title: `Tool path ${title}`, subtitle: `${axes[a]} and ${axes[b]} in the stock frame (mm), ${num(frames.count)} of the frames`,
      chart: (el) => new XYPlot(el, { xLabel: `${axes[a]} (mm)`, yLabel: `${axes[b]} (mm)`, height: 300, equalAspect: true, label: `tool path ${title}` }).setData(view(a, b)),
      tableView: () => kv([['frames read', `${num(frames.count)} (every ${frames.stride}th L1 tick)`], ['first tip', [0, 1, 2].map((k) => fixed(tip.data[k] * 1e3, 3)).join(', ') + ' mm'],
        ['last tip', [0, 1, 2].map((k) => fixed(tip.data[(frames.count - 1) * tip.width + k] * 1e3, 3)).join(', ') + ' mm']]),
    }));
  }
  return box;
}

const REPORTS = [
  ['report.html', 'Report (HTML)'], ['report.pdf', 'Report (PDF)'], ['setup_sheet.html', 'Setup sheet (HTML)'], ['setup_sheet.pdf', 'Setup sheet (PDF)'],
  ['uncertainty/uncertainty.html', 'Uncertainty study (HTML)'], ['compare.html', 'Comparison (HTML)'],
];

export async function renderFiles(root, { api, data }) {
  const files = data.files;
  const present = REPORTS.filter(([path]) => files.has(path));
  root.append(section('Reports', present.length
    ? h('div', { class: 'link-grid' }, present.map(([path, label]) => link(api.fileUrl(data.id, path), label, { kind: 'tile-link', newTab: true, iconName: path.endsWith('.pdf') ? 'download' : 'external' })))
    : notice(data.finished ? 'This run wrote no report (outputs.report in the project).' : 'Reports are written when the run ends.')));
  const media = section('Video and sound');
  if (files.has('video.mp4')) {
    media.append(h('video', { class: 'media', controls: true, preload: 'metadata', playsinline: true, src: api.fileUrl(data.id, 'video.mp4') }),
      link(api.fileUrl(data.id, 'video.mp4'), 'Download the video', { download: 'video.mp4', kind: 'ghost small', iconName: 'download' }));
  }
  if (files.has('audio.wav')) {
    media.append(h('audio', { class: 'media-audio', controls: true, preload: 'metadata', src: api.fileUrl(data.id, 'audio.wav') }),
      link(api.fileUrl(data.id, 'audio.wav'), 'Download the sound', { download: 'audio.wav', kind: 'ghost small', iconName: 'download' }));
  }
  if (!files.has('video.mp4') && !files.has('audio.wav')) media.append(notice('No video.mp4 or audio.wav in this run.'));
  root.append(media);
  const images = [...files.keys()].filter((p) => /^images\/.+\.(png|jpe?g|gif|webp|svg)$/i.test(p)).sort();
  root.append(section('Pictures', images.length
    ? h('div', { class: 'gallery' }, images.map((p) => h('a', { href: api.fileUrl(data.id, p), target: '_blank', rel: 'noopener', class: 'shot' },
      h('img', { src: api.fileUrl(data.id, p), alt: p.replace(/^images\//, ''), loading: 'lazy' }), h('span', null, p.replace(/^images\//, '')))))
    : notice('No pictures (images/).'), await toolPath(api, data)));
  const parts = [...files.keys()].filter((p) => /^part_.+\.(glb|stl|3mf|obj|step)$/i.test(p)).sort();
  root.append(section('Machined part', parts.length
    ? h('div', { class: 'link-grid' }, parts.map((p) => link(api.fileUrl(data.id, p), `${p} · ${bytes(files.get(p))}`, { download: p, kind: 'tile-link', iconName: 'download' })))
    : notice('No part files yet (written at each operation end).')));
  const exportBox = section('Series exports (full resolution)');
  for (const tier of ['tick', 'macro', 'fast']) {
    const channels = data.channels.filter((c) => c.tier === tier || (tier === 'tick' && c.tier === 'fast'));
    if (!channels.length) continue;
    const select = h('select', { class: 'input', multiple: true, size: String(Math.min(6, channels.length)), 'aria-label': `${tier} channels to export` },
      channels.map((c) => h('option', { value: c.name }, `${c.name}${c.tier !== tier ? ' (envelopes)' : ''}`)));
    const csv = link(api.exportUrl(data.id, { format: 'csv', tier }), 'CSV', { kind: 'btn small', download: `${data.id}-${tier}.csv`, iconName: 'download' });
    const parquet = link(api.exportUrl(data.id, { format: 'parquet', tier }), 'Parquet', { kind: 'btn small', download: `${data.id}-${tier}.parquet`, iconName: 'download' });
    const refresh = () => {
      const chosen = [...select.selectedOptions].map((o) => o.value);
      csv.href = api.exportUrl(data.id, { format: 'csv', tier, channels: chosen });
      parquet.href = api.exportUrl(data.id, { format: 'parquet', tier, channels: chosen });
      label.textContent = chosen.length ? `${chosen.length} channel(s)` : 'every column';
    };
    const label = h('span', { class: 'muted small' }, 'every column');
    select.addEventListener('change', refresh);
    exportBox.append(h('div', { class: 'export-row' }, h('b', null, `${tier} tier`), csv, parquet, label,
      h('details', { class: 'disclosure' }, h('summary', null, 'Choose channels'), select, button('All columns', () => { for (const o of select.options) o.selected = false; refresh(); }, { kind: 'small ghost' }))));
  }
  if (!data.channels.length) exportBox.append(notice('No channels recorded.'));
  root.append(exportBox);
  const all = [...files.entries()].map(([path, size]) => ({ path, size }));
  root.append(section(`Every file (${all.length})`, table({
    columns: [{ key: 'path', label: 'file', render: (r) => link(api.fileUrl(data.id, r.path), r.path, { newTab: !/\.(npz|parquet|bin|glb|stl|3mf|idx|jsonl)$/.test(r.path) }) },
      { key: 'size', label: 'size', num: true, render: (r) => bytes(r.size) }],
    rows: all, maxRows: 200, empty: 'No files listed.',
  })));
}
