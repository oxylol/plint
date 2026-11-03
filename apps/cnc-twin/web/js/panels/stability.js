/* Chatter and stability: the stability lobe diagram of every tool (sld/<tool>.json: the limiting
 * depth of cut over spindle speed at the stated radial immersion) with the program's operating
 * points, stable ones filled and unstable ones hollow, and the chatter findings. */

import { h, section, table, notice, errorBox, figure, badge, humanize, kv } from './dom.js';
import { XYPlot } from '../plot/charts.js';
import { chrome } from '../plot/colormap.js';
import { lineLinks } from './results.js';
import { DASH, num, fmt, fixed } from '../plot/units.js';

const RPM = 60 / (2 * Math.PI);

export async function renderStability(root, { api, data, page }) {
  let list;
  try { list = await api.sldList(data.id); } catch (error) { root.append(section('Stability lobes', errorBox(error))); return; }
  const tools = list.tools || [];
  if (!tools.length) root.append(section('Stability lobes', notice('This run has no stability lobe diagrams (sld/).')));
  for (const tool of tools) {
    const box = section(`Stability lobes of ${tool}`);
    root.append(box);
    let doc;
    try { doc = await api.sld(data.id, tool); } catch (error) { box.append(errorBox(error)); continue; }
    const speed = (doc.speed_rad_s || []).map((v) => Math.abs(v) * RPM);
    const depth = (doc.depth_limit_m || []).map((v) => (v === null ? NaN : v * 1e3));
    const points = doc.operating_points || [];
    const c = chrome();
    box.append(kv([['radial immersion', doc.radial_immersion_m !== undefined ? fmt.mm(doc.radial_immersion_m, 3) : DASH],
      ['tool', `${doc.tool || tool}${doc.number !== undefined ? ` (T${doc.number})` : ''}`], ['operating points', String(points.length)],
      ['unstable', String(points.filter((p) => p.stable === false).length)]]));
    box.append(figure({
      title: 'Limiting depth of cut over spindle speed', subtitle: 'stable below the line',
      chart: (b) => new XYPlot(b, { xLabel: 'spindle speed (rpm)', yLabel: 'depth limit (mm)', height: 280, label: `stability lobes of ${tool}` }).setData({
        lines: [{ x: speed, y: depth, color: c.series[0], fill: true, label: 'stability limit' }],
        points: points.map((p) => {
          const stable = p.stable !== false;
          const x = Math.abs(Number(p.speed_rad_s)) * RPM;
          const y = Number(p.ap_m) * 1e3;
          return {
            x, y, color: stable ? c.series[0] : c.error, hollow: !stable, text: `line ${p.line}`,
            tip: [[stable ? c.series[0] : c.error, `${fixed(x, 0)} rpm · ap ${num(y)} mm`, `${p.op || ''} line ${p.line}: ${stable ? 'stable' : 'unstable'}`]],
          };
        }),
        includeZeroY: true,
      }),
      tableView: () => table({ columns: [{ key: 'rpm', label: 'speed (rpm)', num: true }, { key: 'mm', label: 'depth limit (mm)', num: true }],
        rows: speed.map((s, i) => ({ rpm: fixed(s, 0), mm: num(depth[i]) })), maxRows: 200 }),
    }));
    box.append(table({
      columns: [{ key: 'op', label: 'operation' }, { key: 'line', label: 'line', sortable: false, render: (p) => lineLinks(page, data, p.program, p.line) },
        { key: 'speed', label: 'speed (rpm)', num: true, value: (p) => Math.abs(p.speed_rad_s), render: (p) => fixed(Math.abs(p.speed_rad_s) * RPM, 0) },
        { key: 'ap', label: 'depth (mm)', num: true, value: (p) => p.ap_m, render: (p) => num(p.ap_m * 1e3) },
        { key: 'stable', label: 'result', render: (p) => (p.stable === false ? h('span', { class: 'badge tone-error' }, 'unstable') : h('span', { class: 'badge tone-ok' }, 'stable')) }],
      rows: points, empty: 'No operating points recorded.', caption: 'Operating points of the program',
    }));
  }
  const chatter = data.findings.filter((f) => f.category === 'chatter');
  root.append(section('Chatter findings', chatter.length ? h('div', null, chatter.map((f) => h('button', { type: 'button', class: 'finding-mini', on: { click: () => page.focusFinding(f) } },
    badge(f.severity), h('span', null, `${humanize(f.category)} at ${num(f.time_s)} s, line ${f.line}: ${f.message}${f.quantities && f.quantities.chatter_freq_Hz ? ` (${num(f.quantities.chatter_freq_Hz)} Hz)` : ''}`)))) : notice('No chatter findings.')));
}
