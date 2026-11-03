/* Production: cycle time broken down by category in exact integer ticks (and whether they sum to
 * the machine time), per operation, per tool and per program line; takt, parts per hour and the
 * operator times; electrical energy by consumer; coolant, air and tool cost per part; the batch and
 * the model's assumptions (production.json, docs/models/production.md section 9). */

import { h, section, table, notice, errorBox, figure, kv, tiles, humanize } from './dom.js';
import { stackedBar, barList } from '../plot/bars.js';
import { XYPlot } from '../plot/charts.js';
import { categorical, chrome } from '../plot/colormap.js';
import { categoryColor } from './transport.js';
import { lineLinks } from './results.js';
import { DASH, num, fmt, duration, pct, asNumber } from '../plot/units.js';

function breakdownRows(breakdown, tickS) {
  const ticks = (breakdown && breakdown.ticks) || {};
  return Object.entries(ticks).map(([category, value]) => ({ category, ticks: value, s: value * tickS }))
    .sort((a, b) => b.ticks - a.ticks);
}

function breakdownFigure(title, breakdown, tickS) {
  const rows = breakdownRows(breakdown, tickS);
  const total = Number(breakdown && breakdown.total_ticks) || rows.reduce((a, r) => a + r.ticks, 0);
  const sum = rows.reduce((a, r) => a + r.ticks, 0);
  const c = chrome();
  return figure({
    title, subtitle: `${duration(total * tickS)} · ${num(total)} ticks`,
    chart: (box) => box.append(stackedBar(rows.map((r) => ({ label: humanize(r.category), value: r.ticks, text: duration(r.s), color: categoryColor(r.category, c) })), { label: title, total })),
    tableView: () => table({
      columns: [{ key: 'category', label: 'category', render: (r) => humanize(r.category) }, { key: 'ticks', label: 'ticks', num: true, render: (r) => String(r.ticks) },
        { key: 's', label: 'time (s)', num: true, render: (r) => num(r.s, 6) }, { key: 'share', label: 'share', num: true, value: (r) => r.ticks, render: (r) => pct(r.ticks / (total || 1), 2) }],
      rows,
    }),
    note: sum === total ? 'The categories sum to the machine time exactly (integer ticks).' : `The categories sum to ${num(sum)} ticks of ${num(total)} (${num(total - sum)} unclassified).`,
  });
}

export async function renderProduction(root, { api, data, page }) {
  if (!data.has('production.json')) { root.append(section('Production', notice('This run directory has no production.json (not recorded, or the run has not finished).'))); return; }
  let doc;
  try { doc = await api.production(data.id); } catch (error) { root.append(section('Production', errorBox(error))); return; }
  const tickS = Number(doc.tick_s) || data.tickS;
  const first = doc.first_part || {};
  const cycle = first.cycle || {};
  const part = Array.isArray(doc.per_part) ? doc.per_part[0] : null;
  const batch = doc.batch || {};
  root.append(section(null, tiles([
    { label: 'Machine time', value: duration(cycle.machine_s ?? doc.cycle_time_s) },
    { label: 'Takt', value: duration(batch.takt_mean_s ?? doc.per_part_s), sub: 'machine time plus operator time per part' },
    { label: 'Parts per hour', value: num(batch.parts_per_hour, 4) },
    { label: 'Machine utilisation', value: pct(batch.machine_utilisation_frac, 1) },
    { label: 'Energy per part', value: fmt.kWh(batch.energy_per_part_J ?? doc.energy_J), sub: fmt.kJ(batch.energy_per_part_J ?? doc.energy_J) },
    { label: 'Tool cost per part', value: fmt.eur(batch.tool_cost_per_part_eur) },
    { label: 'Chip mass per part', value: fmt.g(part ? part.removed_mass_kg : doc.removed_mass_kg) },
    { label: 'Parts', value: `${num(doc.parts_simulated)} simulated`, sub: `${num(doc.parts_planned)} planned, ${num(doc.parts_projected)} projected` },
  ])));
  const cycleBox = section('Cycle time');
  cycleBox.append(breakdownFigure('First part by category', cycle.breakdown, tickS));
  root.append(cycleBox);
  const ops = Array.isArray(cycle.operations) ? cycle.operations : [];
  if (ops.length) {
    root.append(section('Per operation', table({
      columns: [{ key: 'op', label: 'operation' }, { key: 'span_s', label: 'machine time', num: true, render: (o) => duration(o.span_s ?? o.span_ticks * tickS) },
        { key: 'critical_path', label: 'longest path', num: true, render: (o) => (o.critical_path === undefined ? DASH : `path ${o.critical_path + 1}`) },
        { key: 'cut', label: 'cutting', num: true, value: (o) => (o.breakdown && o.breakdown.ticks && o.breakdown.ticks.cutting) || 0, render: (o) => duration(((o.breakdown && o.breakdown.ticks && o.breakdown.ticks.cutting) || 0) * tickS) },
        { key: 'paths', label: 'paths', num: true, render: (o) => String((o.paths || []).length) }],
      rows: ops,
    })));
    for (const o of ops) if (ops.length > 1) cycleBox.append(breakdownFigure(`${o.op} by category`, o.breakdown, tickS));
  }
  const tools = Array.isArray(cycle.tools) ? cycle.tools : [];
  if (tools.length) {
    root.append(section('Per tool', table({
      columns: [{ key: 'tool', label: 'tool', render: (t) => (t.tool === null || t.tool === undefined ? 'no tool' : `T${t.tool}`) },
        { key: 'total', label: 'time', num: true, value: (t) => t.breakdown && t.breakdown.total_ticks, render: (t) => duration((t.breakdown && t.breakdown.total_ticks || 0) * tickS) },
        { key: 'cut', label: 'cutting', num: true, value: (t) => (t.breakdown && t.breakdown.ticks && t.breakdown.ticks.cutting) || 0, render: (t) => duration(((t.breakdown && t.breakdown.ticks && t.breakdown.ticks.cutting) || 0) * tickS) },
        { key: 'share', label: 'share', num: true, render: (t) => pct((t.breakdown && t.breakdown.total_ticks || 0) / ((cycle.breakdown && cycle.breakdown.total_ticks) || 1), 1) }],
      rows: tools,
    })));
  }
  const lines = (Array.isArray(first.lines) ? first.lines : []).map((l) => ({ ...l, total: (l.breakdown && l.breakdown.total_ticks) || 0 }))
    .sort((a, b) => b.total - a.total).slice(0, 30);
  root.append(section('Longest program lines', table({
    columns: [{ key: 'where', label: 'line', sortable: false, render: (l) => lineLinks(page, data, l.program, l.line) },
      { key: 'block', label: 'block', render: (l) => h('code', null, l.block || '') },
      { key: 'executions', label: 'runs', num: true },
      { key: 'total', label: 'time', num: true, render: (l) => duration(l.total * tickS) },
      { key: 'main', label: 'mostly', sortable: false, render: (l) => { const r = breakdownRows(l.breakdown, tickS)[0]; return r ? humanize(r.category) : DASH; } }],
    rows: lines, empty: 'production.json lists no program lines.',
  })));
  if (part && part.operator) {
    const op = part.operator;
    root.append(section('Operator', kv([
      ...Object.entries(op.components_s || {}).map(([k, v]) => [humanize(k.replace(/_s$/, '')), duration(v)]),
      ['operator time', duration(op.operator_s)], ['machine waiting', duration(op.machine_wait_s)], ['takt', duration(op.takt_s)],
      ['M00 stops', num(op.m00)], ['M01 stops', num(op.m01)], ['inspected', op.inspected ? 'yes' : 'no']])));
  }
  if (cycle.stops) root.append(section('Program stops', kv(Object.entries(cycle.stops).map(([k, v]) => [k, num(v)]))));
  const energy = part && part.energy;
  const consumers = batch.energy_by_consumer_J || {};
  if (Object.keys(consumers).length || energy) {
    const box = section('Energy');
    const entries = Object.entries(consumers).sort((a, b) => b[1] - a[1]);
    box.append(figure({
      title: 'Electrical energy by consumer', subtitle: `${fmt.kWh(batch.energy_J)} for the batch, ${fmt.kWh(batch.energy_per_part_J)} per part`,
      chart: (b) => b.append(barList(entries.map(([k, v]) => ({ label: humanize(k), value: v, text: `${fmt.kJ(v)} · ${fmt.kWh(v)}`, color: categorical(0) })), { label: 'energy by consumer' })),
      tableView: () => table({ columns: [{ key: 'k', label: 'consumer', render: (r) => humanize(r.k) }, { key: 'v', label: 'energy (kJ)', num: true, render: (r) => num(r.v / 1000, 5) }, { key: 's', label: 'share', num: true, render: (r) => pct(r.v / (batch.energy_J || 1), 1) }],
        rows: entries.map(([k, v]) => ({ k, v })) }),
    }));
    if (energy && energy.spindle) {
      box.append(h('h3', null, 'Spindle'), kv(Object.entries(energy.spindle).map(([k, v]) => [humanize(k.replace(/_J$/, '')), fmt.kJ(v)])));
    }
    if (energy && energy.axes && energy.axes.axes) {
      const rows = Object.entries(energy.axes.axes).map(([axis, v]) => ({ axis, ...v }));
      box.append(h('h3', null, 'Axes'), table({
        columns: [{ key: 'axis', label: 'axis' }, ...['inertial_J', 'gravity_J', 'friction_J', 'process_J', 'mechanical_J', 'electrical_J']
          .map((k) => ({ key: k, label: `${humanize(k.replace(/_J$/, ''))} (J)`, num: true, render: (r) => num(r[k], 4) }))],
        rows,
      }));
    }
    if (energy) {
      box.append(h('h3', null, 'Other consumers'), kv([
        ['base load', fmt.kJ(energy.base_J)], ['servo standby', fmt.kJ(energy.servo_standby_J)], ['compressed air', fmt.kJ(energy.air_J)],
        ...Object.entries(energy.pumps || {}).map(([id, p]) => [`pump ${id}`, `${fmt.kJ(p.electrical_J)} (${duration(p.on_s)} on at ${num(p.power_W, 4)} W)`]),
        ['chip conveyor', energy.conveyor ? `${fmt.kJ(energy.conveyor.electrical_J)} (${duration(energy.conveyor.on_s)} on)` : DASH],
        ['not modelled', (energy.missing || []).join(', ') || 'nothing']]));
    }
    root.append(box);
  }
  const consumables = part && part.consumables;
  if (consumables) {
    const coolant = consumables.coolant || {};
    const air = consumables.air || {};
    const ml = (m3) => (asNumber(m3) === null ? DASH : `${num(m3 * 1e6, 4)} ml`);
    root.append(section('Consumables per part', kv([
      ['coolant pumped', fmt.litres(coolant.pumped_m3)], ['drag-out on chips', ml(coolant.drag_out_m3)], ['mist', ml(coolant.mist_m3)],
      ['evaporated water', ml(coolant.evaporated_water_m3)], ['concentrate', ml(coolant.concentrate_m3)], ['water', ml(coolant.water_m3)], ['oil', ml(coolant.oil_m3)],
      ['coolant cost', fmt.eur(coolant.cost_eur)], ['compressed air', `${fmt.litres(air.total_m3)} (machine ${fmt.litres(air.machine_m3)}, air blast ${fmt.litres(air.air_blast_m3)}, MQL ${fmt.litres(air.mql_m3)})`],
      ...Object.entries(consumables.pump_on_s || {}).map(([id, s]) => [`${id} pump on`, duration(s)]),
      ['air blast on', duration(consumables.air_blast_on_s)], ['MQL on', duration(consumables.mql_on_s)],
      ['not modelled', (coolant.missing || []).join(', ') || 'nothing']])));
  }
  const toolCosts = (part && part.tools) || [];
  if (toolCosts.length) {
    root.append(section('Tool cost per part', table({
      columns: [{ key: 'number', label: 'tool', render: (t) => `T${t.number} ${t.tool_id || ''}` },
        { key: 'cut_time_s', label: 'cutting', num: true, render: (t) => duration(t.cut_time_s) },
        { key: 'life', label: 'life used per part', num: true, value: (t) => t.life_used_part_frac, render: (t) => pct(t.life_used_part_frac, 4) },
        { key: 'edge', label: 'cost per edge', num: true, value: (t) => t.cost && t.cost.cost_per_edge_eur, render: (t) => fmt.eur(t.cost && t.cost.cost_per_edge_eur) },
        { key: 'ppe', label: 'parts per edge', num: true, value: (t) => t.cost && t.cost.parts_per_edge, render: (t) => num(t.cost && t.cost.parts_per_edge, 4) },
        { key: 'cpp', label: 'cost per part', num: true, value: (t) => t.cost && t.cost.cost_per_part_eur, render: (t) => fmt.eur(t.cost && t.cost.cost_per_part_eur) }],
      rows: toolCosts,
    })));
  }
  root.append(section('Batch', kv([
    ['parts', num(batch.parts)], ['machine time', duration(batch.machine_s)], ['total time', duration(batch.total_s)], ['mean takt', duration(batch.takt_mean_s)],
    ['parts per hour', num(batch.parts_per_hour, 4)], ['machine utilisation', pct(batch.machine_utilisation_frac, 1)], ['energy', fmt.kWh(batch.energy_J)],
    ['tool cost', fmt.eur(batch.tool_cost_eur)], ['tool changes', num(batch.tool_changes)]])));
  const progression = doc.progression || {};
  if (Array.isArray(progression.part) && progression.part.length > 1) {
    const c = chrome();
    root.append(section('Part by part', figure({
      title: 'Takt per part', subtitle: 'simulated parts filled, projected parts hollow',
      chart: (box) => new XYPlot(box, { xLabel: 'part', yLabel: 'takt (s)', height: 220 }).setData({
        lines: [{ x: progression.part.map((p) => p + 1), y: progression.takt_s, color: c.series[0], label: 'takt' }],
        points: progression.part.map((p, i) => ({ x: p + 1, y: progression.takt_s[i], color: c.series[0], hollow: !progression.simulated[i] })),
      }),
      tableView: () => table({ columns: [{ key: 'part', label: 'part', num: true }, { key: 'sim', label: 'simulated' }, { key: 'takt', label: 'takt (s)', num: true },
        { key: 'energy', label: 'energy (kJ)', num: true }, { key: 'cost', label: 'tool cost', num: true }],
      rows: progression.part.map((p, i) => ({ part: p + 1, sim: progression.simulated[i] ? 'yes' : 'projected', takt: num(progression.takt_s[i]),
        energy: num((progression.energy_J || [])[i] / 1000, 5), cost: fmt.eur((progression.tool_cost_eur || [])[i]) })) }),
    })));
  }
  const assumptions = Array.isArray(doc.assumptions) ? doc.assumptions : [];
  root.append(section('Assumptions', assumptions.length ? h('ul', { class: 'plain' }, assumptions.map((a) => h('li', null, a))) : notice('production.json lists no assumptions.')));
}
