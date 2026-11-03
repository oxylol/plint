/* Overview of a run: the headline numbers (cycle time, findings, characteristics passed, removed
 * mass, energy and tool cost per part, takt), the findings by category and severity, and every
 * finding with its line, quantities and chain of cause. */

import { tiles, section, table, humanize, kv, notice } from './dom.js';
import { FindingsList } from './findings.js';
import { duration, fmt, quantity, num } from '../plot/units.js';

async function optional(promise) {
  try { return await promise; } catch { return null; }
}

export async function renderOverview(root, { api, data, page }) {
  const run = data.run || {};
  const summary = run.summary || {};
  const [inspection, production] = await Promise.all([
    data.has('inspection.json') ? optional(api.inspection(data.id)) : null,
    data.has('production.json') ? optional(api.production(data.id)) : null,
  ]);
  const counts = { fatal: 0, error: 0, warning: 0, info: 0 };
  for (const f of data.findings) counts[f.severity] = (counts[f.severity] || 0) + 1;
  const part = production && Array.isArray(production.per_part) ? production.per_part[0] : null;
  const batch = (production && production.batch) || {};
  const checked = inspection && inspection.summary ? inspection.summary.checked : null;
  const failed = inspection && inspection.summary ? inspection.summary.failed : null;
  const cycle = summary.cycle_time_s ?? (production && production.cycle_time_s) ?? (data.finished ? null : data.duration());
  root.append(section(null, tiles([
    { label: data.finished ? 'Cycle time' : 'Simulated so far', value: duration(cycle) },
    { label: 'Findings', value: String(data.findings.length), sub: ['fatal', 'error', 'warning', 'info'].filter((s) => counts[s]).map((s) => `${counts[s]} ${s}`).join(' · ') || 'none',
      tone: counts.fatal || counts.error ? 'error' : counts.warning ? 'warning' : 'ok' },
    { label: 'Characteristics passed', value: checked === null ? null : `${checked - (failed || 0)} of ${checked}`, tone: failed ? 'error' : checked ? 'ok' : null },
    { label: 'Removed mass', value: summary.removed_mass_kg !== undefined ? fmt.g(summary.removed_mass_kg) : null },
    { label: 'Energy per part', value: part && part.energy ? fmt.kWh(part.energy.total_J) : (batch.energy_per_part_J !== undefined ? fmt.kWh(batch.energy_per_part_J) : null),
      sub: part && part.energy ? fmt.kJ(part.energy.total_J) : null },
    { label: 'Tool cost per part', value: batch.tool_cost_per_part_eur !== undefined ? fmt.eur(batch.tool_cost_per_part_eur) : null },
    { label: 'Takt', value: batch.takt_mean_s !== undefined ? duration(batch.takt_mean_s) : null,
      sub: batch.parts_per_hour !== undefined ? `${num(batch.parts_per_hour, 3)} parts per hour` : null },
  ])));
  const byCategory = new Map();
  for (const f of data.findings) {
    if (!byCategory.has(f.category)) byCategory.set(f.category, { category: f.category, info: 0, warning: 0, error: 0, fatal: 0, total: 0, first: f });
    const row = byCategory.get(f.category);
    row[f.severity] = (row[f.severity] || 0) + 1;
    row.total++;
  }
  root.append(section('Findings by category', table({
    columns: [
      { key: 'category', label: 'category', render: (r) => humanize(r.category) },
      ...['fatal', 'error', 'warning', 'info'].map((s) => ({ key: s, label: s, num: true, render: (r) => (r[s] ? String(r[s]) : '') })),
      { key: 'total', label: 'total', num: true },
      { key: 'first', label: 'first at', num: true, value: (r) => r.first.time_s, render: (r) => `${num(r.first.time_s)} s` },
    ],
    rows: [...byCategory.values()].sort((a, b) => b.fatal - a.fatal || b.error - a.error || b.warning - a.warning || b.total - a.total),
    empty: 'No findings.',
    onRow: (r) => page.focusFinding(r.first),
  })));
  const holder = section('Findings');
  const list = new FindingsList(holder, {
    api, data,
    onJump: (f) => page.focusFinding(f),
    onEvent: (event) => page.seekTo(event.time_s, { line: event.line || null }),
  });
  root.append(holder);
  root.findingsList = list;
  const extra = Object.entries(summary).filter(([, v]) => typeof v === 'number' || typeof v === 'string');
  if (extra.length) {
    root.append(section('Run summary', kv(extra.map(([key, value]) => {
      const q = quantity(key, value);
      return [q.label, q.text];
    }))));
  }
  if (!data.finished) root.append(notice('The run has not finished: results appear when it does (this page updates by itself).'));
}
