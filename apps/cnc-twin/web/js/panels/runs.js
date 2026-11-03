/* Runs: every run, newest first, with status, progress while running, verdict, cycle time and
 * findings; filters by project and status; pick two finished runs to compare. The list refreshes
 * every 2 s while a run is active. */

import { h, clear, section, table, errorBox, button, link, loading, statusBadge, verdictBadge } from './dom.js';
import { meter } from '../plot/bars.js';
import { clock, duration, pct } from '../plot/units.js';

export async function renderRuns(root, { api, navigate, query = {} }) {
  const project = h('select', { class: 'input', 'aria-label': 'Project' }, h('option', { value: '' }, 'all projects'));
  const status = h('select', { class: 'input', 'aria-label': 'Status' }, ['', 'queued', 'running', 'done', 'failed', 'cancelled'].map((s) => h('option', { value: s }, s || 'any status')));
  if (query.project) project.append(h('option', { value: query.project }, query.project));
  project.value = query.project || '';
  status.value = query.status || '';
  const picked = new Set();
  const compareBtn = button('Compare the two picked', () => { const [a, b] = [...picked]; navigate(`#/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`); }, { kind: 'primary', iconName: 'compare', disabled: true });
  const listBox = h('div');
  root.append(h('div', { class: 'page-head' }, h('h1', null, 'Runs')),
    h('div', { class: 'filters' }, project, status, compareBtn), listBox);
  try {
    const projects = await api.projects();
    for (const p of projects.projects || []) if (p.id !== query.project) project.append(h('option', { value: p.id }, p.name || p.id));
    project.value = query.project || '';
  } catch { /* the filter offers what the runs name */ }
  let timer = null;
  let alive = true;
  const draw = async () => {
    if (!alive) return;
    let doc;
    try { doc = await api.runs({ projectId: project.value || undefined, status: status.value || undefined, limit: 500 }); } catch (error) { clear(listBox); listBox.append(errorBox(error)); return; }
    if (!alive) return;
    const runs = doc.runs || [];
    clear(listBox);
    listBox.append(section(`${runs.length} run${runs.length === 1 ? '' : 's'}`, table({
      columns: [
        { key: 'pick', label: 'compare', sortable: false, render: (r) => {
          const box = h('input', { type: 'checkbox', checked: picked.has(r.id), disabled: !r.verdict, 'aria-label': `Pick ${r.id} to compare`,
            on: { change: () => { if (box.checked) picked.add(r.id); else picked.delete(r.id); while (picked.size > 2) picked.delete(picked.values().next().value); compareBtn.disabled = picked.size !== 2; draw(); } } });
          return box;
        } },
        { key: 'id', label: 'run', render: (r) => link(`#/runs/${encodeURIComponent(r.id)}/overview`, r.id) },
        { key: 'name', label: 'project', render: (r) => r.name || r.project_id || '' },
        { key: 'status', label: 'status', render: (r) => statusBadge(r.status) },
        { key: 'progress', label: 'progress', sortable: false, render: (r) => {
          const p = r.progress || {};
          if (!r.active) return p.sim_time_s !== undefined && r.status !== 'done' ? `${clock(p.sim_time_s)} simulated` : '';
          const frac = p.est_total_s ? (p.sim_time_s || 0) / p.est_total_s : null;
          return meter(frac, { text: frac === null ? `${clock(p.sim_time_s || 0)}` : `${pct(frac, 0)}${p.eta_s ? ` · ${duration(p.eta_s)} left` : ''}`, tone: 'ok', label: 'progress' });
        } },
        { key: 'verdict', label: 'verdict', value: (r) => (r.verdict ? (r.verdict.safe ? 1 : 0) : -1), render: (r) => verdictBadge(r.verdict) },
        { key: 'cycle', label: 'cycle time', num: true, value: (r) => r.summary && r.summary.cycle_time_s, render: (r) => (r.summary && r.summary.cycle_time_s !== undefined ? duration(r.summary.cycle_time_s) : '') },
        { key: 'findings', label: 'findings', num: true, sortable: false, render: (r) => {
          const f = (r.summary && r.summary.findings) || null;
          if (!f) return '';
          return ['fatal', 'error', 'warning', 'info'].filter((s) => f[s]).map((s) => `${f[s]} ${s}`).join(', ') || 'none';
        } },
      ],
      rows: runs, empty: 'No runs. Start one from a project.',
    })));
    clearTimeout(timer);
    if (runs.some((r) => r.active)) timer = setTimeout(draw, 2000);
  };
  project.addEventListener('change', () => { navigate(`#/runs${project.value || status.value ? `?${new URLSearchParams({ ...(project.value ? { project: project.value } : {}), ...(status.value ? { status: status.value } : {}) })}` : ''}`, { replace: true }); draw(); });
  status.addEventListener('change', () => { navigate(`#/runs${project.value || status.value ? `?${new URLSearchParams({ ...(project.value ? { project: project.value } : {}), ...(status.value ? { status: status.value } : {}) })}` : ''}`, { replace: true }); draw(); });
  listBox.append(loading());
  await draw();
  return { destroy: () => { alive = false; clearTimeout(timer); } };
}

