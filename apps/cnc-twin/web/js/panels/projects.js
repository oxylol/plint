/* Projects: the list (examples, always read-only, then uploads) with the upload of a zipped project,
 * and a project's page: its files (programs and project.yaml readable in place), the check result
 * (twin check as a job), starting a run with options, and the project's runs. */

import { h, clear, section, table, notice, errorBox, kv, button, link, loading, toast, statusBadge, verdictBadge, jsonView, humanize, lazyDetails } from './dom.js';
import { bytes, duration, num } from '../plot/units.js';

const TEXT_FILES = /\.(nc|ngc|tap|cnc|eia|min|mpf|spf|h|i|iso|ptp|gcode|yaml|yml|json|csv|txt|md|py|log)$/i;
const VIEW_LIMIT_LINES = 5000;

export async function renderProjects(root, { api, navigate }) {
  const file = h('input', { type: 'file', class: 'input', accept: '.zip,application/zip', 'aria-label': 'Zipped project' });
  const name = h('input', { type: 'text', class: 'input', placeholder: 'id (optional)', 'aria-label': 'Project id', autocomplete: 'off' });
  const message = h('div');
  const upload = button('Upload', async () => {
    clear(message);
    if (!file.files || !file.files[0]) { message.append(errorBox('Choose a .zip with project.yaml at its top.')); return; }
    upload.disabled = true;
    try {
      const doc = await api.uploadProject(file.files[0], name.value.trim() || null);
      toast(`Project ${doc.id} uploaded`, 'ok');
      if ((doc.problems || []).length) toast(`${doc.problems.length} problem(s): ${doc.problems[0]}`, 'warning', 8000);
      navigate(`#/projects/${encodeURIComponent(doc.id)}`);
    } catch (error) {
      message.append(errorBox(error, 'Upload: '));
    } finally { upload.disabled = false; }
  }, { kind: 'primary', iconName: 'upload' });
  root.append(h('div', { class: 'page-head' }, h('h1', null, 'Projects')));
  root.append(section('Upload a project', h('p', { class: 'muted small' }, 'A zip of a project folder: project.yaml at the top (or in its one folder), with the programs, CAD and meshes it names.'),
    h('form', { class: 'form-row', on: { submit: (e) => { e.preventDefault(); upload.click(); } } }, file, name, upload), message));
  const listBox = h('div');
  root.append(listBox);
  listBox.append(loading());
  let doc;
  try { doc = await api.projects(); } catch (error) { clear(listBox); listBox.append(errorBox(error)); return; }
  clear(listBox);
  const projects = doc.projects || [];
  for (const [title, list] of [['Examples', projects.filter((p) => p.source === 'example')], ['Uploads', projects.filter((p) => p.source !== 'example')]]) {
    const box = section(`${title} (${list.length})`);
    if (!list.length) box.append(notice(title === 'Uploads' ? 'No uploads yet.' : 'No examples in this build.'));
    const grid = h('div', { class: 'cards' });
    for (const p of list) {
      grid.append(h('article', { class: 'project-card' },
        h('a', { href: `#/projects/${encodeURIComponent(p.id)}`, class: 'project-name' }, p.name || p.id),
        h('div', { class: 'muted small' }, p.id, p.read_only ? ' · read-only' : ''),
        p.description ? h('p', { class: 'small' }, p.description) : null,
        h('div', { class: 'small' }, [p.machine, p.material].filter(Boolean).join(' · ') || ''),
        h('div', { class: 'small muted' }, `${(p.operations || []).length} operation(s) · ${p.tools} tool(s)`),
        p.error ? errorBox(p.error) : null,
        (p.operations || []).some((o) => !o.program_found) ? notice('A program is missing from the project folder.', 'warning') : null));
    }
    box.append(grid);
    listBox.append(box);
  }
}

function checkResult(result) {
  const box = h('div', { class: 'check-result' });
  if (!result) { box.append(notice('The check returned nothing.')); return box; }
  const ok = result.ok;
  box.append(h('p', null, ok === false ? h('span', { class: 'badge tone-error' }, 'problems') : ok === true ? h('span', { class: 'badge tone-ok' }, 'ok') : h('span', { class: 'badge tone-muted' }, 'done'),
    ' ', result.summary || result.message || ''));
  for (const key of ['errors', 'warnings', 'problems']) {
    const items = result[key];
    if (Array.isArray(items) && items.length) {
      box.append(h('h4', null, `${humanize(key)} (${items.length})`), h('ul', { class: 'plain' }, items.map((item) => h('li', null, typeof item === 'string' ? item : JSON.stringify(item)))));
    }
  }
  const scalars = Object.entries(result).filter(([k, v]) => !['summary', 'message', 'errors', 'warnings', 'problems', 'ok'].includes(k) && (typeof v !== 'object' || v === null));
  if (scalars.length) box.append(kv(scalars.map(([k, v]) => [humanize(k), v === null ? null : String(v)])));
  const nested = Object.entries(result).filter(([k, v]) => !['errors', 'warnings', 'problems'].includes(k) && v && typeof v === 'object');
  for (const [k, v] of nested) box.append(lazyDetails(humanize(k), (b) => b.append(jsonView(v))));
  return box;
}

function fileViewer(api, id) {
  const head = h('div', { class: 'viewer-head muted small' });
  const pre = h('pre', { class: 'code-view' });
  const node = h('div', { class: 'viewer' }, head, pre);
  node.show = async (path) => {
    head.textContent = `Loading ${path}…`;
    clear(pre);
    try {
      const text = await api.projectFile(id, path);
      const lines = text.replace(/\r\n?/g, '\n').split('\n');
      const shown = lines.slice(0, VIEW_LIMIT_LINES);
      head.replaceChildren(h('b', null, path), ` · ${lines.length} lines${lines.length > VIEW_LIMIT_LINES ? ` (the first ${VIEW_LIMIT_LINES} shown)` : ''} · `,
        link(api.projectFileUrl(id, path), 'open', { newTab: true }));
      const width = String(shown.length).length;
      pre.textContent = shown.map((l, i) => `${String(i + 1).padStart(width, ' ')}  ${l}`).join('\n');
    } catch (error) {
      head.replaceChildren(errorBox(error, `${path}: `));
    }
  };
  return node;
}

export async function renderProject(root, { api, id, navigate }) {
  root.append(loading());
  let doc;
  try { doc = await api.project(id); } catch (error) { clear(root); root.append(errorBox(error, `Project ${id}: `), link('#/', 'All projects', { kind: 'ghost' })); return; }
  clear(root);
  root.append(h('div', { class: 'page-head' }, h('h1', null, doc.name || doc.id),
    h('span', { class: 'muted' }, doc.id), h('span', { class: `badge ${doc.read_only ? 'tone-muted' : 'tone-info'}` }, doc.source === 'example' ? 'example (read-only)' : 'upload')));
  if (doc.description) root.append(h('p', { class: 'lead' }, doc.description));
  if (doc.error) root.append(errorBox(doc.error, 'project.yaml: '));
  root.append(section('Project', kv([
    ['machine', doc.machine || '—'], ['material', doc.material || '—'], ['tools', String(doc.tools)],
    ['operations', h('span', null, (doc.operations || []).map((o) => h('span', { class: 'op-pill' }, `${o.name} · ${o.program}${o.program_found ? '' : ' (missing)'}`)))],
  ])));
  const checkBox = section('Check');
  const checkOut = h('div');
  const checkBtn = button('Check the project', async () => {
    clear(checkOut);
    checkBtn.disabled = true;
    checkOut.append(loading('Checking…'));
    try {
      let outcome = await api.checkProject(id, {}, { waitS: 20 });
      if (outcome.status === 202) {
        clear(checkOut);
        const line = h('p', { class: 'muted' }, `Check job ${outcome.job.id} running…`);
        checkOut.append(line);
        outcome = await api.waitJob(outcome.job.id, { onProgress: (j) => { line.textContent = `Check job ${j.id}: ${j.state}${j.progress && j.progress.phase ? ` · ${j.progress.phase}` : ''}`; } });
      }
      clear(checkOut);
      if (outcome.job && outcome.job.state !== 'done') checkOut.append(errorBox(outcome.job.error ? outcome.job.error.message : `the check ${outcome.job.state}`));
      else checkOut.append(checkResult(outcome.result));
    } catch (error) {
      clear(checkOut);
      checkOut.append(errorBox(error, 'Check: '));
    } finally { checkBtn.disabled = false; }
  }, { iconName: 'check' });
  checkBox.append(h('p', { class: 'muted small' }, 'twin check: timestep, memory and size estimates, program sanity, without running the simulation.'), checkBtn, checkOut);
  root.append(checkBox);
  const optionsKey = `cnctwin.options.${id}`;
  let stored = '{}';
  try { stored = localStorage.getItem(optionsKey) || '{}'; } catch { stored = '{}'; }
  const options = h('textarea', { class: 'input mono', rows: '4', spellcheck: 'false', 'aria-label': 'Run options (JSON)' }, stored);
  const runMessage = h('div');
  const runBtn = button('Start a run', async () => {
    clear(runMessage);
    let parsed;
    try { parsed = options.value.trim() ? JSON.parse(options.value) : {}; } catch (error) { runMessage.append(errorBox(`options: ${error.message}`)); return; }
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) { runMessage.append(errorBox('options must be a JSON object')); return; }
    try { localStorage.setItem(optionsKey, JSON.stringify(parsed)); } catch { /* private mode */ }
    runBtn.disabled = true;
    try {
      const record = await api.createRun(id, parsed);
      toast(`Run ${record.id} queued`, 'ok');
      navigate(`#/runs/${encodeURIComponent(record.id)}/overview`);
    } catch (error) { runMessage.append(errorBox(error, 'Run: ')); } finally { runBtn.disabled = false; }
  }, { kind: 'primary', iconName: 'play' });
  root.append(section('Run', h('label', { class: 'field' }, h('span', null, 'Options (JSON object, passed to the simulation unchanged; docs/api.md)'), options), runBtn, runMessage));
  const viewer = fileViewer(api, id);
  const files = doc.files || [];
  root.append(section(`Files (${files.length})`, table({
    columns: [{ key: 'path', label: 'file', render: (f) => (TEXT_FILES.test(f.path)
      ? h('button', { type: 'button', class: 'linkish', on: { click: () => viewer.show(f.path) } }, f.path)
      : link(api.projectFileUrl(id, f.path), f.path, { download: f.path.split('/').pop() })) },
    { key: 'bytes', label: 'size', num: true, render: (f) => bytes(f.bytes) }],
    rows: files, maxRows: 300,
  }), viewer));
  if (files.some((f) => f.path === 'project.yaml')) viewer.show('project.yaml');
  const runsBox = section('Runs of this project');
  root.append(runsBox);
  try {
    const runs = await api.runs({ projectId: id });
    runsBox.append(table({
      columns: [{ key: 'id', label: 'run', render: (r) => link(`#/runs/${encodeURIComponent(r.id)}/overview`, r.id) }, { key: 'status', label: 'status', render: (r) => statusBadge(r.status) },
        { key: 'verdict', label: 'verdict', render: (r) => verdictBadge(r.verdict) },
        { key: 'cycle', label: 'cycle time', num: true, value: (r) => r.summary && r.summary.cycle_time_s, render: (r) => (r.summary && r.summary.cycle_time_s !== undefined ? duration(r.summary.cycle_time_s) : num(r.progress && r.progress.sim_time_s)) }],
      rows: runs.runs || [], empty: 'No runs yet.',
    }));
  } catch (error) { runsBox.append(errorBox(error)); }
}
