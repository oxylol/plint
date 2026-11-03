/* CNC Twin UI shell: the header, the hash router (#/ projects, #/projects/<id>, #/runs,
 * #/runs/<id>/<tab>?t=<s>, #/compare?a=&b=) and the pages. Vanilla ES modules, no build step,
 * nothing loaded from outside this server. Behind Plint's Caddy theme injection the Plint pill is
 * placed into the header (window.Plint.mount); standalone the page looks the same without it. */

import { api } from './api.js';
import { h, clear, errorBox, toast, link } from './panels/dom.js';
import { renderProjects, renderProject } from './panels/projects.js';
import { renderRuns } from './panels/runs.js';
import { RunPage } from './panels/run.js';
import { renderCompare } from './panels/compare.js';

const main = document.getElementById('app');
const nav = document.querySelector('.app-nav');
let current = null;

export function parseHash(hash = location.hash) {
  const raw = hash.replace(/^#\/?/, '');
  const [path, search = ''] = raw.split('?');
  const parts = path.split('/').filter(Boolean).map(decodeURIComponent);
  const query = Object.fromEntries(new URLSearchParams(search));
  return { parts, query };
}

export function navigate(hash, { replace = false } = {}) {
  if (replace) {
    history.replaceState(null, '', hash);
    route();
  } else if (location.hash === hash) {
    route();
  } else {
    location.hash = hash;
  }
}

function setNav(section) {
  if (!nav) return;
  for (const a of nav.querySelectorAll('a')) {
    const on = a.dataset.section === section;
    a.classList.toggle('on', on);
    if (on) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
  }
}

function teardown() {
  if (!current) return;
  try { if (current.destroy) current.destroy(); } catch { /* already gone */ }
  current = null;
}

async function show(kind, key, build, title) {
  teardown();
  clear(main);
  const holder = h('div', { class: `page page-${kind}` });
  main.append(holder);
  document.title = title ? `${title} · CNC Twin` : 'CNC Twin';
  current = { kind, key, holder };
  try {
    const result = await build(holder);
    if (current && current.key === key && result && typeof result.destroy === 'function') current.destroy = result.destroy;
  } catch (error) {
    holder.append(errorBox(error));
  }
}

export async function route() {
  const { parts, query } = parseHash();
  window.scrollTo({ top: window.scrollY });
  if (parts[0] === 'runs' && parts[1]) {
    const id = parts[1];
    const tab = parts[2] || 'overview';
    setNav('runs');
    if (current && current.kind === 'run' && current.key === id && current.page) {
      await current.page.showTab(tab, query);
      return;
    }
    teardown();
    clear(main);
    const page = new RunPage(main, { api, runId: id, navigate });
    current = { kind: 'run', key: id, page, destroy: () => page.destroy() };
    document.title = `Run ${id} · CNC Twin`;
    await page.mount(tab, query);
    return;
  }
  if (parts[0] === 'runs') {
    setNav('runs');
    await show('runs', `runs?${new URLSearchParams(query)}`, (root) => renderRuns(root, { api, navigate, query }), 'Runs');
    return;
  }
  if (parts[0] === 'projects' && parts[1]) {
    setNav('projects');
    await show('project', parts[1], (root) => renderProject(root, { api, id: parts[1], navigate }), parts[1]);
    return;
  }
  if (parts[0] === 'compare') {
    setNav('compare');
    await show('compare', `compare?${new URLSearchParams(query)}`, (root) => renderCompare(root, { api, query, navigate }), 'Compare');
    return;
  }
  setNav('projects');
  await show('projects', 'projects', (root) => renderProjects(root, { api, navigate }), 'Projects');
}

function placePlintPill() {
  if (window.Plint && typeof window.Plint === 'object') window.Plint.mount = ['.app-head .pill-slot'];
}

async function footer() {
  const foot = document.getElementById('foot');
  if (!foot) return;
  try {
    const health = await api.health();
    foot.replaceChildren(h('span', null, `CNC Twin ${health.version || ''}`), ' · ',
      h('span', null, `${health.jobs ? health.jobs.running : 0} running, ${health.jobs ? health.jobs.queued : 0} queued`), ' · ',
      link('/api/openapi.json', 'API', { newTab: true }));
  } catch (error) {
    foot.replaceChildren(errorBox(error, 'The API does not answer: '));
  }
}

window.addEventListener('hashchange', () => route());
window.addEventListener('unhandledrejection', (event) => {
  const reason = event.reason;
  if (reason && reason.name === 'AbortError') { event.preventDefault(); return; }
  toast(`Something failed: ${reason && reason.message ? reason.message : reason}`, 'error', 7000);
});
placePlintPill();
window.addEventListener('load', placePlintPill);
footer();
route();
