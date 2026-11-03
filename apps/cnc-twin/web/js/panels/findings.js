/* Findings: what the run predicts, each with severity, category, time, program line, message and
 * quantities in display units; a click jumps the playback to its time and highlights its line;
 * "Causes" expands the chain of cause from trace.jsonl (breadth first, each event once). */

import { h, append, clear, badge, button, notice, humanize, lazyDetails, where } from './dom.js';
import { SEVERITY_RANK } from './rundata.js';
import { num, fixed, quantity } from '../plot/units.js';

const PAGE = 100;

function quantities(q) {
  const entries = Object.entries(q || {});
  if (!entries.length) return null;
  return h('ul', { class: 'qty' }, entries.map(([key, value]) => {
    const { label, text } = quantity(key, value);
    return h('li', null, h('span', null, label), ' ', h('b', null, text));
  }));
}

function position(f) {
  if (!Array.isArray(f.pos_W_m)) return null;
  return `tool tip in W: ${f.pos_W_m.map((v) => fixed(v * 1e3, 3)).join(', ')} mm`;
}

export async function causeChain(api, runId, f, onEvent) {
  const root = Array.isArray(f.cause_chain) && f.cause_chain.length ? f.cause_chain[0] : null;
  if (root === null || root === undefined) return notice('This finding records no chain of cause.');
  const doc = await api.trace(runId, root);
  const chain = doc.chain || [];
  const list = h('ol', { class: 'chain' });
  chain.forEach((event, i) => {
    const parents = (event.parents || []).length ? `from ${(event.parents || []).map((p) => `#${p}`).join(', ')}` : 'root cause';
    const q = Object.entries(event.quantities || {}).map(([key, value]) => { const x = quantity(key, value); return `${x.label} ${x.text}`; });
    list.append(h('li', { class: i === 0 ? 'chain-head' : null },
      h('button', { type: 'button', class: 'linkish', on: { click: () => onEvent && onEvent(event) } },
        h('b', null, humanize(event.event)), ` · ${event.domain} · ${num(event.time_s)} s${event.line ? ` · line ${event.line}` : ''}`),
      h('span', { class: 'muted' }, ` #${event.id}, ${parents}`),
      q.length ? h('div', { class: 'chain-q' }, q.join('; ')) : null));
  });
  const ids = (f.cause_chain || []).length;
  return h('div', null, list, ids > chain.length ? notice(`${ids - chain.length} further events not shown (limit of the request).`) : null);
}

export function findingCard(api, runId, f, { onJump, onEvent } = {}) {
  const card = h('article', { class: `finding sev-${f.severity}`, id: `finding-${f.id}` },
    h('div', { class: 'finding-head' },
      badge(f.severity),
      h('b', { class: 'finding-cat' }, humanize(f.category)),
      h('span', { class: 'muted' }, `${num(f.time_s)} s`),
      h('button', { type: 'button', class: 'loc', on: { click: () => onJump && onJump(f) }, title: 'Jump to this time and line' }, where(f))),
    h('p', { class: 'finding-msg' }, f.message),
    f.block ? h('code', { class: 'finding-block' }, f.block) : null,
    quantities(f.quantities),
    position(f) ? h('p', { class: 'muted small' }, position(f)) : null,
    lazyDetails('Causes', async (body) => {
      body.append(await causeChain(api, runId, f, onEvent));
    }, { className: 'causes' }));
  return card;
}

/** The findings list with filters (minimum severity, category, operation) and paging. */
export class FindingsList {
  constructor(parent, { api, data, onJump, onEvent, compact = false }) {
    this.api = api;
    this.data = data;
    this.onJump = onJump;
    this.onEvent = onEvent;
    this.limit = PAGE;
    this.el = h('div', { class: 'findings' });
    this.severity = h('select', { class: 'input', 'aria-label': 'Minimum severity', on: { change: () => this.render() } },
      ['info', 'warning', 'error', 'fatal'].map((s) => h('option', { value: s }, s === 'info' ? 'all severities' : `${s} and worse`)));
    this.category = h('select', { class: 'input', 'aria-label': 'Category', on: { change: () => this.render() } });
    this.order = h('select', { class: 'input', 'aria-label': 'Order', on: { change: () => this.render() } },
      h('option', { value: 'time' }, 'in time order'), h('option', { value: 'worst' }, 'worst first'));
    this.summary = h('div', { class: 'findings-summary' });
    this.listEl = h('div', { class: 'findings-list' });
    this.el.append(h('div', { class: 'filters' }, this.severity, this.category, this.order), this.summary, this.listEl);
    parent.append(this.el);
    this.compact = compact;
    data.addEventListener('findings', () => this.render());
    this.render();
  }

  render() {
    const all = this.data.findings;
    const categories = [...new Set(all.map((f) => f.category))].sort();
    const chosen = this.category.value;
    clear(this.category);
    this.category.append(h('option', { value: '' }, 'all categories'), categories.map((c) => h('option', { value: c }, humanize(c))));
    this.category.value = categories.includes(chosen) ? chosen : '';
    const min = SEVERITY_RANK[this.severity.value] || 0;
    let list = all.filter((f) => (SEVERITY_RANK[f.severity] || 0) >= min && (!this.category.value || f.category === this.category.value));
    if (this.order.value === 'worst') list = list.slice().sort((a, b) => (SEVERITY_RANK[b.severity] - SEVERITY_RANK[a.severity]) || (a.time_s - b.time_s));
    clear(this.summary);
    const counts = { fatal: 0, error: 0, warning: 0, info: 0 };
    for (const f of all) counts[f.severity] = (counts[f.severity] || 0) + 1;
    append(this.summary, [['fatal', 'error', 'warning', 'info'].filter((s) => counts[s]).map((s) => h('span', { class: 'count' }, badge(s), ` ${counts[s]}`)),
      all.length ? null : notice('No findings.')]);
    clear(this.listEl);
    if (all.length && !list.length) this.listEl.append(notice('No finding matches these filters.'));
    for (const f of list.slice(0, this.limit)) this.listEl.append(findingCard(this.api, this.data.id, f, { onJump: this.onJump, onEvent: this.onEvent }));
    if (list.length > this.limit) {
      this.listEl.append(button(`Show ${Math.min(PAGE, list.length - this.limit)} more of ${list.length - this.limit}`, () => { this.limit += PAGE; this.render(); }, { kind: 'ghost' }));
    }
  }

  focus(f) {
    if (this.limit < this.data.findings.length) { this.limit = this.data.findings.length; this.render(); }
    const card = this.el.querySelector(`#finding-${f.id}`);
    if (card) {
      card.classList.add('flash');
      card.scrollIntoView({ block: 'nearest' });
      setTimeout(() => card.classList.remove('flash'), 1600);
    }
  }
}
