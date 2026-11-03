/* DOM helpers of the panels: an element builder, tables (sortable, scroll sideways inside their
 * own box on phones), figures with a table view, badges with icon + text, tiles, notices, toasts.
 * Text from the server always goes in as text nodes (never as HTML). */

import { num, DASH } from '../plot/units.js';

const SVG_NS = 'http://www.w3.org/2000/svg';

export function h(tag, attrs = null, ...children) {
  const node = document.createElement(tag);
  if (attrs) applyAttrs(node, attrs);
  append(node, children);
  return node;
}

function applyAttrs(node, attrs) {
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === 'class') node.className = Array.isArray(value) ? value.filter(Boolean).join(' ') : value;
    else if (key === 'style' && typeof value === 'object') Object.assign(node.style, value);
    else if (key === 'on') for (const [event, fn] of Object.entries(value)) node.addEventListener(event, fn);
    else if (key === 'dataset') Object.assign(node.dataset, value);
    else if (['value', 'checked', 'disabled', 'hidden', 'selected', 'open', 'multiple', 'indeterminate'].includes(key)) node[key] = value;
    else if (key === 'text') node.textContent = value;
    else node.setAttribute(key, value === true ? '' : String(value));
  }
}

export function append(node, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function clear(node) {
  while (node.firstChild) node.firstChild.remove();
  return node;
}

// ------------------------------------------------------------------ icons
const PATHS = {
  play: 'M7 5v14l12-7z',
  pause: 'M7 5h3.5v14H7zM13.5 5H17v14h-3.5z',
  prev: 'M6 5h2v14H6zM19 5v14L9 12z',
  next: 'M16 5h2v14h-2zM5 5l10 7-10 7z',
  check: 'M5 12.5l4.5 4.5L19 7.5',
  x: 'M6 6l12 12M18 6L6 18',
  alert: 'M12 3l9.5 17h-19zM12 10v4.5M12 17.6v.4',
  info: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM12 11v6M12 7.6v.4',
  fatal: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM8.5 8.5l7 7M15.5 8.5l-7 7',
  chevron: 'M9 6l6 6-6 6',
  download: 'M12 4v11M7 10.5l5 5 5-5M5 20h14',
  external: 'M14 4h6v6M20 4l-9 9M19 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h5',
  upload: 'M12 20V9M7 13.5l5-5 5 5M5 4h14',
  refresh: 'M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7',
  stop: 'M7 7h10v10H7z',
  ruler: 'M3 17L17 3l4 4L7 21zM7 13l2 2M10 10l2 2M13 7l2 2',
  target: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8zM12 1v4M12 19v4M1 12h4M19 12h4',
  home: 'M4 20V9.5L12 4l8 5.5V20h-5v-6H9v6z',
  compare: 'M8 4v16M16 4v16M4 8h8M12 16h8',
  layers: 'M12 3l9 5-9 5-9-5zM3 13l9 5 9-5',
  camera: 'M4 7h3l2-2.5h6L17 7h3v12H4zM12 9.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7z',
  search: 'M10.5 4a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13zM15.5 15.5L21 21',
  play_circle: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM10 8.5v7l5.5-3.5z',
};

export function icon(name, label = null) {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('class', `ico ico-${name}`);
  if (label) { svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', label); } else { svg.setAttribute('aria-hidden', 'true'); }
  const path = document.createElementNS(SVG_NS, 'path');
  path.setAttribute('d', PATHS[name] || PATHS.info);
  if (['play', 'pause', 'prev', 'next', 'stop'].includes(name)) path.setAttribute('class', 'fill');
  svg.append(path);
  return svg;
}

// ------------------------------------------------------------------ small pieces
export function button(label, onClick, { kind = '', iconName = null, title = null, disabled = false, type = 'button', ariaLabel = null } = {}) {
  return h('button', {
    type, class: ['btn', kind], title, disabled, 'aria-label': ariaLabel,
    on: onClick ? { click: onClick } : null,
  }, iconName ? icon(iconName) : null, label ? h('span', null, label) : null);
}

export function link(href, label, { newTab = false, download = null, kind = '', iconName = null } = {}) {
  return h('a', {
    href, class: kind ? ['btn', kind] : null, target: newTab ? '_blank' : null, rel: newTab ? 'noopener' : null,
    download: download === null ? null : download,
  }, iconName ? icon(iconName) : null, label);
}

const SEVERITY_ICON = { info: 'info', warning: 'alert', error: 'x', fatal: 'fatal' };

export function badge(severity) {
  const s = String(severity || 'info');
  return h('span', { class: `badge sev-${s}` }, icon(SEVERITY_ICON[s] || 'info'), s);
}

const STATUS_TONE = { done: 'ok', running: 'info', queued: 'info', cancelling: 'warning', cancelled: 'warning', failed: 'error' };
export function statusBadge(status) {
  const s = String(status || 'unknown');
  return h('span', { class: `badge tone-${STATUS_TONE[s] || 'info'}` }, s);
}

export function verdictBadge(verdict) {
  if (!verdict || verdict.safe === undefined || verdict.safe === null) return h('span', { class: 'badge tone-muted' }, 'no verdict');
  return verdict.safe
    ? h('span', { class: 'badge tone-ok' }, icon('check'), 'safe to run')
    : h('span', { class: 'badge tone-error' }, icon('x'), 'not safe');
}

export function humanize(text) {
  return String(text || '').replace(/_/g, ' ');
}

export function notice(text, tone = 'muted') {
  return h('p', { class: ['notice', tone !== 'muted' ? `notice-${tone}` : ''] }, text);
}

export function loading(text = 'Loading…') {
  return h('div', { class: 'loading', role: 'status' }, h('span', { class: 'spinner', 'aria-hidden': 'true' }), text);
}

export function errorBox(error, prefix = '') {
  const message = error && error.detail ? error.detail : (error && error.message) || String(error);
  const status = error && error.status ? ` (HTTP ${error.status})` : '';
  return h('div', { class: 'errorbox', role: 'alert' }, icon('alert'), h('span', null, `${prefix}${message}${status}`));
}

export function kv(pairs) {
  const list = h('dl', { class: 'kv' });
  for (const [key, value] of pairs) {
    if (value === undefined) continue;
    list.append(h('dt', null, key), h('dd', null, value === null || value === '' ? DASH : value));
  }
  return list;
}

/** Headline tiles: [{label, value, sub, tone}]. */
export function tiles(items) {
  return h('div', { class: 'tiles' }, items.map((t) => h('div', { class: ['tile', t.tone ? `tone-${t.tone}` : ''] },
    h('span', { class: 'tile-label' }, t.label),
    h('span', { class: 'tile-value' }, t.value === null || t.value === undefined ? DASH : t.value),
    t.sub ? h('span', { class: 'tile-sub' }, t.sub) : null)));
}

export function section(title, ...children) {
  return h('section', { class: 'card' }, title ? h('h2', { class: 'card-title' }, title) : null, ...children);
}

/** A <details> whose body is built on first open. */
export function lazyDetails(summary, build, { open = false, className = '' } = {}) {
  const body = h('div', { class: 'details-body' });
  const node = h('details', { class: ['disclosure', className] }, h('summary', null, summary), body);
  let built = false;
  const ensure = () => {
    if (built || !node.open) return;
    built = true;
    Promise.resolve(build(body)).catch((error) => body.append(errorBox(error)));
  };
  node.addEventListener('toggle', ensure);
  if (open) { node.open = true; ensure(); }
  return node;
}

/** A sortable table in a sideways-scrolling box. columns: [{key, label, num?, render?(row), value?(row), title?}]. */
export function table({ columns, rows, empty = 'Nothing to show.', caption = null, onRow = null, rowClass = null, maxRows = null, className = '' }) {
  const wrap = h('div', { class: ['table-wrap', className] });
  if (!rows || !rows.length) { wrap.append(notice(empty)); return wrap; }
  const tableEl = h('table', { class: 'data' });
  if (caption) tableEl.append(h('caption', null, caption));
  const head = h('tr');
  let sortKey = null;
  let sortDir = 1;
  const body = h('tbody');
  const fill = () => {
    clear(body);
    let list = rows.slice();
    if (sortKey !== null) {
      const col = columns[sortKey];
      const value = col.value || ((r) => r[col.key]);
      list.sort((a, b) => {
        const va = value(a); const vb = value(b);
        if (va === vb) return 0;
        if (va === null || va === undefined) return 1;
        if (vb === null || vb === undefined) return -1;
        return (va < vb ? -1 : 1) * sortDir;
      });
    }
    const shown = maxRows ? list.slice(0, maxRows) : list;
    for (const row of shown) {
      const tr = h('tr', { class: rowClass ? rowClass(row) : null });
      if (onRow) { tr.classList.add('clickable'); tr.tabIndex = 0; tr.addEventListener('click', () => onRow(row)); tr.addEventListener('keydown', (e) => { if (e.key === 'Enter') onRow(row); }); }
      for (const col of columns) {
        const content = col.render ? col.render(row) : row[col.key];
        tr.append(h('td', { class: col.num ? 'num' : null }, content === null || content === undefined || content === '' ? DASH : content));
      }
      body.append(tr);
    }
    if (maxRows && list.length > maxRows) {
      body.append(h('tr', null, h('td', { colspan: String(columns.length), class: 'muted' },
        `${list.length - maxRows} more rows not shown`, ' ',
        button('Show all', () => { maxRows = null; fill(); }, { kind: 'small' }))));
    }
  };
  columns.forEach((col, index) => {
    const th = h('th', { class: col.num ? 'num' : null, scope: 'col', title: col.title || null });
    if (col.sortable === false) th.append(col.label);
    else {
      th.append(h('button', {
        type: 'button', class: 'sort',
        on: { click: () => { sortDir = sortKey === index ? -sortDir : 1; sortKey = index; fill(); head.querySelectorAll('th').forEach((x, i) => x.setAttribute('aria-sort', i === index ? (sortDir > 0 ? 'ascending' : 'descending') : 'none')); } },
      }, col.label));
    }
    head.append(th);
  });
  tableEl.append(h('thead', null, head), body);
  fill();
  wrap.append(tableEl);
  return wrap;
}

/** A chart with a title and a Table toggle: chart(container) draws, tableView() builds the numbers. */
export function figure({ title, subtitle = null, chart, tableView = null, note = null, className = '' }) {
  const body = h('div', { class: 'figure-body' });
  const tableBox = h('div', { class: 'figure-table', hidden: true });
  let built = false;
  const toggle = tableView ? button('Table', () => {
    tableBox.hidden = !tableBox.hidden;
    toggle.setAttribute('aria-pressed', String(!tableBox.hidden));
    if (!built && !tableBox.hidden) { built = true; tableBox.append(tableView()); }
  }, { kind: 'small ghost', title: 'Show the numbers as a table' }) : null;
  if (toggle) toggle.setAttribute('aria-pressed', 'false');
  const node = h('figure', { class: ['figure', className] },
    h('figcaption', { class: 'figure-head' }, h('span', { class: 'figure-title' }, title), subtitle ? h('span', { class: 'figure-sub' }, subtitle) : null, toggle),
    body, tableBox, note ? h('p', { class: 'figure-note' }, note) : null);
  chart(body);
  return node;
}

export function jsonView(value) {
  return h('pre', { class: 'json' }, JSON.stringify(value, null, 1));
}

// ------------------------------------------------------------------ toasts
let toastRoot = null;
export function toast(message, tone = 'info', ms = 4500) {
  if (!toastRoot) {
    toastRoot = h('div', { class: 'toasts', role: 'status', 'aria-live': 'polite' });
    document.body.append(toastRoot);
  }
  const item = h('div', { class: `toast tone-${tone}` }, message);
  toastRoot.append(item);
  setTimeout(() => item.remove(), ms);
}

export function formatCount(n, word) {
  return `${num(n)} ${word}${n === 1 ? '' : 's'}`;
}

/** Map a finding's location to text: "OP10 · op10.nc:17". */
export function where(f) {
  const parts = [];
  if (f.op) parts.push(f.op);
  if (f.program || f.line) parts.push(`${f.program || ''}${f.line ? `:${f.line}` : ''}`);
  return parts.join(' · ') || DASH;
}
