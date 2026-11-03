/* Bar charts in HTML (crisp text, accessible, fluid on phones): a list of horizontal bars, one
 * stacked bar with its legend, and a meter against a limit. Bars are at most 22 px thick with a
 * rounded data end and a square base; stacked segments are separated by a 2 px gap; values stay
 * in ink colours beside the marks. Each element has a title (hover) and the same values in text. */

import { num } from './units.js';

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

/** rows: [{label, value, text?, color?, title?, onClick?}]; value in any unit (text shows it). */
export function barList(rows, { signed = false, max = null, emptyText = 'nothing to show', label = 'bar chart' } = {}) {
  const root = el('div', 'bars');
  root.setAttribute('role', 'list');
  root.setAttribute('aria-label', label);
  if (!rows.length) { root.append(el('p', 'muted', emptyText)); return root; }
  const finite = rows.map((r) => r.value).filter(Number.isFinite);
  const top = max !== null ? max : Math.max(1e-300, ...finite.map(Math.abs));
  for (const row of rows) {
    const item = el(row.onClick ? 'button' : 'div', 'bar-row');
    item.setAttribute('role', 'listitem');
    if (row.onClick) { item.type = 'button'; item.addEventListener('click', row.onClick); }
    if (row.title) item.title = row.title;
    const name = el('span', 'bar-label', row.label);
    const track = el('span', `bar-track${signed ? ' signed' : ''}`);
    const fill = el('span', 'bar-fill');
    const value = Number.isFinite(row.value) ? row.value : 0;
    const share = Math.min(1, Math.abs(value) / top);
    if (signed) {
      fill.style.width = `${share * 50}%`;
      if (value < 0) { fill.style.left = 'auto'; fill.style.right = '50%'; fill.classList.add('neg'); } else { fill.style.left = '50%'; }
    } else {
      fill.style.width = `${share * 100}%`;
    }
    if (row.color) fill.style.background = row.color;
    track.append(fill);
    const text = el('span', 'bar-value', row.text !== undefined ? row.text : num(row.value));
    item.append(name, track, text);
    root.append(item);
  }
  return root;
}

/** One stacked bar: segments [{label, value, text, color}] (value >= 0), then a legend with values. */
export function stackedBar(segments, { label = 'share', total = null } = {}) {
  const root = el('div', 'stack');
  const sum = total !== null ? total : segments.reduce((a, s) => a + (Number.isFinite(s.value) ? Math.max(0, s.value) : 0), 0);
  const barEl = el('div', 'stack-bar');
  barEl.setAttribute('role', 'img');
  barEl.setAttribute('aria-label', `${label}: ${segments.map((s) => `${s.label} ${s.text || num(s.value)}`).join(', ')}`);
  for (const s of segments) {
    if (!(s.value > 0) || !(sum > 0)) continue;
    const part = el('span', 'stack-seg');
    part.style.flexGrow = String(s.value / sum);
    part.style.background = s.color;
    part.title = `${s.label}: ${s.text || num(s.value)} (${num(100 * s.value / sum, 3)} %)`;
    barEl.append(part);
  }
  const legend = el('ul', 'stack-legend');
  for (const s of segments) {
    const item = el('li');
    const key = el('i');
    key.style.background = s.color;
    item.append(key, el('span', '', s.label), el('b', '', s.text || num(s.value)),
      el('span', 'muted', sum > 0 && Number.isFinite(s.value) ? `${num(100 * s.value / sum, 3)} %` : ''));
    legend.append(item);
  }
  root.append(barEl, legend);
  return root;
}

/** A meter: value against a limit (fraction filled), text beside it; tone ok | warn | bad. */
export function meter(fraction, { text = '', tone = 'ok', label = '' } = {}) {
  const root = el('div', 'meter');
  const track = el('span', 'meter-track');
  track.setAttribute('role', 'meter');
  track.setAttribute('aria-valuemin', '0');
  track.setAttribute('aria-valuemax', '100');
  const share = Number.isFinite(fraction) ? Math.max(0, Math.min(1, fraction)) : 0;
  track.setAttribute('aria-valuenow', String(Math.round(share * 100)));
  if (label) track.setAttribute('aria-label', label);
  const fill = el('span', `meter-fill ${tone}`);
  fill.style.width = `${share * 100}%`;
  track.append(fill);
  root.append(track, el('span', 'meter-text', text));
  return root;
}
