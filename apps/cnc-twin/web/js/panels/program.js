/* Program viewer: the NC program as it was run, with line numbers, the time each line took (all
 * its executions, from timeline.json), the findings on each line, the active line following
 * playback, and a selected line (from a finding or a click). Rows are virtualised, so programs of
 * tens of thousands of lines scroll smoothly; long lines scroll sideways inside the viewer. */

import { h, clear, badge, button, notice, errorBox, humanize } from './dom.js';
import { lineKey, SEVERITY_RANK } from './rundata.js';
import { duration, num } from '../plot/units.js';

const ROW_H = 24;
const OVERSCAN = 12;

function lineTime(seconds) {
  if (!(seconds > 0)) return '';
  if (seconds < 1e-3) return `${num(seconds * 1e6, 3)} µs`;
  if (seconds < 1) return `${num(seconds * 1e3, 3)} ms`;
  return duration(seconds);
}

export class ProgramViewer {
  constructor(parent, { data, onSeek, onFinding }) {
    this.data = data;
    this.onSeek = onSeek;
    this.onFinding = onFinding;
    this.lines = [];
    this.program = null;
    this.op = null;
    this.active = null;
    this.selected = null;
    this.follow = true;
    this._userScroll = 0;
    this.el = h('div', { class: 'program' });
    this.opSelect = h('select', { class: 'input', 'aria-label': 'Operation', on: { change: () => this.show({ op: this.opSelect.value }) } });
    this.followBox = h('input', { type: 'checkbox', checked: true, on: { change: () => { this.follow = this.followBox.checked; if (this.follow) this.scrollTo(this.active, true); } } });
    this.gotoInput = h('input', { class: 'input', type: 'number', min: '1', inputmode: 'numeric', placeholder: 'line', 'aria-label': 'Go to line',
      on: { keydown: (e) => { if (e.key === 'Enter') this.gotoLine(Number(this.gotoInput.value)); } } });
    const head = h('div', { class: 'program-head' },
      h('label', { class: 'field inline' }, h('span', null, 'Operation'), this.opSelect),
      h('label', { class: 'check' }, this.followBox, h('span', null, 'Follow')),
      h('span', { class: 'program-goto' }, this.gotoInput, button('Go', () => this.gotoLine(Number(this.gotoInput.value)), { kind: 'small' })));
    this.title = h('div', { class: 'program-title muted' });
    this.viewport = h('div', { class: 'program-viewport', tabindex: '0', role: 'list', 'aria-label': 'Program lines' });
    this.inner = h('div', { class: 'program-inner' });
    this.viewport.append(this.inner);
    this.detail = h('div', { class: 'program-detail' });
    this.el.append(head, this.title, this.viewport, this.detail);
    parent.append(this.el);
    this.viewport.addEventListener('scroll', () => {
      if (this._autoScroll) this._autoScroll = false; else this._userScroll = performance.now();
      this._schedule();
    });
    this.viewport.addEventListener('keydown', (e) => {
      if (!this.selected) return;
      if (e.key === 'ArrowDown') { this.selectLine(this.selected + 1, { seek: false }); e.preventDefault(); }
      if (e.key === 'ArrowUp') { this.selectLine(Math.max(1, this.selected - 1), { seek: false }); e.preventDefault(); }
      if (e.key === 'Enter') { this.selectLine(this.selected, { seek: true }); e.preventDefault(); }
    });
    this.data.addEventListener('findings', () => this.render(true));
    this.data.addEventListener('timeline', () => this.render(true));
    const ops = this.data.operations();
    for (const o of ops) this.opSelect.append(h('option', { value: o.name }, `${o.name}${o.program ? ` · ${o.program}` : ''}`));
    if (!ops.length) this.opSelect.append(h('option', { value: '' }, 'no operations recorded'));
  }

  async show({ op = null, program = null } = {}) {
    if (program && program === this.program && this.lines.length) return;
    if (!op && !program) {
      const ops = this.data.operations();
      if (!ops.length) { clear(this.inner); this.title.replaceChildren(notice('This run records no operations yet.')); return; }
      op = ops[0].name;
    }
    this.title.textContent = 'Loading the program…';
    try {
      const doc = await this.data.programText({ op, program });
      this.program = doc.program;
      this.op = doc.op || op;
      if (this.op) this.opSelect.value = this.op;
      this.lines = String(doc.text).replace(/\r\n?/g, '\n').split('\n');
      if (this.lines.length && this.lines[this.lines.length - 1] === '') this.lines.pop();
      const longest = this.lines.reduce((m, l) => Math.max(m, l.length), 0);
      this.inner.style.height = `${this.lines.length * ROW_H}px`;
      this.inner.style.width = `calc(${Math.min(longest, 400)}ch + 190px)`;
      this.title.textContent = `${this.program} · ${this.lines.length} lines${doc.source === 'project' ? ' (from the project folder)' : ''}`;
      this.render(true);
      if (this.active && this.active.program === this.program) this.scrollTo(this.active.line, true);
    } catch (error) {
      this.lines = [];
      clear(this.inner);
      this.title.replaceChildren(errorBox(error, 'Program: '));
    }
  }

  /** The line being executed now (from playback). */
  setActive(program, line) {
    const next = program ? { program, line } : null;
    if (next && this.active && next.program === this.active.program && next.line === this.active.line) return;
    this.active = next;
    if (next && next.program !== this.program && this.lines.length) {
      const op = (this.data.operations().find((o) => o.program === next.program) || {}).name;
      this.show({ op, program: next.program });
      return;
    }
    this.render(false);
    if (this.follow && next && performance.now() - this._userScroll > 1500) this.scrollTo(next.line, false);
  }

  /** Highlight a line (from a finding) and show its details; seek to its first execution if asked. */
  async selectLine(line, { program = null, seek = true } = {}) {
    if (program && program !== this.program) {
      const op = (this.data.operations().find((o) => o.program === program) || {}).name;
      await this.show({ op, program });
    }
    this.selected = line;
    this.render(false);
    this.scrollTo(line, true);
    this.renderDetail();
    if (seek) {
      const stat = this.data.lineStat(this.program, line);
      if (stat && stat.starts.length && this.onSeek) {
        const now = Math.round((this._now || 0) / this.data.tickS);
        const next = stat.starts.find((s) => s > now) ?? stat.starts[0];
        this.onSeek(next * this.data.tickS);
      }
    }
  }

  setTime(t) { this._now = t; }

  gotoLine(line) {
    if (!Number.isFinite(line) || line < 1 || !this.lines.length) return;
    this.selectLine(Math.min(this.lines.length, Math.round(line)), { seek: false });
  }

  scrollTo(line, center) {
    if (!line || !this.lines.length) return;
    const top = (line - 1) * ROW_H;
    const view = this.viewport.clientHeight || 300;
    const current = this.viewport.scrollTop;
    if (!center && top >= current + ROW_H && top <= current + view - 2 * ROW_H) return;
    const next = Math.max(0, Math.min(top - view / 2 + ROW_H / 2, this.viewport.scrollHeight - view));
    if (Math.abs(next - current) < 1) return;
    this._autoScroll = true;
    this.viewport.scrollTop = next;
    this._schedule();
  }

  _schedule() {
    if (this._frame) return;
    this._frame = requestAnimationFrame(() => { this._frame = null; this.render(false); });
  }

  render(full) {
    if (!this.lines.length) return;
    const view = this.viewport.clientHeight || 360;
    const first = Math.max(0, Math.floor(this.viewport.scrollTop / ROW_H) - OVERSCAN);
    const last = Math.min(this.lines.length, Math.ceil((this.viewport.scrollTop + view) / ROW_H) + OVERSCAN);
    if (!full && this._window && this._window[0] === first && this._window[1] === last && this._window[2] === this.active?.line && this._window[3] === this.selected) return;
    this._window = [first, last, this.active?.line, this.selected];
    const rows = [];
    for (let i = first; i < last; i++) rows.push(this._row(i + 1));
    this.inner.replaceChildren(...rows);
  }

  _row(line) {
    const text = this.lines[line - 1];
    const stat = this.data.lineStat(this.program, line);
    const found = this.data.findingsByLine.get(lineKey(this.program, line)) || [];
    let worst = null;
    for (const f of found) if (worst === null || SEVERITY_RANK[f.severity] > SEVERITY_RANK[worst]) worst = f.severity;
    const isActive = this.active && this.active.program === this.program && this.active.line === line;
    const row = h('div', {
      class: ['program-row', isActive ? 'active' : '', this.selected === line ? 'selected' : '', worst ? `has-${worst}` : ''],
      style: { top: `${(line - 1) * ROW_H}px` }, role: 'listitem',
    },
    h('span', { class: 'ln' }, String(line)),
    h('span', { class: 'lt', title: stat ? `${stat.executions} execution(s), ${num(stat.ticks * this.data.tickS)} s in total` : 'not executed' },
      stat ? `${lineTime(stat.ticks * this.data.tickS)}${stat.executions > 1 ? ` ×${stat.executions}` : ''}` : ''),
    h('span', { class: 'lf' }, worst ? h('i', { class: `dot sev-${worst}`, title: `${found.length} finding(s), worst ${worst}` }) : null,
      found.length > 1 ? h('b', null, String(found.length)) : null),
    h('code', { class: 'lc' }, text));
    row.addEventListener('click', () => this.selectLine(line, { seek: true }));
    return row;
  }

  renderDetail() {
    clear(this.detail);
    if (!this.selected) return;
    const stat = this.data.lineStat(this.program, this.selected);
    const found = this.data.findingsByLine.get(lineKey(this.program, this.selected)) || [];
    this.detail.append(h('div', { class: 'program-detail-head' },
      h('b', null, `Line ${this.selected}`), ' ',
      h('code', null, this.lines[this.selected - 1] || ''), ' ',
      h('span', { class: 'muted' }, stat ? `executed ${stat.executions}×, ${lineTime(stat.ticks * this.data.tickS)} in total` : 'not executed in this run')));
    for (const f of found) {
      this.detail.append(h('button', {
        type: 'button', class: 'finding-mini',
        on: { click: () => { if (this.onFinding) this.onFinding(f); } },
      }, badge(f.severity), h('span', null, `${humanize(f.category)} at ${num(f.time_s)} s: ${f.message}`)));
    }
  }
}
