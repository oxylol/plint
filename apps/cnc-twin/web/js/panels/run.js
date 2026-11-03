/* The run page: verdict banner (safe to run or not, and why), progress and time remaining while
 * it runs (server-sent events), and the tabs: overview with the findings, playback (3D view,
 * transport, DRO, program, plots, layers), the result panels, studies and files. The page owns the
 * Timebase; the URL keeps the tab and the time (#/runs/<id>/<tab>?t=<s>). */

import { h, append, clear, button, link, statusBadge, loading, errorBox, toast, humanize, icon } from './dom.js';
import { RunData } from './rundata.js';
import { Timebase } from './timebase.js';
import { Playback } from './playback.js';
import { renderOverview } from './overview.js';
import { renderInspection, renderSurface, renderTools } from './results.js';
import { renderProduction } from './production.js';
import { renderStability } from './stability.js';
import { renderCoolant, renderChips, renderThermal } from './process.js';
import { renderChecks, renderInfo } from './checks.js';
import { renderFiles } from './files.js';
import { renderUncertainty } from './uncertainty.js';
import { renderCalibration } from './calibration.js';
import { clock, duration, num, pct } from '../plot/units.js';

export const TABS = [
  ['overview', 'Overview', renderOverview],
  ['playback', 'Playback', null],
  ['inspection', 'Inspection', renderInspection],
  ['surface', 'Surface', renderSurface],
  ['tools', 'Tools', renderTools],
  ['production', 'Production', renderProduction],
  ['stability', 'Stability', renderStability],
  ['coolant', 'Coolant', renderCoolant],
  ['chips', 'Chips', renderChips],
  ['thermal', 'Thermal', renderThermal],
  ['checks', 'Checks', renderChecks],
  ['uncertainty', 'Uncertainty', renderUncertainty],
  ['calibration', 'Calibration', renderCalibration],
  ['files', 'Reports & files', renderFiles],
  ['info', 'Run info', renderInfo],
];
const RESULT_TABS = new Set(['overview', 'inspection', 'surface', 'tools', 'production', 'stability', 'coolant', 'chips', 'thermal', 'checks', 'files', 'info']);

export class RunPage {
  constructor(root, { api, runId, navigate }) {
    this.api = api;
    this.id = runId;
    this.navigate = navigate;
    this.root = root;
    this.panels = new Map();
    this.tab = null;
    this.stream = null;
    this.el = h('div', { class: 'run-page' });
    root.append(this.el);
  }

  async mount(tab = 'overview', query = {}) {
    this.el.append(loading('Loading the run…'));
    const data = new RunData(this.api, this.id);
    try {
      await data.load();
    } catch (error) {
      clear(this.el);
      this.el.append(errorBox(error, `Run ${this.id}: `), link('#/runs', 'All runs', { kind: 'ghost' }));
      return;
    }
    this.data = data;
    const t = Number(query.t);
    this.time = new Timebase({ duration: data.duration(), t: Number.isFinite(t) ? t : 0 });
    this.time.addEventListener('state', (e) => { if (!e.detail.playing) this._remember(); });
    this.time.addEventListener('time', (e) => { if (e.detail.source !== 'play') this._rememberSoon(); });
    clear(this.el);
    this.header = h('header', { class: 'run-head' });
    this.banner = h('div', { class: 'verdict' });
    this.progress = h('div', { class: 'progress-box' });
    this.tabsEl = h('nav', { class: 'tabs', 'aria-label': 'Run sections' });
    this.body = h('div', { class: 'run-body' });
    this.el.append(this.header, this.banner, this.progress, this.tabsEl, this.body);
    this.renderHeader();
    this.renderBanner();
    this.renderProgress();
    this.renderTabs();
    this._keys = (e) => this._onKey(e);
    document.addEventListener('keydown', this._keys);
    if (data.active) this.follow();
    await this.showTab(TABS.some(([name]) => name === tab) ? tab : 'overview', query);
  }

  renderHeader() {
    const data = this.data;
    const name = data.project.name || (this.data.record.job && this.data.record.job.project_id) || 'Run';
    clear(this.header);
    const projectId = data.projectId;
    this.header.append(
      h('div', { class: 'run-title' },
        h('h1', null, name),
        h('span', { class: 'muted' }, this.id), statusBadge(data.state)),
      h('div', { class: 'run-actions' },
        projectId ? link(`#/projects/${encodeURIComponent(projectId)}`, 'Project', { kind: 'ghost small' }) : null,
        data.has('report.html') ? link(this.api.fileUrl(this.id, 'report.html'), 'Report', { kind: 'ghost small', newTab: true, iconName: 'external' }) : null,
        data.has('report.pdf') ? link(this.api.fileUrl(this.id, 'report.pdf'), 'PDF', { kind: 'ghost small', newTab: true, iconName: 'download' }) : null,
        data.finished ? link(`#/compare?a=${encodeURIComponent(this.id)}`, 'Compare', { kind: 'ghost small', iconName: 'compare' }) : null,
        ['queued', 'running'].includes(data.state) ? button('Cancel', () => this.cancel(), { kind: 'small danger', iconName: 'stop' }) : null,
        ['cancelled', 'failed'].includes(data.state) ? button('Resume', () => this.resume(), { kind: 'small', iconName: 'refresh' }) : null));
  }

  _reasonFinding(reason) {
    return this.data.findings.find((f) => String(reason).startsWith(`${f.category} (${f.severity})`) && String(reason).includes(f.message))
      || this.data.findings.find((f) => String(reason).startsWith(f.category) && f.line && String(reason).includes(`line ${f.line}`));
  }

  renderBanner() {
    const data = this.data;
    clear(this.banner);
    const verdict = data.run && data.run.verdict;
    const counts = { fatal: 0, error: 0, warning: 0, info: 0 };
    for (const f of data.findings) counts[f.severity] = (counts[f.severity] || 0) + 1;
    const countLine = ['fatal', 'error', 'warning', 'info'].filter((s) => counts[s]).map((s) => `${counts[s]} ${s}`).join(' · ') || 'no findings';
    if (verdict && verdict.safe !== undefined) {
      this.banner.className = `verdict ${verdict.safe ? 'safe' : 'unsafe'}`;
      const reasons = (verdict.reasons || []).map((reason) => {
        const f = this._reasonFinding(reason);
        return h('li', null, f ? h('button', { type: 'button', class: 'linkish', on: { click: () => this.focusFinding(f) } }, reason) : reason);
      });
      append(this.banner, [
        h('div', { class: 'verdict-main' }, icon(verdict.safe ? 'check' : 'x'),
          h('div', null, h('strong', null, verdict.safe ? 'Safe to run' : 'Not safe to run'),
            h('span', null, verdict.safe ? `No finding is an error or fatal. ${countLine}.` : `${countLine}. Why:`))),
        reasons.length ? h('ul', { class: 'verdict-reasons' }, reasons) : null]);
      return;
    }
    const state = data.state;
    const worst = data.worstSeverity();
    if (state === 'failed') {
      const error = data.status.error || {};
      this.banner.className = 'verdict unsafe';
      append(this.banner, [h('div', { class: 'verdict-main' }, icon('alert'),
        h('div', null, h('strong', null, 'The run failed'), h('span', null, `${error.type ? `${error.type}: ` : ''}${error.message || 'no error recorded'}`))),
      error.traceback ? h('details', { class: 'disclosure' }, h('summary', null, 'Traceback'), h('pre', { class: 'json' }, error.traceback)) : null,
      h('p', { class: 'small' }, `Findings so far: ${countLine}.`)]);
      return;
    }
    this.banner.className = `verdict pending ${worst === 'error' || worst === 'fatal' ? 'unsafe' : ''}`;
    const so = worst === 'error' || worst === 'fatal' ? 'not safe to run (an error or fatal finding already)' : 'no error so far';
    this.banner.append(h('div', { class: 'verdict-main' }, icon('info'),
      h('div', null, h('strong', null, state === 'cancelled' ? 'Cancelled' : state === 'queued' ? 'Queued' : 'Running'),
        h('span', null, `Verdict so far: ${so}. ${countLine}.`))));
  }

  renderProgress() {
    const data = this.data;
    clear(this.progress);
    const status = data.status;
    if (!['queued', 'running', 'cancelling', 'cancelled', 'failed'].includes(data.state)) { this.progress.hidden = true; return; }
    this.progress.hidden = false;
    const p = status.progress || {};
    const sim = Number(p.sim_time_s) || 0;
    const total = Number(p.est_total_s) || null;
    const frac = total ? Math.min(1, sim / total) : null;
    let eta = Number.isFinite(Number(p.eta_s)) && p.eta_s !== null ? Number(p.eta_s) : null;
    if (eta === null && total && p.wall_s && sim > 0) eta = (total - sim) * (Number(p.wall_s) / sim);
    const bar = h('div', { class: 'meter-track big', role: 'progressbar', 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-valuenow': frac === null ? null : String(Math.round(frac * 100)) },
      h('span', { class: 'meter-fill info', style: { width: `${(frac || 0) * 100}%` } }));
    this.progress.append(
      h('div', { class: 'progress-line' },
        h('b', null, frac === null ? `${clock(sim)} simulated` : `${pct(frac, 0)} · ${clock(sim)} of ${clock(total)} simulated`),
        h('span', { class: 'muted' }, eta !== null && data.active ? ` · about ${duration(eta)} left` : ''),
        p.wall_s ? h('span', { class: 'muted' }, ` · ${duration(Number(p.wall_s))} wall time`) : null),
      bar,
      h('div', { class: 'muted small' },
        [status.op ? `operation ${status.op}` : null, status.phase ? `phase ${status.phase}` : null, status.message || null,
          status.queue_position !== undefined && status.queue_position !== null ? `${status.queue_position} job(s) ahead` : null,
          status.checkpoint ? `last checkpoint at ${num((status.checkpoint.tick || 0) * data.tickS)} s` : null].filter(Boolean).join(' · ')));
  }

  renderTabs() {
    clear(this.tabsEl);
    for (const [name, label] of TABS) {
      const count = name === 'overview' && this.data.findings.length ? ` (${this.data.findings.length})` : '';
      this.tabsEl.append(h('a', {
        href: `#/runs/${encodeURIComponent(this.id)}/${name}`, class: ['tab', this.tab === name ? 'on' : ''],
        'aria-current': this.tab === name ? 'page' : null, dataset: { tab: name },
      }, `${label}${count}`));
    }
  }

  async showTab(name, query = {}) {
    if (!this.data) return;
    const t = Number(query.t);
    if (Number.isFinite(t)) this.time.seek(t, 'url');
    if (this.tab === name && this.panels.has(name)) return;
    this.tab = name;
    this.renderTabs();
    for (const [key, panel] of this.panels) panel.el.hidden = key !== name;
    if (this.panels.has(name)) return;
    const el = h('section', { class: `panel panel-${name}`, 'aria-label': name });
    this.body.append(el);
    const panel = { el, built: null };
    this.panels.set(name, panel);
    await this._build(name, panel);
  }

  async _build(name, panel) {
    clear(panel.el);
    const ctx = { api: this.api, data: this.data, time: this.time, page: this };
    try {
      if (name === 'playback') {
        panel.el.append(loading('Loading the playback…'));
        const playback = new Playback(panel.el, ctx);
        panel.built = playback;
        this.playback = playback;
        await playback.init();
        panel.el.querySelector(':scope > .loading')?.remove();
      } else {
        const entry = TABS.find(([key]) => key === name);
        panel.el.append(loading());
        const holder = h('div');
        await entry[2](holder, ctx);
        clear(panel.el);
        panel.el.append(holder);
        if (holder.findingsList) panel.el.findingsList = holder.findingsList;
      }
    } catch (error) {
      clear(panel.el);
      panel.el.append(errorBox(error, `${humanize(name)}: `));
    }
  }

  _remember() {
    if (!this.tab) return;
    const url = `#/runs/${encodeURIComponent(this.id)}/${this.tab}?t=${Number(this.time.t.toFixed(6))}`;
    if (location.hash !== url) history.replaceState(null, '', url);
  }

  _rememberSoon() {
    clearTimeout(this._rememberTimer);
    this._rememberTimer = setTimeout(() => this._remember(), 400);
  }

  /** Jump the playback to a finding: its time, its program line highlighted. */
  async focusFinding(f, { tab = 'playback' } = {}) {
    if (tab === 'overview' && this.tab !== 'overview') {
      await this.showTab('overview');
      const panel = this.panels.get('overview');
      if (panel && panel.el.findingsList) panel.el.findingsList.focus(f);
      return;
    }
    this.time.pause();
    this.time.seek(f.time_s, 'finding');
    await this.showTab('playback');
    history.replaceState(null, '', `#/runs/${encodeURIComponent(this.id)}/playback?t=${f.time_s}`);
    if (this.playback && f.line) await this.playback.showLine(f.program, f.line);
  }

  /** Jump to a time and (optionally) a program line, from any panel. */
  async seekTo(t, { program = null, line = null } = {}) {
    this.time.pause();
    this.time.seek(t, 'link');
    await this.showTab('playback');
    if (this.playback && line) await this.playback.showLine(program, line);
  }

  follow() {
    if (this.stream) return;
    let refreshTimer = null;
    const refresh = () => {
      clearTimeout(refreshTimer);
      refreshTimer = setTimeout(async () => {
        try {
          await this.data.loadFiles();
          await Promise.all([this.data.loadTimeline(), this.data.loadKeyframes(),
            this.data.channels.length ? null : this.data.loadChannels(), this.data.scene ? null : this.data.loadScene()]);
          this.data.dispatchEvent(new Event('refresh'));
        } catch { /* next event */ }
        this.time.setDuration(this.data.duration());
      }, 1500);
    };
    this.stream = this.api.runEvents(this.id, {
      progress: (status) => {
        if (status && status.status) this.data.record.status = status;
        this.time.setDuration(this.data.duration());
        this.renderProgress();
        this.renderBanner();
        refresh();
      },
      finding: (f) => { this.data.addFinding(f); this.renderBanner(); this.renderTabs(); },
      keyframe: () => refresh(),
      done: async (done) => {
        this.stream = null;
        try { await this.data.refresh(); } catch (error) { toast(`Could not reload the run: ${error.message}`, 'error'); }
        this.time.setDuration(this.data.duration());
        this.renderHeader();
        this.renderBanner();
        this.renderProgress();
        this.renderTabs();
        for (const [name, panel] of this.panels) if (RESULT_TABS.has(name)) await this._build(name, panel);
        toast(`Run ${this.id}: ${done && done.status ? done.status : 'finished'}`, done && done.status === 'done' ? 'ok' : 'warning');
      },
      error: (_event, readyState) => {
        // EventSource reconnects by itself (Last-Event-ID); a stream the server refused is replaced by polling
        if (readyState === 2) { this.stream = null; this._poll(); }
      },
    });
  }

  async _poll() {
    if (this.stream || this._polling) return;
    this._polling = true;
    try {
      while (!this.stream && this.el.isConnected) {
        await new Promise((r) => setTimeout(r, 2000));
        try {
          this.data.record = await this.api.run(this.id);
          await this.data.loadFindings();
        } catch { continue; }
        this.time.setDuration(this.data.duration());
        this.renderProgress();
        this.renderBanner();
        if (!this.data.active) {
          await this.data.refresh();
          this.renderHeader(); this.renderBanner(); this.renderProgress(); this.renderTabs();
          for (const [name, panel] of this.panels) if (RESULT_TABS.has(name)) await this._build(name, panel);
          break;
        }
      }
    } finally { this._polling = false; }
  }

  async cancel() {
    try { await this.api.cancelRun(this.id); toast('Cancel requested', 'info'); this.data.record = await this.api.run(this.id); this.renderHeader(); this.renderProgress(); this.follow(); } catch (error) { toast(error.message, 'error'); }
  }

  async resume() {
    try {
      await this.api.resumeRun(this.id);
      toast('Resuming from the last checkpoint', 'info');
      this.data.record = await this.api.run(this.id);
      this.renderHeader(); this.renderBanner(); this.renderProgress();
      this.follow();
    } catch (error) { toast(error.message, 'error'); }
  }

  _onKey(e) {
    if (this.tab !== 'playback' || e.defaultPrevented) return;
    const target = e.target;
    if (target && (target.closest('input, select, textarea, button, a, summary, [contenteditable]') || target.tagName === 'CANVAS')) return;
    if (e.key === ' ') { this.time.toggle(); e.preventDefault(); }
    else if (e.key === ',') { this.time.pause(); this.time.seek(this.data.stepBlock(this.time.t, -1, this.playback ? this.playback.path : 0), 'key'); }
    else if (e.key === '.') { this.time.pause(); this.time.seek(this.data.stepBlock(this.time.t, 1, this.playback ? this.playback.path : 0), 'key'); }
  }

  destroy() {
    if (this.stream) this.stream.close();
    document.removeEventListener('keydown', this._keys);
    if (this.time) this.time.destroy();
    for (const panel of this.panels.values()) if (panel.built && panel.built.destroy) panel.built.destroy();
    this.el.remove();
  }
}

