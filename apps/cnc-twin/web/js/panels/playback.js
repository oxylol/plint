/* The playback workbench of a run: the 3D view (web/js/view3d), the transport, the DRO, the layer
 * controls, the program viewer following the active line, and the stacked plots, all driven by the
 * run page's Timebase. Keys (outside text fields): space plays or pauses, "," and "." step a block. */

import { h, notice, errorBox } from './dom.js';
import { mountView3D } from './viewhost.js';
import { Transport } from './transport.js';
import { Dro, droColumns } from './dro.js';
import { LayerControls } from './layers.js';
import { ProgramViewer } from './program.js';
import { PlotsPanel } from './plots.js';
import { FrameStore, ChannelSampler } from './rundata.js';

export class Playback {
  constructor(parent, { api, data, time, page }) {
    this.api = api;
    this.data = data;
    this.time = time;
    this.page = page;
    this.path = data.pathList[0] || 0;
    this.view = null;
    this.el = h('div', { class: 'playback' });
    parent.append(this.el);
    this.viewEl = h('div', { class: 'pb-view', 'aria-label': '3D view of the machine and the part' });
    this.layersEl = h('div', { class: 'pb-layers card' });
    this.droEl = h('div', { class: 'pb-dro card' });
    this.programEl = h('div', { class: 'pb-program card' });
    this.transportEl = h('div', { class: 'pb-transport' });
    this.plotsEl = h('div', { class: 'pb-plots card' });
    const side = h('aside', { class: 'pb-side' }, this.droEl, this.programEl);
    this.el.append(h('div', { class: 'pb-top' }, h('div', { class: 'pb-main' }, this.viewEl, this.transportEl, this.layersEl), side), this.plotsEl);
  }

  async init() {
    const { api, data, time } = this;
    this.transport = new Transport(this.transportEl, { time, data, path: () => this.path });
    if (data.pathList.length > 1) {
      const select = h('select', { class: 'input', 'aria-label': 'Controller path', on: { change: () => { this.path = Number(select.value); this.dro.path = this.path; this.transport.drawBase(); this.onTime(); } } },
        data.pathList.map((p) => h('option', { value: String(p) }, `path ${p + 1}`)));
      this.droEl.append(h('label', { class: 'field inline' }, h('span', null, 'Path'), select));
    }
    const header = data.scene && data.scene.frames;
    let frames = null;
    if (header) {
      try { frames = await new FrameStore(api, data.id, header).init(); } catch (error) { this.droEl.append(errorBox(error, 'Frames: ')); }
    } else {
      this.droEl.append(notice('This run has no frames header in scene.json: the DRO shows no positions.'));
    }
    const columns = droColumns(data.channels);
    const tickCols = [columns.load, columns.power, columns.feed, ...columns.overrides.filter((o) => o.tier === 'tick').map((o) => o.name)].filter(Boolean);
    const macroCols = columns.overrides.filter((o) => o.tier === 'macro').map((o) => o.name);
    const tick = new ChannelSampler(api, data.id, tickCols, 'tick');
    const macro = new ChannelSampler(api, data.id, macroCols, 'macro');
    await Promise.all([tick.init().catch(() => tick), macro.init().catch(() => macro)]);
    const tickSet = new Set(tickCols);
    this.samplers = { tick, macro };
    const sampler = { value: (column, t) => (tickSet.has(column) ? tick : macro).value(column, t) };
    this.dro = new Dro(this.droEl, { data, frames, sampler, columns });
    this.dro.path = this.path;
    this.program = new ProgramViewer(this.programEl, {
      data,
      onSeek: (t) => { time.pause(); time.seek(t, 'program'); },
      onFinding: (f) => this.page.focusFinding(f, { tab: 'overview' }),
    });
    this.layers = new LayerControls(this.layersEl, { api, data, getView: () => this.view });
    this.plots = new PlotsPanel(this.plotsEl, { api, data, time, height: matchMedia('(max-width: 640px)').matches ? 128 : 150 });
    const first = data.blockAt(time.t, this.path);
    await this.program.show(first ? { op: first.op, program: first.program } : {});
    time.addEventListener('time', () => this.onTime());
    time.addEventListener('state', (e) => { if (!e.detail.playing) this.refine(); });
    // a running run grows: the DRO's frames and channel overviews follow it (refresh events of run.js)
    data.addEventListener('refresh', () => {
      if (frames) frames.init().catch(() => {});
      tick.init().catch(() => {});
      macro.init().catch(() => {});
    });
    this.onTime();
    this.refine();
    if (data.scene) {
      this.mountView();
    } else {
      this.viewEl.append(notice('The 3D view starts when the run has written its scene (scene.json).'));
      const wait = () => { if (data.scene && !this.view && !this._mounting) { data.removeEventListener('refresh', wait); this.mountView(); } };
      data.addEventListener('refresh', wait);
    }
  }

  mountView() {
    this._mounting = true;
    mountView3D(this.viewEl, {
      api: this.api, runId: this.data.id,
      onPick: (e) => this.layers.showPick(e),
      onMeasure: (e) => this.layers.showMeasure(e),
    }).then((view) => {
      this.view = view;
      if (view) { this.layers.attach(view); view.setTime(this.time.t); }
    });
  }

  onTime() {
    const t = this.time.t;
    if (this.view) this.view.setTime(t);
    if (this.dro) this.dro.update(t);
    if (this.program) {
      this.program.setTime(t);
      const block = this.data.blockAt(t, this.path);
      if (block) this.program.setActive(block.program, block.line);
    }
  }

  refine() {
    const t = this.time.t;
    const ipo = this.data.scene && this.data.scene.frames ? (Number(this.data.scene.frames.ipo_ticks) || 1000) * this.data.tickS : 1e-3;
    if (!this.samplers) return;
    Promise.all([this.samplers.tick.refine(t, 2 * ipo), this.samplers.macro.refine(t, 0.05)])
      .then(() => { if (this.dro && Math.abs(this.time.t - t) < 1e-12) this.dro.update(t); })
      .catch(() => {});
  }

  /** Show a program line (from a finding): selected in the viewer. */
  async showLine(program, line) {
    if (!this.program) return;
    await this.program.selectLine(line, { program, seek: false });
  }

  destroy() {
    if (this.view) this.view.destroy();
    if (this.plots) this.plots.destroy();
    if (this.transport) this.transport.destroy();
  }
}
