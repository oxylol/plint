/* Hosts the 3D view (web/js/view3d/, its own module) inside the run page.
 *
 * Contract (docs/ui.md "3D view"): createView3D(container, {api, runId}) -> view with setTime(t_s),
 * setLayer(name | null), setSection({axis, offset_m, flip}), setMeasure(on), followTool(on),
 * resetCamera(), on('pick' | 'measure', cb), setVisible({...}), destroy(); optional legend(layer).
 * The shell owns time and calls setTime; every call is guarded so a failing view never stops the
 * playback, DRO, program and plots. When the module is missing or cannot start, the panel says so. */

import { h, clear, errorBox } from './dom.js';

const METHODS = ['setTime', 'setLayer', 'setSection', 'setMeasure', 'followTool', 'resetCamera', 'setVisible'];

export async function mountView3D(container, { api, runId, onPick, onMeasure, onStatus }) {
  clear(container);
  const status = h('div', { class: 'view-status', role: 'status' }, 'Loading the 3D view…');
  container.append(status);
  let mod;
  try {
    mod = await import('../view3d/index.js');
  } catch (error) {
    status.replaceWith(errorBox(`The 3D view is not available in this build (${error.message || error}).`));
    if (onStatus) onStatus('missing', error);
    return null;
  }
  if (!mod || typeof mod.createView3D !== 'function') {
    status.replaceWith(errorBox('The 3D view module has no createView3D function.'));
    if (onStatus) onStatus('missing', null);
    return null;
  }
  let view;
  try {
    view = await mod.createView3D(container, { api, runId });
  } catch (error) {
    status.remove();
    container.append(errorBox(`The 3D view could not start: ${error.message || error}`));
    if (onStatus) onStatus('failed', error);
    return null;
  }
  status.remove();
  const failures = new Set();
  const guard = (name, fn) => (...args) => {
    if (!view || typeof view[name] !== 'function') return undefined;
    try {
      return fn ? fn(...args) : view[name](...args);
    } catch (error) {
      if (!failures.has(name)) {
        failures.add(name);
        container.append(errorBox(`3D view: ${name} failed (${error.message || error})`));
      }
      return undefined;
    }
  };
  const host = { raw: view };
  for (const name of METHODS) host[name] = guard(name);
  host.legend = (layer) => {
    for (const name of ['legend', 'layerLegend', 'getLegend']) {
      if (typeof view[name] === 'function') {
        try { return view[name](layer) || null; } catch { return null; }
      }
    }
    return null;
  };
  host.destroy = () => { try { if (typeof view.destroy === 'function') view.destroy(); } catch { /* already gone */ } view = null; };
  if (typeof view.on === 'function') {
    try {
      if (onPick) view.on('pick', (event) => onPick(event));
      if (onMeasure) view.on('measure', (event) => onMeasure(event));
    } catch (error) {
      container.append(errorBox(`3D view: events unavailable (${error.message || error})`));
    }
  }
  if (onStatus) onStatus('ready', null);
  return host;
}
