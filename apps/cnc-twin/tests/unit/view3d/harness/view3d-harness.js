// Test page of the 3D view (tests/unit/view3d): mounts web/js/view3d on one run, as the shell does,
// with the shell's API client (default) or with the contract's minimal api {get, getBinary} (?api=minimal),
// and exposes the view and its events to the browser test (window.view3d, window.__events).
import { createApi } from './js/api.js';
import { createView3D } from './js/view3d/index.js';

const query = new URLSearchParams(location.search);
const runId = query.get('run');
const minimal = {
  async get(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`${response.status} ${path}`);
    return response.json();
  },
  async getBinary(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`${response.status} ${path}`);
    return response.arrayBuffer();
  },
};
const api = query.get('api') === 'minimal' ? minimal : createApi({ base: '' });
window.__events = { pick: [], measure: [] };
try {
  const started = performance.now();
  const view = await createView3D(document.getElementById('view'), { api, runId });
  view.on('pick', (event) => window.__events.pick.push(event));
  view.on('measure', (event) => window.__events.measure.push(event));
  view.setVisible({ machine: true, fixture: true, chips: true, coolant: true, shadows: false, tool: true, deflectedTool: false });
  view.setTime(Number(query.get('t') || 0));
  window.view3d = view;
  window.__mountMs = performance.now() - started;
  window.__ready = true;
} catch (error) {
  window.__error = String((error && error.stack) || error);
  window.__ready = true;
}
