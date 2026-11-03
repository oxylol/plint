/* Test double of the 3D view (web/js/view3d/index.js), served by the UI test runner only when the
 * real module is absent, so the shell's side of the contract can be tested on its own (docs/ui.md
 * "3D view"). It reads the scene, the keyframes and the frames through the shell's api object as
 * the real view does, draws the tool path seen from above and the tool at the time it is given on
 * a 2-D canvas, records every call in window.__view3d.calls, and emits 'pick' on a click and
 * 'measure' after two clicks in measure mode. It is not part of the UI. */

export async function createView3D(container, { api, runId }) {
  const scene = await api.scene(runId);
  const keyframes = await api.keyframes(runId);
  const frames = await api.frames(runId, { maxFrames: 4000 });
  const tip = frames.fields.tip_S_m;
  const handlers = { pick: [], measure: [] };
  const state = { t: 0, layer: null, section: null, measure: false, follow: false, visible: {}, picks: [] };
  const record = { calls: [], scene, keyframes: keyframes.count, frames: frames.count, state };
  window.__view3d = record;
  const canvas = document.createElement('canvas');
  canvas.className = 'view3d-stub';
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  canvas.style.display = 'block';
  container.append(canvas);
  const size = scene.stock && scene.stock.size_m ? scene.stock.size_m : [0.1, 0.1, 0.1];
  const draw = () => {
    const w = container.clientWidth || 400;
    const h = container.clientHeight || 300;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#1f2a25';
    ctx.fillRect(0, 0, w, h);
    const scale = Math.min((w - 40) / size[0], (h - 40) / size[1]) * 0.6;
    const ox = (w - size[0] * scale) / 2;
    const oy = (h + size[1] * scale) / 2;
    ctx.fillStyle = state.layer ? '#c47a3a' : '#8a958f';
    ctx.fillRect(ox, oy - size[1] * scale, size[0] * scale, size[1] * scale);
    ctx.strokeStyle = '#e8eeea';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < frames.count; i++) {
      const x = ox + tip.data[i * tip.width] * scale;
      const y = oy - tip.data[i * tip.width + 1] * scale;
      if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    }
    ctx.stroke();
    let k = 0;
    while (k < frames.count - 1 && frames.time_s[k + 1] <= state.t) k++;
    ctx.fillStyle = '#ffd166';
    ctx.beginPath();
    ctx.arc(ox + tip.data[k * tip.width] * scale, oy - tip.data[k * tip.width + 1] * scale, 6, 0, 2 * Math.PI);
    ctx.fill();
  };
  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const point = [(e.clientX - rect.left) / rect.width * size[0], (1 - (e.clientY - rect.top) / rect.height) * size[1], size[2]];
    for (const cb of handlers.pick) cb({ point_m: point, normal: [0, 0, 1], layerValue: state.layer === 'temperature' ? 310.15 : 2.5e-5 });
    if (state.measure) {
      state.picks.push(point);
      if (state.picks.length === 2) {
        const [a, b] = state.picks;
        for (const cb of handlers.measure) cb({ distance_m: Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]), points: [a, b] });
        state.picks = [];
      }
    }
  });
  const call = (name, args) => { record.calls.push([name, ...args]); };
  draw();
  return {
    setTime(t) { call('setTime', [t]); state.t = t; draw(); },
    setLayer(name) { call('setLayer', [name]); state.layer = name; draw(); },
    setSection(section) { call('setSection', [section]); state.section = section; },
    setMeasure(on) { call('setMeasure', [on]); state.measure = !!on; state.picks = []; },
    followTool(on) { call('followTool', [on]); state.follow = !!on; },
    resetCamera() { call('resetCamera', []); },
    setVisible(visible) { call('setVisible', [visible]); Object.assign(state.visible, visible); },
    on(event, cb) { call('on', [event]); if (handlers[event]) handlers[event].push(cb); },
    legend(layer) { call('legend', [layer]); return null; },
    destroy() { call('destroy', []); canvas.remove(); },
  };
}
