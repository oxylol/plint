// The work kept on the server (when server.js runs with DATA_DIR): everything the app stores in
// the browser — programs, the job, settings, tool libraries — is mirrored to /api/store, and the
// target model's triangles to /api/blob. The server's copy wins when the page loads, so every
// device opens the same job; localStorage and IndexedDB stay as local caches.
const PRE = 'cncstudio.';
const parse = (raw) => { try { return JSON.parse(raw); } catch { return undefined; } };

export async function openServerStore() {
  try {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), 4000);
    const r = await fetch('/api/store', { cache: 'no-store', signal: ctl.signal });
    clearTimeout(timer);
    const data = r.ok ? await r.json() : null;
    return data?.values ? new ServerStore(data) : null;
  } catch {
    return null; // no server storage (or unreachable): the browser keeps the work as before
  }
}

class ServerStore {
  constructor({ rev, values }) {
    this.rev = rev;
    this.values = values;
    this.pending = new Map();
    this.timer = null;
    this.known = new WeakMap(); // triangle arrays the server already has → their blob id
    this.held = null;
    this.seed();
    addEventListener('pagehide', () => this.flush(true));
    document.addEventListener('visibilitychange', () => (document.hidden ? this.flush(true) : this.checkElsewhere()));
  }

  local() {
    const out = {};
    try {
      for (let i = 0; i < localStorage.length; i++) { const k = localStorage.key(i); if (k?.startsWith(PRE)) out[k.slice(PRE.length)] = localStorage.getItem(k); }
    } catch { /* storage blocked */ }
    return out;
  }

  // what only this browser has goes up (its first visit since the server keeps the work); a
  // different job this browser had is saved in a server-side backup before the server's replaces it
  seed() {
    const local = this.local();
    const differs = (k) => k in local && k in this.values && local[k] !== JSON.stringify(this.values[k]);
    if (differs('ops')) {
      const values = {};
      for (const [k, raw] of Object.entries(local)) values[k] = parse(raw);
      if (values.settings) delete values.settings.apiKey;
      fetch('/api/backup', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ at: new Date().toISOString(), userAgent: navigator.userAgent, values }) }).then((r) => r.text()).catch(() => {});
    }
    for (const [k, raw] of Object.entries(local)) if (!(k in this.values) && !k.startsWith('idb.')) this.put(k, parse(raw));
    // a fresh browser opens with an example: that must not become everyone's job until the user
    // works on it (the app saves while it boots; the first change after a tap or key press goes up)
    if (!('ops' in local) && !('ops' in this.values)) {
      this.held = { last: null, armed: false };
      const arm = () => { if (this.held) this.held.armed = true; };
      addEventListener('pointerdown', arm, { capture: true, once: true });
      addEventListener('keydown', arm, { capture: true, once: true });
    }
    try {
      for (const k of Object.keys(this.values)) if (!k.startsWith('idb.')) localStorage.setItem(PRE + k, JSON.stringify(this.get(k)));
    } catch { /* storage blocked: the server copy is still used */ }
  }

  // a key typed into Settings stays in this browser only
  get(k, d) {
    if (!(k in this.values)) return d;
    if (k !== 'settings') return this.values[k];
    let apiKey = '';
    try { apiKey = parse(localStorage.getItem(PRE + 'settings'))?.apiKey || ''; } catch { /* none */ }
    return { ...this.values.settings, apiKey };
  }

  put(k, v) {
    if (v === undefined) v = null;
    if (k === 'ops' && this.held) {
      const json = JSON.stringify(v);
      if (!this.held.armed || json === this.held.last) { this.held.last = json; return; }
      this.held = null;
    }
    if (k === 'settings' && v) { const { apiKey, ...rest } = v; v = rest; }
    if (v === null) delete this.values[k]; else this.values[k] = v;
    this.pending.set(k, v);
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.flush(), 500);
  }

  async flush(leaving = false) {
    clearTimeout(this.timer);
    const batch = [...this.pending];
    this.pending.clear();
    for (const [k, v] of batch) {
      const body = JSON.stringify(v);
      try {
        // keepalive lets a write finish while the page closes (up to 64 KB)
        const r = await fetch(`/api/store/${encodeURIComponent(k)}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body, keepalive: leaving && body.length < 60000 });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        this.rev = (await r.json()).rev;
      } catch {
        if (!this.pending.has(k)) this.pending.set(k, v);
        if (!leaving) { clearTimeout(this.timer); this.timer = setTimeout(() => this.flush(), 5000); }
      }
    }
  }

  // back on this tab after another device saved: offer the newer work instead of overwriting it
  async checkElsewhere() {
    if (this.pending.size) return;
    try {
      const { rev } = await (await fetch('/api/store/rev', { cache: 'no-store' })).json();
      if (rev === this.rev) return;
      this.rev = rev;
      if (confirm('This job was changed on another device. Load the latest version?')) location.reload();
    } catch { /* offline: keep working */ }
  }

  // the target model: its fields as JSON under idb.<key>, the triangles as a blob sent only when they change
  async getBlob(k, local, cache) {
    const meta = this.values[`idb.${k}`];
    if (!meta) {
      if (local?.tris) this.putBlob(k, local).then((id) => id && cache(k, { ...local, blob: id }));
      return local;
    }
    if (local?.tris && local.blob === meta.blob) {
      this.known.set(local.tris, meta.blob);
      return { ...local, ...meta };
    }
    try {
      const r = await fetch(`/api/blob/${encodeURIComponent(k)}`, { cache: 'no-store' });
      if (!r.ok) return local;
      const v = { ...meta, tris: new Float32Array(await r.arrayBuffer()) };
      this.known.set(v.tris, meta.blob);
      cache(k, v);
      return v;
    } catch {
      return local;
    }
  }

  // → the blob id the server has the triangles under (null: not sent, they stay in this browser)
  async putBlob(k, v) {
    if (!v) {
      this.put(`idb.${k}`, null);
      fetch(`/api/blob/${encodeURIComponent(k)}`, { method: 'DELETE' }).then((r) => r.text()).catch(() => {});
      return null;
    }
    const { tris, ...meta } = v;
    let id = this.known.get(tris);
    if (!id) {
      id = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
      try {
        const arr = tris instanceof Float32Array ? tris : Float32Array.from(tris);
        const r = await fetch(`/api/blob/${encodeURIComponent(k)}`, { method: 'PUT', headers: { 'Content-Type': 'application/octet-stream' }, body: arr });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        this.rev = (await r.json()).rev;
        this.known.set(tris, id);
      } catch {
        return null; // sent again the next time the model changes
      }
    }
    this.put(`idb.${k}`, { ...meta, blob: id });
    return id;
  }

  // route the app's own storage helpers through the server
  wrap(store, idb) {
    const lsSet = store.set.bind(store);
    store.get = (k, d) => this.get(k, d);
    store.set = (k, v) => { lsSet(k, v); this.put(k, v); };
    const dbGet = idb.get.bind(idb), dbSet = idb.set.bind(idb);
    idb.get = async (k) => this.getBlob(k, await dbGet(k), dbSet);
    idb.set = async (k, v) => { const id = await this.putBlob(k, v); await dbSet(k, v && id ? { ...v, blob: id } : v); };
  }
}
