// Projects: the job — operations, subprograms, the target model and its placement — saved under
// a name on the server (DATA_DIR/projects/<id>/: project.json + model.bin), to open again from any
// device. The work itself still lives in the store as before; a project is a named copy of it, and
// the app remembers which one is open so Save goes back into it and the button shows its name.
export function initProjects({ enabled, state, store, idb, syncEditorToOp, loadJob, toast, escapeHtml }) {
  const $ = (s) => document.querySelector(s);
  const btn = $('#btnProjects'), dlg = $('#projectsDlg');
  if (!enabled) { btn.hidden = true; return { changed() {}, detach() {}, open() {}, save() {} }; }

  let current = store.get('project', null); // { id, name, sig, modelSig, savedAt }
  let list = [];

  const hash = (s) => { let h = 5381; for (let i = 0; i < s.length; i++) h = ((h * 33) ^ s.charCodeAt(i)) >>> 0; return h.toString(36); };
  const modelSig = () => (state.model ? `${state.model.name}|${state.model.triCount}` : '');
  const signature = () => hash(JSON.stringify([state.ops.map((o) => [o.fileName, o.machine, o.text]), state.subprograms.map((s) => [s.name, s.text]), modelSig(), state.placement]));
  const dirty = () => !!current && signature() !== current.sig;
  const when = (iso) => (iso ? new Date(iso).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' }) : '');
  const newId = () => `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
  const suggestedName = () => (state.ops[0]?.fileName || 'Job').replace(/\.[^.]+$/, '');

  async function api(path, opt) {
    const r = await fetch(`/api/projects${path}`, { cache: 'no-store', ...opt });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `HTTP ${r.status}`);
    return r;
  }

  // the button in the top bar carries the open project's name, with a dot while it has unsaved changes
  function updateChip() {
    const label = $('#projectName');
    btn.classList.toggle('open-project', !!current);
    if (!current) { label.textContent = 'Projects'; btn.classList.remove('dirty'); btn.title = 'Projects kept on the server: open one, or save this job as a project'; return; }
    const d = dirty();
    label.textContent = current.name;
    btn.classList.toggle('dirty', d);
    btn.title = d ? `Project “${current.name}” — changed since it was saved` : `Project “${current.name}” — saved ${when(current.savedAt)}`;
  }
  let chipTimer = null;
  const changed = () => { clearTimeout(chipTimer); chipTimer = setTimeout(updateChip, 150); };
  // every change to the job goes through the store (operations, subprograms) or idb (the model)
  const set = store.set.bind(store);
  store.set = (k, v) => { set(k, v); if (k !== 'project') changed(); };
  const dbSet = idb.set.bind(idb);
  idb.set = async (k, v) => { await dbSet(k, v); changed(); };

  async function save({ id = current?.id, name = current?.name } = {}) {
    name = (name || '').trim();
    if (!name) { toast('Give the project a name'); return false; }
    syncEditorToOp();
    const ops = state.ops.map(({ fileName, machine, text }) => ({ fileName, machine, text }));
    if (!ops.some((o) => o.text.trim())) { toast('Nothing to save yet'); return false; }
    if (!id) id = newId();
    const m = state.model;
    const body = { name, ops, subprograms: state.subprograms, model: m ? { name: m.name, format: m.format, triCount: m.triCount, bounds: m.bounds, warnings: m.warnings || [], placement: state.placement } : null };
    try {
      // the triangles go up only when the project does not have this model yet
      if (m && !(id === current?.id && current.modelSig === modelSig())) {
        const tris = m.tris instanceof Float32Array ? m.tris : Float32Array.from(m.tris);
        await api(`/${id}/model`, { method: 'PUT', headers: { 'Content-Type': 'application/octet-stream' }, body: tris });
      }
      const { savedAt } = await (await api(`/${id}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })).json();
      current = { id, name, sig: signature(), modelSig: modelSig(), savedAt };
      store.set('project', current);
      updateChip();
      toast(`Saved project “${name}”`);
      return true;
    } catch (e) {
      toast(`Could not save the project: ${e.message}`, 5000);
      return false;
    }
  }

  async function open(id, name) {
    if (dirty() && !confirm(`“${current.name}” has unsaved changes that will be lost. Open “${name}” anyway?`)) return;
    try {
      const p = await (await api(`/${id}`)).json();
      let model = null;
      if (p.model) model = { ...p.model, tris: new Float32Array(await (await api(`/${id}/model`)).arrayBuffer()) };
      loadJob({ ops: p.ops, subprograms: p.subprograms, model }, p.name, { quiet: true });
      current = { id, name: p.name, sig: signature(), modelSig: modelSig(), savedAt: p.savedAt };
      store.set('project', current);
      updateChip();
      dlg.close();
      toast(`Opened project “${p.name}”`);
    } catch (e) {
      toast(`Could not open the project: ${e.message}`, 5000);
    }
  }

  async function rename(id, oldName) {
    const name = (prompt('Project name', oldName) || '').trim();
    if (!name || name === oldName) return;
    try {
      await api(`/${id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }) });
      if (current?.id === id) { current = { ...current, name }; store.set('project', current); updateChip(); }
      await refresh();
    } catch (e) {
      toast(`Could not rename: ${e.message}`, 5000);
    }
  }

  async function remove(id, name) {
    if (!confirm(`Delete project “${name}” from the server? This cannot be undone.`)) return;
    try {
      await api(`/${id}`, { method: 'DELETE' });
      if (current?.id === id) detach();
      await refresh();
      toast(`Deleted “${name}”`);
    } catch (e) {
      toast(`Could not delete: ${e.message}`, 5000);
    }
  }

  // the job stops being the open project (New, or a job file opened from disk)
  function detach() {
    if (!current) return;
    current = null;
    store.set('project', null);
    updateChip();
  }

  async function refresh() {
    try {
      list = (await (await api('')).json()).projects;
    } catch (e) {
      $('#projectList').innerHTML = `<div class="muted">Could not load the projects: ${escapeHtml(e.message)}</div>`;
      return;
    }
    render();
  }

  function render() {
    $('#projectCurrent').innerHTML = current
      ? `<div><b>${escapeHtml(current.name)}</b> is open · ${dirty() ? '<span class="warn-text">changed since it was saved</span>' : `saved ${when(current.savedAt)}`}</div><button type="button" class="btn small primary" data-act="save">Save</button>`
      : '<div class="muted">This job is not a project yet: give it a name and save it, or open one below.</div>';
    $('#projectList').innerHTML = list.length
      ? list.map((p) => {
        const facts = [`${p.ops} operation${p.ops === 1 ? '' : 's'}`, p.machines.join(' + ')];
        if (p.model) facts.push(`model ${p.model}`);
        if (p.subprograms) facts.push(`${p.subprograms} subprogram${p.subprograms === 1 ? '' : 's'}`);
        facts.push(`saved ${when(p.savedAt)}`);
        return `<div class="project-row${p.id === current?.id ? ' current' : ''}" data-id="${p.id}"><div class="project-main"><b>${escapeHtml(p.name)}</b><div class="muted">${escapeHtml(facts.join(' · '))}</div></div><div class="project-acts"><button type="button" class="btn small" data-act="open">Open</button><button type="button" class="btn small ghost" data-act="rename">Rename</button><button type="button" class="btn small ghost" data-act="delete" title="Delete from the server">Delete</button></div></div>`;
      }).join('')
      : '<div class="muted">No projects saved yet.</div>';
  }

  btn.onclick = () => {
    $('#projectNewName').value = current ? '' : suggestedName();
    $('#projectNewName').placeholder = current ? `Save a copy as…` : 'Project name';
    $('#projectList').innerHTML = '<div class="muted">Loading…</div>';
    render();
    dlg.showModal();
    refresh();
  };
  $('#btnProjectSaveAs').onclick = async () => {
    const name = $('#projectNewName').value.trim();
    if (!name) { $('#projectNewName').focus(); toast('Give the project a name'); return; }
    if (await save({ id: null, name })) { $('#projectNewName').value = ''; refresh(); }
  };
  $('#projectNewName').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); $('#btnProjectSaveAs').click(); } });
  dlg.addEventListener('click', (e) => {
    const b = e.target.closest('button[data-act]');
    if (!b) return;
    const row = b.closest('.project-row');
    const id = row?.dataset.id, name = row?.querySelector('b')?.textContent;
    if (b.dataset.act === 'save') save().then((ok) => ok && refresh());
    else if (b.dataset.act === 'open') open(id, name);
    else if (b.dataset.act === 'rename') rename(id, name);
    else if (b.dataset.act === 'delete') remove(id, name);
  });
  updateChip();
  return { changed, detach, open, save };
}
