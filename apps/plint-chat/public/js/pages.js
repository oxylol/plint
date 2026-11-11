// Full pages (Chats, Projects, a project, Artifacts, Scheduled tasks, Customize) and
// the Search and Settings dialogs.
import { h, icon, menu, modal, closeModal, formModal, confirmModal, toast, ago, agoLong, when, bytes } from "./ui.js";
import { api, upload } from "./api.js";
import { state, on, refreshChats, refreshProjects, refreshStyles, projectName, prefs } from "./state.js";
import { composer } from "./chat.js";
import { artifactTile, openArtifact, closePanel } from "./artifacts.js";
import { chatMenu, setTheme } from "./sidebar.js";
import { topBar } from "./main.js";

// The frame every list page shares: a serif title, then tabs on the left and actions on the right.
// tabs: [[key, label]], onTab(key); actions: nodes; search: { placeholder, onInput } adds the search action.
export function page(view, { title, tabs = null, tab = null, onTab = null, actions = [], search = null }, ...body) {
  topBar({ title: "" });
  document.getElementById("app").classList.add("no-top");
  let field = null;
  const searchBtn = search ? h("button.icon-btn", { type: "button", "aria-label": search.placeholder, title: "Search", on: { click: () => {
    if (field) { field.remove(); field = null; search.onInput(""); return; }
    field = h("input.input", { type: "search", placeholder: search.placeholder, "aria-label": search.placeholder, on: { input: (e) => search.onInput(e.target.value.toLowerCase()), keydown: (e) => { if (e.key === "Escape") { e.stopPropagation(); searchBtn.click(); } } } });
    searchBtn.before(field);
    field.focus();
  } } }, icon("search")) : null;
  const acts = h("div.page-actions", searchBtn, actions);
  const tabBar = tabs ? h("div.tabs", { role: "tablist" }, tabs.map(([k, l]) => h(`button${k === tab ? ".on" : ""}`, { type: "button", role: "tab", "aria-selected": String(k === tab), on: { click: (e) => { for (const b of tabBar.children) { b.classList.toggle("on", b === e.currentTarget); b.setAttribute("aria-selected", String(b === e.currentTarget)); } onTab(k); } } }, l))) : null;
  const head = h("div.page-head", tabs
    ? [h("div.page-title", h("h1", title)), h("div.page-bars", tabBar, acts)]
    : h("div.page-title", h("h1", title), h("span.grow"), acts));
  const p = h("div.page", head, ...body);
  view.replaceChildren(h("div.scroll", p));
  return p;
}
const chatRow = (c) => h("a.list-item", { href: `#/chat/${c.id}` },
  h("span.t", c.title),
  c.running ? h("i.dot", { title: "Running" }) : null,
  h("button.icon-btn", { type: "button", "aria-label": `More options for ${c.title}`, "aria-haspopup": "menu", on: { click: (e) => { e.preventDefault(); e.stopPropagation(); chatMenu(e.currentTarget, c, { header: true }); } } }, icon("more")),
  h("span.s", agoLong(c.updated)));

// --- projects -------------------------------------------------------------------

export function newProject() {
  return formModal({
    title: "Create a project",
    fields: [
      { name: "name", label: "What are you working on?", placeholder: "Name your project", autofocus: true },
      { name: "description", label: "What are you trying to achieve?", type: "textarea", rows: 3, placeholder: "Describe your project, goals, subject, etc..." },
    ],
    submit: "Create project",
    onSubmit: async (d) => { if (!d.name.trim()) throw new Error("Give the project a name."); const p = await api("projects", { body: d }); await refreshProjects(); location.hash = `#/project/${p.id}`; },
  });
}

export function renderProjects(view, { create = false } = {}) {
  closePanel();
  let q = "";
  const grid = h("div.card-grid");
  const draw = () => {
    const list = state.projects.filter((p) => !q || `${p.name} ${p.description}`.toLowerCase().includes(q));
    grid.replaceChildren(...list.map((p) => h("a.card.one", { href: `#/project/${p.id}` },
      h("span.tile", icon("projects", "s16")),
      h("span.col", h("span.t", p.name), h("span.s", `Updated ${agoLong(p.updated)}`)))));
    if (!list.length) grid.replaceChildren(h("div.empty-line", { style: { gridColumn: "1/-1" } }, q ? "No projects match your search." : "No projects yet. Upload materials, set custom instructions, and organize chats in one space."));
  };
  page(view, { title: "Projects", search: { placeholder: "Search projects", onInput: (v) => { q = v; draw(); } },
    actions: [h("button.btn.primary", { type: "button", on: { click: newProject } }, "New project")] }, grid);
  draw();
  const off = on("projects", draw);
  refreshProjects();
  if (create) newProject();
  return off;
}

export function renderProject(view, id) {
  closePanel();
  let p = state.projects.find((x) => x.id === id);
  const chatsBox = h("div.list");
  const filesBox = h("div");
  const instr = h("div.body");
  const fileInput = h("input", { type: "file", multiple: true, hidden: true });
  const c = composer({
    projectId: id,
    onSend: async (text, files, settings) => {
      const chat = await api("chats", { body: { text, files, ...settings, projectId: id } });
      refreshChats();
      location.hash = `#/chat/${chat.id}`;
    },
  });
  const title = h("h1", p?.name || "");
  const desc = h("p.proj-desc", { hidden: true });
  const crumb = () => topBar({ title: "Projects", onTitleClick: () => { location.hash = "#/projects"; }, badges: [h("span.faint", "/"), h("span.btn.title-btn", { style: { cursor: "default" } }, h("span.t", p?.name || ""))] });
  async function load() {
    try { p = await api(`projects/${id}`); } catch { toast("That project is gone."); location.hash = "#/projects"; return; }
    title.textContent = p.name;
    desc.textContent = p.description || "";
    desc.hidden = !p.description;
    instr.textContent = p.instructions || "Add instructions to tailor replies in this project.";
    const files = await api(`projects/${id}/files`);
    filesBox.replaceChildren(...(files.length ? files.map((f) => h("div.file-row", icon("file", "s16"),
      h("a.grow.ellipsis", { href: `/api/projects/${id}/files/${encodeURIComponent(f.name)}`, target: "_blank" }, f.name), h("span.faint", bytes(f.size)),
      h("button.icon-btn.sm", { type: "button", "aria-label": `Remove ${f.name}`, on: { click: async () => { await api(`projects/${id}/files/${encodeURIComponent(f.name)}`, { method: "DELETE" }); load(); } } }, icon("x", "s16"))))
      : [h("div.body", "Plint can read these in every chat in this project.")]));
    drawChats();
    crumb();
  }
  function drawChats() {
    const list = state.chats.filter((x) => x.projectId === id);
    chatsBox.replaceChildren(...(list.length ? list.map(chatRow) : [h("div.empty-line", "Start a chat to keep conversations organized and re-use reference materials.")]));
  }
  fileInput.onchange = async () => {
    for (const f of fileInput.files) { try { await upload(`projects/${id}/files`, f); } catch (e) { toast(e.message); } }
    fileInput.value = "";
    load();
  };
  const editInstructions = () => formModal({
    title: "Instructions", submit: "Save", text: `Give Plint relevant instructions and information for chats within ${p.name}.`,
    fields: [{ name: "instructions", type: "textarea", rows: 10, value: p.instructions || "", placeholder: "E.g. Work in /opt/plint-table. Answer in Swedish. Always run the tests.", autofocus: true }],
    onSubmit: async (d) => { await api(`projects/${id}`, { body: d }); load(); },
  });
  const projMenu = (e) => menu(e.currentTarget, [
    { label: "Edit details", onClick: () => formModal({ title: "Edit details", fields: [{ name: "name", label: "Name", value: p.name }, { name: "description", label: "Description", type: "textarea", rows: 6, value: p.description }],
      onSubmit: async (d) => { await api(`projects/${id}`, { body: d }); refreshProjects(); load(); } }) },
    { label: "Working folder", right: p.cwd, onClick: () => formModal({ title: "Working folder", fields: [{ name: "cwd", label: "Claude Code starts in", value: p.cwd, hint: "A folder on Plint, e.g. /opt/plint-table" }],
      onSubmit: async (d) => { const r = await api(`projects/${id}`, { body: d }); if (r.cwd !== d.cwd) throw new Error("That folder doesn't exist on Plint."); load(); } }) },
    "sep",
    { label: "Delete", danger: true, onClick: async () => {
      if (!(await confirmModal("Delete project", `“${p.name}” will be deleted. Its chats stay, but leave the project.`))) return;
      await api(`projects/${id}`, { method: "DELETE" }); refreshProjects(); refreshChats(); toast("Project deleted"); location.hash = "#/projects";
    } },
  ], { align: "end" });
  document.getElementById("app").classList.remove("no-top");
  crumb();
  view.replaceChildren(h("div.scroll", h("div.proj",
    h("div.proj-head", title, h("button.icon-btn", { type: "button", "aria-label": "Project options", "aria-haspopup": "menu", on: { click: projMenu } }, icon("more"))),
    desc,
    h("div.proj-pair",
      h("div.proj-main", c.el, h("div", h("div.lbl", "Chats"), chatsBox)),
      h("div.proj-rail",
        h("div.rr", h("span.grow", "Instructions"), h("button.icon-btn", { type: "button", "aria-label": "Edit instructions", title: "Edit instructions", on: { click: editInstructions } }, icon("edit"))), instr,
        h("div.rr", h("span.grow", "Files"), h("button.icon-btn", { type: "button", "aria-label": "Add files", title: "Add files", on: { click: () => fileInput.click() } }, icon("plus"))), h("div", { style: { padding: "8px 0" } }, filesBox), fileInput)))));
  load();
  return on("chats", drawChats);
}

// --- artifacts ------------------------------------------------------------------

export function renderArtifacts(view) {
  const grid = h("div.art-grid");
  let q = "";
  let all = [];
  const draw = () => {
    const list = all.filter((a) => !q || `${a.title} ${a.chatTitle}`.toLowerCase().includes(q));
    grid.replaceChildren(...list.map((a) => artifactTile(a, { onMenu: (anchor) => menu(anchor, [
      { label: "Open", onClick: () => openArtifact(a.id) },
      { label: "Go to chat", onClick: () => { location.hash = `#/chat/${a.chatId}`; } },
      { label: "Download", onClick: () => { location.href = `/api/artifacts/${a.id}/raw?download=1`; } },
      "sep",
      { label: "Delete", danger: true, onClick: async () => { if (await confirmModal("Delete artifact", `Are you sure you want to delete “${a.title}”?`)) { await api(`artifacts/${a.id}`, { method: "DELETE" }); load(); } } },
    ], { align: "end" }) })));
    if (!list.length) grid.replaceChildren(h("div.empty-line", { style: { gridColumn: "1/-1" } }, q ? "No artifacts match your search." : "Ask for a page, a diagram, a chart or a document and it shows up here."));
  };
  const load = async () => { all = await api("artifacts").catch(() => []); draw(); };
  page(view, { title: "Artifacts", search: { placeholder: "Search artifacts", onInput: (v) => { q = v; draw(); } },
    actions: [h("a.btn.primary", { href: "#/" }, "New artifact")] }, grid);
  load();
  return on("artifacts", load);
}

// --- scheduled ------------------------------------------------------------------

const EVERY = [["hourly", "Every hour"], ["daily", "Every day"], ["weekdays", "Weekdays"], ["weekly", "Every week"], ["interval", "Every N minutes"]];
const DAYS = [[1, "Monday"], [2, "Tuesday"], [3, "Wednesday"], [4, "Thursday"], [5, "Friday"], [6, "Saturday"], [0, "Sunday"]];
function taskForm(t = {}) {
  return formModal({
    title: t.id ? "Edit task" : "New scheduled task",
    submit: t.id ? "Save" : "Create task",
    fields: [
      { name: "name", label: "Name", value: t.name || "", placeholder: "Morning check", autofocus: !t.id },
      { name: "prompt", label: "Prompt", type: "textarea", rows: 5, value: t.prompt || "", placeholder: "Check plint status and the last backup. Tell me only if something is wrong." },
      { name: "every", label: "Repeat", type: "select", value: t.every || "daily", options: EVERY },
      { name: "time", label: "Time", type: "time", value: t.time || "08:00", show: (v, d) => (d?.every ?? t.every ?? "daily") !== "interval" },
      { name: "day", label: "Day", type: "select", value: t.day ?? 1, options: DAYS, show: (v, d) => (d?.every ?? t.every) === "weekly" },
      { name: "minutes", label: "Every how many minutes", type: "number", value: t.minutes || 60, show: (v, d) => (d?.every ?? t.every) === "interval", hint: "At least 15." },
      { name: "model", label: "Model", type: "select", value: t.model || "auto-sonnet", options: Object.entries(state.models).map(([id, m]) => [id, m.name]) },
      { name: "effort", label: "Effort", type: "select", value: t.effort || "low", options: state.efforts.map((e) => [e, e[0].toUpperCase() + e.slice(1)]), show: (v, d) => state.models[d?.model ?? t.model]?.efforts },
      { name: "mode", label: "When it wants to run something", type: "select", value: t.mode || "manual", options: [["manual", "Ask me (sent to my phone)"], ["auto", "Just do it"]] },
      { name: "projectId", label: "Project", type: "select", value: t.projectId || "", options: [["", "No project"], ...state.projects.map((p) => [p.id, p.name])] },
    ],
    onSubmit: async (d) => {
      const body = { ...d, projectId: d.projectId || null, day: Number(d.day), minutes: Number(d.minutes) };
      await api(t.id ? `schedules/${t.id}` : "schedules", { body });
    },
  });
}

export function renderScheduled(view, { create = false } = {}) {
  closePanel();
  const list = h("div.card-grid.wide");
  let tab = "active";
  let q = "";
  let tasks = [];
  const draw = () => {
    const shown = tasks.filter((t) => (tab === "active" ? t.enabled : !t.enabled) && (!q || `${t.name} ${t.prompt}`.toLowerCase().includes(q)));
    list.replaceChildren(...(shown.length ? shown.map((t) => {
      const sw = h(`button.sw${t.enabled ? ".on" : ""}`, { type: "button", role: "switch", "aria-checked": String(!!t.enabled), "aria-label": t.enabled ? "Pause" : "Enable", title: t.enabled ? "Pause" : "Enable",
        on: { click: async () => { await api(`schedules/${t.id}`, { body: { enabled: !t.enabled } }).catch(() => toast(`Schedule couldn’t be ${t.enabled ? "paused" : "enabled"}. You can try again.`)); load(); } } });
      return h("div.card",
        h("div.top-row", h("span.t", t.name), sw,
          h("button.icon-btn", { type: "button", "aria-label": `More options for ${t.name}`, "aria-haspopup": "menu", on: { click: (e) => menu(e.currentTarget, [
            { label: "Run now", onClick: async () => { try { await api(`schedules/${t.id}/run`, { body: {} }); toast("Run started."); refreshChats(); load(); } catch (err) { toast(err.message); } } },
            { label: t.enabled ? "Pause" : "Enable", onClick: () => sw.click() },
            { label: "Edit", onClick: () => taskForm(t).then((r) => r && load()) },
            "sep",
            { label: "Delete", danger: true, onClick: async () => { if (await confirmModal("Delete scheduled task?", `Delete “${t.name}”? Chats it made stay.`)) { await api(`schedules/${t.id}`, { method: "DELETE" }); load(); } } },
          ], { align: "end" }) } }, icon("more"))),
        h("div.d", t.prompt),
        h("div.s",
          h("span", t.when),
          t.enabled ? h("span", `Next run ${when(t.nextRun)}`) : h("span", "Paused"),
          t.lastChatId ? h("a", { href: `#/chat/${t.lastChatId}` }, `Last run ${agoLong(t.lastRun)}`) : h("span", "No runs yet"),
          t.mode === "auto" ? h("span", "Auto") : null,
          t.projectId ? h("span", projectName(t.projectId) || "Project") : null));
    }) : [h("div.empty-line", { style: { gridColumn: "1/-1" } }, q ? "No scheduled tasks match your search." : tab === "active" ? "No scheduled tasks yet." : "Nothing is paused.")]));
  };
  const load = async () => { tasks = await api("schedules").catch(() => []); draw(); };
  const add = () => taskForm().then((r) => r && load());
  page(view, { title: "Scheduled tasks", tabs: [["active", "Active"], ["paused", "Paused"]], tab, onTab: (k) => { tab = k; draw(); },
    search: { placeholder: "Search scheduled tasks…", onInput: (v) => { q = v; draw(); } },
    actions: [h("button.btn.primary", { type: "button", on: { click: add } }, "New task")] }, list);
  load();
  if (create) { history.replaceState(null, "", "#/scheduled"); add(); }
  const t = setInterval(load, 30000);
  return () => clearInterval(t);
}

// --- customize ------------------------------------------------------------------

export function renderCustomize(view, tab = "skills") {
  closePanel();
  const body = h("div");
  const TABS = [["skills", "Skills"], ["styles", "Styles"], ["instructions", "Instructions"], ["connectors", "Connectors"]];
  if (!TABS.some(([k]) => k === tab)) tab = "skills";
  const addBtn = h("button.btn.primary", { type: "button", hidden: true }, "Add");
  async function draw() {
    const s = await api("settings");
    addBtn.hidden = tab !== "styles";
    if (tab === "instructions") {
      const ta = h("textarea.input", { rows: 10, placeholder: "E.g. Keep answers short, I read on my phone. I prefer Swedish for chat, English for code." }, s.instructions || "");
      body.replaceChildren(h("section.s-group", h("h3", "Instructions"), h("p", "Added to every chat, on top of CLAUDE.md and any project instructions."),
        h("div.s-row.stack", h("div.ctl", ta)),
        h("div.s-row", h("div.txt"), h("div.ctl", h("button.btn.primary", { type: "button", on: { click: async () => { await api("settings", { body: { instructions: ta.value } }); toast("Instructions saved."); } } }, "Save")))));
    } else if (tab === "styles") {
      const custom = s.styles;
      const edit = (st) => formModal({
        title: st ? "Edit style" : "Create a style",
        fields: [{ name: "name", label: "Name", value: st?.name || "", autofocus: true }, { name: "prompt", label: "Instructions", type: "textarea", rows: 6, value: st?.prompt || "", placeholder: "E.g. Write like a laid-back sysadmin. Short lines. No emoji." }],
        onSubmit: async (d) => {
          const styles = st ? custom.map((x) => (x.id === st.id ? { ...x, ...d } : x)) : [...custom, d];
          await api("settings", { body: { styles } }); await refreshStyles(); draw();
        },
      });
      addBtn.onclick = () => edit(null);
      body.replaceChildren(h("section.s-group", h("h3", "Yours"), h("div.c-list", s.allStyles.map((st) => h("div.c-row",
        h("span.tile", icon("styles")),
        h("div.col", h("span.t", st.name), h("span.p", st.prompt || "Plain replies, no extra style.")),
        h("div.end", s.defaultStyle === st.id ? h("span", "Default") : null,
          h("button.icon-btn", { type: "button", "aria-label": `More options for ${st.name}`, "aria-haspopup": "menu", on: { click: (e) => menu(e.currentTarget, [
            { label: "Make default", disabled: s.defaultStyle === st.id, onClick: async () => { await api("settings", { body: { defaultStyle: st.id } }); draw(); } },
            st.builtin ? null : { label: "Edit", onClick: () => edit(st) },
            st.builtin ? null : "sep",
            st.builtin ? null : { label: "Delete", danger: true, onClick: async () => { await api("settings", { body: { styles: custom.filter((x) => x.id !== st.id) } }); await refreshStyles(); draw(); } },
          ], { align: "end" }) } }, icon("more"))))))));
    } else if (tab === "skills") {
      const skills = await api("skills").catch(() => []);
      body.replaceChildren(h("section.s-group", h("h3", "Yours"), h("p", "Skills Claude Code on Plint can load, from ~/.claude/skills and /opt/plint/.claude/skills."),
        skills.length ? h("div.c-list", skills.map((sk) => h("div.c-row", h("span.tile", icon("book")), h("div.col", h("span.t", sk.name), h("span.p", sk.description)), h("div.end", h("span", sk.where === "project" ? "Project" : "Personal"))))) : h("div.empty-line", "No skills yet.")));
    } else {
      body.replaceChildren(h("section.s-group", h("h3", "Connectors"), h("p", "Plint Chat runs Claude Code on Plint, so it uses the tools and MCP servers set up for Claude Code there (claude mcp list). claude.ai's own connectors (Gmail, Drive…) need Anthropic's servers and aren't available with Silk models.")));
    }
  }
  page(view, { title: "Customize", tabs: TABS, tab, onTab: (k) => { tab = k; history.replaceState(null, "", `#/customize/${k}`); draw(); }, actions: [addBtn] }, body);
  draw();
}

// --- chats (view all) ---------------------------------------------------------------

export function renderRecents(view) {
  closePanel();
  let q = "";
  let tab = "chat";
  const list = h("div.list");
  const draw = () => {
    const items = state.chats.filter((c) => (tab === "code") === (c.kind === "code") && (!q || c.title.toLowerCase().includes(q)));
    list.replaceChildren(...(items.length ? items.map(chatRow) : [h("div.empty-line", q ? "No chats match your search." : tab === "code" ? "Sessions you start will show up here" : "Chats you start will show up here")]));
  };
  page(view, { title: "Chats", tabs: [["chat", "Chats"], ["code", "Code"]], tab, onTab: (k) => { tab = k; draw(); },
    search: { placeholder: "Search chats…", onInput: (v) => { q = v; draw(); } },
    actions: [h("a.btn.primary", { href: "#/" }, "New chat")] }, list);
  draw();
  return on("chats", draw);
}

// --- search ------------------------------------------------------------------------

export function openSearch() {
  let sel = 0;
  let items = [];
  const input = h("input", { placeholder: "Search chats and projects", autofocus: true, "aria-label": "Search" });
  const results = h("div.search-results");
  const go = (c) => { closeModal(); location.hash = c.href || `#/chat/${c.id}`; };
  const draw = () => {
    const q = input.value.trim();
    results.replaceChildren(
      items.length ? h("div.sec", q ? "Search results" : "Recents") : null,
      ...(items.length ? items.map((c, i) => h(`a.sr${i === sel ? ".sel" : ""}`, { href: c.href || `#/chat/${c.id}`, on: { click: (e) => { e.preventDefault(); go(c); }, pointermove: () => { if (sel !== i) { sel = i; results.querySelector(".sel")?.classList.remove("sel"); results.querySelectorAll(".sr")[i]?.classList.add("sel"); } } } },
        icon(c.href ? "projects" : c.kind === "code" ? "code" : "chat"),
        h("div.grow", h("div.t", c.title || c.name), c.snippet ? h("div.s", c.snippet) : null),
        c.updated ? h("span.d", ago(c.updated)) : null)) : [h("div.empty-note", q ? "No results" : "No chats yet")]));
    results.querySelector(".sel")?.scrollIntoView({ block: "nearest" });
  };
  let timer;
  input.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(async () => {
      const q = input.value.trim();
      const projects = q ? state.projects.filter((p) => p.name.toLowerCase().includes(q.toLowerCase())).map((p) => ({ ...p, title: p.name, href: `#/project/${p.id}` })) : [];
      items = q ? [...projects, ...(await api(`search?q=${encodeURIComponent(q)}`).catch(() => []))] : state.chats.slice(0, 10);
      sel = 0; draw();
    }, 120);
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { sel = Math.min(items.length - 1, sel + 1); draw(); e.preventDefault(); }
    if (e.key === "ArrowUp") { sel = Math.max(0, sel - 1); draw(); e.preventDefault(); }
    if (e.key === "Enter" && items[sel]) go(items[sel]);
  });
  items = state.chats.slice(0, 10);
  modal([
    h("div.search-top", icon("search"), input, h("button.icon-btn", { type: "button", "aria-label": "Close", on: { click: closeModal } }, icon("x"))),
    results,
    h("div.search-foot", h("span", h("kbd", "↑"), h("kbd", "↓"), "Navigate"), h("span", h("kbd", "⏎"), "Open"), h("span", h("kbd", "Esc"), "Close")),
  ], { cls: "search-modal" });
  draw();
}

// --- settings ----------------------------------------------------------------------

export function openSettings() {
  const root = document.documentElement;
  let sec = "general";
  const pane = h("div.settings-pane");
  const nav = h("nav.settings-nav", { "aria-label": "Settings" });
  const seg = (opts, cur, set) => { const el = h("div.seg", { role: "radiogroup" }, opts.map(([v, l]) => h(`button${v === cur ? ".on" : ""}`, { type: "button", role: "radio", "aria-checked": String(v === cur), on: { click: (e) => { for (const b of el.children) b.classList.toggle("on", b === e.currentTarget); set(v); } } }, l))); return el; };
  const select = (opts, cur, set) => h("select.input", { on: { change: (e) => set(e.target.value) } }, opts.map(([v, l]) => h("option", { value: v, selected: v === cur }, l)));
  const row = (label, sub, ctl) => h("div.s-row", h("div.txt", h("div", label), sub ? h("span", sub) : null), h("div.ctl", ctl));
  const SECTIONS = {
    general: ["General", "gear", () => [
      h("section.s-group", h("h3", "Appearance"),
        row("Color mode", null, seg([["light", "Light"], ["system", "Auto"], ["dark", "Dark"]], root.dataset.theme || "system", setTheme)),
        row("Reply font", "The typeface used for replies in chats.", select([["serif", "Serif"], ["sans", "Sans"]], root.dataset.reply || "serif", (v) => { try { localStorage.setItem("reply", v); } catch {} if (v === "sans") root.dataset.reply = "sans"; else delete root.dataset.reply; }))),
      h("section.s-group", h("h3", "Composer"),
        row("Enter key", "On a computer. On a phone, Enter always adds a new line.", select([["send", "Sends the message"], ["newline", "Adds a new line"]], prefs.get("enterSends", true) ? "send" : "newline", (v) => prefs.set("enterSends", v === "send")))),
    ]],
    account: ["Account", "user", () => {
      const name = h("input.input", { value: state.me.name || "", "aria-label": "Name", on: { change: async () => { const v = name.value.trim(); if (!v) return; await api("settings", { body: { name: v } }); await refreshStyles(); toast("Saved."); } } });
      return [h("section.s-group", h("h3", "Profile"),
        row("What should Plint call you?", null, name)),
      h("section.s-group", h("h3", "This device"),
        row("Log out", "You'll need a new code from your phone to sign in again.", h("button.btn.secondary", { type: "button", on: { click: async () => { await api("logout", { body: {} }).catch(() => {}); location.reload(); } } }, "Log out")))];
    }],
    models: ["Models", "bolt", () => [h("section.s-group", h("h3", "Models"), h("p", "Served by silk-router on Plint. Pick one under the composer."),
      ...Object.values(state.models).map((m) => row(m.name, m.sub, m.efforts ? h("span.muted", "Effort: low, medium, high") : null)))]],
  };
  const draw = () => {
    nav.replaceChildren(h("div.gt", "Preferences"), ...Object.entries(SECTIONS).map(([k, [label, ico]]) => h(`button${k === sec ? ".on" : ""}`, { type: "button", "aria-current": k === sec ? "page" : null, on: { click: () => { sec = k; draw(); } } }, icon(ico), label)));
    pane.replaceChildren(...SECTIONS[sec][2]());
  };
  modal([h("div.settings-top", "Settings", h("button.icon-btn", { type: "button", "aria-label": "Close", on: { click: closeModal } }, icon("x"))), nav, pane], { cls: "settings" });
  draw();
}
