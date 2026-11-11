// Full pages (Projects, a project, Artifacts, Scheduled, Customize, Recents) and
// the Search and Settings dialogs.
import { h, icon, menu, modal, closeModal, formModal, confirmModal, toast, ago, when, bytes, narrow } from "./ui.js";
import { api, upload } from "./api.js";
import { state, on, refreshChats, refreshProjects, refreshStyles, projectName, prefs } from "./state.js";
import { composer } from "./chat.js";
import { artifactTile, openArtifact, closePanel } from "./artifacts.js";
import { chatMenu } from "./sidebar.js";
import { topBar } from "./main.js";

const page = (view, ...kids) => { const p = h("div.page", ...kids); view.replaceChildren(h("div.scroll", p)); return p; };
const chatRow = (c, { showProject = true } = {}) => h("a.list-item", { href: `#/chat/${c.id}` },
  h("div.grow", h("div.t", c.title), h("div.s", [`Last message ${ago(c.updated)}`, showProject && c.projectId ? projectName(c.projectId) : null, c.scheduleId ? "Scheduled" : null].filter(Boolean).join(" · "))),
  c.running ? h("i.dot", { style: { width: "7px", height: "7px", borderRadius: "50%", background: "var(--brand)" } }) : null,
  h("button.icon-btn.sm", { "aria-label": `More options for ${c.title}`, on: { click: (e) => { e.preventDefault(); e.stopPropagation(); chatMenu(e.currentTarget, c); } } }, icon("more")));

// --- projects -------------------------------------------------------------------

export function newProject() {
  return formModal({
    title: "Create a project",
    fields: [
      { name: "name", label: "What are you working on?", placeholder: "Name your project", autofocus: true },
      { name: "description", label: "What are you trying to achieve?", type: "textarea", rows: 3, placeholder: "Describe your project, goals, subject, etc." },
    ],
    submit: "Create project",
    onSubmit: async (d) => { const p = await api("projects", { body: d }); await refreshProjects(); location.hash = `#/project/${p.id}`; },
  });
}

export function renderProjects(view, { create = false } = {}) {
  closePanel();
  topBar({ title: "" });
  let q = "";
  const grid = h("div.card-grid");
  const draw = () => {
    const list = state.projects.filter((p) => !q || `${p.name} ${p.description}`.toLowerCase().includes(q));
    grid.replaceChildren(...list.map((p) => h("a.card", { href: `#/project/${p.id}` }, h("div.t", p.name), h("div.d", p.description || "No description"), h("div.s", `${p.chats} chat${p.chats === 1 ? "" : "s"} · Updated ${ago(p.updated)}`))));
    if (!list.length) grid.replaceChildren(h("div.empty-note", { style: { gridColumn: "1/-1" } }, icon("folder"), h("div", q ? "No projects match." : "No projects yet. A project keeps chats, files and instructions together.")));
  };
  page(view,
    h("div.page-head", h("h1", "Projects"), h("button.btn.primary", { on: { click: newProject } }, icon("plus"), "New project")),
    h("label.search-field", icon("search"), h("input", { placeholder: "Search projects…", on: { input: (e) => { q = e.target.value.toLowerCase(); draw(); } } })),
    grid);
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
    placeholder: "Start a new chat in this project…",
    onSend: async (text, files, settings) => {
      const chat = await api("chats", { body: { text, files, ...settings, projectId: id } });
      refreshChats();
      location.hash = `#/chat/${chat.id}`;
    },
  });
  const title = h("h1");
  const desc = h("p.page-sub");
  async function load() {
    try { p = await api(`projects/${id}`); } catch { toast("That project is gone."); location.hash = "#/projects"; return; }
    title.textContent = p.name;
    desc.textContent = p.description || "";
    instr.textContent = p.instructions || "Add instructions to tailor replies in this project.";
    instr.classList.toggle("faint", !p.instructions);
    const files = await api(`projects/${id}/files`);
    filesBox.replaceChildren(...(files.length ? files.map((f) => h("div.file-row", icon("file"),
      h("a.grow", { href: `/api/projects/${id}/files/${encodeURIComponent(f.name)}`, target: "_blank" }, f.name), h("span.faint.small", bytes(f.size)),
      h("button.icon-btn.sm", { "aria-label": `Remove ${f.name}`, on: { click: async () => { await api(`projects/${id}/files/${encodeURIComponent(f.name)}`, { method: "DELETE" }); load(); } } }, icon("x"))))
      : [h("div.faint.small", "Files here are available to every chat in the project.")]));
    drawChats();
    topBar({ title: p.name });
  }
  function drawChats() {
    const list = state.chats.filter((x) => x.projectId === id);
    chatsBox.replaceChildren(...(list.length ? list.map((x) => chatRow(x, { showProject: false })) : [h("div.empty-note", "No chats in this project yet.")]));
  }
  fileInput.onchange = async () => {
    for (const f of fileInput.files) { try { await upload(`projects/${id}/files`, f); } catch (e) { toast(e.message); } }
    fileInput.value = "";
    load();
  };
  const editInstructions = () => formModal({
    title: "Set project instructions", submit: "Save instructions",
    fields: [{ name: "instructions", label: "How should replies in this project behave?", type: "textarea", rows: 10, value: p.instructions || "", placeholder: "E.g. Work in /opt/plint-table. Answer in Swedish. Always run the tests." }],
    onSubmit: async (d) => { await api(`projects/${id}`, { body: d }); load(); },
  });
  const projMenu = (e) => menu(e.currentTarget, [
    { label: "Edit details", icon: "edit", onClick: () => formModal({ title: "Edit details", fields: [{ name: "name", label: "Name", value: p.name }, { name: "description", label: "Description", type: "textarea", rows: 3, value: p.description }],
      onSubmit: async (d) => { await api(`projects/${id}`, { body: d }); refreshProjects(); load(); } }) },
    { label: "Working folder", icon: "folder", sub: p.cwd, onClick: () => formModal({ title: "Working folder", fields: [{ name: "cwd", label: "Claude Code starts in", value: p.cwd, hint: "A folder on Plint, e.g. /opt/plint-table" }],
      onSubmit: async (d) => { const r = await api(`projects/${id}`, { body: d }); if (r.cwd !== d.cwd) throw new Error("That folder doesn't exist on Plint."); load(); } }) },
    "sep",
    { label: "Delete project", icon: "trash", danger: true, onClick: async () => {
      if (!(await confirmModal("Delete project?", "Its chats stay, but leave the project. Project files are deleted."))) return;
      await api(`projects/${id}`, { method: "DELETE" }); refreshProjects(); refreshChats(); location.hash = "#/projects";
    } },
  ], { align: "end" });
  page(view,
    h("a.muted.small", { href: "#/projects", style: { textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "4px" } }, h("span", { style: { display: "inline-flex", transform: "rotate(180deg)" } }, icon("chev")), "All projects"),
    h("div.page-head", title, h("button.icon-btn", { "aria-label": "Project options", on: { click: projMenu } }, icon("more"))),
    desc,
    h("div.split",
      h("div", c.el, h("div", { style: { height: "22px" } }), chatsBox),
      h("div",
        h("div.side-card", h("h3", "Instructions", h("button.btn.ghost", { style: { height: "28px", padding: "0 8px" }, on: { click: editInstructions } }, "Edit")), instr),
        h("div.side-card", h("h3", "Files", h("button.btn.ghost", { style: { height: "28px", padding: "0 8px" }, on: { click: () => fileInput.click() } }, icon("plus"), "Add")), filesBox, fileInput))));
  load();
  return on("chats", drawChats);
}

// --- artifacts ------------------------------------------------------------------

export function renderArtifacts(view) {
  topBar({ title: "" });
  const grid = h("div.art-grid");
  let q = "";
  let all = [];
  const draw = () => {
    const list = all.filter((a) => !q || `${a.title} ${a.chatTitle}`.toLowerCase().includes(q));
    grid.replaceChildren(...list.map((a) => artifactTile(a, { onMenu: (anchor) => menu(anchor, [
      { label: "Open", icon: "eye", onClick: () => openArtifact(a.id) },
      { label: "Go to chat", icon: "external", onClick: () => { location.hash = `#/chat/${a.chatId}`; } },
      { label: "Download", icon: "download", onClick: () => { location.href = `/api/artifacts/${a.id}/raw?download=1`; } },
      "sep",
      { label: "Delete", icon: "trash", danger: true, onClick: async () => { if (await confirmModal("Delete artifact?", `"${a.title}" and its versions will be removed.`)) { await api(`artifacts/${a.id}`, { method: "DELETE" }); load(); } } },
    ], { align: "end" }) })));
    if (!list.length) grid.replaceChildren(h("div.empty-note", { style: { gridColumn: "1/-1" } }, icon("artifacts"), h("div", q ? "No artifacts match." : "Ask for a page, a diagram, a chart or a document and it shows up here.")));
  };
  const load = async () => { all = await api("artifacts").catch(() => []); draw(); };
  page(view,
    h("div.page-head", h("h1", "Artifacts"), h("a.btn.primary", { href: "#/", on: { click: () => setTimeout(() => document.querySelector(".composer textarea")?.focus(), 50) } }, icon("plus"), "New artifact")),
    h("label.search-field", icon("search"), h("input", { placeholder: "Search artifacts…", on: { input: (e) => { q = e.target.value.toLowerCase(); draw(); } } })),
    grid);
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

export function renderScheduled(view) {
  closePanel();
  topBar({ title: "" });
  const list = h("div");
  const load = async () => {
    const tasks = await api("schedules").catch(() => []);
    list.replaceChildren(...(tasks.length ? tasks.map((t) => {
      const sw = h(`button.sw${t.enabled ? ".on" : ""}`, { "aria-label": t.enabled ? "Pause task" : "Resume task", title: t.enabled ? "On" : "Paused",
        on: { click: async () => { await api(`schedules/${t.id}`, { body: { enabled: !t.enabled } }); load(); } } });
      return h("div.task",
        h("div.top-row", h("div.t.ellipsis", t.name), sw,
          h("button.icon-btn.sm", { "aria-label": "Task options", on: { click: (e) => menu(e.currentTarget, [
            { label: "Run now", icon: "play", onClick: async () => { try { await api(`schedules/${t.id}/run`, { body: {} }); toast("Started"); refreshChats(); load(); } catch (err) { toast(err.message); } } },
            { label: "Edit", icon: "edit", onClick: () => taskForm(t).then((r) => r && load()) },
            "sep",
            { label: "Delete", icon: "trash", danger: true, onClick: async () => { if (await confirmModal("Delete task?", `"${t.name}" won't run again. Chats it made stay.`)) { await api(`schedules/${t.id}`, { method: "DELETE" }); load(); } } },
          ], { align: "end" }) } }, icon("more"))),
        h("div.p", t.prompt),
        h("div.s",
          h("span", icon("clock"), " ", t.when),
          t.enabled ? h("span", `Next: ${when(t.nextRun)}`) : h("span", "Paused"),
          t.lastChatId ? h("a", { href: `#/chat/${t.lastChatId}` }, `Last run ${ago(t.lastRun)}`) : h("span", "Never run"),
          t.mode === "auto" ? h("span.pill.brand", icon("bolt"), "Auto") : null,
          t.projectId ? h("span.pill", icon("folder"), projectName(t.projectId) || "Project") : null));
    }) : [h("div.empty-note", icon("clock"), h("div", "Nothing scheduled. A task runs a prompt on its own, as a new chat, and pings your phone."))]));
  };
  page(view,
    h("div.page-head", h("h1", "Scheduled"), h("button.btn.primary", { on: { click: () => taskForm().then((r) => r && load()) } }, icon("plus"), "New task")),
    h("p.page-sub", "Prompts that run on their own. Each run is a new chat; anything that needs your OK goes to your phone."),
    list);
  load();
  const t = setInterval(load, 30000);
  return () => clearInterval(t);
}

// --- customize ------------------------------------------------------------------

export function renderCustomize(view, tab = "instructions") {
  closePanel();
  topBar({ title: "" });
  const body = h("div");
  const tabs = h("div.tabs");
  const TABS = [["instructions", "Instructions"], ["styles", "Styles"], ["skills", "Skills"], ["connectors", "Connectors"]];
  async function draw() {
    tabs.replaceChildren(...TABS.map(([k, l]) => h(`button${tab === k ? ".on" : ""}`, { type: "button", on: { click: () => { tab = k; history.replaceState(null, "", `#/customize/${k}`); draw(); } } }, l)));
    const s = await api("settings");
    if (tab === "instructions") {
      const name = h("input", { value: s.name });
      const ta = h("textarea", { rows: 10, placeholder: "E.g. I'm Filip. Keep answers short, I read on my phone. I prefer Swedish for chat, English for code." }, s.instructions || "");
      body.replaceChildren(
        h("label.field", h("span", "What should Plint call you?"), name),
        h("label.field", h("span", "What personal preferences should replies follow?"), ta, h("span.hint", "Added to every chat, on top of CLAUDE.md.")),
        h("button.btn.primary", { on: { click: async () => { await api("settings", { body: { name: name.value, instructions: ta.value } }); await refreshStyles(); toast("Saved"); } } }, "Save"));
    } else if (tab === "styles") {
      const custom = s.styles;
      const edit = (st) => formModal({
        title: st ? "Edit style" : "Create a style",
        fields: [{ name: "name", label: "Name", value: st?.name || "", autofocus: true }, { name: "prompt", label: "How should replies sound?", type: "textarea", rows: 6, value: st?.prompt || "", placeholder: "E.g. Write like a laid-back sysadmin. Short lines. No emoji." }],
        onSubmit: async (d) => {
          const styles = st ? custom.map((x) => (x.id === st.id ? { ...x, ...d } : x)) : [...custom, d];
          await api("settings", { body: { styles } }); await refreshStyles(); draw();
        },
      });
      body.replaceChildren(
        h("p.muted", "Pick a style from the + menu in the composer. Custom styles are yours to write."),
        ...s.allStyles.map((st) => h("div.style-item",
          h("div.grow", h("div.t", st.name, s.defaultStyle === st.id ? h("span.pill", { style: { marginLeft: "8px" } }, "Default") : null), h("div.p", st.prompt || "Plain replies, no extra style.")),
          h("button.icon-btn.sm", { "aria-label": "Style options", on: { click: (e) => menu(e.currentTarget, [
            { label: "Make default", icon: "check", onClick: async () => { await api("settings", { body: { defaultStyle: st.id } }); draw(); } },
            st.builtin ? null : { label: "Edit", icon: "edit", onClick: () => edit(st) },
            st.builtin ? null : { label: "Delete", icon: "trash", danger: true, onClick: async () => { await api("settings", { body: { styles: custom.filter((x) => x.id !== st.id) } }); await refreshStyles(); draw(); } },
          ], { align: "end" }) } }, icon("more")))),
        h("div", { style: { marginTop: "16px" } }, h("button.btn.primary", { on: { click: () => edit(null) } }, icon("plus"), "Create style")));
    } else if (tab === "skills") {
      const skills = await api("skills").catch(() => []);
      body.replaceChildren(h("p.muted", "Skills Claude Code on Plint can load. Add them under ~/.claude/skills or /opt/plint/.claude/skills."),
        h("div.list", skills.map((sk) => h("div.list-item", icon("sparkle"), h("div.grow", h("div.t", sk.name), h("div.s", sk.description))), h("span.pill", sk.where))));
    } else {
      body.replaceChildren(h("p.muted", "Plint Chat runs Claude Code on Plint, so it uses the tools and MCP servers configured for Claude Code there (claude mcp list). claude.ai's own connectors (Gmail, Drive…) need Anthropic's servers and aren't available with Silk models."));
    }
  }
  page(view, h("div.page-head", h("h1", "Customize")), tabs, body);
  draw();
}

// --- recents (view all) -------------------------------------------------------------

export function renderRecents(view) {
  closePanel();
  topBar({ title: "" });
  let q = "";
  const list = h("div.list");
  const draw = () => {
    const items = state.chats.filter((c) => !q || c.title.toLowerCase().includes(q));
    list.replaceChildren(...(items.length ? items.map((c) => chatRow(c)) : [h("div.empty-note", "No chats.")]));
  };
  page(view,
    h("div.page-head", h("h1", "Your chats"), h("a.btn.primary", { href: "#/" }, icon("plus"), "New chat")),
    h("label.search-field", icon("search"), h("input", { placeholder: "Search your chats…", on: { input: (e) => { q = e.target.value.toLowerCase(); draw(); } } })),
    h("p.faint.small", `${state.chats.length} chats`), list);
  draw();
  return on("chats", draw);
}

// --- search ------------------------------------------------------------------------

export function openSearch() {
  let sel = 0;
  let items = [];
  const input = h("input", { placeholder: "Search chats and messages…", autofocus: true, "aria-label": "Search" });
  const results = h("div.search-results");
  const go = (c) => { closeModal(); location.hash = `#/chat/${c.id}`; };
  const draw = () => {
    results.replaceChildren(...(items.length ? items.map((c, i) => h(`a.sr${i === sel ? ".sel" : ""}`, { href: `#/chat/${c.id}`, on: { click: (e) => { e.preventDefault(); go(c); } } },
      h("span.d", ago(c.updated)), h("div.t", c.title), c.snippet ? h("div.s", c.snippet) : null)) : [h("div.empty-note", input.value ? "Nothing found." : "No chats yet.")]));
  };
  let timer;
  input.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(async () => {
      const q = input.value.trim();
      items = q ? await api(`search?q=${encodeURIComponent(q)}`).catch(() => []) : state.chats.slice(0, 10);
      sel = 0; draw();
    }, 120);
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { sel = Math.min(items.length - 1, sel + 1); draw(); e.preventDefault(); }
    if (e.key === "ArrowUp") { sel = Math.max(0, sel - 1); draw(); e.preventDefault(); }
    if (e.key === "Enter" && items[sel]) go(items[sel]);
  });
  items = state.chats.slice(0, 10);
  modal([h("label.search-field", icon("search"), input, h("button.icon-btn.sm", { "aria-label": "Close", on: { click: closeModal } }, icon("x"))), results], { cls: "search-modal" });
  draw();
}

// --- settings ----------------------------------------------------------------------

export function openSettings() {
  const theme = document.documentElement.dataset.theme || "system";
  const reply = document.documentElement.dataset.reply || "serif";
  formModal({
    title: "Settings",
    submit: "Save",
    fields: [
      { name: "name", label: "Your name", value: state.me.name },
      { name: "theme", label: "Appearance", type: "select", value: theme, options: [["system", "Match device"], ["light", "Light"], ["dark", "Dark"]] },
      { name: "reply", label: "Reply font", type: "select", value: reply, options: [["serif", "Serif (like Claude)"], ["sans", "Sans"]] },
      { name: "enter", label: "Enter key (computer)", type: "select", value: prefs.get("enterSends", true) ? "send" : "newline", options: [["send", "Sends the message"], ["newline", "Adds a new line"]] },
    ],
    onSubmit: async (d) => {
      try {
        d.theme === "system" ? localStorage.removeItem("theme") : localStorage.setItem("theme", d.theme);
        localStorage.setItem("reply", d.reply);
      } catch {}
      if (d.theme === "system") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = d.theme;
      if (d.reply === "sans") document.documentElement.dataset.reply = "sans"; else delete document.documentElement.dataset.reply;
      prefs.set("enterSends", d.enter === "send");
      if (d.name.trim() && d.name !== state.me.name) { await api("settings", { body: { name: d.name } }); await refreshStyles(); }
    },
  });
}
