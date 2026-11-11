// Code: sessions that run Claude Code in a folder on Plint or on another computer
// (through plint-runner), with Changes (git diff) and Files beside the chat.
import { h, icon, menu, modal, closeModal, formModal, confirmModal, toast, ago, copyText, esc, bytes, narrow } from "./ui.js";
import { api } from "./api.js";
import { highlight } from "./md.js";
import { state, prefs, refreshChats, on } from "./state.js";
import { composer } from "./chat.js";
import { closePanel } from "./artifacts.js";
import { topBar } from "./main.js";
import { setMachineNames } from "./sidebar.js";

export let machinesCache = [];
export async function refreshMachines() {
  try { machinesCache = await api("machines"); setMachineNames(machinesCache); } catch {}
  return machinesCache;
}
export const machineName = (id) => machinesCache.find((m) => m.id === id)?.name || (id === "plint" ? "Plint" : "Computer");
export const baseName = (p) => String(p || "").replace(/[\\/]+$/, "").split(/[\\/]/).pop() || p;

// --- folder picker ------------------------------------------------------------

export function pickFolder(machineId, start) {
  return new Promise((resolve) => {
    let cur = null;
    const list = h("div.fp-list");
    const pathInput = h("input.fp-path", { "aria-label": "Folder path", spellcheck: false });
    const gitPill = h("span.pill", { hidden: true }, icon("code"), "git");
    const showHidden = h("button.btn.ghost", { type: "button", style: { height: "30px", padding: "0 8px", fontSize: "12.5px" } }, "Show hidden");
    let hidden = false;
    async function go(p) {
      list.replaceChildren(h("div.faint.small", { style: { padding: "12px" } }, "Loading…"));
      try { cur = await api(`machines/${machineId}/ls?path=${encodeURIComponent(p || "")}`); }
      catch (e) { list.replaceChildren(h("div.note", e.message)); return; }
      pathInput.value = cur.path;
      gitPill.hidden = !cur.git;
      list.replaceChildren(
        cur.parent ? h("button.fp-row", { type: "button", on: { click: () => go(cur.parent) } }, icon("chev", "flip"), h("span", "..")) : null,
        ...cur.entries.filter((e) => hidden || !e.hidden).map((e) => e.dir
          ? h("button.fp-row", { type: "button", on: { click: () => go(`${cur.path}${cur.path.endsWith("/") || cur.path.endsWith("\\") ? "" : (cur.path.includes("\\") ? "\\" : "/")}${e.name}`) } }, icon("folder"), h("span.grow.ellipsis", e.name), icon("chev"))
          : h("div.fp-row.file", icon("file"), h("span.grow.ellipsis", e.name), h("span.faint.small", bytes(e.size)))));
    }
    showHidden.onclick = () => { hidden = !hidden; showHidden.textContent = hidden ? "Hide hidden" : "Show hidden"; go(cur?.path); };
    pathInput.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); go(pathInput.value); } });
    modal([
      h("h2", `Choose a folder on ${machineName(machineId)}`),
      h("div.fp-bar", pathInput, gitPill),
      list,
      h("div.actions", showHidden, h("span.grow"),
        h("button.btn.ghost", { type: "button", on: { click: () => { closeModal(); resolve(null); } } }, "Cancel"),
        h("button.btn.primary", { type: "button", on: { click: () => { closeModal(); resolve(cur?.path || null); } } }, "Use this folder")),
    ], { cls: "fp-modal" });
    go(start);
  });
}

// --- code home ------------------------------------------------------------------

export function renderCodeHome(view) {
  state.current = null;
  closePanel();
  topBar({ title: "" });
  let machine = prefs.get("codeMachine", "plint");
  let cwd = prefs.get(`codeCwd.${machine}`, machine === "plint" ? "/opt/plint" : "");
  const machBtn = h("button.c-btn.code-chip", { type: "button" });
  const folderBtn = h("button.c-btn.code-chip", { type: "button" });
  const branch = h("span.pill", { hidden: true });
  const c = composer({
    code: true,
    placeholder: "Describe a task or ask about the code…",
    onSend: async (text, files, settings) => {
      if (!cwd) { toast("Pick a folder first."); throw new Error("Pick a folder first."); }
      const chat = await api("chats", { body: { text, files, ...settings, projectId: null, kind: "code", machine, cwd } });
      rememberFolder(machine, cwd);
      refreshChats();
      location.hash = `#/chat/${chat.id}`;
    },
  });
  async function paint() {
    const m = machinesCache.find((x) => x.id === machine);
    machBtn.replaceChildren(icon(machine === "plint" ? "box" : "terminal"), h("span.t", machineName(machine)), m && !m.connected ? h("span.offline", "offline") : null, icon("down", "chev"));
    folderBtn.replaceChildren(icon("folder"), h("span.t", cwd ? baseName(cwd) : "Choose folder"), icon("down", "chev"));
    folderBtn.title = cwd || "";
    branch.hidden = true;
    if (cwd && m?.connected) {
      api(`machines/${machine}/git?path=${encodeURIComponent(cwd)}`).then((g) => {
        if (!g.git) return;
        branch.hidden = false;
        branch.replaceChildren(icon("code"), g.branch || "detached");
      }).catch(() => {});
    }
    drawRecents();
  }
  machBtn.onclick = (e) => menu(e.currentTarget, [
    { heading: "Run on" },
    ...machinesCache.map((m) => ({ label: m.name, sub: m.connected ? (m.builtin ? "This server" : m.info?.os || "Connected") : `Offline${m.lastSeen ? ` · seen ${ago(m.lastSeen)}` : ""}`, icon: m.builtin ? "box" : "terminal", checked: m.id === machine,
      onClick: () => { machine = m.id; prefs.set("codeMachine", machine); cwd = prefs.get(`codeCwd.${machine}`, m.builtin ? "/opt/plint" : m.info?.roots?.[0] || ""); paint(); } })),
    "sep",
    { label: "Add a computer", icon: "plus", onClick: () => { location.hash = "#/computers"; } },
  ], { align: "start" });
  folderBtn.onclick = async () => {
    const m = machinesCache.find((x) => x.id === machine);
    if (m && !m.connected) return toast(`${m.name} is offline. Start plint-runner on it.`);
    const p = await pickFolder(machine, cwd);
    if (p) { cwd = p; prefs.set(`codeCwd.${machine}`, cwd); paint(); }
  };
  const recents = h("div.code-recents");
  function drawRecents() {
    const seen = new Map();
    for (const s of state.chats.filter((x) => x.kind === "code")) {
      const k = `${s.machine}|${s.cwd}`;
      if (!seen.has(k)) seen.set(k, { machine: s.machine, cwd: s.cwd, sessions: [] });
      seen.get(k).sessions.push(s);
    }
    for (const f of prefs.get("codeFolders", [])) if (!seen.has(`${f.machine}|${f.cwd}`)) seen.set(`${f.machine}|${f.cwd}`, { ...f, sessions: [] });
    const items = [...seen.values()].slice(0, 8);
    recents.replaceChildren(items.length ? h("div.head", "Recent folders") : null, ...items.map((f) => h("button.code-folder", { type: "button", on: { click: () => { machine = f.machine; cwd = f.cwd; prefs.set("codeMachine", machine); prefs.set(`codeCwd.${machine}`, cwd); paint(); c.focus(); } } },
      icon("folder"), h("span.grow", h("div.t", baseName(f.cwd)), h("div.s", `${machineName(f.machine)} · ${f.cwd}`)), f.sessions.length ? h("span.faint.small", `${f.sessions.length} session${f.sessions.length === 1 ? "" : "s"}`) : null)));
  }
  const box = h("div.home.code-home",
    h("div.greet", h("span.spark.code-mark", icon("code")), h("h1", "What should we build?")),
    h("div.composer-slot", h("div.code-target", machBtn, h("span.faint", "/"), folderBtn, branch), c.el),
    recents);
  view.replaceChildren(h("div.scroll.home-scroll", box));
  refreshMachines().then(paint);
  paint();
  c.focus();
  return on("chats", drawRecents);
}

export function rememberFolder(machine, cwd) {
  const list = prefs.get("codeFolders", []).filter((f) => !(f.machine === machine && f.cwd === cwd));
  list.unshift({ machine, cwd });
  prefs.set("codeFolders", list.slice(0, 12));
}

// --- computers --------------------------------------------------------------------

function installSteps(m) {
  const srv = m.server;
  const unix = `mkdir -p ~/plint-runner && cd ~/plint-runner && \\
curl -fsSO ${srv}/runner/plint-runner.mjs -fsSO ${srv}/runner/ops.mjs -fsSO ${srv}/runner/package.json && \\
npm install --omit=dev && \\
node plint-runner.mjs --server ${srv} --token ${m.token} --name "${m.name}"`;
  const win = `mkdir $HOME\\plint-runner -Force; cd $HOME\\plint-runner
foreach ($f in "plint-runner.mjs","ops.mjs","package.json") { iwr ${srv}/runner/$f -OutFile $f }
npm install --omit=dev
node plint-runner.mjs --server ${srv} --token ${m.token} --name "${m.name}"`;
  const block = (label, text) => h("div.code", h("div.code-head", h("span", label), h("button", { type: "button", on: { click: () => copyText(text) } }, icon("copy"), "Copy")), h("pre", h("code", text)));
  return [
    h("p.muted", "Run this once on the computer. It needs Node 18 or newer and has to reach Plint (home network or Tailscale). Keep the window open; it reconnects by itself."),
    block("Mac / Linux (Terminal)", unix),
    block("Windows (PowerShell)", win),
    h("p.faint.small", "Add --root <folder> (more than once if you like) to share only those folders. By default it shares your home folder. The token is shown only now."),
  ];
}

export function renderComputers(view) {
  closePanel();
  topBar({ title: "" });
  const list = h("div");
  const draw = async () => {
    const ms = await refreshMachines();
    list.replaceChildren(...ms.map((m) => h("div.task",
      h("div.top-row",
        h("span.conn", { class: m.connected ? "on" : "" }),
        h("div.t.ellipsis", m.name),
        m.builtin ? h("span.pill", "This server") : h("button.icon-btn.sm", { "aria-label": "Computer options", on: { click: (e) => menu(e.currentTarget, [
          { label: "Rename", icon: "edit", onClick: () => formModal({ title: "Rename computer", fields: [{ name: "name", label: "Name", value: m.name, autofocus: true }], onSubmit: async (d) => { await api(`machines/${m.id}`, { body: d }); draw(); } }) },
          "sep",
          { label: "Remove", icon: "trash", danger: true, onClick: async () => { if (await confirmModal("Remove computer?", `Its token stops working. Sessions on ${m.name} stay but can't continue.`, "Remove")) { await api(`machines/${m.id}`, { method: "DELETE" }); draw(); } } },
        ], { align: "end" }) } }, icon("more"))),
      h("div.s",
        h("span", m.connected ? "Connected" : m.lastSeen ? `Offline · seen ${ago(m.lastSeen)}` : "Never connected"),
        m.info?.os ? h("span", m.info.os) : null,
        m.info?.roots?.length ? h("span", `Shares ${m.info.roots.join(", ")}`) : null))));
  };
  const add = () => formModal({
    title: "Add a computer", submit: "Create token",
    fields: [{ name: "name", label: "Name", placeholder: "Work laptop", autofocus: true }],
    onSubmit: async (d) => {
      const m = await api("machines", { body: { name: d.name || "Computer" } });
      setTimeout(() => modal([h("h2", `Connect ${m.name}`), ...installSteps(m), h("div.actions", h("button.btn.primary", { on: { click: () => { closeModal(); draw(); } } }, "Done"))], { cls: "wide-modal" }), 50);
    },
  });
  const p = h("div.page",
    h("div.page-head", h("h1", "Computers"), h("button.btn.primary", { on: { click: add } }, icon("plus"), "Add a computer")),
    h("p.page-sub", "Code sessions run Claude Code in a folder on one of these. Other computers connect with plint-runner; nothing is opened to the internet."),
    list);
  view.replaceChildren(h("div.scroll", p));
  draw();
  const t = setInterval(draw, 10000);
  return () => clearInterval(t);
}

// --- Changes and Files panels -------------------------------------------------------

let panelState = null; // { kind: "changes"|"files", chat, path }
export const codePanelOpen = (chatId) => panelState?.chat.id === chatId && !document.getElementById("panel").hidden;
export function refreshCodePanel(chat) { if (panelState && panelState.chat.id === chat.id && panelState.kind === "changes" && !document.getElementById("panel").hidden) openChanges(chat, { quiet: true }); }

function panelShell(title, tools, body) {
  const p = document.getElementById("panel");
  p.replaceChildren(
    h("div.panel-head", h("span.t", title), ...tools, h("button.icon-btn", { title: "Close", "aria-label": "Close", on: { click: () => { panelState = null; closePanel(); } } }, icon("x"))),
    h("div.panel-body", body));
  p.hidden = false;
}

function diffBlock(text) {
  const pre = h("pre.diff");
  pre.innerHTML = text.split("\n").map((l) => {
    const cls = l.startsWith("+") && !l.startsWith("+++") ? "add" : l.startsWith("-") && !l.startsWith("---") ? "del" : l.startsWith("@@") ? "hunk" : l.startsWith("diff ") || l.startsWith("index ") || l.startsWith("+++") || l.startsWith("---") || l === "new file" ? "meta" : "";
    return `<span class="${cls}">${esc(l) || " "}</span>`;
  }).join("");
  return pre;
}

export async function openChanges(chat, { quiet = false } = {}) {
  panelState = { kind: "changes", chat };
  const body = h("div.code-view.changes", h("div.faint.small", "Reading changes…"));
  const seg = h("div.seg", h("button.on", { type: "button" }, "Changes"), h("button", { type: "button", on: { click: () => openFiles(chat) } }, "Files"));
  if (!quiet || document.getElementById("panel").hidden) panelShell(`${baseName(chat.cwd)}`, [seg, h("button.icon-btn", { title: "Refresh", "aria-label": "Refresh", on: { click: () => openChanges(chat) } }, icon("retry"))], body);
  let d;
  try { d = await api(`machines/${chat.machine}/diff?path=${encodeURIComponent(chat.cwd)}`); }
  catch (e) { body.replaceChildren(h("div.note", e.message)); return; }
  if (panelState?.kind !== "changes" || panelState.chat.id !== chat.id) return;
  const target = document.querySelector("#panel .changes") || body;
  if (!d.git) return target.replaceChildren(h("div.empty-note", icon("folder"), h("div", "This folder isn't a git repository, so there's no diff to show. Use Files to look around.")));
  if (!d.files.length) return target.replaceChildren(h("div.empty-note", icon("check"), h("div", "No changes yet. The working tree matches the last commit.")));
  const parts = {};
  for (const chunk of d.patch.split(/^(?=diff --git )/m)) {
    const m = chunk.match(/^diff --git a\/(.+?) b\/(.+)$/m);
    if (m) parts[m[2]] = chunk;
  }
  const add = d.files.reduce((n, f) => n + (f.additions || 0), 0), del = d.files.reduce((n, f) => n + (f.deletions || 0), 0);
  target.replaceChildren(
    h("div.diff-sum", `${d.files.length} file${d.files.length === 1 ? "" : "s"} changed`, h("span.add", ` +${add}`), h("span.del", ` −${del}`)),
    ...d.files.map((f) => {
      const row = h("div.diff-file",
        h("button.diff-head", { type: "button", on: { click: () => row.classList.toggle("open") } },
          icon("chev", "chev"), h("span.st", { class: f.status }, f.status[0].toUpperCase()), h("span.grow.ellipsis", f.path),
          h("span.add", f.additions ? `+${f.additions}` : ""), h("span.del", f.deletions ? `−${f.deletions}` : "")),
        parts[f.path] ? diffBlock(parts[f.path]) : h("div.faint.small", { style: { padding: "8px 12px" } }, f.status === "deleted" ? "Deleted." : "No text diff (binary or too large)."));
      if (d.files.length <= 3) row.classList.add("open");
      return row;
    }));
}

export async function openFiles(chat, p = chat.cwd) {
  panelState = { kind: "files", chat, path: p };
  const seg = h("div.seg", h("button", { type: "button", on: { click: () => openChanges(chat) } }, "Changes"), h("button.on", { type: "button" }, "Files"));
  const body = h("div.code-view.files", h("div.faint.small", "Loading…"));
  panelShell(baseName(chat.cwd), [seg], body);
  let d;
  try { d = await api(`machines/${chat.machine}/ls?path=${encodeURIComponent(p)}`); }
  catch (e) { body.replaceChildren(h("div.note", e.message)); return; }
  const join = (name) => `${d.path}${/[\\/]$/.test(d.path) ? "" : d.path.includes("\\") ? "\\" : "/"}${name}`;
  const rel = d.path.startsWith(chat.cwd) ? d.path.slice(chat.cwd.length).replace(/^[\\/]/, "") : d.path;
  body.replaceChildren(
    h("div.fp-crumb", icon("folder"), h("span.ellipsis", rel || baseName(chat.cwd))),
    d.parent && d.path !== chat.cwd ? h("button.fp-row", { type: "button", on: { click: () => openFiles(chat, d.parent) } }, icon("chev", "flip"), h("span", "..")) : null,
    ...d.entries.filter((e) => !e.hidden || e.name === ".github").map((e) => e.dir
      ? h("button.fp-row", { type: "button", on: { click: () => openFiles(chat, join(e.name)) } }, icon("folder"), h("span.grow.ellipsis", e.name), icon("chev"))
      : h("button.fp-row", { type: "button", on: { click: () => viewFile(chat, join(e.name), d.path) } }, icon("file"), h("span.grow.ellipsis", e.name), h("span.faint.small", bytes(e.size)))));
}

async function viewFile(chat, file, back) {
  const body = document.querySelector("#panel .panel-body");
  body.replaceChildren(h("div.code-view", h("div.faint.small", "Loading…")));
  let f;
  try { f = await api(`machines/${chat.machine}/read?path=${encodeURIComponent(file)}`); }
  catch (e) { body.replaceChildren(h("div.code-view", h("div.note", e.message))); return; }
  const ext = file.split(".").pop();
  body.replaceChildren(h("div.code-view",
    h("div.fp-crumb", h("button.icon-btn.sm", { "aria-label": "Back", on: { click: () => openFiles(chat, back) } }, icon("chev", "flip")), h("span.ellipsis", baseName(file)),
      f.binary ? null : h("button.icon-btn.sm", { "aria-label": "Copy", on: { click: () => copyText(f.text) } }, icon("copy"))),
    f.binary ? h("div.empty-note", "Binary file.") : h("pre", h("code", { html: highlight(f.text, ext) })),
    f.truncated ? h("div.faint.small", "Showing the first 512 KB.") : null));
}
