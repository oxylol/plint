// The left sidebar: title bar with the Chat | Code switch, the nav rows, Pinned and
// Recents, and the account button. A drawer on phones; hidden when collapsed.
import { h, icon, menu, toast, narrow, phone, touch, confirmModal, formModal } from "./ui.js";
import { api } from "./api.js";
import { state, on, refreshChats, projectName, prefs } from "./state.js";
import { openSearch, openSettings, newProject } from "./pages.js";

const app = () => document.getElementById("app");
// On narrow screens the sidebar opens over the page; everything behind it is inert meanwhile.
export function openDrawer() {
  app().classList.add("drawer");
  document.querySelector(".main").inert = true;
  if (!touch) document.querySelector("#sidebar .side-top .icon-btn")?.focus();
}
export function closeDrawer() {
  if (!app().classList.contains("drawer")) return;
  app().classList.remove("drawer");
  document.querySelector(".main").inert = false;
}
export function toggleRail() {
  if (narrow()) return app().classList.contains("drawer") ? closeDrawer() : openDrawer();
  const rail = !app().classList.contains("rail");
  app().classList.remove("peek");
  peekHold = rail; // the pointer is still on the button: don't peek until it has left once
  app().classList.toggle("rail", rail);
  prefs.set("rail", rail);
}

let filter = prefs.get("recentsFilter", "all"); // all | chats | scheduled | projects
let mode = "chat"; // which app the sidebar shows: chat | code
const codeRoute = () => /^#\/(code|computers)/.test(location.hash) || state.chats.find((c) => c.id === state.current)?.kind === "code";
// Re-render the sidebar when the route moves between Chat and Code.
export function syncMode() {
  const want = codeRoute() ? "code" : "chat";
  if (want !== mode) { mode = want; renderSidebar(); }
}
const RECENT_LIMIT = 30;
const mac = /Mac|iPhone|iPad/.test(navigator.platform);
const keys = (...k) => h("span.kbd", k.flatMap((x, i) => (i ? ["+", h("kbd", x)] : [h("kbd", x)])));

// One nav row: icon, label, optional shortcut hint and trailing button.
function navRow({ href, ico, label, kbd, onClick, trail }) {
  const inner = [h("span.lead", icon(ico)), h("span.label", label), kbd || null];
  const row = href
    ? h("a.row", { href, "data-nav": href }, inner)
    : h("button.row", { type: "button", on: { click: onClick } }, inner);
  if (trail) row.append(h("span.trail", h("button.icon-btn.sm", { type: "button", "aria-label": trail.label, title: trail.label, on: { click: (e) => { e.preventDefault(); e.stopPropagation(); trail.onClick(e); } } }, icon("plus"))));
  return row;
}

export function renderSidebar() {
  const side = document.getElementById("sidebar");
  app().classList.toggle("rail", prefs.get("rail", false) && !narrow());
  const search = () => { closeDrawer(); openSearch(); };
  const nav = mode === "code"
    ? [
      navRow({ href: "#/code", ico: "plus", label: "New session", kbd: mac ? keys("⇧", "⌘", "O") : keys("Ctrl", "Shift", "O") }),
      navRow({ ico: "search", label: "Search", kbd: mac ? keys("⌘", "K") : keys("Ctrl", "K"), onClick: search }),
      navRow({ href: "#/customize", ico: "customize", label: "Customize" }),
    ]
    : [
      navRow({ href: "#/", ico: "plus", label: "New", kbd: mac ? keys("⇧", "⌘", "O") : keys("Ctrl", "Shift", "O") }),
      navRow({ ico: "search", label: "Search", kbd: mac ? keys("⌘", "K") : keys("Ctrl", "K"), onClick: search }),
      navRow({ href: "#/projects", ico: "projects", label: "Projects", trail: { label: "New project", onClick: () => { closeDrawer(); newProject(); } } }),
      navRow({ href: "#/artifacts", ico: "artifacts", label: "Artifacts" }),
      navRow({ href: "#/scheduled", ico: "clock", label: "Scheduled", trail: { label: "New task", onClick: () => { closeDrawer(); location.hash = "#/scheduled?new"; } } }),
      navRow({ href: "#/customize", ico: "customize", label: "Customize" }),
    ];
  nav[0].addEventListener("click", () => { state.incognito = false; });
  nav[1].classList.add("wide-only"); // on a phone Search is the big button under the title instead
  const small = narrow();
  side.replaceChildren(
    h("div.side-top",
      h("button.icon-btn", { type: "button", "aria-label": small ? "Close sidebar" : "Hide sidebar", title: small ? "Close sidebar" : "Hide sidebar", on: { click: toggleRail } }, icon(phone() ? "x" : "sidebar")),
      h("a.wordmark", { href: mode === "code" ? "#/code" : "#/" }, "Plint"),
      h("div.app-switch", { role: "radiogroup", "aria-label": "Mode", "data-mode": mode },
        h("span.thumb"),
        h(`a${mode === "chat" ? ".on" : ""}`, { href: "#/", role: "radio", "aria-checked": String(mode === "chat"), "aria-label": "Chat", title: "Chat" }, icon("chats")),
        h(`a${mode === "code" ? ".on" : ""}`, { href: "#/code", role: "radio", "aria-checked": String(mode === "code"), "aria-label": "Code", title: "Code" }, icon("code")))),
    h("button.side-search", { type: "button", on: { click: search } }, icon("search"), "Search"),
    h("div.side-body",
      h("div.side-scroll",
        h("nav.side-nav", { "aria-label": "Navigation" }, nav),
        h("div#side-lists"))),
    h("div.side-foot",
      h("button.user-btn", { type: "button", "aria-haspopup": "menu", on: { click: (e) => accountMenu(e.currentTarget) } },
        h("span.av", h("span.avatar", (state.me.name || "F")[0])),
        h("span.who", h("b", state.me.name || "Filip"), h("span", "·"), h("span", "Silk")),
        icon("down", "s12")),
      h("div.end", h("a.icon-btn", { href: "#/computers", "aria-label": "Computers", title: "Computers" }, icon("devices")))));
  side.append(h("div.side-resize", { role: "separator", "aria-orientation": "vertical", "aria-label": "Resize sidebar", on: { pointerdown: startResize } }));
  renderLists();
  highlight();
}

// Drag the sidebar's edge to resize it (232 to 420px).
const applyWidth = () => { const w = prefs.get("sidebarW", 288); document.documentElement.style.setProperty("--sidebar-w", `${w}px`); };
applyWidth();
function startResize(e) {
  if (narrow()) return;
  e.preventDefault();
  const handle = e.currentTarget;
  handle.setPointerCapture(e.pointerId);
  handle.classList.add("on");
  document.body.classList.add("resizing");
  const move = (ev) => document.documentElement.style.setProperty("--sidebar-w", `${Math.max(232, Math.min(420, ev.clientX))}px`);
  const up = (ev) => {
    handle.removeEventListener("pointermove", move);
    handle.classList.remove("on");
    document.body.classList.remove("resizing");
    prefs.set("sidebarW", Math.max(232, Math.min(420, Math.round(ev.clientX))));
  };
  handle.addEventListener("pointermove", move);
  handle.addEventListener("pointerup", up, { once: true });
  handle.addEventListener("dblclick", () => { prefs.set("sidebarW", 288); applyWidth(); }, { once: true });
}

// Collapsed on a computer: hovering the button in the header lets the sidebar peek out over the page.
let peekTimer = null, peekHold = false;
export function peek(on) {
  clearTimeout(peekTimer);
  if (narrow() || !app().classList.contains("rail")) return app().classList.remove("peek");
  if (on) app().classList.add("peek");
  else peekTimer = setTimeout(() => { if (!document.querySelector(".menu")) app().classList.remove("peek"); }, 220);
}
document.addEventListener("pointerover", (e) => {
  if (!app()?.classList.contains("rail") || narrow()) return;
  if (e.target.closest?.(".open-side, #sidebar")) { if (!peekHold) peek(true); }
  else if (app().classList.contains("peek") && !e.target.closest?.(".menu, .tip")) peek(false);
});

document.addEventListener("pointermove", (e) => { if (peekHold && !e.target.closest?.(".open-side")) peekHold = false; });

function chatRow(c) {
  const more = h("button.icon-btn.sm", { type: "button", "aria-label": `More options for ${c.title}`, on: { click: (e) => { e.preventDefault(); e.stopPropagation(); chatMenu(e.currentTarget, c); } } }, icon("more"));
  const row = h("a.row.chat", { href: `#/chat/${c.id}`, "data-id": c.id, "data-tip": c.title, "data-tip-if": "overflow" },
    h("span.label", c.title),
    c.waiting ? h("span.mark.wait", { title: "Awaiting input" }, icon("hand", "s16")) : c.running ? h("span.mark", { title: "Running" }, h("i.dot")) : c.hasArtifact ? h("span.mark", icon("file", "s16"), h("span.sr-only", "Includes an artifact")) : null,
    h("span.trail", more));
  // Touch: press and hold a row for its menu.
  let hold = null;
  row.addEventListener("touchstart", () => { hold = setTimeout(() => { hold = "fired"; navigator.vibrate?.(10); chatMenu(more, c); }, 480); }, { passive: true });
  for (const ev of ["touchmove", "touchend", "touchcancel"]) row.addEventListener(ev, (e) => { if (hold === "fired" && ev === "touchend") e.preventDefault(); else clearTimeout(hold); hold = null; });
  row.addEventListener("contextmenu", (e) => { e.preventDefault(); if (!touch) chatMenu(more, c); });
  return row;
}

// Rename in place: the row's label becomes a field.
function renameInPlace(c) {
  const row = document.querySelector(`#sidebar .row.chat[data-id="${c.id}"]`);
  const label = row?.querySelector(".label");
  if (!label) return formModal({ title: "Rename chat", submit: "Save", fields: [{ name: "title", value: c.title, autofocus: true }], onSubmit: (d) => api(`chats/${c.id}/settings`, { body: { title: d.title } }).then(refreshChats) });
  const input = h("input.row-input", { value: c.title, "aria-label": "Name", maxlength: 100 });
  let done = false;
  const finish = async (save) => {
    if (done) return;
    done = true;
    const v = input.value.trim();
    input.replaceWith(label);
    if (save && v && v !== c.title) { label.textContent = v; await api(`chats/${c.id}/settings`, { body: { title: v } }).catch((e) => toast(e.message)); refreshChats(); }
  };
  input.addEventListener("keydown", (e) => { e.stopPropagation(); if (e.key === "Enter") { e.preventDefault(); finish(true); } if (e.key === "Escape") finish(false); });
  input.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); });
  input.addEventListener("blur", () => finish(true));
  label.replaceWith(input);
  input.focus(); input.select();
}

export function chatMenu(anchor, c, { onDeleted, header = false } = {}) {
  const setProject = (projectId) => api(`chats/${c.id}/settings`, { body: { projectId } }).then(() => { refreshChats(); toast(projectId ? `Moved to ${projectName(projectId)}` : "Removed from project"); });
  menu(anchor, [
    { label: c.pinned ? "Unpin" : "Pin", key: "P", onClick: () => api(`chats/${c.id}/settings`, { body: { pinned: !c.pinned } }).then(refreshChats) },
    { label: "Rename", key: "R", onClick: () => (header ? formModal({ title: "Rename chat", submit: "Save", fields: [{ name: "title", value: c.title, autofocus: true }], onSubmit: (d) => api(`chats/${c.id}/settings`, { body: { title: d.title } }).then(refreshChats) }) : renameInPlace(c)) },
    c.kind === "code" ? null : { label: c.projectId ? "Change project" : "Add to project", children: () => [
      ...state.projects.map((p) => ({ label: p.name, checked: c.projectId === p.id, onClick: () => setProject(p.id) })),
      state.projects.length ? null : { heading: "No projects yet" },
      "sep",
      { label: "New project", icon: "plus", onClick: () => newProject() },
    ] },
    c.projectId ? { label: "Remove from project", onClick: () => setProject(null) } : null,
    "sep",
    { label: "Delete", key: "D", danger: true, onClick: async () => {
      if (!(await confirmModal("Delete chat", "Are you sure you want to delete this chat?"))) return;
      await api(`chats/${c.id}`, { method: "DELETE" });
      if (state.current === c.id) location.hash = c.kind === "code" ? "#/code" : "#/";
      refreshChats();
      onDeleted?.();
    } },
  ], { align: header ? "start" : "end" });
}

// A foldable group: "Pinned ⌄", "Recents ⌄", or a folder in Code.
function group(key, title, rows, { tool = null, tip = "" } = {}) {
  const folded = prefs.get(`fold.${key}`, false);
  const g = h(`div.side-group${folded ? ".folded" : ""}`);
  const fold = h("button.fold", { type: "button", "aria-expanded": String(!folded), title: tip, on: { click: () => {
    const now = !g.classList.contains("folded");
    g.classList.toggle("folded", now);
    fold.setAttribute("aria-expanded", String(!now));
    prefs.set(`fold.${key}`, now);
  } } }, h("span.ellipsis", title), icon("downSmall"));
  g.append(h("div.side-head", fold, tool), h("div.rows", rows));
  return g;
}

function renderCodeLists(box) {
  const sessions = state.chats.filter((c) => c.kind === "code");
  const groups = new Map();
  for (const c of sessions) {
    const k = `${c.machine}|${c.cwd}`;
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(c);
  }
  const name = (id) => (id === "plint" ? "Plint" : machineNames[id] || "Computer");
  box.replaceChildren(
    ...[...groups.entries()].map(([k, list]) => {
      const [machine, cwd] = k.split("|");
      const folder = String(cwd).replace(/[\\/]+$/, "").split(/[\\/]/).pop();
      return group(`code.${k}`, machine === "plint" ? folder : `${folder} · ${name(machine)}`, list.slice(0, 12).map(chatRow), { tip: `${name(machine)}: ${cwd}` });
    }),
    sessions.length ? null : h("div.side-empty", { style: { paddingTop: "16px" } }, "No sessions yet. Pick a folder and describe a task."));
}
let machineNames = {};
export function setMachineNames(list) { machineNames = Object.fromEntries(list.map((m) => [m.id, m.name])); if (mode === "code") renderLists(); }

function renderLists() {
  const box = document.getElementById("side-lists");
  if (!box) return;
  if (mode === "code") { renderCodeLists(box); highlight(); return; }
  const chatsOnly = state.chats.filter((c) => c.kind !== "code");
  const pinned = chatsOnly.filter((c) => c.pinned);
  let recents = chatsOnly.filter((c) => !c.pinned);
  if (filter === "chats") recents = recents.filter((c) => !c.scheduleId && !c.projectId);
  if (filter === "scheduled") recents = recents.filter((c) => c.scheduleId);
  if (filter === "projects") recents = recents.filter((c) => c.projectId);
  const TYPES = [["all", "All"], ["chats", "Chat"], ["projects", "In a project"], ["scheduled", "Scheduled"]];
  const filterBtn = h("button.icon-btn.sm", { type: "button", "aria-label": "Filter recents", title: "Filter recents", "aria-haspopup": "menu", on: { click: (e) => menu(e.currentTarget, [
    { label: "Type", right: TYPES.find(([k]) => k === filter)?.[1], children: TYPES.map(([k, l]) => ({ label: l, checked: filter === k, onClick: () => { filter = k; prefs.set("recentsFilter", k); renderLists(); } })) },
    filter === "all" ? null : "sep",
    filter === "all" ? null : { label: "Reset to defaults", onClick: () => { filter = "all"; prefs.set("recentsFilter", "all"); renderLists(); } },
  ], { align: "end" }) } }, icon("filter"));
  box.replaceChildren(
    pinned.length ? group("pinned", "Pinned", pinned.map(chatRow)) : null,
    group("recents", "Recents", [
      ...recents.slice(0, RECENT_LIMIT).map(chatRow),
      recents.length === 0 ? h("div.side-empty", "Nothing here yet.") : null,
      chatsOnly.length > RECENT_LIMIT || recents.length ? h("a.row.quiet", { href: "#/recents" }, "View all") : null,
    ], { tool: filterBtn }));
  highlight();
}

export function highlight() {
  const hash = (location.hash || "#/").split("?")[0];
  for (const a of document.querySelectorAll("#sidebar .row.chat")) a.classList.toggle("active", a.dataset.id === state.current);
  for (const a of document.querySelectorAll("#sidebar [data-nav]")) {
    const nav = a.dataset.nav;
    const home = nav === "#/" || nav === "#/code";
    a.classList.toggle("active", !state.current && (home ? hash === nav || (nav === "#/" && hash === "#") : hash.startsWith(nav)));
  }
}

export function setTheme(t) {
  try { t === "system" ? localStorage.removeItem("theme") : localStorage.setItem("theme", t); } catch {}
  if (t === "system") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = t;
}

function accountMenu(anchor) {
  const m = menu(anchor, [
    { heading: `${state.me.name || "Filip"} on Plint` },
    { label: "Settings", icon: "gear", right: mac ? "⌘," : "Ctrl+,", onClick: openSettings },
    { label: "Customize", icon: "customize", onClick: () => { location.hash = "#/customize"; } },
    { label: "Computers", icon: "devices", onClick: () => { location.hash = "#/computers"; } },
    "sep",
    { label: "Keyboard shortcuts", icon: "keyboard", onClick: showShortcuts },
    "sep",
    { label: "Log out", icon: "logout", onClick: async () => { await api("logout", { body: {} }).catch(() => {}); location.reload(); } },
  ], { align: "start", side: "above" });
  if (m && !m.classList.contains("sheet")) m.style.width = "272px";
}
function showShortcuts() {
  const mod = mac ? "⌘" : "Ctrl";
  const rows = [["New chat", `${mod}+Shift+O`], ["Search", `${mod}+K`], ["Hide or show the sidebar", `${mod}+.`], ["Settings", `${mod}+,`], ["Send message", "Enter"], ["New line", "Shift+Enter"], ["Stop response", "Esc"], ["Allow once (when asked)", `${mod}+Enter`], ["Deny (when asked)", "Esc"]];
  formModal({ title: "Keyboard shortcuts", fields: [], submit: "Done", cls: "confirm", extra: h("div.keys", rows.map(([l, k]) => h("div", h("span", l), h("span.muted", k)))) });
}

on("chats", () => { const was = mode; syncMode(); if (was === mode) renderLists(); });
on("projects", renderLists);
document.addEventListener("click", (e) => { if (narrow() && e.target.closest("#sidebar a[href]")) closeDrawer(); });

// The sheet follows a finger dragging it left and closes past a third of its width or on a quick flick.
{
  const side = () => document.getElementById("sidebar");
  let start = null;
  document.addEventListener("touchstart", (e) => {
    if (!app()?.classList.contains("drawer") || !e.target.closest("#sidebar") || document.querySelector(".menu")) return;
    const t = e.touches[0];
    start = { x: t.clientX, y: t.clientY, at: Date.now(), drag: false, dx: 0 };
  }, { passive: true });
  document.addEventListener("touchmove", (e) => {
    if (!start) return;
    const t = e.touches[0], dx = t.clientX - start.x, dy = t.clientY - start.y;
    if (!start.drag) {
      if (Math.abs(dx) < 10 && Math.abs(dy) < 10) return;
      if (dx > 0 || Math.abs(dx) < Math.abs(dy)) { start = null; return; }
      start.drag = true;
      side().classList.add("dragging");
    }
    start.dx = Math.min(0, dx);
    side().style.transform = `translateX(${start.dx}px)`;
  }, { passive: true });
  const end = () => {
    if (!start?.drag) { start = null; return; }
    const w = side().offsetWidth, fast = -start.dx / Math.max(1, Date.now() - start.at) > 0.5;
    side().classList.remove("dragging");
    side().style.transform = "";
    if (-start.dx > w / 3 || fast) closeDrawer();
    start = null;
  };
  document.addEventListener("touchend", end);
  document.addEventListener("touchcancel", end);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && app()?.classList.contains("drawer") && !document.querySelector(".menu, .modal-back")) closeDrawer(); });
  addEventListener("popstate", closeDrawer);
}
