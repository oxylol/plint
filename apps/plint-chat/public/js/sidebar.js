// The left sidebar: New chat, Search, Projects, Artifacts, Scheduled, Customize,
// Pinned and Recents, and the account menu. A drawer on phones, a rail when collapsed.
import { h, icon, spark, menu, toast, narrow, confirmModal, formModal } from "./ui.js";
import { api } from "./api.js";
import { state, on, refreshChats, projectName, prefs } from "./state.js";
import { openSearch, openSettings } from "./pages.js";

const app = () => document.getElementById("app");
export function openDrawer() { app().classList.add("drawer"); }
export function closeDrawer() { app().classList.remove("drawer"); }
export function toggleRail() {
  if (narrow()) return openDrawer();
  const rail = !app().classList.contains("rail");
  app().classList.toggle("rail", rail);
  prefs.set("rail", rail);
}

let filter = prefs.get("recentsFilter", "all"); // all | chats | scheduled | projects
const RECENT_LIMIT = 30;

export function renderSidebar() {
  const side = document.getElementById("sidebar");
  app().classList.toggle("rail", prefs.get("rail", false) && !narrow());
  const navItem = (href, ico, label, kbd) => h("a.nav-item", { href, "data-nav": href, title: label }, icon(ico), h("span.label", label), kbd ? h("span.kbd", kbd) : null);
  const mac = /Mac|iPhone|iPad/.test(navigator.platform);
  side.replaceChildren(
    h("div.side-top",
      h("a.wordmark", { href: "#/" }, spark(), h("span.w", "Plint")),
      h("button.icon-btn.collapse", { title: "Toggle sidebar", "aria-label": "Toggle sidebar", on: { click: toggleRail } }, icon("sidebar")),
      h("button.icon-btn", { title: "Close", "aria-label": "Close", style: narrow() ? {} : { display: "none" }, on: { click: closeDrawer } }, icon("x"))),
    h("nav.side-nav",
      h("a.nav-item.new", { href: "#/", title: "New chat", on: { click: () => { state.incognito = false; } } }, icon("plus"), h("span.label", "New chat"), h("span.kbd", mac ? "⇧⌘O" : "Ctrl+Shift+O")),
      h("button.nav-item", { title: "Search", on: { click: () => { closeDrawer(); openSearch(); } } }, icon("search"), h("span.label", "Search"), h("span.kbd", mac ? "⌘K" : "Ctrl+K")),
      navItem("#/projects", "folder", "Projects"),
      navItem("#/artifacts", "artifacts", "Artifacts"),
      navItem("#/scheduled", "clock", "Scheduled"),
      navItem("#/customize", "sliders", "Customize")),
    h("div.side-scroll#side-lists"),
    h("div.side-foot",
      h("button.user-btn", { on: { click: (e) => accountMenu(e.currentTarget) } },
        h("span.avatar", (state.me.name || "F")[0].toUpperCase()),
        h("span.meta.grow", h("div.who.ellipsis", state.me.name || "Filip"), h("div.plan", "Plint · Silk models")),
        h("span.label", icon("down")))));
  renderLists();
  highlight();
}

function chatLink(c) {
  const a = h("a.chat-link", { href: `#/chat/${c.id}`, "data-id": c.id, title: c.title },
    h("span.t", c.title),
    c.running ? h("i.dot", { title: "Working" }) : null,
    h("button.icon-btn.sm.more", { "aria-label": `More options for ${c.title}`, on: { click: (e) => { e.preventDefault(); e.stopPropagation(); chatMenu(e.currentTarget, c); } } }, icon("more")));
  return a;
}

export function chatMenu(anchor, c, { onDeleted } = {}) {
  menu(anchor, [
    { label: c.pinned ? "Unpin" : "Pin", icon: "pin", onClick: () => api(`chats/${c.id}/settings`, { body: { pinned: !c.pinned } }).then(refreshChats) },
    { label: "Rename", icon: "edit", onClick: () => formModal({ title: "Rename chat", fields: [{ name: "title", label: "Name", value: c.title, autofocus: true }],
      onSubmit: (d) => api(`chats/${c.id}/settings`, { body: { title: d.title } }).then(refreshChats) }) },
    { label: c.projectId ? "Move to project" : "Add to project", icon: "folder", onClick: () => moveToProject(c) },
    "sep",
    { label: "Delete", icon: "trash", danger: true, onClick: async () => {
      if (!(await confirmModal("Delete chat?", `"${c.title}" will be gone for good.`))) return;
      await api(`chats/${c.id}`, { method: "DELETE" });
      if (state.current === c.id) location.hash = "#/";
      refreshChats();
      onDeleted?.();
    } },
  ], { align: "start" });
}

export function moveToProject(c) {
  return formModal({
    title: "Move to project",
    fields: [{ name: "projectId", label: "Project", type: "select", value: c.projectId || "", options: [["", "No project"], ...state.projects.map((p) => [p.id, p.name])] }],
    submit: "Move",
    onSubmit: (d) => api(`chats/${c.id}/settings`, { body: { projectId: d.projectId || null } }).then(() => { refreshChats(); toast(d.projectId ? `Moved to ${projectName(d.projectId)}` : "Removed from project"); }),
  });
}

function renderLists() {
  const box = document.getElementById("side-lists");
  if (!box) return;
  const pinned = state.chats.filter((c) => c.pinned);
  let recents = state.chats.filter((c) => !c.pinned);
  if (filter === "chats") recents = recents.filter((c) => !c.scheduleId && !c.projectId);
  if (filter === "scheduled") recents = recents.filter((c) => c.scheduleId);
  if (filter === "projects") recents = recents.filter((c) => c.projectId);
  const filterBtn = h("button.icon-btn.sm", { "aria-label": "Filter recents", title: "Filter", on: { click: (e) => menu(e.currentTarget, [
    { heading: "Show" },
    ...[["all", "All chats"], ["chats", "Chats only"], ["projects", "In projects"], ["scheduled", "Scheduled runs"]].map(([k, l]) => ({ label: l, checked: filter === k, onClick: () => { filter = k; prefs.set("recentsFilter", k); renderLists(); highlight(); } })),
  ], { align: "end" }) } }, icon("filter"));
  box.replaceChildren(
    pinned.length ? h("div.side-head", h("span", "Pinned")) : null,
    ...pinned.map(chatLink),
    h("div.side-head", h("span", filter === "all" ? "Recents" : `Recents · ${{ chats: "chats", projects: "projects", scheduled: "scheduled" }[filter]}`), filterBtn),
    ...recents.slice(0, RECENT_LIMIT).map(chatLink),
    recents.length === 0 ? h("div.faint.small", { style: { padding: "6px 10px" } }, "Nothing here yet.") : null,
    state.chats.length > RECENT_LIMIT ? h("a.view-all", { href: "#/recents" }, "View all") : null);
}

export function highlight() {
  const hash = location.hash || "#/";
  for (const a of document.querySelectorAll("#sidebar .chat-link")) a.classList.toggle("active", a.dataset.id === state.current);
  for (const a of document.querySelectorAll("#sidebar [data-nav]")) a.classList.toggle("active", hash.startsWith(a.dataset.nav) && !state.current);
}

function accountMenu(anchor) {
  const theme = document.documentElement.dataset.theme || "system";
  const setTheme = (t) => {
    try { t === "system" ? localStorage.removeItem("theme") : localStorage.setItem("theme", t); } catch {}
    if (t === "system") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = t;
  };
  menu(anchor, [
    { heading: state.me.name || "Filip" },
    { label: "Settings", icon: "gear", onClick: openSettings },
    { label: "Customize", icon: "sliders", onClick: () => { location.hash = "#/customize"; } },
    "sep",
    { heading: "Appearance" },
    { label: "System", icon: "gear", checked: theme === "system", onClick: () => setTheme("system") },
    { label: "Light", icon: "sun", checked: theme === "light", onClick: () => setTheme("light") },
    { label: "Dark", icon: "moon", checked: theme === "dark", onClick: () => setTheme("dark") },
    "sep",
    { label: "Sign out", icon: "logout", onClick: async () => { await api("logout", { body: {} }).catch(() => {}); location.reload(); } },
  ], { align: "start", side: "auto" });
}

on("chats", () => { renderLists(); highlight(); });
on("projects", renderLists);
document.addEventListener("click", (e) => { if (narrow() && e.target.closest("#sidebar a[href]")) closeDrawer(); });
