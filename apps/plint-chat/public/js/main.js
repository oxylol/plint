// Boot, sign-in, routing and the top bar.
import { h, icon, $, toast, narrow, closeMenu, closeModal } from "./ui.js";
import { api, setUnauthorized } from "./api.js";
import { state, refreshChats, refreshProjects, refreshStyles, connectEvents } from "./state.js";
import { renderSidebar, highlight, openDrawer, closeDrawer } from "./sidebar.js";
import { renderHome, renderChat } from "./chat.js";
import { renderProjects, renderProject, renderArtifacts, renderScheduled, renderCustomize, renderRecents, openSearch } from "./pages.js";

// The bar above every view. title: text; onTitle: click handler (menu); right: nodes.
export function topBar({ title = "", onTitle = null, badges = [], right = [] } = {}) {
  const top = $("#top");
  top.replaceChildren(
    h("button.icon-btn.menu-btn", { "aria-label": "Open sidebar", on: { click: openDrawer } }, icon("menu")),
    h("div.center",
      title ? h("button.title-btn", { disabled: !onTitle, on: onTitle ? { click: onTitle } : {} }, h("span.t", title), icon("down")) : null,
      ...badges.filter(Boolean).map((b) => (b.style.marginLeft = "4px", b.style.alignSelf = "center", b))),
    ...right.filter(Boolean),
    h("a.icon-btn", { href: "#/", title: "New chat", "aria-label": "New chat", style: narrow() ? {} : { display: "none" }, on: { click: () => { state.incognito = false; } } }, icon("edit")));
}

let cleanup = null;
function route() {
  closeMenu();
  closeModal();
  if (narrow()) closeDrawer();
  cleanup?.();
  cleanup = null;
  const view = $("#view");
  const hash = location.hash.replace(/^#/, "") || "/";
  const [path, query = ""] = hash.split("?");
  const params = new URLSearchParams(query);
  const seg = path.split("/").filter(Boolean);
  state.current = null;
  document.title = "Plint";
  if (seg[0] === "chat" && seg[1]) cleanup = renderChat(view, seg[1]);
  else if (seg[0] === "projects") cleanup = renderProjects(view, { create: seg[1] === "new" });
  else if (seg[0] === "project" && seg[1]) cleanup = renderProject(view, seg[1]);
  else if (seg[0] === "artifacts") cleanup = renderArtifacts(view);
  else if (seg[0] === "scheduled") cleanup = renderScheduled(view);
  else if (seg[0] === "customize") cleanup = renderCustomize(view, seg[1]);
  else if (seg[0] === "recents") cleanup = renderRecents(view);
  else renderHome(view, { projectId: params.get("project") });
  highlight();
}

// --- sign in ---------------------------------------------------------------------

function showLogin() {
  $("#app").hidden = true;
  $("#login").hidden = false;
  const sparkSlot = $("#login [data-spark]");
  if (sparkSlot && !sparkSlot.firstChild) import("./ui.js").then(({ spark }) => sparkSlot.replaceWith(Object.assign(spark(), { style: "width:40px;height:40px" })));
}
setUnauthorized(showLogin);
$("#send-code").onclick = async () => {
  const b = $("#send-code");
  b.disabled = true;
  try {
    await api("login/send", { body: {} });
    $("#login-msg").textContent = "Code sent. Check your phone.";
    $("#code-form").hidden = false;
    b.textContent = "Send a new code";
    $("#code").focus();
  } catch (e) { $("#login-msg").textContent = e.message; }
  setTimeout(() => (b.disabled = false), 30000);
};
$("#code-form").onsubmit = async (e) => {
  e.preventDefault();
  try { await api("login", { body: { code: $("#code").value } }); start(); }
  catch (err) { $("#login-msg").textContent = err.message; $("#code").select(); }
};

// --- keyboard ------------------------------------------------------------------------

document.addEventListener("keydown", (e) => {
  const mod = e.metaKey || e.ctrlKey;
  if (!state.me) return;
  if (mod && e.key.toLowerCase() === "k") { e.preventDefault(); openSearch(); }
  if (mod && e.shiftKey && e.key.toLowerCase() === "o") { e.preventDefault(); state.incognito = false; location.hash = "#/"; }
  if (mod && e.key === ".") { e.preventDefault(); import("./sidebar.js").then((m) => m.toggleRail()); }
});

// --- start -----------------------------------------------------------------------------

async function start() {
  let me;
  try { me = await api("me"); } catch { toast("Can't reach Plint Chat."); return; }
  if (!me.signedIn) return showLogin();
  Object.assign(state, { me, models: me.models, efforts: me.efforts, styles: me.styles });
  $("#login").hidden = true;
  $("#app").hidden = false;
  await Promise.all([refreshChats(), refreshProjects()]);
  renderSidebar();
  connectEvents();
  route();
}
$("#scrim").addEventListener("click", closeDrawer);
addEventListener("hashchange", route);
let wasNarrow = narrow();
addEventListener("resize", () => { if (narrow() !== wasNarrow) { wasNarrow = narrow(); if (state.me) { renderSidebar(); route(); } } });
start();
