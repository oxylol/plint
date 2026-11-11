// Boot, sign-in, routing and the top bar.
import { h, icon, $, toast, narrow, closeMenu, closeModal, useGlyphs, spark } from "./ui.js";
import { api, setUnauthorized } from "./api.js";
import { state, refreshChats, refreshProjects, refreshStyles, connectEvents } from "./state.js";
import { renderSidebar, highlight, closeDrawer, syncMode, toggleRail } from "./sidebar.js";
import { renderHome, renderChat } from "./chat.js";
import { renderCodeHome, renderComputers, refreshMachines } from "./code.js";
import { renderProjects, renderProject, renderArtifacts, renderScheduled, renderCustomize, renderRecents, openSearch, openSettings } from "./pages.js";

// The bar above every view. title: text; onRename(newTitle): makes the title editable in place;
// onTitle: click handler for the caret beside it (menu); badges, right: nodes.
export function topBar({ title = "", onRename = null, onTitleClick = null, onTitle = null, badges = [], right = [] } = {}) {
  const top = $("#top");
  const titleBtn = title ? h("button.btn.title-btn", { type: "button", disabled: !onRename && !onTitleClick, title: onRename ? "Rename" : null, on: onTitleClick ? { click: onTitleClick } : {} }, h("span.t", title)) : null;
  if (titleBtn && onRename) titleBtn.onclick = () => {
    const input = h("input.title-input", { value: title, "aria-label": "Chat name", maxlength: 100 });
    let done = false;
    const finish = async (save) => {
      if (done) return;
      done = true;
      const v = input.value.trim();
      if (save && v && v !== title) { titleBtn.firstChild.textContent = v; try { await onRename(v); } catch (e) { toast(e.message); } }
      input.replaceWith(titleBtn);
    };
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") finish(true); if (e.key === "Escape") { e.stopPropagation(); finish(false); } });
    input.addEventListener("blur", () => finish(true));
    titleBtn.replaceWith(input);
    input.focus(); input.select();
  };
  top.replaceChildren(
    h("button.icon-btn.open-side", { type: "button", "aria-label": "Open sidebar", title: "Open sidebar", on: { click: toggleRail } }, icon("sidebar")),
    h("div.lead-slot",
      title ? h("div.title", titleBtn, onTitle ? h("button.icon-btn", { type: "button", "aria-label": "Chat options", "aria-haspopup": "menu", on: { click: onTitle } }, icon("down", "s16")) : null) : null,
      ...badges.filter(Boolean)),
    h("div.actions", ...right.filter(Boolean)));
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
  $("#app").classList.remove("no-top");
  if (seg[0] === "chat" && seg[1]) cleanup = renderChat(view, seg[1]);
  else if (seg[0] === "projects") cleanup = renderProjects(view, { create: seg[1] === "new" });
  else if (seg[0] === "project" && seg[1]) cleanup = renderProject(view, seg[1]);
  else if (seg[0] === "artifacts") cleanup = renderArtifacts(view);
  else if (seg[0] === "scheduled") cleanup = renderScheduled(view, { create: params.has("new") });
  else if (seg[0] === "customize") cleanup = renderCustomize(view, seg[1]);
  else if (seg[0] === "recents") cleanup = renderRecents(view);
  else if (seg[0] === "code") cleanup = renderCodeHome(view);
  else if (seg[0] === "computers") cleanup = renderComputers(view);
  else renderHome(view, { projectId: params.get("project") });
  syncMode();
  highlight();
}

// --- sign in ---------------------------------------------------------------------

function showLogin() {
  $("#app").hidden = true;
  $("#login").hidden = false;
  const sparkSlot = $("#login [data-spark]");
  if (sparkSlot) sparkSlot.replaceWith(Object.assign(spark(), { style: "width:32px;height:32px" }));
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
  if (mod && (e.key === "." || e.key === "\\")) { e.preventDefault(); toggleRail(); }
  if (mod && e.key === ",") { e.preventDefault(); openSettings(); }
});

// --- start -----------------------------------------------------------------------------

async function start() {
  let me;
  try { me = await api("me"); } catch { toast("Can't reach Plint Chat."); return; }
  useGlyphs(me.brand);
  if (!me.signedIn) return showLogin();
  Object.assign(state, { me, models: me.models, efforts: me.efforts, styles: me.styles });
  $("#login").hidden = true;
  $("#app").hidden = false;
  await Promise.all([refreshChats(), refreshProjects()]);
  renderSidebar();
  refreshMachines();
  connectEvents();
  route();
}
addEventListener("hashchange", route);
let wasNarrow = narrow();
addEventListener("resize", () => { if (narrow() !== wasNarrow) { wasNarrow = narrow(); if (state.me) { renderSidebar(); route(); } } });
start();
