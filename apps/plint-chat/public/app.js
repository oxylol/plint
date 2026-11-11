// Plint Chat front end: one page, hash routes (#/ new chat, #/c/<id> a chat).
const $ = (s) => document.querySelector(s);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};
const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const touch = matchMedia("(pointer: coarse)").matches;

async function api(path, opts = {}) {
  const r = await fetch(`/api/${path}`, {
    method: opts.method || (opts.body ? "POST" : "GET"),
    headers: { "content-type": "application/json", "x-plint": "1" },
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const data = await r.json().catch(() => ({}));
  if (r.status === 401 && path !== "login") showLogin();
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
  return data;
}

// --- markdown (escaped first, so model text can never inject HTML) ----------

function inline(s) {
  const codes = [];
  s = s.replace(/`([^`\n]+)`/g, (_, c) => `\u0000${codes.push(c) - 1}\u0000`);
  s = esc(s)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, "$1<em>$2</em>")
    .replace(/(^|[^\w])_([^_\n]+)_(?!\w)/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
    .replace(/(^|[\s(])(https?:\/\/[^\s<)]+)/g, '$1<a href="$2" target="_blank" rel="noopener">$2</a>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${esc(codes[i])}</code>`);
}

function markdown(src) {
  const out = [];
  const lines = src.replace(/\r/g, "").split("\n");
  let i = 0;
  const para = [];
  const flush = () => { if (para.length) { out.push(`<p>${inline(para.join("\n")).replace(/\n/g, "<br>")}</p>`); para.length = 0; } };
  while (i < lines.length) {
    const line = lines[i];
    let m;
    if ((m = line.match(/^\s*```\s*([\w+-]*)/))) {
      flush();
      const body = [];
      i++;
      while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) body.push(lines[i++]);
      i++;
      out.push(`<div class="code"><div class="code-head"><span>${esc(m[1] || "code")}</span><button data-copy>Copy</button></div><pre><code>${esc(body.join("\n"))}</code></pre></div>`);
      continue;
    }
    if ((m = line.match(/^(#{1,4})\s+(.*)/))) { flush(); out.push(`<h${m[1].length}>${inline(m[2])}</h${m[1].length}>`); i++; continue; }
    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { flush(); out.push("<hr>"); i++; continue; }
    if (/^\s*>/.test(line)) {
      flush();
      const q = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) q.push(lines[i++].replace(/^\s*>\s?/, ""));
      out.push(`<blockquote>${markdown(q.join("\n"))}</blockquote>`);
      continue;
    }
    if (/^\s*\|.*\|\s*$/.test(line) && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1] || "")) {
      flush();
      const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
      const head = cells(line);
      i += 2;
      const rows = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) rows.push(cells(lines[i++]));
      out.push(`<div class="table-wrap"><table><thead><tr>${head.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
      continue;
    }
    if ((m = line.match(/^(\s*)([-*+]|\d+[.)])\s+/))) {
      flush();
      const ordered = /\d/.test(m[2]);
      const items = [];
      while (i < lines.length && (m = lines[i].match(/^(\s*)([-*+]|\d+[.)])\s+(.*)/))) {
        const item = [m[3]];
        i++;
        while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*([-*+]|\d+[.)])\s+/.test(lines[i])) item.push(lines[i++].trim());
        items.push(`<li>${inline(item.join(" "))}</li>`);
      }
      out.push(ordered ? `<ol>${items.join("")}</ol>` : `<ul>${items.join("")}</ul>`);
      continue;
    }
    if (!line.trim()) { flush(); i++; continue; }
    para.push(line);
    i++;
  }
  flush();
  return out.join("");
}

document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-copy]");
  if (!b) return;
  navigator.clipboard?.writeText(b.closest(".code").querySelector("code").textContent);
  b.textContent = "Copied";
  setTimeout(() => (b.textContent = "Copy"), 1400);
});

// --- state -----------------------------------------------------------------

let models = {};
let current = null; // { id, model, auto, running, lastI }
let pendingModel = localStorage.getItem("model") || "auto-sonnet";
let es = null;
let draft = null; // { node, text }
const tools = new Map(); // tool_use id -> node
const asks = new Map();
let stick = true;

const thread = $("#thread");
const scroller = $("#scroll");
const input = $("#input");

function greet() {
  const h = new Date().getHours();
  const part = h < 5 ? "Up late" : h < 12 ? "Good morning" : h < 18 ? "Good afternoon" : "Good evening";
  $("#greeting").textContent = `${part}, Filip`;
}

function setModel(m) {
  $("#model-name").textContent = models[m] || m;
  if (current) current.model = m; else { pendingModel = m; try { localStorage.setItem("model", m); } catch {} }
}
function setRunning(running) {
  if (current) current.running = running;
  $("#send").hidden = running;
  $("#stop").hidden = !running;
  updateSend();
  const w = $(".working");
  if (running && !w) thread.append(working());
  if (!running && w) w.remove();
}
function setAuto(auto) {
  if (current) current.auto = auto;
  $("#auto").checked = !!auto;
  $("#auto-badge").hidden = !auto;
}
function updateSend() { $("#send").disabled = !input.value.trim(); }

function working() {
  const w = el("div", "working");
  w.innerHTML = `<svg class="spark" viewBox="0 0 24 24"><path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M5.6 18.4l2.8-2.8M15.6 8.4l2.8-2.8"/></svg><span>Working…</span>`;
  return w;
}
function append(node) {
  const w = $(".working");
  if (w) thread.insertBefore(node, w); else thread.append(node);
  if (stick) requestAnimationFrame(toBottom);
}
function toBottom() { scroller.scrollTop = scroller.scrollHeight; }
scroller.addEventListener("scroll", () => {
  stick = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 80;
  $("#jump").hidden = stick;
}, { passive: true });
$("#jump").onclick = () => { stick = true; scroller.scrollTo({ top: scroller.scrollHeight, behavior: "smooth" }); };

// --- rendering events ------------------------------------------------------

function toolSummary(name, input) {
  if (!input) return "";
  if (name === "Bash") return input.command;
  if (input.file_path) return input.file_path.replace(/^\/opt\/plint\//, "");
  return input.pattern || input.path || input.url || input.query || input.description || input.prompt || "";
}
function toolNode(ev) {
  const t = el("div", "tool busy");
  const head = el("button");
  head.append(el("span", "state"), el("span", "name", ev.name), el("span", "sum", toolSummary(ev.name, ev.input)));
  head.onclick = () => t.classList.toggle("open");
  const body = el("div", "body");
  const inp = ev.name === "Bash" ? ev.input.command : ev.name === "Edit" ? `${ev.input.file_path}\n- ${ev.input.old_string}\n+ ${ev.input.new_string}` : JSON.stringify(ev.input, null, 2);
  body.append(el("pre", "", inp));
  t.append(head, body);
  return t;
}
function askNode(ev) {
  const a = el("div", "ask");
  const head = el("div", "ask-head");
  head.innerHTML = `Allow <b>${esc(ev.name)}</b>? <span>needs your OK</span>`;
  const pre = el("pre", "", ev.name === "Bash" ? ev.input.command : JSON.stringify(ev.input, null, 2));
  const btns = el("div", "ask-btns");
  const answer = (allow, always) => async () => {
    btns.querySelectorAll("button").forEach((b) => (b.disabled = true));
    try { await api(`chats/${current.id}/answer`, { body: { id: ev.id, allow, always } }); }
    catch (e) { btns.querySelectorAll("button").forEach((b) => (b.disabled = false)); alert(e.message); }
  };
  const deny = el("button", "", "Deny"); deny.onclick = answer(false);
  const always = el("button", "", "Always"); always.onclick = answer(true, true);
  const allow = el("button", "allow", "Allow"); allow.onclick = answer(true);
  btns.append(deny, always, allow);
  a.append(head, pre, btns);
  if (navigator.userActivation?.hasBeenActive) navigator.vibrate?.(30);
  return a;
}

function handle(ev) {
  if (ev.i !== undefined) {
    if (ev.i <= current.lastI) return;
    current.lastI = ev.i;
  }
  switch (ev.t) {
    case "user": {
      document.body.classList.add("has-chat");
      append(el("div", "msg-user", ev.text));
      break;
    }
    case "delta": {
      if (!draft) { draft = { node: el("div", "msg-ai"), text: "" }; append(draft.node); }
      draft.text += ev.text;
      draft.node.innerHTML = markdown(draft.text);
      if (stick) toBottom();
      break;
    }
    case "text": {
      const node = draft?.node || el("div", "msg-ai");
      node.innerHTML = markdown(ev.text);
      if (!draft) append(node);
      draft = null;
      break;
    }
    case "thinking": {
      draft?.node.remove(); draft = null;
      const d = el("details", "thinking");
      d.innerHTML = `<summary>✻ Thought</summary>`;
      d.append(el("div", "", ev.text));
      append(d);
      break;
    }
    case "tool": {
      if (draft && !draft.text.trim()) draft.node.remove();
      draft = null;
      const n = toolNode(ev);
      tools.set(ev.id, n);
      append(n);
      break;
    }
    case "result": {
      const n = tools.get(ev.id);
      if (!n) break;
      n.classList.remove("busy");
      n.classList.add(ev.error ? "err" : "ok");
      n.querySelector(".body").append(el("pre", "", ev.output || "(no output)"));
      if (ev.error) n.classList.add("open");
      break;
    }
    case "ask": {
      const n = askNode(ev);
      asks.set(ev.id, n);
      append(n);
      break;
    }
    case "answer": {
      const n = asks.get(ev.id);
      if (!n) break;
      // The tool row above already shows how it went; the card only stays while it waits.
      n.remove();
      asks.delete(ev.id);
      break;
    }
    case "done": if (ev.error) append(el("div", "note", ev.error)); break;
    case "error": append(el("div", "note", ev.text)); break;
    case "idle": setRunning(false); break;
    case "state": setRunning(ev.running); setAuto(ev.auto); setModel(ev.model); break;
  }
}

// --- views -----------------------------------------------------------------

function connect() {
  es?.close();
  const id = current.id;
  es = new EventSource(`/api/chats/${id}/stream?after=${current.lastI}`);
  es.onmessage = (m) => { if (current?.id === id) handle(JSON.parse(m.data)); };
  es.onerror = () => { if (es.readyState === EventSource.CLOSED) setTimeout(() => current?.id === id && connect(), 2000); };
}
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && current && (!es || es.readyState === EventSource.CLOSED)) connect();
});

function reset() {
  es?.close(); es = null;
  thread.replaceChildren();
  tools.clear(); asks.clear(); draft = null; stick = true;
  closeMenus();
}

async function openChat(id) {
  reset();
  document.body.classList.add("has-chat");
  current = { id, lastI: -1, model: "auto-sonnet", auto: false, running: false };
  $("#title-btn").disabled = false;
  try {
    const chat = await api(`chats/${id}`);
    $("#title").textContent = chat.title;
    document.title = `${chat.title} · Plint`;
  } catch { location.hash = "#/"; return; }
  connect();
  highlightList();
}

function newChat() {
  reset();
  current = null;
  document.body.classList.remove("has-chat");
  $("#title").textContent = "New chat";
  $("#title-btn").disabled = true;
  document.title = "Plint";
  setModel(pendingModel);
  setRunning(false);
  setAuto(false);
  greet();
  highlightList();
  if (!touch) input.focus();
}

async function loadList() {
  let list = [];
  try { list = await api("chats"); } catch { return; }
  const nav = $("#chat-list");
  nav.replaceChildren(...list.map((c) => {
    const a = el("a");
    a.href = `#/c/${c.id}`;
    a.dataset.id = c.id;
    a.append(el("span", "", c.title));
    if (c.running) a.append(el("i", "dot"));
    return a;
  }));
  if (!list.length) nav.append(el("p", "muted small", "  No chats yet."));
  highlightList();
}
function highlightList() {
  document.querySelectorAll("#chat-list a").forEach((a) => a.classList.toggle("active", a.dataset.id === current?.id));
}

function route() {
  closeDrawer();
  const m = location.hash.match(/^#\/c\/([\w-]+)/);
  if (m) openChat(m[1]); else newChat();
}

// --- composer --------------------------------------------------------------

function autosize() { input.style.height = "auto"; input.style.height = `${input.scrollHeight}px`; }
input.addEventListener("input", () => { autosize(); updateSend(); });
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !touch && !e.isComposing) { e.preventDefault(); $("#composer").requestSubmit(); }
});
$("#composer").addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text || current?.running) return;
  input.value = ""; autosize(); updateSend();
  if (touch) input.blur();
  try {
    if (current) {
      setRunning(true);
      await api(`chats/${current.id}/send`, { body: { text, model: current.model } });
    } else {
      const c = await api("chats", { body: { text, model: pendingModel } });
      location.hash = `#/c/${c.id}`;
      loadList();
    }
  } catch (err) {
    input.value = text; autosize(); updateSend();
    setRunning(false);
    alert(err.message);
  }
});
$("#stop").onclick = () => current && api(`chats/${current.id}/stop`, { body: {} }).catch(() => {});

function closeMenus() { $("#model-menu").hidden = true; $("#chat-menu").hidden = true; }
$("#model-btn").onclick = (e) => {
  e.stopPropagation();
  const menu = $("#model-menu");
  const was = !menu.hidden;
  closeMenus();
  if (was) return;
  const active = current?.model || pendingModel;
  menu.replaceChildren(...Object.entries(models).map(([id, name]) => {
    const b = el("button", "menu-row");
    b.type = "button";
    b.innerHTML = `<span>${esc(name)}<small>${id === "auto-opus" ? "Kimi K3.1 Pro · deeper" : id === "auto-haiku" ? "DeepSeek Flash · quick" : "DeepSeek Flash · everyday"}</small></span>${id === active ? '<span class="check">✓</span>' : ""}`;
    b.onclick = () => {
      setModel(id); closeMenus();
      if (current) api(`chats/${current.id}/settings`, { body: { model: id } }).catch(() => {});
    };
    return b;
  }));
  menu.hidden = false;
};
$("#title-btn").onclick = (e) => { e.stopPropagation(); const m = $("#chat-menu"); const was = !m.hidden; closeMenus(); m.hidden = was; };
document.addEventListener("click", (e) => { if (!e.target.closest(".menu")) closeMenus(); });
$("#auto").onchange = (e) => current && api(`chats/${current.id}/settings`, { body: { auto: e.target.checked } }).catch((err) => alert(err.message));
$("#rename").onclick = async () => {
  const t = prompt("Rename chat", $("#title").textContent);
  if (!t?.trim() || !current) return;
  await api(`chats/${current.id}/settings`, { body: { title: t } });
  $("#title").textContent = t.trim();
  loadList();
};
$("#delete").onclick = async () => {
  if (!current || !confirm("Delete this chat?")) return;
  try { await api(`chats/${current.id}`, { method: "DELETE" }); location.hash = "#/"; loadList(); }
  catch (err) { alert(err.message); }
};

// --- drawer ----------------------------------------------------------------

function openDrawer() { $("#drawer").classList.add("open"); loadList(); }
function closeDrawer() { $("#drawer").classList.remove("open"); }
$("#open-drawer").onclick = openDrawer;
$("#close-drawer").onclick = closeDrawer;
$("#scrim").onclick = closeDrawer;
const goNew = () => { if (location.hash === "#/" || !location.hash) newChat(); location.hash = "#/"; closeDrawer(); };
$("#new-chat").onclick = goNew;
$("#new-chat-2").onclick = goNew;
$("#logout").onclick = async () => { await api("logout", { body: {} }).catch(() => {}); showLogin(); };

// --- sign in ---------------------------------------------------------------

function showLogin() {
  es?.close();
  $("#main").hidden = true;
  $("#login").hidden = false;
}
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
  try {
    await api("login", { body: { code: $("#code").value } });
    start();
  } catch (err) { $("#login-msg").textContent = err.message; $("#code").select(); }
};

async function start() {
  const me = await api("me");
  models = me.models;
  if (!me.signedIn) return showLogin();
  $("#login").hidden = true;
  $("#main").hidden = false;
  loadList();
  route();
}
window.addEventListener("hashchange", route);
start();
