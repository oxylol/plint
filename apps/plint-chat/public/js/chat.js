// The composer, the home screen and the chat thread.
import { h, icon, spark, menu, toast, touch, narrow, copyText, bytes, esc, $ } from "./ui.js";
import { api, upload } from "./api.js";
import { markdown } from "./md.js";
import { state, prefs, refreshChats, projectName, on } from "./state.js";
import { openArtifact, artifactCard, artifactTile, closePanel, panelChat } from "./artifacts.js";
import { chatMenu } from "./sidebar.js";
import { topBar } from "./main.js";

const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
const modelLabel = (id, short) => { const n = state.models[id]?.name || id; return short ? n.replace(/^Auto /, "") : n; };
const cap = (s) => s[0].toUpperCase() + s.slice(1);

// --- composer -----------------------------------------------------------------
// opts: { chat (summary) | null, projectId, placeholder, onSend(text, fileIds, settings) }
export function composer(opts) {
  const s = {
    model: opts.chat?.model || prefs.get("model", "auto-sonnet"),
    effort: opts.chat?.effort || prefs.get("effort", "medium"),
    mode: opts.chat?.mode || prefs.get("mode", "manual"),
    style: opts.chat?.style || prefs.get("style", "normal"),
    projectId: opts.chat ? opts.chat.projectId : opts.projectId ?? null,
    files: [],
    running: false,
  };
  if (!state.models[s.model]) s.model = "auto-sonnet";
  const isNew = !opts.chat;
  const ta = h("textarea", { rows: 1, placeholder: opts.placeholder || "How can I help you today?", "aria-label": "Write your prompt", enterkeyhint: touch ? "enter" : "send" });
  const atts = h("div.attachments", { hidden: true });
  const fileInput = h("input", { type: "file", multiple: true, hidden: true });
  const photoInput = h("input", { type: "file", accept: "image/*", capture: "environment", hidden: true });
  const send = h("button.send", { type: "submit", "aria-label": "Send message", disabled: true }, icon("up"));
  const stop = h("button.send.stop", { type: "button", "aria-label": "Stop", hidden: true }, icon("stop"));
  const plusBtn = h("button.c-btn.sq", { type: "button", "aria-label": "Add files and more", title: "Add files and more" }, icon("plus"));
  const projBtn = h("button.c-btn", { type: "button", title: "Project" });
  const modelBtn = h("button.c-btn", { type: "button", title: "Model" });
  const modeBtn = h("button.c-btn", { type: "button", title: "Permission mode" });
  const micBtn = h("button.c-btn", { type: "button", title: "Dictate", "aria-label": "Dictate", hidden: !SR }, icon("mic"));
  const form = h("form.composer", atts, ta,
    h("div.c-row", plusBtn, projBtn, h("span.grow"), modelBtn, modeBtn, micBtn, send, stop),
    fileInput, photoInput);

  const persist = (patch) => {
    Object.assign(s, patch);
    if (isNew) for (const k of ["model", "effort", "mode", "style"]) if (k in patch) prefs.set(k, s[k]);
    if (!isNew && opts.chat) api(`chats/${opts.chat.id}/settings`, { body: patch }).catch((e) => toast(e.message));
    paint();
  };
  function paint() {
    const short = narrow();
    modelBtn.replaceChildren(h("span.t", modelLabel(s.model, short)), state.models[s.model]?.efforts ? h("span.faint", ` ${cap(s.effort)}`) : null, icon("down", "chev"));
    modeBtn.replaceChildren(icon(s.mode === "auto" ? "bolt" : "hand"), short ? null : h("span.t", s.mode === "auto" ? "Auto" : "Manual"));
    modeBtn.classList.toggle("on", s.mode === "auto");
    const pname = s.projectId && projectName(s.projectId);
    projBtn.replaceChildren(icon("folder"), pname ? h("span.t", pname) : short ? null : h("span.t", "Project"));
    projBtn.classList.toggle("on", !!pname);
    projBtn.hidden = !isNew && !pname;
    projBtn.disabled = !isNew;
    plusBtn.classList.toggle("on", s.style !== "normal");
    const busy = s.files.some((f) => f.uploading);
    send.disabled = busy || !ta.value.trim();
    send.hidden = s.running;
    stop.hidden = !s.running;
    atts.hidden = !s.files.length;
  }
  function autosize() { ta.style.height = "auto"; ta.style.height = `${Math.min(ta.scrollHeight, innerHeight * 0.38)}px`; }

  // files
  function drawFiles() {
    atts.replaceChildren(...s.files.map((f) => h(`div.att${f.uploading ? ".loading" : ""}`,
      f.type?.startsWith("image/") && f.preview ? h("img", { src: f.preview, alt: "" }) : h("span.ficon", (f.name.split(".").pop() || "file").slice(0, 4)),
      h("div.meta", h("div", f.name), h("div.faint", f.uploading ? `Uploading ${Math.round((f.progress || 0) * 100)}%` : bytes(f.size))),
      h("button.x", { type: "button", "aria-label": `Remove ${f.name}`, on: { click: () => { s.files = s.files.filter((x) => x !== f); drawFiles(); paint(); } } }, icon("x")))));
    paint();
  }
  async function addFiles(list) {
    for (const file of list) {
      if (file.size > 25e6) { toast(`${file.name} is over 25 MB`); continue; }
      const f = { name: file.name || "pasted.png", size: file.size, type: file.type, uploading: true, progress: 0, preview: file.type.startsWith("image/") ? URL.createObjectURL(file) : null };
      s.files.push(f);
      drawFiles();
      try {
        const meta = await upload("uploads", file, (p) => { f.progress = p; drawFiles(); });
        Object.assign(f, meta, { uploading: false });
      } catch (e) { toast(e.message); s.files = s.files.filter((x) => x !== f); }
      drawFiles();
    }
  }
  fileInput.onchange = () => { addFiles([...fileInput.files]); fileInput.value = ""; };
  photoInput.onchange = () => { addFiles([...photoInput.files]); photoInput.value = ""; };
  ta.addEventListener("paste", (e) => {
    const files = [...(e.clipboardData?.files || [])];
    if (files.length) { e.preventDefault(); addFiles(files); }
  });
  form.addEventListener("dragover", (e) => { e.preventDefault(); if (!form.querySelector(".drop-hint")) form.append(h("div.drop-hint", "Drop files to attach")); });
  form.addEventListener("dragleave", (e) => { if (!form.contains(e.relatedTarget)) form.querySelector(".drop-hint")?.remove(); });
  form.addEventListener("drop", (e) => { e.preventDefault(); form.querySelector(".drop-hint")?.remove(); addFiles([...e.dataTransfer.files]); });

  // menus
  plusBtn.onclick = (e) => menu(e.currentTarget, [
    { label: "Upload files", icon: "paperclip", onClick: () => fileInput.click() },
    touch ? { label: "Take a photo", icon: "image", onClick: () => photoInput.click() } : null,
    "sep",
    { heading: "Use style" },
    ...state.styles.map((st) => ({ label: st.name, checked: s.style === st.id, onClick: () => persist({ style: st.id }) })),
    { label: "Create & edit styles", icon: "sliders", onClick: () => { location.hash = "#/customize/styles"; } },
    isNew ? "sep" : null,
    isNew ? { label: "Incognito chat", sub: "Not saved to your chats", icon: "ghost", switch: true, on: state.incognito, onClick: (v) => { state.incognito = v; opts.onIncognito?.(v); } } : null,
  ], { align: "start" });
  projBtn.onclick = (e) => menu(e.currentTarget, [
    { heading: "Start in a project" },
    { label: "No project", checked: !s.projectId, onClick: () => persist({ projectId: null }) },
    ...state.projects.map((p) => ({ label: p.name, icon: "folder", checked: s.projectId === p.id, onClick: () => persist({ projectId: p.id }) })),
    "sep",
    { label: "New project", icon: "plus", onClick: () => { location.hash = "#/projects/new"; } },
  ], { align: "start" });
  modelBtn.onclick = (e) => menu(e.currentTarget, [
    ...Object.entries(state.models).map(([id, m]) => ({ label: m.name, sub: m.sub, checked: s.model === id, onClick: () => persist({ model: id }) })),
    state.models[s.model]?.efforts ? "sep" : null,
    state.models[s.model]?.efforts ? { heading: "Effort" } : null,
    ...(state.models[s.model]?.efforts ? state.efforts.map((ef) => ({ label: cap(ef), sub: { low: "Fastest answers", medium: "Balanced", high: "Thinks longest" }[ef], checked: s.effort === ef, onClick: () => persist({ effort: ef }) })) : []),
  ], { align: "end" });
  modeBtn.onclick = (e) => menu(e.currentTarget, [
    { heading: "Mode" },
    { label: "Manual", sub: "Ask before running commands or editing files", icon: "hand", checked: s.mode !== "auto", onClick: () => persist({ mode: "manual" }) },
    { label: "Auto", sub: "Run everything without asking", icon: "bolt", checked: s.mode === "auto", onClick: () => persist({ mode: "auto" }) },
  ], { align: "end" });

  // dictation (Web Speech API)
  let rec = null;
  micBtn.onclick = () => {
    if (rec) { rec.stop(); return; }
    rec = new SR();
    rec.lang = navigator.language || "en-US";
    rec.continuous = true;
    rec.interimResults = true;
    const base = ta.value ? `${ta.value.replace(/\s+$/, "")} ` : "";
    rec.onresult = (ev) => {
      let text = "";
      for (const r of ev.results) text += r[0].transcript;
      ta.value = base + text;
      autosize(); paint();
    };
    rec.onend = () => { rec = null; micBtn.classList.remove("listening"); };
    rec.onerror = (ev) => { if (ev.error !== "aborted") toast(ev.error === "not-allowed" ? "Microphone access was blocked" : `Dictation: ${ev.error}`); };
    rec.start();
    micBtn.classList.add("listening");
  };

  // send
  ta.addEventListener("input", () => { autosize(); paint(); });
  ta.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing && !touch && prefs.get("enterSends", true)) { e.preventDefault(); form.requestSubmit(); }
    if (e.key === "Escape" && s.running) opts.onStop?.();
  });
  form.addEventListener("click", (e) => { if (e.target === form) ta.focus(); });
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = ta.value.trim();
    if (!text || s.running || s.files.some((f) => f.uploading)) return;
    rec?.stop();
    const files = s.files;
    ta.value = ""; s.files = []; drawFiles(); autosize();
    if (touch) ta.blur();
    try {
      await opts.onSend(text, files.map((f) => f.id), { model: s.model, effort: s.effort, mode: s.mode, style: s.style, projectId: s.projectId });
    } catch (err) {
      ta.value = text; s.files = files; drawFiles(); autosize();
      toast(err.message);
    }
  });
  stop.onclick = () => opts.onStop?.();
  addEventListener("resize", paint);
  paint();
  return {
    el: form, ta,
    focus: () => { if (!touch) ta.focus(); },
    setText: (t) => { ta.value = t; autosize(); paint(); ta.focus(); ta.setSelectionRange(t.length, t.length); },
    setRunning: (r) => { s.running = r; paint(); },
    setChat: (c) => { Object.assign(s, { model: c.model, effort: c.effort, mode: c.mode, style: c.style, projectId: c.projectId }); paint(); },
    repaint: paint,
  };
}

// --- home ---------------------------------------------------------------------

const CHIPS = [
  ["terminal", "Check Plint", "How is Plint doing? Run plint status and tell me anything that needs attention."],
  ["game", "Game servers", "Which game servers are running right now, and is anyone playing?"],
  ["box", "Updates", "Are there container or OS updates waiting on Plint? Don't install anything yet."],
  ["shield", "Backups", "When did the last backup run and did it succeed?"],
  ["pen", "Write", "Help me write "],
];
function greeting(name) {
  const hr = new Date().getHours();
  const timely = hr < 5 ? "Up late" : hr < 12 ? "Good morning" : hr < 18 ? "Good afternoon" : "Good evening";
  const options = [`${timely}, ${name}`, `Back at it, ${name}`, `${timely}, ${name}`, `How can I help, ${name}?`];
  return options[Math.floor(Date.now() / 3.6e6) % options.length];
}

export function renderHome(view, { projectId = null } = {}) {
  state.current = null;
  closePanel();
  let incog = state.incognito;
  const incogBtn = h("button.icon-btn", { title: "Incognito chat", "aria-label": "Use incognito" }, icon("ghost"));
  const box = h("div.home");
  const greet = h("div.greet");
  const c = composer({
    projectId,
    onIncognito: (v) => setIncog(v),
    onSend: async (text, files, settings) => {
      const chat = await api("chats", { body: { text, files, ...settings, incognito: incog } });
      if (!chat.incognito) refreshChats();
      location.hash = `#/chat/${chat.id}`;
    },
  });
  function setIncog(v) {
    incog = state.incognito = v;
    box.classList.toggle("incognito", v);
    incogBtn.classList.toggle("on", v);
    incogBtn.style.color = v ? "var(--text)" : "";
    greet.replaceChildren(
      v ? h("span.spark", { style: { width: "34px", height: "34px" } }, icon("ghost")) : spark(),
      h("h1", v ? "You're incognito" : greeting(state.me.name || "Filip")));
    c.el.closest(".composer-slot")?.querySelector(".fine")?.replaceChildren(v ? "Incognito chats aren't saved to your chats and are deleted after an hour." : "");
  }
  incogBtn.onclick = () => setIncog(!incog);
  topBar({ title: projectId ? projectName(projectId) || "Project" : "", right: [incogBtn] });
  const chips = h("div.chips", CHIPS.map(([ico, label, prompt]) => h("button.chip", { type: "button", on: { click: () => c.setText(prompt) } }, icon(ico), label)));
  const arts = h("div.recent-arts", { hidden: true });
  box.append(greet, h("div.composer-slot", c.el, h("p.fine")), chips, arts);
  view.replaceChildren(h("div.scroll.home-scroll", box));
  setIncog(incog);
  c.focus();
  api("artifacts").then((list) => {
    list = list.filter((a) => !projectId || a.projectId === projectId).slice(0, 3);
    if (!list.length) return;
    arts.hidden = false;
    arts.replaceChildren(h("div.head", h("span", "Recent artifacts"), h("a", { href: "#/artifacts" }, "View all")), h("div.art-row", list.map((a) => artifactTile(a))));
  }).catch(() => {});
}

// --- chat thread --------------------------------------------------------------

const TOOL_ICON = { Bash: "terminal", Read: "fileText", Write: "pen", Edit: "edit", MultiEdit: "edit", Glob: "search", Grep: "search", WebFetch: "globe", WebSearch: "globe", Task: "sparkle", Agent: "sparkle", TodoWrite: "check", NotebookEdit: "edit" };
function toolSummary(name, input = {}) {
  if (name === "Bash") return input.description ? `${input.command}` : input.command;
  if (input.file_path) return String(input.file_path).replace(/^\/opt\/plint\//, "");
  if (name === "TodoWrite") return `${(input.todos || []).length} items`;
  return input.pattern || input.path || input.url || input.query || input.description || input.prompt || "";
}
function toolInput(name, input = {}) {
  if (name === "Bash") return `$ ${input.command}`;
  if (name === "Edit") return `${input.file_path}\n\n- ${String(input.old_string || "").split("\n").join("\n- ")}\n+ ${String(input.new_string || "").split("\n").join("\n+ ")}`;
  if (name === "Write") return `${input.file_path}\n\n${String(input.content || "").slice(0, 4000)}`;
  if (name === "TodoWrite") return (input.todos || []).map((t) => `${t.status === "completed" ? "☑" : t.status === "in_progress" ? "◐" : "☐"} ${t.content}`).join("\n");
  return JSON.stringify(input, null, 2);
}

export function renderChat(view, id) {
  state.current = id;
  if (panelChat() && panelChat() !== id) closePanel();
  let chat = null;
  let lastI = -1;
  let es = null;
  let stick = true;
  let turn = null; // current .turn-ai
  let group = null; // current .tools group
  let draft = null; // { node, text }
  let think = null; // { node, text, started }
  const tools = new Map();
  const asks = new Map();
  let lastUserIndex = -1;
  let alive = true;

  const thread = h("div.thread");
  const scroller = h("div.scroll", thread);
  const jump = h("button.jump", { hidden: true, "aria-label": "Scroll to bottom", on: { click: () => { stick = true; scroller.scrollTo({ top: scroller.scrollHeight, behavior: "smooth" }); } } }, icon("arrowDown"));
  const c = composer({
    chat: { id, model: "auto-sonnet", effort: "medium", mode: "manual", style: "normal", projectId: null },
    placeholder: "Reply…",
    onSend: async (text, files, settings) => {
      stick = true;
      scroller.scrollTop = scroller.scrollHeight;
      c.setRunning(true);
      try { await api(`chats/${id}/send`, { body: { text, files, ...settings } }); }
      catch (e) { c.setRunning(false); throw e; }
    },
    onStop: () => api(`chats/${id}/stop`, { body: {} }).catch(() => {}),
  });
  const wrap = h("div.composer-wrap", h("div.inner", jump, c.el, h("p.fine", "Plint Chat runs commands on Plint. Check what it does.")));
  view.replaceChildren(scroller, wrap);
  scroller.addEventListener("scroll", () => {
    stick = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 90;
    jump.hidden = stick;
  }, { passive: true });
  const toBottom = () => { if (stick) scroller.scrollTop = scroller.scrollHeight; };

  function header() {
    const artBtn = h("button.icon-btn", { title: "Artifacts in this chat", "aria-label": "Artifacts in this chat", hidden: !chat?.hasArtifact, on: { click: (e) => chatArtifacts(e.currentTarget) } }, icon("artifacts"));
    topBar({
      title: chat?.title || "",
      onTitle: chat && !chat.incognito ? (e) => chatMenu(e.currentTarget, chat) : null,
      badges: [chat?.incognito ? h("span.pill.incog", icon("ghost"), "Incognito") : null, chat?.projectId ? h("a.pill", { href: `#/project/${chat.projectId}` }, icon("folder"), projectName(chat.projectId) || "Project") : null],
      right: [artBtn],
    });
  }
  async function chatArtifacts(anchor) {
    const list = (await api("artifacts")).filter((a) => a.chatId === id);
    menu(anchor, [{ heading: "Artifacts" }, ...list.map((a) => ({ label: a.title, sub: `${a.kind} · v${a.versions.length}`, icon: "artifacts", onClick: () => openArtifact(a.id) }))], { align: "end" });
  }

  // --- drawing ---
  function working(on) {
    let w = thread.querySelector(".working");
    if (on && !w) { w = h("div.working", spark(), h("span.shimmer", "Thinking…")); thread.append(w); toBottom(); }
    if (!on && w) w.remove();
  }
  function add(node, into) {
    const parent = into || thread;
    const w = parent === thread ? thread.querySelector(".working") : null;
    w ? thread.insertBefore(node, w) : parent.append(node);
    requestAnimationFrame(toBottom);
    return node;
  }
  function aiTurn() {
    if (!turn) turn = add(h("div.turn-ai"));
    return turn;
  }
  function closeGroup() { group = null; }
  function endTurn() {
    if (!turn) return;
    draft?.node.classList.remove("cursor");
    draft = null;
    finishThink();
    for (const g of turn.querySelectorAll(".tools:not(.single)")) if (g.querySelectorAll(".tool").length >= 3) g.classList.remove("open");
    if (!turn.querySelector(".msg-actions")) {
      const t = turn;
      const userIndex = lastUserIndex;
      const prose = () => [...t.querySelectorAll(":scope > .prose")].map((p) => p.dataset.md).join("\n\n");
      t.append(h("div.msg-actions",
        h("button.icon-btn", { title: "Copy", "aria-label": "Copy", on: { click: () => copyText(prose()) } }, icon("copy")),
        h("button.icon-btn.retry", { title: "Retry", "aria-label": "Retry", on: { click: () => retry(userIndex) } }, icon("retry"))));
    }
    // Only the last turn offers Retry.
    const all = thread.querySelectorAll(".turn-ai .retry");
    all.forEach((b, i) => { b.hidden = i !== all.length - 1; });
    turn = null;
    group = null;
  }
  function finishThink() {
    if (!think) return;
    if (think.started && !think.node.dataset.done) {
      const secs = Math.max(1, Math.round((Date.now() - think.started) / 1000));
      think.node.querySelector(".label").textContent = `Thought for ${secs}s`;
    }
    think.node.querySelector(".label").classList.remove("shimmer");
    think.node.dataset.done = "1";
    think = null;
  }
  function thinkBlock(text, live) {
    const node = h("div.think", h("button", { type: "button", on: { click: () => node.classList.toggle("open") } }, icon("chev", "chev"), h("span.label", live ? "Thinking…" : "Thought process")), h("div.body", text));
    if (live) node.querySelector(".label").classList.add("shimmer");
    return node;
  }

  function userTurn(ev) {
    endTurn();
    if (chat?.running) stick = true;
    lastUserIndex = ev.i;
    const files = (ev.files || []).map((f) => f.type?.startsWith("image/") && f.url
      ? h("a", { href: f.url, target: "_blank" }, h("img", { src: f.url, alt: f.name, loading: "lazy" }))
      : h("a.att", { href: f.url || "#", target: "_blank", style: { textDecoration: "none" } }, h("span.ficon", (f.name.split(".").pop() || "").slice(0, 4)), h("div.meta", h("div", f.name), h("div.faint", bytes(f.size || 0)))));
    const bubble = h("div.bubble", ev.text);
    const node = h("div.turn-user", files.length ? h("div.bubble-files", files) : null, bubble,
      h("div.msg-actions",
        h("button.icon-btn", { title: "Copy", "aria-label": "Copy", on: { click: () => copyText(ev.text) } }, icon("copy")),
        chat?.incognito ? null : h("button.icon-btn", { title: "Edit", "aria-label": "Edit", on: { click: () => editUser(node, ev) } }, icon("edit"))));
    add(node);
  }
  function editUser(node, ev) {
    if (chat.running) return toast("Wait for the reply to finish, or stop it.");
    const ta = h("textarea", { rows: 2 }, ev.text);
    const box = h("div.edit-box", ta, h("div.row",
      h("span.note", "Starts a new branch from here"),
      h("button.btn.ghost", { type: "button", on: { click: () => box.replaceWith(node) } }, "Cancel"),
      h("button.btn.primary", { type: "button", on: { click: async () => {
        const text = ta.value.trim();
        if (!text) return;
        try { await api(`chats/${id}/edit`, { body: { index: ev.i, text } }); } catch (e) { toast(e.message); }
      } } }, "Save")));
    node.replaceWith(box);
    ta.style.height = `${Math.min(ta.scrollHeight + 4, 320)}px`;
    ta.focus();
  }
  async function retry(index) {
    if (index < 0) return;
    try { await api(`chats/${id}/edit`, { body: { index } }); } catch (e) { toast(e.message); }
  }

  function handle(ev) {
    if (ev.i !== undefined) { if (ev.i <= lastI) return; lastI = ev.i; }
    switch (ev.t) {
      case "user": userTurn(ev); break;
      case "thinking_delta": {
        if (!think) { closeGroup(); think = { node: add(thinkBlock("", true), aiTurn()), text: "", started: Date.now() }; }
        think.text += ev.text;
        think.node.querySelector(".body").textContent = think.text;
        working(false);
        break;
      }
      case "thinking": {
        if (think) { think.node.querySelector(".body").textContent = ev.text; finishThink(); }
        else { closeGroup(); add(thinkBlock(ev.text, false), aiTurn()); }
        break;
      }
      case "delta": {
        finishThink();
        if (!draft) { closeGroup(); draft = { node: add(h("div.prose.cursor"), aiTurn()), text: "" }; }
        draft.text += ev.text;
        draft.node.innerHTML = markdown(draft.text);
        working(false);
        toBottom();
        break;
      }
      case "text": {
        finishThink();
        const node = draft?.node || add(h("div.prose"), aiTurn());
        node.classList.remove("cursor");
        node.innerHTML = markdown(ev.text);
        node.dataset.md = ev.text;
        draft = null;
        closeGroup();
        if (chat?.running) working(true);
        break;
      }
      case "tool": {
        finishThink();
        if (draft && !draft.text.trim()) draft.node.remove();
        draft = null;
        if (!group) {
          group = add(h("div.tools.single.open", h("button.tools-head", { type: "button", on: { click: (e) => e.currentTarget.parentNode.classList.toggle("open") } }, icon("terminal"), h("span.count"), icon("chev", "chev")), h("div.list")), aiTurn());
        }
        const n = h("div.tool.busy",
          h("button", { type: "button", on: { click: () => n.classList.toggle("open") } },
            h("span.ti", icon(TOOL_ICON[ev.name] || "sparkle")), h("span.name", ev.name), h("span.sum", toolSummary(ev.name, ev.input)), h("span.st")),
          h("div.detail", h("pre", toolInput(ev.name, ev.input))));
        group.querySelector(".list").append(n);
        const count = group.querySelectorAll(".tool").length;
        group.classList.toggle("single", count === 1);
        group.querySelector(".count").textContent = `${count} steps`;
        tools.set(ev.id, n);
        working(true);
        toBottom();
        break;
      }
      case "result": {
        const n = tools.get(ev.id);
        if (!n) break;
        n.classList.remove("busy");
        n.classList.add(ev.error ? "err" : "ok");
        n.querySelector(".st").replaceChildren(icon(ev.error ? "x" : "check"));
        n.querySelector(".detail").append(h("pre.out", ev.output || "(no output)"));
        break;
      }
      case "artifact": {
        closeGroup();
        add(artifactCard(ev), aiTurn());
        if (chat) { chat.hasArtifact = true; header(); }
        if (!narrow() && chat?.running) openArtifact(ev.id, null, { quiet: true });
        break;
      }
      case "ask": {
        closeGroup();
        const n = askCard(ev);
        asks.set(ev.id, n);
        add(n, aiTurn());
        working(false);
        break;
      }
      case "answer": asks.get(ev.id)?.remove(); asks.delete(ev.id); if (chat?.running) working(true); break;
      case "done": if (ev.error) add(h(`div.note${ev.error === "Stopped." ? ".quiet" : ""}`, ev.error), aiTurn()); break;
      case "error": add(h(`div.note${ev.text === "Stopped." ? ".quiet" : ""}`, ev.text), aiTurn()); break;
      case "idle": endTurn(); working(false); setRunning(false); break;
      case "title": if (chat) { chat.title = ev.title; header(); document.title = `${ev.title} · Plint`; } break;
      case "reset": reload(); break;
      case "state": {
        const was = chat?.running;
        chat = { ...chat, ...ev };
        delete chat.t;
        c.setChat(chat);
        setRunning(!!ev.running);
        if (!was && ev.running && !thread.querySelector(".working") && !draft) working(true);
        header();
        document.title = `${chat.title} · Plint`;
        break;
      }
    }
  }
  function setRunning(r) {
    if (chat) chat.running = r;
    c.setRunning(r);
    if (!r) working(false);
  }
  function askCard(ev) {
    const btns = h("div.ask-btns");
    const answer = (allow, always) => async () => {
      btns.querySelectorAll("button").forEach((b) => (b.disabled = true));
      try { await api(`chats/${id}/answer`, { body: { id: ev.id, allow, always } }); }
      catch (e) { btns.querySelectorAll("button").forEach((b) => (b.disabled = false)); toast(e.message); }
    };
    btns.append(
      h("button.btn", { type: "button", on: { click: answer(false) } }, "Deny"),
      h("button.btn", { type: "button", on: { click: answer(true, true) } }, "Always allow"),
      h("button.btn.primary", { type: "button", on: { click: answer(true) } }, "Allow"));
    if (navigator.userActivation?.hasBeenActive) navigator.vibrate?.(25);
    return h("div.ask",
      h("div.ask-head", icon(TOOL_ICON[ev.name] || "shield"), h("span", h("b", ev.name), " needs your OK")),
      h("pre", toolInput(ev.name, ev.input)), btns);
  }

  function connect() {
    es?.close();
    es = new EventSource(`/api/chats/${id}/stream?after=${lastI}`);
    es.onmessage = (m) => { if (alive) handle(JSON.parse(m.data)); };
    es.onerror = () => { if (alive && es.readyState === EventSource.CLOSED) setTimeout(() => alive && connect(), 2000); };
  }
  function reload() {
    es?.close();
    lastI = -1; turn = null; group = null; draft = null; think = null; lastUserIndex = -1;
    tools.clear(); asks.clear();
    thread.replaceChildren();
    stick = true;
    connect();
  }
  const onVisible = () => { if (document.visibilityState === "visible" && alive && (!es || es.readyState === EventSource.CLOSED)) connect(); };
  document.addEventListener("visibilitychange", onVisible);

  header();
  api(`chats/${id}`).then((data) => {
    if (!alive) return;
    chat = data;
    delete chat.events;
    header();
    c.setChat(chat);
    document.title = `${chat.title} · Plint`;
    connect();
  }).catch(() => { toast("That chat is gone."); location.hash = "#/"; });
  c.focus();

  return () => { alive = false; es?.close(); document.removeEventListener("visibilitychange", onVisible); };
}
