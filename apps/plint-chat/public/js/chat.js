// The composer, the home screen and the chat thread.
import { h, icon, spark, menu, toast, touch, narrow, phone, copyText, bytes, agoLong, when } from "./ui.js";
import { api, upload } from "./api.js";
import { markdown, fadeFresh } from "./md.js";
import { state, prefs, refreshChats, projectName, on } from "./state.js";
import { openArtifact, artifactCard, artifactTile, closePanel, panelChat } from "./artifacts.js";
import { chatMenu } from "./sidebar.js";
import { topBar } from "./main.js";
import { openChanges, openFiles, refreshCodePanel, machineName, baseName, refreshMachines } from "./code.js";

const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
const modelLabel = (id, short) => { const n = state.models[id]?.name || id; return short ? n.replace(/^Auto /, "") : n; };
const cap = (s) => s[0].toUpperCase() + s.slice(1);
const MODE_LABEL = { manual: "Manual", edits: "Accept edits", auto: "Auto" };

// --- composer -----------------------------------------------------------------
// opts: { chat (summary) | null, projectId, placeholder, code, onSend(text, fileIds, settings), onStop }
export function composer(opts) {
  const s = {
    model: opts.chat?.model || prefs.get("model", "auto-sonnet"),
    effort: opts.chat?.effort || prefs.get("effort", "medium"),
    mode: opts.chat?.mode || (opts.code ? prefs.get("codeMode", "edits") : prefs.get("mode", "manual")),
    style: opts.chat?.style || prefs.get("style", "normal"),
    projectId: opts.chat ? opts.chat.projectId : opts.projectId ?? null,
    files: [],
    running: false,
    code: !!opts.code,
  };
  if (!state.models[s.model]) s.model = "auto-sonnet";
  const isNew = !opts.chat;
  const ta = h("textarea", { rows: 1, placeholder: opts.placeholder || "How can I help you today?", "aria-label": "Write your prompt", enterkeyhint: touch ? "enter" : "send" });
  const atts = h("div.attachments", { hidden: true });
  const fileInput = h("input", { type: "file", multiple: true, hidden: true });
  const photoInput = h("input", { type: "file", accept: "image/*", capture: "environment", hidden: true });
  const plusBtn = h("button.icon-btn", { type: "button", "aria-label": "Add files and more", title: "Add files and more", "aria-haspopup": "menu" }, icon("plus"));
  const micBtn = h("button.icon-btn", { type: "button", "aria-label": "Dictate", title: "Dictate" }, icon("mic"));
  const micMore = h("button.icon-btn", { type: "button", "aria-label": "Dictation language", "aria-haspopup": "menu" }, icon("downSmall", "s16"));
  const mic = h("div.split", { role: "group", "aria-label": "Dictate", hidden: !SR }, micBtn, micMore);
  const send = h("button.icon-btn.send", { type: "submit", "aria-label": "Send message", title: "Send message", disabled: true }, icon("up"));
  const stop = h("button.icon-btn.send.stop", { type: "button", "aria-label": "Stop response", title: "Stop response", hidden: true }, icon("stop"));
  const projBtn = h("button.btn", { type: "button", "aria-haspopup": "menu" });
  const modelBtn = h("button.btn.model", { type: "button", "aria-haspopup": "menu" });
  const modeBtn = h("button.btn", { type: "button", "aria-haspopup": "menu" });
  const box = h("div.cbox", atts, ta, h("div.cbar", h("div.l", plusBtn), h("div.r", mic, send, stop)));
  // In a chat the line under the box first shows a short caution, then gives way to the model and mode pickers.
  const noteEl = opts.note ? h("p.chin-note", opts.note) : null;
  const form = h(`form.composer${opts.reply ? ".reply" : ""}${noteEl ? ".noting" : ""}`, box, h("div.chin", h("div.l", opts.chinLeft || projBtn), h("div.r", modelBtn, modeBtn), noteEl), fileInput, photoInput);
  if (noteEl) {
    const reveal = () => form.classList.remove("noting");
    setTimeout(reveal, 4000);
    for (const ev of ["pointerenter", "focusin", "keydown"]) form.addEventListener(ev, () => { form.classList.add("quick"); reveal(); }, { once: true });
  }

  const persist = (patch) => {
    Object.assign(s, patch);
    if (isNew) for (const k of ["model", "effort", "mode", "style"]) if (k in patch) prefs.set(s.code && k === "mode" ? "codeMode" : k, s[k]);
    if (!isNew && opts.chat) api(`chats/${opts.chat.id}/settings`, { body: patch }).catch((e) => toast(e.message));
    paint();
  };
  function paint() {
    const m = state.models[s.model];
    modelBtn.replaceChildren(h("span.t", modelLabel(s.model), m?.efforts ? h("span.sub", cap(s.effort)) : null));
    modelBtn.setAttribute("aria-label", `Model: ${modelLabel(s.model)}${m?.efforts ? ` ${cap(s.effort)}` : ""}`);
    modeBtn.replaceChildren(h("span.t", MODE_LABEL[s.mode] || "Manual"));
    const pname = s.projectId && projectName(s.projectId);
    projBtn.replaceChildren(h("span.t", pname || "Project"), isNew ? icon("down", "s12") : null);
    projBtn.hidden = s.code || (!isNew && !pname);
    projBtn.disabled = !isNew;
    const busy = s.files.some((f) => f.uploading);
    send.disabled = busy || !ta.value.trim();
    send.hidden = s.running;
    stop.hidden = !s.running;
    atts.hidden = !s.files.length;
  }
  function autosize() { ta.style.height = "auto"; ta.style.height = `${Math.min(ta.scrollHeight, 384)}px`; }

  // files
  function drawFiles() {
    atts.replaceChildren(...s.files.map((f) => h(`div.att${f.uploading ? ".loading" : ""}`,
      f.type?.startsWith("image/") && f.preview ? h("img", { src: f.preview, alt: "" }) : h("span.ficon", icon("file")),
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
  box.addEventListener("dragover", (e) => { e.preventDefault(); if (!box.querySelector(".drop-hint")) box.append(h("div.drop-hint", "Drop files here to add to chat")); });
  box.addEventListener("dragleave", (e) => { if (!box.contains(e.relatedTarget)) box.querySelector(".drop-hint")?.remove(); });
  box.addEventListener("drop", (e) => { e.preventDefault(); box.querySelector(".drop-hint")?.remove(); addFiles([...e.dataTransfer.files]); });

  // menus
  plusBtn.onclick = (e) => menu(e.currentTarget, [
    { label: "Add files or photos", icon: "paperclip", onClick: () => fileInput.click() },
    touch ? { label: "Take a photo", icon: "camera", onClick: () => photoInput.click() } : null,
    isNew && !s.code ? { label: "Add to project", icon: "projects", right: s.projectId ? projectName(s.projectId) : "", onClick: () => projBtn.click() } : null,
    "sep",
    { label: "Use style", icon: "styles", right: state.styles.find((st) => st.id === s.style)?.name || "Normal", onClick: () => menu(plusBtn, [
      { heading: "Use style" },
      ...state.styles.map((st) => ({ label: st.name, checked: s.style === st.id, onClick: () => persist({ style: st.id }) })),
      "sep",
      { label: "Create & edit styles", icon: "edit", onClick: () => { location.hash = "#/customize/styles"; } },
    ], { align: "start" }) },
    isNew ? { label: "Incognito chat", icon: "ghost", switch: true, on: state.incognito, onClick: (v) => { state.incognito = v; opts.onIncognito?.(v); } } : null,
  ], { align: "start" });
  projBtn.onclick = (e) => menu(e.currentTarget, [
    { label: "No project", checked: !s.projectId, onClick: () => persist({ projectId: null }) },
    ...state.projects.map((p) => ({ label: p.name, icon: "projects", checked: s.projectId === p.id, onClick: () => persist({ projectId: p.id }) })),
    "sep",
    { label: "New project", icon: "plus", onClick: () => { location.hash = "#/projects/new"; } },
  ], { align: "start" });
  modelBtn.onclick = (e) => menu(e.currentTarget, [
    ...Object.entries(state.models).map(([id, m]) => ({ label: m.name, sub: m.sub, checked: s.model === id, onClick: () => persist({ model: id }) })),
    state.models[s.model]?.efforts ? "sep" : null,
    state.models[s.model]?.efforts ? { label: "Effort", right: cap(s.effort), children: [
      { heading: "Higher effort means more thorough responses, but takes longer.", wrap: true },
      ...state.efforts.map((ef) => ({ label: cap(ef), sub: { low: "Light, casual tasks", medium: "Balanced for everyday work", high: "Complex, detailed work" }[ef], checked: s.effort === ef, onClick: () => persist({ effort: ef }) })),
    ] } : null,
  ], { align: "end", wide: true });
  modeBtn.onclick = (e) => menu(e.currentTarget, [
    { heading: "Permission mode" },
    { label: "Manual", sub: "Plint asks before running commands or changing files.", checked: s.mode === "manual", onClick: () => persist({ mode: "manual" }) },
    { label: "Accept edits", sub: "File edits in the folder go through; other commands still ask.", checked: s.mode === "edits", onClick: () => persist({ mode: "edits" }) },
    { label: "Auto", sub: "Plint runs on its own and never pauses to ask.", checked: s.mode === "auto", onClick: () => persist({ mode: "auto" }) },
  ], { align: "end", gap: 4 });

  // dictation (Web Speech API)
  let rec = null;
  const dictLang = () => prefs.get("dictLang", navigator.language || "en-US");
  micMore.onclick = (e) => {
    const langs = [...new Set([dictLang(), ...(navigator.languages || []), "sv-SE", "en-US"])];
    let names; try { names = new Intl.DisplayNames(["en"], { type: "language" }); } catch {}
    menu(e.currentTarget, [{ heading: "Dictation language" }, ...langs.map((l) => ({ label: names?.of(l) || l, checked: l === dictLang(), onClick: () => prefs.set("dictLang", l) }))], { align: "end" });
  };
  micBtn.onclick = () => {
    if (rec) { rec.stop(); return; }
    rec = new SR();
    rec.lang = dictLang();
    rec.continuous = true;
    rec.interimResults = true;
    const base = ta.value ? `${ta.value.replace(/\s+$/, "")} ` : "";
    rec.onresult = (ev) => {
      let text = "";
      for (const r of ev.results) text += r[0].transcript;
      ta.value = base + text;
      autosize(); paint();
    };
    rec.onend = () => { rec = null; mic.classList.remove("listening"); micBtn.setAttribute("aria-label", "Dictate"); };
    rec.onerror = (ev) => { if (ev.error !== "aborted") toast(ev.error === "not-allowed" ? "Microphone access was blocked" : `Dictation: ${ev.error}`); };
    rec.start();
    mic.classList.add("listening");
    micBtn.setAttribute("aria-label", "Stop dictating");
  };

  // send
  ta.addEventListener("input", () => { autosize(); paint(); });
  ta.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing && !touch && prefs.get("enterSends", true)) { e.preventDefault(); form.requestSubmit(); }
    if (e.key === "Escape" && s.running) opts.onStop?.();
  });
  box.addEventListener("click", (e) => { if (!e.target.closest("button, a, .att")) ta.focus(); });
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
    setChat: (c) => { Object.assign(s, { model: c.model, effort: c.effort, mode: c.mode, style: c.style, projectId: c.projectId, code: c.kind === "code" }); paint(); },
    setNote: (t) => { if (noteEl) noteEl.textContent = t; },
    repaint: paint,
  };
}

// --- home ---------------------------------------------------------------------

function greeting(name) {
  const hr = new Date().getHours();
  const timely = hr < 5 ? "Up late" : hr < 12 ? "Good morning" : hr < 18 ? "Good afternoon" : "Good evening";
  const options = [`${timely}, ${name}`, `Back at it, ${name}`, `${timely}, ${name}`, `How can I help, ${name}?`];
  return options[Math.floor(Date.now() / 3.6e6) % options.length];
}

export { greeting };
export function renderHome(view, { projectId = null } = {}) {
  state.current = null;
  closePanel();
  let incog = state.incognito;
  const incogBtn = h("button.icon-btn", { type: "button", "aria-label": "Use incognito", title: "Use incognito" }, icon("ghost"));
  const box = h("div.home");
  const title = h("h1");
  const fine = h("p.fine");
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
    incogBtn.setAttribute("aria-label", v ? "Leave incognito" : "Use incognito");
    incogBtn.title = v ? "Leave incognito" : "Use incognito";
    title.replaceChildren(v ? h("span.mark", icon("ghost")) : spark(), h("span", v ? "You’re incognito" : greeting(state.me.name || "Filip")));
    fine.textContent = v ? "Incognito chats aren’t saved to your chats and are deleted after an hour." : "";
    more.hidden = v || !more.childElementCount;
  }
  incogBtn.onclick = () => setIncog(!incog);
  topBar({ title: projectId ? projectName(projectId) || "Project" : "", right: [incogBtn] });
  const more = h("div.home-more", { hidden: true });
  box.append(h("div.greet", title), h("div.composer-slot", c.el, fine), more);
  view.replaceChildren(h("div.scroll", box));
  setIncog(incog);
  c.focus();
  api("artifacts").then((list) => {
    list = list.filter((a) => !projectId || a.projectId === projectId).slice(0, 3);
    if (!list.length) return;
    more.replaceChildren(
      h("div.sec-head", h("h3", "Recent artifacts"), h("a.btn", { href: "#/artifacts" }, "View all")),
      h("div.art-row", list.map((a) => artifactTile(a))));
    more.hidden = incog;
  }).catch(() => {});
}

// --- chat thread --------------------------------------------------------------

const TOOL_ICON = { Bash: "terminal", Read: "file", Write: "edit", Edit: "edit", MultiEdit: "edit", Glob: "search", Grep: "search", WebFetch: "globe", WebSearch: "globe", Task: "agent", Agent: "agent", TodoWrite: "tasks", NotebookEdit: "edit" };
// Per tool: [row label, group phrase for one, group phrase for many, phrase while it runs]
const TOOL_WORDS = {
  Bash: ["Ran command", "Ran a command", (n) => `Ran ${n} commands`, "Running a command…"],
  Read: ["Read", "Read a file", (n) => `Read ${n} files`, "Reading a file…"],
  Write: ["Wrote", "Wrote a file", (n) => `Wrote ${n} files`, "Writing a file…"],
  Edit: ["Edited", "Edited a file", (n) => `Made ${n} edits`, "Editing a file…"],
  MultiEdit: ["Edited", "Edited a file", (n) => `Made ${n} edits`, "Editing a file…"],
  NotebookEdit: ["Edited", "Edited a notebook", (n) => `Made ${n} notebook edits`, "Editing a notebook…"],
  Glob: ["Looked for files", "Looked for files", () => "Looked for files", "Looking for files…"],
  Grep: ["Searched", "Searched the code", () => "Searched the code", "Searching the code…"],
  WebSearch: ["Searched the web", "Searched the web", () => "Searched the web", "Searching the web…"],
  WebFetch: ["Read page", "Read a web page", (n) => `Read ${n} web pages`, "Reading a web page…"],
  Task: ["Ran agent", "Ran an agent", (n) => `Ran ${n} agents`, "Working with an agent…"],
  Agent: ["Ran agent", "Ran an agent", (n) => `Ran ${n} agents`, "Working with an agent…"],
  TodoWrite: ["Updated to-dos", "Updated to-dos", () => "Updated to-dos", "Updating to-dos…"],
};
const words = (name) => TOOL_WORDS[name] || [name, `Used ${name}`, (n) => `Used ${name} ${n} times`, `Using ${name}…`];
function toolSubject(name, input = {}) {
  if (name === "Bash") return input.command || "";
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
const lower = (t) => t[0].toLowerCase() + t.slice(1);

// One run of thinking and tool steps inside a reply. Collapsed it is a single quiet line
// ("Thought for 8s", "Ran a command", "Searched the web and read a file"); open, a card of rows.
function statusBlock(flat = false) {
  const steps = []; // { kind: "think" | "tool", name, node, done, error }
  let busy = false, waiting = false;
  let thinkStart = 0, thinkSecs = 0;
  const label = h("span.label");
  const subject = h("span.subject");
  const mark = h("span.mark", spark());
  const card = h("div.steps");
  const node = h(`div.status${flat ? ".flat" : ""}`,
    h("button.status-row", { type: "button", "aria-expanded": "false", on: { click: () => { const open = node.classList.toggle("open"); node.firstChild.setAttribute("aria-expanded", String(open)); } } }, mark, label, subject, icon("downSmall", "s16 chev")),
    h("div.status-panel", card));
  function title() {
    const tools = steps.filter((s) => s.kind === "tool");
    if (waiting) return "Waiting for your approval";
    if (busy) {
      const last = steps.at(-1);
      return !last || last.kind === "think" || last.done ? "Thinking…" : words(last.name)[3];
    }
    if (!tools.length) return thinkSecs ? `Thought for ${thinkSecs}s` : "Thought process";
    const counts = new Map();
    for (const t of tools) counts.set(t.name, (counts.get(t.name) || 0) + 1);
    const parts = [...counts].map(([name, n]) => (n === 1 ? words(name)[1] : words(name)[2](n)));
    if (parts.length > 2) return `${tools.length} steps`;
    return parts.map((p, i) => (i ? lower(p) : p)).join(" and ");
  }
  function paint() {
    node.classList.toggle("busy", busy && !waiting);
    node.classList.toggle("empty", !steps.length);
    const only = flat && steps.length === 1 && steps[0].kind === "tool" ? steps[0] : null;
    label.textContent = only && !waiting ? (busy && !only.done ? words(only.name)[3].replace("…", "") : words(only.name)[0]) + (only.error ? " (failed)" : "") : title();
    subject.textContent = only ? only.subject : "";
    label.classList.toggle("shimmer", busy && !waiting);
  }
  function row(ico, text, subject) {
    const detail = h("div.step-detail");
    const st = h("span.st");
    const n = h("div.step", h("button.step-row", { type: "button", on: { click: () => n.classList.toggle("open") } }, icon(ico, "s16"), h("span.l", text), h("span.s", subject || ""), st), detail);
    card.append(n);
    return { node: n, detail, st };
  }
  const api = {
    node,
    get empty() { return !steps.length; },
    setBusy(b) { if (busy === b) return; busy = b; if (!b) waiting = false; if (!b && thinkStart) api.endThinking(); paint(); },
    setWaiting(w) { waiting = w; paint(); },
    think(text, live) {
      let t = steps.at(-1);
      if (!t || t.kind !== "think" || t.done) {
        const r = row("brain", "Thought", "");
        t = { kind: "think", ...r, done: !live };
        t.detail.append(h("div.think-text"));
        steps.push(t);
        if (live) thinkStart = Date.now();
      }
      t.detail.firstChild.textContent = text;
      if (!live) t.done = true;
      paint();
    },
    endThinking() {
      if (thinkStart) { thinkSecs += Math.max(1, Math.round((Date.now() - thinkStart) / 1000)); thinkStart = 0; }
      const t = steps.at(-1);
      if (t?.kind === "think") t.done = true;
      paint();
    },
    tool(ev) {
      api.endThinking();
      const r = row(TOOL_ICON[ev.name] || "wrench", words(ev.name)[0], toolSubject(ev.name, ev.input));
      r.detail.append(h("pre", toolInput(ev.name, ev.input)));
      const t = { kind: "tool", name: ev.name, subject: toolSubject(ev.name, ev.input), ...r, done: false };
      steps.push(t);
      paint();
      return t;
    },
    result(t, ev) {
      t.done = true;
      t.error = !!ev.error;
      t.node.classList.toggle("err", t.error);
      if (t.error) t.st.replaceChildren(icon("warning", "s16"));
      t.detail.append(h("pre.out", ev.output || "(no output)"));
      paint();
    },
  };
  paint();
  return api;
}

export function renderChat(view, id) {
  state.current = id;
  if (panelChat() && panelChat() !== id) closePanel();
  let chat = null;
  let lastI = -1;
  let es = null;
  let stick = true;
  let turn = null; // current reply: { node, blocks }
  let status = null; // current statusBlock in that reply
  let draft = null; // { node, text, shown }
  const tools = new Map(); // tool id -> [statusBlock, step]
  const asks = new Map();
  let lastUserIndex = -1;
  let alive = true;
  let live = false; // true once history has replayed; only then do things animate in

  const thread = h("div.thread");
  const scroller = h("div.scroll.thread-scroll", h("div.thread-col", thread));
  const jump = h("button.jump", { type: "button", hidden: true, "aria-label": "Scroll to bottom", on: { click: () => { stick = true; scroller.scrollTo({ top: scroller.scrollHeight, behavior: "smooth" }); } } }, icon("arrowDown"));
  const c = composer({
    chat: { id, model: "auto-sonnet", effort: "medium", mode: "manual", style: "normal", projectId: null },
    placeholder: "Reply",
    reply: true,
    note: phone() ? "Plint is AI and can make mistakes." : "Plint is AI and can make mistakes. Please double-check what it runs.",
    onSend: async (text, files, settings) => {
      stick = true;
      scroller.scrollTop = scroller.scrollHeight;
      c.setRunning(true);
      try { await api(`chats/${id}/send`, { body: { text, files, ...settings } }); }
      catch (e) { c.setRunning(false); throw e; }
    },
    onStop: () => api(`chats/${id}/stop`, { body: {} }).catch(() => {}),
  });
  const askBox = h("div.asks");
  const dock = h("div.dock", h("div.dock-in", jump, askBox, c.el));
  view.replaceChildren(scroller, dock);
  scroller.addEventListener("scroll", () => {
    stick = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 90;
    jump.hidden = stick;
  }, { passive: true });
  const toBottom = () => { if (stick) scroller.scrollTop = scroller.scrollHeight; };

  const rename = async (title) => { await api(`chats/${id}/settings`, { body: { title } }); refreshChats(); };
  function header() {
    if (chat?.kind === "code") {
      const gitPill = h("span.pill", { hidden: true });
      if (chat.cwd) api(`machines/${chat.machine}/git?path=${encodeURIComponent(chat.cwd)}`).then((g) => { if (g.git) { gitPill.hidden = false; gitPill.replaceChildren(icon("branch"), g.branch || "detached"); } }).catch(() => {});
      topBar({
        title: chat.title,
        onRename: rename,
        onTitle: (e) => chatMenu(e.currentTarget, chat, { header: true }),
        badges: [narrow() ? null : h("span.pill", { title: `${machineName(chat.machine)}: ${chat.cwd}` }, icon("folder"), `${machineName(chat.machine)} · ${baseName(chat.cwd)}`), narrow() ? null : gitPill],
        right: [
          h("button.icon-btn", { type: "button", title: "Changes", "aria-label": "Changes", on: { click: () => openChanges(chat) } }, icon("changes")),
          h("button.icon-btn", { type: "button", title: "Files", "aria-label": "Files", on: { click: () => openFiles(chat) } }, icon("folder")),
        ],
      });
      return;
    }
    const artBtn = h("button.icon-btn", { type: "button", title: "Artifacts in this chat", "aria-label": "Artifacts in this chat", hidden: !chat?.hasArtifact, on: { click: (e) => chatArtifacts(e.currentTarget) } }, icon("artifacts"));
    topBar({
      title: chat ? chat.title || "Untitled" : "",
      onRename: chat && !chat.incognito ? rename : null,
      onTitle: chat && !chat.incognito ? (e) => chatMenu(e.currentTarget, chat, { header: true }) : null,
      badges: [chat?.incognito ? h("span.pill.incog", icon("ghost"), "Incognito") : null, chat?.projectId ? h("a.pill", { href: `#/project/${chat.projectId}` }, icon("projects"), projectName(chat.projectId) || "Project") : null],
      right: [artBtn],
    });
  }
  async function chatArtifacts(anchor) {
    const list = (await api("artifacts")).filter((a) => a.chatId === id);
    menu(anchor, [{ heading: "Artifacts" }, ...list.map((a) => ({ label: a.title, right: a.versions.length > 1 ? `v${a.versions.length}` : "", icon: "artifacts", onClick: () => openArtifact(a.id) }))], { align: "end" });
  }

  // --- drawing ---
  function add(node, into = thread) {
    into.append(node);
    requestAnimationFrame(toBottom);
    return node;
  }
  function aiTurn() {
    if (!turn) {
      const blocks = h("div.blocks");
      turn = { node: add(h(`div.turn-ai${live ? ".enter" : ""}`, blocks)), blocks };
    }
    return turn.blocks;
  }
  // The status line this reply is currently adding steps to; a new one starts after any text or card.
  function curStatus() {
    if (!status) { status = statusBlock(chat?.kind === "code"); add(status.node, aiTurn()); }
    return status;
  }
  function closeStatus() {
    if (!status) return;
    status.setBusy(false);
    if (status.empty) status.node.remove();
    status = null;
  }
  function working(on) {
    if (on) curStatus().setBusy(true);
    else if (status) { status.setBusy(false); if (status.empty) { status.node.remove(); status = null; } }
  }
  const timeEl = (ts) => (ts ? h("time", { datetime: new Date(ts).toISOString(), title: when(ts) }, agoLong(ts)) : null);
  const act = (ico, text, onClick, cls = "") => h(`button.icon-btn.sm${cls}`, { type: "button", title: text, "aria-label": text, on: { click: onClick } }, icon(ico, "s16"));
  const copyAct = (get) => { const b = act("copy", "Copy", async () => { await copyText(get(), { quiet: true }); b.replaceChildren(icon("check", "s16")); setTimeout(() => b.replaceChildren(icon("copy", "s16")), 1600); }); return b; };
  function endTurn(doneAt) {
    if (!turn) return;
    draft = null;
    closeStatus();
    if (!turn.node.querySelector(":scope > .msg-actions") && turn.blocks.childElementCount) {
      const t = turn;
      const userIndex = lastUserIndex;
      const prose = () => [...t.blocks.querySelectorAll(":scope > .prose")].map((p) => p.dataset.md).join("\n\n");
      const speak = act("speaker", "Read aloud", () => {
        if (speechSynthesis.speaking) { speechSynthesis.cancel(); return; }
        const u = new SpeechSynthesisUtterance([...t.blocks.querySelectorAll(":scope > .prose")].map((p) => p.innerText).join("\n"));
        u.onend = u.onerror = () => speak.replaceChildren(icon("speaker", "s16"));
        speak.replaceChildren(icon("pause", "s16"));
        speechSynthesis.speak(u);
      });
      t.node.append(h("div.msg-actions",
        prose() ? copyAct(prose) : null,
        prose() && "speechSynthesis" in window ? speak : null,
        chat?.incognito || userIndex < 0 ? null : act("retry", "Retry", () => retry(userIndex)),
        timeEl(doneAt)));
    }
    thread.querySelectorAll(".turn-ai.latest").forEach((n) => n.classList.remove("latest"));
    turn.node.classList.add("latest");
    turn = null;
  }

  function userTurn(ev) {
    endTurn();
    if (chat?.running) stick = true;
    lastUserIndex = ev.i;
    const files = (ev.files || []).map((f) => f.type?.startsWith("image/") && f.url
      ? h("a.msg-img", { href: f.url, target: "_blank" }, h("img", { src: f.url, alt: f.name, loading: "lazy" }))
      : h("a.att", { href: f.url || "#", target: "_blank" }, h("span.ficon", icon("file")), h("div.meta", h("div", f.name), h("div.faint", bytes(f.size || 0)))));
    const body = h("div.bubble-text", ev.text);
    const bubble = h("div.bubble", body);
    const node = h(`div.turn-user${live ? ".enter" : ""}`, files.length ? h("div.msg-files", files) : null, bubble,
      h("div.msg-actions",
        timeEl(ev.at),
        chat?.incognito ? null : act("retry", "Retry", () => retry(ev.i)),
        chat?.incognito ? null : act("edit", "Edit", () => editUser(node, bubble, ev)),
        copyAct(() => ev.text)));
    add(node);
    // Long messages fold to a few lines with "Show more".
    requestAnimationFrame(() => {
      if (body.scrollHeight <= 300) return;
      bubble.classList.add("folded");
      const more = h("button.btn.xs", { type: "button", on: { click: () => { const f = bubble.classList.toggle("folded"); more.textContent = f ? "Show more" : "Show less"; } } }, "Show more");
      bubble.append(more);
    });
  }
  function editUser(node, bubble, ev) {
    if (chat.running) return toast("Wait for the reply to finish, or stop it.");
    const ta = h("textarea", { rows: 1, "aria-label": "Edit message" }, ev.text);
    const size = () => { ta.style.height = "auto"; ta.style.height = `${ta.scrollHeight}px`; };
    const cancel = () => { box.replaceWith(bubble); foot.remove(); node.classList.remove("editing"); };
    const save = async () => {
      const text = ta.value.trim();
      if (!text || text === ev.text) return cancel();
      try { await api(`chats/${id}/edit`, { body: { index: ev.i, text } }); } catch (e) { toast(e.message); }
    };
    const box = h("div.bubble.edit", ta);
    const foot = h("div.edit-foot",
      h("button.btn.xs", { type: "button", on: { click: cancel } }, "Cancel"),
      h("button.btn.xs.primary", { type: "button", on: { click: save } }, "Save"));
    ta.addEventListener("input", size);
    ta.addEventListener("keydown", (e) => {
      if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); cancel(); }
      if (e.key === "Enter" && !e.shiftKey && !e.altKey && !e.isComposing && !touch) { e.preventDefault(); save(); }
    });
    node.classList.add("editing");
    bubble.replaceWith(box);
    box.after(foot);
    size();
    ta.focus();
    ta.setSelectionRange(ta.value.length, ta.value.length);
  }
  async function retry(index) {
    if (index < 0) return;
    if (chat?.running) return toast("Wait for the reply to finish, or stop it.");
    try { await api(`chats/${id}/edit`, { body: { index } }); } catch (e) { toast(e.message); }
  }
  const note = (text) => h(`div.note${text === "Stopped." ? ".quiet" : ""}`, text === "Stopped." ? null : icon("warning", "s16"), h("span", text === "Stopped." ? "Response stopped" : text));

  function handle(ev) {
    if (ev.i !== undefined) { if (ev.i <= lastI) return; lastI = ev.i; }
    switch (ev.t) {
      case "user": userTurn(ev); break;
      case "thinking_delta": {
        const st = curStatus();
        st._think = (st._think || "") + ev.text;
        st.think(st._think, true);
        st.setBusy(true);
        break;
      }
      case "thinking": {
        const st = curStatus();
        st.think(ev.text, false);
        st.endThinking();
        st._think = "";
        break;
      }
      case "delta": {
        closeStatus();
        if (!draft) draft = { node: add(h("div.prose"), aiTurn()), text: "", shown: 0 };
        draft.text += ev.text;
        draft.node.innerHTML = markdown(draft.text);
        draft.shown = fadeFresh(draft.node, draft.shown);
        toBottom();
        break;
      }
      case "text": {
        closeStatus();
        const node = draft?.node || add(h("div.prose"), aiTurn());
        node.innerHTML = markdown(ev.text);
        node.dataset.md = ev.text;
        draft = null;
        if (chat?.running && live) working(true);
        break;
      }
      case "tool": {
        if (draft && !draft.text.trim()) draft.node.remove();
        draft = null;
        if (chat?.kind === "code") closeStatus(); // in Code every step is its own line
        const st = curStatus();
        tools.set(ev.id, [st, st.tool(ev)]);
        if (chat?.running) st.setBusy(true);
        toBottom();
        break;
      }
      case "result": {
        const hit = tools.get(ev.id);
        if (hit) hit[0].result(hit[1], ev);
        break;
      }
      case "artifact": {
        closeStatus();
        add(artifactCard(ev), aiTurn());
        if (chat) { chat.hasArtifact = true; header(); }
        if (!narrow() && chat?.running && live) openArtifact(ev.id, null, { quiet: true });
        break;
      }
      case "ask": {
        const n = askCard(ev);
        asks.set(ev.id, n);
        askBox.append(n);
        requestAnimationFrame(toBottom);
        status?.setWaiting(true);
        break;
      }
      case "answer": asks.get(ev.id)?.remove(); asks.delete(ev.id); status?.setWaiting(false); break;
      case "done": closeStatus(); if (ev.error) add(note(ev.error), aiTurn()); endTurn(ev.at); break;
      case "error": closeStatus(); add(note(ev.text), aiTurn()); break;
      case "idle": for (const n of asks.values()) n.remove(); asks.clear(); endTurn(); setRunning(false); if (chat?.kind === "code") refreshCodePanel(chat); break;
      case "changed": if (chat?.kind === "code") refreshCodePanel(chat); break;
      case "title": if (chat) { chat.title = ev.title; header(); document.title = `${ev.title} · Plint`; } break;
      case "reset": reload(); break;
      case "state": {
        // Sent after the stored events have replayed: from here on things are live.
        const was = chat?.running;
        chat = { ...chat, ...ev };
        delete chat.t;
        c.setChat(chat);
        setRunning(!!ev.running);
        if (ev.running && (!was || !live) && !draft && !asks.size) working(true);
        if (!ev.running) endTurn();
        live = true;
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
  // A request to run something waits above the composer: Deny on the left, the two ways to allow on the right.
  function askCard(ev) {
    const shown = Date.now();
    const answer = (allow, always) => async () => {
      if (n.dataset.sent) return;
      n.dataset.sent = "1";
      n.querySelectorAll("button").forEach((b) => (b.disabled = true));
      try { await api(`chats/${id}/answer`, { body: { id: ev.id, allow, always } }); }
      catch (e) { delete n.dataset.sent; n.querySelectorAll("button").forEach((b) => (b.disabled = false)); toast(e.message); }
    };
    if (navigator.userActivation?.hasBeenActive) navigator.vibrate?.(25);
    const hint = (t) => (touch ? null : h("span.key", t));
    const n = h("div.ask", { role: "group", "aria-label": `Permission request: ${ev.name}` },
      h("p.ask-title", `Plint wants to use ${ev.name}`),
      h("pre.ask-detail", toolInput(ev.name, ev.input)),
      h("div.ask-row",
        h("button.btn.secondary", { type: "button", on: { click: answer(false) } }, "Deny", hint("Esc")),
        h("span.grow"),
        h("button.btn.secondary", { type: "button", on: { click: answer(true, true) } }, "Always allow"),
        h("button.btn.primary", { type: "button", on: { click: answer(true) } }, "Allow once", hint(/Mac|iPhone|iPad/.test(navigator.platform) ? "⌘⏎" : "Ctrl+⏎"))));
    n.deny = answer(false); n.always = answer(true, true); n.once = answer(true); n.armed = () => Date.now() - shown > 400;
    return n;
  }
  // Esc denies and Ctrl/⌘+Enter allows the request on top; 1, 2, 3 pick a button when not typing.
  const onAskKey = (e) => {
    const n = askBox.firstElementChild;
    if (!alive || !n || document.querySelector(".menu, .modal-back")) return;
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || "");
    let act = null;
    if (e.key === "Escape") act = n.deny;
    else if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && n.armed()) act = n.once;
    else if (!typing && !e.metaKey && !e.ctrlKey && !e.altKey) act = { 1: n.deny, 2: n.always, 3: n.once }[e.key];
    if (!act) return;
    e.preventDefault(); e.stopPropagation();
    act();
  };
  document.addEventListener("keydown", onAskKey, true);

  function connect() {
    es?.close();
    es = new EventSource(`/api/chats/${id}/stream?after=${lastI}`);
    es.onmessage = (m) => { if (alive) handle(JSON.parse(m.data)); };
    es.onerror = () => { if (alive && es.readyState === EventSource.CLOSED) setTimeout(() => alive && connect(), 2000); };
  }
  function reload() {
    es?.close();
    lastI = -1; turn = null; status = null; draft = null; lastUserIndex = -1; live = false;
    tools.clear(); asks.clear();
    thread.replaceChildren(); askBox.replaceChildren();
    stick = true;
    connect();
  }
  const onVisible = () => { if (document.visibilityState === "visible" && alive && (!es || es.readyState === EventSource.CLOSED)) connect(); };
  document.addEventListener("visibilitychange", onVisible);
  const tick = setInterval(() => { for (const t of thread.querySelectorAll("time[datetime]")) t.textContent = agoLong(Date.parse(t.dateTime)); }, 60000);

  header();
  api(`chats/${id}`).then((data) => {
    if (!alive) return;
    chat = data;
    delete chat.events;
    thread.classList.toggle("code", chat.kind === "code");
    if (chat.kind === "code") { refreshMachines().then(header); }
    header();
    c.setChat(chat);
    document.title = `${chat.title} · Plint`;
    connect();
  }).catch(() => { toast("That chat is gone."); location.hash = "#/"; });
  c.focus();

  return () => { alive = false; document.removeEventListener("keydown", onAskKey, true); es?.close(); clearInterval(tick); speechSynthesis?.cancel?.(); document.removeEventListener("visibilitychange", onVisible); };
}
