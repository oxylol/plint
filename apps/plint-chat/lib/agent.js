// Runs Claude Code turns for a chat through the Agent SDK and turns its messages
// into the small events the page draws. Live listeners, pending permission asks
// and running queries live here; everything stored goes through store.js.
import { query } from "@anthropic-ai/claude-agent-sdk";
import fs from "node:fs";
import path from "node:path";
import { randomBytes } from "node:crypto";
import { chats, projects, artifacts, settings, styleById, chatSummary, projectFilesDir, DIRS, newId, now } from "./store.js";
import { notify, log } from "./notify.js";

export const CWD = process.env.CHAT_CWD || "/opt/plint";
export const ROUTER = process.env.ROUTER_URL || "http://127.0.0.1:4141";
export const MODELS = {
  "auto-sonnet": { name: "Auto Sonnet", sub: "DeepSeek V4.1 Flash · everyday" },
  "auto-opus": { name: "Auto Opus", sub: "Kimi K3.1 Pro · deeper, with effort", efforts: true },
  "auto-haiku": { name: "Auto Haiku", sub: "DeepSeek V4.1 Flash · quick" },
};
export const EFFORTS = ["low", "medium", "high"];
const MAX_RUNNING = 3;
const ARTIFACT_EXT = { html: "html", htm: "html", svg: "svg", md: "markdown", markdown: "markdown", mmd: "mermaid", mermaid: "mermaid" };

const BASE_PROMPT = `You are talking to Filip through Plint Chat, a Claude-style web page he usually opens on his phone.
Keep answers short and lead with the result. Tool calls that need permission are shown to him with
Allow / Deny buttons; if he denies one, ask what he wants instead of retrying it.

Artifacts: when Filip asks for something to look at or keep — a page, a small app, a diagram, a chart,
a document, a report — write it as ONE self-contained file in the chat's artifact folder (given below)
with a .html, .svg, .md or .mermaid extension. It opens beside the chat as an artifact; writing there
needs no permission. Update the same file to make a new version. HTML artifacts may load libraries
from cdnjs.cloudflare.com or cdn.jsdelivr.net. Mention the artifact in one line; don't paste its code.`;

// --- live state -----------------------------------------------------------------

const live = new Map(); // chatId -> { listeners, pending, q, abort, stopped }
const globalListeners = new Set();
export function lv(id) {
  if (!live.has(id)) live.set(id, { listeners: new Set(), pending: new Map(), q: null, abort: null, stopped: false });
  return live.get(id);
}
function sse(res, ev, i) {
  res.write(`${i === undefined ? "" : `id: ${i}\n`}data: ${JSON.stringify(ev)}\n\n`);
}
export function emit(chat, ev, { store = true } = {}) {
  let i;
  if (store) {
    chat.events.push(ev);
    chat.updated = now();
    chats.save(chat);
    i = chat.events.length - 1;
  }
  for (const res of lv(chat.id).listeners) sse(res, i === undefined ? ev : { ...ev, i }, i);
}
export function subscribe(chat, req, res, after) {
  res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache", "x-accel-buffering": "no" });
  chat.events.forEach((ev, i) => { if (i > after) sse(res, { ...ev, i }, i); });
  sse(res, { t: "state", ...chatSummary(chat) });
  const state = lv(chat.id);
  state.listeners.add(res);
  const ping = setInterval(() => res.write(": ping\n\n"), 20000);
  req.on("close", () => { clearInterval(ping); state.listeners.delete(res); });
}
export function subscribeAll(req, res) {
  res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache", "x-accel-buffering": "no" });
  res.write(": hi\n\n");
  globalListeners.add(res);
  const ping = setInterval(() => res.write(": ping\n\n"), 20000);
  req.on("close", () => { clearInterval(ping); globalListeners.delete(res); });
}
// Anything the sidebar shows changed (titles, running dots, new chats, artifacts).
let bTimer = null;
export function broadcast(kind = "list") {
  clearTimeout(bTimer);
  bTimer = setTimeout(() => { for (const res of globalListeners) sse(res, { t: kind }); }, 150);
}
const nobodyWatching = (chat) => lv(chat.id).listeners.size === 0;
export const runningCount = () => chats.all().filter((c) => c.running).length;

// --- helpers --------------------------------------------------------------------

const clip = (s, n = 8000) => (s.length > n ? `${s.slice(0, n)}\n… (${s.length - n} more characters)` : s);
function resultText(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content.map((b) => (b.type === "text" ? b.text : b.type === "image" ? "[image]" : "")).join("\n");
}
function askSummary(name, input) {
  if (name === "Bash") return input.command || "";
  return input.file_path || input.path || input.url || input.pattern || JSON.stringify(input).slice(0, 200);
}
export function artifactDir(chat) {
  const d = path.join(DIRS.artifacts, "files", chat.id);
  fs.mkdirSync(d, { recursive: true });
  return d;
}
function inside(file, dir) {
  const rel = path.relative(dir, path.resolve(file || ""));
  return rel && !rel.startsWith("..") && !path.isAbsolute(rel);
}
function artifactKind(file, chat) {
  const ext = path.extname(file || "").slice(1).toLowerCase();
  if (inside(file, artifactDir(chat))) return ARTIFACT_EXT[ext] || "code";
  return ext === "html" || ext === "htm" || ext === "svg" ? ARTIFACT_EXT[ext] : null;
}
function artifactTitle(file, text, kind) {
  const t = kind === "html" ? text.match(/<title>([^<]{1,80})<\/title>/i)?.[1]
    : kind === "markdown" ? text.match(/^#\s+(.{1,80})$/m)?.[1] : null;
  return (t || path.basename(file).replace(/\.[^.]+$/, "").replace(/[-_]+/g, " ")).trim();
}

// Snapshot an artifact file after a successful write: one version per write.
function recordArtifact(chat, file) {
  const kind = artifactKind(file, chat);
  if (!kind) return null;
  let text;
  try {
    if (fs.statSync(file).size > 2e6) return null;
    text = fs.readFileSync(file, "utf8");
  } catch { return null; }
  let a = artifacts.all().find((x) => x.path === path.resolve(file) && x.chatId === chat.id);
  if (!a) a = { id: newId(), chatId: chat.id, projectId: chat.projectId || null, path: path.resolve(file), created: now(), versions: [] };
  if (a.versions.length && fs.existsSync(versionFile(a, a.versions.length)) && fs.readFileSync(versionFile(a, a.versions.length), "utf8") === text) return a;
  a.kind = kind;
  a.title = artifactTitle(file, text, kind);
  a.updated = now();
  a.versions.push({ n: a.versions.length + 1, at: now(), size: text.length });
  fs.mkdirSync(path.dirname(versionFile(a, 1)), { recursive: true });
  fs.writeFileSync(versionFile(a, a.versions.length), text);
  artifacts.put(a);
  chat.hasArtifact = true;
  return a;
}
export function versionFile(a, n) {
  return path.join(DIRS.artifacts, "versions", a.id, `v${n}`);
}

function systemAppend(chat) {
  const parts = [BASE_PROMPT, `Artifact folder for this chat: ${artifactDir(chat)}`];
  const project = chat.projectId && projects.get(chat.projectId);
  if (project) {
    parts.push(`This chat is in the project "${project.name}".${project.description ? ` ${project.description}` : ""}
Project files (reference material Filip uploaded) are in ${projectFilesDir(project.id)}.`);
    if (project.instructions?.trim()) parts.push(`Project instructions from Filip:\n${project.instructions.trim()}`);
  }
  if (settings.instructions?.trim()) parts.push(`Filip's personal preferences:\n${settings.instructions.trim()}`);
  const style = styleById(chat.style);
  if (style.prompt) parts.push(`Response style "${style.name}": ${style.prompt}`);
  return parts.join("\n\n");
}

function promptText(text, files) {
  if (!files?.length) return text;
  const list = files.map((f) => `- ${f.path} (${f.type || "file"}, ${f.size} bytes)`).join("\n");
  return `${text}\n\nFiles Filip attached (saved on Plint, read them as needed):\n${list}`;
}

// --- a turn ------------------------------------------------------------------------

async function runTurn(chat, { text, files = [], fork = null }) {
  const state = lv(chat.id);
  chat.running = true;
  state.stopped = false;
  emit(chat, { t: "user", text, files: files.map(({ name, size, type, url }) => ({ name, size, type, url })), at: now() });
  broadcast();
  const abort = new AbortController();
  state.abort = abort;
  const tools = new Map(); // tool_use id -> { name, input }
  let reported = false;
  let lastText = "";
  const started = now();

  const canUseTool = async (name, input, { signal, suggestions }) => {
    const target = input?.file_path || input?.notebook_path;
    if (["Write", "Edit", "MultiEdit"].includes(name) && target && inside(target, artifactDir(chat))) return { behavior: "allow", updatedInput: input };
    if (chat.mode === "auto") return { behavior: "allow", updatedInput: input };
    const id = randomBytes(6).toString("hex");
    emit(chat, { t: "ask", id, name, input });
    if (nobodyWatching(chat)) notify(`Plint Chat · ${chat.title}`, `Wants to run ${name}: ${askSummary(name, input)}`.slice(0, 300), "high", "warning");
    const answer = await new Promise((resolve) => {
      state.pending.set(id, resolve);
      signal.addEventListener("abort", () => resolve({ allow: false, aborted: true }), { once: true });
    });
    state.pending.delete(id);
    emit(chat, { t: "answer", id, allow: answer.allow, always: !!answer.always });
    if (answer.allow) return { behavior: "allow", updatedInput: input, ...(answer.always && suggestions ? { updatedPermissions: suggestions } : {}) };
    return { behavior: "deny", message: answer.aborted ? "Stopped." : "Filip declined this tool call." };
  };

  const model = MODELS[chat.model]?.efforts && EFFORTS.includes(chat.effort) ? `${chat.model}@${chat.effort}` : chat.model;
  const project = chat.projectId && projects.get(chat.projectId);
  try {
    const q = query({
      prompt: promptText(text, files),
      options: {
        cwd: project?.cwd && fs.existsSync(project.cwd) ? project.cwd : CWD,
        additionalDirectories: [artifactDir(chat), DIRS.uploads, ...(project ? [projectFilesDir(project.id)] : [])],
        model,
        resume: fork ? fork.sessionId || undefined : chat.sessionId || undefined,
        ...(fork?.sessionId && fork.at ? { resumeSessionAt: fork.at, forkSession: true } : {}),
        includePartialMessages: true,
        permissionMode: "default",
        settingSources: ["user", "project", "local"],
        systemPrompt: { type: "preset", preset: "claude_code", append: systemAppend(chat) },
        disallowedTools: ["AskUserQuestion", "EnterPlanMode", "ExitPlanMode"],
        canUseTool,
        abortController: abort,
        env: {
          ...process.env,
          ANTHROPIC_BASE_URL: ROUTER,
          ANTHROPIC_AUTH_TOKEN: process.env.SILK_ROUTER_TOKEN || "silk",
          ANTHROPIC_DEFAULT_OPUS_MODEL: "auto-opus",
          ANTHROPIC_DEFAULT_SONNET_MODEL: "auto-sonnet",
          ANTHROPIC_DEFAULT_HAIKU_MODEL: "auto-haiku",
          CLAUDE_CODE_SUBAGENT_MODEL: "auto-sonnet",
          CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC: "1",
          CLAUDE_AGENT_SDK_CLIENT_APP: "plint-chat/2",
        },
        stderr: (d) => log(`[${chat.id}] stderr:`, d.trim().slice(0, 400)),
      },
    });
    state.q = q;
    for await (const m of q) {
      if (m.parent_tool_use_id) continue; // subagent frames; their asks still reach canUseTool
      if (m.type === "system" && m.subtype === "init") {
        chat.sessionId = m.session_id;
        chats.save(chat);
      } else if (m.type === "stream_event") {
        const e = m.event;
        if (e.type === "content_block_delta" && e.delta?.type === "text_delta") emit(chat, { t: "delta", text: e.delta.text }, { store: false });
        else if (e.type === "content_block_delta" && e.delta?.type === "thinking_delta") emit(chat, { t: "thinking_delta", text: e.delta.thinking }, { store: false });
      } else if (m.type === "assistant") {
        for (const b of m.message.content || []) {
          if (b.type === "text" && b.text.trim()) { emit(chat, { t: "text", text: b.text, uuid: m.uuid }); lastText = b.text; }
          else if (b.type === "thinking" && b.thinking?.trim()) emit(chat, { t: "thinking", text: clip(b.thinking, 6000), uuid: m.uuid });
          else if (b.type === "tool_use") { tools.set(b.id, b); emit(chat, { t: "tool", id: b.id, name: b.name, input: b.input, uuid: m.uuid }); }
        }
        if (m.error) emit(chat, { t: "error", text: `Model error: ${m.error}` });
      } else if (m.type === "user" && Array.isArray(m.message?.content)) {
        for (const b of m.message.content) {
          if (b.type !== "tool_result") continue;
          emit(chat, { t: "result", id: b.tool_use_id, output: clip(resultText(b.content)), error: !!b.is_error, uuid: m.uuid });
          const call = tools.get(b.tool_use_id);
          const file = call?.input?.file_path;
          if (!b.is_error && file && ["Write", "Edit", "MultiEdit"].includes(call.name)) {
            const a = recordArtifact(chat, file);
            if (a) { emit(chat, { t: "artifact", id: a.id, title: a.title, kind: a.kind, version: a.versions.length }); broadcast("artifacts"); }
          }
        }
      } else if (m.type === "result") {
        const failed = m.subtype !== "success";
        reported = failed;
        const error = !failed ? undefined : state.stopped ? "Stopped." : (m.errors || [m.subtype]).join("; ");
        emit(chat, { t: "done", ms: now() - started, cost: m.total_cost_usd, error });
      }
    }
  } catch (err) {
    log(`[${chat.id}] turn failed:`, err?.stack || err);
    if (!reported) emit(chat, { t: "error", text: state.stopped || abort.signal.aborted ? "Stopped." : `Something broke: ${err?.message || err}` });
  } finally {
    chat.running = false;
    state.q = null;
    state.abort = null;
    for (const resolve of state.pending.values()) resolve({ allow: false, aborted: true });
    emit(chat, { t: "idle" });
    chats.save(chat);
    broadcast();
    if (nobodyWatching(chat) && lastText && !chat.incognito) notify(`Plint Chat · ${chat.title}`, lastText.slice(0, 300));
  }
}

export function startTurn(chat, payload) {
  if (chat.running) return "This chat is still working.";
  if (runningCount() >= MAX_RUNNING) return `${MAX_RUNNING} chats are already working; wait for one to finish.`;
  runTurn(chat, payload);
  return null;
}

// Edit or retry: drop everything from the user message at `index` on, branch the
// Claude Code session just before it, and run the (possibly new) text again.
export function editTurn(chat, index, text, files) {
  if (chat.running) return "This chat is still working.";
  const ev = chat.events[index];
  if (!ev || ev.t !== "user") return "That message is gone.";
  let at = null;
  for (let i = index - 1; i >= 0; i--) if (chat.events[i].uuid) { at = chat.events[i].uuid; break; }
  const fork = { sessionId: at ? chat.sessionId : null, at };
  chat.events = chat.events.slice(0, index);
  if (!at) chat.sessionId = null;
  chats.save(chat);
  emit(chat, { t: "reset", length: index }, { store: false });
  if (runningCount() >= MAX_RUNNING) return `${MAX_RUNNING} chats are already working; wait for one to finish.`;
  runTurn(chat, { text: text ?? ev.text, files: files ?? ev.files ?? [], fork });
  return null;
}

export async function stopTurn(chat) {
  const state = lv(chat.id);
  state.stopped = true;
  for (const resolve of state.pending.values()) resolve({ allow: false, aborted: true });
  try { await state.q?.interrupt(); } catch { state.abort?.abort(); }
}

export function answerAsk(chat, id, allow, always) {
  const resolve = lv(chat.id).pending.get(id);
  if (!resolve) return false;
  resolve({ allow, always });
  return true;
}
export function approveAllPending(chat) {
  for (const resolve of lv(chat.id).pending.values()) resolve({ allow: true });
}

// A short title from the first message, by the quick model. Falls back to the text.
export async function autoTitle(chat, text) {
  try {
    const r = await fetch(`${ROUTER}/v1/messages`, {
      method: "POST",
      headers: { "content-type": "application/json", "x-api-key": "silk", "anthropic-version": "2023-06-01" },
      body: JSON.stringify({
        model: "auto-haiku", max_tokens: 24, stream: false,
        messages: [{ role: "user", content: `Write a title of 2 to 6 words for a chat that starts with the message below. Reply with the title only, no quotes or punctuation at the end.\n\n${text.slice(0, 1500)}` }],
      }),
      signal: AbortSignal.timeout(20000),
    });
    if (!r.ok) return;
    const data = await r.json();
    const t = (data.content || []).filter((b) => b.type === "text").map((b) => b.text).join(" ").replace(/["“”*#]/g, "").replace(/\s+/g, " ").trim().replace(/[.:]$/, "");
    if (t && t.length <= 80 && !/sorry|cannot|can't/i.test(t)) {
      chat.title = t;
      chats.save(chat);
      emit(chat, { t: "title", title: t }, { store: false });
      broadcast();
    }
  } catch (err) { log("title failed:", err.message); }
}
