// Plint Chat: a Claude-style chat page that drives Claude Code on Plint through
// silk-router. Each chat is a Claude Code session (resumed per message); turns
// keep running when the phone sleeps, and tool calls that need permission wait
// for Allow / Deny on the page. Login is a one-time code sent by ntfy.
import { query } from "@anthropic-ai/claude-agent-sdk";
import { createServer } from "node:http";
import { createHash, randomBytes, randomInt, timingSafeEqual } from "node:crypto";
import { execFile } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const PORT = Number(process.env.PORT || 5220);
const DATA = process.env.DATA_DIR || "/opt/plint/data/plint-chat";
const CWD = process.env.CHAT_CWD || "/opt/plint";
const ROUTER = process.env.ROUTER_URL || "http://127.0.0.1:4141";
const NOTIFY = process.env.NOTIFY_CMD || "/opt/plint/scripts/notify.sh";
const DEV_CODE = process.env.PLINT_CHAT_DEV_CODE || ""; // tests only: fixed login code, no ntfy
const PUBLIC = path.join(path.dirname(new URL(import.meta.url).pathname), "public");
const MODELS = { "auto-sonnet": "Auto Sonnet", "auto-opus": "Auto Opus", "auto-haiku": "Auto Haiku" };
const MAX_RUNNING = 2;
const APPEND = `You are talking to Filip through Plint Chat, a web page he usually opens on his phone.
Keep answers short and lead with the result. Tool calls that need permission are shown to him
with Allow / Deny buttons; if he denies one, ask what he wants instead of retrying it.`;

fs.mkdirSync(path.join(DATA, "chats"), { recursive: true });
const sha = (s) => createHash("sha256").update(s).digest("hex");
const now = () => Date.now();
const log = (...a) => console.log(new Date().toISOString(), ...a);

// --- storage ----------------------------------------------------------------

function readJson(file, fallback) {
  try { return JSON.parse(fs.readFileSync(file, "utf8")); } catch { return fallback; }
}
function writeJson(file, value) {
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(value), { mode: 0o600 });
  fs.renameSync(tmp, file);
}

const authFile = path.join(DATA, "auth.json");
const auth = readJson(authFile, { sessions: {} });
const chatFile = (id) => path.join(DATA, "chats", `${id}.json`);
const chats = new Map();
for (const f of fs.readdirSync(path.join(DATA, "chats"))) {
  if (!f.endsWith(".json")) continue;
  const c = readJson(path.join(DATA, "chats", f), null);
  if (!c) continue;
  if (c.running) { // the service stopped mid-turn
    c.running = false;
    c.events.push({ t: "error", text: "Plint Chat restarted, so this turn stopped. Send it again to carry on." });
  }
  chats.set(c.id, c);
}

const saveTimers = new Map();
function save(chat) {
  clearTimeout(saveTimers.get(chat.id));
  saveTimers.set(chat.id, setTimeout(() => writeJson(chatFile(chat.id), chat), 250));
}
function summary(c) {
  return { id: c.id, title: c.title, model: c.model, updated: c.updated, running: c.running, auto: c.auto };
}

// --- live state: listeners, pending permission asks, running queries --------

const live = new Map(); // id -> { listeners:Set<res>, pending:Map<askId,resolve>, q, abort }
function lv(id) {
  if (!live.has(id)) live.set(id, { listeners: new Set(), pending: new Map(), q: null, abort: null });
  return live.get(id);
}
function send(res, ev, i) {
  res.write(`${i === undefined ? "" : `id: ${i}\n`}data: ${JSON.stringify(ev)}\n\n`);
}
function emit(chat, ev, { store = true } = {}) {
  if (store) {
    chat.events.push(ev);
    chat.updated = now();
    save(chat);
  }
  const i = store ? chat.events.length - 1 : undefined;
  for (const res of lv(chat.id).listeners) send(res, i === undefined ? ev : { ...ev, i }, i);
}
function nobodyWatching(chat) {
  return lv(chat.id).listeners.size === 0;
}
function notify(title, body, prio = "default", tags = "speech_balloon") {
  execFile(NOTIFY, [title, body, prio, tags], { timeout: 15000 }, (err) => err && log("notify failed:", err.message));
}

// --- running a turn -----------------------------------------------------------

const clip = (s, n = 6000) => (s.length > n ? `${s.slice(0, n)}\n… (${s.length - n} more characters)` : s);
function resultText(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content.map((b) => (b.type === "text" ? b.text : b.type === "image" ? "[image]" : "")).join("\n");
}
function askSummary(name, input) {
  if (name === "Bash") return input.command || "";
  return input.file_path || input.path || input.url || input.pattern || JSON.stringify(input).slice(0, 200);
}

async function runTurn(chat, text) {
  const state = lv(chat.id);
  chat.running = true;
  emit(chat, { t: "user", text });
  const abort = new AbortController();
  state.abort = abort;
  state.stopped = false;
  let reported = false;

  const canUseTool = async (name, input, { signal, suggestions }) => {
    if (chat.auto) return { behavior: "allow", updatedInput: input };
    const id = randomBytes(6).toString("hex");
    emit(chat, { t: "ask", id, name, input });
    if (nobodyWatching(chat)) notify("Plint Chat wants to run", `${name}: ${askSummary(name, input)}`.slice(0, 300), "high", "warning");
    const answer = await new Promise((resolve) => {
      state.pending.set(id, resolve);
      signal.addEventListener("abort", () => resolve({ allow: false, aborted: true }), { once: true });
    });
    state.pending.delete(id);
    emit(chat, { t: "answer", id, allow: answer.allow, always: !!answer.always });
    if (answer.allow) {
      return { behavior: "allow", updatedInput: input, ...(answer.always && suggestions ? { updatedPermissions: suggestions } : {}) };
    }
    return { behavior: "deny", message: answer.aborted ? "Stopped." : "Filip declined this tool call." };
  };

  let lastText = "";
  try {
    const q = query({
      prompt: text,
      options: {
        cwd: CWD,
        model: chat.model,
        resume: chat.sessionId || undefined,
        includePartialMessages: true,
        permissionMode: "default",
        settingSources: ["user", "project", "local"],
        systemPrompt: { type: "preset", preset: "claude_code", append: APPEND },
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
          CLAUDE_AGENT_SDK_CLIENT_APP: "plint-chat/1",
        },
        stderr: (d) => log(`[${chat.id}] stderr:`, d.trim().slice(0, 500)),
      },
    });
    state.q = q;
    for await (const m of q) {
      if (m.parent_tool_use_id) continue; // subagent frames: their tool asks still come through canUseTool
      if (m.type === "system" && m.subtype === "init") {
        chat.sessionId = m.session_id;
        save(chat);
      } else if (m.type === "stream_event") {
        const e = m.event;
        if (e.type === "content_block_delta" && e.delta?.type === "text_delta") emit(chat, { t: "delta", text: e.delta.text }, { store: false });
      } else if (m.type === "assistant") {
        for (const b of m.message.content || []) {
          if (b.type === "text" && b.text.trim()) { emit(chat, { t: "text", text: b.text }); lastText = b.text; }
          else if (b.type === "thinking" && b.thinking?.trim()) emit(chat, { t: "thinking", text: clip(b.thinking, 4000) });
          else if (b.type === "tool_use") emit(chat, { t: "tool", id: b.id, name: b.name, input: b.input });
        }
        if (m.error) emit(chat, { t: "error", text: `Model error: ${m.error}` });
      } else if (m.type === "user" && Array.isArray(m.message?.content)) {
        for (const b of m.message.content) {
          if (b.type === "tool_result") emit(chat, { t: "result", id: b.tool_use_id, output: clip(resultText(b.content)), error: !!b.is_error });
        }
      } else if (m.type === "result") {
        const failed = m.subtype !== "success";
        reported = failed;
        const error = !failed ? undefined : state.stopped ? "Stopped." : (m.errors || [m.subtype]).join("; ");
        emit(chat, { t: "done", ms: m.duration_ms, error });
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
    if (nobodyWatching(chat) && lastText) notify(`Plint Chat · ${chat.title}`, lastText.slice(0, 300));
  }
}

function startTurn(chat, text, model) {
  if (chat.running) return "This chat is still working.";
  if ([...chats.values()].filter((c) => c.running).length >= MAX_RUNNING) return "Two chats are already working; wait for one to finish.";
  if (model && MODELS[model]) chat.model = model;
  runTurn(chat, text);
  return null;
}

// --- auth -----------------------------------------------------------------------

const COOKIE = "plint_chat";
const codes = []; // { hash, expires, tries }
const sends = []; // timestamps of codes sent, for rate limiting
function cookies(req) {
  return Object.fromEntries((req.headers.cookie || "").split(/;\s*/).filter(Boolean).map((p) => {
    const i = p.indexOf("=");
    return [p.slice(0, i), decodeURIComponent(p.slice(i + 1))];
  }));
}
function signedIn(req) {
  const t = cookies(req)[COOKIE];
  return !!(t && auth.sessions[sha(t)]);
}
function sendCode() {
  const t = now();
  while (sends.length && sends[0] < t - 3600e3) sends.shift();
  if (sends.length && sends.at(-1) > t - 30e3) return "Wait half a minute before asking for another code.";
  if (sends.length >= 6) return "Too many codes this hour.";
  sends.push(t);
  const code = DEV_CODE || String(randomInt(0, 1e6)).padStart(6, "0");
  codes.push({ hash: sha(code), expires: t + 10 * 60e3, tries: 0 });
  if (!DEV_CODE) notify("Plint Chat login", `Your code: ${code} (valid 10 minutes)`, "high", "key");
  return null;
}
function checkCode(code) {
  const t = now();
  const h = Buffer.from(sha(String(code || "").trim()));
  for (const c of codes) {
    if (c.expires < t || c.tries >= 5) continue;
    c.tries++;
    if (timingSafeEqual(Buffer.from(c.hash), h)) { c.expires = 0; return true; }
  }
  return false;
}

// --- http -------------------------------------------------------------------------

const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml", ".png": "image/png", ".webmanifest": "application/manifest+json" };
function json(res, code, body, headers = {}) {
  res.writeHead(code, { "content-type": "application/json", "cache-control": "no-store", ...headers });
  res.end(JSON.stringify(body));
}
async function body(req) {
  let s = "";
  for await (const c of req) { s += c; if (s.length > 1e6) throw new Error("too big"); }
  return s ? JSON.parse(s) : {};
}
function serveStatic(req, res) {
  let p = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (p === "/" || !path.extname(p)) p = "/index.html";
  const file = path.join(PUBLIC, path.normalize(p));
  if (!file.startsWith(PUBLIC) || !fs.existsSync(file)) return json(res, 404, { error: "not found" });
  res.writeHead(200, { "content-type": TYPES[path.extname(file)] || "application/octet-stream", "cache-control": "no-cache" });
  fs.createReadStream(file).pipe(res);
}

async function api(req, res, url) {
  const parts = url.pathname.split("/").filter(Boolean); // ["api", ...]
  const m = req.method;
  if (m !== "GET" && req.headers["x-plint"] !== "1") return json(res, 403, { error: "missing header" });

  if (parts[1] === "login") {
    const b = await body(req);
    if (parts[2] === "send") {
      const err = sendCode();
      return err ? json(res, 429, { error: err }) : json(res, 200, { ok: true });
    }
    if (!checkCode(b.code)) return json(res, 401, { error: "That code didn't work." });
    const token = randomBytes(32).toString("base64url");
    auth.sessions[sha(token)] = { created: now(), ua: String(req.headers["user-agent"] || "").slice(0, 120) };
    writeJson(authFile, auth);
    return json(res, 200, { ok: true }, { "set-cookie": `${COOKIE}=${token}; Path=/; Max-Age=31536000; HttpOnly; Secure; SameSite=Lax` });
  }
  if (parts[1] === "me") return json(res, 200, { signedIn: signedIn(req), models: MODELS });
  if (!signedIn(req)) return json(res, 401, { error: "sign in" });

  if (parts[1] === "logout") {
    delete auth.sessions[sha(cookies(req)[COOKIE])];
    writeJson(authFile, auth);
    return json(res, 200, { ok: true }, { "set-cookie": `${COOKIE}=; Path=/; Max-Age=0` });
  }
  if (parts[1] !== "chats") return json(res, 404, { error: "not found" });

  if (parts.length === 2) {
    if (m === "GET") return json(res, 200, [...chats.values()].sort((a, b) => b.updated - a.updated).map(summary));
    if (m === "POST") {
      const b = await body(req);
      const text = String(b.text || "").trim();
      if (!text) return json(res, 400, { error: "empty" });
      const chat = { id: randomBytes(8).toString("hex"), title: text.replace(/\s+/g, " ").slice(0, 60), model: MODELS[b.model] ? b.model : "auto-sonnet", auto: false, created: now(), updated: now(), running: false, sessionId: null, events: [] };
      chats.set(chat.id, chat);
      const err = startTurn(chat, text);
      if (err) { chats.delete(chat.id); return json(res, 409, { error: err }); }
      return json(res, 200, summary(chat));
    }
  }
  const chat = chats.get(parts[2]);
  if (!chat) return json(res, 404, { error: "no such chat" });
  const action = parts[3];

  if (!action && m === "GET") return json(res, 200, { ...summary(chat), events: chat.events });
  if (!action && m === "DELETE") {
    if (chat.running) return json(res, 409, { error: "Stop it first." });
    chats.delete(chat.id);
    fs.rmSync(chatFile(chat.id), { force: true });
    return json(res, 200, { ok: true });
  }
  if (action === "stream") {
    const after = Number(url.searchParams.get("after") ?? req.headers["last-event-id"] ?? -1);
    res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache", "x-accel-buffering": "no" });
    chat.events.forEach((ev, i) => { if (i > after) send(res, { ...ev, i }, i); });
    send(res, { t: "state", running: chat.running, auto: chat.auto, model: chat.model });
    const state = lv(chat.id);
    state.listeners.add(res);
    const ping = setInterval(() => res.write(": ping\n\n"), 20000);
    req.on("close", () => { clearInterval(ping); state.listeners.delete(res); });
    return;
  }
  const b = await body(req);
  if (action === "send") {
    const text = String(b.text || "").trim();
    if (!text) return json(res, 400, { error: "empty" });
    const err = startTurn(chat, text, b.model);
    return err ? json(res, 409, { error: err }) : json(res, 200, { ok: true });
  }
  if (action === "stop") {
    const state = lv(chat.id);
    state.stopped = true;
    for (const resolve of state.pending.values()) resolve({ allow: false, aborted: true });
    try { await state.q?.interrupt(); } catch { state.abort?.abort(); }
    return json(res, 200, { ok: true });
  }
  if (action === "answer") {
    const resolve = lv(chat.id).pending.get(b.id);
    if (!resolve) return json(res, 410, { error: "That request is gone." });
    resolve({ allow: !!b.allow, always: !!b.always });
    return json(res, 200, { ok: true });
  }
  if (action === "settings") {
    if (typeof b.auto === "boolean") chat.auto = b.auto;
    if (MODELS[b.model]) chat.model = b.model;
    if (typeof b.title === "string" && b.title.trim()) chat.title = b.title.trim().slice(0, 80);
    save(chat);
    emit(chat, { t: "state", running: chat.running, auto: chat.auto, model: chat.model }, { store: false });
    if (chat.auto) for (const resolve of lv(chat.id).pending.values()) resolve({ allow: true });
    return json(res, 200, summary(chat));
  }
  return json(res, 404, { error: "not found" });
}

createServer(async (req, res) => {
  const url = new URL(req.url, "http://x");
  try {
    if (url.pathname.startsWith("/api/")) return await api(req, res, url);
    return serveStatic(req, res);
  } catch (err) {
    log("request failed:", err?.stack || err);
    if (!res.headersSent) json(res, 500, { error: String(err?.message || err) });
    else res.end();
  }
}).listen(PORT, "127.0.0.1", () => log(`plint-chat on 127.0.0.1:${PORT}, router ${ROUTER}`));

for (const sig of ["SIGTERM", "SIGINT"]) {
  process.on(sig, () => {
    for (const c of chats.values()) writeJson(chatFile(c.id), c);
    process.exit(0);
  });
}
