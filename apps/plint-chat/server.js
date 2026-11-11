// Plint Chat: a claude.ai-style page that drives Claude Code on Plint through
// silk-router. HTTP routes only; the work lives in lib/.
import { createServer } from "node:http";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { chats, projects, artifacts, settings, saveSettings, allStyles, schedules, saveSchedules, chatSummary,
  projectFilesDir, listFiles, safeName, DIRS, newId, now } from "./lib/store.js";
import { signedIn, sendCode, login, logout } from "./lib/auth.js";
import { MODELS, EFFORTS, CWD, ROUTER, emit, subscribe, subscribeAll, broadcast, startTurn, editTurn, stopTurn,
  answerAsk, approveAllPending, autoTitle, versionFile, lv } from "./lib/agent.js";
import { KINDS, nextRun, describe, runSchedule, startScheduler } from "./lib/scheduler.js";
import { log } from "./lib/notify.js";

const PORT = Number(process.env.PORT || 5220);
const PUBLIC = path.join(path.dirname(new URL(import.meta.url).pathname), "public");
const MAX_UPLOAD = 25e6;

// A service restart stops running turns; say so in those chats.
for (const c of chats.all()) {
  if (!c.running) continue;
  c.running = false;
  c.events.push({ t: "error", text: "Plint Chat restarted, so this turn stopped. Send it again to carry on." }, { t: "idle" });
  chats.save(c);
}

// --- http helpers ------------------------------------------------------------------

const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp",
  ".webmanifest": "application/manifest+json", ".json": "application/json", ".pdf": "application/pdf", ".txt": "text/plain; charset=utf-8" };
function json(res, code, body, headers = {}) {
  res.writeHead(code, { "content-type": "application/json", "cache-control": "no-store", ...headers });
  res.end(JSON.stringify(body));
}
async function readBody(req, limit = 2e6) {
  const chunks = [];
  let n = 0;
  for await (const c of req) { n += c.length; if (n > limit) throw Object.assign(new Error("Too big."), { status: 413 }); chunks.push(c); }
  return Buffer.concat(chunks);
}
async function body(req) {
  const b = await readBody(req);
  return b.length ? JSON.parse(b.toString("utf8")) : {};
}
function sendFile(res, file, { download = false, name } = {}) {
  if (!fs.existsSync(file) || !fs.statSync(file).isFile()) return json(res, 404, { error: "not found" });
  const ext = path.extname(name || file).toLowerCase();
  // Never let an uploaded or generated file run as a page on this origin.
  const type = TYPES[ext] && !/html|svg/.test(TYPES[ext]) ? TYPES[ext] : "application/octet-stream";
  res.writeHead(200, {
    "content-type": download ? "application/octet-stream" : type,
    "content-disposition": `${download || type === "application/octet-stream" ? "attachment" : "inline"}; filename*=UTF-8''${encodeURIComponent(name || path.basename(file))}`,
    "x-content-type-options": "nosniff", "content-security-policy": "sandbox", "cache-control": "private, max-age=3600",
  });
  fs.createReadStream(file).pipe(res);
}
function serveStatic(req, res) {
  let p = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (p === "/" || !path.extname(p)) p = "/index.html";
  const file = path.join(PUBLIC, path.normalize(p));
  if (!file.startsWith(PUBLIC) || !fs.existsSync(file)) return json(res, 404, { error: "not found" });
  res.writeHead(200, { "content-type": TYPES[path.extname(file)] || "application/octet-stream", "cache-control": "no-cache" });
  fs.createReadStream(file).pipe(res);
}
const pick = (obj, keys) => Object.fromEntries(keys.filter((k) => obj[k] !== undefined).map((k) => [k, obj[k]]));

// Apply the settings a chat can carry (from the composer or the chat menu).
function applyChatSettings(chat, b) {
  if (MODELS[b.model]) chat.model = b.model;
  if (EFFORTS.includes(b.effort)) chat.effort = b.effort;
  if (b.mode === "auto" || b.mode === "manual") chat.mode = b.mode;
  if (typeof b.style === "string" && allStyles().some((s) => s.id === b.style)) chat.style = b.style;
  if (typeof b.title === "string" && b.title.trim()) chat.title = b.title.trim().slice(0, 100);
  if (typeof b.pinned === "boolean") chat.pinned = b.pinned;
  if (b.projectId === null || (typeof b.projectId === "string" && projects.get(b.projectId))) chat.projectId = b.projectId;
}
function uploadsFor(ids) {
  return (Array.isArray(ids) ? ids : []).map((id) => {
    const dir = path.join(DIRS.uploads, String(id).replace(/[^a-f0-9]/g, ""));
    const meta = fs.existsSync(path.join(dir, "meta.json")) ? JSON.parse(fs.readFileSync(path.join(dir, "meta.json"), "utf8")) : null;
    return meta && { ...meta, path: path.join(dir, meta.name), url: `/api/uploads/${meta.id}/${encodeURIComponent(meta.name)}` };
  }).filter(Boolean);
}
function skills() {
  const out = [];
  for (const root of [path.join(os.homedir(), ".claude", "skills"), path.join(CWD, ".claude", "skills")]) {
    for (const d of fs.existsSync(root) ? fs.readdirSync(root) : []) {
      const f = path.join(root, d, "SKILL.md");
      if (!fs.existsSync(f)) continue;
      const head = fs.readFileSync(f, "utf8").slice(0, 3000);
      out.push({ name: head.match(/^name:\s*(.+)$/m)?.[1]?.trim() || d, description: head.match(/^description:\s*(.+)$/m)?.[1]?.trim().slice(0, 300) || "", where: root.startsWith(CWD) ? "project" : "personal" });
    }
  }
  return out;
}
function search(q) {
  const needle = q.toLowerCase();
  const out = [];
  for (const c of chats.all()) {
    if (c.incognito) continue;
    let snippet = null;
    if (c.title.toLowerCase().includes(needle)) snippet = "";
    for (const e of c.events) {
      if (snippet !== null) break;
      if ((e.t === "user" || e.t === "text") && e.text.toLowerCase().includes(needle)) {
        const i = e.text.toLowerCase().indexOf(needle);
        snippet = `${i > 40 ? "…" : ""}${e.text.slice(Math.max(0, i - 40), i + 80).replace(/\s+/g, " ")}`;
      }
    }
    if (snippet !== null) out.push({ ...chatSummary(c), snippet });
  }
  return out.sort((a, b) => b.updated - a.updated).slice(0, 40);
}

// --- api ---------------------------------------------------------------------------

async function api(req, res, url) {
  const parts = url.pathname.split("/").filter(Boolean).slice(1).map(decodeURIComponent);
  const [top, id, action, sub] = parts;
  const m = req.method;
  if (m !== "GET" && req.headers["x-plint"] !== "1") return json(res, 403, { error: "missing header" });

  if (top === "login") {
    if (id === "send") { const err = sendCode(); return err ? json(res, 429, { error: err }) : json(res, 200, { ok: true }); }
    const b = await body(req);
    const cookie = login(b.code, req.headers["user-agent"]);
    return cookie ? json(res, 200, { ok: true }, { "set-cookie": cookie }) : json(res, 401, { error: "That code didn't work." });
  }
  if (top === "me") {
    return json(res, 200, { signedIn: signedIn(req), name: settings.name, models: MODELS, efforts: EFFORTS, styles: signedIn(req) ? allStyles() : [] });
  }
  if (!signedIn(req)) return json(res, 401, { error: "sign in" });
  if (top === "logout") return json(res, 200, { ok: true }, { "set-cookie": logout(req) });
  if (top === "events") return subscribeAll(req, res);
  if (top === "search") return json(res, 200, search(url.searchParams.get("q") || ""));

  // uploads: raw body, name in x-filename
  if (top === "uploads") {
    if (m === "POST" && !id) {
      const data = await readBody(req, MAX_UPLOAD);
      const uid = newId();
      const name = safeName(decodeURIComponent(req.headers["x-filename"] || "file"));
      const dir = path.join(DIRS.uploads, uid);
      fs.mkdirSync(dir, { recursive: true });
      fs.writeFileSync(path.join(dir, name), data);
      const meta = { id: uid, name, size: data.length, type: String(req.headers["content-type"] || "application/octet-stream").slice(0, 100) };
      fs.writeFileSync(path.join(dir, "meta.json"), JSON.stringify(meta));
      return json(res, 200, { ...meta, url: `/api/uploads/${uid}/${encodeURIComponent(name)}` });
    }
    if (m === "GET" && id && action) return sendFile(res, path.join(DIRS.uploads, id.replace(/[^a-f0-9]/g, ""), safeName(action)));
  }

  if (top === "settings") {
    if (m === "GET") return json(res, 200, { ...settings, allStyles: allStyles() });
    const b = await body(req);
    if (typeof b.instructions === "string") settings.instructions = b.instructions.slice(0, 20000);
    if (typeof b.name === "string" && b.name.trim()) settings.name = b.name.trim().slice(0, 40);
    if (Array.isArray(b.styles)) settings.styles = b.styles.filter((s) => s?.name && s?.prompt).slice(0, 20).map((s) => ({ id: s.id || newId(4), name: String(s.name).slice(0, 40), prompt: String(s.prompt).slice(0, 4000) }));
    if (typeof b.defaultStyle === "string") settings.defaultStyle = b.defaultStyle;
    saveSettings();
    return json(res, 200, { ...settings, allStyles: allStyles() });
  }
  if (top === "skills") return json(res, 200, skills());

  // --- chats ---
  if (top === "chats") {
    if (!id) {
      if (m === "GET") {
        const pid = url.searchParams.get("project");
        return json(res, 200, chats.all().filter((c) => !c.incognito && (!pid || c.projectId === pid)).sort((a, b) => b.updated - a.updated).map(chatSummary));
      }
      const b = await body(req);
      const text = String(b.text || "").trim();
      if (!text) return json(res, 400, { error: "empty" });
      const chat = { id: newId(), title: text.replace(/\s+/g, " ").slice(0, 60), model: "auto-sonnet", effort: "low", mode: "manual",
        style: settings.defaultStyle || "normal", projectId: null, pinned: false, incognito: !!b.incognito,
        created: now(), updated: now(), running: false, sessionId: null, events: [] };
      applyChatSettings(chat, b);
      chats.put(chat);
      const err = startTurn(chat, { text, files: uploadsFor(b.files) });
      if (err) { chats.delete(chat.id); return json(res, 409, { error: err }); }
      if (!chat.incognito) autoTitle(chat, text);
      return json(res, 200, chatSummary(chat));
    }
    const chat = chats.get(id);
    if (!chat) return json(res, 404, { error: "No such chat." });
    if (!action) {
      if (m === "GET") return json(res, 200, { ...chatSummary(chat), events: chat.events });
      if (m === "DELETE") {
        if (chat.running) await stopTurn(chat);
        chats.delete(chat.id);
        broadcast();
        return json(res, 200, { ok: true });
      }
    }
    if (action === "stream") return subscribe(chat, req, res, Number(url.searchParams.get("after") ?? -1));
    const b = await body(req);
    if (action === "send") {
      const text = String(b.text || "").trim();
      if (!text) return json(res, 400, { error: "empty" });
      applyChatSettings(chat, b);
      const err = startTurn(chat, { text, files: uploadsFor(b.files) });
      return err ? json(res, 409, { error: err }) : json(res, 200, { ok: true });
    }
    if (action === "edit") {
      applyChatSettings(chat, b);
      const err = editTurn(chat, Number(b.index), typeof b.text === "string" && b.text.trim() ? b.text.trim() : undefined);
      return err ? json(res, 409, { error: err }) : json(res, 200, { ok: true });
    }
    if (action === "stop") { await stopTurn(chat); return json(res, 200, { ok: true }); }
    if (action === "answer") return answerAsk(chat, b.id, !!b.allow, !!b.always) ? json(res, 200, { ok: true }) : json(res, 410, { error: "That request is gone." });
    if (action === "settings") {
      applyChatSettings(chat, b);
      chats.save(chat);
      emit(chat, { t: "state", ...chatSummary(chat) }, { store: false });
      if (chat.mode === "auto") approveAllPending(chat);
      broadcast();
      return json(res, 200, chatSummary(chat));
    }
  }

  // --- projects ---
  if (top === "projects") {
    const summary = (p) => ({ ...p, chats: chats.all().filter((c) => c.projectId === p.id && !c.incognito).length, files: listFiles(projectFilesDir(p.id)).length });
    if (!id) {
      if (m === "GET") return json(res, 200, projects.all().sort((a, b) => b.updated - a.updated).map(summary));
      const b = await body(req);
      if (!String(b.name || "").trim()) return json(res, 400, { error: "A project needs a name." });
      const p = projects.put({ id: newId(), name: b.name.trim().slice(0, 80), description: String(b.description || "").slice(0, 500), instructions: "", cwd: CWD, created: now(), updated: now() });
      broadcast();
      return json(res, 200, summary(p));
    }
    const p = projects.get(id);
    if (!p) return json(res, 404, { error: "No such project." });
    if (action === "files") {
      const dir = projectFilesDir(p.id);
      if (m === "GET" && !sub) return json(res, 200, listFiles(dir));
      if (m === "GET" && sub) return sendFile(res, path.join(dir, safeName(sub)));
      if (m === "POST") {
        const data = await readBody(req, MAX_UPLOAD);
        const name = safeName(decodeURIComponent(req.headers["x-filename"] || "file"));
        fs.writeFileSync(path.join(dir, name), data);
        p.updated = now(); projects.save(p);
        return json(res, 200, listFiles(dir));
      }
      if (m === "DELETE" && sub) { fs.rmSync(path.join(dir, safeName(sub)), { force: true }); return json(res, 200, listFiles(dir)); }
    }
    if (!action && m === "GET") return json(res, 200, summary(p));
    if (!action && m === "DELETE") {
      for (const c of chats.all()) if (c.projectId === p.id) { c.projectId = null; chats.save(c); }
      projects.delete(p.id);
      fs.rmSync(path.join(DIRS.projects, p.id), { recursive: true, force: true });
      broadcast();
      return json(res, 200, { ok: true });
    }
    if (!action && m === "POST") {
      const b = await body(req);
      Object.assign(p, pick({ name: b.name?.trim?.().slice(0, 80) || undefined, description: typeof b.description === "string" ? b.description.slice(0, 500) : undefined,
        instructions: typeof b.instructions === "string" ? b.instructions.slice(0, 20000) : undefined,
        cwd: typeof b.cwd === "string" && fs.existsSync(b.cwd) ? b.cwd : undefined }, ["name", "description", "instructions", "cwd"]));
      p.updated = now();
      projects.save(p);
      broadcast();
      return json(res, 200, summary(p));
    }
  }

  // --- artifacts ---
  if (top === "artifacts") {
    const summary = (a) => ({ ...a, chatTitle: chats.get(a.chatId)?.title || "Deleted chat" });
    if (!id) return json(res, 200, artifacts.all().filter((a) => !chats.get(a.chatId)?.incognito).sort((a, b) => b.updated - a.updated).map(summary));
    const a = artifacts.get(id);
    if (!a) return json(res, 404, { error: "No such artifact." });
    if (action === "raw") {
      const n = Math.min(a.versions.length, Math.max(1, Number(url.searchParams.get("v")) || a.versions.length));
      const file = versionFile(a, n);
      if (url.searchParams.get("download")) return sendFile(res, file, { download: true, name: path.basename(a.path) });
      res.writeHead(200, { "content-type": "text/plain; charset=utf-8", "x-content-type-options": "nosniff", "cache-control": "no-store" });
      return fs.createReadStream(file).pipe(res);
    }
    if (m === "GET") return json(res, 200, summary(a));
    if (m === "DELETE") {
      artifacts.delete(a.id);
      fs.rmSync(path.dirname(versionFile(a, 1)), { recursive: true, force: true });
      broadcast("artifacts");
      return json(res, 200, { ok: true });
    }
  }

  // --- scheduled tasks ---
  if (top === "schedules") {
    const view = (s) => ({ ...s, when: describe(s), lastChatTitle: s.lastChatId ? chats.get(s.lastChatId)?.title : null });
    const clean = (b, s = {}) => {
      if (typeof b.name === "string" && b.name.trim()) s.name = b.name.trim().slice(0, 80);
      if (typeof b.prompt === "string" && b.prompt.trim()) s.prompt = b.prompt.trim().slice(0, 20000);
      if (KINDS.includes(b.every)) s.every = b.every;
      if (/^\d{1,2}:\d{2}$/.test(b.time || "")) s.time = b.time;
      if (b.day !== undefined) s.day = Math.min(6, Math.max(0, Number(b.day) || 0));
      if (b.minutes !== undefined) s.minutes = Math.max(15, Number(b.minutes) || 60);
      if (MODELS[b.model]) s.model = b.model;
      if (EFFORTS.includes(b.effort)) s.effort = b.effort;
      if (b.mode === "auto" || b.mode === "manual") s.mode = b.mode;
      if (b.projectId === null || projects.get(b.projectId)) s.projectId = b.projectId;
      if (typeof b.enabled === "boolean") s.enabled = b.enabled;
      return s;
    };
    if (!id) {
      if (m === "GET") return json(res, 200, schedules.map(view));
      const s = clean(await body(req), { id: newId(), every: "daily", time: "08:00", day: 1, minutes: 60, model: "auto-sonnet", effort: "low", mode: "manual", projectId: null, enabled: true, created: now(), runs: 0 });
      if (!s.name || !s.prompt) return json(res, 400, { error: "A task needs a name and a prompt." });
      s.nextRun = nextRun(s);
      schedules.push(s);
      saveSchedules();
      return json(res, 200, view(s));
    }
    const s = schedules.find((x) => x.id === id);
    if (!s) return json(res, 404, { error: "No such task." });
    if (action === "run") { const err = runSchedule(s); return err ? json(res, 409, { error: err }) : json(res, 200, view(s)); }
    if (m === "DELETE") { schedules.splice(schedules.indexOf(s), 1); saveSchedules(); return json(res, 200, { ok: true }); }
    if (m === "POST") { clean(await body(req), s); s.nextRun = nextRun(s); saveSchedules(); return json(res, 200, view(s)); }
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
    if (!res.headersSent) json(res, err.status || 500, { error: String(err?.message || err) });
    else res.end();
  }
}).listen(PORT, "127.0.0.1", () => log(`plint-chat on 127.0.0.1:${PORT}, router ${ROUTER}`));

startScheduler();

// Incognito chats live in memory and go away an hour after they were last used.
setInterval(() => {
  for (const c of chats.all()) if (c.incognito && !c.running && now() - c.updated > 3600e3 && lv(c.id).listeners.size === 0) chats.delete(c.id);
}, 10 * 60e3);

for (const sig of ["SIGTERM", "SIGINT"]) {
  process.on(sig, () => { chats.flush(); projects.flush(); artifacts.flush(); process.exit(0); });
}
