#!/usr/bin/env node
// plint-runner: lets Plint Chat run Claude Code in folders on this computer.
// It connects out to Plint Chat (no ports opened here), takes turns and file/git
// requests, and sends Claude Code's messages back. Model calls go through Plint.
//
//   node plint-runner.mjs --server https://ai.example.com --token <token> [--name "My PC"] [--root <folder> ...]
import { query } from "@anthropic-ai/claude-agent-sdk";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { randomBytes } from "node:crypto";
import { op } from "./ops.mjs";

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(`--${k}`); return i >= 0 ? args[i + 1] : d; };
const SERVER = (opt("server") || process.env.PLINT_SERVER || "").replace(/\/$/, "");
const TOKEN = opt("token") || process.env.PLINT_RUNNER_TOKEN;
const NAME = opt("name") || os.hostname();
const ROOTS = args.flatMap((a, i) => (a === "--root" ? [path.resolve(args[i + 1])] : []));
if (!ROOTS.length) ROOTS.push(os.homedir());
if (!SERVER || !TOKEN) { console.error("Usage: node plint-runner.mjs --server https://ai.<domain> --token <token> [--name NAME] [--root FOLDER]"); process.exit(1); }

const log = (...a) => console.log(new Date().toLocaleTimeString(), ...a);
const headers = { authorization: `Bearer ${TOKEN}`, "x-plint": "1", "content-type": "application/json" };
const running = new Map(); // chatId -> { q, abort }
const pending = new Map(); // askId -> resolve

// Posts keep their order per chat.
const queues = new Map();
function post(pathname, body, key = "_") {
  const prev = queues.get(key) || Promise.resolve();
  const next = prev.then(() => fetch(`${SERVER}${pathname}`, { method: "POST", headers, body: JSON.stringify(body) }).then((r) => r.ok || log("post failed", r.status)).catch((e) => log("post failed", e.message)));
  queues.set(key, next);
  return next;
}

// Only what Plint Chat draws, to keep the traffic small.
function slim(m) {
  if (m.type === "stream_event") {
    const d = m.event?.delta;
    if (m.event?.type !== "content_block_delta" || !(d?.type === "text_delta" || d?.type === "thinking_delta")) return null;
    return { type: "stream_event", parent_tool_use_id: m.parent_tool_use_id, event: { type: m.event.type, delta: d } };
  }
  if (m.type === "assistant") return { type: "assistant", uuid: m.uuid, parent_tool_use_id: m.parent_tool_use_id, error: m.error, message: { content: m.message?.content } };
  if (m.type === "user") return { type: "user", uuid: m.uuid, parent_tool_use_id: m.parent_tool_use_id, message: { content: Array.isArray(m.message?.content) ? m.message.content : [] } };
  if (m.type === "system" && m.subtype === "init") return { type: "system", subtype: "init", session_id: m.session_id };
  if (m.type === "result") return { type: "result", subtype: m.subtype, errors: m.errors, total_cost_usd: m.total_cost_usd };
  return null;
}

async function download(files, chatId) {
  if (!files?.length) return [];
  const dir = path.join(os.tmpdir(), "plint-runner", chatId);
  fs.mkdirSync(dir, { recursive: true });
  const out = [];
  for (const f of files) {
    try {
      const r = await fetch(`${SERVER}/api/runner/uploads/${f.id}/${encodeURIComponent(f.name)}`, { headers });
      if (!r.ok) continue;
      const file = path.join(dir, `${f.id}-${f.name}`);
      fs.writeFileSync(file, Buffer.from(await r.arrayBuffer()));
      out.push({ ...f, path: file });
    } catch {}
  }
  return out;
}

async function turn(job) {
  const { chatId } = job;
  const send = (body) => post("/api/runner/event", { chatId, ...body }, chatId);
  const abort = new AbortController();
  try {
    if (!fs.existsSync(job.cwd)) throw new Error(`The folder ${job.cwd} isn't there.`);
    const files = await download(job.files, chatId);
    const prompt = files.length ? `${job.text}\n\nFiles Filip attached (saved on this computer):\n${files.map((f) => `- ${f.path}`).join("\n")}` : job.text;
    const q = query({
      prompt,
      options: {
        ...job.options,
        cwd: job.cwd,
        abortController: abort,
        canUseTool: async (name, input, { signal, suggestions }) => {
          const id = randomBytes(6).toString("hex");
          send({ kind: "ask", id, name, input, suggestions });
          return new Promise((resolve) => {
            pending.set(id, resolve);
            signal.addEventListener("abort", () => resolve({ behavior: "deny", message: "Stopped." }), { once: true });
          });
        },
        env: {
          ...process.env,
          ANTHROPIC_BASE_URL: `${SERVER}/api/runner/llm`,
          ANTHROPIC_AUTH_TOKEN: TOKEN,
          ANTHROPIC_API_KEY: "",
          ...(job.env || {}),
        },
        stderr: (d) => log(`[${chatId}]`, d.trim().slice(0, 300)),
      },
    });
    running.set(chatId, { q, abort });
    log(`turn ${chatId} in ${job.cwd}`);
    for await (const m of q) {
      const s = slim(m);
      if (s) send({ kind: "msg", m: s });
    }
  } catch (err) {
    send({ kind: "error", text: abort.signal.aborted ? "Stopped." : String(err?.message || err) });
  } finally {
    running.delete(chatId);
    send({ kind: "end" });
  }
}

async function handle(msg) {
  if (msg.type === "turn") return turn(msg);
  if (msg.type === "answer") { pending.get(msg.id)?.(msg.result); pending.delete(msg.id); return; }
  if (msg.type === "stop") {
    const r = running.get(msg.chatId);
    try { await r?.q.interrupt(); } catch { r?.abort.abort(); }
    return;
  }
  if (msg.type === "rpc") {
    try { post("/api/runner/reply", { reqId: msg.reqId, ok: true, data: await op(msg.op, msg.payload || {}, ROOTS) }); }
    catch (err) { post("/api/runner/reply", { reqId: msg.reqId, ok: false, error: String(err.message || err) }); }
  }
}

// One long-lived event stream from Plint Chat; reconnect with backoff.
async function connect() {
  let delay = 1000;
  for (;;) {
    try {
      const qs = new URLSearchParams({ name: NAME, info: JSON.stringify(await op("info", {}, ROOTS)) });
      const r = await fetch(`${SERVER}/api/runner/connect?${qs}`, { headers });
      if (r.status === 401) { console.error("Plint Chat refused the token. Make a new one under Code → Computers."); process.exit(2); }
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      log(`connected to ${SERVER} as "${NAME}", sharing ${ROOTS.join(", ")}`);
      delay = 1000;
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf("\n\n")) >= 0) {
          const chunk = buf.slice(0, i);
          buf = buf.slice(i + 2);
          const data = chunk.split("\n").filter((l) => l.startsWith("data:")).map((l) => l.slice(5).trim()).join("\n");
          if (data) handle(JSON.parse(data)).catch((e) => log("error:", e.message));
        }
      }
      log("disconnected");
    } catch (err) {
      log(`can't reach Plint Chat (${err.message}); retrying in ${Math.round(delay / 1000)}s`);
    }
    await new Promise((r) => setTimeout(r, delay));
    delay = Math.min(delay * 2, 30000);
  }
}

process.on("SIGINT", () => process.exit(0));
connect();
