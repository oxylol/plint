// Computers Code sessions can run on: Plint itself, plus any computer running
// plint-runner. A runner holds one event stream open; we push turns, answers and
// file/git requests down it and it posts results back.
import path from "node:path";
import { createHash, randomBytes } from "node:crypto";
import { DATA, readJson, writeJson, now } from "./store.js";
import { op } from "../runner/ops.mjs";
import { log } from "./notify.js";

const file = path.join(DATA, "machines.json");
const saved = readJson(file, []); // [{ id, name, tokenHash, created, lastSeen, info }]
const live = new Map(); // id -> { res, ping }
const rpcs = new Map(); // reqId -> { resolve, reject, timer }
const sha = (s) => createHash("sha256").update(s).digest("hex");
const save = () => writeJson(file, saved);
export const PLINT_ROOTS = (process.env.CODE_ROOTS || "/opt,/srv,/home/plint").split(",").filter(Boolean);
const onEvents = new Set();
export function onRunnerChange(fn) { onEvents.add(fn); }
const changed = () => { for (const fn of onEvents) fn(); };

export function list() {
  return [
    { id: "plint", name: "Plint", builtin: true, connected: true, info: { ...opInfo(), roots: PLINT_ROOTS } },
    ...saved.map((m) => ({ id: m.id, name: m.name, connected: live.has(m.id), lastSeen: m.lastSeen, created: m.created, info: m.info || null })),
  ];
}
let infoCache = null;
function opInfo() { return (infoCache ||= { os: "Plint (Ubuntu)", platform: "linux", home: "/opt/plint", sep: "/" }); }
export const get = (id) => list().find((m) => m.id === id);
export const isConnected = (id) => id === "plint" || live.has(id);

export function create(name) {
  const token = `pr_${randomBytes(24).toString("base64url")}`;
  const m = { id: randomBytes(5).toString("hex"), name: String(name || "Computer").slice(0, 40), tokenHash: sha(token), created: now(), lastSeen: null, info: null };
  saved.push(m);
  save();
  return { ...m, tokenHash: undefined, token };
}
export function remove(id) {
  const i = saved.findIndex((m) => m.id === id);
  if (i < 0) return false;
  saved.splice(i, 1);
  save();
  live.get(id)?.res.end();
  live.delete(id);
  changed();
  return true;
}
export function rename(id, name) {
  const m = saved.find((x) => x.id === id);
  if (m && name) { m.name = String(name).slice(0, 40); save(); }
}

export function byToken(req) {
  const t = (req.headers.authorization || "").replace(/^Bearer\s+/i, "") || req.headers["x-api-key"] || "";
  if (!t) return null;
  const h = sha(t);
  return saved.find((m) => m.tokenHash === h) || null;
}

export function connect(m, req, res, url) {
  res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache", "x-accel-buffering": "no" });
  res.write(": hello\n\n");
  live.get(m.id)?.res.end();
  const ping = setInterval(() => res.write(": ping\n\n"), 20000);
  const entry = { res, ping };
  live.set(m.id, entry);
  try { m.info = JSON.parse(url.searchParams.get("info") || "null"); } catch {}
  const name = url.searchParams.get("name");
  if (name && (!m.name || m.name === "Computer")) m.name = name.slice(0, 40);
  m.lastSeen = now();
  save();
  log(`runner ${m.name} connected`);
  changed();
  req.on("close", () => {
    clearInterval(ping);
    if (live.get(m.id) === entry) live.delete(m.id);
    m.lastSeen = now();
    save();
    log(`runner ${m.name} disconnected`);
    changed();
  });
}

export function send(id, msg) {
  const l = live.get(id);
  if (!l) return false;
  l.res.write(`data: ${JSON.stringify(msg)}\n\n`);
  return true;
}

// A file/git request on a machine. Plint answers in-process.
export function rpc(id, type, payload = {}, timeout = 20000) {
  if (id === "plint") return op(type, payload, PLINT_ROOTS);
  return new Promise((resolve, reject) => {
    const reqId = randomBytes(6).toString("hex");
    const timer = setTimeout(() => { rpcs.delete(reqId); reject(new Error("That computer didn't answer.")); }, timeout);
    rpcs.set(reqId, { resolve, reject, timer });
    if (!send(id, { type: "rpc", reqId, op: type, payload })) { clearTimeout(timer); rpcs.delete(reqId); reject(new Error("That computer isn't connected.")); }
  });
}
export function reply({ reqId, ok, data, error }) {
  const r = rpcs.get(reqId);
  if (!r) return;
  clearTimeout(r.timer);
  rpcs.delete(reqId);
  ok ? r.resolve(data) : r.reject(new Error(error || "Failed"));
}
