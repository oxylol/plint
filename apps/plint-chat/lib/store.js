// Everything Plint Chat keeps on disk, under DATA_DIR. One JSON file per chat,
// project and artifact; settings and schedules are single files. Incognito chats
// stay in memory only.
import fs from "node:fs";
import path from "node:path";
import { randomBytes } from "node:crypto";

export const DATA = process.env.DATA_DIR || "/opt/plint/data/plint-chat";
export const DIRS = {
  chats: path.join(DATA, "chats"),
  projects: path.join(DATA, "projects"),
  artifacts: path.join(DATA, "artifacts"),
  uploads: path.join(DATA, "uploads"),
};
for (const d of Object.values(DIRS)) fs.mkdirSync(d, { recursive: true });

export const newId = (bytes = 8) => randomBytes(bytes).toString("hex");
export const now = () => Date.now();

export function readJson(file, fallback) {
  try { return JSON.parse(fs.readFileSync(file, "utf8")); } catch { return fallback; }
}
export function writeJson(file, value) {
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(value), { mode: 0o600 });
  fs.renameSync(tmp, file);
}

class Collection {
  constructor(dir) {
    this.dir = dir;
    this.items = new Map();
    this.timers = new Map();
    for (const f of fs.readdirSync(dir)) {
      if (!f.endsWith(".json")) continue;
      const v = readJson(path.join(dir, f), null);
      if (v?.id) this.items.set(v.id, v);
    }
  }
  file(id) { return path.join(this.dir, `${id}.json`); }
  get(id) { return this.items.get(id); }
  all() { return [...this.items.values()]; }
  put(item) { this.items.set(item.id, item); this.save(item); return item; }
  save(item) {
    if (item.incognito) return;
    clearTimeout(this.timers.get(item.id));
    this.timers.set(item.id, setTimeout(() => { this.timers.delete(item.id); if (this.items.has(item.id)) writeJson(this.file(item.id), item); }, 250));
  }
  flush() { for (const item of this.items.values()) if (!item.incognito) writeJson(this.file(item.id), item); }
  delete(id) {
    clearTimeout(this.timers.get(id));
    this.items.delete(id);
    fs.rmSync(this.file(id), { force: true });
  }
}

export const chats = new Collection(DIRS.chats);
export const projects = new Collection(DIRS.projects);
export const artifacts = new Collection(path.join(DIRS.artifacts));

// --- settings ----------------------------------------------------------------

export const BUILTIN_STYLES = [
  { id: "normal", name: "Normal", prompt: "", builtin: true },
  { id: "concise", name: "Concise", prompt: "Answer as briefly as you can: short sentences, no preamble, no recap.", builtin: true },
  { id: "explanatory", name: "Explanatory", prompt: "Explain your reasoning and the why behind each step, like a patient teacher. Define terms on first use.", builtin: true },
  { id: "formal", name: "Formal", prompt: "Write in a clear, formal, professional register.", builtin: true },
];
const settingsFile = path.join(DATA, "settings.json");
export const settings = Object.assign({ name: "Filip", instructions: "", styles: [], defaultStyle: "normal" }, readJson(settingsFile, {}));
export function saveSettings() { writeJson(settingsFile, settings); }
export function allStyles() { return [...BUILTIN_STYLES, ...settings.styles]; }
export function styleById(id) { return allStyles().find((s) => s.id === id) || BUILTIN_STYLES[0]; }

// --- schedules ----------------------------------------------------------------

const schedulesFile = path.join(DATA, "schedules.json");
export const schedules = readJson(schedulesFile, []);
export function saveSchedules() { writeJson(schedulesFile, schedules); }

// --- helpers ------------------------------------------------------------------

export function chatSummary(c) {
  return {
    id: c.id, title: c.title, model: c.model, effort: c.effort, mode: c.mode, style: c.style,
    projectId: c.projectId || null, pinned: !!c.pinned, incognito: !!c.incognito,
    created: c.created, updated: c.updated, running: !!c.running, scheduleId: c.scheduleId || null,
    hasArtifact: !!c.hasArtifact,
  };
}

export function projectFilesDir(id) {
  const d = path.join(DIRS.projects, id, "files");
  fs.mkdirSync(d, { recursive: true });
  return d;
}
export function listFiles(dir) {
  try {
    return fs.readdirSync(dir).filter((f) => !f.startsWith(".")).map((name) => {
      const st = fs.statSync(path.join(dir, name));
      return { name, size: st.size, updated: st.mtimeMs };
    }).sort((a, b) => b.updated - a.updated);
  } catch { return []; }
}
export function safeName(name) {
  return String(name || "file").replace(/[\\/\0]/g, "_").replace(/^\.+/, "_").slice(0, 120) || "file";
}
