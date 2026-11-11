// File and git operations for Code sessions. Shared by Plint Chat (for folders on
// Plint) and plint-runner (for folders on other computers). Paths must stay inside
// the allowed roots.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFile } from "node:child_process";

const MAX_READ = 512 * 1024;
const MAX_PATCH = 400 * 1024;

export function info(roots) {
  return { os: `${os.type()} ${os.release()}`, platform: process.platform, home: os.homedir(), roots, sep: path.sep, host: os.hostname() };
}

function allowed(p, roots) {
  const abs = path.resolve(p);
  if (!roots?.length) return abs;
  for (const r of roots) {
    const rel = path.relative(path.resolve(r), abs);
    if (!rel.startsWith("..") && !path.isAbsolute(rel)) return abs;
  }
  throw new Error(`${abs} is outside the folders this computer shares.`);
}

export function ls(p, roots) {
  const dir = allowed(p || roots?.[0] || os.homedir(), roots);
  const entries = fs.readdirSync(dir, { withFileTypes: true })
    .filter((e) => e.name !== ".git" && e.name !== "node_modules")
    .map((e) => {
      let size = 0;
      const isDir = e.isDirectory() || (e.isSymbolicLink() && (() => { try { return fs.statSync(path.join(dir, e.name)).isDirectory(); } catch { return false; } })());
      if (!isDir) { try { size = fs.statSync(path.join(dir, e.name)).size; } catch {} }
      return { name: e.name, dir: isDir, size, hidden: e.name.startsWith(".") };
    })
    .sort((a, b) => (b.dir - a.dir) || a.name.localeCompare(b.name));
  const parent = path.dirname(dir);
  let up = parent !== dir ? parent : null;
  if (up) { try { allowed(up, roots); } catch { up = null; } }
  return { path: dir, parent: up, entries, git: fs.existsSync(path.join(dir, ".git")) };
}

export function read(p, roots) {
  const file = allowed(p, roots);
  const st = fs.statSync(file);
  if (st.isDirectory()) throw new Error("That's a folder.");
  const buf = Buffer.alloc(Math.min(st.size, MAX_READ));
  const fd = fs.openSync(file, "r");
  fs.readSync(fd, buf, 0, buf.length, 0);
  fs.closeSync(fd);
  const binary = buf.subarray(0, 8000).includes(0);
  return { path: file, size: st.size, truncated: st.size > MAX_READ, binary, text: binary ? "" : buf.toString("utf8") };
}

function git(cwd, args, max = MAX_PATCH) {
  return new Promise((resolve) => {
    execFile("git", ["-C", cwd, ...args], { maxBuffer: 8 * 1024 * 1024, timeout: 15000 }, (err, stdout) => {
      resolve(err ? null : stdout.length > max ? `${stdout.slice(0, max)}\n… (diff cut off)` : stdout);
    });
  });
}

export async function gitInfo(p, roots) {
  const cwd = allowed(p, roots);
  const top = (await git(cwd, ["rev-parse", "--show-toplevel"]))?.trim();
  if (!top) return { git: false, path: cwd };
  const branch = (await git(cwd, ["rev-parse", "--abbrev-ref", "HEAD"]))?.trim();
  const remote = (await git(cwd, ["remote", "get-url", "origin"]))?.trim() || null;
  return { git: true, path: cwd, root: top, branch, remote };
}

export async function diff(p, roots) {
  const cwd = allowed(p, roots);
  const status = await git(cwd, ["status", "--porcelain=v1", "-uall"]);
  if (status === null) return { git: false, files: [], patch: "" };
  const num = (await git(cwd, ["diff", "HEAD", "--numstat"])) || "";
  const stats = Object.fromEntries(num.split("\n").filter(Boolean).map((l) => {
    const [a, d, ...f] = l.split("\t");
    return [f.join("\t"), { additions: Number(a) || 0, deletions: Number(d) || 0 }];
  }));
  const files = status.split("\n").filter(Boolean).map((l) => {
    const code = l.slice(0, 2).trim() || "M";
    const file = l.slice(3).replace(/^"|"$/g, "").split(" -> ").pop();
    return { path: file, status: code === "??" ? "new" : code.includes("D") ? "deleted" : code.includes("A") ? "added" : code.includes("R") ? "renamed" : "modified", ...(stats[file] || {}) };
  });
  let patch = (await git(cwd, ["diff", "HEAD"])) || "";
  for (const f of files.filter((x) => x.status === "new").slice(0, 20)) {
    try {
      const r = read(path.join(cwd, f.path), roots);
      if (r.binary) continue;
      const lines = r.text.split("\n");
      f.additions = lines.length;
      patch += `diff --git a/${f.path} b/${f.path}\nnew file\n--- /dev/null\n+++ b/${f.path}\n@@ -0,0 +1,${lines.length} @@\n${lines.map((x) => `+${x}`).join("\n")}\n`;
    } catch {}
  }
  if (patch.length > MAX_PATCH) patch = `${patch.slice(0, MAX_PATCH)}\n… (diff cut off)`;
  return { git: true, files, patch };
}

export async function op(type, payload, roots) {
  if (type === "info") return info(roots);
  if (type === "ls") return ls(payload.path, roots);
  if (type === "read") return read(payload.path, roots);
  if (type === "git") return gitInfo(payload.path, roots);
  if (type === "diff") return diff(payload.path, roots);
  throw new Error(`Unknown request: ${type}`);
}
