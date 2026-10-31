// Hertz on Plint: serves the instrument and keeps saved songs in DATA_DIR,
// so the phone, the TV Mac and the laptop all see the same list. No deps.
"use strict";
const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = Number(process.env.PORT || 5200);
const DATA = process.env.DATA_DIR || path.join(__dirname, "data");
const PUBLIC = path.join(__dirname, "public");
const SONGS = path.join(DATA, "songs.json");

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".webmanifest": "application/manifest+json",
};

function readSongs() {
  try { return JSON.parse(fs.readFileSync(SONGS, "utf8")); } catch { return []; }
}
function writeSongs(list) {
  fs.mkdirSync(DATA, { recursive: true });
  const tmp = SONGS + ".tmp";
  fs.writeFileSync(tmp, JSON.stringify(list, null, 1));
  fs.renameSync(tmp, SONGS);
}
function send(res, code, body, type = "application/json") {
  res.writeHead(code, { "Content-Type": type, "Cache-Control": "no-cache" });
  res.end(typeof body === "string" || Buffer.isBuffer(body) ? body : JSON.stringify(body));
}
function body(req) {
  return new Promise((resolve, reject) => {
    let n = 0; const parts = [];
    req.on("data", (c) => { n += c.length; if (n > 256 * 1024) { reject(new Error("too big")); req.destroy(); } else parts.push(c); });
    req.on("end", () => { try { resolve(JSON.parse(Buffer.concat(parts).toString("utf8") || "{}")); } catch (e) { reject(e); } });
  });
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://x");
  try {
    if (url.pathname === "/health") return send(res, 200, { ok: true });

    if (url.pathname === "/api/songs") {
      if (req.method === "GET") {
        return send(res, 200, readSongs().map(({ id, name, saved }) => ({ id, name, saved })));
      }
      if (req.method === "POST") {
        const b = await body(req);
        const name = String(b.name || "Untitled").slice(0, 60).trim() || "Untitled";
        if (!b.patch || typeof b.patch !== "object") return send(res, 400, { error: "patch missing" });
        const list = readSongs();
        let song = b.id && list.find((s) => s.id === b.id);
        if (!song) { song = { id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6) }; list.unshift(song); }
        Object.assign(song, { name, patch: b.patch, saved: new Date().toISOString() });
        writeSongs(list.slice(0, 200));
        return send(res, 200, { id: song.id, name: song.name, saved: song.saved });
      }
    }
    const m = url.pathname.match(/^\/api\/songs\/([a-z0-9]+)$/);
    if (m) {
      const list = readSongs();
      const i = list.findIndex((s) => s.id === m[1]);
      if (i < 0) return send(res, 404, { error: "not found" });
      if (req.method === "GET") return send(res, 200, list[i]);
      if (req.method === "DELETE") { list.splice(i, 1); writeSongs(list); return send(res, 200, { ok: true }); }
    }
    if (url.pathname.startsWith("/api/")) return send(res, 404, { error: "not found" });

    // Static files; everything unknown falls back to the app.
    let file = path.normalize(path.join(PUBLIC, decodeURIComponent(url.pathname)));
    if (!file.startsWith(PUBLIC)) return send(res, 403, "no", "text/plain");
    if (!fs.existsSync(file) || fs.statSync(file).isDirectory()) file = path.join(PUBLIC, "index.html");
    send(res, 200, fs.readFileSync(file), TYPES[path.extname(file)] || "application/octet-stream");
  } catch (e) {
    send(res, 500, { error: String(e.message || e) });
  }
});

server.listen(PORT, "0.0.0.0", () => console.log(`hertz on :${PORT}, songs in ${DATA}`));
