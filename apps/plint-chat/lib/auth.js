// Login: a 6-digit code sent to Filip's phone by ntfy, then a year-long cookie.
// Only hashes of codes and session tokens are kept.
import path from "node:path";
import { createHash, randomBytes, randomInt, timingSafeEqual } from "node:crypto";
import { DATA, readJson, writeJson, now } from "./store.js";
import { notify } from "./notify.js";

const COOKIE = "plint_chat";
const DEV_CODE = process.env.PLINT_CHAT_DEV_CODE || ""; // tests only: fixed code, no ntfy
const file = path.join(DATA, "auth.json");
const auth = readJson(file, { sessions: {} });
const codes = []; // { hash, expires, tries }
const sends = [];
const sha = (s) => createHash("sha256").update(s).digest("hex");

function cookies(req) {
  return Object.fromEntries((req.headers.cookie || "").split(/;\s*/).filter(Boolean).map((p) => {
    const i = p.indexOf("=");
    return [p.slice(0, i), decodeURIComponent(p.slice(i + 1))];
  }));
}

export function signedIn(req) {
  const t = cookies(req)[COOKIE];
  return !!(t && auth.sessions[sha(t)]);
}

export function sendCode() {
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

export function login(code, userAgent) {
  const t = now();
  const h = Buffer.from(sha(String(code || "").trim()));
  let ok = false;
  for (const c of codes) {
    if (c.expires < t || c.tries >= 5) continue;
    c.tries++;
    if (timingSafeEqual(Buffer.from(c.hash), h)) { c.expires = 0; ok = true; break; }
  }
  if (!ok) return null;
  const token = randomBytes(32).toString("base64url");
  auth.sessions[sha(token)] = { created: t, ua: String(userAgent || "").slice(0, 120) };
  writeJson(file, auth);
  return `${COOKIE}=${token}; Path=/; Max-Age=31536000; HttpOnly; Secure; SameSite=Lax`;
}

export function logout(req) {
  const t = cookies(req)[COOKIE];
  if (t) { delete auth.sessions[sha(t)]; writeJson(file, auth); }
  return `${COOKIE}=; Path=/; Max-Age=0`;
}
