// Shared app state: who's signed in, the model list, chats and projects for the
// sidebar. Views subscribe to "chats" / "projects" / "artifacts" changes.
import { api } from "./api.js";

export const state = {
  me: null, models: {}, efforts: [], styles: [],
  chats: [], projects: [],
  current: null, // id of the open chat
  incognito: false,
};
const subs = {};
export function on(kind, fn) { (subs[kind] ||= new Set()).add(fn); return () => subs[kind].delete(fn); }
export function fire(kind) { for (const fn of subs[kind] || []) fn(); }

let chatsTimer = null;
export async function refreshChats() {
  clearTimeout(chatsTimer);
  try { state.chats = await api("chats"); fire("chats"); } catch {}
}
export async function refreshProjects() {
  try { state.projects = await api("projects"); fire("projects"); } catch {}
}
export async function refreshStyles() {
  try { const s = await api("settings"); state.styles = s.allStyles; state.me.name = s.name; fire("styles"); } catch {}
}
export const projectName = (id) => state.projects.find((p) => p.id === id)?.name;

// Live updates for the sidebar and galleries.
let es = null;
export function connectEvents() {
  es?.close();
  es = new EventSource("/api/events");
  es.onmessage = (m) => {
    const ev = JSON.parse(m.data);
    if (ev.t === "list") { clearTimeout(chatsTimer); chatsTimer = setTimeout(() => { refreshChats(); refreshProjects(); }, 100); }
    if (ev.t === "artifacts") fire("artifacts");
  };
  es.onerror = () => { if (es.readyState === EventSource.CLOSED) setTimeout(connectEvents, 3000); };
}
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && state.me && (!es || es.readyState === EventSource.CLOSED)) { connectEvents(); refreshChats(); }
});

// Composer defaults remembered per device.
export const prefs = {
  get(k, d) { try { const v = localStorage.getItem(`pc.${k}`); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(`pc.${k}`, JSON.stringify(v)); } catch {} },
};
