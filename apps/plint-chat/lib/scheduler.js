// Scheduled tasks: a prompt that runs on its own as a new chat. Checked every 30 s.
import { schedules, saveSchedules, chats, newId, now } from "./store.js";
import { startTurn, broadcast } from "./agent.js";
import { log } from "./notify.js";

export const KINDS = ["hourly", "daily", "weekdays", "weekly", "interval"];

// Next run strictly after `from`, in the server's local time.
export function nextRun(s, from = now()) {
  const d = new Date(from);
  const [hh, mm] = String(s.time || "08:00").split(":").map(Number);
  if (s.every === "interval") return from + Math.max(15, Number(s.minutes) || 60) * 60e3;
  if (s.every === "hourly") {
    const n = new Date(d); n.setSeconds(0, 0); n.setMinutes(mm || 0);
    if (n.getTime() <= from) n.setHours(n.getHours() + 1);
    return n.getTime();
  }
  const n = new Date(d); n.setSeconds(0, 0); n.setHours(hh || 0, mm || 0);
  for (let i = 0; i < 8; i++) {
    const ok = n.getTime() > from && (s.every === "daily"
      || (s.every === "weekdays" && n.getDay() >= 1 && n.getDay() <= 5)
      || (s.every === "weekly" && n.getDay() === Number(s.day ?? 1)));
    if (ok) return n.getTime();
    n.setDate(n.getDate() + 1);
  }
  return from + 86400e3;
}

export function describe(s) {
  const days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
  if (s.every === "interval") return `Every ${s.minutes} minutes`;
  if (s.every === "hourly") return `Every hour at :${String(s.time || "00:00").split(":")[1] || "00"}`;
  if (s.every === "weekdays") return `Weekdays at ${s.time}`;
  if (s.every === "weekly") return `${days[Number(s.day ?? 1)]}s at ${s.time}`;
  return `Every day at ${s.time}`;
}

export function runSchedule(s) {
  const stamp = new Date().toLocaleString("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
  const chat = chats.put({
    id: newId(), title: `${s.name} · ${stamp}`, model: s.model || "auto-sonnet", effort: s.effort || "low",
    mode: s.mode === "auto" ? "auto" : "manual", style: "normal", projectId: s.projectId || null, scheduleId: s.id,
    created: now(), updated: now(), running: false, sessionId: null, events: [],
  });
  const err = startTurn(chat, { text: s.prompt });
  if (err) { chats.delete(chat.id); log(`schedule ${s.name}: ${err}`); return err; }
  s.lastRun = now();
  s.lastChatId = chat.id;
  s.runs = (s.runs || 0) + 1;
  saveSchedules();
  broadcast();
  return null;
}

export function startScheduler() {
  for (const s of schedules) if (!s.nextRun || s.nextRun < now()) s.nextRun = nextRun(s);
  saveSchedules();
  setInterval(() => {
    for (const s of schedules) {
      if (!s.enabled || !s.nextRun || s.nextRun > now()) continue;
      s.nextRun = nextRun(s);
      saveSchedules();
      log(`schedule ${s.name}: running`);
      runSchedule(s);
    }
  }, 30e3);
}
