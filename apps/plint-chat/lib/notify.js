import { execFile } from "node:child_process";

const NOTIFY = process.env.NOTIFY_CMD || "/opt/plint/scripts/notify.sh";

export const log = (...a) => console.log(new Date().toISOString(), ...a);

export function notify(title, body, prio = "default", tags = "speech_balloon") {
  execFile(NOTIFY, [title, body, prio, tags], { timeout: 15000 }, (err) => err && log("notify failed:", err.message));
}
