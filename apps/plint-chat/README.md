# Plint Chat

`https://ai.${DOMAIN}`: a Claude-style chat page that runs Claude Code on Plint, so Plint can be
driven from a phone. Models come from silk-router (`apps/silk-router`, localhost:4141):
Auto Sonnet / Auto Haiku → DeepSeek Flash, Auto Opus → Kimi K3.1 Pro.

- **How it works**: `server.js` (node 18, no framework) starts one Claude Code session per chat with the
  Agent SDK (`cwd /opt/plint`, user/project settings and CLAUDE.md loaded) and resumes it on every
  message. Turns run server-side, so a locked phone doesn't stop them. The page follows over SSE
  and catches up from the stored events after a reload.
- **Permissions**: whatever Claude Code would ask about shows up as an Allow / Always / Deny card.
  Read-only commands run without asking, as they do in Claude Code. The chat menu has
  *Auto-approve tools* for one chat. Nobody watching → the ask goes to ntfy too, and so does the
  reply when the turn ends.
- **Login**: "Send code to my phone" sends a 6-digit code by ntfy (10 min, 5 tries, 1 per 30 s, 6 per
  hour). The cookie lasts a year. Sessions are stored as hashes in `data/plint-chat/auth.json`;
  delete that file to sign everyone out.
- **Runs as** the native `plint-chat.service` (user `plint`, port 5220, `MemoryMax=2500M`), not Docker,
  because it has to act on the host. At most two chats work at once.
- **Data**: `data/plint-chat/chats/<id>.json` (the page's events); the Claude Code transcripts live in
  `~plint/.claude/projects/-opt-plint/`.

Logs: `journalctl -u plint-chat -n 100`. Restart: `sudo systemctl restart plint-chat` (stops running turns).
Not built yet: photo/file uploads, AskUserQuestion (it's turned off, so the model asks in text instead).
