# Plint Chat

`https://ai.${DOMAIN}`: a claude.ai-style app that runs Claude Code on Plint, so Plint can be driven
from a phone. Models come from silk-router (`apps/silk-router`, localhost:4141): Auto Sonnet / Auto
Haiku → DeepSeek V4.1 Flash, Auto Opus → Kimi K3.1 Pro (effort low/medium/high → `auto-opus@high`).

Layout and colours follow claude.ai's current neutral look (tokens taken from its stylesheet); the code,
icons and fonts are our own (system sans, a serif for replies).

## What's in it
- **Sidebar**: New chat (⇧⌘O / Ctrl+Shift+O), Search (⌘K), Projects, Artifacts, Scheduled, Customize,
  Pinned, Recents with a filter, View all, account menu (settings, theme, sign out). Drawer on phones,
  collapsible rail on desktop (⌘.).
- **Composer**: + menu (upload files, take a photo, response style, incognito), project picker, model +
  effort, Manual/Auto mode, dictation (Web Speech API), paste/drop files. Files land in
  `data/plint-chat/uploads/` and their paths go to the model.
- **Chat**: streaming replies in serif, thinking blocks, tool steps grouped, Allow / Always / Deny
  cards (Manual mode), copy, retry and edit (both fork the Claude Code session at that point), auto
  titles, scroll-to-bottom.
- **Artifacts**: the model writes .html/.svg/.md/.mermaid into the chat's artifact folder (no
  permission needed); each write is a version. They open in a side panel (full screen on phones) with
  Preview/Code, versions, copy, download. HTML runs in a sandboxed iframe with no access to this origin.
- **Projects**: name, description, instructions, files (`data/plint-chat/projects/<id>/files`), working
  folder; chats started in a project get all of it.
- **Scheduled**: prompts that run hourly / daily / weekdays / weekly / every N minutes as new chats;
  Manual tasks send their approval asks to ntfy.
- **Customize**: personal instructions, styles (built-in + your own, default), skills list, connectors note.
- **Incognito**: not listed, not written to disk, gone an hour after last use.

## How it runs
`server.js` routes; `lib/agent.js` runs turns with the Agent SDK (cwd /opt/plint or the project's
folder, user/project settings and CLAUDE.md loaded) and resumes the session per message; turns keep
running when the phone sleeps and nobody-watching asks/replies go to ntfy. `lib/scheduler.js`,
`lib/store.js` (JSON files under `data/plint-chat`), `lib/auth.js` (6-digit code by ntfy, year-long
cookie, hashes only). Front end: `public/js/*.js` ES modules, no build step.

Native `plint-chat.service` (user `plint`, port 5220, `MemoryMax=2500M`), at most three chats working at
once. Logs: `journalctl -u plint-chat -n 100`. Restart: `sudo systemctl restart plint-chat` (stops
running turns).

Not available with Silk models: claude.ai connectors, Anthropic's web search, voice mode, sharing.
