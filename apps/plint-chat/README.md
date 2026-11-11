# Plint Chat

`https://ai.${DOMAIN}`: a claude.ai-style app that runs Claude Code on Plint, so Plint can be driven
from a phone. Models come from silk-router (`apps/silk-router`, localhost:4141): Auto Sonnet / Auto
Haiku → DeepSeek V4.1 Flash, Auto Opus → Kimi K3.1 Pro (effort low/medium/high → `auto-opus@high`).

The look follows the Claude app as of November 2026: every size, colour, radius and timing was measured from a
saved copy of claude.ai (sidebar, home and composer from the saved page itself; chat view, menus, dialogs,
pages and the phone layout rebuilt from its stylesheet and component code). The code is our own. Two things are
deliberately not Claude's: the name and spark say Plint, and there is no voice mode.

**Typefaces and icon font.** If `data/plint-chat/brand/` holds `sans.woff2`, `sans-fallback.woff2`,
`sans-italic.woff2`, `serif.woff2`, `serif-italic.woff2`, `mono.woff2`, `mono-italic.woff2` and `icons.woff2`
(Anthropic's fonts as the browser downloaded them from claude.ai), the page uses them. They are Anthropic's
property, so they live in `data/` (gitignored, backed up by Kopia) and never in this repo. Without that folder
the page falls back to system fonts and its own line icons (`ICONS` in `public/js/ui.js` holds both).

## What's in it
- **Sidebar**: Chat | Code switch, New (⇧⌘O / Ctrl+Shift+O), Search (⌘K), Projects, Artifacts, Scheduled,
  Customize, Pinned and Recents (foldable, filter), View all, account menu. Row menu with letter keys
  (P pin, R rename in place, D delete), right-click or press-and-hold. Drag the edge to resize, ⌘. hides it,
  hovering the button then lets it peek. On a phone it is a full-screen sheet (swipe left to close).
- **Composer**: + menu (files, photo, project, style, incognito), dictation with a language picker, send /
  stop; under the box: project picker, model (+ Effort submenu) and permission mode. Files land in
  `data/plint-chat/uploads/` and their paths go to the model. Docked to the bottom on a phone.
- **Chat**: replies stream in serif with words fading in; thinking and tool calls fold into one status line
  ("Thought for 8s", "Ran a command") that opens to a card of steps; copy, read aloud, retry, edit (edit and
  retry fork the Claude Code session at that point), timestamps, rename by clicking the title, auto titles,
  scroll-to-bottom. A tool that needs an OK docks a request above the composer: Deny (Esc), Always allow,
  Allow once (Ctrl/⌘+Enter), or 1 / 2 / 3.
- **Artifacts**: the model writes .html/.svg/.md/.mermaid into the chat's artifact folder (no
  permission needed); each write is a version. They open in a side panel beside the chat (half the width, drag to resize; full screen on phones) with
  Preview/Code, copy, download and versions. HTML runs in a sandboxed iframe with no access to this origin.
- **Projects**: name, description, instructions, files (`data/plint-chat/projects/<id>/files`), working
  folder; chats started in a project get all of it.
- **Scheduled**: prompts that run hourly / daily / weekdays / weekly / every N minutes as new chats;
  Manual tasks send their approval asks to ntfy.
- **Customize**: skills list, styles (built-in + your own, default), personal instructions, connectors note.
- **Settings** (account menu or ⌘,): colour mode, reply font, Enter key, name, log out.
- **Incognito**: not listed, not written to disk, gone an hour after last use.
- **Code** (Chat | Code switch at the top of the sidebar): a session is Claude Code in one folder on one
  computer. Pick the computer and folder (folder browser, recent folders), then work like in a chat;
  modes Manual / Accept edits / Auto. Beside it: **Changes** (git status + diff of the folder) and
  **Files** (browse and read). Plint's own folders are limited to `CODE_ROOTS` (default /opt, /srv,
  /home/plint).
- **Computers** (Code → Computers): other machines run `runner/plint-runner.mjs` (Node 18+, Mac / Windows
  / Linux). "Add a computer" makes a token and shows the install command. The runner connects out to
  `https://ai.${DOMAIN}` (LAN or Tailscale), runs Claude Code there, and sends its model calls back
  through `/api/runner/llm` to silk-router. `--root <folder>` limits what it shares (default: home).
  Tokens are stored hashed in `data/plint-chat/machines.json`; removing a computer revokes it.

## How it runs
`server.js` routes; `lib/agent.js` runs turns with the Agent SDK (cwd /opt/plint or the project's
folder, user/project settings and CLAUDE.md loaded) and resumes the session per message; turns keep
running when the phone sleeps and nobody-watching asks/replies go to ntfy. `lib/scheduler.js`,
`lib/store.js` (JSON files under `data/plint-chat`), `lib/auth.js` (6-digit code by ntfy, year-long
cookie, hashes only). Front end: `public/js/*.js` ES modules and four stylesheets (`base` tokens and primitives, `shell`, `thread`,
`pages`), no build step. Menus are bottom sheets and controls grow to 32px on touch screens.

Native `plint-chat.service` (user `plint`, port 5220, `MemoryMax=2500M`), at most three chats working at
once. Logs: `journalctl -u plint-chat -n 100`. Restart: `sudo systemctl restart plint-chat` (stops
running turns).

Not available with Silk models: claude.ai connectors, Anthropic's web search, voice mode, sharing.
