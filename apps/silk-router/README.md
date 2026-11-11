# silk-router

One Silk Compute key (`SILK_API_KEY` in `/opt/plint/.env`) behind `GET /v1/models`,
`POST /v1/chat/completions` and `POST /v1/messages` on `127.0.0.1:4141` (stack `ai`).

| alias | model | fallback |
|---|---|---|
| auto-sonnet, auto-haiku | deepseek-v4.1-flash-1017 | deepseek-v4.1-flash-1017-uncensored |
| auto-opus | kimi-k3.1-pro (`reasoning_effort: low` unless set) | kimi-k3.1-pro-uncensored |

`claude-*` model ids are routed by family (opus/sonnet/haiku); other names pass through.

Streams: the first 240 characters of text are held back. A tool call releases them at once.
If the held text matches the refusal pattern, the stream is dropped and run again on the
fallback (nothing has reached the client yet). Non-streamed replies get the same check.
An upstream 404/429/5xx on the primary also goes to the fallback.

- Claude Code on it: `apps/silk-router/silk-claude` (same flags as `claude`)
- Health: `curl localhost:4141/health` (`key: false` = no `SILK_API_KEY`)
- Logs: `plint logs silk-router` (refusal swaps are logged)
- Optional `SILK_ROUTER_TOKEN` in `.env`: clients must then send it as their API key.
