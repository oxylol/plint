"""silk-router: one Silk Compute key behind OpenAI- and Anthropic-shaped endpoints.

Aliases map to Silk models; a stream that opens with a refusal is dropped and
re-streamed from the alias's -uncensored fallback.
"""
import json
import os
import re

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

UPSTREAM = os.environ.get("SILK_BASE_URL", "https://api.silkcompute.com").rstrip("/")
API_KEY = os.environ.get("SILK_API_KEY", "")
TOKEN = os.environ.get("SILK_ROUTER_TOKEN", "")  # optional: require clients to send it
BUFFER_CHARS = 240

FLASH = "deepseek-v4.1-flash-1017"
PRO = "kimi-k3.1-pro"
ALIASES = {
    "auto-sonnet": {"model": FLASH, "fallback": f"{FLASH}-uncensored"},
    "auto-haiku": {"model": FLASH, "fallback": f"{FLASH}-uncensored"},
    "auto-opus": {"model": PRO, "fallback": f"{PRO}-uncensored", "defaults": {"reasoning_effort": "low"}},
}
REFUSAL = re.compile(
    r"(i('m| am) sorry,? (but )?i (cannot|can't|won't)|i (cannot|can't|won't) (assist|help|fulfill|provide)"
    r"|as an ai|against my safety)",
    re.I,
)

app = FastAPI(title="silk-router")
client = httpx.AsyncClient(timeout=httpx.Timeout(600, connect=15))


def refused(text: str) -> bool:
    # Models often use a curly apostrophe; the pattern is written with a straight one.
    return bool(REFUSAL.search(text.replace("’", "'")))


def resolve(model: str) -> tuple[str, str | None, dict]:
    # Claude Code also asks for its own ids (e.g. claude-sonnet-...); route those by family.
    name = model or "auto-sonnet"
    if name not in ALIASES:
        for fam in ("opus", "sonnet", "haiku"):
            if name.startswith("claude") and fam in name:
                name = f"auto-{fam}"
    a = ALIASES.get(name)
    if not a:
        return model, None, {}
    return a["model"], a["fallback"], a.get("defaults", {})


def upstream_headers(request: Request) -> dict:
    h = {"Authorization": f"Bearer {API_KEY}", "x-api-key": API_KEY, "content-type": "application/json"}
    for k in ("anthropic-version", "anthropic-beta"):
        if k in request.headers:
            h[k] = request.headers[k]
    if "/messages" in request.url.path:
        h.setdefault("anthropic-version", "2023-06-01")
    return h


def authorized(request: Request) -> bool:
    if not TOKEN:
        return True
    sent = request.headers.get("x-api-key") or request.headers.get("authorization", "").removeprefix("Bearer ")
    return sent == TOKEN


# --- reading one SSE event --------------------------------------------------

def event_data(event: bytes) -> dict | None:
    for line in event.decode("utf-8", "replace").splitlines():
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload and payload != "[DONE]":
                try:
                    return json.loads(payload)
                except ValueError:
                    return None
    return None


def openai_delta(d: dict) -> tuple[str, bool]:
    text, tools = "", False
    for c in d.get("choices") or []:
        delta = c.get("delta") or {}
        text += delta.get("content") or ""
        tools = tools or bool(delta.get("tool_calls"))
    return text, tools


def anthropic_delta(d: dict) -> tuple[str, bool]:
    t = d.get("type")
    if t == "content_block_start":
        return "", (d.get("content_block") or {}).get("type") == "tool_use"
    if t == "content_block_delta":
        delta = d.get("delta") or {}
        if delta.get("type") == "text_delta":
            return delta.get("text") or "", False
        if delta.get("type") == "input_json_delta":
            return "", True
    return "", False


async def sse_events(resp: httpx.Response):
    buf = b""
    async for chunk in resp.aiter_bytes():
        buf += chunk.replace(b"\r\n", b"\n")
        while b"\n\n" in buf:
            event, buf = buf.split(b"\n\n", 1)
            yield event + b"\n\n"
    if buf.strip():
        yield buf + b"\n\n"


# --- proxying ---------------------------------------------------------------

async def open_stream(path: str, body: dict, headers: dict) -> httpx.Response:
    req = client.build_request("POST", UPSTREAM + path, json=body, headers=headers)
    return await client.send(req, stream=True)


async def error_response(resp: httpx.Response) -> Response:
    content = await resp.aread()
    await resp.aclose()
    return Response(content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))


def retryable(status: int) -> bool:
    return status >= 500 or status in (404, 429)


async def proxy(request: Request, path: str, parse) -> Response:
    if not authorized(request):
        return JSONResponse({"error": {"message": "bad router token"}}, status_code=401)
    body = await request.json()
    primary, fallback, defaults = resolve(body.get("model", ""))
    for k, v in defaults.items():
        body.setdefault(k, v)
    headers = upstream_headers(request)

    if not body.get("stream"):
        return await proxy_once(path, body, headers, primary, fallback, parse)

    body["model"] = primary
    resp = await open_stream(path, body, headers)
    if resp.status_code >= 400:
        if fallback and retryable(resp.status_code):
            await resp.aclose()
            body["model"] = fallback
            resp = await open_stream(path, body, headers)
        if resp.status_code >= 400:
            return await error_response(resp)

    async def relay():
        nonlocal resp
        held, text, committed = [], "", fallback is None
        events = sse_events(resp)
        try:
            async for ev in events:
                if committed:
                    yield ev
                    continue
                held.append(ev)
                d = event_data(ev)
                if d is None:
                    continue
                t, tools = parse(d)
                text += t
                if tools or (len(text) >= BUFFER_CHARS and not refused(text)):
                    committed = True
                    yield b"".join(held)
                    held = []
                elif refused(text):
                    break
            else:
                if held:
                    if not refused(text):
                        yield b"".join(held)
                        return
                else:
                    return
        finally:
            await resp.aclose()

        # The primary opened with a refusal: nothing was sent, start over on the fallback.
        print(f"silk-router: refusal from {primary}, re-streaming from {fallback}", flush=True)
        body["model"] = fallback
        resp = await open_stream(path, body, headers)
        try:
            if resp.status_code >= 400:
                err = (await resp.aread()).decode("utf-8", "replace")[:500]
                yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': err}})}\n\n".encode()
                return
            async for ev in sse_events(resp):
                yield ev
        finally:
            await resp.aclose()

    return StreamingResponse(relay(), media_type="text/event-stream", headers={"cache-control": "no-cache"})


def full_text(data: dict) -> tuple[str, bool]:
    if "choices" in data:
        msgs = [c.get("message") or {} for c in data["choices"]]
        return "".join(m.get("content") or "" for m in msgs), any(m.get("tool_calls") for m in msgs)
    blocks = data.get("content") or []
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    return text, any(b.get("type") == "tool_use" for b in blocks)


async def proxy_once(path, body, headers, primary, fallback, parse) -> Response:
    body["model"] = primary
    r = await client.post(UPSTREAM + path, json=body, headers=headers)
    if fallback:
        retry = retryable(r.status_code)
        if r.status_code < 400:
            text, tools = full_text(r.json())
            retry = not tools and refused(text[:BUFFER_CHARS])
        if retry:
            body["model"] = fallback
            r = await client.post(UPSTREAM + path, json=body, headers=headers)
    return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await proxy(request, "/v1/chat/completions", openai_delta)


@app.post("/v1/messages")
async def messages(request: Request):
    return await proxy(request, "/v1/messages", anthropic_delta)


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    # Claude Code asks for this; a rough estimate is enough.
    body = await request.json()
    return {"input_tokens": max(1, len(json.dumps(body)) // 4)}


@app.get("/v1/models")
async def models(request: Request):
    data = [{"id": a, "object": "model", "owned_by": "silk-router", "type": "model", "display_name": a} for a in ALIASES]
    try:
        r = await client.get(UPSTREAM + "/v1/models", headers=upstream_headers(request), timeout=15)
        if r.status_code == 200:
            data += r.json().get("data", [])
    except httpx.HTTPError:
        pass
    return {"object": "list", "data": data}


@app.get("/health")
async def health():
    return {"ok": True, "key": bool(API_KEY)}
