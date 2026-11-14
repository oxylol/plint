"""The one module that talks to Anthropic, and the only one that imports its SDK: inside the functions, so
the rest of the app loads, and its tests run, with no key and no network.

Every request has the same shape: streamed, adaptive thinking, the effort set in output_config, and the
answer as JSON through output_config.format (the 5.5 models reject forced tool calls, prefill and sampling
parameters). The main model also carries the server-side refusal fallback; the small model has none."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

from .budget import Bill

FALLBACK_BETA = "server-side-fallback-2026-07-01"   # the header that goes with fallbacks="default"
WEB_SEARCH = "web_search_20260209"                  # runs its own result filtering: no code execution tool beside it
CACHE = {"type": "ephemeral"}                       # a breakpoint: the prefix up to here is kept for 5 minutes


class LLMError(Exception):
    """A call that came back with no response at all. `kind` says which way it failed."""

    def __init__(self, kind: str, detail: str = ""):
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind = kind


@dataclass(frozen=True)
class Ask:
    """One request. `system` is the frozen brief followed by tonight's context; a cached system or user
    part is marked at its end, so everything before the mark is read back at the cache price."""
    model: str
    system: tuple[str, ...]
    user: str
    schema: dict
    max_tokens: int
    effort: str
    fallback: bool = False          # the main model: beta endpoint with fallbacks="default"
    cache_system: bool = False
    cache_user: bool = False
    searches: int = 0               # the web search tool's max_uses; 0 leaves the tool out
    paused: tuple = ()              # this turn's earlier paused replies, sent back exactly as received
    container: str | None = None    # the search tool's sandbox, kept across a pause


@dataclass
class Reply:
    stop: str                       # the API's stop_reason
    data: dict | None               # the answer object; None unless the turn ended normally with valid JSON
    top: Bill                       # the response's own usage
    parts: list[Bill]               # usage per model when the API lists the attempts, else empty
    refusal: str | None = None      # the refusal's category when stop is "refusal"
    urls: dict[str, str] = field(default_factory=dict)   # url -> title of what the search really returned
    content: list = field(default_factory=list)          # the blocks as received
    container: str | None = None
    usd: float = 0.0                # filled in by the desk once the call is booked

    @property
    def searches(self) -> int:
        return self.top.searches


def make_client(key: str):
    """The real client. The SDK itself retries connection errors, 408, 409, 429 and 5xx (twice, backing off)."""
    import anthropic
    return anthropic.Anthropic(api_key=key)


def request(ask: Ask) -> dict:
    """The keyword arguments of messages.stream for this ask."""
    system = [{"type": "text", "text": t} for t in ask.system if t]
    if ask.cache_system and system:
        system[-1]["cache_control"] = dict(CACHE)
    user = {"type": "text", "text": ask.user}
    if ask.cache_user:
        user["cache_control"] = dict(CACHE)
    messages = [{"role": "user", "content": [user]}]
    messages += [{"role": "assistant", "content": content} for content in ask.paused]
    kw = {
        "model": ask.model,
        "max_tokens": ask.max_tokens,
        "system": system,
        "messages": messages,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": ask.effort, "format": {"type": "json_schema", "schema": ask.schema}},
    }
    if ask.searches:
        kw["tools"] = [{"type": WEB_SEARCH, "name": "web_search", "max_uses": ask.searches}]
    if ask.container:
        kw["container"] = ask.container
    if ask.fallback:
        kw["betas"] = [FALLBACK_BETA]
        kw["fallbacks"] = "default"
    return kw


class LLM:
    def __init__(self, client):
        self.client = client

    def call(self, ask: Ask) -> Reply:
        """Sends one request and reads the whole response. A refusal, a paused turn or a cut-off answer is
        a Reply to look at, not an error: only a call with no response raises."""
        import anthropic as sdk
        api = self.client.beta.messages if ask.fallback else self.client.messages
        try:
            with api.stream(**request(ask)) as stream:
                message = stream.get_final_message()
        except sdk.BadRequestError as e:                # our request, or a feature this key does not have
            raise LLMError("bad request", e.message) from e
        except sdk.AuthenticationError as e:
            raise LLMError("auth", e.message) from e
        except sdk.PermissionDeniedError as e:
            raise LLMError("permission", e.message) from e
        except sdk.NotFoundError as e:                  # a model id this key cannot use
            raise LLMError("not found", e.message) from e
        except sdk.RateLimitError as e:                 # still limited after the SDK's own retries
            raise LLMError("rate limit", e.message) from e
        except sdk.APIStatusError as e:
            raise LLMError(f"status {e.status_code}", e.message) from e
        except sdk.APIConnectionError as e:             # includes timeouts
            raise LLMError("connection", str(e)) from e
        except sdk.APIError as e:
            raise LLMError("sdk", str(e)) from e
        return read(message, ask)


def read(message, ask: Ask) -> Reply:
    """A response as the desk needs it. Fields are read through _get: the beta and the plain message
    differ, the SDK hands a block it has no type for over as plain data, and a test's stand-in has only
    what it needs."""
    usage = message.usage
    content = list(message.content or [])
    searches = int(_get(_get(usage, "server_tool_use"), "web_search_requests") or 0)
    attempts = [i for i in _get(usage, "iterations") or [] if _get(i, "type") in ("message", "fallback_message")]
    served = (next((_get(i, "model") for i in reversed(attempts) if _get(i, "type") == "fallback_message" and _get(i, "model")), None)
              or next((_get(_get(b, "to"), "model") for b in reversed(content) if _get(b, "type") == "fallback"), None)
              or _get(message, "model") or ask.model)
    top = Bill(served, usage.input_tokens or 0, usage.output_tokens or 0, _get(usage, "cache_read_input_tokens") or 0,
               _get(usage, "cache_creation_input_tokens") or 0, searches)
    stop = message.stop_reason or ""
    refusal = (_get(_get(message, "stop_details"), "category") or "no category given") if stop == "refusal" else None
    return Reply(stop=stop, data=_answer(content) if stop == "end_turn" else None, top=top,
                 parts=_per_model(attempts, served, searches), refusal=refusal, urls=_urls(content), content=content,
                 container=_get(_get(message, "container"), "id"))


def _get(obj, name: str):
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _per_model(attempts: list, served: str, searches: int) -> list[Bill]:
    """The attempts' usage summed per model. A model that declined and the one that answered are billed
    at their own rates, and the response's top-level usage shows only the one that answered."""
    by: dict[str, Bill] = {}
    for i in attempts:
        model = _get(i, "model") or served
        b = by.get(model, Bill(model))
        by[model] = Bill(model, b.in_tok + (_get(i, "input_tokens") or 0), b.out_tok + (_get(i, "output_tokens") or 0),
                         b.cache_read + (_get(i, "cache_read_input_tokens") or 0),
                         b.cache_write + (_get(i, "cache_creation_input_tokens") or 0))
    if by and searches:
        payer = served if served in by else next(reversed(by))
        by[payer] = replace(by[payer], searches=searches)
    return list(by.values())


def _answer(content: list) -> dict | None:
    """The JSON object of a finished turn. It arrives as text, but a search turn can spread the text over
    several blocks and a fallback splits it where one model handed over to the next, so the text after
    the last tool block is joined before it is parsed."""
    texts, tail = [], []
    for block in content:
        kind = _get(block, "type")
        if kind == "text":
            texts.append(_get(block, "text") or "")
            tail.append(texts[-1])
        elif kind not in ("thinking", "redacted_thinking", "fallback"):
            tail = []
    for candidate in ("".join(tail), texts[-1] if texts else "", "".join(texts)):
        try:
            obj = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _urls(content: list) -> dict[str, str]:
    """The pages a search returned and the model cited, so a read can list only sources that exist."""
    out: dict[str, str] = {}
    for block in content:
        kind, found = _get(block, "type"), []
        if kind == "web_search_tool_result" and isinstance(_get(block, "content"), list):   # an error is an object, not a list
            found = _get(block, "content")
        elif kind == "text":
            found = _get(block, "citations") or []
        for item in found:
            url = _get(item, "url")
            if url:
                out.setdefault(url, _get(item, "title") or "")
    return out
