"""A tiny server: the page, its two JSON endpoints, a health check."""
from __future__ import annotations

import json
import logging
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .config import Config
from .store import Store

log = logging.getLogger("stocks.web")
STATIC = Path(__file__).resolve().parent.parent / "static"
TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "application/javascript; charset=utf-8",
         ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon"}
MODEL_ID = re.compile(r"^[a-z0-9_]{1,24}$")


def starting(cfg: Config) -> dict:
    """What the page gets before the first run has written anything."""
    return {"app": "stocks", "generated": None, "asof": None, "phase": "keys" if not cfg.has_market_data else "backfill",
            "phase_text": "Starting", "keys": cfg.keys(), "notices": [], "clock": {"last_run": None, "next_run": None, "busy": "starting"},
            "start_cash": cfg.start_cash, "index": {"symbol": cfg.index_symbol, "title": cfg.index_symbol},
            "spend": {"month": "", "used": 0.0, "cap": cfg.agent_cap_usd, "today": 0.0, "calls": 0, "stopped": None, "by_day": []},
            "models": [], "log": []}


def make_handler(store: Store, cfg: Config):
    class Handler(BaseHTTPRequestHandler):
        server_version = "stocks"

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            try:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass        # the client (often Docker's health probe) hung up first

        def handle_error(self, request, client_address):  # noqa: D401
            pass

        def do_HEAD(self):  # noqa: N802
            self.do_GET()

        def do_GET(self):  # noqa: N802
            path = urlparse(self.path).path
            if path == "/health":
                return self._send(200, b"ok", "text/plain")
            if path == "/api/state":
                return self._send(200, json.dumps(store.read_json("state.json") or starting(cfg)).encode(), "application/json")
            if path.startswith("/api/model/"):
                mid = path[len("/api/model/"):]
                got = store.read_json(f"model_{mid}.json") if MODEL_ID.match(mid) else None
                if got is None:
                    return self._send(404, b'{"error":"no such model"}', "application/json")
                return self._send(200, json.dumps(got).encode(), "application/json")
            rel = "index.html" if path in ("/", "") else path.lstrip("/")
            target = (STATIC / rel).resolve()
            if STATIC.resolve() not in target.parents or not target.is_file():
                return self._send(404, b"not found", "text/plain")
            self._send(200, target.read_bytes(), TYPES.get(target.suffix, "application/octet-stream"))

        def log_message(self, fmt, *args):
            log.debug(fmt, *args)

    return Handler


def serve(cfg: Config, store: Store) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer(("0.0.0.0", cfg.port), make_handler(store, cfg))
    threading.Thread(target=httpd.serve_forever, name="web", daemon=True).start()
    log.info("page on :%d", cfg.port)
    return httpd
