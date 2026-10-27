"""A tiny status server: the page, its JSON, and a health check."""
from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .store import Store

log = logging.getLogger("polybot.web")
STATIC = Path(__file__).resolve().parent.parent / "static"


def make_handler(store: Store, mode: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "polybot"

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def do_HEAD(self):  # noqa: N802
            self.do_GET()

        def do_GET(self):  # noqa: N802
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/status":
                status = store.read_status() or {"mode": mode, "starting": True, "signals": [], "positions": [], "trades": [], "leaders": []}
                self._send(200, json.dumps(status).encode(), "application/json")
            elif path == "/health":
                self._send(200, b"ok", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

        def log_message(self, fmt, *args):  # keep docker logs for the bot itself
            log.debug(fmt, *args)

    return Handler


def serve(port: int, store: Store, mode: str) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer(("0.0.0.0", port), make_handler(store, mode))
    threading.Thread(target=httpd.serve_forever, name="web", daemon=True).start()
    log.info("status page on :%d", port)
    return httpd
