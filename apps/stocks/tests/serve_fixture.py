"""Serves the page over the fixtures, the way Caddy will serve it over the app. No dependencies.

    python tests/serve_fixture.py [port] [fixture]      fixture: running (default) | keys | stopped

`?fixture=keys` on a request overrides the default for that page load (the page's fetches inherit it through
the Referer). Caddy's part is copied here: the three theme tags are slipped in before </head>, and
/_plint/* comes from the repo's theme/ folder."""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
APP = HERE.parent
THEME = APP.parent.parent / "theme"
FIX = HERE / "fixtures"
INJECT = ('<link rel="stylesheet" href="/_plint/plint.css"><link rel="stylesheet" href="/_plint/apps/stocks.css">'
          '<script src="/_plint/plint.js" data-app="stocks" defer></script></head>')
TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css", ".js": "application/javascript", ".json": "application/json",
         ".woff2": "font/woff2", ".svg": "image/svg+xml", ".png": "image/png"}
DEFAULT = sys.argv[2] if len(sys.argv) > 2 else "running"


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _fixture(self) -> str:
        for src in (self.path, self.headers.get("Referer", "")):
            q = parse_qs(urlparse(src).query).get("fixture")
            if q:
                return q[0]
        return DEFAULT

    def _file(self, path: Path) -> None:
        if not path.is_file():
            return self._send(404, b"not found", "text/plain")
        body = path.read_bytes()
        if path.suffix == ".html":
            body = body.replace(b"</head>", INJECT.encode(), 1)
        self._send(200, body, TYPES.get(path.suffix, "application/octet-stream"))

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        fx = self._fixture()
        suffix = "" if fx == "running" else f"_{fx}"
        if path == "/api/state":
            return self._file(FIX / f"state{suffix}.json")
        if path.startswith("/api/model/"):
            mid = path.rsplit("/", 1)[1]
            p = FIX / f"model_{mid}{suffix}.json"
            return self._file(p if p.is_file() else FIX / f"model_{mid}.json")
        if path == "/health":
            return self._send(200, b"ok", "text/plain")
        if path.startswith("/_plint/"):
            return self._file((THEME / path[len("/_plint/"):]).resolve())
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        target = (APP / "static" / rel).resolve()
        if APP / "static" not in target.parents and target != APP / "static" / "index.html":
            return self._send(404, b"not found", "text/plain")
        self._file(target)

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5258
    print(f"fixture page on :{port} ({DEFAULT})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
