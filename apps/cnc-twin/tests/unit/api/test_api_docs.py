"""GET /api/docs/{path}: the build's Markdown documents (the report links the full approximation table here)."""

from __future__ import annotations

from api_support import ROOT


def test_markdown_documents_are_served_read_only(client):
    response = client.get("/api/docs/approximations.md")
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/markdown")
    assert response.text == (ROOT / "docs" / "approximations.md").read_text(encoding="utf-8")
    assert client.get("/api/docs/models/loop.md").status_code == 200
    assert client.get("/api/docs/openapi.json").status_code == 404           # only the .md documents
    assert client.get("/api/docs/no-such-document.md").status_code == 404
    assert client.get("/api/docs/%2E%2E/README.md").status_code in (400, 403, 404)
