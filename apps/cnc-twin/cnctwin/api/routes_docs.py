"""/api/docs: the build's documentation as Markdown (read-only), so that a report's links reach the full documents
(the report's "Model approximations" section links the full table, /api/docs/approximations.md)."""

from __future__ import annotations

from fastapi import FastAPI
from starlette.responses import Response

from .models import errors
from .safety import ApiError, resolve_inside
from .views import ApiContext, file_response

MARKDOWN = {"text/markdown": {"schema": {"type": "string"}}}


def register(app: FastAPI, ctx: ApiContext) -> None:
    @app.get("/api/docs/{path:path}", tags=["health"], response_class=Response,
             responses={200: {"content": MARKDOWN, "description": "The Markdown document"}, **errors(400, 403, 404)})
    def get_doc(path: str):
        """A document of the build's docs/ folder (``approximations.md``, ``models/loop.md`` ...) as Markdown text."""
        if not path.endswith(".md"):
            raise ApiError(404, f"no document {path!r}: the documents are the .md files of docs/")
        return file_response(resolve_inside(ctx.settings.root / "docs", path))


__all__ = ["register"]
