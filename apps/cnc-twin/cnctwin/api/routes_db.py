"""/api/db: the databases (machines, tools, holders, materials, coolants ...) and the bibliography."""

from __future__ import annotations

from fastapi import FastAPI

from .jsonio import json_response
from .models import DbItem, DbKinds, DbList, errors
from .views import ApiContext


def register(app: FastAPI, ctx: ApiContext) -> None:
    @app.get("/api/db", tags=["database"], response_model=DbKinds)
    def db_kinds():
        """The database kinds and how many items each has."""
        return json_response(ctx.db.kinds())

    @app.get("/api/db/{kind}", tags=["database"], response_model=DbList, responses=errors(400, 404))
    def db_list(kind: str):
        """Items of one kind (``machines`` or ``machine``, ``tools`` ...; ``bibliography``)."""
        return json_response(ctx.db.list(kind))

    @app.get("/api/db/{kind}/{item_id}", tags=["database"], response_model=DbItem, responses=errors(400, 404))
    def db_item(kind: str, item_id: str):
        """One item as written, its SI view and problems, and the bibliography entries it cites."""
        return json_response(ctx.db.get(kind, item_id))
