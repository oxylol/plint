"""HTTP API of CNC Twin (docs/architecture.md 17, docs/api.md).

* :func:`create_app` (``settings=None``) -> FastAPI application (routes, job manager, UI at /)
* :func:`serve` (``argv=None``) -> runs it with uvicorn (host, port and DATA_DIR from args or env)
* :func:`export_openapi` (``path=None``) -> the OpenAPI document, written to docs/openapi.json

Importing this package is light (no FastAPI, no NumPy): the job worker
(``python -m cnctwin.api.worker``) lives here too.
"""

from __future__ import annotations

__all__ = ["create_app", "serve", "export_openapi", "Settings"]


def __getattr__(name: str):
    if name == "create_app":
        from .app import create_app
        return create_app
    if name == "Settings":
        from .settings import Settings
        return Settings
    raise AttributeError(f"module 'cnctwin.api' has no attribute {name!r}")


def serve(argv=None) -> int:
    """Run the API server (see cnctwin.api.server)."""
    from .server import serve as _serve
    return _serve(argv)


def export_openapi(path=None) -> dict:
    """Build the OpenAPI document; write it to ``path`` (default docs/openapi.json; False: do not write)."""
    from .openapi import export_openapi as _export
    return _export(path)
