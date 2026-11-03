"""The OpenAPI document of the API, as docs/openapi.json (deterministic: same code, same bytes)."""

from __future__ import annotations

import json
from pathlib import Path

from .. import paths
from .settings import Settings

DEFAULT_PATH = paths.DOCS / "openapi.json"


def openapi_document() -> dict:
    from .app import create_app
    app = create_app(Settings.from_env())          # building the app touches no files
    return app.openapi()


def render(document: dict) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def export_openapi(path=None) -> dict:
    """The document; written to ``path`` (default docs/openapi.json) unless ``path`` is False."""
    document = openapi_document()
    if path is not False:
        target = Path(path) if path is not None else DEFAULT_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render(document), encoding="utf-8")
    return document


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="write the OpenAPI document of the CNC Twin API")
    parser.add_argument("path", nargs="?", default=str(DEFAULT_PATH))
    args = parser.parse_args(argv)
    export_openapi(args.path)
    print(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
