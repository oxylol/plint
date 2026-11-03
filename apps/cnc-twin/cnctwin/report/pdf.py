"""PDF through WeasyPrint, deterministically: the same HTML gives the same bytes.

WeasyPrint writes no creation or modification date unless the HTML carries ``dcterms.created`` /
``dcterms.modified`` (the report never does) and no random file identifier, so the bytes depend only
on the HTML, the fonts and the WeasyPrint version. Warnings about CSS that only screens use
(``overflow-x``, ``-webkit-*``) are silenced: the print rules of the page replace them.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

_QUIET = ("weasyprint", "fontTools", "fontTools.subset", "fontTools.ttLib")


def render_pdf(html_text: str, base_url: str | None = None) -> tuple[bytes, int]:
    """(PDF bytes, page count)."""
    from weasyprint import HTML
    loggers = [logging.getLogger(name) for name in _QUIET]
    levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.ERROR)
    try:
        document = HTML(string=html_text, base_url=base_url).render()
        data = document.write_pdf()
        return data, len(document.pages)
    finally:
        for logger, level in zip(loggers, levels):
            logger.setLevel(level)


def write_atomic(path: Path, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def write_pdf(html_text: str, path: Path, base_url: str | None = None) -> int:
    data, pages = render_pdf(html_text, base_url)
    write_atomic(path, data)
    return pages


__all__ = ["render_pdf", "write_pdf", "write_atomic"]
