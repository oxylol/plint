"""Names and paths that come from a request: ids, relative file paths, and where they may lead.

Every file the API serves is addressed by a relative path inside a root (a run directory, a
project, the UI folder). :func:`resolve_inside` refuses absolute paths, ``..`` and ``.``
segments, empty segments, backslashes and NUL bytes, then resolves symbolic links and refuses a
result outside the root, so neither the request nor a link inside the folder can reach elsewhere.
"""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath

ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DRIVE = re.compile(r"^[A-Za-z]:")


class ApiError(Exception):
    """An error with its HTTP status; the app turns it into ``{"detail": message, ...extra}``."""

    def __init__(self, status: int, message: str, **extra) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.extra = extra


def check_id(value: str, what: str = "id") -> str:
    if not isinstance(value, str) or not ID_PATTERN.match(value):
        raise ApiError(400, f"invalid {what} {value!r}: use letters, digits, '.', '_' and '-' "
                            f"(starting with a letter or digit, at most 128 characters)")
    return value


def safe_relpath(relative: str) -> PurePosixPath:
    """The relative path as segments, or ApiError 400 if it could leave its root."""
    if not isinstance(relative, str) or not relative:
        raise ApiError(400, "empty path")
    if "\x00" in relative:
        raise ApiError(400, "invalid path: NUL byte")
    if "\\" in relative:
        raise ApiError(400, f"invalid path {relative!r}: use '/' as the separator")
    if relative.startswith("/") or _DRIVE.match(relative):
        raise ApiError(400, f"invalid path {relative!r}: must be relative")
    parts = relative.split("/")
    for part in parts:
        if part in ("", ".", ".."):
            raise ApiError(400, f"invalid path {relative!r}: empty, '.' or '..' segment")
    return PurePosixPath(*parts)


def resolve_inside(root: Path, relative: str, *, must_exist: bool = True, want: str = "file") -> Path:
    """Absolute path of ``relative`` under ``root`` (symbolic links resolved and checked)."""
    parts = safe_relpath(relative).parts
    root_real = os.path.realpath(root)
    target = os.path.realpath(os.path.join(root_real, *parts))
    if os.path.commonpath([root_real, target]) != root_real:
        raise ApiError(403, f"path {relative!r} leads outside its folder")
    if must_exist:
        if want == "file" and not os.path.isfile(target):
            raise ApiError(404, f"no file {relative!r}")
        if want == "dir" and not os.path.isdir(target):
            raise ApiError(404, f"no folder {relative!r}")
        if want == "any" and not os.path.exists(target):
            raise ApiError(404, f"no file or folder {relative!r}")
    return Path(target)


def is_inside(root: Path, path: Path) -> bool:
    root_real = os.path.realpath(root)
    target = os.path.realpath(path)
    return os.path.commonpath([root_real, target]) == root_real


def list_files(root: Path, *, limit: int = 100000) -> list[dict]:
    """Files under ``root`` (relative POSIX paths, sizes), sorted, links pointing outside left out."""
    root = Path(root)
    found: list[dict] = []
    if not root.is_dir():
        return found
    for directory, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            path = Path(directory) / name
            if not is_inside(root, path) or not path.is_file():
                continue
            found.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size})
            if len(found) >= limit:
                return found
    return found


def slugify(text: str, *, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    slug = slug[:max_len].strip("-")
    return slug or "project"
