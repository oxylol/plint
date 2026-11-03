"""Uploaded project archives: the request body to a file, and a zip to a folder, safely.

The body is either the zip itself (``Content-Type: application/zip`` or ``application/octet-stream``)
or ``multipart/form-data`` with the zip in a file field (an HTML form). Both are streamed to disk
with a size cap, so a large upload never sits in memory. The multipart parser is a small streaming
state machine (RFC 7578; the python-multipart package is not a dependency).

Extraction refuses, before writing anything: absolute names, ``..`` segments, drive letters,
symbolic links, encrypted members, duplicate names, more than ``max_files`` members, more than
``max_bytes`` in total, and members above 1 MiB that expand more than ``max_ratio`` times (zip
bombs). While writing, each member is counted again (a zip header can lie about its size).
macOS archive metadata (``__MACOSX/`` and ``.DS_Store``) is dropped and listed in the report.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from .safety import ApiError

ZIP_TYPES = ("application/zip", "application/x-zip-compressed", "application/octet-stream", "")
_PARAM = re.compile(r';\s*([A-Za-z0-9_-]+)\s*=\s*("(?:[^"\\]|\\.)*"|[^;]*)')
_RATIO_FLOOR_BYTES = 1024 * 1024
_CHUNK = 1024 * 1024
_METADATA_DIRS = ("__MACOSX",)
_METADATA_FILES = (".DS_Store",)


def header_params(value: str) -> tuple[str, dict]:
    """``'form-data; name="file"; filename="a.zip"'`` -> ("form-data", {"name": "file", ...})."""
    main = value.split(";", 1)[0].strip().lower()
    params = {}
    for key, raw in _PARAM.findall(value):
        raw = raw.strip()
        if raw.startswith('"') and raw.endswith('"'):
            raw = re.sub(r"\\(.)", r"\1", raw[1:-1])
        params[key.lower()] = raw
    return main, params


class MultipartError(ApiError):
    def __init__(self, message: str) -> None:
        super().__init__(400, f"multipart body: {message}")


class MultipartFileSink:
    """Streams a multipart/form-data body: the first file part goes to ``out``, small text fields
    are kept (``fields``), other parts are discarded."""

    MAX_HEADER_BYTES = 16 * 1024
    MAX_FIELD_BYTES = 64 * 1024

    def __init__(self, boundary: str, out, file_fields=("file", "project", "zip", "archive")) -> None:
        if not boundary or len(boundary) > 200:
            raise MultipartError("missing or invalid boundary")
        self.delimiter = b"--" + boundary.encode("latin-1")
        self.out = out
        self.file_fields = file_fields
        self.buffer = b""
        self.state = "preamble"
        self.part: dict | None = None
        self.fields: dict[str, str] = {}
        self.filename: str | None = None
        self.file_bytes = 0
        self.captured = False
        self._field_value = bytearray()

    def feed(self, data: bytes) -> None:
        self.buffer += data
        while True:
            if self.state == "preamble":
                at = self.buffer.find(self.delimiter)
                if at < 0:
                    self.buffer = self.buffer[-len(self.delimiter):]
                    return
                self.buffer = self.buffer[at + len(self.delimiter):]
                self.state = "after_delimiter"
            elif self.state == "after_delimiter":
                if len(self.buffer) < 2:
                    return
                if self.buffer.startswith(b"--"):
                    self.state = "done"
                    self.buffer = b""
                    return
                end = self.buffer.find(b"\r\n")
                if end < 0:
                    if len(self.buffer) > 1024:
                        raise MultipartError("garbage after a boundary")
                    return
                if self.buffer[:end].strip(b" \t"):
                    raise MultipartError("garbage after a boundary")
                self.buffer = self.buffer[end + 2:]
                self.state = "headers"
            elif self.state == "headers":
                end = self.buffer.find(b"\r\n\r\n")
                if end < 0:
                    if len(self.buffer) > self.MAX_HEADER_BYTES:
                        raise MultipartError("part headers too long")
                    return
                self._start_part(self.buffer[:end].decode("latin-1"))
                self.buffer = self.buffer[end + 4:]
                self.state = "body"
            elif self.state == "body":
                marker = b"\r\n" + self.delimiter
                at = self.buffer.find(marker)
                if at < 0:
                    keep = len(marker) - 1
                    if len(self.buffer) > keep:
                        self._part_data(self.buffer[:-keep])
                        self.buffer = self.buffer[-keep:]
                    return
                self._part_data(self.buffer[:at])
                self._end_part()
                self.buffer = self.buffer[at + len(marker):]
                self.state = "after_delimiter"
            else:
                self.buffer = b""
                return

    def _start_part(self, headers_text: str) -> None:
        headers = {}
        for line in headers_text.split("\r\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip().lower()] = value.strip()
        disposition, params = header_params(headers.get("content-disposition", ""))
        if disposition != "form-data" or "name" not in params:
            raise MultipartError("a part without Content-Disposition: form-data; name=...")
        name = params["name"]
        is_file = "filename" in params or name in self.file_fields
        target = "file" if (is_file and not self.captured) else ("field" if not is_file else "skip")
        self.part = {"name": name, "target": target, "filename": params.get("filename")}
        self._field_value = bytearray()

    def _part_data(self, data: bytes) -> None:
        if not data or self.part is None:
            return
        if self.part["target"] == "file":
            self.out.write(data)
            self.file_bytes += len(data)
        elif self.part["target"] == "field":
            if len(self._field_value) + len(data) > self.MAX_FIELD_BYTES:
                raise MultipartError(f"form field {self.part['name']!r} is too long")
            self._field_value += data

    def _end_part(self) -> None:
        if self.part is None:
            return
        if self.part["target"] == "file":
            self.captured = True
            self.filename = self.part["filename"]
        elif self.part["target"] == "field":
            self.fields[self.part["name"]] = self._field_value.decode("utf-8", "replace")
        self.part = None

    def close(self) -> None:
        if self.state != "done":
            raise MultipartError("the body ended before the closing boundary")
        if not self.captured:
            raise MultipartError("no file part (send the zip in a field named 'file')")


@dataclass
class SavedUpload:
    path: Path
    bytes: int
    filename: str | None
    fields: dict = field(default_factory=dict)


async def save_request_body(request, target: Path, max_bytes: int) -> SavedUpload:
    """Stream the request's zip (raw or multipart) to ``target``; 413 above ``max_bytes``."""
    content_type = request.headers.get("content-type", "")
    main, params = header_params(content_type)
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes + 64 * 1024:
        raise ApiError(413, f"upload of {int(declared)} bytes exceeds the limit of {max_bytes} bytes")
    filename = None
    disposition = request.headers.get("content-disposition")
    if disposition:
        filename = header_params(disposition)[1].get("filename")
    target.parent.mkdir(parents=True, exist_ok=True)
    received = 0
    with open(target, "wb") as out:
        if main == "multipart/form-data":
            sink = MultipartFileSink(params.get("boundary", ""), out)
            async for chunk in request.stream():
                received += len(chunk)
                if received > max_bytes + 64 * 1024:
                    raise ApiError(413, f"upload exceeds the limit of {max_bytes} bytes")
                sink.feed(chunk)
            sink.close()
            size = sink.file_bytes
            filename = sink.filename or filename
            fields_ = sink.fields
        elif main in ZIP_TYPES:
            async for chunk in request.stream():
                received += len(chunk)
                if received > max_bytes:
                    raise ApiError(413, f"upload exceeds the limit of {max_bytes} bytes")
                out.write(chunk)
            size = received
            fields_ = {}
        else:
            raise ApiError(415, f"unsupported Content-Type {content_type!r}: send application/zip "
                                f"or multipart/form-data")
    if size > max_bytes:
        raise ApiError(413, f"upload exceeds the limit of {max_bytes} bytes")
    if size == 0:
        raise ApiError(400, "empty upload")
    return SavedUpload(target, size, filename, fields_)


@dataclass
class ExtractReport:
    files: list[str]
    skipped: list[str]
    total_bytes: int


def _member_name(info: zipfile.ZipInfo) -> str:
    name = info.filename.replace("\\", "/")
    if "\x00" in name:
        raise ApiError(400, "archive member name contains a NUL byte")
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise ApiError(400, f"archive member {info.filename!r} has an absolute path")
    parts = [p for p in name.split("/")]
    if parts and parts[-1] == "":
        parts = parts[:-1]
    for part in parts:
        if part in ("..", "."):
            raise ApiError(400, f"archive member {info.filename!r} leaves the archive folder ('..')")
        if part == "":
            raise ApiError(400, f"archive member {info.filename!r} has an empty path segment")
    return "/".join(parts)


def _is_metadata(name: str) -> bool:
    parts = name.split("/")
    return parts[0] in _METADATA_DIRS or parts[-1] in _METADATA_FILES or parts[-1].startswith("._")


def extract_zip(archive: Path, dest: Path, *, max_bytes: int, max_files: int, max_ratio: float) -> ExtractReport:
    """Extract ``archive`` into the (new, empty) folder ``dest`` after checking every member."""
    try:
        zf = zipfile.ZipFile(archive)
    except zipfile.BadZipFile:
        raise ApiError(400, "the upload is not a zip archive") from None
    with zf:
        infos = zf.infolist()
        if len(infos) > max_files:
            raise ApiError(413, f"archive has {len(infos)} members, more than the limit of {max_files}")
        plan: list[tuple[zipfile.ZipInfo, str]] = []
        skipped: list[str] = []
        seen: set[str] = set()
        total = 0
        for info in infos:
            name = _member_name(info)
            if not name:
                continue
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ApiError(400, f"archive member {info.filename!r} is a symbolic link")
            if info.flag_bits & 0x1:
                raise ApiError(400, f"archive member {info.filename!r} is encrypted")
            if _is_metadata(name):
                skipped.append(name)
                continue
            if name in seen:
                raise ApiError(400, f"archive member {name!r} appears twice")
            seen.add(name)
            if info.is_dir():
                plan.append((info, name))
                continue
            total += info.file_size
            if total > max_bytes:
                raise ApiError(413, f"archive expands to more than {max_bytes} bytes")
            if info.file_size > _RATIO_FLOOR_BYTES and info.file_size > max_ratio * max(info.compress_size, 1):
                raise ApiError(413, f"archive member {name!r} expands {info.file_size / max(info.compress_size, 1):.0f}"
                                    f" times (limit {max_ratio:.0f}): refusing a possible zip bomb")
            plan.append((info, name))
        file_names = {name for info, name in plan if not info.is_dir()}
        for _, name in plan:
            parts = name.split("/")
            for i in range(1, len(parts)):
                parent = "/".join(parts[:i])
                if parent in file_names:
                    raise ApiError(400, f"archive member {parent!r} is both a file and a folder")
        dest.mkdir(parents=True, exist_ok=False)
        written = 0
        files: list[str] = []
        for info, name in plan:
            target = dest.joinpath(*name.split("/"))
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            member_bytes = 0
            try:
                with zf.open(info) as source, open(target, "xb") as out:
                    while True:
                        chunk = source.read(_CHUNK)
                        if not chunk:
                            break
                        member_bytes += len(chunk)
                        written += len(chunk)
                        if member_bytes > info.file_size or written > max_bytes:
                            raise ApiError(413, f"archive member {name!r} is larger than its header says")
                        out.write(chunk)
            except (zipfile.BadZipFile, zlib.error, EOFError, NotImplementedError) as error:
                raise ApiError(400, f"archive member {name!r} cannot be read: {error}") from None
            files.append(name)
    return ExtractReport(sorted(files), sorted(skipped), written)


def flatten_single_folder(dest: Path) -> None:
    """If everything sits in one top folder (a zipped project folder), move it up a level."""
    entries = [p for p in dest.iterdir()]
    if len(entries) == 1 and entries[0].is_dir() and not (dest / "project.yaml").exists():
        inner = entries[0]
        holding = dest / f".flatten-{os.getpid()}"
        inner.rename(holding)
        for child in sorted(holding.iterdir()):
            child.rename(dest / child.name)
        holding.rmdir()


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
