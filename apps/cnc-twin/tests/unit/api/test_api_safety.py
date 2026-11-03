"""Path traversal, ids, zip extraction and the multipart parser."""

from __future__ import annotations

import io
import os
import stat
import warnings
import zipfile

import pytest

from cnctwin.api.safety import ApiError, check_id, resolve_inside, safe_relpath, slugify
from cnctwin.api.upload import MultipartFileSink, extract_zip, flatten_single_folder, header_params

LIMITS = dict(max_bytes=10 * 1024 * 1024, max_files=100, max_ratio=200.0)


def make_zip(path, members, *, symlink=None, encrypted=None, compress=zipfile.ZIP_DEFLATED):
    with zipfile.ZipFile(path, "w", compression=compress) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        if symlink:
            info = zipfile.ZipInfo(symlink[0])
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, symlink[1])
    if encrypted:
        data = bytearray(path.read_bytes())
        offset = data.find(b"PK\x01\x02")
        while offset >= 0:          # set the encryption flag in the central directory
            name_len = int.from_bytes(data[offset + 28:offset + 30], "little")
            name = bytes(data[offset + 46:offset + 46 + name_len]).decode()
            if name == encrypted:
                data[offset + 8] |= 0x01
            offset = data.find(b"PK\x01\x02", offset + 4)
        path.write_bytes(bytes(data))
    return path


@pytest.mark.parametrize("value", ["run-1", "00001-pocket", "a.b_c", "X" * 128])
def test_valid_ids(value):
    assert check_id(value) == value


@pytest.mark.parametrize("value", ["", ".", "..", "-a", "a/b", "a b", "../x", "x" * 129, "ä"])
def test_invalid_ids(value):
    with pytest.raises(ApiError) as error:
        check_id(value)
    assert error.value.status == 400


@pytest.mark.parametrize("value", ["../etc/passwd", "a/../../b", "/etc/passwd", "a//b", "a/./b", "a\\b", "C:/x",
                                   "a\x00b", "", "a/"])
def test_relative_paths_that_could_escape_are_refused(value):
    with pytest.raises(ApiError) as error:
        safe_relpath(value)
    assert error.value.status == 400


def test_resolve_inside_follows_links_and_refuses_escapes(tmp_path):
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "f.txt").write_text("x")
    (tmp_path / "secret.txt").write_text("s")
    os.symlink(tmp_path / "secret.txt", root / "escape.txt")
    os.symlink(root / "sub" / "f.txt", root / "inner.txt")
    os.symlink(tmp_path, root / "up")
    assert resolve_inside(root, "sub/f.txt") == (root / "sub" / "f.txt").resolve()
    assert resolve_inside(root, "inner.txt") == (root / "sub" / "f.txt").resolve()
    for path, status in (("escape.txt", 403), ("up/secret.txt", 403), ("nothing.txt", 404), ("sub", 404)):
        with pytest.raises(ApiError) as error:
            resolve_inside(root, path)
        assert error.value.status == status, path
    assert resolve_inside(root, "sub", want="dir").is_dir()


def test_slugify():
    assert slugify("Pocket with islands, Al 6061-T6") == "pocket-with-islands-al-6061-t6"
    assert slugify("!!!") == "project"
    assert len(slugify("x" * 100)) == 48


def test_extract_good_archive_and_flatten(tmp_path):
    archive = make_zip(tmp_path / "p.zip", {"pocket/project.yaml": "format: cnctwin-project\n",
                                            "pocket/op10.nc": "G0 X0\n", "pocket/cad/": "",
                                            "__MACOSX/pocket/._op10.nc": "junk", "pocket/.DS_Store": "junk"})
    report = extract_zip(archive, tmp_path / "out", **LIMITS)
    assert report.files == ["pocket/op10.nc", "pocket/project.yaml"]
    assert report.skipped == ["__MACOSX/pocket/._op10.nc", "pocket/.DS_Store"]
    flatten_single_folder(tmp_path / "out")
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["cad", "op10.nc", "project.yaml"]


@pytest.mark.parametrize("members, message", [
    ({"../evil.txt": "x"}, "leaves the archive"),
    ({"a/../../evil.txt": "x"}, "leaves the archive"),
    ({"/etc/evil.txt": "x"}, "absolute"),
    ({"C:/evil.txt": "x"}, "absolute"),
    ({"a\\..\\..\\evil.txt": "x"}, "leaves the archive"),
    ({"a//b.txt": "x"}, "empty path segment"),
    ({"a": "file", "a/b": "inside"}, "both a file and a folder"),
])
def test_zip_slip_and_bad_names_are_refused(tmp_path, members, message):
    archive = make_zip(tmp_path / "bad.zip", members)
    with pytest.raises(ApiError) as error:
        extract_zip(archive, tmp_path / "out", **LIMITS)
    assert error.value.status == 400 and message in error.value.message
    assert not (tmp_path / "out").exists() and not (tmp_path / "evil.txt").exists()


def test_duplicate_symlink_and_encrypted_members_are_refused(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)        # zipfile warns about the duplicate we want
        with zipfile.ZipFile(tmp_path / "dup.zip", "w") as archive:
            archive.writestr("a.txt", "1")
            archive.writestr("a.txt", "2")
    cases = [(tmp_path / "dup.zip", "twice"),
             (make_zip(tmp_path / "link.zip", {"a.txt": "x"}, symlink=("link", "/etc/passwd")), "symbolic link"),
             (make_zip(tmp_path / "enc.zip", {"a.txt": "secret"}, encrypted="a.txt"), "encrypted")]
    for archive, message in cases:
        with pytest.raises(ApiError) as error:
            extract_zip(archive, tmp_path / f"out-{archive.stem}", **LIMITS)
        assert message in error.value.message


def test_size_count_and_ratio_limits(tmp_path):
    bomb = make_zip(tmp_path / "bomb.zip", {"zeros.bin": b"\0" * (8 * 1024 * 1024)})
    with pytest.raises(ApiError) as error:
        extract_zip(bomb, tmp_path / "o1", **LIMITS)
    assert error.value.status == 413 and "zip bomb" in error.value.message
    big = make_zip(tmp_path / "big.zip", {"a.bin": os.urandom(600_000), "b.bin": os.urandom(600_000)},
                   compress=zipfile.ZIP_STORED)
    with pytest.raises(ApiError) as error:
        extract_zip(big, tmp_path / "o2", max_bytes=1_000_000, max_files=100, max_ratio=200.0)
    assert error.value.status == 413
    many = make_zip(tmp_path / "many.zip", {f"f{i}.txt": "x" for i in range(5)})
    with pytest.raises(ApiError) as error:
        extract_zip(many, tmp_path / "o3", max_bytes=10_000, max_files=4, max_ratio=200.0)
    assert error.value.status == 413 and "members" in error.value.message


def test_a_lying_size_header_is_caught_while_writing(tmp_path):
    archive = make_zip(tmp_path / "lie.zip", {"a.txt": b"0123456789" * 100}, compress=zipfile.ZIP_STORED)
    data = bytearray(archive.read_bytes())
    central = data.find(b"PK\x01\x02")
    data[central + 24:central + 28] = (10).to_bytes(4, "little")      # uncompressed size: 10
    archive.write_bytes(bytes(data))
    with pytest.raises(ApiError) as error:
        extract_zip(archive, tmp_path / "out", **LIMITS)
    assert error.value.status in (400, 413)


def test_not_a_zip(tmp_path):
    (tmp_path / "x.zip").write_bytes(b"not a zip at all")
    with pytest.raises(ApiError) as error:
        extract_zip(tmp_path / "x.zip", tmp_path / "out", **LIMITS)
    assert error.value.status == 400


def multipart_body(boundary: str, parts) -> bytes:
    out = io.BytesIO()
    for name, filename, data in parts:
        out.write(f"--{boundary}\r\n".encode())
        disposition = f'form-data; name="{name}"' + (f'; filename="{filename}"' if filename else "")
        out.write(f"Content-Disposition: {disposition}\r\n".encode())
        if filename:
            out.write(b"Content-Type: application/zip\r\n")
        out.write(b"\r\n" + data + b"\r\n")
    out.write(f"--{boundary}--\r\n".encode())
    return out.getvalue()


@pytest.mark.parametrize("chunk", [1, 7, 64, 100000])
def test_multipart_parser_in_any_chunking(chunk):
    payload = os.urandom(5000) + b"\r\n--notquite" + os.urandom(100)
    body = multipart_body("XyZ123", [("name", None, b"my-pocket"), ("file", "p.zip", payload),
                                     ("other", "x.bin", b"ignored")])
    out = io.BytesIO()
    sink = MultipartFileSink("XyZ123", out)
    for i in range(0, len(body), chunk):
        sink.feed(body[i:i + chunk])
    sink.close()
    assert out.getvalue() == payload and sink.filename == "p.zip" and sink.fields == {"name": "my-pocket"}


def test_multipart_errors():
    out = io.BytesIO()
    sink = MultipartFileSink("B", out)
    sink.feed(multipart_body("B", [("name", None, b"x")]))
    with pytest.raises(ApiError, match="no file part"):
        sink.close()
    sink = MultipartFileSink("B", io.BytesIO())
    sink.feed(b"--B\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a\"\r\n\r\nabc")
    with pytest.raises(ApiError, match="ended before"):
        sink.close()
    with pytest.raises(ApiError, match="boundary"):
        MultipartFileSink("", io.BytesIO())
    sink = MultipartFileSink("B", io.BytesIO())
    with pytest.raises(ApiError, match="Content-Disposition"):
        sink.feed(b"--B\r\nContent-Type: text/plain\r\n\r\nabc\r\n--B--\r\n")


def test_header_params():
    assert header_params('form-data; name="file"; filename="a \\"b\\".zip"') == \
        ("form-data", {"name": "file", "filename": 'a "b".zip'})
    assert header_params("multipart/form-data; boundary=abc")[1]["boundary"] == "abc"
