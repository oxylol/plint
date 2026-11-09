"""Program generators of a sweep: a make_program.py style script called with arguments (docs/formats.md "Sweep").

A generator runs in its own process (``python -m cnctwin.sweep.generate <request.json>``), in a copy of the base
project's folder (so whatever the script reads beside itself is there, and nothing it writes reaches the base
project), with that folder first on ``sys.path`` and as the working directory:

* ``call: constants`` (make_program.py style, the default): every top-level assignment of a named constant
  (``TROCHOID_STEP = 1.0``) is replaced by the value before the module runs, so the constants derived from it
  (``T1_F = round(RPM * 3 * T1_FZ)``) follow; the module runs with ``__name__`` not ``"__main__"``, then
  ``function()`` (default ``main``) is called; a constant the script does not assign at top level is an error;
* ``call: kwargs``: ``function(**values)`` is called after the module ran unchanged.

The program is the function's return value when it returns text, else the file ``output`` (default the
operation's program file name) the call wrote in the folder. Standard output and error go to the request's log.
The values are written as Python literals (numbers, text, lists, mappings), so a generator is deterministic when
its script is.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import traceback
import types
from pathlib import Path

from ..core.errors import InputError

RESPONSE = "response.json"
LOG_TAIL = 4000


def inject_constants(source: str, values: dict, filename: str = "<generator>"):
    """The module's AST with every top-level assignment of a name of ``values`` given that value; raises
    InputError naming constants the script does not assign at top level."""
    tree = ast.parse(source, filename=filename)
    found = set()
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            targets = [node.targets[0].id]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            targets = [node.target.id]
        for name in targets:
            if name in values:
                literal = ast.parse(repr(values[name]), mode="eval").body
                ast.copy_location(literal, node.value)
                node.value = literal
                found.add(name)
    missing = sorted(set(values) - found)
    if missing:
        raise InputError(f"{filename}: no top-level assignment of {', '.join(missing)} (call: constants replaces "
                         f"constants the script assigns, e.g. NAME = 1.0)")
    ast.fix_missing_locations(tree)
    return tree


def _literal_ok(value) -> bool:
    try:
        return ast.literal_eval(repr(value)) == value
    except (ValueError, SyntaxError):
        return False


def run_request(request: dict) -> dict:
    """In the generator process: run the script as the request says; returns {program, log}."""
    work = Path(request["work_dir"])
    script = work / request["script"]
    values = dict(request.get("constants") or {})
    values.update(request.get("values") or {})
    log = io.StringIO()
    sys.path.insert(0, str(script.parent))
    os.chdir(script.parent)
    module = types.ModuleType("cnctwin_sweep_generator")
    module.__file__ = str(script)
    source = script.read_text(encoding="utf-8")
    if request.get("call", "constants") == "constants":
        code = compile(inject_constants(source, values, str(script)), str(script), "exec")
    else:
        code = compile(source, str(script), "exec")
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        exec(code, module.__dict__)
        function = getattr(module, request.get("function") or "main", None)
        if not callable(function):
            raise InputError(f"{request['script']} has no function {request.get('function') or 'main'}()")
        result = function(**values) if request.get("call") == "kwargs" else function()
    if isinstance(result, str) and result.strip():
        program = result
    else:
        output = script.parent / request["output"]
        if not output.is_file():
            raise InputError(f"{request['script']}: {request.get('function') or 'main'}() returned no program text and "
                             f"wrote no {request['output']}")
        program = output.read_text(encoding="utf-8")
    return {"program": program, "log": log.getvalue()[-LOG_TAIL:]}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: python -m cnctwin.sweep.generate <request.json>", file=sys.stderr)
        return 2
    request_path = Path(argv[0]).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    response = request_path.with_name(RESPONSE)
    try:
        out = run_request(request)
        out["ok"] = True
    except BaseException as error:      # the driver reads the reason from the response
        out = {"ok": False, "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(limit=20)}
    response.write_text(json.dumps(out), encoding="utf-8")
    return 0 if out["ok"] else 1


# =============================================================================== the driver's side

def _ignore(skip: list[Path]):
    def ignore(directory, names):
        out = {n for n in names if n.startswith(".") or n == "__pycache__"}
        for name in names:
            path = (Path(directory) / name).resolve()
            if any(path == s for s in skip):
                out.add(name)
        return out
    return ignore


_DIGESTS: dict = {}


def _folder_digest(folder: Path, skip: list[Path]) -> str:
    """sha256 of every file the generator's copy of the folder holds (memoised by sizes and mtimes)."""
    files = []
    for directory, dirnames, filenames in os.walk(folder):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "__pycache__"
                             and (Path(directory) / d).resolve() not in skip)
        for name in sorted(filenames):
            if not name.startswith("."):
                path = Path(directory) / name
                st = path.stat()
                files.append((path.relative_to(folder).as_posix(), st.st_size, st.st_mtime_ns))
    signature = (str(folder), tuple(files))
    if signature in _DIGESTS:
        return _DIGESTS[signature]
    digest = hashlib.sha256()
    for relative, _, _ in files:
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256((folder / relative).read_bytes()).digest())
    _DIGESTS[signature] = digest.hexdigest()
    return _DIGESTS[signature]


def generate(generator, base_dir: Path, values: dict, point_dir: Path, *, output: str, skip=(),
             python: str | None = None) -> tuple[str, dict]:
    """Run ``generator`` (spec.Generator) with ``values`` {argument: value} for one point; returns (program text,
    record). The record (``request_sha256``: the script folder's content, the call and the values) lets a resumed
    sweep reuse the program a point already has."""
    base_dir = Path(base_dir).resolve()
    point_dir = Path(point_dir)
    skip = [Path(s).resolve() for s in skip] + [point_dir.resolve()]
    for name, value in list(values.items()) + list(generator.constants.items()):
        if not _literal_ok(value):
            raise InputError(f"generator {generator.name}: {name} = {value!r} cannot be written as a Python literal")
    script = (base_dir / generator.script).resolve()
    try:
        script.relative_to(base_dir)
    except ValueError:
        raise InputError(f"generator {generator.name}: {generator.script} is not in the base project's folder") from None
    if not script.is_file():
        raise InputError(f"generator {generator.name}: no script {script}")
    request = {"script": script.relative_to(base_dir).as_posix(), "function": generator.function,
               "call": generator.call, "output": output, "values": dict(values), "constants": dict(generator.constants)}
    request_sha = hashlib.sha256((json.dumps(request, sort_keys=True, default=repr) + "\n"
                                  + _folder_digest(base_dir, skip)).encode("utf-8")).hexdigest()
    record_path = point_dir / f".generated-{generator.name}.json"
    target = point_dir / output
    if record_path.is_file() and target.is_file():
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except ValueError:
            record = {}
        text = target.read_text(encoding="utf-8")
        if record.get("request_sha256") == request_sha and \
                record.get("program_sha256") == hashlib.sha256(text.encode("utf-8")).hexdigest():
            return text, record
    work_root = point_dir / f".generator-{generator.name}"
    if work_root.exists():
        shutil.rmtree(work_root)
    work = work_root / base_dir.name
    shutil.copytree(base_dir, work, ignore=_ignore(skip), symlinks=True)
    request["work_dir"] = str(work)
    request_path = work_root / "request.json"
    request_path.write_text(json.dumps(request, default=repr), encoding="utf-8")
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    try:
        completed = subprocess.run([python or sys.executable, "-m", "cnctwin.sweep.generate", str(request_path)],
                                   env=env, cwd=str(work), capture_output=True, text=True, timeout=generator.timeout_s,
                                   check=False)
    except subprocess.TimeoutExpired:
        raise InputError(f"generator {generator.name} ran longer than {generator.timeout_s:g} s") from None
    response_path = work_root / RESPONSE
    if not response_path.is_file():
        raise InputError(f"generator {generator.name} wrote no response (exit {completed.returncode}): "
                         f"{(completed.stderr or completed.stdout)[-LOG_TAIL:]}")
    response = json.loads(response_path.read_text(encoding="utf-8"))
    if not response.get("ok"):
        raise InputError(f"generator {generator.name} failed: {response.get('error')}\n{response.get('traceback', '')}")
    shutil.rmtree(work_root, ignore_errors=True)
    program = response["program"]
    record = {"generator": generator.name, "request_sha256": request_sha,
              "program_sha256": hashlib.sha256(program.encode("utf-8")).hexdigest(), "values": dict(values),
              "log": response.get("log", "")}
    return program, record


RECORD_FORMAT = "cnctwin-sweep-generated"      # points/p<n>/.generated-<name>.json; older records have no header


def write_record(point_dir: Path, record: dict) -> None:
    """``.generated-<name>.json``: ``{format: cnctwin-sweep-generated, version: 1, generator, request_sha256,
    program_sha256, values, log}`` (a record without the header, written before it, is reused the same)."""
    from ..core.rundir import write_json_atomic
    body = {k: v for k, v in record.items() if k not in ("format", "version")}
    write_json_atomic(Path(point_dir) / f".generated-{record['generator']}.json",
                      {"format": RECORD_FORMAT, "version": 1, **body})


__all__ = ["generate", "inject_constants", "run_request", "write_record", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
