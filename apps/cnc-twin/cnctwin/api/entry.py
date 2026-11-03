"""Entry points of jobs: ``"package.module:function"`` or ``"/path/to/file.py:function"``.

The worker resolves them in the job's own process (never in the server, which must not import
the simulator). :func:`describe` tells /api/health whether the module exists without importing it.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
from pathlib import Path

import cnctwin

_PACKAGE_DIR = Path(cnctwin.__file__).resolve().parent


class EntryPointMissing(Exception):
    """The configured module or function is not in this build."""


def split(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"entry point {spec!r}: expected 'module:function' or '/path/file.py:function'")
    module, attr = spec.rsplit(":", 1)
    if not module or not attr.isidentifier():
        raise ValueError(f"entry point {spec!r}: expected 'module:function' or '/path/file.py:function'")
    return module, attr


def resolve(spec: str):
    """The callable named by ``spec``; EntryPointMissing (with the reason) when it is not there."""
    module_name, attr = split(spec)
    if module_name.endswith(".py"):
        path = Path(module_name)
        if not path.is_file():
            raise EntryPointMissing(f"{spec} is not available: no file {path}")
        name = "cnctwin_entry_" + hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
        spec_obj = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            missing = getattr(error, "name", None) or str(error)
            if missing and (module_name == missing or module_name.startswith(missing + ".")):
                raise EntryPointMissing(f"{module_name}.{attr} is not available in this build "
                                        f"(no module {missing!r})") from None
            raise
    function = getattr(module, attr, None)
    if function is None or not callable(function):
        raise EntryPointMissing(f"{module_name}.{attr} is not available in this build "
                                f"(module {module_name!r} has no callable {attr!r})")
    return function


def describe(spec: str) -> dict:
    """{"entry": spec, "module_found": bool} without importing anything of the simulator."""
    try:
        module_name, _ = split(spec)
    except ValueError as error:
        return {"entry": spec, "module_found": False, "error": str(error)}
    if module_name.endswith(".py"):
        return {"entry": spec, "module_found": Path(module_name).is_file()}
    parts = module_name.split(".")
    if parts[0] == "cnctwin":
        base = _PACKAGE_DIR.joinpath(*parts[1:])
        found = base.with_suffix(".py").is_file() or (base / "__init__.py").is_file()
        return {"entry": spec, "module_found": found}
    return {"entry": spec, "module_found": importlib.util.find_spec(parts[0]) is not None}
