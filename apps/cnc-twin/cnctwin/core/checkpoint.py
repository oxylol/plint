"""Checkpoints: a consistent cut of the whole run (docs/architecture.md 9 and 16).

A checkpoint is taken at a macro-step boundary, after the recorder has closed its series parts, and
holds everything needed to continue bit-identically in a fresh process:

* the whole :class:`cnctwin.core.state.SimState` (clock, ledger, findings, tool table, ``bus_prev`` and
  ``bus_next``, RNG counters, every domain's state): a domain state with ``to_arrays()`` /
  ``from_arrays(arrays, meta)`` is stored through them; anything else through the generic codec
  (:class:`Encoder`): dataclasses, NamedTuples, plain objects, dicts (any key type, insertion order
  kept), lists, tuples, sets, deques, Enums, NumPy arrays and scalars, nested; shared objects
  and arrays stay shared, views stay views of their base, floats keep their bits (NaN payloads too);
* the state of every *participant* (recorder, frames writer, keyframe writer, timeline, run
  directory: objects with ``checkpoint_cut()``, ``checkpoint_state()``, ``restore_state(state)``,
  ``truncate_outputs(cut)``) and the *cut* of their outputs: byte lengths of trace.jsonl, frames.bin,
  keyframes/index.jsonl and series/index.jsonl, the keyframe and timeline counts, the closed series
  files, the outputs registered so far.

Files: ``checkpoints/ck_<tick>.npz`` (arrays ``a<n>.npy`` stored uncompressed, the state tree
``__state__.json``) and ``checkpoints/ck_<tick>.json``, the commit marker. Each is written to a
temporary file, fsynced and renamed, the marker last, so a checkpoint exists only when complete.
Retention keeps the newest ``checkpoints_keep_last`` (2) plus the one written at every operation end.
A checkpoint whose project hash or build differs from the run's is refused (it could not continue
bit-identically); :meth:`Checkpointer.resume` truncates every output after the cut.
"""

from __future__ import annotations

import base64
import collections
import dataclasses
import enum
import hashlib
import importlib
import io
import json
import math
import os
import re
import struct
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePath

import numpy as np

from .. import FORMAT_VERSION, __version__
from ..db.models import params as model_params
from .errors import CheckpointError
from .rundir import (TICK_S, ZIP_DATE, build_id, fsync_dir, fsync_file, numba_cache_key, read_json, tick_time_s,
                     write_json_atomic)

FORMAT = "cnctwin-checkpoint"
STATE_MEMBER = "__state__.json"
_MARKER = re.compile(r"^ck_(\d+)\.json$")
ALLOWED_MODULES = ("cnctwin",)
_DEFAULT_FACTORIES = {"list": list, "dict": dict, "set": set, "int": int, "float": float, "str": str, "tuple": tuple}
_RESERVED = ("$", "#")


# =============================================================================== the generic codec

def _qualname(cls) -> str:
    if "<locals>" in cls.__qualname__:
        raise CheckpointError(f"{cls.__module__}.{cls.__qualname__} is a local class: it cannot be checkpointed")
    return f"{cls.__module__}:{cls.__qualname__}"


def _float_node(value: float):
    if math.isfinite(value):
        return value
    return {"$f": struct.unpack("<Q", struct.pack("<d", value))[0]}


def _float_from(node) -> float:
    return struct.unpack("<d", struct.pack("<Q", int(node["$f"])))[0]


def _bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def _from_bits(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", int(bits)))[0]


def _is_checkpointable(obj) -> bool:
    cls = type(obj)
    return callable(getattr(cls, "to_arrays", None)) and callable(getattr(cls, "from_arrays", None))


def _root(array: np.ndarray) -> np.ndarray:
    base = array
    while isinstance(base.base, np.ndarray):
        base = base.base
    return base


def _address(array: np.ndarray) -> int:
    return int(array.__array_interface__["data"][0])


class Encoder:
    """Object graph -> (JSON tree, arrays). Objects reached twice are written once (``"#": n`` at the
    first place, ``{"$r": n}`` after); arrays are listed in a table, views of a common base as views."""

    def __init__(self, allowed_modules=ALLOWED_MODULES) -> None:
        self.allowed = tuple(allowed_modules)
        self.arrays: list[np.ndarray] = []
        self._array_index: dict[int, int] = {}
        self._counts: dict[int, int] = {}
        self._refs: dict[int, int] = {}
        self._alive: list = []

    # ------------------------------------------------------------------ pass 1: who is shared
    def _count(self, obj) -> None:
        stack = [obj]
        while stack:
            item = stack.pop()
            if item is None or isinstance(item, (bool, int, float, str, bytes, complex, np.generic, enum.Enum,
                                                 PurePath, type)):
                continue
            if isinstance(item, np.ndarray):
                continue
            key = id(item)
            self._counts[key] = self._counts.get(key, 0) + 1
            if self._counts[key] > 1:
                continue
            self._alive.append(item)
            stack.extend(self._children(item))

    def _children(self, item) -> list:
        """What :meth:`encode` will visit inside ``item`` (the same dispatch, so counts match)."""
        if isinstance(item, dict):
            out = []
            for k, v in item.items():
                out.extend((k, v))
            return out
        if isinstance(item, (list, tuple, set, frozenset, collections.deque)):
            return list(item)
        if _is_checkpointable(item):
            return []
        if dataclasses.is_dataclass(item) or hasattr(item, "__dict__") or hasattr(type(item), "__slots__"):
            return [v for _, v in self._attributes(item)]
        return []

    # ------------------------------------------------------------------ pass 2: the tree
    def encode_root(self, obj):
        self._count(obj)
        return self.encode(obj, "state")

    def _array(self, array: np.ndarray, path: str) -> dict:
        if array.dtype.hasobject:
            raise CheckpointError(f"{path}: arrays of Python objects cannot be checkpointed")
        if type(array) is not np.ndarray and not isinstance(array, np.memmap):
            raise CheckpointError(f"{path}: {type(array).__name__} (an ndarray subclass) would lose what it adds "
                                  f"to its data: keep plain arrays in the state")
        key = id(array)
        if key not in self._array_index:
            self._array_index[key] = len(self.arrays)
            self.arrays.append(array)
        return {"$a": self._array_index[key]}

    def _shared(self, obj, node: dict) -> dict:
        key = id(obj)
        if self._counts.get(key, 0) > 1:
            self._refs[key] = len(self._refs)
            node["#"] = self._refs[key]
        return node

    def _allowed(self, cls) -> str:
        name = _qualname(cls)
        top = cls.__module__.split(".")[0]
        if not any(cls.__module__ == m or cls.__module__.startswith(m + ".") or top == m for m in self.allowed):
            raise CheckpointError(f"{name}: instances of classes outside {', '.join(self.allowed)} are not "
                                  f"checkpointed (pass allowed_modules)")
        return name

    def _attributes(self, obj) -> list[tuple[str, object]]:
        exclude = set(getattr(type(obj), "__checkpoint_exclude__", ()))
        items: dict[str, object] = {}
        if dataclasses.is_dataclass(obj):
            for f in dataclasses.fields(obj):
                if f.name not in exclude:
                    items[f.name] = getattr(obj, f.name)
        if hasattr(obj, "__dict__"):
            for name, value in vars(obj).items():
                if name not in exclude and name not in items:
                    items[name] = value
        for cls in type(obj).__mro__:
            slots = cls.__dict__.get("__slots__", ())
            for name in ((slots,) if isinstance(slots, str) else tuple(slots)):
                if name in ("__dict__", "__weakref__") or name in exclude or name in items:
                    continue
                if hasattr(obj, name):
                    items[name] = getattr(obj, name)
        return list(items.items())

    def encode(self, obj, path: str):
        if isinstance(obj, np.generic):                  # before float/complex: np.float64 subclasses float
            return self._numpy_scalar(obj, path)
        if isinstance(obj, enum.Enum):                   # before str/int: str and int Enums subclass them
            return {"$e": self._allowed(type(obj)), "v": self.encode(obj.value, path)}
        if obj is None or type(obj) is bool or type(obj) is str:
            return obj
        if isinstance(obj, (bool, str)):
            raise CheckpointError(f"{path}: {type(obj).__name__} (a subclass of {type(obj).__mro__[1].__name__}) "
                                  f"cannot be checkpointed")
        if type(obj) is int:
            return obj
        if type(obj) is float:
            return _float_node(obj)
        if isinstance(obj, (int, float)):
            raise CheckpointError(f"{path}: {type(obj).__name__} (a subclass of a number type) cannot be checkpointed")
        if isinstance(obj, complex):
            return {"$c": [_bits(obj.real), _bits(obj.imag)]}
        if isinstance(obj, (bytes, bytearray)):
            return {"$b": base64.b64encode(bytes(obj)).decode("ascii"), "mutable": isinstance(obj, bytearray)}
        if isinstance(obj, np.ndarray):
            return self._array(obj, path)
        if isinstance(obj, PurePath):
            return {"$p": str(obj), "cls": type(obj).__name__}
        if isinstance(obj, type):
            return {"$type": self._allowed(obj)}
        key = id(obj)
        if key in self._refs:
            return {"$r": self._refs[key]}
        if isinstance(obj, tuple) and hasattr(type(obj), "_fields"):
            node = {"$nt": self._allowed(type(obj)), "v": None}
            self._shared(obj, node)
            node["v"] = [self.encode(v, f"{path}.{n}") for n, v in zip(type(obj)._fields, obj)]
            return node
        if isinstance(obj, tuple):
            node = self._shared(obj, {"$t": None})
            node["$t"] = [self.encode(v, f"{path}[{i}]") for i, v in enumerate(obj)]
            return node
        if isinstance(obj, list):
            values = self._fast_list(obj, path)
            if values is not None:
                return self._shared(obj, values)
            if self._counts.get(key, 0) > 1:
                node = self._shared(obj, {"$l": None})
                node["$l"] = [self.encode(v, f"{path}[{i}]") for i, v in enumerate(obj)]
                return node
            return [self.encode(v, f"{path}[{i}]") for i, v in enumerate(obj)]
        if isinstance(obj, collections.deque):
            node = self._shared(obj, {"$dq": None, "maxlen": obj.maxlen})
            node["$dq"] = [self.encode(v, f"{path}[{i}]") for i, v in enumerate(obj)]
            return node
        if isinstance(obj, (set, frozenset)):
            tag = "$fset" if isinstance(obj, frozenset) else "$set"
            node = self._shared(obj, {tag: None})
            # iteration order (deterministic under PYTHONHASHSEED=0), so a shared element is defined before
            # it is referred to; architecture 9 forbids results that depend on set order anyway
            node[tag] = [self.encode(v, f"{path}{{}}") for v in obj]
            return node
        if isinstance(obj, dict):
            return self._dict(obj, path)
        if _is_checkpointable(obj):
            return self._checkpointable(obj, path)
        if dataclasses.is_dataclass(obj):
            node = self._shared(obj, {"$dc": self._allowed(type(obj)), "f": None})
            node["f"] = {name: self.encode(value, f"{path}.{name}") for name, value in self._attributes(obj)}
            return node
        if callable(obj) or isinstance(obj, (io.IOBase,)) or type(obj).__name__ in ("generator", "module", "lock"):
            raise CheckpointError(f"{path}: {type(obj).__name__} cannot be checkpointed (keep state as data)")
        if hasattr(obj, "__dict__") or hasattr(type(obj), "__slots__"):
            node = self._shared(obj, {"$o": self._allowed(type(obj)), "f": None})
            node["f"] = {name: self.encode(value, f"{path}.{name}") for name, value in self._attributes(obj)}
            return node
        raise CheckpointError(f"{path}: {type(obj).__name__} cannot be checkpointed")

    def _numpy_scalar(self, value: np.generic, path: str) -> dict:
        dtype = value.dtype
        if dtype.kind == "f":
            return {"$n": dtype.str, "bits": _bits(float(value))}
        if dtype.kind in "iu":
            return {"$n": dtype.str, "i": int(value)}
        if dtype.kind == "b":
            return {"$n": dtype.str, "b": bool(value)}
        if dtype.kind == "c":
            return {"$n": dtype.str, "c": [_bits(float(value.real)), _bits(float(value.imag))]}
        return {"$n0": self._array(np.array(value), path)}

    def _fast_list(self, obj: list, path: str):
        """Long homogeneous lists of floats or ints as one array (exact; ints within int64)."""
        if len(obj) < 64:
            return None
        if all(type(v) is float for v in obj):
            return {"$lf": self._array(np.array(obj, dtype=np.float64), path)["$a"]}
        if all(type(v) is int for v in obj) and all(-2 ** 63 <= v < 2 ** 63 for v in obj):
            return {"$li": self._array(np.array(obj, dtype=np.int64), path)["$a"]}
        return None

    def _dict(self, obj: dict, path: str):
        if type(obj) is collections.OrderedDict:
            node = self._shared(obj, {"$od": None})
            node["$od"] = [[self.encode(k, path), self.encode(v, f"{path}[{k!r}]")] for k, v in obj.items()]
            return node
        if isinstance(obj, collections.defaultdict):
            factory = obj.default_factory
            name = None if factory is None else next((n for n, f in _DEFAULT_FACTORIES.items() if f is factory), "")
            if name == "":
                raise CheckpointError(f"{path}: a defaultdict with factory {factory!r} cannot be checkpointed")
            node = self._shared(obj, {"$dd": name, "items": None})
            node["items"] = [[self.encode(k, path), self.encode(v, f"{path}[{k!r}]")] for k, v in obj.items()]
            return node
        if type(obj) is not dict:
            raise CheckpointError(f"{path}: dict subclass {type(obj).__name__} cannot be checkpointed")
        plain = all(isinstance(k, str) and not k.startswith(_RESERVED) for k in obj)
        if plain and self._counts.get(id(obj), 0) <= 1:
            return {k: self.encode(v, f"{path}.{k}") for k, v in obj.items()}
        node = self._shared(obj, {"$d": None})
        node["$d"] = [[self.encode(k, path), self.encode(v, f"{path}[{k!r}]")] for k, v in obj.items()]
        return node

    def _checkpointable(self, obj, path: str) -> dict:
        result = obj.to_arrays()
        if not (isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], Mapping)):
            raise CheckpointError(f"{path}: {type(obj).__name__}.to_arrays() must return (arrays, meta)")
        arrays, meta = result
        node = self._shared(obj, {"$cp": self._allowed(type(obj)), "arrays": {}, "meta": None})
        for name, array in arrays.items():
            if not isinstance(array, np.ndarray):
                array = np.asarray(array)
            node["arrays"][str(name)] = self._array(array, f"{path}.{name}")["$a"]
        node["meta"] = self.encode(meta, f"{path}.meta")
        return node

    # ------------------------------------------------------------------ the array table
    def array_table(self) -> tuple[list[dict], dict[str, np.ndarray]]:
        """(specs per array index, arrays to store by member name): an array sharing memory with others
        (a view, or the base of a view) is stored once as its base plus views."""
        groups: dict[int, list[int]] = {}
        roots: dict[int, np.ndarray] = {}
        for index, array in enumerate(self.arrays):
            root = _root(array)
            if array.size == 0 or root.size == 0:
                root = array
            groups.setdefault(id(root), []).append(index)
            roots[id(root)] = root
        specs: list[dict] = [{} for _ in self.arrays]
        store: dict[str, np.ndarray] = {}
        for key in sorted(groups, key=lambda k: groups[k][0]):
            members = groups[key]
            root = roots[key]
            if len(members) == 1 or not (root.flags.c_contiguous or root.flags.f_contiguous):
                for index in members:
                    name = f"a{len(store)}"
                    store[name] = self.arrays[index]
                    specs[index] = {"store": name}
                continue
            name = f"a{len(store)}"
            store[name] = root
            base = _address(root)
            for index in members:
                array = self.arrays[index]
                if array is root:
                    specs[index] = {"store": name}
                    continue
                specs[index] = {"view": name, "offset": _address(array) - base, "shape": list(array.shape),
                                "strides": list(array.strides), "dtype": array.dtype.str}
        return specs, store


class Decoder:
    """(JSON tree, array specs, stored arrays) -> object graph (the inverse of :class:`Encoder`)."""

    def __init__(self, specs: list[dict], load, allowed_modules=ALLOWED_MODULES) -> None:
        self.specs = specs
        self.load = load
        self.allowed = tuple(allowed_modules)
        self._arrays: dict[int, np.ndarray] = {}
        self._stored: dict[str, np.ndarray] = {}
        self._refs: dict[int, object] = {}
        self._viewed = {s["view"] for s in specs if "view" in s}
        self.converted_to_c = 0

    def _member(self, name: str, contiguous: bool) -> np.ndarray:
        if name not in self._stored:
            array = self.load(name)
            if not array.dtype.isnative:
                array = array.astype(array.dtype.newbyteorder("="))
            if not array.flags.writeable:
                array = array.copy()
            self._stored[name] = array
        array = self._stored[name]
        if contiguous and not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
            self._stored[name] = array
            self.converted_to_c += 1
        return array

    def array(self, index: int) -> np.ndarray:
        if index in self._arrays:
            return self._arrays[index]
        spec = self.specs[index]
        if "store" in spec:
            array = self._member(spec["store"], contiguous=spec["store"] not in self._viewed)
        else:
            root = self._member(spec["view"], contiguous=False)
            dtype = np.dtype(spec["dtype"])
            if not dtype.isnative:
                dtype = dtype.newbyteorder("=")
            array = np.ndarray(tuple(spec["shape"]), dtype=dtype, buffer=root, offset=int(spec["offset"]),
                               strides=tuple(spec["strides"]))
        self._arrays[index] = array
        return array

    def _class(self, name: str):
        module_name, _, qualname = name.partition(":")
        top = module_name.split(".")[0]
        if not any(module_name == m or module_name.startswith(m + ".") or top == m for m in self.allowed):
            raise CheckpointError(f"the checkpoint names {name}, outside {', '.join(self.allowed)}")
        try:
            obj = importlib.import_module(module_name)
            for part in qualname.split("."):
                obj = getattr(obj, part)
        except (ImportError, AttributeError) as error:
            raise CheckpointError(f"the checkpoint names {name}, which this build does not have ({error})") from None
        return obj

    def _register(self, node: dict, obj):
        if isinstance(node, dict) and "#" in node:
            self._refs[int(node["#"])] = obj
        return obj

    def decode(self, node):
        if node is None or isinstance(node, (bool, str, int, float)):
            return node
        if isinstance(node, list):
            return [self.decode(v) for v in node]
        if not isinstance(node, dict):
            raise CheckpointError(f"unexpected node {node!r} in the checkpoint")
        if not any(k.startswith("$") for k in node):
            return {k: self.decode(v) for k, v in node.items()}
        if "$r" in node:
            if int(node["$r"]) not in self._refs:
                raise CheckpointError(f"the checkpoint refers to object {node['$r']} before it is complete "
                                      f"(a cycle through a tuple)")
            return self._refs[int(node["$r"])]
        if "$a" in node:
            return self.array(int(node["$a"]))
        if "$f" in node:
            return _float_from(node)
        if "$c" in node:
            return complex(_from_bits(node["$c"][0]), _from_bits(node["$c"][1]))
        if "$b" in node:
            data = base64.b64decode(node["$b"])
            return bytearray(data) if node.get("mutable") else data
        if "$n" in node:
            dtype = np.dtype(node["$n"])
            if "bits" in node:
                return dtype.type(_from_bits(node["bits"]))
            if "i" in node:
                return dtype.type(int(node["i"]))
            if "b" in node:
                return dtype.type(bool(node["b"]))
            return dtype.type(complex(_from_bits(node["c"][0]), _from_bits(node["c"][1])))
        if "$n0" in node:
            return self.array(int(node["$n0"]["$a"]))[()]
        if "$p" in node:
            return Path(node["$p"])
        if "$type" in node:
            return self._class(node["$type"])
        if "$e" in node:
            return self._class(node["$e"])(self.decode(node["v"]))
        if "$lf" in node or "$li" in node:
            values = self.array(int(node.get("$lf", node.get("$li")))).tolist()
            return self._register(node, values)
        if "$t" in node:
            return self._register(node, tuple(self.decode(v) for v in node["$t"]))
        if "$nt" in node:
            cls = self._class(node["$nt"])
            return self._register(node, cls(*[self.decode(v) for v in node["v"]]))
        if "$l" in node:
            out: list = self._register(node, [])
            out.extend(self.decode(v) for v in node["$l"])
            return out
        if "$dq" in node:
            out = self._register(node, collections.deque(maxlen=node.get("maxlen")))
            out.extend(self.decode(v) for v in node["$dq"])
            return out
        if "$set" in node or "$fset" in node:
            if "$fset" in node:
                return self._register(node, frozenset(self.decode(v) for v in node["$fset"]))
            out = self._register(node, set())
            out.update(self.decode(v) for v in node["$set"])
            return out
        if "$d" in node or "$od" in node or "$dd" in node:
            if "$od" in node:
                out = self._register(node, collections.OrderedDict())
                pairs = node["$od"]
            elif "$dd" in node:
                factory = _DEFAULT_FACTORIES.get(node["$dd"]) if node["$dd"] is not None else None
                out = self._register(node, collections.defaultdict(factory))
                pairs = node["items"]
            else:
                out = self._register(node, {})
                pairs = node["$d"]
            for k, v in pairs:
                out[self._hashable(self.decode(k))] = self.decode(v)
            return out
        if "$cp" in node:
            cls = self._class(node["$cp"])
            arrays = {name: self.array(int(index)) for name, index in node["arrays"].items()}
            return self._register(node, cls.from_arrays(arrays, self.decode(node["meta"])))
        if "$dc" in node or "$o" in node:
            cls = self._class(node.get("$dc") or node.get("$o"))
            obj = cls.__new__(cls)
            self._register(node, obj)
            for name, value in node["f"].items():
                object.__setattr__(obj, name, self.decode(value))
            return obj
        raise CheckpointError(f"unknown node {sorted(node)} in the checkpoint")

    @staticmethod
    def _hashable(key):
        return tuple(key) if isinstance(key, list) else key


def encode(obj, allowed_modules=ALLOWED_MODULES) -> tuple[dict, dict[str, np.ndarray]]:
    """(document, arrays): ``document = {tree, arrays: specs}`` is JSON; arrays by member name."""
    encoder = Encoder(allowed_modules)
    tree = encoder.encode_root(obj)
    specs, store = encoder.array_table()
    return {"tree": tree, "arrays": specs}, store


def decode(document: Mapping, arrays: Mapping[str, np.ndarray], allowed_modules=ALLOWED_MODULES):
    decoder = Decoder(list(document["arrays"]), lambda name: arrays[name], allowed_modules)
    return decoder.decode(document["tree"])


# =============================================================================== files

def _write_checkpoint_npz(path: Path, document: dict, store: Mapping[str, np.ndarray]) -> tuple[int, str]:
    """Write the arrays (stored) and the state tree (deflated) to ``path``; (bytes, sha256 of the tree)."""
    # insertion order (not sorted): a shared object is defined ("#") before it is referred to ("$r")
    text = json.dumps(document, separators=(",", ":"), allow_nan=False).encode("utf-8")
    with open(path, "wb") as raw:
        with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            info = zipfile.ZipInfo(STATE_MEMBER, date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, text)
            for name in sorted(store, key=lambda n: int(n[1:])):
                array = store[name]
                info = zipfile.ZipInfo(f"{name}.npy", date_time=ZIP_DATE)
                info.compress_type = zipfile.ZIP_STORED
                with archive.open(info, "w", force_zip64=True) as member:
                    np.lib.format.write_array(member, array, allow_pickle=False)
        raw.flush()
        os.fsync(raw.fileno())
    return path.stat().st_size, hashlib.sha256(text).hexdigest()


def _read_checkpoint_npz(path: Path, expected_sha256: str | None):
    archive = zipfile.ZipFile(path, "r")
    text = archive.read(STATE_MEMBER)
    if expected_sha256 is not None and hashlib.sha256(text).hexdigest() != expected_sha256:
        archive.close()
        raise CheckpointError(f"{path.name}: its state does not match its commit marker")
    document = json.loads(text)

    def load(name: str) -> np.ndarray:
        with archive.open(f"{name}.npy") as member:
            return np.lib.format.read_array(member, allow_pickle=False)
    return archive, document, load


# =============================================================================== the checkpointer

@dataclass
class Restored:
    """What a restore gives back: the SimState, the marker, the participants' states (already applied
    to the participants passed in) and the trace log reopened at the cut (resume only)."""

    state: object
    marker: dict
    participants: dict
    trace: object = None
    path: Path | None = None
    converted_to_c: int = 0         # arrays stored in another memory order, restored C-contiguous

    @property
    def tick(self) -> int:
        return int(self.marker["tick"])


class Checkpointer:
    """Writes, prunes and restores the checkpoints of one run directory (module docstring).

    ``project_hash``: the run's ResolvedProject.hash; ``build``: :func:`cnctwin.core.rundir.build_id` by
    default; ``interval_ticks``: base ticks between checkpoints (default ``checkpoint_interval_s``);
    ``allowed_modules``: packages whose classes the generic codec may store and recreate."""

    def __init__(self, run_dir, *, project_hash: str, build: str | None = None, interval_ticks: int | None = None,
                 keep_last: int | None = None, tick_s: float = TICK_S, allowed_modules=ALLOWED_MODULES) -> None:
        settings = model_params("recording")
        self.run_dir = Path(run_dir)
        self.dir = self.run_dir / "checkpoints"
        self.project_hash = str(project_hash)
        self.build = build if build is not None else build_id()
        self.tick_s = float(tick_s)
        self.interval_ticks = int(interval_ticks if interval_ticks is not None
                                  else round(settings["checkpoint_interval_s"] / self.tick_s))
        if self.interval_ticks <= 0:
            raise CheckpointError("the checkpoint interval is at least one tick")
        self.keep_last = int(keep_last if keep_last is not None else settings["checkpoints_keep_last"])
        self.allowed = tuple(allowed_modules)
        self.last_tick = 0

    def due(self, tick: int) -> bool:
        """A checkpoint is due at a macro boundary once ``interval_ticks`` have passed since the last."""
        return int(tick) - self.last_tick >= self.interval_ticks

    # ------------------------------------------------------------------ writing
    def write(self, state, *, tick: int, op: str = "", op_index: int = 0, op_end: bool = False,
              participants: Mapping | None = None, trace=None, extra: Mapping | None = None) -> dict:
        """Take the cut at ``tick`` and commit ``checkpoints/ck_<tick>``; returns the marker."""
        tick = int(tick)
        participants = dict(participants or {})
        self.dir.mkdir(parents=True, exist_ok=True)
        cuts = {name: participant.checkpoint_cut() for name, participant in sorted(participants.items())}
        trace_cut = None
        if trace is not None:
            trace_cut = {"bytes": int(trace.byte_length()), "events": len(getattr(trace, "_offsets", []))}
            path = getattr(trace, "path", None)
            if path is not None and Path(path).exists():
                fsync_file(Path(path))
        payload = {"state": state,
                   "participants": {name: p.checkpoint_state() for name, p in sorted(participants.items())}}
        document, store = encode(payload, self.allowed)
        npz = self.dir / f"ck_{tick}.npz"
        tmp = self.dir / f".ck_{tick}.npz.tmp"
        size, state_sha = _write_checkpoint_npz(tmp, document, store)
        os.replace(tmp, npz)
        fsync_dir(self.dir)
        findings = getattr(state, "findings", None)
        clock = getattr(state, "clock", None)
        marker = {"format": FORMAT, "version": FORMAT_VERSION, "tick": tick, "time_s": tick_time_s(tick, self.tick_s),
                  "macro": int(getattr(clock, "macro", 0)) if clock is not None else 0, "op": str(op),
                  "op_index": int(op_index), "op_end": bool(op_end), "npz": npz.name, "npz_bytes": int(size),
                  "state_sha256": state_sha, "arrays": len(store), "project_hash": self.project_hash,
                  "build": self.build, "numba_cache_key": numba_cache_key(), "cnctwin": __version__,
                  "findings": len(findings) if findings is not None else 0, "trace": trace_cut, "cuts": cuts}
        marker.update(_flat_cuts(cuts, trace_cut))
        if extra:
            marker["extra"] = dict(extra)
        write_json_atomic(self.dir / f"ck_{tick}.json", marker, fsync=True)
        fsync_dir(self.dir)
        self.last_tick = tick
        self.prune()
        return marker

    # ------------------------------------------------------------------ listing, retention
    def markers(self) -> list[tuple[int, Path]]:
        """Committed checkpoints (marker and its .npz present), oldest first."""
        if not self.dir.is_dir():
            return []
        found = []
        for path in self.dir.iterdir():
            match = _MARKER.match(path.name)
            if match and path.with_suffix(".npz").is_file():
                found.append((int(match.group(1)), path))
        return sorted(found)

    def latest(self) -> Path | None:
        found = self.markers()
        return found[-1][1] if found else None

    def nearest(self, tick: int) -> Path | None:
        """The newest committed checkpoint at or before ``tick`` (for a replay)."""
        chosen = None
        for marker_tick, path in self.markers():
            if marker_tick <= int(tick):
                chosen = path
        return chosen

    def prune(self) -> list[Path]:
        """Keep the newest ``keep_last`` checkpoints and every operation-end checkpoint; remove the rest
        (marker first, so a reader never sees a marker without its arrays) and stale temporary files."""
        found = self.markers()
        keep = {path for _, path in found[-self.keep_last:]} if self.keep_last > 0 else set()
        removed = []
        for _, path in found:
            if path in keep:
                continue
            try:
                if read_json(path).get("op_end"):
                    continue
            except (OSError, ValueError):
                pass
            path.unlink()
            npz = path.with_suffix(".npz")
            if npz.exists():
                npz.unlink()
            removed.append(path)
        for stale in self.dir.glob(".ck_*.tmp") if self.dir.is_dir() else ():
            stale.unlink()
        if removed:
            fsync_dir(self.dir)
        return removed

    # ------------------------------------------------------------------ restoring
    def read_marker(self, marker_path) -> dict:
        marker_path = Path(marker_path)
        try:
            marker = read_json(marker_path)
        except (OSError, ValueError) as error:
            raise CheckpointError(f"{marker_path}: not a readable checkpoint marker ({error})") from None
        if marker.get("format") != FORMAT:
            raise CheckpointError(f"{marker_path.name}: format {marker.get('format')!r} is not {FORMAT}")
        if int(marker.get("version", 0)) > FORMAT_VERSION:
            raise CheckpointError(f"{marker_path.name}: checkpoint version {marker.get('version')} is newer than "
                                  f"this build reads ({FORMAT_VERSION})")
        if marker.get("project_hash") != self.project_hash:
            raise CheckpointError(f"{marker_path.name} was written for another project (hash "
                                  f"{str(marker.get('project_hash'))[:12]}, this run's {self.project_hash[:12]}): "
                                  f"it cannot continue this run")
        if marker.get("build") != self.build:
            raise CheckpointError(f"{marker_path.name} was written by another build (build "
                                  f"{str(marker.get('build'))[:12]}, this one {self.build[:12]}): a continuation "
                                  f"would not be bit-identical, start the run again")
        return marker

    def load(self, marker_path, participants: Mapping | None = None) -> Restored:
        """Read a checkpoint without touching the run's outputs (a replay): the state, and the
        participants' states applied to the participants given."""
        marker_path = Path(marker_path)
        marker = self.read_marker(marker_path)
        npz = marker_path.with_name(marker["npz"])
        if not npz.is_file() or npz.stat().st_size != int(marker["npz_bytes"]):
            raise CheckpointError(f"{npz.name} is missing or does not have the size its marker gives")
        archive, document, load = _read_checkpoint_npz(npz, marker.get("state_sha256"))
        decoder = Decoder(list(document["arrays"]), load, self.allowed)
        try:
            payload = decoder.decode(document["tree"])
        except (KeyError, ValueError, zipfile.BadZipFile) as error:
            raise CheckpointError(f"{npz.name} cannot be read: {error}") from None
        finally:
            archive.close()
        saved = payload["participants"]
        participants = dict(participants or {})
        unknown = sorted(set(participants) - set(saved))
        if unknown:
            raise CheckpointError(f"{marker_path.name} has no state for {', '.join(unknown)}")
        for name, participant in participants.items():
            participant.restore_state(saved[name])
        self.last_tick = int(marker["tick"])
        return Restored(payload["state"], marker, saved, None, marker_path, decoder.converted_to_c)

    def resume(self, marker_path, participants: Mapping | None = None, trace_path=None, run_directory=None) -> Restored:
        """Restore a checkpoint to continue the run: everything written after the cut is removed (each
        participant truncates its outputs), the trace log is reopened at its cut (``Restored.trace``),
        findings.json is rewritten from the restored findings, later checkpoints are removed."""
        participants = dict(participants or {})
        marker_path = Path(marker_path)
        restored = self.load(marker_path, participants)
        missing = sorted(set(restored.marker.get("cuts", {})) - set(participants))
        if missing:
            raise CheckpointError(f"{marker_path.name} also cut {', '.join(missing)}: pass them as participants "
                                  f"so their outputs are truncated too")
        for name, participant in sorted(participants.items()):
            participant.truncate_outputs(restored.marker["cuts"][name])
        trace_cut = restored.marker.get("trace")
        if trace_cut is not None:
            from .trace import TraceLog
            path = Path(trace_path) if trace_path is not None else self.run_dir / "trace.jsonl"
            if not path.exists() or path.stat().st_size < int(trace_cut["bytes"]):
                raise CheckpointError(f"{path.name} is shorter than the checkpoint's cut ({trace_cut['bytes']} bytes)")
            restored.trace = TraceLog.resume(path, int(trace_cut["bytes"]))
        findings = getattr(restored.state, "findings", None)
        if findings is not None:
            from .rundir import RunDirectory
            target = run_directory if run_directory is not None else RunDirectory(self.run_dir)
            target.write_findings(findings)
        for marker_tick, path in self.markers():
            if marker_tick > restored.tick:
                path.unlink()
                if path.with_suffix(".npz").exists():
                    path.with_suffix(".npz").unlink()
        fsync_dir(self.dir)
        return restored


def _flat_cuts(cuts: Mapping, trace_cut) -> dict:
    """The cut values readers look for at the top of the marker (docs/formats.md "Run directory")."""
    flat = {}
    if trace_cut is not None:
        flat["trace_bytes"] = trace_cut["bytes"]
        flat["trace_events"] = trace_cut["events"]
    frames = cuts.get("frames")
    if frames is not None:
        flat["frames_bytes"] = frames["bytes"]
    keyframes = cuts.get("keyframes")
    if keyframes is not None:
        flat["keyframes"] = keyframes["keyframes"]
    timeline = cuts.get("timeline")
    if timeline is not None:
        flat["timeline_blocks"] = timeline["blocks"]
    recorder = cuts.get("recorder")
    if recorder is not None:
        flat["series"] = recorder["series"]
        flat["series_index_lines"] = recorder["index_lines"]
        flat["series_index_bytes"] = recorder["index_bytes"]
    return flat


__all__ = ["Checkpointer", "Restored", "Encoder", "Decoder", "encode", "decode", "FORMAT", "ALLOWED_MODULES"]
