"""Plain-data (JSON) form of the controller's dataclasses, driven by their type hints.

Used for checkpoints (``ControllerState.to_arrays``) and for commands held in the look-ahead
buffer. Floats survive exactly (JSON writes the shortest repr, which reads back to the same
double); tuples, enums, nested dataclasses and ``dict[int, ...]`` keys are restored from the
annotations. Untyped payloads (``dict``/``list``/``Any``) must hold JSON types only (lists, not
tuples); the executor only ever writes such values into them.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import types
import typing

import numpy as np


def dump(obj):
    if obj is None or isinstance(obj, (bool, int, str, float)):
        return obj
    if isinstance(obj, enum.Enum):
        return obj.value
    if dataclasses.is_dataclass(obj):
        return {f.name: dump(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [dump(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): dump(v) for k, v in obj.items()}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"cannot serialise {type(obj).__name__}")


@functools.lru_cache(maxsize=None)
def _hints(cls) -> dict:
    return typing.get_type_hints(cls)


def load(tp, data):
    if data is None:
        return None
    if tp is typing.Any or tp is object:
        return data
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        options = [a for a in args if a is not type(None)]
        if len(options) == 1:
            return load(options[0], data)
        return data
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        hints = _hints(tp)
        values = {}
        for f in dataclasses.fields(tp):
            if f.name in data:
                values[f.name] = load(hints[f.name], data[f.name])
        return tp(**values)
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return tp(data)
    if origin is list:
        return [load(args[0], v) for v in data] if args else list(data)
    if origin is tuple:
        if not args:
            return tuple(data)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(load(args[0], v) for v in data)
        return tuple(load(a, v) for a, v in zip(args, data))
    if origin is dict:
        if not args:
            return dict(data)
        key_type, value_type = args
        return {load(key_type, k) if key_type is not str else k: load(value_type, v) for k, v in data.items()}
    if tp is float:
        return float(data)
    if tp is int:
        return int(data)
    if tp is bool:
        return bool(data)
    if tp is str:
        return str(data)
    if tp in (dict, list):
        return data
    return data
