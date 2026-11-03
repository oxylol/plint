"""Numerical defaults of the geometry library, read from ``db/models/geom.yaml`` (docs/models/geom.md).

Every geometry function takes its tolerances as arguments; ``None`` means the default here. The
values are numerical settings (tessellation chord error, tree build constants, selector matching
tolerances), not physical coefficients, and each carries its source and the reason for its value.
"""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def _params() -> dict:
    from ..db.models import params

    return params("geom")


def param(name: str):
    """The SI value of one ``geom`` parameter."""
    return _params()[name]


def resolve(value, name: str):
    """``value`` if given, else the ``geom`` default ``name``."""
    return param(name) if value is None else value


def clear_cache() -> None:
    _params.cache_clear()


__all__ = ["param", "resolve", "clear_cache"]
