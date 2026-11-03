"""Comparisons for the parity tests."""

from __future__ import annotations

import numpy as np


def differing(a, b, skip=()) -> list[str]:
    """Names of the fields of two NamedTuples of arrays that are not bit-identical (NaN equal to NaN)."""
    out = []
    for name, x, y in zip(type(a)._fields, a, b):
        if name in skip:
            continue
        x = np.asarray(x)
        y = np.asarray(y)
        if x.shape != y.shape or x.dtype != y.dtype or not np.array_equal(x, y, equal_nan=x.dtype.kind == "f"):
            out.append(name)
    return out


def identical(a, b, skip=()) -> bool:
    return not differing(a, b, skip)
