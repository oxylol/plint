"""Small things used everywhere."""
from __future__ import annotations

import warnings
from contextlib import contextmanager

import numpy as np


@contextmanager
def quiet():
    """Arithmetic where NaN is an ordinary answer ("no bar that day", "no stocks that day"), not news: numpy's
    floating-point errors and its empty-slice warnings are both switched off inside."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        yield
