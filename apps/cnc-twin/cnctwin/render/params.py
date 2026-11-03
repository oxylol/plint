"""Parameters of the renderer: ``db/models/render.yaml`` (appearance, sizes, cameras) and
``db/models/render-cues.yaml`` (visual cue model), read in SI through :func:`cnctwin.db.models.params`."""

from __future__ import annotations

import functools

import numpy as np

from ..db import models

VIEW = "render"
CUES = "render-cues"


@functools.lru_cache(maxsize=None)
def _get(model: str) -> dict:
    return models.params(model)


def view() -> dict:
    """``render`` parameters (a copy)."""
    return dict(_get(VIEW))


def cues() -> dict:
    """``render-cues`` parameters (a copy)."""
    return dict(_get(CUES))


def number(model: str, name: str) -> float:
    return float(_get(model)[name])


def integer(model: str, name: str) -> int:
    return int(round(float(_get(model)[name])))


def vector(model: str, name: str) -> np.ndarray:
    return np.asarray(_get(model)[name], dtype=np.float64)


def text(model: str, name: str) -> str:
    return str(_get(model)[name])


def mapping(model: str, name: str) -> dict:
    return dict(_get(model)[name])


def clear_cache() -> None:
    _get.cache_clear()
