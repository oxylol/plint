"""Parameters of the sound model: ``db/models/sound.yaml`` (acoustics, listener, enclosure, structures, DSP)
and ``db/models/sound-sources.yaml`` (sources), read in SI through :func:`cnctwin.db.models.params`."""

from __future__ import annotations

import functools

from ..db import models

SOUND = "sound"
SOURCES = "sound-sources"


@functools.lru_cache(maxsize=None)
def _get(model: str) -> dict:
    return models.params(model)


def sound() -> dict:
    return dict(_get(SOUND))


def sources() -> dict:
    return dict(_get(SOURCES))


def number(model: str, name: str) -> float:
    return float(_get(model)[name])


def integer(model: str, name: str) -> int:
    return int(round(float(_get(model)[name])))


def clear_cache() -> None:
    _get.cache_clear()
