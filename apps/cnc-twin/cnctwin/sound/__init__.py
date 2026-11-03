"""The sound of a run: what an operator (or a microphone in the enclosure) would hear, computed from the
simulated physics (docs/models/sound.md).

* :func:`render_audio` writes ``audio.wav`` (44.1 kHz, 16-bit PCM, calibrated: full scale = ``full_scale_Pa``)
  and ``audio.json`` (calibration, listener, levels).
* :func:`render_window` the pressure (Pa) over a time window, sample-identical with the whole run's audio.
* :class:`SoundInputs` / :func:`from_run` what is read from the run directory (or built by hand for studies).
* :mod:`~cnctwin.sound.acoustics` the radiation, propagation and source formulas.

``python -m cnctwin.sound RUN_DIR [--t0 T0 --t1 T1 --out window.wav]``.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "render_audio": "synth", "render_window": "synth", "SoundWindow": "synth", "Synth": "synth", "SOURCES": "synth",
    "LISTENERS": "synth",
    "SoundInputs": "inputs", "FastWindow": "inputs", "Jet": "inputs", "from_run": "inputs",
}


def __getattr__(name: str):
    if name in _EXPORTS:
        module = importlib.import_module(f"{__name__}.{_EXPORTS[name]}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_EXPORTS)
