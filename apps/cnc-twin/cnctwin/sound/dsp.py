"""Signal processing of the sound synthesis (docs/models/sound.md "Signal processing").

* **Filters** are zero-phase (centred, symmetric) FIR filters designed by frequency sampling of a real
  magnitude response on a dense FFT grid, truncated to ``fir_taps`` and Blackman-windowed (Oppenheim & Schafer
  2010, 7.5). A noise-shaping filter is normalised to unit output power for unit white input.
* **Block convolution on a fixed grid**: the audio is cut into blocks of ``block_samples`` at absolute sample
  indices ``[j B, (j + 1) B)``; output block J is the sum, in this order, of the centred FFT convolutions of input
  blocks J - 1, J and J + 1 (taps - 1 < 2 B). Every output sample is therefore computed by the same operations
  whether the whole run or one window is rendered, and the two agree bit for bit.
* **WAV**: 16-bit PCM mono through the standard ``wave`` module with a LIST/INFO ``ICMT`` comment carrying the
  calibration (RIFF 1.0 lets readers ignore it); triangular dither from counter-based noise.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import numpy as np

from ..core.rng import stream_key
from . import kernels as K
from . import params as P

INT16_FULL_SCALE = 32767
STREAM_DITHER = "sound.dither"


def next_pow2(n: int) -> int:
    return 1 << max(int(n) - 1, 1).bit_length()


def design_fir(magnitude, fs: float, taps: int | None = None) -> np.ndarray:
    """Zero-phase FIR (odd length, centre tap at the middle) approximating the real gain ``magnitude(f)``."""
    taps = int(P.integer(P.SOUND, "fir_taps") if taps is None else taps)
    if taps % 2 == 0:
        taps += 1
    n_fft = next_pow2(4 * taps)
    freqs = np.arange(n_fft // 2 + 1) * fs / n_fft
    gain = np.asarray(magnitude(freqs), dtype=np.float64)
    gain = np.where(np.isfinite(gain), gain, 0.0)
    impulse = np.fft.irfft(gain, n_fft)
    half = (taps - 1) // 2
    centred = np.concatenate([impulse[-half:], impulse[: half + 1]]) if half else impulse[:1].copy()
    window = np.blackman(taps + 2)[1:-1]          # exclude the zero end points
    return np.ascontiguousarray(centred * window)


def unit_power(h: np.ndarray) -> float:
    """Output power of a unit-variance white input through ``h`` (sum of squared taps)."""
    return float(np.dot(h, h))


def log_band(freqs, f_lo: float, f_hi: float) -> np.ndarray:
    """Pink (1/f power) band between ``f_lo`` and ``f_hi`` with 12 dB/octave skirts outside: amplitude gain."""
    f = np.maximum(np.asarray(freqs, dtype=np.float64), 1e-9)
    pink = 1.0 / np.sqrt(f / f_lo)
    low = np.where(f < f_lo, (f / f_lo) ** 2, 1.0)
    high = np.where(f > f_hi, (f_hi / f) ** 2, 1.0)
    return pink * low * high


def log_hump(freqs, f_centre: float, octaves: float) -> np.ndarray:
    """Band centred on ``f_centre`` (Gaussian in log2 f, ``octaves`` the standard deviation): amplitude gain."""
    f = np.maximum(np.asarray(freqs, dtype=np.float64), 1e-9)
    x = np.log2(f / f_centre) / octaves
    return np.exp(-0.5 * x * x)


class CentredFIR:
    """One zero-phase FIR applied block by block on the absolute grid (see the module docstring)."""

    def __init__(self, h: np.ndarray, block: int) -> None:
        self.h = np.ascontiguousarray(h, dtype=np.float64)
        self.half = (len(self.h) - 1) // 2
        self.block = int(block)
        if 2 * self.half >= 2 * self.block:
            raise ValueError("filter longer than two blocks")
        self.n_fft = next_pow2(self.block + len(self.h) - 1)
        self.spectrum = np.fft.rfft(self.h, self.n_fft)
        self._cache: dict[int, np.ndarray] = {}

    def convolve(self, j: int, x: np.ndarray | None) -> np.ndarray | None:
        """Full convolution of input block j (length B + taps - 1), cached for the three outputs that use it."""
        if j in self._cache:
            return self._cache[j]
        if x is None:
            out = None
        else:
            out = np.fft.irfft(np.fft.rfft(x, self.n_fft) * self.spectrum, self.n_fft)[: self.block + len(self.h) - 1]
        self._cache[j] = out
        for old in [k for k in self._cache if k < j - 2]:
            del self._cache[old]
        return out

    def output(self, J: int, block_input) -> np.ndarray:
        """Output block J; ``block_input(j)`` returns input block j (or None for an all-zero block)."""
        y = np.zeros(self.block)
        B = self.block
        for j in (J - 1, J, J + 1):
            conv = self.convolve(j, block_input(j))
            if conv is None:
                continue
            start = (j - J) * B - self.half          # index in y of conv[0]
            lo = max(start, 0)
            hi = min(start + len(conv), B)
            if hi > lo:
                y[lo:hi] += conv[lo - start: hi - start]
        return y


def quantise(pressure_Pa: np.ndarray, n0: int, seed: int, full_scale_Pa: float, dither: bool) -> tuple[np.ndarray, int]:
    """16-bit samples of a pressure block (absolute first sample ``n0``) and the number of clipped samples."""
    x = np.asarray(pressure_Pa, dtype=np.float64) / full_scale_Pa * INT16_FULL_SCALE
    if dither:
        d = np.empty(len(x))
        K.hash_tpdf(int(seed), stream_key(STREAM_DITHER), int(n0), d)
        x = x + d
    rounded = np.floor(x + 0.5)
    clipped = int(np.count_nonzero((rounded > INT16_FULL_SCALE) | (rounded < -INT16_FULL_SCALE - 1)))
    return np.clip(rounded, -INT16_FULL_SCALE - 1, INT16_FULL_SCALE).astype("<i2"), clipped


class WavWriter:
    """Mono 16-bit PCM WAV, streamed; the LIST/INFO comment is appended on close (deterministic bytes)."""

    def __init__(self, path, rate_Hz: int, comment: str = "") -> None:
        self.path = Path(path)
        self.rate = int(rate_Hz)
        self.comment = comment
        self.tmp = self.path.with_name(f".{self.path.name}.tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._wav = wave.open(str(self.tmp), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(self.rate)
        self.frames = 0

    def write(self, samples: np.ndarray) -> None:
        self._wav.writeframes(np.ascontiguousarray(samples, dtype="<i2").tobytes())
        self.frames += len(samples)

    def close(self) -> Path:
        self._wav.close()
        if self.comment:
            append_info_comment(self.tmp, self.comment)
        self.tmp.replace(self.path)
        return self.path


def append_info_comment(path: Path, comment: str) -> None:
    """Append a LIST/INFO/ICMT chunk after the data chunk and fix the RIFF size."""
    text = comment.encode("ascii", "replace") + b"\x00"
    if len(text) % 2:
        text += b"\x00"
    info = b"INFO" + b"ICMT" + struct.pack("<I", len(text)) + text
    chunk = b"LIST" + struct.pack("<I", len(info)) + info
    with open(path, "r+b") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        if size % 2:
            handle.write(b"\x00")
            size += 1
        handle.write(chunk)
        handle.seek(4)
        handle.write(struct.pack("<I", size + len(chunk) - 8))


def read_wav(path) -> tuple[np.ndarray, int]:
    """(int16 samples, rate) of a mono WAV (tests and checks)."""
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        data = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    return data, rate


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return math.sqrt(float(np.mean(x * x))) if len(x) else 0.0
