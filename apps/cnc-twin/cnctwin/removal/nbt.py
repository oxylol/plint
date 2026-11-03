"""Numba types of the removal layouts (for the explicit kernel signatures, architecture 5.5).

Each type is ``numba.typeof`` of a small instance built with the exact dtypes, dimensions and layout
the Python side creates (C-contiguous, writable), so every kernel compiles once and the cache keeps it.
"""

from __future__ import annotations

import numpy as np
from numba import typeof, types

from ..core.abi import EngagedIntervals, EventBuffer, SurfaceMemory
from ..tooling.abi import Envelope
from . import abi as A

OPTS = dict(cache=True, nogil=True, error_model="numpy")
# small helpers of the hot loops, inlined at the Numba IR level: no call boundary, so no reference counting
# of their array arguments per call (a NamedTuple argument costs ~7 ns per member array and call otherwise)
INLINE = dict(OPTS, inline="always")
# hot loops that allocate nothing are compiled with Numba's reference counting off for that function
# (``_nrt=False``): every array they touch arrives as an argument (Scratch, the layouts), and an inlined
# helper otherwise pays an atomic incref / decref per array argument and call (measured: the envelope
# intervals of one ray 274 ns with, 30 ns without). Numba refuses to compile an allocation in such a function.
RAW = dict(OPTS, _nrt=False)

f8 = types.float64
i8 = types.int64
i4 = types.int32
i1 = types.int8
b1 = types.boolean
void = types.void
F1 = types.float64[::1]
F2 = types.float64[:, ::1]
F3 = types.float64[:, :, ::1]
F4 = types.float64[:, :, :, ::1]
I1 = types.int64[::1]
I2 = types.int64[:, ::1]
I4_1 = types.int32[::1]
I4_2 = types.int32[:, ::1]
I4_3 = types.int32[:, :, ::1]
B1 = types.int8[::1]
B3 = types.int8[:, :, ::1]
U8_3 = types.uint8[:, :, ::1]


def _f(*shape):
    return np.zeros(shape)


def example_field() -> A.DexelField:
    k = A.K_INLINE
    return A.DexelField(_f(1, 3), _f(1), np.ones(1, np.int64), np.ones((1, 3), np.int64), np.zeros((1, 3), np.int64),
                        np.ones((1, 3), np.int64), np.zeros(1, np.int64), _f(1, 3), _f(1, 3),
                        np.zeros(1, np.int32), np.zeros((1, k, 2), np.int32), np.zeros((1, k, 2), np.int32),
                        np.full(1, -1, np.int32), np.zeros((1, k, 2), np.int32), np.zeros((1, k, 2), np.int32),
                        np.full(1, -1, np.int32), np.zeros((1, 3), np.int64), np.zeros(1, np.int8),
                        np.zeros(1, np.int8), np.zeros(1, np.int32), np.zeros(A.N_FI, np.int64))


def example_target() -> A.TargetField:
    return A.TargetField(np.zeros(2, np.int64), np.zeros((1, 2), np.int32), np.zeros(4, np.int64))


def example_profile() -> A.Profile:
    z = _f(1)
    return A.Profile(z, z.copy(), np.zeros(1, np.int32), z.copy(), z.copy(), z.copy(), z.copy(),
                     np.zeros(1, np.int32), z.copy(), _f(A.N_PI))


def example_accum() -> A.RemovalAccum:
    return A.RemovalAccum(np.zeros((1, 3), np.int64), np.zeros(A.N_IA, np.int64), _f(A.N_FA))


def example_spec() -> A.SubtractSpec:
    return A.SubtractSpec(_f(3), _f(1, 3), _f(2), _f(1), _f(3), _f(3), np.zeros(A.N_SI, np.int64), _f(A.N_SF))


def example_helix() -> A.HelixSweep:
    return A.HelixSweep(_f(1, 3, 2), np.zeros(1, np.int32), _f(1, 4), np.zeros(2, np.int32), _f(1), _f(4))


def example_polar() -> A.PolarField:
    return A.PolarField(_f(A.N_PG), np.zeros(1, np.int8), np.zeros((1, 2, 2), np.int32),
                        np.zeros((1, 2, 2), np.int32), np.zeros(1, np.int8), np.zeros((1, 2, 2), np.int32),
                        np.zeros((1, 2, 2), np.int32), np.zeros(1, np.int64), np.zeros(A.N_PINFO, np.int64))


def example_insert() -> A.InsertRegion:
    return A.InsertRegion(*(_f(2) for _ in range(8)), _f(3), _f(6))


def example_mesh() -> A.Mesh3:
    return A.Mesh3(_f(1, 3), np.zeros((1, 3), np.int32), np.zeros(1, np.int32), np.zeros(1, np.int64))


def example_envelope() -> Envelope:
    i1_ = np.zeros(1, np.int32)
    return Envelope(i1_.copy(), i1_.copy(), np.zeros(1, np.int8), *(_f(1) for _ in range(12)), chain=i1_.copy())


SCRATCH_T = typeof(A.make_scratch())
FIELD_T = typeof(example_field())
TARGET_T = typeof(example_target())
PROFILE_T = typeof(example_profile())
ACCUM_T = typeof(example_accum())
SPEC_T = typeof(example_spec())
HELIX_T = typeof(example_helix())
POLAR_T = typeof(example_polar())
SLABS_T = typeof(A.empty_slabs())
INSERT_T = typeof(example_insert())
MESH_T = typeof(example_mesh())
ENVELOPE_T = typeof(example_envelope())
EVENTS_T = typeof(EventBuffer(_f(1, A.EVENT_WIDTH), np.zeros(2, np.int64)))
ENGAGED_T = typeof(EngagedIntervals(_f(1, 1), _f(1, 1), np.zeros(1, np.int32), np.zeros(1, np.int64)))
MEMORY_T = typeof(SurfaceMemory(_f(1, 1, 3), np.zeros((1, 1), np.int64), np.zeros((1, 1), np.int8)))
