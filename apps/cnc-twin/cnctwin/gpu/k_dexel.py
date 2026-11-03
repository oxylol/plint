"""GPU twin of the dexel subtraction ``cnctwin.removal.k_subtract.subtract_envelope`` (docs/models/gpu.md
"Dexel subtraction"; layouts: cnctwin/removal/abi.py, architecture 5.5 and 10: int32 endpoints in 10 nm
quanta, K = 4 segments inline per ray plus the overflow pool).

The CPU kernel walks the rays of the engaged-sector box level by level, family by family, (iu, iv) in
order; per ray: quick rejects, the exact envelope intervals (``_env_ray``: quadratics, the torus quartic),
quantisation, subtraction from the ray's segments, the write back (inline, then pool blocks from a shared
free list), and the piece records ``post_pieces`` books into bricks, target checks and the contact's sums.
Only the interval geometry is independent per ray; the pool, the bricks and every sum are shared and
order-dependent. So the GPU path splits the call in three:

1. :func:`gather` (CPU) - the candidate rays in the CPU kernel's order with its quick rejects, and the ray
   origins as the CPU computes them (an optimisation only: see 3).
2. ``k_rays`` (GPU, one thread per candidate) - :func:`ray_intervals`: removal's own ``_env_ray`` compiled as
   a device twin (cnctwin.gpu.twins). A ray with more than :data:`MAX_RAW` raw or :data:`MAX_OUT` merged
   intervals is marked :data:`SPILL`.
3. the commit (CPU) - **removal's own subtract_envelope**, compiled from its current source with one global
   rebound: ``_env_ray`` -> :func:`env_ray_lookup`, which returns the GPU's intervals for the ray it is asked
   about (matched on family, origin and range) and computes any ray it has no GPU result for (spilled, or
   one gather did not foresee) with the real ``_env_ray`` and the CPU kernel's own capacity. The intervals
   reach it through two extra regions of the scratch arrays ``ivt`` / ``ivp`` that only ``_env_ray`` uses.

So everything the CPU kernel does after the geometry is the CPU kernel itself (no mirrored copy to drift),
and the field arrays, accumulators, events and status are bit-identical to it (no transcendental functions
in the geometry; FMA contraction off on the device). If removal ever stops calling ``_env_ray`` by that
name, the lookup is not used: the result stays exact, the speed-up is lost, and the call's stats show it.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, types

from ..removal import abi as A
from ..removal import k_core as C
from ..removal import k_subtract as KS
from ..removal.nbt import FIELD_T, PROFILE_T, SPEC_T
from . import device as D
from .twins import builder, twin

# the functions the CPU kernel itself calls (wherever removal defines them)
_CPU_NS = KS.subtract_envelope.py_func.__globals__
_env_ray = _CPU_NS["_env_ray"]
_aabb = _CPU_NS["_aabb"]
node_range = _CPU_NS["node_range"]

Q = A.QUANTUM_M
MAX_RAW = 64          # raw envelope intervals a GPU thread holds (removal's scratch holds A.MAX_IV)
MAX_OUT = 16          # merged, quantised intervals a GPU thread returns
SPILL = -1            # the host computes this ray (more intervals than a GPU thread holds)
WORK = 16             # root-isolation workspace of _env_ray (removal allocates np.empty(16))
# candidate columns: cand_i [n, 6] int64, cand_f [n, 5] float64
CI_RAY, CI_LVL, CI_FAM, CI_IU, CI_IV, CI_LQ = 0, 1, 2, 3, 4, 5
CF_OX, CF_OY, CF_OZ, CF_TLO, CF_THI = 0, 1, 2, 3, 4
NCI = 6
NCF = 5

i8 = types.int64
F2 = types.float64[:, ::1]
I2 = types.int64[:, ::1]
OPTS = dict(cache=True, nogil=True, error_model="numpy")


# ============================================================================ 1. candidates (CPU)
@njit(i8(FIELD_T, PROFILE_T, SPEC_T, F2, I2, i8), **OPTS)
def _walk(f, prof, spec, cand_f, cand_i, record):
    """The rays subtract_envelope visits: with ``record`` 0 the number of rays in the node ranges (an upper
    bound of the candidates); with 1 the candidates themselves (written to cand_f, cand_i), their number
    returned. -1: the CPU kernel returns before its loop (nothing to do, nothing booked)."""
    cnt = f.cnt
    seg = f.seg
    link = f.link
    pseg = f.pseg
    pnext = f.pnext
    info = f.info
    lv_cell_m = f.lv_cell_m
    lv_origin = f.lv_origin_m
    lv_dims = f.lv_dims
    lv_cell_q = f.lv_cell_q
    lv_ray0 = f.lv_ray0
    w = spec.axis_S
    slab_c = spec.slab_c
    slab_z = spec.slab_z
    slab_dr = spec.slab_dr
    ns = slab_dr.shape[0]
    if ns == 0 or prof.z0_m.shape[0] == 0:
        return -1
    elo_ = np.empty(3)
    ehi_ = np.empty(3)
    Rq = _aabb(w, slab_dr, slab_c, slab_z, slab_dr, prof.info[A.PI_ZMIN], prof.info[A.PI_ZMAX],
               prof.info[A.PI_RMAX], spec.fvals[A.SF_SLACK], elo_, ehi_)
    blo = np.empty(3)
    bhi = np.empty(3)
    for i in range(3):
        blo[i] = max(elo_[i], spec.box_lo[i])
        bhi[i] = min(ehi_[i], spec.box_hi[i])
        if bhi[i] < blo[i]:
            return -1
    mask = spec.ivals[A.SI_LEVEL_MASK]
    c0x = slab_c[0, 0]
    c0y = slab_c[0, 1]
    c0z = slab_c[0, 2]
    p = np.empty(3)
    n = 0
    nl = info[A.FI_NLEVELS]
    for lvl in range(nl):
        if mask >= 0 and (mask >> lvl) & 1 == 0:
            continue
        a = lv_cell_m[lvl]
        for fam in range(3):
            u = 1 if fam == 0 else 0
            v = 2 if fam < 2 else 1
            ou = lv_origin[lvl, u]
            ov = lv_origin[lvl, v]
            of = lv_origin[lvl, fam]
            nu = lv_dims[lvl, u]
            nv = lv_dims[lvl, v]
            iu0, iu1 = node_range((blo[u] - ou) / a, (bhi[u] - ou) / a, 0.0, nu)
            iv0, iv1 = node_range((blo[v] - ov) / a, (bhi[v] - ov) / a, 0.0, nv)
            if iu1 < iu0 or iv1 < iv0:
                continue
            lq = (lv_dims[lvl, fam] - 1) * lv_cell_q[lvl]
            t_lo = blo[fam] - of
            t_hi = bhi[fam] - of
            if t_lo < 0.0:
                t_lo = 0.0
            if t_hi > lq * Q:
                t_hi = lq * Q
            if t_hi <= t_lo:
                continue
            if record == 0:
                n += (iu1 - iu0 + 1) * (iv1 - iv0 + 1)
                continue
            q_lo = np.int64(np.floor(t_lo / Q))
            q_hi = np.int64(np.ceil(t_hi / Q))
            if fam == 0:
                dx0 = 0.0
                dx1 = -w[2]
                dx2 = w[1]
            elif fam == 1:
                dx0 = w[2]
                dx1 = 0.0
                dx2 = -w[0]
            else:
                dx0 = -w[1]
                dx1 = w[0]
                dx2 = 0.0
            ndx = math.sqrt(dx0 * dx0 + dx1 * dx1 + dx2 * dx2)
            r0 = lv_ray0[lvl, fam]
            for iu in range(iu0, iu1 + 1):
                for iv in range(iv0, iv1 + 1):
                    ray = r0 + iu * nv + iv
                    if cnt[ray] == 0:
                        continue
                    p[fam] = of
                    p[u] = ou + iu * a
                    p[v] = ov + iv * a
                    rx = p[0] - c0x
                    ry = p[1] - c0y
                    rz = p[2] - c0z
                    if ndx > 1e-12:
                        dist = abs(rx * dx0 + ry * dx1 + rz * dx2) / ndx
                    else:
                        zz = rx * w[0] + ry * w[1] + rz * w[2]
                        px = rx - zz * w[0]
                        py = ry - zz * w[1]
                        pz = rz - zz * w[2]
                        dist = math.sqrt(px * px + py * py + pz * pz)
                    if dist > Rq:
                        continue
                    mat, touched = C.overlap(cnt, seg, link, pseg, pnext, ray, q_lo, q_hi)
                    if mat == 0:
                        continue
                    cand_i[n, CI_RAY] = ray
                    cand_i[n, CI_LVL] = lvl
                    cand_i[n, CI_FAM] = fam
                    cand_i[n, CI_IU] = iu
                    cand_i[n, CI_IV] = iv
                    cand_i[n, CI_LQ] = lq
                    cand_f[n, CF_OX] = p[0]
                    cand_f[n, CF_OY] = p[1]
                    cand_f[n, CF_OZ] = p[2]
                    cand_f[n, CF_TLO] = t_lo
                    cand_f[n, CF_THI] = t_hi
                    n += 1
    return n


def candidate_bound(f, prof, spec) -> int:
    """Rays in the subtraction's node ranges (the size ``auto`` compares with ``min_rays``); -1: nothing to do."""
    return int(_walk(f, prof, spec, np.zeros((1, NCF)), np.zeros((1, NCI), np.int64), np.int64(0)))


def gather(f, prof, spec) -> tuple[int, np.ndarray, np.ndarray]:
    """(n, cand_f, cand_i): the candidate rays in the CPU kernel's order (n = -1: nothing to do)."""
    bound = candidate_bound(f, prof, spec)
    if bound < 0:
        return -1, np.zeros((1, NCF)), np.zeros((1, NCI), np.int64)
    cand_f = np.zeros((max(bound, 1), NCF))
    cand_i = np.zeros((max(bound, 1), NCI), np.int64)
    n = int(_walk(f, prof, spec, cand_f, cand_i, np.int64(1)))
    return n, cand_f, cand_i


# ============================================================================ 2. per-ray intervals (twins)
@twin
def ray_intervals(c, cand_f, cand_i, pz0, pz1, pkind, pc0, pc1, pc2, pc3, ppart, w, slab_c, slab_z, slab_dr, out_n,
                  out_iv, ivt, ivp, work):
    """Candidate c: removal's _env_ray (merged, as the CPU kernel calls it) into out_iv[c], or SPILL."""
    niv = _env_ray(pz0, pz1, pkind, pc0, pc1, pc2, pc3, ppart, cand_f[c, CF_OX], cand_f[c, CF_OY], cand_f[c, CF_OZ],
                   cand_i[c, CI_FAM], w, slab_c, slab_z, slab_dr, cand_f[c, CF_TLO], cand_f[c, CF_THI], ivt, ivp, 1,
                   work)
    if niv < 0 or niv > out_iv.shape[1]:
        out_n[c] = SPILL
        return
    for i in range(niv):
        out_iv[c, i, 0] = ivt[i, 0]
        out_iv[c, i, 1] = ivt[i, 1]
    out_n[c] = niv


def k_rays(n, cand_f, cand_i, pz0, pz1, pkind, pc0, pc1, pc2, pc3, ppart, w, slab_c, slab_z, slab_dr, out_n, out_iv):
    """One thread per candidate ray."""
    c = cuda.grid(1)
    if c >= n:
        return
    ivt = cuda.local.array((MAX_RAW, 2), types.float64)
    ivp = cuda.local.array(MAX_RAW, types.int32)
    work = cuda.local.array(WORK, types.float64)
    ray_intervals(c, cand_f, cand_i, pz0, pz1, pkind, pc0, pc1, pc2, pc3, ppart, w, slab_c, slab_z, slab_dr, out_n,
                  out_iv, ivt, ivp, work)


@twin
def rays_all(n, cand_f, cand_i, pz0, pz1, pkind, pc0, pc1, pc2, pc3, ppart, w, slab_c, slab_z, slab_dr, out_n, out_iv,
             ivt, ivp, work):
    """What k_rays does, on the CPU (the tests' reference of the GPU phase)."""
    for c in range(n):
        ray_intervals(c, cand_f, cand_i, pz0, pz1, pkind, pc0, pc1, pc2, pc3, ppart, w, slab_c, slab_z, slab_dr,
                      out_n, out_iv, ivt, ivp, work)


_F1 = "float64[::1]"
_I4 = "int32[::1]"
SIGNATURES = {"k_rays": (f"void(int64, float64[:, ::1], int64[:, ::1], {_F1}, {_F1}, {_I4}, {_F1}, {_F1}, {_F1}, {_F1}, "
                         f"{_I4}, {_F1}, float64[:, ::1], {_F1}, {_F1}, int64[::1], float64[:, :, ::1])")}
SOURCES = {"k_rays": k_rays}


# ============================================================================ 3. the commit: removal's own kernel
WORK_ROWS = A.MAX_IV           # rows of the scratch ivt the CPU kernel's _env_ray works in (its capacity)
KEY_ROWS = 4                   # per candidate: (fam, ox), (oy, oz), (t_lo, t_hi), (n or SPILL, first interval row)
LOOK_SPAN = 4096               # candidates the lookup searches ahead (gather may list rays the CPU skips)
T_CURSOR, T_HITS, T_MISSES = 0, 1, 2   # counters in ivp[WORK_ROWS + .]


@njit(_env_ray.nopython_signatures[0], cache=True, nogil=True, error_model="numpy", inline="always")
def env_ray_lookup(pz0, pz1, pkind, pc0, pc1, pc2, pc3, ppart, ox, oy, oz, fam, w, slab_c, slab_z, slab_dr, t_lo, t_hi,
                   out, tags, merge, work):
    """removal's _env_ray for the commit: the GPU's intervals of this ray from the table in out[WORK_ROWS:]
    (written to out[0:n] as _env_ray would), else _env_ray itself with the CPU kernel's capacity."""
    n_cand = np.int64(out[WORK_ROWS, 0])
    cur = np.int64(tags[WORK_ROWS + T_CURSOR])
    stop = min(n_cand, cur + LOOK_SPAN)
    c = cur
    fam_f = float(fam)
    while c < stop:
        r = WORK_ROWS + 1 + c * KEY_ROWS
        if (out[r, 0] == fam_f and out[r, 1] == ox and out[r + 1, 0] == oy and out[r + 1, 1] == oz
                and out[r + 2, 0] == t_lo and out[r + 2, 1] == t_hi):
            break
        c += 1
    if c < stop and merge == 1:
        r = WORK_ROWS + 1 + c * KEY_ROWS
        tags[WORK_ROWS + T_CURSOR] = c + 1
        niv = np.int64(out[r + 3, 0])
        if niv != SPILL:
            first = np.int64(out[r + 3, 1])
            for i in range(niv):
                out[i, 0] = out[first + i, 0]
                out[i, 1] = out[first + i, 1]
            tags[WORK_ROWS + T_HITS] += 1
            return niv
    tags[WORK_ROWS + T_MISSES] += 1
    tmp = np.empty((WORK_ROWS, 2))
    ttag = np.empty(WORK_ROWS, np.int32)
    niv = _env_ray(pz0, pz1, pkind, pc0, pc1, pc2, pc3, ppart, ox, oy, oz, fam, w, slab_c, slab_z, slab_dr, t_lo, t_hi,
                   tmp, ttag, merge, work)
    for i in range(niv):
        out[i, 0] = tmp[i, 0]
        out[i, 1] = tmp[i, 1]
        tags[i] = ttag[i]
    return niv


@njit(types.UniTuple(types.Array(types.float64, 2, "C"), 1)(i8, F2, I2, types.int64[::1],
                                                              types.float64[:, :, ::1]),
      cache=True, nogil=True, error_model="numpy")
def _table(n, cand_f, cand_i, out_n, out_iv):
    total = 0
    for c in range(n):
        if out_n[c] > 0:
            total += out_n[c]
    base = WORK_ROWS + 1 + KEY_ROWS * n
    ivt = np.empty((base + total, 2))
    for r in range(WORK_ROWS + 1):
        ivt[r, 0] = 0.0
        ivt[r, 1] = 0.0
    ivt[WORK_ROWS, 0] = float(n)
    nxt = base
    for c in range(n):
        r = WORK_ROWS + 1 + c * KEY_ROWS
        ivt[r, 0] = float(cand_i[c, CI_FAM])
        ivt[r, 1] = cand_f[c, CF_OX]
        ivt[r + 1, 0] = cand_f[c, CF_OY]
        ivt[r + 1, 1] = cand_f[c, CF_OZ]
        ivt[r + 2, 0] = cand_f[c, CF_TLO]
        ivt[r + 2, 1] = cand_f[c, CF_THI]
        m = out_n[c]
        ivt[r + 3, 0] = float(m)
        ivt[r + 3, 1] = float(nxt)
        for i in range(m):
            ivt[nxt, 0] = out_iv[c, i, 0]
            ivt[nxt, 1] = out_iv[c, i, 1]
            nxt += 1
    return (ivt,)


def table(n: int, cand_f: np.ndarray, cand_i: np.ndarray, out_n: np.ndarray, out_iv: np.ndarray, sc):
    """The scratch the commit runs with: ``ivt``/``ivp`` extended by the lookup table (KEY_ROWS per candidate,
    then the intervals packed) and its counters. The working rows [0, WORK_ROWS) are what _env_ray uses."""
    (ivt,) = _table(np.int64(max(int(n), 0)), cand_f, cand_i, np.ascontiguousarray(out_n, dtype=np.int64),
                    np.ascontiguousarray(out_iv))
    return sc._replace(ivt=ivt, ivp=np.zeros(WORK_ROWS + 4, np.int32))


_CPU: dict[str, object] = {}
_GPU: dict[str, object] = {}


def commit_kernel():
    """removal's subtract_envelope compiled from its current source with _env_ray -> env_ray_lookup (once per
    process; not cached on disk: it shares its source file and line with removal's own kernel)."""
    import types as pytypes

    with D.LAUNCH_LOCK:
        found = _CPU.get("commit")
        if found is None:
            src = KS.subtract_envelope.py_func
            if "_env_ray" not in src.__code__.co_names:
                raise D.GpuError("removal's subtract_envelope no longer calls _env_ray by name: the GPU dexel path "
                                 "has nothing to replace (cnctwin.gpu.k_dexel must follow removal)")
            ns = dict(src.__globals__)
            ns["_env_ray"] = env_ray_lookup
            copy = pytypes.FunctionType(src.__code__, ns, src.__name__, src.__defaults__, src.__closure__)
            copy.__qualname__ = "gpu_commit_subtract_envelope"
            found = njit(KS.subtract_envelope.nopython_signatures[0], nogil=True, error_model="numpy")(copy)
            _CPU["commit"] = found
        return found


def commit(f, prof, spec, tgt, acc, ev, sc, n, cand_f, cand_i, out_n, out_iv) -> tuple[int, int, int]:
    """(status, hits, misses): removal's subtract_envelope with the candidates' intervals from the GPU."""
    looked = table(n, cand_f, cand_i, out_n, out_iv, sc)
    status = int(commit_kernel()(f, prof, spec, tgt, acc, ev, looked))
    return status, int(looked.ivp[WORK_ROWS + T_HITS]), int(looked.ivp[WORK_ROWS + T_MISSES])


def _profile_args(prof, spec) -> tuple:
    return (prof.z0_m, prof.z1_m, prof.kind, prof.c0, prof.c1, prof.c2, prof.c3, prof.part, spec.axis_S, spec.slab_c,
            spec.slab_z, spec.slab_dr)


def cpu_rays():
    """The CPU twin of the GPU phase (rays_all)."""
    with D.LAUNCH_LOCK:
        if "rays_all" not in _CPU:
            import numba
            from ..removal.nbt import example_profile, example_spec
            prof, spec = example_profile(), example_spec()
            args = (np.int64(1), np.zeros((1, NCF)), np.zeros((1, NCI), np.int64), *_profile_args(prof, spec),
                    np.zeros(1, np.int64), np.zeros((1, MAX_OUT, 2)), np.zeros((MAX_RAW, 2)),
                    np.zeros(MAX_RAW, np.int32), np.zeros(WORK))
            sig = types.void(*[numba.typeof(a) for a in args])
            _CPU["rays_all"] = builder("cpu").kernel(rays_all, sig)
        return _CPU["rays_all"]


def gpu_kernel():
    with D.LAUNCH_LOCK:
        if "k_rays" not in _GPU:
            cuda = D.cuda_module()
            b = builder("cuda", cuda, D.simulator_mode())
            with D.exact_compile():
                _GPU["k_rays"] = b.kernel(k_rays, SIGNATURES["k_rays"])
        return _GPU["k_rays"]


def _results(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(max(n, 1), np.int64), np.zeros((max(n, 1), MAX_OUT, 2))


def subtract_envelope_cpu_twin(f, prof, spec, tgt, acc, ev, sc, stats: np.ndarray | None = None) -> int:
    """The GPU path's three phases with the per-ray phase on the CPU (tests; the reference of k_rays)."""
    n, cand_f, cand_i = gather(f, prof, spec)
    if n < 0:
        return int(KS.subtract_envelope(f, prof, spec, tgt, acc, ev, sc))
    out_n, out_iv = _results(n)
    if n > 0:
        cpu_rays()(np.int64(n), cand_f, cand_i, *_profile_args(prof, spec), out_n, out_iv, np.zeros((MAX_RAW, 2)),
                   np.zeros(MAX_RAW, np.int32), np.zeros(WORK))
    status, hits, misses = commit(f, prof, spec, tgt, acc, ev, sc, n, cand_f, cand_i, out_n, out_iv)
    if stats is not None:
        stats[0], stats[1], stats[2] = n, misses, hits
    return status


def subtract_envelope(f, prof, spec, tgt, acc, ev, sc, stats: np.ndarray | None = None) -> int:
    """``cnctwin.removal.k_subtract.subtract_envelope`` with the ray geometry on the GPU (same arguments, same
    results, same return value). ``stats`` (optional int64[3]) receives [candidate rays, rays the host
    computed, rays taken from the GPU]. The field and accumulators change only in the commit, after the GPU
    phase succeeded; a CUDA error raises GpuError first."""
    n, cand_f, cand_i = gather(f, prof, spec)
    if n < 0:                                # the CPU kernel returns at once: let it
        return int(KS.subtract_envelope(f, prof, spec, tgt, acc, ev, sc))
    out_n, out_iv = _results(n)
    try:
        if n > 0:
            kernel = gpu_kernel()
            cuda = D.cuda_module()
            with D.LAUNCH_LOCK:
                d_n = cuda.to_device(out_n)
                d_iv = cuda.to_device(out_iv)
                args = [cuda.to_device(np.ascontiguousarray(a)) for a in (cand_f, cand_i) + _profile_args(prof, spec)]
                blocks, tpb = D.grid(n)
                kernel[blocks, tpb](np.int64(n), *args, d_n, d_iv)
                cuda.synchronize()
                out_n = d_n.copy_to_host()
                out_iv = d_iv.copy_to_host()
        commit_kernel()
    except D.GpuError:
        raise
    except Exception as exc:
        raise D.GpuError(f"dexel subtraction on the GPU failed ({type(exc).__name__}: {exc})") from exc
    status, hits, misses = commit(f, prof, spec, tgt, acc, ev, sc, n, cand_f, cand_i, out_n, out_iv)
    if stats is not None:
        stats[0], stats[1], stats[2] = n, misses, hits
    return status
