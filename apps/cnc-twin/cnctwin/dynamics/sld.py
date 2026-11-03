"""Stability lobe diagrams (docs/models/dynamics.md section 5).

Linearised regenerative model of a cut with depth a (the axial depth of a milling cut, the depth of cut in
turning): the dynamic force on the tool is ``F(t) = a sum_j K_j(t) (u(t) - u(t - tau_j))`` with u the relative
tool-workpiece displacement at the contact (T0) and K_j the force sensitivity of tooth j per unit depth (N/m^2),
``K_j = -sum_e (db_e / a_ref) (K_t,e c_e + K_r,e n_e - K_a,e a_e) n_e^T`` over the tooth's engaged edge elements
(incremental coefficients d(K h)/dh at the mean chip thickness; c the cutting direction, n the chip-thickness
normal, a the axial force direction), rotating with the spindle: ``R(theta) M_e R(theta)^T`` while the element is
in the engaged arc. ``u = G(i omega) F`` with G the relative FRF from the modes (:class:`ModalFRF`).

Methods:

* **ZOA** (Altintas & Budak 1995), uniform pitch: ``det[I - a (1 - e^(-i omega tau)) K0 G] = 0`` with the mean
  ``K0``. For an eigenvalue lambda of K0 G with Re > 0: ``a_lim = 1 / (2 Re lambda)``, ``omega tau = pi + 2 psi
  + 2 k pi`` (psi = arg lambda), spindle speed ``2 pi / (N tau)`` - lobes straight from a frequency scan
  (:func:`zoa_lobes`). With N = 1 and a constant K this is Tlusty's turning / boring limit
  ``a_lim = -1 / (2 K_f Re G_or)``.
* **General solver at a speed** (:func:`limit_at_speed`): the smallest real a > 0 with
  ``det[I - a B(omega)] = 0`` over the chatter frequency, B from:
  - variable pitch ZOA (Altintas, Engin & Budak 1999): ``B = sum_j (1 - e^(-i omega tau_j)) Kbar_j G``;
  - multi-frequency (Budak & Altintas 1998): ``B = (1 - e^(-i omega tau)) [G(omega + p w_T) K_(p-q)]_pq`` over
    the tooth-passing harmonics -h..h of K(t);
  - process damping (Altintas, Eynian & Onozuka 2008): G replaced by ``(I + i omega a C_pd G)^-1 G`` with the mean
    indentation damping ``C_pd = sum_e K_sp db_e L_w^2 / (4 V_e a_ref) n n^T``, iterated on a (:func:`limit_with_process_damping`).
* **Mode coupling** (Tobias 1965; Tlusty & Polacek 1963): the smallest a for which the modes with the current
  displacement's cutting stiffness only (no regeneration) become unstable (:func:`mode_coupling_limit`).
* **Semi-discretisation** check (Insperger & Stepan 2002, 2004): cnctwin.dynamics.semidisc.

Angles: right-handed about +z_T (T0); the spindle turns with sign ``spin`` (M03: -1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.errors import InputError

TWO_PI = 2.0 * math.pi

# R(theta) = P0 + e^(i theta) Pp + e^(-i theta) Pm (rotation about z)
_P0 = np.diag([0.0, 0.0, 1.0]).astype(complex)
_PP = np.array([[0.5, 0.5j, 0.0], [-0.5j, 0.5, 0.0], [0.0, 0.0, 0.0]])
_PM = _PP.conj()


def rotation_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def harmonic_coefficients(M: np.ndarray) -> dict[int, np.ndarray]:
    """C_m (m = -2..2) with R(theta) M R(theta)^T = sum_m C_m e^(i m theta)."""
    parts = ((0, _P0), (1, _PP), (-1, _PM))
    out = {m: np.zeros((3, 3), complex) for m in range(-2, 3)}
    for na, Pa in parts:
        for nb, Pb in parts:
            out[na + nb] += Pa @ M @ Pb.T
    return out


def _exp_integral(k: float, a: float, b: float) -> complex:
    """int_a^b e^(i k theta) d theta."""
    if k == 0.0:
        return complex(b - a)
    return (np.exp(1j * k * b) - np.exp(1j * k * a)) / (1j * k)


def rotated_integral(M: np.ndarray, a: float, b: float, q: int = 0, C: dict | None = None) -> np.ndarray:
    """int_a^b R(theta) M R(theta)^T e^(-i q theta) d theta (complex 3x3; real for q = 0). ``C``: the harmonic
    coefficients of M when already known."""
    if C is None:
        C = harmonic_coefficients(np.asarray(M, float))
    out = np.zeros((3, 3), complex)
    for m, Cm in C.items():
        out += Cm * _exp_integral(float(m - q), a, b)
    return out


def engaged_pieces(alpha0: float, lo: float, hi: float, a: float, b: float) -> list[tuple[float, float]]:
    """Sub-intervals of the spindle angle range [a, b] (a <= b) where the element angle alpha0 + theta lies in the
    engaged arc [lo, hi] (mod 2 pi; hi - lo <= 2 pi)."""
    width = hi - lo
    if width >= TWO_PI - 1e-15:
        return [(a, b)]
    if width <= 0.0:
        return []
    out = []
    # theta with alpha0 + theta in [lo + 2 pi k, hi + 2 pi k]
    k0 = math.floor((a + alpha0 - hi) / TWO_PI)
    k1 = math.ceil((b + alpha0 - lo) / TWO_PI)
    for k in range(int(k0), int(k1) + 1):
        t0 = lo + TWO_PI * k - alpha0
        t1 = hi + TWO_PI * k - alpha0
        s0 = max(a, t0)
        s1 = min(b, t1)
        if s1 > s0:
            out.append((s0, s1))
    return out


@dataclass
class CutModel:
    """Linearised cut per unit depth (module doc). Element matrices are in T at spindle angle 0."""

    M: np.ndarray                    # [ne, 3, 3] force sensitivity per unit depth (N/m^2)
    alpha0_rad: np.ndarray           # [ne] right-handed angle of the element about +z_T at spindle angle 0
    tooth: np.ndarray                # [ne] int
    n_teeth: int
    pitch_rad: np.ndarray            # [n_teeth] angle (cutting sense) from tooth j forward to the tooth that passed
                                     #   the same point just before it: tooth j's delay is pitch_rad[j] / |Omega|
    arc_lo_rad: float = 0.0          # engaged arc (T0, right-handed), [lo, hi]
    arc_hi_rad: float = TWO_PI
    spin: float = -1.0               # sign of the spindle angle rate (M03: -1)
    rotating: bool = True            # False: turning / boring (K constant, one delay = one work revolution)
    damping_M: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3)))   # [ne, 3, 3] PD at unit speed
    radius_m: np.ndarray = field(default_factory=lambda: np.zeros(0))            # [ne] element radius (PD speed)
    work_radius_m: float = 0.0       # turning: cutting radius on the workpiece (PD speed)

    def _harm(self, damping: bool = False) -> list:
        """Harmonic coefficients of every element matrix (cached)."""
        name = "_harm_d" if damping else "_harm_m"
        cached = self.__dict__.get(name)
        mats = self.damping_M if damping else self.M
        if cached is None or len(cached) != len(mats):
            cached = [harmonic_coefficients(np.asarray(m, float)) for m in mats]
            self.__dict__[name] = cached
        return cached

    @property
    def uniform_pitch(self) -> bool:
        p = np.asarray(self.pitch_rad, float)
        return bool(len(p) == 0 or np.allclose(p, TWO_PI / self.n_teeth, rtol=0.0, atol=1e-9))

    # ---------------------------------------------------------------- means
    def mean(self, tooth: int | None = None, damping: bool = False) -> np.ndarray:
        """Mean over one revolution of K (or of tooth j's K_j), per unit depth (real 3x3)."""
        mats = self.damping_M if damping else self.M
        out = np.zeros((3, 3))
        if len(mats) == 0:
            return out
        harm = self._harm(damping) if self.rotating else None
        for e in range(len(mats)):
            if tooth is not None and int(self.tooth[e]) != tooth:
                continue
            if not self.rotating:
                out += mats[e]
                continue
            lo = self.arc_lo_rad - float(self.alpha0_rad[e])
            hi = self.arc_hi_rad - float(self.alpha0_rad[e])
            out += rotated_integral(mats[e], lo, hi, 0, harm[e]).real / TWO_PI
        return out

    def harmonic(self, r: int) -> np.ndarray:
        """Fourier coefficient of K(t) at r times the tooth-passing frequency (uniform pitch), per unit depth:
        K(t) = sum_r K_r e^(i r w_T t), w_T = N |Omega|."""
        if not self.rotating:
            return self.mean() if r == 0 else np.zeros((3, 3), complex)
        q = int(round(r * self.n_teeth * self.spin))
        out = np.zeros((3, 3), complex)
        harm = self._harm()
        for e in range(len(self.M)):
            lo = self.arc_lo_rad - float(self.alpha0_rad[e])
            hi = self.arc_hi_rad - float(self.alpha0_rad[e])
            out += rotated_integral(self.M[e], lo, hi, q, harm[e]) / TWO_PI
        return out

    def interval_mean(self, theta_a: float, theta_b: float, tooth: int | None = None) -> np.ndarray:
        """Mean of K (tooth j's K_j) over the spindle angles between theta_a and theta_b (either order)."""
        a, b = (theta_a, theta_b) if theta_a <= theta_b else (theta_b, theta_a)
        if b <= a:
            return self.at(a, tooth)
        out = np.zeros((3, 3))
        harm = self._harm() if self.rotating else None
        for e in range(len(self.M)):
            if tooth is not None and int(self.tooth[e]) != tooth:
                continue
            if not self.rotating:
                out += self.M[e] * (b - a)
                continue
            for s0, s1 in engaged_pieces(float(self.alpha0_rad[e]), self.arc_lo_rad, self.arc_hi_rad, a, b):
                out += rotated_integral(self.M[e], s0, s1, 0, harm[e]).real
        return out / (b - a)

    def damping_interval_mean(self, theta_a: float, theta_b: float, speed_rad_s: float) -> np.ndarray:
        """Mean indentation damping matrix (per unit depth, N s / m^2) over the spindle angles between theta_a and
        theta_b at |Omega| (each element's coefficient scales as 1 / V_e)."""
        if len(self.damping_M) == 0:
            return np.zeros((3, 3))
        a, b = (theta_a, theta_b) if theta_a <= theta_b else (theta_b, theta_a)
        w = abs(float(speed_rad_s))
        if not self.rotating:
            V = w * self.work_radius_m
            return self.mean(damping=True) / V if V > 0.0 else np.zeros((3, 3))
        out = np.zeros((3, 3))
        harm = self._harm(damping=True)
        for e in range(len(self.damping_M)):
            V = w * float(self.radius_m[e])
            if V <= 0.0:
                continue
            for s0, s1 in engaged_pieces(float(self.alpha0_rad[e]), self.arc_lo_rad, self.arc_hi_rad, a, b):
                out += rotated_integral(self.damping_M[e], s0, s1, 0, harm[e]).real / V
        return out / (b - a) if b > a else out

    def at(self, theta: float, tooth: int | None = None) -> np.ndarray:
        """K (tooth j's K_j) at spindle angle theta."""
        out = np.zeros((3, 3))
        R = rotation_z(theta)
        for e in range(len(self.M)):
            if tooth is not None and int(self.tooth[e]) != tooth:
                continue
            if not self.rotating:
                out += self.M[e]
                continue
            alpha = (float(self.alpha0_rad[e]) + theta - self.arc_lo_rad) % TWO_PI
            if alpha <= self.arc_hi_rad - self.arc_lo_rad:
                out += R @ self.M[e] @ R.T
        return out

    def delays_s(self, speed_rad_s: float) -> np.ndarray:
        """Regenerative delay of each tooth at |Omega| (turning: one revolution of the workpiece)."""
        w = abs(float(speed_rad_s))
        if not self.rotating:
            return np.array([TWO_PI / w])
        return np.asarray(self.pitch_rad, float) / w

    def process_damping(self, speed_rad_s: float) -> np.ndarray:
        """Mean indentation damping matrix per unit depth (N s / m^2) at |Omega| (scales as 1/V)."""
        if len(self.damping_M) == 0:
            return np.zeros((3, 3))
        w = abs(float(speed_rad_s))
        if not self.rotating:
            V = w * self.work_radius_m
            return self.mean(damping=True) / V if V > 0.0 else np.zeros((3, 3))
        out = np.zeros((3, 3))
        harm = self._harm(damping=True)
        for e in range(len(self.damping_M)):
            V = w * float(self.radius_m[e])
            if V <= 0.0:
                continue
            lo = self.arc_lo_rad - float(self.alpha0_rad[e])
            hi = self.arc_hi_rad - float(self.alpha0_rad[e])
            out += rotated_integral(self.damping_M[e], lo, hi, 0, harm[e]).real / (TWO_PI * V)
        return out


def classic_milling(n_teeth: int, Kt_Pa: float, Kr_Pa: float, phi_st_rad: float, phi_ex_rad: float,
                    pitch_deg=None, K_sp_N_per_m3: float = 0.0, land_m: float = 0.0, radius_m: float = 0.0) -> CutModel:
    """Straight-tooth end mill in Altintas' convention (x feed, y normal; immersion phi clockwise from +y,
    phi_st..phi_ex in cut; the spindle turns clockwise seen from +z, i.e. spin -1 about +z): one element per tooth
    with F = -(K_t c + K_r n) h, h = n . (u - u_tau), n = (sin phi, cos phi), c = (cos phi, -sin phi).
    Right-handed angle alpha = pi/2 - phi, so the engaged arc is [pi/2 - phi_ex, pi/2 - phi_st]."""
    N = int(n_teeth)
    if pitch_deg is None:
        pitch = np.full(N, TWO_PI / N)
    else:
        pitch = np.radians(np.asarray(pitch_deg, float))
        if len(pitch) != N or abs(float(np.sum(pitch)) - TWO_PI) > 1e-9:
            raise InputError("pitch angles must be one per tooth and sum to 360 deg")
    # tooth j at phi_j = phi_0 + sum of the pitches before it: tooth j + 1 is pitch_j ahead of tooth j (phi grows
    # with time), so it passes every point pitch_j / |Omega| before tooth j
    phi = np.concatenate([[0.0], np.cumsum(pitch)[:-1]])
    alpha0 = 0.5 * math.pi - phi
    M = np.zeros((N, 3, 3))
    D = np.zeros((N, 3, 3))
    for j in range(N):
        a = alpha0[j]
        n = np.array([math.cos(a), math.sin(a), 0.0])
        c = np.array([math.sin(a), -math.cos(a), 0.0])
        M[j] = -np.outer(Kt_Pa * c + Kr_Pa * n, n)
        if K_sp_N_per_m3 > 0.0:
            D[j] = K_sp_N_per_m3 * land_m * land_m / 4.0 * np.outer(n, n)
    return CutModel(M=M, alpha0_rad=alpha0, tooth=np.arange(N), n_teeth=N, pitch_rad=pitch.copy(),
                    arc_lo_rad=0.5 * math.pi - phi_ex_rad, arc_hi_rad=0.5 * math.pi - phi_st_rad, spin=-1.0,
                    rotating=True, damping_M=D if K_sp_N_per_m3 > 0.0 else np.zeros((0, 3, 3)),
                    radius_m=np.full(N, radius_m))


def immersion_angles(ae_over_D: float, down: bool) -> tuple[float, float]:
    """(phi_st, phi_ex) of a straight cut: down milling arccos(2 a/D - 1)..pi, up milling 0..arccos(1 - 2 a/D)."""
    r = min(max(float(ae_over_D), 0.0), 1.0)
    if down:
        return math.acos(2.0 * r - 1.0), math.pi
    return 0.0, math.acos(1.0 - 2.0 * r)


def turning_cut(K_f_Pa: np.ndarray, normal_T0: np.ndarray, force_dir_T0: np.ndarray, K_sp_N_per_m3: float = 0.0,
                land_m: float = 0.0, work_radius_m: float = 0.0) -> CutModel:
    """Turning / boring: one constant element F = -K_f h u_F, h = n . (u - u_tau), per unit depth."""
    n = np.asarray(normal_T0, float)
    n = n / np.linalg.norm(n)
    d = np.asarray(force_dir_T0, float)
    d = d / np.linalg.norm(d)
    M = np.array([-float(K_f_Pa) * np.outer(d, n)])
    D = np.array([K_sp_N_per_m3 * land_m * land_m / 4.0 * np.outer(n, n)]) if K_sp_N_per_m3 > 0.0 \
        else np.zeros((0, 3, 3))
    return CutModel(M=M, alpha0_rad=np.zeros(1), tooth=np.zeros(1, int), n_teeth=1, pitch_rad=np.array([TWO_PI]),
                    rotating=False, damping_M=D, work_radius_m=work_radius_m)


# =============================================================================== FRF
@dataclass
class ModalFRF:
    """Relative tool-workpiece FRF at the contact (T0): G = sum_k psi_k psi_k^T / (w_k^2 - w^2 + 2 i z_k w_k w)
    + G_static (residual flexibility). psi [nm, 3] mass-normalised relative shapes."""

    omega_rad_s: np.ndarray
    zeta: np.ndarray
    psi: np.ndarray
    static: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))

    def __call__(self, omega) -> np.ndarray:
        """G(omega): [3, 3] for a scalar, [n, 3, 3] for an array."""
        w = np.atleast_1d(np.asarray(omega, float))
        wk = np.asarray(self.omega_rad_s, float)
        zk = np.asarray(self.zeta, float)
        den = wk[None, :] ** 2 - w[:, None] ** 2 + 2j * zk[None, :] * wk[None, :] * w[:, None]   # [n, nm]
        outer = np.einsum("ki,kj->kij", self.psi, self.psi)                                     # [nm, 3, 3]
        G = np.einsum("nk,kij->nij", 1.0 / den, outer) + np.asarray(self.static, float)[None, :, :]
        return G[0] if np.ndim(omega) == 0 else G

    @classmethod
    def single(cls, freq_Hz: float, zeta: float, mass_kg: float, direction) -> "ModalFRF":
        d = np.asarray(direction, float)
        d = d / np.linalg.norm(d)
        return cls(np.array([TWO_PI * freq_Hz]), np.array([zeta]), (d / math.sqrt(mass_kg))[None, :])

    @classmethod
    def combine(cls, *frfs: "ModalFRF") -> "ModalFRF":
        return cls(np.concatenate([f.omega_rad_s for f in frfs]), np.concatenate([f.zeta for f in frfs]),
                   np.concatenate([f.psi for f in frfs]), sum((f.static for f in frfs), np.zeros((3, 3))))

    def frequency_grid(self, n: int, lo_factor: float, hi_factor: float) -> np.ndarray:
        """Chatter-frequency grid: geometric between lo_factor x the lowest and hi_factor x the highest mode,
        with points concentrated around every mode (+-6 damped half-bandwidths)."""
        wk = np.asarray(self.omega_rad_s, float)
        lo = lo_factor * float(np.min(wk))
        hi = hi_factor * float(np.max(wk))
        base = np.geomspace(lo, hi, max(n // 2, 16))
        dense = []
        per = max(n // (2 * max(len(wk), 1)), 8)
        for w, z in zip(wk, self.zeta):
            half = max(6.0 * float(z) * w, 1e-3 * w)
            dense.append(np.linspace(max(w - half, lo), min(w + half, hi), per))
        return np.unique(np.concatenate([base] + dense))


# =============================================================================== ZOA lobes
def _eigs(A: np.ndarray) -> np.ndarray:
    return np.linalg.eigvals(A)


def _match(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """Reorder cur to follow prev (nearest neighbour, greedy, deterministic)."""
    n = len(cur)
    out = np.empty(n, complex)
    used = np.zeros(n, bool)
    for i in range(n):
        d = np.abs(cur - prev[i])
        d[used] = np.inf
        j = int(np.argmin(d))
        out[i] = cur[j]
        used[j] = True
    return out


@dataclass
class Lobes:
    """ZOA lobe curves and their lower envelope on a speed grid."""

    lobes: list                       # [{k, speed_rad_s [n], depth_limit_m [n], chatter_freq_Hz [n]}]
    speed_rad_s: np.ndarray           # envelope grid
    depth_limit_m: np.ndarray
    chatter_freq_Hz: np.ndarray
    lobe_index: np.ndarray
    method: str = "zoa"


def zoa_lobes(cut: CutModel, frf: ModalFRF, speeds_rad_s: np.ndarray, *, n_freq: int, n_lobes: int,
              lo_factor: float, hi_factor: float) -> Lobes:
    """Zero-order (Altintas & Budak 1995) lobes of a uniform-pitch cut from a chatter-frequency scan, with their
    lower envelope on ``speeds_rad_s`` (|Omega|)."""
    if not cut.uniform_pitch:
        raise InputError("zoa_lobes needs a uniform pitch: use limit_at_speed for variable pitch")
    K0 = cut.mean()
    omegas = frf.frequency_grid(n_freq, lo_factor, hi_factor)
    G = frf(omegas)
    lam = np.array([_eigs(K0 @ G[i]) for i in range(len(omegas))])
    for i in range(1, len(omegas)):
        lam[i] = _match(lam[i - 1], lam[i])
    N = cut.n_teeth if cut.rotating else 1
    speeds = np.asarray(speeds_rad_s, float)
    env = np.full(len(speeds), np.inf)
    env_f = np.zeros(len(speeds))
    env_k = np.full(len(speeds), -1)
    curves = []
    for k in range(int(n_lobes)):
        for b in range(lam.shape[1]):
            lr = lam[:, b].real
            ok = lr > 0.0
            if not np.any(ok):
                continue
            a = np.where(ok, 1.0 / (2.0 * np.where(ok, lr, 1.0)), np.inf)
            psi = np.arctan2(lam[:, b].imag, lr)
            eps = math.pi + 2.0 * psi
            tau = (eps + TWO_PI * k) / omegas
            Om = TWO_PI / (N * tau)
            _envelope_from_curve(Om, a, omegas, ok, speeds, env, env_f, env_k, k)
            order = np.argsort(Om[ok], kind="stable")
            curves.append({"k": k, "branch": b, "speed_rad_s": Om[ok][order], "depth_limit_m": a[ok][order],
                           "chatter_freq_Hz": (omegas[ok] / TWO_PI)[order]})
    return Lobes(curves, speeds, env, env_f, env_k)


def _envelope_from_curve(Om, a, omegas, ok, speeds, env, env_f, env_k, k) -> None:
    """Lower envelope update from one parametric curve (Om(w), a(w)) over its valid runs (linear interpolation
    on monotonic pieces)."""
    n = len(Om)
    i = 0
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ok[j + 1]:
            j += 1
        # split the run [i, j] into monotonic pieces in Om
        s = i
        while s < j:
            e = s + 1
            direction = np.sign(Om[e] - Om[s])
            while e + 1 <= j and np.sign(Om[e + 1] - Om[e]) == direction:
                e += 1
            xs = Om[s:e + 1]
            ys = a[s:e + 1]
            fs = omegas[s:e + 1]
            if direction < 0:
                xs, ys, fs = xs[::-1], ys[::-1], fs[::-1]
            if len(xs) >= 2 and xs[-1] > xs[0]:
                inside = (speeds >= xs[0]) & (speeds <= xs[-1])
                if np.any(inside):
                    ya = np.interp(speeds[inside], xs, ys)
                    yf = np.interp(speeds[inside], xs, fs)
                    idx = np.nonzero(inside)[0]
                    better = ya < env[idx]
                    env[idx[better]] = ya[better]
                    env_f[idx[better]] = yf[better] / TWO_PI
                    env_k[idx[better]] = k
            s = e
        i = j + 1


# =============================================================================== general solver at a speed
def characteristic_ZOA(cut: CutModel, G: np.ndarray, omega: float, speed_rad_s: float) -> np.ndarray:
    """B(omega) of the (variable pitch) zero-order solution: sum_j (1 - e^(-i omega tau_j)) Kbar_j G."""
    taus = cut.delays_s(speed_rad_s)
    if cut.rotating and not cut.uniform_pitch:
        B = np.zeros((3, 3), complex)
        for j in range(cut.n_teeth):
            B += (1.0 - np.exp(-1j * omega * taus[j])) * (cut.mean(tooth=j) @ G)
        return B
    tau = taus[0] if not cut.rotating else TWO_PI / (cut.n_teeth * abs(speed_rad_s))
    return (1.0 - np.exp(-1j * omega * tau)) * (cut.mean() @ G)


def characteristic_MF(cut: CutModel, G_of, omega: float, speed_rad_s: float, harmonics: int,
                      K_r: dict | None = None) -> np.ndarray:
    """B(omega) of the multi-frequency solution (uniform pitch): (1 - e^(-i omega tau)) [G(omega + p w_T) K_(p-q)];
    G_of(w) is the FRF (3x3) at w."""
    wT = cut.n_teeth * abs(speed_rad_s)
    tau = TWO_PI / wT
    h = int(harmonics)
    if K_r is None:
        K_r = {r: cut.harmonic(r) for r in range(-2 * h, 2 * h + 1)}
    n = 2 * h + 1
    B = np.zeros((3 * n, 3 * n), complex)
    for ip, p in enumerate(range(-h, h + 1)):
        Gp = G_of(omega + p * wT)
        for iq, q in enumerate(range(-h, h + 1)):
            B[3 * ip:3 * ip + 3, 3 * iq:3 * iq + 3] = Gp @ K_r[p - q]
    return (1.0 - np.exp(-1j * omega * tau)) * B


def _real_positive_crossings(omegas: np.ndarray, mats_fn) -> list[tuple[float, float]]:
    """(omega, a) where an eigenvalue mu of B(omega) is real and positive (a = 1 / mu): sign changes of Im(mu)
    along nearest-neighbour branches, refined by bisection on the branch."""
    mus = [np.linalg.eigvals(mats_fn(w)) for w in omegas]
    for i in range(1, len(mus)):
        mus[i] = _match(mus[i - 1], mus[i])
    out = []
    for i in range(len(omegas) - 1):
        a0 = mus[i]
        a1 = mus[i + 1]
        for b in range(len(a0)):
            m0, m1 = a0[b], a1[b]
            if m0.imag == 0.0 and m0.real > 0.0:
                out.append((float(omegas[i]), 1.0 / m0.real))
                continue
            if m0.imag * m1.imag >= 0.0 or (m0.real <= 0.0 and m1.real <= 0.0):
                continue
            lo, hi = float(omegas[i]), float(omegas[i + 1])
            mlo, mhi = m0, m1
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                cand = np.linalg.eigvals(mats_fn(mid))
                t = 0.5
                guess = mlo + t * (mhi - mlo)
                mm = cand[int(np.argmin(np.abs(cand - guess)))]
                if mm.imag * mlo.imag <= 0.0:
                    hi, mhi = mid, mm
                else:
                    lo, mlo = mid, mm
                if hi - lo <= 1e-12 * hi:
                    break
            if abs(mlo.imag) <= abs(mhi.imag):
                mu, w_c = mlo, lo
            else:
                mu, w_c = mhi, hi
            if mu.real > 0.0:
                out.append((w_c, 1.0 / mu.real))
    return out


@dataclass
class SpeedLimit:
    speed_rad_s: float
    depth_limit_m: float
    chatter_freq_Hz: float


def limit_at_speed(cut: CutModel, frf, speed_rad_s: float, omegas: np.ndarray, *, method: str = "zoa",
                   harmonics: int = 0, depth_for_pd: float | None = None, K_r: dict | None = None) -> SpeedLimit:
    """Smallest stable-to-unstable depth at one speed (method zoa: variable pitch ZOA; mf: multi-frequency).
    ``depth_for_pd``: include process damping at that depth (G -> (I + i w a C G)^-1 G)."""
    C = cut.process_damping(speed_rad_s) if depth_for_pd is not None else None

    def G_of(w):
        G = frf(w)
        if C is not None and np.any(C != 0.0):
            G = np.linalg.solve(np.eye(3) + 1j * w * depth_for_pd * (G @ C), G)
        return G

    if method == "mf":
        if not cut.uniform_pitch:
            raise InputError("the multi-frequency solution here needs a uniform pitch")
        fn = (lambda w: characteristic_MF(cut, G_of, w, speed_rad_s, harmonics, K_r))
    else:
        fn = (lambda w: characteristic_ZOA(cut, G_of(w), w, speed_rad_s))
    crossings = _real_positive_crossings(np.asarray(omegas, float), fn)
    if not crossings:
        return SpeedLimit(float(speed_rad_s), math.inf, 0.0)
    w_c, a = min(crossings, key=lambda c: c[1])
    return SpeedLimit(float(speed_rad_s), float(a), w_c / TWO_PI)


def limit_with_process_damping(cut: CutModel, frf, speed_rad_s: float, omegas: np.ndarray, *, tol_rel: float,
                               max_iter: int, method: str = "zoa", harmonics: int = 0,
                               window: float = 0.15) -> tuple[SpeedLimit, int]:
    """Process-damped limit at one speed: the depth a solves a = limit(G_pd(a)) (the indentation damping grows with
    the depth). Fixed-point iteration from the undamped limit with secant acceleration; after the first step the
    chatter-frequency scan is narrowed to +-window around the last chatter frequency. Returns (limit, iterations)."""
    first = limit_at_speed(cut, frf, speed_rad_s, omegas, method=method, harmonics=harmonics)
    a0 = first.depth_limit_m
    if not math.isfinite(a0):
        return first, 0
    grid = np.asarray(omegas, float)

    def g(a, w_c):
        lo, hi = (1.0 - window) * TWO_PI * w_c, (1.0 + window) * TWO_PI * w_c
        om = grid[(grid >= lo) & (grid <= hi)] if w_c > 0.0 else grid
        if len(om) < 16:
            om = grid
        res = limit_at_speed(cut, frf, speed_rad_s, om, method=method, harmonics=harmonics, depth_for_pd=a)
        if not math.isfinite(res.depth_limit_m) and len(om) < len(grid):
            res = limit_at_speed(cut, frf, speed_rad_s, grid, method=method, harmonics=harmonics, depth_for_pd=a)
        return res

    x0 = a0
    r0 = g(x0, first.chatter_freq_Hz)
    if not math.isfinite(r0.depth_limit_m):
        return r0, 1
    x1 = r0.depth_limit_m
    w_c = r0.chatter_freq_Hz
    f0 = x1 - x0
    last = r0
    for it in range(2, int(max_iter) + 1):
        r1 = g(x1, w_c)
        if not math.isfinite(r1.depth_limit_m):
            return r1, it
        f1 = r1.depth_limit_m - x1
        last = r1
        if abs(f1) <= tol_rel * x1:
            return SpeedLimit(r1.speed_rad_s, r1.depth_limit_m, r1.chatter_freq_Hz), it
        # secant step on f(a) = limit(a) - a, kept positive and within a factor 4 of the plain step
        if f1 != f0:
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        else:
            x2 = r1.depth_limit_m
        plain = r1.depth_limit_m
        if not (x2 > 0.0 and 0.25 * plain <= x2 <= 4.0 * plain):
            x2 = plain
        x0, f0 = x1, f1
        x1 = x2
        w_c = r1.chatter_freq_Hz
    return SpeedLimit(last.speed_rad_s, x1, last.chatter_freq_Hz), int(max_iter)


# =============================================================================== mode coupling
def mode_coupling_limit(cut: CutModel, frf: ModalFRF, *, a_max: float, n_scan: int, tol_rel: float) -> float:
    """Smallest depth at which the modes with the current-displacement cutting stiffness a K0 alone (no
    regeneration) lose stability: max Re eig of [[0, I], [-W^2 + a Psi^T K0 Psi, -2 Z W]] > 0. inf if none up to
    a_max."""
    K0 = cut.mean()
    Psi = np.asarray(frf.psi, float).T                 # [3, nm]
    Kq = Psi.T @ K0 @ Psi
    w = np.asarray(frf.omega_rad_s, float)
    z = np.asarray(frf.zeta, float)
    n = len(w)

    def growth(a: float) -> float:
        A = np.zeros((2 * n, 2 * n))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -np.diag(w * w) + a * Kq
        A[n:, n:] = -np.diag(2.0 * z * w)
        return float(np.max(np.linalg.eigvals(A).real))

    grid = np.geomspace(a_max * 1e-6, a_max, int(n_scan))
    prev = 0.0
    for a in grid:
        if growth(a) > 0.0:
            lo, hi = prev, a
            while hi - lo > tol_rel * hi:
                mid = 0.5 * (lo + hi)
                if growth(mid) > 0.0:
                    hi = mid
                else:
                    lo = mid
            return float(hi)
        prev = a
    return math.inf


__all__ = ["CutModel", "ModalFRF", "Lobes", "SpeedLimit", "classic_milling", "immersion_angles", "turning_cut",
           "zoa_lobes", "limit_at_speed", "limit_with_process_damping", "mode_coupling_limit", "rotated_integral",
           "harmonic_coefficients", "engaged_pieces", "rotation_z", "characteristic_ZOA", "characteristic_MF"]
