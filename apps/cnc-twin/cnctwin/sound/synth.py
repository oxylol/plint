"""The machining sound of a run, sample by sample (docs/models/sound.md).

Sources (each can be switched off for a study or a test, :data:`SOURCES`):

========== ===================================================================================================
cutting    the cutting force on the tool: the fast tier resampled where it was recorded, elsewhere the tick-tier
           min/mean/max envelopes shaped by the tooth-period waveform of the nearest fast window (synchronous
           average), or by a raised-cosine pulse whose duty matches the envelope, at the tooth phase N theta(t)
           of the actual spindle angle. No contact: zero envelope, no tooth pass (air cutting).
chatter    from the dynamics channels: amplitude 0.5 vibration_pp sqrt(chatter_index) at chatter_freq_Hz, turned
           into the force that drives the dominant tool mode to it (inverse receptance); not inside fast windows
unbalance  residual unbalance m G omega^2 / omega_max (ISO 21940-11 grade G) rotating with the spindle
bearings   varying compliance at the outer-race ball passage frequency (Sunnersjo 1978)
packing    chips packed in the flutes: band noise of rms ``packing_fluctuation`` x packing torque / tool radius
spindle_motor, axis_motors, pump
           empirical sound power (Bies & Hansen 2003) split into the magnetic tone 2 f_e, inverter tones
           f_sw +- 2 f_e and broadband noise; axis motors turn with the ballscrew (angle 2 pi x / lead)
splash     liquid coolant streams: acoustic efficiency x kinetic power, band around the Minnaert frequency
air_jets   air blast / MQL: Lighthill efficiency 1e-4 M^5 of the kinetic power, hump at St = 0.2
impacts    chips striking the enclosure sheet: Hertzian half-sine force pulses, one chip per tooth pass
========== ===================================================================================================

Forces at the tool (cutting, chatter, unbalance, bearings, packing) reach the listener through the
structure-borne filters (SEA of the work and tool sides + the tool's own dipole radiation); impact forces through
the enclosure sheet; airborne sources directly. Sources in the cutting zone are heard by the operator through the
closed enclosure (insertion loss); the motors and the pump are outside it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.rng import stream_key
from . import acoustics as A
from . import dsp as D
from . import kernels as K
from . import params as P
from .inputs import AIR_JET_KINDS, LIQUID_JET_KINDS, SoundInputs, from_run

SOURCES = ("cutting", "chatter", "unbalance", "bearings", "packing", "spindle_motor", "axis_motors", "pump",
           "splash", "air_jets", "impacts")
CONTACT_SOURCES = ("cutting", "chatter", "packing")      # act between tool and part: both sides and the tool radiate
SPINDLE_SOURCES = ("unbalance", "bearings")              # act in the spindle bearings: the tool side radiates
FORCE_SOURCES = CONTACT_SOURCES + SPINDLE_SOURCES
LISTENERS = ("operator", "inside")
IMPACT_PULSE_POINTS = 8            # a chip impact is a half-sine force pulse integrated with 8 midpoints
IMPACT_MAX_PULSE_S = 1e-3          # impacts drawn up to this long before a block can still reach into it
MIN_DUTY = 0.02                    # shortest raised-cosine pulse of the envelope synthesis (tooth period fraction)
STREAMS = {"motor": "sound.spindle_motor", "pump": "sound.pump", "splash": "sound.splash", "air": "sound.air_jets",
           "packing": "sound.packing", "impacts": "sound.impacts"}


@dataclass
class SoundWindow:
    rate_Hz: float
    n0: int
    pressure_Pa: np.ndarray
    buses: dict = field(default_factory=dict)       # "inside", "outside" pressure at the listener (Pa)
    notes: list = field(default_factory=list)

    @property
    def time_s(self) -> np.ndarray:
        return (self.n0 + np.arange(len(self.pressure_Pa))) / self.rate_Hz


def _axis_stream(name: str) -> str:
    return f"sound.axis_motor.{name}"


class Synth:
    """Block synthesis of one run's sound for one listener and a set of sources."""

    def __init__(self, inputs: SoundInputs, sources=None, listener: str | None = None) -> None:
        self.inp = inputs
        self.sources = tuple(SOURCES if sources is None else sources)
        unknown = [s for s in self.sources if s not in SOURCES]
        if unknown:
            raise ValueError(f"unknown sound sources {unknown} (one of {', '.join(SOURCES)})")
        self.p = P.sound()
        self.q = P.sources()
        self.listener = str(self.p["listener"] if listener is None else listener)
        if self.listener not in LISTENERS:
            raise ValueError(f"listener {self.listener!r} is not one of {LISTENERS}")
        self.fs = float(self.p["sample_rate_Hz"])
        self.B = int(round(float(self.p["block_samples"])))
        self.n_total = int(math.ceil(inputs.end_s * self.fs))
        self.n_blocks = (self.n_total + self.B - 1) // self.B
        self.seed = int(inputs.seed)
        self.rho0, self.c0 = A.air()
        self.spindle_power_W = float(inputs.spindle_power_W or self.q["spindle_power_W"])
        self.spindle_max_rad_s = float(inputs.spindle_max_rad_s or self.q["spindle_max_speed_rad_s"])
        self._setup_geometry()
        self._setup_spindle()
        self._setup_envelopes()
        self._setup_shapes()
        self._setup_filters()
        self._cache: dict = {}

    # ------------------------------------------------------------------ set-up
    def _setup_geometry(self) -> None:
        p = self.p
        q_dir = float(p["directivity"])
        room = A.room_constant(float(p["room_surface_m2"]), float(p["room_absorption"]))
        area = float(self.inp.enclosure_area_m2 or p["enclosure_area_m2"])
        enclosure = A.room_constant(area, float(p["enclosure_absorption"]))
        self.enclosure_area_m2 = area
        op = float(p["operator_distance_m"])
        if self.listener == "operator":
            self.factor_inside = A.listener_factor(op + float(p["cut_zone_depth_m"]), q_dir, room)
            self.factor_panel = A.listener_factor(op, q_dir, room)
            self.through_enclosure_inside = True
        else:
            r = float(p["inside_distance_m"])
            self.factor_inside = A.listener_factor(r, q_dir, enclosure)
            self.factor_panel = A.listener_factor(r, q_dir, enclosure)
            self.through_enclosure_inside = False
        self.factor_motor = A.listener_factor(float(p["motor_distance_m"]), q_dir, room)
        self.factor_pump = A.listener_factor(float(p["pump_distance_m"]), q_dir, room)

    def transmission(self, freq_Hz) -> np.ndarray:
        """Power transmission of the closed enclosure (inside sources for the operator, outside for 'inside')."""
        return A.enclosure_power_transmission(freq_Hz, self.p)

    def _inside_T(self, f) -> np.ndarray:
        return self.transmission(f) if self.through_enclosure_inside else np.ones_like(np.asarray(f, float))

    def _outside_T(self, f) -> np.ndarray:
        return np.ones_like(np.asarray(f, float)) if self.through_enclosure_inside else self.transmission(f)

    def _setup_spindle(self) -> None:
        inp = self.inp
        t = np.ascontiguousarray(inp.frame_t_s, dtype=np.float64)
        theta = np.ascontiguousarray(inp.spindle_angle_rad, dtype=np.float64)
        self.frame_t = t
        self.theta = theta
        self.dtheta = np.ascontiguousarray(np.gradient(theta, t) if len(t) > 1 else np.zeros(len(t)))
        self.axis_pos = np.asarray(inp.axis_pos, dtype=np.float64).reshape(len(t), -1) if len(t) else np.zeros((0, 0))
        self.axis_vel = (np.gradient(self.axis_pos, t, axis=0) if len(t) > 1 else np.zeros_like(self.axis_pos))
        q = self.q
        self.bpfo_ratio = A.ball_pass_outer_ratio(int(round(float(q["bearing_balls"]))),
                                                  float(q["bearing_ball_diameter_m"]),
                                                  float(q["bearing_pitch_diameter_m"]),
                                                  float(q["bearing_contact_angle_rad"]))
        modes = [m for m in inp.spindle_modes if m[3] in ("x", "y")]
        if not modes:
            modes = [(float(q["chatter_default_frequency_Hz"]), float(q["chatter_default_damping"]),
                      float(q["chatter_default_stiffness_N_per_m"]), "x")]
        self.tool_modes = modes
        chattering = np.asarray(inp.chatter_freq_Hz, dtype=np.float64)
        chattering = chattering[chattering > 0.0]
        typical = float(np.median(chattering)) if len(chattering) else modes[0][0]
        self.chatter_mode = min(modes, key=lambda m: abs(m[0] - typical))[:3]

    def _setup_envelopes(self) -> None:
        """Centred moving average (``envelope_smoothing_s``) of the tick-tier min/mean/max."""
        inp = self.inp
        rows = max(int(round(float(self.p["envelope_smoothing_s"]) / max(inp.env_span_s, 1e-12))), 1)
        self.env = {}
        for name, values in (("min", inp.env_min_N), ("mean", inp.env_mean_N), ("max", inp.env_max_N)):
            v = np.asarray(values, dtype=np.float64).reshape(-1, 3)
            if rows > 1 and len(v) >= rows:
                kernel = np.full(rows, 1.0 / rows)
                v = np.stack([np.convolve(v[:, c], kernel, mode="same") for c in range(3)], axis=1)
            self.env[name] = v

    def _setup_shapes(self) -> None:
        """Tooth-period waveform of every fast window (synchronous average over the tooth phase)."""
        bins = int(round(float(self.p["shape_bins"])))
        self.shapes = []
        for window in self.inp.fast:
            t = window.t_first_s + window.dt_s * np.arange(len(window.force_N))
            theta, omega = self._spindle_at(t)
            teeth = self._teeth_at(t)
            if not np.any(teeth > 0):
                continue
            phase = np.mod(teeth * np.sign(omega) * theta, 2.0 * math.pi)
            index = np.minimum((phase / (2.0 * math.pi) * bins).astype(np.int64), bins - 1)
            counts = np.bincount(index, minlength=bins)
            if np.any(counts == 0):
                continue
            table = np.zeros((3, bins))
            for c in range(3):
                mean = np.bincount(index, weights=window.force_N[:, c], minlength=bins) / counts
                span = float(mean.max() - mean.min())
                table[c] = (mean - mean.min()) / span if span > 0.0 else np.zeros(bins)
            self.shapes.append((window.t_first_s, window.t_last_s, table, table.mean(axis=1)))
        self.shape_starts = np.array([s[0] for s in self.shapes])

    def _setup_filters(self) -> None:
        p, q, fs, B = self.p, self.q, self.fs, self.B
        taps = int(round(float(p["fir_taps"])))
        inside_T = self._inside_T
        sheet = (float(p["sheet_youngs_modulus_Pa"]), float(p["sheet_density_kg_m3"]), float(p["sheet_poisson"]),
                 float(p["sheet_thickness_m"]), float(p["sheet_loss_factor"]))
        work = (float(p["work_plate_youngs_modulus_Pa"]), float(p["work_plate_density_kg_m3"]),
                float(p["work_plate_poisson"]), float(p["work_plate_thickness_m"]), float(p["work_loss_factor"]),
                float(p["work_radiating_area_m2"]))
        tool = (float(p["work_plate_youngs_modulus_Pa"]), float(p["work_plate_density_kg_m3"]),
                float(p["work_plate_poisson"]), float(p["tool_plate_thickness_m"]), float(p["tool_loss_factor"]),
                float(p["tool_radiating_area_m2"]))
        radius = float(p["tool_radiator_radius_m"])
        rho_c = self.rho0 * self.c0
        self.firs: dict[str, D.CentredFIR] = {}
        modes_by_dir = {d: [(m[0], m[1], m[2]) for m in self.inp.spindle_modes if m[3] == d] for d in "xyz"}
        if not modes_by_dir["x"] and not modes_by_dir["y"]:
            modes_by_dir["x"] = modes_by_dir["y"] = [m[:3] for m in self.tool_modes]
        for axis, name in enumerate("xyz"):
            modes = modes_by_dir[name]

            def contact_gain(f, modes=modes):
                power = A.sea_power_per_force2(f, *work) + A.sea_power_per_force2(f, *tool)
                if modes:
                    power = power + A.dipole_power_per_force2(f, modes, radius)
                return np.sqrt(power * rho_c * self.factor_inside * inside_T(f))

            self.firs[f"contact_{name}"] = D.CentredFIR(D.design_fir(contact_gain, fs, taps), B)
        for name in "xy":
            self.firs[f"spindle_{name}"] = D.CentredFIR(D.design_fir(
                lambda f: np.sqrt(A.sea_power_per_force2(f, *tool) * rho_c * self.factor_inside * inside_T(f)), fs,
                taps), B)
        area = self.enclosure_area_m2
        self.firs["panel"] = D.CentredFIR(D.design_fir(
            lambda f: np.sqrt(0.5 * A.sea_power_per_force2(f, *sheet, area) * rho_c * self.factor_panel), fs, taps), B)
        self._shapers = {}
        lo, hi = (float(v) for v in q["motor_broadband_band_Hz"])
        self._add_shaper("motor", lambda f: D.log_band(f, lo, hi), self._outside_T, taps)
        for name in self.inp.axis_names:
            self._add_shaper(f"axis:{name}", lambda f: D.log_band(f, lo, hi), self._outside_T, taps)
        plo, phi = (float(v) for v in q["pump_broadband_band_Hz"])
        self._add_shaper("pump", lambda f: D.log_band(f, plo, phi), self._outside_T, taps)
        rho_l = float(self.inp.liquid_density_kg_m3 or q["liquid_density_kg_m3"])
        f_minnaert = A.minnaert_Hz(float(q["splash_bubble_radius_m"]), rho_l)
        octaves = float(q["splash_band_octaves"])
        self._add_shaper("splash", lambda f: D.log_hump(f, f_minnaert, octaves), inside_T, taps)
        klo, khi = (float(v) for v in q["packing_band_Hz"])
        self._add_shaper("packing", lambda f: D.log_band(f, klo, khi), None, taps)
        self.liquid_density = rho_l

    def _add_shaper(self, name: str, shape, transmission, taps: int) -> None:
        """Unit-power noise shaper (``shape`` normalised), with the enclosure transmission folded in."""
        h_shape = D.design_fir(shape, self.fs, taps)
        norm = math.sqrt(D.unit_power(h_shape))
        if transmission is None:
            h = h_shape / norm
        else:
            h = D.design_fir(lambda f: shape(f) * np.sqrt(transmission(f)), self.fs, taps) / norm
        self.firs[f"shape:{name}"] = D.CentredFIR(h, self.B)
        self._shapers[name] = True

    def _air_shaper(self, speed: float, diameter: float) -> str:
        key = f"air:{speed:.3g}:{diameter:.3g}"
        if f"shape:{key}" not in self.firs:
            centre = float(self.q["jet_strouhal"]) * speed / diameter
            octaves = float(self.q["jet_band_octaves"])
            self._add_shaper(key, lambda f: D.log_hump(f, centre, octaves), self._inside_T,
                             int(round(float(self.p["fir_taps"]))))
        return key

    # ------------------------------------------------------------------ state at times
    def _spindle_at(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = np.ascontiguousarray(t, dtype=np.float64)
        theta = np.empty(len(t))
        omega = np.empty(len(t))
        K.hermite(self.frame_t, self.theta, self.dtheta, t, theta, omega)
        return theta, omega

    def _frame_index(self, t: np.ndarray) -> np.ndarray:
        if not len(self.frame_t):
            return np.zeros(len(t), np.int64)
        return np.clip(np.searchsorted(self.frame_t, t, side="right") - 1, 0, len(self.frame_t) - 1)

    def _teeth_at(self, t: np.ndarray) -> np.ndarray:
        if not len(self.inp.teeth):
            return np.zeros(len(t), np.int64)
        return np.asarray(self.inp.teeth)[self._frame_index(t)]

    def _radius_at(self, t: np.ndarray) -> np.ndarray:
        if not len(self.inp.tool_radius_m):
            return np.zeros(len(t))
        return np.asarray(self.inp.tool_radius_m)[self._frame_index(t)]

    def _macro_at(self, values: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Macro values linearly interpolated between the centres of their steps (0 outside)."""
        if not len(self.inp.macro_t_s) or values is None or not len(values):
            return np.zeros(len(t))
        centres = self.inp.macro_t_s - 0.5 * self.inp.macro_span_s
        return np.interp(t, centres, np.asarray(values, dtype=np.float64), left=0.0, right=0.0)

    def _coolant_on_at(self, t: np.ndarray) -> np.ndarray:
        if self.inp.coolant_on is not None and len(self.inp.macro_t_s):
            return self._macro_at(self.inp.coolant_on.astype(np.float64), t)
        return np.ones(len(t))

    def _jets_at(self, t_mid: float) -> list:
        if not len(self.inp.jets_t_s):
            return []
        i = int(np.searchsorted(self.inp.jets_t_s, t_mid, side="right")) - 1
        return self.inp.jets[i] if i >= 0 else []

    # ------------------------------------------------------------------ blocks of noise and forces
    def _times(self, j: int) -> np.ndarray:
        return (j * self.B + np.arange(self.B)) / self.fs

    def _valid(self, j: int) -> np.ndarray:
        n = j * self.B + np.arange(self.B)
        return ((n >= 0) & (n < self.n_total)).astype(np.float64)

    def white(self, name: str, j: int) -> np.ndarray | None:
        if j < 0 or j >= self.n_blocks:
            return None
        key = ("white", name, j)
        if key not in self._cache:
            out = np.empty(self.B)
            stream = _axis_stream(name[len("axis:"):]) if name.startswith("axis:") else STREAMS[name]
            K.hash_noise(self.seed, stream_key(stream), j * self.B, out)
            self._cache[key] = out
        return self._cache[key]

    def shaped(self, name: str, j: int) -> np.ndarray:
        key = ("shaped", name, j)
        if key not in self._cache:
            fir = self.firs[f"shape:{name}"]
            stream = name.split(":")[0] if name.startswith("air:") else name
            self._cache[key] = fir.output(j, lambda jj: self.white(stream, jj))
        return self._cache[key]

    def force_block(self, j: int, kind: str = "contact") -> np.ndarray | None:
        """Force [B, 3] (T0) of the selected contact sources (cutting, chatter, packing) or spindle sources
        (unbalance, bearings); None outside the run."""
        if j < 0 or j >= self.n_blocks:
            return None
        key = (kind, j)
        if key not in self._cache:
            t = self._times(j)
            theta, omega = self._spindle_at(t)
            force = np.zeros((self.B, 3))
            if kind == "contact":
                fast_weight = self._fast_weight(t)
                if "cutting" in self.sources:
                    force += self._cutting(t, theta, omega, fast_weight)
                if "chatter" in self.sources:
                    force += self._chatter(t) * (1.0 - fast_weight)[:, None]
                if "packing" in self.sources:
                    torque = self._macro_at(self.inp.packing_torque_Nm, t)
                    radius = self._radius_at(t)
                    tangential = np.divide(torque, radius, out=np.zeros(len(t)), where=radius > 0.0)
                    force[:, 0] += float(self.q["packing_fluctuation"]) * tangential * self.shaped("packing", j)
            else:
                if "unbalance" in self.sources:
                    amp = A.unbalance_force_N(float(self.q["spindle_rotor_mass_kg"]),
                                              float(self.q["balance_grade_m_s"]), self.spindle_max_rad_s, omega)
                    force[:, 0] += amp * np.cos(theta)
                    force[:, 1] += amp * np.sin(theta)
                if "bearings" in self.sources:
                    amp = float(self.q["bearing_compliance_variation"]) * float(self.q["bearing_preload_N"])
                    spinning = (np.abs(omega) > 0.0).astype(np.float64)
                    phase = self.bpfo_ratio * theta
                    force[:, 0] += amp * spinning * np.cos(phase)
                    force[:, 1] += amp * spinning * np.sin(phase)
            self._cache[key] = force * self._valid(j)[:, None]
        return self._cache[key]

    def _fast_weight(self, t: np.ndarray) -> np.ndarray:
        """1 inside a fast window, raised-cosine ramps of fast_crossfade_s at its ends, 0 elsewhere."""
        weight = np.zeros(len(t))
        ramp = float(self.p["fast_crossfade_s"])
        for window in self.inp.fast:
            a, b = window.t_first_s, window.t_last_s
            if b < t[0] or a > t[-1]:
                continue
            inside = (t >= a) & (t <= b)
            if ramp > 0.0:
                edge = np.minimum((t - a) / ramp, (b - t) / ramp)
                w = np.where(inside, 0.5 - 0.5 * np.cos(math.pi * np.clip(edge, 0.0, 1.0)), 0.0)
            else:
                w = inside.astype(np.float64)
            weight = np.maximum(weight, w)
        return weight

    def _cutting(self, t, theta, omega, fast_weight) -> np.ndarray:
        inp = self.inp
        out = np.zeros((len(t), 3))
        if len(inp.env_t_s):
            centres = inp.env_t_s + 0.5 * inp.env_span_s
            env = {}
            for name, values in self.env.items():
                env[name] = np.stack([np.interp(t, centres, values[:, c], left=0.0, right=0.0) for c in range(3)],
                                     axis=1)
            teeth = self._teeth_at(t)
            phase = np.mod(teeth * np.sign(omega) * theta, 2.0 * math.pi)
            span = env["max"] - env["min"]
            shape = self._shape_values(t, phase)
            if shape is not None:
                waveform, mean_w = shape
                out = env["mean"] + span * (waveform - mean_w)
            else:
                out = env["mean"] + span * self._pulse(phase, env)
            out = out * (teeth > 0)[:, None]
        if inp.fast:
            fast = np.zeros((len(t), 3))
            cutoff = float(self.p["resample_cutoff_Hz"])
            half = int(round(float(self.p["resample_half_width"])))
            n0 = int(round(t[0] * self.fs))
            for window in inp.fast:
                if window.t_last_s < t[0] or window.t_first_s > t[-1]:
                    continue
                for c in range(3):
                    column = np.empty(len(t))
                    K.sinc_resample(window.t_first_s, window.dt_s, np.ascontiguousarray(window.force_N[:, c]), n0,
                                    self.fs, cutoff, half, column)
                    inside = (t >= window.t_first_s) & (t <= window.t_last_s)
                    fast[inside, c] = column[inside]
            out = fast_weight[:, None] * fast + (1.0 - fast_weight)[:, None] * out
        return out

    def _shape_values(self, t: np.ndarray, phase: np.ndarray):
        """(waveform [n, 3], its period mean [n, 3]) from the latest fast window at or before each sample (the first
        one before it); None when the run recorded no usable window."""
        if not self.shapes:
            return None
        index = np.clip(np.searchsorted(self.shape_starts, t, side="right") - 1, 0, len(self.shapes) - 1)
        bins = self.shapes[0][2].shape[1]
        x = phase / (2.0 * math.pi) * bins - 0.5
        i0 = np.floor(x).astype(np.int64)
        frac = x - i0
        i0 = np.mod(i0, bins)
        i1 = np.mod(i0 + 1, bins)
        waveform = np.zeros((len(t), 3))
        mean_w = np.zeros((len(t), 3))
        for s in np.unique(index):
            rows = index == s
            table, mean = self.shapes[int(s)][2], self.shapes[int(s)][3]
            waveform[rows] = (table[:, i0[rows]] * (1.0 - frac[rows]) + table[:, i1[rows]] * frac[rows]).T
            mean_w[rows] = mean
        return waveform, mean_w

    @staticmethod
    def _pulse(phase: np.ndarray, env: dict) -> np.ndarray:
        """Raised-cosine tooth pulse minus its mean, per component, whose duty reproduces the envelope mean."""
        span = env["max"] - env["min"]
        safe = np.where(span > 0.0, span, 1.0)
        rel_mean = np.clip((env["mean"] - env["min"]) / safe, 0.0, 1.0)
        u = (phase / (2.0 * math.pi))[:, None]
        low = rel_mean <= 0.5
        duty = np.clip(np.where(low, 2.0 * rel_mean, 2.0 * (1.0 - rel_mean)), MIN_DUTY, 1.0)
        pulse = np.where(u < duty, 0.5 - 0.5 * np.cos(2.0 * math.pi * np.minimum(u / duty, 1.0)), 0.0)
        waveform = np.where(low, pulse, 1.0 - pulse)
        mean_w = np.where(low, 0.5 * duty, 1.0 - 0.5 * duty)
        return np.where(span > 0.0, waveform - mean_w, 0.0)

    def _chatter(self, t: np.ndarray) -> np.ndarray:
        inp = self.inp
        out = np.zeros((len(t), 3))
        if not len(inp.macro_t_s) or not np.any(inp.chatter_freq_Hz > 0.0):
            return out
        span = inp.macro_span_s
        f = np.asarray(inp.chatter_freq_Hz, dtype=np.float64)
        starts = inp.macro_t_s - span
        cumulative = np.concatenate([[0.0], np.cumsum(2.0 * math.pi * f * span)])
        m = np.clip(np.searchsorted(inp.macro_t_s, t, side="left"), 0, len(f) - 1)
        phase = cumulative[m] + 2.0 * math.pi * f[m] * (t - starts[m])
        amplitude_rows = 0.5 * inp.vibration_pp_m * np.sqrt(np.clip(inp.chatter_index, 0.0, 1.0)) * (f > 0.0)
        amplitude = self._macro_at(amplitude_rows, t)
        freq_now = f[m]
        mode = self.chatter_mode
        force = A.chatter_force_N(amplitude, np.where(freq_now > 0.0, freq_now, mode[0]), mode)
        force = np.where(freq_now > 0.0, force, 0.0)
        out[:, 0] = force * np.cos(phase)
        out[:, 1] = force * np.sin(phase)
        return out

    def impact_block(self, j: int) -> np.ndarray | None:
        """Force on the enclosure sheet [B] from chip impacts (band-limited half-sine pulses)."""
        if j < 0 or j >= self.n_blocks or "impacts" not in self.sources:
            return None
        key = ("impact", j)
        if key in self._cache:
            return self._cache[key]
        q = self.q
        cutoff = float(self.p["resample_cutoff_Hz"])
        half = int(round(float(self.p["resample_half_width"])))
        slot = float(q["impact_subslot_s"])
        pad = (half + 1) / self.fs
        t0 = j * self.B / self.fs - pad - IMPACT_MAX_PULSE_S
        t1 = (j + 1) * self.B / self.fs + pad
        q0 = int(math.floor(max(t0, 0.0) / slot))
        q1 = max(int(math.ceil(t1 / slot)), q0)
        centres = (q0 + np.arange(q1 - q0) + 0.5) * slot
        theta, omega = self._spindle_at(centres)
        teeth = self._teeth_at(centres)
        radius = self._radius_at(centres)
        chips_per_s = teeth * np.abs(omega) / (2.0 * math.pi) * self._contact_at(centres)
        rate = chips_per_s * float(q["chip_impact_fraction"])
        stream = stream_key(STREAMS["impacts"])
        draws = [np.empty(len(centres)) for _ in range(3)]
        for b, draw in enumerate(draws):
            K.hash_uniforms(self.seed, stream, q0, b, draw)
        hit = (rate > 0.0) & (draws[0] < np.minimum(rate * slot, 1.0))
        mass_rate = self._macro_at(self.inp.chip_rate_kg_s, centres)
        per_chip = np.divide(mass_rate, chips_per_s, out=np.zeros(len(centres)), where=chips_per_s > 0.0)
        mass = np.where(per_chip > 0.0, per_chip, float(q["chip_default_mass_kg"]))
        speed_ref = (np.full(len(centres), self.inp.chip_speed_m_s) if self.inp.chip_speed_m_s else
                     float(q["chip_speed_ratio"]) * np.abs(omega) * radius)
        spread = float(q["chip_speed_spread"])
        speed = speed_ref * (1.0 - spread + 2.0 * spread * draws[2])
        hit &= speed > 0.0
        when = (q0 + np.arange(len(centres)) + draws[1]) * slot
        times = when[hit]
        strengths = (mass * (1.0 + float(q["chip_restitution"])) * speed)[hit]
        durations = list(zip(mass[hit], speed[hit]))
        out = np.zeros(self.B)
        if len(times):
            density = float(self.inp.chip_density_kg_m3 or self.p["sheet_density_kg_m3"])
            masses = np.array([d[0] for d in durations])
            speeds = np.array([d[1] for d in durations])
            radii = np.cbrt(3.0 * masses / (4.0 * math.pi * density))
            tau = A.hertz_duration_s(masses, radii, speeds, float(self.inp.chip_youngs_Pa or q["chip_youngs_modulus_Pa"]),
                                     float(self.inp.chip_poisson or q["chip_poisson"]),
                                     float(self.p["sheet_youngs_modulus_Pa"]), float(self.p["sheet_poisson"]))
            points = (np.arange(IMPACT_PULSE_POINTS) + 0.5) / IMPACT_PULSE_POINTS
            weights = np.sin(math.pi * points)
            weights = weights / weights.sum()
            pulse_t = (np.asarray(times)[:, None] + tau[:, None] * points[None, :]).reshape(-1)
            pulse_j = (np.asarray(strengths)[:, None] * weights[None, :]).reshape(-1)
            K.place_impulses(np.ascontiguousarray(pulse_t), np.ascontiguousarray(pulse_j), j * self.B, self.fs, cutoff,
                             half, out)
        out *= self._valid(j)
        self._cache[key] = out
        return out

    def _contact_at(self, t: np.ndarray) -> np.ndarray:
        """1 where the tool cuts (the force envelope spans a range or the fast force varies), else 0."""
        inp = self.inp
        if len(inp.env_t_s):
            span = np.abs(inp.env_max_N - inp.env_min_N).sum(axis=1)
            index = np.clip(np.searchsorted(inp.env_t_s, t, side="right") - 1, 0, len(inp.env_t_s) - 1)
            inside = (t >= inp.env_t_s[0]) & (t < inp.env_t_s[-1] + inp.env_span_s)
            return ((span[index] > 0.0) & inside).astype(np.float64)
        return np.zeros(len(t))

    # ------------------------------------------------------------------ airborne sources
    def _tone_T(self, freq: np.ndarray) -> np.ndarray:
        return np.sqrt(self._outside_T(np.abs(freq)))

    def _motor(self, j: int, t: np.ndarray, angle: np.ndarray, speed: np.ndarray, power_W: float, pole_pairs: int,
               f_switch: float, factor: float, offset_dB: float, shaper: str) -> np.ndarray:
        """One electric motor at mechanical ``angle`` (rad) and ``speed`` (rad/s): magnetic tone at 2 f_e, inverter
        tones at f_sw +- 2 f_e and broadband noise, their shares of the empirical sound power."""
        q = self.q
        power = A.motor_power_W(power_W, speed, offset_dB)
        p_rms = np.sqrt(self.rho0 * self.c0 * power * factor)
        tonal = float(q["motor_tonal_fraction"])
        pwm = float(q["motor_pwm_fraction"])
        electrical = angle * pole_pairs
        f_e = speed * pole_pairs / (2.0 * math.pi)
        tone = math.sqrt(2.0 * tonal) * p_rms * np.sin(2.0 * electrical) * self._tone_T(2.0 * f_e)
        carrier = 2.0 * math.pi * f_switch * t
        side = math.sqrt(pwm) * p_rms * (np.sin(carrier + 2.0 * electrical) * self._tone_T(f_switch + 2.0 * f_e)
                                         + np.sin(carrier - 2.0 * electrical) * self._tone_T(f_switch - 2.0 * f_e))
        broadband = math.sqrt(max(1.0 - tonal - pwm, 0.0)) * p_rms * self.shaped(shaper, j)
        return tone + side + broadband

    def outside_block(self, j: int) -> np.ndarray:
        """Airborne sources outside the enclosure (motors, pump) at the listener, block j."""
        t = self._times(j)
        theta, omega = self._spindle_at(t)
        q = self.q
        out = np.zeros(self.B)
        if "spindle_motor" in self.sources:
            ratio = float(q["spindle_motor_ratio"])
            out += self._motor(j, t, theta * ratio, np.abs(omega) * ratio, self.spindle_power_W,
                               int(round(float(q["spindle_motor_pole_pairs"]))), float(q["spindle_inverter_frequency_Hz"]),
                               self.factor_motor, float(q["spindle_motor_level_offset_dB"]), "motor")
        if "axis_motors" in self.sources and self.axis_pos.size:
            index = self._frame_index(t)
            for a, name in enumerate(self.inp.axis_names):
                lead = float(self.inp.axis_lead_m.get(name, q["ballscrew_lead_m"]))
                linear = self.inp.axis_linear[a] if a < len(self.inp.axis_linear) else True
                pos = np.interp(t, self.frame_t, self.axis_pos[:, a]) if len(self.frame_t) > 1 else np.zeros(len(t))
                vel = self.axis_vel[index, a]
                angle = 2.0 * math.pi * pos / lead if linear else pos
                speed = 2.0 * math.pi * np.abs(vel) / lead if linear else np.abs(vel)
                power = float(self.inp.axis_motor_power_W.get(name, q["axis_motor_power_W"]))
                out += self._motor(j, t, angle, speed, power, int(round(float(q["axis_motor_pole_pairs"]))),
                                   float(q["axis_inverter_frequency_Hz"]), self.factor_motor,
                                   float(q["axis_motor_level_offset_dB"]), f"axis:{name}")
        if "pump" in self.sources:
            out += self._pump(j, t)
        return out * self._valid(j)

    def _pump(self, j: int, t: np.ndarray) -> np.ndarray:
        q = self.q
        jets = self._jets_at(float(t[len(t) // 2]))
        kinds = {jet.kind for jet in jets if jet.flow_m3_s > 0.0 and jet.kind in LIQUID_JET_KINDS}
        if not kinds:
            return np.zeros(len(t))
        power = 0.0
        for kind in sorted(kinds):
            power += A.pump_power_W(float(self.inp.pumps.get(kind, self.inp.pumps.get("flood", q["pump_power_W"]))))
        on = self._coolant_on_at(t)
        p_rms = math.sqrt(self.rho0 * self.c0 * power * self.factor_pump) * on
        tonal = float(q["pump_tonal_fraction"])
        f_blade = float(q["pump_blades"]) * float(q["pump_speed_rad_s"]) / (2.0 * math.pi)
        tone = math.sqrt(2.0 * tonal) * p_rms * np.sin(2.0 * math.pi * f_blade * t) * self._tone_T(np.array([f_blade]))[0]
        return tone + math.sqrt(1.0 - tonal) * p_rms * self.shaped("pump", j)

    def inside_airborne_block(self, j: int) -> np.ndarray:
        """Airborne sources in the cutting zone (splash, air jets) at the listener (enclosure folded in)."""
        t = self._times(j)
        out = np.zeros(self.B)
        jets = self._jets_at(float(t[len(t) // 2]))
        if not jets:
            return out
        on = self._coolant_on_at(t)
        if "splash" in self.sources:
            power = sum(A.splash_power_W(jet.flow_m3_s, jet.speed_m_s, self.liquid_density) for jet in jets
                        if jet.kind in LIQUID_JET_KINDS and jet.flow_m3_s > 0.0)
            if power > 0.0:
                out += math.sqrt(self.rho0 * self.c0 * power * self.factor_inside) * on * self.shaped("splash", j)
        if "air_jets" in self.sources:
            for jet in jets:
                if jet.kind in AIR_JET_KINDS and jet.speed_m_s > 0.0:
                    power = A.lighthill_power_W(jet.speed_m_s, jet.diameter_m)
                    key = self._air_shaper(jet.speed_m_s, jet.diameter_m)
                    out += math.sqrt(self.rho0 * self.c0 * power * self.factor_inside) * on * self.shaped(key, j)
        return out * self._valid(j)

    # ------------------------------------------------------------------ output
    def _component(self, j: int, kind: str, axis: int) -> np.ndarray | None:
        force = self.force_block(j, kind)
        return None if force is None else np.ascontiguousarray(force[:, axis])

    def output_block(self, J: int) -> dict:
        inside = np.zeros(self.B)
        if any(s in self.sources for s in CONTACT_SOURCES):
            for axis, name in enumerate("xyz"):
                inside += self.firs[f"contact_{name}"].output(J, lambda j, axis=axis: self._component(j, "contact", axis))
        if any(s in self.sources for s in SPINDLE_SOURCES):
            for axis, name in enumerate("xy"):
                inside += self.firs[f"spindle_{name}"].output(J, lambda j, axis=axis: self._component(j, "spindle", axis))
        inside += self.inside_airborne_block(J)
        outside = self.outside_block(J)
        if "impacts" in self.sources:
            outside += self.firs["panel"].output(J, self.impact_block)
        for key in [k for k in self._cache if isinstance(k, tuple) and k[-1] < J - 3]:
            del self._cache[key]
        return {"inside": inside, "outside": outside, "total": inside + outside}


# ----------------------------------------------------------------------------- public functions
def render_window(run_or_inputs, t0_s: float, t1_s: float, sources=None, listener: str | None = None,
                  split: bool = False) -> SoundWindow:
    """Pressure at the listener (Pa, 44.1 kHz) over [t0_s, t1_s): the same samples the whole run's audio has there."""
    inputs = run_or_inputs if isinstance(run_or_inputs, SoundInputs) else from_run(run_or_inputs)
    synth = Synth(inputs, sources, listener)
    n0 = max(int(math.floor(t0_s * synth.fs)), 0)
    n1 = min(int(math.ceil(t1_s * synth.fs)), synth.n_total)
    if n1 <= n0:
        return SoundWindow(synth.fs, n0, np.zeros(0), {}, list(inputs.notes))
    parts = {"inside": [], "outside": [], "total": []}
    for J in range(n0 // synth.B, (n1 - 1) // synth.B + 1):
        block = synth.output_block(J)
        lo = max(n0 - J * synth.B, 0)
        hi = min(n1 - J * synth.B, synth.B)
        for key in parts:
            parts[key].append(block[key][lo:hi])
    buses = {k: np.concatenate(v) for k, v in parts.items() if k != "total"} if split else {}
    return SoundWindow(synth.fs, n0, np.concatenate(parts["total"]), buses, list(inputs.notes))


def render_audio(run_dir, out=None, sources=None, listener: str | None = None, write_json: bool = True) -> dict:
    """Write the run's ``audio.wav`` (44.1 kHz, 16-bit PCM, calibrated: see docs/models/sound.md "Levels") and
    ``audio.json`` (calibration, listener, levels, clipped samples); returns the summary."""
    from ..render.rundir import RunData

    run = RunData(run_dir)
    inputs = from_run(run)
    synth = Synth(inputs, sources, listener)
    full_scale = float(synth.p["full_scale_Pa"])
    dither = bool(synth.p["dither"])
    path = Path(out) if out is not None else run.path / "audio.wav"
    comment = (f"cnctwin sound: full scale {full_scale:g} Pa peak (32767), reference 2e-5 Pa; "
               f"listener {synth.listener}; sources {','.join(synth.sources)}")
    writer = D.WavWriter(path, int(round(synth.fs)), comment)
    weighting = D.CentredFIR(D.design_fir(lambda f: 10.0 ** (A.a_weighting_dB(f, synth.p) / 20.0), synth.fs), synth.B)
    totals = {}
    energy = {"total": 0.0, "inside": 0.0, "outside": 0.0, "a": 0.0}
    peak = 0.0
    clipped = 0
    count = 0

    def total_block(J: int):
        if J < 0 or J >= synth.n_blocks:
            return None
        if J not in totals:
            totals[J] = synth.output_block(J)
        return totals[J]["total"]

    for J in range(synth.n_blocks):
        a_weighted = weighting.output(J, total_block)      # computes output blocks J - 1 .. J + 1 as needed
        block = totals[J]
        n_valid = min(synth.B, synth.n_total - J * synth.B)
        pressure = block["total"][:n_valid]
        samples, clipped_block = D.quantise(pressure, J * synth.B, synth.seed, full_scale, dither)
        writer.write(samples)
        clipped += clipped_block
        peak = max(peak, float(np.max(np.abs(pressure))) if n_valid else 0.0)
        energy["total"] += float(np.dot(pressure, pressure))
        energy["inside"] += float(np.dot(block["inside"][:n_valid], block["inside"][:n_valid]))
        energy["outside"] += float(np.dot(block["outside"][:n_valid], block["outside"][:n_valid]))
        energy["a"] += float(np.dot(a_weighted[:n_valid], a_weighted[:n_valid]))
        count += n_valid
        for old in [k for k in totals if k < J - 1]:
            del totals[old]
    writer.close()
    mean = {k: (v / count if count else 0.0) for k, v in energy.items()}
    summary = {
        "format": "cnctwin-audio", "version": 1, "file": path.name if out is None else str(path),
        "sample_rate_Hz": synth.fs, "samples": count, "duration_s": count / synth.fs, "bits": 16,
        "calibration": {"full_scale_Pa": full_scale, "lsb_Pa": full_scale / D.INT16_FULL_SCALE,
                        "reference_Pa": float(synth.p["reference_pressure_Pa"]),
                        "full_scale_sine_dB": A.level_dB(full_scale / math.sqrt(2.0)), "dither": dither},
        "listener": {"kind": synth.listener, "distance_m": float(synth.p["operator_distance_m"] if synth.listener ==
                                                                   "operator" else synth.p["inside_distance_m"])},
        "sources": list(synth.sources),
        "levels": {"Leq_dB": A.level_dB(math.sqrt(mean["total"])), "LAeq_dB": A.level_dB(math.sqrt(mean["a"])),
                   "inside_Leq_dB": A.level_dB(math.sqrt(mean["inside"])),
                   "outside_Leq_dB": A.level_dB(math.sqrt(mean["outside"])), "peak_Pa": peak},
        "clipped_samples": clipped, "notes": list(inputs.notes)}
    if write_json:
        target = path.with_name("audio.json") if out is None else path.with_suffix(".json")
        tmp = target.with_name(f".{target.name}.tmp")
        tmp.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(target)
    return summary
