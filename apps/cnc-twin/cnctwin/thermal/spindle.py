"""Physical spindle thermal model (docs/models/thermal.md 4.1): bearing friction, motor losses, a lumped
network of shaft, bearing sets and housing (with a cooling jacket when the spindle has one) and the axial
growth of the nose from the shaft and housing temperatures and lengths.

Heat sources at the spindle speed n (rpm):

* **bearing friction**, per bearing (Palmgren 1959; Harris & Kotzalas 2007 ch. 6):
  ``M = M_v + M_l``, ``M_v = 1e-7 f0 (nu n)^(2/3) dm^3`` [N mm] for ``nu n >= 2000`` (mm2/s rpm), else
  ``160e-7 f0 dm^3``; ``M_l = f1 F_beta dm``, ``f1 = z (P0 / C0)^y`` with the static equivalent load P0
  (ISO 76) and the friction load F_beta of the bearing type (angular-contact ball:
  ``max(0.9 Fa cot(alpha) - 0.1 Fr, Fr)``); ``nu`` is the lubricant's kinematic viscosity at the bearing
  node's temperature (Walther's relation, ASTM D341, through the 40 and 100 degC values). Heat ``M omega``:
  the share ``bearing_heat_rolling_fraction`` enters the rolling elements (the bearing node), the rest the
  rings, ``bearing_heat_inner_ring_fraction`` of it the inner ring (shaft) and the remainder the outer ring
  (housing) (Blok's equal partition at a rolling contact of like bodies).
* **motor** no-load loss ``r P_rated min(1, n / n_base)``: a built-in motor puts its rotor share into the
  shaft and the stator share into the housing; a separate motor (coupled, belt, gear) a fraction into the
  housing (the rest leaves with its own fan).
* the heat the **tool holder** conducts into the nose (thermal's tool ladder) enters the shaft.

Links: bearing node <-> shaft and <-> housing through the rolling contacts (per bearing and side,
``G_ref (dm / dm_ref) sqrt(omega / omega_ref)``: the mean rise of a high-Peclet moving contact falls as
1/sqrt(Pe), Jaeger 1942); shaft <-> housing through the rotating air annulus between the bearing sets and
shaft <-> ambient at the exposed nose (rotating cylinder, Kays & Bjorklund 1958); housing <-> ambient (free
convection of a vertical plate, Churchill & Chu 1975, plus linearised radiation); housing <-> jacket
coolant (a stream over the wetted jacket: ``G = m c_p (1 - exp(-h A / (m c_p)))``, h of pipe flow in the
channels); housing <-> machine structure (the head's mounting).

Integration over a step: the convection coefficients at the start of each sub-step (at most
``spindle_network_max_step_s``), the bearing heat linearised in its node temperature (dP/dT < 0 through the
viscosity: it stiffens the system), then the linear network integrated exactly for constant sources and
boundary temperatures (symmetric eigen-decomposition of C^-1/2 K C^-1/2). Every loss is the exact integral
of its link over the step, so the energy ledger closes to round-off.

Growth toward the work: ``eps_s(T_S) L_n + eps_h(T_H) L_h`` (free strains from the mean-CTE tables relative
to the reference temperature), L_n = nose to the axially fixed bearing set, L_h = housing length from the
head's axial datum to that bearing set.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

import numpy as np

from ..db.models import params
from ..units import REFERENCE_TEMP_K, ZERO_DEGC_K
from . import convection as cv
from . import props
from .inputs import MM, number, plain
from .ledger import SolverLedger

RAD_S_PER_RPM = 2.0 * math.pi / 60.0
DRIVE_TYPES = ("built_in", "coupled", "belt", "gear")
BOUNDARIES = ("air", "structure", "jacket")
REF_K = REFERENCE_TEMP_K


@functools.lru_cache(maxsize=1)
def _p() -> dict:
    return dict(params("thermal-machine"))


@functools.lru_cache(maxsize=None)
def _mat(item_id: str):
    return props.material_table_by_id(item_id)


# ----------------------------------------------------------------------------------------------- lubricant
def walther_coefficients(nu40_m2_s: float, nu100_m2_s: float) -> tuple[float, float]:
    """(A, B) of Walther's relation log10 log10(nu + 0.7) = A - B log10(T) (nu in mm2/s, T in K;
    ASTM D341) through the kinematic viscosities at 40 and 100 degC."""
    z40 = math.log10(math.log10(nu40_m2_s * 1e6 + 0.7))
    z100 = math.log10(math.log10(nu100_m2_s * 1e6 + 0.7))
    t40 = math.log10(ZERO_DEGC_K + 40.0)
    t100 = math.log10(ZERO_DEGC_K + 100.0)
    b = (z40 - z100) / (t100 - t40)
    return z40 + b * t40, b


def viscosity_m2_s(a: float, b: float, temp_K: float) -> tuple[float, float]:
    """(nu, d nu / dT) of a lubricant (m2/s, m2/s per K) at temp_K (held at the ends of the validity range)."""
    p = _p()
    lo, hi = float(p["walther_min_temp_K"]), float(p["walther_max_temp_K"])
    t = min(max(float(temp_K), lo), hi)
    x = 10.0 ** (a - b * math.log10(t))              # log10(nu + 0.7)
    y = 10.0 ** x                                    # nu + 0.7 (mm2/s)
    nu_mm = y - 0.7
    slope = 0.0 if (temp_K <= lo or temp_K >= hi) else -y * x * b * math.log(10.0) / t
    return nu_mm * 1e-6, slope * 1e-6


# ----------------------------------------------------------------------------------------------- bearings
@dataclass
class BearingNode:
    """One bearing set (identical bearings) and its node in the network."""

    name: str                 # front | rear | ... (from the machine file)
    kind: str                 # angular_contact | cylindrical_roller | tapered_roller | deep_groove | thrust_ball
    count: int
    bore_m: float
    outer_m: float
    contact_angle_rad: float
    preload_N: float          # axial preload carried by each bearing
    c0_N: float               # basic static load rating of one bearing
    lubrication: str          # grease | oil_mist | oil_air | oil_bath | oil_jet
    f0: float
    f1_z: float
    f1_y: float
    visc_a: float             # Walther coefficients of the lubricant
    visc_b: float
    mass_kg: float            # whole set (rings and rolling elements)
    fixed: bool               # the axially locating set

    @property
    def dm_m(self) -> float:
        return 0.5 * (self.bore_m + self.outer_m)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "BearingNode":
        return cls(**values)


def _type_row(kind: str) -> int:
    kinds = list(_p()["bearing_types"])
    if kind not in kinds:
        raise ValueError(f"thermal: unknown bearing type {kind!r} (one of {', '.join(kinds)})")
    return kinds.index(kind)


def _lub_col(lubrication: str) -> int:
    lubs = list(_p()["bearing_lubrications"])
    if lubrication not in lubs:
        raise ValueError(f"thermal: unknown bearing lubrication {lubrication!r} (one of {', '.join(lubs)})")
    return lubs.index(lubrication)


def static_equivalent_N(kind: str, alpha_rad: float, axial_N: float, radial_N: float) -> float:
    """Static equivalent load P0 of one bearing (ISO 76)."""
    p = _p()
    if kind == "angular_contact":
        y0 = float(np.interp(alpha_rad, p["iso76_acbb_angles_rad"], p["iso76_acbb_y0"]))
        return max(radial_N, p["iso76_acbb_x0"] * radial_N + y0 * axial_N)
    if kind == "deep_groove":
        return max(radial_N, p["iso76_deep_groove_x0"] * radial_N + p["iso76_deep_groove_y0"] * axial_N)
    if kind == "thrust_ball":
        return axial_N
    if kind == "tapered_roller":
        y0 = p["iso76_tapered_y0_cot_factor"] / math.tan(max(alpha_rad, 1e-3))
        return max(radial_N, p["iso76_acbb_x0"] * radial_N + y0 * axial_N)
    return radial_N


def friction_load_N(kind: str, alpha_rad: float, axial_N: float, radial_N: float) -> float:
    """Palmgren's friction load F_beta of one bearing (harris-kotzalas-2007-bearings ch. 6)."""
    p = _p()
    if kind == "angular_contact":
        return max(p["fbeta_acbb_axial_factor"] * axial_N / math.tan(max(alpha_rad, 1e-3))
                   - p["fbeta_radial_factor"] * radial_N, radial_N)
    if kind == "deep_groove":
        return max(p["fbeta_deep_groove_axial_factor"] * axial_N - p["fbeta_radial_factor"] * radial_N, radial_N)
    if kind == "thrust_ball":
        return axial_N
    if kind == "tapered_roller":
        return max(2.0 * p["iso281_tapered_y_cot_factor"] / math.tan(max(alpha_rad, 1e-3)) * axial_N, radial_N)
    return radial_N


def bearing_moment_Nm(b: BearingNode, rpm, nu_m2_s: float, radial_N: float = 0.0):
    """(M, dM/dnu) of one bearing of the set at the speeds ``rpm`` (array; N m and N m per m2/s)."""
    p = _p()
    rpm = np.abs(np.asarray(rpm, dtype=float))
    dm_mm = b.dm_m * 1e3
    nu_mm = nu_m2_s * 1e6
    x = nu_mm * rpm
    c = float(p["palmgren_mv_coefficient"])
    e = float(p["palmgren_mv_exponent"])
    high = x >= float(p["palmgren_mv_limit"])
    mv = np.where(high, c * b.f0 * np.power(np.maximum(x, 1e-300), e) * dm_mm ** 3,
                  float(p["palmgren_mv_low_coefficient"]) * b.f0 * dm_mm ** 3)
    dmv = np.where(high, e * mv / max(nu_mm, 1e-12), 0.0) * 1e6       # per (m2/s)
    p0 = static_equivalent_N(b.kind, b.contact_angle_rad, b.preload_N, radial_N)
    f1 = b.f1_z * (p0 / b.c0_N) ** b.f1_y if (b.f1_y > 0.0 and b.c0_N > 0.0 and p0 > 0.0) else \
        (b.f1_z if b.f1_y == 0.0 else 0.0)
    ml = f1 * friction_load_N(b.kind, b.contact_angle_rad, b.preload_N, radial_N) * dm_mm
    return (mv + ml) * 1e-3, dmv * 1e-3


# ----------------------------------------------------------------------------------------------- network
def _add_link(k: np.ndarray, i: int, j: int, g: float) -> None:
    k[i, i] += g
    k[j, j] += g
    k[i, j] -= g
    k[j, i] -= g


def exact_step(cap: np.ndarray, k_eff: np.ndarray, rhs: np.ndarray, theta0: np.ndarray,
               h: float) -> tuple[np.ndarray, np.ndarray]:
    """(theta(h), integral of theta over [0, h]) of C theta' = rhs - K theta, exact (K symmetric positive
    definite, C diagonal): theta = theta_eq + C^-1/2 Q exp(-L t) Q^T C^1/2 (theta0 - theta_eq)."""
    theta_eq = np.linalg.solve(k_eff, rhs)
    sq = np.sqrt(cap)
    lam, q = np.linalg.eigh(k_eff / np.outer(sq, sq))
    coef = q.T @ (sq * (theta0 - theta_eq))
    lh = lam * h
    decay = np.exp(-lh)
    integ = np.where(lh > 1e-12, -np.expm1(-lh) / np.where(lam > 0.0, lam, 1.0), h)
    theta1 = theta_eq + (q @ (decay * coef)) / sq
    integral = theta_eq * h + (q @ (integ * coef)) / sq
    return theta1, integral


@dataclass
class SpindleNetwork:
    """Lumped network of one spindle: node 0 the shaft, then one node per bearing set, last the housing."""

    sid: str
    names: list
    cap_J_K_nodes: list
    theta_K: list                     # excess temperatures over 20 degC
    bearings: list                    # BearingNode per bearing node (nodes 1..k)
    shaft_d_m: float
    span_m: float                     # shaft length between the bearing sets (annulus)
    nose_d_m: float
    nose_area_m2: float
    housing_area_m2: float
    housing_height_m: float
    housing_emissivity: float
    shaft_emissivity: float
    g_struct_W_K: float
    jacket_g_W_K: float
    jacket_offset_dK: float
    jacket_inlet_K: float | None
    bc_W_K: float                     # rolling-contact conductance per bearing and side at the reference
    bc_ref_dm_m: float
    bc_ref_rad_s: float
    bc_min_rad_s: float
    rolling_frac: float
    inner_frac: float
    motor_W: float                    # no-load loss at and above the base speed
    motor_base_rad_s: float
    motor_to_shaft: float
    motor_to_housing: float
    shaft_material: str
    housing_material: str
    nose_length_m: float              # nose to the axially fixed bearing set
    housing_length_m: float           # head's axial datum to the fixed bearing set
    T_ref_K: float
    max_step_s: float
    ledger: SolverLedger = field(default_factory=lambda: SolverLedger("spindle"))
    totals: dict = field(default_factory=dict)
    last: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- body interface (as LumpedBody)
    @property
    def n_nodes(self) -> int:
        return len(self.cap_J_K_nodes)

    @property
    def T_K(self) -> float:
        """Shaft temperature (the nose the tool holder sits in)."""
        return REF_K + float(self.theta_K[0])

    @property
    def housing_T_K(self) -> float:
        return REF_K + float(self.theta_K[-1])

    @property
    def cap_J_K(self) -> float:
        return float(sum(self.cap_J_K_nodes))

    @property
    def g_W_K(self) -> float:
        """Conductance of the boundary links at rest and at the reference temperature (W/K, informative)."""
        _, gb = self._links(0.0, REF_K)
        return float(gb.sum())

    def temps_K(self) -> np.ndarray:
        return REF_K + np.asarray(self.theta_K, dtype=float)

    def energy(self) -> float:
        return float(np.dot(self.cap_J_K_nodes, self.theta_K))

    def set_temperature(self, temp_K: float) -> None:
        self.theta_K = [float(temp_K) - REF_K] * self.n_nodes

    def set_temperatures(self, temps_K) -> None:
        self.theta_K = [float(t) - REF_K for t in temps_K]

    # ---------------------------------------------------------------- physics
    def _sources(self, omega_abs: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        """(power per node at the current temperatures, dP/dT at the bearing nodes, parts in W)."""
        n = self.n_nodes
        power = np.zeros(n)
        slope = np.zeros(n)
        w = np.abs(np.asarray(omega_abs, dtype=float)).ravel()
        if w.size == 0:
            w = np.zeros(1)
        elif w.size > 1 and float(w.max()) == float(w.min()):
            w = w[:1]                                  # a constant speed over the step: one evaluation
        rpm = w / RAD_S_PER_RPM
        bearing_total = 0.0
        for i, b in enumerate(self.bearings, start=1):
            nu, dnu = viscosity_m2_s(b.visc_a, b.visc_b, REF_K + float(self.theta_K[i]))
            moment, dmoment = bearing_moment_Nm(b, rpm, nu)
            heat = b.count * float(np.mean(moment * w))
            dheat = b.count * float(np.mean(dmoment * w)) * dnu
            power[i] += self.rolling_frac * heat
            ring = (1.0 - self.rolling_frac) * heat
            power[0] += self.inner_frac * ring
            power[-1] += (1.0 - self.inner_frac) * ring
            slope[i] = min(dheat, 0.0)
            bearing_total += heat
        motor = 0.0
        if self.motor_W > 0.0 and self.motor_base_rad_s > 0.0:
            motor = self.motor_W * float(np.mean(np.minimum(w / self.motor_base_rad_s, 1.0)))
        power[0] += self.motor_to_shaft * motor
        power[-1] += self.motor_to_housing * motor
        return power, slope, {"bearing_W": bearing_total, "motor_W": motor * (self.motor_to_shaft + self.motor_to_housing)}

    def _links(self, omega_mean: float, T_env_K: float) -> tuple[np.ndarray, np.ndarray]:
        """(internal conductance Laplacian, boundary conductances [node, air/structure/jacket]) at the current
        temperatures."""
        n = self.n_nodes
        temps = self.temps_K()
        k = np.zeros((n, n))
        speed = math.sqrt(max(abs(omega_mean), self.bc_min_rad_s) / self.bc_ref_rad_s)
        for i, b in enumerate(self.bearings, start=1):
            g = self.bc_W_K * b.count * (b.dm_m / self.bc_ref_dm_m) * speed
            _add_link(k, i, 0, g)
            _add_link(k, i, n - 1, g)
        if self.span_m > 0.0:
            h = float(cv.h_rotating_cylinder(omega_mean, self.shaft_d_m, temps[0], temps[-1]))
            _add_link(k, 0, n - 1, h * math.pi * self.shaft_d_m * self.span_m)
        gb = np.zeros((n, len(BOUNDARIES)))
        h_nose = float(cv.h_rotating_cylinder(omega_mean, self.nose_d_m, temps[0], T_env_K)) + \
            float(cv.h_radiation(self.shaft_emissivity, temps[0], T_env_K))
        gb[0, 0] = h_nose * self.nose_area_m2
        h_out = cv.h_vertical_plate(temps[-1], T_env_K, self.housing_height_m) + \
            float(cv.h_radiation(self.housing_emissivity, temps[-1], T_env_K))
        gb[n - 1, 0] = h_out * self.housing_area_m2
        gb[n - 1, 1] = self.g_struct_W_K
        gb[n - 1, 2] = self.jacket_g_W_K
        return k, gb

    def heat_W(self, omega_abs) -> float:
        """Heat into the network at the current temperatures (W, time mean over the samples)."""
        power, _, _ = self._sources(np.asarray(omega_abs, dtype=float))
        return float(power.sum())

    def bearing_heat_W(self, omega_abs) -> float:
        """Bearing friction power at the current temperatures (W, time mean over the samples)."""
        return float(self._sources(np.asarray(omega_abs, dtype=float))[2]["bearing_W"])

    def jacket_inlet_temp_K(self, T_env_K: float) -> float:
        return float(self.jacket_inlet_K) if self.jacket_inlet_K is not None else T_env_K + self.jacket_offset_dK

    def step(self, dt: float, omega_abs, heat_in_J: float, tol: float, time_s: float, T_env_K: float,
             T_struct_K: float) -> dict:
        """Advance dt with the spindle speeds of the step (samples) and the holder heat (J, into the shaft);
        returns the energy that left over the step by boundary {air, structure, jacket} (J)."""
        u0 = self.energy()
        out = {name: 0.0 for name in BOUNDARIES}
        cap = np.asarray(self.cap_J_K_nodes, dtype=float)
        if dt <= 0.0:
            if heat_in_J:
                theta = list(self.theta_K)
                theta[0] += heat_in_J / cap[0]
                self.theta_K = theta
                self.ledger.book(u0, self.energy(), heat_in_J, 0.0, 0.0, tol, time_s)
            return out
        w = np.abs(np.asarray(omega_abs, dtype=float)).ravel()
        omega_mean = float(w.mean()) if w.size else 0.0
        n_sub = max(1, int(math.ceil(dt / self.max_step_s - 1e-12)))
        h = dt / n_sub
        sinks = np.array([T_env_K - REF_K, T_struct_K - REF_K, self.jacket_inlet_temp_K(T_env_K) - REF_K])
        sources = 0.0
        parts = {"bearing_J": 0.0, "motor_J": 0.0, "holder_J": float(heat_in_J)}
        for _ in range(n_sub):
            theta0 = np.asarray(self.theta_K, dtype=float)
            power, slope, info = self._sources(w)
            power[0] += heat_in_J / dt
            k, gb = self._links(omega_mean, T_env_K)
            k_eff = k + np.diag(gb.sum(axis=1) - slope)
            rhs = power - slope * theta0 + gb @ sinks
            theta1, integral = exact_step(cap, k_eff, rhs, theta0, h)
            correction = float(np.dot(slope, integral - theta0 * h))
            sources += float(power.sum()) * h + correction
            parts["bearing_J"] += info["bearing_W"] * h + correction
            parts["motor_J"] += info["motor_W"] * h
            for b, name in enumerate(BOUNDARIES):
                out[name] += float(np.dot(gb[:, b], integral - sinks[b] * h))
            self.theta_K = [float(x) for x in theta1]
        losses = sum(out.values())
        self.ledger.book(u0, self.energy(), sources, losses, 0.0, tol, time_s)
        for key, value in parts.items():
            self.totals[key] = self.totals.get(key, 0.0) + value
        for name, value in out.items():
            self.totals[f"to_{name}_J"] = self.totals.get(f"to_{name}_J", 0.0) + value
        self.last = {"bearing_W": parts["bearing_J"] / dt, "motor_W": parts["motor_J"] / dt,
                     "holder_W": heat_in_J / dt, **{f"to_{k}_W": v / dt for k, v in out.items()}}
        return out

    def warm_up(self, omega_rad_s: float, duration_s: float, T_env_K: float, T_struct_K: float, tol: float) -> None:
        """Run at a constant speed for duration_s (the warm-up program of a 'warm' start); the ledger is
        restarted at the reached state."""
        t = 0.0
        while t < duration_s - 1e-9:
            dt = min(self.max_step_s, duration_s - t)
            self.step(dt, np.array([omega_rad_s]), 0.0, tol, t, T_env_K, T_struct_K)
            t += dt
        self.reset_ledger()

    def reset_ledger(self) -> None:
        name = self.ledger.name
        self.ledger = SolverLedger(name)
        self.ledger.u_J = self.energy()
        self.ledger.u0_J = self.ledger.u_J
        self.totals = {}
        self.last = {}

    def growth_m(self) -> float:
        """Axial growth of the nose toward the work relative to the reference temperature (m)."""
        e_shaft = props.thermal_strain(_mat(self.shaft_material), self.T_K, self.T_ref_K)
        e_house = props.thermal_strain(_mat(self.housing_material), self.housing_T_K, self.T_ref_K)
        return e_shaft * self.nose_length_m + e_house * self.housing_length_m

    def growth_parts_m(self) -> dict:
        e_shaft = props.thermal_strain(_mat(self.shaft_material), self.T_K, self.T_ref_K)
        e_house = props.thermal_strain(_mat(self.housing_material), self.housing_T_K, self.T_ref_K)
        return {"shaft_m": e_shaft * self.nose_length_m, "housing_m": e_house * self.housing_length_m}

    # ---------------------------------------------------------------- checkpoint
    def to_dict(self) -> dict:
        out = {}
        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            if key == "bearings":
                out[key] = [b.to_dict() for b in value]
            elif key == "ledger":
                out[key] = value.to_dict()
            elif isinstance(value, np.ndarray):
                out[key] = [float(x) for x in value]
            elif isinstance(value, (list, tuple)):
                out[key] = [float(x) if isinstance(x, (float, np.floating)) else x for x in value]
            elif isinstance(value, dict):
                out[key] = {k: float(v) for k, v in sorted(value.items())}
            else:
                out[key] = value
        return out

    @classmethod
    def from_dict(cls, values: dict) -> "SpindleNetwork":
        values = dict(values)
        values["bearings"] = [BearingNode.from_dict(b) for b in values["bearings"]]
        values["ledger"] = SolverLedger.from_dict(values["ledger"])
        values["cap_J_K_nodes"] = [float(x) for x in values["cap_J_K_nodes"]]
        values["theta_K"] = [float(x) for x in values["theta_K"]]
        return cls(**values)

    def describe(self) -> dict:
        """Resolved parameters (SI) for reports."""
        return {"nodes": list(self.names), "capacity_J_K": [float(c) for c in self.cap_J_K_nodes],
                "bearings": [{"name": b.name, "type": b.kind, "count": b.count, "bore_m": b.bore_m,
                              "outer_m": b.outer_m, "contact_angle_rad": b.contact_angle_rad,
                              "preload_N": b.preload_N, "c0_N": b.c0_N, "lubrication": b.lubrication,
                              "f0": b.f0, "fixed": b.fixed} for b in self.bearings],
                "shaft_d_m": self.shaft_d_m, "span_m": self.span_m, "nose_length_m": self.nose_length_m,
                "housing_length_m": self.housing_length_m, "housing_area_m2": self.housing_area_m2,
                "housing_mass_kg": self.cap_J_K_nodes[-1] / _cp(self.housing_material),
                "g_struct_W_K": self.g_struct_W_K, "jacket_g_W_K": self.jacket_g_W_K,
                "motor_noload_W": self.motor_W, "motor_base_rad_s": self.motor_base_rad_s,
                "motor_to_shaft": self.motor_to_shaft, "motor_to_housing": self.motor_to_housing}


def _cp(item_id: str) -> float:
    mat = _mat(item_id)
    return float(np.interp(REF_K, mat.T_K, mat.cp_J_kgK))


def _rho(item_id: str) -> float:
    return float(_mat(item_id).rho_kg_m3[0])


# ----------------------------------------------------------------------------------------------- from a machine file
def lubrication_of(text) -> str | None:
    """Bearing lubrication from the machine file's free text (``bearing_lubrication``)."""
    if text is None:
        return None
    t = str(text).lower()
    if "grease" in t:
        return "grease"
    if "air" in t and "oil" in t:
        return "oil_air"
    if "mist" in t:
        return "oil_mist"
    if "jet" in t:
        return "oil_jet"
    if "bath" in t:
        return "oil_bath"
    return None


def drive_type_of(spindle: Mapping) -> str | None:
    """``drive_type`` of the spindle, else read from the free text ``drive``."""
    value = plain(spindle.get("drive_type"))
    if value:
        return str(value)
    text = plain(spindle.get("drive"))
    if not text:
        return None
    t = str(text).lower()
    if "motor spindle" in t or "built-in" in t or "built in" in t or "electro" in t:
        return "built_in"
    if "belt" in t:
        return "belt"
    if "gear" in t:
        return "gear"
    if "direct" in t or "inline" in t or "coupl" in t:
        return "coupled"
    return None


def _cooled(text) -> bool | None:
    if text is None:
        return None
    t = str(text).lower()
    if any(word in t for word in ("liquid", "water", "oil cool", "chiller", "jacket", "cooled motor")):
        return True
    if "air" in t or "none" in t or "uncooled" in t:
        return False
    return None


def _taper_size(nose) -> int | None:
    import re
    text = str(plain(nose) or "")
    m = re.search(r"(?:BT|CAT|CT|SK|ISO|DIN|NT|JT)\s*-?(30|40|45|50|60)\b", text, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"HSK-?[A-Z]?\s*(32|40|50|63|80|100|125)", text, re.I)
    if m:
        return {32: 30, 40: 30, 50: 30, 63: 40, 80: 45, 100: 50, 125: 50}[int(m.group(1))]
    return None


def motor_rating(spindle: Mapping) -> tuple[float | None, float | None]:
    """(rated power W, base speed rad/s) of the spindle drive: ``drive_power_kW``, else the S1 rating (or the
    main curve) of the first gear range; the base speed where that curve first reaches the fraction
    ``motor_base_power_fraction`` of its maximum."""
    p = _p()
    rated = number(spindle, (("drive_power_kW", 1e3, 0.0), ("drive_power_W", 1.0, 0.0)))
    base = None
    gears = spindle.get("gear_ranges") or []
    gear = gears[0] if gears and isinstance(gears[0], Mapping) else None
    if gear is not None:
        curve = plain(gear.get("power_kW"))
        if isinstance(curve, list) and curve and isinstance(curve[0], (list, tuple)):
            rpm = np.array([float(r[0]) for r in curve])
            kw = np.array([float(r[1]) for r in curve])
            top = float(kw.max()) if kw.size else 0.0
            if top > 0.0:
                reach = np.nonzero(kw >= p["motor_base_power_fraction"] * top)[0]
                base = float(rpm[reach[0]]) * RAD_S_PER_RPM if reach.size else None
                if rated is None:
                    rated = top * 1e3
                    for extra in gear.get("additional_ratings") or []:
                        name = str(plain(extra.get("rating")) or "")
                        if "S1" in name.upper():
                            scale = plain(extra.get("scale_of_peak"))
                            own = plain(extra.get("power_kW"))
                            if isinstance(own, list) and own and isinstance(own[0], (list, tuple)):
                                rated = max(float(r[1]) for r in own) * 1e3
                            elif scale is not None:
                                rated = float(scale) * top * 1e3
                            break
    return rated, base


def _reference_bore(spindle: Mapping, estimated: list) -> float:
    p = _p()
    bearings = spindle.get("bearings") or []
    for b in bearings:
        if isinstance(b, Mapping) and str(plain(b.get("position")) or "front") == "front":
            bore = number(b, (("bore_mm", MM, 0.0),))
            if bore:
                return float(bore)
    bore = number(spindle, (("bearing_id_mm", MM, 0.0),))
    if bore:
        return float(bore)
    sizes = list(p["default_front_bore_taper_sizes"])
    bores = list(p["default_front_bore_m"])
    size = _taper_size(spindle.get("nose"))
    if size is not None:
        estimated.append(f"front bearing bore (from the {size}-taper class)")
        return float(np.interp(size, sizes, bores))
    estimated.append("front bearing bore (40-taper class)")
    return float(np.interp(40, sizes, bores))


def _bearing(raw: Mapping, name: str, d_ref: float, lubrication: str, kind_default: str,
             estimated: list) -> BearingNode:
    p = _p()
    kind = str(plain(raw.get("type")) or kind_default)
    row = _type_row(kind)
    bore = number(raw, (("bore_mm", MM, 0.0),))
    if bore is None:
        bore = d_ref
        estimated.append(f"{name} bearing bore")
    outer = number(raw, (("outer_mm", MM, 0.0),))
    if outer is None:
        outer = bore * p["default_bearing_outer_ratio"]
        estimated.append(f"{name} bearing outer diameter")
    count = int(plain(raw.get("count")) or 0)
    if count <= 0:
        count = int(p["default_front_bearing_count"] if name == "front" else p["default_rear_bearing_count"])
        estimated.append(f"{name} bearing count")
    default_alpha = p["default_contact_angle_rad"] if kind == "angular_contact" else \
        (p["default_tapered_contact_angle_rad"] if kind == "tapered_roller" else 0.0)
    alpha = number(raw, (("contact_angle_deg", math.pi / 180.0, 0.0),))
    if alpha is None:
        alpha = default_alpha
        if kind in ("angular_contact", "tapered_roller"):
            estimated.append(f"{name} bearing contact angle")
    dm = 0.5 * (bore + outer)
    c0 = number(raw, (("static_load_rating_kN", 1e3, 0.0), ("static_load_rating_N", 1.0, 0.0)))
    if c0 is None:
        c0 = p["default_c0_per_dm2_Pa"] * dm * dm
        estimated.append(f"{name} bearing static load rating")
    preload = number(raw, (("preload_N", 1.0, 0.0), ("preload_kN", 1e3, 0.0)))
    if preload is None:
        preload = p["default_preload_c0_ratio"] * c0 if kind in ("angular_contact", "tapered_roller",
                                                                  "thrust_ball") else 0.0
        if kind in ("angular_contact", "tapered_roller", "thrust_ball"):
            estimated.append(f"{name} bearing preload")
    lub = lubrication_of(plain(raw.get("lubrication"))) or str(plain(raw.get("lubrication")) or "") or lubrication
    col = _lub_col(lub)
    f0 = float(np.asarray(p["bearing_f0"], dtype=float)[row, col])
    nu40 = number(raw, (("oil_viscosity_40C_mm2_s", 1e-6, 0.0),))
    nu100 = number(raw, (("oil_viscosity_100C_mm2_s", 1e-6, 0.0),))
    if nu40 is None or nu100 is None:
        nu40 = float(p["lubricant_viscosity_40C_m2_s"][col]) if nu40 is None else nu40
        nu100 = float(p["lubricant_viscosity_100C_m2_s"][col]) if nu100 is None else nu100
        estimated.append(f"{name} bearing lubricant viscosity ({lub})")
    a, b = walther_coefficients(nu40, nu100)
    width = p["bearing_width_ratio"] * (outer - bore)
    mass = number(raw, (("mass_kg", 1.0, 0.0),))
    if mass is None:
        mass = _rho(str(p["bearing_material"])) * math.pi / 4.0 * (outer * outer - bore * bore) * width * \
            p["bearing_solid_fraction"]
    fixed_raw = plain(raw.get("axially_fixed"))
    fixed = bool(fixed_raw) if fixed_raw is not None else (name == "front")
    return BearingNode(name, kind, count, float(bore), float(outer), float(alpha), float(preload), float(c0), lub,
                       f0, float(np.asarray(p["bearing_f1_z"])[row]), float(np.asarray(p["bearing_f1_y"])[row]),
                       a, b, float(mass) * count, fixed)


def build_network(sid: str, spindle: Mapping, ambient_K: float, start_K: float | None = None
                  ) -> tuple[SpindleNetwork, list]:
    """The network of one spindle of a machine file (raw mapping) and the list of the parameters taken from
    the model's defaults (db/models/thermal-machine.yaml) because the file does not give them."""
    p = _p()
    estimated: list = []
    thermal = spindle.get("thermal") or {}
    if not isinstance(thermal, Mapping):
        thermal = {}
    kind = str(plain(spindle.get("kind")) or "milling")
    d0 = _reference_bore(spindle, estimated)
    scale = d0 / p["default_reference_bore_m"]
    lub_text = lubrication_of(plain(spindle.get("bearing_lubrication")))
    lubrication = lub_text or str(p["default_lubrication"])
    if lub_text is None:
        estimated.append(f"bearing lubrication ({lubrication})")
    raw_sets = [b for b in (spindle.get("bearings") or []) if isinstance(b, Mapping)]
    if not raw_sets:
        estimated.append("bearing arrangement (default front and rear angular-contact sets)")
        raw_sets = [{"position": "front"}, {"position": "rear", "bore_mm": p["default_rear_bore_ratio"] * d0 * 1e3}]
    bearings = []
    for i, raw in enumerate(raw_sets):
        name = str(plain(raw.get("position")) or ("front" if i == 0 else f"set{i}"))
        bearings.append(_bearing(raw, name, d0, lubrication, "angular_contact", estimated))
    if not any(b.fixed for b in bearings):
        bearings[0].fixed = True

    def dim(key: str, default: float, label: str) -> float:
        value = number(thermal, ((key, MM, 0.0),))
        if value is None:
            estimated.append(label)
            return float(default)
        return float(value)

    shaft_d = dim("shaft_diameter_mm", d0, "shaft diameter")
    shaft_bore = dim("shaft_bore_mm", p["default_shaft_bore_ratio"] * shaft_d, "shaft bore")
    shaft_len = dim("shaft_length_mm", p["default_shaft_length_to_bore"] * d0, "shaft length")
    span = dim("bearing_span_mm", p["default_bearing_span_to_bore"] * d0, "bearing span")
    nose_d = dim("nose_diameter_mm", p["default_nose_diameter_ratio"] * shaft_d, "nose diameter")
    fixed = next(b for b in bearings if b.fixed)
    nose_default = p["default_nose_to_fixed_bearing_to_bore"] * d0 if fixed.name == "front" else \
        span + p["default_nose_to_fixed_bearing_to_bore"] * d0
    nose_len = dim("nose_to_fixed_bearing_mm", nose_default, "nose to the axially fixed bearing")
    nose_exposed = p["default_nose_exposed_to_bore"] * d0
    nose_area = math.pi * nose_d * nose_exposed + math.pi * nose_d * nose_d / 4.0
    housing_material = str(plain(thermal.get("housing_material")) or p["housing_material"])
    housing_mass = number(thermal, (("housing_mass_kg", 1.0, 0.0),))
    if housing_mass is None:
        housing_mass = p["default_housing_mass_kg"] * scale ** 3
        estimated.append("housing mass")
    size = plain(thermal.get("housing_size_mm"))
    if isinstance(size, (list, tuple)) and len(size) == 3:
        w, d, hgt = (float(v) * MM for v in size)
        housing_area = 2.0 * (w * d + w * hgt + d * hgt) - w * hgt
        housing_height = hgt
    else:
        housing_area = p["default_housing_area_m2"] * scale ** 2
        housing_height = p["default_housing_height_m"] * scale
        estimated.append("housing size (exterior area, height)")
    housing_len = dim("housing_growth_length_mm", p["default_housing_growth_length_to_bore"] * d0,
                      "housing growth length (head datum to the fixed bearing)")
    g_struct = p["default_housing_to_structure_W_K"]
    estimated.append("housing to structure conductance")
    # cooling jacket
    jacket_raw = thermal.get("jacket")
    cooled = _cooled(plain(spindle.get("cooling")))
    jacket_g = 0.0
    jacket_offset = 0.0
    jacket_inlet = None
    if isinstance(jacket_raw, Mapping) or cooled:
        jr = jacket_raw if isinstance(jacket_raw, Mapping) else {}
        flow = number(jr, (("flow_l_min", 1e-3 / 60.0, 0.0),))
        if flow is None:
            flow = p["default_jacket_flow_m3_s"]
            estimated.append("jacket flow")
        dh = number(jr, (("channel_diameter_mm", MM, 0.0),))
        if dh is None:
            dh = p["default_jacket_channel_diameter_m"]
            estimated.append("jacket channel diameter")
        channels = int(plain(jr.get("channels")) or 0) or int(p["default_jacket_channels"])
        area = number(jr, (("area_cm2", 1e-4, 0.0),))
        if area is None:
            area = p["default_jacket_area_m2"] * scale ** 2
            estimated.append("jacket wetted area")
        rho, cp = p["jacket_fluid_density_kg_m3"], p["jacket_fluid_cp_J_kgK"]
        velocity = flow / (channels * math.pi * dh * dh / 4.0)
        h_j = cv.h_pipe(velocity, dh, rho, p["jacket_fluid_viscosity_Pa_s"], p["jacket_fluid_k_W_mK"], cp)
        mcp = rho * flow * cp
        jacket_g = mcp * (-math.expm1(-h_j * area / mcp)) if mcp > 0.0 else 0.0
        inlet = number(jr, (("inlet_degC", 1.0, ZERO_DEGC_K), ("inlet_K", 1.0, 0.0)))
        if inlet is not None:
            jacket_inlet = float(inlet)
        else:
            jacket_offset = number(jr, (("inlet_offset_dK", 1.0, 0.0),), None)
            if jacket_offset is None:
                jacket_offset = p["default_jacket_inlet_offset_dK"]
                estimated.append("jacket inlet temperature (follows the ambient)")
    # motor
    drive = drive_type_of(spindle)
    if drive is None:
        drive = str(p["default_drive_type"])
        estimated.append(f"drive type ({drive})")
    if drive not in DRIVE_TYPES:
        raise ValueError(f"thermal: spindle {sid}: unknown drive_type {drive!r} (one of {', '.join(DRIVE_TYPES)})")
    rated, base = motor_rating(spindle)
    if rated is None:
        rated = p["default_motor_rated_W"]
        estimated.append("drive rated power")
    if base is None:
        base = p["default_motor_base_rad_s"]
        estimated.append("drive base speed")
    loss_frac = number(thermal, (("motor_no_load_loss_pct", 1e-2, 0.0),))
    if loss_frac is None:
        loss_frac = p["motor_no_load_loss_fraction"]
        estimated.append("drive no-load loss")
    if drive == "built_in":
        to_shaft = p["motor_rotor_loss_fraction"]
        to_housing = 1.0 - to_shaft
    else:
        to_shaft = 0.0
        share = number(thermal, (("motor_heat_to_head_pct", 1e-2, 0.0),))
        if share is None:
            share = float(np.asarray(p["motor_heat_to_housing_fraction"])[list(p["drive_types"]).index(drive)])
            estimated.append("share of the motor loss entering the head")
        to_housing = share
    # capacities
    shaft_mat = str(p["spindle_material"])
    bearing_mat = str(p["bearing_material"])
    cap_shaft = _rho(shaft_mat) * _cp(shaft_mat) * math.pi / 4.0 * (shaft_d ** 2 - shaft_bore ** 2) * shaft_len
    caps = [cap_shaft]
    names = ["shaft"]
    for b in bearings:
        caps.append(b.mass_kg * p["bearing_rolling_mass_fraction"] * _cp(bearing_mat))
        names.append(f"bearings.{b.name}")
    caps.append(housing_mass * _cp(housing_material))
    names.append("housing")
    start = ambient_K if start_K is None else float(start_K)
    net = SpindleNetwork(
        sid=sid, names=names, cap_J_K_nodes=[float(c) for c in caps], theta_K=[start - REF_K] * len(caps),
        bearings=bearings, shaft_d_m=shaft_d, span_m=span, nose_d_m=nose_d, nose_area_m2=nose_area,
        housing_area_m2=float(housing_area), housing_height_m=float(housing_height),
        housing_emissivity=float(p["housing_emissivity"]), shaft_emissivity=float(cv.emissivity("steel")),
        g_struct_W_K=float(g_struct), jacket_g_W_K=float(jacket_g), jacket_offset_dK=float(jacket_offset),
        jacket_inlet_K=jacket_inlet, bc_W_K=float(p["bearing_contact_conductance_W_K"]),
        bc_ref_dm_m=float(p["bearing_contact_ref_dm_m"]), bc_ref_rad_s=float(p["bearing_contact_ref_rad_s"]),
        bc_min_rad_s=float(p["bearing_contact_min_rad_s"]), rolling_frac=float(p["bearing_heat_rolling_fraction"]),
        inner_frac=float(p["bearing_heat_inner_ring_fraction"]), motor_W=float(loss_frac * rated),
        motor_base_rad_s=float(base), motor_to_shaft=float(to_shaft), motor_to_housing=float(to_housing),
        shaft_material=shaft_mat, housing_material=housing_material, nose_length_m=float(nose_len),
        housing_length_m=float(housing_len), T_ref_K=float(ambient_K), max_step_s=float(p["spindle_network_max_step_s"]),
        ledger=SolverLedger(f"spindle.{sid}"))
    net.reset_ledger()
    if kind not in ("milling", "turning", "live_tool", "sub"):
        raise ValueError(f"thermal: spindle {sid}: unknown kind {kind!r}")
    return net, estimated


__all__ = ["BearingNode", "SpindleNetwork", "build_network", "bearing_moment_Nm", "viscosity_m2_s",
           "walther_coefficients", "static_equivalent_N", "friction_load_N", "exact_step", "lubrication_of",
           "drive_type_of", "motor_rating", "DRIVE_TYPES"]
