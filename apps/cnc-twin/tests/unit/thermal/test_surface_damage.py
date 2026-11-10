"""Machined-surface temperature history and the thermal-damage criteria (docs/models/thermal.md 3.4).

Analytic checks of the moving sources (surfacetemp.py), the criteria (damage.py: Hollomon-Jaffe tempering parameter
with additivity, austenite start at the heating rate, minimum affected depth), and the chain mechanics -> partition ->
surface history -> criteria through the validation harness's ContactModel (the coefficient pipeline the run uses):
a severely worn CBN tool in hard turning of AISI 52100 (Chou & Evans 1999) and the hard-turning measurements of Chen
et al. (2017). The material data are the database's (db/materials).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cnctwin.core.findings import Severity
from cnctwin.thermal import damage as DM
from cnctwin.thermal import surfacetemp as ST
from cnctwin.thermal.response import gauss_legendre

GLX, GLW = gauss_legendre(24)
K_STEEL, RHO, CP = 42.0, 7850.0, 475.0
ALPHA = K_STEEL / (RHO * CP)
ZERO = 273.15


def _band_only(q_W_m2, length, speed, width=2e-3, where="arc"):
    power = q_W_m2 * length * width
    if where == "arc":
        return ST.EdgeSources(speed_m_s=speed, width_m=width, duration_s=1.0, arc_W=power, arc_length_m=length)
    return ST.EdgeSources(speed_m_s=speed, width_m=width, duration_s=1.0, land_W=power, land_length_m=length,
                          arc_length_m=0.0)


def test_trailing_edge_of_a_fast_band_is_the_one_dimensional_flash():
    """At a high Peclet number every surface point sees the flux q for l / V: the trailing-edge rise is the 1-D
    half-space solution 2 q sqrt(t / pi) / sqrt(k rho c) (Carslaw & Jaeger 1959 2.9; Jaeger's 1.128 q l / (k sqrt L))."""
    q, length = 1e9, 200e-6
    for speed in (5.0, 20.0):
        src = _band_only(q, length, speed)
        t = length / speed
        flash = 2.0 * q * math.sqrt(t / math.pi) / math.sqrt(K_STEEL * RHO * CP)
        rise = max(ST.rise(src, K_STEEL, ALPHA, -length * a, 0.0, GLX, GLW) for a in (0.9, 0.95, 0.98, 1.0))
        peclet = speed * 0.5 * length / (2.0 * ALPHA)
        assert peclet > 20.0
        assert rise == pytest.approx(flash, rel=0.03)
        assert ST.surface_peak(src, K_STEEL, ALPHA, GLX, GLW) == pytest.approx(rise, rel=0.02)   # sampled
        # the land sits behind the arc: the same band as a land of a contact without arc gives the same field
        land = _band_only(q, length, speed, where="land")
        assert ST.rise(land, K_STEEL, ALPHA, -0.95 * length, 0.0, GLX, GLW) == \
            pytest.approx(ST.rise(src, K_STEEL, ALPHA, -0.95 * length, 0.0, GLX, GLW), rel=1e-9)


def test_history_is_the_moving_frame_field_sampled_along_the_path():
    src = ST.EdgeSources(speed_m_s=2.0, width_m=1.5e-3, duration_s=0.5, shear_W=100.0, shear_length_m=4e-4,
                         arc_W=60.0, plough_W=100.0, arc_length_m=3e-5, plough_depth_m=2e-5, land_W=40.0,
                         land_length_m=5e-5)
    depths = ST.default_depths(src, ALPHA, (1e-6,))
    hist = ST.history(src, 300.0, K_STEEL, ALPHA, depths, GLX, GLW)
    assert np.all(np.diff(hist.time_s) > 0.0)
    assert hist.time_s[hist.edge_index] == pytest.approx(src.shear_length_m / src.speed_m_s, rel=1e-9)
    peaks = hist.peak_K()
    assert np.all(np.diff(peaks) <= 1e-9)                     # the peak falls with depth
    assert hist.temp_K[0, 0] - 300.0 < 0.05 * (peaks[0] - 300.0)  # ahead of the shear band: hardly heated
    assert hist.temp_K[0, -1] - 300.0 < 0.1 * (peaks[0] - 300.0)  # long behind: cooled into the bulk
    assert ST.surface_peak(src, K_STEEL, ALPHA, GLX, GLW) == pytest.approx(peaks[0] - 300.0, rel=0.02)


def test_plastic_work_in_the_ploughed_layer_heats_the_surface_less_than_at_the_surface():
    """The edge pressure's plastic work is generated to the depth delta (Carslaw & Jaeger plane sources with images):
    -> the surface source as delta -> 0, below it at the surface for a finite layer, alike deep below."""
    base = dict(speed_m_s=2.0, width_m=1.5e-3, duration_s=1.0, arc_length_m=3e-5)
    surface = ST.EdgeSources(**base, arc_W=100.0)
    thin = ST.EdgeSources(**base, plough_W=100.0, plough_depth_m=1e-8)
    layer = ST.EdgeSources(**base, plough_W=100.0, plough_depth_m=2e-5)
    at = -3e-5
    s0 = ST.rise(surface, K_STEEL, ALPHA, at, 0.0, GLX, GLW)
    assert ST.rise(thin, K_STEEL, ALPHA, at, 0.0, GLX, GLW) == pytest.approx(s0, rel=1e-3)
    assert ST.rise(layer, K_STEEL, ALPHA, at, 0.0, GLX, GLW) < 0.9 * s0
    deep = 2e-4
    assert ST.rise(layer, K_STEEL, ALPHA, at - 1e-3, deep, GLX, GLW) == \
        pytest.approx(ST.rise(surface, K_STEEL, ALPHA, at - 1e-3, deep, GLX, GLW), rel=0.05)


def test_tempering_parameter_is_hollomon_jaffe_with_additivity():
    C = 20.0
    t = np.array([0.0, 3600.0])
    assert DM.tempering_parameter(t, [873.15, 873.15], C) == pytest.approx(873.15 * C, rel=1e-12)
    # two hours in four pieces = two hours at once (additivity at one temperature)
    pieces = DM.tempering_parameter(np.linspace(0.0, 7200.0, 5), [800.0] * 5, C)
    assert pieces == pytest.approx(800.0 * (C + math.log10(2.0)), rel=1e-12)
    # a hold appended at the same temperature continues the same equivalent time
    assert DM.tempering_parameter(t, [800.0, 800.0], C, 800.0, 3600.0) == pytest.approx(pieces, rel=1e-12)
    # a hotter interval dominates: the parameter after a cool then a hot hour exceeds the hot hour alone
    both = DM.tempering_parameter(np.array([0.0, 3600.0, 3601.0, 7201.0]), [700.0, 700.0, 900.0, 900.0], C)
    assert 900.0 * C < both < 900.0 * (C + math.log10(2.0))


def test_heating_rate_and_austenite_start():
    t = np.linspace(0.0, 1e-4, 11)
    temps = 300.0 + 1e7 * t                     # 1e7 K/s ramp
    assert DM.heating_rate(t, temps) == pytest.approx(1e7, rel=1e-9)
    limit = DM.Threshold("white_layer", 1013.15, Severity.ERROR, "", fast_K=1053.15, fast_rate=3000.0)
    slow = float(DM.damage_params()["slow_heating_rate_kelvin_per_second"])
    assert DM.ac1_at_rate(limit, slow) == pytest.approx(1013.15)
    assert DM.ac1_at_rate(limit, 3000.0) == pytest.approx(1053.15)
    assert DM.ac1_at_rate(limit, 1e7) == pytest.approx(1053.15)      # no extrapolation beyond the data
    mid = math.sqrt(slow * 3000.0)
    assert DM.ac1_at_rate(limit, mid) == pytest.approx(1033.15, rel=1e-9)
    assert DM.ac1_at_rate(DM.Threshold("white_layer", 1013.15, Severity.ERROR, ""), 1e7) == 1013.15


def _history(peak_rise_K, depth_scale_m, contact_s=5e-5, base=300.0):
    """A synthetic history: a skin heated to base + rise exp(-z / depth_scale) for contact_s, then cooled."""
    depths = np.array([0.0, 0.5e-6, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5])
    times = np.array([0.0, 0.5 * contact_s, contact_s, 2.0 * contact_s, 10.0 * contact_s])
    shape = np.array([0.0, 0.6, 1.0, 0.5, 0.1])
    temps = base + peak_rise_K * np.exp(-depths[:, None] / depth_scale_m) * shape[None, :]
    return ST.SurfaceHistory(depths, times, temps, base, 2)


def test_white_layer_needs_the_minimum_depth():
    limits = DM.thresholds({"thermal_damage": {"white_layer_degC": 740}})
    zmin = float(DM.damage_params()["min_depth_m"])
    skin = DM.judge(limits[0], _history(900.0, 0.2 * zmin), 20.0)   # hot only in a sub-micron skin
    assert skin.details["surface_peak_K"] > limits[0].temp_K and not skin.damaged
    layer = DM.judge(limits[0], _history(900.0, 5.0 * zmin), 20.0)
    assert layer.damaged and layer.depth_m >= zmin


def test_short_flash_does_not_overtemper_a_steel_tempered_at_600C():
    """4140 tempered at 600 C (2 h): a 1000 C flash of 50 us has P = 1273 (20 + log10(1.4e-8)) ~ 15 600 K, below
    the 17 700 K of the tempering; the burn colours' 200 C / 1 h (9 460 K) are exceeded."""
    from cnctwin.db.database import Database
    material = Database.load().material("aisi4140-qt30")
    monitor = DM.DamageMonitor("4140", DM.thresholds(material), DM.hj_constant(material))
    hist = _history(1000.0 - 27.0, 2e-5, contact_s=5e-5)
    verdicts = {v.kind: v for v in monitor.verdicts(hist)}
    assert not verdicts["overtemper"].damaged
    assert verdicts["overtemper"].limit == pytest.approx(873.15 * (20.0 + math.log10(2.0)), rel=1e-9)
    assert verdicts["burn"].damaged
    assert verdicts["white_layer"].damaged          # 1000 C over microns: above Ac1 (740 C)


# ------------------------------------------------------------------ the chain through the coefficient pipeline
@pytest.fixture(scope="module")
def db():
    from cnctwin.db.database import Database
    return Database.load()


def _contact(db, material_id, tool_material, rake_deg, edge_radius_m, speed_m_s, h_m, width_m, VB_m, coat=None):
    from cnctwin.db.simap import wrap
    from cnctwin.mechanics.materials import ToolSide, WorkMaterial
    from cnctwin.validation.harnesses.contact import ContactModel
    work = WorkMaterial.from_db(db.material(material_id))
    item = {"type": "turning_insert", "coating": {"id": coat or "none"}, "geometry": {"insert": {}}}
    tool = ToolSide.from_db(wrap(item), db.tool_material(tool_material), db.coating(coat) if coat else None)
    model = ContactModel(work, tool, rake_rad=math.radians(rake_deg), clearance_rad=math.radians(6.0),
                         edge_radius_m=edge_radius_m, work_temp_K=293.15, speeds_m_s=[speed_m_s], h_values_m=[h_m])
    st = model.state(h_m, speed_m_s, 600.0, width_m, 1.0, VB_m=VB_m)
    src = model.edge_sources(st)
    return work, st, src


def _judge(db, material_id, work, src):
    k, alpha = work.conductivity(293.15), work.diffusivity(293.15)
    zmin = float(DM.damage_params()["min_depth_m"])
    hist = ST.history(src, 293.15, k, alpha, ST.default_depths(src, alpha, (zmin,)), GLX, GLW)
    material = db.material(material_id)
    monitor = DM.DamageMonitor(material_id, DM.thresholds(material), DM.hj_constant(material))
    return hist, {v.kind: v for v in monitor.verdicts(hist, 293.15, 0.1)}


def test_severely_worn_cbn_tool_in_hard_turning_leaves_a_white_layer(db):
    """AISI 52100 at 62 HRC, CBN, 200 m/min, f 0.1 mm, ap 0.2 mm: with a 0.3 mm wear land the land's friction
    heats the new surface far above Ac1 to tens of microns, a white layer as Chou & Evans (1999) report for worn
    tools in hard turning (their depth grew with flank wear); the layer is deeper than with the sharp tool."""
    kr = math.radians(95.0)
    worn = _contact(db, "aisi52100-62hrc", "cbn-low", -6.0, 20e-6, 200 / 60, 1e-4 * math.sin(kr), 2e-4 / math.sin(kr),
                    3e-4)
    hist, verdicts = _judge(db, "aisi52100-62hrc", worn[0], worn[2])
    white = verdicts["white_layer"]
    assert white.damaged and white.depth_m >= 5e-6
    assert verdicts["overtemper"].damaged         # tempered at 180 C: the dark layer below
    sharp = _contact(db, "aisi52100-62hrc", "cbn-low", -6.0, 20e-6, 200 / 60, 1e-4 * math.sin(kr),
                     2e-4 / math.sin(kr), 0.0)
    _, sharp_verdicts = _judge(db, "aisi52100-62hrc", sharp[0], sharp[2])
    assert white.depth_m > 2.0 * sharp_verdicts["white_layer"].depth_m


CHEN_FEEDS_MM = (0.05, 0.1, 0.15)


def test_edge_temperature_against_chen_2017_hard_turning(db):
    """Chen et al. (2017): AISI 52100 58-62 HRC, PCBN (-5 deg rake), 100 m/min, radial depth 0.1 mm on the 0.8 mm
    nose, feeds 0.05 / 0.1 / 0.15 mm/rev: peak machined surface temperatures 500-800 C at the intersection of the
    cutting edge and the machined surface (tool-foil and embedded thermocouples, 30 C apart), rising with feed. The
    chain predicts the surface temperature at the edge (xi = 0) within that range +- 30 C, rising with feed.
    Known (docs/models/thermal.md 3.4, open): the mean over the edge contact behind it is 250-400 K above the range
    (the edge heat into the work, about 5 GW/m2 over the 16 um arc against their inverse flank fluxes of 0.13-0.73
    GW/m2); asserted as found so that a change of it is seen."""
    at_edge, means = [], []
    for f_mm in CHEN_FEEDS_MM:
        engaged = 0.8e-3 * math.acos(1.0 - 0.1 / 0.8) + 0.5 * f_mm * 1e-3
        h = f_mm * 1e-3 * 0.1e-3 / engaged
        work, st, src = _contact(db, "aisi52100-62hrc", "cbn-low", -5.0, 25e-6, 100 / 60, h, engaged, 0.0)
        k, alpha = work.conductivity(293.15), work.diffusivity(293.15)
        at_edge.append(293.15 + ST.rise(src, k, alpha, 0.0, 0.0, GLX, GLW) - ZERO)
        means.append(293.15 + ST.contact_mean_rise(src, k, alpha, GLX, GLW) - ZERO)
    assert all(470.0 <= t <= 830.0 for t in at_edge), at_edge
    assert at_edge[0] < at_edge[1] < at_edge[2]
    assert all(m > 800.0 for m in means), means


@pytest.mark.xfail(strict=True, reason="open (docs/models/thermal.md 3.4): the edge heat the coefficient pipeline "
                   "puts into the work (about 180 W on a 29 um arc, 4 GW/m2) heats the new surface of the warmup's "
                   "4140 pass to about 900 C at 1 um depth, above Ac1")
def test_warmup_od_pass_on_4140_is_not_damaged(db):
    """The warmup's lathe pass (cnctwin.core.warmup.lathe_project): AISI 4140 QT ~30 HRC, CNMG120408-TM carbide
    (6 deg chip-breaker rake on the -6 deg holder, 40 um hone), 132 m/min, f 0.25 mm/rev, ap 1.5 mm, 95 deg approach,
    the run's initial 50 um land. Published machined-surface temperatures of such cuts are a few hundred degC and
    white layers are a hard-turning, worn-tool phenomenon: no white layer and no over-tempering expected."""
    kr = math.radians(95.0)
    work, st, src = _contact(db, "aisi4140-qt30", "carbide-p30", 0.0, 40e-6, 132 / 60, 0.25e-3 * math.sin(kr),
                             1.5e-3 / math.sin(kr), 50e-6, coat="al2o3-cvd")
    _, verdicts = _judge(db, "aisi4140-qt30", work, src)
    assert not verdicts["overtemper"].damaged
    assert not verdicts["white_layer"].damaged
