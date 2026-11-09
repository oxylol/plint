"""Validation against measured orthogonal cutting data.

Source: Klippel, Suessmaier, Kuffa & Wegener (2022), "Dry cutting experiments database Ti6Al4V and
Ck45", arXiv:2209.04197 (ETH Zurich), Table B.15 (Ck45) and Table B.14 (Ti6Al4V): quasi-orthogonal
tube turning, uncoated Sandvik CCMW 09T304 H13A (rake 0 deg, clearance 7 deg, measured edge radii
33-44 um, mean 39 um), dry, forces per mm of width. Values are means of the repetitions marked
'ok' (cutting force F_c, feed force F_f in N/mm, average chip thickness h_avg in um).

Model: Oxley's own interface condition (the chip's shear flow stress at the interface; friction an
output), the modified-Merchant fallback continued from the edge of the admissible Oxley solutions
where the condition has none (here: the small-h and low-speed points, regime REGIME_MERCHANT).
Tolerances and why (docs/models/mechanics.md, "Validation"), measured on 2026-11-03:
* uncalibrated Ck45, Oxley-regime points (12): F_c -20 % to -7 %, F_f -32 % to -13 %, h_avg
  -20 % to +43 %; bands -25/+10 %, -40/+20 %, +-50 %: the Jaspers & Dautzenberg JC set (other 1045
  sets change forces by 10-20 %), the estimated h_min ratio and edge pressure;
* uncalibrated Ck45, fallback points (3): within -5/+55 % (F_c), -5/+165 % (F_f), +5/+215 %
  (h_avg); the 30 m/min point lies in the BUE window of Ck45 (10-50 m/min), where the chip flows
  over a built-up edge that this BUE-free solution does not have (the theory has no admissible
  solution there; its edge state gives a chip about three times too thick);
* calibrated Ck45 (Kienzle at 100 m/min, the same database): Oxley-regime points within 9 % (band 15 %), the
  fallback points within 80 % (the 30 m/min BUE point; the others within 25 %);
* Ti6Al4V (Seo et al. JC set): uncalibrated F_c +32 to +95 %, F_f +34 to +201 %: a continuous-chip
  theory with a seized interface over-predicts the segmented Ti6Al4V chip (the former Coulomb
  limit with mu = 0.3 came from the same ETH tests and compensated); bands F_c 0/+50 % and F_f
  0/+85 % at the Oxley points, F_c 0/+100 % and F_f 0/+210 % at the two fallback points; calibrated
  with the five measured Kienzle speeds within 12 % (F_c) and 25 % (F_f);
* chip thickness: Ti6Al4V +46 % to +215 % (continuous-chip thickness against segmented chips).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cnctwin.mechanics import coefficients as CF

from conftest import STEEL_1045, work_from

CK45 = {(30, 0.15): (374.7, 214.8, 240.0), (50, 0.20): (643.6, 605.6, 503.9), (100, 0.06): (197.0, 175.3, 161.5),
        (150, 0.06): (225.0, 241.4, 211.7), (150, 0.10): (344.7, 335.8, 339.1), (150, 0.15): (428.1, 351.7, 303.8),
        (150, 0.20): (532.5, 390.9, 453.9), (150, 0.40): (910.0, 507.3, 720.3), (200, 0.04): (158.8, 171.3, 166.2),
        (250, 0.10): (304.0, 256.9, 312.8), (250, 0.12): (333.7, 260.3, 336.2), (350, 0.10): (283.8, 214.9, 263.8),
        (350, 0.15): (372.5, 233.3, 346.4), (450, 0.10): (263.4, 180.9, 237.7), (500, 0.10): (259.8, 172.4, 234.5)}

TI64 = {(20, 0.15): (292.8, 176.4, 211.6), (50, 0.10): (210.9, 140.6, 152.1), (70, 0.20): (334.4, 181.7, 230.3),
        (100, 0.10): (200.0, 122.0, 180.3), (100, 0.20): (329.4, 177.2, 241.5), (125, 0.15): (259.2, 154.2, 171.6),
        (150, 0.10): (196.3, 137.2, 141.2)}

# db/materials/ti6al4v.yaml (2026-11-03): Seo et al. JC set via Ducobu et al. 2017; ETH Kienzle entries
TI_KIENZLE = [(20, 1265, 0.22), (60, 1146, 0.27), (100, 932, 0.33), (212, 859, 0.32), (318, 712, 0.36)]
TI_FEED = {20: (421, 0.54), 60: (300, 0.69), 100: (226, 0.77), 212: (248, 0.72), 318: (186, 0.78)}
TI = {
    "id": "fixture-ti64", "family": "titanium", "iso_group": "S",
    "physical": {"density_kg_m3": 4430, "specific_heat_J_kgK": {"table": [[20, 580], [425, 670], [870, 930]]},
                 "conductivity_W_mK": {"table": [[20, 6.6], [315, 10.6]]}, "youngs_modulus_GPa": 114.5,
                 "poisson": 0.31, "yield_MPa": 869, "ultimate_MPa": 952, "fracture_strain": 0.49},
    "flow_stress": {"model": "johnson_cook", "A_MPa": 997.9, "B_MPa": 653.1, "n": 0.45, "C": 0.0198, "m": 0.7,
                    "ref_strain_rate_1_s": 1.0, "T_ref_degC": 20, "T_melt_degC": 1660},
    "cutting": {"kienzle": [{"tool_class": "carbide", "coating": "uncoated", "kc11_N_mm2": kc, "mc": mc,
                             "kf11_N_mm2": TI_FEED[v][0], "mf": TI_FEED[v][1], "h_ref_mm": 1, "b_ref_mm": 1,
                             "rake_ref_deg": 0, "vc_ref_m_min": v, "src": "klippel-2024-ck45-orthogonal",
                             "calibration": {"h_mm": [0.01, 0.4], "edge_radius_um": 37.5, "coolant": "dry"}}
                            for v, kc, mc in TI_KIENZLE],
                "min_chip_thickness_ratio": 0.37},
    "friction": [{"tool_class": "carbide", "coating": "uncoated", "mu_sliding": 0.3, "sticking_ratio": 0.6}],
    "chip": {"tendency": "segmented", "segmentation": {"criterion": "critical_speed", "vc_onset_m_min": 11},
             "breakability": "medium"},
}

GROUP = CF.EdgeGroup(rake_rad=0.0, clearance_rad=math.radians(7), edge_radius_m=39e-6)


def _predict(work, tool, data):
    speeds = np.array(sorted({v for v, _ in data})) / 60.0
    pmap = CF.build_process_map(work, tool, 0.0, speed_grid_m_s=speeds)
    cal = CF.build_calibration(work, tool, GROUP, "turning")
    out = {}
    for (v, h), meas in sorted(data.items()):
        state = CF.SlowState(speed_m_s=v / 60.0, tool_temp_K=293.15)
        hh = np.array([h * 1e-3])
        raw = CF.group_coefficients(work, tool, GROUP, state, hh, pmap)
        calibrated = CF.group_coefficients(work, tool, GROUP, state, hh, pmap, calibration=cal)
        f_raw = raw.forces_per_width(hh)[:, 0] / 1e3
        f_cal = calibrated.forces_per_width(hh)[:, 0] / 1e3
        t2 = h * 1e-3 / raw.chip_ratio[0] * 1e6
        out[(v, h)] = (f_raw, f_cal, t2, meas)
    return out, cal


@pytest.fixture(scope="module")
def ck45(carbide):
    return _predict(work_from(STEEL_1045), carbide, CK45)


@pytest.fixture(scope="module")
def ti64(carbide):
    return _predict(work_from(TI), carbide, TI64)


def _regimes(work, tool, data):
    speeds = np.array(sorted({v for v, _ in data})) / 60.0
    pmap = CF.build_process_map(work, tool, 0.0, speed_grid_m_s=speeds)
    out = {}
    for (v, h) in sorted(data):
        gc = CF.group_coefficients(work, tool, GROUP, CF.SlowState(speed_m_s=v / 60.0, tool_temp_K=293.15),
                                   np.array([h * 1e-3]), pmap)
        out[(v, h)] = int(gc.regime[0])
    return out


@pytest.fixture(scope="module")
def ck45_regimes(carbide):
    return _regimes(work_from(STEEL_1045), carbide, CK45)


@pytest.fixture(scope="module")
def ti64_regimes(carbide):
    return _regimes(work_from(TI), carbide, TI64)


def test_ck45_uncalibrated(ck45, ck45_regimes):
    results, _ = ck45
    errors = []
    for key, (f_raw, _, t2, (fc, ff, t2_meas)) in results.items():
        e_c = f_raw[0] / fc - 1.0
        e_f = f_raw[1] / ff - 1.0
        e_t = t2 / t2_meas - 1.0
        if ck45_regimes[key] == CF.REGIME_OXLEY:
            assert -0.25 <= e_c <= 0.10, (key, e_c)
            assert -0.40 <= e_f <= 0.20, (key, e_f)
            assert abs(e_t) <= 0.5, (key, t2, t2_meas)
            errors.append(abs(e_c))
        else:
            assert ck45_regimes[key] == CF.REGIME_MERCHANT, key
            assert -0.05 <= e_c <= 0.55 and -0.05 <= e_f <= 1.65 and 0.05 <= e_t <= 2.15, (key, e_c, e_f, e_t)
    assert len(errors) >= 12 and np.mean(errors) < 0.20
    # the BUE-window point (30 m/min) has no admissible Oxley solution
    assert ck45_regimes[(30, 0.15)] == CF.REGIME_MERCHANT


def test_ck45_calibrated(ck45, ck45_regimes):
    results, cal = ck45
    assert cal.kind == "kienzle"
    for key, (_, f_cal, _, (fc, ff, _)) in results.items():
        bound = 0.15 if ck45_regimes[key] == CF.REGIME_OXLEY else 0.80
        assert abs(f_cal[0] / fc - 1.0) <= bound, (key, f_cal[0], fc)


def test_ti64_uncalibrated_and_calibrated(ti64, ti64_regimes):
    results, cal = ti64
    assert cal.kind == "kienzle" and cal.speeds_m_s.size == 5
    for key, (f_raw, f_cal, t2, (fc, ff, t2_meas)) in results.items():
        oxley_point = ti64_regimes[key] == CF.REGIME_OXLEY
        assert 0.0 <= f_raw[0] / fc - 1.0 <= (0.50 if oxley_point else 1.00), (key, f_raw[0], fc)
        assert 0.0 <= f_raw[1] / ff - 1.0 <= (0.85 if oxley_point else 2.10), (key, f_raw[1], ff)
        assert abs(f_cal[0] / fc - 1.0) <= 0.12, (key, f_cal[0], fc)
        assert abs(f_cal[1] / ff - 1.0) <= 0.25, (key, f_cal[1], ff)
        assert 0.4 <= t2 / t2_meas - 1.0 <= 2.2, (key, t2, t2_meas)
