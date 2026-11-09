"""Convergence of the numerical settings (docs/convergence.md), one test per study.

Each test runs the study's shipped default and the finer levels its reference rests on (tools/convergence/study.py:
the cases, levels and analysis that wrote docs/convergence.md) and asserts that the shipped default lies within the
documented tolerance of the Richardson-extrapolated value (or the finest level, or the exact value where the case has
one) for every quantity docs/convergence.md records as inside the converged range. The quantities it records as
outside, each with the default it recommends, are asserted by a second test of that study marked as a strict expected
failure: when the shipped default comes inside it passes, fails as XPASS, and the document is corrected with the code.

Slow (marked): about 35 CPU minutes in all on the i5-4690 (2016 s on 2026-11-09), half of it the whole-loop studies
(macro step, lag, thermal grid). Run with ``bin/test --all convergence`` (or ``TSAFE_MEM=1800M bin/tsafe python -m
pytest -m slow tests/convergence``: the whole-loop runs need more than bin/test's default 1.5 GB cap).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "convergence"))

import study as S  # noqa: E402  (tools/convergence/study.py on the tree these tests run in)

pytestmark = pytest.mark.slow

# docs/convergence.md: quantities whose shipped default is outside the tolerance, with the recommendation
_PEAK = "the step-mean peak; max_deg_per_step 0.125 (docs/convergence.md 1, 8)"
_TOP = "the element at the depth-of-cut line is engaged whole or not at all (docs/convergence.md 3, 9)"
OUTSIDE: dict[str, dict[str, str]] = {
    "stride_slot": {"F_peak_N": _PEAK},
    "stride_side": {"F_peak_N": _PEAK},
    "lattice_side": {"F_mean_N": "at the tolerance: -1.50 % on the copy of 14:28, -1.59 % on the working tree of "
                                 "16:00; 0.125 mm",
                     "F_peak_N": "34 % high at 0.25 mm, inside only at 0.03125 mm",
                     "chip_volume_analytic_mm3": "1.7 % low at 0.25 mm; 0.125 mm",
                     "engaged_deg": "oscillates by +-1.3 % between neighbouring lattices",
                     "removed_mm3": "the z family samples a 0.3 mm layer at lattice nodes: -18 % at 0.25 mm, +2.3 % at "
                                    "0.03125 mm (docs/convergence.md 2, 9)"},
    "lattice_corner": {"F_mean_N": "1.8 % low at 0.25 mm; 0.125 mm",
                       "engaged_max_deg": "oscillates by +-2 % between neighbouring lattices",
                       "removed_mm3": "the z family samples a 0.3 mm layer at lattice nodes: -16 % at 0.25 mm"},
    "element_flat": {q: _TOP for q in ("F_mean_N", "Fx_N", "Fy_N", "Fz_N", "F_peak_N", "torque_Nm",
                                       "chip_volume_analytic_mm3")},
    "bins_stride": {"F_peak_N": _PEAK, "vib_pp_um": "+7 % at 720 bins and 0.5 deg; 1440 bins with 0.25 deg",
                    "chip_tooth_spread": "0.206 against 0.239; 1440 bins with 0.25 deg"},
    "macro": {q: "the rake temperature cutting reads depends on the macro step and the tool's wear and temperature "
                 "run away at a time the macro step sets (docs/convergence.md 5, 9)"
              for q in ("tool_rise_K", "edge_rise_K", "rake_rise_K", "VB_growth_um", "F_mean_N", "bulk_rise_K",
                        "expansion_um")},
    "thermal_cell": {"expansion_um": "inside on the copy of 14:28 (-0.36 to -0.48 um); the working tree of 16:00 gives "
                                     "-0.16 um at 0.5 mm (docs/convergence.md 6)",
                     "bulk_max_rise_K": "the hottest cell of a field under a moving source: grows as the cell shrinks",
                     "bulk_rise_K": "5 % from 0.5 mm, whose tool has run away (docs/convergence.md 6)",
                     "surface_rise_K": "the 0.5 mm level's tool has run away (docs/convergence.md 6)",
                     "tool_rise_K": "at 1 and 0.5 mm cells the tool's wear and temperature run away by 6 s"},
    "thermal_every": {"bulk_max_rise_K": "the hottest cell: 15 % between every 5 and every macro step",
                      "bulk_rise_K": "2.3 % from the Richardson value",
                      "surface_rise_K": "7 % from the every-step level"},
}

_DOCS: dict[str, dict] = {}


def _doc(name: str) -> dict:
    if name not in _DOCS:
        _DOCS[name] = S.evaluate(name)
    return _DOCS[name]


def _check_inside(name: str) -> None:
    doc = _doc(name)
    analysis = doc["analysis"]
    missing = sorted(set(S.STUDIES[name].quantities) - set(analysis))
    assert not missing, f"{name}: quantities not measured: {missing}"
    problems = [f"{q}: default {a.get('default_value')!r} against {a.get('reference')!r} ({a.get('status')}): error "
                f"{a.get('default_error')!r} > tolerance {a['tol']!r}"
                for q, a in sorted(analysis.items())
                if q not in OUTSIDE.get(name, {}) and not a.get("default_inside")]
    assert not problems, f"{name}: the shipped default left its documented tolerance:\n  " + "\n  ".join(problems)


def _check_outside(name: str) -> None:
    analysis = _doc(name)["analysis"]
    still = [f"{q} ({why})" for q, why in OUTSIDE[name].items() if not analysis.get(q, {}).get("default_inside")]
    assert not still, f"{name}: outside the tolerance at the shipped default: " + "; ".join(still)


def _xfail(name: str):
    reasons = "; ".join(f"{q}: {why}" for q, why in OUTSIDE.get(name, {}).items())
    return pytest.mark.xfail(strict=True, reason=f"docs/convergence.md: {reasons}")


def test_stride_slot():
    _check_inside("stride_slot")


@_xfail("stride_slot")
def test_stride_slot_outside():
    _check_outside("stride_slot")


def test_stride_side():
    _check_inside("stride_side")


@_xfail("stride_side")
def test_stride_side_outside():
    _check_outside("stride_side")


def test_lattice_side():
    _check_inside("lattice_side")


@_xfail("lattice_side")
def test_lattice_side_outside():
    _check_outside("lattice_side")


def test_lattice_corner():
    _check_inside("lattice_corner")


@_xfail("lattice_corner")
def test_lattice_corner_outside():
    _check_outside("lattice_corner")


def test_element_flat():
    _check_inside("element_flat")


@_xfail("element_flat")
def test_element_flat_outside():
    _check_outside("element_flat")


def test_element_ball():
    _check_inside("element_ball")


def test_bins_stride():
    _check_inside("bins_stride")


@_xfail("bins_stride")
def test_bins_stride_outside():
    _check_outside("bins_stride")


def test_bins_chatter():
    _check_inside("bins_chatter")


def test_turn_theta():
    _check_inside("turn_theta")


def test_turn_dz():
    _check_inside("turn_dz")


def test_lag():
    # lag_macro 2 is computed as lag 1: the two levels agree exactly
    _check_inside("lag")


def test_macro():
    _check_inside("macro")


@_xfail("macro")
def test_macro_outside():
    _check_outside("macro")


def test_thermal_cell():
    _check_inside("thermal_cell")


@_xfail("thermal_cell")
def test_thermal_cell_outside():
    _check_outside("thermal_cell")


def test_thermal_every():
    _check_inside("thermal_every")


@_xfail("thermal_every")
def test_thermal_every_outside():
    _check_outside("thermal_every")
