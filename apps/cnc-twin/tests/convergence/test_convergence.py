"""Convergence of the numerical settings (docs/convergence.md), one test per study.

Each test runs the study's shipped default and every finer level of its setting (tools/convergence/study.py, the
same cases, levels and analysis that wrote docs/convergence.md) and asserts that the shipped default lies within the
documented tolerance of the Richardson-extrapolated value (or the finest level, or the exact value where the case has
one) for every quantity docs/convergence.md records as inside its converged range. Quantities it records as outside
(each with the default it recommends) are asserted by a second test per study marked as a strict expected failure:
it starts to pass, and fails as XPASS, when the default comes inside, so the document is corrected with the code.

Slow (marked): the finest levels are the expensive ones; the whole-loop studies (macro step, lag, thermal grid) take
most of the time. Run with ``bin/test --all convergence`` or ``bin/tsafe python -m pytest -m slow tests/convergence``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "convergence"))

import study as S  # noqa: E402  (tools/convergence/study.py: the tree these tests run in)

pytestmark = pytest.mark.slow

# quantities docs/convergence.md records as outside their tolerance at the shipped default, with the recommendation
OUTSIDE: dict[str, dict[str, str]] = {}


def _evaluate(name: str) -> dict:
    return S.evaluate(name)


def _inside(name: str, doc: dict) -> None:
    analysis = doc["analysis"]
    expected = set(S.STUDIES[name].quantities)
    assert expected <= set(analysis), f"{name}: quantities not measured: {sorted(expected - set(analysis))}"
    problems = []
    for q, a in sorted(analysis.items()):
        if q in OUTSIDE.get(name, {}):
            continue
        if not a.get("default_inside"):
            problems.append(f"{q}: default {a.get('default_value')!r}, reference {a.get('reference')!r} "
                            f"({a.get('status')}), error {a.get('default_error')!r} > tolerance {a['tol']!r}")
    assert not problems, f"{name}: the shipped default left its documented tolerance:\n  " + "\n  ".join(problems)


def _outside(name: str, doc: dict) -> None:
    analysis = doc["analysis"]
    still = [q for q in OUTSIDE.get(name, {}) if not analysis.get(q, {}).get("default_inside", False)]
    assert not still, f"{name}: still outside the tolerance at the shipped default: " + \
        "; ".join(f"{q} ({OUTSIDE[name][q]})" for q in still)


_CACHE: dict[str, dict] = {}


def _doc(name: str) -> dict:
    if name not in _CACHE:
        _CACHE[name] = _evaluate(name)
    return _CACHE[name]


def _pair(name: str):
    def test_inside():
        _inside(name, _doc(name))

    reason = "; ".join(f"{q}: {why}" for q, why in OUTSIDE.get(name, {}).items()) or "none documented"

    @pytest.mark.xfail(strict=True, reason=f"docs/convergence.md: shipped default outside the tolerance ({reason})")
    def test_outside():
        _outside(name, _doc(name))
    return test_inside, test_outside


def test_stride_slot():
    _inside("stride_slot", _doc("stride_slot"))


def test_stride_side():
    _inside("stride_side", _doc("stride_side"))


def test_lattice_side():
    _inside("lattice_side", _doc("lattice_side"))


def test_lattice_corner():
    _inside("lattice_corner", _doc("lattice_corner"))


def test_element_flat():
    _inside("element_flat", _doc("element_flat"))


def test_element_ball():
    _inside("element_ball", _doc("element_ball"))


def test_bins_runout():
    _inside("bins_runout", _doc("bins_runout"))


def test_bins_chatter():
    _inside("bins_chatter", _doc("bins_chatter"))


def test_turn_theta():
    _inside("turn_theta", _doc("turn_theta"))


def test_turn_dz():
    _inside("turn_dz", _doc("turn_dz"))


def test_macro():
    _inside("macro", _doc("macro"))


def test_lag():
    _inside("lag", _doc("lag"))


def test_thermal_cell():
    _inside("thermal_cell", _doc("thermal_cell"))


def test_thermal_every():
    _inside("thermal_every", _doc("thermal_every"))
