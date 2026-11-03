"""Compiled kernels against their plain-Python references; one signature each (architecture 5.5)."""

import math

import numpy as np
import pytest

from cnctwin.tooling import build_tool, element_positions_T, kernels as K
from cnctwin.tooling.update import edge_state, effective_edge
from tests.unit.tooling.fixtures import HOLDER, TOOLS, instance


class Rec:
    def __init__(self, **values):
        self.__dict__.update(values)


@pytest.fixture(scope="module")
def tool():
    return build_tool(TOOLS["toroidal_endmill"], HOLDER,
                      instance(runout_um=10, runout_phase_deg=30, tilt_urad=50))


@pytest.mark.parametrize("theta", [0.0, 0.7, -2.4, 5.9])
def test_edge_points_match_reference(tool, theta):
    e = tool.edge
    out = np.zeros(e.elem_r_m.shape + (3,))
    ref = np.zeros_like(out)
    K.edge_points_T0(e, tool.trig, math.cos(theta), math.sin(theta), out)
    K.edge_points_T0_ref(e, tool.trig, math.cos(theta), math.sin(theta), ref)
    assert np.allclose(out, ref, atol=1e-15)
    assert np.allclose(out, element_positions_T(e, theta), atol=1e-15)


def test_edge_frames_match_reference(tool):
    shape = tool.frames.tangent.shape
    outs = [np.zeros(shape) for _ in range(4)]
    refs = [np.zeros(shape) for _ in range(4)]
    K.edge_frames_T0(tool.frames, math.cos(1.1), math.sin(1.1), *outs)
    K.edge_frames_T0_ref(tool.frames, math.cos(1.1), math.sin(1.1), *refs)
    for a, b in zip(outs, refs):
        assert np.allclose(a, b, atol=1e-15)
    # normal and tangent are perpendicular (to second order in the sub-chord on helical arcs), unit vectors
    valid = tool.edge.elem_valid == 1
    assert np.allclose(np.einsum("jki,jki->jk", outs[0], outs[1])[valid], 0.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(outs[2], axis=-1)[valid], 1.0)


def test_compose_matches_reference(tool):
    nt, ne = tool.edge.elem_r_m.shape
    wear = Rec(radial_wear_m=np.full((nt, ne), 2e-6), axial_wear_m=np.full((nt, ne), 1e-6),
               edge_radius_m=np.full((nt, ne), 12e-6), broken=False)
    heat = Rec(radial_growth_m=np.full((nt, ne), 3e-6), length_growth_m=8e-6, holder_growth_m=2e-6)
    hold = Rec(pullout_m=5e-6, slip_rad=0.01)
    state = edge_state(tool, wear, heat, hold)
    out = effective_edge(tool.edge, state)
    ref = K.compose_effective_ref(tool.edge, state)
    for name in out._fields:
        assert np.array_equal(np.asarray(getattr(out, name)), np.asarray(getattr(ref, name))), name


def test_envelope_radius_matches_reference(tool):
    z = np.linspace(-1e-3, 0.2, 400)
    for chain in (-1, 0, 1):
        out = np.zeros_like(z)
        K.envelope_radius_profile(tool.envelope, z, chain, out)
        assert np.array_equal(out, K.envelope_radius_ref(tool.envelope, z, chain))
        assert K.envelope_radius_at(tool.envelope, float(z[123]), chain) == out[123]


def test_one_signature_each():
    for kernel in (K.edge_points_T0, K.edge_frames_T0, K.compose_effective, K.envelope_radius_at,
                   K.envelope_radius_profile):
        assert len(kernel.signatures) == 1, kernel.__name__


def test_kernels_accept_every_tool_type():
    """Edges of all builders have the dtypes the explicit signatures were compiled for."""
    for kind, item in TOOLS.items():
        g = build_tool(item, HOLDER if kind not in ("turning_insert", "boring_bar", "grooving_insert",
                                                    "parting_insert", "threading_insert", "form_tool") else None,
                       instance())
        out = np.zeros(g.edge.elem_r_m.shape + (3,))
        K.edge_points_T0(g.edge, g.trig, 1.0, 0.0, out)
        assert np.all(np.isfinite(out)), kind
