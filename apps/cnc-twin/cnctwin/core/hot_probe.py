"""G31 probing in the compiled loop (docs/models/hotloop.md "Probe"; architecture 5.3 steps 1 and 3).

A path whose spindle holds a probe (a ``kind: probe`` entry of the project's tool list naming one of the machine's
``probes``; the resolved project lists them in ``tools.probes``) gets a probe-only contact in the hot loop's contact tuple: ``CI_ACTIVE`` 0 (it cuts nothing, every
cutting, removal and dynamics loop of run_macro passes it by), ``CI_PROBE`` 1 and the stylus as removal's
:class:`~cnctwin.removal.abi.Profile`. run_macro sweeps it with removal's ``sweep_contact`` against the
workpiece's field along every L1 tick of a G31 move of that path (``MotionSetpoint.probe``); the first contact
with the material sets ``MotionOut.probe_hit`` and returns at that sync tick, where motion latches the encoder
position (plus the stylus pre-travel along the move) for the controller's skip feedback.

The stylus: the ruby ball of the probe's definition (``stylus_ball_mm`` of the entry's resolved probe, else of the
machine's ``<probe>`` or ``<probe>_extra``) on its stem up to ``stylus_length_mm`` above the ball's centre. The probe's tool control
point is the ball's lowest point: its length offset (the tool entry's ``length_offset_mm``, the calibrated
electrical trigger point) runs from the gauge line to the bottom of the ball. Motion's tip of a path is its tool
slot's gauge-line-to-tip length below the gauge line, and a probe has no tool slot (it is not in the tool table:
the tip is the gauge line), so the profile is placed that far below motion's tip: z from motion's tip, the ball's
bottom at ``motion length - probe length`` (negative). The stem diameter is not in the machine files:
``probe_stem_diameter_ratio`` x the ball diameter (db/models/loop.yaml).
"""

from __future__ import annotations

import math

import numpy as np

from ..removal import abi as RA

DEFAULT_PROBE = "spindle_probe"


def _get(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        value = obj.get(name, default)
    else:
        try:
            value = obj[name]
        except (KeyError, TypeError, IndexError):
            value = getattr(obj, name, default)
    return default if value is None else value


def _num(value) -> float | None:
    if value is None:
        return None
    value = getattr(value, "value", value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _plain(value):
    return getattr(value, "value", value)


def probe_definition(project, number):
    """(probe name, length offset in m, definition or None) of probe T``number``, else None. A resolved
    project lists its probes in ``tools.probes`` (the loader moves the ``kind: probe`` entries out of
    ``tools.list``; each entry's ``probe`` is the definition resolved against the machine's ``probes``, its
    ``name`` the machine probe it extends, None for an inline one); a raw project keeps them in ``tools.list``
    with ``probe`` naming the machine's probe."""
    if number is None:
        return None
    tools = _get(project, "tools")
    entries = list(_get(tools, "probes", []) or [])
    entries += [e for e in (_get(tools, "list", []) or []) if str(_plain(_get(e, "kind", "")) or "") == "probe"]
    for entry in entries:
        n = _num(_get(entry, "number"))
        if n is None or int(n) != int(number):
            continue
        definition = _get(entry, "probe")
        if definition is None or isinstance(_plain(definition), str):
            name, spec = str(_plain(definition) or DEFAULT_PROBE), None
        else:
            name = str(_plain(_get(definition, "name")) or _plain(_get(definition, "extends")) or "inline")
            spec = definition
        length = _num(_get(entry, "length_offset_m"))
        return name, float(length or 0.0), spec
    return None


def probe_entry(project, number):
    """(machine probe name, length offset in m) of probe T``number`` (:func:`probe_definition`), else None."""
    found = probe_definition(project, number)
    return None if found is None else (found[0], found[1])


def stylus_dimensions(machine, name: str, definition=None) -> tuple[float, float]:
    """(ball diameter, stylus length) in m of probe ``name``: the probe entry's resolved definition, else the
    machine's ``probes`` entry ``name`` and ``<name>_extra``, else db/models/loop.yaml's defaults."""
    from ..db.models import params
    probes = _get(machine, "probes")
    entries = [definition, _get(probes, name), _get(probes, f"{name}_extra")]
    ball = length = None
    for entry in entries:
        ball = ball if ball is not None else _num(_get(entry, "stylus_ball_m"))
        length = length if length is not None else _num(_get(entry, "stylus_length_m"))
    p = params("loop")
    if ball is None or ball <= 0.0:
        ball = float(p["probe_ball_diameter_m"])
    if length is None or length <= 0.0:
        length = float(p["probe_stylus_length_m"])
    return float(ball), float(length)


def stylus_profile(ball_m: float, length_m: float, stem_ratio: float, z0_m: float = 0.0) -> RA.Profile:
    """The stylus as a max-radius profile R(z) about the probe axis, the ball's lowest point at z0_m: the ball's
    lower half, its upper half down to the stem radius, the stem to ``length_m`` above the ball's centre."""
    from ..removal.profile import Piece, build_profile
    from ..tooling.abi import PART_EDGE, PART_SHANK
    R = 0.5 * float(ball_m)
    rs = min(max(float(stem_ratio), 0.0), 1.0) * R
    z_stem = R + math.sqrt(max(R * R - rs * rs, 0.0))
    top = R + max(float(length_m), z_stem - R)
    b = float(z0_m)
    pieces = [Piece(RA.PK_ARC, b, b + R, b + R, 0.0, R, 1.0, PART_EDGE, 0)]
    if z_stem > R:
        pieces.append(Piece(RA.PK_ARC, b + R, b + z_stem, b + R, 0.0, R, 1.0, PART_EDGE, 1))
    if top > z_stem and rs > 0.0:
        pieces.append(Piece(RA.PK_LINE, b + z_stem, b + top, rs, 0.0, 0.0, 0.0, PART_SHANK, 2))
    return build_profile(pieces)


class ProbeContacts:
    """The probe-only contacts of the hot loop, one per path whose spindle holds a probe (profiles cached by
    stylus size and placement)."""

    def __init__(self, project):
        self.project = project
        self.profiles: dict[tuple, RA.Profile] = {}

    def profile(self, ball_m: float, length_m: float, z0_m: float) -> RA.Profile:
        key = (float(ball_m), float(length_m), float(z0_m))
        prof = self.profiles.get(key)
        if prof is None:
            from ..db.models import params
            prof = stylus_profile(ball_m, length_m, float(params("loop")["probe_stem_diameter_ratio"]), z0_m)
            self.profiles[key] = prof
        return prof

    def find(self, number, tip_length_m: float):
        """(probe name, z of the ball's bottom from motion's tip, ball diameter, stylus length) when tool
        ``number`` is a probe, else None; ``tip_length_m``: motion's gauge-line-to-tip length of the path's
        tool slot (0 without a slot)."""
        found = probe_definition(self.project, number)
        if found is None:
            return None
        name, length, definition = found
        ball, stylus = stylus_dimensions(_get(self.project, "machine"), name, definition)
        return name, float(tip_length_m) - length, ball, stylus

    def contact(self, pad, H, path: int, workpiece: int, found):
        """A copy of the padding contact ``pad`` (an inactive contact of the right types) that sweeps the stylus
        ``found`` (:meth:`find`) for ``path`` against ``workpiece``: its own index arrays and scratch (the
        padding contact is shared)."""
        _, z0_m, ball_m, length_m = found
        ip = np.zeros(H.N_CI, np.int64)
        ip[H.CI_ACTIVE] = 0
        ip[H.CI_PROBE] = 1
        ip[H.CI_PATH] = int(path)
        ip[H.CI_WORK] = int(workpiece)
        ip[H.CI_SLOT] = -1
        return pad._replace(ipar=ip, work=np.zeros(H.WORK_SIZE), probe=self.profile(ball_m, length_m, z0_m),
                            probe_out=np.zeros(8))
