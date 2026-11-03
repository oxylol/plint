"""Chordal deviation of the linearised toolpath against the CAD surface (docs/models/inspect.md "Chordal
deviation").

CAM systems approximate curved surfaces by G01 chords whose end points lie on the tool's offset surface; between
them the path cuts the corner of a convex surface (too deep) or leaves material on a concave one. With d(p) the
signed distance of a path point to the CAD (+ outside the material) and the segment's end points at d0, d1, the
**chord error** is

    e(s) = d(P(s)) - ((1 - s) d0 + s d1),   s in [0, 1]

its most negative value (toward / into the part) and most positive value (away from it) are found by sampling and
golden-section refinement (cnctwin.inspect.kernels.chord_deviation_nb). For a chord of length L on a curve of
radius R it is the sagitta ``R (1 - cos(L / 2R)) ~ L^2 / 8R``. The reference point is the programmed tool tip,
optionally moved along the tool axis (``reference_offset_m``: the ball centre of a ball nose is the tip + R), so
the result does not depend on the tool model; the offset itself (d0, d1) is reported. Arcs (G02/G03) are evaluated
along the exact helical arc the control interpolates, so their "chord error" measures how the programmed arc
departs from the CAD.

Tip positions come from :class:`cnctwin.inspect.protocols.TipResolver` (the default
:class:`~cnctwin.inspect.protocols.CartesianTips` follows sanity's convention: TIP-space commands carry the tip, a
3-axis mill's machine axes minus the work offset and the active tool length) and are mapped to S by the stock pose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..controller.ir import Kind
from . import abi as A
from .cadref import CadReference
from .kernels import chord_deviation_nb
from .schema import InspectSettings


@dataclass
class PathSegment:
    kind: int                       # abi.SEG_LINE | SEG_ARC
    p0: np.ndarray                  # S
    p1: np.ndarray
    centre: np.ndarray | None = None
    normal: np.ndarray | None = None
    sweep_rad: float = 0.0
    program: str = ""
    line: int = 0
    block: str = ""
    tool: int | None = None

    def row(self) -> np.ndarray:
        r = np.zeros(A.N_SG)
        r[A.SG_KIND] = self.kind
        r[A.SG_P0:A.SG_P0 + 3] = self.p0
        r[A.SG_P1:A.SG_P1 + 3] = self.p1
        if self.kind == A.SEG_ARC:
            r[A.SG_C:A.SG_C + 3] = self.centre
            r[A.SG_N:A.SG_N + 3] = self.normal
            r[A.SG_SWEEP] = self.sweep_rad
        return r


def segments_from_commands(commands, resolver, pose_S_W=None, reference_offset_m: float = 0.0) -> list[PathSegment]:
    """Feed segments (G01 lines and G02/G03 arcs) of a command stream as tool reference points in S."""
    T = np.eye(4) if pose_S_W is None else np.asarray(pose_S_W, dtype=np.float64)
    R, t = T[:3, :3], T[:3, 3]
    out = []
    previous = None
    for cmd in commands:
        if cmd.kind not in (Kind.RAPID, Kind.LINEAR, Kind.ARC):
            continue
        end = resolver.tip_W(cmd, previous)
        if end is None or not np.all(np.isfinite(end)):
            previous = None
            continue
        end = np.asarray(end, dtype=np.float64)
        if previous is not None and cmd.kind in (Kind.LINEAR, Kind.ARC):
            axis = resolver.axis_W(cmd)
            off = reference_offset_m * (R @ np.asarray(axis, dtype=np.float64))
            p0 = R @ previous + t + off
            p1 = R @ end + t + off
            tool = (cmd.modal or {}).get("tool")
            if cmd.kind == Kind.ARC and cmd.arc is not None:
                c = R @ np.asarray(resolver.point_W(cmd, cmd.arc.centre), dtype=np.float64) + t + off
                n = R @ np.asarray(cmd.arc.normal, dtype=np.float64)
                out.append(PathSegment(A.SEG_ARC, p0, p1, c, n / np.linalg.norm(n), float(cmd.arc.sweep_rad),
                                       cmd.ref.program, cmd.ref.line, cmd.ref.block, tool))
            elif float(np.linalg.norm(p1 - p0)) > 0.0:
                out.append(PathSegment(A.SEG_LINE, p0, p1, program=cmd.ref.program, line=cmd.ref.line,
                                       block=cmd.ref.block, tool=tool))
        previous = end
    return out


def chordal_deviation(segments: list[PathSegment], cad: CadReference, settings: InspectSettings,
                      limit_m: float | None = None, max_listed: int = 50) -> dict:
    """Chord errors of path segments against the CAD (module documentation): totals, the worst segments, every
    segment beyond the limit (default ``chordal_tolerance_m``)."""
    limit = settings.chordal_tolerance_m if limit_m is None else float(limit_m)
    if not segments:
        return {"segments": 0, "evaluated": 0, "limit_m": limit, "worst": [], "exceeding": 0}
    rows = A.contiguous(np.array([s.row() for s in segments]))
    out = np.zeros((len(rows), A.N_CH))
    chord_deviation_nb(cad.tree, rows, int(settings.chordal_samples), int(settings.chordal_iterations),
                       float(settings.chordal_max_distance_m), out)
    evaluated = np.isfinite(out[:, A.CH_EMIN])
    err = np.where(evaluated, np.maximum(-out[:, A.CH_EMIN], out[:, A.CH_EMAX]), -np.inf)
    order = np.argsort(-err, kind="stable")
    listed = []
    for i in order[:max_listed]:
        if not evaluated[i]:
            break
        s = segments[i]
        neg = -out[i, A.CH_EMIN] >= out[i, A.CH_EMAX]
        par = out[i, A.CH_SMIN] if neg else out[i, A.CH_SMAX]
        rows_i = rows[i]
        from .kernels import segment_point_nb

        q = np.zeros(3)
        segment_point_nb(rows, int(i), float(par), q)
        face = int(cad.face_of([int(out[i, A.CH_FMIN] if neg else out[i, A.CH_FMAX])])[0])
        listed.append({"program": s.program, "line": s.line, "block": s.block, "tool": s.tool,
                       "kind": "arc" if rows_i[A.SG_KIND] == A.SEG_ARC else "line",
                       "chord_error_m": float(err[i]), "toward_part": bool(neg),
                       "error_min_m": float(out[i, A.CH_EMIN]), "error_max_m": float(out[i, A.CH_EMAX]),
                       "at_S_m": [float(v) for v in q], "parameter": float(par), "length_m": float(out[i, A.CH_LEN]),
                       "end_distances_m": [float(out[i, A.CH_D0]), float(out[i, A.CH_D1])], "face": face})
    exceeding = int(np.count_nonzero(err > limit))
    return {"segments": len(segments), "evaluated": int(evaluated.sum()), "limit_m": limit, "exceeding": exceeding,
            "max_m": float(err[evaluated].max()) if evaluated.any() else 0.0,
            "mean_m": float(err[evaluated].mean()) if evaluated.any() else 0.0, "worst": listed,
            "definition": "e(s) = d(P(s)) - ((1 - s) d0 + s d1): deviation from the end points' offset, worst sign"}


__all__ = ["PathSegment", "segments_from_commands", "chordal_deviation"]
