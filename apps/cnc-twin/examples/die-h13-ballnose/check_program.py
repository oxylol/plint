"""Geometric check of die_h13.h against part.step, independent of the toolpath generator's bookkeeping.

Run: cd apps/cnc-twin/examples/die-h13-ballnose && TSAFE_MEM=1500M ../../bin/tsafe python check_program.py

Reads the Klartext program (TOOL CALL, PLANE SPATIAL SPB, PLANE RESET, L blocks; FMAX moves skipped),
turns every programmed ball tip of the tilted working plane into a ball centre in the part (datum)
frame, samples the straight blocks every 0.05 mm, and measures for every vertex and facet centroid of
the cavity and rim of part.step the distance to the nearest ball centre: material left = distance - R
- stock (negative = the ball passed inside the part). It is the ideal-machine result of the program
(no deflection, runout, wear or machine errors) that EXPECTED.md starts from.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from make_program import DATUM_S, R, TOOLS, rot_y, triangles_from_step

HERE = Path(__file__).resolve().parent


def ball_centres(path: Path):
    tool, spb = None, 0.0
    pos = {"X": 0.0, "Y": 0.0, "Z": 0.0}
    paths: dict[int, list] = {}
    for line in path.read_text().splitlines():
        s = line.split(";")[0]
        if "TOOL CALL" in s:
            tool = int(re.search(r"TOOL CALL (\d+)", s).group(1))
            paths.setdefault(tool, [[]])
            continue
        if "PLANE SPATIAL" in s:
            spb = float(re.search(r"SPB([+-][\d.]+)", s).group(1))
            paths[tool].append([])
            continue
        if "PLANE RESET" in s:
            spb = 0.0
            continue
        m = re.match(r"\d+ L (.*)", s)
        if not m or "M91" in s:
            continue
        for ax, v in re.findall(r"([XYZ])([+-][\d.]+)", m.group(1)):
            pos[ax] = float(v)
        if "FMAX" in s:
            continue
        tip = np.array([pos["X"], pos["Y"], pos["Z"]])
        paths[tool][-1].append(rot_y(spb) @ (tip + np.array([0.0, 0.0, R])))
    return {t: [np.array(p) for p in runs if p] for t, runs in paths.items()}


def densify(run, step=0.05):
    out = [run[:1]]
    for a, b in zip(run[:-1], run[1:]):
        k = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        u = np.linspace(0.0, 1.0, k + 1)[1:, None]
        out.append(a + u * (b - a))
    return np.concatenate(out)


def main():
    tris = triangles_from_step(HERE / "part.step") - DATUM_S
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nz = np.abs(n[:, 2]) / np.maximum(np.linalg.norm(n, axis=1), 1e-30)
    cav = (tris[:, :, 2] < -1e-6).any(axis=1) & (nz > 0.2)
    pts = np.unique(np.round(np.concatenate([tris[cav].mean(axis=1), tris[cav].reshape(-1, 3)]), 6), axis=0)
    rim = pts[:, 2] > -2.0
    stock = {spec["T"]: spec["stock"] for spec in TOOLS.values()}
    for tool, runs in sorted(ball_centres(HERE / "die_h13.h").items()):
        dense = np.concatenate([densify(r) for r in runs])
        dist, _ = cKDTree(dense).query(pts, k=1)
        left = (dist - R - stock[tool]) * 1000.0
        print(f"T{tool}: {sum(len(r) for r in runs)} programmed points, stock {stock[tool]} mm, {len(pts)} surface points")
        for name, sel in (("cavity (z < -2)", ~rim), ("rim band (z >= -2)", rim)):
            v = left[sel]
            print(f"  {name}: material left um  min {v.min():.2f}  p1 {np.percentile(v, 1):.2f}  p50 "
                  f"{np.percentile(v, 50):.2f}  p99 {np.percentile(v, 99):.2f}  max {v.max():.2f}")


if __name__ == "__main__":
    main()
