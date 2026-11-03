"""Section definition of the ruled Ti-6Al-4V blade (used by make_cad.py; make_program.py reads the CAD).

Frame (CAD frame = stock frame S): z is the stacking axis, the aerodynamic span runs from the hub
section at z = 0 to the tip section at z = 20; the blade continues as the same ruled surface down to the
platform (base block top) at z = -7. x runs along the hub chord (leading edge at -x), y completes.

Each section is a smooth (C-infinity) cambered airfoil, parametrised by an angle phi:
x = (c/2) cos(phi), y = y_c(x) + (t c / 2) sin(phi) (1 - k cos(phi)), with the parabolic camber line
y_c = m (1 - (2x/c)^2) (max camber m = 4 % of the chord at mid-chord) and k = 0.2, which moves the
thickness forward: leading-edge radius (t c (1 + k) / 2)^2 / (c / 2), trailing-edge radius
(t c (1 - k) / 2)^2 / (c / 2). The stacking point (mid-chord on the camber line) is on the z axis. The tip
section is geometrically similar (same thickness ratio, chord 36), turned by the twist about z and shifted by the lean. Hub and tip points are
taken at the same phi, so straight lines between them are the rulings.
"""

from __future__ import annotations

import math

import numpy as np

HUB = {"z": 0.0, "chord": 40.0, "t": 0.10, "camber": 0.04, "twist_deg": 0.0, "shift": (0.0, 0.0)}
TIP = {"z": 20.0, "chord": 36.0, "t": 0.10, "camber": 0.04, "twist_deg": 8.0, "shift": (0.0, -6.0)}
K_THICKNESS = 0.2
ROOT_Z = -7.0                         # platform (base block top); the ruled surface continues down to it
BASE = {"x": (-30.0, 30.0), "y": (-16.0, 16.0), "z": (-27.0, ROOT_Z)}   # platform block held in the vise


def section_points(sec, n=480):
    """Closed section, counter-clockwise seen from +z, starting at the trailing edge (phi = 0), as an
    [n, 2] array without the repeated first point."""
    c, t, m = sec["chord"], sec["t"], sec["camber"] * sec["chord"]
    phi = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    x = 0.5 * c * np.cos(phi)
    y = m * (1.0 - (2.0 * x / c) ** 2) - m + 0.5 * t * c * np.sin(phi) * (1.0 - K_THICKNESS * np.cos(phi))
    p = np.stack([x, y], axis=1)
    rot = math.radians(sec["twist_deg"])
    R = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
    return p @ R.T + np.asarray(sec["shift"])


def edge_radii(sec):
    c, t, k = sec["chord"], sec["t"], K_THICKNESS
    return {"le_mm": (0.5 * t * c * (1 + k)) ** 2 / (0.5 * c), "te_mm": (0.5 * t * c * (1 - k)) ** 2 / (0.5 * c)}


def section_3d(sec):
    p = section_points(sec)
    return np.column_stack([p, np.full(len(p), sec["z"])])


def ruled_point(hub, tip, z):
    """Points of the ruled surface at height z (linear in z along each ruling)."""
    w = (z - HUB["z"]) / (TIP["z"] - HUB["z"])
    return hub + w * (tip - hub)


def offset_closed(points_2d, distance):
    """In-plane offset of a closed counter-clockwise polygon by ``distance`` (outwards for > 0)."""
    p = np.asarray(points_2d)
    nxt, prv = np.roll(p, -1, axis=0), np.roll(p, 1, axis=0)
    t = nxt - prv
    t /= np.linalg.norm(t, axis=1)[:, None]
    normal = np.stack([t[:, 1], -t[:, 0]], axis=1)          # right of the tangent = outwards for CCW
    return p + distance * normal


__all__ = ["HUB", "TIP", "ROOT_Z", "BASE", "section_points", "section_3d", "ruled_point", "offset_closed",
           "edge_radii"]
