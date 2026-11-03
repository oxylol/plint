"""Index of the removal kernels (docs/models/removal.md "Kernels"). Importing this module compiles (or loads
from the cache) every kernel; the domain and the tests import the k_* modules they need instead.

| module | kernels |
|---|---|
| k_core | array-level primitives, inlined everywhere (ray lookups, pool load/store, bricks, D_f, classify) |
| k_ray | FIELD_T wrappers of the primitives, interval algebra, family sums |
| k_stock | analytic stock fills, watertight mesh rasterisation onto the rays |
| k_envelope | ray / revolved-envelope interval mathematics (line, sphere, torus quartic) |
| k_subtract | subtract_envelope (hot pass) and post_pieces (bricks, target, extent, centroid) |
| k_helix | tap swept-profile subtraction along the helical path |
| k_collide | non-cutting envelope parts, meshes and swept meshes against the field |
| k_sd / k_gate | interpolated boundary, engaged intervals, contact geometry, sector box |
| k_query | ray cast, surface point, surface height, fill fraction, wall thickness, broad phase, probe |
| k_connect | connectivity and loose-body removal |
| k_mesh | marching cubes per brick |
| k_polar | polar field cut, radius, polar -> tri-dexel rebuild |
| k_transfer | exact lattice remap |
| k_drive | tooth_pass, l1_contact, excess_material (composition for core.hotloop.run_macro) |
"""

from __future__ import annotations

from . import (k_collide, k_connect, k_core, k_drive, k_envelope, k_gate, k_helix, k_mesh, k_polar, k_query,
               k_ray, k_sd, k_stock, k_subtract, k_transfer)

MODULES = (k_ray, k_stock, k_envelope, k_subtract, k_helix, k_collide, k_sd, k_gate, k_query, k_connect, k_mesh,
           k_polar, k_transfer, k_drive)
KERNELS = tuple(k for m in MODULES for k in m.KERNELS)
INLINED = tuple(getattr(k_envelope, "INLINED", ())) + tuple(getattr(k_gate, "INLINED", ()))

__all__ = ["KERNELS", "INLINED", "MODULES"]
