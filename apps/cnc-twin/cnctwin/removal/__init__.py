"""Material removal: the in-process workpiece (docs/models/removal.md).

Tri-dexel field with refine levels and an overflow pool (abi, field, stock), exact subtraction of revolved
tool envelopes per tooth and slab (k_envelope, k_subtract), tap sweeps (k_helix), turning on a polar field
(polar, k_polar), outer-boundary gating and the contact geometry (k_gate), queries (k_query, views),
collisions (k_collide, collision), connectivity and loose bodies (k_connect, connect), meshing (mesh,
k_mesh), transfers between operations (transfer), and the RemovalDomain (domain).

Kernel modules compile on import; this package imports nothing heavy by itself.
"""
