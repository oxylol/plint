"""Controller dialects: registry and the dialects of the core (FANUC 0i-F, Haas NGC).

Other dialects (Siemens 840D sl, Heidenhain) register themselves from their own sub-packages
(``cnctwin.controller.dialects.siemens840d``, ``...heidenhain``), loaded by
``cnctwin.controller.hooks.load_plugins``.
"""

from .base import Dialect, GSpec, build_dialect, dialect_names, normalize_g, register_dialect
from . import fanuc  # noqa: F401  (registers "fanuc")
from . import haas  # noqa: F401  (registers "haas")

__all__ = ["Dialect", "GSpec", "build_dialect", "dialect_names", "normalize_g", "register_dialect"]
