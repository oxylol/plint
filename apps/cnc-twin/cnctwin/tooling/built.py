"""What a type builder hands to the generic assembly: cutting pieces, teeth, revolved outline."""

from __future__ import annotations

from dataclasses import dataclass, field

from .abi import InsertProfile, ThreadProfile
from .edges import ToothDef
from .profile import Seg


@dataclass
class Built:
    """Output of a type builder.

    ``segs``: the cutting pieces of the layout (EdgeModel). ``outline``: the revolved outline of
    the cutting head from the axis at the tip to the top of the head (pieces with ``kind`` set
    are swept by cutting edges), in the envelope convention of cnctwin.tooling.abi; the envelope
    builder continues it with body, neck, shank, holder and spindle nose. ``info`` carries the
    numbers the property models need (diameter, core, gullet, flutes, head length, ...).
    """

    segs: list[Seg]
    teeth: list[ToothDef]
    outline: list[Seg]
    info: dict = field(default_factory=dict)
    thread: ThreadProfile | None = None
    insert: InsertProfile | None = None
    rotating: bool = True
