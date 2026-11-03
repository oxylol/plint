"""Time-series channels: every recorded quantity is registered with its unit, tier and shape.

Columns in the series files: a scalar channel is one column ``<name>``; a vector channel of
``max_shape`` (n,) is ``<name>[0]`` ... ``<name>[n-1]``, padded with NaN where the current shape
is smaller (n_teeth varies by tool). The tick tier stores ``<name>.min``, ``.mean``, ``.max`` of
each fast channel. Each domain registers its channels from ``<domain>/channels.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Tier(str, Enum):
    FAST = "fast"     # every L0 cutting step (windows from a ring buffer by default, or all)
    TICK = "tick"     # every L1 interpolation tick
    MACRO = "macro"   # every L2 coupling step


@dataclass(frozen=True)
class Channel:
    name: str                      # dotted, unit last: "cutting.force_x_N", "thermal.edge_temp_K"
    unit: str                      # SI unit text as shown in plots ("N", "K", "m", "W", "-")
    tier: Tier
    domain: str
    doc: str
    max_shape: tuple[int, ...] = ()   # () scalar; (16,) per tooth up to 16 teeth ...
    dtype: str = "float64"            # float64 | float32 | int64 | int32 | int8 | bool

    def columns(self) -> list[str]:
        if not self.max_shape:
            return [self.name]
        count = 1
        for size in self.max_shape:
            count *= size
        return [f"{self.name}[{i}]" for i in range(count)]


class ChannelRegistry:
    def __init__(self) -> None:
        self._channels: dict[str, Channel] = {}

    def register(self, channel: Channel) -> None:
        existing = self._channels.get(channel.name)
        if existing is not None and existing != channel:
            raise ValueError(f"channel {channel.name} registered twice with different definitions")
        self._channels[channel.name] = channel

    def register_all(self, channels: list[Channel]) -> None:
        for channel in channels:
            self.register(channel)

    def tier(self, tier: Tier) -> list[Channel]:
        return sorted((c for c in self._channels.values() if c.tier == tier), key=lambda c: c.name)

    def get(self, name: str) -> Channel:
        return self._channels[name]

    def all(self) -> list[Channel]:
        return sorted(self._channels.values(), key=lambda c: c.name)

    def __contains__(self, name: str) -> bool:
        return name in self._channels
