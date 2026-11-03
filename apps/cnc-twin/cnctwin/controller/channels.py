"""Channels the controller records (macro tier, one column per controller path, up to MAX_PATHS)."""

from __future__ import annotations

from ..core.channels import Channel, Tier

MAX_PATHS = 4

WAIT_CODES = {None: 0, "probe": 1, "live": 2, "stop": 3, "path_wait": 4, "setter": 5, "end": 6}

CHANNELS = [
    Channel("controller.buffer_fill", "-", Tier.MACRO, "controller",
            "look-ahead buffer occupancy: commands produced and not yet retired / capacity", (MAX_PATHS,)),
    Channel("controller.interp_lead_s", "s", Tier.MACRO, "controller",
            "how far the interpreter's block-time clock is ahead of simulated time (negative: starving)",
            (MAX_PATHS,)),
    Channel("controller.block_rate_Hz", "Hz", Tier.MACRO, "controller",
            "NC blocks and macro statements interpreted per second over the macro step", (MAX_PATHS,)),
    Channel("controller.waiting", "-", Tier.MACRO, "controller",
            "what the path waits for: 0 none, 1 probe, 2 live read, 3 stop, 4 path wait, 5 tool setter, 6 end",
            (MAX_PATHS,), "int8"),
    Channel("controller.alarm", "-", Tier.MACRO, "controller", "the path is stopped by an alarm", (MAX_PATHS,), "bool"),
]


def channels() -> list[Channel]:
    return list(CHANNELS)
