"""The memory cap of a generator subprocess (cnctwin.core.outputs.capped_command)."""

from __future__ import annotations

from cnctwin.core.outputs import capped_command


def _swap_bytes(command: list[str]) -> int:
    return int(next(a for a in command if a.startswith("MemorySwapMax=")).split("=", 1)[1])


def test_the_swap_allowance_is_never_more_than_the_cap():
    small, env = capped_command(["-c", "pass"], "systemd", 8 * 2 ** 20, env={"A": "1"})
    large, _ = capped_command(["-c", "pass"], "systemd", 2 ** 30, env={"A": "1"})
    assert env is None and "MemoryMax=8388608" in small and "A=1" in small
    assert _swap_bytes(small) == 8 * 2 ** 20            # a tiny cap fails instead of paging its working set
    assert _swap_bytes(large) == 256 * 2 ** 20          # db/models/outputs.yaml memory_swap_max_bytes


def test_rlimit_and_none_run_the_plain_command():
    for mode in ("rlimit", "none"):
        command, env = capped_command(["-c", "pass"], mode, 8 * 2 ** 20, env={"A": "1"})
        assert command[1:] == ["-c", "pass"] and env == {"A": "1"}
