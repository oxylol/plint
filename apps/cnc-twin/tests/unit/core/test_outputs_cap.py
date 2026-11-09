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


def test_blas_under_an_rlimit_cap_works_and_running_out_fails_fast():
    """Under RLIMIT_DATA, OpenBLAS retries a work buffer it cannot map forever (without warm_blas this subprocess
    spins until the timeout); warm_blas maps the buffer before the cap, so BLAS keeps working under the cap and an
    allocation beyond it is a MemoryError at once."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    code = ("from cnctwin.core.outputs import warm_blas, cap_memory\n"
            "warm_blas()\n"
            "import numpy as np\n"
            "small = np.ones((256, 256))\n"
            "data = next(int(l.split()[1]) * 1024 for l in open('/proc/self/status') if l.startswith('VmData:'))\n"
            "cap_memory(data + 4 * 2 ** 20)\n"
            "print('product', float((small @ small)[0, 0]))\n"
            "try:\n"
            "    np.ones((4096, 4096))\n"
            "except MemoryError:\n"
            "    print('MemoryError')\n")
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[3]), OPENBLAS_NUM_THREADS="1")
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30)
    assert done.stdout.split() == ["product", "256.0", "MemoryError"], done.stderr
