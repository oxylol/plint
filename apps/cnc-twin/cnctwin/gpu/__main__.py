"""``python -m cnctwin.gpu [info | bench [smoke|default] | ptx [major.minor] | warmup]``: device detection,
the CPU vs GPU benchmark, the offline PTX check, or compiling the kernels. Prints JSON."""

from __future__ import annotations

import json
import sys


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    command = argv[0] if argv else "info"
    if command == "info":
        from .device import available, info
        ok, reason = available()
        print(json.dumps({"available": ok, "reason": reason, "device": info().to_dict()}, indent=2))
        return 0
    if command == "bench":
        from .bench import benchmark
        print(json.dumps(benchmark(sizes=argv[1] if len(argv) > 1 else "default"), indent=2))
        return 0
    if command == "ptx":
        from .ptxcheck import check
        cc = tuple(int(v) for v in argv[1].split(".")) if len(argv) > 1 else (5, 2)
        report = check(cc)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report.get("ok") else 1
    if command == "warmup":
        from .accel import Accelerator
        from .schema import GpuSettings
        print(json.dumps(Accelerator(GpuSettings(mode="on")).warmup(), indent=2))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
