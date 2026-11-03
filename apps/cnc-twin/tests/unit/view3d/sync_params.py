"""web/js/view3d/params.js from db/models/view3d.yaml: the browser's copy of the 3D view's settings.

The YAML file is the one source (units, sources, ranges, validated by cnctwin.db.models.params); the
browser cannot read it, so its SI values are written into an ES module. test_view3d_params.py fails
when the module is stale.

    bin/tsafe python tests/unit/view3d/sync_params.py           # rewrite params.js
    bin/tsafe python tests/unit/view3d/sync_params.py --check   # exit 1 when it is stale
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cnctwin.db.models import params  # noqa: E402

TARGET = ROOT / "web" / "js" / "view3d" / "params.js"
HEADER = """/* Settings of the 3D view in SI: written from db/models/view3d.yaml (units, sources and ranges are
 * there) by tests/unit/view3d/sync_params.py. Do not edit: change the YAML file and run the script. */

"""


def render() -> str:
    values = params("view3d")
    body = json.dumps(values, indent=2, sort_keys=True, ensure_ascii=False)
    return f"{HEADER}export const PARAMS = Object.freeze({body});\n"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    text = render()
    if "--check" in argv:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != text:
            print(f"{TARGET} is stale: run tests/unit/view3d/sync_params.py")
            return 1
        return 0
    TARGET.write_text(text, encoding="utf-8")
    print(f"wrote {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
